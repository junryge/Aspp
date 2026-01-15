# -*- coding: utf-8 -*-
"""
================================================================================
M14 예측 평가 모듈
- 내부: data 폴더의 m14_data_YYYYMMDD.csv 파일 사용
- 외부: 로그프레소 API 직접 조회
- 10분/30분 예측 평가
- 백그라운드 실행 지원 (실시간 모니터링에 영향 없음)
================================================================================
"""

import os
import pickle
import warnings
import gc
import threading
import numpy as np
import pandas as pd
import requests
import urllib.parse
from io import StringIO
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
requests.packages.urllib3.disable_warnings()

# ============================================================================
# 로그프레소 설정 (외부 조회용)
# ============================================================================
LOGPRESSO_HOST = "10.40.42.27"
LOGPRESSO_PORT = 8888
LOGPRESSO_API_KEY = "db1d2335-49cf-e859-3519-1ca132922e38"

FINAL_COLUMNS = [
    'CURRTIME', 'TOTALCNT',
    'M14AM10A', 'M10AM14A', 'M14AM10ASUM',
    'M14AM14B', 'M14BM14A', 'M14AM14BSUM',
    'M14AM16', 'M16M14A', 'M14AM16SUM',
    'M14.QUE.ALL.CURRENTQCREATED',
    'M14.QUE.ALL.CURRENTQCOMPLETED',
    'M14.QUE.OHT.OHTUTIL',
    'M14.QUE.ALL.TRANSPORT4MINOVERCNT',
    'M14B.QUE.SENDFAB.VERTICALQUEUECOUNT'
]

# ============================================================================
# 모델 설정
# ============================================================================
CONFIG_10 = {
    'model_file': 'models/v10_4_m14_model.pkl',
    'sequence_length': 280,
    'prediction_offset': 10,
    'limit_value': 1700,
    'target_column': 'TOTALCNT',
}

CONFIG_30 = {
    'model_file': 'models/v10_4_30min_m14_model.pkl',
    'sequence_length': 280,
    'prediction_offset': 30,
    'limit_value': 1700,
    'target_column': 'TOTALCNT',
}

# 전역 모델 캐시
_model_data_10 = None
_model_data_30 = None


# ============================================================================
# 로그프레소 API 함수 (외부 조회용)
# ============================================================================
def query_logpresso(query, timeout=180):
    """로그프레소 쿼리 실행"""
    query_clean = ' '.join(query.split())
    encoded = urllib.parse.quote(query_clean, safe='')
    url = f"http://{LOGPRESSO_HOST}:{LOGPRESSO_PORT}/logpresso/httpexport/query.csv?_apikey={LOGPRESSO_API_KEY}&_q={encoded}"
    
    try:
        resp = requests.get(url, verify=False, timeout=timeout)
        
        if resp.status_code == 200 and resp.text.strip() and not resp.text.startswith('<!'):
            df = pd.read_csv(StringIO(resp.text))
            return df
        else:
            print(f"[평가-외부] 쿼리 에러: Status {resp.status_code}")
            return None
            
    except Exception as e:
        print(f"[평가-외부] 쿼리 예외: {e}")
        return None


def get_logpresso_data_range(from_time, to_time):
    """로그프레소에서 특정 시간대 데이터 조회"""
    print(f"[평가-외부] 로그프레소 조회: {from_time} ~ {to_time}")
    
    # ts_current_job 집계
    query_job = f'''
    table from={from_time} to={to_time} ts_current_job
    | search FAB == "M14"
    | eval A = case(trim(DESTMACHINENAME) == "4ABL_M10", 1, 0)
    | eval B = case(substr(trim(SOURCEMACHINENAME), 0, 7) == "4ABL330", 1, 0)
    | eval C = case(substr(trim(DESTMACHINENAME), 0, 4) == "4ALF", 1, 0)
    | eval D = case(substr(trim(SOURCEMACHINENAME), 0, 4) == "4ALF", 1, 0)
    | eval E = case(substr(trim(DESTMACHINENAME), 0, 4) == "4AFC", 1, 0)
    | eval F = case(substr(trim(SOURCEMACHINENAME), 0, 4) == "4AFC", 1, 0)
    | stats sum(A), sum(B), sum(C), sum(D), sum(E), sum(F), count by CURRTIME
    | rename count as TOTALCNT, sum(A) as M14AM10A, sum(B) as M10AM14A, sum(C) as M14AM14B, sum(D) as M14BM14A, sum(E) as M14AM16, sum(F) as M16M14A
    | eval M14AM10ASUM = M10AM14A + M14AM10A,
           M14AM14BSUM = M14AM14B + M14BM14A,
           M14AM16SUM = M14AM16 + M16M14A
    | sort CURRTIME
    '''
    df_job = query_logpresso(query_job)
    
    if df_job is None or len(df_job) == 0:
        return None
    
    # star_transport_view
    query_star = f'''
    table from={from_time} to={to_time} star_transport_view
    | eval CURRTIME = string(CRT_TM, "yyyyMMddHHmm")
    | pivot last(IDC_VAL) for IDC_NM by CURRTIME
    | sort CURRTIME
    '''
    df_star = query_logpresso(query_star)
    
    # Merge
    if df_star is not None and len(df_star) > 0:
        df_merged = pd.merge(df_job, df_star, on='CURRTIME', how='left')
    else:
        df_merged = df_job
    
    # 컬럼 정리
    for col in FINAL_COLUMNS:
        if col not in df_merged.columns:
            df_merged[col] = 0
    
    df_final = df_merged[FINAL_COLUMNS].copy()
    df_final = df_final.sort_values('CURRTIME').reset_index(drop=True)
    df_final = df_final.fillna(0)
    df_final['CURRTIME'] = df_final['CURRTIME'].astype(str)
    
    print(f"  → {len(df_final)} rows")
    return df_final


# ============================================================================
# 백그라운드 작업 관리
# ============================================================================
class EvaluationManager:
    """백그라운드 평가 작업 관리"""
    
    def __init__(self):
        self.is_running = False
        self.progress = 0
        self.total = 0
        self.status = 'idle'  # idle, running, completed, error
        self.result = None
        self.error = None
        self.lock = threading.Lock()
        self.thread = None
    
    def start(self, data_dir, date_start, date_end, time_start, time_end, pred_type, data_source='internal'):
        """백그라운드 평가 시작
        
        Args:
            data_source: 'internal' (파일) 또는 'external' (로그프레소)
        """
        with self.lock:
            if self.is_running:
                return False, '이미 평가가 진행 중입니다'
            
            self.is_running = True
            self.progress = 0
            self.total = 0
            self.status = 'running'
            self.result = None
            self.error = None
        
        self.thread = threading.Thread(
            target=self._run_evaluation,
            args=(data_dir, date_start, date_end, time_start, time_end, pred_type, data_source),
            daemon=True
        )
        self.thread.start()
        return True, '평가 시작됨'
    
    def _run_evaluation(self, data_dir, date_start, date_end, time_start, time_end, pred_type, data_source):
        """실제 평가 수행 (백그라운드 스레드)"""
        try:
            result = evaluate(
                data_dir=data_dir,
                date_start=date_start,
                date_end=date_end,
                time_start=time_start,
                time_end=time_end,
                pred_type=pred_type,
                data_source=data_source,
                progress_callback=self._update_progress
            )
            
            with self.lock:
                self.result = result
                self.status = 'completed'
                self.is_running = False
                
        except Exception as e:
            import traceback
            with self.lock:
                self.error = str(e)
                self.status = 'error'
                self.is_running = False
            traceback.print_exc()
    
    def _update_progress(self, current, total):
        """진행 상태 업데이트"""
        with self.lock:
            self.progress = current
            self.total = total
    
    def get_status(self):
        """현재 상태 반환"""
        with self.lock:
            return {
                'status': self.status,
                'progress': self.progress,
                'total': self.total,
                'percent': round(self.progress / self.total * 100, 1) if self.total > 0 else 0
            }
    
    def get_result(self):
        """결과 반환"""
        with self.lock:
            if self.status == 'completed':
                return self.result
            elif self.status == 'error':
                return {'error': self.error}
            else:
                return None
    
    def reset(self):
        """상태 초기화"""
        with self.lock:
            if not self.is_running:
                self.status = 'idle'
                self.progress = 0
                self.total = 0
                self.result = None
                self.error = None


# 전역 매니저 인스턴스
eval_manager = EvaluationManager()


def load_model_10():
    """10분 모델 로드"""
    global _model_data_10
    if _model_data_10 is not None:
        return _model_data_10
    
    if not os.path.exists(CONFIG_10['model_file']):
        print(f"[평가] 10분 모델 파일 없음: {CONFIG_10['model_file']}")
        return None
    
    with open(CONFIG_10['model_file'], 'rb') as f:
        _model_data_10 = pickle.load(f)
    print(f"[평가] 10분 모델 로드 완료")
    return _model_data_10


def load_model_30():
    """30분 모델 로드"""
    global _model_data_30
    if _model_data_30 is not None:
        return _model_data_30
    
    if not os.path.exists(CONFIG_30['model_file']):
        print(f"[평가] 30분 모델 파일 없음: {CONFIG_30['model_file']}")
        return None
    
    with open(CONFIG_30['model_file'], 'rb') as f:
        _model_data_30 = pickle.load(f)
    print(f"[평가] 30분 모델 로드 완료")
    return _model_data_30


def create_sequence_features(df, feature_cols, seq_len, idx, limit_val=1700):
    """시퀀스 피처 생성"""
    features = []
    for col in feature_cols:
        if col not in df.columns:
            seq = np.zeros(seq_len)
        else:
            seq = df[col].iloc[idx - seq_len:idx].values
        
        if len(seq) < seq_len:
            seq = np.pad(seq, (seq_len - len(seq), 0), 'constant', constant_values=0)
        
        current_val = seq[-1] if len(seq) > 0 else 0
        features.extend([
            np.mean(seq), np.std(seq), np.min(seq), np.max(seq), current_val,
            seq[-1] - seq[0] if len(seq) > 0 else 0,
            np.percentile(seq, 25), np.percentile(seq, 75),
            np.mean(seq[-10:]) - np.mean(seq[:10]) if len(seq) >= 10 else 0,
            np.max(seq[-30:]) if len(seq) >= 30 else np.max(seq),
            seq[-1] - seq[-10] if len(seq) >= 10 else 0,
            seq[-1] - seq[-30] if len(seq) >= 30 else 0,
            seq[-1] - seq[-60] if len(seq) >= 60 else 0,
            (seq[-1] - seq[-10]) / 10 if len(seq) >= 10 else 0,
            (seq[-1] - seq[-30]) / 30 if len(seq) >= 30 else 0,
            np.max(seq[-10:]) - np.min(seq[-10:]) if len(seq) >= 10 else 0,
            np.max(seq[-30:]) - np.min(seq[-30:]) if len(seq) >= 30 else 0,
            limit_val - current_val,
            1 if current_val >= 1500 else 0,
            1 if current_val >= 1600 else 0,
        ])
    return features


def preprocess_data(df, feature_groups):
    """데이터 전처리"""
    df = df.copy()
    
    if not pd.api.types.is_datetime64_any_dtype(df['CURRTIME']):
        df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M', errors='coerce')
    
    df = df.dropna(subset=['CURRTIME']).sort_values('CURRTIME').reset_index(drop=True)
    
    if 'M14.QUE.ALL.CURRENTQCREATED' in df.columns and 'M14.QUE.ALL.CURRENTQCOMPLETED' in df.columns:
        df['QUEUE_GAP'] = df['M14.QUE.ALL.CURRENTQCREATED'] - df['M14.QUE.ALL.CURRENTQCOMPLETED']
    
    all_cols = []
    for group in feature_groups.values():
        all_cols.extend(group)
    
    for col in list(set(all_cols)):
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df


def load_data_files(data_dir, date_start, date_end):
    """내부: 날짜 범위의 데이터 파일 로드"""
    all_data = []
    
    start = datetime.strptime(date_start, "%Y%m%d")
    end = datetime.strptime(date_end, "%Y%m%d")
    
    current = start
    while current <= end:
        date_str = current.strftime("%Y%m%d")
        file_path = os.path.join(data_dir, f'm14_data_{date_str}.csv')
        
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                df['CURRTIME'] = df['CURRTIME'].astype(str)
                all_data.append(df)
                print(f"  [내부] 로드: {file_path} ({len(df)}행)")
            except Exception as e:
                print(f"  [내부] 에러: {file_path} - {e}")
        
        current += timedelta(days=1)
    
    if not all_data:
        return None
    
    df = pd.concat(all_data, ignore_index=True)
    df = df.drop_duplicates(subset=['CURRTIME']).sort_values('CURRTIME').reset_index(drop=True)
    
    return df


def load_data_logpresso(date_start, date_end, time_start, time_end, progress_callback=None):
    """외부: 로그프레소에서 데이터 조회"""
    print(f"[평가-외부] 로그프레소 데이터 조회 시작")
    
    # 날짜 범위 계산 (일 단위로 나눠서 조회 - 대용량 대응)
    start = datetime.strptime(date_start, "%Y%m%d")
    end = datetime.strptime(date_end, "%Y%m%d")
    total_days = (end - start).days + 1
    
    all_data = []
    current = start
    day_count = 0
    
    while current <= end:
        day_count += 1
        date_str = current.strftime("%Y%m%d")
        
        # 해당 날짜의 시작/종료 시간 설정
        if current == start:
            day_from = f"{date_str}{time_start}"
        else:
            day_from = f"{date_str}0000"
        
        if current == end:
            day_to = f"{date_str}{time_end}"
        else:
            day_to = f"{date_str}2359"
        
        print(f"  [{day_count}/{total_days}] {day_from} ~ {day_to} 조회 중...")
        
        if progress_callback:
            progress_callback(day_count, total_days * 2)  # 데이터 로드 50%, 평가 50%
        
        df = get_logpresso_data_range(day_from, day_to)
        
        if df is not None and len(df) > 0:
            all_data.append(df)
            print(f"      → {len(df)}행 로드")
        
        current += timedelta(days=1)
    
    if not all_data:
        return None
    
    df = pd.concat(all_data, ignore_index=True)
    df = df.drop_duplicates(subset=['CURRTIME']).sort_values('CURRTIME').reset_index(drop=True)
    
    print(f"[평가-외부] 총 {len(df)}행 로드 완료")
    return df


def evaluate(data_dir, date_start, date_end, time_start='0000', time_end='2359', 
             pred_type='10', data_source='internal', progress_callback=None):
    """
    평가 수행
    
    Args:
        data_dir: 데이터 폴더 경로 (내부 조회 시)
        date_start: 시작 날짜 (YYYYMMDD)
        date_end: 종료 날짜 (YYYYMMDD)
        time_start: 시작 시간 (HHMM)
        time_end: 종료 시간 (HHMM)
        pred_type: '10' 또는 '30'
        data_source: 'internal' (파일) 또는 'external' (로그프레소)
        progress_callback: 진행 상태 콜백 함수
    
    Returns:
        dict: 평가 결과
    """
    source_name = "내부(파일)" if data_source == 'internal' else "외부(로그프레소)"
    print(f"\n[평가] {pred_type}분 예측 평가 시작 - {source_name}")
    print(f"  기간: {date_start} {time_start} ~ {date_end} {time_end}")
    
    if pred_type == '10':
        config = CONFIG_10
        model_data = load_model_10()
        analysis_range = 5
    else:
        config = CONFIG_30
        model_data = load_model_30()
        analysis_range = 20
    
    if model_data is None:
        return {'error': f'{pred_type}분 모델을 찾을 수 없습니다'}
    
    models = model_data['models']
    scalers = model_data['scalers']
    feature_groups = model_data['feature_groups']
    
    # 데이터 로드 (내부 vs 외부)
    print(f"\n[1] 데이터 로드... ({source_name})")
    
    if data_source == 'external':
        # 외부: 로그프레소 직접 조회
        df = load_data_logpresso(date_start, date_end, time_start, time_end, progress_callback)
    else:
        # 내부: 파일에서 로드
        df = load_data_files(data_dir, date_start, date_end)
    
    if df is None or len(df) == 0:
        if data_source == 'external':
            return {'error': '로그프레소에서 데이터를 조회할 수 없습니다. 네트워크 또는 API 연결을 확인하세요.'}
        else:
            return {'error': '해당 기간에 데이터 파일이 없습니다. 외부(로그프레소) 조회를 사용해보세요.'}
    
    print(f"  총 {len(df):,}행 로드됨")
    
    # 시간 필터링 (내부 조회 시에만 필요)
    if data_source == 'internal':
        start_datetime = date_start + time_start
        end_datetime = date_end + time_end
        
        df_filtered = df[
            (df['CURRTIME'].astype(str).str[:12] >= start_datetime) & 
            (df['CURRTIME'].astype(str).str[:12] <= end_datetime)
        ].copy()
    else:
        # 외부 조회는 이미 시간 필터링됨
        df_filtered = df.copy()
    
    if len(df_filtered) == 0:
        return {'error': f'선택한 시간 범위에 데이터가 없습니다 ({date_start} {time_start} ~ {date_end} {time_end})'}
    
    print(f"  시간 필터 후: {len(df_filtered):,}행")
    
    # 전처리
    print(f"\n[2] 데이터 전처리...")
    df_filtered = preprocess_data(df_filtered, feature_groups)
    print(f"  전처리 후: {len(df_filtered):,}행")
    
    # 누락 컬럼 확인
    missing_cols = []
    for group_name, cols in feature_groups.items():
        for col in cols:
            if col not in df_filtered.columns or df_filtered[col].sum() == 0:
                missing_cols.append(col)
    
    if missing_cols:
        print(f"  ⚠ 누락/0인 컬럼: {len(missing_cols)}개")
    
    # 평가 수행
    print(f"\n[3] 예측 수행...")
    seq_len = config['sequence_length']
    pred_offset = config['prediction_offset']
    limit_val = config['limit_value']
    target_col = config['target_column']
    
    if len(df_filtered) < seq_len + pred_offset:
        return {'error': f'데이터가 부족합니다 (최소 {seq_len + pred_offset}행 필요, 현재 {len(df_filtered)}행)'}
    
    results = []
    total = len(df_filtered) - seq_len - pred_offset
    
    # 외부 조회 시 진행률 조정 (데이터 로드가 50% 차지)
    progress_offset = total if data_source == 'external' else 0
    
    if progress_callback:
        progress_callback(progress_offset, total + progress_offset)
    
    for idx in range(seq_len, len(df_filtered) - pred_offset):
        current_progress = idx - seq_len
        
        if current_progress % 100 == 0:
            if progress_callback:
                progress_callback(current_progress + progress_offset, total + progress_offset)
            print(f"    진행: {current_progress:,}/{total:,}")
            gc.collect()
        
        current_time = df_filtered['CURRTIME'].iloc[idx - 1]
        current_total = float(df_filtered[target_col].iloc[idx - 1]) if pd.notna(df_filtered[target_col].iloc[idx - 1]) else 0
        prediction_time = current_time + timedelta(minutes=pred_offset)
        
        future_end = min(idx - 1 + pred_offset, len(df_filtered))
        actual_max = df_filtered[target_col].iloc[idx - 1:future_end].max()
        
        actual_single_idx = idx - 1 + pred_offset
        actual_single = df_filtered[target_col].iloc[actual_single_idx] if actual_single_idx < len(df_filtered) else df_filtered[target_col].iloc[-1]
        
        try:
            feat_target = create_sequence_features(df_filtered, feature_groups['target'], seq_len, idx)
            feat_important = create_sequence_features(df_filtered, feature_groups['important'], seq_len, idx)
            feat_auxiliary = create_sequence_features(df_filtered, feature_groups['auxiliary'], seq_len, idx)
            
            X_target = scalers['target'].transform([feat_target])
            X_important = scalers['important'].transform([feat_important])
            X_auxiliary = scalers['auxiliary'].transform([feat_auxiliary])
            
            pred_xgb_target = models['xgb_target'].predict(X_target)[0]
            pred_xgb_important = models['xgb_important'].predict(X_important)[0]
            pred_xgb_auxiliary = models['xgb_auxiliary'].predict(X_auxiliary)[0]
            
            pred_lgb_target = models['lgb_target'].predict(X_target)[0]
            pred_lgb_important = models['lgb_important'].predict(X_important)[0]
            pred_lgb_auxiliary = models['lgb_auxiliary'].predict(X_auxiliary)[0]
            prob_lgb_important = models['lgb_important'].predict_proba(X_important)[0][1]
            
            pred_xgb_pdt = None
            if 'xgb_pdt_new' in models and 'pdt_new' in scalers and 'pdt_new' in feature_groups:
                feat_pdt = create_sequence_features(df_filtered, feature_groups.get('pdt_new', []), seq_len, idx)
                if feat_pdt:
                    X_pdt = scalers['pdt_new'].transform([feat_pdt])
                    pred_xgb_pdt = models['xgb_pdt_new'].predict(X_pdt)[0]
            
            votes = [
                1 if pred_xgb_target >= limit_val else 0,
                1 if pred_xgb_important >= limit_val else 0,
                1 if pred_xgb_auxiliary >= limit_val else 0,
                pred_lgb_target,
                pred_lgb_important,
                pred_lgb_auxiliary,
            ]
            if pred_xgb_pdt is not None:
                votes.append(1 if pred_xgb_pdt >= limit_val else 0)
            
            vote_sum = sum(votes)
            
            rule1 = vote_sum >= 3
            rule2 = (prob_lgb_important >= 0.50) and (current_total >= 1450)
            rule3 = (pred_xgb_important >= 1680) and (current_total >= 1500)
            rule4 = (current_total >= 1600) and (vote_sum >= 2)
            rule5 = (pred_xgb_important >= 1700)
            
            final_danger = 1 if (rule1 or rule2 or rule3 or rule4 or rule5) else 0
            
        except Exception as e:
            pred_xgb_target = 0
            prob_lgb_important = 0
            vote_sum = 0
            final_danger = 0
        
        results.append({
            'currtime': current_time.strftime('%Y-%m-%d %H:%M') if hasattr(current_time, 'strftime') else str(current_time),
            'current': round(current_total, 0),
            'pred_time': prediction_time.strftime('%Y-%m-%d %H:%M') if hasattr(prediction_time, 'strftime') else str(prediction_time),
            'actual_max': round(float(actual_max), 0),
            'actual_single': round(float(actual_single), 0),
            'predict': round(float(pred_xgb_target), 0),
            'prob': round(float(prob_lgb_important), 3),
            'vote': int(vote_sum),
            'danger': int(final_danger),
            'actual_danger': 1 if actual_max >= limit_val else 0,
        })
    
    if progress_callback:
        progress_callback(total + progress_offset, total + progress_offset)
    
    print(f"  ✅ 예측 완료: {len(results):,}개")
    
    # 통계 계산
    print(f"\n[4] 통계 계산...")
    df_result = pd.DataFrame(results)
    
    actual_danger = df_result['actual_danger'] == 1
    pred_danger = df_result['danger'] == 1
    
    TP = int((actual_danger & pred_danger).sum())
    TN = int((~actual_danger & ~pred_danger).sum())
    FP = int((~actual_danger & pred_danger).sum())
    FN = int((actual_danger & ~pred_danger).sum())
    
    # 예측상태 분류
    df_result['currtime_dt'] = pd.to_datetime(df_result['currtime'])
    
    def get_status(row):
        actual = row['actual_danger']
        pred = row['danger']
        current_time = row['currtime_dt']
        
        if actual == 1 and pred == 1:
            return 'TP'
        elif actual == 0 and pred == 0:
            return 'TN'
        elif actual == 1 and pred == 0:
            for mins in range(1, analysis_range + 1):
                time_ago = current_time - timedelta(minutes=mins)
                prev = df_result[df_result['currtime_dt'] == time_ago]
                if len(prev) > 0 and prev['danger'].values[0] == 1:
                    return f'FN_{mins}분전'
            return 'FN_놓침'
        else:
            for mins in range(1, analysis_range + 1):
                time_later = current_time + timedelta(minutes=mins)
                later = df_result[df_result['currtime_dt'] == time_later]
                if len(later) > 0 and later['actual_danger'].values[0] == 1:
                    return f'FP_{mins}분후'
            return 'FP_오탐'
    
    df_result['status'] = df_result.apply(get_status, axis=1)
    
    status_counts = df_result['status'].value_counts().to_dict()
    
    real_fn = status_counts.get('FN_놓침', 0)
    real_fp = status_counts.get('FP_오탐', 0)
    
    fn_early = sum(status_counts.get(f'FN_{i}분전', 0) for i in range(1, analysis_range + 1))
    fp_valid = sum(status_counts.get(f'FP_{i}분후', 0) for i in range(1, analysis_range + 1))
    
    total_count = TP + TN + FP + FN
    accuracy = round((TP + TN) / total_count * 100, 2) if total_count > 0 else 0
    recall = round(TP / actual_danger.sum() * 100, 2) if actual_danger.sum() > 0 else 0
    precision = round(TP / pred_danger.sum() * 100, 2) if pred_danger.sum() > 0 else 0
    
    return {
        'pred_type': pred_type,
        'data_source': data_source,
        'data_source_name': source_name,
        'date_start': date_start,
        'date_end': date_end,
        'time_start': time_start,
        'time_end': time_end,
        'total_count': len(results),
        'actual_danger_count': int(actual_danger.sum()),
        'missing_cols': missing_cols[:10],
        'stats': {
            'TP': TP,
            'TN': TN,
            'FP': FP,
            'FN': FN,
            'accuracy': accuracy,
            'recall': recall,
            'precision': precision,
            'real_fn': real_fn,
            'real_fp': real_fp,
            'fn_early': fn_early,
            'fp_valid': fp_valid,
        },
        'status_counts': status_counts,
        'analysis_range': analysis_range,
        'data': results[-100:],
    }


def get_available_dates(data_dir):
    """사용 가능한 날짜 목록 반환 (내부용)"""
    dates = []
    
    if not os.path.exists(data_dir):
        return dates
    
    for f in os.listdir(data_dir):
        if f.startswith('m14_data_') and f.endswith('.csv'):
            date_str = f.replace('m14_data_', '').replace('.csv', '')
            if len(date_str) == 8 and date_str.isdigit():
                dates.append(date_str)
    
    return sorted(dates)


# 테스트
if __name__ == "__main__":
    # 내부 테스트
    result = evaluate('data', '20250114', '20250114', '0000', '2359', '10', 'internal')
    print(result)