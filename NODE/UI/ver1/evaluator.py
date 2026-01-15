# -*- coding: utf-8 -*-
"""
================================================================================
M14 예측 평가 모듈
- data 폴더의 m14_data_YYYYMMDD.csv 파일 사용
- 10분/30분 예측 평가
- 웹 API용
================================================================================
"""

import os
import pickle
import warnings
import gc
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# ============================================================================
# 설정
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
            # 컬럼 없으면 0으로 채움
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
    
    # CURRTIME datetime 변환
    if not pd.api.types.is_datetime64_any_dtype(df['CURRTIME']):
        df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M', errors='coerce')
    
    df = df.dropna(subset=['CURRTIME']).sort_values('CURRTIME').reset_index(drop=True)
    
    # QUEUE_GAP 파생
    if 'M14.QUE.ALL.CURRENTQCREATED' in df.columns and 'M14.QUE.ALL.CURRENTQCOMPLETED' in df.columns:
        df['QUEUE_GAP'] = df['M14.QUE.ALL.CURRENTQCREATED'] - df['M14.QUE.ALL.CURRENTQCOMPLETED']
    
    # 모든 feature 컬럼 확인 및 생성
    all_cols = []
    for group in feature_groups.values():
        all_cols.extend(group)
    
    for col in list(set(all_cols)):
        if col not in df.columns:
            df[col] = 0  # 없는 컬럼은 0으로 생성
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df


def load_data_files(data_dir, date_start, date_end):
    """날짜 범위의 데이터 파일 로드"""
    all_data = []
    
    # 날짜 범위 생성
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
                print(f"  로드: {file_path} ({len(df)}행)")
            except Exception as e:
                print(f"  에러: {file_path} - {e}")
        
        current += timedelta(days=1)
    
    if not all_data:
        return None
    
    # 병합 및 정렬
    df = pd.concat(all_data, ignore_index=True)
    df = df.drop_duplicates(subset=['CURRTIME']).sort_values('CURRTIME').reset_index(drop=True)
    
    return df


def evaluate(data_dir, date_start, date_end, pred_type='10'):
    """
    평가 수행
    
    Args:
        data_dir: 데이터 폴더 경로
        date_start: 시작 날짜 (YYYYMMDD)
        date_end: 종료 날짜 (YYYYMMDD)
        pred_type: '10' 또는 '30'
    
    Returns:
        dict: 평가 결과
    """
    print(f"\n[평가] {pred_type}분 예측 평가 시작")
    print(f"  기간: {date_start} ~ {date_end}")
    
    # 설정 및 모델 선택
    if pred_type == '10':
        config = CONFIG_10
        model_data = load_model_10()
        analysis_range = 5  # FN/FP 분석 범위
    else:
        config = CONFIG_30
        model_data = load_model_30()
        analysis_range = 20
    
    if model_data is None:
        return {'error': f'{pred_type}분 모델을 찾을 수 없습니다'}
    
    models = model_data['models']
    scalers = model_data['scalers']
    feature_groups = model_data['feature_groups']
    
    # 데이터 로드
    print(f"\n[1] 데이터 로드...")
    df = load_data_files(data_dir, date_start, date_end)
    
    if df is None or len(df) == 0:
        return {'error': '해당 기간에 데이터가 없습니다'}
    
    print(f"  총 {len(df):,}행 로드됨")
    
    # 전처리
    print(f"\n[2] 데이터 전처리...")
    df = preprocess_data(df, feature_groups)
    print(f"  전처리 후: {len(df):,}행")
    
    # 누락 컬럼 확인
    missing_cols = []
    for group_name, cols in feature_groups.items():
        for col in cols:
            if col not in df.columns or df[col].sum() == 0:
                missing_cols.append(col)
    
    if missing_cols:
        print(f"  ⚠ 누락/0인 컬럼: {len(missing_cols)}개")
    
    # 평가 수행
    print(f"\n[3] 예측 수행...")
    seq_len = config['sequence_length']
    pred_offset = config['prediction_offset']
    limit_val = config['limit_value']
    target_col = config['target_column']
    
    if len(df) < seq_len + pred_offset:
        return {'error': f'데이터가 부족합니다 (최소 {seq_len + pred_offset}행 필요, 현재 {len(df)}행)'}
    
    results = []
    total = len(df) - seq_len - pred_offset
    
    for idx in range(seq_len, len(df) - pred_offset):
        if (idx - seq_len) % 500 == 0:
            print(f"    진행: {idx - seq_len:,}/{total:,}")
            gc.collect()
        
        current_time = df['CURRTIME'].iloc[idx - 1]
        current_total = float(df[target_col].iloc[idx - 1]) if pd.notna(df[target_col].iloc[idx - 1]) else 0
        prediction_time = current_time + timedelta(minutes=pred_offset)
        
        # 실제값: N분 내 최대값
        future_end = min(idx - 1 + pred_offset, len(df))
        actual_max = df[target_col].iloc[idx - 1:future_end].max()
        
        # 실제단일값: 정확히 N분 후
        actual_single_idx = idx - 1 + pred_offset
        actual_single = df[target_col].iloc[actual_single_idx] if actual_single_idx < len(df) else df[target_col].iloc[-1]
        
        try:
            # Feature 생성
            feat_target = create_sequence_features(df, feature_groups['target'], seq_len, idx)
            feat_important = create_sequence_features(df, feature_groups['important'], seq_len, idx)
            feat_auxiliary = create_sequence_features(df, feature_groups['auxiliary'], seq_len, idx)
            
            X_target = scalers['target'].transform([feat_target])
            X_important = scalers['important'].transform([feat_important])
            X_auxiliary = scalers['auxiliary'].transform([feat_auxiliary])
            
            # XGB 예측
            pred_xgb_target = models['xgb_target'].predict(X_target)[0]
            pred_xgb_important = models['xgb_important'].predict(X_important)[0]
            pred_xgb_auxiliary = models['xgb_auxiliary'].predict(X_auxiliary)[0]
            
            # LGBM 예측
            pred_lgb_target = models['lgb_target'].predict(X_target)[0]
            pred_lgb_important = models['lgb_important'].predict(X_important)[0]
            pred_lgb_auxiliary = models['lgb_auxiliary'].predict(X_auxiliary)[0]
            prob_lgb_important = models['lgb_important'].predict_proba(X_important)[0][1]
            
            # PDT 모델
            pred_xgb_pdt = None
            if 'xgb_pdt_new' in models and 'pdt_new' in scalers and 'pdt_new' in feature_groups:
                feat_pdt = create_sequence_features(df, feature_groups.get('pdt_new', []), seq_len, idx)
                if feat_pdt:
                    X_pdt = scalers['pdt_new'].transform([feat_pdt])
                    pred_xgb_pdt = models['xgb_pdt_new'].predict(X_pdt)[0]
            
            # 투표
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
            
            # 최종 판정
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
    
    # 상태별 집계
    status_counts = df_result['status'].value_counts().to_dict()
    
    # 실질 FN/FP
    real_fn = status_counts.get('FN_놓침', 0)
    real_fp = status_counts.get('FP_오탐', 0)
    
    # 조기감지/유효경고 합계
    fn_early = sum(status_counts.get(f'FN_{i}분전', 0) for i in range(1, analysis_range + 1))
    fp_valid = sum(status_counts.get(f'FP_{i}분후', 0) for i in range(1, analysis_range + 1))
    
    # 정확도
    total_count = TP + TN + FP + FN
    accuracy = round((TP + TN) / total_count * 100, 2) if total_count > 0 else 0
    recall = round(TP / actual_danger.sum() * 100, 2) if actual_danger.sum() > 0 else 0
    precision = round(TP / pred_danger.sum() * 100, 2) if pred_danger.sum() > 0 else 0
    
    # 결과 반환
    return {
        'pred_type': pred_type,
        'date_start': date_start,
        'date_end': date_end,
        'total_count': len(results),
        'actual_danger_count': int(actual_danger.sum()),
        'missing_cols': missing_cols[:10],  # 최대 10개만
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
        'data': results[-100:],  # 최근 100개만 반환 (UI용)
    }


def get_available_dates(data_dir):
    """사용 가능한 날짜 목록 반환"""
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
    result = evaluate('data', '20250114', '20250114', '10')
    print(result)