# -*- coding: utf-8 -*-
"""
================================================================================
M14 데이터 조회 모듈
- 로그프레소 API로 280분(시퀀스용) 데이터 조회
- main.py에서 import해서 사용
================================================================================
"""

import os
import requests
import urllib.parse
import pandas as pd
from io import StringIO
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')
requests.packages.urllib3.disable_warnings()

# ============================================================================
# 로그프레소 설정
# ============================================================================
HOST = "10.40.42.27"
PORT = 8888
API_KEY = "db1d2335-49cf-e859-3519-1ca132922e38"

# 출력 컬럼
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


def query_logpresso(query, timeout=180):
    """로그프레소 쿼리 실행"""
    query_clean = ' '.join(query.split())
    encoded = urllib.parse.quote(query_clean, safe='')
    url = f"http://{HOST}:{PORT}/logpresso/httpexport/query.csv?_apikey={API_KEY}&_q={encoded}"
    
    try:
        resp = requests.get(url, verify=False, timeout=timeout)
        
        if resp.status_code == 200 and resp.text.strip() and not resp.text.startswith('<!'):
            df = pd.read_csv(StringIO(resp.text))
            return df
        else:
            print(f"[M14] 쿼리 에러: Status {resp.status_code}")
            return None
            
    except Exception as e:
        print(f"[M14] 쿼리 예외: {e}")
        return None


def get_realtime_data(minutes=280):
    """현재 시간 기준 N분 데이터 조회"""
    now = datetime.now()
    from_time = (now - timedelta(minutes=minutes)).strftime("%Y%m%d%H%M")
    to_time = now.strftime("%Y%m%d%H%M")
    return get_realtime_data_range(from_time, to_time)


def get_realtime_data_range(from_time, to_time):
    """특정 시간대 데이터 조회"""
    print(f"[M14] 조회: {from_time} ~ {to_time}")
    
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


def get_latest_row():
    """
    최신 1개 row만 조회 (빠른 업데이트용)
    """
    now = datetime.now()
    from_time = (now - timedelta(minutes=5)).strftime("%Y%m%d%H%M")
    to_time = now.strftime("%Y%m%d%H%M")
    
    query = f'''
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
    | sort -CURRTIME
    | limit 1
    '''
    df = query_logpresso(query, timeout=30)
    
    if df is not None and len(df) > 0:
        return df.iloc[0].to_dict()
    return None


class M14DataManager:
    """
    M14 데이터 관리자
    - 날짜별 파일 저장 (m14_data_20250114.csv)
    - 알람 기록도 저장 (m14_alert_20250114.csv)
    - 알람 상태 저장 (쿨타임, 알람 횟수)
    """
    
    ALARM_COOLDOWN = 60 * 60  # 쿨타임 1시간 (초)
    
    def __init__(self, window_minutes=280, data_dir='data'):
        self.window_minutes = window_minutes
        self.data_dir = data_dir
        self.data = None
        self.predict_10_list = []
        self.predict_30_list = []
        self.alert_10_list = []  # 10분 예측 1700+ 알람 기록
        self.alert_30_list = []  # 30분 예측 1700+ 알람 기록
        self.alarm_count = 0     # 총 알람 발생 횟수
        self.last_alarm_time = None  # 마지막 알람 시간 (쿨타임용)
        self.last_update = None
        self._predictor_10 = None
        self._predictor_30 = None
        
        # 폴더 생성
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
    
    def _get_file_path(self, date_str):
        return os.path.join(self.data_dir, f'm14_data_{date_str}.csv')
    
    def _get_pred_file_path(self, date_str):
        return os.path.join(self.data_dir, f'm14_pred_{date_str}.csv')
    
    def _get_alert_file_path(self, date_str):
        return os.path.join(self.data_dir, f'm14_alert_{date_str}.csv')
    
    def set_predictors(self, pred_10, pred_30):
        self._predictor_10 = pred_10
        self._predictor_30 = pred_30
    
    def _calc_alarm_state_from_alerts(self):
        """alert 기록에서 알람 상태 계산 (서버 재시작 시 사용)"""
        all_alerts = self.alert_10_list + self.alert_30_list
        
        # IS_ALARM=True인 것만 필터
        alarm_records = [a for a in all_alerts if a.get('IS_ALARM') == True]
        
        if not alarm_records:
            self.alarm_count = 0
            self.last_alarm_time = None
            print(f"[M14] 알람 상태 계산: count=0, last=None")
            return
        
        # 알람 번호 중 최대값 = 오늘 알람 횟수
        self.alarm_count = max(a.get('ALARM_NO', 0) for a in alarm_records)
        
        # 가장 마지막 알람 시간 찾기
        last_timestamp = None
        for a in alarm_records:
            ts = a.get('TIMESTAMP')
            if ts:
                try:
                    if isinstance(ts, str):
                        t = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
                    else:
                        t = ts
                    if last_timestamp is None or t > last_timestamp:
                        last_timestamp = t
                except:
                    pass
        
        self.last_alarm_time = last_timestamp
        print(f"[M14] 알람 상태 계산: count={self.alarm_count}, last={self.last_alarm_time}")
    
    def _is_in_cooldown(self):
        """쿨타임 중인지 확인"""
        if self.last_alarm_time is None:
            return False
        elapsed = (datetime.now() - self.last_alarm_time).total_seconds()
        return elapsed < self.ALARM_COOLDOWN
    
    def _get_cooldown_mins(self):
        """남은 쿨타임 (분)"""
        if self.last_alarm_time is None:
            return 0
        elapsed = (datetime.now() - self.last_alarm_time).total_seconds()
        remaining = self.ALARM_COOLDOWN - elapsed
        return max(0, int(remaining / 60))
    
    def initialize(self):
        """초기화"""
        today = datetime.now().strftime("%Y%m%d")
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
        
        all_data = []
        all_pred_10 = []
        all_pred_30 = []
        
        for date_str in [yesterday, today]:
            data_file = self._get_file_path(date_str)
            pred_file = self._get_pred_file_path(date_str)
            
            if os.path.exists(data_file):
                print(f"[M14] 파일 로드: {data_file}")
                df = pd.read_csv(data_file)
                df['CURRTIME'] = df['CURRTIME'].astype(str)
                all_data.append(df)
                
                if os.path.exists(pred_file):
                    pred_df = pd.read_csv(pred_file)
                    all_pred_10.extend(pred_df['PREDICT_10'].tolist())
                    all_pred_30.extend(pred_df['PREDICT_30'].tolist())
        
        # 오늘 알람 기록 로드
        alert_file = self._get_alert_file_path(today)
        if os.path.exists(alert_file):
            alert_df = pd.read_csv(alert_file)
            # TYPE을 문자열로 변환 (CSV에서 정수로 읽힐 수 있음)
            alert_df['TYPE'] = alert_df['TYPE'].astype(str)
            self.alert_10_list = alert_df[alert_df['TYPE'] == '10'].to_dict('records')
            self.alert_30_list = alert_df[alert_df['TYPE'] == '30'].to_dict('records')
            print(f"[M14] 알람 로드: 10분={len(self.alert_10_list)}개, 30분={len(self.alert_30_list)}개")
        else:
            # 파일 없으면 빈 리스트로 초기화
            self.alert_10_list = []
            self.alert_30_list = []
            print(f"[M14] 알람 기록 없음 (신규)")
        
        # alert 기록에서 알람 상태 계산 (alarm_count, last_alarm_time)
        self._calc_alarm_state_from_alerts()
        
        if all_data:
            self.data = pd.concat(all_data, ignore_index=True)
            self.data = self.data.drop_duplicates(subset=['CURRTIME']).sort_values('CURRTIME').reset_index(drop=True)
            self.predict_10_list = all_pred_10
            self.predict_30_list = all_pred_30
            print(f"[M14] 로드 완료: {len(self.data)} rows")
            self._fetch_missing()
        else:
            print(f"[M14] 파일 없음, API 조회...")
            self.data = get_realtime_data(minutes=self.window_minutes)
            
            if self.data is None or len(self.data) == 0:
                return False
            
            self._calculate_predictions_for_new(len(self.data))
            self._save()
        
        if len(self.data) > self.window_minutes:
            self.data = self.data.tail(self.window_minutes).reset_index(drop=True)
            self.predict_10_list = self.predict_10_list[-self.window_minutes:] if len(self.predict_10_list) > self.window_minutes else self.predict_10_list
            self.predict_30_list = self.predict_30_list[-self.window_minutes:] if len(self.predict_30_list) > self.window_minutes else self.predict_30_list
        
        self.last_update = datetime.now()
        return True
    
    def _fetch_missing(self):
        if self.data is None or len(self.data) == 0:
            return
        
        last_time = str(self.data['CURRTIME'].iloc[-1])
        print(f"[M14] 마지막 데이터: {last_time}")
        
        try:
            from_dt = datetime.strptime(last_time, "%Y%m%d%H%M") + timedelta(minutes=1)
            from_time = from_dt.strftime("%Y%m%d%H%M")
            to_time = datetime.now().strftime("%Y%m%d%H%M")
            
            if from_time >= to_time:
                print(f"[M14] 빠진 데이터 없음")
                return
            
            print(f"[M14] 빠진 데이터 조회: {from_time} ~ {to_time}")
            new_data = get_realtime_data_range(from_time, to_time)
            
            if new_data is not None and len(new_data) > 0:
                print(f"  → {len(new_data)}개 추가")
                
                old_len = len(self.data)
                self.data = pd.concat([self.data, new_data], ignore_index=True)
                self.data = self.data.drop_duplicates(subset=['CURRTIME']).sort_values('CURRTIME').reset_index(drop=True)
                
                new_count = len(self.data) - old_len
                if new_count > 0:
                    self._calculate_predictions_for_new(new_count)
                
                self._save()
                
        except Exception as e:
            print(f"[M14] 빠진 데이터 조회 실패: {e}")
    
    def _calculate_predictions_for_new(self, new_count):
        if self._predictor_10 is None or self._predictor_30 is None:
            self.predict_10_list.extend([0] * new_count)
            self.predict_30_list.extend([0] * new_count)
            return
        
        seq_len = 280
        start_idx = len(self.data) - new_count
        
        for i in range(start_idx, len(self.data)):
            if i + 1 >= seq_len:
                df_slice = self.data.iloc[:i + 1]
                p10 = self._predictor_10.predict(df_slice)
                p30 = self._predictor_30.predict(df_slice)
                pred_10_val = p10['predict_value']
                pred_30_val = p30['predict_value']
                self.predict_10_list.append(pred_10_val)
                self.predict_30_list.append(pred_30_val)
                
                # 1700+ 알람 기록
                curr_time = str(self.data['CURRTIME'].iloc[i])
                if pred_10_val >= 1700:
                    self._add_alert('10', curr_time, pred_10_val)
                if pred_30_val >= 1700:
                    self._add_alert('30', curr_time, pred_30_val)
            else:
                self.predict_10_list.append(0)
                self.predict_30_list.append(0)
    
    def _add_alert(self, alert_type, curr_time, value):
        """알람 기록 추가 (쿨타임, 알람번호 포함)"""
        alert_list = self.alert_10_list if alert_type == '10' else self.alert_30_list
        
        # 중복 체크
        if any(str(a.get('CURRTIME')) == curr_time for a in alert_list):
            return
        
        # 쿨타임 체크
        in_cooldown = self._is_in_cooldown()
        cooldown_mins = self._get_cooldown_mins() if in_cooldown else 0
        
        # 알람 발생 여부
        is_alarm = not in_cooldown
        
        if is_alarm:
            self.alarm_count += 1
            self.last_alarm_time = datetime.now()
        
        alert_list.append({
            'TYPE': alert_type,
            'CURRTIME': curr_time,
            'VALUE': value,
            'TIMESTAMP': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'ALARM_NO': self.alarm_count,
            'IS_ALARM': is_alarm,
            'COOLDOWN_MINS': cooldown_mins if not is_alarm else 0
        })
    
    def _save(self):
        if self.data is None or len(self.data) == 0:
            return
        
        self.data['DATE'] = self.data['CURRTIME'].str[:8]
        
        for date_str, group in self.data.groupby('DATE'):
            data_file = self._get_file_path(date_str)
            pred_file = self._get_pred_file_path(date_str)
            
            group_save = group.drop(columns=['DATE'])
            group_save.to_csv(data_file, index=False)
            
            # 예측 파일: CURRTIME, TOTALCNT, 예측시간, 예측값
            indices = group.index.tolist()
            pred_data = []
            
            for idx in indices:
                curr_time = str(self.data['CURRTIME'].iloc[idx])
                total_cnt = int(self.data['TOTALCNT'].iloc[idx]) if pd.notna(self.data['TOTALCNT'].iloc[idx]) else 0
                pred_10 = self.predict_10_list[idx] if idx < len(self.predict_10_list) else 0
                pred_30 = self.predict_30_list[idx] if idx < len(self.predict_30_list) else 0
                
                pred_time_10 = self._add_minutes_to_time(curr_time, 10)
                pred_time_30 = self._add_minutes_to_time(curr_time, 30)
                
                pred_data.append({
                    'CURRTIME': curr_time,
                    'TOTALCNT': total_cnt,
                    'PRED_TIME_10': pred_time_10,
                    'PREDICT_10': pred_10,
                    'PRED_TIME_30': pred_time_30,
                    'PREDICT_30': pred_30
                })
            
            pd.DataFrame(pred_data).to_csv(pred_file, index=False)
        
        self.data = self.data.drop(columns=['DATE'])
        self._save_alerts()
        print(f"[M14] 파일 저장 완료")
    
    def _add_minutes_to_time(self, time_str, mins):
        """YYYYMMDDHHMM 형식에 분 더하기"""
        try:
            if len(time_str) >= 12:
                dt = datetime.strptime(time_str[:12], "%Y%m%d%H%M")
                dt = dt + timedelta(minutes=mins)
                return dt.strftime("%Y%m%d%H%M")
        except:
            pass
        return time_str
        
        self.data = self.data.drop(columns=['DATE'])
        self._save_alerts()
        print(f"[M14] 파일 저장 완료")
    
    def _save_alerts(self):
        today = datetime.now().strftime("%Y%m%d")
        alert_file = self._get_alert_file_path(today)
        
        all_alerts = self.alert_10_list + self.alert_30_list
        
        if all_alerts:
            pd.DataFrame(all_alerts).to_csv(alert_file, index=False)
    
    def update(self):
        self._fetch_missing()
        
        if len(self.data) > self.window_minutes:
            self.data = self.data.tail(self.window_minutes).reset_index(drop=True)
            self.predict_10_list = self.predict_10_list[-self.window_minutes:]
            self.predict_30_list = self.predict_30_list[-self.window_minutes:]
        
        self.last_update = datetime.now()
        return True
    
    def get_data(self):
        return self.data
    
    def get_predictions(self):
        return self.predict_10_list, self.predict_30_list
    
    def get_alerts(self):
        return self.alert_10_list, self.alert_30_list
    
    def get_alarm_state(self):
        """알람 상태 반환 (웹에서 사용)"""
        return {
            'alarm_count': self.alarm_count,
            'last_alarm_time': self.last_alarm_time.strftime('%Y-%m-%d %H:%M:%S') if self.last_alarm_time else None,
            'in_cooldown': self._is_in_cooldown(),
            'cooldown_mins': self._get_cooldown_mins()
        }
    
    def get_latest(self):
        if self.data is not None and len(self.data) > 0:
            return self.data.iloc[-1].to_dict()
        return None
    
    def load_date(self, date_str):
        data_file = self._get_file_path(date_str)
        pred_file = self._get_pred_file_path(date_str)
        alert_file = self._get_alert_file_path(date_str)
        
        if os.path.exists(data_file):
            df = pd.read_csv(data_file)
            df['CURRTIME'] = df['CURRTIME'].astype(str)
            
            pred_10, pred_30 = [], []
            if os.path.exists(pred_file):
                pred_df = pd.read_csv(pred_file)
                pred_10 = pred_df['PREDICT_10'].tolist()
                pred_30 = pred_df['PREDICT_30'].tolist()
            
            alert_10, alert_30 = [], []
            if os.path.exists(alert_file):
                alert_df = pd.read_csv(alert_file)
                alert_10 = alert_df[alert_df['TYPE'] == '10'].to_dict('records')
                alert_30 = alert_df[alert_df['TYPE'] == '30'].to_dict('records')
            
            return df, pred_10, pred_30, alert_10, alert_30
        return None, [], [], [], []
    
    def refresh(self):
        today = datetime.now().strftime("%Y%m%d")
        for f in [self._get_file_path(today), self._get_pred_file_path(today), self._get_alert_file_path(today)]:
            if os.path.exists(f):
                os.remove(f)
        
        self.data = None
        self.predict_10_list = []
        self.predict_30_list = []
        self.alert_10_list = []
        self.alert_30_list = []
        # 알람 상태는 유지 (쿨타임 때문에)
        return self.initialize()


# 테스트
if __name__ == "__main__":
    print("=" * 60)
    print("M14 데이터 모듈 테스트")
    print("=" * 60)
    
    # 280분 데이터 조회
    df = get_realtime_data(minutes=280)
    if df is not None:
        print(f"\n결과: {len(df)} rows")
        print(df[['CURRTIME', 'TOTALCNT']].tail(5))