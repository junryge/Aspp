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
    """
    현재 시간 기준 N분 데이터 조회 (ML 모델용 280분)
    
    Args:
        minutes: 조회할 분 수 (기본 280분 = 시퀀스 길이)
    
    Returns:
        DataFrame (CURRTIME, TOTALCNT 등 포함)
    """
    now = datetime.now()
    from_time = (now - timedelta(minutes=minutes)).strftime("%Y%m%d%H%M")
    to_time = now.strftime("%Y%m%d%H%M")
    
    print(f"[M14] 데이터 조회: {from_time} ~ {to_time} ({minutes}분)")
    
    # Step 1: ts_current_job 집계
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
        print("[M14] ts_current_job 조회 실패")
        return None
    
    print(f"  → ts_current_job: {len(df_job)} rows")
    
    # Step 2: star_transport_view pivot
    query_star = f'''
    table from={from_time} to={to_time} star_transport_view
    | eval CURRTIME = string(CRT_TM, "yyyyMMddHHmm")
    | pivot last(IDC_VAL) for IDC_NM by CURRTIME
    | sort CURRTIME
    '''
    df_star = query_logpresso(query_star)
    
    # Step 3: Merge
    if df_star is not None and len(df_star) > 0:
        print(f"  → star_transport_view: {len(df_star)} rows")
        df_merged = pd.merge(df_job, df_star, on='CURRTIME', how='left')
    else:
        print("  → star_transport_view: 없음")
        df_merged = df_job
    
    # 최종 컬럼 선택
    for col in FINAL_COLUMNS:
        if col not in df_merged.columns:
            df_merged[col] = 0
    
    df_final = df_merged[FINAL_COLUMNS].copy()
    df_final = df_final.sort_values('CURRTIME').reset_index(drop=True)
    df_final = df_final.fillna(0)
    
    # CURRTIME 문자열 유지 (예측 모듈에서 datetime 변환)
    df_final['CURRTIME'] = df_final['CURRTIME'].astype(str)
    
    print(f"[M14] 최종: {len(df_final)} rows")
    
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
    - 파일로 데이터 저장 (서버 재시작해도 유지)
    - API로 새 데이터만 가져와서 추가
    """
    
    def __init__(self, window_minutes=280, data_file='m14_data.csv'):
        self.window_minutes = window_minutes
        self.data_file = data_file
        self.data = None
        self.predict_10_list = []
        self.predict_30_list = []
        self.last_update = None
        self._predictor_10 = None
        self._predictor_30 = None
    
    def set_predictors(self, pred_10, pred_30):
        """예측 모듈 설정"""
        self._predictor_10 = pred_10
        self._predictor_30 = pred_30
    
    def initialize(self):
        """초기화 - 파일 있으면 로드, 없으면 API에서 가져오기"""
        
        # 1) 파일 있으면 로드
        if os.path.exists(self.data_file):
            print(f"[M14Manager] 파일 로드: {self.data_file}")
            self.data = pd.read_csv(self.data_file)
            self.data['CURRTIME'] = self.data['CURRTIME'].astype(str)
            print(f"  → {len(self.data)} rows 로드")
            
            # 예측값 파일도 로드
            pred_file = self.data_file.replace('.csv', '_pred.csv')
            if os.path.exists(pred_file):
                pred_df = pd.read_csv(pred_file)
                self.predict_10_list = pred_df['PREDICT_10'].tolist()
                self.predict_30_list = pred_df['PREDICT_30'].tolist()
                print(f"  → 예측값 {len(self.predict_10_list)}개 로드")
            else:
                self._calculate_all_predictions()
        
        # 2) 파일 없으면 API에서 가져오기
        else:
            print(f"[M14Manager] 파일 없음, API에서 {self.window_minutes}분 데이터 조회...")
            self.data = get_realtime_data(minutes=self.window_minutes)
            
            if self.data is None or len(self.data) == 0:
                print("[M14Manager] 로드 실패")
                return False
            
            print(f"  → {len(self.data)} rows 조회")
            
            # 예측값 계산
            self._calculate_all_predictions()
            
            # 파일 저장
            self._save_to_file()
        
        self.last_update = datetime.now()
        return True
    
    def _save_to_file(self):
        """데이터 + 예측값 파일 저장"""
        # 데이터 저장
        self.data.to_csv(self.data_file, index=False)
        
        # 예측값 저장
        pred_file = self.data_file.replace('.csv', '_pred.csv')
        pred_df = pd.DataFrame({
            'PREDICT_10': self.predict_10_list,
            'PREDICT_30': self.predict_30_list
        })
        pred_df.to_csv(pred_file, index=False)
        print(f"[M14Manager] 파일 저장 완료")
    
    def _calculate_all_predictions(self):
        """전체 예측값 계산"""
        self.predict_10_list = []
        self.predict_30_list = []
        
        if self._predictor_10 is None or self._predictor_30 is None:
            print("[M14Manager] 예측 모듈 없음, 0으로 채움")
            self.predict_10_list = [0] * len(self.data)
            self.predict_30_list = [0] * len(self.data)
            return
        
        seq_len = 280
        print(f"[M14Manager] 예측값 계산 중... ({len(self.data)}개)")
        
        for i in range(len(self.data)):
            if i + 1 >= seq_len:
                df_slice = self.data.iloc[:i + 1]
                p10 = self._predictor_10.predict(df_slice)
                p30 = self._predictor_30.predict(df_slice)
                self.predict_10_list.append(p10['predict_value'])
                self.predict_30_list.append(p30['predict_value'])
            else:
                self.predict_10_list.append(0)
                self.predict_30_list.append(0)
            
            # 진행상황 (50개마다)
            if (i + 1) % 50 == 0:
                print(f"  → {i + 1}/{len(self.data)}")
        
        print(f"[M14Manager] 예측 완료")
    
    def update(self):
        """새 데이터 1개 추가 (API 호출)"""
        new_row = get_latest_row()
        
        if new_row is None:
            return False
        
        # 중복 체크
        if self.data is not None and len(self.data) > 0:
            if str(new_row.get('CURRTIME')) == str(self.data.iloc[-1]['CURRTIME']):
                return False
        
        # 새 row 추가
        new_df = pd.DataFrame([new_row])
        for col in FINAL_COLUMNS:
            if col not in new_df.columns:
                new_df[col] = 0
        new_df = new_df[FINAL_COLUMNS]
        
        self.data = pd.concat([self.data, new_df], ignore_index=True)
        
        # 예측 1개 계산
        if self._predictor_10 and self._predictor_30 and len(self.data) >= 280:
            p10 = self._predictor_10.predict(self.data)
            p30 = self._predictor_30.predict(self.data)
            self.predict_10_list.append(p10['predict_value'])
            self.predict_30_list.append(p30['predict_value'])
        else:
            self.predict_10_list.append(0)
            self.predict_30_list.append(0)
        
        # 윈도우 유지
        if len(self.data) > self.window_minutes:
            self.data = self.data.tail(self.window_minutes).reset_index(drop=True)
            self.predict_10_list = self.predict_10_list[-self.window_minutes:]
            self.predict_30_list = self.predict_30_list[-self.window_minutes:]
        
        # 파일 저장
        self._save_to_file()
        
        self.last_update = datetime.now()
        return True
    
    def get_data(self):
        return self.data
    
    def get_predictions(self):
        return self.predict_10_list, self.predict_30_list
    
    def get_latest(self):
        if self.data is not None and len(self.data) > 0:
            return self.data.iloc[-1].to_dict()
        return None
    
    def refresh(self):
        """전체 새로고침 (파일 삭제 후 다시)"""
        if os.path.exists(self.data_file):
            os.remove(self.data_file)
        pred_file = self.data_file.replace('.csv', '_pred.csv')
        if os.path.exists(pred_file):
            os.remove(pred_file)
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