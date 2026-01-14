# -*- coding: utf-8 -*-
"""
================================================================================
M14 데이터 조회 모듈
- 로그프레소 API로 280분(시퀀스용) 데이터 조회
- main.py에서 import해서 사용
================================================================================
"""

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
    - 초기: 280분 데이터 로드
    - 업데이트: 새 row 추가 + 윈도우 유지
    """
    
    def __init__(self, window_minutes=280):
        self.window_minutes = window_minutes
        self.data = None
        self.last_update = None
    
    def initialize(self):
        """초기 데이터 로드 (280분)"""
        print(f"[M14Manager] 초기 로드 ({self.window_minutes}분)...")
        self.data = get_realtime_data(minutes=self.window_minutes)
        
        if self.data is not None and len(self.data) > 0:
            self.last_update = datetime.now()
            print(f"[M14Manager] 로드 완료: {len(self.data)} rows")
            return True
        else:
            print("[M14Manager] 로드 실패")
            return False
    
    def update(self):
        """새 데이터 1개 추가"""
        new_row = get_latest_row()
        
        if new_row is None:
            return False
        
        # 새 row DataFrame
        new_df = pd.DataFrame([new_row])
        for col in FINAL_COLUMNS:
            if col not in new_df.columns:
                new_df[col] = 0
        new_df = new_df[FINAL_COLUMNS]
        
        # 중복 체크
        if self.data is not None and len(self.data) > 0:
            if str(new_row.get('CURRTIME')) == str(self.data.iloc[-1]['CURRTIME']):
                return False
        
        # 추가
        if self.data is None:
            self.data = new_df
        else:
            self.data = pd.concat([self.data, new_df], ignore_index=True)
            
            # 윈도우 크기 유지
            if len(self.data) > self.window_minutes:
                self.data = self.data.tail(self.window_minutes).reset_index(drop=True)
        
        self.last_update = datetime.now()
        return True
    
    def get_data(self):
        """현재 데이터 반환"""
        return self.data
    
    def get_latest(self):
        """최신 row 반환"""
        if self.data is not None and len(self.data) > 0:
            return self.data.iloc[-1].to_dict()
        return None
    
    def refresh(self):
        """전체 데이터 새로고침"""
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