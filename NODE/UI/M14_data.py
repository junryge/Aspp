"""
M14 데이터 조회 모듈
- 로그프레소 API를 통한 실시간 데이터 조회
- 1분 간격 데이터 수집
"""

import requests
import urllib.parse
import pandas as pd
from io import StringIO
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')
requests.packages.urllib3.disable_warnings()

# 로그프레소 접속 정보
HOST = "10.40.42.27"
PORT = 8888
API_KEY = "db1d2335-49cf-e859-3519-1ca132922e38"

# 출력 컬럼 정의
FINAL_COLUMNS = [
    'CURRTIME',
    'TOTALCNT',
    'M14AM10A',
    'M10AM14A',
    'M14AM10ASUM',
    'M14AM14B',
    'M14BM14A',
    'M14AM14BSUM',
    'M14AM16',
    'M16M14A',
    'M14AM16SUM',
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
    현재 시간 기준 N분 데이터 조회
    
    Args:
        minutes: 조회할 분 수 (기본 280분)
    
    Returns:
        DataFrame 또는 None
    """
    now = datetime.now()
    from_time = (now - timedelta(minutes=minutes)).strftime("%Y%m%d%H%M")
    to_time = now.strftime("%Y%m%d%H%M")
    
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
        return None
    
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
        df_merged = pd.merge(df_job, df_star, on='CURRTIME', how='left')
    else:
        df_merged = df_job
    
    # 최종 컬럼 선택
    for col in FINAL_COLUMNS:
        if col not in df_merged.columns:
            df_merged[col] = 0
    
    df_final = df_merged[FINAL_COLUMNS].copy()
    df_final = df_final.sort_values('CURRTIME').reset_index(drop=True)
    df_final = df_final.fillna(0)
    
    return df_final


def get_latest_row():
    """
    가장 최신 1개 row만 조회 (빠른 업데이트용)
    
    Returns:
        dict 또는 None
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
    M14 데이터 관리 클래스
    - 초기 데이터 로드
    - 1분마다 새 데이터 추가
    """
    
    def __init__(self, window_minutes=60):
        """
        Args:
            window_minutes: 유지할 데이터 윈도우 크기 (분)
        """
        self.window_minutes = window_minutes
        self.data = None
        self.last_update = None
        self.update_count = 0
    
    def initialize(self):
        """초기 데이터 로드"""
        print(f"[M14] 초기 데이터 로드 중... ({self.window_minutes}분)")
        self.data = get_realtime_data(minutes=self.window_minutes)
        
        if self.data is not None:
            self.last_update = datetime.now()
            print(f"[M14] 초기 로드 완료: {len(self.data)} rows")
            return True
        else:
            print("[M14] 초기 로드 실패")
            return False
    
    def update(self):
        """새 데이터 추가 (1분마다 호출)"""
        new_row = get_latest_row()
        
        if new_row is None:
            print("[M14] 업데이트 실패: 데이터 없음")
            return False
        
        # 새 row 추가
        new_df = pd.DataFrame([new_row])
        
        # 컬럼 맞추기
        for col in FINAL_COLUMNS:
            if col not in new_df.columns:
                new_df[col] = 0
        
        new_df = new_df[FINAL_COLUMNS]
        
        # 중복 체크 (같은 CURRTIME이면 스킵)
        if self.data is not None and len(self.data) > 0:
            if new_row.get('CURRTIME') == self.data.iloc[-1]['CURRTIME']:
                return False  # 같은 시간 데이터
        
        # 데이터 추가
        if self.data is None:
            self.data = new_df
        else:
            self.data = pd.concat([self.data, new_df], ignore_index=True)
            
            # 윈도우 크기 유지
            if len(self.data) > self.window_minutes:
                self.data = self.data.tail(self.window_minutes).reset_index(drop=True)
        
        self.last_update = datetime.now()
        self.update_count += 1
        
        return True
    
    def get_data(self):
        """현재 데이터 반환"""
        return self.data
    
    def get_column(self, col_name):
        """특정 컬럼 데이터 반환"""
        if self.data is not None and col_name in self.data.columns:
            return self.data[col_name].values
        return None
    
    def get_latest(self):
        """최신 row 반환"""
        if self.data is not None and len(self.data) > 0:
            return self.data.iloc[-1].to_dict()
        return None


# 테스트용
if __name__ == "__main__":
    print("=" * 50)
    print("M14 데이터 모듈 테스트")
    print("=" * 50)
    
    # 방법 1: 직접 조회
    print("\n[테스트 1] 60분 데이터 조회")
    df = get_realtime_data(minutes=60)
    if df is not None:
        print(f"결과: {len(df)} rows")
        print(df.tail(3))
    
    # 방법 2: 최신 1개
    print("\n[테스트 2] 최신 데이터 조회")
    latest = get_latest_row()
    if latest:
        print(f"CURRTIME: {latest.get('CURRTIME')}")
        print(f"TOTALCNT: {latest.get('TOTALCNT')}")
    
    # 방법 3: 데이터 매니저
    print("\n[테스트 3] 데이터 매니저")
    mgr = M14DataManager(window_minutes=30)
    if mgr.initialize():
        print(f"TOTALCNT 마지막 5개: {mgr.get_column('TOTALCNT')[-5:]}")