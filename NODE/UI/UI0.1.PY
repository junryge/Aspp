"""
반송 큐 예측 모델 - 최종 데이터 조회
출력 컬럼: CURRTIME, TOTALCNT, M14AM10A, M10AM14A, M14AM10ASUM, 
          M14AM14B, M14BM14A, M14AM14BSUM, M14AM16, M16M14A, M14AM16SUM,
          M14.QUE.ALL.CURRENTQCREATED, M14.QUE.ALL.CURRENTQCOMPLETED,
          M14.QUE.OHT.OHTUTIL, M14.QUE.ALL.TRANSPORT4MINOVERCNT,
          M14B.QUE.SENDFAB.VERTICALQUEUECOUNT
"""

import requests
import urllib.parse
import pandas as pd
from io import StringIO
from datetime import datetime, timedelta

requests.packages.urllib3.disable_warnings()

HOST = "10.40.42.27"
PORT = 8888
API_KEY = "db1d2335-49cf-e859-3519-1ca132922e38"

# 최종 출력 컬럼
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
    
    print(f"쿼리 실행 중...")
    
    try:
        resp = requests.get(url, verify=False, timeout=timeout)
        
        if resp.status_code == 200 and resp.text.strip() and not resp.text.startswith('<!'):
            df = pd.read_csv(StringIO(resp.text))
            print(f"결과: {len(df)} rows")
            return df
        else:
            print(f"에러: Status {resp.status_code}")
            return None
            
    except Exception as e:
        print(f"에러: {e}")
        return None


def get_transport_queue_data(from_time, to_time):
    """
    반송 큐 데이터 전체 조회
    
    Args:
        from_time: 시작시간 (yyyyMMddHHmm)
        to_time: 종료시간 (yyyyMMddHHmm)
    """
    
    # Step 1: ts_current_job 집계
    print("\n[1/3] ts_current_job 조회...")
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
        print("ts_current_job 조회 실패")
        return None
    
    # Step 2: star_transport_view pivot
    print("\n[2/3] star_transport_view 조회...")
    query_star = f'''
    table from={from_time} to={to_time} star_transport_view
    | eval CURRTIME = string(CRT_TM, "yyyyMMddHHmm")
    | pivot last(IDC_VAL) for IDC_NM by CURRTIME
    | sort CURRTIME
    '''
    df_star = query_logpresso(query_star)
    
    # Step 3: Merge
    print("\n[3/3] 데이터 Merge...")
    if df_star is not None and len(df_star) > 0:
        df_merged = pd.merge(df_job, df_star, on='CURRTIME', how='left')
    else:
        print("star_transport_view 없음, ts_current_job만 사용")
        df_merged = df_job
    
    # 최종 컬럼 선택 (없는 컬럼은 NaN)
    for col in FINAL_COLUMNS:
        if col not in df_merged.columns:
            df_merged[col] = None
    
    df_final = df_merged[FINAL_COLUMNS].copy()
    df_final = df_final.sort_values('CURRTIME').reset_index(drop=True)
    
    print(f"\n최종: {len(df_final)} rows, {len(df_final.columns)} cols")
    
    return df_final


if __name__ == "__main__":
    
    # ============ 조회 기간 설정 ============
    FROM_TIME = "202601051800"  # 시작
    TO_TIME = "202601070700"    # 종료
    
    # 또는 최근 N시간
    # now = datetime.now()
    # FROM_TIME = (now - timedelta(hours=48)).strftime("%Y%m%d%H%M")
    # TO_TIME = now.strftime("%Y%m%d%H%M")
    
    print("=" * 60)
    print("반송 큐 예측 모델 데이터 조회")
    print(f"기간: {FROM_TIME} ~ {TO_TIME}")
    print("=" * 60)
    
    # 데이터 조회
    df = get_transport_queue_data(FROM_TIME, TO_TIME)
    
    if df is not None and len(df) > 0:
        print("\n" + "=" * 60)
        print("결과 미리보기:")
        print(df.head(10))
        
        print(f"\n컬럼: {list(df.columns)}")
        
        # CSV 저장
        filename = f"transport_queue_{FROM_TIME}_{TO_TIME}.csv"
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"\n저장 완료: {filename}")
    else:
        print("\n데이터 조회 실패")
    
    print("\n" + "=" * 60)
    print("완료!")