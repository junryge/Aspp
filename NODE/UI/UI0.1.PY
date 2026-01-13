"""
반송 큐 예측 모델 - 로그프레소 데이터 조회
(절대시간 방식 - from/to 직접 지정)
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


def query_logpresso(query, timeout=120):
    """로그프레소 쿼리 실행"""
    query_clean = ' '.join(query.split())
    encoded = urllib.parse.quote(query_clean, safe='')
    url = f"http://{HOST}:{PORT}/logpresso/httpexport/query.csv?_apikey={API_KEY}&_q={encoded}"
    
    print(f"쿼리 실행 중...")
    print(f"쿼리: {query_clean[:100]}...")
    
    try:
        resp = requests.get(url, verify=False, timeout=timeout)
        print(f"Status: {resp.status_code}")
        
        if resp.status_code == 200 and resp.text.strip() and not resp.text.startswith('<!'):
            df = pd.read_csv(StringIO(resp.text))
            print(f"결과: {len(df)} rows, {len(df.columns)} cols")
            return df
        else:
            print(f"에러 또는 빈 데이터")
            if resp.status_code != 200:
                print(f"응답: {resp.text[:300]}")
            return None
            
    except Exception as e:
        print(f"에러: {e}")
        return None


def get_ts_current_job(from_time, to_time):
    """
    ts_current_job 데이터 조회 + 집계
    """
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
    | sort CURRTIME
    '''
    return query_logpresso(query)


def get_star_transport_view(from_time, to_time):
    """
    star_transport_view 데이터 조회 + pivot
    """
    query = f'''
    table from={from_time} to={to_time} star_transport_view
    | eval CURRTIME = string(CRT_TM, "yyyyMMddHHmm")
    | pivot last(IDC_VAL) for IDC_NM by CURRTIME
    | sort CURRTIME
    '''
    return query_logpresso(query)


def get_star_transport_simple(from_time, to_time):
    """
    star_transport_view 원본 데이터 (pivot 없이)
    """
    query = f'''
    table from={from_time} to={to_time} star_transport_view
    | fields CRT_TM, IDC_NM, IDC_VAL
    | sort CRT_TM
    '''
    return query_logpresso(query)


def merge_data(df_job, df_star):
    """
    Python에서 두 데이터 merge
    """
    if df_job is None or df_star is None:
        print("merge 불가: 데이터 없음")
        return None
    
    # CURRTIME 기준 merge
    df_merged = pd.merge(df_job, df_star, on='CURRTIME', how='left')
    print(f"Merged: {len(df_merged)} rows")
    return df_merged


if __name__ == "__main__":
    
    # ============ 조회 기간 설정 ============
    # 방법 1: 직접 지정
    FROM_TIME = "202601051800"
    TO_TIME = "202601070700"
    
    # 방법 2: 현재 시간 기준 (최근 N시간)
    # now = datetime.now()
    # FROM_TIME = (now - timedelta(hours=24)).strftime("%Y%m%d%H%M")
    # TO_TIME = now.strftime("%Y%m%d%H%M")
    
    print("=" * 60)
    print("반송 큐 예측 모델 데이터 조회")
    print(f"기간: {FROM_TIME} ~ {TO_TIME}")
    print("=" * 60)
    
    # Step 1: ts_current_job 조회
    print("\n[Step 1] ts_current_job 조회")
    df_job = get_ts_current_job(FROM_TIME, TO_TIME)
    
    if df_job is not None and len(df_job) > 0:
        print("\nts_current_job 결과:")
        print(df_job.head())
        print(f"\n컬럼: {list(df_job.columns)}")
        df_job.to_csv("ts_current_job_result.csv", index=False, encoding='utf-8-sig')
        print("저장: ts_current_job_result.csv")
    
    # Step 2: star_transport_view 조회
    print("\n" + "=" * 60)
    print("[Step 2] star_transport_view 조회")
    df_star = get_star_transport_view(FROM_TIME, TO_TIME)
    
    if df_star is not None and len(df_star) > 0:
        print("\nstar_transport_view 결과:")
        print(df_star.head())
        print(f"\n컬럼: {list(df_star.columns)}")
        df_star.to_csv("star_transport_view_result.csv", index=False, encoding='utf-8-sig')
        print("저장: star_transport_view_result.csv")
    else:
        # pivot이 안 되면 원본 조회
        print("\npivot 실패, 원본 조회 시도...")
        df_star_raw = get_star_transport_simple(FROM_TIME, TO_TIME)
        if df_star_raw is not None:
            print(df_star_raw.head(10))
            df_star_raw.to_csv("star_transport_raw.csv", index=False, encoding='utf-8-sig')
    
    # Step 3: Python에서 merge
    print("\n" + "=" * 60)
    print("[Step 3] 데이터 Merge")
    if df_job is not None and df_star is not None:
        df_merged = merge_data(df_job, df_star)
        if df_merged is not None:
            print("\nMerged 결과:")
            print(df_merged.head())
            df_merged.to_csv("transport_queue_final.csv", index=False, encoding='utf-8-sig')
            print("\n최종 저장: transport_queue_final.csv")
    
    print("\n" + "=" * 60)
    print("완료!")