"""
반송 큐 예측 모델 - 로그프레소 데이터 조회
"""

import requests
import urllib.parse
import pandas as pd
from io import StringIO
from datetime import datetime

requests.packages.urllib3.disable_warnings()

# ============ 설정 ============
HOST = "10.40.42.27"
PORT = 8888
API_KEY = "db1d2335-49cf-e859-3519-1ca132922e38"


def query_logpresso(query, timeout=120):
    """로그프레소 쿼리 실행 → DataFrame 반환"""
    
    # 쿼리 정리 (줄바꿈 → 공백)
    query_clean = ' '.join(query.split())
    
    # URL 인코딩
    encoded = urllib.parse.quote(query_clean, safe='')
    url = f"http://{HOST}:{PORT}/logpresso/httpexport/query.csv?_apikey={API_KEY}&_q={encoded}"
    
    print(f"쿼리 실행 중...")
    print(f"URL 길이: {len(url)}")
    
    try:
        resp = requests.get(url, verify=False, timeout=timeout)
        print(f"Status: {resp.status_code}")
        
        if resp.status_code == 200:
            if resp.text.strip():
                df = pd.read_csv(StringIO(resp.text))
                print(f"결과: {len(df)} rows, {len(df.columns)} cols")
                return df
            else:
                print("결과 없음")
                return pd.DataFrame()
        else:
            print(f"에러: {resp.text[:500]}")
            return None
            
    except Exception as e:
        print(f"에러: {e}")
        return None


def get_transport_queue_data(from_time, to_time):
    """
    반송 큐 예측 모델 입력 데이터 조회
    
    Args:
        from_time: 시작시간 (yyyyMMddHHmm 형식, 예: "202601051800")
        to_time: 종료시간 (yyyyMMddHHmm 형식, 예: "202601070700")
    
    Returns:
        DataFrame with transport queue data
    """
    
    query = f'''
    | set FR = "{from_time}"
    | set TO = "{to_time}"
    | table from=$("FR") to=$("TO") ts_current_job
    | search FAB == "M14"
    | eval A = case(trim(DESTMACHINENAME) == "4ABL_M10" ,1,0)
    | eval B = case(substr(trim(SOURCEMACHINENAME),0,7) == "4ABL330" ,1,0)
    | eval C = case(substr(trim(DESTMACHINENAME),0,4) == "4ALF" ,1,0)
    | eval D = case(substr(trim(SOURCEMACHINENAME),0,4) == "4ALF" ,1,0)
    | eval E = case(substr(trim(DESTMACHINENAME),0,4) == "4AFC" ,1,0)
    | eval F = case(substr(trim(SOURCEMACHINENAME),0,4) == "4AFC" ,1,0)
    | stats sum(A), sum(B), sum(C), sum(D), sum(E), sum(F), count by CURRTIME
    | rename count as TOTALCNT, sum(A) as M14AM10A, sum(B) as M10AM14A, sum(C) as M14AM14B, sum(D) as M14BM14A, sum(E) as M14AM16, sum(F) as M16M14A
    | eval M14AM10ASUM = M10AM14A + M14AM10A,
        M14AM14BSUM = M14AM14B + M14BM14A,
        M14AM16SUM = M14AM16 + M16M14A,
        TIME = CURRTIME
    | join CURRTIME [
        table from=$("FR") to=$("TO") star_transport_view
        | eval CURRTIME = string(CRT_TM, "yyyyMMddHHmm")
        | pivot last(IDC_VAL) for IDC_NM by CURRTIME
    ]
    | fields CURRTIME, TOTALCNT, M14AM10A, M10AM14A, M14AM10ASUM, M14AM14B, M14BM14A, M14AM14BSUM, M14AM16, M16M14A, M14AM16SUM,
        M14.QUE.ALL.CURRENTQCREATED,
        M14.QUE.ALL.CURRENTQCOMPLETED,
        M14.QUE.OHT.OHTUTIL,
        M14.QUE.ALL.TRANSPORT4MINOVERCNT,
        M14B.QUE.SENDFAB.VERTICALQUEUECOUNT
    '''
    
    return query_logpresso(query)


def get_transport_queue_data_simple(from_time, to_time):
    """
    간단 버전 - join 없이 ts_current_job만 조회
    (join이 문제될 경우 사용)
    """
    
    query = f'''
    | set FR = "{from_time}"
    | set TO = "{to_time}"
    | table from=$("FR") to=$("TO") ts_current_job
    | search FAB == "M14"
    | eval A = case(trim(DESTMACHINENAME) == "4ABL_M10" ,1,0)
    | eval B = case(substr(trim(SOURCEMACHINENAME),0,7) == "4ABL330" ,1,0)
    | eval C = case(substr(trim(DESTMACHINENAME),0,4) == "4ALF" ,1,0)
    | eval D = case(substr(trim(SOURCEMACHINENAME),0,4) == "4ALF" ,1,0)
    | eval E = case(substr(trim(DESTMACHINENAME),0,4) == "4AFC" ,1,0)
    | eval F = case(substr(trim(SOURCEMACHINENAME),0,4) == "4AFC" ,1,0)
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
    star_transport_view 테이블만 별도 조회
    """
    
    query = f'''
    | set FR = "{from_time}"
    | set TO = "{to_time}"
    | table from=$("FR") to=$("TO") star_transport_view
    | eval CURRTIME = string(CRT_TM, "yyyyMMddHHmm")
    | pivot last(IDC_VAL) for IDC_NM by CURRTIME
    | sort CURRTIME
    '''
    
    return query_logpresso(query)


if __name__ == "__main__":
    
    # ============ 조회 기간 설정 ============
    FROM_TIME = "202601051800"  # 2026년 1월 5일 18:00
    TO_TIME = "202601070700"    # 2026년 1월 7일 07:00
    
    print("=" * 60)
    print("반송 큐 예측 모델 데이터 조회")
    print(f"기간: {FROM_TIME} ~ {TO_TIME}")
    print("=" * 60)
    
    # 방법 1: 전체 쿼리 (join 포함)
    print("\n[방법 1] 전체 쿼리 실행...")
    df_full = get_transport_queue_data(FROM_TIME, TO_TIME)
    
    if df_full is not None and len(df_full) > 0:
        print("\n결과 미리보기:")
        print(df_full.head(10))
        print(f"\n컬럼: {list(df_full.columns)}")
        
        # CSV 저장
        df_full.to_csv("transport_queue_data.csv", index=False, encoding='utf-8-sig')
        print("\n저장 완료: transport_queue_data.csv")
    
    else:
        # 방법 2: 분리 조회 (join이 안 될 경우)
        print("\n[방법 2] 분리 조회 시도...")
        
        print("\n--- ts_current_job 조회 ---")
        df_job = get_transport_queue_data_simple(FROM_TIME, TO_TIME)
        if df_job is not None and len(df_job) > 0:
            print(df_job.head())
            df_job.to_csv("ts_current_job_data.csv", index=False, encoding='utf-8-sig')
        
        print("\n--- star_transport_view 조회 ---")
        df_star = get_star_transport_view(FROM_TIME, TO_TIME)
        if df_star is not None and len(df_star) > 0:
            print(df_star.head())
            df_star.to_csv("star_transport_view_data.csv", index=False, encoding='utf-8-sig')
        
        # Python에서 merge
        if df_job is not None and df_star is not None:
            print("\n--- Python에서 merge ---")
            df_merged = pd.merge(df_job, df_star, on='CURRTIME', how='left')
            print(f"Merged: {len(df_merged)} rows")
            print(df_merged.head())
            df_merged.to_csv("transport_queue_merged.csv", index=False, encoding='utf-8-sig')
            print("저장 완료: transport_queue_merged.csv")