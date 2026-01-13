"""
M14 V8.3 반송 큐 예측 모델 - 실시간 280분 데이터 조회
현재 시간 기준 280분(약 4시간 40분) 데이터 자동 조회
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

# V8.3 최종 출력 컬럼
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
            print(f"  → 결과: {len(df)} rows")
            return df
        else:
            print(f"  → 에러: Status {resp.status_code}")
            return None
            
    except Exception as e:
        print(f"  → 에러: {e}")
        return None


def get_realtime_280min_data():
    """
    현재 시간 기준 280분 데이터 조회 (M14 V8.3용)
    
    Returns:
        DataFrame: 280개 row (1분 간격)
    """
    # 현재 시간 기준 280분 전 ~ 현재
    now = datetime.now()
    from_time = (now - timedelta(minutes=280)).strftime("%Y%m%d%H%M")
    to_time = now.strftime("%Y%m%d%H%M")
    
    print("=" * 60)
    print("M14 V8.3 실시간 280분 데이터 조회")
    print(f"현재 시간: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"조회 기간: {from_time} ~ {to_time}")
    print("=" * 60)
    
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
        print("  → star_transport_view 없음, ts_current_job만 사용")
        df_merged = df_job
    
    # 최종 컬럼 선택 (없는 컬럼은 NaN)
    for col in FINAL_COLUMNS:
        if col not in df_merged.columns:
            df_merged[col] = None
    
    df_final = df_merged[FINAL_COLUMNS].copy()
    df_final = df_final.sort_values('CURRTIME').reset_index(drop=True)
    
    print(f"\n✓ 최종: {len(df_final)} rows, {len(df_final.columns)} cols")
    
    # 280개 row 확인
    if len(df_final) < 280:
        print(f"⚠ 주의: 280분 미만 데이터 ({len(df_final)}개)")
    
    return df_final


def get_data_for_prediction():
    """
    V8.3 모델 입력용 데이터 반환
    - 최신 280개 row만 반환
    - CURRTIME 컬럼 제외한 숫자 데이터만 반환
    """
    df = get_realtime_280min_data()
    
    if df is None:
        return None
    
    # 최신 280개만 (부족하면 전체)
    df_280 = df.tail(280).reset_index(drop=True)
    
    # 숫자 컬럼만 추출 (CURRTIME 제외)
    numeric_cols = [c for c in FINAL_COLUMNS if c != 'CURRTIME']
    df_numeric = df_280[numeric_cols].copy()
    
    # NaN을 0으로 채우기 (모델 입력용)
    df_numeric = df_numeric.fillna(0)
    
    print(f"\n모델 입력 데이터: {df_numeric.shape}")
    
    return df_numeric


if __name__ == "__main__":
    
    # 방법 1: 전체 데이터 조회 (CURRTIME 포함)
    df = get_realtime_280min_data()
    
    if df is not None and len(df) > 0:
        print("\n" + "=" * 60)
        print("결과 미리보기 (처음 5개):")
        print(df.head())
        
        print("\n결과 미리보기 (마지막 5개):")
        print(df.tail())
        
        # CSV 저장
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"M14_V83_280min_{now_str}.csv"
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"\n저장 완료: {filename}")
        
        # 방법 2: 모델 입력용 데이터 (숫자만)
        print("\n" + "-" * 60)
        print("모델 입력용 데이터 예시:")
        df_model = get_data_for_prediction()
        if df_model is not None:
            print(df_model.tail())
    
    print("\n" + "=" * 60)
    print("완료!")