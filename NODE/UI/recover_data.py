# -*- coding: utf-8 -*-
"""
================================================================================
M14 데이터 복구 스크립트
- 로그프레소에서 데이터 조회하여 data, pred CSV 파일 재생성
- 사용법: python recover_data.py 20250115 20250119
================================================================================
"""

import os
import sys
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

DATA_DIR = 'data'

# ============================================================================
# 예측 모듈 로드
# ============================================================================
try:
    import predictor_10min
    import predictor_30min
    PREDICTOR_AVAILABLE = True
    print("[복구] 예측 모듈 로드 완료")
except ImportError as e:
    PREDICTOR_AVAILABLE = False
    print(f"[복구] 예측 모듈 로드 실패: {e}")
    print("       → pred 파일은 예측값 0으로 생성됩니다")


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
            print(f"  [에러] Status {resp.status_code}")
            return None
            
    except Exception as e:
        print(f"  [예외] {e}")
        return None


def get_data_for_date(date_str):
    """특정 날짜의 전체 데이터 조회 (00:00 ~ 23:59)"""
    from_time = f"{date_str}0000"
    to_time = f"{date_str}2359"
    
    print(f"\n[조회] {date_str} ({from_time} ~ {to_time})")
    
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
    
    print("  → ts_current_job 조회 중...")
    df_job = query_logpresso(query_job)
    
    if df_job is None or len(df_job) == 0:
        print("  → 데이터 없음")
        return None
    
    print(f"  → {len(df_job)}개 row")
    
    # star_transport_view
    query_star = f'''
    table from={from_time} to={to_time} star_transport_view
    | eval CURRTIME = string(CRT_TM, "yyyyMMddHHmm")
    | pivot last(IDC_VAL) for IDC_NM by CURRTIME
    | sort CURRTIME
    '''
    
    print("  → star_transport_view 조회 중...")
    df_star = query_logpresso(query_star)
    
    # Merge
    if df_star is not None and len(df_star) > 0:
        print(f"  → star 데이터 {len(df_star)}개 병합")
        df_merged = pd.merge(df_job, df_star, on='CURRTIME', how='left')
    else:
        print("  → star 데이터 없음 (job만 사용)")
        df_merged = df_job
    
    # 컬럼 정리
    for col in FINAL_COLUMNS:
        if col not in df_merged.columns:
            df_merged[col] = 0
    
    df_final = df_merged[FINAL_COLUMNS].copy()
    df_final = df_final.sort_values('CURRTIME').reset_index(drop=True)
    df_final = df_final.fillna(0)
    df_final['CURRTIME'] = df_final['CURRTIME'].astype(str)
    
    print(f"  → 최종 {len(df_final)}개 row")
    return df_final


def add_minutes_to_time(time_str, mins):
    """YYYYMMDDHHMM 형식에 분 더하기"""
    try:
        if len(time_str) >= 12:
            dt = datetime.strptime(time_str[:12], "%Y%m%d%H%M")
            dt = dt + timedelta(minutes=mins)
            return dt.strftime("%Y%m%d%H%M")
    except:
        pass
    return time_str


def calculate_predictions(df_all, target_date):
    """예측값 계산 (280분 시퀀스 필요)"""
    seq_len = 280
    
    # 해당 날짜 데이터만 필터
    df_target = df_all[df_all['CURRTIME'].str.startswith(target_date)].copy()
    
    if len(df_target) == 0:
        return []
    
    pred_data = []
    
    for i, row in df_target.iterrows():
        curr_time = str(row['CURRTIME'])
        total_cnt = int(row['TOTALCNT']) if pd.notna(row['TOTALCNT']) else 0
        
        # 예측값 계산
        pred_10 = 0
        pred_30 = 0
        
        if PREDICTOR_AVAILABLE:
            # 현재 위치까지의 데이터 (시퀀스용)
            idx_in_all = df_all[df_all['CURRTIME'] == curr_time].index
            if len(idx_in_all) > 0:
                idx = idx_in_all[0]
                if idx + 1 >= seq_len:
                    df_slice = df_all.iloc[:idx + 1].copy()
                    try:
                        p10 = predictor_10min.predict(df_slice)
                        p30 = predictor_30min.predict(df_slice)
                        pred_10 = p10['predict_value']
                        pred_30 = p30['predict_value']
                    except Exception as e:
                        pass
        
        pred_time_10 = add_minutes_to_time(curr_time, 10)
        pred_time_30 = add_minutes_to_time(curr_time, 30)
        
        pred_data.append({
            'CURRTIME': curr_time,
            'TOTALCNT': total_cnt,
            'PRED_TIME_10': pred_time_10,
            'PREDICT_10': pred_10,
            'PRED_TIME_30': pred_time_30,
            'PREDICT_30': pred_30
        })
    
    return pred_data


def recover_date(date_str, df_all=None):
    """특정 날짜 데이터 복구"""
    
    # 데이터 조회
    df = get_data_for_date(date_str)
    
    if df is None or len(df) == 0:
        print(f"  [실패] {date_str} 데이터 조회 실패")
        return False
    
    # 폴더 생성
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    # data 파일 저장
    data_file = os.path.join(DATA_DIR, f'm14_data_{date_str}.csv')
    df.to_csv(data_file, index=False)
    print(f"  [저장] {data_file} ({len(df)}개)")
    
    # 예측값 계산을 위한 전체 데이터 (이전 데이터 + 현재 데이터)
    if df_all is not None:
        df_combined = pd.concat([df_all, df], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=['CURRTIME']).sort_values('CURRTIME').reset_index(drop=True)
    else:
        df_combined = df
    
    # pred 파일 생성
    print(f"  → 예측값 계산 중...")
    pred_data = calculate_predictions(df_combined, date_str)
    
    if pred_data:
        pred_file = os.path.join(DATA_DIR, f'm14_pred_{date_str}.csv')
        pd.DataFrame(pred_data).to_csv(pred_file, index=False)
        
        # 예측 통계
        pred_10_vals = [p['PREDICT_10'] for p in pred_data]
        pred_30_vals = [p['PREDICT_30'] for p in pred_data]
        over_1700_10 = sum(1 for v in pred_10_vals if v >= 1700)
        over_1700_30 = sum(1 for v in pred_30_vals if v >= 1700)
        
        print(f"  [저장] {pred_file} ({len(pred_data)}개)")
        print(f"         10분 예측 1700+ : {over_1700_10}개")
        print(f"         30분 예측 1700+ : {over_1700_30}개")
    
    return df_combined


def recover_range(start_date, end_date):
    """날짜 범위 복구"""
    print("=" * 60)
    print("M14 데이터 복구 시작")
    print("=" * 60)
    print(f"기간: {start_date} ~ {end_date}")
    print(f"저장 경로: {DATA_DIR}/")
    print("=" * 60)
    
    # 날짜 리스트 생성
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    
    print(f"복구 대상: {len(dates)}일")
    
    # 시퀀스를 위해 시작일 하루 전 데이터도 로드
    prev_date = (start - timedelta(days=1)).strftime("%Y%m%d")
    print(f"\n[사전 로드] {prev_date} (시퀀스용)")
    df_all = get_data_for_date(prev_date)
    
    # 각 날짜 복구
    success_count = 0
    for i, date_str in enumerate(dates):
        print(f"\n[{i+1}/{len(dates)}] {date_str} 복구 중...")
        result = recover_date(date_str, df_all)
        if result is not None and not isinstance(result, bool):
            df_all = result
            success_count += 1
        elif result == True:
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"복구 완료: {success_count}/{len(dates)}일")
    print("=" * 60)


def main():
    if len(sys.argv) < 2:
        print("사용법:")
        print("  python recover_data.py 20250115           # 특정 날짜")
        print("  python recover_data.py 20250115 20250119  # 날짜 범위")
        print("")
        
        # 대화형 모드
        print("날짜를 입력하세요:")
        start_date = input("시작 날짜 (YYYYMMDD): ").strip()
        if not start_date:
            print("취소됨")
            return
        
        end_date = input("종료 날짜 (YYYYMMDD, 같으면 Enter): ").strip()
        if not end_date:
            end_date = start_date
        
        recover_range(start_date, end_date)
    
    elif len(sys.argv) == 2:
        # 단일 날짜
        date_str = sys.argv[1]
        recover_range(date_str, date_str)
    
    else:
        # 날짜 범위
        start_date = sys.argv[1]
        end_date = sys.argv[2]
        recover_range(start_date, end_date)


if __name__ == "__main__":
    main()