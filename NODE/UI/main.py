"""
M14 반송 큐 모니터링 - 로그프레소 API 연동
"""

from flask import Flask, jsonify, send_file
import requests
import urllib.parse
import pandas as pd
from io import StringIO
from datetime import datetime, timedelta

requests.packages.urllib3.disable_warnings()

app = Flask(__name__)

# 로그프레소 설정
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


def fetch_realtime_data(minutes=60):
    """
    로그프레소에서 실시간 데이터 조회
    
    Args:
        minutes: 조회할 분 수 (기본 60분)
    
    Returns:
        DataFrame
    """
    now = datetime.now()
    from_time = (now - timedelta(minutes=minutes)).strftime("%Y%m%d%H%M")
    to_time = now.strftime("%Y%m%d%H%M")
    
    print(f"\n[데이터 조회] {from_time} ~ {to_time}")
    
    # Step 1: ts_current_job 집계
    print("[1/3] ts_current_job 조회...")
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
    print("[2/3] star_transport_view 조회...")
    query_star = f'''
    table from={from_time} to={to_time} star_transport_view
    | eval CURRTIME = string(CRT_TM, "yyyyMMddHHmm")
    | pivot last(IDC_VAL) for IDC_NM by CURRTIME
    | sort CURRTIME
    '''
    df_star = query_logpresso(query_star)
    
    # Step 3: Merge
    print("[3/3] 데이터 Merge...")
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
    
    # CURRTIME을 문자열로
    df_final['CURRTIME'] = df_final['CURRTIME'].astype(str)
    
    print(f"✓ 최종: {len(df_final)} rows")
    
    return df_final


def calculate_prediction(df):
    """
    예측값 계산 (임시 로직 - 나중에 ML 모델로 교체)
    
    현재는 간단한 이동평균 + 트렌드 기반 예측
    """
    if df is None or len(df) == 0:
        return [], []
    
    values = df['TOTALCNT'].fillna(0).tolist()
    predict_10 = []
    predict_30 = []
    
    for i in range(len(values)):
        # 최근 5개 평균
        start_idx = max(0, i - 4)
        recent = values[start_idx:i+1]
        avg = sum(recent) / len(recent)
        
        # 트렌드 계산
        if len(recent) >= 2:
            trend = (recent[-1] - recent[0]) / len(recent)
        else:
            trend = 0
        
        # 예측값 (평균 + 트렌드 * 시간)
        p10 = int(avg + trend * 10)
        p30 = int(avg + trend * 30)
        
        # 범위 제한
        p10 = max(1000, min(2000, p10))
        p30 = max(1000, min(2000, p30))
        
        predict_10.append(p10)
        predict_30.append(p30)
    
    return predict_10, predict_30


@app.route('/')
def index():
    return send_file('index.html')


@app.route('/api/data')
def get_data():
    """실시간 데이터 API"""
    
    # 데이터 조회
    df = fetch_realtime_data(60)
    
    if df is None or len(df) == 0:
        return jsonify({'error': 'No data'}), 500
    
    # 예측값 계산
    predict_10_list, predict_30_list = calculate_prediction(df)
    
    # CURRTIME을 HH:MM 형식으로 변환 (그래프용)
    times = []
    times_full = []
    for t in df['CURRTIME'].values:
        t_str = str(t)
        if len(t_str) >= 12:
            times.append(f"{t_str[8:10]}:{t_str[10:12]}")
            times_full.append(f"{t_str[0:4]}-{t_str[4:6]}-{t_str[6:8]} {t_str[8:10]}:{t_str[10:12]}")
        else:
            times.append(t_str)
            times_full.append(t_str)
    
    # 마지막 데이터 시간
    last_t = str(df['CURRTIME'].iloc[-1])
    if len(last_t) >= 12:
        full_time = f"{last_t[0:4]}-{last_t[4:6]}-{last_t[6:8]} {last_t[8:10]}:{last_t[10:12]}"
    else:
        full_time = last_t
    
    current_val = int(df['TOTALCNT'].iloc[-1]) if pd.notna(df['TOTALCNT'].iloc[-1]) else 0
    
    return jsonify({
        'x': times,
        'x_full': times_full,
        'y': df['TOTALCNT'].fillna(0).astype(int).tolist(),
        'predict_10_list': predict_10_list,
        'predict_30_list': predict_30_list,
        'current': current_val,
        'predict_10': predict_10_list[-1] if predict_10_list else current_val,
        'predict_30': predict_30_list[-1] if predict_30_list else current_val,
        'currtime': full_time,
        'idx': len(df),
        'total': len(df)
    })


@app.route('/api/next')
def next_step():
    """다음 스텝 (실시간 모드에서는 최신 데이터 조회)"""
    return get_data()


@app.route('/api/reset')
def reset():
    """리셋"""
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    print('=' * 50)
    print('M14 반송 큐 모니터링 서버 (실시간)')
    print('로그프레소 API 연동')
    print(f'HOST: {HOST}:{PORT}')
    print('http://localhost:5000')
    print('=' * 50)
    app.run(debug=False, port=5000)