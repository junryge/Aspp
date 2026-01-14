"""
M14 반송 큐 모니터링 - Plotly
"""

from flask import Flask, jsonify, send_file
import pandas as pd
import os

app = Flask(__name__)

CSV_FILE = 'sample_m14_data.csv'
current_idx = 30
df = None


def load_csv():
    global df
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        df['CURRTIME'] = df['CURRTIME'].astype(str)
        print(f'[OK] CSV 로드: {len(df)} rows')
        return True
    print(f'[ERROR] {CSV_FILE} 없음')
    return False


@app.route('/')
def index():
    return send_file('index.html')


@app.route('/api/data')
def get_data():
    global current_idx, df
    
    if df is None:
        return jsonify({'error': 'No data'}), 500
    
    # 최근 60개 (1시간)
    start = max(0, current_idx - 59)
    end = current_idx + 1
    
    window = df.iloc[start:end]
    
    # CURRTIME을 HH:MM 형식으로 변환 (그래프용)
    times = []
    times_full = []  # 전체 날짜시간 (툴팁용)
    for t in window['CURRTIME'].values:
        t_str = str(t)
        if len(t_str) >= 12:
            times.append(f"{t_str[8:10]}:{t_str[10:12]}")
            times_full.append(f"{t_str[0:4]}-{t_str[4:6]}-{t_str[6:8]} {t_str[8:10]}:{t_str[10:12]}")
        else:
            times.append(t_str)
            times_full.append(t_str)
    
    # 전체 날짜시간 (알람/표시용)
    last_t = str(window['CURRTIME'].iloc[-1])
    if len(last_t) >= 12:
        full_time = f"{last_t[0:4]}-{last_t[4:6]}-{last_t[6:8]} {last_t[8:10]}:{last_t[10:12]}"
    else:
        full_time = last_t
    
    # 예측값 (샘플 데이터에서 가져오기)
    current_val = int(window['TOTALCNT'].iloc[-1])
    
    if 'PREDICT_10' in window.columns:
        predict_10 = int(window['PREDICT_10'].iloc[-1])
        predict_10_list = window['PREDICT_10'].tolist()
    else:
        predict_10 = current_val
        predict_10_list = window['TOTALCNT'].tolist()
    
    if 'PREDICT_30' in window.columns:
        predict_30 = int(window['PREDICT_30'].iloc[-1])
        predict_30_list = window['PREDICT_30'].tolist()
    else:
        predict_30 = current_val
        predict_30_list = window['TOTALCNT'].tolist()
    
    return jsonify({
        'x': times,
        'x_full': times_full,
        'y': window['TOTALCNT'].tolist(),
        'predict_10_list': predict_10_list,
        'predict_30_list': predict_30_list,
        'current': current_val,
        'predict_10': predict_10,
        'predict_30': predict_30,
        'currtime': full_time,
        'idx': current_idx,
        'total': len(df)
    })


@app.route('/api/next')
def next_step():
    global current_idx, df
    
    if df is None:
        return jsonify({'error': 'No data'}), 500
    
    if current_idx < len(df) - 1:
        current_idx += 1
    else:
        current_idx = 30
    
    return jsonify({'idx': current_idx})


@app.route('/api/reset')
def reset():
    global current_idx
    current_idx = 30
    return jsonify({'idx': current_idx})


if __name__ == '__main__':
    load_csv()
    print('=' * 40)
    print('M14 모니터링 서버')
    print('http://localhost:5000')
    print('=' * 40)
    app.run(debug=False, port=5000)