"""
================================================================================
M14 반송 큐 모니터링 서버
- Flask 웹 서버
- m14_data.py: 로그프레소에서 280분 데이터 조회
- predictor_10min.py: 10분 예측
- predictor_30min.py: 30분 예측
================================================================================
"""

from flask import Flask, jsonify, send_file
import pandas as pd
from datetime import datetime

# 모듈 import
import m14_data
import predictor_10min
import predictor_30min

app = Flask(__name__)

# 데이터 매니저 (280분 윈도우)
data_manager = m14_data.M14DataManager(window_minutes=280)


@app.route('/')
def index():
    return send_file('index.html')


@app.route('/api/data')
def get_data():
    """
    실시간 데이터 + 예측값 API
    - 280분 데이터 조회
    - 10분, 30분 예측 수행
    """
    
    # 데이터가 없으면 초기화
    if data_manager.data is None or len(data_manager.data) == 0:
        if not data_manager.initialize():
            return jsonify({'error': 'Data load failed'}), 500
    
    df = data_manager.get_data()
    
    if df is None or len(df) == 0:
        return jsonify({'error': 'No data'}), 500
    
    # 10분 예측
    pred_10 = predictor_10min.predict(df)
    
    # 30분 예측
    pred_30 = predictor_30min.predict(df)
    
    # 차트용 데이터 (최근 60분만)
    df_chart = df.tail(60).reset_index(drop=True)
    
    # 시간 포맷 변환
    times = []
    times_full = []
    for t in df_chart['CURRTIME'].values:
        t_str = str(t)
        if len(t_str) >= 12:
            times.append(f"{t_str[8:10]}:{t_str[10:12]}")
            times_full.append(f"{t_str[0:4]}-{t_str[4:6]}-{t_str[6:8]} {t_str[8:10]}:{t_str[10:12]}")
        else:
            times.append(t_str)
            times_full.append(t_str)
    
    # 예측 리스트 (차트용 - 60개)
    predict_10_list = []
    predict_30_list = []
    
    # 각 시점별 예측값 계산 (차트 표시용)
    for i in range(len(df_chart)):
        # 간단한 폴백: 현재값 + 트렌드
        if i < 5:
            predict_10_list.append(int(df_chart['TOTALCNT'].iloc[i]))
            predict_30_list.append(int(df_chart['TOTALCNT'].iloc[i]))
        else:
            recent = df_chart['TOTALCNT'].iloc[max(0, i-5):i+1].tolist()
            avg = sum(recent) / len(recent)
            trend = (recent[-1] - recent[0]) / len(recent) if len(recent) >= 2 else 0
            p10 = int(max(1000, min(2000, avg + trend * 10)))
            p30 = int(max(1000, min(2000, avg + trend * 30)))
            predict_10_list.append(p10)
            predict_30_list.append(p30)
    
    # 마지막 값은 실제 ML 예측값으로 대체
    if predict_10_list:
        predict_10_list[-1] = pred_10['predict_value']
    if predict_30_list:
        predict_30_list[-1] = pred_30['predict_value']
    
    # 현재 시간 포맷
    last_t = str(df['CURRTIME'].iloc[-1])
    if len(last_t) >= 12:
        full_time = f"{last_t[0:4]}-{last_t[4:6]}-{last_t[6:8]} {last_t[8:10]}:{last_t[10:12]}"
    else:
        full_time = last_t
    
    return jsonify({
        'x': times,
        'x_full': times_full,
        'y': df_chart['TOTALCNT'].fillna(0).astype(int).tolist(),
        'predict_10_list': predict_10_list,
        'predict_30_list': predict_30_list,
        'current': pred_10['current_value'],
        'predict_10': pred_10['predict_value'],
        'predict_30': pred_30['predict_value'],
        'danger_10': pred_10['danger'],
        'danger_30': pred_30['danger'],
        'prob_10': pred_10.get('prob', 0),
        'prob_30': pred_30.get('prob', 0),
        'currtime': full_time,
        'idx': len(df),
        'total': len(df)
    })


@app.route('/api/next')
def next_step():
    """다음 스텝 - 새 데이터 추가 후 조회"""
    data_manager.update()
    return get_data()


@app.route('/api/reset')
def reset():
    """리셋 - 전체 데이터 새로고침"""
    data_manager.refresh()
    return jsonify({'status': 'ok'})


@app.route('/api/status')
def status():
    """서버 상태"""
    return jsonify({
        'status': 'running',
        'data_count': len(data_manager.data) if data_manager.data is not None else 0,
        'last_update': str(data_manager.last_update) if data_manager.last_update else None,
        'sequence_length': 280,
    })


if __name__ == '__main__':
    print('=' * 60)
    print('M14 반송 큐 모니터링 서버')
    print('=' * 60)
    print('📦 모듈:')
    print('  - m14_data.py: 로그프레소 280분 데이터 조회')
    print('  - predictor_10min.py: V10_4 10분 예측')
    print('  - predictor_30min.py: V10_4 30분 예측')
    print('=' * 60)
    print('🌐 http://localhost:5000')
    print('=' * 60)
    
    # 초기 데이터 로드
    print('\n[초기화] 280분 데이터 로드 중...')
    data_manager.initialize()
    
    # 서버 시작
    app.run(debug=False, port=5000, host='0.0.0.0')