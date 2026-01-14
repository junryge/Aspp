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

# 데이터 매니저 (280분 윈도우, data 폴더에 저장)
data_manager = m14_data.M14DataManager(window_minutes=280, data_dir='data')

# 예측 모듈 연결
data_manager.set_predictors(predictor_10min, predictor_30min)


@app.route('/')
def index():
    return send_file('index.html')


@app.route('/mini')
def mini():
    return send_file('mini.html')


@app.route('/api/data')
def get_data():
    """
    실시간 데이터 + 예측값 API
    - 데이터 매니저에서 저장된 데이터/예측값 반환
    """
    
    # 데이터가 없으면 초기화
    if data_manager.data is None or len(data_manager.data) == 0:
        if not data_manager.initialize():
            return jsonify({'error': 'Data load failed'}), 500
    
    df = data_manager.get_data()
    predict_10_all, predict_30_all = data_manager.get_predictions()
    
    if df is None or len(df) == 0:
        return jsonify({'error': 'No data'}), 500
    
    # 차트용 데이터 (최근 60분만)
    chart_len = min(60, len(df))
    df_chart = df.tail(chart_len).reset_index(drop=True)
    predict_10_list = predict_10_all[-chart_len:]
    predict_30_list = predict_30_all[-chart_len:]
    
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
    
    # 현재 시간 포맷
    last_t = str(df['CURRTIME'].iloc[-1])
    if len(last_t) >= 12:
        full_time = f"{last_t[0:4]}-{last_t[4:6]}-{last_t[6:8]} {last_t[8:10]}:{last_t[10:12]}"
    else:
        full_time = last_t
    
    current_val = int(df['TOTALCNT'].iloc[-1]) if pd.notna(df['TOTALCNT'].iloc[-1]) else 0
    
    # 알람 기록 + 상태
    alert_10, alert_30 = data_manager.get_alerts()
    alarm_state = data_manager.get_alarm_state()
    
    return jsonify({
        'x': times,
        'x_full': times_full,
        'y': df_chart['TOTALCNT'].fillna(0).astype(int).tolist(),
        'predict_10_list': predict_10_list,
        'predict_30_list': predict_30_list,
        'current': current_val,
        'predict_10': predict_10_list[-1] if predict_10_list else 0,
        'predict_30': predict_30_list[-1] if predict_30_list else 0,
        'currtime': full_time,
        'idx': len(df),
        'total': len(df),
        'alert_10': alert_10,
        'alert_30': alert_30,
        'alarm_state': alarm_state
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


@app.route('/history')
def history():
    """과거 데이터 조회 페이지"""
    return send_file('history.html')


@app.route('/api/history')
def get_history():
    """
    과거 데이터 조회 API
    
    Parameters:
        date: YYYYMMDD 형식의 날짜
    
    Returns:
        data: 해당 날짜의 데이터 + 예측값
        alerts_10: 10분 예측 알람 기록
        alerts_30: 30분 예측 알람 기록
    """
    from flask import request
    import os
    
    date_str = request.args.get('date', '')
    
    if not date_str or len(date_str) != 8:
        return jsonify({'error': '날짜 형식이 잘못되었습니다 (YYYYMMDD)'}), 400
    
    data_dir = data_manager.data_dir
    data_file = os.path.join(data_dir, f'm14_data_{date_str}.csv')
    pred_file = os.path.join(data_dir, f'm14_pred_{date_str}.csv')
    alert_file = os.path.join(data_dir, f'm14_alert_{date_str}.csv')
    
    # 데이터 파일 확인
    if not os.path.exists(data_file):
        return jsonify({'error': f'{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} 데이터가 없습니다'})
    
    try:
        # 데이터 로드
        df_data = pd.read_csv(data_file)
        
        # 예측 파일 로드
        if os.path.exists(pred_file):
            df_pred = pd.read_csv(pred_file)
            # 데이터와 예측 merge
            if 'CURRTIME' in df_pred.columns:
                df_merged = pd.merge(df_data, df_pred, on='CURRTIME', how='left')
            else:
                # 기존 형식 (CURRTIME 없는 경우)
                for col in ['PREDICT_10', 'PREDICT_30', 'PRED_TIME_10', 'PRED_TIME_30']:
                    if col in df_pred.columns:
                        df_data[col] = df_pred[col].values[:len(df_data)]
                df_merged = df_data
        else:
            df_merged = df_data
        
        # NaN 처리
        df_merged = df_merged.fillna(0)
        
        # 알람 기록 로드
        alerts_10 = []
        alerts_30 = []
        
        if os.path.exists(alert_file):
            df_alert = pd.read_csv(alert_file)
            for _, row in df_alert.iterrows():
                alert_item = {
                    'CURRTIME': row.get('CURRTIME', ''),
                    'VALUE': int(row.get('VALUE', 0)),
                    'ALARM_NO': int(row.get('ALARM_NO', 0)),
                    'IS_ALARM': bool(row.get('IS_ALARM', False)),
                    'COOLDOWN_MINS': int(row.get('COOLDOWN_MINS', 0)) if pd.notna(row.get('COOLDOWN_MINS')) else 0
                }
                if row.get('TYPE') == 'PRED_10':
                    alerts_10.append(alert_item)
                elif row.get('TYPE') == 'PRED_30':
                    alerts_30.append(alert_item)
        
        # 결과 반환
        return jsonify({
            'date': date_str,
            'data': df_merged.to_dict('records'),
            'alerts_10': alerts_10,
            'alerts_30': alerts_30
        })
        
    except Exception as e:
        return jsonify({'error': f'데이터 로드 실패: {str(e)}'}), 500


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