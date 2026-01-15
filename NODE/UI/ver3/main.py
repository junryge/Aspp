"""
================================================================================
M14 반송 큐 모니터링 서버
- Flask 웹 서버
- m14_data.py: 로그프레소에서 280분 데이터 조회
- predictor_10min.py: 10분 예측
- predictor_30min.py: 30분 예측
- evaluator.py: 예측 평가 (내부/외부 데이터 소스 지원)
- logpresso_alarm.py: 로그프레소 알람 조회
================================================================================
"""

from flask import Flask, jsonify, send_file, request
import pandas as pd
from datetime import datetime

# 모듈 import
import m14_data
import predictor_10min
import predictor_30min
import evaluator
import logpresso_alarm

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
        # 예측 파일 먼저 확인 (TOTALCNT 포함된 경우)
        if os.path.exists(pred_file):
            df_pred = pd.read_csv(pred_file)
            
            # pred 파일에 TOTALCNT 있으면 바로 사용
            if 'TOTALCNT' in df_pred.columns and 'CURRTIME' in df_pred.columns:
                df_merged = df_pred
            else:
                # 없으면 data 파일과 merge
                df_data = pd.read_csv(data_file)
                if 'CURRTIME' in df_pred.columns:
                    df_merged = pd.merge(df_data, df_pred, on='CURRTIME', how='left')
                else:
                    for col in ['PREDICT_10', 'PREDICT_30', 'PRED_TIME_10', 'PRED_TIME_30']:
                        if col in df_pred.columns:
                            df_data[col] = df_pred[col].values[:len(df_data)]
                    df_merged = df_data
        else:
            # pred 파일 없으면 data만
            df_merged = pd.read_csv(data_file)
        
        # NaN 처리
        df_merged = df_merged.fillna(0)
        
        # 알람 기록 로드
        alerts_10 = []
        alerts_30 = []
        
        if os.path.exists(alert_file):
            df_alert = pd.read_csv(alert_file)
            # TYPE을 문자열로 변환 (CSV에서 정수로 읽힐 수 있음)
            df_alert['TYPE'] = df_alert['TYPE'].astype(str)
            for _, row in df_alert.iterrows():
                alert_item = {
                    'CURRTIME': row.get('CURRTIME', ''),
                    'VALUE': int(row.get('VALUE', 0)),
                    'ALARM_NO': int(row.get('ALARM_NO', 0)),
                    'IS_ALARM': bool(row.get('IS_ALARM', False)),
                    'COOLDOWN_MINS': int(row.get('COOLDOWN_MINS', 0)) if pd.notna(row.get('COOLDOWN_MINS')) else 0
                }
                # TYPE이 '10' 또는 '30'으로 저장됨
                if row.get('TYPE') == '10':
                    alerts_10.append(alert_item)
                elif row.get('TYPE') == '30':
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


# ============================================================================
# 로그프레소 알람 API
# ============================================================================

@app.route('/api/logpresso_alarm')
def get_logpresso_alarm():
    """
    로그프레소 알람 조회 API
    
    Parameters:
        from: 시작 시간 (YYYYMMDDHHMM00)
        to: 종료 시간 (YYYYMMDDHHMM00)
    
    Returns:
        data: 알람 리스트 [{MEAS_TM, LSTM_FCAST_TM, ALARM_DESC, ALARM_YN}, ...]
    """
    from_time = request.args.get('from', '')
    to_time = request.args.get('to', '')
    
    if not from_time or not to_time:
        return jsonify({'error': 'from, to 파라미터 필요'}), 400
    
    try:
        alarms = logpresso_alarm.get_alarm_data(from_time, to_time)
        return jsonify({
            'from': from_time,
            'to': to_time,
            'count': len(alarms),
            'data': alarms
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# 평가 관련 라우트 (백그라운드 실행)
# ============================================================================

@app.route('/evaluate')
def evaluate_page():
    """예측 평가 페이지"""
    return send_file('evaluate.html')


@app.route('/api/evaluate/start', methods=['POST', 'GET'])
def start_evaluate():
    """
    백그라운드 평가 시작 API
    
    Parameters:
        date_start: 시작 날짜 (YYYYMMDD)
        date_end: 종료 날짜 (YYYYMMDD)
        time_start: 시작 시간 (HHMM)
        time_end: 종료 시간 (HHMM)
        pred_type: '10' 또는 '30'
        data_source: 'internal' (파일) 또는 'external' (로그프레소)
    """
    date_start = request.args.get('date_start', '')
    date_end = request.args.get('date_end', '')
    time_start = request.args.get('time_start', '0000')
    time_end = request.args.get('time_end', '2359')
    pred_type = request.args.get('pred_type', '10')
    data_source = request.args.get('data_source', 'internal')  # 기본값: 내부(파일)
    
    if not date_start or not date_end:
        return jsonify({'error': '시작/종료 날짜를 지정해주세요'}), 400
    
    if len(date_start) != 8 or len(date_end) != 8:
        return jsonify({'error': '날짜 형식이 잘못되었습니다 (YYYYMMDD)'}), 400
    
    if pred_type not in ['10', '30']:
        return jsonify({'error': 'pred_type은 10 또는 30이어야 합니다'}), 400
    
    if data_source not in ['internal', 'external']:
        return jsonify({'error': 'data_source는 internal 또는 external이어야 합니다'}), 400
    
    success, msg = evaluator.eval_manager.start(
        data_dir=data_manager.data_dir,
        date_start=date_start,
        date_end=date_end,
        time_start=time_start,
        time_end=time_end,
        pred_type=pred_type,
        data_source=data_source
    )
    
    if success:
        return jsonify({'status': 'started', 'message': msg, 'data_source': data_source})
    else:
        return jsonify({'error': msg}), 400


@app.route('/api/evaluate/status')
def get_evaluate_status():
    """평가 진행 상태 조회"""
    return jsonify(evaluator.eval_manager.get_status())


@app.route('/api/evaluate/result')
def get_evaluate_result():
    """평가 결과 조회"""
    result = evaluator.eval_manager.get_result()
    if result:
        return jsonify(result)
    else:
        return jsonify({'error': '결과가 없습니다 (평가 진행 중이거나 시작되지 않음)'}), 400


@app.route('/api/evaluate/reset')
def reset_evaluate():
    """평가 상태 초기화"""
    evaluator.eval_manager.reset()
    return jsonify({'status': 'reset'})


@app.route('/api/evaluate/dates')
def get_available_dates():
    """사용 가능한 날짜 목록 반환 (내부 파일용)"""
    dates = evaluator.get_available_dates(data_manager.data_dir)
    return jsonify({'dates': dates})


if __name__ == '__main__':
    print('=' * 60)
    print('M14 반송 큐 모니터링 서버')
    print('=' * 60)
    print('📦 모듈:')
    print('  - m14_data.py: 로그프레소 280분 데이터 조회')
    print('  - predictor_10min.py: V10_4 10분 예측')
    print('  - predictor_30min.py: V10_4 30분 예측')
    print('  - evaluator.py: 예측 평가 (내부/외부 지원)')
    print('  - logpresso_alarm.py: 로그프레소 알람 조회')
    print('=' * 60)
    print('🌐 http://localhost:5000')
    print('   /evaluate - 예측 평가 페이지')
    print('     📁 내부: data 폴더 CSV 파일 사용')
    print('     🌐 외부: 로그프레소 API 직접 조회')
    print('   /api/logpresso_alarm - 로그프레소 알람 조회')
    print('=' * 60)
    
    # 초기 데이터 로드
    print('\n[초기화] 280분 데이터 로드 중...')
    data_manager.initialize()
    
    # 서버 시작
    app.run(debug=False, port=5000)#, host='0.0.0.0')