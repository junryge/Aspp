#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 실시간 예측 대시보드 - V9.0 (queue_gap 복원!)
10분/15분/25분 후 예측 + 풍부한 그래프
"""

import numpy as np
import pandas as pd
import pickle
import os
import webbrowser
from datetime import datetime, timedelta

# ========================================
# 설정
# ========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, 'data', '2025_DATA.CSV')

# 모델 파일 (V9.0)
MODEL_FILES = {
    10: 'model_v9_10min.pkl',
    15: 'model_v9_15min.pkl',
    25: 'model_v9_25min.pkl'
}

# 필수 컬럼 8개 (V9.0)
REQUIRED_COLS = [
    'M14AM14B', 'M14AM14BSUM', 'M10AM14A', 'TOTALCNT',
    'M14.QUE.ALL.TRANSPORT4MINOVERCNT', 'M14.QUE.OHT.OHTUTIL',
    'M14.QUE.ALL.CURRENTQCREATED', 'M14.QUE.ALL.CURRENTQCOMPLETED'
]

# ========================================
# Feature 생성 함수 (V9.0 - 82개 Feature)
# ========================================
def create_features_v9(row_dict):
    """V9.0 Feature 생성 (82개)"""
    features = {}
    
    seq_m14b = np.array(row_dict['M14AM14B'])
    seq_m14bsum = np.array(row_dict['M14AM14BSUM'])
    seq_m10arev = np.array(row_dict['M10AM14A'])
    seq_totalcnt = np.array(row_dict['TOTALCNT'])
    seq_transport = np.array(row_dict['TRANSPORT'])
    seq_oht = np.array(row_dict['OHT'])
    seq_q_created = np.array(row_dict['Q_CREATED'])
    seq_q_completed = np.array(row_dict['Q_COMPLETED'])
    seq_gap = seq_q_created - seq_q_completed
    
    seq_len = len(seq_m14b)
    
    # M14AM14B (8)
    features['m14b_mean'] = np.mean(seq_m14b)
    features['m14b_max'] = np.max(seq_m14b)
    features['m14b_current'] = seq_m14b[-1]
    features['m14b_last10'] = np.mean(seq_m14b[-10:])
    features['m14b_last30'] = np.mean(seq_m14b[-30:])
    features['m14b_slope'] = np.polyfit(np.arange(seq_len), seq_m14b, 1)[0]
    features['m14b_std'] = np.std(seq_m14b)
    features['m14b_trend10'] = seq_m14b[-1] - seq_m14b[-10]
    
    # M14AM14BSUM (8)
    features['m14bsum_mean'] = np.mean(seq_m14bsum)
    features['m14bsum_max'] = np.max(seq_m14bsum)
    features['m14bsum_current'] = seq_m14bsum[-1]
    features['m14bsum_last10'] = np.mean(seq_m14bsum[-10:])
    features['m14bsum_last30'] = np.mean(seq_m14bsum[-30:])
    features['m14bsum_slope'] = np.polyfit(np.arange(seq_len), seq_m14bsum, 1)[0]
    features['m14bsum_std'] = np.std(seq_m14bsum)
    features['m14bsum_trend10'] = seq_m14bsum[-1] - seq_m14bsum[-10]
    
    # TOTALCNT (10)
    features['total_mean'] = np.mean(seq_totalcnt)
    features['total_max'] = np.max(seq_totalcnt)
    features['total_min'] = np.min(seq_totalcnt)
    features['total_current'] = seq_totalcnt[-1]
    features['total_last5'] = np.mean(seq_totalcnt[-5:])
    features['total_last10'] = np.mean(seq_totalcnt[-10:])
    features['total_last30'] = np.mean(seq_totalcnt[-30:])
    features['total_slope'] = np.polyfit(np.arange(seq_len), seq_totalcnt, 1)[0]
    features['total_std'] = np.std(seq_totalcnt)
    features['total_trend10'] = seq_totalcnt[-1] - seq_totalcnt[-10]
    
    # M10AM14A (5)
    features['m10arev_mean'] = np.mean(seq_m10arev)
    features['m10arev_max'] = np.max(seq_m10arev)
    features['m10arev_current'] = seq_m10arev[-1]
    features['m10arev_last10'] = np.mean(seq_m10arev[-10:])
    features['m10arev_slope'] = np.polyfit(np.arange(seq_len), seq_m10arev, 1)[0]
    
    # TRANSPORT (5)
    features['trans_mean'] = np.mean(seq_transport)
    features['trans_max'] = np.max(seq_transport)
    features['trans_current'] = seq_transport[-1]
    features['trans_last10'] = np.mean(seq_transport[-10:])
    features['trans_slope'] = np.polyfit(np.arange(seq_len), seq_transport, 1)[0]
    
    # OHT (4)
    features['oht_mean'] = np.mean(seq_oht)
    features['oht_max'] = np.max(seq_oht)
    features['oht_current'] = seq_oht[-1]
    features['oht_last10'] = np.mean(seq_oht[-10:])
    
    # Queue Gap (10) - 복원!
    features['gap_mean'] = np.mean(seq_gap)
    features['gap_max'] = np.max(seq_gap)
    features['gap_min'] = np.min(seq_gap)
    features['gap_current'] = seq_gap[-1]
    features['gap_last5'] = np.mean(seq_gap[-5:])
    features['gap_last10'] = np.mean(seq_gap[-10:])
    features['gap_last30'] = np.mean(seq_gap[-30:])
    features['gap_slope'] = np.polyfit(np.arange(seq_len), seq_gap, 1)[0]
    features['gap_std'] = np.std(seq_gap)
    features['gap_trend10'] = seq_gap[-1] - seq_gap[-10]
    
    # Interaction (12)
    features['m14b_x_sum'] = seq_m14b[-1] * seq_m14bsum[-1] / 1000
    features['m14b_x_sum_mean'] = np.mean(seq_m14b * seq_m14bsum) / 1000
    features['sum_per_m14b'] = seq_m14bsum[-1] / (seq_m14b[-1] + 1)
    features['m14b_plus_sum'] = seq_m14b[-1] + seq_m14bsum[-1]
    features['m10arev_x_m14b'] = seq_m10arev[-1] * seq_m14b[-1] / 100
    features['trans_x_m14b'] = seq_transport[-1] * seq_m14b[-1] / 100
    features['ratio_m14b_total'] = seq_m14b[-1] / (seq_totalcnt[-1] + 1)
    features['ratio_sum_total'] = seq_m14bsum[-1] / (seq_totalcnt[-1] + 1)
    features['gap_x_m14b'] = seq_gap[-1] * seq_m14b[-1] / 1000
    features['gap_x_total'] = seq_gap[-1] * seq_totalcnt[-1] / 1000
    features['gap_x_trans'] = seq_gap[-1] * seq_transport[-1] / 100
    features['ratio_gap_total'] = seq_gap[-1] / (seq_totalcnt[-1] + 1)
    
    # 임계값 (12)
    features['m14b_over_520'] = np.sum(seq_m14b > 520)
    features['m14b_over_540'] = np.sum(seq_m14b > 540)
    features['m14bsum_over_600'] = np.sum(seq_m14bsum > 600)
    features['m14bsum_over_620'] = np.sum(seq_m14bsum > 620)
    features['m10arev_over_55'] = np.sum(seq_m10arev > 55)
    features['total_over_1600'] = np.sum(seq_totalcnt >= 1600)
    features['total_over_1650'] = np.sum(seq_totalcnt >= 1650)
    features['total_over_1700'] = np.sum(seq_totalcnt >= 1700)
    features['gap_over_200'] = np.sum(seq_gap > 200)
    features['gap_over_250'] = np.sum(seq_gap > 250)
    features['gap_over_300'] = np.sum(seq_gap > 300)
    features['gap_over_350'] = np.sum(seq_gap > 350)
    
    # 황금 패턴 (8)
    features['gold_strict'] = 1 if (seq_m14b[-1] > 540 and seq_m14bsum[-1] > 620) else 0
    features['gold_normal'] = 1 if (seq_m14b[-1] > 520 and seq_m14bsum[-1] > 600) else 0
    features['in_danger'] = 1 if seq_totalcnt[-1] >= 1700 else 0
    features['near_danger'] = 1 if seq_totalcnt[-1] >= 1600 else 0
    features['danger_gap'] = 1 if seq_gap[-1] > 300 else 0
    features['danger_trans'] = 1 if seq_transport[-1] > 151 else 0
    features['triple_check'] = 1 if (seq_m14b[-1] > 520 and seq_m14bsum[-1] > 600 and seq_gap[-1] > 250) else 0
    features['quad_check'] = 1 if (seq_m14b[-1] > 520 and seq_m14bsum[-1] > 600 and seq_gap[-1] > 250 and seq_transport[-1] > 145) else 0
    
    return features

# ========================================
# Boost 함수 (V9.0 - 시간대별 하한선)
# ========================================
def adjust_prediction_v9(pred, current_total, m14b, m14bsum, gap, trans, horizon_min):
    """V9.0 예측 보정 (시간대별 하한선)"""
    
    # 시간대별 하한선
    floor_offsets = {10: 80, 15: 100, 25: 130}
    floor_offset = floor_offsets.get(horizon_min, 80)
    floor = current_total - floor_offset
    if pred < floor:
        pred = floor
    
    # 1650~1699 구간 boost
    if 1650 <= pred < 1700:
        boost = 0
        
        # 황금패턴
        if m14b > 540 and m14bsum > 620:
            boost += 50
        elif m14b > 520 and m14bsum > 600:
            boost += 40
        
        # Gap
        if gap > 350:
            boost += 40
        elif gap > 300:
            boost += 35
        elif gap > 250:
            boost += 25
        
        # Transport
        if trans > 180:
            boost += 30
        elif trans > 151:
            boost += 20
        
        pred = pred + boost
    
    # 위험시 보정
    danger_offsets = {10: 50, 15: 70, 25: 100}
    danger_offset = danger_offsets.get(horizon_min, 50)
    if current_total >= 1700 and pred < 1680:
        pred = max(pred, current_total - danger_offset)
    
    return min(pred, 2000)

# ========================================
# 상태 판정
# ========================================
def get_status_info(value):
    """물류량에 따른 상태"""
    if value < 900:
        return 'LOW'
    elif value < 1600:
        return 'NORMAL'
    elif value < 1700:
        return 'CAUTION'
    else:
        return 'CRITICAL'

# ========================================
# 예측 실행
# ========================================
def predict_all_horizons():
    """10/15/25분 예측 실행"""
    
    print("="*60)
    print("🚀 실시간 예측 대시보드 - V9.0 (queue_gap 복원!)")
    print("="*60)
    
    # 데이터 로드
    if not os.path.exists(DATA_PATH):
        return {'error': 'No data', 'message': f'CSV file not found: {DATA_PATH}'}
    
    try:
        df = pd.read_csv(DATA_PATH, on_bad_lines='skip', dtype={'CURRTIME': str})
    except Exception as e:
        return {'error': 'No data', 'message': f'Failed to load data: {e}'}
    
    if len(df) == 0:
        return {'error': 'No data', 'message': 'CSV file is empty'}
    
    # 필수 컬럼 확인
    missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing_cols:
        return {'error': 'No data', 'message': f'Missing columns: {", ".join(missing_cols)}'}
    
    # 데이터 부족 확인
    if len(df) < 280:
        return {'error': 'Lack of data', 'message': f'Need 280 rows, got {len(df)}'}
    
    print(f"✅ Data: {len(df):,} rows")
    
    # CURRTIME 파싱
    if 'CURRTIME' in df.columns:
        try:
            df['CURRTIME'] = df['CURRTIME'].astype(str).str.strip()
            df = df[df['CURRTIME'].str.len() == 12].copy()
            df['CURRTIME'] = pd.to_datetime(df['CURRTIME'], format='%Y%m%d%H%M', errors='coerce')
            df = df.dropna(subset=['CURRTIME']).copy()
        except:
            base_time = datetime.now()
            df['CURRTIME'] = [base_time - timedelta(minutes=len(df)-1-i) for i in range(len(df))]
    else:
        base_time = datetime.now()
        df['CURRTIME'] = [base_time - timedelta(minutes=len(df)-1-i) for i in range(len(df))]
    
    # 최종 데이터 부족 재확인
    if len(df) < 280:
        return {'error': 'Lack of data', 'message': f'Need 280 rows, got {len(df)} after parsing'}
    
    # 280분 데이터 추출 (V9.0 형식)
    row_dict = {
        'M14AM14B': df['M14AM14B'].iloc[-280:].values,
        'M14AM14BSUM': df['M14AM14BSUM'].iloc[-280:].values,
        'M10AM14A': df['M10AM14A'].iloc[-280:].values,
        'TOTALCNT': df['TOTALCNT'].iloc[-280:].values,
        'TRANSPORT': df['M14.QUE.ALL.TRANSPORT4MINOVERCNT'].iloc[-280:].values,
        'OHT': df['M14.QUE.OHT.OHTUTIL'].iloc[-280:].values,
        'Q_CREATED': df['M14.QUE.ALL.CURRENTQCREATED'].iloc[-280:].values,
        'Q_COMPLETED': df['M14.QUE.ALL.CURRENTQCOMPLETED'].iloc[-280:].values,
    }
    
    # 시간 정보
    current_time = df['CURRTIME'].iloc[-1]
    if pd.isna(current_time):
        current_time = datetime.now()
    
    # 현재 상태
    seq_totalcnt = row_dict['TOTALCNT']
    seq_m14b = row_dict['M14AM14B']
    seq_m14b_sum = row_dict['M14AM14BSUM']
    seq_gap = row_dict['Q_CREATED'] - row_dict['Q_COMPLETED']
    seq_trans = row_dict['TRANSPORT']
    
    current_totalcnt = seq_totalcnt[-1]
    current_m14b = seq_m14b[-1]
    current_m14bsum = seq_m14b_sum[-1]
    current_gap = seq_gap[-1]
    current_trans = seq_trans[-1]
    
    print(f"Current TOTALCNT: {current_totalcnt:.1f}")
    
    # Feature 생성 (V9.0)
    try:
        features = create_features_v9(row_dict)
        X_pred = pd.DataFrame([features])
        print(f"✅ Features: {len(features)}")
    except Exception as e:
        return {'error': 'Model operation failure', 'message': f'Feature generation failed: {e}'}
    
    # 🎨 그래프용 데이터 (최근 60분!)
    totalcnt_data = seq_totalcnt[-60:].tolist()
    m14b_data = seq_m14b[-60:].tolist()
    m14bsum_data = seq_m14b_sum[-60:].tolist()
    gap_data = seq_gap[-60:].tolist()
    trans_data = seq_trans[-60:].tolist()
    
    # 시간 라벨 (60개)
    time_labels = df['CURRTIME'].tail(60).tolist()
    time_labels = [t.strftime('%H:%M') for t in time_labels]
    
    # 각 예측 실행
    results = []
    
    for horizon_min in [10, 15, 25]:
        model_file = MODEL_FILES[horizon_min]
        
        if not os.path.exists(model_file):
            print(f"⚠️ {horizon_min}min: Model not found")
            continue
        
        try:
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            
            # 원본 예측
            pred_raw = model.predict(X_pred)[0]
            
            # V9.0 보정
            pred = adjust_prediction_v9(pred_raw, current_totalcnt, current_m14b, current_m14bsum, current_gap, current_trans, horizon_min)
            
            # 상태 판정
            pred_status = get_status_info(pred)
            
            # 위험 확률 계산
            danger_prob = 0
            if pred >= 1750:
                danger_prob = 100
            elif pred >= 1700:
                danger_prob = 95
            elif pred >= 1680:
                danger_prob = 75
            elif pred >= 1650:
                danger_prob = 50
            elif pred >= 1620:
                danger_prob = 30
            elif pred >= 1600:
                danger_prob = 15
            else:
                danger_prob = 5
            
            # 황금 패턴 보정 (V9.0 임계값)
            gold_strict = (current_m14b > 540 and current_m14bsum > 620)
            gold_normal = (current_m14b > 520 and current_m14bsum > 600)
            
            if gold_strict:
                danger_prob = min(100, danger_prob + 20)
            elif gold_normal:
                danger_prob = min(100, danger_prob + 15)
            
            # Gap/Trans 보정
            if current_gap > 300:
                danger_prob = min(100, danger_prob + 15)
            elif current_gap > 250:
                danger_prob = min(100, danger_prob + 10)
            if current_trans > 151:
                danger_prob = min(100, danger_prob + (10 if current_trans > 180 else 5))
            
            # triple_check 보정
            if current_m14b > 520 and current_m14bsum > 600 and current_gap > 250:
                danger_prob = min(100, danger_prob + 10)
            
            # 현재값 보정
            if current_totalcnt >= 1700:
                danger_prob = max(danger_prob, 85)
            elif current_totalcnt >= 1650:
                danger_prob = max(danger_prob, 60)
            
            danger_prob = max(0, min(100, danger_prob))
            
            result = {
                'horizon': horizon_min,
                'current_time': current_time.strftime('%Y-%m-%d %H:%M'),
                'pred_time': (current_time + timedelta(minutes=horizon_min)).strftime('%Y-%m-%d %H:%M'),
                'pred_time_label': (current_time + timedelta(minutes=horizon_min)).strftime('%H:%M'),
                'prediction': int(pred),
                'status': pred_status,
                'danger_probability': danger_prob,
                'change': int(pred - current_totalcnt)
            }
            
            results.append(result)
            print(f"✅ {horizon_min}min: {pred:.1f} ({pred_status}) - Danger: {danger_prob}%")
            
        except Exception as e:
            print(f"❌ {horizon_min}min: {e}")
            continue
    
    if not results:
        return {'error': 'Model operation failure', 'message': 'All predictions failed'}
    
    return {
        'current_value': int(current_totalcnt),
        'current_time': current_time.strftime('%Y-%m-%d %H:%M'),
        'current_status': get_status_info(current_totalcnt),
        'current_m14b': float(current_m14b),
        'current_m14bsum': float(current_m14bsum),
        'current_gap': float(current_gap),
        'current_trans': float(current_trans),
        'predictions': results,
        'totalcnt_data': totalcnt_data,
        'm14b_data': m14b_data,
        'm14bsum_data': m14bsum_data,
        'gap_data': gap_data,
        'trans_data': trans_data,
        'time_labels': time_labels
    }

# ========================================
# HTML 생성
# ========================================
def generate_html(data):
    """대시보드 HTML 생성 - 풍부한 그래프!"""
    
    if 'error' in data:
        return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Error</title></head>
<body style="font-family:sans-serif;padding:50px;text-align:center;">
<h1 style="color:#e53e3e;">❌ {data['error']}</h1>
<p style="font-size:18px;color:#718096;">{data['message']}</p>
</body></html>"""
    
    current_value = data['current_value']
    current_status = data['current_status']
    predictions = data['predictions']
    
    # 최대 위험도
    max_danger = max(p['danger_probability'] for p in predictions)
    
    if max_danger >= 85:
        risk, risk_color = "CRITICAL", "#c53030"
    elif max_danger >= 60:
        risk, risk_color = "WARNING", "#dd6b20"
    elif max_danger >= 30:
        risk, risk_color = "CAUTION", "#d69e2e"
    else:
        risk, risk_color = "NORMAL", "#38a169"
    
    # 그래프 데이터
    import json
    
    # 1. TOTALCNT: 60분 + 예측 3개
    chart1_labels = data['time_labels'] + [p['pred_time_label'] for p in predictions]
    chart1_historical = data['totalcnt_data'] + [None] * len(predictions)
    chart1_predictions = [None] * len(data['totalcnt_data']) + [p['prediction'] for p in predictions]
    
    # 2. M14AM14B: 60분
    chart2_labels = data['time_labels']
    chart2_data = data['m14b_data']
    
    # 3. M14AM14BSUM: 60분
    chart3_data = data['m14bsum_data']
    
    # 4. queue_gap: 60분
    chart4_data = data['gap_data']
    
    # 5. TRANSPORT: 60분
    chart5_data = data['trans_data']
    
    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>V9.0 실시간 예측 대시보드</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:sans-serif;background:linear-gradient(135deg,#667eea,#764ba2);padding:20px}}
.container{{max-width:1600px;margin:0 auto}}
.card{{background:#fff;border-radius:15px;padding:30px;margin-bottom:20px;box-shadow:0 10px 30px rgba(0,0,0,0.2)}}
.header{{text-align:center;font-size:36px;color:#2d3748;margin-bottom:10px;font-weight:700}}
.subtitle{{text-align:center;font-size:18px;color:#718096;margin-bottom:20px}}
.current{{background:#f7fafc;border-radius:10px;padding:20px;margin-bottom:20px}}
.current-grid{{display:grid;grid-template-columns:repeat(5,1fr);gap:15px}}
.current-item{{text-align:center}}
.current-label{{font-size:12px;color:#718096;margin-bottom:5px}}
.current-value{{font-size:28px;font-weight:700;color:#2d3748}}
.current-status{{display:inline-block;padding:8px 16px;border-radius:20px;font-weight:700;margin-top:5px;font-size:14px}}
.status-low{{background:#c6f6d5;color:#2f855a}}
.status-normal{{background:#bee3f8;color:#2c5282}}
.status-caution{{background:#fefcbf;color:#b7791f}}
.status-critical{{background:#fed7d7;color:#c53030}}
.risk{{background:{risk_color};color:#fff;padding:20px;border-radius:10px;text-align:center;font-size:24px;font-weight:700;margin-bottom:20px}}
.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:20px;margin-bottom:30px}}
.pred{{background:#f7fafc;border-radius:10px;padding:20px;text-align:center}}
.time{{font-size:20px;font-weight:700;color:#667eea;margin-bottom:5px}}
.pred-value{{font-size:42px;font-weight:700;color:#2d3748;margin:15px 0}}
.danger{{font-size:24px;font-weight:700;margin:10px 0}}
.danger-high{{color:#e53e3e}}
.danger-medium{{color:#dd6b20}}
.danger-low{{color:#38a169}}
.metric{{display:flex;justify-content:space-between;padding:8px;background:#fff;border-radius:8px;margin:5px 0;font-size:16px}}
.chart-container{{background:#fff;border-radius:15px;padding:30px;box-shadow:0 10px 30px rgba(0,0,0,0.2);margin-bottom:20px}}
.chart-title{{font-size:24px;font-weight:700;color:#2d3748;margin-bottom:20px;text-align:center}}
.chart-grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:20px;margin-bottom:20px}}
.help-btn{{position:fixed;top:20px;right:20px;background:#667eea;color:#fff;border:none;padding:15px 25px;border-radius:50px;font-size:18px;font-weight:700;cursor:pointer;box-shadow:0 4px 15px rgba(102,126,234,0.4);z-index:1000;transition:all 0.3s}}
.help-btn:hover{{background:#5568d3;transform:translateY(-2px);box-shadow:0 6px 20px rgba(102,126,234,0.6)}}
.modal{{display:none;position:fixed;z-index:2000;left:0;top:0;width:100%;height:100%;overflow:auto;background-color:rgba(0,0,0,0.7)}}
.modal-content{{background:#fff;margin:2% auto;padding:0;border-radius:15px;width:90%;max-width:1200px;max-height:90vh;overflow-y:auto;box-shadow:0 10px 50px rgba(0,0,0,0.5)}}
.modal-header{{background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;padding:25px 30px;border-radius:15px 15px 0 0;position:sticky;top:0;z-index:10}}
.modal-title{{font-size:28px;font-weight:700;margin:0}}
.close{{color:#fff;float:right;font-size:35px;font-weight:700;cursor:pointer;line-height:28px}}
.close:hover{{color:#ffd700}}
.modal-body{{padding:30px;line-height:1.8;color:#2d3748}}
.modal-body h2{{color:#667eea;font-size:22px;margin:30px 0 15px 0;border-bottom:3px solid #667eea;padding-bottom:10px}}
.modal-body h3{{color:#4a5568;font-size:18px;margin:20px 0 10px 0}}
.modal-body ul{{margin-left:20px}}
.modal-body li{{margin:8px 0}}
.section{{background:#f7fafc;padding:20px;border-radius:10px;margin:15px 0;border-left:5px solid #667eea}}
.warning{{background:#fff5f5;border-left:5px solid #e53e3e}}
.success{{background:#f0fff4;border-left:5px solid #38a169}}
.info{{background:#ebf8ff;border-left:5px solid #4299e1}}
</style></head><body>

<!-- 도움말 버튼 -->
<button class="help-btn" onclick="openModal()">❓ 지표 설명</button>

<!-- 모달 (팝업) -->
<div id="helpModal" class="modal">
<div class="modal-content">
<div class="modal-header">
<span class="close" onclick="closeModal()">&times;</span>
<h2 class="modal-title">📊 V9.0 핵심 지표 설명 - 1700+ 위험 감지</h2>
</div>
<div class="modal-body">

<div class="section success">
<h2>🎯 V9.0 핵심 요약</h2>
<p><strong>TOTALCNT가 1700을 넘으면 위험한 상황입니다.</strong></p>
<p>V9.0은 8개 컬럼, 82개 Feature로 간소화하면서 queue_gap을 복원했습니다.</p>
<p><strong>4개 지표를 실시간으로 모니터링</strong>하여 위험을 사전에 예측합니다!</p>
</div>

<h2>1️⃣ queue_gap (가장 중요! ⭐⭐⭐)</h2>
<div class="section">
<h3>■ 정의</h3>
<p><code>queue_gap = 생성된 작업 - 완료된 작업 = 현재 대기 중인 작업량</code></p>

<h3>■ 위험 신호 (V9.0)</h3>
<ul>
<li>gap < 200 → 정상 ✅</li>
<li>gap 200~250 → 주의 ⚠️</li>
<li>gap 250~300 → 위험 🔥</li>
<li>gap > 300 → 매우 위험! 🚨</li>
<li>gap > 350 → 폭발 직전! 💥</li>
</ul>
</div>

<h2>2️⃣ M14AM14B (두 번째로 중요! ⭐⭐)</h2>
<div class="section">
<h3>■ 위험 신호 (V9.0 조정)</h3>
<ul>
<li>M14AM14B < 520 → 정상 ✅</li>
<li>M14AM14B > 520 → 주의 ⚠️</li>
<li>M14AM14B > 540 → 위험! 🔥</li>
</ul>
</div>

<h2>3️⃣ M14AM14BSUM (세 번째로 중요! ⭐⭐)</h2>
<div class="section">
<h3>■ 위험 신호 (V9.0 조정)</h3>
<ul>
<li>M14AM14BSUM < 600 → 정상 ✅</li>
<li>M14AM14BSUM > 600 → 주의 ⚠️</li>
<li>M14AM14BSUM > 620 → 위험! 🔥</li>
</ul>
</div>

<h2>4️⃣ TRANSPORT</h2>
<div class="section">
<h3>■ 위험 신호</h3>
<ul>
<li>TRANSPORT < 145 → 정상 ✅</li>
<li>TRANSPORT > 145 → 주의 ⚠️</li>
<li>TRANSPORT > 151 → 위험 🔥</li>
<li>TRANSPORT > 180 → 매우 위험! 🚨</li>
</ul>
</div>

<h2>⭐ V9.0 황금 패턴</h2>
<div class="section warning">
<h3>■ 엄격 패턴 (1700+ 확률 높음)</h3>
<p style="font-size:20px;font-weight:700;color:#e53e3e;">M14AM14B > 540 AND M14AM14BSUM > 620</p>

<h3>■ 보통 패턴</h3>
<p style="font-size:18px;font-weight:700;color:#dd6b20;">M14AM14B > 520 AND M14AM14BSUM > 600</p>

<h3>■ 3중 체크 (복합 위험)</h3>
<p style="font-size:18px;font-weight:700;color:#c53030;">M14AM14B > 520 AND M14AM14BSUM > 600 AND queue_gap > 250</p>

<h3>■ 4중 체크 (최고 위험)</h3>
<p style="font-size:16px;font-weight:700;color:#c53030;">위 조건 + TRANSPORT > 145</p>
</div>

<div class="section success">
<h2>✅ V9.0 변경사항</h2>
<ul style="font-size:16px;line-height:2;">
<li>✅ 컬럼: 13개 → 8개 (간소화)</li>
<li>✅ Feature: 222개 → 82개 (최적화)</li>
<li>✅ queue_gap 복원 (가장 중요한 지표!)</li>
<li>✅ 황금패턴 임계값 재조정 (520/600 → 540/620)</li>
<li>✅ 시간대별 하한선 추가 (10분:-80, 15분:-100, 25분:-130)</li>
</ul>
</div>

</div>
</div>
</div>

<div class="container">

<div class="card">
<div class="header">📊 V9.0 실시간 예측 대시보드</div>
<div class="subtitle">{data['current_time']} | 82 Features | queue_gap 복원!</div>

<div class="current">
<div class="current-grid">
<div class="current-item">
<div class="current-label">TOTALCNT</div>
<div class="current-value">{current_value:,}</div>
<span class="current-status status-{current_status.lower()}">{current_status}</span>
</div>
<div class="current-item">
<div class="current-label">M14AM14B</div>
<div class="current-value">{data['current_m14b']:.0f}</div>
</div>
<div class="current-item">
<div class="current-label">M14AM14BSUM</div>
<div class="current-value">{data['current_m14bsum']:.0f}</div>
</div>
<div class="current-item">
<div class="current-label">queue_gap</div>
<div class="current-value">{data['current_gap']:.0f}</div>
</div>
<div class="current-item">
<div class="current-label">TRANSPORT</div>
<div class="current-value">{data['current_trans']:.0f}</div>
</div>
</div>
</div>
</div>

<div class="risk">Maximum Risk: {risk} ({max_danger}%)</div>

<div class="grid">"""
    
    for p in predictions:
        danger_class = 'danger-high' if p['danger_probability'] >= 60 else ('danger-medium' if p['danger_probability'] >= 30 else 'danger-low')
        
        html += f"""<div class="pred">
<div class="time">⏱️ {p['horizon']} Minutes</div>
<div style="font-size:14px;color:#718096;margin-bottom:10px">{p['pred_time']}</div>
<div class="pred-value">{p['prediction']:,}</div>
<div class="danger {danger_class}">🚨 {p['danger_probability']}%</div>
<div class="metric"><span>Change</span><span style="font-weight:700">{p['change']:+,}</span></div>
<div class="metric"><span>Status</span><span style="font-weight:700">{p['status']}</span></div>
<span class="current-status status-{p['status'].lower()}">{p['status']}</span>
</div>"""
    
    html += f"""</div>

<!-- 메인 그래프: 60분 과거 + 예측 -->
<div class="chart-container">
<div class="chart-title">📈 최근 60분 TOTALCNT + 예측 (10/15/25분)</div>
<canvas id="chart1" height="100"></canvas>
</div>

<!-- 서브 그래프 그리드 -->
<div class="chart-grid">
<div class="chart-container">
<div class="chart-title">M14AM14B (최근 60분)</div>
<canvas id="chart2" height="150"></canvas>
</div>

<div class="chart-container">
<div class="chart-title">M14AM14BSUM (최근 60분)</div>
<canvas id="chart3" height="150"></canvas>
</div>

<div class="chart-container">
<div class="chart-title">queue_gap (최근 60분)</div>
<canvas id="chart4" height="150"></canvas>
</div>

<div class="chart-container">
<div class="chart-title">TRANSPORT (최근 60분)</div>
<canvas id="chart5" height="150"></canvas>
</div>
</div>

</div>

<script>
// Chart 1: TOTALCNT + 예측
const ctx1 = document.getElementById('chart1').getContext('2d');
new Chart(ctx1, {{
    type: 'line',
    data: {{
        labels: {json.dumps(chart1_labels)},
        datasets: [
            {{
                label: '최근 60분 데이터',
                data: {json.dumps(chart1_historical)},
                borderColor: '#667eea',
                backgroundColor: 'rgba(102, 126, 234, 0.1)',
                borderWidth: 3,
                pointRadius: 3,
                pointBackgroundColor: '#667eea',
                tension: 0.4,
                fill: false
            }},
            {{
                label: '예측 (10/15/25분)',
                data: {json.dumps(chart1_predictions)},
                borderColor: '#e53e3e',
                backgroundColor: 'rgba(229, 62, 62, 0.1)',
                borderWidth: 3,
                borderDash: [5, 5],
                pointRadius: 8,
                pointBackgroundColor: '#e53e3e',
                tension: 0,
                fill: false,
                spanGaps: true
            }}
        ]
    }},
    options: {{
        responsive: true,
        maintainAspectRatio: true,
        interaction: {{mode: 'index', intersect: false}},
        plugins: {{
            legend: {{display: true, position: 'top', labels: {{font: {{size: 14, weight: 'bold'}}, usePointStyle: true, padding: 20}}}},
            tooltip: {{backgroundColor: 'rgba(0, 0, 0, 0.8)', padding: 12, titleFont: {{size: 14, weight: 'bold'}}, bodyFont: {{size: 13}}}}
        }},
        scales: {{
            x: {{grid: {{display: true, color: 'rgba(0, 0, 0, 0.05)'}}, ticks: {{font: {{size: 11}}, maxRotation: 45, minRotation: 45}}}},
            y: {{beginAtZero: false, grid: {{display: true, color: 'rgba(0, 0, 0, 0.1)'}}, ticks: {{font: {{size: 12}}}}}}
        }}
    }}
}});

// Chart 2: M14AM14B
const ctx2 = document.getElementById('chart2').getContext('2d');
new Chart(ctx2, {{
    type: 'line',
    data: {{
        labels: {json.dumps(chart2_labels)},
        datasets: [{{label: 'M14AM14B', data: {json.dumps(chart2_data)}, borderColor: '#48bb78', backgroundColor: 'rgba(72, 187, 120, 0.1)', borderWidth: 2, pointRadius: 0, tension: 0.4, fill: true}}]
    }},
    options: {{responsive: true, maintainAspectRatio: true, plugins: {{legend: {{display: false}}}}, scales: {{x: {{grid: {{display: false}}, ticks: {{maxRotation: 0, autoSkip: true, maxTicksLimit: 10, font: {{size: 9}}}}}}, y: {{grid: {{display: true, color: 'rgba(0, 0, 0, 0.05)'}}, ticks: {{font: {{size: 10}}}}}}}}}}
}});

// Chart 3: M14AM14BSUM
const ctx3 = document.getElementById('chart3').getContext('2d');
new Chart(ctx3, {{
    type: 'line',
    data: {{
        labels: {json.dumps(chart2_labels)},
        datasets: [{{label: 'M14AM14BSUM', data: {json.dumps(chart3_data)}, borderColor: '#4299e1', backgroundColor: 'rgba(66, 153, 225, 0.1)', borderWidth: 2, pointRadius: 0, tension: 0.4, fill: true}}]
    }},
    options: {{responsive: true, maintainAspectRatio: true, plugins: {{legend: {{display: false}}}}, scales: {{x: {{grid: {{display: false}}, ticks: {{maxRotation: 0, autoSkip: true, maxTicksLimit: 10, font: {{size: 9}}}}}}, y: {{grid: {{display: true, color: 'rgba(0, 0, 0, 0.05)'}}, ticks: {{font: {{size: 10}}}}}}}}}}
}});

// Chart 4: queue_gap
const ctx4 = document.getElementById('chart4').getContext('2d');
new Chart(ctx4, {{
    type: 'line',
    data: {{
        labels: {json.dumps(chart2_labels)},
        datasets: [{{label: 'queue_gap', data: {json.dumps(chart4_data)}, borderColor: '#ed8936', backgroundColor: 'rgba(237, 137, 54, 0.1)', borderWidth: 2, pointRadius: 0, tension: 0.4, fill: true}}]
    }},
    options: {{responsive: true, maintainAspectRatio: true, plugins: {{legend: {{display: false}}}}, scales: {{x: {{grid: {{display: false}}, ticks: {{maxRotation: 0, autoSkip: true, maxTicksLimit: 10, font: {{size: 9}}}}}}, y: {{grid: {{display: true, color: 'rgba(0, 0, 0, 0.05)'}}, ticks: {{font: {{size: 10}}}}}}}}}}
}});

// Chart 5: TRANSPORT
const ctx5 = document.getElementById('chart5').getContext('2d');
new Chart(ctx5, {{
    type: 'line',
    data: {{
        labels: {json.dumps(chart2_labels)},
        datasets: [{{label: 'TRANSPORT', data: {json.dumps(chart5_data)}, borderColor: '#9f7aea', backgroundColor: 'rgba(159, 122, 234, 0.1)', borderWidth: 2, pointRadius: 0, tension: 0.4, fill: true}}]
    }},
    options: {{responsive: true, maintainAspectRatio: true, plugins: {{legend: {{display: false}}}}, scales: {{x: {{grid: {{display: false}}, ticks: {{maxRotation: 0, autoSkip: true, maxTicksLimit: 10, font: {{size: 9}}}}}}, y: {{grid: {{display: true, color: 'rgba(0, 0, 0, 0.05)'}}, ticks: {{font: {{size: 10}}}}}}}}}}
}});
</script>

<script>
function openModal() {{document.getElementById('helpModal').style.display = 'block'; document.body.style.overflow = 'hidden';}}
function closeModal() {{document.getElementById('helpModal').style.display = 'none'; document.body.style.overflow = 'auto';}}
window.onclick = function(event) {{var modal = document.getElementById('helpModal'); if (event.target == modal) {{closeModal();}}}};
document.addEventListener('keydown', function(event) {{if (event.key === 'Escape') {{closeModal();}}}});
</script>
</body></html>"""
    
    return html

# ========================================
# 메인
# ========================================
def main():
    # 예측 실행
    data = predict_all_horizons()
    
    # HTML 생성
    html = generate_html(data)
    
    # 파일 저장
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'Dashboard_V9_{timestamp}.html'
    filepath = os.path.join(SCRIPT_DIR, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"\n✅ Dashboard saved: {filename}")
    
    # 브라우저 열기
    try:
        webbrowser.open('file://' + os.path.abspath(filepath))
        print("✅ Browser opened")
    except:
        print("⚠️ Please open the HTML file manually")

if __name__ == '__main__':
    main()