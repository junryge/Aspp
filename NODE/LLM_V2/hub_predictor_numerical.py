#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HUB 수치형 예측 모듈"""

import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime, timedelta
from io import StringIO
import json

# 스크립트 위치 기준 경로
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, 'HUB_MODEL')

TARGET_COL = 'CURRENT_M16A_3F_JOB_2'

FEATURE_COLS = {
    'storage': ['M16A_3F_STORAGE_UTIL'],
    'fs_storage': ['CD_M163FSTORAGEUSE', 'CD_M163FSTORAGETOTAL', 'CD_M163FSTORAGEUTIL'],
    'hub': ['HUBROOMTOTAL'],
    'cmd': ['M16A_3F_CMD', 'M16A_6F_TO_HUB_CMD'],
    'inflow': ['M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2', 'M14A_3F_TO_HUB_JOB2'],
    'outflow': ['M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB', 'M16A_3F_TO_M14A_3F_JOB'],
    'maxcapa': ['M16A_6F_LFT_MAXCAPA', 'M16A_2F_LFT_MAXCAPA'],
    'que': ['M16HUB.QUE.ALL.CURRENTQCNT', 'M16HUB.QUE.TIME.AVGTOTALTIME1MIN'],
    'target_alt': ['CURRENT_M16A_3F_JOB'],
    'm16hub_que': [
        'M16HUB.QUE.ALL.CURRENTQCOMPLETED', 'M16HUB.QUE.ALL.FABTRANSJOBCNT',
        'M16HUB.QUE.TIME.AVGTOTALTIME', 'M16HUB.QUE.OHT.CURRENTOHTQCNT', 'M16HUB.QUE.OHT.OHTUTIL'
    ],
    'm16a_que': [
        'M16A.QUE.ALL.CURRENTQCOMPLETED', 'M16A.QUE.ALL.CURRENTQCREATED',
        'M16A.QUE.OHT.CURRENTOHTQCNT', 'M16A.QUE.OHT.OHTUTIL',
        'M16A.QUE.LOAD.AVGLOADTIME1MIN', 'M16A.QUE.ALL.TRANSPORT4MINOVERCNT', 'M16A.QUE.ABN.QUETIMEDELAY'
    ]
}

def create_features_v8(df, available_cols):
    """Feature 생성"""
    if len(df) < 30:
        raise ValueError(f"데이터 부족: {len(df)}개 (최소 30개 필요)")
    
    i = len(df)
    seq_target = df[TARGET_COL].iloc[i-30:i].values
    
    features = {
        'target_mean': np.mean(seq_target),
        'target_std': np.std(seq_target),
        'target_max': np.max(seq_target),
        'target_min': np.min(seq_target),
        'target_last_value': seq_target[-1],
        'target_last_5_mean': np.mean(seq_target[-5:]),
        'target_slope': np.polyfit(np.arange(30), seq_target, 1)[0],
        'target_acceleration': (seq_target[-5:].mean() - seq_target[-10:-5].mean()) / 5,
        'target_is_rising': 1 if seq_target[-1] > seq_target[-5] else 0,
        'target_rapid_rise': 1 if (seq_target[-1] - seq_target[-5] > 10) else 0,
        'target_last_10_mean': np.mean(seq_target[-10:])
    }
    
    for group_name, cols in FEATURE_COLS.items():
        for col in cols:
            if col not in available_cols:
                continue
            col_seq = df[col].iloc[i-30:i].values
            
            if group_name == 'maxcapa':
                features[f'{col}_last_value'] = col_seq[-1]
            elif group_name in ['cmd', 'storage', 'fs_storage', 'hub', 'que', 'target_alt', 'm16hub_que', 'm16a_que']:
                features[f'{col}_mean'] = np.mean(col_seq)
                features[f'{col}_std'] = np.std(col_seq)
                features[f'{col}_max'] = np.max(col_seq)
                features[f'{col}_min'] = np.min(col_seq)
                features[f'{col}_last_value'] = col_seq[-1]
                features[f'{col}_last_5_mean'] = np.mean(col_seq[-5:])
                features[f'{col}_slope'] = np.polyfit(np.arange(30), col_seq, 1)[0]
            else:
                features[f'{col}_mean'] = np.mean(col_seq)
                features[f'{col}_last_value'] = col_seq[-1]
                features[f'{col}_slope'] = np.polyfit(np.arange(30), col_seq, 1)[0]
    
    return features, seq_target[-1]

def predict_hub_numerical(csv_data):
    """HUB 수치형 예측"""
    print(f"[수치형] 모델 디렉토리: {MODEL_DIR}")
    
    try:
        df = pd.read_csv(StringIO(csv_data))
        print(f"[수치형] CSV 로드: {len(df)}행")
    except Exception as e:
        return {'error': 'CSV parsing failed', 'message': str(e)}
    
    if len(df) < 30:
        return {'error': 'Insufficient data', 'message': f"최소 30행 필요 (현재: {len(df)}행)"}
    
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')
    df = df.dropna(subset=[TARGET_COL])
    
    available_cols = set(df.columns)
    features, current_value = create_features_v8(df, available_cols)
    X_pred = pd.DataFrame([features])
    
    if 'STAT_DT' in df.columns:
        try:
            df['STAT_DT'] = pd.to_datetime(df['STAT_DT'].astype(str), format='%Y%m%d%H%M')
            current_time = df['STAT_DT'].iloc[-1]
        except:
            current_time = datetime.now()
    else:
        current_time = datetime.now()
    
    predictions = []
    
    for horizon_min in [10, 15, 25]:
        model_path = os.path.join(MODEL_DIR, f'xgboost_Numerical_V8_{horizon_min}min.pkl')
        print(f"[수치형] 모델: {model_path} → 존재: {os.path.exists(model_path)}")
        
        if not os.path.exists(model_path):
            continue
        
        try:
            with open(model_path, 'rb') as f:
                model_dict = pickle.load(f)
            models = model_dict['models']
            
            pred_min = models[0].predict(X_pred)[0]
            pred_max = models[1].predict(X_pred)[0]
            
            pred_value_max = current_value + pred_max
            
            if pred_value_max >= 300:
                status = "CRITICAL"
            elif pred_value_max >= 280:
                status = "WARNING"
            elif pred_value_max >= 270:
                status = "CAUTION"
            else:
                status = "NORMAL"
            
            predictions.append({
                "horizon": horizon_min,
                "current_time": current_time.strftime('%Y-%m-%d %H:%M'),
                "pred_time": (current_time + timedelta(minutes=horizon_min)).strftime('%Y-%m-%d %H:%M'),
                "pred_time_label": (current_time + timedelta(minutes=horizon_min)).strftime('%H:%M'),
                "current_value": round(current_value, 2),
                "pred_min": round(current_value + pred_min, 2),
                "pred_max": round(current_value + pred_max, 2),
                "change_min": round(pred_min, 2),
                "change_max": round(pred_max, 2),
                "status": status
            })
            print(f"[수치형] {horizon_min}분 예측 성공")
        except Exception as e:
            print(f"[수치형] {horizon_min}분 실패: {e}")
    
    if not predictions:
        return {'error': 'No predictions', 'message': '예측 모델을 찾을 수 없습니다.'}
    
    historical_data = df[TARGET_COL].tail(30).values.tolist()
    if 'STAT_DT' in df.columns:
        historical_times = [t.strftime('%H:%M') for t in df['STAT_DT'].tail(30)]
    else:
        historical_times = [f"-{30-i}분" for i in range(30)]
    
    dashboard_html = generate_html(predictions, historical_data, historical_times)
    
    return {
        'predictions': predictions,
        'current_value': current_value,
        'dashboard_html': dashboard_html
    }

def generate_html(data, historical_data, historical_times):
    """HTML 대시보드"""
    if not data:
        return "<html><body><h1>No Data</h1></body></html>"
    
    current_value = data[0]['current_value']
    max_value = max(p['pred_max'] for p in data)
    
    if max_value >= 300:
        risk, risk_color = "CRITICAL", "#c53030"
    elif max_value >= 280:
        risk, risk_color = "WARNING", "#dd6b20"
    else:
        risk, risk_color = "NORMAL", "#38a169"
    
    chart_labels = historical_times + [p['pred_time_label'] for p in data]
    chart_historical = historical_data + [None] * len(data)
    chart_pred_min = [None] * len(historical_times) + [p['pred_min'] for p in data]
    chart_pred_max = [None] * len(historical_times) + [p['pred_max'] for p in data]
    
    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>HUB 수치형</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>*{{margin:0;padding:0}}body{{font-family:sans-serif;background:linear-gradient(135deg,#667eea,#764ba2);padding:20px}}.container{{max-width:1400px;margin:0 auto}}.card{{background:#fff;border-radius:15px;padding:30px;margin-bottom:20px;box-shadow:0 10px 30px rgba(0,0,0,0.2)}}.header{{text-align:center;font-size:36px;color:#2d3748}}.risk{{background:{risk_color};color:#fff;padding:20px;border-radius:10px;text-align:center;font-size:24px;font-weight:700;margin:20px 0}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:20px}}.pred{{background:#f7fafc;border-radius:10px;padding:20px;text-align:center}}.time{{font-size:20px;font-weight:700;color:#667eea;margin-bottom:10px}}.range{{font-size:36px;font-weight:700;color:#e53e3e;margin:15px 0}}.status{{display:inline-block;padding:8px 16px;border-radius:20px;font-weight:700;margin-top:10px}}.status-normal{{background:#c6f6d5;color:#2f855a}}.status-caution{{background:#fefcbf;color:#b7791f}}.status-warning{{background:#feebc8;color:#c05621}}.status-critical{{background:#fed7d7;color:#c53030}}</style></head><body><div class="container">
<div class="card"><div class="header">📊 HUB 수치형</div><div style="text-align:center;font-size:20px;color:#4299e1;margin-top:10px">Current: {current_value} → Max: {max_value:.1f}</div></div>
<div class="risk">{risk}</div>
<div class="grid">"""
    
    for p in data:
        html += f"""<div class="pred"><div class="time">⏱️ {p['horizon']}분</div><div style="font-size:14px;color:#718096">{p['pred_time']}</div>
<div class="range">{p['pred_min']:.1f} ~ {p['pred_max']:.1f}</div>
<span class="status status-{p['status'].lower()}">{p['status']}</span></div>"""
    
    html += f"""</div>
<div class="card"><canvas id="chart"></canvas></div>
</div>
<script>
new Chart(document.getElementById('chart'), {{
    type: 'line',
    data: {{
        labels: {json.dumps(chart_labels)},
        datasets: [
            {{label: '과거', data: {json.dumps(chart_historical)}, borderColor: '#667eea', borderWidth: 3}},
            {{label: '예측 MAX', data: {json.dumps(chart_pred_max)}, borderColor: '#e53e3e', borderDash: [5, 5]}},
            {{label: '예측 MIN', data: {json.dumps(chart_pred_min)}, borderColor: '#38a169', borderDash: [5, 5]}}
        ]
    }},
    options: {{responsive: true, aspectRatio: 2.5}}
}});
</script>
</body></html>"""
    
    return html