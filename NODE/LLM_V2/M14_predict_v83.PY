#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M14 물류 예측 모듈 - V8.3 (queue_gap 복원!)
"""

import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime, timedelta

# 모델 파일 (V8.3)
MODEL_FILES = {
    10: 'M14_MODEL/model_v83_10min.pkl',
    15: 'M14_MODEL/model_v83_15min.pkl',
    25: 'M14_MODEL/model_v83_25min.pkl'
}

# 필수 컬럼 8개 (V8.3)
REQUIRED_COLS = [
    'M14AM14B', 'M14AM14BSUM', 'M10AM14A', 'TOTALCNT',
    'M14.QUE.ALL.TRANSPORT4MINOVERCNT', 'M14.QUE.OHT.OHTUTIL',
    'M14.QUE.ALL.CURRENTQCREATED', 'M14.QUE.ALL.CURRENTQCOMPLETED'
]

def create_features_v83(row_dict):
    """V8.3 Feature 생성 (82개)"""
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

def adjust_prediction_v83(pred, current_total, m14b, m14bsum, gap, trans, horizon_min):
    """V8.3 예측 보정 (시간대별 하한선 포함)"""
    
    # 1. 시간대별 하한선
    floor_offsets = {10: 80, 15: 100, 25: 130}
    floor_offset = floor_offsets.get(horizon_min, 80)
    floor = current_total - floor_offset
    if pred < floor:
        pred = floor
    
    # 2. 1650~1699 구간 boost
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
    
    # 3. 위험 구간 보정
    danger_offsets = {10: 50, 15: 70, 25: 100}
    danger_offset = danger_offsets.get(horizon_min, 50)
    if current_total >= 1700 and pred < 1680:
        pred = max(pred, current_total - danger_offset)
    
    return min(pred, 2000)

def get_status_info(value):
    """상태 판정"""
    if value < 900:
        return 'LOW'
    elif value < 1600:
        return 'NORMAL'
    elif value < 1700:
        return 'CAUTION'
    else:
        return 'CRITICAL'

def generate_dashboard_html(result):
    """상세 HTML 대시보드 생성"""
    
    import json
    
    current_value = result['current_value']
    current_status = result['current_status']
    current_time = result['current_time']
    predictions = result['predictions']
    
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
    chart1_labels = result['time_labels'] + [p['pred_time_label'] if 'pred_time_label' in p else f"{p['horizon']}min" for p in predictions]
    chart1_historical = result['totalcnt_data'] + [None] * len(predictions)
    chart1_predictions = [None] * len(result['totalcnt_data']) + [p['prediction'] for p in predictions]
    
    chart2_labels = result['time_labels']
    chart2_data = result['m14b_data']
    chart3_data = result['m14bsum_data']
    chart4_data = result['gap_data']
    chart5_data = result['trans_data']
    
    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>M14 V8.3 예측 대시보드</title>
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
.pred{{background:#f7fafc;border-radius:10px;padding:25px;text-align:center}}
.time{{font-size:20px;font-weight:700;color:#667eea;margin-bottom:10px}}
.pred-value{{font-size:48px;font-weight:700;color:#2d3748;margin:15px 0}}
.danger{{font-size:28px;font-weight:700;margin:10px 0}}
.danger-high{{color:#e53e3e}}
.danger-medium{{color:#dd6b20}}
.danger-low{{color:#38a169}}
.metric{{display:flex;justify-content:space-between;padding:10px;background:#fff;border-radius:8px;margin:8px 0;font-size:16px}}
.chart-container{{background:#fff;border-radius:15px;padding:30px;box-shadow:0 10px 30px rgba(0,0,0,0.2);margin-bottom:20px}}
.chart-title{{font-size:24px;font-weight:700;color:#2d3748;margin-bottom:20px;text-align:center}}
.chart-grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:20px}}
</style></head><body>

<div class="container">

<div class="card">
<div class="header">📊 M14 V8.3 예측 대시보드</div>
<div class="subtitle">{current_time} | 82 Features | queue_gap 복원!</div>

<div class="current">
<div class="current-grid">
<div class="current-item">
<div class="current-label">TOTALCNT</div>
<div class="current-value">{current_value:,}</div>
<span class="current-status status-{current_status.lower()}">{current_status}</span>
</div>
<div class="current-item">
<div class="current-label">M14AM14B</div>
<div class="current-value">{result['current_m14b']:.0f}</div>
</div>
<div class="current-item">
<div class="current-label">M14AM14BSUM</div>
<div class="current-value">{result['current_m14bsum']:.0f}</div>
</div>
<div class="current-item">
<div class="current-label">queue_gap</div>
<div class="current-value">{result['current_gap']:.0f}</div>
</div>
<div class="current-item">
<div class="current-label">TRANSPORT</div>
<div class="current-value">{result['current_trans']:.0f}</div>
</div>
</div>
</div>
</div>

<div class="risk">위험도: {risk} ({max_danger}%)</div>

<div class="grid">"""
    
    for p in predictions:
        danger_class = 'danger-high' if p['danger_probability'] >= 60 else ('danger-medium' if p['danger_probability'] >= 30 else 'danger-low')
        
        html += f"""<div class="pred">
<div class="time">⏱️ {p['horizon']}분 후</div>
<div class="pred-value">{p['prediction']:,}</div>
<div class="danger {danger_class}">🚨 {p['danger_probability']}%</div>
<div class="metric"><span>변화</span><span style="font-weight:700">{p['change']:+,}</span></div>
<div class="metric"><span>상태</span><span style="font-weight:700">{p['status']}</span></div>
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
            tooltip: {{backgroundColor: 'rgba(0, 0, 0, 0.8)', padding: 12}}
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
    data: {{labels: {json.dumps(chart2_labels)}, datasets: [{{label: 'M14AM14B', data: {json.dumps(chart2_data)}, borderColor: '#48bb78', backgroundColor: 'rgba(72, 187, 120, 0.1)', borderWidth: 2, pointRadius: 0, tension: 0.4, fill: true}}]}},
    options: {{responsive: true, maintainAspectRatio: true, plugins: {{legend: {{display: false}}}}, scales: {{x: {{grid: {{display: false}}, ticks: {{maxRotation: 0, autoSkip: true, maxTicksLimit: 10, font: {{size: 9}}}}}}, y: {{grid: {{display: true, color: 'rgba(0, 0, 0, 0.05)'}}, ticks: {{font: {{size: 10}}}}}}}}}}
}});

// Chart 3: M14AM14BSUM
const ctx3 = document.getElementById('chart3').getContext('2d');
new Chart(ctx3, {{
    type: 'line',
    data: {{labels: {json.dumps(chart2_labels)}, datasets: [{{label: 'M14AM14BSUM', data: {json.dumps(chart3_data)}, borderColor: '#4299e1', backgroundColor: 'rgba(66, 153, 225, 0.1)', borderWidth: 2, pointRadius: 0, tension: 0.4, fill: true}}]}},
    options: {{responsive: true, maintainAspectRatio: true, plugins: {{legend: {{display: false}}}}, scales: {{x: {{grid: {{display: false}}, ticks: {{maxRotation: 0, autoSkip: true, maxTicksLimit: 10, font: {{size: 9}}}}}}, y: {{grid: {{display: true, color: 'rgba(0, 0, 0, 0.05)'}}, ticks: {{font: {{size: 10}}}}}}}}}}
}});

// Chart 4: queue_gap
const ctx4 = document.getElementById('chart4').getContext('2d');
new Chart(ctx4, {{
    type: 'line',
    data: {{labels: {json.dumps(chart2_labels)}, datasets: [{{label: 'queue_gap', data: {json.dumps(chart4_data)}, borderColor: '#ed8936', backgroundColor: 'rgba(237, 137, 54, 0.1)', borderWidth: 2, pointRadius: 0, tension: 0.4, fill: true}}]}},
    options: {{responsive: true, maintainAspectRatio: true, plugins: {{legend: {{display: false}}}}, scales: {{x: {{grid: {{display: false}}, ticks: {{maxRotation: 0, autoSkip: true, maxTicksLimit: 10, font: {{size: 9}}}}}}, y: {{grid: {{display: true, color: 'rgba(0, 0, 0, 0.05)'}}, ticks: {{font: {{size: 10}}}}}}}}}}
}});

// Chart 5: TRANSPORT
const ctx5 = document.getElementById('chart5').getContext('2d');
new Chart(ctx5, {{
    type: 'line',
    data: {{labels: {json.dumps(chart2_labels)}, datasets: [{{label: 'TRANSPORT', data: {json.dumps(chart5_data)}, borderColor: '#9f7aea', backgroundColor: 'rgba(159, 122, 234, 0.1)', borderWidth: 2, pointRadius: 0, tension: 0.4, fill: true}}]}},
    options: {{responsive: true, maintainAspectRatio: true, plugins: {{legend: {{display: false}}}}, scales: {{x: {{grid: {{display: false}}, ticks: {{maxRotation: 0, autoSkip: true, maxTicksLimit: 10, font: {{size: 9}}}}}}, y: {{grid: {{display: true, color: 'rgba(0, 0, 0, 0.05)'}}, ticks: {{font: {{size: 10}}}}}}}}}}
}});
</script>

</body></html>"""
    
    return html

def predict_m14(csv_data):
    """
    M14 예측 실행 (V8.3)
    csv_data: CSV 문자열 또는 DataFrame
    """
    
    # DataFrame 변환
    if isinstance(csv_data, str):
        from io import StringIO
        df = pd.read_csv(StringIO(csv_data))
    else:
        df = csv_data
    
    # 필수 컬럼 확인
    missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing_cols:
        return {'error': 'Missing columns', 'message': f"Missing: {', '.join(missing_cols)}"}
    
    # 데이터 부족 확인
    if len(df) < 280:
        return {'error': 'Insufficient data', 'message': f'Need 280 rows, got {len(df)}'}
    
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
    
    if len(df) < 280:
        return {'error': 'Insufficient data', 'message': f'Need 280 rows after parsing, got {len(df)}'}
    
    # 280분 데이터 추출 (V8.3 형식)
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
    
    # Feature 생성 (V8.3)
    try:
        features = create_features_v83(row_dict)
        X_pred = pd.DataFrame([features])
    except Exception as e:
        return {'error': 'Feature generation failed', 'message': str(e)}
    
    # 그래프용 데이터 (최근 60분!)
    totalcnt_data = seq_totalcnt[-60:].tolist()
    m14b_data = seq_m14b[-60:].tolist()
    m14bsum_data = seq_m14b_sum[-60:].tolist()
    gap_data = seq_gap[-60:].tolist()
    trans_data = seq_trans[-60:].tolist()
    
    # 시간 라벨 (60개)
    time_labels = []
    for i in range(60):
        t = current_time - timedelta(minutes=59-i)
        time_labels.append(t.strftime('%H:%M'))
    
    # 예측 실행
    results = []
    
    for horizon_min in [10, 15, 25]:
        model_file = MODEL_FILES[horizon_min]
        
        if not os.path.exists(model_file):
            continue
        
        try:
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            
            pred_raw = model.predict(X_pred)[0]
            pred = adjust_prediction_v83(pred_raw, current_totalcnt, current_m14b, current_m14bsum, current_gap, current_trans, horizon_min)
            pred_status = get_status_info(pred)
            
            # 위험 확률
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
            
            # 황금 패턴 보정 (V8.3 임계값)
            if (current_m14b > 540 and current_m14bsum > 620):
                danger_prob = min(100, danger_prob + 20)
            elif (current_m14b > 520 and current_m14bsum > 600):
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
            
            if current_totalcnt >= 1700:
                danger_prob = max(danger_prob, 85)
            elif current_totalcnt >= 1650:
                danger_prob = max(danger_prob, 60)
            
            danger_prob = max(0, min(100, danger_prob))
            
            results.append({
                'horizon': horizon_min,
                'prediction': int(pred),
                'status': pred_status,
                'danger_probability': danger_prob,
                'change': int(pred - current_totalcnt)
            })
            
        except Exception as e:
            continue
    
    if not results:
        return {'error': 'Prediction failed', 'message': 'All models failed'}
    
    result = {
        'success': True,
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
    
    # HTML 대시보드 생성
    result['dashboard_html'] = generate_dashboard_html(result)
    
    return result