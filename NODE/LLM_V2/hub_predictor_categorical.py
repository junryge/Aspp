#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HUB 범주형 예측 모듈 (서버용)"""

import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime, timedelta
from io import StringIO
import json

TARGET_COL = 'CURRENT_M16A_3F_JOB_2'

REQUIRED_COLS = [
    'STAT_DT', 'CURRENT_M16A_3F_JOB', 'CURRENT_M16A_3F_JOB_2',
    'M14A_3F_CNV_MAXCAPA', 'M14A_3F_TO_HUB_CMD', 'M14A_3F_TO_HUB_JOB2', 'M14A_3F_TO_HUB_JOB_ALT',
    'M14B_7F_LFT_MAXCAPA', 'M14B_7F_TO_HUB_CMD', 'M14B_7F_TO_HUB_JOB2', 'M14B_7F_TO_HUB_JOB_ALT',
    'M14_TO_M16_OFS_CUR', 'M16A_2F_LFT_MAXCAPA', 'M16A_2F_TO_6F_JOB', 'M16A_2F_TO_HUB_CMD',
    'M16A_2F_TO_HUB_JOB2', 'M16A_2F_TO_HUB_JOB_ALT', 'M16A_3F_CMD', 'M16A_3F_CNV_MAXCAPA',
    'M16A_3F_LFT_MAXCAPA', 'M16A_3F_M14BLFT_MAXCAPA', 'M16A_3F_STORAGE_UTIL',
    'M16A_3F_TO_3F_MLUD_JOB', 'M16A_3F_TO_M14A_3F_JOB', 'M16A_3F_TO_M14A_CNV_AI_CMD',
    'M16A_3F_TO_M14B_7F_JOB', 'M16A_3F_TO_M14B_LFT_AI_CMD', 'M16A_3F_TO_M16A_2F_JOB',
    'M16A_3F_TO_M16A_3F_STB_CMD', 'M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_LFT_AI_CMD',
    'M16A_3F_TO_M16A_MLUD_AI_CMD', 'M16A_6F_LFT_MAXCAPA', 'M16A_6F_TO_2F_JOB',
    'M16A_6F_TO_HUB_CMD', 'M16A_6F_TO_HUB_JOB', 'M16A_6F_TO_HUB_JOB_ALT', 'M16B_10F_TO_HUB_JOB',
    'M16_TO_M14_OFS_CUR', 'HUBROOMTOTAL', 'CD_M163FSTORAGEUSE', 'CD_M163FSTORAGETOTAL',
    'CD_M163FSTORAGEUTIL', 'M16HUB.QUE.ALL.CURRENTQCNT', 'M16HUB.QUE.TIME.AVGTOTALTIME1MIN',
    'M16HUB.QUE.ALL.CURRENTQCOMPLETED', 'M16HUB.QUE.ALL.FABTRANSJOBCNT',
    'M16HUB.QUE.TIME.AVGTOTALTIME', 'M16HUB.QUE.OHT.CURRENTOHTQCNT', 'M16HUB.QUE.OHT.OHTUTIL',
    'M16A.QUE.ALL.CURRENTQCOMPLETED', 'M16A.QUE.ALL.CURRENTQCREATED',
    'M16A.QUE.OHT.CURRENTOHTQCNT', 'M16A.QUE.OHT.OHTUTIL', 'M16A.QUE.LOAD.AVGLOADTIME1MIN',
    'M16A.QUE.ALL.TRANSPORT4MINOVERCNT', 'M16A.QUE.ABN.QUETIMEDELAY'
]

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

CLASS_NAMES = ['Class 0 (하락/정체)', 'Class 1 (소폭증가)', 'Class 2 (급증)']

def create_features_v8(df, available_cols):
    """Feature 생성 - hub_predictor_numerical.py와 동일"""
    
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
    
    # 추가 Feature (동일)
    if all(col in available_cols for col in ['CD_M163FSTORAGEUSE', 'CD_M163FSTORAGETOTAL', 'CD_M163FSTORAGEUTIL']):
        storage_use = df['CD_M163FSTORAGEUSE'].iloc[i-30:i].values
        storage_total = df['CD_M163FSTORAGETOTAL'].iloc[i-30:i].values
        storage_util = df['CD_M163FSTORAGEUTIL'].iloc[i-30:i].values
        features['storage_use_rate'] = (storage_use[-1] - storage_use[0]) / 30
        features['storage_remaining'] = storage_total[-1] - storage_use[-1]
        features['storage_util_last'] = storage_util[-1]
        features['storage_util_high'] = 1 if storage_util[-1] >= 7 else 0
        features['storage_util_critical'] = 1 if storage_util[-1] >= 10 else 0
    
    if 'HUBROOMTOTAL' in available_cols:
        hub_seq = df['HUBROOMTOTAL'].iloc[i-30:i].values
        hub_last = hub_seq[-1]
        features['hub_critical'] = 1 if hub_last < 590 else 0
        features['hub_high'] = 1 if hub_last < 610 else 0
        features['hub_warning'] = 1 if hub_last < 620 else 0
        features['hub_decrease_rate'] = (hub_seq[0] - hub_last) / 30
        if 'CD_M163FSTORAGEUTIL' in available_cols:
            storage_util_last = df['CD_M163FSTORAGEUTIL'].iloc[i-1]
            features['hub_storage_risk'] = 1 if (hub_last < 610 and storage_util_last >= 7) else 0
    
    inflow_sum = sum(df[col].iloc[i-1] for col in FEATURE_COLS['inflow'] if col in available_cols)
    outflow_sum = sum(df[col].iloc[i-1] for col in FEATURE_COLS['outflow'] if col in available_cols)
    features['net_flow'] = inflow_sum - outflow_sum
    
    cmd_sum = sum(df[col].iloc[i-1] for col in FEATURE_COLS['cmd'] if col in available_cols)
    features['total_cmd'] = cmd_sum
    features['total_cmd_low'] = 1 if cmd_sum < 220 else 0
    features['total_cmd_very_low'] = 1 if cmd_sum < 200 else 0
    
    if 'HUBROOMTOTAL' in available_cols:
        hub_last = df['HUBROOMTOTAL'].iloc[i-1]
        features['hub_cmd_bottleneck'] = 1 if (hub_last < 610 and cmd_sum < 220) else 0
    
    if 'M16A_3F_STORAGE_UTIL' in available_cols:
        storage_util = df['M16A_3F_STORAGE_UTIL'].iloc[i-1]
        features['storage_util_critical'] = 1 if storage_util >= 205 else 0
        features['storage_util_high_risk'] = 1 if storage_util >= 207 else 0
    
    features['surge_risk_score'] = (
        features.get('hub_high', 0) * 3 + features.get('storage_util_critical', 0) * 2 +
        features.get('total_cmd_low', 0) * 1 + features.get('storage_util_high', 0) * 1
    )
    
    features['surge_imminent'] = 1 if (
        seq_target[-1] > 280 and features.get('target_acceleration', 0) > 0.5 and features.get('hub_high', 0) == 1
    ) else 0
    
    # M16HUB 큐
    if 'M16HUB.QUE.ALL.CURRENTQCNT' in available_cols:
        currentq = df['M16HUB.QUE.ALL.CURRENTQCNT'].iloc[i-1]
        features['currentq_high'] = 1 if currentq >= 1200 else 0
        features['currentq_critical'] = 1 if currentq >= 1400 else 0
    else:
        features['currentq_high'] = 0
        features['currentq_critical'] = 0
    
    if 'M16HUB.QUE.TIME.AVGTOTALTIME1MIN' in available_cols:
        avgtime = df['M16HUB.QUE.TIME.AVGTOTALTIME1MIN'].iloc[i-1]
        features['avgtime1min_high'] = 1 if avgtime >= 4.0 else 0
        features['avgtime1min_critical'] = 1 if avgtime >= 4.5 else 0
    else:
        features['avgtime1min_high'] = 0
        features['avgtime1min_critical'] = 0
    
    if 'M16HUB.QUE.ALL.CURRENTQCNT' in available_cols and 'M16HUB.QUE.TIME.AVGTOTALTIME1MIN' in available_cols:
        currentq = df['M16HUB.QUE.ALL.CURRENTQCNT'].iloc[i-1]
        avgtime = df['M16HUB.QUE.TIME.AVGTOTALTIME1MIN'].iloc[i-1]
        features['que_severe_bottleneck'] = 1 if (currentq >= 1200 and avgtime >= 4.0) else 0
    else:
        features['que_severe_bottleneck'] = 0
    
    # M16A 큐
    if 'M16A.QUE.OHT.OHTUTIL' in available_cols:
        ohtutil = df['M16A.QUE.OHT.OHTUTIL'].iloc[i-1]
        features['m16a_ohtutil_high'] = 1 if ohtutil >= 85.0 else 0
        features['m16a_ohtutil_critical'] = 1 if ohtutil >= 90.0 else 0
    else:
        features['m16a_ohtutil_high'] = 0
        features['m16a_ohtutil_critical'] = 0
    
    return features, seq_target[-1]

def predict_hub_categorical(csv_data):
    """HUB 범주형 예측"""
    
    # CSV 파싱
    try:
        df = pd.read_csv(StringIO(csv_data))
    except Exception as e:
        return {'error': 'CSV parsing failed', 'message': str(e)}
    
    # 컬럼 검증
    missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing_cols:
        return {
            'error': 'Missing columns',
            'message': f"필수 컬럼 누락: {', '.join(missing_cols[:5])}..."
        }
    
    # 데이터 길이 검증
    if len(df) < 30:
        return {
            'error': 'Insufficient data',
            'message': f"최소 30행 필요 (현재: {len(df)}행)"
        }
    
    # 타겟 변환
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')
    df = df.dropna(subset=[TARGET_COL])
    
    if len(df) < 30:
        return {
            'error': 'Data validation failed',
            'message': "타겟 컬럼에 유효한 데이터 부족"
        }
    
    # Feature 생성
    available_cols = set(df.columns)
    features, current_value = create_features_v8(df, available_cols)
    
    X_pred = pd.DataFrame([features])
    
    # 현재 시간
    if 'STAT_DT' in df.columns:
        try:
            df['STAT_DT'] = pd.to_datetime(df['STAT_DT'].astype(str), format='%Y%m%d%H%M')
            current_time = df['STAT_DT'].iloc[-1]
        except:
            current_time = datetime.now()
    else:
        current_time = datetime.now()
    
    # 예측
    predictions = []
    
    for horizon_min in [10, 15, 25]:
        model_path = f'HUB_MODEL/xgboost_범주형_V8_{horizon_min}분.pkl'
        
        if not os.path.exists(model_path):
            continue
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            pred_class = model.predict(X_pred)[0]
            pred_proba = model.predict_proba(X_pred)[0]
            
            prob0 = pred_proba[0] * 100
            prob1 = pred_proba[1] * 100
            prob2 = pred_proba[2] * 100
            
            if prob2 >= 70:
                status = "CRITICAL"
            elif prob2 >= 50:
                status = "HIGH"
            elif prob2 >= 30:
                status = "MEDIUM"
            else:
                status = "LOW"
            
            predictions.append({
                "horizon": horizon_min,
                "current_time": current_time.strftime('%Y-%m-%d %H:%M'),
                "pred_time": (current_time + timedelta(minutes=horizon_min)).strftime('%Y-%m-%d %H:%M'),
                "current_value": round(current_value, 2),
                "pred_class": int(pred_class),
                "class_name": CLASS_NAMES[pred_class],
                "prob0": round(prob0, 1),
                "prob1": round(prob1, 1),
                "prob2": round(prob2, 1),
                "status": status
            })
            
        except Exception as e:
            print(f"모델 로드 실패 ({horizon_min}분): {e}")
            continue
    
    if not predictions:
        return {
            'error': 'No predictions',
            'message': '예측 모델을 찾을 수 없습니다.'
        }
    
    # HTML 생성
    dashboard_html = generate_html(predictions)
    
    return {
        'predictions': predictions,
        'current_value': current_value,
        'dashboard_html': dashboard_html
    }

def generate_html(data):
    """HTML 대시보드 생성"""
    
    if not data:
        return "<html><body><h1>No Data</h1></body></html>"
    
    current_value = data[0]['current_value']
    max_prob2 = max(p['prob2'] for p in data)
    
    if max_prob2 >= 70:
        risk, risk_color = "CRITICAL", "#c53030"
    elif max_prob2 >= 50:
        risk, risk_color = "HIGH", "#dd6b20"
    elif max_prob2 >= 30:
        risk, risk_color = "MEDIUM", "#d69e2e"
    else:
        risk, risk_color = "LOW", "#38a169"
    
    chart_labels = [f"{p['horizon']}분" for p in data]
    chart_prob0 = [p['prob0'] for p in data]
    chart_prob1 = [p['prob1'] for p in data]
    chart_prob2 = [p['prob2'] for p in data]
    
    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>HUB 범주형 예측</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}body{{font-family:sans-serif;background:linear-gradient(135deg,#f093fb,#f5576c);padding:20px}}.container{{max-width:1400px;margin:0 auto}}.card{{background:#fff;border-radius:15px;padding:30px;margin-bottom:20px;box-shadow:0 10px 30px rgba(0,0,0,0.2)}}.header{{text-align:center;font-size:36px;color:#2d3748;margin-bottom:20px}}.risk{{background:{risk_color};color:#fff;padding:20px;border-radius:10px;text-align:center;font-size:24px;font-weight:700;margin-bottom:20px}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:20px;margin-bottom:30px}}.pred{{background:#f7fafc;border-radius:10px;padding:20px;text-align:center}}.time{{font-size:20px;font-weight:700;color:#f5576c;margin-bottom:10px}}.result{{font-size:18px;font-weight:700;padding:15px;background:linear-gradient(135deg,#f093fb,#f5576c);color:#fff;border-radius:10px;margin:10px 0}}.prob{{display:flex;justify-content:space-between;padding:8px;background:#fff;border-radius:8px;margin:5px 0}}.status{{display:inline-block;padding:8px 16px;border-radius:20px;font-weight:700;margin-top:10px}}.status-low{{background:#c6f6d5;color:#2f855a}}.status-medium{{background:#fefcbf;color:#b7791f}}.status-high{{background:#feebc8;color:#c05621}}.status-critical{{background:#fed7d7;color:#c53030}}.chart-container{{background:#fff;border-radius:15px;padding:30px;box-shadow:0 10px 30px rgba(0,0,0,0.2);margin-bottom:20px}}.chart-title{{font-size:24px;font-weight:700;color:#2d3748;margin-bottom:20px;text-align:center}}.chart-grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
</style></head><body><div class="container">
<div class="card"><div class="header">🎯 HUB 범주형 예측</div><div style="text-align:center;font-size:24px;color:#4299e1">Current: {current_value} → Max Surge: {max_prob2:.1f}%</div></div>
<div class="risk">{risk}</div>
<div class="grid">"""
    
    for p in data:
        html += f"""<div class="pred"><div class="time">⏱️ {p['horizon']} min</div><div style="font-size:14px;color:#718096;margin-bottom:10px">{p['pred_time']}</div>
<div class="result">{p['class_name']}</div>
<div class="prob"><span>🔴 Class 2 (급증)</span><span style="font-weight:700;color:#f5576c">{p['prob2']}%</span></div>
<div class="prob"><span>🟡 Class 1 (소폭)</span><span style="font-weight:700;color:#f5576c">{p['prob1']}%</span></div>
<div class="prob"><span>🟢 Class 0 (하락)</span><span style="font-weight:700;color:#f5576c">{p['prob0']}%</span></div>
<span class="status status-{p['status'].lower()}">{p['status']}</span></div>"""
    
    html += f"""</div>

<div class="chart-grid">
<div class="chart-container">
<div class="chart-title">🔴 급증 확률 추이 (Class 2)</div>
<canvas id="surgeChart"></canvas>
</div>

<div class="chart-container">
<div class="chart-title">📊 클래스 분포</div>
<canvas id="distributionChart"></canvas>
</div>
</div>

</div>
<script>
// 급증 확률 추이
const ctx1 = document.getElementById('surgeChart').getContext('2d');
const chart1 = new Chart(ctx1, {{
    type: 'line',
    data: {{
        labels: {json.dumps(chart_labels)},
        datasets: [{{
            label: 'Class 2 (급증) 확률',
            data: {json.dumps(chart_prob2)},
            borderColor: '#e53e3e',
            backgroundColor: 'rgba(229, 62, 62, 0.2)',
            borderWidth: 3,
            pointRadius: 8,
            pointBackgroundColor: '#e53e3e',
            pointBorderColor: '#fff',
            pointBorderWidth: 2,
            tension: 0.4,
            fill: true
        }}]
    }},
    options: {{
        responsive: true,
        maintainAspectRatio: true,
        aspectRatio: 1.5,
        plugins: {{
            legend: {{
                display: false
            }},
            tooltip: {{
                backgroundColor: 'rgba(0, 0, 0, 0.8)',
                padding: 12,
                titleFont: {{
                    size: 14,
                    weight: 'bold'
                }},
                bodyFont: {{
                    size: 13
                }},
                callbacks: {{
                    label: function(context) {{
                        return '급증 확률: ' + context.parsed.y.toFixed(1) + '%';
                    }}
                }}
            }}
        }},
        scales: {{
            x: {{
                grid: {{
                    display: false
                }},
                ticks: {{
                    font: {{
                        size: 14,
                        weight: 'bold'
                    }}
                }}
            }},
            y: {{
                beginAtZero: true,
                max: 100,
                grid: {{
                    display: true,
                    color: 'rgba(0, 0, 0, 0.1)'
                }},
                ticks: {{
                    font: {{
                        size: 12
                    }},
                    callback: function(value) {{
                        return value + '%';
                    }}
                }}
            }}
        }}
    }}
}});

// 클래스 분포
const ctx2 = document.getElementById('distributionChart').getContext('2d');
const chart2 = new Chart(ctx2, {{
    type: 'bar',
    data: {{
        labels: {json.dumps(chart_labels)},
        datasets: [
            {{
                label: 'Class 2 (급증)',
                data: {json.dumps(chart_prob2)},
                backgroundColor: 'rgba(229, 62, 62, 0.8)',
                borderColor: '#e53e3e',
                borderWidth: 2
            }},
            {{
                label: 'Class 1 (소폭)',
                data: {json.dumps(chart_prob1)},
                backgroundColor: 'rgba(237, 137, 54, 0.8)',
                borderColor: '#ed8936',
                borderWidth: 2
            }},
            {{
                label: 'Class 0 (하락)',
                data: {json.dumps(chart_prob0)},
                backgroundColor: 'rgba(56, 161, 105, 0.8)',
                borderColor: '#38a169',
                borderWidth: 2
            }}
        ]
    }},
    options: {{
        responsive: true,
        maintainAspectRatio: true,
        aspectRatio: 1.5,
        plugins: {{
            legend: {{
                display: true,
                position: 'top',
                labels: {{
                    font: {{
                        size: 13,
                        weight: 'bold'
                    }},
                    usePointStyle: true,
                    padding: 15
                }}
            }},
            tooltip: {{
                backgroundColor: 'rgba(0, 0, 0, 0.8)',
                padding: 12,
                titleFont: {{
                    size: 14,
                    weight: 'bold'
                }},
                bodyFont: {{
                    size: 13
                }},
                callbacks: {{
                    label: function(context) {{
                        return context.dataset.label + ': ' + context.parsed.y.toFixed(1) + '%';
                    }}
                }}
            }}
        }},
        scales: {{
            x: {{
                stacked: true,
                grid: {{
                    display: false
                }},
                ticks: {{
                    font: {{
                        size: 14,
                        weight: 'bold'
                    }}
                }}
            }},
            y: {{
                stacked: true,
                beginAtZero: true,
                max: 100,
                grid: {{
                    display: true,
                    color: 'rgba(0, 0, 0, 0.1)'
                }},
                ticks: {{
                    font: {{
                        size: 12
                    }},
                    callback: function(value) {{
                        return value + '%';
                    }}
                }}
            }}
        }}
    }}
}});
</script>
</body></html>"""
    
    return html