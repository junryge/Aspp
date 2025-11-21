#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HUB 수치형 예측 모듈 (서버용)"""

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

def create_features_v8(df, available_cols):
    """Feature 생성 - 마지막 30개 사용"""
    
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
    
    # 추가 Feature
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

def predict_hub_numerical(csv_data):
    """HUB 수치형 예측"""
    
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
        model_path = f'HUB_MODEL/xgboost_수치형_V8_{horizon_min}분.pkl'
        
        if not os.path.exists(model_path):
            continue
        
        try:
            with open(model_path, 'rb') as f:
                model_dict = pickle.load(f)
            models = model_dict['models']
            
            pred_min = models[0].predict(X_pred)[0]
            pred_max = models[1].predict(X_pred)[0]
            pred_avg = models[2].predict(X_pred)[0]
            
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
                "pred_avg": round(current_value + pred_avg, 2),
                "change_min": round(pred_min, 2),
                "change_max": round(pred_max, 2),
                "change_avg": round(pred_avg, 2),
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
    
    # 과거 30분 데이터
    historical_data = df[TARGET_COL].tail(30).values.tolist()
    if 'STAT_DT' in df.columns:
        historical_times = df['STAT_DT'].tail(30).tolist()
        historical_times = [t.strftime('%H:%M') for t in historical_times]
    else:
        historical_times = [f"-{30-i}분" for i in range(30)]
    
    # HTML 생성
    dashboard_html = generate_html(predictions, historical_data, historical_times)
    
    return {
        'predictions': predictions,
        'current_value': current_value,
        'dashboard_html': dashboard_html
    }

def generate_html(data, historical_data, historical_times):
    """HTML 대시보드 생성"""
    
    if not data:
        return "<html><body><h1>No Data</h1></body></html>"
    
    current_value = data[0]['current_value']
    max_value = max(p['pred_max'] for p in data)
    
    if max_value >= 300:
        risk, risk_color = "CRITICAL", "#c53030"
    elif max_value >= 280:
        risk, risk_color = "WARNING", "#dd6b20"
    elif max_value >= 270:
        risk, risk_color = "CAUTION", "#d69e2e"
    else:
        risk, risk_color = "NORMAL", "#38a169"
    
    chart_labels = historical_times + [p['pred_time_label'] for p in data]
    chart_historical = historical_data + [None] * len(data)
    chart_pred_min = [None] * len(historical_times) + [p['pred_min'] for p in data]
    chart_pred_max = [None] * len(historical_times) + [p['pred_max'] for p in data]
    
    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>HUB 수치형 예측</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}body{{font-family:sans-serif;background:linear-gradient(135deg,#667eea,#764ba2);padding:20px}}.container{{max-width:1400px;margin:0 auto}}.card{{background:#fff;border-radius:15px;padding:30px;margin-bottom:20px;box-shadow:0 10px 30px rgba(0,0,0,0.2)}}.header{{text-align:center;font-size:36px;color:# 2d3748;margin-bottom:20px}}.risk{{background:{risk_color};color:#fff;padding:20px;border-radius:10px;text-align:center;font-size:24px;font-weight:700;margin-bottom:20px}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:20px;margin-bottom:30px}}.pred{{background:#f7fafc;border-radius:10px;padding:20px;text-align:center}}.time{{font-size:20px;font-weight:700;color:#667eea;margin-bottom:10px}}.range{{font-size:36px;font-weight:700;color:#e53e3e;margin:15px 0;line-height:1.2}}.range-label{{font-size:14px;color:#718096;margin-top:5px}}.metric{{display:flex;justify-content:space-between;padding:8px;background:#fff;border-radius:8px;margin:5px 0;font-size:16px}}.status{{display:inline-block;padding:8px 16px;border-radius:20px;font-weight:700;margin-top:10px}}.status-normal{{background:#c6f6d5;color:#2f855a}}.status-caution{{background:#fefcbf;color:#b7791f}}.status-warning{{background:#feebc8;color:#c05621}}.status-critical{{background:#fed7d7;color:#c53030}}.chart-container{{background:#fff;border-radius:15px;padding:30px;box-shadow:0 10px 30px rgba(0,0,0,0.2)}}.chart-title{{font-size:24px;font-weight:700;color:#2d3748;margin-bottom:20px;text-align:center}}
</style></head><body><div class="container">
<div class="card"><div class="header">📊 HUB 수치형 예측</div><div style="text-align:center;font-size:24px;color:#4299e1">Current: {current_value} → Max: {max_value:.1f}</div></div>
<div class="risk">{risk}</div>
<div class="grid">"""
    
    for p in data:
        html += f"""<div class="pred"><div class="time">⏱️ {p['horizon']} min</div><div style="font-size:14px;color:#718096;margin-bottom:10px">{p['pred_time']}</div>
<div class="range">{p['pred_min']:.1f} ~ {p['pred_max']:.1f}<div class="range-label">예측 범위</div></div>
<div class="metric"><span>Change (MIN)</span><span style="font-weight:700">{p['change_min']:+.1f}</span></div>
<div class="metric"><span>Change (MAX)</span><span style="font-weight:700">{p['change_max']:+.1f}</span></div>
<span class="status status-{p['status'].lower()}">{p['status']}</span></div>"""
    
    html += f"""</div>
<div class="chart-container">
<div class="chart-title">📈 30분 과거 데이터 + 예측 범위 (MIN ~ MAX)</div>
<canvas id="timeSeriesChart"></canvas>
</div>
</div>
<script>
const ctx = document.getElementById('timeSeriesChart').getContext('2d');
const chart = new Chart(ctx, {{
    type: 'line',
    data: {{
        labels: {json.dumps(chart_labels)},
        datasets: [
            {{
                label: '과거 데이터',
                data: {json.dumps(chart_historical)},
                borderColor: '#667eea',
                backgroundColor: 'rgba(102, 126, 234, 0.1)',
                borderWidth: 3,
                pointRadius: 3,
                pointBackgroundColor: '#667eea',
                tension: 0.4,
                fill: false
            }},
            {{
                label: '예측 MAX',
                data: {json.dumps(chart_pred_max)},
                borderColor: '#e53e3e',
                backgroundColor: 'rgba(229, 62, 62, 0.1)',
                borderWidth: 3,
                borderDash: [5, 5],
                pointRadius: 6,
                pointBackgroundColor: '#e53e3e',
                tension: 0.4,
                fill: '+1'
            }},
            {{
                label: '예측 MIN',
                data: {json.dumps(chart_pred_min)},
                borderColor: '#38a169',
                backgroundColor: 'rgba(56, 161, 105, 0.2)',
                borderWidth: 3,
                borderDash: [5, 5],
                pointRadius: 6,
                pointBackgroundColor: '#38a169',
                tension: 0.4,
                fill: false
            }}
        ]
    }},
    options: {{
        responsive: true,
        maintainAspectRatio: true,
        aspectRatio: 2.5,
        interaction: {{
            mode: 'index',
            intersect: false
        }},
        plugins: {{
            legend: {{
                display: true,
                position: 'top',
                labels: {{
                    font: {{
                        size: 14,
                        weight: 'bold'
                    }},
                    usePointStyle: true,
                    padding: 20
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
                        if (context.parsed.y === null) return null;
                        return context.dataset.label + ': ' + context.parsed.y.toFixed(1);
                    }}
                }}
            }}
        }},
        scales: {{
            x: {{
                grid: {{
                    display: true,
                    color: 'rgba(0, 0, 0, 0.05)'
                }},
                ticks: {{
                    font: {{
                        size: 12
                    }},
                    maxRotation: 45,
                    minRotation: 45
                }}
            }},
            y: {{
                beginAtZero: false,
                grid: {{
                    display: true,
                    color: 'rgba(0, 0, 0, 0.1)'
                }},
                ticks: {{
                    font: {{
                        size: 12
                    }}
                }}
            }}
        }}
    }}
}});
</script>
</body></html>"""
    
    return html