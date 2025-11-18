# -*- coding: utf-8 -*-
"""
V8 수치형 대시보드 - 독립 실행 버전
직접 모델 로드하고 예측해서 HTML 생성
"""

import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
import os
import webbrowser

# 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, 'model')
data_path = os.path.join(script_dir, 'data', 'HUBROOM_PIVOT_DATA.csv')

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
        'M16HUB.QUE.ALL.CURRENTQCOMPLETED',
        'M16HUB.QUE.ALL.FABTRANSJOBCNT',
        'M16HUB.QUE.TIME.AVGTOTALTIME',
        'M16HUB.QUE.OHT.CURRENTOHTQCNT',
        'M16HUB.QUE.OHT.OHTUTIL'
    ],
    'm16a_que': [
        'M16A.QUE.ALL.CURRENTQCOMPLETED',
        'M16A.QUE.ALL.CURRENTQCREATED',
        'M16A.QUE.OHT.CURRENTOHTQCNT',
        'M16A.QUE.OHT.OHTUTIL',
        'M16A.QUE.LOAD.AVGLOADTIME1MIN',
        'M16A.QUE.ALL.TRANSPORT4MINOVERCNT',
        'M16A.QUE.ABN.QUETIMEDELAY'
    ]
}

TARGET_COL = 'CURRENT_M16A_3F_JOB_2'

def create_features(recent_30):
    """Feature 생성 (학습과 동일)"""
    seq_target = recent_30[TARGET_COL].values
    
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
            if col not in recent_30.columns:
                continue
            
            col_seq = recent_30[col].values
            
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
    
    # FS Storage
    if all(col in recent_30.columns for col in ['CD_M163FSTORAGEUSE', 'CD_M163FSTORAGETOTAL', 'CD_M163FSTORAGEUTIL']):
        storage_use = recent_30['CD_M163FSTORAGEUSE'].values
        storage_total = recent_30['CD_M163FSTORAGETOTAL'].values
        storage_util = recent_30['CD_M163FSTORAGEUTIL'].values
        features['storage_use_rate'] = (storage_use[-1] - storage_use[0]) / 30
        features['storage_remaining'] = storage_total[-1] - storage_use[-1]
        features['storage_util_last'] = storage_util[-1]
        features['storage_util_high'] = 1 if storage_util[-1] >= 7 else 0
        features['storage_util_critical'] = 1 if storage_util[-1] >= 10 else 0
    
    # HUBROOMTOTAL
    if 'HUBROOMTOTAL' in recent_30.columns:
        hub_seq = recent_30['HUBROOMTOTAL'].values
        hub_last = hub_seq[-1]
        features['hub_critical'] = 1 if hub_last < 590 else 0
        features['hub_high'] = 1 if hub_last < 610 else 0
        features['hub_warning'] = 1 if hub_last < 620 else 0
        features['hub_decrease_rate'] = (hub_seq[0] - hub_last) / 30
        if 'CD_M163FSTORAGEUTIL' in recent_30.columns:
            storage_util_last = recent_30['CD_M163FSTORAGEUTIL'].iloc[-1]
            features['hub_storage_risk'] = 1 if (hub_last < 610 and storage_util_last >= 7) else 0
    
    # Flows
    inflow_sum = sum(recent_30[col].iloc[-1] for col in FEATURE_COLS['inflow'] if col in recent_30.columns)
    outflow_sum = sum(recent_30[col].iloc[-1] for col in FEATURE_COLS['outflow'] if col in recent_30.columns)
    features['net_flow'] = inflow_sum - outflow_sum
    
    cmd_sum = sum(recent_30[col].iloc[-1] for col in FEATURE_COLS['cmd'] if col in recent_30.columns)
    features['total_cmd'] = cmd_sum
    features['total_cmd_low'] = 1 if cmd_sum < 220 else 0
    features['total_cmd_very_low'] = 1 if cmd_sum < 200 else 0
    
    if 'HUBROOMTOTAL' in recent_30.columns:
        hub_last = recent_30['HUBROOMTOTAL'].iloc[-1]
        features['hub_cmd_bottleneck'] = 1 if (hub_last < 610 and cmd_sum < 220) else 0
    
    if 'M16A_3F_STORAGE_UTIL' in recent_30.columns:
        storage_util = recent_30['M16A_3F_STORAGE_UTIL'].iloc[-1]
        features['storage_util_critical'] = 1 if storage_util >= 205 else 0
        features['storage_util_high_risk'] = 1 if storage_util >= 207 else 0
    
    features['surge_risk_score'] = (
        features.get('hub_high', 0) * 3 +
        features.get('storage_util_critical', 0) * 2 +
        features.get('total_cmd_low', 0) * 1 +
        features.get('storage_util_high', 0) * 1
    )
    
    features['surge_imminent'] = 1 if (
        seq_target[-1] > 280 and
        features.get('target_acceleration', 0) > 0.5 and
        features.get('hub_high', 0) == 1
    ) else 0
    
    # Boolean features (간결하게)
    bool_features = [
        ('M16HUB.QUE.ALL.CURRENTQCNT', [(1200, 'currentq_high'), (1400, 'currentq_critical')]),
        ('M16HUB.QUE.TIME.AVGTOTALTIME1MIN', [(4.0, 'avgtime1min_high'), (4.5, 'avgtime1min_critical')]),
        ('M16HUB.QUE.OHT.OHTUTIL', [(85.0, 'm16hub_ohtutil_high'), (90.0, 'm16hub_ohtutil_critical')]),
        ('M16HUB.QUE.TIME.AVGTOTALTIME', [(5.0, 'm16hub_avgtime_high'), (6.0, 'm16hub_avgtime_critical')]),
        ('M16A.QUE.OHT.OHTUTIL', [(85.0, 'm16a_ohtutil_high'), (90.0, 'm16a_ohtutil_critical')]),
        ('M16A.QUE.LOAD.AVGLOADTIME1MIN', [(2.5, 'm16a_loadtime_high'), (2.8, 'm16a_loadtime_critical')]),
        ('M16A.QUE.ALL.TRANSPORT4MINOVERCNT', [(40, 'm16a_transport4min_high'), (50, 'm16a_transport4min_critical')]),
        ('M16A.QUE.ABN.QUETIMEDELAY', [(1, 'm16a_delay_warning'), (3, 'm16a_delay_critical')]),
    ]
    
    for col, thresholds in bool_features:
        if col in recent_30.columns:
            val = recent_30[col].iloc[-1]
            for thr, name in thresholds:
                features[name] = 1 if val >= thr else 0
        else:
            for thr, name in thresholds:
                features[name] = 0
    
    # 복합 bottleneck
    if 'M16HUB.QUE.ALL.CURRENTQCNT' in recent_30.columns and 'M16HUB.QUE.TIME.AVGTOTALTIME1MIN' in recent_30.columns:
        currentq = recent_30['M16HUB.QUE.ALL.CURRENTQCNT'].iloc[-1]
        avgtime = recent_30['M16HUB.QUE.TIME.AVGTOTALTIME1MIN'].iloc[-1]
        features['que_severe_bottleneck'] = 1 if (currentq >= 1200 and avgtime >= 4.0) else 0
    else:
        features['que_severe_bottleneck'] = 0
    
    if 'M16HUB.QUE.OHT.OHTUTIL' in recent_30.columns and 'M16HUB.QUE.TIME.AVGTOTALTIME' in recent_30.columns:
        ohtutil = recent_30['M16HUB.QUE.OHT.OHTUTIL'].iloc[-1]
        avgtime = recent_30['M16HUB.QUE.TIME.AVGTOTALTIME'].iloc[-1]
        features['m16hub_severe_bottleneck'] = 1 if (ohtutil >= 85.0 and avgtime >= 5.0) else 0
    else:
        features['m16hub_severe_bottleneck'] = 0
    
    if 'M16A.QUE.OHT.OHTUTIL' in recent_30.columns and 'M16A.QUE.ALL.TRANSPORT4MINOVERCNT' in recent_30.columns:
        ohtutil = recent_30['M16A.QUE.OHT.OHTUTIL'].iloc[-1]
        transport4min = recent_30['M16A.QUE.ALL.TRANSPORT4MINOVERCNT'].iloc[-1]
        features['m16a_severe_bottleneck'] = 1 if (ohtutil >= 85.0 and transport4min >= 40) else 0
    else:
        features['m16a_severe_bottleneck'] = 0
    
    return features

def predict_all_horizons(df):
    """10분, 15분, 25분 모두 예측"""
    
    recent_30 = df.tail(30).copy()
    
    if 'STAT_DT' in recent_30.columns:
        try:
            recent_30['STAT_DT'] = pd.to_datetime(recent_30['STAT_DT'].astype(str), format='%Y%m%d%H%M')
        except:
            base_time = datetime.now() - timedelta(minutes=29)
            recent_30['STAT_DT'] = [base_time + timedelta(minutes=i) for i in range(30)]
    else:
        base_time = datetime.now() - timedelta(minutes=29)
        recent_30['STAT_DT'] = [base_time + timedelta(minutes=i) for i in range(30)]
    
    current_time = recent_30['STAT_DT'].iloc[-1]
    current_value = recent_30[TARGET_COL].iloc[-1]
    
    features = create_features(recent_30)
    X_pred = pd.DataFrame([features])
    
    results = []
    
    for horizon_min, horizon_name in [(10, '10MIN'), (15, '15MIN'), (25, '25MIN')]:
        model_path = os.path.join(model_dir, f'xgboost_수치형_V8_{horizon_min}분.pkl')
        
        try:
            with open(model_path, 'rb') as f:
                model_dict = pickle.load(f)
            models = model_dict['models']
            
            pred_min = models[0].predict(X_pred)[0]
            pred_max = models[1].predict(X_pred)[0]
            pred_avg = models[2].predict(X_pred)[0]
            
            if pred_max >= 300:
                status = "CRITICAL"
                surge_warning = 1
            elif pred_max >= 280:
                status = "WARNING"
                surge_warning = 1
            elif pred_max >= 270:
                status = "CAUTION"
                surge_warning = 0
            else:
                status = "NORMAL"
                surge_warning = 0
            
            result = {
                "TIME_HORIZON": horizon_name,
                "CURRENT_TIME": current_time.strftime('%Y-%m-%d %H:%M'),
                "CURRENT_VALUE": round(current_value, 2),
                "PREDICTION_TIME": (current_time + timedelta(minutes=horizon_min)).strftime('%Y-%m-%d %H:%M'),
                "PREDICT_MIN": round(current_value + pred_min, 2),
                "PREDICT_MAX": round(current_value + pred_max, 2),
                "PREDICT_AVG": round(current_value + pred_avg, 2),
                "CHANGE_MIN": round(pred_min, 2),
                "CHANGE_MAX": round(pred_max, 2),
                "CHANGE_AVG": round(pred_avg, 2),
                "SURGE_WARNING": surge_warning,
                "STATUS": status
            }
            results.append(result)
            
        except Exception as e:
            print(f"❌ {horizon_name} failed: {e}")
            continue
    
    return results

def generate_html(data):
    """HTML 생성"""
    if not data:
        return "<html><body><h1>No Data</h1></body></html>"
    
    current_time = data[0]['CURRENT_TIME']
    current_value = data[0]['CURRENT_VALUE']
    
    pred_10 = data[0] if len(data) > 0 else {}
    pred_15 = data[1] if len(data) > 1 else {}
    pred_25 = data[2] if len(data) > 2 else {}
    
    max_value = max(pred_10.get('PREDICT_MAX', 0), pred_15.get('PREDICT_MAX', 0), pred_25.get('PREDICT_MAX', 0))
    
    if pred_10.get('PREDICT_MAX', 0) < pred_15.get('PREDICT_MAX', 0) < pred_25.get('PREDICT_MAX', 0):
        trend, trend_color, trend_alert = "📈 지속 상승", "#e53e3e", "위험도 증가 중"
    elif pred_10.get('PREDICT_MAX', 0) > pred_15.get('PREDICT_MAX', 0) > pred_25.get('PREDICT_MAX', 0):
        trend, trend_color, trend_alert = "📉 지속 하락", "#38a169", "안정화 중"
    else:
        trend, trend_color, trend_alert = "📊 변동", "#d69e2e", "불규칙 변화"
    
    if max_value >= 300:
        risk_level, risk_color, risk_icon = "CRITICAL", "#c53030", "🔴"
    elif max_value >= 280:
        risk_level, risk_color, risk_icon = "WARNING", "#dd6b20", "🟠"
    elif max_value >= 270:
        risk_level, risk_color, risk_icon = "CAUTION", "#d69e2e", "🟡"
    else:
        risk_level, risk_color, risk_icon = "NORMAL", "#38a169", "🟢"
    
    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>V8 Numerical Dashboard</title><style>
*{{margin:0;padding:0;box-sizing:border-box}}body{{font-family:-apple-system,sans-serif;background:linear-gradient(135deg,#667eea,#764ba2);padding:20px;min-height:100vh}}.container{{max-width:1400px;margin:0 auto}}.header{{background:#fff;border-radius:15px;padding:30px;margin-bottom:20px;box-shadow:0 10px 30px rgba(0,0,0,0.2)}}.header h1{{font-size:36px;color:#2d3748;margin-bottom:10px}}.header-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:20px;margin-top:20px}}.header-item{{text-align:center}}.header-label{{font-size:12px;color:#718096;text-transform:uppercase;margin-bottom:8px}}.header-value{{font-size:32px;font-weight:700;color:#2d3748}}.current-value{{color:#4299e1}}.max-value{{color:{risk_color}}}.risk-banner{{background:{risk_color};color:#fff;padding:20px;border-radius:10px;text-align:center;margin-bottom:20px}}.risk-banner-icon{{font-size:48px;margin-bottom:10px}}.risk-banner-text{{font-size:24px;font-weight:700}}.summary-card{{background:#fff;border-radius:15px;padding:25px;margin-bottom:20px;box-shadow:0 10px 30px rgba(0,0,0,0.2)}}.trend-card{{background:#f7fafc;border-left:4px solid {trend_color};padding:20px;border-radius:8px}}.trend-title{{font-size:20px;font-weight:700;color:{trend_color};margin-bottom:10px}}.chart-container{{background:#fff;border-radius:15px;padding:25px;margin-bottom:20px;box-shadow:0 10px 30px rgba(0,0,0,0.2)}}.timeline{{display:flex;justify-content:space-around;align-items:flex-end;height:250px;background:#f7fafc;border-radius:10px;padding:20px}}.timeline-item{{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:flex-end}}.bar{{width:80%;background:linear-gradient(to top,#667eea,#764ba2);border-radius:8px 8px 0 0}}.bar-label{{margin-top:15px;font-weight:700;font-size:16px}}.bar-value{{margin-top:5px;font-size:24px;font-weight:700;color:#667eea}}.predictions-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:20px;margin-bottom:20px}}.prediction-card{{background:#fff;border-radius:15px;padding:25px;box-shadow:0 10px 30px rgba(0,0,0,0.2)}}.card-header{{text-align:center;margin-bottom:20px;padding-bottom:15px;border-bottom:2px solid #e2e8f0}}.time-badge{{font-size:24px;font-weight:700;color:#667eea}}.metric-big{{font-size:48px;font-weight:700;text-align:center;margin:15px 0;color:#e53e3e}}.metric-row{{display:flex;justify-content:space-between;padding:12px;background:#f7fafc;border-radius:8px;margin-bottom:8px}}.status-badge{{display:inline-block;padding:8px 16px;border-radius:20px;font-weight:700;font-size:14px;margin-top:10px}}.status-normal{{background:#c6f6d5;color:#2f855a}}.status-caution{{background:#fefcbf;color:#b7791f}}.status-warning{{background:#feebc8;color:#c05621}}.status-critical{{background:#fed7d7;color:#c53030}}.footer{{text-align:center;color:#fff;margin-top:30px;font-size:14px}}
</style></head><body><div class="container">
<div class="header"><h1>📊 V8 Numerical Dashboard</h1><div class="header-grid">
<div class="header-item"><div class="header-label">Current</div><div class="header-value current-value">{current_value}</div></div>
<div class="header-item"><div class="header-label">Max Prediction</div><div class="header-value max-value">{max_value:.1f}</div></div>
<div class="header-item"><div class="header-label">Risk</div><div class="header-value" style="color:{risk_color}">{risk_icon} {risk_level}</div></div>
</div></div>
<div class="risk-banner"><div class="risk-banner-icon">{risk_icon}</div><div class="risk-banner-text">{risk_level}</div><div style="margin-top:10px">{current_time}</div></div>
<div class="summary-card"><div class="trend-card"><div class="trend-title">{trend}</div><div>{trend_alert}</div></div></div>
<div class="chart-container"><h2 style="margin-bottom:20px">📈 MAX Values</h2><div class="timeline">
<div class="timeline-item"><div class="bar" style="height:{pred_10.get('PREDICT_MAX',0)/max_value*100 if max_value>0 else 0}%"></div><div class="bar-label">10 min</div><div class="bar-value">{pred_10.get('PREDICT_MAX',0):.1f}</div></div>
<div class="timeline-item"><div class="bar" style="height:{pred_15.get('PREDICT_MAX',0)/max_value*100 if max_value>0 else 0}%"></div><div class="bar-label">15 min</div><div class="bar-value">{pred_15.get('PREDICT_MAX',0):.1f}</div></div>
<div class="timeline-item"><div class="bar" style="height:{pred_25.get('PREDICT_MAX',0)/max_value*100 if max_value>0 else 0}%"></div><div class="bar-label">25 min</div><div class="bar-value">{pred_25.get('PREDICT_MAX',0):.1f}</div></div>
</div></div>
<div class="predictions-grid">"""

    for pred in [pred_10, pred_15, pred_25]:
        time_label = pred.get('TIME_HORIZON', '').replace('MIN', ' min')
        html += f"""
<div class="prediction-card"><div class="card-header"><div class="time-badge">⏱️ {time_label}</div><div style="font-size:14px;color:#718096;margin-top:5px">{pred.get('PREDICTION_TIME','')}</div></div>
<div class="metric-big">{pred.get('PREDICT_MAX',0):.1f}</div>
<div class="metric-row"><span>MIN</span><span style="font-weight:700">{pred.get('PREDICT_MIN',0):.1f}</span></div>
<div class="metric-row"><span>AVG</span><span style="font-weight:700">{pred.get('PREDICT_AVG',0):.1f}</span></div>
<div class="metric-row"><span>Change</span><span style="font-weight:700">{pred.get('CHANGE_MAX',0):+.1f}</span></div>
<div style="text-align:center"><span class="status-badge status-{pred.get('STATUS','').lower()}">{pred.get('STATUS','')}</span></div>
{'<div style="margin-top:15px;padding:10px;background:#fed7d7;border-radius:8px;text-align:center;color:#c53030;font-weight:700">🚨 SURGE</div>' if pred.get('SURGE_WARNING',0)==1 else ''}
</div>"""

    html += f"""
</div><div class="footer"><p>🤖 V8 Numerical Prediction</p><p>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p></div>
</div></body></html>"""
    
    return html

def main():
    print("="*60)
    print("V8 Numerical Dashboard")
    print("="*60)
    
    print("\n[1/3] Loading data...")
    try:
        df = pd.read_csv(data_path, on_bad_lines='skip', encoding='utf-8', low_memory=False)
        df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')
        df = df.dropna(subset=[TARGET_COL])
        print(f"✅ {len(df)} rows")
    except Exception as e:
        print(f"❌ Load failed: {e}")
        return
    
    print("\n[2/3] Predicting...")
    data = predict_all_horizons(df)
    
    if not data:
        print("❌ Prediction failed!")
        return
    
    print(f"✅ {len(data)} predictions")
    
    print("\n[3/3] Generating HTML...")
    html_content = generate_html(data)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'V8_Numerical_Dashboard_{timestamp}.html'
    filepath = os.path.join(script_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n✅ Created: {filename}")
    
    try:
        webbrowser.open('file://' + os.path.abspath(filepath))
        print(f"🌐 Opening browser...")
    except:
        print(f"⚠️ Open manually: {filepath}")

if __name__ == '__main__':
    main()