#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V8 수치형 대시보드
직접 모델 로드 & 예측 & HTML 생성
"""

import numpy as np
import pandas as pd
import pickle
import os
import sys
import webbrowser
from datetime import datetime, timedelta

# =============================================================================
# 경로 설정
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, 'model')
DATA_PATH = os.path.join(SCRIPT_DIR, 'data', 'HUBROOM_PIVOT_DATA.csv')

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

# =============================================================================
# Feature 생성
# =============================================================================
def create_features(df_30):
    """최근 30분 데이터로 Feature 생성"""
    seq_target = df_30[TARGET_COL].values
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
    
    # 각 컬럼 그룹별 Feature
    for group_name, cols in FEATURE_COLS.items():
        for col in cols:
            if col not in df_30.columns:
                continue
            col_seq = df_30[col].values
            
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
    if all(c in df_30.columns for c in ['CD_M163FSTORAGEUSE', 'CD_M163FSTORAGETOTAL', 'CD_M163FSTORAGEUTIL']):
        use_seq = df_30['CD_M163FSTORAGEUSE'].values
        total_seq = df_30['CD_M163FSTORAGETOTAL'].values
        util_seq = df_30['CD_M163FSTORAGEUTIL'].values
        features['storage_use_rate'] = (use_seq[-1] - use_seq[0]) / 30
        features['storage_remaining'] = total_seq[-1] - use_seq[-1]
        features['storage_util_last'] = util_seq[-1]
        features['storage_util_high'] = 1 if util_seq[-1] >= 7 else 0
        features['storage_util_critical'] = 1 if util_seq[-1] >= 10 else 0
    
    # HUBROOMTOTAL
    if 'HUBROOMTOTAL' in df_30.columns:
        hub_seq = df_30['HUBROOMTOTAL'].values
        hub_last = hub_seq[-1]
        features['hub_critical'] = 1 if hub_last < 590 else 0
        features['hub_high'] = 1 if hub_last < 610 else 0
        features['hub_warning'] = 1 if hub_last < 620 else 0
        features['hub_decrease_rate'] = (hub_seq[0] - hub_last) / 30
        if 'CD_M163FSTORAGEUTIL' in df_30.columns:
            storage_util_last = df_30['CD_M163FSTORAGEUTIL'].iloc[-1]
            features['hub_storage_risk'] = 1 if (hub_last < 610 and storage_util_last >= 7) else 0
    
    # Flow & CMD
    inflow_sum = sum(df_30[c].iloc[-1] for c in FEATURE_COLS['inflow'] if c in df_30.columns)
    outflow_sum = sum(df_30[c].iloc[-1] for c in FEATURE_COLS['outflow'] if c in df_30.columns)
    features['net_flow'] = inflow_sum - outflow_sum
    
    cmd_sum = sum(df_30[c].iloc[-1] for c in FEATURE_COLS['cmd'] if c in df_30.columns)
    features['total_cmd'] = cmd_sum
    features['total_cmd_low'] = 1 if cmd_sum < 220 else 0
    features['total_cmd_very_low'] = 1 if cmd_sum < 200 else 0
    
    if 'HUBROOMTOTAL' in df_30.columns:
        features['hub_cmd_bottleneck'] = 1 if (df_30['HUBROOMTOTAL'].iloc[-1] < 610 and cmd_sum < 220) else 0
    
    if 'M16A_3F_STORAGE_UTIL' in df_30.columns:
        storage_util = df_30['M16A_3F_STORAGE_UTIL'].iloc[-1]
        features['storage_util_critical'] = 1 if storage_util >= 205 else 0
        features['storage_util_high_risk'] = 1 if storage_util >= 207 else 0
    
    features['surge_risk_score'] = (
        features.get('hub_high', 0) * 3 + features.get('storage_util_critical', 0) * 2 +
        features.get('total_cmd_low', 0) * 1 + features.get('storage_util_high', 0) * 1
    )
    
    features['surge_imminent'] = 1 if (
        seq_target[-1] > 280 and features.get('target_acceleration', 0) > 0.5 and features.get('hub_high', 0) == 1
    ) else 0
    
    # Boolean features
    bool_features = [
        ('M16HUB.QUE.ALL.CURRENTQCNT', 1200, 'currentq_high'), ('M16HUB.QUE.ALL.CURRENTQCNT', 1400, 'currentq_critical'),
        ('M16HUB.QUE.TIME.AVGTOTALTIME1MIN', 4.0, 'avgtime1min_high'), ('M16HUB.QUE.TIME.AVGTOTALTIME1MIN', 4.5, 'avgtime1min_critical'),
        ('M16HUB.QUE.OHT.OHTUTIL', 85.0, 'm16hub_ohtutil_high'), ('M16HUB.QUE.OHT.OHTUTIL', 90.0, 'm16hub_ohtutil_critical'),
        ('M16HUB.QUE.TIME.AVGTOTALTIME', 5.0, 'm16hub_avgtime_high'), ('M16HUB.QUE.TIME.AVGTOTALTIME', 6.0, 'm16hub_avgtime_critical'),
        ('M16A.QUE.OHT.OHTUTIL', 85.0, 'm16a_ohtutil_high'), ('M16A.QUE.OHT.OHTUTIL', 90.0, 'm16a_ohtutil_critical'),
        ('M16A.QUE.LOAD.AVGLOADTIME1MIN', 2.5, 'm16a_loadtime_high'), ('M16A.QUE.LOAD.AVGLOADTIME1MIN', 2.8, 'm16a_loadtime_critical'),
        ('M16A.QUE.ALL.TRANSPORT4MINOVERCNT', 40, 'm16a_transport4min_high'), ('M16A.QUE.ALL.TRANSPORT4MINOVERCNT', 50, 'm16a_transport4min_critical'),
        ('M16A.QUE.ABN.QUETIMEDELAY', 1, 'm16a_delay_warning'), ('M16A.QUE.ABN.QUETIMEDELAY', 3, 'm16a_delay_critical'),
    ]
    
    for col, threshold, name in bool_features:
        features[name] = 1 if (col in df_30.columns and df_30[col].iloc[-1] >= threshold) else 0
    
    # 복합 bottleneck
    features['que_severe_bottleneck'] = 1 if (
        'M16HUB.QUE.ALL.CURRENTQCNT' in df_30.columns and 'M16HUB.QUE.TIME.AVGTOTALTIME1MIN' in df_30.columns and
        df_30['M16HUB.QUE.ALL.CURRENTQCNT'].iloc[-1] >= 1200 and df_30['M16HUB.QUE.TIME.AVGTOTALTIME1MIN'].iloc[-1] >= 4.0
    ) else 0
    
    features['m16hub_severe_bottleneck'] = 1 if (
        'M16HUB.QUE.OHT.OHTUTIL' in df_30.columns and 'M16HUB.QUE.TIME.AVGTOTALTIME' in df_30.columns and
        df_30['M16HUB.QUE.OHT.OHTUTIL'].iloc[-1] >= 85.0 and df_30['M16HUB.QUE.TIME.AVGTOTALTIME'].iloc[-1] >= 5.0
    ) else 0
    
    features['m16a_severe_bottleneck'] = 1 if (
        'M16A.QUE.OHT.OHTUTIL' in df_30.columns and 'M16A.QUE.ALL.TRANSPORT4MINOVERCNT' in df_30.columns and
        df_30['M16A.QUE.OHT.OHTUTIL'].iloc[-1] >= 85.0 and df_30['M16A.QUE.ALL.TRANSPORT4MINOVERCNT'].iloc[-1] >= 40
    ) else 0
    
    return features

# =============================================================================
# 예측
# =============================================================================
def predict_horizons(df):
    """10/15/25분 예측"""
    print("\n[STEP 1] 데이터 준비")
    print("-" * 60)
    
    # 최근 30개
    df_30 = df.tail(30).copy()
    
    # 시간 처리
    if 'STAT_DT' in df_30.columns:
        try:
            df_30['STAT_DT'] = pd.to_datetime(df_30['STAT_DT'].astype(str), format='%Y%m%d%H%M')
        except:
            base_time = datetime.now() - timedelta(minutes=29)
            df_30['STAT_DT'] = [base_time + timedelta(minutes=i) for i in range(30)]
    else:
        base_time = datetime.now() - timedelta(minutes=29)
        df_30['STAT_DT'] = [base_time + timedelta(minutes=i) for i in range(30)]
    
    current_time = df_30['STAT_DT'].iloc[-1]
    current_value = df_30[TARGET_COL].iloc[-1]
    
    print(f"현재 시간: {current_time.strftime('%Y-%m-%d %H:%M')}")
    print(f"현재 값: {current_value:.2f}")
    
    # Feature 생성
    print("\n[STEP 2] Feature 생성")
    print("-" * 60)
    features = create_features(df_30)
    X_pred = pd.DataFrame([features])
    print(f"✅ Feature: {len(features)}개")
    
    # 예측
    print("\n[STEP 3] 예측 실행")
    print("-" * 60)
    
    results = []
    for horizon_min in [10, 15, 25]:
        model_path = os.path.join(MODEL_DIR, f'xgboost_수치형_V8_{horizon_min}분.pkl')
        
        print(f"\n{horizon_min}분 예측:")
        if not os.path.exists(model_path):
            print(f"  ❌ 모델 파일 없음: {model_path}")
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
            
            result = {
                "horizon": horizon_min,
                "current_time": current_time.strftime('%Y-%m-%d %H:%M'),
                "pred_time": (current_time + timedelta(minutes=horizon_min)).strftime('%Y-%m-%d %H:%M'),
                "current_value": round(current_value, 2),
                "pred_min": round(current_value + pred_min, 2),
                "pred_max": round(current_value + pred_max, 2),
                "pred_avg": round(current_value + pred_avg, 2),
                "change_min": round(pred_min, 2),
                "change_max": round(pred_max, 2),
                "change_avg": round(pred_avg, 2),
                "status": status
            }
            
            results.append(result)
            print(f"  ✅ MAX: {result['pred_max']:.2f} ({result['status']})")
            
        except Exception as e:
            print(f"  ❌ 예측 실패: {e}")
            continue
    
    return results

# =============================================================================
# HTML 생성
# =============================================================================
def generate_html(data):
    """HTML 생성"""
    if not data:
        return "<html><body><h1>❌ No Data</h1></body></html>"
    
    current_value = data[0]['current_value']
    current_time = data[0]['current_time']
    
    pred_10 = data[0] if len(data) > 0 else {}
    pred_15 = data[1] if len(data) > 1 else {}
    pred_25 = data[2] if len(data) > 2 else {}
    
    max_value = max(p.get('pred_max', 0) for p in data)
    
    # 추세
    if len(data) == 3:
        v10, v15, v25 = data[0]['pred_max'], data[1]['pred_max'], data[2]['pred_max']
        if v10 < v15 < v25:
            trend, trend_color = "📈 지속 상승", "#e53e3e"
        elif v10 > v15 > v25:
            trend, trend_color = "📉 지속 하락", "#38a169"
        else:
            trend, trend_color = "📊 변동", "#d69e2e"
    else:
        trend, trend_color = "📊 변동", "#d69e2e"
    
    # 위험도
    if max_value >= 300:
        risk, risk_color, risk_icon = "CRITICAL", "#c53030", "🔴"
    elif max_value >= 280:
        risk, risk_color, risk_icon = "WARNING", "#dd6b20", "🟠"
    elif max_value >= 270:
        risk, risk_color, risk_icon = "CAUTION", "#d69e2e", "🟡"
    else:
        risk, risk_color, risk_icon = "NORMAL", "#38a169", "🟢"
    
    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>V8 Numerical Dashboard</title><style>
*{{margin:0;padding:0;box-sizing:border-box}}body{{font-family:-apple-system,sans-serif;background:linear-gradient(135deg,#667eea,#764ba2);padding:20px;min-height:100vh}}.container{{max-width:1400px;margin:0 auto}}.header{{background:#fff;border-radius:15px;padding:30px;margin-bottom:20px;box-shadow:0 10px 30px rgba(0,0,0,0.2)}}.header h1{{font-size:36px;color:#2d3748;margin-bottom:20px}}.header-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:20px}}.header-item{{text-align:center}}.header-label{{font-size:12px;color:#718096;text-transform:uppercase;margin-bottom:8px}}.header-value{{font-size:32px;font-weight:700;color:#2d3748}}.current-value{{color:#4299e1}}.max-value{{color:{risk_color}}}.risk-banner{{background:{risk_color};color:#fff;padding:20px;border-radius:10px;text-align:center;margin-bottom:20px}}.risk-icon{{font-size:48px;margin-bottom:10px}}.risk-text{{font-size:24px;font-weight:700}}.card{{background:#fff;border-radius:15px;padding:25px;margin-bottom:20px;box-shadow:0 10px 30px rgba(0,0,0,0.2)}}.trend{{background:#f7fafc;border-left:4px solid {trend_color};padding:20px;border-radius:8px}}.trend-title{{font-size:20px;font-weight:700;color:{trend_color};margin-bottom:10px}}.chart{{display:flex;justify-content:space-around;align-items:flex-end;height:250px;background:#f7fafc;border-radius:10px;padding:20px}}.chart-item{{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:flex-end}}.bar{{width:80%;background:linear-gradient(to top,#667eea,#764ba2);border-radius:8px 8px 0 0}}.bar-label{{margin-top:15px;font-weight:700;font-size:16px}}.bar-value{{margin-top:5px;font-size:24px;font-weight:700;color:#667eea}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:20px}}.pred-card{{background:#fff;border-radius:15px;padding:25px;box-shadow:0 10px 30px rgba(0,0,0,0.2)}}.pred-header{{text-align:center;margin-bottom:20px;padding-bottom:15px;border-bottom:2px solid #e2e8f0}}.time-badge{{font-size:24px;font-weight:700;color:#667eea}}.big-value{{font-size:48px;font-weight:700;text-align:center;margin:15px 0;color:#e53e3e}}.metric-row{{display:flex;justify-content:space-between;padding:12px;background:#f7fafc;border-radius:8px;margin-bottom:8px}}.status{{display:inline-block;padding:8px 16px;border-radius:20px;font-weight:700;font-size:14px;margin-top:10px}}.status-normal{{background:#c6f6d5;color:#2f855a}}.status-caution{{background:#fefcbf;color:#b7791f}}.status-warning{{background:#feebc8;color:#c05621}}.status-critical{{background:#fed7d7;color:#c53030}}.footer{{text-align:center;color:#fff;margin-top:30px;font-size:14px}}
</style></head><body><div class="container">
<div class="header"><h1>📊 V8 Numerical Dashboard</h1><div class="header-grid">
<div class="header-item"><div class="header-label">Current</div><div class="header-value current-value">{current_value}</div></div>
<div class="header-item"><div class="header-label">Max Prediction</div><div class="header-value max-value">{max_value:.1f}</div></div>
<div class="header-item"><div class="header-label">Risk</div><div class="header-value" style="color:{risk_color}">{risk_icon} {risk}</div></div>
</div></div>
<div class="risk-banner"><div class="risk-icon">{risk_icon}</div><div class="risk-text">{risk}</div><div style="margin-top:10px">{current_time}</div></div>
<div class="card"><div class="trend"><div class="trend-title">{trend}</div></div></div>
<div class="card"><h2 style="margin-bottom:20px">📈 MAX Values</h2><div class="chart">"""
    
    for p in data:
        height = (p['pred_max'] / max_value * 100) if max_value > 0 else 0
        html += f"""<div class="chart-item"><div class="bar" style="height:{height}%"></div><div class="bar-label">{p['horizon']} min</div><div class="bar-value">{p['pred_max']:.1f}</div></div>"""
    
    html += """</div></div><div class="grid">"""
    
    for p in data:
        html += f"""<div class="pred-card"><div class="pred-header"><div class="time-badge">⏱️ {p['horizon']} min</div><div style="font-size:14px;color:#718096;margin-top:5px">{p['pred_time']}</div></div>
<div class="big-value">{p['pred_max']:.1f}</div>
<div class="metric-row"><span>MIN</span><span style="font-weight:700">{p['pred_min']:.1f}</span></div>
<div class="metric-row"><span>AVG</span><span style="font-weight:700">{p['pred_avg']:.1f}</span></div>
<div class="metric-row"><span>Change</span><span style="font-weight:700">{p['change_max']:+.1f}</span></div>
<div style="text-align:center"><span class="status status-{p['status'].lower()}">{p['status']}</span></div>
</div>"""
    
    html += f"""</div><div class="footer"><p>🤖 V8 Numerical Prediction</p><p>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p></div></div></body></html>"""
    
    return html

# =============================================================================
# 메인
# =============================================================================
def main():
    print("=" * 60)
    print("V8 수치형 대시보드")
    print("=" * 60)
    
    # 데이터 로드
    print("\n📂 데이터 로드 중...")
    if not os.path.exists(DATA_PATH):
        print(f"❌ 데이터 파일 없음: {DATA_PATH}")
        return
    
    try:
        df = pd.read_csv(DATA_PATH, on_bad_lines='skip', encoding='utf-8', low_memory=False)
        df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')
        df = df.dropna(subset=[TARGET_COL])
        print(f"✅ {len(df):,}개 행 로드")
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return
    
    # 예측
    try:
        results = predict_horizons(df)
    except Exception as e:
        print(f"\n❌ 예측 실패: {e}")
        import traceback
        traceback.print_exc()
        return
    
    if not results:
        print("\n❌ 예측 결과 없음!")
        return
    
    # HTML 생성
    print("\n[STEP 4] HTML 생성")
    print("-" * 60)
    html = generate_html(results)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'V8_Numerical_{timestamp}.html'
    filepath = os.path.join(SCRIPT_DIR, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ HTML 생성: {filename}")
    
    # 브라우저
    try:
        webbrowser.open('file://' + os.path.abspath(filepath))
        print(f"🌐 브라우저 열기...")
    except:
        print(f"⚠️ 수동으로 여세요: {filepath}")
    
    print("\n" + "=" * 60)
    print("✅ 완료!")
    print("=" * 60)

if __name__ == '__main__':
    main()