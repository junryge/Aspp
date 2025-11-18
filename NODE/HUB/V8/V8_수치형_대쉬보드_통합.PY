#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V8 수치형 대시보드"""

import numpy as np
import pandas as pd
import pickle
import os
import webbrowser
from datetime import datetime, timedelta

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

def create_features_v8(df, available_cols):
    """학습과 동일한 Feature 생성 - 마지막 30개 사용"""
    
    # 마지막 30개 데이터 사용
    if len(df) < 30:
        raise ValueError(f"데이터 부족: {len(df)}개 (최소 30개 필요)")
    
    # 학습과 동일하게: i는 예측 시점, i-30:i는 시퀀스
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
    
    if 'M16HUB.QUE.OHT.OHTUTIL' in available_cols:
        ohtutil = df['M16HUB.QUE.OHT.OHTUTIL'].iloc[i-1]
        features['m16hub_ohtutil_high'] = 1 if ohtutil >= 85.0 else 0
        features['m16hub_ohtutil_critical'] = 1 if ohtutil >= 90.0 else 0
    else:
        features['m16hub_ohtutil_high'] = 0
        features['m16hub_ohtutil_critical'] = 0
    
    if 'M16HUB.QUE.TIME.AVGTOTALTIME' in available_cols:
        avgtime = df['M16HUB.QUE.TIME.AVGTOTALTIME'].iloc[i-1]
        features['m16hub_avgtime_high'] = 1 if avgtime >= 5.0 else 0
        features['m16hub_avgtime_critical'] = 1 if avgtime >= 6.0 else 0
    else:
        features['m16hub_avgtime_high'] = 0
        features['m16hub_avgtime_critical'] = 0
    
    if 'M16HUB.QUE.OHT.OHTUTIL' in available_cols and 'M16HUB.QUE.TIME.AVGTOTALTIME' in available_cols:
        ohtutil = df['M16HUB.QUE.OHT.OHTUTIL'].iloc[i-1]
        avgtime = df['M16HUB.QUE.TIME.AVGTOTALTIME'].iloc[i-1]
        features['m16hub_severe_bottleneck'] = 1 if (ohtutil >= 85.0 and avgtime >= 5.0) else 0
    else:
        features['m16hub_severe_bottleneck'] = 0
    
    if 'M16A.QUE.OHT.OHTUTIL' in available_cols:
        ohtutil = df['M16A.QUE.OHT.OHTUTIL'].iloc[i-1]
        features['m16a_ohtutil_high'] = 1 if ohtutil >= 85.0 else 0
        features['m16a_ohtutil_critical'] = 1 if ohtutil >= 90.0 else 0
    else:
        features['m16a_ohtutil_high'] = 0
        features['m16a_ohtutil_critical'] = 0
    
    if 'M16A.QUE.LOAD.AVGLOADTIME1MIN' in available_cols:
        loadtime = df['M16A.QUE.LOAD.AVGLOADTIME1MIN'].iloc[i-1]
        features['m16a_loadtime_high'] = 1 if loadtime >= 2.5 else 0
        features['m16a_loadtime_critical'] = 1 if loadtime >= 2.8 else 0
    else:
        features['m16a_loadtime_high'] = 0
        features['m16a_loadtime_critical'] = 0
    
    if 'M16A.QUE.ALL.TRANSPORT4MINOVERCNT' in available_cols:
        transport4min = df['M16A.QUE.ALL.TRANSPORT4MINOVERCNT'].iloc[i-1]
        features['m16a_transport4min_high'] = 1 if transport4min >= 40 else 0
        features['m16a_transport4min_critical'] = 1 if transport4min >= 50 else 0
    else:
        features['m16a_transport4min_high'] = 0
        features['m16a_transport4min_critical'] = 0
    
    if 'M16A.QUE.ABN.QUETIMEDELAY' in available_cols:
        delay = df['M16A.QUE.ABN.QUETIMEDELAY'].iloc[i-1]
        features['m16a_delay_warning'] = 1 if delay >= 1 else 0
        features['m16a_delay_critical'] = 1 if delay >= 3 else 0
    else:
        features['m16a_delay_warning'] = 0
        features['m16a_delay_critical'] = 0
    
    if 'M16A.QUE.OHT.OHTUTIL' in available_cols and 'M16A.QUE.ALL.TRANSPORT4MINOVERCNT' in available_cols:
        ohtutil = df['M16A.QUE.OHT.OHTUTIL'].iloc[i-1]
        transport4min = df['M16A.QUE.ALL.TRANSPORT4MINOVERCNT'].iloc[i-1]
        features['m16a_severe_bottleneck'] = 1 if (ohtutil >= 85.0 and transport4min >= 40) else 0
    else:
        features['m16a_severe_bottleneck'] = 0
    
    return features, seq_target[-1]

def predict_horizons(df):
    """10/15/25분 예측"""
    
    if 'STAT_DT' in df.columns:
        try:
            df['STAT_DT'] = pd.to_datetime(df['STAT_DT'].astype(str), format='%Y%m%d%H%M')
            current_time = df['STAT_DT'].iloc[-1]
        except:
            current_time = datetime.now()
    else:
        current_time = datetime.now()
    
    print(f"현재 시간: {current_time.strftime('%Y-%m-%d %H:%M')}")
    print(f"데이터: {len(df)}개")
    
    available_cols = set(df.columns)
    features, current_value = create_features_v8(df, available_cols)
    
    print(f"현재값: {current_value:.2f}")
    print(f"Feature: {len(features)}개")
    
    X_pred = pd.DataFrame([features])
    results = []
    
    for horizon_min in [10, 15, 25]:
        model_path = os.path.join(MODEL_DIR, f'xgboost_수치형_V8_{horizon_min}분.pkl')
        
        if not os.path.exists(model_path):
            print(f"❌ {horizon_min}분: 모델 없음")
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
            print(f"✅ {horizon_min}분: {result['pred_max']:.1f} ({status})")
            
        except Exception as e:
            print(f"❌ {horizon_min}분: {e}")
            continue
    
    return results

def generate_html(data):
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
    
    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>V8 Numerical</title><style>
*{{margin:0;padding:0;box-sizing:border-box}}body{{font-family:sans-serif;background:linear-gradient(135deg,#667eea,#764ba2);padding:20px}}.container{{max-width:1200px;margin:0 auto}}.card{{background:#fff;border-radius:15px;padding:30px;margin-bottom:20px;box-shadow:0 10px 30px rgba(0,0,0,0.2)}}.header{{text-align:center;font-size:36px;color:#2d3748;margin-bottom:20px}}.risk{{background:{risk_color};color:#fff;padding:20px;border-radius:10px;text-align:center;font-size:24px;font-weight:700;margin-bottom:20px}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:20px}}.pred{{background:#f7fafc;border-radius:10px;padding:20px;text-align:center}}.time{{font-size:20px;font-weight:700;color:#667eea;margin-bottom:10px}}.range{{font-size:36px;font-weight:700;color:#e53e3e;margin:15px 0;line-height:1.2}}.range-label{{font-size:14px;color:#718096;margin-top:5px}}.metric{{display:flex;justify-content:space-between;padding:8px;background:#fff;border-radius:8px;margin:5px 0;font-size:16px}}.status{{display:inline-block;padding:8px 16px;border-radius:20px;font-weight:700;margin-top:10px}}.status-normal{{background:#c6f6d5;color:#2f855a}}.status-caution{{background:#fefcbf;color:#b7791f}}.status-warning{{background:#feebc8;color:#c05621}}.status-critical{{background:#fed7d7;color:#c53030}}
</style></head><body><div class="container">
<div class="card"><div class="header">📊 V8 Numerical</div><div style="text-align:center;font-size:24px;color:#4299e1">Current: {current_value} → Max: {max_value:.1f}</div></div>
<div class="risk">{risk}</div><div class="grid">"""
    
    for p in data:
        html += f"""<div class="pred"><div class="time">⏱️ {p['horizon']} min</div><div style="font-size:14px;color:#718096;margin-bottom:10px">{p['pred_time']}</div>
<div class="range">{p['pred_min']:.1f} ~ {p['pred_max']:.1f}<div class="range-label">예측 범위</div></div>
<div class="metric"><span>Change (MIN)</span><span style="font-weight:700">{p['change_min']:+.1f}</span></div>
<div class="metric"><span>Change (MAX)</span><span style="font-weight:700">{p['change_max']:+.1f}</span></div>
<span class="status status-{p['status'].lower()}">{p['status']}</span></div>"""
    
    html += f"""</div></div></body></html>"""
    return html

def main():
    print("="*60)
    print("V8 Numerical Dashboard")
    print("="*60)
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ 파일 없음: {DATA_PATH}")
        return
    
    try:
        df = pd.read_csv(DATA_PATH, on_bad_lines='skip', encoding='utf-8', low_memory=False)
        df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')
        df = df.dropna(subset=[TARGET_COL])
        print(f"✅ 로드: {len(df):,}행")
    except Exception as e:
        print(f"❌ 로드 실패: {e}")
        return
    
    try:
        results = predict_horizons(df)
    except Exception as e:
        print(f"❌ 예측 실패: {e}")
        import traceback
        traceback.print_exc()
        return
    
    if not results:
        print("❌ 예측 결과 없음!")
        return
    
    html = generate_html(results)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'V8_Numerical_{timestamp}.html'
    filepath = os.path.join(SCRIPT_DIR, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ 완료: {filename}")
    
    try:
        webbrowser.open('file://' + os.path.abspath(filepath))
    except:
        pass

if __name__ == '__main__':
    main()