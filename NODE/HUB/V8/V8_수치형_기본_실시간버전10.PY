# -*- coding: utf-8 -*-
"""
V8 실시간 예측 - 수치형 10분 (MAX만 사용)
"""

import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, 'model', 'xgboost_수치형_V8_10분.pkl')
data_dir = os.path.join(script_dir, 'data', 'HUBROOM_PIVOT_DATA.csv')

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

def realtime_prediction():
    try:
        with open(model_dir, 'rb') as f:
            model_dict = pickle.load(f)
        models = model_dict['models']  # [MIN, MAX, AVG] 3개 모델
    except:
        return None, "model is not found"
   
    try:
        df = pd.read_csv(data_dir, on_bad_lines='skip')
    except FileNotFoundError:
        return None, "file is not found"
    except Exception as e:
        return None, "exception occurred while reading file"
   
    recent_30 = df.tail(30).copy()
    TARGET_COL = 'CURRENT_M16A_3F_JOB_2'
   
    if TARGET_COL not in recent_30.columns:
        return None, "target column is not included"
   
    seq_target = recent_30[TARGET_COL].values
    current_value = seq_target[-1]
   
    if 'STAT_DT' in recent_30.columns:
        try:
            recent_30['STAT_DT'] = pd.to_datetime(recent_30['STAT_DT'].astype(str), format='%Y%m%d%H%M')
        except:
            base_time = datetime(2024, 1, 1, 0, 0)
            recent_30['STAT_DT'] = [base_time + timedelta(minutes=i) for i in range(len(recent_30))]
   
    # Feature 생성
    features = {
        'target_mean': np.mean(seq_target),
        'target_std': np.std(seq_target),
        'target_max': np.max(seq_target),
        'target_min': np.min(seq_target),
        'target_last_value': seq_target[-1],
        'target_last_5_mean': np.mean(seq_target[-5:]),
        'target_slope': np.polyfit(np.arange(30), seq_target, 1)[0],
    }
   
    features['target_acceleration'] = (seq_target[-5:].mean() - seq_target[-10:-5].mean()) / 5
    features['target_is_rising'] = 1 if seq_target[-1] > seq_target[-5] else 0
    features['target_rapid_rise'] = 1 if (seq_target[-1] - seq_target[-5] > 10) else 0
    features['target_last_10_mean'] = np.mean(seq_target[-10:])
   
    for group_name, cols in FEATURE_COLS.items():
        for col in cols:
            if col not in recent_30.columns:
                continue
           
            col_seq = recent_30[col].values
           
            if group_name == 'maxcapa':
                features[f'{col}_last_value'] = col_seq[-1]
           
            elif group_name in ['cmd', 'storage', 'fs_storage', 'hub', 'que', 
                               'target_alt', 'm16hub_que', 'm16a_que']:
                features[f'{col}_mean'] = np.mean(col_seq)
                features[f'{col}_std'] = np.std(col_seq)
                features[f'{col}_max'] = np.max(col_seq)
                features[f'{col}_min'] = np.min(col_seq)
                features[f'{col}_last_value'] = col_seq[-1]
                features[f'{col}_last_5_mean'] = np.mean(col_seq[-5:])
                features[f'{col}_slope'] = np.polyfit(np.arange(30), col_seq, 1)[0]
           
            else:  # inflow, outflow
                features[f'{col}_mean'] = np.mean(col_seq)
                features[f'{col}_last_value'] = col_seq[-1]
                features[f'{col}_slope'] = np.polyfit(np.arange(30), col_seq, 1)[0]
   
    if 'CD_M163FSTORAGEUSE' in recent_30.columns and 'CD_M163FSTORAGETOTAL' in recent_30.columns and 'CD_M163FSTORAGEUTIL' in recent_30.columns:
        use_seq = recent_30['CD_M163FSTORAGEUSE'].values
        total_seq = recent_30['CD_M163FSTORAGETOTAL'].values
        util_seq = recent_30['CD_M163FSTORAGEUTIL'].values
       
        features['storage_use_rate'] = (use_seq[-1] - use_seq[0]) / 30
        features['storage_remaining'] = total_seq[-1] - use_seq[-1]
        features['storage_util_last'] = util_seq[-1]
        features['storage_util_high'] = 1 if util_seq[-1] >= 7 else 0
        features['storage_util_critical'] = 1 if util_seq[-1] >= 10 else 0
   
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
   
    # Boolean features
    if 'M16HUB.QUE.ALL.CURRENTQCNT' in recent_30.columns:
        currentq = recent_30['M16HUB.QUE.ALL.CURRENTQCNT'].iloc[-1]
        features['currentq_high'] = 1 if currentq >= 1200 else 0
        features['currentq_critical'] = 1 if currentq >= 1400 else 0
    else:
        features['currentq_high'] = 0
        features['currentq_critical'] = 0
   
    if 'M16HUB.QUE.TIME.AVGTOTALTIME1MIN' in recent_30.columns:
        avgtime = recent_30['M16HUB.QUE.TIME.AVGTOTALTIME1MIN'].iloc[-1]
        features['avgtime1min_high'] = 1 if avgtime >= 4.0 else 0
        features['avgtime1min_critical'] = 1 if avgtime >= 4.5 else 0
    else:
        features['avgtime1min_high'] = 0
        features['avgtime1min_critical'] = 0
   
    if 'M16HUB.QUE.ALL.CURRENTQCNT' in recent_30.columns and 'M16HUB.QUE.TIME.AVGTOTALTIME1MIN' in recent_30.columns:
        currentq = recent_30['M16HUB.QUE.ALL.CURRENTQCNT'].iloc[-1]
        avgtime = recent_30['M16HUB.QUE.TIME.AVGTOTALTIME1MIN'].iloc[-1]
        features['que_severe_bottleneck'] = 1 if (currentq >= 1200 and avgtime >= 4.0) else 0
    else:
        features['que_severe_bottleneck'] = 0
   
    if 'M16HUB.QUE.OHT.OHTUTIL' in recent_30.columns:
        ohtutil = recent_30['M16HUB.QUE.OHT.OHTUTIL'].iloc[-1]
        features['m16hub_ohtutil_high'] = 1 if ohtutil >= 85.0 else 0
        features['m16hub_ohtutil_critical'] = 1 if ohtutil >= 90.0 else 0
    else:
        features['m16hub_ohtutil_high'] = 0
        features['m16hub_ohtutil_critical'] = 0
   
    if 'M16HUB.QUE.TIME.AVGTOTALTIME' in recent_30.columns:
        avgtime = recent_30['M16HUB.QUE.TIME.AVGTOTALTIME'].iloc[-1]
        features['m16hub_avgtime_high'] = 1 if avgtime >= 5.0 else 0
        features['m16hub_avgtime_critical'] = 1 if avgtime >= 6.0 else 0
    else:
        features['m16hub_avgtime_high'] = 0
        features['m16hub_avgtime_critical'] = 0
   
    if 'M16HUB.QUE.OHT.OHTUTIL' in recent_30.columns and 'M16HUB.QUE.TIME.AVGTOTALTIME' in recent_30.columns:
        ohtutil = recent_30['M16HUB.QUE.OHT.OHTUTIL'].iloc[-1]
        avgtime = recent_30['M16HUB.QUE.TIME.AVGTOTALTIME'].iloc[-1]
        features['m16hub_severe_bottleneck'] = 1 if (ohtutil >= 85.0 and avgtime >= 5.0) else 0
    else:
        features['m16hub_severe_bottleneck'] = 0
   
    if 'M16A.QUE.OHT.OHTUTIL' in recent_30.columns:
        ohtutil = recent_30['M16A.QUE.OHT.OHTUTIL'].iloc[-1]
        features['m16a_ohtutil_high'] = 1 if ohtutil >= 85.0 else 0
        features['m16a_ohtutil_critical'] = 1 if ohtutil >= 90.0 else 0
    else:
        features['m16a_ohtutil_high'] = 0
        features['m16a_ohtutil_critical'] = 0
   
    if 'M16A.QUE.LOAD.AVGLOADTIME1MIN' in recent_30.columns:
        loadtime = recent_30['M16A.QUE.LOAD.AVGLOADTIME1MIN'].iloc[-1]
        features['m16a_loadtime_high'] = 1 if loadtime >= 2.5 else 0
        features['m16a_loadtime_critical'] = 1 if loadtime >= 2.8 else 0
    else:
        features['m16a_loadtime_high'] = 0
        features['m16a_loadtime_critical'] = 0
   
    if 'M16A.QUE.ALL.TRANSPORT4MINOVERCNT' in recent_30.columns:
        transport4min = recent_30['M16A.QUE.ALL.TRANSPORT4MINOVERCNT'].iloc[-1]
        features['m16a_transport4min_high'] = 1 if transport4min >= 40 else 0
        features['m16a_transport4min_critical'] = 1 if transport4min >= 50 else 0
    else:
        features['m16a_transport4min_high'] = 0
        features['m16a_transport4min_critical'] = 0
   
    if 'M16A.QUE.ABN.QUETIMEDELAY' in recent_30.columns:
        delay = recent_30['M16A.QUE.ABN.QUETIMEDELAY'].iloc[-1]
        features['m16a_delay_warning'] = 1 if delay >= 1 else 0
        features['m16a_delay_critical'] = 1 if delay >= 3 else 0
    else:
        features['m16a_delay_warning'] = 0
        features['m16a_delay_critical'] = 0
   
    if 'M16A.QUE.OHT.OHTUTIL' in recent_30.columns and 'M16A.QUE.ALL.TRANSPORT4MINOVERCNT' in recent_30.columns:
        ohtutil = recent_30['M16A.QUE.OHT.OHTUTIL'].iloc[-1]
        transport4min = recent_30['M16A.QUE.ALL.TRANSPORT4MINOVERCNT'].iloc[-1]
        features['m16a_severe_bottleneck'] = 1 if (ohtutil >= 85.0 and transport4min >= 40) else 0
    else:
        features['m16a_severe_bottleneck'] = 0
   
    X_pred = pd.DataFrame([features])
    
    # MAX 모델 예측 (가장 중요한 최대값)
    pred_change_max = models[1].predict(X_pred)[0]
    prediction = current_value + pred_change_max
   
    # 상태 판정
    if prediction >= 300:
        status = "CRITICAL"
    elif prediction >= 280:
        status = "WARNING"
    elif prediction >= 270:
        status = "CAUTION"
    else:
        status = "NORMAL"
   
    return prediction, status

def predict_realtime():
    prediction, status_text = realtime_prediction()
   
    if prediction is None:
        row = {"EXCEPT": status_text}
        print([row])
    else:
        classification_label = int(prediction >= 300)  # 300 이상이면 1
        row = {
            "PREDICTVAL": int(prediction),
            "JUDGEVAL": classification_label,
            "MODEL": "V8_NUMERICAL_10MIN",
            "STATUS": status_text
        }
        print([row])

if __name__ == '__main__':
    predict_realtime()