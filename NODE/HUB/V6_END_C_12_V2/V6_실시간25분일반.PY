# -*- coding: utf-8 -*-
"""
V6 실시간 예측 코드 - 25분 버전
- 30분 시퀀스 → 25분 후 예측
- Momentum Features 포함
- 임계값 280
- 가중치 5배 학습 모델 사용
"""

import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
import os

# 현 디렉터리를 기준으로 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, 'model', 'xgboost_완벽버전_가중치5배_임계값280_momentum_25min.pkl')
data_dir = os.path.join(script_dir, 'data', 'HUBROOM_PIVOT_DATA.csv')

# V6 Feature 컬럼 정의
FEATURE_COLS = {
    'storage': ['M16A_3F_STORAGE_UTIL'],
    'fs_storage': ['CD_M163FSTORAGEUSE', 'CD_M163FSTORAGETOTAL', 'CD_M163FSTORAGEUTIL'],
    'hub': ['HUBROOMTOTAL'],
    'cmd': ['M16A_3F_CMD', 'M16A_6F_TO_HUB_CMD'],
    'inflow': ['M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2', 'M14A_3F_TO_HUB_JOB2'],
    'outflow': ['M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB', 'M16A_3F_TO_M14A_3F_JOB'],
    'maxcapa': ['M16A_6F_LFT_MAXCAPA', 'M16A_2F_LFT_MAXCAPA']
}

def realtime_prediction_25min():
    """V6 모델 기반 실시간 예측 - 25분 버전"""
    # 모델 로드
    try:
        with open(model_dir, 'rb') as f:
            model = pickle.load(f)
    except:
        return None, "model is not found"
   
    # 데이터 로드
    try:
        df = pd.read_csv(data_dir, on_bad_lines='skip')
    except FileNotFoundError:
        return None, "file is not found"
    except Exception as e:
        return None, "exception occurred while reading file"
   
    # 최근 30개 데이터
    recent_30 = df.tail(30).copy()
    TARGET_COL = 'CURRENT_M16A_3F_JOB_2'
   
    if TARGET_COL not in recent_30.columns:
        return None, "target column is not included"
   
    seq_target = recent_30[TARGET_COL].values
   
    # STAT_DT 처리
    if 'STAT_DT' in recent_30.columns:
        try:
            recent_30['STAT_DT'] = pd.to_datetime(recent_30['STAT_DT'].astype(str), format='%Y%m%d%H%M')
        except:
            base_time = datetime(2024, 1, 1, 0, 0)
            recent_30['STAT_DT'] = [base_time + timedelta(minutes=i) for i in range(len(recent_30))]
   
    seq_end = recent_30['STAT_DT'].iloc[-1]
    prediction_time = seq_end + timedelta(minutes=25)  # ✅ 25분 후
   
    # ===== Feature 생성 (V6와 동일) =====
    
    # 1. 타겟 기본 통계
    features = {
        'target_mean': np.mean(seq_target),
        'target_std': np.std(seq_target),
        'target_max': np.max(seq_target),
        'target_min': np.min(seq_target),
        'target_last_value': seq_target[-1],
        'target_last_5_mean': np.mean(seq_target[-5:]),
        'target_slope': np.polyfit(np.arange(30), seq_target, 1)[0],
    }
   
    # 2. 🆕 Momentum Features
    features['target_acceleration'] = (seq_target[-5:].mean() - seq_target[-10:-5].mean()) / 5
    features['target_is_rising'] = 1 if seq_target[-1] > seq_target[-5] else 0
    features['target_rapid_rise'] = 1 if (seq_target[-1] - seq_target[-5] > 10) else 0
    features['target_last_10_mean'] = np.mean(seq_target[-10:])
   
    # 3. 각 컬럼 그룹별 Features
    for group_name, cols in FEATURE_COLS.items():
        for col in cols:
            if col not in recent_30.columns:
                continue
           
            col_seq = recent_30[col].values
           
            if group_name == 'maxcapa':
                features[f'{col}_last_value'] = col_seq[-1]
           
            elif group_name in ['cmd', 'storage', 'fs_storage', 'hub']:
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
   
    # 4. FS Storage 특수 Features
    if 'CD_M163FSTORAGEUSE' in recent_30.columns and 'CD_M163FSTORAGETOTAL' in recent_30.columns and 'CD_M163FSTORAGEUTIL' in recent_30.columns:
        use_seq = recent_30['CD_M163FSTORAGEUSE'].values
        total_seq = recent_30['CD_M163FSTORAGETOTAL'].values
        util_seq = recent_30['CD_M163FSTORAGEUTIL'].values
       
        features['storage_use_rate'] = (use_seq[-1] - use_seq[0]) / 30
        features['storage_remaining'] = total_seq[-1] - use_seq[-1]
        features['storage_util_last'] = util_seq[-1]
        features['storage_util_high'] = 1 if util_seq[-1] >= 7 else 0
        features['storage_util_critical'] = 1 if util_seq[-1] >= 10 else 0
   
    # 5. HUBROOMTOTAL 특수 Features
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
   
    # 6. 유입-유출, CMD
    inflow_sum = sum(recent_30[col].iloc[-1] for col in FEATURE_COLS['inflow'] if col in recent_30.columns)
    outflow_sum = sum(recent_30[col].iloc[-1] for col in FEATURE_COLS['outflow'] if col in recent_30.columns)
    features['net_flow'] = inflow_sum - outflow_sum
   
    cmd_sum = sum(recent_30[col].iloc[-1] for col in FEATURE_COLS['cmd'] if col in recent_30.columns)
    features['total_cmd'] = cmd_sum
    features['total_cmd_low'] = 1 if cmd_sum < 220 else 0
    features['total_cmd_very_low'] = 1 if cmd_sum < 200 else 0
   
    # 7. HUB × CMD 복합
    if 'HUBROOMTOTAL' in recent_30.columns:
        hub_last = recent_30['HUBROOMTOTAL'].iloc[-1]
        features['hub_cmd_bottleneck'] = 1 if (hub_last < 610 and cmd_sum < 220) else 0
   
    # 8. Storage Util 위험
    if 'M16A_3F_STORAGE_UTIL' in recent_30.columns:
        storage_util = recent_30['M16A_3F_STORAGE_UTIL'].iloc[-1]
        features['storage_util_critical'] = 1 if storage_util >= 205 else 0
        features['storage_util_high_risk'] = 1 if storage_util >= 207 else 0
   
    # 9. 🔥 급증 위험도 점수
    features['surge_risk_score'] = (
        features.get('hub_high', 0) * 3 +
        features.get('storage_util_critical', 0) * 2 +
        features.get('total_cmd_low', 0) * 1 +
        features.get('storage_util_high', 0) * 1
    )
   
    # 10. 🆕 급증 임박 신호
    features['surge_imminent'] = 1 if (
        seq_target[-1] > 272 and
        features.get('target_acceleration', 0) > 0.2 and
        features.get('hub_high', 0) == 1
    ) else 0
   
    X_pred = pd.DataFrame([features])
   
    # 예측
    prediction = model.predict(X_pred)[0]
   
    # 상태 결정
    if prediction >= 300:
        status = "CRITICAL"
    elif prediction >= 280:
        status = "WARNING"
    elif prediction >= 270:
        status = "CAUTION"
    else:
        status = "NORMAL"
   
    return prediction, status

def predict_realtime_25min():
    """실시간 예측 실행 및 출력 - 25분"""
    prediction, status_text = realtime_prediction_25min()
   
    if prediction is None:
        row = {"EXCEPT": status_text}
        print([row])
    else:
        surge_warning = int(prediction >= 280)
        extreme_warning = int(prediction >= 300)
       
        row = {
            "PREDICTVAL": round(prediction, 2),
            "SURGE_WARNING": surge_warning,
            "EXTREME_WARNING": extreme_warning,
            "MODEL": "V6_MOMENTUM_280_25MIN",  # ✅ 25분 모델
            "STATUS": status_text,
            "PREDICTION_HORIZON": "25MIN",  # ✅ 25분 예측
            "THRESHOLD_280": "V6_PRIMARY"
        }
        print([row])

if __name__ == '__main__':
    predict_realtime_25min()