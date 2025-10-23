# -*- coding: utf-8 -*-
"""
V6 실시간 예측 시스템
- 30분 시퀀스 → 10분 후 예측
- Momentum Feature 포함
- 300 이상 달성 확률 계산
- 상승 원인 분석
"""

import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
import os

# 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'model', 'xgboost_완벽버전_가중치5배_임계값280_momentum.pkl')
data_path = os.path.join(script_dir, 'data', 'HUBROOM_PIVOT_DATA.csv')

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

TARGET_COL = 'CURRENT_M16A_3F_JOB_2'

def calculate_exceed_300_probability(prediction, features):
    """
    10분 내 300 이상 달성 확률 계산
    """
    # 1. 예측값 기반 확률
    if prediction >= 310:
        base_prob = 95
    elif prediction >= 300:
        base_prob = 85
    elif prediction >= 290:
        base_prob = 60
    elif prediction >= 280:
        base_prob = 35
    elif prediction >= 270:
        base_prob = 20
    else:
        base_prob = 5
    
    # 2. Momentum 보정
    momentum_boost = 0
    if features.get('surge_imminent', 0) == 1:
        momentum_boost += 15
    if features.get('target_acceleration', 0) > 1.0:
        momentum_boost += 10
    elif features.get('target_acceleration', 0) > 0.5:
        momentum_boost += 5
    
    # 3. 위험 요인 보정
    risk_boost = 0
    if features.get('hub_high', 0) == 1:
        risk_boost += 10
    if features.get('storage_util_critical', 0) == 1:
        risk_boost += 5
    if features.get('total_cmd_low', 0) == 1:
        risk_boost += 5
    
    # 4. 최종 확률 (0~100% 범위)
    final_prob = min(100, max(0, base_prob + momentum_boost + risk_boost))
    
    return f"{final_prob}%"

def analyze_surge_cause(features, seq_target):
    """
    상승 원인 분석
    """
    causes = []
    
    # HUB 병목
    hub_last = features.get('HUBROOMTOTAL_last_value', 650)
    if hub_last < 590:
        causes.append(f"HUB심각병목({int(hub_last)})")
    elif hub_last < 610:
        causes.append(f"HUB병목({int(hub_last)})")
    
    # 가속도
    acceleration = features.get('target_acceleration', 0)
    if acceleration > 1.0:
        causes.append(f"매우높은가속도({acceleration:.1f})")
    elif acceleration > 0.5:
        causes.append(f"높은가속도({acceleration:.1f})")
    
    # CMD 부족
    total_cmd = features.get('total_cmd', 300)
    if total_cmd < 200:
        causes.append(f"CMD심각부족({int(total_cmd)})")
    elif total_cmd < 220:
        causes.append(f"CMD부족({int(total_cmd)})")
    
    # 스토리지 과부하
    storage_util = features.get('M16A_3F_STORAGE_UTIL_last_value', 0)
    if storage_util >= 207:
        causes.append(f"스토리지심각({int(storage_util)})")
    elif storage_util >= 205:
        causes.append(f"스토리지과부하({int(storage_util)})")
    
    # FS 저장공간 부족
    fs_util = features.get('CD_M163FSTORAGEUTIL_last_value', 0)
    if fs_util >= 10:
        causes.append(f"FS심각부족({fs_util:.1f})")
    elif fs_util >= 7:
        causes.append(f"FS부족({fs_util:.1f})")
    
    # 급등 패턴
    if features.get('target_rapid_rise', 0) == 1:
        causes.append("5분급등패턴")
    
    # 급증 임박
    if features.get('surge_imminent', 0) == 1:
        causes.append("급증임박신호")
    
    # 원인이 없으면
    if not causes:
        # 현재값 확인
        current_value = seq_target[-1]
        if current_value < 250:
            causes.append("정상범위")
        else:
            causes.append("완만한상승추세")
    
    return " + ".join(causes)

def create_features_realtime(df_30):
    """
    실시간 Feature 생성 (학습 코드와 동일)
    """
    seq_target = df_30[TARGET_COL].values
    
    # 타겟 기본 통계
    features = {
        'target_mean': np.mean(seq_target),
        'target_std': np.std(seq_target),
        'target_max': np.max(seq_target),
        'target_min': np.min(seq_target),
        'target_last_value': seq_target[-1],
        'target_last_5_mean': np.mean(seq_target[-5:]),
        'target_slope': np.polyfit(np.arange(30), seq_target, 1)[0],
    }
    
    # 🆕 Momentum Feature 추가
    features['target_acceleration'] = (seq_target[-5:].mean() - seq_target[-10:-5].mean()) / 5
    features['target_is_rising'] = 1 if seq_target[-1] > seq_target[-5] else 0
    features['target_rapid_rise'] = 1 if (seq_target[-1] - seq_target[-5] > 10) else 0
    features['target_last_10_mean'] = np.mean(seq_target[-10:])
    
    # 각 컬럼 그룹별 Feature
    for group_name, cols in FEATURE_COLS.items():
        for col in cols:
            if col not in df_30.columns:
                continue
            
            col_seq = df_30[col].values
            
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
    
    # FS Storage Feature
    if 'CD_M163FSTORAGEUSE' in df_30.columns and 'CD_M163FSTORAGETOTAL' in df_30.columns:
        storage_use = df_30['CD_M163FSTORAGEUSE'].values
        storage_total = df_30['CD_M163FSTORAGETOTAL'].values
        storage_util = df_30['CD_M163FSTORAGEUTIL'].values
        
        features['storage_use_rate'] = (storage_use[-1] - storage_use[0]) / 30
        features['storage_remaining'] = storage_total[-1] - storage_use[-1]
        features['storage_util_last'] = storage_util[-1]
        features['storage_util_high'] = 1 if storage_util[-1] >= 7 else 0
        features['storage_util_critical'] = 1 if storage_util[-1] >= 10 else 0
    
    # HUBROOMTOTAL Feature
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
    
    # 유입-유출, CMD
    inflow_sum = sum(df_30[col].iloc[-1] for col in FEATURE_COLS['inflow'] if col in df_30.columns)
    outflow_sum = sum(df_30[col].iloc[-1] for col in FEATURE_COLS['outflow'] if col in df_30.columns)
    features['net_flow'] = inflow_sum - outflow_sum
    
    cmd_sum = sum(df_30[col].iloc[-1] for col in FEATURE_COLS['cmd'] if col in df_30.columns)
    features['total_cmd'] = cmd_sum
    features['total_cmd_low'] = 1 if cmd_sum < 220 else 0
    features['total_cmd_very_low'] = 1 if cmd_sum < 200 else 0
    
    # HUB × CMD 복합
    if 'HUBROOMTOTAL' in df_30.columns:
        hub_last = df_30['HUBROOMTOTAL'].iloc[-1]
        features['hub_cmd_bottleneck'] = 1 if (hub_last < 610 and cmd_sum < 220) else 0
    
    # Storage Util 위험
    if 'M16A_3F_STORAGE_UTIL' in df_30.columns:
        storage_util = df_30['M16A_3F_STORAGE_UTIL'].iloc[-1]
        features['storage_util_critical'] = 1 if storage_util >= 205 else 0
        features['storage_util_high_risk'] = 1 if storage_util >= 207 else 0
    
    # 🔥 급증 위험도 점수
    features['surge_risk_score'] = (
        features.get('hub_high', 0) * 3 +
        features.get('storage_util_critical', 0) * 2 +
        features.get('total_cmd_low', 0) * 1 +
        features.get('storage_util_high', 0) * 1
    )
    
    # 🆕 급증 임박 신호
    features['surge_imminent'] = 1 if (
        seq_target[-1] > 272 and
        features.get('target_acceleration', 0) > 0.2 and
        features.get('hub_high', 0) == 1
    ) else 0
    
    return features, seq_target

def realtime_prediction():
    """
    실시간 예측 메인 함수
    """
    # 모델 로드
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        return None, "MODEL_FILE_NOT_FOUND"
    except Exception as e:
        return None, f"MODEL_LOAD_ERROR: {str(e)}"
    
    # 데이터 로드
    try:
        df = pd.read_csv(data_path, on_bad_lines='skip', encoding='utf-8', low_memory=False)
    except FileNotFoundError:
        return None, "DATA_FILE_NOT_FOUND"
    except Exception as e:
        return None, f"DATA_LOAD_ERROR: {str(e)}"
    
    # 타겟 컬럼 확인
    if TARGET_COL not in df.columns:
        return None, "TARGET_COLUMN_NOT_FOUND"
    
    # 최근 30개 데이터 추출
    if len(df) < 30:
        return None, "INSUFFICIENT_DATA"
    
    df_30 = df.tail(30).copy()
    
    # 타겟 컬럼 전처리
    df_30[TARGET_COL] = pd.to_numeric(df_30[TARGET_COL], errors='coerce')
    df_30 = df_30.dropna(subset=[TARGET_COL])
    
    if len(df_30) < 30:
        return None, "INSUFFICIENT_VALID_DATA"
    
    # STAT_DT 처리
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
    prediction_time = current_time + timedelta(minutes=10)
    
    # Feature 생성
    features, seq_target = create_features_realtime(df_30)
    
    # DataFrame 변환
    X_pred = pd.DataFrame([features])
    
    # 예측
    prediction = model.predict(X_pred)[0]
    current_value = seq_target[-1]
    
    # 상승률 계산
    increase_rate = (prediction - current_value) / current_value * 100
    
    # STATUS 결정
    if prediction >= 300:
        status = "DANGEROUS"
    elif prediction >= 280:
        status = "WARNING"
    else:
        status = "NORMAL"
    
    # SURGE_RISK 결정
    risk_score = features.get('surge_risk_score', 0)
    if risk_score >= 5:
        surge_risk = "CRITICAL"
    elif risk_score >= 3:
        surge_risk = "HIGH"
    elif risk_score >= 1:
        surge_risk = "MEDIUM"
    else:
        surge_risk = "LOW"
    
    # 300 이상 달성 확률
    exceed_300_prob = calculate_exceed_300_probability(prediction, features)
    
    # 상승 원인 분석
    cause_analysis = analyze_surge_cause(features, seq_target)
    
    # 결과 생성
    result = {
        "CURRENT_VALUE": round(current_value, 1),
        "PREDICTED_VALUE": round(prediction, 1),
        "INCREASE_RATE": f"{increase_rate:+.1f}%",
        "EXCEED_300_PROBABILITY": exceed_300_prob,
        "STATUS": status,
        "SURGE_RISK": surge_risk,
        "CAUSE_ANALYSIS": cause_analysis,
        "MOMENTUM_SIGNAL": {
            "acceleration": round(features.get('target_acceleration', 0), 2),
            "is_rising": bool(features.get('target_is_rising', 0)),
            "rapid_rise": bool(features.get('target_rapid_rise', 0)),
            "surge_imminent": bool(features.get('surge_imminent', 0))
        },
        "KEY_FACTORS": {
            "HUBROOMTOTAL": int(features.get('HUBROOMTOTAL_last_value', 0)),
            "FS_STORAGE_UTIL": round(features.get('CD_M163FSTORAGEUTIL_last_value', 0), 1),
            "TOTAL_CMD": int(features.get('total_cmd', 0)),
            "STORAGE_UTIL": int(features.get('M16A_3F_STORAGE_UTIL_last_value', 0))
        },
        "PREDICTION_TIME": prediction_time.strftime('%Y-%m-%d %H:%M:%S'),
        "MODEL_VERSION": "V6_MOMENTUM_W5X_T280"
    }
    
    return result, None

def predict_realtime():
    """
    실시간 예측 실행 및 출력
    """
    result, error = realtime_prediction()
    
    if error:
        print([{"ERROR": error}])
    else:
        print([result])

if __name__ == '__main__':
    predict_realtime()