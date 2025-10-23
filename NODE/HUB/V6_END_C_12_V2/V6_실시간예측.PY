# -*- coding: utf-8 -*-
"""
V6 실시간 예측 코드
- 30분 시퀀스 → 10분 후 예측
- Momentum Feature 4개 포함
- surge_imminent 포함
- 임계값: 280
"""

import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
import os

# 현 디렉터리를 기준으로 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, 'model', 'xgboost_완벽버전_가중치5배_임계값280_momentum.pkl')
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

TARGET_COL = 'CURRENT_M16A_3F_JOB_2'

def realtime_prediction():
    """
    실시간 예측 함수
    - 최근 30분 데이터로 10분 후 예측
    - V6 평가 코드의 Feature 생성 로직 사용
    """
    # 모델 로드
    try:
        with open(model_dir, 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        return None, "model is not found", None, None
    except Exception as e:
        return None, f"model load error: {str(e)}", None, None
   
    # 데이터 로드
    try:
        df = pd.read_csv(data_dir, on_bad_lines='skip', encoding='utf-8', low_memory=False)
    except FileNotFoundError:
        return None, "file is not found", None, None
    except Exception as e:
        # 다른 인코딩 시도
        try:
            df = pd.read_csv(data_dir, on_bad_lines='skip', encoding='cp949', low_memory=False)
        except:
            try:
                df = pd.read_csv(data_dir, on_bad_lines='skip', encoding='euc-kr', low_memory=False)
            except:
                return None, "exception occurred while reading file", None, None
   
    # 타겟 컬럼 확인 및 전처리
    if TARGET_COL not in df.columns:
        return None, "target column is not included", None, None
   
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')
    df = df.dropna(subset=[TARGET_COL])
   
    if len(df) < 30:
        return None, "insufficient data (need 30 rows)", None, None
   
    # 최근 30개 데이터 추출
    recent_30 = df.tail(30).copy()
    seq_target = recent_30[TARGET_COL].values
   
    # STAT_DT 처리
    if 'STAT_DT' in recent_30.columns:
        try:
            recent_30['STAT_DT'] = pd.to_datetime(recent_30['STAT_DT'].astype(str), format='%Y%m%d%H%M')
            seq_end = recent_30['STAT_DT'].iloc[-1]
        except:
            seq_end = datetime.now()
    else:
        seq_end = datetime.now()
   
    prediction_time = seq_end + timedelta(minutes=10)
   
    # ==================== Feature 생성 (V6_평가.PY와 동일) ====================
    
    features = {}
    
    # 1. 타겟 기본 통계
    features['target_mean'] = np.mean(seq_target)
    features['target_std'] = np.std(seq_target)
    features['target_max'] = np.max(seq_target)
    features['target_min'] = np.min(seq_target)
    features['target_last_value'] = seq_target[-1]
    features['target_last_5_mean'] = np.mean(seq_target[-5:])
    features['target_slope'] = np.polyfit(np.arange(30), seq_target, 1)[0]
    
    # 2. 🆕 Momentum Feature (V6 핵심!)
    features['target_acceleration'] = (seq_target[-5:].mean() - seq_target[-10:-5].mean()) / 5
    features['target_is_rising'] = 1 if seq_target[-1] > seq_target[-5] else 0
    features['target_rapid_rise'] = 1 if (seq_target[-1] - seq_target[-5] > 10) else 0
    features['target_last_10_mean'] = np.mean(seq_target[-10:])
    
    # 3. 각 컬럼 그룹별 Feature
    for group_name, cols in FEATURE_COLS.items():
        for col in cols:
            if col not in df.columns:
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
    
    # 4. FS Storage Feature
    if 'CD_M163FSTORAGEUSE' in df.columns and 'CD_M163FSTORAGETOTAL' in df.columns and 'CD_M163FSTORAGEUTIL' in df.columns:
        use_seq = recent_30['CD_M163FSTORAGEUSE'].values
        total_seq = recent_30['CD_M163FSTORAGETOTAL'].values
        util_seq = recent_30['CD_M163FSTORAGEUTIL'].values
        
        # 추가 Feature (V6_평가.PY에서 누락된 부분 보완)
        features['CD_M163FSTORAGEUSE_mean'] = np.mean(use_seq)
        features['CD_M163FSTORAGEUSE_std'] = np.std(use_seq)
        features['CD_M163FSTORAGEUSE_max'] = np.max(use_seq)
        features['CD_M163FSTORAGEUSE_min'] = np.min(use_seq)
        features['CD_M163FSTORAGEUSE_last_value'] = use_seq[-1]
        features['CD_M163FSTORAGEUSE_last_5_mean'] = np.mean(use_seq[-5:])
        features['CD_M163FSTORAGEUSE_slope'] = np.polyfit(np.arange(30), use_seq, 1)[0]
        
        features['CD_M163FSTORAGETOTAL_mean'] = np.mean(total_seq)
        features['CD_M163FSTORAGETOTAL_std'] = np.std(total_seq)
        features['CD_M163FSTORAGETOTAL_max'] = np.max(total_seq)
        features['CD_M163FSTORAGETOTAL_min'] = np.min(total_seq)
        features['CD_M163FSTORAGETOTAL_last_value'] = total_seq[-1]
        features['CD_M163FSTORAGETOTAL_last_5_mean'] = np.mean(total_seq[-5:])
        features['CD_M163FSTORAGETOTAL_slope'] = np.polyfit(np.arange(30), total_seq, 1)[0]
        
        features['CD_M163FSTORAGEUTIL_mean'] = np.mean(util_seq)
        features['CD_M163FSTORAGEUTIL_std'] = np.std(util_seq)
        features['CD_M163FSTORAGEUTIL_max'] = np.max(util_seq)
        features['CD_M163FSTORAGEUTIL_min'] = np.min(util_seq)
        features['CD_M163FSTORAGEUTIL_last_value'] = util_seq[-1]
        features['CD_M163FSTORAGEUTIL_last_5_mean'] = np.mean(util_seq[-5:])
        features['CD_M163FSTORAGEUTIL_slope'] = np.polyfit(np.arange(30), util_seq, 1)[0]
        
        features['storage_use_rate'] = (use_seq[-1] - use_seq[0]) / 30
        features['storage_remaining'] = total_seq[-1] - use_seq[-1]
        features['storage_util_last'] = util_seq[-1]
        features['storage_util_high'] = 1 if util_seq[-1] >= 7 else 0
        features['storage_util_critical'] = 1 if util_seq[-1] >= 10 else 0
    
    # 5. HUBROOMTOTAL Feature
    if 'HUBROOMTOTAL' in df.columns:
        hub_seq = recent_30['HUBROOMTOTAL'].values
        hub_last = hub_seq[-1]
        
        features['hub_critical'] = 1 if hub_last < 590 else 0
        features['hub_high'] = 1 if hub_last < 610 else 0
        features['hub_warning'] = 1 if hub_last < 620 else 0
        features['hub_decrease_rate'] = (hub_seq[0] - hub_last) / 30
        
        if 'CD_M163FSTORAGEUTIL' in df.columns:
            storage_util_last = recent_30['CD_M163FSTORAGEUTIL'].iloc[-1]
            features['hub_storage_risk'] = 1 if (hub_last < 610 and storage_util_last >= 7) else 0
    
    # 6. 유입-유출, CMD
    i = len(df) - 1  # 마지막 인덱스
    
    inflow_sum = sum(df[col].iloc[i] for col in FEATURE_COLS['inflow'] if col in df.columns)
    outflow_sum = sum(df[col].iloc[i] for col in FEATURE_COLS['outflow'] if col in df.columns)
    features['net_flow'] = inflow_sum - outflow_sum
    
    cmd_sum = sum(df[col].iloc[i] for col in FEATURE_COLS['cmd'] if col in df.columns)
    features['total_cmd'] = cmd_sum
    features['total_cmd_low'] = 1 if cmd_sum < 220 else 0
    features['total_cmd_very_low'] = 1 if cmd_sum < 200 else 0
    
    if 'HUBROOMTOTAL' in df.columns:
        hub_last = df['HUBROOMTOTAL'].iloc[i]
        features['hub_cmd_bottleneck'] = 1 if (hub_last < 610 and cmd_sum < 220) else 0
    
    if 'M16A_3F_STORAGE_UTIL' in df.columns:
        storage_util = df['M16A_3F_STORAGE_UTIL'].iloc[i]
        features['storage_util_critical'] = 1 if storage_util >= 205 else 0
        features['storage_util_high_risk'] = 1 if storage_util >= 207 else 0
    
    # 7. 🔥 급증 위험도 점수
    features['surge_risk_score'] = (
        features.get('hub_high', 0) * 3 +
        features.get('storage_util_critical', 0) * 2 +
        features.get('total_cmd_low', 0) * 1 +
        features.get('storage_util_high', 0) * 1
    )
    
    # 8. 🆕 급증 임박 신호 (평가 코드 버전 - 임계값 272로 완화)
    features['surge_imminent'] = 1 if (
        seq_target[-1] > 272 and
        features.get('target_acceleration', 0) > 0.2 and
        features.get('hub_high', 0) == 1
    ) else 0
    
    # ==================== 예측 실행 ====================
    
    X_pred = pd.DataFrame([features])
    prediction = model.predict(X_pred)[0]
    
    # 상태 결정 (임계값 280 기준)
    if prediction >= 300:
        status = "DANGEROUS"
    elif prediction >= 280:
        status = "WARNING"
    else:
        status = "NORMAL"
    
    # 시간 정보 포함하여 반환
    return prediction, status, seq_end, prediction_time

def predict_realtime():
    """
    실시간 예측 실행 및 결과 출력
    - 기존 포맷 준수
    - 현재시간, 예측시간 추가
    """
    result = realtime_prediction()
    
    if result[0] is None:
        # 예외 상황
        row = {"EXCEPT": result[1]}
        print([row])
    else:
        # 정상 예측
        prediction, status_text, current_time, pred_time = result
        classification_label = int(prediction >= 300)  # 300 이상이면 1 (DANGEROUS)
        row = {
            "CURRENT_TIME": current_time.strftime('%Y-%m-%d %H:%M'),
            "PREDICTION_TIME": pred_time.strftime('%Y-%m-%d %H:%M'),
            "PREDICTVAL": int(prediction),
            "JUDGEVAL": classification_label,
            "MODEL": "V6_HUBROOM_MODEL_MOMENTUM",
            "STATUS": status_text
        }
        print([row])

if __name__ == '__main__':
    predict_realtime()