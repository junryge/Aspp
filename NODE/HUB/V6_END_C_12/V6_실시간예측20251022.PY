# -*- coding: utf-8 -*-
"""
V6 실시간 예측 코드 + 급증 위험도 기반 보정
- 최근 30개 데이터로 10분 후 예측
- 보정 전/후 예측값 모두 반환
- 위험도 평가 포함
"""

import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
import os

# 현 디렉터리를 기준으로 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, 'model', 'xgboost_model_30min_10min.pkl')
data_dir = os.path.join(script_dir, 'data', 'HUBROOM_PIVOT_DATA.csv')

# V6 필수 컬럼 정의 (12개 핵심 컬럼)
FEATURE_COLS = {
    'storage': ['M16A_3F_STORAGE_UTIL'],
    'cmd': ['M16A_3F_CMD', 'M16A_6F_TO_HUB_CMD'],
    'inflow': ['M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2', 'M14A_3F_TO_HUB_JOB2'],
    'outflow': ['M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB', 'M16A_3F_TO_M14A_3F_JOB'],
    'maxcapa': ['M16A_6F_LFT_MAXCAPA', 'M16A_2F_LFT_MAXCAPA']
}

TARGET_COL = 'CURRENT_M16A_3F_JOB_2'


def calculate_surge_risk(features):
    """급증 위험도 및 예상 급증량 계산"""
    cmd_last = features.get('M16A_3F_CMD_last_value', 999)
    storage_last = features.get('M16A_3F_STORAGE_UTIL_last_value', 0)
    cmd_slope = features.get('M16A_3F_CMD_slope', 0)
    
    # 위험도 판정 (분석 결과 기반)
    if cmd_last < 180:
        risk_level = "HIGH"
        surge_prob = 0.70
        expected_surge = 60
    elif storage_last >= 30 and storage_last <= 40:
        risk_level = "HIGH"
        surge_prob = 0.45
        expected_surge = 50
    elif cmd_last >= 260 and cmd_last <= 280:
        risk_level = "HIGH"
        surge_prob = 0.67
        expected_surge = 60
    elif cmd_last < 220:
        risk_level = "MEDIUM"
        surge_prob = 0.30
        expected_surge = 35
    else:
        risk_level = "LOW"
        surge_prob = 0.05
        expected_surge = 20
    
    # 추가 보정 요소
    if cmd_slope < -5:
        surge_prob *= 1.2
        expected_surge *= 1.1
    
    storage_slope = features.get('M16A_3F_STORAGE_UTIL_slope', 0)
    if storage_slope > 2:
        surge_prob *= 1.1
    
    surge_prob = min(surge_prob, 0.85)
    
    return risk_level, surge_prob, expected_surge


def apply_adaptive_correction(prediction, features, seq_target):
    """적응형 예측값 보정"""
    risk_level, surge_prob, expected_surge = calculate_surge_risk(features)
    current_value = seq_target[-1]
    seq_mean = np.mean(seq_target)
    seq_max = np.max(seq_target)
    recent_mean = np.mean(seq_target[-5:])
    
    correction = 0
    
    # 1. 기본 보정: 위험도에 따른 급증 반영
    if current_value < 300:
        if risk_level == "HIGH":
            if prediction < 300:
                potential_value = current_value + expected_surge * surge_prob
                if potential_value >= 300:
                    correction = (potential_value - prediction) * 0.7
                else:
                    correction = expected_surge * surge_prob * 0.5
        elif risk_level == "MEDIUM":
            if prediction < 300 and surge_prob > 0.25:
                correction = expected_surge * surge_prob * 0.3
        else:
            correction = expected_surge * surge_prob * 0.1
    
    # 2. 추가 보정: 시퀀스 패턴 반영
    recent_increase = recent_mean - seq_mean
    if recent_increase > 10:
        correction += recent_increase * 0.2
    
    if seq_max >= 300:
        correction += 5
    
    # 3. 보정값 제한
    corrected_prediction = prediction + correction
    corrected_prediction = max(corrected_prediction, current_value - 50)
    corrected_prediction = min(corrected_prediction, current_value + 100)
    
    if current_value < 250 and corrected_prediction > 350:
        corrected_prediction = 320
    
    return corrected_prediction, risk_level, surge_prob, correction


def realtime_prediction():
    """실시간 예측 함수 (보정 포함)"""
    
    # 모델 로드
    try:
        with open(model_dir, 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        return None, "model is not found"
    except Exception as e:
        return None, f"model load error: {str(e)}"
    
    # 데이터 로드
    try:
        df = pd.read_csv(data_dir, on_bad_lines='skip', encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(data_dir, on_bad_lines='skip', encoding='cp949')
        except:
            df = pd.read_csv(data_dir, on_bad_lines='skip', encoding='euc-kr')
    except FileNotFoundError:
        return None, "file is not found"
    except Exception as e:
        return None, f"file read error: {str(e)}"
    
    # 최근 30개 데이터 확인
    if len(df) < 30:
        return None, "insufficient data (need at least 30 rows)"
    
    recent_30 = df.tail(30).copy()
    
    if TARGET_COL not in recent_30.columns:
        return None, "target column is not included"
    
    seq_target = recent_30[TARGET_COL].values
    
    # NaN 체크
    if np.any(np.isnan(seq_target)) or np.any(np.isinf(seq_target)):
        return None, "target column contains NaN or Inf"
    
    # STAT_DT 처리
    if 'STAT_DT' in recent_30.columns:
        try:
            recent_30['STAT_DT'] = pd.to_datetime(recent_30['STAT_DT'].astype(str), format='%Y%m%d%H%M')
        except:
            base_time = datetime.now()
            recent_30['STAT_DT'] = [base_time - timedelta(minutes=(29-i)) for i in range(30)]
    else:
        base_time = datetime.now()
        recent_30['STAT_DT'] = [base_time - timedelta(minutes=(29-i)) for i in range(30)]
    
    seq_end = recent_30['STAT_DT'].iloc[-1]
    prediction_time = seq_end + timedelta(minutes=10)
    current_value = seq_target[-1]
    
    # Feature 생성 - 타겟 컬럼 기본 특성
    features = {
        'target_mean': np.mean(seq_target),
        'target_std': np.std(seq_target),
        'target_last_5_mean': np.mean(seq_target[-5:]),
        'target_max': np.max(seq_target),
        'target_min': np.min(seq_target),
        'target_slope': np.polyfit(np.arange(30), seq_target, 1)[0],
        'target_last_10_mean': np.mean(seq_target[-10:]),
        'target_first_10_mean': np.mean(seq_target[:10])
    }
    
    # 각 컬럼 그룹별 특성 추가 (12개 핵심 컬럼)
    for group_name, cols in FEATURE_COLS.items():
        for col in cols:
            if col in recent_30.columns:
                col_seq = recent_30[col].values
                
                # 기본 통계
                features[f'{col}_mean'] = np.mean(col_seq)
                features[f'{col}_std'] = np.std(col_seq)
                features[f'{col}_max'] = np.max(col_seq)
                features[f'{col}_min'] = np.min(col_seq)
                
                # 최근 특성
                features[f'{col}_last_5_mean'] = np.mean(col_seq[-5:])
                features[f'{col}_last_10_mean'] = np.mean(col_seq[-10:])
                
                # 추세
                features[f'{col}_slope'] = np.polyfit(np.arange(30), col_seq, 1)[0]
                
                # 구간별 평균
                features[f'{col}_first_10_mean'] = np.mean(col_seq[:10])
                features[f'{col}_mid_10_mean'] = np.mean(col_seq[10:20])
                features[f'{col}_last_value'] = col_seq[-1]
    
    # 유입-유출 차이 (Net Flow)
    inflow_sum = 0
    outflow_sum = 0
    for col in FEATURE_COLS['inflow']:
        if col in recent_30.columns:
            inflow_sum += recent_30[col].iloc[-1]
    for col in FEATURE_COLS['outflow']:
        if col in recent_30.columns:
            outflow_sum += recent_30[col].iloc[-1]
    features['net_flow'] = inflow_sum - outflow_sum
    
    # CMD 총합
    cmd_sum = 0
    for col in FEATURE_COLS['cmd']:
        if col in recent_30.columns:
            cmd_sum += recent_30[col].iloc[-1]
    features['total_cmd'] = cmd_sum
    
    # 데이터프레임 생성
    X_pred = pd.DataFrame([features])
    
    # 기본 예측
    prediction_original = model.predict(X_pred)[0]
    
    # 보정된 예측
    prediction_corrected, risk_level, surge_prob, correction_amount = \
        apply_adaptive_correction(prediction_original, features, seq_target)
    
    # 상태 결정 (보정 후 예측값 기준)
    if prediction_corrected >= 300:
        status = "DANGEROUS"
    elif prediction_corrected >= 280:
        status = "WARNING"
    else:
        status = "NORMAL"
    
    # 결과 반환
    result = {
        'current_time': seq_end.strftime('%Y-%m-%d %H:%M'),
        'prediction_time': prediction_time.strftime('%Y-%m-%d %H:%M'),
        'current_value': round(current_value, 2),
        'prediction_original': round(prediction_original, 2),
        'prediction_corrected': round(prediction_corrected, 2),
        'correction_amount': round(correction_amount, 2),
        'risk_level': risk_level,
        'surge_probability': f"{surge_prob*100:.0f}%",
        'status': status,
        'cmd_last': round(features.get('M16A_3F_CMD_last_value', 0), 2),
        'storage_last': round(features.get('M16A_3F_STORAGE_UTIL_last_value', 0), 2)
    }
    
    return result, None


def predict_realtime():
    """실시간 예측 실행 및 결과 출력"""
    result, error = realtime_prediction()
    
    if error:
        row = {"EXCEPT": error}
        print([row])
    else:
        # 운영 환경 출력 형식 (보정된 예측값 사용)
        classification_label = int(result['prediction_corrected'] >= 300)
        row = {
            "PREDICTVAL": int(result['prediction_corrected']),
            "PREDICTVAL_ORIGINAL": int(result['prediction_original']),
            "CORRECTION": int(result['correction_amount']),
            "JUDGEVAL": classification_label,
            "RISK_LEVEL": result['risk_level'],
            "SURGE_PROB": result['surge_probability'],
            "CURRENT_VALUE": int(result['current_value']),
            "MODEL": "V6_HUBROOM_MODEL_C12_WITH_CORRECTION",
            "STATUS": result['status'],
            "PREDICTION_TIME": result['prediction_time']
        }
        print([row])


def predict_realtime_detail():
    """상세 정보 포함 실시간 예측 (디버깅/모니터링용)"""
    result, error = realtime_prediction()
    
    if error:
        print(f"❌ 에러 발생: {error}")
        return
    
    print("="*80)
    print("🚀 V6 실시간 예측 결과 (급증 위험도 보정 포함)")
    print("="*80)
    print(f"\n📅 현재 시간: {result['current_time']}")
    print(f"📅 예측 시점: {result['prediction_time']}")
    print(f"\n현재값: {result['current_value']}")
    print(f"\n[보정 전] 예측값: {result['prediction_original']}")
    print(f"[보정 후] 예측값: {result['prediction_corrected']}")
    print(f"보정량: +{result['correction_amount']}")
    print(f"\n⚠️ 위험도: {result['risk_level']}")
    print(f"📊 급증 확률: {result['surge_probability']}")
    print(f"🚦 상태: {result['status']}")
    print(f"\n주요 지표:")
    print(f"  - M16A_3F_CMD: {result['cmd_last']}")
    print(f"  - M16A_3F_STORAGE_UTIL: {result['storage_last']}")
    print("="*80)


if __name__ == '__main__':
    # 운영 환경: 간단한 출력
    predict_realtime()
    
    # 디버깅/모니터링: 상세 출력 (필요시 주석 해제)
    # print("\n")
    # predict_realtime_detail()