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
import webbrowser  # 브라우저 자동 열기용

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
    prediction_start = current_time + timedelta(minutes=1)
    prediction_end = current_time + timedelta(minutes=10)
    
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
        "PREDICTED_VALUE": int(round(prediction)),  # 정수로 변경!
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
        "PREDICTION_TIME": f"{prediction_start.strftime('%Y-%m-%d %H:%M:%S')}~{prediction_end.strftime('%Y-%m-%d %H:%M:%S')}",
        "MODEL_VERSION": "V6_MOMENTUM_W5X_T280"
    }
    
    return result, None

def generate_html_dashboard(result):
    """
    예측 결과를 HTML 대시보드로 생성
    """
    # 확률에서 % 제거하고 숫자만 추출
    prob_value = int(result['EXCEED_300_PROBABILITY'].replace('%', ''))
    
    # 상승률에서 % 제거하고 숫자만 추출
    increase_rate_text = result['INCREASE_RATE']
    
    # 원인 분석을 항목별로 분리
    causes = result['CAUSE_ANALYSIS'].split(' + ')
    cause_items_html = ""
    for cause in causes:
        if any(word in cause for word in ['심각', '매우높은', '급증임박']):
            icon_class = "cause-icon-critical"
            icon = "🔴"
        else:
            icon_class = "cause-icon-warning"
            icon = "⚠️"
        
        cause_items_html += f"""
                <div class="cause-item">
                    <div class="cause-icon {icon_class}">{icon}</div>
                    <div class="cause-text">{cause}</div>
                </div>
        """
    
    # STATUS에 따른 클래스 결정
    status_class = f"status-{result['STATUS'].lower()}"
    
    # SURGE_RISK에 따른 클래스 결정
    risk_class = f"risk-{result['SURGE_RISK'].lower()}"
    
    # 확률에 따른 조치 상황
    if prob_value >= 80:
        action_level = "⚠️ 즉시 대응 필요"
        action_desc = "매우 높은 확률로 300 초과 예상"
        action_class = "action-needed"
    elif prob_value >= 60:
        action_level = "⚡ 사전 준비 필요"
        action_desc = "높은 확률로 발생 가능"
        action_class = "action-needed"
    else:
        action_level = "📊 모니터링 강화"
        action_desc = "주의 관찰 필요"
        action_class = "action-needed"
    
    # Momentum 체크 표시
    momentum_checks = {
        'acceleration': '✅' if result['MOMENTUM_SIGNAL']['acceleration'] > 0.5 else '❌',
        'is_rising': '✅' if result['MOMENTUM_SIGNAL']['is_rising'] else '❌',
        'rapid_rise': '✅' if result['MOMENTUM_SIGNAL']['rapid_rise'] else '❌',
        'surge_imminent': '✅' if result['MOMENTUM_SIGNAL']['surge_imminent'] else '❌'
    }
    
    # KEY_FACTORS 색상 결정
    hub_class = "factor-critical" if result['KEY_FACTORS']['HUBROOMTOTAL'] < 610 else ""
    fs_class = "factor-warning" if result['KEY_FACTORS']['FS_STORAGE_UTIL'] >= 7 else ""
    storage_class = "factor-warning" if result['KEY_FACTORS']['STORAGE_UTIL'] >= 205 else ""
    
    # 예측 시간 범위 분리
    time_parts = result['PREDICTION_TIME'].split('~')
    pred_start = time_parts[0] if len(time_parts) > 0 else ""
    pred_end = time_parts[1] if len(time_parts) > 1 else ""
    
    # 현재 시간 (예측 시작 시간 - 1분)
    current_time_str = pred_start.split(' ')[1] if ' ' in pred_start else "00:00:00"
    
    html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>V6 실시간 급증 예측 대시보드</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .header {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        
        .header h1 {{
            font-size: 32px;
            color: #2d3748;
            margin-bottom: 10px;
        }}
        
        .header .subtitle {{
            color: #718096;
            font-size: 16px;
        }}
        
        .model-info {{
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
            margin-top: 10px;
        }}
        
        .timestamp {{
            float: right;
            color: #a0aec0;
            font-size: 14px;
        }}
        
        .main-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        
        .card {{
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        
        .card-title {{
            font-size: 14px;
            color: #718096;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 15px;
            font-weight: 600;
        }}
        
        .big-number {{
            font-size: 48px;
            font-weight: 700;
            margin-bottom: 10px;
        }}
        
        .current-value {{
            color: #4299e1;
        }}
        
        .predicted-value {{
            color: #f56565;
        }}
        
        .increase-rate {{
            color: #ed8936;
        }}
        
        .sub-info {{
            font-size: 14px;
            color: #a0aec0;
            margin-top: 5px;
        }}
        
        .status-badge {{
            display: inline-block;
            padding: 8px 20px;
            border-radius: 25px;
            font-weight: 700;
            font-size: 16px;
            margin-top: 10px;
        }}
        
        .status-normal {{
            background: #c6f6d5;
            color: #2f855a;
        }}
        
        .status-warning {{
            background: #feebc8;
            color: #c05621;
        }}
        
        .status-dangerous {{
            background: #fed7d7;
            color: #c53030;
        }}
        
        .probability-section {{
            text-align: center;
        }}
        
        .probability-circle {{
            width: 180px;
            height: 180px;
            border-radius: 50%;
            background: conic-gradient(#f56565 0% {prob_value}%, #e2e8f0 {prob_value}% 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 20px auto;
            position: relative;
            box-shadow: 0 5px 15px rgba(245, 101, 101, 0.3);
        }}
        
        .probability-inner {{
            width: 140px;
            height: 140px;
            border-radius: 50%;
            background: white;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
        }}
        
        .probability-number {{
            font-size: 48px;
            font-weight: 700;
            color: #f56565;
        }}
        
        .probability-label {{
            font-size: 12px;
            color: #718096;
            margin-top: 5px;
        }}
        
        .action-needed {{
            background: #fed7d7;
            border-left: 4px solid #f56565;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
        }}
        
        .action-needed-title {{
            font-weight: 700;
            color: #c53030;
            margin-bottom: 5px;
        }}
        
        .action-needed-text {{
            color: #742a2a;
            font-size: 14px;
        }}
        
        .risk-indicator {{
            display: flex;
            align-items: center;
            gap: 15px;
            margin-top: 15px;
        }}
        
        .risk-badge {{
            padding: 10px 20px;
            border-radius: 25px;
            font-weight: 700;
            font-size: 18px;
        }}
        
        .risk-low {{
            background: #c6f6d5;
            color: #2f855a;
        }}
        
        .risk-medium {{
            background: #feebc8;
            color: #c05621;
        }}
        
        .risk-high {{
            background: #ffd4b3;
            color: #dd6b20;
        }}
        
        .risk-critical {{
            background: #fed7d7;
            color: #c53030;
        }}
        
        .cause-analysis {{
            background: #f7fafc;
            border-radius: 10px;
            padding: 20px;
            margin-top: 15px;
        }}
        
        .cause-item {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px;
            background: white;
            border-radius: 8px;
            margin-bottom: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }}
        
        .cause-icon {{
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            flex-shrink: 0;
        }}
        
        .cause-icon-critical {{
            background: #fed7d7;
        }}
        
        .cause-icon-warning {{
            background: #feebc8;
        }}
        
        .cause-text {{
            font-size: 15px;
            color: #2d3748;
            font-weight: 500;
        }}
        
        .momentum-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin-top: 15px;
        }}
        
        .momentum-item {{
            background: #f7fafc;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }}
        
        .momentum-label {{
            font-size: 12px;
            color: #718096;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .momentum-value {{
            font-size: 24px;
            font-weight: 700;
            color: #2d3748;
        }}
        
        .momentum-active {{
            color: #f56565;
        }}
        
        .factors-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin-top: 15px;
        }}
        
        .factor-item {{
            background: #f7fafc;
            padding: 15px;
            border-radius: 10px;
        }}
        
        .factor-label {{
            font-size: 12px;
            color: #718096;
            margin-bottom: 5px;
        }}
        
        .factor-value {{
            font-size: 24px;
            font-weight: 700;
            color: #2d3748;
        }}
        
        .factor-critical {{
            color: #f56565;
        }}
        
        .factor-warning {{
            color: #ed8936;
        }}
        
        .prediction-range {{
            background: #fef5e7;
            border: 2px dashed #f39c12;
            border-radius: 10px;
            padding: 15px;
            margin-top: 15px;
            text-align: center;
        }}
        
        .prediction-range-title {{
            font-size: 12px;
            color: #d68910;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 5px;
        }}
        
        .prediction-range-time {{
            font-size: 16px;
            font-weight: 700;
            color: #c77b00;
        }}
        
        .alert-box {{
            background: linear-gradient(135deg, #f56565 0%, #ed8936 100%);
            border-radius: 15px;
            padding: 25px;
            color: white;
            box-shadow: 0 10px 30px rgba(245, 101, 101, 0.3);
            margin-bottom: 20px;
        }}
        
        .alert-title {{
            font-size: 24px;
            font-weight: 700;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .alert-message {{
            font-size: 16px;
            line-height: 1.6;
        }}
        
        .footer {{
            text-align: center;
            color: white;
            margin-top: 30px;
            font-size: 14px;
        }}
        
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
        }}
        
        .alert-icon {{
            animation: pulse 2s infinite;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <div class="timestamp">생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
            <h1>🚨 V6 실시간 급증 예측 대시보드</h1>
            <p class="subtitle">30분 시퀀스 → 10분 후 예측 | Momentum Feature 기반 분석</p>
            <span class="model-info">{result['MODEL_VERSION']}</span>
        </div>

        <!-- Alert Box (DANGEROUS일 때만) -->
        {f'''<div class="alert-box">
            <div class="alert-title">
                <span class="alert-icon">🚨</span> 긴급 알림: 급증 위험 감지
            </div>
            <div class="alert-message">
                현재 시스템에서 급격한 상승 추세가 감지되었습니다. 다음 10분 내에 300을 초과할 확률이 <strong>{result['EXCEED_300_PROBABILITY']}</strong>입니다. {action_level.split()[1] if len(action_level.split()) > 1 else '주의가'} 필요합니다.
            </div>
        </div>''' if result['STATUS'] == 'DANGEROUS' else ''}

        <!-- Main Metrics Grid -->
        <div class="main-grid">
            <!-- Current Value -->
            <div class="card">
                <div class="card-title">현재값</div>
                <div class="big-number current-value">{result['CURRENT_VALUE']}</div>
                <div class="sub-info">30분 시퀀스의 마지막 값</div>
                <div class="sub-info">현재 시간: {current_time_str}</div>
            </div>

            <!-- Predicted Value -->
            <div class="card">
                <div class="card-title">예측값 (10분 후)</div>
                <div class="big-number predicted-value">{result['PREDICTED_VALUE']}</div>
                <div class="sub-info">예측 구간 최대값</div>
                <span class="status-badge {status_class}">{result['STATUS']}</span>
            </div>

            <!-- Increase Rate -->
            <div class="card">
                <div class="card-title">상승률</div>
                <div class="big-number increase-rate">{increase_rate_text}</div>
                <div class="sub-info">({result['PREDICTED_VALUE']} - {result['CURRENT_VALUE']}) / {result['CURRENT_VALUE']}</div>
                <div class="risk-indicator">
                    <span class="risk-badge {risk_class}">{result['SURGE_RISK']}</span>
                </div>
            </div>

            <!-- 300 Exceed Probability -->
            <div class="card probability-section">
                <div class="card-title">300 달성 확률</div>
                <div class="probability-circle">
                    <div class="probability-inner">
                        <div class="probability-number">{result['EXCEED_300_PROBABILITY']}</div>
                        <div class="probability-label">예측 구간 내</div>
                    </div>
                </div>
                <div class="{action_class}">
                    <div class="action-needed-title">{action_level}</div>
                    <div class="action-needed-text">{action_desc}</div>
                </div>
            </div>
        </div>

        <!-- Prediction Range -->
        <div class="card">
            <div class="prediction-range">
                <div class="prediction-range-title">🎯 예측 범위</div>
                <div class="prediction-range-time">{result['PREDICTION_TIME']}</div>
            </div>
        </div>

        <!-- Cause Analysis -->
        <div class="card">
            <div class="card-title">🔍 상승 원인 분석</div>
            <div class="cause-analysis">
                {cause_items_html}
            </div>
        </div>

        <!-- Momentum & Key Factors Grid -->
        <div class="main-grid">
            <!-- Momentum Signals -->
            <div class="card">
                <div class="card-title">🎯 Momentum 신호</div>
                <div class="momentum-grid">
                    <div class="momentum-item">
                        <div class="momentum-label">가속도</div>
                        <div class="momentum-value {'momentum-active' if result['MOMENTUM_SIGNAL']['acceleration'] > 0.5 else ''}">{result['MOMENTUM_SIGNAL']['acceleration']}</div>
                    </div>
                    <div class="momentum-item">
                        <div class="momentum-label">상승 중</div>
                        <div class="momentum-value {'momentum-active' if result['MOMENTUM_SIGNAL']['is_rising'] else ''}">{momentum_checks['is_rising']}</div>
                    </div>
                    <div class="momentum-item">
                        <div class="momentum-label">급등 여부</div>
                        <div class="momentum-value {'momentum-active' if result['MOMENTUM_SIGNAL']['rapid_rise'] else ''}">{momentum_checks['rapid_rise']}</div>
                    </div>
                    <div class="momentum-item">
                        <div class="momentum-label">급증 임박</div>
                        <div class="momentum-value {'momentum-active' if result['MOMENTUM_SIGNAL']['surge_imminent'] else ''}">{momentum_checks['surge_imminent']}</div>
                    </div>
                </div>
            </div>

            <!-- Key Factors -->
            <div class="card">
                <div class="card-title">📊 주요 지표</div>
                <div class="factors-grid">
                    <div class="factor-item">
                        <div class="factor-label">HUBROOMTOTAL</div>
                        <div class="factor-value {hub_class}">{result['KEY_FACTORS']['HUBROOMTOTAL']}</div>
                    </div>
                    <div class="factor-item">
                        <div class="factor-label">FS STORAGE UTIL</div>
                        <div class="factor-value {fs_class}">{result['KEY_FACTORS']['FS_STORAGE_UTIL']}</div>
                    </div>
                    <div class="factor-item">
                        <div class="factor-label">TOTAL CMD</div>
                        <div class="factor-value">{result['KEY_FACTORS']['TOTAL_CMD']}</div>
                    </div>
                    <div class="factor-item">
                        <div class="factor-label">STORAGE UTIL</div>
                        <div class="factor-value {storage_class}">{result['KEY_FACTORS']['STORAGE_UTIL']}</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Footer -->
        <div class="footer">
            <p>🤖 V6 Momentum Model | Powered by XGBoost + Momentum Features</p>
            <p>30분 시퀀스 기반 10분 후 예측 시스템</p>
        </div>
    </div>
</body>
</html>"""
    
    return html_content

def predict_realtime():
    """
    실시간 예측 실행 및 출력 + HTML 대시보드 자동 생성
    """
    result, error = realtime_prediction()
    
    if error:
        print([{"ERROR": error}])
    else:
        # 1. 콘솔 출력
        print([result])
        
        # 2. HTML 대시보드 자동 생성
        try:
            html_content = generate_html_dashboard(result)
            
            # 타임스탬프를 포함한 파일명
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            html_filename = f'V6_실시간예측_대시보드_{timestamp}.html'
            
            # HTML 파일 저장
            html_path = os.path.join(script_dir, html_filename)
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"\n✅ HTML 대시보드 생성 완료: {html_filename}")
            print(f"📂 파일 위치: {html_path}")
            
            # 🌐 브라우저에서 자동으로 열기!
            try:
                webbrowser.open('file://' + os.path.abspath(html_path))
                print(f"🌐 브라우저에서 대시보드를 열었습니다!")
            except Exception as browser_error:
                print(f"⚠️ 브라우저 자동 열기 실패: {str(browser_error)}")
                print(f"💡 수동으로 파일을 열어주세요: {html_path}")
            
        except Exception as e:
            print(f"\n⚠️ HTML 생성 실패: {str(e)}")

if __name__ == '__main__':
    predict_realtime()