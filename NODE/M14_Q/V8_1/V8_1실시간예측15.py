# -*- coding: utf-8 -*-
"""280분 → 15분 후 실시간 예측"""

import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
import os

def create_single_prediction_features(seq_m14b, seq_m10a, seq_m16, seq_totalcnt):
    """하나의 280분 시퀀스로부터 Feature 생성 (85개 - 조기 경보 포함)"""
    features = {}
    
    # ========== M14AM14B 기본 8개 ==========
    features['m14b_mean'] = np.mean(seq_m14b)
    features['m14b_std'] = np.std(seq_m14b)
    features['m14b_last_5_mean'] = np.mean(seq_m14b[-5:])
    features['m14b_max'] = np.max(seq_m14b)
    features['m14b_min'] = np.min(seq_m14b)
    features['m14b_slope'] = np.polyfit(np.arange(280), seq_m14b, 1)[0]
    features['m14b_last_10_mean'] = np.mean(seq_m14b[-10:])
    features['m14b_first_10_mean'] = np.mean(seq_m14b[:10])
    
    # ========== M14AM10A 기본 8개 ==========
    features['m10a_mean'] = np.mean(seq_m10a)
    features['m10a_std'] = np.std(seq_m10a)
    features['m10a_last_5_mean'] = np.mean(seq_m10a[-5:])
    features['m10a_max'] = np.max(seq_m10a)
    features['m10a_min'] = np.min(seq_m10a)
    features['m10a_slope'] = np.polyfit(np.arange(280), seq_m10a, 1)[0]
    features['m10a_last_10_mean'] = np.mean(seq_m10a[-10:])
    features['m10a_first_10_mean'] = np.mean(seq_m10a[:10])
    
    # ========== M14AM16 기본 8개 ==========
    features['m16_mean'] = np.mean(seq_m16)
    features['m16_std'] = np.std(seq_m16)
    features['m16_last_5_mean'] = np.mean(seq_m16[-5:])
    features['m16_max'] = np.max(seq_m16)
    features['m16_min'] = np.min(seq_m16)
    features['m16_slope'] = np.polyfit(np.arange(280), seq_m16, 1)[0]
    features['m16_last_10_mean'] = np.mean(seq_m16[-10:])
    features['m16_first_10_mean'] = np.mean(seq_m16[:10])
    
    # ========== TOTALCNT 기본 8개 ==========
    features['totalcnt_mean'] = np.mean(seq_totalcnt)
    features['totalcnt_std'] = np.std(seq_totalcnt)
    features['totalcnt_last_5_mean'] = np.mean(seq_totalcnt[-5:])
    features['totalcnt_max'] = np.max(seq_totalcnt)
    features['totalcnt_min'] = np.min(seq_totalcnt)
    features['totalcnt_slope'] = np.polyfit(np.arange(280), seq_totalcnt, 1)[0]
    features['totalcnt_last_10_mean'] = np.mean(seq_totalcnt[-10:])
    features['totalcnt_first_10_mean'] = np.mean(seq_totalcnt[:10])
    
    # ========== 비율 Feature (8개) ==========
    features['ratio_m14b_m10a'] = seq_m14b[-1] / (seq_m10a[-1] + 1)
    features['ratio_m14b_m16'] = seq_m14b[-1] / (seq_m16[-1] + 1)
    features['ratio_m10a_m16'] = seq_m10a[-1] / (seq_m16[-1] + 1)
    features['ratio_m14b_m10a_mean'] = np.mean(seq_m14b) / (np.mean(seq_m10a) + 1)
    features['ratio_m14b_m16_mean'] = np.mean(seq_m14b) / (np.mean(seq_m16) + 1)
    features['ratio_m14b_m10a_max'] = np.max(seq_m14b) / (np.max(seq_m10a) + 1)
    features['volatility_m14b'] = np.std(seq_m14b) / (np.mean(seq_m14b) + 1)
    features['volatility_totalcnt'] = np.std(seq_totalcnt) / (np.mean(seq_totalcnt) + 1)
    
    # ========== M14AM14B 임계값 카운트 (8개) ==========
    features['m14b_over_250'] = np.sum(seq_m14b > 250)
    features['m14b_over_300'] = np.sum(seq_m14b > 300)
    features['m14b_over_350'] = np.sum(seq_m14b > 350)
    features['m14b_over_400'] = np.sum(seq_m14b > 400)
    features['m14b_over_450'] = np.sum(seq_m14b > 450)
    features['m14b_over_300_last30'] = np.sum(seq_m14b[-30:] > 300)
    features['m14b_over_350_last30'] = np.sum(seq_m14b[-30:] > 350)
    features['m14b_over_400_last30'] = np.sum(seq_m14b[-30:] > 400)
    
    # ========== M14AM10A 임계값 카운트 (4개) ==========
    features['m10a_over_70'] = np.sum(seq_m10a > 70)
    features['m10a_over_80'] = np.sum(seq_m10a > 80)
    features['m10a_under_80'] = np.sum(seq_m10a < 80)
    features['m10a_under_70'] = np.sum(seq_m10a < 70)
    
    # ========== TOTALCNT 임계값 카운트 (8개) ==========
    features['totalcnt_over_1400'] = np.sum(seq_totalcnt >= 1400)
    features['totalcnt_over_1500'] = np.sum(seq_totalcnt >= 1500)
    features['totalcnt_over_1600'] = np.sum(seq_totalcnt >= 1600)
    features['totalcnt_over_1700'] = np.sum(seq_totalcnt >= 1700)
    features['totalcnt_over_1400_last30'] = np.sum(seq_totalcnt[-30:] >= 1400)
    features['totalcnt_over_1500_last30'] = np.sum(seq_totalcnt[-30:] >= 1500)
    features['totalcnt_over_1600_last30'] = np.sum(seq_totalcnt[-30:] >= 1600)
    features['totalcnt_over_1700_last30'] = np.sum(seq_totalcnt[-30:] >= 1700)
    
    # ========== 황금 패턴 (4개) ==========
    features['golden_pattern_300_80'] = 1 if (seq_m14b[-1] > 300 and seq_m10a[-1] < 80) else 0
    features['golden_pattern_350_80'] = 1 if (seq_m14b[-1] > 350 and seq_m10a[-1] < 80) else 0
    features['golden_pattern_400_70'] = 1 if (seq_m14b[-1] > 400 and seq_m10a[-1] < 70) else 0
    features['danger_zone'] = 1 if seq_totalcnt[-1] >= 1700 else 0
    
    # ========== 변화율/가속도 (8개) ==========
    features['m14b_change_rate'] = (seq_m14b[-1] - seq_m14b[-30]) / 30 if len(seq_m14b) >= 30 else 0
    features['totalcnt_change_rate'] = (seq_totalcnt[-1] - seq_totalcnt[-30]) / 30 if len(seq_totalcnt) >= 30 else 0
    
    recent_30_m14b = np.mean(seq_m14b[-30:])
    previous_30_m14b = np.mean(seq_m14b[-60:-30]) if len(seq_m14b) >= 60 else np.mean(seq_m14b[-30:])
    features['m14b_acceleration'] = recent_30_m14b - previous_30_m14b
    
    recent_30_totalcnt = np.mean(seq_totalcnt[-30:])
    previous_30_totalcnt = np.mean(seq_totalcnt[-60:-30]) if len(seq_totalcnt) >= 60 else np.mean(seq_totalcnt[-30:])
    features['totalcnt_acceleration'] = recent_30_totalcnt - previous_30_totalcnt
    
    features['m14b_range'] = np.max(seq_m14b) - np.min(seq_m14b)
    features['totalcnt_range'] = np.max(seq_totalcnt) - np.min(seq_totalcnt)
    features['m14b_recent_vs_mean'] = np.mean(seq_m14b[-30:]) / (np.mean(seq_m14b) + 1)
    features['totalcnt_recent_vs_mean'] = np.mean(seq_totalcnt[-30:]) / (np.mean(seq_totalcnt) + 1)
    
    # ========== 시간대별 통계 (8개) ==========
    q1 = seq_totalcnt[:70]
    q2 = seq_totalcnt[70:140]
    q3 = seq_totalcnt[140:210]
    q4 = seq_totalcnt[210:280]
    
    features['totalcnt_q1_mean'] = np.mean(q1)
    features['totalcnt_q2_mean'] = np.mean(q2)
    features['totalcnt_q3_mean'] = np.mean(q3)
    features['totalcnt_q4_mean'] = np.mean(q4)
    features['totalcnt_trend_q1_q2'] = np.mean(q2) - np.mean(q1)
    features['totalcnt_trend_q2_q3'] = np.mean(q3) - np.mean(q2)
    features['totalcnt_trend_q3_q4'] = np.mean(q4) - np.mean(q3)
    features['totalcnt_trend_overall'] = np.mean(q4) - np.mean(q1)
    
    # ========== 🔥 조기 경보 Feature (5개) ==========
    last_15min = seq_totalcnt[-10:]
    
    features['last_15min_max'] = np.max(last_15min)
    features['last_15min_min'] = np.min(last_15min)
    features['last_15min_mean'] = np.mean(last_15min)
    features['last_15min_rise'] = last_15min[-1] - last_15min[0]
    
    early_warning = (np.max(last_15min) >= 1650) and ((last_15min[-1] - last_15min[0]) > 20)
    features['early_warning_1650_rising'] = 1 if early_warning else 0
    
    return features

def get_status_info(value):
    """물류량에 따른 상태 정보 반환"""
    if value < 900:
        return 'Low'
    elif value < 1600:
        return 'Normal'
    elif value < 1700:
        return 'Warning'
    else:
        return 'Critical'

def predict_latest_only():
    """최신 280분 데이터로 15분 후 예측"""
    
    # 모델 로드
    model_files = [
        'xgboost_280to15_1year_augmented.pkl',
        'xgboost_280to15_enhanced_earlywarning.pkl',
        '/mnt/user-data/outputs/xgboost_280to15_1year_augmented.pkl'
    ]
    
    model = None
    for mf in model_files:
        if os.path.exists(mf):
            try:
                with open(mf, 'rb') as f:
                    model = pickle.load(f)
                break
            except:
                continue
    
    if model is None:
        print("❌ 모델 파일 없음")
        return None
    
    # 데이터 로드
    csv_files = [
        '/mnt/project/V6_6결과.CSV',
        'V6_6결과.CSV',
        'data/222.csv'
    ]
    
    df = None
    for cf in csv_files:
        if os.path.exists(cf):
            try:
                df = pd.read_csv(cf, on_bad_lines='skip')
                break
            except:
                continue
    
    if df is None:
        print("❌ CSV 파일 없음")
        return None
    
    # 필수 컬럼 확인
    required_cols = ['M14AM14B', 'M14AM10A', 'M14AM16', 'TOTALCNT']
    if not all(col in df.columns for col in required_cols):
        print("❌ 필수 컬럼 누락")
        return None
    
    if len(df) < 280:
        print("❌ 데이터 부족")
        return None
    
    # CURRTIME 처리
    if 'CURRTIME' in df.columns:
        try:
            df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M')
        except:
            try:
                df['CURRTIME'] = pd.to_datetime(df['CURRTIME'])
            except:
                base_time = datetime.now()
                df['CURRTIME'] = [base_time + timedelta(minutes=i) for i in range(len(df))]
    else:
        base_time = datetime.now()
        df['CURRTIME'] = [base_time + timedelta(minutes=i) for i in range(len(df))]
    
    # 최근 280분 데이터 추출
    seq_m14b = df['M14AM14B'].iloc[-280:].values
    seq_m10a = df['M14AM10A'].iloc[-280:].values
    seq_m16 = df['M14AM16'].iloc[-280:].values
    seq_totalcnt = df['TOTALCNT'].iloc[-280:].values
    
    # 현재 시간
    current_time = df['CURRTIME'].iloc[-1]
    prediction_time = current_time + timedelta(minutes=15)
    
    # Feature 생성
    features = create_single_prediction_features(seq_m14b, seq_m10a, seq_m16, seq_totalcnt)
    X_pred = pd.DataFrame([features])
    
    # 예측
    prediction = model.predict(X_pred)[0]
    
    # 상태 판정
    pred_status = get_status_info(prediction)
    
    # 결과 출력
    simple_dict = {
        'prediction_time': prediction_time.strftime('%Y-%m-%d %H:%M'),
        'prediction': int(prediction),
        'status': pred_status
    }
    
    print(simple_dict)
    return simple_dict

if __name__ == '__main__':
    predict_latest_only()