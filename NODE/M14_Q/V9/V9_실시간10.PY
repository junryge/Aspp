#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 실시간 예측 시스템 - V9 (Feature 간소화 버전)
222개 → 60개 Feature
"""

import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
import os

# ========================================
# Feature 생성 함수 (V9 - 60개!)
# ========================================
def create_features_v9(row_dict):
    """핵심 컬럼 기반 60개 Feature"""
    features = {}
    
    seq_m14b = np.array(row_dict['M14AM14B'])
    seq_m14bsum = np.array(row_dict['M14AM14BSUM'])
    seq_m10arev = np.array(row_dict['M10AM14A'])
    seq_totalcnt = np.array(row_dict['TOTALCNT'])
    seq_transport = np.array(row_dict['TRANSPORT'])
    seq_oht = np.array(row_dict['OHT'])
    
    seq_len = len(seq_m14b)
    
    # 1. M14AM14B (8개)
    features['m14b_mean'] = np.mean(seq_m14b)
    features['m14b_max'] = np.max(seq_m14b)
    features['m14b_current'] = seq_m14b[-1]
    features['m14b_last10'] = np.mean(seq_m14b[-10:])
    features['m14b_last30'] = np.mean(seq_m14b[-30:])
    features['m14b_slope'] = np.polyfit(np.arange(seq_len), seq_m14b, 1)[0]
    features['m14b_std'] = np.std(seq_m14b)
    features['m14b_trend10'] = seq_m14b[-1] - seq_m14b[-10]
    
    # 2. M14AM14BSUM (8개)
    features['m14bsum_mean'] = np.mean(seq_m14bsum)
    features['m14bsum_max'] = np.max(seq_m14bsum)
    features['m14bsum_current'] = seq_m14bsum[-1]
    features['m14bsum_last10'] = np.mean(seq_m14bsum[-10:])
    features['m14bsum_last30'] = np.mean(seq_m14bsum[-30:])
    features['m14bsum_slope'] = np.polyfit(np.arange(seq_len), seq_m14bsum, 1)[0]
    features['m14bsum_std'] = np.std(seq_m14bsum)
    features['m14bsum_trend10'] = seq_m14bsum[-1] - seq_m14bsum[-10]
    
    # 3. TOTALCNT (10개)
    features['total_mean'] = np.mean(seq_totalcnt)
    features['total_max'] = np.max(seq_totalcnt)
    features['total_min'] = np.min(seq_totalcnt)
    features['total_current'] = seq_totalcnt[-1]
    features['total_last5'] = np.mean(seq_totalcnt[-5:])
    features['total_last10'] = np.mean(seq_totalcnt[-10:])
    features['total_last30'] = np.mean(seq_totalcnt[-30:])
    features['total_slope'] = np.polyfit(np.arange(seq_len), seq_totalcnt, 1)[0]
    features['total_std'] = np.std(seq_totalcnt)
    features['total_trend10'] = seq_totalcnt[-1] - seq_totalcnt[-10]
    
    # 4. M10AM14A (5개)
    features['m10arev_mean'] = np.mean(seq_m10arev)
    features['m10arev_max'] = np.max(seq_m10arev)
    features['m10arev_current'] = seq_m10arev[-1]
    features['m10arev_last10'] = np.mean(seq_m10arev[-10:])
    features['m10arev_slope'] = np.polyfit(np.arange(seq_len), seq_m10arev, 1)[0]
    
    # 5. TRANSPORT (5개)
    features['trans_mean'] = np.mean(seq_transport)
    features['trans_max'] = np.max(seq_transport)
    features['trans_current'] = seq_transport[-1]
    features['trans_last10'] = np.mean(seq_transport[-10:])
    features['trans_slope'] = np.polyfit(np.arange(seq_len), seq_transport, 1)[0]
    
    # 6. OHT (4개)
    features['oht_mean'] = np.mean(seq_oht)
    features['oht_max'] = np.max(seq_oht)
    features['oht_current'] = seq_oht[-1]
    features['oht_last10'] = np.mean(seq_oht[-10:])
    
    # Interaction (8개)
    features['m14b_x_sum'] = seq_m14b[-1] * seq_m14bsum[-1] / 1000
    features['m14b_x_sum_mean'] = np.mean(seq_m14b * seq_m14bsum) / 1000
    features['sum_per_m14b'] = seq_m14bsum[-1] / (seq_m14b[-1] + 1)
    features['m14b_plus_sum'] = seq_m14b[-1] + seq_m14bsum[-1]
    features['m10arev_x_m14b'] = seq_m10arev[-1] * seq_m14b[-1] / 100
    features['trans_x_m14b'] = seq_transport[-1] * seq_m14b[-1] / 100
    features['ratio_m14b_total'] = seq_m14b[-1] / (seq_totalcnt[-1] + 1)
    features['ratio_sum_total'] = seq_m14bsum[-1] / (seq_totalcnt[-1] + 1)
    
    # 임계값 (8개)
    features['m14b_over_520'] = np.sum(seq_m14b > 520)
    features['m14b_over_540'] = np.sum(seq_m14b > 540)
    features['m14bsum_over_600'] = np.sum(seq_m14bsum > 600)
    features['m14bsum_over_620'] = np.sum(seq_m14bsum > 620)
    features['m10arev_over_55'] = np.sum(seq_m10arev > 55)
    features['total_over_1600'] = np.sum(seq_totalcnt >= 1600)
    features['total_over_1650'] = np.sum(seq_totalcnt >= 1650)
    features['total_over_1700'] = np.sum(seq_totalcnt >= 1700)
    
    # 황금 패턴 (4개)
    features['gold_strict'] = 1 if (seq_m14b[-1] > 540 and seq_m14bsum[-1] > 620) else 0
    features['gold_normal'] = 1 if (seq_m14b[-1] > 520 and seq_m14bsum[-1] > 600) else 0
    features['in_danger'] = 1 if seq_totalcnt[-1] >= 1700 else 0
    features['near_danger'] = 1 if seq_totalcnt[-1] >= 1600 else 0
    
    return features

# ========================================
# 보정 함수 (V9!)
# ========================================
def adjust_prediction_v9(pred, current_total, m14b, m14bsum):
    """V9 보정 - 하한선 + 황금패턴 boost"""
    # 하한선 (10분이라 80)
    floor = current_total - 80
    if pred < floor:
        pred = floor
    
    # 황금패턴 boost (1650~1699)
    if 1650 <= pred < 1700:
        if m14b > 540 and m14bsum > 620:
            pred += 50
        elif m14b > 520 and m14bsum > 600:
            pred += 40
    
    # 현재 1700+ 보정
    if current_total >= 1700 and pred < 1680:
        pred = max(pred, current_total - 50)
    
    return pred

# ========================================
# 상태 판정 함수
# ========================================
def get_status_info(value):
    if value < 900:
        return 'LOW'
    elif value < 1600:
        return 'NORMAL'
    elif value < 1700:
        return 'CAUTION'
    else:
        return 'CRITICAL'

# ========================================
# 실시간 예측 함수
# ========================================
def predict_latest():
    """
    가장 최근 280분 데이터로 10분 후 예측
    """
    
    # 1. 모델 로드
    model_file = 'model_v9.pkl'
    try:
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"❌ Model file error: {e}")
        return {
            'prediction': 0,
            'status': 'Model operation failure',
            'prediction_time': '',
            'danger_probability': 0,
            'error_message': f'No model file: {model_file}'
        }
    
    # 2. CSV 파일 확인
    csv_file = 'data/2025_DATA.CSV'
    if not os.path.exists(csv_file):
        print(f"❌ No CSV file: {csv_file}")
        return {
            'prediction': 0,
            'status': 'No data',
            'prediction_time': '',
            'danger_probability': 0,
            'error_message': f'No data: {csv_file}'
        }
    
    # 3. 데이터 로드
    try:
        df = pd.read_csv(csv_file, on_bad_lines='skip', dtype={'CURRTIME': str})
    except Exception as e:
        print(f"❌ Data Load Error: {e}")
        return {
            'prediction': 0,
            'status': 'No data',
            'prediction_time': '',
            'danger_probability': 0,
            'error_message': f'Failed to load data: {e}'
        }
    
    # 4. 데이터가 비어있는 경우
    if len(df) == 0:
        print(f"❌ No data: CSV file empty")
        return {
            'prediction': 0,
            'status': 'No data',
            'prediction_time': '',
            'danger_probability': 0,
            'error_message': 'CSV file is empty'
        }
    
    # 5. 필수 컬럼 확인 (V9: 6개!)
    required_cols = [
        'M14AM14B', 'M14AM14BSUM', 'M10AM14A', 'TOTALCNT',
        'M14.QUE.ALL.TRANSPORT4MINOVERCNT', 'M14.QUE.OHT.OHTUTIL'
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ Missing required column: {missing_cols}")
        return {
            'prediction': 0,
            'status': 'No data',
            'prediction_time': '',
            'danger_probability': 0,
            'error_message': f'Missing required column: {", ".join(missing_cols)}'
        }
    
    # 6. 데이터 부족 체크
    if len(df) < 280:
        print(f"❌ Lack of data: {len(df)}EA (Minimum 280 EA pieces)")
        return {
            'prediction': 0,
            'status': 'Lack of data',
            'prediction_time': '',
            'danger_probability': 0,
            'error_message': f'Lack of data: {len(df)}EA (Minimum 280 EA pieces)'
        }
    
    # 7. CURRTIME 파싱
    if 'CURRTIME' in df.columns:
        try:
            df['CURRTIME'] = df['CURRTIME'].astype(str).str.strip()
            df = df[df['CURRTIME'].str.len() == 12].copy()
            if len(df) == 0:
                raise ValueError("유효한 CURRTIME 없음")
            df['CURRTIME'] = pd.to_datetime(df['CURRTIME'], format='%Y%m%d%H%M', errors='coerce')
            df = df.dropna(subset=['CURRTIME']).copy()
            if len(df) < 280:
                raise ValueError(f"CURRTIME 파싱 후 데이터 부족: {len(df)}개")
        except Exception as e:
            base_time = datetime.now()
            df['CURRTIME'] = [base_time - timedelta(minutes=len(df)-1-i) for i in range(len(df))]
    else:
        base_time = datetime.now()
        df['CURRTIME'] = [base_time - timedelta(minutes=len(df)-1-i) for i in range(len(df))]
    
    # 최종 데이터 부족 재확인
    if len(df) < 280:
        print(f"❌ Lack of data: {len(df)}EA (Minimum 280 EA pieces)")
        return {
            'prediction': 0,
            'status': 'Lack of data',
            'prediction_time': '',
            'danger_probability': 0,
            'error_message': f'Lack of data: {len(df)}EA (Minimum 280 EA pieces)'
        }
    
    # 8. Feature 추출 및 예측
    try:
        # 최근 280분 데이터 추출 (V9: 6개 컬럼!)
        row_dict = {
            'M14AM14B': df['M14AM14B'].iloc[-280:].values,
            'M14AM14BSUM': df['M14AM14BSUM'].iloc[-280:].values,
            'M10AM14A': df['M10AM14A'].iloc[-280:].values,
            'TOTALCNT': df['TOTALCNT'].iloc[-280:].values,
            'TRANSPORT': df['M14.QUE.ALL.TRANSPORT4MINOVERCNT'].iloc[-280:].values,
            'OHT': df['M14.QUE.OHT.OHTUTIL'].iloc[-280:].values,
        }
        
        # 시간 정보
        current_time = df['CURRTIME'].iloc[-1]
        if pd.isna(current_time):
            current_time = datetime.now()
        prediction_time = current_time + timedelta(minutes=10)
        
        # 현재 상태
        seq_totalcnt = row_dict['TOTALCNT']
        seq_m14b = row_dict['M14AM14B']
        seq_m14bsum = row_dict['M14AM14BSUM']
        
        current_totalcnt = seq_totalcnt[-1]
        current_m14b = seq_m14b[-1]
        current_m14bsum = seq_m14bsum[-1]
        
        # Feature 생성 (V9!)
        features = create_features_v9(row_dict)
        X_pred = pd.DataFrame([features])
        
        # 원본 예측
        pred_raw = model.predict(X_pred)[0]
        
        # V9 보정
        pred = adjust_prediction_v9(pred_raw, current_totalcnt, current_m14b, current_m14bsum)
        
        # 상태 판정
        pred_status = get_status_info(pred)
        
        # 패턴 감지 (V9 임계값!)
        gold_strict = (current_m14b > 540 and current_m14bsum > 620)
        gold_normal = (current_m14b > 520 and current_m14bsum > 600)
        
        # 1700+ 위험 확률 계산
        danger_prob = 0
        
        if pred >= 1750:
            danger_prob = 100
        elif pred >= 1700:
            danger_prob = 95
        elif pred >= 1680:
            danger_prob = 75
        elif pred >= 1650:
            danger_prob = 50
        elif pred >= 1620:
            danger_prob = 30
        elif pred >= 1600:
            danger_prob = 15
        else:
            danger_prob = 5
        
        # 황금 패턴 보정
        if gold_strict:
            danger_prob = min(100, danger_prob + 20)
        elif gold_normal:
            danger_prob = min(100, danger_prob + 15)
        elif (current_m14b > 509 and current_m14bsum > 570):
            danger_prob = min(100, danger_prob + 10)
        
        # 현재값 보정
        if current_totalcnt >= 1700:
            danger_prob = max(danger_prob, 85)
        elif current_totalcnt >= 1650:
            danger_prob = max(danger_prob, 60)
        
        danger_prob = max(0, min(100, danger_prob))
        
        # 정상 결과 반환
        result = {
            'prediction': int(pred),
            'status': pred_status,
            'prediction_time': prediction_time.strftime('%Y-%m-%d %H:%M'),
            'danger_probability': danger_prob
        }
        
        return result
        
    except Exception as e:
        print(f"❌ Predictive execution error: {e}")
        return {
            'prediction': 0,
            'status': 'Model operation failure',
            'prediction_time': '',
            'danger_probability': 0,
            'error_message': f'Model operation failure: {e}'
        }

if __name__ == '__main__':
    result = predict_latest()
    
    if result is not None:
        print(f"prediction: {result['prediction']:,}")
        print(f"status: {result['status']}")
        print(f"prediction_time: {result['prediction_time']}")
        print(f"1700+ danger_probability: {result['danger_probability']}%")