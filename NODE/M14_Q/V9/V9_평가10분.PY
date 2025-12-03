#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 V9 평가 - 현재값 하한선 적용!

핵심 수정:
1. 예측값 >= 현재값 - 80 (물리적 하한선!)
2. 상한선 제거 - 있는 그대로!
3. Feature 간소화 (학습과 동일)
"""

import numpy as np
import pandas as pd
import pickle
from datetime import timedelta
import os
import gc
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🚀 V9 평가 - 현재값 하한선 + 상한선 제거")
print("="*70)

# ========================================
# Feature 생성 함수 (학습과 100% 동일!)
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
# 보정 함수 - 현재값 하한선!
# ========================================
def adjust_prediction(pred, current_total, m14b, m14bsum):
    """
    🔥 핵심 보정:
    1. 현재값 하한선: 예측 >= 현재 - 80 (10분에 80 이상 하락 불가능!)
    2. 황금패턴 시 상향 boost
    3. 상한선 없음!
    """
    
    # 🔥 1. 현재값 하한선 (가장 중요!)
    # 10분 동안 80 이상 떨어지는 건 물리적으로 드묾
    floor = current_total - 80
    if pred < floor:
        pred = floor
    
    # 2. 황금패턴 boost (1650~1699 구간만)
    if 1650 <= pred < 1700:
        boost = 0
        
        # 엄격 황금패턴
        if m14b > 540 and m14bsum > 620:
            boost += 50
        # 보통 황금패턴
        elif m14b > 520 and m14bsum > 600:
            boost += 40
        
        # 고값 추가 boost
        if m14b > 560:
            boost += 20
        if m14bsum > 650:
            boost += 20
            
        pred = pred + boost
    
    # 3. 이미 위험 구간일 때
    if current_total >= 1700:
        # 현재 1700+인데 예측이 낮으면 올림
        if pred < 1680:
            pred = max(pred, current_total - 50)
    
    # 상한선 없음! 있는 그대로!
    return pred

# ========================================
# 메인
# ========================================
# 모델 로드
model = None
for mf in ['model_v9.pkl', 'model_13col_final.pkl']:
    if os.path.exists(mf):
        with open(mf, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ 모델: {mf}")
        break

if not model:
    print("❌ 모델 없음! model_v9.pkl 또는 model_13col_final.pkl 필요")
    exit(1)

# 데이터 로드
DATA_FILE = 'data/M14_Q_20251014_TO_20251015.CSV'
df = pd.read_csv(DATA_FILE, on_bad_lines='skip')
print(f"✅ 데이터: {len(df):,}행")
df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M')

# 평가
results = []
seq_len = 280
total = len(df) - seq_len - 10
print(f"\n🔄 평가 중... {total:,}개")

for i in range(seq_len, len(df) - 10):
    row_dict = {
        'M14AM14B': df['M14AM14B'].iloc[i-seq_len:i].values,
        'M14AM14BSUM': df['M14AM14BSUM'].iloc[i-seq_len:i].values,
        'M10AM14A': df['M10AM14A'].iloc[i-seq_len:i].values,
        'TOTALCNT': df['TOTALCNT'].iloc[i-seq_len:i].values,
        'TRANSPORT': df['M14.QUE.ALL.TRANSPORT4MINOVERCNT'].iloc[i-seq_len:i].values,
        'OHT': df['M14.QUE.OHT.OHTUTIL'].iloc[i-seq_len:i].values,
    }
    
    seq_totalcnt = row_dict['TOTALCNT']
    seq_m14b = row_dict['M14AM14B']
    seq_m14bsum = row_dict['M14AM14BSUM']
    
    current_time = df['CURRTIME'].iloc[i-1]
    current_total = seq_totalcnt[-1]
    actual = df['TOTALCNT'].iloc[i+9]
    
    seq_start_time = df['CURRTIME'].iloc[i-seq_len]
    prediction_time = current_time + timedelta(minutes=10)
    actual_time = df['CURRTIME'].iloc[i+9]
    
    # Feature 생성
    features = create_features_v9(row_dict)
    X = pd.DataFrame([features])
    
    # 예측
    pred_raw = model.predict(X)[0]
    
    # 🔥 보정 적용 (현재값 하한선!)
    pred = adjust_prediction(pred_raw, current_total, seq_m14b[-1], seq_m14bsum[-1])
    
    # 패턴
    gold_strict = (seq_m14b[-1] > 540 and seq_m14bsum[-1] > 620)
    gold_normal = (seq_m14b[-1] > 520 and seq_m14bsum[-1] > 600)
    
    # 결과 저장
    results.append({
        '시퀀스시작': seq_start_time.strftime('%Y-%m-%d %H:%M'),
        '현재시간': current_time.strftime('%Y-%m-%d %H:%M'),
        '현재TOTALCNT': round(current_total, 2),
        '예측시점': prediction_time.strftime('%Y-%m-%d %H:%M'),
        '실제시점': actual_time.strftime('%Y-%m-%d %H:%M'),
        '실제값': round(actual, 2),
        '원본예측': round(pred_raw, 2),
        '보정예측': round(pred, 2),
        '오차': round(actual - pred, 2),
        '절대오차': round(abs(actual - pred), 2),
        '오차율(%)': round(abs(actual - pred) / max(actual, 1) * 100, 2),
        'M14AM14B': round(seq_m14b[-1], 2),
        'M14AM14BSUM': round(seq_m14bsum[-1], 2),
        'M10AM14A': round(row_dict['M10AM14A'][-1], 2),
        'TRANSPORT': round(row_dict['TRANSPORT'][-1], 2),
        'OHT_UTIL': round(row_dict['OHT'][-1], 2),
        '황금패턴(엄격)': 'O' if gold_strict else '',
        '황금패턴(보통)': 'O' if gold_normal else '',
        '실제위험(1700+)': 'O' if actual >= 1700 else '',
        '예측위험(1700+)': 'O' if pred >= 1700 else ''
    })
    
    if (i - seq_len) % 1000 == 0:
        print(f"  {i-seq_len:,}/{total:,}")
        gc.collect()

print(f"✅ 완료!")

df_result = pd.DataFrame(results)
output = 'evaluation_V9.csv'
df_result.to_csv(output, index=False, encoding='utf-8-sig')
print(f"✅ 저장: {output}")

del df
gc.collect()

# ===== 통계 =====
print("\n" + "="*70)
print("📊 V9 평가 통계")
print("="*70)
print(f"총 예측: {len(df_result):,}개")
print(f"MAE: {df_result['절대오차'].mean():.2f}")
print(f"오차율: {df_result['오차율(%)'].mean():.2f}%")

# 1700+ 분석
actual_danger = df_result['실제위험(1700+)'] == 'O'
pred_danger = df_result['예측위험(1700+)'] == 'O'

actual_count = actual_danger.sum()
detected = (actual_danger & pred_danger).sum()

print(f"\n🔥 1700+ 감지:")
print(f"  실제: {actual_count}개")
if actual_count > 0:
    print(f"  감지: {detected}개")
    print(f"  감지율: {detected/actual_count*100:.1f}%")
    
    # 미감지 케이스
    missed = df_result[actual_danger & ~pred_danger]
    if len(missed) > 0:
        print(f"\n❌ 미감지 {len(missed)}개:")
        cols = ['현재시간', '현재TOTALCNT', '실제값', '원본예측', '보정예측', 'M14AM14B', 'M14AM14BSUM']
        print(missed[cols].head(10).to_string(index=False))

# 보정 효과
if actual_count > 0:
    danger_rows = df_result[actual_danger]
    raw_detected = (danger_rows['원본예측'] >= 1700).sum()
    adj_detected = (danger_rows['보정예측'] >= 1700).sum()
    
    print(f"\n📈 보정 효과:")
    print(f"  원본 감지율: {raw_detected/actual_count*100:.1f}%")
    print(f"  보정 감지율: {adj_detected/actual_count*100:.1f}%")
    print(f"  개선: +{(adj_detected-raw_detected)/actual_count*100:.1f}%p")

# 물리적 불가능 케이스 확인
absurd = df_result[(df_result['현재TOTALCNT'] - df_result['보정예측']) > 100]
print(f"\n🔍 물리적 이상 케이스 (현재-예측 > 100): {len(absurd)}개")

print("\n" + "="*70)
print(f"✅ V9 평가 완료! → {output}")
print("="*70)