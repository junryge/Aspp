#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 V9 평가 - 25분 후 예측
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
print("🚀 V9 평가 - 25분 후 예측")
print("="*70)

PRED_OFFSET = 25

# ========================================
# Feature 생성 함수
# ========================================
def create_features_v9(row_dict):
    features = {}
    
    seq_m14b = np.array(row_dict['M14AM14B'])
    seq_m14bsum = np.array(row_dict['M14AM14BSUM'])
    seq_m10arev = np.array(row_dict['M10AM14A'])
    seq_totalcnt = np.array(row_dict['TOTALCNT'])
    seq_transport = np.array(row_dict['TRANSPORT'])
    seq_oht = np.array(row_dict['OHT'])
    
    seq_len = len(seq_m14b)
    
    features['m14b_mean'] = np.mean(seq_m14b)
    features['m14b_max'] = np.max(seq_m14b)
    features['m14b_current'] = seq_m14b[-1]
    features['m14b_last10'] = np.mean(seq_m14b[-10:])
    features['m14b_last30'] = np.mean(seq_m14b[-30:])
    features['m14b_slope'] = np.polyfit(np.arange(seq_len), seq_m14b, 1)[0]
    features['m14b_std'] = np.std(seq_m14b)
    features['m14b_trend10'] = seq_m14b[-1] - seq_m14b[-10]
    
    features['m14bsum_mean'] = np.mean(seq_m14bsum)
    features['m14bsum_max'] = np.max(seq_m14bsum)
    features['m14bsum_current'] = seq_m14bsum[-1]
    features['m14bsum_last10'] = np.mean(seq_m14bsum[-10:])
    features['m14bsum_last30'] = np.mean(seq_m14bsum[-30:])
    features['m14bsum_slope'] = np.polyfit(np.arange(seq_len), seq_m14bsum, 1)[0]
    features['m14bsum_std'] = np.std(seq_m14bsum)
    features['m14bsum_trend10'] = seq_m14bsum[-1] - seq_m14bsum[-10]
    
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
    
    features['m10arev_mean'] = np.mean(seq_m10arev)
    features['m10arev_max'] = np.max(seq_m10arev)
    features['m10arev_current'] = seq_m10arev[-1]
    features['m10arev_last10'] = np.mean(seq_m10arev[-10:])
    features['m10arev_slope'] = np.polyfit(np.arange(seq_len), seq_m10arev, 1)[0]
    
    features['trans_mean'] = np.mean(seq_transport)
    features['trans_max'] = np.max(seq_transport)
    features['trans_current'] = seq_transport[-1]
    features['trans_last10'] = np.mean(seq_transport[-10:])
    features['trans_slope'] = np.polyfit(np.arange(seq_len), seq_transport, 1)[0]
    
    features['oht_mean'] = np.mean(seq_oht)
    features['oht_max'] = np.max(seq_oht)
    features['oht_current'] = seq_oht[-1]
    features['oht_last10'] = np.mean(seq_oht[-10:])
    
    features['m14b_x_sum'] = seq_m14b[-1] * seq_m14bsum[-1] / 1000
    features['m14b_x_sum_mean'] = np.mean(seq_m14b * seq_m14bsum) / 1000
    features['sum_per_m14b'] = seq_m14bsum[-1] / (seq_m14b[-1] + 1)
    features['m14b_plus_sum'] = seq_m14b[-1] + seq_m14bsum[-1]
    features['m10arev_x_m14b'] = seq_m10arev[-1] * seq_m14b[-1] / 100
    features['trans_x_m14b'] = seq_transport[-1] * seq_m14b[-1] / 100
    features['ratio_m14b_total'] = seq_m14b[-1] / (seq_totalcnt[-1] + 1)
    features['ratio_sum_total'] = seq_m14bsum[-1] / (seq_totalcnt[-1] + 1)
    
    features['m14b_over_520'] = np.sum(seq_m14b > 520)
    features['m14b_over_540'] = np.sum(seq_m14b > 540)
    features['m14bsum_over_600'] = np.sum(seq_m14bsum > 600)
    features['m14bsum_over_620'] = np.sum(seq_m14bsum > 620)
    features['m10arev_over_55'] = np.sum(seq_m10arev > 55)
    features['total_over_1600'] = np.sum(seq_totalcnt >= 1600)
    features['total_over_1650'] = np.sum(seq_totalcnt >= 1650)
    features['total_over_1700'] = np.sum(seq_totalcnt >= 1700)
    
    features['gold_strict'] = 1 if (seq_m14b[-1] > 540 and seq_m14bsum[-1] > 620) else 0
    features['gold_normal'] = 1 if (seq_m14b[-1] > 520 and seq_m14bsum[-1] > 600) else 0
    features['in_danger'] = 1 if seq_totalcnt[-1] >= 1700 else 0
    features['near_danger'] = 1 if seq_totalcnt[-1] >= 1600 else 0
    
    return features

# ========================================
# 보정 함수
# ========================================
def adjust_prediction(pred, current_total, m14b, m14bsum):
    # 하한선 (25분이라 130으로 확대)
    floor = current_total - 130
    if pred < floor:
        pred = floor
    
    if 1650 <= pred < 1700:
        if m14b > 540 and m14bsum > 620:
            pred += 50
        elif m14b > 520 and m14bsum > 600:
            pred += 40
    
    if current_total >= 1700 and pred < 1680:
        pred = max(pred, current_total - 100)
    
    return pred

# ========================================
# 메인
# ========================================
model = None
for mf in ['model_v9_25min.pkl', 'model_13col_final.pkl']:
    if os.path.exists(mf):
        with open(mf, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ 모델: {mf}")
        break

if not model:
    print("❌ 모델 없음!")
    exit(1)

DATA_FILE = 'data/M14_Q_20251014_TO_20251015.CSV'
df = pd.read_csv(DATA_FILE, on_bad_lines='skip')
print(f"✅ 데이터: {len(df):,}행")
df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M')

results = []
seq_len = 280
total = len(df) - seq_len - PRED_OFFSET
print(f"\n🔄 평가 중... {total:,}개")

for i in range(seq_len, len(df) - PRED_OFFSET):
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
    actual = df['TOTALCNT'].iloc[i + PRED_OFFSET - 1]
    
    seq_start_time = df['CURRTIME'].iloc[i-seq_len]
    prediction_time = current_time + timedelta(minutes=PRED_OFFSET)
    actual_time = df['CURRTIME'].iloc[i + PRED_OFFSET - 1]
    
    features = create_features_v9(row_dict)
    X = pd.DataFrame([features])
    
    pred_raw = model.predict(X)[0]
    pred = adjust_prediction(pred_raw, current_total, seq_m14b[-1], seq_m14bsum[-1])
    
    gold_strict = (seq_m14b[-1] > 540 and seq_m14bsum[-1] > 620)
    gold_normal = (seq_m14b[-1] > 520 and seq_m14bsum[-1] > 600)
    
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
        'M14AM14B': round(seq_m14b[-1], 2),
        'M14AM14BSUM': round(seq_m14bsum[-1], 2),
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
output = 'evaluation_V9_25min.csv'
df_result.to_csv(output, index=False, encoding='utf-8-sig')
print(f"✅ 저장: {output}")

print("\n" + "="*70)
print(f"📊 V9 25분 평가 통계")
print("="*70)
print(f"MAE: {df_result['절대오차'].mean():.2f}")

actual_danger = df_result['실제위험(1700+)'] == 'O'
pred_danger = df_result['예측위험(1700+)'] == 'O'
actual_count = actual_danger.sum()
detected = (actual_danger & pred_danger).sum()

if actual_count > 0:
    print(f"1700+ 감지율: {detected/actual_count*100:.1f}% ({detected}/{actual_count})")

print(f"\n✅ 완료! → {output}")