#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 V10.0 평가 - 10분 예측
시퀀스: 100분, 예측: 10분 후
"""

import numpy as np
import pandas as pd
import pickle
from datetime import timedelta
import glob
import gc
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🚀 V10.0 평가 - 10분 예측 (시퀀스 100분)")
print("="*70)

SEQ_LEN = 100
PRED_OFFSET = 10

def create_features(row_dict):
    features = {}
    seq_m14b = np.array(row_dict['M14AM14B'])
    seq_m14bsum = np.array(row_dict['M14AM14BSUM'])
    seq_m10a = np.array(row_dict['M10AM14A'])
    seq_total = np.array(row_dict['TOTALCNT'])
    seq_trans = np.array(row_dict['TRANSPORT'])
    seq_oht = np.array(row_dict['OHT'])
    seq_created = np.array(row_dict['Q_CREATED'])
    seq_completed = np.array(row_dict['Q_COMPLETED'])
    seq_gap = seq_created - seq_completed
    n = len(seq_m14b)
    
    features['m14b_mean'] = np.mean(seq_m14b)
    features['m14b_max'] = np.max(seq_m14b)
    features['m14b_current'] = seq_m14b[-1]
    features['m14b_last10'] = np.mean(seq_m14b[-10:])
    features['m14b_last30'] = np.mean(seq_m14b[-30:])
    features['m14b_slope'] = np.polyfit(np.arange(n), seq_m14b, 1)[0]
    features['m14b_std'] = np.std(seq_m14b)
    features['m14b_trend'] = seq_m14b[-1] - seq_m14b[-10]
    
    features['m14bsum_mean'] = np.mean(seq_m14bsum)
    features['m14bsum_max'] = np.max(seq_m14bsum)
    features['m14bsum_current'] = seq_m14bsum[-1]
    features['m14bsum_last10'] = np.mean(seq_m14bsum[-10:])
    features['m14bsum_last30'] = np.mean(seq_m14bsum[-30:])
    features['m14bsum_slope'] = np.polyfit(np.arange(n), seq_m14bsum, 1)[0]
    features['m14bsum_std'] = np.std(seq_m14bsum)
    features['m14bsum_trend'] = seq_m14bsum[-1] - seq_m14bsum[-10]
    
    features['total_mean'] = np.mean(seq_total)
    features['total_max'] = np.max(seq_total)
    features['total_min'] = np.min(seq_total)
    features['total_current'] = seq_total[-1]
    features['total_last5'] = np.mean(seq_total[-5:])
    features['total_last10'] = np.mean(seq_total[-10:])
    features['total_last30'] = np.mean(seq_total[-30:])
    features['total_slope'] = np.polyfit(np.arange(n), seq_total, 1)[0]
    features['total_std'] = np.std(seq_total)
    features['total_trend'] = seq_total[-1] - seq_total[-10]
    
    features['m10a_mean'] = np.mean(seq_m10a)
    features['m10a_max'] = np.max(seq_m10a)
    features['m10a_current'] = seq_m10a[-1]
    features['m10a_last10'] = np.mean(seq_m10a[-10:])
    features['m10a_slope'] = np.polyfit(np.arange(n), seq_m10a, 1)[0]
    
    features['trans_mean'] = np.mean(seq_trans)
    features['trans_max'] = np.max(seq_trans)
    features['trans_current'] = seq_trans[-1]
    features['trans_last10'] = np.mean(seq_trans[-10:])
    features['trans_slope'] = np.polyfit(np.arange(n), seq_trans, 1)[0]
    
    features['oht_mean'] = np.mean(seq_oht)
    features['oht_max'] = np.max(seq_oht)
    features['oht_current'] = seq_oht[-1]
    features['oht_last10'] = np.mean(seq_oht[-10:])
    
    features['gap_mean'] = np.mean(seq_gap)
    features['gap_max'] = np.max(seq_gap)
    features['gap_min'] = np.min(seq_gap)
    features['gap_current'] = seq_gap[-1]
    features['gap_last5'] = np.mean(seq_gap[-5:])
    features['gap_last10'] = np.mean(seq_gap[-10:])
    features['gap_last30'] = np.mean(seq_gap[-30:])
    features['gap_slope'] = np.polyfit(np.arange(n), seq_gap, 1)[0]
    features['gap_std'] = np.std(seq_gap)
    features['gap_trend'] = seq_gap[-1] - seq_gap[-10]
    
    features['m14b_x_sum'] = seq_m14b[-1] * seq_m14bsum[-1] / 1000
    features['sum_per_m14b'] = seq_m14bsum[-1] / (seq_m14b[-1] + 1)
    features['m14b_plus_sum'] = seq_m14b[-1] + seq_m14bsum[-1]
    features['gap_x_m14b'] = seq_gap[-1] * seq_m14b[-1] / 1000
    features['gap_x_total'] = seq_gap[-1] * seq_total[-1] / 10000
    features['ratio_m14b_total'] = seq_m14b[-1] / (seq_total[-1] + 1)
    features['ratio_sum_total'] = seq_m14bsum[-1] / (seq_total[-1] + 1)
    features['ratio_gap_total'] = seq_gap[-1] / (seq_total[-1] + 1)
    features['trans_x_gap'] = seq_trans[-1] * seq_gap[-1] / 100
    features['oht_x_total'] = seq_oht[-1] * seq_total[-1] / 10000
    
    features['m14b_over_400'] = np.sum(seq_m14b > 400)
    features['m14b_over_500'] = np.sum(seq_m14b > 500)
    features['m14bsum_over_500'] = np.sum(seq_m14bsum > 500)
    features['m14bsum_over_600'] = np.sum(seq_m14bsum > 600)
    features['total_over_1500'] = np.sum(seq_total >= 1500)
    features['total_over_1600'] = np.sum(seq_total >= 1600)
    features['total_over_1700'] = np.sum(seq_total >= 1700)
    features['gap_over_100'] = np.sum(seq_gap > 100)
    features['gap_over_200'] = np.sum(seq_gap > 200)
    features['gap_over_300'] = np.sum(seq_gap > 300)
    features['trans_over_100'] = np.sum(seq_trans > 100)
    features['trans_over_150'] = np.sum(seq_trans > 150)
    
    features['gold_pattern'] = 1 if (seq_m14b[-1] > 500 and seq_m14bsum[-1] > 600) else 0
    features['danger_gap'] = 1 if seq_gap[-1] > 250 else 0
    features['danger_total'] = 1 if seq_total[-1] >= 1600 else 0
    features['triple_check'] = 1 if (seq_m14b[-1] > 450 and seq_m14bsum[-1] > 550 and seq_gap[-1] > 200) else 0
    features['rising_fast'] = 1 if features['total_slope'] > 2 else 0
    features['gap_rising'] = 1 if features['gap_slope'] > 1 else 0
    
    return features

def adjust_prediction(pred, current_total, m14b, m14bsum, gap, trans):
    floor = current_total - 80
    if pred < floor:
        pred = floor
    if 1650 <= pred < 1700:
        boost = 0
        if m14b > 500 and m14bsum > 600:
            boost += 50
        elif m14b > 450 and m14bsum > 550:
            boost += 30
        if gap > 300:
            boost += 40
        elif gap > 250:
            boost += 25
        if trans > 150:
            boost += 20
        pred = pred + boost
    if current_total >= 1700 and pred < 1680:
        pred = max(pred, current_total - 50)
    return pred

# 모델 로드
print("\n📂 모델 로드...")
with open('model_v10_10min.pkl', 'rb') as f:
    model = pickle.load(f)
print("✅ model_v10_10min.pkl")

# 데이터 로드
print("\n📂 데이터 로드...")
files = glob.glob('/mnt/user-data/uploads/SO*')
dfs = []
for f in sorted(files):
    try:
        df = pd.read_csv(f, on_bad_lines='skip')
        dfs.append(df)
    except:
        pass
df = pd.concat(dfs, ignore_index=True)
df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M', errors='coerce')

required = ['TOTALCNT', 'M14AM14B', 'M14AM14BSUM', 'M10AM14A',
            'M14.QUE.ALL.TRANSPORT4MINOVERCNT', 'M14.QUE.OHT.OHTUTIL',
            'M14.QUE.ALL.CURRENTQCREATED', 'M14.QUE.ALL.CURRENTQCOMPLETED']
df = df.dropna(subset=required)
print(f"✅ 데이터: {len(df):,}행")

# 평가
print(f"\n🔄 평가 중...")
results = []
total = len(df) - SEQ_LEN - PRED_OFFSET

for i in range(SEQ_LEN, len(df) - PRED_OFFSET):
    row_dict = {
        'M14AM14B': df['M14AM14B'].iloc[i-SEQ_LEN:i].values,
        'M14AM14BSUM': df['M14AM14BSUM'].iloc[i-SEQ_LEN:i].values,
        'M10AM14A': df['M10AM14A'].iloc[i-SEQ_LEN:i].values,
        'TOTALCNT': df['TOTALCNT'].iloc[i-SEQ_LEN:i].values,
        'TRANSPORT': df['M14.QUE.ALL.TRANSPORT4MINOVERCNT'].iloc[i-SEQ_LEN:i].values,
        'OHT': df['M14.QUE.OHT.OHTUTIL'].iloc[i-SEQ_LEN:i].values,
        'Q_CREATED': df['M14.QUE.ALL.CURRENTQCREATED'].iloc[i-SEQ_LEN:i].values,
        'Q_COMPLETED': df['M14.QUE.ALL.CURRENTQCOMPLETED'].iloc[i-SEQ_LEN:i].values,
    }
    
    seq_total = row_dict['TOTALCNT']
    seq_m14b = row_dict['M14AM14B']
    seq_m14bsum = row_dict['M14AM14BSUM']
    seq_gap = row_dict['Q_CREATED'] - row_dict['Q_COMPLETED']
    seq_trans = row_dict['TRANSPORT']
    
    current_time = df['CURRTIME'].iloc[i-1]
    current_total = seq_total[-1]
    actual = df['TOTALCNT'].iloc[i + PRED_OFFSET - 1]
    
    features = create_features(row_dict)
    X = pd.DataFrame([features]).fillna(0).replace([np.inf, -np.inf], 0)
    
    pred_raw = model.predict(X)[0]
    pred = adjust_prediction(pred_raw, current_total, seq_m14b[-1], seq_m14bsum[-1], seq_gap[-1], seq_trans[-1])
    
    results.append({
        '현재시간': current_time.strftime('%Y-%m-%d %H:%M') if pd.notna(current_time) else '',
        '현재TOTALCNT': round(current_total, 2),
        '실제값': round(actual, 2),
        '원본예측': round(pred_raw, 2),
        '보정예측': round(pred, 2),
        '오차': round(actual - pred, 2),
        '절대오차': round(abs(actual - pred), 2),
        'M14AM14B': round(seq_m14b[-1], 2),
        'M14AM14BSUM': round(seq_m14bsum[-1], 2),
        'queue_gap': round(seq_gap[-1], 2),
        'TRANSPORT': round(seq_trans[-1], 2),
        '실제위험': 'O' if actual >= 1700 else '',
        '예측위험': 'O' if pred >= 1700 else ''
    })
    
    if (i - SEQ_LEN) % 5000 == 0:
        print(f"  {i-SEQ_LEN:,}/{total:,}")
        gc.collect()

print(f"✅ 완료!")

df_result = pd.DataFrame(results)
df_result.to_csv('evaluation_V10_10min.csv', index=False, encoding='utf-8-sig')
print(f"\n💾 저장: evaluation_V10_10min.csv")

# 통계
print("\n" + "="*70)
print("📊 V10.0 평가 통계")
print("="*70)
print(f"\n총 예측: {len(df_result):,}개")
print(f"MAE: {df_result['절대오차'].mean():.2f}")

actual_danger = df_result['실제위험'] == 'O'
pred_danger = df_result['예측위험'] == 'O'
actual_count = actual_danger.sum()

print(f"\n🔥 1700+ 감지:")
print(f"  실제 1700+: {actual_count}개")
if actual_count > 0:
    detected = (actual_danger & pred_danger).sum()
    print(f"  감지: {detected}개 ({detected/actual_count*100:.1f}%)")
    missed = actual_count - detected
    print(f"  미감지: {missed}개 ({missed/actual_count*100:.1f}%)")
    false_alarm = ((~actual_danger) & pred_danger).sum()
    print(f"  오탐: {false_alarm}개 ({false_alarm/(~actual_danger).sum()*100:.2f}%)")
    raw_detected = (df_result[actual_danger]['원본예측'] >= 1700).sum()
    print(f"\n📈 보정 효과:")
    print(f"  원본 감지율: {raw_detected/actual_count*100:.1f}%")
    print(f"  보정 감지율: {detected/actual_count*100:.1f}%")

print(f"\n📊 구간별 MAE:")
for low, high in [(0, 1400), (1400, 1600), (1600, 1700), (1700, 9999)]:
    mask = (df_result['실제값'] >= low) & (df_result['실제값'] < high)
    if mask.sum() > 0:
        mae = df_result.loc[mask, '절대오차'].mean()
        print(f"  {low:4d}~{high:4d}: {mask.sum():>6,}개, MAE={mae:.1f}")

print(f"\n✅ V10.0 평가 완료!")