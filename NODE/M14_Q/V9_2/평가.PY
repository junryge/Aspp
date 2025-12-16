#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 V9.4 평가
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
print("🚀 V9.4 평가")
print("="*70)

# ============================================
# 설정
# ============================================
SEQ_LEN = 100
PRED_MINUTES = 10
DATA_FILE = "YOUR_EVAL_FILE.csv"  # ← 평가 데이터!
MODEL_FILE = "model_v94.pkl"

# ============================================
# Feature 생성
# ============================================
def create_features_v94(row_dict):
    """40개 Feature"""
    features = {}
    
    seq_m14b = np.array(row_dict['M14B'])
    seq_m14bsum = np.array(row_dict['M14BSUM'])
    seq_totalcnt = np.array(row_dict['TOTALCNT'])
    seq_q_created = np.array(row_dict['Q_CREATED'])
    seq_q_completed = np.array(row_dict['Q_COMPLETED'])
    seq_gap = seq_q_created - seq_q_completed
    
    seq_len = len(seq_m14b)
    
    cur_m14b = seq_m14b[-1]
    cur_m14bsum = seq_m14bsum[-1]
    cur_gap = seq_gap[-1]
    cur_total = seq_totalcnt[-1]
    
    # TOTALCNT (8개)
    features['total_current'] = cur_total
    features['total_mean'] = np.mean(seq_totalcnt)
    features['total_max'] = np.max(seq_totalcnt)
    features['total_min'] = np.min(seq_totalcnt)
    features['total_last10'] = np.mean(seq_totalcnt[-10:])
    features['total_slope'] = np.polyfit(np.arange(seq_len), seq_totalcnt, 1)[0]
    features['total_std'] = np.std(seq_totalcnt)
    features['total_trend10'] = seq_totalcnt[-1] - seq_totalcnt[-10]
    
    # M14AM14B (6개)
    features['m14b_current'] = cur_m14b
    features['m14b_mean'] = np.mean(seq_m14b)
    features['m14b_max'] = np.max(seq_m14b)
    features['m14b_last10'] = np.mean(seq_m14b[-10:])
    features['m14b_slope'] = np.polyfit(np.arange(seq_len), seq_m14b, 1)[0]
    features['m14b_trend10'] = seq_m14b[-1] - seq_m14b[-10]
    
    # M14AM14BSUM (6개)
    features['m14bsum_current'] = cur_m14bsum
    features['m14bsum_mean'] = np.mean(seq_m14bsum)
    features['m14bsum_max'] = np.max(seq_m14bsum)
    features['m14bsum_last10'] = np.mean(seq_m14bsum[-10:])
    features['m14bsum_slope'] = np.polyfit(np.arange(seq_len), seq_m14bsum, 1)[0]
    features['m14bsum_trend10'] = seq_m14bsum[-1] - seq_m14bsum[-10]
    
    # Queue Gap (8개)
    features['gap_current'] = cur_gap
    features['gap_mean'] = np.mean(seq_gap)
    features['gap_max'] = np.max(seq_gap)
    features['gap_min'] = np.min(seq_gap)
    features['gap_last10'] = np.mean(seq_gap[-10:])
    features['gap_slope'] = np.polyfit(np.arange(seq_len), seq_gap, 1)[0]
    features['gap_std'] = np.std(seq_gap)
    features['gap_trend10'] = seq_gap[-1] - seq_gap[-10]
    
    # 조합 (6개)
    features['m14b_x_sum'] = cur_m14b * cur_m14bsum / 1000
    features['gap_x_m14b'] = cur_gap * cur_m14b / 1000
    features['ratio_gap_total'] = cur_gap / (cur_total + 1)
    features['ratio_m14b_total'] = cur_m14b / (cur_total + 1)
    features['m14b_plus_sum'] = cur_m14b + cur_m14bsum
    features['gap_per_m14b'] = cur_gap / (cur_m14b + 1)
    
    # 패턴 (6개)
    features['is_gold_pattern'] = 1 if (cur_m14b > 520 and cur_m14bsum > 600) else 0
    features['is_triple_check'] = 1 if (cur_m14b > 520 and cur_m14bsum > 600 and cur_gap > 250) else 0
    features['is_gap_critical'] = 1 if cur_gap > 400 else 0
    features['is_gap_extreme'] = 1 if cur_gap > 500 else 0
    features['is_near_danger'] = 1 if cur_total >= 1650 else 0
    features['is_in_danger'] = 1 if cur_total >= 1700 else 0
    
    return features

# ============================================
# 모델 로드
# ============================================
if not os.path.exists(MODEL_FILE):
    print(f"❌ 모델 없음: {MODEL_FILE}")
    exit(1)

with open(MODEL_FILE, 'rb') as f:
    model = pickle.load(f)
print(f"✅ 모델: {MODEL_FILE}")

# ============================================
# 데이터 로드
# ============================================
df = pd.read_csv(DATA_FILE, on_bad_lines='skip')
print(f"✅ 데이터: {len(df):,}행")

if 'CURRTIME' in df.columns:
    df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M', errors='coerce')

df['queue_gap'] = df['M14.QUE.ALL.CURRENTQCREATED'] - df['M14.QUE.ALL.CURRENTQCOMPLETED']

print(f"\n📊 평가 데이터:")
print(f"  1700+: {(df['TOTALCNT'] >= 1700).sum()}개")
print(f"  1800+: {(df['TOTALCNT'] >= 1800).sum()}개")

# ============================================
# 평가 실행
# ============================================
results = []
total = len(df) - SEQ_LEN - PRED_MINUTES
print(f"\n🔄 평가 중... {total:,}개")

for i in range(SEQ_LEN, len(df) - PRED_MINUTES):
    row_dict = {
        'TOTALCNT': df['TOTALCNT'].iloc[i-SEQ_LEN:i].values,
        'M14B': df['M14AM14B'].iloc[i-SEQ_LEN:i].values,
        'M14BSUM': df['M14AM14BSUM'].iloc[i-SEQ_LEN:i].values,
        'Q_CREATED': df['M14.QUE.ALL.CURRENTQCREATED'].iloc[i-SEQ_LEN:i].values,
        'Q_COMPLETED': df['M14.QUE.ALL.CURRENTQCOMPLETED'].iloc[i-SEQ_LEN:i].values,
    }
    
    current_total = row_dict['TOTALCNT'][-1]
    current_m14b = row_dict['M14B'][-1]
    current_m14bsum = row_dict['M14BSUM'][-1]
    current_gap = row_dict['Q_CREATED'][-1] - row_dict['Q_COMPLETED'][-1]
    
    if 'CURRTIME' in df.columns and pd.notna(df['CURRTIME'].iloc[i-1]):
        current_time = df['CURRTIME'].iloc[i-1]
        time_str = current_time.strftime('%Y-%m-%d %H:%M')
        pred_time_str = (current_time + timedelta(minutes=PRED_MINUTES)).strftime('%Y-%m-%d %H:%M')
    else:
        time_str = ''
        pred_time_str = ''
    
    actual = df['TOTALCNT'].iloc[i + PRED_MINUTES - 1]
    
    # Feature + 예측
    features = create_features_v94(row_dict)
    X = pd.DataFrame([features])
    pred = model.predict(X)[0]
    
    # 하한선
    floor = current_total - 80
    if pred < floor:
        pred = floor
    
    # 패턴
    is_gold = (current_m14b > 520 and current_m14bsum > 600)
    is_triple = is_gold and current_gap > 250
    
    results.append({
        '현재시간': time_str,
        '예측시점': pred_time_str,
        '현재TOTALCNT': round(current_total, 2),
        '예측값': round(pred, 2),
        '실제값': round(actual, 2),
        '오차': round(actual - pred, 2),
        '절대오차': round(abs(actual - pred), 2),
        'M14B': round(current_m14b, 2),
        'M14BSUM': round(current_m14bsum, 2),
        'queue_gap': round(current_gap, 2),
        '황금패턴': 'O' if is_gold else '',
        'Triple': 'O' if is_triple else '',
        '현재1650+': 'O' if current_total >= 1650 else '',
        '실제1700+': 'O' if actual >= 1700 else '',
        '예측1700+': 'O' if pred >= 1700 else '',
        '실제1800+': 'O' if actual >= 1800 else ''
    })
    
    if (i - SEQ_LEN) % 2000 == 0:
        print(f"  {i-SEQ_LEN:,}/{total:,}")
        gc.collect()

print(f"✅ 완료!")

# ============================================
# 저장
# ============================================
df_result = pd.DataFrame(results)
output_file = 'evaluation_V94.csv'
df_result.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n💾 저장: {output_file}")

# ============================================
# 통계
# ============================================
print("\n" + "="*70)
print("📊 V9.4 평가 통계")
print("="*70)

print(f"\n총 예측: {len(df_result):,}개")
print(f"MAE: {df_result['절대오차'].mean():.2f}")
print(f"RMSE: {np.sqrt((df_result['오차']**2).mean()):.2f}")

# 구간별 MAE
print(f"\n📊 구간별 MAE:")
for low, high, name in [(0, 1400, '~1400'), (1400, 1600, '1400~1600'),
                        (1600, 1700, '1600~1700'), (1700, 1800, '1700~1800'),
                        (1800, 9999, '1800+')]:
    mask = (df_result['실제값'] >= low) & (df_result['실제값'] < high)
    if mask.sum() > 0:
        seg_mae = df_result.loc[mask, '절대오차'].mean()
        print(f"  {name}: MAE={seg_mae:.2f} ({mask.sum()}개)")

# 1700+ 감지
actual_1700 = df_result['실제1700+'] == 'O'
pred_1700 = df_result['예측1700+'] == 'O'
pred_1680 = df_result['예측값'] >= 1680
pred_1650 = df_result['예측값'] >= 1650

actual_count = actual_1700.sum()
detected_1700 = (actual_1700 & pred_1700).sum()
detected_1680 = (actual_1700 & pred_1680).sum()
detected_1650 = (actual_1700 & pred_1650).sum()
false_alarm = (pred_1700 & ~actual_1700).sum()

print(f"\n🔥 1700+ 감지:")
print(f"  실제: {actual_count}개")
print(f"\n  감지율:")
print(f"    예측>=1700: {detected_1700}/{actual_count} ({detected_1700/actual_count*100:.1f}%)" if actual_count > 0 else "")
print(f"    예측>=1680: {detected_1680}/{actual_count} ({detected_1680/actual_count*100:.1f}%)" if actual_count > 0 else "")
print(f"    예측>=1650: {detected_1650}/{actual_count} ({detected_1650/actual_count*100:.1f}%)" if actual_count > 0 else "")

pred_count = pred_1700.sum()
if pred_count > 0:
    print(f"\n  정밀도: {detected_1700}/{pred_count} ({detected_1700/pred_count*100:.1f}%)")

print(f"  오탐: {false_alarm}개")

# 1800+ 감지
actual_1800 = df_result['실제1800+'] == 'O'
if actual_1800.sum() > 0:
    pred_1750 = df_result['예측값'] >= 1750
    detected_1800 = (actual_1800 & pred_1750).sum()
    print(f"\n🔥 1800+ 감지:")
    print(f"  실제: {actual_1800.sum()}개")
    print(f"  예측>=1750: {detected_1800}/{actual_1800.sum()} ({detected_1800/actual_1800.sum()*100:.1f}%)")

# 패턴 분석
print(f"\n📊 패턴별 1700+ 실제 발생률:")
gold = df_result['황금패턴'] == 'O'
cur_1650 = df_result['현재1650+'] == 'O'

for name, mask in [('황금패턴', gold), ('현재>=1650', cur_1650)]:
    if mask.sum() > 0:
        actual_in = (df_result.loc[mask, '실제1700+'] == 'O').sum()
        print(f"  {name}: {actual_in}/{mask.sum()} ({actual_in/mask.sum()*100:.1f}%)")

# 미감지 분석
missed = actual_1700 & ~pred_1700
if missed.sum() > 0:
    print(f"\n📊 미감지 {missed.sum()}개:")
    missed_df = df_result[missed]
    print(f"  현재값 평균: {missed_df['현재TOTALCNT'].mean():.0f}")
    print(f"  예측값 평균: {missed_df['예측값'].mean():.0f}")
    print(f"  실제값 평균: {missed_df['실제값'].mean():.0f}")

print(f"\n" + "="*70)
print(f"✅ V9.4 평가 완료!")
print("="*70)