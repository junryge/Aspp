#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 V9.2 평가 - 100분/10분 예측
패턴을 Feature로! (학습과 동일!)
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
print("🚀 V9.2 평가 - 100분/10분 예측")
print("   패턴 Feature 포함!")
print("="*70)

# ============================================
# 설정
# ============================================
SEQ_LEN = 100
PRED_MINUTES = 10
DATA_FILE = "YOUR_CSV_FILE.csv"  # ← 평가 데이터!

# ============================================
# Feature 생성 (40개 = 학습과 동일!)
# ============================================
def create_features_v92(row_dict):
    """34개 기본 + 6개 패턴 Feature"""
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
    
    # 🔥 패턴 Feature (6개)
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
model = None
model_files = ['model_v92_100min_10min.pkl']
for mf in model_files:
    if os.path.exists(mf):
        with open(mf, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ 모델: {mf}")
        break

if not model:
    print("❌ 모델 없음! 먼저 V92_학습 실행!")
    exit(1)

# ============================================
# 데이터 로드
# ============================================
df = pd.read_csv(DATA_FILE, on_bad_lines='skip')
print(f"✅ 데이터: {len(df):,}행")

if 'CURRTIME' in df.columns:
    df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M', errors='coerce')

df['queue_gap'] = df['M14.QUE.ALL.CURRENTQCREATED'] - df['M14.QUE.ALL.CURRENTQCOMPLETED']

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
    
    # Feature 생성 (패턴 포함!)
    features = create_features_v92(row_dict)
    X = pd.DataFrame([features])
    
    # ML 예측
    pred = model.predict(X)[0]
    
    # 하한선만
    floor = current_total - 80
    if pred < floor:
        pred = floor
    
    # 패턴 정보 (출력용)
    is_triple = features['is_triple_check']
    is_gold = features['is_gold_pattern']
    is_gap_crit = features['is_gap_critical']
    
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
        'Triple체크': 'O' if is_triple else '',
        '황금패턴': 'O' if is_gold else '',
        'Gap위험': 'O' if is_gap_crit else '',
        '실제1700+': 'O' if actual >= 1700 else '',
        '예측1700+': 'O' if pred >= 1700 else ''
    })
    
    if (i - SEQ_LEN) % 2000 == 0:
        print(f"  {i-SEQ_LEN:,}/{total:,}")
        gc.collect()

print(f"✅ 완료!")

# ============================================
# 결과 저장
# ============================================
df_result = pd.DataFrame(results)
output_file = 'evaluation_V92_100min_10min.csv'
df_result.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n💾 저장: {output_file}")

# ============================================
# 통계
# ============================================
print("\n" + "="*70)
print("📊 V9.2 평가 통계")
print("="*70)

print(f"\n총 예측: {len(df_result):,}개")
print(f"MAE: {df_result['절대오차'].mean():.2f}")
print(f"RMSE: {np.sqrt((df_result['오차']**2).mean()):.2f}")

# 1700+ 감지
actual_danger = df_result['실제1700+'] == 'O'
pred_danger = df_result['예측1700+'] == 'O'
actual_count = actual_danger.sum()
pred_count = pred_danger.sum()
detected = (actual_danger & pred_danger).sum()
false_alarm = (pred_danger & ~actual_danger).sum()
missed = (actual_danger & ~pred_danger).sum()

print(f"\n🔥 1700+ 감지:")
print(f"  실제 1700+: {actual_count:,}개")
print(f"  예측 1700+: {pred_count:,}개")
if actual_count > 0:
    print(f"  ✅ 감지율(Recall): {detected}/{actual_count} ({detected/actual_count*100:.1f}%)")
    print(f"  ❌ 미감지: {missed}개")
if pred_count > 0:
    print(f"  🎯 정밀도(Precision): {detected}/{pred_count} ({detected/pred_count*100:.1f}%)")
print(f"  ⚠️ 오탐: {false_alarm}개")

# 패턴별 분석
print(f"\n📊 패턴별 실제 1700+ 발생률:")
triple = df_result['Triple체크'] == 'O'
gold = df_result['황금패턴'] == 'O'
gap_crit = df_result['Gap위험'] == 'O'

for name, mask in [('Triple체크', triple), ('황금패턴', gold), ('Gap위험', gap_crit)]:
    subset = df_result[mask]
    if len(subset) > 0:
        actual_1700 = (subset['실제1700+'] == 'O').sum()
        print(f"  {name}: {actual_1700}/{len(subset)} ({actual_1700/len(subset)*100:.1f}%)")

# 패턴 없는 경우
no_pattern = ~triple & ~gold & ~gap_crit
subset = df_result[no_pattern]
if len(subset) > 0:
    actual_1700 = (subset['실제1700+'] == 'O').sum()
    print(f"  패턴없음: {actual_1700}/{len(subset)} ({actual_1700/len(subset)*100:.2f}%)")

# 미감지 분석
if missed > 0:
    print(f"\n📊 미감지 {missed}개 분석:")
    missed_df = df_result[actual_danger & ~pred_danger]
    print(f"  현재값 평균: {missed_df['현재TOTALCNT'].mean():.0f}")
    print(f"  예측값 평균: {missed_df['예측값'].mean():.0f}")
    print(f"  실제값 평균: {missed_df['실제값'].mean():.0f}")
    
    # 패턴 분포
    print(f"\n  미감지 중 패턴 분포:")
    print(f"    Triple체크: {(missed_df['Triple체크']=='O').sum()}개")
    print(f"    황금패턴: {(missed_df['황금패턴']=='O').sum()}개")
    print(f"    Gap위험: {(missed_df['Gap위험']=='O').sum()}개")

print(f"\n" + "="*70)
print(f"✅ V9.2 평가 완료!")
print("="*70)