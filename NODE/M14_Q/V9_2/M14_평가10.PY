#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 V9.1 평가 - 100분/10분 예측
패턴 = 경보 + Boost / 예측값 = ML!

수정:
- 패턴 매칭 → 경보 레벨만!
- 예측값 → 항상 ML + 패턴 boost!
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
print("🚀 V9.1 평가 - 100분/10분 예측")
print("   패턴=경보 / 예측값=ML+Boost!")
print("="*70)

# ============================================
# 설정
# ============================================
SEQ_LEN = 100
PRED_MINUTES = 10
DATA_FILE = "YOUR_CSV_FILE.csv"  # ← 평가용 데이터!

# ============================================
# 패턴 체크 (경보 레벨만!)
# ============================================
def check_pattern(m14b, m14bsum, gap, totalcnt):
    """
    패턴 → 경보 레벨 + Boost값 반환
    예측값은 ML이 담당!
    """
    # 1단계: 황금패턴 & gap>400 (100%)
    if m14b > 520 and m14bsum > 600 and gap > 400:
        return ('CRITICAL', 150)  # boost +150
    
    # 2단계: gap>500 (99.7%)
    if gap > 500:
        return ('CRITICAL', 130)
    
    # 3단계: Triple Check (96.5%)
    if m14b > 520 and m14bsum > 600 and gap > 250:
        return ('CRITICAL', 100)
    
    # 4단계: gap>400 (96.2%)
    if gap > 400:
        return ('HIGH', 80)
    
    # 5단계: 황금패턴 (gap 없이)
    if m14b > 520 and m14bsum > 600:
        return ('HIGH', 60)
    
    # 6단계: 현재값 높음
    if totalcnt >= 1680:
        return ('HIGH', 50)
    
    if totalcnt >= 1650:
        return ('WARNING', 30)
    
    if totalcnt >= 1600:
        return ('CAUTION', 15)
    
    return ('NORMAL', 0)

# ============================================
# Feature 생성 (34개)
# ============================================
def create_features_v9(row_dict):
    """필수 5개 컬럼 기반 34개 Feature"""
    features = {}
    
    seq_m14b = np.array(row_dict['M14B'])
    seq_m14bsum = np.array(row_dict['M14BSUM'])
    seq_totalcnt = np.array(row_dict['TOTALCNT'])
    seq_q_created = np.array(row_dict['Q_CREATED'])
    seq_q_completed = np.array(row_dict['Q_COMPLETED'])
    seq_gap = seq_q_created - seq_q_completed
    
    seq_len = len(seq_m14b)
    
    # TOTALCNT (8개)
    features['total_current'] = seq_totalcnt[-1]
    features['total_mean'] = np.mean(seq_totalcnt)
    features['total_max'] = np.max(seq_totalcnt)
    features['total_min'] = np.min(seq_totalcnt)
    features['total_last10'] = np.mean(seq_totalcnt[-10:])
    features['total_slope'] = np.polyfit(np.arange(seq_len), seq_totalcnt, 1)[0]
    features['total_std'] = np.std(seq_totalcnt)
    features['total_trend10'] = seq_totalcnt[-1] - seq_totalcnt[-10]
    
    # M14AM14B (6개)
    features['m14b_current'] = seq_m14b[-1]
    features['m14b_mean'] = np.mean(seq_m14b)
    features['m14b_max'] = np.max(seq_m14b)
    features['m14b_last10'] = np.mean(seq_m14b[-10:])
    features['m14b_slope'] = np.polyfit(np.arange(seq_len), seq_m14b, 1)[0]
    features['m14b_trend10'] = seq_m14b[-1] - seq_m14b[-10]
    
    # M14AM14BSUM (6개)
    features['m14bsum_current'] = seq_m14bsum[-1]
    features['m14bsum_mean'] = np.mean(seq_m14bsum)
    features['m14bsum_max'] = np.max(seq_m14bsum)
    features['m14bsum_last10'] = np.mean(seq_m14bsum[-10:])
    features['m14bsum_slope'] = np.polyfit(np.arange(seq_len), seq_m14bsum, 1)[0]
    features['m14bsum_trend10'] = seq_m14bsum[-1] - seq_m14bsum[-10]
    
    # Queue Gap (8개)
    features['gap_current'] = seq_gap[-1]
    features['gap_mean'] = np.mean(seq_gap)
    features['gap_max'] = np.max(seq_gap)
    features['gap_min'] = np.min(seq_gap)
    features['gap_last10'] = np.mean(seq_gap[-10:])
    features['gap_slope'] = np.polyfit(np.arange(seq_len), seq_gap, 1)[0]
    features['gap_std'] = np.std(seq_gap)
    features['gap_trend10'] = seq_gap[-1] - seq_gap[-10]
    
    # 조합 (6개)
    features['m14b_x_sum'] = seq_m14b[-1] * seq_m14bsum[-1] / 1000
    features['gap_x_m14b'] = seq_gap[-1] * seq_m14b[-1] / 1000
    features['ratio_gap_total'] = seq_gap[-1] / (seq_totalcnt[-1] + 1)
    features['ratio_m14b_total'] = seq_m14b[-1] / (seq_totalcnt[-1] + 1)
    features['m14b_plus_sum'] = seq_m14b[-1] + seq_m14bsum[-1]
    features['gap_per_m14b'] = seq_gap[-1] / (seq_m14b[-1] + 1)
    
    return features

# ============================================
# 예측 보정
# ============================================
def adjust_prediction(pred_raw, current_total, boost, level):
    """ML 예측값 보정"""
    pred = pred_raw
    
    # 1. 패턴 boost 적용
    if boost > 0:
        # 1650 이상일 때만 boost 적용 (과도한 boost 방지)
        if pred >= 1600:
            pred = pred + boost
        elif pred >= 1550:
            pred = pred + boost * 0.5
    
    # 2. 하한선 (10분 = -80)
    floor = current_total - 80
    if pred < floor:
        pred = floor
    
    # 3. 상한선 (비현실적 예측 방지)
    ceiling = current_total + 200
    if pred > ceiling:
        pred = ceiling
    
    return pred

# ============================================
# 모델 로드
# ============================================
model = None
model_files = ['model_v9_100min_10min.pkl', 'model_v9.pkl']
for mf in model_files:
    if os.path.exists(mf):
        with open(mf, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ 모델: {mf}")
        break

if not model:
    print("❌ 모델 없음!")
    exit(1)

# ============================================
# 데이터 로드
# ============================================
df = pd.read_csv(DATA_FILE, on_bad_lines='skip')
print(f"✅ 데이터: {len(df):,}행")

# CURRTIME 파싱
if 'CURRTIME' in df.columns:
    df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M', errors='coerce')

# queue_gap 계산
df['queue_gap'] = df['M14.QUE.ALL.CURRENTQCREATED'] - df['M14.QUE.ALL.CURRENTQCOMPLETED']

# ============================================
# 평가 실행
# ============================================
results = []
level_counts = {'CRITICAL': 0, 'HIGH': 0, 'WARNING': 0, 'CAUTION': 0, 'NORMAL': 0}

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
    
    # 현재 상태
    current_total = row_dict['TOTALCNT'][-1]
    current_m14b = row_dict['M14B'][-1]
    current_m14bsum = row_dict['M14BSUM'][-1]
    current_gap = row_dict['Q_CREATED'][-1] - row_dict['Q_COMPLETED'][-1]
    
    # 시간 정보
    if 'CURRTIME' in df.columns and pd.notna(df['CURRTIME'].iloc[i-1]):
        current_time = df['CURRTIME'].iloc[i-1]
        prediction_time = current_time + timedelta(minutes=PRED_MINUTES)
        time_str = current_time.strftime('%Y-%m-%d %H:%M')
        pred_time_str = prediction_time.strftime('%Y-%m-%d %H:%M')
    else:
        time_str = ''
        pred_time_str = ''
    
    # 실제값
    actual = df['TOTALCNT'].iloc[i + PRED_MINUTES - 1]
    
    # ========== 패턴 체크 (경보 레벨만!) ==========
    level, boost = check_pattern(current_m14b, current_m14bsum, current_gap, current_total)
    level_counts[level] += 1
    
    # ========== ML 예측 (항상!) ==========
    features = create_features_v9(row_dict)
    X = pd.DataFrame([features])
    pred_raw = model.predict(X)[0]
    
    # ========== 보정 (ML + boost) ==========
    pred = adjust_prediction(pred_raw, current_total, boost, level)
    
    # 결과 저장
    results.append({
        '현재시간': time_str,
        '예측시점': pred_time_str,
        '현재TOTALCNT': round(current_total, 2),
        'ML원본': round(pred_raw, 2),
        'Boost': boost,
        '최종예측': round(pred, 2),
        '실제값': round(actual, 2),
        '오차': round(actual - pred, 2),
        '절대오차': round(abs(actual - pred), 2),
        '경보레벨': level,
        'M14B': round(current_m14b, 2),
        'M14BSUM': round(current_m14bsum, 2),
        'queue_gap': round(current_gap, 2),
        '실제1700+': 'O' if actual >= 1700 else '',
        '예측1700+': 'O' if pred >= 1700 else ''
    })
    
    if (i - SEQ_LEN) % 2000 == 0:
        print(f"  {i-SEQ_LEN:,}/{total:,}")
        gc.collect()

print(f"\n✅ 완료!")
print(f"\n📊 경보 레벨 분포:")
for level, count in level_counts.items():
    if count > 0:
        print(f"  {level}: {count:,}개 ({count/total*100:.1f}%)")

# ============================================
# 결과 저장
# ============================================
df_result = pd.DataFrame(results)
output_file = 'evaluation_V91_100min_10min.csv'
df_result.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n💾 저장: {output_file}")

# ============================================
# 통계 분석
# ============================================
print("\n" + "="*70)
print("📊 V9.1 평가 통계")
print("="*70)

print(f"\n총 예측: {len(df_result):,}개")
print(f"MAE (전체): {df_result['절대오차'].mean():.2f}")
print(f"MAE (ML원본): {abs(df_result['실제값'] - df_result['ML원본']).mean():.2f}")

# 1700+ 감지 통계
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
    print(f"  ✅ 감지(Recall): {detected:,}/{actual_count:,} ({detected/actual_count*100:.1f}%)")
    print(f"  ❌ 미감지: {missed:,}개")
if pred_count > 0:
    print(f"  🎯 정밀도(Precision): {detected:,}/{pred_count:,} ({detected/pred_count*100:.1f}%)")
print(f"  ⚠️ 오탐(False Alarm): {false_alarm:,}개")

# 경보 레벨별 정확도
print(f"\n📊 경보 레벨별 1700+ 실제 발생률:")
for level in ['CRITICAL', 'HIGH', 'WARNING', 'CAUTION', 'NORMAL']:
    subset = df_result[df_result['경보레벨'] == level]
    if len(subset) > 0:
        actual_1700 = (subset['실제1700+'] == 'O').sum()
        rate = actual_1700 / len(subset) * 100
        print(f"  {level}: {actual_1700:,}/{len(subset):,} ({rate:.1f}%)")

# Boost 효과
print(f"\n📊 Boost 효과:")
boosted = df_result[df_result['Boost'] > 0]
if len(boosted) > 0:
    mae_raw = abs(boosted['실제값'] - boosted['ML원본']).mean()
    mae_final = boosted['절대오차'].mean()
    print(f"  Boost 적용: {len(boosted):,}개")
    print(f"  ML원본 MAE: {mae_raw:.2f}")
    print(f"  Boost후 MAE: {mae_final:.2f}")
    print(f"  개선: {mae_raw - mae_final:+.2f}")

# 미감지 분석
if missed > 0:
    print(f"\n📊 미감지 {missed}개 분석:")
    missed_df = df_result[actual_danger & ~pred_danger]
    print(f"  현재값 평균: {missed_df['현재TOTALCNT'].mean():.0f}")
    print(f"  ML원본 평균: {missed_df['ML원본'].mean():.0f}")
    print(f"  최종예측 평균: {missed_df['최종예측'].mean():.0f}")
    print(f"  실제값 평균: {missed_df['실제값'].mean():.0f}")
    
    # 현재값 분포
    print(f"\n  현재값 분포 (미감지):")
    for thresh in [1600, 1650, 1680]:
        cnt = (missed_df['현재TOTALCNT'] >= thresh).sum()
        print(f"    >= {thresh}: {cnt}개 ({cnt/missed*100:.1f}%)")

print(f"\n" + "="*70)
print(f"✅ V9.1 평가 완료!")
print("="*70)