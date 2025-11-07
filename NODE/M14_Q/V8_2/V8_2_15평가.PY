#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 평가 - 280분 → 15분 후 예측
"""

import numpy as np
import pandas as pd
import pickle
from datetime import timedelta
import os
import gc
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🚀 평가 - 280분 → 15분 후 예측")
print("="*80)

def create_features_13col_optimized(row_dict):
    """Feature 생성 (학습과 100% 동일)"""
    features = {}
    
    seq_m14b = np.array(row_dict['M14AM14B'])
    seq_m14b_sum = np.array(row_dict['M14AM14BSUM'])
    seq_m14b_rev = np.array(row_dict['M14BM14A'])
    seq_m10a = np.array(row_dict['M14AM10A'])
    seq_m10a_rev = np.array(row_dict['M10AM14A'])
    seq_m16_rev = np.array(row_dict['M16M14A'])
    seq_m16_sum = np.array(row_dict['M14AM16SUM'])
    seq_totalcnt = np.array(row_dict['TOTALCNT'])
    seq_q_created = np.array(row_dict['M14.QUE.ALL.CURRENTQCREATED'])
    seq_q_completed = np.array(row_dict['M14.QUE.ALL.CURRENTQCOMPLETED'])
    seq_oht = np.array(row_dict['M14.QUE.OHT.OHTUTIL'])
    seq_transport = np.array(row_dict['M14.QUE.ALL.TRANSPORT4MINOVERCNT'])
    seq_queue_gap = seq_q_created - seq_q_completed
    
    # M14AM14B (10)
    features['m14b_mean'] = np.mean(seq_m14b)
    features['m14b_std'] = np.std(seq_m14b)
    features['m14b_max'] = np.max(seq_m14b)
    features['m14b_min'] = np.min(seq_m14b)
    features['m14b_current'] = seq_m14b[-1]
    features['m14b_last_5'] = np.mean(seq_m14b[-5:])
    features['m14b_last_10'] = np.mean(seq_m14b[-10:])
    features['m14b_last_30'] = np.mean(seq_m14b[-30:])
    features['m14b_slope'] = np.polyfit(np.arange(280), seq_m14b, 1)[0]
    features['m14b_range'] = np.max(seq_m14b) - np.min(seq_m14b)
    
    # M14AM14BSUM (10)
    features['m14bsum_mean'] = np.mean(seq_m14b_sum)
    features['m14bsum_std'] = np.std(seq_m14b_sum)
    features['m14bsum_max'] = np.max(seq_m14b_sum)
    features['m14bsum_min'] = np.min(seq_m14b_sum)
    features['m14bsum_current'] = seq_m14b_sum[-1]
    features['m14bsum_last_5'] = np.mean(seq_m14b_sum[-5:])
    features['m14bsum_last_10'] = np.mean(seq_m14b_sum[-10:])
    features['m14bsum_last_30'] = np.mean(seq_m14b_sum[-30:])
    features['m14bsum_slope'] = np.polyfit(np.arange(280), seq_m14b_sum, 1)[0]
    features['m14bsum_range'] = np.max(seq_m14b_sum) - np.min(seq_m14b_sum)
    
    # M14BM14A (10)
    features['m14brev_mean'] = np.mean(seq_m14b_rev)
    features['m14brev_std'] = np.std(seq_m14b_rev)
    features['m14brev_max'] = np.max(seq_m14b_rev)
    features['m14brev_min'] = np.min(seq_m14b_rev)
    features['m14brev_current'] = seq_m14b_rev[-1]
    features['m14brev_last_5'] = np.mean(seq_m14b_rev[-5:])
    features['m14brev_last_10'] = np.mean(seq_m14b_rev[-10:])
    features['m14brev_last_30'] = np.mean(seq_m14b_rev[-30:])
    features['m14brev_slope'] = np.polyfit(np.arange(280), seq_m14b_rev, 1)[0]
    features['m14brev_range'] = np.max(seq_m14b_rev) - np.min(seq_m14b_rev)
    
    # M14AM10A (10)
    features['m10a_mean'] = np.mean(seq_m10a)
    features['m10a_std'] = np.std(seq_m10a)
    features['m10a_max'] = np.max(seq_m10a)
    features['m10a_min'] = np.min(seq_m10a)
    features['m10a_current'] = seq_m10a[-1]
    features['m10a_last_5'] = np.mean(seq_m10a[-5:])
    features['m10a_last_10'] = np.mean(seq_m10a[-10:])
    features['m10a_last_30'] = np.mean(seq_m10a[-30:])
    features['m10a_slope'] = np.polyfit(np.arange(280), seq_m10a, 1)[0]
    features['m10a_range'] = np.max(seq_m10a) - np.min(seq_m10a)
    
    # M10AM14A (10)
    features['m10arev_mean'] = np.mean(seq_m10a_rev)
    features['m10arev_std'] = np.std(seq_m10a_rev)
    features['m10arev_max'] = np.max(seq_m10a_rev)
    features['m10arev_min'] = np.min(seq_m10a_rev)
    features['m10arev_current'] = seq_m10a_rev[-1]
    features['m10arev_last_5'] = np.mean(seq_m10a_rev[-5:])
    features['m10arev_last_10'] = np.mean(seq_m10a_rev[-10:])
    features['m10arev_last_30'] = np.mean(seq_m10a_rev[-30:])
    features['m10arev_slope'] = np.polyfit(np.arange(280), seq_m10a_rev, 1)[0]
    features['m10arev_range'] = np.max(seq_m10a_rev) - np.min(seq_m10a_rev)
    
    # M16M14A (10)
    features['m16rev_mean'] = np.mean(seq_m16_rev)
    features['m16rev_std'] = np.std(seq_m16_rev)
    features['m16rev_max'] = np.max(seq_m16_rev)
    features['m16rev_min'] = np.min(seq_m16_rev)
    features['m16rev_current'] = seq_m16_rev[-1]
    features['m16rev_last_5'] = np.mean(seq_m16_rev[-5:])
    features['m16rev_last_10'] = np.mean(seq_m16_rev[-10:])
    features['m16rev_last_30'] = np.mean(seq_m16_rev[-30:])
    features['m16rev_slope'] = np.polyfit(np.arange(280), seq_m16_rev, 1)[0]
    features['m16rev_range'] = np.max(seq_m16_rev) - np.min(seq_m16_rev)
    
    # M14AM16SUM (10)
    features['m16sum_mean'] = np.mean(seq_m16_sum)
    features['m16sum_std'] = np.std(seq_m16_sum)
    features['m16sum_max'] = np.max(seq_m16_sum)
    features['m16sum_min'] = np.min(seq_m16_sum)
    features['m16sum_current'] = seq_m16_sum[-1]
    features['m16sum_last_5'] = np.mean(seq_m16_sum[-5:])
    features['m16sum_last_10'] = np.mean(seq_m16_sum[-10:])
    features['m16sum_last_30'] = np.mean(seq_m16_sum[-30:])
    features['m16sum_slope'] = np.polyfit(np.arange(280), seq_m16_sum, 1)[0]
    features['m16sum_range'] = np.max(seq_m16_sum) - np.min(seq_m16_sum)
    
    # TOTALCNT (10)
    features['total_mean'] = np.mean(seq_totalcnt)
    features['total_std'] = np.std(seq_totalcnt)
    features['total_max'] = np.max(seq_totalcnt)
    features['total_min'] = np.min(seq_totalcnt)
    features['total_current'] = seq_totalcnt[-1]
    features['total_last_5'] = np.mean(seq_totalcnt[-5:])
    features['total_last_10'] = np.mean(seq_totalcnt[-10:])
    features['total_last_30'] = np.mean(seq_totalcnt[-30:])
    features['total_slope'] = np.polyfit(np.arange(280), seq_totalcnt, 1)[0]
    features['total_range'] = np.max(seq_totalcnt) - np.min(seq_totalcnt)
    
    # Queue Created (10)
    features['qc_mean'] = np.mean(seq_q_created)
    features['qc_std'] = np.std(seq_q_created)
    features['qc_max'] = np.max(seq_q_created)
    features['qc_min'] = np.min(seq_q_created)
    features['qc_current'] = seq_q_created[-1]
    features['qc_last_5'] = np.mean(seq_q_created[-5:])
    features['qc_last_10'] = np.mean(seq_q_created[-10:])
    features['qc_last_30'] = np.mean(seq_q_created[-30:])
    features['qc_slope'] = np.polyfit(np.arange(280), seq_q_created, 1)[0]
    features['qc_range'] = np.max(seq_q_created) - np.min(seq_q_created)
    
    # Queue Completed (10)
    features['qd_mean'] = np.mean(seq_q_completed)
    features['qd_std'] = np.std(seq_q_completed)
    features['qd_max'] = np.max(seq_q_completed)
    features['qd_min'] = np.min(seq_q_completed)
    features['qd_current'] = seq_q_completed[-1]
    features['qd_last_5'] = np.mean(seq_q_completed[-5:])
    features['qd_last_10'] = np.mean(seq_q_completed[-10:])
    features['qd_last_30'] = np.mean(seq_q_completed[-30:])
    features['qd_slope'] = np.polyfit(np.arange(280), seq_q_completed, 1)[0]
    features['qd_range'] = np.max(seq_q_completed) - np.min(seq_q_completed)
    
    # OHT (10)
    features['oht_mean'] = np.mean(seq_oht)
    features['oht_std'] = np.std(seq_oht)
    features['oht_max'] = np.max(seq_oht)
    features['oht_min'] = np.min(seq_oht)
    features['oht_current'] = seq_oht[-1]
    features['oht_last_5'] = np.mean(seq_oht[-5:])
    features['oht_last_10'] = np.mean(seq_oht[-10:])
    features['oht_last_30'] = np.mean(seq_oht[-30:])
    features['oht_slope'] = np.polyfit(np.arange(280), seq_oht, 1)[0]
    features['oht_range'] = np.max(seq_oht) - np.min(seq_oht)
    
    # Transport (10)
    features['trans_mean'] = np.mean(seq_transport)
    features['trans_std'] = np.std(seq_transport)
    features['trans_max'] = np.max(seq_transport)
    features['trans_min'] = np.min(seq_transport)
    features['trans_current'] = seq_transport[-1]
    features['trans_last_5'] = np.mean(seq_transport[-5:])
    features['trans_last_10'] = np.mean(seq_transport[-10:])
    features['trans_last_30'] = np.mean(seq_transport[-30:])
    features['trans_slope'] = np.polyfit(np.arange(280), seq_transport, 1)[0]
    features['trans_range'] = np.max(seq_transport) - np.min(seq_transport)
    
    # Queue Gap (10)
    features['gap_mean'] = np.mean(seq_queue_gap)
    features['gap_std'] = np.std(seq_queue_gap)
    features['gap_max'] = np.max(seq_queue_gap)
    features['gap_min'] = np.min(seq_queue_gap)
    features['gap_current'] = seq_queue_gap[-1]
    features['gap_last_5'] = np.mean(seq_queue_gap[-5:])
    features['gap_last_10'] = np.mean(seq_queue_gap[-10:])
    features['gap_last_30'] = np.mean(seq_queue_gap[-30:])
    features['gap_slope'] = np.polyfit(np.arange(280), seq_queue_gap, 1)[0]
    features['gap_range'] = np.max(seq_queue_gap) - np.min(seq_queue_gap)
    
    # Interaction (30)
    features['m14b_x_m14bsum'] = seq_m14b[-1] * seq_m14b_sum[-1] / 1000
    features['m14b_x_m14bsum_mean'] = np.mean(seq_m14b * seq_m14b_sum) / 1000
    features['m14bsum_per_m14b'] = seq_m14b_sum[-1] / (seq_m14b[-1] + 1)
    features['m14b_plus_m14bsum'] = seq_m14b[-1] + seq_m14b_sum[-1]
    features['gap_x_m14b'] = seq_queue_gap[-1] * seq_m14b[-1] / 1000
    features['gap_x_m14bsum'] = seq_queue_gap[-1] * seq_m14b_sum[-1] / 1000
    features['gap_x_total'] = seq_queue_gap[-1] * seq_totalcnt[-1] / 1000
    features['gap_per_total'] = seq_queue_gap[-1] / (seq_totalcnt[-1] + 1)
    features['trans_x_m14b'] = seq_transport[-1] * seq_m14b[-1] / 100
    features['trans_x_m14bsum'] = seq_transport[-1] * seq_m14b_sum[-1] / 100
    features['trans_x_gap'] = seq_transport[-1] * seq_queue_gap[-1] / 100
    features['trans_x_oht'] = seq_transport[-1] * seq_oht[-1] / 10
    features['triple_danger'] = seq_m14b[-1] * seq_m14b_sum[-1] * seq_transport[-1] / 100000
    features['gap_trans_m14b'] = seq_queue_gap[-1] * seq_transport[-1] * seq_m14b[-1] / 100000
    features['m10arev_x_m14b'] = seq_m10a_rev[-1] * seq_m14b[-1] / 100
    features['oht_x_m14bsum'] = seq_oht[-1] * seq_m14b_sum[-1] / 10
    features['m16rev_x_total'] = seq_m16_rev[-1] * seq_totalcnt[-1] / 100
    features['ratio_m14b_total'] = seq_m14b[-1] / (seq_totalcnt[-1] + 1)
    features['ratio_m14bsum_total'] = seq_m14b_sum[-1] / (seq_totalcnt[-1] + 1)
    features['ratio_gap_m14b'] = seq_queue_gap[-1] / (seq_m14b[-1] + 1)
    features['ratio_trans_total'] = seq_transport[-1] / (seq_totalcnt[-1] + 1)
    features['vol_m14b'] = np.std(seq_m14b) / (np.mean(seq_m14b) + 1)
    features['vol_m14bsum'] = np.std(seq_m14b_sum) / (np.mean(seq_m14b_sum) + 1)
    features['vol_total'] = np.std(seq_totalcnt) / (np.mean(seq_totalcnt) + 1)
    features['vol_gap'] = np.std(seq_queue_gap) / (np.mean(np.abs(seq_queue_gap)) + 1)
    features['vol_trans'] = np.std(seq_transport) / (np.mean(seq_transport) + 1)
    features['corr_m14b_total'] = np.corrcoef(seq_m14b, seq_totalcnt)[0, 1]
    features['corr_m14bsum_total'] = np.corrcoef(seq_m14b_sum, seq_totalcnt)[0, 1]
    features['corr_gap_total'] = np.corrcoef(seq_queue_gap, seq_totalcnt)[0, 1]
    features['corr_trans_total'] = np.corrcoef(seq_transport, seq_totalcnt)[0, 1]
    
    # 임계값 (35)
    features['m14b_over_497'] = np.sum(seq_m14b > 497)
    features['m14b_over_517'] = np.sum(seq_m14b > 517)
    features['m14b_over_520'] = np.sum(seq_m14b > 520)
    features['m14b_over_539'] = np.sum(seq_m14b > 539)
    features['m14bsum_over_566'] = np.sum(seq_m14b_sum > 566)
    features['m14bsum_over_576'] = np.sum(seq_m14b_sum > 576)
    features['m14bsum_over_588'] = np.sum(seq_m14b_sum > 588)
    features['m14bsum_over_602'] = np.sum(seq_m14b_sum > 602)
    features['gap_over_200'] = np.sum(seq_queue_gap > 200)
    features['gap_over_250'] = np.sum(seq_queue_gap > 250)
    features['gap_over_300'] = np.sum(seq_queue_gap > 300)
    features['gap_over_350'] = np.sum(seq_queue_gap > 350)
    features['trans_over_145'] = np.sum(seq_transport > 145)
    features['trans_over_151'] = np.sum(seq_transport > 151)
    features['trans_over_171'] = np.sum(seq_transport > 171)
    features['trans_over_180'] = np.sum(seq_transport > 180)
    features['oht_over_83'] = np.sum(seq_oht > 83.6)
    features['oht_over_84'] = np.sum(seq_oht > 84.6)
    features['oht_over_86'] = np.sum(seq_oht > 85.6)
    features['total_over_1500'] = np.sum(seq_totalcnt >= 1500)
    features['total_over_1600'] = np.sum(seq_totalcnt >= 1600)
    features['total_over_1700'] = np.sum(seq_totalcnt >= 1700)
    features['total_over_1600_last30'] = np.sum(seq_totalcnt[-30:] >= 1600)
    features['m10arev_over_55'] = np.sum(seq_m10a_rev > 55)
    features['m10arev_over_59'] = np.sum(seq_m10a_rev > 59)
    features['m10a_under_80'] = np.sum(seq_m10a < 80)
    features['m10a_under_70'] = np.sum(seq_m10a < 70)
    features['m16rev_over_128'] = np.sum(seq_m16_rev > 128)
    features['m16rev_over_136'] = np.sum(seq_m16_rev > 136)
    features['m14b_over_517_last30'] = np.sum(seq_m14b[-30:] > 517)
    features['m14bsum_over_576_last30'] = np.sum(seq_m14b_sum[-30:] > 576)
    features['gap_over_250_last30'] = np.sum(seq_queue_gap[-30:] > 250)
    features['trans_over_151_last30'] = np.sum(seq_transport[-30:] > 151)
    
    # 황금 패턴 (20)
    features['must_condition'] = 1 if (seq_m14b[-1] > 497 and seq_m14b_sum[-1] > 566) else 0
    features['gold_strict'] = 1 if (seq_m14b[-1] > 520 and seq_m14b_sum[-1] > 588) else 0
    features['gold_normal'] = 1 if (seq_m14b[-1] > 517 and seq_m14b_sum[-1] > 576) else 0
    features['gold_loose'] = 1 if (seq_m14b[-1] > 509 and seq_m14b_sum[-1] > 570) else 0
    features['danger_gap'] = 1 if seq_queue_gap[-1] > 300 else 0
    features['danger_trans'] = 1 if seq_transport[-1] > 151 else 0
    features['danger_oht'] = 1 if seq_oht[-1] > 84.6 else 0
    features['triple_check'] = 1 if (seq_m14b[-1] > 517 and seq_m14b_sum[-1] > 576 and seq_queue_gap[-1] > 250) else 0
    features['quad_check'] = 1 if (seq_m14b[-1] > 517 and seq_m14b_sum[-1] > 576 and seq_queue_gap[-1] > 250 and seq_transport[-1] > 145) else 0
    features['danger_1700'] = 1 if seq_totalcnt[-1] >= 1700 else 0
    features['danger_1600'] = 1 if seq_totalcnt[-1] >= 1600 else 0
    features['in_1700'] = 1 if seq_totalcnt[-1] >= 1700 else 0
    features['rising_1700'] = 1 if (seq_totalcnt[-1] >= 1700 and seq_totalcnt[-1] - seq_totalcnt[-10] > 20) else 0
    features['stable_1700'] = 1 if (seq_totalcnt[-1] >= 1700 and abs(seq_totalcnt[-1] - seq_totalcnt[-10]) <= 20) else 0
    features['falling_1700'] = 1 if (seq_totalcnt[-1] >= 1700 and seq_totalcnt[-1] - seq_totalcnt[-10] < -20) else 0
    features['trend_10min'] = seq_totalcnt[-1] - seq_totalcnt[-10]
    features['trend_30min'] = seq_totalcnt[-1] - seq_totalcnt[-30]
    features['high_m14b_low_m10a'] = 1 if (seq_m14b[-1] > 517 and seq_m10a[-1] < 80) else 0
    features['high_gap_high_trans'] = 1 if (seq_queue_gap[-1] > 250 and seq_transport[-1] > 145) else 0
    
    # 시간대별 (10)
    q1 = seq_totalcnt[:70]
    q2 = seq_totalcnt[70:140]
    q3 = seq_totalcnt[140:210]
    q4 = seq_totalcnt[210:280]
    features['q1_mean'] = np.mean(q1)
    features['q2_mean'] = np.mean(q2)
    features['q3_mean'] = np.mean(q3)
    features['q4_mean'] = np.mean(q4)
    features['q_trend_1_2'] = np.mean(q2) - np.mean(q1)
    features['q_trend_2_3'] = np.mean(q3) - np.mean(q2)
    features['q_trend_3_4'] = np.mean(q4) - np.mean(q3)
    features['q_trend_overall'] = np.mean(q4) - np.mean(q1)
    features['q4_vs_mean'] = np.mean(q4) / (np.mean(seq_totalcnt) + 1)
    features['q_accel'] = (np.mean(q4) - np.mean(q3)) - (np.mean(q3) - np.mean(q2))
    
    return features

def adjust_light_plus(pred, m14b, m14bsum, gap, trans):
    """🎯 1650~1699만 집중 boost!"""
    boost = 0
    
    if 1650 <= pred < 1700:
        if (m14b > 520 and m14bsum > 588):
            boost += 50
        elif (m14b > 517 and m14bsum > 576):
            boost += 45
        elif (m14b > 509 and m14bsum > 570):
            boost += 35
        elif (m14b > 497 and m14bsum > 566):
            boost += 30
        
        if gap > 400:
            boost += 47
        elif gap > 350:
            boost += 40
        elif gap > 300:
            boost += 40
        elif gap > 250:
            boost += 30
        
        if trans > 200:
            boost += 43
        elif trans > 180:
            boost += 40
        
        if m14b > 550:
            boost += 37
        
        if gap > 300 and trans > 151:
            boost += 37
        
        if m14b > 520 and m14bsum > 588 and gap > 300 and trans > 151:
            boost += 45
    
    pred = pred + boost
    pred = min(pred, 2000)
    return pred

# 모델 로드 (⭐ 15분용 모델)
model = None
for mf in ['model_13col_15min.pkl', 'model_13col_final.pkl']:
    if os.path.exists(mf):
        with open(mf, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ 모델: {mf}")
        break

if not model:
    print("❌ 모델 없음!")
    exit(1)

# 데이터
df = pd.read_csv('data/M14_Q_20251014_TO_20251015.CSV', on_bad_lines='skip')
print(f"✅ 데이터: {len(df):,}행")
df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M')

results = []
total = len(df) - 295  # ⭐ 280 + 15
print(f"\n🔄 평가 중... {total:,}개 (280분 → 15분 후)")

for i in range(280, len(df) - 15):  # ⭐ 15분 후
    row_dict = {
        'M14AM14B': df['M14AM14B'].iloc[i-280:i].values,
        'M14AM14BSUM': df['M14AM14BSUM'].iloc[i-280:i].values,
        'M14BM14A': df['M14BM14A'].iloc[i-280:i].values,
        'M14AM10A': df['M14AM10A'].iloc[i-280:i].values,
        'M10AM14A': df['M10AM14A'].iloc[i-280:i].values,
        'M16M14A': df['M16M14A'].iloc[i-280:i].values,
        'M14AM16SUM': df['M14AM16SUM'].iloc[i-280:i].values,
        'TOTALCNT': df['TOTALCNT'].iloc[i-280:i].values,
        'M14.QUE.ALL.CURRENTQCREATED': df['M14.QUE.ALL.CURRENTQCREATED'].iloc[i-280:i].values,
        'M14.QUE.ALL.CURRENTQCOMPLETED': df['M14.QUE.ALL.CURRENTQCOMPLETED'].iloc[i-280:i].values,
        'M14.QUE.OHT.OHTUTIL': df['M14.QUE.OHT.OHTUTIL'].iloc[i-280:i].values,
        'M14.QUE.ALL.TRANSPORT4MINOVERCNT': df['M14.QUE.ALL.TRANSPORT4MINOVERCNT'].iloc[i-280:i].values,
    }
    
    seq_totalcnt = row_dict['TOTALCNT']
    seq_m14b = row_dict['M14AM14B']
    seq_m14b_sum = row_dict['M14AM14BSUM']
    seq_qc = row_dict['M14.QUE.ALL.CURRENTQCREATED']
    seq_qd = row_dict['M14.QUE.ALL.CURRENTQCOMPLETED']
    seq_gap = seq_qc - seq_qd
    seq_trans = row_dict['M14.QUE.ALL.TRANSPORT4MINOVERCNT']
    
    current_time = df['CURRTIME'].iloc[i-1]
    actual = df['TOTALCNT'].iloc[i+14]  # ⭐ 15분 후 (i+14)
    
    seq_start_time = df['CURRTIME'].iloc[i-280]
    prediction_time = current_time + timedelta(minutes=15)  # ⭐ 15분
    actual_time = df['CURRTIME'].iloc[i+14]  # ⭐ i+14
    
    # Feature 생성
    features = create_features_13col_optimized(row_dict)
    X = pd.DataFrame([features])
    
    # 예측
    pred_raw = model.predict(X)[0]
    pred = adjust_light_plus(pred_raw, seq_m14b[-1], seq_m14b_sum[-1], seq_gap[-1], seq_trans[-1])
    
    # 패턴 감지
    gold_strict = (seq_m14b[-1] > 520 and seq_m14b_sum[-1] > 588)
    gold_normal = (seq_m14b[-1] > 517 and seq_m14b_sum[-1] > 576)
    gold_loose = (seq_m14b[-1] > 509 and seq_m14b_sum[-1] > 570)
    must_condition = (seq_m14b[-1] > 497 and seq_m14b_sum[-1] > 566)
    danger_in_seq = np.sum(seq_totalcnt >= 1700) > 0
    
    last_10min = seq_totalcnt[-10:]
    early_warning = (np.max(last_10min) >= 1650) and ((last_10min[-1] - last_10min[0]) > 20)
    
    danger_gap = seq_gap[-1] > 300
    danger_trans = seq_trans[-1] > 151
    
    triple_check = (seq_m14b[-1] > 517 and seq_m14b_sum[-1] > 576 and seq_gap[-1] > 250)
    quad_check = (seq_m14b[-1] > 517 and seq_m14b_sum[-1] > 576 and seq_gap[-1] > 250 and seq_trans[-1] > 145)
    
    # 결과 저장
    results.append({
        '시퀀스시작': seq_start_time.strftime('%Y-%m-%d %H:%M'),
        '현재시간': current_time.strftime('%Y-%m-%d %H:%M'),
        '현재TOTALCNT': round(seq_totalcnt[-1], 2),
        '예측시점': prediction_time.strftime('%Y-%m-%d %H:%M'),
        '실제시점': actual_time.strftime('%Y-%m-%d %H:%M'),
        '실제값': round(actual, 2),
        '원본예측': round(pred_raw, 2),
        '보정예측': round(pred, 2),
        '오차': round(actual - pred, 2),
        '절대오차': round(abs(actual - pred), 2),
        '오차율(%)': round(abs(actual - pred) / max(actual, 1) * 100, 2),
        'M14AM14B': round(seq_m14b[-1], 2),
        'M14AM14BSUM': round(seq_m14b_sum[-1], 2),
        'M14BM14A': round(row_dict['M14BM14A'][-1], 2),
        'M14AM10A': round(row_dict['M14AM10A'][-1], 2),
        'M10AM14A': round(row_dict['M10AM14A'][-1], 2),
        'M16M14A': round(row_dict['M16M14A'][-1], 2),
        'M14AM16SUM': round(row_dict['M14AM16SUM'][-1], 2),
        'queue_gap': round(seq_gap[-1], 2),
        'TRANSPORT': round(seq_trans[-1], 2),
        'OHT_UTIL': round(row_dict['M14.QUE.OHT.OHTUTIL'][-1], 2),
        '시퀀스TOTALCNT_MAX': round(np.max(seq_totalcnt), 2),
        '시퀀스TOTALCNT_MIN': round(np.min(seq_totalcnt), 2),
        '시퀀스TOTALCNT_평균': round(np.mean(seq_totalcnt), 2),
        '마지막10분_MAX': round(np.max(last_10min), 2),
        '마지막10분_상승폭': round(last_10min[-1] - last_10min[0], 2),
        '필수조건': 'O' if must_condition else '',
        '황금패턴(엄격)': 'O' if gold_strict else '',
        '황금패턴(보통)': 'O' if gold_normal else '',
        '황금패턴(느슨)': 'O' if gold_loose else '',
        'Gap위험': 'O' if danger_gap else '',
        'Trans위험': 'O' if danger_trans else '',
        '3중체크': 'O' if triple_check else '',
        '4중체크': 'O' if quad_check else '',
        '시퀀스위험': 'O' if danger_in_seq else '',
        '조기경보': 'O' if early_warning else '',
        '실제위험(1700+)': 'O' if actual >= 1700 else '',
        '예측위험(1700+)': 'O' if pred >= 1700 else ''
    })
    
    if (i - 280) % 1000 == 0:
        print(f"  {i-280:,}/{total:,}")
        gc.collect()

print(f"✅ 완료!")

df_result = pd.DataFrame(results)
output = 'evaluation_15MIN.csv'  # ⭐ 파일명 변경
df_result.to_csv(output, index=False, encoding='utf-8-sig')
print(f"✅ 저장: {output}")

del df
gc.collect()

# ===== 통계 =====
print("\n" + "="*80)
print("📊 평가 통계 (280분 → 15분 후)")
print("="*80)
print(f"총 예측 수: {len(df_result):,}개")
print(f"평균 절대 오차(MAE): {df_result['절대오차'].mean():.2f}")
print(f"평균 오차율: {df_result['오차율(%)'].mean():.2f}%")
print(f"최대 절대 오차: {df_result['절대오차'].max():.2f}")
print(f"최소 절대 오차: {df_result['절대오차'].min():.2f}")

actual_danger = df_result['실제위험(1700+)'] == 'O'
pred_danger = df_result['예측위험(1700+)'] == 'O'
early_warning = df_result['조기경보'] == 'O'

actual_danger_count = actual_danger.sum()
pred_danger_count = pred_danger.sum()
danger_detected = (actual_danger & pred_danger).sum()
early_warning_count = early_warning.sum()

print(f"\n실제 위험(1700+): {actual_danger_count:,}개")
print(f"예측 위험(1700+): {pred_danger_count:,}개")
print(f"위험 감지 성공: {danger_detected:,}개")
if actual_danger_count > 0:
    print(f"🎯 위험 감지율: {danger_detected/actual_danger_count*100:.1f}%")

if actual_danger_count > 0:
    danger_rows = df_result[actual_danger]
    print(f"\n🔥 보정 효과 (1700+ 구간):")
    print(f"  원본 예측 평균: {danger_rows['원본예측'].mean():.1f}")
    print(f"  보정 예측 평균: {danger_rows['보정예측'].mean():.1f}")
    print(f"  실제값 평균: {danger_rows['실제값'].mean():.1f}")
    print(f"  평균 보정량: +{(danger_rows['보정예측'] - danger_rows['원본예측']).mean():.1f}")
    
    raw_detected = ((actual_danger) & (df_result['원본예측'] >= 1700)).sum()
    print(f"\n  원본 감지율: {raw_detected/actual_danger_count*100:.1f}% ({raw_detected:,}/{actual_danger_count:,})")
    print(f"  보정 감지율: {danger_detected/actual_danger_count*100:.1f}% ({danger_detected:,}/{actual_danger_count:,})")
    print(f"  개선: +{(danger_detected-raw_detected)/actual_danger_count*100:.1f}%p")

print("\n" + "="*80)
print("✅ 최종 평가 요약 (280분 → 15분 후)")
print("="*80)
print(f"1. 전체 성능:")
print(f"   - MAE: {df_result['절대오차'].mean():.2f}")
if actual_danger_count > 0:
    print(f"   - 위험 감지율: {danger_detected/actual_danger_count*100:.1f}% ({danger_detected:,}/{actual_danger_count:,})")
print(f"\n2. 저장 파일:")
print(f"   - {output}")
print("="*80)