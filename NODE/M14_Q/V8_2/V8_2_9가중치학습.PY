#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 XGBoost 학습 - 13개 컬럼 최종 강화 버전
VERTICAL 제거, multiplier=50!
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🚀 XGBoost 학습 - 13개 컬럼 최종 강화!")
print("="*80)
print("multiplier=50 - 1700+ 대폭 증강!")
print("Queue Gap 4.2배 / TRANSPORT 1.29배 활용!")
print("="*80)

# ========================================
# Feature 생성 함수
# ========================================
def create_features_13col_optimized(row_dict):
    """13개 컬럼 최적화 Feature 생성 (~222개)"""
    features = {}
    
    # 시퀀스 추출
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
    
    # 파생 컬럼
    seq_queue_gap = seq_q_created - seq_q_completed
    
    # ========================================
    # 1. M14AM14B (10개)
    # ========================================
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
    
    # ========================================
    # 2. M14AM14BSUM (10개)
    # ========================================
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
    
    # ========================================
    # 3. M14BM14A (10개)
    # ========================================
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
    
    # ========================================
    # 4. M14AM10A (10개)
    # ========================================
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
    
    # ========================================
    # 5. M10AM14A (10개)
    # ========================================
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
    
    # ========================================
    # 6. M16M14A (10개)
    # ========================================
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
    
    # ========================================
    # 7. M14AM16SUM (10개)
    # ========================================
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
    
    # ========================================
    # 8. TOTALCNT (10개)
    # ========================================
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
    
    # ========================================
    # 9. Queue Created (10개)
    # ========================================
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
    
    # ========================================
    # 10. Queue Completed (10개)
    # ========================================
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
    
    # ========================================
    # 11. OHT Util (10개)
    # ========================================
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
    
    # ========================================
    # 12. Transport (10개)
    # ========================================
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
    
    # ========================================
    # 13. Queue Gap (10개)
    # ========================================
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
    
    # ========================================
    # Interaction Features (30개)
    # ========================================
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
    
    # ========================================
    # 임계값 카운트 (35개)
    # ========================================
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
    
    # ========================================
    # 황금 패턴 (20개)
    # ========================================
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
    
    # ========================================
    # 시간대별 통계 (10개)
    # ========================================
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

# ========================================
# 데이터 준비
# ========================================
def prepare_data(csv_path):
    """데이터 로딩 및 Feature 생성"""
    print(f"\n📂 데이터 로딩: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✅ {len(df):,}행 × {df.shape[1]}열")
    
    # 필수 컬럼 (13개)
    required = [
        'CURRTIME', 'M14AM14B', 'M14AM14BSUM', 'M14BM14A',
        'M14AM10A', 'M10AM14A', 'M16M14A', 'M14AM16SUM', 'TOTALCNT',
        'M14.QUE.ALL.CURRENTQCREATED', 'M14.QUE.ALL.CURRENTQCOMPLETED',
        'M14.QUE.OHT.OHTUTIL', 'M14.QUE.ALL.TRANSPORT4MINOVERCNT'
    ]
    
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"❌ 필수 컬럼 누락: {missing}")
        return None, None
    
    print("✅ 13개 컬럼 확인 완료")
    
    # CURRTIME 변환
    df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M')
    
    print(f"\n🔄 Feature 생성 중...")
    X_list, y_list = [], []
    total = len(df) - 280 - 10
    
    for i in range(280, len(df) - 10):
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
        
        X_list.append(create_features_13col_optimized(row_dict))
        y_list.append(df['TOTALCNT'].iloc[i+9])
        
        if (i - 280) % 1000 == 0:
            print(f"  {i-280:,}/{total:,} ({(i-280)/total*100:.1f}%)", end='\r')
    
    print(f"\n✅ 완료!")
    
    X = pd.DataFrame(X_list)
    y = pd.Series(y_list)
    
    print(f"\n📊 결과:")
    print(f"  Feature: {X.shape[1]}개")
    print(f"  샘플: {len(X):,}개")
    print(f"  타겟 범위: {y.min():.0f}~{y.max():.0f}")
    print(f"  1700+: {(y>=1700).sum()}개 ({(y>=1700).sum()/len(y)*100:.3f}%)")
    
    return X, y

# ========================================
# 데이터 증강
# ========================================
def augment_data(X, y, threshold=1700, multiplier=50):
    """🔥 1700+ 데이터 50배 증강!"""
    print(f"\n🔥 1700+ 데이터 {multiplier}배 증강 (강력!)")
    
    high_mask = y >= threshold
    count = high_mask.sum()
    
    print(f"1700+ 원본: {count:,}개")
    
    if count == 0:
        print("⚠️ 증강 불가")
        return X, y
    
    X_aug = pd.concat([X] + [X[high_mask]] * (multiplier - 1), ignore_index=True)
    y_aug = pd.concat([y] + [y[high_mask]] * (multiplier - 1), ignore_index=True)
    
    print(f"증강 후: {len(X_aug):,}개")
    print(f"1700+ 비율: {(y_aug>=threshold).sum()/len(y_aug)*100:.2f}%")
    
    return X_aug, y_aug

# ========================================
# 학습
# ========================================
def train_model(X, y):
    """XGBoost 학습"""
    print(f"\n🚀 XGBoost 학습")
    
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        max_depth=8,
        learning_rate=0.05,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method='hist'
    )
    
    model.fit(X, y, verbose=50)
    print(f"\n✅ 학습 완료!")
    
    return model

# ========================================
# 평가
# ========================================
def evaluate(model, X, y):
    """성능 평가"""
    print(f"\n" + "="*80)
    print(f"📊 성능 평가")
    print("="*80)
    
    y_pred = model.predict(X)
    
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    print(f"\n전체:")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²:   {r2:.4f}")
    
    # 1700+
    high = y >= 1700
    if high.sum() > 0:
        mae_h = mean_absolute_error(y[high], y_pred[high])
        detected = ((y >= 1700) & (y_pred >= 1680)).sum()
        rate = detected / high.sum() * 100
        
        print(f"\n1700+:")
        print(f"  샘플: {high.sum():,}개")
        print(f"  MAE: {mae_h:.2f}")
        print(f"  감지율 (1680+): {rate:.1f}% ({detected:,}/{high.sum():,})")
        print(f"  평균 예측: {y_pred[high].mean():.1f}")
        print(f"  최소 예측: {y_pred[high].min():.1f}")
        print(f"  최대 예측: {y_pred[high].max():.1f}")
    
    print("="*80)
    return mae, rmse, r2

# ========================================
# Feature 중요도
# ========================================
def show_importance(model, X):
    """TOP 30 Feature 중요도"""
    print(f"\n" + "="*80)
    print(f"🏆 Feature 중요도 TOP 30")
    print("="*80)
    
    imp = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(imp.head(30).to_string(index=False))

# ========================================
# 메인
# ========================================
if __name__ == '__main__':
    # CSV 파일 경로
    CSV_FILE = "/mnt/user-data/uploads/ssss.CSV"
    
    # 데이터 준비
    X, y = prepare_data(CSV_FILE)
    if X is None:
        exit(1)
    
    # 증강 (50배!)
    X_aug, y_aug = augment_data(X, y, threshold=1700, multiplier=50)
    
    # 학습
    model = train_model(X_aug, y_aug)
    
    # 평가
    mae, rmse, r2 = evaluate(model, X_aug, y_aug)
    
    # Feature 중요도
    show_importance(model, X_aug)
    
    # 저장
    with open('/mnt/user-data/outputs/model_13col_final.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('/mnt/user-data/outputs/features_13col_final.pkl', 'wb') as f:
        pickle.dump(list(X.columns), f)
    
    print(f"\n💾 저장:")
    print(f"  model_13col_final.pkl")
    print(f"  features_13col_final.pkl")
    
    print(f"\n" + "="*80)
    print(f"✅ 완료! multiplier=50")
    print(f"Feature: {X.shape[1]}개")
    print(f"MAE: {mae:.2f}, R²: {r2:.4f}")
    print("="*80)