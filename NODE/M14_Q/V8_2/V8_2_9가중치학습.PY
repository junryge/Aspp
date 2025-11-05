#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 XGBoost 학습 - 14개 컬럼 (280분→10분 예측)
M14AM14BSUM + VERTICALQUEUE 추가, 양방향 흐름 포함
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🚀 XGBoost 학습 - 14개 컬럼")
print("="*80)

# ========================================
# Feature 생성 함수
# ========================================
def create_features_14col(row_dict):
    """14개 컬럼 280분 시퀀스로부터 약 240개 Feature 생성"""
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
    seq_vertical = np.array(row_dict['M14B.QUE.SENDFAB.VERTICALQUEUECOUNT'])
    
    # 파생
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
    features['m14b_sum_mean'] = np.mean(seq_m14b_sum)
    features['m14b_sum_std'] = np.std(seq_m14b_sum)
    features['m14b_sum_max'] = np.max(seq_m14b_sum)
    features['m14b_sum_min'] = np.min(seq_m14b_sum)
    features['m14b_sum_current'] = seq_m14b_sum[-1]
    features['m14b_sum_last_5'] = np.mean(seq_m14b_sum[-5:])
    features['m14b_sum_last_10'] = np.mean(seq_m14b_sum[-10:])
    features['m14b_sum_last_30'] = np.mean(seq_m14b_sum[-30:])
    features['m14b_sum_slope'] = np.polyfit(np.arange(280), seq_m14b_sum, 1)[0]
    features['m14b_sum_range'] = np.max(seq_m14b_sum) - np.min(seq_m14b_sum)
    
    # ========================================
    # 3. M14BM14A (10개)
    # ========================================
    features['m14b_rev_mean'] = np.mean(seq_m14b_rev)
    features['m14b_rev_std'] = np.std(seq_m14b_rev)
    features['m14b_rev_max'] = np.max(seq_m14b_rev)
    features['m14b_rev_min'] = np.min(seq_m14b_rev)
    features['m14b_rev_current'] = seq_m14b_rev[-1]
    features['m14b_rev_last_5'] = np.mean(seq_m14b_rev[-5:])
    features['m14b_rev_last_10'] = np.mean(seq_m14b_rev[-10:])
    features['m14b_rev_last_30'] = np.mean(seq_m14b_rev[-30:])
    features['m14b_rev_slope'] = np.polyfit(np.arange(280), seq_m14b_rev, 1)[0]
    features['m14b_rev_range'] = np.max(seq_m14b_rev) - np.min(seq_m14b_rev)
    
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
    features['m10a_rev_mean'] = np.mean(seq_m10a_rev)
    features['m10a_rev_std'] = np.std(seq_m10a_rev)
    features['m10a_rev_max'] = np.max(seq_m10a_rev)
    features['m10a_rev_min'] = np.min(seq_m10a_rev)
    features['m10a_rev_current'] = seq_m10a_rev[-1]
    features['m10a_rev_last_5'] = np.mean(seq_m10a_rev[-5:])
    features['m10a_rev_last_10'] = np.mean(seq_m10a_rev[-10:])
    features['m10a_rev_last_30'] = np.mean(seq_m10a_rev[-30:])
    features['m10a_rev_slope'] = np.polyfit(np.arange(280), seq_m10a_rev, 1)[0]
    features['m10a_rev_range'] = np.max(seq_m10a_rev) - np.min(seq_m10a_rev)
    
    # ========================================
    # 6. M16M14A (10개)
    # ========================================
    features['m16_rev_mean'] = np.mean(seq_m16_rev)
    features['m16_rev_std'] = np.std(seq_m16_rev)
    features['m16_rev_max'] = np.max(seq_m16_rev)
    features['m16_rev_min'] = np.min(seq_m16_rev)
    features['m16_rev_current'] = seq_m16_rev[-1]
    features['m16_rev_last_5'] = np.mean(seq_m16_rev[-5:])
    features['m16_rev_last_10'] = np.mean(seq_m16_rev[-10:])
    features['m16_rev_last_30'] = np.mean(seq_m16_rev[-30:])
    features['m16_rev_slope'] = np.polyfit(np.arange(280), seq_m16_rev, 1)[0]
    features['m16_rev_range'] = np.max(seq_m16_rev) - np.min(seq_m16_rev)
    
    # ========================================
    # 7. M14AM16SUM (10개)
    # ========================================
    features['m16_sum_mean'] = np.mean(seq_m16_sum)
    features['m16_sum_std'] = np.std(seq_m16_sum)
    features['m16_sum_max'] = np.max(seq_m16_sum)
    features['m16_sum_min'] = np.min(seq_m16_sum)
    features['m16_sum_current'] = seq_m16_sum[-1]
    features['m16_sum_last_5'] = np.mean(seq_m16_sum[-5:])
    features['m16_sum_last_10'] = np.mean(seq_m16_sum[-10:])
    features['m16_sum_last_30'] = np.mean(seq_m16_sum[-30:])
    features['m16_sum_slope'] = np.polyfit(np.arange(280), seq_m16_sum, 1)[0]
    features['m16_sum_range'] = np.max(seq_m16_sum) - np.min(seq_m16_sum)
    
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
    # 13. Vertical Queue (10개) ⭐ 신규!
    # ========================================
    features['vertical_mean'] = np.mean(seq_vertical)
    features['vertical_std'] = np.std(seq_vertical)
    features['vertical_max'] = np.max(seq_vertical)
    features['vertical_min'] = np.min(seq_vertical)
    features['vertical_current'] = seq_vertical[-1]
    features['vertical_last_5'] = np.mean(seq_vertical[-5:])
    features['vertical_last_10'] = np.mean(seq_vertical[-10:])
    features['vertical_last_30'] = np.mean(seq_vertical[-30:])
    features['vertical_slope'] = np.polyfit(np.arange(280), seq_vertical, 1)[0]
    features['vertical_range'] = np.max(seq_vertical) - np.min(seq_vertical)
    
    # ========================================
    # 14. Queue Gap (10개)
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
    # 비율 Feature (25개)
    # ========================================
    features['ratio_m14b_m14bsum'] = seq_m14b[-1] / (seq_m14b_sum[-1] + 1)
    features['ratio_m14b_m10a'] = seq_m14b[-1] / (seq_m10a[-1] + 1)
    features['ratio_m14b_total'] = seq_m14b[-1] / (seq_totalcnt[-1] + 1)
    features['ratio_m14bsum_total'] = seq_m14b_sum[-1] / (seq_totalcnt[-1] + 1)
    features['ratio_gap_total'] = seq_queue_gap[-1] / (seq_totalcnt[-1] + 1)
    features['ratio_m10a_rev_m10a'] = seq_m10a_rev[-1] / (seq_m10a[-1] + 1)
    features['ratio_m14b_rev_m14b'] = seq_m14b_rev[-1] / (seq_m14b[-1] + 1)
    features['ratio_m16_rev_m16sum'] = seq_m16_rev[-1] / (seq_m16_sum[-1] + 1)
    features['ratio_vertical_m14b'] = seq_vertical[-1] / (seq_m14b[-1] + 1)
    features['ratio_vertical_total'] = seq_vertical[-1] / (seq_totalcnt[-1] + 1)
    
    features['vol_m14b'] = np.std(seq_m14b) / (np.mean(seq_m14b) + 1)
    features['vol_m14bsum'] = np.std(seq_m14b_sum) / (np.mean(seq_m14b_sum) + 1)
    features['vol_total'] = np.std(seq_totalcnt) / (np.mean(seq_totalcnt) + 1)
    features['vol_gap'] = np.std(seq_queue_gap) / (np.mean(np.abs(seq_queue_gap)) + 1)
    features['vol_oht'] = np.std(seq_oht) / (np.mean(seq_oht) + 1)
    features['vol_vertical'] = np.std(seq_vertical) / (np.mean(seq_vertical) + 1)
    
    features['corr_m14b_total'] = np.corrcoef(seq_m14b, seq_totalcnt)[0, 1]
    features['corr_m14bsum_total'] = np.corrcoef(seq_m14b_sum, seq_totalcnt)[0, 1]
    features['corr_gap_total'] = np.corrcoef(seq_queue_gap, seq_totalcnt)[0, 1]
    features['corr_m10a_rev_total'] = np.corrcoef(seq_m10a_rev, seq_totalcnt)[0, 1]
    features['corr_oht_total'] = np.corrcoef(seq_oht, seq_totalcnt)[0, 1]
    features['corr_trans_total'] = np.corrcoef(seq_transport, seq_totalcnt)[0, 1]
    features['corr_m16sum_total'] = np.corrcoef(seq_m16_sum, seq_totalcnt)[0, 1]
    features['corr_vertical_total'] = np.corrcoef(seq_vertical, seq_totalcnt)[0, 1]
    features['corr_vertical_m14b'] = np.corrcoef(seq_vertical, seq_m14b)[0, 1]
    
    # ========================================
    # 임계값 카운트 (30개)
    # ========================================
    features['m14b_over_300'] = np.sum(seq_m14b > 300)
    features['m14b_over_400'] = np.sum(seq_m14b > 400)
    features['m14b_over_450'] = np.sum(seq_m14b > 450)
    features['m14b_over_500'] = np.sum(seq_m14b > 500)
    
    features['m14bsum_over_400'] = np.sum(seq_m14b_sum > 400)
    features['m14bsum_over_500'] = np.sum(seq_m14b_sum > 500)
    features['m14bsum_over_550'] = np.sum(seq_m14b_sum > 550)
    
    features['total_over_1500'] = np.sum(seq_totalcnt >= 1500)
    features['total_over_1600'] = np.sum(seq_totalcnt >= 1600)
    features['total_over_1700'] = np.sum(seq_totalcnt >= 1700)
    features['total_over_1600_last30'] = np.sum(seq_totalcnt[-30:] >= 1600)
    features['total_over_1700_last30'] = np.sum(seq_totalcnt[-30:] >= 1700)
    
    features['gap_over_100'] = np.sum(seq_queue_gap > 100)
    features['gap_over_150'] = np.sum(seq_queue_gap > 150)
    features['gap_over_200'] = np.sum(seq_queue_gap > 200)
    
    features['oht_over_80'] = np.sum(seq_oht > 80)
    features['oht_over_85'] = np.sum(seq_oht > 85)
    features['oht_over_90'] = np.sum(seq_oht > 90)
    
    features['trans_over_150'] = np.sum(seq_transport > 150)
    features['trans_over_180'] = np.sum(seq_transport > 180)
    
    features['vertical_over_140'] = np.sum(seq_vertical > 140)
    features['vertical_over_160'] = np.sum(seq_vertical > 160)
    features['vertical_over_180'] = np.sum(seq_vertical > 180)
    features['vertical_over_160_last30'] = np.sum(seq_vertical[-30:] > 160)
    
    features['m10a_rev_over_55'] = np.sum(seq_m10a_rev > 55)
    features['m10a_rev_over_60'] = np.sum(seq_m10a_rev > 60)
    
    features['m10a_under_80'] = np.sum(seq_m10a < 80)
    features['m10a_under_70'] = np.sum(seq_m10a < 70)
    
    # ========================================
    # 황금 패턴 & 위험 신호 (20개)
    # ========================================
    features['golden_m14b_450'] = 1 if seq_m14b[-1] > 450 else 0
    features['golden_m14b_500'] = 1 if seq_m14b[-1] > 500 else 0
    features['golden_m14bsum_550'] = 1 if seq_m14b_sum[-1] > 550 else 0
    features['golden_vertical_160'] = 1 if seq_vertical[-1] > 160 else 0
    
    features['golden_pattern_1'] = 1 if (seq_m14b[-1] > 450 and seq_m10a[-1] < 80) else 0
    features['golden_pattern_2'] = 1 if (seq_m14b_sum[-1] > 550 and seq_oht[-1] > 80) else 0
    features['golden_pattern_3'] = 1 if (seq_m14b[-1] > 450 and seq_vertical[-1] > 160) else 0
    
    features['danger_1700'] = 1 if seq_totalcnt[-1] >= 1700 else 0
    features['danger_1600'] = 1 if seq_totalcnt[-1] >= 1600 else 0
    features['danger_signal_1'] = 1 if (seq_queue_gap[-1] > 150 and seq_m14b[-1] > 400) else 0
    features['danger_signal_2'] = 1 if (seq_m14b_sum[-1] > 500 and seq_oht[-1] > 85) else 0
    features['danger_signal_3'] = 1 if (seq_vertical[-1] > 160 and seq_m14b[-1] > 400) else 0
    
    # 3중 위험 신호 (슈퍼 위험!)
    features['super_danger'] = 1 if (
        seq_m14b[-1] > 450 and 
        seq_m14b_sum[-1] > 550 and 
        seq_vertical[-1] > 160
    ) else 0
    
    features['in_1700'] = 1 if seq_totalcnt[-1] >= 1700 else 0
    features['rising_1700'] = 1 if (seq_totalcnt[-1] >= 1700 and seq_totalcnt[-1] - seq_totalcnt[-10] > 20) else 0
    features['stable_1700'] = 1 if (seq_totalcnt[-1] >= 1700 and abs(seq_totalcnt[-1] - seq_totalcnt[-10]) <= 20) else 0
    features['falling_1700'] = 1 if (seq_totalcnt[-1] >= 1700 and seq_totalcnt[-1] - seq_totalcnt[-10] < -20) else 0
    
    features['trend_10min'] = seq_totalcnt[-1] - seq_totalcnt[-10]
    features['trend_30min'] = seq_totalcnt[-1] - seq_totalcnt[-30]
    
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
    
    # 필수 컬럼 (14개)
    required = [
        'CURRTIME', 'M14AM14B', 'M14AM14BSUM', 'M14BM14A',
        'M14AM10A', 'M10AM14A', 'M16M14A', 'M14AM16SUM', 'TOTALCNT',
        'M14.QUE.ALL.CURRENTQCREATED', 'M14.QUE.ALL.CURRENTQCOMPLETED',
        'M14.QUE.OHT.OHTUTIL', 'M14.QUE.ALL.TRANSPORT4MINOVERCNT',
        'M14B.QUE.SENDFAB.VERTICALQUEUECOUNT'
    ]
    
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"❌ 필수 컬럼 누락: {missing}")
        return None, None
    
    print("✅ 14개 컬럼 확인 완료")
    
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
            'M14B.QUE.SENDFAB.VERTICALQUEUECOUNT': df['M14B.QUE.SENDFAB.VERTICALQUEUECOUNT'].iloc[i-280:i].values,
        }
        
        X_list.append(create_features_14col(row_dict))
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
def augment_data(X, y, threshold=1700, multiplier=10):
    """1700+ 데이터 증강"""
    print(f"\n🔥 1700+ 데이터 {multiplier}배 증강")
    
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
        detected = ((y >= 1700) & (y_pred >= 1650)).sum()
        rate = detected / high.sum() * 100
        
        print(f"\n1700+:")
        print(f"  샘플: {high.sum():,}개")
        print(f"  MAE: {mae_h:.2f}")
        print(f"  감지율: {rate:.1f}% ({detected:,}/{high.sum():,})")
    
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
    # CSV 파일 경로 (수정하세요!)
    CSV_FILE = "당신의파일.csv"
    
    # 데이터 준비
    X, y = prepare_data(CSV_FILE)
    if X is None:
        exit(1)
    
    # 증강
    X_aug, y_aug = augment_data(X, y, threshold=1700, multiplier=10)
    
    # 학습
    model = train_model(X_aug, y_aug)
    
    # 평가
    mae, rmse, r2 = evaluate(model, X_aug, y_aug)
    
    # Feature 중요도
    show_importance(model, X_aug)
    
    # 저장
    with open('model_14col.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('features_14col.pkl', 'wb') as f:
        pickle.dump(list(X.columns), f)
    
    print(f"\n💾 저장:")
    print(f"  model_14col.pkl")
    print(f"  features_14col.pkl")
    
    print(f"\n" + "="*80)
    print(f"✅ 완료! Feature: {X.shape[1]}개, MAE: {mae:.2f}, R²: {r2:.4f}")
    print("="*80)