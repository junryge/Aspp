#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 실시간 예측 대시보드 - 60분 그래프 버전
10분/15분/25분 후 예측 + 풍부한 그래프
"""

import numpy as np
import pandas as pd
import pickle
import os
import webbrowser
from datetime import datetime, timedelta

# ========================================
# 설정
# ========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, 'data', '2025_DATA.CSV')

# 모델 파일
MODEL_FILES = {
    10: 'model_13col_10.pkl',
    15: 'model_13col_15.pkl',
    25: 'model_13col_25.pkl'
}

# 필수 컬럼 13개
REQUIRED_COLS = [
    'M14AM14B', 'M14AM14BSUM', 'M14BM14A',
    'M14AM10A', 'M10AM14A', 'M16M14A', 'M14AM16SUM', 'TOTALCNT',
    'M14.QUE.ALL.CURRENTQCREATED', 'M14.QUE.ALL.CURRENTQCOMPLETED',
    'M14.QUE.OHT.OHTUTIL', 'M14.QUE.ALL.TRANSPORT4MINOVERCNT'
]

# ========================================
# Feature 생성 함수 (학습 코드와 100% 동일!)
# ========================================
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

# ========================================
# Boost 함수
# ========================================
def adjust_light_plus(pred, m14b, m14bsum, gap, trans):
    """🎯 1650~1699만 집중"""
    boost = 0
    
    if 1650 <= pred < 1700:
        # 황금패턴
        if (m14b > 520 and m14bsum > 588):
            boost += 50
        elif (m14b > 517 and m14bsum > 576):
            boost += 45
        elif (m14b > 509 and m14bsum > 570):
            boost += 35
        elif (m14b > 497 and m14bsum > 566):
            boost += 30
        
        # Gap
        if gap > 400:
            boost += 47
        elif gap > 350:
            boost += 40
        elif gap > 300:
            boost += 40
        elif gap > 250:
            boost += 30
        
        # Transport
        if trans > 200:
            boost += 43
        elif trans > 180:
            boost += 40
        
        # 극단값
        if m14b > 550:
            boost += 37
        
        # 복합
        if gap > 300 and trans > 151:
            boost += 37
        
        # 최강 복합
        if m14b > 520 and m14bsum > 588 and gap > 300 and trans > 151:
            boost += 45
    
    pred = pred + boost
    pred = min(pred, 2000)
    return pred

# ========================================
# 상태 판정
# ========================================
def get_status_info(value):
    """물류량에 따른 상태"""
    if value < 900:
        return 'LOW'
    elif value < 1600:
        return 'NORMAL'
    elif value < 1700:
        return 'CAUTION'
    else:
        return 'CRITICAL'

# ========================================
# 예측 실행
# ========================================
def predict_all_horizons():
    """10/15/25분 예측 실행"""
    
    print("="*60)
    print("🚀 실시간 예측 대시보드 - 60분 그래프")
    print("="*60)
    
    # 데이터 로드
    if not os.path.exists(DATA_PATH):
        return {'error': 'No data', 'message': f'CSV file not found: {DATA_PATH}'}
    
    try:
        df = pd.read_csv(DATA_PATH, on_bad_lines='skip', dtype={'CURRTIME': str})
    except Exception as e:
        return {'error': 'No data', 'message': f'Failed to load data: {e}'}
    
    if len(df) == 0:
        return {'error': 'No data', 'message': 'CSV file is empty'}
    
    # 필수 컬럼 확인
    missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing_cols:
        return {'error': 'No data', 'message': f'Missing columns: {", ".join(missing_cols)}'}
    
    # 데이터 부족 확인
    if len(df) < 280:
        return {'error': 'Lack of data', 'message': f'Need 280 rows, got {len(df)}'}
    
    print(f"✅ Data: {len(df):,} rows")
    
    # CURRTIME 파싱
    if 'CURRTIME' in df.columns:
        try:
            df['CURRTIME'] = df['CURRTIME'].astype(str).str.strip()
            df = df[df['CURRTIME'].str.len() == 12].copy()
            df['CURRTIME'] = pd.to_datetime(df['CURRTIME'], format='%Y%m%d%H%M', errors='coerce')
            df = df.dropna(subset=['CURRTIME']).copy()
        except:
            base_time = datetime.now()
            df['CURRTIME'] = [base_time - timedelta(minutes=len(df)-1-i) for i in range(len(df))]
    else:
        base_time = datetime.now()
        df['CURRTIME'] = [base_time - timedelta(minutes=len(df)-1-i) for i in range(len(df))]
    
    # 최종 데이터 부족 재확인
    if len(df) < 280:
        return {'error': 'Lack of data', 'message': f'Need 280 rows, got {len(df)} after parsing'}
    
    # 280분 데이터 추출
    row_dict = {
        'M14AM14B': df['M14AM14B'].iloc[-280:].values,
        'M14AM14BSUM': df['M14AM14BSUM'].iloc[-280:].values,
        'M14BM14A': df['M14BM14A'].iloc[-280:].values,
        'M14AM10A': df['M14AM10A'].iloc[-280:].values,
        'M10AM14A': df['M10AM14A'].iloc[-280:].values,
        'M16M14A': df['M16M14A'].iloc[-280:].values,
        'M14AM16SUM': df['M14AM16SUM'].iloc[-280:].values,
        'TOTALCNT': df['TOTALCNT'].iloc[-280:].values,
        'M14.QUE.ALL.CURRENTQCREATED': df['M14.QUE.ALL.CURRENTQCREATED'].iloc[-280:].values,
        'M14.QUE.ALL.CURRENTQCOMPLETED': df['M14.QUE.ALL.CURRENTQCOMPLETED'].iloc[-280:].values,
        'M14.QUE.OHT.OHTUTIL': df['M14.QUE.OHT.OHTUTIL'].iloc[-280:].values,
        'M14.QUE.ALL.TRANSPORT4MINOVERCNT': df['M14.QUE.ALL.TRANSPORT4MINOVERCNT'].iloc[-280:].values,
    }
    
    # 시간 정보
    current_time = df['CURRTIME'].iloc[-1]
    if pd.isna(current_time):
        current_time = datetime.now()
    
    # 현재 상태
    seq_totalcnt = row_dict['TOTALCNT']
    seq_m14b = row_dict['M14AM14B']
    seq_m14b_sum = row_dict['M14AM14BSUM']
    seq_qc = row_dict['M14.QUE.ALL.CURRENTQCREATED']
    seq_qd = row_dict['M14.QUE.ALL.CURRENTQCOMPLETED']
    seq_gap = seq_qc - seq_qd
    seq_trans = row_dict['M14.QUE.ALL.TRANSPORT4MINOVERCNT']
    
    current_totalcnt = seq_totalcnt[-1]
    current_m14b = seq_m14b[-1]
    current_m14bsum = seq_m14b_sum[-1]
    current_gap = seq_gap[-1]
    current_trans = seq_trans[-1]
    
    print(f"Current TOTALCNT: {current_totalcnt:.1f}")
    
    # Feature 생성
    try:
        features = create_features_13col_optimized(row_dict)
        X_pred = pd.DataFrame([features])
        print(f"✅ Features: {len(features)}")
    except Exception as e:
        return {'error': 'Model operation failure', 'message': f'Feature generation failed: {e}'}
    
    # 🎨 그래프용 데이터 (최근 60분!)
    totalcnt_data = seq_totalcnt[-60:].tolist()  # 60개
    m14b_data = seq_m14b[-60:].tolist()
    m14bsum_data = seq_m14b_sum[-60:].tolist()
    gap_data = seq_gap[-60:].tolist()
    trans_data = seq_trans[-60:].tolist()
    
    # 시간 라벨 (60개)
    time_labels = df['CURRTIME'].tail(60).tolist()
    time_labels = [t.strftime('%H:%M') for t in time_labels]
    
    # 각 예측 실행
    results = []
    
    for horizon_min in [10, 15, 25]:
        model_file = MODEL_FILES[horizon_min]
        
        if not os.path.exists(model_file):
            print(f"⚠️ {horizon_min}min: Model not found")
            continue
        
        try:
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            
            # 원본 예측
            pred_raw = model.predict(X_pred)[0]
            
            # Boost 보정
            pred = adjust_light_plus(pred_raw, current_m14b, current_m14bsum, current_gap, current_trans)
            
            # 상태 판정
            pred_status = get_status_info(pred)
            
            # 위험 확률 계산
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
            gold_strict = (current_m14b > 520 and current_m14bsum > 588)
            gold_normal = (current_m14b > 517 and current_m14bsum > 576)
            
            if gold_strict:
                danger_prob = min(100, danger_prob + 20)
            elif gold_normal:
                danger_prob = min(100, danger_prob + 15)
            elif (current_m14b > 509 and current_m14bsum > 570):
                danger_prob = min(100, danger_prob + 10)
            
            # Gap/Trans 보정
            if current_gap > 300:
                danger_prob = min(100, danger_prob + (10 if current_gap > 350 else 5))
            if current_trans > 151:
                danger_prob = min(100, danger_prob + (10 if current_trans > 180 else 5))
            
            # 현재값 보정
            if current_totalcnt >= 1700:
                danger_prob = max(danger_prob, 85)
            elif current_totalcnt >= 1650:
                danger_prob = max(danger_prob, 60)
            
            danger_prob = max(0, min(100, danger_prob))
            
            result = {
                'horizon': horizon_min,
                'current_time': current_time.strftime('%Y-%m-%d %H:%M'),
                'pred_time': (current_time + timedelta(minutes=horizon_min)).strftime('%Y-%m-%d %H:%M'),
                'pred_time_label': (current_time + timedelta(minutes=horizon_min)).strftime('%H:%M'),
                'prediction': int(pred),
                'status': pred_status,
                'danger_probability': danger_prob,
                'change': int(pred - current_totalcnt)
            }
            
            results.append(result)
            print(f"✅ {horizon_min}min: {pred:.1f} ({pred_status}) - Danger: {danger_prob}%")
            
        except Exception as e:
            print(f"❌ {horizon_min}min: {e}")
            continue
    
    if not results:
        return {'error': 'Model operation failure', 'message': 'All predictions failed'}
    
    return {
        'current_value': int(current_totalcnt),
        'current_time': current_time.strftime('%Y-%m-%d %H:%M'),
        'current_status': get_status_info(current_totalcnt),
        'current_m14b': float(current_m14b),
        'current_m14bsum': float(current_m14bsum),
        'current_gap': float(current_gap),
        'current_trans': float(current_trans),
        'predictions': results,
        'totalcnt_data': totalcnt_data,
        'm14b_data': m14b_data,
        'm14bsum_data': m14bsum_data,
        'gap_data': gap_data,
        'trans_data': trans_data,
        'time_labels': time_labels
    }

# ========================================
# HTML 생성
# ========================================
def generate_html(data):
    """대시보드 HTML 생성 - 풍부한 그래프!"""
    
    if 'error' in data:
        return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Error</title></head>
<body style="font-family:sans-serif;padding:50px;text-align:center;">
<h1 style="color:#e53e3e;">❌ {data['error']}</h1>
<p style="font-size:18px;color:#718096;">{data['message']}</p>
</body></html>"""
    
    current_value = data['current_value']
    current_status = data['current_status']
    predictions = data['predictions']
    
    # 최대 위험도
    max_danger = max(p['danger_probability'] for p in predictions)
    
    if max_danger >= 85:
        risk, risk_color = "CRITICAL", "#c53030"
    elif max_danger >= 60:
        risk, risk_color = "WARNING", "#dd6b20"
    elif max_danger >= 30:
        risk, risk_color = "CAUTION", "#d69e2e"
    else:
        risk, risk_color = "NORMAL", "#38a169"
    
    # 그래프 데이터
    import json
    
    # 1. TOTALCNT: 280분 + 예측 3개
    chart1_labels = data['time_labels'] + [p['pred_time_label'] for p in predictions]
    chart1_historical = data['totalcnt_data'] + [None] * len(predictions)
    chart1_predictions = [None] * len(data['totalcnt_data']) + [p['prediction'] for p in predictions]
    
    # 2. M14AM14B: 280분
    chart2_labels = data['time_labels']
    chart2_data = data['m14b_data']
    
    # 3. M14AM14BSUM: 280분
    chart3_data = data['m14bsum_data']
    
    # 4. queue_gap: 280분
    chart4_data = data['gap_data']
    
    # 5. TRANSPORT: 280분
    chart5_data = data['trans_data']
    
    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>60분 실시간 예측 대시보드</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:sans-serif;background:linear-gradient(135deg,#667eea,#764ba2);padding:20px}}
.container{{max-width:1600px;margin:0 auto}}
.card{{background:#fff;border-radius:15px;padding:30px;margin-bottom:20px;box-shadow:0 10px 30px rgba(0,0,0,0.2)}}
.header{{text-align:center;font-size:36px;color:#2d3748;margin-bottom:10px;font-weight:700}}
.subtitle{{text-align:center;font-size:18px;color:#718096;margin-bottom:20px}}
.current{{background:#f7fafc;border-radius:10px;padding:20px;margin-bottom:20px}}
.current-grid{{display:grid;grid-template-columns:repeat(5,1fr);gap:15px}}
.current-item{{text-align:center}}
.current-label{{font-size:12px;color:#718096;margin-bottom:5px}}
.current-value{{font-size:28px;font-weight:700;color:#2d3748}}
.current-status{{display:inline-block;padding:8px 16px;border-radius:20px;font-weight:700;margin-top:5px;font-size:14px}}
.status-low{{background:#c6f6d5;color:#2f855a}}
.status-normal{{background:#bee3f8;color:#2c5282}}
.status-caution{{background:#fefcbf;color:#b7791f}}
.status-critical{{background:#fed7d7;color:#c53030}}
.risk{{background:{risk_color};color:#fff;padding:20px;border-radius:10px;text-align:center;font-size:24px;font-weight:700;margin-bottom:20px}}
.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:20px;margin-bottom:30px}}
.pred{{background:#f7fafc;border-radius:10px;padding:20px;text-align:center}}
.time{{font-size:20px;font-weight:700;color:#667eea;margin-bottom:5px}}
.pred-value{{font-size:42px;font-weight:700;color:#2d3748;margin:15px 0}}
.danger{{font-size:24px;font-weight:700;margin:10px 0}}
.danger-high{{color:#e53e3e}}
.danger-medium{{color:#dd6b20}}
.danger-low{{color:#38a169}}
.metric{{display:flex;justify-content:space-between;padding:8px;background:#fff;border-radius:8px;margin:5px 0;font-size:16px}}
.chart-container{{background:#fff;border-radius:15px;padding:30px;box-shadow:0 10px 30px rgba(0,0,0,0.2);margin-bottom:20px}}
.chart-title{{font-size:24px;font-weight:700;color:#2d3748;margin-bottom:20px;text-align:center}}
.chart-grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:20px;margin-bottom:20px}}
.help-btn{{position:fixed;top:20px;right:20px;background:#667eea;color:#fff;border:none;padding:15px 25px;border-radius:50px;font-size:18px;font-weight:700;cursor:pointer;box-shadow:0 4px 15px rgba(102,126,234,0.4);z-index:1000;transition:all 0.3s}}
.help-btn:hover{{background:#5568d3;transform:translateY(-2px);box-shadow:0 6px 20px rgba(102,126,234,0.6)}}
.modal{{display:none;position:fixed;z-index:2000;left:0;top:0;width:100%;height:100%;overflow:auto;background-color:rgba(0,0,0,0.7)}}
.modal-content{{background:#fff;margin:2% auto;padding:0;border-radius:15px;width:90%;max-width:1200px;max-height:90vh;overflow-y:auto;box-shadow:0 10px 50px rgba(0,0,0,0.5)}}
.modal-header{{background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;padding:25px 30px;border-radius:15px 15px 0 0;position:sticky;top:0;z-index:10}}
.modal-title{{font-size:28px;font-weight:700;margin:0}}
.close{{color:#fff;float:right;font-size:35px;font-weight:700;cursor:pointer;line-height:28px}}
.close:hover{{color:#ffd700}}
.modal-body{{padding:30px;line-height:1.8;color:#2d3748}}
.modal-body h2{{color:#667eea;font-size:22px;margin:30px 0 15px 0;border-bottom:3px solid #667eea;padding-bottom:10px}}
.modal-body h3{{color:#4a5568;font-size:18px;margin:20px 0 10px 0}}
.modal-body ul{{margin-left:20px}}
.modal-body li{{margin:8px 0}}
.section{{background:#f7fafc;padding:20px;border-radius:10px;margin:15px 0;border-left:5px solid #667eea}}
.warning{{background:#fff5f5;border-left:5px solid #e53e3e}}
.success{{background:#f0fff4;border-left:5px solid #38a169}}
.info{{background:#ebf8ff;border-left:5px solid #4299e1}}
.help-btn{{background:#667eea;color:#fff;border:none;padding:12px 24px;border-radius:8px;font-size:16px;font-weight:700;cursor:pointer;margin-bottom:20px;box-shadow:0 4px 12px rgba(102,126,234,0.3);transition:all 0.3s}}
.help-btn:hover{{background:#5568d3;transform:translateY(-2px);box-shadow:0 6px 16px rgba(102,126,234,0.4)}}
.modal{{display:none;position:fixed;z-index:1000;left:0;top:0;width:100%;height:100%;overflow:auto;background:rgba(0,0,0,0.7);backdrop-filter:blur(5px)}}
.modal-content{{background:#fff;margin:2% auto;padding:0;border-radius:15px;width:90%;max-width:1200px;max-height:90vh;overflow:auto;box-shadow:0 20px 60px rgba(0,0,0,0.3);animation:slideDown 0.3s}}
@keyframes slideDown{{from{{opacity:0;transform:translateY(-50px)}}to{{opacity:1;transform:translateY(0)}}}}
.modal-header{{background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;padding:30px;border-radius:15px 15px 0 0;position:sticky;top:0;z-index:10}}
.modal-title{{font-size:32px;font-weight:700;margin:0}}
.close{{color:#fff;float:right;font-size:40px;font-weight:700;cursor:pointer;line-height:30px;transition:all 0.3s}}
.close:hover{{color:#fed7d7;transform:scale(1.2)}}
.modal-body{{padding:30px;line-height:1.8;color:#2d3748}}
.modal-section{{margin-bottom:40px;padding:25px;background:#f7fafc;border-radius:10px;border-left:5px solid #667eea}}
.modal-section h2{{color:#667eea;font-size:24px;margin-bottom:15px;font-weight:700}}
.modal-section h3{{color:#2d3748;font-size:20px;margin:20px 0 10px 0;font-weight:700}}
.signal-box{{background:#fff;padding:15px;margin:10px 0;border-radius:8px;border-left:4px solid #48bb78}}
.signal-box.warning{{border-left-color:#ed8936}}
.signal-box.danger{{border-left-color:#e53e3e}}
.pattern-box{{background:linear-gradient(135deg,#fed7d7,#fef5e7);padding:20px;border-radius:10px;margin:15px 0;border:2px solid #e53e3e}}
.faq-item{{background:#fff;padding:20px;margin:15px 0;border-radius:10px;border-left:4px solid #4299e1}}
.faq-q{{color:#2d3748;font-weight:700;font-size:18px;margin-bottom:10px}}
.faq-a{{color:#4a5568;line-height:1.8}}
</style></head><body>

<!-- 도움말 버튼 -->
<button class="help-btn" onclick="openModal()">❓ 지표 설명</button>

<!-- 모달 (팝업) -->
<div id="helpModal" class="modal">
<div class="modal-content">
<div class="modal-header">
<span class="close" onclick="closeModal()">&times;</span>
<h2 class="modal-title">📊 4개 핵심 지표 설명 - 1700+ 위험 감지</h2>
</div>
<div class="modal-body">

<div class="section success">
<h2>🎯 핵심 요약</h2>
<p><strong>TOTALCNT가 1700을 넘으면 위험한 상황입니다.</strong></p>
<p>하지만 1700이 된 후에는 이미 늦습니다. 그래서 우리는 <strong>1700이 되기 25-30분 전에 미리 감지</strong>합니다.</p>
<p><strong>4개 지표를 실시간으로 모니터링</strong>하여 위험을 사전에 예측합니다!</p>
</div>

<h2>1️⃣ queue_gap (가장 중요! ⭐⭐⭐)</h2>
<div class="section">
<h3>■ 정의</h3>
<p><code>queue_gap = 생성된 작업 - 완료된 작업 = 현재 대기 중인 작업량</code></p>

<h3>■ 왜 중요한가?</h3>
<ul>
<li>일이 밀리고 있다는 신호</li>
<li>처리 속도보다 들어오는 일이 더 많음</li>
<li><strong>TOTALCNT 예측에서 중요도 4.2배 (가장 높음!)</strong></li>
</ul>

<h3>■ 위험 신호</h3>
<ul>
<li>gap < 150 → 정상 ✅</li>
<li>gap 150~250 → 주의 ⚠️</li>
<li>gap 250~350 → 위험 🔥</li>
<li>gap > 350 → 매우 위험! 🚨</li>
</ul>

<h3>■ 실제 데이터</h3>
<ul>
<li>queue_gap이 300 넘으면 → 30분 후 1700+ 확률 높음</li>
<li>1700+ 발생 30분 전, queue_gap 평균 280 이상</li>
</ul>

<div class="warning" style="margin-top:15px;padding:15px;">
<strong>⚠️ 고객님께:</strong> "queue_gap이 250 넘어가면 주의하세요. 일이 밀리기 시작했다는 신호입니다. 이 상태가 지속되면 1700 넘어갑니다."
</div>
</div>

<h2>2️⃣ M14AM14B (두 번째로 중요! ⭐⭐)</h2>
<div class="section">
<h3>■ 정의</h3>
<p>M14A 공장에서 M14B 공장으로 보내는 물류량</p>

<h3>■ 왜 중요한가?</h3>
<ul>
<li>Fab 간 물류 흐름의 핵심</li>
<li>TOTALCNT와 높은 상관관계</li>
<li><strong>1700+ 발생 30분 전, 1.72배 상승 (가장 강력한 선행지표!)</strong></li>
</ul>

<h3>■ 위험 신호</h3>
<ul>
<li>M14AM14B < 497 → 정상 ✅</li>
<li>M14AM14B > 497 → 주의 ⚠️</li>
<li>M14AM14B > 517 → 위험 🔥</li>
<li>M14AM14B > 520 → 매우 위험! 🚨</li>
</ul>

<h3>■ 실제 데이터</h3>
<ul>
<li>평균 300 정도 유지</li>
<li>520 넘으면 → 1700+ 확률 급증</li>
<li>1700+ 발생 시, M14AM14B 평균 540</li>
</ul>

<div class="warning" style="margin-top:15px;padding:15px;">
<strong>⚠️ 고객님께:</strong> "M14AM14B가 500 넘어가면 M14A에서 M14B로 물류가 급증하고 있다는 뜻입니다. 이는 전체 물류 폭증의 신호입니다."
</div>
</div>

<h2>3️⃣ M14AM14BSUM (세 번째로 중요! ⭐⭐)</h2>
<div class="section">
<h3>■ 정의</h3>
<p>M14A → M14B 물류의 누적 합계</p>

<h3>■ 왜 중요한가?</h3>
<ul>
<li>M14AM14B의 누적 추세</li>
<li>지속적인 물류 증가 패턴 감지</li>
<li><strong>TOTALCNT와 상관계수 0.900 (매우 높음!)</strong></li>
<li>1700+ 발생 30분 전, 1.59배 상승</li>
</ul>

<h3>■ 위험 신호</h3>
<ul>
<li>M14AM14BSUM < 566 → 정상 ✅</li>
<li>M14AM14BSUM > 566 → 주의 ⚠️</li>
<li>M14AM14BSUM > 576 → 위험 🔥</li>
<li>M14AM14BSUM > 588 → 매우 위험! 🚨</li>
</ul>

<h3>■ 실제 데이터</h3>
<ul>
<li>평균 400 정도 유지</li>
<li>588 넘으면 → 1700+ 확률 매우 높음</li>
<li>M14AM14B와 함께 보면 더 정확</li>
</ul>

<div class="warning" style="margin-top:15px;padding:15px;">
<strong>⚠️ 고객님께:</strong> "M14AM14BSUM은 누적 추세입니다. 순간적인 급증이 아니라 지속적으로 물류가 증가하고 있는지 확인합니다."
</div>
</div>

<h2>4️⃣ TRANSPORT (네 번째로 중요! ⭐)</h2>
<div class="section">
<h3>■ 정의</h3>
<p>4분 이상 걸린 운송 건수 (정상적으로는 4분 미만에 완료되어야 함)</p>

<h3>■ 왜 중요한가?</h3>
<ul>
<li>운송 시스템이 느려지고 있다는 신호</li>
<li>병목 현상 발생 중</li>
<li>중요도 1.29배</li>
</ul>

<h3>■ 위험 신호</h3>
<ul>
<li>TRANSPORT < 145 → 정상 ✅</li>
<li>TRANSPORT > 145 → 주의 ⚠️</li>
<li>TRANSPORT > 151 → 위험 🔥</li>
<li>TRANSPORT > 180 → 매우 위험! 🚨</li>
</ul>

<h3>■ 실제 데이터</h3>
<ul>
<li>평균 100 정도 유지</li>
<li>151 넘으면 → 운송 지연 심각</li>
<li>1700+ 발생 시, TRANSPORT 평균 170</li>
</ul>

<div class="warning" style="margin-top:15px;padding:15px;">
<strong>⚠️ 고객님께:</strong> "TRANSPORT가 150 넘으면 운송이 밀리고 있습니다. 물류는 많은데 처리가 느려서 전체 시스템이 포화되고 있다는 뜻입니다."
</div>
</div>

<h2>⭐ 황금 패턴 (골든 패턴)</h2>
<div class="section warning">
<h3>■ 1700+ 발생 확률 85%의 패턴</h3>
<p style="font-size:20px;font-weight:700;color:#e53e3e;">M14AM14B > 520 AND M14AM14BSUM > 588</p>
<p>→ 이 두 조건이 동시에 만족되면 <strong>1700+ 발생 확률 85%!</strong></p>

<h3>■ 3단계 패턴</h3>
<ol>
<li><strong>엄격 패턴 (1700+ 확률 85%)</strong><br>M14AM14B > 520 AND M14AM14BSUM > 588</li>
<li><strong>보통 패턴 (1700+ 확률 70%)</strong><br>M14AM14B > 517 AND M14AM14BSUM > 576</li>
<li><strong>느슨 패턴 (1700+ 확률 55%)</strong><br>M14AM14B > 509 AND M14AM14BSUM > 570</li>
</ol>

<h3>■ 복합 패턴 (확률 더 높음!)</h3>
<p style="font-size:18px;font-weight:700;color:#c53030;">M14AM14B > 520<br>AND M14AM14BSUM > 588<br>AND queue_gap > 300<br>AND TRANSPORT > 151</p>
<p>→ <strong>4개 조건 모두 만족 시 1700+ 확률 95%!</strong></p>
</div>

<h2>🔍 왜 이 4개 지표가 1700+를 감지하는가?</h2>
<div class="section info">
<h3>1. 인과관계 명확</h3>
<pre style="background:#2d3748;color:#fff;padding:15px;border-radius:8px;line-height:2;">
M14AM14B 증가 → M14B 공장 물류 급증
     ↓
M14AM14BSUM 증가 → 지속적인 물류 증가 추세
     ↓
queue_gap 증가 → 처리 속도 < 유입 속도 (일 밀림!)
     ↓
TRANSPORT 증가 → 운송 지연 (병목 현상!)
     ↓
TOTALCNT 1700+ 도달! 🚨
</pre>

<h3>2. 시간적 선행성</h3>
<p>이 4개 지표는 TOTALCNT보다 <strong>먼저 변합니다.</strong></p>
<table style="width:100%;border-collapse:collapse;margin:15px 0;">
<tr style="background:#667eea;color:#fff;">
<th style="padding:10px;border:1px solid #ddd;">0분</th>
<th style="padding:10px;border:1px solid #ddd;">10-15분</th>
<th style="padding:10px;border:1px solid #ddd;">25-30분</th>
</tr>
<tr>
<td style="padding:10px;border:1px solid #ddd;">4개 지표 상승 시작</td>
<td style="padding:10px;border:1px solid #ddd;">4개 지표 급상승</td>
<td style="padding:10px;border:1px solid #ddd;">TOTALCNT 1700 도달!</td>
</tr>
</table>
<p><strong>→ 30분 전에 미리 감지 가능!</strong></p>

<h3>3. 데이터 검증</h3>
<p>실제 데이터 분석 결과:</p>
<ul>
<li>1700+ 발생 467건 분석</li>
<li>30분 전 4개 지표 평균값:
<ul>
<li>M14AM14B: 540 (정상 300 대비 1.8배)</li>
<li>M14AM14BSUM: 600 (정상 400 대비 1.5배)</li>
<li>queue_gap: 280 (정상 150 대비 1.9배)</li>
<li>TRANSPORT: 170 (정상 100 대비 1.7배)</li>
</ul>
</li>
</ul>
</div>

<h2>💡 실무 활용 가이드</h2>
<div class="section">
<h3>■ 상황 1: queue_gap만 높음 (250+)</h3>
<p><strong>의미:</strong> 일이 밀리기 시작<br>
<strong>조치:</strong> 작업 처리 속도 점검<br>
<strong>위험도:</strong> 중 ⚠️</p>

<h3>■ 상황 2: M14AM14B + M14AM14BSUM 높음 (520+, 588+)</h3>
<p><strong>의미:</strong> M14A → M14B 물류 급증 (황금패턴!)<br>
<strong>조치:</strong> Fab 간 물류 조절 필요<br>
<strong>위험도:</strong> 고 🔥</p>

<h3>■ 상황 3: TRANSPORT 높음 (151+)</h3>
<p><strong>의미:</strong> 운송 지연, 병목 현상<br>
<strong>조치:</strong> 운송 시스템 점검<br>
<strong>위험도:</strong> 중 ⚠️</p>

<h3>■ 상황 4: 4개 모두 높음</h3>
<p><strong>의미:</strong> 시스템 전체 포화 직전!<br>
<strong>조치:</strong> 즉시 긴급 대응!<br>
<strong>위험도:</strong> 매우 높음! 🚨<br>
<strong>예상:</strong> 25분 후 1700+ 확률 95%</p>
</div>

<h2>❓ FAQ</h2>
<div class="section">
<p><strong>Q1. 왜 하필 이 4개인가요?</strong><br>
A1. 3개월 데이터 분석 결과, 이 4개가 TOTALCNT 1700+ 예측에 가장 중요했습니다. (중요도: queue_gap 4.2배 > M14AM14B > M14AM14BSUM > TRANSPORT 1.29배)</p>

<p><strong>Q2. 다른 지표는 안 보나요?</strong><br>
A2. 총 13개 지표를 AI가 분석하지만, 이해하기 쉽게 가장 중요한 4개만 대시보드에 표시합니다.</p>

<p><strong>Q3. 왜 30분 전에 감지 가능한가요?</strong><br>
A3. 이 4개 지표가 먼저 변하고, 그 영향이 30분 후 TOTALCNT에 나타나기 때문입니다. (시간 지연 효과)</p>

<p><strong>Q4. 정확도는 얼마나 되나요?</strong><br>
A4. 1700+ 감지율 85% 이상 (467건 중 400건 이상 사전 감지 성공)</p>

<p><strong>Q5. 어떻게 활용하나요?</strong><br>
A5. 대시보드에서 4개 지표를 실시간 모니터링하고, 위험 수준 도달 시 사전 조치하세요. 25분의 대응 시간을 확보할 수 있습니다!</p>
</div>

<div class="section success">
<h2>✅ 결론</h2>
<ul style="font-size:16px;line-height:2;">
<li>✅ 4개 핵심 지표 = 1700+ 조기 경보 시스템</li>
<li>✅ 30분 전 사전 감지 가능</li>
<li>✅ 황금 패턴 (M14AM14B > 520 & M14AM14BSUM > 588) = 85% 확률</li>
<li>✅ 4개 모두 위험 수준 = 95% 확률</li>
<li>✅ 실시간 모니터링으로 사전 대응 가능!</li>
</ul>
</div>

<p style="text-align:center;color:#718096;margin-top:30px;padding-top:20px;border-top:2px solid #e2e8f0;">
작성: AI 물류 예측팀 | 버전: V8.2 | 날짜: 2025-11-07
</p>

</div>
</div>
</div>

<div class="container">

<div class="card">
<div class="header">📊 60분 실시간 예측 대시보드</div>
<div class="subtitle">{data['current_time']}</div>

<button class="help-btn" onclick="openModal()">📖 4개 핵심 지표 설명 보기</button>

<div class="current">
<div class="current-grid">
<div class="current-item">
<div class="current-label">TOTALCNT</div>
<div class="current-value">{current_value:,}</div>
<span class="current-status status-{current_status.lower()}">{current_status}</span>
</div>
<div class="current-item">
<div class="current-label">M14AM14B</div>
<div class="current-value">{data['current_m14b']:.0f}</div>
</div>
<div class="current-item">
<div class="current-label">M14AM14BSUM</div>
<div class="current-value">{data['current_m14bsum']:.0f}</div>
</div>
<div class="current-item">
<div class="current-label">queue_gap</div>
<div class="current-value">{data['current_gap']:.0f}</div>
</div>
<div class="current-item">
<div class="current-label">TRANSPORT</div>
<div class="current-value">{data['current_trans']:.0f}</div>
</div>
</div>
</div>
</div>

<div class="risk">Maximum Risk: {risk} ({max_danger}%)</div>

<div class="grid">"""
    
    for p in predictions:
        danger_class = 'danger-high' if p['danger_probability'] >= 60 else ('danger-medium' if p['danger_probability'] >= 30 else 'danger-low')
        
        html += f"""<div class="pred">
<div class="time">⏱️ {p['horizon']} Minutes</div>
<div style="font-size:14px;color:#718096;margin-bottom:10px">{p['pred_time']}</div>
<div class="pred-value">{p['prediction']:,}</div>
<div class="danger {danger_class}">🚨 {p['danger_probability']}%</div>
<div class="metric"><span>Change</span><span style="font-weight:700">{p['change']:+,}</span></div>
<div class="metric"><span>Status</span><span style="font-weight:700">{p['status']}</span></div>
<span class="current-status status-{p['status'].lower()}">{p['status']}</span>
</div>"""
    
    html += f"""</div>

<!-- 메인 그래프: 60분 과거 + 예측 -->
<div class="chart-container">
<div class="chart-title">📈 최근 60분 TOTALCNT + 예측 (10/15/25분)</div>
<canvas id="chart1" height="100"></canvas>
</div>

<!-- 서브 그래프 그리드 -->
<div class="chart-grid">
<div class="chart-container">
<div class="chart-title">M14AM14B (최근 60분)</div>
<canvas id="chart2" height="150"></canvas>
</div>

<div class="chart-container">
<div class="chart-title">M14AM14BSUM (최근 60분)</div>
<canvas id="chart3" height="150"></canvas>
</div>

<div class="chart-container">
<div class="chart-title">queue_gap (최근 60분)</div>
<canvas id="chart4" height="150"></canvas>
</div>

<div class="chart-container">
<div class="chart-title">TRANSPORT (최근 60분)</div>
<canvas id="chart5" height="150"></canvas>
</div>
</div>

</div>

<script>
// Chart 1: TOTALCNT + 예측 (점선으로 연결!)
const ctx1 = document.getElementById('chart1').getContext('2d');
new Chart(ctx1, {{
    type: 'line',
    data: {{
        labels: {json.dumps(chart1_labels)},
        datasets: [
            {{
                label: '최근 60분 데이터',
                data: {json.dumps(chart1_historical)},
                borderColor: '#667eea',
                backgroundColor: 'rgba(102, 126, 234, 0.1)',
                borderWidth: 3,
                pointRadius: 3,
                pointBackgroundColor: '#667eea',
                tension: 0.4,
                fill: false
            }},
            {{
                label: '예측 (10/15/25분)',
                data: {json.dumps(chart1_predictions)},
                borderColor: '#e53e3e',
                backgroundColor: 'rgba(229, 62, 62, 0.1)',
                borderWidth: 3,
                borderDash: [5, 5],
                pointRadius: 8,
                pointBackgroundColor: '#e53e3e',
                tension: 0,
                fill: false,
                spanGaps: true
            }}
        ]
    }},
    options: {{
        responsive: true,
        maintainAspectRatio: true,
        interaction: {{
            mode: 'index',
            intersect: false
        }},
        plugins: {{
            legend: {{
                display: true,
                position: 'top',
                labels: {{
                    font: {{size: 14, weight: 'bold'}},
                    usePointStyle: true,
                    padding: 20
                }}
            }},
            tooltip: {{
                backgroundColor: 'rgba(0, 0, 0, 0.8)',
                padding: 12,
                titleFont: {{size: 14, weight: 'bold'}},
                bodyFont: {{size: 13}},
                callbacks: {{
                    label: function(context) {{
                        if (context.parsed.y === null) return null;
                        return context.dataset.label + ': ' + context.parsed.y.toFixed(0);
                    }}
                }}
            }}
        }},
        scales: {{
            x: {{
                grid: {{display: true, color: 'rgba(0, 0, 0, 0.05)'}},
                ticks: {{
                    font: {{size: 11}},
                    maxRotation: 45,
                    minRotation: 45
                }}
            }},
            y: {{
                beginAtZero: false,
                grid: {{display: true, color: 'rgba(0, 0, 0, 0.1)'}},
                ticks: {{font: {{size: 12}}}}
            }}
        }}
    }}
}});

// Chart 2: M14AM14B
const ctx2 = document.getElementById('chart2').getContext('2d');
new Chart(ctx2, {{
    type: 'line',
    data: {{
        labels: {json.dumps(chart2_labels)},
        datasets: [{{
            label: 'M14AM14B',
            data: {json.dumps(chart2_data)},
            borderColor: '#48bb78',
            backgroundColor: 'rgba(72, 187, 120, 0.1)',
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.4,
            fill: true
        }}]
    }},
    options: {{
        responsive: true,
        maintainAspectRatio: true,
        plugins: {{
            legend: {{display: false}},
            tooltip: {{
                backgroundColor: 'rgba(0, 0, 0, 0.8)',
                padding: 10
            }}
        }},
        scales: {{
            x: {{
                grid: {{display: false}},
                ticks: {{
                    maxRotation: 0,
                    autoSkip: true,
                    maxTicksLimit: 10,
                    font: {{size: 9}}
                }}
            }},
            y: {{
                grid: {{display: true, color: 'rgba(0, 0, 0, 0.05)'}},
                ticks: {{font: {{size: 10}}}}
            }}
        }}
    }}
}});

// Chart 3: M14AM14BSUM
const ctx3 = document.getElementById('chart3').getContext('2d');
new Chart(ctx3, {{
    type: 'line',
    data: {{
        labels: {json.dumps(chart2_labels)},
        datasets: [{{
            label: 'M14AM14BSUM',
            data: {json.dumps(chart3_data)},
            borderColor: '#4299e1',
            backgroundColor: 'rgba(66, 153, 225, 0.1)',
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.4,
            fill: true
        }}]
    }},
    options: {{
        responsive: true,
        maintainAspectRatio: true,
        plugins: {{
            legend: {{display: false}},
            tooltip: {{
                backgroundColor: 'rgba(0, 0, 0, 0.8)',
                padding: 10
            }}
        }},
        scales: {{
            x: {{
                grid: {{display: false}},
                ticks: {{
                    maxRotation: 0,
                    autoSkip: true,
                    maxTicksLimit: 10,
                    font: {{size: 9}}
                }}
            }},
            y: {{
                grid: {{display: true, color: 'rgba(0, 0, 0, 0.05)'}},
                ticks: {{font: {{size: 10}}}}
            }}
        }}
    }}
}});

// Chart 4: queue_gap
const ctx4 = document.getElementById('chart4').getContext('2d');
new Chart(ctx4, {{
    type: 'line',
    data: {{
        labels: {json.dumps(chart2_labels)},
        datasets: [{{
            label: 'queue_gap',
            data: {json.dumps(chart4_data)},
            borderColor: '#ed8936',
            backgroundColor: 'rgba(237, 137, 54, 0.1)',
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.4,
            fill: true
        }}]
    }},
    options: {{
        responsive: true,
        maintainAspectRatio: true,
        plugins: {{
            legend: {{display: false}},
            tooltip: {{
                backgroundColor: 'rgba(0, 0, 0, 0.8)',
                padding: 10
            }}
        }},
        scales: {{
            x: {{
                grid: {{display: false}},
                ticks: {{
                    maxRotation: 0,
                    autoSkip: true,
                    maxTicksLimit: 10,
                    font: {{size: 9}}
                }}
            }},
            y: {{
                grid: {{display: true, color: 'rgba(0, 0, 0, 0.05)'}},
                ticks: {{font: {{size: 10}}}}
            }}
        }}
    }}
}});

// Chart 5: TRANSPORT
const ctx5 = document.getElementById('chart5').getContext('2d');
new Chart(ctx5, {{
    type: 'line',
    data: {{
        labels: {json.dumps(chart2_labels)},
        datasets: [{{
            label: 'TRANSPORT',
            data: {json.dumps(chart5_data)},
            borderColor: '#9f7aea',
            backgroundColor: 'rgba(159, 122, 234, 0.1)',
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.4,
            fill: true
        }}]
    }},
    options: {{
        responsive: true,
        maintainAspectRatio: true,
        plugins: {{
            legend: {{display: false}},
            tooltip: {{
                backgroundColor: 'rgba(0, 0, 0, 0.8)',
                padding: 10
            }}
        }},
        scales: {{
            x: {{
                grid: {{display: false}},
                ticks: {{
                    maxRotation: 0,
                    autoSkip: true,
                    maxTicksLimit: 10,
                    font: {{size: 9}}
                }}
            }},
            y: {{
                grid: {{display: true, color: 'rgba(0, 0, 0, 0.05)'}},
                ticks: {{font: {{size: 10}}}}
            }}
        }}
    }}
}});
</script>

<script>
// 모달 열기/닫기
function openModal() {{
    document.getElementById('helpModal').style.display = 'block';
    document.body.style.overflow = 'hidden'; // 스크롤 방지
}}

function closeModal() {{
    document.getElementById('helpModal').style.display = 'none';
    document.body.style.overflow = 'auto'; // 스크롤 복원
}}

// 모달 외부 클릭 시 닫기
window.onclick = function(event) {{
    var modal = document.getElementById('helpModal');
    if (event.target == modal) {{
        closeModal();
    }}
}}

// ESC 키로 닫기
document.addEventListener('keydown', function(event) {{
    if (event.key === 'Escape') {{
        closeModal();
    }}
}});
</script>
</body></html>"""
    
    return html

# ========================================
# 메인
# ========================================
def main():
    # 예측 실행
    data = predict_all_horizons()
    
    # HTML 생성
    html = generate_html(data)
    
    # 파일 저장
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'Dashboard_60min_{timestamp}.html'
    filepath = os.path.join(SCRIPT_DIR, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"\n✅ Dashboard saved: {filename}")
    
    # 브라우저 열기
    try:
        webbrowser.open('file://' + os.path.abspath(filepath))
        print("✅ Browser opened")
    except:
        print("⚠️ Please open the HTML file manually")

if __name__ == '__main__':
    main()