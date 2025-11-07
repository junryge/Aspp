#!/usr/bin/env python3

    # 1700+ 위험 확률 계산
    danger_prob = 0
    
    # 황금 패턴 기반
    if gold_strict:
        danger_prob = 85  # ⭐⭐⭐ 엄격 패턴
    elif gold_normal:
        danger_prob = 70  # ⭐⭐ 보통 패턴
    elif (current_m14b > 509 and current_m14bsum > 570):
        danger_prob = 55  # ⭐ 느슨 패턴
    elif (current_m14b > 497 and current_m14bsum > 566):
        danger_prob = 40  # 필수 조건
    
    # Gap 위험도 추가
    if danger_gap:
        if current_gap > 350:
            danger_prob += 15
        else:
            danger_prob += 10
    
    # Transport 위험도 추가
    if danger_trans:
        if current_trans > 180:
            danger_prob += 15
        else:
            danger_prob += 10
    
    # 현재 값이 이미 1700+ 인 경우
    if current_totalcnt >= 1700:
        danger_prob = max(danger_prob, 90)
    
    # 현재 값이 1600+ 인 경우
    elif current_totalcnt >= 1600:
        danger_prob = max(danger_prob, 60)
    
    # 최대 95%로 제한
    danger_prob = min(danger_prob, 95)
    
    # 결과 딕셔너리 반환
    result = {
        'prediction': int(pred),
        'status': pred_status,
        'prediction_time': prediction_time.strftime('%Y-%m-%d %H:%M'),
        'danger_probability': danger_prob
    }
    
    return result

# -*- coding: utf-8 -*-
"""
🚀 실시간 예측 시스템 - V8.2 평가 코드 기반
multiplier=90, boost 보정 적용
"""

import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
import os

# ========================================
# Feature 생성 함수 (평가 코드와 100% 동일!)
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
# Boost 함수 (평가 코드와 100% 동일!)
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
# 상태 판정 함수 (영문!)
# ========================================
def get_status_info(value):
    """
    물류량에 따른 상태 정보 반환 (영문)
    < 900: LOW
    900-1599: NORMAL
    1600-1699: CAUTION
    1700+: CRITICAL
    """
    if value < 900:
        return 'LOW'
    elif value < 1600:
        return 'NORMAL'
    elif value < 1700:
        return 'CAUTION'
    else:
        return 'CRITICAL'

# ========================================
# 실시간 예측 함수
# ========================================
def predict_latest():
    """
    가장 최근 280분 데이터로 10분 후 예측
    반환: {'예측물류량': int, '예측상태': str}
    """
    
    print("="*80)
    print("🚀 실시간 예측 시스템 - V8.2")
    print("="*80)
    
    # 모델 로드
    model_file = 'model_13col_final.pkl'
    try:
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ 모델 로드 완료: {model_file}")
    except Exception as e:
        print(f"❌ 모델 파일 없음: {e}")
        print("먼저 V8_2학습.py를 실행하여 모델을 학습하세요.")
        return None
    
    # 데이터 로드
    csv_file = 'data/M14_Q_20251014_TO_20251015.CSV'
    if not os.path.exists(csv_file):
        print(f"❌ CSV 파일 없음: {csv_file}")
        return None
    
    df = pd.read_csv(csv_file, on_bad_lines='skip')
    print(f"✅ 데이터 로드 완료: {len(df):,}개 행")
    
    # 필수 컬럼 확인 (13개)
    required_cols = [
        'M14AM14B', 'M14AM14BSUM', 'M14BM14A',
        'M14AM10A', 'M10AM14A', 'M16M14A', 'M14AM16SUM', 'TOTALCNT',
        'M14.QUE.ALL.CURRENTQCREATED', 'M14.QUE.ALL.CURRENTQCOMPLETED',
        'M14.QUE.OHT.OHTUTIL', 'M14.QUE.ALL.TRANSPORT4MINOVERCNT'
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ 필수 컬럼 누락: {missing_cols}")
        print(f"현재 컬럼: {list(df.columns)}")
        return None
    
    print(f"✅ 필수 컬럼 확인 완료 (13개)")
    
    # 데이터 길이 확인
    if len(df) < 280:
        print(f"❌ 데이터 부족: 최소 280개 필요, 현재 {len(df)}개")
        return None
    
    # CURRTIME 처리
    if 'CURRTIME' in df.columns:
        try:
            df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M')
            print("✅ CURRTIME 파싱 완료")
        except:
            try:
                df['CURRTIME'] = pd.to_datetime(df['CURRTIME'])
                print("✅ CURRTIME 자동 파싱 완료")
            except:
                print("⚠️ CURRTIME 변환 실패, 가상 시간 생성")
                base_time = datetime(2024, 1, 1, 0, 0)
                df['CURRTIME'] = [base_time + timedelta(minutes=i) for i in range(len(df))]
    else:
        print("⚠️ CURRTIME 없음, 가상 시간 생성")
        base_time = datetime(2024, 1, 1, 0, 0)
        df['CURRTIME'] = [base_time + timedelta(minutes=i) for i in range(len(df))]
    
    # 최근 280분 데이터 추출
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
    prediction_time = current_time + timedelta(minutes=10)
    
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
    
    # Feature 생성
    print(f"\n🔄 Feature 생성 중...")
    features = create_features_13col_optimized(row_dict)
    X_pred = pd.DataFrame([features])
    
    # 원본 예측
    print(f"🔮 예측 중...")
    pred_raw = model.predict(X_pred)[0]
    
    # boost 보정 적용
    pred = adjust_light_plus(pred_raw, current_m14b, current_m14bsum, current_gap, current_trans)
    
    # 상태 판정
    current_status = get_status_info(current_totalcnt)
    pred_status = get_status_info(pred)
    
    # 패턴 감지
    gold_strict = (current_m14b > 520 and current_m14bsum > 588)
    gold_normal = (current_m14b > 517 and current_m14bsum > 576)
    danger_gap = current_gap > 300
    danger_trans = current_trans > 151
    
    # 결과 출력
    print("\n" + "="*80)
    print("🔮 예측 결과")
    print("="*80)
    print(f"현재 시간:        {current_time.strftime('%Y-%m-%d %H:%M')}")
    print(f"예측 시점:        {prediction_time.strftime('%Y-%m-%d %H:%M')} (10분 후)")
    print("-"*80)
    print(f"현재 물류량:      {int(current_totalcnt):,}  [{current_status}]")
    print(f"원본 예측:        {int(pred_raw):,}")
    print(f"보정 예측:        {int(pred):,}  [{pred_status}]")
    print(f"변화량:          {int(pred - current_totalcnt):+,}")
    print("-"*80)
    print(f"M14AM14B:        {int(current_m14b):,}")
    print(f"M14AM14BSUM:     {int(current_m14bsum):,}")
    print(f"queue_gap:       {int(current_gap):,}")
    print(f"TRANSPORT:       {int(current_trans):,}")
    if gold_strict:
        print(f"황금 패턴:        ⭐⭐⭐ 엄격 (1700+ 확률 85%!)")
    elif gold_normal:
        print(f"황금 패턴:        ⭐⭐ 보통 (1700+ 확률 70%)")
    if danger_gap:
        print(f"Gap 위험:        🚨 매우 높음 (>300)")
    if danger_trans:
        print(f"Transport 위험:  🚨 매우 높음 (>151)")
    print("="*80)
    
    # 결과 딕셔너리 반환
    result = {
        '예측물류량': int(pred),
        '예측상태': pred_status
    }
    
    return result

if __name__ == '__main__':
    print("\n🚀 실시간 예측 시스템 V8.2 시작...\n")
    
    # 최신 예측 실행
    result = predict_latest()
    
    if result is not None:
        print(f"\n✅ 예측 완료!")
        print(f"   예측물류량: {result['예측물류량']:,}")
        print(f"   예측상태: {result['예측상태']}")
        print("\n" + "="*80)