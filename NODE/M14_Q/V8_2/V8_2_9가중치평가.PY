#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 XGBoost 평가 - 13개 컬럼 최적화 버전
VERTICAL 제거, Queue Gap & TRANSPORT 강화!
"""

import numpy as np
import pandas as pd
import pickle
from datetime import timedelta
import os
import gc
import warnings
warnings.filterwarnings('ignore')

def create_features_13col_optimized(row_dict):
    """13개 컬럼 최적화 Feature 생성 (학습 코드와 완전 동일)"""
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
    
    # 파생 컬럼 (Queue Gap - 4.2배 차이!)
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
    # 12. Transport (10개) - 1.29배 중요!
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
    # 13. Queue Gap (10개) - 4.2배 차이!
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
    # Interaction Features (30개) - 핵심!
    # ========================================
    
    # M14AM14B × M14AM14BSUM (100% 필수 조건!)
    features['m14b_x_m14bsum'] = seq_m14b[-1] * seq_m14b_sum[-1] / 1000
    features['m14b_x_m14bsum_mean'] = np.mean(seq_m14b * seq_m14b_sum) / 1000
    features['m14bsum_per_m14b'] = seq_m14b_sum[-1] / (seq_m14b[-1] + 1)
    features['m14b_plus_m14bsum'] = seq_m14b[-1] + seq_m14b_sum[-1]
    
    # Queue Gap 조합 (4.2배 차이!)
    features['gap_x_m14b'] = seq_queue_gap[-1] * seq_m14b[-1] / 1000
    features['gap_x_m14bsum'] = seq_queue_gap[-1] * seq_m14b_sum[-1] / 1000
    features['gap_x_total'] = seq_queue_gap[-1] * seq_totalcnt[-1] / 1000
    features['gap_per_total'] = seq_queue_gap[-1] / (seq_totalcnt[-1] + 1)
    
    # Transport 조합 (1.29배!)
    features['trans_x_m14b'] = seq_transport[-1] * seq_m14b[-1] / 100
    features['trans_x_m14bsum'] = seq_transport[-1] * seq_m14b_sum[-1] / 100
    features['trans_x_gap'] = seq_transport[-1] * seq_queue_gap[-1] / 100
    features['trans_x_oht'] = seq_transport[-1] * seq_oht[-1] / 10
    
    # 3중 조합
    features['triple_danger'] = seq_m14b[-1] * seq_m14b_sum[-1] * seq_transport[-1] / 100000
    features['gap_trans_m14b'] = seq_queue_gap[-1] * seq_transport[-1] * seq_m14b[-1] / 100000
    
    # 기타 조합
    features['m10arev_x_m14b'] = seq_m10a_rev[-1] * seq_m14b[-1] / 100
    features['oht_x_m14bsum'] = seq_oht[-1] * seq_m14b_sum[-1] / 10
    features['m16rev_x_total'] = seq_m16_rev[-1] * seq_totalcnt[-1] / 100
    
    # 비율
    features['ratio_m14b_total'] = seq_m14b[-1] / (seq_totalcnt[-1] + 1)
    features['ratio_m14bsum_total'] = seq_m14b_sum[-1] / (seq_totalcnt[-1] + 1)
    features['ratio_gap_m14b'] = seq_queue_gap[-1] / (seq_m14b[-1] + 1)
    features['ratio_trans_total'] = seq_transport[-1] / (seq_totalcnt[-1] + 1)
    
    # 변동성
    features['vol_m14b'] = np.std(seq_m14b) / (np.mean(seq_m14b) + 1)
    features['vol_m14bsum'] = np.std(seq_m14b_sum) / (np.mean(seq_m14b_sum) + 1)
    features['vol_total'] = np.std(seq_totalcnt) / (np.mean(seq_totalcnt) + 1)
    features['vol_gap'] = np.std(seq_queue_gap) / (np.mean(np.abs(seq_queue_gap)) + 1)
    features['vol_trans'] = np.std(seq_transport) / (np.mean(seq_transport) + 1)
    
    # 상관
    features['corr_m14b_total'] = np.corrcoef(seq_m14b, seq_totalcnt)[0, 1]
    features['corr_m14bsum_total'] = np.corrcoef(seq_m14b_sum, seq_totalcnt)[0, 1]
    features['corr_gap_total'] = np.corrcoef(seq_queue_gap, seq_totalcnt)[0, 1]
    features['corr_trans_total'] = np.corrcoef(seq_transport, seq_totalcnt)[0, 1]
    
    # ========================================
    # 임계값 카운트 (35개)
    # ========================================
    
    # M14AM14B (데이터 기반!)
    features['m14b_over_497'] = np.sum(seq_m14b > 497)
    features['m14b_over_517'] = np.sum(seq_m14b > 517)
    features['m14b_over_520'] = np.sum(seq_m14b > 520)
    features['m14b_over_539'] = np.sum(seq_m14b > 539)
    
    # M14AM14BSUM (데이터 기반!)
    features['m14bsum_over_566'] = np.sum(seq_m14b_sum > 566)
    features['m14bsum_over_576'] = np.sum(seq_m14b_sum > 576)
    features['m14bsum_over_588'] = np.sum(seq_m14b_sum > 588)
    features['m14bsum_over_602'] = np.sum(seq_m14b_sum > 602)
    
    # Queue Gap (데이터 기반!)
    features['gap_over_200'] = np.sum(seq_queue_gap > 200)
    features['gap_over_250'] = np.sum(seq_queue_gap > 250)
    features['gap_over_300'] = np.sum(seq_queue_gap > 300)
    features['gap_over_350'] = np.sum(seq_queue_gap > 350)
    
    # Transport (데이터 기반!)
    features['trans_over_145'] = np.sum(seq_transport > 145)
    features['trans_over_151'] = np.sum(seq_transport > 151)
    features['trans_over_171'] = np.sum(seq_transport > 171)
    features['trans_over_180'] = np.sum(seq_transport > 180)
    
    # OHT
    features['oht_over_83'] = np.sum(seq_oht > 83.6)
    features['oht_over_84'] = np.sum(seq_oht > 84.6)
    features['oht_over_86'] = np.sum(seq_oht > 85.6)
    
    # TOTALCNT
    features['total_over_1500'] = np.sum(seq_totalcnt >= 1500)
    features['total_over_1600'] = np.sum(seq_totalcnt >= 1600)
    features['total_over_1700'] = np.sum(seq_totalcnt >= 1700)
    features['total_over_1600_last30'] = np.sum(seq_totalcnt[-30:] >= 1600)
    
    # 기타
    features['m10arev_over_55'] = np.sum(seq_m10a_rev > 55)
    features['m10arev_over_59'] = np.sum(seq_m10a_rev > 59)
    features['m10a_under_80'] = np.sum(seq_m10a < 80)
    features['m10a_under_70'] = np.sum(seq_m10a < 70)
    features['m16rev_over_128'] = np.sum(seq_m16_rev > 128)
    features['m16rev_over_136'] = np.sum(seq_m16_rev > 136)
    
    # 최근 30분
    features['m14b_over_517_last30'] = np.sum(seq_m14b[-30:] > 517)
    features['m14bsum_over_576_last30'] = np.sum(seq_m14b_sum[-30:] > 576)
    features['gap_over_250_last30'] = np.sum(seq_queue_gap[-30:] > 250)
    features['trans_over_151_last30'] = np.sum(seq_transport[-30:] > 151)
    
    # ========================================
    # 황금 패턴 (20개) - 데이터 기반!
    # ========================================
    
    # 필수 조건 (100%!)
    features['must_condition'] = 1 if (seq_m14b[-1] > 497 and seq_m14b_sum[-1] > 566) else 0
    
    # 강력한 신호
    features['gold_strict'] = 1 if (seq_m14b[-1] > 520 and seq_m14b_sum[-1] > 588) else 0
    features['gold_normal'] = 1 if (seq_m14b[-1] > 517 and seq_m14b_sum[-1] > 576) else 0
    features['gold_loose'] = 1 if (seq_m14b[-1] > 509 and seq_m14b_sum[-1] > 570) else 0
    
    # 복합 신호
    features['danger_gap'] = 1 if seq_queue_gap[-1] > 300 else 0
    features['danger_trans'] = 1 if seq_transport[-1] > 151 else 0
    features['danger_oht'] = 1 if seq_oht[-1] > 84.6 else 0
    
    # 3중 체크
    features['triple_check'] = 1 if (seq_m14b[-1] > 517 and seq_m14b_sum[-1] > 576 and seq_queue_gap[-1] > 250) else 0
    features['quad_check'] = 1 if (seq_m14b[-1] > 517 and seq_m14b_sum[-1] > 576 and seq_queue_gap[-1] > 250 and seq_transport[-1] > 145) else 0
    
    # 현재 상태
    features['danger_1700'] = 1 if seq_totalcnt[-1] >= 1700 else 0
    features['danger_1600'] = 1 if seq_totalcnt[-1] >= 1600 else 0
    features['in_1700'] = 1 if seq_totalcnt[-1] >= 1700 else 0
    features['rising_1700'] = 1 if (seq_totalcnt[-1] >= 1700 and seq_totalcnt[-1] - seq_totalcnt[-10] > 20) else 0
    features['stable_1700'] = 1 if (seq_totalcnt[-1] >= 1700 and abs(seq_totalcnt[-1] - seq_totalcnt[-10]) <= 20) else 0
    features['falling_1700'] = 1 if (seq_totalcnt[-1] >= 1700 and seq_totalcnt[-1] - seq_totalcnt[-10] < -20) else 0
    
    # 추세
    features['trend_10min'] = seq_totalcnt[-1] - seq_totalcnt[-10]
    features['trend_30min'] = seq_totalcnt[-1] - seq_totalcnt[-30]
    
    # 특수 패턴
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

def evaluate_all_predictions():
    """전체 데이터를 슬라이딩 윈도우로 평가 - 13개 컬럼"""
    
    print("="*80)
    print("🚀 XGBoost 평가 - 13개 컬럼 최적화 (280분→10분)")
    print("="*80)
    print("VERTICAL 제거, Queue Gap & TRANSPORT 강화!")
    print("="*80)
    
    # 모델 로드
    model_files = [
        '/mnt/user-data/outputs/model_13col_optimized.pkl',
        'model_13col_optimized.pkl',
        '/home/claude/model_13col_optimized.pkl'
    ]
    
    model = None
    model_file = None
    
    for mf in model_files:
        if os.path.exists(mf):
            try:
                with open(mf, 'rb') as f:
                    model = pickle.load(f)
                model_file = mf
                print(f"✅ 모델 로드: {model_file}")
                break
            except Exception as e:
                print(f"⚠️ 로드 실패 ({mf}): {e}")
    
    if model is None:
        print(f"❌ 모델 파일 없음!")
        print(f"\n먼저 학습_13컬럼_최적화.py를 실행하세요.")
        return None
    
    # 데이터 로드
    csv_file = '/mnt/user-data/uploads/ssss.CSV'
    
    if not os.path.exists(csv_file):
        print(f"❌ CSV 파일 없음: {csv_file}")
        return None
    
    try:
        print(f"데이터 로딩: {csv_file}...")
        df = pd.read_csv(csv_file, on_bad_lines='skip')
        print(f"✅ 데이터 로드: {len(df):,}행")
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return None
    
    # 필수 컬럼 확인 (13개)
    required_cols = [
        'CURRTIME', 'M14AM14B', 'M14AM14BSUM', 'M14BM14A',
        'M14AM10A', 'M10AM14A', 'M16M14A', 'M14AM16SUM', 'TOTALCNT',
        'M14.QUE.ALL.CURRENTQCREATED', 'M14.QUE.ALL.CURRENTQCOMPLETED',
        'M14.QUE.OHT.OHTUTIL', 'M14.QUE.ALL.TRANSPORT4MINOVERCNT'
    ]
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ 필수 컬럼 누락: {missing_cols}")
        return None
    
    print(f"✅ 13개 컬럼 확인 완료 (VERTICAL 제외)")
    
    # CURRTIME 처리
    df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M')
    print("✅ CURRTIME 파싱 완료")
    
    results = []
    
    total_predictions = len(df) - 280 - 10
    print(f"\n🔄 슬라이딩 윈도우 평가 시작...")
    print(f"총 예측 수: {total_predictions:,}개")
    
    # 슬라이딩 윈도우
    for i in range(280, len(df) - 10):
        # 280분 시퀀스
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
        seq_m10a = row_dict['M14AM10A']
        seq_m10a_rev = row_dict['M10AM14A']
        seq_transport = row_dict['M14.QUE.ALL.TRANSPORT4MINOVERCNT']
        
        seq_queue_created = row_dict['M14.QUE.ALL.CURRENTQCREATED']
        seq_queue_completed = row_dict['M14.QUE.ALL.CURRENTQCOMPLETED']
        seq_oht = row_dict['M14.QUE.OHT.OHTUTIL']
        
        seq_queue_gap = seq_queue_created - seq_queue_completed
        
        # 시간 정보
        current_time = df['CURRTIME'].iloc[i-1]
        seq_start_time = df['CURRTIME'].iloc[i-280]
        prediction_time = current_time + timedelta(minutes=10)
        actual_time = df['CURRTIME'].iloc[i+9]
        
        # 실제값 (10분 후)
        actual_value = df['TOTALCNT'].iloc[i+9]
        
        # Feature 생성
        features = create_features_13col_optimized(row_dict)
        X_pred = pd.DataFrame([features])
        
        # 예측
        prediction = model.predict(X_pred)[0]
        
        # 패턴 감지 (데이터 기반!)
        gold_strict = (seq_m14b[-1] > 520 and seq_m14b_sum[-1] > 588)
        gold_normal = (seq_m14b[-1] > 517 and seq_m14b_sum[-1] > 576)
        danger_gap = (seq_queue_gap[-1] > 300)
        danger_trans = (seq_transport[-1] > 151)
        quad_check = (seq_m14b[-1] > 517 and seq_m14b_sum[-1] > 576 and seq_queue_gap[-1] > 250 and seq_transport[-1] > 145)
        
        # 조기 경보
        last_10min = seq_totalcnt[-10:]
        early_warning = (np.max(last_10min) >= 1640) and ((last_10min[-1] - last_10min[0]) > 20)
        
        # 결과 저장
        results.append({
            '시퀀스시작': seq_start_time.strftime('%Y-%m-%d %H:%M'),
            '현재시간': current_time.strftime('%Y-%m-%d %H:%M'),
            '현재TOTALCNT': round(seq_totalcnt[-1], 2),
            '예측시점': prediction_time.strftime('%Y-%m-%d %H:%M'),
            '실제시점': actual_time.strftime('%Y-%m-%d %H:%M'),
            '실제값': round(actual_value, 2),
            '예측값': round(prediction, 2),
            '오차': round(actual_value - prediction, 2),
            '절대오차': round(abs(actual_value - prediction), 2),
            '오차율(%)': round(abs(actual_value - prediction) / max(actual_value, 1) * 100, 2),
            'M14AM14B': round(seq_m14b[-1], 2),
            'M14AM14BSUM': round(seq_m14b_sum[-1], 2),
            'M14AM10A': round(seq_m10a[-1], 2),
            'M10AM14A': round(seq_m10a_rev[-1], 2),
            'queue_gap': round(seq_queue_gap[-1], 2),
            'TRANSPORT': round(seq_transport[-1], 2),
            'OHT_UTIL': round(seq_oht[-1], 2),
            '시퀀스MAX': round(np.max(seq_totalcnt), 2),
            '시퀀스MIN': round(np.min(seq_totalcnt), 2),
            '시퀀스평균': round(np.mean(seq_totalcnt), 2),
            '마지막10분MAX': round(np.max(last_10min), 2),
            '마지막10분상승': round(last_10min[-1] - last_10min[0], 2),
            '황금패턴(엄격)': 'O' if gold_strict else '',
            '황금패턴(보통)': 'O' if gold_normal else '',
            'Gap위험': 'O' if danger_gap else '',
            'Trans위험': 'O' if danger_trans else '',
            '4중체크': 'O' if quad_check else '',
            '조기경보': 'O' if early_warning else '',
            '실제위험(1700+)': 'O' if actual_value >= 1700 else '',
            '예측위험(1640+)': 'O' if prediction >= 1640 else ''
        })
        
        # 진행상황
        if (i - 280) % 5000 == 0 and i > 280:
            progress = (i - 280) / total_predictions * 100
            print(f"  진행중... {i-280:,}/{total_predictions:,} ({progress:.1f}%)")
            gc.collect()
    
    print(f"✅ 평가 완료!")
    
    # DataFrame 변환
    results_df = pd.DataFrame(results)
    
    # CSV 저장
    output_file = '/mnt/user-data/outputs/evaluation_13col_optimized.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✅ 결과 저장: {output_file}")
    
    # 메모리 정리
    del df
    gc.collect()
    
    # ===== 통계 분석 =====
    print("\n" + "="*80)
    print("📊 평가 통계 (13개 컬럼, 280분 → 10분 후)")
    print("="*80)
    print(f"총 예측 수: {len(results_df):,}개")
    print(f"평균 절대 오차(MAE): {results_df['절대오차'].mean():.2f}")
    print(f"평균 오차율: {results_df['오차율(%)'].mean():.2f}%")
    print(f"최대 절대 오차: {results_df['절대오차'].max():.2f}")
    print(f"최소 절대 오차: {results_df['절대오차'].min():.2f}")
    
    print(f"\n황금 패턴(엄격): {results_df['황금패턴(엄격)'].value_counts().get('O', 0):,}개")
    print(f"황금 패턴(보통): {results_df['황금패턴(보통)'].value_counts().get('O', 0):,}개")
    print(f"Gap 위험: {results_df['Gap위험'].value_counts().get('O', 0):,}개")
    print(f"Transport 위험: {results_df['Trans위험'].value_counts().get('O', 0):,}개")
    print(f"🔥 4중 체크: {results_df['4중체크'].value_counts().get('O', 0):,}개")
    print(f"🔥 조기 경보: {results_df['조기경보'].value_counts().get('O', 0):,}개")
    
    # 위험 구간 분석
    actual_danger = results_df['실제위험(1700+)'] == 'O'
    pred_danger = results_df['예측위험(1640+)'] == 'O'
    quad_check = results_df['4중체크'] == 'O'
    
    actual_danger_count = actual_danger.sum()
    pred_danger_count = pred_danger.sum()
    danger_detected = (actual_danger & pred_danger).sum()
    quad_count = quad_check.sum()
    
    # 4중 체크의 1700+ 예측력
    quad_to_danger = (quad_check & actual_danger).sum()
    
    print(f"\n실제 위험(1700+): {actual_danger_count:,}개")
    print(f"예측 위험(1640+): {pred_danger_count:,}개")
    print(f"위험 감지 성공: {danger_detected:,}개")
    if actual_danger_count > 0:
        print(f"🎯 위험 감지율: {danger_detected/actual_danger_count*100:.1f}% ({danger_detected:,}/{actual_danger_count:,})")
    
    if quad_count > 0:
        print(f"\n🔥 4중 체크 → 실제 1700+ 발생: {quad_to_danger:,}개 ({quad_to_danger/quad_count*100:.1f}%)")
    
    # 오차 상위 10개
    print("\n" + "="*80)
    print("오차 상위 10개 구간")
    print("="*80)
    top_errors = results_df.nlargest(10, '절대오차')
    print(top_errors[['현재시간', '현재TOTALCNT', '실제값', '예측값', '절대오차', 'M14AM14B', 'M14AM14BSUM', 'queue_gap', 'TRANSPORT', '4중체크']].to_string(index=False))
    
    # 최종 요약
    print("\n" + "="*80)
    print("✅ 최종 평가 요약")
    print("="*80)
    print(f"1. 전체 성능:")
    print(f"   - MAE: {results_df['절대오차'].mean():.2f}")
    if actual_danger_count > 0:
        print(f"   - 🎯 위험 감지율: {danger_detected/actual_danger_count*100:.1f}% ({danger_detected:,}/{actual_danger_count:,})")
    
    print(f"\n2. 저장 파일:")
    print(f"   - {output_file}")
    print("="*80)
    
    return results_df

if __name__ == '__main__':
    print("\n🚀 XGBoost 평가 시작 (13개 컬럼 최적화)...\n")
    results = evaluate_all_predictions()
    
    if results is not None:
        print(f"\n✅ 평가 완료! 총 {len(results):,}개 예측 생성")
        print(f"✅ 결과 파일: evaluation_13col_optimized.csv")