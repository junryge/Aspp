#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 XGBoost 280분 시퀀스 → 10분 후 예측 학습
데이터: Ois.csv (43,200행)
11개 컬럼, 약 220개 Feature 생성
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
from datetime import datetime
import gc
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🚀 XGBoost 학습 - Ois.csv (280분→10분 예측)")
print("="*80)
print("11개 컬럼, 약 220개 Feature")
print("="*80)

def create_features_v2(row_dict):
    """11개 컬럼 280분 시퀀스로부터 Feature 생성"""
    features = {}
    
    # 시퀀스 추출
    seq_m14b = np.array(row_dict['M14AM14B'])
    seq_m10a = np.array(row_dict['M14AM10A'])
    seq_m16 = np.array(row_dict['M14AM16'])
    seq_totalcnt = np.array(row_dict['TOTALCNT'])
    
    # 신규 컬럼
    seq_queue_created = np.array(row_dict['M14.QUE.ALL.CURRENTQCREATED'])
    seq_queue_completed = np.array(row_dict['M14.QUE.ALL.CURRENTQCOMPLETED'])
    seq_oht_util = np.array(row_dict['M14.QUE.OHT.OHTUTIL'])
    seq_transport_over = np.array(row_dict['M14.QUE.ALL.TRANSPORT4MINOVERCNT'])
    seq_m14b_vertical = np.array(row_dict['M14B.QUE.SENDFAB.VERTICALQUEUECOUNT'])
    seq_m10a_cnt = np.array(row_dict['M10A.QUE.ALL.CURRENTQCNT'])
    seq_m10a_oht_util = np.array(row_dict['M10A.QUE.OHT_OHS.OHTUTIL'])
    
    # 파생 컬럼
    seq_queue_gap = seq_queue_created - seq_queue_completed
    
    # ========================================
    # 1. 기존 4개 컬럼 Feature (각 10개씩 = 40개)
    # ========================================
    
    # M14AM14B (10개)
    features['m14b_mean'] = np.mean(seq_m14b)
    features['m14b_std'] = np.std(seq_m14b)
    features['m14b_max'] = np.max(seq_m14b)
    features['m14b_min'] = np.min(seq_m14b)
    features['m14b_current'] = seq_m14b[-1]
    features['m14b_last_5_mean'] = np.mean(seq_m14b[-5:])
    features['m14b_last_10_mean'] = np.mean(seq_m14b[-10:])
    features['m14b_last_30_mean'] = np.mean(seq_m14b[-30:])
    features['m14b_slope'] = np.polyfit(np.arange(280), seq_m14b, 1)[0]
    features['m14b_range'] = np.max(seq_m14b) - np.min(seq_m14b)
    
    # M14AM10A (10개)
    features['m10a_mean'] = np.mean(seq_m10a)
    features['m10a_std'] = np.std(seq_m10a)
    features['m10a_max'] = np.max(seq_m10a)
    features['m10a_min'] = np.min(seq_m10a)
    features['m10a_current'] = seq_m10a[-1]
    features['m10a_last_5_mean'] = np.mean(seq_m10a[-5:])
    features['m10a_last_10_mean'] = np.mean(seq_m10a[-10:])
    features['m10a_last_30_mean'] = np.mean(seq_m10a[-30:])
    features['m10a_slope'] = np.polyfit(np.arange(280), seq_m10a, 1)[0]
    features['m10a_range'] = np.max(seq_m10a) - np.min(seq_m10a)
    
    # M14AM16 (10개)
    features['m16_mean'] = np.mean(seq_m16)
    features['m16_std'] = np.std(seq_m16)
    features['m16_max'] = np.max(seq_m16)
    features['m16_min'] = np.min(seq_m16)
    features['m16_current'] = seq_m16[-1]
    features['m16_last_5_mean'] = np.mean(seq_m16[-5:])
    features['m16_last_10_mean'] = np.mean(seq_m16[-10:])
    features['m16_last_30_mean'] = np.mean(seq_m16[-30:])
    features['m16_slope'] = np.polyfit(np.arange(280), seq_m16, 1)[0]
    features['m16_range'] = np.max(seq_m16) - np.min(seq_m16)
    
    # TOTALCNT (10개)
    features['totalcnt_mean'] = np.mean(seq_totalcnt)
    features['totalcnt_std'] = np.std(seq_totalcnt)
    features['totalcnt_max'] = np.max(seq_totalcnt)
    features['totalcnt_min'] = np.min(seq_totalcnt)
    features['totalcnt_current'] = seq_totalcnt[-1]
    features['totalcnt_last_5_mean'] = np.mean(seq_totalcnt[-5:])
    features['totalcnt_last_10_mean'] = np.mean(seq_totalcnt[-10:])
    features['totalcnt_last_30_mean'] = np.mean(seq_totalcnt[-30:])
    features['totalcnt_slope'] = np.polyfit(np.arange(280), seq_totalcnt, 1)[0]
    features['totalcnt_range'] = np.max(seq_totalcnt) - np.min(seq_totalcnt)
    
    # ========================================
    # 2. 신규 7개 컬럼 Feature (각 10개씩 = 70개)
    # ========================================
    
    # CURRENTQCREATED (10개)
    features['q_created_mean'] = np.mean(seq_queue_created)
    features['q_created_std'] = np.std(seq_queue_created)
    features['q_created_max'] = np.max(seq_queue_created)
    features['q_created_min'] = np.min(seq_queue_created)
    features['q_created_current'] = seq_queue_created[-1]
    features['q_created_last_5_mean'] = np.mean(seq_queue_created[-5:])
    features['q_created_last_10_mean'] = np.mean(seq_queue_created[-10:])
    features['q_created_last_30_mean'] = np.mean(seq_queue_created[-30:])
    features['q_created_slope'] = np.polyfit(np.arange(280), seq_queue_created, 1)[0]
    features['q_created_range'] = np.max(seq_queue_created) - np.min(seq_queue_created)
    
    # CURRENTQCOMPLETED (10개)
    features['q_completed_mean'] = np.mean(seq_queue_completed)
    features['q_completed_std'] = np.std(seq_queue_completed)
    features['q_completed_max'] = np.max(seq_queue_completed)
    features['q_completed_min'] = np.min(seq_queue_completed)
    features['q_completed_current'] = seq_queue_completed[-1]
    features['q_completed_last_5_mean'] = np.mean(seq_queue_completed[-5:])
    features['q_completed_last_10_mean'] = np.mean(seq_queue_completed[-10:])
    features['q_completed_last_30_mean'] = np.mean(seq_queue_completed[-30:])
    features['q_completed_slope'] = np.polyfit(np.arange(280), seq_queue_completed, 1)[0]
    features['q_completed_range'] = np.max(seq_queue_completed) - np.min(seq_queue_completed)
    
    # OHT.OHTUTIL (10개)
    features['oht_util_mean'] = np.mean(seq_oht_util)
    features['oht_util_std'] = np.std(seq_oht_util)
    features['oht_util_max'] = np.max(seq_oht_util)
    features['oht_util_min'] = np.min(seq_oht_util)
    features['oht_util_current'] = seq_oht_util[-1]
    features['oht_util_last_5_mean'] = np.mean(seq_oht_util[-5:])
    features['oht_util_last_10_mean'] = np.mean(seq_oht_util[-10:])
    features['oht_util_last_30_mean'] = np.mean(seq_oht_util[-30:])
    features['oht_util_slope'] = np.polyfit(np.arange(280), seq_oht_util, 1)[0]
    features['oht_util_range'] = np.max(seq_oht_util) - np.min(seq_oht_util)
    
    # TRANSPORT4MINOVERCNT (10개)
    features['transport_over_mean'] = np.mean(seq_transport_over)
    features['transport_over_std'] = np.std(seq_transport_over)
    features['transport_over_max'] = np.max(seq_transport_over)
    features['transport_over_min'] = np.min(seq_transport_over)
    features['transport_over_current'] = seq_transport_over[-1]
    features['transport_over_last_5_mean'] = np.mean(seq_transport_over[-5:])
    features['transport_over_last_10_mean'] = np.mean(seq_transport_over[-10:])
    features['transport_over_last_30_mean'] = np.mean(seq_transport_over[-30:])
    features['transport_over_slope'] = np.polyfit(np.arange(280), seq_transport_over, 1)[0]
    features['transport_over_range'] = np.max(seq_transport_over) - np.min(seq_transport_over)
    
    # M14B.VERTICALQUEUECOUNT (10개)
    features['m14b_vertical_mean'] = np.mean(seq_m14b_vertical)
    features['m14b_vertical_std'] = np.std(seq_m14b_vertical)
    features['m14b_vertical_max'] = np.max(seq_m14b_vertical)
    features['m14b_vertical_min'] = np.min(seq_m14b_vertical)
    features['m14b_vertical_current'] = seq_m14b_vertical[-1]
    features['m14b_vertical_last_5_mean'] = np.mean(seq_m14b_vertical[-5:])
    features['m14b_vertical_last_10_mean'] = np.mean(seq_m14b_vertical[-10:])
    features['m14b_vertical_last_30_mean'] = np.mean(seq_m14b_vertical[-30:])
    features['m14b_vertical_slope'] = np.polyfit(np.arange(280), seq_m14b_vertical, 1)[0]
    features['m14b_vertical_range'] = np.max(seq_m14b_vertical) - np.min(seq_m14b_vertical)
    
    # M10A.CURRENTQCNT (10개)
    features['m10a_cnt_mean'] = np.mean(seq_m10a_cnt)
    features['m10a_cnt_std'] = np.std(seq_m10a_cnt)
    features['m10a_cnt_max'] = np.max(seq_m10a_cnt)
    features['m10a_cnt_min'] = np.min(seq_m10a_cnt)
    features['m10a_cnt_current'] = seq_m10a_cnt[-1]
    features['m10a_cnt_last_5_mean'] = np.mean(seq_m10a_cnt[-5:])
    features['m10a_cnt_last_10_mean'] = np.mean(seq_m10a_cnt[-10:])
    features['m10a_cnt_last_30_mean'] = np.mean(seq_m10a_cnt[-30:])
    features['m10a_cnt_slope'] = np.polyfit(np.arange(280), seq_m10a_cnt, 1)[0]
    features['m10a_cnt_range'] = np.max(seq_m10a_cnt) - np.min(seq_m10a_cnt)
    
    # M10A.OHT_OHS.OHTUTIL (10개)
    features['m10a_oht_util_mean'] = np.mean(seq_m10a_oht_util)
    features['m10a_oht_util_std'] = np.std(seq_m10a_oht_util)
    features['m10a_oht_util_max'] = np.max(seq_m10a_oht_util)
    features['m10a_oht_util_min'] = np.min(seq_m10a_oht_util)
    features['m10a_oht_util_current'] = seq_m10a_oht_util[-1]
    features['m10a_oht_util_last_5_mean'] = np.mean(seq_m10a_oht_util[-5:])
    features['m10a_oht_util_last_10_mean'] = np.mean(seq_m10a_oht_util[-10:])
    features['m10a_oht_util_last_30_mean'] = np.mean(seq_m10a_oht_util[-30:])
    features['m10a_oht_util_slope'] = np.polyfit(np.arange(280), seq_m10a_oht_util, 1)[0]
    features['m10a_oht_util_range'] = np.max(seq_m10a_oht_util) - np.min(seq_m10a_oht_util)
    
    # ========================================
    # 3. 파생 Feature - queue_gap (10개)
    # ========================================
    features['queue_gap_mean'] = np.mean(seq_queue_gap)
    features['queue_gap_std'] = np.std(seq_queue_gap)
    features['queue_gap_max'] = np.max(seq_queue_gap)
    features['queue_gap_min'] = np.min(seq_queue_gap)
    features['queue_gap_current'] = seq_queue_gap[-1]
    features['queue_gap_last_5_mean'] = np.mean(seq_queue_gap[-5:])
    features['queue_gap_last_10_mean'] = np.mean(seq_queue_gap[-10:])
    features['queue_gap_last_30_mean'] = np.mean(seq_queue_gap[-30:])
    features['queue_gap_slope'] = np.polyfit(np.arange(280), seq_queue_gap, 1)[0]
    features['queue_gap_range'] = np.max(seq_queue_gap) - np.min(seq_queue_gap)
    
    # ========================================
    # 4. 비율 Feature (20개)
    # ========================================
    features['ratio_m14b_m10a'] = seq_m14b[-1] / (seq_m10a[-1] + 1)
    features['ratio_m14b_m16'] = seq_m14b[-1] / (seq_m16[-1] + 1)
    features['ratio_m10a_m16'] = seq_m10a[-1] / (seq_m16[-1] + 1)
    features['ratio_m14b_totalcnt'] = seq_m14b[-1] / (seq_totalcnt[-1] + 1)
    features['ratio_queue_gap_totalcnt'] = seq_queue_gap[-1] / (seq_totalcnt[-1] + 1)
    features['ratio_m10a_cnt_totalcnt'] = seq_m10a_cnt[-1] / (seq_totalcnt[-1] + 1)
    features['ratio_oht_util_m14b'] = seq_oht_util[-1] / (seq_m14b[-1] + 1)
    features['ratio_m10a_oht_m14b'] = seq_m10a_oht_util[-1] / (seq_m14b[-1] + 1)
    
    features['ratio_m14b_m10a_mean'] = np.mean(seq_m14b) / (np.mean(seq_m10a) + 1)
    features['ratio_queue_gap_m14b'] = seq_queue_gap[-1] / (seq_m14b[-1] + 1)
    features['ratio_vertical_m14b'] = seq_m14b_vertical[-1] / (seq_m14b[-1] + 1)
    features['ratio_transport_totalcnt'] = seq_transport_over[-1] / (seq_totalcnt[-1] + 1)
    
    features['volatility_m14b'] = np.std(seq_m14b) / (np.mean(seq_m14b) + 1)
    features['volatility_totalcnt'] = np.std(seq_totalcnt) / (np.mean(seq_totalcnt) + 1)
    features['volatility_queue_gap'] = np.std(seq_queue_gap) / (np.mean(np.abs(seq_queue_gap)) + 1)
    features['volatility_oht_util'] = np.std(seq_oht_util) / (np.mean(seq_oht_util) + 1)
    features['volatility_m10a_cnt'] = np.std(seq_m10a_cnt) / (np.mean(seq_m10a_cnt) + 1)
    
    features['correlation_m14b_totalcnt'] = np.corrcoef(seq_m14b, seq_totalcnt)[0, 1]
    features['correlation_queue_gap_totalcnt'] = np.corrcoef(seq_queue_gap, seq_totalcnt)[0, 1]
    features['correlation_m10a_cnt_totalcnt'] = np.corrcoef(seq_m10a_cnt, seq_totalcnt)[0, 1]
    
    # ========================================
    # 5. 임계값 카운트 Feature (30개)
    # ========================================
    
    # M14AM14B 임계값
    features['m14b_over_250'] = np.sum(seq_m14b > 250)
    features['m14b_over_300'] = np.sum(seq_m14b > 300)
    features['m14b_over_350'] = np.sum(seq_m14b > 350)
    features['m14b_over_400'] = np.sum(seq_m14b > 400)
    features['m14b_over_450'] = np.sum(seq_m14b > 450)
    features['m14b_over_300_last30'] = np.sum(seq_m14b[-30:] > 300)
    features['m14b_over_400_last30'] = np.sum(seq_m14b[-30:] > 400)
    
    # TOTALCNT 임계값
    features['totalcnt_over_1500'] = np.sum(seq_totalcnt >= 1500)
    features['totalcnt_over_1600'] = np.sum(seq_totalcnt >= 1600)
    features['totalcnt_over_1700'] = np.sum(seq_totalcnt >= 1700)
    features['totalcnt_over_1600_last30'] = np.sum(seq_totalcnt[-30:] >= 1600)
    features['totalcnt_over_1700_last30'] = np.sum(seq_totalcnt[-30:] >= 1700)
    
    # queue_gap 임계값
    features['queue_gap_over_100'] = np.sum(seq_queue_gap > 100)
    features['queue_gap_over_150'] = np.sum(seq_queue_gap > 150)
    features['queue_gap_over_200'] = np.sum(seq_queue_gap > 200)
    features['queue_gap_over_100_last30'] = np.sum(seq_queue_gap[-30:] > 100)
    features['queue_gap_over_150_last30'] = np.sum(seq_queue_gap[-30:] > 150)
    
    # OHT UTIL 임계값
    features['oht_util_over_85'] = np.sum(seq_oht_util > 85)
    features['oht_util_over_90'] = np.sum(seq_oht_util > 90)
    features['oht_util_over_85_last30'] = np.sum(seq_oht_util[-30:] > 85)
    
    # M10A CNT 임계값
    features['m10a_cnt_under_350'] = np.sum(seq_m10a_cnt < 350)
    features['m10a_cnt_under_320'] = np.sum(seq_m10a_cnt < 320)
    features['m10a_cnt_under_300'] = np.sum(seq_m10a_cnt < 300)
    features['m10a_cnt_under_350_last30'] = np.sum(seq_m10a_cnt[-30:] < 350)
    
    # M10A 임계값
    features['m10a_under_80'] = np.sum(seq_m10a < 80)
    features['m10a_under_70'] = np.sum(seq_m10a < 70)
    
    # TRANSPORT 임계값
    features['transport_over_150'] = np.sum(seq_transport_over > 150)
    features['transport_over_180'] = np.sum(seq_transport_over > 180)
    
    # M14B VERTICAL 임계값
    features['m14b_vertical_over_160'] = np.sum(seq_m14b_vertical > 160)
    features['m14b_vertical_over_180'] = np.sum(seq_m14b_vertical > 180)
    
    # ========================================
    # 6. 황금 패턴 & 위험 신호 (15개)
    # ========================================
    features['golden_pattern_300_80'] = 1 if (seq_m14b[-1] > 300 and seq_m10a[-1] < 80) else 0
    features['golden_pattern_400_70'] = 1 if (seq_m14b[-1] > 400 and seq_m10a[-1] < 70) else 0
    features['golden_pattern_450_80'] = 1 if (seq_m14b[-1] > 450 and seq_m10a[-1] < 80) else 0
    
    features['danger_zone_1700'] = 1 if seq_totalcnt[-1] >= 1700 else 0
    features['danger_zone_1600'] = 1 if seq_totalcnt[-1] >= 1600 else 0
    
    # 복합 위험 신호
    features['danger_signal_1'] = 1 if (seq_queue_gap[-1] > 150 and seq_m14b[-1] > 400) else 0
    features['danger_signal_2'] = 1 if (seq_oht_util[-1] > 85 and seq_m10a_cnt[-1] < 350) else 0
    features['danger_signal_3'] = 1 if (seq_m14b[-1] > 400 and seq_m10a_cnt[-1] < 350) else 0
    features['danger_signal_4'] = 1 if (seq_queue_gap[-1] > 100 and seq_oht_util[-1] > 85) else 0
    
    # 1700+ 구간 패턴
    features['in_1700_zone'] = 1 if seq_totalcnt[-1] >= 1700 else 0
    features['rising_in_1700'] = 1 if (seq_totalcnt[-1] >= 1700 and seq_totalcnt[-1] - seq_totalcnt[-10] > 20) else 0
    features['stable_in_1700'] = 1 if (seq_totalcnt[-1] >= 1700 and abs(seq_totalcnt[-1] - seq_totalcnt[-10]) <= 20) else 0
    features['falling_in_1700'] = 1 if (seq_totalcnt[-1] >= 1700 and seq_totalcnt[-1] - seq_totalcnt[-10] < -20) else 0
    
    # 최근 추세
    features['last_10min_trend'] = seq_totalcnt[-1] - seq_totalcnt[-10]
    features['last_30min_trend'] = seq_totalcnt[-1] - seq_totalcnt[-30]
    
    # ========================================
    # 7. 시간대별 통계 (10개)
    # ========================================
    q1 = seq_totalcnt[:70]
    q2 = seq_totalcnt[70:140]
    q3 = seq_totalcnt[140:210]
    q4 = seq_totalcnt[210:280]
    
    features['totalcnt_q1_mean'] = np.mean(q1)
    features['totalcnt_q2_mean'] = np.mean(q2)
    features['totalcnt_q3_mean'] = np.mean(q3)
    features['totalcnt_q4_mean'] = np.mean(q4)
    features['totalcnt_trend_q1_q2'] = np.mean(q2) - np.mean(q1)
    features['totalcnt_trend_q2_q3'] = np.mean(q3) - np.mean(q2)
    features['totalcnt_trend_q3_q4'] = np.mean(q4) - np.mean(q3)
    features['totalcnt_trend_overall'] = np.mean(q4) - np.mean(q1)
    features['totalcnt_q4_vs_mean'] = np.mean(q4) / (np.mean(seq_totalcnt) + 1)
    features['totalcnt_q4_acceleration'] = (np.mean(q4) - np.mean(q3)) - (np.mean(q3) - np.mean(q2))
    
    return features

def prepare_training_data(csv_path):
    """학습 데이터 준비"""
    print(f"\n📂 데이터 로딩: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✅ 원본: {len(df):,}행 × {df.shape[1]}열")
    
    # 필수 컬럼 확인
    required_cols = [
        'CURRTIME', 'M14AM14B', 'M14AM10A', 'M14AM16', 'TOTALCNT',
        'M14.QUE.ALL.CURRENTQCREATED', 'M14.QUE.ALL.CURRENTQCOMPLETED',
        'M14.QUE.OHT.OHTUTIL', 'M14.QUE.ALL.TRANSPORT4MINOVERCNT',
        'M14B.QUE.SENDFAB.VERTICALQUEUECOUNT', 
        'M10A.QUE.ALL.CURRENTQCNT', 'M10A.QUE.OHT_OHS.OHTUTIL'
    ]
    
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"❌ 필수 컬럼 누락: {missing}")
        return None, None
    
    print("✅ 필수 컬럼 확인 완료")
    
    # CURRTIME 변환
    df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M')
    
    print(f"\n🔄 Feature 생성 중... (280분 시퀀스 → 10분 후 타겟)")
    X_list = []
    y_list = []
    
    total_samples = len(df) - 280 - 10
    print(f"총 {total_samples:,}개 샘플 생성 예정")
    
    for i in range(280, len(df) - 10):
        # 280분 시퀀스: iloc[i-280:i] = 과거 280분
        row_dict = {
            'M14AM14B': df['M14AM14B'].iloc[i-280:i].values,
            'M14AM10A': df['M14AM10A'].iloc[i-280:i].values,
            'M14AM16': df['M14AM16'].iloc[i-280:i].values,
            'TOTALCNT': df['TOTALCNT'].iloc[i-280:i].values,
            'M14.QUE.ALL.CURRENTQCREATED': df['M14.QUE.ALL.CURRENTQCREATED'].iloc[i-280:i].values,
            'M14.QUE.ALL.CURRENTQCOMPLETED': df['M14.QUE.ALL.CURRENTQCOMPLETED'].iloc[i-280:i].values,
            'M14.QUE.OHT.OHTUTIL': df['M14.QUE.OHT.OHTUTIL'].iloc[i-280:i].values,
            'M14.QUE.ALL.TRANSPORT4MINOVERCNT': df['M14.QUE.ALL.TRANSPORT4MINOVERCNT'].iloc[i-280:i].values,
            'M14B.QUE.SENDFAB.VERTICALQUEUECOUNT': df['M14B.QUE.SENDFAB.VERTICALQUEUECOUNT'].iloc[i-280:i].values,
            'M10A.QUE.ALL.CURRENTQCNT': df['M10A.QUE.ALL.CURRENTQCNT'].iloc[i-280:i].values,
            'M10A.QUE.OHT_OHS.OHTUTIL': df['M10A.QUE.OHT_OHS.OHTUTIL'].iloc[i-280:i].values,
        }
        
        # Feature 생성
        features = create_features_v2(row_dict)
        
        # 타겟 (10분 후): iloc[i+9]
        # i-1이 현재, i+9는 10분 후
        target = df['TOTALCNT'].iloc[i+9]
        
        X_list.append(features)
        y_list.append(target)
        
        # 진행상황
        if (i - 280) % 1000 == 0:
            progress = (i - 280) / total_samples * 100
            print(f"  진행: {i-280:,}/{total_samples:,} ({progress:.1f}%)", end='\r')
    
    print(f"\n✅ Feature 생성 완료!")
    
    # DataFrame 변환
    X = pd.DataFrame(X_list)
    y = pd.Series(y_list)
    
    print(f"\n📊 Feature 정보:")
    print(f"  Feature 수: {X.shape[1]}개")
    print(f"  샘플 수: {len(X):,}개")
    print(f"  타겟 범위: {y.min():.0f} ~ {y.max():.0f}")
    print(f"  1700+ 타겟: {(y >= 1700).sum()}개 ({(y >= 1700).sum()/len(y)*100:.3f}%)")
    
    return X, y

def augment_high_risk_data(X, y, threshold=1700, multiplier=10):
    """1700+ 고위험 구간 데이터 증강"""
    print(f"\n🔥 데이터 증강 (1700+ 구간 x{multiplier})")
    
    high_risk_mask = y >= threshold
    high_risk_count = high_risk_mask.sum()
    
    print(f"1700+ 구간: {high_risk_count:,}개 ({high_risk_count/len(y)*100:.3f}%)")
    
    if high_risk_count == 0:
        print("⚠️ 1700+ 구간 없음, 증강 불가")
        return X, y
    
    X_high = X[high_risk_mask]
    y_high = y[high_risk_mask]
    
    # 증강
    X_augmented = pd.concat([X] + [X_high] * (multiplier - 1), ignore_index=True)
    y_augmented = pd.concat([y] + [y_high] * (multiplier - 1), ignore_index=True)
    
    print(f"증강 후: {len(X_augmented):,}개")
    print(f"1700+ 비율: {(y_augmented >= threshold).sum() / len(y_augmented) * 100:.2f}%")
    
    return X_augmented, y_augmented

def train_xgboost_v2(X_train, y_train):
    """XGBoost 학습"""
    print(f"\n🚀 XGBoost 학습 시작")
    
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 8,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist',
    }
    
    print(f"파라미터:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    
    model = xgb.XGBRegressor(**params)
    
    print(f"\n학습 중...")
    model.fit(
        X_train, y_train,
        verbose=50
    )
    
    print(f"\n✅ 학습 완료!")
    
    return model

def evaluate_training_data(model, X_train, y_train):
    """학습 데이터 평가"""
    print(f"\n" + "="*80)
    print(f"📊 학습 데이터 성능 평가")
    print("="*80)
    
    y_pred = model.predict(X_train)
    
    mae = mean_absolute_error(y_train, y_pred)
    rmse = np.sqrt(mean_squared_error(y_train, y_pred))
    r2 = r2_score(y_train, y_pred)
    
    print(f"\n전체 성능:")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²:   {r2:.4f}")
    
    # 1700+ 구간 평가
    high_mask = y_train >= 1700
    if high_mask.sum() > 0:
        mae_high = mean_absolute_error(y_train[high_mask], y_pred[high_mask])
        rmse_high = np.sqrt(mean_squared_error(y_train[high_mask], y_pred[high_mask]))
        
        print(f"\n1700+ 구간 성능:")
        print(f"  샘플 수: {high_mask.sum():,}개 ({high_mask.sum()/len(y_train)*100:.1f}%)")
        print(f"  MAE:  {mae_high:.2f}")
        print(f"  RMSE: {rmse_high:.2f}")
        
        # 1700+ 감지율
        pred_high_mask = y_pred >= 1650
        detected = (high_mask & pred_high_mask).sum()
        detection_rate = detected / high_mask.sum() * 100
        print(f"  1700+ 감지율: {detection_rate:.1f}% ({detected:,}/{high_mask.sum():,})")
        
        # 1700+ 실제 예측값 분포
        print(f"\n  1700+ 실제값 예측 분포:")
        print(f"    평균 예측: {y_pred[high_mask].mean():.1f}")
        print(f"    최소 예측: {y_pred[high_mask].min():.1f}")
        print(f"    최대 예측: {y_pred[high_mask].max():.1f}")
    
    # 1600+ 구간 평가
    mid_mask = (y_train >= 1600) & (y_train < 1700)
    if mid_mask.sum() > 0:
        mae_mid = mean_absolute_error(y_train[mid_mask], y_pred[mid_mask])
        print(f"\n1600-1699 구간 성능:")
        print(f"  샘플 수: {mid_mask.sum():,}개")
        print(f"  MAE:  {mae_mid:.2f}")
    
    print(f"\n" + "="*80)
    
    return mae, rmse, r2

def show_feature_importance(model, X):
    """Feature 중요도 출력"""
    print(f"\n" + "="*80)
    print(f"🏆 Feature 중요도 TOP 30")
    print("="*80)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(30).to_string(index=False))

# ========================================
# 메인 실행
# ========================================

if __name__ == '__main__':
    # 데이터 준비
    X, y = prepare_training_data('/mnt/user-data/uploads/Ois.csv')
    
    if X is None:
        print("❌ 데이터 준비 실패")
        exit(1)
    
    # 데이터 증강
    X_aug, y_aug = augment_high_risk_data(X, y, threshold=1700, multiplier=10)
    
    # 학습
    print(f"\n🎯 전체 데이터 학습")
    print(f"  총 샘플: {len(X_aug):,}개")
    print(f"  1700+ 비율: {(y_aug >= 1700).sum() / len(y_aug) * 100:.2f}%")
    
    model = train_xgboost_v2(X_aug, y_aug)
    
    # 학습 데이터 평가
    mae, rmse, r2 = evaluate_training_data(model, X_aug, y_aug)
    
    # Feature 중요도 출력
    show_feature_importance(model, X_aug)
    
    # 모델 저장
    model_filename = '/home/claude/xgboost_ois_280to10.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n💾 모델 저장: {model_filename}")
    
    # Feature 리스트 저장
    feature_filename = '/home/claude/xgboost_ois_features.pkl'
    with open(feature_filename, 'wb') as f:
        pickle.dump(list(X.columns), f)
    print(f"💾 Feature 저장: {feature_filename}")
    
    print("\n" + "="*80)
    print("✅ 학습 완료!")
    print("="*80)
    print(f"모델: {model_filename}")
    print(f"Feature: {feature_filename}")
    print(f"총 샘플: {len(X_aug):,}개")
    print(f"Feature 수: {X_aug.shape[1]}개")
    print(f"\n성능:")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²:   {r2:.4f}")
    print("="*80)