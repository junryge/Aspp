# -*- coding: utf-8 -*-
"""
🚀 XGBoost V2 평가 - 신규 7개 컬럼 포함
280분 시퀀스 → 10분 후 예측 평가
"""

import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
import os
import gc
import warnings
warnings.filterwarnings('ignore')

def create_features_v2(row_dict):
    """
    11개 컬럼 280분 시퀀스로부터 Feature 생성
    
    기존 4개: M14AM14B, M14AM10A, M14AM16, TOTALCNT
    신규 7개: CURRENTQCREATED, CURRENTQCOMPLETED, OHTUTIL, 
             TRANSPORT4MINOVERCNT, VERTICALQUEUECOUNT, 
             M10A_CURRENTQCNT, M10A_OHTUTIL
    """
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
    features['volatility_queue_gap'] = np.std(seq_queue_gap) / (np.mean(seq_queue_gap) + 1)
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

def evaluate_all_predictions_v2():
    """전체 데이터를 슬라이딩 윈도우로 평가 - 신규 11개 컬럼 버전"""
    
    print("="*80)
    print("🚀 XGBoost V2 평가 - 신규 7개 컬럼 포함")
    print("="*80)
    print("11개 컬럼: 기존 4개 + 신규 7개")
    print("약 220개 Feature")
    print("="*80)
    
    # 모델 로드
    model_files = [
        'xgboost_280to10_v2_11columns.pkl',
        '/mnt/user-data/outputs/xgboost_280to10_v2_11columns.pkl'
    ]
    
    model = None
    model_file = None
    
    for mf in model_files:
        if os.path.exists(mf):
            try:
                with open(mf, 'rb') as f:
                    model = pickle.load(f)
                model_file = mf
                print(f"✅ 모델 로드 완료: {model_file}")
                break
            except Exception as e:
                print(f"⚠️ 모델 로드 실패 ({mf}): {e}")
    
    if model is None:
        print(f"❌ 모델 파일 없음!")
        print(f"시도한 파일:")
        for mf in model_files:
            print(f"  - {mf}")
        print("\n먼저 xgb_280to10_v2_학습.py를 실행하여 모델을 학습하세요.")
        return None
    
    # 데이터 로드
    csv_file = '/mnt/user-data/uploads/TRS.CSV'
    
    if not os.path.exists(csv_file):
        print(f"❌ CSV 파일 없음: {csv_file}")
        return None
    
    try:
        print(f"데이터 로딩 중: {csv_file}...")
        df = pd.read_csv(csv_file, on_bad_lines='skip')
        print(f"✅ 데이터 로드 완료: {len(df):,}개 행")
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return None
    
    # 메모리 사용량 확인
    memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"메모리 사용량: {memory_mb:.1f} MB")
    
    # 필수 컬럼 확인
    required_cols = [
        'CURRTIME', 'M14AM14B', 'M14AM10A', 'M14AM16', 'TOTALCNT',
        'M14.QUE.ALL.CURRENTQCREATED', 'M14.QUE.ALL.CURRENTQCOMPLETED',
        'M14.QUE.OHT.OHTUTIL', 'M14.QUE.ALL.TRANSPORT4MINOVERCNT',
        'M14B.QUE.SENDFAB.VERTICALQUEUECOUNT', 
        'M10A.QUE.ALL.CURRENTQCNT', 'M10A.QUE.OHT_OHS.OHTUTIL'
    ]
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ 필수 컬럼 누락: {missing_cols}")
        print(f"현재 컬럼: {list(df.columns)}")
        return None
    
    print(f"✅ 필수 컬럼 확인 완료")
    
    # CURRTIME 처리
    df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M')
    print("✅ CURRTIME 파싱 완료")
    
    results = []
    
    total_predictions = len(df) - 280 - 10
    print(f"\n🔄 슬라이딩 윈도우 평가 시작...")
    print(f"총 예측 수: {total_predictions:,}개")
    
    # 슬라이딩 윈도우
    for i in range(280, len(df) - 10):
        # 280분 시퀀스 (11개 컬럼)
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
        
        seq_totalcnt = row_dict['TOTALCNT']
        seq_m14b = row_dict['M14AM14B']
        seq_m10a = row_dict['M14AM10A']
        seq_m16 = row_dict['M14AM16']
        
        # 신규 컬럼
        seq_queue_created = row_dict['M14.QUE.ALL.CURRENTQCREATED']
        seq_queue_completed = row_dict['M14.QUE.ALL.CURRENTQCOMPLETED']
        seq_oht_util = row_dict['M14.QUE.OHT.OHTUTIL']
        seq_m10a_cnt = row_dict['M10A.QUE.ALL.CURRENTQCNT']
        
        seq_queue_gap = seq_queue_created - seq_queue_completed
        
        # 시간 정보
        current_time = df['CURRTIME'].iloc[i-1]
        seq_start_time = df['CURRTIME'].iloc[i-280]
        prediction_time = current_time + timedelta(minutes=10)
        actual_time = df['CURRTIME'].iloc[i+9]
        
        # 실제값 (10분 후)
        actual_value = df['TOTALCNT'].iloc[i+9]
        
        # Feature 생성
        features = create_features_v2(row_dict)
        X_pred = pd.DataFrame([features])
        
        # 예측 (수동 보정 없음!)
        prediction = model.predict(X_pred)[0]
        
        # 패턴 감지 (통계용)
        golden_pattern = (seq_m14b[-1] > 300 and seq_m10a[-1] < 80)
        danger_in_seq = np.sum(seq_totalcnt >= 1700) > 0
        
        # 조기 경보 감지 (통계용)
        last_10min = seq_totalcnt[-10:]
        early_warning_detected = (np.max(last_10min) >= 1650) and ((last_10min[-1] - last_10min[0]) > 20)
        
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
            'M14AM10A': round(seq_m10a[-1], 2),
            'M14AM16': round(seq_m16[-1], 2),
            'queue_gap': round(seq_queue_gap[-1], 2),
            'OHT_UTIL': round(seq_oht_util[-1], 2),
            'M10A_CNT': round(seq_m10a_cnt[-1], 2),
            '시퀀스TOTALCNT_MAX': round(np.max(seq_totalcnt), 2),
            '시퀀스TOTALCNT_MIN': round(np.min(seq_totalcnt), 2),
            '시퀀스TOTALCNT_평균': round(np.mean(seq_totalcnt), 2),
            '마지막10분_MAX': round(np.max(last_10min), 2),
            '마지막10분_상승폭': round(last_10min[-1] - last_10min[0], 2),
            '황금패턴': 'O' if golden_pattern else '',
            '시퀀스위험': 'O' if danger_in_seq else '',
            '조기경보': 'O' if early_warning_detected else '',
            '실제위험(1700+)': 'O' if actual_value >= 1700 else '',
            '예측위험(1650+)': 'O' if prediction >= 1650 else ''
        })
        
        # 진행상황 출력
        if (i - 280) % 10000 == 0 and i > 280:
            progress = (i - 280) / total_predictions * 100
            print(f"  진행중... {i-280:,}/{total_predictions:,} ({progress:.1f}%)")
            gc.collect()
    
    print(f"✅ 평가 완료!")
    
    # DataFrame 변환
    results_df = pd.DataFrame(results)
    
    # CSV 저장
    output_file = 'evaluation_v2_11columns.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✅ 결과 저장 완료: {output_file}")
    
    # 메모리 정리
    del df
    gc.collect()
    
    # ===== 통계 분석 =====
    print("\n" + "="*80)
    print("📊 평가 통계 (280분 → 10분 후)")
    print("="*80)
    print(f"총 예측 수: {len(results_df):,}개")
    print(f"평균 절대 오차(MAE): {results_df['절대오차'].mean():.2f}")
    print(f"평균 오차율: {results_df['오차율(%)'].mean():.2f}%")
    print(f"최대 절대 오차: {results_df['절대오차'].max():.2f}")
    print(f"최소 절대 오차: {results_df['절대오차'].min():.2f}")
    
    print(f"\n황금 패턴 발생: {results_df['황금패턴'].value_counts().get('O', 0):,}개")
    print(f"시퀀스 위험 구간: {results_df['시퀀스위험'].value_counts().get('O', 0):,}개")
    print(f"🔥 조기 경보 발생: {results_df['조기경보'].value_counts().get('O', 0):,}개")
    
    # 위험 구간 분석
    actual_danger = results_df['실제위험(1700+)'] == 'O'
    pred_danger = results_df['예측위험(1650+)'] == 'O'
    early_warning = results_df['조기경보'] == 'O'
    
    actual_danger_count = actual_danger.sum()
    pred_danger_count = pred_danger.sum()
    danger_detected = (actual_danger & pred_danger).sum()
    early_warning_count = early_warning.sum()
    
    # 조기 경보의 1700+ 예측력
    early_to_danger = (early_warning & actual_danger).sum()
    
    print(f"\n실제 위험(1700+): {actual_danger_count:,}개")
    print(f"예측 위험(1650+): {pred_danger_count:,}개")
    print(f"위험 감지 성공: {danger_detected:,}개")
    if actual_danger_count > 0:
        print(f"위험 감지율: {danger_detected/actual_danger_count*100:.1f}%")
    
    if early_warning_count > 0:
        print(f"\n🔥 조기 경보 → 실제 1700+ 발생: {early_to_danger:,}개 ({early_to_danger/early_warning_count*100:.1f}%)")
    
    # M14AM14B 450+ 구간 분석
    high_m14b_mask = results_df['M14AM14B'] >= 450
    high_m14b_count = high_m14b_mask.sum()
    if high_m14b_count > 0:
        print(f"\n🔥 M14AM14B 450+ 구간: {high_m14b_count:,}개")
        high_m14b_mae = results_df[high_m14b_mask]['절대오차'].mean()
        print(f"   평균 MAE: {high_m14b_mae:.2f}")
        high_m14b_danger = (high_m14b_mask & actual_danger).sum()
        print(f"   1700+ 발생: {high_m14b_danger:,}개 ({high_m14b_danger/high_m14b_count*100:.1f}%)")
    
    # queue_gap 구간 분석
    high_queue_gap_mask = results_df['queue_gap'] >= 150
    high_queue_gap_count = high_queue_gap_mask.sum()
    if high_queue_gap_count > 0:
        print(f"\n🔥 queue_gap 150+ 구간: {high_queue_gap_count:,}개")
        high_queue_gap_mae = results_df[high_queue_gap_mask]['절대오차'].mean()
        print(f"   평균 MAE: {high_queue_gap_mae:.2f}")
        high_queue_gap_danger = (high_queue_gap_mask & actual_danger).sum()
        print(f"   1700+ 발생: {high_queue_gap_danger:,}개 ({high_queue_gap_danger/high_queue_gap_count*100:.1f}%)")
    
    # 오차 상위 10개
    print("\n" + "="*80)
    print("오차 상위 10개 구간")
    print("="*80)
    top_errors = results_df.nlargest(10, '절대오차')
    print(top_errors[['현재시간', '현재TOTALCNT', '실제값', '예측값', '절대오차', 'M14AM14B', 'queue_gap', '조기경보']].to_string(index=False))
    
    # 10칸 내려서 비교 검증
    print("\n" + "="*80)
    print("🔍 10칸 내려서 예측 검증 (샘플 10개)")
    print("="*80)
    print("Line 1의 예측값 vs Line 11의 현재TOTALCNT 비교")
    print("-"*80)
    
    for idx in [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]:
        if idx + 10 < len(results_df):
            line1_time = results_df.iloc[idx]['현재시간']
            line1_pred = results_df.iloc[idx]['예측값']
            line11_time = results_df.iloc[idx+10]['현재시간']
            line11_current = results_df.iloc[idx+10]['현재TOTALCNT']
            diff = abs(line11_current - line1_pred)
            
            print(f"Line {idx+1:4d} (예측): {line1_time} → 예측값: {line1_pred:7.2f}")
            print(f"Line {idx+11:4d} (실제): {line11_time} → 현재값: {line11_current:7.2f}")
            print(f"           차이: {diff:.2f}")
            print("-"*40)
    
    # 최종 요약
    print("\n" + "="*80)
    print("✅ 최종 평가 요약")
    print("="*80)
    print(f"1. 전체 성능:")
    print(f"   - MAE: {results_df['절대오차'].mean():.2f}")
    print(f"   - 위험 감지율: {danger_detected/actual_danger_count*100:.1f}% ({danger_detected:,}/{actual_danger_count:,})")
    
    if high_m14b_count > 0:
        print(f"\n2. M14AM14B 450+ 구간:")
        print(f"   - 발생 횟수: {high_m14b_count:,}개")
        print(f"   - MAE: {high_m14b_mae:.2f}")
        print(f"   - 1700+ 비율: {high_m14b_danger/high_m14b_count*100:.1f}%")
    
    if high_queue_gap_count > 0:
        print(f"\n3. queue_gap 150+ 구간:")
        print(f"   - 발생 횟수: {high_queue_gap_count:,}개")
        print(f"   - MAE: {high_queue_gap_mae:.2f}")
        print(f"   - 1700+ 비율: {high_queue_gap_danger/high_queue_gap_count*100:.1f}%")
    
    if early_warning_count > 0:
        print(f"\n4. 조기 경보:")
        print(f"   - 발생 횟수: {early_warning_count:,}개")
        print(f"   - 정확도: {early_to_danger/early_warning_count*100:.1f}% (1700+ 예측)")
    
    print(f"\n5. 저장 파일:")
    print(f"   - {output_file}")
    print("="*80)
    
    return results_df

if __name__ == '__main__':
    print("\n🚀 XGBoost V2 평가 시작 (신규 11개 컬럼)...\n")
    results = evaluate_all_predictions_v2()
    
    if results is not None:
        print(f"\n✅ 평가 완료! 총 {len(results):,}개 예측 생성")
        print(f"✅ 결과 파일: evaluation_v2_11columns.csv")