#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 XGBoost 평가 - 가중치 모델 (완전판)
모든 컬럼 포함 + 100배 가중치 모델용
"""

import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
import os
import gc
import warnings
warnings.filterwarnings('ignore')

def create_features_v3(row_dict):
    """9개 컬럼 280분 시퀀스로부터 Feature 생성"""
    features = {}
    
    seq_m14b = np.array(row_dict['M14AM14B'])
    seq_m10a = np.array(row_dict['M14AM10A'])
    seq_m16 = np.array(row_dict['M14AM16'])
    seq_totalcnt = np.array(row_dict['TOTALCNT'])
    seq_queue_created = np.array(row_dict['M14.QUE.ALL.CURRENTQCREATED'])
    seq_queue_completed = np.array(row_dict['M14.QUE.ALL.CURRENTQCOMPLETED'])
    seq_oht_util = np.array(row_dict['M14.QUE.OHT.OHTUTIL'])
    seq_transport_over = np.array(row_dict['M14.QUE.ALL.TRANSPORT4MINOVERCNT'])
    seq_m14b_vertical = np.array(row_dict['M14B.QUE.SENDFAB.VERTICALQUEUECOUNT'])
    seq_queue_gap = seq_queue_created - seq_queue_completed
    
    # M14AM14B
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
    
    # M14AM10A
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
    
    # M14AM16
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
    
    # TOTALCNT
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
    
    # CURRENTQCREATED
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
    
    # CURRENTQCOMPLETED
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
    
    # OHT.OHTUTIL
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
    
    # TRANSPORT4MINOVERCNT
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
    
    # M14B.VERTICALQUEUECOUNT
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
    
    # queue_gap
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
    
    # 비율
    features['ratio_m14b_m10a'] = seq_m14b[-1] / (seq_m10a[-1] + 1)
    features['ratio_m14b_m16'] = seq_m14b[-1] / (seq_m16[-1] + 1)
    features['ratio_m10a_m16'] = seq_m10a[-1] / (seq_m16[-1] + 1)
    features['ratio_m14b_totalcnt'] = seq_m14b[-1] / (seq_totalcnt[-1] + 1)
    features['ratio_queue_gap_totalcnt'] = seq_queue_gap[-1] / (seq_totalcnt[-1] + 1)
    features['ratio_oht_util_m14b'] = seq_oht_util[-1] / (seq_m14b[-1] + 1)
    features['ratio_m14b_m10a_mean'] = np.mean(seq_m14b) / (np.mean(seq_m10a) + 1)
    features['ratio_queue_gap_m14b'] = seq_queue_gap[-1] / (seq_m14b[-1] + 1)
    features['ratio_vertical_m14b'] = seq_m14b_vertical[-1] / (seq_m14b[-1] + 1)
    features['ratio_transport_totalcnt'] = seq_transport_over[-1] / (seq_totalcnt[-1] + 1)
    features['volatility_m14b'] = np.std(seq_m14b) / (np.mean(seq_m14b) + 1)
    features['volatility_totalcnt'] = np.std(seq_totalcnt) / (np.mean(seq_totalcnt) + 1)
    features['volatility_queue_gap'] = np.std(seq_queue_gap) / (np.mean(np.abs(seq_queue_gap)) + 1)
    features['volatility_oht_util'] = np.std(seq_oht_util) / (np.mean(seq_oht_util) + 1)
    features['correlation_m14b_totalcnt'] = np.corrcoef(seq_m14b, seq_totalcnt)[0, 1]
    features['correlation_queue_gap_totalcnt'] = np.corrcoef(seq_queue_gap, seq_totalcnt)[0, 1]
    
    # 임계값
    features['m14b_over_250'] = np.sum(seq_m14b > 250)
    features['m14b_over_300'] = np.sum(seq_m14b > 300)
    features['m14b_over_350'] = np.sum(seq_m14b > 350)
    features['m14b_over_400'] = np.sum(seq_m14b > 400)
    features['m14b_over_450'] = np.sum(seq_m14b > 450)
    features['m14b_over_300_last30'] = np.sum(seq_m14b[-30:] > 300)
    features['m14b_over_400_last30'] = np.sum(seq_m14b[-30:] > 400)
    features['totalcnt_over_1500'] = np.sum(seq_totalcnt >= 1500)
    features['totalcnt_over_1600'] = np.sum(seq_totalcnt >= 1600)
    features['totalcnt_over_1700'] = np.sum(seq_totalcnt >= 1700)
    features['totalcnt_over_1600_last30'] = np.sum(seq_totalcnt[-30:] >= 1600)
    features['totalcnt_over_1700_last30'] = np.sum(seq_totalcnt[-30:] >= 1700)
    features['queue_gap_over_100'] = np.sum(seq_queue_gap > 100)
    features['queue_gap_over_150'] = np.sum(seq_queue_gap > 150)
    features['queue_gap_over_200'] = np.sum(seq_queue_gap > 200)
    features['queue_gap_over_100_last30'] = np.sum(seq_queue_gap[-30:] > 100)
    features['queue_gap_over_150_last30'] = np.sum(seq_queue_gap[-30:] > 150)
    features['oht_util_over_85'] = np.sum(seq_oht_util > 85)
    features['oht_util_over_90'] = np.sum(seq_oht_util > 90)
    features['oht_util_over_85_last30'] = np.sum(seq_oht_util[-30:] > 85)
    features['m10a_under_80'] = np.sum(seq_m10a < 80)
    features['m10a_under_70'] = np.sum(seq_m10a < 70)
    features['transport_over_150'] = np.sum(seq_transport_over > 150)
    features['transport_over_180'] = np.sum(seq_transport_over > 180)
    features['m14b_vertical_over_160'] = np.sum(seq_m14b_vertical > 160)
    features['m14b_vertical_over_180'] = np.sum(seq_m14b_vertical > 180)
    
    # 패턴
    features['golden_pattern_300_80'] = 1 if (seq_m14b[-1] > 300 and seq_m10a[-1] < 80) else 0
    features['golden_pattern_400_70'] = 1 if (seq_m14b[-1] > 400 and seq_m10a[-1] < 70) else 0
    features['golden_pattern_450_80'] = 1 if (seq_m14b[-1] > 450 and seq_m10a[-1] < 80) else 0
    features['danger_zone_1700'] = 1 if seq_totalcnt[-1] >= 1700 else 0
    features['danger_zone_1600'] = 1 if seq_totalcnt[-1] >= 1600 else 0
    features['danger_signal_1'] = 1 if (seq_queue_gap[-1] > 150 and seq_m14b[-1] > 400) else 0
    features['danger_signal_4'] = 1 if (seq_queue_gap[-1] > 100 and seq_oht_util[-1] > 85) else 0
    features['in_1700_zone'] = 1 if seq_totalcnt[-1] >= 1700 else 0
    features['rising_in_1700'] = 1 if (seq_totalcnt[-1] >= 1700 and seq_totalcnt[-1] - seq_totalcnt[-10] > 20) else 0
    features['stable_in_1700'] = 1 if (seq_totalcnt[-1] >= 1700 and abs(seq_totalcnt[-1] - seq_totalcnt[-10]) <= 20) else 0
    features['falling_in_1700'] = 1 if (seq_totalcnt[-1] >= 1700 and seq_totalcnt[-1] - seq_totalcnt[-10] < -20) else 0
    features['last_10min_trend'] = seq_totalcnt[-1] - seq_totalcnt[-10]
    features['last_30min_trend'] = seq_totalcnt[-1] - seq_totalcnt[-30]
    
    # 시간대별
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

def evaluate_all_predictions():
    """전체 데이터를 슬라이딩 윈도우로 평가 - 모든 컬럼 포함"""
    
    print("="*80)
    print("🔥 XGBoost 평가 - 가중치 100배 모델")
    print("="*80)
    
    model_file = 'xgboost_9col_weighted.pkl'
    
    if not os.path.exists(model_file):
        print(f"❌ 모델 파일 없음: {model_file}")
        return None
    
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    print(f"✅ 모델 로드")
    
    csv_file = 'Ois.csv'
    if not os.path.exists(csv_file):
        print(f"❌ CSV 파일 없음")
        return None
    
    df = pd.read_csv(csv_file, on_bad_lines='skip')
    print(f"✅ 데이터 로드: {len(df):,}행")
    
    df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M')
    
    results = []
    total_predictions = len(df) - 280 - 10
    
    print(f"🔄 평가 시작... (총 {total_predictions:,}개)")
    
    for i in range(280, len(df) - 10):
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
        }
        
        seq_totalcnt = row_dict['TOTALCNT']
        seq_m14b = row_dict['M14AM14B']
        seq_m10a = row_dict['M14AM10A']
        seq_m16 = row_dict['M14AM16']
        seq_oht_util = row_dict['M14.QUE.OHT.OHTUTIL']
        seq_queue_created = row_dict['M14.QUE.ALL.CURRENTQCREATED']
        seq_queue_completed = row_dict['M14.QUE.ALL.CURRENTQCOMPLETED']
        seq_queue_gap = seq_queue_created - seq_queue_completed
        
        current_time = df['CURRTIME'].iloc[i-1]
        seq_start_time = df['CURRTIME'].iloc[i-280]
        prediction_time = current_time + timedelta(minutes=10)
        actual_time = df['CURRTIME'].iloc[i+9]
        actual_value = df['TOTALCNT'].iloc[i+9]
        
        features = create_features_v3(row_dict)
        X_pred = pd.DataFrame([features])
        prediction = model.predict(X_pred)[0]
        
        golden_pattern = (seq_m14b[-1] > 300 and seq_m10a[-1] < 80)
        danger_in_seq = np.sum(seq_totalcnt >= 1700) > 0
        
        last_10min = seq_totalcnt[-10:]
        early_warning_detected = (np.max(last_10min) >= 1650) and ((last_10min[-1] - last_10min[0]) > 20)
        
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
        
        if (i - 280) % 5000 == 0 and i > 280:
            print(f"  {i-280:,}/{total_predictions:,} ({(i-280)/total_predictions*100:.1f}%)")
            gc.collect()
    
    print(f"✅ 평가 완료!")
    
    results_df = pd.DataFrame(results)
    
    output_file = 'evaluation_weighted_100x.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✅ 저장: {output_file}")
    
    del df
    gc.collect()
    
    print("\n" + "="*80)
    print("📊 통계")
    print("="*80)
    print(f"총 예측: {len(results_df):,}개")
    print(f"MAE: {results_df['절대오차'].mean():.2f}")
    
    actual_danger = results_df['실제위험(1700+)'] == 'O'
    pred_danger = results_df['예측위험(1650+)'] == 'O'
    pred_1700 = results_df['예측값'] >= 1700
    
    actual_count = actual_danger.sum()
    detected = (actual_danger & pred_danger).sum()
    detected_1700 = (actual_danger & pred_1700).sum()
    pred_1700_count = pred_1700.sum()
    
    print(f"\n🔥 1700+ 분석")
    print(f"실제 1700+: {actual_count:,}개")
    print(f"예측 1650+: {pred_danger.sum():,}개")
    print(f"예측 1700+: {pred_1700_count:,}개 🎯")
    
    if actual_count > 0:
        print(f"\n감지율 (1650+): {detected/actual_count*100:.1f}%")
        print(f"감지율 (1700+): {detected_1700/actual_count*100:.1f}% 🎯")
    
    print("="*80)
    
    return results_df

if __name__ == '__main__':
    results = evaluate_all_predictions()
    if results is not None:
        print(f"\n✅ 완료!")