#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 V9.0 평가 - 수동 보정 로직 (Cheat Key) 적용
"""
import numpy as np
import pandas as pd
import pickle
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🚀 V9.0 평가 - 보정 로직 적용 (유리천장 깨기)")
print("="*70)

# 평가할 파일명 (여기만 수정하세요!)
EVAL_FILE = '20251211_16평가.CSV' 

# Feature 생성 함수 (학습과 동일)
def create_features_v9(row_dict):
    # ... (학습 코드의 create_features_v9 함수 내용을 여기에 복붙해야 합니다)
    # 편의상 생략했지만, 학습 코드의 함수를 그대로 가져오시면 됩니다.
    # 만약 귀찮으시면 학습 코드와 같은 파일에 두고 import 해도 됩니다.
    # 여기서는 안전하게 전체 코드를 다시 써드릴까요? (너무 길어져서.. 
    # 위 학습 코드의 create_features_v9 함수를 그대로 쓰시면 됩니다!)
    features = {}
    seq_m14b = np.array(row_dict['M14AM14B'])
    seq_m14bsum = np.array(row_dict['M14AM14BSUM'])
    seq_m10arev = np.array(row_dict['M10AM14A'])
    seq_totalcnt = np.array(row_dict['TOTALCNT'])
    seq_transport = np.array(row_dict['TRANSPORT'])
    seq_oht = np.array(row_dict['OHT'])
    seq_q_created = np.array(row_dict['Q_CREATED'])
    seq_q_completed = np.array(row_dict['Q_COMPLETED'])
    seq_gap = seq_q_created - seq_q_completed
    seq_len = len(seq_m14b)

    features['m14b_mean'] = np.mean(seq_m14b)
    features['m14b_max'] = np.max(seq_m14b)
    features['m14b_current'] = seq_m14b[-1]
    features['m14b_last10'] = np.mean(seq_m14b[-10:])
    features['m14b_last30'] = np.mean(seq_m14b[-30:])
    features['m14b_slope'] = np.polyfit(np.arange(seq_len), seq_m14b, 1)[0]
    features['m14b_std'] = np.std(seq_m14b)
    features['m14b_trend10'] = seq_m14b[-1] - seq_m14b[-10]

    features['m14bsum_mean'] = np.mean(seq_m14bsum)
    features['m14bsum_max'] = np.max(seq_m14bsum)
    features['m14bsum_current'] = seq_m14bsum[-1]
    features['m14bsum_last10'] = np.mean(seq_m14bsum[-10:])
    features['m14bsum_last30'] = np.mean(seq_m14bsum[-30:])
    features['m14bsum_slope'] = np.polyfit(np.arange(seq_len), seq_m14bsum, 1)[0]
    features['m14bsum_std'] = np.std(seq_m14bsum)
    features['m14bsum_trend10'] = seq_m14bsum[-1] - seq_m14bsum[-10]

    features['total_mean'] = np.mean(seq_totalcnt)
    features['total_max'] = np.max(seq_totalcnt)
    features['total_min'] = np.min(seq_totalcnt)
    features['total_current'] = seq_totalcnt[-1]
    features['total_last5'] = np.mean(seq_totalcnt[-5:])
    features['total_last10'] = np.mean(seq_totalcnt[-10:])
    features['total_last30'] = np.mean(seq_totalcnt[-30:])
    features['total_slope'] = np.polyfit(np.arange(seq_len), seq_totalcnt, 1)[0]
    features['total_std'] = np.std(seq_totalcnt)
    features['total_trend10'] = seq_totalcnt[-1] - seq_totalcnt[-10]

    features['m10arev_mean'] = np.mean(seq_m10arev)
    features['m10arev_max'] = np.max(seq_m10arev)
    features['m10arev_current'] = seq_m10arev[-1]
    features['m10arev_last10'] = np.mean(seq_m10arev[-10:])
    features['m10arev_slope'] = np.polyfit(np.arange(seq_len), seq_m10arev, 1)[0]

    features['trans_mean'] = np.mean(seq_transport)
    features['trans_max'] = np.max(seq_transport)
    features['trans_current'] = seq_transport[-1]
    features['trans_last10'] = np.mean(seq_transport[-10:])
    features['trans_slope'] = np.polyfit(np.arange(seq_len), seq_transport, 1)[0]

    features['oht_mean'] = np.mean(seq_oht)
    features['oht_max'] = np.max(seq_oht)
    features['oht_current'] = seq_oht[-1]
    features['oht_last10'] = np.mean(seq_oht[-10:])

    features['gap_mean'] = np.mean(seq_gap)
    features['gap_max'] = np.max(seq_gap)
    features['gap_min'] = np.min(seq_gap)
    features['gap_current'] = seq_gap[-1]
    features['gap_last5'] = np.mean(seq_gap[-5:])
    features['gap_last10'] = np.mean(seq_gap[-10:])
    features['gap_last30'] = np.mean(seq_gap[-30:])
    features['gap_slope'] = np.polyfit(np.arange(seq_len), seq_gap, 1)[0]
    features['gap_std'] = np.std(seq_gap)
    features['gap_trend10'] = seq_gap[-1] - seq_gap[-10]

    features['m14b_x_sum'] = seq_m14b[-1] * seq_m14bsum[-1] / 1000
    features['m14b_x_sum_mean'] = np.mean(seq_m14b * seq_m14bsum) / 1000
    features['sum_per_m14b'] = seq_m14bsum[-1] / (seq_m14b[-1] + 1)
    features['m14b_plus_sum'] = seq_m14b[-1] + seq_m14bsum[-1]
    features['m10arev_x_m14b'] = seq_m10arev[-1] * seq_m14b[-1] / 100
    features['trans_x_m14b'] = seq_transport[-1] * seq_m14b[-1] / 100
    features['ratio_m14b_total'] = seq_m14b[-1] / (seq_totalcnt[-1] + 1)
    features['ratio_sum_total'] = seq_m14bsum[-1] / (seq_totalcnt[-1] + 1)
    features['gap_x_m14b'] = seq_gap[-1] * seq_m14b[-1] / 1000
    features['gap_x_total'] = seq_gap[-1] * seq_totalcnt[-1] / 1000
    features['gap_x_trans'] = seq_gap[-1] * seq_transport[-1] / 100
    features['ratio_gap_total'] = seq_gap[-1] / (seq_totalcnt[-1] + 1)

    features['m14b_over_520'] = np.sum(seq_m14b > 520)
    features['m14b_over_540'] = np.sum(seq_m14b > 540)
    features['m14bsum_over_600'] = np.sum(seq_m14bsum > 600)
    features['m14bsum_over_620'] = np.sum(seq_m14bsum > 620)
    features['m10arev_over_55'] = np.sum(seq_m10arev > 55)
    features['total_over_1600'] = np.sum(seq_totalcnt >= 1600)
    features['total_over_1650'] = np.sum(seq_totalcnt >= 1650)
    features['total_over_1700'] = np.sum(seq_totalcnt >= 1700)
    features['gap_over_200'] = np.sum(seq_gap > 200)
    features['gap_over_250'] = np.sum(seq_gap > 250)
    features['gap_over_300'] = np.sum(seq_gap > 300)
    features['gap_over_350'] = np.sum(seq_gap > 350)

    features['gold_strict'] = 1 if (seq_m14b[-1] > 540 and seq_m14bsum[-1] > 620) else 0
    features['gold_normal'] = 1 if (seq_m14b[-1] > 520 and seq_m14bsum[-1] > 600) else 0
    features['in_danger'] = 1 if seq_totalcnt[-1] >= 1700 else 0
    features['near_danger'] = 1 if seq_totalcnt[-1] >= 1600 else 0
    features['danger_gap'] = 1 if seq_gap[-1] > 300 else 0
    features['danger_trans'] = 1 if seq_transport[-1] > 151 else 0
    features['triple_check'] = 1 if (seq_m14b[-1] > 520 and seq_m14bsum[-1] > 600 and seq_gap[-1] > 250) else 0
    features['quad_check'] = 1 if (seq_m14b[-1] > 520 and seq_m14bsum[-1] > 600 and seq_gap[-1] > 250 and seq_transport[-1] > 145) else 0

    return features

# 🔥 [핵심] 강제 보정 함수 (Glass Ceiling Break)
def adjust_prediction(pred, current_total, m14b, m14bsum, gap, trans):
    # 1. 하한선 안전장치
    floor = current_total - 80
    if pred < floor: pred = floor
    
    # 2. 부스트 로직 (조건 만족 시 강제 +점수)
    boost = 0
    # 황금 패턴
    if m14b > 540 and m14bsum > 620: boost += 50
    elif m14b > 520 and m14bsum > 600: boost += 40
    
    # Gap 부스트
    if gap > 350: boost += 40
    elif gap > 300: boost += 35
    elif gap > 250: boost += 25
    
    # Transport 부스트
    if trans > 180: boost += 30
    elif trans > 151: boost += 20
    
    pred = pred + boost
    
    # 3. 현재 이미 1700 넘었으면, 예측값도 못 떨어지게 막음
    if current_total >= 1700:
        if pred < 1680: pred = max(pred, current_total - 50)
        
    return pred

# 메인 실행
if __name__ == '__main__':
    with open('model_v9_10min.pkl', 'rb') as f: model = pickle.load(f)
    try: df = pd.read_csv(EVAL_FILE, encoding='utf-8')
    except: df = pd.read_csv(EVAL_FILE, encoding='cp949')
    
    df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M')
    
    results = []
    seq_len = 280
    pred_offset = 10
    
    print("🚀 평가 시작...")
    for i in range(seq_len, len(df) - pred_offset):
        row_dict = {
            'M14AM14B': df['M14AM14B'].iloc[i-seq_len:i].values,
            'M14AM14BSUM': df['M14AM14BSUM'].iloc[i-seq_len:i].values,
            'M10AM14A': df['M10AM14A'].iloc[i-seq_len:i].values,
            'TOTALCNT': df['TOTALCNT'].iloc[i-seq_len:i].values,
            'TRANSPORT': df['M14.QUE.ALL.TRANSPORT4MINOVERCNT'].iloc[i-seq_len:i].values,
            'OHT': df['M14.QUE.OHT.OHTUTIL'].iloc[i-seq_len:i].values,
            'Q_CREATED': df['M14.QUE.ALL.CURRENTQCREATED'].iloc[i-seq_len:i].values,
            'Q_COMPLETED': df['M14.QUE.ALL.CURRENTQCOMPLETED'].iloc[i-seq_len:i].values,
        }
        
        # 기본 예측
        feat = create_features_v9(row_dict)
        pred_raw = model.predict(pd.DataFrame([feat]))[0]
        
        # 🔥 보정 예측 적용
        current_total = df['TOTALCNT'].iloc[i-1]
        gap_val = feat['gap_current']
        trans_val = feat['trans_current']
        m14b_val = feat['m14b_current']
        m14bsum_val = feat['m14bsum_current']
        
        pred_adj = adjust_prediction(pred_raw, current_total, m14b_val, m14bsum_val, gap_val, trans_val)
        
        actual = df['TOTALCNT'].iloc[i+pred_offset-1]
        
        results.append({
            '현재시간': df['CURRTIME'].iloc[i-1],
            '실제값': actual,
            '예측값': pred_adj, # 보정된 값 사용
            '오차': actual - pred_adj,
            '예측위험': 'O' if pred_adj >= 1650 else '' # 1650 기준
        })
        
    res_df = pd.DataFrame(results)
    res_df.to_csv('Evaluation_Result_V9.CSV', index=False)
    print("✅ 완료! 저장됨: Evaluation_Result_V9.CSV")
