#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 V9.0 학습 - 10분 예측 (Queue Gap 복원 + 경로 수정판)
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import glob
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🚀 V9.0 학습 - 10분 예측 (경로 수정 완료)")
print("="*70)

# ========================================
# Feature 생성 함수 (그대로 유지)
# ========================================
def create_features_v9(row_dict):
    features = {}
    # numpy array 변환
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

    # 1. M14AM14B
    features['m14b_mean'] = np.mean(seq_m14b)
    features['m14b_max'] = np.max(seq_m14b)
    features['m14b_current'] = seq_m14b[-1]
    features['m14b_last10'] = np.mean(seq_m14b[-10:])
    features['m14b_last30'] = np.mean(seq_m14b[-30:])
    features['m14b_slope'] = np.polyfit(np.arange(seq_len), seq_m14b, 1)[0]
    features['m14b_std'] = np.std(seq_m14b)
    features['m14b_trend10'] = seq_m14b[-1] - seq_m14b[-10]

    # 2. M14AM14BSUM
    features['m14bsum_mean'] = np.mean(seq_m14bsum)
    features['m14bsum_max'] = np.max(seq_m14bsum)
    features['m14bsum_current'] = seq_m14bsum[-1]
    features['m14bsum_last10'] = np.mean(seq_m14bsum[-10:])
    features['m14bsum_last30'] = np.mean(seq_m14bsum[-30:])
    features['m14bsum_slope'] = np.polyfit(np.arange(seq_len), seq_m14bsum, 1)[0]
    features['m14bsum_std'] = np.std(seq_m14bsum)
    features['m14bsum_trend10'] = seq_m14bsum[-1] - seq_m14bsum[-10]

    # 3. TOTALCNT
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

    # 4. M10AM14A
    features['m10arev_mean'] = np.mean(seq_m10arev)
    features['m10arev_max'] = np.max(seq_m10arev)
    features['m10arev_current'] = seq_m10arev[-1]
    features['m10arev_last10'] = np.mean(seq_m10arev[-10:])
    features['m10arev_slope'] = np.polyfit(np.arange(seq_len), seq_m10arev, 1)[0]

    # 5. TRANSPORT
    features['trans_mean'] = np.mean(seq_transport)
    features['trans_max'] = np.max(seq_transport)
    features['trans_current'] = seq_transport[-1]
    features['trans_last10'] = np.mean(seq_transport[-10:])
    features['trans_slope'] = np.polyfit(np.arange(seq_len), seq_transport, 1)[0]

    # 6. OHT
    features['oht_mean'] = np.mean(seq_oht)
    features['oht_max'] = np.max(seq_oht)
    features['oht_current'] = seq_oht[-1]
    features['oht_last10'] = np.mean(seq_oht[-10:])

    # 7. 🔥 Queue Gap
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

    # Interaction
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

    # 임계값
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

    # 황금 패턴
    features['gold_strict'] = 1 if (seq_m14b[-1] > 540 and seq_m14bsum[-1] > 620) else 0
    features['gold_normal'] = 1 if (seq_m14b[-1] > 520 and seq_m14bsum[-1] > 600) else 0
    features['in_danger'] = 1 if seq_totalcnt[-1] >= 1700 else 0
    features['near_danger'] = 1 if seq_totalcnt[-1] >= 1600 else 0
    features['danger_gap'] = 1 if seq_gap[-1] > 300 else 0
    features['danger_trans'] = 1 if seq_transport[-1] > 151 else 0
    features['triple_check'] = 1 if (seq_m14b[-1] > 520 and seq_m14bsum[-1] > 600 and seq_gap[-1] > 250) else 0
    features['quad_check'] = 1 if (seq_m14b[-1] > 520 and seq_m14bsum[-1] > 600 and seq_gap[-1] > 250 and seq_transport[-1] > 145) else 0

    return features

# ========================================
# 데이터 로드 및 학습
# ========================================
def load_data_from_pattern(pattern='M14_Q_*.CSV'):
    print(f"\n📂 데이터 로딩: {pattern}")
    files = glob.glob(pattern)
    if not files: raise ValueError("파일 없음!")
    dfs = []
    for f in sorted(files):
        try: dfs.append(pd.read_csv(f, on_bad_lines='skip', encoding='utf-8'))
        except: dfs.append(pd.read_csv(f, on_bad_lines='skip', encoding='cp949'))
    return pd.concat(dfs, ignore_index=True)

def prepare_data(seq_len=280, pred_minutes=10):
    df = load_data_from_pattern()
    print(f"✅ {len(df):,}행")
    
    required = ['M14AM14B', 'M14AM14BSUM', 'M10AM14A', 'TOTALCNT', 
                'M14.QUE.ALL.TRANSPORT4MINOVERCNT', 'M14.QUE.OHT.OHTUTIL', 
                'M14.QUE.ALL.CURRENTQCREATED', 'M14.QUE.ALL.CURRENTQCOMPLETED']
    df = df.dropna(subset=required).reset_index(drop=True)
    
    print(f"\n🔄 Feature 생성 중 (다소 시간이 걸릴 수 있습니다)...")
    X_list, y_list = [], []
    pred_offset = pred_minutes
    total_len = len(df) - seq_len - pred_offset
    
    # 반복문 사용 (V9 방식)
    for i in range(seq_len, len(df) - pred_offset):
        # 10분 단위로만 샘플링하여 속도 향상 (선택사항, 지금은 전체 다 함)
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
        target = df['TOTALCNT'].iloc[i + pred_offset - 1]
        
        X_list.append(create_features_v9(row_dict))
        y_list.append(target)
        
        if len(X_list) % 10000 == 0:
            print(f"  진행률: {len(X_list)/total_len*100:.1f}%")

    X = pd.DataFrame(X_list)
    y = pd.Series(y_list)
    return X, y

def augment_data(X, y, threshold=1700, multiplier=20): # V9에 증폭 추가
    print(f"\n🔥 1700+ 데이터 {multiplier}배 증강")
    high_mask = y >= threshold
    if high_mask.sum() == 0: return X, y
    X_aug = pd.concat([X] + [X[high_mask]] * (multiplier - 1), ignore_index=True)
    y_aug = pd.concat([y] + [y[high_mask]] * (multiplier - 1), ignore_index=True)
    return X_aug, y_aug

if __name__ == '__main__':
    # 1. 데이터 준비
    X, y = prepare_data()
    
    # 2. 증폭 (V9에도 증폭을 섞으면 최강)
    X_aug, y_aug = augment_data(X, y, multiplier=20)
    
    # 3. 학습
    print(f"\n🚀 XGBoost 학습 시작...")
    model = xgb.XGBRegressor(
        objective='reg:squarederror', max_depth=8, learning_rate=0.05, 
        n_estimators=1000, subsample=0.8, colsample_bytree=0.8, 
        min_child_weight=3, n_jobs=-1, tree_method='hist', random_state=42
    )
    model.fit(X_aug, y_aug)
    
    # 4. 저장
    with open('model_v9_10min.pkl', 'wb') as f: pickle.dump(model, f)
    with open('features_v9_10min.pkl', 'wb') as f: pickle.dump(list(X.columns), f)
    print(f"\n✅ 저장 완료: model_v9_10min.pkl")
