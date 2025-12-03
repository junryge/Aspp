#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 V9 학습 - 25분 후 예측
Feature 간소화 (222개 → 60개)
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🚀 V9 학습 - 25분 후 예측")
print("="*70)

# ========================================
# Feature 생성 함수
# ========================================
def create_features_v9(row_dict):
    features = {}
    
    seq_m14b = np.array(row_dict['M14AM14B'])
    seq_m14bsum = np.array(row_dict['M14AM14BSUM'])
    seq_m10arev = np.array(row_dict['M10AM14A'])
    seq_totalcnt = np.array(row_dict['TOTALCNT'])
    seq_transport = np.array(row_dict['TRANSPORT'])
    seq_oht = np.array(row_dict['OHT'])
    
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
    
    features['m14b_x_sum'] = seq_m14b[-1] * seq_m14bsum[-1] / 1000
    features['m14b_x_sum_mean'] = np.mean(seq_m14b * seq_m14bsum) / 1000
    features['sum_per_m14b'] = seq_m14bsum[-1] / (seq_m14b[-1] + 1)
    features['m14b_plus_sum'] = seq_m14b[-1] + seq_m14bsum[-1]
    features['m10arev_x_m14b'] = seq_m10arev[-1] * seq_m14b[-1] / 100
    features['trans_x_m14b'] = seq_transport[-1] * seq_m14b[-1] / 100
    features['ratio_m14b_total'] = seq_m14b[-1] / (seq_totalcnt[-1] + 1)
    features['ratio_sum_total'] = seq_m14bsum[-1] / (seq_totalcnt[-1] + 1)
    
    features['m14b_over_520'] = np.sum(seq_m14b > 520)
    features['m14b_over_540'] = np.sum(seq_m14b > 540)
    features['m14bsum_over_600'] = np.sum(seq_m14bsum > 600)
    features['m14bsum_over_620'] = np.sum(seq_m14bsum > 620)
    features['m10arev_over_55'] = np.sum(seq_m10arev > 55)
    features['total_over_1600'] = np.sum(seq_totalcnt >= 1600)
    features['total_over_1650'] = np.sum(seq_totalcnt >= 1650)
    features['total_over_1700'] = np.sum(seq_totalcnt >= 1700)
    
    features['gold_strict'] = 1 if (seq_m14b[-1] > 540 and seq_m14bsum[-1] > 620) else 0
    features['gold_normal'] = 1 if (seq_m14b[-1] > 520 and seq_m14bsum[-1] > 600) else 0
    features['in_danger'] = 1 if seq_totalcnt[-1] >= 1700 else 0
    features['near_danger'] = 1 if seq_totalcnt[-1] >= 1600 else 0
    
    return features

# ========================================
# 데이터 준비
# ========================================
def prepare_data(csv_path, seq_len=280, pred_offset=25):
    print(f"\n📂 데이터 로딩: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✅ {len(df):,}행 × {df.shape[1]}열")
    
    required = ['M14AM14B', 'M14AM14BSUM', 'M10AM14A', 'TOTALCNT',
                'M14.QUE.ALL.TRANSPORT4MINOVERCNT', 'M14.QUE.OHT.OHTUTIL']
    
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"❌ 누락 컬럼: {missing}")
        return None, None
    
    print(f"✅ 핵심 6개 컬럼 확인")
    print(f"📊 예측 목표: {pred_offset}분 후")
    
    print(f"\n🔄 Feature 생성 중...")
    X_list, y_list = [], []
    total = len(df) - seq_len - pred_offset
    
    for i in range(seq_len, len(df) - pred_offset):
        row_dict = {
            'M14AM14B': df['M14AM14B'].iloc[i-seq_len:i].values,
            'M14AM14BSUM': df['M14AM14BSUM'].iloc[i-seq_len:i].values,
            'M10AM14A': df['M10AM14A'].iloc[i-seq_len:i].values,
            'TOTALCNT': df['TOTALCNT'].iloc[i-seq_len:i].values,
            'TRANSPORT': df['M14.QUE.ALL.TRANSPORT4MINOVERCNT'].iloc[i-seq_len:i].values,
            'OHT': df['M14.QUE.OHT.OHTUTIL'].iloc[i-seq_len:i].values,
        }
        
        X_list.append(create_features_v9(row_dict))
        y_list.append(df['TOTALCNT'].iloc[i + pred_offset - 1])
        
        if (i - seq_len) % 5000 == 0:
            print(f"  {i-seq_len:,}/{total:,}")
    
    print(f"✅ 완료!")
    
    X = pd.DataFrame(X_list)
    y = pd.Series(y_list)
    
    print(f"\n📊 결과:")
    print(f"  Feature: {X.shape[1]}개")
    print(f"  샘플: {len(X):,}개")
    print(f"  1700+: {(y>=1700).sum()}개 ({(y>=1700).sum()/len(y)*100:.2f}%)")
    
    return X, y

def augment_data(X, y, threshold=1700, multiplier=90):
    print(f"\n🔥 1700+ 데이터 {multiplier}배 증강")
    high_mask = y >= threshold
    count = high_mask.sum()
    print(f"1700+ 원본: {count:,}개")
    if count == 0:
        return X, y
    X_aug = pd.concat([X] + [X[high_mask]] * (multiplier - 1), ignore_index=True)
    y_aug = pd.concat([y] + [y[high_mask]] * (multiplier - 1), ignore_index=True)
    print(f"증강 후: {len(X_aug):,}개")
    return X_aug, y_aug

def train_model(X, y):
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

def evaluate(model, X, y):
    print(f"\n" + "="*70)
    print(f"📊 성능 평가")
    print("="*70)
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f"\n전체: MAE={mae:.2f}, R²={r2:.4f}")
    high = y >= 1700
    if high.sum() > 0:
        detected = ((y >= 1700) & (y_pred >= 1680)).sum()
        print(f"1700+ 감지율: {detected/high.sum()*100:.1f}% ({detected}/{high.sum()})")
    return mae, r2

if __name__ == '__main__':
    CSV_FILE = "data/M14_Q_20250802_TO_20251015.csv"
    
    X, y = prepare_data(CSV_FILE, seq_len=280, pred_offset=25)
    if X is None:
        exit(1)
    
    X_aug, y_aug = augment_data(X, y, threshold=1700, multiplier=90)
    model = train_model(X_aug, y_aug)
    mae, r2 = evaluate(model, X_aug, y_aug)
    
    with open('model_v9_25min.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('features_v9_25min.pkl', 'wb') as f:
        pickle.dump(list(X.columns), f)
    
    print(f"\n💾 저장: model_v9_25min.pkl")
    print(f"\n✅ V9 25분 학습 완료! MAE={mae:.2f}, R²={r2:.4f}")