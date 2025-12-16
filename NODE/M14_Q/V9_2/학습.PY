#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 V10.0 학습 - 10분 예측
시퀀스: 100분, 예측: 10분 후
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import glob
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🚀 V10.0 학습 - 10분 예측 (시퀀스 100분)")
print("="*70)

SEQ_LEN = 100
PRED_OFFSET = 10
MULTIPLIER = 90

def create_features(row_dict):
    features = {}
    seq_m14b = np.array(row_dict['M14AM14B'])
    seq_m14bsum = np.array(row_dict['M14AM14BSUM'])
    seq_m10a = np.array(row_dict['M10AM14A'])
    seq_total = np.array(row_dict['TOTALCNT'])
    seq_trans = np.array(row_dict['TRANSPORT'])
    seq_oht = np.array(row_dict['OHT'])
    seq_created = np.array(row_dict['Q_CREATED'])
    seq_completed = np.array(row_dict['Q_COMPLETED'])
    seq_gap = seq_created - seq_completed
    n = len(seq_m14b)
    
    # M14AM14B
    features['m14b_mean'] = np.mean(seq_m14b)
    features['m14b_max'] = np.max(seq_m14b)
    features['m14b_current'] = seq_m14b[-1]
    features['m14b_last10'] = np.mean(seq_m14b[-10:])
    features['m14b_last30'] = np.mean(seq_m14b[-30:])
    features['m14b_slope'] = np.polyfit(np.arange(n), seq_m14b, 1)[0]
    features['m14b_std'] = np.std(seq_m14b)
    features['m14b_trend'] = seq_m14b[-1] - seq_m14b[-10]
    
    # M14AM14BSUM
    features['m14bsum_mean'] = np.mean(seq_m14bsum)
    features['m14bsum_max'] = np.max(seq_m14bsum)
    features['m14bsum_current'] = seq_m14bsum[-1]
    features['m14bsum_last10'] = np.mean(seq_m14bsum[-10:])
    features['m14bsum_last30'] = np.mean(seq_m14bsum[-30:])
    features['m14bsum_slope'] = np.polyfit(np.arange(n), seq_m14bsum, 1)[0]
    features['m14bsum_std'] = np.std(seq_m14bsum)
    features['m14bsum_trend'] = seq_m14bsum[-1] - seq_m14bsum[-10]
    
    # TOTALCNT
    features['total_mean'] = np.mean(seq_total)
    features['total_max'] = np.max(seq_total)
    features['total_min'] = np.min(seq_total)
    features['total_current'] = seq_total[-1]
    features['total_last5'] = np.mean(seq_total[-5:])
    features['total_last10'] = np.mean(seq_total[-10:])
    features['total_last30'] = np.mean(seq_total[-30:])
    features['total_slope'] = np.polyfit(np.arange(n), seq_total, 1)[0]
    features['total_std'] = np.std(seq_total)
    features['total_trend'] = seq_total[-1] - seq_total[-10]
    
    # M10AM14A
    features['m10a_mean'] = np.mean(seq_m10a)
    features['m10a_max'] = np.max(seq_m10a)
    features['m10a_current'] = seq_m10a[-1]
    features['m10a_last10'] = np.mean(seq_m10a[-10:])
    features['m10a_slope'] = np.polyfit(np.arange(n), seq_m10a, 1)[0]
    
    # TRANSPORT
    features['trans_mean'] = np.mean(seq_trans)
    features['trans_max'] = np.max(seq_trans)
    features['trans_current'] = seq_trans[-1]
    features['trans_last10'] = np.mean(seq_trans[-10:])
    features['trans_slope'] = np.polyfit(np.arange(n), seq_trans, 1)[0]
    
    # OHT
    features['oht_mean'] = np.mean(seq_oht)
    features['oht_max'] = np.max(seq_oht)
    features['oht_current'] = seq_oht[-1]
    features['oht_last10'] = np.mean(seq_oht[-10:])
    
    # Queue Gap
    features['gap_mean'] = np.mean(seq_gap)
    features['gap_max'] = np.max(seq_gap)
    features['gap_min'] = np.min(seq_gap)
    features['gap_current'] = seq_gap[-1]
    features['gap_last5'] = np.mean(seq_gap[-5:])
    features['gap_last10'] = np.mean(seq_gap[-10:])
    features['gap_last30'] = np.mean(seq_gap[-30:])
    features['gap_slope'] = np.polyfit(np.arange(n), seq_gap, 1)[0]
    features['gap_std'] = np.std(seq_gap)
    features['gap_trend'] = seq_gap[-1] - seq_gap[-10]
    
    # Interaction
    features['m14b_x_sum'] = seq_m14b[-1] * seq_m14bsum[-1] / 1000
    features['sum_per_m14b'] = seq_m14bsum[-1] / (seq_m14b[-1] + 1)
    features['m14b_plus_sum'] = seq_m14b[-1] + seq_m14bsum[-1]
    features['gap_x_m14b'] = seq_gap[-1] * seq_m14b[-1] / 1000
    features['gap_x_total'] = seq_gap[-1] * seq_total[-1] / 10000
    features['ratio_m14b_total'] = seq_m14b[-1] / (seq_total[-1] + 1)
    features['ratio_sum_total'] = seq_m14bsum[-1] / (seq_total[-1] + 1)
    features['ratio_gap_total'] = seq_gap[-1] / (seq_total[-1] + 1)
    features['trans_x_gap'] = seq_trans[-1] * seq_gap[-1] / 100
    features['oht_x_total'] = seq_oht[-1] * seq_total[-1] / 10000
    
    # 임계값 카운트
    features['m14b_over_400'] = np.sum(seq_m14b > 400)
    features['m14b_over_500'] = np.sum(seq_m14b > 500)
    features['m14bsum_over_500'] = np.sum(seq_m14bsum > 500)
    features['m14bsum_over_600'] = np.sum(seq_m14bsum > 600)
    features['total_over_1500'] = np.sum(seq_total >= 1500)
    features['total_over_1600'] = np.sum(seq_total >= 1600)
    features['total_over_1700'] = np.sum(seq_total >= 1700)
    features['gap_over_100'] = np.sum(seq_gap > 100)
    features['gap_over_200'] = np.sum(seq_gap > 200)
    features['gap_over_300'] = np.sum(seq_gap > 300)
    features['trans_over_100'] = np.sum(seq_trans > 100)
    features['trans_over_150'] = np.sum(seq_trans > 150)
    
    # 패턴
    features['gold_pattern'] = 1 if (seq_m14b[-1] > 500 and seq_m14bsum[-1] > 600) else 0
    features['danger_gap'] = 1 if seq_gap[-1] > 250 else 0
    features['danger_total'] = 1 if seq_total[-1] >= 1600 else 0
    features['triple_check'] = 1 if (seq_m14b[-1] > 450 and seq_m14bsum[-1] > 550 and seq_gap[-1] > 200) else 0
    features['rising_fast'] = 1 if features['total_slope'] > 2 else 0
    features['gap_rising'] = 1 if features['gap_slope'] > 1 else 0
    
    return features

# 데이터 로드
print("\n📂 데이터 로딩...")
files = glob.glob('/mnt/user-data/uploads/SO*')
dfs = []
for f in sorted(files):
    try:
        df = pd.read_csv(f, on_bad_lines='skip')
        print(f"  {f.split('/')[-1]}: {len(df):,}행")
        dfs.append(df)
    except:
        print(f"  {f}: 실패")

df = pd.concat(dfs, ignore_index=True)
print(f"\n✅ 총 {len(df):,}행")

required = ['TOTALCNT', 'M14AM14B', 'M14AM14BSUM', 'M10AM14A',
            'M14.QUE.ALL.TRANSPORT4MINOVERCNT', 'M14.QUE.OHT.OHTUTIL',
            'M14.QUE.ALL.CURRENTQCREATED', 'M14.QUE.ALL.CURRENTQCOMPLETED']
df = df.dropna(subset=required)
print(f"📊 유효: {len(df):,}행, 1700+: {(df['TOTALCNT']>=1700).sum():,}개")

# Feature 생성
print(f"\n🔄 Feature 생성...")
X_list, y_list = [], []
total = len(df) - SEQ_LEN - PRED_OFFSET

for i in range(SEQ_LEN, len(df) - PRED_OFFSET):
    row_dict = {
        'M14AM14B': df['M14AM14B'].iloc[i-SEQ_LEN:i].values,
        'M14AM14BSUM': df['M14AM14BSUM'].iloc[i-SEQ_LEN:i].values,
        'M10AM14A': df['M10AM14A'].iloc[i-SEQ_LEN:i].values,
        'TOTALCNT': df['TOTALCNT'].iloc[i-SEQ_LEN:i].values,
        'TRANSPORT': df['M14.QUE.ALL.TRANSPORT4MINOVERCNT'].iloc[i-SEQ_LEN:i].values,
        'OHT': df['M14.QUE.OHT.OHTUTIL'].iloc[i-SEQ_LEN:i].values,
        'Q_CREATED': df['M14.QUE.ALL.CURRENTQCREATED'].iloc[i-SEQ_LEN:i].values,
        'Q_COMPLETED': df['M14.QUE.ALL.CURRENTQCOMPLETED'].iloc[i-SEQ_LEN:i].values,
    }
    target = df['TOTALCNT'].iloc[i + PRED_OFFSET - 1]
    if pd.isna(target):
        continue
    X_list.append(create_features(row_dict))
    y_list.append(target)
    if (i - SEQ_LEN) % 20000 == 0:
        print(f"  {i-SEQ_LEN:,}/{total:,}")

X = pd.DataFrame(X_list).fillna(0).replace([np.inf, -np.inf], 0)
y = pd.Series(y_list)
print(f"✅ 샘플: {len(X):,}, Feature: {X.shape[1]}개, 1700+: {(y>=1700).sum():,}개")

# 증강
print(f"\n🔥 1700+ {MULTIPLIER}배 증강...")
high_mask = y >= 1700
X_aug = pd.concat([X] + [X[high_mask]] * (MULTIPLIER - 1), ignore_index=True)
y_aug = pd.concat([y] + [y[high_mask]] * (MULTIPLIER - 1), ignore_index=True)
print(f"  증강 후: {len(X_aug):,}개")

# 학습
print(f"\n🚀 XGBoost 학습...")
model = xgb.XGBRegressor(
    max_depth=8, learning_rate=0.05, n_estimators=500,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, n_jobs=-1, tree_method='hist'
)
model.fit(X_aug, y_aug, verbose=100)

# 성능
y_pred = model.predict(X_aug)
print(f"\n📊 학습 성능:")
print(f"  MAE: {mean_absolute_error(y_aug, y_pred):.2f}")
print(f"  R²: {r2_score(y_aug, y_pred):.4f}")
detected = ((y_aug >= 1700) & (y_pred >= 1700)).sum()
print(f"  1700+ 감지: {detected}/{(y_aug>=1700).sum()} ({detected/(y_aug>=1700).sum()*100:.1f}%)")

# 중요도
print(f"\n🏆 Feature 중요도 TOP 15:")
imp = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
imp = imp.sort_values('importance', ascending=False)
for i, (_, r) in enumerate(imp.head(15).iterrows(), 1):
    print(f"  {i:2d}. {r['feature']}: {r['importance']:.4f}")

# 저장
with open('model_v10_10min.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('features_v10_10min.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)
print(f"\n💾 저장: model_v10_10min.pkl")
print(f"✅ V10.0 학습 완료!")