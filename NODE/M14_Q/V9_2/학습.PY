#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 V11.5 학습 - 280분 시퀀스 & 10분 예측
특징: 과거 280분(4시간 40분) 데이터를 보고 10분 뒤 변화량(Delta)을 예측
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import glob
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🚀 V11.5 학습 - 280분 시퀀스 / 10분 예측")
print("="*70)

PRED_OFFSET = 10 # 10분 뒤 예측

def load_data(path_pattern):
    print(f"\n📂 데이터 로딩: {path_pattern}")
    files = glob.glob(path_pattern)
    if not files: files = glob.glob('M14_Q_*.CSV')
    dfs = []
    for f in sorted(files):
        try: dfs.append(pd.read_csv(f, on_bad_lines='skip', encoding='utf-8'))
        except: dfs.append(pd.read_csv(f, on_bad_lines='skip', encoding='cp949'))
    return pd.concat(dfs, ignore_index=True)

def create_vectorized_features(df):
    """280분 시퀀스 피처 생성"""
    df = df.copy()
    
    # 숫자형 변환
    cols = ['TOTALCNT', 'M14AM14B', 'M14AM14BSUM', 'M10AM14A', 
            'M14.QUE.ALL.TRANSPORT4MINOVERCNT', 'M14.QUE.OHT.OHTUTIL',
            'M14.QUE.ALL.CURRENTQCREATED', 'M14.QUE.ALL.CURRENTQCOMPLETED']
    for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    df['Gap'] = df['M14.QUE.ALL.CURRENTQCREATED'] - df['M14.QUE.ALL.CURRENTQCOMPLETED']
    
    # 🔥 [수정] 280분까지 확장
    lags = [1, 3, 5, 10, 20, 50, 100, 280] 
    for lag in lags:
        df[f'Total_Lag_{lag}'] = df['TOTALCNT'].shift(lag)
        df[f'M14B_Lag_{lag}'] = df['M14AM14B'].shift(lag)
    
    # 🔥 [수정] 280분까지 확장
    windows = [10, 30, 60, 100, 280]
    for w in windows:
        df[f'Total_Mean_{w}'] = df['TOTALCNT'].rolling(w).mean()
        df[f'Total_Std_{w}'] = df['TOTALCNT'].rolling(w).std()

    # 변화량 및 기울기 (Delta 집중)
    df['Slope_1'] = df['TOTALCNT'] - df['TOTALCNT'].shift(1)
    df['Slope_3'] = df['TOTALCNT'] - df['TOTALCNT'].shift(3)
    df['Accel_1'] = df['Slope_1'] - df['Slope_1'].shift(1)
    df['Total_Delta_10'] = df['TOTALCNT'] - df['Total_Lag_10']

    # 타겟 생성 (10분 후 예측)
    df['Target_Total_Future'] = df['TOTALCNT'].shift(-PRED_OFFSET)
    df['Target_Delta'] = df['Target_Total_Future'] - df['TOTALCNT']
    
    return df.dropna().reset_index(drop=True)

# 1. 실행
df = load_data('M14_Q_*.CSV')
# (만약 1:1 파일 쓰실거면: df = pd.read_csv('Final_Train_Data_1to1.csv') 하시면 됩니다)

print("\n🔄 280분 시퀀스 피처 생성 중...")
df_features = create_vectorized_features(df)

# 2. 학습 준비
exclude = ['CURRTIME', 'Target_Total_Future', 'Target_Delta']
cols = [c for c in df_features.columns if c not in exclude]
X = df_features[cols]
y_delta = df_features['Target_Delta']
y_total = df_features['Target_Total_Future']

# 3. 가중치 (고부하 + 급상승)
weights = np.ones(len(y_total))
weights[y_total >= 1500] = 5
weights[y_total >= 1700] = 20
weights[y_delta >= 50] = 10 

print(f"\n🚀 모델 학습 시작 (280분 Sequence)...")
model = xgb.XGBRegressor(
    n_estimators=2000,
    learning_rate=0.015,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    n_jobs=4,
    device='cuda',
    tree_method='hist',
    random_state=42
)

model.fit(X, y_delta, sample_weight=weights, verbose=False)

# 저장
with open('model_v11_280seq.pkl', 'wb') as f: pickle.dump(model, f)
with open('features_v11_280seq.pkl', 'wb') as f: pickle.dump(cols, f)
print("✅ 학습 완료! (model_v11_280seq.pkl)")