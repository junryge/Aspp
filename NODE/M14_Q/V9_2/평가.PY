#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 V11.5 평가 - 280분 시퀀스 & 10분 예측 (수정됨)
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import warnings
from datetime import timedelta
warnings.filterwarnings('ignore')

print("="*70)
print("🚀 V11.5 평가 - 280분 시퀀스 / 10분 예측")
print("="*70)

PRED_OFFSET = 10
EVAL_FILE = '20251211_16평가.CSV'

def create_vectorized_features(df):
    """학습과 동일한 280분 시퀀스 피처 생성"""
    df = df.copy()
    
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

    # 변화량
    df['Slope_1'] = df['TOTALCNT'] - df['TOTALCNT'].shift(1)
    df['Slope_3'] = df['TOTALCNT'] - df['TOTALCNT'].shift(3)
    df['Accel_1'] = df['Slope_1'] - df['Slope_1'].shift(1)
    df['Total_Delta_10'] = df['TOTALCNT'] - df['Total_Lag_10']

    # 타겟
    df['Target_Total_Future'] = df['TOTALCNT'].shift(-PRED_OFFSET)
    return df

# 1. 모델 로드
print("📂 모델 로딩 (model_v11_280seq.pkl)...")
try:
    with open('model_v11_280seq.pkl', 'rb') as f:
        model_base = pickle.load(f)
    with open('features_v11_280seq.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
except FileNotFoundError:
    print("❌ 모델 파일이 없습니다.")
    exit()

# 2. 데이터 로드
print(f"📂 평가 데이터: {EVAL_FILE}")
try:
    df_raw = pd.read_csv(EVAL_FILE, on_bad_lines='skip', encoding='utf-8')
except:
    df_raw = pd.read_csv(EVAL_FILE, on_bad_lines='skip', encoding='cp949')

df_raw['CURRTIME_OBJ'] = pd.to_datetime(df_raw['CURRTIME'].astype(str), format='%Y%m%d%H%M', errors='coerce')

# 3. 피처 생성
print("🔄 280분 시퀀스 피처 생성 중...")
df_feat = create_vectorized_features(df_raw)
df_valid = df_feat.dropna(subset=feature_cols + ['Target_Total_Future']).copy()

print(f"✅ 유효 샘플: {len(df_valid):,}개")
X_eval = df_valid[feature_cols]
y_true = df_valid['Target_Total_Future']
current_total = df_valid['TOTALCNT']

# 4. 예측 (변화량)
pred_delta = model_base.predict(X_eval)
pred_total = current_total + pred_delta

# 5. 결과 저장
results = pd.DataFrame()
results['현재시간'] = df_valid['CURRTIME_OBJ'].dt.strftime('%Y-%m-%d %H:%M')
results['현재값'] = current_total.round(2)
results['예측시점'] = (df_valid['CURRTIME_OBJ'] + timedelta(minutes=PRED_OFFSET)).dt.strftime('%Y-%m-%d %H:%M')
results['실제값'] = y_true.round(2)
results['예측값'] = pred_total.round(2)
results['오차'] = (results['실제값'] - results['예측값']).round(2)
results['절대오차'] = results['오차'].abs()

# 통계
mae = results['절대오차'].mean()
print(f"\n📊 전체 MAE: {mae:.2f}")

high_val_mask = results['실제값'] >= 1650
if high_val_mask.sum() > 0:
    mae_high = results.loc[high_val_mask, '절대오차'].mean()
    print(f"🔥 고부하(>=1650) MAE: {mae_high:.2f}")

# CSV 저장
outfile = 'Evaluation_Result_V11_280seq.CSV'
results.to_csv(outfile, index=False, encoding='utf-8-sig')
print(f"\n💾 저장 완료: {outfile}")