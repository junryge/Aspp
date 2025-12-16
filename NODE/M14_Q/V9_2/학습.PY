#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 V12.0 평가 - 구간별 성능 분석
특징: Load_Category별로 MAE와 정확도를 상세 분석
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import warnings
from datetime import timedelta
warnings.filterwarnings('ignore')

print("="*70)
print("🚀 V12.0 평가 - 구간별 상세 분석")
print("="*70)

PRED_OFFSET = 10

# --- 구간 분류 함수 (평가용) ---
def get_load_category(cnt):
    if cnt <= 1200: return 'Under_1200'
    elif 1200 < cnt <= 1300: return '1200_1300'
    elif 1300 < cnt <= 1400: return '1300_1400'
    elif 1400 < cnt <= 1500: return '1400_1500'
    elif 1500 < cnt <= 1600: return '1500_1600'
    elif 1600 < cnt < 1700: return '1600_1700'
    else: return 'Over_1700'

def create_vectorized_features(df):
    """학습과 동일한 피처 생성"""
    df = df.copy()
    cols = ['TOTALCNT', 'M14AM14B', 'M14AM14BSUM', 'M10AM14A', 
            'M14.QUE.ALL.TRANSPORT4MINOVERCNT', 'M14.QUE.OHT.OHTUTIL',
            'M14.QUE.ALL.CURRENTQCREATED', 'M14.QUE.ALL.CURRENTQCOMPLETED']
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    df['Gap'] = df['M14.QUE.ALL.CURRENTQCREATED'] - df['M14.QUE.ALL.CURRENTQCOMPLETED']
    
    lags = [1, 3, 5, 10, 20, 50, 100, 200, 280]
    for lag in lags:
        df[f'Total_Lag_{lag}'] = df['TOTALCNT'].shift(lag)
        df[f'M14B_Lag_{lag}'] = df['M14AM14B'].shift(lag)
        df[f'Sum_Lag_{lag}'] = df['M14AM14BSUM'].shift(lag)
        df[f'Gap_Lag_{lag}'] = df['Gap'].shift(lag)

    windows = [10, 30, 60, 100, 200, 280]
    for w in windows:
        df[f'Total_Mean_{w}'] = df['TOTALCNT'].rolling(w).mean()
        df[f'Total_Std_{w}'] = df['TOTALCNT'].rolling(w).std()
        df[f'M14B_Mean_{w}'] = df['M14AM14B'].rolling(w).mean()
        df[f'Gap_Mean_{w}'] = df['Gap'].rolling(w).mean()
        
    df['Target_Total_Future'] = df['TOTALCNT'].shift(-PRED_OFFSET)
    return df

# 1. 모델 및 피처 로드
print("📂 모델 로딩...")
with open('model_v12_balanced.pkl', 'rb') as f:
    model = pickle.load(f)
with open('features_v12_balanced.pkl', 'rb') as f:
    feature_cols = pickle.load(f)

# 2. 평가 데이터 로드
EVAL_FILE = '20251211_16평가.CSV' # 평가할 파일명
print(f"📂 평가 데이터: {EVAL_FILE}")
try:
    df_raw = pd.read_csv(EVAL_FILE, on_bad_lines='skip', encoding='utf-8')
except:
    df_raw = pd.read_csv(EVAL_FILE, on_bad_lines='skip', encoding='cp949')

# 3. 피처 생성 및 전처리
print("🔄 피처 생성 중...")
df_feat = create_vectorized_features(df_raw)
df_valid = df_feat.dropna(subset=feature_cols + ['Target_Total_Future']).copy()

# 4. 예측 수행
print("🚀 예측 중...")
X_eval = df_valid[feature_cols]
pred_delta = model.predict(X_eval)
df_valid['Pred_Total'] = df_valid['TOTALCNT'] + pred_delta

# 5. Load_Category 라벨링 (실제값 기준)
df_valid['Load_Category'] = df_valid['Target_Total_Future'].apply(get_load_category)
df_valid['Error'] = (df_valid['Target_Total_Future'] - df_valid['Pred_Total']).abs()

# 6. 결과 집계 및 출력
print("\n📊 [구간별 예측 성능 분석]")
print("-" * 65)
print(f"{'Category':<15} | {'Count':<8} | {'MAE (오차)':<10} | {'RMSE':<10}")
print("-" * 65)

categories = ['Under_1200', '1200_1300', '1300_1400', '1400_1500', '1500_1600', '1600_1700', 'Over_1700']

total_mae = df_valid['Error'].mean()
print(f"{'TOTAL (전체)':<15} | {len(df_valid):<8} | {total_mae:<10.2f} | -")
print("-" * 65)

for cat in categories:
    subset = df_valid[df_valid['Load_Category'] == cat]
    if len(subset) > 0:
        mae = subset['Error'].mean()
        rmse = np.sqrt((subset['Error']**2).mean())
        print(f"{cat:<15} | {len(subset):<8} | {mae:<10.2f} | {rmse:<10.2f}")
    else:
        print(f"{cat:<15} | {'0':<8} | {'-':<10} | -")

print("-" * 65)

# 1700+ 구간 상세 정확도
over_1700 = df_valid[df_valid['Load_Category'] == 'Over_1700']
if len(over_1700) > 0:
    acc = (over_1700['Error'] <= 20).mean() * 100
    print(f"\n🔥 Over_1700 구간 상세:")
    print(f"   - 샘플 수: {len(over_1700)}개")
    print(f"   - 평균 오차: {over_1700['Error'].mean():.2f}")
    print(f"   - ±20 이내 정확도: {acc:.1f}%")

# 결과 저장
outfile = 'Evaluation_Result_Categorized.CSV'
df_valid.to_csv(outfile, index=False)
print(f"\n💾 상세 결과 저장 완료: {outfile}")
