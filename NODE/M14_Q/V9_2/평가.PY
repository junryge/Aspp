#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 V11.5 평가 - 변화량 예측 검증 (수정됨)
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import warnings
from datetime import timedelta
warnings.filterwarnings('ignore')

print("="*70)
print("🚀 V11.5 평가 - 10분 예측 (Delta 기반)")
print("="*70)

PRED_OFFSET = 10
EVAL_FILE = '20251211_16평가.CSV' # 평가할 파일명 확인

def create_vectorized_features(df):
    """학습과 동일한 피처 생성 (중요: Slope, Accel 포함)"""
    df = df.copy()
    
    cols = ['TOTALCNT', 'M14AM14B', 'M14AM14BSUM', 'M10AM14A', 
            'M14.QUE.ALL.TRANSPORT4MINOVERCNT', 'M14.QUE.OHT.OHTUTIL',
            'M14.QUE.ALL.CURRENTQCREATED', 'M14.QUE.ALL.CURRENTQCOMPLETED']
    for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    
    df['Gap'] = df['M14.QUE.ALL.CURRENTQCREATED'] - df['M14.QUE.ALL.CURRENTQCOMPLETED']
    
    # 1. 시차 변수
    lags = [1, 3, 5, 10, 20]
    for lag in lags:
        df[f'Total_Lag_{lag}'] = df['TOTALCNT'].shift(lag)
        df[f'M14B_Lag_{lag}'] = df['M14AM14B'].shift(lag)
    
    # 2. 이동 평균
    windows = [10, 30, 60]
    for w in windows:
        df[f'Total_Mean_{w}'] = df['TOTALCNT'].rolling(w).mean()
        df[f'Total_Std_{w}'] = df['TOTALCNT'].rolling(w).std()

    # 3. 🔥 변화량 및 기울기 (학습 코드와 동일하게!)
    df['Slope_1'] = df['TOTALCNT'] - df['TOTALCNT'].shift(1)
    df['Slope_3'] = df['TOTALCNT'] - df['TOTALCNT'].shift(3)
    df['Accel_1'] = df['Slope_1'] - df['Slope_1'].shift(1)
    df['Total_Delta_10'] = df['TOTALCNT'] - df['Total_Lag_10']

    # 타겟 생성 (검증용)
    df['Target_Total_Future'] = df['TOTALCNT'].shift(-PRED_OFFSET)
    
    return df

# 1. 모델 로드
print("📂 모델 로딩 (model_v11_delta.pkl)...")
try:
    with open('model_v11_delta.pkl', 'rb') as f:
        model_base = pickle.load(f)
    with open('features_v11_delta.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
except FileNotFoundError:
    print("❌ 모델 파일이 없습니다. 학습 코드를 먼저 실행하세요.")
    exit()

# 2. 데이터 로드
print(f"📂 데이터 로딩: {EVAL_FILE}")
try:
    df = pd.read_csv(EVAL_FILE, on_bad_lines='skip', encoding='utf-8')
except:
    df = pd.read_csv(EVAL_FILE, on_bad_lines='skip', encoding='cp949')

df['CURRTIME_OBJ'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M', errors='coerce')

# 3. 피처 생성
print("🔄 피처 생성 중...")
df_feat = create_vectorized_features(df)

# 유효 데이터 필터링 (Feature가 NaN인 초기 데이터 제거)
df_valid = df_feat.dropna(subset=feature_cols).copy()
# 타겟값(실제값)이 있는 구간만 평가 (마지막 10분은 정답이 없으므로 제외)
df_valid = df_valid.dropna(subset=['Target_Total_Future'])

print(f"✅ 유효 샘플: {len(df_valid):,}개")
X_eval = df_valid[feature_cols]
y_true = df_valid['Target_Total_Future']
current_total = df_valid['TOTALCNT']

# 4. 예측
print("🚀 예측 수행...")
# 모델은 변화량(Delta)을 예측함
pred_delta = model_base.predict(X_eval)

# 최종 예측값 = 현재값 + 예측된 변화량
pred_total = current_total + pred_delta

# 5. 결과 저장
results = pd.DataFrame()
results['현재시간'] = df_valid['CURRTIME_OBJ'].dt.strftime('%Y-%m-%d %H:%M')
results['현재값'] = current_total.round(2)
results['예측시점'] = (df_valid['CURRTIME_OBJ'] + timedelta(minutes=PRED_OFFSET)).dt.strftime('%Y-%m-%d %H:%M')
results['실제값'] = y_true.round(2)
results['예측값'] = pred_total.round(2)
results['예측Delta'] = pred_delta.round(2) # 모델이 예측한 변화량 확인용
results['오차'] = (results['실제값'] - results['예측값']).round(2)
results['절대오차'] = results['오차'].abs()

# 위험 감지 여부
results['실제위험'] = np.where(results['실제값'] >= 1700, 'O', '')
results['예측위험'] = np.where(results['예측값'] >= 1650, 'O', '') # 1650으로 알람 기준 잡음

# 통계 출력
mae = results['절대오차'].mean()
print(f"\n📊 전체 평균 절대 오차 (MAE): {mae:.2f}")

high_val_mask = results['실제값'] >= 1650
if high_val_mask.sum() > 0:
    mae_high = results.loc[high_val_mask, '절대오차'].mean()
    acc_high = (results.loc[high_val_mask, '절대오차'] <= 20).mean() * 100
    print(f"🔥 고부하 구간(>=1650) MAE: {mae_high:.2f}")
    print(f"🔥 고부하 구간 정확도(±20): {acc_high:.1f}%")

# CSV 저장
outfile = 'Evaluation_Result_V11_delta.CSV'
results.to_csv(outfile, index=False, encoding='utf-8-sig')
print(f"\n💾 저장 완료: {outfile}")