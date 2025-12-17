#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 V11.5 평가 - 변화량(Delta) 예측 성능 검증
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import warnings
from datetime import timedelta
warnings.filterwarnings('ignore')

print("="*70)
print("🚀 V11.5 평가 - 변화량 예측 검증")
print("="*70)

PRED_OFFSET = 10
EVAL_FILE = '20251211_16평가.CSV' # 평가할 파일명 확인하세요

def create_vectorized_features(df):
    """학습과 동일한 피처 생성"""
    df = df.copy()
    cols = ['TOTALCNT', 'M14AM14B', 'M14AM14BSUM', 'M10AM14A', 
            'M14.QUE.ALL.TRANSPORT4MINOVERCNT', 'M14.QUE.OHT.OHTUTIL',
            'M14.QUE.ALL.CURRENTQCREATED', 'M14.QUE.ALL.CURRENTQCOMPLETED']
    for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    
    df['Gap'] = df['M14.QUE.ALL.CURRENTQCREATED'] - df['M14.QUE.ALL.CURRENTQCOMPLETED']
    
    lags = [1, 3, 5, 10, 20]
    for lag in lags:
        df[f'Total_Lag_{lag}'] = df['TOTALCNT'].shift(lag)
        df[f'M14B_Lag_{lag}'] = df['M14AM14B'].shift(lag)
    
    windows = [10, 30, 60]
    for w in windows:
        df[f'Total_Mean_{w}'] = df['TOTALCNT'].rolling(w).mean()
        df[f'Total_Std_{w}'] = df['TOTALCNT'].rolling(w).std()

    # 🔥 변화량 피처 (필수)
    df['Slope_1'] = df['TOTALCNT'] - df['TOTALCNT'].shift(1)
    df['Slope_3'] = df['TOTALCNT'] - df['TOTALCNT'].shift(3)
    df['Slope_5'] = df['TOTALCNT'] - df['TOTALCNT'].shift(5)
    df['Accel_1'] = df['Slope_1'] - df['Slope_1'].shift(1)

    df['Target_Total_Future'] = df['TOTALCNT'].shift(-PRED_OFFSET)
    return df

# 1. 모델 로드
print("📂 모델 로딩 (model_v11_delta_focused.pkl)...")
try:
    with open('model_v11_delta_focused.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('features_v11_delta_focused.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
except FileNotFoundError:
    print("❌ 모델 파일이 없습니다! 학습(train_v11_5.py)을 먼저 실행해주세요.")
    exit()

# 2. 데이터 로드
print(f"📂 평가 데이터: {EVAL_FILE}")
try:
    df_raw = pd.read_csv(EVAL_FILE, on_bad_lines='skip', encoding='utf-8')
except:
    df_raw = pd.read_csv(EVAL_FILE, on_bad_lines='skip', encoding='cp949')

df_raw['CURRTIME_OBJ'] = pd.to_datetime(df_raw['CURRTIME'].astype(str), format='%Y%m%d%H%M', errors='coerce')

# 3. 피처 생성
print("🔄 피처 생성 중...")
df_feat = create_vectorized_features(df_raw)

# 유효 데이터 필터링
df_valid = df_feat.dropna(subset=feature_cols + ['Target_Total_Future', 'CURRTIME_OBJ']).copy()
print(f"✅ 유효 평가 샘플: {len(df_valid):,}개")

# 4. 예측 수행
print("🚀 예측 수행 중...")
X_eval = df_valid[feature_cols]

# 변화량(Delta) 예측
pred_delta = model.predict(X_eval)

# 최종값 = 현재값 + 예측된 변화량
pred_total = df_valid['TOTALCNT'] + pred_delta

# 5. 결과 정리
results = pd.DataFrame()
current_time = df_valid['CURRTIME_OBJ']

results['현재시간'] = current_time.dt.strftime('%Y-%m-%d %H:%M')
results['현재TOTALCNT'] = df_valid['TOTALCNT'].round(2)
results['예측시점'] = (current_time + timedelta(minutes=PRED_OFFSET)).dt.strftime('%Y-%m-%d %H:%M')
results['실제값'] = df_valid['Target_Total_Future'].round(2)
results['예측값'] = pred_total.round(2)
results['예측Delta'] = pred_delta.round(2) # 모델이 예측한 변화량
results['실제Delta'] = (results['실제값'] - results['현재TOTALCNT']).round(2) # 실제 변화량
results['오차'] = (results['실제값'] - results['예측값']).round(2)
results['절대오차'] = results['오차'].abs()

# 위험 감지 여부 (1650 기준)
results['실제위험'] = np.where(results['실제값'] >= 1700, 'O', '')
results['예측위험'] = np.where(results['예측값'] >= 1650, 'O', '')

# 6. 통계 출력
mae = results['절대오차'].mean()
print(f"\n📊 전체 평균 절대 오차 (MAE): {mae:.2f}")

high_val_mask = results['실제값'] >= 1650
if high_val_mask.sum() > 0:
    mae_high = results.loc[high_val_mask, '절대오차'].mean()
    acc_high = (results.loc[high_val_mask, '절대오차'] <= 20).mean() * 100
    print(f"🔥 고부하 구간(>=1650) MAE: {mae_high:.2f}")
    print(f"🔥 고부하 구간 정확도(±20): {acc_high:.1f}%")
    
    # 변화량 예측력 체크
    print("\n[변화량(Delta) 예측 샘플]")
    print(results.loc[high_val_mask, ['현재시간', '현재TOTALCNT', '예측Delta', '실제Delta', '예측값', '실제값']].head().to_string(index=False))

# CSV 저장
outfile = 'Evaluation_Result_V11_5.CSV'
results.to_csv(outfile, index=False, encoding='utf-8-sig')
print(f"\n💾 상세 결과 저장 완료: {outfile}")
