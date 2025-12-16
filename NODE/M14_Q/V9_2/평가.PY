#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 V11.0 평가 - 10분 예측 (시퀀스 280분)
시퀀스: 280분, 예측: 10분 후
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import warnings
from datetime import timedelta
warnings.filterwarnings('ignore')
print("="*70)
print("🚀 V11.0 평가 - 10분 예측 (시퀀스 280분)")
print("="*70)
PRED_OFFSET = 10
SEQ_LEN = 280

def create_vectorized_features(df):
    """학습과 동일한 피처 생성 로직 (280분 시퀀스)"""
    df = df.copy()
    
    cols = ['TOTALCNT', 'M14AM14B', 'M14AM14BSUM', 'M10AM14A', 
            'M14.QUE.ALL.TRANSPORT4MINOVERCNT', 'M14.QUE.OHT.OHTUTIL',
            'M14.QUE.ALL.CURRENTQCREATED', 'M14.QUE.ALL.CURRENTQCOMPLETED']
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    df['Gap'] = df['M14.QUE.ALL.CURRENTQCREATED'] - df['M14.QUE.ALL.CURRENTQCOMPLETED']
    df['Transport'] = df['M14.QUE.ALL.TRANSPORT4MINOVERCNT']
    df['OHT'] = df['M14.QUE.OHT.OHTUTIL']
    
    # 1. 시차 변수 (Lags) - 280분까지 확장
    lags = [1, 3, 5, 10, 20, 50, 100, 200, 280]
    for lag in lags:
        df[f'Total_Lag_{lag}'] = df['TOTALCNT'].shift(lag)
        df[f'M14B_Lag_{lag}'] = df['M14AM14B'].shift(lag)
        df[f'Sum_Lag_{lag}'] = df['M14AM14BSUM'].shift(lag)
        df[f'Gap_Lag_{lag}'] = df['Gap'].shift(lag)
    # 2. 이동 평균/표준편차 (Rolling Stats) - 280분까지 확장
    windows = [10, 30, 60, 100, 200, 280]
    for w in windows:
        df[f'Total_Mean_{w}'] = df['TOTALCNT'].rolling(w).mean()
        df[f'Total_Std_{w}'] = df['TOTALCNT'].rolling(w).std()
        df[f'Total_Max_{w}'] = df['TOTALCNT'].rolling(w).max()
        df[f'Total_Min_{w}'] = df['TOTALCNT'].rolling(w).min()
        df[f'M14B_Mean_{w}'] = df['M14AM14B'].rolling(w).mean()
        df[f'Gap_Mean_{w}'] = df['Gap'].rolling(w).mean()
    # 3. 변화량 (Delta)
    df['Total_Delta_1'] = df['TOTALCNT'] - df['Total_Lag_1']
    df['Total_Delta_5'] = df['TOTALCNT'] - df['Total_Lag_5']
    df['Total_Delta_10'] = df['TOTALCNT'] - df['Total_Lag_10']
    df['Total_Delta_50'] = df['TOTALCNT'] - df['Total_Lag_50']
    df['Total_Delta_100'] = df['TOTALCNT'] - df['Total_Lag_100']
    df['Total_Delta_280'] = df['TOTALCNT'] - df['Total_Lag_280']
    
    df['M14B_Delta_10'] = df['M14AM14B'] - df['M14B_Lag_10']
    df['M14B_Delta_100'] = df['M14AM14B'] - df['M14B_Lag_100']
    df['Sum_Delta_10'] = df['M14AM14BSUM'] - df['Sum_Lag_10']
    df['Sum_Delta_100'] = df['M14AM14BSUM'] - df['Sum_Lag_100']
    
    # 4. 상호작용 (Interaction)
    df['Load_Ratio'] = df['M14AM14B'] / (df['TOTALCNT'] + 1)
    df['Gap_Load'] = df['Gap'] * df['M14AM14B']
    
    # 5. 패턴 (Explicit Patterns)
    df['Gold_Pattern'] = ((df['M14AM14B'] > 500) & (df['M14AM14BSUM'] > 600)).astype(int)
    df['Danger_Gap'] = (df['Gap'] > 250).astype(int)
    df['Danger_Total'] = (df['TOTALCNT'] >= 1600).astype(int)
    df['Triple_Check'] = ((df['M14AM14B'] > 450) & (df['M14AM14BSUM'] > 550) & (df['Gap'] > 200)).astype(int)
    # 6. 가속도 (Acceleration) 및 단기 변화
    df['Accel_1'] = df['Total_Delta_1'] - df['Total_Delta_1'].shift(1)
    df['Accel_3'] = df['Total_Delta_1'].diff(3)
    
    # 기울기 (Slope)
    df['Slope_3'] = (df['TOTALCNT'] - df['TOTALCNT'].shift(3)) / 3
    df['Slope_5'] = (df['TOTALCNT'] - df['TOTALCNT'].shift(5)) / 5
    df['Slope_30'] = (df['TOTALCNT'] - df['TOTALCNT'].shift(30)) / 30
    df['Slope_100'] = (df['TOTALCNT'] - df['TOTALCNT'].shift(100)) / 100
    df['Gap_Slope_3'] = (df['Gap'] - df['Gap'].shift(3)) / 3
    df['Gap_Slope_30'] = (df['Gap'] - df['Gap'].shift(30)) / 30
    df['Target_Total_Future'] = df['TOTALCNT'].shift(-PRED_OFFSET)
    df['Target_Delta'] = df['Target_Total_Future'] - df['TOTALCNT']
    return df

# 1. 모델 로드
print("📂 모델 로딩...")
with open('model_v11_280min.pkl', 'rb') as f:
    model_base = pickle.load(f)
with open('features_v11_280min.pkl', 'rb') as f:
    feature_cols = pickle.load(f)
print(f"📊 피처 수: {len(feature_cols)}개")
# 2. 데이터 로드
EVAL_FILE = '20251211_16평가.CSV'
print(f"📂 데이터 로딩: {EVAL_FILE}")
try:
    df = pd.read_csv(EVAL_FILE, on_bad_lines='skip', encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(EVAL_FILE, on_bad_lines='skip', encoding='cp949')
# 시간 파싱 (CURRTIME: YYYYMMDDHHMM)
df['CURRTIME_OBJ'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M', errors='coerce')
# 3. 피처 생성
print(f"🔄 피처 생성 중 (280분 시퀀스)...")
df_feat = create_vectorized_features(df)
# 유효 데이터 필터링
df_valid = df_feat.dropna(subset=feature_cols).copy()
df_valid = df_valid.dropna(subset=['Target_Total_Future'])
print(f"✅ 유효 샘플: {len(df_valid):,}개")
X_eval = df_valid[feature_cols]
y_true = df_valid['Target_Total_Future']
current_time = df_valid['CURRTIME_OBJ']
current_total = df_valid['TOTALCNT']
m14b = df_valid['M14AM14B']
m14bsum = df_valid['M14AM14BSUM']
gap = df_valid['Gap']
trans = df_valid['Transport']
# 4. 예측
print("🚀 예측 수행...")
pred_delta = model_base.predict(X_eval)
pred_total = current_total + pred_delta
# 5. 결과 저장 (한글 컬럼)
results = pd.DataFrame()
results['현재시간'] = current_time.dt.strftime('%Y-%m-%d %H:%M')
results['현재TOTALCNT'] = current_total.round(2)
results['예측시점'] = (current_time + timedelta(minutes=PRED_OFFSET)).dt.strftime('%Y-%m-%d %H:%M')
results['실제값'] = y_true.round(2)
results['예측값'] = pred_total.round(2)
results['오차'] = (results['실제값'] - results['예측값']).round(2)
results['절대오차'] = (results['오차'].abs()).round(2)
results['M14AM14B'] = m14b
results['M14AM14BSUM'] = m14bsum
results['queue_gap'] = gap
results['TRANSPORT'] = trans
# 위험 감지 여부
results['실제위험'] = np.where(y_true >= 1700, 'O', '')
results['예측위험'] = np.where(pred_total >= 1700, 'O', '')
# 결과 출력
mae = results['절대오차'].mean()
print(f"\n📊 전체 평균 절대 오차 (MAE): {mae:.2f}")
high_val_mask = results['실제값'] >= 1650
if high_val_mask.sum() > 0:
    mae_high = results.loc[high_val_mask, '절대오차'].mean()
    acc_high = (results.loc[high_val_mask, '절대오차'] <= 20).mean() * 100
    print(f"🔥 고부하 구간(>=1650) MAE: {mae_high:.2f}")
    print(f"🔥 고부하 구간 정확도(±20): {acc_high:.1f}%")
# 1700+ 감지율
actual_danger = (y_true >= 1700).sum()
if actual_danger > 0:
    detected = ((y_true >= 1700) & (pred_total >= 1700)).sum()
    print(f"\n🔥 1700+ 감지:")
    print(f"  실제: {actual_danger}개")
    print(f"  감지: {detected}개 ({detected/actual_danger*100:.1f}%)")
# CSV 저장
outfile = '20251211_16평가_결과_280분.CSV'
results.to_csv(outfile, index=False, encoding='utf-8-sig')
print(f"\n💾 저장 완료: {outfile}")
print(f"📊 시퀀스: {SEQ_LEN}분")