#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 V12.2 평가 - 결과 상세 저장 및 구간 분석
특징: Load_Category별 오차 분석 + 엑셀용 상세 결과 파일 생성
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import warnings
from datetime import timedelta
warnings.filterwarnings('ignore')

print("="*70)
print("🚀 V12.2 평가 - 결과 상세 저장 및 구간 분석")
print("="*70)

PRED_OFFSET = 10
EVAL_FILE = '20251211_16평가.CSV'  # 🔴 평가할 파일명을 여기에 입력하세요

# --- 구간 분류 함수 (평가 통계용) ---
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

    df['Slope_1'] = df['TOTALCNT'] - df['TOTALCNT'].shift(1)
    df['Slope_3'] = df['TOTALCNT'] - df['TOTALCNT'].shift(3)
    df['Slope_10'] = df['TOTALCNT'] - df['TOTALCNT'].shift(10)
        
    df['Target_Total_Future'] = df['TOTALCNT'].shift(-PRED_OFFSET)
    return df

# 1. 모델 및 피처 로드
print("📂 모델 로딩 (model_v12_weighted.pkl)...")
try:
    with open('model_v12_weighted.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('features_v12_weighted.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
except FileNotFoundError:
    print("❌ 모델 파일이 없습니다! 학습(train_v12_2.py)을 먼저 실행해주세요.")
    exit()

# 2. 평가 데이터 로드
print(f"📂 평가 데이터 로딩: {EVAL_FILE}")
try:
    df_raw = pd.read_csv(EVAL_FILE, on_bad_lines='skip', encoding='utf-8')
except:
    df_raw = pd.read_csv(EVAL_FILE, on_bad_lines='skip', encoding='cp949')

# 시간 파싱
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
pred_delta = model.predict(X_eval)
pred_total = df_valid['TOTALCNT'] + pred_delta

# -------------------------------------------------------------
# 5. 상세 결과 저장 (Results DataFrame)
# -------------------------------------------------------------
results = pd.DataFrame()
current_time = df_valid['CURRTIME_OBJ']

results['현재시간'] = current_time.dt.strftime('%Y-%m-%d %H:%M')
results['현재TOTALCNT'] = df_valid['TOTALCNT'].round(2)
results['예측시점'] = (current_time + timedelta(minutes=PRED_OFFSET)).dt.strftime('%Y-%m-%d %H:%M')
results['실제값'] = df_valid['Target_Total_Future'].round(2)
results['예측값'] = pred_total.round(2)
results['오차'] = (results['실제값'] - results['예측값']).round(2)
results['절대오차'] = (results['오차'].abs()).round(2)

# 보조 지표
results['M14AM14B'] = df_valid['M14AM14B']
results['M14AM14BSUM'] = df_valid['M14AM14BSUM']
results['queue_gap'] = df_valid['Gap']
results['TRANSPORT'] = df_valid['M14.QUE.ALL.TRANSPORT4MINOVERCNT']

# 위험 여부 (O/X) - 1650 기준 적용 (튜닝)
results['실제위험'] = np.where(results['실제값'] >= 1700, 'O', '')
results['예측위험'] = np.where(results['예측값'] >= 1650, 'O', '')  # ⭐️ 예측은 1650으로 기준 완화

# 통계용 구간 정보
results['Load_Category'] = results['실제값'].apply(get_load_category)

# -------------------------------------------------------------
# 6. 결과 리포트 출력
# -------------------------------------------------------------
print("\n📊 [구간별 예측 성능 분석]")
print("-" * 65)
print(f"{'Category':<15} | {'Count':<8} | {'MAE (오차)':<10} | {'RMSE':<10}")
print("-" * 65)

categories = ['Under_1200', '1200_1300', '1300_1400', '1400_1500', '1500_1600', '1600_1700', 'Over_1700']

total_mae = results['절대오차'].mean()
print(f"{'TOTAL (전체)':<15} | {len(results):<8} | {total_mae:<10.2f} | -")
print("-" * 65)

for cat in categories:
    subset = results[results['Load_Category'] == cat]
    if len(subset) > 0:
        mae = subset['절대오차'].mean()
        rmse = np.sqrt((subset['오차']**2).mean())
        print(f"{cat:<15} | {len(subset):<8} | {mae:<10.2f} | {rmse:<10.2f}")
    else:
        print(f"{cat:<15} | {'0':<8} | {'-':<10} | -")
print("-" * 65)

# Over_1700 상세
over_1700 = results[results['Load_Category'] == 'Over_1700']
if len(over_1700) > 0:
    acc = (over_1700['절대오차'] <= 20).mean() * 100
    print(f"\n🔥 [Over_1700 구간 상세]")
    print(f"   - 샘플 수: {len(over_1700)}개")
    print(f"   - 평균 오차: {over_1700['절대오차'].mean():.2f}")
    print(f"   - ±20 이내 정확도: {acc:.1f}%")

# CSV 저장
outfile = 'Evaluation_Result_Weighted.CSV'
results.to_csv(outfile, index=False, encoding='utf-8-sig')
print(f"\n💾 상세 결과 파일 저장 완료: {outfile}")
