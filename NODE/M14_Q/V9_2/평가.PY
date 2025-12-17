#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 V11.8 최종 평가 - AI vs Rule 판단형 시스템
핵심 기능:
  - 결과 파일에 'Source' 컬럼 추가 (AI / Rule)
  - 사용자는 어떤 로직이 선택되었는지 눈으로 확인 가능
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import warnings
from datetime import timedelta
warnings.filterwarnings('ignore')

print("="*70)
print("🚀 V11.8 평가 - 예측 출처(Source) 판단 시스템")
print("="*70)

PRED_OFFSET = 10
EVAL_FILE = '20251211_16평가.CSV'

# 1. 모델 로드
try:
    with open('model_v11_280seq.pkl', 'rb') as f: model_base = pickle.load(f)
    with open('features_v11_280seq.pkl', 'rb') as f: feature_cols = pickle.load(f)
except:
    try: 
        with open('model_v10_base.pkl', 'rb') as f: model_base = pickle.load(f)
        with open('features_v10.pkl', 'rb') as f: feature_cols = pickle.load(f)
    except:
        print("❌ 모델 파일 없음."); exit()

# 2. 데이터 로드
try: df_raw = pd.read_csv(EVAL_FILE, encoding='utf-8')
except: df_raw = pd.read_csv(EVAL_FILE, encoding='cp949')
df_raw['CURRTIME_OBJ'] = pd.to_datetime(df_raw['CURRTIME'].astype(str), format='%Y%m%d%H%M', errors='coerce')

# 3. 피처 생성
def create_features(df):
    df = df.copy()
    cols = ['TOTALCNT', 'M14AM14B', 'M14AM14BSUM', 'M10AM14A', 'M14.QUE.ALL.TRANSPORT4MINOVERCNT', 
            'M14.QUE.OHT.OHTUTIL', 'M14.QUE.ALL.CURRENTQCREATED', 'M14.QUE.ALL.CURRENTQCOMPLETED']
    for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    df['Gap'] = df['M14.QUE.ALL.CURRENTQCREATED'] - df['M14.QUE.ALL.CURRENTQCOMPLETED']
    
    lags = [1, 3, 5, 10, 20, 50, 100, 280]
    for lag in lags:
        df[f'Total_Lag_{lag}'] = df['TOTALCNT'].shift(lag)
        df[f'M14B_Lag_{lag}'] = df['M14AM14B'].shift(lag)
    
    windows = [10, 30, 60, 100, 280]
    for w in windows:
        df[f'Total_Mean_{w}'] = df['TOTALCNT'].rolling(w).mean()
        df[f'Total_Std_{w}'] = df['TOTALCNT'].rolling(w).std()

    df['Slope_1'] = df['TOTALCNT'] - df['TOTALCNT'].shift(1)
    df['Slope_3'] = df['TOTALCNT'] - df['TOTALCNT'].shift(3)
    df['Accel_1'] = df['Slope_1'] - df['Slope_1'].shift(1)
    df['Total_Delta_10'] = df['TOTALCNT'] - df['Total_Lag_10']
    df['Target_Total_Future'] = df['TOTALCNT'].shift(-PRED_OFFSET)
    return df

print("🔄 피처 생성 중...")
df_feat = create_features(df_raw)
df_valid = df_feat.dropna(subset=feature_cols + ['Target_Total_Future']).copy()

# 4. 예측 수행
print("🚀 AI 및 Rule 예측 계산 중...")
X_eval = df_valid[feature_cols]

# (1) AI 예측값
pred_ai = df_valid['TOTALCNT'] + model_base.predict(X_eval)

# (2) Rule 예측값 (패턴 보정)
# 공식: (입고량 - 500) * 2.2
def calculate_rule(row, ai_val):
    inp = row['M14AM14B']
    if inp > 500:
        boost = (inp - 500) * 2.2
        if ai_val > 1680: boost *= 0.5 # AI가 이미 높으면 보정 완화
        return min(ai_val + boost, 1850)
    return ai_val # 입고량 낮으면 AI값 그대로

pred_rule = df_valid.apply(lambda row: calculate_rule(row, pred_ai.loc[row.name]), axis=1)

# 5. [핵심] 소스 결정 (Selection Logic)
# 입고량 520 이상이면 'Rule' 선택, 아니면 'AI' 선택
print("⚖️ Source(출처) 결정 중...")

df_valid['AI_Pred'] = pred_ai
df_valid['Rule_Pred'] = pred_rule
df_valid['M14AM14B'] = df_valid['M14AM14B']

# 조건문으로 Source 결정
df_valid['Source'] = np.where(df_valid['M14AM14B'] >= 520, 'Rule', 'AI')

# 최종값 매핑
df_valid['Final_Pred'] = np.where(df_valid['Source'] == 'Rule', df_valid['Rule_Pred'], df_valid['AI_Pred'])

# 6. 결과 정리 및 저장
results = pd.DataFrame()
results['현재시간'] = df_valid['CURRTIME_OBJ'].dt.strftime('%Y-%m-%d %H:%M')
results['입고량'] = df_valid['M14AM14B']
results['실제값'] = df_valid['Target_Total_Future']
results['AI예측'] = df_valid['AI_Pred'].round(2)
results['Rule예측'] = df_valid['Rule_Pred'].round(2)
results['Source'] = df_valid['Source']  # ✅ 여기가 핵심 (AI vs Rule)
results['최종예측'] = df_valid['Final_Pred'].round(2)
results['오차'] = (results['실제값'] - results['최종예측']).abs().round(2)

# 통계 출력
mae = results['오차'].mean()
print(f"\n📊 [최종 분석 결과]")
print(f"   - 전체 데이터 수: {len(results)}개")
print(f"   - AI 선택 횟수: {(results['Source']=='AI').sum()}회 (안정 구간)")
print(f"   - Rule 선택 횟수: {(results['Source']=='Rule').sum()}회 (위험 구간)")
print(f"   - 전체 MAE: {mae:.2f}")

# 10:20분 구간 확인
print("\n[🔎 10:20분 구간 판단 결과]")
mask_10 = (results['현재시간'] >= '2025-12-11 10:20') & (results['현재시간'] <= '2025-12-11 10:23')
print(results.loc[mask_10, ['현재시간', '입고량', 'AI예측', 'Rule예측', 'Source', '최종예측']].to_string(index=False))

# 17:34분 구간 확인
print("\n[🔎 17:34분 구간 판단 결과]")
mask_17 = (results['현재시간'] >= '2025-12-11 17:33') & (results['현재시간'] <= '2025-12-11 17:36')
print(results.loc[mask_17, ['현재시간', '입고량', 'AI예측', 'Rule예측', 'Source', '최종예측']].to_string(index=False))

# 저장
results.to_csv('Final_Eval_With_Source.CSV', index=False, encoding='utf-8-sig')
print("\n💾 저장 완료: Final_Eval_With_Source.CSV")
print("   👉 엑셀에서 'Source' 컬럼을 필터링해보세요!")