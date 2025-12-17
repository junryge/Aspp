#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 V13.0 평가 - 12월 11일 실전 테스트
특징: 실전 알람 기준(1650) 적용하여 방어율 측정
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import warnings
from datetime import timedelta
warnings.filterwarnings('ignore')

print("="*70)
print("🚀 V13.0 평가 - 12월 11일 집중 분석")
print("="*70)

PRED_OFFSET = 10
# 🔴 평가할 파일명을 정확히 입력하세요!
EVAL_FILE = '20251211_16평가.CSV' 

def create_vectorized_features(df):
    """피처 생성"""
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

    df['Target_Total_Future'] = df['TOTALCNT'].shift(-PRED_OFFSET)
    return df

# 1. 모델 로드
print("📂 모델 로딩 (model_v13_final.pkl)...")
try:
    with open('model_v13_final.pkl', 'rb') as f: model = pickle.load(f)
    with open('features_v13_final.pkl', 'rb') as f: cols = pickle.load(f)
except:
    print("❌ 모델 파일이 없습니다! 'train_v13_final.py'를 먼저 실행하세요.")
    exit()

# 2. 데이터 로드
print(f"📂 평가 데이터: {EVAL_FILE}")
try: df_raw = pd.read_csv(EVAL_FILE, encoding='utf-8')
except: df_raw = pd.read_csv(EVAL_FILE, encoding='cp949')

df_raw['CURRTIME_OBJ'] = pd.to_datetime(df_raw['CURRTIME'].astype(str), format='%Y%m%d%H%M', errors='coerce')

# 3. 피처 생성 및 예측
print("🔄 피처 생성 및 예측 중...")
df_feat = create_vectorized_features(df_raw)
df_valid = df_feat.dropna(subset=cols + ['Target_Total_Future', 'CURRTIME_OBJ']).copy()

# 예측값 계산
pred_total = df_valid['TOTALCNT'] + model.predict(df_valid[cols])

# 4. 결과 정리
results = pd.DataFrame()
current_time = df_valid['CURRTIME_OBJ']
results['현재시간'] = current_time.dt.strftime('%Y-%m-%d %H:%M')
results['현재TOTALCNT'] = df_valid['TOTALCNT']
results['예측시점'] = (current_time + timedelta(minutes=PRED_OFFSET)).dt.strftime('%Y-%m-%d %H:%M')
results['실제값'] = df_valid['Target_Total_Future']
results['예측값'] = pred_total
results['오차'] = (results['실제값'] - results['예측값']).round(2)
results['절대오차'] = results['오차'].abs()

# 🔥 [실전 알람 튜닝]
# 예측값이 1650만 넘어도 "위험(O)"으로 판단
results['실제위험'] = np.where(results['실제값'] >= 1700, 'O', '')
results['예측위험'] = np.where(results['예측값'] >= 1650, 'O', '') 

# 5. 리포트 출력
print("\n📊 [12월 11일 실전 평가 결과]")
high_risk = results[results['실제값'] >= 1700]

if len(high_risk) > 0:
    detected = high_risk[high_risk['예측위험'] == 'O']
    print(f"🔥 실제 1700+ 발생 건수: {len(high_risk)}건")
    print(f"✅ 알람 발송 성공(예측>=1650): {len(detected)}건")
    print(f"🎯 방어율: {len(detected)/len(high_risk)*100:.1f}%")
    
    print("\n[상세 내역 - 처음 5건]")
    print(high_risk[['현재시간', '현재TOTALCNT', '실제값', '예측값', '예측위험']].head().to_string(index=False))
else:
    print("평가 데이터에 1700+ 구간이 없습니다.")

outfile = 'Final_Evaluation_Result_Dec11.CSV'
results.to_csv(outfile, index=False, encoding='utf-8-sig')
print(f"\n💾 결과 저장: {outfile}")
