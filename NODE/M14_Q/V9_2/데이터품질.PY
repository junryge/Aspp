#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🛠️ 고부하(Over 1700) 학습 데이터 추출기 (V2 - 280분 시퀀스 반영)
기능: '10분 뒤 1700 이상'이 되는 시점의 데이터를 280분 과거 흐름까지 포함해 추출
"""
import numpy as np
import pandas as pd
import glob
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🛠️ 고부하(1700+) 학습 데이터 추출 (280분 시퀀스 & 10분 예측)")
print("="*70)

PRED_OFFSET = 10 # 10분 뒤 예측

def load_data(path_pattern):
    print(f"\n📂 원본 데이터 로딩: {path_pattern}")
    files = glob.glob(path_pattern)
    if not files: 
        print("⚠ 파일이 없어 기본 파일(M14_Q_*.CSV)을 찾습니다.")
        files = glob.glob('M14_Q_*.CSV')
    
    dfs = []
    for f in sorted(files):
        try: dfs.append(pd.read_csv(f, on_bad_lines='skip', encoding='utf-8'))
        except: dfs.append(pd.read_csv(f, on_bad_lines='skip', encoding='cp949'))
    
    if not dfs: raise ValueError("❌ 데이터 파일이 없습니다!")
    return pd.concat(dfs, ignore_index=True)

def create_vectorized_features(df):
    """280분 시퀀스를 반영한 피처 생성"""
    df = df.copy()
    cols = ['TOTALCNT', 'M14AM14B', 'M14AM14BSUM', 'M10AM14A', 
            'M14.QUE.ALL.TRANSPORT4MINOVERCNT', 'M14.QUE.OHT.OHTUTIL',
            'M14.QUE.ALL.CURRENTQCREATED', 'M14.QUE.ALL.CURRENTQCOMPLETED']
    for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    
    df['Gap'] = df['M14.QUE.ALL.CURRENTQCREATED'] - df['M14.QUE.ALL.CURRENTQCOMPLETED']
    
    # 1. Lag (시차 변수)
    # 280분 전의 상태와 비교하기 위해 Lag 280 추가
    lags = [1, 3, 5, 10, 20, 50, 100, 280] 
    for lag in lags:
        df[f'Total_Lag_{lag}'] = df['TOTALCNT'].shift(lag)
        df[f'M14B_Lag_{lag}'] = df['M14AM14B'].shift(lag)
    
    # 2. Rolling (이동평균) - 280분 추가!
    windows = [10, 30, 60, 100, 280] 
    for w in windows:
        df[f'Total_Mean_{w}'] = df['TOTALCNT'].rolling(w).mean()
        df[f'Total_Std_{w}'] = df['TOTALCNT'].rolling(w).std()

    # 3. 변화량 (Delta & Slope)
    df['Slope_1'] = df['TOTALCNT'] - df['TOTALCNT'].shift(1)
    df['Slope_3'] = df['TOTALCNT'] - df['TOTALCNT'].shift(3)
    df['Accel_1'] = df['Slope_1'] - df['Slope_1'].shift(1)

    # 4. 타겟 (10분 뒤 미래 값)
    df['Target_Total_Future'] = df['TOTALCNT'].shift(-PRED_OFFSET)
    
    return df.dropna().reset_index(drop=True)

# --- 메인 실행 ---
df_raw = load_data('M14_Q_*.CSV')

print("\n🔄 280분 시퀀스 피처 생성 중...")
df_features = create_vectorized_features(df_raw)

print("\n🔍 1700 이상 고부하 데이터 추출 중...")
# 1700 이상인 데이터만 추출
high_load_df = df_features[df_features['Target_Total_Future'] >= 1700].copy()

count = len(high_load_df)
print(f"\n📊 추출 결과")
print(f"   - 1700+ 데이터: {count:,}개 추출됨")

if count > 0:
    outfile = 'High_Load_Training_Data.csv'
    high_load_df.to_csv(outfile, index=False, encoding='utf-8-sig')
    print(f"\n💾 저장 완료: {outfile}")
    print("   (이 파일은 280분 전의 데이터까지 포함하고 있어 V11.5 학습에 최적입니다)")
else:
    print("\n⚠ 1700 이상인 데이터가 없습니다.")
