#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🛠️ 순수 고부하 데이터 추출기 (학습 X, 추출 O)
기능: 전체 데이터에서 280분 시퀀스 피처를 생성한 뒤,
      '12월 11일'을 제외한 나머지 날짜의 '1700 이상' 데이터만 뽑아서 저장함.
"""
import numpy as np
import pandas as pd
import glob
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🛡️ 학습용 1700+ 데이터 추출 (12월 11일 철저 배제)")
print("="*70)

# 🔴 설정
PRED_OFFSET = 10              # 10분 뒤 예측
EXCLUDE_DATE = '2025-12-11'   # 평가용 날짜 (학습 데이터에서 무조건 제외)
TARGET_THRESHOLD = 1700       # 추출할 목표값 기준

def load_data(path_pattern):
    print(f"\n📂 원본 데이터 로딩 중...")
    files = glob.glob(path_pattern)
    if not files: files = glob.glob('M14_Q_*.CSV')
    
    dfs = []
    for f in sorted(files):
        try: dfs.append(pd.read_csv(f, on_bad_lines='skip', encoding='utf-8'))
        except: dfs.append(pd.read_csv(f, on_bad_lines='skip', encoding='cp949'))
        
    if not dfs: raise ValueError("❌ 파일이 없습니다.")
    return pd.concat(dfs, ignore_index=True)

def create_vectorized_features(df):
    """
    학습에 필요한 280분 시퀀스 피처 생성
    """
    df = df.copy()
    
    # 1. 숫자형 변환
    cols = ['TOTALCNT', 'M14AM14B', 'M14AM14BSUM', 'M10AM14A', 
            'M14.QUE.ALL.TRANSPORT4MINOVERCNT', 'M14.QUE.OHT.OHTUTIL',
            'M14.QUE.ALL.CURRENTQCREATED', 'M14.QUE.ALL.CURRENTQCOMPLETED']
    for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    
    df['Gap'] = df['M14.QUE.ALL.CURRENTQCREATED'] - df['M14.QUE.ALL.CURRENTQCOMPLETED']
    
    # 2. Lag & Rolling (280분까지 포함)
    lags = [1, 3, 5, 10, 20, 50, 100, 280]
    for lag in lags:
        df[f'Total_Lag_{lag}'] = df['TOTALCNT'].shift(lag)
        df[f'M14B_Lag_{lag}'] = df['M14AM14B'].shift(lag)
    
    windows = [10, 30, 60, 100, 280]
    for w in windows:
        df[f'Total_Mean_{w}'] = df['TOTALCNT'].rolling(w).mean()
        df[f'Total_Std_{w}'] = df['TOTALCNT'].rolling(w).std()

    # 3. 변화량 (Slope, Accel) - V11.5 핵심 피처
    df['Slope_1'] = df['TOTALCNT'] - df['TOTALCNT'].shift(1)
    df['Slope_3'] = df['TOTALCNT'] - df['TOTALCNT'].shift(3)
    df['Slope_5'] = df['TOTALCNT'] - df['TOTALCNT'].shift(5)
    df['Accel_1'] = df['Slope_1'] - df['Slope_1'].shift(1)

    # 4. 타겟 (10분 뒤 미래 값)
    df['Target_Total_Future'] = df['TOTALCNT'].shift(-PRED_OFFSET)
    
    # 결측 제거 (280분치 데이터가 확보된 이후부터 사용)
    return df.dropna().reset_index(drop=True)

# --- 메인 실행 ---
if __name__ == '__main__':
    # 1. 데이터 로드
    df_raw = load_data('M14_Q_*.CSV')
    
    # 2. 날짜 파싱 (필터링용)
    df_raw['datetime'] = pd.to_datetime(df_raw['CURRTIME'].astype(str), format='%Y%m%d%H%M', errors='coerce')
    
    # 3. 피처 생성
    print("\n🔄 피처 생성 중 (280분 시퀀스 계산)...")
    df_features = create_vectorized_features(df_raw)
    
    # 4. [핵심] 데이터 필터링
    print(f"\n🔍 데이터 추출 조건:")
    print(f"   1) 날짜: {EXCLUDE_DATE} 제외")
    print(f"   2) 값: {TARGET_THRESHOLD} 이상")
    
    # 4-1. 평가일 제외
    mask_train_date = df_features['datetime'].dt.date.astype(str) != EXCLUDE_DATE
    df_train_pool = df_features[mask_train_date].copy()
    
    # 4-2. 1700 이상만 추출
    df_high_load = df_train_pool[df_train_pool['Target_Total_Future'] >= TARGET_THRESHOLD].copy()
    
    # 5. 결과 저장
    count = len(df_high_load)
    print(f"\n📊 최종 추출 결과")
    print(f"   - 원본 전체: {len(df_features):,}개")
    print(f"   - 평가일 제외 후: {len(df_train_pool):,}개")
    print(f"   - 최종 추출(1700+): {count:,}개")
    
    if count > 0:
        outfile = 'High_Load_Train_Only.csv'
        df_high_load.to_csv(outfile, index=False, encoding='utf-8-sig')
        print(f"\n✅ 저장 완료: {outfile}")
        print("   -> 이 파일을 학습 코드에서 불러와서 증폭(concat)시키면 됩니다.")
    else:
        print("\n⚠ 조건에 맞는 데이터가 없습니다.")
