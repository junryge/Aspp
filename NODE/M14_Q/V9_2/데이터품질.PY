#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🛠️ 고부하(Over 1700) 학습 데이터 추출기
기능: 전체 데이터에서 '10분 뒤 1700 이상'이 되는 시점의 피처(X)와 타겟(y)만 추출하여 저장
용도: 1. 데이터 증폭(Oversampling)용 소스
      2. 1700 이상 패턴 분석 (엑셀 열람용)
"""
import numpy as np
import pandas as pd
import glob
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🛠️ 고부하(1700+) 학습 시퀀스 데이터 추출")
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
    """학습에 사용되는 피처와 동일하게 생성"""
    df = df.copy()
    cols = ['TOTALCNT', 'M14AM14B', 'M14AM14BSUM', 'M10AM14A', 
            'M14.QUE.ALL.TRANSPORT4MINOVERCNT', 'M14.QUE.OHT.OHTUTIL',
            'M14.QUE.ALL.CURRENTQCREATED', 'M14.QUE.ALL.CURRENTQCOMPLETED']
    for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    
    df['Gap'] = df['M14.QUE.ALL.CURRENTQCREATED'] - df['M14.QUE.ALL.CURRENTQCOMPLETED']
    
    # 1. Lag (시차 변수)
    lags = [1, 3, 5, 10, 20]
    for lag in lags:
        df[f'Total_Lag_{lag}'] = df['TOTALCNT'].shift(lag)
        df[f'M14B_Lag_{lag}'] = df['M14AM14B'].shift(lag)
    
    # 2. Rolling (이동평균)
    windows = [10, 30, 60]
    for w in windows:
        df[f'Total_Mean_{w}'] = df['TOTALCNT'].rolling(w).mean()
        df[f'Total_Std_{w}'] = df['TOTALCNT'].rolling(w).std()

    # 3. 변화량 (Delta & Slope)
    df['Slope_1'] = df['TOTALCNT'] - df['TOTALCNT'].shift(1)
    df['Slope_3'] = df['TOTALCNT'] - df['TOTALCNT'].shift(3)
    df['Slope_5'] = df['TOTALCNT'] - df['TOTALCNT'].shift(5)
    df['Accel_1'] = df['Slope_1'] - df['Slope_1'].shift(1)

    # 4. 타겟 (미래 값)
    df['Target_Total_Future'] = df['TOTALCNT'].shift(-PRED_OFFSET)
    df['Target_Delta'] = df['Target_Total_Future'] - df['TOTALCNT']
    
    return df.dropna().reset_index(drop=True)

# --- 메인 실행 ---

# 1. 전체 데이터 로드
df_raw = load_data('M14_Q_*.CSV')

# 2. 피처 생성 (시간 좀 걸림)
print("\n🔄 피처 생성 및 시퀀스 변환 중...")
df_features = create_vectorized_features(df_raw)
print(f"   -> 전체 학습 가능 시퀀스: {len(df_features):,}개")

# 3. 1700 이상 데이터만 필터링
print("\n🔍 1700 이상 고부하 데이터 추출 중...")
high_load_df = df_features[df_features['Target_Total_Future'] >= 1700].copy()

# 4. 결과 출력 및 저장
count = len(high_load_df)
total = len(df_features)
ratio = count / total * 100

print(f"\n📊 추출 결과 요약")
print(f"   - 전체 데이터: {total:,}개")
print(f"   - 1700+ 데이터: {count:,}개")
print(f"   - 희소 비율: {ratio:.2f}% (이러니 모델이 못 맞추죠!)")

if count > 0:
    # 날짜별 분포 확인을 위해 시간 컬럼 복원 (원본 인덱스 매칭 필요하지만 여기선 간단히 저장)
    outfile = 'High_Load_Training_Data.csv'
    high_load_df.to_csv(outfile, index=False, encoding='utf-8-sig')
    print(f"\n💾 저장 완료: {outfile}")
    print("   👉 이 파일을 엑셀로 열어서 어떤 패턴일 때 1700이 되는지 분석해보세요.")
    print("   👉 학습 코드에서 이 파일을 로드해서 50배 복사해서 쓰면 됩니다.")
else:
    print("\n⚠ 1700 이상인 데이터가 하나도 없습니다! 원본 데이터를 확인해주세요.")
