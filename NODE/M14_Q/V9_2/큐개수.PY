#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 V12.0 학습 - Load_Category 밸런싱 적용
특징: 구간별 데이터 균형을 맞춰 고부하(1700+) 패턴 학습 강화
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import glob
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🚀 V12.0 학습 - Load_Category 밸런싱 적용 (280분 시퀀스)")
print("="*70)

PRED_OFFSET = 10
SEQ_LEN = 280

# --- 구간 분류 함수 ---
def get_load_category(cnt):
    if cnt <= 1200: return 'Under_1200'
    elif 1200 < cnt <= 1300: return '1200_1300'
    elif 1300 < cnt <= 1400: return '1300_1400'
    elif 1400 < cnt <= 1500: return '1400_1500'
    elif 1500 < cnt <= 1600: return '1500_1600'
    elif 1600 < cnt < 1700: return '1600_1700'
    else: return 'Over_1700'

def load_data(path_pattern):
    print(f"\n📂 데이터 로딩: {path_pattern}")
    files = glob.glob(path_pattern)
    if not files:
        # 파일 없으면 기본 파일 시도
        files = glob.glob('M14_Q_*.CSV')
        
    dfs = []
    for f in sorted(files):
        try:
            df = pd.read_csv(f, on_bad_lines='skip', encoding='utf-8')
            dfs.append(df)
        except:
            df = pd.read_csv(f, on_bad_lines='skip', encoding='cp949')
            dfs.append(df)
    
    if not dfs: raise ValueError("데이터 파일 없음!")
    return pd.concat(dfs, ignore_index=True)

def create_vectorized_features(df):
    """벡터화된 피처 생성 (280분 시퀀스)"""
    df = df.copy()
    
    # 숫자형 변환
    cols = ['TOTALCNT', 'M14AM14B', 'M14AM14BSUM', 'M10AM14A', 
            'M14.QUE.ALL.TRANSPORT4MINOVERCNT', 'M14.QUE.OHT.OHTUTIL',
            'M14.QUE.ALL.CURRENTQCREATED', 'M14.QUE.ALL.CURRENTQCOMPLETED']
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    df['Gap'] = df['M14.QUE.ALL.CURRENTQCREATED'] - df['M14.QUE.ALL.CURRENTQCOMPLETED']
    
    # Lag 및 Rolling 피처 (기존 로직 유지)
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

    # 타겟 생성
    df['Target_Total_Future'] = df['TOTALCNT'].shift(-PRED_OFFSET)
    df['Target_Delta'] = df['Target_Total_Future'] - df['TOTALCNT']
    
    # 결측 제거 (시퀀스 길이만큼 앞부분 데이터 삭제됨)
    df_clean = df.dropna().reset_index(drop=True)
    return df_clean

# --- 메인 실행 ---
# 1. 데이터 로드
df = load_data('M14_Q_*.CSV')

# 2. 피처 생성
print("\n🔄 피처 생성 중...")
df_features = create_vectorized_features(df)

# 3. Load_Category 생성 및 데이터 밸런싱
print("\n⚖️ 데이터 밸런싱 작업 시작 (Load_Category 기반)...")

# 타겟 값(미래 시점의 부하)을 기준으로 카테고리 분류
df_features['Load_Category'] = df_features['Target_Total_Future'].apply(get_load_category)

# 카테고리별 개수 확인
counts = df_features['Load_Category'].value_counts()
print("  [원본 분포]")
print(counts)

# 최소 샘플 수 기준 설정 (너무 적으면 학습 안되므로 최소 5000개 혹은 가장 적은 클래스 개수 사용)
# 여기서는 'Over_1700' 개수 혹은 10000개 중 작은 값으로 설정하여 언더샘플링
min_samples = counts.min() 
target_samples = max(min_samples, 5000) # 최소 5000개는 확보 시도, 데이터가 모자르면 min값 사용

print(f"  👉 밸런싱 목표 샘플 수: {target_samples} (각 클래스별)")

# 밸런싱 수행 (각 그룹별로 샘플링)
df_balanced = df_features.groupby('Load_Category', group_keys=False).apply(
    lambda x: x.sample(n=min(len(x), target_samples), random_state=42)
)
# 데이터 섞기
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"✅ 밸런싱 완료: 총 {len(df_balanced):,}행")
print("  [밸런싱 후 분포]")
print(df_balanced['Load_Category'].value_counts())

# 4. 학습 준비
exclude_cols = ['CURRTIME', 'Target_Total_Future', 'Target_Delta', 'Load_Category']
feature_cols = [c for c in df_balanced.columns if c not in exclude_cols]

X = df_balanced[feature_cols]
y_delta = df_balanced['Target_Delta']

# 5. 모델 학습
print(f"\n🚀 모델 학습 시작 (피처 {len(feature_cols)}개)...")
model = xgb.XGBRegressor(
    n_estimators=2000,     # 밸런싱 되었으므로 학습량 충분히
    learning_rate=0.01,
    max_depth=12,          # 복잡한 패턴 학습
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=4,
    device='cuda',         # GPU 사용 (없으면 'cpu')
    tree_method='hist',
    random_state=42
)

model.fit(X, y_delta, verbose=False)

# 6. 저장
print("\n💾 모델 저장 중...")
with open('model_v12_balanced.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('features_v12_balanced.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)
    
print("✅ 학습 및 저장 완료!")
