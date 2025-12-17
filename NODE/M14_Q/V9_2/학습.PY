#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 V13.0 학습 - 데이터 증폭 (Oversampling) & 일반화
특징: 평가일(12월 11일) 제외하고, 다른 고부하 날짜를 5배 증폭하여 학습
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import glob
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🚀 V13.0 학습 - 데이터 증폭 (쪽집게 과외 모드)")
print("="*70)

PRED_OFFSET = 10
EVAL_DATE = '2025-12-11' # 🔴 이 날짜는 학습에서 철저히 배제 (평가용)

def load_data(path_pattern):
    print(f"\n📂 데이터 로딩: {path_pattern}")
    files = glob.glob(path_pattern)
    if not files: files = glob.glob('M14_Q_*.CSV')
    
    dfs = []
    for f in sorted(files):
        try: dfs.append(pd.read_csv(f, on_bad_lines='skip', encoding='utf-8'))
        except: dfs.append(pd.read_csv(f, on_bad_lines='skip', encoding='cp949'))
    
    if not dfs: raise ValueError("❌ 데이터 파일 없음!")
    return pd.concat(dfs, ignore_index=True)

def create_vectorized_features(df):
    """V11 기반 피처 생성 (검증된 로직)"""
    df = df.copy()
    cols = ['TOTALCNT', 'M14AM14B', 'M14AM14BSUM', 'M10AM14A', 
            'M14.QUE.ALL.TRANSPORT4MINOVERCNT', 'M14.QUE.OHT.OHTUTIL',
            'M14.QUE.ALL.CURRENTQCREATED', 'M14.QUE.ALL.CURRENTQCOMPLETED']
    for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    
    df['Gap'] = df['M14.QUE.ALL.CURRENTQCREATED'] - df['M14.QUE.ALL.CURRENTQCOMPLETED']
    
    # Lag & Rolling (V11 설정 유지)
    lags = [1, 3, 5, 10, 20]
    for lag in lags:
        df[f'Total_Lag_{lag}'] = df['TOTALCNT'].shift(lag)
        df[f'M14B_Lag_{lag}'] = df['M14AM14B'].shift(lag)
    
    windows = [10, 30, 60]
    for w in windows:
        df[f'Total_Mean_{w}'] = df['TOTALCNT'].rolling(w).mean()
        df[f'Total_Std_{w}'] = df['TOTALCNT'].rolling(w).std()

    # 타겟 생성
    df['Target_Total_Future'] = df['TOTALCNT'].shift(-PRED_OFFSET)
    df['Target_Delta'] = df['Target_Total_Future'] - df['TOTALCNT']
    
    return df.dropna().reset_index(drop=True)

# 1. 데이터 로드
df_raw = load_data('M14_Q_*.CSV')

# 2. 날짜 파싱 및 평가일 제외 (Cheating 방지)
df_raw['datetime'] = pd.to_datetime(df_raw['CURRTIME'].astype(str), format='%Y%m%d%H%M', errors='coerce')
df_train_source = df_raw[df_raw['datetime'].dt.date.astype(str) != EVAL_DATE].copy()

print(f"📉 학습 데이터 필터링: {EVAL_DATE} 데이터 제외 완료")
print(f"   (남은 데이터로 12월 패턴을 학습합니다)")

# 3. 피처 생성
print("\n🔄 피처 생성 중...")
df_features = create_vectorized_features(df_train_source)

# 4. [핵심] 데이터 증폭 (Oversampling)
# 1650 이상인 '진짜' 위험 데이터를 찾아서 5배로 복사합니다.
print("\n⚖️ 데이터 증폭 작업 (Oversampling)...")

high_load_indices = df_features[df_features['Target_Total_Future'] >= 1650].index
if len(high_load_indices) > 0:
    high_load_data = df_features.loc[high_load_indices]
    # 5배 뻥튀기 (원본 포함 총 6배 효과)
    augmented_data = pd.concat([high_load_data] * 5, ignore_index=True)
    df_final = pd.concat([df_features, augmented_data], ignore_index=True)
    print(f"🔥 고부하(1650+) 데이터 {len(high_load_data)}개를 5배 복사하여 추가함!")
else:
    df_final = df_features
    print("⚠ 고부하 데이터가 부족하여 증폭하지 못했습니다.")

# 순서 섞기
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

# 5. 가중치 설정 (V11 로직)
y_future = df_final['Target_Total_Future'].values
weights = np.ones(len(df_final))
weights[y_future >= 1500] = 5
weights[y_future >= 1700] = 20  # 중요도 유지

# 6. 학습 준비
exclude = ['CURRTIME', 'datetime', 'Target_Total_Future', 'Target_Delta']
cols = [c for c in df_final.columns if c not in exclude]
X = df_final[cols]
y = df_final['Target_Delta']

print(f"\n🚀 모델 학습 시작 (증폭된 데이터 {len(df_final):,}건)...")
model = xgb.XGBRegressor(
    n_estimators=2000,
    learning_rate=0.02,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    n_jobs=4,
    device='cuda',
    tree_method='hist',
    random_state=42
)

model.fit(X, y, sample_weight=weights, verbose=False)

# 7. 저장
with open('model_v13_final.pkl', 'wb') as f: pickle.dump(model, f)
with open('features_v13_final.pkl', 'wb') as f: pickle.dump(cols, f)
print("✅ V13.0 학습 완료! (model_v13_final.pkl)")
