#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 V11.5 학습 - 변화량(Delta) 집중 공략
특징: 고부하(1700) 및 '급상승(Delta 폭발)' 구간에 가중치를 부여하여
      모델이 현재값을 따라가는 게 아니라 '변화'를 예측하도록 강제함.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import glob
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🚀 V11.5 학습 - 변화량(Delta) 집중 학습")
print("="*70)

PRED_OFFSET = 10

def load_data(path_pattern):
    print(f"\n📂 데이터 로딩: {path_pattern}")
    files = glob.glob(path_pattern)
    if not files: 
        print("⚠ 지정된 파일이 없어 기본 패턴(M14_Q_*.CSV)을 찾습니다.")
        files = glob.glob('M14_Q_*.CSV')
    
    dfs = []
    for f in sorted(files):
        try: dfs.append(pd.read_csv(f, on_bad_lines='skip', encoding='utf-8'))
        except: dfs.append(pd.read_csv(f, on_bad_lines='skip', encoding='cp949'))
    
    if not dfs: raise ValueError("❌ 학습할 데이터 파일이 없습니다!")
    return pd.concat(dfs, ignore_index=True)

def create_vectorized_features(df):
    df = df.copy()
    cols = ['TOTALCNT', 'M14AM14B', 'M14AM14BSUM', 'M10AM14A', 
            'M14.QUE.ALL.TRANSPORT4MINOVERCNT', 'M14.QUE.OHT.OHTUTIL',
            'M14.QUE.ALL.CURRENTQCREATED', 'M14.QUE.ALL.CURRENTQCOMPLETED']
    for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    
    df['Gap'] = df['M14.QUE.ALL.CURRENTQCREATED'] - df['M14.QUE.ALL.CURRENTQCOMPLETED']
    
    # Lag & Rolling
    lags = [1, 3, 5, 10, 20]
    for lag in lags:
        df[f'Total_Lag_{lag}'] = df['TOTALCNT'].shift(lag)
        df[f'M14B_Lag_{lag}'] = df['M14AM14B'].shift(lag)
    
    windows = [10, 30, 60]
    for w in windows:
        df[f'Total_Mean_{w}'] = df['TOTALCNT'].rolling(w).mean()
        df[f'Total_Std_{w}'] = df['TOTALCNT'].rolling(w).std()

    # 🔥 [중요] 변화량 Feature 강화 (따라가기 방지용)
    # 기울기(Slope)
    df['Slope_1'] = df['TOTALCNT'] - df['TOTALCNT'].shift(1)  # 1분간 변화량
    df['Slope_3'] = df['TOTALCNT'] - df['TOTALCNT'].shift(3)  # 3분간 변화량
    df['Slope_5'] = df['TOTALCNT'] - df['TOTALCNT'].shift(5)  # 5분간 변화량
    
    # 가속도(Accel): 변화량이 더 커지고 있는지?
    df['Accel_1'] = df['Slope_1'] - df['Slope_1'].shift(1)

    # 타겟 생성
    df['Target_Total_Future'] = df['TOTALCNT'].shift(-PRED_OFFSET)
    df['Target_Delta'] = df['Target_Total_Future'] - df['TOTALCNT'] # 변화량 예측
    
    return df.dropna().reset_index(drop=True)

# 1. 실행
df = load_data('M14_Q_*.CSV')
print("\n🔄 피처 생성 중...")
df_features = create_vectorized_features(df)

# 2. 학습 준비
exclude = ['CURRTIME', 'Target_Total_Future', 'Target_Delta']
cols = [c for c in df_features.columns if c not in exclude]
X = df_features[cols]
y_delta = df_features['Target_Delta']
y_total = df_features['Target_Total_Future'] # 가중치용

# -------------------------------------------------------------
# 🔥 [핵심 수정] 가중치 전략 변경
# -------------------------------------------------------------
weights = np.ones(len(y_total))

# 1. 고부하 구간 중요 (기존 유지)
weights[y_total >= 1500] = 5
weights[y_total >= 1700] = 20

# 2. 급상승 구간 중요 (신규 추가)
# 미래에 +50 이상 급등하는 구간은 놓치면 안 됨!
weights[y_delta >= 50] = 10 

print(f"\n⚖️ 가중치 적용: 1700+ 구간(20배) & 급상승 구간(10배)")

print(f"\n🚀 모델 학습 시작 (변화량 집중)...")
model = xgb.XGBRegressor(
    n_estimators=2000,
    learning_rate=0.015,  # 정밀 학습
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    n_jobs=4,
    device='cuda',        # GPU 없으면 'cpu'로 변경
    tree_method='hist',
    random_state=42
)

# 변화량(Delta)을 학습 목표로 설정
model.fit(X, y_delta, sample_weight=weights, verbose=False)

# 저장
with open('model_v11_delta_focused.pkl', 'wb') as f: pickle.dump(model, f)
with open('features_v11_delta_focused.pkl', 'wb') as f: pickle.dump(cols, f)
print("✅ 학습 완료! (model_v11_delta_focused.pkl)")
