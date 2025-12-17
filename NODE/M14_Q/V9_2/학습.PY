#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 V11.5 학습 - 변화량(Delta) 집중 공략 (수정됨)
특징: 고부하(1700) + '급상승(Delta 폭발)' 구간 동시 가중치 부여
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
        print("⚠ 지정된 파일이 없어 기본 파일(M14_Q_*.CSV)을 찾습니다.")
        files = glob.glob('M14_Q_*.CSV')
        
    dfs = []
    for f in sorted(files):
        try: dfs.append(pd.read_csv(f, on_bad_lines='skip', encoding='utf-8'))
        except: dfs.append(pd.read_csv(f, on_bad_lines='skip', encoding='cp949'))
            
    if not dfs: raise ValueError("❌ 데이터 파일을 찾을 수 없습니다!")
    return pd.concat(dfs, ignore_index=True)

def create_vectorized_features(df):
    """벡터화된 피처 생성"""
    df = df.copy()
    
    # 숫자형 변환
    cols = ['TOTALCNT', 'M14AM14B', 'M14AM14BSUM', 'M10AM14A', 
            'M14.QUE.ALL.TRANSPORT4MINOVERCNT', 'M14.QUE.OHT.OHTUTIL',
            'M14.QUE.ALL.CURRENTQCREATED', 'M14.QUE.ALL.CURRENTQCOMPLETED']
    for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    df['Gap'] = df['M14.QUE.ALL.CURRENTQCREATED'] - df['M14.QUE.ALL.CURRENTQCOMPLETED']
    
    # 1. 시차 변수 (Lags)
    lags = [1, 3, 5, 10, 20]
    for lag in lags:
        df[f'Total_Lag_{lag}'] = df['TOTALCNT'].shift(lag)
        df[f'M14B_Lag_{lag}'] = df['M14AM14B'].shift(lag)
    
    # 2. 이동 평균 (Rolling)
    windows = [10, 30, 60]
    for w in windows:
        df[f'Total_Mean_{w}'] = df['TOTALCNT'].rolling(w).mean()
        df[f'Total_Std_{w}'] = df['TOTALCNT'].rolling(w).std()

    # 3. 🔥 [강화] 변화량 및 기울기 (Lag 현상 방지 핵심)
    # 1분 전 대비 얼마나 변했나? (가장 민감한 지표)
    df['Slope_1'] = df['TOTALCNT'] - df['TOTALCNT'].shift(1)
    df['Slope_3'] = df['TOTALCNT'] - df['TOTALCNT'].shift(3)
    
    # 가속도 (변화의 속도가 빨라지는가?)
    df['Accel_1'] = df['Slope_1'] - df['Slope_1'].shift(1)
    
    # 기존 변화량 지표 유지
    df['Total_Delta_10'] = df['TOTALCNT'] - df['Total_Lag_10']

    # 4. 타겟 생성 (10분 후 예측)
    df['Target_Total_Future'] = df['TOTALCNT'].shift(-PRED_OFFSET)
    df['Target_Delta'] = df['Target_Total_Future'] - df['TOTALCNT']
    
    return df.dropna().reset_index(drop=True)

# --- 메인 실행 ---
# 1. 데이터 로드
df = load_data('M14_Q_*.CSV')

# 2. 피처 생성
print("\n🔄 피처 생성 중...")
df_features = create_vectorized_features(df)
print(f"✅ 생성 완료: {len(df_features):,}행, 컬럼 {df_features.shape[1]}개")

# 3. 학습 준비
exclude_cols = ['CURRTIME', 'Target_Total_Future', 'Target_Delta']
feature_cols = [c for c in df_features.columns if c not in exclude_cols]
X = df_features[feature_cols]
y_delta = df_features['Target_Delta']        # 목표: 변화량 맞추기
y_total = df_features['Target_Total_Future'] # 가중치 설정용 참조값

# 4. 🔥 [핵심] 가중치 전략 (Weighting)
# 목표: 모델이 "급변하는 순간"에 집중하게 만듦
weights = np.ones(len(y_total))

# 규칙 A: 절대값이 높으면 중요 (기존 유지)
weights[y_total >= 1500] = 5
weights[y_total >= 1700] = 20

# 규칙 B: 🔥 변화량이 크면 중요 (Delta >= 50이면 가중치 10배)
# -> 이걸 추가해야 모델이 "멍때리다가 1700 가는 것"을 잡습니다.
weights[y_delta >= 50] = 10 

print("\n🚀 모델 학습 시작 (변화량 + 고부하 집중)...")
model_base = xgb.XGBRegressor(
    n_estimators=2000,      # 학습량 충분히
    learning_rate=0.015,    # 정교하게
    max_depth=10,           # 복잡한 패턴 인식
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    n_jobs=4,
    device='cuda',          # GPU 사용 (없으면 'cpu')
    tree_method='hist',
    random_state=42
)

# 학습 (변화량 y_delta 학습)
model_base.fit(X, y_delta, sample_weight=weights, verbose=False)

# 저장
print("\n💾 모델 저장 중...")
with open('model_v11_delta.pkl', 'wb') as f:
    pickle.dump(model_base, f)
with open('features_v11_delta.pkl', 'wb') as f:
    pickle.dump(list(feature_cols), f)
    
print("✅ 학습 완료! (model_v11_delta.pkl)")