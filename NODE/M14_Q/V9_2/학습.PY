#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 V11.0 학습 - 10분 예측 (시퀀스 280분)
시퀀스: 280분, 예측: 10분 후
특징: 벡터화 연산, 가중치 학습, 280분 시계열 활용
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import glob
import warnings
from sklearn.metrics import mean_absolute_error
warnings.filterwarnings('ignore')
print("="*70)
print("🚀 V11.0 학습 - 10분 예측 (시퀀스 280분)")
print("="*70)
PRED_OFFSET = 10
SEQ_LEN = 280

def load_data(path_pattern):
    print(f"\n📂 데이터 로딩: {path_pattern}")
    files = glob.glob(path_pattern)
    if not files:
        print("⚠ 파일이 없습니다. 기본 파일명을 시도합니다.")
        files = glob.glob('M14_Q_20241201_202512010_학습.CSV')
        
    dfs = []
    for f in sorted(files):
        try:
            df = pd.read_csv(f, on_bad_lines='skip', encoding='utf-8')
            print(f"  {f}: {len(df):,}행")
            dfs.append(df)
        except UnicodeDecodeError:
            df = pd.read_csv(f, on_bad_lines='skip', encoding='cp949')
            print(f"  {f} (cp949): {len(df):,}행")
            dfs.append(df)
            
    if not dfs:
        raise ValueError("데이터 파일을 찾을 수 없습니다!")
        
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
    # 기본 변수
    df['Gap'] = df['M14.QUE.ALL.CURRENTQCREATED'] - df['M14.QUE.ALL.CURRENTQCOMPLETED']
    df['Transport'] = df['M14.QUE.ALL.TRANSPORT4MINOVERCNT']
    df['OHT'] = df['M14.QUE.OHT.OHTUTIL']
    
    # 1. 시차 변수 (Lags) - 280분까지 확장
    lags = [1, 3, 5, 10, 20, 50, 100, 200, 280]
    for lag in lags:
        df[f'Total_Lag_{lag}'] = df['TOTALCNT'].shift(lag)
        df[f'M14B_Lag_{lag}'] = df['M14AM14B'].shift(lag)
        df[f'Sum_Lag_{lag}'] = df['M14AM14BSUM'].shift(lag)
        df[f'Gap_Lag_{lag}'] = df['Gap'].shift(lag)
    # 2. 이동 평균/표준편차 (Rolling Stats) - 280분까지 확장
    windows = [10, 30, 60, 100, 200, 280]
    for w in windows:
        df[f'Total_Mean_{w}'] = df['TOTALCNT'].rolling(w).mean()
        df[f'Total_Std_{w}'] = df['TOTALCNT'].rolling(w).std()
        df[f'Total_Max_{w}'] = df['TOTALCNT'].rolling(w).max()
        df[f'Total_Min_{w}'] = df['TOTALCNT'].rolling(w).min()
        df[f'M14B_Mean_{w}'] = df['M14AM14B'].rolling(w).mean()
        df[f'Gap_Mean_{w}'] = df['Gap'].rolling(w).mean()
    # 3. 변화량 (Delta)
    df['Total_Delta_1'] = df['TOTALCNT'] - df['Total_Lag_1']
    df['Total_Delta_5'] = df['TOTALCNT'] - df['Total_Lag_5']
    df['Total_Delta_10'] = df['TOTALCNT'] - df['Total_Lag_10']
    df['Total_Delta_50'] = df['TOTALCNT'] - df['Total_Lag_50']
    df['Total_Delta_100'] = df['TOTALCNT'] - df['Total_Lag_100']
    df['Total_Delta_280'] = df['TOTALCNT'] - df['Total_Lag_280']
    
    df['M14B_Delta_10'] = df['M14AM14B'] - df['M14B_Lag_10']
    df['M14B_Delta_100'] = df['M14AM14B'] - df['M14B_Lag_100']
    df['Sum_Delta_10'] = df['M14AM14BSUM'] - df['Sum_Lag_10']
    df['Sum_Delta_100'] = df['M14AM14BSUM'] - df['Sum_Lag_100']
    
    # 4. 상호작용 (Interaction)
    df['Load_Ratio'] = df['M14AM14B'] / (df['TOTALCNT'] + 1)
    df['Gap_Load'] = df['Gap'] * df['M14AM14B']
    
    # 5. 패턴 (Explicit Patterns)
    df['Gold_Pattern'] = ((df['M14AM14B'] > 500) & (df['M14AM14BSUM'] > 600)).astype(int)
    df['Danger_Gap'] = (df['Gap'] > 250).astype(int)
    df['Danger_Total'] = (df['TOTALCNT'] >= 1600).astype(int)
    df['Triple_Check'] = ((df['M14AM14B'] > 450) & (df['M14AM14BSUM'] > 550) & (df['Gap'] > 200)).astype(int)
    
    # 6. 가속도 (Acceleration) 및 단기 변화
    df['Accel_1'] = df['Total_Delta_1'] - df['Total_Delta_1'].shift(1)
    df['Accel_3'] = df['Total_Delta_1'].diff(3)
    
    # 기울기 (Slope)
    df['Slope_3'] = (df['TOTALCNT'] - df['TOTALCNT'].shift(3)) / 3
    df['Slope_5'] = (df['TOTALCNT'] - df['TOTALCNT'].shift(5)) / 5
    df['Slope_30'] = (df['TOTALCNT'] - df['TOTALCNT'].shift(30)) / 30
    df['Slope_100'] = (df['TOTALCNT'] - df['TOTALCNT'].shift(100)) / 100
    df['Gap_Slope_3'] = (df['Gap'] - df['Gap'].shift(3)) / 3
    df['Gap_Slope_30'] = (df['Gap'] - df['Gap'].shift(30)) / 30
    # 타겟 생성 (10분 후 예측)
    df['Target_Total_Future'] = df['TOTALCNT'].shift(-PRED_OFFSET)
    df['Target_Delta'] = df['Target_Total_Future'] - df['TOTALCNT']
    
    # 결측 제거
    df_clean = df.dropna().reset_index(drop=True)
    return df_clean

# --- 메인 실행 ---
# 1. 데이터 로드
df = load_data('M14_Q_*.CSV')
# 2. 피처 생성
print("\n🔄 피처 생성 중 (280분 시퀀스)...")
df_features = create_vectorized_features(df)
print(f"✅ 생성 완료: {len(df_features):,}행, 컬럼 {df_features.shape[1]}개")
# 3. 학습 준비
exclude_cols = ['CURRTIME', 'Target_Total_Future', 'Target_Delta']
feature_cols = [c for c in df_features.columns if c not in exclude_cols]
X = df_features[feature_cols]
y_delta = df_features['Target_Delta']
y_total = df_features['Target_Total_Future']
print(f"📊 피처 수: {len(feature_cols)}개")
# 가중치: 고부하 20배, 중부하 5배
weights = np.ones(len(y_total))
weights[y_total >= 1500] = 5
weights[y_total >= 1700] = 20
# 모델 학습 (전체 데이터 사용)
print("\n🚀 모델 학습 시작 (GPU 가속, 반응성 강화)...")
# 원본 모델 (Base Model)
print("  메인 모델 학습...")
model_base = xgb.XGBRegressor(
    n_estimators=1500, # 학습량 증가
    learning_rate=0.02, # 정교하게
    max_depth=10, # 깊이 증가 (복잡한 패턴)
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1, # 민감하게
    reg_alpha=0.0, # 규제 완화
    reg_lambda=0.5, # 규제 완화
    n_jobs=4,
    device='cuda',
    tree_method='hist',
    random_state=42
)
model_base.fit(X, y_delta, sample_weight=weights, verbose=False)
# 저장
print("\n💾 모델 저장 중...")
with open('model_v11_280min.pkl', 'wb') as f:
    pickle.dump(model_base, f)
with open('features_v11_280min.pkl', 'wb') as f:
    pickle.dump(list(feature_cols), f)
print(f"✅ 학습 완료! (model_v11_280min.pkl)")
print(f"📊 시퀀스: {SEQ_LEN}분, 피처: {len(feature_cols)}개")