#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 V12.2 학습 - 가중치 폭탄 (Weight Bomb) 적용
특징: 밸런싱된 데이터에 1700+ 구간 가중치 50배를 부여하여 고부하 예측 강제화
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import glob
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🚀 V12.2 학습 - 가중치 폭탄 (Over_1700 집중 타격)")
print("="*70)

PRED_OFFSET = 10
SEQ_LEN = 280

def get_load_category(cnt):
    """부하 구간 분류 (밸런싱용)"""
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
        print("⚠ 지정된 파일이 없어 기본 패턴(M14_Q_*.CSV)을 찾습니다.")
        files = glob.glob('M14_Q_*.CSV')
    
    dfs = []
    for f in sorted(files):
        try: dfs.append(pd.read_csv(f, on_bad_lines='skip', encoding='utf-8'))
        except: dfs.append(pd.read_csv(f, on_bad_lines='skip', encoding='cp949'))
    
    if not dfs: raise ValueError("❌ 학습할 데이터 파일이 없습니다!")
    return pd.concat(dfs, ignore_index=True)

def create_vectorized_features(df):
    """피처 생성 (280분 시퀀스 & 기울기 강화)"""
    df = df.copy()
    
    # 숫자형 변환
    cols = ['TOTALCNT', 'M14AM14B', 'M14AM14BSUM', 'M10AM14A', 
            'M14.QUE.ALL.TRANSPORT4MINOVERCNT', 'M14.QUE.OHT.OHTUTIL',
            'M14.QUE.ALL.CURRENTQCREATED', 'M14.QUE.ALL.CURRENTQCOMPLETED']
    for c in cols: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    
    df['Gap'] = df['M14.QUE.ALL.CURRENTQCREATED'] - df['M14.QUE.ALL.CURRENTQCOMPLETED']
    
    # 1. Lag (시차)
    lags = [1, 3, 5, 10, 20, 50, 100, 200, 280]
    for lag in lags:
        df[f'Total_Lag_{lag}'] = df['TOTALCNT'].shift(lag)
        df[f'M14B_Lag_{lag}'] = df['M14AM14B'].shift(lag)
        df[f'Sum_Lag_{lag}'] = df['M14AM14BSUM'].shift(lag)
        df[f'Gap_Lag_{lag}'] = df['Gap'].shift(lag)
    
    # 2. Rolling (이동평균)
    windows = [10, 30, 60, 100, 200, 280]
    for w in windows:
        df[f'Total_Mean_{w}'] = df['TOTALCNT'].rolling(w).mean()
        df[f'Total_Std_{w}'] = df['TOTALCNT'].rolling(w).std()
        df[f'M14B_Mean_{w}'] = df['M14AM14B'].rolling(w).mean()
        df[f'Gap_Mean_{w}'] = df['Gap'].rolling(w).mean()

    # 3. Slope (급상승 감지용 기울기)
    df['Slope_1'] = df['TOTALCNT'] - df['TOTALCNT'].shift(1)
    df['Slope_3'] = df['TOTALCNT'] - df['TOTALCNT'].shift(3)
    df['Slope_10'] = df['TOTALCNT'] - df['TOTALCNT'].shift(10)
    
    # 타겟 생성 (10분 뒤 예측)
    df['Target_Total_Future'] = df['TOTALCNT'].shift(-PRED_OFFSET)
    df['Target_Delta'] = df['Target_Total_Future'] - df['TOTALCNT']
    
    # 결측 제거
    return df.dropna().reset_index(drop=True)

# --- 메인 실행 ---
# 1. 데이터 로드 및 전처리
df = load_data('M14_Q_*.CSV') 
print("\n🔄 피처 생성 중...")
df_features = create_vectorized_features(df)

# 2. 데이터 밸런싱 (Under-sampling)
print("\n⚖️ 데이터 밸런싱 작업...")
# 타겟값 기준으로 카테고리 분류
df_features['Load_Category'] = df_features['Target_Total_Future'].apply(get_load_category)

# 최소 샘플 수 계산 (너무 적으면 5000개로 고정 시도)
counts = df_features['Load_Category'].value_counts()
min_samples = counts.min()
target_samples = max(min_samples, 5000) 

print(f"  👉 클래스별 목표 샘플 수: {target_samples} (최소 클래스: {min_samples})")

df_balanced = df_features.groupby('Load_Category', group_keys=False).apply(
    lambda x: x.sample(n=min(len(x), target_samples), random_state=42)
)
# 순서 섞기
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# 3. [핵심] 가중치(Sample Weight) 생성 - 금융 치료 적용
# 기본 가중치 1.0
weights = np.ones(len(df_balanced))
y_future = df_balanced['Target_Total_Future'].values

# 구간별 가중치 할증
weights[(y_future >= 1500) & (y_future < 1600)] = 2.0   # 주의 구간: 2배
weights[(y_future >= 1600) & (y_future < 1700)] = 10.0  # 위험 구간: 10배
weights[y_future >= 1700] = 50.0                        # 🚨 고부하 구간: 50배 (강력 처방)

print(f"🔥 가중치 적용 완료: 1700+ 샘플 {sum(y_future >= 1700)}개에 50배 가중치 부여")

# 4. 학습 준비
exclude_cols = ['CURRTIME', 'Target_Total_Future', 'Target_Delta', 'Load_Category']
feature_cols = [c for c in df_balanced.columns if c not in exclude_cols]

X = df_balanced[feature_cols]
y = df_balanced['Target_Delta']

print(f"\n🚀 모델 학습 시작 (XGBoost, 가중치 적용)...")
model = xgb.XGBRegressor(
    n_estimators=3000,      # 충분한 학습량
    learning_rate=0.01,     # 정교한 학습
    max_depth=10,           # 복잡한 패턴 인식
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,     # 소수 데이터 무시 방지
    n_jobs=4,
    device='cuda',          # GPU 사용 (없으면 'cpu'로 변경)
    tree_method='hist',
    random_state=42
)

# 학습 시 weights 전달
model.fit(X, y, sample_weight=weights, verbose=False)

# 5. 저장
print("\n💾 모델 저장 중...")
with open('model_v12_weighted.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('features_v12_weighted.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)
    
print("✅ 학습 완료! (model_v12_weighted.pkl)")
