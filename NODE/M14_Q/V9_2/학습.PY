#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 V13 학습: 값 예측(Regression) 전용
목표: 10분 뒤 '최대값(MAX)'이 정확히 몇이 될지 숫자로 예측
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import warnings

warnings.filterwarnings('ignore')

# ============================================
# ⚙️ 설정 (학습 파일)
# ============================================
TRAIN_FILE = "UF09.CSV"   # 3개월치 데이터
PRED_MINUTES = 10         # 10분 예지

# 파일에 있는 정확한 컬럼명 지정
COL_TOTAL = 'TOTALCNT'
COL_M14B  = 'M14AM14B'
COL_SUM   = 'M14AM14BSUM'
COL_Q_CR  = 'M14.QUE.ALL.CURRENTQCREATED'
COL_Q_CP  = 'M14.QUE.ALL.CURRENTQCOMPLETED'

print("="*70)
print(f"🚀 V13 숫자 예측 학습 시작: {TRAIN_FILE}")
print("="*70)

def create_features(df):
    df = df.copy()
    
    # 큐 갭
    df['Queue_Gap'] = df[COL_Q_CR] - df[COL_Q_CP]
    
    # 속도(Velocity) 변수 (매우 중요)
    for gap in [1, 3, 5, 10]:
        df[f'Vel_Total_{gap}m'] = df[COL_TOTAL].diff(gap)
        df[f'Vel_Gap_{gap}m'] = df['Queue_Gap'].diff(gap)
        df[f'Vel_M14B_{gap}m'] = df[COL_M14B].diff(gap)
    
    # 가속도 및 변동성
    df['Acc_Total'] = df['Vel_Total_3m'].diff(1)
    df['Vol_Total_10m'] = df[COL_TOTAL].rolling(10).std()
    
    # 패턴 점수화
    df['Triple_Score'] = (df[COL_M14B]/500) + (df[COL_SUM]/600) + (df['Queue_Gap']/200)
    
    return df

def train():
    print("🔄 데이터 로딩 중...")
    try:
        df = pd.read_csv(TRAIN_FILE, on_bad_lines='skip')
    except FileNotFoundError:
        print("❌ 파일을 찾을 수 없습니다.")
        return

    # 1. Feature 생성
    print("🔄 특징 추출 중...")
    df = create_features(df)
    
    # 2. Target 생성 (핵심!)
    # 목표: 10분 뒤 MAX값 - 현재값 = "상승폭(Delta)"을 예측
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=PRED_MINUTES)
    df['Future_Max'] = df[COL_TOTAL].rolling(window=indexer).max().shift(-1)
    
    # Target = 미래최대값 - 현재값 (얼마나 오를 것인가?)
    df['Target_Delta'] = df['Future_Max'] - df[COL_TOTAL]
    
    # 학습 변수
    features = [
        COL_TOTAL, COL_M14B, COL_SUM, 'Queue_Gap',
        'Vel_Total_1m', 'Vel_Total_3m', 'Vel_Total_5m', 'Vel_Total_10m',
        'Vel_Gap_3m', 'Acc_Total', 'Vol_Total_10m', 'Triple_Score'
    ]
    
    # 결측치 제거
    df_train = df.dropna(subset=features + ['Target_Delta']).reset_index(drop=True)
    print(f"✅ 학습 데이터: {len(df_train):,}개")
    
    # XGBoost 회귀(Regressor) 모델
    print("🚀 숫자 예측 모델 학습 중...")
    model = xgb.XGBRegressor(
        n_estimators=1000,       # 정밀한 값 예측을 위해 트리 많이
        max_depth=7,
        learning_rate=0.02,      # 천천히 꼼꼼하게 학습
        objective='reg:absoluteerror', # 오차 줄이기 (MAE)
        n_jobs=-1,
        random_state=42,
        tree_method='hist'
    )
    
    model.fit(df_train[features], df_train['Target_Delta'])
    
    # 저장
    with open('model_v13_reg.pkl', 'wb') as f: pickle.dump(model, f)
    with open('features_v13_reg.pkl', 'wb') as f: pickle.dump(features, f)
    
    print("\n✅ 학습 완료! (model_v13_reg.pkl)")

if __name__ == "__main__":
    train()
