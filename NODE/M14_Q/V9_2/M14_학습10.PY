#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 1단계: 학습 전용 (train_v12.py) - 최종 수정
기능: UF09.CSV 파일의 실제 컬럼명을 그대로 사용하여 학습 (오류 해결)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import warnings

warnings.filterwarnings('ignore')

# ============================================
# ⚙️ 설정
# ============================================
TRAIN_FILE = "UF09.CSV"   # 학습 파일명
PRED_MINUTES = 10         # 10분 예지
THRESHOLD = 1700          # 위험 기준

print("="*70)
print(f"🚀 V12 학습 시작: {TRAIN_FILE}")
print("="*70)

def create_features(df):
    df = df.copy()
    
    # 시간 변환
    if 'CURRTIME' in df.columns:
        df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M', errors='coerce')

    # 1. 큐 갭 계산 (파일에 있는 실제 컬럼명 사용)
    # Q_CREATED - Q_COMPLETED
    df['Queue_Gap'] = df['M14.QUE.ALL.CURRENTQCREATED'] - df['M14.QUE.ALL.CURRENTQCOMPLETED']
    
    # 2. 거리 정보
    df['Dist_to_Limit'] = THRESHOLD - df['TOTALCNT']
    
    # 3. 속도 및 가속도 (M14AM14B 그대로 사용)
    for gap in [1, 3, 5, 10]:
        df[f'Vel_Total_{gap}m'] = df['TOTALCNT'].diff(gap)
        df[f'Vel_Gap_{gap}m'] = df['Queue_Gap'].diff(gap)
        df[f'Vel_M14B_{gap}m'] = df['M14AM14B'].diff(gap)
    
    df['Acc_Total'] = df['Vel_Total_3m'].diff(1)
    df['Vol_Total_10m'] = df['TOTALCNT'].rolling(10).std()
    
    # 4. 패턴 (M14AM14B, M14AM14BSUM 그대로 사용)
    df['Is_Triple_Danger'] = ((df['M14AM14B'] > 500) & (df['M14AM14BSUM'] > 600) & (df['Queue_Gap'] > 200)).astype(int)
    df['Is_Gold_Pattern'] = ((df['M14AM14B'] > 520) & (df['M14AM14BSUM'] > 600)).astype(int)
    
    return df

def train():
    print("🔄 데이터 로딩 중...")
    try:
        df = pd.read_csv(TRAIN_FILE, on_bad_lines='skip')
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {TRAIN_FILE}")
        return

    # Feature 생성
    print("🔄 특징(Feature) 생성 중...")
    try:
        df = create_features(df)
    except KeyError as e:
        print(f"❌ 컬럼 에러: {e} 가 파일에 없습니다. 파일 내용을 확인해주세요.")
        return

    # Target 생성
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=PRED_MINUTES)
    df['Future_Max'] = df['TOTALCNT'].rolling(window=indexer).max().shift(-1)
    df['Target'] = (df['Future_Max'] >= THRESHOLD).astype(int)
    
    # 학습에 사용할 Feature 목록 (실제 컬럼명 포함)
    features = [
        'TOTALCNT', 'M14AM14B', 'M14AM14BSUM', 'Queue_Gap', 'Dist_to_Limit',
        'Vel_Total_1m', 'Vel_Total_3m', 'Vel_Total_5m', 'Vel_Total_10m',
        'Vel_Gap_3m', 'Acc_Total', 'Vol_Total_10m', 
        'Is_Triple_Danger', 'Is_Gold_Pattern'
    ]
    
    df_train = df.dropna(subset=features + ['Target']).reset_index(drop=True)
    print(f"✅ 학습 데이터 확보: {len(df_train):,}행")
    
    # 가중치 계산
    neg, pos = np.bincount(df_train['Target'])
    scale_weight = neg / pos if pos > 0 else 1
    print(f"⚖️ 가중치 설정: {scale_weight:.2f}")
    
    print("🚀 XGBoost 모델 학습 중...")
    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.03,
        scale_pos_weight=scale_weight * 1.5, random_state=42, n_jobs=-1, tree_method='hist'
    )
    model.fit(df_train[features], df_train['Target'])
    
    # 저장
    with open('model_v12.pkl', 'wb') as f: pickle.dump(model, f)
    with open('features_v12.pkl', 'wb') as f: pickle.dump(features, f)
    
    print("\n✅ 학습 완료! (model_v12.pkl 저장됨)")
    print("   이제 eval_v12.py를 실행하세요.")

if __name__ == "__main__":
    train()
