#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 1단계: 학습 전용 (train_v12.py)
기능: 대량의 데이터를 읽어 XGBoost 모델을 학습하고 저장합니다.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import warnings

warnings.filterwarnings('ignore')

# ============================================
# ⚙️ 설정 (학습용 데이터 지정)
# ============================================
TRAIN_FILE = "UF09.CSV"   # 3개월치 데이터 파일명 (학습용)
PRED_MINUTES = 10         # 10분 후 예측
THRESHOLD = 1700          # 위험 기준값

print("="*70)
print(f"🚀 V12 학습 모드: {TRAIN_FILE}")
print("="*70)

# --------------------------------------------
# 1. Feature Engineering (평가 코드와 동일해야 함!)
# --------------------------------------------
def create_features(df):
    df = df.copy()
    
    # 시간 변환
    if 'CURRTIME' in df.columns:
        df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M', errors='coerce')

    # 컬럼 매핑 (파일마다 이름이 다를 수 있음)
    q_created = df['M14.QUE.ALL.CURRENTQCREATED'] if 'M14.QUE.ALL.CURRENTQCREATED' in df.columns else df['Q_CREATED']
    q_completed = df['M14.QUE.ALL.CURRENTQCOMPLETED'] if 'M14.QUE.ALL.CURRENTQCOMPLETED' in df.columns else df['Q_COMPLETED']
    
    df['Queue_Gap'] = q_created - q_completed
    df['Dist_to_Limit'] = THRESHOLD - df['TOTALCNT']
    
    # 속도(Velocity) 및 가속도
    for gap in [1, 3, 5, 10]:
        df[f'Vel_Total_{gap}m'] = df['TOTALCNT'].diff(gap)
        df[f'Vel_Gap_{gap}m'] = df['Queue_Gap'].diff(gap)
        df[f'Vel_M14B_{gap}m'] = df['M14AM14B'].diff(gap)
    
    df['Acc_Total'] = df['Vel_Total_3m'].diff(1)
    df['Vol_Total_10m'] = df['TOTALCNT'].rolling(10).std()
    
    # 위험 패턴
    df['Is_Triple_Danger'] = ((df['M14AM14B'] > 500) & (df['M14AM14BSUM'] > 600) & (df['Queue_Gap'] > 200)).astype(int)
    df['Is_Gold_Pattern'] = ((df['M14AM14B'] > 520) & (df['M14AM14BSUM'] > 600)).astype(int)
    
    return df

# --------------------------------------------
# 2. 데이터 준비 및 학습
# --------------------------------------------
def train():
    print("🔄 데이터 로딩 중...")
    df = pd.read_csv(TRAIN_FILE, on_bad_lines='skip')
    
    # 컬럼 이름 통일
    rename_dict = {}
    cols_map = {'TOTALCNT': 'TOTALCNT', 'M14B': 'M14AM14B', 'M14BSUM': 'M14AM14BSUM'}
    for k, v in cols_map.items():
        if v in df.columns: rename_dict[v] = k
    df = df.rename(columns=rename_dict)
    
    # 변수 생성
    df = create_features(df)
    
    # Target 생성 (미래 10분 MAX >= 1700)
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=PRED_MINUTES)
    df['Future_Max'] = df['TOTALCNT'].rolling(window=indexer).max().shift(-1)
    df['Target'] = (df['Future_Max'] >= THRESHOLD).astype(int)
    
    # 학습에 쓸 변수 지정
    features = [
        'TOTALCNT', 'M14B', 'M14BSUM', 'Queue_Gap', 'Dist_to_Limit',
        'Vel_Total_1m', 'Vel_Total_3m', 'Vel_Total_5m', 'Vel_Total_10m',
        'Vel_Gap_3m', 'Acc_Total', 'Vol_Total_10m', 
        'Is_Triple_Danger', 'Is_Gold_Pattern'
    ]
    
    # 결측치 제거
    df_train = df.dropna(subset=features + ['Target']).reset_index(drop=True)
    
    print(f"✅ 학습 데이터: {len(df_train):,}행")
    
    # 가중치 계산 (불균형 데이터 해결)
    neg, pos = np.bincount(df_train['Target'])
    scale_weight = neg / pos if pos > 0 else 1
    print(f"⚖️ 가중치(scale_pos_weight): {scale_weight:.2f}")
    
    # XGBoost 학습
    print("🚀 학습 시작...")
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        scale_pos_weight=scale_weight * 1.5, # Recall 강화
        random_state=42,
        n_jobs=-1,
        tree_method='hist'
    )
    
    model.fit(df_train[features], df_train['Target'])
    
    # 저장
    with open('model_v12.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('features_v12.pkl', 'wb') as f:
        pickle.dump(features, f)
        
    print("\n💾 저장 완료:")
    print("  1. 모델: model_v12.pkl")
    print("  2. 변수목록: features_v12.pkl")
    print("✅ 학습 끝! 이제 eval_v12.py를 실행하세요.")

if __name__ == "__main__":
    train()
