#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 1단계: 학습 전용 (train_v12.py) - 수정버전
기능: 컬럼명 오류 해결 (M14AM14B -> M14B 자동 변환)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import warnings

warnings.filterwarnings('ignore')

# ============================================
# ⚙️ 설정 (학습 파일 지정)
# ============================================
TRAIN_FILE = "UF09.CSV"   # 3개월치 데이터 파일명
PRED_MINUTES = 10         # 10분 후 예측
THRESHOLD = 1700          # 위험 기준

print("="*70)
print(f"🚀 V12 학습 시작: {TRAIN_FILE}")
print("="*70)

# --------------------------------------------
# 1. 컬럼명 표준화 함수 (KeyError 방지)
# --------------------------------------------
def standardize_columns(df):
    """
    CSV 파일의 긴 컬럼명을 코드가 사용하는 짧은 이름으로 통일합니다.
    """
    # (표준이름 : [CSV에서 가능한 이름 후보들])
    mapping = {
        'TOTALCNT': ['TOTALCNT'],
        'M14B': ['M14AM14B', 'M14B'],
        'M14BSUM': ['M14AM14BSUM', 'M14BSUM'],
        'Q_CREATED': ['M14.QUE.ALL.CURRENTQCREATED', 'Q_CREATED'],
        'Q_COMPLETED': ['M14.QUE.ALL.CURRENTQCOMPLETED', 'Q_COMPLETED']
    }
    
    rename_dict = {}
    for std_name, candidates in mapping.items():
        for col in candidates:
            if col in df.columns:
                rename_dict[col] = std_name
                break
    
    df = df.rename(columns=rename_dict)
    return df

# --------------------------------------------
# 2. Feature 생성 (표준화된 이름 M14B 사용)
# --------------------------------------------
def create_features(df):
    df = df.copy()
    if 'CURRTIME' in df.columns:
        df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M', errors='coerce')

    # 표준화된 이름을 사용하므로 안전함
    df['Queue_Gap'] = df['Q_CREATED'] - df['Q_COMPLETED']
    df['Dist_to_Limit'] = THRESHOLD - df['TOTALCNT']
    
    # 속도 및 패턴
    for gap in [1, 3, 5, 10]:
        df[f'Vel_Total_{gap}m'] = df['TOTALCNT'].diff(gap)
        df[f'Vel_Gap_{gap}m'] = df['Queue_Gap'].diff(gap)
        # 여기서 M14AM14B 대신 M14B 사용
        df[f'Vel_M14B_{gap}m'] = df['M14B'].diff(gap)
    
    df['Acc_Total'] = df['Vel_Total_3m'].diff(1)
    df['Vol_Total_10m'] = df['TOTALCNT'].rolling(10).std()
    
    # 패턴 (M14B, M14BSUM 사용)
    df['Is_Triple_Danger'] = ((df['M14B'] > 500) & (df['M14BSUM'] > 600) & (df['Queue_Gap'] > 200)).astype(int)
    df['Is_Gold_Pattern'] = ((df['M14B'] > 520) & (df['M14BSUM'] > 600)).astype(int)
    
    return df

# --------------------------------------------
# 3. 학습 실행
# --------------------------------------------
def train():
    print("🔄 데이터 로딩 중...")
    try:
        df = pd.read_csv(TRAIN_FILE, on_bad_lines='skip')
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {TRAIN_FILE}")
        return

    # 1. 컬럼 이름부터 통일 (여기서 에러 해결!)
    df = standardize_columns(df)
    
    # 필수 컬럼 확인
    required = ['TOTALCNT', 'M14B', 'M14BSUM', 'Q_CREATED', 'Q_COMPLETED']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"❌ 필수 컬럼 누락: {missing}")
        print("   CSV 파일의 컬럼명을 확인해주세요.")
        return

    # 2. Feature 생성
    df = create_features(df)
    
    # 3. Target 생성
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=PRED_MINUTES)
    df['Future_Max'] = df['TOTALCNT'].rolling(window=indexer).max().shift(-1)
    df['Target'] = (df['Future_Max'] >= THRESHOLD).astype(int)
    
    # 4. 학습할 Feature 선정
    features = [
        'TOTALCNT', 'M14B', 'M14BSUM', 'Queue_Gap', 'Dist_to_Limit',
        'Vel_Total_1m', 'Vel_Total_3m', 'Vel_Total_5m', 'Vel_Total_10m',
        'Vel_Gap_3m', 'Acc_Total', 'Vol_Total_10m', 
        'Is_Triple_Danger', 'Is_Gold_Pattern'
    ]
    
    df_train = df.dropna(subset=features + ['Target']).reset_index(drop=True)
    
    print(f"✅ 학습 데이터 확보: {len(df_train):,}행")
    
    # 가중치 계산
    neg, pos = np.bincount(df_train['Target'])
    if pos == 0:
        print("⚠️ 주의: 학습 데이터에 위험(1700+) 구간이 없습니다. 모델이 제대로 학습되지 않을 수 있습니다.")
        scale_weight = 1
    else:
        scale_weight = neg / pos
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
