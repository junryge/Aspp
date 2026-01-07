# -*- coding: utf-8 -*-
"""
================================================================================
V13 ML 예측 모델 - 학습 코드
N2_RATIO + CNV 2개 신규 컬럼 추가
================================================================================
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')

# ============================================================================
# 설정
# ============================================================================
CONFIG = {
    'train_file': 'M14_학습_V13_20250909_20251231.CSV',  # V13 병합 데이터
    'model_file': 'models/v13_m14_model.pkl',
    'sequence_length': 240,
    'prediction_window': 30,
    'limit_value': 1700,
    'target_column': 'TOTALCNT',
}

# V13: 기존 V10 + 신규 2개 컬럼
FEATURE_GROUPS = {
    'target': ['TOTALCNT'],
    'important': [
        'M14.QUE.LOAD.CURRENTLOADQCNT',
        'M14.QUE.LOAD.AVGLOADTIME',
        'M14.QUE.LOAD.AVGLOADTIME1MIN',
        'M14.QUE.LOAD.AVGFOUPLOADTIME',
        'M14.QUE.LOAD.AVGRETICLELOADTIME',
        'M14.QUE.LOAD.CURRENTRETICLELOADQCNT',
        'M14.QUE.ALL.CURRENTQCOMPLETED',
        'M14.QUE.ALL.CURRENTQCREATED',
        'M14.QUE.ALL.TRANSPORT4MINOVERCNT',
        'M14.QUE.ALL.TRANSPORT4MINOVERTIMEAVG',
        'M14.QUE.ALL.TRANSPORT4MINOVERRATIO',
        'M16HUB.QUE.M16TOM14B.CURRENTQCREATED',
        'M16HUB.QUE.M14BTOM16.CURRENTQCREATED',
        'M16HUB.QUE.M14TOM16.CURRENTQCREATED',
        'M16HUB.QUE.M16TOM14.CURRENTQCREATED',
        'M16HUB.QUE.M14TOM16.MESCURRENTQCNT',
        'M16HUB.QUE.M16TOM14.MESCURRENTQCNT',
    ],
    'auxiliary': [
        'M14AM10A', 'M10AM14A', 'M14AM10ASUM',
        'M14AM14B', 'M14BM14A', 'M14AM14BSUM',
        'M14AM16', 'M16M14A', 'M14AM16SUM',
        'M14.QUE.SFAB.SENDQUEUETOTAL',
        'M14.QUE.SFAB.RECEIVEQUEUETOTAL',
        'M14.QUE.SFAB.RETURNQUEUETOTAL',
        'M14.QUE.SFAB.COMPLETEQUEUETOTAL',
        'M14.QUE.OHT.OHTUTIL',
        'M14.QUE.OHT.RTCOHTUTIL',
        'M14.QUE.OHT.CURRENTOHTQCNT',
        'M14.QUE.OHT.CURRENTRETICLEOHTQCNT',
        'M14.QUE.CNV.NORTHCURRENTQCNT',
        'M14.QUE.CNV.SOUTHCURRENTQCNT',
        'M14.QUE.CNV.ALLTONORTHCNVCURRENTQCNT',
        'M14.QUE.CNV.ALLTOSOUTHCNVCURRENTQCNT',
    ],
    # ★ V13 신규 그룹
    'v13_new': [
        'M14.STRATE.N2.STORAGERATIO',                        # N2 저장율
        'M14.PDT.LAYOUT.M14A_M14ATOM14ACNV_CURRENTQCNT',     # CNV 큐 (선행지표)
    ]
}

print("=" * 70)
print("V13 ML 예측 모델 - 학습 시작 (N2_RATIO + CNV 추가)")
print("=" * 70)

# ============================================================================
# 데이터 로드
# ============================================================================
print("\n[1/6] 데이터 로드 중...")

try:
    df = pd.read_csv(CONFIG['train_file'], encoding='utf-8')
except:
    try:
        df = pd.read_csv(CONFIG['train_file'], encoding='cp949')
    except:
        df = pd.read_csv(CONFIG['train_file'], encoding='euc-kr')

df['CURRTIME'] = pd.to_datetime(df['CURRTIME'], format='%Y%m%d%H%M')
df = df.sort_values('CURRTIME').reset_index(drop=True)

print(f"  - 데이터: {len(df):,}행")

# 컬럼 필터링
for group_name in FEATURE_GROUPS:
    FEATURE_GROUPS[group_name] = [f for f in FEATURE_GROUPS[group_name] if f in df.columns]
    print(f"  - {group_name} 컬럼: {len(FEATURE_GROUPS[group_name])}개")

# 수치형 변환
all_cols = []
for group in FEATURE_GROUPS.values():
    all_cols.extend(group)
all_cols = list(set(all_cols))

for col in all_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# ============================================================================
# 시퀀스 피처 생성 함수
# ============================================================================
def create_sequence_features(df, feature_cols, seq_len, idx, limit_val=1700):
    features = []
    for col in feature_cols:
        seq = df[col].iloc[idx - seq_len:idx].values
        current_val = seq[-1]
        features.extend([
            np.mean(seq), np.std(seq), np.min(seq), np.max(seq), current_val,
            seq[-1] - seq[0],
            np.percentile(seq, 25), np.percentile(seq, 75),
            np.mean(seq[-10:]) - np.mean(seq[:10]),
            np.max(seq[-30:]) if len(seq) >= 30 else np.max(seq),
            seq[-1] - seq[-10] if len(seq) >= 10 else 0,
            seq[-1] - seq[-30] if len(seq) >= 30 else 0,
            seq[-1] - seq[-60] if len(seq) >= 60 else 0,
            (seq[-1] - seq[-10]) / 10 if len(seq) >= 10 else 0,
            (seq[-1] - seq[-30]) / 30 if len(seq) >= 30 else 0,
            np.max(seq[-10:]) - np.min(seq[-10:]) if len(seq) >= 10 else 0,
            np.max(seq[-30:]) - np.min(seq[-30:]) if len(seq) >= 30 else 0,
            limit_val - current_val,
            1 if current_val >= 1500 else 0,
            1 if current_val >= 1600 else 0,
        ])
    return features

# ============================================================================
# 학습 데이터 생성
# ============================================================================
print("\n[2/6] 학습 데이터 생성 중...")

seq_len = CONFIG['sequence_length']
pred_window = CONFIG['prediction_window']
limit_val = CONFIG['limit_value']
target_col = CONFIG['target_column']

X_target, X_important, X_auxiliary, X_v13_new = [], [], [], []
y_reg, y_clf = [], []

for idx in range(seq_len, len(df) - pred_window):
    if idx % 10000 == 0:
        print(f"    진행: {idx:,}/{len(df) - pred_window:,}")
    
    # 타겟 (30분 후 최대값)
    future_max = df[target_col].iloc[idx:idx + pred_window].max()
    y_reg.append(future_max)
    y_clf.append(1 if future_max >= limit_val else 0)
    
    # 피처 생성
    X_target.append(create_sequence_features(df, FEATURE_GROUPS['target'], seq_len, idx))
    X_important.append(create_sequence_features(df, FEATURE_GROUPS['important'], seq_len, idx))
    X_auxiliary.append(create_sequence_features(df, FEATURE_GROUPS['auxiliary'], seq_len, idx))
    X_v13_new.append(create_sequence_features(df, FEATURE_GROUPS['v13_new'], seq_len, idx))

X_target = np.array(X_target)
X_important = np.array(X_important)
X_auxiliary = np.array(X_auxiliary)
X_v13_new = np.array(X_v13_new)
y_reg = np.array(y_reg)
y_clf = np.array(y_clf)

print(f"  - 샘플 수: {len(y_reg):,}")
print(f"  - 돌파 케이스: {sum(y_clf):,} ({100*sum(y_clf)/len(y_clf):.1f}%)")

# ============================================================================
# 스케일링
# ============================================================================
print("\n[3/6] 스케일링...")

scalers = {
    'target': StandardScaler(),
    'important': StandardScaler(),
    'auxiliary': StandardScaler(),
    'v13_new': StandardScaler(),
}

X_target_scaled = scalers['target'].fit_transform(X_target)
X_important_scaled = scalers['important'].fit_transform(X_important)
X_auxiliary_scaled = scalers['auxiliary'].fit_transform(X_auxiliary)
X_v13_new_scaled = scalers['v13_new'].fit_transform(X_v13_new)

# ============================================================================
# 모델 학습
# ============================================================================
print("\n[4/6] 모델 학습 중...")

models = {}

# XGBoost 회귀 모델
print("  - XGB Target...")
models['xgb_target'] = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1)
models['xgb_target'].fit(X_target_scaled, y_reg)

print("  - XGB Important...")
models['xgb_important'] = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1)
models['xgb_important'].fit(X_important_scaled, y_reg)

print("  - XGB Auxiliary...")
models['xgb_auxiliary'] = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1)
models['xgb_auxiliary'].fit(X_auxiliary_scaled, y_reg)

print("  - XGB V13 New...")
models['xgb_v13_new'] = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1)
models['xgb_v13_new'].fit(X_v13_new_scaled, y_reg)

# LightGBM 분류 모델
print("  - LGBM Target...")
models['lgb_target'] = LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1)
models['lgb_target'].fit(X_target_scaled, y_clf)

print("  - LGBM Important...")
models['lgb_important'] = LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1)
models['lgb_important'].fit(X_important_scaled, y_clf)

print("  - LGBM Auxiliary...")
models['lgb_auxiliary'] = LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1)
models['lgb_auxiliary'].fit(X_auxiliary_scaled, y_clf)

print("  - LGBM V13 New...")
models['lgb_v13_new'] = LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1)
models['lgb_v13_new'].fit(X_v13_new_scaled, y_clf)

# ============================================================================
# 피처 인덱스 저장
# ============================================================================
feature_indices = {
    'target_end': X_target.shape[1],
    'important_end': X_target.shape[1] + X_important.shape[1],
    'auxiliary_end': X_target.shape[1] + X_important.shape[1] + X_auxiliary.shape[1],
    'v13_new_end': X_target.shape[1] + X_important.shape[1] + X_auxiliary.shape[1] + X_v13_new.shape[1],
}

# ============================================================================
# 모델 저장
# ============================================================================
print("\n[5/6] 모델 저장...")

os.makedirs('models', exist_ok=True)

model_data = {
    'models': models,
    'scalers': scalers,
    'feature_indices': feature_indices,
    'config': CONFIG,
    'feature_groups': FEATURE_GROUPS,
    'version': 'V13',
    'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
}

with open(CONFIG['model_file'], 'wb') as f:
    pickle.dump(model_data, f)

print(f"  - 저장: {CONFIG['model_file']}")

# ============================================================================
# 완료
# ============================================================================
print("\n[6/6] 학습 완료!")
print("=" * 70)
print(f"V13 모델 학습 완료!")
print(f"  - 신규 컬럼: N2_RATIO, CNV")
print(f"  - 모델 수: {len(models)}개")
print("=" * 70)