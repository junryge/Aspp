# -*- coding: utf-8 -*-
"""
================================================================================
V11 ML 예측 모델 - 학습 코드 (새 피처 추가)
M14 TOTALCNT 30분 내 리미트(1700) 초과 예측 시스템
================================================================================

V10 → V11 변경사항:
    - Important 그룹에 급증 신호 강한 5개 컬럼 추가
    - M14.QUE.ALL.M14ATOM16MANUAL_CURRENTQCNT (+212% 차이)
    - M14.QUE.STK.MANUALOUTCOUNT (+66% 차이)
    - M16HUB.QUE.M16TOM14A.MESCURRENTQCNT (+36%, 선행효과)
    - M14B.QUE.ALL.M14BTOM16CURRENTQCNT (+67% 차이)
    - M14.QUE.SENDFAB.VERTICALQUEUECOUNT (상관 0.842)

사용법:
    python m14_v11_train.py
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import xgboost as xgb
import lightgbm as lgb

from pyod.models.iforest import IForest
from pyod.models.copod import COPOD

print("=" * 80)
print("V11 ML 예측 모델 - 학습 시작 (새 피처 추가)")
print("=" * 80)

# ============================================================================
# 설정
# ============================================================================
CONFIG = {
    'train_file': 'M14_학습_20250909_20251231_C_109.CSV',  # 파일명 수정 필요
    'sequence_length': 240,
    'prediction_window': 30,
    'limit_value': 1700,
    'target_column': 'TOTALCNT',
    'xgb_params': {
        'n_estimators': 250,
        'max_depth': 8,
        'learning_rate': 0.03,
        'random_state': 42,
        'tree_method': 'hist',
        'device': 'cuda',
        'verbosity': 0
    },
    'lgb_params': {
        'n_estimators': 250,
        'max_depth': 8,
        'learning_rate': 0.03,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    },
    'surge_weight': 10,
    'model_dir': 'models',
    'log_dir': 'logs'
}

# Feature 그룹 정의 (V11 - 새 피처 추가)
FEATURE_GROUPS = {
    'target': [
        'TOTALCNT'
    ],
    'important': [
        # 기존 Load 관련
        'M14.QUE.LOAD.CURRENTLOADQCNT',
        'M14.QUE.LOAD.AVGLOADTIME',
        'M14.QUE.LOAD.AVGLOADTIME1MIN',
        'M14.QUE.LOAD.AVGFOUPLOADTIME',
        'M14.QUE.LOAD.AVGRETICLELOADTIME',
        'M14.QUE.LOAD.CURRENTRETICLELOADQCNT',
        # 기존 CMD 관련
        'M14.QUE.ALL.CURRENTQCOMPLETED',
        'M14.QUE.ALL.CURRENTQCREATED',
        'M14.QUE.ALL.TRANSPORT4MINOVERCNT',
        'M14.QUE.ALL.TRANSPORT4MINOVERTIMEAVG',
        'M14.QUE.ALL.TRANSPORT4MINOVERRATIO',
        # 기존 Hub 관련
        'M16HUB.QUE.M16TOM14B.CURRENTQCREATED',
        'M16HUB.QUE.M14BTOM16.CURRENTQCREATED',
        'M16HUB.QUE.M14TOM16.CURRENTQCREATED',
        'M16HUB.QUE.M16TOM14.CURRENTQCREATED',
        'M16HUB.QUE.M14TOM16.MESCURRENTQCNT',
        'M16HUB.QUE.M16TOM14.MESCURRENTQCNT',
        # ★ V11 신규 추가 (급증 신호 강한 컬럼)
        'M14.QUE.ALL.M14ATOM16MANUAL_CURRENTQCNT',  # +212% 급증 차이
        'M14.QUE.STK.MANUALOUTCOUNT',               # +66% 급증 차이
        'M16HUB.QUE.M16TOM14A.MESCURRENTQCNT',      # +36% 급증 차이, 선행효과
        'M14B.QUE.ALL.M14BTOM16CURRENTQCNT',        # +67% 급증 차이
        'M14.QUE.SENDFAB.VERTICALQUEUECOUNT',       # 상관관계 0.842
    ],
    'auxiliary': [
        # Inflow
        'M14AM10A', 'M10AM14A', 'M14AM10ASUM',
        'M14AM14B', 'M14BM14A', 'M14AM14BSUM',
        'M14AM16', 'M16M14A', 'M14AM16SUM',
        # Outflow
        'M14.QUE.SFAB.SENDQUEUETOTAL',
        'M14.QUE.SFAB.RECEIVEQUEUETOTAL',
        'M14.QUE.SFAB.RETURNQUEUETOTAL',
        'M14.QUE.SFAB.COMPLETEQUEUETOTAL',
        # OHT
        'M14.QUE.OHT.OHTUTIL',
        'M14.QUE.OHT.RTCOHTUTIL',
        'M14.QUE.OHT.CURRENTOHTQCNT',
        'M14.QUE.OHT.CURRENTRETICLEOHTQCNT',
        # CNV
        'M14.QUE.CNV.NORTHCURRENTQCNT',
        'M14.QUE.CNV.SOUTHCURRENTQCNT',
        'M14.QUE.CNV.ALLTONORTHCNVCURRENTQCNT',
        'M14.QUE.CNV.ALLTOSOUTHCNVCURRENTQCNT',
    ]
}

os.makedirs(CONFIG['model_dir'], exist_ok=True)
os.makedirs(CONFIG['log_dir'], exist_ok=True)

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

print(f"  - 원본 데이터: {len(df):,}행")

df['CURRTIME'] = pd.to_datetime(df['CURRTIME'], format='%Y%m%d%H%M')
df = df.sort_values('CURRTIME').reset_index(drop=True)

# Feature 그룹별로 실제 사용 가능한 것만 필터링
all_features = FEATURE_GROUPS['target'] + FEATURE_GROUPS['important'] + FEATURE_GROUPS['auxiliary']
available_features = [col for col in all_features if col in df.columns]
missing_features = [col for col in all_features if col not in df.columns]

if missing_features:
    print(f"  - 누락된 Feature: {missing_features}")

for group_name in FEATURE_GROUPS:
    FEATURE_GROUPS[group_name] = [f for f in FEATURE_GROUPS[group_name] if f in df.columns]

print(f"  - Target: {len(FEATURE_GROUPS['target'])}개")
print(f"  - Important: {len(FEATURE_GROUPS['important'])}개")
print(f"  - Auxiliary: {len(FEATURE_GROUPS['auxiliary'])}개")

# ============================================================================
# 결측치 처리
# ============================================================================
print("\n[2/6] 결측치 처리 중...")

all_feature_cols = FEATURE_GROUPS['target'] + FEATURE_GROUPS['important'] + FEATURE_GROUPS['auxiliary']
for col in all_feature_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

print(f"  - 결측치 처리 완료")

# ============================================================================
# 시퀀스 및 타겟 생성
# ============================================================================
print("\n[3/6] 시퀀스 및 타겟 생성 중...")

seq_len = CONFIG['sequence_length']
pred_window = CONFIG['prediction_window']
limit_val = CONFIG['limit_value']
target_col = CONFIG['target_column']

def create_sequence_features(df, feature_cols, seq_len, idx, limit_val=1700):
    """특정 인덱스에서 seq_len 만큼의 시퀀스 Feature 추출"""
    features = []
    for col in feature_cols:
        seq = df[col].iloc[idx - seq_len:idx].values
        current_val = seq[-1]

        # 기본 통계 Feature (10개)
        features.extend([
            np.mean(seq),
            np.std(seq),
            np.min(seq),
            np.max(seq),
            current_val,
            seq[-1] - seq[0],
            np.percentile(seq, 25),
            np.percentile(seq, 75),
            np.mean(seq[-10:]) - np.mean(seq[:10]),
            np.max(seq[-30:]) if len(seq) >= 30 else np.max(seq),
        ])

        # 급증 감지 Feature (10개)
        features.extend([
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

def create_target(df, target_col, pred_window, limit_val, idx):
    """30분 내 최대값과 돌파 여부 반환"""
    future_vals = df[target_col].iloc[idx:idx + pred_window].values
    if len(future_vals) == 0:
        return None, None
    max_val = np.max(future_vals)
    breach = 1 if max_val >= limit_val else 0
    return max_val, breach

print("  - 시퀀스 Feature 추출 중... (시간이 소요됩니다)")

X_target, X_important, X_auxiliary = [], [], []
y_regression, y_classification = [], []
current_values = []

valid_count = 0
breach_count = 0
surge_count = 0

for idx in range(seq_len, len(df) - pred_window):
    if idx % 10000 == 0:
        print(f"    진행률: {idx:,}/{len(df) - pred_window:,} ({100*idx/(len(df)-pred_window):.1f}%)")

    max_val, breach = create_target(df, target_col, pred_window, limit_val, idx)
    if max_val is None:
        continue

    current_val = df[target_col].iloc[idx]
    current_values.append(current_val)

    feat_target = create_sequence_features(df, FEATURE_GROUPS['target'], seq_len, idx)
    feat_important = create_sequence_features(df, FEATURE_GROUPS['important'], seq_len, idx)
    feat_auxiliary = create_sequence_features(df, FEATURE_GROUPS['auxiliary'], seq_len, idx)

    X_target.append(feat_target)
    X_important.append(feat_important)
    X_auxiliary.append(feat_auxiliary)
    y_regression.append(max_val)
    y_classification.append(breach)

    valid_count += 1
    if breach == 1:
        breach_count += 1
        if current_val < 1600:
            surge_count += 1

current_values = np.array(current_values)
X_target = np.array(X_target)
X_important = np.array(X_important)
X_auxiliary = np.array(X_auxiliary)
y_regression = np.array(y_regression)
y_classification = np.array(y_classification)

print(f"\n  - 총 샘플 수: {valid_count:,}")
print(f"  - 돌파 샘플 수: {breach_count:,} ({100*breach_count/valid_count:.2f}%)")
print(f"  - 급증 케이스: {surge_count:,}건")

# ============================================================================
# 스케일링
# ============================================================================
print("\n[4/6] 스케일링 중...")

scaler_target = StandardScaler()
scaler_important = StandardScaler()
scaler_auxiliary = StandardScaler()

X_target_scaled = scaler_target.fit_transform(X_target)
X_important_scaled = scaler_important.fit_transform(X_important)
X_auxiliary_scaled = scaler_auxiliary.fit_transform(X_auxiliary)

X_all = np.hstack([X_target_scaled, X_important_scaled, X_auxiliary_scaled])

print(f"  - 타겟 Feature: {X_target_scaled.shape}")
print(f"  - 중요 Feature: {X_important_scaled.shape}")
print(f"  - 보조 Feature: {X_auxiliary_scaled.shape}")
print(f"  - 전체 Feature: {X_all.shape}")

# ============================================================================
# PyOD 이상 탐지
# ============================================================================
print("\n[5/6] PyOD 이상 탐지 모델 학습 중...")

print("  - IsolationForest 학습 중...")
pyod_iforest = IForest(n_estimators=100, random_state=42, n_jobs=-1)
pyod_iforest.fit(X_target_scaled)

print("  - COPOD 학습 중...")
pyod_copod = COPOD()
pyod_copod.fit(X_target_scaled)

# ============================================================================
# 6개 모델 학습
# ============================================================================
print("\n[6/6] 6개 예측 모델 학습 중...")

sample_weights = np.ones(len(y_classification))
sample_weights[y_classification == 1] = CONFIG['surge_weight']

surge_mask = (current_values < 1600) & (y_classification == 1)
sample_weights[surge_mask] = 20
print(f"  - 급증 케이스 가중치 20배 적용: {np.sum(surge_mask):,}건")

X_train, X_val, y_reg_train, y_reg_val, y_cls_train, y_cls_val, w_train, w_val = train_test_split(
    X_all, y_regression, y_classification, sample_weights,
    test_size=0.2, random_state=42, stratify=y_classification
)

target_end = X_target_scaled.shape[1]
important_end = target_end + X_important_scaled.shape[1]
auxiliary_end = important_end + X_auxiliary_scaled.shape[1]

models = {}

# XGBoost 회귀 모델 3개
print("\n  [XGBoost 회귀 모델]")

print("    - XGB 타겟 모델 학습 중...")
xgb_target = xgb.XGBRegressor(**CONFIG['xgb_params'])
xgb_target.fit(X_train[:, :target_end], y_reg_train, sample_weight=w_train)
models['xgb_target'] = xgb_target

print("    - XGB 중요 모델 학습 중...")
xgb_important = xgb.XGBRegressor(**CONFIG['xgb_params'])
xgb_important.fit(X_train[:, target_end:important_end], y_reg_train, sample_weight=w_train)
models['xgb_important'] = xgb_important

print("    - XGB 보조 모델 학습 중...")
xgb_auxiliary = xgb.XGBRegressor(**CONFIG['xgb_params'])
xgb_auxiliary.fit(X_train[:, important_end:auxiliary_end], y_reg_train, sample_weight=w_train)
models['xgb_auxiliary'] = xgb_auxiliary

# LightGBM 분류 모델 3개
print("\n  [LightGBM 분류 모델]")

print("    - LGBM 타겟 모델 학습 중...")
lgb_target = lgb.LGBMClassifier(**CONFIG['lgb_params'])
lgb_target.fit(X_train[:, :target_end], y_cls_train, sample_weight=w_train)
models['lgb_target'] = lgb_target

print("    - LGBM 중요 모델 학습 중...")
lgb_important = lgb.LGBMClassifier(**CONFIG['lgb_params'])
lgb_important.fit(X_train[:, target_end:important_end], y_cls_train, sample_weight=w_train)
models['lgb_important'] = lgb_important

print("    - LGBM 보조 모델 학습 중...")
lgb_auxiliary = lgb.LGBMClassifier(**CONFIG['lgb_params'])
lgb_auxiliary.fit(X_train[:, important_end:auxiliary_end], y_cls_train, sample_weight=w_train)
models['lgb_auxiliary'] = lgb_auxiliary

# ============================================================================
# Validation 성능 평가
# ============================================================================
print("\n  [Validation 성능 평가]")

pred_xgb_target = models['xgb_target'].predict(X_val[:, :target_end])
pred_xgb_important = models['xgb_important'].predict(X_val[:, target_end:important_end])
pred_xgb_auxiliary = models['xgb_auxiliary'].predict(X_val[:, important_end:auxiliary_end])

pred_lgb_target = models['lgb_target'].predict(X_val[:, :target_end])
pred_lgb_important = models['lgb_important'].predict(X_val[:, target_end:important_end])
pred_lgb_auxiliary = models['lgb_auxiliary'].predict(X_val[:, important_end:auxiliary_end])

votes = np.vstack([
    (pred_xgb_target >= limit_val).astype(int),
    (pred_xgb_important >= limit_val).astype(int),
    (pred_xgb_auxiliary >= limit_val).astype(int),
    pred_lgb_target,
    pred_lgb_important,
    pred_lgb_auxiliary
])
vote_sum = votes.sum(axis=0)

val_pred = (vote_sum >= 4).astype(int)

acc = accuracy_score(y_cls_val, val_pred)
prec = precision_score(y_cls_val, val_pred, zero_division=0)
rec = recall_score(y_cls_val, val_pred, zero_division=0)
f1 = f1_score(y_cls_val, val_pred, zero_division=0)

print(f"    Accuracy:  {acc:.4f}")
print(f"    Precision: {prec:.4f}")
print(f"    Recall:    {rec:.4f}")
print(f"    F1 Score:  {f1:.4f}")

# ============================================================================
# 모델 저장
# ============================================================================
print("\n" + "=" * 80)
print("모델 저장")
print("=" * 80)

save_data = {
    'models': models,
    'pyod_iforest': pyod_iforest,
    'pyod_copod': pyod_copod,
    'scalers': {
        'target': scaler_target,
        'important': scaler_important,
        'auxiliary': scaler_auxiliary
    },
    'config': CONFIG,
    'feature_groups': FEATURE_GROUPS,
    'feature_indices': {
        'target_end': target_end,
        'important_end': important_end,
        'auxiliary_end': auxiliary_end
    },
    'training_info': {
        'version': 'V11',
        'total_samples': valid_count,
        'breach_samples': breach_count,
        'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'validation_accuracy': acc,
        'validation_recall': rec,
        'validation_f1': f1
    }
}

model_path = os.path.join(CONFIG['model_dir'], 'v11_m14_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(save_data, f)

print(f"  - 모델 저장 완료: {model_path}")

print("\n" + "=" * 80)
print("V11 학습 완료!")
print("=" * 80)