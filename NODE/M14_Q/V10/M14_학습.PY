# -*- coding: utf-8 -*-
"""
================================================================================
V10 ML 예측 모델 - 학습 코드 (GPU 버전)
M14 Q 30분 내 리미트(1700) 초과 예측 시스템
================================================================================

사용법:
    python m14_v10_train.py

출력:
    - models/ 폴더에 학습된 모델 저장
    - logs/ 폴더에 학습 로그 저장

필수 패키지:
    pip install xgboost lightgbm  (GPU 버전)
    # CUDA 설치 필요
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# 경고 무시
warnings.filterwarnings('ignore')

# scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# XGBoost & LightGBM
import xgboost as xgb
import lightgbm as lgb

# PyOD (이상 탐지)
from pyod.models.iforest import IForest
from pyod.models.copod import COPOD

print("=" * 80)
print("V10 ML 예측 모델 - 학습 시작 (GPU 모드)")
print("=" * 80)

# GPU 사용 가능 여부 확인
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        print(f"  GPU 감지: {torch.cuda.get_device_name(0)}")
except:
    GPU_AVAILABLE = True  # XGBoost/LightGBM은 CUDA 직접 사용
    print("  GPU 모드로 학습 시도...")

# ============================================================================
# 설정
# ============================================================================
CONFIG = {
    # 파일 경로
    'train_file': 'M14_학습_20250909_20251231_C_109.CSV',

    # 시퀀스 설정
    'sequence_length': 240,  # 240분 (4시간)
    'prediction_window': 30,  # 30분 예측
    'limit_value': 1700,      # 리미트 값

    # 타겟 컬럼
    'target_column': 'TOTALCNT',  # TOTALCNT

    # 모델 파라미터 (GPU)
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

    # 급증 케이스 가중치
    'surge_weight': 10,

    # 출력 폴더
    'model_dir': 'models',
    'log_dir': 'logs'
}

# Feature 그룹 정의
FEATURE_GROUPS = {
    'target': [
        'TOTALCNT'  # TOTALCNT (타겟)
    ],
    'important': [
        # Storage 관련
        'M14.QUE.LOAD.CURRENTLOADQCNT',
        'M14.QUE.LOAD.AVGLOADTIME',
        'M14.QUE.LOAD.AVGLOADTIME1MIN',
        'M14.QUE.LOAD.AVGFOUPLOADTIME',
        'M14.QUE.LOAD.AVGRETICLELOADTIME',
        'M14.QUE.LOAD.CURRENTRETICLELOADQCNT',
        # CMD 관련
        'M14.QUE.ALL.CURRENTQCOMPLETED',
        'M14.QUE.ALL.CURRENTQCREATED',
        'M14.QUE.ALL.TRANSPORT4MINOVERCNT',
        'M14.QUE.ALL.TRANSPORT4MINOVERTIMEAVG',
        'M14.QUE.ALL.TRANSPORT4MINOVERRATIO',
        # Hub 관련
        'M16HUB.QUE.M16TOM14B.CURRENTQCREATED',
        'M16HUB.QUE.M14BTOM16.CURRENTQCREATED',
        'M16HUB.QUE.M14TOM16.CURRENTQCREATED',
        'M16HUB.QUE.M16TOM14.CURRENTQCREATED',
        'M16HUB.QUE.M14TOM16.MESCURRENTQCNT',
        'M16HUB.QUE.M16TOM14.MESCURRENTQCNT',
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
        # Maxcapa
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

# ============================================================================
# 폴더 생성
# ============================================================================
os.makedirs(CONFIG['model_dir'], exist_ok=True)
os.makedirs(CONFIG['log_dir'], exist_ok=True)

# ============================================================================
# 데이터 로드
# ============================================================================
print("\n[1/6] 데이터 로드 중...")

df = pd.read_csv(CONFIG['train_file'], encoding='utf-8')
print(f"  - 원본 데이터: {len(df):,}행")

# CURRTIME을 datetime으로 변환
df['CURRTIME'] = pd.to_datetime(df['CURRTIME'], format='%Y%m%d%H%M')
df = df.sort_values('CURRTIME').reset_index(drop=True)

# 사용할 컬럼 확인
all_features = FEATURE_GROUPS['target'] + FEATURE_GROUPS['important'] + FEATURE_GROUPS['auxiliary']
available_features = [col for col in all_features if col in df.columns]
missing_features = [col for col in all_features if col not in df.columns]

if missing_features:
    print(f"  - 누락된 Feature: {missing_features}")

print(f"  - 사용 가능한 Feature: {len(available_features)}개")

# Feature 그룹별로 실제 사용 가능한 것만 필터링
for group_name in FEATURE_GROUPS:
    FEATURE_GROUPS[group_name] = [f for f in FEATURE_GROUPS[group_name] if f in df.columns]

# ============================================================================
# 결측치 처리
# ============================================================================
print("\n[2/6] 결측치 처리 중...")

# 숫자형 컬럼만 선택하고 모든 Feature 컬럼을 숫자로 변환
all_feature_cols = FEATURE_GROUPS['target'] + FEATURE_GROUPS['important'] + FEATURE_GROUPS['auxiliary']
for col in all_feature_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[numeric_cols] = df[numeric_cols].fillna(0)

print(f"  - 결측치 처리 완료")

# ============================================================================
# 시퀀스 및 타겟 생성
# ============================================================================
print("\n[3/6] 시퀀스 및 타겟 생성 중...")

seq_len = CONFIG['sequence_length']
pred_window = CONFIG['prediction_window']
limit_val = CONFIG['limit_value']
target_col = CONFIG['target_column']

# 시퀀스 Feature 생성 함수 (급증 감지 Feature 강화)
def create_sequence_features(df, feature_cols, seq_len, idx, limit_val=1700):
    """특정 인덱스에서 seq_len 만큼의 시퀀스 Feature 추출 (급증 감지 강화)"""
    features = []
    for col in feature_cols:
        seq = df[col].iloc[idx - seq_len:idx].values
        current_val = seq[-1]

        # 기본 통계 Feature
        features.extend([
            np.mean(seq),           # 평균
            np.std(seq),            # 표준편차
            np.min(seq),            # 최소
            np.max(seq),            # 최대
            current_val,            # 현재값
            seq[-1] - seq[0],       # 변화량 (240분 전 대비)
            np.percentile(seq, 25), # 25% 분위수
            np.percentile(seq, 75), # 75% 분위수
            np.mean(seq[-10:]) - np.mean(seq[:10]),  # 최근 추세
            np.max(seq[-30:]) if len(seq) >= 30 else np.max(seq),  # 최근 30분 최대
        ])

        # 급증 감지 Feature 추가
        features.extend([
            seq[-1] - seq[-10] if len(seq) >= 10 else 0,    # 최근 10분 변화량
            seq[-1] - seq[-30] if len(seq) >= 30 else 0,    # 최근 30분 변화량
            seq[-1] - seq[-60] if len(seq) >= 60 else 0,    # 최근 60분 변화량
            (seq[-1] - seq[-10]) / 10 if len(seq) >= 10 else 0,  # 최근 10분 기울기
            (seq[-1] - seq[-30]) / 30 if len(seq) >= 30 else 0,  # 최근 30분 기울기
            np.max(seq[-10:]) - np.min(seq[-10:]) if len(seq) >= 10 else 0,  # 최근 10분 변동폭
            np.max(seq[-30:]) - np.min(seq[-30:]) if len(seq) >= 30 else 0,  # 최근 30분 변동폭
            limit_val - current_val,  # 리미트까지 남은 거리
            1 if current_val >= 1500 else 0,  # 1500 이상 여부
            1 if current_val >= 1600 else 0,  # 1600 이상 여부
        ])
    return features

# 타겟 생성 (30분 내 최대값 및 돌파 여부)
def create_target(df, target_col, pred_window, limit_val, idx):
    """30분 내 최대값과 돌파 여부 반환"""
    future_vals = df[target_col].iloc[idx:idx + pred_window].values
    if len(future_vals) == 0:
        return None, None
    max_val = np.max(future_vals)
    breach = 1 if max_val >= limit_val else 0
    return max_val, breach

# 데이터 생성
print("  - 시퀀스 Feature 추출 중... (시간이 소요됩니다)")

X_target, X_important, X_auxiliary = [], [], []
y_regression, y_classification = [], []
current_values = []  # 현재값 저장 (급증 케이스 가중치용)

valid_count = 0
breach_count = 0
surge_count = 0  # 급증 케이스 카운트

for idx in range(seq_len, len(df) - pred_window):
    if idx % 10000 == 0:
        print(f"    진행률: {idx:,}/{len(df) - pred_window:,} ({100*idx/(len(df)-pred_window):.1f}%)")

    # 타겟 생성
    max_val, breach = create_target(df, target_col, pred_window, limit_val, idx)
    if max_val is None:
        continue

    # 현재값 저장
    current_val = df[target_col].iloc[idx]
    current_values.append(current_val)

    # 시퀀스 Feature 생성
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
        # 급증 케이스: 현재 < 1600 & 30분 후 >= 1700
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
print(f"  - 급증 케이스 (현재<1600 → 1700돌파): {surge_count:,}건")
print(f"  - 안전 샘플 수: {valid_count - breach_count:,} ({100*(valid_count-breach_count)/valid_count:.2f}%)")

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

# 전체 Feature 결합 (각 모델별로 다르게 사용)
X_all = np.hstack([X_target_scaled, X_important_scaled, X_auxiliary_scaled])

print(f"  - 타겟 Feature shape: {X_target_scaled.shape}")
print(f"  - 중요 Feature shape: {X_important_scaled.shape}")
print(f"  - 보조 Feature shape: {X_auxiliary_scaled.shape}")
print(f"  - 전체 Feature shape: {X_all.shape}")

# ============================================================================
# PyOD 이상 탐지 모델 학습
# ============================================================================
print("\n[5/6] PyOD 이상 탐지 모델 학습 중...")

# IsolationForest
print("  - IsolationForest 학습 중...")
pyod_iforest = IForest(n_estimators=100, random_state=42, n_jobs=-1)
pyod_iforest.fit(X_target_scaled)

# COPOD
print("  - COPOD 학습 중...")
pyod_copod = COPOD()
pyod_copod.fit(X_target_scaled)

# ============================================================================
# 6개 모델 학습
# ============================================================================
print("\n[6/6] 6개 예측 모델 학습 중...")

# 샘플 가중치 계산
# - 일반 돌파 케이스: 10배
# - 급증 케이스 (현재 < 1600 & 30분 후 >= 1700): 20배
sample_weights = np.ones(len(y_classification))
sample_weights[y_classification == 1] = CONFIG['surge_weight']  # 돌파는 10배

# 급증 케이스에 추가 가중치 (20배)
surge_mask = (current_values < 1600) & (y_classification == 1)
sample_weights[surge_mask] = 20
print(f"  - 급증 케이스 가중치 20배 적용: {np.sum(surge_mask):,}건")

# Train/Validation 분할
X_train, X_val, y_reg_train, y_reg_val, y_cls_train, y_cls_val, w_train, w_val, cv_train, cv_val = train_test_split(
    X_all, y_regression, y_classification, sample_weights, current_values,
    test_size=0.2, random_state=42, stratify=y_classification
)

# 각 Feature 그룹별 인덱스
target_end = X_target_scaled.shape[1]
important_end = target_end + X_important_scaled.shape[1]
auxiliary_end = important_end + X_auxiliary_scaled.shape[1]

# 모델 저장 딕셔너리
models = {}

# --- XGBoost 회귀 모델 3개 ---
print("\n  [XGBoost 회귀 모델]")

# 1. 타겟 지표 모델
print("    - XGB 타겟 모델 학습 중...")
xgb_target = xgb.XGBRegressor(**CONFIG['xgb_params'])
xgb_target.fit(X_train[:, :target_end], y_reg_train, sample_weight=w_train)
models['xgb_target'] = xgb_target

# 2. 중요 지표 모델
print("    - XGB 중요 모델 학습 중...")
xgb_important = xgb.XGBRegressor(**CONFIG['xgb_params'])
xgb_important.fit(X_train[:, target_end:important_end], y_reg_train, sample_weight=w_train)
models['xgb_important'] = xgb_important

# 3. 보조 지표 모델
print("    - XGB 보조 모델 학습 중...")
xgb_auxiliary = xgb.XGBRegressor(**CONFIG['xgb_params'])
xgb_auxiliary.fit(X_train[:, important_end:auxiliary_end], y_reg_train, sample_weight=w_train)
models['xgb_auxiliary'] = xgb_auxiliary

# --- LightGBM 분류 모델 3개 ---
print("\n  [LightGBM 분류 모델]")

# scale_pos_weight 계산
pos_ratio = np.sum(y_cls_train == 0) / np.sum(y_cls_train == 1) if np.sum(y_cls_train == 1) > 0 else 1
lgb_params = CONFIG['lgb_params'].copy()
lgb_params['scale_pos_weight'] = pos_ratio * CONFIG['surge_weight']

# 1. 타겟 지표 모델
print("    - LGBM 타겟 모델 학습 중...")
lgb_target = lgb.LGBMClassifier(**lgb_params)
lgb_target.fit(X_train[:, :target_end], y_cls_train, sample_weight=w_train)
models['lgb_target'] = lgb_target

# 2. 중요 지표 모델
print("    - LGBM 중요 모델 학습 중...")
lgb_important = lgb.LGBMClassifier(**lgb_params)
lgb_important.fit(X_train[:, target_end:important_end], y_cls_train, sample_weight=w_train)
models['lgb_important'] = lgb_important

# 3. 보조 지표 모델
print("    - LGBM 보조 모델 학습 중...")
lgb_auxiliary = lgb.LGBMClassifier(**lgb_params)
lgb_auxiliary.fit(X_train[:, important_end:auxiliary_end], y_cls_train, sample_weight=w_train)
models['lgb_auxiliary'] = lgb_auxiliary

# ============================================================================
# Validation 성능 평가
# ============================================================================
print("\n" + "=" * 80)
print("Validation 성능 평가")
print("=" * 80)

# XGBoost 회귀 예측
pred_xgb_target = xgb_target.predict(X_val[:, :target_end])
pred_xgb_important = xgb_important.predict(X_val[:, target_end:important_end])
pred_xgb_auxiliary = xgb_auxiliary.predict(X_val[:, important_end:auxiliary_end])

# LightGBM 분류 예측
pred_lgb_target = lgb_target.predict(X_val[:, :target_end])
pred_lgb_important = lgb_important.predict(X_val[:, target_end:important_end])
pred_lgb_auxiliary = lgb_auxiliary.predict(X_val[:, important_end:auxiliary_end])

# XGBoost 결과를 이진 분류로 변환
xgb_target_breach = (pred_xgb_target >= limit_val).astype(int)
xgb_important_breach = (pred_xgb_important >= limit_val).astype(int)
xgb_auxiliary_breach = (pred_xgb_auxiliary >= limit_val).astype(int)

# 다수결
votes = np.vstack([
    xgb_target_breach,
    xgb_important_breach,
    xgb_auxiliary_breach,
    pred_lgb_target,
    pred_lgb_important,
    pred_lgb_auxiliary
])

vote_sum = votes.sum(axis=0)
final_pred = (vote_sum >= 4).astype(int)

# 성능 계산
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

print("\n[개별 모델 성능]")
model_preds = {
    'XGB 타겟': xgb_target_breach,
    'XGB 중요': xgb_important_breach,
    'XGB 보조': xgb_auxiliary_breach,
    'LGBM 타겟': pred_lgb_target,
    'LGBM 중요': pred_lgb_important,
    'LGBM 보조': pred_lgb_auxiliary,
}

for name, pred in model_preds.items():
    acc = accuracy_score(y_cls_val, pred)
    prec = precision_score(y_cls_val, pred, zero_division=0)
    rec = recall_score(y_cls_val, pred, zero_division=0)
    f1 = f1_score(y_cls_val, pred, zero_division=0)
    print(f"  {name:12s}: Acc={acc:.4f}, Prec={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")

print("\n[다수결 앙상블 성능]")
acc = accuracy_score(y_cls_val, final_pred)
prec = precision_score(y_cls_val, final_pred, zero_division=0)
rec = recall_score(y_cls_val, final_pred, zero_division=0)
f1 = f1_score(y_cls_val, final_pred, zero_division=0)
print(f"  다수결(4/6): Acc={acc:.4f}, Prec={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")

print("\n[혼동 행렬]")
cm = confusion_matrix(y_cls_val, final_pred)
print(f"  TN={cm[0,0]:,}, FP={cm[0,1]:,}")
print(f"  FN={cm[1,0]:,}, TP={cm[1,1]:,}")

# ============================================================================
# 모델 저장
# ============================================================================
print("\n" + "=" * 80)
print("모델 저장")
print("=" * 80)

# 저장할 객체
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
        'total_samples': valid_count,
        'breach_samples': breach_count,
        'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'validation_accuracy': acc,
        'validation_recall': rec,
        'validation_f1': f1
    }
}

model_path = os.path.join(CONFIG['model_dir'], 'v10_m14_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(save_data, f)

print(f"  - 모델 저장 완료: {model_path}")

# 학습 로그 저장
log_path = os.path.join(CONFIG['log_dir'], f'train_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
with open(log_path, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("V10 ML 예측 모델 학습 로그\n")
    f.write("=" * 80 + "\n")
    f.write(f"학습 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"학습 파일: {CONFIG['train_file']}\n")
    f.write(f"총 샘플 수: {valid_count:,}\n")
    f.write(f"돌파 샘플 수: {breach_count:,} ({100*breach_count/valid_count:.2f}%)\n")
    f.write(f"\nValidation 성능:\n")
    f.write(f"  Accuracy: {acc:.4f}\n")
    f.write(f"  Precision: {prec:.4f}\n")
    f.write(f"  Recall: {rec:.4f}\n")
    f.write(f"  F1: {f1:.4f}\n")

print(f"  - 로그 저장 완료: {log_path}")

print("\n" + "=" * 80)
print("학습 완료!")
print("=" * 80)