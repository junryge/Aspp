# -*- coding: utf-8 -*-
"""
================================================================================
V10_4c ML 예측 모델 - 학습 코드 (30분 예측)
V10_4b + LGBM 분류기 튜닝 (class_weight, threshold 조정)
================================================================================
"""

import os
import pickle
import warnings
import glob
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')

# ============================================================================
# 설정
# ============================================================================
CONFIG = {
    'train_file': 'M14_학습*.CSV',
    'model_file': 'models/v10_4c_30min_m14_model.pkl',
    'sequence_length': 280,
    'prediction_offset': 30,
    'limit_value': 1700,
    'target_column': 'TOTALCNT',
}

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
    'pdt_new': [
        'M14.STRATE.N2.STORAGERATIO',
        'M14.PDT.LAYOUT.M14A_M14ATOM14ACNV_CURRENTQCNT',
        'M14.PDT.LAYOUT.HUBROOM_M14TOM16_CURRENTQCNT',
    ],
    'job_features': [
        'JobPrep_Count',
        'Reserved_Count', 
        'JobEnd_Count',
        'Job_Total',
        'Job_Total_ma10',
    ]
}

print("=" * 70)
print("🚀 V10_4c ML 예측 모델 - 학습 시작")
print("   V10_4b + LGBM 튜닝 (class_weight, scale_pos_weight)")
print("   시퀀스: 280분, 예측: 30분 후")
print("=" * 70)

# ============================================================================
# 데이터 로드
# ============================================================================
def load_data(path_pattern):
    print(f"\n[1/6] 데이터 로드 중... ({path_pattern})")
    files = glob.glob(path_pattern)
    if not files:
        raise ValueError("데이터 파일을 찾을 수 없습니다!")
    
    dfs = []
    for f in sorted(files):
        try:
            df = pd.read_csv(f, on_bad_lines='skip', encoding='utf-8')
        except:
            try:
                df = pd.read_csv(f, on_bad_lines='skip', encoding='cp949')
            except:
                df = pd.read_csv(f, on_bad_lines='skip', encoding='euc-kr')
        print(f"  - {f}: {len(df):,}행")
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)

df = load_data(CONFIG['train_file'])
print(f"  → 총 데이터: {len(df):,}행")

df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M', errors='coerce')
df = df.dropna(subset=['CURRTIME']).sort_values('CURRTIME').reset_index(drop=True)

# ============================================================================
# 파생변수 생성
# ============================================================================
print("\n[2/6] Feature 그룹 확인 및 파생변수 생성...")

# QUEUE_GAP 파생
if 'M14.QUE.ALL.CURRENTQCREATED' in df.columns and 'M14.QUE.ALL.CURRENTQCOMPLETED' in df.columns:
    df['QUEUE_GAP'] = df['M14.QUE.ALL.CURRENTQCREATED'] - df['M14.QUE.ALL.CURRENTQCOMPLETED']
    FEATURE_GROUPS['auxiliary'].append('QUEUE_GAP')
    print("  - QUEUE_GAP 파생 변수 추가!")

# Job Feature 파생변수 생성
job_cols_exist = all(col in df.columns for col in ['JobPrep_Count', 'Reserved_Count', 'JobEnd_Count'])
if job_cols_exist:
    df['Job_Total'] = df['JobPrep_Count'] + df['Reserved_Count'] + df['JobEnd_Count']
    df['Job_Total_ma10'] = df['Job_Total'].rolling(10, min_periods=1).mean()
    print("  - Job_Total, Job_Total_ma10 파생 변수 추가!")
else:
    print("  ⚠️ Job Feature 컬럼 없음 - Job 그룹 비활성화")
    FEATURE_GROUPS['job_features'] = []

# Feature 그룹 필터링
for group_name in FEATURE_GROUPS:
    original = len(FEATURE_GROUPS[group_name])
    FEATURE_GROUPS[group_name] = [f for f in FEATURE_GROUPS[group_name] if f in df.columns]
    print(f"  - {group_name}: {len(FEATURE_GROUPS[group_name])}/{original}개")

# 숫자형 변환
all_cols = []
for group in FEATURE_GROUPS.values():
    all_cols.extend(group)
for col in list(set(all_cols)):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# ============================================================================
# Feature 생성 함수
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
print("\n[3/6] 학습 데이터 생성 중...")

seq_len = CONFIG['sequence_length']
pred_offset = CONFIG['prediction_offset']
limit_val = CONFIG['limit_value']
target_col = CONFIG['target_column']

X_target, X_important, X_auxiliary, X_pdt_new, X_job = [], [], [], [], []
y_reg, y_clf = [], []

for idx in range(seq_len, len(df) - pred_offset):
    if idx % 10000 == 0:
        print(f"    진행: {idx:,}/{len(df) - pred_offset:,}")
    
    # 타겟: 30분 내 최대값
    future_val = df[target_col].iloc[idx:idx + pred_offset].max()
    y_reg.append(future_val)
    y_clf.append(1 if future_val >= limit_val else 0)
    
    X_target.append(create_sequence_features(df, FEATURE_GROUPS['target'], seq_len, idx))
    X_important.append(create_sequence_features(df, FEATURE_GROUPS['important'], seq_len, idx))
    X_auxiliary.append(create_sequence_features(df, FEATURE_GROUPS['auxiliary'], seq_len, idx))
    
    if FEATURE_GROUPS['pdt_new']:
        X_pdt_new.append(create_sequence_features(df, FEATURE_GROUPS['pdt_new'], seq_len, idx))
    
    if FEATURE_GROUPS['job_features']:
        X_job.append(create_sequence_features(df, FEATURE_GROUPS['job_features'], seq_len, idx))

X_target = np.array(X_target)
X_important = np.array(X_important)
X_auxiliary = np.array(X_auxiliary)
X_pdt_new = np.array(X_pdt_new) if X_pdt_new else np.array([])
X_job = np.array(X_job) if X_job else np.array([])
y_reg = np.array(y_reg)
y_clf = np.array(y_clf)

# ★ 클래스 불균형 계산
n_danger = sum(y_clf)
n_safe = len(y_clf) - n_danger
class_ratio = n_safe / n_danger if n_danger > 0 else 1

print(f"  → 샘플: {len(y_reg):,}개")
print(f"  → 1700+: {n_danger:,}개 ({100*n_danger/len(y_clf):.2f}%)")
print(f"  → 클래스 비율: 1:{class_ratio:.1f} (scale_pos_weight에 사용)")

# ============================================================================
# 스케일링
# ============================================================================
print("\n[4/6] 스케일링...")

scalers = {
    'target': StandardScaler(),
    'important': StandardScaler(),
    'auxiliary': StandardScaler(),
}

X_target_scaled = scalers['target'].fit_transform(X_target)
X_important_scaled = scalers['important'].fit_transform(X_important)
X_auxiliary_scaled = scalers['auxiliary'].fit_transform(X_auxiliary)

if len(X_pdt_new) > 0:
    scalers['pdt_new'] = StandardScaler()
    X_pdt_new_scaled = scalers['pdt_new'].fit_transform(X_pdt_new)

if len(X_job) > 0:
    scalers['job_features'] = StandardScaler()
    X_job_scaled = scalers['job_features'].fit_transform(X_job)

# ============================================================================
# 모델 학습 (★ LGBM 튜닝!)
# ============================================================================
print("\n[5/6] 모델 학습 중...")

models = {}

# XGB 회귀 파라미터 (기존과 동일)
xgb_reg_params = {
    'n_estimators': 200, 
    'max_depth': 6, 
    'learning_rate': 0.05, 
    'random_state': 42, 
    'n_jobs': -1
}

# ★ LGBM 분류 파라미터 튜닝 (클래스 불균형 처리)
lgb_clf_params = {
    'n_estimators': 300,           # 더 많은 트리
    'max_depth': 8,                # 더 깊게
    'learning_rate': 0.03,         # 더 작은 학습률
    'num_leaves': 50,              # 더 많은 리프
    'min_child_samples': 20,       # 과적합 방지
    'scale_pos_weight': class_ratio,  # ★ 클래스 불균형 보정
    'random_state': 42, 
    'n_jobs': -1, 
    'verbose': -1,
    'force_col_wise': True,
}

# ★ XGB 분류도 추가 (LGBM 보완용)
xgb_clf_params = {
    'n_estimators': 300,
    'max_depth': 8,
    'learning_rate': 0.03,
    'scale_pos_weight': class_ratio,  # ★ 클래스 불균형 보정
    'random_state': 42,
    'n_jobs': -1,
    'use_label_encoder': False,
    'eval_metric': 'logloss',
}

# Target 그룹
print("  - XGB Target (회귀)...")
models['xgb_target'] = XGBRegressor(**xgb_reg_params)
models['xgb_target'].fit(X_target_scaled, y_reg)

print("  - LGBM Target (분류, 튜닝)...")
models['lgb_target'] = LGBMClassifier(**lgb_clf_params)
models['lgb_target'].fit(X_target_scaled, y_clf)

print("  - XGB Target (분류, 추가)...")
models['xgb_target_clf'] = XGBClassifier(**xgb_clf_params)
models['xgb_target_clf'].fit(X_target_scaled, y_clf)

# Important 그룹
print("  - XGB Important (회귀)...")
models['xgb_important'] = XGBRegressor(**xgb_reg_params)
models['xgb_important'].fit(X_important_scaled, y_reg)

print("  - LGBM Important (분류, 튜닝)...")
models['lgb_important'] = LGBMClassifier(**lgb_clf_params)
models['lgb_important'].fit(X_important_scaled, y_clf)

print("  - XGB Important (분류, 추가)...")
models['xgb_important_clf'] = XGBClassifier(**xgb_clf_params)
models['xgb_important_clf'].fit(X_important_scaled, y_clf)

# Auxiliary 그룹
print("  - XGB Auxiliary (회귀)...")
models['xgb_auxiliary'] = XGBRegressor(**xgb_reg_params)
models['xgb_auxiliary'].fit(X_auxiliary_scaled, y_reg)

print("  - LGBM Auxiliary (분류, 튜닝)...")
models['lgb_auxiliary'] = LGBMClassifier(**lgb_clf_params)
models['lgb_auxiliary'].fit(X_auxiliary_scaled, y_clf)

# PDT 그룹
if len(X_pdt_new) > 0:
    print("  - XGB PDT (회귀)...")
    models['xgb_pdt_new'] = XGBRegressor(**xgb_reg_params)
    models['xgb_pdt_new'].fit(X_pdt_new_scaled, y_reg)
    
    print("  - LGBM PDT (분류, 튜닝)...")
    models['lgb_pdt_new'] = LGBMClassifier(**lgb_clf_params)
    models['lgb_pdt_new'].fit(X_pdt_new_scaled, y_clf)

# Job 그룹
if len(X_job) > 0:
    print("  - XGB Job (회귀)...")
    models['xgb_job'] = XGBRegressor(**xgb_reg_params)
    models['xgb_job'].fit(X_job_scaled, y_reg)
    
    print("  - LGBM Job (분류, 튜닝)...")
    models['lgb_job'] = LGBMClassifier(**lgb_clf_params)
    models['lgb_job'].fit(X_job_scaled, y_clf)

# ============================================================================
# 저장
# ============================================================================
print("\n[6/6] 모델 저장...")

os.makedirs('models', exist_ok=True)

model_data = {
    'models': models,
    'scalers': scalers,
    'config': CONFIG,
    'feature_groups': FEATURE_GROUPS,
    'training_info': {
        'version': 'V10_4c',
        'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'samples': len(y_reg),
        'danger_samples': int(n_danger),
        'class_ratio': round(class_ratio, 2),
        'job_features_enabled': len(X_job) > 0,
        'tuning': 'LGBM scale_pos_weight + XGB분류 추가',
    }
}

with open(CONFIG['model_file'], 'wb') as f:
    pickle.dump(model_data, f)

print(f"  → 저장: {CONFIG['model_file']}")
print("\n" + "=" * 70)
print("✅ V10_4c 학습 완료! (30분 예측, LGBM 튜닝)")
print(f"   클래스 비율: 1:{class_ratio:.1f}")
print(f"   모델 수: {len(models)}개")
print("=" * 70)