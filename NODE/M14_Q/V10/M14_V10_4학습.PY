# -*- coding: utf-8 -*-
"""
================================================================================
V11 ML 예측 모델 - 학습 코드
V8.3.1 + V10.3 통합 (XGBoost 전용)

특징:
    - 시퀀스: 280분 (V8.3.1 기준)
    - 예측: 10분 후 단일값 (V8.3.1 기준)
    - Feature 그룹: target, important, auxiliary (V10.3 구조)
    - 모델: XGBoost만 사용 (LightGBM 제외)
================================================================================
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import glob
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

# ============================================================================
# 설정
# ============================================================================
CONFIG = {
    'train_file': 'M14_학습*.CSV',  # glob 패턴
    'model_file': 'models/v11_m14_model.pkl',
    'sequence_length': 280,      # V8.3.1 기준
    'prediction_offset': 10,     # V8.3.1 기준 (10분 후 단일값)
    'limit_value': 1700,
    'target_column': 'TOTALCNT',
}

# Feature 그룹 (V10.3 동일)
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
    # ★ 신규 컬럼 (3개)
    'pdt_new': [
        'M14.STRATE.N2.STORAGERATIO',
        'M14.PDT.LAYOUT.M14A_M14ATOM14ACNV_CURRENTQCNT',
        'M14.PDT.LAYOUT.HUBROOM_M14TOM16_CURRENTQCNT',
    ]
}

print("=" * 70)
print("🚀 V11 ML 예측 모델 - 학습 시작")
print("   V8.3.1 + V10.3 통합 (XGBoost 전용)")
print("   시퀀스: 280분, 예측: 10분 후")
print("=" * 70)

# ============================================================================
# 데이터 로드
# ============================================================================
def load_data(path_pattern):
    print(f"\n[1/6] 데이터 로드 중... ({path_pattern})")
    files = glob.glob(path_pattern)
    
    if not files:
        # 대안 패턴 시도
        alt_patterns = ['M14_Q_*.CSV', '*.CSV']
        for alt in alt_patterns:
            files = glob.glob(alt)
            if files:
                print(f"  → 대안 패턴 사용: {alt}")
                break
    
    if not files:
        raise ValueError("데이터 파일을 찾을 수 없습니다!")
    
    dfs = []
    for f in sorted(files):
        try:
            df = pd.read_csv(f, on_bad_lines='skip', encoding='utf-8')
            print(f"  - {f}: {len(df):,}행")
            dfs.append(df)
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(f, on_bad_lines='skip', encoding='cp949')
                print(f"  - {f} (cp949): {len(df):,}행")
                dfs.append(df)
            except:
                df = pd.read_csv(f, on_bad_lines='skip', encoding='euc-kr')
                print(f"  - {f} (euc-kr): {len(df):,}행")
                dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)

df = load_data(CONFIG['train_file'])
print(f"  → 총 데이터: {len(df):,}행")

# CURRTIME 파싱
if 'CURRTIME' in df.columns:
    df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M', errors='coerce')
    df = df.dropna(subset=['CURRTIME'])
    df = df.sort_values('CURRTIME').reset_index(drop=True)

# Feature 그룹 필터링 (존재하는 컬럼만)
print("\n[2/6] Feature 그룹 확인...")
for group_name in FEATURE_GROUPS:
    original_count = len(FEATURE_GROUPS[group_name])
    FEATURE_GROUPS[group_name] = [f for f in FEATURE_GROUPS[group_name] if f in df.columns]
    print(f"  - {group_name}: {len(FEATURE_GROUPS[group_name])}/{original_count}개")

# 숫자형 변환
all_cols = []
for group in FEATURE_GROUPS.values():
    all_cols.extend(group)
all_cols = list(set(all_cols))

for col in all_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# queue_gap 파생 변수 생성
if 'M14.QUE.ALL.CURRENTQCREATED' in df.columns and 'M14.QUE.ALL.CURRENTQCOMPLETED' in df.columns:
    df['QUEUE_GAP'] = df['M14.QUE.ALL.CURRENTQCREATED'] - df['M14.QUE.ALL.CURRENTQCOMPLETED']
    FEATURE_GROUPS['auxiliary'].append('QUEUE_GAP')
    print(f"  - QUEUE_GAP 파생 변수 추가!")

# ============================================================================
# Feature 생성 함수 (V10.3 기반 + V8.3.1 시퀀스)
# ============================================================================
def create_sequence_features(df, feature_cols, seq_len, idx, limit_val=1700):
    """시퀀스 기반 Feature 생성 (컬럼당 20개)"""
    features = []
    
    for col in feature_cols:
        seq = df[col].iloc[idx - seq_len:idx].values
        current_val = seq[-1]
        
        # 기본 통계 (10개)
        features.extend([
            np.mean(seq),                                           # 평균
            np.std(seq),                                            # 표준편차
            np.min(seq),                                            # 최소값
            np.max(seq),                                            # 최대값
            current_val,                                            # 현재값
            seq[-1] - seq[0],                                       # 전체 변화량
            np.percentile(seq, 25),                                 # 25% 분위수
            np.percentile(seq, 75),                                 # 75% 분위수
            np.mean(seq[-10:]) - np.mean(seq[:10]),                # 최근 vs 초기 평균 차이
            np.max(seq[-30:]) if len(seq) >= 30 else np.max(seq),  # 최근 30분 최대값
        ])
        
        # 추세/변화 (10개)
        features.extend([
            seq[-1] - seq[-10] if len(seq) >= 10 else 0,           # 10분 변화량
            seq[-1] - seq[-30] if len(seq) >= 30 else 0,           # 30분 변화량
            seq[-1] - seq[-60] if len(seq) >= 60 else 0,           # 60분 변화량
            (seq[-1] - seq[-10]) / 10 if len(seq) >= 10 else 0,    # 10분 기울기
            (seq[-1] - seq[-30]) / 30 if len(seq) >= 30 else 0,    # 30분 기울기
            np.max(seq[-10:]) - np.min(seq[-10:]) if len(seq) >= 10 else 0,  # 10분 범위
            np.max(seq[-30:]) - np.min(seq[-30:]) if len(seq) >= 30 else 0,  # 30분 범위
            limit_val - current_val,                                # 리미트까지 남은 값
            1 if current_val >= 1500 else 0,                       # 1500 이상 플래그
            1 if current_val >= 1600 else 0,                       # 1600 이상 플래그
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

X_target, X_important, X_auxiliary, X_pdt_new = [], [], [], []
y_values = []

total_samples = len(df) - seq_len - pred_offset
print(f"  → 예상 샘플 수: {total_samples:,}개")

for idx in range(seq_len, len(df) - pred_offset):
    if idx % 10000 == 0:
        print(f"    진행: {idx:,}/{len(df) - pred_offset:,} ({100*idx/(len(df)-pred_offset):.1f}%)")
    
    # 타겟: 10분 후 TOTALCNT 단일값 (V8.3.1 방식)
    future_val = df[target_col].iloc[idx + pred_offset]
    y_values.append(future_val)
    
    # Feature 생성
    X_target.append(create_sequence_features(df, FEATURE_GROUPS['target'], seq_len, idx))
    X_important.append(create_sequence_features(df, FEATURE_GROUPS['important'], seq_len, idx))
    X_auxiliary.append(create_sequence_features(df, FEATURE_GROUPS['auxiliary'], seq_len, idx))
    X_pdt_new.append(create_sequence_features(df, FEATURE_GROUPS['pdt_new'], seq_len, idx))

X_target = np.array(X_target)
X_important = np.array(X_important)
X_auxiliary = np.array(X_auxiliary)
X_pdt_new = np.array(X_pdt_new)
y_values = np.array(y_values)

print(f"\n  → 생성 완료:")
print(f"    - 샘플 수: {len(y_values):,}개")
print(f"    - Target Features: {X_target.shape[1]}개")
print(f"    - Important Features: {X_important.shape[1]}개")
print(f"    - Auxiliary Features: {X_auxiliary.shape[1]}개")
print(f"    - PDT New Features: {X_pdt_new.shape[1]}개")
print(f"    - 총 Features: {X_target.shape[1] + X_important.shape[1] + X_auxiliary.shape[1] + X_pdt_new.shape[1]}개")

# 1700+ 케이스 통계
danger_count = np.sum(y_values >= limit_val)
print(f"    - 1700+ 케이스: {danger_count:,}개 ({100*danger_count/len(y_values):.2f}%)")

# ============================================================================
# 스케일링
# ============================================================================
print("\n[4/6] 스케일링...")

scalers = {
    'target': StandardScaler(),
    'important': StandardScaler(),
    'auxiliary': StandardScaler(),
    'pdt_new': StandardScaler(),
}

X_target_scaled = scalers['target'].fit_transform(X_target)
X_important_scaled = scalers['important'].fit_transform(X_important)
X_auxiliary_scaled = scalers['auxiliary'].fit_transform(X_auxiliary)
X_pdt_new_scaled = scalers['pdt_new'].fit_transform(X_pdt_new)

# ============================================================================
# 가중치 설정 (V8.3.1 방식)
# ============================================================================
print("\n[5/6] 모델 학습 중 (XGBoost)...")

# 가중치: 고부하 구간 강조
weights = np.ones(len(y_values))
weights[y_values >= 1500] = 3
weights[y_values >= 1600] = 10
weights[y_values >= 1700] = 30

print(f"  → 가중치 분포:")
print(f"    - 기본(~1500): {np.sum(weights == 1):,}개")
print(f"    - 중간(1500~1600): {np.sum(weights == 3):,}개")
print(f"    - 높음(1600~1700): {np.sum(weights == 10):,}개")
print(f"    - 최고(1700+): {np.sum(weights == 30):,}개")

# ============================================================================
# XGBoost 모델 학습 (LightGBM 제외!)
# ============================================================================
models = {}

# XGBoost 하이퍼파라미터 (V8.3.1 참고)
xgb_params = {
    'n_estimators': 500,
    'max_depth': 8,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
}

print(f"\n  → XGBoost 파라미터:")
for k, v in xgb_params.items():
    print(f"    - {k}: {v}")

print("\n  - XGB Target 학습...")
models['xgb_target'] = XGBRegressor(**xgb_params)
models['xgb_target'].fit(X_target_scaled, y_values, sample_weight=weights)

print("  - XGB Important 학습...")
models['xgb_important'] = XGBRegressor(**xgb_params)
models['xgb_important'].fit(X_important_scaled, y_values, sample_weight=weights)

print("  - XGB Auxiliary 학습...")
models['xgb_auxiliary'] = XGBRegressor(**xgb_params)
models['xgb_auxiliary'].fit(X_auxiliary_scaled, y_values, sample_weight=weights)

print("  - XGB PDT New 학습...")
models['xgb_pdt_new'] = XGBRegressor(**xgb_params)
models['xgb_pdt_new'].fit(X_pdt_new_scaled, y_values, sample_weight=weights)

# ============================================================================
# 모델 저장
# ============================================================================
print("\n[6/6] 모델 저장...")

os.makedirs('models', exist_ok=True)

model_data = {
    'models': models,
    'scalers': scalers,
    'feature_indices': {
        'target_end': X_target.shape[1],
        'important_end': X_target.shape[1] + X_important.shape[1],
        'auxiliary_end': X_target.shape[1] + X_important.shape[1] + X_auxiliary.shape[1],
        'pdt_new_end': X_target.shape[1] + X_important.shape[1] + X_auxiliary.shape[1] + X_pdt_new.shape[1],
    },
    'config': CONFIG,
    'feature_groups': FEATURE_GROUPS,
    'xgb_params': xgb_params,
    'training_info': {
        'version': 'V11',
        'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'samples': len(y_values),
        'danger_samples': int(danger_count),
        'sequence_length': seq_len,
        'prediction_offset': pred_offset,
    }
}

with open(CONFIG['model_file'], 'wb') as f:
    pickle.dump(model_data, f)

print(f"  → 저장: {CONFIG['model_file']}")

# ============================================================================
# 완료
# ============================================================================
print("\n" + "=" * 70)
print("✅ V11 학습 완료!")
print("=" * 70)
print(f"  - 버전: V11 (V8.3.1 + V10.3 통합)")
print(f"  - 시퀀스: {seq_len}분")
print(f"  - 예측: {pred_offset}분 후")
print(f"  - 모델: XGBoost 4개 (target, important, auxiliary, pdt_new)")
print(f"  - 총 샘플: {len(y_values):,}개")
print(f"  - 1700+ 샘플: {danger_count:,}개")
print(f"  - 모델 파일: {CONFIG['model_file']}")
print("=" * 70)