#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
🚀 10단계: 모델 학습
================================================================================
9단계에서 선택된 31개 Feature로 XGBoost 모델 학습
- 학습/검증 분리
- 1700+ 가중치 학습
- 모델 저장
================================================================================
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import warnings
import gc

warnings.filterwarnings('ignore')

print("="*80)
print("🚀 10단계: 모델 학습")
print("   31개 선택 Feature로 XGBoost 학습")
print("="*80)

# ==============================================================================
# 1. 선택된 Feature 목록 (9단계 결과)
# ==============================================================================

SELECTED_FEATURES = [
    'total_last10',
    'M14AM14BSUM_current',
    'total_mean',
    'total_min',
    'SENDFAB_VERTICALQUEUECOUNT_over_280',
    'M10AM14A_over_55',
    'M10AM14A_over_60',
    'M10AM14A_over_50',
    'total_over_1600',
    'OHT_OHTUTIL_over_80',
    'vertical_gold',
    'inflow_surge',
    'M10AM14A_min',
    'gold_normal',
    'gold_strict',
    'triple_check',
    'total_slope',
    'OHT_CURRENTOHTQCNT_over_800',
    'm16_surge',
    'queue_gap_min',
    'cnv_total_high',
    'quad_check',
    'oht_overload',
    'queue_accel_danger',
    'M14AM14BSUM_trend20',
    'SENDFAB_VERTICALQUEUECOUNT_trend10',
    'M14AM14BSUM_trend10',
    'cnv_imbalance',
    'SENDFAB_VERTICALQUEUECOUNT_trend5',
    'OHT_OHTUTIL_trend20',
    'OHT_OHTUTIL_trend10'
]

print(f"\n📋 선택된 Feature: {len(SELECTED_FEATURES)}개")

# ==============================================================================
# 2. 데이터 로드
# ==============================================================================

print("\n📂 데이터 로딩...")
INPUT_FILE = 'step8_features_10min.csv'

# 필요한 컬럼만 로드 (메모리 절약)
use_cols = SELECTED_FEATURES + ['target_TOTALCNT', 'is_danger', 'current_TOTALCNT']
df = pd.read_csv(INPUT_FILE, usecols=use_cols)

print(f"  ✅ 로드 완료: {len(df):,}행 × {len(df.columns)}컬럼")

# 결측치 처리
df = df.fillna(0)
df = df.replace([np.inf, -np.inf], 0)

# X, y 분리
X = df[SELECTED_FEATURES]
y = df['target_TOTALCNT']

print(f"  ✅ X shape: {X.shape}")
print(f"  ✅ y 범위: {y.min():.0f} ~ {y.max():.0f}")

# ==============================================================================
# 3. 학습/검증 분리 (시간순)
# ==============================================================================

print("\n📊 학습/검증 분리...")

# 시간순 분리 (마지막 20%를 검증용)
split_idx = int(len(X) * 0.8)
X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"  학습: {len(X_train):,}개")
print(f"  검증: {len(X_val):,}개")

# 1700+ 샘플 수
train_danger = (y_train >= 1700).sum()
val_danger = (y_val >= 1700).sum()
print(f"  학습 1700+: {train_danger:,}개 ({train_danger/len(y_train)*100:.2f}%)")
print(f"  검증 1700+: {val_danger:,}개 ({val_danger/len(y_val)*100:.2f}%)")

# ==============================================================================
# 4. 가중치 설정 (1700+ 강조)
# ==============================================================================

print("\n⚖️ 가중치 설정...")

weights = np.ones(len(y_train))
weights[y_train >= 1500] = 3
weights[y_train >= 1600] = 10
weights[y_train >= 1700] = 30
weights[y_train >= 1800] = 50

print(f"  일반: 1배")
print(f"  1500+: 3배")
print(f"  1600+: 10배")
print(f"  1700+: 30배")
print(f"  1800+: 50배")

# ==============================================================================
# 5. XGBoost 학습
# ==============================================================================

print("\n🚀 XGBoost 학습 시작...")

# GPU 사용 가능 여부 확인
try:
    test_model = xgb.XGBRegressor(device='cuda', n_estimators=10)
    test_model.fit(X_train[:100], y_train[:100])
    USE_GPU = True
    print("  ✅ GPU 사용 가능!")
except:
    USE_GPU = False
    print("  ⚠️ GPU 불가, CPU 사용")

# 모델 정의
model = xgb.XGBRegressor(
    n_estimators=1000,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    n_jobs=-1,
    device='cuda' if USE_GPU else 'cpu',
    tree_method='hist',
    random_state=42
)

# 학습
model.fit(
    X_train, y_train,
    sample_weight=weights,
    eval_set=[(X_val, y_val)],
    verbose=100
)

print("  ✅ 학습 완료!")

# ==============================================================================
# 6. Feature Importance
# ==============================================================================

print("\n📊 Feature Importance...")

importance = dict(zip(SELECTED_FEATURES, model.feature_importances_))
importance_sorted = sorted(importance.items(), key=lambda x: x[1], reverse=True)

print("\n  TOP 15 Feature:")
for i, (feat, imp) in enumerate(importance_sorted[:15], 1):
    bar = '█' * int(imp * 100)
    print(f"  {i:2d}. {feat[:40]:40s} {imp:.4f} {bar}")

# ==============================================================================
# 7. 검증 평가
# ==============================================================================

print("\n" + "="*80)
print("📈 검증 결과")
print("="*80)

y_pred = model.predict(X_val)

# 기본 메트릭
mae = mean_absolute_error(y_val, y_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)

print(f"\n  📊 기본 메트릭:")
print(f"     MAE: {mae:.2f}")
print(f"     RMSE: {rmse:.2f}")
print(f"     R²: {r2:.4f}")

# 1700+ 감지율
actual_danger = y_val >= 1700
pred_danger = y_pred >= 1700

tp = ((actual_danger) & (pred_danger)).sum()
fn = ((actual_danger) & (~pred_danger)).sum()
fp = ((~actual_danger) & (pred_danger)).sum()
tn = ((~actual_danger) & (~pred_danger)).sum()

recall = tp / (tp + fn) if (tp + fn) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n  🔥 1700+ 감지 성능:")
print(f"     실제 1700+: {actual_danger.sum():,}개")
print(f"     예측 1700+: {pred_danger.sum():,}개")
print(f"     TP (정탐): {tp:,}개")
print(f"     FN (미탐): {fn:,}개")
print(f"     FP (오탐): {fp:,}개")
print(f"     ─────────────────────")
print(f"     감지율 (Recall): {recall*100:.1f}%")
print(f"     정밀도 (Precision): {precision*100:.1f}%")
print(f"     F1 Score: {f1:.4f}")

# 구간별 MAE
print(f"\n  📊 구간별 MAE:")
for low, high in [(0, 1600), (1600, 1700), (1700, 1800), (1800, 9999)]:
    mask = (y_val >= low) & (y_val < high)
    if mask.sum() > 0:
        segment_mae = mean_absolute_error(y_val[mask], y_pred[mask])
        print(f"     {low:4d} ~ {high:4d}: MAE {segment_mae:.2f} ({mask.sum():,}개)")

# ==============================================================================
# 8. 모델 저장
# ==============================================================================

print("\n💾 모델 저장...")

# 모델 저장
with open('model_v10_31feat.pkl', 'wb') as f:
    pickle.dump(model, f)
print("  ✅ model_v10_31feat.pkl")

# Feature 목록 저장
with open('model_v10_features.pkl', 'wb') as f:
    pickle.dump(SELECTED_FEATURES, f)
print("  ✅ model_v10_features.pkl")

# Feature Importance 저장
importance_df = pd.DataFrame(importance_sorted, columns=['feature', 'importance'])
importance_df.to_csv('model_v10_importance.csv', index=False, encoding='utf-8-sig')
print("  ✅ model_v10_importance.csv")

# ==============================================================================
# 9. 결과 요약
# ==============================================================================

print("\n" + "="*80)
print("📋 10단계 결과 요약")
print("="*80)
print(f"  Feature 수: {len(SELECTED_FEATURES)}개")
print(f"  학습 샘플: {len(X_train):,}개")
print(f"  검증 샘플: {len(X_val):,}개")
print(f"  ─────────────────────────")
print(f"  MAE: {mae:.2f}")
print(f"  R²: {r2:.4f}")
print(f"  1700+ 감지율: {recall*100:.1f}%")
print(f"  1700+ 정밀도: {precision*100:.1f}%")
print(f"  ─────────────────────────")
print(f"  모델 파일: model_v10_31feat.pkl")

print("\n✅ 10단계 완료!")
print("   → 11단계: 별도 평가 데이터로 최종 검증")