#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
🚀 9단계: Feature Selection 2차
================================================================================
346개 Feature → 80~120개 축소
- 타겟 상관관계 분석
- Feature 간 다중공선성 제거
- Tree Importance (XGBoost)
- LASSO
- 최종 선택
================================================================================
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
import gc

warnings.filterwarnings('ignore')

print("="*80)
print("🚀 9단계: Feature Selection 2차")
print("   346개 Feature → 80~120개 축소")
print("="*80)

# ==============================================================================
# 1. 데이터 로드
# ==============================================================================

print("\n📂 데이터 로딩...")
INPUT_FILE = 'step8_features_10min.csv'

# 메타 컬럼 (제외할 것들)
META_COLS = ['CURRTIME', 'current_TOTALCNT', 'target_TOTALCNT', 'target_delta', 'is_danger']

# 샘플링해서 로드 (메모리 절약)
df = pd.read_csv(INPUT_FILE)
print(f"  ✅ 전체 데이터: {len(df):,}행")

# Feature 컬럼만 추출
feature_cols = [c for c in df.columns if c not in META_COLS]
print(f"  ✅ Feature 수: {len(feature_cols)}개")

# X, y 분리
X = df[feature_cols].copy()
y = df['target_TOTALCNT'].copy()

# 결측치 처리
X = X.fillna(0)
X = X.replace([np.inf, -np.inf], 0)

del df
gc.collect()

# ==============================================================================
# 2. 타겟 상관관계 분석
# ==============================================================================

print("\n📊 1단계: 타겟 상관관계 분석...")

correlations = {}
for col in feature_cols:
    try:
        corr = X[col].corr(y)
        if pd.notna(corr):
            correlations[col] = abs(corr)
    except:
        correlations[col] = 0

# 정렬
corr_sorted = sorted(correlations.items(), key=lambda x: x[1], reverse=True)

print(f"\n  📈 상관관계 TOP 30:")
for i, (col, corr) in enumerate(corr_sorted[:30], 1):
    print(f"    {i:2d}. {col[:50]:50s} {corr:.4f}")

# 상관관계 0.1 이상만 선택
high_corr_features = [col for col, corr in corr_sorted if corr >= 0.1]
print(f"\n  ✅ 상관계수 >= 0.1: {len(high_corr_features)}개")

# ==============================================================================
# 3. 다중공선성 제거
# ==============================================================================

print("\n📊 2단계: 다중공선성 제거...")

def remove_multicollinearity(X_subset, threshold=0.95):
    """상관계수 높은 Feature 쌍 중 하나 제거"""
    corr_matrix = X_subset.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = set()
    for col in upper.columns:
        high_corr = upper[col][upper[col] > threshold].index.tolist()
        for hc in high_corr:
            # 타겟 상관이 낮은 쪽 제거
            if correlations.get(col, 0) >= correlations.get(hc, 0):
                to_drop.add(hc)
            else:
                to_drop.add(col)
    
    return list(to_drop)

# 상관관계 높은 Feature들만 대상
X_high = X[high_corr_features]
to_drop = remove_multicollinearity(X_high, threshold=0.95)
print(f"  ⚠️ 다중공선성 제거: {len(to_drop)}개")

after_multicollinearity = [c for c in high_corr_features if c not in to_drop]
print(f"  ✅ 남은 Feature: {len(after_multicollinearity)}개")

# ==============================================================================
# 4. Tree Importance (XGBoost)
# ==============================================================================

print("\n📊 3단계: Tree Importance (XGBoost)...")

try:
    import xgboost as xgb
    
    # 샘플링 (속도)
    sample_size = min(50000, len(X))
    idx = np.random.choice(len(X), sample_size, replace=False)
    X_sample = X.iloc[idx]
    y_sample = y.iloc[idx]
    
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_sample, y_sample, verbose=False)
    
    importances = dict(zip(feature_cols, model.feature_importances_))
    imp_sorted = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n  📈 XGBoost Importance TOP 30:")
    for i, (col, imp) in enumerate(imp_sorted[:30], 1):
        print(f"    {i:2d}. {col[:50]:50s} {imp:.4f}")
    
    # Importance > 0.001
    xgb_features = [col for col, imp in imp_sorted if imp >= 0.001]
    print(f"\n  ✅ Importance >= 0.001: {len(xgb_features)}개")
    
except Exception as e:
    print(f"  ⚠️ XGBoost 오류: {e}")
    xgb_features = after_multicollinearity

gc.collect()

# ==============================================================================
# 5. ExtraTrees Importance
# ==============================================================================

print("\n📊 4단계: ExtraTrees Importance...")

try:
    model_et = ExtraTreesRegressor(
        n_estimators=100,
        max_depth=10,
        n_jobs=-1,
        random_state=42
    )
    model_et.fit(X_sample, y_sample)
    
    et_importances = dict(zip(feature_cols, model_et.feature_importances_))
    et_sorted = sorted(et_importances.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n  📈 ExtraTrees Importance TOP 30:")
    for i, (col, imp) in enumerate(et_sorted[:30], 1):
        print(f"    {i:2d}. {col[:50]:50s} {imp:.4f}")
    
    et_features = [col for col, imp in et_sorted if imp >= 0.001]
    print(f"\n  ✅ Importance >= 0.001: {len(et_features)}개")
    
except Exception as e:
    print(f"  ⚠️ ExtraTrees 오류: {e}")
    et_features = xgb_features

gc.collect()

# ==============================================================================
# 6. LASSO
# ==============================================================================

print("\n📊 5단계: LASSO Selection...")

try:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sample)
    
    lasso = LassoCV(cv=5, random_state=42, n_jobs=-1, max_iter=1000)
    lasso.fit(X_scaled, y_sample)
    
    lasso_coef = dict(zip(feature_cols, np.abs(lasso.coef_)))
    lasso_sorted = sorted(lasso_coef.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n  📈 LASSO Coefficient TOP 30:")
    for i, (col, coef) in enumerate(lasso_sorted[:30], 1):
        print(f"    {i:2d}. {col[:50]:50s} {coef:.4f}")
    
    lasso_features = [col for col, coef in lasso_sorted if coef > 0]
    print(f"\n  ✅ Non-zero coefficient: {len(lasso_features)}개")
    
except Exception as e:
    print(f"  ⚠️ LASSO 오류: {e}")
    lasso_features = []

gc.collect()

# ==============================================================================
# 7. 앙상블 선택 (2개 이상 방법에서 선택)
# ==============================================================================

print("\n📊 6단계: 앙상블 선택...")

# 각 방법에서 TOP 100 선택
top_corr = [c for c, _ in corr_sorted[:100]]
top_xgb = [c for c, _ in imp_sorted[:100]] if 'imp_sorted' in dir() else []
top_et = [c for c, _ in et_sorted[:100]] if 'et_sorted' in dir() else []
top_lasso = [c for c, _ in lasso_sorted[:100]] if 'lasso_sorted' in dir() else []

# 카운트
from collections import Counter
all_features = top_corr + top_xgb + top_et + top_lasso
feature_counts = Counter(all_features)

# 2개 이상에서 선택된 Feature
ensemble_features = [f for f, count in feature_counts.items() if count >= 2]
print(f"  ✅ 2개+ 방법에서 선택: {len(ensemble_features)}개")

# 3개 이상에서 선택된 Feature
strong_features = [f for f, count in feature_counts.items() if count >= 3]
print(f"  ✅ 3개+ 방법에서 선택: {len(strong_features)}개")

# ==============================================================================
# 8. 최종 선택
# ==============================================================================

print("\n" + "="*80)
print("📋 최종 Feature 선택")
print("="*80)

# 최종 선택 기준:
# 1. 3개+ 방법에서 선택된 Feature (strong)
# 2. 다중공선성 제거 후 남은 것
# 3. 상호작용 패턴 Feature는 보존

# 상호작용 패턴 Feature (무조건 포함)
interaction_features = [
    'gold_strict', 'gold_normal', 'triple_check', 'quad_check',
    'vertical_gold', 'cnv_imbalance', 'cnv_total_high', 'oht_overload',
    'm16_surge', 'inflow_surge', 'queue_accel_danger'
]

# 최종 선택
final_features = set(strong_features)

# 상호작용 패턴 추가
for f in interaction_features:
    if f in feature_cols:
        final_features.add(f)

# 다중공선성 제거
final_features = [f for f in final_features if f not in to_drop]

# 상관관계순 정렬
final_features = sorted(final_features, key=lambda x: correlations.get(x, 0), reverse=True)

print(f"\n🎯 최종 선택: {len(final_features)}개")
print("\n" + "-"*80)

for i, col in enumerate(final_features, 1):
    corr = correlations.get(col, 0)
    xgb_imp = importances.get(col, 0) if 'importances' in dir() else 0
    count = feature_counts.get(col, 0)
    print(f"  {i:3d}. {col[:55]:55s} 상관:{corr:.3f} XGB:{xgb_imp:.3f} 선택:{count}회")

# ==============================================================================
# 9. 결과 저장
# ==============================================================================

print("\n💾 결과 저장...")

# Feature 목록 저장
with open('step9_selected_features.txt', 'w', encoding='utf-8') as f:
    f.write(f"9단계 Feature Selection 결과\n")
    f.write(f"="*60 + "\n")
    f.write(f"총 선택: {len(final_features)}개\n\n")
    
    for i, col in enumerate(final_features, 1):
        corr = correlations.get(col, 0)
        f.write(f"{i:3d}. {col} (상관:{corr:.4f})\n")

print(f"  ✅ step9_selected_features.txt 저장")

# 상세 결과 저장
results_df = pd.DataFrame({
    'feature': feature_cols,
    'correlation': [correlations.get(c, 0) for c in feature_cols],
    'xgb_importance': [importances.get(c, 0) for c in feature_cols] if 'importances' in dir() else [0]*len(feature_cols),
    'et_importance': [et_importances.get(c, 0) for c in feature_cols] if 'et_importances' in dir() else [0]*len(feature_cols),
    'lasso_coef': [lasso_coef.get(c, 0) for c in feature_cols] if 'lasso_coef' in dir() else [0]*len(feature_cols),
    'selection_count': [feature_counts.get(c, 0) for c in feature_cols],
    'is_selected': [c in final_features for c in feature_cols]
})
results_df = results_df.sort_values('correlation', ascending=False)
results_df.to_csv('step9_feature_analysis.csv', index=False, encoding='utf-8-sig')
print(f"  ✅ step9_feature_analysis.csv 저장")

# 선택된 Feature만 있는 데이터셋 저장
print("\n📂 선택된 Feature 데이터셋 생성...")
df_full = pd.read_csv(INPUT_FILE)
selected_cols = final_features + META_COLS
selected_cols = [c for c in selected_cols if c in df_full.columns]
df_selected = df_full[selected_cols]
df_selected.to_csv('step9_selected_data.csv', index=False, encoding='utf-8-sig')
print(f"  ✅ step9_selected_data.csv 저장 ({len(df_selected):,}행 × {len(selected_cols)}컬럼)")

# ==============================================================================
# 10. 요약
# ==============================================================================

print("\n" + "="*80)
print("📊 9단계 요약")
print("="*80)
print(f"  원본 Feature: 346개")
print(f"  상관관계 >= 0.1: {len(high_corr_features)}개")
print(f"  다중공선성 제거: -{len(to_drop)}개")
print(f"  XGBoost 선택: {len(xgb_features)}개")
print(f"  ExtraTrees 선택: {len(et_features)}개")
print(f"  LASSO 선택: {len(lasso_features)}개")
print(f"  앙상블 (2개+): {len(ensemble_features)}개")
print(f"  앙상블 (3개+): {len(strong_features)}개")
print(f"  ─────────────────────────")
print(f"  🎯 최종 선택: {len(final_features)}개")

print("\n✅ 9단계 완료!")
print("   → 10단계: 선택된 Feature로 모델 학습")