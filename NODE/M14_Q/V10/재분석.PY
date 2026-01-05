# -*- coding: utf-8 -*-
"""
================================================================================
Feature Importance 분석 - 불필요한 피처 찾기
================================================================================
"""

import pickle
import numpy as np
import pandas as pd

# ============================================================================
# 설정
# ============================================================================
MODEL_FILE = 'models/v10_m14_model.pkl'  # V10 모델 파일 경로

# Feature 그룹 (V10 기준)
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
    ]
}

# 피처 이름 (컬럼당 20개 통계)
STAT_NAMES = [
    'mean', 'std', 'min', 'max', 'current', 'change_240m',
    'pct25', 'pct75', 'trend', 'max_30m',
    'change_10m', 'change_30m', 'change_60m', 'slope_10m', 'slope_30m',
    'vol_10m', 'vol_30m', 'dist_limit', 'above_1500', 'above_1600'
]

print("=" * 80)
print("Feature Importance 분석")
print("=" * 80)

# ============================================================================
# 모델 로드
# ============================================================================
print("\n[1/3] 모델 로드 중...")

with open(MODEL_FILE, 'rb') as f:
    model_data = pickle.load(f)

models = model_data['models']
print(f"  - 로드 완료")

# ============================================================================
# Feature Importance 분석
# ============================================================================
print("\n[2/3] Feature Importance 분석...")

def analyze_importance(model, model_name, col_list):
    """모델별 Feature Importance 분석"""
    importance = model.feature_importances_
    
    # 피처 이름 생성
    feature_names = []
    for col in col_list:
        col_short = col.split('.')[-1][:20] if '.' in col else col[:20]
        for stat in STAT_NAMES:
            feature_names.append(f"{col_short}_{stat}")
    
    # DataFrame 생성
    df = pd.DataFrame({
        'feature': feature_names[:len(importance)],
        'importance': importance
    })
    df = df.sort_values('importance', ascending=False)
    
    print(f"\n[{model_name}]")
    print(f"  총 피처 수: {len(importance)}")
    print(f"  importance = 0: {(importance == 0).sum()}개")
    print(f"  importance < 0.001: {(importance < 0.001).sum()}개")
    
    print(f"\n  ★ Top 15 피처:")
    for i, row in df.head(15).iterrows():
        print(f"    {row['feature']:<35} {row['importance']:.4f}")
    
    print(f"\n  ✗ Bottom 15 피처 (기여도 낮음):")
    for i, row in df.tail(15).iterrows():
        print(f"    {row['feature']:<35} {row['importance']:.6f}")
    
    return df

# XGB Important 모델 분석 (가장 중요)
df_important = analyze_importance(
    models['xgb_important'], 
    'XGB_Important (핵심)', 
    FEATURE_GROUPS['important']
)

# XGB Target 모델 분석
df_target = analyze_importance(
    models['xgb_target'], 
    'XGB_Target', 
    FEATURE_GROUPS['target']
)

# XGB Auxiliary 모델 분석
df_auxiliary = analyze_importance(
    models['xgb_auxiliary'], 
    'XGB_Auxiliary', 
    FEATURE_GROUPS['auxiliary']
)

# ============================================================================
# 컬럼별 합산 Importance
# ============================================================================
print("\n" + "=" * 80)
print("[3/3] 컬럼별 합산 Importance (XGB_Important)")
print("=" * 80)

# 컬럼별로 20개 피처 importance 합산
col_importance = {}
importance = models['xgb_important'].feature_importances_

for i, col in enumerate(FEATURE_GROUPS['important']):
    col_short = col.split('.')[-1][:25] if '.' in col else col[:25]
    start_idx = i * 20
    end_idx = start_idx + 20
    if end_idx <= len(importance):
        col_importance[col_short] = importance[start_idx:end_idx].sum()

# 정렬
sorted_cols = sorted(col_importance.items(), key=lambda x: x[1], reverse=True)

print(f"\n{'컬럼':<30} {'합산 Importance':>15} {'판정'}")
print("-" * 55)

for col, imp in sorted_cols:
    if imp < 0.01:
        status = "❌ 제거 검토"
    elif imp < 0.03:
        status = "⚠️ 낮음"
    else:
        status = "✓"
    print(f"{col:<30} {imp:>15.4f} {status}")

# Auxiliary 그룹도 분석
print("\n" + "=" * 80)
print("컬럼별 합산 Importance (XGB_Auxiliary)")
print("=" * 80)

col_importance_aux = {}
importance_aux = models['xgb_auxiliary'].feature_importances_

for i, col in enumerate(FEATURE_GROUPS['auxiliary']):
    col_short = col.split('.')[-1][:25] if '.' in col else col[:25]
    start_idx = i * 20
    end_idx = start_idx + 20
    if end_idx <= len(importance_aux):
        col_importance_aux[col_short] = importance_aux[start_idx:end_idx].sum()

sorted_cols_aux = sorted(col_importance_aux.items(), key=lambda x: x[1], reverse=True)

print(f"\n{'컬럼':<30} {'합산 Importance':>15} {'판정'}")
print("-" * 55)

for col, imp in sorted_cols_aux:
    if imp < 0.01:
        status = "❌ 제거 검토"
    elif imp < 0.03:
        status = "⚠️ 낮음"
    else:
        status = "✓"
    print(f"{col:<30} {imp:>15.4f} {status}")

print("\n" + "=" * 80)
print("분석 완료!")
print("=" * 80)
print("\n★ 제거 검토 대상: 합산 Importance < 0.01인 컬럼들")