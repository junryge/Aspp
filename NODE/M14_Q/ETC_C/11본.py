#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
🚀 11단계: 모델 후보군 선정 및 성능지표 수립
================================================================================
여러 모델 비교하여 최적 모델 찾기
- XGBoost
- LightGBM  
- ExtraTrees
- RandomForest
- 규칙 기반 (V8.3 방식)
- 앙상블 (ML + 규칙)
================================================================================
"""

import numpy as np
import pandas as pd
import pickle
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

warnings.filterwarnings('ignore')

print("="*80)
print("🚀 11단계: 모델 후보군 선정 및 성능지표 수립")
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

# ==============================================================================
# 2. 데이터 로드
# ==============================================================================

print("\n📂 데이터 로딩...")
import os
if os.path.exists('step9_selected_data.csv'):
    df = pd.read_csv('step9_selected_data.csv')
    print("  ✅ step9_selected_data.csv 사용")
else:
    use_cols = SELECTED_FEATURES + ['target_TOTALCNT', 'is_danger', 'current_TOTALCNT']
    df = pd.read_csv('step8_features_10min.csv', usecols=use_cols)
    print("  ✅ step8_features_10min.csv 사용")

print(f"  ✅ 로드 완료: {len(df):,}행")

# 결측치 처리
df = df.fillna(0).replace([np.inf, -np.inf], 0)

# X, y 분리
X = df[SELECTED_FEATURES]
y = df['target_TOTALCNT']

# 시간순 분리 (80/20)
split_idx = int(len(X) * 0.8)
X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"  학습: {len(X_train):,}개 / 검증: {len(X_val):,}개")

# ==============================================================================
# 3. 성능 지표 정의
# ==============================================================================

def evaluate_model(y_true, y_pred, model_name=""):
    """모델 성능 평가"""
    results = {}
    
    # 기본 메트릭
    results['MAE'] = mean_absolute_error(y_true, y_pred)
    results['RMSE'] = np.sqrt(mean_squared_error(y_true, y_pred))
    results['R2'] = r2_score(y_true, y_pred)
    
    # 1700+ 감지 성능
    actual_danger = y_true >= 1700
    pred_danger = y_pred >= 1700
    
    tp = ((actual_danger) & (pred_danger)).sum()
    fn = ((actual_danger) & (~pred_danger)).sum()
    fp = ((~actual_danger) & (pred_danger)).sum()
    
    results['TP'] = tp
    results['FN'] = fn
    results['FP'] = fp
    results['Recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    results['Precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    results['F1'] = 2 * results['Precision'] * results['Recall'] / (results['Precision'] + results['Recall']) if (results['Precision'] + results['Recall']) > 0 else 0
    
    # 구간별 MAE
    for low, high, name in [(0, 1600, 'MAE_0_1600'), (1600, 1700, 'MAE_1600_1700'), (1700, 9999, 'MAE_1700+')]:
        mask = (y_true >= low) & (y_true < high)
        if mask.sum() > 0:
            results[name] = mean_absolute_error(y_true[mask], y_pred[mask])
        else:
            results[name] = 0
    
    return results

def print_results(results, model_name):
    """결과 출력"""
    print(f"\n{'─'*60}")
    print(f"📊 {model_name}")
    print(f"{'─'*60}")
    print(f"  MAE: {results['MAE']:.2f} | RMSE: {results['RMSE']:.2f} | R²: {results['R2']:.4f}")
    print(f"  🔥 1700+ 감지율: {results['Recall']*100:.1f}% | 정밀도: {results['Precision']*100:.1f}% | F1: {results['F1']:.4f}")
    print(f"  TP: {results['TP']} | FN: {results['FN']} | FP: {results['FP']}")

# ==============================================================================
# 4. 가중치 설정
# ==============================================================================

weights = np.ones(len(y_train))
weights[y_train >= 1500] = 3
weights[y_train >= 1600] = 10
weights[y_train >= 1700] = 50   # 30 → 50으로 증가
weights[y_train >= 1800] = 100  # 50 → 100으로 증가

# ==============================================================================
# 5. 모델 후보군 학습
# ==============================================================================

all_results = {}

# -----------------------------------------------------------------------------
# 모델 1: XGBoost
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("🔷 모델 1: XGBoost")
print("="*80)

try:
    import xgboost as xgb
    
    model_xgb = xgb.XGBRegressor(
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        n_jobs=-1,
        random_state=42
    )
    model_xgb.fit(X_train, y_train, sample_weight=weights, verbose=False)
    y_pred_xgb = model_xgb.predict(X_val)
    
    results_xgb = evaluate_model(y_val, y_pred_xgb, "XGBoost")
    print_results(results_xgb, "XGBoost")
    all_results['XGBoost'] = results_xgb
    
except Exception as e:
    print(f"  ❌ XGBoost 오류: {e}")

# -----------------------------------------------------------------------------
# 모델 2: LightGBM
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("🔷 모델 2: LightGBM")
print("="*80)

try:
    import lightgbm as lgb
    
    model_lgb = lgb.LGBMRegressor(
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
        verbose=-1
    )
    model_lgb.fit(X_train, y_train, sample_weight=weights)
    y_pred_lgb = model_lgb.predict(X_val)
    
    results_lgb = evaluate_model(y_val, y_pred_lgb, "LightGBM")
    print_results(results_lgb, "LightGBM")
    all_results['LightGBM'] = results_lgb
    
except Exception as e:
    print(f"  ❌ LightGBM 오류: {e}")

# -----------------------------------------------------------------------------
# 모델 3: ExtraTrees
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("🔷 모델 3: ExtraTrees")
print("="*80)

try:
    model_et = ExtraTreesRegressor(
        n_estimators=500,
        max_depth=15,
        min_samples_split=5,
        n_jobs=-1,
        random_state=42
    )
    model_et.fit(X_train, y_train, sample_weight=weights)
    y_pred_et = model_et.predict(X_val)
    
    results_et = evaluate_model(y_val, y_pred_et, "ExtraTrees")
    print_results(results_et, "ExtraTrees")
    all_results['ExtraTrees'] = results_et
    
except Exception as e:
    print(f"  ❌ ExtraTrees 오류: {e}")

# -----------------------------------------------------------------------------
# 모델 4: RandomForest
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("🔷 모델 4: RandomForest")
print("="*80)

try:
    model_rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=15,
        min_samples_split=5,
        n_jobs=-1,
        random_state=42
    )
    model_rf.fit(X_train, y_train, sample_weight=weights)
    y_pred_rf = model_rf.predict(X_val)
    
    results_rf = evaluate_model(y_val, y_pred_rf, "RandomForest")
    print_results(results_rf, "RandomForest")
    all_results['RandomForest'] = results_rf
    
except Exception as e:
    print(f"  ❌ RandomForest 오류: {e}")

# -----------------------------------------------------------------------------
# 모델 5: 규칙 기반 (V8.3 방식)
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("🔷 모델 5: 규칙 기반 (V8.3 스타일)")
print("="*80)

def rule_based_predict(X, y_current=None):
    """V8.3 스타일 규칙 기반 예측"""
    predictions = []
    
    for idx in range(len(X)):
        row = X.iloc[idx]
        
        # 기본값: 최근 10분 평균
        pred = row['total_last10']
        
        # 추세 반영
        pred += row['total_slope'] * 10  # 10분 후 예측
        
        # 황금 패턴 boost
        if row['gold_strict'] == 1:
            pred += 80
        elif row['gold_normal'] == 1:
            pred += 60
        
        # triple/quad check
        if row['quad_check'] == 1:
            pred += 50
        elif row['triple_check'] == 1:
            pred += 40
        
        # 신규 패턴 boost
        if row['vertical_gold'] == 1:
            pred += 50
        if row['inflow_surge'] == 1:
            pred += 40
        if row['oht_overload'] == 1:
            pred += 30
        
        # 현재 1600 이상이면 boost
        if row['total_over_1600'] > 0:
            pred += 30
        
        predictions.append(pred)
    
    return np.array(predictions)

y_pred_rule = rule_based_predict(X_val)
results_rule = evaluate_model(y_val, y_pred_rule, "Rule-Based")
print_results(results_rule, "규칙 기반 (V8.3)")
all_results['Rule-Based'] = results_rule

# -----------------------------------------------------------------------------
# 모델 6: 앙상블 (XGBoost + 규칙 기반)
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("🔷 모델 6: 앙상블 (XGBoost + 규칙)")
print("="*80)

if 'y_pred_xgb' in dir():
    # XGBoost 예측에 규칙 기반 보정 추가
    y_pred_ensemble = y_pred_xgb.copy()
    
    for idx in range(len(X_val)):
        row = X_val.iloc[idx]
        pred = y_pred_ensemble[idx]
        
        # 1650~1700 구간에서 boost 적용
        if 1650 <= pred < 1700:
            boost = 0
            if row['gold_strict'] == 1:
                boost += 60
            elif row['gold_normal'] == 1:
                boost += 50
            if row['triple_check'] == 1:
                boost += 40
            if row['vertical_gold'] == 1:
                boost += 40
            if row['inflow_surge'] == 1:
                boost += 30
            
            y_pred_ensemble[idx] = pred + boost
        
        # 현재 1600 이상 + 패턴 있으면 boost
        elif 1600 <= pred < 1650:
            if row['gold_normal'] == 1 or row['vertical_gold'] == 1:
                y_pred_ensemble[idx] = pred + 50
    
    results_ensemble = evaluate_model(y_val, y_pred_ensemble, "Ensemble")
    print_results(results_ensemble, "앙상블 (XGB + Rule)")
    all_results['Ensemble'] = results_ensemble

# -----------------------------------------------------------------------------
# 모델 7: 앙상블 v2 (더 공격적)
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("🔷 모델 7: 앙상블 v2 (공격적)")
print("="*80)

if 'y_pred_xgb' in dir():
    y_pred_ensemble2 = y_pred_xgb.copy()
    
    for idx in range(len(X_val)):
        row = X_val.iloc[idx]
        pred = y_pred_ensemble2[idx]
        
        # 1600 이상이면 무조건 boost 검토
        if pred >= 1600:
            boost = 0
            if row['gold_strict'] == 1:
                boost += 80
            elif row['gold_normal'] == 1:
                boost += 60
            if row['triple_check'] == 1:
                boost += 50
            if row['quad_check'] == 1:
                boost += 30
            if row['vertical_gold'] == 1:
                boost += 50
            if row['inflow_surge'] == 1:
                boost += 40
            if row['oht_overload'] == 1:
                boost += 30
            if row['m16_surge'] == 1:
                boost += 20
            
            y_pred_ensemble2[idx] = pred + boost
    
    results_ensemble2 = evaluate_model(y_val, y_pred_ensemble2, "Ensemble_v2")
    print_results(results_ensemble2, "앙상블 v2 (공격적)")
    all_results['Ensemble_v2'] = results_ensemble2

# ==============================================================================
# 6. 모델 비교 요약
# ==============================================================================

print("\n" + "="*80)
print("📊 모델 비교 요약")
print("="*80)

# 결과 테이블 생성
comparison_df = pd.DataFrame(all_results).T
comparison_df = comparison_df[['MAE', 'RMSE', 'R2', 'Recall', 'Precision', 'F1', 'TP', 'FN', 'FP']]
comparison_df['Recall'] = comparison_df['Recall'] * 100
comparison_df['Precision'] = comparison_df['Precision'] * 100

print("\n" + comparison_df.round(2).to_string())

# 감지율 기준 정렬
print("\n\n🏆 1700+ 감지율 순위:")
recall_rank = comparison_df.sort_values('Recall', ascending=False)
for i, (name, row) in enumerate(recall_rank.iterrows(), 1):
    star = "⭐" if row['Recall'] >= 70 else ""
    print(f"  {i}. {name:20s} 감지율: {row['Recall']:.1f}% | 정밀도: {row['Precision']:.1f}% | MAE: {row['MAE']:.1f} {star}")

# F1 기준 정렬  
print("\n\n🏆 F1 Score 순위:")
f1_rank = comparison_df.sort_values('F1', ascending=False)
for i, (name, row) in enumerate(f1_rank.iterrows(), 1):
    print(f"  {i}. {name:20s} F1: {row['F1']:.4f} | 감지율: {row['Recall']:.1f}% | 정밀도: {row['Precision']:.1f}%")

# ==============================================================================
# 7. 결과 저장
# ==============================================================================

print("\n💾 결과 저장...")

comparison_df.to_csv('step11_model_comparison.csv', encoding='utf-8-sig')
print("  ✅ step11_model_comparison.csv")

# 최고 감지율 모델 저장
best_recall_model = recall_rank.index[0]
print(f"\n🏆 최고 감지율 모델: {best_recall_model} ({recall_rank.iloc[0]['Recall']:.1f}%)")

# ==============================================================================
# 8. 성능 지표 정의 문서
# ==============================================================================

print("\n" + "="*80)
print("📋 성능 지표 정의")
print("="*80)
print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│ 지표         │ 설명                           │ 목표        │ 우선순위    │
├─────────────────────────────────────────────────────────────────────────────┤
│ Recall       │ 실제 1700+ 중 감지한 비율      │ ≥ 85%      │ ⭐⭐⭐ 최우선│
│ Precision    │ 예측 1700+ 중 실제 1700+ 비율  │ ≥ 60%      │ ⭐⭐ 중요   │
│ F1 Score     │ Recall × Precision 조화평균    │ ≥ 0.70     │ ⭐⭐ 중요   │
│ MAE          │ 평균 절대 오차                 │ ≤ 50       │ ⭐ 참고    │
│ MAE(1700+)   │ 1700+ 구간 MAE                 │ ≤ 80       │ ⭐⭐ 중요   │
└─────────────────────────────────────────────────────────────────────────────┘

※ 감지율(Recall)이 가장 중요! 놓치면 공장 사고!
※ 오탐(FP)은 어느 정도 허용 가능 (경고만 하면 됨)
""")

print("\n✅ 11단계 완료!")
print("   → 12단계: 최적 모델 선정 및 상세 학습")