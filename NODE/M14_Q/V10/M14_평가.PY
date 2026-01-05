# -*- coding: utf-8 -*-
"""
================================================================================
V10 ML 예측 모델 - 평가 코드
M14 Q 30분 내 리미트(1700) 초과 예측 시스템
================================================================================

사용법:
    python m14_v10_evaluate.py

입력:
    - models/v10_m14_model.pkl (학습된 모델)
    - 평가 데이터 파일

출력:
    - 평가 결과 리포트
    - results/ 폴더에 예측 결과 저장
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

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

print("=" * 80)
print("V10 ML 예측 모델 - 평가 시작")
print("=" * 80)

# ============================================================================
# 설정
# ============================================================================
EVAL_CONFIG = {
    'eval_file': 'M14_평가_20260101_20260102_C_109.CSV',
    'model_file': 'models/v10_m14_model.pkl',
    'result_dir': 'results'
}

os.makedirs(EVAL_CONFIG['result_dir'], exist_ok=True)

# ============================================================================
# 모델 로드
# ============================================================================
print("\n[1/5] 모델 로드 중...")

with open(EVAL_CONFIG['model_file'], 'rb') as f:
    save_data = pickle.load(f)

models = save_data['models']
pyod_iforest = save_data['pyod_iforest']
pyod_copod = save_data['pyod_copod']
scalers = save_data['scalers']
config = save_data['config']
feature_groups = save_data['feature_groups']
feature_indices = save_data['feature_indices']
training_info = save_data['training_info']

print(f"  - 모델 로드 완료")
print(f"  - 학습 일시: {training_info['train_date']}")
print(f"  - 학습 샘플 수: {training_info['total_samples']:,}")

# ============================================================================
# 평가 데이터 로드
# ============================================================================
print("\n[2/5] 평가 데이터 로드 중...")

df = pd.read_csv(EVAL_CONFIG['eval_file'], encoding='utf-8')
print(f"  - 원본 데이터: {len(df):,}행")

# CURRTIME을 datetime으로 변환
df['CURRTIME'] = pd.to_datetime(df['CURRTIME'], format='%Y%m%d%H%M')
df = df.sort_values('CURRTIME').reset_index(drop=True)

# 숫자형 컬럼만 선택
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df[numeric_cols] = df[numeric_cols].fillna(0)

# ============================================================================
# 시퀀스 Feature 생성
# ============================================================================
print("\n[3/5] 시퀀스 Feature 생성 중...")

seq_len = config['sequence_length']
pred_window = config['prediction_window']
limit_val = config['limit_value']
target_col = config['target_column']

def create_sequence_features(df, feature_cols, seq_len, idx, limit_val=1700):
    """특정 인덱스에서 seq_len 만큼의 시퀀스 Feature 추출 (급증 감지 강화)"""
    features = []
    for col in feature_cols:
        if col not in df.columns:
            # 컬럼이 없으면 0으로 채움 (기본 10개 + 급증감지 10개 = 20개)
            features.extend([0] * 20)
            continue
        seq = df[col].iloc[idx - seq_len:idx].values
        current_val = seq[-1]

        # 기본 통계 Feature
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

        # 급증 감지 Feature 추가
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

# 데이터 생성
print("  - 시퀀스 Feature 추출 중...")

X_target, X_important, X_auxiliary = [], [], []
y_regression, y_classification = [], []
timestamps = []
current_values = []  # 현재 시점 M14 Q 값

for idx in range(seq_len, len(df) - pred_window):
    max_val, breach = create_target(df, target_col, pred_window, limit_val, idx)
    if max_val is None:
        continue

    feat_target = create_sequence_features(df, feature_groups['target'], seq_len, idx)
    feat_important = create_sequence_features(df, feature_groups['important'], seq_len, idx)
    feat_auxiliary = create_sequence_features(df, feature_groups['auxiliary'], seq_len, idx)

    X_target.append(feat_target)
    X_important.append(feat_important)
    X_auxiliary.append(feat_auxiliary)
    y_regression.append(max_val)
    y_classification.append(breach)
    timestamps.append(df['CURRTIME'].iloc[idx])
    current_values.append(df[target_col].iloc[idx])  # 현재 시점 값 저장

X_target = np.array(X_target)
X_important = np.array(X_important)
X_auxiliary = np.array(X_auxiliary)
y_regression = np.array(y_regression)
y_classification = np.array(y_classification)

print(f"  - 총 샘플 수: {len(y_classification):,}")
print(f"  - 돌파 샘플 수: {np.sum(y_classification):,} ({100*np.mean(y_classification):.2f}%)")

# ============================================================================
# 스케일링
# ============================================================================
print("\n[4/5] 예측 수행 중...")

X_target_scaled = scalers['target'].transform(X_target)
X_important_scaled = scalers['important'].transform(X_important)
X_auxiliary_scaled = scalers['auxiliary'].transform(X_auxiliary)

X_all = np.hstack([X_target_scaled, X_important_scaled, X_auxiliary_scaled])

target_end = feature_indices['target_end']
important_end = feature_indices['important_end']
auxiliary_end = feature_indices['auxiliary_end']

# ============================================================================
# PyOD 이상 탐지
# ============================================================================
print("\n  [PyOD 이상 탐지]")

pyod_iforest_pred = pyod_iforest.predict(X_target_scaled)
pyod_copod_pred = pyod_copod.predict(X_target_scaled)

print(f"    - IsolationForest 이상 감지: {np.sum(pyod_iforest_pred):,}건 ({100*np.mean(pyod_iforest_pred):.2f}%)")
print(f"    - COPOD 이상 감지: {np.sum(pyod_copod_pred):,}건 ({100*np.mean(pyod_copod_pred):.2f}%)")

# ============================================================================
# 6개 모델 예측
# ============================================================================
print("\n  [6개 모델 예측]")

# XGBoost 회귀 예측
pred_xgb_target = models['xgb_target'].predict(X_all[:, :target_end])
pred_xgb_important = models['xgb_important'].predict(X_all[:, target_end:important_end])
pred_xgb_auxiliary = models['xgb_auxiliary'].predict(X_all[:, important_end:auxiliary_end])

# LightGBM 분류 예측
pred_lgb_target = models['lgb_target'].predict(X_all[:, :target_end])
pred_lgb_important = models['lgb_important'].predict(X_all[:, target_end:important_end])
pred_lgb_auxiliary = models['lgb_auxiliary'].predict(X_all[:, important_end:auxiliary_end])

# LightGBM 확률 예측
prob_lgb_target = models['lgb_target'].predict_proba(X_all[:, :target_end])[:, 1]
prob_lgb_important = models['lgb_important'].predict_proba(X_all[:, target_end:important_end])[:, 1]
prob_lgb_auxiliary = models['lgb_auxiliary'].predict_proba(X_all[:, important_end:auxiliary_end])[:, 1]

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

# 최종 판정 (다수결 + 추가 규칙)
# 규칙1: 다수결 4/6 이상
# 규칙2: LGBM_중요_확률 >= 0.95 AND 현재값 >= 1550 (FP 감소 강화)
# 규칙3: XGB_중요_예측값 >= 1720 AND 현재값 >= 1550 (과대예측 보정)
current_values_arr = np.array(current_values)

rule1 = (vote_sum >= 4)
rule2 = (prob_lgb_important >= 0.95) & (current_values_arr >= 1550)
rule3 = (pred_xgb_important >= 1720) & (current_values_arr >= 1550)

final_pred = (rule1 | rule2 | rule3).astype(int)

print(f"\n  [최종 판정 규칙]")
print(f"    - 규칙1) 다수결 4/6 이상: {rule1.sum()}건")
print(f"    - 규칙2) LGBM_중요_확률 >= 0.95 AND 현재값 >= 1550: {rule2.sum()}건")
print(f"    - 규칙3) XGB_중요_예측값 >= 1720 AND 현재값 >= 1550: {rule3.sum()}건")
print(f"    - 최종 돌파 예측: {final_pred.sum()}건")

# ============================================================================
# 성능 평가
# ============================================================================
print("\n[5/5] 성능 평가...")

print("\n" + "=" * 80)
print("평가 결과")
print("=" * 80)

print("\n[개별 모델 성능]")
model_results = {
    'XGB 타겟': xgb_target_breach,
    'XGB 중요': xgb_important_breach,
    'XGB 보조': xgb_auxiliary_breach,
    'LGBM 타겟': pred_lgb_target,
    'LGBM 중요': pred_lgb_important,
    'LGBM 보조': pred_lgb_auxiliary,
}

print(f"{'모델':<12} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 52)

for name, pred in model_results.items():
    acc = accuracy_score(y_classification, pred)
    prec = precision_score(y_classification, pred, zero_division=0)
    rec = recall_score(y_classification, pred, zero_division=0)
    f1 = f1_score(y_classification, pred, zero_division=0)
    print(f"{name:<12} {acc:>10.4f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f}")

print("\n[다수결 앙상블 성능]")
print("-" * 52)

acc = accuracy_score(y_classification, final_pred)
prec = precision_score(y_classification, final_pred, zero_division=0)
rec = recall_score(y_classification, final_pred, zero_division=0)
f1 = f1_score(y_classification, final_pred, zero_division=0)

print(f"{'다수결(4/6)':<12} {acc:>10.4f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f}")

# ROC-AUC (평균 확률 기반)
avg_prob = (prob_lgb_target + prob_lgb_important + prob_lgb_auxiliary) / 3
if len(np.unique(y_classification)) > 1:
    auc = roc_auc_score(y_classification, avg_prob)
    print(f"\n  ROC-AUC: {auc:.4f}")

print("\n[혼동 행렬]")
cm = confusion_matrix(y_classification, final_pred)
print(f"                예측 안전   예측 돌파")
print(f"  실제 안전     {cm[0,0]:>8,}   {cm[0,1]:>8,}")
print(f"  실제 돌파     {cm[1,0]:>8,}   {cm[1,1]:>8,}")

print("\n[분류 리포트]")
print(classification_report(y_classification, final_pred,
                          target_names=['안전(0)', '돌파(1)'],
                          zero_division=0))

# ============================================================================
# 결과 저장
# ============================================================================
print("\n" + "=" * 80)
print("결과 저장")
print("=" * 80)

# 상세 결과 DataFrame 생성 (한글 컬럼명)
# 예측 범위 시간 생성 (현재시간 ~ 현재시간+pred_window분)
pred_end_times = [t + pd.Timedelta(minutes=pred_window) for t in timestamps]
pred_range = [f"{s.strftime('%Y-%m-%d %H:%M')} ~ {e.strftime('%H:%M')}" for s, e in zip(timestamps, pred_end_times)]

result_df = pd.DataFrame({
    '현재시간': timestamps,
    '예측범위': pred_range,
    '현재_TOTALCNT': current_values,
    '실제_최대값': y_regression,
    '실제_돌파여부': y_classification,
    'XGB_타겟_예측값': pred_xgb_target,
    'XGB_중요_예측값': pred_xgb_important,
    'XGB_보조_예측값': pred_xgb_auxiliary,
    'LGBM_타겟_돌파': pred_lgb_target,
    'LGBM_중요_돌파': pred_lgb_important,
    'LGBM_보조_돌파': pred_lgb_auxiliary,
    'LGBM_타겟_확률': prob_lgb_target,
    'LGBM_중요_확률': prob_lgb_important,
    'LGBM_보조_확률': prob_lgb_auxiliary,
    '투표수_6개중': vote_sum,
    '최종예측_돌파여부': final_pred,
    'PyOD_IForest_이상': pyod_iforest_pred,
    'PyOD_COPOD_이상': pyod_copod_pred
})

# 예측 결과 저장
result_path = os.path.join(EVAL_CONFIG['result_dir'],
                          f'eval_result_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
result_df.to_csv(result_path, index=False, encoding='utf-8-sig')
print(f"  - 예측 결과 저장: {result_path}")

# 요약 리포트 저장
report_path = os.path.join(EVAL_CONFIG['result_dir'],
                          f'eval_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("V10 ML 예측 모델 평가 리포트\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"평가 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"평가 파일: {EVAL_CONFIG['eval_file']}\n")
    f.write(f"총 샘플 수: {len(y_classification):,}\n")
    f.write(f"돌파 샘플 수: {np.sum(y_classification):,} ({100*np.mean(y_classification):.2f}%)\n\n")

    f.write("=" * 80 + "\n")
    f.write("다수결 앙상블 성능\n")
    f.write("=" * 80 + "\n")
    f.write(f"  Accuracy:  {acc:.4f}\n")
    f.write(f"  Precision: {prec:.4f}\n")
    f.write(f"  Recall:    {rec:.4f}\n")
    f.write(f"  F1 Score:  {f1:.4f}\n")
    if len(np.unique(y_classification)) > 1:
        f.write(f"  ROC-AUC:   {auc:.4f}\n")

    f.write("\n[혼동 행렬]\n")
    f.write(f"  TN={cm[0,0]:,}, FP={cm[0,1]:,}\n")
    f.write(f"  FN={cm[1,0]:,}, TP={cm[1,1]:,}\n")

    f.write("\n[분류 리포트]\n")
    f.write(classification_report(y_classification, final_pred,
                                target_names=['안전(0)', '돌파(1)'],
                                zero_division=0))

print(f"  - 평가 리포트 저장: {report_path}")

# ============================================================================
# 오분류 분석
# ============================================================================
print("\n" + "=" * 80)
print("오분류 분석")
print("=" * 80)

# False Negative (돌파인데 안전으로 예측) - 가장 위험
fn_mask = (y_classification == 1) & (final_pred == 0)
fn_count = np.sum(fn_mask)
print(f"\n[False Negative (놓친 돌파)] : {fn_count}건")
if fn_count > 0:
    fn_df = result_df[fn_mask].head(10)
    print(f"  - 상위 {min(10, fn_count)}건:")
    for _, row in fn_df.iterrows():
        print(f"    {row['현재시간']} | 예측범위: {row['예측범위']} | 현재값: {row['현재_TOTALCNT']:.0f} | 실제최대: {row['실제_최대값']:.0f} | "
              f"투표: {row['투표수_6개중']}/6")

# False Positive (안전인데 돌파로 예측)
fp_mask = (y_classification == 0) & (final_pred == 1)
fp_count = np.sum(fp_mask)
print(f"\n[False Positive (잘못된 경고)] : {fp_count}건")
if fp_count > 0:
    fp_df = result_df[fp_mask].head(10)
    print(f"  - 상위 {min(10, fp_count)}건:")
    for _, row in fp_df.iterrows():
        print(f"    {row['현재시간']} | 예측범위: {row['예측범위']} | 현재값: {row['현재_TOTALCNT']:.0f} | 실제최대: {row['실제_최대값']:.0f} | "
              f"투표: {row['투표수_6개중']}/6")

print("\n" + "=" * 80)
print("평가 완료!")
print("=" * 80)