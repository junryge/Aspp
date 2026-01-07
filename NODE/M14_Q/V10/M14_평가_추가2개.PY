# -*- coding: utf-8 -*-
"""
================================================================================
V12 ML 예측 모델 - 평가 코드 (불필요 피처 제거)
M14 TOTALCNT 30분 내 리미트(1700) 초과 예측 시스템
================================================================================

V12 변경사항:
    - Important 그룹: 5개 컬럼 제거
    - Auxiliary 그룹: 11개 컬럼 제거

사용법:
    python m14_v12_evaluate.py
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# ============================================================================
# 설정
# ============================================================================
CONFIG = {
    'model_file': 'models/v12_m14_model.pkl',
    'eval_file': 'M14_평가_20260101_20260102_C_110_컬럼4개추가.CSV',  # 평가 파일명 수정 필요
    'output_file': f'v12_평가결과_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
    'sequence_length': 240,
    'prediction_window': 30,
    'limit_value': 1700,
    'target_column': 'TOTALCNT',
}

# Feature 그룹 (V12 - 불필요 피처 제거)
FEATURE_GROUPS = {
    'target': [
        'TOTALCNT'
    ],
    'important': [
        'M14.QUE.ALL.CURRENTQCOMPLETED',
        'M16HUB.QUE.M16TOM14.MESCURRENTQCNT',
        'M16HUB.QUE.M14TOM16.MESCURRENTQCNT',
        'M14.QUE.LOAD.AVGFOUPLOADTIME',
        'M14.QUE.LOAD.CURRENTLOADQCNT',
        'M14.QUE.LOAD.AVGLOADTIME',
        'M14.QUE.ALL.TRANSPORT4MINOVERRATIO',
        'M14.QUE.LOAD.AVGLOADTIME1MIN',
        'M16HUB.QUE.M16TOM14B.CURRENTQCREATED',
        'M16HUB.QUE.M14BTOM16.CURRENTQCREATED',
        'M16HUB.QUE.M14TOM16.CURRENTQCREATED',
        'M16HUB.QUE.M16TOM14.CURRENTQCREATED',
    ],
    'auxiliary': [
        'M10AM14A',
        'M14.QUE.CNV.SOUTHCURRENTQCNT',
        'M14AM14BSUM',
        'M14AM10A',
        'M14.QUE.CNV.ALLTOSOUTHCNVCURRENTQCNT',
        'M14AM14B',
        'M14.QUE.OHT.CURRENTOHTQCNT',
        'M14.QUE.CNV.ALLTONORTHCNVCURRENTQCNT',
        'M14AM10ASUM',
        'M14AM16SUM',
    ]
}

print("=" * 80)
print("V12 ML 예측 모델 - 평가 시작 (불필요 피처 제거)")
print("=" * 80)

# ============================================================================
# 모델 로드
# ============================================================================
print("\n[1/5] 모델 로드 중...")

with open(CONFIG['model_file'], 'rb') as f:
    model_data = pickle.load(f)

models = model_data['models']
scalers = model_data['scalers']
feature_indices = model_data['feature_indices']

print(f"  - 모델 버전: {model_data['training_info'].get('version', 'V12')}")
print(f"  - 학습일: {model_data['training_info']['train_date']}")

# ============================================================================
# 데이터 로드
# ============================================================================
print("\n[2/5] 평가 데이터 로드 중...")

try:
    df = pd.read_csv(CONFIG['eval_file'], encoding='utf-8')
except:
    try:
        df = pd.read_csv(CONFIG['eval_file'], encoding='cp949')
    except:
        df = pd.read_csv(CONFIG['eval_file'], encoding='euc-kr')

print(f"  - 원본 데이터: {len(df):,}행")

df['CURRTIME'] = pd.to_datetime(df['CURRTIME'], format='%Y%m%d%H%M')
df = df.sort_values('CURRTIME').reset_index(drop=True)

# Feature 그룹 필터링
for group_name in FEATURE_GROUPS:
    FEATURE_GROUPS[group_name] = [f for f in FEATURE_GROUPS[group_name] if f in df.columns]

# 숫자형 변환
all_feature_cols = FEATURE_GROUPS['target'] + FEATURE_GROUPS['important'] + FEATURE_GROUPS['auxiliary']
for col in all_feature_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

print(f"  - Important 컬럼 수: {len(FEATURE_GROUPS['important'])}개")
print(f"  - Auxiliary 컬럼 수: {len(FEATURE_GROUPS['auxiliary'])}개")

# ============================================================================
# Feature 생성 함수
# ============================================================================
def create_sequence_features(df, feature_cols, seq_len, idx, limit_val=1700):
    features = []
    for col in feature_cols:
        seq = df[col].iloc[idx - seq_len:idx].values
        current_val = seq[-1]

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

# ============================================================================
# 평가 수행
# ============================================================================
print("\n[3/5] 예측 수행 중...")

seq_len = CONFIG['sequence_length']
pred_window = CONFIG['prediction_window']
limit_val = CONFIG['limit_value']
target_col = CONFIG['target_column']

target_end = feature_indices['target_end']
important_end = feature_indices['important_end']

results = []

for idx in range(seq_len, len(df)):
    if idx % 1000 == 0:
        print(f"    진행률: {idx:,}/{len(df):,} ({100*idx/len(df):.1f}%)")

    current_time = df['CURRTIME'].iloc[idx]
    current_val = df[target_col].iloc[idx]

    # 실제 30분 후 최대값
    future_end = min(idx + pred_window, len(df))
    future_vals = df[target_col].iloc[idx:future_end].values
    actual_max = np.max(future_vals) if len(future_vals) > 0 else current_val
    actual_breach = 1 if actual_max >= limit_val else 0

    # Feature 생성
    feat_target = create_sequence_features(df, FEATURE_GROUPS['target'], seq_len, idx)
    feat_important = create_sequence_features(df, FEATURE_GROUPS['important'], seq_len, idx)
    feat_auxiliary = create_sequence_features(df, FEATURE_GROUPS['auxiliary'], seq_len, idx)

    X_target = scalers['target'].transform([feat_target])
    X_important = scalers['important'].transform([feat_important])
    X_auxiliary = scalers['auxiliary'].transform([feat_auxiliary])

    # 예측
    pred_xgb_target = models['xgb_target'].predict(X_target)[0]
    pred_xgb_important = models['xgb_important'].predict(X_important)[0]
    pred_xgb_auxiliary = models['xgb_auxiliary'].predict(X_auxiliary)[0]

    pred_lgb_target = models['lgb_target'].predict(X_target)[0]
    pred_lgb_important = models['lgb_important'].predict(X_important)[0]
    pred_lgb_auxiliary = models['lgb_auxiliary'].predict(X_auxiliary)[0]

    prob_lgb_target = models['lgb_target'].predict_proba(X_target)[0][1]
    prob_lgb_important = models['lgb_important'].predict_proba(X_important)[0][1]
    prob_lgb_auxiliary = models['lgb_auxiliary'].predict_proba(X_auxiliary)[0][1]

    # 투표
    votes = [
        1 if pred_xgb_target >= limit_val else 0,
        1 if pred_xgb_important >= limit_val else 0,
        1 if pred_xgb_auxiliary >= limit_val else 0,
        pred_lgb_target,
        pred_lgb_important,
        pred_lgb_auxiliary
    ]
    vote_sum = sum(votes)

    # # 최종 판정 규칙
    # rule1 = vote_sum >= 4
    # rule2 = (prob_lgb_important >= 0.92) and (current_val >= 1520)
    # rule3 = (pred_xgb_important >= 1710) and (current_val >= 1520)
    # rule4 = (current_val >= 1650) and (vote_sum >= 3)
   
    rule1 = vote_sum >= 3                                              # 4→3
    rule2 = (prob_lgb_important >= 0.50) and (current_val >= 1550)     # 0.92→0.50
    rule3 = (pred_xgb_important >= 1680) and (current_val >= 1500)     # 1710→1680
    rule4 = (current_val >= 1600) and (vote_sum >= 2)                  # 1650→1600, 3→2

    final_pred = 1 if (rule1 or rule2 or rule3 or rule4) else 0

    # 예측 범위
    pred_start = current_time
    pred_end = current_time + timedelta(minutes=30)
    pred_range = f"{pred_start.strftime('%Y-%m-%d %H:%M')} ~ {pred_end.strftime('%H:%M')}"

    results.append({
        '현재시간': current_time.strftime('%Y-%m-%d %H:%M'),
        '예측범위': pred_range,
        '현재_TOTALCNT': current_val,
        '실제_최대값': actual_max,
        '실제_돌파여부': actual_breach,
        'XGB_타겟_예측값': round(pred_xgb_target, 1),
        'XGB_중요_예측값': round(pred_xgb_important, 1),
        'XGB_보조_예측값': round(pred_xgb_auxiliary, 1),
        'LGBM_타겟_확률': round(prob_lgb_target, 3),
        'LGBM_중요_확률': round(prob_lgb_important, 3),
        'LGBM_보조_확률': round(prob_lgb_auxiliary, 3),
        '투표수_6개중': vote_sum,
        '최종예측_돌파여부': final_pred,
    })

result_df = pd.DataFrame(results)

# ============================================================================
# 예측상태 분류 (10분 전/후 분석)
# ============================================================================
print("\n[4/5] 예측상태 분류 중 (10분 전/후 분석)...")

result_df['현재시간_dt'] = pd.to_datetime(result_df['현재시간'])

def get_prediction_status(row, all_df):
    actual = row['실제_돌파여부']
    pred = row['최종예측_돌파여부']
    current_time = row['현재시간_dt']
   
    if actual == 1 and pred == 1:
        return '정상예측_TP'
    elif actual == 0 and pred == 0:
        return '정상예측_TN'
    elif actual == 1 and pred == 0:
        time_10min_ago = current_time - timedelta(minutes=10)
        prev_data = all_df[all_df['현재시간_dt'] == time_10min_ago]
        if len(prev_data) > 0 and prev_data['최종예측_돌파여부'].values[0] == 1:
            return 'FN_10분전예측'
        else:
            return 'FN_완전놓침'
    else:
        time_10min_later = current_time + timedelta(minutes=10)
        later_data = all_df[all_df['현재시간_dt'] == time_10min_later]
        if len(later_data) > 0 and later_data['실제_돌파여부'].values[0] == 1:
            return 'FP_10분후돌파'
        else:
            return 'FP_잘못된경고'

result_df['예측상태'] = result_df.apply(lambda row: get_prediction_status(row, result_df), axis=1)
result_df = result_df.drop(columns=['현재시간_dt'])

# ============================================================================
# 성능 평가
# ============================================================================
print("\n[5/5] 성능 평가...")

actual = result_df['실제_돌파여부'].values
pred = result_df['최종예측_돌파여부'].values

TP = ((actual == 1) & (pred == 1)).sum()
TN = ((actual == 0) & (pred == 0)).sum()
FP = ((actual == 0) & (pred == 1)).sum()
FN = ((actual == 1) & (pred == 0)).sum()

accuracy = (TP + TN) / len(actual) if len(actual) > 0 else 0
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print("\n" + "=" * 80)
print("V12 평가 결과 (불필요 피처 제거)")
print("=" * 80)

print(f"\n[기본 성능]")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1 Score:  {f1:.4f}")

print(f"\n[혼동 행렬]")
print(f"  TP (정확한 돌파 예측): {TP:,}건")
print(f"  TN (정확한 안전 예측): {TN:,}건")
print(f"  FP (잘못된 경고):      {FP:,}건")
print(f"  FN (놓친 돌파):        {FN:,}건")

# 예측상태별 집계
print(f"\n[예측상태 분류]")
status_counts = result_df['예측상태'].value_counts()
for status, count in status_counts.items():
    print(f"  {status}: {count:,}건")

# 실질 FN/FP
real_fn = status_counts.get('FN_완전놓침', 0)
real_fp = status_counts.get('FP_잘못된경고', 0)

print(f"\n[실질적 성능 (10분 기준)]")
print(f"  실질 FN (완전 놓침):    {real_fn:,}건")
print(f"  실질 FP (잘못된 경고):  {real_fp:,}건")

# V10 대비 비교
print(f"\n[V10 대비 비교]")
print(f"  V10: 실질 FN=7건, 실질 FP=338건")
print(f"  V12: 실질 FN={real_fn}건, 실질 FP={real_fp}건")

# ============================================================================
# 결과 저장
# ============================================================================
result_df.to_csv(CONFIG['output_file'], index=False, encoding='utf-8-sig')
print(f"\n결과 저장: {CONFIG['output_file']}")

print("\n" + "=" * 80)
print("V12 평가 완료!")
print("=" * 80)