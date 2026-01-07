# -*- coding: utf-8 -*-
"""
================================================================================
V12 ML 예측 모델 - 수정 평가 코드
판정 규칙 민감도 높임 (FN 감소, FP 증가 예상)
================================================================================
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
    'eval_file': 'M14_평가_20250101_20250104.CSV',
    'output_file': f'V12_수정_평가결과_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
    'sequence_length': 240,
    'prediction_window': 30,
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
        'M14.QUE.ALL.CURRENTQCOMPLETED',
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
        'M14.QUE.SFAB.SENDQUEUETOTAL',
        'M14.QUE.SFAB.RECEIVEQUEUETOTAL',
        'M14.QUE.CNV.SOUTHCURRENTQCNT',
        'M14.QUE.CNV.ALLTOSOUTHCNVCURRENTQCNT',
    ],
}

print("=" * 70)
print("V12 ML 예측 모델 - 수정 평가 (판정 규칙 민감도 ↑)")
print("=" * 70)

# ============================================================================
# 모델 로드
# ============================================================================
print("\n[1/5] 모델 로드 중...")

with open(CONFIG['model_file'], 'rb') as f:
    model_data = pickle.load(f)

models = model_data['models']
scalers = model_data['scalers']

print(f"  - 모델 버전: {model_data.get('version', 'V12')}")
print(f"  - 학습일: {model_data.get('trained_at', 'N/A')}")

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

df['CURRTIME'] = pd.to_datetime(df['CURRTIME'], format='%Y%m%d%H%M')
df = df.sort_values('CURRTIME').reset_index(drop=True)

print(f"  - 원본 데이터: {len(df):,}행")

for group_name in FEATURE_GROUPS:
    FEATURE_GROUPS[group_name] = [f for f in FEATURE_GROUPS[group_name] if f in df.columns]
    print(f"  - {group_name} 컬럼 수: {len(FEATURE_GROUPS[group_name])}개")

all_cols = []
for group in FEATURE_GROUPS.values():
    all_cols.extend(group)
all_cols = list(set(all_cols))

for col in all_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# ============================================================================
# 시퀀스 피처 생성
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
# 예측 수행
# ============================================================================
print("\n[3/5] 예측 수행 중...")

seq_len = CONFIG['sequence_length']
pred_window = CONFIG['prediction_window']
limit_val = CONFIG['limit_value']
target_col = CONFIG['target_column']

results = []

for idx in range(seq_len, len(df)):
    if idx % 1000 == 0:
        print(f"    진행률: {idx:,}/{len(df):,} ({100*idx/len(df):.1f}%)")
    
    current_time = df['CURRTIME'].iloc[idx]
    current_val = df[target_col].iloc[idx]
    
    future_end = min(idx + pred_window, len(df))
    future_vals = df[target_col].iloc[idx:future_end].values
    actual_max = np.max(future_vals) if len(future_vals) > 0 else current_val
    actual_breach = 1 if actual_max >= limit_val else 0
    
    feat_target = create_sequence_features(df, FEATURE_GROUPS['target'], seq_len, idx)
    feat_important = create_sequence_features(df, FEATURE_GROUPS['important'], seq_len, idx)
    feat_auxiliary = create_sequence_features(df, FEATURE_GROUPS['auxiliary'], seq_len, idx)
    
    X_target = scalers['target'].transform([feat_target])
    X_important = scalers['important'].transform([feat_important])
    X_auxiliary = scalers['auxiliary'].transform([feat_auxiliary])
    
    pred_xgb_target = models['xgb_target'].predict(X_target)[0]
    pred_xgb_important = models['xgb_important'].predict(X_important)[0]
    pred_xgb_auxiliary = models['xgb_auxiliary'].predict(X_auxiliary)[0]
    
    pred_lgb_target = models['lgb_target'].predict(X_target)[0]
    pred_lgb_important = models['lgb_important'].predict(X_important)[0]
    pred_lgb_auxiliary = models['lgb_auxiliary'].predict(X_auxiliary)[0]
    
    prob_lgb_target = models['lgb_target'].predict_proba(X_target)[0][1]
    prob_lgb_important = models['lgb_important'].predict_proba(X_important)[0][1]
    prob_lgb_auxiliary = models['lgb_auxiliary'].predict_proba(X_auxiliary)[0][1]
    
    # 투표 (6개 모델)
    votes = [
        1 if pred_xgb_target >= limit_val else 0,
        1 if pred_xgb_important >= limit_val else 0,
        1 if pred_xgb_auxiliary >= limit_val else 0,
        pred_lgb_target,
        pred_lgb_important,
        pred_lgb_auxiliary,
    ]
    vote_sum = sum(votes)
    
    # ================================================================
    # ★ 수정된 판정 규칙 (민감도 높임)
    # ================================================================
    rule1 = vote_sum >= 3                                              # 4→3
    rule2 = (prob_lgb_important >= 0.50) and (current_val >= 1550)     # 0.92→0.50
    rule3 = (pred_xgb_important >= 1680) and (current_val >= 1500)     # 1710→1680
    rule4 = (current_val >= 1600) and (vote_sum >= 2)                  # 1650→1600, 3→2
    
    final_pred = 1 if (rule1 or rule2 or rule3 or rule4) else 0
    
    pred_end = current_time + timedelta(minutes=30)
    
    results.append({
        '현재시간': current_time.strftime('%Y-%m-%d %H:%M'),
        '예측범위': f"{current_time.strftime('%Y-%m-%d %H:%M')} ~ {pred_end.strftime('%H:%M')}",
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

# ============================================================================
# 예측상태 분류
# ============================================================================
print("\n[4/5] 예측상태 분류 중...")

result_df = pd.DataFrame(results)
result_df['현재시간_dt'] = pd.to_datetime(result_df['현재시간'])

def get_status(row, all_df):
    actual = row['실제_돌파여부']
    pred = row['최종예측_돌파여부']
    current_time = row['현재시간_dt']
    
    if actual == 1 and pred == 1:
        return '정상예측_TP'
    elif actual == 0 and pred == 0:
        return '정상예측_TN'
    elif actual == 1 and pred == 0:
        prev = all_df[all_df['현재시간_dt'] == current_time - timedelta(minutes=10)]
        if len(prev) > 0 and prev['최종예측_돌파여부'].values[0] == 1:
            return 'FN_10분전예측'
        return 'FN_완전놓침'
    else:
        later = all_df[all_df['현재시간_dt'] == current_time + timedelta(minutes=10)]
        if len(later) > 0 and later['실제_돌파여부'].values[0] == 1:
            return 'FP_10분후돌파'
        return 'FP_잘못된경고'

result_df['예측상태'] = result_df.apply(lambda row: get_status(row, result_df), axis=1)
result_df = result_df.drop(columns=['현재시간_dt'])

# ============================================================================
# 성능 평가
# ============================================================================
print("\n[5/5] 성능 평가...")

status_counts = result_df['예측상태'].value_counts()

tp = status_counts.get('정상예측_TP', 0)
tn = status_counts.get('정상예측_TN', 0)
fp_wrong = status_counts.get('FP_잘못된경고', 0)
fn_miss = status_counts.get('FN_완전놓침', 0)
fn_10min = status_counts.get('FN_10분전예측', 0)

real_fn = fn_miss
real_fp = fp_wrong

print("\n" + "=" * 70)
print("V12 수정 평가 결과 (판정 규칙 민감도 ↑)")
print("=" * 70)

print("\n[수정된 판정 규칙]")
print("  rule1: 투표수 >= 3 (기존 4)")
print("  rule2: LGBM확률 >= 0.50 & 현재값 >= 1550 (기존 0.92, 1520)")
print("  rule3: XGB예측 >= 1680 & 현재값 >= 1500 (기존 1710, 1520)")
print("  rule4: 현재값 >= 1600 & 투표수 >= 2 (기존 1650, 3)")

print(f"\n[혼동 행렬]")
print(f"  TP (정확한 돌파 예측): {tp}건")
print(f"  TN (정확한 안전 예측): {tn}건")
print(f"  FP (잘못된 경고):      {fp_wrong}건")
print(f"  FN (놓친 돌파):        {fn_miss + fn_10min}건")

print(f"\n[예측상태 분류]")
for status, cnt in status_counts.items():
    print(f"  {status}: {cnt}건")

print(f"\n[실질적 성능 (10분 기준)]")
print(f"  실질 FN (완전 놓침):    {real_fn}건")
print(f"  실질 FP (잘못된 경고):  {real_fp}건")

print(f"\n[비교]")
print(f"  V10 기존:  실질 FN=7건, 실질 FP=338건")
print(f"  V12 기존:  실질 FN=92건, 실질 FP=118건")
print(f"  V12 수정:  실질 FN={real_fn}건, 실질 FP={real_fp}건")

# ============================================================================
# 저장
# ============================================================================
result_df.to_csv(CONFIG['output_file'], index=False, encoding='utf-8-sig')
print(f"\n결과 저장: {CONFIG['output_file']}")

print("\n" + "=" * 70)
print("V12 수정 평가 완료!")
print("=" * 70)