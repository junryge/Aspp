# -*- coding: utf-8 -*-
"""
================================================================================
V10_4 ML 예측 모델 - 평가 코드 (30분 예측)
V10_3 방식 (XGBoost 회귀 + LightGBM 분류 + 투표) + 30분 예측
================================================================================
"""

import os
import pickle
import warnings
import gc
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# ============================================================================
# 설정
# ============================================================================
CONFIG = {
    'model_file': 'models/v10_4_30min_m14_model.pkl',
    'eval_file': 'M14_평가.CSV',
    'output_file': f'V10_4_30min_평가결과_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
    'sequence_length': 280,
    'prediction_offset': 30,  # 30분 후!
    'limit_value': 1700,
    'target_column': 'TOTALCNT',
}

print("=" * 70)
print("🚀 V10_4 ML 예측 모델 - 평가 시작 (30분 예측)")
print("   V10_3 방식 (XGB 회귀 + LGBM 분류 + 투표)")
print("   시퀀스: 280분, 예측: 30분 후")
print("=" * 70)

# ============================================================================
# 모델 로드
# ============================================================================
print("\n[1/5] 모델 로드 중...")

with open(CONFIG['model_file'], 'rb') as f:
    model_data = pickle.load(f)

models = model_data['models']
scalers = model_data['scalers']
FEATURE_GROUPS = model_data['feature_groups']
training_info = model_data.get('training_info', {})

print(f"  - 모델 버전: {training_info.get('version', 'V10_4')}")
print(f"  - 학습일: {training_info.get('train_date', 'N/A')}")
print(f"  - 모델 수: {len(models)}개")

# ============================================================================
# 데이터 로드
# ============================================================================
print("\n[2/5] 평가 데이터 로드 중...")

try:
    df = pd.read_csv(CONFIG['eval_file'], encoding='utf-8', on_bad_lines='skip')
except:
    try:
        df = pd.read_csv(CONFIG['eval_file'], encoding='cp949', on_bad_lines='skip')
    except:
        df = pd.read_csv(CONFIG['eval_file'], encoding='euc-kr', on_bad_lines='skip')

print(f"  - 원본 데이터: {len(df):,}행")

df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M', errors='coerce')
df = df.dropna(subset=['CURRTIME']).sort_values('CURRTIME').reset_index(drop=True)
print(f"  - 파싱 후 데이터: {len(df):,}행")

# QUEUE_GAP 파생 (먼저!)
if 'M14.QUE.ALL.CURRENTQCREATED' in df.columns and 'M14.QUE.ALL.CURRENTQCOMPLETED' in df.columns:
    df['QUEUE_GAP'] = df['M14.QUE.ALL.CURRENTQCREATED'] - df['M14.QUE.ALL.CURRENTQCOMPLETED']
    print("  - QUEUE_GAP 파생 변수 생성!")

# Feature 그룹 필터링
for group_name in FEATURE_GROUPS:
    original = FEATURE_GROUPS[group_name].copy()
    FEATURE_GROUPS[group_name] = [f for f in FEATURE_GROUPS[group_name] if f in df.columns]
    missing = set(original) - set(FEATURE_GROUPS[group_name])
    if missing:
        print(f"  ⚠ {group_name} 누락: {missing}")

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
# 평가 수행
# ============================================================================
print("\n[3/5] 예측 수행 중...")

seq_len = CONFIG['sequence_length']
pred_offset = CONFIG['prediction_offset']
limit_val = CONFIG['limit_value']
target_col = CONFIG['target_column']

results = []
total = len(df) - seq_len - pred_offset
print(f"  → 예상 평가 수: {total:,}개")

for idx in range(seq_len, len(df) - pred_offset):
    if (idx - seq_len) % 1000 == 0:
        print(f"    진행: {idx - seq_len:,}/{total:,}")
        gc.collect()
    
    current_time = df['CURRTIME'].iloc[idx - 1]
    current_total = df[target_col].iloc[idx - 1]
    prediction_time = current_time + timedelta(minutes=pred_offset)
    
    # 실제값: 30분 내 최대값
    future_end = min(idx - 1 + pred_offset, len(df))
    actual_max = df[target_col].iloc[idx - 1:future_end].max()
    
    # 실제단일값: 정확히 30분 후 값 (예측 시점의 실제값)
    actual_single_idx = idx - 1 + pred_offset
    if actual_single_idx < len(df):
        actual_single = df[target_col].iloc[actual_single_idx]
    else:
        actual_single = df[target_col].iloc[-1]
    
    # Feature 생성
    feat_target = create_sequence_features(df, FEATURE_GROUPS['target'], seq_len, idx)
    feat_important = create_sequence_features(df, FEATURE_GROUPS['important'], seq_len, idx)
    feat_auxiliary = create_sequence_features(df, FEATURE_GROUPS['auxiliary'], seq_len, idx)
    
    X_target = scalers['target'].transform([feat_target])
    X_important = scalers['important'].transform([feat_important])
    X_auxiliary = scalers['auxiliary'].transform([feat_auxiliary])
    
    # XGB 회귀 예측
    pred_xgb_target = models['xgb_target'].predict(X_target)[0]
    pred_xgb_important = models['xgb_important'].predict(X_important)[0]
    pred_xgb_auxiliary = models['xgb_auxiliary'].predict(X_auxiliary)[0]
    
    # LGBM 분류 예측
    pred_lgb_target = models['lgb_target'].predict(X_target)[0]
    pred_lgb_important = models['lgb_important'].predict(X_important)[0]
    pred_lgb_auxiliary = models['lgb_auxiliary'].predict(X_auxiliary)[0]
    
    prob_lgb_target = models['lgb_target'].predict_proba(X_target)[0][1]
    prob_lgb_important = models['lgb_important'].predict_proba(X_important)[0][1]
    prob_lgb_auxiliary = models['lgb_auxiliary'].predict_proba(X_auxiliary)[0][1]
    
    # PDT 모델 (있으면)
    pred_xgb_pdt, pred_lgb_pdt, prob_lgb_pdt = None, None, None
    if 'xgb_pdt_new' in models and 'pdt_new' in scalers:
        feat_pdt = create_sequence_features(df, FEATURE_GROUPS.get('pdt_new', []), seq_len, idx)
        if feat_pdt:
            X_pdt = scalers['pdt_new'].transform([feat_pdt])
            pred_xgb_pdt = models['xgb_pdt_new'].predict(X_pdt)[0]
            pred_lgb_pdt = models['lgb_pdt_new'].predict(X_pdt)[0]
            prob_lgb_pdt = models['lgb_pdt_new'].predict_proba(X_pdt)[0][1]
    
    # ============================================
    # 투표 (V10_3 방식)
    # ============================================
    votes = [
        1 if pred_xgb_target >= limit_val else 0,
        1 if pred_xgb_important >= limit_val else 0,
        1 if pred_xgb_auxiliary >= limit_val else 0,
        pred_lgb_target,
        pred_lgb_important,
        pred_lgb_auxiliary,
    ]
    
    if pred_xgb_pdt is not None:
        votes.append(1 if pred_xgb_pdt >= limit_val else 0)
        votes.append(pred_lgb_pdt)
    
    vote_sum = sum(votes)
    total_votes = len(votes)
    
    # ============================================
    # 최종 판정 규칙 (V10_3 방식)
    # ============================================
    rule1 = vote_sum >= 3
    rule2 = (prob_lgb_important >= 0.50) and (current_total >= 1450)
    rule3 = (pred_xgb_important >= 1680) and (current_total >= 1500)
    rule4 = (current_total >= 1600) and (vote_sum >= 2)
    rule5 = (pred_xgb_important >= 1700)
    
    final_pred_danger = 1 if (rule1 or rule2 or rule3 or rule4 or rule5) else 0
    
    # 앙상블 평균 (회귀값)
    if pred_xgb_pdt is not None:
        ensemble_pred = (pred_xgb_target + pred_xgb_important + pred_xgb_auxiliary + pred_xgb_pdt) / 4
    else:
        ensemble_pred = (pred_xgb_target + pred_xgb_important + pred_xgb_auxiliary) / 3
    
    # 결과 저장
    results.append({
        '현재시간': current_time.strftime('%Y-%m-%d %H:%M'),
        '현재TOTALCNT': round(current_total, 2),
        '예측시점': prediction_time.strftime('%Y-%m-%d %H:%M'),
        '실제값10min': round(actual_max, 2),
        '실제단일값': round(actual_single, 2),
        'XGB_타겟': round(pred_xgb_target, 2),
        'XGB_중요': round(pred_xgb_important, 2),
        'XGB_보조': round(pred_xgb_auxiliary, 2),
        'XGB_PDT': round(pred_xgb_pdt, 2) if pred_xgb_pdt else '',
        'LGBM_중요_확률': round(prob_lgb_important, 3),
        '앙상블예측': round(ensemble_pred, 2),
        f'투표({total_votes}개중)': vote_sum,
        '최종판정': final_pred_danger,
        '실제위험(1700+)': 1 if actual_max >= limit_val else 0,
    })

print(f"  ✅ 완료!")

# ============================================================================
# 결과 저장
# ============================================================================
print("\n[4/5] 결과 저장...")

df_result = pd.DataFrame(results)

# ============================================================================
# 예측상태 분류 (10분 전/후 분석)
# ============================================================================
print("\n  → 예측상태 분류 중 (10분 전/후 분석)...")

df_result['현재시간_dt'] = pd.to_datetime(df_result['현재시간'])

def get_prediction_status(row, all_df):
    actual = row['실제위험(1700+)']
    pred = row['최종판정']
    current_time = row['현재시간_dt']
    
    if actual == 1 and pred == 1:
        return '정상예측_TP'
    elif actual == 0 and pred == 0:
        return '정상예측_TN'
    elif actual == 1 and pred == 0:
        # 30분 예측: 20분 이내에 예측된 값이 있는지 확인
        for minutes_ago in range(1, 21):  # 1~20분 전 확인
            time_ago = current_time - timedelta(minutes=minutes_ago)
            prev_data = all_df[all_df['현재시간_dt'] == time_ago]
            if len(prev_data) > 0 and prev_data['최종판정'].values[0] == 1:
                return 'FN_20분내예측'
        return 'FN_완전놓침'
    else:
        # 30분 예측: 20분 이내에 실제 돌파가 있는지 확인
        for minutes_later in range(1, 21):  # 1~20분 후 확인
            time_later = current_time + timedelta(minutes=minutes_later)
            later_data = all_df[all_df['현재시간_dt'] == time_later]
            if len(later_data) > 0 and later_data['실제위험(1700+)'].values[0] == 1:
                return 'FP_20분내돌파'
        return 'FP_잘못된경고'

df_result['예측상태'] = df_result.apply(lambda row: get_prediction_status(row, df_result), axis=1)
df_result = df_result.drop(columns=['현재시간_dt'])

df_result.to_csv(CONFIG['output_file'], index=False, encoding='utf-8-sig')
print(f"  → 저장: {CONFIG['output_file']}")

# ============================================================================
# 성능 평가
# ============================================================================
print("\n[5/5] 성능 평가...")

print("\n" + "=" * 70)
print("📊 V10_4 30분 평가 통계")
print("=" * 70)

actual_danger = df_result['실제위험(1700+)'] == 1
pred_danger = df_result['최종판정'] == 1

TP = (actual_danger & pred_danger).sum()
TN = (~actual_danger & ~pred_danger).sum()
FP = (~actual_danger & pred_danger).sum()
FN = (actual_danger & ~pred_danger).sum()

print(f"총 예측: {len(df_result):,}개")
print(f"실제 1700+: {actual_danger.sum()}개")

print(f"\n🔥 1700+ 감지:")
if actual_danger.sum() > 0:
    recall = TP / actual_danger.sum() * 100
    print(f"  감지(TP): {TP}개 ({recall:.1f}%)")
    print(f"  미감지(FN): {FN}개 ({100-recall:.1f}%)")

print(f"\n⚠️ 오탐(FP): {FP}개")
if pred_danger.sum() > 0:
    precision = TP / pred_danger.sum() * 100
    print(f"정밀도: {precision:.1f}%")

print(f"\n[혼동 행렬]")
print(f"  TP: {TP:,}건 | FN: {FN:,}건")
print(f"  FP: {FP:,}건 | TN: {TN:,}건")

# 예측상태별 집계
print(f"\n[예측상태 분류]")
status_counts = df_result['예측상태'].value_counts()
for status, count in status_counts.items():
    print(f"  {status}: {count:,}건")

# 실질 FN/FP
real_fn = status_counts.get('FN_완전놓침', 0)
real_fp = status_counts.get('FP_잘못된경고', 0)
fn_early = status_counts.get('FN_20분내예측', 0)
fp_early = status_counts.get('FP_20분내돌파', 0)

print(f"\n[실질적 성능 (20분 기준)]")
print(f"  FN 20분내예측 (조기감지): {fn_early:,}건")
print(f"  FP 20분내돌파 (유효경고): {fp_early:,}건")
print(f"  실질 FN (완전 놓침):    {real_fn:,}건")
print(f"  실질 FP (잘못된 경고):  {real_fp:,}건")

if (TP + TN + FP + FN) > 0:
    accuracy = (TP + TN) / (TP + TN + FP + FN) * 100
    print(f"\n정확도: {accuracy:.2f}%")

print(f"\n✅ V10_4 평가 완료! → {CONFIG['output_file']}")
print("=" * 70)