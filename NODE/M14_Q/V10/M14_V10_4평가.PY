# -*- coding: utf-8 -*-
"""
================================================================================
V11 ML 예측 모델 - 평가 코드
V8.3.1 + V10.3 통합 (XGBoost 전용)

특징:
    - 시퀀스: 280분 (V8.3.1 기준)
    - 예측: 10분 후 단일값 (V8.3.1 기준)
    - 출력: V8.3.1 형식 (추세예측 포함)
    - Feature 그룹: target, important, auxiliary (V10.3 구조)
    - 모델: XGBoost만 사용
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
    'model_file': 'models/v11_m14_model.pkl',
    'eval_file': 'M14_평가.CSV',  # 평가 파일명 수정 필요
    'output_file': f'V11_평가결과_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
    'sequence_length': 280,
    'prediction_offset': 10,
    'limit_value': 1700,
    'target_column': 'TOTALCNT',
}

print("=" * 70)
print("🚀 V11 ML 예측 모델 - 평가 시작")
print("   V8.3.1 + V10.3 통합 (XGBoost 전용)")
print("   시퀀스: 280분, 예측: 10분 후")
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

print(f"  - 모델 버전: {training_info.get('version', 'V11')}")
print(f"  - 학습일: {training_info.get('train_date', 'N/A')}")
print(f"  - 학습 샘플: {training_info.get('samples', 'N/A'):,}개")

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

# CURRTIME 파싱
df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M', errors='coerce')
df = df.dropna(subset=['CURRTIME'])
df = df.sort_values('CURRTIME').reset_index(drop=True)
print(f"  - 파싱 후 데이터: {len(df):,}행")

# Feature 그룹 필터링
for group_name in FEATURE_GROUPS:
    original = FEATURE_GROUPS[group_name].copy()
    FEATURE_GROUPS[group_name] = [f for f in FEATURE_GROUPS[group_name] if f in df.columns]
    missing = set(original) - set(FEATURE_GROUPS[group_name])
    if missing:
        print(f"  ⚠ {group_name} 누락 컬럼: {missing}")

# 숫자형 변환
all_cols = []
for group in FEATURE_GROUPS.values():
    all_cols.extend(group)
all_cols = list(set(all_cols))

for col in all_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# queue_gap 파생 변수
if 'M14.QUE.ALL.CURRENTQCREATED' in df.columns and 'M14.QUE.ALL.CURRENTQCOMPLETED' in df.columns:
    df['QUEUE_GAP'] = df['M14.QUE.ALL.CURRENTQCREATED'] - df['M14.QUE.ALL.CURRENTQCOMPLETED']
    if 'QUEUE_GAP' not in FEATURE_GROUPS['auxiliary']:
        FEATURE_GROUPS['auxiliary'].append('QUEUE_GAP')

print(f"  - Target 컬럼: {len(FEATURE_GROUPS['target'])}개")
print(f"  - Important 컬럼: {len(FEATURE_GROUPS['important'])}개")
print(f"  - Auxiliary 컬럼: {len(FEATURE_GROUPS['auxiliary'])}개")

# ============================================================================
# Feature 생성 함수
# ============================================================================
def create_sequence_features(df, feature_cols, seq_len, idx, limit_val=1700):
    """시퀀스 기반 Feature 생성"""
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
        print(f"    진행: {idx - seq_len:,}/{total:,} ({100*(idx-seq_len)/total:.1f}%)")
        gc.collect()
    
    # 현재 시간 및 값
    current_time = df['CURRTIME'].iloc[idx - 1]
    current_total = df[target_col].iloc[idx - 1]
    prediction_time = current_time + timedelta(minutes=pred_offset)
    
    # 실제값 (10분 후)
    actual = df[target_col].iloc[idx + pred_offset - 1]
    
    # 보조 변수 추출
    m14b = df['M14AM14B'].iloc[idx - 1] if 'M14AM14B' in df.columns else None
    m14bsum = df['M14AM14BSUM'].iloc[idx - 1] if 'M14AM14BSUM' in df.columns else None
    gap = df['QUEUE_GAP'].iloc[idx - 1] if 'QUEUE_GAP' in df.columns else None
    trans = df['M14.QUE.ALL.TRANSPORT4MINOVERCNT'].iloc[idx - 1] if 'M14.QUE.ALL.TRANSPORT4MINOVERCNT' in df.columns else None
    
    # Feature 생성
    feat_target = create_sequence_features(df, FEATURE_GROUPS['target'], seq_len, idx)
    feat_important = create_sequence_features(df, FEATURE_GROUPS['important'], seq_len, idx)
    feat_auxiliary = create_sequence_features(df, FEATURE_GROUPS['auxiliary'], seq_len, idx)
    feat_pdt_new = create_sequence_features(df, FEATURE_GROUPS.get('pdt_new', []), seq_len, idx)
    
    # 스케일링
    X_target = scalers['target'].transform([feat_target])
    X_important = scalers['important'].transform([feat_important])
    X_auxiliary = scalers['auxiliary'].transform([feat_auxiliary])
    
    # XGBoost 예측
    pred_xgb_target = models['xgb_target'].predict(X_target)[0]
    pred_xgb_important = models['xgb_important'].predict(X_important)[0]
    pred_xgb_auxiliary = models['xgb_auxiliary'].predict(X_auxiliary)[0]
    
    # pdt_new 모델 예측 (있는 경우)
    if 'xgb_pdt_new' in models and 'pdt_new' in scalers and len(feat_pdt_new) > 0:
        X_pdt_new = scalers['pdt_new'].transform([feat_pdt_new])
        pred_xgb_pdt_new = models['xgb_pdt_new'].predict(X_pdt_new)[0]
        # 앙상블 (4개 평균)
        final_pred = (pred_xgb_target + pred_xgb_important + pred_xgb_auxiliary + pred_xgb_pdt_new) / 4
    else:
        pred_xgb_pdt_new = None
        # 앙상블 (3개 평균)
        final_pred = (pred_xgb_target + pred_xgb_important + pred_xgb_auxiliary) / 3
    
    # 결과 저장
    results.append({
        '현재시간': current_time.strftime('%Y-%m-%d %H:%M'),
        '현재TOTALCNT': round(current_total, 2),
        '예측시점': prediction_time.strftime('%Y-%m-%d %H:%M'),
        '실제값': round(actual, 2),
        'XGB_타겟': round(pred_xgb_target, 2),
        'XGB_중요': round(pred_xgb_important, 2),
        'XGB_보조': round(pred_xgb_auxiliary, 2),
        'XGB_PDT': round(pred_xgb_pdt_new, 2) if pred_xgb_pdt_new is not None else '',
        '최종예측': round(final_pred, 2),
        '오차': round(actual - final_pred, 2),
        '절대오차': round(abs(actual - final_pred), 2),
        'M14AM14B': round(m14b, 2) if m14b else '',
        'M14AM14BSUM': round(m14bsum, 2) if m14bsum else '',
        'queue_gap': round(gap, 2) if gap else '',
        'TRANSPORT': round(trans, 2) if trans else '',
        '실제위험(1700+)': 'O' if actual >= 1700 else '',
        '예측위험(1700+)': 'O' if final_pred >= 1700 else ''
    })

print(f"  ✅ 완료!")

# ============================================================================
# 결과 저장
# ============================================================================
print("\n[4/5] 결과 저장...")

df_result = pd.DataFrame(results)
df_result.to_csv(CONFIG['output_file'], index=False, encoding='utf-8-sig')
print(f"  → 저장: {CONFIG['output_file']}")

# ============================================================================
# 성능 평가
# ============================================================================
print("\n[5/5] 성능 평가...")

print("\n" + "=" * 70)
print("📊 V11 10분 평가 통계")
print("=" * 70)

print(f"총 예측: {len(df_result):,}개")
print(f"MAE: {df_result['절대오차'].mean():.2f}")

actual_danger = df_result['실제위험(1700+)'] == 'O'
pred_danger = df_result['예측위험(1700+)'] == 'O'
actual_count = actual_danger.sum()
detected = (actual_danger & pred_danger).sum()

print(f"\n🔥 1700+ 감지 (최종예측 기준):")
print(f"  실제: {actual_count}개")

if actual_count > 0:
    recall = detected / actual_count * 100
    print(f"  감지(TP): {detected}개 ({recall:.1f}%)")
    
    # 미감지 분석
    fn = actual_count - detected
    print(f"  미감지(FN): {fn}개 ({fn/actual_count*100:.1f}%)")

# 오탐 분석
fp = (pred_danger & ~actual_danger).sum()
print(f"\n⚠️ 오탐(FP): {fp}개")

if pred_danger.sum() > 0:
    precision = detected / pred_danger.sum() * 100
    print(f"정밀도: {precision:.1f}%")

# 구간별 MAE
print(f"\n📊 구간별 MAE:")
for low, high in [(0, 1500), (1500, 1600), (1600, 1700), (1700, 2000)]:
    mask = (df_result['실제값'] >= low) & (df_result['실제값'] < high)
    if mask.sum() > 0:
        mae = df_result.loc[mask, '절대오차'].mean()
        print(f"  {low}~{high}: MAE={mae:.2f} ({mask.sum():,}개)")

# 혼동 행렬
TP = detected
TN = (~actual_danger & ~pred_danger).sum()
FP = fp
FN = (actual_danger & ~pred_danger).sum()

print(f"\n[혼동 행렬]")
print(f"  TP (정확한 위험 예측): {TP:,}건")
print(f"  TN (정확한 안전 예측): {TN:,}건")
print(f"  FP (오탐):              {FP:,}건")
print(f"  FN (미감지):            {FN:,}건")

if (TP + TN + FP + FN) > 0:
    accuracy = (TP + TN) / (TP + TN + FP + FN) * 100
    print(f"\n정확도: {accuracy:.2f}%")

print(f"\n✅ V11 10분 평가 완료! → {CONFIG['output_file']}")
print("=" * 70)