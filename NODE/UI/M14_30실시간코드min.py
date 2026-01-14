# -*- coding: utf-8 -*-
"""
================================================================================
V10_4 ML 예측 모델 - 실시간 예측 코드 (30분 예측)
V10_3 방식 (XGBoost 회귀 + LightGBM 분류 + 투표)
예측값: XGB_타겟
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
    'sequence_length': 280,
    'prediction_offset': 30,
    'limit_value': 1700,
    'target_column': 'TOTALCNT',
}

# ============================================================================
# 모델 로드
# ============================================================================
def load_model():
    """학습된 모델 로드"""
    print("=" * 70)
    print("🚀 V10_4 실시간 예측 시스템 - 30분 예측")
    print("   V10_3 방식 (XGB 회귀 + LGBM 분류 + 투표)")
    print("   예측값: XGB_타겟")
    print("=" * 70)
    
    if not os.path.exists(CONFIG['model_file']):
        raise FileNotFoundError(f"모델 파일이 없습니다: {CONFIG['model_file']}")
    
    with open(CONFIG['model_file'], 'rb') as f:
        model_data = pickle.load(f)
    
    training_info = model_data.get('training_info', {})
    print(f"✅ 모델 로드 완료")
    print(f"  - 모델 버전: {training_info.get('version', 'V10_4')}")
    print(f"  - 학습일: {training_info.get('train_date', 'N/A')}")
    print(f"  - 모델 수: {len(model_data['models'])}개")
    
    return model_data

# ============================================================================
# Feature 생성 함수 (평가 코드와 동일)
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
# 데이터 전처리 함수
# ============================================================================
def preprocess_data(df, feature_groups):
    """데이터 전처리 (평가 코드와 동일)"""
    df = df.copy()
    
    df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M', errors='coerce')
    df = df.dropna(subset=['CURRTIME']).sort_values('CURRTIME').reset_index(drop=True)
    
    if 'M14.QUE.ALL.CURRENTQCREATED' in df.columns and 'M14.QUE.ALL.CURRENTQCOMPLETED' in df.columns:
        df['QUEUE_GAP'] = df['M14.QUE.ALL.CURRENTQCREATED'] - df['M14.QUE.ALL.CURRENTQCOMPLETED']
    
    # 없는 컬럼은 0으로 생성 (feature 개수 맞추기 위해 필수!)
    all_cols = []
    for group in feature_groups.values():
        all_cols.extend(group)
    for col in list(set(all_cols)):
        if col not in df.columns:
            df[col] = 0
            print(f"  ⚠ 컬럼 없음, 0으로 생성: {col}")
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df

# ============================================================================
# 단일 시점 예측 함수 (평가 코드 로직 그대로)
# ============================================================================
def predict_single(df, idx, model_data):
    """
    단일 시점에 대한 예측 수행
    Returns: dict (예측값 = XGB_타겟)
    """
    models = model_data['models']
    scalers = model_data['scalers']
    FEATURE_GROUPS = model_data['feature_groups']
    
    seq_len = CONFIG['sequence_length']
    pred_offset = CONFIG['prediction_offset']
    limit_val = CONFIG['limit_value']
    target_col = CONFIG['target_column']
    
    if idx < seq_len:
        raise ValueError(f"인덱스 {idx}는 시퀀스 길이 {seq_len}보다 작습니다.")
    
    current_time = df['CURRTIME'].iloc[idx - 1]
    current_total = df[target_col].iloc[idx - 1]
    prediction_time = current_time + timedelta(minutes=pred_offset)
    
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
    
    pred_xgb_pdt, pred_lgb_pdt, prob_lgb_pdt = None, None, None
    if 'xgb_pdt_new' in models and 'pdt_new' in scalers:
        feat_pdt = create_sequence_features(df, FEATURE_GROUPS.get('pdt_new', []), seq_len, idx)
        if feat_pdt:
            X_pdt = scalers['pdt_new'].transform([feat_pdt])
            pred_xgb_pdt = models['xgb_pdt_new'].predict(X_pdt)[0]
            pred_lgb_pdt = models['lgb_pdt_new'].predict(X_pdt)[0]
            prob_lgb_pdt = models['lgb_pdt_new'].predict_proba(X_pdt)[0][1]
    
    # 투표 (V10_3 방식)
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
    
    # 최종 판정 규칙 (V10_3 방식)
    rule1 = vote_sum >= 3
    rule2 = (prob_lgb_important >= 0.50) and (current_total >= 1450)
    rule3 = (pred_xgb_important >= 1680) and (current_total >= 1500)
    rule4 = (current_total >= 1600) and (vote_sum >= 2)
    rule5 = (pred_xgb_important >= 1700)
    
    final_pred_danger = 1 if (rule1 or rule2 or rule3 or rule4 or rule5) else 0
    
    if pred_xgb_pdt is not None:
        ensemble_pred = (pred_xgb_target + pred_xgb_important + pred_xgb_auxiliary + pred_xgb_pdt) / 4
    else:
        ensemble_pred = (pred_xgb_target + pred_xgb_important + pred_xgb_auxiliary) / 3
    
    return {
        '현재시간': current_time.strftime('%Y-%m-%d %H:%M'),
        '현재TOTALCNT': round(current_total, 2),
        '예측시점': prediction_time.strftime('%Y-%m-%d %H:%M'),
        '예측값': round(pred_xgb_target, 2),
        'XGB_타겟': round(pred_xgb_target, 2),
        'XGB_중요': round(pred_xgb_important, 2),
        'XGB_보조': round(pred_xgb_auxiliary, 2),
        'XGB_PDT': round(pred_xgb_pdt, 2) if pred_xgb_pdt else None,
        'LGBM_중요_확률': round(prob_lgb_important, 3),
        '앙상블예측': round(ensemble_pred, 2),
        f'투표({total_votes}개중)': vote_sum,
        '최종판정': final_pred_danger,
        '위험여부': '🔴 위험' if final_pred_danger == 1 else '🟢 안전',
    }

# ============================================================================
# 실시간 모니터링
# ============================================================================
def realtime_monitoring(data_file, model_data):
    print(f"\n📡 실시간 모니터링 시작 - 30분 예측")
    print("-" * 70)
    
    try:
        df = pd.read_csv(data_file, encoding='utf-8', on_bad_lines='skip')
    except:
        try:
            df = pd.read_csv(data_file, encoding='cp949', on_bad_lines='skip')
        except:
            df = pd.read_csv(data_file, encoding='euc-kr', on_bad_lines='skip')
    
    FEATURE_GROUPS = model_data['feature_groups'].copy()
    df = preprocess_data(df, FEATURE_GROUPS)
    model_data['feature_groups'] = FEATURE_GROUPS
    
    seq_len = CONFIG['sequence_length']
    print(f"데이터: {len(df):,}행, 시작 인덱스: {seq_len}")
    print("-" * 70)
    
    for idx in range(seq_len, len(df)):
        try:
            result = predict_single(df, idx, model_data)
            print(f"\n[{result['현재시간']}] 현재: {result['현재TOTALCNT']:.0f}")
            print(f"  예측값(XGB_타겟): {result['예측값']:.0f}, LGBM확률: {result['LGBM_중요_확률']:.1%}")
            print(f"  ▶ {result['위험여부']}")
            if result['최종판정'] == 1:
                print("  🚨 30분 내 1700 초과 예상!")
        except Exception as e:
            print(f"  ❌ 오류: {e}")

if __name__ == "__main__":
    model_data = load_model()
    data_file = input("\n데이터 파일 경로: ").strip()
    if data_file and os.path.exists(data_file):
        realtime_monitoring(data_file, model_data)
    else:
        print("❌ 파일 없음")