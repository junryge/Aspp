# -*- coding: utf-8 -*-
"""
#Predictor_10min.py
================================================================================
V10_4 ML 예측 모듈 - 10분 예측
main.py에서 import해서 사용
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
    'model_file': 'models/v10_4_m14_model.pkl',
    'sequence_length': 280,
    'prediction_offset': 10,
    'limit_value': 1700,
    'target_column': 'TOTALCNT',
}

# 전역 모델 캐시
_model_data = None


def load_model():
    """모델 로드 (한 번만)"""
    global _model_data
    
    if _model_data is not None:
        return _model_data
    
    print("[10분 예측] 모델 로드 중...")
    
    if not os.path.exists(CONFIG['model_file']):
        print(f"  ⚠ 모델 파일 없음: {CONFIG['model_file']}")
        return None
    
    with open(CONFIG['model_file'], 'rb') as f:
        _model_data = pickle.load(f)
    
    print(f"  ✅ 모델 로드 완료 (모델 수: {len(_model_data['models'])}개)")
    return _model_data


def create_sequence_features(df, feature_cols, seq_len, idx, limit_val=1700):
    """시퀀스 피처 생성"""
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


def preprocess_data(df, feature_groups):
    """데이터 전처리"""
    df = df.copy()
    
    # CURRTIME이 datetime이 아니면 변환
    if not pd.api.types.is_datetime64_any_dtype(df['CURRTIME']):
        df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M', errors='coerce')
    
    df = df.dropna(subset=['CURRTIME']).sort_values('CURRTIME').reset_index(drop=True)
    
    # QUEUE_GAP 생성
    if 'M14.QUE.ALL.CURRENTQCREATED' in df.columns and 'M14.QUE.ALL.CURRENTQCOMPLETED' in df.columns:
        df['QUEUE_GAP'] = df['M14.QUE.ALL.CURRENTQCREATED'] - df['M14.QUE.ALL.CURRENTQCOMPLETED']
    
    # 없는 컬럼 0으로 생성
    all_cols = []
    for group in feature_groups.values():
        all_cols.extend(group)
    for col in list(set(all_cols)):
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df


def predict(df):
    """
    10분 후 예측 수행
    
    Args:
        df: DataFrame (280개 이상의 row, CURRTIME/TOTALCNT 포함)
    
    Returns:
        dict: 예측 결과
            - predict_value: 예측값 (XGB_타겟)
            - current_value: 현재 TOTALCNT
            - current_time: 현재 시간
            - predict_time: 예측 시점 (10분 후)
            - danger: 위험 여부 (0 or 1)
            - prob: LGBM 확률
    """
    model_data = load_model()
    
    if model_data is None:
        # 모델 없으면 단순 이동평균 폴백
        return _fallback_predict(df, 10)
    
    try:
        models = model_data['models']
        scalers = model_data['scalers']
        feature_groups = model_data['feature_groups']
        
        # 데이터 전처리
        df = preprocess_data(df, feature_groups)
        
        seq_len = CONFIG['sequence_length']
        limit_val = CONFIG['limit_value']
        
        if len(df) < seq_len:
            print(f"  ⚠ 데이터 부족: {len(df)} < {seq_len}")
            return _fallback_predict(df, 10)
        
        idx = len(df)  # 마지막 인덱스 + 1
        
        current_time = df['CURRTIME'].iloc[-1]
        current_total = float(df['TOTALCNT'].iloc[-1])
        prediction_time = current_time + timedelta(minutes=10)
        
        # Feature 생성
        feat_target = create_sequence_features(df, feature_groups['target'], seq_len, idx)
        feat_important = create_sequence_features(df, feature_groups['important'], seq_len, idx)
        feat_auxiliary = create_sequence_features(df, feature_groups['auxiliary'], seq_len, idx)
        
        # 스케일링
        X_target = scalers['target'].transform([feat_target])
        X_important = scalers['important'].transform([feat_important])
        X_auxiliary = scalers['auxiliary'].transform([feat_auxiliary])
        
        # XGBoost 예측
        pred_xgb_target = models['xgb_target'].predict(X_target)[0]
        pred_xgb_important = models['xgb_important'].predict(X_important)[0]
        pred_xgb_auxiliary = models['xgb_auxiliary'].predict(X_auxiliary)[0]
        
        # LightGBM 예측
        pred_lgb_target = models['lgb_target'].predict(X_target)[0]
        pred_lgb_important = models['lgb_important'].predict(X_important)[0]
        pred_lgb_auxiliary = models['lgb_auxiliary'].predict(X_auxiliary)[0]
        
        # 확률
        prob_lgb_important = models['lgb_important'].predict_proba(X_important)[0][1]
        
        # PDT 모델 (있으면)
        pred_xgb_pdt = None
        if 'xgb_pdt_new' in models and 'pdt_new' in scalers:
            feat_pdt = create_sequence_features(df, feature_groups.get('pdt_new', []), seq_len, idx)
            if feat_pdt:
                X_pdt = scalers['pdt_new'].transform([feat_pdt])
                pred_xgb_pdt = models['xgb_pdt_new'].predict(X_pdt)[0]
        
        # 투표
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
        
        vote_sum = sum(votes)
        
        # 최종 판정 규칙
        rule1 = vote_sum >= 3
        rule2 = (prob_lgb_important >= 0.50) and (current_total >= 1450)
        rule3 = (pred_xgb_important >= 1680) and (current_total >= 1500)
        rule4 = (current_total >= 1600) and (vote_sum >= 2)
        rule5 = (pred_xgb_important >= 1700)
        
        final_danger = 1 if (rule1 or rule2 or rule3 or rule4 or rule5) else 0
        
        return {
            'predict_value': int(round(pred_xgb_target)),
            'current_value': int(round(current_total)),
            'current_time': current_time,
            'predict_time': prediction_time,
            'danger': final_danger,
            'prob': round(prob_lgb_important, 3),
            'vote': vote_sum,
        }
        
    except Exception as e:
        print(f"  ❌ 10분 예측 오류: {e}")
        return _fallback_predict(df, 10)


def _fallback_predict(df, offset):
    """모델 없을 때 단순 이동평균 폴백"""
    if len(df) < 5:
        return {
            'predict_value': 0,
            'current_value': 0,
            'current_time': datetime.now(),
            'predict_time': datetime.now() + timedelta(minutes=offset),
            'danger': 0,
            'prob': 0,
            'vote': 0,
        }
    
    values = df['TOTALCNT'].fillna(0).tolist()
    recent = values[-5:]
    avg = sum(recent) / len(recent)
    trend = (recent[-1] - recent[0]) / len(recent) if len(recent) >= 2 else 0
    pred = int(avg + trend * offset)
    pred = max(1000, min(2000, pred))
    
    current_val = int(values[-1])
    
    # CURRTIME 처리
    if 'CURRTIME' in df.columns:
        curr_time = df['CURRTIME'].iloc[-1]
        if isinstance(curr_time, str):
            try:
                curr_time = datetime.strptime(str(curr_time), '%Y%m%d%H%M')
            except:
                curr_time = datetime.now()
    else:
        curr_time = datetime.now()
    
    return {
        'predict_value': pred,
        'current_value': current_val,
        'current_time': curr_time,
        'predict_time': curr_time + timedelta(minutes=offset),
        'danger': 1 if pred >= 1700 else 0,
        'prob': 0,
        'vote': 0,
    }