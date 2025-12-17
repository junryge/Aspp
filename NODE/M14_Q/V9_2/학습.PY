#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 V8.3.1 학습 - 10분 예측
핵심 변경: 변화량 예측 방식

변경사항:
1. 타겟 = 변화량 (10분 후 - 현재)
2. Feature = 33개 (82개 → 33개)
3. 시계열 = 60분 (280분 → 60분)
4. 증강 = 제거
5. 예측값 = 현재값 + 예측_변화량
6. M16M14A 추가, 새 임계값(1550/1580/1600/1650)
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🚀 V8.3.1 학습 - 10분 예측")
print("   전략: 현재값 + 변화량 예측")
print("="*70)

# ========================================
# 설정
# ========================================
SEQ_LEN = 60       # 60분 시계열
PRED_MINUTES = 10  # 10분 후 예측

# ========================================
# Feature 생성 함수 (33개)
# ========================================
def create_features_v831(row_dict):
    """핵심 Feature만 (33개)"""
    features = {}
    
    seq_totalcnt = np.array(row_dict['TOTALCNT'])
    seq_m14b = np.array(row_dict['M14AM14B'])
    seq_m14bsum = np.array(row_dict['M14AM14BSUM'])
    seq_m10arev = np.array(row_dict['M10AM14A'])
    seq_m16m14a = np.array(row_dict['M16M14A'])
    seq_transport = np.array(row_dict['TRANSPORT'])
    seq_oht = np.array(row_dict['OHT'])
    seq_gap = np.array(row_dict['Q_CREATED']) - np.array(row_dict['Q_COMPLETED'])
    
    seq_len = len(seq_totalcnt)
    
    # 1. TOTALCNT (7개)
    features['total_current'] = seq_totalcnt[-1]
    features['total_last5'] = np.mean(seq_totalcnt[-5:])
    features['total_last10'] = np.mean(seq_totalcnt[-10:])
    features['total_trend5'] = seq_totalcnt[-1] - seq_totalcnt[-5]
    features['total_trend10'] = seq_totalcnt[-1] - seq_totalcnt[-10]
    features['total_slope'] = np.polyfit(np.arange(min(30, seq_len)), seq_totalcnt[-30:], 1)[0]
    features['total_std10'] = np.std(seq_totalcnt[-10:])
    
    # 2. M14AM14B (4개)
    features['m14b_current'] = seq_m14b[-1]
    features['m14b_last10'] = np.mean(seq_m14b[-10:])
    features['m14b_trend10'] = seq_m14b[-1] - seq_m14b[-10]
    features['m14b_slope'] = np.polyfit(np.arange(min(30, seq_len)), seq_m14b[-30:], 1)[0]
    
    # 3. M14AM14BSUM (4개)
    features['m14bsum_current'] = seq_m14bsum[-1]
    features['m14bsum_last10'] = np.mean(seq_m14bsum[-10:])
    features['m14bsum_trend10'] = seq_m14bsum[-1] - seq_m14bsum[-10]
    features['m14bsum_slope'] = np.polyfit(np.arange(min(30, seq_len)), seq_m14bsum[-30:], 1)[0]
    
    # 4. M16M14A (3개)
    features['m16_current'] = seq_m16m14a[-1]
    features['m16_last10'] = np.mean(seq_m16m14a[-10:])
    features['m16_trend10'] = seq_m16m14a[-1] - seq_m16m14a[-10]
    
    # 5. M10AM14A (2개)
    features['m10arev_current'] = seq_m10arev[-1]
    features['m10arev_trend10'] = seq_m10arev[-1] - seq_m10arev[-10]
    
    # 6. TRANSPORT (2개)
    features['trans_current'] = seq_transport[-1]
    features['trans_trend10'] = seq_transport[-1] - seq_transport[-10]
    
    # 7. OHT (2개)
    features['oht_current'] = seq_oht[-1]
    features['oht_last10'] = np.mean(seq_oht[-10:])
    
    # 8. Queue Gap (3개)
    features['gap_current'] = seq_gap[-1]
    features['gap_last10'] = np.mean(seq_gap[-10:])
    features['gap_trend10'] = seq_gap[-1] - seq_gap[-10]
    
    # 9. 핵심 임계값 (4개)
    features['is_over_1550'] = 1 if seq_totalcnt[-1] >= 1550 else 0
    features['is_over_1580'] = 1 if seq_totalcnt[-1] >= 1580 else 0
    features['is_over_1600'] = 1 if seq_totalcnt[-1] >= 1600 else 0
    features['is_over_1650'] = 1 if seq_totalcnt[-1] >= 1650 else 0
    
    # 10. 복합 지표 (2개)
    features['m14b_plus_sum'] = seq_m14b[-1] + seq_m14bsum[-1]
    features['total_momentum'] = features['total_trend5'] + features['total_trend10']
    
    return features

# ========================================
# 데이터 준비
# ========================================
def prepare_data(csv_path, seq_len=SEQ_LEN, pred_minutes=PRED_MINUTES):
    print(f"\n📂 데이터 로딩: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✅ {len(df):,}행 × {df.shape[1]}열")
    
    required = ['M14AM14B', 'M14AM14BSUM', 'M10AM14A', 'TOTALCNT', 'M16M14A',
                'M14.QUE.ALL.TRANSPORT4MINOVERCNT', 'M14.QUE.OHT.OHTUTIL',
                'M14.QUE.ALL.CURRENTQCREATED', 'M14.QUE.ALL.CURRENTQCOMPLETED']
    
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"❌ 누락 컬럼: {missing}")
        return None, None, None
    
    print(f"✅ 9개 컬럼 확인")
    
    before = len(df)
    df = df.dropna(subset=required)
    after = len(df)
    if before != after:
        print(f"⚠️ NaN 제거: {before:,} → {after:,}")
    
    print(f"\n📊 TOTALCNT 분포:")
    print(f"  범위: {df['TOTALCNT'].min():.0f} ~ {df['TOTALCNT'].max():.0f}")
    print(f"  1700+: {(df['TOTALCNT']>=1700).sum()}개 ({(df['TOTALCNT']>=1700).sum()/len(df)*100:.2f}%)")
    
    print(f"\n🔄 Feature 생성 중...")
    
    X_list = []
    y_change_list = []
    y_actual_list = []
    current_list = []
    
    pred_offset = pred_minutes
    total = len(df) - seq_len - pred_offset
    
    for i in range(seq_len, len(df) - pred_offset):
        row_dict = {
            'M14AM14B': df['M14AM14B'].iloc[i-seq_len:i].values,
            'M14AM14BSUM': df['M14AM14BSUM'].iloc[i-seq_len:i].values,
            'M10AM14A': df['M10AM14A'].iloc[i-seq_len:i].values,
            'TOTALCNT': df['TOTALCNT'].iloc[i-seq_len:i].values,
            'M16M14A': df['M16M14A'].iloc[i-seq_len:i].values,
            'TRANSPORT': df['M14.QUE.ALL.TRANSPORT4MINOVERCNT'].iloc[i-seq_len:i].values,
            'OHT': df['M14.QUE.OHT.OHTUTIL'].iloc[i-seq_len:i].values,
            'Q_CREATED': df['M14.QUE.ALL.CURRENTQCREATED'].iloc[i-seq_len:i].values,
            'Q_COMPLETED': df['M14.QUE.ALL.CURRENTQCOMPLETED'].iloc[i-seq_len:i].values,
        }
        
        current_total = df['TOTALCNT'].iloc[i-1]
        future_total = df['TOTALCNT'].iloc[i + pred_offset - 1]
        
        if pd.isna(future_total) or pd.isna(current_total):
            continue
        
        change = future_total - current_total
        
        X_list.append(create_features_v831(row_dict))
        y_change_list.append(change)
        y_actual_list.append(future_total)
        current_list.append(current_total)
        
        if (i - seq_len) % 5000 == 0:
            print(f"  {i-seq_len:,}/{total:,}")
    
    print(f"✅ 완료!")
    
    X = pd.DataFrame(X_list)
    y_change = pd.Series(y_change_list)
    y_actual = pd.Series(y_actual_list)
    current = pd.Series(current_list)
    
    mask = ~(X.isna().any(axis=1) | y_change.isna())
    X = X[mask].reset_index(drop=True)
    y_change = y_change[mask].reset_index(drop=True)
    y_actual = y_actual[mask].reset_index(drop=True)
    current = current[mask].reset_index(drop=True)
    
    print(f"\n📊 결과:")
    print(f"  Feature: {X.shape[1]}개")
    print(f"  샘플: {len(X):,}개")
    print(f"  변화량: {y_change.min():.0f} ~ {y_change.max():.0f}")
    print(f"  10분 후 1700+: {(y_actual>=1700).sum()}개")
    
    return X, y_change, pd.DataFrame({'current': current, 'actual': y_actual})

# ========================================
# 학습
# ========================================
def train_model(X, y):
    print(f"\n🚀 XGBoost 학습")
    
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        max_depth=6,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method='hist'
    )
    
    model.fit(X, y, verbose=50)
    print(f"\n✅ 학습 완료!")
    return model

# ========================================
# 평가
# ========================================
def evaluate(model, X, y_change, meta):
    print(f"\n" + "="*70)
    print(f"📊 성능 평가")
    print("="*70)
    
    pred_change = model.predict(X)
    pred_actual = meta['current'] + pred_change
    actual = meta['actual']
    
    mae = mean_absolute_error(actual, pred_actual)
    rmse = np.sqrt(mean_squared_error(actual, pred_actual))
    r2 = r2_score(actual, pred_actual)
    
    print(f"\nMAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
    
    actual_danger = actual >= 1700
    pred_danger = pred_actual >= 1700
    
    if actual_danger.sum() > 0:
        tp = (actual_danger & pred_danger).sum()
        recall = tp / actual_danger.sum() * 100
        print(f"\n1700+ 감지율: {recall:.1f}% ({tp}/{actual_danger.sum()})")
    
    return mae, rmse, r2

def show_importance(model, X):
    print(f"\n🏆 Feature 중요도 TOP 15")
    imp = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(imp.head(15).to_string(index=False))
    return imp

if __name__ == '__main__':
    CSV_FILE = "data/M14_Q_20241108_TO_20251210.csv"  # 수정 필요!
    
    X, y_change, meta = prepare_data(CSV_FILE)
    if X is None:
        exit(1)
    
    model = train_model(X, y_change)
    evaluate(model, X, y_change, meta)
    show_importance(model, X)
    
    with open('model_V8.3.1_10min.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('features_V8.3.1_10min.pkl', 'wb') as f:
        pickle.dump(list(X.columns), f)
    
    print(f"\n💾 저장: model_V8.3.1_10min.pkl")
    print(f"✅ V8.3.1 학습 완료! Feature: {X.shape[1]}개")