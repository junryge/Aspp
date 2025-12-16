#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 V9.3 학습 - 품질 데이터 기반

핵심:
1. 균형 데이터 사용 (증강 X!)
2. Sample Weight로 중요 구간 강조
3. 패턴 Feature 포함
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🚀 V9.3 학습 - 품질 데이터 기반")
print("   증강 없이! Sample Weight 사용!")
print("="*70)

# ============================================
# 설정
# ============================================
SEQ_LEN = 100
PRED_MINUTES = 10
CSV_FILE = "train_balanced.csv"  # ← 품질 처리된 데이터!

# ============================================
# Feature 생성 (40개)
# ============================================
def create_features_v93(row_dict):
    """40개 Feature (34 기본 + 6 패턴)"""
    features = {}
    
    seq_m14b = np.array(row_dict['M14B'])
    seq_m14bsum = np.array(row_dict['M14BSUM'])
    seq_totalcnt = np.array(row_dict['TOTALCNT'])
    seq_q_created = np.array(row_dict['Q_CREATED'])
    seq_q_completed = np.array(row_dict['Q_COMPLETED'])
    seq_gap = seq_q_created - seq_q_completed
    
    seq_len = len(seq_m14b)
    
    cur_m14b = seq_m14b[-1]
    cur_m14bsum = seq_m14bsum[-1]
    cur_gap = seq_gap[-1]
    cur_total = seq_totalcnt[-1]
    
    # TOTALCNT (8개)
    features['total_current'] = cur_total
    features['total_mean'] = np.mean(seq_totalcnt)
    features['total_max'] = np.max(seq_totalcnt)
    features['total_min'] = np.min(seq_totalcnt)
    features['total_last10'] = np.mean(seq_totalcnt[-10:])
    features['total_slope'] = np.polyfit(np.arange(seq_len), seq_totalcnt, 1)[0]
    features['total_std'] = np.std(seq_totalcnt)
    features['total_trend10'] = seq_totalcnt[-1] - seq_totalcnt[-10]
    
    # M14AM14B (6개)
    features['m14b_current'] = cur_m14b
    features['m14b_mean'] = np.mean(seq_m14b)
    features['m14b_max'] = np.max(seq_m14b)
    features['m14b_last10'] = np.mean(seq_m14b[-10:])
    features['m14b_slope'] = np.polyfit(np.arange(seq_len), seq_m14b, 1)[0]
    features['m14b_trend10'] = seq_m14b[-1] - seq_m14b[-10]
    
    # M14AM14BSUM (6개)
    features['m14bsum_current'] = cur_m14bsum
    features['m14bsum_mean'] = np.mean(seq_m14bsum)
    features['m14bsum_max'] = np.max(seq_m14bsum)
    features['m14bsum_last10'] = np.mean(seq_m14bsum[-10:])
    features['m14bsum_slope'] = np.polyfit(np.arange(seq_len), seq_m14bsum, 1)[0]
    features['m14bsum_trend10'] = seq_m14bsum[-1] - seq_m14bsum[-10]
    
    # Queue Gap (8개)
    features['gap_current'] = cur_gap
    features['gap_mean'] = np.mean(seq_gap)
    features['gap_max'] = np.max(seq_gap)
    features['gap_min'] = np.min(seq_gap)
    features['gap_last10'] = np.mean(seq_gap[-10:])
    features['gap_slope'] = np.polyfit(np.arange(seq_len), seq_gap, 1)[0]
    features['gap_std'] = np.std(seq_gap)
    features['gap_trend10'] = seq_gap[-1] - seq_gap[-10]
    
    # 조합 (6개)
    features['m14b_x_sum'] = cur_m14b * cur_m14bsum / 1000
    features['gap_x_m14b'] = cur_gap * cur_m14b / 1000
    features['ratio_gap_total'] = cur_gap / (cur_total + 1)
    features['ratio_m14b_total'] = cur_m14b / (cur_total + 1)
    features['m14b_plus_sum'] = cur_m14b + cur_m14bsum
    features['gap_per_m14b'] = cur_gap / (cur_m14b + 1)
    
    # 패턴 Feature (6개)
    features['is_gold_pattern'] = 1 if (cur_m14b > 520 and cur_m14bsum > 600) else 0
    features['is_triple_check'] = 1 if (cur_m14b > 520 and cur_m14bsum > 600 and cur_gap > 250) else 0
    features['is_gap_critical'] = 1 if cur_gap > 400 else 0
    features['is_gap_extreme'] = 1 if cur_gap > 500 else 0
    features['is_near_danger'] = 1 if cur_total >= 1650 else 0
    features['is_in_danger'] = 1 if cur_total >= 1700 else 0
    
    return features

# ============================================
# Sample Weight 계산
# ============================================
def calculate_weights(y):
    """타겟 구간별 가중치"""
    weights = np.ones(len(y))
    
    # 1700+ 가중치 높임
    weights[y >= 1700] = 5.0
    
    # 1650~1700 (전이 구간) 가중치
    weights[(y >= 1650) & (y < 1700)] = 3.0
    
    # 1600~1650 가중치
    weights[(y >= 1600) & (y < 1650)] = 2.0
    
    print(f"\n📊 Sample Weight 분포:")
    print(f"  weight=1.0: {(weights == 1.0).sum():,}개")
    print(f"  weight=2.0: {(weights == 2.0).sum():,}개")
    print(f"  weight=3.0: {(weights == 3.0).sum():,}개")
    print(f"  weight=5.0: {(weights == 5.0).sum():,}개")
    
    return weights

# ============================================
# 데이터 준비
# ============================================
def prepare_data(csv_path):
    print(f"\n📂 데이터 로딩: {csv_path}")
    df = pd.read_csv(csv_path, on_bad_lines='skip')
    print(f"✅ {len(df):,}행")
    
    # 필수 컬럼
    required = {
        'TOTALCNT': 'TOTALCNT',
        'M14B': 'M14AM14B',
        'M14BSUM': 'M14AM14BSUM',
        'Q_CREATED': 'M14.QUE.ALL.CURRENTQCREATED',
        'Q_COMPLETED': 'M14.QUE.ALL.CURRENTQCOMPLETED',
    }
    
    for key, col in required.items():
        if col not in df.columns:
            print(f"❌ 누락: {col}")
            return None, None, None
    
    df = df.dropna(subset=list(required.values()))
    
    # 분포 확인
    print(f"\n📊 타겟 분포:")
    print(f"  ~1400: {(df['TOTALCNT'] < 1400).sum():,}개")
    print(f"  1400~1600: {((df['TOTALCNT'] >= 1400) & (df['TOTALCNT'] < 1600)).sum():,}개")
    print(f"  1600~1700: {((df['TOTALCNT'] >= 1600) & (df['TOTALCNT'] < 1700)).sum():,}개")
    print(f"  1700+: {(df['TOTALCNT'] >= 1700).sum():,}개")
    
    # Feature 생성
    print(f"\n🔄 Feature 생성 중...")
    X_list, y_list = [], []
    
    total = len(df) - SEQ_LEN - PRED_MINUTES
    
    for i in range(SEQ_LEN, len(df) - PRED_MINUTES):
        row_dict = {
            'TOTALCNT': df['TOTALCNT'].iloc[i-SEQ_LEN:i].values,
            'M14B': df['M14AM14B'].iloc[i-SEQ_LEN:i].values,
            'M14BSUM': df['M14AM14BSUM'].iloc[i-SEQ_LEN:i].values,
            'Q_CREATED': df['M14.QUE.ALL.CURRENTQCREATED'].iloc[i-SEQ_LEN:i].values,
            'Q_COMPLETED': df['M14.QUE.ALL.CURRENTQCOMPLETED'].iloc[i-SEQ_LEN:i].values,
        }
        
        target = df['TOTALCNT'].iloc[i + PRED_MINUTES - 1]
        if pd.isna(target):
            continue
        
        X_list.append(create_features_v93(row_dict))
        y_list.append(target)
        
        if (i - SEQ_LEN) % 2000 == 0:
            print(f"  {i-SEQ_LEN:,}/{total:,}")
    
    X = pd.DataFrame(X_list)
    y = pd.Series(y_list)
    
    # NaN 제거
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)
    
    # Sample Weight
    weights = calculate_weights(y.values)
    
    print(f"\n📊 최종:")
    print(f"  Feature: {X.shape[1]}개")
    print(f"  샘플: {len(X):,}개")
    
    return X, y, weights

# ============================================
# 모델 학습
# ============================================
def train_model(X, y, weights):
    print(f"\n🚀 XGBoost 학습 (Sample Weight 적용!)")
    
    # 분할
    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X, y, weights, test_size=0.2, random_state=42
    )
    print(f"  Train: {len(X_train):,}, Val: {len(X_val):,}")
    
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        max_depth=6,
        learning_rate=0.05,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method='hist',
        early_stopping_rounds=50
    )
    
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        verbose=50
    )
    
    # 평가
    y_val_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_val_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    r2 = r2_score(y_val, y_val_pred)
    
    print(f"\n📊 검증 성능:")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²: {r2:.4f}")
    
    # 1700+ 감지
    val_high = y_val >= 1700
    if val_high.sum() > 0:
        # 1680 이상이면 감지로 간주
        detected_1680 = ((y_val >= 1700) & (y_val_pred >= 1680)).sum()
        detected_1700 = ((y_val >= 1700) & (y_val_pred >= 1700)).sum()
        print(f"\n🔥 1700+ 감지율:")
        print(f"  예측>=1680: {detected_1680}/{val_high.sum()} ({detected_1680/val_high.sum()*100:.1f}%)")
        print(f"  예측>=1700: {detected_1700}/{val_high.sum()} ({detected_1700/val_high.sum()*100:.1f}%)")
    
    # 구간별 MAE
    print(f"\n📊 구간별 MAE:")
    for low, high, name in [(0, 1400, '~1400'), (1400, 1600, '1400~1600'), 
                            (1600, 1700, '1600~1700'), (1700, 9999, '1700+')]:
        mask = (y_val >= low) & (y_val < high)
        if mask.sum() > 0:
            seg_mae = mean_absolute_error(y_val[mask], y_val_pred[mask])
            print(f"  {name}: MAE={seg_mae:.2f} ({mask.sum()}개)")
    
    return model

# ============================================
# Feature 중요도
# ============================================
def show_importance(model, X):
    print(f"\n🏆 Feature 중요도 TOP 15")
    imp = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(imp.head(15).to_string(index=False))
    
    # 패턴 Feature
    print(f"\n📊 패턴 Feature 중요도:")
    patterns = ['is_gold_pattern', 'is_triple_check', 'is_gap_critical', 
                'is_gap_extreme', 'is_near_danger', 'is_in_danger']
    for pf in patterns:
        val = imp[imp['feature'] == pf]['importance'].values[0]
        marker = "⭐" if val > 0.01 else ""
        print(f"  {pf}: {val:.4f} {marker}")
    
    return imp

# ============================================
# 메인
# ============================================
if __name__ == '__main__':
    X, y, weights = prepare_data(CSV_FILE)
    if X is None:
        exit(1)
    
    model = train_model(X, y, weights)
    show_importance(model, X)
    
    # 저장
    with open('model_v93.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('features_v93.pkl', 'wb') as f:
        pickle.dump(list(X.columns), f)
    
    print(f"\n💾 저장:")
    print(f"  모델: model_v93.pkl")
    print(f"  Feature: features_v93.pkl")
    
    print(f"\n" + "="*70)
    print(f"✅ V9.3 학습 완료!")
    print(f"   데이터: 균형 샘플링 (증강 없음!)")
    print(f"   Sample Weight: 1700+ = 5배 가중치")
    print("="*70)