#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 V9.2 학습 - 100분/10분 예측
패턴을 Feature로! ML이 패턴 학습!

Feature 40개:
- 기본 34개
- 패턴 6개 (신규!)
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
print("🚀 V9.2 학습 - 100분/10분 예측")
print("   패턴을 Feature로! ML이 패턴 학습!")
print("="*70)

# ============================================
# 설정
# ============================================
SEQ_LEN = 100
PRED_MINUTES = 10
AUGMENT_MULTIPLIER = 50
CSV_FILE = "YOUR_CSV_FILE.csv"  # ← 학습 데이터!

# ============================================
# Feature 생성 (40개 = 34 + 패턴 6개)
# ============================================
def create_features_v92(row_dict):
    """34개 기본 + 6개 패턴 Feature"""
    features = {}
    
    seq_m14b = np.array(row_dict['M14B'])
    seq_m14bsum = np.array(row_dict['M14BSUM'])
    seq_totalcnt = np.array(row_dict['TOTALCNT'])
    seq_q_created = np.array(row_dict['Q_CREATED'])
    seq_q_completed = np.array(row_dict['Q_COMPLETED'])
    seq_gap = seq_q_created - seq_q_completed
    
    seq_len = len(seq_m14b)
    
    # 현재값 (패턴 체크용)
    cur_m14b = seq_m14b[-1]
    cur_m14bsum = seq_m14bsum[-1]
    cur_gap = seq_gap[-1]
    cur_total = seq_totalcnt[-1]
    
    # ========== TOTALCNT (8개) ==========
    features['total_current'] = cur_total
    features['total_mean'] = np.mean(seq_totalcnt)
    features['total_max'] = np.max(seq_totalcnt)
    features['total_min'] = np.min(seq_totalcnt)
    features['total_last10'] = np.mean(seq_totalcnt[-10:])
    features['total_slope'] = np.polyfit(np.arange(seq_len), seq_totalcnt, 1)[0]
    features['total_std'] = np.std(seq_totalcnt)
    features['total_trend10'] = seq_totalcnt[-1] - seq_totalcnt[-10]
    
    # ========== M14AM14B (6개) ==========
    features['m14b_current'] = cur_m14b
    features['m14b_mean'] = np.mean(seq_m14b)
    features['m14b_max'] = np.max(seq_m14b)
    features['m14b_last10'] = np.mean(seq_m14b[-10:])
    features['m14b_slope'] = np.polyfit(np.arange(seq_len), seq_m14b, 1)[0]
    features['m14b_trend10'] = seq_m14b[-1] - seq_m14b[-10]
    
    # ========== M14AM14BSUM (6개) ==========
    features['m14bsum_current'] = cur_m14bsum
    features['m14bsum_mean'] = np.mean(seq_m14bsum)
    features['m14bsum_max'] = np.max(seq_m14bsum)
    features['m14bsum_last10'] = np.mean(seq_m14bsum[-10:])
    features['m14bsum_slope'] = np.polyfit(np.arange(seq_len), seq_m14bsum, 1)[0]
    features['m14bsum_trend10'] = seq_m14bsum[-1] - seq_m14bsum[-10]
    
    # ========== Queue Gap (8개) ==========
    features['gap_current'] = cur_gap
    features['gap_mean'] = np.mean(seq_gap)
    features['gap_max'] = np.max(seq_gap)
    features['gap_min'] = np.min(seq_gap)
    features['gap_last10'] = np.mean(seq_gap[-10:])
    features['gap_slope'] = np.polyfit(np.arange(seq_len), seq_gap, 1)[0]
    features['gap_std'] = np.std(seq_gap)
    features['gap_trend10'] = seq_gap[-1] - seq_gap[-10]
    
    # ========== 조합 (6개) ==========
    features['m14b_x_sum'] = cur_m14b * cur_m14bsum / 1000
    features['gap_x_m14b'] = cur_gap * cur_m14b / 1000
    features['ratio_gap_total'] = cur_gap / (cur_total + 1)
    features['ratio_m14b_total'] = cur_m14b / (cur_total + 1)
    features['m14b_plus_sum'] = cur_m14b + cur_m14bsum
    features['gap_per_m14b'] = cur_gap / (cur_m14b + 1)
    
    # ========== 🔥 패턴 Feature (6개) - 신규! ==========
    # ML이 이걸 보고 학습!
    features['is_gold_pattern'] = 1 if (cur_m14b > 520 and cur_m14bsum > 600) else 0
    features['is_triple_check'] = 1 if (cur_m14b > 520 and cur_m14bsum > 600 and cur_gap > 250) else 0
    features['is_gap_critical'] = 1 if cur_gap > 400 else 0
    features['is_gap_extreme'] = 1 if cur_gap > 500 else 0
    features['is_near_danger'] = 1 if cur_total >= 1650 else 0
    features['is_in_danger'] = 1 if cur_total >= 1700 else 0
    
    return features

# ============================================
# 데이터 준비
# ============================================
def prepare_data(csv_path):
    print(f"\n📂 데이터 로딩: {csv_path}")
    df = pd.read_csv(csv_path, on_bad_lines='skip')
    print(f"✅ {len(df):,}행 × {df.shape[1]}열")
    
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
            return None, None
    
    print(f"✅ 필수 5개 컬럼 확인!")
    
    df = df.dropna(subset=list(required.values()))
    print(f"✅ NaN 제거 후: {len(df):,}행")
    
    df['queue_gap'] = df['M14.QUE.ALL.CURRENTQCREATED'] - df['M14.QUE.ALL.CURRENTQCOMPLETED']
    high_ratio = (df['TOTALCNT'] >= 1700).sum() / len(df) * 100
    print(f"📊 현재 1700+ 비율: {high_ratio:.2f}%")
    
    print(f"\n🔄 Feature 생성 중... (시퀀스={SEQ_LEN}분, 예측={PRED_MINUTES}분 후)")
    
    X_list, y_list = [], []
    pattern_count = 0
    
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
        
        features = create_features_v92(row_dict)
        
        # 패턴 카운트
        if features['is_triple_check'] == 1:
            pattern_count += 1
        
        X_list.append(features)
        y_list.append(target)
        
        if (i - SEQ_LEN) % 10000 == 0:
            print(f"  {i-SEQ_LEN:,}/{total:,} ({(i-SEQ_LEN)/total*100:.1f}%)")
    
    print(f"✅ 완료!")
    print(f"📊 Triple Check 패턴: {pattern_count:,}개 ({pattern_count/len(X_list)*100:.2f}%)")
    
    X = pd.DataFrame(X_list)
    y = pd.Series(y_list)
    
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)
    
    print(f"\n📊 최종 데이터:")
    print(f"  Feature: {X.shape[1]}개")
    print(f"  샘플: {len(X):,}개")
    print(f"  타겟 1700+: {(y>=1700).sum():,}개 ({(y>=1700).sum()/len(y)*100:.2f}%)")
    
    return X, y

# ============================================
# 데이터 증강
# ============================================
def augment_data(X, y, threshold=1700, multiplier=AUGMENT_MULTIPLIER):
    print(f"\n🔥 1700+ 데이터 {multiplier}배 증강")
    high_mask = y >= threshold
    count = high_mask.sum()
    print(f"  1700+ 원본: {count:,}개")
    
    if count == 0:
        print("  ⚠️ 1700+ 데이터 없음!")
        return X, y
    
    X_aug = pd.concat([X] + [X[high_mask]] * (multiplier - 1), ignore_index=True)
    y_aug = pd.concat([y] + [y[high_mask]] * (multiplier - 1), ignore_index=True)
    
    print(f"  증강 후: {len(X_aug):,}개")
    print(f"  1700+ 비율: {(y_aug>=threshold).sum()/len(y_aug)*100:.1f}%")
    
    return X_aug, y_aug

# ============================================
# 모델 학습
# ============================================
def train_model(X, y):
    print(f"\n🚀 XGBoost 학습")
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"  Train: {len(X_train):,}, Val: {len(X_val):,}")
    
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        max_depth=6,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method='hist'
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50
    )
    
    y_val_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_val_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    r2 = r2_score(y_val, y_val_pred)
    
    print(f"\n📊 검증 성능:")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²: {r2:.4f}")
    
    val_high = y_val >= 1700
    if val_high.sum() > 0:
        detected = ((y_val >= 1700) & (y_val_pred >= 1680)).sum()
        print(f"  1700+ 감지율: {detected}/{val_high.sum()} ({detected/val_high.sum()*100:.1f}%)")
    
    return model

# ============================================
# Feature 중요도
# ============================================
def show_importance(model, X):
    print(f"\n🏆 Feature 중요도 TOP 20")
    imp = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(imp.head(20).to_string(index=False))
    
    # 패턴 Feature 중요도
    print(f"\n📊 패턴 Feature 중요도:")
    pattern_features = ['is_gold_pattern', 'is_triple_check', 'is_gap_critical', 
                        'is_gap_extreme', 'is_near_danger', 'is_in_danger']
    for pf in pattern_features:
        if pf in imp['feature'].values:
            val = imp[imp['feature'] == pf]['importance'].values[0]
            print(f"  {pf}: {val:.4f}")
    
    return imp

# ============================================
# 메인
# ============================================
if __name__ == '__main__':
    X, y = prepare_data(CSV_FILE)
    if X is None:
        print("❌ 데이터 로드 실패!")
        exit(1)
    
    X_aug, y_aug = augment_data(X, y)
    model = train_model(X_aug, y_aug)
    show_importance(model, X_aug)
    
    # 저장
    with open('model_v92_100min_10min.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('features_v92.pkl', 'wb') as f:
        pickle.dump(list(X.columns), f)
    
    print(f"\n💾 저장:")
    print(f"  모델: model_v92_100min_10min.pkl")
    print(f"  Feature: features_v92.pkl ({len(X.columns)}개)")
    
    print(f"\n" + "="*70)
    print(f"✅ V9.2 학습 완료!")
    print(f"   시퀀스: {SEQ_LEN}분 / 예측: {PRED_MINUTES}분")
    print(f"   Feature: {X.shape[1]}개 (패턴 6개 포함!)")
    print("="*70)