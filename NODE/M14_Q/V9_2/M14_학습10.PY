#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 V9.0 학습 - 100분 시퀀스 / 10분 후 예측
패턴 우선 + ML 보조 하이브리드!

핵심 변경 (V8.3 → V9.0):
1. 시퀀스: 280분 → 100분
2. 컬럼: 8개 → 5개 필수만
3. Feature: 82개 → 34개
4. 패턴: ML Feature → 즉시 경보!
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
print("🚀 V9.0 학습 - 100분/10분 예측")
print("   패턴 우선 + ML 보조 하이브리드!")
print("="*70)

# ============================================
# 설정
# ============================================
SEQ_LEN = 100          # 280 → 100분!
PRED_MINUTES = 10      # 10분 후 예측
AUGMENT_MULTIPLIER = 50
CSV_FILE = "YOUR_CSV_FILE.csv"  # ← 실제 파일 경로!

# ============================================
# 패턴 체크 함수 (즉시 경보용)
# ============================================
def check_pattern(m14b, m14bsum, gap, totalcnt):
    """
    패턴 매칭 → 즉시 경보!
    반환: (레벨, 예측값, 확률) or None
    """
    # 1단계: 황금패턴 & gap>400 (100% 정밀도)
    if m14b > 520 and m14bsum > 600 and gap > 400:
        return ('CRITICAL', 1800, 100.0)
    
    # 2단계: gap만 매우 높음 (99.7%)
    if gap > 500:
        return ('CRITICAL', 1780, 99.7)
    
    # 3단계: Triple Check (96.5%)
    if m14b > 520 and m14bsum > 600 and gap > 250:
        return ('CRITICAL', 1750, 96.5)
    
    # 4단계: gap>400 단독 (96.2%)
    if gap > 400:
        return ('HIGH', 1730, 96.2)
    
    # 5단계: 현재값 이미 높음
    if totalcnt >= 1680:
        return ('HIGH', 1720, 85.0)
    
    if totalcnt >= 1650:
        return ('WARNING', 1700, 71.0)
    
    # 패턴 없음 → ML로
    return None

# ============================================
# Feature 생성 (34개)
# ============================================
def create_features_v9(row_dict):
    """필수 5개 컬럼 기반 34개 Feature"""
    features = {}
    
    seq_m14b = np.array(row_dict['M14B'])
    seq_m14bsum = np.array(row_dict['M14BSUM'])
    seq_totalcnt = np.array(row_dict['TOTALCNT'])
    seq_q_created = np.array(row_dict['Q_CREATED'])
    seq_q_completed = np.array(row_dict['Q_COMPLETED'])
    seq_gap = seq_q_created - seq_q_completed
    
    seq_len = len(seq_m14b)
    
    # ========== TOTALCNT (8개) ==========
    features['total_current'] = seq_totalcnt[-1]
    features['total_mean'] = np.mean(seq_totalcnt)
    features['total_max'] = np.max(seq_totalcnt)
    features['total_min'] = np.min(seq_totalcnt)
    features['total_last10'] = np.mean(seq_totalcnt[-10:])
    features['total_slope'] = np.polyfit(np.arange(seq_len), seq_totalcnt, 1)[0]
    features['total_std'] = np.std(seq_totalcnt)
    features['total_trend10'] = seq_totalcnt[-1] - seq_totalcnt[-10]
    
    # ========== M14AM14B (6개) ==========
    features['m14b_current'] = seq_m14b[-1]
    features['m14b_mean'] = np.mean(seq_m14b)
    features['m14b_max'] = np.max(seq_m14b)
    features['m14b_last10'] = np.mean(seq_m14b[-10:])
    features['m14b_slope'] = np.polyfit(np.arange(seq_len), seq_m14b, 1)[0]
    features['m14b_trend10'] = seq_m14b[-1] - seq_m14b[-10]
    
    # ========== M14AM14BSUM (6개) ==========
    features['m14bsum_current'] = seq_m14bsum[-1]
    features['m14bsum_mean'] = np.mean(seq_m14bsum)
    features['m14bsum_max'] = np.max(seq_m14bsum)
    features['m14bsum_last10'] = np.mean(seq_m14bsum[-10:])
    features['m14bsum_slope'] = np.polyfit(np.arange(seq_len), seq_m14bsum, 1)[0]
    features['m14bsum_trend10'] = seq_m14bsum[-1] - seq_m14bsum[-10]
    
    # ========== Queue Gap (8개) - 핵심! ==========
    features['gap_current'] = seq_gap[-1]
    features['gap_mean'] = np.mean(seq_gap)
    features['gap_max'] = np.max(seq_gap)
    features['gap_min'] = np.min(seq_gap)
    features['gap_last10'] = np.mean(seq_gap[-10:])
    features['gap_slope'] = np.polyfit(np.arange(seq_len), seq_gap, 1)[0]
    features['gap_std'] = np.std(seq_gap)
    features['gap_trend10'] = seq_gap[-1] - seq_gap[-10]
    
    # ========== 핵심 조합 (6개) ==========
    features['m14b_x_sum'] = seq_m14b[-1] * seq_m14bsum[-1] / 1000
    features['gap_x_m14b'] = seq_gap[-1] * seq_m14b[-1] / 1000
    features['ratio_gap_total'] = seq_gap[-1] / (seq_totalcnt[-1] + 1)
    features['ratio_m14b_total'] = seq_m14b[-1] / (seq_totalcnt[-1] + 1)
    features['m14b_plus_sum'] = seq_m14b[-1] + seq_m14bsum[-1]
    features['gap_per_m14b'] = seq_gap[-1] / (seq_m14b[-1] + 1)
    
    return features

# ============================================
# 데이터 준비
# ============================================
def prepare_data(csv_path):
    print(f"\n📂 데이터 로딩: {csv_path}")
    df = pd.read_csv(csv_path, on_bad_lines='skip')
    print(f"✅ {len(df):,}행 × {df.shape[1]}열")
    
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
            return None, None
    
    print(f"✅ 필수 5개 컬럼 확인!")
    
    # NaN 제거
    df = df.dropna(subset=list(required.values()))
    print(f"✅ NaN 제거 후: {len(df):,}행")
    
    # 통계
    df['queue_gap'] = df['M14.QUE.ALL.CURRENTQCREATED'] - df['M14.QUE.ALL.CURRENTQCOMPLETED']
    high_ratio = (df['TOTALCNT'] >= 1700).sum() / len(df) * 100
    print(f"📊 현재 1700+ 비율: {high_ratio:.2f}%")
    print(f"📊 queue_gap 범위: {df['queue_gap'].min():.0f} ~ {df['queue_gap'].max():.0f}")
    
    # Feature 생성
    print(f"\n🔄 Feature 생성 중... (시퀀스={SEQ_LEN}분, 예측={PRED_MINUTES}분 후)")
    
    X_list, y_list = [], []
    pattern_stats = {'CRITICAL': 0, 'HIGH': 0, 'WARNING': 0}
    
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
        
        # 현재 상태
        current_total = row_dict['TOTALCNT'][-1]
        current_m14b = row_dict['M14B'][-1]
        current_m14bsum = row_dict['M14BSUM'][-1]
        current_gap = row_dict['Q_CREATED'][-1] - row_dict['Q_COMPLETED'][-1]
        
        # 패턴 통계
        pattern = check_pattern(current_m14b, current_m14bsum, current_gap, current_total)
        if pattern:
            pattern_stats[pattern[0]] += 1
        
        # Feature 생성
        X_list.append(create_features_v9(row_dict))
        y_list.append(target)
        
        if (i - SEQ_LEN) % 10000 == 0:
            print(f"  {i-SEQ_LEN:,}/{total:,} ({(i-SEQ_LEN)/total*100:.1f}%)")
    
    print(f"✅ 완료!")
    print(f"\n📊 패턴 매칭 통계 (실시간에서 즉시 처리):")
    for level, count in pattern_stats.items():
        print(f"  {level}: {count:,}개")
    
    X = pd.DataFrame(X_list)
    y = pd.Series(y_list)
    
    # NaN 제거
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
    
    # Train/Val 분리
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
    
    # 검증
    y_val_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_val_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    r2 = r2_score(y_val, y_val_pred)
    
    print(f"\n📊 검증 성능:")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²: {r2:.4f}")
    
    # 1700+ 감지율
    val_high = y_val >= 1700
    if val_high.sum() > 0:
        detected = ((y_val >= 1700) & (y_val_pred >= 1680)).sum()
        print(f"  1700+ 감지율: {detected}/{val_high.sum()} ({detected/val_high.sum()*100:.1f}%)")
    
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
    return imp

# ============================================
# 메인
# ============================================
if __name__ == '__main__':
    # 데이터 준비
    X, y = prepare_data(CSV_FILE)
    if X is None:
        print("❌ 데이터 로드 실패!")
        exit(1)
    
    # 증강
    X_aug, y_aug = augment_data(X, y)
    
    # 학습
    model = train_model(X_aug, y_aug)
    
    # Feature 중요도
    show_importance(model, X_aug)
    
    # 저장
    with open('model_v9_100min_10min.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('features_v9.pkl', 'wb') as f:
        pickle.dump(list(X.columns), f)
    
    print(f"\n💾 저장 완료:")
    print(f"  모델: model_v9_100min_10min.pkl")
    print(f"  Feature: features_v9.pkl")
    
    print(f"\n" + "="*70)
    print(f"✅ V9.0 학습 완료!")
    print(f"   시퀀스: {SEQ_LEN}분 / 예측: {PRED_MINUTES}분 후")
    print(f"   Feature: {X.shape[1]}개")
    print("="*70)