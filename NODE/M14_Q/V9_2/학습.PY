#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 V9.0 학습 - 10분 예측 (Deep 버전 + 과적합 방지)
Feature 82개 (queue_gap 복원!)
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
print("🚀 V9.0 학습 - 10분 예측 (Deep + 과적합 방지)")
print("="*70)

# ========================================
# Feature 생성 함수 (82개)
# ========================================
def create_features_v9(row_dict):
    features = {}
    
    seq_m14b = np.array(row_dict['M14AM14B'])
    seq_m14bsum = np.array(row_dict['M14AM14BSUM'])
    seq_m10arev = np.array(row_dict['M10AM14A'])
    seq_totalcnt = np.array(row_dict['TOTALCNT'])
    seq_transport = np.array(row_dict['TRANSPORT'])
    seq_oht = np.array(row_dict['OHT'])
    seq_q_created = np.array(row_dict['Q_CREATED'])
    seq_q_completed = np.array(row_dict['Q_COMPLETED'])
    seq_gap = seq_q_created - seq_q_completed
    
    seq_len = len(seq_m14b)
    
    # M14AM14B (8개)
    features['m14b_mean'] = np.mean(seq_m14b)
    features['m14b_max'] = np.max(seq_m14b)
    features['m14b_current'] = seq_m14b[-1]
    features['m14b_last10'] = np.mean(seq_m14b[-10:])
    features['m14b_last30'] = np.mean(seq_m14b[-30:])
    features['m14b_slope'] = np.polyfit(np.arange(seq_len), seq_m14b, 1)[0]
    features['m14b_std'] = np.std(seq_m14b)
    features['m14b_trend10'] = seq_m14b[-1] - seq_m14b[-10]
    
    # M14AM14BSUM (8개)
    features['m14bsum_mean'] = np.mean(seq_m14bsum)
    features['m14bsum_max'] = np.max(seq_m14bsum)
    features['m14bsum_current'] = seq_m14bsum[-1]
    features['m14bsum_last10'] = np.mean(seq_m14bsum[-10:])
    features['m14bsum_last30'] = np.mean(seq_m14bsum[-30:])
    features['m14bsum_slope'] = np.polyfit(np.arange(seq_len), seq_m14bsum, 1)[0]
    features['m14bsum_std'] = np.std(seq_m14bsum)
    features['m14bsum_trend10'] = seq_m14bsum[-1] - seq_m14bsum[-10]
    
    # TOTALCNT (10개)
    features['total_mean'] = np.mean(seq_totalcnt)
    features['total_max'] = np.max(seq_totalcnt)
    features['total_min'] = np.min(seq_totalcnt)
    features['total_current'] = seq_totalcnt[-1]
    features['total_last5'] = np.mean(seq_totalcnt[-5:])
    features['total_last10'] = np.mean(seq_totalcnt[-10:])
    features['total_last30'] = np.mean(seq_totalcnt[-30:])
    features['total_slope'] = np.polyfit(np.arange(seq_len), seq_totalcnt, 1)[0]
    features['total_std'] = np.std(seq_totalcnt)
    features['total_trend10'] = seq_totalcnt[-1] - seq_totalcnt[-10]
    
    # M10AM14A (5개)
    features['m10arev_mean'] = np.mean(seq_m10arev)
    features['m10arev_max'] = np.max(seq_m10arev)
    features['m10arev_current'] = seq_m10arev[-1]
    features['m10arev_last10'] = np.mean(seq_m10arev[-10:])
    features['m10arev_slope'] = np.polyfit(np.arange(seq_len), seq_m10arev, 1)[0]
    
    # TRANSPORT (5개)
    features['trans_mean'] = np.mean(seq_transport)
    features['trans_max'] = np.max(seq_transport)
    features['trans_current'] = seq_transport[-1]
    features['trans_last10'] = np.mean(seq_transport[-10:])
    features['trans_slope'] = np.polyfit(np.arange(seq_len), seq_transport, 1)[0]
    
    # OHT (4개)
    features['oht_mean'] = np.mean(seq_oht)
    features['oht_max'] = np.max(seq_oht)
    features['oht_current'] = seq_oht[-1]
    features['oht_last10'] = np.mean(seq_oht[-10:])
    
    # Queue Gap (10개)
    features['gap_mean'] = np.mean(seq_gap)
    features['gap_max'] = np.max(seq_gap)
    features['gap_min'] = np.min(seq_gap)
    features['gap_current'] = seq_gap[-1]
    features['gap_last5'] = np.mean(seq_gap[-5:])
    features['gap_last10'] = np.mean(seq_gap[-10:])
    features['gap_last30'] = np.mean(seq_gap[-30:])
    features['gap_slope'] = np.polyfit(np.arange(seq_len), seq_gap, 1)[0]
    features['gap_std'] = np.std(seq_gap)
    features['gap_trend10'] = seq_gap[-1] - seq_gap[-10]
    
    # Interaction (12개)
    features['m14b_x_sum'] = seq_m14b[-1] * seq_m14bsum[-1] / 1000
    features['m14b_x_sum_mean'] = np.mean(seq_m14b * seq_m14bsum) / 1000
    features['sum_per_m14b'] = seq_m14bsum[-1] / (seq_m14b[-1] + 1)
    features['m14b_plus_sum'] = seq_m14b[-1] + seq_m14bsum[-1]
    features['m10arev_x_m14b'] = seq_m10arev[-1] * seq_m14b[-1] / 100
    features['trans_x_m14b'] = seq_transport[-1] * seq_m14b[-1] / 100
    features['ratio_m14b_total'] = seq_m14b[-1] / (seq_totalcnt[-1] + 1)
    features['ratio_sum_total'] = seq_m14bsum[-1] / (seq_totalcnt[-1] + 1)
    features['gap_x_m14b'] = seq_gap[-1] * seq_m14b[-1] / 1000
    features['gap_x_total'] = seq_gap[-1] * seq_totalcnt[-1] / 1000
    features['gap_x_trans'] = seq_gap[-1] * seq_transport[-1] / 100
    features['ratio_gap_total'] = seq_gap[-1] / (seq_totalcnt[-1] + 1)
    
    # 임계값 (12개)
    features['m14b_over_520'] = np.sum(seq_m14b > 520)
    features['m14b_over_540'] = np.sum(seq_m14b > 540)
    features['m14bsum_over_600'] = np.sum(seq_m14bsum > 600)
    features['m14bsum_over_620'] = np.sum(seq_m14bsum > 620)
    features['m10arev_over_55'] = np.sum(seq_m10arev > 55)
    features['total_over_1600'] = np.sum(seq_totalcnt >= 1600)
    features['total_over_1650'] = np.sum(seq_totalcnt >= 1650)
    features['total_over_1700'] = np.sum(seq_totalcnt >= 1700)
    features['gap_over_200'] = np.sum(seq_gap > 200)
    features['gap_over_250'] = np.sum(seq_gap > 250)
    features['gap_over_300'] = np.sum(seq_gap > 300)
    features['gap_over_350'] = np.sum(seq_gap > 350)
    
    # 황금 패턴 (8개)
    features['gold_strict'] = 1 if (seq_m14b[-1] > 540 and seq_m14bsum[-1] > 620) else 0
    features['gold_normal'] = 1 if (seq_m14b[-1] > 520 and seq_m14bsum[-1] > 600) else 0
    features['in_danger'] = 1 if seq_totalcnt[-1] >= 1700 else 0
    features['near_danger'] = 1 if seq_totalcnt[-1] >= 1600 else 0
    features['danger_gap'] = 1 if seq_gap[-1] > 300 else 0
    features['danger_trans'] = 1 if seq_transport[-1] > 151 else 0
    features['triple_check'] = 1 if (seq_m14b[-1] > 520 and seq_m14bsum[-1] > 600 and seq_gap[-1] > 250) else 0
    features['quad_check'] = 1 if (seq_m14b[-1] > 520 and seq_m14bsum[-1] > 600 and seq_gap[-1] > 250 and seq_transport[-1] > 145) else 0
    
    return features

# ========================================
# 데이터 준비
# ========================================
def prepare_data(csv_path, seq_len=280, pred_minutes=10):
    print(f"\n📂 데이터 로딩: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✅ {len(df):,}행 × {df.shape[1]}열")
    
    required = ['M14AM14B', 'M14AM14BSUM', 'M10AM14A', 'TOTALCNT',
                'M14.QUE.ALL.TRANSPORT4MINOVERCNT', 'M14.QUE.OHT.OHTUTIL',
                'M14.QUE.ALL.CURRENTQCREATED', 'M14.QUE.ALL.CURRENTQCOMPLETED']
    
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"❌ 누락 컬럼: {missing}")
        return None, None
    
    print(f"✅ 8개 컬럼 확인 (queue 포함!)")
    
    before = len(df)
    df = df.dropna(subset=required)
    after = len(df)
    if before != after:
        print(f"⚠️ NaN 제거: {before:,} → {after:,}")
    
    high_ratio = (df['TOTALCNT'] >= 1700).sum() / len(df) * 100
    print(f"📊 1700+ 비율: {high_ratio:.2f}%")
    
    print(f"\n🔄 Feature 생성 중... ({pred_minutes}분 예측)")
    X_list, y_list = [], []
    pred_offset = pred_minutes
    total = len(df) - seq_len - pred_offset
    
    for i in range(seq_len, len(df) - pred_offset):
        row_dict = {
            'M14AM14B': df['M14AM14B'].iloc[i-seq_len:i].values,
            'M14AM14BSUM': df['M14AM14BSUM'].iloc[i-seq_len:i].values,
            'M10AM14A': df['M10AM14A'].iloc[i-seq_len:i].values,
            'TOTALCNT': df['TOTALCNT'].iloc[i-seq_len:i].values,
            'TRANSPORT': df['M14.QUE.ALL.TRANSPORT4MINOVERCNT'].iloc[i-seq_len:i].values,
            'OHT': df['M14.QUE.OHT.OHTUTIL'].iloc[i-seq_len:i].values,
            'Q_CREATED': df['M14.QUE.ALL.CURRENTQCREATED'].iloc[i-seq_len:i].values,
            'Q_COMPLETED': df['M14.QUE.ALL.CURRENTQCOMPLETED'].iloc[i-seq_len:i].values,
        }
        
        target = df['TOTALCNT'].iloc[i + pred_offset - 1]
        if pd.isna(target):
            continue
            
        X_list.append(create_features_v9(row_dict))
        y_list.append(target)
        
        if (i - seq_len) % 5000 == 0:
            print(f"  {i-seq_len:,}/{total:,} ({(i-seq_len)/total*100:.1f}%)")
    
    print(f"✅ 완료!")
    
    X = pd.DataFrame(X_list)
    y = pd.Series(y_list)
    
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)
    
    print(f"\n📊 결과:")
    print(f"  Feature: {X.shape[1]}개")
    print(f"  샘플: {len(X):,}개")
    print(f"  타겟 범위: {y.min():.0f}~{y.max():.0f}")
    print(f"  1700+: {(y>=1700).sum()}개 ({(y>=1700).sum()/len(y)*100:.2f}%)")
    
    return X, y

# ========================================
# 데이터 증강 (1:1 비율)
# ========================================
def augment_data(X, y, threshold=1700, multiplier=10):
    print(f"\n🔥 1700+ 데이터 {multiplier}배 증강 (1:1 비율)")
    high_mask = y >= threshold
    count = high_mask.sum()
    print(f"1700+ 원본: {count:,}개")
    
    if count == 0:
        return X, y
    
    X_aug = pd.concat([X] + [X[high_mask]] * (multiplier - 1), ignore_index=True)
    y_aug = pd.concat([y] + [y[high_mask]] * (multiplier - 1), ignore_index=True)
    
    print(f"증강 후: {len(X_aug):,}개")
    print(f"1700+ 비율: {(y_aug>=threshold).sum()/len(y_aug)*100:.1f}%")
    
    return X_aug, y_aug

# ========================================
# 모델 학습 (Deep + 과적합 방지)
# ========================================
def train_model(X, y):
    print(f"\n🚀 XGBoost Deep 학습 (과적합 방지)")
    print(f"  X NaN: {X.isna().sum().sum()}, y NaN: {y.isna().sum()}")
    
    # 학습/검증 분리
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    print(f"  학습: {len(X_train):,}개, 검증: {len(X_val):,}개")
    
    # Deep + 과적합 방지 설정
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        
        # Deep 설정
        max_depth=10,              # 깊게 but 과적합 방지
        n_estimators=2000,         # 충분히 많이
        learning_rate=0.01,        # 천천히 세밀하게
        
        # 과적합 방지 강화
        subsample=0.6,
        colsample_bytree=0.6,
        colsample_bylevel=0.5,
        min_child_weight=10,
        gamma=0.3,
        reg_alpha=0.5,
        reg_lambda=3.0,
        
        random_state=42,
        n_jobs=-1,
        tree_method='hist'
    )
    
    # Early Stopping으로 학습
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        early_stopping_rounds=100,  # 100회 개선 없으면 중단
        verbose=100
    )
    
    print(f"\n✅ Deep 학습 완료!")
    return model, X_val, y_val

# ========================================
# 성능 평가
# ========================================
def evaluate(model, X, y, X_val=None, y_val=None):
    print(f"\n" + "="*70)
    print(f"📊 성능 평가")
    print("="*70)
    
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    print(f"\n[전체] MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
    
    # 검증 성능
    if X_val is not None and y_val is not None:
        y_val_pred = model.predict(X_val)
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_r2 = r2_score(y_val, y_val_pred)
        print(f"[검증] MAE: {val_mae:.2f}, RMSE: {val_rmse:.2f}, R²: {val_r2:.4f}")
        
        # 과적합 체크
        gap = val_rmse - rmse
        if gap > 5:
            print(f"⚠️ 과적합 주의! (차이: {gap:.2f})")
        else:
            print(f"✅ 과적합 없음 (차이: {gap:.2f})")
    
    # 1700+ 감지율
    high = y >= 1700
    if high.sum() > 0:
        detected = ((y >= 1700) & (y_pred >= 1680)).sum()
        print(f"\n🔥 1700+ 감지율: {detected/high.sum()*100:.1f}% ({detected:,}/{high.sum():,})")
    
    return mae, rmse, r2

# ========================================
# Feature 중요도
# ========================================
def show_importance(model, X):
    print(f"\n🏆 Feature 중요도 TOP 20")
    imp = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(imp.head(20).to_string(index=False))
    return imp

# ========================================
# 메인 실행
# ========================================
if __name__ == '__main__':
    # 데이터 파일 경로
    CSV_FILE = "1700_데이터_고품질데이터.csv"
    
    # 1. 데이터 준비
    X, y = prepare_data(CSV_FILE, pred_minutes=10)
    if X is None:
        exit(1)
    
    # 2. 데이터 증강 (1:1 비율)
    X_aug, y_aug = augment_data(X, y, multiplier=10)
    
    # 3. 모델 학습
    model, X_val, y_val = train_model(X_aug, y_aug)
    
    # 4. 성능 평가
    mae, rmse, r2 = evaluate(model, X_aug, y_aug, X_val, y_val)
    
    # 5. Feature 중요도
    show_importance(model, X_aug)
    
    # 6. 모델 저장
    with open('model_v9_10min.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('features_v9_10min.pkl', 'wb') as f:
        pickle.dump(list(X.columns), f)
    
    print(f"\n💾 저장: model_v9_10min.pkl, features_v9_10min.pkl")
    print(f"✅ V9.0 Deep 학습 완료! Feature: {X.shape[1]}개")