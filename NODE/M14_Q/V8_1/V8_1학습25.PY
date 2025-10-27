# -*- coding: utf-8 -*-
"""
280분 시퀀스 → 25분 후 TOTALCNT 최대값 예측
조기 경보 Feature 추가: 마지막 25분에 1650+ 상승 추세
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def check_gpu_availability():
    """GPU 사용 가능 여부 확인"""
    print("\n" + "="*80)
    print("GPU 사용 가능 여부 확인")
    print("="*80)
    
    try:
        test_X = np.random.rand(100, 10)
        test_y = np.random.rand(100)
        
        test_model = xgb.XGBRegressor(
            tree_method='gpu_hist',
            gpu_id=0,
            n_estimators=10
        )
        test_model.fit(test_X, test_y)
        
        print("GPU 사용 가능! XGBoost GPU 모드로 학습")
        return 'gpu_hist', 0
        
    except Exception as e:
        print(f"GPU 사용 불가: {str(e)[:100]}")
        print("CPU 모드로 학습")
        return 'hist', None

def create_features_280min_enhanced(df, start_idx=280):
    """
    280분 시퀀스 + 강화된 Feature + 조기 경보 Feature
    - 기본 8개 통계값 (각 컬럼)
    - 비율 Feature (m14b/m10a 등)
    - 임계값 카운트
    - 변화율, 가속도 등
    - 🔥 조기 경보: 마지막 25분에 1650+ 상승 추세
    """
    
    print(f"\nFeature 생성 중... (280분 시퀀스 + 강화 Feature + 조기 경보)")
    
    # 필수 컬럼 확인
    required_cols = ['M14AM14B', 'M14AM10A', 'M14AM16', 'TOTALCNT']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"필수 컬럼 누락: {missing_cols}")
    
    print(f"필수 컬럼 확인: {required_cols}")
    
    features_list = []
    labels = []
    
    seq_totalcnt_max_list = []
    seq_totalcnt_min_list = []
    indices = []
    
    total_sequences = len(df) - start_idx - 25
    print(f"생성 가능한 시퀀스: {total_sequences:,}개")
    
    # Feature 생성
    for i in range(start_idx, len(df) - 25):
        if i % 10000 == 0:
            progress = (i - start_idx) / total_sequences * 100
            print(f"  진행률: {i-start_idx}/{total_sequences} ({progress:.1f}%)", end='\r')
        
        # 280분 시퀀스 추출
        seq_m14b = df['M14AM14B'].iloc[i-280:i].values
        seq_m10a = df['M14AM10A'].iloc[i-280:i].values
        seq_m16 = df['M14AM16'].iloc[i-280:i].values
        seq_totalcnt = df['TOTALCNT'].iloc[i-280:i].values
        
        features = {}
        
        # ========== M14AM14B 기본 8개 ==========
        features['m14b_mean'] = np.mean(seq_m14b)
        features['m14b_std'] = np.std(seq_m14b)
        features['m14b_last_5_mean'] = np.mean(seq_m14b[-5:])
        features['m14b_max'] = np.max(seq_m14b)
        features['m14b_min'] = np.min(seq_m14b)
        features['m14b_slope'] = np.polyfit(np.arange(280), seq_m14b, 1)[0]
        features['m14b_last_10_mean'] = np.mean(seq_m14b[-10:])
        features['m14b_first_10_mean'] = np.mean(seq_m14b[:10])
        
        # ========== M14AM10A 기본 8개 ==========
        features['m10a_mean'] = np.mean(seq_m10a)
        features['m10a_std'] = np.std(seq_m10a)
        features['m10a_last_5_mean'] = np.mean(seq_m10a[-5:])
        features['m10a_max'] = np.max(seq_m10a)
        features['m10a_min'] = np.min(seq_m10a)
        features['m10a_slope'] = np.polyfit(np.arange(280), seq_m10a, 1)[0]
        features['m10a_last_10_mean'] = np.mean(seq_m10a[-10:])
        features['m10a_first_10_mean'] = np.mean(seq_m10a[:10])
        
        # ========== M14AM16 기본 8개 ==========
        features['m16_mean'] = np.mean(seq_m16)
        features['m16_std'] = np.std(seq_m16)
        features['m16_last_5_mean'] = np.mean(seq_m16[-5:])
        features['m16_max'] = np.max(seq_m16)
        features['m16_min'] = np.min(seq_m16)
        features['m16_slope'] = np.polyfit(np.arange(280), seq_m16, 1)[0]
        features['m16_last_10_mean'] = np.mean(seq_m16[-10:])
        features['m16_first_10_mean'] = np.mean(seq_m16[:10])
        
        # ========== TOTALCNT 기본 8개 ==========
        features['totalcnt_mean'] = np.mean(seq_totalcnt)
        features['totalcnt_std'] = np.std(seq_totalcnt)
        features['totalcnt_last_5_mean'] = np.mean(seq_totalcnt[-5:])
        features['totalcnt_max'] = np.max(seq_totalcnt)
        features['totalcnt_min'] = np.min(seq_totalcnt)
        features['totalcnt_slope'] = np.polyfit(np.arange(280), seq_totalcnt, 1)[0]
        features['totalcnt_last_10_mean'] = np.mean(seq_totalcnt[-10:])
        features['totalcnt_first_10_mean'] = np.mean(seq_totalcnt[:10])
        
        # ========== 비율 Feature (8개) ==========
        features['ratio_m14b_m10a'] = seq_m14b[-1] / (seq_m10a[-1] + 1)
        features['ratio_m14b_m16'] = seq_m14b[-1] / (seq_m16[-1] + 1)
        features['ratio_m10a_m16'] = seq_m10a[-1] / (seq_m16[-1] + 1)
        features['ratio_m14b_m10a_mean'] = np.mean(seq_m14b) / (np.mean(seq_m10a) + 1)
        features['ratio_m14b_m16_mean'] = np.mean(seq_m14b) / (np.mean(seq_m16) + 1)
        features['ratio_m14b_m10a_max'] = np.max(seq_m14b) / (np.max(seq_m10a) + 1)
        features['volatility_m14b'] = np.std(seq_m14b) / (np.mean(seq_m14b) + 1)
        features['volatility_totalcnt'] = np.std(seq_totalcnt) / (np.mean(seq_totalcnt) + 1)
        
        # ========== M14AM14B 임계값 카운트 (8개) ==========
        features['m14b_over_250'] = np.sum(seq_m14b > 250)
        features['m14b_over_300'] = np.sum(seq_m14b > 300)
        features['m14b_over_350'] = np.sum(seq_m14b > 350)
        features['m14b_over_400'] = np.sum(seq_m14b > 400)
        features['m14b_over_450'] = np.sum(seq_m14b > 450)
        features['m14b_over_300_last30'] = np.sum(seq_m14b[-30:] > 300)
        features['m14b_over_350_last30'] = np.sum(seq_m14b[-30:] > 350)
        features['m14b_over_400_last30'] = np.sum(seq_m14b[-30:] > 400)
        
        # ========== M14AM10A 임계값 카운트 (4개) ==========
        features['m10a_over_70'] = np.sum(seq_m10a > 70)
        features['m10a_over_80'] = np.sum(seq_m10a > 80)
        features['m10a_under_80'] = np.sum(seq_m10a < 80)
        features['m10a_under_70'] = np.sum(seq_m10a < 70)
        
        # ========== TOTALCNT 임계값 카운트 (8개) ==========
        features['totalcnt_over_1400'] = np.sum(seq_totalcnt >= 1400)
        features['totalcnt_over_1500'] = np.sum(seq_totalcnt >= 1500)
        features['totalcnt_over_1600'] = np.sum(seq_totalcnt >= 1600)
        features['totalcnt_over_1700'] = np.sum(seq_totalcnt >= 1700)
        features['totalcnt_over_1400_last30'] = np.sum(seq_totalcnt[-30:] >= 1400)
        features['totalcnt_over_1500_last30'] = np.sum(seq_totalcnt[-30:] >= 1500)
        features['totalcnt_over_1600_last30'] = np.sum(seq_totalcnt[-30:] >= 1600)
        features['totalcnt_over_1700_last30'] = np.sum(seq_totalcnt[-30:] >= 1700)
        
        # ========== 황금 패턴 (4개) ==========
        features['golden_pattern_300_80'] = 1 if (seq_m14b[-1] > 300 and seq_m10a[-1] < 80) else 0
        features['golden_pattern_350_80'] = 1 if (seq_m14b[-1] > 350 and seq_m10a[-1] < 80) else 0
        features['golden_pattern_400_70'] = 1 if (seq_m14b[-1] > 400 and seq_m10a[-1] < 70) else 0
        features['danger_zone'] = 1 if seq_totalcnt[-1] >= 1700 else 0
        
        # ========== 변화율/가속도 (8개) ==========
        features['m14b_change_rate'] = (seq_m14b[-1] - seq_m14b[-30]) / 30 if len(seq_m14b) >= 30 else 0
        features['totalcnt_change_rate'] = (seq_totalcnt[-1] - seq_totalcnt[-30]) / 30 if len(seq_totalcnt) >= 30 else 0
        
        recent_30_m14b = np.mean(seq_m14b[-30:])
        previous_30_m14b = np.mean(seq_m14b[-60:-30]) if len(seq_m14b) >= 60 else np.mean(seq_m14b[-30:])
        features['m14b_acceleration'] = recent_30_m14b - previous_30_m14b
        
        recent_30_totalcnt = np.mean(seq_totalcnt[-30:])
        previous_30_totalcnt = np.mean(seq_totalcnt[-60:-30]) if len(seq_totalcnt) >= 60 else np.mean(seq_totalcnt[-30:])
        features['totalcnt_acceleration'] = recent_30_totalcnt - previous_30_totalcnt
        
        features['m14b_range'] = np.max(seq_m14b) - np.min(seq_m14b)
        features['totalcnt_range'] = np.max(seq_totalcnt) - np.min(seq_totalcnt)
        features['m14b_recent_vs_mean'] = np.mean(seq_m14b[-30:]) / (np.mean(seq_m14b) + 1)
        features['totalcnt_recent_vs_mean'] = np.mean(seq_totalcnt[-30:]) / (np.mean(seq_totalcnt) + 1)
        
        # ========== 시간대별 통계 (8개) ==========
        q1 = seq_totalcnt[:70]
        q2 = seq_totalcnt[70:140]
        q3 = seq_totalcnt[140:210]
        q4 = seq_totalcnt[210:280]
        
        features['totalcnt_q1_mean'] = np.mean(q1)
        features['totalcnt_q2_mean'] = np.mean(q2)
        features['totalcnt_q3_mean'] = np.mean(q3)
        features['totalcnt_q4_mean'] = np.mean(q4)
        features['totalcnt_trend_q1_q2'] = np.mean(q2) - np.mean(q1)
        features['totalcnt_trend_q2_q3'] = np.mean(q3) - np.mean(q2)
        features['totalcnt_trend_q3_q4'] = np.mean(q4) - np.mean(q3)
        features['totalcnt_trend_overall'] = np.mean(q4) - np.mean(q1)
        
        # ========== 🔥 조기 경보 Feature (5개) ==========
        # 마지막 25분 분석
        last_10min = seq_totalcnt[-10:]
        
        features['last_10min_max'] = np.max(last_10min)
        features['last_10min_min'] = np.min(last_10min)
        features['last_10min_mean'] = np.mean(last_10min)
        features['last_10min_rise'] = last_10min[-1] - last_10min[0]  # 25분간 상승폭
        
        # 조기 경보: 1650 이상 + 20 이상 상승
        early_warning = (np.max(last_10min) >= 1650) and ((last_10min[-1] - last_10min[0]) > 20)
        features['early_warning_1650_rising'] = 1 if early_warning else 0
        
        features_list.append(features)
        
        # 라벨: 25분 후 TOTALCNT 최대값
        future_totalcnt = df['TOTALCNT'].iloc[i:i+25].values
        labels.append(np.max(future_totalcnt))
        
        # 시퀀스 정보 저장
        seq_totalcnt_max_list.append(np.max(seq_totalcnt))
        seq_totalcnt_min_list.append(np.min(seq_totalcnt))
        indices.append(i)
    
    print(f"\nFeature 생성 완료: {len(features_list):,}개 시퀀스")
    
    X = pd.DataFrame(features_list)
    y = np.array(labels)
    
    seq_info = {
        'seq_max': seq_totalcnt_max_list,
        'seq_min': seq_totalcnt_min_list,
        'indices': indices
    }
    
    
    print(f"Feature 개수: {X.shape[1]}개")
    print(f"  - 기본 통계 (4컬럼 × 8): 32개")
    print(f"  - 비율: 8개")
    print(f"  - M14AM14B 임계값: 8개")
    print(f"  - M14AM10A 임계값: 4개")
    print(f"  - TOTALCNT 임계값: 8개")
    print(f"  - 황금 패턴: 4개")
    print(f"  - 변화율/가속도: 8개")
    print(f"  - 시간대별 통계: 8개")
    print(f"  - 🔥 조기 경보: 5개")
    print(f"데이터 범위: 인덱스 {indices[0]} ~ {indices[-1]}")
    
    return X, y, seq_info

def train_and_evaluate_complete():
    """
    280분 시퀀스 + 강화된 Feature + 조기 경보로 25분 후 TOTALCNT 예측
    """
    print("="*80)
    print("XGBoost 280분 → 25분 후 TOTALCNT 최대값 예측 (조기 경보 추가)")
    print("컬럼: M14AM14B, M14AM10A, M14AM16, TOTALCNT")
    print("Feature: 기본 80개 + 조기 경보 5개 = 총 85개")
    print("="*80)
    
    # GPU 확인
    tree_method, gpu_id = check_gpu_availability()
    
    # ===== 1. 데이터 로딩 =====
    print("\n[STEP 1] 데이터 로딩")
    print("-"*40)
    
    csv_file = '/mnt/project/V6_6결과.CSV'
    print(f"사용 파일: {csv_file}")
    
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV 파일이 없습니다: {csv_file}")
    
    df = pd.read_csv(csv_file, on_bad_lines='skip')
    print(f"데이터 로딩: {len(df):,}행")
    print(f"컬럼: {list(df.columns)}")
    
    # ===== 2. Feature 생성 =====
    print("\n[STEP 2] Feature 생성 (280분 시퀀스 + 강화 Feature + 조기 경보)")
    print("-"*40)
    
    X, y, seq_info = create_features_280min_enhanced(df)
    
    print(f"\nTOTALCNT 분석 (25분 후 최대값):")
    print(f"  평균: {y.mean():.2f}")
    print(f"  표준편차: {y.std():.2f}")
    print(f"  최소: {y.min():.2f}")
    print(f"  최대: {y.max():.2f}")
    print(f"  위험(1700+): {np.sum(y >= 1700)}개 ({np.sum(y >= 1700)/len(y)*100:.2f}%)")
    print(f"  극단(2000+): {np.sum(y >= 2000)}개 ({np.sum(y >= 2000)/len(y)*100:.2f}%)")
    
    # 조기 경보 통계
    early_warning_count = X['early_warning_1650_rising'].sum()
    print(f"\n조기 경보 발생: {early_warning_count}개 ({early_warning_count/len(X)*100:.2f}%)")
    
    # ===== 2000 이상 극단값 제거 =====
    print("\n극단값(2000+) 필터링...")
    filter_mask = y < 2000
    X_filtered = X[filter_mask]
    y_filtered = y[filter_mask]
    
    removed_count = len(y) - len(y_filtered)
    print(f"제거된 데이터: {removed_count:,}개 ({removed_count/len(y)*100:.3f}%)")
    print(f"남은 데이터: {len(y_filtered):,}개")
    print(f"새로운 최대값: {y_filtered.max():.2f}")
    
    # ===== 3. Train/Test 분할 =====
    print("\n[STEP 3] Train/Test 분할")
    print("-"*40)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_filtered, y_filtered, test_size=0.2, random_state=42, shuffle=True
    )
    
    print(f"학습 데이터: {X_train.shape[0]:,}개")
    print(f"테스트 데이터: {X_test.shape[0]:,}개")
    
    # ===== 4. 모델 학습 =====
    print("\n[STEP 4] XGBoost 모델 학습")
    print("-"*40)
    
    if gpu_id is not None:
        print(f"GPU 모드: tree_method={tree_method}, gpu_id={gpu_id}")
        model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method=tree_method,
            gpu_id=gpu_id,
            random_state=42
        )
    else:
        print(f"CPU 모드: tree_method={tree_method}")
        model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method=tree_method,
            random_state=42,
            n_jobs=4
        )
    
    print("학습 시작...")
    start_time = datetime.now()
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n학습 완료! 소요 시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)")
    
    # ===== 5. 모델 평가 =====
    print("\n[STEP 5] 모델 평가")
    print("-"*40)
    
    # Train 성능
    y_train_pred = model.predict(X_train)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    
    print(f"Train 성능:")
    print(f"  MAE:  {train_mae:.2f}")
    print(f"  RMSE: {train_rmse:.2f}")
    print(f"  R²:   {train_r2:.4f}")
    
    # Test 성능
    y_test_pred = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"\nTest 성능:")
    print(f"  MAE:  {test_mae:.2f}")
    print(f"  RMSE: {test_rmse:.2f}")
    print(f"  R²:   {test_r2:.4f}")
    
    # ===== 6. 모델 저장 =====
    print("\n[STEP 6] 모델 저장")
    print("-"*40)
    
    with open('xgboost_280to25_enhanced_earlywarning.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("모델 저장: xgboost_280to25_enhanced_earlywarning.pkl")
    
    # ===== 7. 상세 분석 =====
    print("\n[STEP 7] 상세 분석")
    print("-"*40)
    
    # 위험 구간 분석
    danger_mask_actual = y_test >= 1700
    danger_mask_pred = y_test_pred >= 1650
    
    danger_actual_count = np.sum(danger_mask_actual)
    danger_detected = np.sum(danger_mask_actual & danger_mask_pred)
    
    print(f"위험 구간 (1700+) 분석:")
    print(f"  실제 위험: {danger_actual_count}개")
    print(f"  감지 성공: {danger_detected}개")
    if danger_actual_count > 0:
        print(f"  감지율: {danger_detected/danger_actual_count*100:.1f}%")
    
    # 오차 분석
    errors = y_test_pred - y_test
    print(f"\n오차 분석:")
    print(f"  평균 오차: {np.mean(errors):.2f}")
    print(f"  오차 표준편차: {np.std(errors):.2f}")
    print(f"  오차 범위: [{np.min(errors):.2f}, {np.max(errors):.2f}]")
    
    # ===== 8. Feature 중요도 =====
    print("\n[STEP 8] Feature 중요도 (Top 30)")
    print("-"*40)
    
    feature_importance_full = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance_full.head(30).to_string(index=False))
    
    feature_importance_full.to_csv('feature_importance_280to25_earlywarning.csv', index=False)
    print("\nFeature 중요도 저장: feature_importance_280to25_earlywarning.csv")
    
    # 조기 경보 Feature 중요도 확인
    early_warning_importance = feature_importance_full[
        feature_importance_full['feature'].str.contains('early_warning|last_10min')
    ]
    
    if len(early_warning_importance) > 0:
        print("\n🔥 조기 경보 Feature 중요도:")
        print(early_warning_importance.to_string(index=False))
    
    # ===== 9. 최종 요약 =====
    print("\n" + "="*80)
    print("최종 평가 요약")
    print("="*80)
    print(f"1. 모델 성능:")
    print(f"   - Train MAE: {train_mae:.2f}")
    print(f"   - Test MAE:  {test_mae:.2f}")
    print(f"   - Test R²:   {test_r2:.4f}")
    
    print(f"\n2. 시퀀스 정보:")
    print(f"   - 시퀀스 길이: 280분")
    print(f"   - 예측 범위: 25분 후 최대값")
    print(f"   - Feature 수: {X.shape[1]}개 (조기 경보 5개 포함)")
    
    print(f"\n3. 저장 파일:")
    print(f"   - 모델: xgboost_280to25_enhanced_earlywarning.pkl")
    print(f"   - Feature 중요도: feature_importance_280to25_earlywarning.csv")
    
    print("\n" + "="*80)
    
    return model, feature_importance_full

# 실행
if __name__ == '__main__':
    model, feature_importance = train_and_evaluate_complete()