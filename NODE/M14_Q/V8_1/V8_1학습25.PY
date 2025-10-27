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
        
        print("✅ GPU 사용 가능! XGBoost GPU 모드로 학습")
        return 'gpu_hist', 0
        
    except Exception as e:
        print(f"⚠️ GPU 사용 불가: {str(e)[:100]}")
        print("CPU 모드로 학습")
        return 'hist', None

def create_features_280min_to_25min(df, start_idx=280):
    """
    280분 시퀀스 → 25분 후 예측
    - 기본 8개 통계값 (각 컬럼)
    - 비율 Feature (m14b/m10a 등)
    - 임계값 카운트
    - 변화율, 가속도 등
    """
    
    print(f"\n⚙️ Feature 생성 중... (280분 시퀀스 → 25분 후 예측)")
    
    # 필수 컬럼 확인
    required_cols = ['M14AM14B', 'M14AM10A', 'M14AM16', 'TOTALCNT']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"필수 컬럼 누락: {missing_cols}")
    
    print(f"✅ 필수 컬럼 확인: {required_cols}")
    
    features_list = []
    labels = []
    
    seq_totalcnt_max_list = []
    seq_totalcnt_min_list = []
    indices = []
    
    # ===== 25분 후 예측으로 변경 =====
    total_sequences = len(df) - start_idx - 25
    print(f"✅ 생성 가능한 시퀀스: {total_sequences:,}개")
    
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
        
        # ========== 시간 통계 (10개) ==========
        for window in [10, 30, 60]:
            features[f'm14b_last_{window}_mean'] = np.mean(seq_m14b[-window:])
            features[f'totalcnt_last_{window}_mean'] = np.mean(seq_totalcnt[-window:])
        
        features['m14b_trend_10_vs_30'] = np.mean(seq_m14b[-10:]) - np.mean(seq_m14b[-30:])
        features['m14b_trend_30_vs_60'] = np.mean(seq_m14b[-30:]) - np.mean(seq_m14b[-60:])
        features['totalcnt_trend_10_vs_30'] = np.mean(seq_totalcnt[-10:]) - np.mean(seq_totalcnt[-30:])
        features['totalcnt_trend_30_vs_60'] = np.mean(seq_totalcnt[-30:]) - np.mean(seq_totalcnt[-60:])
        
        # ========== 추가 통계 (10개) ==========
        features['m14b_median'] = np.median(seq_m14b)
        features['m14b_q25'] = np.percentile(seq_m14b, 25)
        features['m14b_q75'] = np.percentile(seq_m14b, 75)
        features['totalcnt_median'] = np.median(seq_totalcnt)
        features['totalcnt_q25'] = np.percentile(seq_totalcnt, 25)
        features['totalcnt_q75'] = np.percentile(seq_totalcnt, 75)
        
        features['m14b_over_mean'] = np.sum(seq_m14b > np.mean(seq_m14b))
        features['totalcnt_over_mean'] = np.sum(seq_totalcnt > np.mean(seq_totalcnt))
        features['m14b_last_vs_max'] = seq_m14b[-1] / (np.max(seq_m14b) + 1)
        features['totalcnt_last_vs_max'] = seq_totalcnt[-1] / (np.max(seq_totalcnt) + 1)
        
        features_list.append(features)
        
        # ===== 25분 후 타겟 =====
        labels.append(df['TOTALCNT'].iloc[i+25])
        
        seq_totalcnt_max_list.append(np.max(seq_totalcnt))
        seq_totalcnt_min_list.append(np.min(seq_totalcnt))
        indices.append(i)
    
    print()  # 줄바꿈
    
    X = pd.DataFrame(features_list)
    y = np.array(labels)
    
    print(f"✅ Feature 생성 완료")
    print(f"  - Feature 수: {X.shape[1]}개")
    print(f"  - 샘플 수: {X.shape[0]:,}개")
    print(f"  - 타겟: 25분 후 TOTALCNT")
    
    return X, y, seq_totalcnt_max_list, seq_totalcnt_min_list, indices

def train_and_evaluate_complete():
    """전체 학습 및 평가"""
    
    print("\n" + "="*80)
    print("🚀 280분 시퀀스 → 25분 후 TOTALCNT 예측")
    print("="*80)
    
    # ===== 1. GPU 확인 =====
    tree_method, gpu_id = check_gpu_availability()
    
    # ===== 2. 데이터 로드 =====
    print("\n[STEP 1] 데이터 로드")
    print("-"*40)
    
    data_path = '/mnt/project/V6_6결과.CSV'
    
    if not os.path.exists(data_path):
        print(f"❌ 데이터 파일 없음: {data_path}")
        return None, None
    
    df = pd.read_csv(data_path)
    print(f"✅ 데이터 로드: {df.shape[0]:,}행 × {df.shape[1]}열")
    
    # CURRTIME 정렬
    if 'CURRTIME' in df.columns:
        df = df.sort_values('CURRTIME').reset_index(drop=True)
        print(f"✅ 시간 정렬 완료")
    
    # ===== 3. Feature 생성 =====
    print("\n[STEP 2] Feature 생성")
    print("-"*40)
    
    X, y, seq_max, seq_min, indices = create_features_280min_to_25min(df)
    
    print(f"\n📊 타겟 통계:")
    print(f"  평균: {y.mean():.2f}")
    print(f"  표준편차: {y.std():.2f}")
    print(f"  최소: {y.min():.0f}")
    print(f"  최대: {y.max():.0f}")
    print(f"  위험(1700+): {np.sum(y >= 1700):,}개 ({np.sum(y >= 1700)/len(y)*100:.1f}%)")
    
    # ===== 4. Train/Test 분할 =====
    print("\n[STEP 3] Train/Test 분할")
    print("-"*40)
    
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"✅ Train: {X_train.shape[0]:,}개")
    print(f"✅ Test:  {X_test.shape[0]:,}개")
    
    # ===== 5. 모델 학습 =====
    print("\n[STEP 4] XGBoost 학습 (280분→25분)")
    print("-"*40)
    
    params = {
        'n_estimators': 500,
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'tree_method': tree_method,
        'random_state': 42,
        'n_jobs': -1
    }
    
    if gpu_id is not None:
        params['gpu_id'] = gpu_id
    
    model = xgb.XGBRegressor(**params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=50
    )
    
    print("\n✅ 학습 완료")
    
    # ===== 6. 평가 =====
    print("\n[STEP 5] 성능 평가")
    print("-"*40)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    
    print(f"Train 성능:")
    print(f"  MAE:  {train_mae:.2f}")
    print(f"  RMSE: {train_rmse:.2f}")
    print(f"  R²:   {train_r2:.4f}")
    
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"\nTest 성능:")
    print(f"  MAE:  {test_mae:.2f}")
    print(f"  RMSE: {test_rmse:.2f}")
    print(f"  R²:   {test_r2:.4f}")
    
    # ===== 7. 모델 저장 =====
    print("\n[STEP 6] 모델 저장")
    print("-"*40)
    
    with open('xgboost_280to25_enhanced.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✅ 모델 저장: xgboost_280to25_enhanced.pkl")
    
    # ===== 8. 상세 분석 =====
    print("\n[STEP 7] 상세 분석")
    print("-"*40)
    
    danger_mask_actual = y_test >= 1700
    danger_mask_pred = y_test_pred >= 1650
    
    danger_actual_count = np.sum(danger_mask_actual)
    danger_detected = np.sum(danger_mask_actual & danger_mask_pred)
    
    print(f"위험 구간 (1700+) 분석:")
    print(f"  실제 위험: {danger_actual_count}개")
    print(f"  감지 성공: {danger_detected}개")
    if danger_actual_count > 0:
        print(f"  감지율: {danger_detected/danger_actual_count*100:.1f}%")
    
    errors = y_test_pred - y_test
    print(f"\n오차 분석:")
    print(f"  평균 오차: {np.mean(errors):.2f}")
    print(f"  오차 표준편차: {np.std(errors):.2f}")
    print(f"  오차 범위: [{np.min(errors):.2f}, {np.max(errors):.2f}]")
    
    # ===== 9. Feature 중요도 =====
    print("\n[STEP 8] Feature 중요도 (Top 30)")
    print("-"*40)
    
    feature_importance_full = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance_full.head(30).to_string(index=False))
    
    feature_importance_full.to_csv('feature_importance_280to25.csv', index=False)
    print("\n✅ Feature 중요도 저장: feature_importance_280to25.csv")
    
    # ===== 10. 최종 요약 =====
    print("\n" + "="*80)
    print("📊 최종 평가 요약 (280분 → 25분 후 예측)")
    print("="*80)
    print(f"1. 모델 성능:")
    print(f"   - Train MAE: {train_mae:.2f}")
    print(f"   - Test MAE:  {test_mae:.2f}")
    print(f"   - Test R²:   {test_r2:.4f}")
    
    print(f"\n2. 시퀀스 정보:")
    print(f"   - 시퀀스 길이: 280분")
    print(f"   - 예측 범위: 25분 후")
    print(f"   - Feature 수: {X.shape[1]}개")
    
    print(f"\n3. 위험 감지:")
    print(f"   - 감지율: {danger_detected/danger_actual_count*100 if danger_actual_count > 0 else 0:.1f}%")
    print(f"   - 실제 위험: {danger_actual_count}개")
    
    print(f"\n4. 저장 파일:")
    print(f"   - 모델: xgboost_280to25_enhanced.pkl")
    print(f"   - Feature 중요도: feature_importance_280to25.csv")
    
    print("\n" + "="*80)
    
    return model, feature_importance_full

# 실행
if __name__ == '__main__':
    model, feature_importance = train_and_evaluate_complete()