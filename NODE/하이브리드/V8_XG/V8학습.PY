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
        print("💻 CPU 모드로 학습")
        return 'hist', None

def create_features_100min(df, start_idx=100):
    """
    100분 시퀀스로 10분 후 TOTALCNT 예측 Feature 생성
    각 컬럼별 8개 통계값만 사용 (성공 코드 패턴)
    
    컬럼: M14AM14B, M14AM10A, M14AM16, TOTALCNT
    """
    
    print(f"\n📊 Feature 생성 중... (100분 시퀀스 → 10분 후 예측)")
    
    # 필수 컬럼 확인
    required_cols = ['M14AM14B', 'M14AM10A', 'M14AM16', 'TOTALCNT']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"❌ 필수 컬럼 누락: {missing_cols}")
    
    print(f"✅ 필수 컬럼 확인: {required_cols}")
    
    features_list = []
    labels = []
    
    seq_totalcnt_max_list = []
    seq_totalcnt_min_list = []
    indices = []
    
    total_sequences = len(df) - start_idx - 10
    print(f"✅ 생성 가능한 시퀀스: {total_sequences:,}개")
    
    # Feature 생성
    for i in range(start_idx, len(df) - 10):
        if i % 10000 == 0:
            progress = (i - start_idx) / total_sequences * 100
            print(f"  진행률: {i-start_idx}/{total_sequences} ({progress:.1f}%)", end='\r')
        
        # 100분 시퀀스 추출
        seq_m14b = df['M14AM14B'].iloc[i-100:i].values
        seq_m10a = df['M14AM10A'].iloc[i-100:i].values
        seq_m16 = df['M14AM16'].iloc[i-100:i].values
        seq_totalcnt = df['TOTALCNT'].iloc[i-100:i].values
        
        features = {}
        
        # ========== M14AM14B (8개 통계값) ==========
        features['m14b_mean'] = np.mean(seq_m14b)                    # 100분 평균
        features['m14b_std'] = np.std(seq_m14b)                      # 100분 표준편차
        features['m14b_last_5_mean'] = np.mean(seq_m14b[-5:])        # 최근 5분 평균
        features['m14b_max'] = np.max(seq_m14b)                      # 100분 최대
        features['m14b_min'] = np.min(seq_m14b)                      # 100분 최소
        features['m14b_slope'] = np.polyfit(np.arange(100), seq_m14b, 1)[0]  # 추세
        features['m14b_last_10_mean'] = np.mean(seq_m14b[-10:])      # 최근 10분 평균
        features['m14b_first_10_mean'] = np.mean(seq_m14b[:10])      # 처음 10분 평균
        
        # ========== M14AM10A (8개 통계값) ==========
        features['m10a_mean'] = np.mean(seq_m10a)
        features['m10a_std'] = np.std(seq_m10a)
        features['m10a_last_5_mean'] = np.mean(seq_m10a[-5:])
        features['m10a_max'] = np.max(seq_m10a)
        features['m10a_min'] = np.min(seq_m10a)
        features['m10a_slope'] = np.polyfit(np.arange(100), seq_m10a, 1)[0]
        features['m10a_last_10_mean'] = np.mean(seq_m10a[-10:])
        features['m10a_first_10_mean'] = np.mean(seq_m10a[:10])
        
        # ========== M14AM16 (8개 통계값) ==========
        features['m16_mean'] = np.mean(seq_m16)
        features['m16_std'] = np.std(seq_m16)
        features['m16_last_5_mean'] = np.mean(seq_m16[-5:])
        features['m16_max'] = np.max(seq_m16)
        features['m16_min'] = np.min(seq_m16)
        features['m16_slope'] = np.polyfit(np.arange(100), seq_m16, 1)[0]
        features['m16_last_10_mean'] = np.mean(seq_m16[-10:])
        features['m16_first_10_mean'] = np.mean(seq_m16[:10])
        
        # ========== TOTALCNT (8개 통계값) ==========
        features['totalcnt_mean'] = np.mean(seq_totalcnt)
        features['totalcnt_std'] = np.std(seq_totalcnt)
        features['totalcnt_last_5_mean'] = np.mean(seq_totalcnt[-5:])
        features['totalcnt_max'] = np.max(seq_totalcnt)
        features['totalcnt_min'] = np.min(seq_totalcnt)
        features['totalcnt_slope'] = np.polyfit(np.arange(100), seq_totalcnt, 1)[0]
        features['totalcnt_last_10_mean'] = np.mean(seq_totalcnt[-10:])
        features['totalcnt_first_10_mean'] = np.mean(seq_totalcnt[:10])
        
        features_list.append(features)
        
        # 라벨: 10분 후 TOTALCNT 최대값
        future_totalcnt = df['TOTALCNT'].iloc[i:i+10].values
        labels.append(np.max(future_totalcnt))
        
        # 시퀀스 정보 저장
        seq_totalcnt_max_list.append(np.max(seq_totalcnt))
        seq_totalcnt_min_list.append(np.min(seq_totalcnt))
        indices.append(i)
    
    print(f"\n✅ Feature 생성 완료: {len(features_list):,}개 시퀀스")
    
    X = pd.DataFrame(features_list)
    y = np.array(labels)
    
    seq_info = {
        'seq_max': seq_totalcnt_max_list,
        'seq_min': seq_totalcnt_min_list,
        'indices': indices
    }
    
    print(f"✅ Feature 개수: {X.shape[1]}개 (4컬럼 × 8개 통계)")
    print(f"✅ 데이터 범위: 인덱스 {indices[0]} ~ {indices[-1]}")
    
    return X, y, seq_info

def train_and_evaluate_complete():
    """
    100분 시퀀스로 10분 후 TOTALCNT 예측
    """
    print("="*80)
    print("XGBoost 100분 → 10분 후 TOTALCNT 예측")
    print("컬럼: M14AM14B, M14AM10A, M14AM16, TOTALCNT")
    print("Feature: 각 컬럼별 8개 통계값 (총 32개)")
    print("="*80)
    
    # GPU 확인
    tree_method, gpu_id = check_gpu_availability()
    
    # ===== 1. 데이터 로딩 =====
    print("\n[STEP 1] 데이터 로딩")
    print("-"*40)
    
    # CSV 파일 경로
    csv_file = 'DATA/202502.CSV'
    print(f"사용 파일: {csv_file}")
    
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"❌ CSV 파일이 없습니다: {csv_file}")
    
    df = pd.read_csv(csv_file, on_bad_lines='skip')
    print(f"✅ 데이터 로딩: {len(df):,}행")
    print(f"✅ 컬럼: {list(df.columns)}")
    
    # ===== 2. Feature 생성 =====
    print("\n[STEP 2] Feature 생성 (100분 시퀀스)")
    print("-"*40)
    
    X, y, seq_info = create_features_100min(df)
    
    print(f"\n📊 TOTALCNT 분석 (10분 후 최대값):")
    print(f"  평균: {y.mean():.2f}")
    print(f"  표준편차: {y.std():.2f}")
    print(f"  최소: {y.min():.2f}")
    print(f"  최대: {y.max():.2f}")
    print(f"  위험(1700+): {np.sum(y >= 1700)}개 ({np.sum(y >= 1700)/len(y)*100:.2f}%)")
    
    # ===== 3. Train/Test 분할 =====
    print("\n[STEP 3] Train/Test 분할")
    print("-"*40)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    print(f"✅ 학습 데이터: {X_train.shape[0]:,}개")
    print(f"✅ 테스트 데이터: {X_test.shape[0]:,}개")
    
    # ===== 4. 모델 학습 =====
    print("\n[STEP 4] XGBoost 모델 학습")
    print("-"*40)
    
    if gpu_id is not None:
        print(f"🎮 GPU 모드: tree_method={tree_method}, gpu_id={gpu_id}")
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
        print(f"💻 CPU 모드: tree_method={tree_method}")
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
    
    print("🚀 학습 시작...")
    start_time = datetime.now()
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n✅ 학습 완료! 소요 시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)")
    
    # ===== 5. 모델 평가 =====
    print("\n[STEP 5] 모델 평가")
    print("-"*40)
    
    # Train 성능
    y_train_pred = model.predict(X_train)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    
    print(f"📊 Train 성능:")
    print(f"  MAE:  {train_mae:.2f}")
    print(f"  RMSE: {train_rmse:.2f}")
    print(f"  R²:   {train_r2:.4f}")
    
    # Test 성능
    y_test_pred = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"\n📊 Test 성능:")
    print(f"  MAE:  {test_mae:.2f}")
    print(f"  RMSE: {test_rmse:.2f}")
    print(f"  R²:   {test_r2:.4f}")
    
    print(f"\n📊 Train vs Test 차이:")
    print(f"  MAE 차이:  {abs(test_mae - train_mae):.2f}")
    print(f"  과적합 여부: {'⚠️ 과적합' if abs(test_mae - train_mae) > 50 else '✅ 정상'}")
    
    # ===== 6. 모델 저장 =====
    print("\n[STEP 6] 모델 저장")
    print("-"*40)
    
    with open('xgboost_100to10_totalcnt.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✅ 모델 저장: xgboost_100to10_totalcnt.pkl")
    
    # ===== 7. 상세 분석 =====
    print("\n[STEP 7] 상세 분석")
    print("-"*40)
    
    # 위험 구간 분석
    danger_mask_actual = y_test >= 1700
    danger_mask_pred = y_test_pred >= 1650  # 예측 임계값
    
    danger_actual_count = np.sum(danger_mask_actual)
    danger_detected = np.sum(danger_mask_actual & danger_mask_pred)
    
    print(f"⚠️ 위험 구간 (1700+) 분석:")
    print(f"  실제 위험: {danger_actual_count}개")
    print(f"  감지 성공: {danger_detected}개")
    if danger_actual_count > 0:
        print(f"  감지율: {danger_detected/danger_actual_count*100:.1f}%")
    
    # 오차 분석
    errors = y_test_pred - y_test
    print(f"\n📊 오차 분석:")
    print(f"  평균 오차: {np.mean(errors):.2f}")
    print(f"  오차 표준편차: {np.std(errors):.2f}")
    print(f"  오차 범위: [{np.min(errors):.2f}, {np.max(errors):.2f}]")
    
    # ===== 8. 시각화 =====
    print("\n[STEP 8] 시각화 생성")
    print("-"*40)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. 예측 vs 실제
    ax1 = axes[0, 0]
    ax1.scatter(y_test, y_test_pred, alpha=0.3, s=5, color='blue')
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax1.axhline(y=1700, color='orange', linestyle='--', alpha=0.5, label='Danger(1700)')
    ax1.axvline(x=1700, color='orange', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Actual TOTALCNT')
    ax1.set_ylabel('Predicted TOTALCNT')
    ax1.set_title(f'Actual vs Predicted\nMAE={test_mae:.2f}, R²={test_r2:.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 시계열 (처음 500개)
    ax2 = axes[0, 1]
    plot_size = min(500, len(y_test))
    ax2.plot(range(plot_size), y_test[:plot_size], 'b-', label='Actual', alpha=0.7, linewidth=1)
    ax2.plot(range(plot_size), y_test_pred[:plot_size], 'r--', label='Predicted', alpha=0.7, linewidth=1)
    ax2.axhline(y=1700, color='orange', linestyle='--', label='Danger(1700)', alpha=0.5)
    ax2.set_xlabel('Time Index')
    ax2.set_ylabel('TOTALCNT')
    ax2.set_title('Time Series (First 500)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 오차 분포
    ax3 = axes[0, 2]
    ax3.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    ax3.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    ax3.set_xlabel('Prediction Error')
    ax3.set_ylabel('Frequency')
    ax3.set_title(f'Error Distribution\nMean={np.mean(errors):.2f}, Std={np.std(errors):.2f}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 위험 구간 성능
    ax4 = axes[1, 0]
    normal_mask = y_test < 1700
    ax4.scatter(y_test[normal_mask], y_test_pred[normal_mask], 
               alpha=0.2, s=5, label='Normal(<1700)', color='blue')
    ax4.scatter(y_test[danger_mask_actual], y_test_pred[danger_mask_actual], 
               alpha=0.8, s=20, label='Danger(1700+)', color='red')
    ax4.plot([1200, 2000], [1200, 2000], 'k--', lw=1)
    ax4.axhline(y=1700, color='orange', linestyle='--', alpha=0.5)
    ax4.axvline(x=1700, color='orange', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Actual')
    ax4.set_ylabel('Predicted')
    ax4.set_title(f'Danger Zone Performance\nDetected: {danger_detected}/{danger_actual_count}')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. MAE by Value Range
    ax5 = axes[1, 1]
    bins = np.array([1000, 1200, 1400, 1600, 1800, 2000])
    bin_labels = ['1000-1200', '1200-1400', '1400-1600', '1600-1800', '1800+']
    mae_by_range = []
    count_by_range = []
    
    for i in range(len(bins)-1):
        mask = (y_test >= bins[i]) & (y_test < bins[i+1])
        count = np.sum(mask)
        count_by_range.append(count)
        if count > 0:
            mae_by_range.append(mean_absolute_error(y_test[mask], y_test_pred[mask]))
        else:
            mae_by_range.append(0)
    
    bars = ax5.bar(bin_labels, mae_by_range, color='steelblue', alpha=0.7)
    ax5.set_xlabel('TOTALCNT Range')
    ax5.set_ylabel('MAE')
    ax5.set_title('MAE by Value Range')
    ax5.tick_params(axis='x', rotation=45)
    
    # 개수 표시
    for i, (bar, count) in enumerate(zip(bars, count_by_range)):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'n={count}',
                ha='center', va='bottom', fontsize=8)
    
    ax5.grid(True, alpha=0.3)
    
    # 6. Feature 중요도 (Top 20)
    ax6 = axes[1, 2]
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(20)
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_importance)))
    ax6.barh(range(len(feature_importance)), feature_importance['importance'].values, color=colors)
    ax6.set_yticks(range(len(feature_importance)))
    ax6.set_yticklabels(feature_importance['feature'].values, fontsize=8)
    ax6.set_xlabel('Importance')
    ax6.set_title('Top 20 Feature Importance')
    ax6.invert_yaxis()
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('100min -> 10min TOTALCNT Prediction (XGBoost)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('xgboost_100to10_evaluation.png', dpi=150, bbox_inches='tight')
    print("✅ 그래프 저장: xgboost_100to10_evaluation.png")
    
    # ===== 9. Feature 중요도 전체 출력 =====
    print("\n[STEP 9] Feature 중요도 (전체)")
    print("-"*40)
    
    feature_importance_full = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance_full.to_string(index=False))
    
    # CSV 저장
    feature_importance_full.to_csv('feature_importance.csv', index=False)
    print("\n✅ Feature 중요도 저장: feature_importance.csv")
    
    # ===== 10. 최종 요약 =====
    print("\n" + "="*80)
    print("최종 평가 요약")
    print("="*80)
    print(f"1. 모델 성능:")
    print(f"   - Train MAE: {train_mae:.2f}")
    print(f"   - Test MAE:  {test_mae:.2f}")
    print(f"   - Test R²:   {test_r2:.4f}")
    
    print(f"\n2. 위험 구간 성능:")
    print(f"   - 실제 위험(1700+): {danger_actual_count}개")
    print(f"   - 감지 성공: {danger_detected}개 ({danger_detected/danger_actual_count*100 if danger_actual_count > 0 else 0:.1f}%)")
    
    print(f"\n3. 시퀀스 정보:")
    print(f"   - 시퀀스 길이: 100분")
    print(f"   - 예측 범위: 10분 후")
    print(f"   - Feature 수: {X.shape[1]}개 (4컬럼 × 8통계)")
    
    print(f"\n4. 저장 파일:")
    print(f"   - 모델: xgboost_100to10_totalcnt.pkl")
    print(f"   - 그래프: xgboost_100to10_evaluation.png")
    print(f"   - Feature 중요도: feature_importance.csv")
    
    print("\n" + "="*80)
    
    return model, feature_importance_full

# 실행
if __name__ == '__main__':
    model, feature_importance = train_and_evaluate_complete()