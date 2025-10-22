import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def train_and_evaluate_complete():
    """
    학습(aas.csv) → 평가(BBB.CSV) 전체 프로세스
    V6 Ultimate 12개 필수 컬럼 활용 + 타겟/각 컬럼별 이상감지 Feature 추가
    """
    print("="*80)
    print("XGBoost 30분→10분 예측 모델 학습 및 평가")
    print("핵심 12개 컬럼 + 타겟 이상감지 + 각 컬럼별 이상감지 Feature")
    print("="*80)
    
    # 핵심 12개 컬럼만 사용 (물리적으로 중요한 것만)
    FEATURE_COLS = {
        'storage': ['M16A_3F_STORAGE_UTIL'],  # 극단값 핵심 지표
        'cmd': ['M16A_3F_CMD', 'M16A_6F_TO_HUB_CMD'],  # 주요 CMD
        'inflow': ['M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2', 'M14A_3F_TO_HUB_JOB2'],  # 주요 유입
        'outflow': ['M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB', 'M16A_3F_TO_M14A_3F_JOB'],  # 주요 유출
        'maxcapa': ['M16A_6F_LFT_MAXCAPA', 'M16A_2F_LFT_MAXCAPA']  # 용량 제한
    }
    
    TARGET_COL = 'CURRENT_M16A_3F_JOB_2'
    
    # Feature 생성 함수
    def create_features(df, start_idx=30):
        features_list = []
        labels = []
        seq_max_list = []
        seq_min_list = []
        indices = []
        
        for i in range(start_idx, len(df) - 10):
            seq_target = df[TARGET_COL].iloc[i-30:i].values
            
            # 🔥 타겟 컬럼 기본 통계 (재사용을 위해 변수로 저장)
            seq_mean = np.mean(seq_target)
            seq_std = np.std(seq_target)
            seq_max = np.max(seq_target)
            seq_min = np.min(seq_target)
            last_5_mean = np.mean(seq_target[-5:])
            last_10_mean = np.mean(seq_target[-10:])
            first_10_mean = np.mean(seq_target[:10])
            slope = np.polyfit(np.arange(30), seq_target, 1)[0]
            
            features = {
                # 타겟 컬럼 기본 특성
                'target_mean': seq_mean,
                'target_std': seq_std,
                'target_last_5_mean': last_5_mean,
                'target_max': seq_max,
                'target_min': seq_min,
                'target_slope': slope,
                'target_last_10_mean': last_10_mean,
                'target_first_10_mean': first_10_mean,
            }
            
            # 🔥🔥🔥 타겟 컬럼 이상감지 Feature 추가! (300미만 → 300이상 급증 감지)
            features['target_near_threshold'] = 1 if 290 <= seq_max < 300 else 0  # 임계값 근접
            features['target_rising_trend'] = 1 if slope > 0.5 else 0  # 상승 추세
            features['target_recent_high'] = 1 if last_5_mean > seq_mean + 10 else 0  # 최근 고점
            features['target_risk_score'] = max(0, (seq_max - 290) / 10)  # 위험 점수 (0~1+)
            features['target_near_300'] = 1 if seq_max >= 295 else 0  # 300 초근접
            features['target_last_5_rising'] = 1 if last_5_mean > last_10_mean else 0  # 최근 5분 상승
            features['target_acceleration'] = np.mean(np.diff(seq_target[-5:])) if len(seq_target) >= 6 else 0  # 가속도
            
            # 각 컬럼 그룹별 특성 추가
            for group_name, cols in FEATURE_COLS.items():
                for col in cols:
                    if col in df.columns:
                        seq_data = df[col].iloc[i-30:i].values
                        
                        # 🔥 컬럼별 기본 통계 (재사용)
                        col_mean = np.mean(seq_data)
                        col_std = np.std(seq_data)
                        col_max = np.max(seq_data)
                        col_min = np.min(seq_data)
                        col_last_5_mean = np.mean(seq_data[-5:])
                        col_last_10_mean = np.mean(seq_data[-10:])
                        col_first_10_mean = np.mean(seq_data[:10])
                        col_slope = np.polyfit(np.arange(30), seq_data, 1)[0]
                        
                        # 기본 통계
                        features[f'{col}_mean'] = col_mean
                        features[f'{col}_std'] = col_std
                        features[f'{col}_max'] = col_max
                        features[f'{col}_min'] = col_min
                        features[f'{col}_last_5_mean'] = col_last_5_mean
                        features[f'{col}_last_10_mean'] = col_last_10_mean
                        features[f'{col}_slope'] = col_slope
                        features[f'{col}_first_10_mean'] = col_first_10_mean
                        features[f'{col}_mid_10_mean'] = np.mean(seq_data[10:20])
                        features[f'{col}_last_value'] = seq_data[-1]
                        
                        # 🔥🔥🔥 각 컬럼별 이상감지 Feature 추가!
                        # 컬럼별 임계값 설정 (도메인 지식 기반)
                        if 'CMD' in col:
                            # CMD는 낮을수록 위험 (180 이하 고위험)
                            col_threshold_low = 180
                            col_threshold_high = 280
                            features[f'{col}_near_low_threshold'] = 1 if col_min <= col_threshold_low else 0
                            features[f'{col}_near_high_threshold'] = 1 if col_max >= col_threshold_high else 0
                            features[f'{col}_in_danger_zone'] = 1 if (col_last_5_mean < 220) else 0
                        elif 'STORAGE' in col:
                            # Storage Util 30~40 위험 구간
                            features[f'{col}_near_threshold'] = 1 if 30 <= col_max <= 40 else 0
                            features[f'{col}_in_danger_zone'] = 1 if (30 <= col_last_5_mean <= 40) else 0
                        else:
                            # 기타 컬럼은 일반적인 이상감지
                            features[f'{col}_near_max'] = 1 if col_max >= (col_mean + 2*col_std) else 0
                        
                        # 공통 이상감지 Feature
                        features[f'{col}_rising_trend'] = 1 if col_slope > 0.5 else 0
                        features[f'{col}_falling_trend'] = 1 if col_slope < -0.5 else 0
                        features[f'{col}_recent_high'] = 1 if col_last_5_mean > col_mean + (col_std * 0.5) else 0
                        features[f'{col}_recent_low'] = 1 if col_last_5_mean < col_mean - (col_std * 0.5) else 0
                        features[f'{col}_last_5_rising'] = 1 if col_last_5_mean > col_last_10_mean else 0
                        features[f'{col}_acceleration'] = np.mean(np.diff(seq_data[-5:])) if len(seq_data) >= 6 else 0
                        features[f'{col}_volatility'] = col_std / (col_mean + 1e-6)  # 변동성 (상대 표준편차)
            
            # 유입-유출 차이 (Net Flow)
            inflow_sum = 0
            outflow_sum = 0
            for col in FEATURE_COLS['inflow']:
                if col in df.columns:
                    inflow_sum += df[col].iloc[i-1]
            for col in FEATURE_COLS['outflow']:
                if col in df.columns:
                    outflow_sum += df[col].iloc[i-1]
            features['net_flow'] = inflow_sum - outflow_sum
            features['net_flow_positive'] = 1 if (inflow_sum - outflow_sum) > 0 else 0  # 🔥 순유입 여부
            
            # CMD 총합
            cmd_sum = 0
            for col in FEATURE_COLS['cmd']:
                if col in df.columns:
                    cmd_sum += df[col].iloc[i-1]
            features['total_cmd'] = cmd_sum
            features['total_cmd_low'] = 1 if cmd_sum < 400 else 0  # 🔥 CMD 총합 낮음
            
            features_list.append(features)
            labels.append(df[TARGET_COL].iloc[i:i+10].max())
            seq_max_list.append(seq_max)
            seq_min_list.append(seq_min)
            indices.append(i)
        
        return pd.DataFrame(features_list), np.array(labels), seq_max_list, seq_min_list, indices
    
    # ===== 1. 학습 단계 (aas.csv) =====
    print("\n[STEP 1] aas.csv로 모델 학습")
    print("-"*40)
    
    df_train = pd.read_csv('HUB_0509_TO_0929_DATA.csv', on_bad_lines='skip')
    
    print(f"학습 데이터: {len(df_train)}개 행")
    print(f"사용 가능한 컬럼 확인:")
    
    all_feature_cols = []
    for group_name, cols in FEATURE_COLS.items():
        available = [col for col in cols if col in df_train.columns]
        all_feature_cols.extend(available)
        print(f"  - {group_name}: {len(available)}/{len(cols)}개")
    
    # 학습 데이터 생성
    X_train, y_train, _, _, _ = create_features(df_train)
    
    print(f"\n생성된 Feature 수: {len(X_train.columns)}개")
    print(f"  → 기존 Feature + 타겟 이상감지 7개 + 각 컬럼별 이상감지 추가!")
    print(f"학습 샘플 수: {len(X_train)}개")
    
    # 🔥 이상감지 Feature 확인
    print(f"\n✅ 추가된 타겟 이상감지 Feature (7개):")
    target_anomaly_features = ['target_near_threshold', 'target_rising_trend', 'target_recent_high', 
                                'target_risk_score', 'target_near_300', 'target_last_5_rising', 'target_acceleration']
    for feat in target_anomaly_features:
        if feat in X_train.columns:
            print(f"  - {feat}: ✓")
    
    print(f"\n✅ 추가된 컬럼별 이상감지 Feature:")
    col_anomaly_count = 0
    for col in X_train.columns:
        if any(keyword in col for keyword in ['_near_', '_rising_', '_falling_', '_recent_', '_acceleration', '_volatility', '_danger_', '_low_threshold', '_high_threshold', '_in_danger']):
            if not col.startswith('target_'):  # 타겟 제외
                col_anomaly_count += 1
    print(f"  - 컬럼별 이상감지 Feature: {col_anomaly_count}개")
    
    # 학습/검증 분할
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # GPU/CPU 자동 선택
    print("\n🔍 학습 환경 감지 중...")
    use_gpu = False
    
    try:
        print("  → GPU 모드 시도 중...")
        test_model = xgb.XGBRegressor(
            n_estimators=5,
            max_depth=3,
            random_state=42,
            tree_method='gpu_hist',
            gpu_id=0
        )
        test_model.fit(X_tr[:100], y_tr[:100], verbose=False)
        use_gpu = True
        print("  ✅ GPU 사용 가능! GPU 모드로 학습합니다.\n")
    except Exception as e:
        print(f"  ⚠️ GPU 사용 불가: {str(e)[:100]}")
        print("  → CPU 모드로 전환합니다.\n")
        use_gpu = False
    
    # 실제 모델 생성
    if use_gpu:
        model = xgb.XGBRegressor(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method='gpu_hist',
            gpu_id=0,
            predictor='gpu_predictor'
        )
    else:
        model = xgb.XGBRegressor(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method='hist',
            n_jobs=-1
        )
    
    print("모델 학습 중...")
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # 학습 데이터 평가
    y_val_pred = model.predict(X_val)
    train_mae = mean_absolute_error(y_val, y_val_pred)
    train_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    train_r2 = r2_score(y_val, y_val_pred)
    
    print(f"\n학습 데이터 성능:")
    print(f"  MAE:  {train_mae:.4f}")
    print(f"  RMSE: {train_rmse:.4f}")
    print(f"  R²:   {train_r2:.4f}")
    
    # 모델 저장
    with open('xgboost_model_30min_10min_이상감지FULL.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✅ 모델 저장 완료: xgboost_model_30min_10min_이상감지FULL.pkl")
    
    # ===== 2. 평가 단계 (BBB.CSV) =====
    print("\n[STEP 2] BBB.CSV로 모델 평가")
    print("-"*40)
    
    df_test = pd.read_csv('HUB_20250916_to_20250929.CSV', on_bad_lines='skip')
    
    print(f"평가 데이터: {len(df_test)}개 행")
    
    # BBB.CSV Feature 생성
    X_test, y_test, seq_max_list, seq_min_list, indices = create_features(df_test)
    
    # 예측
    y_pred = model.predict(X_test)
    
    # 평가 지표
    test_mae = mean_absolute_error(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_r2 = r2_score(y_test, y_pred)
    
    print(f"\nBBB.CSV 평가 결과:")
    print(f"  MAE:  {test_mae:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  R²:   {test_r2:.4f}")
    
    # ===== 3. 상세 분석 결과 생성 =====
    print("\n[STEP 3] 상세 분석 결과 생성")
    print("-"*40)
    
    # STAT_DT 처리
    if 'STAT_DT' in df_test.columns:
        try:
            df_test['STAT_DT'] = pd.to_datetime(df_test['STAT_DT'], format='%Y%m%d%H%M')
        except:
            try:
                df_test['STAT_DT'] = pd.to_datetime(df_test['STAT_DT'])
            except:
                base_date = datetime(2024, 1, 1)
                df_test['STAT_DT'] = [base_date + timedelta(minutes=i) for i in range(len(df_test))]
    else:
        base_date = datetime(2024, 1, 1)
        df_test['STAT_DT'] = [base_date + timedelta(minutes=i) for i in range(len(df_test))]
    
    # 결과 DataFrame 생성
    results = []
    jump_count = 0
    extreme_count = 0
    extreme_detected = 0
    
    for i, idx in enumerate(indices):
        current_time = df_test['STAT_DT'].iloc[idx]
        seq_start_time = df_test['STAT_DT'].iloc[idx-30]
        prediction_time = current_time + timedelta(minutes=10)
        
        is_jump = (seq_max_list[i] < 280) and (y_test[i] >= 300)
        if is_jump:
            jump_count += 1
        
        if y_test[i] >= 300:
            extreme_count += 1
            if y_pred[i] >= 290:
                extreme_detected += 1
        
        results.append({
            '현재시간': current_time.strftime('%Y-%m-%d %H:%M'),
            '예측시간(+10분)': prediction_time.strftime('%Y-%m-%d %H:%M'),
            '시퀀스시작': seq_start_time.strftime('%Y-%m-%d %H:%M'),
            '시퀀스완료': current_time.strftime('%Y-%m-%d %H:%M'),
            '실제값': round(y_test[i], 2),
            '예측값': round(y_pred[i], 2),
            '오차': round(abs(y_test[i] - y_pred[i]), 2),
            '오차율(%)': round(abs(y_test[i] - y_pred[i]) / y_test[i] * 100, 2) if y_test[i] > 0 else 0,
            '시퀀스MAX': round(seq_max_list[i], 2),
            '시퀀스MIN': round(seq_min_list[i], 2),
            '시퀀스범위': round(seq_max_list[i] - seq_min_list[i], 2),
            '점프케이스': 'O' if is_jump else '-',
            '극단값(300+)': 'O' if y_test[i] >= 300 else '-',
            '극단값감지': 'O' if (y_test[i] >= 300 and y_pred[i] >= 290) else '-'
        })
    
    df_results = pd.DataFrame(results)
    
    # 통계 출력
    print(f"\n📊 특수 케이스 분석:")
    print(f"  - 전체 예측: {len(df_results)}개")
    print(f"  - 극단값(300+): {extreme_count}개 ({extreme_count/len(df_results)*100:.1f}%)")
    print(f"  - 극단값 감지: {extreme_detected}/{extreme_count}개 ({extreme_detected/extreme_count*100 if extreme_count > 0 else 0:.1f}%)")
    print(f"  - 점프 케이스: {jump_count}개 ({jump_count/len(df_results)*100:.1f}%)")
    
    # CSV 저장
    df_results.to_csv('BBB_evaluation_results_이상감지FULL.csv', index=False, encoding='utf-8-sig')
    print(f"\n✅ 상세 결과 저장: BBB_evaluation_results_이상감지FULL.csv")
    
    # ===== 4. 그래프 생성 =====
    print("\n[STEP 4] 평가 그래프 생성")
    print("-"*40)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. 예측 vs 실제 산점도
    ax1 = axes[0, 0]
    ax1.scatter(y_test, y_pred, alpha=0.5, s=10)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax1.set_xlabel('Actual')
    ax1.set_ylabel('Predicted')
    ax1.set_title(f'Actual vs Predicted (With Anomaly Features)\nMAE={test_mae:.2f}, R²={test_r2:.3f}')
    ax1.grid(True, alpha=0.3)
    
    # 2. 시계열 비교
    ax2 = axes[0, 1]
    plot_size = min(300, len(y_test))
    ax2.plot(range(plot_size), y_test[:plot_size], 'b-', label='Actual', alpha=0.7, linewidth=1)
    ax2.plot(range(plot_size), y_pred[:plot_size], 'r--', label='Predicted', alpha=0.7, linewidth=1)
    ax2.axhline(y=300, color='orange', linestyle='--', label='Extreme(300)', alpha=0.5)
    ax2.set_xlabel('Time Index')
    ax2.set_ylabel('Value')
    ax2.set_title('Time Series Comparison (First 300)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 오차 분포
    ax3 = axes[0, 2]
    errors = y_pred - y_test
    ax3.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    ax3.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax3.set_xlabel('Prediction Error')
    ax3.set_ylabel('Frequency')
    ax3.set_title(f'Error Distribution\nMean={np.mean(errors):.2f}, Std={np.std(errors):.2f}')
    ax3.grid(True, alpha=0.3)
    
    # 4. 극단값 성능
    ax4 = axes[1, 0]
    extreme_mask = y_test >= 300
    if extreme_mask.any():
        ax4.scatter(y_test[~extreme_mask], y_pred[~extreme_mask], 
                   alpha=0.3, s=5, label='Normal', color='blue')
        ax4.scatter(y_test[extreme_mask], y_pred[extreme_mask], 
                   alpha=0.8, s=20, label='Extreme(300+)', color='red')
        ax4.plot([200, 500], [200, 500], 'k--', lw=1)
        ax4.axhline(y=300, color='orange', linestyle='--', alpha=0.5)
        ax4.axvline(x=300, color='orange', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Actual')
        ax4.set_ylabel('Predicted')
        ax4.set_title(f'Extreme Value Performance\nDetected: {extreme_detected}/{extreme_count}')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # 5. 점프 케이스 분석
    ax5 = axes[1, 1]
    if jump_count > 0:
        jump_mask = df_results['점프케이스'] == 'O'
        jump_actual = df_results[jump_mask]['실제값'].values
        jump_pred = df_results[jump_mask]['예측값'].values
        jump_seq_max = df_results[jump_mask]['시퀀스MAX'].values
        
        ax5.scatter(jump_seq_max, jump_actual, label='Actual Jump', s=50, alpha=0.8, color='red')
        ax5.scatter(jump_seq_max, jump_pred, label='Predicted', s=30, alpha=0.6, color='blue')
        
        for i in range(len(jump_seq_max)):
            ax5.plot([jump_seq_max[i], jump_seq_max[i]], 
                    [jump_pred[i], jump_actual[i]], 
                    'gray', alpha=0.3, linewidth=0.5)
        
        ax5.axhline(y=300, color='orange', linestyle='--', label='Threshold')
        ax5.axvline(x=280, color='green', linestyle='--', alpha=0.5, label='SeqMax=280')
        ax5.set_xlabel('Sequence MAX')
        ax5.set_ylabel('Value')
        ax5.set_title(f'Jump Cases Analysis\n(SeqMax<280 → Actual≥300: {jump_count} cases)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'No Jump Cases', ha='center', va='center', fontsize=14)
        ax5.set_title('Jump Cases Analysis')
    
    # 6. Feature 중요도 (Top 20)
    ax6 = axes[1, 2]
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(20)
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_importance)))
    ax6.barh(range(len(feature_importance)), feature_importance['importance'].values[::-1], color=colors[::-1])
    ax6.set_yticks(range(len(feature_importance)))
    ax6.set_yticklabels(feature_importance['feature'].values[::-1], fontsize=8)
    ax6.set_xlabel('Importance')
    ax6.set_title('Top 20 Feature Importance')
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('BBB.CSV Evaluation Results (With Full Anomaly Detection Features)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('BBB_evaluation_graphs_이상감지FULL.png', dpi=150, bbox_inches='tight')
    print("✅ 그래프 저장: BBB_evaluation_graphs_이상감지FULL.png")
    
    # ===== 5. 최종 요약 =====
    print("\n" + "="*80)
    print("📊 최종 평가 요약 (이상감지 Feature FULL)")
    print("="*80)
    print(f"1. 모델 성능:")
    print(f"   - 학습 MAE: {train_mae:.2f}")
    print(f"   - 평가 MAE: {test_mae:.2f}")
    print(f"   - 성능 차이: {abs(test_mae - train_mae):.2f}")
    
    print(f"\n2. 특수 케이스 성능:")
    print(f"   - 극단값(300+): {extreme_count}개 발생")
    print(f"   - 극단값 감지: {extreme_detected}개 ({extreme_detected/extreme_count*100 if extreme_count > 0 else 0:.1f}%)")
    print(f"   - 점프 케이스: {jump_count}개 발생")
    
    print(f"\n3. Feature 정보:")
    print(f"   - 총 Feature 수: {len(X_train.columns)}개")
    print(f"   - 타겟 이상감지: 7개")
    print(f"   - 컬럼별 이상감지: {col_anomaly_count}개")
    
    print(f"\n4. Top 10 중요 Features:")
    top_features = feature_importance.head(10)
    for idx, row in top_features.iterrows():
        # 🔥 이상감지 Feature 표시
        marker = "🔥" if any(keyword in row['feature'] for keyword in ['_near_', '_rising_', '_falling_', '_recent_', '_acceleration', '_volatility', '_danger_', '_risk_', '_threshold']) else ""
        print(f"   {marker} {row['feature']}: {row['importance']:.4f}")
    
    print(f"\n5. 저장 파일:")
    print(f"   - 모델: xgboost_model_30min_10min_이상감지FULL.pkl")
    print(f"   - 결과: BBB_evaluation_results_이상감지FULL.csv")
    print(f"   - 그래프: BBB_evaluation_graphs_이상감지FULL.png")
    
    return model, df_results

# 실행
if __name__ == '__main__':
    model, results = train_and_evaluate_complete()