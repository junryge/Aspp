import numpy as np
import pandas as pd
import lightgbm as lgb
import pickle
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def train_and_evaluate_quantile():
    """
    LightGBM Quantile Regression (alpha=0.9)
    보수적 예측으로 사전감지 성능 향상
    """
    print("="*80)
    print("LightGBM 분위 회귀 (Quantile Regression) - alpha=0.9")
    print("30분 → 10분 예측 모델 학습 및 평가")
    print("="*80)
    
    # 핵심 12개 컬럼
    FEATURE_COLS = {
        'storage': ['M16A_3F_STORAGE_UTIL'],
        'cmd': ['M16A_3F_CMD', 'M16A_6F_TO_HUB_CMD'],
        'inflow': ['M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2', 'M14A_3F_TO_HUB_JOB2'],
        'outflow': ['M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB', 'M16A_3F_TO_M14A_3F_JOB'],
        'maxcapa': ['M16A_6F_LFT_MAXCAPA', 'M16A_2F_LFT_MAXCAPA']
    }
    
    TARGET_COL = 'CURRENT_M16A_3F_JOB_2'
    
    # Feature 생성 함수 (기존과 동일)
    def create_features(df, start_idx=30):
        features_list = []
        labels = []
        seq_max_list = []
        seq_min_list = []
        indices = []
        
        for i in range(start_idx, len(df) - 10):
            seq_target = df[TARGET_COL].iloc[i-30:i].values
            
            features = {
                # 타겟 컬럼 특성
                'target_mean': np.mean(seq_target),
                'target_std': np.std(seq_target),
                'target_last_5_mean': np.mean(seq_target[-5:]),
                'target_max': np.max(seq_target),
                'target_min': np.min(seq_target),
                'target_slope': np.polyfit(np.arange(30), seq_target, 1)[0],
                'target_last_10_mean': np.mean(seq_target[-10:]),
                'target_first_10_mean': np.mean(seq_target[:10]),
            }
            
            # 각 컬럼 그룹별 특성 추가
            for group_name, cols in FEATURE_COLS.items():
                for col in cols:
                    if col in df.columns:
                        seq_data = df[col].iloc[i-30:i].values
                        
                        features[f'{col}_mean'] = np.mean(seq_data)
                        features[f'{col}_std'] = np.std(seq_data)
                        features[f'{col}_max'] = np.max(seq_data)
                        features[f'{col}_min'] = np.min(seq_data)
                        features[f'{col}_last_5_mean'] = np.mean(seq_data[-5:])
                        features[f'{col}_last_10_mean'] = np.mean(seq_data[-10:])
                        features[f'{col}_slope'] = np.polyfit(np.arange(30), seq_data, 1)[0]
                        features[f'{col}_first_10_mean'] = np.mean(seq_data[:10])
                        features[f'{col}_mid_10_mean'] = np.mean(seq_data[10:20])
                        features[f'{col}_last_value'] = seq_data[-1]
            
            # 유입-유출 차이
            inflow_sum = 0
            outflow_sum = 0
            for col in FEATURE_COLS['inflow']:
                if col in df.columns:
                    inflow_sum += df[col].iloc[i-1]
            for col in FEATURE_COLS['outflow']:
                if col in df.columns:
                    outflow_sum += df[col].iloc[i-1]
            features['net_flow'] = inflow_sum - outflow_sum
            
            # CMD 총합
            cmd_sum = 0
            for col in FEATURE_COLS['cmd']:
                if col in df.columns:
                    cmd_sum += df[col].iloc[i-1]
            features['total_cmd'] = cmd_sum
            
            features_list.append(features)
            labels.append(df[TARGET_COL].iloc[i:i+10].max())
            seq_max_list.append(np.max(seq_target))
            seq_min_list.append(np.min(seq_target))
            indices.append(i)
        
        return pd.DataFrame(features_list), np.array(labels), seq_max_list, seq_min_list, indices
    
    # ===== 1. 학습 단계 =====
    print("\n[STEP 1] 학습 데이터 로드 및 Feature 생성")
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
    print(f"학습 샘플 수: {len(X_train)}개")
    
    # 학습/검증 분할
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # ===== 2. LightGBM Quantile Regression 학습 =====
    print("\n[STEP 2] LightGBM 분위 회귀 모델 학습")
    print("-"*40)
    print("📊 모델 설정:")
    print("  - objective: quantile (분위 회귀)")
    print("  - alpha: 0.9 (90% 분위수 예측)")
    print("  - 효과: 보수적 예측 (높게 예측)")
    print()
    
    model = lgb.LGBMRegressor(
        objective='quantile',  # 분위 회귀!
        alpha=0.9,              # 90% 분위수
        n_estimators=150,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    print("모델 학습 중...")
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.log_evaluation(period=0)]
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
    
    # 보수적 예측 확인
    over_predict = np.sum(y_val_pred > y_val)
    print(f"\n보수적 예측 확인:")
    print(f"  실제보다 높게 예측: {over_predict}/{len(y_val)} ({over_predict/len(y_val)*100:.1f}%)")
    print(f"  평균 예측 - 실제: {np.mean(y_val_pred - y_val):.2f}")
    
    # 모델 저장
    model_file = 'lightgbm_quantile_model_30min_10min_12컬럼.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ 모델 저장 완료: {model_file}")
    
    # ===== 3. 평가 단계 (BBB.CSV) =====
    print("\n[STEP 3] 평가 데이터로 모델 테스트")
    print("-"*40)
    
    df_test = pd.read_csv('HUB_20250916_to_20250929.CSV', on_bad_lines='skip')
    
    print(f"평가 데이터: {len(df_test)}개 행")
    
    # Feature 생성
    X_test, y_test, seq_max_list, seq_min_list, indices = create_features(df_test)
    
    # 예측
    y_pred = model.predict(X_test)
    
    # 평가 지표
    test_mae = mean_absolute_error(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_r2 = r2_score(y_test, y_pred)
    
    print(f"\n평가 데이터 성능:")
    print(f"  MAE:  {test_mae:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  R²:   {test_r2:.4f}")
    
    # 보수적 예측 확인
    over_predict_test = np.sum(y_pred > y_test)
    print(f"\n보수적 예측 확인:")
    print(f"  실제보다 높게 예측: {over_predict_test}/{len(y_test)} ({over_predict_test/len(y_test)*100:.1f}%)")
    print(f"  평균 예측 - 실제: {np.mean(y_pred - y_test):.2f}")
    
    # 사전감지 성능
    extreme_actual = np.sum(y_test >= 300)
    extreme_predicted = np.sum(y_pred >= 300)
    extreme_detected = np.sum((y_test >= 300) & (y_pred >= 300))
    
    print(f"\n🚨 사전감지 성능:")
    print(f"  실제 300+: {extreme_actual}개")
    print(f"  예측 300+: {extreme_predicted}개")
    print(f"  정확 감지: {extreme_detected}/{extreme_actual}개 ({extreme_detected/extreme_actual*100 if extreme_actual > 0 else 0:.1f}%)")
    
    # ===== 4. 상세 결과 생성 =====
    print("\n[STEP 4] 상세 분석 결과 생성")
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
    
    for i, idx in enumerate(indices):
        current_time = df_test['STAT_DT'].iloc[idx]
        seq_start_time = df_test['STAT_DT'].iloc[idx-30]
        prediction_time = current_time + timedelta(minutes=10)
        
        results.append({
            '현재시간': current_time.strftime('%Y-%m-%d %H:%M'),
            '예측시간(+10분)': prediction_time.strftime('%Y-%m-%d %H:%M'),
            '시퀀스시작': seq_start_time.strftime('%Y-%m-%d %H:%M'),
            '실제값': round(y_test[i], 2),
            '예측값': round(y_pred[i], 2),
            '오차': round(y_pred[i] - y_test[i], 2),  # 예측 - 실제
            '절대오차': round(abs(y_test[i] - y_pred[i]), 2),
            '오차율(%)': round(abs(y_test[i] - y_pred[i]) / y_test[i] * 100, 2),
            '시퀀스MAX': round(seq_max_list[i], 2),
            '시퀀스MIN': round(seq_min_list[i], 2),
            '보수적예측': 'O' if y_pred[i] > y_test[i] else 'X',
            '실제_300+': 'O' if y_test[i] >= 300 else '-',
            '예측_300+': 'O' if y_pred[i] >= 300 else '-',
            '사전감지성공': 'O' if (y_test[i] >= 300 and y_pred[i] >= 300) else '-'
        })
    
    df_results = pd.DataFrame(results)
    
    # CSV 저장
    output_file = 'BBB_evaluation_quantile_results.csv'
    df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✅ 상세 결과 저장: {output_file}")
    
    # ===== 5. 그래프 생성 =====
    print("\n[STEP 5] 평가 그래프 생성")
    print("-"*40)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 예측 vs 실제
    ax1 = axes[0, 0]
    ax1.scatter(y_test, y_pred, alpha=0.5, s=10)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax1.axhline(y=300, color='orange', linestyle='--', alpha=0.5)
    ax1.axvline(x=300, color='orange', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Actual')
    ax1.set_ylabel('Predicted (Quantile 0.9)')
    ax1.set_title(f'Quantile Regression (alpha=0.9)\nMAE={test_mae:.2f}, R²={test_r2:.3f}')
    ax1.grid(True, alpha=0.3)
    
    # 2. 시계열 비교
    ax2 = axes[0, 1]
    plot_size = min(300, len(y_test))
    ax2.plot(range(plot_size), y_test[:plot_size], 'b-', label='Actual', alpha=0.7, linewidth=1)
    ax2.plot(range(plot_size), y_pred[:plot_size], 'r--', label='Predicted', alpha=0.7, linewidth=1)
    ax2.axhline(y=300, color='orange', linestyle='--', label='Critical(300)', alpha=0.5)
    ax2.set_xlabel('Time Index')
    ax2.set_ylabel('Value')
    ax2.set_title('Time Series Comparison (First 300)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 오차 분포
    ax3 = axes[1, 0]
    errors = y_pred - y_test
    ax3.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    ax3.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax3.axvline(x=np.mean(errors), color='g', linestyle='--', linewidth=2, label=f'Mean={np.mean(errors):.2f}')
    ax3.set_xlabel('Prediction Error (Predicted - Actual)')
    ax3.set_ylabel('Frequency')
    ax3.set_title(f'Error Distribution (Quantile 0.9)\nMean={np.mean(errors):.2f}, Std={np.std(errors):.2f}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 극단값 성능
    ax4 = axes[1, 1]
    extreme_mask = y_test >= 300
    if extreme_mask.any():
        ax4.scatter(y_test[~extreme_mask], y_pred[~extreme_mask], 
                   alpha=0.3, s=5, label='Normal', color='blue')
        ax4.scatter(y_test[extreme_mask], y_pred[extreme_mask], 
                   alpha=0.8, s=20, label='Critical(300+)', color='red')
        ax4.plot([200, 500], [200, 500], 'k--', lw=1)
        ax4.axhline(y=300, color='orange', linestyle='--', alpha=0.5)
        ax4.axvline(x=300, color='orange', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Actual')
        ax4.set_ylabel('Predicted')
        ax4.set_title(f'Critical Value Detection\nDetected: {extreme_detected}/{extreme_actual}')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.suptitle('LightGBM Quantile Regression (alpha=0.9) Evaluation', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    graph_file = 'BBB_evaluation_quantile_graphs.png'
    plt.savefig(graph_file, dpi=150, bbox_inches='tight')
    print(f"✅ 그래프 저장: {graph_file}")
    
    # ===== 6. 최종 요약 =====
    print("\n" + "="*80)
    print("📊 최종 평가 요약")
    print("="*80)
    print(f"1. 모델 성능:")
    print(f"   - 학습 MAE: {train_mae:.2f}")
    print(f"   - 평가 MAE: {test_mae:.2f}")
    
    print(f"\n2. 보수적 예측 특성:")
    print(f"   - 평균 (예측 - 실제): {np.mean(y_pred - y_test):.2f}")
    print(f"   - 실제보다 높게 예측: {over_predict_test}/{len(y_test)} ({over_predict_test/len(y_test)*100:.1f}%)")
    
    print(f"\n3. 사전감지 성능:")
    print(f"   - 실제 300+: {extreme_actual}개")
    print(f"   - 감지 성공: {extreme_detected}개 ({extreme_detected/extreme_actual*100 if extreme_actual > 0 else 0:.1f}%)")
    
    print(f"\n4. 저장 파일:")
    print(f"   - 모델: {model_file}")
    print(f"   - 결과: {output_file}")
    print(f"   - 그래프: {graph_file}")
    
    return model, df_results

# 실행
if __name__ == '__main__':
    model, results = train_and_evaluate_quantile()