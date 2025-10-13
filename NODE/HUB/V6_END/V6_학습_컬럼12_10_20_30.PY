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

# 핵심 12개 컬럼 정의 (물리적으로 중요한 것만)
FEATURE_COLS = {
    'storage': ['M16A_3F_STORAGE_UTIL'],  # 극단값 핵심 지표
    'cmd': ['M16A_3F_CMD', 'M16A_6F_TO_HUB_CMD'],  # 주요 CMD
    'inflow': ['M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2', 'M14A_3F_TO_HUB_JOB2'],  # 주요 유입
    'outflow': ['M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB', 'M16A_3F_TO_M14A_3F_JOB'],  # 주요 유출
    'maxcapa': ['M16A_6F_LFT_MAXCAPA', 'M16A_2F_LFT_MAXCAPA']  # 용량 제한
}

TARGET_COL = 'CURRENT_M16A_3F_JOB_2'

def create_features(df, horizon, start_idx=30):
    """
    Feature 생성 함수 (12개 컬럼 기반)
    
    Args:
        df: 데이터프레임
        horizon: 예측 시점 (10, 20, 30분)
        start_idx: 시작 인덱스
    """
    features_list = []
    labels = []
    seq_max_list = []
    seq_min_list = []
    indices = []
    
    # horizon에 따라 루프 범위 조정
    for i in range(start_idx, len(df) - horizon):
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
                    
                    # 기본 통계
                    features[f'{col}_mean'] = np.mean(seq_data)
                    features[f'{col}_std'] = np.std(seq_data)
                    features[f'{col}_max'] = np.max(seq_data)
                    features[f'{col}_min'] = np.min(seq_data)
                    
                    # 최근 특성
                    features[f'{col}_last_5_mean'] = np.mean(seq_data[-5:])
                    features[f'{col}_last_10_mean'] = np.mean(seq_data[-10:])
                    
                    # 추세
                    features[f'{col}_slope'] = np.polyfit(np.arange(30), seq_data, 1)[0]
                    
                    # 구간별 평균 (초기/중간/최근)
                    features[f'{col}_first_10_mean'] = np.mean(seq_data[:10])
                    features[f'{col}_mid_10_mean'] = np.mean(seq_data[10:20])
                    features[f'{col}_last_value'] = seq_data[-1]
        
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
        
        # CMD 총합
        cmd_sum = 0
        for col in FEATURE_COLS['cmd']:
            if col in df.columns:
                cmd_sum += df[col].iloc[i-1]
        features['total_cmd'] = cmd_sum
        
        features_list.append(features)
        
        # horizon에 따라 타겟 설정
        if horizon == 10:
            labels.append(df[TARGET_COL].iloc[i:i+10].max())
        elif horizon == 20:
            labels.append(df[TARGET_COL].iloc[i+10:i+20].max())
        elif horizon == 30:
            labels.append(df[TARGET_COL].iloc[i+20:i+30].max())
        
        seq_max_list.append(np.max(seq_target))
        seq_min_list.append(np.min(seq_target))
        indices.append(i)
    
    return pd.DataFrame(features_list), np.array(labels), seq_max_list, seq_min_list, indices

def train_model_for_horizon(horizon_name, horizon_minutes):
    """
    특정 horizon에 대한 모델 학습 및 평가 (12개 컬럼 버전)
    
    Args:
        horizon_name: '10min', '20min', '30min'
        horizon_minutes: 10, 20, 30
    """
    print("\n" + "="*80)
    print(f"🚀 {horizon_name} 모델 학습 시작 (12개 컬럼, 과거 30분 → {horizon_minutes}분 후 예측)")
    print("="*80)
    
    # ===== 1. 학습 단계 =====
    print(f"\n[STEP 1] 학습 데이터로 {horizon_name} 모델 학습")
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
    X_train, y_train, _, _, _ = create_features(df_train, horizon_minutes)
    
    print(f"\n생성된 Feature 수: {len(X_train.columns)}개")
    print(f"학습 샘플 수: {len(X_train)}개")
    
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
        print(f"  ⚠️ GPU 사용 불가")
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
    
    print(f"모델 학습 중 ({horizon_name})...")
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
    model_filename = f'xgboost_model_30min_{horizon_name}_12col.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ 모델 저장 완료: {model_filename}")
    
    # ===== 2. 평가 단계 =====
    print(f"\n[STEP 2] 평가 데이터로 {horizon_name} 모델 평가")
    print("-"*40)
    
    df_test = pd.read_csv('HUB_20250916_to_20250929.CSV', on_bad_lines='skip')
    
    print(f"평가 데이터: {len(df_test)}개 행")
    
    # 평가 데이터 생성
    X_test, y_test, seq_max_list, seq_min_list, indices = create_features(df_test, horizon_minutes)
    
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
    
    # ===== 3. 상세 분석 결과 생성 =====
    print(f"\n[STEP 3] {horizon_name} 상세 결과 생성")
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
        prediction_time = current_time + timedelta(minutes=horizon_minutes)
        
        # 점프 케이스 판단
        is_jump = (seq_max_list[i] < 280) and (y_test[i] >= 300)
        if is_jump:
            jump_count += 1
        
        # 극단값 케이스
        if y_test[i] >= 300:
            extreme_count += 1
            if y_pred[i] >= 290:
                extreme_detected += 1
        
        results.append({
            '현재시간': current_time.strftime('%Y-%m-%d %H:%M'),
            f'예측시간(+{horizon_minutes}분)': prediction_time.strftime('%Y-%m-%d %H:%M'),
            '시퀀스시작': seq_start_time.strftime('%Y-%m-%d %H:%M'),
            '시퀀스완료': current_time.strftime('%Y-%m-%d %H:%M'),
            '실제값': round(y_test[i], 2),
            '예측값': round(y_pred[i], 2),
            '오차': round(abs(y_test[i] - y_pred[i]), 2),
            '오차율(%)': round(abs(y_test[i] - y_pred[i]) / y_test[i] * 100, 2),
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
    
    # 점프 케이스 상세
    if jump_count > 0:
        jump_df = df_results[df_results['점프케이스'] == 'O']
        print(f"\n🔥 점프 케이스 상세:")
        display_cols = ['현재시간', '시퀀스MAX', '실제값', '예측값', '오차']
        print(jump_df[display_cols].head(5).to_string(index=False))
    
    # 극단값 케이스 상세
    if extreme_count > 0:
        extreme_df = df_results[df_results['극단값(300+)'] == 'O']
        print(f"\n⚠️ 극단값 케이스 상세:")
        display_cols = ['현재시간', '실제값', '예측값', '오차', '극단값감지']
        print(extreme_df[display_cols].head(5).to_string(index=False))
    
    # CSV 저장
    csv_filename = f'BBB_evaluation_results_{horizon_name}_12col.csv'
    df_results.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    print(f"\n✅ 상세 결과 저장: {csv_filename}")
    
    # ===== 4. 그래프 생성 =====
    print(f"\n[STEP 4] {horizon_name} 평가 그래프 생성")
    print("-"*40)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. 예측 vs 실제 산점도
    ax1 = axes[0, 0]
    ax1.scatter(y_test, y_pred, alpha=0.5, s=10)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax1.set_xlabel('Actual')
    ax1.set_ylabel('Predicted')
    ax1.set_title(f'Actual vs Predicted ({horizon_name}, 12col)\nMAE={test_mae:.2f}, R²={test_r2:.3f}')
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
        ax4.set_title(f'Extreme Value Performance ({horizon_name})\nDetected: {extreme_detected}/{extreme_count}')
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
        ax5.set_title(f'Jump Cases Analysis ({horizon_name})\n(SeqMax<280 → Actual≥300: {jump_count} cases)')
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
    
    plt.suptitle(f'Evaluation Results ({horizon_name}, 12 Columns)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    graph_filename = f'BBB_evaluation_graphs_{horizon_name}_12col.png'
    plt.savefig(graph_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 그래프 저장: {graph_filename}")
    
    # ===== 5. 최종 요약 =====
    print("\n" + "-"*80)
    print(f"📊 {horizon_name} 최종 요약 (12개 컬럼)")
    print("-"*80)
    print(f"모델 성능:")
    print(f"  - 학습 MAE: {train_mae:.2f}")
    print(f"  - 평가 MAE: {test_mae:.2f}")
    print(f"  - 성능 차이: {abs(test_mae - train_mae):.2f}")
    
    print(f"\n특수 케이스:")
    print(f"  - 극단값(300+): {extreme_count}개")
    print(f"  - 극단값 감지율: {extreme_detected/extreme_count*100 if extreme_count > 0 else 0:.1f}%")
    print(f"  - 점프 케이스: {jump_count}개")
    
    print(f"\nFeature 정보:")
    print(f"  - 총 Feature 수: {len(X_train.columns)}개")
    print(f"  - 12개 컬럼 기반")
    
    print(f"\n저장 파일:")
    print(f"  - 모델: {model_filename}")
    print(f"  - 결과: {csv_filename}")
    print(f"  - 그래프: {graph_filename}")
    
    return {
        'model': model,
        'results': df_results,
        'metrics': {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'extreme_count': extreme_count,
            'extreme_detected': extreme_detected,
            'jump_count': jump_count
        }
    }

def main():
    """메인 실행 함수 - 10분/20분/30분 모델 순차 학습 (12개 컬럼)"""
    
    print("="*80)
    print("🚀 XGBoost 다중 Horizon 예측 모델 통합 학습 시작 (12개 컬럼)")
    print("="*80)
    print("📋 학습 계획:")
    print("  1. 10분 후 예측 모델 (xgboost_model_30min_10min_12col.pkl)")
    print("  2. 20분 후 예측 모델 (xgboost_model_30min_20min_12col.pkl)")
    print("  3. 30분 후 예측 모델 (xgboost_model_30min_30min_12col.pkl)")
    print("\n📦 사용 컬럼:")
    print("  - storage(1): M16A_3F_STORAGE_UTIL")
    print("  - cmd(2): M16A_3F_CMD, M16A_6F_TO_HUB_CMD")
    print("  - inflow(3): M16A_6F_TO_HUB_JOB, M16A_2F_TO_HUB_JOB2, M14A_3F_TO_HUB_JOB2")
    print("  - outflow(3): M16A_3F_TO_M16A_6F_JOB, M16A_3F_TO_M16A_2F_JOB, M16A_3F_TO_M14A_3F_JOB")
    print("  - maxcapa(2): M16A_6F_LFT_MAXCAPA, M16A_2F_LFT_MAXCAPA")
    print("  - 기타: net_flow, total_cmd")
    print("="*80)
    
    results_all = {}
    
    # 10분 모델
    print("\n\n" + "🔵"*40)
    results_all['10min'] = train_model_for_horizon('10min', 10)
    
    # 20분 모델
    print("\n\n" + "🟢"*40)
    results_all['20min'] = train_model_for_horizon('20min', 20)
    
    # 30분 모델
    print("\n\n" + "🟡"*40)
    results_all['30min'] = train_model_for_horizon('30min', 30)
    
    # 전체 요약
    print("\n\n" + "="*80)
    print("🎉 전체 학습 완료! (12개 컬럼 버전)")
    print("="*80)
    print("\n📊 전체 성능 비교:")
    print("-"*80)
    print(f"{'Horizon':<10} {'Train MAE':<12} {'Test MAE':<12} {'Test RMSE':<12} {'R²':<8} {'극단값 감지율':<15}")
    print("-"*80)
    
    for horizon in ['10min', '20min', '30min']:
        metrics = results_all[horizon]['metrics']
        detection_rate = metrics['extreme_detected']/metrics['extreme_count']*100 if metrics['extreme_count'] > 0 else 0
        print(f"{horizon:<10} {metrics['train_mae']:<12.2f} {metrics['test_mae']:<12.2f} "
              f"{metrics['test_rmse']:<12.2f} {metrics['test_r2']:<8.3f} {detection_rate:<15.1f}%")
    
    print("\n📁 생성된 파일 (12개 컬럼 버전):")
    print("-"*80)
    for horizon in ['10min', '20min', '30min']:
        print(f"\n[{horizon}]")
        print(f"  - xgboost_model_30min_{horizon}_12col.pkl")
        print(f"  - BBB_evaluation_results_{horizon}_12col.csv")
        print(f"  - BBB_evaluation_graphs_{horizon}_12col.png")
    
    print("\n" + "="*80)
    print("✅ 모든 작업 완료! (12개 컬럼)")
    print("="*80)
    
    return results_all

# 실행
if __name__ == '__main__':
    results = main()