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
    시퀀스 시작/완료 시간 포함 (컬럼 순서 조정)
    """
    print("="*80)
    print("XGBoost 30분→10분 예측 모델 학습 및 평가")
    print("="*80)
    
    # ===== 1. 학습 단계 (aas.csv) =====
    print("\n[STEP 1] aas.csv로 모델 학습")
    print("-"*40)
    
    df_train = pd.read_csv('aas.csv', on_bad_lines='skip')
    
    # Feature 생성 함수
    def create_features(df, start_idx=30):
        features_list = []
        labels = []
        seq_max_list = []
        seq_min_list = []
        indices = []
        
        TARGET_COL = 'CURRENT_M16A_3F_JOB_2'
        
        for i in range(start_idx, len(df) - 10):
            seq_target = df[TARGET_COL].iloc[i-30:i].values
            
            features = {
                'target_mean': np.mean(seq_target),
                'target_std': np.std(seq_target),
                'target_last_5_mean': np.mean(seq_target[-5:]),
                'target_max': np.max(seq_target),
                'target_min': np.min(seq_target),
                'target_slope': np.polyfit(np.arange(30), seq_target, 1)[0],
                'target_last_10_mean': np.mean(seq_target[-10:]),
                'target_first_10_mean': np.mean(seq_target[:10])
            }
            
            features_list.append(features)
            labels.append(df[TARGET_COL].iloc[i:i+10].max())
            seq_max_list.append(np.max(seq_target))
            seq_min_list.append(np.min(seq_target))
            indices.append(i)
        
        return pd.DataFrame(features_list), np.array(labels), seq_max_list, seq_min_list, indices
    
    # 학습 데이터 생성
    X_train, y_train, _, _, _ = create_features(df_train)
    
    # 모델 학습
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    
    # 학습/검증 분할
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    model.fit(X_tr, y_tr)
    
    # 학습 데이터 평가
    y_val_pred = model.predict(X_val)
    train_mae = mean_absolute_error(y_val, y_val_pred)
    train_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    train_r2 = r2_score(y_val, y_val_pred)
    
    print(f"학습 데이터 성능:")
    print(f"  MAE:  {train_mae:.4f}")
    print(f"  RMSE: {train_rmse:.4f}")
    print(f"  R²:   {train_r2:.4f}")
    
    # 모델 저장
    with open('xgboost_model_30min_10min.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✅ 모델 저장 완료")
    
    # ===== 2. 평가 단계 (BBB.CSV) =====
    print("\n[STEP 2] BBB.CSV로 모델 평가")
    print("-"*40)
    
    df_test = pd.read_csv('BBB.CSV', on_bad_lines='skip')
    
    # BBB.CSV Feature 생성
    X_test, y_test, seq_max_list, seq_min_list, indices = create_features(df_test)
    
    # 예측
    y_pred = model.predict(X_test)
    
    # 평가 지표
    test_mae = mean_absolute_error(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_r2 = r2_score(y_test, y_pred)
    
    print(f"BBB.CSV 평가 결과:")
    print(f"  MAE:  {test_mae:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  R²:   {test_r2:.4f}")
    
    # ===== 3. 상세 분석 결과 생성 =====
    print("\n[STEP 3] 상세 분석 결과 생성")
    print("-"*40)
    
    # STAT_DT 처리 (날짜 시간 형식)
    TARGET_COL = 'CURRENT_M16A_3F_JOB_2'
    
    # 날짜 컬럼이 있는 경우 처리
    if 'STAT_DT' in df_test.columns:
        try:
            # YYYYMMDDHHMM 형식 시도
            df_test['STAT_DT'] = pd.to_datetime(df_test['STAT_DT'], format='%Y%m%d%H%M')
        except:
            try:
                # YYYY-MM-DD HH:MM:SS 형식 시도
                df_test['STAT_DT'] = pd.to_datetime(df_test['STAT_DT'])
            except:
                # 실패시 인덱스 기반으로 생성
                base_date = datetime(2024, 1, 1)
                df_test['STAT_DT'] = [base_date + timedelta(minutes=i) for i in range(len(df_test))]
    else:
        # STAT_DT가 없으면 가상으로 생성
        base_date = datetime(2024, 1, 1)
        df_test['STAT_DT'] = [base_date + timedelta(minutes=i) for i in range(len(df_test))]
    
    # 결과 DataFrame 생성
    results = []
    jump_count = 0
    extreme_count = 0
    extreme_detected = 0
    
    for i, idx in enumerate(indices):
        # 시간 계산
        current_time = df_test['STAT_DT'].iloc[idx]  # 시퀀스 완료 시점 (현재)
        seq_start_time = df_test['STAT_DT'].iloc[idx-30]  # 시퀀스 시작 시점
        prediction_time = current_time + timedelta(minutes=10)  # 예측 시점
        
        # 점프 케이스 판단 (시퀀스 MAX < 280 & 실제값 >= 300)
        is_jump = (seq_max_list[i] < 280) and (y_test[i] >= 300)
        if is_jump:
            jump_count += 1
        
        # 극단값 케이스
        if y_test[i] >= 300:
            extreme_count += 1
            if y_pred[i] >= 290:  # 극단값 감지 성공
                extreme_detected += 1
        
        # 컬럼 순서 조정: 현재시간 → 예측시간 → 시퀀스시작 → 시퀀스완료 → 나머지
        results.append({
            '현재시간': current_time.strftime('%Y-%m-%d %H:%M'),
            '예측시간(+10분)': prediction_time.strftime('%Y-%m-%d %H:%M'),
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
    
    # 점프 케이스 상세 출력
    if jump_count > 0:
        jump_df = df_results[df_results['점프케이스'] == 'O']
        print(f"\n🔥 점프 케이스 상세 (시퀀스MAX<280 → 실제값≥300):")
        display_cols = ['현재시간', '시퀀스시작', '시퀀스완료', '시퀀스MAX', '실제값', '예측값', '오차']
        print(jump_df[display_cols].head(5).to_string())
    
    # 극단값 케이스 상세 출력
    if extreme_count > 0:
        extreme_df = df_results[df_results['극단값(300+)'] == 'O']
        print(f"\n⚠️ 극단값 케이스 상세 (실제값 ≥ 300):")
        display_cols = ['현재시간', '시퀀스완료', '실제값', '예측값', '오차', '극단값감지']
        print(extreme_df[display_cols].head(5).to_string())
    
    # CSV 저장
    df_results.to_csv('BBB_evaluation_results.csv', index=False, encoding='utf-8-sig')
    print(f"\n✅ 상세 결과 저장: BBB_evaluation_results.csv")
    
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
    ax1.set_title(f'Actual vs Predicted\nMAE={test_mae:.2f}, R²={test_r2:.3f}')
    ax1.grid(True, alpha=0.3)
    
    # 2. 시계열 비교 (처음 300개)
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
        
        # 점프 감지선
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
    
    # 6. Feature 중요도
    ax6 = axes[1, 2]
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_importance)))
    ax6.barh(range(len(feature_importance)), feature_importance['importance'].values, color=colors)
    ax6.set_yticks(range(len(feature_importance)))
    ax6.set_yticklabels(feature_importance['feature'].values)
    ax6.set_xlabel('Importance')
    ax6.set_title('Feature Importance')
    
    # 중요도 값 표시
    for i, v in enumerate(feature_importance['importance'].values):
        ax6.text(v + 0.001, i, f'{v:.3f}', va='center')
    
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('BBB.CSV Evaluation Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('BBB_evaluation_graphs.png', dpi=150, bbox_inches='tight')
    print("✅ 그래프 저장: BBB_evaluation_graphs.png")
    
    # ===== 5. 최종 요약 =====
    print("\n" + "="*80)
    print("📊 최종 평가 요약")
    print("="*80)
    print(f"1. 모델 성능:")
    print(f"   - 학습 MAE: {train_mae:.2f}")
    print(f"   - 평가 MAE: {test_mae:.2f}")
    print(f"   - 성능 차이: {abs(test_mae - train_mae):.2f}")
    
    print(f"\n2. 특수 케이스 성능:")
    print(f"   - 극단값(300+): {extreme_count}개 발생")
    print(f"   - 극단값 감지: {extreme_detected}개 ({extreme_detected/extreme_count*100 if extreme_count > 0 else 0:.1f}%)")
    print(f"   - 점프 케이스: {jump_count}개 발생")
    
    print(f"\n3. 시간 정보:")
    if len(df_results) > 0:
        print(f"   - 데이터 시작: {df_results['시퀀스시작'].iloc[0]}")
        print(f"   - 데이터 종료: {df_results['예측시간(+10분)'].iloc[-1]}")
    
    print(f"\n4. 저장 파일:")
    print(f"   - 모델: xgboost_model_30min_10min.pkl")
    print(f"   - 결과: BBB_evaluation_results.csv (컬럼 순서 조정)")
    print(f"   - 그래프: BBB_evaluation_graphs.png")
    
    # 컬럼 순서 확인 출력
    print(f"\n5. CSV 컬럼 순서:")
    print(f"   {' | '.join(df_results.columns[:6])}")
    print(f"   {' | '.join(df_results.columns[6:])}")
    
    return model, df_results

# 실행
if __name__ == '__main__':
    model, results = train_and_evaluate_complete()