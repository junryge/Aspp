import numpy as np
import pandas as pd
import pickle
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def create_features(df, horizon, start_idx=30):
    """
    Feature 생성 함수 (1개 컬럼 - 타겟만)
    실제값 = 정확히 그 시점의 값! (MAX 아님)
    """
    features_list = []
    labels = []
    seq_max_list = []
    seq_min_list = []
    indices = []
    
    TARGET_COL = 'CURRENT_M16A_3F_JOB_2'
    
    for i in range(start_idx, len(df) - horizon):
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
        
        # 🔥 수정: 정확히 그 시점의 실제값!
        if horizon == 10:
            labels.append(df[TARGET_COL].iloc[i+10])  # 10분 후 정확한 값
        elif horizon == 20:
            labels.append(df[TARGET_COL].iloc[i+20])  # 20분 후 정확한 값
        elif horizon == 30:
            labels.append(df[TARGET_COL].iloc[i+30])  # 30분 후 정확한 값
        
        seq_max_list.append(np.max(seq_target))
        seq_min_list.append(np.min(seq_target))
        indices.append(i)
    
    return pd.DataFrame(features_list), np.array(labels), seq_max_list, seq_min_list, indices

def evaluate_all_models():
    """
    이미 학습된 3개 모델(10분/20분/30분)을 불러와서 평가만!
    """
    print("="*80)
    print("🚀 3개 모델 통합 평가 (실제값 = 정확한 시점!)")
    print("="*80)
    
    # ===== 1. 모델 로드 =====
    print("\n[STEP 1] 모델 로드")
    print("-"*40)
    
    models = {}
    for horizon_name in ['10min', '20min', '30min']:
        model_filename = f'model_30min_{horizon_name}.pkl'
        try:
            with open(model_filename, 'rb') as f:
                models[horizon_name] = pickle.load(f)
            print(f"✅ {model_filename} 로드 완료")
        except Exception as e:
            print(f"❌ {model_filename} 로드 실패: {e}")
            print(f"   먼저 학습을 진행해주세요!")
            return None
    
    # ===== 2. 평가 데이터 로드 =====
    print("\n[STEP 2] 평가 데이터 로드")
    print("-"*40)
    
    df_test = pd.read_csv('HUB0906_0929.CSV', on_bad_lines='skip')
    print(f"✅ 평가 데이터 로드: {len(df_test)}개 행")
    
    TARGET_COL = 'CURRENT_M16A_3F_JOB_2'
    
    # STAT_DT 처리
    if 'STAT_DT' in df_test.columns:
        try:
            df_test['STAT_DT'] = pd.to_datetime(df_test['STAT_DT'].astype(str), format='%Y%m%d%H%M')
        except:
            try:
                df_test['STAT_DT'] = pd.to_datetime(df_test['STAT_DT'])
            except:
                print("⚠️ STAT_DT 변환 실패, 가상 시간 생성")
                base_date = datetime(2024, 9, 6, 0, 0)
                df_test['STAT_DT'] = [base_date + timedelta(minutes=i) for i in range(len(df_test))]
    else:
        print("⚠️ STAT_DT 없음, 가상 시간 생성")
        base_date = datetime(2024, 9, 6, 0, 0)
        df_test['STAT_DT'] = [base_date + timedelta(minutes=i) for i in range(len(df_test))]
    
    # ===== 3. 각 모델로 예측 =====
    print("\n[STEP 3] 각 모델로 예측")
    print("-"*40)
    
    predictions = {}
    seq_info = {}
    
    for horizon_name, horizon_minutes in [('10min', 10), ('20min', 20), ('30min', 30)]:
        print(f"\n{horizon_name} 모델 예측 중...")
        
        X_test, y_test, seq_max, seq_min, indices = create_features(df_test, horizon_minutes)
        
        print(f"  생성된 샘플 수: {len(X_test)}개")
        print(f"  실제값 = {horizon_minutes}분 후 정확한 값")
        
        y_pred = models[horizon_name].predict(X_test)
        
        predictions[horizon_name] = {
            'y_pred': y_pred,
            'y_test': y_test,
            'indices': indices
        }
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²:   {r2:.4f}")
        
        if horizon_name == '10min':
            seq_info = {
                'seq_max': seq_max,
                'seq_min': seq_min,
                'indices': indices
            }
    
    # ===== 4. 통합 결과 생성 =====
    print("\n[STEP 4] 통합 결과 CSV 생성")
    print("-"*40)
    
    min_length = len(predictions['30min']['indices'])
    print(f"통합 가능한 샘플 수: {min_length}개")
    
    results = []
    stats = {
        '10min': {'jump': 0, 'extreme': 0, 'extreme_detected': 0},
        '20min': {'jump': 0, 'extreme': 0, 'extreme_detected': 0},
        '30min': {'jump': 0, 'extreme': 0, 'extreme_detected': 0}
    }
    
    for i in range(min_length):
        idx = predictions['30min']['indices'][i]
        
        current_time = df_test['STAT_DT'].iloc[idx]
        seq_start_time = df_test['STAT_DT'].iloc[idx-30]
        
        seq_max_val = seq_info['seq_max'][i]
        seq_min_val = seq_info['seq_min'][i]
        
        result = {
            '현재시간': current_time.strftime('%Y-%m-%d %H:%M'),
            '시퀀스시작': seq_start_time.strftime('%Y-%m-%d %H:%M'),
            '시퀀스MAX': round(seq_max_val, 2),
            '시퀀스MIN': round(seq_min_val, 2),
            '시퀀스범위': round(seq_max_val - seq_min_val, 2),
        }
        
        for horizon_name, horizon_minutes in [('10min', 10), ('20min', 20), ('30min', 30)]:
            pred_time = current_time + timedelta(minutes=horizon_minutes)
            
            y_pred_val = predictions[horizon_name]['y_pred'][i]
            y_test_val = predictions[horizon_name]['y_test'][i]
            
            error = abs(y_test_val - y_pred_val)
            error_rate = (error / y_test_val * 100) if y_test_val > 0 else 0
            
            is_jump = (seq_max_val < 280) and (y_test_val >= 300)
            if is_jump:
                stats[horizon_name]['jump'] += 1
            
            if y_test_val >= 300:
                stats[horizon_name]['extreme'] += 1
                if y_pred_val >= 290:
                    stats[horizon_name]['extreme_detected'] += 1
            
            result[f'예측시간_{horizon_name}'] = pred_time.strftime('%Y-%m-%d %H:%M')
            result[f'실제값_{horizon_name}'] = round(y_test_val, 2)
            result[f'예측값_{horizon_name}'] = round(y_pred_val, 2)
            result[f'오차_{horizon_name}'] = round(error, 2)
            result[f'오차율_{horizon_name}(%)'] = round(error_rate, 2)
            result[f'점프케이스_{horizon_name}'] = 'O' if is_jump else '-'
            result[f'극단값_{horizon_name}'] = 'O' if y_test_val >= 300 else '-'
            result[f'극단값감지_{horizon_name}'] = 'O' if (y_test_val >= 300 and y_pred_val >= 290) else '-'
        
        results.append(result)
        
        if (i + 1) % 500 == 0:
            print(f"  진행 중... {i+1}/{min_length} ({(i+1)/min_length*100:.1f}%)")
    
    df_results = pd.DataFrame(results)
    
    csv_filename = 'evaluation_results_all_models_1col.csv'
    df_results.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    print(f"✅ 통합 결과 저장: {csv_filename}")
    print(f"   총 {len(df_results)}개 예측 생성")
    
    # ===== 5. 통계 출력 =====
    print("\n" + "="*80)
    print("📊 모델별 통계")
    print("="*80)
    
    for horizon_name in ['10min', '20min', '30min']:
        stat = stats[horizon_name]
        y_test = predictions[horizon_name]['y_test'][:min_length]
        y_pred = predictions[horizon_name]['y_pred'][:min_length]
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"\n[{horizon_name}]")
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  극단값(300+): {stat['extreme']}개")
        print(f"  극단값 감지: {stat['extreme_detected']}/{stat['extreme']}개 ({stat['extreme_detected']/stat['extreme']*100 if stat['extreme'] > 0 else 0:.1f}%)")
        print(f"  점프 케이스: {stat['jump']}개")
    
    # ===== 6. 비교 그래프 생성 =====
    print("\n[STEP 5] 비교 그래프 생성")
    print("-"*40)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    colors = {'10min': 'blue', '20min': 'green', '30min': 'orange'}
    
    # 1. MAE 비교
    ax1 = axes[0, 0]
    maes = []
    for horizon_name in ['10min', '20min', '30min']:
        y_test = predictions[horizon_name]['y_test'][:min_length]
        y_pred = predictions[horizon_name]['y_pred'][:min_length]
        maes.append(mean_absolute_error(y_test, y_pred))
    
    ax1.bar(['10min', '20min', '30min'], maes, color=[colors[h] for h in ['10min', '20min', '30min']])
    ax1.set_ylabel('MAE')
    ax1.set_title('MAE Comparison')
    ax1.grid(True, alpha=0.3)
    
    for i, v in enumerate(maes):
        ax1.text(i, v + 1, f'{v:.2f}', ha='center', va='bottom')
    
    # 2. 극단값 감지율 비교
    ax2 = axes[0, 1]
    detection_rates = []
    for horizon_name in ['10min', '20min', '30min']:
        stat = stats[horizon_name]
        rate = (stat['extreme_detected']/stat['extreme']*100) if stat['extreme'] > 0 else 0
        detection_rates.append(rate)
    
    ax2.bar(['10min', '20min', '30min'], detection_rates, color=[colors[h] for h in ['10min', '20min', '30min']])
    ax2.set_ylabel('Detection Rate (%)')
    ax2.set_title('Extreme Value Detection Rate')
    ax2.grid(True, alpha=0.3)
    
    for i, v in enumerate(detection_rates):
        ax2.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
    
    # 3. 시계열 비교
    ax3 = axes[0, 2]
    plot_size = min(200, min_length)
    
    y_test_plot = predictions['10min']['y_test'][:plot_size]
    ax3.plot(range(plot_size), y_test_plot, 'k-', label='Actual', alpha=0.7, linewidth=2)
    
    for horizon_name in ['10min', '20min', '30min']:
        y_pred_plot = predictions[horizon_name]['y_pred'][:plot_size]
        ax3.plot(range(plot_size), y_pred_plot, '--', label=f'Pred {horizon_name}', 
                alpha=0.7, linewidth=1, color=colors[horizon_name])
    
    ax3.axhline(y=300, color='red', linestyle='--', label='Extreme(300)', alpha=0.5)
    ax3.set_xlabel('Time Index')
    ax3.set_ylabel('Value')
    ax3.set_title('Time Series Comparison (First 200)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 점프 케이스 수 비교
    ax4 = axes[1, 0]
    jump_counts = [stats[h]['jump'] for h in ['10min', '20min', '30min']]
    ax4.bar(['10min', '20min', '30min'], jump_counts, color=[colors[h] for h in ['10min', '20min', '30min']])
    ax4.set_ylabel('Count')
    ax4.set_title('Jump Cases Count')
    ax4.grid(True, alpha=0.3)
    
    for i, v in enumerate(jump_counts):
        ax4.text(i, v + 0.5, f'{v}', ha='center', va='bottom')
    
    # 5. 오차 분포 비교
    ax5 = axes[1, 1]
    for horizon_name in ['10min', '20min', '30min']:
        y_test = predictions[horizon_name]['y_test'][:min_length]
        y_pred = predictions[horizon_name]['y_pred'][:min_length]
        errors = y_pred - y_test
        ax5.hist(errors, bins=30, alpha=0.5, label=horizon_name, color=colors[horizon_name])
    
    ax5.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax5.set_xlabel('Prediction Error')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Error Distribution Comparison')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 극단값 예측 산점도
    ax6 = axes[1, 2]
    y_test_30 = predictions['30min']['y_test'][:min_length]
    y_pred_30 = predictions['30min']['y_pred'][:min_length]
    
    extreme_mask = y_test_30 >= 300
    ax6.scatter(y_test_30[~extreme_mask], y_pred_30[~extreme_mask],
               alpha=0.3, s=5, label='Normal', color='blue')
    ax6.scatter(y_test_30[extreme_mask], y_pred_30[extreme_mask],
               alpha=0.8, s=20, label='Extreme(300+)', color='red')
    ax6.plot([200, 500], [200, 500], 'k--', lw=1)
    ax6.axhline(y=300, color='orange', linestyle='--', alpha=0.5)
    ax6.axvline(x=300, color='orange', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Actual')
    ax6.set_ylabel('Predicted')
    ax6.set_title('Extreme Value Performance (30min)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('3 Models Evaluation (Exact Time Point Values)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    graph_filename = 'evaluation_graphs_all_models_1col.png'
    plt.savefig(graph_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 비교 그래프 저장: {graph_filename}")
    
    # ===== 7. 최종 요약 =====
    print("\n" + "="*80)
    print("🎉 평가 완료!")
    print("="*80)
    print(f"\n📊 전체 성능 비교:")
    print("-"*80)
    print(f"{'Model':<10} {'MAE':<10} {'RMSE':<10} {'극단값감지율':<15} {'점프케이스':<12}")
    print("-"*80)
    
    for horizon_name in ['10min', '20min', '30min']:
        y_test = predictions[horizon_name]['y_test'][:min_length]
        y_pred = predictions[horizon_name]['y_pred'][:min_length]
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        stat = stats[horizon_name]
        detection_rate = (stat['extreme_detected']/stat['extreme']*100) if stat['extreme'] > 0 else 0
        
        print(f"{horizon_name:<10} {mae:<10.2f} {rmse:<10.2f} {detection_rate:<15.1f}% {stat['jump']:<12}")
    
    print("\n📁 생성된 파일:")
    print(f"  - {csv_filename}")
    print(f"  - {graph_filename}")
    
    print("\n📋 평가 데이터 정보:")
    print(f"  - 파일: HUB0906_0929.CSV")
    print(f"  - 기간: {df_results['현재시간'].iloc[0]} ~ {df_results['현재시간'].iloc[-1]}")
    print(f"\n✅ 실제값 = 정확히 그 시점의 값 (MAX 아님!)")
    
    return df_results

# 실행
if __name__ == '__main__':
    print("🚀 3개 모델 평가 시작 (정확한 시점 실제값!)\n")
    results = evaluate_all_models()
    
    if results is not None:
        print(f"\n✅ 평가 완료! 총 {len(results)}개 예측 생성")