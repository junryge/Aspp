import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_and_save_csv(test_file='BBB.CSV'):
    """
    실시간 예측 로직과 동일하게 평가
    """
    # 모델 로드
    try:
        with open('xgboost_model_30min_10min.pkl', 'rb') as f:
            model = pickle.load(f)
    except:
        print("❌ 모델 파일 없음")
        return None
    
    # 데이터 로드
    df_test = pd.read_csv(test_file, on_bad_lines='skip')
    TARGET_COL = 'CURRENT_M16A_3F_JOB_2'
    
    # STAT_DT 처리
    if 'STAT_DT' in df_test.columns:
        try:
            df_test['STAT_DT'] = pd.to_datetime(df_test['STAT_DT'].astype(str), format='%Y%m%d%H%M')
        except:
            base_date = datetime(2024, 1, 1)
            df_test['STAT_DT'] = [base_date + timedelta(minutes=i) for i in range(len(df_test))]
    
    results = []
    jump_count = 0
    
    for i in range(30, len(df_test) - 10):
        # 30개 시퀀스 추출 (실시간 예측과 동일)
        seq_target = df_test[TARGET_COL].iloc[i-30:i].values
        
        # 시간 정보
        seq_start = df_test['STAT_DT'].iloc[i-30]
        seq_end = df_test['STAT_DT'].iloc[i-1]
        current_time = df_test['STAT_DT'].iloc[i]
        prediction_start = current_time
        prediction_end = df_test['STAT_DT'].iloc[i+9]
        
        # Feature 생성 (실시간 예측과 동일)
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
        
        X_pred = pd.DataFrame([features])
        
        # 10분 MAX 예측 (실시간 예측과 동일)
        prediction_max = model.predict(X_pred)[0]
        
        # 실제 10분 MAX
        actual_max = df_test[TARGET_COL].iloc[i:i+10].max()
        
        # 점프 케이스 체크
        seq_max = np.max(seq_target)
        is_jump = (seq_max < 280) and (actual_max >= 300)
        if is_jump:
            jump_count += 1
        
        results.append({
            '현재시간': current_time.strftime('%Y-%m-%d %H:%M'),
            '예측시간(+10분)': prediction_end.strftime('%Y-%m-%d %H:%M'),
            '시퀀스시작': seq_start.strftime('%Y-%m-%d %H:%M'),
            '시퀀스완료': seq_end.strftime('%Y-%m-%d %H:%M'),
            '실제값': round(df_test[TARGET_COL].iloc[i+9], 2),  # i+9 시점 실제값
            '10분MAX실제': round(actual_max, 2),
            '10분MAX예측': round(prediction_max, 2),
            '오차': round(abs(actual_max - prediction_max), 2),
            '시퀀스MAX': round(seq_max, 2),
            '시퀀스MIN': round(np.min(seq_target), 2),
            '점프케이스': 'O' if is_jump else '-',
            '극단값(300+)': 'O' if actual_max >= 300 else '-'
        })
    
    df_results = pd.DataFrame(results)
    
    # 평가 지표 (10분 MAX 기준)
    y_true = df_results['10분MAX실제'].values
    y_pred = df_results['10분MAX예측'].values
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print("\n" + "="*60)
    print("📊 평가 결과 (실시간 예측 로직)")
    print("="*60)
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")
    
    # 극단값 평가
    extreme_count = sum(y_true >= 300)
    if extreme_count > 0:
        extreme_detected = sum((y_true >= 300) & (y_pred >= 290))
        print(f"\n극단값(300+): {extreme_count}개")
        print(f"극단값 감지: {extreme_detected}개 ({extreme_detected/extreme_count*100:.1f}%)")
    
    print(f"점프 케이스: {jump_count}개")
    
    # CSV 저장
    df_results.to_csv('evaluation_results.csv', index=False, encoding='utf-8-sig')
    print(f"\n✅ 저장: evaluation_results.csv")
    
    return df_results

# 실행
if __name__ == '__main__':
    results = evaluate_and_save_csv('BBB.CSV')