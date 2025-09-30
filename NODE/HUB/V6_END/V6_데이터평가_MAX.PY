import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_and_save_csv(test_file='BBB.CSV'):
    """
    평가 - 예측값이 10분MAX값
    """
    # 모델 로드
    with open('xgboost_model_30min_10min.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # 데이터 로드
    df_test = pd.read_csv(test_file, on_bad_lines='skip')
    TARGET_COL = 'CURRENT_M16A_3F_JOB_2'
    
    # STAT_DT 처리
    if 'STAT_DT' in df_test.columns:
        df_test['STAT_DT'] = pd.to_datetime(df_test['STAT_DT'].astype(str), format='%Y%m%d%H%M')
    
    results = []
    jump_count = 0
    
    for i in range(30, len(df_test) - 10):
        seq = df_test[TARGET_COL].iloc[i-30:i].values
        
        current_time = df_test['STAT_DT'].iloc[i]
        seq_start = df_test['STAT_DT'].iloc[i-30]
        seq_end = df_test['STAT_DT'].iloc[i-1]
        prediction_time = df_test['STAT_DT'].iloc[i+9]
        
        # 실제 10분 MAX
        actual_max = df_test[TARGET_COL].iloc[i:i+10].max()
        
        # Features
        features = pd.DataFrame([{
            'target_mean': np.mean(seq),
            'target_std': np.std(seq),
            'target_last_5_mean': np.mean(seq[-5:]),
            'target_max': np.max(seq),
            'target_min': np.min(seq),
            'target_slope': np.polyfit(np.arange(30), seq, 1)[0],
            'target_last_10_mean': np.mean(seq[-10:]),
            'target_first_10_mean': np.mean(seq[:10])
        }])
        
        # 예측 (10분 MAX)
        pred_value = model.predict(features)[0]
        
        seq_max = np.max(seq)
        is_jump = (seq_max < 280) and (actual_max >= 300)
        if is_jump:
            jump_count += 1
        
        results.append({
            '현재시간': current_time.strftime('%Y-%m-%d %H:%M'),
            '예측시간(+10분)': prediction_time.strftime('%Y-%m-%d %H:%M'),
            '시퀀스시작': seq_start.strftime('%Y-%m-%d %H:%M'),
            '시퀀스완료': seq_end.strftime('%Y-%m-%d %H:%M'),
            '실제값(10분MAX)': round(actual_max, 2),
            '예측값(10분MAX)': round(pred_value, 2),
            '오차': round(abs(actual_max - pred_value), 2),
            '시퀀스MAX': round(seq_max, 2),
            '시퀀스MIN': round(np.min(seq), 2),
            '점프케이스': 'O' if is_jump else '-',
            '극단값(300+)': 'O' if actual_max >= 300 else '-'
        })
    
    df_results = pd.DataFrame(results)
    
    # 평가
    y_true = df_results['실제값(10분MAX)'].values
    y_pred = df_results['예측값(10분MAX)'].values
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"\nMAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"점프: {jump_count}개")
    
    df_results.to_csv('evaluation_results.csv', index=False, encoding='utf-8-sig')
    print(f"\n✅ 저장: evaluation_results.csv")
    
    return df_results

# 실행
if __name__ == '__main__':
    results = evaluate_and_save_csv('BBB.CSV')