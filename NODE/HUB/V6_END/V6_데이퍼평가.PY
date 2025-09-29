import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_and_save_csv(test_file='BBB.CSV'):
    """
    기존 예측 로직 유지 + 실제값 정확히 표시
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
    
    # 예측 수행 (기존 로직 그대로)
    for i in range(30, len(df_test) - 10):
        seq = df_test[TARGET_COL].iloc[i-30:i].values
        
        current_time = df_test['STAT_DT'].iloc[i]
        seq_start = df_test['STAT_DT'].iloc[i-30]
        seq_end = df_test['STAT_DT'].iloc[i-1]
        prediction_time = df_test['STAT_DT'].iloc[i+9]
        
        # 실제값 - 각 시점의 값
        actual_at_time = df_test[TARGET_COL].iloc[i+9]  # 예측 시점의 실제값
        actual_max = df_test[TARGET_COL].iloc[i:i+10].max()  # 10분 구간 MAX
        
        # Features (기존 그대로)
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
        
        # 예측 (기존 그대로)
        pred_value = model.predict(features)[0]
        
        results.append({
            '현재시간': current_time.strftime('%Y-%m-%d %H:%M'),
            '예측시간(+10분)': prediction_time.strftime('%Y-%m-%d %H:%M'),
            '시퀀스시작': seq_start.strftime('%Y-%m-%d %H:%M'),
            '시퀀스완료': seq_end.strftime('%Y-%m-%d %H:%M'),
            '실제값': round(actual_at_time, 2),  # 해당 시점 실제값
            '예측값': round(pred_value, 2),
            '오차': round(abs(actual_max - pred_value), 2),  # MAX 기준
            '시퀀스MAX': round(np.max(seq), 2),
            '시퀀스MIN': round(np.min(seq), 2),
            '점프케이스': 'O' if (np.max(seq) < 280 and actual_max >= 300) else '-',
            '극단값(300+)': 'O' if actual_max >= 300 else '-'
        })
    
    df_results = pd.DataFrame(results)
    
    # CSV 저장
    df_results.to_csv('evaluation_results.csv', index=False, encoding='utf-8-sig')
    print("✅ 저장 완료: evaluation_results.csv")
    
    return df_results

# 실행
if __name__ == '__main__':
    results = evaluate_and_save_csv('BBB.CSV')