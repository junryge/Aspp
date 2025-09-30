import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_and_save_csv(test_file='BBB.CSV'):
    """
    점프 케이스 정확히 감지
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
        
        # 실제값들
        actual_at_time = df_test[TARGET_COL].iloc[i+9]
        actual_max_10min = df_test[TARGET_COL].iloc[i:i+10].max()
        
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
        
        pred_value = model.predict(features)[0]
        
        # 점프 케이스: 시퀀스 MAX < 280 AND 10분 구간 MAX >= 300
        seq_max = np.max(seq)
        is_jump = (seq_max < 280) and (actual_max_10min >= 300)
        
        if is_jump:
            jump_count += 1
            print(f"🔥 점프 발견! 시퀀스MAX: {seq_max:.0f} → 실제MAX: {actual_max_10min:.0f}")
        
        results.append({
            '현재시간': current_time.strftime('%Y-%m-%d %H:%M'),
            '예측시간(+10분)': prediction_time.strftime('%Y-%m-%d %H:%M'),
            '시퀀스시작': seq_start.strftime('%Y-%m-%d %H:%M'),
            '시퀀스완료': seq_end.strftime('%Y-%m-%d %H:%M'),
            '실제값': round(actual_at_time, 2),
            '10분구간MAX': round(actual_max_10min, 2),
            '예측값': round(pred_value, 2),
            '오차': round(abs(actual_max_10min - pred_value), 2),
            '시퀀스MAX': round(seq_max, 2),
            '시퀀스MIN': round(np.min(seq), 2),
            '점프케이스': 'O' if is_jump else '-',
            '극단값(300+)': 'O' if actual_max_10min >= 300 else '-'
        })
    
    df_results = pd.DataFrame(results)
    
    print(f"\n총 점프 케이스: {jump_count}개")
    
    # 점프 케이스만 출력
    if jump_count > 0:
        jump_df = df_results[df_results['점프케이스'] == 'O']
        print("\n점프 케이스 상세:")
        print(jump_df[['현재시간', '시퀀스MAX', '10분구간MAX', '예측값', '오차']])
    
    # CSV 저장
    df_results.to_csv('evaluation_results.csv', index=False, encoding='utf-8-sig')
    print("\n✅ 저장: evaluation_results.csv")
    
    return df_results

# 실행
if __name__ == '__main__':
    results = evaluate_and_save_csv('BBB.CSV')