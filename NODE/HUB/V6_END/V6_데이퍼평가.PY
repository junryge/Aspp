import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_and_save_csv(test_file='BBB.CSV'):
    """
    전체 실제값 + 10분 후 예측 평가
    """
    # 모델 로드
    try:
        with open('xgboost_model_30min_10min.pkl', 'rb') as f:
            model = pickle.load(f)
    except:
        print("❌ 모델 파일 없음")
        return
    
    # 테스트 데이터 로드
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
    
    # 30분 시퀀스로 예측
    for i in range(30, len(df_test) - 10):
        # 시퀀스 (i-30 ~ i-1)
        seq = df_test[TARGET_COL].iloc[i-30:i].values
        
        # 시간 정보
        seq_start_time = df_test['STAT_DT'].iloc[i-30]
        seq_end_time = df_test['STAT_DT'].iloc[i-1]
        current_time = df_test['STAT_DT'].iloc[i]
        
        # 10분 후 실제값들 (i ~ i+9 각각의 값)
        future_10_values = df_test[TARGET_COL].iloc[i:i+10].values
        future_10_max = np.max(future_10_values)
        
        # 각 시점의 시간과 실제값
        for j in range(10):
            future_time = df_test['STAT_DT'].iloc[i+j]
            actual_value = df_test[TARGET_COL].iloc[i+j]
            
            # 예측은 10분 후 최대값 (첫 번째만)
            if j == 0:
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
            
            results.append({
                '시점': future_time.strftime('%Y-%m-%d %H:%M'),
                '분후': f'+{j+1}분',
                '실제값': round(actual_value, 2),
                '10분MAX실제': round(future_10_max, 2),
                '10분MAX예측': round(pred_value, 2) if j == 0 else '-',
                '오차': round(abs(future_10_max - pred_value), 2) if j == 0 else '-',
                '시퀀스시작': seq_start_time.strftime('%Y-%m-%d %H:%M'),
                '시퀀스종료': seq_end_time.strftime('%Y-%m-%d %H:%M'),
                '시퀀스MAX': round(np.max(seq), 2),
                '시퀀스MIN': round(np.min(seq), 2),
                '극단값': 'O' if actual_value >= 300 else '-'
            })
    
    df_results = pd.DataFrame(results)
    
    # 평가 (10분 MAX 기준)
    eval_df = df_results[df_results['10분MAX예측'] != '-'].copy()
    actual = eval_df['10분MAX실제'].astype(float).values
    predicted = eval_df['10분MAX예측'].astype(float).values
    
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    
    print(f"\n📊 평가 결과:")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²:   {r2:.3f}")
    
    # CSV 저장
    df_results.to_csv('evaluation_full_results.csv', index=False, encoding='utf-8-sig')
    print(f"\n✅ 전체 저장: evaluation_full_results.csv")
    
    # 샘플 출력
    print("\n샘플 (처음 10개):")
    print(df_results[['시점', '분후', '실제값', '10분MAX실제', '10분MAX예측']].head(10))
    
    return df_results

# 실행
if __name__ == '__main__':
    results = evaluate_and_save_csv('BBB.CSV')