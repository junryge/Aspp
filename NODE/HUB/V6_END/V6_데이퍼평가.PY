import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_and_save_csv(test_file='BBB.CSV'):
    """
    각 시점의 실제값 그대로 + 예측값 비교
    """
    # 모델 로드
    try:
        with open('xgboost_model_30min_10min.pkl', 'rb') as f:
            model = pickle.load(f)
    except:
        print("❌ 모델 파일 없음")
        return
    
    # 데이터 로드
    df_test = pd.read_csv(test_file, on_bad_lines='skip')
    TARGET_COL = 'CURRENT_M16A_3F_JOB_2'
    
    # STAT_DT 처리
    if 'STAT_DT' in df_test.columns:
        try:
            df_test['STAT_DT'] = pd.to_datetime(df_test['STAT_DT'].astype(str), format='%Y%m%d%H%M')
        except:
            base_date = datetime(2024, 1, 1, 10, 0)  # 10:00 시작
            df_test['STAT_DT'] = [base_date + timedelta(minutes=i) for i in range(len(df_test))]
    
    # 전체 데이터에 실제값 추가
    all_results = []
    
    # 모든 시점의 실제값 저장
    for i in range(len(df_test)):
        all_results.append({
            '날짜시간': df_test['STAT_DT'].iloc[i].strftime('%Y-%m-%d %H:%M'),
            '실제값': round(df_test[TARGET_COL].iloc[i], 2)
        })
    
    # 예측 추가 (30분 시퀀스 필요)
    for i in range(30, len(df_test) - 10):
        seq = df_test[TARGET_COL].iloc[i-30:i].values
        
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
        
        # 예측 (10분 구간 MAX)
        pred_value = model.predict(features)[0]
        max_10min = df_test[TARGET_COL].iloc[i:i+10].max()
        
        # i+10 시점에 예측값 추가
        if i + 10 < len(all_results):
            all_results[i + 10]['예측값'] = round(pred_value, 2)
            all_results[i + 10]['10분구간MAX'] = round(max_10min, 2)
            all_results[i + 10]['시퀀스MAX'] = round(np.max(seq), 2)
            all_results[i + 10]['시퀀스MIN'] = round(np.min(seq), 2)
    
    df_results = pd.DataFrame(all_results)
    
    # 예측값이 있는 행만 평가
    eval_df = df_results.dropna(subset=['예측값'])
    
    if len(eval_df) > 0:
        # MAX 기준 평가 (모델이 MAX 예측하도록 학습됨)
        actual = eval_df['10분구간MAX'].values
        predicted = eval_df['예측값'].values
        
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        r2 = r2_score(actual, predicted)
        
        print(f"\n📊 평가 결과:")
        print(f"  MAE:  {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R²:   {r2:.3f}")
    
    # CSV 저장
    df_results.to_csv('evaluation_results.csv', index=False, encoding='utf-8-sig')
    print(f"\n✅ 저장: evaluation_results.csv")
    
    # 샘플 출력
    print("\n샘플:")
    print(df_results[['날짜시간', '실제값', '예측값']].head(50))
    
    return df_results

# 실행
if __name__ == '__main__':
    results = evaluate_and_save_csv('BBB.CSV')