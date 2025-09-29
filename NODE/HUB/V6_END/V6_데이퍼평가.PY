import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_and_save_csv(test_file='BBB.CSV'):
    """
    시간과 실제값 정확히 매칭하여 평가
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
    
    # 30분 시퀀스로 10분 후 예측
    for i in range(30, len(df_test) - 10):
        # 시퀀스 (i-30 ~ i-1)
        seq = df_test[TARGET_COL].iloc[i-30:i].values
        
        # 시간 정보
        seq_start_time = df_test['STAT_DT'].iloc[i-30]  # 시퀀스 시작
        seq_end_time = df_test['STAT_DT'].iloc[i-1]     # 시퀀스 끝
        current_time = df_test['STAT_DT'].iloc[i]       # 현재 (예측 시작점)
        target_time = df_test['STAT_DT'].iloc[i+9]      # 10분 후
        
        # 실제값: i ~ i+9 구간의 최대값
        actual_value = df_test[TARGET_COL].iloc[i:i+10].max()
        
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
        
        # 예측
        pred_value = model.predict(features)[0]
        
        # 결과 저장
        results.append({
            '현재시간': current_time.strftime('%Y-%m-%d %H:%M'),
            '예측구간시작': current_time.strftime('%Y-%m-%d %H:%M'),
            '예측구간종료': target_time.strftime('%Y-%m-%d %H:%M'),
            '시퀀스시작': seq_start_time.strftime('%Y-%m-%d %H:%M'),
            '시퀀스종료': seq_end_time.strftime('%Y-%m-%d %H:%M'),
            '실제값(10분MAX)': round(actual_value, 2),
            '예측값': round(pred_value, 2),
            '오차': round(abs(actual_value - pred_value), 2),
            '시퀀스MAX': round(np.max(seq), 2),
            '시퀀스MIN': round(np.min(seq), 2),
            '점프케이스': 'O' if (np.max(seq) < 280 and actual_value >= 300) else '-',
            '극단값(300+)': 'O' if actual_value >= 300 else '-'
        })
    
    df_results = pd.DataFrame(results)
    
    # 평가 지표
    actual = df_results['실제값(10분MAX)'].values
    predicted = df_results['예측값'].values
    
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    
    print(f"\n📊 평가 결과:")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²:   {r2:.3f}")
    
    extreme_count = sum(actual >= 300)
    extreme_detected = sum((actual >= 300) & (predicted >= 290))
    
    print(f"\n  극단값: {extreme_count}개 중 {extreme_detected}개 감지")
    
    # CSV 저장
    df_results.to_csv('evaluation_results.csv', index=False, encoding='utf-8-sig')
    print(f"\n✅ 저장: evaluation_results.csv")
    
    # 검증 출력 (처음 5개)
    print("\n검증 (처음 5개):")
    for _, row in df_results.head(5).iterrows():
        print(f"{row['현재시간']} → {row['예측구간종료']}: 실제={row['실제값(10분MAX)']}, 예측={row['예측값']}")
    
    return df_results

# 실행
if __name__ == '__main__':
    results = evaluate_and_save_csv('BBB.CSV')