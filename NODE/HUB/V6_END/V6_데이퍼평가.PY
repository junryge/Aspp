import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_and_save_csv(test_file='BBB.CSV'):
    """
    모델 평가 + 상세 CSV 저장 (시간 컬럼 포함)
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
    else:
        base_date = datetime(2024, 1, 1)
        df_test['STAT_DT'] = [base_date + timedelta(minutes=i) for i in range(len(df_test))]
    
    # Feature 생성 및 예측
    results = []
    indices = []
    X_test = []
    y_test = []
    seq_max_list = []
    seq_min_list = []
    
    for i in range(30, len(df_test) - 10):
        seq = df_test[TARGET_COL].iloc[i-30:i].values
        
        features = {
            'target_mean': np.mean(seq),
            'target_std': np.std(seq),
            'target_last_5_mean': np.mean(seq[-5:]),
            'target_max': np.max(seq),
            'target_min': np.min(seq),
            'target_slope': np.polyfit(np.arange(30), seq, 1)[0],
            'target_last_10_mean': np.mean(seq[-10:]),
            'target_first_10_mean': np.mean(seq[:10])
        }
        
        X_test.append(features)
        y_test.append(df_test[TARGET_COL].iloc[i:i+10].max())
        seq_max_list.append(np.max(seq))
        seq_min_list.append(np.min(seq))
        indices.append(i)
    
    X_test = pd.DataFrame(X_test)
    y_test = np.array(y_test)
    
    # 예측
    y_pred = model.predict(X_test)
    
    # 평가 지표
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # 결과 DataFrame 생성 (시간 정보 포함)
    jump_count = 0
    extreme_count = 0
    extreme_detected = 0
    
    for i, idx in enumerate(indices):
        current_time = df_test['STAT_DT'].iloc[idx]
        seq_start_time = df_test['STAT_DT'].iloc[idx-30]
        prediction_time = current_time + timedelta(minutes=10)
        
        is_jump = (seq_max_list[i] < 280) and (y_test[i] >= 300)
        if is_jump:
            jump_count += 1
        
        if y_test[i] >= 300:
            extreme_count += 1
            if y_pred[i] >= 290:
                extreme_detected += 1
        
        # CSV 저장 컬럼 (원하는 순서대로)
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
    
    # 평가 결과 출력
    print(f"\n📊 평가 결과 ({test_file}):")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²:   {r2:.3f}")
    print(f"\n  전체: {len(df_results)}개")
    print(f"  극단값: {extreme_count}개 ({extreme_detected}개 감지, {extreme_detected/extreme_count*100 if extreme_count > 0 else 0:.1f}%)")
    print(f"  점프: {jump_count}개")
    
    # CSV 저장
    df_results.to_csv('evaluation_results.csv', index=False, encoding='utf-8-sig')
    print(f"\n✅ 상세 결과 저장: evaluation_results.csv")
    
    # 요약 저장
    summary = pd.DataFrame([{
        '평가파일': test_file,
        '평가시간': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'R2': round(r2, 3),
        '전체샘플': len(df_results),
        '극단값개수': extreme_count,
        '극단값감지': extreme_detected,
        '감지율(%)': round(extreme_detected/extreme_count*100 if extreme_count > 0 else 0, 1),
        '점프케이스': jump_count
    }])
    
    summary.to_csv('evaluation_summary.csv', index=False, encoding='utf-8-sig')
    print("✅ 요약 저장: evaluation_summary.csv")
    
    return df_results

# 실행
if __name__ == '__main__':
    results = evaluate_and_save_csv('BBB.CSV')
    
    # 다른 파일도 평가하려면
    # results = evaluate_and_save_csv('다른파일.csv')