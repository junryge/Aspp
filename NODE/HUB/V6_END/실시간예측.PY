import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta

def realtime_prediction():
    # 모델 로드
    try:
        with open('xgboost_model_30min_10min.pkl', 'rb') as f:
            model = pickle.load(f)
    except:
        print("❌ 모델 파일 없음")
        return None
    
    # 데이터 로드
    df = pd.read_csv('data/HUBROOM_PIVOT_DATA.CSV', on_bad_lines='skip')
    
    # 최근 30개 데이터
    recent_30 = df.tail(30).copy()
    TARGET_COL = 'CURRENT_M16A_3F_JOB_2'
    seq_target = recent_30[TARGET_COL].values
    
    # STAT_DT 처리
    if 'STAT_DT' in recent_30.columns:
        try:
            recent_30['STAT_DT'] = pd.to_datetime(recent_30['STAT_DT'].astype(str), format='%Y%m%d%H%M')
        except:
            base_time = datetime(2024, 1, 1, 0, 0)
            recent_30['STAT_DT'] = [base_time + timedelta(minutes=i) for i in range(30)]
    
    seq_end = recent_30['STAT_DT'].iloc[-1]
    prediction_time = seq_end + timedelta(minutes=10)
    
    # Feature 생성
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
    
    # 예측
    prediction = model.predict(X_pred)[0]
    
    # 결과 출력
    print(f"\n현재시간: {seq_end.strftime('%Y-%m-%d %H:%M')}")
    print(f"예측시점: {prediction_time.strftime('%Y-%m-%d %H:%M')}")
    print(f"예측값: {prediction:.2f}")
    
    if prediction >= 300:
        print("상태: 🔴 극단값")
    elif prediction >= 280:
        print("상태: 🟡 주의")
    else:
        print("상태: 🟢 정상")
    
    return prediction

# 실행
if __name__ == '__main__':
    result = realtime_prediction()
    print(f"\n결과: {result}")