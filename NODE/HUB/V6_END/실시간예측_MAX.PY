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
    
    seq_start = recent_30['STAT_DT'].iloc[0]
    seq_end = recent_30['STAT_DT'].iloc[-1]
    prediction_start = seq_end + timedelta(minutes=1)
    prediction_end = seq_end + timedelta(minutes=10)
    
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
    
    # 예측 (10분 구간 MAX)
    prediction_max = model.predict(X_pred)[0]
    
    # 결과 출력
    print("\n" + "="*60)
    print("🚀 실시간 HUBROOM 예측")
    print("="*60)
    print(f"현재시간: {seq_end.strftime('%Y-%m-%d %H:%M')}")
    print(f"시퀀스: {seq_start.strftime('%H:%M')} ~ {seq_end.strftime('%H:%M')} (30분)")
    print(f"예측구간: {prediction_start.strftime('%H:%M')} ~ {prediction_end.strftime('%H:%M')} (10분)")
    print(f"\n📊 10분 구간 MAX 예측값: {prediction_max:.2f}")
    
    if prediction_max >= 300:
        print("⚠️ 상태: 🔴 극단값 경고! (10분 내 300+ 예상)")
    elif prediction_max >= 280:
        print("⚠️ 상태: 🟡 주의 필요 (10분 내 280+ 예상)")
    else:
        print("✅ 상태: 🟢 정상 범위")
    
    # 점프 가능성 체크
    if features['target_max'] < 280 and prediction_max >= 300:
        print("\n🔥 점프 경고: 현재 280 미만 → 10분 내 300+ 예상!")
    
    print("="*60)
    
    return prediction_max

# 실행
if __name__ == '__main__':
    result = realtime_prediction()
    print(f"\n최종 결과: 10분 내 최대값 {result:.2f} 예상")