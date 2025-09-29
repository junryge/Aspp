import numpy as np
import pandas as pd
import pickle
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def realtime_prediction():
    """
    학습된 모델로 실시간 예측 수행
    data/HUBROOM_PIVOT_DATA.CSV 사용
    """
    print("="*80)
    print("🚀 실시간 HUBROOM 예측 시스템")
    print("="*80)
    
    # ===== 1. 모델 로드 =====
    print("\n[STEP 1] 학습된 모델 로드")
    print("-"*40)
    
    try:
        with open('xgboost_model_30min_10min.pkl', 'rb') as f:
            model = pickle.load(f)
        print("✅ 모델 로드 성공: xgboost_model_30min_10min.pkl")
    except FileNotFoundError:
        print("❌ 모델 파일이 없습니다. 먼저 모델을 학습시켜주세요.")
        return None
    
    # ===== 2. 실시간 데이터 로드 =====
    print("\n[STEP 2] 실시간 데이터 로드")
    print("-"*40)
    
    try:
        df = pd.read_csv('data/HUBROOM_PIVOT_DATA.CSV', on_bad_lines='skip')
        print(f"✅ 데이터 로드 성공: {len(df)}개 행")
    except FileNotFoundError:
        print("❌ 데이터 파일을 찾을 수 없습니다: data/HUBROOM_PIVOT_DATA.CSV")
        return None
    
    # 타겟 컬럼 확인
    TARGET_COL = 'CURRENT_M16A_3F_JOB_2'
    
    if TARGET_COL not in df.columns:
        print(f"❌ 타겟 컬럼 '{TARGET_COL}'이 없습니다.")
        print(f"사용 가능한 컬럼: {df.columns.tolist()}")
        return None
    
    # ===== 3. 최근 30분 데이터 추출 =====
    print("\n[STEP 3] 최근 30분 시퀀스 추출")
    print("-"*40)
    
    # 최근 30개 데이터 확인
    if len(df) < 30:
        print(f"❌ 데이터가 부족합니다. 최소 30개 필요 (현재: {len(df)}개)")
        return None
    
    # 최근 30개 추출
    recent_30 = df.tail(30).copy()
    
    # 시간 정보 처리
    if 'STAT_DT' in recent_30.columns:
        try:
            recent_30['STAT_DT'] = pd.to_datetime(recent_30['STAT_DT'], format='%Y%m%d%H%M')
        except:
            try:
                recent_30['STAT_DT'] = pd.to_datetime(recent_30['STAT_DT'])
            except:
                base_time = datetime.now() - timedelta(minutes=29)
                recent_30['STAT_DT'] = [base_time + timedelta(minutes=i) for i in range(30)]
    else:
        base_time = datetime.now() - timedelta(minutes=29)
        recent_30['STAT_DT'] = [base_time + timedelta(minutes=i) for i in range(30)]
    
    # 시퀀스 정보 출력
    seq_start = recent_30['STAT_DT'].iloc[0]
    seq_end = recent_30['STAT_DT'].iloc[-1]
    
    print(f"📅 시퀀스 기간:")
    print(f"   시작: {seq_start.strftime('%Y-%m-%d %H:%M')}")
    print(f"   종료: {seq_end.strftime('%Y-%m-%d %H:%M')}")
    
    # ===== 4. Feature 생성 =====
    print("\n[STEP 4] Feature 생성")
    print("-"*40)
    
    # 타겟 시퀀스
    seq_target = recent_30[TARGET_COL].values
    
    # Feature 계산
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
    
    # 현재 상태 분석
    print("📊 현재 시퀀스 분석:")
    print(f"   평균값: {features['target_mean']:.2f}")
    print(f"   최근 5분 평균: {features['target_last_5_mean']:.2f}")
    print(f"   최대값: {features['target_max']:.2f}")
    print(f"   최소값: {features['target_min']:.2f}")
    print(f"   변동폭: {features['target_max'] - features['target_min']:.2f}")
    print(f"   추세: {'상승' if features['target_slope'] > 0 else '하락'} ({features['target_slope']:.3f})")
    
    # DataFrame으로 변환
    X_pred = pd.DataFrame([features])
    
    # ===== 5. 예측 수행 =====
    print("\n[STEP 5] 예측 수행")
    print("-"*40)
    
    # 예측
    prediction = model.predict(X_pred)[0]
    
    # 예측 시점
    prediction_time = seq_end + timedelta(minutes=10)
    
    print("\n🎯 예측 결과:")
    print("="*60)
    print(f"  예측 시점: {prediction_time.strftime('%Y-%m-%d %H:%M')}")
    print(f"  예측값: {prediction:.2f}")
    
    # 위험도 평가
    if prediction >= 300:
        risk_level = "🔴 극단값 경고!"
        risk_color = 'red'
    elif prediction >= 280:
        risk_level = "🟡 주의 필요"
        risk_color = 'orange'
    else:
        risk_level = "🟢 정상 범위"
        risk_color = 'green'
    
    print(f"  위험도: {risk_level}")
    print("="*60)
    
    # 점프 가능성 평가
    if features['target_max'] < 280 and prediction >= 300:
        print("\n⚠️ 점프 경고: 현재 280 미만에서 300+ 예측!")
    
    # ===== 6. 시각화 =====
    print("\n[STEP 6] 예측 시각화")
    print("-"*40)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 왼쪽: 시퀀스 + 예측
    ax1.plot(range(30), seq_target, 'b-', linewidth=2, label='Current Sequence')
    ax1.scatter(29, seq_target[-1], color='blue', s=100, zorder=5)
    
    # 예측값 표시
    ax1.scatter(39, prediction, color=risk_color, s=200, marker='*', 
               label=f'Prediction: {prediction:.1f}', zorder=5)
    ax1.plot([29, 39], [seq_target[-1], prediction], 'r--', alpha=0.5)
    
    # 위험 구간 표시
    ax1.axhline(y=300, color='red', linestyle='--', alpha=0.3, label='Extreme (300)')
    ax1.axhline(y=280, color='orange', linestyle='--', alpha=0.3, label='Warning (280)')
    
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('HUBROOM Value')
    ax1.set_title(f'Real-time Prediction\n{seq_end.strftime("%Y-%m-%d %H:%M")} → +10min')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 오른쪽: Feature 중요도
    feature_names = list(features.keys())
    feature_values = list(features.values())
    
    # 정규화 (시각화용)
    normalized_values = [(v - min(feature_values)) / (max(feature_values) - min(feature_values) + 0.001) 
                        for v in feature_values]
    
    colors = ['red' if 'last_5' in name else 'blue' for name in feature_names]
    bars = ax2.barh(range(len(feature_names)), normalized_values, color=colors, alpha=0.6)
    
    ax2.set_yticks(range(len(feature_names)))
    ax2.set_yticklabels(feature_names)
    ax2.set_xlabel('Normalized Value')
    ax2.set_title('Current Feature Values')
    
    # 값 표시
    for i, (bar, val) in enumerate(zip(bars, feature_values)):
        ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{val:.1f}', va='center')
    
    plt.tight_layout()
    plt.savefig('realtime_prediction.png', dpi=150, bbox_inches='tight')
    print("✅ 그래프 저장: realtime_prediction.png")
    plt.show()
    
    # ===== 7. 결과 저장 =====
    print("\n[STEP 7] 예측 결과 저장")
    print("-"*40)
    
    # 결과 DataFrame
    result_df = pd.DataFrame([{
        '예측시각': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        '시퀀스시작': seq_start.strftime('%Y-%m-%d %H:%M'),
        '시퀀스종료': seq_end.strftime('%Y-%m-%d %H:%M'),
        '예측시점': prediction_time.strftime('%Y-%m-%d %H:%M'),
        '예측값': round(prediction, 2),
        '위험도': risk_level.split()[1] if len(risk_level.split()) > 1 else risk_level,
        '현재평균': round(features['target_mean'], 2),
        '최근5분평균': round(features['target_last_5_mean'], 2),
        '최대값': round(features['target_max'], 2),
        '최소값': round(features['target_min'], 2),
        '추세': round(features['target_slope'], 3)
    }])
    
    # 기존 예측 기록에 추가
    try:
        history = pd.read_csv('realtime_predictions_history.csv')
        history = pd.concat([history, result_df], ignore_index=True)
    except:
        history = result_df
    
    history.to_csv('realtime_predictions_history.csv', index=False, encoding='utf-8-sig')
    print("✅ 예측 기록 저장: realtime_predictions_history.csv")
    
    # 현재 예측만 별도 저장
    result_df.to_csv('latest_prediction.csv', index=False, encoding='utf-8-sig')
    print("✅ 최신 예측 저장: latest_prediction.csv")
    
    # ===== 8. 추천 조치사항 =====
    print("\n" + "="*80)
    print("💡 추천 조치사항")
    print("="*80)
    
    if prediction >= 300:
        print("🔴 극단값 예상:")
        print("   1. 즉시 모니터링 강화")
        print("   2. 대응 인력 준비")
        print("   3. 시스템 부하 분산 검토")
    elif prediction >= 280:
        print("🟡 주의 단계:")
        print("   1. 모니터링 주시")
        print("   2. 예비 조치 준비")
    else:
        print("🟢 정상 운영:")
        print("   1. 정기 모니터링 유지")
    
    print("\n" + "="*80)
    print("✅ 실시간 예측 완료!")
    print("="*80)
    
    return {
        'prediction': prediction,
        'prediction_time': prediction_time,
        'risk_level': risk_level,
        'features': features,
        'sequence': seq_target
    }

def continuous_monitoring(interval_minutes=1):
    """
    지속적인 모니터링 모드 (선택사항)
    """
    import time
    
    print("🔄 연속 모니터링 모드 시작 (Ctrl+C로 종료)")
    print("-"*60)
    
    while True:
        try:
            result = realtime_prediction()
            if result:
                print(f"\n⏰ 다음 예측까지 {interval_minutes}분 대기...")
                time.sleep(interval_minutes * 60)
                print("\n" + "="*80)
        except KeyboardInterrupt:
            print("\n\n모니터링 종료")
            break
        except Exception as e:
            print(f"오류 발생: {e}")
            time.sleep(10)

# 실행
if __name__ == '__main__':
    # 단일 예측
    result = realtime_prediction()
    
    # 연속 모니터링 원할 경우 아래 주석 해제
    # continuous_monitoring(interval_minutes=1)