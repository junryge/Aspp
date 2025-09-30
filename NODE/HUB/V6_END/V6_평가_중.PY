import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta

def evaluate_all_predictions():
    """전체 데이터를 슬라이딩 윈도우로 평가"""
    
    # 모델 로드
    try:
        with open('xgboost_model_30min_10min.pkl', 'rb') as f:
            model = pickle.load(f)
        print("✅ 모델 로드 완료")
    except Exception as e:
        print(f"❌ 모델 파일 없음: {e}")
        return None
    
    # 데이터 로드
    df = pd.read_csv('data/HUBROOM_PIVOT_DATA.CSV', on_bad_lines='skip')
    print(f"✅ 데이터 로드 완료: {len(df)}개 행")
    
    TARGET_COL = 'CURRENT_M16A_3F_JOB_2'
    
    # STAT_DT 처리
    if 'STAT_DT' in df.columns:
        try:
            df['STAT_DT'] = pd.to_datetime(df['STAT_DT'].astype(str), format='%Y%m%d%H%M')
        except:
            print("⚠️ STAT_DT 변환 실패, 가상 시간 생성")
            base_time = datetime(2024, 1, 1, 0, 0)
            df['STAT_DT'] = [base_time + timedelta(minutes=i) for i in range(len(df))]
    
    results = []
    
    # 슬라이딩 윈도우: 30개 시퀀스 → 10분 후 예측
    # i=30부터 시작 (0~29로 첫 예측, 실제값은 30번째)
    for i in range(30, len(df)):
        # 과거 30개 데이터
        seq_data = df.iloc[i-30:i].copy()
        seq_target = seq_data[TARGET_COL].values
        
        # 현재 시점 (시퀀스 마지막)
        current_time = seq_data['STAT_DT'].iloc[-1]
        
        # 예측 시점 (10분 후)
        prediction_time = current_time + timedelta(minutes=10)
        
        # 실제값 (i번째 행)
        actual_value = df.iloc[i][TARGET_COL]
        actual_time = df.iloc[i]['STAT_DT']
        
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
        
        # 300 이상 점프 감지 (시퀀스 내)
        jump_detected = np.any(seq_target >= 300)
        
        # 결과 저장
        results.append({
            '현재시간': current_time.strftime('%Y-%m-%d %H:%M'),
            '예측시점': prediction_time.strftime('%Y-%m-%d %H:%M'),
            '실제시점': actual_time.strftime('%Y-%m-%d %H:%M'),
            '실제값': actual_value,
            '예측값': round(prediction, 2),
            '오차': round(actual_value - prediction, 2),
            '오차율(%)': round(abs(actual_value - prediction) / max(actual_value, 1) * 100, 2),
            '시퀀스MAX': np.max(seq_target),
            '시퀀스MIN': np.min(seq_target),
            '시퀀스평균': round(np.mean(seq_target), 2),
            '300이상점프': '🔴' if jump_detected else '',
            '실제값상태': '🔴극단' if actual_value >= 300 else ('🟡주의' if actual_value >= 280 else '🟢정상'),
            '예측값상태': '🔴극단' if prediction >= 300 else ('🟡주의' if prediction >= 280 else '🟢정상')
        })
        
        # 진행상황 출력
        if (i - 30) % 100 == 0:
            print(f"진행중... {i-30}/{len(df)-30} ({(i-30)/(len(df)-30)*100:.1f}%)")
    
    # DataFrame 변환
    results_df = pd.DataFrame(results)
    
    # CSV 저장
    output_file = 'prediction_evaluation_results.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 결과 저장 완료: {output_file}")
    
    # 통계 출력
    print("\n" + "="*80)
    print("📊 평가 통계")
    print("="*80)
    print(f"총 예측 수: {len(results_df)}")
    print(f"평균 오차: {results_df['오차'].abs().mean():.2f}")
    print(f"평균 오차율: {results_df['오차율(%)'].mean():.2f}%")
    print(f"최대 오차: {results_df['오차'].abs().max():.2f}")
    print(f"\n300이상 점프 구간: {results_df['300이상점프'].value_counts().get('🔴', 0)}개")
    print(f"실제값 극단(≥300): {(results_df['실제값'] >= 300).sum()}개")
    print(f"예측값 극단(≥300): {(results_df['예측값'] >= 300).sum()}개")
    
    # 상위 오차 구간
    print("\n" + "="*80)
    print("❌ 오차 상위 10개 구간")
    print("="*80)
    top_errors = results_df.nlargest(10, '오차율(%)')
    print(top_errors[['현재시간', '실제값', '예측값', '오차', '오차율(%)', '시퀀스MAX', '300이상점프']].to_string(index=False))
    
    # 극단값 구간
    extreme_cases = results_df[results_df['실제값'] >= 300]
    if len(extreme_cases) > 0:
        print("\n" + "="*80)
        print("🔴 실제 극단값(≥300) 구간")
        print("="*80)
        print(extreme_cases[['현재시간', '실제값', '예측값', '오차', '시퀀스MAX', '시퀀스MIN']].to_string(index=False))
    
    return results_df

if __name__ == '__main__':
    print("🚀 실시간 예측 평가 시작...\n")
    results = evaluate_all_predictions()
    
    if results is not None:
        print(f"\n✅ 평가 완료! 총 {len(results)}개 예측 생성")
        print(f"📁 결과 파일: prediction_evaluation_results.csv")