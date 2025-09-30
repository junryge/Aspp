import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
import os

def create_single_prediction_features_100min(seq_m14b, seq_m10a, seq_m16, seq_totalcnt):
    """
    하나의 100분 시퀀스로부터 Feature 생성 (32개 기본 통계값만)
    """
    features = {}
    
    # ========== M14AM14B 기본 8개 ==========
    features['m14b_mean'] = np.mean(seq_m14b)
    features['m14b_std'] = np.std(seq_m14b)
    features['m14b_last_5_mean'] = np.mean(seq_m14b[-5:])
    features['m14b_max'] = np.max(seq_m14b)
    features['m14b_min'] = np.min(seq_m14b)
    features['m14b_slope'] = np.polyfit(np.arange(100), seq_m14b, 1)[0]
    features['m14b_last_10_mean'] = np.mean(seq_m14b[-10:])
    features['m14b_first_10_mean'] = np.mean(seq_m14b[:10])
    
    # ========== M14AM10A 기본 8개 ==========
    features['m10a_mean'] = np.mean(seq_m10a)
    features['m10a_std'] = np.std(seq_m10a)
    features['m10a_last_5_mean'] = np.mean(seq_m10a[-5:])
    features['m10a_max'] = np.max(seq_m10a)
    features['m10a_min'] = np.min(seq_m10a)
    features['m10a_slope'] = np.polyfit(np.arange(100), seq_m10a, 1)[0]
    features['m10a_last_10_mean'] = np.mean(seq_m10a[-10:])
    features['m10a_first_10_mean'] = np.mean(seq_m10a[:10])
    
    # ========== M14AM16 기본 8개 ==========
    features['m16_mean'] = np.mean(seq_m16)
    features['m16_std'] = np.std(seq_m16)
    features['m16_last_5_mean'] = np.mean(seq_m16[-5:])
    features['m16_max'] = np.max(seq_m16)
    features['m16_min'] = np.min(seq_m16)
    features['m16_slope'] = np.polyfit(np.arange(100), seq_m16, 1)[0]
    features['m16_last_10_mean'] = np.mean(seq_m16[-10:])
    features['m16_first_10_mean'] = np.mean(seq_m16[:10])
    
    # ========== TOTALCNT 기본 8개 ==========
    features['totalcnt_mean'] = np.mean(seq_totalcnt)
    features['totalcnt_std'] = np.std(seq_totalcnt)
    features['totalcnt_last_5_mean'] = np.mean(seq_totalcnt[-5:])
    features['totalcnt_max'] = np.max(seq_totalcnt)
    features['totalcnt_min'] = np.min(seq_totalcnt)
    features['totalcnt_slope'] = np.polyfit(np.arange(100), seq_totalcnt, 1)[0]
    features['totalcnt_last_10_mean'] = np.mean(seq_totalcnt[-10:])
    features['totalcnt_first_10_mean'] = np.mean(seq_totalcnt[:10])
    
    return features

def evaluate_all_predictions():
    """전체 데이터를 슬라이딩 윈도우로 평가 (100분 시퀀스)"""
    
    print("="*80)
    print("100분 시퀀스 모델 평가")
    print("="*80)
    
    # 모델 로드
    model_file = 'xgboost_100to10_totalcnt.pkl'
    try:
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ 모델 로드 완료: {model_file}")
    except Exception as e:
        print(f"❌ 모델 파일 없음: {e}")
        print("먼저 xgboost_100to10_simple.py를 실행하여 모델을 학습하세요.")
        return None
    
    # 데이터 로드
    csv_file = 'data/aasf.csv'
    if not os.path.exists(csv_file):
        print(f"❌ CSV 파일 없음: {csv_file}")
        return None
    
    df = pd.read_csv(csv_file, on_bad_lines='skip')
    print(f"✅ 데이터 로드 완료: {len(df):,}개 행")
    
    # 필수 컬럼 확인
    required_cols = ['M14AM14B', 'M14AM10A', 'M14AM16', 'TOTALCNT']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ 필수 컬럼 누락: {missing_cols}")
        print(f"현재 컬럼: {list(df.columns)}")
        return None
    
    print(f"✅ 필수 컬럼 확인 완료")
    
    # CURRTIME 처리
    if 'CURRTIME' in df.columns:
        try:
            df['CURRTIME'] = pd.to_datetime(df['CURRTIME'])
            print("✅ CURRTIME 컬럼 사용")
        except:
            print("⚠️ CURRTIME 변환 실패, 가상 시간 생성")
            base_time = datetime(2024, 1, 1, 0, 0)
            df['CURRTIME'] = [base_time + timedelta(minutes=i) for i in range(len(df))]
    else:
        print("⚠️ CURRTIME 없음, 가상 시간 생성")
        base_time = datetime(2024, 1, 1, 0, 0)
        df['CURRTIME'] = [base_time + timedelta(minutes=i) for i in range(len(df))]
    
    results = []
    
    print(f"\n📊 슬라이딩 윈도우 평가 시작...")
    print(f"총 예측 수: {len(df) - 100:,}개")
    
    # 슬라이딩 윈도우: i=100부터 시작
    for i in range(100, len(df)):
        # 100분 시퀀스
        seq_m14b = df['M14AM14B'].iloc[i-100:i].values
        seq_m10a = df['M14AM10A'].iloc[i-100:i].values
        seq_m16 = df['M14AM16'].iloc[i-100:i].values
        seq_totalcnt = df['TOTALCNT'].iloc[i-100:i].values
        
        # 시간 정보
        current_time = df['CURRTIME'].iloc[i-1]  # 시퀀스 마지막
        seq_start_time = df['CURRTIME'].iloc[i-100]  # 시퀀스 시작
        prediction_time = current_time + timedelta(minutes=10)
        actual_time = df['CURRTIME'].iloc[i]
        
        # 실제값
        actual_value = df['TOTALCNT'].iloc[i]
        
        # Feature 생성
        features = create_single_prediction_features_100min(seq_m14b, seq_m10a, seq_m16, seq_totalcnt)
        X_pred = pd.DataFrame([features])
        
        # 예측
        prediction = model.predict(X_pred)[0]
        
        # 황금 패턴 감지
        golden_pattern = (seq_m14b[-1] > 300 and seq_m10a[-1] < 80)
        
        # 위험 구간 감지
        danger_in_seq = np.sum(seq_totalcnt >= 1700) > 0
        
        # 결과 저장
        results.append({
            '시퀀스시작': seq_start_time.strftime('%Y-%m-%d %H:%M'),
            '현재시간': current_time.strftime('%Y-%m-%d %H:%M'),
            '예측시점': prediction_time.strftime('%Y-%m-%d %H:%M'),
            '실제시점': actual_time.strftime('%Y-%m-%d %H:%M'),
            '실제값': round(actual_value, 2),
            '예측값': round(prediction, 2),
            '오차': round(actual_value - prediction, 2),
            '절대오차': round(abs(actual_value - prediction), 2),
            '오차율(%)': round(abs(actual_value - prediction) / max(actual_value, 1) * 100, 2),
            'M14AM14B': round(seq_m14b[-1], 2),
            'M14AM10A': round(seq_m10a[-1], 2),
            'M14AM16': round(seq_m16[-1], 2),
            '시퀀스TOTALCNT_MAX': round(np.max(seq_totalcnt), 2),
            '시퀀스TOTALCNT_MIN': round(np.min(seq_totalcnt), 2),
            '시퀀스TOTALCNT_평균': round(np.mean(seq_totalcnt), 2),
            '황금패턴': 'O' if golden_pattern else '',
            '시퀀스위험': 'O' if danger_in_seq else '',
            '실제위험(1700+)': 'O' if actual_value >= 1700 else '',
            '예측위험(1700+)': 'O' if prediction >= 1650 else ''
        })
        
        # 진행상황
        if (i - 100) % 1000 == 0:
            progress = (i - 100) / (len(df) - 100) * 100
            print(f"  진행중... {i-100}/{len(df)-100} ({progress:.1f}%)")
    
    # DataFrame 변환
    results_df = pd.DataFrame(results)
    
    # CSV 저장
    output_file = 'aasf_evaluation_100min_results.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 결과 저장 완료: {output_file}")
    
    # ===== 통계 분석 =====
    print("\n" + "="*80)
    print("📊 평가 통계")
    print("="*80)
    print(f"총 예측 수: {len(results_df):,}개")
    print(f"평균 절대 오차(MAE): {results_df['절대오차'].mean():.2f}")
    print(f"평균 오차율: {results_df['오차율(%)'].mean():.2f}%")
    print(f"최대 절대 오차: {results_df['절대오차'].max():.2f}")
    print(f"최소 절대 오차: {results_df['절대오차'].min():.2f}")
    
    print(f"\n황금 패턴 발생: {results_df['황금패턴'].value_counts().get('O', 0)}개")
    print(f"시퀀스 위험 구간: {results_df['시퀀스위험'].value_counts().get('O', 0)}개")
    
    # 위험 구간 분석
    actual_danger = results_df['실제위험(1700+)'] == 'O'
    pred_danger = results_df['예측위험(1700+)'] == 'O'
    
    actual_danger_count = actual_danger.sum()
    pred_danger_count = pred_danger.sum()
    danger_detected = (actual_danger & pred_danger).sum()
    
    print(f"\n⚠️ 위험 구간 분석:")
    print(f"  실제 위험(1700+): {actual_danger_count}개")
    print(f"  예측 위험(1650+): {pred_danger_count}개")
    print(f"  위험 감지 성공: {danger_detected}개")
    if actual_danger_count > 0:
        print(f"  위험 감지율: {danger_detected/actual_danger_count*100:.1f}%")
    
    # 오차 상위 10개
    print("\n" + "="*80)
    print("❌ 오차 상위 10개 구간")
    print("="*80)
    top_errors = results_df.nlargest(10, '절대오차')
    print(top_errors[['현재시간', '실제값', '예측값', '절대오차', '오차율(%)', 'M14AM14B', '황금패턴']].to_string(index=False))
    
    # 위험 구간 상세
    if actual_danger_count > 0:
        print("\n" + "="*80)
        print("⚠️ 실제 위험(1700+) 구간 상세")
        print("="*80)
        danger_cases = results_df[actual_danger]
        print(f"총 {len(danger_cases)}개")
        print(danger_cases[['현재시간', '실제값', '예측값', '절대오차', 'M14AM14B', 'M14AM10A', '예측위험(1700+)']].head(20).to_string(index=False))
    
    # 황금 패턴 분석
    golden_cases = results_df[results_df['황금패턴'] == 'O']
    if len(golden_cases) > 0:
        print("\n" + "="*80)
        print("🏆 황금 패턴 발생 구간")
        print("="*80)
        print(f"총 {len(golden_cases)}개")
        print(f"평균 실제값: {golden_cases['실제값'].mean():.2f}")
        print(f"평균 예측값: {golden_cases['예측값'].mean():.2f}")
        print(f"평균 절대오차: {golden_cases['절대오차'].mean():.2f}")
        print(golden_cases[['현재시간', '실제값', '예측값', '절대오차', 'M14AM14B', 'M14AM10A']].head(10).to_string(index=False))
    
    return results_df

if __name__ == '__main__':
    print("\n🚀 100분 시퀀스 모델 실시간 예측 평가 시작...\n")
    results = evaluate_all_predictions()
    
    if results is not None:
        print(f"\n✅ 평가 완료! 총 {len(results):,}개 예측 생성")
        print(f"📁 결과 파일: aasf_evaluation_100min_results.csv")