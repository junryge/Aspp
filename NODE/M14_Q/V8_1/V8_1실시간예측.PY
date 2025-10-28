import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
import os

def create_single_prediction_features(seq_m14b, seq_m10a, seq_m16, seq_totalcnt):
    """
    하나의 280분 시퀀스로부터 Feature 생성 (조기 경보 포함 85개)
    """
    features = {}
    
    # ========== M14AM14B 기본 8개 ==========
    features['m14b_mean'] = np.mean(seq_m14b)
    features['m14b_std'] = np.std(seq_m14b)
    features['m14b_last_5_mean'] = np.mean(seq_m14b[-5:])
    features['m14b_max'] = np.max(seq_m14b)
    features['m14b_min'] = np.min(seq_m14b)
    features['m14b_slope'] = np.polyfit(np.arange(280), seq_m14b, 1)[0]
    features['m14b_last_10_mean'] = np.mean(seq_m14b[-10:])
    features['m14b_first_10_mean'] = np.mean(seq_m14b[:10])
    
    # ========== M14AM10A 기본 8개 ==========
    features['m10a_mean'] = np.mean(seq_m10a)
    features['m10a_std'] = np.std(seq_m10a)
    features['m10a_last_5_mean'] = np.mean(seq_m10a[-5:])
    features['m10a_max'] = np.max(seq_m10a)
    features['m10a_min'] = np.min(seq_m10a)
    features['m10a_slope'] = np.polyfit(np.arange(280), seq_m10a, 1)[0]
    features['m10a_last_10_mean'] = np.mean(seq_m10a[-10:])
    features['m10a_first_10_mean'] = np.mean(seq_m10a[:10])
    
    # ========== M14AM16 기본 8개 ==========
    features['m16_mean'] = np.mean(seq_m16)
    features['m16_std'] = np.std(seq_m16)
    features['m16_last_5_mean'] = np.mean(seq_m16[-5:])
    features['m16_max'] = np.max(seq_m16)
    features['m16_min'] = np.min(seq_m16)
    features['m16_slope'] = np.polyfit(np.arange(280), seq_m16, 1)[0]
    features['m16_last_10_mean'] = np.mean(seq_m16[-10:])
    features['m16_first_10_mean'] = np.mean(seq_m16[:10])
    
    # ========== TOTALCNT 기본 8개 ==========
    features['totalcnt_mean'] = np.mean(seq_totalcnt)
    features['totalcnt_std'] = np.std(seq_totalcnt)
    features['totalcnt_last_5_mean'] = np.mean(seq_totalcnt[-5:])
    features['totalcnt_max'] = np.max(seq_totalcnt)
    features['totalcnt_min'] = np.min(seq_totalcnt)
    features['totalcnt_slope'] = np.polyfit(np.arange(280), seq_totalcnt, 1)[0]
    features['totalcnt_last_10_mean'] = np.mean(seq_totalcnt[-10:])
    features['totalcnt_first_10_mean'] = np.mean(seq_totalcnt[:10])
    
    # ========== 비율 Feature (8개) ==========
    features['ratio_m14b_m10a'] = seq_m14b[-1] / (seq_m10a[-1] + 1)
    features['ratio_m14b_m16'] = seq_m14b[-1] / (seq_m16[-1] + 1)
    features['ratio_m10a_m16'] = seq_m10a[-1] / (seq_m16[-1] + 1)
    features['ratio_m14b_m10a_mean'] = np.mean(seq_m14b) / (np.mean(seq_m10a) + 1)
    features['ratio_m14b_m16_mean'] = np.mean(seq_m14b) / (np.mean(seq_m16) + 1)
    features['ratio_m14b_m10a_max'] = np.max(seq_m14b) / (np.max(seq_m10a) + 1)
    features['volatility_m14b'] = np.std(seq_m14b) / (np.mean(seq_m14b) + 1)
    features['volatility_totalcnt'] = np.std(seq_totalcnt) / (np.mean(seq_totalcnt) + 1)
    
    # ========== M14AM14B 임계값 카운트 (8개) ==========
    features['m14b_over_250'] = np.sum(seq_m14b > 250)
    features['m14b_over_300'] = np.sum(seq_m14b > 300)
    features['m14b_over_350'] = np.sum(seq_m14b > 350)
    features['m14b_over_400'] = np.sum(seq_m14b > 400)
    features['m14b_over_450'] = np.sum(seq_m14b > 450)
    features['m14b_over_300_last30'] = np.sum(seq_m14b[-30:] > 300)
    features['m14b_over_350_last30'] = np.sum(seq_m14b[-30:] > 350)
    features['m14b_over_400_last30'] = np.sum(seq_m14b[-30:] > 400)
    
    # ========== M14AM10A 임계값 카운트 (4개) ==========
    features['m10a_over_70'] = np.sum(seq_m10a > 70)
    features['m10a_over_80'] = np.sum(seq_m10a > 80)
    features['m10a_under_80'] = np.sum(seq_m10a < 80)
    features['m10a_under_70'] = np.sum(seq_m10a < 70)
    
    # ========== TOTALCNT 임계값 카운트 (8개) ==========
    features['totalcnt_over_1400'] = np.sum(seq_totalcnt >= 1400)
    features['totalcnt_over_1500'] = np.sum(seq_totalcnt >= 1500)
    features['totalcnt_over_1600'] = np.sum(seq_totalcnt >= 1600)
    features['totalcnt_over_1700'] = np.sum(seq_totalcnt >= 1700)
    features['totalcnt_over_1400_last30'] = np.sum(seq_totalcnt[-30:] >= 1400)
    features['totalcnt_over_1500_last30'] = np.sum(seq_totalcnt[-30:] >= 1500)
    features['totalcnt_over_1600_last30'] = np.sum(seq_totalcnt[-30:] >= 1600)
    features['totalcnt_over_1700_last30'] = np.sum(seq_totalcnt[-30:] >= 1700)
    
    # ========== 황금 패턴 (4개) ==========
    features['golden_pattern_300_80'] = 1 if (seq_m14b[-1] > 300 and seq_m10a[-1] < 80) else 0
    features['golden_pattern_350_80'] = 1 if (seq_m14b[-1] > 350 and seq_m10a[-1] < 80) else 0
    features['golden_pattern_400_70'] = 1 if (seq_m14b[-1] > 400 and seq_m10a[-1] < 70) else 0
    features['danger_zone'] = 1 if seq_totalcnt[-1] >= 1700 else 0
    
    # ========== 변화율/가속도 (8개) ==========
    features['m14b_change_rate'] = (seq_m14b[-1] - seq_m14b[-30]) / 30 if len(seq_m14b) >= 30 else 0
    features['totalcnt_change_rate'] = (seq_totalcnt[-1] - seq_totalcnt[-30]) / 30 if len(seq_totalcnt) >= 30 else 0
    
    recent_30_m14b = np.mean(seq_m14b[-30:])
    previous_30_m14b = np.mean(seq_m14b[-60:-30]) if len(seq_m14b) >= 60 else np.mean(seq_m14b[-30:])
    features['m14b_acceleration'] = recent_30_m14b - previous_30_m14b
    
    recent_30_totalcnt = np.mean(seq_totalcnt[-30:])
    previous_30_totalcnt = np.mean(seq_totalcnt[-60:-30]) if len(seq_totalcnt) >= 60 else np.mean(seq_totalcnt[-30:])
    features['totalcnt_acceleration'] = recent_30_totalcnt - previous_30_totalcnt
    
    features['m14b_range'] = np.max(seq_m14b) - np.min(seq_m14b)
    features['totalcnt_range'] = np.max(seq_totalcnt) - np.min(seq_totalcnt)
    features['m14b_recent_vs_mean'] = np.mean(seq_m14b[-30:]) / (np.mean(seq_m14b) + 1)
    features['totalcnt_recent_vs_mean'] = np.mean(seq_totalcnt[-30:]) / (np.mean(seq_totalcnt) + 1)
    
    # ========== 시간대별 통계 (8개) ==========
    q1 = seq_totalcnt[:70]
    q2 = seq_totalcnt[70:140]
    q3 = seq_totalcnt[140:210]
    q4 = seq_totalcnt[210:280]
    
    features['totalcnt_q1_mean'] = np.mean(q1)
    features['totalcnt_q2_mean'] = np.mean(q2)
    features['totalcnt_q3_mean'] = np.mean(q3)
    features['totalcnt_q4_mean'] = np.mean(q4)
    features['totalcnt_trend_q1_q2'] = np.mean(q2) - np.mean(q1)
    features['totalcnt_trend_q2_q3'] = np.mean(q3) - np.mean(q2)
    features['totalcnt_trend_q3_q4'] = np.mean(q4) - np.mean(q3)
    features['totalcnt_trend_overall'] = np.mean(q4) - np.mean(q1)
    
    # ========== 🔥 조기 경보 Feature (5개) ==========
    # 마지막 10분 분석
    last_10min = seq_totalcnt[-10:]
    
    features['last_10min_max'] = np.max(last_10min)
    features['last_10min_min'] = np.min(last_10min)
    features['last_10min_mean'] = np.mean(last_10min)
    features['last_10min_rise'] = last_10min[-1] - last_10min[0]  # 10분간 상승폭
    
    # 조기 경보: 1650 이상 + 20 이상 상승
    early_warning = (np.max(last_10min) >= 1650) and ((last_10min[-1] - last_10min[0]) > 20)
    features['early_warning_1650_rising'] = 1 if early_warning else 0
    
    return features

def get_status_info(value):
    """
    물류량에 따른 상태 정보 반환
    900-1599: 정상
    1600-1699: 주의
    1700+: 심각
    """
    if value < 900:
        return '🔵저조', 'blue'
    elif value < 1600:
        return '🟢정상', 'green'
    elif value < 1700:
        return '🟡주의', 'yellow'
    else:
        return '🔴심각', 'red'

def realtime_prediction():
    """실시간 예측 시스템 (조기 경보 포함)"""
    
    print("="*80)
    print("280분 → 10분 후 물류량 실시간 예측 시스템 (조기 경보 포함)")
    print("="*80)
    print("Feature: 85개 (조기 경보 5개 포함)")
    print("="*80)
    
    # 모델 로드 (여러 파일명 시도)
    model_files = [
        'xgboost_280to10_1year_augmented.pkl',
        'xgboost_280to10_enhanced_earlywarning.pkl',
        '280to10_enhanced.pkl'
    ]
    
    model = None
    model_file = None
    
    for mf in model_files:
        if os.path.exists(mf):
            try:
                with open(mf, 'rb') as f:
                    model = pickle.load(f)
                model_file = mf
                print(f"✅ 모델 로드 완료: {model_file}")
                break
            except Exception as e:
                print(f"⚠️ 모델 로드 실패 ({mf}): {e}")
    
    if model is None:
        print(f"❌ 모델 파일 없음!")
        print(f"시도한 파일:")
        for mf in model_files:
            print(f"  - {mf}")
        return None
    
    # 데이터 로드
    csv_files = [
        'data/222.csv',
        'data/20250807_DATA.csv',
        '222.csv'
    ]
    
    csv_file = None
    for cf in csv_files:
        if os.path.exists(cf):
            csv_file = cf
            break
    
    if csv_file is None:
        print(f"❌ CSV 파일 없음!")
        print(f"시도한 파일:")
        for cf in csv_files:
            print(f"  - {cf}")
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
            df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M')
            print("✅ CURRTIME 파싱 완료 (YYYYMMDDHHMM 형식)")
        except:
            try:
                df['CURRTIME'] = pd.to_datetime(df['CURRTIME'])
                print("✅ CURRTIME 자동 파싱 완료")
            except:
                print("⚠️ CURRTIME 변환 실패, 가상 시간 생성")
                base_time = datetime(2024, 1, 1, 0, 0)
                df['CURRTIME'] = [base_time + timedelta(minutes=i) for i in range(len(df))]
    else:
        print("⚠️ CURRTIME 없음, 가상 시간 생성")
        base_time = datetime(2024, 1, 1, 0, 0)
        df['CURRTIME'] = [base_time + timedelta(minutes=i) for i in range(len(df))]
    
    # 데이터 길이 확인
    if len(df) < 280:
        print(f"❌ 데이터 부족: 최소 280개 필요, 현재 {len(df)}개")
        return None
    
    results = []
    
    print(f"\n🔮 실시간 예측 시작...")
    print(f"총 예측 가능: {len(df) - 280 + 1:,}개")
    print(f"예측 방식: 280분 시퀀스 → 10분 후 예측")
    print(f"조기 경보: 1650+ & 20+ 상승 감지 시 예측값 2% 증가")
    print("-"*80)
    
    # 슬라이딩 윈도우 예측
    for i in range(280, len(df) + 1):
        # 280분 시퀀스 추출
        seq_m14b = df['M14AM14B'].iloc[i-280:i].values
        seq_m10a = df['M14AM10A'].iloc[i-280:i].values
        seq_m16 = df['M14AM16'].iloc[i-280:i].values
        seq_totalcnt = df['TOTALCNT'].iloc[i-280:i].values
        
        # 시간 정보
        current_time = df['CURRTIME'].iloc[i-1]
        prediction_time = current_time + timedelta(minutes=10)
        
        # 현재 상태
        current_totalcnt = seq_totalcnt[-1]
        current_m14b = seq_m14b[-1]
        current_m10a = seq_m10a[-1]
        
        # Feature 생성 (85개)
        features = create_single_prediction_features(seq_m14b, seq_m10a, seq_m16, seq_totalcnt)
        X_pred = pd.DataFrame([features])
        
        # 예측
        prediction = model.predict(X_pred)[0]
        
        # 🔥 조기 경보 감지
        last_10min = seq_totalcnt[-10:]
        early_warning_detected = (np.max(last_10min) >= 1650) and ((last_10min[-1] - last_10min[0]) > 20)
        
        # 🔥🔥 조기경보 발생 시 예측값 * 1.02 (2% 증가)
        if early_warning_detected:
            prediction = prediction * 1.02
        
        # 황금 패턴 감지
        golden_pattern = (current_m14b > 300 and current_m10a < 80)
        
        # 상태 판정
        current_status, current_color = get_status_info(current_totalcnt)
        pred_status, pred_color = get_status_info(prediction)
        
        # 상태 변화 감지
        status_change = ''
        if current_color != pred_color:
            if pred_color == 'red' and current_color != 'red':
                status_change = '⚠️상승경보'
            elif pred_color == 'yellow' and current_color == 'green':
                status_change = '⚠️주의전환'
            elif pred_color == 'green' and current_color in ['yellow', 'red']:
                status_change = '✅안정화'
        
        # 결과 저장
        results.append({
            '현재시간': current_time.strftime('%Y-%m-%d %H:%M'),
            '예측시점(10분후)': prediction_time.strftime('%Y-%m-%d %H:%M'),
            '현재물류량': int(current_totalcnt),
            '현재상태': current_status,
            '예측물류량': int(prediction),
            '예측상태': pred_status,
            '변화량': int(prediction - current_totalcnt),
            '상태변화': status_change,
            'M14AM14B': int(current_m14b),
            'M14AM10A': int(current_m10a),
            '황금패턴': '⭐' if golden_pattern else '',
            '🔥조기경보': '🔥' if early_warning_detected else '',
            '조기경보2%적용': 'O' if early_warning_detected else '',
            '마지막10분_MAX': int(np.max(last_10min)),
            '마지막10분_상승폭': int(last_10min[-1] - last_10min[0]),
            '시퀀스최대': int(np.max(seq_totalcnt)),
            '시퀀스최소': int(np.min(seq_totalcnt)),
            '시퀀스평균': int(np.mean(seq_totalcnt))
        })
        
        # 진행상황 (100개마다)
        if (i - 280) % 100 == 0:
            progress = (i - 280) / (len(df) - 280 + 1) * 100
            print(f"  진행중... {i-280}/{len(df)-280+1} ({progress:.1f}%)")
    
    # DataFrame 변환
    results_df = pd.DataFrame(results)
    
    # ===== 통계 분석 =====
    print("\n" + "="*80)
    print("📊 예측 통계")
    print("="*80)
    print(f"총 예측 수: {len(results_df):,}개")
    print(f"\n현재 상태 분포:")
    print(results_df['현재상태'].value_counts())
    print(f"\n예측 상태 분포:")
    print(results_df['예측상태'].value_counts())
    
    # 조기 경보 분석
    early_warning_cases = results_df[results_df['🔥조기경보'] == '🔥']
    if len(early_warning_cases) > 0:
        print(f"\n🔥 조기 경보 발생: {len(early_warning_cases):,}건")
        print(f"   평균 예측값: {early_warning_cases['예측물류량'].mean():.0f}")
        print(f"   평균 마지막10분_MAX: {early_warning_cases['마지막10분_MAX'].mean():.0f}")
        print(f"   평균 상승폭: {early_warning_cases['마지막10분_상승폭'].mean():.0f}")
    
    # 상태 변화 분석
    status_changes = results_df[results_df['상태변화'] != '']
    if len(status_changes) > 0:
        print(f"\n⚠️ 상태 변화 발생: {len(status_changes):,}건")
        print(status_changes['상태변화'].value_counts())
    
    # 심각 상태 분석
    critical_cases = results_df[results_df['예측상태'] == '🔴심각']
    if len(critical_cases) > 0:
        print(f"\n🔴 심각(1700+) 예측: {len(critical_cases):,}건")
        print(f"   평균 예측값: {critical_cases['예측물류량'].mean():.0f}")
        print(f"   최대 예측값: {critical_cases['예측물류량'].max():.0f}")
        
        # 조기 경보 & 심각 동시 발생
        early_and_critical = critical_cases[critical_cases['🔥조기경보'] == '🔥']
        if len(early_and_critical) > 0:
            print(f"   🔥 조기경보+심각: {len(early_and_critical):,}건 ({len(early_and_critical)/len(critical_cases)*100:.1f}%)")
    
    # 주의 상태 분석
    warning_cases = results_df[results_df['예측상태'] == '🟡주의']
    if len(warning_cases) > 0:
        print(f"\n🟡 주의(1600-1699) 예측: {len(warning_cases):,}건")
        print(f"   평균 예측값: {warning_cases['예측물류량'].mean():.0f}")
    
    # 황금 패턴 분석
    golden_cases = results_df[results_df['황금패턴'] == '⭐']
    if len(golden_cases) > 0:
        print(f"\n⭐ 황금 패턴 발생: {len(golden_cases):,}건")
        print(f"   평균 예측값: {golden_cases['예측물류량'].mean():.0f}")
    
    # 최근 10개 예측 출력
    print("\n" + "="*80)
    print("📋 최근 10개 예측 결과")
    print("="*80)
    print(results_df[['현재시간', '현재물류량', '현재상태', '예측물류량', '예측상태', '변화량', '🔥조기경보', '상태변화']].tail(10).to_string(index=False))
    
    # 조기 경보 발생 구간 (상위 20개)
    if len(early_warning_cases) > 0:
        print("\n" + "="*80)
        print("🔥 조기 경보 발생 구간 (상위 20개)")
        print("="*80)
        print(early_warning_cases[['현재시간', '마지막10분_MAX', '마지막10분_상승폭', '현재물류량', '예측물류량', '예측상태']].head(20).to_string(index=False))
    
    # 심각 예측 상위 10개
    if len(critical_cases) > 0:
        print("\n" + "="*80)
        print("🔴 심각 예측 상위 10개")
        print("="*80)
        top_critical = critical_cases.nlargest(10, '예측물류량')
        print(top_critical[['현재시간', '현재물류량', '예측물류량', '변화량', 'M14AM14B', 'M14AM10A', '🔥조기경보']].to_string(index=False))
    
    # 상승 경보 발생
    alerts = results_df[results_df['상태변화'].str.contains('상승경보', na=False)]
    if len(alerts) > 0:
        print("\n" + "="*80)
        print("⚠️ 상승 경보 발생 구간")
        print("="*80)
        print(alerts[['현재시간', '현재상태', '예측상태', '현재물류량', '예측물류량', '변화량', '🔥조기경보']].to_string(index=False))
    
    return results_df

def predict_latest_only():
    """
    가장 최근 280분 데이터로 10분 후 예측 (조기 경보 포함)
    """
    
    print("="*80)
    print("최신 데이터 예측 (조기 경보 포함)")
    print("="*80)
    
    # 모델 로드
    model_files = [
        'xgboost_280to10_1year_augmented.pkl',
        'xgboost_280to10_enhanced_earlywarning.pkl',
        '280to10_enhanced.pkl'
    ]
    
    model = None
    for mf in model_files:
        if os.path.exists(mf):
            try:
                with open(mf, 'rb') as f:
                    model = pickle.load(f)
                print(f"✅ 모델 로드 완료: {mf}")
                break
            except:
                pass
    
    if model is None:
        print(f"❌ 모델 파일 없음!")
        return None
    
    # 데이터 로드
    csv_files = [
        'data/20250807_DATA.csv',
        'data/222.csv',
        '222.csv'
    ]
    
    csv_file = None
    for cf in csv_files:
        if os.path.exists(cf):
            csv_file = cf
            break
    
    if csv_file is None:
        print(f"❌ CSV 파일 없음!")
        return None
    
    df = pd.read_csv(csv_file, on_bad_lines='skip')
    print(f"✅ 데이터 로드 완료: {len(df):,}개 행")
    
    # 필수 컬럼 확인
    required_cols = ['M14AM14B', 'M14AM10A', 'M14AM16', 'TOTALCNT']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ 필수 컬럼 누락: {missing_cols}")
        return None
    
    # 데이터 길이 확인
    if len(df) < 280:
        print(f"❌ 데이터 부족: 최소 280개 필요, 현재 {len(df)}개")
        return None
    
    # CURRTIME 처리
    if 'CURRTIME' in df.columns:
        try:
            df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M')
        except:
            try:
                df['CURRTIME'] = pd.to_datetime(df['CURRTIME'])
            except:
                base_time = datetime(2024, 1, 1, 0, 0)
                df['CURRTIME'] = [base_time + timedelta(minutes=i) for i in range(len(df))]
    else:
        base_time = datetime(2024, 1, 1, 0, 0)
        df['CURRTIME'] = [base_time + timedelta(minutes=i) for i in range(len(df))]
    
    # 최근 280분 데이터 추출
    seq_m14b = df['M14AM14B'].iloc[-280:].values
    seq_m10a = df['M14AM10A'].iloc[-280:].values
    seq_m16 = df['M14AM16'].iloc[-280:].values
    seq_totalcnt = df['TOTALCNT'].iloc[-280:].values
    
    # 현재 시간 정보
    current_time = df['CURRTIME'].iloc[-1]
    prediction_time = current_time + timedelta(minutes=10)
    
    # 현재 상태
    current_totalcnt = seq_totalcnt[-1]
    current_m14b = seq_m14b[-1]
    current_m10a = seq_m10a[-1]
    
    # Feature 생성 (85개)
    features = create_single_prediction_features(seq_m14b, seq_m10a, seq_m16, seq_totalcnt)
    X_pred = pd.DataFrame([features])
    
    # 예측
    prediction = model.predict(X_pred)[0]
    
    # 🔥 조기 경보 감지
    last_10min = seq_totalcnt[-10:]
    early_warning_detected = (np.max(last_10min) >= 1650) and ((last_10min[-1] - last_10min[0]) > 20)
    
    # 🔥🔥 조기경보 발생 시 예측값 * 1.02 (2% 증가)
    if early_warning_detected:
        prediction = prediction * 1.02
    
    # 황금 패턴 감지
    golden_pattern = (current_m14b > 300 and current_m10a < 80)
    
    # 상태 판정
    current_status, current_color = get_status_info(current_totalcnt)
    pred_status, pred_color = get_status_info(prediction)
    
    # 간단한 결과만 출력
    if early_warning_detected:
        print("🔥 조기 경보: 발생!")
    
    simple_dict = {
        '예측물류량': int(prediction),
        '예측상태': pred_status
    }
    
    print(simple_dict)
    
    return simple_dict

if __name__ == '__main__':
    # 전체 데이터 실시간 예측
    print("\n🚀 실시간 예측 시스템 시작 (조기 경보 포함)...\n")
    results = realtime_prediction()
    
    if results is not None:
        print(f"\n✅ 예측 완료! 총 {len(results):,}개 예측 생성")