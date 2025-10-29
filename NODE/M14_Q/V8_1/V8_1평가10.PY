# -*- coding: utf-8 -*-
"""
280분 시퀀스 → 10분 후 예측 평가 (1년치 데이터)
조기 경보 + 1550→1700 패턴 Feature 포함 (96개)
"""

import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
import os
import gc

def create_single_prediction_features(seq_m14b, seq_m10a, seq_m16, seq_totalcnt, df=None, idx=None):
    """하나의 280분 시퀀스로부터 Feature 생성 (96개: 조기 경보 + 1550→1700 패턴)"""
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
    
    # ========== 🎯 1550→1700 패턴 Feature (11개) ==========
    # 마지막 5분에서 TOTALCNT 체크
    last_5min_totalcnt = seq_totalcnt[-5:]
    current_totalcnt_max = np.max(last_5min_totalcnt)  # 마지막 5분 최대값
    current_totalcnt_mean = np.mean(last_5min_totalcnt)  # 마지막 5분 평균
    
    # 상승 추세 체크 (마지막 5분 평균 vs 이전 5분 평균)
    prev_5min_totalcnt = seq_totalcnt[-10:-5]
    prev_5min_mean = np.mean(prev_5min_totalcnt)
    is_rising = current_totalcnt_mean > prev_5min_mean
    rise_amount = current_totalcnt_mean - prev_5min_mean
    
    features['current_totalcnt'] = current_totalcnt_max  # 마지막 5분 최대값
    features['totalcnt_rise_5min'] = rise_amount  # 5분 상승폭
    
    # 🔥 1550+ 조건: TOTALCNT 1550+ AND 상승 추세 AND M14AM16 조건
    is_1550_plus_with_conditions = (
        (current_totalcnt_max >= 1550) and 
        is_rising and 
        (seq_m16[-1] >= 217)  # M14AM16이 217 이상
    )
    features['is_1550_plus'] = 1 if is_1550_plus_with_conditions else 0
    
    # 🔥 1600+ 조건: TOTALCNT 1600+ AND 상승 추세 AND 강한 M14AM16 조건
    is_1600_plus_with_conditions = (
        (current_totalcnt_max >= 1600) and 
        is_rising and 
        (seq_m16[-1] >= 272)  # M14AM16이 272 이상 (75% 분위수)
    )
    features['is_1600_plus'] = 1 if is_1600_plus_with_conditions else 0
    
    # M14AM16 핵심 변수 (46.2% 차이)
    features['m16_last'] = seq_m16[-1]
    features['m16_last_vs_mean'] = seq_m16[-1] / (np.mean(seq_m16) + 1)
    
    # M14AM16SUM 계산 (28.7% 차이)
    if df is not None and 'M14AM16SUM' in df.columns and idx is not None:
        m16sum_current = df['M14AM16SUM'].iloc[idx-1]
    else:
        # M14AM16SUM이 없으면 계산
        m16sum_current = np.sum(seq_m16[-10:])  # 최근 10분 합
    features['m16sum_current'] = m16sum_current
    
    # 1700+ 예측 규칙 (LEVEL 2 기준)
    rule_level2 = (seq_m16[-1] >= 272) and (m16sum_current >= 494) and (seq_m14b[-1] <= 194)
    features['pattern_1700_level2'] = 1 if rule_level2 else 0
    
    # 1700+ 예측 규칙 (LEVEL 3 긴급)
    rule_level3 = (seq_m16[-1] >= 641) and (m16sum_current >= 853)
    features['pattern_1700_level3'] = 1 if rule_level3 else 0
    
    # M14BM14A 비율 (15.3% 차이)
    if df is not None and 'M14BM14A' in df.columns and idx is not None:
        m14bm14a_current = df['M14BM14A'].iloc[idx-1]
    else:
        m14bm14a_current = 0
    features['m14bm14a_current'] = m14bm14a_current
    
    # 복합 패턴: 1550+ 구간에서 위험 신호 (마지막 5분 기준)
    high_base_danger = (
        (current_totalcnt_max >= 1550) and 
        (seq_m16[-1] >= 217) and 
        (m16sum_current >= 430)
    )
    features['pattern_1550_danger'] = 1 if high_base_danger else 0
    
    return features

def evaluate_all_predictions():
    """전체 데이터를 슬라이딩 윈도우로 평가 - 10분 후 (1년치 데이터)"""
    
    print("="*80)
    print("🚀 280분 시퀀스 → 10분 후 예측 평가 (1년치 데이터)")
    print("="*80)
    print("Feature: 96개 (조기 경보 5 + 1550→1700 패턴 11)")
    print("="*80)
    
    # 모델 로드 (여러 파일명 시도)
    model_files = [
        'xgboost_280to10_1year_augmented.pkl',
        'xgboost_280to10_enhanced_earlywarning.pkl',
        '/mnt/user-data/outputs/xgboost_280to10_1year_augmented.pkl'
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
        print("\n먼저 FINAL_xgb_280_COMPLETE.py를 실행하여 모델을 학습하세요.")
        return None
    
    # 데이터 로드 (여러 경로 시도)
    csv_files = [
        '/mnt/project/V6_6결과.CSV',
        'V6_6결과.CSV',
        '/mnt/project/uu.csv',
        'uu.csv'
    ]
    
    df = None
    csv_file = None
    
    for cf in csv_files:
        if os.path.exists(cf):
            try:
                print(f"데이터 로딩 중: {cf}...")
                df = pd.read_csv(cf, on_bad_lines='skip')
                csv_file = cf
                print(f"✅ 데이터 로드 완료: {len(df):,}개 행")
                break
            except Exception as e:
                print(f"⚠️ 데이터 로드 실패 ({cf}): {e}")
    
    if df is None:
        print(f"❌ CSV 파일 없음!")
        print(f"시도한 파일:")
        for cf in csv_files:
            print(f"  - {cf}")
        return None
    
    # 메모리 사용량 확인
    memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"메모리 사용량: {memory_mb:.1f} MB")
    
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
            print("✅ CURRTIME 파싱 완료")
        except:
            print("⚠️ CURRTIME 변환 실패, 가상 시간 생성")
            base_time = datetime(2024, 1, 1, 0, 0)
            df['CURRTIME'] = [base_time + timedelta(minutes=i) for i in range(len(df))]
    else:
        print("⚠️ CURRTIME 없음, 가상 시간 생성")
        base_time = datetime(2024, 1, 1, 0, 0)
        df['CURRTIME'] = [base_time + timedelta(minutes=i) for i in range(len(df))]
    
    results = []
    
    total_predictions = len(df) - 280 - 10
    print(f"\n🔄 슬라이딩 윈도우 평가 시작...")
    print(f"총 예측 수: {total_predictions:,}개")
    print(f"예상 소요 시간: 약 {total_predictions / 10000:.1f}분")
    
    # 슬라이딩 윈도우
    for i in range(280, len(df) - 10):
        # 280분 시퀀스
        seq_m14b = df['M14AM14B'].iloc[i-280:i].values
        seq_m10a = df['M14AM10A'].iloc[i-280:i].values
        seq_m16 = df['M14AM16'].iloc[i-280:i].values
        seq_totalcnt = df['TOTALCNT'].iloc[i-280:i].values
        
        # 시간 정보
        current_time = df['CURRTIME'].iloc[i-1]  # 시퀀스 마지막 (현재)
        seq_start_time = df['CURRTIME'].iloc[i-280]
        prediction_time = current_time + timedelta(minutes=10)
        actual_time = df['CURRTIME'].iloc[i+9]  # ✅ 10분 후 (i+9)
        
        # ✅ 실제값 (10분 후)
        actual_value = df['TOTALCNT'].iloc[i+9]  # i는 1분 후, i+9는 10분 후
        
        # Feature 생성 (96개: 조기 경보 + 1550→1700 패턴)
        features = create_single_prediction_features(seq_m14b, seq_m10a, seq_m16, seq_totalcnt, df, i)
        X_pred = pd.DataFrame([features])
        
        # 예측
        prediction = model.predict(X_pred)[0]
        
        # 황금 패턴 감지
        golden_pattern = (seq_m14b[-1] > 300 and seq_m10a[-1] < 80)
        
        # 위험 구간 감지
        danger_in_seq = np.sum(seq_totalcnt >= 1700) > 0
        
        # 🔥 조기 경보 감지
        last_10min = seq_totalcnt[-10:]
        early_warning_detected = (np.max(last_10min) >= 1650) and ((last_10min[-1] - last_10min[0]) > 20)
        
        # 🎯 1550→1700 패턴 감지
        pattern_1550_detected = features['is_1550_plus'] == 1
        pattern_1700_level2 = features['pattern_1700_level2'] == 1
        pattern_1700_level3 = features['pattern_1700_level3'] == 1
        
        # 🔥🔥 조기경보 발생 시 예측값 * 1.02 (2% 증가)
        if early_warning_detected:
            prediction = prediction * 1.02
        
        # 결과 저장
        results.append({
            '시퀀스시작': seq_start_time.strftime('%Y-%m-%d %H:%M'),
            '현재시간': current_time.strftime('%Y-%m-%d %H:%M'),
            '현재TOTALCNT': round(seq_totalcnt[-1], 2),  # ✅ 현재 시점의 실제 TOTALCNT
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
            '마지막10분_MAX': round(np.max(last_10min), 2),
            '마지막10분_상승폭': round(last_10min[-1] - last_10min[0], 2),
            '마지막5분_MAX': round(features['current_totalcnt'], 2),
            '5분상승폭': round(features['totalcnt_rise_5min'], 2),
            '황금패턴': 'O' if golden_pattern else '',
            '시퀀스위험': 'O' if danger_in_seq else '',
            '조기경보': 'O' if early_warning_detected else '',
            '조기경보2%적용': 'O' if early_warning_detected else '',
            '1550+패턴': 'O' if pattern_1550_detected else '',
            '1700_LEVEL2': 'O' if pattern_1700_level2 else '',
            '1700_LEVEL3': 'O' if pattern_1700_level3 else '',
            '실제위험(1700+)': 'O' if actual_value >= 1700 else '',
            '예측위험(1650+)': 'O' if prediction >= 1650 else ''
        })
        
        # 진행상황 출력 (1년치 데이터용 - 50000개마다)
        if (i - 280) % 50000 == 0 and i > 280:
            progress = (i - 280) / total_predictions * 100
            elapsed = (i - 280) / 10000  # 대략적인 경과 시간
            print(f"  진행중... {i-280:,}/{total_predictions:,} ({progress:.1f}%) - 약 {elapsed:.1f}분 경과")
            
            # 메모리 정리
            gc.collect()
    
    print(f"✅ 평가 완료!")
    
    # DataFrame 변환
    results_df = pd.DataFrame(results)
    
    # CSV 저장
    output_file = 'evaluation_280to10_1year_with_1550pattern.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✅ 결과 저장 완료: {output_file}")
    
    # 메모리 정리
    del df
    gc.collect()
    
    # ===== 통계 분석 =====
    print("\n" + "="*80)
    print("📊 평가 통계 (280분 → 10분 후)")
    print("="*80)
    print(f"총 예측 수: {len(results_df):,}개")
    print(f"평균 절대 오차(MAE): {results_df['절대오차'].mean():.2f}")
    print(f"평균 오차율: {results_df['오차율(%)'].mean():.2f}%")
    print(f"최대 절대 오차: {results_df['절대오차'].max():.2f}")
    print(f"최소 절대 오차: {results_df['절대오차'].min():.2f}")
    
    print(f"\n황금 패턴 발생: {results_df['황금패턴'].value_counts().get('O', 0):,}개")
    print(f"시퀀스 위험 구간: {results_df['시퀀스위험'].value_counts().get('O', 0):,}개")
    print(f"🔥 조기 경보 발생: {results_df['조기경보'].value_counts().get('O', 0):,}개")
    print(f"🎯 1550+ 패턴 발생: {results_df['1550+패턴'].value_counts().get('O', 0):,}개")
    print(f"🎯 1700 LEVEL2 발생: {results_df['1700_LEVEL2'].value_counts().get('O', 0):,}개")
    print(f"🎯 1700 LEVEL3 발생: {results_df['1700_LEVEL3'].value_counts().get('O', 0):,}개")
    
    # 위험 구간 분석
    actual_danger = results_df['실제위험(1700+)'] == 'O'
    pred_danger = results_df['예측위험(1650+)'] == 'O'
    early_warning = results_df['조기경보'] == 'O'
    pattern_1550 = results_df['1550+패턴'] == 'O'
    pattern_level2 = results_df['1700_LEVEL2'] == 'O'
    pattern_level3 = results_df['1700_LEVEL3'] == 'O'
    
    actual_danger_count = actual_danger.sum()
    pred_danger_count = pred_danger.sum()
    danger_detected = (actual_danger & pred_danger).sum()
    early_warning_count = early_warning.sum()
    pattern_1550_count = pattern_1550.sum()
    pattern_level2_count = pattern_level2.sum()
    pattern_level3_count = pattern_level3.sum()
    
    # 조기 경보의 1700+ 예측력
    early_to_danger = (early_warning & actual_danger).sum()
    
    # 1550+ 패턴의 1700+ 예측력
    pattern_1550_to_danger = (pattern_1550 & actual_danger).sum()
    
    # LEVEL2의 1700+ 예측력
    level2_to_danger = (pattern_level2 & actual_danger).sum()
    
    # LEVEL3의 1700+ 예측력
    level3_to_danger = (pattern_level3 & actual_danger).sum()
    
    print(f"\n실제 위험(1700+): {actual_danger_count:,}개")
    print(f"예측 위험(1650+): {pred_danger_count:,}개")
    print(f"위험 감지 성공: {danger_detected:,}개")
    if actual_danger_count > 0:
        print(f"위험 감지율: {danger_detected/actual_danger_count*100:.1f}%")
    
    if early_warning_count > 0:
        print(f"\n🔥 조기 경보 → 실제 1700+ 발생: {early_to_danger:,}개 ({early_to_danger/early_warning_count*100:.1f}%)")
    
    if pattern_1550_count > 0:
        print(f"🎯 1550+ 패턴 → 실제 1700+ 발생: {pattern_1550_to_danger:,}개 ({pattern_1550_to_danger/pattern_1550_count*100:.1f}%)")
    
    if pattern_level2_count > 0:
        print(f"🎯 LEVEL2 패턴 → 실제 1700+ 발생: {level2_to_danger:,}개 ({level2_to_danger/pattern_level2_count*100:.1f}%)")
    
    if pattern_level3_count > 0:
        print(f"🎯 LEVEL3 패턴 → 실제 1700+ 발생: {level3_to_danger:,}개 ({level3_to_danger/pattern_level3_count*100:.1f}%)")
    
    # M14AM14B 450+ 구간 분석
    high_m14b_mask = results_df['M14AM14B'] >= 450
    high_m14b_count = high_m14b_mask.sum()
    if high_m14b_count > 0:
        print(f"\n🔥 M14AM14B 450+ 구간: {high_m14b_count:,}개")
        high_m14b_mae = results_df[high_m14b_mask]['절대오차'].mean()
        print(f"   평균 MAE: {high_m14b_mae:.2f}")
        high_m14b_danger = (high_m14b_mask & actual_danger).sum()
        print(f"   1700+ 발생: {high_m14b_danger:,}개 ({high_m14b_danger/high_m14b_count*100:.1f}%)")
    
    # 오차 상위 10개
    print("\n" + "="*80)
    print("오차 상위 10개 구간")
    print("="*80)
    top_errors = results_df.nlargest(10, '절대오차')
    print(top_errors[['현재시간', '현재TOTALCNT', '실제값', '예측값', '절대오차', 'M14AM14B', '1550+패턴', '조기경보']].to_string(index=False))
    
    # ✅ 10칸 내려서 비교 검증 (샘플 10개)
    print("\n" + "="*80)
    print("🔍 10칸 내려서 예측 검증 (샘플 10개)")
    print("="*80)
    print("Line 1의 예측값 vs Line 11의 현재TOTALCNT 비교")
    print("-"*80)
    
    for idx in [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]:
        if idx + 10 < len(results_df):
            line1_time = results_df.iloc[idx]['현재시간']
            line1_pred = results_df.iloc[idx]['예측값']
            line11_time = results_df.iloc[idx+10]['현재시간']
            line11_current = results_df.iloc[idx+10]['현재TOTALCNT']
            diff = abs(line11_current - line1_pred)
            
            print(f"Line {idx+1:4d} (예측): {line1_time} → 예측값: {line1_pred:7.2f}")
            print(f"Line {idx+11:4d} (실제): {line11_time} → 현재값: {line11_current:7.2f}")
            print(f"           차이: {diff:.2f}")
            print("-"*40)
    
    # 1550+ 패턴 발생 구간 분석
    if pattern_1550_count > 0:
        print("\n" + "="*80)
        print("🎯 1550+ 패턴 발생 구간 (샘플 20개)")
        print("="*80)
        pattern_cases = results_df[results_df['1550+패턴'] == 'O']
        print(f"총 {len(pattern_cases):,}개")
        print(pattern_cases[['현재시간', '마지막5분_MAX', '5분상승폭', 'M14AM16', '실제값', '예측값', '실제위험(1700+)']].head(20).to_string(index=False))
    
    # 조기 경보 발생 구간 분석
    if early_warning_count > 0:
        print("\n" + "="*80)
        print("🔥 조기 경보 발생 구간 (샘플 20개)")
        print("="*80)
        early_cases = results_df[results_df['조기경보'] == 'O']
        print(f"총 {len(early_cases):,}개")
        print(early_cases[['현재시간', '마지막10분_MAX', '마지막10분_상승폭', '실제값', '예측값', '실제위험(1700+)']].head(20).to_string(index=False))
    
    # 최종 요약
    print("\n" + "="*80)
    print("✅ 최종 평가 요약")
    print("="*80)
    print(f"1. 전체 성능:")
    print(f"   - MAE: {results_df['절대오차'].mean():.2f}")
    print(f"   - 위험 감지율: {danger_detected/actual_danger_count*100:.1f}% ({danger_detected:,}/{actual_danger_count:,})")
    
    if high_m14b_count > 0:
        print(f"\n2. M14AM14B 450+ 구간:")
        print(f"   - 발생 횟수: {high_m14b_count:,}개")
        print(f"   - MAE: {high_m14b_mae:.2f}")
        print(f"   - 1700+ 비율: {high_m14b_danger/high_m14b_count*100:.1f}%")
    
    if early_warning_count > 0:
        print(f"\n3. 조기 경보:")
        print(f"   - 발생 횟수: {early_warning_count:,}개")
        print(f"   - 정확도: {early_to_danger/early_warning_count*100:.1f}% (1700+ 예측)")
    
    if pattern_1550_count > 0:
        print(f"\n4. 1550→1700 패턴:")
        print(f"   - 발생 횟수: {pattern_1550_count:,}개")
        print(f"   - 정확도: {pattern_1550_to_danger/pattern_1550_count*100:.1f}% (1700+ 예측)")
        
        if pattern_level2_count > 0:
            print(f"   - LEVEL2: {pattern_level2_count:,}개 ({level2_to_danger/pattern_level2_count*100:.1f}% 정확도)")
        
        if pattern_level3_count > 0:
            print(f"   - LEVEL3: {pattern_level3_count:,}개 ({level3_to_danger/pattern_level3_count*100:.1f}% 정확도)")
    
    print(f"\n5. 저장 파일:")
    print(f"   - {output_file}")
    print("="*80)
    
    return results_df

if __name__ == '__main__':
    print("\n🚀 280분 시퀀스 → 10분 후 예측 평가 시작 (1년치 데이터)...\n")
    results = evaluate_all_predictions()
    
    if results is not None:
        print(f"\n✅ 평가 완료! 총 {len(results):,}개 예측 생성")
        print(f"✅ 결과 파일: evaluation_280to10_1year_with_1550pattern.csv")