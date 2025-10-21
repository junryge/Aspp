# -*- coding: utf-8 -*-
"""
V6 평가 전용 코드 + 급증 위험도 기반 보정
전체 데이터를 슬라이딩 윈도우로 평가 + 적응형 보정 적용
"""

import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta


def calculate_surge_risk(features):
    """
    급증 위험도 및 예상 급증량 계산
    
    Returns:
        risk_level: 위험도 (HIGH/MEDIUM/LOW)
        surge_prob: 급증 확률 (0~1)
        expected_surge: 예상 급증량
    """
    # 주요 지표 추출
    cmd_last = features.get('M16A_3F_CMD_last_value', 999)
    cmd_mean = features.get('M16A_3F_CMD_mean', 999)
    storage_last = features.get('M16A_3F_STORAGE_UTIL_last_value', 0)
    storage_mean = features.get('M16A_3F_STORAGE_UTIL_mean', 0)
    
    # CMD 변화량 계산
    cmd_slope = features.get('M16A_3F_CMD_slope', 0)
    
    # 위험도 판정 (분석 결과 기반)
    if cmd_last < 180:
        # CMD가 매우 낮음 → 큰 급증 가능성
        risk_level = "HIGH"
        surge_prob = 0.70
        expected_surge = 60
    elif storage_last >= 30 and storage_last <= 40:
        # Storage Util 위험 구간
        risk_level = "HIGH"
        surge_prob = 0.45
        expected_surge = 50
    elif cmd_last >= 260 and cmd_last <= 280:
        # CMD 높은 구간에서도 급증 가능
        risk_level = "HIGH"
        surge_prob = 0.67
        expected_surge = 60
    elif cmd_last < 220:
        # CMD 중간 구간
        risk_level = "MEDIUM"
        surge_prob = 0.30
        expected_surge = 35
    else:
        # 일반 상황
        risk_level = "LOW"
        surge_prob = 0.05
        expected_surge = 20
    
    # 추가 보정 요소
    # 1. CMD 급감 패턴 (slope가 음수이고 큼)
    if cmd_slope < -5:
        surge_prob *= 1.2  # 20% 증가
        expected_surge *= 1.1
    
    # 2. Storage Util이 증가 추세
    storage_slope = features.get('M16A_3F_STORAGE_UTIL_slope', 0)
    if storage_slope > 2:
        surge_prob *= 1.1  # 10% 증가
    
    # 확률 상한 조정
    surge_prob = min(surge_prob, 0.85)
    
    return risk_level, surge_prob, expected_surge

def apply_adaptive_correction(prediction, features, seq_target):
    """
    적응형 예측값 보정
    
    Args:
        prediction: 원래 모델 예측값
        features: 특성 딕셔너리
        seq_target: 30개 시퀀스 타겟값
    
    Returns:
        보정된 예측값, 위험도, 급증 확률, 보정량
    """
    # 위험도 계산
    risk_level, surge_prob, expected_surge = calculate_surge_risk(features)
    
    # 현재 값 (시퀀스 마지막)
    current_value = seq_target[-1]
    
    # 시퀀스 통계
    seq_mean = np.mean(seq_target)
    seq_std = np.std(seq_target)
    seq_max = np.max(seq_target)
    recent_mean = np.mean(seq_target[-5:])
    
    # 보정 전략
    correction = 0
    
    # 1. 기본 보정: 위험도에 따른 급증 반영
    if current_value < 300:  # 현재 300 미만일 때만
        if risk_level == "HIGH":
            # 높은 위험도: 급증 가능성 적극 반영
            if prediction < 300:
                # 예측값이 300 미만이면 급증 가능성 반영
                potential_value = current_value + expected_surge * surge_prob
                
                # 300 도달 가능성 체크
                if potential_value >= 300:
                    # 300 넘을 가능성이 높으면 보정
                    correction = (potential_value - prediction) * 0.7  # 70% 반영
                else:
                    # 300 미달이어도 상향 보정
                    correction = expected_surge * surge_prob * 0.5
                    
        elif risk_level == "MEDIUM":
            # 중간 위험도: 보수적 보정
            if prediction < 300 and surge_prob > 0.25:
                correction = expected_surge * surge_prob * 0.3
                
        else:  # LOW
            # 낮은 위험도: 최소 보정
            correction = expected_surge * surge_prob * 0.1
    
    # 2. 추가 보정: 시퀀스 패턴 반영
    # 최근 급증 추세
    recent_increase = recent_mean - seq_mean
    if recent_increase > 10:  # 최근 급증 중
        correction += recent_increase * 0.2
    
    # 시퀀스 내 300 이상 있었던 경우
    if seq_max >= 300:
        correction += 5  # 추가 상향
    
    # 3. 보정값 제한
    corrected_prediction = prediction + correction
    
    # 상한/하한 제한
    corrected_prediction = max(corrected_prediction, current_value - 50)  # 급락 제한
    corrected_prediction = min(corrected_prediction, current_value + 100)  # 급등 제한
    
    # 현실적 범위 제한
    if current_value < 250 and corrected_prediction > 350:
        corrected_prediction = 320  # 과도한 급증 제한
    
    return corrected_prediction, risk_level, surge_prob, correction


def evaluate_all_predictions_with_correction():
    """전체 데이터를 슬라이딩 윈도우로 평가 + 급증 위험도 보정"""
   
    # V4 Ultimate 필수 컬럼 정의
    # 핵심 12개 컬럼만 사용 (물리적으로 중요한 것만)
    FEATURE_COLS = {
        'storage': ['M16A_3F_STORAGE_UTIL'],  # 극단값 핵심 지표
        'cmd': ['M16A_3F_CMD', 'M16A_6F_TO_HUB_CMD'],  # 주요 CMD
        'inflow': ['M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2', 'M14A_3F_TO_HUB_JOB2'],  # 주요 유입
        'outflow': ['M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB', 'M16A_3F_TO_M14A_3F_JOB'],  # 주요 유출
        'maxcapa': ['M16A_6F_LFT_MAXCAPA', 'M16A_2F_LFT_MAXCAPA']  # 용량 제한
    }
     
    # 모델 로드
    try:
        with open('xgboost_model_30min_10min.pkl', 'rb') as f:
            model = pickle.load(f)
        print("✅ 모델 로드 완료: xgboost_model_30min_10min.pkl")
    except Exception as e:
        print(f"❌ 모델 파일 없음: {e}")
        return None
   
    # 데이터 로드 (인코딩 자동 감지)
    try:
        df = pd.read_csv('HUB0905101512.CSV', on_bad_lines='skip', encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv('HUB0905101512.CSV', on_bad_lines='skip', encoding='cp949')
            print("⚠️ 인코딩: CP949로 파일 로드")
        except:
            df = pd.read_csv('HUB0905101512.CSV', on_bad_lines='skip', encoding='euc-kr')
            print("⚠️ 인코딩: EUC-KR로 파일 로드")
    print(f"✅ 데이터 로드 완료: {len(df)}개 행")
   
    TARGET_COL = 'CURRENT_M16A_3F_JOB_2'
    
    # 사용 가능한 컬럼 확인
    print(f"\n사용 가능한 컬럼 확인:")
    all_feature_cols = []
    for group_name, cols in FEATURE_COLS.items():
        available = [col for col in cols if col in df.columns]
        all_feature_cols.extend(available)
        print(f"  - {group_name}: {len(available)}/{len(cols)}개")
   
    # STAT_DT 처리
    if 'STAT_DT' in df.columns:
        try:
            df['STAT_DT'] = pd.to_datetime(df['STAT_DT'].astype(str), format='%Y%m%d%H%M')
        except:
            print("⚠️ STAT_DT 변환 실패, 가상 시간 생성")
            base_time = datetime(2024, 1, 1, 0, 0)
            df['STAT_DT'] = [base_time + timedelta(minutes=i) for i in range(len(df))]
   
    results = []
   
    print("\n슬라이딩 윈도우 평가 시작...")
    # 슬라이딩 윈도우: 30개 시퀀스 → 10분 후 예측
    # i=30부터 시작 (0~29로 첫 예측, 실제값은 30번째)
    for i in range(30, len(df)):
        # 과거 30개 데이터
        seq_data = df.iloc[i-30:i].copy()
        seq_target = seq_data[TARGET_COL].values
       
        # 현재 시점 (시퀀스 마지막)
        current_time = seq_data['STAT_DT'].iloc[-1]
        current_value = seq_target[-1]
       
        # 예측 시점 (10분 후)
        prediction_time = current_time + timedelta(minutes=10)
       
        # 실제값 (i번째 행)
        actual_value = df.iloc[i][TARGET_COL]
        actual_time = df.iloc[i]['STAT_DT']
       
        # Feature 생성 (타겟 컬럼)
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
        
        # 각 컬럼 그룹별 특성 추가
        for group_name, cols in FEATURE_COLS.items():
            for col in cols:
                if col in df.columns:
                    col_seq = seq_data[col].values
                    
                    # 기본 통계
                    features[f'{col}_mean'] = np.mean(col_seq)
                    features[f'{col}_std'] = np.std(col_seq)
                    features[f'{col}_max'] = np.max(col_seq)
                    features[f'{col}_min'] = np.min(col_seq)
                    
                    # 최근 특성
                    features[f'{col}_last_5_mean'] = np.mean(col_seq[-5:])
                    features[f'{col}_last_10_mean'] = np.mean(col_seq[-10:])
                    
                    # 추세
                    features[f'{col}_slope'] = np.polyfit(np.arange(30), col_seq, 1)[0]
                    
                    # 구간별 평균
                    features[f'{col}_first_10_mean'] = np.mean(col_seq[:10])
                    features[f'{col}_mid_10_mean'] = np.mean(col_seq[10:20])
                    features[f'{col}_last_value'] = col_seq[-1]
        
        # 유입-유출 차이 (Net Flow)
        inflow_sum = 0
        outflow_sum = 0
        for col in FEATURE_COLS['inflow']:
            if col in df.columns:
                inflow_sum += df[col].iloc[i-1]
        for col in FEATURE_COLS['outflow']:
            if col in df.columns:
                outflow_sum += df[col].iloc[i-1]
        features['net_flow'] = inflow_sum - outflow_sum
        
        # CMD 총합
        cmd_sum = 0
        for col in FEATURE_COLS['cmd']:
            if col in df.columns:
                cmd_sum += df[col].iloc[i-1]
        features['total_cmd'] = cmd_sum
       
        X_pred = pd.DataFrame([features])
       
        # 기본 예측
        prediction_original = model.predict(X_pred)[0]
        
        # 보정된 예측
        prediction_corrected, risk_level, surge_prob, correction_amount = \
            apply_adaptive_correction(prediction_original, features, seq_target)
       
        # 300 이상 점프 감지 (시퀀스 내)
        jump_detected = np.any(seq_target >= 300)
        
        # 급증 여부 판정
        is_surge = (current_value < 300 and actual_value >= 300)
       
        # 결과 저장
        results.append({
            '현재시간': current_time.strftime('%Y-%m-%d %H:%M'),
            '예측시점': prediction_time.strftime('%Y-%m-%d %H:%M'),
            '실제시점': actual_time.strftime('%Y-%m-%d %H:%M'),
            '현재값': round(current_value, 2),
            '실제값': round(actual_value, 2),
            '예측값': round(prediction_original, 2),
            '예측값_보정후': round(prediction_corrected, 2),
            '위험도': risk_level,
            '급증확률': f"{surge_prob*100:.0f}%",
            '보정량': round(correction_amount, 2),
            '오차': round(actual_value - prediction_original, 2),
            '오차_보정후': round(actual_value - prediction_corrected, 2),
            '오차율(%)': round(abs(actual_value - prediction_original) / max(actual_value, 1) * 100, 2),
            '오차율_보정후(%)': round(abs(actual_value - prediction_corrected) / max(actual_value, 1) * 100, 2),
            '시퀀스MAX': round(np.max(seq_target), 2),
            '시퀀스MIN': round(np.min(seq_target), 2),
            '시퀀스평균': round(np.mean(seq_target), 2),
            '시퀀스증가': round(seq_target[-1] - seq_target[0], 2),
            '300이상점프': '🔴' if jump_detected else '',
            '300급증': '✅' if is_surge else '',
            '급증예측성공_보정전': '⭕' if (is_surge and prediction_original >= 300) else '',
            '급증예측성공_보정후': '✅' if (is_surge and prediction_corrected >= 300) else '',
            '실제값상태': '🔴극단' if actual_value >= 300 else ('🟡주의' if actual_value >= 280 else '🟢정상'),
            '예측상태_보정전': '🔴극단' if prediction_original >= 300 else ('🟡주의' if prediction_original >= 280 else '🟢정상'),
            '예측상태_보정후': '🔴극단' if prediction_corrected >= 300 else ('🟡주의' if prediction_corrected >= 280 else '🟢정상'),
            'M16A_3F_CMD': round(features.get('M16A_3F_CMD_last_value', 0), 2),
            'M16A_3F_STORAGE_UTIL': round(features.get('M16A_3F_STORAGE_UTIL_last_value', 0), 2)
        })
       
        # 진행상황 출력
        if (i - 30) % 100 == 0:
            print(f"진행중... {i-30}/{len(df)-30} ({(i-30)/(len(df)-30)*100:.1f}%)")
   
    # DataFrame 변환
    results_df = pd.DataFrame(results)
   
    # CSV 저장
    output_file = 'prediction_evaluation_with_correction.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 결과 저장 완료: {output_file}")
   
    # ============ 평가 통계 ============
    print("\n" + "="*80)
    print("📊 전체 평가 통계")
    print("="*80)
    print(f"총 예측 수: {len(results_df)}")
    
    print(f"\n[보정 전]")
    print(f"  평균 오차: {results_df['오차'].abs().mean():.2f}")
    print(f"  평균 오차율: {results_df['오차율(%)'].mean():.2f}%")
    print(f"  최대 오차: {results_df['오차'].abs().max():.2f}")
    
    print(f"\n[보정 후]")
    print(f"  평균 오차: {results_df['오차_보정후'].abs().mean():.2f}")
    print(f"  평균 오차율: {results_df['오차율_보정후(%)'].mean():.2f}%")
    print(f"  최대 오차: {results_df['오차_보정후'].abs().max():.2f}")
    print(f"  평균 보정량: {results_df['보정량'].abs().mean():.2f}")
    
    improvement = results_df['오차율(%)'].mean() - results_df['오차율_보정후(%)'].mean()
    print(f"  오차율 개선: {improvement:.2f}% 포인트")
    
    # 300 급증 예측 성능
    surge_cases = results_df[results_df['300급증'] == '✅']
    if len(surge_cases) > 0:
        print("\n" + "="*80)
        print("🚨 300 급증 예측 성능 (300미만 → 300이상)")
        print("="*80)
        print(f"총 급증 케이스: {len(surge_cases)}개 ({len(surge_cases)/len(results_df)*100:.1f}%)")
        
        # 보정 전 성능
        pred_success_before = (surge_cases['예측값'] >= 300).sum()
        pred_280_before = (surge_cases['예측값'] >= 280).sum()
        
        # 보정 후 성능
        pred_success_after = (surge_cases['예측값_보정후'] >= 300).sum()
        pred_280_after = (surge_cases['예측값_보정후'] >= 280).sum()
        
        print(f"\n[보정 전]")
        print(f"  300 이상 예측 성공: {pred_success_before}/{len(surge_cases)}개 ({pred_success_before/len(surge_cases)*100:.1f}%)")
        print(f"  280 이상 예측 (주의): {pred_280_before}/{len(surge_cases)}개 ({pred_280_before/len(surge_cases)*100:.1f}%)")
        
        print(f"\n[보정 후]")
        print(f"  300 이상 예측 성공: {pred_success_after}/{len(surge_cases)}개 ({pred_success_after/len(surge_cases)*100:.1f}%)")
        print(f"  280 이상 예측 (주의): {pred_280_after}/{len(surge_cases)}개 ({pred_280_after/len(surge_cases)*100:.1f}%)")
        print(f"  성능 개선: +{(pred_success_after - pred_success_before)}개 ({(pred_success_after - pred_success_before)/len(surge_cases)*100:.1f}% 포인트)")
    
    # 위험도별 통계
    print("\n" + "="*80)
    print("⚠️ 위험도별 예측 성능")
    print("="*80)
    for risk in ['HIGH', 'MEDIUM', 'LOW']:
        risk_df = results_df[results_df['위험도'] == risk]
        if len(risk_df) > 0:
            actual_surge = (risk_df['300급증'] == '✅').sum()
            print(f"\n[{risk} 위험도] - {len(risk_df)}개 ({len(risk_df)/len(results_df)*100:.1f}%)")
            print(f"  실제 급증 발생: {actual_surge}개 ({actual_surge/len(risk_df)*100:.1f}%)")
            print(f"  평균 오차(보정전): {risk_df['오차'].abs().mean():.2f}")
            print(f"  평균 오차(보정후): {risk_df['오차_보정후'].abs().mean():.2f}")
    
    # 300이상 점프 구간
    print(f"\n300이상 점프 구간: {results_df['300이상점프'].value_counts().get('🔴', 0)}개")
    
    # 극단값 분석
    extreme_cases = results_df[results_df['실제값'] >= 300]
    if len(extreme_cases) > 0:
        print(f"\n실제값 극단(≥300): {len(extreme_cases)}개 ({len(extreme_cases)/len(results_df)*100:.1f}%)")
        pred_extreme_before = (results_df['예측값'] >= 300).sum()
        pred_extreme_after = (results_df['예측값_보정후'] >= 300).sum()
        print(f"예측값 극단(≥300) 보정전: {pred_extreme_before}개")
        print(f"예측값 극단(≥300) 보정후: {pred_extreme_after}개")
   
    # 급증 예측 실패 케이스 분석
    if len(surge_cases) > 0:
        surge_failed = surge_cases[surge_cases['급증예측성공_보정후'] != '✅']
        if len(surge_failed) > 0:
            print("\n" + "="*80)
            print("❌ 급증 예측 실패 케이스 (상위 10개)")
            print("="*80)
            failed_top = surge_failed.nsmallest(10, '예측값_보정후')
            display_cols = ['현재시간', '현재값', '실제값', '예측값_보정후', '위험도', 'M16A_3F_CMD', 'M16A_3F_STORAGE_UTIL']
            print(failed_top[display_cols].to_string(index=False))
    
    # 극단값 구간별 분석
    if len(extreme_cases) > 0:
        print("\n" + "="*80)
        print("🔴 극단값(≥300) 구간별 예측 정확도")
        print("="*80)
        
        # 구간별 분석
        ranges = [(300, 320), (320, 350), (350, 400), (400, 500)]
        for low, high in ranges:
            range_df = extreme_cases[(extreme_cases['실제값'] >= low) & (extreme_cases['실제값'] < high)]
            if len(range_df) > 0:
                print(f"\n[{low}~{high} 구간] - {len(range_df)}개")
                print(f"  평균 오차(보정전): {range_df['오차'].abs().mean():.2f}")
                print(f"  평균 오차(보정후): {range_df['오차_보정후'].abs().mean():.2f}")
                pred_success = (range_df['예측값_보정후'] >= 300).sum()
                print(f"  300이상 예측 성공률: {pred_success}/{len(range_df)}개 ({pred_success/len(range_df)*100:.1f}%)")
    
    # 상위 오차 구간
    print("\n" + "="*80)
    print("❌ 오차 상위 10개 구간 (보정 후 기준)")
    print("="*80)
    top_errors = results_df.nlargest(10, '오차율_보정후(%)')
    display_cols = ['현재시간', '현재값', '실제값', '예측값_보정후', '오차_보정후', '오차율_보정후(%)', '위험도', '300급증']
    print(top_errors[display_cols].to_string(index=False))
   
    return results_df

if __name__ == '__main__':
    print("🚀 V6 실시간 예측 평가 + 급증 위험도 보정 시작...\n")
    print("="*80)
    print("💡 보정 로직:")
    print("  - M16A_3F_CMD < 180: HIGH 위험도 (70% 급증 확률)")
    print("  - STORAGE_UTIL 30~40: HIGH 위험도 (45% 급증 확률)")
    print("  - M16A_3F_CMD 260~280: HIGH 위험도 (67% 급증 확률)")
    print("  - M16A_3F_CMD < 220: MEDIUM 위험도 (30% 급증 확률)")
    print("  - 기타: LOW 위험도 (5% 급증 확률)")
    print("="*80)
    print()
    
    results = evaluate_all_predictions_with_correction()
   
    if results is not None:
        print(f"\n✅ 평가 완료! 총 {len(results)}개 예측 생성")
        print(f"📁 결과 파일: prediction_evaluation_with_correction.csv")
        
        # 최종 요약
        print("\n" + "="*80)
        print("💡 최종 요약")
        print("="*80)
        
        improvement = results['오차율(%)'].mean() - results['오차율_보정후(%)'].mean()
        print(f"✅ 오차율 개선: {improvement:.2f}% 포인트")
        
        high_risk = results[results['위험도'] == 'HIGH']
        if len(high_risk) > 0:
            surge_in_high = (high_risk['300급증'] == '✅').sum()
            print(f"✅ HIGH 위험도 적중률: {surge_in_high}/{len(high_risk)}개 ({surge_in_high/len(high_risk)*100:.1f}%)")
        
        surge_all = results[results['300급증'] == '✅']
        if len(surge_all) > 0:
            surge_detected = (surge_all['급증예측성공_보정후'] == '✅').sum()
            print(f"✅ 300 급증 예측 성공률: {surge_detected}/{len(surge_all)}개 ({surge_detected/len(surge_all)*100:.1f}%)")