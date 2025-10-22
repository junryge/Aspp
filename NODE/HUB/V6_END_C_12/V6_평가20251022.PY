# -*- coding: utf-8 -*-
"""
V6 평가 전용 코드 - 이상감지 Feature FULL 버전 + 급증 위험도 보정
전체 데이터를 슬라이딩 윈도우로 평가
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
        risk_level = "HIGH"
        surge_prob = 0.70
        expected_surge = 60
    elif storage_last >= 30 and storage_last <= 40:
        risk_level = "HIGH"
        surge_prob = 0.45
        expected_surge = 50
    elif cmd_last >= 260 and cmd_last <= 280:
        risk_level = "HIGH"
        surge_prob = 0.67
        expected_surge = 60
    elif cmd_last < 220:
        risk_level = "MEDIUM"
        surge_prob = 0.30
        expected_surge = 35
    else:
        risk_level = "LOW"
        surge_prob = 0.05
        expected_surge = 20
    
    # 추가 보정 요소
    if cmd_slope < -5:
        surge_prob *= 1.2
        expected_surge *= 1.1
    
    storage_slope = features.get('M16A_3F_STORAGE_UTIL_slope', 0)
    if storage_slope > 2:
        surge_prob *= 1.1
    
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
    risk_level, surge_prob, expected_surge = calculate_surge_risk(features)
    
    current_value = seq_target[-1]
    seq_mean = np.mean(seq_target)
    seq_max = np.max(seq_target)
    recent_mean = np.mean(seq_target[-5:])
    
    correction = 0
    
    if current_value < 300:
        if risk_level == "HIGH":
            if prediction < 300:
                potential_value = current_value + expected_surge * surge_prob
                if potential_value >= 300:
                    correction = (potential_value - prediction) * 0.7
                else:
                    correction = expected_surge * surge_prob * 0.5
        elif risk_level == "MEDIUM":
            if prediction < 300 and surge_prob > 0.25:
                correction = expected_surge * surge_prob * 0.3
        else:
            correction = expected_surge * surge_prob * 0.1
    
    recent_increase = recent_mean - seq_mean
    if recent_increase > 10:
        correction += recent_increase * 0.2
    
    if seq_max >= 300:
        correction += 5
    
    corrected_prediction = prediction + correction
    corrected_prediction = max(corrected_prediction, current_value - 50)
    corrected_prediction = min(corrected_prediction, current_value + 100)
    
    if current_value < 250 and corrected_prediction > 350:
        corrected_prediction = 320
    
    return corrected_prediction, risk_level, surge_prob, correction


def evaluate_all_predictions():
    """전체 데이터를 슬라이딩 윈도우로 평가 (이상감지 Feature FULL)"""
   
    # V6 Ultimate 필수 컬럼 정의
    FEATURE_COLS = {
        'storage': ['M16A_3F_STORAGE_UTIL'],
        'cmd': ['M16A_3F_CMD', 'M16A_6F_TO_HUB_CMD'],
        'inflow': ['M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2', 'M14A_3F_TO_HUB_JOB2'],
        'outflow': ['M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB', 'M16A_3F_TO_M14A_3F_JOB'],
        'maxcapa': ['M16A_6F_LFT_MAXCAPA', 'M16A_2F_LFT_MAXCAPA']
    }
     
    # 모델 로드
    try:
        with open('xgboost_model_30min_10min_이상감지FULL.pkl', 'rb') as f:
            model = pickle.load(f)
        print("✅ 모델 로드 완료: xgboost_model_30min_10min_이상감지FULL.pkl")
    except Exception as e:
        print(f"❌ 모델 파일 없음: {e}")
        return None
   
    # 데이터 로드
    df = pd.read_csv('HUB0905101512.CSV', on_bad_lines='skip')
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
   
    # 슬라이딩 윈도우: 30개 시퀀스 → 10분 후 예측
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
       
        # 🔥 타겟 컬럼 기본 통계 (재사용)
        seq_mean = np.mean(seq_target)
        seq_std = np.std(seq_target)
        seq_max = np.max(seq_target)
        seq_min = np.min(seq_target)
        last_5_mean = np.mean(seq_target[-5:])
        last_10_mean = np.mean(seq_target[-10:])
        first_10_mean = np.mean(seq_target[:10])
        slope = np.polyfit(np.arange(30), seq_target, 1)[0]
        
        features = {
            # 타겟 컬럼 기본 특성
            'target_mean': seq_mean,
            'target_std': seq_std,
            'target_last_5_mean': last_5_mean,
            'target_max': seq_max,
            'target_min': seq_min,
            'target_slope': slope,
            'target_last_10_mean': last_10_mean,
            'target_first_10_mean': first_10_mean,
        }
        
        # 🔥🔥🔥 타겟 컬럼 이상감지 Feature
        features['target_near_threshold'] = 1 if 290 <= seq_max < 300 else 0
        features['target_rising_trend'] = 1 if slope > 0.5 else 0
        features['target_recent_high'] = 1 if last_5_mean > seq_mean + 10 else 0
        features['target_risk_score'] = max(0, (seq_max - 290) / 10)
        features['target_near_300'] = 1 if seq_max >= 295 else 0
        features['target_last_5_rising'] = 1 if last_5_mean > last_10_mean else 0
        features['target_acceleration'] = np.mean(np.diff(seq_target[-5:])) if len(seq_target) >= 6 else 0
        
        # 각 컬럼 그룹별 특성 추가
        for group_name, cols in FEATURE_COLS.items():
            for col in cols:
                if col in df.columns:
                    seq_data_col = seq_data[col].values
                    
                    # 🔥 컬럼별 기본 통계 (재사용)
                    col_mean = np.mean(seq_data_col)
                    col_std = np.std(seq_data_col)
                    col_max = np.max(seq_data_col)
                    col_min = np.min(seq_data_col)
                    col_last_5_mean = np.mean(seq_data_col[-5:])
                    col_last_10_mean = np.mean(seq_data_col[-10:])
                    col_first_10_mean = np.mean(seq_data_col[:10])
                    col_slope = np.polyfit(np.arange(30), seq_data_col, 1)[0]
                    
                    # 기본 통계
                    features[f'{col}_mean'] = col_mean
                    features[f'{col}_std'] = col_std
                    features[f'{col}_max'] = col_max
                    features[f'{col}_min'] = col_min
                    features[f'{col}_last_5_mean'] = col_last_5_mean
                    features[f'{col}_last_10_mean'] = col_last_10_mean
                    features[f'{col}_slope'] = col_slope
                    features[f'{col}_first_10_mean'] = col_first_10_mean
                    features[f'{col}_mid_10_mean'] = np.mean(seq_data_col[10:20])
                    features[f'{col}_last_value'] = seq_data_col[-1]
                    
                    # 🔥🔥🔥 각 컬럼별 이상감지 Feature
                    if 'CMD' in col:
                        col_threshold_low = 180
                        col_threshold_high = 280
                        features[f'{col}_near_low_threshold'] = 1 if col_min <= col_threshold_low else 0
                        features[f'{col}_near_high_threshold'] = 1 if col_max >= col_threshold_high else 0
                        features[f'{col}_in_danger_zone'] = 1 if (col_last_5_mean < 220) else 0
                    elif 'STORAGE' in col:
                        features[f'{col}_near_threshold'] = 1 if 30 <= col_max <= 40 else 0
                        features[f'{col}_in_danger_zone'] = 1 if (30 <= col_last_5_mean <= 40) else 0
                    else:
                        features[f'{col}_near_max'] = 1 if col_max >= (col_mean + 2*col_std) else 0
                    
                    # 공통 이상감지 Feature
                    features[f'{col}_rising_trend'] = 1 if col_slope > 0.5 else 0
                    features[f'{col}_falling_trend'] = 1 if col_slope < -0.5 else 0
                    features[f'{col}_recent_high'] = 1 if col_last_5_mean > col_mean + (col_std * 0.5) else 0
                    features[f'{col}_recent_low'] = 1 if col_last_5_mean < col_mean - (col_std * 0.5) else 0
                    features[f'{col}_last_5_rising'] = 1 if col_last_5_mean > col_last_10_mean else 0
                    features[f'{col}_acceleration'] = np.mean(np.diff(seq_data_col[-5:])) if len(seq_data_col) >= 6 else 0
                    features[f'{col}_volatility'] = col_std / (col_mean + 1e-6)
        
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
        features['net_flow_positive'] = 1 if (inflow_sum - outflow_sum) > 0 else 0
        
        # CMD 총합
        cmd_sum = 0
        for col in FEATURE_COLS['cmd']:
            if col in df.columns:
                cmd_sum += df[col].iloc[i-1]
        features['total_cmd'] = cmd_sum
        features['total_cmd_low'] = 1 if cmd_sum < 400 else 0
       
        X_pred = pd.DataFrame([features])
       
        # 기본 예측
        prediction_original = model.predict(X_pred)[0]
        
        # 🔥 보정된 예측
        prediction_corrected, risk_level, surge_prob, correction_amount = \
            apply_adaptive_correction(prediction_original, features, seq_target)
       
        # 300 이상 점프 감지 (시퀀스 내)
        jump_detected = np.any(seq_target >= 300)
        
        # 급증 여부 판정
        current_value = seq_target[-1]
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
    output_file = 'prediction_evaluation_이상감지FULL.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 결과 저장 완료: {output_file}")
   
    # ============ 평가 통계 ============
    print("\n" + "="*80)
    print("📊 전체 평가 통계 (이상감지 Feature FULL + 급증 위험도 보정)")
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
    print("🚀 V6 실시간 예측 평가 시작 (이상감지 Feature FULL + 급증 위험도 보정)...\n")
    print("="*80)
    print("💡 보정 로직:")
    print("  - M16A_3F_CMD < 180: HIGH 위험도 (70% 급증 확률)")
    print("  - STORAGE_UTIL 30~40: HIGH 위험도 (45% 급증 확률)")
    print("  - M16A_3F_CMD 260~280: HIGH 위험도 (67% 급증 확률)")
    print("  - M16A_3F_CMD < 220: MEDIUM 위험도 (30% 급증 확률)")
    print("  - 기타: LOW 위험도 (5% 급증 확률)")
    print("="*80)
    print()
    
    results = evaluate_all_predictions()
   
    if results is not None:
        print(f"\n✅ 평가 완료! 총 {len(results)}개 예측 생성")
        print(f"📁 결과 파일: prediction_evaluation_이상감지FULL.csv")
        
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