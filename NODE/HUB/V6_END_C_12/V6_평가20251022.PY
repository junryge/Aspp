# -*- coding: utf-8 -*-
"""
V6 평가 전용 코드 + 급증 위험도 기반 보정 (완전판)
실제값: i번째 행의 단일 값 (반복 없음!)
그래프 + 진행도 + 상세 통계 모두 포함
"""

import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta


def calculate_surge_risk(features):
    """급증 위험도 및 예상 급증량 계산"""
    cmd_last = features.get('M16A_3F_CMD_last_value', 999)
    cmd_mean = features.get('M16A_3F_CMD_mean', 999)
    storage_last = features.get('M16A_3F_STORAGE_UTIL_last_value', 0)
    storage_mean = features.get('M16A_3F_STORAGE_UTIL_mean', 0)
    cmd_slope = features.get('M16A_3F_CMD_slope', 0)
    
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
    
    if cmd_slope < -5:
        surge_prob *= 1.2
        expected_surge *= 1.1
    
    storage_slope = features.get('M16A_3F_STORAGE_UTIL_slope', 0)
    if storage_slope > 2:
        surge_prob *= 1.1
    
    surge_prob = min(surge_prob, 0.85)
    
    return risk_level, surge_prob, expected_surge


def apply_adaptive_correction(prediction, features, seq_target):
    """적응형 예측값 보정"""
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


def evaluate_all_predictions_with_correction():
    """보정된 예측 평가 시스템 (완전판)"""
    
    FEATURE_COLS = {
        'storage': ['M16A_3F_STORAGE_UTIL'],
        'cmd': ['M16A_3F_CMD', 'M16A_6F_TO_HUB_CMD'],
        'inflow': ['M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2', 'M14A_3F_TO_HUB_JOB2'],
        'outflow': ['M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB', 'M16A_3F_TO_M14A_3F_JOB'],
        'maxcapa': ['M16A_6F_LFT_MAXCAPA', 'M16A_2F_LFT_MAXCAPA']
    }
    
    # 모델 로드
    try:
        with open('xgboost_model_30min_10min.pkl', 'rb') as f:
            model = pickle.load(f)
        print("✅ 모델 로드 완료")
    except Exception as e:
        print(f"❌ 모델 파일 없음: {e}")
        return None
    
    # 데이터 로드
    try:
        df = pd.read_csv('HUB0905101512.CSV', on_bad_lines='skip', encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv('HUB0905101512.CSV', on_bad_lines='skip', encoding='cp949')
            print("⚠️ CP949 인코딩")
        except:
            df = pd.read_csv('HUB0905101512.CSV', on_bad_lines='skip', encoding='euc-kr')
            print("⚠️ EUC-KR 인코딩")
    
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
    
    print("\n슬라이딩 윈도우 평가 시작...")
    
    # 슬라이딩 윈도우: 30개 시퀀스 → 실제값은 i번째 행
    for i in range(30, len(df)):
        # 과거 30개 데이터
        seq_data = df.iloc[i-30:i].copy()
        seq_target = seq_data[TARGET_COL].values
        
        # 현재 시점 (시퀀스 마지막)
        current_time = seq_data['STAT_DT'].iloc[-1]
        current_value = seq_target[-1]
        
        # 예측 시점 (10분 후)
        prediction_time = current_time + timedelta(minutes=10)
        
        # 실제값 (i번째 행) ★★★ 핵심: 단일값만 가져옴 ★★★
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
        
        # 각 컬럼 그룹별 특성 추가
        for group_name, cols in FEATURE_COLS.items():
            for col in cols:
                if col in df.columns:
                    col_seq = seq_data[col].values
                    
                    features[f'{col}_mean'] = np.mean(col_seq)
                    features[f'{col}_std'] = np.std(col_seq)
                    features[f'{col}_max'] = np.max(col_seq)
                    features[f'{col}_min'] = np.min(col_seq)
                    features[f'{col}_last_5_mean'] = np.mean(col_seq[-5:])
                    features[f'{col}_last_10_mean'] = np.mean(col_seq[-10:])
                    features[f'{col}_slope'] = np.polyfit(np.arange(30), col_seq, 1)[0]
                    features[f'{col}_first_10_mean'] = np.mean(col_seq[:10])
                    features[f'{col}_mid_10_mean'] = np.mean(col_seq[10:20])
                    features[f'{col}_last_value'] = col_seq[-1]
        
        # 유입-유출 차이
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
            '300급증': '✅' if is_surge else '',
            '급증예측성공': '✅' if (is_surge and prediction_corrected >= 300) else '',
            '실제값상태': '🔴극단' if actual_value >= 300 else ('🟡주의' if actual_value >= 280 else '🟢정상'),
            '예측상태': '🔴극단' if prediction_corrected >= 300 else ('🟡주의' if prediction_corrected >= 280 else '🟢정상'),
            'M16A_3F_CMD': round(features.get('M16A_3F_CMD_last_value', 0), 2),
            'M16A_3F_STORAGE_UTIL': round(features.get('M16A_3F_STORAGE_UTIL_last_value', 0), 2)
        })
        
        # 진행상황 출력 (100개마다)
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
    
    # 300 급증 예측 성능
    surge_cases = results_df[results_df['300급증'] == '✅']
    if len(surge_cases) > 0:
        print("\n" + "="*80)
        print("🚨 300 급증 예측 성능 (300미만 → 300이상)")
        print("="*80)
        print(f"총 급증 케이스: {len(surge_cases)}개")
        
        # 보정 전 성능
        pred_success_before = (surge_cases['예측값'] >= 300).sum()
        pred_280_before = (surge_cases['예측값'] >= 280).sum()
        
        # 보정 후 성능
        pred_success_after = (surge_cases['예측값_보정후'] >= 300).sum()
        pred_280_after = (surge_cases['예측값_보정후'] >= 280).sum()
        
        print(f"\n[보정 전]")
        print(f"  300 이상 예측 성공: {pred_success_before}개 ({pred_success_before/len(surge_cases)*100:.1f}%)")
        print(f"  280 이상 예측 (주의): {pred_280_before}개 ({pred_280_before/len(surge_cases)*100:.1f}%)")
        
        print(f"\n[보정 후]")
        print(f"  300 이상 예측 성공: {pred_success_after}개 ({pred_success_after/len(surge_cases)*100:.1f}%)")
        print(f"  280 이상 예측 (주의): {pred_280_after}개 ({pred_280_after/len(surge_cases)*100:.1f}%)")
        print(f"  성능 개선: +{(pred_success_after - pred_success_before)}개")
    
    # 위험도별 통계
    print("\n" + "="*80)
    print("⚠️ 위험도별 예측 성능")
    print("="*80)
    for risk in ['HIGH', 'MEDIUM', 'LOW']:
        risk_df = results_df[results_df['위험도'] == risk]
        if len(risk_df) > 0:
            actual_surge = (risk_df['300급증'] == '✅').sum()
            print(f"\n[{risk} 위험도] - {len(risk_df)}개")
            print(f"  실제 급증 발생: {actual_surge}개 ({actual_surge/len(risk_df)*100:.1f}%)")
            print(f"  평균 오차(보정후): {risk_df['오차_보정후'].abs().mean():.2f}")
    
    # 급증 예측 실패 케이스 분석
    if len(surge_cases) > 0:
        surge_failed = surge_cases[surge_cases['급증예측성공'] != '✅']
        if len(surge_failed) > 0:
            print("\n" + "="*80)
            print("❌ 급증 예측 실패 케이스 (상위 10개)")
            print("="*80)
            failed_top = surge_failed.nsmallest(10, '예측값_보정후')
            print(failed_top[['현재시간', '실제값', '예측값_보정후', '위험도', 
                              'M16A_3F_CMD', 'M16A_3F_STORAGE_UTIL']].to_string(index=False))
    
    # 극단값 구간 분석
    extreme_cases = results_df[results_df['실제값'] >= 300]
    if len(extreme_cases) > 0:
        print("\n" + "="*80)
        print("🔴 극단값(≥300) 예측 정확도")
        print("="*80)
        
        # 구간별 분석
        ranges = [(300, 320), (320, 350), (350, 400), (400, 500)]
        for low, high in ranges:
            range_df = extreme_cases[(extreme_cases['실제값'] >= low) & (extreme_cases['실제값'] < high)]
            if len(range_df) > 0:
                print(f"\n[{low}~{high} 구간] - {len(range_df)}개")
                print(f"  평균 오차(보정전): {range_df['오차'].abs().mean():.2f}")
                print(f"  평균 오차(보정후): {range_df['오차_보정후'].abs().mean():.2f}")
                print(f"  300이상 예측 성공률: {(range_df['예측값_보정후'] >= 300).sum()/len(range_df)*100:.1f}%")
    
    # 상위 오차 케이스
    print("\n" + "="*80)
    print("❌ 오차 상위 10개 구간")
    print("="*80)
    top_errors = results_df.nlargest(10, '오차율_보정후(%)')
    print(top_errors[['현재시간', '실제값', '예측값_보정후', '오차_보정후', '오차율_보정후(%)', '위험도']].to_string(index=False))
    
    return results_df


if __name__ == '__main__':
    print("🚀 급증 위험도 기반 예측 보정 평가 시작...\n")
    print("="*80)
    print("보정 로직:")
    print("  - M16A_3F_CMD < 180: HIGH 위험도 (70% 급증)")
    print("  - STORAGE_UTIL 30~40: HIGH 위험도 (45% 급증)")
    print("  - M16A_3F_CMD 260~280: HIGH 위험도 (67% 급증)")
    print("  - M16A_3F_CMD < 220: MEDIUM 위험도 (30% 급증)")
    print("  - 기타: LOW 위험도 (5% 급증)")
    print("="*80)
    print()
    
    results = evaluate_all_predictions_with_correction()
    
    if results is not None:
        print(f"\n✅ 평가 완료!")
        print(f"📁 결과 파일: prediction_evaluation_with_correction.csv")
        
        # 최종 요약
        print("\n" + "="*80)
        print("💡 최종 요약")
        print("="*80)
        
        improvement = results['오차율(%)'].mean() - results['오차율_보정후(%)'].mean()
        print(f"오차율 개선: {improvement:.2f}% 포인트")
        
        high_risk = results[results['위험도'] == 'HIGH']
        if len(high_risk) > 0:
            surge_in_high = (high_risk['300급증'] == '✅').sum()
            print(f"HIGH 위험도 적중률: {surge_in_high/len(high_risk)*100:.1f}%")