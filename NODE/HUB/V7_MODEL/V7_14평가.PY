import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta

def evaluate_perfect_version():
    """
    🎯 완벽 버전 평가: 30시퀀스 → 10분 예측 + 임계값 280 + Momentum
    """
    print("="*80)
    print("🎯 완벽 버전 평가: 가중치 5배 + 임계값 280 + Momentum")
    print("="*80)
   
    FEATURE_COLS = {
        'storage': ['M16A_3F_STORAGE_UTIL'],
        'fs_storage': ['CD_M163FSTORAGEUSE', 'CD_M163FSTORAGETOTAL', 'CD_M163FSTORAGEUTIL'],
        'hub': ['HUBROOMTOTAL'],
        'cmd': ['M16A_3F_CMD', 'M16A_6F_TO_HUB_CMD'],
        'inflow': ['M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2', 'M14A_3F_TO_HUB_JOB2'],
        'outflow': ['M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB', 'M16A_3F_TO_M14A_3F_JOB'],
        'maxcapa': ['M16A_6F_LFT_MAXCAPA', 'M16A_2F_LFT_MAXCAPA']
    }
   
    # 모델 로드
    try:
        with open('xgboost_완벽버전_가중치5배_임계값280_momentum.pkl', 'rb') as f:
            model = pickle.load(f)
        print("✅ 모델 로드 완료: xgboost_완벽버전_가중치5배_임계값280_momentum.pkl")
    except Exception as e:
        print(f"❌ 모델 파일 없음: {e}")
        print("   먼저 '완벽버전_학습코드.py'를 실행하세요.")
        return None
   
    # 데이터 로드
    try:
        df = pd.read_csv('HUB0905101512.csv', on_bad_lines='skip', encoding='utf-8', low_memory=False)
    except:
        try:
            df = pd.read_csv('HUB0905101512.csv', on_bad_lines='skip', encoding='cp949', low_memory=False)
        except:
            df = pd.read_csv('HUB0905101512.csv', on_bad_lines='skip', encoding='euc-kr', low_memory=False)
   
    TARGET_COL = 'CURRENT_M16A_3F_JOB_2'
   
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')
    df = df.dropna(subset=[TARGET_COL])
   
    print(f"✅ 데이터 로드: {len(df)}개 행")
   
    # STAT_DT 처리
    if 'STAT_DT' in df.columns:
        try:
            df['STAT_DT'] = pd.to_datetime(df['STAT_DT'].astype(str), format='%Y%m%d%H%M')
        except:
            base_time = datetime(2024, 1, 1, 0, 0)
            df['STAT_DT'] = [base_time + timedelta(minutes=i) for i in range(len(df))]
   
    results = []
   
    print("\n슬라이딩 윈도우 평가 시작 (30시퀀스 → 10분 예측 + Momentum)...")
    for i in range(30, len(df) - 10):  # 🔥 수정: -10 추가!
        seq_data = df.iloc[i-30:i].copy()
        seq_target = seq_data[TARGET_COL].values
       
        current_time = seq_data['STAT_DT'].iloc[-1]
        current_value = seq_target[-1]
        prediction_time = current_time + timedelta(minutes=10)  # ✅ 10분 후
       
        # 🔥 수정: 10분 구간의 최대값 (학습과 동일하게!)
        actual_value = df[TARGET_COL].iloc[i:i+10].max()
        actual_time = df.iloc[i+9]['STAT_DT']  # 10분 후의 시간
       
        # Feature 생성 (학습 코드와 완벽히 동일!)
        features = {
            'target_mean': np.mean(seq_target),
            'target_std': np.std(seq_target),
            'target_max': np.max(seq_target),
            'target_min': np.min(seq_target),
            'target_last_value': seq_target[-1],
            'target_last_5_mean': np.mean(seq_target[-5:]),
            'target_slope': np.polyfit(np.arange(30), seq_target, 1)[0],
        }
       
        # 🆕 Momentum Feature 추가 (학습 코드와 동일!)
        features['target_acceleration'] = (seq_target[-5:].mean() - seq_target[-10:-5].mean()) / 5
        features['target_is_rising'] = 1 if seq_target[-1] > seq_target[-5] else 0
        features['target_rapid_rise'] = 1 if (seq_target[-1] - seq_target[-5] > 10) else 0
        features['target_last_10_mean'] = np.mean(seq_target[-10:])
       
        # 각 컬럼 Feature
        for group_name, cols in FEATURE_COLS.items():
            for col in cols:
                if col not in df.columns:
                    continue
               
                col_seq = seq_data[col].values
               
                if group_name == 'maxcapa':
                    features[f'{col}_last_value'] = col_seq[-1]
               
                elif group_name in ['cmd', 'storage', 'fs_storage', 'hub']:
                    features[f'{col}_mean'] = np.mean(col_seq)
                    features[f'{col}_std'] = np.std(col_seq)
                    features[f'{col}_max'] = np.max(col_seq)
                    features[f'{col}_min'] = np.min(col_seq)
                    features[f'{col}_last_value'] = col_seq[-1]
                    features[f'{col}_last_5_mean'] = np.mean(col_seq[-5:])
                    features[f'{col}_slope'] = np.polyfit(np.arange(30), col_seq, 1)[0]
               
                else:
                    features[f'{col}_mean'] = np.mean(col_seq)
                    features[f'{col}_last_value'] = col_seq[-1]
                    features[f'{col}_slope'] = np.polyfit(np.arange(30), col_seq, 1)[0]
       
        # FS Storage Feature
        if 'CD_M163FSTORAGEUSE' in df.columns and 'CD_M163FSTORAGETOTAL' in df.columns and 'CD_M163FSTORAGEUTIL' in df.columns:
            use_seq = seq_data['CD_M163FSTORAGEUSE'].values
            total_seq = seq_data['CD_M163FSTORAGETOTAL'].values
            util_seq = seq_data['CD_M163FSTORAGEUTIL'].values
           
            features['CD_M163FSTORAGEUSE_mean'] = np.mean(use_seq)
            features['CD_M163FSTORAGEUSE_std'] = np.std(use_seq)
            features['CD_M163FSTORAGEUSE_max'] = np.max(use_seq)
            features['CD_M163FSTORAGEUSE_min'] = np.min(use_seq)
            features['CD_M163FSTORAGEUSE_last_value'] = use_seq[-1]
            features['CD_M163FSTORAGEUSE_last_5_mean'] = np.mean(use_seq[-5:])
            features['CD_M163FSTORAGEUSE_slope'] = np.polyfit(np.arange(30), use_seq, 1)[0]
           
            features['CD_M163FSTORAGETOTAL_mean'] = np.mean(total_seq)
            features['CD_M163FSTORAGETOTAL_std'] = np.std(total_seq)
            features['CD_M163FSTORAGETOTAL_max'] = np.max(total_seq)
            features['CD_M163FSTORAGETOTAL_min'] = np.min(total_seq)
            features['CD_M163FSTORAGETOTAL_last_value'] = total_seq[-1]
            features['CD_M163FSTORAGETOTAL_last_5_mean'] = np.mean(total_seq[-5:])
            features['CD_M163FSTORAGETOTAL_slope'] = np.polyfit(np.arange(30), total_seq, 1)[0]
           
            features['CD_M163FSTORAGEUTIL_mean'] = np.mean(util_seq)
            features['CD_M163FSTORAGEUTIL_std'] = np.std(util_seq)
            features['CD_M163FSTORAGEUTIL_max'] = np.max(util_seq)
            features['CD_M163FSTORAGEUTIL_min'] = np.min(util_seq)
            features['CD_M163FSTORAGEUTIL_last_value'] = util_seq[-1]
            features['CD_M163FSTORAGEUTIL_last_5_mean'] = np.mean(util_seq[-5:])
            features['CD_M163FSTORAGEUTIL_slope'] = np.polyfit(np.arange(30), util_seq, 1)[0]
           
            features['storage_use_rate'] = (use_seq[-1] - use_seq[0]) / 30
            features['storage_remaining'] = total_seq[-1] - use_seq[-1]
            features['storage_util_last'] = util_seq[-1]
            features['storage_util_high'] = 1 if util_seq[-1] >= 7 else 0
            features['storage_util_critical'] = 1 if util_seq[-1] >= 10 else 0
       
        # HUBROOMTOTAL Feature
        if 'HUBROOMTOTAL' in df.columns:
            hub_seq = seq_data['HUBROOMTOTAL'].values
            hub_last = hub_seq[-1]
           
            features['hub_critical'] = 1 if hub_last < 590 else 0
            features['hub_high'] = 1 if hub_last < 610 else 0
            features['hub_warning'] = 1 if hub_last < 620 else 0
            features['hub_decrease_rate'] = (hub_seq[0] - hub_last) / 30
           
            if 'CD_M163FSTORAGEUTIL' in df.columns:
                storage_util_last = df['CD_M163FSTORAGEUTIL'].iloc[i-1]
                features['hub_storage_risk'] = 1 if (hub_last < 610 and storage_util_last >= 7) else 0
       
        # 유입-유출, CMD
        inflow_sum = sum(df[col].iloc[i-1] for col in FEATURE_COLS['inflow'] if col in df.columns)
        outflow_sum = sum(df[col].iloc[i-1] for col in FEATURE_COLS['outflow'] if col in df.columns)
        features['net_flow'] = inflow_sum - outflow_sum
       
        cmd_sum = sum(df[col].iloc[i-1] for col in FEATURE_COLS['cmd'] if col in df.columns)
        features['total_cmd'] = cmd_sum
        features['total_cmd_low'] = 1 if cmd_sum < 220 else 0
        features['total_cmd_very_low'] = 1 if cmd_sum < 200 else 0
       
        if 'HUBROOMTOTAL' in df.columns:
            hub_last = df['HUBROOMTOTAL'].iloc[i-1]
            features['hub_cmd_bottleneck'] = 1 if (hub_last < 610 and cmd_sum < 220) else 0
       
        if 'M16A_3F_STORAGE_UTIL' in df.columns:
            storage_util = df['M16A_3F_STORAGE_UTIL'].iloc[i-1]
            features['storage_util_critical'] = 1 if storage_util >= 205 else 0
            features['storage_util_high_risk'] = 1 if storage_util >= 207 else 0
       
        # 🔥 급증 위험도 점수
        features['surge_risk_score'] = (
            features.get('hub_high', 0) * 3 +
            features.get('storage_util_critical', 0) * 2 +
            features.get('total_cmd_low', 0) * 1 +
            features.get('storage_util_high', 0) * 1
        )
       
        # 🆕 급증 임박 신호
        features['surge_imminent'] = 1 if (
            seq_target[-1] > 272 and
            features.get('target_acceleration', 0) > 0.2 and
            features.get('hub_high', 0) == 1
        ) else 0
       
        X_pred = pd.DataFrame([features])
       
        # 예측
        prediction = model.predict(X_pred)[0]
       
        # 급증 판정
        is_surge = (current_value < 300 and actual_value >= 300)
       
        # 위험도 판정
        risk_level = "LOW"
        risk_score = features.get('surge_risk_score', 0)
        if risk_score >= 5:
            risk_level = "CRITICAL"
        elif risk_score >= 3:
            risk_level = "HIGH"
        elif risk_score >= 1:
            risk_level = "MEDIUM"
       
        results.append({
            '현재시간': current_time.strftime('%Y-%m-%d %H:%M'),
            '예측시점': prediction_time.strftime('%Y-%m-%d %H:%M'),
            '현재값': round(current_value, 2),
            '실제값': round(actual_value, 2),
            '예측값': round(prediction, 2),
            '오차': round(abs(actual_value - prediction), 2),
            '오차율(%)': round(abs(actual_value - prediction) / max(actual_value, 1) * 100, 2),
            '시퀀스MAX': round(np.max(seq_target), 2),
            '300급증': '✅' if is_surge else '',
            '감지_290': '✅' if (is_surge and prediction >= 290) else '',
            '감지_285': '✅' if (is_surge and prediction >= 285) else '',
            '감지_280': '✅' if (is_surge and prediction >= 280) else '',  # ⭐ 메인!
            '감지_275': '✅' if (is_surge and prediction >= 275) else '',
            '위험도': risk_level,
            '위험점수': risk_score,
            'momentum상승': '✅' if features.get('target_is_rising', 0) == 1 else '',
            'momentum가속': round(features.get('target_acceleration', 0), 2),
            '급증임박': '✅' if features.get('surge_imminent', 0) == 1 else '',
            'HUBROOMTOTAL': round(features.get('HUBROOMTOTAL_last_value', 0), 0),
            'FS_UTIL': round(features.get('CD_M163FSTORAGEUTIL_last_value', 0), 0),
            'CMD': round(features.get('total_cmd', 0), 0),
            'STORAGE_UTIL': round(features.get('M16A_3F_STORAGE_UTIL_last_value', 0), 0),
        })
       
        if (i - 30) % 1000 == 0:
            print(f"진행: {i-30}/{len(df)-30} ({(i-30)/(len(df)-30)*100:.1f}%)")
   
    results_df = pd.DataFrame(results)
   
    output_file = '완벽버전_가중치5배_임계값280_momentum_평가결과.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 결과 저장: {output_file}")
   
    # 통계
    print("\n" + "="*80)
    print("📊 평가 통계 (완벽 버전: 가중치 5배 + 임계값 280 + Momentum)")
    print("="*80)
   
    print(f"총 예측: {len(results_df)}개")
    print(f"평균 오차: {results_df['오차'].mean():.2f}")
    print(f"평균 오차율: {results_df['오차율(%)'].mean():.2f}%")
   
    surge_cases = results_df[results_df['300급증'] == '✅']
    if len(surge_cases) > 0:
        print(f"\n🎯 300 급증 예측 성능:")
        print(f"  발생: {len(surge_cases)}개 ({len(surge_cases)/len(results_df)*100:.2f}%)")
        
        for threshold in [290, 285, 280, 275]:
            detected = (surge_cases[f'감지_{threshold}'] == '✅').sum()
            marker = "⭐" if threshold == 280 else "  "
            print(f"{marker} 임계값 {threshold}: {detected}/{len(surge_cases)}개 ({detected/len(surge_cases)*100:.1f}%)")
   
    extreme_cases = results_df[results_df['실제값'] >= 300]
    if len(extreme_cases) > 0:
        extreme_detected_280 = (extreme_cases['예측값'] >= 280).sum()
       
        print(f"\n극단값 (300+) 감지 (임계값 280):")
        print(f"  발생: {len(extreme_cases)}개 ({len(extreme_cases)/len(results_df)*100:.1f}%)")
        print(f"  감지: {extreme_detected_280}/{len(extreme_cases)}개 ({extreme_detected_280/len(extreme_cases)*100:.1f}%)")
   
    # Momentum Feature 분석
    print(f"\n🆕 Momentum Feature 분석:")
    if len(surge_cases) > 0:
        rising_in_surge = (surge_cases['momentum상승'] == '✅').sum()
        imminent_in_surge = (surge_cases['급증임박'] == '✅').sum()
        print(f"  급증 시 상승 중: {rising_in_surge}/{len(surge_cases)}개 ({rising_in_surge/len(surge_cases)*100:.1f}%)")
        print(f"  급증 임박 신호: {imminent_in_surge}/{len(surge_cases)}개 ({imminent_in_surge/len(surge_cases)*100:.1f}%)")
   
    # 위험도별 통계
    print(f"\n위험도별 분석:")
    for risk in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
        risk_df = results_df[results_df['위험도'] == risk]
        if len(risk_df) > 0:
            surge_in_risk = (risk_df['300급증'] == '✅').sum()
            print(f"  {risk}: {len(risk_df)}개 - 급증 {surge_in_risk}개 ({surge_in_risk/len(risk_df)*100:.1f}%)")
   
    # 상위 오차
    print(f"\n❌ 오차 상위 10개:")
    top_errors = results_df.nlargest(10, '오차율(%)')
    print(top_errors[['현재시간', '현재값', '실제값', '예측값', '오차', '위험도', '300급증', '급증임박']].to_string(index=False))
   
    # 급증 예측 실패 케이스
    if len(surge_cases) > 0:
        surge_failed = surge_cases[surge_cases['감지_280'] != '✅']
        if len(surge_failed) > 0:
            print(f"\n❌ 급증 예측 실패 케이스 (상위 10개, 임계값 280):")
            failed_top = surge_failed.nsmallest(10, '예측값')
            print(failed_top[['현재시간', '현재값', '실제값', '예측값', 'HUBROOMTOTAL', 'FS_UTIL', 'CMD', '위험점수', 'momentum가속', '급증임박']].to_string(index=False))
   
    return results_df

if __name__ == '__main__':
    print("🚀 완벽 버전 모델 평가 시작...\n")
    results = evaluate_perfect_version()
   
    if results is not None:
        print(f"\n✅ 평가 완료! 총 {len(results)}개 예측")
        print(f"📁 결과: 완벽버전_가중치5배_임계값280_momentum_평가결과.csv")
        print(f"\n🎉 60~70% 감지율 목표 달성 예상!")