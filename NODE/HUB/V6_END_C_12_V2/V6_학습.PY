import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta

def evaluate_change_prediction():
    """
    🎯 변화량 예측 모델 평가
    - 단일 모델로 급증/급감 동시 감지
    """
    print("="*80)
    print("🎯 변화량 예측 모델 평가")
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
        with open('xgboost_변화량예측.pkl', 'rb') as f:
            model = pickle.load(f)
        print("✅ 모델 로드 완료: xgboost_변화량예측.pkl")
    except Exception as e:
        print(f"❌ 모델 파일 없음: {e}")
        print("   먼저 'V7_변화량_학습코드.py'를 실행하세요.")
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
   
    # 타입 변환
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
   
    print("\n슬라이딩 윈도우 평가 시작...")
    for i in range(30, len(df) - 10):
        if (i - 30) % 1000 == 0:
            print(f"진행: {i-30}/{len(df)-40} ({(i-30)/(len(df)-40)*100:.1f}%)")
       
        seq_data = df.iloc[i-30:i].copy()
        seq_target = seq_data[TARGET_COL].values
       
        current_time = seq_data['STAT_DT'].iloc[-1]
        current_value = seq_target[-1]
        prediction_time = current_time + timedelta(minutes=10)
       
        # 10분 후 실제 값
        actual_value = df.iloc[i+10][TARGET_COL]
        actual_time = df.iloc[i+10]['STAT_DT']
        actual_change = actual_value - current_value
       
        # Feature 생성 (학습 코드와 동일)
        features = {
            'target_mean': np.mean(seq_target),
            'target_std': np.std(seq_target),
            'target_max': np.max(seq_target),
            'target_min': np.min(seq_target),
            'target_last_value': seq_target[-1],
            'target_last_5_mean': np.mean(seq_target[-5:]),
            'target_slope': np.polyfit(np.arange(30), seq_target, 1)[0],
        }
       
        # 변화량 예측용 추가 Feature
        features['target_from_max'] = np.max(seq_target) - seq_target[-1]
        features['target_decreasing'] = 1 if features['target_slope'] < -1 else 0
        features['target_increasing'] = 1 if features['target_slope'] > 1 else 0
        features['target_volatility'] = np.std(np.diff(seq_target))
        features['target_momentum'] = np.mean(seq_target[-5:]) - np.mean(seq_target[:5])
       
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
       
        # CMD Feature
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
       
        # Storage Util
        if 'M16A_3F_STORAGE_UTIL' in df.columns:
            storage_util = df['M16A_3F_STORAGE_UTIL'].iloc[i-1]
            features['storage_util_critical'] = 1 if storage_util >= 205 else 0
            features['storage_util_high_risk'] = 1 if storage_util >= 207 else 0
       
        # DataFrame으로 변환
        X_pred = pd.DataFrame([features])
       
        # 변화량 예측
        predicted_change = model.predict(X_pred)[0]
        predicted_value = current_value + predicted_change
       
        # 급증/급감 판정
        is_surge = (current_value < 300 and actual_value >= 300)
        is_drop = (current_value >= 300 and actual_value < 300)
        is_extreme = (actual_value >= 300)
       
        surge_detected = (is_surge and predicted_value >= 290)
        drop_detected = (is_drop and predicted_value < 310)
        extreme_detected = (is_extreme and predicted_value >= 290)
       
        # 위험도 판정
        risk_level = "LOW"
        if abs(predicted_change) >= 40:
            risk_level = "HIGH"
        elif abs(predicted_change) >= 25:
            risk_level = "MEDIUM"
       
        results.append({
            '현재시간': current_time.strftime('%Y-%m-%d %H:%M'),
            '예측시점': prediction_time.strftime('%Y-%m-%d %H:%M'),
            '현재값': round(current_value, 2),
            '실제_10분후': round(actual_value, 2),
            '예측_10분후': round(predicted_value, 2),
            '실제_변화량': round(actual_change, 2),
            '예측_변화량': round(predicted_change, 2),
            '변화량_오차': round(abs(actual_change - predicted_change), 2),
            '예측값_오차': round(abs(actual_value - predicted_value), 2),
            '급증': '✅' if is_surge else '',
            '급증감지': '✅' if surge_detected else '',
            '급감': '✅' if is_drop else '',
            '급감감지': '✅' if drop_detected else '',
            '극단값': '✅' if is_extreme else '',
            '극단값감지': '✅' if extreme_detected else '',
            '위험도': risk_level,
            'HUBROOMTOTAL': round(features.get('HUBROOMTOTAL_last_value', 0), 0),
            'FS_UTIL': round(features.get('CD_M163FSTORAGEUTIL_last_value', 0), 0),
            'CMD': round(features.get('total_cmd', 0), 0),
            'STORAGE_UTIL': round(features.get('M16A_3F_STORAGE_UTIL_last_value', 0), 0),
        })
   
    results_df = pd.DataFrame(results)
   
    output_file = '변화량예측_평가결과.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 결과 저장: {output_file}")
   
    # 통계
    print("\n" + "="*80)
    print("📊 평가 통계")
    print("="*80)
   
    print(f"총 예측: {len(results_df)}개")
    print(f"\n변화량 예측 정확도:")
    print(f"  평균 변화량 오차: {results_df['변화량_오차'].mean():.2f}")
    print(f"  평균 예측값 오차: {results_df['예측값_오차'].mean():.2f}")
   
    # 급증 성능
    surge_cases = results_df[results_df['급증'] == '✅']
    if len(surge_cases) > 0:
        surge_detected = (surge_cases['급증감지'] == '✅').sum()
       
        print(f"\n🔺 급증 예측 성능:")
        print(f"  발생: {len(surge_cases)}개 ({len(surge_cases)/len(results_df)*100:.2f}%)")
        print(f"  감지: {surge_detected}/{len(surge_cases)}개 ({surge_detected/len(surge_cases)*100:.1f}%)")
       
        if surge_detected/len(surge_cases) >= 0.78:
            print(f"  ✅ 목표 78% 이상 달성!")
        else:
            print(f"  ⚠️ 목표 78% 미달")
   
    # 급감 성능
    drop_cases = results_df[results_df['급감'] == '✅']
    if len(drop_cases) > 0:
        drop_detected = (drop_cases['급감감지'] == '✅').sum()
       
        print(f"\n🔻 급감 예측 성능:")
        print(f"  발생: {len(drop_cases)}개 ({len(drop_cases)/len(results_df)*100:.2f}%)")
        print(f"  감지: {drop_detected}/{len(drop_cases)}개 ({drop_detected/len(drop_cases)*100:.1f}%)")
   
    # 극단값 성능
    extreme_cases = results_df[results_df['극단값'] == '✅']
    if len(extreme_cases) > 0:
        extreme_detected = (extreme_cases['극단값감지'] == '✅').sum()
       
        print(f"\n극단값 (300+) 감지:")
        print(f"  발생: {len(extreme_cases)}개 ({len(extreme_cases)/len(results_df)*100:.1f}%)")
        print(f"  감지: {extreme_detected}/{len(extreme_cases)}개 ({extreme_detected/len(extreme_cases)*100:.1f}%)")
   
    # 위험도별 통계
    print(f"\n위험도별 분석:")
    for risk in ['HIGH', 'MEDIUM', 'LOW']:
        risk_df = results_df[results_df['위험도'] == risk]
        if len(risk_df) > 0:
            surge_in_risk = (risk_df['급증'] == '✅').sum()
            drop_in_risk = (risk_df['급감'] == '✅').sum()
            print(f"  {risk}: {len(risk_df)}개 - 급증 {surge_in_risk}개, 급감 {drop_in_risk}개")
   
    # 오차 상위 케이스
    print(f"\n❌ 변화량 오차 상위 10개:")
    top_errors = results_df.nlargest(10, '변화량_오차')
    print(top_errors[['현재시간', '현재값', '실제_변화량', '예측_변화량', '변화량_오차', '급증', '급감']].to_string(index=False))
   
    # 급증 예측 실패 케이스
    if len(surge_cases) > 0:
        surge_failed = surge_cases[surge_cases['급증감지'] != '✅']
        if len(surge_failed) > 0:
            print(f"\n❌ 급증 예측 실패 케이스 (상위 5개):")
            failed_top = surge_failed.nlargest(5, '실제_변화량')
            print(failed_top[['현재시간', '현재값', '실제_10분후', '예측_10분후', '실제_변화량', '예측_변화량']].to_string(index=False))
   
    # 급감 예측 실패 케이스
    if len(drop_cases) > 0:
        drop_failed = drop_cases[drop_cases['급감감지'] != '✅']
        if len(drop_failed) > 0:
            print(f"\n❌ 급감 예측 실패 케이스 (상위 5개):")
            failed_top = drop_failed.nsmallest(5, '실제_변화량')
            print(failed_top[['현재시간', '현재값', '실제_10분후', '예측_10분후', '실제_변화량', '예측_변화량']].to_string(index=False))
   
    # 변화량 분포
    print(f"\n📊 변화량 분포:")
    print(f"  실제 변화량:")
    print(f"    평균: {results_df['실제_변화량'].mean():.2f}")
    print(f"    표준편차: {results_df['실제_변화량'].std():.2f}")
    print(f"    최소: {results_df['실제_변화량'].min():.2f}")
    print(f"    최대: {results_df['실제_변화량'].max():.2f}")
   
    print(f"\n  예측 변화량:")
    print(f"    평균: {results_df['예측_변화량'].mean():.2f}")
    print(f"    표준편차: {results_df['예측_변화량'].std():.2f}")
    print(f"    최소: {results_df['예측_변화량'].min():.2f}")
    print(f"    최대: {results_df['예측_변화량'].max():.2f}")
   
    return results_df

if __name__ == '__main__':
    print("🚀 변화량 예측 모델 평가 시작...\n")
    results = evaluate_change_prediction()
   
    if results is not None:
        print(f"\n✅ 평가 완료! 총 {len(results)}개 예측")
        print(f"📁 결과: 변화량예측_평가결과.csv")
        
        # 최종 요약
        surge_total = (results['급증'] == '✅').sum()
        surge_detected = ((results['급증'] == '✅') & (results['급증감지'] == '✅')).sum()
        drop_total = (results['급감'] == '✅').sum()
        drop_detected = ((results['급감'] == '✅') & (results['급감감지'] == '✅')).sum()
        extreme_total = (results['극단값'] == '✅').sum()
        extreme_detected = ((results['극단값'] == '✅') & (results['극단값감지'] == '✅')).sum()
        
        print("\n" + "="*80)
        print("🎯 최종 성능 요약")
        print("="*80)
        print(f"\n🔺 급증: {surge_detected}/{surge_total} ({surge_detected/surge_total*100 if surge_total > 0 else 0:.1f}%)")
        print(f"🔻 급감: {drop_detected}/{drop_total} ({drop_detected/drop_total*100 if drop_total > 0 else 0:.1f}%)")
        print(f"⚡ 극단값 (300+): {extreme_detected}/{extreme_total} ({extreme_detected/extreme_total*100 if extreme_total > 0 else 0:.1f}%)")
        
        if surge_total > 0 and surge_detected/surge_total >= 0.78:
            print("\n✅ 급증 감지율 78% 이상 달성!")
        elif surge_total > 0:
            print(f"\n⚠️ 급증 감지율 {surge_detected/surge_total*100:.1f}% - 78% 목표 미달")