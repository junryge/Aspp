import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta

def evaluate_v9_regression_balanced():
    """
    🎯 V9-수치형 밸런스: MIN/MAX/AVG 구간 예측 평가
    - 최종_예측: 복합 로직으로 MIN/MAX/AVG 중 선택
    """
    print("="*80)
    print("🎯 V9-수치형 밸런스 평가 (최종_예측 포함)")
    print("   ⚖️ 복합 로직: 예측값 + 추세 + HUB 종합 판단")
    print("="*80)
    
    # ========== V8 기존 14개 ==========
    FEATURE_COLS_V8 = {
        'storage': ['M16A_3F_STORAGE_UTIL'],
        'fs_storage': ['CD_M163FSTORAGEUSE', 'CD_M163FSTORAGETOTAL', 'CD_M163FSTORAGEUTIL'],
        'hub': ['HUBROOMTOTAL'],
        'cmd': ['M16A_3F_CMD', 'M16A_6F_TO_HUB_CMD'],
        'inflow': ['M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2', 'M14A_3F_TO_HUB_JOB2'],
        'outflow': ['M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB', 'M16A_3F_TO_M14A_3F_JOB'],
        'maxcapa': ['M16A_6F_LFT_MAXCAPA', 'M16A_2F_LFT_MAXCAPA'],
    }
    
    # ========== V9 신규 20개 ==========
    V9_NEW_COLS = [
        'M16A_LFTTOALL',
        'M16HUB_M16TOM14_CREATED',
        'M16A_ALLTONORTHCNV',
        'M16A_NORTHCNVTOALL',
        'M16A_M16ATOM14A',
        'M14_TOTALCNVCURRENTQCNT',
        'M16HUB_M16TOM14B_CREATED',
        'M16HUB_M14TOM16_CREATED',
        'M16A_CURRENTQCNT',
        'M16A_ALLTOSOUTHCNV',
        'M14_ALLTOSOUTHCNV',
        'M14_ALLTONORTHCNV',
        'M14_CURRENTQCREATED',
        'M14_RETURNTOM16',
        'M14B_LFTTOALL',
        'M14B_M14BTOM16A',
        'M16B_CURRENTQCREATED',
        'M16_SENDTOM14',
        'M16HUB_M14TOM16_MESCNT',
        'M16HUB_MESCURRENTQCNT',
    ]
    
    TARGET_COL = 'CURRENT_M16A_3F_JOB_2'
    
    def determine_final_prediction(pred_min, pred_max, pred_avg, 
                                    pred_min_change, pred_max_change, pred_avg_change):
        """
        최종 예측값 결정 (데이터 검증 기반)
        
        검증 결과:
        - 급등 시: MAX 오차 19.7 (최적)
        - 급락 시: MIN 오차 15.0 (최적)
        - 평상시: AVG 오차 11.6 (최적)
        
        기준:
        - 예측_MIN변화량 <= -40 → 급락 징후 → MIN 사용
        - 예측_MAX변화량 >= 40 → 급등 징후 → MAX 사용
        - 그 외 → 평상시 → AVG 사용
        
        Returns:
            (최종_예측값, 최종_변화량, 판단근거, 사용모델)
        """
        # 급락 징후: MIN변화량 <= -40
        is_drop_signal = pred_min_change <= -40
        
        # 급등 징후: MAX변화량 >= 40
        is_surge_signal = pred_max_change >= 40
        
        # 둘 다 있으면 더 강한 쪽 선택
        if is_drop_signal and is_surge_signal:
            # 절대값 비교
            if abs(pred_min_change) > abs(pred_max_change):
                return pred_min, pred_min_change, f"급락우세(MIN:{pred_min_change:.0f},MAX:{pred_max_change:.0f})", "MIN"
            else:
                return pred_max, pred_max_change, f"급등우세(MIN:{pred_min_change:.0f},MAX:{pred_max_change:.0f})", "MAX"
        
        # 급락 징후만
        elif is_drop_signal:
            return pred_min, pred_min_change, f"급락징후(MIN변화량:{pred_min_change:.0f})", "MIN"
        
        # 급등 징후만
        elif is_surge_signal:
            return pred_max, pred_max_change, f"급등징후(MAX변화량:{pred_max_change:.0f})", "MAX"
        
        # 평상시
        else:
            return pred_avg, pred_avg_change, "평상시", "AVG"
    
    # ========== 모델 로드 ==========
    try:
        with open('xgboost_수치형_V9_밸런스.pkl', 'rb') as f:
            model_dict = pickle.load(f)
        models = model_dict['models']
        print("✅ 모델 로드: xgboost_수치형_V9_밸런스.pkl")
        
        if 'quantile_settings' in model_dict:
            qs = model_dict['quantile_settings']
            print(f"   Quantile 설정: MIN={qs['MIN']}, MAX={qs['MAX']}, AVG={qs['AVG']}")
    except Exception as e:
        print(f"❌ 밸런스 모델 없음: {e}")
        print("   → 기존 모델 시도...")
        try:
            with open('xgboost_수치형_V9.pkl', 'rb') as f:
                model_dict = pickle.load(f)
            models = model_dict['models']
            print("✅ 기존 모델 로드: xgboost_수치형_V9.pkl")
        except:
            print("❌ 모델 없음")
            return None
    
    # ========== 데이터 로드 ==========
    try:
        df = pd.read_csv('test_data.csv', on_bad_lines='skip', encoding='utf-8', low_memory=False)
    except:
        try:
            df = pd.read_csv('test_data.csv', on_bad_lines='skip', encoding='cp949', low_memory=False)
        except:
            df = pd.read_csv('test_data.csv', on_bad_lines='skip', encoding='euc-kr', low_memory=False)
    
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')
    df = df.dropna(subset=[TARGET_COL])
    
    print(f"✅ 데이터 로드: {len(df)}개 행")
    
    available_cols = set(df.columns)
    
    # 시간 처리
    if 'STAT_DT' in df.columns:
        try:
            df['STAT_DT'] = pd.to_datetime(df['STAT_DT'].astype(str), format='%Y%m%d%H%M')
        except:
            base_time = datetime(2024, 1, 1, 0, 0)
            df['STAT_DT'] = [base_time + timedelta(minutes=i) for i in range(len(df))]
    
    results = []
    
    print("\n평가 시작...")
    
    total_samples = len(df) - 40
    
    for idx, i in enumerate(range(30, len(df) - 10)):
        if idx % 500 == 0:
            print(f"  진행: {idx}/{total_samples} ({idx/total_samples*100:.1f}%)")
        
        seq_data = df.iloc[i-30:i].copy()
        seq_target = seq_data[TARGET_COL].values
        
        current_time = seq_data['STAT_DT'].iloc[-1]
        current_value = seq_target[-1]
        prediction_time = current_time + timedelta(minutes=10)
        
        # 실제 10분 후
        future_10min = df[TARGET_COL].iloc[i:i+10].values
        actual_value = df[TARGET_COL].iloc[i+9]
        actual_min = np.min(future_10min)
        actual_max = np.max(future_10min)
        actual_change = actual_value - current_value
        
        # 추세 계산
        target_slope = np.polyfit(np.arange(30), seq_target, 1)[0]
        
        # HUB 값
        hub_value = 999
        if 'HUBROOMTOTAL' in available_cols:
            hub_value = seq_data['HUBROOMTOTAL'].iloc[-1]
        
        # ========== Feature 생성 ==========
        features = {
            'target_mean': np.mean(seq_target),
            'target_std': np.std(seq_target),
            'target_max': np.max(seq_target),
            'target_min': np.min(seq_target),
            'target_last_value': seq_target[-1],
            'target_slope': target_slope,
        }
        
        # V8 기존 컬럼
        for group_name, cols in FEATURE_COLS_V8.items():
            for col in cols:
                if col not in available_cols:
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
                    features[f'{col}_slope'] = np.polyfit(np.arange(30), col_seq, 1)[0]
                else:
                    features[f'{col}_mean'] = np.mean(col_seq)
                    features[f'{col}_last_value'] = col_seq[-1]
                    features[f'{col}_slope'] = np.polyfit(np.arange(30), col_seq, 1)[0]
        
        # V9 신규 컬럼
        for col in V9_NEW_COLS:
            if col not in available_cols:
                continue
            
            col_seq = seq_data[col].values
            features[f'{col}_mean'] = np.mean(col_seq)
            features[f'{col}_std'] = np.std(col_seq)
            features[f'{col}_max'] = np.max(col_seq)
            features[f'{col}_min'] = np.min(col_seq)
            features[f'{col}_last_value'] = col_seq[-1]
            features[f'{col}_slope'] = np.polyfit(np.arange(30), col_seq, 1)[0]
        
        # 특수 Feature
        if all(col in available_cols for col in ['CD_M163FSTORAGEUSE', 'CD_M163FSTORAGETOTAL', 'CD_M163FSTORAGEUTIL']):
            use_seq = seq_data['CD_M163FSTORAGEUSE'].values
            total_seq = seq_data['CD_M163FSTORAGETOTAL'].values
            util_seq = seq_data['CD_M163FSTORAGEUTIL'].values
            features['storage_use_rate'] = (use_seq[-1] - use_seq[0]) / 30
            features['storage_remaining'] = total_seq[-1] - use_seq[-1]
            features['storage_util_last'] = util_seq[-1]
            features['storage_util_high'] = 1 if util_seq[-1] >= 7 else 0
            features['storage_util_critical'] = 1 if util_seq[-1] >= 10 else 0
        
        if 'HUBROOMTOTAL' in available_cols:
            hub_seq = seq_data['HUBROOMTOTAL'].values
            hub_last = hub_seq[-1]
            features['hub_critical'] = 1 if hub_last < 590 else 0
            features['hub_high'] = 1 if hub_last < 610 else 0
            features['hub_warning'] = 1 if hub_last < 620 else 0
            features['hub_decrease_rate'] = (hub_seq[0] - hub_last) / 30
        
        # 복합 Feature
        inflow_sum = sum(df[col].iloc[i-1] for col in FEATURE_COLS_V8['inflow'] if col in available_cols)
        outflow_sum = sum(df[col].iloc[i-1] for col in FEATURE_COLS_V8['outflow'] if col in available_cols)
        features['net_flow'] = inflow_sum - outflow_sum
        
        cmd_sum = sum(df[col].iloc[i-1] for col in FEATURE_COLS_V8['cmd'] if col in available_cols)
        features['total_cmd'] = cmd_sum
        features['total_cmd_low'] = 1 if cmd_sum < 220 else 0
        features['total_cmd_very_low'] = 1 if cmd_sum < 200 else 0
        
        features['surge_risk_score'] = (
            features.get('hub_high', 0) * 3 +
            features.get('storage_util_critical', 0) * 2 +
            features.get('total_cmd_low', 0) * 1 +
            features.get('storage_util_high', 0) * 1
        )
        
        # V9 Boolean
        if 'M16A_LFTTOALL' in available_cols:
            lft = df['M16A_LFTTOALL'].iloc[i-1]
            features['lfttoall_high'] = 1 if lft >= 100 else 0
            features['lfttoall_critical'] = 1 if lft >= 150 else 0
        
        if 'M16HUB_M16TOM14_CREATED' in available_cols:
            m16tom14 = df['M16HUB_M16TOM14_CREATED'].iloc[i-1]
            features['m16tom14_high'] = 1 if m16tom14 >= 50 else 0
            features['m16tom14_critical'] = 1 if m16tom14 >= 80 else 0
        
        if 'M16HUB_M14TOM16_CREATED' in available_cols:
            m14tom16 = df['M16HUB_M14TOM16_CREATED'].iloc[i-1]
            features['m14tom16_high'] = 1 if m14tom16 >= 50 else 0
            features['m14tom16_critical'] = 1 if m14tom16 >= 80 else 0
        
        X_pred = pd.DataFrame([features])
        
        # ========== 예측 ==========
        pred_change_min = models[0].predict(X_pred)[0]
        pred_change_max = models[1].predict(X_pred)[0]
        pred_change_avg = models[2].predict(X_pred)[0]
        
        pred_min = current_value + pred_change_min
        pred_max = current_value + pred_change_max
        pred_avg = current_value + pred_change_avg
        
        # ========== 최종 예측 결정 ==========
        final_pred, final_change, reason, used_model = determine_final_prediction(
            pred_min, pred_max, pred_avg,
            pred_change_min, pred_change_max, pred_change_avg
        )
        
        # 범위 내 포함 여부
        in_range = (pred_min <= actual_max and pred_max >= actual_min)
        
        # 범위폭
        pred_width = pred_change_max - pred_change_min
        actual_width = actual_max - actual_min
        
        # 급등/급락 판정
        is_surge = actual_change >= 50
        is_drop = actual_change <= -50
        
        # 감지 여부
        surge_detected = pred_change_max >= 40 if is_surge else None
        drop_detected = pred_change_min <= -40 if is_drop else None
        
        # 최종예측 정확도 (오차 범위 내)
        final_error = abs(actual_value - final_pred)
        final_accurate_10 = final_error <= 10
        final_accurate_20 = final_error <= 20
        final_accurate_30 = final_error <= 30
        
        results.append({
            '현재시간': current_time.strftime('%Y-%m-%d %H:%M'),
            '예측시점': prediction_time.strftime('%Y-%m-%d %H:%M'),
            '현재값': round(current_value, 2),
            '실제값': round(actual_value, 2),
            '실제_변화량': round(actual_change, 2),
            '예측_MIN': round(pred_min, 2),
            '예측_MAX': round(pred_max, 2),
            '예측_AVG': round(pred_avg, 2),
            '최종_예측': round(final_pred, 2),
            '최종_변화량': round(final_change, 2),
            '사용모델': used_model,
            '판단근거': reason,
            '최종_오차': round(final_error, 2),
            '오차≤10': '✅' if final_accurate_10 else '',
            '오차≤20': '✅' if final_accurate_20 else '',
            '오차≤30': '✅' if final_accurate_30 else '',
            '범위적중': '✅' if in_range else '❌',
            '예측범위폭': round(pred_width, 2),
            '급등여부': '📈' if is_surge else '',
            '급락여부': '📉' if is_drop else '',
            '급등감지': '✅' if surge_detected == True else ('❌' if surge_detected == False else ''),
            '급락감지': '✅' if drop_detected == True else ('❌' if drop_detected == False else ''),
            '추세': round(target_slope, 4),
            'HUBROOMTOTAL': round(hub_value, 0),
        })
    
    results_df = pd.DataFrame(results)
    
    output_file = '수치형_V9_최종예측_평가결과.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 결과 저장: {output_file}")
    
    # ========== 통계 ==========
    print("\n" + "="*80)
    print("📊 평가 통계 (V9-수치형 최종예측)")
    print("="*80)
    
    print(f"총 예측: {len(results_df)}개")
    
    # ========== 최종예측 정확도 (핵심!) ==========
    print(f"\n" + "="*60)
    print(f"🎯 최종_예측 정확도 (핵심 지표)")
    print("="*60)
    
    acc_10 = (results_df['오차≤10'] == '✅').sum()
    acc_20 = (results_df['오차≤20'] == '✅').sum()
    acc_30 = (results_df['오차≤30'] == '✅').sum()
    
    print(f"  오차 ≤ 10: {acc_10}/{len(results_df)}개 ({acc_10/len(results_df)*100:.1f}%)")
    print(f"  오차 ≤ 20: {acc_20}/{len(results_df)}개 ({acc_20/len(results_df)*100:.1f}%)")
    print(f"  오차 ≤ 30: {acc_30}/{len(results_df)}개 ({acc_30/len(results_df)*100:.1f}%)")
    
    avg_error = results_df['최종_오차'].mean()
    median_error = results_df['최종_오차'].median()
    print(f"\n  평균 오차: {avg_error:.2f}")
    print(f"  중앙값 오차: {median_error:.2f}")
    
    # 사용 모델 분포
    print(f"\n📊 사용 모델 분포:")
    model_counts = results_df['사용모델'].value_counts()
    for model, count in model_counts.items():
        print(f"  {model}: {count}개 ({count/len(results_df)*100:.1f}%)")
    
    # 모델별 정확도
    print(f"\n📊 모델별 정확도 (오차 ≤ 20):")
    for model in ['MIN', 'MAX', 'AVG']:
        subset = results_df[results_df['사용모델'] == model]
        if len(subset) > 0:
            acc = (subset['오차≤20'] == '✅').sum() / len(subset) * 100
            avg_err = subset['최종_오차'].mean()
            print(f"  {model}: {acc:.1f}% (평균오차: {avg_err:.1f})")
    
    # 범위 적중률
    in_range_count = (results_df['범위적중'] == '✅').sum()
    print(f"\n🎯 범위 적중률: {in_range_count}/{len(results_df)}개 ({in_range_count/len(results_df)*100:.1f}%)")
    
    # ========== 급등/급락 감지 ==========
    print(f"\n" + "="*60)
    print(f"📈📉 급등/급락 감지 성능")
    print("="*60)
    
    # 급등 감지
    surge_samples = results_df[results_df['급등여부'] == '📈']
    if len(surge_samples) > 0:
        surge_detected = (surge_samples['급등감지'] == '✅').sum()
        print(f"📈 급등 (+50 이상): {surge_detected}/{len(surge_samples)}개 감지 ({surge_detected/len(surge_samples)*100:.1f}%)")
    
    # 급락 감지
    drop_samples = results_df[results_df['급락여부'] == '📉']
    if len(drop_samples) > 0:
        drop_detected = (drop_samples['급락감지'] == '✅').sum()
        print(f"📉 급락 (-50 이하): {drop_detected}/{len(drop_samples)}개 감지 ({drop_detected/len(drop_samples)*100:.1f}%)")
    
    # ========== 판단근거별 분석 ==========
    print(f"\n📊 판단근거별 분포:")
    reason_counts = results_df['판단근거'].value_counts().head(10)
    for reason, count in reason_counts.items():
        print(f"  {reason}: {count}개")
    
    # ========== 샘플 출력 ==========
    print(f"\n📋 샘플 데이터 (급등/급락 각 5개):")
    
    print(f"\n[급등 샘플]")
    surge_sample = results_df[results_df['급등여부'] == '📈'].head(5)
    if len(surge_sample) > 0:
        display_cols = ['현재시간', '실제_변화량', '최종_예측', '최종_변화량', '사용모델', '판단근거', '최종_오차']
        print(surge_sample[display_cols].to_string(index=False))
    
    print(f"\n[급락 샘플]")
    drop_sample = results_df[results_df['급락여부'] == '📉'].head(5)
    if len(drop_sample) > 0:
        display_cols = ['현재시간', '실제_변화량', '최종_예측', '최종_변화량', '사용모델', '판단근거', '최종_오차']
        print(drop_sample[display_cols].to_string(index=False))
    
    return results_df

if __name__ == '__main__':
    print("🚀 V9-수치형 최종예측 평가 시작...\n")
    results = evaluate_v9_regression_balanced()
    
    if results is not None:
        print(f"\n✅ 평가 완료!")
        print(f"📁 결과: 수치형_V9_최종예측_평가결과.csv")