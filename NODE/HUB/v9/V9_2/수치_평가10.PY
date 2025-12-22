import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta

def evaluate_v9_new_logic():
    """
    🎯 V9-수치형 평가 (Storage/HUB 기반 새 로직)
    
    핵심 발견:
    - CD_M163FSTORAGEUTIL 낮으면 → 급등 가능성
    - HUBROOMTOTAL 낮으면 → 급등 가능성
    - 이 조건에서 MIN 사용하면 대참사!
    
    새 로직:
    1. Storage < 5 → MIN 사용 금지 (AVG 강제)
    2. HUB < 550 → MIN 사용 금지 (AVG 강제)  
    3. 그 외 → 기존 기준값 40 로직
    """
    print("="*80)
    print("🎯 V9-수치형 평가 (Storage/HUB 기반 새 로직)")
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
        'M16A_LFTTOALL', 'M16HUB_M16TOM14_CREATED', 'M16A_ALLTONORTHCNV',
        'M16A_NORTHCNVTOALL', 'M16A_M16ATOM14A', 'M14_TOTALCNVCURRENTQCNT',
        'M16HUB_M16TOM14B_CREATED', 'M16HUB_M14TOM16_CREATED', 'M16A_CURRENTQCNT',
        'M16A_ALLTOSOUTHCNV', 'M14_ALLTOSOUTHCNV', 'M14_ALLTONORTHCNV',
        'M14_CURRENTQCREATED', 'M14_RETURNTOM16', 'M14B_LFTTOALL',
        'M14B_M14BTOM16A', 'M16B_CURRENTQCREATED', 'M16_SENDTOM14',
        'M16HUB_M14TOM16_MESCNT', 'M16HUB_MESCURRENTQCNT',
    ]
    
    TARGET_COL = 'CURRENT_M16A_3F_JOB_2'
    
    def determine_final_prediction(pred_min, pred_max, pred_avg,
                                    pred_min_change, pred_max_change, pred_avg_change,
                                    storage_util, hub_value, slope):
        """
        최종 예측값 결정 (Storage/HUB 기반)
        
        새 로직:
        1. Storage < 5 → 급등 가능성 → MIN 금지
        2. HUB < 550 → 급등 가능성 → MIN 금지
        3. 그 외 → 기준값 40 로직
        """
        # 급락 징후 (기준값 40)
        is_drop_signal = pred_min_change <= -40
        # 급등 징후 (기준값 40)
        is_surge_signal = pred_max_change >= 40
        
        # ========== 새 조건: 급등 가능성 체크 ==========
        # Storage 낮으면 급등 가능성
        storage_low = storage_util < 5
        # HUB 낮으면 급등 가능성
        hub_low = hub_value < 550
        
        # 급등 가능성 있으면 MIN 사용 금지!
        surge_possible = storage_low or hub_low
        
        # ========== 최종 판단 ==========
        # 둘 다 신호 있으면
        if is_drop_signal and is_surge_signal:
            if abs(pred_min_change) > abs(pred_max_change):
                if surge_possible:
                    return pred_avg, pred_avg_change, f"급락신호+급등가능→AVG(Storage:{storage_util:.0f},HUB:{hub_value:.0f})", "AVG"
                return pred_min, pred_min_change, f"급락우세", "MIN"
            else:
                return pred_max, pred_max_change, f"급등우세", "MAX"
        
        # 급락 신호만
        elif is_drop_signal:
            if surge_possible:
                return pred_avg, pred_avg_change, f"급락신호+급등가능→AVG(Storage:{storage_util:.0f},HUB:{hub_value:.0f})", "AVG"
            return pred_min, pred_min_change, f"급락징후(MIN:{pred_min_change:.0f})", "MIN"
        
        # 급등 신호만
        elif is_surge_signal:
            return pred_max, pred_max_change, f"급등징후(MAX:{pred_max_change:.0f})", "MAX"
        
        # 평상시
        else:
            return pred_avg, pred_avg_change, "평상시", "AVG"
    
    # ========== 모델 로드 ==========
    try:
        with open('xgboost_수치형_V9_밸런스.pkl', 'rb') as f:
            model_dict = pickle.load(f)
        models = model_dict['models']
        print("✅ 모델 로드: xgboost_수치형_V9_밸런스.pkl")
    except Exception as e:
        print(f"❌ 모델 없음: {e}")
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
    print("새 로직: Storage<5 또는 HUB<550 → MIN 사용 금지")
    
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
        
        # Storage, HUB 값 (새 로직용)
        storage_util = 999
        hub_value = 999
        
        if 'CD_M163FSTORAGEUTIL' in available_cols:
            storage_util = seq_data['CD_M163FSTORAGEUTIL'].iloc[-1]
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
        
        # ========== 최종 예측 결정 (새 로직) ==========
        final_pred, final_change, reason, used_model = determine_final_prediction(
            pred_min, pred_max, pred_avg,
            pred_change_min, pred_change_max, pred_change_avg,
            storage_util, hub_value, target_slope
        )
        
        # 범위폭
        pred_width = pred_change_max - pred_change_min
        
        # 급등/급락 판정
        is_surge = actual_change >= 50
        is_drop = actual_change <= -50
        
        # 최종예측 오차
        final_error = abs(actual_value - final_pred)
        
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
            '오차≤10': '✅' if final_error <= 10 else '',
            '오차≤20': '✅' if final_error <= 20 else '',
            '오차≤30': '✅' if final_error <= 30 else '',
            '예측범위폭': round(pred_width, 2),
            '급등여부': '📈' if is_surge else '',
            '급락여부': '📉' if is_drop else '',
            'Storage': round(storage_util, 1),
            'HUB': round(hub_value, 0),
            '추세': round(target_slope, 4),
        })
    
    results_df = pd.DataFrame(results)
    
    output_file = '수치형_V9_새로직_평가결과.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 결과 저장: {output_file}")
    
    # ========== 통계 ==========
    print("\n" + "="*80)
    print("📊 평가 통계 (V9-수치형 새 로직)")
    print("="*80)
    
    print(f"총 예측: {len(results_df)}개")
    
    # 최종예측 정확도
    acc_10 = (results_df['오차≤10'] == '✅').sum()
    acc_20 = (results_df['오차≤20'] == '✅').sum()
    acc_30 = (results_df['오차≤30'] == '✅').sum()
    avg_error = results_df['최종_오차'].mean()
    
    print(f"\n🎯 최종_예측 정확도:")
    print(f"  오차 ≤ 10: {acc_10}/{len(results_df)}개 ({acc_10/len(results_df)*100:.1f}%)")
    print(f"  오차 ≤ 20: {acc_20}/{len(results_df)}개 ({acc_20/len(results_df)*100:.1f}%)")
    print(f"  오차 ≤ 30: {acc_30}/{len(results_df)}개 ({acc_30/len(results_df)*100:.1f}%)")
    print(f"  평균 오차: {avg_error:.2f}")
    
    # 사용 모델 분포
    print(f"\n📊 사용 모델 분포:")
    model_counts = results_df['사용모델'].value_counts()
    for model, count in model_counts.items():
        print(f"  {model}: {count}개 ({count/len(results_df)*100:.1f}%)")
    
    # 모델별 정확도
    print(f"\n📊 모델별 오차:")
    for model in ['MIN', 'MAX', 'AVG']:
        subset = results_df[results_df['사용모델'] == model]
        if len(subset) > 0:
            avg_err = subset['최종_오차'].mean()
            print(f"  {model}: 평균오차 {avg_err:.1f}")
    
    # 급등/급락 분석
    print(f"\n📈📉 급등/급락 성능:")
    
    surge_df = results_df[results_df['급등여부'] == '📈']
    drop_df = results_df[results_df['급락여부'] == '📉']
    
    if len(surge_df) > 0:
        surge_error = surge_df['최종_오차'].mean()
        surge_max_used = (surge_df['사용모델'] == 'MAX').sum()
        print(f"  📈 급등 ({len(surge_df)}개): 평균오차 {surge_error:.1f}, MAX사용 {surge_max_used}개")
    
    if len(drop_df) > 0:
        drop_error = drop_df['최종_오차'].mean()
        drop_min_used = (drop_df['사용모델'] == 'MIN').sum()
        print(f"  📉 급락 ({len(drop_df)}개): 평균오차 {drop_error:.1f}, MIN사용 {drop_min_used}개")
    
    # 새 로직 효과 분석
    print(f"\n🆕 새 로직 효과 (급등가능→AVG 강제):")
    avg_forced = results_df[results_df['판단근거'].str.contains('급등가능', na=False)]
    if len(avg_forced) > 0:
        print(f"  AVG 강제 적용: {len(avg_forced)}개")
        print(f"  평균 오차: {avg_forced['최종_오차'].mean():.1f}")
        # 그 중 실제 급등
        avg_forced_surge = avg_forced[avg_forced['급등여부'] == '📈']
        if len(avg_forced_surge) > 0:
            print(f"  → 실제 급등이었던 케이스: {len(avg_forced_surge)}개, 오차 {avg_forced_surge['최종_오차'].mean():.1f}")
    
    return results_df

if __name__ == '__main__':
    print("🚀 V9-수치형 새 로직 평가 시작...\n")
    results = evaluate_v9_new_logic()
    
    if results is not None:
        print(f"\n✅ 평가 완료!")
        print(f"📁 결과: 수치형_V9_새로직_평가결과.csv")