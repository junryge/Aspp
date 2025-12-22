import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta

def evaluate_combined():
    """
    🎯 범주형 + 수치형 결합 평가
    - 범주형: 급등/급락/평상시 판단
    - 수치형: MIN/MAX/AVG 값 사용
    
    로직:
    - Class 0 (하락) → MIN 사용
    - Class 1 (소폭 증가) → AVG 사용
    - Class 2 (급증) → MAX 사용
    """
    print("="*80)
    print("🎯 범주형 + 수치형 결합 평가")
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
    
    # ========== 모델 로드 ==========
    print("\n[모델 로드]")
    
    # 수치형 모델
    try:
        with open('xgboost_수치형_V9_밸런스.pkl', 'rb') as f:
            num_model_dict = pickle.load(f)
        num_models = num_model_dict['models']
        print("✅ 수치형 모델: xgboost_수치형_V9_밸런스.pkl")
    except:
        print("❌ 수치형 모델 없음")
        return None
    
    # 범주형 모델
    try:
        with open('xgboost_범주형_V9.pkl', 'rb') as f:
            cat_model = pickle.load(f)
        print("✅ 범주형 모델: xgboost_범주형_V9.pkl")
    except:
        print("❌ 범주형 모델 없음")
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
    
    print(f"✅ 데이터: {len(df)}개 행")
    
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
        
        # ========== 수치형 예측 ==========
        pred_change_min = num_models[0].predict(X_pred)[0]
        pred_change_max = num_models[1].predict(X_pred)[0]
        pred_change_avg = num_models[2].predict(X_pred)[0]
        
        pred_min = current_value + pred_change_min
        pred_max = current_value + pred_change_max
        pred_avg = current_value + pred_change_avg
        
        # ========== 범주형 예측 ==========
        cat_pred = cat_model.predict(X_pred)[0]
        cat_proba = cat_model.predict_proba(X_pred)[0]
        
        # ========== 결합: 범주형으로 모델 선택 ==========
        if cat_pred == 0:  # 하락
            final_pred = pred_min
            final_change = pred_change_min
            used_model = "MIN"
            reason = f"범주형=하락(Class0, {cat_proba[0]*100:.0f}%)"
        elif cat_pred == 2:  # 급증
            final_pred = pred_max
            final_change = pred_change_max
            used_model = "MAX"
            reason = f"범주형=급증(Class2, {cat_proba[2]*100:.0f}%)"
        else:  # 소폭 증가
            final_pred = pred_avg
            final_change = pred_change_avg
            used_model = "AVG"
            reason = f"범주형=소폭(Class1, {cat_proba[1]*100:.0f}%)"
        
        # 범위 내 포함 여부
        in_range = (pred_min <= actual_max and pred_max >= actual_min)
        
        # 범위폭
        pred_width = pred_change_max - pred_change_min
        
        # 급등/급락 판정
        is_surge = actual_change >= 50
        is_drop = actual_change <= -50
        
        # 최종예측 정확도
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
            '범주형_Class': cat_pred,
            'Class0_확률': round(cat_proba[0] * 100, 1),
            'Class1_확률': round(cat_proba[1] * 100, 1),
            'Class2_확률': round(cat_proba[2] * 100, 1),
            '최종_오차': round(final_error, 2),
            '오차≤10': '✅' if final_accurate_10 else '',
            '오차≤20': '✅' if final_accurate_20 else '',
            '오차≤30': '✅' if final_accurate_30 else '',
            '범위적중': '✅' if in_range else '❌',
            '예측범위폭': round(pred_width, 2),
            '급등여부': '📈' if is_surge else '',
            '급락여부': '📉' if is_drop else '',
            'HUBROOMTOTAL': round(hub_value, 0),
        })
    
    results_df = pd.DataFrame(results)
    
    output_file = '결합_V9_평가결과.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 결과 저장: {output_file}")
    
    # ========== 통계 ==========
    print("\n" + "="*80)
    print("📊 결합 평가 통계")
    print("="*80)
    
    print(f"총 예측: {len(results_df)}개")
    
    # 전체 정확도
    print(f"\n🎯 전체 정확도:")
    print(f"  평균 오차: {results_df['최종_오차'].mean():.2f}")
    print(f"  중앙값 오차: {results_df['최종_오차'].median():.2f}")
    print(f"  오차≤10: {(results_df['오차≤10']=='✅').sum()}/{len(results_df)}개 ({(results_df['오차≤10']=='✅').mean()*100:.1f}%)")
    print(f"  오차≤20: {(results_df['오차≤20']=='✅').sum()}/{len(results_df)}개 ({(results_df['오차≤20']=='✅').mean()*100:.1f}%)")
    print(f"  오차≤30: {(results_df['오차≤30']=='✅').sum()}/{len(results_df)}개 ({(results_df['오차≤30']=='✅').mean()*100:.1f}%)")
    
    # 사용 모델 분포
    print(f"\n📊 사용 모델 분포:")
    print(results_df['사용모델'].value_counts())
    
    # 급등/급락 분석
    급등 = results_df[results_df['급등여부'] == '📈']
    급락 = results_df[results_df['급락여부'] == '📉']
    
    print(f"\n🔥 급등/급락 감지:")
    
    if len(급등) > 0:
        급등_MAX = 급등[급등['사용모델'] == 'MAX']
        print(f"  📈 급등 ({len(급등)}개):")
        print(f"     MAX 선택: {len(급등_MAX)}/{len(급등)}개 ({len(급등_MAX)/len(급등)*100:.1f}%)")
        print(f"     평균 오차: {급등['최종_오차'].mean():.1f}")
    
    if len(급락) > 0:
        급락_MIN = 급락[급락['사용모델'] == 'MIN']
        print(f"  📉 급락 ({len(급락)}개):")
        print(f"     MIN 선택: {len(급락_MIN)}/{len(급락)}개 ({len(급락_MIN)/len(급락)*100:.1f}%)")
        print(f"     평균 오차: {급락['최종_오차'].mean():.1f}")
    
    # 기준값30 vs 결합 비교용 출력
    print(f"\n" + "="*80)
    print("🆚 비교용 (기준값30 결과 참고)")
    print("="*80)
    print(f"{'항목':<20} {'기준값30':<12} {'결합':<12}")
    print("-"*44)
    print(f"{'전체 오차':<20} {'17.30':<12} {results_df['최종_오차'].mean():<12.2f}")
    print(f"{'오차≤20':<20} {'67.4%':<12} {(results_df['오차≤20']=='✅').mean()*100:<12.1f}%")
    if len(급등) > 0:
        급등_MAX = 급등[급등['사용모델'] == 'MAX']
        print(f"{'급등 감지율':<20} {'75.9%':<12} {len(급등_MAX)/len(급등)*100:<12.1f}%")
        print(f"{'급등 오차':<20} {'28.7':<12} {급등['최종_오차'].mean():<12.1f}")
    if len(급락) > 0:
        급락_MIN = 급락[급락['사용모델'] == 'MIN']
        print(f"{'급락 감지율':<20} {'72.4%':<12} {len(급락_MIN)/len(급락)*100:<12.1f}%")
        print(f"{'급락 오차':<20} {'20.7':<12} {급락['최종_오차'].mean():<12.1f}")
    
    return results_df

if __name__ == '__main__':
    print("🚀 범주형 + 수치형 결합 평가 시작...\n")
    results = evaluate_combined()
    
    if results is not None:
        print(f"\n✅ 평가 완료!")
        print(f"📁 결과: 결합_V9_평가결과.csv")