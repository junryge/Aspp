import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta

def evaluate_v9_regression():
    """
    🎯 V9-수치형: MIN/MAX 구간 예측 평가 (단순화 버전)
    """
    print("="*80)
    print("🎯 V9-수치형 평가 (단순화 버전)")
    print("="*80)
    
    # V8 기존 14개
    FEATURE_COLS_V8 = {
        'storage': ['M16A_3F_STORAGE_UTIL'],
        'fs_storage': ['CD_M163FSTORAGEUSE', 'CD_M163FSTORAGETOTAL', 'CD_M163FSTORAGEUTIL'],
        'hub': ['HUBROOMTOTAL'],
        'cmd': ['M16A_3F_CMD', 'M16A_6F_TO_HUB_CMD'],
        'inflow': ['M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2', 'M14A_3F_TO_HUB_JOB2'],
        'outflow': ['M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB', 'M16A_3F_TO_M14A_3F_JOB'],
        'maxcapa': ['M16A_6F_LFT_MAXCAPA', 'M16A_2F_LFT_MAXCAPA'],
    }
    
    # V9 신규 20개
    V9_ESSENTIAL = [
        'M16A_LFTTOALL',
        'M16HUB_M16TOM14_CREATED',
        'M16A_ALLTONORTHCNV',
        'M16A_NORTHCNVTOALL',
        'M16A_M16ATOM14A',
        'M14_TOTALCNVCURRENTQCNT',
        'M16HUB_M16TOM14B_CREATED',
        'M16HUB_M14TOM16_CREATED',
    ]
    
    V9_RECOMMENDED = [
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
    
    V9_NEW_COLS = V9_ESSENTIAL + V9_RECOMMENDED
    
    TARGET_COL = 'CURRENT_M16A_3F_JOB_2'
    
    # 모델 로드
    try:
        with open('xgboost_수치형_V9_Simple.pkl', 'rb') as f:
            model_dict = pickle.load(f)
        models = model_dict['models']
        print("✅ 모델 로드: xgboost_수치형_V9_Simple.pkl")
    except Exception as e:
        print(f"❌ 모델 없음: {e}")
        return None
    
    # 데이터 로드
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
        
        # 실제 10분 후 구간
        future_10min = df[TARGET_COL].iloc[i:i+10].values
        actual_value = df[TARGET_COL].iloc[i+9]  # 🔥 10분 후 실제값
        actual_min = np.min(future_10min)
        actual_max = np.max(future_10min)
        actual_change = actual_value - current_value
        
        # 🔥 Feature 생성 (단순화 - 핵심만)
        features = {
            'target_mean': np.mean(seq_target),
            'target_std': np.std(seq_target),
            'target_max': np.max(seq_target),
            'target_min': np.min(seq_target),
            'target_last_value': seq_target[-1],
            'target_slope': np.polyfit(np.arange(30), seq_target, 1)[0],
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
        
        # FS Storage 특수 Feature
        if all(col in available_cols for col in ['CD_M163FSTORAGEUSE', 'CD_M163FSTORAGETOTAL', 'CD_M163FSTORAGEUTIL']):
            use_seq = seq_data['CD_M163FSTORAGEUSE'].values
            total_seq = seq_data['CD_M163FSTORAGETOTAL'].values
            util_seq = seq_data['CD_M163FSTORAGEUTIL'].values
            features['storage_use_rate'] = (use_seq[-1] - use_seq[0]) / 30
            features['storage_remaining'] = total_seq[-1] - use_seq[-1]
            features['storage_util_last'] = util_seq[-1]
            features['storage_util_high'] = 1 if util_seq[-1] >= 7 else 0
            features['storage_util_critical'] = 1 if util_seq[-1] >= 10 else 0
        
        # HUBROOMTOTAL 특수 Feature
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
        
        # 예측
        pred_change_min = models[0].predict(X_pred)[0]
        pred_change_max = models[1].predict(X_pred)[0]
        pred_change_avg = models[2].predict(X_pred)[0]  # 🔥 AVG 추가
        
        pred_min = current_value + pred_change_min
        pred_max = current_value + pred_change_max
        pred_avg = current_value + pred_change_avg  # 🔥 AVG 추가
        
        # 범위 내 포함 여부
        in_range = (pred_min <= actual_max and pred_max >= actual_min)
        
        # 범위폭
        pred_width = pred_change_max - pred_change_min
        actual_width = actual_max - actual_min
        
        results.append({
            '현재시간': current_time.strftime('%Y-%m-%d %H:%M'),
            '예측시점': prediction_time.strftime('%Y-%m-%d %H:%M'),
            '현재값': round(current_value, 2),
            '실제값': round(actual_value, 2),  # 🔥 10분 후 실제값
            '실제_MIN': round(actual_min, 2),
            '실제_MAX': round(actual_max, 2),
            '실제_변화량': round(actual_change, 2),
            '예측_MIN': round(pred_min, 2),
            '예측_MAX': round(pred_max, 2),
            '예측_AVG': round(pred_avg, 2),  # 🔥 AVG 추가
            '예측_MIN변화량': round(pred_change_min, 2),
            '예측_MAX변화량': round(pred_change_max, 2),
            '예측_AVG변화량': round(pred_change_avg, 2),  # 🔥 AVG 추가
            '범위적중': '✅' if in_range else '❌',
            '예측범위폭': round(pred_width, 2),
            '실제범위폭': round(actual_width, 2),
            'HUBROOMTOTAL': round(features.get('HUBROOMTOTAL_last_value', 0), 0),
        })
    
    results_df = pd.DataFrame(results)
    
    output_file = '수치형_V9_Simple_평가결과.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 결과 저장: {output_file}")
    
    # 통계
    print("\n" + "="*80)
    print("📊 평가 통계 (V9-수치형 단순화)")
    print("="*80)
    
    print(f"총 예측: {len(results_df)}개")
    
    # 범위 적중률
    in_range_count = (results_df['범위적중'] == '✅').sum()
    print(f"\n🎯 범위 적중률: {in_range_count}/{len(results_df)}개 ({in_range_count/len(results_df)*100:.1f}%)")
    
    # 범위폭 분석
    print(f"\n📏 예측 범위폭:")
    print(f"  - 평균: {results_df['예측범위폭'].mean():.1f}")
    print(f"  - 중앙값: {results_df['예측범위폭'].median():.1f}")
    print(f"  - 최소: {results_df['예측범위폭'].min():.1f}")
    print(f"  - 최대: {results_df['예측범위폭'].max():.1f}")
    
    print(f"\n📏 실제 범위폭:")
    print(f"  - 평균: {results_df['실제범위폭'].mean():.1f}")
    print(f"  - 중앙값: {results_df['실제범위폭'].median():.1f}")
    
    # 급증/급락 감지
    print(f"\n🔥 급증/급락 감지:")
    
    large_increase = results_df[results_df['실제_변화량'] >= 50]
    if len(large_increase) > 0:
        detected = large_increase[large_increase['예측_MAX변화량'] >= 40]
        print(f"  - 급증(+50) 감지: {len(detected)}/{len(large_increase)}개 ({len(detected)/len(large_increase)*100:.1f}%)")
    
    large_decrease = results_df[results_df['실제_변화량'] <= -50]
    if len(large_decrease) > 0:
        detected = large_decrease[large_decrease['예측_MIN변화량'] <= -40]
        print(f"  - 급락(-50) 감지: {len(detected)}/{len(large_decrease)}개 ({len(detected)/len(large_decrease)*100:.1f}%)")
    
    # 범위폭별 적중률
    print(f"\n📊 범위폭별 적중률:")
    for width_threshold in [20, 30, 40, 50]:
        subset = results_df[results_df['예측범위폭'] <= width_threshold]
        if len(subset) > 0:
            hit_rate = (subset['범위적중'] == '✅').sum() / len(subset) * 100
            print(f"  - 범위폭 ≤{width_threshold}: {len(subset)}개 중 {hit_rate:.1f}% 적중")
    
    return results_df

if __name__ == '__main__':
    print("🚀 V9-수치형 평가 시작 (단순화)...\n")
    results = evaluate_v9_regression()
    
    if results is not None:
        print(f"\n✅ 평가 완료!")
        print(f"📁 결과: 수치형_V9_Simple_평가결과.csv")