import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta

def evaluate_v9_regression():
    """
    🎯 V9-수치형: MIN/MAX/AVG 구간 변화량 평가
    - 실제 데이터 컬럼명 사용!
    - 구간값과 비교 (올바른 평가!)
    """
    print("="*80)
    print("🎯 V9-수치형: MIN/MAX/AVG 구간 변화량 평가")
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
    
    # ========== V9 신규 20개 (실제 컬럼명!) ==========
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
        with open('xgboost_수치형_V9.pkl', 'rb') as f:
            model_dict = pickle.load(f)
        models = model_dict['models']
        print("✅ 모델 로드: xgboost_수치형_V9.pkl (3개 모델)")
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
    
    # V9 컬럼 확인
    v8_count = sum(1 for cols in FEATURE_COLS_V8.values() for col in cols if col in available_cols)
    v9_essential_count = sum(1 for col in V9_ESSENTIAL if col in available_cols)
    v9_recommend_count = sum(1 for col in V9_RECOMMENDED if col in available_cols)
    
    print(f"\n📋 V9 컬럼 확인:")
    print(f"  V8 기존: {v8_count}/14개")
    print(f"  V9 필수(8): {v9_essential_count}개")
    print(f"  V9 권장(12): {v9_recommend_count}개")
    
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
        
        # 실제 구간 MIN/MAX/AVG
        future_10min = df[TARGET_COL].iloc[i:i+10].values
        actual_min = np.min(future_10min)
        actual_max = np.max(future_10min)
        actual_avg = np.mean(future_10min)
        
        actual_change_min = actual_min - current_value
        actual_change_max = actual_max - current_value
        actual_change_avg = actual_avg - current_value
        
        # 10분 후 단일값 (참고용)
        actual_10min_value = df[TARGET_COL].iloc[i+9]
        actual_10min_change = actual_10min_value - current_value
        
        # Feature 생성
        features = {
            'target_mean': np.mean(seq_target),
            'target_std': np.std(seq_target),
            'target_max': np.max(seq_target),
            'target_min': np.min(seq_target),
            'target_last_value': seq_target[-1],
            'target_last_5_mean': np.mean(seq_target[-5:]),
            'target_slope': np.polyfit(np.arange(30), seq_target, 1)[0],
            'target_acceleration': (seq_target[-5:].mean() - seq_target[-10:-5].mean()) / 5,
            'target_is_rising': 1 if seq_target[-1] > seq_target[-5] else 0,
            'target_rapid_rise': 1 if (seq_target[-1] - seq_target[-5] > 10) else 0,
            'target_last_10_mean': np.mean(seq_target[-10:]),
        }
        
        # V8 기존 컬럼 Feature
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
                    features[f'{col}_last_5_mean'] = np.mean(col_seq[-5:])
                    features[f'{col}_slope'] = np.polyfit(np.arange(30), col_seq, 1)[0]
                else:
                    features[f'{col}_mean'] = np.mean(col_seq)
                    features[f'{col}_last_value'] = col_seq[-1]
                    features[f'{col}_slope'] = np.polyfit(np.arange(30), col_seq, 1)[0]
        
        # V9 신규 컬럼 Feature
        for col in V9_NEW_COLS:
            if col not in available_cols:
                continue
            
            col_seq = seq_data[col].values
            features[f'{col}_mean'] = np.mean(col_seq)
            features[f'{col}_std'] = np.std(col_seq)
            features[f'{col}_max'] = np.max(col_seq)
            features[f'{col}_min'] = np.min(col_seq)
            features[f'{col}_last_value'] = col_seq[-1]
            features[f'{col}_last_5_mean'] = np.mean(col_seq[-5:])
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
        
        # V9 핵심 컬럼 Boolean
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
        
        # 예측 (3개 모델)
        pred_change_min = models[0].predict(X_pred)[0]
        pred_change_max = models[1].predict(X_pred)[0]
        pred_change_avg = models[2].predict(X_pred)[0]
        
        pred_min = current_value + pred_change_min
        pred_max = current_value + pred_change_max
        pred_avg = current_value + pred_change_avg
        
        # 구간값과 비교!
        error_min = abs(actual_change_min - pred_change_min)
        error_max = abs(actual_change_max - pred_change_max)
        error_avg = abs(actual_change_avg - pred_change_avg)
        
        # 10분후 값이 예측 범위 내인지
        in_range = pred_min <= actual_10min_value <= pred_max
        
        results.append({
            '현재시간': current_time.strftime('%Y-%m-%d %H:%M'),
            '예측시점': prediction_time.strftime('%Y-%m-%d %H:%M'),
            '현재값': round(current_value, 2),
            '실제_구간MIN': round(actual_min, 2),
            '실제_구간MAX': round(actual_max, 2),
            '실제_구간AVG': round(actual_avg, 2),
            '예측_MIN': round(pred_min, 2),
            '예측_MAX': round(pred_max, 2),
            '예측_AVG': round(pred_avg, 2),
            '실제_변화량_MIN': round(actual_change_min, 2),
            '실제_변화량_MAX': round(actual_change_max, 2),
            '실제_변화량_AVG': round(actual_change_avg, 2),
            '예측_변화량_MIN': round(pred_change_min, 2),
            '예측_변화량_MAX': round(pred_change_max, 2),
            '예측_변화량_AVG': round(pred_change_avg, 2),
            '오차_MIN': round(error_min, 2),
            '오차_MAX': round(error_max, 2),
            '오차_AVG': round(error_avg, 2),
            '10분후_실제값': round(actual_10min_value, 2),
            '10분후_변화량': round(actual_10min_change, 2),
            '범위내': '✅' if in_range else '❌',
            'HUBROOMTOTAL': round(features.get('HUBROOMTOTAL_last_value', 0), 0),
        })
    
    results_df = pd.DataFrame(results)
    
    output_file = '수치형_V9_평가결과.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 결과 저장: {output_file}")
    
    # 통계
    print("\n" + "="*80)
    print("📊 평가 통계 (V9-수치형)")
    print("="*80)
    
    print(f"총 예측: {len(results_df)}개")
    
    # 구간값 오차
    print(f"\n🎯 구간값 예측 성능 (올바른 비교!):")
    for name in ['MIN', 'MAX', 'AVG']:
        avg_error = results_df[f'오차_{name}'].mean()
        print(f"  {name} 평균 오차: {avg_error:.2f}")
    
    # 범위 내 적중률
    in_range_count = (results_df['범위내'] == '✅').sum()
    in_range_rate = in_range_count / len(results_df) * 100
    print(f"\n🎯 10분후 값이 예측 범위 내:")
    print(f"  - 적중: {in_range_count}/{len(results_df)}개 ({in_range_rate:.1f}%)")
    
    # 큰 변화 감지
    print(f"\n🔥 큰 변화 감지 (구간 MAX 기준):")
    
    large_increase = results_df[results_df['실제_변화량_MAX'] >= 50]
    if len(large_increase) > 0:
        detected = large_increase[large_increase['예측_변화량_MAX'] >= 30]
        print(f"  - 큰 상승(+50) 감지: {len(detected)}/{len(large_increase)}개 ({len(detected)/len(large_increase)*100:.1f}%)")
    
    large_decrease = results_df[results_df['실제_변화량_MIN'] <= -50]
    if len(large_decrease) > 0:
        detected = large_decrease[large_decrease['예측_변화량_MIN'] <= -30]
        print(f"  - 큰 하락(-50) 감지: {len(detected)}/{len(large_decrease)}개 ({len(detected)/len(large_decrease)*100:.1f}%)")
    
    # 샘플 출력
    print(f"\n📋 예측 샘플 (처음 10개):")
    print(f"{'현재':>6} {'실MIN':>6} {'실MAX':>6} {'예MIN':>6} {'예MAX':>6} {'10분후':>6} {'범위':>4}")
    print("-"*50)
    for i in range(min(10, len(results_df))):
        row = results_df.iloc[i]
        print(f"{row['현재값']:>6.0f} {row['실제_구간MIN']:>6.0f} {row['실제_구간MAX']:>6.0f} {row['예측_MIN']:>6.1f} {row['예측_MAX']:>6.1f} {row['10분후_실제값']:>6.0f} {row['범위내']:>4}")
    
    return results_df

if __name__ == '__main__':
    print("🚀 V9-수치형 평가 시작...\n")
    results = evaluate_v9_regression()
    
    if results is not None:
        print(f"\n✅ 평가 완료!")
        print(f"📁 결과: 수치형_V9_평가결과.csv")