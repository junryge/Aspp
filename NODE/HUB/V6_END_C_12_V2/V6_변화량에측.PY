import pandas as pd
import numpy as np
import pickle

print("="*80)
print("🔍 실제 예측 검증 - 10분 후 예측값 확인")
print("="*80)

# 모델 로드
try:
    with open('xgboost_변화량예측.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✅ 모델 로드 완료")
except:
    print("❌ 모델 파일이 없습니다!")
    print("   먼저 V7_변화량_학습코드.py를 실행하세요.")
    exit()

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

FEATURE_COLS = {
    'storage': ['M16A_3F_STORAGE_UTIL'],
    'fs_storage': ['CD_M163FSTORAGEUSE', 'CD_M163FSTORAGETOTAL', 'CD_M163FSTORAGEUTIL'],
    'hub': ['HUBROOMTOTAL'],
    'cmd': ['M16A_3F_CMD', 'M16A_6F_TO_HUB_CMD'],
    'inflow': ['M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2', 'M14A_3F_TO_HUB_JOB2'],
    'outflow': ['M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB', 'M16A_3F_TO_M14A_3F_JOB'],
    'maxcapa': ['M16A_6F_LFT_MAXCAPA', 'M16A_2F_LFT_MAXCAPA']
}

print("\n🎯 샘플 20개 직접 검증 (무작위 선택)")
print("-"*80)

# 무작위로 20개 샘플 선택
np.random.seed(42)
test_indices = np.random.choice(range(30, len(df)-10), 20, replace=False)

results = []

for idx, i in enumerate(test_indices):
    seq_target = df[TARGET_COL].iloc[i-30:i].values
    current_value = seq_target[-1]
    
    # 실제 10분 후 값
    actual_10min = df[TARGET_COL].iloc[i+10]
    actual_change = actual_10min - current_value
    
    # Feature 생성 (간소화)
    features = {
        'target_mean': np.mean(seq_target),
        'target_std': np.std(seq_target),
        'target_max': np.max(seq_target),
        'target_min': np.min(seq_target),
        'target_last_value': seq_target[-1],
        'target_last_5_mean': np.mean(seq_target[-5:]),
        'target_slope': np.polyfit(np.arange(30), seq_target, 1)[0],
        'target_from_max': np.max(seq_target) - seq_target[-1],
        'target_decreasing': 1 if np.polyfit(np.arange(30), seq_target, 1)[0] < -1 else 0,
        'target_increasing': 1 if np.polyfit(np.arange(30), seq_target, 1)[0] > 1 else 0,
        'target_volatility': np.std(np.diff(seq_target)) if len(seq_target) > 1 else 0,
        'target_momentum': np.mean(seq_target[-5:]) - np.mean(seq_target[:5]),
    }
    
    seq_data = df.iloc[i-30:i]
    
    # 각 Feature 생성
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
    
    # FS Storage
    if all(c in df.columns for c in ['CD_M163FSTORAGEUSE', 'CD_M163FSTORAGETOTAL', 'CD_M163FSTORAGEUTIL']):
        storage_use = seq_data['CD_M163FSTORAGEUSE'].values
        storage_total = seq_data['CD_M163FSTORAGETOTAL'].values
        storage_util = seq_data['CD_M163FSTORAGEUTIL'].values
        
        features['storage_use_rate'] = (storage_use[-1] - storage_use[0]) / 30 if len(storage_use) > 0 else 0
        features['storage_remaining'] = storage_total[-1] - storage_use[-1]
        features['storage_util_last'] = storage_util[-1]
        features['storage_util_high'] = 1 if storage_util[-1] >= 7 else 0
        features['storage_util_critical'] = 1 if storage_util[-1] >= 10 else 0
    
    # HUBROOMTOTAL
    if 'HUBROOMTOTAL' in df.columns:
        hub_seq = seq_data['HUBROOMTOTAL'].values
        hub_last = hub_seq[-1]
        
        features['hub_critical'] = 1 if hub_last < 590 else 0
        features['hub_high'] = 1 if hub_last < 610 else 0
        features['hub_warning'] = 1 if hub_last < 620 else 0
        features['hub_decrease_rate'] = (hub_seq[0] - hub_last) / 30 if len(hub_seq) > 0 else 0
        
        if 'CD_M163FSTORAGEUTIL' in df.columns:
            storage_util_last = df['CD_M163FSTORAGEUTIL'].iloc[i-1]
            features['hub_storage_risk'] = 1 if (hub_last < 610 and storage_util_last >= 7) else 0
    
    # CMD
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
    
    X_pred = pd.DataFrame([features])
    
    # 예측
    predicted_change = model.predict(X_pred)[0]
    predicted_10min = current_value + predicted_change
    
    # 급증/급감 판정
    is_surge = (current_value < 300 and actual_10min >= 300)
    surge_detected = (current_value < 300 and predicted_10min >= 290)
    
    is_drop = (current_value >= 300 and actual_10min < 300)
    drop_detected = (current_value >= 300 and predicted_10min < 310)
    
    # 오차
    change_error = abs(actual_change - predicted_change)
    value_error = abs(actual_10min - predicted_10min)
    
    results.append({
        '샘플': idx+1,
        '현재값': round(current_value, 1),
        '실제_10분후': round(actual_10min, 1),
        '예측_10분후': round(predicted_10min, 1),
        '실제_변화량': round(actual_change, 1),
        '예측_변화량': round(predicted_change, 1),
        '변화량_오차': round(change_error, 1),
        '예측값_오차': round(value_error, 1),
        '급증': 'O' if is_surge else '-',
        '급증감지': 'O' if (is_surge and surge_detected) else '-',
        '급감': 'O' if is_drop else '-',
        '급감감지': 'O' if (is_drop and drop_detected) else '-'
    })
    
    if idx < 5:  # 처음 5개만 상세 출력
        print(f"\n샘플 {idx+1}:")
        print(f"  현재값: {current_value:.1f}")
        print(f"  실제 10분 후: {actual_10min:.1f} (변화: {actual_change:+.1f})")
        print(f"  예측 10분 후: {predicted_10min:.1f} (변화: {predicted_change:+.1f})")
        print(f"  오차: {value_error:.1f}")
        if is_surge:
            print(f"  ✅ 급증! 감지: {'성공' if surge_detected else '실패'}")
        elif is_drop:
            print(f"  ✅ 급감! 감지: {'성공' if drop_detected else '실패'}")

df_results = pd.DataFrame(results)

print("\n\n" + "="*80)
print("📊 전체 결과")
print("="*80)
print(df_results.to_string(index=False))

print("\n\n" + "="*80)
print("📈 통계")
print("="*80)
print(f"변화량 오차 평균: {df_results['변화량_오차'].mean():.2f}")
print(f"예측값 오차 평균: {df_results['예측값_오차'].mean():.2f}")

surge_count = (df_results['급증'] == 'O').sum()
surge_detected_count = ((df_results['급증'] == 'O') & (df_results['급증감지'] == 'O')).sum()
drop_count = (df_results['급감'] == 'O').sum()
drop_detected_count = ((df_results['급감'] == 'O') & (df_results['급감감지'] == 'O')).sum()

if surge_count > 0:
    print(f"\n급증: {surge_detected_count}/{surge_count}개 감지 ({surge_detected_count/surge_count*100:.1f}%)")
if drop_count > 0:
    print(f"급감: {drop_detected_count}/{drop_count}개 감지 ({drop_detected_count/drop_count*100:.1f}%)")

print("\n✅ 10분 후 예측 검증 완료!")