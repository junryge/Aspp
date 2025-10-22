import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta

print("="*80)
print("🎯 전체 평가 CSV 생성")
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
    print("✅ 모델 로드 완료")
except:
    print("❌ 모델 파일 없음!")
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

# STAT_DT 처리
if 'STAT_DT' in df.columns:
    try:
        df['STAT_DT'] = pd.to_datetime(df['STAT_DT'].astype(str), format='%Y%m%d%H%M')
    except:
        base_time = datetime(2024, 1, 1, 0, 0)
        df['STAT_DT'] = [base_time + timedelta(minutes=i) for i in range(len(df))]

results = []

print("\n전체 데이터 평가 중...")
total = len(df) - 40

for i in range(30, len(df) - 10):
    if (i - 30) % 5000 == 0:
        print(f"진행: {i-30}/{total} ({(i-30)/total*100:.1f}%)")
    
    seq_data = df.iloc[i-30:i].copy()
    seq_target = seq_data[TARGET_COL].values
    
    current_time = seq_data['STAT_DT'].iloc[-1]
    current_value = seq_target[-1]
    prediction_time = current_time + timedelta(minutes=10)
    
    # 10분 후 실제값
    actual_value = df.iloc[i+10][TARGET_COL]
    actual_time = df.iloc[i+10]['STAT_DT']
    actual_change = actual_value - current_value
    
    # Feature 생성
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
        'target_volatility': np.std(np.diff(seq_target)),
        'target_momentum': np.mean(seq_target[-5:]) - np.mean(seq_target[:5]),
    }
    
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
    
    if 'CD_M163FSTORAGEUSE' in df.columns and 'CD_M163FSTORAGETOTAL' in df.columns and 'CD_M163FSTORAGEUTIL' in df.columns:
        storage_use = seq_data['CD_M163FSTORAGEUSE'].values
        storage_total = seq_data['CD_M163FSTORAGETOTAL'].values
        storage_util = seq_data['CD_M163FSTORAGEUTIL'].values
        
        features['storage_use_rate'] = (storage_use[-1] - storage_use[0]) / 30
        features['storage_remaining'] = storage_total[-1] - storage_use[-1]
        features['storage_util_last'] = storage_util[-1]
        features['storage_util_high'] = 1 if storage_util[-1] >= 7 else 0
        features['storage_util_critical'] = 1 if storage_util[-1] >= 10 else 0
    
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
        '급증': 'O' if is_surge else '',
        '급증감지': 'O' if surge_detected else '',
        '급감': 'O' if is_drop else '',
        '급감감지': 'O' if drop_detected else '',
        '극단값': 'O' if is_extreme else '',
        '극단값감지': 'O' if extreme_detected else '',
    })

results_df = pd.DataFrame(results)

# CSV 저장
output_file = '변화량예측_전체결과.csv'
results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n✅ 결과 저장: {output_file}")
print(f"   총 {len(results_df)}개 예측")

# 통계
print("\n" + "="*80)
print("📊 통계")
print("="*80)

print(f"\n평균 오차:")
print(f"  변화량 오차: {results_df['변화량_오차'].mean():.2f}")
print(f"  예측값 오차: {results_df['예측값_오차'].mean():.2f}")

surge_total = (results_df['급증'] == 'O').sum()
surge_detected = ((results_df['급증'] == 'O') & (results_df['급증감지'] == 'O')).sum()

drop_total = (results_df['급감'] == 'O').sum()
drop_detected = ((results_df['급감'] == 'O') & (results_df['급감감지'] == 'O')).sum()

extreme_total = (results_df['극단값'] == 'O').sum()
extreme_detected = ((results_df['극단값'] == 'O') & (results_df['극단값감지'] == 'O')).sum()

if surge_total > 0:
    print(f"\n🔺 급증: {surge_detected}/{surge_total}개 ({surge_detected/surge_total*100:.1f}%)")
if drop_total > 0:
    print(f"🔻 급감: {drop_detected}/{drop_total}개 ({drop_detected/drop_total*100:.1f}%)")
if extreme_total > 0:
    print(f"⚡ 극단값: {extreme_detected}/{extreme_total}개 ({extreme_detected/extreme_total*100:.1f}%)")

print("\n✅ 완료!")