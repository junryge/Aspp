"""
ExtremeNet V6.5 평가 - data/20250732_to20250806.csv
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
import os
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

print("="*80)
print("ExtremeNet V6.5 평가")
print("="*80)

# 특성 컬럼
feature_columns = [
    'TOTALCNT', 'M14AM14B', 'M14AM14BSUM', 'M14AM10A', 'M14AM16',
    'RATIO', 'GOLDEN', 'SIGNAL_ZONE', 'PRE_SIGNAL', 'DANGER_ZONE',
    'PATTERN', 'TREND',
    'MA_5', 'MA_10', 'MA_20',
    'STD_5', 'STD_10', 'STD_20',
    'M14B_MA_5', 'M14B_MA_10', 'M14B_MA_20',
    'CHANGE_1', 'CHANGE_5', 'CHANGE_10',
    'M14B_CHANGE_1', 'M14B_CHANGE_5', 'M14B_CHANGE_10',
    'RISE_COUNT'
]

# 1. 모델 로드
print("\n모델 로딩...")
model = tf.keras.models.load_model(
    'models_final/extremenet_v65_final.keras',
    safe_mode=False  # Lambda 레이어 허용
)
print("✓ 모델 로드 완료")

# 2. 스케일러 로드
print("\n스케일러 로딩...")
with open('scalers_final/scaler_X.pkl', 'rb') as f:
    scaler_X = pickle.load(f)
with open('scalers_final/scaler_y.pkl', 'rb') as f:
    scaler_y = pickle.load(f)
print("✓ 스케일러 로드 완료")

# 3. 데이터 로드
print("\n데이터 로딩...")
data_path = 'data/20250732_to20250806.csv'
if not os.path.exists(data_path):
    # 대체 경로들
    for alt_path in ['20250732_to20250806.csv', 'data/20240201_TO_202507281705.csv', '20240201_TO_202507281705.csv']:
        if os.path.exists(alt_path):
            data_path = alt_path
            break

df = pd.read_csv(data_path)
df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M', errors='coerce')
df = df.sort_values('CURRTIME').reset_index(drop=True)
df = df[df['TOTALCNT'] > 0].reset_index(drop=True)
print(f"✓ 데이터: {len(df)}행")

# 4. 특성 생성
print("\n특성 생성...")
df['RATIO'] = df['M14AM14B'] / (df['M14AM10A'] + 1)
df['GOLDEN'] = ((df['M14AM14B'] > 300) & (df['M14AM10A'] < 80)).astype(int)
df['SIGNAL_ZONE'] = ((df['TOTALCNT'] >= 1651) & (df['TOTALCNT'] <= 1699)).astype(int)
df['PRE_SIGNAL'] = ((df['TOTALCNT'] >= 1600) & (df['TOTALCNT'] < 1651)).astype(int)
df['DANGER_ZONE'] = (df['TOTALCNT'] >= 1700).astype(int)

for w in [5, 10, 20]:
    df[f'MA_{w}'] = df['TOTALCNT'].rolling(w, min_periods=1).mean()
    df[f'STD_{w}'] = df['TOTALCNT'].rolling(w, min_periods=1).std().fillna(0)
    df[f'M14B_MA_{w}'] = df['M14AM14B'].rolling(w, min_periods=1).mean()

for lag in [1, 5, 10]:
    df[f'CHANGE_{lag}'] = df['TOTALCNT'].diff(lag).fillna(0)
    df[f'M14B_CHANGE_{lag}'] = df['M14AM14B'].diff(lag).fillna(0)

df['RISE'] = (df['TOTALCNT'] > df['TOTALCNT'].shift(1)).astype(int)
df['RISE_COUNT'] = df['RISE'].rolling(10, min_periods=1).sum()

df['PATTERN'] = 0
signal_mask = ((df['TOTALCNT'] >= 1651) & (df['TOTALCNT'] <= 1699))
high_m14b = df['M14AM14B'].rolling(100, min_periods=1).mean() > 380
df.loc[signal_mask, 'PATTERN'] = 1
df.loc[high_m14b, 'PATTERN'] = 2

df['TREND'] = 1
for i in range(20, len(df)):
    recent = df['TOTALCNT'].iloc[i-20:i].values
    if len(recent) > 1:
        slope = np.polyfit(range(len(recent)), recent, 1)[0]
        if slope > 5:
            df.loc[i, 'TREND'] = 2
        elif slope < -5:
            df.loc[i, 'TREND'] = 0

print("✓ 특성 생성 완료")

# 5. 예측
print("\n예측 시작...")
results = []
seq_len = 100
pred_len = 10

for i in range(len(df) - seq_len - pred_len + 1):
    # 100분 데이터
    seq_data = df.iloc[i:i+seq_len]
    current_time = seq_data['CURRTIME'].iloc[-1]
    target_time = current_time + timedelta(minutes=10)
    
    # 실제값 (10분 후)
    if i+seq_len+pred_len-1 < len(df):
        actual_value = df['TOTALCNT'].iloc[i+seq_len+pred_len-1]
    else:
        actual_value = np.nan
    
    # 특성 추출
    X = seq_data[feature_columns].values
    X_scaled = scaler_X.transform(X.reshape(-1, len(feature_columns)))
    X_scaled = X_scaled.reshape(1, seq_len, len(feature_columns))
    
    # 예측
    preds = model.predict(X_scaled, verbose=0)
    y_pred_scaled = preds[0][0, 0]
    y_pred = scaler_y.inverse_transform([[y_pred_scaled]])[0, 0]
    
    # 일별 최대값
    current_date = current_time.date()
    day_data = df[df['CURRTIME'].dt.date == current_date]
    daily_max = day_data['TOTALCNT'].max()
    
    # 결과 저장
    results.append({
        'DateTime': current_time.strftime('%Y-%m-%d %H:%M'),
        'TargetDateTime': target_time.strftime('%Y-%m-%d %H:%M'),
        'DailyMax': daily_max,
        'ActualValue': actual_value,
        'PredictedValue': y_pred
    })
    
    if (i+1) % 100 == 0:
        print(f"  진행: {i+1}/{len(df)-seq_len-pred_len+1}")

print(f"✓ 예측 완료: {len(results)}개")

# 6. 결과 저장
results_df = pd.DataFrame(results)
results_df.to_csv('prediction_results.csv', index=False, encoding='utf-8-sig')
print("\n결과 저장: prediction_results.csv")

# 7. 샘플 출력
print("\n[결과 샘플]")
print(results_df.head(10))

# 8. 평가 메트릭 (실제값이 있는 경우만)
valid_results = results_df.dropna(subset=['ActualValue'])
if len(valid_results) > 0:
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(valid_results['ActualValue'], valid_results['PredictedValue'])
    print(f"\nMAE: {mae:.2f}")

print("\n완료!")