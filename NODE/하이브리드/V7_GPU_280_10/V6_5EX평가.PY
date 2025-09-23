"""
ExtremeNet V6.5 평가 - 빠른 버전 (배치 예측)
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
import pickle
import os
from datetime import timedelta
import time

print("="*80)
print("ExtremeNet V6.5 평가 - 빠른 버전")
print("대상: data/20250732_to20250806.csv")
print("="*80)

# weighted_mae_signal 함수
def weighted_mae_signal(y_true, y_pred):
    weights = tf.ones_like(y_true)
    signal_mask = tf.logical_and(y_true >= 0.65, y_true <= 0.75)
    weights = tf.where(signal_mask, 5.0, weights)
    danger_mask = y_true > 0.75
    weights = tf.where(danger_mask, 10.0, weights)
    mae = tf.abs(y_true - y_pred)
    weighted_mae = mae * weights
    return tf.reduce_mean(weighted_mae)

# 파일 체크
data_path = 'data/20250732_to20250806.csv'
if not os.path.exists(data_path):
    print(f"❌ {data_path} 파일이 없습니다!")
    exit()

# 모델 로드
print("\n모델 로딩...")
start = time.time()
model = tf.keras.models.load_model(
    'models_final/extremenet_v65_final.keras',
    safe_mode=False,
    custom_objects={'weighted_mae_signal': weighted_mae_signal}
)
print(f"✓ 모델 로드 ({time.time()-start:.1f}초)")

# 스케일러 로드
with open('scalers_final/scaler_X.pkl', 'rb') as f:
    scaler_X = pickle.load(f)
with open('scalers_final/scaler_y.pkl', 'rb') as f:
    scaler_y = pickle.load(f)

# 데이터 로드
df = pd.read_csv(data_path)
df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M')
df = df.sort_values('CURRTIME').reset_index(drop=True)
df = df[df['TOTALCNT'] > 0].reset_index(drop=True)
print(f"✓ 데이터: {len(df)}행")

# 특성 생성
print("\n특성 생성...")
start = time.time()

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
        if slope > 5: df.loc[i, 'TREND'] = 2
        elif slope < -5: df.loc[i, 'TREND'] = 0

print(f"✓ 특성 생성 ({time.time()-start:.1f}초)")

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

# 파라미터
seq_len = 100
pred_len = 10
batch_size = 32  # 배치 크기

# 데이터 체크
if len(df) < seq_len + pred_len:
    print(f"⚠️ 데이터 부족! {seq_len + pred_len}개 필요, 현재 {len(df)}개")
    exit()

# 배치 준비
print("\n배치 준비...")
start = time.time()

X_batch = []
times = []
actual_values = []

for i in range(len(df) - seq_len - pred_len + 1):
    seq_data = df.iloc[i:i+seq_len]
    X = seq_data[feature_columns].values
    X_batch.append(X)
    times.append(seq_data['CURRTIME'].iloc[-1])
    actual_values.append(df['TOTALCNT'].iloc[i+seq_len+pred_len-1])

X_batch = np.array(X_batch)
print(f"✓ 배치 준비: {X_batch.shape} ({time.time()-start:.1f}초)")

# 스케일링
print("\n스케일링...")
start = time.time()
X_batch_flat = X_batch.reshape(-1, len(feature_columns))
X_batch_scaled = scaler_X.transform(X_batch_flat)
X_batch_scaled = X_batch_scaled.reshape(X_batch.shape)
print(f"✓ 스케일링 ({time.time()-start:.1f}초)")

# 배치 예측
print("\n예측 중... (학습 아님, predict만 수행)")
start = time.time()
predictions = []

# 배치 단위로 예측
for i in range(0, len(X_batch_scaled), batch_size):
    batch = X_batch_scaled[i:i+batch_size]
    preds = model.predict(batch, verbose=0)
    y_pred_scaled = preds[0][:, 0]
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    predictions.extend(y_pred)
    
    # 진행 표시
    if (i//batch_size + 1) % 10 == 0:
        print(f"  진행: {i+batch_size}/{len(X_batch_scaled)}")

print(f"✓ 예측 완료: {len(predictions)}개 ({time.time()-start:.1f}초)")

# 결과 정리
results = []
for i, (time_val, actual, pred) in enumerate(zip(times, actual_values, predictions)):
    target_time = time_val + timedelta(minutes=10)
    current_date = time_val.date()
    day_data = df[df['CURRTIME'].dt.date == current_date]
    daily_max = day_data['TOTALCNT'].max()
    
    results.append({
        'DateTime': time_val.strftime('%Y-%m-%d %H:%M'),
        'TargetDateTime': target_time.strftime('%Y-%m-%d %H:%M'),
        'DailyMax': daily_max,
        'ActualValue': actual,
        'PredictedValue': pred
    })

# 저장
results_df = pd.DataFrame(results)
results_df.to_csv('prediction_results.csv', index=False, encoding='utf-8-sig')
print("\n✓ 저장: prediction_results.csv")

# 평가
mae = mean_absolute_error(results_df['ActualValue'], results_df['PredictedValue'])
print(f"MAE: {mae:.2f}")

# 샘플
print("\n[결과 샘플]")
print(results_df[['DateTime', 'ActualValue', 'PredictedValue']].head(5))

print("\n완료!")