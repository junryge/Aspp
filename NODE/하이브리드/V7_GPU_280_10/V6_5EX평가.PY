"""
ExtremeNet V6.5 평가
100개 시퀀스 → 10분 후 예측
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from datetime import timedelta
from sklearn.metrics import mean_absolute_error

# weighted_mae_signal
def weighted_mae_signal(y_true, y_pred):
    weights = tf.ones_like(y_true)
    signal_mask = tf.logical_and(y_true >= 0.65, y_true <= 0.75)
    weights = tf.where(signal_mask, 5.0, weights)
    danger_mask = y_true > 0.75
    weights = tf.where(danger_mask, 10.0, weights)
    return tf.reduce_mean(tf.abs(y_true - y_pred) * weights)

# 모델 로드
model = tf.keras.models.load_model(
    'models_final/extremenet_v65_final.keras',
    safe_mode=False,
    custom_objects={'weighted_mae_signal': weighted_mae_signal}
)

# 스케일러 로드
with open('scalers_final/scaler_X.pkl', 'rb') as f:
    scaler_X = pickle.load(f)
with open('scalers_final/scaler_y.pkl', 'rb') as f:
    scaler_y = pickle.load(f)

# 데이터 로드
df = pd.read_csv('data/20250732_to20250806.csv')
df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M')
df = df.sort_values('CURRTIME').reset_index(drop=True)
df = df[df['TOTALCNT'] > 0].reset_index(drop=True)

print(f"데이터: {len(df)}행")

# 특성 생성
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
df['TREND'] = 1

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

# 예측
results = []

# 시퀀스 100개로 10분 후 예측
# 마지막 100개로 예측하려면 len(df)가 최소 110개 필요
for i in range(len(df) - 109):
    # 100개 입력
    X = df[feature_columns].iloc[i:i+100].values
    X_scaled = scaler_X.transform(X.reshape(-1, 28)).reshape(1, 100, 28)
    
    # 예측
    pred = model.predict(X_scaled, verbose=0)
    y_pred = scaler_y.inverse_transform([[pred[0][0,0]]])[0,0]
    
    # 시간
    current = df['CURRTIME'].iloc[i+99]
    target = current + timedelta(minutes=10)
    
    # 실제값
    actual = df['TOTALCNT'].iloc[i+109]
    
    # 일 최대값
    day_max = df[df['CURRTIME'].dt.date == current.date()]['TOTALCNT'].max()
    
    results.append({
        'DateTime': current.strftime('%Y-%m-%d %H:%M'),
        'TargetDateTime': target.strftime('%Y-%m-%d %H:%M'),
        'DailyMax': day_max,
        'ActualValue': actual,
        'PredictedValue': y_pred
    })

# 저장
import pandas as pd
results_df = pd.DataFrame(results)
results_df.to_csv('prediction_results.csv', index=False, encoding='utf-8-sig')

print(f"예측: {len(results)}개")
print(f"MAE: {mean_absolute_error(results_df['ActualValue'], results_df['PredictedValue']):.2f}")
print("\n[샘플]")
print(results_df.head())