import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. 데이터 로드
# 업로드하신 파일명을 지정합니다.
file_path = 'M14_Q_20241201_20251201610.CSV'
df = pd.read_csv(file_path)

# 2. 전처리 (Preprocessing)
# CURRTIME을 날짜/시간 형식으로 변환 (YYYYMMDDHHMM 형식 가정)
df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M')
df = df.sort_values('CURRTIME')

# 결측치 처리 (앞뒤 값으로 채움)
df = df.ffill().bfill()

# 3. 철철이 분석 (계절성/시간 패턴 반영)
# 시간, 분, 요일 정보를 추출하여 모델이 주기성을 학습하도록 함
df['Hour'] = df['CURRTIME'].dt.hour
df['Minute'] = df['CURRTIME'].dt.minute
df['DayOfWeek'] = df['CURRTIME'].dt.dayofweek

# 4. 시퀀스 280분 분석 (Feature Engineering)
target_col = 'TOTALCNT'

# (1) 과거 특정 시점의 데이터 (Lag Features)
# 직전, 10분전, 1시간전, 280분전 데이터를 변수로 추가
lags = [1, 10, 60, 280]
for lag in lags:
    df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)

# (2) 280분간의 흐름 요약 (Rolling Window Statistics)
# 최근 280분의 평균, 표준편차 등을 변수로 추가하여 '시퀀스'의 상태를 반영
window_size = 280
df[f'{target_col}_roll_mean_{window_size}'] = df[target_col].rolling(window=window_size).mean()
df[f'{target_col}_roll_std_{window_size}'] = df[target_col].rolling(window=window_size).std()
df[f'{target_col}_roll_max_{window_size}'] = df[target_col].rolling(window=window_size).max()
df[f'{target_col}_roll_min_{window_size}'] = df[target_col].rolling(window=window_size).min()

# 5. 타겟 생성 (10분 후 예측)
# 현재 시점(t)의 데이터로 10분 뒤(t+10)의 TOTALCNT를 예측해야 하므로
# 타겟 컬럼을 -10만큼 shift 하여 생성합니다.
df['Target'] = df[target_col].shift(-10)

# 시프트로 인해 생긴 결측치(NaN) 제거
df_model = df.dropna()

# 6. 학습/테스트 데이터 분리
# 시계열 데이터이므로 랜덤 섞기(Shuffle) 대신 시간 순서대로 분리합니다. (마지막 20%를 테스트용으로 사용)
train_size = int(len(df_model) * 0.8)
train = df_model.iloc[:train_size]
test = df_model.iloc[train_size:]

# 학습에 사용할 변수 선택 (날짜와 타겟 제외)
features = [c for c in df_model.columns if c not in ['CURRTIME', 'Target']]
X_train, y_train = train[features], train['Target']
X_test, y_test = test[features], test['Target']

# 7. XGBoost 모델 학습 (XGBOOTING)
print("모델 학습을 시작합니다...")
model = xgb.XGBRegressor(
    n_estimators=500,       # 트리 개수
    learning_rate=0.05,     # 학습률
    max_depth=6,            # 트리의 깊이
    subsample=0.8,          # 데이터 샘플링 비율
    colsample_bytree=0.8,   # 컬럼 샘플링 비율
    random_state=42,
    n_jobs=-1               # 모든 CPU 코어 사용
)

model.fit(X_train, y_train, 
          eval_set=[(X_train, y_train), (X_test, y_test)],
          early_stopping_rounds=50, # 성능 향상이 없으면 조기 종료
          verbose=100)

# 8. 예측 및 평가
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\n[Evaluation Result]")
print(f"RMSE (평균 오차): {rmse:.4f}")
print(f"R2 Score (설명력): {r2:.4f}")

# 9. 결과 시각화 (마지막 300분)
plt.figure(figsize=(15, 6))
plt.plot(test['CURRTIME'].iloc[-300:], y_test.iloc[-300:].values, label='Actual (10min Later)', color='black')
plt.plot(test['CURRTIME'].iloc[-300:], y_pred[-300:], label='Predicted (XGBoost)', color='red', linestyle='--')
plt.title('TOTALCNT Prediction: Actual vs Predicted (Next 10 min)')
plt.legend()
plt.show()

# 피처 중요도 확인 (어떤 변수가 예측에 가장 중요했는지)
xgb.plot_importance(model, max_num_features=15)
plt.show()
