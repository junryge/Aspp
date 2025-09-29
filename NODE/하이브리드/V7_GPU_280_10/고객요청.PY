# -*- coding: utf-8 -*-
"""
ExtraTrees 모델 분석 - CSV 파일의 실제 날짜 데이터 사용
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (20, 12)

print("="*80)
print("ExtraTrees 모델 분석 - 실제 CSV 데이터 날짜 사용")
print("="*80)

# 1. 데이터 로드
print("\n[1] 데이터 로딩...")
data_file = 'data/20240201_to_202507271705.csv'
df = pd.read_csv(data_file)

print(f"✓ 데이터 파일: {data_file}")
print(f"✓ 전체 행 수: {len(df):,}개")
print(f"✓ 컬럼: {df.columns.tolist()}")

# CSV 파일에서 실제 날짜/시간 컬럼 확인
date_columns = []
for col in df.columns:
    if 'date' in col.lower() or 'time' in col.lower() or 'datetime' in col.lower():
        date_columns.append(col)
        print(f"✓ 날짜 컬럼 발견: {col}")

# 날짜 컬럼이 있으면 사용, 없으면 분 단위로 생성
if date_columns:
    # 실제 날짜 컬럼이 있는 경우
    date_col = date_columns[0]
    df['timestamp'] = pd.to_datetime(df[date_col])
    print(f"✓ 날짜 컬럼 '{date_col}' 사용")
else:
    # 날짜 컬럼이 없으면 분 단위로 가정 (데이터가 1분 간격으로 수집된다고 가정)
    print("⚠️ 날짜 컬럼이 없어 인덱스 기반으로 시간 생성 (1분 간격 가정)")
    # 첫 행을 임의의 시작 시간으로 설정 (실제 데이터에 맞게 조정 필요)
    # 데이터가 분 단위라고 가정
    base_date = datetime(2024, 2, 1, 0, 0)  # 임시 시작일
    df['timestamp'] = [base_date + timedelta(minutes=i) for i in range(len(df))]

# 실제 날짜 범위 확인
actual_start = df['timestamp'].iloc[0]
actual_end = df['timestamp'].iloc[-1]
total_duration = actual_end - actual_start

print(f"\n📅 데이터 실제 기간:")
print(f"  시작: {actual_start}")
print(f"  종료: {actual_end}")
print(f"  총 기간: {total_duration.days}일 ({total_duration.days/30:.1f}개월)")
print(f"  데이터 포인트: {len(df):,}개")

# 2. 원본 코드의 시퀀스 생성 함수 사용
def assign_level(totalcnt_value):
    if totalcnt_value < 1400:
        return 0  # 정상
    elif totalcnt_value < 1700:
        return 1  # 확인
    else:
        return 2  # 위험

def create_sequences_280to10(data, seq_length=280, pred_horizon=10):
    """원본 코드와 동일한 시퀀스 생성"""
    
    print(f"\n시퀀스 생성 중... (280분 → 10분 후)")
    
    feature_cols = ['M14AM14B', 'M14AM10A', 'M14AM16', 'M14AM14BSUM', 
                   'M14AM10ASUM', 'M14AM16SUM', 'M14BM14A', 'M10AM14A', 'M16M14A', 'TOTALCNT']
    
    # 필요한 컬럼이 있는지 확인
    missing_cols = [col for col in feature_cols if col not in data.columns]
    if missing_cols:
        print(f"⚠️ 누락된 컬럼: {missing_cols}")
        # 누락된 컬럼이 있으면 사용 가능한 컬럼만 사용
        feature_cols = [col for col in feature_cols if col in data.columns]
        print(f"✓ 사용 가능한 컬럼: {feature_cols}")
    
    # 파생 변수 생성
    data = data.copy()
    
    if 'M14AM14B' in data.columns and 'M14AM10A' in data.columns:
        data['ratio_M14B_M14A'] = np.clip(data['M14AM14B'] / (data['M14AM10A'] + 1), 0, 1000)
    if 'M14AM14B' in data.columns and 'M14AM16' in data.columns:
        data['ratio_M14B_M16'] = np.clip(data['M14AM14B'] / (data['M14AM16'] + 1), 0, 1000)
    
    if 'TOTALCNT' in data.columns:
        data['totalcnt_change'] = data['TOTALCNT'].diff().fillna(0)
        data['totalcnt_pct_change'] = data['TOTALCNT'].pct_change().fillna(0)
        data['totalcnt_pct_change'] = np.clip(data['totalcnt_pct_change'], -10, 10)
    
    data = data.replace([np.inf, -np.inf], 0)
    data = data.fillna(0)
    
    X_list = []
    y_list = []
    timestamps = []
    
    n_sequences = len(data) - seq_length - pred_horizon + 1
    print(f"✓ 생성 가능한 시퀀스: {n_sequences:,}개")
    
    for i in range(n_sequences):
        if i % 10000 == 0:
            print(f"  진행률: {i/n_sequences*100:.1f}%", end='\r')
        
        start_idx = i
        end_idx = i + seq_length
        seq_data = data.iloc[start_idx:end_idx]
        
        features = []
        
        for col in feature_cols:
            values = seq_data[col].values
            
            features.extend([
                np.mean(values),
                np.std(values) if len(values) > 1 else 0,
                np.min(values),
                np.max(values),
                np.percentile(values, 25),
                np.percentile(values, 50),
                np.percentile(values, 75),
                values[-1],
                values[-1] - values[0],
                np.mean(values[-60:]),
                np.max(values[-60:]),
                np.mean(values[-30:]),
                np.max(values[-30:]),
            ])
            
            if col == 'TOTALCNT':
                features.append(np.sum((values >= 1650) & (values < 1700)))
                features.append(np.sum(values >= 1700))
                features.append(np.max(values[-20:]))
                features.append(np.sum(values < 1400))
                features.append(np.sum((values >= 1400) & (values < 1700)))
                features.append(np.sum(values >= 1700))
                features.append(np.sum((values >= 1651) & (values <= 1682)))
                
                anomaly_values = values[(values >= 1651) & (values <= 1682)]
                features.append(np.max(anomaly_values) if len(anomaly_values) > 0 else 0)
                
                normal_vals = values[values < 1400]
                check_vals = values[(values >= 1400) & (values < 1700)]
                danger_vals = values[values >= 1700]
                
                features.append(np.mean(normal_vals) if len(normal_vals) > 0 else 0)
                features.append(np.mean(check_vals) if len(check_vals) > 0 else 0)
                features.append(np.mean(danger_vals) if len(danger_vals) > 0 else 0)
                
                try:
                    slope = np.polyfit(np.arange(len(values)), values, 1)[0]
                    features.append(np.clip(slope, -100, 100))
                except:
                    features.append(0)
                
                try:
                    recent_slope = np.polyfit(np.arange(60), values[-60:], 1)[0]
                    features.append(np.clip(recent_slope, -100, 100))
                except:
                    features.append(0)
        
        # 파생 변수 추가
        last_idx = end_idx - 1
        if 'ratio_M14B_M14A' in data.columns:
            features.append(np.clip(data['ratio_M14B_M14A'].iloc[last_idx], 0, 1000))
        if 'ratio_M14B_M16' in data.columns:
            features.append(np.clip(data['ratio_M14B_M16'].iloc[last_idx], 0, 1000))
        if 'totalcnt_change' in data.columns:
            features.append(np.clip(data['totalcnt_change'].iloc[last_idx], -1000, 1000))
        if 'totalcnt_pct_change' in data.columns:
            features.append(np.clip(data['totalcnt_pct_change'].iloc[last_idx], -10, 10))
        
        # 타겟
        target_idx = end_idx + pred_horizon - 1
        if target_idx < len(data) and 'TOTALCNT' in data.columns:
            X_list.append(features)
            y_list.append(data['TOTALCNT'].iloc[target_idx])
            timestamps.append(data['timestamp'].iloc[target_idx])
    
    X = np.array(X_list)
    y = np.array(y_list)
    X = np.nan_to_num(X, nan=0.0, posinf=1000.0, neginf=-1000.0)
    
    print(f"\n✓ 시퀀스 생성 완료! Shape: {X.shape}")
    
    return X, y, timestamps

# 3. 시퀀스 생성
print("\n[2] 시퀀스 데이터 생성...")
X, y, timestamps = create_sequences_280to10(df)

# 4. 시계열 순서로 분할 (60:20:20)
print("\n[3] 시계열 순서 유지 데이터 분할...")

n_samples = len(X)
train_size = int(n_samples * 0.6)
val_size = int(n_samples * 0.2)

X_train = X[:train_size]
y_train = y[:train_size]
timestamps_train = timestamps[:train_size]

X_val = X[train_size:train_size+val_size]
y_val = y[train_size:train_size+val_size]
timestamps_val = timestamps[train_size:train_size+val_size]

X_test = X[train_size+val_size:]
y_test = y[train_size+val_size:]
timestamps_test = timestamps[train_size+val_size:]

# 5. 실제 데이터 기간 출력
print("\n" + "="*80)
print("📅 실제 데이터셋 날짜 범위 (CSV 파일 기준)")
print("="*80)

print(f"\n[Train Set] - 60%")
print(f"  시작: {timestamps_train[0]}")
print(f"  종료: {timestamps_train[-1]}")
print(f"  기간: {(timestamps_train[-1] - timestamps_train[0]).days}일")
print(f"  샘플: {len(X_train):,}개")

print(f"\n[Validation Set] - 20%")
print(f"  시작: {timestamps_val[0]}")
print(f"  종료: {timestamps_val[-1]}")
print(f"  기간: {(timestamps_val[-1] - timestamps_val[0]).days}일")
print(f"  샘플: {len(X_val):,}개")

print(f"\n[Test Set] - 20%")
print(f"  시작: {timestamps_test[0]}")
print(f"  종료: {timestamps_test[-1]}")
print(f"  기간: {(timestamps_test[-1] - timestamps_test[0]).days}일")
print(f"  샘플: {len(X_test):,}개")

# 6. 정규화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 7. ExtraTrees 학습 곡선
print("\n[4] ExtraTrees 학습 곡선 생성...")

n_estimators_list = [50, 100, 150, 200, 250, 300]
train_losses = []
val_losses = []

for n_est in n_estimators_list:
    print(f"  n_estimators={n_est} 학습...", end=' ')
    
    model = ExtraTreesRegressor(
        n_estimators=n_est,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    train_mae = mean_absolute_error(y_train, model.predict(X_train_scaled))
    val_mae = mean_absolute_error(y_val, model.predict(X_val_scaled))
    
    train_losses.append(train_mae)
    val_losses.append(val_mae)
    
    print(f"Train: {train_mae:.2f}, Val: {val_mae:.2f}, Gap: {val_mae-train_mae:.2f}")

# 8. 최종 모델 (n_estimators=300)
print("\n[5] 최종 모델 학습 (n_estimators=300)...")

final_model = ExtraTreesRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

final_model.fit(X_train_scaled, y_train)

# 9. 성능 평가
y_train_pred = final_model.predict(X_train_scaled)
y_val_pred = final_model.predict(X_val_scaled)
y_test_pred = final_model.predict(X_test_scaled)

train_mae = mean_absolute_error(y_train, y_train_pred)
val_mae = mean_absolute_error(y_val, y_val_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

train_r2 = r2_score(y_train, y_train_pred)
val_r2 = r2_score(y_val, y_val_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\n" + "="*80)
print("📊 최종 성능")
print("="*80)
print(f"Train MAE: {train_mae:.2f} (R²: {train_r2:.4f})")
print(f"Val MAE: {val_mae:.2f} (R²: {val_r2:.4f})")
print(f"Test MAE: {test_mae:.2f} (R²: {test_r2:.4f})")

print(f"\n🔍 Overfitting 체크:")
print(f"Val-Train Gap: {val_mae-train_mae:.2f}")
print(f"Test-Train Gap: {test_mae-train_mae:.2f}")

if val_mae-train_mae < 30:
    print("✅ Overfitting 상태: 양호")
elif val_mae-train_mae < 50:
    print("⚠️ Overfitting 상태: 주의")
else:
    print("❌ Overfitting 상태: 심각")

# 10. 그래프
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Loss 곡선
ax1 = axes[0, 0]
ax1.plot(n_estimators_list, train_losses, 'b-', marker='o', label='Train MAE')
ax1.plot(n_estimators_list, val_losses, 'r-', marker='s', label='Val MAE')
ax1.set_xlabel('n_estimators')
ax1.set_ylabel('MAE')
ax1.set_title('Learning Curve')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Overfitting Gap
ax2 = axes[0, 1]
gaps = [val_losses[i] - train_losses[i] for i in range(len(n_estimators_list))]
ax2.bar(n_estimators_list, gaps, color=['green' if g < 30 else 'orange' if g < 50 else 'red' for g in gaps])
ax2.axhline(y=30, color='orange', linestyle='--', alpha=0.5)
ax2.axhline(y=50, color='red', linestyle='--', alpha=0.5)
ax2.set_xlabel('n_estimators')
ax2.set_ylabel('Val MAE - Train MAE')
ax2.set_title('Overfitting Gap')
ax2.grid(True, alpha=0.3)

# Test 예측
ax3 = axes[1, 0]
sample_idx = np.random.choice(len(y_test), min(1000, len(y_test)), replace=False)
ax3.scatter(y_test[sample_idx], y_test_pred[sample_idx], alpha=0.5, s=10)
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
ax3.set_xlabel('Actual')
ax3.set_ylabel('Predicted')
ax3.set_title(f'Test Predictions (R²={test_r2:.3f})')
ax3.grid(True, alpha=0.3)

# 성능 요약
ax4 = axes[1, 1]
datasets = ['Train', 'Val', 'Test']
maes = [train_mae, val_mae, test_mae]
r2s = [train_r2, val_r2, test_r2]
x = np.arange(3)
ax4.bar(x - 0.2, maes, 0.4, label='MAE', color='blue', alpha=0.7)
ax4.bar(x + 0.2, [r*100 for r in r2s], 0.4, label='R² (%)', color='green', alpha=0.7)
ax4.set_xticks(x)
ax4.set_xticklabels(datasets)
ax4.set_ylabel('Value')
ax4.set_title('Performance Summary')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('extratrees_analysis.png', dpi=150)
plt.show()

print("\n✅ 분석 완료! (extratrees_analysis.png 저장됨)")