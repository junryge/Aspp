# -*- coding: utf-8 -*-
"""
ExtraTrees Overfitting 분석 - Loss 그래프 생성
원본 코드의 데이터와 함수 그대로 사용
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (15, 10)

print("="*80)
print("ExtraTrees Overfitting 분석 - Loss 그래프")
print("="*80)

# 1. 원본 코드와 동일한 데이터 로드
print("\n[1] 데이터 로딩...")
data_file = 'data/20240201_to_202507271705.csv'
df = pd.read_csv(data_file)
print(f"✓ 데이터: {len(df):,}행")

# 2. 원본 코드의 시퀀스 생성 함수 (그대로 복사)
def create_sequences_280to10(data, seq_length=280, pred_horizon=10):
    print(f"시퀀스 생성 중...")
    
    feature_cols = ['M14AM14B', 'M14AM10A', 'M14AM16', 'M14AM14BSUM', 
                   'M14AM10ASUM', 'M14AM16SUM', 'M14BM14A', 'M10AM14A', 'M16M14A', 'TOTALCNT']
    
    data = data.copy()
    data['ratio_M14B_M14A'] = np.clip(data['M14AM14B'] / (data['M14AM10A'] + 1), 0, 1000)
    data['ratio_M14B_M16'] = np.clip(data['M14AM14B'] / (data['M14AM16'] + 1), 0, 1000)
    data['totalcnt_change'] = data['TOTALCNT'].diff().fillna(0)
    data['totalcnt_pct_change'] = data['TOTALCNT'].pct_change().fillna(0)
    data['totalcnt_pct_change'] = np.clip(data['totalcnt_pct_change'], -10, 10)
    data = data.replace([np.inf, -np.inf], 0).fillna(0)
    
    X_list = []
    y_list = []
    
    n_sequences = len(data) - seq_length - pred_horizon + 1
    
    for i in range(n_sequences):
        if i % 10000 == 0:
            print(f"  {i}/{n_sequences}", end='\r')
        
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
                # 원본 코드의 특별 처리
                features.extend([
                    np.sum((values >= 1650) & (values < 1700)),
                    np.sum(values >= 1700),
                    np.max(values[-20:]),
                    np.sum(values < 1400),
                    np.sum((values >= 1400) & (values < 1700)),
                    np.sum(values >= 1700),
                    np.sum((values >= 1651) & (values <= 1682)),
                    np.max(values[(values >= 1651) & (values <= 1682)]) if len(values[(values >= 1651) & (values <= 1682)]) > 0 else 0,
                    np.mean(values[values < 1400]) if len(values[values < 1400]) > 0 else 0,
                    np.mean(values[(values >= 1400) & (values < 1700)]) if len(values[(values >= 1400) & (values < 1700)]) > 0 else 0,
                    np.mean(values[values >= 1700]) if len(values[values >= 1700]) > 0 else 0,
                ])
                
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
        
        last_idx = end_idx - 1
        features.extend([
            np.clip(data['ratio_M14B_M14A'].iloc[last_idx], 0, 1000),
            np.clip(data['ratio_M14B_M16'].iloc[last_idx], 0, 1000),
            np.clip(data['totalcnt_change'].iloc[last_idx], -1000, 1000),
            np.clip(data['totalcnt_pct_change'].iloc[last_idx], -10, 10),
        ])
        
        target_idx = end_idx + pred_horizon - 1
        if target_idx < len(data):
            X_list.append(features)
            y_list.append(data['TOTALCNT'].iloc[target_idx])
    
    X = np.array(X_list)
    y = np.array(y_list)
    X = np.nan_to_num(X, nan=0.0, posinf=1000.0, neginf=-1000.0)
    
    print(f"\n✓ 시퀀스 생성 완료: {X.shape}")
    return X, y

# 3. 시퀀스 생성
print("\n[2] 시퀀스 생성...")
X, y = create_sequences_280to10(df)

# 4. 시계열 순서로 분할 (60:20:20)
n = len(X)
train_end = int(n * 0.6)
val_end = int(n * 0.8)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

print(f"\n[3] 데이터 분할:")
print(f"  Train: {len(X_train):,}개 (60%)")
print(f"  Val: {len(X_val):,}개 (20%)")
print(f"  Test: {len(X_test):,}개 (20%)")

# 5. 예상 날짜 범위 계산
total_days = 541  # 2024.2.1 ~ 2025.7.27 = 약 541일
train_days = int(total_days * 0.6)
val_days = int(total_days * 0.2)
test_days = int(total_days * 0.2)

print(f"\n[4] 예상 날짜 범위:")
print(f"  Train: 2024-02-01 ~ 2024-12-15 (약 {train_days}일)")
print(f"  Val: 2024-12-15 ~ 2025-04-05 (약 {val_days}일)")
print(f"  Test: 2025-04-05 ~ 2025-07-27 (약 {test_days}일)")

# 6. 정규화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 7. Loss 곡선을 위한 학습 (여러 n_estimators)
print(f"\n[5] Loss 곡선 생성 중...")
n_estimators_list = [10, 20, 30, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500]
train_losses = []
val_losses = []

for n_est in n_estimators_list:
    print(f"  n_estimators={n_est}...", end=' ')
    
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
    
    print(f"Train={train_mae:.1f}, Val={val_mae:.1f}")

# 8. 최종 모델 (n_estimators=300)
print(f"\n[6] 최종 모델 학습...")
final_model = ExtraTreesRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

final_model.fit(X_train_scaled, y_train)

y_train_pred = final_model.predict(X_train_scaled)
y_val_pred = final_model.predict(X_val_scaled)
y_test_pred = final_model.predict(X_test_scaled)

# 9. 성능 계산
train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)

val_mae = mean_absolute_error(y_val, y_val_pred)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
val_r2 = r2_score(y_val, y_val_pred)

test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)

# 10. 결과 출력
print("\n" + "="*80)
print("📊 최종 성능 결과")
print("="*80)

print(f"\n[Train]")
print(f"  MAE: {train_mae:.2f}")
print(f"  RMSE: {train_rmse:.2f}")
print(f"  R²: {train_r2:.4f}")

print(f"\n[Validation]")
print(f"  MAE: {val_mae:.2f} (Gap: {val_mae-train_mae:+.2f})")
print(f"  RMSE: {val_rmse:.2f}")
print(f"  R²: {val_r2:.4f}")

print(f"\n[Test]")
print(f"  MAE: {test_mae:.2f} (Gap: {test_mae-train_mae:+.2f})")
print(f"  RMSE: {test_rmse:.2f}")
print(f"  R²: {test_r2:.4f}")

# 11. Overfitting 진단
print(f"\n[Overfitting 진단]")
overfitting_gap = val_mae - train_mae

if overfitting_gap < 30:
    status = "✅ 양호 - Overfitting 없음"
elif overfitting_gap < 50:
    status = "⚠️ 주의 - 약간의 Overfitting"
else:
    status = "❌ 위험 - 심각한 Overfitting"

print(f"  Val-Train MAE Gap: {overfitting_gap:.2f}")
print(f"  상태: {status}")

# 12. 그래프 그리기
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. Loss/Val_Loss 곡선 (중요!)
ax1 = axes[0, 0]
ax1.plot(n_estimators_list, train_losses, 'b-', marker='o', label='Train Loss', linewidth=2, markersize=6)
ax1.plot(n_estimators_list, val_losses, 'r-', marker='s', label='Validation Loss', linewidth=2, markersize=6)
ax1.fill_between(n_estimators_list, train_losses, val_losses, alpha=0.2, color='gray')
ax1.set_xlabel('n_estimators', fontsize=12)
ax1.set_ylabel('MAE Loss', fontsize=12)
ax1.set_title('Train vs Validation Loss Curve', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.axvline(x=300, color='green', linestyle='--', alpha=0.5, label='Final Model')

# 2. Overfitting Gap
ax2 = axes[0, 1]
gaps = [val_losses[i] - train_losses[i] for i in range(len(n_estimators_list))]
colors = ['green' if g < 30 else 'orange' if g < 50 else 'red' for g in gaps]
ax2.bar(n_estimators_list, gaps, color=colors, alpha=0.7)
ax2.axhline(y=30, color='orange', linestyle='--', label='Warning')
ax2.axhline(y=50, color='red', linestyle='--', label='Critical')
ax2.set_xlabel('n_estimators', fontsize=12)
ax2.set_ylabel('Overfitting Gap (Val - Train)', fontsize=12)
ax2.set_title('Overfitting Gap Analysis', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# 3. 성능 비교
ax3 = axes[0, 2]
datasets = ['Train', 'Val', 'Test']
mae_values = [train_mae, val_mae, test_mae]
colors = ['blue', 'orange', 'green']
bars = ax3.bar(datasets, mae_values, color=colors, alpha=0.7)
for bar, val in zip(bars, mae_values):
    ax3.text(bar.get_x() + bar.get_width()/2, val + 1,
            f'{val:.1f}', ha='center', va='bottom', fontsize=11)
ax3.set_ylabel('MAE', fontsize=12)
ax3.set_title('MAE Comparison', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 4. Train 예측
ax4 = axes[1, 0]
sample_idx = np.random.choice(len(y_train), min(1000, len(y_train)), replace=False)
ax4.scatter(y_train[sample_idx], y_train_pred[sample_idx], alpha=0.4, s=10, color='blue')
ax4.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2)
ax4.set_xlabel('Actual', fontsize=12)
ax4.set_ylabel('Predicted', fontsize=12)
ax4.set_title(f'Train (R²={train_r2:.3f})', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)

# 5. Validation 예측
ax5 = axes[1, 1]
sample_idx = np.random.choice(len(y_val), min(1000, len(y_val)), replace=False)
ax5.scatter(y_val[sample_idx], y_val_pred[sample_idx], alpha=0.4, s=10, color='orange')
ax5.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=2)
ax5.set_xlabel('Actual', fontsize=12)
ax5.set_ylabel('Predicted', fontsize=12)
ax5.set_title(f'Validation (R²={val_r2:.3f})', fontsize=14, fontweight='bold')
ax5.grid(True, alpha=0.3)

# 6. Test 예측
ax6 = axes[1, 2]
sample_idx = np.random.choice(len(y_test), min(1000, len(y_test)), replace=False)
ax6.scatter(y_test[sample_idx], y_test_pred[sample_idx], alpha=0.4, s=10, color='green')
ax6.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
ax6.set_xlabel('Actual', fontsize=12)
ax6.set_ylabel('Predicted', fontsize=12)
ax6.set_title(f'Test (R²={test_r2:.3f})', fontsize=14, fontweight='bold')
ax6.grid(True, alpha=0.3)

plt.suptitle('ExtraTrees Model - Overfitting Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('extratrees_overfitting_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n✅ 그래프 저장: extratrees_overfitting_analysis.png")