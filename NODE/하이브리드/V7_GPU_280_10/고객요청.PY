# -*- coding: utf-8 -*-
"""
ExtraTrees 모델 상세 분석
- Train/Val/Test 시계열 순서대로 분할
- 각 데이터셋 날짜 범위 파악
- Overfitting 체크
- Loss 그래프 시각화
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (15, 10)

print("="*80)
print("ExtraTrees 모델 상세 분석 시스템")
print("="*80)

# 모델 디렉토리
model_dir = 'extratrees_analysis'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# 1. 데이터 로드 및 날짜 정보 추가
print("\n[Step 1] 데이터 로딩 및 시간 정보 생성...")
df1 = pd.read_csv('uu.csv')
df2 = pd.read_csv('uu2.csv')
df = pd.concat([df1, df2], ignore_index=True).reset_index(drop=True)

# 시작 시간을 가정 (실제 데이터에 날짜가 없으므로 임의로 설정)
# 2024년 1월 1일 00:00부터 시작한다고 가정
start_date = datetime(2024, 1, 1, 0, 0)
df['timestamp'] = [start_date + timedelta(minutes=i) for i in range(len(df))]

print(f"✓ 전체 데이터: {len(df):,}행")
print(f"✓ 시간 범위: {df['timestamp'].iloc[0]} ~ {df['timestamp'].iloc[-1]}")
print(f"✓ 총 기간: {(df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).days}일")

# 2. 구간 분류 함수
def assign_level(totalcnt_value):
    """TOTALCNT 값을 3구간으로 분류"""
    if totalcnt_value < 1400:
        return 0  # 정상
    elif totalcnt_value < 1700:
        return 1  # 확인
    else:
        return 2  # 위험

# 3. 개선된 시퀀스 생성 함수
def create_sequences_with_timestamps(data, seq_length=280, pred_horizon=10):
    """시간 정보를 포함한 시퀀스 생성"""
    
    print(f"\n시퀀스 생성 중... (280분 → 10분 후)")
    
    feature_cols = ['M14AM14B', 'M14AM10A', 'M14AM16', 'M14AM14BSUM', 
                   'M14AM10ASUM', 'M14AM16SUM', 'M14BM14A', 'M10AM14A', 'M16M14A', 'TOTALCNT']
    
    # 파생 변수 생성
    data = data.copy()
    data['ratio_M14B_M14A'] = np.clip(data['M14AM14B'] / (data['M14AM10A'] + 1), 0, 1000)
    data['ratio_M14B_M16'] = np.clip(data['M14AM14B'] / (data['M14AM16'] + 1), 0, 1000)
    data['totalcnt_change'] = data['TOTALCNT'].diff().fillna(0)
    data['totalcnt_pct_change'] = data['TOTALCNT'].pct_change().fillna(0)
    data['totalcnt_pct_change'] = np.clip(data['totalcnt_pct_change'], -10, 10)
    
    # NaN/inf 처리
    data = data.replace([np.inf, -np.inf], 0)
    data = data.fillna(0)
    
    X_list = []
    y_reg_list = []
    y_cls_list = []
    timestamps = []  # 각 시퀀스의 타임스탬프 저장
    
    n_sequences = len(data) - seq_length - pred_horizon + 1
    
    for i in range(n_sequences):
        if i % 1000 == 0:
            print(f"  진행률: {i/n_sequences*100:.1f}%", end='\r')
        
        start_idx = i
        end_idx = i + seq_length
        seq_data = data.iloc[start_idx:end_idx]
        
        features = []
        
        # 특징 추출 (기존 코드와 동일)
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
                    x = np.arange(len(values))
                    slope, _ = np.polyfit(x, values, 1)
                    features.append(np.clip(slope, -100, 100))
                except:
                    features.append(0)
                
                try:
                    recent_slope = np.polyfit(np.arange(60), values[-60:], 1)[0]
                    features.append(np.clip(recent_slope, -100, 100))
                except:
                    features.append(0)
        
        # 마지막 시점의 파생 변수들
        last_idx = end_idx - 1
        features.extend([
            np.clip(data['ratio_M14B_M14A'].iloc[last_idx], 0, 1000),
            np.clip(data['ratio_M14B_M16'].iloc[last_idx], 0, 1000),
            np.clip(data['totalcnt_change'].iloc[last_idx], -1000, 1000),
            np.clip(data['totalcnt_pct_change'].iloc[last_idx], -10, 10),
        ])
        
        # 타겟
        target_idx = end_idx + pred_horizon - 1
        if target_idx < len(data):
            future_totalcnt = data['TOTALCNT'].iloc[target_idx]
            
            X_list.append(features)
            y_reg_list.append(future_totalcnt)
            y_cls_list.append(assign_level(future_totalcnt))
            timestamps.append(data['timestamp'].iloc[target_idx])  # 예측 시점의 타임스탬프
    
    X = np.array(X_list)
    y_reg = np.array(y_reg_list)
    y_cls = np.array(y_cls_list)
    
    X = np.nan_to_num(X, nan=0.0, posinf=1000.0, neginf=-1000.0)
    
    print(f"\n✓ 시퀀스 생성 완료!")
    
    return X, y_reg, y_cls, timestamps

# 4. 데이터 생성
print("\n[Step 2] 시퀀스 데이터 생성...")
X, y_reg, y_cls, timestamps = create_sequences_with_timestamps(df, seq_length=280, pred_horizon=10)

# 5. 시계열 순서대로 Train/Val/Test 분할 (중요!)
print("\n[Step 3] 시계열 순서대로 데이터 분할...")

# 60% train, 20% validation, 20% test
n_samples = len(X)
train_size = int(n_samples * 0.6)
val_size = int(n_samples * 0.2)

# 시계열 순서 유지하여 분할
X_train = X[:train_size]
y_train = y_reg[:train_size]
y_cls_train = y_cls[:train_size]
timestamps_train = timestamps[:train_size]

X_val = X[train_size:train_size+val_size]
y_val = y_reg[train_size:train_size+val_size]
y_cls_val = y_cls[train_size:train_size+val_size]
timestamps_val = timestamps[train_size:train_size+val_size]

X_test = X[train_size+val_size:]
y_test = y_reg[train_size+val_size:]
y_cls_test = y_cls[train_size+val_size:]
timestamps_test = timestamps[train_size+val_size:]

# 날짜 범위 출력
print("\n[데이터셋별 날짜 범위]")
print("="*60)
print(f"Train Set: {timestamps_train[0]} ~ {timestamps_train[-1]}")
print(f"  - 샘플 수: {len(X_train):,}개")
print(f"  - 기간: {(timestamps_train[-1] - timestamps_train[0]).days}일")
print(f"  - TOTALCNT 범위: {y_train.min():.0f} ~ {y_train.max():.0f}")
print(f"  - 평균±표준편차: {y_train.mean():.0f} ± {y_train.std():.0f}")

print(f"\nValidation Set: {timestamps_val[0]} ~ {timestamps_val[-1]}")
print(f"  - 샘플 수: {len(X_val):,}개")
print(f"  - 기간: {(timestamps_val[-1] - timestamps_val[0]).days}일")
print(f"  - TOTALCNT 범위: {y_val.min():.0f} ~ {y_val.max():.0f}")
print(f"  - 평균±표준편차: {y_val.mean():.0f} ± {y_val.std():.0f}")

print(f"\nTest Set: {timestamps_test[0]} ~ {timestamps_test[-1]}")
print(f"  - 샘플 수: {len(X_test):,}개")
print(f"  - 기간: {(timestamps_test[-1] - timestamps_test[0]).days}일")
print(f"  - TOTALCNT 범위: {y_test.min():.0f} ~ {y_test.max():.0f}")
print(f"  - 평균±표준편차: {y_test.mean():.0f} ± {y_test.std():.0f}")

# 6. 정규화
print("\n[Step 4] 데이터 정규화...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # train 데이터로만 fit
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 7. ExtraTrees 모델 학습 및 검증
print("\n[Step 5] ExtraTrees 모델 학습...")

# 다양한 n_estimators로 학습 곡선 확인
n_estimators_list = [50, 100, 150, 200, 250, 300, 350, 400]
train_losses = []
val_losses = []

for n_est in n_estimators_list:
    print(f"\n학습 중... (n_estimators={n_est})")
    
    model = ExtraTreesRegressor(
        n_estimators=n_est,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    # 학습
    model.fit(X_train_scaled, y_train)
    
    # 예측
    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_val_scaled)
    
    # Loss 계산 (MAE 사용)
    train_loss = mean_absolute_error(y_train, y_train_pred)
    val_loss = mean_absolute_error(y_val, y_val_pred)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    print(f"  Train MAE: {train_loss:.2f}")
    print(f"  Val MAE: {val_loss:.2f}")
    print(f"  Overfitting Gap: {val_loss - train_loss:.2f}")

# 8. 최적 모델 선택 및 최종 학습
best_n_estimators = n_estimators_list[np.argmin(val_losses)]
print(f"\n최적 n_estimators: {best_n_estimators}")

final_model = ExtraTreesRegressor(
    n_estimators=best_n_estimators,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

final_model.fit(X_train_scaled, y_train)

# 9. 최종 성능 평가
print("\n[Step 6] 최종 모델 성능 평가...")

# 각 데이터셋별 예측
y_train_pred = final_model.predict(X_train_scaled)
y_val_pred = final_model.predict(X_val_scaled)
y_test_pred = final_model.predict(X_test_scaled)

# 성능 메트릭 계산
def calculate_metrics(y_true, y_pred, dataset_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # 구간별 정확도
    y_true_cls = np.array([assign_level(val) for val in y_true])
    y_pred_cls = np.array([assign_level(val) for val in y_pred])
    cls_accuracy = np.mean(y_true_cls == y_pred_cls)
    
    print(f"\n[{dataset_name} Set 성능]")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²: {r2:.4f}")
    print(f"  3구간 정확도: {cls_accuracy:.3f}")
    
    # 1700+ 예측 성능
    actual_danger = y_true >= 1700
    pred_danger = y_pred >= 1700
    
    if np.sum(actual_danger) > 0:
        tp = np.sum(actual_danger & pred_danger)
        fp = np.sum(~actual_danger & pred_danger)
        fn = np.sum(actual_danger & ~pred_danger)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"  1700+ 예측: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
        print(f"  1700+ 상세: 정확예측={tp}, 오탐={fp}, 놓침={fn}")
    
    return mae, rmse, r2

train_mae, train_rmse, train_r2 = calculate_metrics(y_train, y_train_pred, "Train")
val_mae, val_rmse, val_r2 = calculate_metrics(y_val, y_val_pred, "Validation")
test_mae, test_rmse, test_r2 = calculate_metrics(y_test, y_test_pred, "Test")

# 10. Overfitting 분석
print("\n[Overfitting 분석]")
print("="*60)
print(f"Train MAE: {train_mae:.2f}")
print(f"Val MAE: {val_mae:.2f}")
print(f"Test MAE: {test_mae:.2f}")
print(f"\nOverfitting 지표:")
print(f"  Val-Train MAE 차이: {val_mae - train_mae:.2f}")
print(f"  Test-Train MAE 차이: {test_mae - train_mae:.2f}")

if val_mae - train_mae > 50:
    print("  ⚠️ Overfitting 징후가 있습니다! (Val-Train > 50)")
elif val_mae - train_mae > 30:
    print("  ⚡ 약간의 Overfitting이 있습니다. (Val-Train > 30)")
else:
    print("  ✅ Overfitting이 잘 제어되고 있습니다.")

# 11. 시각화
print("\n[Step 7] 결과 시각화...")

# 큰 캔버스 생성
fig = plt.figure(figsize=(20, 15))

# 1. Learning Curve (Train vs Validation Loss)
ax1 = plt.subplot(3, 3, 1)
ax1.plot(n_estimators_list, train_losses, 'b-', label='Train MAE', marker='o')
ax1.plot(n_estimators_list, val_losses, 'r-', label='Validation MAE', marker='s')
ax1.set_xlabel('n_estimators')
ax1.set_ylabel('MAE')
ax1.set_title('Learning Curve: Train vs Validation Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axvline(x=best_n_estimators, color='green', linestyle='--', label='Best n_estimators')

# 2. Overfitting Gap 변화
ax2 = plt.subplot(3, 3, 2)
overfitting_gaps = [val_losses[i] - train_losses[i] for i in range(len(n_estimators_list))]
ax2.plot(n_estimators_list, overfitting_gaps, 'g-', marker='D')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.axhline(y=30, color='orange', linestyle='--', alpha=0.7, label='Warning threshold')
ax2.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Overfitting threshold')
ax2.set_xlabel('n_estimators')
ax2.set_ylabel('Validation MAE - Train MAE')
ax2.set_title('Overfitting Gap Analysis')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. 데이터셋별 성능 비교
ax3 = plt.subplot(3, 3, 3)
datasets = ['Train', 'Validation', 'Test']
mae_scores = [train_mae, val_mae, test_mae]
rmse_scores = [train_rmse, val_rmse, test_rmse]
x = np.arange(len(datasets))
width = 0.35
bars1 = ax3.bar(x - width/2, mae_scores, width, label='MAE', color='skyblue')
bars2 = ax3.bar(x + width/2, rmse_scores, width, label='RMSE', color='lightcoral')
ax3.set_xlabel('Dataset')
ax3.set_ylabel('Error')
ax3.set_title('Performance Across Datasets')
ax3.set_xticks(x)
ax3.set_xticklabels(datasets)
ax3.legend()
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom')

# 4. Train Set 예측 산점도
ax4 = plt.subplot(3, 3, 4)
colors_train = ['green' if val < 1400 else 'orange' if val < 1700 else 'red' for val in y_train]
ax4.scatter(y_train, y_train_pred, alpha=0.5, s=10, c=colors_train)
ax4.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2)
ax4.axhline(y=1400, color='orange', linestyle=':', lw=1)
ax4.axvline(x=1400, color='orange', linestyle=':', lw=1)
ax4.axhline(y=1700, color='red', linestyle=':', lw=1)
ax4.axvline(x=1700, color='red', linestyle=':', lw=1)
ax4.set_xlabel('Actual TOTALCNT')
ax4.set_ylabel('Predicted TOTALCNT')
ax4.set_title(f'Train Set (R²={train_r2:.3f})')
ax4.grid(True, alpha=0.3)

# 5. Validation Set 예측 산점도
ax5 = plt.subplot(3, 3, 5)
colors_val = ['green' if val < 1400 else 'orange' if val < 1700 else 'red' for val in y_val]
ax5.scatter(y_val, y_val_pred, alpha=0.5, s=10, c=colors_val)
ax5.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=2)
ax5.axhline(y=1400, color='orange', linestyle=':', lw=1)
ax5.axvline(x=1400, color='orange', linestyle=':', lw=1)
ax5.axhline(y=1700, color='red', linestyle=':', lw=1)
ax5.axvline(x=1700, color='red', linestyle=':', lw=1)
ax5.set_xlabel('Actual TOTALCNT')
ax5.set_ylabel('Predicted TOTALCNT')
ax5.set_title(f'Validation Set (R²={val_r2:.3f})')
ax5.grid(True, alpha=0.3)

# 6. Test Set 예측 산점도
ax6 = plt.subplot(3, 3, 6)
colors_test = ['green' if val < 1400 else 'orange' if val < 1700 else 'red' for val in y_test]
ax6.scatter(y_test, y_test_pred, alpha=0.5, s=10, c=colors_test)
ax6.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
ax6.axhline(y=1400, color='orange', linestyle=':', lw=1)
ax6.axvline(x=1400, color='orange', linestyle=':', lw=1)
ax6.axhline(y=1700, color='red', linestyle=':', lw=1)
ax6.axvline(x=1700, color='red', linestyle=':', lw=1)
ax6.set_xlabel('Actual TOTALCNT')
ax6.set_ylabel('Predicted TOTALCNT')
ax6.set_title(f'Test Set (R²={test_r2:.3f})')
ax6.grid(True, alpha=0.3)

# 7. 특징 중요도 (Top 20)
ax7 = plt.subplot(3, 3, 7)
feature_importance = final_model.feature_importances_
top_20_idx = np.argsort(feature_importance)[-20:]
top_20_importance = feature_importance[top_20_idx]
ax7.barh(range(len(top_20_importance)), top_20_importance)
ax7.set_xlabel('Feature Importance')
ax7.set_ylabel('Feature Index')
ax7.set_title('Top 20 Feature Importances')
ax7.grid(True, alpha=0.3)

# 8. 에러 분포 히스토그램
ax8 = plt.subplot(3, 3, 8)
train_errors = y_train_pred - y_train
val_errors = y_val_pred - y_val
test_errors = y_test_pred - y_test
ax8.hist(train_errors, bins=50, alpha=0.5, label='Train', color='blue')
ax8.hist(val_errors, bins=50, alpha=0.5, label='Validation', color='orange')
ax8.hist(test_errors, bins=50, alpha=0.5, label='Test', color='green')
ax8.axvline(x=0, color='black', linestyle='--')
ax8.set_xlabel('Prediction Error (Predicted - Actual)')
ax8.set_ylabel('Frequency')
ax8.set_title('Error Distribution')
ax8.legend()
ax8.grid(True, alpha=0.3)

# 9. 시간별 예측 성능
ax9 = plt.subplot(3, 3, 9)
# Test set에서 시간순으로 100개 샘플 추출
sample_size = min(100, len(y_test))
indices = np.linspace(0, len(y_test)-1, sample_size, dtype=int)
ax9.plot(range(sample_size), y_test[indices], 'b-', label='Actual', alpha=0.7)
ax9.plot(range(sample_size), y_test_pred[indices], 'r-', label='Predicted', alpha=0.7)
ax9.axhline(y=1400, color='orange', linestyle=':', lw=1, alpha=0.7)
ax9.axhline(y=1700, color='red', linestyle=':', lw=1, alpha=0.7)
ax9.set_xlabel('Sample Index')
ax9.set_ylabel('TOTALCNT')
ax9.set_title('Test Set Time Series Comparison (100 samples)')
ax9.legend()
ax9.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(model_dir, 'extratrees_analysis_results.png'), dpi=150, bbox_inches='tight')
plt.show()

# 12. 모델 저장
print("\n[Step 8] 모델 및 설정 저장...")
joblib.dump(final_model, os.path.join(model_dir, 'extratrees_final_model.pkl'))
joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))

# 분석 결과 저장
analysis_results = {
    'dataset_info': {
        'train': {
            'start': timestamps_train[0].strftime('%Y-%m-%d %H:%M'),
            'end': timestamps_train[-1].strftime('%Y-%m-%d %H:%M'),
            'samples': len(X_train),
            'days': (timestamps_train[-1] - timestamps_train[0]).days
        },
        'validation': {
            'start': timestamps_val[0].strftime('%Y-%m-%d %H:%M'),
            'end': timestamps_val[-1].strftime('%Y-%m-%d %H:%M'),
            'samples': len(X_val),
            'days': (timestamps_val[-1] - timestamps_val[0]).days
        },
        'test': {
            'start': timestamps_test[0].strftime('%Y-%m-%d %H:%M'),
            'end': timestamps_test[-1].strftime('%Y-%m-%d %H:%M'),
            'samples': len(X_test),
            'days': (timestamps_test[-1] - timestamps_test[0]).days
        }
    },
    'performance': {
        'train': {'mae': train_mae, 'rmse': train_rmse, 'r2': train_r2},
        'validation': {'mae': val_mae, 'rmse': val_rmse, 'r2': val_r2},
        'test': {'mae': test_mae, 'rmse': test_rmse, 'r2': test_r2}
    },
    'overfitting_analysis': {
        'val_train_gap': val_mae - train_mae,
        'test_train_gap': test_mae - train_mae,
        'is_overfitting': (val_mae - train_mae) > 30,
        'overfitting_severity': 'High' if (val_mae - train_mae) > 50 else 
                                'Medium' if (val_mae - train_mae) > 30 else 'Low'
    },
    'model_params': {
        'n_estimators': best_n_estimators,
        'max_depth': 20,
        'min_samples_split': 5,
        'min_samples_leaf': 2
    }
}

import json
with open(os.path.join(model_dir, 'analysis_results.json'), 'w') as f:
    json.dump(analysis_results, f, indent=2, default=str)

# 13. 최종 요약 레포트
print("\n" + "="*80)
print("📊 ExtraTrees 모델 분석 최종 보고서")
print("="*80)

print("\n1. 데이터셋 정보:")
print(f"   • Train: {timestamps_train[0].strftime('%Y-%m-%d')} ~ {timestamps_train[-1].strftime('%Y-%m-%d')} ({len(X_train):,}개)")
print(f"   • Val:   {timestamps_val[0].strftime('%Y-%m-%d')} ~ {timestamps_val[-1].strftime('%Y-%m-%d')} ({len(X_val):,}개)")
print(f"   • Test:  {timestamps_test[0].strftime('%Y-%m-%d')} ~ {timestamps_test[-1].strftime('%Y-%m-%d')} ({len(X_test):,}개)")

print("\n2. 모델 성능:")
print(f"   • Train MAE: {train_mae:.2f}")
print(f"   • Val MAE:   {val_mae:.2f} (차이: {val_mae-train_mae:+.2f})")
print(f"   • Test MAE:  {test_mae:.2f} (차이: {test_mae-train_mae:+.2f})")

print("\n3. Overfitting 진단:")
if val_mae - train_mae > 50:
    print(f"   ⚠️ 심각한 Overfitting 발견!")
    print(f"   권장사항:")
    print(f"   - max_depth 감소 (현재: 20 → 추천: 10-15)")
    print(f"   - min_samples_split 증가 (현재: 5 → 추천: 10-20)")
    print(f"   - min_samples_leaf 증가 (현재: 2 → 추천: 5-10)")
elif val_mae - train_mae > 30:
    print(f"   ⚡ 약간의 Overfitting 존재")
    print(f"   권장사항:")
    print(f"   - 정규화 파라미터 조정 고려")
    print(f"   - 교차 검증 적용 권장")
else:
    print(f"   ✅ Overfitting 잘 제어됨")

print("\n4. 최적 하이퍼파라미터:")
print(f"   • n_estimators: {best_n_estimators}")
print(f"   • max_depth: 20")
print(f"   • min_samples_split: 5")
print(f"   • min_samples_leaf: 2")

print("\n5. 파일 저장 위치:")
print(f"   • 모델: {model_dir}/extratrees_final_model.pkl")
print(f"   • 스케일러: {model_dir}/scaler.pkl")
print(f"   • 분석 결과: {model_dir}/analysis_results.json")
print(f"   • 시각화: {model_dir}/extratrees_analysis_results.png")

print("\n" + "="*80)
print("✅ 분석 완료!")
print("="*80)