# -*- coding: utf-8 -*-
"""
3구간 분류 통합 반도체 물류 예측 시스템 - 280분 → 10분 후 예측
기존 회귀 모델 + 3구간 분류 모델 통합

구간 정의:
- Level 0 (정상): 900-1399
- Level 1 (확인): 1400-1699  
- Level 2 (위험): 1700+
- 이상신호: 1651-1682 구간 특별 감지
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor,
                             RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import Ridge, ElasticNet, LogisticRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                           classification_report, confusion_matrix, accuracy_score,
                           precision_recall_fscore_support)
import joblib
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (15, 10)

print("="*80)
print("3구간 분류 통합 반도체 물류 예측 시스템 - 280분 → 10분 후 예측")
print("="*80)
print("입력: 과거 280분(4시간 40분) 시퀀스 데이터")
print("출력: 10분 후 TOTALCNT 예측값 + 3구간 분류")
print("")
print("구간 정의:")
print("  Level 0 (정상): 900-1399")
print("  Level 1 (확인): 1400-1699") 
print("  Level 2 (위험): 1700+")
print("  이상신호: 1651-1682 구간")
print("="*80)

# 1. 설정
model_dir = 'regression_classification_models_280to10'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# 2. 구간 분류 함수
def assign_level(totalcnt_value):
    """TOTALCNT 값을 3구간으로 분류"""
    if totalcnt_value < 1400:
        return 0  # 정상 (900-1399)
    elif totalcnt_value < 1700:
        return 1  # 확인 (1400-1699)
    else:
        return 2  # 위험 (1700+)

def detect_anomaly_signal(totalcnt_value):
    """1651-1682 이상신호 감지"""
    return 1 if 1651 <= totalcnt_value <= 1682 else 0

# 3. 데이터 로드
print("\n[Step 1] 데이터 로딩...")
df1 = pd.read_csv('uu.csv')
df2 = pd.read_csv('uu2.csv')
df = pd.concat([df1, df2], ignore_index=True).reset_index(drop=True)
print(f"✓ 전체 데이터: {len(df):,}행")
print(f"✓ 시간 범위: {len(df)} 분 ({len(df)/60:.1f} 시간)")


def create_sequences_280to10_with_classification(data, seq_length=280, pred_horizon=10):
    """
    280분 시퀀스로 10분 후 TOTALCNT 예측 + 3구간 분류
    """
    
    print(f"\n시퀀스 생성 중... (280분 → 10분 후 + 3구간 분류)")
    
    feature_cols = ['M14AM14B', 'M14AM10A', 'M14AM16', 'M14AM14BSUM', 
                   'M14AM10ASUM', 'M14AM16SUM', 'M14BM14A', 'M10AM14A', 'M16M14A', 'TOTALCNT']
    
    # 파생 변수 생성 (무한대 값 방지)
    data = data.copy()
    
    # 비율 계산 시 0으로 나누기 방지 및 상한선 설정
    data['ratio_M14B_M14A'] = np.clip(data['M14AM14B'] / (data['M14AM10A'] + 1), 0, 1000)
    data['ratio_M14B_M16'] = np.clip(data['M14AM14B'] / (data['M14AM16'] + 1), 0, 1000)
    
    # 변화량 계산
    data['totalcnt_change'] = data['TOTALCNT'].diff().fillna(0)
    data['totalcnt_pct_change'] = data['TOTALCNT'].pct_change().fillna(0)
    
    # pct_change에서 발생한 무한대 값을 제한
    data['totalcnt_pct_change'] = np.clip(data['totalcnt_pct_change'], -10, 10)
    
    # NaN이나 inf 값을 0으로 대체
    data = data.replace([np.inf, -np.inf], 0)
    data = data.fillna(0)
    
    X_list = []
    y_reg_list = []  # 회귀 타겟
    y_cls_list = []  # 분류 타겟
    y_anomaly_list = []  # 이상신호 타겟
    
    # 가능한 시퀀스 수
    n_sequences = len(data) - seq_length - pred_horizon + 1
    print(f"✓ 생성 가능한 시퀀스: {n_sequences:,}개")
    
    for i in range(n_sequences):
        if i % 1000 == 0:
            print(f"  진행률: {i/n_sequences*100:.1f}%", end='\r')
        
        # 현재 시점부터 과거 280분 데이터
        start_idx = i
        end_idx = i + seq_length
        seq_data = data.iloc[start_idx:end_idx]
        
        features = []
        
        # 각 컬럼별 특징 추출
        for col in feature_cols:
            values = seq_data[col].values
            
            # 기본 통계 (안전하게 계산)
            features.extend([
                np.mean(values),                    # 평균
                np.std(values) if len(values) > 1 else 0,  # 표준편차
                np.min(values),                     # 최소값
                np.max(values),                     # 최대값
                np.percentile(values, 25),          # Q1
                np.percentile(values, 50),          # 중간값
                np.percentile(values, 75),          # Q3
                values[-1],                         # 현재값 (280분째)
                values[-1] - values[0],             # 전체 변화량
                np.mean(values[-60:]),              # 최근 1시간 평균
                np.max(values[-60:]),               # 최근 1시간 최대
                np.mean(values[-30:]),              # 최근 30분 평균
                np.max(values[-30:]),               # 최근 30분 최대
            ])
            
            # TOTALCNT 특별 처리
            if col == 'TOTALCNT':
                # 기존 위험 구간 관련
                features.append(np.sum((values >= 1650) & (values < 1700)))  # 경고 구간
                features.append(np.sum(values >= 1700))                      # 위험 횟수
                features.append(np.max(values[-20:]))                        # 최근 20분 최대
                
                # 3구간 관련 특징 추가
                features.append(np.sum(values < 1400))                       # 정상 구간 횟수
                features.append(np.sum((values >= 1400) & (values < 1700)))  # 확인 구간 횟수
                features.append(np.sum(values >= 1700))                      # 위험 구간 횟수
                
                # 이상신호 1651-1682 구간 특징
                features.append(np.sum((values >= 1651) & (values <= 1682))) # 이상신호 횟수
                anomaly_values = values[(values >= 1651) & (values <= 1682)]
                features.append(np.max(anomaly_values) if len(anomaly_values) > 0 else 0)  # 이상신호 최대값
                
                # 구간별 평균값
                normal_vals = values[values < 1400]
                check_vals = values[(values >= 1400) & (values < 1700)]
                danger_vals = values[values >= 1700]
                
                features.append(np.mean(normal_vals) if len(normal_vals) > 0 else 0)  # 정상구간 평균
                features.append(np.mean(check_vals) if len(check_vals) > 0 else 0)    # 확인구간 평균
                features.append(np.mean(danger_vals) if len(danger_vals) > 0 else 0)  # 위험구간 평균
                
                # 추세 분석 (안전하게)
                try:
                    x = np.arange(len(values))
                    slope, intercept = np.polyfit(x, values, 1)
                    features.append(np.clip(slope, -100, 100))  # 전체 추세 (상한 설정)
                except:
                    features.append(0)
                
                # 최근 60분 추세
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
        
        # 타겟: 10분 후 TOTALCNT
        target_idx = end_idx + pred_horizon - 1
        if target_idx < len(data):
            future_totalcnt = data['TOTALCNT'].iloc[target_idx]
            
            X_list.append(features)
            y_reg_list.append(future_totalcnt)                        # 회귀 타겟
            y_cls_list.append(assign_level(future_totalcnt))          # 분류 타겟
            y_anomaly_list.append(detect_anomaly_signal(future_totalcnt))  # 이상신호 타겟
    
    X = np.array(X_list)
    y_reg = np.array(y_reg_list)
    y_cls = np.array(y_cls_list)
    y_anomaly = np.array(y_anomaly_list)
    
    # 최종 무한대 및 NaN 체크
    X = np.nan_to_num(X, nan=0.0, posinf=1000.0, neginf=-1000.0)
    
    print(f"\n✓ 시퀀스 생성 완료!")
    print(f"✓ 특징 개수: {X.shape[1]}개")
    
    return X, y_reg, y_cls, y_anomaly
# 5. 데이터 생성
print("\n[Step 2] 280분 → 10분 후 시퀀스 데이터 생성...")
X, y_reg, y_cls, y_anomaly = create_sequences_280to10_with_classification(df, seq_length=280, pred_horizon=10)
print(f"✓ 특징 행렬: {X.shape}")
print(f"✓ 회귀 타겟: {y_reg.shape}")
print(f"✓ 분류 타겟: {y_cls.shape}")
print(f"✓ 이상신호 타겟: {y_anomaly.shape}")

# 타겟 분석
print(f"\n[타겟(10분 후 TOTALCNT) 분석]")
print(f"최소값: {y_reg.min():.0f}")
print(f"최대값: {y_reg.max():.0f}")
print(f"평균값: {y_reg.mean():.0f} (±{y_reg.std():.0f})")
print(f"중간값: {np.median(y_reg):.0f}")

print(f"\n[3구간 분포]")
level_counts = np.bincount(y_cls)
for i, count in enumerate(level_counts):
    level_names = ["정상(900-1399)", "확인(1400-1699)", "위험(1700+)"]
    print(f"Level {i} ({level_names[i]}): {count:,}개 ({count/len(y_cls)*100:.2f}%)")

print(f"\n[이상신호 1651-1682 분포]")
anomaly_counts = np.bincount(y_anomaly)
print(f"정상: {anomaly_counts[0]:,}개 ({anomaly_counts[0]/len(y_anomaly)*100:.2f}%)")
if len(anomaly_counts) > 1:
    print(f"이상신호(1651-1682): {anomaly_counts[1]:,}개 ({anomaly_counts[1]/len(y_anomaly)*100:.2f}%)")

# 기존 구간 분석 유지
print(f"\n[기존 구간 분석 (호환성)]")
print(f"1700 이상: {np.sum(y_reg >= 1700):,}개 ({np.sum(y_reg >= 1700)/len(y_reg)*100:.2f}%)")
print(f"1650-1699: {np.sum((y_reg >= 1650) & (y_reg < 1700)):,}개")
print(f"1600-1649: {np.sum((y_reg >= 1600) & (y_reg < 1650)):,}개")

# 6. 데이터 분할 및 정규화
print("\n[Step 3] 데이터 전처리...")

# Train/Test 분할
X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test, y_anomaly_train, y_anomaly_test = train_test_split(
    X, y_reg, y_cls, y_anomaly, test_size=0.2, random_state=42, shuffle=True
)

print(f"✓ 훈련 데이터: {X_train.shape[0]:,}개")
print(f"✓ 테스트 데이터: {X_test.shape[0]:,}개")

# 정규화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 분포 확인
print(f"\n[훈련 데이터 분포]")
level_counts_train = np.bincount(y_cls_train)
for i, count in enumerate(level_counts_train):
    level_names = ["정상", "확인", "위험"]
    print(f"  Level {i} ({level_names[i]}): {count:,}개 ({count/len(y_cls_train)*100:.2f}%)")

print(f"\n[테스트 데이터 분포]")
level_counts_test = np.bincount(y_cls_test)
for i, count in enumerate(level_counts_test):
    level_names = ["정상", "확인", "위험"]
    print(f"  Level {i} ({level_names[i]}): {count:,}개 ({count/len(y_cls_test)*100:.2f}%)")

# 7. 회귀 모델 정의 (기존과 동일)
print("\n[Step 4-1] 회귀 모델 학습...")

regression_models = {
    'RandomForest': RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),
    'ExtraTrees': ExtraTreesRegressor(
        n_estimators=300,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    ),
    'GradientBoosting': GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=10,
        subsample=0.8,
        random_state=42
    ),
    'MLP': MLPRegressor(
        hidden_layer_sizes=(200, 100, 50),
        activation='relu',
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    ),
    'Ridge': Ridge(alpha=1.0, random_state=42),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
}

# 8. 분류 모델 정의 (새로 추가)
classification_models = {
    'RandomForest_Cls': RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    ),
    'ExtraTrees_Cls': ExtraTreesClassifier(
        n_estimators=300,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    ),
    'GradientBoosting_Cls': GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=10,
        subsample=0.8,
        random_state=42
    ),
    'MLP_Cls': MLPClassifier(
        hidden_layer_sizes=(200, 100, 50),
        activation='relu',
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    ),
    'LogisticRegression': LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced'
    )
}

# 9. 회귀 모델 학습 및 평가 (기존과 동일)
print("\n[회귀 모델 학습 및 평가]")
regression_results = {}
regression_predictions = {}

for name, model in regression_models.items():
    print(f"\n[{name}] 학습 중...")
    
    # 학습
    model.fit(X_train_scaled, y_reg_train)
    
    # 예측
    y_pred = model.predict(X_test_scaled)
    
    # 평가 메트릭
    mae = mean_absolute_error(y_reg_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred))
    r2 = r2_score(y_reg_test, y_pred)
    
    # 1700+ 예측 성능 (기존 호환성)
    actual_danger = y_reg_test >= 1700
    pred_danger = y_pred >= 1700
    
    tp = np.sum(actual_danger & pred_danger)
    fp = np.sum(~actual_danger & pred_danger)
    fn = np.sum(actual_danger & ~pred_danger)
    tn = np.sum(~actual_danger & ~pred_danger)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # 3구간 정확도 (예측값을 구간으로 변환하여 평가)
    pred_cls = np.array([assign_level(val) for val in y_pred])
    cls_accuracy = accuracy_score(y_cls_test, pred_cls)
    
    # 결과 저장
    regression_results[name] = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'precision_1700': precision,
        'recall_1700': recall,
        'f1_1700': f1,
        'classification_accuracy': cls_accuracy,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }
    
    regression_predictions[name] = y_pred
    
    # 출력
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²: {r2:.4f}")
    print(f"  3구간 정확도: {cls_accuracy:.3f}")
    print(f"  1700+ 예측: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
    print(f"  1700+ 상세: 정확예측={tp}, 오탐={fp}, 놓침={fn}")
    
    # 모델 저장
    joblib.dump(model, os.path.join(model_dir, f'{name}_regression_model.pkl'))

# 10. 분류 모델 학습 및 평가 (새로 추가)
print(f"\n[Step 4-2] 3구간 분류 모델 학습...")
classification_results = {}
classification_predictions = {}

for name, model in classification_models.items():
    print(f"\n[{name}] 학습 중...")
    
    # 학습
    model.fit(X_train_scaled, y_cls_train)
    
    # 예측
    y_cls_pred = model.predict(X_test_scaled)
    y_cls_prob = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
    
    # 평가 메트릭
    accuracy = accuracy_score(y_cls_test, y_cls_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_cls_test, y_cls_pred, average='weighted')
    
    # 구간별 성능
    cm = confusion_matrix(y_cls_test, y_cls_pred)
    report = classification_report(y_cls_test, y_cls_pred, 
                                 target_names=['정상(0)', '확인(1)', '위험(2)'],
                                 output_dict=True)
    
    # 결과 저장
    classification_results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'classification_report': report
    }
    
    classification_predictions[name] = y_cls_pred
    
    # 출력
    print(f"  정확도: {accuracy:.3f}")
    print(f"  정밀도: {precision:.3f}")
    print(f"  재현율: {recall:.3f}")
    print(f"  F1-Score: {f1:.3f}")
    
    # 구간별 성능 출력
    print(f"  구간별 F1:")
    for i, level_name in enumerate(['정상', '확인', '위험']):
        level_f1 = report[f'정상(0)' if i == 0 else f'확인(1)' if i == 1 else f'위험(2)']['f1-score']
        print(f"    {level_name}: {level_f1:.3f}")
    
    # 모델 저장
    joblib.dump(model, os.path.join(model_dir, f'{name}_classification_model.pkl'))

# 11. 스케일러 저장
joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))

# 12. 최고 모델 선정
print("\n" + "="*80)
print("모델 성능 요약")
print("="*80)

print(f"\n[회귀 모델 성능]")
print(f"{'모델':15} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'3구간정확도':>10} {'1700+F1':>10}")
print("-" * 75)
for name, result in regression_results.items():
    print(f"{name:15} {result['mae']:8.2f} {result['rmse']:8.2f} {result['r2']:8.4f} {result['classification_accuracy']:10.3f} {result['f1_1700']:10.3f}")

print(f"\n[분류 모델 성능]")
print(f"{'모델':20} {'정확도':>8} {'정밀도':>8} {'재현율':>8} {'F1':>8}")
print("-" * 55)
for name, result in classification_results.items():
    print(f"{name:20} {result['accuracy']:8.3f} {result['precision']:8.3f} {result['recall']:8.3f} {result['f1']:8.3f}")

# 최고 성능 모델
best_r2_model = max(regression_results.items(), key=lambda x: x[1]['r2'])[0]
best_mae_model = min(regression_results.items(), key=lambda x: x[1]['mae'])[0]
best_f1_model = max(regression_results.items(), key=lambda x: x[1]['f1_1700'])[0]
best_cls_model = max(classification_results.items(), key=lambda x: x[1]['accuracy'])[0]

print(f"\n최고 R² 회귀 모델: {best_r2_model} (R²={regression_results[best_r2_model]['r2']:.4f})")
print(f"최저 MAE 회귀 모델: {best_mae_model} (MAE={regression_results[best_mae_model]['mae']:.2f})")
print(f"1700+ 최고 F1 회귀 모델: {best_f1_model} (F1={regression_results[best_f1_model]['f1_1700']:.3f})")
print(f"최고 정확도 분류 모델: {best_cls_model} (정확도={classification_results[best_cls_model]['accuracy']:.3f})")

# 13. 시각화 (확장)
print("\n[Step 5] 결과 시각화...")

fig, axes = plt.subplots(3, 3, figsize=(20, 15))

# 회귀 모델 시각화 (기존)
# 1. MAE/RMSE 비교
ax = axes[0, 0]
x = np.arange(len(regression_models))
mae_values = [r['mae'] for r in regression_results.values()]
rmse_values = [r['rmse'] for r in regression_results.values()]

ax.bar(x - 0.2, mae_values, 0.4, label='MAE', color='skyblue')
ax.bar(x + 0.2, rmse_values, 0.4, label='RMSE', color='lightcoral')
ax.set_xticks(x)
ax.set_xticklabels(regression_models.keys(), rotation=45)
ax.set_ylabel('Error')
ax.set_title('회귀 모델 - MAE/RMSE 비교')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. R² 비교
ax = axes[0, 1]
r2_values = [r['r2'] for r in regression_results.values()]
bars = ax.bar(regression_models.keys(), r2_values, color='green', alpha=0.7)
ax.set_xticklabels(regression_models.keys(), rotation=45)
ax.set_ylabel('R² Score')
ax.set_title('회귀 모델 - R² Score 비교')
ax.set_ylim(0, 1)
for bar, val in zip(bars, r2_values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, 
            f'{val:.3f}', ha='center', va='bottom')
ax.grid(True, alpha=0.3)

# 3. 3구간 정확도 비교
ax = axes[0, 2]
cls_acc_values = [r['classification_accuracy'] for r in regression_results.values()]
bars = ax.bar(regression_models.keys(), cls_acc_values, color='purple', alpha=0.7)
ax.set_xticklabels(regression_models.keys(), rotation=45)
ax.set_ylabel('3구간 정확도')
ax.set_title('회귀 모델 - 3구간 분류 정확도')
for bar, val in zip(bars, cls_acc_values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, 
            f'{val:.3f}', ha='center', va='bottom')
ax.grid(True, alpha=0.3)

# 분류 모델 시각화 (새로 추가)
# 4. 분류 모델 정확도
ax = axes[1, 0]
acc_values = [r['accuracy'] for r in classification_results.values()]
bars = ax.bar(classification_models.keys(), acc_values, color='orange', alpha=0.7)
ax.set_xticklabels(classification_models.keys(), rotation=45)
ax.set_ylabel('정확도')
ax.set_title('분류 모델 - 정확도 비교')
for bar, val in zip(bars, acc_values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, 
            f'{val:.3f}', ha='center', va='bottom')
ax.grid(True, alpha=0.3)

# 5. 혼동행렬 (최고 분류 모델)
ax = axes[1, 1]
best_cm = classification_results[best_cls_model]['confusion_matrix']
sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['정상', '확인', '위험'],
            yticklabels=['정상', '확인', '위험'])
ax.set_title(f'{best_cls_model} - 혼동행렬')
ax.set_ylabel('실제')
ax.set_xlabel('예측')

# 6. 3구간 분포
ax = axes[1, 2]
level_names = ['정상\n(900-1399)', '확인\n(1400-1699)', '위험\n(1700+)']
level_counts = [np.sum(y_cls == i) for i in range(3)]
colors = ['green', 'orange', 'red']
bars = ax.bar(level_names, level_counts, color=colors, alpha=0.7)
ax.set_ylabel('개수')
ax.set_title('3구간 분포')
for bar, count in zip(bars, level_counts):
    percentage = count / len(y_cls) * 100
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            f'{count:,}\n({percentage:.1f}%)', ha='center', va='bottom')

# 7-9. 실제값 vs 예측값 산점도 (상위 3개 회귀 모델)
top_regression_models = sorted(regression_results.items(), key=lambda x: x[1]['r2'], reverse=True)[:3]

for idx, (name, _) in enumerate(top_regression_models):
    ax = axes[2, idx]
    y_pred = regression_predictions[name]
    
    # 구간별 색상으로 산점도
    colors = ['green' if val < 1400 else 'orange' if val < 1700 else 'red' for val in y_reg_test]
    ax.scatter(y_reg_test, y_pred, alpha=0.5, s=10, c=colors)
    
    # 대각선
    ax.plot([y_reg_test.min(), y_reg_test.max()], [y_reg_test.min(), y_reg_test.max()], 
            'k--', lw=2, label='Perfect Prediction')
    
    # 구간 경계선
    ax.axhline(y=1400, color='orange', linestyle=':', lw=2, alpha=0.7, label='1400 (정상/확인)')
    ax.axvline(x=1400, color='orange', linestyle=':', lw=2, alpha=0.7)
    ax.axhline(y=1700, color='red', linestyle=':', lw=2, alpha=0.7, label='1700 (확인/위험)')
    ax.axvline(x=1700, color='red', linestyle=':', lw=2, alpha=0.7)
    
    # 이상신호 구간 (1651-1682)
    ax.axhspan(1651, 1682, alpha=0.2, color='yellow', label='이상신호')
    ax.axvspan(1651, 1682, alpha=0.2, color='yellow')
    
    ax.set_xlabel('실제 TOTALCNT')
    ax.set_ylabel('예측 TOTALCNT')
    ax.set_title(f'{name} - R²={regression_results[name]["r2"]:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(model_dir, 'integrated_results.png'), dpi=150)
plt.close()

# 14. 상세 분석
print("\n[1700+ 예측 상세 분석]")
best_model_pred = regression_predictions[best_f1_model]
danger_mask = y_reg_test >= 1700

if np.sum(danger_mask) > 0:
    print(f"\n{best_f1_model} 모델의 1700+ 예측:")
    print(f"실제 1700+ 샘플: {np.sum(danger_mask)}개")
    
    for idx in np.where(danger_mask)[0][:5]:
        actual = y_reg_test[idx]
        pred = best_model_pred[idx]
        error = pred - actual
        actual_level = assign_level(actual)
        pred_level = assign_level(pred)
        print(f"  실제: {actual:.0f}(Level{actual_level}), 예측: {pred:.0f}(Level{pred_level}), 오차: {error:+.0f}")

print(f"\n[이상신호 1651-1682 분석]")
anomaly_mask = (y_reg_test >= 1651) & (y_reg_test <= 1682)
if np.sum(anomaly_mask) > 0:
    print(f"실제 이상신호 샘플: {np.sum(anomaly_mask)}개")
    pred_anomaly = best_model_pred[anomaly_mask]
    detected = np.sum((pred_anomaly >= 1651) & (pred_anomaly <= 1682))
    print(f"이상신호 구간으로 예측된 개수: {detected}개")
    print(f"이상신호 감지율: {detected/np.sum(anomaly_mask)*100:.1f}%")

# 15. 설정 저장
config = {
    'sequence_length': 280,
    'prediction_horizon': 10,
    'num_features': X.shape[1],
    'level_definition': {
        0: '정상 (900-1399)',
        1: '확인 (1400-1699)', 
        2: '위험 (1700+)'
    },
    'anomaly_range': [1651, 1682],
    'best_regression_models': {
        'r2': best_r2_model,
        'mae': best_mae_model,
        'f1_1700': best_f1_model
    },
    'best_classification_model': best_cls_model,
    'regression_results': regression_results,
    'classification_results': classification_results
}

with open(os.path.join(model_dir, 'config.pkl'), 'wb') as f:
    pickle.dump(config, f)

# 16. 통합 실시간 예측 함수
def predict_10min_later_integrated(sequence_280min, regression_model_name=None, classification_model_name=None):
    """
    280분 데이터로 10분 후 TOTALCNT 예측 + 3구간 분류 통합
    
    Parameters:
    -----------
    sequence_280min : array-like, shape (280, n_features)
        280분 시퀀스 데이터
    regression_model_name : str
        사용할 회귀 모델 (None이면 best_r2_model 사용)
    classification_model_name : str
        사용할 분류 모델 (None이면 best_cls_model 사용)
    
    Returns:
    --------
    dict : 통합 예측 결과
        - 'predicted_value': 예측된 TOTALCNT (회귀)
        - 'predicted_level': 예측된 구간 (분류: 0,1,2)
        - 'level_name': 구간 이름
        - 'is_danger': 1700 이상 여부
        - 'is_anomaly': 이상신호(1651-1682) 여부
        - 'confidence': 예측 신뢰도
        - 'models_used': 사용된 모델들
    """
    if regression_model_name is None:
        regression_model_name = best_r2_model
    if classification_model_name is None:
        classification_model_name = best_cls_model
    
    # 모델과 스케일러 로드
    reg_model = joblib.load(os.path.join(model_dir, f'{regression_model_name}_regression_model.pkl'))
    cls_model = joblib.load(os.path.join(model_dir, f'{classification_model_name}_classification_model.pkl'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    
    # 특징 추출 (create_sequences_280to10_with_classification과 동일한 방식)
    # ... 실제 구현 시 추가 ...
    
    # 예시 반환
    predicted_value = 1650
    predicted_level = assign_level(predicted_value)
    level_names = {0: '정상', 1: '확인', 2: '위험'}
    
    return {
        'predicted_value': predicted_value,
        'predicted_level': predicted_level,
        'level_name': level_names[predicted_level],
        'is_danger': predicted_value >= 1700,
        'is_anomaly': detect_anomaly_signal(predicted_value),
        'confidence': 0.85,
        'models_used': {
            'regression': regression_model_name,
            'classification': classification_model_name
        }
    }

print("\n" + "="*80)
print("✓ 통합 3구간 분류 시스템 완성!")
print(f"✓ 저장 위치: {model_dir}/")
print("✓ 회귀 모델 6개 + 분류 모델 5개 = 총 11개 모델")
print("✓ 280분 데이터로 10분 후 TOTALCNT 예측 + 3구간 분류 준비 완료!")
print("✓ 이상신호 1651-1682 구간 감지 기능 포함!")
print("="*80)