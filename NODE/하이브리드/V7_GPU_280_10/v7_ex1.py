# -*- coding: utf-8 -*-
"""
ExtraTrees 완전체 - R2 문제 해결 버전
280분 → 10분 후 예측 (3구간 분류 + 이상신호 감지)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ExtraTrees 완전체 - R2 문제 해결 버전")
print("="*80)
print("구간 정의:")
print("  Level 0 (정상): 900-1399")
print("  Level 1 (확인): 1400-1699") 
print("  Level 2 (위험): 1700+")
print("  이상신호: 1651-1682 구간")
print("="*80)

# 모델 저장 폴더
model_dir = 'extratrees_fixed_model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# 구간 분류 함수
def assign_level(totalcnt_value):
    """TOTALCNT 값을 3구간으로 분류"""
    if totalcnt_value < 1400:
        return 0  # 정상
    elif totalcnt_value < 1700:
        return 1  # 확인
    else:
        return 2  # 위험

def detect_anomaly_signal(totalcnt_value):
    """1651-1682 이상신호 감지"""
    return 1 if 1651 <= totalcnt_value <= 1682 else 0

# 데이터 로드
print("\n[1] 데이터 로딩...")
df = pd.read_csv('data/20240201_TO_202507281705.csv')
print(f"✓ 전체 데이터: {len(df):,}행")

# TOTALCNT 실제 범위 확인 (R2 문제 해결 핵심)
print(f"\nTOTALCNT 실제 범위:")
print(f"  최소값: {df['TOTALCNT'].min():.0f}")
print(f"  최대값: {df['TOTALCNT'].max():.0f}")
print(f"  평균값: {df['TOTALCNT'].mean():.0f}")
print(f"  중간값: {df['TOTALCNT'].median():.0f}")

# 전역 변수로 평균값 저장 (0 대신 사용)
TOTALCNT_MEAN = df['TOTALCNT'].mean()
TOTALCNT_MEDIAN = df['TOTALCNT'].median()

# 280분 시퀀스 → 10분 후 예측 데이터 생성
def create_sequences_280to10_fixed(data, seq_length=280, pred_horizon=10):
    """R2 문제 해결한 시퀀스 생성"""
    
    print(f"\n시퀀스 생성 중... (280분 → 10분 후)")
    
    feature_cols = ['M14AM14B', 'M14AM10A', 'M14AM16', 'M14AM14BSUM', 
                   'M14AM10ASUM', 'M14AM16SUM', 'M14BM14A', 'M10AM14A', 'M16M14A', 'TOTALCNT']
    
    # 파생 변수 (개선된 전처리)
    data = data.copy()
    
    # 0으로 나누기 방지 + 비율 제한
    data['ratio_M14B_M14A'] = np.clip(data['M14AM14B'] / (data['M14AM10A'] + 1), 0, 100)
    data['ratio_M14B_M16'] = np.clip(data['M14AM14B'] / (data['M14AM16'] + 1), 0, 100)
    
    # 변화량 계산
    data['totalcnt_change'] = data['TOTALCNT'].diff()
    data['totalcnt_pct_change'] = data['TOTALCNT'].pct_change()
    
    # ★ R2 개선: NaN을 0이 아닌 중간값으로 채우기
    data['totalcnt_change'].fillna(0, inplace=True)  # 첫 변화량만 0
    data['totalcnt_pct_change'].fillna(0, inplace=True)  # 첫 변화율만 0
    data['totalcnt_pct_change'] = np.clip(data['totalcnt_pct_change'], -1, 1)  # ±100% 제한
    
    # inf 값 처리 (0이 아닌 중간값으로)
    data = data.replace([np.inf, -np.inf], np.nan)
    
    # NaN 값을 각 컬럼의 중간값으로 채우기
    for col in data.columns:
        if data[col].isna().any():
            col_median = data[col].median()
            data[col].fillna(col_median, inplace=True)
    
    X_list = []
    y_reg_list = []
    y_cls_list = []
    y_anomaly_list = []
    
    n_sequences = len(data) - seq_length - pred_horizon + 1
    print(f"✓ 생성 가능한 시퀀스: {n_sequences:,}개")
    
    for i in range(n_sequences):
        if i % 5000 == 0:
            print(f"  진행: {i}/{n_sequences} ({i/n_sequences*100:.1f}%)", end='\r')
        
        start_idx = i
        end_idx = i + seq_length
        seq_data = data.iloc[start_idx:end_idx]
        
        features = []
        
        # 각 컬럼별 특징 추출
        for col in feature_cols:
            values = seq_data[col].values
            
            # 기본 통계
            features.extend([
                np.mean(values),
                np.std(values) if len(values) > 1 else 0,
                np.min(values),
                np.max(values),
                np.percentile(values, 25),
                np.percentile(values, 50),
                np.percentile(values, 75),
                values[-1],  # 현재값
                values[-1] - values[0],  # 변화량
                np.mean(values[-60:]),  # 최근 1시간
                np.max(values[-60:]),
                np.mean(values[-30:]),  # 최근 30분
                np.max(values[-30:]),
            ])
            
            # TOTALCNT 특별 처리
            if col == 'TOTALCNT':
                # 위험 구간
                features.append(np.sum((values >= 1650) & (values < 1700)))
                features.append(np.sum(values >= 1700))
                features.append(np.max(values[-20:]))
                
                # 3구간 특징
                features.append(np.sum(values < 1400))
                features.append(np.sum((values >= 1400) & (values < 1700)))
                features.append(np.sum(values >= 1700))
                
                # 이상신호 1651-1682
                features.append(np.sum((values >= 1651) & (values <= 1682)))
                anomaly_values = values[(values >= 1651) & (values <= 1682)]
                features.append(np.max(anomaly_values) if len(anomaly_values) > 0 else TOTALCNT_MEDIAN)  # 0 대신 중간값
                
                # 구간별 평균 (0 대신 전체 평균 사용)
                normal_vals = values[values < 1400]
                check_vals = values[(values >= 1400) & (values < 1700)]
                danger_vals = values[values >= 1700]
                
                features.append(np.mean(normal_vals) if len(normal_vals) > 0 else TOTALCNT_MEAN)
                features.append(np.mean(check_vals) if len(check_vals) > 0 else TOTALCNT_MEAN)
                features.append(np.mean(danger_vals) if len(danger_vals) > 0 else TOTALCNT_MEAN)
                
                # 추세 (제한된 범위)
                try:
                    x = np.arange(len(values))
                    slope = np.polyfit(x, values, 1)[0]
                    features.append(np.clip(slope, -50, 50))
                except:
                    features.append(0)
                
                try:
                    recent_slope = np.polyfit(np.arange(60), values[-60:], 1)[0]
                    features.append(np.clip(recent_slope, -50, 50))
                except:
                    features.append(0)
        
        # 마지막 시점 파생 변수
        last_idx = end_idx - 1
        features.extend([
            np.clip(data['ratio_M14B_M14A'].iloc[last_idx], 0, 100),
            np.clip(data['ratio_M14B_M16'].iloc[last_idx], 0, 100),
            np.clip(data['totalcnt_change'].iloc[last_idx], -500, 500),
            np.clip(data['totalcnt_pct_change'].iloc[last_idx], -1, 1),
        ])
        
        # 타겟
        target_idx = end_idx + pred_horizon - 1
        if target_idx < len(data):
            future_totalcnt = data['TOTALCNT'].iloc[target_idx]
            
            X_list.append(features)
            y_reg_list.append(future_totalcnt)
            y_cls_list.append(assign_level(future_totalcnt))
            y_anomaly_list.append(detect_anomaly_signal(future_totalcnt))
    
    X = np.array(X_list)
    y_reg = np.array(y_reg_list)
    y_cls = np.array(y_cls_list)
    y_anomaly = np.array(y_anomaly_list)
    
    print(f"\n✓ 시퀀스 생성 완료!")
    print(f"✓ 특징 개수: {X.shape[1]}개")
    print(f"✓ 샘플 수: {X.shape[0]:,}개")
    
    return X, y_reg, y_cls, y_anomaly

# 데이터 생성
print("\n[2] 시퀀스 데이터 생성...")
X, y_reg, y_cls, y_anomaly = create_sequences_280to10_fixed(df)

# 타겟 분석
print(f"\n[타겟 분석]")
print(f"최소값: {y_reg.min():.0f}")
print(f"최대값: {y_reg.max():.0f}")
print(f"평균값: {y_reg.mean():.0f} (±{y_reg.std():.0f})")
print(f"중간값: {np.median(y_reg):.0f}")

print(f"\n[3구간 분포]")
for i in range(3):
    count = np.sum(y_cls == i)
    level_names = ["정상(900-1399)", "확인(1400-1699)", "위험(1700+)"]
    print(f"{level_names[i]}: {count:,}개 ({count/len(y_cls)*100:.2f}%)")

# 데이터 분할
print("\n[3] 데이터 분할...")
X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test, y_anomaly_train, y_anomaly_test = train_test_split(
    X, y_reg, y_cls, y_anomaly, test_size=0.2, random_state=42, shuffle=True
)

print(f"✓ 훈련: {X_train.shape[0]:,}개")
print(f"✓ 테스트: {X_test.shape[0]:,}개")

# 정규화 - RobustScaler 사용 (이상치에 강함)
print("\n[4] 데이터 정규화 (RobustScaler)...")
scaler = RobustScaler()  # StandardScaler 대신 RobustScaler
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✓ 정규화 완료")

# ExtraTrees 회귀 모델 (하이퍼파라미터 최적화)
print("\n[5-1] ExtraTrees 회귀 모델 학습...")
reg_model = ExtraTreesRegressor(
    n_estimators=500,  # 300 → 500
    max_depth=30,      # 20 → 30
    min_samples_split=3,  # 5 → 3
    min_samples_leaf=1,   # 2 → 1
    max_features='sqrt',
    bootstrap=False,
    random_state=42,
    n_jobs=-1
)

reg_model.fit(X_train_scaled, y_reg_train)
print("✓ 회귀 모델 학습 완료!")

# 3구간 분류 모델
print("\n[5-2] ExtraTrees 3구간 분류 모델 학습...")
cls_model = ExtraTreesClassifier(
    n_estimators=500,
    max_depth=30,
    min_samples_split=3,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

cls_model.fit(X_train_scaled, y_cls_train)
print("✓ 3구간 분류 모델 학습 완료!")

# 이상신호 감지 모델
print("\n[5-3] ExtraTrees 이상신호 감지 모델 학습...")
anomaly_model = ExtraTreesClassifier(
    n_estimators=500,
    max_depth=30,
    min_samples_split=3,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

anomaly_model.fit(X_train_scaled, y_anomaly_train)
print("✓ 이상신호 감지 모델 학습 완료!")

# 평가
print("\n[6] 모델 평가...")

# 회귀 평가
y_reg_pred_train = reg_model.predict(X_train_scaled)
y_reg_pred_test = reg_model.predict(X_test_scaled)

mae_train = mean_absolute_error(y_reg_train, y_reg_pred_train)
r2_train = r2_score(y_reg_train, y_reg_pred_train)

mae_test = mean_absolute_error(y_reg_test, y_reg_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred_test))
r2_test = r2_score(y_reg_test, y_reg_pred_test)

print("\n[회귀 모델 성능]")
print(f"훈련 - MAE: {mae_train:.2f}, R²: {r2_train:.4f}")
print(f"테스트 - MAE: {mae_test:.2f}, RMSE: {rmse_test:.2f}, R²: {r2_test:.4f}")

# R2가 여전히 음수인지 확인
if r2_test < 0:
    print(f"\n⚠️ R² 음수 감지: {r2_test:.4f}")
    print("평균 예측 대비 성능 분석:")
    mean_pred = np.full_like(y_reg_test, y_reg_train.mean())
    baseline_mse = mean_squared_error(y_reg_test, mean_pred)
    model_mse = mean_squared_error(y_reg_test, y_reg_pred_test)
    print(f"  평균 예측 MSE: {baseline_mse:.2f}")
    print(f"  모델 MSE: {model_mse:.2f}")
    print(f"  개선율: {(baseline_mse - model_mse) / baseline_mse * 100:.2f}%")

# 3구간 분류 평가
y_cls_pred = cls_model.predict(X_test_scaled)
cls_accuracy = accuracy_score(y_cls_test, y_cls_pred)

print("\n[3구간 분류 성능]")
print(f"정확도: {cls_accuracy:.3f}")

# 이상신호 감지 평가
y_anomaly_pred = anomaly_model.predict(X_test_scaled)
anomaly_actual = y_anomaly_test == 1
anomaly_predicted = y_anomaly_pred == 1

tp = np.sum(anomaly_actual & anomaly_predicted)
fp = np.sum(~anomaly_actual & anomaly_predicted)
fn = np.sum(anomaly_actual & ~anomaly_predicted)

if tp + fp > 0:
    precision = tp / (tp + fp)
else:
    precision = 0

if tp + fn > 0:
    recall = tp / (tp + fn)
else:
    recall = 0

if precision + recall > 0:
    f1 = 2 * (precision * recall) / (precision + recall)
else:
    f1 = 0

print("\n[이상신호(1651-1682) 감지 성능]")
print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

# 1700+ 위험 예측
actual_danger = y_reg_test >= 1700
pred_danger = y_reg_pred_test >= 1700

tp_d = np.sum(actual_danger & pred_danger)
fp_d = np.sum(~actual_danger & pred_danger)
fn_d = np.sum(actual_danger & ~pred_danger)

precision_d = tp_d / (tp_d + fp_d) if (tp_d + fp_d) > 0 else 0
recall_d = tp_d / (tp_d + fn_d) if (tp_d + fn_d) > 0 else 0
f1_d = 2 * (precision_d * recall_d) / (precision_d + recall_d) if (precision_d + recall_d) > 0 else 0

print(f"\n[1700+ 위험 예측 성능]")
print(f"F1-Score: {f1_d:.3f} (Precision: {precision_d:.3f}, Recall: {recall_d:.3f})")

# 모델 저장
print("\n[7] 모델 저장...")
joblib.dump(reg_model, os.path.join(model_dir, 'ExtraTrees_regression.pkl'))
joblib.dump(cls_model, os.path.join(model_dir, 'ExtraTrees_classifier.pkl'))
joblib.dump(anomaly_model, os.path.join(model_dir, 'ExtraTrees_anomaly.pkl'))
joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))

print("\n" + "="*80)
print("✅ ExtraTrees 완전체 학습 완료!")
print(f"✅ 회귀 R²: {r2_test:.4f} (음수 문제 {'해결!' if r2_test > 0 else '개선 중'})")
print(f"✅ 3구간 분류: {cls_accuracy:.3f}")
print(f"✅ 이상신호 F1: {f1:.3f}")
print(f"✅ 1700+ 예측 F1: {f1_d:.3f}")
print("="*80)