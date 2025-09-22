# -*- coding: utf-8 -*-
"""
ExtraTrees 완전체 - 280분 → 10분 후 예측
3구간 분류 + 1651-1682 이상신호 감지 포함!
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ExtraTrees 완전체 - 3구간 + 이상신호 감지 포함!")
print("="*80)
print("구간 정의:")
print("  Level 0 (정상): 900-1399")
print("  Level 1 (확인): 1400-1699") 
print("  Level 2 (위험): 1700+")
print("  이상신호: 1651-1682 구간 특별 감지")
print("="*80)

# 모델 저장 폴더
model_dir = 'extratrees_complete_model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# 구간 분류 함수
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

# 데이터 로드
print("\n[1] 데이터 로딩...")
df = pd.read_csv('data/20240201_TO_202507281705.csv')
print(f"✓ 전체 데이터: {len(df):,}행")

# 280분 시퀀스 → 10분 후 예측 데이터 생성 (3구간 + 이상신호 포함)
def create_sequences_280to10_with_all_features(data, seq_length=280, pred_horizon=10):
    """280분 시퀀스로 10분 후 TOTALCNT 예측 - 3구간 + 이상신호 포함"""
    
    print(f"\n시퀀스 생성 중... (280분 → 10분 후)")
    
    feature_cols = ['M14AM14B', 'M14AM10A', 'M14AM16', 'M14AM14BSUM', 
                   'M14AM10ASUM', 'M14AM16SUM', 'M14BM14A', 'M10AM14A', 'M16M14A', 'TOTALCNT']
    
    # 파생 변수
    data = data.copy()
    data['ratio_M14B_M14A'] = np.clip(data['M14AM14B'] / (data['M14AM10A'] + 1), 0, 1000)
    data['ratio_M14B_M16'] = np.clip(data['M14AM14B'] / (data['M14AM16'] + 1), 0, 1000)
    data['totalcnt_change'] = data['TOTALCNT'].diff().fillna(0)
    data['totalcnt_pct_change'] = np.clip(data['TOTALCNT'].pct_change().fillna(0), -10, 10)
    
    # NaN과 inf 처리
    data = data.replace([np.inf, -np.inf], 0)
    data = data.fillna(0)
    
    X_list = []
    y_reg_list = []  # 회귀 타겟
    y_cls_list = []  # 분류 타겟 (3구간)
    y_anomaly_list = []  # 이상신호 타겟 (1651-1682)
    
    n_sequences = len(data) - seq_length - pred_horizon + 1
    print(f"✓ 생성 가능한 시퀀스: {n_sequences:,}개")
    
    for i in range(n_sequences):
        if i % 5000 == 0:
            print(f"  진행: {i}/{n_sequences} ({i/n_sequences*100:.1f}%)", end='\r')
        
        # 280분 데이터
        start_idx = i
        end_idx = i + seq_length
        seq_data = data.iloc[start_idx:end_idx]
        
        features = []
        
        # 각 컬럼별 특징 추출
        for col in feature_cols:
            values = seq_data[col].values
            
            # 기본 통계
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
            
            # TOTALCNT 특별 처리 - 3구간 + 이상신호
            if col == 'TOTALCNT':
                # 기존 위험 구간
                features.append(np.sum((values >= 1650) & (values < 1700)))  # 경고 구간
                features.append(np.sum(values >= 1700))                      # 위험 횟수
                features.append(np.max(values[-20:]))                        # 최근 20분 최대
                
                # ★ 3구간 관련 특징
                features.append(np.sum(values < 1400))                       # 정상 구간 횟수
                features.append(np.sum((values >= 1400) & (values < 1700)))  # 확인 구간 횟수  
                features.append(np.sum(values >= 1700))                      # 위험 구간 횟수
                
                # ★ 이상신호 1651-1682 구간 특징
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
                
                # 추세 분석
                try:
                    x = np.arange(len(values))
                    slope, intercept = np.polyfit(x, values, 1)
                    features.append(np.clip(slope, -100, 100))  # 전체 추세
                except:
                    features.append(0)
                
                # 최근 60분 추세
                try:
                    recent_slope = np.polyfit(np.arange(60), values[-60:], 1)[0]
                    features.append(np.clip(recent_slope, -100, 100))
                except:
                    features.append(0)
        
        # 마지막 시점 파생 변수
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
            y_cls_list.append(assign_level(future_totalcnt))          # 3구간 분류 타겟
            y_anomaly_list.append(detect_anomaly_signal(future_totalcnt))  # 이상신호 타겟
    
    X = np.array(X_list)
    y_reg = np.array(y_reg_list)
    y_cls = np.array(y_cls_list)
    y_anomaly = np.array(y_anomaly_list)
    
    # 최종 무한대 및 NaN 체크
    X = np.nan_to_num(X, nan=0.0, posinf=1000.0, neginf=-1000.0)
    
    print(f"\n✓ 시퀀스 생성 완료!")
    print(f"✓ 특징 개수: {X.shape[1]}개 (3구간 + 이상신호 특징 포함)")
    print(f"✓ 샘플 수: {X.shape[0]:,}개")
    
    return X, y_reg, y_cls, y_anomaly

# 데이터 생성
print("\n[2] 280분 → 10분 후 시퀀스 데이터 생성...")
X, y_reg, y_cls, y_anomaly = create_sequences_280to10_with_all_features(df)

# 타겟 분석
print(f"\n[타겟 분석 - 회귀]")
print(f"최소값: {y_reg.min():.0f}")
print(f"최대값: {y_reg.max():.0f}")
print(f"평균값: {y_reg.mean():.0f} (±{y_reg.std():.0f})")

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

# 데이터 분할
print("\n[3] 데이터 분할...")
X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test, y_anomaly_train, y_anomaly_test = train_test_split(
    X, y_reg, y_cls, y_anomaly, test_size=0.2, random_state=42, shuffle=True
)

print(f"✓ 훈련 데이터: {X_train.shape[0]:,}개")
print(f"✓ 테스트 데이터: {X_test.shape[0]:,}개")

# 정규화 (스케일러)
print("\n[4] 데이터 정규화 (StandardScaler)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✓ 정규화 완료")

# ExtraTrees 회귀 모델
print("\n[5-1] ExtraTrees 회귀 모델 학습...")
reg_model = ExtraTreesRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=False,
    random_state=42,
    n_jobs=-1
)

print("회귀 모델 학습 시작...")
reg_model.fit(X_train_scaled, y_reg_train)
print("✓ 회귀 모델 학습 완료!")

# ExtraTrees 분류 모델 (3구간)
print("\n[5-2] ExtraTrees 3구간 분류 모델 학습...")
cls_model = ExtraTreesClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

print("3구간 분류 모델 학습 시작...")
cls_model.fit(X_train_scaled, y_cls_train)
print("✓ 3구간 분류 모델 학습 완료!")

# ExtraTrees 이상신호 감지 모델
print("\n[5-3] ExtraTrees 이상신호(1651-1682) 감지 모델 학습...")
anomaly_model = ExtraTreesClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

print("이상신호 감지 모델 학습 시작...")
anomaly_model.fit(X_train_scaled, y_anomaly_train)
print("✓ 이상신호 감지 모델 학습 완료!")

# 평가
print("\n[6] 모델 평가...")

# 회귀 모델 평가
y_reg_pred = reg_model.predict(X_test_scaled)
mae = mean_absolute_error(y_reg_test, y_reg_pred)
rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
r2 = r2_score(y_reg_test, y_reg_pred)

print("\n[회귀 모델 성능]")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")

# 3구간 분류 평가
y_cls_pred = cls_model.predict(X_test_scaled)
cls_accuracy = accuracy_score(y_cls_test, y_cls_pred)

print("\n[3구간 분류 성능]")
print(f"정확도: {cls_accuracy:.3f}")
print("\n구간별 성능:")
report = classification_report(y_cls_test, y_cls_pred, 
                              target_names=['정상(0)', '확인(1)', '위험(2)'])
print(report)

# 이상신호 감지 평가
y_anomaly_pred = anomaly_model.predict(X_test_scaled)
anomaly_accuracy = accuracy_score(y_anomaly_test, y_anomaly_pred)

# 이상신호 감지 성능
anomaly_actual = y_anomaly_test == 1
anomaly_predicted = y_anomaly_pred == 1

tp = np.sum(anomaly_actual & anomaly_predicted)
fp = np.sum(~anomaly_actual & anomaly_predicted)
fn = np.sum(anomaly_actual & ~anomaly_predicted)

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("\n[이상신호(1651-1682) 감지 성능]")
print(f"정확도: {anomaly_accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-Score: {f1:.3f}")
if np.sum(anomaly_actual) > 0:
    print(f"실제 이상신호: {np.sum(anomaly_actual)}개")
    print(f"정확히 감지: {tp}개, 오탐: {fp}개, 놓침: {fn}개")

# 1700+ 위험 예측 성능
actual_danger = y_reg_test >= 1700
pred_danger = y_reg_pred >= 1700

tp_danger = np.sum(actual_danger & pred_danger)
fp_danger = np.sum(~actual_danger & pred_danger)
fn_danger = np.sum(actual_danger & ~pred_danger)

precision_danger = tp_danger / (tp_danger + fp_danger) if (tp_danger + fp_danger) > 0 else 0
recall_danger = tp_danger / (tp_danger + fn_danger) if (tp_danger + fn_danger) > 0 else 0
f1_danger = 2 * (precision_danger * recall_danger) / (precision_danger + recall_danger) if (precision_danger + recall_danger) > 0 else 0

print(f"\n[1700+ 위험 구간 예측 성능]")
print(f"Precision: {precision_danger:.3f}")
print(f"Recall: {recall_danger:.3f}")
print(f"F1-Score: {f1_danger:.3f}")

# 모델과 스케일러 저장
print("\n[7] 모델 저장...")
joblib.dump(reg_model, os.path.join(model_dir, 'ExtraTrees_regression_model.pkl'))
joblib.dump(cls_model, os.path.join(model_dir, 'ExtraTrees_3level_classifier.pkl'))
joblib.dump(anomaly_model, os.path.join(model_dir, 'ExtraTrees_anomaly_detector.pkl'))
joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))

print(f"✓ 회귀 모델: {model_dir}/ExtraTrees_regression_model.pkl")
print(f"✓ 3구간 분류 모델: {model_dir}/ExtraTrees_3level_classifier.pkl")
print(f"✓ 이상신호 감지 모델: {model_dir}/ExtraTrees_anomaly_detector.pkl")
print(f"✓ 스케일러: {model_dir}/scaler.pkl")

# 통합 예측 함수
def predict_10min_integrated(sequence_280min):
    """280분 데이터로 10분 후 통합 예측"""
    # 모델들 로드
    reg_model = joblib.load(os.path.join(model_dir, 'ExtraTrees_regression_model.pkl'))
    cls_model = joblib.load(os.path.join(model_dir, 'ExtraTrees_3level_classifier.pkl'))
    anomaly_model = joblib.load(os.path.join(model_dir, 'ExtraTrees_anomaly_detector.pkl'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    
    # 특징 추출과 정규화
    # features = extract_features(sequence_280min)  # 위와 동일한 방식
    # features_scaled = scaler.transform([features])
    
    # 예측
    # value_pred = reg_model.predict(features_scaled)[0]
    # level_pred = cls_model.predict(features_scaled)[0]
    # anomaly_pred = anomaly_model.predict(features_scaled)[0]
    
    # level_names = {0: '정상(900-1399)', 1: '확인(1400-1699)', 2: '위험(1700+)'}
    
    # return {
    #     'predicted_value': value_pred,
    #     'predicted_level': level_pred,
    #     'level_name': level_names[level_pred],
    #     'is_anomaly_1651_1682': anomaly_pred == 1,
    #     'is_danger_1700plus': value_pred >= 1700
    # }
    pass

print("\n" + "="*80)
print("✅ ExtraTrees 완전체 학습 완료!")
print(f"✅ 회귀 성능: MAE={mae:.2f}, R²={r2:.4f}")
print(f"✅ 3구간 분류 정확도: {cls_accuracy:.3f}")
print(f"✅ 이상신호(1651-1682) 감지 F1: {f1:.3f}")
print(f"✅ 1700+ 위험 예측 F1: {f1_danger:.3f}")
print("✅ 모든 기능 포함 완료!")
print("="*80)