# -*- coding: utf-8 -*-
"""
ExtraTrees 완전체 - 모델 평가 스크립트
- 저장된 모델과 스케일러를 로드
- 학습 시와 동일한 방법으로 평가용 데이터 전처리
- 회귀, 분류, 이상신호 예측 성능을 종합적으로 평가
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# --- 설정 ---
# 모델과 스케일러가 저장된 폴더 경로
MODEL_DIR = 'extratrees_fixed_model'
# 평가할 데이터 파일 경로
DATA_PATH = 'data/20240201_TO_202507281705.csv'
# --- 설정 끝 ---


print("="*80)
print("ExtraTrees 완전체 - 모델 평가 시작")
print(f"모델 경로: {MODEL_DIR}")
print(f"데이터 경로: {DATA_PATH}")
print("="*80)


# ==============================================================================
# 1. 모델 및 스케일러 로드
# ==============================================================================
print("\n[1] 모델 및 스케일러 로딩...")
try:
    reg_model = joblib.load(os.path.join(MODEL_DIR, 'ExtraTrees_regression.pkl'))
    cls_model = joblib.load(os.path.join(MODEL_DIR, 'ExtraTrees_classifier.pkl'))
    anomaly_model = joblib.load(os.path.join(MODEL_DIR, 'ExtraTrees_anomaly.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    print("✓ 모델 4개 파일 로딩 완료!")
except FileNotFoundError as e:
    print(f"오류: 모델 파일을 찾을 수 없습니다. 경로를 확인하세요. -> {e}")
    exit()


# ==============================================================================
# 2. 학습 코드와 동일한 데이터 처리 함수
# (이 함수들은 학습 시 사용된 것과 반드시 동일해야 합니다)
# ==============================================================================
def assign_level(totalcnt_value):
    if totalcnt_value < 1400: return 0
    elif totalcnt_value < 1700: return 1
    else: return 2

def detect_anomaly_signal(totalcnt_value):
    return 1 if 1651 <= totalcnt_value <= 1682 else 0

def create_sequences_for_evaluation(data, seq_length=280, pred_horizon=10):
    """평가를 위한 시퀀스 생성 함수 (학습 코드와 100% 동일)"""
    print("\n[2] 평가용 시퀀스 데이터 생성 시작...")
    
    # 학습 시 사용된 전역 변수와 동일하게 계산
    TOTALCNT_MEAN = data['TOTALCNT'].mean()
    TOTALCNT_MEDIAN = data['TOTALCNT'].median()
    
    feature_cols = ['M14AM14B', 'M14AM10A', 'M14AM16', 'M14AM14BSUM', 
                   'M14AM10ASUM', 'M14AM16SUM', 'M14BM14A', 'M10AM14A', 'M16M14A', 'TOTALCNT']
    
    data = data.copy()
    data['ratio_M14B_M14A'] = np.clip(data['M14AM14B'] / (data['M14AM10A'] + 1), 0, 100)
    data['ratio_M14B_M16'] = np.clip(data['M14AM14B'] / (data['M14AM16'] + 1), 0, 100)
    data['totalcnt_change'] = data['TOTALCNT'].diff().fillna(0)
    data['totalcnt_pct_change'] = data['TOTALCNT'].pct_change().fillna(0)
    data['totalcnt_pct_change'] = np.clip(data['totalcnt_pct_change'], -1, 1)
    
    data = data.replace([np.inf, -np.inf], np.nan)
    for col in data.columns:
        if data[col].isna().any():
            col_median = data[col].median()
            data[col].fillna(col_median, inplace=True)
    
    X_list, y_reg_list, y_cls_list, y_anomaly_list = [], [], [], []
    
    n_sequences = len(data) - seq_length - pred_horizon + 1
    print(f"✓ 생성 가능한 시퀀스: {n_sequences:,}개")
    
    for i in range(n_sequences):
        if i % 5000 == 0:
            print(f"  진행: {i}/{n_sequences} ({i/n_sequences*100:.1f}%)", end='\r')
        
        seq_data = data.iloc[i : i + seq_length]
        features = []
        
        for col in feature_cols:
            values = seq_data[col].values
            features.extend([
                np.mean(values), np.std(values), np.min(values), np.max(values),
                np.percentile(values, 25), np.percentile(values, 50), np.percentile(values, 75),
                values[-1], values[-1] - values[0],
                np.mean(values[-60:]), np.max(values[-60:]),
                np.mean(values[-30:]), np.max(values[-30:]),
            ])
            if col == 'TOTALCNT':
                features.extend([
                    np.sum((values >= 1650) & (values < 1700)),
                    np.sum(values >= 1700),
                    np.max(values[-20:]),
                    np.sum(values < 1400),
                    np.sum((values >= 1400) & (values < 1700)),
                    np.sum(values >= 1700),
                    np.sum((values >= 1651) & (values <= 1682)),
                ])
                anomaly_values = values[(values >= 1651) & (values <= 1682)]
                features.append(np.max(anomaly_values) if len(anomaly_values) > 0 else TOTALCNT_MEDIAN)
                
                normal_vals = values[values < 1400]
                check_vals = values[(values >= 1400) & (values < 1700)]
                danger_vals = values[values >= 1700]
                features.extend([
                    np.mean(normal_vals) if len(normal_vals) > 0 else TOTALCNT_MEAN,
                    np.mean(check_vals) if len(check_vals) > 0 else TOTALCNT_MEAN,
                    np.mean(danger_vals) if len(danger_vals) > 0 else TOTALCNT_MEAN,
                ])
                try:
                    features.append(np.clip(np.polyfit(np.arange(len(values)), values, 1)[0], -50, 50))
                    features.append(np.clip(np.polyfit(np.arange(60), values[-60:], 1)[0], -50, 50))
                except:
                    features.extend([0, 0])

        last_idx = i + seq_length - 1
        features.extend([
            np.clip(data['ratio_M14B_M14A'].iloc[last_idx], 0, 100),
            np.clip(data['ratio_M14B_M16'].iloc[last_idx], 0, 100),
            np.clip(data['totalcnt_change'].iloc[last_idx], -500, 500),
            np.clip(data['totalcnt_pct_change'].iloc[last_idx], -1, 1),
        ])
        
        target_idx = i + seq_length + pred_horizon - 1
        if target_idx < len(data):
            future_totalcnt = data['TOTALCNT'].iloc[target_idx]
            X_list.append(features)
            y_reg_list.append(future_totalcnt)
            y_cls_list.append(assign_level(future_totalcnt))
            y_anomaly_list.append(detect_anomaly_signal(future_totalcnt))
            
    print(f"\n✓ 시퀀스 생성 완료! (총 {len(X_list):,}개)")
    return np.array(X_list), np.array(y_reg_list), np.array(y_cls_list), np.array(y_anomaly_list)


# ==============================================================================
# 3. 데이터 로드 및 예측 수행
# ==============================================================================
# 데이터 로드
print(f"\n[3] 평가용 데이터 로딩: {DATA_PATH}")
try:
    eval_df = pd.read_csv(DATA_PATH)
    print(f"✓ 데이터 로드 완료: {len(eval_df):,}행")
except FileNotFoundError:
    print(f"오류: 데이터 파일을 찾을 수 없습니다. 경로를 확인하세요.")
    exit()

# 시퀀스 생성
X_eval, y_reg_true, y_cls_true, y_anomaly_true = create_sequences_for_evaluation(eval_df)

# 데이터 정규화 (★매우 중요: fit_transform이 아닌 transform 사용)
print("\n[4] 데이터 정규화 적용...")
X_eval_scaled = scaler.transform(X_eval)
print("✓ 정규화 완료!")

# 예측 수행
print("\n[5] 모델 예측 수행...")
y_reg_pred = reg_model.predict(X_eval_scaled)
y_cls_pred = cls_model.predict(X_eval_scaled)
y_anomaly_pred = anomaly_model.predict(X_eval_scaled)
print("✓ 예측 완료!")


# ==============================================================================
# 4. 성능 평가 및 결과 출력
# ==============================================================================
print("\n" + "="*80)
print("종합 평가 결과")
print("="*80)

# 회귀 모델 평가
mae = mean_absolute_error(y_reg_true, y_reg_pred)
rmse = np.sqrt(mean_squared_error(y_reg_true, y_reg_pred))
r2 = r2_score(y_reg_true, y_reg_pred)

print("\n[📈 회귀 모델 성능]")
print(f"  MAE:  {mae:.2f}")
print(f"  RMSE: {rmse:.2f}")
print(f"  R²:   {r2:.4f}")

# 3구간 분류 모델 평가
cls_accuracy = accuracy_score(y_cls_true, y_cls_pred)
print("\n[📊 3구간 분류 성능]")
print(f"  정확도: {cls_accuracy:.3f}")
print("\n  Classification Report:")
print(classification_report(y_cls_true, y_cls_pred, target_names=['정상(0)', '확인(1)', '위험(2)']))

# 이상신호 감지 모델 평가
print("[🔥 이상신호(1651-1682) 감지 성능]")
print(classification_report(y_anomaly_true, y_anomaly_pred, target_names=['정상', '이상신호']))

# 1700+ 위험 예측 성능 (회귀 모델 기반)
actual_danger = y_reg_true >= 1700
pred_danger = y_reg_pred >= 1700
tp_d = np.sum(actual_danger & pred_danger)
fp_d = np.sum(~actual_danger & pred_danger)
fn_d = np.sum(actual_danger & ~pred_danger)
precision_d = tp_d / (tp_d + fp_d) if (tp_d + fp_d) > 0 else 0
recall_d = tp_d / (tp_d + fn_d) if (tp_d + fn_d) > 0 else 0
f1_d = 2 * (precision_d * recall_d) / (precision_d + recall_d) if (precision_d + recall_d) > 0 else 0

print(f"[🚨 1700+ 위험 예측 성능 (회귀 기반)]")
print(f"  F1-Score: {f1_d:.3f} (Precision: {precision_d:.3f}, Recall: {recall_d:.3f})")


# ==============================================================================
# 5. 예측 결과 DataFrame으로 확인
# ==============================================================================
print("\n" + "="*80)
print("예측 결과 샘플 확인")
print("="*80)

results_df = pd.DataFrame({
    '실제 TOTALCNT': y_reg_true,
    '예측 TOTALCNT': y_reg_pred,
    '오차': y_reg_true - y_reg_pred,
    '실제 구간': y_cls_true,
    '예측 구간': y_cls_pred,
    '실제 이상신호': y_anomaly_true,
    '예측 이상신호': y_anomaly_pred
})

# 소수점 둘째 자리까지 반올림하여 보기 좋게 설정
pd.set_option('display.float_format', '{:.2f}'.format)
print(results_df.tail(15)) # 마지막 15개 결과 출력


print("\n✅ 평가가 성공적으로 완료되었습니다.")
