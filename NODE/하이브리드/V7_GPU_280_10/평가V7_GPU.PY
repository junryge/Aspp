# -*- coding: utf-8 -*-
"""
4개 모델만 사용하여 평가 (1651-1682 구간 4% 상향 보정)
RandomForest, ExtraTrees, GradientBoosting, MLP
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("4개 모델 평가 시스템 (1651-1682 구간 보정 적용)")
print("="*80)

# 설정
model_dir = 'regression_classification_models_280to10'
data_file = 'data/20250731_to_20250806.csv'
output_file = 'evaluation_results_4models_adjusted.csv'

# 사용할 모델 (4개만)
MODEL_NAMES = ['RandomForest', 'ExtraTrees', 'GradientBoosting', 'MLP']

# 이상신호 구간 보정 함수
def adjust_anomaly_prediction(prediction, current_totalcnt):
    """
    1651-1682 구간 근처에서 예측값을 4% 상향 조정
    """
    # 현재값이 1600-1700 범위이고 예측값이 1500-1700 범위일 때
    if 1600 <= current_totalcnt <= 1700 and 1500 <= prediction <= 1700:
        # 4% 상향 조정
        adjusted = prediction * 1.04
        
        # 특히 1651-1682 구간으로 예측될 가능성이 높으면 더 보정
        if 1640 <= prediction <= 1690:
            adjusted = prediction * 1.045  # 4.5% 상향
            
        return adjusted
    
    # 예측값이 이미 1651-1682 구간이면 약간만 상향
    elif 1651 <= prediction <= 1682:
        return prediction * 1.02  # 2% 추가 상향
    
    return prediction

# 시퀀스 생성 함수
def create_sequences_for_evaluation(data, seq_length=280, pred_horizon=10):
    """평가용 280분 시퀀스 생성"""
    
    print(f"시퀀스 생성 중...")
    
    feature_cols = ['M14AM14B', 'M14AM10A', 'M14AM16', 'M14AM14BSUM', 
                   'M14AM10ASUM', 'M14AM16SUM', 'M14BM14A', 'M10AM14A', 'M16M14A', 'TOTALCNT']
    
    # 파생 변수 생성
    data = data.copy()
    data['ratio_M14B_M14A'] = np.clip(data['M14AM14B'] / (data['M14AM10A'] + 1), 0, 1000)
    data['ratio_M14B_M16'] = np.clip(data['M14AM14B'] / (data['M14AM16'] + 1), 0, 1000)
    data['totalcnt_change'] = data['TOTALCNT'].diff().fillna(0)
    data['totalcnt_pct_change'] = np.clip(data['TOTALCNT'].pct_change().fillna(0), -10, 10)
    
    data = data.replace([np.inf, -np.inf], 0).fillna(0)
    
    X_list = []
    y_reg_list = []
    time_info_list = []
    
    n_sequences = len(data) - seq_length - pred_horizon + 1
    
    for i in range(n_sequences):
        if i % 500 == 0:
            print(f"  {i}/{n_sequences}", end='\r')
        
        start_idx = i
        end_idx = i + seq_length
        seq_data = data.iloc[start_idx:end_idx]
        
        features = []
        
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
                values[-1],
                values[-1] - values[0],
                np.mean(values[-60:]),
                np.max(values[-60:]),
                np.mean(values[-30:]),
                np.max(values[-30:]),
            ])
            
            if col == 'TOTALCNT':
                # 구간 관련 특징
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
        
        last_idx = end_idx - 1
        features.extend([
            np.clip(data['ratio_M14B_M14A'].iloc[last_idx], 0, 1000),
            np.clip(data['ratio_M14B_M16'].iloc[last_idx], 0, 1000),
            np.clip(data['totalcnt_change'].iloc[last_idx], -1000, 1000),
            np.clip(data['totalcnt_pct_change'].iloc[last_idx], -10, 10),
        ])
        
        target_idx = end_idx + pred_horizon - 1
        if target_idx < len(data):
            time_info = {
                'current_idx': end_idx - 1,
                'target_idx': target_idx,
                'current_totalcnt': data['TOTALCNT'].iloc[end_idx - 1],  # 현재값 추가
                'seq_max': np.max(seq_data['TOTALCNT'].values),
                'seq_min': np.min(seq_data['TOTALCNT'].values),
            }
            
            X_list.append(features)
            y_reg_list.append(data['TOTALCNT'].iloc[target_idx])
            time_info_list.append(time_info)
    
    X = np.nan_to_num(np.array(X_list), nan=0.0, posinf=1000.0, neginf=-1000.0)
    
    print(f"\n✓ 시퀀스 생성 완료: {len(X)}개")
    return X, np.array(y_reg_list), time_info_list

# 메인 실행
print(f"\n1. 데이터 로딩: {data_file}")
df = pd.read_csv(data_file)
print(f"   데이터 크기: {len(df):,}행")

# DateTime 생성
base_date = datetime(2025, 7, 31, 0, 0)
df['DateTime'] = [base_date + timedelta(minutes=i) for i in range(len(df))]

print(f"\n2. 시퀀스 생성 (280분 → 10분 후)")
X, y_true, time_info = create_sequences_for_evaluation(df, seq_length=280, pred_horizon=10)

print(f"\n3. 스케일러 로딩")
scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
X_scaled = scaler.transform(X)

print(f"\n4. 4개 모델 예측 (1651-1682 구간 보정 적용)")
print("-"*60)
predictions = {}
adjusted_predictions = {}

for name in MODEL_NAMES:
    print(f"   {name} 예측...", end='')
    model = joblib.load(os.path.join(model_dir, f'{name}_regression_model.pkl'))
    
    # 기본 예측
    raw_pred = model.predict(X_scaled)
    
    # 1651-1682 구간 보정 적용
    adjusted_pred = []
    for i in range(len(raw_pred)):
        current_val = time_info[i]['current_totalcnt']
        adj_val = adjust_anomaly_prediction(raw_pred[i], current_val)
        adjusted_pred.append(adj_val)
    
    adjusted_pred = np.array(adjusted_pred)
    
    predictions[name] = raw_pred
    adjusted_predictions[name] = adjusted_pred
    
    # 성능 비교
    mae_raw = np.mean(np.abs(raw_pred - y_true))
    mae_adj = np.mean(np.abs(adjusted_pred - y_true))
    
    print(f" MAE: {mae_raw:.2f} → {mae_adj:.2f} (보정 후)")
    
    # 1651-1682 구간 성능 확인
    anomaly_mask = (y_true >= 1651) & (y_true <= 1682)
    if np.sum(anomaly_mask) > 0:
        anomaly_mae_raw = np.mean(np.abs(raw_pred[anomaly_mask] - y_true[anomaly_mask]))
        anomaly_mae_adj = np.mean(np.abs(adjusted_pred[anomaly_mask] - y_true[anomaly_mask]))
        print(f"      1651-1682 구간 MAE: {anomaly_mae_raw:.2f} → {anomaly_mae_adj:.2f}")

print("-"*60)

print(f"\n5. 결과 CSV 생성")
results = []

for i in range(len(X)):
    current_time = df['DateTime'].iloc[time_info[i]['current_idx']]
    target_time = df['DateTime'].iloc[time_info[i]['target_idx']]
    
    row = {
        '날짜': current_time.strftime('%Y-%m-%d %H:%M'),
        '타켓날짜': target_time.strftime('%Y-%m-%d %H:%M'),
        '실제값': round(y_true[i], 0),
        '시퀀스MAX': round(time_info[i]['seq_max'], 0),
        '시퀀스MIN': round(time_info[i]['seq_min'], 0),
    }
    
    # 4개 모델 예측값 (보정 적용)
    for name in MODEL_NAMES:
        row[f'{name}_예측'] = round(adjusted_predictions[name][i], 0)
        row[f'{name}_오차'] = round(adjusted_predictions[name][i] - y_true[i], 0)
        row[f'{name}_오차율(%)'] = round(abs(adjusted_predictions[name][i] - y_true[i]) / y_true[i] * 100, 2)
    
    results.append(row)

results_df = pd.DataFrame(results)
results_df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"\n✅ 완료! 결과 파일: {output_file}")
print(f"   총 {len(results_df):,}개 예측 생성")

# 성능 요약
print(f"\n[모델별 성능 요약 - 보정 적용]")
print("="*80)
print(f"{'모델':20} {'MAE':>10} {'RMSE':>10} {'1700+정확도':>12} {'1651-1682정확도':>15}")
print("-"*80)

for name in MODEL_NAMES:
    errors = adjusted_predictions[name] - y_true
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    
    # 1700+ 예측 정확도
    danger_mask = y_true >= 1700
    if np.sum(danger_mask) > 0:
        danger_acc = np.mean(np.abs(errors[danger_mask]) < 50) * 100  # 오차 50 이내를 정확으로 봄
    else:
        danger_acc = 0
    
    # 1651-1682 구간 정확도
    anomaly_mask = (y_true >= 1651) & (y_true <= 1682)
    if np.sum(anomaly_mask) > 0:
        anomaly_acc = np.mean(np.abs(errors[anomaly_mask]) < 30) * 100  # 오차 30 이내를 정확으로 봄
    else:
        anomaly_acc = 0
    
    print(f"{name:20} {mae:10.2f} {rmse:10.2f} {danger_acc:11.1f}% {anomaly_acc:14.1f}%")

print("="*80)

# 1651-1682 구간 상세 분석
print(f"\n[1651-1682 이상신호 구간 상세 분석]")
print("-"*80)
anomaly_mask = (y_true >= 1651) & (y_true <= 1682)
anomaly_count = np.sum(anomaly_mask)

if anomaly_count > 0:
    print(f"이상신호 구간 샘플 수: {anomaly_count}개")
    print(f"\n모델별 이상신호 구간 예측 성능:")
    
    for name in MODEL_NAMES:
        anomaly_pred = adjusted_predictions[name][anomaly_mask]
        anomaly_true = y_true[anomaly_mask]
        
        mae = np.mean(np.abs(anomaly_pred - anomaly_true))
        rmse = np.sqrt(np.mean((anomaly_pred - anomaly_true)**2))
        
        # 예측이 1651-1682 구간에 들어간 비율
        correct_range = np.sum((anomaly_pred >= 1651) & (anomaly_pred <= 1682))
        range_accuracy = correct_range / anomaly_count * 100
        
        print(f"\n{name}:")
        print(f"  - MAE: {mae:.2f}")
        print(f"  - RMSE: {rmse:.2f}")
        print(f"  - 구간 내 예측 비율: {range_accuracy:.1f}%")
        
        # 샘플 출력 (처음 5개)
        if anomaly_count > 0:
            print(f"  - 예측 샘플 (처음 5개):")
            for j in range(min(5, anomaly_count)):
                actual = anomaly_true[j]
                pred = anomaly_pred[j]
                error = pred - actual
                print(f"    실제: {actual:.0f} → 예측: {pred:.0f} (오차: {error:+.0f})")
else:
    print("이상신호 구간 샘플이 없습니다.")

# 최종 권장사항
print(f"\n[최종 권장사항]")
print("="*80)
print("✅ 1651-1682 구간 4% 상향 보정 적용 완료")
print("✅ 4개 핵심 모델만 사용 (RandomForest, ExtraTrees, GradientBoosting, MLP)")
print("✅ Ridge, ElasticNet 제외")
print(f"\n📊 최고 성능 모델 (보정 후 기준):")

# 최고 성능 모델 찾기
best_mae = float('inf')
best_model = ""
for name in MODEL_NAMES:
    mae = np.mean(np.abs(adjusted_predictions[name] - y_true))
    if mae < best_mae:
        best_mae = mae
        best_model = name

print(f"   → {best_model} (MAE: {best_mae:.2f})")
print("="*80)