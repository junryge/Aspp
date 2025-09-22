# -*- coding: utf-8 -*-
"""
추세 기반 동적 보정 시스템
시퀀스 마지막 100개 데이터의 추세에 따라 3-5% 차등 보정
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("추세 기반 동적 보정 시스템 (상승 5% / 보합 4% / 하락 3%)")
print("="*80)

# 설정
model_dir = 'regression_classification_models_280to10'
data_file = 'data/20250731_to_20250806.csv'
output_file = 'evaluation_trend_based.csv'

# 사용할 모델 (4개)
MODEL_NAMES = ['RandomForest', 'ExtraTrees', 'GradientBoosting', 'MLP']

def calculate_trend(sequence_last_100):
    """
    마지막 100개 데이터의 추세 계산
    상승(+1), 보합(0), 하락(-1) 반환
    """
    if len(sequence_last_100) < 100:
        sequence_last_100 = sequence_last_100[-min(len(sequence_last_100), 100):]
    
    # 1. 선형 회귀로 추세 계산
    x = np.arange(len(sequence_last_100))
    slope, _ = np.polyfit(x, sequence_last_100, 1)
    
    # 2. 이동평균 비교
    if len(sequence_last_100) >= 50:
        first_half_avg = np.mean(sequence_last_100[:50])
        second_half_avg = np.mean(sequence_last_100[50:])
        ma_trend = second_half_avg - first_half_avg
    else:
        ma_trend = 0
    
    # 3. 최근 20개 vs 이전 20개 비교
    if len(sequence_last_100) >= 40:
        recent_20_avg = np.mean(sequence_last_100[-20:])
        prev_20_avg = np.mean(sequence_last_100[-40:-20])
        short_trend = recent_20_avg - prev_20_avg
    else:
        short_trend = 0
    
    # 종합 판단
    # slope가 양수이고 이동평균도 상승 → 상승 추세
    if slope > 0.5 and ma_trend > 10:
        return 1  # 상승
    # slope가 음수이고 이동평균도 하락 → 하락 추세
    elif slope < -0.5 and ma_trend < -10:
        return -1  # 하락
    # 그 외는 보합
    else:
        return 0  # 보합

def trend_based_correction(prediction, current_totalcnt, sequence_last_100, seq_max):
    """
    추세 기반 동적 보정
    """
    # 추세 계산
    trend = calculate_trend(sequence_last_100)
    
    # 추세에 따른 보정률 결정
    if trend == 1:  # 상승 추세
        base_correction = 1.05  # 5%
        trend_name = "상승"
    elif trend == -1:  # 하락 추세
        base_correction = 1.03  # 3%
        trend_name = "하락"
    else:  # 보합
        base_correction = 1.04  # 4%
        trend_name = "보합"
    
    # 1651-1682 구간 (기존 요구사항)
    if 1651 <= current_totalcnt <= 1682:
        adjusted = prediction * base_correction
        return adjusted, trend_name
    
    # 1680-1750 구간 (강화 보정)
    if 1680 <= current_totalcnt <= 1750:
        if prediction < 1600:
            # 극심한 과소예측 - 추세 관계없이 강력 보정
            adjusted = prediction * 1.15
            # 추세가 상승이면 추가 보정
            if trend == 1:
                adjusted *= 1.03
        elif 1600 <= prediction < 1680:
            # 중간 과소예측 - 추세 기반 보정 강화
            if trend == 1:
                adjusted = prediction * 1.12  # 상승 시 12%
            elif trend == -1:
                adjusted = prediction * 1.08  # 하락 시 8%
            else:
                adjusted = prediction * 1.10  # 보합 시 10%
        else:
            # 적정 예측 - 기본 추세 보정
            adjusted = prediction * base_correction
        
        return adjusted, trend_name
    
    # 1700+ 구간
    if current_totalcnt >= 1700:
        if prediction < 1650:
            # 과소예측 - 추세 반영 강화
            if trend == 1:
                adjusted = prediction * 1.13
            else:
                adjusted = prediction * 1.10
        elif prediction < 1700:
            # 약간 낮은 예측 - 추세 기반
            adjusted = prediction * (base_correction + 0.03)
        else:
            # 적정 예측 - 가벼운 보정
            adjusted = prediction * base_correction
        
        return adjusted, trend_name
    
    # 일반 구간
    if current_totalcnt >= 1600 and prediction < current_totalcnt * 0.95:
        # 현재값보다 5% 이상 낮게 예측 시 추세 보정
        return prediction * base_correction, trend_name
    
    return prediction, "없음"

def create_sequences_for_evaluation(data, seq_length=280, pred_horizon=10):
    """평가용 280분 시퀀스 생성 (마지막 100개 데이터 추가)"""
    
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
            # 마지막 100개 데이터 저장
            last_100_data = seq_data['TOTALCNT'].values[-100:]
            
            time_info = {
                'current_idx': end_idx - 1,
                'target_idx': target_idx,
                'current_totalcnt': data['TOTALCNT'].iloc[end_idx - 1],
                'seq_max': np.max(seq_data['TOTALCNT'].values),
                'seq_min': np.min(seq_data['TOTALCNT'].values),
                'last_100': last_100_data,  # 추세 분석용
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
X, y_true, time_info = create_sequences_for_evaluation(df)

print(f"\n3. 스케일러 로딩")
scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))

# 특징 개수 맞추기
expected_features = scaler.n_features_in_
if X.shape[1] < expected_features:
    X = np.hstack([X, np.zeros((X.shape[0], expected_features - X.shape[1]))])
elif X.shape[1] > expected_features:
    X = X[:, :expected_features]

X_scaled = scaler.transform(X)

print(f"\n4. 추세 기반 동적 보정 예측")
print("-"*80)

predictions = {}
corrected_predictions = {}
trend_stats = {'상승': 0, '보합': 0, '하락': 0, '없음': 0}

for name in MODEL_NAMES:
    print(f"\n{name}:")
    model = joblib.load(os.path.join(model_dir, f'{name}_regression_model.pkl'))
    
    # 기본 예측
    raw_pred = model.predict(X_scaled)
    
    # 추세 기반 보정 적용
    corrected_pred = []
    for i in range(len(raw_pred)):
        current_val = time_info[i]['current_totalcnt']
        seq_last_100 = time_info[i]['last_100']
        seq_max = time_info[i]['seq_max']
        
        corrected_val, trend_type = trend_based_correction(
            raw_pred[i], current_val, seq_last_100, seq_max
        )
        corrected_pred.append(corrected_val)
        
        if name == 'GradientBoosting':  # 한 모델만 통계
            trend_stats[trend_type] += 1
    
    corrected_pred = np.array(corrected_pred)
    
    predictions[name] = raw_pred
    corrected_predictions[name] = corrected_pred
    
    # 전체 성능
    mae_raw = np.mean(np.abs(raw_pred - y_true))
    mae_corrected = np.mean(np.abs(corrected_pred - y_true))
    print(f"  전체 MAE: {mae_raw:.2f} → {mae_corrected:.2f}")
    
    # 1680+ 구간 성능
    high_mask = y_true >= 1680
    if np.sum(high_mask) > 0:
        high_mae_raw = np.mean(np.abs(raw_pred[high_mask] - y_true[high_mask]))
        high_mae_corrected = np.mean(np.abs(corrected_pred[high_mask] - y_true[high_mask]))
        print(f"  1680+ MAE: {high_mae_raw:.2f} → {high_mae_corrected:.2f}")

# 추세 통계
print(f"\n추세 분포:")
for trend_type, count in trend_stats.items():
    if count > 0:
        print(f"  {trend_type}: {count}개 ({count/len(y_true)*100:.1f}%)")

print("-"*80)

print(f"\n5. 결과 CSV 생성")
results = []

for i in range(len(X)):
    current_time = df['DateTime'].iloc[time_info[i]['current_idx']]
    target_time = df['DateTime'].iloc[time_info[i]['target_idx']]
    
    # 추세 계산
    trend = calculate_trend(time_info[i]['last_100'])
    trend_name = {1: "상승", 0: "보합", -1: "하락"}[trend]
    
    row = {
        '날짜': current_time.strftime('%Y-%m-%d %H:%M'),
        '타켓날짜': target_time.strftime('%Y-%m-%d %H:%M'),
        '실제값': round(y_true[i], 0),
        '시퀀스MAX': round(time_info[i]['seq_max'], 0),
        '시퀀스MIN': round(time_info[i]['seq_min'], 0),
        '추세': trend_name,
    }
    
    # 4개 모델 예측 (보정 적용)
    for name in MODEL_NAMES:
        row[f'{name}_예측'] = round(corrected_predictions[name][i], 0)
    
    results.append(row)

results_df = pd.DataFrame(results)
results_df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"\n✅ 완료! 결과 파일: {output_file}")

# 성능 요약
print(f"\n[최종 성능 요약]")
print("="*80)
print(f"{'모델':20} {'전체 MAE':>10} {'1680+ MAE':>12}")
print("-"*60)

for name in MODEL_NAMES:
    mae = np.mean(np.abs(corrected_predictions[name] - y_true))
    
    high_mask = y_true >= 1680
    if np.sum(high_mask) > 0:
        high_mae = np.mean(np.abs(corrected_predictions[name][high_mask] - y_true[high_mask]))
    else:
        high_mae = 0
    
    print(f"{name:20} {mae:10.2f} {high_mae:12.2f}")

print("="*80)

# 문제 샘플 분석
print(f"\n[1680+ 구간 예측 샘플]")
print("-"*100)
problem_samples = results_df[results_df['실제값'] >= 1680].head(15)

if len(problem_samples) > 0:
    display_cols = ['날짜', '실제값', '추세', 'RandomForest_예측', 
                   'ExtraTrees_예측', 'GradientBoosting_예측', 'MLP_예측']
    print(problem_samples[display_cols].to_string(index=False))
    
    # 평균 오차 계산
    print(f"\n각 모델 평균 오차 (1680+ 구간):")
    for name in MODEL_NAMES:
        errors = problem_samples[f'{name}_예측'] - problem_samples['실제값']
        print(f"  {name}: {errors.mean():.1f}")

print(f"\n💡 추세 기반 보정 효과:")
print("  상승 추세 → 5% 보정")
print("  보합 추세 → 4% 보정")
print("  하락 추세 → 3% 보정")