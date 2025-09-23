# -*- coding: utf-8 -*-
"""
1700+ 구간 강화 보정 - 앙상블 없음
개별 모델별 결과만 제공
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("1700+ 구간 강화 보정 (앙상블 없음)")
print("="*80)

# 설정
model_dir = 'regression_classification_models_280to10'
data_file = 'data/20250731_to_20250806.csv'
output_file = 'evaluation_no_ensemble.csv'

MODEL_NAMES = ['RandomForest', 'ExtraTrees', 'GradientBoosting', 'MLP']

def correction_1700_plus(prediction, current_totalcnt, seq_max):
    """
    1700+ 구간 단순 강력 보정
    """
    # 1700+ 구간
    if current_totalcnt >= 1700:
        if prediction < 1550:  # 너무 낮게 예측
            return prediction * 1.15  # 15% 상향
        elif prediction < 1650:
            return prediction * 1.10  # 10% 상향
        elif prediction < 1700:
            return prediction * 1.06  # 6% 상향
        else:
            return prediction * 1.02  # 2% 상향
    
    # 1680-1699 구간
    elif 1680 <= current_totalcnt < 1700:
        if prediction < current_totalcnt - 50:
            return prediction * 1.08  # 8% 상향
        elif prediction < current_totalcnt:
            return prediction * 1.04  # 4% 상향
    
    # 1651-1682 구간
    elif 1651 <= current_totalcnt <= 1682:
        if prediction < current_totalcnt:
            return prediction * 1.04  # 4% 상향
    
    return prediction

def create_sequences_for_evaluation(data, seq_length=280, pred_horizon=10):
    """평가용 시퀀스 생성"""
    
    print(f"시퀀스 생성 중...")
    
    feature_cols = ['M14AM14B', 'M14AM10A', 'M14AM16', 'M14AM14BSUM', 
                   'M14AM10ASUM', 'M14AM16SUM', 'M14BM14A', 'M10AM14A', 'M16M14A', 'TOTALCNT']
    
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
                'current_totalcnt': data['TOTALCNT'].iloc[end_idx - 1],
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

base_date = datetime(2025, 7, 31, 0, 0)
df['DateTime'] = [base_date + timedelta(minutes=i) for i in range(len(df))]

print(f"\n2. 시퀀스 생성")
X, y_true, time_info = create_sequences_for_evaluation(df)

print(f"\n3. 스케일러 로딩")
scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))

expected_features = scaler.n_features_in_
if X.shape[1] < expected_features:
    X = np.hstack([X, np.zeros((X.shape[0], expected_features - X.shape[1]))])
elif X.shape[1] > expected_features:
    X = X[:, :expected_features]

X_scaled = scaler.transform(X)

print(f"\n4. 개별 모델별 예측 (앙상블 없음)")
print("="*80)

predictions = {}
corrected_predictions = {}

for name in MODEL_NAMES:
    print(f"\n{name}:")
    model = joblib.load(os.path.join(model_dir, f'{name}_regression_model.pkl'))
    
    raw_pred = model.predict(X_scaled)
    
    # 1700+ 보정
    corrected_pred = []
    for i in range(len(raw_pred)):
        current_val = time_info[i]['current_totalcnt']
        seq_max = time_info[i]['seq_max']
        
        corrected_val = correction_1700_plus(raw_pred[i], current_val, seq_max)
        corrected_pred.append(corrected_val)
    
    corrected_pred = np.array(corrected_pred)
    
    predictions[name] = raw_pred
    corrected_predictions[name] = corrected_pred
    
    # 전체 성능
    mae_raw = np.mean(np.abs(raw_pred - y_true))
    mae_corrected = np.mean(np.abs(corrected_pred - y_true))
    print(f"  전체 MAE: {mae_raw:.2f} → {mae_corrected:.2f}")
    
    # 1700+ 구간
    high_mask = y_true >= 1700
    if np.sum(high_mask) > 0:
        high_mae_raw = np.mean(np.abs(raw_pred[high_mask] - y_true[high_mask]))
        high_mae_corrected = np.mean(np.abs(corrected_pred[high_mask] - y_true[high_mask]))
        print(f"  1700+ MAE: {high_mae_raw:.2f} → {high_mae_corrected:.2f}")
        
        # 정확도
        acc_50 = np.mean(np.abs(corrected_pred[high_mask] - y_true[high_mask]) <= 50) * 100
        acc_100 = np.mean(np.abs(corrected_pred[high_mask] - y_true[high_mask]) <= 100) * 100
        print(f"  1700+ 정확도: ±50={acc_50:.1f}%, ±100={acc_100:.1f}%")

print("="*80)

print(f"\n5. 결과 CSV 생성 (앙상블 없음)")
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
    
    # 각 모델 개별 예측값
    for name in MODEL_NAMES:
        row[f'{name}_예측'] = round(corrected_predictions[name][i], 0)
        row[f'{name}_오차'] = round(corrected_predictions[name][i] - y_true[i], 0)
    
    results.append(row)

results_df = pd.DataFrame(results)
results_df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"\n✅ 완료! 결과 파일: {output_file}")

# 1700+ 구간 분석
print(f"\n[1700+ 구간 개별 모델 성능]")
print("="*100)
high_samples = results_df[results_df['실제값'] >= 1700]

if len(high_samples) > 0:
    print(f"1700+ 샘플 수: {len(high_samples)}개\n")
    
    # 각 모델별 성능
    for name in MODEL_NAMES:
        errors = high_samples[f'{name}_오차'].abs()
        mae = errors.mean()
        acc_50 = (errors <= 50).mean() * 100
        acc_100 = (errors <= 100).mean() * 100
        
        print(f"{name}:")
        print(f"  MAE: {mae:.1f}")
        print(f"  정확도: ±50={acc_50:.1f}%, ±100={acc_100:.1f}%")
        print()
    
    # 샘플 출력
    print("\n예측 결과 샘플 (처음 15개):")
    print("-"*100)
    display_cols = ['날짜', '실제값', 'RandomForest_예측', 'RandomForest_오차',
                   'GradientBoosting_예측', 'GradientBoosting_오차']
    print(high_samples.head(15)[display_cols].to_string(index=False))

print(f"\n💡 보정 규칙:")
print("  1700+: 예측값에 따라 2-15% 상향")
print("  1680-1699: 4-8% 상향")
print("  1651-1682: 4% 상향")
print("  앙상블 없음 - 개별 모델별 결과만 제공")