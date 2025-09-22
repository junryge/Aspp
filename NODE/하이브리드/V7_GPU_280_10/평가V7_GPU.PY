# -*- coding: utf-8 -*-
"""
학습된 280분 모델로 평가만 수행
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("280분 시퀀스 모델 평가 (학습된 모델 사용)")
print("="*80)

# 설정
model_dir = 'regression_classification_models_280to10'
data_file = 'data/20250731_to_20250806.csv'
output_file = 'evaluation_results.csv'

# 시퀀스 생성 함수 (학습 시와 동일)
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

# DateTime 생성 (1분 간격 가정)
base_date = datetime(2025, 7, 31, 0, 0)
df['DateTime'] = [base_date + timedelta(minutes=i) for i in range(len(df))]

print(f"\n2. 시퀀스 생성 (280분 → 10분 후)")
X, y_true, time_info = create_sequences_for_evaluation(df, seq_length=280, pred_horizon=10)

print(f"\n3. 스케일러 로딩")
scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
X_scaled = scaler.transform(X)

print(f"\n4. 모델별 예측")
model_names = ['RandomForest', 'ExtraTrees', 'GradientBoosting', 'MLP', 'Ridge', 'ElasticNet']
predictions = {}

for name in model_names:
    print(f"   {name} 예측...", end='')
    model = joblib.load(os.path.join(model_dir, f'{name}_regression_model.pkl'))
    predictions[name] = model.predict(X_scaled)
    mae = np.mean(np.abs(predictions[name] - y_true))
    print(f" MAE: {mae:.2f}")

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
    
    for name in model_names:
        row[f'{name}_예측'] = round(predictions[name][i], 0)
    
    results.append(row)

results_df = pd.DataFrame(results)
results_df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"\n✅ 완료! 결과 파일: {output_file}")
print(f"   총 {len(results_df):,}개 예측 생성")

# 성능 요약
print(f"\n[모델별 성능 요약]")
print("="*60)
print(f"{'모델':15} {'MAE':>10} {'RMSE':>10}")
print("-"*40)

for name in model_names:
    errors = predictions[name] - y_true
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    print(f"{name:15} {mae:10.2f} {rmse:10.2f}")

print("="*60)

# 샘플 출력
print(f"\n[결과 샘플]")
print(results_df.head(10).to_string(index=False))