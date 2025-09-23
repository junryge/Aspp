# -*- coding: utf-8 -*-
"""
최종 보정 시스템 v2 - 시퀀스 50개 추세 분석
1700+ 구간에서 상승/하락 추세에 따른 동적 보정
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("최종 보정 v2 - 시퀀스 50개 추세 기반 동적 보정")
print("="*80)

# 설정
model_dir = 'regression_classification_models_280to10'
data_file = 'data/20250731_to_20250806.csv'
output_file = 'evaluation_trend_final.csv'

MODEL_NAMES = ['RandomForest', 'ExtraTrees', 'GradientBoosting', 'MLP']

def analyze_trend_50(seq_last_50):
    """
    마지막 50개 데이터로 추세 분석
    return: 'rising', 'falling', 'stable'
    """
    if len(seq_last_50) < 50:
        seq_last_50 = seq_last_50[-min(len(seq_last_50), 50):]
    
    # 1. 선형 회귀 기울기
    x = np.arange(len(seq_last_50))
    slope, _ = np.polyfit(x, seq_last_50, 1)
    
    # 2. 전반부 vs 후반부 평균
    mid = len(seq_last_50) // 2
    first_half_avg = np.mean(seq_last_50[:mid])
    second_half_avg = np.mean(seq_last_50[mid:])
    avg_diff = second_half_avg - first_half_avg
    
    # 3. 최근 10개 vs 이전 10개
    recent_10 = np.mean(seq_last_50[-10:])
    prev_10 = np.mean(seq_last_50[-20:-10]) if len(seq_last_50) >= 20 else np.mean(seq_last_50[:10])
    recent_trend = recent_10 - prev_10
    
    # 종합 판단
    if slope > 1.0 and avg_diff > 20 and recent_trend > 10:
        return 'rising'  # 강한 상승
    elif slope < -1.0 and avg_diff < -20 and recent_trend < -10:
        return 'falling'  # 강한 하락
    else:
        return 'stable'  # 보합

def trend_based_correction(prediction, current_totalcnt, seq_max, seq_last_50):
    """
    추세 기반 동적 보정 (시퀀스 50개 분석)
    """
    
    # 추세 분석
    trend = analyze_trend_50(seq_last_50)
    
    # 1700+ 구간 - 추세별 차등 보정
    if current_totalcnt >= 1700:
        error = current_totalcnt - prediction
        
        if trend == 'rising':  # 상승 추세
            # 더 강한 보정
            if prediction < 1550:
                return prediction * 1.20  # 20% (상승 시)
            elif prediction < 1600:
                return prediction * 1.15  # 15%
            elif prediction < 1650:
                return prediction * 1.10  # 10%
            elif prediction < 1700:
                return prediction * 1.07  # 7%
            else:
                return prediction * 1.03  # 3%
                
        elif trend == 'falling':  # 하락 추세
            # 보수적 보정
            if prediction < 1550:
                return prediction * 1.15  # 15% (하락 시)
            elif prediction < 1600:
                return prediction * 1.10  # 10%
            elif prediction < 1650:
                return prediction * 1.06  # 6%
            elif prediction < 1700:
                return prediction * 1.04  # 4%
            else:
                return prediction * 1.01  # 1%
                
        else:  # stable (보합)
            # 중간 보정
            if prediction < 1550:
                return prediction * 1.18  # 18%
            elif prediction < 1600:
                return prediction * 1.12  # 12%
            elif prediction < 1650:
                return prediction * 1.08  # 8%
            elif prediction < 1700:
                return prediction * 1.05  # 5%
            else:
                return prediction * 1.02  # 2%
    
    # 1680-1699 구간
    elif 1680 <= current_totalcnt < 1700:
        trend = analyze_trend_50(seq_last_50)
        
        if seq_max >= 1700:  # 시퀀스에 1700+ 있음
            if trend == 'rising':
                # 상승 가능성 높음
                if prediction < 1600:
                    return prediction * 1.12  # 12%
                elif prediction < 1650:
                    return prediction * 1.08  # 8%
                else:
                    return prediction * 1.05  # 5%
            else:
                # 일반 보정
                if prediction < 1600:
                    return prediction * 1.10  # 10%
                elif prediction < 1650:
                    return prediction * 1.07  # 7%
                else:
                    return prediction * 1.04  # 4%
        else:
            if prediction < current_totalcnt - 30:
                return prediction * 1.06  # 6%
            else:
                return prediction * 1.03  # 3%
    
    # 1651-1682 구간
    elif 1651 <= current_totalcnt <= 1682:
        if prediction < 1600:
            return prediction * 1.08  # 8%
        elif prediction < 1650:
            return prediction * 1.05  # 5%
        else:
            return prediction * 1.03  # 3%
    
    # 1600-1650 구간
    elif 1600 <= current_totalcnt < 1651:
        if seq_max >= 1700 and prediction < current_totalcnt:
            trend = analyze_trend_50(seq_last_50)
            if trend == 'rising':
                return prediction * 1.05  # 5%
            else:
                return prediction * 1.04  # 4%
    
    return prediction

def create_sequences_for_evaluation(data, seq_length=280, pred_horizon=10):
    """평가용 시퀀스 생성 (마지막 50개 포함)"""
    
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
            # 마지막 50개 데이터 저장
            seq_last_50 = seq_data['TOTALCNT'].values[-50:]
            
            time_info = {
                'current_idx': end_idx - 1,
                'target_idx': target_idx,
                'current_totalcnt': data['TOTALCNT'].iloc[end_idx - 1],
                'seq_max': np.max(seq_data['TOTALCNT'].values),
                'seq_min': np.min(seq_data['TOTALCNT'].values),
                'seq_last_50': seq_last_50,  # 추세 분석용
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

print(f"\n4. 추세 기반 동적 보정 예측")
print("="*80)

predictions = {}
corrected_predictions = {}
trend_stats = {'rising': 0, 'falling': 0, 'stable': 0}

for name in MODEL_NAMES:
    print(f"\n{name}:")
    model = joblib.load(os.path.join(model_dir, f'{name}_regression_model.pkl'))
    
    raw_pred = model.predict(X_scaled)
    
    # 추세 기반 보정
    corrected_pred = []
    for i in range(len(raw_pred)):
        current_val = time_info[i]['current_totalcnt']
        seq_max = time_info[i]['seq_max']
        seq_last_50 = time_info[i]['seq_last_50']
        
        corrected_val = trend_based_correction(raw_pred[i], current_val, seq_max, seq_last_50)
        corrected_pred.append(corrected_val)
        
        # 추세 통계 (한 모델만)
        if name == 'GradientBoosting' and current_val >= 1700:
            trend = analyze_trend_50(seq_last_50)
            trend_stats[trend] += 1
    
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

# 추세 통계 출력
print(f"\n1700+ 구간 추세 분포:")
total_1700 = sum(trend_stats.values())
if total_1700 > 0:
    for trend, count in trend_stats.items():
        print(f"  {trend}: {count}개 ({count/total_1700*100:.1f}%)")

print("="*80)

print(f"\n5. 결과 CSV 생성")
results = []

for i in range(len(X)):
    current_time = df['DateTime'].iloc[time_info[i]['current_idx']]
    target_time = df['DateTime'].iloc[time_info[i]['target_idx']]
    
    # 추세 계산
    trend = analyze_trend_50(time_info[i]['seq_last_50'])
    
    row = {
        '날짜': current_time.strftime('%Y-%m-%d %H:%M'),
        '타켓날짜': target_time.strftime('%Y-%m-%d %H:%M'),
        '실제값': round(y_true[i], 0),
        '시퀀스MAX': round(time_info[i]['seq_max'], 0),
        '시퀀스MIN': round(time_info[i]['seq_min'], 0),
        '추세': trend,
    }
    
    # 각 모델 예측값
    for name in MODEL_NAMES:
        row[f'{name}_예측'] = round(corrected_predictions[name][i], 0)
        row[f'{name}_오차'] = round(corrected_predictions[name][i] - y_true[i], 0)
    
    results.append(row)

results_df = pd.DataFrame(results)
results_df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"\n✅ 완료! 결과 파일: {output_file}")

# 1700+ 구간 상세 분석
print(f"\n[1700+ 구간 추세별 성능]")
print("-"*100)
high_samples = results_df[results_df['실제값'] >= 1700]

if len(high_samples) > 0:
    # 추세별 그룹화
    for trend in ['rising', 'falling', 'stable']:
        trend_samples = high_samples[high_samples['추세'] == trend]
        if len(trend_samples) > 0:
            print(f"\n{trend.upper()} 추세 ({len(trend_samples)}개):")
            for name in MODEL_NAMES:
                errors = trend_samples[f'{name}_오차'].abs()
                mae = errors.mean()
                print(f"  {name}: MAE={mae:.1f}")
    
    # 샘플 출력
    print(f"\n예측 결과 샘플 (처음 15개):")
    display_cols = ['날짜', '실제값', '추세', 'RandomForest_예측', 'GradientBoosting_예측']
    print(high_samples.head(15)[display_cols].to_string(index=False))

print(f"\n💡 추세별 보정 전략:")
print("  상승(rising): 1700+ 구간에서 최대 20% 보정")
print("  하락(falling): 1700+ 구간에서 최대 15% 보정 (보수적)")
print("  보합(stable): 1700+ 구간에서 최대 18% 보정")