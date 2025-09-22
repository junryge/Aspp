# -*- coding: utf-8 -*-
"""
스케일러 생성 스크립트
학습 시 사용한 것과 동일한 스케일러 재생성
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

print("="*70)
print("스케일러 생성 스크립트")
print("="*70)

def create_sequences_for_scaler(data, seq_length=280, pred_horizon=10, max_samples=5000):
    """스케일러 학습용 시퀀스 생성 (일부만)"""
    
    feature_cols = ['M14AM14B', 'M14AM10A', 'M14AM16', 'M14AM14BSUM', 
                   'M14AM10ASUM', 'M14AM16SUM', 'M14BM14A', 'M10AM14A', 
                   'M16M14A', 'TOTALCNT']
    
    data = data.copy()
    
    # 파생 변수
    data['ratio_M14B_M14A'] = np.clip(data['M14AM14B'] / (data['M14AM10A'] + 1), 0, 1000)
    data['ratio_M14B_M16'] = np.clip(data['M14AM14B'] / (data['M14AM16'] + 1), 0, 1000)
    data['totalcnt_change'] = data['TOTALCNT'].diff().fillna(0)
    data['totalcnt_pct_change'] = data['TOTALCNT'].pct_change().fillna(0).clip(-10, 10)
    
    data = data.replace([np.inf, -np.inf], 0).fillna(0)
    
    X_list = []
    n_sequences = min(max_samples, len(data) - seq_length - pred_horizon + 1)
    
    print(f"  {n_sequences}개 샘플로 스케일러 학습")
    
    for i in range(n_sequences):
        if i % 100 == 0:
            print(f"  진행: {i}/{n_sequences}", end='\r')
        
        start = i
        end = i + seq_length
        seq_data = data.iloc[start:end]
        
        features = []
        
        # 각 컬럼별 특징 (13개씩)
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
                np.max(values[-30:])
            ])
            
            # TOTALCNT 특별 처리
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
        
        # 파생 변수
        last_idx = end - 1
        features.extend([
            np.clip(data['ratio_M14B_M14A'].iloc[last_idx], 0, 1000),
            np.clip(data['ratio_M14B_M16'].iloc[last_idx], 0, 1000),
            np.clip(data['totalcnt_change'].iloc[last_idx], -1000, 1000),
            np.clip(data['totalcnt_pct_change'].iloc[last_idx], -10, 10),
        ])
        
        X_list.append(features)
    
    X = np.array(X_list)
    X = np.nan_to_num(X, nan=0.0, posinf=1000.0, neginf=-1000.0)
    
    print(f"\n  특징 행렬 생성 완료: {X.shape}")
    return X

# 메인 실행
print("\n[Step 1] 데이터 로딩...")

# uu.csv와 uu2.csv 확인
if os.path.exists('uu.csv') and os.path.exists('uu2.csv'):
    df1 = pd.read_csv('uu.csv')
    df2 = pd.read_csv('uu2.csv')
    df = pd.concat([df1, df2], ignore_index=True).reset_index(drop=True)
    print(f"  uu.csv + uu2.csv 로드 완료: {len(df):,}행")
elif os.path.exists('uu.csv'):
    df = pd.read_csv('uu.csv')
    print(f"  uu.csv 로드 완료: {len(df):,}행")
else:
    # 다른 CSV 파일 찾기
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    if csv_files:
        print(f"  uu.csv 없음 - {csv_files[0]} 사용")
        df = pd.read_csv(csv_files[0])
        print(f"  {csv_files[0]} 로드 완료: {len(df):,}행")
    else:
        print("  CSV 파일이 없습니다!")
        exit()

# 필수 컬럼 확인
required_cols = ['M14AM14B', 'M14AM10A', 'M14AM16', 'M14AM14BSUM', 
                'M14AM10ASUM', 'M14AM16SUM', 'M14BM14A', 'M10AM14A', 
                'M16M14A', 'TOTALCNT']

missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f"  필수 컬럼 누락: {missing_cols}")
    exit()

print("\n[Step 2] 특징 생성 (스케일러 학습용)...")
X_for_scaler = create_sequences_for_scaler(df, seq_length=280, pred_horizon=10, max_samples=5000)

print("\n[Step 3] 스케일러 학습...")
scaler = StandardScaler()
scaler.fit(X_for_scaler)

print(f"  평균: {scaler.mean_[:5]} ... (처음 5개)")
print(f"  표준편차: {scaler.scale_[:5]} ... (처음 5개)")

print("\n[Step 4] 스케일러 저장...")

# 여러 위치에 저장
save_paths = [
    'scaler.pkl',
    'regression_classification_models_280to10/scaler.pkl',
    'checkpoint/scaler.pkl'
]

for path in save_paths:
    try:
        # 폴더 생성
        dir_name = os.path.dirname(path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)
        
        # 저장
        joblib.dump(scaler, path)
        print(f"  저장: {path}")
    except Exception as e:
        print(f"  실패: {path} - {e}")

print("\n" + "="*70)
print("스케일러 생성 완료!")
print("="*70)
print("\n사용 방법:")
print("  1. evaluate_extratrees.py 실행")
print("  2. 또는 학습 코드에서 사용")