# -*- coding: utf-8 -*-
"""
ExtraTrees 모델로 CSV 데이터 평가
280분 → 10분 후 예측
"""

import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import warnings
warnings.filterwarnings('ignore')

def create_features_from_280min(data_280min):
    """280분 데이터에서 147개 특징 추출 (학습 때와 동일)"""
    
    feature_cols = ['M14AM14B', 'M14AM10A', 'M14AM16', 'M14AM14BSUM', 
                   'M14AM10ASUM', 'M14AM16SUM', 'M14BM14A', 'M10AM14A', 
                   'M16M14A', 'TOTALCNT']
    
    # DataFrame으로 변환
    if isinstance(data_280min, np.ndarray):
        df = pd.DataFrame(data_280min, columns=feature_cols)
    else:
        df = data_280min
    
    # 파생 변수
    df['ratio_M14B_M14A'] = np.clip(df['M14AM14B'] / (df['M14AM10A'] + 1), 0, 1000)
    df['ratio_M14B_M16'] = np.clip(df['M14AM14B'] / (df['M14AM16'] + 1), 0, 1000)
    df['totalcnt_change'] = df['TOTALCNT'].diff().fillna(0)
    df['totalcnt_pct_change'] = df['TOTALCNT'].pct_change().fillna(0).clip(-10, 10)
    
    features = []
    
    # 각 컬럼별 13개 특징 (학습 때와 동일)
    for col in feature_cols:
        values = df[col].values
        
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
    
    # 파생 변수 추가
    features.extend([
        np.clip(df['ratio_M14B_M14A'].iloc[-1], 0, 1000),
        np.clip(df['ratio_M14B_M16'].iloc[-1], 0, 1000),
        np.clip(df['totalcnt_change'].iloc[-1], -1000, 1000),
        np.clip(df['totalcnt_pct_change'].iloc[-1], -10, 10),
    ])
    
    return np.array(features)

def evaluate_csv_with_extratrees(csv_file, model_path, scaler_path=None):
    """CSV 파일을 ExtraTrees 모델로 평가"""
    
    print("="*70)
    print("🌲 ExtraTrees 모델 평가")
    print("="*70)
    
    # 1. 모델 로드
    print("\n📁 모델 로딩...")
    model = joblib.load(model_path)
    print(f"  ✅ 모델: {model_path}")
    
    # 스케일러 로드 또는 생성
    if scaler_path and os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"  ✅ 스케일러: {scaler_path}")
    else:
        print("  ⚠️ 스케일러 없음 - 새로 생성합니다")
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler_fitted = False
    
    # 2. CSV 데이터 로드
    print(f"\n📂 데이터 로딩: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"  원본: {len(df):,}행")
    
    # 필수 컬럼 확인
    required_cols = ['M14AM14B', 'M14AM10A', 'M14AM16', 'M14AM14BSUM', 
                     'M14AM10ASUM', 'M14AM16SUM', 'M14BM14A', 'M10AM14A', 
                     'M16M14A', 'TOTALCNT']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ 필수 컬럼 누락: {missing_cols}")
        return None
    
    # 3. 예측 가능 범위
    seq_length = 280
    pred_horizon = 10
    
    start_idx = seq_length
    end_idx = len(df) - pred_horizon
    total = end_idx - start_idx
    
    if total <= 0:
        print("❌ 데이터가 부족합니다 (최소 290분 필요)")
        return None
    
    print(f"\n🔮 예측 시작...")
    print(f"  시퀀스: 280분 → 10분 후")
    print(f"  예측 개수: {total:,}개")
    
    # 4. 스케일러가 없으면 데이터로 학습
    if scaler_path is None or not os.path.exists(scaler_path):
        print("\n📊 스케일러 생성 중...")
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        
        # 모든 특징 수집하여 스케일러 학습
        all_features = []
        sample_size = min(1000, total)  # 최대 1000개 샘플로 학습
        
        for i in range(start_idx, min(start_idx + sample_size, end_idx)):
            seq_data = df.iloc[i-seq_length:i]
            features = create_features_from_280min(seq_data)
            all_features.append(features)
        
        all_features = np.array(all_features)
        scaler.fit(all_features)
        
        # 스케일러 저장
        joblib.dump(scaler, 'generated_scaler.pkl')
        print(f"  ✅ 스케일러 생성 완료! (generated_scaler.pkl로 저장)")
    
    # 5. 예측 수행
    predictions = []
    actuals = []
    
    for i in range(start_idx, end_idx):
        if i % 100 == 0:
            print(f"  진행: {i-start_idx}/{total} ({(i-start_idx)/total*100:.1f}%)", end='\r')
        
        # 280분 시퀀스 추출
        seq_data = df.iloc[i-seq_length:i]
        
        # 특징 추출
        features = create_features_from_280min(seq_data)
        
        # 스케일링
        features_scaled = scaler.transform([features])
        
        # 예측
        pred = model.predict(features_scaled)[0]
        predictions.append(pred)
        
        # 실제값 (10분 후)
        actual_idx = i + pred_horizon
        if actual_idx < len(df):
            actuals.append(df.iloc[actual_idx]['TOTALCNT'])
    
    print("\n✅ 예측 완료!")
    
    # 5. 성능 평가
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    r2 = r2_score(actuals, predictions)
    mape = np.mean(abs(predictions - actuals) / actuals) * 100
    
    print(f"\n📊 성능 지표:")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²: {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  정확도: {100-mape:.2f}%")
    
    # 6. 결과 저장
    results = pd.DataFrame({
        '실제값': actuals,
        '예측값': predictions.round(),
        '오차': (predictions - actuals).round(),
        '오차율(%)': (abs(predictions - actuals) / actuals * 100).round(2)
    })
    
    # 위험 구간 분석
    danger_actual = (actuals >= 1700).sum()
    danger_pred = (predictions >= 1700).sum()
    
    print(f"\n🎯 위험 구간(1700+) 분석:")
    print(f"  실제: {danger_actual}개 ({danger_actual/len(actuals)*100:.2f}%)")
    print(f"  예측: {danger_pred}개 ({danger_pred/len(predictions)*100:.2f}%)")
    
    # CSV 저장
    output_file = 'extratrees_evaluation_results.csv'
    results.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n💾 결과 저장: {output_file}")
    
    return results

# 사용 예시
if __name__ == "__main__":
    
    # 파일 경로 설정
    csv_file = 'M14_20250916_20250918.csv'  # 평가할 CSV
    model_path = 'regression_classification_models_280to10/ExtraTrees_model.pkl'
    scaler_path = 'regression_classification_models_280to10/scaler.pkl'  # 없어도 됨
    
    # 파일 존재 확인
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일이 없습니다: {model_path}")
        print("먼저 학습을 완료하세요!")
    elif not os.path.exists(csv_file):
        print(f"❌ CSV 파일이 없습니다: {csv_file}")
    else:
        # 스케일러 없어도 실행
        if not os.path.exists(scaler_path):
            print(f"⚠️ 스케일러 없음 - 자동 생성됩니다")
            scaler_path = None
        
        # 평가 실행
        results = evaluate_csv_with_extratrees(csv_file, model_path, scaler_path)
        
        if results is not None:
            print("\n✅ 평가 완료!")
            print(f"총 {len(results)}개 예측 수행")