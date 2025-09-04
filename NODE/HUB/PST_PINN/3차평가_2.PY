# -*- coding: utf-8 -*-
"""
HUBROOM 모델 평가 - 실제 CSV 데이터 평가
- 202509월.csv 데이터 로드
- 과거 20분 데이터로 10분 후 예측
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🎯 HUBROOM 모델 평가 - 실제 CSV 데이터")
print("📅 과거 20분 → 10분 후 예측")
print("="*80)

# ========================================
# Model 1: PatchTST
# ========================================

class PatchTSTModel(keras.Model):
    def __init__(self, config):
        super().__init__()
        
        self.seq_len = config['seq_len']
        self.n_features = config['n_features']
        self.patch_len = config['patch_len']
        self.n_patches = self.seq_len // self.patch_len
        
        self.patch_embedding = layers.Dense(128, activation='relu')
        self.attention = layers.MultiHeadAttention(num_heads=8, key_dim=16)
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        
        self.ffn = keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(128)
        ])
        
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout = layers.Dropout(0.3)
        self.dense2 = layers.Dense(64, activation='relu')
        self.output_layer = layers.Dense(1)
        
    def call(self, x, training=False):
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, [batch_size, self.n_patches, self.patch_len * self.n_features])
        x = self.patch_embedding(x)
        attn = self.attention(x, x, training=training)
        x = self.norm1(x + attn)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        output = self.output_layer(x)
        return tf.squeeze(output, axis=-1)

# ========================================
# Model 2: PatchTST + PINN
# ========================================

class PatchTSTPINN(keras.Model):
    def __init__(self, config):
        super().__init__()
        
        self.seq_len = config['seq_len']
        self.n_features = config['n_features']
        self.patch_len = config['patch_len']
        self.n_patches = self.seq_len // self.patch_len
        
        self.patch_embedding = layers.Dense(128, activation='relu')
        self.attention = layers.MultiHeadAttention(num_heads=8, key_dim=16)
        self.norm = layers.LayerNormalization()
        self.flatten = layers.Flatten()
        self.temporal_dense = layers.Dense(64, activation='relu')
        
        self.physics_net = keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(16, activation='relu')
        ])
        
        self.fusion = keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])
        
        self.extreme_boost = layers.Dense(1, activation='sigmoid')
        
    def call(self, inputs, training=False):
        x_seq, x_physics = inputs
        batch_size = tf.shape(x_seq)[0]
        
        x = tf.reshape(x_seq, [batch_size, self.n_patches, self.patch_len * self.n_features])
        x = self.patch_embedding(x)
        attn = self.attention(x, x, training=training)
        x = self.norm(x + attn)
        x = self.flatten(x)
        temporal_features = self.temporal_dense(x)
        
        physics_features = self.physics_net(x_physics)
        combined = tf.concat([temporal_features, physics_features], axis=-1)
        output = self.fusion(combined)
        
        boost_factor = self.extreme_boost(combined)
        output = output * (1 + boost_factor * 0.2)
        
        return tf.squeeze(output, axis=-1)

# ========================================
# 데이터 처리 클래스
# ========================================

class DataProcessor:
    def __init__(self):
        self.target_col = 'CURRENT_M16A_3F_JOB_2'
        self.scaler_X = RobustScaler()
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))
        self.scaler_physics = StandardScaler()
        
        # 물리 법칙용 컬럼
        self.inflow_cols = [
            'M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2',
            'M14A_3F_TO_HUB_JOB2', 'M14B_7F_TO_HUB_JOB2'
        ]
        self.outflow_cols = [
            'M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB',
            'M16A_3F_TO_M14A_3F_JOB', 'M16A_3F_TO_M14B_7F_JOB'
        ]
    
    def create_sequences(self, df, seq_len=20, pred_len=10):
        """과거 20분 데이터로 10분 후 예측을 위한 시퀀스 생성"""
        
        X, y, X_physics = [], [], []
        timestamps = []
        
        # 타겟 컬럼 확인
        if self.target_col not in df.columns:
            print(f"⚠️ 타겟 컬럼 '{self.target_col}'이 없습니다.")
            # 타겟 컬럼이 다른 이름일 수 있으므로 확인
            possible_targets = [col for col in df.columns if 'CURRENT' in col and 'M16A_3F' in col]
            if possible_targets:
                self.target_col = possible_targets[0]
                print(f"✅ 대체 타겟 컬럼 사용: {self.target_col}")
            else:
                raise ValueError("타겟 컬럼을 찾을 수 없습니다.")
        
        # 특징 컬럼 선택
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'datetime']]
        
        # 물리 법칙 컬럼 확인
        available_inflow = [col for col in self.inflow_cols if col in df.columns]
        available_outflow = [col for col in self.outflow_cols if col in df.columns]
        
        if not available_inflow:
            available_inflow = [col for col in df.columns if 'TO_HUB' in col][:4]
        if not available_outflow:
            available_outflow = [col for col in df.columns if 'M16A_3F_TO' in col][:4]
        
        # 시퀀스 생성
        for i in range(len(df) - seq_len - pred_len + 1):
            # 입력: i부터 i+seq_len까지 (20분)
            X.append(df[feature_cols].iloc[i:i+seq_len].values)
            
            # 출력: i+seq_len+pred_len-1 시점 (10분 후)
            y.append(df[self.target_col].iloc[i+seq_len+pred_len-1])
            
            # 물리 특징
            if available_inflow and available_outflow:
                inflow = df[available_inflow].iloc[i+seq_len-1].sum()
                outflow = df[available_outflow].iloc[i+seq_len-1].sum()
                net_flow = inflow - outflow
                X_physics.append([inflow, outflow, net_flow])
            else:
                X_physics.append([0, 0, 0])
            
            # 타임스탬프
            if 'timestamp' in df.columns:
                timestamps.append(df['timestamp'].iloc[i+seq_len+pred_len-1])
            else:
                timestamps.append(df.index[i+seq_len+pred_len-1])
        
        return np.array(X), np.array(y), np.array(X_physics), timestamps

# ========================================
# CSV 데이터 로드 및 평가
# ========================================

def evaluate_csv_data(csv_path='data/202509월.csv'):
    """실제 CSV 데이터로 평가"""
    
    print(f"\n📂 CSV 데이터 로드 중: {csv_path}")
    
    # CSV 로드
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ CSV 로드 완료: {df.shape}")
    except Exception as e:
        print(f"❌ CSV 로드 실패: {e}")
        print("💡 다른 경로 시도...")
        
        # 다른 가능한 경로들
        possible_paths = [
            './data/202509월.csv',
            '../data/202509월.csv',
            './202509월.csv',
            'data/20250901_TO_20250930.CSV',
            'data/HUB_0509_TO_0730_DATA.CSV'
        ]
        
        for path in possible_paths:
            try:
                df = pd.read_csv(path)
                print(f"✅ CSV 로드 성공: {path}")
                csv_path = path
                break
            except:
                continue
        else:
            raise FileNotFoundError("CSV 파일을 찾을 수 없습니다.")
    
    print(f"\n📊 데이터 정보:")
    print(f"  - 행 수: {len(df)}")
    print(f"  - 컬럼 수: {len(df.columns)}")
    print(f"  - 컬럼 목록: {list(df.columns[:10])}...")
    
    # 타임스탬프 처리
    if df.columns[0].isdigit() or 'timestamp' not in df.columns:
        # 첫 번째 컬럼이 타임스탬프인 경우
        df['timestamp'] = pd.to_datetime(df.iloc[:, 0], format='%Y%m%d%H%M', errors='coerce')
    
    # 데이터 정렬 및 전처리
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp').reset_index(drop=True)
    
    df = df.fillna(method='ffill').fillna(0)
    
    # 숫자형 컬럼만 선택
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df_numeric = df[numeric_cols]
    
    if 'timestamp' in df.columns:
        df_numeric['timestamp'] = df['timestamp']
    
    # 데이터 프로세서 생성
    processor = DataProcessor()
    
    # 시퀀스 생성
    print("\n📈 시퀀스 생성 중...")
    X, y, X_physics, timestamps = processor.create_sequences(df_numeric, seq_len=20, pred_len=10)
    
    print(f"✅ 시퀀스 생성 완료:")
    print(f"  - 샘플 수: {len(X)}")
    print(f"  - 입력 shape: {X.shape}")
    print(f"  - 타겟 shape: {y.shape}")
    print(f"  - 물리 특징 shape: {X_physics.shape}")
    
    # 타겟 분석
    print(f"\n🎯 타겟({processor.target_col}) 분석:")
    print(f"  - 범위: {y.min():.0f} ~ {y.max():.0f}")
    print(f"  - 평균: {y.mean():.1f}")
    print(f"  - 극단값(310+): {(y >= 310).sum()}개 ({(y >= 310).sum()/len(y)*100:.2f}%)")
    print(f"  - 초극단값(335+): {(y >= 335).sum()}개 ({(y >= 335).sum()/len(y)*100:.2f}%)")
    
    # 데이터 스케일링
    print("\n🔧 데이터 스케일링 중...")
    
    # 스케일러 로드 또는 새로 생성
    try:
        processor.scaler_X = joblib.load('./scalers/scaler_X.pkl')
        processor.scaler_y = joblib.load('./scalers/scaler_y.pkl')
        processor.scaler_physics = joblib.load('./scalers/scaler_physics.pkl')
        print("✅ 기존 스케일러 로드")
    except:
        print("💡 새 스케일러 생성")
        # Fit scalers
        X_flat = X.reshape(-1, X.shape[2])
        processor.scaler_X.fit(X_flat)
        processor.scaler_y.fit(y.reshape(-1, 1))
        processor.scaler_physics.fit(X_physics)
    
    # 스케일링 적용
    X_scaled = processor.scaler_X.transform(X.reshape(-1, X.shape[2])).reshape(X.shape)
    y_scaled = processor.scaler_y.transform(y.reshape(-1, 1)).flatten()
    X_physics_scaled = processor.scaler_physics.transform(X_physics)
    
    # 모델 설정
    n_features = X.shape[2]
    config = {
        'seq_len': 20,
        'n_features': n_features,
        'patch_len': 5
    }
    
    # Model 1 예측
    print("\n🤖 Model 1 (PatchTST) 예측 중...")
    model1 = PatchTSTModel(config)
    model1.compile(optimizer='adam', loss='mse')
    
    # 더미 입력으로 모델 빌드
    dummy_input = np.zeros((1, 20, n_features))
    _ = model1(dummy_input)
    
    try:
        model1.load_weights('./checkpoints/model1_v3.h5')
        print("✅ Model1 가중치 로드")
    except:
        print("⚠️ Model1 가중치 없음 - 랜덤 초기화 상태로 예측")
    
    y_pred1_scaled = model1.predict(X_scaled, batch_size=32, verbose=0)
    y_pred1 = processor.scaler_y.inverse_transform(y_pred1_scaled.reshape(-1, 1)).flatten()
    print("✅ Model 1 예측 완료")
    
    # Model 2 예측
    print("\n🤖 Model 2 (PatchTST + PINN) 예측 중...")
    model2 = PatchTSTPINN(config)
    model2.compile(optimizer='adam', loss='mse')
    
    # 더미 입력으로 모델 빌드
    dummy_seq = np.zeros((1, 20, n_features))
    dummy_physics = np.zeros((1, 3))
    _ = model2([dummy_seq, dummy_physics])
    
    try:
        model2.load_weights('./checkpoints/model2_v3.h5')
        print("✅ Model2 가중치 로드")
    except:
        print("⚠️ Model2 가중치 없음 - 랜덤 초기화 상태로 예측")
    
    y_pred2_scaled = model2.predict([X_scaled, X_physics_scaled], batch_size=32, verbose=0)
    y_pred2 = processor.scaler_y.inverse_transform(y_pred2_scaled.reshape(-1, 1)).flatten()
    print("✅ Model 2 예측 완료")
    
    # 앙상블
    y_ensemble = 0.4 * y_pred1 + 0.6 * y_pred2
    print("✅ Ensemble 예측 완료")
    
    # 결과 DataFrame 생성
    print("\n📊 결과 정리 중...")
    
    results = []
    for i in range(len(y)):
        # 타임스탬프 처리
        if isinstance(timestamps[i], pd.Timestamp):
            target_time = timestamps[i]
            timestamp_str = target_time.strftime('%Y%m%d%H%M')
            readable_time = target_time.strftime('%Y-%m-%d %H:%M')
        else:
            timestamp_str = str(timestamps[i])
            try:
                target_time = datetime.strptime(timestamp_str, '%Y%m%d%H%M')
                readable_time = target_time.strftime('%Y-%m-%d %H:%M')
            except:
                readable_time = timestamp_str
                target_time = datetime.now()
        
        # 입력 시간 계산
        input_start = target_time - timedelta(minutes=30)
        input_end = target_time - timedelta(minutes=10)
        
        result = {
            'index': i,
            'timestamp': timestamp_str,
            'readable_time': readable_time,
            'actual': float(y[i]),
            'predicted_model1': float(y_pred1[i]),
            'predicted_model2': float(y_pred2[i]),
            'predicted_ensemble': float(y_ensemble[i]),
            'Target_time': readable_time,
            'input_start': input_start.strftime('%Y-%m-%d %H:%M'),
            'input_end': input_end.strftime('%Y-%m-%d %H:%M'),
            'error_model1': abs(float(y[i]) - float(y_pred1[i])),
            'error_model2': abs(float(y[i]) - float(y_pred2[i])),
            'error_ensemble': abs(float(y[i]) - float(y_ensemble[i])),
            'is_extreme': float(y[i]) >= 310,
            'is_very_extreme': float(y[i]) >= 335
        }
        
        # 최고 성능 모델
        min_error = min(result['error_model1'], result['error_model2'], result['error_ensemble'])
        if result['error_model1'] == min_error:
            result['best_model'] = 'Model1'
        elif result['error_model2'] == min_error:
            result['best_model'] = 'Model2'
        else:
            result['best_model'] = 'Ensemble'
        
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    # 결과 저장
    os.makedirs('./evaluation_results', exist_ok=True)
    results_df.to_csv('./evaluation_results/csv_predictions.csv', index=False)
    print(f"✅ 결과 저장: ./evaluation_results/csv_predictions.csv")
    
    # 극단값만 저장
    extreme_df = results_df[results_df['is_extreme']].copy()
    if len(extreme_df) > 0:
        extreme_df.to_csv('./evaluation_results/csv_extreme_predictions.csv', index=False)
        print(f"✅ 극단값 저장: ./evaluation_results/csv_extreme_predictions.csv")
    
    # 성능 요약
    print("\n" + "="*80)
    print("📈 전체 성능 요약")
    print("="*80)
    
    print(f"\n📊 평가 데이터: {len(results_df)}개 샘플")
    
    mae1 = results_df['error_model1'].mean()
    mae2 = results_df['error_model2'].mean()
    mae_ensemble = results_df['error_ensemble'].mean()
    
    print(f"\nMAE (Mean Absolute Error):")
    print(f"  Model 1: {mae1:.2f}")
    print(f"  Model 2: {mae2:.2f}")
    print(f"  Ensemble: {mae_ensemble:.2f}")
    
    # 극단값 성능
    if len(extreme_df) > 0:
        print(f"\n🔴 극단값 성능 ({len(extreme_df)}개):")
        print(f"  Model 1 MAE: {extreme_df['error_model1'].mean():.2f}")
        print(f"  Model 2 MAE: {extreme_df['error_model2'].mean():.2f}")
        print(f"  Ensemble MAE: {extreme_df['error_ensemble'].mean():.2f}")
        
        # 극단값 감지율
        detected_m1 = (extreme_df['predicted_model1'] >= 305).sum()
        detected_m2 = (extreme_df['predicted_model2'] >= 305).sum()
        print(f"\n극단값 감지율 (예측 >= 305):")
        print(f"  Model 1: {detected_m1}/{len(extreme_df)} ({detected_m1/len(extreme_df)*100:.1f}%)")
        print(f"  Model 2: {detected_m2}/{len(extreme_df)} ({detected_m2/len(extreme_df)*100:.1f}%)")
    
    # 샘플 출력
    print("\n" + "="*80)
    print("📋 예측 결과 샘플")
    print("="*80)
    
    # 처음 10개
    print("\n[처음 10개]")
    for idx, row in results_df.head(10).iterrows():
        mark = "🔴" if row['is_extreme'] else "⚪"
        print(f"{mark} {row['readable_time']}: 실제={row['actual']:.0f}, M1={row['predicted_model1']:.0f}, M2={row['predicted_model2']:.0f}, Ens={row['predicted_ensemble']:.0f}")
    
    # 극단값 샘플
    if len(extreme_df) > 0:
        print("\n[극단값 샘플]")
        for idx, row in extreme_df.head(5).iterrows():
            print(f"🔴 {row['readable_time']}: 실제={row['actual']:.0f}, M1={row['predicted_model1']:.0f}, M2={row['predicted_model2']:.0f}, Best={row['best_model']}")
    
    print("\n✅ 평가 완료!")
    return results_df

# ========================================
# 메인 실행
# ========================================

if __name__ == "__main__":
    try:
        # CSV 파일 경로 설정
        csv_path = input("CSV 파일 경로를 입력하세요 (기본: data/202509월.csv): ").strip()
        if not csv_path:
            csv_path = 'data/202509월.csv'
        
        # 평가 실행
        results = evaluate_csv_data(csv_path)
        
        print("\n" + "="*80)
        print("🎉 모든 평가가 완료되었습니다!")
        print(f"📁 결과 파일: ./evaluation_results/csv_predictions.csv")
        print(f"📊 총 {len(results)}개 데이터 평가 완료")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()