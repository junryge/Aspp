# -*- coding: utf-8 -*-
"""
HUBROOM 모델 평가 - CSV 데이터 예측
- CSV 파일 로드
- 과거 20분 데이터로 10분 후 예측 (시퀀스 생성)
- 전체 데이터 평가
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🎯 HUBROOM 모델 평가 - CSV 데이터")
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
# 메인 평가 함수 - CSV 데이터
# ========================================

def evaluate_csv_data(csv_path='data/202509월.csv'):
    """CSV 파일로 평가 - 과거 20분 → 10분 후 예측"""
    
    print(f"\n📂 CSV 파일 로드: {csv_path}")
    
    # CSV 로드
    df = pd.read_csv(csv_path)
    print(f"✅ 로드 완료: {df.shape[0]}행 x {df.shape[1]}열")
    
    # 타임스탬프 처리
    df['timestamp'] = pd.to_datetime(df.iloc[:, 0], format='%Y%m%d%H%M', errors='coerce')
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df.fillna(method='ffill').fillna(0)
    
    # 타겟 컬럼
    target_col = 'CURRENT_M16A_3F_JOB_2'
    if target_col not in df.columns:
        # 타겟 컬럼 찾기
        possible = [col for col in df.columns if 'CURRENT' in col and 'M16A_3F' in col]
        if possible:
            target_col = possible[0]
        else:
            # 숫자 컬럼 중 적절한 것 선택
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 27:  # 27번째 컬럼 (0부터 시작하면 26)
                target_col = numeric_cols[26]
    
    print(f"✅ 타겟 컬럼: {target_col}")
    
    # 특징 컬럼 선택 (숫자형만)
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"✅ 특징 컬럼: {len(feature_cols)}개")
    
    # 물리 법칙 컬럼 (유입/유출)
    inflow_cols = [col for col in feature_cols if 'TO_HUB' in col][:4]
    outflow_cols = [col for col in feature_cols if 'M16A_3F_TO' in col and 'HUB' not in col][:4]
    
    if not inflow_cols:
        inflow_cols = feature_cols[1:5]  # 임시
    if not outflow_cols:
        outflow_cols = feature_cols[30:34]  # 임시
    
    # ========================================
    # 시퀀스 생성 (과거 20분 → 10분 후)
    # ========================================
    
    print("\n📈 시퀀스 생성 중...")
    
    X, y, X_physics = [], [], []
    timestamps = []
    
    seq_len = 20  # 과거 20분
    pred_len = 10  # 10분 후 예측
    
    # 시퀀스 생성
    for i in range(len(df) - seq_len - pred_len + 1):
        # 입력: i ~ i+seq_len (20개 시점)
        X.append(df[feature_cols].iloc[i:i+seq_len].values)
        
        # 타겟: i+seq_len+pred_len-1 (10분 후)
        y.append(df[target_col].iloc[i+seq_len+pred_len-1])
        
        # 타임스탬프
        timestamps.append(df['timestamp'].iloc[i+seq_len+pred_len-1])
        
        # 물리 특징
        inflow = df[inflow_cols].iloc[i+seq_len-1].sum() if inflow_cols else 0
        outflow = df[outflow_cols].iloc[i+seq_len-1].sum() if outflow_cols else 0
        net_flow = inflow - outflow
        X_physics.append([inflow, outflow, net_flow])
    
    X = np.array(X)
    y = np.array(y)
    X_physics = np.array(X_physics)
    
    print(f"✅ 시퀀스 생성 완료:")
    print(f"  - 샘플 수: {len(X)}개")
    print(f"  - 입력 shape: {X.shape}")
    print(f"  - 타겟 shape: {y.shape}")
    
    # 타겟 분석
    print(f"\n🎯 타겟 분석:")
    print(f"  - 범위: {y.min():.0f} ~ {y.max():.0f}")
    print(f"  - 평균: {y.mean():.1f}")
    print(f"  - 극단값(310+): {(y >= 310).sum()}개 ({(y >= 310).sum()/len(y)*100:.2f}%)")
    print(f"  - 초극단값(335+): {(y >= 335).sum()}개 ({(y >= 335).sum()/len(y)*100:.2f}%)")
    
    # ========================================
    # 스케일링
    # ========================================
    
    print("\n🔧 데이터 스케일링...")
    
    # 스케일러 로드
    try:
        scaler_X = joblib.load('./scalers/scaler_X.pkl')
        scaler_y = joblib.load('./scalers/scaler_y.pkl')
        scaler_physics = joblib.load('./scalers/scaler_physics.pkl')
        print("✅ 기존 스케일러 로드")
    except:
        print("💡 새 스케일러 생성")
        scaler_X = RobustScaler()
        scaler_y = MinMaxScaler(feature_range=(0, 1))
        scaler_physics = StandardScaler()
        
        # Fit
        X_flat = X.reshape(-1, X.shape[2])
        scaler_X.fit(X_flat)
        scaler_y.fit(y.reshape(-1, 1))
        scaler_physics.fit(X_physics)
    
    # Transform
    X_scaled = scaler_X.transform(X.reshape(-1, X.shape[2])).reshape(X.shape)
    X_physics_scaled = scaler_physics.transform(X_physics)
    
    # ========================================
    # 모델 예측
    # ========================================
    
    n_features = X.shape[2]
    config = {
        'seq_len': 20,
        'n_features': n_features,
        'patch_len': 5
    }
    
    # Model 1
    print("\n🤖 Model 1 (PatchTST) 예측 중...")
    model1 = PatchTSTModel(config)
    model1.compile(optimizer='adam', loss='mse')
    dummy_input = np.zeros((1, 20, n_features))
    _ = model1(dummy_input)
    
    try:
        model1.load_weights('./checkpoints/model1_v3.h5')
        print("✅ Model1 가중치 로드")
    except:
        print("⚠️ Model1 가중치 없음")
    
    y_pred1_scaled = model1.predict(X_scaled, batch_size=32, verbose=0)
    y_pred1 = scaler_y.inverse_transform(y_pred1_scaled.reshape(-1, 1)).flatten()
    
    # Model 2
    print("\n🤖 Model 2 (PINN) 예측 중...")
    model2 = PatchTSTPINN(config)
    model2.compile(optimizer='adam', loss='mse')
    dummy_seq = np.zeros((1, 20, n_features))
    dummy_physics = np.zeros((1, 3))
    _ = model2([dummy_seq, dummy_physics])
    
    try:
        model2.load_weights('./checkpoints/model2_v3.h5')
        print("✅ Model2 가중치 로드")
    except:
        print("⚠️ Model2 가중치 없음")
    
    y_pred2_scaled = model2.predict([X_scaled, X_physics_scaled], batch_size=32, verbose=0)
    y_pred2 = scaler_y.inverse_transform(y_pred2_scaled.reshape(-1, 1)).flatten()
    
    # Ensemble
    y_ensemble = 0.4 * y_pred1 + 0.6 * y_pred2
    
    print("✅ 예측 완료!")
    
    # ========================================
    # 결과 정리
    # ========================================
    
    print("\n📊 결과 생성 중...")
    
    results = []
    for i in range(len(y)):
        target_time = timestamps[i]
        
        # 시간 정보
        if pd.notna(target_time):
            timestamp_str = target_time.strftime('%Y%m%d%H%M')
            readable_time = target_time.strftime('%Y-%m-%d %H:%M')
            input_start = (target_time - timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M')
            input_end = (target_time - timedelta(minutes=10)).strftime('%Y-%m-%d %H:%M')
        else:
            timestamp_str = f"index_{i}"
            readable_time = f"Index {i}"
            input_start = "N/A"
            input_end = "N/A"
        
        result = {
            'timestamp': timestamp_str,
            'readable_time': readable_time,
            'actual': float(y[i]),
            'predicted_model1': float(y_pred1[i]),
            'predicted_model2': float(y_pred2[i]),
            'predicted_ensemble': float(y_ensemble[i]),
            'Target_time': readable_time,
            'input_start': input_start,
            'input_end': input_end,
            'error_model1': abs(float(y[i]) - float(y_pred1[i])),
            'error_model2': abs(float(y[i]) - float(y_pred2[i])),
            'error_ensemble': abs(float(y[i]) - float(y_ensemble[i]))
        }
        
        # 최적 모델
        min_error = min(result['error_model1'], result['error_model2'], result['error_ensemble'])
        if result['error_model1'] == min_error:
            result['best_model'] = 'Model1'
        elif result['error_model2'] == min_error:
            result['best_model'] = 'Model2'
        else:
            result['best_model'] = 'Ensemble'
        
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    # 저장
    os.makedirs('./evaluation_results', exist_ok=True)
    results_df.to_csv('./evaluation_results/csv_predictions_all.csv', index=False)
    print(f"✅ 전체 결과 저장: ./evaluation_results/csv_predictions_all.csv ({len(results_df)}개)")
    
    # 극단값만
    extreme_df = results_df[results_df['actual'] >= 310]
    if len(extreme_df) > 0:
        extreme_df.to_csv('./evaluation_results/csv_predictions_extreme.csv', index=False)
        print(f"✅ 극단값 저장: ./evaluation_results/csv_predictions_extreme.csv ({len(extreme_df)}개)")
    
    # ========================================
    # 성능 출력
    # ========================================
    
    print("\n" + "="*80)
    print("📈 전체 성능 평가")
    print("="*80)
    
    print(f"\n📊 전체 데이터: {len(results_df)}개")
    
    mae1 = results_df['error_model1'].mean()
    mae2 = results_df['error_model2'].mean()
    mae_ens = results_df['error_ensemble'].mean()
    
    print(f"\nMAE (Mean Absolute Error):")
    print(f"  Model 1: {mae1:.2f}")
    print(f"  Model 2: {mae2:.2f}")
    print(f"  Ensemble: {mae_ens:.2f}")
    
    # 극단값 성능
    if len(extreme_df) > 0:
        print(f"\n🔴 극단값 성능 ({len(extreme_df)}개):")
        print(f"  Model 1: {extreme_df['error_model1'].mean():.2f}")
        print(f"  Model 2: {extreme_df['error_model2'].mean():.2f}")
        print(f"  Ensemble: {extreme_df['error_ensemble'].mean():.2f}")
    
    # ========================================
    # 샘플 출력
    # ========================================
    
    print("\n" + "="*80)
    print("📋 예측 결과 샘플")
    print("="*80)
    
    # 단일 결과 예시
    print("\n[대표 예시 1개]")
    if len(extreme_df) > 0:
        sample = extreme_df.iloc[0]
    else:
        sample = results_df.iloc[0]
    
    print(f"""
Timestamp: {sample['timestamp']}
Readable_time: {sample['readable_time']}
Input_start: {sample['input_start']} (20분 전)
Input_end: {sample['input_end']} (10분 전)
Target_time: {sample['Target_time']} (예측 시점)

CURRENT_M16A_3F_JOB_2:
- Actual: {sample['actual']:.0f}
- Predicted_Model1: {sample['predicted_model1']:.0f}
- Predicted_Model2: {sample['predicted_model2']:.0f}
- Predicted_Ensemble: {sample['predicted_ensemble']:.0f}

Error:
- Model1: {sample['error_model1']:.1f}
- Model2: {sample['error_model2']:.1f}
- Ensemble: {sample['error_ensemble']:.1f}

Best: {sample['best_model']}
""")
    
    # 추가 샘플들
    print("\n[처음 10개]")
    for idx, row in results_df.head(10).iterrows():
        mark = "🔴" if row['actual'] >= 310 else "⚪"
        print(f"{mark} {row['readable_time']}: 실제={row['actual']:.0f}, M1={row['predicted_model1']:.0f}, M2={row['predicted_model2']:.0f}, Ens={row['predicted_ensemble']:.0f}")
    
    print(f"\n✅ 평가 완료! 총 {len(results_df)}개 데이터 처리")
    return results_df

# ========================================
# 메인 실행
# ========================================

if __name__ == "__main__":
    try:
        # CSV 파일 경로
        csv_path = input("CSV 파일 경로 (기본: data/202509월.csv): ").strip()
        if not csv_path:
            csv_path = 'data/202509월.csv'
        
        # 평가 실행
        results = evaluate_csv_data(csv_path)
        
        print("\n🎉 모든 평가 완료!")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()