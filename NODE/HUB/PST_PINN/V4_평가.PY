# -*- coding: utf-8 -*-
"""
V4 모델 평가 데이터 생성
- 202509월.csv 데이터 사용 (학습되지 않은 데이터)
- 예측 결과와 실제값 비교
- 시간 정보를 사람이 읽을 수 있는 형식으로 변환
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from datetime import datetime, timedelta
import joblib
import warnings
from tqdm import tqdm
import os

warnings.filterwarnings('ignore')

# V4 모델 클래스들 (동일한 구조 필요)
class PatchTSTModel(keras.Model):
    def __init__(self, config):
        super().__init__()
        self.seq_len = config['seq_len']
        self.n_features = config['n_features']
        self.patch_len = config['patch_len']
        self.n_patches = self.seq_len // self.patch_len
        
        self.patch_embedding = keras.layers.Dense(128, activation='relu')
        self.attention = keras.layers.MultiHeadAttention(num_heads=8, key_dim=16)
        self.norm1 = keras.layers.LayerNormalization()
        self.norm2 = keras.layers.LayerNormalization()
        
        self.ffn = keras.Sequential([
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(128)
        ])
        
        self.flatten = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(128, activation='relu')
        self.dropout = keras.layers.Dropout(0.3)
        self.dense2 = keras.layers.Dense(64, activation='relu')
        self.output_layer = keras.layers.Dense(1)
        
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

class PatchTSTPINN(keras.Model):
    def __init__(self, config):
        super().__init__()
        self.seq_len = config['seq_len']
        self.n_features = config['n_features']
        self.patch_len = config['patch_len']
        self.n_patches = self.seq_len // self.patch_len
        
        self.patch_embedding = keras.layers.Dense(128, activation='relu')
        self.attention = keras.layers.MultiHeadAttention(num_heads=8, key_dim=16)
        self.norm = keras.layers.LayerNormalization()
        self.flatten = keras.layers.Flatten()
        self.temporal_dense = keras.layers.Dense(64, activation='relu')
        
        self.physics_net = keras.Sequential([
            keras.layers.Dense(64, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation='relu')
        ])
        
        self.fusion = keras.Sequential([
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1)
        ])
        
        self.extreme_boost = keras.layers.Dense(1, activation='sigmoid')
        
    def call(self, inputs, training=False):
        if isinstance(inputs, list):
            x_seq, x_physics = inputs
        else:
            x_seq = inputs[0]
            x_physics = inputs[1]
        
        batch_size = tf.shape(x_seq)[0]
        x = tf.reshape(x_seq, [batch_size, self.n_patches, self.patch_len * self.n_features])
        x = self.patch_embedding(x)
        attn = self.attention(x, x, training=training)
        x = self.norm(x + attn)
        x = self.flatten(x)
        x_temporal = self.temporal_dense(x)
        x_physics = self.physics_net(x_physics)
        x_combined = tf.concat([x_temporal, x_physics], axis=-1)
        output = self.fusion(x_combined)
        boost_factor = self.extreme_boost(x_physics)
        output = output * (1 + boost_factor * 0.2)
        return tf.squeeze(output, axis=-1)

def create_evaluation_data():
    """202509월.csv로 평가 데이터 생성"""
    
    print("="*80)
    print("🔬 V4 모델 평가 데이터 생성")
    print("📅 데이터: 202509월.csv (학습되지 않은 데이터)")
    print("="*80)
    
    # 1. 데이터 로드
    print("\n[1] 데이터 로드 중...")
    df = pd.read_csv('data/202509월.csv')
    print(f"✅ 데이터 로드: {df.shape[0]}행 x {df.shape[1]}열")
    
    # 타임스탬프 처리 (첫 번째 컬럼이 시간이라고 가정)
    df['timestamp'] = pd.to_datetime(df.iloc[:, 0], format='%Y%m%d%H%M', errors='coerce')
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 2. V4 필수 컬럼 확인
    target_col = 'CURRENT_M16A_3F_JOB_2'
    
    # V4 컬럼 리스트
    v4_cols = [
        'CURRENT_M16A_3F_JOB_2',
        'M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2', 'M14A_3F_TO_HUB_JOB2',
        'M14B_7F_TO_HUB_JOB2', 'M16B_10F_TO_HUB_JOB',
        'M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB',
        'M16A_3F_TO_M14A_3F_JOB', 'M16A_3F_TO_M14B_7F_JOB',
        'M16A_3F_TO_3F_MLUD_JOB', 'M16A_3F_CMD',
        'M16A_6F_TO_HUB_CMD', 'M16A_2F_TO_HUB_CMD',
        'M14A_3F_TO_HUB_CMD', 'M14B_7F_TO_HUB_CMD',
        'M16A_6F_LFT_MAXCAPA', 'M16A_2F_LFT_MAXCAPA',
        'M16A_3F_STORAGE_UTIL', 'BRIDGE_TIME',
        'M14_TO_M16_OFS_CUR', 'M16_TO_M14_OFS_CUR'
    ]
    
    # 누락된 컬럼 0으로 채우기
    for col in v4_cols:
        if col not in df.columns:
            df[col] = 0
            
    # BRIDGE_TIME 기본값
    if 'BRIDGE_TIME' not in df.columns or df['BRIDGE_TIME'].isna().all():
        df['BRIDGE_TIME'] = 3.5
    
    # 3. 연속 패턴 분석
    print("\n[2] 연속 패턴 분석 중...")
    pattern_probability = {
        0: 0.003, 1: 0.15, 2: 0.25, 3: 0.31, 4: 0.43, 5: 0.43,
        6: 0.35, 7: 0.42, 8: 0.53, 9: 0.49, 10: 0.42,
        11: 0.47, 12: 0.52, 13: 0.60, 14: 0.54, 15: 0.66,
        16: 0.62, 17: 0.71, 18: 0.79, 19: 0.83, 20: 0.987
    }
    
    seq_len = 20
    consecutive_300_counts = []
    consecutive_300_probs = []
    
    for i in range(len(df) - seq_len):
        window = df[target_col].iloc[i:i+seq_len]
        count_300 = (window >= 300).sum()
        consecutive_300_counts.append(count_300)
        prob = pattern_probability.get(count_300, 0.5)
        consecutive_300_probs.append(prob)
    
    df['consecutive_300_count'] = 0
    df['consecutive_300_prob'] = 0.5
    df.loc[seq_len:, 'consecutive_300_count'] = consecutive_300_counts
    df.loc[seq_len:, 'consecutive_300_prob'] = consecutive_300_probs
    
    # 4. 스케일러 로드
    print("\n[3] 스케일러 로드 중...")
    scaler_X = joblib.load('./scalers/scaler_X_v4.pkl')
    scaler_y = joblib.load('./scalers/scaler_y_v4.pkl')
    scaler_physics = joblib.load('./scalers/scaler_physics_v4.pkl')
    print("✅ 스케일러 로드 완료")
    
    # 5. 모델 로드
    print("\n[4] 모델 로드 중...")
    
    # 특징 수 확인
    numeric_cols = v4_cols + ['consecutive_300_count', 'consecutive_300_prob']
    n_features = len(numeric_cols)
    
    config = {
        'seq_len': 20,
        'n_features': n_features,
        'patch_len': 5
    }
    
    # Model 1 로드
    model1 = PatchTSTModel(config)
    model1.compile(optimizer='adam', loss='mse')
    dummy_input = np.zeros((1, 20, n_features))
    _ = model1(dummy_input)
    model1.load_weights('./checkpoints/model1_v4.h5')
    print("✅ Model 1 로드 완료")
    
    # Model 2 로드
    model2 = PatchTSTPINN(config)
    model2.compile(optimizer='adam', loss='mse')
    dummy_seq = np.zeros((1, 20, n_features))
    dummy_physics = np.zeros((1, 9))
    _ = model2([dummy_seq, dummy_physics])
    model2.load_weights('./checkpoints/model2_v4.h5')
    print("✅ Model 2 로드 완료")
    
    # 6. 예측 수행
    print("\n[5] 예측 수행 중...")
    
    results = []
    pred_len = 10
    
    # 유입/유출 컬럼
    inflow_cols = ['M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2', 
                   'M14A_3F_TO_HUB_JOB2', 'M14B_7F_TO_HUB_JOB2', 'M16B_10F_TO_HUB_JOB']
    outflow_cols = ['M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB',
                    'M16A_3F_TO_M14A_3F_JOB', 'M16A_3F_TO_M14B_7F_JOB', 'M16A_3F_TO_3F_MLUD_JOB']
    cmd_cols = ['M16A_3F_CMD', 'M16A_6F_TO_HUB_CMD', 'M16A_2F_TO_HUB_CMD',
                'M14A_3F_TO_HUB_CMD', 'M14B_7F_TO_HUB_CMD']
    
    # 모든 유효한 인덱스 처리 (1분마다)
    valid_indices = range(seq_len, len(df) - pred_len)
    
    for idx in tqdm(valid_indices, desc="예측 생성"):
        i = idx
        
        # 과거 20분 데이터 (i-20 ~ i-1)
        X_seq = df[numeric_cols].iloc[i-seq_len:i].values
        X_seq_scaled = scaler_X.transform(X_seq.reshape(-1, n_features)).reshape(1, seq_len, n_features)
        
        # 물리 데이터
        physics = [
            df[target_col].iloc[i-1],  # 현재값
            df[inflow_cols].iloc[i:i+pred_len].sum().sum(),  # 향후 10분 유입
            df[outflow_cols].iloc[i:i+pred_len].sum().sum(),  # 향후 10분 유출
            df['BRIDGE_TIME'].iloc[i-1],
            df['consecutive_300_count'].iloc[i-1],
            df['consecutive_300_prob'].iloc[i-1],
            df['M16A_3F_STORAGE_UTIL'].iloc[i-1],
            df[cmd_cols].iloc[i-1].sum(),
            df[target_col].iloc[i-5:i].mean()  # 최근 5개 평균
        ]
        
        X_physics_scaled = scaler_physics.transform(np.array(physics).reshape(1, -1))
        
        # 예측
        pred1_scaled = model1.predict(X_seq_scaled, verbose=0)
        pred1 = scaler_y.inverse_transform(pred1_scaled.reshape(-1, 1))[0, 0]
        
        pred2_scaled = model2.predict([X_seq_scaled, X_physics_scaled], verbose=0)
        pred2 = scaler_y.inverse_transform(pred2_scaled.reshape(-1, 1))[0, 0]
        
        # 과거 20분 실제값 전체
        actual_20min_values = df[target_col].iloc[i-seq_len:i].tolist()
        actual_20min_times = [df['timestamp'].iloc[j].strftime('%Y-%m-%d %H:%M') for j in range(i-seq_len, i)]
        
        # 10분 후 실제값
        actual_target = df[target_col].iloc[i + pred_len - 1]
        target_time = df['timestamp'].iloc[i + pred_len - 1].strftime('%Y-%m-%d %H:%M')
        
        # 현재 시점 정보
        current_time = df['timestamp'].iloc[i].strftime('%Y-%m-%d %H:%M')
        current_value = df[target_col].iloc[i]
        
        # 입력 데이터 통계
        input_max = df[target_col].iloc[i-seq_len:i].max()
        input_min = df[target_col].iloc[i-seq_len:i].min()
        input_avg = df[target_col].iloc[i-seq_len:i].mean()
        input_std = df[target_col].iloc[i-seq_len:i].std()
        
        # 300+ 판정
        threshold_300 = 300
        actual_ng = actual_target >= threshold_300
        pred1_ng = pred1 >= threshold_300
        pred2_ng = pred2 >= threshold_300
        
        # 결과 저장
        result = {
            # 시간 정보
            'prediction_time': current_time,
            'target_time': target_time,
            
            # 실제값들
            'current_actual': round(current_value, 2),
            'target_actual': round(actual_target, 2),
            
            # 과거 20분 실제값 (각 분별로)
            **{f'actual_t-{20-j}': round(actual_20min_values[j], 2) for j in range(20)},
            
            # 예측값
            'model1_predicted': round(pred1, 2),
            'model2_predicted': round(pred2, 2),
            
            # 오차
            'model1_error': round(actual_target - pred1, 2),
            'model2_error': round(actual_target - pred2, 2),
            'model1_error_pct': round((actual_target - pred1) / actual_target * 100, 2) if actual_target != 0 else 0,
            'model2_error_pct': round((actual_target - pred2) / actual_target * 100, 2) if actual_target != 0 else 0,
            
            # 판정
            'model1_OK_NG': 'OK' if (actual_ng == pred1_ng) else 'NG',
            'model2_OK_NG': 'OK' if (actual_ng == pred2_ng) else 'NG',
            
            # 통계
            'input_max': round(input_max, 2),
            'input_min': round(input_min, 2),
            'input_avg': round(input_avg, 2),
            'input_std': round(input_std, 2),
            'consecutive_300_count': int(df['consecutive_300_count'].iloc[i-1]),
            'consecutive_300_prob': round(df['consecutive_300_prob'].iloc[i-1], 3)
        }
        
        results.append(result)
    
    # 7. 결과 저장
    print("\n[6] 결과 저장 중...")
    
    results_df = pd.DataFrame(results)
    
    # 전체 결과를 하나의 CSV로 저장
    results_df.to_csv('evaluation_results_202509_full.csv', index=False, encoding='utf-8-sig')
    
    print(f"✅ 평가 데이터 저장 완료: evaluation_results_202509_full.csv ({len(results_df)}행)")
    
    # 8. 성능 요약
    print("\n[7] 성능 요약:")
    print("="*50)
    
    # Model 1 성능
    mae1 = np.mean(np.abs(results_df['model1_error']))
    rmse1 = np.sqrt(np.mean(results_df['model1_error']**2))
    accuracy1 = (results_df['model1_OK_NG'] == 'OK').mean() * 100
    
    print(f"\nModel 1 (PatchTST):")
    print(f"  MAE: {mae1:.2f}")
    print(f"  RMSE: {rmse1:.2f}")
    print(f"  300+ 판정 정확도: {accuracy1:.1f}%")
    
    # Model 2 성능
    mae2 = np.mean(np.abs(results_df['model2_error']))
    rmse2 = np.sqrt(np.mean(results_df['model2_error']**2))
    accuracy2 = (results_df['model2_OK_NG'] == 'OK').mean() * 100
    
    print(f"\nModel 2 (PINN):")
    print(f"  MAE: {mae2:.2f}")
    print(f"  RMSE: {rmse2:.2f}")
    print(f"  300+ 판정 정확도: {accuracy2:.1f}%")
    
    # 극단값 감지율
    extreme_actual = results_df[results_df['target_actual'] >= 335]
    if len(extreme_actual) > 0:
        print(f"\n335+ 극단값 감지:")
        detected1 = (extreme_actual['model1_predicted'] >= 335).sum()
        detected2 = (extreme_actual['model2_predicted'] >= 335).sum()
        print(f"  Model 1: {detected1}/{len(extreme_actual)} ({detected1/len(extreme_actual)*100:.1f}%)")
        print(f"  Model 2: {detected2}/{len(extreme_actual)} ({detected2/len(extreme_actual)*100:.1f}%)")
    
    # 구간별 성능
    print(f"\n구간별 MAE:")
    low = results_df[results_df['target_actual'] < 200]
    normal = results_df[(results_df['target_actual'] >= 200) & (results_df['target_actual'] < 300)]
    high = results_df[results_df['target_actual'] >= 300]
    
    if len(low) > 0:
        print(f"  <200: Model1={np.mean(np.abs(low['model1_error'])):.2f}, Model2={np.mean(np.abs(low['model2_error'])):.2f}")
    if len(normal) > 0:
        print(f"  200-300: Model1={np.mean(np.abs(normal['model1_error'])):.2f}, Model2={np.mean(np.abs(normal['model2_error'])):.2f}")
    if len(high) > 0:
        print(f"  300+: Model1={np.mean(np.abs(high['model1_error'])):.2f}, Model2={np.mean(np.abs(high['model2_error'])):.2f}")
    
    print("\n" + "="*80)
    print("✅ 평가 완료!")

if __name__ == "__main__":
    create_evaluation_data()