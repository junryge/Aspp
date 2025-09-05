# -*- coding: utf-8 -*-
"""
HUBROOM 모델 평가 - 2025년 8월 데이터
- data/20250801_TO_20250831.CSV에서 실제값 로드 (전체)
- 예측 가능한 부분만 예측값 표기
- 예측 불가능한 부분은 실제값만 표기
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🎯 HUBROOM 모델 평가 시스템")
print("📅 2025년 8월 데이터 (20250801_TO_20250831.CSV)")
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
# 메인 평가 함수
# ========================================

def evaluate_august_data():
    """8월 CSV에서 실제값 로드하여 예측값과 비교"""
    
    # 8월 CSV 파일 경로
    csv_path = 'data/20250801_TO_20250831.CSV'
    
    print(f"\n📂 8월 CSV 데이터 로드 중: {csv_path}")
    
    # CSV 로드
    try:
        df_august = pd.read_csv(csv_path)
        print(f"✅ 8월 데이터 로드 완료: {df_august.shape}")
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {csv_path}")
        return None
    
    # 타임스탬프 처리
    df_august['timestamp'] = pd.to_datetime(df_august.iloc[:, 0], format='%Y%m%d%H%M', errors='coerce')
    df_august = df_august.sort_values('timestamp').reset_index(drop=True)
    df_august = df_august.fillna(method='ffill').fillna(0)
    
    # 타겟 컬럼 찾기
    target_col = 'CURRENT_M16A_3F_JOB_2'
    if target_col not in df_august.columns:
        # 타겟 컬럼 찾기 시도
        possible = [col for col in df_august.columns if 'CURRENT' in col and 'M16A_3F' in col]
        if possible:
            target_col = possible[0]
        else:
            # 보통 27번째 컬럼
            numeric_cols = df_august.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 26:
                target_col = numeric_cols[26]
    
    print(f"✅ 타겟 컬럼: {target_col}")
    
    # 전체 실제값 분석
    actual_values_all = df_august[target_col].values
    print(f"\n🎯 8월 전체 실제값 분석:")
    print(f"  - 전체 데이터 수: {len(actual_values_all)}개")
    print(f"  - 범위: {actual_values_all.min():.0f} ~ {actual_values_all.max():.0f}")
    print(f"  - 평균: {actual_values_all.mean():.1f}")
    print(f"  - 극단값(310+): {(actual_values_all >= 310).sum()}개")
    print(f"  - 초극단값(335+): {(actual_values_all >= 335).sum()}개")
    
    # 특징 컬럼 선택
    feature_cols = df_august.select_dtypes(include=[np.number]).columns.tolist()
    
    # 물리 법칙 컬럼 (유입/유출)
    inflow_cols = [col for col in feature_cols if 'TO_HUB' in col][:4]
    outflow_cols = [col for col in feature_cols if 'M16A_3F_TO' in col and 'HUB' not in col][:4]
    
    if not inflow_cols:
        inflow_cols = feature_cols[1:5]
    if not outflow_cols:
        outflow_cols = feature_cols[30:34]
    
    # ========================================
    # 예측 가능한 부분 시퀀스 생성
    # ========================================
    
    print("\n📈 예측 가능한 시퀀스 생성 중...")
    
    X_august = []
    X_physics_august = []
    predictable_indices = []  # 예측 가능한 인덱스 저장
    
    seq_len = 20  # 과거 20분
    pred_len = 10  # 10분 후 예측
    
    # 예측 가능한 시점만 추출
    for i in range(len(df_august) - seq_len - pred_len + 1):
        # 입력: i ~ i+seq_len (20개 시점)
        X_august.append(df_august[feature_cols].iloc[i:i+seq_len].values)
        
        # 예측 시점 인덱스 저장
        predictable_indices.append(i + seq_len + pred_len - 1)
        
        # 물리 특징
        inflow = df_august[inflow_cols].iloc[i+seq_len-1].sum()
        outflow = df_august[outflow_cols].iloc[i+seq_len-1].sum()
        net_flow = inflow - outflow
        X_physics_august.append([inflow, outflow, net_flow])
    
    X_august = np.array(X_august)
    X_physics_august = np.array(X_physics_august)
    
    print(f"✅ 시퀀스 생성 완료:")
    print(f"  - 예측 가능한 샘플 수: {len(X_august)}개")
    print(f"  - 예측 불가능한 샘플 수: {len(df_august) - len(X_august)}개")
    
    # ========================================
    # 저장된 스케일러 로드
    # ========================================
    
    print("\n🔧 저장된 스케일러 로드...")
    
    scaler_X = joblib.load('./scalers/scaler_X.pkl')
    scaler_y = joblib.load('./scalers/scaler_y.pkl')
    scaler_physics = joblib.load('./scalers/scaler_physics.pkl')
    print("✅ 스케일러 로드 완료")
    
    # Transform만 수행 (Fit 하지 않음!)
    X_august_scaled = scaler_X.transform(X_august.reshape(-1, X_august.shape[2])).reshape(X_august.shape)
    X_physics_august_scaled = scaler_physics.transform(X_physics_august)
    
    # ========================================
    # 모델 예측
    # ========================================
    
    n_features = X_august.shape[2]
    config = {
        'seq_len': 20,
        'n_features': n_features,
        'patch_len': 5
    }
    
    # Model 1 예측
    print("\n🤖 Model 1 (PatchTST) 예측 중...")
    model1 = PatchTSTModel(config)
    model1.compile(optimizer='adam', loss='mse')
    dummy_input = np.zeros((1, 20, n_features))
    _ = model1(dummy_input)
    model1.load_weights('./checkpoints/model1_v3.h5')
    
    y_pred1_scaled = model1.predict(X_august_scaled, batch_size=32, verbose=0)
    y_pred1 = scaler_y.inverse_transform(y_pred1_scaled.reshape(-1, 1)).flatten()
    print("✅ Model 1 예측 완료")
    
    # Model 2 예측
    print("\n🤖 Model 2 (PINN) 예측 중...")
    model2 = PatchTSTPINN(config)
    model2.compile(optimizer='adam', loss='mse')
    dummy_seq = np.zeros((1, 20, n_features))
    dummy_physics = np.zeros((1, 3))
    _ = model2([dummy_seq, dummy_physics])
    model2.load_weights('./checkpoints/model2_v3.h5')
    
    y_pred2_scaled = model2.predict([X_august_scaled, X_physics_august_scaled], batch_size=32, verbose=0)
    y_pred2 = scaler_y.inverse_transform(y_pred2_scaled.reshape(-1, 1)).flatten()
    print("✅ Model 2 예측 완료")
    
    # 앙상블
    y_ensemble = 0.4 * y_pred1 + 0.6 * y_pred2
    print("✅ Ensemble 예측 완료")
    
    # ========================================
    # 전체 결과 생성 (실제값은 모두, 예측값은 가능한 부분만)
    # ========================================
    
    print("\n📊 전체 결과 정리 중...")
    
    results = []
    pred_idx = 0  # 예측값 인덱스
    
    for i in range(len(df_august)):
        # 기본 정보
        timestamp = df_august['timestamp'].iloc[i]
        
        if pd.notna(timestamp):
            timestamp_str = timestamp.strftime('%Y%m%d%H%M')
            readable_time = timestamp.strftime('%Y-%m-%d %H:%M')
        else:
            timestamp_str = f"index_{i}"
            readable_time = f"Index {i}"
        
        # 실제값은 항상 있음
        actual = float(df_august[target_col].iloc[i])
        
        # 예측값은 가능한 경우만
        if i in predictable_indices:
            # 예측 가능한 경우
            target_time = df_august['timestamp'].iloc[i]
            input_start = (target_time - timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M') if pd.notna(target_time) else "N/A"
            input_end = (target_time - timedelta(minutes=10)).strftime('%Y-%m-%d %H:%M') if pd.notna(target_time) else "N/A"
            
            result = {
                'index': i,
                'timestamp': timestamp_str,
                'readable_time': readable_time,
                'actual': actual,
                'predicted_model1': float(y_pred1[pred_idx]),
                'predicted_model2': float(y_pred2[pred_idx]),
                'predicted_ensemble': float(y_ensemble[pred_idx]),
                'input_start': input_start,
                'input_end': input_end,
                'error_model1': abs(actual - float(y_pred1[pred_idx])),
                'error_model2': abs(actual - float(y_pred2[pred_idx])),
                'error_ensemble': abs(actual - float(y_ensemble[pred_idx])),
                'is_predictable': True
            }
            
            # 최적 모델
            min_error = min(result['error_model1'], result['error_model2'], result['error_ensemble'])
            if result['error_model1'] == min_error:
                result['best_model'] = 'Model1'
            elif result['error_model2'] == min_error:
                result['best_model'] = 'Model2'
            else:
                result['best_model'] = 'Ensemble'
            
            pred_idx += 1
        else:
            # 예측 불가능한 경우 (처음 29개 또는 마지막 부분)
            result = {
                'index': i,
                'timestamp': timestamp_str,
                'readable_time': readable_time,
                'actual': actual,
                'predicted_model1': np.nan,
                'predicted_model2': np.nan,
                'predicted_ensemble': np.nan,
                'input_start': "N/A",
                'input_end': "N/A",
                'error_model1': np.nan,
                'error_model2': np.nan,
                'error_ensemble': np.nan,
                'is_predictable': False,
                'best_model': "N/A"
            }
        
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    # 저장
    os.makedirs('./evaluation_results', exist_ok=True)
    results_df.to_csv('./evaluation_results/august_complete_results.csv', index=False)
    print(f"✅ 전체 결과 저장: ./evaluation_results/august_complete_results.csv ({len(results_df)}개)")
    
    # 예측 가능한 부분만
    predictable_df = results_df[results_df['is_predictable']].copy()
    predictable_df.to_csv('./evaluation_results/august_predictable_only.csv', index=False)
    print(f"✅ 예측 가능 결과 저장: ./evaluation_results/august_predictable_only.csv ({len(predictable_df)}개)")
    
    # 극단값만
    extreme_df = results_df[(results_df['actual'] >= 310) & (results_df['is_predictable'])].copy()
    if len(extreme_df) > 0:
        extreme_df.to_csv('./evaluation_results/august_extreme_predictions.csv', index=False)
        print(f"✅ 극단값 예측 저장: ./evaluation_results/august_extreme_predictions.csv ({len(extreme_df)}개)")
    
    # ========================================
    # 성능 평가 (예측 가능한 부분만)
    # ========================================
    
    print("\n" + "="*80)
    print("📈 성능 평가 (예측 가능한 부분)")
    print("="*80)
    
    # 예측 가능한 실제값
    y_actual_predictable = predictable_df['actual'].values
    y_pred1_eval = predictable_df['predicted_model1'].values
    y_pred2_eval = predictable_df['predicted_model2'].values
    y_ensemble_eval = predictable_df['predicted_ensemble'].values
    
    mae1 = mean_absolute_error(y_actual_predictable, y_pred1_eval)
    mae2 = mean_absolute_error(y_actual_predictable, y_pred2_eval)
    mae_ens = mean_absolute_error(y_actual_predictable, y_ensemble_eval)
    
    rmse1 = np.sqrt(mean_squared_error(y_actual_predictable, y_pred1_eval))
    rmse2 = np.sqrt(mean_squared_error(y_actual_predictable, y_pred2_eval))
    rmse_ens = np.sqrt(mean_squared_error(y_actual_predictable, y_ensemble_eval))
    
    print(f"\n📊 예측 가능 샘플: {len(predictable_df)}개")
    print(f"\n              MAE     RMSE")
    print(f"Model 1:   {mae1:7.2f} {rmse1:7.2f}")
    print(f"Model 2:   {mae2:7.2f} {rmse2:7.2f}")
    print(f"Ensemble:  {mae_ens:7.2f} {rmse_ens:7.2f}")
    
    # 극단값 성능
    if len(extreme_df) > 0:
        print(f"\n🔴 극단값 성능 ({len(extreme_df)}개):")
        print(f"  Model 1 MAE: {extreme_df['error_model1'].mean():.2f}")
        print(f"  Model 2 MAE: {extreme_df['error_model2'].mean():.2f}")
        print(f"  Ensemble MAE: {extreme_df['error_ensemble'].mean():.2f}")
        
        # 감지율
        threshold = 305
        detected_m1 = (extreme_df['predicted_model1'] >= threshold).sum()
        detected_m2 = (extreme_df['predicted_model2'] >= threshold).sum()
        detected_ens = (extreme_df['predicted_ensemble'] >= threshold).sum()
        
        print(f"\n극단값 감지율 (예측 >= {threshold}):")
        print(f"  Model 1: {detected_m1}/{len(extreme_df)} ({detected_m1/len(extreme_df)*100:.1f}%)")
        print(f"  Model 2: {detected_m2}/{len(extreme_df)} ({detected_m2/len(extreme_df)*100:.1f}%)")
        print(f"  Ensemble: {detected_ens}/{len(extreme_df)} ({detected_ens/len(extreme_df)*100:.1f}%)")
    
    # ========================================
    # 결과 출력 샘플
    # ========================================
    
    print("\n" + "="*80)
    print("📋 예측 결과 샘플 (1개)")
    print("="*80)
    
    # 극단값 샘플 우선
    if len(extreme_df) > 0:
        sample = extreme_df.iloc[0]
    else:
        sample = predictable_df.iloc[0]
    
    print(f"""
📅 8월 데이터 예측 결과
━━━━━━━━━━━━━━━━━━━━━━
Timestamp: {sample['timestamp']}
Readable_time: {sample['readable_time']}
Input_start: {sample['input_start']}
Input_end: {sample['input_end']}

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
    
    # 전체 요약
    print("\n" + "="*80)
    print("📊 전체 요약")
    print("="*80)
    print(f"  - 전체 데이터: {len(results_df)}개")
    print(f"  - 예측 가능: {len(predictable_df)}개")
    print(f"  - 예측 불가능: {len(results_df) - len(predictable_df)}개 (처음 29개 + 마지막 부분)")
    print(f"  - 극단값 예측: {len(extreme_df)}개")
    
    print("\n✅ 8월 데이터 평가 완료!")
    return results_df

# ========================================
# 메인 실행
# ========================================

if __name__ == "__main__":
    try:
        print("\n8월 데이터(20250801_TO_20250831.CSV) 평가를 시작합니다...")
        
        # 평가 실행
        results = evaluate_august_data()
        
        if results is not None:
            print("\n🎉 평가 완료!")
            print("\n📁 생성된 파일:")
            print("  1. august_complete_results.csv - 전체 결과 (실제값 전체 + 예측 가능 부분)")
            print("  2. august_predictable_only.csv - 예측 가능한 부분만")
            print("  3. august_extreme_predictions.csv - 극단값 예측 결과")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()