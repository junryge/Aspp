# -*- coding: utf-8 -*-
"""
HUBROOM 극단값 예측 모델 평가
- 평가 데이터: 2025년 8월 (20250801_TO_20250831.CSV)
- 학습된 Model 1, Model 2 사용
- 결과를 CSV로 저장
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🎯 HUBROOM 모델 평가 시스템")
print("📅 평가 데이터: 2025년 8월")
print("="*80)

# ========================================
# Model 1: PatchTST (코드에서 복사)
# ========================================

class PatchTSTModel(keras.Model):
    def __init__(self, config):
        super().__init__()
        
        self.seq_len = config['seq_len']
        self.n_features = config['n_features']
        self.patch_len = config['patch_len']
        self.n_patches = self.seq_len // self.patch_len
        
        # 패치 임베딩
        self.patch_embedding = layers.Dense(128, activation='relu')
        
        # Transformer
        self.attention = layers.MultiHeadAttention(num_heads=8, key_dim=16)
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        
        self.ffn = keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(128)
        ])
        
        # 출력
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout = layers.Dropout(0.3)
        self.dense2 = layers.Dense(64, activation='relu')
        self.output_layer = layers.Dense(1)
        
    def call(self, x, training=False):
        batch_size = tf.shape(x)[0]
        
        # 패치 생성
        x = tf.reshape(x, [batch_size, self.n_patches, self.patch_len * self.n_features])
        
        # 패치 임베딩
        x = self.patch_embedding(x)
        
        # Transformer
        attn = self.attention(x, x, training=training)
        x = self.norm1(x + attn)
        
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        # 출력
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        output = self.output_layer(x)
        
        return tf.squeeze(output, axis=-1)

# ========================================
# Model 2: PatchTST + PINN (코드에서 복사)
# ========================================

class PatchTSTPINN(keras.Model):
    def __init__(self, config):
        super().__init__()
        
        # PatchTST 부분
        self.seq_len = config['seq_len']
        self.n_features = config['n_features']
        self.patch_len = config['patch_len']
        self.n_patches = self.seq_len // self.patch_len
        
        # 패치 임베딩
        self.patch_embedding = layers.Dense(128, activation='relu')
        
        # Transformer
        self.attention = layers.MultiHeadAttention(num_heads=8, key_dim=16)
        self.norm = layers.LayerNormalization()
        
        # 시계열 처리
        self.flatten = layers.Flatten()
        self.temporal_dense = layers.Dense(64, activation='relu')
        
        # 물리 정보 처리 (PINN)
        self.physics_net = keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(16, activation='relu')
        ])
        
        # 융합 및 극단값 보정
        self.fusion = keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])
        
        # 극단값 부스팅 레이어
        self.extreme_boost = layers.Dense(1, activation='sigmoid')
        
    def call(self, inputs, training=False):
        x_seq, x_physics = inputs
        
        batch_size = tf.shape(x_seq)[0]
        
        # PatchTST 처리
        x = tf.reshape(x_seq, [batch_size, self.n_patches, self.patch_len * self.n_features])
        x = self.patch_embedding(x)
        
        attn = self.attention(x, x, training=training)
        x = self.norm(x + attn)
        
        x = self.flatten(x)
        temporal_features = self.temporal_dense(x)
        
        # 물리 정보 처리
        physics_features = self.physics_net(x_physics)
        
        # 융합
        combined = tf.concat([temporal_features, physics_features], axis=-1)
        output = self.fusion(combined)
        
        # 극단값 부스팅
        boost_factor = self.extreme_boost(combined)
        output = output * (1 + boost_factor * 0.2)
        
        return tf.squeeze(output, axis=-1)

# ========================================
# 평가 함수
# ========================================

def evaluate_august_data():
    """2025년 8월 데이터 평가"""
    
    # 1. 데이터 로드
    print("\n📂 데이터 로드 중...")
    df = pd.read_csv('data/20250801_TO_20250831.CSV')
    print(f"✅ 8월 데이터 로드 완료: {df.shape}")
    
    # 타겟 컬럼
    target_col = 'CURRENT_M16A_3F_JOB_2'
    
    # 데이터 전처리
    df['timestamp'] = pd.to_datetime(df.iloc[:, 0], format='%Y%m%d%H%M', errors='coerce')
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df.fillna(method='ffill').fillna(0)
    
    # 데이터 분석
    print(f"\n📊 8월 데이터 분석:")
    print(f"  기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
    print(f"  전체 샘플: {len(df)}")
    print(f"  타겟 범위: {df[target_col].min():.0f} ~ {df[target_col].max():.0f}")
    print(f"  평균: {df[target_col].mean():.1f}, 중앙값: {df[target_col].median():.1f}")
    print(f"\n  극단값 분포:")
    print(f"    300+: {(df[target_col] >= 300).sum()}개 ({(df[target_col] >= 300).sum()/len(df)*100:.2f}%)")
    print(f"    310+: {(df[target_col] >= 310).sum()}개 ({(df[target_col] >= 310).sum()/len(df)*100:.2f}%)")
    print(f"    335+: {(df[target_col] >= 335).sum()}개 ({(df[target_col] >= 335).sum()/len(df)*100:.2f}%)")
    
    # 2. 스케일러 로드
    print("\n🔧 스케일러 로드 중...")
    scaler_X = joblib.load('./scalers/scaler_X.pkl')
    scaler_y = joblib.load('./scalers/scaler_y.pkl')
    scaler_physics = joblib.load('./scalers/scaler_physics.pkl')
    print("✅ 스케일러 로드 완료")
    
    # 3. 시퀀스 생성
    print("\n🔄 시퀀스 생성 중...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 물리 데이터용 컬럼
    inflow_cols = ['M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2',
                   'M14A_3F_TO_HUB_JOB2', 'M14B_7F_TO_HUB_JOB2']
    outflow_cols = ['M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB',
                    'M16A_3F_TO_M14A_3F_JOB', 'M16A_3F_TO_M14B_7F_JOB']
    
    available_inflow = [col for col in inflow_cols if col in df.columns]
    available_outflow = [col for col in outflow_cols if col in df.columns]
    
    seq_len = 20
    pred_len = 10
    
    X, y, X_physics, timestamps = [], [], [], []
    
    for i in range(len(df) - seq_len - pred_len + 1):
        # 시계열 데이터
        X.append(df[numeric_cols].iloc[i:i+seq_len].values)
        
        # 타겟
        y_val = df[target_col].iloc[i + seq_len + pred_len - 1]
        y.append(y_val)
        
        # 타임스탬프
        timestamps.append(df['timestamp'].iloc[i + seq_len + pred_len - 1])
        
        # 물리 데이터
        physics = [
            df[target_col].iloc[i + seq_len - 1],  # 현재값
            df[available_inflow].iloc[i+seq_len:i+seq_len+pred_len].sum().sum() if available_inflow else 0,
            df[available_outflow].iloc[i+seq_len:i+seq_len+pred_len].sum().sum() if available_outflow else 0
        ]
        X_physics.append(physics)
    
    X = np.array(X)
    y = np.array(y)
    X_physics = np.array(X_physics)
    
    print(f"✅ 시퀀스 생성 완료: {len(X)}개")
    
    # 4. 데이터 스케일링
    print("\n📐 데이터 스케일링 중...")
    n_features = X.shape[2]
    
    X_scaled = scaler_X.transform(X.reshape(-1, n_features)).reshape(len(X), seq_len, n_features)
    y_scaled = scaler_y.transform(y.reshape(-1, 1)).flatten()
    X_physics_scaled = scaler_physics.transform(X_physics)
    
    print("✅ 스케일링 완료")
    
    # 5. 모델 로드 및 예측
    print("\n🤖 모델 로드 및 예측 중...")
    
    config = {
        'seq_len': 20,
        'n_features': n_features,
        'patch_len': 5
    }
    
    # Model 1 예측
    print("  Model 1 (PatchTST) 예측 중...")
    model1 = PatchTSTModel(config)
    model1.compile(optimizer='adam', loss='mse')
    dummy_input = np.zeros((1, 20, n_features))
    _ = model1(dummy_input)
    model1.load_weights('./checkpoints/model1_v3.h5')
    
    y_pred1_scaled = model1.predict(X_scaled, batch_size=64, verbose=0)
    y_pred1 = scaler_y.inverse_transform(y_pred1_scaled.reshape(-1, 1)).flatten()
    
    # Model 2 예측
    print("  Model 2 (PatchTST + PINN) 예측 중...")
    model2 = PatchTSTPINN(config)
    model2.compile(optimizer='adam', loss='mse')
    dummy_seq = np.zeros((1, 20, n_features))
    dummy_physics = np.zeros((1, 3))
    _ = model2([dummy_seq, dummy_physics])
    model2.load_weights('./checkpoints/model2_v3.h5')
    
    y_pred2_scaled = model2.predict([X_scaled, X_physics_scaled], batch_size=64, verbose=0)
    y_pred2 = scaler_y.inverse_transform(y_pred2_scaled.reshape(-1, 1)).flatten()
    
    # 앙상블 예측 (가중 평균)
    y_ensemble = 0.4 * y_pred1 + 0.6 * y_pred2  # Model 2에 더 높은 가중치
    
    print("✅ 예측 완료")
    
    # 6. 평가 메트릭 계산
    print("\n📈 성능 평가 중...")
    
    results = {
        'Model': [],
        'MAE': [],
        'RMSE': [],
        'R2': [],
        'MAE_Low(<200)': [],
        'MAE_Normal(200-300)': [],
        'MAE_High(300+)': [],
        'MAE_Extreme(310+)': [],
        'Detection_310+': [],
        'Detection_335+': []
    }
    
    # 각 모델별 평가
    for model_name, y_pred in [('Model1_PatchTST', y_pred1), 
                               ('Model2_PINN', y_pred2), 
                               ('Ensemble', y_ensemble)]:
        
        results['Model'].append(model_name)
        results['MAE'].append(mean_absolute_error(y, y_pred))
        results['RMSE'].append(np.sqrt(mean_squared_error(y, y_pred)))
        results['R2'].append(r2_score(y, y_pred))
        
        # 구간별 MAE
        mask_low = y < 200
        mask_normal = (y >= 200) & (y < 300)
        mask_high = y >= 300
        mask_extreme = y >= 310
        
        results['MAE_Low(<200)'].append(mean_absolute_error(y[mask_low], y_pred[mask_low]) if mask_low.sum() > 0 else 0)
        results['MAE_Normal(200-300)'].append(mean_absolute_error(y[mask_normal], y_pred[mask_normal]) if mask_normal.sum() > 0 else 0)
        results['MAE_High(300+)'].append(mean_absolute_error(y[mask_high], y_pred[mask_high]) if mask_high.sum() > 0 else 0)
        results['MAE_Extreme(310+)'].append(mean_absolute_error(y[mask_extreme], y_pred[mask_extreme]) if mask_extreme.sum() > 0 else 0)
        
        # 극단값 감지율
        mask_310 = y >= 310
        mask_335 = y >= 335
        
        if mask_310.sum() > 0:
            detection_310 = ((y_pred >= 305)[mask_310].sum() / mask_310.sum()) * 100
            results['Detection_310+'].append(f"{detection_310:.1f}%")
        else:
            results['Detection_310+'].append("N/A")
            
        if mask_335.sum() > 0:
            detection_335 = ((y_pred >= 330)[mask_335].sum() / mask_335.sum()) * 100
            results['Detection_335+'].append(f"{detection_335:.1f}%")
        else:
            results['Detection_335+'].append("N/A")
    
    # 결과 DataFrame 생성
    results_df = pd.DataFrame(results)
    
    # 7. 상세 예측 결과 DataFrame
    detail_df = pd.DataFrame({
        'Timestamp': timestamps,
        'Actual': y,
        'Pred_Model1': y_pred1,
        'Pred_Model2': y_pred2,
        'Pred_Ensemble': y_ensemble,
        'Error_Model1': np.abs(y - y_pred1),
        'Error_Model2': np.abs(y - y_pred2),
        'Error_Ensemble': np.abs(y - y_ensemble),
        'Is_Extreme(310+)': y >= 310,
        'Is_VeryExtreme(335+)': y >= 335
    })
    
    # 8. 결과 저장
    print("\n💾 결과 저장 중...")
    
    # 결과 폴더 생성
    os.makedirs('./evaluation_results', exist_ok=True)
    
    # 성능 요약 저장
    results_df.to_csv('./evaluation_results/model_performance_summary.csv', index=False)
    print("✅ 성능 요약 저장: ./evaluation_results/model_performance_summary.csv")
    
    # 상세 예측 결과 저장
    detail_df.to_csv('./evaluation_results/detailed_predictions_august.csv', index=False)
    print("✅ 상세 예측 저장: ./evaluation_results/detailed_predictions_august.csv")
    
    # 극단값만 필터링한 결과
    extreme_df = detail_df[detail_df['Is_Extreme(310+)']].copy()
    extreme_df.to_csv('./evaluation_results/extreme_predictions_august.csv', index=False)
    print("✅ 극단값 예측 저장: ./evaluation_results/extreme_predictions_august.csv")
    
    # 9. 결과 출력
    print("\n" + "="*80)
    print("📊 평가 결과 요약")
    print("="*80)
    print(results_df.to_string(index=False))
    
    print("\n🎯 극단값 예측 성능 (310+ 구간)")
    print("-"*50)
    extreme_only = detail_df[detail_df['Is_Extreme(310+)']]
    if len(extreme_only) > 0:
        print(f"  총 극단값 샘플: {len(extreme_only)}개")
        print(f"  Model 1 평균 오차: {extreme_only['Error_Model1'].mean():.2f}")
        print(f"  Model 2 평균 오차: {extreme_only['Error_Model2'].mean():.2f}")
        print(f"  Ensemble 평균 오차: {extreme_only['Error_Ensemble'].mean():.2f}")
        
        # 정확 예측 (오차 10 이내)
        accurate_m1 = (extreme_only['Error_Model1'] <= 10).sum()
        accurate_m2 = (extreme_only['Error_Model2'] <= 10).sum()
        accurate_ens = (extreme_only['Error_Ensemble'] <= 10).sum()
        
        print(f"\n  오차 10 이내 예측:")
        print(f"    Model 1: {accurate_m1}/{len(extreme_only)} ({accurate_m1/len(extreme_only)*100:.1f}%)")
        print(f"    Model 2: {accurate_m2}/{len(extreme_only)} ({accurate_m2/len(extreme_only)*100:.1f}%)")
        print(f"    Ensemble: {accurate_ens}/{len(extreme_only)} ({accurate_ens/len(extreme_only)*100:.1f}%)")
    
    print("\n✅ 평가 완료! 결과는 ./evaluation_results/ 폴더에 저장되었습니다.")
    print("="*80)
    
    return results_df, detail_df

# 실행
if __name__ == "__main__":
    try:
        results_df, detail_df = evaluate_august_data()
        print("\n🎉 모든 작업이 성공적으로 완료되었습니다!")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()