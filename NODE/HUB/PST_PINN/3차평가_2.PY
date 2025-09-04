# -*- coding: utf-8 -*-
"""
HUBROOM 모델 평가
- 이미 저장된 테스트 데이터 사용 (시퀀스 생성 X)
- CSV는 실제값 비교용으로만 사용
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🎯 HUBROOM 모델 평가")
print("📅 저장된 테스트 데이터 사용")
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

def evaluate_model(csv_path=None):
    """저장된 테스트 데이터로 평가 + CSV에서 타임스탬프만 참조"""
    
    print("\n📂 저장된 테스트 데이터 로드 중...")
    
    # 이미 준비된 테스트 데이터 로드
    X_test_scaled = np.load('./test_data/X_test_scaled.npy')
    y_test_scaled = np.load('./test_data/y_test_scaled.npy')
    y_test = np.load('./test_data/y_test.npy')
    X_physics_test_scaled = np.load('./test_data/X_physics_test_scaled.npy')
    
    print(f"✅ 테스트 데이터 로드 완료")
    print(f"  샘플 수: {len(y_test)}개")
    print(f"  입력 shape: {X_test_scaled.shape}")
    
    # 스케일러 로드
    scaler_y = joblib.load('./scalers/scaler_y.pkl')
    
    # CSV에서 타임스탬프 가져오기 (있으면)
    timestamps = None
    if csv_path:
        try:
            print(f"\n📂 CSV 파일에서 타임스탬프 로드: {csv_path}")
            df = pd.read_csv(csv_path)
            df['timestamp'] = pd.to_datetime(df.iloc[:, 0], format='%Y%m%d%H%M', errors='coerce')
            # 테스트 데이터 크기만큼 타임스탬프 추출
            timestamps = df['timestamp'].values[-len(y_test):]
            print(f"✅ 타임스탬프 로드 완료")
        except:
            print("⚠️ 타임스탬프 로드 실패 - 기본값 사용")
    
    # 타임스탬프가 없으면 생성
    if timestamps is None:
        base_date = datetime(2025, 5, 9, 6, 0)
        timestamps = [base_date + timedelta(minutes=i) for i in range(len(y_test))]
    
    # 데이터 분석
    print(f"\n📊 테스트 데이터 분석:")
    print(f"  타겟 범위: {y_test.min():.0f} ~ {y_test.max():.0f}")
    print(f"  평균: {y_test.mean():.1f}")
    print(f"  극단값(310+): {(y_test >= 310).sum()}개 ({(y_test >= 310).sum()/len(y_test)*100:.2f}%)")
    print(f"  초극단값(335+): {(y_test >= 335).sum()}개 ({(y_test >= 335).sum()/len(y_test)*100:.2f}%)")
    
    # 모델 설정
    n_features = X_test_scaled.shape[2]
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
    
    y_pred1_scaled = model1.predict(X_test_scaled, batch_size=64, verbose=0)
    y_pred1 = scaler_y.inverse_transform(y_pred1_scaled.reshape(-1, 1)).flatten()
    print("✅ Model 1 예측 완료")
    
    # Model 2 예측
    print("\n🤖 Model 2 (PatchTST + PINN) 예측 중...")
    model2 = PatchTSTPINN(config)
    model2.compile(optimizer='adam', loss='mse')
    dummy_seq = np.zeros((1, 20, n_features))
    dummy_physics = np.zeros((1, 3))
    _ = model2([dummy_seq, dummy_physics])
    model2.load_weights('./checkpoints/model2_v3.h5')
    
    y_pred2_scaled = model2.predict([X_test_scaled, X_physics_test_scaled], batch_size=64, verbose=0)
    y_pred2 = scaler_y.inverse_transform(y_pred2_scaled.reshape(-1, 1)).flatten()
    print("✅ Model 2 예측 완료")
    
    # 앙상블
    y_ensemble = 0.4 * y_pred1 + 0.6 * y_pred2
    print("✅ Ensemble 예측 완료")
    
    # 결과 생성
    print("\n📊 결과 정리 중...")
    
    results = []
    for i in range(len(y_test)):
        # 타임스탬프 처리
        if pd.notna(timestamps[i]) and hasattr(timestamps[i], 'strftime'):
            target_time = timestamps[i]
            timestamp_str = target_time.strftime('%Y%m%d%H%M')
            readable_time = target_time.strftime('%Y-%m-%d %H:%M')
            input_start = (target_time - timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M')
            input_end = (target_time - timedelta(minutes=10)).strftime('%Y-%m-%d %H:%M')
        else:
            target_time = timestamps[i] if isinstance(timestamps[i], datetime) else datetime.now() + timedelta(minutes=i)
            timestamp_str = target_time.strftime('%Y%m%d%H%M')
            readable_time = target_time.strftime('%Y-%m-%d %H:%M')
            input_start = (target_time - timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M')
            input_end = (target_time - timedelta(minutes=10)).strftime('%Y-%m-%d %H:%M')
        
        result = {
            'index': i,
            'timestamp': timestamp_str,
            'readable_time': readable_time,
            'actual': float(y_test[i]),
            'predicted_model1': float(y_pred1[i]),
            'predicted_model2': float(y_pred2[i]),
            'predicted_ensemble': float(y_ensemble[i]),
            'Target_time': readable_time,
            'input_start': input_start,
            'input_end': input_end,
            'error_model1': abs(float(y_test[i]) - float(y_pred1[i])),
            'error_model2': abs(float(y_test[i]) - float(y_pred2[i])),
            'error_ensemble': abs(float(y_test[i]) - float(y_ensemble[i]))
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
    results_df.to_csv('./evaluation_results/predictions_all.csv', index=False)
    print(f"✅ 전체 결과 저장: ./evaluation_results/predictions_all.csv ({len(results_df)}개)")
    
    # 극단값만
    extreme_df = results_df[results_df['actual'] >= 310]
    if len(extreme_df) > 0:
        extreme_df.to_csv('./evaluation_results/predictions_extreme.csv', index=False)
        print(f"✅ 극단값 저장: ./evaluation_results/predictions_extreme.csv ({len(extreme_df)}개)")
    
    # ========================================
    # 성능 평가
    # ========================================
    
    print("\n" + "="*80)
    print("📈 전체 성능 평가")
    print("="*80)
    
    print(f"\n📊 전체: {len(y_test)}개 샘플")
    
    mae1 = mean_absolute_error(y_test, y_pred1)
    mae2 = mean_absolute_error(y_test, y_pred2)
    mae_ens = mean_absolute_error(y_test, y_ensemble)
    
    print(f"\nMAE (Mean Absolute Error):")
    print(f"  Model 1: {mae1:.2f}")
    print(f"  Model 2: {mae2:.2f}")
    print(f"  Ensemble: {mae_ens:.2f}")
    
    # 극단값 성능
    extreme_mask = y_test >= 310
    if extreme_mask.sum() > 0:
        print(f"\n🔴 극단값 성능 ({extreme_mask.sum()}개):")
        print(f"  Model 1: {mean_absolute_error(y_test[extreme_mask], y_pred1[extreme_mask]):.2f}")
        print(f"  Model 2: {mean_absolute_error(y_test[extreme_mask], y_pred2[extreme_mask]):.2f}")
        print(f"  Ensemble: {mean_absolute_error(y_test[extreme_mask], y_ensemble[extreme_mask]):.2f}")
        
        # 감지율
        detected_m1 = (y_pred1[extreme_mask] >= 305).sum()
        detected_m2 = (y_pred2[extreme_mask] >= 305).sum()
        print(f"\n극단값 감지율 (예측 >= 305):")
        print(f"  Model 1: {detected_m1}/{extreme_mask.sum()} ({detected_m1/extreme_mask.sum()*100:.1f}%)")
        print(f"  Model 2: {detected_m2}/{extreme_mask.sum()} ({detected_m2/extreme_mask.sum()*100:.1f}%)")
    
    # ========================================
    # 결과 출력
    # ========================================
    
    print("\n" + "="*80)
    print("📋 예측 결과 (1개 샘플)")
    print("="*80)
    
    # 대표 샘플 1개
    if len(extreme_df) > 0:
        sample = extreme_df.iloc[0]
    else:
        sample = results_df.iloc[0]
    
    print(f"""
Timestamp: {sample['timestamp']}
Readable_time: {sample['readable_time']}
Input_start: {sample['input_start']}
Input_end: {sample['input_end']}
Target_time: {sample['Target_time']}

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
    
    # 추가 샘플
    print("\n[처음 10개]")
    for idx, row in results_df.head(10).iterrows():
        mark = "🔴" if row['actual'] >= 310 else "⚪"
        print(f"{mark} {row['readable_time']}: 실제={row['actual']:.0f}, M1={row['predicted_model1']:.0f}, M2={row['predicted_model2']:.0f}")
    
    print(f"\n✅ 평가 완료! 총 {len(results_df)}개 테스트 데이터")
    return results_df

# ========================================
# 메인 실행
# ========================================

if __name__ == "__main__":
    try:
        # CSV 파일 옵션
        use_csv = input("CSV 파일에서 타임스탬프를 가져올까요? (y/n): ").strip().lower()
        
        csv_path = None
        if use_csv == 'y':
            csv_path = input("CSV 파일 경로 (기본: data/202509월.csv): ").strip()
            if not csv_path:
                csv_path = 'data/202509월.csv'
        
        # 평가 실행
        results = evaluate_model(csv_path)
        
        print("\n🎉 평가 완료!")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()