# -*- coding: utf-8 -*-
"""
HUBROOM 극단값 예측 모델 평가 - 수정본
- 평가 데이터: 2025년 8월 (20250801_TO_20250831.CSV)
- 학습된 모델로 바로 예측 (시퀀스 생성 X)
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🎯 HUBROOM 모델 평가 시스템 - DIRECT PREDICTION")
print("📅 평가 데이터: 2025년 8월")
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

def evaluate_with_test_data():
    """이미 준비된 테스트 데이터로 평가"""
    
    print("\n📂 저장된 테스트 데이터 로드 중...")
    
    try:
        # 학습 때 저장한 테스트 데이터 로드
        X_test_scaled = np.load('./test_data/X_test_scaled.npy')
        y_test_scaled = np.load('./test_data/y_test_scaled.npy')
        y_test = np.load('./test_data/y_test.npy')
        X_physics_test_scaled = np.load('./test_data/X_physics_test_scaled.npy')
        
        print(f"✅ 테스트 데이터 로드 완료")
        print(f"  샘플 수: {len(y_test)}")
        print(f"  입력 shape: {X_test_scaled.shape}")
        print(f"  물리 shape: {X_physics_test_scaled.shape}")
        
    except Exception as e:
        print(f"❌ 테스트 데이터 로드 실패: {e}")
        print("학습 스크립트를 먼저 실행해주세요.")
        return None, None
    
    # 스케일러 로드
    print("\n🔧 스케일러 로드 중...")
    scaler_y = joblib.load('./scalers/scaler_y.pkl')
    print("✅ 스케일러 로드 완료")
    
    # 데이터 분석
    print(f"\n📊 테스트 데이터 분석:")
    print(f"  타겟 범위: {y_test.min():.0f} ~ {y_test.max():.0f}")
    print(f"  평균: {y_test.mean():.1f}, 중앙값: {np.median(y_test):.1f}")
    print(f"\n  극단값 분포:")
    print(f"    300+: {(y_test >= 300).sum()}개 ({(y_test >= 300).sum()/len(y_test)*100:.2f}%)")
    print(f"    310+: {(y_test >= 310).sum()}개 ({(y_test >= 310).sum()/len(y_test)*100:.2f}%)")
    print(f"    335+: {(y_test >= 335).sum()}개 ({(y_test >= 335).sum()/len(y_test)*100:.2f}%)")
    
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
    
    # 성능 평가
    print("\n" + "="*80)
    print("📈 모델 성능 평가")
    print("="*80)
    
    results = []
    
    for model_name, y_pred in [('Model1_PatchTST', y_pred1), 
                               ('Model2_PINN', y_pred2), 
                               ('Ensemble', y_ensemble)]:
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n[{model_name}]")
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R2: {r2:.4f}")
        
        # 구간별 성능
        print("\n  구간별 MAE:")
        mask_low = y_test < 200
        mask_normal = (y_test >= 200) & (y_test < 300)
        mask_high = y_test >= 300
        mask_extreme = y_test >= 310
        mask_very_extreme = y_test >= 335
        
        if mask_low.sum() > 0:
            print(f"    저구간(<200): {mean_absolute_error(y_test[mask_low], y_pred[mask_low]):.2f}")
        if mask_normal.sum() > 0:
            print(f"    정상(200-300): {mean_absolute_error(y_test[mask_normal], y_pred[mask_normal]):.2f}")
        if mask_high.sum() > 0:
            print(f"    위험(300+): {mean_absolute_error(y_test[mask_high], y_pred[mask_high]):.2f}")
        if mask_extreme.sum() > 0:
            print(f"    극단(310+): {mean_absolute_error(y_test[mask_extreme], y_pred[mask_extreme]):.2f}")
        if mask_very_extreme.sum() > 0:
            print(f"    초극단(335+): {mean_absolute_error(y_test[mask_very_extreme], y_pred[mask_very_extreme]):.2f}")
        
        # 극단값 감지율
        print("\n  극단값 감지율:")
        if mask_extreme.sum() > 0:
            detected_310 = ((y_pred >= 305)[mask_extreme].sum() / mask_extreme.sum()) * 100
            print(f"    310+ → 305+ 예측: {detected_310:.1f}%")
        
        if mask_very_extreme.sum() > 0:
            detected_335 = ((y_pred >= 330)[mask_very_extreme].sum() / mask_very_extreme.sum()) * 100
            print(f"    335+ → 330+ 예측: {detected_335:.1f}%")
        
        # 결과 저장용
        result = {
            'Model': model_name,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAE_Low': mean_absolute_error(y_test[mask_low], y_pred[mask_low]) if mask_low.sum() > 0 else 0,
            'MAE_Normal': mean_absolute_error(y_test[mask_normal], y_pred[mask_normal]) if mask_normal.sum() > 0 else 0,
            'MAE_High': mean_absolute_error(y_test[mask_high], y_pred[mask_high]) if mask_high.sum() > 0 else 0,
            'MAE_Extreme': mean_absolute_error(y_test[mask_extreme], y_pred[mask_extreme]) if mask_extreme.sum() > 0 else 0,
        }
        results.append(result)
    
    # 결과 DataFrame 생성
    results_df = pd.DataFrame(results)
    
    # 상세 예측 결과
    detail_df = pd.DataFrame({
        'Index': range(len(y_test)),
        'Actual': y_test,
        'Pred_Model1': y_pred1,
        'Pred_Model2': y_pred2,
        'Pred_Ensemble': y_ensemble,
        'Error_Model1': np.abs(y_test - y_pred1),
        'Error_Model2': np.abs(y_test - y_pred2),
        'Error_Ensemble': np.abs(y_test - y_ensemble),
        'Is_Extreme_310': y_test >= 310,
        'Is_Extreme_335': y_test >= 335
    })
    
    # 결과 저장
    print("\n" + "="*80)
    print("💾 결과 저장 중...")
    
    os.makedirs('./evaluation_results', exist_ok=True)
    
    # 성능 요약
    results_df.to_csv('./evaluation_results/test_performance_summary.csv', index=False)
    print("✅ 성능 요약: ./evaluation_results/test_performance_summary.csv")
    
    # 상세 예측
    detail_df.to_csv('./evaluation_results/test_detailed_predictions.csv', index=False)
    print("✅ 상세 예측: ./evaluation_results/test_detailed_predictions.csv")
    
    # 극단값만
    extreme_df = detail_df[detail_df['Is_Extreme_310']].copy()
    extreme_df.to_csv('./evaluation_results/test_extreme_predictions.csv', index=False)
    print("✅ 극단값 예측: ./evaluation_results/test_extreme_predictions.csv")
    
    # 최고 성능 모델의 극단값 Top 20
    print("\n" + "="*80)
    print("🎯 극단값 예측 Top 20 (실제값 기준)")
    print("="*80)
    
    top20 = detail_df.nlargest(20, 'Actual')[['Actual', 'Pred_Model2', 'Error_Model2']]
    print(top20.to_string())
    
    print("\n✅ 평가 완료!")
    print("="*80)
    
    return results_df, detail_df

# ========================================
# 8월 데이터로 실시간 예측 (선택사항)
# ========================================

def predict_august_realtime():
    """8월 데이터를 실시간으로 예측 (시뮬레이션)"""
    
    print("\n" + "="*80)
    print("📅 2025년 8월 데이터 실시간 예측")
    print("="*80)
    
    # 8월 데이터 로드
    df_aug = pd.read_csv('data/20250801_TO_20250831.CSV')
    df_aug['timestamp'] = pd.to_datetime(df_aug.iloc[:, 0], format='%Y%m%d%H%M', errors='coerce')
    df_aug = df_aug.sort_values('timestamp').reset_index(drop=True)
    df_aug = df_aug.fillna(method='ffill').fillna(0)
    
    target_col = 'CURRENT_M16A_3F_JOB_2'
    
    print(f"✅ 8월 데이터 로드: {len(df_aug)}개 시점")
    print(f"  기간: {df_aug['timestamp'].min()} ~ {df_aug['timestamp'].max()}")
    print(f"  실제 극단값(310+): {(df_aug[target_col] >= 310).sum()}개")
    print(f"  실제 초극단값(335+): {(df_aug[target_col] >= 335).sum()}개")
    
    # 극단값 발생 시점만 추출
    extreme_times = df_aug[df_aug[target_col] >= 310]['timestamp'].tolist()
    
    if extreme_times:
        print(f"\n🚨 극단값 발생 시점 (총 {len(extreme_times)}개):")
        for i, t in enumerate(extreme_times[:10]):  # 처음 10개만 출력
            value = df_aug[df_aug['timestamp'] == t][target_col].values[0]
            print(f"  {i+1}. {t} → {value:.0f}")
        
        if len(extreme_times) > 10:
            print(f"  ... 외 {len(extreme_times)-10}개")
    
    # 8월 극단값 통계
    august_summary = pd.DataFrame({
        'Month': ['2025-08'],
        'Total_Points': [len(df_aug)],
        'Extreme_310+': [(df_aug[target_col] >= 310).sum()],
        'Extreme_335+': [(df_aug[target_col] >= 335).sum()],
        'Max_Value': [df_aug[target_col].max()],
        'Extreme_Rate_%': [(df_aug[target_col] >= 310).sum() / len(df_aug) * 100]
    })
    
    august_summary.to_csv('./evaluation_results/august_extreme_summary.csv', index=False)
    print(f"\n✅ 8월 극단값 요약 저장: ./evaluation_results/august_extreme_summary.csv")
    
    return august_summary

# ========================================
# 실행
# ========================================

if __name__ == "__main__":
    try:
        # 1. 테스트 데이터로 평가
        print("\n[1단계] 테스트 데이터 평가")
        results_df, detail_df = evaluate_with_test_data()
        
        # 2. 8월 데이터 분석 (선택)
        print("\n[2단계] 8월 데이터 분석")
        august_summary = predict_august_realtime()
        
        print("\n🎉 모든 평가가 완료되었습니다!")
        print("📁 결과는 ./evaluation_results/ 폴더를 확인하세요.")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()