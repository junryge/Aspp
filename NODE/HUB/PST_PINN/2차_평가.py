# -*- coding: utf-8 -*-
"""
학습된 데이터 평가 시스템
기존 학습 데이터(HUB_0509_TO_0730_DATA.CSV)로 모델 성능 평가
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("📊 학습 데이터 평가 시스템")
print("🎯 ExtremePatchTST & ImprovedPINN 성능 평가")
print("="*80)

# ========================================
# 모델 정의
# ========================================

class ExtremePatchTST(keras.Model):
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

class ImprovedPINN(keras.Model):
    def __init__(self, config):
        super().__init__()
        
        self.lstm = layers.LSTM(64, return_sequences=False)
        self.physics_net = keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(16, activation='relu')
        ])
        self.fusion = keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        
    def call(self, inputs, training=False):
        x_seq, x_physics = inputs
        seq_features = self.lstm(x_seq)
        physics_features = self.physics_net(x_physics)
        combined = tf.concat([seq_features, physics_features], axis=-1)
        output = self.fusion(combined)
        return tf.squeeze(output, axis=-1)

# ========================================
# 평가 시스템
# ========================================

class ModelEvaluator:
    def __init__(self):
        self.target_col = 'CURRENT_M16A_3F_JOB_2'
        
        # 스케일러 로드
        print("\n📂 스케일러 로드 중...")
        self.scaler_X = joblib.load('./scalers/scaler_X.pkl')
        self.scaler_y = joblib.load('./scalers/scaler_y.pkl')
        self.scaler_physics = joblib.load('./scalers/scaler_physics.pkl')
        print("✅ 스케일러 로드 완료")
    
    def load_test_data(self):
        """저장된 테스트 데이터 로드"""
        print("\n📂 테스트 데이터 로드 중...")
        
        # Step 3에서 저장된 스케일된 데이터 로드
        import pickle
        with open('./checkpoints/training_state.pkl', 'rb') as f:
            state = pickle.load(f)
        
        # 테스트 데이터 추출
        X_test_scaled = state['X_test_scaled']
        y_test_scaled = state['y_test_scaled']
        X_physics_test_scaled = state['X_physics_test_scaled']
        y_test = state['y_test']  # 원본 y값
        
        print(f"  테스트 데이터 크기: {X_test_scaled.shape[0]}개")
        print(f"  310+ 데이터: {(y_test >= 310).sum()}개")
        print(f"  350+ 데이터: {(y_test >= 350).sum()}개")
        
        return X_test_scaled, y_test_scaled, X_physics_test_scaled, y_test
    
    def load_models(self, n_features):
        """학습된 모델 로드"""
        print("\n🤖 모델 로드 중...")
        
        config = {
            'seq_len': 20,
            'n_features': n_features,
            'patch_len': 5
        }
        
        # ExtremePatchTST
        print("  ExtremePatchTST 로드 중...")
        model1 = ExtremePatchTST(config)
        dummy = np.zeros((1, 20, n_features))
        _ = model1(dummy)
        model1.load_weights('./checkpoints/model1_final.h5')
        
        # ImprovedPINN
        print("  ImprovedPINN 로드 중...")
        model2 = ImprovedPINN(config)
        dummy_seq = np.zeros((1, 20, n_features))
        dummy_physics = np.zeros((1, 3))
        _ = model2([dummy_seq, dummy_physics])
        model2.load_weights('./checkpoints/model2_final.h5')
        
        print("✅ 모델 로드 완료")
        return model1, model2
    
    def evaluate_model(self, model, model_name, X_test, y_test_true, X_physics=None):
        """모델 평가"""
        print(f"\n{'='*60}")
        print(f"📊 {model_name} 평가")
        print('='*60)
        
        # 예측
        if model_name == 'ImprovedPINN':
            y_pred_scaled = model.predict([X_test, X_physics], batch_size=32, verbose=0)
        else:
            y_pred_scaled = model.predict(X_test, batch_size=32, verbose=0)
        
        # 역변환
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        # 메트릭 계산
        mae = mean_absolute_error(y_test_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_true, y_pred))
        r2 = r2_score(y_test_true, y_pred)
        
        print(f"\n📈 전체 성능:")
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R²: {r2:.4f}")
        
        # 임계값별 분석
        thresholds = [300, 310, 350]
        for threshold in thresholds:
            mask = y_test_true >= threshold
            if mask.sum() > 0:
                mae_th = mean_absolute_error(y_test_true[mask], y_pred[mask])
                detected = (y_pred >= threshold)[mask].sum()
                total = mask.sum()
                rate = detected / total * 100
                
                print(f"\n🎯 {threshold}+ 분석:")
                print(f"  실제: {total}개")
                print(f"  감지: {detected}개 ({rate:.1f}%)")
                print(f"  MAE: {mae_th:.2f}")
        
        return y_pred, {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'y_pred': y_pred,
            'y_true': y_test_true
        }
    
    def save_predictions(self, y_test, y_pred1, y_pred2):
        """예측 결과 CSV 저장"""
        print("\n💾 예측 결과 저장 중...")
        
        # 데이터프레임 생성
        df_results = pd.DataFrame({
            'actual': y_test,
            'pred_ExtremePatchTST': y_pred1,
            'pred_ImprovedPINN': y_pred2,
            'error_ExtremePatchTST': y_pred1 - y_test,
            'error_ImprovedPINN': y_pred2 - y_test,
            'is_310+': y_test >= 310,
            'is_350+': y_test >= 350
        })
        
        # 알람 상태 추가
        df_results['alarm_status'] = df_results.apply(
            lambda row: 'CRITICAL' if row['actual'] >= 350 
            else 'WARNING' if row['actual'] >= 310 
            else 'NORMAL', axis=1
        )
        
        # 예측 정확도
        df_results['model1_correct_310'] = (
            (df_results['actual'] >= 310) == (df_results['pred_ExtremePatchTST'] >= 310)
        )
        df_results['model2_correct_310'] = (
            (df_results['actual'] >= 310) == (df_results['pred_ImprovedPINN'] >= 310)
        )
        
        # 저장
        output_path = 'test_predictions_result.csv'
        df_results.to_csv(output_path, index=False)
        print(f"✅ 저장 완료: {output_path}")
        
        # 요약 통계
        print(f"\n📊 요약:")
        print(f"  전체 데이터: {len(df_results)}개")
        print(f"  310+ 데이터: {df_results['is_310+'].sum()}개")
        print(f"  350+ 데이터: {df_results['is_350+'].sum()}개")
        print(f"  Model1 310+ 정확도: {df_results['model1_correct_310'].mean():.1%}")
        print(f"  Model2 310+ 정확도: {df_results['model2_correct_310'].mean():.1%}")
        
        return df_results

def main():
    # 평가기 생성
    evaluator = ModelEvaluator()
    
    # 테스트 데이터 로드
    X_test, y_test_scaled, X_physics_test, y_test = evaluator.load_test_data()
    
    # 특성 수 확인
    n_features = X_test.shape[2]
    
    # 모델 로드
    model1, model2 = evaluator.load_models(n_features)
    
    # ExtremePatchTST 평가
    y_pred1, results1 = evaluator.evaluate_model(
        model1, 'ExtremePatchTST', X_test, y_test
    )
    
    # ImprovedPINN 평가
    y_pred2, results2 = evaluator.evaluate_model(
        model2, 'ImprovedPINN', X_test, y_test, X_physics_test
    )
    
    # 결과 저장
    df_results = evaluator.save_predictions(y_test, y_pred1, y_pred2)
    
    # 최종 비교
    print("\n" + "="*80)
    print("📊 최종 모델 비교")
    print("="*80)
    
    print(f"\n{'모델':<20} {'MAE':<10} {'RMSE':<10} {'R²':<10}")
    print("-"*50)
    print(f"{'ExtremePatchTST':<20} {results1['mae']:<10.2f} {results1['rmse']:<10.2f} {results1['r2']:<10.4f}")
    print(f"{'ImprovedPINN':<20} {results2['mae']:<10.2f} {results2['rmse']:<10.2f} {results2['r2']:<10.4f}")
    
    # 우수 모델
    if results2['mae'] < results1['mae']:
        print(f"\n🏆 우수 모델: ImprovedPINN (MAE {results2['mae']:.2f})")
    else:
        print(f"\n🏆 우수 모델: ExtremePatchTST (MAE {results1['mae']:.2f})")
    
    print("\n✅ 평가 완료!")
    print(f"📁 결과 파일: test_predictions_result.csv")

if __name__ == "__main__":
    main()