# -*- coding: utf-8 -*-
"""
202509월 CSV 파일 평가 시스템
과거 20분 데이터로 10분 후 예측 → CSV 저장
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("📊 202509월 CSV 평가 시스템")
print("🎯 과거 20분 → 10분 후 예측")
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
# 평가 클래스
# ========================================

class CSVEvaluator:
    def __init__(self):
        self.seq_len = 20  # 과거 20분
        self.pred_len = 10  # 10분 후 예측
        self.target_col = 'CURRENT_M16A_3F_JOB_2'
        
        # 물리 컬럼
        self.inflow_cols = [
            'M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2',
            'M14A_3F_TO_HUB_JOB2', 'M14B_7F_TO_HUB_JOB2'
        ]
        self.outflow_cols = [
            'M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB',
            'M16A_3F_TO_M14A_3F_JOB', 'M16A_3F_TO_M14B_7F_JOB'
        ]
        
    def load_data(self, csv_path):
        """CSV 데이터 로드 및 전처리"""
        print(f"\n📂 CSV 로드: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # 시간 처리
        df['timestamp'] = pd.to_datetime(df.iloc[:, 0], format='%Y%m%d%H%M', errors='coerce')
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = df.fillna(method='ffill').fillna(0)
        
        print(f"  데이터 크기: {df.shape}")
        print(f"  기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
        
        return df
    
    def prepare_sequences(self, df):
        """예측을 위한 시퀀스 준비"""
        print("\n🔄 시퀀스 준비 중...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        n_features = len(numeric_cols)
        
        # 물리 컬럼 확인
        available_inflow = [col for col in self.inflow_cols if col in df.columns]
        available_outflow = [col for col in self.outflow_cols if col in df.columns]
        
        X_list = []
        X_physics_list = []
        y_actual_list = []
        valid_indices = []
        
        # 예측 가능한 범위 계산
        total = len(df) - self.seq_len - self.pred_len + 1
        
        for i in tqdm(range(total), desc="시퀀스 생성"):
            # 과거 20분 데이터
            X = df[numeric_cols].iloc[i:i+self.seq_len].values
            
            # 10분 후 실제값
            y_actual = df[self.target_col].iloc[i + self.seq_len + self.pred_len - 1]
            
            # 물리 데이터 (현재 상태 + 미래 유입/유출)
            current_val = df[self.target_col].iloc[i + self.seq_len - 1]
            inflow = df[available_inflow].iloc[i+self.seq_len:i+self.seq_len+self.pred_len].sum().sum() if available_inflow else 0
            outflow = df[available_outflow].iloc[i+self.seq_len:i+self.seq_len+self.pred_len].sum().sum() if available_outflow else 0
            
            X_list.append(X)
            X_physics_list.append([current_val, inflow, outflow])
            y_actual_list.append(y_actual)
            valid_indices.append(i + self.seq_len + self.pred_len - 1)
        
        print(f"  생성된 시퀀스: {len(X_list)}개")
        
        return (np.array(X_list), np.array(X_physics_list), 
                np.array(y_actual_list), valid_indices, n_features)
    
    def load_models(self, n_features):
        """두 모델 모두 로드"""
        print("\n🤖 모델 로드 중...")
        
        config = {
            'seq_len': 20,
            'n_features': n_features,
            'patch_len': 5
        }
        
        models = {}
        
        # ExtremePatchTST
        try:
            model1 = ExtremePatchTST(config)
            dummy = np.zeros((1, 20, n_features))
            _ = model1(dummy)
            model1.load_weights('./checkpoints/model1_final.h5')
            models['ExtremePatchTST'] = model1
            print("  ✅ ExtremePatchTST 로드 완료")
        except Exception as e:
            print(f"  ❌ ExtremePatchTST 로드 실패: {e}")
        
        # ImprovedPINN
        try:
            model2 = ImprovedPINN(config)
            dummy_seq = np.zeros((1, 20, n_features))
            dummy_physics = np.zeros((1, 3))
            _ = model2([dummy_seq, dummy_physics])
            model2.load_weights('./checkpoints/model2_final.h5')
            models['ImprovedPINN'] = model2
            print("  ✅ ImprovedPINN 로드 완료")
        except Exception as e:
            print(f"  ❌ ImprovedPINN 로드 실패: {e}")
        
        return models
    
    def load_scalers(self):
        """스케일러 로드"""
        print("\n📂 스케일러 로드 중...")
        try:
            scaler_X = joblib.load('./scalers/scaler_X.pkl')
            scaler_y = joblib.load('./scalers/scaler_y.pkl')
            scaler_physics = joblib.load('./scalers/scaler_physics.pkl')
            print("  ✅ 스케일러 로드 완료")
            return scaler_X, scaler_y, scaler_physics
        except Exception as e:
            print(f"  ❌ 스케일러 로드 실패: {e}")
            return None, None, None
    
    def predict_and_evaluate(self, models, X, X_physics, y_actual, n_features):
        """예측 수행 및 평가"""
        # 스케일러 로드
        scaler_X, scaler_y, scaler_physics = self.load_scalers()
        if scaler_X is None:
            return None
        
        # 데이터 스케일링
        print("\n📏 데이터 스케일링 중...")
        X_scaled = scaler_X.transform(X.reshape(-1, n_features)).reshape(X.shape[0], 20, n_features)
        X_physics_scaled = scaler_physics.transform(X_physics)
        
        results = {}
        
        # 각 모델로 예측
        for model_name, model in models.items():
            print(f"\n🔮 {model_name} 예측 중...")
            
            if model_name == 'ImprovedPINN':
                y_pred_scaled = model.predict([X_scaled, X_physics_scaled], batch_size=32, verbose=1)
            else:
                y_pred_scaled = model.predict(X_scaled, batch_size=32, verbose=1)
            
            # 역변환
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            
            # 평가
            mae = mean_absolute_error(y_actual, y_pred)
            rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
            r2 = r2_score(y_actual, y_pred)
            
            # 310+ 분석
            mask_310 = y_actual >= 310
            if mask_310.sum() > 0:
                mae_310 = mean_absolute_error(y_actual[mask_310], y_pred[mask_310])
                detected_310 = (y_pred >= 310)[mask_310].sum()
                rate_310 = detected_310 / mask_310.sum() * 100
            else:
                mae_310 = 0
                detected_310 = 0
                rate_310 = 0
            
            results[model_name] = {
                'predictions': y_pred,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'mae_310': mae_310,
                'detection_rate_310': rate_310
            }
            
            print(f"\n📊 {model_name} 성능:")
            print(f"  MAE: {mae:.2f}")
            print(f"  RMSE: {rmse:.2f}")
            print(f"  R²: {r2:.4f}")
            print(f"  310+ MAE: {mae_310:.2f}")
            print(f"  310+ 감지율: {rate_310:.1f}%")
        
        return results
    
    def save_results(self, df, results, y_actual, valid_indices):
        """결과를 CSV로 저장"""
        print("\n💾 결과 저장 중...")
        
        # 전체 데이터프레임 복사
        df_result = df.copy()
        
        # 예측 컬럼 초기화
        df_result['actual_10min_later'] = np.nan
        df_result['pred_ExtremePatchTST'] = np.nan
        df_result['pred_ImprovedPINN'] = np.nan
        df_result['error_ExtremePatchTST'] = np.nan
        df_result['error_ImprovedPINN'] = np.nan
        
        # 유효한 인덱스에만 값 채우기
        df_result.loc[valid_indices, 'actual_10min_later'] = y_actual
        
        if 'ExtremePatchTST' in results:
            df_result.loc[valid_indices, 'pred_ExtremePatchTST'] = results['ExtremePatchTST']['predictions']
            df_result.loc[valid_indices, 'error_ExtremePatchTST'] = results['ExtremePatchTST']['predictions'] - y_actual
        
        if 'ImprovedPINN' in results:
            df_result.loc[valid_indices, 'pred_ImprovedPINN'] = results['ImprovedPINN']['predictions']
            df_result.loc[valid_indices, 'error_ImprovedPINN'] = results['ImprovedPINN']['predictions'] - y_actual
        
        # 알람 상태 추가
        df_result['alarm_status'] = df_result.apply(
            lambda row: 'CRITICAL' if row['actual_10min_later'] >= 350 
            else 'WARNING' if row['actual_10min_later'] >= 310
            else 'NORMAL' if pd.notna(row['actual_10min_later'])
            else 'NO_DATA', axis=1
        )
        
        # 저장
        output_path = '202509_evaluation_results.csv'
        df_result.to_csv(output_path, index=False)
        
        print(f"✅ 결과 저장 완료: {output_path}")
        print(f"  전체 행: {len(df_result)}")
        print(f"  예측 가능한 행: {len(valid_indices)}")
        print(f"  310+ 데이터: {(y_actual >= 310).sum()}개")
        
        return df_result

# ========================================
# 메인 실행
# ========================================

def main():
    # 평가기 생성
    evaluator = CSVEvaluator()
    
    # 1. 데이터 로드
    csv_path = '202509.csv'  # 또는 입력받기
    if not os.path.exists(csv_path):
        csv_path = input("CSV 파일 경로 입력: ").strip()
    
    df = evaluator.load_data(csv_path)
    
    # 2. 시퀀스 준비
    X, X_physics, y_actual, valid_indices, n_features = evaluator.prepare_sequences(df)
    
    # 3. 모델 로드
    models = evaluator.load_models(n_features)
    
    if not models:
        print("❌ 로드된 모델이 없습니다!")
        return
    
    # 4. 예측 및 평가
    results = evaluator.predict_and_evaluate(models, X, X_physics, y_actual, n_features)
    
    if results:
        # 5. 결과 저장
        df_result = evaluator.save_results(df, results, y_actual, valid_indices)
        
        # 6. 최종 요약
        print("\n" + "="*80)
        print("📊 최종 평가 요약")
        print("="*80)
        
        for model_name, result in results.items():
            print(f"\n[{model_name}]")
            print(f"  MAE: {result['mae']:.2f}")
            print(f"  RMSE: {result['rmse']:.2f}")
            print(f"  R²: {result['r2']:.4f}")
            print(f"  310+ 감지율: {result['detection_rate_310']:.1f}%")
        
        print("\n✅ 평가 완료!")
        print("📁 결과 파일: 202509_evaluation_results.csv")

if __name__ == "__main__":
    main()