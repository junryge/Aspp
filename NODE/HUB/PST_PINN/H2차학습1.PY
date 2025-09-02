# -*- coding: utf-8 -*-
"""
HUBROOM 극단값 예측 시스템 - 300+ 감지 특화 버전
- 극단값(300+) 예측 정확도 최우선
- 스케일러 저장/로드 완벽 지원
- 데이터 불균형 해결
- PINN 모델 완전 재설계
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import warnings
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle
import os
import json
from tqdm import tqdm
import joblib
from collections import Counter

warnings.filterwarnings('ignore')

# TensorFlow 설정
print(f"TensorFlow Version: {tf.__version__}")
tf.random.set_seed(42)
np.random.seed(42)

# GPU 메모리 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# ===========================
# 🎯 극단값 특화 손실함수
# ===========================

class ExtremeLoss(tf.keras.losses.Loss):
    """300 이상 극단값에 가중치를 부여하는 손실함수"""
    
    def __init__(self, threshold=300, extreme_weight=10.0, name='extreme_loss'):
        super().__init__(name=name)
        self.threshold = threshold
        self.extreme_weight = extreme_weight
    
    def call(self, y_true, y_pred):
        # 기본 MSE
        mse = tf.square(y_true - y_pred)
        
        # 극단값 마스크 (300 이상)
        extreme_mask = tf.cast(y_true >= self.threshold, tf.float32)
        
        # 가중치 적용
        weights = 1.0 + extreme_mask * (self.extreme_weight - 1.0)
        
        # 가중 MSE
        weighted_mse = mse * weights
        
        return tf.reduce_mean(weighted_mse)

class FocalMSELoss(tf.keras.losses.Loss):
    """예측이 어려운 샘플에 집중하는 Focal Loss의 MSE 버전"""
    
    def __init__(self, gamma=2.0, threshold=300, name='focal_mse_loss'):
        super().__init__(name=name)
        self.gamma = gamma
        self.threshold = threshold
    
    def call(self, y_true, y_pred):
        # 절대 오차
        abs_error = tf.abs(y_true - y_pred)
        
        # Focal 가중치 (오차가 클수록 높은 가중치)
        focal_weight = tf.pow(abs_error / 100.0, self.gamma)
        
        # 극단값 추가 가중치
        extreme_mask = tf.cast(y_true >= self.threshold, tf.float32)
        extreme_weight = 1.0 + extreme_mask * 4.0
        
        # 최종 손실
        mse = tf.square(y_true - y_pred)
        weighted_loss = mse * focal_weight * extreme_weight
        
        return tf.reduce_mean(weighted_loss)

# ===========================
# 📊 극단값 특화 데이터 처리
# ===========================

class ExtremeValueProcessor:
    """극단값 처리에 특화된 데이터 프로세서"""
    
    def __init__(self, file_path='data/HUB_0509_TO_0730_DATA.CSV'):
        self.file_path = file_path
        self.target_col = 'CURRENT_M16A_3F_JOB_2'
        
        # 임계값 설정
        self.thresholds = {
            'normal': 250,
            'warning': 280,
            'critical': 300,
            'extreme': 400,
            'disaster': 500
        }
        
        # 스케일러 (극단값에 강건한 RobustScaler 사용)
        self.scaler_X = RobustScaler()
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))  # 0-1로 정규화
        self.scaler_physics = StandardScaler()
        
        # 스케일러 저장 경로
        self.scaler_dir = './scalers'
        os.makedirs(self.scaler_dir, exist_ok=True)
        
        # 물리적 컬럼
        self.inflow_cols = [
            'M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2',
            'M14A_3F_TO_HUB_JOB2', 'M14B_7F_TO_HUB_JOB2'
        ]
        
        self.outflow_cols = [
            'M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB',
            'M16A_3F_TO_M14A_3F_JOB', 'M16A_3F_TO_M14B_7F_JOB'
        ]
    
    def save_scalers(self):
        """스케일러 저장"""
        joblib.dump(self.scaler_X, os.path.join(self.scaler_dir, 'scaler_X.pkl'))
        joblib.dump(self.scaler_y, os.path.join(self.scaler_dir, 'scaler_y.pkl'))
        joblib.dump(self.scaler_physics, os.path.join(self.scaler_dir, 'scaler_physics.pkl'))
        print(f"✅ 스케일러 저장 완료: {self.scaler_dir}")
    
    def load_scalers(self):
        """스케일러 로드"""
        try:
            self.scaler_X = joblib.load(os.path.join(self.scaler_dir, 'scaler_X.pkl'))
            self.scaler_y = joblib.load(os.path.join(self.scaler_dir, 'scaler_y.pkl'))
            self.scaler_physics = joblib.load(os.path.join(self.scaler_dir, 'scaler_physics.pkl'))
            print(f"✅ 스케일러 로드 완료: {self.scaler_dir}")
            return True
        except:
            print("⚠️ 저장된 스케일러 없음. 새로 학습 필요.")
            return False
    
    def analyze_extremes(self, df):
        """극단값 상세 분석"""
        if self.target_col not in df.columns:
            return
        
        target = df[self.target_col]
        
        print("\n" + "="*60)
        print("🔍 극단값 분석 (HUBROOM 반송량)")
        print("="*60)
        
        print(f"\n📊 기본 통계:")
        print(f"  - 전체 범위: {target.min():.0f} ~ {target.max():.0f}")
        print(f"  - 평균: {target.mean():.2f}")
        print(f"  - 중앙값: {target.median():.0f}")
        print(f"  - 표준편차: {target.std():.2f}")
        
        print(f"\n🚨 임계값별 분포:")
        for name, threshold in self.thresholds.items():
            count = (target >= threshold).sum()
            ratio = count / len(target) * 100
            if ratio > 0:
                print(f"  - {name:8} (≥{threshold:3}): {count:5}개 ({ratio:5.2f}%)")
        
        # 극단값 구간별 상세
        extreme_data = target[target >= 300]
        if len(extreme_data) > 0:
            print(f"\n💥 극단값 구간 상세 (≥300):")
            bins = [300, 350, 400, 450, 500, 600, 700, 1000]
            for i in range(len(bins)-1):
                mask = (extreme_data >= bins[i]) & (extreme_data < bins[i+1])
                count = mask.sum()
                if count > 0:
                    print(f"    [{bins[i]:3}-{bins[i+1]:3}): {count:4}개 - "
                          f"평균: {extreme_data[mask].mean():.1f}, "
                          f"최대: {extreme_data[mask].max():.0f}")
    
    def create_balanced_sequences(self, df, numeric_cols, seq_len=20, pred_len=10):
        """극단값 오버샘플링이 적용된 시퀀스 생성"""
        
        X, y, X_physics, sample_weights = [], [], [], []
        
        available_inflow = [col for col in self.inflow_cols if col in df.columns]
        available_outflow = [col for col in self.outflow_cols if col in df.columns]
        
        total_sequences = len(df) - seq_len - pred_len + 1
        
        # 극단값 인덱스 찾기
        extreme_indices = []
        normal_indices = []
        
        print("\n📦 균형잡힌 시퀀스 생성 중...")
        
        for i in tqdm(range(total_sequences), desc="시퀀스 스캔"):
            target_value = df[self.target_col].iloc[i + seq_len + pred_len - 1]
            
            if target_value >= 300:
                extreme_indices.append(i)
            else:
                normal_indices.append(i)
        
        print(f"  - 정상 시퀀스: {len(normal_indices)}개")
        print(f"  - 극단값 시퀀스 (≥300): {len(extreme_indices)}개")
        
        # 극단값 오버샘플링 (3배)
        oversample_rate = 3
        extreme_indices_oversampled = extreme_indices * oversample_rate
        all_indices = normal_indices + extreme_indices_oversampled
        np.random.shuffle(all_indices)
        
        print(f"  - 오버샘플링 후 극단값: {len(extreme_indices_oversampled)}개")
        print(f"  - 전체 학습 시퀀스: {len(all_indices)}개")
        
        # 시퀀스 생성
        for i in tqdm(all_indices, desc="시퀀스 생성"):
            # 입력 시퀀스
            X_seq = df[numeric_cols].iloc[i:i+seq_len].values
            X.append(X_seq)
            
            # 타겟 (10분 후)
            y_val = df[self.target_col].iloc[i + seq_len + pred_len - 1]
            y.append(y_val)
            
            # 가중치 (극단값은 높은 가중치)
            if y_val >= 300:
                weight = 10.0
            elif y_val >= 280:
                weight = 5.0
            else:
                weight = 1.0
            sample_weights.append(weight)
            
            # 물리 데이터
            physics_data = np.array([
                df[self.target_col].iloc[i + seq_len - 1],  # 현재 HUBROOM
                df[available_inflow].iloc[i+seq_len:i+seq_len+pred_len].sum().sum() if available_inflow else 0,
                df[available_outflow].iloc[i+seq_len:i+seq_len+pred_len].sum().sum() if available_outflow else 0
            ])
            X_physics.append(physics_data)
        
        return (np.array(X), np.array(y), np.array(X_physics), np.array(sample_weights))

# ===========================
# 🧠 극단값 특화 PatchTST 모델
# ===========================

class ExtremePatchTST(keras.Model):
    """극단값 예측에 특화된 PatchTST"""
    
    def __init__(self, config):
        super(ExtremePatchTST, self).__init__()
        
        self.seq_len = config['seq_len']
        self.n_features = config['n_features']
        self.patch_len = config['patch_len']
        self.n_patches = self.seq_len // self.patch_len
        
        self.d_model = config['d_model']
        self.n_heads = config['n_heads']
        self.d_ff = config['d_ff']
        self.n_layers = config['n_layers']
        self.dropout_rate = config['dropout']
        
        # 패치 임베딩
        self.patch_embedding = layers.Dense(self.d_model, activation='relu')
        
        # 위치 인코딩
        self.pos_embedding = self.add_weight(
            name='pos_embedding',
            shape=(1, self.n_patches, self.d_model),
            initializer='random_normal',
            trainable=True
        )
        
        # Transformer 인코더
        self.encoder_layers = []
        for _ in range(self.n_layers):
            self.encoder_layers.append(
                layers.MultiHeadAttention(
                    num_heads=self.n_heads,
                    key_dim=self.d_model // self.n_heads,
                    dropout=self.dropout_rate
                )
            )
            self.encoder_layers.append(layers.LayerNormalization())
            self.encoder_layers.append(layers.Dropout(self.dropout_rate))
        
        # 극단값 감지 브랜치
        self.extreme_detector = keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')  # 300 이상 확률
        ])
        
        # 값 예측 브랜치
        self.value_predictor = keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])
        
        self.flatten = layers.Flatten()
    
    def create_patches(self, x):
        batch_size = tf.shape(x)[0]
        x_reshaped = tf.reshape(x, [batch_size, self.n_patches, self.patch_len, self.n_features])
        x_patches = tf.reshape(x_reshaped, [batch_size, self.n_patches, self.patch_len * self.n_features])
        return x_patches
    
    def call(self, x, training=False):
        # 패치 생성
        x_patches = self.create_patches(x)
        
        # 패치 임베딩
        x_embed = self.patch_embedding(x_patches)
        x_embed = x_embed + self.pos_embedding
        
        # Transformer 인코더
        for i in range(0, len(self.encoder_layers), 3):
            attention = self.encoder_layers[i](x_embed, x_embed, training=training)
            norm = self.encoder_layers[i+1](attention + x_embed)
            x_embed = self.encoder_layers[i+2](norm, training=training)
        
        # Flatten
        x_flat = self.flatten(x_embed)
        
        # 극단값 감지와 값 예측
        extreme_prob = self.extreme_detector(x_flat)
        value_pred = self.value_predictor(x_flat)
        
        # 극단값 확률을 이용한 보정
        # 극단값 확률이 높으면 예측값을 상향 조정
        adjusted_pred = value_pred + extreme_prob * 50  # 극단값이면 +50 보정
        
        return tf.squeeze(adjusted_pred, axis=-1)

# ===========================
# 🔬 개선된 PINN 모델
# ===========================

class ImprovedPINN(keras.Model):
    """물리 법칙이 개선된 PINN 모델"""
    
    def __init__(self, config):
        super(ImprovedPINN, self).__init__()
        
        # 시계열 처리 (간단한 LSTM)
        self.lstm = layers.LSTM(64, return_sequences=False)
        
        # 물리 정보 처리
        self.physics_net = keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(16, activation='relu')
        ])
        
        # 통합 네트워크
        self.fusion_net = keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
    
    def call(self, inputs, training=False):
        x_seq, x_physics = inputs
        
        # 시계열 특징
        seq_features = self.lstm(x_seq)
        
        # 물리 특징
        physics_features = self.physics_net(x_physics)
        
        # 특징 결합
        combined = tf.concat([seq_features, physics_features], axis=-1)
        
        # 최종 예측
        output = self.fusion_net(combined)
        
        return tf.squeeze(output, axis=-1)

# ===========================
# 📈 극단값 모니터링 콜백
# ===========================

class ExtremeValueMonitor(Callback):
    """학습 중 극단값 예측 성능 모니터링"""
    
    def __init__(self, X_val, y_val, scaler_y, threshold=300):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.scaler_y = scaler_y
        self.threshold = threshold
    
    def on_epoch_end(self, epoch, logs=None):
        # 예측
        y_pred_scaled = self.model.predict(self.X_val, verbose=0)
        
        # 역정규화
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_true = self.scaler_y.inverse_transform(self.y_val.reshape(-1, 1)).flatten()
        
        # 극단값 정확도
        true_extreme = y_true >= self.threshold
        pred_extreme = y_pred >= self.threshold
        
        if true_extreme.sum() > 0:
            recall = (true_extreme & pred_extreme).sum() / true_extreme.sum()
            print(f"\n  🎯 Epoch {epoch+1} - 극단값(≥{self.threshold}) Recall: {recall:.2%}")
            
            # 극단값 샘플 출력
            extreme_indices = np.where(true_extreme)[0][:3]
            for idx in extreme_indices:
                print(f"     실제: {y_true[idx]:.0f}, 예측: {y_pred[idx]:.0f}")

# ===========================
# 🎯 메인 실행
# ===========================

def main():
    print("="*80)
    print("🏭 HUBROOM 극단값 예측 시스템")
    print("🎯 목표: 300 이상 극단값 정확한 예측")
    print("="*80)
    
    # 데이터 프로세서
    processor = ExtremeValueProcessor()
    
    # 1. 데이터 로드
    print("\n📂 데이터 로드 중...")
    df = pd.read_csv(processor.file_path)
    print(f"✅ 데이터 로드 완료: {df.shape}")
    
    # 2. 극단값 분석
    processor.analyze_extremes(df)
    
    # 3. 전처리
    print("\n🔧 데이터 전처리 중...")
    time_col = df.columns[0]
    df['timestamp'] = pd.to_datetime(df[time_col], format='%Y%m%d%H%M', errors='coerce')
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df.fillna(method='ffill').fillna(0)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 4. 균형잡힌 시퀀스 생성
    X, y, X_physics, sample_weights = processor.create_balanced_sequences(
        df, numeric_cols, seq_len=20, pred_len=10
    )
    
    print(f"\n📊 시퀀스 생성 완료:")
    print(f"  - X shape: {X.shape}")
    print(f"  - y shape: {y.shape}")
    print(f"  - 극단값 가중치 적용: {(sample_weights > 1).sum()}개")
    
    # 5. 데이터 분할
    # 가중치도 함께 분할
    indices = np.arange(len(X))
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    X_physics_train = X_physics[train_idx]
    X_physics_val = X_physics[val_idx]
    X_physics_test = X_physics[test_idx]
    
    weights_train = sample_weights[train_idx]
    
    print(f"\n🔀 데이터 분할:")
    print(f"  - Train: {len(X_train)} ({(y_train >= 300).sum()} 극단값)")
    print(f"  - Valid: {len(X_val)} ({(y_val >= 300).sum()} 극단값)")
    print(f"  - Test: {len(X_test)} ({(y_test >= 300).sum()} 극단값)")
    
    # 6. 스케일링
    print("\n📏 데이터 정규화 중...")
    n_samples_train, seq_len, n_features = X_train.shape
    
    # X 스케일링 (RobustScaler)
    X_train_scaled = processor.scaler_X.fit_transform(
        X_train.reshape(-1, n_features)
    ).reshape(n_samples_train, seq_len, n_features)
    
    X_val_scaled = processor.scaler_X.transform(
        X_val.reshape(-1, n_features)
    ).reshape(X_val.shape[0], seq_len, n_features)
    
    X_test_scaled = processor.scaler_X.transform(
        X_test.reshape(-1, n_features)
    ).reshape(X_test.shape[0], seq_len, n_features)
    
    # y 스케일링 (MinMaxScaler)
    y_train_scaled = processor.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = processor.scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = processor.scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    # 물리 데이터 스케일링
    X_physics_train_scaled = processor.scaler_physics.fit_transform(X_physics_train)
    X_physics_val_scaled = processor.scaler_physics.transform(X_physics_val)
    X_physics_test_scaled = processor.scaler_physics.transform(X_physics_test)
    
    # 스케일러 저장
    processor.save_scalers()
    
    # 7. 모델 구성
    config = {
        'seq_len': 20,
        'n_features': n_features,
        'patch_len': 5,
        'd_model': 128,
        'n_heads': 8,
        'd_ff': 256,
        'n_layers': 4,
        'dropout': 0.2
    }
    
    # 콜백
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
        ExtremeValueMonitor(X_val_scaled, y_val_scaled, processor.scaler_y)
    ]
    
    results = {}
    
    # 모델 1: ExtremePatchTST 학습
    print("\n" + "="*60)
    print("🚀 모델 1: ExtremePatchTST 학습 시작")
    print("="*60)
    model1 = ExtremePatchTST(config)
    
    # 극단값 특화 손실함수 사용
    model1.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=ExtremeLoss(threshold=300, extreme_weight=10),
        metrics=['mae']
    )
    
    history1 = model1.fit(
        X_train_scaled, y_train_scaled,
        validation_data=(X_val_scaled, y_val_scaled),
        sample_weight=weights_train,  # 가중치 적용
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # 평가
    y_pred1_scaled = model1.predict(X_test_scaled)
    y_pred1 = processor.scaler_y.inverse_transform(y_pred1_scaled.reshape(-1, 1)).flatten()
    results['ExtremePatchTST'] = y_pred1
    
    # 모델 2: ImprovedPINN 학습
    print("\n" + "="*60)
    print("🚀 모델 2: ImprovedPINN 학습 시작")
    print("="*60)
    model2 = ImprovedPINN(config)
    
    model2.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=FocalMSELoss(gamma=2.0, threshold=300),
        metrics=['mae']
    )
    
    history2 = model2.fit(
        [X_train_scaled, X_physics_train_scaled], y_train_scaled,
        validation_data=([X_val_scaled, X_physics_val_scaled], y_val_scaled),
        sample_weight=weights_train,
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # 평가
    y_pred2_scaled = model2.predict([X_test_scaled, X_physics_test_scaled])
    y_pred2 = processor.scaler_y.inverse_transform(y_pred2_scaled.reshape(-1, 1)).flatten()
    results['ImprovedPINN'] = y_pred2
    
    # 8. 최종 평가
    print("\n" + "="*60)
    print("📊 최종 성능 평가")
    print("="*60)
    
    for model_name, y_pred in results.items():
        print(f"\n🔍 {model_name}:")
        
        # 전체 성능
        mae = np.mean(np.abs(y_test - y_pred))
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        
        print(f"  전체 MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        
        # 극단값 성능
        extreme_mask = y_test >= 300
        if extreme_mask.sum() > 0:
            extreme_mae = np.mean(np.abs(y_test[extreme_mask] - y_pred[extreme_mask]))
            
            # 극단값 감지 성능
            pred_extreme = y_pred >= 300
            tp = (extreme_mask & pred_extreme).sum()
            fp = (~extreme_mask & pred_extreme).sum()
            fn = (extreme_mask & ~pred_extreme).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"\n  🚨 극단값(≥300) 성능:")
            print(f"    - MAE: {extreme_mae:.2f}")
            print(f"    - 정밀도: {precision:.2%}")
            print(f"    - 재현율: {recall:.2%}")
            print(f"    - F1-Score: {f1:.2%}")
            print(f"    - 감지: {tp}/{extreme_mask.sum()} ({recall:.1%})")
            
            # 극단값 예측 샘플
            print(f"\n  📝 극단값 예측 샘플:")
            extreme_indices = np.where(extreme_mask)[0][:5]
            for idx in extreme_indices:
                status = "✅" if abs(y_test[idx] - y_pred[idx]) < 50 else "❌"
                print(f"    {status} 실제: {y_test[idx]:.0f}, 예측: {y_pred[idx]:.0f}, "
                      f"오차: {abs(y_test[idx] - y_pred[idx]):.0f}")
    
    print("\n✅ 모든 작업 완료!")
    print("💾 스케일러가 ./scalers/ 폴더에 저장되었습니다.")

if __name__ == "__main__":
    main()