# -*- coding: utf-8 -*-
"""
HUBROOM 극단값 예측 시스템 V3
- Model 1: PatchTST (전체 구간 균형)
- Model 2: PatchTST + PINN (310+ 극단값 특화)
- 310-335 구간 10배, 335+ 구간 15배 오버샘플링
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import os
import pickle
import warnings
from tqdm import tqdm
import joblib

warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)
print(f"TensorFlow Version: {tf.__version__}")
print("="*80)
print("🏭 HUBROOM 극단값 예측 시스템 V3")
print("🎯 목표: 310+ 극단값 정확 예측")
print("="*80)

# ========================================
# 극단값 모니터링 콜백
# ========================================

class ExtremeValueCallback(Callback):
    """310+ 예측 성능 모니터링"""
    
    def __init__(self, X_val, y_val, scaler_y):
        super().__init__()
        self.X_val = X_val[:500] if len(X_val) > 500 else X_val
        self.y_val = y_val[:500] if len(y_val) > 500 else y_val
        self.scaler_y = scaler_y
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            # 예측
            if isinstance(self.X_val, tuple):
                y_pred_scaled = self.model.predict(self.X_val, verbose=0)
            else:
                y_pred_scaled = self.model.predict(self.X_val, verbose=0)
            
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_true = self.scaler_y.inverse_transform(self.y_val.reshape(-1, 1)).flatten()
            
            # 구간별 성능
            print(f"\n[Epoch {epoch}] 극단값 감지:")
            for threshold in [310, 335]:
                mask = y_true >= threshold
                if mask.sum() > 0:
                    detected = (y_pred >= threshold - 5)[mask].sum()
                    print(f"  {threshold}+: {detected}/{mask.sum()} ({detected/mask.sum()*100:.1f}%)")

# ========================================
# 데이터 처리 (극단값 특화 V3)
# ========================================

class DataProcessorV3:
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
        
    def save_scalers(self):
        os.makedirs('./scalers', exist_ok=True)
        joblib.dump(self.scaler_X, './scalers/scaler_X.pkl')
        joblib.dump(self.scaler_y, './scalers/scaler_y.pkl')
        joblib.dump(self.scaler_physics, './scalers/scaler_physics.pkl')
        print("✅ 스케일러 저장 완료")
        
    def load_scalers(self):
        try:
            self.scaler_X = joblib.load('./scalers/scaler_X.pkl')
            self.scaler_y = joblib.load('./scalers/scaler_y.pkl')
            self.scaler_physics = joblib.load('./scalers/scaler_physics.pkl')
            print("✅ 스케일러 로드 완료")
            return True
        except:
            print("❌ 스케일러 로드 실패")
            return False
    
    def analyze_data(self, df):
        """데이터 분석"""
        target = df[self.target_col]
        print("\n📊 데이터 분석:")
        print(f"  범위: {target.min():.0f} ~ {target.max():.0f}")
        print(f"  평균: {target.mean():.1f}")
        print(f"  중앙값: {target.median():.1f}")
        
        print("\n🎯 3구간 분포:")
        print(f"  저구간(<200): {(target < 200).sum():6}개 ({(target < 200).sum()/len(target)*100:5.2f}%)")
        print(f"  정상(200-300): {((target >= 200) & (target < 300)).sum():6}개 ({((target >= 200) & (target < 300)).sum()/len(target)*100:5.2f}%)")
        print(f"  위험(300+): {(target >= 300).sum():6}개 ({(target >= 300).sum()/len(target)*100:5.2f}%)")
        
        print("\n🚨 극단값 세부:")
        print(f"  310+: {(target >= 310).sum():6}개 ({(target >= 310).sum()/len(target)*100:5.2f}%)")
        print(f"  335+: {(target >= 335).sum():6}개 ({(target >= 335).sum()/len(target)*100:5.2f}%)")
    
    def create_sequences_v3(self, df, seq_len=20, pred_len=10):
        """극단값 집중 시퀀스 생성 V3"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 물리 데이터용 컬럼
        available_inflow = [col for col in self.inflow_cols if col in df.columns]
        available_outflow = [col for col in self.outflow_cols if col in df.columns]
        
        X, y, X_physics, weights = [], [], [], []
        total = len(df) - seq_len - pred_len + 1
        
        # 구간별 인덱스
        indices = {
            'low': [],      # <200
            'normal': [],   # 200-300
            '300': [],      # 300-310
            '310': [],      # 310-335
            '335': []       # 335+
        }
        
        print("\n📦 시퀀스 분류 중...")
        for i in tqdm(range(total)):
            target_val = df[self.target_col].iloc[i + seq_len + pred_len - 1]
            
            if target_val < 200:
                indices['low'].append(i)
            elif target_val < 300:
                indices['normal'].append(i)
            elif target_val < 310:
                indices['300'].append(i)
            elif target_val < 335:
                indices['310'].append(i)
            else:
                indices['335'].append(i)
        
        # V3 오버샘플링 전략
        all_indices = []
        all_indices.extend(indices['low'])         # 1배
        all_indices.extend(indices['normal'])      # 1배
        all_indices.extend(indices['300'] * 3)     # 3배
        all_indices.extend(indices['310'] * 10)    # 10배 (핵심!)
        all_indices.extend(indices['335'] * 15)    # 15배 (최대!)
        
        print(f"\n📊 오버샘플링 결과:")
        print(f"  <200: {len(indices['low'])} → {len(indices['low'])}")
        print(f"  200-300: {len(indices['normal'])} → {len(indices['normal'])}")
        print(f"  300-310: {len(indices['300'])} → {len(indices['300'])*3}")
        print(f"  310-335: {len(indices['310'])} → {len(indices['310'])*10} ⭐")
        print(f"  335+: {len(indices['335'])} → {len(indices['335'])*15} ⭐⭐")
        
        np.random.shuffle(all_indices)
        
        # 시퀀스 생성
        print(f"\n시퀀스 생성 중... (총 {len(all_indices)}개)")
        for i in tqdm(all_indices):
            # 시계열 데이터
            X.append(df[numeric_cols].iloc[i:i+seq_len].values)
            
            # 타겟
            y_val = df[self.target_col].iloc[i + seq_len + pred_len - 1]
            y.append(y_val)
            
            # 물리 데이터 (현재값, 유입합, 유출합)
            physics = [
                df[self.target_col].iloc[i + seq_len - 1],  # 현재값
                df[available_inflow].iloc[i+seq_len:i+seq_len+pred_len].sum().sum() if available_inflow else 0,
                df[available_outflow].iloc[i+seq_len:i+seq_len+pred_len].sum().sum() if available_outflow else 0
            ]
            X_physics.append(physics)
            
            # V3 가중치
            if y_val >= 335:
                weights.append(20.0)  # 335+ 최대 가중치
            elif y_val >= 310:
                weights.append(15.0)  # 310-335 강한 가중치
            elif y_val >= 300:
                weights.append(5.0)   # 300-310 중간 가중치
            else:
                weights.append(1.0)   # 정상 구간
        
        return np.array(X), np.array(y), np.array(X_physics), np.array(weights)

# ========================================
# Model 1: PatchTST (전체 균형)
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
# Model 2: PatchTST + PINN (극단값 특화)
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
        
        # 극단값 부스팅 (310+ 구간 강화)
        boost_factor = self.extreme_boost(combined)
        output = output * (1 + boost_factor * 0.2)  # 최대 20% 부스팅
        
        return tf.squeeze(output, axis=-1)

# ========================================
# 손실 함수 V3
# ========================================

class ExtremeLossV3(tf.keras.losses.Loss):
    """극단값 특화 손실함수 V3"""
    
    def __init__(self, extreme_focus=False):
        super().__init__()
        self.extreme_focus = extreme_focus
        
    def call(self, y_true, y_pred):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        
        mse = tf.square(y_true - y_pred)
        
        if self.extreme_focus:
            # Model 2용 - 극단값 더 강조
            weight = tf.where(y_true > 0.52, 20.0,    # 335+ (스케일 후)
                     tf.where(y_true > 0.45, 15.0,    # 310+
                     tf.where(y_true > 0.4, 5.0,      # 300+
                     1.0)))
        else:
            # Model 1용 - 전체 균형
            weight = tf.where(y_true > 0.52, 10.0,    # 335+
                     tf.where(y_true > 0.45, 8.0,     # 310+
                     tf.where(y_true > 0.4, 3.0,      # 300+
                     1.0)))
        
        return mse * weight

# ========================================
# 메인 실행
# ========================================

def main():
    # 데이터 처리
    processor = DataProcessorV3()
    
    # 1. 데이터 로드
    print("\n[Step 1/5] 데이터 로드")
    df = pd.read_csv('data/HUB_0509_TO_0730_DATA.CSV')
    print(f"✅ 데이터 로드: {df.shape}")
    
    # 2. 데이터 분석
    processor.analyze_data(df)
    
    # 3. 전처리
    print("\n[Step 2/5] 데이터 전처리")
    df['timestamp'] = pd.to_datetime(df.iloc[:, 0], format='%Y%m%d%H%M', errors='coerce')
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df.fillna(method='ffill').fillna(0)
    
    # 4. 시퀀스 생성
    print("\n[Step 3/5] 시퀀스 생성")
    X, y, X_physics, weights = processor.create_sequences_v3(df)
    
    # 5. 데이터 분할
    print("\n[Step 4/5] 데이터 분할 및 스케일링")
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=42)
    
    # 데이터 분할
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    X_physics_train = X_physics[train_idx]
    X_physics_val = X_physics[val_idx]
    X_physics_test = X_physics[test_idx]
    
    weights_train = weights[train_idx]
    
    # 스케일링
    n_features = X.shape[2]
    
    X_train_flat = X_train.reshape(-1, n_features)
    X_train_scaled = processor.scaler_X.fit_transform(X_train_flat)
    X_train_scaled = X_train_scaled.reshape(len(X_train), 20, n_features)
    
    X_val_scaled = processor.scaler_X.transform(X_val.reshape(-1, n_features)).reshape(len(X_val), 20, n_features)
    X_test_scaled = processor.scaler_X.transform(X_test.reshape(-1, n_features)).reshape(len(X_test), 20, n_features)
    
    y_train_scaled = processor.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = processor.scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = processor.scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    X_physics_train_scaled = processor.scaler_physics.fit_transform(X_physics_train)
    X_physics_val_scaled = processor.scaler_physics.transform(X_physics_val)
    X_physics_test_scaled = processor.scaler_physics.transform(X_physics_test)
    
    processor.save_scalers()
    
    print(f"\n📊 데이터셋 크기:")
    print(f"  Train: {len(train_idx)} (310+: {(y_train >= 310).sum()}, 335+: {(y_train >= 335).sum()})")
    print(f"  Valid: {len(val_idx)} (310+: {(y_val >= 310).sum()}, 335+: {(y_val >= 335).sum()})")
    print(f"  Test: {len(test_idx)} (310+: {(y_test >= 310).sum()}, 335+: {(y_test >= 335).sum()})")
    
    # 6. 모델 학습
    print("\n[Step 5/5] 모델 학습")
    
    config = {
        'seq_len': 20,
        'n_features': n_features,
        'patch_len': 5
    }
    
    # 콜백
    callbacks_model1 = [
        EarlyStopping(patience=20, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6),
        ModelCheckpoint('./checkpoints/model1_v3.h5', save_best_only=True, save_weights_only=True),
        ExtremeValueCallback(X_val_scaled, y_val_scaled, processor.scaler_y)
    ]
    
    callbacks_model2 = [
        EarlyStopping(patience=20, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6),
        ModelCheckpoint('./checkpoints/model2_v3.h5', save_best_only=True, save_weights_only=True),
        ExtremeValueCallback((X_val_scaled, X_physics_val_scaled), y_val_scaled, processor.scaler_y)
    ]
    
    # Model 1: PatchTST
    print("\n🤖 Model 1: PatchTST 학습")
    model1 = PatchTSTModel(config)
    model1.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=ExtremeLossV3(extreme_focus=False),
        metrics=['mae']
    )
    
    history1 = model1.fit(
        X_train_scaled, y_train_scaled,
        validation_data=(X_val_scaled, y_val_scaled),
        sample_weight=weights_train,
        epochs=50,
        batch_size=32,
        callbacks=callbacks_model1,
        verbose=1
    )
    
    # Model 2: PatchTST + PINN
    print("\n🤖 Model 2: PatchTST + PINN 학습")
    model2 = PatchTSTPINN(config)
    model2.compile(
        optimizer=Adam(learning_rate=0.0008),
        loss=ExtremeLossV3(extreme_focus=True),
        metrics=['mae']
    )
    
    history2 = model2.fit(
        [X_train_scaled, X_physics_train_scaled], y_train_scaled,
        validation_data=([X_val_scaled, X_physics_val_scaled], y_val_scaled),
        sample_weight=weights_train,
        epochs=60,
        batch_size=32,
        callbacks=callbacks_model2,
        verbose=1
    )
    
    # 7. 평가
    print("\n" + "="*80)
    print("📊 최종 평가")
    print("="*80)
    
    # Model 1 평가
    print("\n[Model 1: PatchTST]")
    y_pred1_scaled = model1.predict(X_test_scaled, verbose=0)
    y_pred1 = processor.scaler_y.inverse_transform(y_pred1_scaled.reshape(-1, 1)).flatten()
    
    mae1 = np.mean(np.abs(y_test - y_pred1))
    print(f"전체 MAE: {mae1:.2f}")
    
    # 3구간 평가
    print("\n3구간 성능:")
    mask_low = y_test < 200
    mask_normal = (y_test >= 200) & (y_test < 300)
    mask_danger = y_test >= 300
    
    if mask_low.sum() > 0:
        print(f"  저구간(<200): MAE={np.mean(np.abs(y_test[mask_low] - y_pred1[mask_low])):.2f}")
    if mask_normal.sum() > 0:
        print(f"  정상(200-300): MAE={np.mean(np.abs(y_test[mask_normal] - y_pred1[mask_normal])):.2f}")
    if mask_danger.sum() > 0:
        print(f"  위험(300+): MAE={np.mean(np.abs(y_test[mask_danger] - y_pred1[mask_danger])):.2f}")
    
    # 극단값 감지
    print("\n극단값 감지:")
    for threshold in [310, 335]:
        mask = y_test >= threshold
        if mask.sum() > 0:
            detected = (y_pred1 >= threshold)[mask].sum()
            print(f"  {threshold}+: {detected}/{mask.sum()} ({detected/mask.sum()*100:.1f}%)")
    
    # Model 2 평가
    print("\n[Model 2: PatchTST + PINN]")
    y_pred2_scaled = model2.predict([X_test_scaled, X_physics_test_scaled], verbose=0)
    y_pred2 = processor.scaler_y.inverse_transform(y_pred2_scaled.reshape(-1, 1)).flatten()
    
    mae2 = np.mean(np.abs(y_test - y_pred2))
    print(f"전체 MAE: {mae2:.2f}")
    
    # 3구간 평가
    print("\n3구간 성능:")
    if mask_low.sum() > 0:
        print(f"  저구간(<200): MAE={np.mean(np.abs(y_test[mask_low] - y_pred2[mask_low])):.2f}")
    if mask_normal.sum() > 0:
        print(f"  정상(200-300): MAE={np.mean(np.abs(y_test[mask_normal] - y_pred2[mask_normal])):.2f}")
    if mask_danger.sum() > 0:
        print(f"  위험(300+): MAE={np.mean(np.abs(y_test[mask_danger] - y_pred2[mask_danger])):.2f}")
    
    # 극단값 감지
    print("\n극단값 감지:")
    for threshold in [310, 335]:
        mask = y_test >= threshold
        if mask.sum() > 0:
            detected = (y_pred2 >= threshold)[mask].sum()
            print(f"  {threshold}+: {detected}/{mask.sum()} ({detected/mask.sum()*100:.1f}%)")
    
    print("\n✅ V3 완료!")

if __name__ == "__main__":
    main()