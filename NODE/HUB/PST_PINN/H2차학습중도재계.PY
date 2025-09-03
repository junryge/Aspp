# -*- coding: utf-8 -*-
"""
HUBROOM 극단값 예측 시스템 - 완전 새 버전
- ExtremePatchTST: Transformer 기반
- ImprovedPINN: 물리 법칙 기반
310+ 극단값 예측 특화
중단/재시작 지원
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import os
import pickle
import warnings
from tqdm import tqdm
import joblib
import signal
import sys

warnings.filterwarnings('ignore')

# 설정
np.random.seed(42)
tf.random.set_seed(42)
print(f"TensorFlow Version: {tf.__version__}")

# ========================================
# 상태 관리
# ========================================

class CheckpointManager:
    def __init__(self, checkpoint_dir='./checkpoints'):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.state_file = os.path.join(checkpoint_dir, 'training_state.pkl')
        self.interrupted = False
        
        # Ctrl+C 핸들러
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        print('\n\n⚠️ 중단 감지! 상태 저장 중...')
        self.interrupted = True
        
    def save_state(self, state):
        with open(self.state_file, 'wb') as f:
            pickle.dump(state, f)
        print(f"💾 상태 저장 완료")
        
    def load_state(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, 'rb') as f:
                return pickle.load(f)
        return None

# ========================================
# 데이터 처리
# ========================================

class DataProcessor:
    def __init__(self):
        self.target_col = 'CURRENT_M16A_3F_JOB_2'
        self.scaler_X = RobustScaler()
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))
        self.scaler_physics = StandardScaler()
        
        # 물리 컬럼
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
            return False
    
    def analyze_data(self, df):
        target = df[self.target_col]
        print("\n📊 데이터 분석:")
        print(f"  범위: {target.min():.0f} ~ {target.max():.0f}")
        print(f"  평균: {target.mean():.1f}")
        
        print("\n🚨 극단값 분포:")
        thresholds = [280, 300, 310, 350, 400]
        for t in thresholds:
            count = (target >= t).sum()
            ratio = count / len(target) * 100
            print(f"  ≥{t}: {count:6}개 ({ratio:5.2f}%)")
    
    def create_sequences(self, df, seq_len=20, pred_len=10):
        """시퀀스 생성 (310+ 오버샘플링)"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 물리 데이터용 컬럼 체크
        available_inflow = [col for col in self.inflow_cols if col in df.columns]
        available_outflow = [col for col in self.outflow_cols if col in df.columns]
        
        X, y, X_physics, weights = [], [], [], []
        total = len(df) - seq_len - pred_len + 1
        
        # 구간별 인덱스
        indices = {'normal': [], '300': [], '310': [], '350': []}
        
        print("\n📦 시퀀스 분류 중...")
        for i in tqdm(range(total)):
            target_val = df[self.target_col].iloc[i + seq_len + pred_len - 1]
            
            if target_val >= 350:
                indices['350'].append(i)
            elif target_val >= 310:
                indices['310'].append(i)
            elif target_val >= 300:
                indices['300'].append(i)
            else:
                indices['normal'].append(i)
        
        # 오버샘플링
        all_indices = []
        all_indices.extend(indices['normal'])     # 1배
        all_indices.extend(indices['300'] * 3)    # 3배
        all_indices.extend(indices['310'] * 5)    # 5배 (핵심)
        all_indices.extend(indices['350'] * 8)    # 8배
        
        print(f"  정상: {len(indices['normal'])}")
        print(f"  300+: {len(indices['300'])} → {len(indices['300'])*3}")
        print(f"  310+: {len(indices['310'])} → {len(indices['310'])*5}")
        print(f"  350+: {len(indices['350'])} → {len(indices['350'])*8}")
        
        np.random.shuffle(all_indices)
        
        # 시퀀스 생성
        print(f"\n시퀀스 생성 중... (총 {len(all_indices)}개)")
        for i in tqdm(all_indices):
            # 시계열 데이터
            X.append(df[numeric_cols].iloc[i:i+seq_len].values)
            
            # 타겟
            y_val = df[self.target_col].iloc[i + seq_len + pred_len - 1]
            y.append(y_val)
            
            # 물리 데이터
            physics = [
                df[self.target_col].iloc[i + seq_len - 1],  # 현재값
                df[available_inflow].iloc[i+seq_len:i+seq_len+pred_len].sum().sum() if available_inflow else 0,
                df[available_outflow].iloc[i+seq_len:i+seq_len+pred_len].sum().sum() if available_outflow else 0
            ]
            X_physics.append(physics)
            
            # 가중치
            if y_val >= 350:
                weights.append(15.0)
            elif y_val >= 310:
                weights.append(10.0)
            elif y_val >= 300:
                weights.append(5.0)
            else:
                weights.append(1.0)
        
        return np.array(X), np.array(y), np.array(X_physics), np.array(weights)

# ========================================
# 모델 1: ExtremePatchTST
# ========================================

class ExtremePatchTST(keras.Model):
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
# 모델 2: ImprovedPINN
# ========================================

class ImprovedPINN(keras.Model):
    def __init__(self, config):
        super().__init__()
        
        # LSTM for 시계열
        self.lstm = layers.LSTM(64, return_sequences=False)
        
        # 물리 정보 처리
        self.physics_net = keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(16, activation='relu')
        ])
        
        # 융합
        self.fusion = keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        
    def call(self, inputs, training=False):
        x_seq, x_physics = inputs
        
        # 시계열 처리
        seq_features = self.lstm(x_seq)
        
        # 물리 정보 처리
        physics_features = self.physics_net(x_physics)
        
        # 결합
        combined = tf.concat([seq_features, physics_features], axis=-1)
        
        # 출력
        output = self.fusion(combined)
        
        return tf.squeeze(output, axis=-1)

# ========================================
# 커스텀 손실함수
# ========================================

class ExtremeLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()
        
    def call(self, y_true, y_pred):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        
        mse = tf.square(y_true - y_pred)
        
        # MinMaxScaler 후 값 (300→0.4, 310→0.45, 350→0.55 대략)
        weight = tf.where(y_true > 0.55, 15.0,    # 350+
                 tf.where(y_true > 0.45, 10.0,    # 310+
                 tf.where(y_true > 0.4, 5.0,      # 300+
                 1.0)))
        
        return mse * weight

# ========================================
# 메인 실행
# ========================================

def main():
    print("="*80)
    print("🏭 HUBROOM 극단값 예측 시스템")
    print("🎯 목표: 310+ 극단값 정확 예측")
    print("="*80)
    
    # 체크포인트 관리
    ckpt = CheckpointManager()
    
    # 상태 로드
    state = ckpt.load_state()
    if state:
        print("\n📂 이전 상태 로드")
        step = state.get('step', 1)
    else:
        state = {}
        step = 1
    
    processor = DataProcessor()
    
    # Step 1: 데이터 준비
    if step == 1:
        print("\n[Step 1/5] 데이터 준비")
        
        # 데이터 로드
        df = pd.read_csv('data/HUB_0509_TO_0730_DATA.CSV')
        print(f"✅ 데이터 로드: {df.shape}")
        
        # 분석
        processor.analyze_data(df)
        
        # 전처리
        df['timestamp'] = pd.to_datetime(df.iloc[:, 0], format='%Y%m%d%H%M', errors='coerce')
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = df.fillna(method='ffill').fillna(0)
        
        # 시퀀스 생성
        X, y, X_physics, weights = processor.create_sequences(df)
        
        state['X'] = X
        state['y'] = y
        state['X_physics'] = X_physics
        state['weights'] = weights
        state['n_features'] = X.shape[2]
        state['step'] = 2
        
        ckpt.save_state(state)
        print("✅ Step 1 완료")
    
    # Step 2: 데이터 분할
    if step <= 2:
        print("\n[Step 2/5] 데이터 분할")
        
        X = state['X']
        y = state['y']
        X_physics = state['X_physics']
        weights = state['weights']
        
        # 인덱스 분할
        indices = np.arange(len(X))
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
        val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=42)
        
        state['train_idx'] = train_idx
        state['val_idx'] = val_idx
        state['test_idx'] = test_idx
        state['step'] = 3
        
        ckpt.save_state(state)
        
        print(f"  Train: {len(train_idx)} ({(y[train_idx] >= 310).sum()} 310+)")
        print(f"  Valid: {len(val_idx)} ({(y[val_idx] >= 310).sum()} 310+)")
        print(f"  Test: {len(test_idx)} ({(y[test_idx] >= 310).sum()} 310+)")
        print("✅ Step 2 완료")
    
    # Step 3: 스케일링
    if step <= 3:
        print("\n[Step 3/5] 데이터 스케일링")
        
        X = state['X']
        y = state['y']
        X_physics = state['X_physics']
        weights = state['weights']
        n_features = state['n_features']
        
        train_idx = state['train_idx']
        val_idx = state['val_idx']
        test_idx = state['test_idx']
        
        # 데이터 분할
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        X_physics_train = X_physics[train_idx]
        X_physics_val = X_physics[val_idx]
        X_physics_test = X_physics[test_idx]
        
        weights_train = weights[train_idx]
        
        # X 스케일링
        X_train_flat = X_train.reshape(-1, n_features)
        X_train_scaled = processor.scaler_X.fit_transform(X_train_flat)
        X_train_scaled = X_train_scaled.reshape(len(X_train), 20, n_features)
        
        X_val_flat = X_val.reshape(-1, n_features)
        X_val_scaled = processor.scaler_X.transform(X_val_flat)
        X_val_scaled = X_val_scaled.reshape(len(X_val), 20, n_features)
        
        X_test_flat = X_test.reshape(-1, n_features)
        X_test_scaled = processor.scaler_X.transform(X_test_flat)
        X_test_scaled = X_test_scaled.reshape(len(X_test), 20, n_features)
        
        # y 스케일링
        y_train_scaled = processor.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = processor.scaler_y.transform(y_val.reshape(-1, 1)).flatten()
        y_test_scaled = processor.scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        
        # 물리 데이터 스케일링
        X_physics_train_scaled = processor.scaler_physics.fit_transform(X_physics_train)
        X_physics_val_scaled = processor.scaler_physics.transform(X_physics_val)
        X_physics_test_scaled = processor.scaler_physics.transform(X_physics_test)
        
        # 스케일러 저장
        processor.save_scalers()
        
        state['X_train_scaled'] = X_train_scaled
        state['y_train_scaled'] = y_train_scaled
        state['X_val_scaled'] = X_val_scaled
        state['y_val_scaled'] = y_val_scaled
        state['X_test_scaled'] = X_test_scaled
        state['y_test_scaled'] = y_test_scaled
        state['X_physics_train_scaled'] = X_physics_train_scaled
        state['X_physics_val_scaled'] = X_physics_val_scaled
        state['X_physics_test_scaled'] = X_physics_test_scaled
        state['weights_train'] = weights_train
        state['y_test'] = y_test
        state['step'] = 4
        
        ckpt.save_state(state)
        print("✅ Step 3 완료")
    
    # Step 4: 모델 학습
    if step <= 4:
        print("\n[Step 4/5] 모델 학습")
        
        # 스케일러 로드
        if not processor.load_scalers():
            print("❌ 스케일러 로드 실패")
            return
        
        X_train_scaled = state['X_train_scaled']
        y_train_scaled = state['y_train_scaled']
        X_val_scaled = state['X_val_scaled']
        y_val_scaled = state['y_val_scaled']
        X_physics_train_scaled = state['X_physics_train_scaled']
        X_physics_val_scaled = state['X_physics_val_scaled']
        weights_train = state['weights_train']
        n_features = state['n_features']
        
        config = {
            'seq_len': 20,
            'n_features': n_features,
            'patch_len': 5
        }
        
        # 콜백
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6),
            ModelCheckpoint('./checkpoints/best_model_{epoch:02d}.h5', 
                          save_best_only=True, save_weights_only=True)
        ]
        
        # 모델 1: ExtremePatchTST
        print("\n🤖 모델 1: ExtremePatchTST 학습")
        model1 = ExtremePatchTST(config)
        model1.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=ExtremeLoss(),
            metrics=['mae']
        )
        
        history1 = model1.fit(
            X_train_scaled, y_train_scaled,
            validation_data=(X_val_scaled, y_val_scaled),
            sample_weight=weights_train,
            epochs=50,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        model1.save_weights('./checkpoints/model1_final.h5')
        
        # 모델 2: ImprovedPINN
        print("\n🤖 모델 2: ImprovedPINN 학습")
        model2 = ImprovedPINN(config)
        model2.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=ExtremeLoss(),
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
        
        model2.save_weights('./checkpoints/model2_final.h5')
        
        state['step'] = 5
        ckpt.save_state(state)
        print("✅ Step 4 완료")
    
    # Step 5: 평가
    print("\n[Step 5/5] 모델 평가")
    
    # 스케일러 로드
    if not processor.load_scalers():
        print("❌ 스케일러 로드 실패")
        return
    
    X_test_scaled = state['X_test_scaled']
    y_test_scaled = state['y_test_scaled']
    X_physics_test_scaled = state['X_physics_test_scaled']
    y_test = state['y_test']
    n_features = state['n_features']
    
    config = {
        'seq_len': 20,
        'n_features': n_features,
        'patch_len': 5
    }
    
    # 모델 1 평가
    print("\n📊 모델 1: ExtremePatchTST 평가")
    model1 = ExtremePatchTST(config)
    model1.compile(optimizer='adam', loss='mse')
    
    # 더미 데이터로 모델 빌드
    dummy_input = np.zeros((1, 20, n_features))
    _ = model1(dummy_input)
    
    # 가중치 로드
    model1.load_weights('./checkpoints/model1_final.h5')
    
    y_pred1_scaled = model1.predict(X_test_scaled, verbose=0)
    y_pred1 = processor.scaler_y.inverse_transform(y_pred1_scaled.reshape(-1, 1)).flatten()
    
    mae1 = np.mean(np.abs(y_test - y_pred1))
    print(f"  전체 MAE: {mae1:.2f}")
    
    mask_310 = y_test >= 310
    if mask_310.sum() > 0:
        mae_310 = np.mean(np.abs(y_test[mask_310] - y_pred1[mask_310]))
        detected = (y_pred1 >= 310)[mask_310].sum()
        print(f"  310+ MAE: {mae_310:.2f}")
        print(f"  310+ 감지: {detected}/{mask_310.sum()} ({detected/mask_310.sum()*100:.1f}%)")
    
    # 모델 2 평가
    print("\n📊 모델 2: ImprovedPINN 평가")
    model2 = ImprovedPINN(config)
    model2.compile(optimizer='adam', loss='mse')
    
    # 더미 데이터로 모델 빌드
    dummy_seq = np.zeros((1, 20, n_features))
    dummy_physics = np.zeros((1, 3))
    _ = model2([dummy_seq, dummy_physics])
    
    # 가중치 로드
    model2.load_weights('./checkpoints/model2_final.h5')
    
    y_pred2_scaled = model2.predict([X_test_scaled, X_physics_test_scaled], verbose=0)
    y_pred2 = processor.scaler_y.inverse_transform(y_pred2_scaled.reshape(-1, 1)).flatten()
    
    mae2 = np.mean(np.abs(y_test - y_pred2))
    print(f"  전체 MAE: {mae2:.2f}")
    
    if mask_310.sum() > 0:
        mae_310 = np.mean(np.abs(y_test[mask_310] - y_pred2[mask_310]))
        detected = (y_pred2 >= 310)[mask_310].sum()
        print(f"  310+ MAE: {mae_310:.2f}")
        print(f"  310+ 감지: {detected}/{mask_310.sum()} ({detected/mask_310.sum()*100:.1f}%)")
    
    print("\n✅ 완료!")

if __name__ == "__main__":
    main()