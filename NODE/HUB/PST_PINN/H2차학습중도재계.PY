# -*- coding: utf-8 -*-
"""
HUBROOM 극단값 예측 시스템 - 최종 정리 버전
- 310+ 극단값 정확한 예측
- 중단/재시작 완벽 지원
- 깔끔한 구조
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
import os
import pickle
from tqdm import tqdm
import joblib

warnings.filterwarnings('ignore')

print(f"TensorFlow Version: {tf.__version__}")
tf.random.set_seed(42)
np.random.seed(42)

# ===========================
# 1. 데이터 처리
# ===========================

class DataProcessor:
    def __init__(self, file_path='data/HUB_0509_TO_0730_DATA.CSV'):
        self.file_path = file_path
        self.target_col = 'CURRENT_M16A_3F_JOB_2'
        
        # 스케일러
        self.scaler_X = RobustScaler()
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))
        
        # 스케일러 경로
        self.scaler_dir = './scalers'
        os.makedirs(self.scaler_dir, exist_ok=True)
        
    def save_scalers(self):
        """스케일러 저장"""
        joblib.dump(self.scaler_X, os.path.join(self.scaler_dir, 'scaler_X.pkl'))
        joblib.dump(self.scaler_y, os.path.join(self.scaler_dir, 'scaler_y.pkl'))
        print(f"✅ 스케일러 저장: {self.scaler_dir}")
    
    def load_scalers(self):
        """스케일러 로드"""
        try:
            self.scaler_X = joblib.load(os.path.join(self.scaler_dir, 'scaler_X.pkl'))
            self.scaler_y = joblib.load(os.path.join(self.scaler_dir, 'scaler_y.pkl'))
            print(f"✅ 스케일러 로드: {self.scaler_dir}")
            return True
        except:
            return False
    
    def analyze_data(self, df):
        """데이터 분석"""
        target = df[self.target_col]
        
        print("\n📊 데이터 분석")
        print(f"  범위: {target.min():.0f} ~ {target.max():.0f}")
        print(f"  평균: {target.mean():.2f}")
        
        # 극단값 분포
        print("\n🚨 극단값 분포:")
        thresholds = [280, 300, 310, 350, 400]
        for t in thresholds:
            count = (target >= t).sum()
            ratio = count / len(target) * 100
            print(f"  ≥{t}: {count}개 ({ratio:.2f}%)")
    
    def create_sequences(self, df, seq_len=20, pred_len=10):
        """시퀀스 생성 (310+ 오버샘플링)"""
        
        # 숫자형 컬럼만
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        X, y, weights = [], [], []
        total_seq = len(df) - seq_len - pred_len + 1
        
        # 구간별 인덱스 수집
        indices_normal = []
        indices_300 = []
        indices_310 = []
        indices_350 = []
        
        print("\n📦 시퀀스 분류 중...")
        for i in tqdm(range(total_seq)):
            target_val = df[self.target_col].iloc[i + seq_len + pred_len - 1]
            
            if target_val >= 350:
                indices_350.append(i)
            elif target_val >= 310:
                indices_310.append(i)
            elif target_val >= 300:
                indices_300.append(i)
            else:
                indices_normal.append(i)
        
        # 오버샘플링
        all_indices = []
        all_indices.extend(indices_normal)  # 1배
        all_indices.extend(indices_300 * 2)  # 2배
        all_indices.extend(indices_310 * 5)  # 5배 (310+ 강화)
        all_indices.extend(indices_350 * 8)  # 8배 (350+ 더 강화)
        
        print(f"  정상: {len(indices_normal)}")
        print(f"  300+: {len(indices_300)} → {len(indices_300)*2}")
        print(f"  310+: {len(indices_310)} → {len(indices_310)*5}")
        print(f"  350+: {len(indices_350)} → {len(indices_350)*8}")
        
        np.random.shuffle(all_indices)
        
        # 시퀀스 생성
        print(f"\n📦 시퀀스 생성 중... (총 {len(all_indices)}개)")
        for i in tqdm(all_indices):
            X_seq = df[numeric_cols].iloc[i:i+seq_len].values
            y_val = df[self.target_col].iloc[i + seq_len + pred_len - 1]
            
            X.append(X_seq)
            y.append(y_val)
            
            # 가중치
            if y_val >= 350:
                weights.append(15.0)
            elif y_val >= 310:
                weights.append(10.0)
            elif y_val >= 300:
                weights.append(5.0)
            else:
                weights.append(1.0)
        
        return np.array(X), np.array(y), np.array(weights)

# ===========================
# 2. 모델 정의
# ===========================

class ExtremePatchTST(keras.Model):
    def __init__(self, config):
        super().__init__()
        
        self.seq_len = config['seq_len']
        self.n_features = config['n_features']
        self.patch_len = config['patch_len']
        self.n_patches = self.seq_len // self.patch_len
        
        # 패치 임베딩
        self.patch_embed = layers.Dense(128, activation='relu')
        
        # Transformer 블록
        self.attention = layers.MultiHeadAttention(num_heads=8, key_dim=16)
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        
        # FFN
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
        
        # 극단값 보정
        self.extreme_detector = keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
    
    def call(self, x, training=False):
        batch_size = tf.shape(x)[0]
        
        # 패치 생성
        x = tf.reshape(x, [batch_size, self.n_patches, self.patch_len * self.n_features])
        
        # 패치 임베딩
        x = self.patch_embed(x)
        
        # Transformer
        attn = self.attention(x, x, training=training)
        x = self.norm1(x + attn)
        
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        # 출력
        x = self.flatten(x)
        
        # 극단값 감지
        extreme_prob = self.extreme_detector(x)
        
        # 값 예측
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        value = self.output_layer(x)
        
        # 극단값 보정 (310+ 강화)
        # extreme_prob이 높으면 더 큰 값으로 보정
        adjustment = extreme_prob * 80  # 최대 80 추가
        final_output = value + adjustment
        
        return tf.squeeze(final_output, axis=-1)

# ===========================
# 3. 손실함수
# ===========================

class ExtremeLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()
        
    def call(self, y_true, y_pred):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        
        # MSE
        mse = tf.square(y_true - y_pred)
        
        # 극단값 가중치 (정규화된 값 기준)
        # MinMaxScaler 후: 300 → ~0.4, 310 → ~0.45, 350 → ~0.55
        weight = tf.where(y_true > 0.55, 20.0,  # 350+
                 tf.where(y_true > 0.45, 10.0,  # 310+
                 tf.where(y_true > 0.4, 5.0,    # 300+
                 1.0)))
        
        return mse * weight

# ===========================
# 4. 메인 실행
# ===========================

def main():
    print("="*80)
    print("🏭 HUBROOM 극단값 예측 시스템 - 최종 버전")
    print("="*80)
    
    # 체크포인트 디렉토리
    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 진행 상태 파일
    state_file = os.path.join(checkpoint_dir, 'training_state.pkl')
    
    # 이전 상태 확인
    if os.path.exists(state_file):
        with open(state_file, 'rb') as f:
            state = pickle.load(f)
        print("📂 이전 상태 로드")
        phase = state.get('phase', 1)
    else:
        state = {}
        phase = 1
    
    processor = DataProcessor()
    
    # Phase 1: 데이터 로드
    if phase == 1:
        print("\n[Phase 1/4] 데이터 준비")
        
        # 데이터 로드
        df = pd.read_csv(processor.file_path)
        print(f"✅ 데이터 로드: {df.shape}")
        
        # 분석
        processor.analyze_data(df)
        
        # 전처리
        df['timestamp'] = pd.to_datetime(df.iloc[:, 0], format='%Y%m%d%H%M', errors='coerce')
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = df.fillna(method='ffill').fillna(0)
        
        # 시퀀스 생성
        X, y, weights = processor.create_sequences(df)
        
        # 데이터 분할
        indices = np.arange(len(X))
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
        val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=42)
        
        state['X_train'] = X[train_idx]
        state['y_train'] = y[train_idx]
        state['X_val'] = X[val_idx]
        state['y_val'] = y[val_idx]
        state['X_test'] = X[test_idx]
        state['y_test'] = y[test_idx]
        state['weights_train'] = weights[train_idx]
        state['n_features'] = X.shape[2]
        state['phase'] = 2
        
        with open(state_file, 'wb') as f:
            pickle.dump(state, f)
        print("💾 Phase 1 완료")
    
    # Phase 2: 스케일링
    if phase <= 2:
        print("\n[Phase 2/4] 데이터 정규화")
        
        # 데이터 로드
        X_train = state['X_train']
        y_train = state['y_train']
        X_val = state['X_val']
        y_val = state['y_val']
        X_test = state['X_test']
        y_test = state['y_test']
        n_features = state['n_features']
        
        # 스케일링
        n_train = len(X_train)
        n_val = len(X_val)
        n_test = len(X_test)
        
        X_train_scaled = processor.scaler_X.fit_transform(X_train.reshape(-1, n_features))
        X_train_scaled = X_train_scaled.reshape(n_train, 20, n_features)
        
        X_val_scaled = processor.scaler_X.transform(X_val.reshape(-1, n_features))
        X_val_scaled = X_val_scaled.reshape(n_val, 20, n_features)
        
        X_test_scaled = processor.scaler_X.transform(X_test.reshape(-1, n_features))
        X_test_scaled = X_test_scaled.reshape(n_test, 20, n_features)
        
        y_train_scaled = processor.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = processor.scaler_y.transform(y_val.reshape(-1, 1)).flatten()
        y_test_scaled = processor.scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        
        # 스케일러 저장
        processor.save_scalers()
        
        state['X_train_scaled'] = X_train_scaled
        state['y_train_scaled'] = y_train_scaled
        state['X_val_scaled'] = X_val_scaled
        state['y_val_scaled'] = y_val_scaled
        state['X_test_scaled'] = X_test_scaled
        state['y_test_scaled'] = y_test_scaled
        state['phase'] = 3
        
        with open(state_file, 'wb') as f:
            pickle.dump(state, f)
        print("💾 Phase 2 완료")
    
    # Phase 3: 학습
    if phase <= 3:
        print("\n[Phase 3/4] 모델 학습")
        
        # 스케일러 로드
        if not processor.load_scalers():
            print("❌ 스케일러 로드 실패")
            return
        
        # 데이터 로드
        X_train_scaled = state['X_train_scaled']
        y_train_scaled = state['y_train_scaled']
        X_val_scaled = state['X_val_scaled']
        y_val_scaled = state['y_val_scaled']
        weights_train = state['weights_train']
        n_features = state['n_features']
        
        # 모델 구성
        config = {
            'seq_len': 20,
            'n_features': n_features,
            'patch_len': 5
        }
        
        model = ExtremePatchTST(config)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=ExtremeLoss(),
            metrics=['mae']
        )
        
        # 콜백
        callbacks = [
            ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, 'best_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True
            ),
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        # 학습
        print(f"학습 데이터: {len(X_train_scaled)}")
        print(f"310+ 샘플: {(state['y_train'] >= 310).sum()}")
        
        history = model.fit(
            X_train_scaled, y_train_scaled,
            validation_data=(X_val_scaled, y_val_scaled),
            sample_weight=weights_train,
            epochs=50,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # 저장
        model.save_weights(os.path.join(checkpoint_dir, 'final_model.h5'))
        state['phase'] = 4
        
        with open(state_file, 'wb') as f:
            pickle.dump(state, f)
        print("💾 Phase 3 완료")
    
    # Phase 4: 평가
    print("\n[Phase 4/4] 모델 평가")
    
    # 스케일러 로드
    if not processor.load_scalers():
        print("❌ 스케일러 로드 실패")
        return
    
    # 모델 재구성
    config = {
        'seq_len': 20,
        'n_features': state['n_features'],
        'patch_len': 5
    }
    
    model = ExtremePatchTST(config)
    model.compile(optimizer='adam', loss='mse')
    
    # 가중치 로드
    model.load_weights(os.path.join(checkpoint_dir, 'final_model.h5'))
    
    # 예측
    X_test_scaled = state['X_test_scaled']
    y_test_scaled = state['y_test_scaled']
    y_test = state['y_test']
    
    y_pred_scaled = model.predict(X_test_scaled, verbose=0)
    y_pred = processor.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    # 평가
    mae = np.mean(np.abs(y_test - y_pred))
    print(f"\n전체 MAE: {mae:.2f}")
    
    # 310+ 성능
    mask_310 = y_test >= 310
    if mask_310.sum() > 0:
        mae_310 = np.mean(np.abs(y_test[mask_310] - y_pred[mask_310]))
        detected = (y_pred >= 310)[mask_310].sum()
        print(f"\n🎯 310+ 극단값 성능:")
        print(f"  MAE: {mae_310:.2f}")
        print(f"  감지: {detected}/{mask_310.sum()} ({detected/mask_310.sum()*100:.1f}%)")
        
        # 샘플
        print(f"\n📝 310+ 예측 샘플:")
        indices = np.where(mask_310)[0][:5]
        for idx in indices:
            print(f"  실제: {y_test[idx]:.0f}, 예측: {y_pred[idx]:.0f}")
    
    print("\n✅ 완료!")

if __name__ == "__main__":
    main()