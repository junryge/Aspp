"""
train_v6_memory_efficient.py - 메모리 효율적인 학습 코드
BATCH_SIZE=32 사용 가능 버전
메모리 매핑과 제너레이터 사용으로 RAM 사용 최소화
"""

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json
import os
import gc
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("🚀 반도체 물류 예측 학습 V6 (메모리 효율 버전)")
print(f"📦 TensorFlow 버전: {tf.__version__}")
print("💾 메모리 효율 모드: BATCH_SIZE=32 사용")
print("="*60)

# 메모리 최적화 설정
gc.collect()
tf.keras.backend.clear_session()

# ============================================
# 1. 설정
# ============================================
class Config:
    # 시퀀스 파일
    SEQUENCE_FILE = './sequences_v6.npz'
    
    # M14 임계값
    M14B_THRESHOLDS = {
        1400: 320,
        1500: 400,
        1600: 450,
        1700: 500
    }
    
    RATIO_THRESHOLDS = {
        1400: 4,
        1500: 5,
        1600: 6,
        1700: 7
    }
    
    # 학습 설정
    BATCH_SIZE = 32  # 32 유지!
    EPOCHS = 100
    LEARNING_RATE = 0.001
    PATIENCE = 15
    
    # 모델 저장 경로
    MODEL_DIR = './models_v6/'
    
    # 가중치 설정
    SPIKE_WEIGHTS = {
        'normal': 1.0,
        'level_1400': 3.0,
        'level_1500': 5.0,
        'level_1600': 8.0,
        'level_1700': 10.0
    }

# 디렉토리 생성
os.makedirs(Config.MODEL_DIR, exist_ok=True)

# ============================================
# 2. 메모리 효율적인 데이터 제너레이터
# ============================================
class MemoryEfficientGenerator(tf.keras.utils.Sequence):
    """메모리 매핑을 사용하는 데이터 제너레이터"""
    
    def __init__(self, npz_path, indices, batch_size=32, for_ensemble=False):
        self.npz_path = npz_path
        self.indices = indices
        self.batch_size = batch_size
        self.for_ensemble = for_ensemble
        
        # 메타데이터만 메모리에 로드
        with np.load(npz_path) as data:
            self.data_shape = data['X'].shape
            self.n_features = data['X'].shape[2]
            
            # y와 m14는 작으니까 메모리에 로드
            self.y_all = data['y'][:]
            self.m14_all = data['m14_features'][:]
    
    def __len__(self):
        return len(self.indices) // self.batch_size
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        # 배치만큼만 메모리에 로드
        with np.load(self.npz_path, mmap_mode='r') as data:
            X_batch = np.array(data['X'][batch_indices], dtype=np.float32)
        
        y_batch = self.y_all[batch_indices]
        
        if self.for_ensemble:
            m14_batch = self.m14_all[batch_indices]
            return [X_batch, m14_batch], y_batch
        else:
            return X_batch, y_batch
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)

class SpikeGenerator(tf.keras.utils.Sequence):
    """Spike Detector용 제너레이터 (이중 출력)"""
    
    def __init__(self, npz_path, indices, batch_size=32):
        self.npz_path = npz_path
        self.indices = indices
        self.batch_size = batch_size
        
        with np.load(npz_path) as data:
            self.y_all = data['y'][:]
            self.y_spike_class = (self.y_all >= 1400).astype(np.float32)
    
    def __len__(self):
        return len(self.indices) // self.batch_size
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        with np.load(self.npz_path, mmap_mode='r') as data:
            X_batch = np.array(data['X'][batch_indices], dtype=np.float32)
        
        y_batch = self.y_all[batch_indices]
        y_class_batch = self.y_spike_class[batch_indices]
        
        return X_batch, {'spike_value': y_batch, 'spike_prob': y_class_batch}

class EnsembleGenerator(tf.keras.utils.Sequence):
    """앙상블 모델용 제너레이터"""
    
    def __init__(self, npz_path, indices, batch_size=32):
        self.npz_path = npz_path
        self.indices = indices
        self.batch_size = batch_size
        
        with np.load(npz_path) as data:
            self.y_all = data['y'][:]
            self.m14_all = data['m14_features'][:]
            self.y_spike_class = (self.y_all >= 1400).astype(np.float32)
    
    def __len__(self):
        return len(self.indices) // self.batch_size
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        with np.load(self.npz_path, mmap_mode='r') as data:
            X_batch = np.array(data['X'][batch_indices], dtype=np.float32)
        
        y_batch = self.y_all[batch_indices]
        m14_batch = self.m14_all[batch_indices]
        y_class_batch = self.y_spike_class[batch_indices]
        
        return [X_batch, m14_batch], [y_batch, y_class_batch]

# ============================================
# 3. 데이터 인덱스 준비
# ============================================
print("\n📂 데이터 준비 중...")

# 메타데이터만 로드
with np.load(Config.SEQUENCE_FILE) as data:
    total_samples = len(data['y'])
    input_shape = data['X'].shape[1:]  # (100, 47)
    m14_shape = data['m14_features'].shape[1]  # 4
    
    # 통계 출력용
    y_temp = data['y'][:]
    print(f"  전체 샘플: {total_samples:,}개")
    print(f"  입력 shape: {input_shape}")
    print(f"  1400+ 비율: {(y_temp >= 1400).mean():.1%}")
    del y_temp

# 학습/검증 인덱스 분할
np.random.seed(42)
all_indices = np.arange(total_samples)
np.random.shuffle(all_indices)
split_point = int(total_samples * 0.8)
train_indices = all_indices[:split_point]
val_indices = all_indices[split_point:]

print(f"\n📊 데이터 분할:")
print(f"  학습: {len(train_indices):,}개")
print(f"  검증: {len(val_indices):,}개")

# ============================================
# 4. 모델 정의
# ============================================
class ModelsV6:
    
    @staticmethod
    def build_lstm_model(input_shape):
        """LSTM 모델"""
        inputs = tf.keras.Input(shape=input_shape, name='lstm_input')
        
        lstm1 = tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.2)(inputs)
        lstm2 = tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.2)(lstm1)
        lstm3 = tf.keras.layers.LSTM(64, dropout=0.2)(lstm2)
        
        dense1 = tf.keras.layers.Dense(128, activation='relu')(lstm3)
        dropout = tf.keras.layers.Dropout(0.3)(dense1)
        dense2 = tf.keras.layers.Dense(64, activation='relu')(dropout)
        
        output = tf.keras.layers.Dense(1, name='lstm_output')(dense2)
        
        model = tf.keras.Model(inputs=inputs, outputs=output, name='LSTM_Model')
        return model
    
    @staticmethod
    def build_enhanced_gru(input_shape):
        """GRU 모델"""
        inputs = tf.keras.Input(shape=input_shape, name='gru_input')
        
        x = tf.keras.layers.LayerNormalization()(inputs)
        
        gru1 = tf.keras.layers.GRU(128, return_sequences=True, dropout=0.2)(x)
        gru2 = tf.keras.layers.GRU(128, return_sequences=True, dropout=0.2)(gru1)
        residual = tf.keras.layers.Add()([gru1, gru2])
        gru3 = tf.keras.layers.GRU(64, dropout=0.2)(residual)
        
        dense1 = tf.keras.layers.Dense(128, activation='relu')(gru3)
        dropout = tf.keras.layers.Dropout(0.3)(dense1)
        dense2 = tf.keras.layers.Dense(64, activation='relu')(dropout)
        
        output = tf.keras.layers.Dense(1, name='gru_output')(dense2)
        
        model = tf.keras.Model(inputs=inputs, outputs=output, name='GRU_Model')
        return model
    
    @staticmethod
    def build_cnn_lstm(input_shape):
        """CNN-LSTM 모델"""
        inputs = tf.keras.Input(shape=input_shape, name='cnn_lstm_input')
        
        conv1 = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
        conv2 = tf.keras.layers.Conv1D(64, 5, activation='relu', padding='same')(inputs)
        conv3 = tf.keras.layers.Conv1D(64, 7, activation='relu', padding='same')(inputs)
        
        concat = tf.keras.layers.Concatenate()([conv1, conv2, conv3])
        pool = tf.keras.layers.MaxPooling1D(pool_size=2)(concat)
        
        lstm1 = tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.2)(pool)
        lstm2 = tf.keras.layers.LSTM(64, dropout=0.2)(lstm1)
        
        dense1 = tf.keras.layers.Dense(128, activation='relu')(lstm2)
        dropout = tf.keras.layers.Dropout(0.3)(dense1)
        dense2 = tf.keras.layers.Dense(64, activation='relu')(dropout)
        
        output = tf.keras.layers.Dense(1, name='cnn_lstm_output')(dense2)
        
        model = tf.keras.Model(inputs=inputs, outputs=output, name='CNN_LSTM_Model')
        return model
    
    @staticmethod
    def build_spike_detector(input_shape):
        """Spike Detector 모델"""
        inputs = tf.keras.Input(shape=input_shape, name='time_series_input')
        
        conv1 = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
        conv2 = tf.keras.layers.Conv1D(64, 5, activation='relu', padding='same')(inputs)
        conv3 = tf.keras.layers.Conv1D(64, 7, activation='relu', padding='same')(inputs)
        
        concat = tf.keras.layers.Concatenate()([conv1, conv2, conv3])
        norm = tf.keras.layers.BatchNormalization()(concat)
        
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=4, key_dim=48, dropout=0.2
        )(norm, norm)
        
        lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.2)
        )(attention)
        
        pooled = tf.keras.layers.GlobalAveragePooling1D()(lstm)
        
        dense1 = tf.keras.layers.Dense(256, activation='relu')(pooled)
        dropout1 = tf.keras.layers.Dropout(0.3)(dense1)
        dense2 = tf.keras.layers.Dense(128, activation='relu')(dropout1)
        dropout2 = tf.keras.layers.Dropout(0.2)(dense2)
        
        regression_output = tf.keras.layers.Dense(1, name='spike_value')(dropout2)
        classification_output = tf.keras.layers.Dense(1, activation='sigmoid', name='spike_prob')(dropout2)
        
        model = tf.keras.Model(
            inputs=inputs,
            outputs=[regression_output, classification_output],
            name='Spike_Detector'
        )
        return model

# ============================================
# 5. 커스텀 레이어 및 손실 함수
# ============================================
class M14RuleCorrection(tf.keras.layers.Layer):
    """M14 규칙 기반 보정 레이어"""
    def __init__(self):
        super().__init__()
        
    def call(self, inputs):
        pred, m14_features = inputs
        
        m14b = m14_features[:, 0:1]
        m10a = m14_features[:, 1:2]
        ratio = m14_features[:, 3:4]
        
        # 규칙 기반 보정
        condition_1700 = tf.logical_and(
            tf.greater_equal(m14b, 500),
            tf.greater_equal(ratio, 7)
        )
        pred = tf.where(condition_1700, tf.maximum(pred, 1700), pred)
        
        condition_1600 = tf.logical_and(
            tf.greater_equal(m14b, 450),
            tf.greater_equal(ratio, 6)
        )
        pred = tf.where(condition_1600, tf.maximum(pred, 1600), pred)
        
        condition_1500 = tf.logical_and(
            tf.greater_equal(m14b, 400),
            tf.greater_equal(ratio, 5)
        )
        pred = tf.where(condition_1500, tf.maximum(pred, 1500), pred)
        
        condition_1400 = tf.greater_equal(m14b, 320)
        pred = tf.where(condition_1400, tf.maximum(pred, 1400), pred)
        
        condition_inverse = tf.logical_and(
            tf.less(m10a, 70),
            tf.greater_equal(m14b, 250)
        )
        pred = tf.where(condition_inverse, pred * 1.08, pred)
        
        return pred

class WeightedLoss(tf.keras.losses.Loss):
    """레벨별 가중 손실 함수"""
    def __init__(self):
        super().__init__()
        
    def call(self, y_true, y_pred):
        weights = tf.where(y_true < 1400, 1.0,
                 tf.where(y_true < 1500, 3.0,
                 tf.where(y_true < 1600, 5.0,
                 tf.where(y_true < 1700, 8.0, 10.0))))
        
        mae = tf.abs(y_true - y_pred)
        weighted_mae = mae * weights
        
        return tf.reduce_mean(weighted_mae)

# ============================================
# 6. 학습 파이프라인
# ============================================
print("\n" + "="*60)
print("🏋️ 모델 학습 시작 (메모리 효율 모드)")
print("="*60)

models = {}
history = {}
evaluation_results = {}

# ============================================
# 1. LSTM 모델
# ============================================
print("\n1️⃣ LSTM 모델 학습")
gc.collect()

# 체크포인트 경로
lstm_checkpoint_path = f"{Config.MODEL_DIR}lstm_checkpoint.h5"

# 이전 체크포인트가 있으면 로드
if os.path.exists(lstm_checkpoint_path):
    print(f"  📂 이전 체크포인트 로드: {lstm_checkpoint_path}")
    lstm_model = tf.keras.models.load_model(
        lstm_checkpoint_path,
        custom_objects={'WeightedLoss': WeightedLoss}
    )
else:
    lstm_model = ModelsV6.build_lstm_model(input_shape)
    lstm_model.compile(
        optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE),
        loss=WeightedLoss(),
        metrics=['mae']
    )

# 제너레이터 생성
train_gen = MemoryEfficientGenerator(Config.SEQUENCE_FILE, train_indices, Config.BATCH_SIZE)
val_gen = MemoryEfficientGenerator(Config.SEQUENCE_FILE, val_indices, Config.BATCH_SIZE)

# 학습
lstm_history = lstm_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=Config.EPOCHS,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            lstm_checkpoint_path,
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(patience=Config.PATIENCE, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
    ],
    verbose=1
)

models['lstm'] = lstm_model
history['lstm'] = lstm_history
gc.collect()

# ============================================
# 2. GRU 모델
# ============================================
print("\n2️⃣ Enhanced GRU 모델 학습")
gc.collect()

gru_checkpoint_path = f"{Config.MODEL_DIR}gru_checkpoint.h5"

if os.path.exists(gru_checkpoint_path):
    print(f"  📂 이전 체크포인트 로드: {gru_checkpoint_path}")
    gru_model = tf.keras.models.load_model(
        gru_checkpoint_path,
        custom_objects={'WeightedLoss': WeightedLoss}
    )
else:
    gru_model = ModelsV6.build_enhanced_gru(input_shape)
    gru_model.compile(
        optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE),
        loss=WeightedLoss(),
        metrics=['mae']
    )

# 새 제너레이터 (인덱스 재셔플)
train_gen = MemoryEfficientGenerator(Config.SEQUENCE_FILE, train_indices, Config.BATCH_SIZE)
val_gen = MemoryEfficientGenerator(Config.SEQUENCE_FILE, val_indices, Config.BATCH_SIZE)

gru_history = gru_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=Config.EPOCHS,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            gru_checkpoint_path,
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(patience=Config.PATIENCE, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
    ],
    verbose=1
)

models['gru'] = gru_model
history['gru'] = gru_history
gc.collect()

# ============================================
# 3. CNN-LSTM 모델
# ============================================
print("\n3️⃣ CNN-LSTM 모델 학습")
gc.collect()

cnn_lstm_checkpoint_path = f"{Config.MODEL_DIR}cnn_lstm_checkpoint.h5"

if os.path.exists(cnn_lstm_checkpoint_path):
    print(f"  📂 이전 체크포인트 로드: {cnn_lstm_checkpoint_path}")
    cnn_lstm_model = tf.keras.models.load_model(
        cnn_lstm_checkpoint_path,
        custom_objects={'WeightedLoss': WeightedLoss}
    )
else:
    cnn_lstm_model = ModelsV6.build_cnn_lstm(input_shape)
    cnn_lstm_model.compile(
        optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE),
        loss=WeightedLoss(),
        metrics=['mae']
    )

train_gen = MemoryEfficientGenerator(Config.SEQUENCE_FILE, train_indices, Config.BATCH_SIZE)
val_gen = MemoryEfficientGenerator(Config.SEQUENCE_FILE, val_indices, Config.BATCH_SIZE)

cnn_lstm_history = cnn_lstm_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=Config.EPOCHS,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            cnn_lstm_checkpoint_path,
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(patience=Config.PATIENCE, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
    ],
    verbose=1
)

models['cnn_lstm'] = cnn_lstm_model
history['cnn_lstm'] = cnn_lstm_history
gc.collect()

# ============================================
# 4. Spike Detector 모델
# ============================================
print("\n4️⃣ Spike Detector 모델 학습 (1400+ 특화)")
gc.collect()

spike_checkpoint_path = f"{Config.MODEL_DIR}spike_checkpoint.h5"

if os.path.exists(spike_checkpoint_path):
    print(f"  📂 이전 체크포인트 로드: {spike_checkpoint_path}")
    spike_model = tf.keras.models.load_model(
        spike_checkpoint_path,
        custom_objects={'WeightedLoss': WeightedLoss}
    )
else:
    spike_model = ModelsV6.build_spike_detector(input_shape)
    spike_model.compile(
        optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE),
        loss={
            'spike_value': WeightedLoss(),
            'spike_prob': 'binary_crossentropy'
        },
        loss_weights={
            'spike_value': 1.0,
            'spike_prob': 0.5
        },
        metrics={
            'spike_value': 'mae',
            'spike_prob': 'accuracy'
        }
    )

# Spike용 제너레이터
train_gen_spike = SpikeGenerator(Config.SEQUENCE_FILE, train_indices, Config.BATCH_SIZE)
val_gen_spike = SpikeGenerator(Config.SEQUENCE_FILE, val_indices, Config.BATCH_SIZE)

spike_history = spike_model.fit(
    train_gen_spike,
    validation_data=val_gen_spike,
    epochs=Config.EPOCHS,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            spike_checkpoint_path,
            save_best_only=True,
            monitor='val_spike_value_loss',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(patience=Config.PATIENCE, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
    ],
    verbose=1
)

models['spike'] = spike_model
history['spike'] = spike_history
gc.collect()

# ============================================
# 5. 최종 앙상블 모델
# ============================================
print("\n5️⃣ 최종 앙상블 모델 구성")
gc.collect()

# 앙상블 모델 구성
time_series_input = tf.keras.Input(shape=input_shape, name='ensemble_input')
m14_input = tf.keras.Input(shape=(m14_shape,), name='m14_features')

# 각 모델 예측
lstm_pred = lstm_model(time_series_input)
gru_pred = gru_model(time_series_input)
cnn_lstm_pred = cnn_lstm_model(time_series_input)
spike_pred, spike_prob = spike_model(time_series_input)

# M14 기반 동적 가중치
weight_dense = tf.keras.layers.Dense(32, activation='relu')(m14_input)
weight_dense = tf.keras.layers.Dense(16, activation='relu')(weight_dense)
weights = tf.keras.layers.Dense(4, activation='softmax', name='ensemble_weights')(weight_dense)

# 가중 평균
w_lstm = tf.keras.layers.Lambda(lambda x: x[:, 0:1])(weights)
w_gru = tf.keras.layers.Lambda(lambda x: x[:, 1:2])(weights)
w_cnn = tf.keras.layers.Lambda(lambda x: x[:, 2:3])(weights)
w_spike = tf.keras.layers.Lambda(lambda x: x[:, 3:4])(weights)

weighted_lstm = tf.keras.layers.Multiply()([lstm_pred, w_lstm])
weighted_gru = tf.keras.layers.Multiply()([gru_pred, w_gru])
weighted_cnn = tf.keras.layers.Multiply()([cnn_lstm_pred, w_cnn])
weighted_spike = tf.keras.layers.Multiply()([spike_pred, w_spike])

# 최종 예측
ensemble_pred = tf.keras.layers.Add()([weighted_lstm, weighted_gru, weighted_cnn, weighted_spike])

# M14 규칙 보정
final_pred = M14RuleCorrection()([ensemble_pred, m14_input])

ensemble_model = tf.keras.Model(
    inputs=[time_series_input, m14_input],
    outputs=[final_pred, spike_prob],
    name='Final_Ensemble'
)

ensemble_model.compile(
    optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE * 0.5),
    loss={
        'm14_rule_correction': WeightedLoss(),
        'spike_prob': 'binary_crossentropy'
    },
    loss_weights={
        'm14_rule_correction': 1.0,
        'spike_prob': 0.3
    },
    metrics=['mae']
)

# 앙상블용 제너레이터
train_gen_ensemble = EnsembleGenerator(Config.SEQUENCE_FILE, train_indices, Config.BATCH_SIZE)
val_gen_ensemble = EnsembleGenerator(Config.SEQUENCE_FILE, val_indices, Config.BATCH_SIZE)

print("\n📊 앙상블 파인튜닝...")
ensemble_history = ensemble_model.fit(
    train_gen_ensemble,
    validation_data=val_gen_ensemble,
    epochs=20,
    verbose=1
)

models['ensemble'] = ensemble_model
history['ensemble'] = ensemble_history

print("\n✅ 모든 모델 학습 완료!")

# ============================================
# 7. 평가 (간단 버전 - 메모리 효율)
# ============================================
print("\n" + "="*60)
print("📊 모델 평가")
print("="*60)

# 평가용 작은 배치만 로드
with np.load(Config.SEQUENCE_FILE) as data:
    eval_size = min(5000, len(val_indices))
    eval_indices = val_indices[:eval_size]
    
    X_eval = data['X'][eval_indices].astype(np.float32)
    y_eval = data['y'][eval_indices]
    m14_eval = data['m14_features'][eval_indices]

for name, model in models.items():
    if name == 'ensemble':
        pred = model.predict([X_eval, m14_eval], verbose=0)[0].flatten()
    else:
        pred = model.predict(X_eval, verbose=0)
        if isinstance(pred, list):
            pred = pred[0]
        pred = pred.flatten()
    
    # 전체 성능
    mae = np.mean(np.abs(y_eval - pred))
    
    # 구간별 성능
    level_performance = {}
    for level in [1400, 1500, 1600, 1700]:
        mask = y_eval >= level
        if np.any(mask):
            recall = np.sum((pred >= level) & mask) / np.sum(mask)
            level_mae = np.mean(np.abs(y_eval[mask] - pred[mask]))
            level_performance[level] = {
                'recall': recall,
                'mae': level_mae,
                'count': np.sum(mask)
            }
    
    evaluation_results[name] = {
        'overall_mae': mae,
        'levels': level_performance
    }
    
    # 출력
    print(f"\n🎯 {name.upper()} 모델:")
    print(f"  전체 MAE: {mae:.2f}")
    for level, perf in level_performance.items():
        print(f"  {level}+: Recall={perf['recall']:.2%}, MAE={perf['mae']:.1f} (n={perf['count']})")

# 최종 선택
best_model = min(evaluation_results.keys(), key=lambda x: evaluation_results[x]['overall_mae'])
print(f"\n🏆 최고 성능: {best_model.upper()} 모델")

# ============================================
# 8. 모델 저장
# ============================================
print("\n💾 모델 저장 중...")

for name, model in models.items():
    model.save(f"{Config.MODEL_DIR}{name}_model.h5")
    print(f"  {name}_model.h5 저장 완료")

# 평가 결과 저장
with open(f"{Config.MODEL_DIR}evaluation_results.json", 'w') as f:
    json.dump(evaluation_results, f, indent=2)

# 설정 저장
config_dict = {k: v for k, v in Config.__dict__.items() if not k.startswith('_')}
with open(f"{Config.MODEL_DIR}config.json", 'w') as f:
    json.dump(config_dict, f, indent=2)

print("  결과 파일 저장 완료")

# ============================================
# 9. 간단 시각화
# ============================================
print("\n📈 결과 시각화 생성 중...")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

# 학습 곡선
for idx, (name, hist) in enumerate(history.items()):
    if idx < 5:
        ax = axes[idx]
        if hasattr(hist, 'history'):
            loss = hist.history.get('loss', hist.history.get('spike_value_loss', []))
            val_loss = hist.history.get('val_loss', hist.history.get('val_spike_value_loss', []))
            
            if loss and val_loss:
                ax.plot(loss, label='Train')
                ax.plot(val_loss, label='Val')
                ax.set_title(f'{name.upper()}')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.legend()
                ax.grid(True, alpha=0.3)

# MAE 비교
ax = axes[5]
model_names = list(evaluation_results.keys())
maes = [evaluation_results[m]['overall_mae'] for m in model_names]
ax.bar(model_names, maes)
ax.set_title('Model MAE Comparison')
ax.set_ylabel('MAE')

plt.suptitle('학습 V6 모델 성능 분석 (메모리 효율 모드)', fontsize=14)
plt.tight_layout()
plt.savefig(f"{Config.MODEL_DIR}training_results.png", dpi=100)
print("  training_results.png 저장 완료")
plt.show()

print("\n" + "="*60)
print("🎉 모든 작업 완료!")
print(f"📁 저장 위치: {Config.MODEL_DIR}")
print(f"📂 시퀀스 파일: {Config.SEQUENCE_FILE}")
print(f"💾 메모리 효율 모드로 BATCH_SIZE=32 학습 성공!")
print("="*60)

# 메모리 정리
gc.collect()
tf.keras.backend.clear_session()