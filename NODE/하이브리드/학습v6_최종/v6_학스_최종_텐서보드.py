"""
V6_학습_TensorBoard_수정완료.py - TensorBoard + 학습 재개 기능이 추가된 5개 모델 앙상블 학습
디렉토리 생성 문제 해결 버전
TensorFlow 2.15.0 + TensorBoard + Resume Training
"""

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json
import os
import warnings
from datetime import datetime
import pickle
import pathlib
warnings.filterwarnings('ignore')

print("="*60)
print("🚀 반도체 물류 예측 앙상블 학습 V6 - TensorBoard Edition")
print(f"📦 TensorFlow 버전: {tf.__version__}")
print("="*60)

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
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    PATIENCE = 15
    
    # 모델 저장 경로
    MODEL_DIR = './models_v6/'
    CHECKPOINT_DIR = './checkpoints_v6/'
    LOG_DIR = './logs/fit/'
    
    # 가중치 설정
    SPIKE_WEIGHTS = {
        'normal': 1.0,
        'level_1400': 3.0,
        'level_1500': 5.0,
        'level_1600': 8.0,
        'level_1700': 10.0
    }
    
    # 학습 재개 설정
    RESUME_TRAINING = True  # True로 설정하면 이전 학습 이어서 진행

# 디렉토리 생성 (Windows 호환)
for dir_path in [Config.MODEL_DIR, Config.CHECKPOINT_DIR, Config.LOG_DIR]:
    pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
    print(f"✅ 디렉토리 생성/확인: {dir_path}")

# ============================================
# 학습 상태 저장/로드 함수
# ============================================
def save_training_state(model_name, epoch, history_dict):
    """학습 상태 저장"""
    state = {
        'epoch': epoch,
        'history': history_dict
    }
    state_path = os.path.join(Config.CHECKPOINT_DIR, f"{model_name}_state.pkl")
    with open(state_path, 'wb') as f:
        pickle.dump(state, f)

def load_training_state(model_name):
    """학습 상태 로드"""
    state_file = os.path.join(Config.CHECKPOINT_DIR, f"{model_name}_state.pkl")
    if os.path.exists(state_file):
        with open(state_file, 'rb') as f:
            return pickle.load(f)
    return None

def get_initial_epoch(model_name):
    """시작 에폭 번호 가져오기"""
    if Config.RESUME_TRAINING:
        state = load_training_state(model_name)
        if state:
            return state['epoch']
    return 0

# ============================================
# 2. 커스텀 레이어 및 손실 함수
# ============================================
class WeightedLoss(tf.keras.losses.Loss):
    """물류량 구간별 가중치 손실 함수"""
    def call(self, y_true, y_pred):
        mae = tf.abs(y_true - y_pred)
        
        # 구간별 가중치
        weights = tf.ones_like(y_true)
        weights = tf.where(y_true >= 1700, 10.0, weights)
        weights = tf.where((y_true >= 1600) & (y_true < 1700), 8.0, weights)
        weights = tf.where((y_true >= 1500) & (y_true < 1600), 5.0, weights)
        weights = tf.where((y_true >= 1400) & (y_true < 1500), 3.0, weights)
        
        return tf.reduce_mean(mae * weights)

class M14RuleCorrection(tf.keras.layers.Layer):
    """M14 규칙 기반 보정 레이어"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def call(self, inputs):
        pred, m14_features = inputs
        
        # M14 특징 분해
        m14b = m14_features[:, 0:1]
        m10a = m14_features[:, 1:2]
        ratio = m14_features[:, 3:4]
        
        # 규칙 기반 보정
        # 1700+ 신호
        condition_1700 = tf.logical_and(
            tf.greater_equal(m14b, 500),
            tf.greater_equal(ratio, 7)
        )
        pred = tf.where(condition_1700, tf.maximum(pred, 1700), pred)
        
        # 1600+ 신호
        condition_1600 = tf.logical_and(
            tf.greater_equal(m14b, 450),
            tf.greater_equal(ratio, 6)
        )
        pred = tf.where(condition_1600, tf.maximum(pred, 1600), pred)
        
        # 1500+ 신호
        condition_1500 = tf.logical_and(
            tf.greater_equal(m14b, 400),
            tf.greater_equal(ratio, 5)
        )
        pred = tf.where(condition_1500, tf.maximum(pred, 1500), pred)
        
        # 1400+ 신호
        condition_1400 = tf.greater_equal(m14b, 320)
        pred = tf.where(condition_1400, tf.maximum(pred, 1400), pred)
        
        # 황금 패턴 보정
        golden_pattern = tf.logical_and(
            tf.greater_equal(m14b, 350),
            tf.less(m10a, 70)
        )
        pred = tf.where(golden_pattern, pred * 1.1, pred)
        
        return pred

class SpikePerformanceCallback(tf.keras.callbacks.Callback):
    """급증 감지 성능 모니터링 + TensorBoard 로깅"""
    def __init__(self, X_val, y_val, log_dir, model_name):
        self.X_val = X_val
        self.y_val = y_val
        self.writer = tf.summary.create_file_writer(log_dir)
        self.model_name = model_name
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            pred = self.model.predict(self.X_val, verbose=0)
            if isinstance(pred, list):
                pred = pred[0]
            pred = pred.flatten()
            
            # 구간별 Recall
            with self.writer.as_default():
                for level in [1400, 1500, 1600, 1700]:
                    mask = self.y_val >= level
                    if np.any(mask):
                        recall = np.sum((pred >= level) & mask) / np.sum(mask)
                        print(f"   {level} Recall: {recall:.2%}", end=" ")
                        tf.summary.scalar(f'{self.model_name}/recall_{level}+', recall, step=epoch)
            print()
            self.writer.flush()

class TrainingStateCallback(tf.keras.callbacks.Callback):
    """학습 상태 저장 콜백"""
    def __init__(self, model_name):
        self.model_name = model_name
        self.history_dict = {}
        
    def on_epoch_end(self, epoch, logs=None):
        # 히스토리 업데이트
        for key, value in logs.items():
            if key not in self.history_dict:
                self.history_dict[key] = []
            self.history_dict[key].append(value)
        
        # 상태 저장
        save_training_state(self.model_name, epoch + 1, self.history_dict)

# ============================================
# 3. 모델 정의
# ============================================
class ModelsV6:
    
    @staticmethod
    def build_lstm_model(input_shape):
        """1. LSTM 모델 - 장기 시계열 패턴 학습"""
        inputs = tf.keras.Input(shape=input_shape, name='lstm_input')
        
        # Stacked LSTM
        lstm1 = tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.2)(inputs)
        lstm2 = tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.2)(lstm1)
        lstm3 = tf.keras.layers.LSTM(64, dropout=0.2)(lstm2)
        
        # Dense layers
        dense1 = tf.keras.layers.Dense(128, activation='relu')(lstm3)
        dropout = tf.keras.layers.Dropout(0.3)(dense1)
        dense2 = tf.keras.layers.Dense(64, activation='relu')(dropout)
        
        # Output
        output = tf.keras.layers.Dense(1, name='lstm_output')(dense2)
        
        model = tf.keras.Model(inputs=inputs, outputs=output, name='LSTM_Model')
        return model
    
    @staticmethod
    def build_enhanced_gru(input_shape):
        """2. GRU 모델 - 단기 변동성 포착"""
        inputs = tf.keras.Input(shape=input_shape, name='gru_input')
        
        # Layer Normalization
        x = tf.keras.layers.LayerNormalization()(inputs)
        
        # Stacked GRU with residual
        gru1 = tf.keras.layers.GRU(128, return_sequences=True, dropout=0.2)(x)
        gru2 = tf.keras.layers.GRU(128, return_sequences=True, dropout=0.2)(gru1)
        
        # Residual connection
        residual = tf.keras.layers.Add()([gru1, gru2])
        
        # Final GRU
        gru3 = tf.keras.layers.GRU(64, dropout=0.2)(residual)
        
        # Dense layers
        dense1 = tf.keras.layers.Dense(128, activation='relu')(gru3)
        dropout = tf.keras.layers.Dropout(0.3)(dense1)
        dense2 = tf.keras.layers.Dense(64, activation='relu')(dropout)
        
        # Output
        output = tf.keras.layers.Dense(1, name='gru_output')(dense2)
        
        model = tf.keras.Model(inputs=inputs, outputs=output, name='GRU_Model')
        return model
    
    @staticmethod
    def build_cnn_lstm(input_shape):
        """3. CNN-LSTM 모델 - 복합 패턴 인식"""
        inputs = tf.keras.Input(shape=input_shape, name='cnn_input')
        
        # Multi-scale CNN
        conv1 = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
        conv2 = tf.keras.layers.Conv1D(64, 5, activation='relu', padding='same')(inputs)
        conv3 = tf.keras.layers.Conv1D(64, 7, activation='relu', padding='same')(inputs)
        
        # Concatenate
        concat = tf.keras.layers.Concatenate()([conv1, conv2, conv3])
        
        # Batch normalization
        norm = tf.keras.layers.BatchNormalization()(concat)
        
        # LSTM
        lstm = tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.2)(norm)
        lstm2 = tf.keras.layers.LSTM(64, dropout=0.2)(lstm)
        
        # Dense layers
        dense1 = tf.keras.layers.Dense(128, activation='relu')(lstm2)
        dropout = tf.keras.layers.Dropout(0.3)(dense1)
        
        # Output
        output = tf.keras.layers.Dense(1, name='cnn_lstm_output')(dropout)
        
        model = tf.keras.Model(inputs=inputs, outputs=output, name='CNN_LSTM_Model')
        return model
    
    @staticmethod
    def build_spike_detector(input_shape):
        """4. Spike Detector - 이상치 감지 전문"""
        inputs = tf.keras.Input(shape=input_shape, name='spike_input')
        
        # Multi-scale CNN for pattern detection
        conv1 = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
        conv2 = tf.keras.layers.Conv1D(64, 5, activation='relu', padding='same')(inputs)
        conv3 = tf.keras.layers.Conv1D(64, 7, activation='relu', padding='same')(inputs)
        
        # Concatenate multi-scale features
        concat = tf.keras.layers.Concatenate()([conv1, conv2, conv3])
        
        # Batch normalization
        norm = tf.keras.layers.BatchNormalization()(concat)
        
        # Attention mechanism
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=4, 
            key_dim=48,
            dropout=0.2
        )(norm, norm)
        
        # BiLSTM
        lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.2)
        )(attention)
        
        # Global pooling
        pooled = tf.keras.layers.GlobalAveragePooling1D()(lstm)
        
        # Dense layers
        dense1 = tf.keras.layers.Dense(256, activation='relu')(pooled)
        dropout1 = tf.keras.layers.Dropout(0.3)(dense1)
        dense2 = tf.keras.layers.Dense(128, activation='relu')(dropout1)
        dropout2 = tf.keras.layers.Dropout(0.2)(dense2)
        
        # Dual output (회귀 + 분류)
        regression_output = tf.keras.layers.Dense(1, name='spike_value')(dropout2)
        classification_output = tf.keras.layers.Dense(1, activation='sigmoid', name='spike_prob')(dropout2)
        
        model = tf.keras.Model(
            inputs=inputs,
            outputs=[regression_output, classification_output],
            name='Spike_Detector'
        )
        return model
    
    @staticmethod
    def build_rule_based_model(input_shape, m14_shape):
        """5. Rule-Based 모델 - 검증된 황금 패턴"""
        # 시계열 입력
        time_input = tf.keras.Input(shape=input_shape, name='time_input')
        # M14 특징 입력
        m14_input = tf.keras.Input(shape=m14_shape, name='m14_input')
        
        # 간단한 시계열 처리
        lstm = tf.keras.layers.LSTM(32, dropout=0.2)(time_input)
        
        # M14 특징 처리
        m14_dense = tf.keras.layers.Dense(16, activation='relu')(m14_input)
        
        # 결합
        combined = tf.keras.layers.Concatenate()([lstm, m14_dense])
        
        # Dense layers
        dense1 = tf.keras.layers.Dense(64, activation='relu')(combined)
        dropout = tf.keras.layers.Dropout(0.2)(dense1)
        dense2 = tf.keras.layers.Dense(32, activation='relu')(dropout)
        
        # 예측
        prediction = tf.keras.layers.Dense(1, name='rule_pred')(dense2)
        
        # M14 규칙 적용
        corrected = M14RuleCorrection()([prediction, m14_input])
        
        model = tf.keras.Model(
            inputs=[time_input, m14_input],
            outputs=corrected,
            name='Rule_Based_Model'
        )
        return model

# ============================================
# 4. 데이터 로드 및 준비
# ============================================
print("\n📂 시퀀스 로딩 중...")

# 시퀀스 로드
if not os.path.exists(Config.SEQUENCE_FILE):
    print(f"❌ 시퀀스 파일이 없습니다: {Config.SEQUENCE_FILE}")
    print("먼저 V6_시퀀스생성_최종본.py를 실행하세요.")
    exit()

data = np.load(Config.SEQUENCE_FILE)
X = data['X']
y = data['y']
m14_features = data['m14_features']

print(f"  ✅ 로드 완료!")
print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")
print(f"  m14_features shape: {m14_features.shape}")

# 학습/검증 분할
X_train, X_val, y_train, y_val, m14_train, m14_val = train_test_split(
    X, y, m14_features, test_size=0.2, random_state=42
)

# 1400+ 여부 레이블 생성
y_spike_class = (y_train >= 1400).astype(float)
y_val_spike_class = (y_val >= 1400).astype(float)

print(f"\n📊 데이터 분할:")
print(f"  학습: {X_train.shape[0]:,}개")
print(f"  검증: {X_val.shape[0]:,}개")
print(f"  1400+ 학습 비율: {y_spike_class.mean():.1%}")
print(f"  1400+ 검증 비율: {y_val_spike_class.mean():.1%}")

# ============================================
# 5. 학습 파이프라인
# ============================================
print("\n" + "="*60)
print("🏋️ 5개 모델 학습 시작")
print("📊 TensorBoard 로그 디렉토리:", Config.LOG_DIR)
print("⚡ 학습 재개 모드:", "ON" if Config.RESUME_TRAINING else "OFF")
print("="*60)

models = {}
history = {}
evaluation_results = {}

# ============================================
# 5.1 LSTM 모델
# ============================================
print("\n1️⃣ LSTM 모델 학습 (장기 시계열 패턴)")

# 모델 생성 또는 로드
lstm_model = ModelsV6.build_lstm_model(X_train.shape[1:])
checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "lstm_checkpoint.h5")

# 이전 가중치 로드 (있으면)
if Config.RESUME_TRAINING and os.path.exists(checkpoint_path):
    print("  ✅ 이전 체크포인트에서 가중치 로드")
    lstm_model.load_weights(checkpoint_path)

lstm_model.compile(
    optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE),
    loss=WeightedLoss(),
    metrics=['mae']
)

# TensorBoard 설정
log_dir = os.path.join(Config.LOG_DIR, "lstm", datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(log_dir, exist_ok=True)

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True,
    write_images=True,
    update_freq='epoch',
    profile_batch='500,520'
)

# 시작 에폭 결정
initial_epoch = get_initial_epoch('lstm')
print(f"  시작 에폭: {initial_epoch}")

lstm_history = lstm_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=Config.EPOCHS,
    initial_epoch=initial_epoch,
    batch_size=Config.BATCH_SIZE,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            save_weights_only=True,
            save_best_only=False,
            verbose=0
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(Config.MODEL_DIR, "lstm_best.h5"),
            save_best_only=True,
            monitor='val_loss',
            verbose=0
        ),
        tf.keras.callbacks.EarlyStopping(patience=Config.PATIENCE, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
        tensorboard_callback,
        SpikePerformanceCallback(X_val, y_val, log_dir, 'lstm'),
        TrainingStateCallback('lstm')
    ],
    verbose=1
)

models['lstm'] = lstm_model
history['lstm'] = lstm_history

# ============================================
# 5.2 GRU 모델
# ============================================
print("\n2️⃣ Enhanced GRU 모델 학습 (단기 변동성)")

gru_model = ModelsV6.build_enhanced_gru(X_train.shape[1:])
checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "gru_checkpoint.h5")

if Config.RESUME_TRAINING and os.path.exists(checkpoint_path):
    print("  ✅ 이전 체크포인트에서 가중치 로드")
    gru_model.load_weights(checkpoint_path)

gru_model.compile(
    optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE),
    loss=WeightedLoss(),
    metrics=['mae']
)

log_dir = os.path.join(Config.LOG_DIR, "gru", datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(log_dir, exist_ok=True)

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True,
    write_images=True,
    update_freq='epoch'
)

initial_epoch = get_initial_epoch('gru')
print(f"  시작 에폭: {initial_epoch}")

gru_history = gru_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=Config.EPOCHS,
    initial_epoch=initial_epoch,
    batch_size=Config.BATCH_SIZE,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            save_weights_only=True,
            save_best_only=False,
            verbose=0
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(Config.MODEL_DIR, "gru_best.h5"),
            save_best_only=True,
            monitor='val_loss',
            verbose=0
        ),
        tf.keras.callbacks.EarlyStopping(patience=Config.PATIENCE, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
        tensorboard_callback,
        SpikePerformanceCallback(X_val, y_val, log_dir, 'gru'),
        TrainingStateCallback('gru')
    ],
    verbose=1
)

models['gru'] = gru_model
history['gru'] = gru_history

# ============================================
# 5.3 CNN-LSTM 모델
# ============================================
print("\n3️⃣ CNN-LSTM 모델 학습 (복합 패턴 인식)")

cnn_lstm_model = ModelsV6.build_cnn_lstm(X_train.shape[1:])
checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "cnn_lstm_checkpoint.h5")

if Config.RESUME_TRAINING and os.path.exists(checkpoint_path):
    print("  ✅ 이전 체크포인트에서 가중치 로드")
    cnn_lstm_model.load_weights(checkpoint_path)

cnn_lstm_model.compile(
    optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE),
    loss=WeightedLoss(),
    metrics=['mae']
)

log_dir = os.path.join(Config.LOG_DIR, "cnn_lstm", datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(log_dir, exist_ok=True)

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True,
    write_images=True,
    update_freq='epoch'
)

initial_epoch = get_initial_epoch('cnn_lstm')
print(f"  시작 에폭: {initial_epoch}")

cnn_lstm_history = cnn_lstm_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=Config.EPOCHS,
    initial_epoch=initial_epoch,
    batch_size=Config.BATCH_SIZE,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            save_weights_only=True,
            save_best_only=False,
            verbose=0
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(Config.MODEL_DIR, "cnn_lstm_best.h5"),
            save_best_only=True,
            monitor='val_loss',
            verbose=0
        ),
        tf.keras.callbacks.EarlyStopping(patience=Config.PATIENCE, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
        tensorboard_callback,
        SpikePerformanceCallback(X_val, y_val, log_dir, 'cnn_lstm'),
        TrainingStateCallback('cnn_lstm')
    ],
    verbose=1
)

models['cnn_lstm'] = cnn_lstm_model
history['cnn_lstm'] = cnn_lstm_history

# ============================================
# 5.4 Spike Detector 모델
# ============================================
print("\n4️⃣ Spike Detector 모델 학습 (이상치 감지)")

spike_model = ModelsV6.build_spike_detector(X_train.shape[1:])
checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "spike_checkpoint.h5")

if Config.RESUME_TRAINING and os.path.exists(checkpoint_path):
    print("  ✅ 이전 체크포인트에서 가중치 로드")
    spike_model.load_weights(checkpoint_path)

spike_model.compile(
    optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE),
    loss={
        'spike_value': WeightedLoss(),
        'spike_prob': 'binary_crossentropy'
    },
    loss_weights={
        'spike_value': 1.0,
        'spike_prob': 0.3
    },
    metrics=['mae']
)

log_dir = os.path.join(Config.LOG_DIR, "spike", datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(log_dir, exist_ok=True)

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True,
    write_images=True,
    update_freq='epoch'
)

initial_epoch = get_initial_epoch('spike')
print(f"  시작 에폭: {initial_epoch}")

spike_history = spike_model.fit(
    X_train, 
    [y_train, y_spike_class],
    validation_data=(X_val, [y_val, y_val_spike_class]),
    epochs=Config.EPOCHS,
    initial_epoch=initial_epoch,
    batch_size=Config.BATCH_SIZE,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            save_weights_only=True,
            save_best_only=False,
            verbose=0
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(Config.MODEL_DIR, "spike_best.h5"),
            save_best_only=True,
            monitor='val_spike_value_loss',
            verbose=0
        ),
        tf.keras.callbacks.EarlyStopping(patience=Config.PATIENCE, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
        tensorboard_callback,
        TrainingStateCallback('spike')
    ],
    verbose=1
)

models['spike'] = spike_model
history['spike'] = spike_history

# ============================================
# 5.5 Rule-Based 모델
# ============================================
print("\n5️⃣ Rule-Based 모델 학습 (검증된 황금 패턴)")

rule_model = ModelsV6.build_rule_based_model(X_train.shape[1:], m14_train.shape[1])
checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "rule_checkpoint.h5")

if Config.RESUME_TRAINING and os.path.exists(checkpoint_path):
    print("  ✅ 이전 체크포인트에서 가중치 로드")
    rule_model.load_weights(checkpoint_path)

rule_model.compile(
    optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE * 0.5),
    loss=WeightedLoss(),
    metrics=['mae']
)

log_dir = os.path.join(Config.LOG_DIR, "rule", datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(log_dir, exist_ok=True)

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True,
    write_images=True,
    update_freq='epoch'
)

initial_epoch = get_initial_epoch('rule')
print(f"  시작 에폭: {initial_epoch}")

rule_history = rule_model.fit(
    [X_train, m14_train], 
    y_train,
    validation_data=([X_val, m14_val], y_val),
    epochs=50,  # Rule-based는 빠르게 수렴
    initial_epoch=initial_epoch,
    batch_size=Config.BATCH_SIZE,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            save_weights_only=True,
            save_best_only=False,
            verbose=0
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(Config.MODEL_DIR, "rule_best.h5"),
            save_best_only=True,
            monitor='val_loss',
            verbose=0
        ),
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tensorboard_callback,
        TrainingStateCallback('rule')
    ],
    verbose=1
)

models['rule'] = rule_model
history['rule'] = rule_history

# ============================================
# 6. 최종 앙상블 모델
# ============================================
print("\n" + "="*60)
print("🎯 최종 앙상블 모델 구성")
print("="*60)

# 입력 정의
time_series_input = tf.keras.Input(shape=X_train.shape[1:], name='ensemble_time_input')
m14_input = tf.keras.Input(shape=m14_train.shape[1], name='ensemble_m14_input')

# 각 모델 예측
lstm_pred = models['lstm'](time_series_input)
gru_pred = models['gru'](time_series_input)
cnn_lstm_pred = models['cnn_lstm'](time_series_input)
spike_pred, spike_prob = models['spike'](time_series_input)
rule_pred = models['rule']([time_series_input, m14_input])

# M14 기반 동적 가중치 생성
weight_dense = tf.keras.layers.Dense(32, activation='relu')(m14_input)
weight_dense = tf.keras.layers.Dense(16, activation='relu')(weight_dense)
weights = tf.keras.layers.Dense(5, activation='softmax', name='ensemble_weights')(weight_dense)

# 가중치 분리
w_lstm = tf.keras.layers.Lambda(lambda x: x[:, 0:1])(weights)
w_gru = tf.keras.layers.Lambda(lambda x: x[:, 1:2])(weights)
w_cnn = tf.keras.layers.Lambda(lambda x: x[:, 2:3])(weights)
w_spike = tf.keras.layers.Lambda(lambda x: x[:, 3:4])(weights)
w_rule = tf.keras.layers.Lambda(lambda x: x[:, 4:5])(weights)

# 가중 평균
weighted_lstm = tf.keras.layers.Multiply()([lstm_pred, w_lstm])
weighted_gru = tf.keras.layers.Multiply()([gru_pred, w_gru])
weighted_cnn = tf.keras.layers.Multiply()([cnn_lstm_pred, w_cnn])
weighted_spike = tf.keras.layers.Multiply()([spike_pred, w_spike])
weighted_rule = tf.keras.layers.Multiply()([rule_pred, w_rule])

# 앙상블 예측
ensemble_pred = tf.keras.layers.Add()([
    weighted_lstm, weighted_gru, weighted_cnn, 
    weighted_spike, weighted_rule
])

# 최종 M14 규칙 보정
final_pred = M14RuleCorrection()([ensemble_pred, m14_input])

# 앙상블 모델 정의
ensemble_model = tf.keras.Model(
    inputs=[time_series_input, m14_input],
    outputs=[final_pred, spike_prob],
    name='Final_Ensemble_Model'
)

# 앙상블 체크포인트
ensemble_checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "ensemble_checkpoint.h5")

if Config.RESUME_TRAINING and os.path.exists(ensemble_checkpoint_path):
    print("  ✅ 이전 앙상블 체크포인트에서 가중치 로드")
    ensemble_model.load_weights(ensemble_checkpoint_path)

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

log_dir = os.path.join(Config.LOG_DIR, "ensemble", datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(log_dir, exist_ok=True)

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True,
    write_images=True,
    update_freq='epoch'
)

initial_epoch = get_initial_epoch('ensemble')
print(f"  앙상블 시작 에폭: {initial_epoch}")

print("\n📊 앙상블 파인튜닝...")
ensemble_history = ensemble_model.fit(
    [X_train, m14_train],
    [y_train, y_spike_class],
    validation_data=(
        [X_val, m14_val],
        [y_val, y_val_spike_class]
    ),
    epochs=20,
    initial_epoch=initial_epoch,
    batch_size=Config.BATCH_SIZE,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            ensemble_checkpoint_path,
            save_weights_only=True,
            save_best_only=False,
            verbose=0
        ),
        tensorboard_callback,
        TrainingStateCallback('ensemble')
    ],
    verbose=1
)

models['ensemble'] = ensemble_model
history['ensemble'] = ensemble_history

print("\n✅ 5개 모델 + 앙상블 학습 완료!")

# ============================================
# 7. 평가
# ============================================
print("\n" + "="*60)
print("📊 모델 평가")
print("="*60)

for name, model in models.items():
    if name == 'ensemble':
        pred = model.predict([X_val, m14_val], verbose=0)[0].flatten()
    elif name == 'spike':
        pred = model.predict(X_val, verbose=0)[0].flatten()
    elif name == 'rule':
        pred = model.predict([X_val, m14_val], verbose=0).flatten()
    else:
        pred = model.predict(X_val, verbose=0).flatten()
    
    # 전체 성능
    mae = np.mean(np.abs(y_val - pred))
    
    # 구간별 성능
    level_performance = {}
    for level in [1400, 1500, 1600, 1700]:
        mask = y_val >= level
        if np.any(mask):
            recall = np.sum((pred >= level) & mask) / np.sum(mask)
            level_mae = np.mean(np.abs(y_val[mask] - pred[mask]))
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
print(f"  MAE: {evaluation_results[best_model]['overall_mae']:.2f}")

# ============================================
# 8. 모델 저장
# ============================================
print("\n💾 최종 모델 저장 중...")

for name, model in models.items():
    model_path = os.path.join(Config.MODEL_DIR, f"{name}_model.h5")
    model.save(model_path)
    print(f"  {name}_model.h5 저장 완료")

# 평가 결과 저장
results_path = os.path.join(Config.MODEL_DIR, "evaluation_results.json")
with open(results_path, 'w') as f:
    json.dump(evaluation_results, f, indent=2, default=str)

# 설정 저장
config_dict = {k: v for k, v in Config.__dict__.items() if not k.startswith('_')}
config_path = os.path.join(Config.MODEL_DIR, "config.json")
with open(config_path, 'w') as f:
    json.dump(config_dict, f, indent=2)

print("  결과 파일 저장 완료")

# ============================================
# 9. 시각화
# ============================================
print("\n📈 결과 시각화 생성 중...")

# 시각화 코드는 기존과 동일하지만 경로 수정
fig = plt.figure(figsize=(20, 12))

# 1-5. 각 모델 학습 곡선
for idx, (name, hist) in enumerate(history.items()):
    if idx < 5:  # 개별 모델들
        ax = plt.subplot(3, 4, idx+1)
        
        if hasattr(hist, 'history'):
            if name == 'spike':
                loss = hist.history.get('spike_value_loss', [])
                val_loss = hist.history.get('val_spike_value_loss', [])
            else:
                loss = hist.history.get('loss', [])
                val_loss = hist.history.get('val_loss', [])
            
            if loss and val_loss:
                ax.plot(loss, label='Train Loss', alpha=0.8)
                ax.plot(val_loss, label='Val Loss', alpha=0.8)
        
        ax.set_title(f'{name.upper()} Learning Curve')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

# 6. 앙상블 학습 곡선
ax = plt.subplot(3, 4, 6)
if 'ensemble' in history and hasattr(history['ensemble'], 'history'):
    loss = history['ensemble'].history.get('m14_rule_correction_loss', [])
    val_loss = history['ensemble'].history.get('val_m14_rule_correction_loss', [])
    if loss and val_loss:
        ax.plot(loss, label='Train Loss', alpha=0.8)
        ax.plot(val_loss, label='Val Loss', alpha=0.8)
ax.set_title('ENSEMBLE Learning Curve')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
ax.grid(True, alpha=0.3)

# 7. 모델별 MAE 비교
ax = plt.subplot(3, 4, 7)
model_names = list(evaluation_results.keys())
maes = [evaluation_results[m]['overall_mae'] for m in model_names]
colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown']

bars = ax.bar(model_names, maes, color=colors[:len(model_names)])
ax.set_title('Model MAE Comparison')
ax.set_ylabel('MAE')
ax.set_ylim(0, max(maes) * 1.2)

for bar, mae in zip(bars, maes):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{mae:.1f}', ha='center', va='bottom')

# 8-11. Recall 비교
for idx, level in enumerate([1400, 1500, 1600, 1700]):
    ax = plt.subplot(3, 4, 8+idx)
    recalls = []
    for m in model_names:
        if level in evaluation_results[m]['levels']:
            recalls.append(evaluation_results[m]['levels'][level]['recall'] * 100)
        else:
            recalls.append(0)
    
    bars = ax.bar(model_names, recalls, color=colors[:len(model_names)])
    ax.set_title(f'{level}+ Recall Comparison (%)')
    ax.set_ylabel('Recall (%)')
    ax.set_ylim(0, 105)
    
    for bar, recall in zip(bars, recalls):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{recall:.1f}%', ha='center', va='bottom')

# 12. 성능 요약
ax = plt.subplot(3, 4, 12)
ax.axis('off')

summary_text = "🏆 Performance Summary\n" + "="*35 + "\n"
summary_text += f"Best Model: {best_model.upper()}\n"
summary_text += f"Overall MAE: {evaluation_results[best_model]['overall_mae']:.2f}\n\n"

summary_text += "Recall by Level:\n"
for level in [1400, 1500, 1600, 1700]:
    if level in evaluation_results[best_model]['levels']:
        recall = evaluation_results[best_model]['levels'][level]['recall']
        mae = evaluation_results[best_model]['levels'][level]['mae']
        summary_text += f"  {level}+: {recall:6.1%} (MAE: {mae:.1f})\n"

summary_text += f"\n5-Model Ensemble Complete!"

ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
       fontsize=11, verticalalignment='top', fontfamily='monospace')

plt.suptitle('V6 Ensemble Model Performance Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()

# 저장 경로
plot_path = os.path.join(Config.MODEL_DIR, "training_results.png")
plt.savefig(plot_path, dpi=100, bbox_inches='tight')
print(f"  training_results.png 저장 완료: {plot_path}")
plt.show()

# ============================================
# 10. 최종 출력
# ============================================
print("\n" + "="*60)
print("🎉 모든 작업 완료!")
print("="*60)
print(f"📁 모델 저장 위치: {Config.MODEL_DIR}")
print(f"📂 시퀀스 파일: {Config.SEQUENCE_FILE}")
print(f"📊 체크포인트 위치: {Config.CHECKPOINT_DIR}")
print(f"📈 TensorBoard 로그: {Config.LOG_DIR}")
print("\n📊 최종 성능:")
print(f"  최고 모델: {best_model.upper()}")
print(f"  전체 MAE: {evaluation_results[best_model]['overall_mae']:.2f}")
print("\n💡 TensorBoard 실행 명령어:")
print(f"  tensorboard --logdir={Config.LOG_DIR}")
print(f"  브라우저에서 http://localhost:6006 접속")
print("\n💡 다음 단계: 실시간 예측 시스템 적용")
print("="*60)

# GPU 정보 출력
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print(f"\n🎮 GPU 사용: {len(gpus)}개")
    for gpu in gpus:
        print(f"  {gpu}")
else:
    print("\n💻 CPU 모드로 실행됨")