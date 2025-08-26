"""
V6_학습_개선판_수정.py - 5개 모델 앙상블 학습 (수정 완료 버전)
미리 생성된 시퀀스를 로드하여 LSTM, GRU, CNN-LSTM, Spike Detector, Rule-Based 학습
TensorFlow 2.15.0
"""

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import json
import os
import warnings
import pickle
from datetime import datetime
warnings.filterwarnings('ignore')

print("="*60)
print("🚀 반도체 물류 예측 앙상블 학습 V6 (수정 완료)")
print(f"📦 TensorFlow 버전: {tf.__version__}")
print("="*60)

# ============================================
# 1. 개선된 설정
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
    
    # 수정된 학습 설정
    BATCH_SIZE = 128  # 증가: 더 빠른 학습
    EPOCHS = 100  # 적절한 에폭 수
    LEARNING_RATE = 0.0005  # 적절한 학습률
    PATIENCE = 25  # 적절한 인내심
    MIN_DELTA = 0.0001  # 최소 개선 폭
    
    # 모델 저장 경로
    MODEL_DIR = './models_v6_improved/'
    CHECKPOINT_DIR = './checkpoints_v6_improved/'
    
    # 정규화 설정 강화
    L1_REG = 0.001
    L2_REG = 0.01
    
    # 드롭아웃 비율 조정
    DROPOUT_RATES = {
        'lstm': 0.4,
        'gru': 0.4,
        'cnn': 0.3,
        'dense': 0.3
    }
    
    # 가중치 설정
    SPIKE_WEIGHTS = {
        'normal': 1.0,
        'level_1400': 3.0,  # 적절히 조정
        'level_1500': 5.0,  
        'level_1600': 8.0,  
        'level_1700': 10.0  
    }

# 디렉토리 생성
os.makedirs(Config.MODEL_DIR, exist_ok=True)
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

# ============================================
# 2. 데이터 증강 클래스 (수정)
# ============================================
class DataAugmentation:
    """데이터 증강을 통한 불균형 해결"""
    
    @staticmethod
    def augment_sequences(X, y, m14_features, target_levels=[1400, 1500, 1600, 1700]):
        """급증 구간 데이터 증강 (보수적으로 수정)"""
        augmented_X = []
        augmented_y = []
        augmented_m14 = []
        
        for level in target_levels:
            # 각 레벨 이상의 데이터 찾기
            if level == 1700:
                indices = np.where(y >= level)[0]
            else:
                next_level = target_levels[target_levels.index(level) + 1] if level != 1700 else float('inf')
                indices = np.where((y >= level) & (y < next_level))[0]
            
            if len(indices) == 0:
                continue
                
            print(f"  {level}+ 구간: {len(indices)}개 데이터")
            
            # 수정된 증강 비율 (보수적)
            if level >= 1700:
                augment_factor = 2  # 10 → 2
            elif level >= 1600:
                augment_factor = 2  # 5 → 2
            elif level >= 1500:
                augment_factor = 1  # 3 → 1 (증강 안함)
            else:
                augment_factor = 1  # 2 → 1 (증강 안함)
            
            for _ in range(augment_factor - 1):
                for idx in indices:
                    # 시계열 노이즈 추가
                    noise_scale = 0.01  # 0.02 → 0.01 (노이즈 감소)
                    noise = np.random.normal(0, noise_scale, X[idx].shape)
                    
                    # 시간 시프트 (±1 timesteps)
                    shift = np.random.randint(-1, 2)  # ±2 → ±1
                    if shift != 0:
                        shifted_x = np.roll(X[idx], shift, axis=0)
                    else:
                        shifted_x = X[idx]
                    
                    augmented_X.append(shifted_x + noise)
                    
                    # 타겟 변동 (작은 범위)
                    y_variation = np.random.uniform(0.99, 1.01)  # 0.98-1.02 → 0.99-1.01
                    augmented_y.append(y[idx] * y_variation)
                    
                    # M14 특징 변동
                    m14_variation = np.random.uniform(0.99, 1.01, m14_features[idx].shape)
                    augmented_m14.append(m14_features[idx] * m14_variation)
        
        if augmented_X:
            return (np.concatenate([X, np.array(augmented_X)]),
                   np.concatenate([y, np.array(augmented_y)]),
                   np.concatenate([m14_features, np.array(augmented_m14)]))
        
        return X, y, m14_features

# ============================================
# 3. 개선된 손실 함수
# ============================================
class ImprovedWeightedLoss(tf.keras.losses.Loss):
    """더 강한 가중치를 가진 손실 함수"""
    def __init__(self, spike_threshold=1400):
        super().__init__()
        self.spike_threshold = spike_threshold
        
    def call(self, y_true, y_pred):
        mae = tf.abs(y_true - y_pred)
        
        # 적절한 가중치
        weights = tf.ones_like(y_true)
        weights = tf.where(y_true >= 1700, 10.0, weights)
        weights = tf.where((y_true >= 1600) & (y_true < 1700), 8.0, weights)
        weights = tf.where((y_true >= 1500) & (y_true < 1600), 5.0, weights)
        weights = tf.where((y_true >= 1400) & (y_true < 1500), 3.0, weights)
        
        # 예측 오차가 큰 경우 추가 페널티
        large_error_penalty = tf.where(mae > 100, mae * 0.1, 0.0)
        
        # 급증 놓친 경우 추가 페널티
        missed_spike_penalty = tf.where(
            (y_true >= self.spike_threshold) & (y_pred < self.spike_threshold),
            (y_true - self.spike_threshold) * 0.5,
            0.0
        )
        
        return tf.reduce_mean(mae * weights + large_error_penalty + missed_spike_penalty)

# ============================================
# 4. 개선된 모델 구조
# ============================================
class ImprovedModels:
    
    @staticmethod
    def build_lstm_model(input_shape):
        """개선된 LSTM 모델"""
        inputs = tf.keras.Input(shape=input_shape, name='lstm_input')
        
        # 입력 정규화
        x = tf.keras.layers.LayerNormalization()(inputs)
        
        # 첫 번째 LSTM 블록
        lstm1 = tf.keras.layers.LSTM(
            256, 
            return_sequences=True,
            kernel_regularizer=tf.keras.regularizers.l1_l2(Config.L1_REG, Config.L2_REG),
            recurrent_regularizer=tf.keras.regularizers.l2(Config.L2_REG),
            dropout=Config.DROPOUT_RATES['lstm'],
            recurrent_dropout=Config.DROPOUT_RATES['lstm']
        )(x)
        lstm1 = tf.keras.layers.LayerNormalization()(lstm1)
        
        # 두 번째 LSTM 블록
        lstm2 = tf.keras.layers.LSTM(
            128, 
            return_sequences=True,
            kernel_regularizer=tf.keras.regularizers.l2(Config.L2_REG),
            dropout=Config.DROPOUT_RATES['lstm'],
            recurrent_dropout=Config.DROPOUT_RATES['lstm']
        )(lstm1)
        lstm2 = tf.keras.layers.LayerNormalization()(lstm2)
        
        # Attention 메커니즘
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=4, 
            key_dim=32,
            dropout=Config.DROPOUT_RATES['lstm']
        )(lstm2, lstm2)
        
        # 세 번째 LSTM
        lstm3 = tf.keras.layers.LSTM(
            64,
            dropout=Config.DROPOUT_RATES['lstm']
        )(attention)
        
        # Dense layers
        dense1 = tf.keras.layers.Dense(128, activation='relu')(lstm3)
        dense1 = tf.keras.layers.LayerNormalization()(dense1)
        dropout1 = tf.keras.layers.Dropout(Config.DROPOUT_RATES['dense'])(dense1)
        
        dense2 = tf.keras.layers.Dense(64, activation='relu')(dropout1)
        dropout2 = tf.keras.layers.Dropout(Config.DROPOUT_RATES['dense'])(dense2)
        
        # Output
        output = tf.keras.layers.Dense(1, name='lstm_output')(dropout2)
        
        model = tf.keras.Model(inputs=inputs, outputs=output, name='Improved_LSTM')
        return model
    
    @staticmethod
    def build_gru_model(input_shape):
        """개선된 GRU 모델"""
        inputs = tf.keras.Input(shape=input_shape, name='gru_input')
        
        # 입력 정규화
        x = tf.keras.layers.LayerNormalization()(inputs)
        
        # Bidirectional GRU
        gru1 = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                128, 
                return_sequences=True,
                kernel_regularizer=tf.keras.regularizers.l1_l2(Config.L1_REG, Config.L2_REG),
                dropout=Config.DROPOUT_RATES['gru'],
                recurrent_dropout=Config.DROPOUT_RATES['gru']
            )
        )(x)
        gru1 = tf.keras.layers.LayerNormalization()(gru1)
        
        # 두 번째 GRU
        gru2 = tf.keras.layers.GRU(
            128, 
            return_sequences=True,
            kernel_regularizer=tf.keras.regularizers.l2(Config.L2_REG),
            dropout=Config.DROPOUT_RATES['gru']
        )(gru1)
        
        # Global pooling
        avg_pool = tf.keras.layers.GlobalAveragePooling1D()(gru2)
        max_pool = tf.keras.layers.GlobalMaxPooling1D()(gru2)
        concatenated = tf.keras.layers.Concatenate()([avg_pool, max_pool])
        
        # Dense layers
        dense1 = tf.keras.layers.Dense(128, activation='relu')(concatenated)
        dense1 = tf.keras.layers.LayerNormalization()(dense1)
        dropout1 = tf.keras.layers.Dropout(Config.DROPOUT_RATES['dense'])(dense1)
        
        dense2 = tf.keras.layers.Dense(64, activation='relu')(dropout1)
        dropout2 = tf.keras.layers.Dropout(Config.DROPOUT_RATES['dense'])(dense2)
        
        # Output
        output = tf.keras.layers.Dense(1, name='gru_output')(dropout2)
        
        model = tf.keras.Model(inputs=inputs, outputs=output, name='Improved_GRU')
        return model
    
    @staticmethod
    def build_cnn_lstm(input_shape):
        """개선된 CNN-LSTM 모델"""
        inputs = tf.keras.Input(shape=input_shape, name='cnn_input')
        
        # 입력 정규화
        x = tf.keras.layers.LayerNormalization()(inputs)
        
        # Multi-scale CNN with residual connections
        conv1 = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(x)
        conv1 = tf.keras.layers.BatchNormalization()(conv1)
        
        conv2 = tf.keras.layers.Conv1D(64, 5, activation='relu', padding='same')(x)
        conv2 = tf.keras.layers.BatchNormalization()(conv2)
        
        conv3 = tf.keras.layers.Conv1D(64, 7, activation='relu', padding='same')(x)
        conv3 = tf.keras.layers.BatchNormalization()(conv3)
        
        # Concatenate multi-scale features
        concat = tf.keras.layers.Concatenate()([conv1, conv2, conv3])
        concat = tf.keras.layers.Dropout(Config.DROPOUT_RATES['cnn'])(concat)
        
        # Second CNN layer
        conv4 = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same')(concat)
        conv4 = tf.keras.layers.BatchNormalization()(conv4)
        conv4 = tf.keras.layers.Dropout(Config.DROPOUT_RATES['cnn'])(conv4)
        
        # LSTM layers
        lstm1 = tf.keras.layers.LSTM(128, return_sequences=True, dropout=Config.DROPOUT_RATES['lstm'])(conv4)
        lstm1 = tf.keras.layers.LayerNormalization()(lstm1)
        
        lstm2 = tf.keras.layers.LSTM(64, dropout=Config.DROPOUT_RATES['lstm'])(lstm1)
        
        # Dense layers
        dense1 = tf.keras.layers.Dense(128, activation='relu')(lstm2)
        dense1 = tf.keras.layers.LayerNormalization()(dense1)
        dropout1 = tf.keras.layers.Dropout(Config.DROPOUT_RATES['dense'])(dense1)
        
        dense2 = tf.keras.layers.Dense(64, activation='relu')(dropout1)
        dropout2 = tf.keras.layers.Dropout(Config.DROPOUT_RATES['dense'])(dense2)
        
        # Output
        output = tf.keras.layers.Dense(1, name='cnn_lstm_output')(dropout2)
        
        model = tf.keras.Model(inputs=inputs, outputs=output, name='Improved_CNN_LSTM')
        return model
    
    @staticmethod
    def build_spike_detector(input_shape):
        """개선된 Spike Detector"""
        inputs = tf.keras.Input(shape=input_shape, name='spike_input')
        
        # 입력 정규화
        x = tf.keras.layers.LayerNormalization()(inputs)
        
        # Feature extraction branch
        # CNN branch
        conv1 = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same')(x)
        conv1 = tf.keras.layers.BatchNormalization()(conv1)
        conv2 = tf.keras.layers.Conv1D(128, 5, activation='relu', padding='same')(conv1)
        conv2 = tf.keras.layers.BatchNormalization()(conv2)
        pool1 = tf.keras.layers.MaxPooling1D(2)(conv2)
        
        # BiLSTM branch  
        lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128, return_sequences=True, dropout=Config.DROPOUT_RATES['lstm'])
        )(x)
        lstm = tf.keras.layers.LayerNormalization()(lstm)
        
        # Combine branches
        lstm_pooled = tf.keras.layers.MaxPooling1D(2)(lstm)
        combined = tf.keras.layers.Concatenate()([pool1, lstm_pooled])
        
        # Attention
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=8, 
            key_dim=32,
            dropout=Config.DROPOUT_RATES['lstm']
        )(combined, combined)
        
        # Global pooling
        avg_pool = tf.keras.layers.GlobalAveragePooling1D()(attention)
        max_pool = tf.keras.layers.GlobalMaxPooling1D()(attention)
        concat_pool = tf.keras.layers.Concatenate()([avg_pool, max_pool])
        
        # Dense layers
        dense1 = tf.keras.layers.Dense(256, activation='relu')(concat_pool)
        dense1 = tf.keras.layers.LayerNormalization()(dense1)
        dropout1 = tf.keras.layers.Dropout(Config.DROPOUT_RATES['dense'])(dense1)
        
        dense2 = tf.keras.layers.Dense(128, activation='relu')(dropout1)
        dropout2 = tf.keras.layers.Dropout(Config.DROPOUT_RATES['dense'])(dense2)
        
        # Dual output
        regression_output = tf.keras.layers.Dense(1, name='spike_value')(dropout2)
        classification_output = tf.keras.layers.Dense(1, activation='sigmoid', name='spike_prob')(dropout2)
        
        model = tf.keras.Model(
            inputs=inputs,
            outputs=[regression_output, classification_output],
            name='Improved_Spike_Detector'
        )
        return model

# ============================================
# 5. 개선된 콜백 (수정)
# ============================================
class ImprovedCallbacks:
    
    @staticmethod
    def get_callbacks(model_name, X_val, y_val):
        """개선된 콜백 세트 (학습률 스케줄러 충돌 수정)"""
        # 로그 디렉토리 생성
        log_dir = f'./logs/{model_name}_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        os.makedirs(log_dir, exist_ok=True)
        
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                f"{Config.MODEL_DIR}{model_name}_best.h5",
                save_best_only=True,
                monitor='val_loss',
                mode='min',
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=Config.PATIENCE,
                restore_best_weights=True,
                min_delta=Config.MIN_DELTA,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,  # 15 → 10
                min_lr=1e-7,
                verbose=1
            ),
            # LearningRateScheduler 제거 (충돌 방지)
            SpikePerformanceCallback(X_val, y_val)
        ]
        
        return callbacks

# SpikePerformanceCallback 유지
class SpikePerformanceCallback(tf.keras.callbacks.Callback):
    """급증 감지 성능 모니터링"""
    def __init__(self, X_val, y_val):
        self.X_val = X_val
        self.y_val = y_val
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            pred = self.model.predict(self.X_val[:1000], verbose=0)  # 샘플링으로 속도 개선
            if isinstance(pred, list):
                pred = pred[0]
            pred = pred.flatten()
            
            # 구간별 Recall과 Precision
            print("\n", end="")
            for level in [1400, 1500, 1600, 1700]:
                mask = self.y_val[:1000] >= level
                if np.any(mask):
                    pred_mask = pred >= level
                    
                    tp = np.sum((pred_mask) & (mask))
                    fp = np.sum((pred_mask) & (~mask))
                    fn = np.sum((~pred_mask) & (mask))
                    
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    
                    print(f"   {level}: R={recall:.2%} P={precision:.2%}", end="")
            print()

# ============================================
# M14RuleCorrection 레이어
# ============================================
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

# ============================================
# 6. 메인 실행 부분
# ============================================
print("\n📂 시퀀스 로딩 중...")

# 시퀀스 로드
data = np.load(Config.SEQUENCE_FILE)
X = data['X']
y = data['y']
m14_features = data['m14_features']

print(f"  ✅ 로드 완료!")
print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")
print(f"  m14_features shape: {m14_features.shape}")

# 타겟값 0 제거
non_zero_idx = np.where(y > 0)[0]
X = X[non_zero_idx]
y = y[non_zero_idx]
m14_features = m14_features[non_zero_idx]

print(f"\n  타겟값 0 제거 후: {len(X)}개")

# 학습/검증 분할
X_train, X_val, y_train, y_val, m14_train, m14_val = train_test_split(
    X, y, m14_features, test_size=0.2, random_state=42, stratify=(y >= 1400)
)

print(f"\n📊 원본 데이터 분포:")
for level in [1400, 1500, 1600, 1700]:
    train_count = np.sum(y_train >= level)
    val_count = np.sum(y_val >= level)
    print(f"  {level}+: 학습 {train_count}개 ({train_count/len(y_train):.1%}), "
          f"검증 {val_count}개 ({val_count/len(y_val):.1%})")

# 데이터 증강
print(f"\n🔄 데이터 증강 중...")
X_train, y_train, m14_train = DataAugmentation.augment_sequences(
    X_train, y_train, m14_train
)

print(f"\n📊 증강 후 데이터:")
print(f"  학습: {X_train.shape[0]:,}개")
for level in [1400, 1500, 1600, 1700]:
    count = np.sum(y_train >= level)
    print(f"  {level}+: {count}개 ({count/len(y_train):.1%})")

# 1400+ 여부 레이블 생성
y_spike_class = (y_train >= 1400).astype(float)
y_val_spike_class = (y_val >= 1400).astype(float)

# 클래스 가중치 계산
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_spike_class),
    y=y_spike_class
)
class_weight_dict = dict(enumerate(class_weights))

print(f"\n⚖️ 클래스 가중치: {class_weight_dict}")

# ============================================
# 7. 모델 학습
# ============================================
print("\n" + "="*60)
print("🏋️ 5개 모델 학습 시작")
print("="*60)

models = {}
history = {}
evaluation_results = {}

# 7.1 LSTM 모델
print("\n1️⃣ LSTM 모델 학습")

lstm_model = ImprovedModels.build_lstm_model(X_train.shape[1:])
lstm_model.compile(
    optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE),
    loss=ImprovedWeightedLoss(),
    metrics=['mae']
)

lstm_history = lstm_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    callbacks=ImprovedCallbacks.get_callbacks('lstm', X_val, y_val),
    verbose=1
)

models['lstm'] = lstm_model
history['lstm'] = lstm_history

# 7.2 GRU 모델
print("\n2️⃣ GRU 모델 학습")

gru_model = ImprovedModels.build_gru_model(X_train.shape[1:])
gru_model.compile(
    optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE),
    loss=ImprovedWeightedLoss(),
    metrics=['mae']
)

gru_history = gru_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    callbacks=ImprovedCallbacks.get_callbacks('gru', X_val, y_val),
    verbose=1
)

models['gru'] = gru_model
history['gru'] = gru_history

# 7.3 CNN-LSTM 모델
print("\n3️⃣ CNN-LSTM 모델 학습")

cnn_lstm_model = ImprovedModels.build_cnn_lstm(X_train.shape[1:])
cnn_lstm_model.compile(
    optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE),
    loss=ImprovedWeightedLoss(),
    metrics=['mae']
)

cnn_lstm_history = cnn_lstm_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    callbacks=ImprovedCallbacks.get_callbacks('cnn_lstm', X_val, y_val),
    verbose=1
)

models['cnn_lstm'] = cnn_lstm_model
history['cnn_lstm'] = cnn_lstm_history

# 7.4 Spike Detector 모델
print("\n4️⃣ Spike Detector 모델 학습")

spike_model = ImprovedModels.build_spike_detector(X_train.shape[1:])
spike_model.compile(
    optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE),
    loss={
        'spike_value': ImprovedWeightedLoss(),
        'spike_prob': 'binary_crossentropy'
    },
    loss_weights={
        'spike_value': 1.0,
        'spike_prob': 0.3  
    },
    metrics=['mae']
)

spike_history = spike_model.fit(
    X_train, 
    [y_train, y_spike_class],
    validation_data=(X_val, [y_val, y_val_spike_class]),
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    class_weight={'spike_prob': class_weight_dict},
    callbacks=ImprovedCallbacks.get_callbacks('spike', X_val, y_val),
    verbose=1
)

models['spike'] = spike_model
history['spike'] = spike_history

# 7.5 Rule-Based 모델
print("\n5️⃣ Rule-Based 모델 학습")

def build_rule_based_model(input_shape, m14_shape):
    """Rule-Based 모델"""
    time_input = tf.keras.Input(shape=input_shape, name='time_input')
    m14_input = tf.keras.Input(shape=m14_shape, name='m14_input')
    
    # 시계열은 간단히 처리
    lstm = tf.keras.layers.LSTM(32, dropout=0.3)(time_input)
    
    # M14 특징 강조
    m14_dense = tf.keras.layers.Dense(32, activation='relu')(m14_input)
    m14_dense = tf.keras.layers.Dropout(0.2)(m14_dense)
    
    # 결합
    combined = tf.keras.layers.Concatenate()([lstm, m14_dense])
    
    # Dense
    dense = tf.keras.layers.Dense(64, activation='relu')(combined)
    dense = tf.keras.layers.Dropout(0.3)(dense)
    
    # 예측
    prediction = tf.keras.layers.Dense(1, name='rule_pred')(dense)
    
    # M14 규칙 적용
    corrected = M14RuleCorrection()([prediction, m14_input])
    
    model = tf.keras.Model(
        inputs=[time_input, m14_input],
        outputs=corrected,
        name='Rule_Based_Model'
    )
    return model

rule_model = build_rule_based_model(X_train.shape[1:], m14_train.shape[1])
rule_model.compile(
    optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE * 0.5),
    loss=ImprovedWeightedLoss(),
    metrics=['mae']
)

rule_history = rule_model.fit(
    [X_train, m14_train], 
    y_train,
    validation_data=([X_val, m14_val], y_val),
    epochs=50,  
    batch_size=Config.BATCH_SIZE,
    callbacks=ImprovedCallbacks.get_callbacks('rule', X_val, y_val)[:3],
    verbose=1
)

models['rule'] = rule_model
history['rule'] = rule_history

# ============================================
# 8. 앙상블 모델
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

# 간단한 가중 평균
ensemble_pred = tf.keras.layers.Average()([
    lstm_pred, gru_pred, cnn_lstm_pred, spike_pred, rule_pred
])

# M14 규칙 보정
final_pred = M14RuleCorrection(name='ensemble_prediction')([ensemble_pred, m14_input])

# spike_prob 출력
spike_prob_output = tf.keras.layers.Lambda(lambda x: x, name='spike_probability')(spike_prob)

# 앙상블 모델 정의
ensemble_model = tf.keras.Model(
    inputs=[time_series_input, m14_input],
    outputs=[final_pred, spike_prob_output],
    name='Ensemble_Model'
)

# 컴파일
ensemble_model.compile(
    optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE * 0.5),
    loss={
        'ensemble_prediction': ImprovedWeightedLoss(),
        'spike_probability': 'binary_crossentropy'
    },
    loss_weights={
        'ensemble_prediction': 1.0,
        'spike_probability': 0.3
    },
    metrics=['mae']
)

print("\n📊 앙상블 파인튜닝...")
ensemble_history = ensemble_model.fit(
    [X_train, m14_train],
    [y_train, y_spike_class],
    validation_data=(
        [X_val, m14_val],
        [y_val, y_val_spike_class]
    ),
    epochs=30,  
    batch_size=Config.BATCH_SIZE,
    callbacks=ImprovedCallbacks.get_callbacks('ensemble', X_val, y_val)[:3],
    verbose=1
)

models['ensemble'] = ensemble_model
history['ensemble'] = ensemble_history

print("\n✅ 5개 모델 + 앙상블 학습 완료!")

# ============================================
# 9. 평가
# ============================================
print("\n" + "="*60)
print("📊 모델 평가")
print("="*60)

for name, model in models.items():
    print(f"\n🎯 {name.upper()} 모델")
    
    if name == 'ensemble':
        pred, _ = model.predict([X_val, m14_val], verbose=0)
        pred = pred.flatten()
    elif name == 'spike':
        pred, _ = model.predict(X_val, verbose=0)
        pred = pred.flatten()
    elif name == 'rule':
        pred = model.predict([X_val, m14_val], verbose=0).flatten()
    else:
        pred = model.predict(X_val, verbose=0).flatten()
    
    # 전체 성능
    mae = np.mean(np.abs(y_val - pred))
    rmse = np.sqrt(np.mean((y_val - pred)**2))
    
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    
    # 구간별 성능
    for level in [1400, 1500, 1600, 1700]:
        mask = y_val >= level
        if np.any(mask):
            pred_mask = pred >= level
            tp = np.sum((pred_mask) & (mask))
            fn = np.sum((~pred_mask) & (mask))
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            print(f"  {level}+ Recall: {recall:.2%} ({tp}/{tp+fn})")
    
    evaluation_results[name] = {'mae': mae, 'rmse': rmse}

# ============================================
# 10. 모델 저장
# ============================================
print("\n💾 모델 저장 중...")

for name, model in models.items():
    model.save(f"{Config.MODEL_DIR}{name}_model.h5")
    print(f"  ✅ {name}_model.h5 저장 완료")

# 평가 결과 저장
with open(f"{Config.MODEL_DIR}evaluation_results.json", 'w') as f:
    json.dump(evaluation_results, f, indent=2, default=str)

print("\n" + "="*60)
print("🎉 모든 작업 완료!")
print("="*60)