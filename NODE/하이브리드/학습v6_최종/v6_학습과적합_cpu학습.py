"""
V6_학습_개선판.py - 5개 모델 앙상블 학습 (성능 개선 버전)
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
print("🚀 반도체 물류 예측 앙상블 학습 V6 (개선판)")
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
    
    # 개선된 학습 설정
    BATCH_SIZE = 64  # 증가: 더 안정적인 그래디언트
    EPOCHS = 200  # 증가: 충분한 학습
    LEARNING_RATE = 0.0005  # 감소: 더 정밀한 학습
    PATIENCE = 30  # 증가: 더 인내심 있게
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
        'level_1400': 5.0,  # 증가
        'level_1500': 8.0,  # 증가
        'level_1600': 12.0,  # 증가
        'level_1700': 15.0  # 증가
    }

# 디렉토리 생성
os.makedirs(Config.MODEL_DIR, exist_ok=True)
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

# ============================================
# 2. 데이터 증강 클래스
# ============================================
class DataAugmentation:
    """데이터 증강을 통한 불균형 해결"""
    
    @staticmethod
    def augment_sequences(X, y, m14_features, target_levels=[1400, 1500, 1600, 1700]):
        """급증 구간 데이터 증강"""
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
            
            # 증강 비율 결정 (희소할수록 더 많이)
            if level >= 1700:
                augment_factor = 10
            elif level >= 1600:
                augment_factor = 5
            elif level >= 1500:
                augment_factor = 3
            else:
                augment_factor = 2
            
            for _ in range(augment_factor - 1):
                for idx in indices:
                    # 시계열 노이즈 추가
                    noise_scale = 0.02 if level >= 1600 else 0.01
                    noise = np.random.normal(0, noise_scale, X[idx].shape)
                    
                    # 시간 시프트 (±2 timesteps)
                    shift = np.random.randint(-2, 3)
                    if shift != 0:
                        shifted_x = np.roll(X[idx], shift, axis=0)
                    else:
                        shifted_x = X[idx]
                    
                    augmented_X.append(shifted_x + noise)
                    
                    # 타겟 변동 (작은 범위)
                    y_variation = np.random.uniform(0.98, 1.02)
                    augmented_y.append(y[idx] * y_variation)
                    
                    # M14 특징 변동
                    m14_variation = np.random.uniform(0.97, 1.03, m14_features[idx].shape)
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
        
        # 더 세분화된 가중치
        weights = tf.ones_like(y_true)
        weights = tf.where(y_true >= 1700, 15.0, weights)
        weights = tf.where((y_true >= 1600) & (y_true < 1700), 12.0, weights)
        weights = tf.where((y_true >= 1500) & (y_true < 1600), 8.0, weights)
        weights = tf.where((y_true >= 1400) & (y_true < 1500), 5.0, weights)
        
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
# 5. 개선된 콜백
# ============================================
# ============================================
# 5. 개선된 콜백 (수정 버전)
# ============================================
class ImprovedCallbacks:
    @staticmethod
    def cosine_annealing_scheduler(epoch, lr):
        """Cosine Annealing 학습률 스케줄러"""
        epochs = Config.EPOCHS
        lr_max = Config.LEARNING_RATE
        lr_min = lr_max * 0.0001
        
        return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(np.pi * epoch / epochs))
    
    @staticmethod
    def get_callbacks(model_name, X_val, y_val):
        """개선된 콜백 세트"""
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
                patience=15,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.LearningRateScheduler(
                ImprovedCallbacks.cosine_annealing_scheduler,
                verbose=0
            ),
            SpikePerformanceCallback(X_val, y_val)
        ]
        
        # TensorBoard는 옵션으로 (CPU에서는 느릴 수 있음)
        # 원하면 주석 해제
        # callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir))
        
        return callbacks

# 기존 SpikePerformanceCallback은 유지
class SpikePerformanceCallback(tf.keras.callbacks.Callback):
    """급증 감지 성능 모니터링"""
    def __init__(self, X_val, y_val):
        self.X_val = X_val
        self.y_val = y_val
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            pred = self.model.predict(self.X_val, verbose=0)
            if isinstance(pred, list):
                pred = pred[0]
            pred = pred.flatten()
            
            # 구간별 Recall과 Precision
            print("\n", end="")
            for level in [1400, 1500, 1600, 1700]:
                mask = self.y_val >= level
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
# M14RuleCorrection 레이어는 기존 유지
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
print("🏋️ 5개 모델 학습 시작 (개선된 버전)")
print("="*60)

models = {}
history = {}
evaluation_results = {}

# 7.1 LSTM 모델
print("\n1️⃣ LSTM 모델 학습 (개선된 구조)")

lstm_model = ImprovedModels.build_lstm_model(X_train.shape[1:])
lstm_model.compile(
    optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE),
    loss=ImprovedWeightedLoss(),
    metrics=['mae']
)

# 모델 구조 출력
lstm_model.summary()

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
print("\n2️⃣ Enhanced GRU 모델 학습 (개선된 구조)")

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
print("\n3️⃣ CNN-LSTM 모델 학습 (개선된 구조)")

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
print("\n4️⃣ Spike Detector 모델 학습 (개선된 구조)")

spike_model = ImprovedModels.build_spike_detector(X_train.shape[1:])
spike_model.compile(
    optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE),
    loss={
        'spike_value': ImprovedWeightedLoss(),
        'spike_prob': 'binary_crossentropy'
    },
    loss_weights={
        'spike_value': 1.0,
        'spike_prob': 0.5  # 증가
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

# 7.5 Rule-Based 모델 (간소화)
print("\n5️⃣ Rule-Based 모델 학습")

# Rule-based는 기존 구조 유지하되 정규화 강화
def build_rule_based_model(input_shape, m14_shape):
    """간소화된 Rule-Based 모델"""
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
    epochs=100,  # Rule-based는 짧게
    batch_size=Config.BATCH_SIZE,
    callbacks=ImprovedCallbacks.get_callbacks('rule', X_val, y_val)[:3],  # 기본 콜백만
    verbose=1
)

models['rule'] = rule_model
history['rule'] = rule_history

# ============================================
# 8. 앙상블 모델 (개선된 버전)
# ============================================
print("\n" + "="*60)
print("🎯 최종 앙상블 모델 구성 (개선된 버전)")
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

# 성능 기반 가중치 학습
ensemble_features = tf.keras.layers.Concatenate()([
    lstm_pred, gru_pred, cnn_lstm_pred, spike_pred, rule_pred, m14_input
])

# 가중치 네트워크
weight_dense = tf.keras.layers.Dense(64, activation='relu')(ensemble_features)
weight_dense = tf.keras.layers.LayerNormalization()(weight_dense)
weight_dense = tf.keras.layers.Dropout(0.3)(weight_dense)
weight_dense = tf.keras.layers.Dense(32, activation='relu')(weight_dense)
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

# 잔차 학습
residual_dense = tf.keras.layers.Dense(32, activation='relu')(ensemble_features)
residual_dense = tf.keras.layers.Dropout(0.2)(residual_dense)
residual = tf.keras.layers.Dense(1, name='residual')(residual_dense)

# 최종 예측
final_pred = tf.keras.layers.Add()([ensemble_pred, residual])

# M14 규칙 보정
final_pred = M14RuleCorrection(name='ensemble_prediction')([final_pred, m14_input])

# spike_prob 출력
spike_prob_output = tf.keras.layers.Lambda(lambda x: x, name='spike_probability')(spike_prob)

# 앙상블 모델 정의
ensemble_model = tf.keras.Model(
    inputs=[time_series_input, m14_input],
    outputs=[final_pred, spike_prob_output],
    name='Improved_Ensemble_Model'
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
        'spike_probability': 0.5
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
    epochs=50,  # 앙상블은 짧게
    batch_size=Config.BATCH_SIZE,
    class_weight={'spike_probability': class_weight_dict},
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
print("📊 모델 평가 (개선된 버전)")
print("="*60)

for name, model in models.items():
    print(f"\n{'='*40}")
    print(f"🎯 {name.upper()} 모델 평가")
    print(f"{'='*40}")
    
    if name == 'ensemble':
        pred, spike_pred = model.predict([X_val, m14_val], verbose=0)
        pred = pred.flatten()
    elif name == 'spike':
        pred, spike_pred = model.predict(X_val, verbose=0)
        pred = pred.flatten()
    elif name == 'rule':
        pred = model.predict([X_val, m14_val], verbose=0).flatten()
    else:
        pred = model.predict(X_val, verbose=0).flatten()
    
    # 전체 성능
    mae = np.mean(np.abs(y_val - pred))
    rmse = np.sqrt(np.mean((y_val - pred)**2))
    
    # 구간별 성능
    level_performance = {}
    for level in [1400, 1500, 1600, 1700]:
        mask = y_val >= level
        if np.any(mask):
            # Recall과 Precision
            pred_mask = pred >= level
            tp = np.sum((pred_mask) & (mask))
            fp = np.sum((pred_mask) & (~mask))
            fn = np.sum((~pred_mask) & (mask))
            tn = np.sum((~pred_mask) & (~mask))
            
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # MAE
            level_mae = np.mean(np.abs(y_val[mask] - pred[mask]))
            
            level_performance[level] = {
                'recall': recall,
                'precision': precision,
                'f1': f1,
                'mae': level_mae,
                'count': np.sum(mask),
                'tp': tp,
                'fp': fp,
                'fn': fn
            }
    
    evaluation_results[name] = {
        'overall_mae': mae,
        'overall_rmse': rmse,
        'levels': level_performance
    }
    
    # 출력
    print(f"\n📈 전체 성능:")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    
    print(f"\n📊 구간별 성능:")
    for level, perf in level_performance.items():
        print(f"\n  🎯 {level}+ 구간 (n={perf['count']}):")
        print(f"    Recall: {perf['recall']:.2%} ({perf['tp']}/{perf['tp']+perf['fn']})")
        print(f"    Precision: {perf['precision']:.2%} ({perf['tp']}/{perf['tp']+perf['fp']})")
        print(f"    F1-Score: {perf['f1']:.4f}")
        print(f"    MAE: {perf['mae']:.1f}")

# 최고 모델 선택 (F1 스코어 기준)
def calculate_weighted_score(results):
    """가중 평균 F1 스코어 계산"""
    weights = {1400: 1.0, 1500: 2.0, 1600: 3.0, 1700: 4.0}
    total_weight = 0
    weighted_f1 = 0
    
    for level, weight in weights.items():
        if level in results['levels']:
            weighted_f1 += results['levels'][level]['f1'] * weight
            total_weight += weight
    
    return weighted_f1 / total_weight if total_weight > 0 else 0

best_model = max(evaluation_results.keys(), 
                key=lambda x: calculate_weighted_score(evaluation_results[x]))

print(f"\n{'='*60}")
print(f"🏆 최고 성능 모델: {best_model.upper()}")
print(f"  가중 F1 스코어: {calculate_weighted_score(evaluation_results[best_model]):.4f}")
print(f"  전체 MAE: {evaluation_results[best_model]['overall_mae']:.2f}")
print(f"{'='*60}")

# ============================================
# 10. 모델 저장
# ============================================
print("\n💾 모델 저장 중...")

for name, model in models.items():
    model.save(f"{Config.MODEL_DIR}{name}_model_improved.h5")
    print(f"  ✅ {name}_model_improved.h5 저장 완료")

# 평가 결과 저장
with open(f"{Config.MODEL_DIR}evaluation_results_improved.json", 'w') as f:
    json.dump(evaluation_results, f, indent=2, default=str)

# 설정 저장
config_dict = {k: v for k, v in Config.__dict__.items() if not k.startswith('_')}
with open(f"{Config.MODEL_DIR}config_improved.json", 'w') as f:
    json.dump(config_dict, f, indent=2)

print("  ✅ 평가 결과 및 설정 저장 완료")

# ============================================
# 11. 시각화
# ============================================
print("\n📈 결과 시각화 생성 중...")

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(24, 16))

# 1-5. 각 모델 학습 곡선
for idx, (name, hist) in enumerate(history.items()):
    if idx < 5:
        ax = plt.subplot(4, 4, idx+1)
        
        if hasattr(hist, 'history'):
            if name == 'spike':
                loss = hist.history.get('spike_value_loss', [])
                val_loss = hist.history.get('val_spike_value_loss', [])
            else:
                loss = hist.history.get('loss', [])
                val_loss = hist.history.get('val_loss', [])
            
            if loss and val_loss:
                ax.plot(loss, label='Train Loss', alpha=0.8, linewidth=2)
                ax.plot(val_loss, label='Val Loss', alpha=0.8, linewidth=2)
                
                # 최소값 표시
                min_val_loss = min(val_loss)
                min_epoch = val_loss.index(min_val_loss)
                ax.axvline(x=min_epoch, color='red', linestyle='--', alpha=0.5)
                ax.text(min_epoch, min_val_loss, f'{min_val_loss:.1f}', 
                       ha='right', va='bottom', fontsize=8)
        
        ax.set_title(f'{name.upper()} Learning Curve', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

# 6. 앙상블 학습 곡선
ax = plt.subplot(4, 4, 6)
if 'ensemble' in history and hasattr(history['ensemble'], 'history'):
    loss = history['ensemble'].history.get('ensemble_prediction_loss', [])
    val_loss = history['ensemble'].history.get('val_ensemble_prediction_loss', [])
    if loss and val_loss:
        ax.plot(loss, label='Train Loss', alpha=0.8, linewidth=2, color='darkred')
        ax.plot(val_loss, label='Val Loss', alpha=0.8, linewidth=2, color='darkblue')
ax.set_title('ENSEMBLE Learning Curve', fontsize=12, fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
ax.grid(True, alpha=0.3)

# 7-8. 성능 비교 차트
# 7. MAE 비교
ax = plt.subplot(4, 4, 7)
model_names = list(evaluation_results.keys())
maes = [evaluation_results[m]['overall_mae'] for m in model_names]
colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))

bars = ax.bar(model_names, maes, color=colors)
ax.set_title('Model MAE Comparison', fontsize=12, fontweight='bold')
ax.set_ylabel('MAE')
ax.set_ylim(0, max(maes) * 1.2)

for bar, mae in zip(bars, maes):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{mae:.1f}', ha='center', va='bottom', fontsize=10)

# 8. RMSE 비교
ax = plt.subplot(4, 4, 8)
rmses = [evaluation_results[m]['overall_rmse'] for m in model_names]

bars = ax.bar(model_names, rmses, color=colors)
ax.set_title('Model RMSE Comparison', fontsize=12, fontweight='bold')
ax.set_ylabel('RMSE')
ax.set_ylim(0, max(rmses) * 1.2)

for bar, rmse in zip(bars, rmses):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{rmse:.1f}', ha='center', va='bottom', fontsize=10)

# 9-12. 각 레벨별 F1 스코어 비교
for idx, level in enumerate([1400, 1500, 1600, 1700]):
    ax = plt.subplot(4, 4, 9 + idx)
    
    f1_scores = []
    recalls = []
    precisions = []
    
    for m in model_names:
        if level in evaluation_results[m]['levels']:
            f1_scores.append(evaluation_results[m]['levels'][level]['f1'])
            recalls.append(evaluation_results[m]['levels'][level]['recall'])
            precisions.append(evaluation_results[m]['levels'][level]['precision'])
        else:
            f1_scores.append(0)
            recalls.append(0)
            precisions.append(0)
    
    x = np.arange(len(model_names))
    width = 0.25
    
    bars1 = ax.bar(x - width, recalls, width, label='Recall', alpha=0.8, color='blue')
    bars2 = ax.bar(x, precisions, width, label='Precision', alpha=0.8, color='green')
    bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8, color='red')
    
    ax.set_title(f'{level}+ Performance', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

# 13. 예측 산점도 (앙상블)
ax = plt.subplot(4, 4, 13)
if 'ensemble' in models:
    pred, _ = models['ensemble'].predict([X_val[:1000], m14_val[:1000]], verbose=0)
    pred = pred.flatten()
    actual = y_val[:1000]
    
    scatter = ax.scatter(actual, pred, alpha=0.5, s=10, c=actual, cmap='viridis')
    ax.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 
            'r--', alpha=0.8, linewidth=2)
    
    # 구간별 색상
    for level in [1400, 1500, 1600, 1700]:
        ax.axvline(x=level, color='gray', linestyle='--', alpha=0.3)
        ax.axhline(y=level, color='gray', linestyle='--', alpha=0.3)
    
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Ensemble Prediction Scatter', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=ax)

# 14. 오차 분포
ax = plt.subplot(4, 4, 14)
if 'ensemble' in models:
    errors = actual - pred
    ax.hist(errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Prediction Error')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Error Distribution (μ={errors.mean():.1f}, σ={errors.std():.1f})', 
                fontsize=12, fontweight='bold')

# 15. 가중치 시각화
ax = plt.subplot(4, 4, 15)
# 샘플 데이터로 가중치 추출
if 'ensemble' in models:
    # 가중치 레이어 찾기
    for layer in models['ensemble'].layers:
        if layer.name == 'ensemble_weights':
            sample_weights = layer(ensemble_features[:100]).numpy()
            avg_weights = sample_weights.mean(axis=0)
            
            model_names_short = ['LSTM', 'GRU', 'CNN', 'Spike', 'Rule']
            bars = ax.bar(model_names_short, avg_weights, color=colors[:5])
            ax.set_title('Average Ensemble Weights', fontsize=12, fontweight='bold')
            ax.set_ylabel('Weight')
            ax.set_ylim(0, 1)
            
            for bar, weight in zip(bars, avg_weights):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{weight:.2%}', ha='center', va='bottom')
            break

# 16. 성능 요약
ax = plt.subplot(4, 4, 16)
ax.axis('off')

summary_text = "🏆 Performance Summary (Improved)\n" + "="*45 + "\n"
summary_text += f"Best Model: {best_model.upper()}\n"
summary_text += f"Weighted F1: {calculate_weighted_score(evaluation_results[best_model]):.4f}\n"
summary_text += f"Overall MAE: {evaluation_results[best_model]['overall_mae']:.2f}\n"
summary_text += f"Overall RMSE: {evaluation_results[best_model]['overall_rmse']:.2f}\n\n"

summary_text += "Performance by Level:\n"
for level in [1400, 1500, 1600, 1700]:
    if level in evaluation_results[best_model]['levels']:
        perf = evaluation_results[best_model]['levels'][level]
        summary_text += f"\n{level}+ (n={perf['count']}):\n"
        summary_text += f"  Recall: {perf['recall']:6.1%}\n"
        summary_text += f"  Precision: {perf['precision']:6.1%}\n"
        summary_text += f"  F1-Score: {perf['f1']:.4f}\n"
        summary_text += f"  MAE: {perf['mae']:.1f}\n"

summary_text += f"\n개선사항:\n"
summary_text += f"• 데이터 증강으로 불균형 해결\n"
summary_text += f"• 강화된 정규화와 드롭아웃\n"
summary_text += f"• 개선된 손실 함수\n"
summary_text += f"• 최적화된 학습률 스케줄링"

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
       fontsize=10, verticalalignment='top', fontfamily='monospace',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('V6 앙상블 모델 성능 분석 (개선 버전)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{Config.MODEL_DIR}training_results_improved.png", dpi=150, bbox_inches='tight')
print("  ✅ training_results_improved.png 저장 완료")
plt.show()

# ============================================
# 12. 최종 출력
# ============================================
print("\n" + "="*60)
print("🎉 모든 작업 완료!")
print("="*60)
print(f"📁 모델 저장 위치: {Config.MODEL_DIR}")
print(f"📂 시퀀스 파일: {Config.SEQUENCE_FILE}")
print("\n📊 최종 성능:")
print(f"  최고 모델: {best_model.upper()}")
print(f"  가중 F1 스코어: {calculate_weighted_score(evaluation_results[best_model]):.4f}")
print(f"  전체 MAE: {evaluation_results[best_model]['overall_mae']:.2f}")
print(f"  전체 RMSE: {evaluation_results[best_model]['overall_rmse']:.2f}")

print("\n💡 주요 개선사항:")
print("  • 데이터 증강으로 급증 구간 학습 강화")
print("  • 개선된 모델 구조 (Attention, Residual 등)")
print("  • 강화된 정규화 및 드롭아웃")
print("  • 최적화된 손실 함수와 학습률 스케줄링")
print("  • 성능 기반 동적 앙상블 가중치")

print("\n💡 다음 단계:")
print("  1. 더 많은 데이터 수집 (특히 1600+, 1700+ 구간)")
print("  2. 하이퍼파라미터 튜닝 (Optuna 등 활용)")
print("  3. 실시간 예측 시스템 적용")
print("="*60)

# GPU 정보 출력
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print(f"\n🎮 GPU 사용: {len(gpus)}개")
    for gpu in gpus:
        print(f"  {gpu}")
else:
    print("\n💻 CPU 모드로 실행됨")