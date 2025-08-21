"""
train_v6.py - 메모리 최적화 학습 코드
배치 처리 및 메모리 관리 강화
TensorFlow 2.15.0
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
print("🚀 반도체 물류 예측 학습 V6 - 메모리 최적화")
print(f"📦 TensorFlow 버전: {tf.__version__}")
print("="*60)

# GPU 메모리 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

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
    
    # 학습 설정 - 메모리 최적화
    BATCH_SIZE = 16  # 32에서 16으로 감소
    EPOCHS = 100
    LEARNING_RATE = 0.001
    PATIENCE = 15
    
    # 메모리 관리
    MAX_SAMPLES = 100000  # 최대 샘플 수 제한
    SUBSET_RATIO = 0.3    # 데이터의 30%만 사용 (필요시)
    
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
# 2. 메모리 효율적 데이터 로드
# ============================================
print("\n📂 시퀀스 로딩 중 (메모리 최적화)...")

def load_data_efficiently():
    """메모리 효율적으로 데이터 로드"""
    # 메모리 맵 모드로 로드
    data = np.load(Config.SEQUENCE_FILE, mmap_mode='r')
    
    # 데이터 크기 확인
    total_samples = data['X'].shape[0]
    print(f"  전체 샘플 수: {total_samples:,}")
    
    # 메모리 제한이 필요한 경우
    if total_samples > Config.MAX_SAMPLES:
        print(f"  ⚠️ 메모리 제한으로 {Config.MAX_SAMPLES:,}개 샘플만 사용")
        
        # 랜덤 인덱스 선택
        np.random.seed(42)
        indices = np.random.choice(total_samples, Config.MAX_SAMPLES, replace=False)
        indices = np.sort(indices)
        
        # float32로 변환하여 메모리 절약
        X = np.array(data['X'][indices], dtype=np.float32)
        y = np.array(data['y'][indices], dtype=np.float32)
        m14_features = np.array(data['m14_features'][indices], dtype=np.float32)
    else:
        # float32로 변환
        X = np.array(data['X'], dtype=np.float32)
        y = np.array(data['y'], dtype=np.float32)
        m14_features = np.array(data['m14_features'], dtype=np.float32)
    
    print(f"  ✅ 로드 완료!")
    print(f"  X shape: {X.shape} (dtype: {X.dtype})")
    print(f"  y shape: {y.shape} (dtype: {y.dtype})")
    print(f"  m14_features shape: {m14_features.shape} (dtype: {m14_features.dtype})")
    
    # 메모리 사용량 출력
    memory_usage = (X.nbytes + y.nbytes + m14_features.nbytes) / (1024**3)
    print(f"  💾 메모리 사용: {memory_usage:.2f} GB")
    
    return X, y, m14_features

# 데이터 로드
X, y, m14_features = load_data_efficiently()

# 가비지 컬렉션
gc.collect()

# 학습/검증 분할 - 메모리 효율적 처리
print("\n📊 데이터 분할 중...")
try:
    X_train, X_val, y_train, y_val, m14_train, m14_val = train_test_split(
        X, y, m14_features, test_size=0.2, random_state=42
    )
    print("  ✅ 분할 완료!")
except MemoryError:
    print("  ⚠️ 메모리 부족 - 더 작은 데이터셋 사용")
    
    # 데이터 크기 추가 감소
    subset_size = int(len(X) * Config.SUBSET_RATIO)
    indices = np.random.choice(len(X), subset_size, replace=False)
    
    X_subset = X[indices]
    y_subset = y[indices]
    m14_subset = m14_features[indices]
    
    # 원본 데이터 메모리 해제
    del X, y, m14_features
    gc.collect()
    
    X_train, X_val, y_train, y_val, m14_train, m14_val = train_test_split(
        X_subset, y_subset, m14_subset, test_size=0.2, random_state=42
    )
    
    del X_subset, y_subset, m14_subset
    gc.collect()
    print(f"  ✅ 축소된 데이터셋으로 분할 완료 ({subset_size:,}개 샘플)")

# 1400+ 여부 레이블 생성
y_spike_class = (y_train >= 1400).astype(np.float32)
y_val_spike_class = (y_val >= 1400).astype(np.float32)

print(f"\n📊 최종 데이터:")
print(f"  학습: {X_train.shape[0]:,}개")
print(f"  검증: {X_val.shape[0]:,}개")
print(f"  1400+ 학습 비율: {y_spike_class.mean():.1%}")
print(f"  1400+ 검증 비율: {y_val_spike_class.mean():.1%}")

# ============================================
# 3. 경량화된 모델 정의
# ============================================
class LightweightModels:
    """메모리 효율적인 경량 모델"""
    
    @staticmethod
    def build_lstm_model(input_shape):
        """경량 LSTM 모델"""
        inputs = tf.keras.Input(shape=input_shape, name='lstm_input')
        
        # 레이어 크기 감소
        lstm1 = tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2)(inputs)
        lstm2 = tf.keras.layers.LSTM(32, dropout=0.2)(lstm1)
        
        # Dense layers
        dense1 = tf.keras.layers.Dense(64, activation='relu')(lstm2)
        dropout = tf.keras.layers.Dropout(0.3)(dense1)
        
        # Output
        output = tf.keras.layers.Dense(1, name='lstm_output')(dropout)
        
        model = tf.keras.Model(inputs=inputs, outputs=output, name='LSTM_Model')
        return model
    
    @staticmethod
    def build_gru_model(input_shape):
        """경량 GRU 모델"""
        inputs = tf.keras.Input(shape=input_shape, name='gru_input')
        
        # Layer Normalization
        x = tf.keras.layers.LayerNormalization()(inputs)
        
        # GRU layers
        gru1 = tf.keras.layers.GRU(64, return_sequences=True, dropout=0.2)(x)
        gru2 = tf.keras.layers.GRU(32, dropout=0.2)(gru1)
        
        # Dense layers
        dense1 = tf.keras.layers.Dense(64, activation='relu')(gru2)
        dropout = tf.keras.layers.Dropout(0.3)(dense1)
        
        # Output
        output = tf.keras.layers.Dense(1, name='gru_output')(dropout)
        
        model = tf.keras.Model(inputs=inputs, outputs=output, name='GRU_Model')
        return model
    
    @staticmethod
    def build_cnn_lstm(input_shape):
        """경량 CNN-LSTM 모델"""
        inputs = tf.keras.Input(shape=input_shape, name='cnn_lstm_input')
        
        # CNN layers
        conv1 = tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same')(inputs)
        conv2 = tf.keras.layers.Conv1D(32, 5, activation='relu', padding='same')(inputs)
        
        # Concatenate
        concat = tf.keras.layers.Concatenate()([conv1, conv2])
        pool = tf.keras.layers.MaxPooling1D(pool_size=2)(concat)
        
        # LSTM layer
        lstm = tf.keras.layers.LSTM(32, dropout=0.2)(pool)
        
        # Dense layers
        dense1 = tf.keras.layers.Dense(32, activation='relu')(lstm)
        dropout = tf.keras.layers.Dropout(0.3)(dense1)
        
        # Output
        output = tf.keras.layers.Dense(1, name='cnn_lstm_output')(dropout)
        
        model = tf.keras.Model(inputs=inputs, outputs=output, name='CNN_LSTM_Model')
        return model
    
    @staticmethod
    def build_spike_detector(input_shape):
        """경량 급변 감지 모델"""
        inputs = tf.keras.Input(shape=input_shape, name='spike_input')
        
        # CNN for pattern detection
        conv = tf.keras.layers.Conv1D(32, 5, activation='relu', padding='same')(inputs)
        
        # Batch normalization
        norm = tf.keras.layers.BatchNormalization()(conv)
        
        # BiLSTM
        lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(32, dropout=0.2)
        )(norm)
        
        # Dense layers
        dense1 = tf.keras.layers.Dense(64, activation='relu')(lstm)
        dropout = tf.keras.layers.Dropout(0.3)(dense1)
        
        # Dual output
        regression_output = tf.keras.layers.Dense(1, name='spike_value')(dropout)
        classification_output = tf.keras.layers.Dense(1, activation='sigmoid', name='spike_prob')(dropout)
        
        model = tf.keras.Model(
            inputs=inputs,
            outputs=[regression_output, classification_output],
            name='Spike_Detector'
        )
        return model

# ============================================
# 4. 커스텀 손실 함수
# ============================================
class WeightedLoss(tf.keras.losses.Loss):
    """레벨별 가중 손실 함수"""
    def __init__(self):
        super().__init__()
        
    def call(self, y_true, y_pred):
        # 레벨별 가중치
        weights = tf.where(y_true < 1400, 1.0,
                 tf.where(y_true < 1500, 3.0,
                 tf.where(y_true < 1600, 5.0,
                 tf.where(y_true < 1700, 8.0, 10.0))))
        
        # 가중 MAE
        mae = tf.abs(y_true - y_pred)
        weighted_mae = mae * weights
        
        return tf.reduce_mean(weighted_mae)

# ============================================
# 5. 메모리 효율적 콜백
# ============================================
class MemoryCallback(tf.keras.callbacks.Callback):
    """메모리 관리 콜백"""
    def on_epoch_end(self, epoch, logs=None):
        # 주기적으로 가비지 컬렉션
        if epoch % 5 == 0:
            gc.collect()
            if gpus:
                tf.keras.backend.clear_session()

class SimpleSpikeCallback(tf.keras.callbacks.Callback):
    """간단한 1400+ 성능 모니터링"""
    def __init__(self, X_val, y_val, check_interval=10):
        super().__init__()
        self.X_val = X_val[:1000]  # 검증용 샘플 제한
        self.y_val = y_val[:1000]
        self.check_interval = check_interval
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.check_interval == 0:
            # 예측
            y_pred = self.model.predict(self.X_val, verbose=0, batch_size=Config.BATCH_SIZE)
            if isinstance(y_pred, list):
                y_pred = y_pred[0]
            y_pred = y_pred.flatten()
            
            # 1400+ 구간 성능
            spike_mask = self.y_val >= 1400
            if np.any(spike_mask):
                spike_mae = np.mean(np.abs(self.y_val[spike_mask] - y_pred[spike_mask]))
                recall = np.sum((y_pred >= 1400) & spike_mask) / np.sum(spike_mask)
                print(f"  [Epoch {epoch+1}] 1400+ MAE: {spike_mae:.1f}, Recall: {recall:.1%}")

# ============================================
# 6. 데이터 생성기 (메모리 효율)
# ============================================
class DataGenerator(tf.keras.utils.Sequence):
    """메모리 효율적인 데이터 생성기"""
    def __init__(self, X, y, m14=None, batch_size=32, shuffle=True):
        self.X = X
        self.y = y
        self.m14 = m14
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(X))
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))
    
    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        
        X_batch = self.X[indices]
        y_batch = self.y[indices]
        
        if self.m14 is not None:
            m14_batch = self.m14[indices]
            return [X_batch, m14_batch], y_batch
        
        return X_batch, y_batch
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# ============================================
# 7. 학습 파이프라인
# ============================================
print("\n" + "="*60)
print("🏋️ 모델 학습 시작 (메모리 최적화)")
print("="*60)

models = {}
history = {}
evaluation_results = {}

# 데이터 생성기 생성
train_gen = DataGenerator(X_train, y_train, batch_size=Config.BATCH_SIZE)
val_gen = DataGenerator(X_val, y_val, batch_size=Config.BATCH_SIZE, shuffle=False)

# ============================================
# 1. LSTM 모델
# ============================================
print("\n1️⃣ LSTM 모델 학습")

try:
    lstm_model = LightweightModels.build_lstm_model(X_train.shape[1:])
    lstm_model.compile(
        optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE),
        loss=WeightedLoss(),
        metrics=['mae']
    )
    
    lstm_history = lstm_model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=Config.EPOCHS,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                f"{Config.MODEL_DIR}lstm_checkpoint.h5",
                save_best_only=True,
                monitor='val_loss',
                verbose=0
            ),
            tf.keras.callbacks.EarlyStopping(
                patience=Config.PATIENCE, 
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                patience=5, 
                factor=0.5,
                verbose=0
            ),
            MemoryCallback(),
            SimpleSpikeCallback(X_val, y_val)
        ],
        verbose=1
    )
    
    models['lstm'] = lstm_model
    history['lstm'] = lstm_history
    print("  ✅ LSTM 학습 완료")
    
except Exception as e:
    print(f"  ❌ LSTM 학습 실패: {e}")
    
# 메모리 정리
gc.collect()

# ============================================
# 2. GRU 모델
# ============================================
print("\n2️⃣ GRU 모델 학습")

try:
    gru_model = LightweightModels.build_gru_model(X_train.shape[1:])
    gru_model.compile(
        optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE),
        loss=WeightedLoss(),
        metrics=['mae']
    )
    
    gru_history = gru_model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=Config.EPOCHS,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                f"{Config.MODEL_DIR}gru_checkpoint.h5",
                save_best_only=True,
                monitor='val_loss',
                verbose=0
            ),
            tf.keras.callbacks.EarlyStopping(
                patience=Config.PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                patience=5,
                factor=0.5,
                verbose=0
            ),
            MemoryCallback(),
            SimpleSpikeCallback(X_val, y_val)
        ],
        verbose=1
    )
    
    models['gru'] = gru_model
    history['gru'] = gru_history
    print("  ✅ GRU 학습 완료")
    
except Exception as e:
    print(f"  ❌ GRU 학습 실패: {e}")

# 메모리 정리
gc.collect()

# ============================================
# 3. CNN-LSTM 모델
# ============================================
print("\n3️⃣ CNN-LSTM 모델 학습")

try:
    cnn_lstm_model = LightweightModels.build_cnn_lstm(X_train.shape[1:])
    cnn_lstm_model.compile(
        optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE),
        loss=WeightedLoss(),
        metrics=['mae']
    )
    
    cnn_lstm_history = cnn_lstm_model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=Config.EPOCHS,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                f"{Config.MODEL_DIR}cnn_lstm_checkpoint.h5",
                save_best_only=True,
                monitor='val_loss',
                verbose=0
            ),
            tf.keras.callbacks.EarlyStopping(
                patience=Config.PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                patience=5,
                factor=0.5,
                verbose=0
            ),
            MemoryCallback()
        ],
        verbose=1
    )
    
    models['cnn_lstm'] = cnn_lstm_model
    history['cnn_lstm'] = cnn_lstm_history
    print("  ✅ CNN-LSTM 학습 완료")
    
except Exception as e:
    print(f"  ❌ CNN-LSTM 학습 실패: {e}")

# 메모리 정리
gc.collect()

# ============================================
# 4. Spike Detector 모델
# ============================================
print("\n4️⃣ Spike Detector 모델 학습")

try:
    # Spike 데이터 생성기
    spike_train_gen = DataGenerator(
        X_train, 
        {'spike_value': y_train, 'spike_prob': y_spike_class},
        batch_size=Config.BATCH_SIZE
    )
    spike_val_gen = DataGenerator(
        X_val,
        {'spike_value': y_val, 'spike_prob': y_val_spike_class},
        batch_size=Config.BATCH_SIZE,
        shuffle=False
    )
    
    spike_model = LightweightModels.build_spike_detector(X_train.shape[1:])
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
    
    # 커스텀 fit 구현 (메모리 효율)
    spike_history = {'loss': [], 'val_loss': []}
    
    for epoch in range(min(Config.EPOCHS, 50)):  # 최대 50 에폭
        print(f"  Epoch {epoch+1}/{min(Config.EPOCHS, 50)}")
        
        # 학습
        train_loss = []
        for batch_idx in range(len(train_gen)):
            X_batch, y_batch = train_gen[batch_idx]
            
            # y_batch가 딕셔너리가 아닌 경우 처리
            if not isinstance(y_batch, dict):
                y_batch = {'spike_value': y_batch, 'spike_prob': y_spike_class[batch_idx*Config.BATCH_SIZE:(batch_idx+1)*Config.BATCH_SIZE]}
            
            loss = spike_model.train_on_batch(X_batch, y_batch)
            train_loss.append(loss[0] if isinstance(loss, list) else loss)
        
        # 검증
        val_loss = []
        for batch_idx in range(len(val_gen)):
            X_batch, y_batch = val_gen[batch_idx]
            
            # y_batch가 딕셔너리가 아닌 경우 처리
            if not isinstance(y_batch, dict):
                y_batch = {'spike_value': y_batch, 'spike_prob': y_val_spike_class[batch_idx*Config.BATCH_SIZE:(batch_idx+1)*Config.BATCH_SIZE]}
            
            loss = spike_model.test_on_batch(X_batch, y_batch)
            val_loss.append(loss[0] if isinstance(loss, list) else loss)
        
        avg_train_loss = np.mean(train_loss)
        avg_val_loss = np.mean(val_loss)
        
        spike_history['loss'].append(avg_train_loss)
        spike_history['val_loss'].append(avg_val_loss)
        
        print(f"    Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
        
        # Early stopping
        if epoch > 10 and avg_val_loss > min(spike_history['val_loss'][:-1]):
            print("  Early stopping")
            break
        
        # 메모리 정리
        if epoch % 5 == 0:
            gc.collect()
    
    models['spike'] = spike_model
    history['spike'] = type('', (), {'history': spike_history})()
    print("  ✅ Spike Detector 학습 완료")
    
except Exception as e:
    print(f"  ❌ Spike Detector 학습 실패: {e}")

# 메모리 정리
gc.collect()

print("\n✅ 모든 모델 학습 완료!")

# ============================================
# 8. 평가 (배치 처리)
# ============================================
print("\n" + "="*60)
print("📊 모델 평가")
print("="*60)

# 평가용 샘플 제한
eval_size = min(5000, len(X_val))
X_eval = X_val[:eval_size]
y_eval = y_val[:eval_size]

for name, model in models.items():
    if model is None:
        continue
        
    # 배치 예측
    predictions = []
    for i in range(0, len(X_eval), Config.BATCH_SIZE):
        batch = X_eval[i:i+Config.BATCH_SIZE]
        pred = model.predict(batch, verbose=0)
        if isinstance(pred, list):
            pred = pred[0]
        predictions.append(pred)
    
    pred = np.concatenate(predictions, axis=0).flatten()
    
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
if evaluation_results:
    best_model = min(evaluation_results.keys(), key=lambda x: evaluation_results[x]['overall_mae'])
    print(f"\n🏆 최고 성능: {best_model.upper()} 모델")

# ============================================
# 9. 모델 저장
# ============================================
print("\n💾 모델 저장 중...")

for name, model in models.items():
    if model is not None:
        try:
            model.save(f"{Config.MODEL_DIR}{name}_model.h5")
            print(f"  {name}_model.h5 저장 완료")
        except Exception as e:
            print(f"  {name} 모델 저장 실패: {e}")

# 평가 결과 저장
with open(f"{Config.MODEL_DIR}evaluation_results.json", 'w') as f:
    json.dump(evaluation_results, f, indent=2)

# 설정 저장
config_dict = {k: v for k, v in Config.__dict__.items() if not k.startswith('_')}
with open(f"{Config.MODEL_DIR}config.json", 'w') as f:
    json.dump(config_dict, f, indent=2)

print("  결과 파일 저장 완료")

# ============================================
# 10. 간단한 시각화
# ============================================
print("\n📈 결과 시각화 생성 중...")

try:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 학습 곡선
    for idx, (name, hist) in enumerate(history.items()):
        if idx >= 4:
            break
        ax = axes[idx // 2, idx % 2]
        
        if hasattr(hist, 'history'):
            loss = hist.history.get('loss', [])
            val_loss = hist.history.get('val_loss', [])
            
            if loss and val_loss:
                ax.plot(loss, label='Train Loss')
                ax.plot(val_loss, label='Val Loss')
                ax.set_title(f'{name.upper()} Learning Curve')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.legend()
                ax.grid(True, alpha=0.3)
    
    plt.suptitle('메모리 최적화 학습 결과', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{Config.MODEL_DIR}training_results.png", dpi=100, bbox_inches='tight')
    print("  training_results.png 저장 완료")
    plt.show()
    
except Exception as e:
    print(f"  시각화 생성 실패: {e}")

print("\n" + "="*60)
print("🎉 모든 작업 완료!")
print(f"📁 저장 위치: {Config.MODEL_DIR}")
print(f"📂 시퀀스 파일: {Config.SEQUENCE_FILE}")
print("="*60)

# 최종 메모리 정리
del X_train, X_val, y_train, y_val, m14_train, m14_val
gc.collect()

# GPU 정보 출력
if gpus:
    print(f"\n🎮 GPU 사용: {len(gpus)}개")
    for gpu in gpus:
        print(f"  {gpu}")
else:
    print("\n💻 CPU 모드로 실행됨")

print("\n💡 메모리 최적화 팁:")
print("  - BATCH_SIZE를 더 줄이기 (8 또는 4)")
print("  - MAX_SAMPLES 값 조정")
print("  - 모델 레이어 크기 추가 축소")
print("  - 시퀀스 파일을 더 작게 재생성")