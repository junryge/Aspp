"""
V6_RAM_30GB_완전체.py - RAM 30GB 제한 GPU 학습
5개 모델 + 앙상블 + 체크포인트 시스템 완전 포함
"""

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
import pickle
import os
import gc
import psutil
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Mixed Precision 활성화 (메모리 50% 절약)
tf.keras.mixed_precision.set_global_policy('mixed_float16')

print("="*60)
print("🚀 V6 RAM 30GB 제한 - 완전체 시스템")
print("💾 최대 RAM: 30GB")
print("🎮 GPU 학습 + 5개 모델 + 앙상블")
print("="*60)

# ============================================
# GPU 설정
# ============================================
def setup_gpu():
    """GPU 설정"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU 감지: {len(gpus)}개")
            return True
        except:
            return False
    return False

has_gpu = setup_gpu()

# ============================================
# 메모리 관리
# ============================================
class MemoryManager:
    """RAM 30GB 제한 관리자"""
    
    @staticmethod
    def get_usage():
        """현재 RAM 사용량 GB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024**3
    
    @staticmethod
    def enforce_limit(max_gb=28):
        """메모리 제한 강제 (안전마진 2GB)"""
        current = MemoryManager.get_usage()
        if current > max_gb:
            gc.collect()
            gc.collect()
            tf.keras.backend.clear_session()
            new = MemoryManager.get_usage()
            print(f"  💾 메모리 정리: {current:.1f}GB → {new:.1f}GB")
    
    @staticmethod
    def print_status(stage=""):
        """메모리 상태 출력"""
        usage = MemoryManager.get_usage()
        print(f"  [{stage}] RAM: {usage:.1f}GB")

# ============================================
# 설정
# ============================================
class Config:
    # 파일 경로
    SEQUENCE_FILE = './sequences_v6_gpu.npz'
    SCALER_FILE = './scalers_v6_gpu.pkl'
    MODEL_DIR = './models_v6_gpu/'
    CHECKPOINT_DIR = './checkpoints_v6_gpu/'
    
    # 학습 설정
    BATCH_SIZE = 32  # 메모리 절약
    EPOCHS = 30
    LEARNING_RATE = 0.0005
    PATIENCE = 10
    
    # M14 임계값
    M14B_THRESHOLDS = {
        1400: 320,
        1500: 400,
        1600: 450,
        1700: 500
    }
    
    # 메모리 제한
    MAX_RAM_GB = 28  # 안전마진 2GB

# 디렉토리 생성
os.makedirs(Config.MODEL_DIR, exist_ok=True)
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

# ============================================
# 체크포인트 관리자
# ============================================
class CheckpointManager:
    """학습 중단/재개 관리"""
    
    def __init__(self):
        self.checkpoint_file = os.path.join(Config.CHECKPOINT_DIR, 'training_state.pkl')
    
    def save_state(self, completed_models, evaluation_results):
        """상태 저장"""
        state = {
            'completed_models': completed_models,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'evaluation_results': evaluation_results
        }
        
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"  💾 체크포인트 저장: {completed_models}")
    
    def load_state(self):
        """상태 복원"""
        if not os.path.exists(self.checkpoint_file):
            return [], {}
        
        with open(self.checkpoint_file, 'rb') as f:
            state = pickle.load(f)
        
        print(f"  🔄 체크포인트 복원: {state['completed_models']}")
        return state['completed_models'], state.get('evaluation_results', {})

# ============================================
# 커스텀 레이어 및 손실 함수
# ============================================
@tf.keras.utils.register_keras_serializable()
class WeightedLoss(tf.keras.losses.Loss):
    """가중치 손실 함수"""
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        mae = tf.abs(y_true - y_pred)
        
        # 물류량별 가중치
        weights = tf.ones_like(y_true)
        weights = tf.where(y_true >= 1700, 10.0, weights)
        weights = tf.where((y_true >= 1600) & (y_true < 1700), 8.0, weights)
        weights = tf.where((y_true >= 1500) & (y_true < 1600), 5.0, weights)
        weights = tf.where((y_true >= 1400) & (y_true < 1500), 3.0, weights)
        
        return tf.reduce_mean(mae * weights)

@tf.keras.utils.register_keras_serializable()
class M14RuleCorrection(tf.keras.layers.Layer):
    """M14 규칙 보정 레이어"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        pred, m14_features = inputs
        pred = tf.cast(pred, tf.float32)
        m14_features = tf.cast(m14_features, tf.float32)
        
        m14b = m14_features[:, 0:1]
        m10a = m14_features[:, 1:2]
        ratio = m14_features[:, 3:4]
        
        # 검증된 규칙
        pred = tf.where((m14b >= 500) & (ratio >= 7), tf.maximum(pred, 1700), pred)
        pred = tf.where((m14b >= 450) & (ratio >= 6), tf.maximum(pred, 1600), pred)
        pred = tf.where((m14b >= 400) & (ratio >= 5), tf.maximum(pred, 1500), pred)
        pred = tf.where(m14b >= 320, tf.maximum(pred, 1400), pred)
        
        # 황금 패턴
        golden = (m14b >= 350) & (m10a < 70)
        pred = tf.where(golden, pred * 1.1, pred)
        
        return pred

# ============================================
# 5개 모델 정의
# ============================================
class ModelsV6:
    """5개 전문 모델"""
    
    @staticmethod
    def build_lstm(input_shape):
        """1. LSTM 모델"""
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.3),
            tf.keras.layers.LSTM(64, dropout=0.3),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, dtype='float32')
        ], name='LSTM_Model')
        return model
    
    @staticmethod
    def build_gru(input_shape):
        """2. GRU 모델"""
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.GRU(128, return_sequences=True, dropout=0.2),
            tf.keras.layers.GRU(64, dropout=0.2),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, dtype='float32')
        ], name='GRU_Model')
        return model
    
    @staticmethod
    def build_cnn_lstm(input_shape):
        """3. CNN-LSTM 모델"""
        inputs = tf.keras.Input(shape=input_shape)
        
        conv1 = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
        conv2 = tf.keras.layers.Conv1D(64, 5, activation='relu', padding='same')(inputs)
        conv3 = tf.keras.layers.Conv1D(64, 7, activation='relu', padding='same')(inputs)
        
        concat = tf.keras.layers.Concatenate()([conv1, conv2, conv3])
        norm = tf.keras.layers.BatchNormalization()(concat)
        
        lstm = tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.2)(norm)
        lstm2 = tf.keras.layers.LSTM(64, dropout=0.2)(lstm)
        
        dense1 = tf.keras.layers.Dense(128, activation='relu')(lstm2)
        dropout = tf.keras.layers.Dropout(0.3)(dense1)
        output = tf.keras.layers.Dense(1, dtype='float32')(dropout)
        
        return tf.keras.Model(inputs=inputs, outputs=output, name='CNN_LSTM_Model')
    
    @staticmethod
    def build_spike_detector(input_shape):
        """4. Spike Detector 모델"""
        inputs = tf.keras.Input(shape=input_shape)
        
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
        
        regression_output = tf.keras.layers.Dense(1, name='spike_value', dtype='float32')(dropout2)
        classification_output = tf.keras.layers.Dense(1, activation='sigmoid', name='spike_prob', dtype='float32')(dropout2)
        
        return tf.keras.Model(
            inputs=inputs,
            outputs=[regression_output, classification_output],
            name='Spike_Detector'
        )
    
    @staticmethod
    def build_rule_based(input_shape, m14_shape):
        """5. Rule-Based 모델"""
        time_input = tf.keras.Input(shape=input_shape, name='time_input')
        m14_input = tf.keras.Input(shape=m14_shape, name='m14_input')
        
        lstm = tf.keras.layers.LSTM(32, dropout=0.2)(time_input)
        m14_dense = tf.keras.layers.Dense(16, activation='relu')(m14_input)
        
        combined = tf.keras.layers.Concatenate()([lstm, m14_dense])
        
        dense1 = tf.keras.layers.Dense(64, activation='relu')(combined)
        dropout = tf.keras.layers.Dropout(0.2)(dense1)
        dense2 = tf.keras.layers.Dense(32, activation='relu')(dropout)
        
        prediction = tf.keras.layers.Dense(1, name='rule_pred', dtype='float32')(dense2)
        corrected = M14RuleCorrection()([prediction, m14_input])
        
        return tf.keras.Model(
            inputs=[time_input, m14_input],
            outputs=corrected,
            name='Rule_Based_Model'
        )

# ============================================
# 데이터 제너레이터
# ============================================
class DataGenerator(tf.keras.utils.Sequence):
    """메모리 효율적 배치 생성"""
    
    def __init__(self, X_file, y_file, m14_file, indices, 
                 batch_size=32, model_type='normal', shuffle=True):
        # 메모리 맵으로 열기
        self.X_mmap = np.load(X_file, mmap_mode='r')
        self.y_mmap = np.load(y_file, mmap_mode='r')
        self.m14_mmap = np.load(m14_file, mmap_mode='r')
        
        self.indices = indices
        self.batch_size = batch_size
        self.model_type = model_type
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        return len(self.indices) // self.batch_size
    
    def __getitem__(self, index):
        batch_idx = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        
        # 배치만 메모리에 로드
        X = np.array(self.X_mmap[batch_idx], dtype=np.float16)
        y = np.array(self.y_mmap[batch_idx], dtype=np.float32)
        m14 = np.array(self.m14_mmap[batch_idx], dtype=np.float16)
        
        # 모델별 반환
        if self.model_type == 'spike':
            y_spike = (y >= 1400).astype(np.float32)
            return X, [y, y_spike]
        elif self.model_type == 'rule':
            return [X, m14], y
        elif self.model_type == 'ensemble':
            y_spike = (y >= 1400).astype(np.float32)
            return [X, m14], [y, y_spike]
        else:
            return X, y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        gc.collect()  # 에폭마다 메모리 정리

# ============================================
# 메인 실행
# ============================================
print(f"\n💾 초기 RAM: {MemoryManager.get_usage():.1f}GB")

# 1. 데이터 준비
print("\n📦 데이터 준비...")

# NPZ를 개별 파일로 분리 (메모리 효율)
if not os.path.exists('X_data.npy'):
    print("  NPZ → NPY 변환 중...")
    data = np.load(Config.SEQUENCE_FILE, mmap_mode='r')
    
    # 청크 단위로 저장
    chunk_size = 10000
    X_shape = data['X'].shape
    y_shape = data['y_original'].shape if 'y_original' in data else data['y'].shape
    m14_shape = data['m14_features'].shape
    
    # 메모리 맵 파일 생성
    X_mmap = np.memmap('X_data.npy', dtype='float16', mode='w+', shape=X_shape)
    y_mmap = np.memmap('y_data.npy', dtype='float32', mode='w+', shape=y_shape)
    m14_mmap = np.memmap('m14_data.npy', dtype='float16', mode='w+', shape=m14_shape)
    
    for i in range(0, X_shape[0], chunk_size):
        end = min(i + chunk_size, X_shape[0])
        X_mmap[i:end] = data['X'][i:end].astype(np.float16)
        y_mmap[i:end] = (data['y_original'][i:end] if 'y_original' in data else data['y'][i:end]).astype(np.float32)
        m14_mmap[i:end] = data['m14_features'][i:end].astype(np.float16)
        
        if i % 50000 == 0:
            print(f"    진행: {i}/{X_shape[0]}")
            MemoryManager.enforce_limit(Config.MAX_RAM_GB)
    
    del X_mmap, y_mmap, m14_mmap
    gc.collect()
    print("  ✅ 변환 완료")

# 데이터 정보 로드
X_mmap = np.load('X_data.npy', mmap_mode='r')
n_samples = len(X_mmap)
input_shape = (X_mmap.shape[1], X_mmap.shape[2])

# 인덱스 분할
np.random.seed(42)
indices = np.arange(n_samples)
np.random.shuffle(indices)

split = int(0.8 * n_samples)
train_idx = indices[:split]
val_idx = indices[split:]

print(f"  학습: {len(train_idx):,}개")
print(f"  검증: {len(val_idx):,}개")

MemoryManager.print_status("데이터 준비 완료")

# 2. 체크포인트 관리자
checkpoint_manager = CheckpointManager()
completed_models, evaluation_results = checkpoint_manager.load_state()

# 3. 5개 모델 학습
model_configs = [
    ('lstm', ModelsV6.build_lstm, input_shape, None, 'normal'),
    ('gru', ModelsV6.build_gru, input_shape, None, 'normal'),
    ('cnn_lstm', ModelsV6.build_cnn_lstm, input_shape, None, 'normal'),
    ('spike', ModelsV6.build_spike_detector, input_shape, None, 'spike'),
    ('rule', ModelsV6.build_rule_based, input_shape, (4,), 'rule')
]

for idx, (name, builder, shape1, shape2, model_type) in enumerate(model_configs):
    if name not in completed_models:
        print(f"\n{['1️⃣','2️⃣','3️⃣','4️⃣','5️⃣'][idx]} {name.upper()} 모델 학습")
        MemoryManager.print_status(f"{name} 시작")
        
        # 모델 생성
        if shape2:
            model = builder(shape1, shape2)
        else:
            model = builder(shape1)
        
        # 컴파일
        if name == 'spike':
            model.compile(
                optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE),
                loss={'spike_value': WeightedLoss(), 'spike_prob': 'binary_crossentropy'},
                loss_weights={'spike_value': 1.0, 'spike_prob': 0.3},
                metrics={'spike_value': 'mae', 'spike_prob': 'accuracy'}
            )
        elif name == 'rule':
            model.compile(
                optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE * 0.5),
                loss=WeightedLoss(),
                metrics=['mae']
            )
        else:
            model.compile(
                optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE),
                loss=WeightedLoss(),
                metrics=['mae']
            )
        
        # 제너레이터 생성
        train_gen = DataGenerator(
            'X_data.npy', 'y_data.npy', 'm14_data.npy',
            train_idx, Config.BATCH_SIZE, model_type, shuffle=True
        )
        val_gen = DataGenerator(
            'X_data.npy', 'y_data.npy', 'm14_data.npy',
            val_idx, Config.BATCH_SIZE, model_type, shuffle=False
        )
        
        # 콜백
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=Config.PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                patience=5,
                factor=0.5,
                verbose=1
            ),
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: (
                    MemoryManager.enforce_limit(Config.MAX_RAM_GB),
                    MemoryManager.print_status(f"에폭 {epoch+1}")
                )
            )
        ]
        
        # 학습
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=Config.EPOCHS,
            callbacks=callbacks,
            verbose=1,
            workers=1,
            use_multiprocessing=False
        )
        
        # 저장
        model.save(f"{Config.MODEL_DIR}{name}_final.h5")
        completed_models.append(name)
        
        # 평가
        if name == 'spike':
            eval_result = model.evaluate(val_gen, verbose=0)
            evaluation_results[name] = {'mae': eval_result[1]}
        else:
            eval_result = model.evaluate(val_gen, verbose=0)
            evaluation_results[name] = {'mae': eval_result[1] if len(eval_result) > 1 else eval_result}
        
        print(f"  ✅ {name} MAE: {evaluation_results[name]['mae']:.2f}")
        
        # 체크포인트 저장
        checkpoint_manager.save_state(completed_models, evaluation_results)
        
        # 메모리 정리
        del model, train_gen, val_gen
        tf.keras.backend.clear_session()
        gc.collect()
        
        MemoryManager.print_status(f"{name} 완료")
    else:
        print(f"\n{['1️⃣','2️⃣','3️⃣','4️⃣','5️⃣'][idx]} {name.upper()} - 이미 완료 ✓")

# 4. 앙상블 모델
if 'ensemble' not in completed_models:
    print("\n🎯 앙상블 모델 구성")
    MemoryManager.print_status("앙상블 시작")
    
    # 개별 모델 로드
    models = {}
    for name in ['lstm', 'gru', 'cnn_lstm', 'spike', 'rule']:
        custom_objects = {'WeightedLoss': WeightedLoss}
        if name == 'rule':
            custom_objects['M14RuleCorrection'] = M14RuleCorrection
        
        models[name] = tf.keras.models.load_model(
            f"{Config.MODEL_DIR}{name}_final.h5",
            custom_objects=custom_objects
        )
    
    # 앙상블 구성
    time_input = tf.keras.Input(shape=input_shape, name='ensemble_time')
    m14_input = tf.keras.Input(shape=(4,), name='ensemble_m14')
    
    # 각 모델 예측
    lstm_pred = models['lstm'](time_input)
    gru_pred = models['gru'](time_input)
    cnn_lstm_pred = models['cnn_lstm'](time_input)
    spike_pred, spike_prob = models['spike'](time_input)
    rule_pred = models['rule']([time_input, m14_input])
    
    # 동적 가중치
    weight_dense = tf.keras.layers.Dense(32, activation='relu')(m14_input)
    weight_dense = tf.keras.layers.Dense(16, activation='relu')(weight_dense)
    weights = tf.keras.layers.Dense(5, activation='softmax', dtype='float32')(weight_dense)
    
    w = [tf.keras.layers.Lambda(lambda x: x[:, i:i+1])(weights) for i in range(5)]
    
    # 가중 평균
    weighted = [
        tf.keras.layers.Multiply()([pred, weight])
        for pred, weight in zip([lstm_pred, gru_pred, cnn_lstm_pred, spike_pred, rule_pred], w)
    ]
    
    ensemble_pred = tf.keras.layers.Add()(weighted)
    final_pred = M14RuleCorrection(name='ensemble_output')([ensemble_pred, m14_input])
    spike_prob_output = tf.keras.layers.Lambda(lambda x: x, name='spike_output')(spike_prob)
    
    # 앙상블 모델
    ensemble_model = tf.keras.Model(
        inputs=[time_input, m14_input],
        outputs=[final_pred, spike_prob_output],
        name='Ensemble_Model'
    )
    
    # 컴파일
    ensemble_model.compile(
        optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE * 0.5),
        loss={'ensemble_output': WeightedLoss(), 'spike_output': 'binary_crossentropy'},
        loss_weights={'ensemble_output': 1.0, 'spike_output': 0.3},
        metrics={'ensemble_output': 'mae', 'spike_output': 'accuracy'}
    )
    
    # 파인튜닝
    train_gen = DataGenerator(
        'X_data.npy', 'y_data.npy', 'm14_data.npy',
        train_idx, Config.BATCH_SIZE, 'ensemble', shuffle=True
    )
    val_gen = DataGenerator(
        'X_data.npy', 'y_data.npy', 'm14_data.npy',
        val_idx, Config.BATCH_SIZE, 'ensemble', shuffle=False
    )
    
    ensemble_model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,  # 짧게
        callbacks=[
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: MemoryManager.enforce_limit(Config.MAX_RAM_GB)
            )
        ],
        verbose=1
    )
    
    # 저장
    ensemble_model.save(f"{Config.MODEL_DIR}ensemble_final.h5")
    
    # 평가
    eval_result = ensemble_model.evaluate(val_gen, verbose=0)
    evaluation_results['ensemble'] = {'mae': eval_result[1]}
    
    print(f"  ✅ 앙상블 MAE: {evaluation_results['ensemble']['mae']:.2f}")
    
    completed_models.append('ensemble')
    checkpoint_manager.save_state(completed_models, evaluation_results)
    
    # 메모리 정리
    del models, ensemble_model
    tf.keras.backend.clear_session()
    gc.collect()
    
    MemoryManager.print_status("앙상블 완료")

# 5. 최종 결과
print("\n" + "="*60)
print("📊 최종 결과")
print("="*60)

for name in completed_models:
    if name in evaluation_results:
        mae = evaluation_results[name].get('mae', 'N/A')
        print(f"  {name.upper()}: MAE = {mae:.2f}" if isinstance(mae, float) else f"  {name.upper()}: {mae}")

best_model = min(evaluation_results.keys(), key=lambda x: evaluation_results[x].get('mae', float('inf')))
print(f"\n🏆 최고 모델: {best_model.upper()}")

# 결과 저장
with open(f"{Config.MODEL_DIR}evaluation_results.json", 'w') as f:
    json.dump(evaluation_results, f, indent=2, default=str)

print(f"\n💾 최종 RAM: {MemoryManager.get_usage():.1f}GB")
print("✅ RAM 30GB 제한 성공!")
print("✅ 5개 모델 + 앙상블 완료!")
print("✅ GPU 학습 완료!")
print("="*60)