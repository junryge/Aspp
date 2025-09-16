"""
V6 GPU 최적화 + 체크포인트 시스템
- GPU 메모리 최대 활용 (배치 512)
- 중단 시 자동 저장 및 재개
- Mixed Precision Training
- 학습 진행 상황 추적
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
import json
import os
import pickle
import warnings
import gc
import time
from datetime import datetime
warnings.filterwarnings('ignore')

print("="*60)
print("🚀 V6 GPU 최적화 + 체크포인트 버전")
print(f"📦 TensorFlow: {tf.__version__}")
print("="*60)

# ============================================
# GPU 최적화 설정
# ============================================
def setup_gpu_optimized():
    """GPU 최적화 설정"""
    print("\n🎮 GPU 최적화 설정...")
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Mixed Precision 활성화
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print(f"✅ Mixed Precision 활성화: {policy.name}")
            
            print(f"✅ GPU 감지: {len(gpus)}개")
            return True
        except Exception as e:
            print(f"⚠️ GPU 설정 오류: {e}")
            return False
    else:
        print("💻 CPU 모드로 실행")
        return False

has_gpu = setup_gpu_optimized()

# ============================================
# 최적화된 설정
# ============================================
class Config:
    # 시퀀스 파일
    SEQUENCE_FILE = './sequences_v6_gpu.npz'
    SCALER_FILE = './scalers_v6_gpu.pkl'
    
    # 시퀀스 설정
    LOOKBACK = 100
    FORECAST = 10
    
    # GPU 최적화 학습 설정
    if has_gpu:
        BATCH_SIZE = 512  # GPU 메모리 활용
        LEARNING_RATE = 0.003  # 큰 배치에 맞게
    else:
        BATCH_SIZE = 64
        LEARNING_RATE = 0.001
    
    EPOCHS = 200
    
    # 저장 경로
    MODEL_DIR = './models_v6_checkpoint/'
    CHECKPOINT_DIR = './checkpoints_v6/'
    
    # 체크포인트 설정
    SAVE_FREQ = 5  # 5 에폭마다 저장
    
    USE_SCALED_Y = False

# 디렉토리 생성
os.makedirs(Config.MODEL_DIR, exist_ok=True)
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

# ============================================
# 체크포인트 관리자
# ============================================
class CheckpointManager:
    """학습 상태 저장 및 복원 관리"""
    
    def __init__(self):
        self.state_file = os.path.join(Config.CHECKPOINT_DIR, 'training_state.json')
        
    def save_state(self, model_name, epoch, history):
        """현재 학습 상태 저장"""
        state = {
            'model_name': model_name,
            'last_epoch': epoch,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'history': {
                'loss': history.history.get('loss', [])[-50:],  # 최근 50개
                'mae': history.history.get('mae', [])[-50:]
            }
        }
        
        # 상태 파일 저장
        state_path = os.path.join(Config.CHECKPOINT_DIR, f'{model_name}_state.json')
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"💾 체크포인트 저장: {model_name} (Epoch {epoch})")
        
    def load_state(self, model_name):
        """저장된 상태 불러오기"""
        state_path = os.path.join(Config.CHECKPOINT_DIR, f'{model_name}_state.json')
        
        if os.path.exists(state_path):
            with open(state_path, 'r') as f:
                state = json.load(f)
            print(f"🔄 체크포인트 로드: {model_name} (Epoch {state['last_epoch']})")
            return state
        return None
    
    def get_initial_epoch(self, model_name):
        """시작 에폭 확인"""
        state = self.load_state(model_name)
        if state:
            return state['last_epoch']
        return 0

# ============================================
# 손실 함수
# ============================================
class WeightedLoss(tf.keras.losses.Loss):
    """가중치 손실 함수"""
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        mae = tf.abs(y_true - y_pred)
        
        # 강화된 가중치
        weights = tf.ones_like(y_true)
        if not Config.USE_SCALED_Y:
            weights = tf.where(y_true >= 1550, 30.0, weights)
            weights = tf.where((y_true >= 1500) & (y_true < 1550), 25.0, weights)
            weights = tf.where((y_true >= 1450) & (y_true < 1500), 20.0, weights)
            weights = tf.where((y_true >= 1400) & (y_true < 1450), 15.0, weights)
            weights = tf.where((y_true >= 1350) & (y_true < 1400), 10.0, weights)
        
        # 큰 오차 페널티
        large_error = tf.where(mae > 100, mae * 0.2, 0.0)
        
        return tf.reduce_mean(mae * weights + large_error)

# ============================================
# M14 규칙 보정 레이어
# ============================================
class M14RuleCorrection(tf.keras.layers.Layer):
    """M14 규칙 기반 보정"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs, training=None):
        pred, m14_features = inputs
        
        pred = tf.cast(pred, tf.float32)
        m14_features = tf.cast(m14_features, tf.float32)
        
        m14b = m14_features[:, 0:1]
        m10a = m14_features[:, 1:2]
        m16 = m14_features[:, 2:3]
        ratio = m14_features[:, 3:4]
        
        if not Config.USE_SCALED_Y:
            # 임계값 규칙
            pred = tf.where(m14b >= 420, tf.maximum(pred, 1550.0), pred)
            pred = tf.where(m14b >= 380, tf.maximum(pred, 1500.0), pred)
            pred = tf.where(m14b >= 350, tf.maximum(pred, 1450.0), pred)
            pred = tf.where(m14b >= 300, tf.maximum(pred, 1400.0), pred)
            
            # 비율 보정
            pred = tf.where(ratio >= 5.5, pred * 1.15, pred)
            pred = tf.where((ratio >= 5.0) & (ratio < 5.5), pred * 1.10, pred)
            pred = tf.where((ratio >= 4.5) & (ratio < 5.0), pred * 1.08, pred)
            pred = tf.where((ratio >= 4.0) & (ratio < 4.5), pred * 1.05, pred)
            
            # 황금 패턴
            golden = (m14b >= 350) & (m10a < 70)
            pred = tf.where(golden, pred * 1.2, pred)
            
            # 범위 제한
            pred = tf.clip_by_value(pred, 1200.0, 2000.0)
        
        return pred

# ============================================
# 모델 정의
# ============================================
class ModelsV6:
    
    @staticmethod
    def build_lstm_model(input_shape):
        """LSTM 모델"""
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.2),
            tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.2),
            tf.keras.layers.LSTM(64, dropout=0.2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, dtype='float32')  # Mixed Precision 대응
        ], name='LSTM_Model')
        return model
    
    @staticmethod
    def build_gru_model(input_shape):
        """GRU 모델"""
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.GRU(256, return_sequences=True, dropout=0.15),
            tf.keras.layers.GRU(128, return_sequences=True, dropout=0.15),
            tf.keras.layers.GRU(64, dropout=0.15),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1, dtype='float32')
        ], name='GRU_Model')
        return model
    
    @staticmethod
    def build_cnn_lstm(input_shape):
        """CNN-LSTM 모델"""
        inputs = tf.keras.Input(shape=input_shape)
        
        # 멀티스케일 CNN
        convs = []
        for kernel_size in [3, 5, 7, 9]:
            conv = tf.keras.layers.Conv1D(128, kernel_size, activation='relu', padding='same')(inputs)
            conv = tf.keras.layers.BatchNormalization()(conv)
            convs.append(conv)
        
        concat = tf.keras.layers.Concatenate()(convs)
        
        lstm1 = tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.15)(concat)
        lstm2 = tf.keras.layers.LSTM(128, dropout=0.15)(lstm1)
        
        dense1 = tf.keras.layers.Dense(256, activation='relu')(lstm2)
        dropout1 = tf.keras.layers.Dropout(0.2)(dense1)
        dense2 = tf.keras.layers.Dense(128, activation='relu')(dropout1)
        dropout2 = tf.keras.layers.Dropout(0.15)(dense2)
        output = tf.keras.layers.Dense(1, dtype='float32')(dropout2)
        
        return tf.keras.Model(inputs=inputs, outputs=output, name='CNN_LSTM_Model')
    
    @staticmethod
    def build_spike_detector(input_shape):
        """Spike Detector"""
        inputs = tf.keras.Input(shape=input_shape)
        
        # CNN
        convs = []
        for kernel_size in [3, 5, 7]:
            conv = tf.keras.layers.Conv1D(96, kernel_size, activation='relu', padding='same')(inputs)
            convs.append(conv)
        
        concat = tf.keras.layers.Concatenate()(convs)
        norm = tf.keras.layers.BatchNormalization()(concat)
        
        # Attention
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=8, key_dim=36, dropout=0.15
        )(norm, norm)
        
        # Bidirectional LSTM
        lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.15)
        )(attention)
        
        # Pooling
        avg_pool = tf.keras.layers.GlobalAveragePooling1D()(lstm)
        max_pool = tf.keras.layers.GlobalMaxPooling1D()(lstm)
        pooled = tf.keras.layers.Concatenate()([avg_pool, max_pool])
        
        dense1 = tf.keras.layers.Dense(256, activation='relu')(pooled)
        dropout1 = tf.keras.layers.Dropout(0.2)(dense1)
        dense2 = tf.keras.layers.Dense(128, activation='relu')(dropout1)
        dropout2 = tf.keras.layers.Dropout(0.15)(dense2)
        
        output = tf.keras.layers.Dense(1, name='spike_value', dtype='float32')(dropout2)
        
        return tf.keras.Model(inputs=inputs, outputs=output, name='Spike_Detector')
    
    @staticmethod
    def build_rule_based_model(input_shape, m14_shape):
        """Rule-Based 모델"""
        time_input = tf.keras.Input(shape=input_shape, name='time_input')
        m14_input = tf.keras.Input(shape=(m14_shape,), name='m14_input')
        
        lstm1 = tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.15)(time_input)
        lstm2 = tf.keras.layers.LSTM(32, dropout=0.15)(lstm1)
        
        m14_dense1 = tf.keras.layers.Dense(32, activation='relu')(m14_input)
        m14_dense2 = tf.keras.layers.Dense(16, activation='relu')(m14_dense1)
        
        combined = tf.keras.layers.Concatenate()([lstm2, m14_dense2])
        
        dense1 = tf.keras.layers.Dense(128, activation='relu')(combined)
        dropout1 = tf.keras.layers.Dropout(0.2)(dense1)
        dense2 = tf.keras.layers.Dense(64, activation='relu')(dropout1)
        dropout2 = tf.keras.layers.Dropout(0.15)(dense2)
        
        prediction = tf.keras.layers.Dense(1, name='rule_pred', dtype='float32')(dropout2)
        corrected = M14RuleCorrection()([prediction, m14_input])
        
        return tf.keras.Model(
            inputs=[time_input, m14_input],
            outputs=corrected,
            name='Rule_Based_Model'
        )

# ============================================
# 데이터 로드
# ============================================
print("\n📦 데이터 로드 중...")

# 기존 NPZ 파일 로드
if os.path.exists(Config.SEQUENCE_FILE):
    print(f"✅ 기존 NPZ 파일 로드: {Config.SEQUENCE_FILE}")
    data = np.load(Config.SEQUENCE_FILE)
    X_train = data['X']
    y_train = data['y_original'] if 'y_original' in data else data['y']
    m14_train = data['m14_features']
    
    print(f"  X shape: {X_train.shape}")
    print(f"  y shape: {y_train.shape}")
    print(f"  m14 shape: {m14_train.shape}")
    print(f"  y 범위: {y_train.min():.0f} ~ {y_train.max():.0f}")
    
else:
    print("⚠️ NPZ 파일 없음 - 새로 생성 필요")
    exit(1)

# 타겟 분포 확인
print("\n📊 타겟 분포 (전체 데이터):")
for level in [1300, 1400, 1450, 1500, 1550]:
    count = (y_train >= level).sum()
    ratio = count / len(y_train) * 100
    print(f"  {level}+: {count:,}개 ({ratio:.1f}%)")

# ============================================
# 학습 함수 (체크포인트 포함)
# ============================================
def train_model_with_checkpoint(model_name, model_builder, X_data, y_data, m14_data=None, epochs=200):
    """체크포인트 기능이 있는 학습 함수"""
    
    print(f"\n{'='*60}")
    print(f"🎯 {model_name.upper()} 모델 학습")
    print(f"{'='*60}")
    
    checkpoint_manager = CheckpointManager()
    
    # 초기 에폭 확인
    initial_epoch = checkpoint_manager.get_initial_epoch(model_name)
    
    # 모델 로드 또는 생성
    model_path = os.path.join(Config.CHECKPOINT_DIR, f'{model_name}_model.keras')
    
    if os.path.exists(model_path) and initial_epoch > 0:
        print(f"🔄 저장된 모델 로드 (Epoch {initial_epoch}부터 재개)")
        custom_objects = {'WeightedLoss': WeightedLoss}
        if model_name == 'rule':
            custom_objects['M14RuleCorrection'] = M14RuleCorrection
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    else:
        print("🆕 새 모델 생성")
        if model_name == 'rule':
            model = model_builder(X_data.shape[1:], m14_data.shape[1])
        else:
            model = model_builder(X_data.shape[1:])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE * (0.5 if model_name == 'rule' else 1.0)),
            loss=WeightedLoss(),
            metrics=['mae']
        )
    
    # 이미 완료된 경우
    if initial_epoch >= epochs:
        print(f"✅ 이미 학습 완료 (Epoch {initial_epoch})")
        return model, None
    
    # 콜백 설정
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            patience=10,
            factor=0.5,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            save_best_only=False,  # 매번 저장
            save_freq='epoch',
            verbose=0
        ),
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: checkpoint_manager.save_state(
                model_name, epoch + 1, {'history': logs}
            ) if (epoch + 1) % Config.SAVE_FREQ == 0 else None
        )
    ]
    
    # 학습 시작
    print(f"📊 학습 시작 (Epoch {initial_epoch + 1}/{epochs})")
    print(f"   배치 사이즈: {Config.BATCH_SIZE}")
    print(f"   학습률: {Config.LEARNING_RATE}")
    
    start_time = time.time()
    
    # 데이터 준비
    if model_name == 'rule':
        train_data = [X_data, m14_data]
    else:
        train_data = X_data
    
    # 학습 실행
    history = model.fit(
        train_data, y_data,
        epochs=epochs,
        initial_epoch=initial_epoch,
        batch_size=Config.BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    elapsed_time = time.time() - start_time
    print(f"✅ {model_name} 완료 - 시간: {elapsed_time/60:.1f}분")
    print(f"   최종 Loss: {history.history['loss'][-1]:.4f}")
    
    # 최종 저장
    final_path = os.path.join(Config.MODEL_DIR, f'{model_name}_final.keras')
    model.save(final_path)
    print(f"💾 최종 모델 저장: {final_path}")
    
    return model, history

# ============================================
# 모든 모델 학습
# ============================================
print("\n" + "="*60)
print("🏋️ 전체 데이터 학습 시작 (체크포인트 지원)")
print("="*60)

models = {}
histories = {}

# 모델 목록
model_configs = [
    ('lstm', ModelsV6.build_lstm_model, 200),
    ('gru', ModelsV6.build_gru_model, 200),
    ('cnn_lstm', ModelsV6.build_cnn_lstm, 200),
    ('spike', ModelsV6.build_spike_detector, 200),
    ('rule', ModelsV6.build_rule_based_model, 100),
]

# 각 모델 학습
for model_name, model_builder, epochs in model_configs:
    if model_name == 'rule':
        model, history = train_model_with_checkpoint(
            model_name, model_builder, X_train, y_train, m14_train, epochs
        )
    else:
        model, history = train_model_with_checkpoint(
            model_name, model_builder, X_train, y_train, epochs=epochs
        )
    
    models[model_name] = model
    if history:
        histories[model_name] = history

# ============================================
# 앙상블 모델
# ============================================
print("\n" + "="*60)
print("🎯 앙상블 모델 생성")
print("="*60)

# 앙상블은 별도 처리 (개별 모델 학습 완료 후)
ensemble_path = os.path.join(Config.MODEL_DIR, 'ensemble_final.keras')

if not os.path.exists(ensemble_path):
    # 앙상블 입력
    time_input = tf.keras.Input(shape=X_train.shape[1:], name='ensemble_time')
    m14_input = tf.keras.Input(shape=(m14_train.shape[1],), name='ensemble_m14')
    
    # 각 모델 예측
    lstm_pred = models['lstm'](time_input)
    gru_pred = models['gru'](time_input)
    cnn_lstm_pred = models['cnn_lstm'](time_input)
    spike_pred = models['spike'](time_input)
    rule_pred = models['rule']([time_input, m14_input])
    
    # 가중 평균
    ensemble_pred = tf.keras.layers.Lambda(
        lambda x: 0.25*x[0] + 0.20*x[1] + 0.25*x[2] + 0.15*x[3] + 0.15*x[4]
    )([lstm_pred, gru_pred, cnn_lstm_pred, spike_pred, rule_pred])
    
    # 최종 보정
    final_pred = M14RuleCorrection()([ensemble_pred, m14_input])
    
    # 앙상블 모델
    ensemble_model = tf.keras.Model(
        inputs=[time_input, m14_input],
        outputs=final_pred,
        name='Ensemble_Model'
    )
    
    ensemble_model.compile(
        optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE * 0.3),
        loss=WeightedLoss(),
        metrics=['mae']
    )
    
    print("📊 앙상블 파인튜닝...")
    ensemble_hist = ensemble_model.fit(
        [X_train, m14_train], y_train,
        epochs=50,
        batch_size=Config.BATCH_SIZE,
        verbose=1
    )
    
    ensemble_model.save(ensemble_path)
    models['ensemble'] = ensemble_model
else:
    print("✅ 앙상블 모델 이미 존재")
    custom_objects = {'WeightedLoss': WeightedLoss, 'M14RuleCorrection': M14RuleCorrection}
    models['ensemble'] = tf.keras.models.load_model(ensemble_path, custom_objects=custom_objects)

# ============================================
# 평가
# ============================================
print("\n" + "="*60)
print("📊 모델 평가")
print("="*60)

# 샘플링 평가
sample_size = min(10000, len(X_train))
sample_idx = np.random.choice(len(X_train), sample_size, replace=False)
X_sample = X_train[sample_idx]
y_sample = y_train[sample_idx]
m14_sample = m14_train[sample_idx]

evaluation_results = {}

for name, model in models.items():
    # 예측
    if name == 'ensemble' or name == 'rule':
        pred = model.predict([X_sample, m14_sample], verbose=0).flatten()
    else:
        pred = model.predict(X_sample, verbose=0).flatten()
    
    # MAE
    mae = np.mean(np.abs(y_sample - pred))
    
    # 구간별 성능
    level_performance = {}
    for level in [1300, 1400, 1450, 1500]:
        mask = y_sample >= level
        if np.any(mask):
            recall = np.sum((pred >= level) & mask) / np.sum(mask)
            precision = np.sum((pred >= level) & mask) / max(np.sum(pred >= level), 1)
            f1 = 2 * (precision * recall) / max(precision + recall, 1e-7)
            level_mae = np.mean(np.abs(y_sample[mask] - pred[mask]))
            
            level_performance[level] = {
                'recall': recall,
                'precision': precision,
                'f1': f1,
                'mae': level_mae
            }
    
    evaluation_results[name] = {
        'overall_mae': mae,
        'levels': level_performance
    }
    
    print(f"\n🎯 {name.upper()}:")
    print(f"  전체 MAE: {mae:.2f}")

# 최고 모델
best_model = min(evaluation_results.keys(), 
                 key=lambda x: evaluation_results[x]['overall_mae'])
print(f"\n🏆 최고 성능: {best_model.upper()}")

# ============================================
# 최종 저장
# ============================================
print("\n💾 평가 결과 저장...")

with open(f"{Config.MODEL_DIR}evaluation_results.json", 'w') as f:
    json.dump(evaluation_results, f, indent=2, default=str)

print("\n" + "="*60)
print("🎉 학습 완료!")
print("="*60)
print(f"📁 모델 위치: {Config.MODEL_DIR}")
print(f"📁 체크포인트: {Config.CHECKPOINT_DIR}")
print("\n💡 중단 시 자동으로 저장되며,")
print("   다시 실행하면 이어서 학습합니다!")
print("="*60)

# 메모리 정리
tf.keras.backend.clear_session()
gc.collect()