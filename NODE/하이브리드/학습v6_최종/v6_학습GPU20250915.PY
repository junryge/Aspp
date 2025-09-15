"""
V6_GPU_통합_체크포인트_완전판.py
시퀀스 생성 + 5개 모델 앙상블 학습
중단 후 재시작 가능 (체크포인트 지원)
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json
import os
import pickle
import warnings
from datetime import datetime
import gc
warnings.filterwarnings('ignore')

print("="*60)
print("🚀 V6 GPU 통합 시스템 - 체크포인트 지원")
print(f"📦 TensorFlow: {tf.__version__}")
print("="*60)

# ============================================
# GPU 설정
# ============================================
def setup_gpu():
    """GPU 설정"""
    print("\n🎮 GPU 환경 확인...")
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU 감지: {len(gpus)}개")
        return True
    print("💻 CPU 모드")
    return False

has_gpu = setup_gpu()

# ============================================
# 통합 설정
# ============================================
class Config:
    DATA_FILE = '20240201_TO_202507281705.CSV'
    
    LOOKBACK = 100
    FORECAST = 10
    
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
    
    BATCH_SIZE = 128 if has_gpu else 64
    EPOCHS = 50
    LEARNING_RATE = 0.0005
    PATIENCE = 15
    
    MODEL_DIR = './models_v6_gpu/'
    CHECKPOINT_DIR = './checkpoints_v6_gpu/'
    SEQUENCE_FILE = './sequences_v6_gpu.npz'
    
os.makedirs(Config.MODEL_DIR, exist_ok=True)
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

# ============================================
# 체크포인트 관리자
# ============================================
class CheckpointManager:
    def __init__(self):
        self.checkpoint_file = os.path.join(Config.CHECKPOINT_DIR, 'training_state.pkl')
    
    def save_state(self, completed_models, models, history, evaluation_results):
        """학습 상태 저장"""
        state = {
            'completed_models': completed_models,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'evaluation_results': evaluation_results
        }
        
        for name, model in models.items():
            model.save(os.path.join(Config.CHECKPOINT_DIR, f'{name}_checkpoint.h5'))
        
        with open(os.path.join(Config.CHECKPOINT_DIR, 'history.pkl'), 'wb') as f:
            pickle.dump(history, f)
        
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"\n💾 체크포인트 저장: {completed_models}")
    
    def load_state(self):
        """저장된 상태 로드"""
        if not os.path.exists(self.checkpoint_file):
            return None, {}, {}, {}
        
        with open(self.checkpoint_file, 'rb') as f:
            state = pickle.load(f)
        
        history = {}
        if os.path.exists(os.path.join(Config.CHECKPOINT_DIR, 'history.pkl')):
            with open(os.path.join(Config.CHECKPOINT_DIR, 'history.pkl'), 'rb') as f:
                history = pickle.load(f)
        
        print(f"\n🔄 체크포인트 로드: {state['completed_models']}")
        print(f"   저장 시간: {state['timestamp']}")
        
        return state['completed_models'], {}, history, state.get('evaluation_results', {})

# ============================================
# PART 1: 시퀀스 생성 또는 로드
# ============================================
print("\n" + "="*60)
print("📦 PART 1: 시퀀스 준비")
print("="*60)

# 시퀀스 파일이 있는지 확인
if os.path.exists(Config.SEQUENCE_FILE):
    print(f"\n✅ 기존 시퀀스 파일 발견: {Config.SEQUENCE_FILE}")
    print("   로딩 중...")
    
    data = np.load(Config.SEQUENCE_FILE)
    X_scaled = data['X']
    y = data['y']
    m14_features = data['m14_features']
    
    print(f"  ✅ 로드 완료: X{X_scaled.shape}, y{y.shape}")
    
else:
    print("\n⚠️ 시퀀스 파일 없음 - 새로 생성")
    
    # 데이터 로드
    print(f"\n📂 데이터 로딩: {Config.DATA_FILE}")
    df = pd.read_csv(Config.DATA_FILE)
    print(f"  ✅ {len(df):,}행 로드")
    
    # 특징 생성
    print("\n🔧 특징 엔지니어링...")
    
    if 'TOTALCNT' in df.columns:
        df['current_value'] = df['TOTALCNT']
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df['current_value'] = df[numeric_cols[0]]
    
    for col in ['M14AM10A', 'M14AM14B', 'M14AM16']:
        if col not in df.columns:
            df[col] = 0
    
    df['target'] = df['current_value'].shift(-Config.FORECAST)
    
    # 비율 특징
    df['ratio_14B_10A'] = df['M14AM14B'] / df['M14AM10A'].clip(lower=1)
    
    # 시계열 특징
    for col in ['current_value', 'M14AM14B', 'M14AM10A', 'M14AM16']:
        if col in df.columns:
            df[f'{col}_change_5'] = df[col].diff(5)
            df[f'{col}_change_10'] = df[col].diff(10)
            df[f'{col}_ma_10'] = df[col].rolling(10, min_periods=1).mean()
            df[f'{col}_std_10'] = df[col].rolling(10, min_periods=1).std()
    
    # 황금 패턴
    df['golden_pattern'] = ((df['M14AM14B'] >= 350) & (df['M14AM10A'] < 70)).astype(float)
    
    # 급증 신호
    for level, threshold in Config.M14B_THRESHOLDS.items():
        df[f'signal_{level}'] = (df['M14AM14B'] >= threshold).astype(float)
    
    df = df.fillna(0)
    df = df.dropna(subset=['target'])
    
    exclude_cols = ['TIME', 'CURRTIME', 'TOTALCNT']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # 시퀀스 생성
    print("\n⚡ 시퀀스 생성 중...")
    data = df[feature_cols].values.astype(np.float32)
    n_samples = len(data) - Config.LOOKBACK - Config.FORECAST + 1
    
    X = np.zeros((n_samples, Config.LOOKBACK, len(feature_cols)), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.float32)
    
    for i in range(n_samples):
        X[i] = data[i:i+Config.LOOKBACK]
        y[i] = data[i+Config.LOOKBACK+Config.FORECAST-1, 0]
        if i % 50000 == 0:
            print(f"    진행: {i/n_samples*100:.1f}%")
    
    print(f"  ✅ 완료: X{X.shape}, y{y.shape}")
    
    # X만 스케일링
    X_scaled = np.zeros_like(X)
    for i in range(X.shape[2]):
        scaler = RobustScaler()
        feature = X[:, :, i].reshape(-1, 1)
        X_scaled[:, :, i] = scaler.fit_transform(feature).reshape(X[:, :, i].shape)
    
    # M14 특징
    m14_features = np.zeros((len(X), 4), dtype=np.float32)
    for i in range(len(X)):
        idx = i + Config.LOOKBACK
        if idx < len(df):
            m14_features[i] = [
                df['M14AM14B'].iloc[idx],
                df['M14AM10A'].iloc[idx],
                df['M14AM16'].iloc[idx],
                df['ratio_14B_10A'].iloc[idx]
            ]
    
    # 시퀀스 저장
    print("\n💾 시퀀스 저장...")
    np.savez_compressed(
        Config.SEQUENCE_FILE,
        X=X_scaled,
        y=y,
        m14_features=m14_features
    )
    print(f"  ✅ 저장: {Config.SEQUENCE_FILE}")

print(f"\n  타겟 분포:")
for level in [1400, 1500, 1600, 1700]:
    count = (y >= level).sum()
    ratio = count / len(y) * 100
    print(f"    {level}+: {count:,}개 ({ratio:.1f}%)")

# ============================================
# PART 2: 5개 모델 학습
# ============================================
print("\n" + "="*60)
print("🏋️ PART 2: 5개 모델 앙상블 학습")
print("="*60)

# 데이터 분할
X_train, X_val, y_train, y_val, m14_train, m14_val = train_test_split(
    X_scaled, y, m14_features, test_size=0.2, random_state=42
)

y_spike_class = (y_train >= 1400).astype(float)
y_val_spike_class = (y_val >= 1400).astype(float)

print(f"\n📊 데이터 분할:")
print(f"  학습: {X_train.shape[0]:,}개")
print(f"  검증: {X_val.shape[0]:,}개")

# ============================================
# 커스텀 레이어
# ============================================
class WeightedLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        mae = tf.abs(y_true - y_pred)
        weights = tf.ones_like(y_true)
        weights = tf.where(y_true >= 1700, 10.0, weights)
        weights = tf.where((y_true >= 1600) & (y_true < 1700), 8.0, weights)
        weights = tf.where((y_true >= 1500) & (y_true < 1600), 5.0, weights)
        weights = tf.where((y_true >= 1400) & (y_true < 1500), 3.0, weights)
        return tf.reduce_mean(mae * weights)

class M14RuleCorrection(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        pred, m14_features = inputs
        m14b = m14_features[:, 0:1]
        m10a = m14_features[:, 1:2]
        ratio = m14_features[:, 3:4]
        
        pred = tf.where((m14b >= 500) & (ratio >= 7), tf.maximum(pred, 1700), pred)
        pred = tf.where((m14b >= 450) & (ratio >= 6), tf.maximum(pred, 1600), pred)
        pred = tf.where((m14b >= 400) & (ratio >= 5), tf.maximum(pred, 1500), pred)
        pred = tf.where(m14b >= 320, tf.maximum(pred, 1400), pred)
        
        golden = (m14b >= 350) & (m10a < 70)
        pred = tf.where(golden, pred * 1.1, pred)
        
        return pred

# ============================================
# V6 5개 모델 정의
# ============================================
class ModelsV6:
    
    @staticmethod
    def build_lstm_model(input_shape):
        """1. LSTM"""
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
            tf.keras.layers.Dense(1)
        ], name='LSTM_Model')
        return model
    
    @staticmethod
    def build_enhanced_gru(input_shape):
        """2. GRU"""
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.GRU(128, return_sequences=True, dropout=0.2),
            tf.keras.layers.GRU(64, dropout=0.2),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ], name='GRU_Model')
        return model
    
    @staticmethod
    def build_cnn_lstm(input_shape):
        """3. CNN-LSTM"""
        inputs = tf.keras.Input(shape=input_shape)
        
        conv1 = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
        conv2 = tf.keras.layers.Conv1D(64, 5, activation='relu', padding='same')(inputs)
        concat = tf.keras.layers.Concatenate()([conv1, conv2])
        norm = tf.keras.layers.BatchNormalization()(concat)
        
        lstm = tf.keras.layers.LSTM(64, dropout=0.2)(norm)
        
        dense = tf.keras.layers.Dense(128, activation='relu')(lstm)
        dropout = tf.keras.layers.Dropout(0.3)(dense)
        output = tf.keras.layers.Dense(1)(dropout)
        
        return tf.keras.Model(inputs=inputs, outputs=output, name='CNN_LSTM_Model')
    
    @staticmethod
    def build_spike_detector(input_shape):
        """4. Spike Detector"""
        inputs = tf.keras.Input(shape=input_shape)
        
        conv1 = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
        conv2 = tf.keras.layers.Conv1D(64, 5, activation='relu', padding='same')(inputs)
        concat = tf.keras.layers.Concatenate()([conv1, conv2])
        
        norm = tf.keras.layers.BatchNormalization()(concat)
        
        # Attention
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=4, key_dim=32, dropout=0.2
        )(norm, norm)
        
        # BiLSTM
        lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2)
        )(attention)
        
        pooled = tf.keras.layers.GlobalAveragePooling1D()(lstm)
        
        dense1 = tf.keras.layers.Dense(128, activation='relu')(pooled)
        dropout1 = tf.keras.layers.Dropout(0.3)(dense1)
        dense2 = tf.keras.layers.Dense(64, activation='relu')(dropout1)
        dropout2 = tf.keras.layers.Dropout(0.2)(dense2)
        
        regression_output = tf.keras.layers.Dense(1, name='spike_value')(dropout2)
        classification_output = tf.keras.layers.Dense(1, activation='sigmoid', name='spike_prob')(dropout2)
        
        return tf.keras.Model(
            inputs=inputs,
            outputs=[regression_output, classification_output],
            name='Spike_Detector'
        )
    
    @staticmethod
    def build_rule_based_model(input_shape, m14_shape):
        """5. Rule-Based"""
        time_input = tf.keras.Input(shape=input_shape, name='time_input')
        m14_input = tf.keras.Input(shape=m14_shape, name='m14_input')
        
        lstm = tf.keras.layers.LSTM(32, dropout=0.2)(time_input)
        m14_dense = tf.keras.layers.Dense(16, activation='relu')(m14_input)
        
        combined = tf.keras.layers.Concatenate()([lstm, m14_dense])
        
        dense1 = tf.keras.layers.Dense(64, activation='relu')(combined)
        dropout = tf.keras.layers.Dropout(0.2)(dense1)
        dense2 = tf.keras.layers.Dense(32, activation='relu')(dropout)
        
        prediction = tf.keras.layers.Dense(1, name='rule_pred')(dense2)
        corrected = M14RuleCorrection()([prediction, m14_input])
        
        return tf.keras.Model(
            inputs=[time_input, m14_input],
            outputs=corrected,
            name='Rule_Based_Model'
        )

# 체크포인트 매니저
checkpoint_manager = CheckpointManager()
completed_models, models, history, evaluation_results = checkpoint_manager.load_state()

if not completed_models:
    completed_models = []
    models = {}
    history = {}
    evaluation_results = {}

# ============================================
# 5개 모델 학습 (체크포인트 지원)
# ============================================

# 1. LSTM
if 'lstm' not in completed_models:
    print("\n1️⃣ LSTM 모델 학습")
    
    lstm_model = ModelsV6.build_lstm_model(X_train.shape[1:])
    lstm_model.compile(
        optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE),
        loss=WeightedLoss(),
        metrics=['mae']
    )
    
    lstm_history = lstm_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=Config.PATIENCE, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
        ],
        verbose=1
    )
    
    models['lstm'] = lstm_model
    history['lstm'] = lstm_history
    completed_models.append('lstm')
    checkpoint_manager.save_state(completed_models, models, history, evaluation_results)
else:
    print("\n1️⃣ LSTM - 이미 완료 ✓")

# 2. GRU
if 'gru' not in completed_models:
    print("\n2️⃣ GRU 모델 학습")
    
    gru_model = ModelsV6.build_enhanced_gru(X_train.shape[1:])
    gru_model.compile(
        optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE),
        loss=WeightedLoss(),
        metrics=['mae']
    )
    
    gru_history = gru_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=Config.PATIENCE, restore_best_weights=True)
        ],
        verbose=1
    )
    
    models['gru'] = gru_model
    history['gru'] = gru_history
    completed_models.append('gru')
    checkpoint_manager.save_state(completed_models, models, history, evaluation_results)
else:
    print("\n2️⃣ GRU - 이미 완료 ✓")

# 3. CNN-LSTM
if 'cnn_lstm' not in completed_models:
    print("\n3️⃣ CNN-LSTM 모델 학습")
    
    cnn_lstm_model = ModelsV6.build_cnn_lstm(X_train.shape[1:])
    cnn_lstm_model.compile(
        optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE),
        loss=WeightedLoss(),
        metrics=['mae']
    )
    
    cnn_lstm_history = cnn_lstm_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=Config.PATIENCE, restore_best_weights=True)
        ],
        verbose=1
    )
    
    models['cnn_lstm'] = cnn_lstm_model
    history['cnn_lstm'] = cnn_lstm_history
    completed_models.append('cnn_lstm')
    checkpoint_manager.save_state(completed_models, models, history, evaluation_results)
else:
    print("\n3️⃣ CNN-LSTM - 이미 완료 ✓")

# 4. Spike Detector
if 'spike' not in completed_models:
    print("\n4️⃣ Spike Detector 모델 학습")
    
    spike_model = ModelsV6.build_spike_detector(X_train.shape[1:])
    spike_model.compile(
        optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE),
        loss={
            'spike_value': WeightedLoss(),
            'spike_prob': 'binary_crossentropy'
        },
        loss_weights={'spike_value': 1.0, 'spike_prob': 0.3},
        metrics=['mae']
    )
    
    spike_history = spike_model.fit(
        X_train, [y_train, y_spike_class],
        validation_data=(X_val, [y_val, y_val_spike_class]),
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=Config.PATIENCE, restore_best_weights=True)
        ],
        verbose=1
    )
    
    models['spike'] = spike_model
    history['spike'] = spike_history
    completed_models.append('spike')
    checkpoint_manager.save_state(completed_models, models, history, evaluation_results)
else:
    print("\n4️⃣ Spike Detector - 이미 완료 ✓")

# 5. Rule-Based
if 'rule' not in completed_models:
    print("\n5️⃣ Rule-Based 모델 학습")
    
    rule_model = ModelsV6.build_rule_based_model(X_train.shape[1:], m14_train.shape[1])
    rule_model.compile(
        optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE * 0.5),
        loss=WeightedLoss(),
        metrics=['mae']
    )
    
    rule_history = rule_model.fit(
        [X_train, m14_train], y_train,
        validation_data=([X_val, m14_val], y_val),
        epochs=30,
        batch_size=Config.BATCH_SIZE,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        ],
        verbose=1
    )
    
    models['rule'] = rule_model
    history['rule'] = rule_history
    completed_models.append('rule')
    checkpoint_manager.save_state(completed_models, models, history, evaluation_results)
else:
    print("\n5️⃣ Rule-Based - 이미 완료 ✓")

# ============================================
# 평가
# ============================================
print("\n📊 모델 평가")

for name, model in models.items():
    if name == 'spike':
        pred = model.predict(X_val, verbose=0)[0].flatten()
    elif name == 'rule':
        pred = model.predict([X_val, m14_val], verbose=0).flatten()
    else:
        pred = model.predict(X_val, verbose=0).flatten()
    
    mae = np.mean(np.abs(y_val - pred))
    
    level_performance = {}
    for level in [1400, 1500, 1600, 1700]:
        mask = y_val >= level
        if np.any(mask):
            recall = np.sum((pred >= level) & mask) / np.sum(mask)
            level_mae = np.mean(np.abs(y_val[mask] - pred[mask]))
            level_performance[level] = {'recall': recall, 'mae': level_mae, 'count': np.sum(mask)}
    
    evaluation_results[name] = {'overall_mae': mae, 'levels': level_performance}
    
    print(f"\n🎯 {name.upper()}:")
    print(f"  전체 MAE: {mae:.2f}")
    for level, perf in level_performance.items():
        print(f"  {level}+: Recall={perf['recall']:.2%}, MAE={perf['mae']:.1f}")

best_model = min(evaluation_results.keys(), key=lambda x: evaluation_results[x]['overall_mae'])
print(f"\n🏆 최고 성능: {best_model.upper()} (MAE: {evaluation_results[best_model]['overall_mae']:.2f})")

# ============================================
# 모델 저장
# ============================================
print("\n💾 모델 저장...")

for name, model in models.items():
    model.save(f"{Config.MODEL_DIR}{name}_v6_gpu.h5")
    print(f"  {name}_v6_gpu.h5 저장 완료")

with open(f"{Config.MODEL_DIR}evaluation_results.json", 'w') as f:
    json.dump(evaluation_results, f, indent=2, default=str)

print("\n" + "="*60)
print("🎉 V6 GPU 통합 시스템 완료!")
print("="*60)
print(f"📁 모델: {Config.MODEL_DIR}")
print(f"📁 체크포인트: {Config.CHECKPOINT_DIR}")
print("\n✅ 5개 모델 앙상블 준비 완료!")