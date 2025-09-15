"""
V6_GPU_통합_전체_최종.py
시퀀스 생성 + 5개 모델 + 앙상블 = 총 6개 모델
체크포인트 지원 - 중단/재시작 가능
파일: 20240201_TO_202507281705.CSV
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
print("🚀 V6 GPU 통합 시스템 - 전체 코드")
print(f"📦 TensorFlow: {tf.__version__}")
print("="*60)

# ============================================
# GPU 설정
# ============================================
def setup_gpu():
    """GPU 설정 및 확인"""
    print("\n🎮 GPU 환경 확인...")
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU 감지: {len(gpus)}개")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
            return True
        except Exception as e:
            print(f"⚠️ GPU 설정 오류: {e}")
            return False
    else:
        print("💻 CPU 모드로 실행")
        return False

has_gpu = setup_gpu()

# ============================================
# 설정
# ============================================
class Config:
    # 데이터 파일 (고정)
    DATA_FILE = '20240201_TO_202507281705.CSV'
    
    # 시퀀스 설정
    LOOKBACK = 100
    FORECAST = 10
    
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
    BATCH_SIZE = 128 if has_gpu else 64
    EPOCHS = 50
    LEARNING_RATE = 0.0005
    PATIENCE = 15
    
    # 저장 경로
    MODEL_DIR = './models_v6_gpu/'
    CHECKPOINT_DIR = './checkpoints_v6_gpu/'
    SEQUENCE_FILE = './sequences_v6_gpu.npz'

# 디렉토리 생성
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
        
        # 모델 저장
        for name, model in models.items():
            save_path = os.path.join(Config.CHECKPOINT_DIR, f'{name}_checkpoint.h5')
            model.save(save_path)
        
        # 히스토리 저장
        with open(os.path.join(Config.CHECKPOINT_DIR, 'history.pkl'), 'wb') as f:
            pickle.dump(history, f)
        
        # 상태 저장
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"\n💾 체크포인트 저장: {completed_models}")
    
    def load_state(self):
        """저장된 상태 로드"""
        if not os.path.exists(self.checkpoint_file):
            return None, {}, {}, {}
        
        with open(self.checkpoint_file, 'rb') as f:
            state = pickle.load(f)
        
        # 히스토리 로드
        history = {}
        history_path = os.path.join(Config.CHECKPOINT_DIR, 'history.pkl')
        if os.path.exists(history_path):
            with open(history_path, 'rb') as f:
                history = pickle.load(f)
        
        print(f"\n🔄 체크포인트 로드: {state['completed_models']}")
        print(f"   저장 시간: {state['timestamp']}")
        
        return state['completed_models'], {}, history, state.get('evaluation_results', {})

# ============================================
# 커스텀 손실 함수 및 레이어
# ============================================
class WeightedLoss(tf.keras.losses.Loss):
    """가중치 손실 함수"""
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        mae = tf.abs(y_true - y_pred)
        
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
        
        pred = tf.cast(pred, tf.float32)
        m14_features = tf.cast(m14_features, tf.float32)
        
        m14b = m14_features[:, 0:1]
        m10a = m14_features[:, 1:2]
        ratio = m14_features[:, 3:4]
        
        # 규칙 기반 보정
        pred = tf.where((m14b >= 500) & (ratio >= 7), tf.maximum(pred, 1700), pred)
        pred = tf.where((m14b >= 450) & (ratio >= 6), tf.maximum(pred, 1600), pred)
        pred = tf.where((m14b >= 400) & (ratio >= 5), tf.maximum(pred, 1500), pred)
        pred = tf.where(m14b >= 320, tf.maximum(pred, 1400), pred)
        
        # 황금 패턴
        golden = (m14b >= 350) & (m10a < 70)
        pred = tf.where(golden, pred * 1.1, pred)
        
        return pred

# ============================================
# V6 모델 정의 (5개 모델)
# ============================================
class ModelsV6:
    
    @staticmethod
    def build_lstm_model(input_shape):
        """1. LSTM 모델"""
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
            tf.keras.layers.LSTM(64, dropout=0.3, recurrent_dropout=0.3),
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
        """2. GRU 모델"""
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
        """3. CNN-LSTM 모델"""
        inputs = tf.keras.Input(shape=input_shape)
        
        # Multi-scale CNN
        conv1 = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
        conv2 = tf.keras.layers.Conv1D(64, 5, activation='relu', padding='same')(inputs)
        conv3 = tf.keras.layers.Conv1D(64, 7, activation='relu', padding='same')(inputs)
        
        # Concatenate
        concat = tf.keras.layers.Concatenate()([conv1, conv2, conv3])
        norm = tf.keras.layers.BatchNormalization()(concat)
        
        # LSTM
        lstm = tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.2)(norm)
        lstm2 = tf.keras.layers.LSTM(64, dropout=0.2)(lstm)
        
        # Dense
        dense1 = tf.keras.layers.Dense(128, activation='relu')(lstm2)
        dropout = tf.keras.layers.Dropout(0.3)(dense1)
        output = tf.keras.layers.Dense(1)(dropout)
        
        return tf.keras.Model(inputs=inputs, outputs=output, name='CNN_LSTM_Model')
    
    @staticmethod
    def build_spike_detector(input_shape):
        """4. Spike Detector 모델"""
        inputs = tf.keras.Input(shape=input_shape)
        
        # Multi-scale CNN
        conv1 = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
        conv2 = tf.keras.layers.Conv1D(64, 5, activation='relu', padding='same')(inputs)
        conv3 = tf.keras.layers.Conv1D(64, 7, activation='relu', padding='same')(inputs)
        
        # Concatenate
        concat = tf.keras.layers.Concatenate()([conv1, conv2, conv3])
        norm = tf.keras.layers.BatchNormalization()(concat)
        
        # Attention
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
        
        # Dual output
        regression_output = tf.keras.layers.Dense(1, name='spike_value')(dropout2)
        classification_output = tf.keras.layers.Dense(1, activation='sigmoid', name='spike_prob')(dropout2)
        
        return tf.keras.Model(
            inputs=inputs,
            outputs=[regression_output, classification_output],
            name='Spike_Detector'
        )
    
    @staticmethod
    def build_rule_based_model(input_shape, m14_shape):
        """5. Rule-Based 모델"""
        time_input = tf.keras.Input(shape=input_shape, name='time_input')
        m14_input = tf.keras.Input(shape=m14_shape, name='m14_input')
        
        # 시계열 처리
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
        
        return tf.keras.Model(
            inputs=[time_input, m14_input],
            outputs=corrected,
            name='Rule_Based_Model'
        )

# ============================================
# PART 1: 시퀀스 생성 또는 로드
# ============================================
print("\n" + "="*60)
print("📦 PART 1: 시퀀스 준비")
print("="*60)

if os.path.exists(Config.SEQUENCE_FILE):
    print(f"\n✅ 기존 시퀀스 파일 로드: {Config.SEQUENCE_FILE}")
    data = np.load(Config.SEQUENCE_FILE)
    X_scaled = data['X']
    y = data['y']
    m14_features = data['m14_features']
    print(f"  로드 완료: X{X_scaled.shape}, y{y.shape}")
    
else:
    print(f"\n⚠️ 시퀀스 파일 없음 - 새로 생성")
    print(f"📂 데이터 로딩: {Config.DATA_FILE}")
    
    df = pd.read_csv(Config.DATA_FILE)
    print(f"  ✅ {len(df):,}행 로드")
    
    # 특징 생성
    print("\n🔧 특징 엔지니어링...")
    
    # 기본 컬럼
    if 'TOTALCNT' in df.columns:
        df['current_value'] = df['TOTALCNT']
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df['current_value'] = df[numeric_cols[0]] if len(numeric_cols) > 0 else 0
    
    # M14 컬럼
    for col in ['M14AM10A', 'M14AM14B', 'M14AM16']:
        if col not in df.columns:
            df[col] = 0
            print(f"  ⚠️ {col} 없음 → 0으로 초기화")
    
    # 타겟 생성
    df['target'] = df['current_value'].shift(-Config.FORECAST)
    
    # 비율 특징
    df['ratio_14B_10A'] = df['M14AM14B'] / df['M14AM10A'].clip(lower=1)
    df['ratio_14B_16'] = df['M14AM14B'] / df['M14AM16'].clip(lower=1)
    
    # 시계열 특징
    for col in ['current_value', 'M14AM14B', 'M14AM10A', 'M14AM16']:
        if col in df.columns:
            df[f'{col}_change_5'] = df[col].diff(5)
            df[f'{col}_change_10'] = df[col].diff(10)
            df[f'{col}_ma_10'] = df[col].rolling(10, min_periods=1).mean()
            df[f'{col}_ma_30'] = df[col].rolling(30, min_periods=1).mean()
            df[f'{col}_std_10'] = df[col].rolling(10, min_periods=1).std()
    
    # M14AM10A 역패턴
    df['M10A_drop_5'] = -df['M14AM10A'].diff(5)
    df['M10A_drop_10'] = -df['M14AM10A'].diff(10)
    
    # 황금 패턴
    df['golden_pattern'] = ((df['M14AM14B'] >= 350) & (df['M14AM10A'] < 70)).astype(float)
    
    # 급증 신호
    for level, threshold in Config.M14B_THRESHOLDS.items():
        df[f'signal_{level}'] = (df['M14AM14B'] >= threshold).astype(float)
    
    # 비율 신호
    for level, threshold in Config.RATIO_THRESHOLDS.items():
        df[f'ratio_signal_{level}'] = (df['ratio_14B_10A'] >= threshold).astype(float)
    
    # NaN 처리
    df = df.fillna(0)
    df = df.dropna(subset=['target'])
    
    # 특징 선택
    exclude_cols = ['TIME', 'CURRTIME', 'TOTALCNT']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    print(f"  ✅ 특징: {len(feature_cols)}개")
    
    # 시퀀스 생성
    print("\n⚡ 시퀀스 생성 중...")
    data = df[feature_cols].values.astype(np.float32)
    n_samples = len(data) - Config.LOOKBACK - Config.FORECAST + 1
    
    X = np.zeros((n_samples, Config.LOOKBACK, len(feature_cols)), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.float32)
    
    for i in range(n_samples):
        X[i] = data[i:i+Config.LOOKBACK]
        y[i] = data[i+Config.LOOKBACK+Config.FORECAST-1, 0]  # target 컬럼
        
        if i % 50000 == 0:
            print(f"    진행: {i/n_samples*100:.1f}%")
    
    print(f"  ✅ 완료: X{X.shape}, y{y.shape}")
    
    # X만 스케일링 (y는 원본 유지!)
    print("\n📏 X 데이터만 스케일링...")
    X_scaled = np.zeros_like(X)
    scalers = {}
    
    for i in range(X.shape[2]):
        scaler = RobustScaler()
        feature = X[:, :, i].reshape(-1, 1)
        X_scaled[:, :, i] = scaler.fit_transform(feature).reshape(X[:, :, i].shape)
        scalers[f'feature_{i}'] = scaler
    
    # M14 특징 추출
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

# 타겟 분포 확인
print(f"\n📊 타겟 분포:")
for level in [1400, 1500, 1600, 1700]:
    count = (y >= level).sum()
    ratio = count / len(y) * 100
    print(f"  {level}+: {count:,}개 ({ratio:.1f}%)")

# ============================================
# PART 2: 데이터 분할
# ============================================
print("\n" + "="*60)
print("📊 데이터 분할")
print("="*60)

X_train, X_val, y_train, y_val, m14_train, m14_val = train_test_split(
    X_scaled, y, m14_features, test_size=0.2, random_state=42
)

y_spike_class = (y_train >= 1400).astype(float)
y_val_spike_class = (y_val >= 1400).astype(float)

print(f"  학습: {X_train.shape[0]:,}개")
print(f"  검증: {X_val.shape[0]:,}개")
print(f"  1400+ 학습 비율: {y_spike_class.mean():.1%}")
print(f"  1400+ 검증 비율: {y_val_spike_class.mean():.1%}")

# ============================================
# PART 3: 5개 모델 학습
# ============================================
print("\n" + "="*60)
print("🏋️ PART 3: 5개 모델 학습")
print("="*60)

# 체크포인트 매니저 초기화
checkpoint_manager = CheckpointManager()
completed_models, models, history, evaluation_results = checkpoint_manager.load_state()

if not completed_models:
    completed_models = []
    models = {}
    history = {}
    evaluation_results = {}

# 1. LSTM 모델
if 'lstm' not in completed_models:
    print("\n1️⃣ LSTM 모델 학습 (장기 시계열 패턴)")
    
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
            tf.keras.callbacks.EarlyStopping(
                patience=Config.PATIENCE,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                patience=5,
                factor=0.5
            )
        ],
        verbose=1
    )
    
    models['lstm'] = lstm_model
    history['lstm'] = lstm_history
    completed_models.append('lstm')
    checkpoint_manager.save_state(completed_models, models, history, evaluation_results)
else:
    print("\n1️⃣ LSTM - 이미 완료 ✓")

# 2. GRU 모델
if 'gru' not in completed_models:
    print("\n2️⃣ GRU 모델 학습 (단기 변동성)")
    
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
            tf.keras.callbacks.EarlyStopping(
                patience=Config.PATIENCE,
                restore_best_weights=True
            )
        ],
        verbose=1
    )
    
    models['gru'] = gru_model
    history['gru'] = gru_history
    completed_models.append('gru')
    checkpoint_manager.save_state(completed_models, models, history, evaluation_results)
else:
    print("\n2️⃣ GRU - 이미 완료 ✓")

# 3. CNN-LSTM 모델
if 'cnn_lstm' not in completed_models:
    print("\n3️⃣ CNN-LSTM 모델 학습 (복합 패턴 인식)")
    
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
            tf.keras.callbacks.EarlyStopping(
                patience=Config.PATIENCE,
                restore_best_weights=True
            )
        ],
        verbose=1
    )
    
    models['cnn_lstm'] = cnn_lstm_model
    history['cnn_lstm'] = cnn_lstm_history
    completed_models.append('cnn_lstm')
    checkpoint_manager.save_state(completed_models, models, history, evaluation_results)
else:
    print("\n3️⃣ CNN-LSTM - 이미 완료 ✓")

# 4. Spike Detector 모델
if 'spike' not in completed_models:
    print("\n4️⃣ Spike Detector 모델 학습 (이상치 감지)")
    
    spike_model = ModelsV6.build_spike_detector(X_train.shape[1:])
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
    
    spike_history = spike_model.fit(
        X_train, 
        [y_train, y_spike_class],
        validation_data=(X_val, [y_val, y_val_spike_class]),
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                patience=Config.PATIENCE,
                restore_best_weights=True
            )
        ],
        verbose=1
    )
    
    models['spike'] = spike_model
    history['spike'] = spike_history
    completed_models.append('spike')
    checkpoint_manager.save_state(completed_models, models, history, evaluation_results)
else:
    print("\n4️⃣ Spike Detector - 이미 완료 ✓")

# 5. Rule-Based 모델
if 'rule' not in completed_models:
    print("\n5️⃣ Rule-Based 모델 학습 (검증된 황금 패턴)")
    
    rule_model = ModelsV6.build_rule_based_model(X_train.shape[1:], m14_train.shape[1])
    rule_model.compile(
        optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE * 0.5),
        loss=WeightedLoss(),
        metrics=['mae']
    )
    
    rule_history = rule_model.fit(
        [X_train, m14_train], 
        y_train,
        validation_data=([X_val, m14_val], y_val),
        epochs=30,  # Rule-based는 빠르게 수렴
        batch_size=Config.BATCH_SIZE,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                patience=10,
                restore_best_weights=True
            )
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
# PART 4: 앙상블 모델
# ============================================
if 'ensemble' not in completed_models:
    print("\n" + "="*60)
    print("🎯 PART 4: 최종 앙상블 모델 구성")
    print("="*60)
    
    # 개별 모델 로드 (필요시)
    for name in ['lstm', 'gru', 'cnn_lstm', 'spike', 'rule']:
        if name not in models:
            checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f'{name}_checkpoint.h5')
            if os.path.exists(checkpoint_path):
                print(f"  모델 로드: {name}")
                custom_objects = {'WeightedLoss': WeightedLoss}
                if name == 'rule':
                    custom_objects['M14RuleCorrection'] = M14RuleCorrection
                models[name] = tf.keras.models.load_model(checkpoint_path, custom_objects=custom_objects)
    
    # 앙상블 모델 구성
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
    final_pred = M14RuleCorrection(name='ensemble_prediction')([ensemble_pred, m14_input])
    
    # spike_prob 출력
    spike_prob_output = tf.keras.layers.Lambda(lambda x: x, name='spike_probability')(spike_prob)
    
    # 앙상블 모델 정의
    ensemble_model = tf.keras.Model(
        inputs=[time_series_input, m14_input],
        outputs=[final_pred, spike_prob_output],
        name='Final_Ensemble_Model'
    )
    
    # 컴파일
    ensemble_model.compile(
        optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE * 0.5),
        loss={
            'ensemble_prediction': WeightedLoss(),
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
        epochs=20,
        batch_size=Config.BATCH_SIZE,
        verbose=1
    )
    
    models['ensemble'] = ensemble_model
    history['ensemble'] = ensemble_history
    completed_models.append('ensemble')
    checkpoint_manager.save_state(completed_models, models, history, evaluation_results)
    
    print("\n✅ 앙상블 모델 완성!")
else:
    print("\n🎯 앙상블 모델 - 이미 완료 ✓")

# ============================================
# PART 5: 평가
# ============================================
print("\n" + "="*60)
print("📊 PART 5: 모델 평가")
print("="*60)

for name, model in models.items():
    # 예측
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

# 최고 모델 선택
best_model = min(evaluation_results.keys(), key=lambda x: evaluation_results[x]['overall_mae'])
print(f"\n🏆 최고 성능: {best_model.upper()} 모델")
print(f"  MAE: {evaluation_results[best_model]['overall_mae']:.2f}")

# ============================================
# PART 6: 모델 저장
# ============================================
print("\n" + "="*60)
print("💾 PART 6: 모델 저장")
print("="*60)

for name, model in models.items():
    save_path = f"{Config.MODEL_DIR}{name}_v6_gpu.h5"
    model.save(save_path)
    print(f"  {name}_v6_gpu.h5 저장 완료")

# 평가 결과 저장
with open(f"{Config.MODEL_DIR}evaluation_results.json", 'w') as f:
    json.dump(evaluation_results, f, indent=2, default=str)

# 설정 저장
config_dict = {k: v for k, v in Config.__dict__.items() if not k.startswith('_')}
with open(f"{Config.MODEL_DIR}config.json", 'w') as f:
    json.dump(config_dict, f, indent=2)

print("\n  결과 파일 저장 완료")

# ============================================
# PART 7: 시각화
# ============================================
print("\n" + "="*60)
print("📈 PART 7: 결과 시각화")
print("="*60)

fig = plt.figure(figsize=(20, 12))

# 학습 곡선
for idx, (name, hist) in enumerate(history.items()):
    if idx < 6:
        ax = plt.subplot(3, 3, idx+1)
        
        if hasattr(hist, 'history'):
            if name == 'spike':
                loss = hist.history.get('spike_value_loss', [])
                val_loss = hist.history.get('val_spike_value_loss', [])
            elif name == 'ensemble':
                loss = hist.history.get('ensemble_prediction_loss', [])
                val_loss = hist.history.get('val_ensemble_prediction_loss', [])
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

# MAE 비교
ax = plt.subplot(3, 3, 7)
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

# 1400+ Recall 비교
ax = plt.subplot(3, 3, 8)
recalls_1400 = []
for m in model_names:
    if 1400 in evaluation_results[m]['levels']:
        recalls_1400.append(evaluation_results[m]['levels'][1400]['recall'] * 100)
    else:
        recalls_1400.append(0)

bars = ax.bar(model_names, recalls_1400, color=colors[:len(model_names)])
ax.set_title('1400+ Recall Comparison (%)')
ax.set_ylabel('Recall (%)')
ax.set_ylim(0, 105)

for bar, recall in zip(bars, recalls_1400):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{recall:.1f}%', ha='center', va='bottom')

# 성능 요약
ax = plt.subplot(3, 3, 9)
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

summary_text += f"\n6개 모델 앙상블 완성!"

ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
       fontsize=11, verticalalignment='top', fontfamily='monospace')

plt.suptitle('V6 GPU 앙상블 모델 성능 분석', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{Config.MODEL_DIR}training_results.png", dpi=100, bbox_inches='tight')
print("  training_results.png 저장 완료")
plt.show()

# ============================================
# 최종 출력
# ============================================
print("\n" + "="*60)
print("🎉 V6 GPU 통합 시스템 완료!")
print("="*60)
print(f"📁 모델 저장: {Config.MODEL_DIR}")
print(f"📁 체크포인트: {Config.CHECKPOINT_DIR}")
print(f"📂 시퀀스: {Config.SEQUENCE_FILE}")
print("\n📊 최종 성능:")
print(f"  최고 모델: {best_model.upper()}")
print(f"  전체 MAE: {evaluation_results[best_model]['overall_mae']:.2f}")

# GPU 정보
if has_gpu:
    gpus = tf.config.list_physical_devices('GPU')
    print(f"\n🎮 GPU 사용: {len(gpus)}개")
else:
    print("\n💻 CPU 모드로 실행됨")

print("\n✅ 6개 모델 (5개 + 앙상블) 준비 완료!")
print("💡 다음 단계: 실시간 예측 시스템 적용")

# 메모리 정리
tf.keras.backend.clear_session()
gc.collect()

print("="*60)