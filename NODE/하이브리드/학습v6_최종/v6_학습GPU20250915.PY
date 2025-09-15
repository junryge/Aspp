"""
V6_GPU_통합_최종_전체.py
시퀀스 생성 + 5개 모델 + 앙상블 = 총 6개 모델
스케일링 완전 처리 + 체크포인트 지원
파일: 20240201_TO_202507281705.CSV
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import RobustScaler, StandardScaler
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
print("🚀 V6 GPU 통합 시스템 - 최종 전체 코드")
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
            
            # GPU 테스트
            with tf.device('/GPU:0'):
                test = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                result = tf.matmul(test, test)
                print(f"  GPU 테스트: ✅ (결과: {result.numpy()[0][0]:.0f})")
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
    
    # 스케일링 옵션
    USE_SCALED_Y = False  # False: 원본 y 사용, True: 스케일된 y 사용
    
    # 저장 경로
    MODEL_DIR = './models_v6_gpu/'
    CHECKPOINT_DIR = './checkpoints_v6_gpu/'
    SEQUENCE_FILE = './sequences_v6_gpu.npz'
    SCALER_FILE = './scalers_v6_gpu.pkl'

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
        
        for name, model in models.items():
            save_path = os.path.join(Config.CHECKPOINT_DIR, f'{name}_checkpoint.h5')
            try:
                model.save(save_path)
            except:
                print(f"  ⚠️ {name} 모델 저장 실패")
        
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
        
        # 원본 스케일 기준 가중치
        weights = tf.ones_like(y_true)
        if not Config.USE_SCALED_Y:
            # 원본 스케일 사용시
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
        
        # 원본 스케일 기준 규칙
        if not Config.USE_SCALED_Y:
            pred = tf.where((m14b >= 500) & (ratio >= 7), tf.maximum(pred, 1700), pred)
            pred = tf.where((m14b >= 450) & (ratio >= 6), tf.maximum(pred, 1600), pred)
            pred = tf.where((m14b >= 400) & (ratio >= 5), tf.maximum(pred, 1500), pred)
            pred = tf.where(m14b >= 320, tf.maximum(pred, 1400), pred)
            
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
        
        conv1 = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
        conv2 = tf.keras.layers.Conv1D(64, 5, activation='relu', padding='same')(inputs)
        conv3 = tf.keras.layers.Conv1D(64, 7, activation='relu', padding='same')(inputs)
        
        concat = tf.keras.layers.Concatenate()([conv1, conv2, conv3])
        norm = tf.keras.layers.BatchNormalization()(concat)
        
        lstm = tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.2)(norm)
        lstm2 = tf.keras.layers.LSTM(64, dropout=0.2)(lstm)
        
        dense1 = tf.keras.layers.Dense(128, activation='relu')(lstm2)
        dropout = tf.keras.layers.Dropout(0.3)(dense1)
        output = tf.keras.layers.Dense(1)(dropout)
        
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

# ============================================
# PART 1: 시퀀스 생성 또는 로드
# ============================================
print("\n" + "="*60)
print("📦 PART 1: 시퀀스 준비")
print("="*60)

# 스케일러 변수
scalers = None
y_scaler = None
m14_scaler = None

if os.path.exists(Config.SEQUENCE_FILE):
    print(f"\n✅ 기존 시퀀스 파일 로드: {Config.SEQUENCE_FILE}")
    data = np.load(Config.SEQUENCE_FILE)
    X_scaled = data['X']
    y_original = data['y_original'] if 'y_original' in data else data['y']
    y_scaled = data['y_scaled'] if 'y_scaled' in data else None
    m14_features = data['m14_features']
    m14_features_scaled = data['m14_features_scaled'] if 'm14_features_scaled' in data else m14_features
    
    # 스케일러 로드
    if os.path.exists(Config.SCALER_FILE):
        with open(Config.SCALER_FILE, 'rb') as f:
            scaler_data = pickle.load(f)
            scalers = scaler_data.get('feature_scalers')
            y_scaler = scaler_data.get('y_scaler')
            m14_scaler = scaler_data.get('m14_scaler')
    
    print(f"  X shape: {X_scaled.shape}")
    print(f"  y 원본 범위: {y_original.min():.0f} ~ {y_original.max():.0f}")
    if y_scaled is not None:
        print(f"  y 스케일 범위: {y_scaled.min():.2f} ~ {y_scaled.max():.2f}")
    
else:
    print(f"\n⚠️ 시퀀스 파일 없음 - 새로 생성")
    print(f"📂 데이터 로딩: {Config.DATA_FILE}")
    
    df = pd.read_csv(Config.DATA_FILE)
    print(f"  ✅ {len(df):,}행 로드")
    
    # 특징 생성
    print("\n🔧 특징 엔지니어링...")
    
    if 'TOTALCNT' in df.columns:
        df['current_value'] = df['TOTALCNT']
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df['current_value'] = df[numeric_cols[0]] if len(numeric_cols) > 0 else 0
    
    for col in ['M14AM10A', 'M14AM14B', 'M14AM16']:
        if col not in df.columns:
            df[col] = 0
            print(f"  ⚠️ {col} 없음 → 0으로 초기화")
    
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
    
    df['M10A_drop_5'] = -df['M14AM10A'].diff(5)
    df['M10A_drop_10'] = -df['M14AM10A'].diff(10)
    
    df['golden_pattern'] = ((df['M14AM14B'] >= 350) & (df['M14AM10A'] < 70)).astype(float)
    
    for level, threshold in Config.M14B_THRESHOLDS.items():
        df[f'signal_{level}'] = (df['M14AM14B'] >= threshold).astype(float)
    
    for level, threshold in Config.RATIO_THRESHOLDS.items():
        df[f'ratio_signal_{level}'] = (df['ratio_14B_10A'] >= threshold).astype(float)
    
    df = df.fillna(0)
    df = df.dropna(subset=['target'])
    
    exclude_cols = ['TIME', 'CURRTIME', 'TOTALCNT']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    print(f"  ✅ 특징: {len(feature_cols)}개")
    
    # 시퀀스 생성
    print("\n⚡ 시퀀스 생성 중...")
    data = df[feature_cols].values.astype(np.float32)
    n_samples = len(data) - Config.LOOKBACK - Config.FORECAST + 1
    
    X = np.zeros((n_samples, Config.LOOKBACK, len(feature_cols)), dtype=np.float32)
    y_original = np.zeros(n_samples, dtype=np.float32)
    
    for i in range(n_samples):
        X[i] = data[i:i+Config.LOOKBACK]
        y_original[i] = data[i+Config.LOOKBACK+Config.FORECAST-1, 0]  # target 컬럼
        
        if i % 50000 == 0:
            print(f"    진행: {i/n_samples*100:.1f}%")
    
    print(f"  ✅ 완료: X{X.shape}, y{y_original.shape}")
    
    # 스케일링
    print("\n📏 데이터 스케일링...")
    
    # X 스케일링
    X_scaled = np.zeros_like(X)
    scalers = {}
    
    for i in range(X.shape[2]):
        scaler = RobustScaler()
        feature = X[:, :, i].reshape(-1, 1)
        X_scaled[:, :, i] = scaler.fit_transform(feature).reshape(X[:, :, i].shape)
        scalers[f'feature_{i}'] = scaler
    
    # y 스케일링 (옵션)
    y_scaler = RobustScaler()
    y_scaled = y_scaler.fit_transform(y_original.reshape(-1, 1)).flatten()
    
    # M14 특징 추출 및 스케일링
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
    
    m14_scaler = RobustScaler()
    m14_features_scaled = m14_scaler.fit_transform(m14_features)
    
    print(f"  ✅ 스케일링 완료")
    print(f"  y 원본 범위: {y_original.min():.0f} ~ {y_original.max():.0f}")
    print(f"  y 스케일 범위: {y_scaled.min():.2f} ~ {y_scaled.max():.2f}")
    
    # 스케일러 저장
    print("\n💾 데이터 저장...")
    
    with open(Config.SCALER_FILE, 'wb') as f:
        pickle.dump({
            'feature_scalers': scalers,
            'y_scaler': y_scaler,
            'm14_scaler': m14_scaler
        }, f)
    
    np.savez_compressed(
        Config.SEQUENCE_FILE,
        X=X_scaled,
        y_original=y_original,
        y_scaled=y_scaled,
        m14_features=m14_features,
        m14_features_scaled=m14_features_scaled
    )
    
    print(f"  ✅ 시퀀스: {Config.SEQUENCE_FILE}")
    print(f"  ✅ 스케일러: {Config.SCALER_FILE}")

# 학습용 y 선택
if Config.USE_SCALED_Y and y_scaled is not None:
    y = y_scaled
    print("\n📊 스케일된 y 사용")
else:
    y = y_original
    print("\n📊 원본 y 사용")

# 타겟 분포 확인
print(f"\n📊 타겟 분포:")
for level in [1400, 1500, 1600, 1700]:
    if Config.USE_SCALED_Y:
        # 스케일된 경우 역변환하여 확인
        level_scaled = y_scaler.transform([[level]])[0][0] if y_scaler else level
        count = (y >= level_scaled).sum()
    else:
        count = (y >= level).sum()
    ratio = count / len(y) * 100
    print(f"  {level}+: {count:,}개 ({ratio:.1f}%)")

# ============================================
# PART 2: 데이터 분할
# ============================================
print("\n" + "="*60)
print("📊 PART 2: 데이터 분할")
print("="*60)

X_train, X_val, y_train, y_val, m14_train, m14_val = train_test_split(
    X_scaled, y, m14_features, test_size=0.2, random_state=42
)

# 원본 y도 분할 (평가용)
_, _, y_train_orig, y_val_orig, _, _ = train_test_split(
    X_scaled, y_original, m14_features, test_size=0.2, random_state=42
)

# 1400+ 분류 레이블
if Config.USE_SCALED_Y:
    threshold_1400 = y_scaler.transform([[1400]])[0][0] if y_scaler else 1400
    y_spike_class = (y_train >= threshold_1400).astype(float)
    y_val_spike_class = (y_val >= threshold_1400).astype(float)
else:
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

checkpoint_manager = CheckpointManager()
completed_models, models, history, evaluation_results = checkpoint_manager.load_state()

if not completed_models:
    completed_models = []
    models = {}
    history = {}
    evaluation_results = {}

# 1-5. 개별 모델 학습
model_configs = [
    ('lstm', ModelsV6.build_lstm_model, X_train.shape[1:], None),
    ('gru', ModelsV6.build_enhanced_gru, X_train.shape[1:], None),
    ('cnn_lstm', ModelsV6.build_cnn_lstm, X_train.shape[1:], None),
    ('spike', ModelsV6.build_spike_detector, X_train.shape[1:], None),
    ('rule', ModelsV6.build_rule_based_model, X_train.shape[1:], m14_train.shape[1])
]

for idx, (name, build_func, input_shape, m14_shape) in enumerate(model_configs):
    if name not in completed_models:
        print(f"\n{['1️⃣','2️⃣','3️⃣','4️⃣','5️⃣'][idx]} {name.upper()} 모델 학습")
        
        # 모델 빌드
        if m14_shape:
            model = build_func(input_shape, m14_shape)
        else:
            model = build_func(input_shape)
        
        # 컴파일
        if name == 'spike':
            model.compile(
                optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE),
                loss={'spike_value': WeightedLoss(), 'spike_prob': 'binary_crossentropy'},
                loss_weights={'spike_value': 1.0, 'spike_prob': 0.3},
                metrics=['mae']
            )
            train_data = X_train
            train_labels = [y_train, y_spike_class]
            val_data = (X_val, [y_val, y_val_spike_class])
        elif name == 'rule':
            model.compile(
                optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE * 0.5),
                loss=WeightedLoss(),
                metrics=['mae']
            )
            train_data = [X_train, m14_train]
            train_labels = y_train
            val_data = ([X_val, m14_val], y_val)
        else:
            model.compile(
                optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE),
                loss=WeightedLoss(),
                metrics=['mae']
            )
            train_data = X_train
            train_labels = y_train
            val_data = (X_val, y_val)
        
        # 학습
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=10 if name == 'rule' else Config.PATIENCE,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                patience=5,
                factor=0.5,
                min_lr=1e-7
            )
        ]
        
        hist = model.fit(
            train_data, train_labels,
            validation_data=val_data,
            epochs=30 if name == 'rule' else Config.EPOCHS,
            batch_size=Config.BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        models[name] = model
        history[name] = hist
        completed_models.append(name)
        checkpoint_manager.save_state(completed_models, models, history, evaluation_results)
    else:
        print(f"\n{['1️⃣','2️⃣','3️⃣','4️⃣','5️⃣'][idx]} {name.upper()} - 이미 완료 ✓")

# ============================================
# PART 4: 앙상블 모델
# ============================================
if 'ensemble' not in completed_models:
    print("\n" + "="*60)
    print("🎯 PART 4: 최종 앙상블 모델")
    print("="*60)
    
    # 모델 로드
    for name in ['lstm', 'gru', 'cnn_lstm', 'spike', 'rule']:
        if name not in models:
            checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f'{name}_checkpoint.h5')
            if os.path.exists(checkpoint_path):
                print(f"  모델 로드: {name}")
                custom_objects = {'WeightedLoss': WeightedLoss}
                if name == 'rule':
                    custom_objects['M14RuleCorrection'] = M14RuleCorrection
                models[name] = tf.keras.models.load_model(checkpoint_path, custom_objects=custom_objects)
    
    # 앙상블 구성
    time_input = tf.keras.Input(shape=X_train.shape[1:], name='ensemble_time')
    m14_input = tf.keras.Input(shape=m14_train.shape[1], name='ensemble_m14')
    
    lstm_pred = models['lstm'](time_input)
    gru_pred = models['gru'](time_input)
    cnn_lstm_pred = models['cnn_lstm'](time_input)
    spike_pred, spike_prob = models['spike'](time_input)
    rule_pred = models['rule']([time_input, m14_input])
    
    # 동적 가중치
    weight_dense = tf.keras.layers.Dense(32, activation='relu')(m14_input)
    weight_dense = tf.keras.layers.Dense(16, activation='relu')(weight_dense)
    weights = tf.keras.layers.Dense(5, activation='softmax')(weight_dense)
    
    w = [tf.keras.layers.Lambda(lambda x: x[:, i:i+1])(weights) for i in range(5)]
    
    weighted = [
        tf.keras.layers.Multiply()([pred, weight])
        for pred, weight in zip([lstm_pred, gru_pred, cnn_lstm_pred, spike_pred, rule_pred], w)
    ]
    
    ensemble_pred = tf.keras.layers.Add()(weighted)
    final_pred = M14RuleCorrection(name='ensemble_output')([ensemble_pred, m14_input])
    spike_prob_output = tf.keras.layers.Lambda(lambda x: x, name='spike_output')(spike_prob)
    
    ensemble_model = tf.keras.Model(
        inputs=[time_input, m14_input],
        outputs=[final_pred, spike_prob_output],
        name='Ensemble_Model'
    )
    
    ensemble_model.compile(
        optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE * 0.5),
        loss={'ensemble_output': WeightedLoss(), 'spike_output': 'binary_crossentropy'},
        loss_weights={'ensemble_output': 1.0, 'spike_output': 0.3},
        metrics=['mae']
    )
    
    print("\n📊 앙상블 파인튜닝...")
    ensemble_hist = ensemble_model.fit(
        [X_train, m14_train],
        [y_train, y_spike_class],
        validation_data=([X_val, m14_val], [y_val, y_val_spike_class]),
        epochs=20,
        batch_size=Config.BATCH_SIZE,
        verbose=1
    )
    
    models['ensemble'] = ensemble_model
    history['ensemble'] = ensemble_hist
    completed_models.append('ensemble')
    checkpoint_manager.save_state(completed_models, models, history, evaluation_results)
else:
    print("\n🎯 앙상블 - 이미 완료 ✓")

# ============================================
# PART 5: 평가 (원본 스케일)
# ============================================
print("\n" + "="*60)
print("📊 PART 5: 모델 평가 (원본 스케일)")
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
    
    # 역변환 (원본 스케일로)
    if Config.USE_SCALED_Y and y_scaler:
        pred_original = y_scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
        y_val_eval = y_val_orig
    else:
        pred_original = pred
        y_val_eval = y_val
    
    # 평가
    mae = np.mean(np.abs(y_val_eval - pred_original))
    
    level_performance = {}
    for level in [1400, 1500, 1600, 1700]:
        mask = y_val_eval >= level
        if np.any(mask):
            recall = np.sum((pred_original >= level) & mask) / np.sum(mask)
            level_mae = np.mean(np.abs(y_val_eval[mask] - pred_original[mask]))
            level_performance[level] = {
                'recall': recall,
                'mae': level_mae,
                'count': np.sum(mask)
            }
    
    evaluation_results[name] = {
        'overall_mae': mae,
        'levels': level_performance
    }
    
    print(f"\n🎯 {name.upper()}:")
    print(f"  전체 MAE: {mae:.2f}")
    for level, perf in level_performance.items():
        print(f"  {level}+: Recall={perf['recall']:.2%}, MAE={perf['mae']:.1f}")

best_model = min(evaluation_results.keys(), key=lambda x: evaluation_results[x]['overall_mae'])
print(f"\n🏆 최고 성능: {best_model.upper()} (MAE: {evaluation_results[best_model]['overall_mae']:.2f})")

# ============================================
# PART 6: 모델 저장
# ============================================
print("\n" + "="*60)
print("💾 PART 6: 최종 저장")
print("="*60)

for name, model in models.items():
    save_path = f"{Config.MODEL_DIR}{name}_final.h5"
    model.save(save_path)
    print(f"  {name}_final.h5 저장")

with open(f"{Config.MODEL_DIR}evaluation_results.json", 'w') as f:
    json.dump(evaluation_results, f, indent=2, default=str)

with open(f"{Config.MODEL_DIR}config.json", 'w') as f:
    config_dict = {k: v for k, v in Config.__dict__.items() if not k.startswith('_')}
    json.dump(config_dict, f, indent=2)

# ============================================
# 최종 출력
# ============================================
print("\n" + "="*60)
print("🎉 V6 GPU 통합 시스템 완료!")
print("="*60)
print(f"📁 모델: {Config.MODEL_DIR}")
print(f"📂 시퀀스: {Config.SEQUENCE_FILE}")
print(f"📂 스케일러: {Config.SCALER_FILE}")
print(f"\n✅ 6개 모델 준비 완료!")
print(f"🏆 최고 모델: {best_model.upper()} (MAE: {evaluation_results[best_model]['overall_mae']:.2f})")

if has_gpu:
    print(f"\n🎮 GPU 사용 완료")
else:
    print(f"\n💻 CPU 모드 완료")

# 메모리 정리
tf.keras.backend.clear_session()
gc.collect()

print("="*60)