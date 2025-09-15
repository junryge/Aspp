"""
V6 개선판 - 반도체 물류 예측 시스템
- 100분 데이터 유지
- 클래스 불균형 해결
- 학습 파라미터 최적화
- 가중치 강화
- .keras 형식 사용 (TensorFlow 2.16+ 호환)
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import json
import os
import pickle
import warnings
from datetime import datetime
import gc
warnings.filterwarnings('ignore')

print("="*60)
print("🚀 V6 개선판 - 반도체 물류 예측 시스템")
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
            return True
        except Exception as e:
            print(f"⚠️ GPU 설정 오류: {e}")
            return False
    else:
        print("💻 CPU 모드로 실행")
        return False

has_gpu = setup_gpu()

# ============================================
# 개선된 설정
# ============================================
class Config:
    # 데이터 파일
    DATA_FILE = '/mnt/user-data/uploads/20250807_DATA.CSV'
    
    # 시퀀스 설정 (100분 유지!)
    LOOKBACK = 100  # ⭐ 과거 100분 데이터 사용
    FORECAST = 10   # ⭐ 10분 후 예측
    
    # M14 임계값 (개선된 값)
    M14B_THRESHOLDS = {
        1300: 250,  # 추가
        1400: 300,  # 조정
        1450: 350,  # 추가
        1500: 380,  # 조정
        1550: 420,  # 추가
    }
    
    # 비율 임계값 (개선된 값)
    RATIO_THRESHOLDS = {
        1300: 3.5,
        1400: 4.0,
        1450: 4.5,
        1500: 5.0,
        1550: 5.5,
    }
    
    # 학습 설정 (최적화)
    BATCH_SIZE = 64  # 줄임 (더 자주 업데이트)
    EPOCHS = 100  # 증가 (충분한 학습)
    LEARNING_RATE = 0.001  # 증가 (빠른 수렴)
    PATIENCE = 20  # 증가 (충분한 기회)
    
    # 저장 경로
    MODEL_DIR = './models_v6_improved/'
    CHECKPOINT_DIR = './checkpoints_v6_improved/'

# 디렉토리 생성
os.makedirs(Config.MODEL_DIR, exist_ok=True)
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

# ============================================
# 강화된 손실 함수
# ============================================
class ImprovedWeightedLoss(tf.keras.losses.Loss):
    """개선된 가중치 손실 함수"""
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        mae = tf.abs(y_true - y_pred)
        
        # 강화된 가중치
        weights = tf.ones_like(y_true)
        weights = tf.where(y_true >= 1550, 25.0, weights)  # 1550+ 매우 중요
        weights = tf.where((y_true >= 1500) & (y_true < 1550), 20.0, weights)
        weights = tf.where((y_true >= 1450) & (y_true < 1500), 15.0, weights)
        weights = tf.where((y_true >= 1400) & (y_true < 1450), 10.0, weights)
        weights = tf.where((y_true >= 1350) & (y_true < 1400), 5.0, weights)
        
        # 예측 오차가 큰 경우 추가 페널티
        large_error_penalty = tf.where(mae > 100, mae * 0.1, 0.0)
        
        return tf.reduce_mean(mae * weights + large_error_penalty)

# ============================================
# M14 규칙 보정 레이어 (개선)
# ============================================
class ImprovedM14RuleCorrection(tf.keras.layers.Layer):
    """개선된 M14 규칙 기반 보정"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs, training=None):
        pred, m14_features = inputs
        
        pred = tf.cast(pred, tf.float32)
        m14_features = tf.cast(m14_features, tf.float32)
        
        m14b = m14_features[:, 0:1]  # M14AM14B
        m10a = m14_features[:, 1:2]  # M14AM10A
        m16 = m14_features[:, 2:3]   # M14AM16
        ratio = tf.where(m10a > 0, m14b / (m10a + 1e-7), 0.0)
        
        # 세밀한 규칙 적용
        corrections = []
        
        # 기본 임계값 규칙
        corrections.append(tf.where(m14b >= 420, tf.maximum(pred, 1550.0), pred))
        corrections.append(tf.where(m14b >= 380, tf.maximum(pred, 1500.0), pred))
        corrections.append(tf.where(m14b >= 350, tf.maximum(pred, 1450.0), pred))
        corrections.append(tf.where(m14b >= 300, tf.maximum(pred, 1400.0), pred))
        
        # 비율 기반 규칙
        corrections.append(tf.where(ratio >= 5.5, pred * 1.15, pred))
        corrections.append(tf.where(ratio >= 5.0, pred * 1.10, pred))
        corrections.append(tf.where(ratio >= 4.5, pred * 1.08, pred))
        corrections.append(tf.where(ratio >= 4.0, pred * 1.05, pred))
        
        # 황금 패턴
        golden = (m14b >= 350) & (m10a < 70)
        corrections.append(tf.where(golden, pred * 1.2, pred))
        
        # 복합 조건
        spike_condition = (m14b >= 320) & (ratio >= 4.0) & (m16 < 90)
        corrections.append(tf.where(spike_condition, tf.maximum(pred, 1450.0), pred))
        
        # 모든 보정 적용
        for correction in corrections:
            pred = correction
        
        # 최종 범위 제한
        pred = tf.clip_by_value(pred, 1200.0, 2000.0)
        
        return pred

# ============================================
# 개선된 모델 정의
# ============================================
class ImprovedModelsV6:
    
    @staticmethod
    def build_lstm_model(input_shape):
        """개선된 LSTM"""
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.3),  # 증가
            tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.3),
            tf.keras.layers.LSTM(64, dropout=0.3),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ], name='LSTM_Model')
        return model
    
    @staticmethod
    def build_gru_model(input_shape):
        """개선된 GRU"""
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.GRU(256, return_sequences=True, dropout=0.2),
            tf.keras.layers.GRU(128, return_sequences=True, dropout=0.2),
            tf.keras.layers.GRU(64, dropout=0.2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1)
        ], name='GRU_Model')
        return model
    
    @staticmethod
    def build_cnn_lstm(input_shape):
        """개선된 CNN-LSTM"""
        inputs = tf.keras.Input(shape=input_shape)
        
        # 멀티스케일 CNN
        convs = []
        for kernel_size in [3, 5, 7, 9]:
            conv = tf.keras.layers.Conv1D(
                128, kernel_size, 
                activation='relu', 
                padding='same'
            )(inputs)
            conv = tf.keras.layers.BatchNormalization()(conv)
            convs.append(conv)
        
        concat = tf.keras.layers.Concatenate()(convs)
        
        # LSTM 처리
        lstm1 = tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.2)(concat)
        lstm2 = tf.keras.layers.LSTM(128, dropout=0.2)(lstm1)
        
        # Dense layers
        dense1 = tf.keras.layers.Dense(256, activation='relu')(lstm2)
        dropout1 = tf.keras.layers.Dropout(0.3)(dense1)
        dense2 = tf.keras.layers.Dense(128, activation='relu')(dropout1)
        dropout2 = tf.keras.layers.Dropout(0.2)(dense2)
        output = tf.keras.layers.Dense(1)(dropout2)
        
        return tf.keras.Model(inputs=inputs, outputs=output, name='CNN_LSTM_Model')
    
    @staticmethod
    def build_spike_detector(input_shape):
        """개선된 Spike Detector"""
        inputs = tf.keras.Input(shape=input_shape)
        
        # 멀티스케일 CNN
        convs = []
        for kernel_size in [3, 5, 7]:
            conv = tf.keras.layers.Conv1D(
                96, kernel_size, 
                activation='relu', 
                padding='same'
            )(inputs)
            convs.append(conv)
        
        concat = tf.keras.layers.Concatenate()(convs)
        norm = tf.keras.layers.BatchNormalization()(concat)
        
        # 어텐션
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=8, key_dim=36, dropout=0.2
        )(norm, norm)
        
        # Bidirectional LSTM
        lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.2)
        )(attention)
        
        # Global pooling
        avg_pool = tf.keras.layers.GlobalAveragePooling1D()(lstm)
        max_pool = tf.keras.layers.GlobalMaxPooling1D()(lstm)
        pooled = tf.keras.layers.Concatenate()([avg_pool, max_pool])
        
        # Dense layers
        dense1 = tf.keras.layers.Dense(256, activation='relu')(pooled)
        dropout1 = tf.keras.layers.Dropout(0.3)(dense1)
        dense2 = tf.keras.layers.Dense(128, activation='relu')(dropout1)
        dropout2 = tf.keras.layers.Dropout(0.2)(dense2)
        
        # 출력
        regression_output = tf.keras.layers.Dense(1, name='spike_value')(dropout2)
        
        return tf.keras.Model(
            inputs=inputs,
            outputs=regression_output,
            name='Spike_Detector'
        )
    
    @staticmethod
    def build_rule_based_model(input_shape, m14_shape):
        """개선된 Rule-Based"""
        time_input = tf.keras.Input(shape=input_shape, name='time_input')
        m14_input = tf.keras.Input(shape=(m14_shape,), name='m14_input')
        
        # 시계열 처리
        lstm1 = tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2)(time_input)
        lstm2 = tf.keras.layers.LSTM(32, dropout=0.2)(lstm1)
        
        # M14 특징 처리 (더 깊게)
        m14_dense1 = tf.keras.layers.Dense(32, activation='relu')(m14_input)
        m14_dense2 = tf.keras.layers.Dense(16, activation='relu')(m14_dense1)
        
        # 결합
        combined = tf.keras.layers.Concatenate()([lstm2, m14_dense2])
        
        dense1 = tf.keras.layers.Dense(128, activation='relu')(combined)
        dropout1 = tf.keras.layers.Dropout(0.3)(dense1)
        dense2 = tf.keras.layers.Dense(64, activation='relu')(dropout1)
        dropout2 = tf.keras.layers.Dropout(0.2)(dense2)
        
        prediction = tf.keras.layers.Dense(1, name='rule_pred')(dropout2)
        
        # M14 규칙 적용
        corrected = ImprovedM14RuleCorrection()([prediction, m14_input])
        
        return tf.keras.Model(
            inputs=[time_input, m14_input],
            outputs=corrected,
            name='Rule_Based_Model'
        )

# ============================================
# 데이터 로드 및 전처리
# ============================================
print("\n📊 데이터 로드 중...")
df = pd.read_csv(Config.DATA_FILE)

# 필요한 컬럼 선택
feature_columns = ['M14AM10A', 'M14AM14B', 'M14AM16', 'M14AM14BSUM', 'TOTALCNT']
for col in feature_columns:
    if col not in df.columns:
        if col == 'M14AM14BSUM':
            df[col] = df['M14AM14B'] + df['M14AM10A']  # 합계 생성
        else:
            print(f"⚠️ {col} 컬럼 없음")

# 데이터 정리
df = df[feature_columns].copy()
for col in feature_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna()

print(f"데이터 shape: {df.shape}")
print(f"TOTALCNT 범위: {df['TOTALCNT'].min():.0f} ~ {df['TOTALCNT'].max():.0f}")
print(f"TOTALCNT 평균: {df['TOTALCNT'].mean():.0f}")

# ============================================
# 특징 엔지니어링
# ============================================
print("\n🔧 특징 엔지니어링...")

# 비율 특징
df['ratio_14B_10A'] = df['M14AM14B'] / (df['M14AM10A'] + 1)
df['ratio_14B_16'] = df['M14AM14B'] / (df['M14AM16'] + 1)
df['ratio_10A_16'] = df['M14AM10A'] / (df['M14AM16'] + 1)

# 시계열 특징
for col in feature_columns:
    # 변화량
    df[f'{col}_diff_1'] = df[col].diff(1)
    df[f'{col}_diff_5'] = df[col].diff(5)
    df[f'{col}_diff_10'] = df[col].diff(10)
    
    # 이동평균
    df[f'{col}_ma_5'] = df[col].rolling(5, min_periods=1).mean()
    df[f'{col}_ma_10'] = df[col].rolling(10, min_periods=1).mean()
    df[f'{col}_ma_20'] = df[col].rolling(20, min_periods=1).mean()
    
    # 표준편차
    df[f'{col}_std_5'] = df[col].rolling(5, min_periods=1).std()
    df[f'{col}_std_10'] = df[col].rolling(10, min_periods=1).std()

# 황금 패턴
df['golden_pattern'] = ((df['M14AM14B'] >= 350) & (df['M14AM10A'] < 70)).astype(float)

# 급증 신호
for level, threshold in Config.M14B_THRESHOLDS.items():
    df[f'signal_{level}'] = (df['M14AM14B'] >= threshold).astype(float)

for level, ratio in Config.RATIO_THRESHOLDS.items():
    df[f'ratio_signal_{level}'] = (df['ratio_14B_10A'] >= ratio).astype(float)

df = df.fillna(0)

print(f"특징 개수: {len(df.columns)}개")

# ============================================
# 시퀀스 생성 (100분!)
# ============================================
print("\n⚡ 시퀀스 생성 중... (100분 데이터)")

def create_sequences(data, lookback=100, forecast=10):
    """100분 시퀀스 생성"""
    X, y = [], []
    
    for i in range(len(data) - lookback - forecast):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback+forecast-1, df.columns.get_loc('TOTALCNT')])
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# 시퀀스 생성
data = df.values
X, y = create_sequences(data, Config.LOOKBACK, Config.FORECAST)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# M14 특징 추출 (현재 시점)
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

# ============================================
# 데이터 스케일링
# ============================================
print("\n📏 데이터 스케일링...")

# X 스케일링
X_scaled = np.zeros_like(X)
scalers = {}

for i in range(X.shape[2]):
    scaler = RobustScaler()
    feature = X[:, :, i].reshape(-1, 1)
    X_scaled[:, :, i] = scaler.fit_transform(feature).reshape(X[:, :, i].shape)
    scalers[f'feature_{i}'] = scaler

# M14 스케일링
m14_scaler = RobustScaler()
m14_features_scaled = m14_scaler.fit_transform(m14_features)

print("✅ 스케일링 완료")

# ============================================
# 데이터 분할
# ============================================
print("\n📊 데이터 분할...")

X_train, X_val, y_train, y_val, m14_train, m14_val = train_test_split(
    X_scaled, y, m14_features_scaled, 
    test_size=0.2, 
    random_state=42,
    stratify=(y >= 1400).astype(int)  # 층화 샘플링
)

print(f"학습: {X_train.shape[0]:,}개")
print(f"검증: {X_val.shape[0]:,}개")

# 타겟 분포 확인
for level in [1300, 1400, 1450, 1500]:
    train_count = (y_train >= level).sum()
    val_count = (y_val >= level).sum()
    print(f"{level}+: 학습 {train_count}개 ({train_count/len(y_train)*100:.1f}%), "
          f"검증 {val_count}개 ({val_count/len(y_val)*100:.1f}%)")

# ============================================
# 모델 학습
# ============================================
print("\n" + "="*60)
print("🏋️ 모델 학습 시작")
print("="*60)

models = {}
history = {}

# 학습 콜백
def get_callbacks(patience=15):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            patience=patience//2,
            factor=0.5,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            f'{Config.CHECKPOINT_DIR}{{epoch:02d}}-{{val_loss:.2f}}.keras',  # ✅ .h5 → .keras
            save_best_only=True,
            monitor='val_loss'
        )
    ]

# 1. LSTM
print("\n1️⃣ LSTM 모델 학습...")
lstm_model = ImprovedModelsV6.build_lstm_model(X_train.shape[1:])
lstm_model.compile(
    optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE),
    loss=ImprovedWeightedLoss(),
    metrics=['mae']
)
lstm_hist = lstm_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    callbacks=get_callbacks(),
    verbose=1
)
models['lstm'] = lstm_model
history['lstm'] = lstm_hist

# 2. GRU
print("\n2️⃣ GRU 모델 학습...")
gru_model = ImprovedModelsV6.build_gru_model(X_train.shape[1:])
gru_model.compile(
    optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE),
    loss=ImprovedWeightedLoss(),
    metrics=['mae']
)
gru_hist = gru_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    callbacks=get_callbacks(),
    verbose=1
)
models['gru'] = gru_model
history['gru'] = gru_hist

# 3. CNN-LSTM
print("\n3️⃣ CNN-LSTM 모델 학습...")
cnn_lstm_model = ImprovedModelsV6.build_cnn_lstm(X_train.shape[1:])
cnn_lstm_model.compile(
    optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE),
    loss=ImprovedWeightedLoss(),
    metrics=['mae']
)
cnn_lstm_hist = cnn_lstm_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    callbacks=get_callbacks(),
    verbose=1
)
models['cnn_lstm'] = cnn_lstm_model
history['cnn_lstm'] = cnn_lstm_hist

# 4. Spike Detector
print("\n4️⃣ Spike Detector 모델 학습...")
spike_model = ImprovedModelsV6.build_spike_detector(X_train.shape[1:])
spike_model.compile(
    optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE),
    loss=ImprovedWeightedLoss(),
    metrics=['mae']
)
spike_hist = spike_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    callbacks=get_callbacks(),
    verbose=1
)
models['spike'] = spike_model
history['spike'] = spike_hist

# 5. Rule-Based
print("\n5️⃣ Rule-Based 모델 학습...")
rule_model = ImprovedModelsV6.build_rule_based_model(X_train.shape[1:], m14_train.shape[1])
rule_model.compile(
    optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE * 0.5),
    loss=ImprovedWeightedLoss(),
    metrics=['mae']
)
rule_hist = rule_model.fit(
    [X_train, m14_train], y_train,
    validation_data=([X_val, m14_val], y_val),
    epochs=50,  # Rule은 빨리 수렴
    batch_size=Config.BATCH_SIZE,
    callbacks=get_callbacks(patience=10),
    verbose=1
)
models['rule'] = rule_model
history['rule'] = rule_hist

# ============================================
# 앙상블 모델
# ============================================
print("\n" + "="*60)
print("🎯 앙상블 모델 생성")
print("="*60)

# 앙상블 입력
time_input = tf.keras.Input(shape=X_train.shape[1:], name='ensemble_time')
m14_input = tf.keras.Input(shape=(m14_train.shape[1],), name='ensemble_m14')

# 각 모델 예측
lstm_pred = models['lstm'](time_input)
gru_pred = models['gru'](time_input)
cnn_lstm_pred = models['cnn_lstm'](time_input)
spike_pred = models['spike'](time_input)
rule_pred = models['rule']([time_input, m14_input])

# 가중 평균 (고정 가중치)
ensemble_pred = tf.keras.layers.Lambda(
    lambda x: 0.25*x[0] + 0.20*x[1] + 0.25*x[2] + 0.15*x[3] + 0.15*x[4]
)([lstm_pred, gru_pred, cnn_lstm_pred, spike_pred, rule_pred])

# 최종 보정
final_pred = ImprovedM14RuleCorrection()([ensemble_pred, m14_input])

# 앙상블 모델
ensemble_model = tf.keras.Model(
    inputs=[time_input, m14_input],
    outputs=final_pred,
    name='Ensemble_Model'
)

ensemble_model.compile(
    optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE * 0.3),
    loss=ImprovedWeightedLoss(),
    metrics=['mae']
)

print("📊 앙상블 파인튜닝...")
ensemble_hist = ensemble_model.fit(
    [X_train, m14_train], y_train,
    validation_data=([X_val, m14_val], y_val),
    epochs=30,
    batch_size=Config.BATCH_SIZE,
    callbacks=get_callbacks(patience=10),
    verbose=1
)

models['ensemble'] = ensemble_model
history['ensemble'] = ensemble_hist

# ============================================
# 평가
# ============================================
print("\n" + "="*60)
print("📊 모델 평가")
print("="*60)

evaluation_results = {}

for name, model in models.items():
    # 예측
    if name == 'ensemble' or name == 'rule':
        pred = model.predict([X_val, m14_val], verbose=0).flatten()
    else:
        pred = model.predict(X_val, verbose=0).flatten()
    
    # 전체 MAE
    mae = np.mean(np.abs(y_val - pred))
    
    # 구간별 성능
    level_performance = {}
    for level in [1300, 1400, 1450, 1500]:
        mask = y_val >= level
        if np.any(mask):
            # Recall: 실제 급증 중 예측 성공 비율
            recall = np.sum((pred >= level) & mask) / np.sum(mask)
            # Precision: 급증 예측 중 실제 급증 비율
            precision = np.sum((pred >= level) & mask) / max(np.sum(pred >= level), 1)
            # F1 Score
            f1 = 2 * (precision * recall) / max(precision + recall, 1e-7)
            # MAE
            level_mae = np.mean(np.abs(y_val[mask] - pred[mask]))
            
            level_performance[level] = {
                'recall': recall,
                'precision': precision,
                'f1': f1,
                'mae': level_mae,
                'count': np.sum(mask)
            }
    
    evaluation_results[name] = {
        'overall_mae': mae,
        'levels': level_performance
    }
    
    # 출력
    print(f"\n🎯 {name.upper()}:")
    print(f"  전체 MAE: {mae:.2f}")
    for level, perf in level_performance.items():
        print(f"  {level}+: Recall={perf['recall']:.2%}, "
              f"Precision={perf['precision']:.2%}, "
              f"F1={perf['f1']:.2%}, MAE={perf['mae']:.1f}")

# 최고 모델
best_model = min(evaluation_results.keys(), 
                 key=lambda x: evaluation_results[x]['overall_mae'])
print(f"\n🏆 최고 성능: {best_model.upper()} "
      f"(MAE: {evaluation_results[best_model]['overall_mae']:.2f})")

# ============================================
# 모델 저장
# ============================================
print("\n💾 모델 저장 중...")

for name, model in models.items():
    save_path = f"{Config.MODEL_DIR}{name}_final.keras"  # ✅ .h5 → .keras
    model.save(save_path)
    print(f"  ✅ {name} 저장: {save_path}")

# 평가 결과 저장
with open(f"{Config.MODEL_DIR}evaluation_results.json", 'w') as f:
    json.dump(evaluation_results, f, indent=2, default=str)
print(f"  ✅ 평가 결과 저장")

# 스케일러 저장
with open(f"{Config.MODEL_DIR}scalers.pkl", 'wb') as f:
    pickle.dump({
        'feature_scalers': scalers,
        'm14_scaler': m14_scaler
    }, f)
print(f"  ✅ 스케일러 저장")

# ============================================
# 최종 출력
# ============================================
print("\n" + "="*60)
print("🎉 V6 개선판 학습 완료!")
print("="*60)
print(f"📁 모델 저장 위치: {Config.MODEL_DIR}")
print(f"🏆 최고 모델: {best_model.upper()}")
print(f"📊 전체 MAE: {evaluation_results[best_model]['overall_mae']:.2f}")

# 100만개 데이터 체크
if len(y) >= 1000000:
    print("\n" + "="*60)
    print("🔔 알림: 100만개 이상 데이터 감지!")
    print("📊 Patch Time Series Transformer 적용 가능합니다.")
    print("="*60)

print("\n✅ 모든 작업 완료!")
print("="*60)

# 메모리 정리
tf.keras.backend.clear_session()
gc.collect()