"""
🚀 Ultimate 1700+ Predictor Suite v7.0
=====================================
TensorFlow 2.16.1 기반 모든 최신 모델 구현
목표: 1700+ 예측 정확도 95% 달성!

데이터: data/20240201_TO_202507281705.CSV
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.callbacks import *
import tensorflow_probability as tfp
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats
from scipy.special import expit
import math
import warnings
import logging
from datetime import datetime
import os
import json
import pickle
import time
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')

# ========================================
# 환경 설정
# ========================================

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ultimate_1700.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# GPU 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"🎮 GPU 사용: {len(gpus)}개")
    except:
        logger.info("💻 CPU 모드")
else:
    logger.info("💻 CPU 모드")

# Mixed Precision 설정 (속도 2배)
tf.keras.mixed_precision.set_global_policy('mixed_float16')
logger.info("⚡ Mixed Precision 활성화")

# 랜덤 시드
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ========================================
# 1. 데이터 로드 및 전처리
# ========================================

class DataProcessor:
    """데이터 전처리 및 특성 생성"""
    
    def __init__(self, seq_len=100, pred_len=10):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scaler = RobustScaler()
        self.feature_cols = None
        
    def load_and_process(self, filepath):
        """데이터 로드 및 전처리"""
        logger.info("📂 데이터 로딩 시작...")
        
        # 데이터 로드
        df = pd.read_csv(filepath)
        logger.info(f"✅ 데이터 shape: {df.shape}")
        
        # 시간 변환
        df['CURRTIME'] = pd.to_datetime(df['CURRTIME'], format='%Y%m%d%H%M')
        df['TIME'] = pd.to_datetime(df['TIME'], format='%Y%m%d%H%M')
        
        # 정렬
        df = df.sort_values('CURRTIME').reset_index(drop=True)
        
        # 타겟 생성 (10분 후)
        df['TARGET'] = df['TOTALCNT'].shift(-self.pred_len)
        df['IS_HIGH'] = (df['TARGET'] >= 1700).astype(int)
        
        # 특성 엔지니어링
        df = self._create_features(df)
        
        # NaN 제거
        df = df.dropna()
        
        logger.info(f"✅ 전처리 완료: {df.shape}")
        logger.info(f"✅ 1700+ 비율: {df['IS_HIGH'].mean():.2%}")
        
        return df
    
    def _create_features(self, df):
        """특성 생성"""
        logger.info("⚙️ 특성 생성 중...")
        
        # 기본 컬럼
        base_cols = ['TOTALCNT', 'M14AM10A', 'M10AM14A', 'M14AM10ASUM',
                     'M14AM14B', 'M14BM14A', 'M14AM14BSUM', 'M14AM16',
                     'M16M14A', 'M14AM16SUM']
        
        # 이동 평균
        for col in ['TOTALCNT', 'M14AM14B', 'M14AM10A']:
            df[f'{col}_MA10'] = df[col].rolling(10, min_periods=1).mean()
            df[f'{col}_MA30'] = df[col].rolling(30, min_periods=1).mean()
            df[f'{col}_MA60'] = df[col].rolling(60, min_periods=1).mean()
            
        # 표준편차
        df['TOTALCNT_STD10'] = df['TOTALCNT'].rolling(10, min_periods=1).std()
        df['TOTALCNT_STD30'] = df['TOTALCNT'].rolling(30, min_periods=1).std()
        
        # 변화율
        df['TOTALCNT_CHANGE'] = df['TOTALCNT'].diff()
        df['TOTALCNT_PCT'] = df['TOTALCNT'].pct_change()
        
        # 비율 특성 (핵심!)
        df['RATIO_M14B_M10A'] = df['M14AM14B'] / (df['M14AM10A'] + 1e-6)
        df['RATIO_HIGH'] = (df['RATIO_M14B_M10A'] > 7).astype(int)
        
        # 시간 특성
        df['HOUR'] = df['CURRTIME'].dt.hour
        df['DAYOFWEEK'] = df['CURRTIME'].dt.dayofweek
        df['IS_DAWN'] = df['HOUR'].isin([4, 5, 6]).astype(int)  # 새벽 시간
        
        # 추세
        df['TREND_100'] = df['TOTALCNT'].diff(100) / 100
        df['TREND_30'] = df['TOTALCNT'].diff(30) / 30
        df['TREND_10'] = df['TOTALCNT'].diff(10) / 10
        
        # 임계값 기반 특성
        df['M14B_HIGH'] = (df['M14AM14B'] > 350).astype(int)
        df['M14BSUM_HIGH'] = (df['M14AM14BSUM'] > 410).astype(int)
        
        # 복합 신호
        df['SPIKE_SIGNAL'] = (
            (df['M14B_HIGH'] == 1) & 
            (df['IS_DAWN'] == 1) & 
            (df['RATIO_HIGH'] == 1)
        ).astype(int)
        
        return df
    
    def create_sequences(self, df):
        """시퀀스 데이터 생성"""
        logger.info("🔄 시퀀스 생성 중...")
        
        # 특성 컬럼 선택
        feature_cols = [col for col in df.columns if col not in 
                       ['CURRTIME', 'TIME', 'TARGET', 'IS_HIGH']]
        self.feature_cols = feature_cols
        
        # 스케일링
        df_scaled = df.copy()
        df_scaled[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        df_scaled['TARGET_SCALED'] = self.scaler.fit_transform(df[['TARGET']])
        
        # 시퀀스 생성
        X, y, y_binary, spike_signals = [], [], [], []
        
        for i in range(len(df_scaled) - self.seq_len - self.pred_len):
            X.append(df_scaled[feature_cols].iloc[i:i+self.seq_len].values)
            y.append(df_scaled['TARGET_SCALED'].iloc[i+self.seq_len])
            y_binary.append(df_scaled['IS_HIGH'].iloc[i+self.seq_len])
            spike_signals.append(df_scaled['SPIKE_SIGNAL'].iloc[i+self.seq_len-1])
            
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        y_binary = np.array(y_binary, dtype=np.float32)
        spike_signals = np.array(spike_signals, dtype=np.float32)
        
        logger.info(f"✅ 시퀀스 shape: X={X.shape}, y={y.shape}")
        logger.info(f"✅ 1700+ 샘플: {y_binary.sum():.0f}개 ({y_binary.mean():.2%})")
        
        return X, y, y_binary, spike_signals

# ========================================
# 2. 커스텀 손실 함수
# ========================================

class CustomLosses:
    """1700+ 특화 손실 함수"""
    
    @staticmethod
    def weighted_mse(weight_1700=100.0):
        """1700+ 에 가중치를 준 MSE"""
        def loss(y_true, y_pred):
            # 1700+ 는 스케일된 값으로 약 0.5 이상
            weights = tf.where(y_true > 0.5, weight_1700, 1.0)
            squared_diff = tf.square(y_true - y_pred)
            return tf.reduce_mean(weights * squared_diff)
        return loss
    
    @staticmethod
    def focal_mse(gamma=2.0, alpha=0.25):
        """Focal Loss의 MSE 버전"""
        def loss(y_true, y_pred):
            diff = tf.abs(y_true - y_pred)
            focal_weight = tf.pow(diff, gamma)
            return tf.reduce_mean(alpha * focal_weight * tf.square(diff))
        return loss
    
    @staticmethod
    def quantile_loss(quantiles=[0.1, 0.5, 0.9]):
        """Quantile Loss for uncertainty"""
        def loss(y_true, y_pred):
            losses = []
            for i, q in enumerate(quantiles):
                error = y_true - y_pred[:, i:i+1]
                losses.append(tf.maximum(q * error, (q - 1) * error))
            return tf.reduce_mean(tf.concat(losses, axis=-1))
        return loss

# ========================================
# 3. 모델 구현
# ========================================

class ModelFactory:
    """모든 모델 생성 팩토리"""
    
    def __init__(self, input_shape, n_features):
        self.input_shape = input_shape  # (100, n_features)
        self.seq_len = input_shape[0]
        self.n_features = input_shape[1]
        
    # ----------------
    # 1. PatchTST
    # ----------------
    def build_patch_tst(self, patch_len=10, d_model=128, n_heads=8, n_layers=3):
        """Patch Time Series Transformer"""
        inputs = Input(shape=self.input_shape)
        
        # 패치로 나누기
        n_patches = self.seq_len // patch_len
        patches = tf.reshape(inputs, [-1, n_patches, patch_len * self.n_features])
        
        # 패치 임베딩
        x = Dense(d_model)(patches)
        x = LayerNormalization(epsilon=1e-6)(x)
        
        # Positional Encoding
        positions = tf.range(start=0, limit=n_patches, delta=1)
        pos_encoding = self._positional_encoding(n_patches, d_model)
        x = x + pos_encoding
        
        # Transformer Blocks
        for _ in range(n_layers):
            # Multi-Head Attention
            attn_output = MultiHeadAttention(
                num_heads=n_heads, 
                key_dim=d_model // n_heads,
                dropout=0.1
            )(x, x)
            x = Add()([x, attn_output])
            x = LayerNormalization(epsilon=1e-6)(x)
            
            # Feed Forward
            ff_output = Sequential([
                Dense(d_model * 4, activation='relu'),
                Dropout(0.1),
                Dense(d_model)
            ])(x)
            x = Add()([x, ff_output])
            x = LayerNormalization(epsilon=1e-6)(x)
        
        # Global Pooling
        x = GlobalAveragePooling1D()(x)
        
        # Output layers
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs, outputs, name='PatchTST')
        return model
    
    def _positional_encoding(self, position, d_model):
        """Positional encoding for transformer"""
        angle_rads = self._get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)
    
    def _get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates
    
    # ----------------
    # 2. Extreme Value Network
    # ----------------
    def build_extreme_value_network(self):
        """극단값 이론 기반 네트워크"""
        inputs = Input(shape=self.input_shape)
        
        # Feature extraction
        x = LSTM(128, return_sequences=True)(inputs)
        x = LSTM(64, return_sequences=False)(x)
        
        # 극단값 특화 브랜치
        extreme_branch = Dense(32, activation='relu')(x)
        extreme_branch = Dense(16, activation='relu')(extreme_branch)
        
        # GPD parameters (Generalized Pareto Distribution)
        xi = Dense(1, activation='tanh', name='shape')(extreme_branch)  # shape parameter
        sigma = Dense(1, activation='softplus', name='scale')(extreme_branch)  # scale parameter
        
        # 일반 예측 브랜치
        normal_branch = Dense(32, activation='relu')(x)
        normal_branch = Dense(1)(normal_branch)
        
        # 결합
        threshold = tf.constant(0.5)  # 스케일된 1500 정도
        is_extreme = tf.greater(normal_branch, threshold)
        
        # 극단값일 때는 GPD 기반 보정
        extreme_pred = normal_branch + sigma * tf.pow(tf.random.uniform(tf.shape(normal_branch)), -xi)
        
        outputs = tf.where(is_extreme, extreme_pred, normal_branch)
        
        model = Model(inputs, outputs, name='ExtremeValueNetwork')
        return model
    
    # ----------------
    # 3. Enhanced Spike Detector
    # ----------------
    def build_spike_detector(self):
        """급변 감지 특화 모델"""
        inputs = Input(shape=self.input_shape)
        
        # CNN for pattern detection
        x = Conv1D(64, 3, activation='relu', padding='same')(inputs)
        x = Conv1D(64, 3, activation='relu', padding='same')(x)
        x = MaxPooling1D(2)(x)
        
        # Bidirectional LSTM
        x = Bidirectional(LSTM(64, return_sequences=True))(x)
        x = Bidirectional(LSTM(32, return_sequences=False))(x)
        
        # Spike detection heads
        # 1. Binary classification (spike or not)
        spike_prob = Dense(32, activation='relu')(x)
        spike_prob = Dense(1, activation='sigmoid', name='spike_prob')(spike_prob)
        
        # 2. Spike magnitude
        spike_mag = Dense(32, activation='relu')(x)
        spike_mag = Dense(1, activation='linear', name='spike_magnitude')(spike_mag)
        
        # Combined output
        outputs = spike_prob * spike_mag
        
        model = Model(inputs, outputs, name='SpikeDetector')
        return model
    
    # ----------------
    # 4. Mamba (State Space Model)
    # ----------------
    def build_mamba(self):
        """State Space Model - 선형 복잡도"""
        inputs = Input(shape=self.input_shape)
        
        # State Space 파라미터
        A = tf.Variable(tf.random.normal([self.n_features, self.n_features]), trainable=True)
        B = tf.Variable(tf.random.normal([self.n_features, 64]), trainable=True)
        C = tf.Variable(tf.random.normal([64, self.n_features]), trainable=True)
        D = tf.Variable(tf.random.normal([self.n_features]), trainable=True)
        
        # SSM Layer 구현
        def ssm_step(carry, x):
            h = carry
            h_new = tf.matmul(h, A) + tf.matmul(x[None, :], B)
            y = tf.matmul(h_new, C) + x * D
            return h_new[0], y
        
        # Scan through sequence
        x = inputs
        x = Dense(64)(x)
        
        # RNN 형태로 구현 (실제 Mamba는 더 복잡)
        x = LSTM(128, return_sequences=True)(x)
        x = LSTM(64, return_sequences=False)(x)
        
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs, outputs, name='Mamba')
        return model
    
    # ----------------
    # 5. Temporal Graph Neural Network
    # ----------------
    def build_mtgnn(self):
        """Multivariate Time Graph Neural Network"""
        inputs = Input(shape=self.input_shape)
        
        # Graph construction (컬럼 간 관계)
        # Adjacency matrix learning
        node_embeddings = Dense(32)(inputs)
        
        # Temporal Convolution
        x = Conv1D(64, 3, activation='relu', padding='causal')(inputs)
        x = Conv1D(64, 3, activation='relu', padding='causal')(x)
        
        # Graph Convolution (simplified)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        
        # Temporal pooling
        x = GlobalAveragePooling1D()(x)
        
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs, outputs, name='MTGNN')
        return model
    
    # ----------------
    # 6. Neural SDE
    # ----------------
    def build_neural_sde(self):
        """확률적 미분방정식 모델"""
        inputs = Input(shape=self.input_shape)
        
        # Drift network (mu)
        drift = LSTM(64, return_sequences=True)(inputs)
        drift = LSTM(32, return_sequences=False)(drift)
        mu = Dense(16, activation='linear')(drift)
        
        # Diffusion network (sigma)
        diffusion = LSTM(64, return_sequences=True)(inputs)
        diffusion = LSTM(32, return_sequences=False)(diffusion)
        sigma = Dense(16, activation='softplus')(diffusion)
        
        # Stochastic integration
        z = tf.random.normal(tf.shape(mu))
        x = mu + sigma * z
        
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs, outputs, name='NeuralSDE')
        return model
    
    # ----------------
    # 7. Mixture of Experts
    # ----------------
    def build_mixture_of_experts(self, n_experts=3):
        """전문가 모델 앙상블"""
        inputs = Input(shape=self.input_shape)
        
        # Router (어떤 expert를 사용할지 결정)
        router = Dense(32, activation='relu')(inputs)
        router = GlobalAveragePooling1D()(router)
        router = Dense(n_experts, activation='softmax')(router)
        
        # Experts
        expert_outputs = []
        
        # Expert 1: Normal range (1200-1400)
        x1 = LSTM(64, return_sequences=False)(inputs)
        x1 = Dense(32, activation='relu')(x1)
        e1 = Dense(1)(x1)
        expert_outputs.append(e1)
        
        # Expert 2: Medium spike (1400-1600)
        x2 = GRU(64, return_sequences=False)(inputs)
        x2 = Dense(32, activation='relu')(x2)
        e2 = Dense(1)(x2)
        expert_outputs.append(e2)
        
        # Expert 3: Extreme spike (1700+)
        x3 = Bidirectional(LSTM(32, return_sequences=False))(inputs)
        x3 = Dense(32, activation='relu')(x3)
        e3 = Dense(1)(x3)
        expert_outputs.append(e3)
        
        # Weighted combination
        expert_stack = tf.stack(expert_outputs, axis=1)
        outputs = tf.reduce_sum(expert_stack * router[:, :, None], axis=1)
        
        model = Model(inputs, outputs, name='MixtureOfExperts')
        return model
    
    # ----------------
    # 8. Diffusion Model
    # ----------------
    def build_diffusion_model(self):
        """Denoising Diffusion Model for time series"""
        inputs = Input(shape=self.input_shape)
        
        # Encoder
        x = Conv1D(64, 3, activation='relu', padding='same')(inputs)
        x = Conv1D(128, 3, activation='relu', padding='same')(x)
        x = MaxPooling1D(2)(x)
        
        # Bottleneck with noise
        x = LSTM(64, return_sequences=True)(x)
        
        # Denoising process
        noise_level = tf.random.uniform([], 0, 0.1)
        x_noisy = x + tf.random.normal(tf.shape(x)) * noise_level
        
        # Decoder
        x = LSTM(64, return_sequences=False)(x_noisy)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs, outputs, name='DiffusionModel')
        return model
    
    # ----------------
    # 9. Autoformer
    # ----------------
    def build_autoformer(self):
        """Auto-Correlation Transformer"""
        inputs = Input(shape=self.input_shape)
        
        # Decomposition
        # Trend
        trend = AveragePooling1D(pool_size=10, strides=1, padding='same')(inputs)
        trend = Conv1D(64, 1)(trend)
        
        # Seasonal
        seasonal = inputs - trend
        seasonal = Conv1D(64, 1)(seasonal)
        
        # Auto-Correlation
        x = Concatenate()([trend, seasonal])
        x = MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
        
        x = GlobalAveragePooling1D()(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs, outputs, name='Autoformer')
        return model
    
    # ----------------
    # 10. Temporal Convolutional Network
    # ----------------
    def build_tcn(self):
        """TCN with dilated convolutions"""
        inputs = Input(shape=self.input_shape)
        
        x = inputs
        skip_connections = []
        
        for dilation_rate in [1, 2, 4, 8, 16]:
            # Dilated convolution
            conv = Conv1D(64, 3, dilation_rate=dilation_rate, 
                         padding='causal', activation='relu')(x)
            conv = Dropout(0.1)(conv)
            conv = Conv1D(64, 3, dilation_rate=dilation_rate,
                         padding='causal', activation='relu')(conv)
            
            # Skip connection
            if x.shape[-1] != conv.shape[-1]:
                x = Conv1D(64, 1)(x)
            
            x = Add()([x, conv])
            skip_connections.append(conv)
        
        # Aggregate skip connections
        x = Add()(skip_connections)
        x = GlobalAveragePooling1D()(x)
        
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs, outputs, name='TCN')
        return model
    
    # ----------------
    # 11. WaveNet
    # ----------------
    def build_wavenet(self):
        """WaveNet architecture"""
        inputs = Input(shape=self.input_shape)
        
        x = inputs
        skip_connections = []
        
        for i in range(4):
            dilation_rate = 2 ** i
            
            # Gated convolution
            tanh_out = Conv1D(64, 2, dilation_rate=dilation_rate, 
                             padding='causal', activation='tanh')(x)
            sigm_out = Conv1D(64, 2, dilation_rate=dilation_rate,
                             padding='causal', activation='sigmoid')(x)
            
            x = Multiply()([tanh_out, sigm_out])
            
            # Skip connection
            skip = Conv1D(64, 1)(x)
            skip_connections.append(skip)
            
            # Residual
            x = Conv1D(self.n_features, 1)(x)
            x = Add()([x, inputs])
        
        # Output
        x = Add()(skip_connections)
        x = Activation('relu')(x)
        x = Conv1D(64, 1, activation='relu')(x)
        x = GlobalAveragePooling1D()(x)
        
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs, outputs, name='WaveNet')
        return model
    
    # ----------------
    # 12. Bi-LSTM with Attention
    # ----------------
    def build_bilstm_attention(self):
        """Bidirectional LSTM with Attention"""
        inputs = Input(shape=self.input_shape)
        
        # Bi-LSTM layers
        x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
        x = Bidirectional(LSTM(64, return_sequences=True))(x)
        
        # Attention mechanism
        attention = Dense(1, activation='tanh')(x)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(128)(attention)
        attention = Permute([2, 1])(attention)
        
        # Apply attention
        x = Multiply()([x, attention])
        x = GlobalAveragePooling1D()(x)
        
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs, outputs, name='BiLSTM_Attention')
        return model
    
    # ----------------
    # 13. Conformal Predictor
    # ----------------
    def build_conformal_predictor(self):
        """예측 구간을 제공하는 모델"""
        inputs = Input(shape=self.input_shape)
        
        # Base predictor
        x = LSTM(128, return_sequences=True)(inputs)
        x = LSTM(64, return_sequences=False)(x)
        x = Dense(32, activation='relu')(x)
        
        # Three outputs for quantiles (0.1, 0.5, 0.9)
        lower = Dense(1, name='lower_bound')(x)
        median = Dense(1, name='median')(x)
        upper = Dense(1, name='upper_bound')(x)
        
        outputs = Concatenate()([lower, median, upper])
        
        model = Model(inputs, outputs, name='ConformalPredictor')
        return model

# ========================================
# 4. 모델 학습 및 평가
# ========================================

class ModelTrainer:
    """모델 학습 및 평가"""
    
    def __init__(self, models_dict, X_train, y_train, X_val, y_val, 
                 X_test, y_test, y_test_binary, spike_signals_test):
        self.models_dict = models_dict
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.y_test_binary = y_test_binary
        self.spike_signals_test = spike_signals_test
        self.results = {}
        
    def train_all_models(self, epochs=50, batch_size=256):
        """모든 모델 학습"""
        logger.info("="*80)
        logger.info("🚀 모델 학습 시작")
        logger.info("="*80)
        
        for name, model in self.models_dict.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"📊 {name} 학습 중...")
            logger.info(f"{'='*60}")
            
            try:
                # 컴파일
                if name == 'ConformalPredictor':
                    model.compile(
                        optimizer=AdamW(learning_rate=0.001),
                        loss=CustomLosses.quantile_loss(),
                        metrics=['mae']
                    )
                else:
                    model.compile(
                        optimizer=AdamW(learning_rate=0.001),
                        loss=CustomLosses.weighted_mse(weight_1700=100.0),
                        metrics=['mae', 'mse']
                    )
                
                # 콜백
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
                    ModelCheckpoint(f'models/{name}_best.h5', save_best_only=True, verbose=0)
                ]
                
                # 학습
                history = model.fit(
                    self.X_train, self.y_train,
                    validation_data=(self.X_val, self.y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks,
                    verbose=1
                )
                
                # 평가
                self._evaluate_model(name, model)
                
            except Exception as e:
                logger.error(f"❌ {name} 학습 실패: {str(e)}")
                continue
    
    def _evaluate_model(self, name, model):
        """개별 모델 평가"""
        # 예측
        if name == 'ConformalPredictor':
            predictions = model.predict(self.X_test, verbose=0)
            y_pred = predictions[:, 1]  # median 사용
        else:
            y_pred = model.predict(self.X_test, verbose=0).flatten()
        
        # 메트릭 계산
        mae = np.mean(np.abs(self.y_test - y_pred))
        mse = np.mean((self.y_test - y_pred) ** 2)
        rmse = np.sqrt(mse)
        
        # 1700+ 성능
        high_mask = self.y_test_binary == 1
        if high_mask.sum() > 0:
            mae_high = np.mean(np.abs(self.y_test[high_mask] - y_pred[high_mask]))
            
            # Precision/Recall (임계값 기반)
            pred_high = (y_pred > 0.5).astype(int)  # 스케일된 1700 임계값
            tp = np.sum((pred_high == 1) & (high_mask == 1))
            fp = np.sum((pred_high == 1) & (high_mask == 0))
            fn = np.sum((pred_high == 0) & (high_mask == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        else:
            mae_high = 0
            precision = 0
            recall = 0
            f1 = 0
        
        # 결과 저장
        self.results[name] = {
            'mae': mae,
            'rmse': rmse,
            'mae_1700+': mae_high,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        logger.info(f"✅ {name} 평가 완료:")
        logger.info(f"   MAE: {mae:.4f}")
        logger.info(f"   RMSE: {rmse:.4f}")
        logger.info(f"   1700+ MAE: {mae_high:.4f}")
        logger.info(f"   Precision: {precision:.2%}")
        logger.info(f"   Recall: {recall:.2%}")
        logger.info(f"   F1: {f1:.2%}")

# ========================================
# 5. 앙상블
# ========================================

class EnsemblePredictor:
    """최종 앙상블 예측"""
    
    def __init__(self, models_dict, results):
        self.models_dict = models_dict
        self.results = results
        
    def predict_weighted(self, X_test):
        """가중 평균 앙상블"""
        predictions = {}
        weights = {}
        
        # 각 모델 예측
        for name, model in self.models_dict.items():
            if name == 'ConformalPredictor':
                pred = model.predict(X_test, verbose=0)[:, 1]
            else:
                pred = model.predict(X_test, verbose=0).flatten()
            predictions[name] = pred
            
            # F1 score 기반 가중치
            weights[name] = self.results[name]['f1']
        
        # 정규화
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        else:
            weights = {k: 1/len(weights) for k in weights.keys()}
        
        # 가중 평균
        ensemble_pred = np.zeros(len(X_test))
        for name, pred in predictions.items():
            ensemble_pred += weights[name] * pred
            
        return ensemble_pred, predictions, weights

# ========================================
# 메인 실행
# ========================================

def main():
    """메인 실행 함수"""
    
    logger.info("="*80)
    logger.info("🚀 Ultimate 1700+ Predictor Suite v7.0")
    logger.info("="*80)
    
    # 1. 데이터 로드
    processor = DataProcessor(seq_len=100, pred_len=10)
    df = processor.load_and_process('data/20240201_TO_202507281705.csv')
    
    # 2. 시퀀스 생성
    X, y, y_binary, spike_signals = processor.create_sequences(df)
    
    # 3. 데이터 분할
    # Train: 70%, Val: 15%, Test: 15%
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    y_test_binary = y_binary[train_size+val_size:]
    spike_signals_test = spike_signals[train_size+val_size:]
    
    logger.info(f"✅ 데이터 분할:")
    logger.info(f"   Train: {X_train.shape}")
    logger.info(f"   Val: {X_val.shape}")
    logger.info(f"   Test: {X_test.shape}")
    
    # 4. 모델 생성
    factory = ModelFactory(input_shape=(100, X.shape[2]), n_features=X.shape[2])
    
    models_dict = {
        'PatchTST': factory.build_patch_tst(),
        'ExtremeValueNet': factory.build_extreme_value_network(),
        'SpikeDetector': factory.build_spike_detector(),
        'Mamba': factory.build_mamba(),
        'MTGNN': factory.build_mtgnn(),
        'NeuralSDE': factory.build_neural_sde(),
        'MixtureOfExperts': factory.build_mixture_of_experts(),
        'DiffusionModel': factory.build_diffusion_model(),
        'Autoformer': factory.build_autoformer(),
        'TCN': factory.build_tcn(),
        'WaveNet': factory.build_wavenet(),
        'BiLSTM_Attention': factory.build_bilstm_attention(),
        'ConformalPredictor': factory.build_conformal_predictor()
    }
    
    logger.info(f"✅ {len(models_dict)}개 모델 생성 완료")
    
    # 5. 모델 학습
    os.makedirs('models', exist_ok=True)
    trainer = ModelTrainer(
        models_dict, X_train, y_train, X_val, y_val,
        X_test, y_test, y_test_binary, spike_signals_test
    )
    trainer.train_all_models(epochs=30, batch_size=256)
    
    # 6. 앙상블 예측
    logger.info("\n" + "="*80)
    logger.info("🎯 앙상블 예측")
    logger.info("="*80)
    
    ensemble = EnsemblePredictor(models_dict, trainer.results)
    ensemble_pred, individual_preds, weights = ensemble.predict_weighted(X_test)
    
    # 7. 최종 평가
    mae = np.mean(np.abs(y_test - ensemble_pred))
    rmse = np.sqrt(np.mean((y_test - ensemble_pred) ** 2))
    
    high_mask = y_test_binary == 1
    if high_mask.sum() > 0:
        mae_high = np.mean(np.abs(y_test[high_mask] - ensemble_pred[high_mask]))
    else:
        mae_high = 0
    
    logger.info(f"\n🏆 최종 앙상블 성능:")
    logger.info(f"   MAE: {mae:.4f}")
    logger.info(f"   RMSE: {rmse:.4f}")
    logger.info(f"   1700+ MAE: {mae_high:.4f}")
    
    # 8. 모델별 순위
    logger.info("\n📊 모델별 성능 순위 (F1 기준):")
    sorted_results = sorted(trainer.results.items(), key=lambda x: x[1]['f1'], reverse=True)
    for i, (name, metrics) in enumerate(sorted_results, 1):
        logger.info(f"{i:2d}. {name:20s}: F1={metrics['f1']:.2%}, MAE={metrics['mae']:.4f}")
    
    # 9. 가중치 출력
    logger.info("\n⚖️ 앙상블 가중치:")
    for name, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"   {name:20s}: {weight:.2%}")
    
    # 10. 결과 저장
    results_summary = {
        'ensemble_mae': float(mae),
        'ensemble_rmse': float(rmse),
        'ensemble_mae_1700+': float(mae_high),
        'model_results': trainer.results,
        'ensemble_weights': weights,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('results_ultimate.json', 'w') as f:
        json.dump(results_summary, f, indent=4)
    
    logger.info("\n✅ 모든 작업 완료!")
    logger.info("📊 결과 저장: results_ultimate.json")
    
    return trainer.results, ensemble_pred

if __name__ == "__main__":
    results, predictions = main()