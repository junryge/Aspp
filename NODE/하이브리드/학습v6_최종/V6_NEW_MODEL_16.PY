"""
🚀 Ultimate 1700+ Predictor v8.0 - Production Ready
==================================================
- Infinity/NaN 오류 완전 해결
- 전체 구간별 평가 (일반, 중간, 1700+)
- 안정적인 데이터 처리
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import *
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
import logging
from datetime import datetime
import os
import json

warnings.filterwarnings('ignore')

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# TensorFlow 설정
tf.random.set_seed(42)
np.random.seed(42)

print("\n" + "="*80)
print("🚀 Ultimate 1700+ Predictor v8.0")
print(f"📦 TensorFlow: {tf.__version__}")
print("="*80)

# ========================================
# 안전한 데이터 처리
# ========================================

class SafeDataProcessor:
    """Infinity/NaN 안전 처리"""
    
    def __init__(self, seq_len=100, pred_len=10):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scaler = MinMaxScaler()  # RobustScaler 대신 MinMaxScaler 사용
        self.target_scaler = MinMaxScaler()
        
    def load_and_process(self, filepath):
        """데이터 로드 - 안전 처리"""
        logger.info("📂 데이터 로딩...")
        
        # CSV 로드
        df = pd.read_csv(filepath)
        logger.info(f"✅ 원본 데이터: {df.shape}")
        
        # 시간 변환
        df['CURRTIME'] = pd.to_datetime(df['CURRTIME'], format='%Y%m%d%H%M', errors='coerce')
        df['TIME'] = pd.to_datetime(df['TIME'], format='%Y%m%d%H%M', errors='coerce')
        
        # NaT 제거
        df = df.dropna(subset=['CURRTIME', 'TIME'])
        df = df.sort_values('CURRTIME').reset_index(drop=True)
        
        # 수치형 컬럼만 선택
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Infinity와 NaN 체크 및 제거
        logger.info("🔍 이상치 체크...")
        for col in numeric_cols:
            # Infinity를 NaN으로 변경
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            
            # NaN을 중앙값으로 채우기
            if df[col].isna().sum() > 0:
                median_val = df[col].median()
                if pd.isna(median_val):
                    median_val = 0
                df[col] = df[col].fillna(median_val)
                logger.info(f"   {col}: {df[col].isna().sum()}개 NaN → {median_val:.2f}로 대체")
        
        # 극단적 이상치 제거 (IQR 방법)
        for col in ['TOTALCNT']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 3 * IQR
            upper = Q3 + 3 * IQR
            
            before = len(df)
            df = df[(df[col] >= lower) & (df[col] <= upper)]
            after = len(df)
            
            if before != after:
                logger.info(f"   {col}: {before-after}개 극단 이상치 제거")
        
        # 타겟 생성
        df['TARGET'] = df['TOTALCNT'].shift(-self.pred_len)
        df['IS_1700'] = (df['TARGET'] >= 1700).astype(int)
        
        # 특성 생성 (안전하게)
        df = self._create_safe_features(df)
        
        # 최종 NaN 제거
        df = df.dropna()
        
        logger.info(f"✅ 전처리 완료: {df.shape}")
        logger.info(f"✅ 1700+ 비율: {df['IS_1700'].mean():.2%} ({df['IS_1700'].sum()}개)")
        
        return df
    
    def _create_safe_features(self, df):
        """안전한 특성 생성"""
        logger.info("⚙️ 특성 생성...")
        
        # 이동평균 (안전한 범위)
        for window in [10, 30]:
            df[f'MA_{window}'] = df['TOTALCNT'].rolling(window, min_periods=1).mean()
            df[f'MA_{window}'].fillna(df['TOTALCNT'].mean(), inplace=True)
        
        # 변화율 (0으로 나누기 방지)
        df['CHANGE'] = df['TOTALCNT'].diff().fillna(0)
        df['PCT_CHANGE'] = df['TOTALCNT'].pct_change().fillna(0)
        
        # 극단값 방지
        df['PCT_CHANGE'] = df['PCT_CHANGE'].clip(-1, 1)
        
        # 비율 계산 (0으로 나누기 방지)
        if 'M14AM14B' in df.columns and 'M14AM10A' in df.columns:
            df['RATIO'] = df['M14AM14B'] / (df['M14AM10A'] + 1)  # +1로 0 방지
            df['RATIO'] = df['RATIO'].clip(0, 20)  # 극단값 제한
        
        # 시간 특성
        df['HOUR'] = df['CURRTIME'].dt.hour
        df['DAYOFWEEK'] = df['CURRTIME'].dt.dayofweek
        df['IS_DAWN'] = df['HOUR'].isin([4, 5, 6]).astype(int)
        
        # 임계값 신호
        if 'M14AM14B' in df.columns:
            df['M14B_HIGH'] = (df['M14AM14B'] > df['M14AM14B'].quantile(0.9)).astype(int)
        
        # 모든 값 체크
        for col in df.select_dtypes(include=[np.number]).columns:
            # Infinity 체크
            df[col] = df[col].replace([np.inf, -np.inf], df[col].median())
            # NaN 체크
            df[col] = df[col].fillna(df[col].median())
            
        return df
    
    def create_sequences(self, df):
        """안전한 시퀀스 생성"""
        logger.info("🔄 시퀀스 생성...")
        
        # 특성 선택 (문제 될 수 있는 컬럼 제외)
        exclude_cols = ['CURRTIME', 'TIME', 'TARGET', 'IS_1700']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # 수치형만 선택
        feature_cols = [col for col in feature_cols if df[col].dtype in [np.float64, np.int64, np.float32, np.int32]]
        
        logger.info(f"   사용할 특성: {len(feature_cols)}개")
        
        # 데이터 복사
        data = df[feature_cols].copy()
        targets = df['TARGET'].copy()
        is_1700 = df['IS_1700'].copy()
        
        # 스케일링 전 체크
        logger.info("   스케일링 전 체크...")
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.fillna(data.median())
        
        # 스케일링
        try:
            data_scaled = self.scaler.fit_transform(data)
            targets_scaled = self.target_scaler.fit_transform(targets.values.reshape(-1, 1)).flatten()
        except Exception as e:
            logger.error(f"❌ 스케일링 오류: {e}")
            # 대체 방법: 수동 정규화
            data_scaled = (data - data.mean()) / (data.std() + 1e-7)
            targets_scaled = (targets - targets.mean()) / (targets.std() + 1e-7)
            data_scaled = data_scaled.fillna(0).values
            targets_scaled = targets_scaled.fillna(0).values
        
        # 시퀀스 생성
        X, y, y_binary = [], [], []
        
        for i in range(len(data_scaled) - self.seq_len - self.pred_len):
            seq = data_scaled[i:i+self.seq_len]
            
            # 시퀀스 체크
            if not np.any(np.isnan(seq)) and not np.any(np.isinf(seq)):
                X.append(seq)
                y.append(targets_scaled[i+self.seq_len])
                y_binary.append(is_1700.iloc[i+self.seq_len])
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        y_binary = np.array(y_binary, dtype=np.float32)
        
        # 최종 체크
        logger.info("   최종 체크...")
        logger.info(f"   X: NaN={np.isnan(X).sum()}, Inf={np.isinf(X).sum()}")
        logger.info(f"   y: NaN={np.isnan(y).sum()}, Inf={np.isinf(y).sum()}")
        
        logger.info(f"✅ 시퀀스 생성 완료: {X.shape}")
        
        return X, y, y_binary, feature_cols

# ========================================
# 안정적인 모델
# ========================================

class StableModels:
    """모든 모델 구현 - 15개"""
    
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.seq_len = input_shape[0]
        self.n_features = input_shape[1]
    
    # 1. PatchTST - Transformer 최강
    def build_patch_tst(self, patch_len=10, d_model=128, n_heads=4):
        """Patch Time Series Transformer - SOTA"""
        inputs = Input(shape=self.input_shape)
        
        # Normalize
        x = BatchNormalization()(inputs)
        
        # 패치 생성 (100 -> 10 patches) - Reshape 레이어 사용
        n_patches = self.seq_len // patch_len
        patch_size = patch_len * self.n_features
        
        # Reshape를 Keras 레이어로 처리
        x = Reshape((n_patches, patch_size))(x)
        
        # 패치 임베딩
        x = Dense(d_model)(x)
        x = LayerNormalization()(x)
        
        # Multi-Head Attention (2 layers)
        for _ in range(2):
            attn = MultiHeadAttention(
                num_heads=n_heads, 
                key_dim=d_model//n_heads,
                dropout=0.1
            )(x, x)
            x = Add()([x, attn])
            x = LayerNormalization()(x)
            
            # FFN
            ff = Dense(d_model * 2, activation='relu')(x)
            ff = Dropout(0.1)(ff)
            ff = Dense(d_model)(ff)
            x = Add()([x, ff])
            x = LayerNormalization()(x)
        
        x = GlobalAveragePooling1D()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs, outputs, name='PatchTST')
    
    # 2. Extreme Value Network - 1700+ 킬러
    def build_extreme_value_network(self):
        """극단값 이론 기반 - 1700+ 특화"""
        inputs = Input(shape=self.input_shape)
        
        x = BatchNormalization()(inputs)
        
        # LSTM backbone
        x = LSTM(128, return_sequences=True)(x)
        x = Dropout(0.2)(x)
        x = LSTM(64)(x)
        x = Dropout(0.2)(x)
        
        # 극단값 브랜치
        extreme = Dense(32, activation='relu')(x)
        extreme = Dense(16, activation='relu')(extreme)
        xi = Dense(1, activation='tanh')(extreme)  # shape parameter
        sigma = Dense(1, activation='softplus')(extreme)  # scale parameter
        
        # 일반 브랜치
        normal = Dense(32, activation='relu')(x)
        normal_pred = Dense(1, activation='linear')(normal)
        
        # GPD 기반 보정
        extreme_adjustment = Multiply()([sigma, xi])
        outputs = Add()([normal_pred, extreme_adjustment])
        outputs = Activation('sigmoid')(outputs)
        
        return Model(inputs, outputs, name='ExtremeValueNet')
    
    # 3. Enhanced Spike Detector - 급변 감지
    def build_spike_detector(self):
        """급변 감지 특화"""
        inputs = Input(shape=self.input_shape)
        
        x = BatchNormalization()(inputs)
        
        # CNN for pattern
        x = Conv1D(64, 3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv1D(64, 3, padding='same', activation='relu')(x)
        x = MaxPooling1D(2)(x)
        
        # Bidirectional LSTM
        x = Bidirectional(LSTM(64, return_sequences=True))(x)
        x = Bidirectional(LSTM(32))(x)
        
        # Detection heads
        spike_prob = Dense(32, activation='relu')(x)
        spike_prob = Dense(1, activation='sigmoid')(spike_prob)
        
        spike_mag = Dense(32, activation='relu')(x)
        spike_mag = Dense(1, activation='linear')(spike_mag)
        
        outputs = Multiply()([spike_prob, spike_mag])
        outputs = Activation('sigmoid')(outputs)
        
        return Model(inputs, outputs, name='SpikeDetector')
    
    # 4. Mamba (State Space Model)
    def build_mamba_ssm(self):
        """State Space Model - 효율적"""
        inputs = Input(shape=self.input_shape)
        
        x = BatchNormalization()(inputs)
        
        # SSM layers (GRU로 근사)
        x = GRU(128, return_sequences=True)(x)
        x = GRU(64, return_sequences=True)(x)
        x = GRU(32)(x)
        
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs, outputs, name='MambaSSM')
    
    # 5. MTGNN (Graph Neural Network)
    def build_mtgnn(self):
        """Multivariate Time Graph NN - 컬럼 관계"""
        inputs = Input(shape=self.input_shape)
        
        x = BatchNormalization()(inputs)
        
        # Graph construction
        node_emb = Dense(64, activation='relu')(x)
        
        # Temporal Conv
        x = Conv1D(64, 3, padding='causal', activation='relu')(x)
        x = Conv1D(64, 3, padding='causal', activation='relu')(x)
        
        # Graph Conv (simplified)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        
        x = GlobalAveragePooling1D()(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs, outputs, name='MTGNN')
    
    # 6. Neural SDE
    def build_neural_sde(self):
        """확률적 미분방정식"""
        inputs = Input(shape=self.input_shape)
        
        x = BatchNormalization()(inputs)
        
        # Drift (mu)
        drift = LSTM(64, return_sequences=True)(x)
        drift = LSTM(32)(drift)
        mu = Dense(16, activation='linear')(drift)
        
        # Diffusion (sigma)  
        diffusion = LSTM(64, return_sequences=True)(x)
        diffusion = LSTM(32)(diffusion)
        sigma = Dense(16, activation='softplus')(diffusion)
        
        # Stochastic integration - Lambda 레이어 사용
        x = Add()([mu, Lambda(lambda s: s * np.random.normal(0, 0.1, (1,)))(sigma)])
        
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs, outputs, name='NeuralSDE')
    
    # 7. Mixture of Experts
    def build_mixture_of_experts(self):
        """전문가 앙상블"""
        inputs = Input(shape=self.input_shape)
        
        x = BatchNormalization()(inputs)
        
        # Shared
        shared = LSTM(64, return_sequences=True)(x)
        shared = LSTM(32)(shared)
        
        # 3 Experts
        e1 = Dense(16, activation='relu')(shared)
        e1_out = Dense(1, activation='sigmoid')(e1)
        
        e2 = Dense(16, activation='relu')(shared)
        e2_out = Dense(1, activation='sigmoid')(e2)
        
        e3 = Dense(16, activation='relu')(shared)
        e3_out = Dense(1, activation='sigmoid')(e3)
        
        # Router
        gate = Dense(16, activation='relu')(shared)
        gate = Dense(3, activation='softmax')(gate)
        
        # Weighted sum - Keras 연산으로 처리
        e1_weighted = Multiply()([e1_out, Lambda(lambda g: g[:, 0:1])(gate)])
        e2_weighted = Multiply()([e2_out, Lambda(lambda g: g[:, 1:2])(gate)])
        e3_weighted = Multiply()([e3_out, Lambda(lambda g: g[:, 2:3])(gate)])
        
        outputs = Add()([e1_weighted, e2_weighted, e3_weighted])
        
        return Model(inputs, outputs, name='MixtureOfExperts')
    
    # 8. Diffusion Model
    def build_diffusion_model(self):
        """Denoising Diffusion"""
        inputs = Input(shape=self.input_shape)
        
        x = BatchNormalization()(inputs)
        
        # Encoder
        x = Conv1D(64, 3, padding='same', activation='relu')(x)
        x = Conv1D(128, 3, padding='same', activation='relu')(x)
        x = MaxPooling1D(2)(x)
        
        # Noising - Lambda 레이어 사용
        x = LSTM(64, return_sequences=True)(x)
        x = Lambda(lambda t: t + np.random.normal(0, 0.05, (1,1,1)))(x)
        
        # Denoiser
        x = LSTM(64)(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs, outputs, name='DiffusionModel')
    
    # 9. Autoformer
    def build_autoformer(self):
        """Auto-Correlation Transformer"""
        inputs = Input(shape=self.input_shape)
        
        x = BatchNormalization()(inputs)
        
        # Decomposition
        trend = AveragePooling1D(10, strides=1, padding='same')(x)
        seasonal = x - trend
        
        # Process
        trend = Conv1D(64, 1, activation='relu')(trend)
        seasonal = Conv1D(64, 1, activation='relu')(seasonal)
        
        # Auto-Correlation
        x = Concatenate()([trend, seasonal])
        x = MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
        
        x = GlobalAveragePooling1D()(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs, outputs, name='Autoformer')
    
    # 10. TCN (Temporal CNN)
    def build_tcn(self):
        """Temporal Convolutional Network"""
        inputs = Input(shape=self.input_shape)
        
        x = BatchNormalization()(inputs)
        
        # Dilated convolutions
        for dilation in [1, 2, 4, 8]:
            conv = Conv1D(64, 3, dilation_rate=dilation, 
                         padding='causal', activation='relu')(x)
            conv = Dropout(0.1)(conv)
            if x.shape[-1] != conv.shape[-1]:
                x = Conv1D(64, 1)(x)
            x = Add()([x, conv])
        
        x = GlobalAveragePooling1D()(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs, outputs, name='TCN')
    
    # 11. WaveNet
    def build_wavenet(self):
        """WaveNet Architecture"""
        inputs = Input(shape=self.input_shape)
        
        x = BatchNormalization()(inputs)
        skip_connections = []
        
        for i in range(4):
            dilation = 2 ** i
            
            # Gated conv
            tanh_out = Conv1D(64, 2, dilation_rate=dilation,
                             padding='causal', activation='tanh')(x)
            sigm_out = Conv1D(64, 2, dilation_rate=dilation,
                             padding='causal', activation='sigmoid')(x)
            
            x = Multiply()([tanh_out, sigm_out])
            
            skip = Conv1D(64, 1)(x)
            skip_connections.append(skip)
            
            x = Conv1D(self.n_features, 1)(x)
            x = Add()([x, inputs])
        
        x = Add()(skip_connections)
        x = Activation('relu')(x)
        x = GlobalAveragePooling1D()(x)
        
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs, outputs, name='WaveNet')
    
    # 12. Bi-LSTM with Attention
    def build_bilstm_attention(self):
        """양방향 LSTM + Attention"""
        inputs = Input(shape=self.input_shape)
        
        x = BatchNormalization()(inputs)
        
        # Bi-LSTM
        x = Bidirectional(LSTM(128, return_sequences=True))(x)
        x = Bidirectional(LSTM(64, return_sequences=True))(x)
        
        # Attention
        attention = Dense(1, activation='tanh')(x)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(128)(attention)
        attention = Permute([2, 1])(attention)
        
        x = Multiply()([x, attention])
        x = GlobalAveragePooling1D()(x)
        
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs, outputs, name='BiLSTM_Attention')
    
    # 13. Informer
    def build_informer(self):
        """Informer - 효율적 Transformer"""
        inputs = Input(shape=self.input_shape)
        
        x = BatchNormalization()(inputs)
        
        # ProbSparse Attention (simplified)
        x = Dense(128)(x)
        x = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
        x = LayerNormalization()(x)
        
        x = Dense(64, activation='relu')(x)
        x = GlobalAveragePooling1D()(x)
        
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs, outputs, name='Informer')
    
    # 14. TimesNet
    def build_timesnet(self):
        """TimesNet - 2D Transform"""
        inputs = Input(shape=self.input_shape)
        
        x = BatchNormalization()(inputs)
        
        # 1D to 2D transform - Keras 레이어 사용
        x = Lambda(lambda t: tf.expand_dims(t, axis=-1))(x)
        x = Conv2D(64, (3, 1), padding='same', activation='relu')(x)
        x = Conv2D(32, (3, 1), padding='same', activation='relu')(x)
        x = Lambda(lambda t: tf.squeeze(t, axis=-1))(x)
        
        x = GlobalAveragePooling1D()(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs, outputs, name='TimesNet')
    
    # 15. Baseline Strong LSTM
    def build_baseline_lstm(self):
        """강력한 LSTM 베이스라인"""
        inputs = Input(shape=self.input_shape)
        
        x = BatchNormalization()(inputs)
        
        x = LSTM(256, return_sequences=True)(x)
        x = Dropout(0.3)(x)
        x = LSTM(128, return_sequences=True)(x)
        x = Dropout(0.3)(x)
        x = LSTM(64)(x)
        x = Dropout(0.2)(x)
        
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.1)(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs, outputs, name='BaselineLSTM')

# ========================================
# 평가 시스템
# ========================================

class ComprehensiveEvaluator:
    """전체 구간 평가"""
    
    def __init__(self, target_scaler):
        self.target_scaler = target_scaler
        self.results = {}
    
    def evaluate(self, model, X_test, y_test, y_binary, model_name):
        """종합 평가"""
        
        # 예측
        y_pred = model.predict(X_test, batch_size=256, verbose=0).flatten()
        
        # Clip predictions to valid range
        y_pred = np.clip(y_pred, 0, 1)
        
        # 역변환
        y_test_real = self.target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_real = self.target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        
        # ===== 전체 성능 =====
        mae_total = np.mean(np.abs(y_test_real - y_pred_real))
        rmse_total = np.sqrt(np.mean((y_test_real - y_pred_real) ** 2))
        
        # ===== 구간별 분석 =====
        metrics = {
            'total': {'mae': mae_total, 'rmse': rmse_total}
        }
        
        # 구간 정의
        ranges = [
            ('low', 0, 1200),
            ('normal', 1200, 1400),
            ('medium', 1400, 1600),
            ('high', 1600, 1700),
            ('extreme', 1700, 9999)
        ]
        
        for name, low, high in ranges:
            mask = (y_test_real >= low) & (y_test_real < high)
            if mask.sum() > 0:
                mae = np.mean(np.abs(y_test_real[mask] - y_pred_real[mask]))
                rmse = np.sqrt(np.mean((y_test_real[mask] - y_pred_real[mask]) ** 2))
                count = mask.sum()
                metrics[name] = {
                    'mae': mae,
                    'rmse': rmse,
                    'count': count,
                    'percentage': (count / len(y_test_real)) * 100
                }
        
        # ===== 1700+ 특별 분석 =====
        actual_1700 = y_test_real >= 1700
        pred_1700 = y_pred_real >= 1700
        
        tp = np.sum(pred_1700 & actual_1700)
        fp = np.sum(pred_1700 & ~actual_1700)
        fn = np.sum(~pred_1700 & actual_1700)
        tn = np.sum(~pred_1700 & ~actual_1700)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # ===== 정상 구간 오탐 =====
        normal_mask = (y_test_real >= 1200) & (y_test_real < 1400)
        if normal_mask.sum() > 0:
            false_alarm = np.sum(y_pred_real[normal_mask] >= 1700)
            false_alarm_rate = false_alarm / normal_mask.sum()
        else:
            false_alarm_rate = 0
        
        # 결과 저장
        result = {
            'metrics': metrics,
            'precision_1700': precision,
            'recall_1700': recall,
            'f1_1700': f1,
            'false_alarm_rate': false_alarm_rate,
            'confusion_matrix': {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}
        }
        
        self.results[model_name] = result
        
        # 출력
        self._print_results(model_name, result)
        
        return result
    
    def _print_results(self, model_name, result):
        """결과 출력"""
        print(f"\n{'='*60}")
        print(f"📊 {model_name} 평가 결과")
        print(f"{'='*60}")
        
        print("\n[전체 성능]")
        print(f"  MAE: {result['metrics']['total']['mae']:.2f}")
        print(f"  RMSE: {result['metrics']['total']['rmse']:.2f}")
        
        print("\n[구간별 성능]")
        for range_name in ['low', 'normal', 'medium', 'high', 'extreme']:
            if range_name in result['metrics']:
                m = result['metrics'][range_name]
                print(f"  {range_name:8s}: MAE={m['mae']:6.2f}, "
                      f"Count={m['count']:4d} ({m['percentage']:5.2f}%)")
        
        print("\n[1700+ 예측 성능]")
        print(f"  Precision: {result['precision_1700']:.2%}")
        print(f"  Recall: {result['recall_1700']:.2%}")
        print(f"  F1 Score: {result['f1_1700']:.2%}")
        
        print(f"\n[정상→1700+ 오탐률]")
        print(f"  False Alarm Rate: {result['false_alarm_rate']:.2%}")
        
        # 종합 점수
        balanced_score = (
            result['f1_1700'] * 0.4 +  # 1700+ 예측
            (1 - result['false_alarm_rate']) * 0.3 +  # 오탐 방지
            (1 - result['metrics']['normal']['mae']/100) * 0.3  # 정상 구간 정확도
        )
        print(f"\n[종합 점수]")
        print(f"  Balanced Score: {balanced_score:.2%}")

# ========================================
# 메인 실행
# ========================================

def main():
    """메인 함수"""
    
    print("\n🚀 시작...")
    
    # 1. 데이터 처리
    processor = SafeDataProcessor(seq_len=100, pred_len=10)
    
    # 데이터 경로 확인
    data_paths = [
        'data/20240201_TO_202507281705.csv',
        '/mnt/user-data/uploads/gs.CSV',
        '/mnt/user-data/uploads/20250807_DATA.CSV'
    ]
    
    data_path = None
    for path in data_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if data_path is None:
        logger.error("❌ 데이터 파일을 찾을 수 없습니다!")
        return
    
    df = processor.load_and_process(data_path)
    X, y, y_binary, feature_cols = processor.create_sequences(df)
    
    # 2. 데이터 분할
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    y_test_binary = y_binary[train_size+val_size:]
    
    print(f"\n📊 데이터 분할:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val: {X_val.shape}")
    print(f"  Test: {X_test.shape}")
    
    # 3. 모델 생성
    model_builder = StableModels(input_shape=(100, X.shape[2]))
    
    # 모든 15개 모델!
    models = {
        '1_PatchTST': model_builder.build_patch_tst(),
        '2_ExtremeValueNet': model_builder.build_extreme_value_network(),
        '3_SpikeDetector': model_builder.build_spike_detector(),
        '4_MambaSSM': model_builder.build_mamba_ssm(),
        '5_MTGNN': model_builder.build_mtgnn(),
        '6_NeuralSDE': model_builder.build_neural_sde(),
        '7_MixtureOfExperts': model_builder.build_mixture_of_experts(),
        '8_DiffusionModel': model_builder.build_diffusion_model(),
        '9_Autoformer': model_builder.build_autoformer(),
        '10_TCN': model_builder.build_tcn(),
        '11_WaveNet': model_builder.build_wavenet(),
        '12_BiLSTM_Attention': model_builder.build_bilstm_attention(),
        '13_Informer': model_builder.build_informer(),
        '14_TimesNet': model_builder.build_timesnet(),
        '15_BaselineLSTM': model_builder.build_baseline_lstm()
    }
    
    print(f"\n✅ {len(models)}개 모델 생성 완료!")
    print("="*60)
    for name in models.keys():
        print(f"  ✓ {name}")
    print("="*60)
    
    # 4. 평가자 생성
    evaluator = ComprehensiveEvaluator(processor.target_scaler)
    
    # 5. 각 모델 학습 및 평가
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"🎯 {name} 학습")
        print(f"{'='*60}")
        
        # 컴파일
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',  # 안정적인 MSE 사용
            metrics=['mae']
        )
        
        # 학습
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=20,
            batch_size=256,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
            ],
            verbose=1
        )
        
        # 평가
        evaluator.evaluate(model, X_test, y_test, y_test_binary, name)
    
    # 6. 최종 순위
    print("\n" + "="*80)
    print("🏆 최종 순위")
    print("="*80)
    
    # F1 기준 정렬
    sorted_models = sorted(
        evaluator.results.items(),
        key=lambda x: x[1]['f1_1700'],
        reverse=True
    )
    
    for i, (name, result) in enumerate(sorted_models, 1):
        print(f"{i}. {name:15s} | F1: {result['f1_1700']:.2%} | "
              f"정상MAE: {result['metrics']['normal']['mae']:.2f} | "
              f"오탐률: {result['false_alarm_rate']:.2%}")
    
    print("\n✅ 완료!")
    
    return evaluator.results

if __name__ == "__main__":
    results = main()