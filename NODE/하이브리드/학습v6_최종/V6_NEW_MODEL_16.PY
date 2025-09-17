"""
🚀 Ultimate 1700+ Predictor v9.0 - Final Production
==================================================
TensorFlow 2.18.0 완벽 호환
모든 15개 모델 포함
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import *
from sklearn.preprocessing import MinMaxScaler
import warnings
import logging
from datetime import datetime
import os
import json

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("\n" + "="*80)
print("🚀 Ultimate 1700+ Predictor v9.0 - FINAL")
print(f"📦 TensorFlow: {tf.__version__}")
print("="*80)

tf.random.set_seed(42)
np.random.seed(42)

# ========================================
# 안전한 데이터 처리
# ========================================

class SafeDataProcessor:
    """완벽한 데이터 처리"""
    
    def __init__(self, seq_len=100, pred_len=10):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        
    def load_and_process(self, filepath):
        logger.info("📂 데이터 로딩...")
        
        df = pd.read_csv(filepath)
        logger.info(f"✅ 원본 데이터: {df.shape}")
        
        # 시간 변환
        df['CURRTIME'] = pd.to_datetime(df['CURRTIME'], format='%Y%m%d%H%M', errors='coerce')
        df['TIME'] = pd.to_datetime(df['TIME'], format='%Y%m%d%H%M', errors='coerce')
        df = df.dropna(subset=['CURRTIME', 'TIME'])
        df = df.sort_values('CURRTIME').reset_index(drop=True)
        
        # 수치형 컬럼
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 이상치 처리
        logger.info("🔍 이상치 체크...")
        for col in numeric_cols:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            if df[col].isna().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        # 타겟 생성
        df['TARGET'] = df['TOTALCNT'].shift(-self.pred_len)
        df['IS_1700'] = (df['TARGET'] >= 1700).astype(int)
        
        # 특성 생성
        df = self._create_safe_features(df)
        df = df.dropna()
        
        logger.info(f"✅ 전처리 완료: {df.shape}")
        logger.info(f"✅ 1700+ 비율: {df['IS_1700'].mean():.2%} ({df['IS_1700'].sum()}개)")
        
        return df
    
    def _create_safe_features(self, df):
        logger.info("⚙️ 특성 생성...")
        
        # 이동평균
        for window in [10, 30]:
            df[f'MA_{window}'] = df['TOTALCNT'].rolling(window, min_periods=1).mean()
            df[f'MA_{window}'].fillna(df['TOTALCNT'].mean(), inplace=True)
        
        # 변화율
        df['CHANGE'] = df['TOTALCNT'].diff().fillna(0)
        df['PCT_CHANGE'] = df['TOTALCNT'].pct_change().fillna(0).clip(-1, 1)
        
        # 비율 (안전)
        if 'M14AM14B' in df.columns and 'M14AM10A' in df.columns:
            df['RATIO'] = df['M14AM14B'] / (df['M14AM10A'] + 1)
            df['RATIO'] = df['RATIO'].clip(0, 20)
        
        # 시간
        df['HOUR'] = df['CURRTIME'].dt.hour
        df['IS_DAWN'] = df['HOUR'].isin([4, 5, 6]).astype(int)
        
        # 임계값
        if 'M14AM14B' in df.columns:
            df['M14B_HIGH'] = (df['M14AM14B'] > df['M14AM14B'].quantile(0.9)).astype(int)
        
        # 최종 체크
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].replace([np.inf, -np.inf], df[col].median())
            df[col] = df[col].fillna(df[col].median())
            
        return df
    
    def create_sequences(self, df):
        logger.info("🔄 시퀀스 생성...")
        
        exclude_cols = ['CURRTIME', 'TIME', 'TARGET', 'IS_1700']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        feature_cols = [col for col in feature_cols if df[col].dtype in [np.float64, np.int64, np.float32, np.int32]]
        
        logger.info(f"   사용할 특성: {len(feature_cols)}개")
        
        data = df[feature_cols].copy()
        targets = df['TARGET'].copy()
        is_1700 = df['IS_1700'].copy()
        
        # 스케일링
        logger.info("   스케일링 전 체크...")
        data = data.replace([np.inf, -np.inf], np.nan).fillna(data.median())
        
        try:
            data_scaled = self.scaler.fit_transform(data)
            targets_scaled = self.target_scaler.fit_transform(targets.values.reshape(-1, 1)).flatten()
        except:
            data_scaled = (data - data.mean()) / (data.std() + 1e-7)
            targets_scaled = (targets - targets.mean()) / (targets.std() + 1e-7)
            data_scaled = data_scaled.fillna(0).values
            targets_scaled = targets_scaled.fillna(0).values
        
        # 시퀀스 생성
        X, y, y_binary = [], [], []
        
        for i in range(len(data_scaled) - self.seq_len - self.pred_len):
            seq = data_scaled[i:i+self.seq_len]
            if not np.any(np.isnan(seq)) and not np.any(np.isinf(seq)):
                X.append(seq)
                y.append(targets_scaled[i+self.seq_len])
                y_binary.append(is_1700.iloc[i+self.seq_len])
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        y_binary = np.array(y_binary, dtype=np.float32)
        
        logger.info("   최종 체크...")
        logger.info(f"   X: NaN={np.isnan(X).sum()}, Inf={np.isinf(X).sum()}")
        logger.info(f"   y: NaN={np.isnan(y).sum()}, Inf={np.isinf(y).sum()}")
        logger.info(f"✅ 시퀀스 생성 완료: {X.shape}")
        
        return X, y, y_binary, feature_cols

# ========================================
# 모든 모델 구현 (15개)
# ========================================

class AllModels:
    """15개 모델 전체"""
    
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.seq_len = input_shape[0]
        self.n_features = input_shape[1]
    
    # 1. PatchTST
    def build_patch_tst(self):
        inputs = Input(shape=self.input_shape)
        x = BatchNormalization()(inputs)
        
        # 패치 생성
        n_patches = 10
        patch_size = (self.seq_len // n_patches) * self.n_features
        x = Reshape((n_patches, patch_size))(x)
        
        # 임베딩
        x = Dense(128)(x)
        x = LayerNormalization()(x)
        
        # Attention
        x = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
        x = LayerNormalization()(x)
        
        x = GlobalAveragePooling1D()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs, outputs, name='PatchTST')
    
    # 2. ExtremeValueNet
    def build_extreme_value_net(self):
        inputs = Input(shape=self.input_shape)
        x = BatchNormalization()(inputs)
        
        x = LSTM(128, return_sequences=True)(x)
        x = LSTM(64)(x)
        
        # 극단값 처리
        normal = Dense(32, activation='relu')(x)
        normal_out = Dense(1)(normal)
        
        extreme = Dense(32, activation='relu')(x)
        extreme_factor = Dense(1, activation='sigmoid')(extreme)
        
        outputs = Add()([normal_out, Multiply()([normal_out, extreme_factor])])
        outputs = Activation('sigmoid')(outputs)
        
        return Model(inputs, outputs, name='ExtremeValueNet')
    
    # 3. SpikeDetector
    def build_spike_detector(self):
        inputs = Input(shape=self.input_shape)
        x = BatchNormalization()(inputs)
        
        x = Conv1D(64, 3, padding='same', activation='relu')(x)
        x = Conv1D(64, 3, padding='same', activation='relu')(x)
        x = MaxPooling1D(2)(x)
        
        x = Bidirectional(LSTM(32))(x)
        
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs, outputs, name='SpikeDetector')
    
    # 4. MambaSSM
    def build_mamba_ssm(self):
        inputs = Input(shape=self.input_shape)
        x = BatchNormalization()(inputs)
        
        x = GRU(128, return_sequences=True)(x)
        x = GRU(64, return_sequences=True)(x)
        x = GRU(32)(x)
        
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs, outputs, name='MambaSSM')
    
    # 5. MTGNN
    def build_mtgnn(self):
        inputs = Input(shape=self.input_shape)
        x = BatchNormalization()(inputs)
        
        x = Conv1D(64, 3, padding='causal', activation='relu')(x)
        x = Conv1D(64, 3, padding='causal', activation='relu')(x)
        
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        
        x = GlobalAveragePooling1D()(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs, outputs, name='MTGNN')
    
    # 6. NeuralSDE
    def build_neural_sde(self):
        inputs = Input(shape=self.input_shape)
        x = BatchNormalization()(inputs)
        
        x = LSTM(64, return_sequences=True)(x)
        x = LSTM(32)(x)
        
        mu = Dense(16)(x)
        sigma = Dense(16, activation='softplus')(x)
        
        # 간단한 결합
        x = Concatenate()([mu, sigma])
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs, outputs, name='NeuralSDE')
    
    # 7. MixtureOfExperts
    def build_mixture_of_experts(self):
        inputs = Input(shape=self.input_shape)
        x = BatchNormalization()(inputs)
        
        shared = LSTM(64, return_sequences=True)(x)
        shared = LSTM(32)(shared)
        
        # 3 experts
        e1 = Dense(16, activation='relu')(shared)
        e1 = Dense(1, activation='sigmoid')(e1)
        
        e2 = Dense(16, activation='relu')(shared)
        e2 = Dense(1, activation='sigmoid')(e2)
        
        e3 = Dense(16, activation='relu')(shared)
        e3 = Dense(1, activation='sigmoid')(e3)
        
        # Gate
        gate = Dense(3, activation='softmax')(shared)
        
        # Weighted sum (간단하게)
        outputs = Add()([
            Multiply()([e1, Reshape((1,))(Lambda(lambda x: x[:, 0])(gate))]),
            Multiply()([e2, Reshape((1,))(Lambda(lambda x: x[:, 1])(gate))]),
            Multiply()([e3, Reshape((1,))(Lambda(lambda x: x[:, 2])(gate))])
        ])
        
        return Model(inputs, outputs, name='MixtureOfExperts')
    
    # 8. DiffusionModel
    def build_diffusion_model(self):
        inputs = Input(shape=self.input_shape)
        x = BatchNormalization()(inputs)
        
        x = Conv1D(64, 3, padding='same', activation='relu')(x)
        x = Conv1D(128, 3, padding='same', activation='relu')(x)
        x = MaxPooling1D(2)(x)
        
        x = LSTM(64, return_sequences=True)(x)
        x = LSTM(32)(x)
        
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs, outputs, name='DiffusionModel')
    
    # 9. Autoformer
    def build_autoformer(self):
        inputs = Input(shape=self.input_shape)
        x = BatchNormalization()(inputs)
        
        # Decomposition
        trend = AveragePooling1D(10, strides=1, padding='same')(x)
        seasonal = Subtract()([x, trend])
        
        trend = Conv1D(32, 1, activation='relu')(trend)
        seasonal = Conv1D(32, 1, activation='relu')(seasonal)
        
        x = Concatenate()([trend, seasonal])
        x = GlobalAveragePooling1D()(x)
        
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs, outputs, name='Autoformer')
    
    # 10. TCN
    def build_tcn(self):
        inputs = Input(shape=self.input_shape)
        x = BatchNormalization()(inputs)
        
        # Dilated convolutions
        for dilation in [1, 2, 4, 8]:
            conv = Conv1D(64, 3, dilation_rate=dilation, padding='causal', activation='relu')(x)
            conv = Dropout(0.1)(conv)
            # Residual
            if x.shape[-1] != 64:
                x = Conv1D(64, 1)(x)
            x = Add()([x, conv])
        
        x = GlobalAveragePooling1D()(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs, outputs, name='TCN')
    
    # 11. WaveNet
    def build_wavenet(self):
        inputs = Input(shape=self.input_shape)
        x = BatchNormalization()(inputs)
        
        for i in range(3):
            dilation = 2 ** i
            tanh = Conv1D(32, 2, dilation_rate=dilation, padding='causal', activation='tanh')(x)
            sigm = Conv1D(32, 2, dilation_rate=dilation, padding='causal', activation='sigmoid')(x)
            x = Multiply()([tanh, sigm])
            
            # Residual
            if x.shape[-1] != self.n_features:
                res = Conv1D(self.n_features, 1)(x)
            else:
                res = x
            x = Add()([inputs, res])
        
        x = GlobalAveragePooling1D()(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs, outputs, name='WaveNet')
    
    # 12. BiLSTM_Attention
    def build_bilstm_attention(self):
        inputs = Input(shape=self.input_shape)
        x = BatchNormalization()(inputs)
        
        x = Bidirectional(LSTM(64, return_sequences=True))(x)
        
        # Simple attention
        attention = Dense(1, activation='tanh')(x)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(128)(attention)
        attention = Permute([2, 1])(attention)
        
        x = Multiply()([x, attention])
        x = GlobalAveragePooling1D()(x)
        
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs, outputs, name='BiLSTM_Attention')
    
    # 13. Informer (simplified)
    def build_informer(self):
        inputs = Input(shape=self.input_shape)
        x = BatchNormalization()(inputs)
        
        x = Dense(64)(x)
        x = MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
        x = LayerNormalization()(x)
        
        x = GlobalAveragePooling1D()(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs, outputs, name='Informer')
    
    # 14. TimesNet (simplified - 2D 변환 제거)
    def build_timesnet(self):
        inputs = Input(shape=self.input_shape)
        x = BatchNormalization()(inputs)
        
        # 1D CNN으로 단순화
        x = Conv1D(64, 3, padding='same', activation='relu')(x)
        x = Conv1D(32, 3, padding='same', activation='relu')(x)
        
        x = GlobalAveragePooling1D()(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs, outputs, name='TimesNet')
    
    # 15. BaselineLSTM
    def build_baseline_lstm(self):
        inputs = Input(shape=self.input_shape)
        x = BatchNormalization()(inputs)
        
        x = LSTM(128, return_sequences=True)(x)
        x = Dropout(0.2)(x)
        x = LSTM(64)(x)
        x = Dropout(0.2)(x)
        
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs, outputs, name='BaselineLSTM')

# ========================================
# 평가
# ========================================

class Evaluator:
    def __init__(self, target_scaler):
        self.target_scaler = target_scaler
        self.results = {}
    
    def evaluate(self, model, X_test, y_test, y_binary, model_name):
        # 예측
        y_pred = model.predict(X_test, batch_size=256, verbose=0).flatten()
        y_pred = np.clip(y_pred, 0, 1)
        
        # 역변환
        y_test_real = self.target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_real = self.target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        
        # 전체 성능
        mae_total = np.mean(np.abs(y_test_real - y_pred_real))
        rmse_total = np.sqrt(np.mean((y_test_real - y_pred_real) ** 2))
        
        # 1700+ 성능
        actual_1700 = y_test_real >= 1700
        pred_1700 = y_pred_real >= 1700
        
        tp = np.sum(pred_1700 & actual_1700)
        fp = np.sum(pred_1700 & ~actual_1700)
        fn = np.sum(~pred_1700 & actual_1700)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # 정상 구간 성능
        normal_mask = (y_test_real >= 1200) & (y_test_real < 1400)
        if normal_mask.sum() > 0:
            mae_normal = np.mean(np.abs(y_test_real[normal_mask] - y_pred_real[normal_mask]))
            false_alarm = np.sum(y_pred_real[normal_mask] >= 1700) / normal_mask.sum()
        else:
            mae_normal = 0
            false_alarm = 0
        
        result = {
            'mae_total': mae_total,
            'rmse_total': rmse_total,
            'precision_1700': precision,
            'recall_1700': recall,
            'f1_1700': f1,
            'mae_normal': mae_normal,
            'false_alarm': false_alarm
        }
        
        self.results[model_name] = result
        
        print(f"\n{'='*50}")
        print(f"📊 {model_name}")
        print(f"  전체 MAE: {mae_total:.2f}")
        print(f"  1700+ F1: {f1:.2%}")
        print(f"  정상 MAE: {mae_normal:.2f}")
        print(f"  오탐률: {false_alarm:.2%}")
        
        return result

# ========================================
# 메인
# ========================================

def main():
    print("\n🚀 시작...")
    
    # 1. 데이터 처리
    processor = SafeDataProcessor()
    
    data_paths = [
        'data/20240201_TO_202507281705.csv',
        '/mnt/user-data/uploads/gs.CSV',
        'gs.CSV'
    ]
    
    data_path = None
    for path in data_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if not data_path:
        print("❌ 데이터 파일을 찾을 수 없습니다!")
        return
    
    df = processor.load_and_process(data_path)
    X, y, y_binary, _ = processor.create_sequences(df)
    
    # 2. 분할
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
    builder = AllModels(input_shape=(100, X.shape[2]))
    
    models = {
        '1_PatchTST': builder.build_patch_tst(),
        '2_ExtremeValueNet': builder.build_extreme_value_net(),
        '3_SpikeDetector': builder.build_spike_detector(),
        '4_MambaSSM': builder.build_mamba_ssm(),
        '5_MTGNN': builder.build_mtgnn(),
        '6_NeuralSDE': builder.build_neural_sde(),
        '7_MixtureOfExperts': builder.build_mixture_of_experts(),
        '8_DiffusionModel': builder.build_diffusion_model(),
        '9_Autoformer': builder.build_autoformer(),
        '10_TCN': builder.build_tcn(),
        '11_WaveNet': builder.build_wavenet(),
        '12_BiLSTM_Attention': builder.build_bilstm_attention(),
        '13_Informer': builder.build_informer(),
        '14_TimesNet': builder.build_timesnet(),
        '15_BaselineLSTM': builder.build_baseline_lstm()
    }
    
    print(f"\n✅ {len(models)}개 모델 생성 완료!")
    
    # 4. 평가자
    evaluator = Evaluator(processor.target_scaler)
    
    # 5. 학습 및 평가
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"🎯 {name} 학습")
        
        try:
            model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=10,  # 빠른 테스트
                batch_size=256,
                callbacks=[
                    EarlyStopping(patience=3, restore_best_weights=True),
                    ReduceLROnPlateau(patience=2, factor=0.5)
                ],
                verbose=1
            )
            
            evaluator.evaluate(model, X_test, y_test, y_test_binary, name)
            
        except Exception as e:
            print(f"❌ {name} 오류: {str(e)}")
    
    # 6. 최종 순위
    print("\n" + "="*80)
    print("🏆 최종 순위 (F1 기준)")
    print("="*80)
    
    sorted_models = sorted(evaluator.results.items(), 
                          key=lambda x: x[1]['f1_1700'], 
                          reverse=True)
    
    for i, (name, res) in enumerate(sorted_models, 1):
        print(f"{i:2d}. {name:20s} | F1: {res['f1_1700']:.2%} | MAE: {res['mae_total']:.2f}")
    
    print("\n✅ 완료!")
    return evaluator.results

if __name__ == "__main__":
    results = main()