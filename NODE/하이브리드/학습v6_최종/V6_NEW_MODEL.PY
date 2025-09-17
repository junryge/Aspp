"""
pip install tensorflow-probability[tf]==0.24.0
🚀 Ultimate 1700+ Predictor Suite v7.1
=====================================
TensorFlow 2.16.1 + tf-keras 호환 버전
Python 3.11 지원
"""

import numpy as np
import pandas as pd
import tensorflow as tf

# TensorFlow 2.16.1 keras 호환성 처리
try:
    from tensorflow import keras
    from tensorflow.keras import layers, Model, Input
    from tensorflow.keras.layers import *
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import *
    print("Using tensorflow.keras")
except:
    try:
        import tf_keras as keras
        from tf_keras import layers, Model, Input
        from tf_keras.layers import *
        from tf_keras.optimizers import Adam
        from tf_keras.callbacks import *
        print("Using tf_keras")
    except:
        print("ERROR: Please install tf-keras: pip install tf-keras")
        exit(1)

# tensorflow_probability는 선택사항
try:
    import tensorflow_probability as tfp
    HAS_TFP = True
except ImportError:
    print("Warning: tensorflow_probability not found. Some models may be limited.")
    HAS_TFP = False

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import warnings
import logging
from datetime import datetime
import os
import json
import pickle
import time

warnings.filterwarnings('ignore')

# ========================================
# 환경 설정
# ========================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# GPU 체크
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    logger.info(f"🎮 GPU 사용 가능: {len(gpus)}개")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except:
        pass
else:
    logger.info("💻 CPU 모드로 실행")

# 랜덤 시드
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

print("\n" + "="*60)
print("✅ TensorFlow 2.16.1 Ultimate 1700+ Predictor")
print(f"✅ TensorFlow Version: {tf.__version__}")
print(f"✅ Keras Backend: {keras.__name__}")
if HAS_TFP:
    print(f"✅ TF-Probability: Available")
print("="*60 + "\n")

# ========================================
# 데이터 전처리
# ========================================

class DataProcessor:
    """데이터 전처리"""
    
    def __init__(self, seq_len=100, pred_len=10):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scaler = RobustScaler()
        
    def load_and_process(self, filepath):
        """데이터 로드"""
        logger.info("📂 데이터 로딩...")
        
        # CSV 로드
        df = pd.read_csv(filepath)
        logger.info(f"✅ 원본 shape: {df.shape}")
        
        # 시간 변환
        df['CURRTIME'] = pd.to_datetime(df['CURRTIME'], format='%Y%m%d%H%M')
        df['TIME'] = pd.to_datetime(df['TIME'], format='%Y%m%d%H%M')
        df = df.sort_values('CURRTIME').reset_index(drop=True)
        
        # 타겟 생성
        df['TARGET'] = df['TOTALCNT'].shift(-self.pred_len)
        df['IS_HIGH'] = (df['TARGET'] >= 1700).astype(int)
        
        # 특성 생성
        df = self._create_features(df)
        df = df.dropna()
        
        logger.info(f"✅ 전처리 완료: {df.shape}")
        logger.info(f"✅ 1700+ 비율: {df['IS_HIGH'].mean():.2%} ({df['IS_HIGH'].sum()}개)")
        
        return df
    
    def _create_features(self, df):
        """특성 엔지니어링"""
        # 이동평균
        for window in [10, 30, 60]:
            df[f'MA_{window}'] = df['TOTALCNT'].rolling(window, min_periods=1).mean()
            df[f'STD_{window}'] = df['TOTALCNT'].rolling(window, min_periods=1).std()
        
        # 변화율
        df['CHANGE_10'] = df['TOTALCNT'].diff(10)
        df['PCT_CHANGE'] = df['TOTALCNT'].pct_change(10)
        
        # 핵심 비율 (1700+ 예측의 핵심!)
        df['RATIO_M14B_M10A'] = df['M14AM14B'] / (df['M14AM10A'] + 1e-6)
        df['RATIO_HIGH'] = (df['RATIO_M14B_M10A'] > 7).astype(int)
        
        # 시간 특성
        df['HOUR'] = df['CURRTIME'].dt.hour
        df['IS_DAWN'] = df['HOUR'].isin([4, 5, 6]).astype(int)
        
        # 임계값 신호
        df['M14B_HIGH'] = (df['M14AM14B'] > 350).astype(int)
        df['SPIKE_SIGNAL'] = ((df['M14B_HIGH'] == 1) & (df['IS_DAWN'] == 1)).astype(int)
        
        return df
    
    def create_sequences(self, df):
        """시퀀스 생성"""
        logger.info("🔄 시퀀스 생성 중...")
        
        # 특성 선택
        feature_cols = [col for col in df.columns 
                       if col not in ['CURRTIME', 'TIME', 'TARGET', 'IS_HIGH']]
        
        # 스케일링
        df_scaled = df.copy()
        df_scaled[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        df_scaled['TARGET_SCALED'] = self.scaler.fit_transform(df[['TARGET']])
        
        # 시퀀스 생성
        X, y, y_binary = [], [], []
        
        for i in range(len(df_scaled) - self.seq_len - self.pred_len):
            X.append(df_scaled[feature_cols].iloc[i:i+self.seq_len].values)
            y.append(df_scaled['TARGET_SCALED'].iloc[i+self.seq_len])
            y_binary.append(df['IS_HIGH'].iloc[i+self.seq_len])
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        y_binary = np.array(y_binary, dtype=np.float32)
        
        logger.info(f"✅ X shape: {X.shape}, y shape: {y.shape}")
        
        return X, y, y_binary, feature_cols

# ========================================
# 커스텀 손실 함수
# ========================================

def weighted_mse_loss(y_true, y_pred, weight_high=100.0):
    """1700+ 가중 MSE"""
    # 스케일된 1700은 약 0.5
    weights = tf.where(y_true > 0.5, weight_high, 1.0)
    return tf.reduce_mean(weights * tf.square(y_true - y_pred))

def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """Focal Loss"""
    diff = tf.abs(y_true - y_pred)
    focal_weight = tf.pow(diff, gamma)
    return tf.reduce_mean(alpha * focal_weight * tf.square(diff))

# ========================================
# 모델 구현
# ========================================

class ModelBuilder:
    """모델 빌더"""
    
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.seq_len = input_shape[0]
        self.n_features = input_shape[1]
    
    # 1. PatchTST - Transformer 기반
    def build_patch_tst(self, patch_len=10, d_model=128, n_heads=4):
        """Patch Time Series Transformer"""
        inputs = Input(shape=self.input_shape)
        
        # 패치 생성 (100 -> 10 patches of 10)
        n_patches = self.seq_len // patch_len
        x = tf.reshape(inputs, [-1, n_patches, patch_len * self.n_features])
        
        # 패치 임베딩
        x = Dense(d_model)(x)
        x = LayerNormalization()(x)
        
        # Multi-Head Attention
        for _ in range(2):
            attn_out = MultiHeadAttention(num_heads=n_heads, key_dim=d_model//n_heads)(x, x)
            x = Add()([x, attn_out])
            x = LayerNormalization()(x)
            
            # FFN
            ff = Dense(d_model * 2, activation='relu')(x)
            ff = Dense(d_model)(ff)
            x = Add()([x, ff])
            x = LayerNormalization()(x)
        
        x = GlobalAveragePooling1D()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(1)(x)
        
        return Model(inputs, outputs, name='PatchTST')
    
    # 2. Extreme Value Network
    def build_extreme_value_network(self):
        """극단값 특화 네트워크"""
        inputs = Input(shape=self.input_shape)
        
        # LSTM backbone
        x = LSTM(128, return_sequences=True)(inputs)
        x = LSTM(64)(x)
        
        # 두 개 브랜치
        # 1. 일반 예측
        normal_branch = Dense(32, activation='relu')(x)
        normal_pred = Dense(1, name='normal')(normal_branch)
        
        # 2. 극단값 보정
        extreme_branch = Dense(32, activation='relu')(x)
        extreme_branch = Dense(16, activation='relu')(extreme_branch)
        extreme_factor = Dense(1, activation='sigmoid', name='extreme')(extreme_branch)
        
        # 결합: 극단값일 때 증폭
        outputs = Add()([normal_pred, Multiply()([normal_pred, extreme_factor])])
        
        return Model(inputs, outputs, name='ExtremeValueNet')
    
    # 3. Enhanced Spike Detector
    def build_spike_detector(self):
        """급변 감지 모델"""
        inputs = Input(shape=self.input_shape)
        
        # CNN for pattern
        x = Conv1D(64, 3, padding='same', activation='relu')(inputs)
        x = Conv1D(64, 3, padding='same', activation='relu')(x)
        x = MaxPooling1D(2)(x)
        
        # Bi-LSTM
        x = Bidirectional(LSTM(64, return_sequences=True))(x)
        x = Bidirectional(LSTM(32))(x)
        
        # Spike detection
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1)(x)
        
        return Model(inputs, outputs, name='SpikeDetector')
    
    # 4. Mamba-inspired SSM
    def build_mamba_ssm(self):
        """State Space Model"""
        inputs = Input(shape=self.input_shape)
        
        # Simplified SSM using LSTM/GRU
        x = GRU(128, return_sequences=True)(inputs)
        x = GRU(64, return_sequences=True)(x)
        x = GRU(32)(x)
        
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1)(x)
        
        return Model(inputs, outputs, name='MambaSSM')
    
    # 5. Graph Neural Network
    def build_graph_nn(self):
        """컬럼 간 관계 학습"""
        inputs = Input(shape=self.input_shape)
        
        # Feature interaction
        x = Dense(128, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        
        # Temporal
        x = LSTM(64, return_sequences=True)(x)
        x = LSTM(32)(x)
        
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1)(x)
        
        return Model(inputs, outputs, name='GraphNN')
    
    # 6. Mixture of Experts
    def build_mixture_of_experts(self):
        """전문가 앙상블"""
        inputs = Input(shape=self.input_shape)
        
        # Shared feature extraction
        shared = LSTM(64, return_sequences=True)(inputs)
        shared = LSTM(32)(shared)
        
        # Expert 1: Low range (< 1400)
        e1 = Dense(16, activation='relu')(shared)
        e1_out = Dense(1)(e1)
        
        # Expert 2: Mid range (1400-1600)
        e2 = Dense(16, activation='relu')(shared)
        e2_out = Dense(1)(e2)
        
        # Expert 3: High range (1700+)
        e3 = Dense(16, activation='relu')(shared)
        e3_out = Dense(1)(e3)
        
        # Gating network
        gate = Dense(16, activation='relu')(shared)
        gate = Dense(3, activation='softmax')(gate)
        
        # Weighted sum
        experts = tf.stack([e1_out, e2_out, e3_out], axis=1)
        outputs = tf.reduce_sum(experts * gate[:, :, None], axis=1)
        
        return Model(inputs, outputs, name='MixtureOfExperts')
    
    # 7. TCN (Temporal CNN)
    def build_tcn(self):
        """Temporal Convolutional Network"""
        inputs = Input(shape=self.input_shape)
        
        x = inputs
        for dilation in [1, 2, 4, 8]:
            conv = Conv1D(64, 3, dilation_rate=dilation, padding='causal', activation='relu')(x)
            conv = Dropout(0.1)(conv)
            if x.shape[-1] != conv.shape[-1]:
                x = Conv1D(64, 1)(x)
            x = Add()([x, conv])
        
        x = GlobalAveragePooling1D()(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1)(x)
        
        return Model(inputs, outputs, name='TCN')
    
    # 8. Bi-LSTM with Attention
    def build_bilstm_attention(self):
        """양방향 LSTM + Attention"""
        inputs = Input(shape=self.input_shape)
        
        # Bi-LSTM
        x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
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
        outputs = Dense(1)(x)
        
        return Model(inputs, outputs, name='BiLSTM_Attention')
    
    # 9. Autoformer
    def build_autoformer(self):
        """Auto-correlation Transformer"""
        inputs = Input(shape=self.input_shape)
        
        # Decomposition
        trend = AveragePooling1D(10, strides=1, padding='same')(inputs)
        seasonal = inputs - trend
        
        # Process
        trend_out = Conv1D(64, 1, activation='relu')(trend)
        trend_out = GlobalAveragePooling1D()(trend_out)
        
        seasonal_out = Conv1D(64, 1, activation='relu')(seasonal)
        seasonal_out = GlobalAveragePooling1D()(seasonal_out)
        
        # Combine
        x = Concatenate()([trend_out, seasonal_out])
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1)(x)
        
        return Model(inputs, outputs, name='Autoformer')
    
    # 10. Simple but Strong Baseline
    def build_baseline_lstm(self):
        """강력한 LSTM 베이스라인"""
        inputs = Input(shape=self.input_shape)
        
        x = LSTM(256, return_sequences=True)(inputs)
        x = Dropout(0.2)(x)
        x = LSTM(128, return_sequences=True)(x)
        x = Dropout(0.2)(x)
        x = LSTM(64)(x)
        x = Dropout(0.2)(x)
        
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.1)(x)
        outputs = Dense(1)(x)
        
        return Model(inputs, outputs, name='Baseline_LSTM')

# ========================================
# 학습 및 평가
# ========================================

class Trainer:
    """모델 학습"""
    
    def __init__(self, models_dict):
        self.models_dict = models_dict
        self.results = {}
    
    def train_model(self, model, X_train, y_train, X_val, y_val, 
                   model_name, epochs=30, batch_size=256):
        """개별 모델 학습"""
        
        logger.info(f"\n{'='*50}")
        logger.info(f"🚀 Training {model_name}")
        logger.info(f"{'='*50}")
        
        # 컴파일
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=weighted_mse_loss,
            metrics=['mae']
        )
        
        # 콜백
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        # 학습
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, model, X_test, y_test, y_test_binary, model_name):
        """모델 평가"""
        
        # 예측
        y_pred = model.predict(X_test, verbose=0).flatten()
        
        # MAE, RMSE
        mae = np.mean(np.abs(y_test - y_pred))
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        
        # 1700+ 성능
        high_mask = y_test_binary == 1
        if high_mask.sum() > 0:
            mae_high = np.mean(np.abs(y_test[high_mask] - y_pred[high_mask]))
            
            # Precision/Recall
            threshold = 0.5  # 스케일된 1700
            pred_high = (y_pred > threshold).astype(int)
            
            tp = np.sum((pred_high == 1) & (high_mask == 1))
            fp = np.sum((pred_high == 1) & (high_mask == 0)) 
            fn = np.sum((pred_high == 0) & (high_mask == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        else:
            mae_high = precision = recall = f1 = 0
        
        results = {
            'mae': mae,
            'rmse': rmse,
            'mae_1700+': mae_high,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        self.results[model_name] = results
        
        logger.info(f"\n📊 {model_name} Results:")
        logger.info(f"   MAE: {mae:.4f}")
        logger.info(f"   RMSE: {rmse:.4f}")
        logger.info(f"   1700+ MAE: {mae_high:.4f}")
        logger.info(f"   Precision: {precision:.2%}")
        logger.info(f"   Recall: {recall:.2%}")
        logger.info(f"   F1: {f1:.2%}")
        
        return results

# ========================================
# 메인 실행
# ========================================

def main():
    """메인 함수"""
    
    print("\n" + "="*80)
    print("🚀 Ultimate 1700+ Predictor Suite v7.1")
    print("="*80 + "\n")
    
    # 1. 데이터 로드
    processor = DataProcessor(seq_len=100, pred_len=10)
    
    # 데이터 파일 경로 확인
    data_path = 'data/20240201_TO_202507281705.csv'
    if not os.path.exists(data_path):
        # 업로드된 파일 사용
        data_path = '/mnt/user-data/uploads/20250807_DATA.CSV'
        if not os.path.exists(data_path):
            data_path = '/mnt/user-data/uploads/gs.CSV'
    
    df = processor.load_and_process(data_path)
    
    # 2. 시퀀스 생성
    X, y, y_binary, feature_cols = processor.create_sequences(df)
    
    # 3. 데이터 분할
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    y_test_binary = y_binary[train_size+val_size:]
    
    logger.info(f"\n📊 데이터 분할:")
    logger.info(f"   Train: {X_train.shape}")
    logger.info(f"   Val: {X_val.shape}")
    logger.info(f"   Test: {X_test.shape}")
    logger.info(f"   1700+ in test: {y_test_binary.sum()}개")
    
    # 4. 모델 생성
    builder = ModelBuilder(input_shape=(100, X.shape[2]))
    
    models = {
        'PatchTST': builder.build_patch_tst(),
        'ExtremeValueNet': builder.build_extreme_value_network(),
        'SpikeDetector': builder.build_spike_detector(),
        'MambaSSM': builder.build_mamba_ssm(),
        'GraphNN': builder.build_graph_nn(),
        'MixtureOfExperts': builder.build_mixture_of_experts(),
        'TCN': builder.build_tcn(),
        'BiLSTM_Attention': builder.build_bilstm_attention(),
        'Autoformer': builder.build_autoformer(),
        'Baseline_LSTM': builder.build_baseline_lstm()
    }
    
    logger.info(f"\n✅ {len(models)}개 모델 생성 완료\n")
    
    # 5. 학습 및 평가
    trainer = Trainer(models)
    
    for name, model in models.items():
        try:
            # 학습
            history = trainer.train_model(
                model, X_train, y_train, X_val, y_val, 
                name, epochs=20, batch_size=256
            )
            
            # 평가
            trainer.evaluate_model(
                model, X_test, y_test, y_test_binary, name
            )
            
        except Exception as e:
            logger.error(f"❌ {name} 실패: {str(e)}")
            continue
    
    # 6. 결과 요약
    print("\n" + "="*80)
    print("📊 최종 결과 요약")
    print("="*80)
    
    # F1 기준 정렬
    sorted_results = sorted(trainer.results.items(), 
                           key=lambda x: x[1]['f1'], 
                           reverse=True)
    
    print("\n🏆 모델 순위 (F1 Score 기준):")
    print("-"*60)
    for i, (name, metrics) in enumerate(sorted_results, 1):
        print(f"{i:2d}. {name:20s} | F1: {metrics['f1']:.2%} | "
              f"MAE: {metrics['mae']:.4f} | "
              f"Recall: {metrics['recall']:.2%}")
    
    # 최고 모델
    best_model = sorted_results[0][0]
    best_metrics = sorted_results[0][1]
    
    print("\n🥇 최고 성능 모델:", best_model)
    print(f"   - F1 Score: {best_metrics['f1']:.2%}")
    print(f"   - Precision: {best_metrics['precision']:.2%}")
    print(f"   - Recall: {best_metrics['recall']:.2%}")
    print(f"   - 1700+ MAE: {best_metrics['mae_1700+']:.4f}")
    
    # 결과 저장
    with open('results_v7.json', 'w') as f:
        json.dump({
            'results': trainer.results,
            'best_model': best_model,
            'timestamp': datetime.now().isoformat()
        }, f, indent=4)
    
    print("\n✅ 완료! 결과 저장: results_v7.json")
    
    return trainer.results

if __name__ == "__main__":
    results = main()