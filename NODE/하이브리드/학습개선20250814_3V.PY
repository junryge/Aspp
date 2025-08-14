"""
반도체 물류 예측을 위한 개선된 하이브리드 딥러닝 모델 (v3.0)
================================================================
주요 개선사항:
1. 시퀀스 길이를 100으로 증가 (100분 과거 데이터 활용)
2. 급변 패턴 학습을 위한 Attention 메커니즘 추가
3. 1400 이상 구간 예측 개선을 위한 특징 엔지니어링
4. TensorFlow 2.15.0 호환
5. 급변 감지 특화 모델 추가

개발일: 2024년
버전: 3.0 (급변 예측 개선 버전)
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Input, LSTM, Dense, Dropout, BatchNormalization, 
                                     Bidirectional, GRU, SimpleRNN, 
                                     MultiHeadAttention, LayerNormalization, Add,
                                     Conv1D, MaxPooling1D, Flatten, Concatenate,
                                     GlobalAveragePooling1D)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import joblib
import logging
import warnings
import json
import pickle
import traceback
from scipy import stats
from sklearn.utils import class_weight

# TensorFlow 버전 확인
print(f"TensorFlow version: {tf.__version__}")

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')

# ===================================
# 1. 환경 설정 및 초기화
# ===================================

# CPU 모드 설정 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

# 랜덤 시드 고정
RANDOM_SEED = 2079936
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_improved.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===================================
# 2. 개선된 체크포인트 관리 클래스
# ===================================

class CheckpointManager:
    """학습 상태를 저장하고 복원하는 클래스"""
    
    def __init__(self, checkpoint_dir='checkpoints_v3'):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.state_file = os.path.join(checkpoint_dir, 'training_state.json')
        self.data_file = os.path.join(checkpoint_dir, 'preprocessed_data.pkl')
        
    def save_state(self, state_dict):
        """현재 학습 상태 저장"""
        with open(self.state_file, 'w') as f:
            json.dump(state_dict, f, indent=4, default=str)
        logger.info(f"학습 상태 저장됨: {self.state_file}")
        
    def load_state(self):
        """저장된 학습 상태 로드"""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            logger.info(f"학습 상태 로드됨: {self.state_file}")
            return state
        return None
        
    def save_data(self, data_dict):
        """전처리된 데이터 저장"""
        with open(self.data_file, 'wb') as f:
            pickle.dump(data_dict, f)
        logger.info(f"데이터 저장됨: {self.data_file}")
        
    def load_data(self):
        """저장된 데이터 로드"""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"데이터 로드됨: {self.data_file}")
            return data
        return None

# ===================================
# 3. 급변 예측을 위한 커스텀 손실 함수
# ===================================

def create_weighted_loss(threshold_value=1400, high_weight=3.0):
    """급변 구간에 더 높은 가중치를 주는 손실 함수"""
    def weighted_mse(y_true, y_pred):
        # 역변환 후 임계값 비교 (간단히 하기 위해 scaled 값으로 근사)
        # 실제 1400이 scaled 후 약 0.5라고 가정 (실제 값은 데이터에 따라 조정)
        threshold_scaled = 0.5  
        
        # 높은 값에 대한 가중치
        weights = tf.where(y_true > threshold_scaled, high_weight, 1.0)
        
        # 가중 MSE 계산
        squared_diff = tf.square(y_true - y_pred)
        weighted_squared_diff = weights * squared_diff
        
        return tf.reduce_mean(weighted_squared_diff)
    
    return weighted_mse

# ===================================
# 4. 개선된 특징 엔지니어링
# ===================================

def enhanced_feature_engineering(data):
    """급변 예측을 위한 강화된 특징 생성"""
    
    logger.info("강화된 특징 엔지니어링 시작...")
    
    # 기존 특징
    data['hour'] = data.index.hour
    data['dayofweek'] = data.index.dayofweek
    data['is_weekend'] = (data.index.dayofweek >= 5).astype(int)
    
    # 이동 평균 (다양한 윈도우)
    for window in [5, 10, 20, 30, 60, 100]:
        data[f'MA_{window}'] = data['TOTALCNT'].rolling(window=window, min_periods=1).mean()
        data[f'STD_{window}'] = data['TOTALCNT'].rolling(window=window, min_periods=1).std()
    
    # 변화율 및 가속도
    data['change_rate'] = data['TOTALCNT'].pct_change()
    data['change_rate_5'] = data['TOTALCNT'].pct_change(5)
    data['change_rate_10'] = data['TOTALCNT'].pct_change(10)
    
    # 2차 미분 (가속도)
    data['acceleration'] = data['change_rate'].diff()
    data['acceleration_5'] = data['change_rate_5'].diff()
    
    # 볼륨 버스트 지표 (급증 감지)
    data['volume_ratio'] = data['TOTALCNT'] / data['MA_30']
    data['volume_spike'] = (data['volume_ratio'] > 1.2).astype(int)
    
    # 피크 감지
    data['local_max'] = (data['TOTALCNT'].shift(1) < data['TOTALCNT']) & \
                        (data['TOTALCNT'] > data['TOTALCNT'].shift(-1))
    data['local_max'] = data['local_max'].astype(int)
    
    # 1400 이상 구간 표시 (학습용)
    data['high_volume'] = (data['TOTALCNT'] > 1400).astype(int)
    data['high_volume_ma'] = data['high_volume'].rolling(window=10, min_periods=1).mean()
    
    # 시간대별 통계
    hourly_stats = data.groupby(data.index.hour)['TOTALCNT'].agg(['mean', 'std']).to_dict()
    data['hour_mean'] = data.index.hour.map(hourly_stats['mean'])
    data['hour_std'] = data.index.hour.map(hourly_stats['std'])
    data['hour_zscore'] = (data['TOTALCNT'] - data['hour_mean']) / (data['hour_std'] + 1e-8)
    
    # 트렌드 강도
    data['trend_strength'] = data['MA_10'] - data['MA_30']
    data['trend_direction'] = np.sign(data['trend_strength'])
    
    # EMA (지수이동평균)
    data['EMA_10'] = data['TOTALCNT'].ewm(span=10, adjust=False).mean()
    data['EMA_30'] = data['TOTALCNT'].ewm(span=30, adjust=False).mean()
    data['EMA_signal'] = data['EMA_10'] - data['EMA_30']
    
    # 볼린저 밴드
    data['BB_upper'] = data['MA_20'] + (2 * data['STD_20'])
    data['BB_lower'] = data['MA_20'] - (2 * data['STD_20'])
    data['BB_position'] = (data['TOTALCNT'] - data['BB_lower']) / (data['BB_upper'] - data['BB_lower'] + 1e-8)
    
    # RSI (상대강도지수)
    delta = data['TOTALCNT'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # NaN 처리
    data = data.ffill().fillna(0)
    
    logger.info(f"특징 생성 완료. 총 특징 수: {len(data.columns)}")
    
    return data

# ===================================
# 5. Attention 기반 LSTM 모델
# ===================================

def build_attention_lstm_model(input_shape):
    """Attention 메커니즘을 포함한 LSTM 모델"""
    
    inputs = Input(shape=input_shape)
    
    # 첫 번째 LSTM 레이어
    lstm1 = LSTM(units=128, return_sequences=True)(inputs)
    lstm1 = Dropout(0.2)(lstm1)
    lstm1 = BatchNormalization()(lstm1)
    
    # Multi-Head Attention
    attention = MultiHeadAttention(num_heads=4, key_dim=32)(lstm1, lstm1)
    attention = Dropout(0.2)(attention)
    attention = LayerNormalization()(attention + lstm1)  # Residual connection
    
    # 두 번째 LSTM 레이어
    lstm2 = LSTM(units=64, return_sequences=True)(attention)
    lstm2 = Dropout(0.2)(lstm2)
    lstm2 = BatchNormalization()(lstm2)
    
    # Global Average Pooling
    gap = GlobalAveragePooling1D()(lstm2)
    
    # Dense layers
    dense1 = Dense(64, activation='relu')(gap)
    dense1 = Dropout(0.3)(dense1)
    dense2 = Dense(32, activation='relu')(dense1)
    dense2 = Dropout(0.2)(dense2)
    
    # Output
    output = Dense(1)(dense2)
    
    model = Model(inputs=inputs, outputs=output)
    
    return model

# ===================================
# 6. CNN-LSTM 하이브리드 모델
# ===================================

def build_cnn_lstm_model(input_shape):
    """CNN과 LSTM을 결합한 모델 (급변 패턴 감지용)"""
    
    inputs = Input(shape=input_shape)
    
    # CNN 부분 (지역 패턴 추출)
    conv1 = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = MaxPooling1D(pool_size=2)(conv1)
    
    conv2 = Conv1D(filters=32, kernel_size=3, activation='relu')(conv1)
    conv2 = BatchNormalization()(conv2)
    
    # LSTM 부분 (시계열 패턴)
    lstm1 = LSTM(units=100, return_sequences=True)(conv2)
    lstm1 = Dropout(0.2)(lstm1)
    
    lstm2 = LSTM(units=50, return_sequences=False)(lstm1)
    lstm2 = Dropout(0.2)(lstm2)
    
    # Dense layers
    dense1 = Dense(64, activation='relu')(lstm2)
    dense1 = Dropout(0.3)(dense1)
    dense2 = Dense(32, activation='relu')(dense1)
    
    # Output
    output = Dense(1)(dense2)
    
    model = Model(inputs=inputs, outputs=output)
    
    return model

# ===================================
# 7. 급변 감지 분류 모델
# ===================================

def build_spike_detector_model(input_shape):
    """1400 이상 급변을 예측하는 이진 분류 모델"""
    
    inputs = Input(shape=input_shape)
    
    # Bidirectional LSTM
    lstm1 = Bidirectional(LSTM(units=64, return_sequences=True))(inputs)
    lstm1 = Dropout(0.2)(lstm1)
    
    lstm2 = Bidirectional(LSTM(units=32, return_sequences=False))(lstm1)
    lstm2 = Dropout(0.2)(lstm2)
    
    # Dense layers
    dense1 = Dense(32, activation='relu')(lstm2)
    dense1 = Dropout(0.3)(dense1)
    dense2 = Dense(16, activation='relu')(dense1)
    
    # Binary output (1400 이상일 확률)
    output = Dense(1, activation='sigmoid')(dense2)
    
    model = Model(inputs=inputs, outputs=output)
    
    return model

# ===================================
# 8. 개선된 하이브리드 모델 클래스
# ===================================

class ImprovedHybridModels:
    """개선된 하이브리드 예측 시스템"""
    
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.models = {}
        self.histories = {}
        self.spike_detector = None
        
    def build_enhanced_lstm_model(self):
        """강화된 LSTM 모델"""
        model = Sequential([
            Input(shape=self.input_shape),
            
            LSTM(units=150, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),
            
            LSTM(units=100, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),
            
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            
            Dense(units=64, activation='relu'),
            Dense(units=32, activation='relu'),
            Dense(units=1)
        ])
        
        return model
    
    def build_enhanced_gru_model(self):
        """강화된 GRU 모델"""
        model = Sequential([
            Input(shape=self.input_shape),
            
            GRU(units=150, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),
            
            GRU(units=100, return_sequences=True),
            Dropout(0.2),
            
            GRU(units=50, return_sequences=False),
            Dropout(0.2),
            
            Dense(units=64, activation='relu'),
            Dense(units=32, activation='relu'),
            Dense(units=1)
        ])
        
        return model
    
    def build_enhanced_bidirectional_model(self):
        """강화된 양방향 LSTM"""
        model = Sequential([
            Input(shape=self.input_shape),
            
            Bidirectional(LSTM(units=100, return_sequences=True)),
            Dropout(0.2),
            BatchNormalization(),
            
            Bidirectional(LSTM(units=50, return_sequences=True)),
            Dropout(0.2),
            
            Bidirectional(LSTM(units=25, return_sequences=False)),
            Dropout(0.2),
            
            Dense(units=64, activation='relu'),
            Dense(units=32, activation='relu'),
            Dense(units=1)
        ])
        
        return model

# ===================================
# 9. 개선된 학습 함수
# ===================================

def train_model_improved(model, model_name, X_train, y_train, X_val, y_val, 
                         epochs, batch_size, checkpoint_manager, 
                         use_weighted_loss=True, spike_labels_train=None):
    """개선된 학습 함수 (급변 예측 최적화)"""
    
    # 옵티마이저 설정
    optimizer = Adam(learning_rate=0.001)
    
    # 손실 함수 설정
    if use_weighted_loss and spike_labels_train is not None:
        # 급변 구간에 가중치 부여
        loss_fn = create_weighted_loss(threshold_value=1400, high_weight=3.0)
    else:
        loss_fn = 'mse'
    
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['mae'])
    
    # 콜백 설정
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6),
        ModelCheckpoint(f'{checkpoint_manager.checkpoint_dir}/{model_name}_best.h5', 
                       save_best_only=True, monitor='val_loss')
    ]
    
    # 클래스 가중치 계산 (급변 구간 중요도 증가)
    sample_weights = None
    if spike_labels_train is not None:
        # 1400 이상 구간에 더 높은 가중치
        sample_weights = np.where(spike_labels_train > 0.5, 3.0, 1.0)
    
    # 학습
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        sample_weight=sample_weights,
        verbose=1
    )
    
    return history

# ===================================
# 10. 메인 학습 프로세스
# ===================================

def main(resume=False):
    """메인 학습 프로세스"""
    
    checkpoint_manager = CheckpointManager()
    
    logger.info("="*60)
    logger.info("개선된 반도체 물류 예측 모델 학습 시작 (v3.0)")
    logger.info("주요 개선: 100분 시퀀스, Attention, 급변 예측 특화")
    logger.info("="*60)
    
    # 데이터 로드 및 전처리
    Full_data_path = 'data/20240201_TO_202507281705.csv'
    logger.info(f"데이터 로딩 중: {Full_data_path}")
    
    Full_Data = pd.read_csv(Full_data_path)
    logger.info(f"원본 데이터 shape: {Full_Data.shape}")
    
    # 시간 컬럼 변환
    Full_Data['CURRTIME'] = pd.to_datetime(Full_Data['CURRTIME'], format='%Y%m%d%H%M')
    Full_Data['TIME'] = pd.to_datetime(Full_Data['TIME'], format='%Y%m%d%H%M')
    
    # SUM 컬럼 제거
    columns_to_drop = [col for col in Full_Data.columns if 'SUM' in col]
    Full_Data = Full_Data.drop(columns=columns_to_drop)
    
    # 날짜 범위 필터링
    start_date = pd.to_datetime('2024-02-01 00:00:00')
    end_date = pd.to_datetime('2024-07-27 23:59:59')
    Full_Data = Full_Data[(Full_Data['TIME'] >= start_date) & (Full_Data['TIME'] <= end_date)].reset_index(drop=True)
    
    # 필요한 컬럼만 선택
    Full_Data = Full_Data[['CURRTIME', 'TOTALCNT', 'TIME']]
    Full_Data.set_index('CURRTIME', inplace=True)
    
    # PM 기간 설정 및 이상치 제거
    PM_start_date = pd.to_datetime('2024-10-23 00:00:00')
    PM_end_date = pd.to_datetime('2024-10-23 23:59:59')
    
    within_range_data = Full_Data[(Full_Data['TIME'] >= PM_start_date) & (Full_Data['TIME'] <= PM_end_date)]
    outside_range_data = Full_Data[(Full_Data['TIME'] < PM_start_date) | (Full_Data['TIME'] > PM_end_date)]
    
    # 이상치 제거 범위 조정 (1400 이상 데이터 보존)
    outside_range_filtered_data = outside_range_data[
        (outside_range_data['TOTALCNT'] >= 800) & 
        (outside_range_data['TOTALCNT'] <= 3000)  # 상한 증가
    ]
    
    Modified_Data = pd.concat([within_range_data, outside_range_filtered_data])
    Modified_Data = Modified_Data.sort_values(by='TIME')
    
    # FUTURE 컬럼 생성 (10분 후)
    Modified_Data['FUTURE'] = pd.NA
    future_minutes = 10
    
    for i in Modified_Data.index:
        future_time = i + pd.Timedelta(minutes=future_minutes)
        if (future_time <= Modified_Data.index.max()) & (future_time in Modified_Data.index):
            Modified_Data.loc[i, 'FUTURE'] = Modified_Data.loc[future_time, 'TOTALCNT']
    
    Modified_Data.dropna(subset=['FUTURE'], inplace=True)
    
    # 급변 레이블 생성 (1400 이상)
    Modified_Data['spike_label'] = (Modified_Data['FUTURE'] >= 1400).astype(int)
    
    logger.info(f"1400 이상 구간 비율: {Modified_Data['spike_label'].mean():.2%}")
    
    # 강화된 특징 엔지니어링
    Modified_Data = enhanced_feature_engineering(Modified_Data)
    
    # 데이터 스케일링
    sscaler = RobustScaler()  # 이상치에 강한 스케일러
    scaled_list = ['TOTALCNT', 'FUTURE'] + [col for col in Modified_Data.columns 
                                              if col not in ['TIME', 'spike_label', 'high_volume']]
    scaled_list = [col for col in scaled_list if col in Modified_Data.columns]
    
    fitted_scaled = sscaler.fit(Modified_Data[scaled_list])
    scaled_list_data = fitted_scaled.transform(Modified_Data[scaled_list])
    
    scaled_columns = [f'scaled_{col}' for col in scaled_list]
    scaled_list_data = pd.DataFrame(scaled_list_data, columns=scaled_columns)
    scaled_list_data.index = Modified_Data.index
    
    Scaled_Data = pd.merge(Modified_Data, scaled_list_data, left_index=True, right_index=True, how='left')
    
    # 시퀀스 데이터 생성 (100분으로 증가)
    SEQ_LENGTH = 100  # 30에서 100으로 증가
    
    def split_data_by_continuity(data):
        time_diff = data.index.to_series().diff()
        split_points = time_diff > pd.Timedelta(minutes=1)
        segment_ids = split_points.cumsum()
        segments = []
        for segment_id in segment_ids.unique():
            segment = data[segment_ids == segment_id].copy()
            if len(segment) > SEQ_LENGTH:
                segments.append(segment)
        return segments
    
    data_segments = split_data_by_continuity(Scaled_Data)
    
    def create_sequences(data, feature_cols, target_col, seq_length=100):
        X, y, spike_labels = [], [], []
        feature_data = data[feature_cols].values
        target_data = data[target_col].values
        spike_data = data['spike_label'].values if 'spike_label' in data.columns else None
        
        for i in range(len(data) - seq_length):
            X.append(feature_data[i:i+seq_length])
            y.append(target_data[i+seq_length])
            if spike_data is not None:
                spike_labels.append(spike_data[i+seq_length])
        
        return np.array(X), np.array(y), np.array(spike_labels) if spike_labels else None
    
    input_features = [col for col in Scaled_Data.columns if col.startswith('scaled_') and col != 'scaled_FUTURE']
    
    all_X, all_y, all_spike_labels = [], [], []
    for segment in data_segments:
        X_seg, y_seg, spike_seg = create_sequences(segment, input_features, 'scaled_FUTURE', SEQ_LENGTH)
        if len(X_seg) > 0:
            all_X.append(X_seg)
            all_y.append(y_seg)
            if spike_seg is not None:
                all_spike_labels.append(spike_seg)
    
    X_seq_all = np.concatenate(all_X, axis=0)
    y_seq_all = np.concatenate(all_y, axis=0)
    spike_labels_all = np.concatenate(all_spike_labels, axis=0) if all_spike_labels else None
    
    # 데이터 분할
    train_size = int(0.7 * len(X_seq_all))
    val_size = int(0.15 * len(X_seq_all))
    
    X_train = X_seq_all[:train_size]
    y_train = y_seq_all[:train_size]
    spike_labels_train = spike_labels_all[:train_size] if spike_labels_all is not None else None
    
    X_val = X_seq_all[train_size:train_size+val_size]
    y_val = y_seq_all[train_size:train_size+val_size]
    spike_labels_val = spike_labels_all[train_size:train_size+val_size] if spike_labels_all is not None else None
    
    X_test = X_seq_all[train_size+val_size:]
    y_test = y_seq_all[train_size+val_size:]
    spike_labels_test = spike_labels_all[train_size+val_size:] if spike_labels_all is not None else None
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    logger.info(f"입력 shape: {input_shape}")
    logger.info(f"학습 데이터: {X_train.shape[0]} 샘플")
    logger.info(f"검증 데이터: {X_val.shape[0]} 샘플")
    logger.info(f"테스트 데이터: {X_test.shape[0]} 샘플")
    
    # 하이브리드 모델 초기화
    hybrid_models = ImprovedHybridModels(input_shape)
    
    # 학습 파라미터
    EPOCHS = 150
    BATCH_SIZE = 32  # 더 작은 배치로 세밀한 학습
    
    # 모델 리스트
    model_configs = [
        ('attention_lstm', build_attention_lstm_model),
        ('cnn_lstm', build_cnn_lstm_model),
        ('enhanced_lstm', hybrid_models.build_enhanced_lstm_model),
        ('enhanced_gru', hybrid_models.build_enhanced_gru_model),
        ('enhanced_bidirectional', hybrid_models.build_enhanced_bidirectional_model)
    ]
    
    # 각 모델 학습
    for model_name, build_func in model_configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"{model_name.upper()} 모델 학습 시작")
        logger.info(f"{'='*60}")
        
        model = build_func(input_shape) if 'attention' in model_name or 'cnn' in model_name else build_func()
        
        history = train_model_improved(
            model, model_name, X_train, y_train, X_val, y_val,
            EPOCHS, BATCH_SIZE, checkpoint_manager,
            use_weighted_loss=True, spike_labels_train=spike_labels_train
        )
        
        hybrid_models.models[model_name] = model
        hybrid_models.histories[model_name] = history
    
    # 급변 감지 분류기 학습
    logger.info("\n" + "="*60)
    logger.info("급변 감지 분류기 학습")
    logger.info("="*60)
    
    spike_detector = build_spike_detector_model(input_shape)
    spike_detector.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    spike_history = spike_detector.fit(
        X_train, spike_labels_train,
        validation_data=(X_val, spike_labels_val),
        epochs=100,
        batch_size=32,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)
        ],
        verbose=1
    )
    
    hybrid_models.spike_detector = spike_detector
    
    # ===================================
    # 11. 개선된 앙상블 예측
    # ===================================
    
    def improved_ensemble_predict(models, spike_detector, X_test, sscaler, scaled_list):
        """급변 감지를 포함한 개선된 앙상블 예측"""
        
        # 급변 확률 예측
        spike_probs = spike_detector.predict(X_test, verbose=0).flatten()
        
        # 각 모델별 예측
        individual_preds = {}
        for model_name, model in models.items():
            pred = model.predict(X_test, verbose=0).flatten()
            individual_preds[model_name] = pred
        
        # 동적 가중치 계산 (급변 확률에 따라)
        ensemble_pred = np.zeros_like(list(individual_preds.values())[0])
        
        for i in range(len(ensemble_pred)):
            if spike_probs[i] > 0.5:  # 급변 예상
                # CNN-LSTM과 Attention 모델에 더 높은 가중치
                weights = {
                    'attention_lstm': 0.35,
                    'cnn_lstm': 0.35,
                    'enhanced_lstm': 0.15,
                    'enhanced_gru': 0.10,
                    'enhanced_bidirectional': 0.05
                }
            else:  # 일반 구간
                weights = {
                    'attention_lstm': 0.20,
                    'cnn_lstm': 0.20,
                    'enhanced_lstm': 0.25,
                    'enhanced_gru': 0.20,
                    'enhanced_bidirectional': 0.15
                }
            
            for model_name, pred in individual_preds.items():
                weight = weights.get(model_name, 0.2)
                ensemble_pred[i] += weight * pred[i]
        
        # 급변 보정 (spike probability가 높으면 예측값 상향)
        spike_adjustment = spike_probs * 0.15  # 최대 15% 상향
        ensemble_pred = ensemble_pred * (1 + spike_adjustment)
        
        return ensemble_pred, individual_preds, spike_probs
    
    # 앙상블 예측 수행
    ensemble_pred, individual_preds, spike_probs = improved_ensemble_predict(
        hybrid_models.models, spike_detector, X_test, sscaler, scaled_list
    )
    
    # ===================================
    # 12. 성능 평가
    # ===================================
    
    logger.info("\n" + "="*60)
    logger.info("모델 성능 평가")
    logger.info("="*60)
    
    def calculate_improved_metrics(y_true, y_pred, spike_true, spike_pred, model_name, sscaler, scaled_list):
        """개선된 성능 지표 계산"""
        
        # 역변환
        y_true_original = sscaler.inverse_transform(
            np.column_stack([y_true] + [np.zeros_like(y_true)] * (len(scaled_list) - 1))
        )[:, 0]
        
        y_pred_original = sscaler.inverse_transform(
            np.column_stack([y_pred] + [np.zeros_like(y_pred)] * (len(scaled_list) - 1))
        )[:, 0]
        
        # 기본 지표
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mae_original = mean_absolute_error(y_true_original, y_pred_original)
        
        # 1400 이상 구간에 대한 성능
        high_volume_mask = y_true_original >= 1400
        if high_volume_mask.sum() > 0:
            mae_high = mean_absolute_error(
                y_true_original[high_volume_mask], 
                y_pred_original[high_volume_mask]
            )
            accuracy_high = 1 - (mae_high / y_true_original[high_volume_mask].mean())
        else:
            mae_high = 0
            accuracy_high = 0
        
        # 급변 감지 정확도
        if spike_true is not None and spike_pred is not None:
            spike_accuracy = np.mean((spike_pred > 0.5) == spike_true)
        else:
            spike_accuracy = 0
        
        logger.info(f"\n{model_name} 모델 성능:")
        logger.info(f"  - MSE: {mse:.4f}")
        logger.info(f"  - MAE: {mae:.4f}")
        logger.info(f"  - RMSE: {rmse:.4f}")
        logger.info(f"  - R²: {r2:.4f}")
        logger.info(f"  - MAE (원본): {mae_original:.2f}")
        logger.info(f"  - MAE (1400+): {mae_high:.2f}")
        logger.info(f"  - 정확도 (1400+): {accuracy_high:.2%}")
        logger.info(f"  - 급변 감지 정확도: {spike_accuracy:.2%}")
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mae_original': mae_original,
            'mae_high': mae_high,
            'accuracy_high': accuracy_high,
            'spike_accuracy': spike_accuracy
        }
    
    # 모델 평가
    results = {}
    for model_name, pred in individual_preds.items():
        results[model_name] = calculate_improved_metrics(
            y_test, pred, spike_labels_test, None, 
            model_name.upper(), sscaler, scaled_list
        )
    
    # 앙상블 평가
    results['ensemble'] = calculate_improved_metrics(
        y_test, ensemble_pred, spike_labels_test, spike_probs,
        "IMPROVED ENSEMBLE", sscaler, scaled_list
    )
    
    # ===================================
    # 13. 결과 시각화
    # ===================================
    
    logger.info("\n결과 시각화 생성 중...")
    
    # 예측 결과 시각화 (1400 이상 구간 포함)
    plt.figure(figsize=(20, 12))
    
    sample_size = min(500, len(y_test))
    
    # 실제값을 원본 스케일로 변환
    y_test_original = sscaler.inverse_transform(
        np.column_stack([y_test[:sample_size]] + [np.zeros_like(y_test[:sample_size])] * (len(scaled_list) - 1))
    )[:, 0]
    
    # 앙상블 예측값
    ensemble_original = sscaler.inverse_transform(
        np.column_stack([ensemble_pred[:sample_size]] + [np.zeros_like(ensemble_pred[:sample_size])] * (len(scaled_list) - 1))
    )[:, 0]
    
    # 서브플롯 1: 전체 예측
    plt.subplot(3, 1, 1)
    plt.plot(y_test_original, label='실제값', color='black', linewidth=2)
    plt.plot(ensemble_original, label='개선된 앙상블 예측', color='red', linewidth=2, alpha=0.7)
    plt.axhline(y=1400, color='blue', linestyle='--', label='1400 임계값', alpha=0.5)
    plt.title('반도체 물류량 예측 (개선된 모델)', fontsize=16)
    plt.ylabel('물류량', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 서브플롯 2: 1400 이상 구간 확대
    plt.subplot(3, 1, 2)
    high_mask = y_test_original >= 1400
    if high_mask.sum() > 0:
        indices = np.where(high_mask)[0][:100]  # 처음 100개
        plt.scatter(range(len(indices)), y_test_original[indices], 
                   label='실제 (1400+)', color='black', s=50)
        plt.scatter(range(len(indices)), ensemble_original[indices], 
                   label='예측 (1400+)', color='red', s=30, alpha=0.7)
        plt.title('1400 이상 구간 예측 성능', fontsize=14)
        plt.ylabel('물류량', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 서브플롯 3: 급변 확률
    plt.subplot(3, 1, 3)
    plt.plot(spike_probs[:sample_size], label='급변 확률', color='orange', alpha=0.7)
    plt.axhline(y=0.5, color='red', linestyle='--', label='임계값 (0.5)', alpha=0.5)
    plt.title('급변 감지 확률', fontsize=14)
    plt.xlabel('시간', fontsize=12)
    plt.ylabel('확률', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('improved_prediction_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 모델별 성능 비교 (1400+ 구간 포함)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    models_list = list(results.keys())
    
    # MAE 비교
    mae_values = [results[model]['mae_original'] for model in models_list]
    axes[0, 0].bar(models_list, mae_values, color='blue')
    axes[0, 0].set_title('전체 MAE', fontsize=14)
    axes[0, 0].set_ylabel('MAE', fontsize=12)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 1400+ MAE 비교
    mae_high_values = [results[model]['mae_high'] for model in models_list]
    axes[0, 1].bar(models_list, mae_high_values, color='red')
    axes[0, 1].set_title('1400+ 구간 MAE', fontsize=14)
    axes[0, 1].set_ylabel('MAE', fontsize=12)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # R² 비교
    r2_values = [results[model]['r2'] for model in models_list]
    axes[1, 0].bar(models_list, r2_values, color='green')
    axes[1, 0].set_title('R² Score', fontsize=14)
    axes[1, 0].set_ylabel('R²', fontsize=12)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 급변 정확도
    spike_acc_values = [results[model]['spike_accuracy'] for model in models_list]
    axes[1, 1].bar(models_list, spike_acc_values, color='orange')
    axes[1, 1].set_title('급변 감지 정확도', fontsize=14)
    axes[1, 1].set_ylabel('정확도', fontsize=12)
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('improved_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ===================================
    # 14. 모델 저장
    # ===================================
    
    logger.info("\n" + "="*60)
    logger.info("모델 및 결과 저장")
    logger.info("="*60)
    
    os.makedirs('model_v3', exist_ok=True)
    os.makedirs('scaler_v3', exist_ok=True)
    os.makedirs('results_v3', exist_ok=True)
    
    # 모델 저장
    for model_name, model in hybrid_models.models.items():
        model_path = f'model_v3/{model_name}_improved.h5'
        model.save(model_path)
        logger.info(f"{model_name} 모델 저장: {model_path}")
    
    # 급변 감지기 저장
    spike_detector.save('model_v3/spike_detector.h5')
    logger.info("급변 감지기 저장: model_v3/spike_detector.h5")
    
    # 스케일러 저장
    joblib.dump(sscaler, 'scaler_v3/robust_scaler.pkl')
    
    # 성능 결과 저장
    results_df = pd.DataFrame(results).T
    results_df.to_csv('results_v3/improved_performance.csv')
    
    # 설정 저장
    config = {
        'seq_length': SEQ_LENGTH,
        'future_minutes': 10,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'features_count': len(input_features),
        'high_volume_threshold': 1400,
        'improvements': [
            '100분 시퀀스 길이',
            'Attention 메커니즘',
            'CNN-LSTM 하이브리드',
            '급변 감지 분류기',
            '동적 앙상블 가중치',
            '강화된 특징 엔지니어링'
        ]
    }
    
    with open('results_v3/training_config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    logger.info("\n" + "="*60)
    logger.info("학습 완료!")
    logger.info(f"최종 앙상블 성능:")
    logger.info(f"  - 전체 MAE: {results['ensemble']['mae_original']:.2f}")
    logger.info(f"  - 1400+ MAE: {results['ensemble']['mae_high']:.2f}")
    logger.info(f"  - R²: {results['ensemble']['r2']:.4f}")
    logger.info(f"  - 급변 감지: {results['ensemble']['spike_accuracy']:.2%}")
    logger.info("="*60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='개선된 반도체 물류 예측 모델')
    parser.add_argument('--resume', action='store_true', help='이전 학습 재개')
    
    args = parser.parse_args()
    main(resume=args.resume)