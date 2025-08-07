"""
CNN-LSTM Multi-Task 기반 반도체 물류 예측 모델 - 재시작 가능 버전
==================================================================
중간 저장과 재시작이 가능한 버전입니다.
중단되어도 이어서 학습할 수 있습니다.
스케일러도 중간 저장됩니다!

사용 데이터: data/20240201_TO_202507281705.csv
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (Input, Conv1D, LSTM, Dense, Dropout,
                                    BatchNormalization, Bidirectional,
                                    MaxPooling1D, Activation)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import pickle
from datetime import datetime, timedelta
import joblib
import logging
import warnings
import traceback

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
        logging.FileHandler('training_multitask.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===================================
# 2. 체크포인트 관리 클래스
# ===================================

class CheckpointManager:
    """학습 상태를 저장하고 복원하는 클래스"""
    
    def __init__(self, checkpoint_dir='checkpoints_multitask'):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 저장 경로들
        self.state_file = os.path.join(checkpoint_dir, 'training_state.json')
        self.data_file = os.path.join(checkpoint_dir, 'preprocessed_data.pkl')
        self.scaler_file = os.path.join(checkpoint_dir, 'scaler_checkpoint.pkl')
        self.history_file = os.path.join(checkpoint_dir, 'training_history.pkl')
        self.class_mapping_file = os.path.join(checkpoint_dir, 'class_mapping.json')
        
    def save_state(self, state_dict):
        """현재 학습 상태 저장"""
        # JSON 직렬화를 위해 numpy 타입 변환
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # 재귀적으로 변환
        state_dict_converted = json.loads(
            json.dumps(state_dict, default=convert_numpy)
        )
        
        with open(self.state_file, 'w') as f:
            json.dump(state_dict_converted, f, indent=4)
        logger.info(f"✓ 학습 상태 저장됨: {self.state_file}")
        
    def load_state(self):
        """저장된 학습 상태 로드"""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            logger.info(f"✓ 학습 상태 로드됨: {self.state_file}")
            return state
        return None
        
    def save_data(self, data_dict):
        """전처리된 데이터 저장"""
        with open(self.data_file, 'wb') as f:
            pickle.dump(data_dict, f)
        logger.info(f"✓ 데이터 저장됨: {self.data_file}")
        
    def load_data(self):
        """저장된 데이터 로드"""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"✓ 데이터 로드됨: {self.data_file}")
            return data
        return None
        
    def save_scaler(self, scaler):
        """스케일러 저장 - 중요!"""
        joblib.dump(scaler, self.scaler_file)
        logger.info(f"✓ 스케일러 저장됨: {self.scaler_file}")
        
    def load_scaler(self):
        """저장된 스케일러 로드"""
        if os.path.exists(self.scaler_file):
            scaler = joblib.load(self.scaler_file)
            logger.info(f"✓ 스케일러 로드됨: {self.scaler_file}")
            return scaler
        return None
        
    def save_class_mapping(self, class_mapping):
        """클래스 매핑 정보 저장"""
        with open(self.class_mapping_file, 'w') as f:
            json.dump(class_mapping, f, indent=4)
        logger.info(f"✓ 클래스 매핑 저장됨: {self.class_mapping_file}")
        
    def load_class_mapping(self):
        """클래스 매핑 정보 로드"""
        if os.path.exists(self.class_mapping_file):
            with open(self.class_mapping_file, 'r') as f:
                mapping = json.load(f)
            logger.info(f"✓ 클래스 매핑 로드됨: {self.class_mapping_file}")
            return mapping
        return None
        
    def save_history(self, history):
        """학습 히스토리 저장"""
        with open(self.history_file, 'wb') as f:
            pickle.dump(history, f)
        logger.info(f"✓ 학습 히스토리 저장됨: {self.history_file}")
        
    def load_history(self):
        """학습 히스토리 로드"""
        if os.path.exists(self.history_file):
            with open(self.history_file, 'rb') as f:
                history = pickle.load(f)
            logger.info(f"✓ 학습 히스토리 로드됨: {self.history_file}")
            return history
        return None
        
    def save_model_checkpoint(self, model, epoch):
        """모델 체크포인트 저장"""
        model_path = os.path.join(self.checkpoint_dir, f'model_epoch_{epoch}.keras')
        model.save(model_path)
        return model_path

# ===================================
# 3. 주기적 체크포인트 콜백
# ===================================

class PeriodicCheckpoint(Callback):
    """주기적으로 모델과 상태를 저장하는 콜백"""
    
    def __init__(self, checkpoint_manager, save_freq=5, scaler=None):
        super().__init__()
        self.checkpoint_manager = checkpoint_manager
        self.save_freq = save_freq
        self.scaler = scaler
        self.history = {'loss': [], 'val_loss': [], 
                       'logistics_output_loss': [], 'val_logistics_output_loss': [],
                       'bottleneck_output_loss': [], 'val_bottleneck_output_loss': [],
                       'logistics_output_mae': [], 'val_logistics_output_mae': [],
                       'bottleneck_output_accuracy': [], 'val_bottleneck_output_accuracy': []}
        
    def on_epoch_end(self, epoch, logs=None):
        # 히스토리 업데이트
        for key in self.history.keys():
            if logs and key in logs:
                self.history[key].append(logs[key])
        
        # 주기적 저장
        if (epoch + 1) % self.save_freq == 0:
            logger.info(f"\n{'='*60}")
            logger.info(f"에폭 {epoch + 1}: 체크포인트 저장 중...")
            
            # 1. 모델 저장
            model_path = self.checkpoint_manager.save_model_checkpoint(self.model, epoch + 1)
            
            # 2. 스케일러 저장 (중요!)
            if self.scaler is not None:
                self.checkpoint_manager.save_scaler(self.scaler)
            
            # 3. 상태 저장
            state = {
                'current_epoch': epoch + 1,
                'model_path': model_path,
                'best_val_loss': min(self.history['val_loss']) if self.history['val_loss'] else float('inf'),
                'training_completed': False,
                'last_save_time': datetime.now().isoformat()
            }
            self.checkpoint_manager.save_state(state)
            
            # 4. 히스토리 저장
            self.checkpoint_manager.save_history(self.history)
            
            logger.info(f"✓ 체크포인트 저장 완료 (에폭 {epoch + 1})")
            logger.info(f"{'='*60}\n")

# ===================================
# 4. 데이터 전처리 함수들 (기존과 동일)
# ===================================

def load_and_preprocess_data(data_path):
    """데이터 로드 및 전처리"""
    logger.info("데이터 로딩 중...")
    
    # 데이터 로드
    Full_Data = pd.read_csv(data_path)
    logger.info(f"원본 데이터 shape: {Full_Data.shape}")
    
    # 시간 컬럼 변환
    Full_Data['CURRTIME'] = pd.to_datetime(Full_Data['CURRTIME'], format='%Y%m%d%H%M')
    Full_Data['TIME'] = pd.to_datetime(Full_Data['TIME'], format='%Y%m%d%H%M')
    
    # SUM 컬럼 제거
    columns_to_drop = [col for col in Full_Data.columns if 'SUM' in col]
    Full_Data = Full_Data.drop(columns=columns_to_drop)
    
    # 특정 날짜 범위만 사용
    start_date = pd.to_datetime('2024-02-01 00:00:00')
    end_date = pd.to_datetime('2025-07-27 23:59:59')
    Full_Data = Full_Data[(Full_Data['TIME'] >= start_date) & (Full_Data['TIME'] <= end_date)]
    
    # 인덱스를 시간으로 설정
    Full_Data.set_index('CURRTIME', inplace=True)
    
    # 이상치 처리
    PM_start_date = pd.to_datetime('2024-10-23 00:00:00')
    PM_end_date = pd.to_datetime('2024-10-23 23:59:59')
    
    within_PM = Full_Data[(Full_Data['TIME'] >= PM_start_date) & (Full_Data['TIME'] <= PM_end_date)]
    outside_PM = Full_Data[(Full_Data['TIME'] < PM_start_date) | (Full_Data['TIME'] > PM_end_date)]
    outside_PM_filtered = outside_PM[(outside_PM['TOTALCNT'] >= 800) & (outside_PM['TOTALCNT'] <= 2500)]
    
    Full_Data = pd.concat([within_PM, outside_PM_filtered])
    Full_Data = Full_Data.sort_index()
    
    logger.info(f"전처리 후 데이터 shape: {Full_Data.shape}")
    return Full_Data

def create_features(data):
    """특징 엔지니어링"""
    logger.info("특징 생성 중...")
    
    features_data = data.copy()
    
    # 시간 특징
    features_data['hour'] = features_data.index.hour
    features_data['dayofweek'] = features_data.index.dayofweek
    features_data['is_weekend'] = (features_data.index.dayofweek >= 5).astype(int)
    features_data['month'] = features_data.index.month
    features_data['day'] = features_data.index.day
    features_data['is_peak_hour'] = features_data.index.hour.isin([8, 9, 14, 15, 16, 17]).astype(int)
    
    # 팹 간 불균형 지표
    features_data['imbalance_M14A_M10A'] = features_data['M14AM10A'] - features_data['M10AM14A']
    features_data['imbalance_M14A_M14B'] = features_data['M14AM14B'] - features_data['M14BM14A']
    features_data['imbalance_M14A_M16'] = features_data['M14AM16'] - features_data['M16M14A']
    
    # 이동 평균
    for window in [5, 10, 30, 60]:
        features_data[f'MA_{window}'] = features_data['TOTALCNT'].rolling(window=window, min_periods=1).mean()
    
    # 표준편차
    for window in [5, 10, 30]:
        features_data[f'STD_{window}'] = features_data['TOTALCNT'].rolling(window=window, min_periods=1).std()
    
    # 최대/최소값
    features_data['MAX_10'] = features_data['TOTALCNT'].rolling(window=10, min_periods=1).max()
    features_data['MIN_10'] = features_data['TOTALCNT'].rolling(window=10, min_periods=1).min()
    
    # 팹별 부하율
    total_safe = features_data['TOTALCNT'].replace(0, 1)
    features_data['load_M14A_out'] = (features_data['M14AM10A'] + features_data['M14AM14B'] +
                                      features_data['M14AM16']) / total_safe
    features_data['load_M14A_in'] = (features_data['M10AM14A'] + features_data['M14BM14A'] +
                                     features_data['M16M14A']) / total_safe
    
    # 경로별 비율
    features_data['ratio_M14A_M10A'] = (features_data['M14AM10A'] + features_data['M10AM14A']) / total_safe
    features_data['ratio_M14A_M14B'] = (features_data['M14AM14B'] + features_data['M14BM14A']) / total_safe
    features_data['ratio_M14A_M16'] = (features_data['M14AM16'] + features_data['M16M14A']) / total_safe
    
    # 변화율
    features_data['change_rate'] = features_data['TOTALCNT'].pct_change()
    features_data['change_rate_5'] = features_data['TOTALCNT'].pct_change(5)
    features_data['change_rate_10'] = features_data['TOTALCNT'].pct_change(10)
    features_data['acceleration'] = features_data['change_rate'].diff()
    
    # 결측값 처리
    features_data = features_data.fillna(method='ffill').fillna(0)
    features_data = features_data.replace([np.inf, -np.inf], 0)
    
    # 이상치 클리핑
    numeric_columns = features_data.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if col not in ['TIME', 'CURRTIME']:
            upper_limit = features_data[col].quantile(0.999)
            lower_limit = features_data[col].quantile(0.001)
            features_data[col] = features_data[col].clip(lower=lower_limit, upper=upper_limit)
    
    logger.info(f"특징 생성 완료 - shape: {features_data.shape}")
    return features_data

def create_targets(data, future_minutes=10):
    """타겟 변수 생성"""
    logger.info("타겟 변수 생성 중...")
    
    # 물류량 타겟
    data['FUTURE_TOTALCNT'] = pd.NA
    for i in data.index:
        future_time = i + pd.Timedelta(minutes=future_minutes)
        if (future_time <= data.index.max()) & (future_time in data.index):
            data.loc[i, 'FUTURE_TOTALCNT'] = data.loc[future_time, 'TOTALCNT']
    
    # 병목 위치 타겟
    thresholds = {
        'total': data['TOTALCNT'].quantile(0.90),
        'm14a_m10a': np.percentile(data['M14AM10A'] + data['M10AM14A'], 90),
        'm14a_m14b': np.percentile(data['M14AM14B'] + data['M14BM14A'], 90),
        'm14a_m16': np.percentile(data['M14AM16'] + data['M16M14A'], 90)
    }
    
    data['BOTTLENECK_LOCATION'] = 0
    for i in data.index:
        future_time = i + pd.Timedelta(minutes=future_minutes)
        if (future_time <= data.index.max()) & (future_time in data.index):
            future_total = data.loc[future_time, 'TOTALCNT']
            if future_total > thresholds['total']:
                route_loads = {
                    1: data.loc[future_time, 'M14AM10A'] + data.loc[future_time, 'M10AM14A'],
                    2: data.loc[future_time, 'M14AM14B'] + data.loc[future_time, 'M14BM14A'],
                    3: data.loc[future_time, 'M14AM16'] + data.loc[future_time, 'M16M14A']
                }
                max_route = max(route_loads.items(), key=lambda x: x[1])
                if max_route[0] == 1 and max_route[1] > thresholds['m14a_m10a']:
                    data.loc[i, 'BOTTLENECK_LOCATION'] = 1
                elif max_route[0] == 2 and max_route[1] > thresholds['m14a_m14b']:
                    data.loc[i, 'BOTTLENECK_LOCATION'] = 2
                elif max_route[0] == 3 and max_route[1] > thresholds['m14a_m16']:
                    data.loc[i, 'BOTTLENECK_LOCATION'] = 3
    
    data = data.dropna(subset=['FUTURE_TOTALCNT'])
    logger.info(f"타겟 생성 완료 - 병목 분포: {data['BOTTLENECK_LOCATION'].value_counts()}")
    return data

def scale_features(data, feature_columns, scaler=None):
    """특징 스케일링"""
    if scaler is None:
        scaler = StandardScaler()
        fit_scaler = True
    else:
        fit_scaler = False
    
    scale_columns = [col for col in feature_columns if col in data.columns]
    scale_data = data[scale_columns].copy()
    
    # 무한대 값 처리
    if np.isinf(scale_data.values).any():
        scale_data = scale_data.replace([np.inf, -np.inf], np.nan)
        scale_data = scale_data.fillna(scale_data.mean())
    
    # NaN 값 처리
    if scale_data.isnull().any().any():
        scale_data = scale_data.fillna(scale_data.mean())
    
    # 스케일링
    if fit_scaler:
        scaled_data = scaler.fit_transform(scale_data)
    else:
        scaled_data = scaler.transform(scale_data)
    
    # 스케일링된 데이터프레임 생성
    scaled_df = pd.DataFrame(
        scaled_data,
        columns=[f'scaled_{col}' for col in scale_columns],
        index=data.index
    )
    
    result = pd.concat([data, scaled_df], axis=1)
    return result, scaler

def create_sequences(data, feature_cols, target_cols, seq_length=60):
    """시퀀스 데이터 생성"""
    X, y_regression, y_classification = [], [], []
    
    time_diff = data.index.to_series().diff()
    split_points = time_diff > pd.Timedelta(minutes=1)
    segment_ids = split_points.cumsum()
    
    for segment_id in segment_ids.unique():
        segment = data[segment_ids == segment_id]
        if len(segment) > seq_length:
            feature_data = segment[feature_cols].values
            regression_data = segment[target_cols[0]].values
            classification_data = segment[target_cols[1]].values
            
            for i in range(len(segment) - seq_length):
                X.append(feature_data[i:i+seq_length])
                y_regression.append(regression_data[i+seq_length])
                y_classification.append(classification_data[i+seq_length])
    
    return np.array(X), np.array(y_regression), np.array(y_classification)

# ===================================
# 5. CNN-LSTM Multi-Task 모델 (기존과 동일)
# ===================================

def build_cnn_lstm_multitask_model(input_shape, num_classes=4):
    """CNN-LSTM Multi-Task 모델 구축"""
    
    inputs = Input(shape=input_shape, name='input')
    
    # CNN 파트
    x = Conv1D(filters=128, kernel_size=5, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    
    x = Conv1D(filters=256, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    
    x = Conv1D(filters=256, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    
    x = Conv1D(filters=128, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = MaxPooling1D(pool_size=2)(x)
    
    # LSTM 파트
    x = Bidirectional(LSTM(256, return_sequences=True))(x)
    x = Dropout(0.4)(x)
    
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.4)(x)
    
    x = Bidirectional(LSTM(64, return_sequences=False))(x)
    x = Dropout(0.4)(x)
    
    # 공유 Dense 레이어
    shared = Dense(256, activation='relu', name='shared_layer')(x)
    shared = BatchNormalization()(shared)
    shared = Dropout(0.4)(shared)
    
    shared = Dense(128, activation='relu')(shared)
    shared = BatchNormalization()(shared)
    shared = Dropout(0.3)(shared)
    
    # Multi-Task 출력
    logistics_branch = Dense(128, activation='relu')(shared)
    logistics_branch = Dropout(0.3)(logistics_branch)
    logistics_branch = Dense(64, activation='relu')(logistics_branch)
    logistics_output = Dense(1, name='logistics_output')(logistics_branch)
    
    bottleneck_branch = Dense(128, activation='relu')(shared)
    bottleneck_branch = Dropout(0.3)(bottleneck_branch)
    bottleneck_branch = Dense(64, activation='relu')(bottleneck_branch)
    bottleneck_output = Dense(num_classes, activation='softmax', name='bottleneck_output')(bottleneck_branch)
    
    model = Model(inputs=inputs, outputs=[logistics_output, bottleneck_output])
    
    return model

# ===================================
# 6. 학습 프로세스 (수정됨)
# ===================================

def train_model_with_checkpoint(model, X_train, y_train_reg, y_train_cls, 
                                X_val, y_val_reg, y_val_cls,
                                checkpoint_manager, scaler, start_epoch=0,
                                epochs=200, batch_size=64):
    """체크포인트를 지원하는 모델 학습"""
    
    # 손실 함수와 가중치 설정
    losses = {
        'logistics_output': 'mse',
        'bottleneck_output': 'sparse_categorical_crossentropy'
    }
    
    loss_weights = {
        'logistics_output': 0.7,
        'bottleneck_output': 0.3
    }
    
    metrics = {
        'logistics_output': ['mae'],
        'bottleneck_output': ['accuracy']
    }
    
    # 컴파일
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics
    )
    
    # 기존 히스토리 로드
    previous_history = checkpoint_manager.load_history()
    
    # 콜백 설정
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=30,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=15,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            'model/cnn_lstm_multitask_best.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        # 주기적 체크포인트 콜백 추가!
        PeriodicCheckpoint(
            checkpoint_manager,
            save_freq=5,  # 5에폭마다 저장
            scaler=scaler  # 스케일러도 함께 저장!
        )
    ]
    
    # 이전 히스토리가 있으면 콜백에 전달
    if previous_history:
        callbacks[-1].history = previous_history
    
    try:
        # 학습
        history = model.fit(
            X_train,
            {'logistics_output': y_train_reg, 'bottleneck_output': y_train_cls},
            validation_data=(
                X_val,
                {'logistics_output': y_val_reg, 'bottleneck_output': y_val_cls}
            ),
            initial_epoch=start_epoch,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # 학습 완료 상태 저장
        state = {
            'current_epoch': epochs,
            'training_completed': True,
            'completion_time': datetime.now().isoformat()
        }
        checkpoint_manager.save_state(state)
        
        return history
        
    except KeyboardInterrupt:
        logger.warning("\n학습이 사용자에 의해 중단되었습니다.")
        # 현재 상태 저장
        current_epoch = len(callbacks[-1].history['loss'])
        state = {
            'current_epoch': start_epoch + current_epoch,
            'training_completed': False,
            'interrupted': True,
            'interrupt_time': datetime.now().isoformat()
        }
        checkpoint_manager.save_state(state)
        checkpoint_manager.save_history(callbacks[-1].history)
        checkpoint_manager.save_scaler(scaler)  # 스케일러도 저장!
        raise
        
    except Exception as e:
        logger.error(f"\n학습 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        # 오류 시점 상태 저장
        current_epoch = len(callbacks[-1].history['loss']) if callbacks else 0
        state = {
            'current_epoch': start_epoch + current_epoch,
            'training_completed': False,
            'error': str(e),
            'error_time': datetime.now().isoformat()
        }
        checkpoint_manager.save_state(state)
        checkpoint_manager.save_history(callbacks[-1].history)
        checkpoint_manager.save_scaler(scaler)  # 스케일러도 저장!
        raise

# ===================================
# 7. 메인 실행 함수 (수정됨)
# ===================================

def main(resume=False):
    """메인 실행 함수"""
    
    checkpoint_manager = CheckpointManager()
    
    # 재시작 모드 확인
    if resume:
        state = checkpoint_manager.load_state()
        if state:
            logger.info("="*60)
            logger.info("이전 학습 상태에서 재시작합니다.")
            logger.info(f"마지막 에폭: {state.get('current_epoch', 0)}")
            logger.info(f"학습 완료 여부: {state.get('training_completed', False)}")
            logger.info("="*60)
            
            # 저장된 데이터 로드
            saved_data = checkpoint_manager.load_data()
            saved_scaler = checkpoint_manager.load_scaler()
            saved_class_mapping = checkpoint_manager.load_class_mapping()
            
            if saved_data and saved_scaler:
                logger.info("저장된 데이터와 스케일러를 성공적으로 로드했습니다.")
                
                # 데이터 복원
                X_train = saved_data['X_train']
                X_val = saved_data['X_val']
                X_test = saved_data['X_test']
                y_train_reg = saved_data['y_train_reg']
                y_val_reg = saved_data['y_val_reg']
                y_test_reg = saved_data['y_test_reg']
                y_train_cls = saved_data['y_train_cls']
                y_val_cls = saved_data['y_val_cls']
                y_test_cls = saved_data['y_test_cls']
                input_shape = saved_data['input_shape']
                num_classes = saved_data['num_classes']
                
                # 스케일러 복원
                scaler = saved_scaler
                
                # 클래스 매핑 복원
                class_mapping = saved_class_mapping if saved_class_mapping else {}
                
                # 모델 재생성 또는 로드
                start_epoch = state.get('current_epoch', 0)
                
                if 'model_path' in state and os.path.exists(state['model_path']):
                    logger.info(f"저장된 모델 로드: {state['model_path']}")
                    model = load_model(state['model_path'])
                else:
                    logger.info("모델을 새로 생성합니다.")
                    model = build_cnn_lstm_multitask_model(input_shape, num_classes)
            else:
                logger.warning("저장된 데이터나 스케일러가 없습니다. 처음부터 시작합니다.")
                resume = False
        else:
            logger.info("저장된 학습 상태가 없습니다. 처음부터 시작합니다.")
            resume = False
    
    # 처음부터 시작하는 경우
    if not resume:
        logger.info("="*60)
        logger.info("CNN-LSTM Multi-Task 모델 학습 시작 (실제 데이터)")
        logger.info("="*60)
        
        # 데이터 로드 및 전처리
        data_path = 'data/20240201_TO_202507281705.csv'
        
        if not os.path.exists(data_path):
            logger.error(f"데이터 파일을 찾을 수 없습니다: {data_path}")
            return None, None, None
        
        data = load_and_preprocess_data(data_path)
        data = create_features(data)
        data = create_targets(data)
        
        # 특징 선택
        scale_features_list = [
            'TOTALCNT', 'M14AM10A', 'M10AM14A', 'M14AM14B', 'M14BM14A', 'M14AM16', 'M16M14A',
            'imbalance_M14A_M10A', 'imbalance_M14A_M14B', 'imbalance_M14A_M16',
            'MA_5', 'MA_10', 'MA_30', 'MA_60',
            'STD_5', 'STD_10', 'STD_30',
            'MAX_10', 'MIN_10',
            'load_M14A_out', 'load_M14A_in',
            'ratio_M14A_M10A', 'ratio_M14A_M14B', 'ratio_M14A_M16',
            'change_rate', 'change_rate_5', 'change_rate_10',
            'acceleration'
        ]
        
        scale_features_list = [col for col in scale_features_list if col in data.columns]
        
        # 스케일링
        data, scaler = scale_features(data, scale_features_list)
        
        # 스케일러 즉시 저장!
        checkpoint_manager.save_scaler(scaler)
        
        # 시퀀스 생성
        sequence_features = [col for col in data.columns if col.startswith('scaled_')]
        target_features = ['FUTURE_TOTALCNT', 'BOTTLENECK_LOCATION']
        
        X, y_regression, y_classification = create_sequences(
            data, sequence_features, target_features, seq_length=60
        )
        
        # 클래스 레이블 재매핑
        unique_classes = np.unique(y_classification)
        class_mapping = {old_class: new_class for new_class, old_class in enumerate(unique_classes)}
        y_classification_mapped = np.array([class_mapping[cls] for cls in y_classification])
        
        # 클래스 매핑 저장
        checkpoint_manager.save_class_mapping(class_mapping)
        
        # 데이터 분할
        train_size = int(0.7 * len(X))
        val_size = int(0.15 * len(X))
        
        X_train = X[:train_size]
        X_val = X[train_size:train_size+val_size]
        X_test = X[train_size+val_size:]
        
        y_train_reg = y_regression[:train_size]
        y_val_reg = y_regression[train_size:train_size+val_size]
        y_test_reg = y_regression[train_size+val_size:]
        
        y_train_cls = y_classification_mapped[:train_size]
        y_val_cls = y_classification_mapped[train_size:train_size+val_size]
        y_test_cls = y_classification_mapped[train_size+val_size:]
        
        # 데이터 저장
        checkpoint_manager.save_data({
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train_reg': y_train_reg, 'y_val_reg': y_val_reg, 'y_test_reg': y_test_reg,
            'y_train_cls': y_train_cls, 'y_val_cls': y_val_cls, 'y_test_cls': y_test_cls,
            'input_shape': (X_train.shape[1], X_train.shape[2]),
            'num_classes': len(np.unique(y_classification_mapped))
        })
        
        # 모델 생성
        input_shape = (X_train.shape[1], X_train.shape[2])
        num_classes = len(np.unique(y_classification_mapped))
        model = build_cnn_lstm_multitask_model(input_shape, num_classes)
        model.summary()
        
        start_epoch = 0
    
    # 모델 학습
    logger.info(f"\n모델 학습 시작... (에폭 {start_epoch}부터)")
    
    try:
        history = train_model_with_checkpoint(
            model,
            X_train, y_train_reg, y_train_cls,
            X_val, y_val_reg, y_val_cls,
            checkpoint_manager, scaler,
            start_epoch=start_epoch,
            epochs=200,
            batch_size=64
        )
    except KeyboardInterrupt:
        logger.warning("\n학습이 중단되었습니다. 현재 상태가 저장되었습니다.")
        logger.info("다시 시작하려면: python script.py --resume")
        return None, None, None
    except Exception as e:
        logger.error(f"\n학습 중 오류 발생: {str(e)}")
        logger.info("다시 시작하려면: python script.py --resume")
        return None, None, None
    
    # 모델 평가
    logger.info("\n모델 평가 중...")
    predictions = model.predict(X_test)
    pred_logistics = predictions[0].flatten()
    pred_bottleneck = predictions[1]
    
    mae = mean_absolute_error(y_test_reg, pred_logistics)
    mse = mean_squared_error(y_test_reg, pred_logistics)
    
    logger.info(f"\n물류량 예측 성능:")
    logger.info(f"  MAE: {mae:.2f}")
    logger.info(f"  MSE: {mse:.2f}")
    logger.info(f"  RMSE: {np.sqrt(mse):.2f}")
    
    pred_bottleneck_classes = np.argmax(pred_bottleneck, axis=1)
    accuracy = accuracy_score(y_test_cls, pred_bottleneck_classes)
    
    logger.info(f"\n병목 위치 예측 성능:")
    logger.info(f"  Accuracy: {accuracy:.2%}")
    
    # 최종 모델 및 스케일러 저장
    logger.info("\n최종 모델 및 스케일러 저장 중...")
    
    os.makedirs('model', exist_ok=True)
    os.makedirs('scaler', exist_ok=True)
    os.makedirs('config', exist_ok=True)
    
    model.save('model/cnn_lstm_multitask_final.keras')
    joblib.dump(scaler, 'scaler/multitask_scaler.pkl')
    
    with open('config/class_mapping.json', 'w') as f:
        json.dump(class_mapping, f, indent=4)
    
    logger.info("\n" + "="*60)
    logger.info("학습 완료!")
    logger.info("="*60)
    
    return model, scaler, history

# ===================================
# 8. 실행
# ===================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='CNN-LSTM Multi-Task 모델 학습')
    parser.add_argument('--resume', action='store_true', 
                       help='이전 학습을 이어서 진행')
    parser.add_argument('--reset', action='store_true',
                       help='체크포인트를 삭제하고 처음부터 시작')
    
    args = parser.parse_args()
    
    if args.reset:
        import shutil
        if os.path.exists('checkpoints_multitask'):
            shutil.rmtree('checkpoints_multitask')
            logger.info("체크포인트가 삭제되었습니다. 처음부터 시작합니다.")
    
    # 실행
    model, scaler, history = main(resume=args.resume)
    
    if model is not None:
        print("\n" + "="*60)
        print("🎉 학습 완료!")
        print("="*60)
        print("\n생성된 파일:")
        print("  - model/cnn_lstm_multitask_final.keras")
        print("  - scaler/multitask_scaler.pkl")
        print("  - checkpoints_multitask/ (중간 저장 파일들)")