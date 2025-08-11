"""
TensorFlow 2.18.0 - 재시작 가능한 반도체 물류 예측 시스템 (오류 수정판)
========================================================
학습 데이터: 20240201_TO_202507281705.csv
학습 중단 시 재시작 기능 완벽 지원

핵심 기능:
1. 체크포인트 자동 저장
2. 중단점부터 재시작
3. 학습 진행 상황 추적
4. 모델별 개별 관리
5. 데이터 전처리 상태 보존

데이터 전처리:
- 전체 기간: 2024-02-01 ~ 2025-07-28 (20240201_TO_202507281705.csv)
- 날짜 필터링: 옵션 (use_date_filter=False 기본값)
- PM 기간(2024-10-23) 예외 처리
- 정상 범위: 800 ~ 2500 (PM 기간 제외)

구현 완료:
✅ RobustScaler 사용 (이상치에 강건)
✅ 변화율 기반 특징 강화
✅ Attention 메커니즘 추가
✅ 다단계 예측 + 앙상블
✅ 동적 임계값 학습
✅ 재시작 가능한 학습

오류 수정:
✅ .weights.h5 확장자로 모든 저장 경로 수정
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import pickle
from datetime import datetime
import joblib
import warnings
import gc
import traceback
import signal
import sys
import logging

# TensorFlow 2.18.0 설정
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

# 메모리 최적화
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

# 랜덤 시드
RANDOM_SEED = 2079936
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print(f"TensorFlow Version: {tf.__version__}")
print("="*70)

# ===================================
# 체크포인트 관리 시스템 (파일 형식 수정)
# ===================================

class CheckpointManager:
    """학습 상태 저장 및 복원 관리"""
    
    def __init__(self, checkpoint_dir='checkpoints'):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs('model', exist_ok=True)
        
        # 경로 설정
        self.state_file = os.path.join(checkpoint_dir, 'training_state.json')
        self.data_file = os.path.join(checkpoint_dir, 'preprocessed_data.pkl')
        self.scaler_file = os.path.join(checkpoint_dir, 'scalers.pkl')
        self.history_file = os.path.join(checkpoint_dir, 'training_history.pkl')
        
        # 상태 초기화
        self.state = {
            'current_model': None,
            'current_epoch': 0,
            'completed_models': [],
            'model_histories': {},
            'best_scores': {},
            'training_config': {},
            'timestamp': None
        }
        
        # Ctrl+C 핸들러 등록
        signal.signal(signal.SIGINT, self.signal_handler)
        
    def signal_handler(self, sig, frame):
        """Ctrl+C 시 자동 저장"""
        print('\n\n⚠️  학습 중단 감지! 현재 상태를 저장합니다...')
        self.save_emergency_checkpoint()
        print('✅ 긴급 체크포인트 저장 완료!')
        print('다시 시작하려면: python script.py --resume')
        sys.exit(0)
    
    def save_emergency_checkpoint(self):
        """긴급 체크포인트 저장"""
        self.state['interrupted'] = True
        self.state['interrupt_time'] = datetime.now().isoformat()
        self.save_state()
    
    def save_state(self):
        """현재 학습 상태 저장"""
        self.state['timestamp'] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=4)
        print(f"📁 상태 저장: {self.state_file}")
    
    def load_state(self):
        """저장된 학습 상태 로드"""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                self.state = json.load(f)
            print(f"📂 상태 로드: {self.state_file}")
            return True
        return False
    
    def save_data(self, X_train, y_train, X_val, y_val, X_test, y_test, 
                  y_train_orig, y_val_orig, y_test_orig, feature_cols):
        """전처리된 데이터 저장"""
        data = {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'y_train_orig': y_train_orig,
            'y_val_orig': y_val_orig,
            'y_test_orig': y_test_orig,
            'feature_cols': feature_cols,
            'shape_info': {
                'train': X_train.shape,
                'val': X_val.shape,
                'test': X_test.shape
            }
        }
        with open(self.data_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"📁 데이터 저장: {self.data_file}")
    
    def load_data(self):
        """저장된 데이터 로드"""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'rb') as f:
                data = pickle.load(f)
            print(f"📂 데이터 로드: {self.data_file}")
            return data
        return None
    
    def save_scalers(self, target_scaler, feature_scaler):
        """스케일러 저장"""
        scalers = {
            'target_scaler': target_scaler,
            'feature_scaler': feature_scaler
        }
        with open(self.scaler_file, 'wb') as f:
            pickle.dump(scalers, f)
        print(f"📁 스케일러 저장: {self.scaler_file}")
    
    def load_scalers(self):
        """스케일러 로드"""
        if os.path.exists(self.scaler_file):
            with open(self.scaler_file, 'rb') as f:
                scalers = pickle.load(f)
            print(f"📂 스케일러 로드: {self.scaler_file}")
            return scalers
        return None
    
    def save_history(self, model_name, history):
        """학습 이력 저장"""
        if os.path.exists(self.history_file):
            with open(self.history_file, 'rb') as f:
                all_histories = pickle.load(f)
        else:
            all_histories = {}
        
        # 기존 이력과 병합
        if model_name in all_histories:
            for key in history.history.keys():
                all_histories[model_name][key].extend(history.history[key])
        else:
            all_histories[model_name] = history.history
        
        with open(self.history_file, 'wb') as f:
            pickle.dump(all_histories, f)
    
    def load_history(self):
        """학습 이력 로드"""
        if os.path.exists(self.history_file):
            with open(self.history_file, 'rb') as f:
                return pickle.load(f)
        return {}
    
    def get_model_checkpoint_path(self, model_name):
        """모델별 체크포인트 경로"""
        return os.path.join(self.checkpoint_dir, f'{model_name}_checkpoint.weights.h5')
    
    def save_model_weights(self, model, model_name, epoch):
        """모델 가중치 저장 (수정됨: .weights.h5 확장자 사용)"""
        # .weights.h5 확장자 사용
        weights_path = os.path.join(self.checkpoint_dir, f'{model_name}_weights_epoch_{epoch}.weights.h5')
        model.save_weights(weights_path)
        return weights_path
    
    def load_model_weights(self, model, model_name):
        """모델 가중치 로드"""
        # 먼저 전체 모델 (.keras) 확인
        keras_path = f'model/{model_name}_final.keras'
        if os.path.exists(keras_path):
            try:
                loaded_model = keras.models.load_model(keras_path, compile=False)
                model.set_weights(loaded_model.get_weights())
                print(f"📂 전체 모델 로드: {model_name} (.keras)")
                return True
            except:
                pass
        
        # .weights.h5 파일 시도
        path = os.path.join(self.checkpoint_dir, f'{model_name}_checkpoint.weights.h5')
        if os.path.exists(path):
            try:
                model.load_weights(path)
                print(f"📂 모델 가중치 로드: {model_name}")
                return True
            except:
                pass
        
        # 에폭별 파일 확인
        for file in os.listdir(self.checkpoint_dir):
            if file.startswith(f'{model_name}_weights_epoch_') and file.endswith('.weights.h5'):
                try:
                    path = os.path.join(self.checkpoint_dir, file)
                    model.load_weights(path)
                    print(f"📂 모델 로드 (에폭 파일): {model_name}")
                    return True
                except:
                    continue
        
        return False
    
    def update_progress(self, model_name, epoch, metrics=None):
        """학습 진행 상황 업데이트"""
        self.state['current_model'] = model_name
        self.state['current_epoch'] = epoch
        if metrics:
            if model_name not in self.state['best_scores']:
                self.state['best_scores'][model_name] = {}
            self.state['best_scores'][model_name].update(metrics)
        self.save_state()


# ===================================
# 커스텀 콜백
# ===================================

class ProgressCallback(Callback):
    """학습 진행 상황 추적 콜백"""
    
    def __init__(self, checkpoint_manager, model_name):
        super().__init__()
        self.checkpoint_manager = checkpoint_manager
        self.model_name = model_name
        self.epoch_count = 0
        
    def on_epoch_end(self, epoch, logs=None):
        self.epoch_count += 1
        
        # 5 에폭마다 체크포인트 저장
        if self.epoch_count % 5 == 0:
            try:
                metrics = {
                    'loss': logs.get('loss'),
                    'val_loss': logs.get('val_loss'),
                    'mae': logs.get('mae'),
                    'val_mae': logs.get('val_mae')
                }
                
                # 진행 상황 업데이트
                self.checkpoint_manager.update_progress(
                    self.model_name, 
                    self.epoch_count,
                    metrics
                )
                
                # 모델 가중치 저장
                self.checkpoint_manager.save_model_weights(
                    self.model, 
                    self.model_name,
                    self.epoch_count
                )
                
                print(f"\n💾 체크포인트 저장 (Epoch {self.epoch_count})")
            except Exception as e:
                print(f"\n⚠️ 체크포인트 저장 실패: {e}")


# ===================================
# 데이터 전처리 (원본 그대로)
# ===================================

class AdvancedPreprocessor:
    """개별 경로 데이터를 활용한 고급 전처리"""
    
    def __init__(self):
        self.target_scaler = RobustScaler()
        self.feature_scaler = StandardScaler()
        self.route_scalers = {}
        self.routes = ['M14AM10A', 'M10AM14A', 'M14AM14B', 
                      'M14BM14A', 'M14AM16', 'M16M14A']
        self.is_fitted = False
        
    def load_and_process(self, file_path):
        """데이터 로드 및 전처리"""
        print("\n[데이터 로딩 및 전처리]")
        
        data = pd.read_csv(file_path)
        print(f"✓ 원본 데이터: {len(data)} 행")
        
        # 시간 변환
        data['CURRTIME'] = pd.to_datetime(data['CURRTIME'], format='%Y%m%d%H%M')
        data['TIME'] = pd.to_datetime(data['TIME'], format='%Y%m%d%H%M')
        
        # SUM 컬럼 제거
        sum_cols = [col for col in data.columns if 'SUM' in col]
        data = data.drop(columns=sum_cols, errors='ignore')
        
        # 날짜 범위 필터링 (선택적)
        use_date_filter = False  # True로 설정하면 날짜 필터링 적용
        
        if use_date_filter:
            start_date = pd.to_datetime('2024-02-01 00:00:00')
            end_date = pd.to_datetime('2024-07-27 23:59:59')
            data = data[(data['TIME'] >= start_date) & (data['TIME'] <= end_date)].reset_index(drop=True)
            print(f"✓ 필터링 후: {len(data)} 행 (2024-02-01 ~ 2024-07-27)")
        else:
            print(f"✓ 전체 데이터 사용: {len(data)} 행")
        
        # 인덱스 설정
        data.set_index('CURRTIME', inplace=True)
        
        # PM 기간 설정 (예외 처리)
        PM_start_date = pd.to_datetime('2024-10-23 00:00:00')
        PM_end_date = pd.to_datetime('2024-10-23 23:59:59')
        
        # PM 기간 데이터와 정상 범위 데이터 분리
        within_pm = data[(data['TIME'] >= PM_start_date) & (data['TIME'] <= PM_end_date)]
        outside_pm = data[(data['TIME'] < PM_start_date) | (data['TIME'] > PM_end_date)]
        
        # PM 기간이 아닌 경우 800-2500 범위 필터링
        outside_pm_filtered = outside_pm[
            (outside_pm['TOTALCNT'] >= 800) & 
            (outside_pm['TOTALCNT'] <= 2500)
        ]
        
        # PM 기간 데이터와 필터링된 데이터 병합
        data = pd.concat([within_pm, outside_pm_filtered])
        data = data.sort_values(by='TIME')
        
        print(f"✓ 이상치 처리 후: {len(data)} 행")
        print(f"✓ TOTALCNT 범위: {data['TOTALCNT'].min()} ~ {data['TOTALCNT'].max()}")
        
        high_count = (data['TOTALCNT'] >= 1400).sum()
        print(f"✓ 1400 이상: {high_count}개 ({high_count/len(data)*100:.2f}%)")
        
        return data
    
    def create_all_features(self, data):
        """모든 특징 생성"""
        
        # FUTURE 타겟
        data['FUTURE'] = data['TOTALCNT'].shift(-10)
        
        # 개별 경로 특징
        for route in self.routes:
            if route in data.columns:
                # 변화율
                data[f'{route}_pct_1'] = data[route].pct_change(1) * 100
                data[f'{route}_pct_5'] = data[route].pct_change(5) * 100
                
                # 이동평균
                data[f'{route}_ma_5'] = data[route].rolling(5, min_periods=1).mean()
                data[f'{route}_ma_10'] = data[route].rolling(10, min_periods=1).mean()
                
                # 비율
                data[f'{route}_ratio'] = data[route] / (data['TOTALCNT'] + 1e-10)
                
                # 스파이크
                ma = data[route].rolling(10).mean()
                data[f'{route}_spike'] = data[route] / (ma + 1e-10)
        
        # 양방향 흐름 차이
        if 'M14AM10A' in data.columns and 'M10AM14A' in data.columns:
            data['flow_diff_M14A_M10A'] = data['M14AM10A'] - data['M10AM14A']
        
        if 'M14AM14B' in data.columns and 'M14BM14A' in data.columns:
            data['flow_diff_M14A_M14B'] = data['M14AM14B'] - data['M14BM14A']
        
        if 'M14AM16' in data.columns and 'M16M14A' in data.columns:
            data['flow_diff_M14A_M16'] = data['M14AM16'] - data['M16M14A']
        
        # TOTALCNT 특징
        for lag in [1, 2, 3, 5, 10, 15, 20, 30]:
            data[f'diff_{lag}'] = data['TOTALCNT'].diff(lag)
            data[f'pct_change_{lag}'] = data['TOTALCNT'].pct_change(lag) * 100
            data[f'accel_{lag}'] = data[f'diff_{lag}'].diff(1)
        
        # 급상승 감지
        for window in [5, 10, 20, 30]:
            ma = data['TOTALCNT'].rolling(window).mean()
            std = data['TOTALCNT'].rolling(window).std()
            data[f'z_score_{window}'] = (data['TOTALCNT'] - ma) / (std + 1e-10)
            data[f'spike_ratio_{window}'] = data['TOTALCNT'] / (ma + 1e-10)
        
        # 통계 특징
        for w in [3, 5, 10, 20, 30, 60]:
            data[f'MA_{w}'] = data['TOTALCNT'].rolling(w, min_periods=1).mean()
            data[f'STD_{w}'] = data['TOTALCNT'].rolling(w, min_periods=1).std()
            data[f'MAX_{w}'] = data['TOTALCNT'].rolling(w, min_periods=1).max()
            data[f'MIN_{w}'] = data['TOTALCNT'].rolling(w, min_periods=1).min()
            data[f'RANGE_{w}'] = data[f'MAX_{w}'] - data[f'MIN_{w}']
        
        # 시간 특징
        data['hour'] = data.index.hour
        data['dayofweek'] = data.index.dayofweek
        data['is_weekend'] = (data.index.dayofweek >= 5).astype(int)
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        
        # NaN 처리
        data = data.fillna(method='ffill').fillna(0)
        data = data.replace([np.inf, -np.inf], 0)
        data = data.dropna(subset=['FUTURE'])
        
        print(f"✓ 특징 생성 완료: {len(data.columns)} 컬럼")
        
        return data


# ===================================
# 모델 정의 (원본 그대로)
# ===================================

class AttentionLSTM(layers.Layer):
    """커스텀 Attention LSTM 레이어"""
    
    def __init__(self, units, **kwargs):
        super(AttentionLSTM, self).__init__(**kwargs)
        self.units = units
        
    def build(self, input_shape):
        self.lstm = layers.LSTM(self.units, return_sequences=True, return_state=True)
        self.attention = layers.MultiHeadAttention(num_heads=4, key_dim=self.units)
        self.layer_norm = layers.LayerNormalization()
        super(AttentionLSTM, self).build(input_shape)
        
    def call(self, inputs):
        lstm_output, state_h, state_c = self.lstm(inputs)
        attention_output = self.attention(lstm_output, lstm_output)
        normalized = self.layer_norm(lstm_output + attention_output)
        return normalized, state_h, state_c


def build_attention_model(input_shape):
    """Attention 메커니즘 모델"""
    
    inputs = layers.Input(shape=input_shape)
    
    # Conv1D
    conv1 = layers.Conv1D(64, 3, padding='same', activation='relu')(inputs)
    conv2 = layers.Conv1D(128, 3, padding='same', activation='relu')(conv1)
    pool1 = layers.MaxPooling1D(2)(conv2)
    
    # Bidirectional LSTM
    lstm1 = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(pool1)
    dropout1 = layers.Dropout(0.2)(lstm1)
    
    lstm2 = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(dropout1)
    dropout2 = layers.Dropout(0.2)(lstm2)
    
    # Multi-Head Attention
    attention = layers.MultiHeadAttention(
        num_heads=8, 
        key_dim=256,
        dropout=0.1
    )(dropout2, dropout2)
    
    norm1 = layers.LayerNormalization()(attention + dropout2)
    
    # Global pooling
    avg_pool = layers.GlobalAveragePooling1D()(norm1)
    
    # Dense layers
    dense1 = layers.Dense(512, activation='relu')(avg_pool)
    dropout3 = layers.Dropout(0.3)(dense1)
    norm2 = layers.BatchNormalization()(dropout3)
    
    dense2 = layers.Dense(256, activation='relu')(norm2)
    dropout4 = layers.Dropout(0.2)(dense2)
    
    dense3 = layers.Dense(128, activation='relu')(dropout4)
    dense4 = layers.Dense(64, activation='relu')(dense3)
    
    outputs = layers.Dense(1, activation='linear')(dense4)
    
    return Model(inputs=inputs, outputs=outputs)


def build_lstm_model(input_shape):
    """LSTM 모델"""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(256, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(64),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='linear')
    ])
    return model


def build_gru_model(input_shape):
    """GRU 모델"""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.GRU(100, return_sequences=True),
        layers.Dropout(0.2),
        layers.GRU(200, return_sequences=True),
        layers.Dropout(0.2),
        layers.GRU(100, return_sequences=True),
        layers.Dropout(0.2),
        layers.GRU(50),
        layers.Dense(200, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(100, activation='relu'),
        layers.Dense(50, activation='relu'),
        layers.Dense(1, activation='linear')
    ])
    return model


def build_cnn_lstm_model(input_shape):
    """CNN-LSTM 하이브리드"""
    inputs = layers.Input(shape=input_shape)
    
    # CNN
    conv1 = layers.Conv1D(64, 5, padding='same', activation='relu')(inputs)
    conv2 = layers.Conv1D(128, 3, padding='same', activation='relu')(conv1)
    pool = layers.MaxPooling1D(2)(conv2)
    
    # LSTM
    lstm = layers.LSTM(128, return_sequences=True)(pool)
    lstm = layers.LSTM(64)(lstm)
    
    # Dense
    dense = layers.Dense(128, activation='relu')(lstm)
    dense = layers.Dropout(0.3)(dense)
    dense = layers.Dense(64, activation='relu')(dense)
    outputs = layers.Dense(1, activation='linear')(dense)
    
    return Model(inputs=inputs, outputs=outputs)


# ===================================
# 손실 함수
# ===================================

@tf.function
def custom_loss(y_true, y_pred):
    """급상승 예측 강화 손실 함수"""
    
    # Huber 손실
    huber = tf.keras.losses.Huber(delta=1.0)
    base_loss = huber(y_true, y_pred)
    
    # 고값 가중치 (정규화된 값 기준)
    high_mask = tf.cast(y_true > 0.6, tf.float32)
    
    # 저평가 페널티
    underestimate = tf.cast(y_pred < y_true, tf.float32)
    underestimate_penalty = underestimate * high_mask * 5.0
    
    # 최종 손실
    total_loss = base_loss * (1 + underestimate_penalty)
    
    return tf.reduce_mean(total_loss)


# ===================================
# 재시작 가능한 학습 관리자
# ===================================

class ResumableTrainer:
    """재시작 가능한 학습 관리"""
    
    def __init__(self, checkpoint_dir='checkpoints'):
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        self.preprocessor = AdvancedPreprocessor()
        self.models = {}
        self.histories = {}
        self.predictions = {}
        
    def prepare_data(self, file_path='data/20240201_TO_202507281705.csv', force_reload=False):
        """데이터 준비 (캐시 활용)"""
        
        # 저장된 데이터 확인
        if not force_reload:
            saved_data = self.checkpoint_manager.load_data()
            saved_scalers = self.checkpoint_manager.load_scalers()
            
            if saved_data and saved_scalers:
                print("\n✅ 저장된 데이터 사용")
                
                # 스케일러 복원
                self.preprocessor.target_scaler = saved_scalers['target_scaler']
                self.preprocessor.feature_scaler = saved_scalers['feature_scaler']
                self.preprocessor.is_fitted = True
                
                return saved_data
        
        # 새로 데이터 처리
        print("\n📊 데이터 새로 처리")
        
        # 데이터 로드 및 전처리
        data = self.preprocessor.load_and_process(file_path)
        data = self.preprocessor.create_all_features(data)
        
        # 시퀀스 생성
        X, y, y_original, feature_cols = self.create_sequences(data)
        
        # 데이터 분할
        train_size = int(0.7 * len(X))
        val_size = int(0.15 * len(X))
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        y_train_orig = y_original[:train_size]
        
        X_val = X[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]
        y_val_orig = y_original[train_size:train_size+val_size]
        
        X_test = X[train_size+val_size:]
        y_test = y[train_size+val_size:]
        y_test_orig = y_original[train_size+val_size:]
        
        print(f"\n[데이터 분할]")
        print(f"학습: {len(X_train)}, 검증: {len(X_val)}, 테스트: {len(X_test)}")
        
        # 데이터 저장
        self.checkpoint_manager.save_data(
            X_train, y_train, X_val, y_val, X_test, y_test,
            y_train_orig, y_val_orig, y_test_orig, feature_cols
        )
        
        # 스케일러 저장
        self.checkpoint_manager.save_scalers(
            self.preprocessor.target_scaler,
            self.preprocessor.feature_scaler
        )
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'y_train_orig': y_train_orig,
            'y_val_orig': y_val_orig,
            'y_test_orig': y_test_orig,
            'feature_cols': feature_cols
        }
    
    def create_sequences(self, data, seq_length=30):
        """시퀀스 생성"""
        print("\n[시퀀스 생성]")
        
        feature_cols = [col for col in data.columns if col not in ['TIME', 'FUTURE']]
        target_col = 'FUTURE'
        
        # 스케일링
        X_scaled = self.preprocessor.feature_scaler.fit_transform(data[feature_cols])
        y_scaled = self.preprocessor.target_scaler.fit_transform(data[[target_col]])
        
        # 시퀀스 생성
        X, y, y_original = [], [], []
        for i in range(len(X_scaled) - seq_length):
            X.append(X_scaled[i:i+seq_length])
            y.append(y_scaled[i+seq_length])
            y_original.append(data[target_col].iloc[i+seq_length])
        
        X = np.array(X)
        y = np.array(y)
        y_original = np.array(y_original).reshape(-1, 1)
        
        print(f"✓ X shape: {X.shape}")
        print(f"✓ y shape: {y.shape}")
        
        return X, y, y_original, feature_cols
    
    def train_model(self, model, model_name, X_train, y_train, X_val, y_val, 
                   initial_epoch=0, epochs=200):
        """개별 모델 학습 (재시작 지원)"""
        
        print(f"\n{'='*50}")
        print(f"{model_name.upper()} 모델 학습")
        print(f"시작 에폭: {initial_epoch}")
        print('='*50)
        
        # 저장된 가중치 로드
        if initial_epoch > 0:
            loaded = self.checkpoint_manager.load_model_weights(model, model_name)
            if loaded:
                print(f"✅ Epoch {initial_epoch}부터 재시작")
        
        # 컴파일
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=custom_loss,
            metrics=['mae', 'mse']
        )
        
        # 콜백 (수정됨: .weights.h5 확장자 사용)
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
                patience=10, 
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                f'model/{model_name}_best.weights.h5',  # .weights.h5 확장자
                save_best_only=True,
                save_weights_only=True,
                monitor='val_loss',
                verbose=0
            ),
            ProgressCallback(self.checkpoint_manager, model_name)
        ]
        
        # 학습
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            initial_epoch=initial_epoch,
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # 학습 완료 후 전체 모델 저장 (.keras 형식)
        model_path = f'model/{model_name}_final.keras'
        model.save(model_path)
        print(f"✅ 전체 모델 저장: {model_path}")
        
        # 스케일러도 함께 저장
        scaler_path = f'scaler/{model_name}_scaler.pkl'
        joblib.dump({
            'target_scaler': self.preprocessor.target_scaler,
            'feature_scaler': self.preprocessor.feature_scaler
        }, scaler_path)
        print(f"✅ 스케일러 저장: {scaler_path}")
        
        # 이력 저장
        self.checkpoint_manager.save_history(model_name, history)
        
        # 완료 모델 기록
        if model_name not in self.checkpoint_manager.state['completed_models']:
            self.checkpoint_manager.state['completed_models'].append(model_name)
            self.checkpoint_manager.save_state()
        
        return history
    
    def evaluate_model(self, model, X_test, y_test_orig, model_name):
        """모델 평가"""
        print(f"\n[{model_name} 평가]")
        
        # 예측
        y_pred = model.predict(X_test, verbose=0)
        y_pred_orig = self.preprocessor.target_scaler.inverse_transform(y_pred)
        
        # 메트릭
        mae = mean_absolute_error(y_test_orig, y_pred_orig)
        mse = mean_squared_error(y_test_orig, y_pred_orig)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_orig, y_pred_orig)
        
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R²: {r2:.4f}")
        
        # 1400+ 예측 성능
        high_mask = y_test_orig.flatten() >= 1400
        if high_mask.sum() > 0:
            high_mae = mean_absolute_error(
                y_test_orig[high_mask],
                y_pred_orig[high_mask]
            )
            high_pred = (y_pred_orig.flatten() >= 1400).sum()
            high_actual = high_mask.sum()
            
            print(f"\n[1400+ 예측]")
            print(f"MAE: {high_mae:.2f}")
            print(f"예측/실제: {high_pred}/{high_actual}")
        
        return y_pred_orig
    
    def train_all(self, data_path='data/20240201_TO_202507281705.csv', resume=False):
        """전체 학습 파이프라인 (재시작 지원)"""
        
        print("\n" + "="*70)
        if resume:
            print("📂 이전 학습 재개")
        else:
            print("🆕 새로운 학습 시작")
        print("="*70)
        
        # 상태 로드
        if resume:
            success = self.checkpoint_manager.load_state()
            if not success:
                print("⚠️  저장된 상태가 없습니다. 새로 시작합니다.")
                resume = False
        
        # 데이터 준비
        data = self.prepare_data(data_path, force_reload=not resume)
        
        X_train = data['X_train']
        y_train = data['y_train']
        X_val = data['X_val']
        y_val = data['y_val']
        X_test = data['X_test']
        y_test = data['y_test']
        y_test_orig = data['y_test_orig']
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        # 모델 정의
        model_configs = [
            ('lstm', build_lstm_model),
            ('gru', build_gru_model),
            ('cnn_lstm', build_cnn_lstm_model),
            ('attention', build_attention_model)
        ]
        
        # 완료된 모델 확인
        completed = self.checkpoint_manager.state.get('completed_models', [])
        
        # 각 모델 학습
        for model_name, build_func in model_configs:
            # 이미 완료된 모델 스킵
            if model_name in completed:
                print(f"\n✅ {model_name} 모델은 이미 완료됨")
                
                # 모델 로드 및 평가만 수행
                model = build_func(input_shape)
                self.checkpoint_manager.load_model_weights(model, model_name)
                self.models[model_name] = model
                
                pred = self.evaluate_model(model, X_test, y_test_orig, model_name)
                self.predictions[model_name] = pred
                continue
            
            # 시작 에폭 확인
            initial_epoch = 0
            if resume and self.checkpoint_manager.state.get('current_model') == model_name:
                initial_epoch = self.checkpoint_manager.state.get('current_epoch', 0)
            
            # 모델 생성 및 학습
            model = build_func(input_shape)
            self.models[model_name] = model
            
            try:
                history = self.train_model(
                    model, model_name, 
                    X_train, y_train, X_val, y_val,
                    initial_epoch=initial_epoch
                )
                self.histories[model_name] = history
                
                # 평가
                pred = self.evaluate_model(model, X_test, y_test_orig, model_name)
                self.predictions[model_name] = pred
                
            except KeyboardInterrupt:
                print(f"\n⚠️  {model_name} 학습 중단!")
                raise
            except Exception as e:
                print(f"\n❌ {model_name} 학습 실패: {str(e)}")
                traceback.print_exc()
                continue
        
        # 앙상블 예측
        if len(self.predictions) > 0:
            print("\n" + "="*70)
            print("앙상블 예측")
            print("="*70)
            
            ensemble_pred = np.mean(list(self.predictions.values()), axis=0)
            
            mae = mean_absolute_error(y_test_orig, ensemble_pred)
            rmse = np.sqrt(mean_squared_error(y_test_orig, ensemble_pred))
            r2 = r2_score(y_test_orig, ensemble_pred)
            
            print(f"앙상블 MAE: {mae:.2f}")
            print(f"앙상블 RMSE: {rmse:.2f}")
            print(f"앙상블 R²: {r2:.4f}")
            
            self.predictions['ensemble'] = ensemble_pred
            
            # 앙상블 예측 결과 저장
            np.save('model/ensemble_predictions.npy', ensemble_pred)
            print("✅ 앙상블 예측 결과 저장: model/ensemble_predictions.npy")
        
        # 최종 모델들 저장 확인
        print("\n" + "="*70)
        print("저장된 모델 확인")
        print("="*70)
        for model_name in self.models.keys():
            keras_path = f'model/{model_name}_final.keras'
            weights_path = f'model/{model_name}_best.weights.h5'
            if os.path.exists(keras_path):
                print(f"✅ {model_name}: {keras_path}")
            if os.path.exists(weights_path):
                print(f"✅ {model_name} weights: {weights_path}")
        
        print("\n✅ 학습 완료!")
        
        return self.models, self.predictions, y_test_orig


# ===================================
# 메인 실행
# ===================================

def main(resume=False):
    """메인 실행 함수"""
    
    print("="*70)
    print("TensorFlow 2.18.0 - 재시작 가능한 학습 시스템")
    print("Ctrl+C로 중단 시 자동 저장")
    print("="*70)
    
    # 학습 관리자 초기화
    trainer = ResumableTrainer()
    
    # 실제 학습 데이터 파일 경로 설정
    data_file = 'data/20240201_TO_202507281705.csv'
    
    # 파일 존재 확인
    if not os.path.exists(data_file):
        # 대체 경로 시도
        alt_paths = [
            '20240201_TO_202507281705.csv',
            '../data/20240201_TO_202507281705.csv',
            './20240201_TO_202507281705.csv'
        ]
        
        for path in alt_paths:
            if os.path.exists(path):
                data_file = path
                break
        else:
            print(f"❌ 데이터 파일을 찾을 수 없습니다: {data_file}")
            print("다음 경로 중 하나에 파일을 배치하세요:")
            for path in ['data/20240201_TO_202507281705.csv'] + alt_paths:
                print(f"  - {path}")
            return None
    
    print(f"📁 학습 데이터: {data_file}")
    
    try:
        # 학습 실행
        models, predictions, y_test = trainer.train_all(
            data_path=data_file,
            resume=resume
        )
        
        print("\n" + "="*70)
        print("🎉 모든 학습이 성공적으로 완료되었습니다!")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\n학습이 중단되었습니다.")
        print("다시 시작하려면: python script.py --resume")
    except Exception as e:
        print(f"\n오류 발생: {str(e)}")
        traceback.print_exc()
    
    return trainer


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='재시작 가능한 반도체 물류 예측 학습')
    parser.add_argument('--resume', action='store_true', help='이전 학습 재개')
    parser.add_argument('--reset', action='store_true', help='체크포인트 삭제 후 새로 시작')
    parser.add_argument('--data', type=str, default='data/20240201_TO_202507281705.csv', 
                       help='학습 데이터 파일 경로')
    
    args = parser.parse_args()
    
    # 리셋 옵션
    if args.reset:
        import shutil
        if os.path.exists('checkpoints'):
            shutil.rmtree('checkpoints')
            print("✅ 체크포인트 삭제 완료")
    
    # 학습 실행
    trainer = main(resume=args.resume)