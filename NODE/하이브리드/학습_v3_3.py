"""
반도체 물류 예측 모델 v3.3 - 메모리 최적화 및 균형잡힌 예측
================================================================
주요 수정사항:
1. float32 사용으로 메모리 50% 절약
2. 과적합 방지를 위한 손실 가중치 조정
3. 더 강한 정규화
4. 메모리 정리 강화
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, Bidirectional, GRU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os
import joblib
import logging
import warnings
import json
import pickle
import gc  # 가비지 컬렉션
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# 환경 설정 - 메모리 최적화
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

# TensorFlow 메모리 최적화
tf.keras.backend.set_floatx('float32')  # float64 대신 float32 사용

RANDOM_SEED = 2079936
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_v3.3.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===================================
# 체크포인트 관리
# ===================================
class CheckpointManager:
    def __init__(self, checkpoint_dir='checkpoints_v33'):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.state_file = os.path.join(checkpoint_dir, 'training_state.json')
        self.data_file = os.path.join(checkpoint_dir, 'preprocessed_data.pkl')
        
    def save_state(self, state_dict):
        with open(self.state_file, 'w') as f:
            json.dump(state_dict, f, indent=4, default=str)
        logger.info(f"학습 상태 저장됨: {self.state_file}")
        
    def load_state(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            logger.info(f"학습 상태 로드됨: {self.state_file}")
            return state
        return None
        
    def save_data(self, data_dict):
        with open(self.data_file, 'wb') as f:
            pickle.dump(data_dict, f, protocol=4)  # 더 효율적인 프로토콜
        logger.info(f"데이터 저장됨: {self.data_file}")
        
    def load_data(self):
        if os.path.exists(self.data_file):
            with open(self.data_file, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"데이터 로드됨: {self.data_file}")
            return data
        return None
        
    def save_model_weights(self, model, model_name, epoch):
        weights_path = os.path.join(self.checkpoint_dir, f'{model_name}_weights_epoch_{epoch}.h5')
        model.save_weights(weights_path)
        return weights_path
        
    def load_model_weights(self, model, weights_path):
        if os.path.exists(weights_path):
            model.load_weights(weights_path)
            logger.info(f"모델 가중치 로드됨: {weights_path}")
            return True
        return False

# ===================================
# 균형잡힌 손실 함수
# ===================================
def balanced_focal_loss(gamma=2.0, alpha=0.75):
    """균형잡힌 Focal Loss - 과적합 방지"""
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # 타입 변환
        y_true = tf.cast(y_true, tf.float32)
        
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_factor = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        focal_weight = tf.pow((1 - p_t), gamma)
        
        cross_entropy = -tf.math.log(p_t)
        loss = alpha_factor * focal_weight * cross_entropy
        
        return tf.reduce_mean(loss)
    
    return focal_loss_fixed

def balanced_binary_crossentropy(y_true, y_pred):
    """균형잡힌 이진 크로스엔트로피"""
    epsilon = tf.keras.backend.epsilon()
    
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    
    # 동적 가중치 계산
    pos_weight = tf.constant(5.0, dtype=tf.float32)  # 과도하지 않은 가중치
    
    loss = -pos_weight * y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
    
    return tf.reduce_mean(loss)

# ===================================
# 균형잡힌 모델 클래스
# ===================================
class BalancedDualOutputModels:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.models = {}
        self.histories = {}
        
    def build_balanced_lstm_model(self):
        """균형잡힌 LSTM 모델 - 더 강한 정규화"""
        inputs = Input(shape=self.input_shape, name='input')
        
        # 적절한 크기의 레이어
        x = LSTM(units=128, return_sequences=True, 
                kernel_regularizer=tf.keras.regularizers.l2(0.01))(inputs)
        x = Dropout(0.5)(x)  # 더 강한 드롭아웃
        x = BatchNormalization()(x)
        
        x = LSTM(units=64, return_sequences=False,
                kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = Dropout(0.5)(x)
        
        # 공유 특징
        shared_features = Dense(units=32, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        shared_features = Dropout(0.4)(shared_features)
        
        # 수치 예측 브랜치
        regression_branch = Dense(units=16, activation='relu')(shared_features)
        regression_output = Dense(units=1, name='regression_output')(regression_branch)
        
        # 급증 분류 브랜치 (단순화)
        classification_branch = Dense(units=16, activation='relu',
                                    kernel_regularizer=tf.keras.regularizers.l2(0.02))(shared_features)
        classification_branch = Dropout(0.5)(classification_branch)
        classification_output = Dense(units=1, activation='sigmoid', name='spike_output')(classification_branch)
        
        model = Model(inputs=inputs, outputs=[regression_output, classification_output])
        return model
    
    def build_balanced_gru_model(self):
        """균형잡힌 GRU 모델"""
        inputs = Input(shape=self.input_shape, name='input')
        
        x = GRU(units=128, return_sequences=True,
                kernel_regularizer=tf.keras.regularizers.l2(0.01))(inputs)
        x = Dropout(0.5)(x)
        x = BatchNormalization()(x)
        
        x = GRU(units=64, return_sequences=False)(x)
        x = Dropout(0.5)(x)
        
        shared_features = Dense(units=32, activation='relu')(x)
        
        # 수치 예측
        regression_output = Dense(units=1, name='regression_output')(
            Dense(units=16, activation='relu')(shared_features)
        )
        
        # 급증 예측
        classification_branch = Dense(units=16, activation='relu')(shared_features)
        classification_branch = Dropout(0.5)(classification_branch)
        classification_output = Dense(units=1, activation='sigmoid', name='spike_output')(classification_branch)
        
        model = Model(inputs=inputs, outputs=[regression_output, classification_output])
        return model

# ===================================
# 메모리 효율적인 학습 함수
# ===================================
def train_balanced_model(model, model_name, X_train, y_train_reg, y_train_cls,
                        X_val, y_val_reg, y_val_cls, epochs, batch_size,
                        checkpoint_manager, class_weight_dict=None,
                        start_epoch=0, initial_lr=0.001):
    
    # 균형잡힌 컴파일
    optimizer = Adam(learning_rate=initial_lr)
    model.compile(
        optimizer=optimizer,
        loss={
            'regression_output': 'mse',
            'spike_output': balanced_binary_crossentropy  # 균형잡힌 손실
        },
        loss_weights={
            'regression_output': 1.0,
            'spike_output': 10.0  # 과도하지 않은 가중치
        },
        metrics={
            'regression_output': 'mae',
            'spike_output': ['accuracy',
                           tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.Recall(name='recall')]
        }
    )
    
    # 개선된 콜백
    callbacks = [
        ReduceLROnPlateau(
            monitor='val_spike_output_recall',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',  # 전체 손실 모니터링
            patience=25,
            mode='min',
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # 학습 이력 초기화
    history = {
        'loss': [], 'val_loss': [],
        'regression_output_loss': [], 'val_regression_output_loss': [],
        'spike_output_loss': [], 'val_spike_output_loss': [],
        'spike_output_accuracy': [], 'val_spike_output_accuracy': [],
        'spike_output_recall': [], 'val_spike_output_recall': [],
        'spike_output_precision': [], 'val_spike_output_precision': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    try:
        for epoch in range(start_epoch, epochs):
            logger.info(f"\n{model_name} - Epoch {epoch+1}/{epochs}")
            
            # 메모리 정리
            if epoch % 10 == 0:
                gc.collect()
                tf.keras.backend.clear_session()
            
            # 샘플 가중치 계산 (급증 샘플에 적절한 가중치)
            sample_weights = np.ones(len(y_train_cls))
            spike_indices = np.where(y_train_cls == 1)[0]
            sample_weights[spike_indices] = 3.0  # 과도하지 않은 가중치
            
            epoch_history = model.fit(
                X_train,
                {'regression_output': y_train_reg, 'spike_output': y_train_cls},
                validation_data=(X_val, {'regression_output': y_val_reg, 'spike_output': y_val_cls}),
                epochs=1,
                batch_size=batch_size,
                sample_weight=sample_weights,
                callbacks=callbacks,
                verbose=1
            )
            
            # 이력 업데이트
            for key in history.keys():
                if key in epoch_history.history:
                    history[key].append(epoch_history.history[key][0])
            
            current_val_loss = epoch_history.history['val_loss'][0]
            current_spike_recall = epoch_history.history.get('val_spike_output_recall', [0])[0]
            current_spike_precision = epoch_history.history.get('val_spike_output_precision', [0])[0]
            
            # F1 스코어 계산
            if current_spike_precision + current_spike_recall > 0:
                f1_score = 2 * (current_spike_precision * current_spike_recall) / (current_spike_precision + current_spike_recall)
            else:
                f1_score = 0
            
            # 최고 성능 모델 저장 (F1 스코어 기준)
            if f1_score > 0.3 and current_val_loss < best_val_loss:  # F1 > 0.3 조건 추가
                best_val_loss = current_val_loss
                best_weights_path = checkpoint_manager.save_model_weights(model, model_name, epoch)
                patience_counter = 0
                
                logger.info(f"✓ 균형잡힌 성능 달성!")
                logger.info(f"  - Recall: {current_spike_recall:.3f}")
                logger.info(f"  - Precision: {current_spike_precision:.3f}")
                logger.info(f"  - F1-Score: {f1_score:.3f}")
            else:
                patience_counter += 1
            
            # 조기 종료
            if patience_counter >= 25:
                logger.info(f"조기 종료 - 25에폭 동안 개선 없음")
                break
            
            # 매 5에폭마다 체크포인트 저장
            if (epoch + 1) % 5 == 0:
                current_state = checkpoint_manager.load_state() or {}
                current_state['current_model'] = model_name
                current_state['current_epoch'] = epoch + 1
                current_state['best_val_loss'] = float(best_val_loss)
                
                if 'model_histories' not in current_state:
                    current_state['model_histories'] = {}
                current_state['model_histories'][model_name] = history
                
                checkpoint_manager.save_state(current_state)
                
    except Exception as e:
        logger.error(f"학습 중 오류 발생: {str(e)}")
        # 메모리 부족 오류 처리
        if "Unable to allocate" in str(e):
            logger.error("메모리 부족! 배치 크기를 줄이거나 모델 크기를 줄여주세요.")
            # 긴급 메모리 정리
            gc.collect()
            tf.keras.backend.clear_session()
        raise
    
    # 학습 완료
    current_state = checkpoint_manager.load_state() or {}
    if 'completed_models' not in current_state:
        current_state['completed_models'] = []
    if model_name not in current_state['completed_models']:
        current_state['completed_models'].append(model_name)
    
    if 'model_histories' not in current_state:
        current_state['model_histories'] = {}
    current_state['model_histories'][model_name] = history
    checkpoint_manager.save_state(current_state)
    
    if 'best_weights_path' in locals():
        checkpoint_manager.load_model_weights(model, best_weights_path)
    
    return history

# ===================================
# 메인 학습 프로세스
# ===================================
def main(resume=False):
    checkpoint_manager = CheckpointManager()
    
    # 기존 데이터 로드 부분은 동일...
    # (이전 코드에서 복사)
    
    if resume:
        state = checkpoint_manager.load_state()
        if state:
            logger.info("="*60)
            logger.info("이전 학습 상태에서 재시작합니다.")
            logger.info(f"마지막 모델: {state.get('current_model', 'Unknown')}")
            logger.info(f"마지막 에폭: {state.get('current_epoch', 0)}")
            logger.info("="*60)
            
            saved_data = checkpoint_manager.load_data()
            if saved_data:
                # float32로 변환
                X_train = saved_data['X_train'].astype(np.float32)
                y_train_reg = saved_data['y_train_reg'].astype(np.float32)
                y_train_cls = saved_data['y_train_cls'].astype(np.float32)
                X_val = saved_data['X_val'].astype(np.float32)
                y_val_reg = saved_data['y_val_reg'].astype(np.float32)
                y_val_cls = saved_data['y_val_cls'].astype(np.float32)
                X_test = saved_data['X_test'].astype(np.float32)
                y_test_reg = saved_data['y_test_reg'].astype(np.float32)
                y_test_cls = saved_data['y_test_cls'].astype(np.float32)
                
                scaler = saved_data['scaler']
                Modified_Data = saved_data['modified_data']
                input_shape = saved_data['input_shape']
                class_weight_dict = saved_data.get('class_weight_dict', None)
                scaling_columns = saved_data.get('scaling_columns', [])
                input_features = saved_data.get('input_features', [])
                
                logger.info("저장된 데이터를 성공적으로 로드했습니다.")
                logger.info(f"데이터 타입: {X_train.dtype}")
            else:
                logger.warning("저장된 데이터가 없습니다.")
                resume = False
        else:
            resume = False
    
    if not resume:
        logger.info("="*60)
        logger.info("반도체 물류 예측 모델 v3.3 - 메모리 최적화 및 균형잡힌 예측")
        logger.info("="*60)
        
        # 데이터 로드
        Full_data_path = '20240201_TO_202507281705.csv'
        logger.info(f"데이터 로딩 중: {Full_data_path}")
        
        Full_Data = pd.read_csv(Full_data_path, dtype=np.float32)  # float32로 로드
        logger.info(f"원본 데이터 shape: {Full_Data.shape}")
        
        # 이후 전처리 과정은 이전과 동일하지만 float32 사용
        # ... (이전 코드의 전처리 부분 복사)
        
        # 시간 컬럼 변환
        Full_Data['CURRTIME'] = pd.to_datetime(Full_Data['CURRTIME'], format='%Y%m%d%H%M')
        Full_Data['TIME'] = pd.to_datetime(Full_Data['TIME'], format='%Y%m%d%H%M')
        
        # 필요한 컬럼 선택
        required_columns = ['CURRTIME', 'TOTALCNT', 'M14AM10A', 'M10AM14A',
                          'M14AM14B', 'M14BM14A', 'M14AM16', 'M16M14A', 'TIME']
        
        available_columns = [col for col in required_columns if col in Full_Data.columns]
        Full_Data = Full_Data[available_columns]
        Full_Data.set_index('CURRTIME', inplace=True)
        
        # 날짜 범위 필터링
        start_date = pd.to_datetime('2024-02-01 00:00:00')
        end_date = pd.to_datetime('2024-07-27 23:59:59')
        Full_Data = Full_Data[(Full_Data['TIME'] >= start_date) & (Full_Data['TIME'] <= end_date)].reset_index()
        Full_Data.set_index('CURRTIME', inplace=True)
        
        # 이상치 제거
        Full_Data = Full_Data[(Full_Data['TOTALCNT'] >= 800) & (Full_Data['TOTALCNT'] <= 2500)]
        
        # FUTURE 컬럼 생성
        Modified_Data = Full_Data.copy()
        Modified_Data['FUTURE'] = pd.NA
        future_minutes = 10
        
        for i in Modified_Data.index:
            future_time = i + pd.Timedelta(minutes=future_minutes)
            if (future_time <= Modified_Data.index.max()) & (future_time in Modified_Data.index):
                Modified_Data.loc[i, 'FUTURE'] = Modified_Data.loc[future_time, 'TOTALCNT']
        
        Modified_Data.dropna(subset=['FUTURE'], inplace=True)
        
        # 급증 라벨 생성
        Modified_Data['future_spike'] = (Modified_Data['FUTURE'] > 1400).astype(int)
        
        spike_ratio = Modified_Data['future_spike'].mean()
        logger.info(f"급증 구간 비율: {spike_ratio:.2%}")
        
        # ===== 특징 엔지니어링 (간소화) =====
        segment_columns = ['M14AM10A', 'M10AM14A', 'M14AM14B', 'M14BM14A', 'M14AM16', 'M16M14A']
        available_segments = [col for col in segment_columns if col in Modified_Data.columns]
        
        for col in available_segments:
            # 기본 특징만
            Modified_Data[f'{col}_ratio'] = Modified_Data[col] / (Modified_Data['TOTALCNT'] + 1e-6)
            Modified_Data[f'{col}_change_10'] = Modified_Data[col].pct_change(10).fillna(0)
            Modified_Data[f'{col}_MA5'] = Modified_Data[col].rolling(window=5, min_periods=1).mean()
            
            # 급증 신호 (단순화)
            Modified_Data[f'{col}_spike_signal'] = (Modified_Data[f'{col}_change_10'] > 0.3).astype(int)
        
        # 기본 특징
        Modified_Data['hour'] = Modified_Data.index.hour
        Modified_Data['dayofweek'] = Modified_Data.index.dayofweek
        Modified_Data['MA_10'] = Modified_Data['TOTALCNT'].rolling(window=10, min_periods=1).mean()
        Modified_Data['STD_10'] = Modified_Data['TOTALCNT'].rolling(window=10, min_periods=1).std().fillna(0)
        Modified_Data['change_rate'] = Modified_Data['TOTALCNT'].pct_change().fillna(0).clip(-2, 2)
        
        # 결측값 처리
        Modified_Data = Modified_Data.ffill().bfill().fillna(0)
        
        # 모든 숫자 컬럼을 float32로 변환
        numeric_cols = Modified_Data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Modified_Data[col] = Modified_Data[col].astype(np.float32)
            Modified_Data[col] = Modified_Data[col].replace([np.inf, -np.inf], 0)
        
        # 스케일링
        scaling_columns = ['TOTALCNT', 'FUTURE'] + available_segments
        scaling_columns += [col for col in Modified_Data.columns if 'MA' in col or 'STD' in col]
        scaling_columns = list(set(scaling_columns))
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(Modified_Data[scaling_columns].astype(np.float32))
        
        scaled_columns = [f'scaled_{col}' for col in scaling_columns]
        scaled_df = pd.DataFrame(scaled_data, columns=scaled_columns, index=Modified_Data.index)
        
        # 비율과 신호는 스케일링하지 않음
        non_scaled_features = [col for col in Modified_Data.columns
                             if ('ratio' in col or 'change' in col or 'signal' in col or
                                 col in ['hour', 'dayofweek'])]
        
        Scaled_Data = pd.concat([Modified_Data[non_scaled_features + ['future_spike']], scaled_df], axis=1)
        
        # 시퀀스 생성 함수들...
        def split_data_by_continuity(data):
            time_diff = data.index.to_series().diff()
            split_points = time_diff > pd.Timedelta(minutes=1)
            segment_ids = split_points.cumsum()
            segments = []
            for segment_id in segment_ids.unique():
                segment = data[segment_ids == segment_id].copy()
                if len(segment) > 30:
                    segments.append(segment)
            return segments
        
        data_segments = split_data_by_continuity(Scaled_Data)
        
        def create_sequences(data, feature_cols, target_col_reg, target_col_cls, seq_length=30):
            X, y_reg, y_cls = [], [], []
            feature_data = data[feature_cols].values.astype(np.float32)
            target_data_reg = data[target_col_reg].values.astype(np.float32)
            target_data_cls = data[target_col_cls].values.astype(np.float32)
            
            for i in range(len(data) - seq_length):
                X.append(feature_data[i:i+seq_length])
                y_reg.append(target_data_reg[i+seq_length])
                y_cls.append(target_data_cls[i+seq_length])
            
            return np.array(X, dtype=np.float32), np.array(y_reg, dtype=np.float32), np.array(y_cls, dtype=np.float32)
        
        SEQ_LENGTH = 30
        input_features = [col for col in Scaled_Data.columns
                         if col not in ['scaled_FUTURE', 'future_spike']]
        
        all_X, all_y_reg, all_y_cls = [], [], []
        for segment in data_segments:
            X_seg, y_reg_seg, y_cls_seg = create_sequences(
                segment, input_features, 'scaled_FUTURE', 'future_spike', SEQ_LENGTH
            )
            if len(X_seg) > 0:
                all_X.append(X_seg)
                all_y_reg.append(y_reg_seg)
                all_y_cls.append(y_cls_seg)
        
        X_seq_all = np.concatenate(all_X, axis=0)
        y_reg_all = np.concatenate(all_y_reg, axis=0)
        y_cls_all = np.concatenate(all_y_cls, axis=0)
        
        # 균등 분할
        spike_indices = np.where(y_cls_all == 1)[0]
        normal_indices = np.where(y_cls_all == 0)[0]
        
        n_spike = len(spike_indices)
        n_normal = len(normal_indices)
        
        train_ratio = 0.7
        val_ratio = 0.15
        
        train_spike_size = int(train_ratio * n_spike)
        val_spike_size = int(val_ratio * n_spike)
        
        train_spike_idx = spike_indices[:train_spike_size]
        val_spike_idx = spike_indices[train_spike_size:train_spike_size + val_spike_size]
        test_spike_idx = spike_indices[train_spike_size + val_spike_size:]
        
        train_normal_size = int(train_ratio * n_normal)
        val_normal_size = int(val_ratio * n_normal)
        
        train_normal_idx = normal_indices[:train_normal_size]
        val_normal_idx = normal_indices[train_normal_size:train_normal_size + val_normal_size]
        test_normal_idx = normal_indices[train_normal_size + val_normal_size:]
        
        train_idx = np.concatenate([train_spike_idx, train_normal_idx])
        val_idx = np.concatenate([val_spike_idx, val_normal_idx])
        test_idx = np.concatenate([test_spike_idx, test_normal_idx])
        
        np.random.shuffle(train_idx)
        np.random.shuffle(val_idx)
        np.random.shuffle(test_idx)
        
        X_train = X_seq_all[train_idx]
        y_train_reg = y_reg_all[train_idx]
        y_train_cls = y_cls_all[train_idx]
        
        X_val = X_seq_all[val_idx]
        y_val_reg = y_reg_all[val_idx]
        y_val_cls = y_cls_all[val_idx]
        
        X_test = X_seq_all[test_idx]
        y_test_reg = y_reg_all[test_idx]
        y_test_cls = y_cls_all[test_idx]
        
        # 적당한 SMOTE 적용 (과도하지 않게)
        logger.info("균형잡힌 SMOTE 적용 중...")
        X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
        
        try:
            spike_ratio = y_train_cls.mean()
            logger.info(f"훈련 데이터 급증 비율 (SMOTE 전): {spike_ratio:.2%}")
            
            if spike_ratio < 0.1:
                # 적당한 오버샘플링 (최대 3배)
                target_ratio = min(0.15, spike_ratio * 3)
                
                n_spike_samples = y_train_cls.sum()
                k_neighbors = min(5, int(n_spike_samples) - 1) if n_spike_samples > 5 else 1
                
                smote = SMOTE(
                    sampling_strategy=target_ratio,
                    random_state=RANDOM_SEED,
                    k_neighbors=k_neighbors
                )
                X_train_resampled, y_train_cls_resampled = smote.fit_resample(X_train_reshaped, y_train_cls)
                
                # y_train_reg도 조정
                original_size = len(y_train_cls)
                resampled_size = len(y_train_cls_resampled)
                
                if resampled_size > original_size:
                    spike_indices = np.where(y_train_cls == 1)[0]
                    spike_reg_mean = y_train_reg[spike_indices].mean() if len(spike_indices) > 0 else 0
                    
                    y_train_reg_extended = np.concatenate([
                        y_train_reg,
                        np.full(resampled_size - original_size, spike_reg_mean, dtype=np.float32)
                    ])
                    y_train_reg = y_train_reg_extended
                
                X_train = X_train_resampled.reshape(-1, SEQ_LENGTH, len(input_features))
                y_train_cls = y_train_cls_resampled
                
                logger.info(f"SMOTE 후 훈련 데이터 급증 비율: {y_train_cls.mean():.2%}")
                
        except Exception as e:
            logger.error(f"SMOTE 적용 실패: {str(e)}")
            logger.warning("SMOTE 없이 원본 데이터로 진행합니다.")
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        # 클래스 가중치 (균형잡힌)
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train_cls),
            y=y_train_cls
        )
        class_weight_dict = {0: class_weights[0], 1: class_weights[1] * 1.5}  # 적당한 가중치
        
        logger.info(f"최종 훈련 데이터 급증 비율: {y_train_cls.mean():.2%}")
        logger.info(f"검증 데이터 급증 비율: {y_val_cls.mean():.2%}")
        logger.info(f"테스트 데이터 급증 비율: {y_test_cls.mean():.2%}")
        
        # 데이터 저장
        checkpoint_manager.save_data({
            'X_train': X_train.astype(np.float32),
            'y_train_reg': y_train_reg.astype(np.float32),
            'y_train_cls': y_train_cls.astype(np.float32),
            'X_val': X_val.astype(np.float32),
            'y_val_reg': y_val_reg.astype(np.float32),
            'y_val_cls': y_val_cls.astype(np.float32),
            'X_test': X_test.astype(np.float32),
            'y_test_reg': y_test_reg.astype(np.float32),
            'y_test_cls': y_test_cls.astype(np.float32),
            'scaler': scaler,
            'modified_data': Modified_Data,
            'input_shape': input_shape,
            'scaling_columns': scaling_columns,
            'input_features': input_features,
            'class_weight_dict': class_weight_dict
        })
        
        # 메모리 정리
        del Full_Data, Modified_Data, Scaled_Data, X_seq_all, y_reg_all, y_cls_all
        gc.collect()
    
    # 모델 초기화
    balanced_models = BalancedDualOutputModels(input_shape)
    
    # 학습 파라미터
    EPOCHS = 150  # 더 적은 에폭
    BATCH_SIZE = 32  # 적절한 배치 크기
    LEARNING_RATE = 0.001  # 더 높은 학습률로 시작
    
    # 모델 리스트
    model_configs = [
        ('balanced_lstm', balanced_models.build_balanced_lstm_model),
        ('balanced_gru', balanced_models.build_balanced_gru_model)
    ]
    
    # 학습 실행
    state = checkpoint_manager.load_state() if resume else {}
    completed_models = state.get('completed_models', [])
    
    for model_name, build_func in model_configs:
        if model_name in completed_models:
            logger.info(f"\n{model_name} 모델은 이미 학습이 완료되었습니다.")
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"{model_name.upper()} 모델 학습 시작")
        logger.info(f"{'='*60}")
        
        model = build_func()
        
        start_epoch = 0
        if resume and state.get('current_model') == model_name:
            start_epoch = state.get('current_epoch', 0)
            weights_path = os.path.join(checkpoint_manager.checkpoint_dir,
                                       f'{model_name}_checkpoint_weights_epoch_{start_epoch-1}.h5')
            if checkpoint_manager.load_model_weights(model, weights_path):
                logger.info(f"{model_name} 모델 가중치 로드 완료. Epoch {start_epoch}부터 재시작")
        
        try:
            history = train_balanced_model(
                model, model_name, X_train, y_train_reg, y_train_cls,
                X_val, y_val_reg, y_val_cls,
                EPOCHS, BATCH_SIZE, checkpoint_manager,
                class_weight_dict=class_weight_dict,
                start_epoch=start_epoch, initial_lr=LEARNING_RATE
            )
            
            balanced_models.models[model_name] = model
            balanced_models.histories[model_name] = history
            
            # 메모리 정리
            gc.collect()
            tf.keras.backend.clear_session()
            
        except KeyboardInterrupt:
            logger.warning("\n학습이 중단되었습니다.")
            return
        except Exception as e:
            logger.error(f"\n{model_name} 모델 학습 중 오류 발생: {str(e)}")
            # 메모리 정리 시도
            gc.collect()
            tf.keras.backend.clear_session()
            return
    
    logger.info("\n" + "="*60)
    logger.info("모든 모델 학습 완료!")
    logger.info("="*60)
    
    # 평가 및 저장 부분...
    # (평가 함수도 메모리 효율적으로 수정)
    
    def evaluate_balanced_model(y_true_reg, y_pred_reg, y_true_cls, y_pred_cls,
                               model_name, scaler, scaling_columns, threshold=0.4):
        """균형잡힌 평가"""
        
        # 회귀 성능
        mse = mean_squared_error(y_true_reg, y_pred_reg)
        mae = mean_absolute_error(y_true_reg, y_pred_reg)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true_reg, y_pred_reg)
        
        # 역변환
        n_features = len(scaling_columns)
        dummy_array = np.zeros((len(y_true_reg), n_features), dtype=np.float32)
        future_idx = scaling_columns.index('FUTURE') if 'FUTURE' in scaling_columns else 0
        
        dummy_array[:, future_idx] = y_true_reg
        y_true_original = scaler.inverse_transform(dummy_array)[:, future_idx]
        
        dummy_array[:, future_idx] = y_pred_reg
        y_pred_original = scaler.inverse_transform(dummy_array)[:, future_idx]
        
        mae_original = mean_absolute_error(y_true_original, y_pred_original)
        
        # 분류 성능 (균형잡힌 임계값)
        y_pred_cls_binary = (y_pred_cls > threshold).astype(int)
        
        cls_report = classification_report(y_true_cls, y_pred_cls_binary,
                                         output_dict=True, zero_division=0)
        
        cm = confusion_matrix(y_true_cls, y_pred_cls_binary)
        
        logger.info(f"\n{model_name} 모델 성능 (임계값: {threshold}):")
        logger.info(f"  회귀 성능:")
        logger.info(f"    - MAE (원본 스케일): {mae_original:.2f}")
        logger.info(f"    - RMSE: {rmse:.4f}")
        logger.info(f"    - R²: {r2:.4f}")
        logger.info(f"  급증 예측 성능:")
        logger.info(f"    - 정확도: {cls_report['accuracy']:.3f}")
        logger.info(f"    - 정밀도 (급증): {cls_report.get('1', {}).get('precision', 0):.3f}")
        logger.info(f"    - 재현율 (급증): {cls_report.get('1', {}).get('recall', 0):.3f}")
        logger.info(f"    - F1-Score: {cls_report.get('1', {}).get('f1-score', 0):.3f}")
        
        return {
            'mae': mae,
            'mae_original': mae_original,
            'rmse': rmse,
            'r2': r2,
            'spike_accuracy': cls_report['accuracy'],
            'spike_precision': cls_report.get('1', {}).get('precision', 0),
            'spike_recall': cls_report.get('1', {}).get('recall', 0),
            'spike_f1': cls_report.get('1', {}).get('f1-score', 0),
            'confusion_matrix': cm
        }
    
    # 모델별 예측 및 평가
    results = {}
    for model_name, model in balanced_models.models.items():
        pred = model.predict(X_test, batch_size=BATCH_SIZE*2)  # 더 큰 배치로 예측
        
        results[model_name] = evaluate_balanced_model(
            y_test_reg, pred[0].flatten(),
            y_test_cls, pred[1].flatten(),
            model_name.upper(), scaler, scaling_columns
        )
        
        # 메모리 정리
        del pred
        gc.collect()
    
    # 앙상블 예측 (간단하게)
    ensemble_reg = 0
    ensemble_spike = 0
    n_models = len(balanced_models.models)
    
    for model_name, model in balanced_models.models.items():
        pred = model.predict(X_test, batch_size=BATCH_SIZE*2)
        ensemble_reg += pred[0].flatten() / n_models
        ensemble_spike += pred[1].flatten() / n_models
        del pred
    
    results['ensemble'] = evaluate_balanced_model(
        y_test_reg, ensemble_reg,
        y_test_cls, ensemble_spike,
        "ENSEMBLE", scaler, scaling_columns
    )
    
    # 결과 저장
    logger.info("\n모델 및 결과 저장 중...")
    
    os.makedirs('model_v33', exist_ok=True)
    os.makedirs('results_v33', exist_ok=True)
    
    # 모델 저장
    for model_name, model in balanced_models.models.items():
        model_path = f'model_v33/{model_name}_final.keras'
        model.save(model_path)
        logger.info(f"{model_name} 모델 저장: {model_path}")
    
    # 스케일러 저장
    scaler_path = 'model_v33/scaler_v33.pkl'
    joblib.dump(scaler, scaler_path)
    
    # 성능 결과 저장
    results_df = pd.DataFrame(results).T
    results_df.to_csv('results_v33/model_performance.csv')
    
    # 최종 요약
    logger.info("\n" + "="*60)
    logger.info("학습 완료 요약")
    logger.info("="*60)
    
    best_f1_model = max(results.items(), key=lambda x: x[1]['spike_f1'])
    
    logger.info(f"\n최고 F1-Score 모델: {best_f1_model[0].upper()}")
    logger.info(f"  - F1-Score: {best_f1_model[1]['spike_f1']:.3f}")
    logger.info(f"  - Recall: {best_f1_model[1]['spike_recall']:.3f}")
    logger.info(f"  - Precision: {best_f1_model[1]['spike_precision']:.3f}")
    
    logger.info(f"\n앙상블 모델 성능:")
    logger.info(f"  - MAE: {results['ensemble']['mae_original']:.2f}")
    logger.info(f"  - 급증 F1-Score: {results['ensemble']['spike_f1']:.3f}")
    logger.info(f"  - 급증 Recall: {results['ensemble']['spike_recall']:.3f}")
    logger.info(f"  - 급증 Precision: {results['ensemble']['spike_precision']:.3f}")
    
    # 목표 확인
    target_recall = 0.7
    if results['ensemble']['spike_recall'] >= target_recall and results['ensemble']['spike_precision'] >= 0.3:
        logger.info(f"\n✅ 균형잡힌 성능 달성!")
    else:
        logger.info(f"\n⚠️  추가 조정 필요")
    
    logger.info("\n" + "="*60)
    logger.info("모든 작업이 완료되었습니다!")
    logger.info("="*60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='반도체 물류 예측 모델 v3.3')
    parser.add_argument('--resume', action='store_true', help='이전 학습을 이어서 진행')
    parser.add_argument('--reset', action='store_true', help='체크포인트를 삭제하고 처음부터 시작')
    
    args = parser.parse_args()
    
    if args.reset:
        import shutil
        if os.path.exists('checkpoints_v33'):
            shutil.rmtree('checkpoints_v33')
            logger.info("체크포인트가 삭제되었습니다. 처음부터 시작합니다.")
    
    try:
        main(resume=args.resume)
    except Exception as e:
        logger.error(f"프로그램 실행 중 오류: {str(e)}")
        # 최종 메모리 정리
        gc.collect()
        tf.keras.backend.clear_session()