"""
반도체 물류 예측을 위한 하이브리드 딥러닝 모델 v3.2 - 급증 재현율 70% 목표
================================================================
주요 개선사항:
1. 손실 가중치 대폭 증가 (15→30)
2. 임계값 추가 하향 (0.3→0.25)
3. 더 강력한 급증 특징 엔지니어링
4. Focal Loss 파라미터 최적화
5. 학습률 및 배치 크기 조정
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, Bidirectional, GRU, SimpleRNN, Concatenate, Attention
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os
import joblib
import logging
import warnings
import json
import pickle
import traceback
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# ===================================
# 1. 환경 설정
# ===================================
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

RANDOM_SEED = 2079936
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_v3.2.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===================================
# 2. 체크포인트 관리 클래스
# ===================================
class CheckpointManager:
    def __init__(self, checkpoint_dir='checkpoints_v32'):
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
            pickle.dump(data_dict, f)
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
# 3. 개선된 손실 함수
# ===================================
def improved_focal_loss(gamma=3.0, alpha=0.9):
    """더 강력한 Focal Loss"""
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_factor = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        focal_weight = tf.pow((1 - p_t), gamma)
        
        cross_entropy = -tf.math.log(p_t)
        loss = alpha_factor * focal_weight * cross_entropy
        
        return tf.reduce_mean(loss)
    
    return focal_loss_fixed

def custom_spike_loss(y_true, y_pred):
    """급증 예측에 특화된 커스텀 손실 함수"""
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    
    # False Negative에 매우 높은 페널티
    fn_weight = 10.0
    fp_weight = 1.0
    
    # Binary crossentropy with custom weights
    loss = -fn_weight * y_true * tf.math.log(y_pred) - fp_weight * (1 - y_true) * tf.math.log(1 - y_pred)
    
    return tf.reduce_mean(loss)

# ===================================
# 4. 개선된 모델 클래스
# ===================================
class EnhancedDualOutputModels:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.models = {}
        self.histories = {}
        
    def build_dual_lstm_model(self):
        """개선된 LSTM 모델 - 급증 예측 브랜치 강화"""
        inputs = Input(shape=self.input_shape, name='input')
        
        # 더 깊은 공통 레이어
        x = LSTM(units=300, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.001))(inputs)
        x = Dropout(0.3)(x)
        x = BatchNormalization()(x)
        
        x = LSTM(units=300, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = Dropout(0.3)(x)
        x = BatchNormalization()(x)
        
        x = LSTM(units=150, return_sequences=False)(x)
        x = Dropout(0.3)(x)
        
        # 더 큰 공유 특징
        shared_features = Dense(units=128, activation='relu')(x)
        shared_features = BatchNormalization()(shared_features)
        shared_features = Dropout(0.2)(shared_features)
        
        # 수치 예측 브랜치
        regression_branch = Dense(units=64, activation='relu')(shared_features)
        regression_branch = Dropout(0.2)(regression_branch)
        regression_output = Dense(units=1, name='regression_output')(regression_branch)
        
        # 급증 분류 브랜치 (더욱 강화)
        classification_branch = Dense(units=128, activation='relu')(shared_features)
        classification_branch = Dropout(0.4)(classification_branch)
        classification_branch = BatchNormalization()(classification_branch)
        
        classification_branch = Dense(units=64, activation='relu')(classification_branch)
        classification_branch = Dropout(0.4)(classification_branch)
        classification_branch = BatchNormalization()(classification_branch)
        
        classification_branch = Dense(units=32, activation='relu')(classification_branch)
        classification_branch = Dropout(0.3)(classification_branch)
        
        classification_output = Dense(units=1, activation='sigmoid', name='spike_output')(classification_branch)
        
        model = Model(inputs=inputs, outputs=[regression_output, classification_output])
        return model
    
    def build_dual_gru_model(self):
        inputs = Input(shape=self.input_shape, name='input')
        
        x = GRU(units=300, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.001))(inputs)
        x = Dropout(0.3)(x)
        x = BatchNormalization()(x)
        
        x = GRU(units=150, return_sequences=False)(x)
        x = Dropout(0.3)(x)
        
        shared_features = Dense(units=128, activation='relu')(x)
        shared_features = BatchNormalization()(shared_features)
        
        # 수치 예측
        regression_branch = Dense(units=64, activation='relu')(shared_features)
        regression_output = Dense(units=1, name='regression_output')(regression_branch)
        
        # 급증 예측 (강화)
        classification_branch = Dense(units=128, activation='relu')(shared_features)
        classification_branch = Dropout(0.4)(classification_branch)
        classification_branch = Dense(units=64, activation='relu')(classification_branch)
        classification_branch = Dropout(0.4)(classification_branch)
        classification_output = Dense(units=1, activation='sigmoid', name='spike_output')(classification_branch)
        
        model = Model(inputs=inputs, outputs=[regression_output, classification_output])
        return model
    
    def build_dual_bilstm_model(self):
        inputs = Input(shape=self.input_shape, name='input')
        
        x = Bidirectional(LSTM(units=150, return_sequences=True))(inputs)
        x = Dropout(0.3)(x)
        x = BatchNormalization()(x)
        
        x = Bidirectional(LSTM(units=75, return_sequences=False))(x)
        x = Dropout(0.3)(x)
        
        shared_features = Dense(units=128, activation='relu')(x)
        shared_features = BatchNormalization()(shared_features)
        
        # 이중 출력
        regression_branch = Dense(units=64, activation='relu')(shared_features)
        regression_output = Dense(units=1, name='regression_output')(regression_branch)
        
        classification_branch = Dense(units=128, activation='relu')(shared_features)
        classification_branch = Dropout(0.4)(classification_branch)
        classification_branch = Dense(units=64, activation='relu')(classification_branch)
        classification_branch = Dropout(0.4)(classification_branch)
        classification_output = Dense(units=1, activation='sigmoid', name='spike_output')(classification_branch)
        
        model = Model(inputs=inputs, outputs=[regression_output, classification_output])
        return model

# ===================================
# 5. 개선된 학습 함수
# ===================================
def train_enhanced_model(model, model_name, X_train, y_train_reg, y_train_cls, 
                        X_val, y_val_reg, y_val_cls, epochs, batch_size, 
                        checkpoint_manager, class_weight_dict=None, 
                        start_epoch=0, initial_lr=0.0003):
    
    # 최적화된 컴파일
    optimizer = Adam(learning_rate=initial_lr)
    model.compile(
        optimizer=optimizer,
        loss={
            'regression_output': 'mse',
            'spike_output': custom_spike_loss  # 커스텀 손실 함수 사용
        },
        loss_weights={
            'regression_output': 1.0,
            'spike_output': 30.0  # 대폭 증가
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
            patience=15,
            min_lr=1e-6,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_spike_output_recall',
            patience=40,  # 더 긴 patience
            mode='max',
            restore_best_weights=True,
            verbose=1,
            min_delta=0.01  # 최소 개선량
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
    
    state = checkpoint_manager.load_state()
    if state and model_name in state.get('model_histories', {}):
        history = state['model_histories'][model_name]
    
    best_spike_recall = 0.0
    patience_counter = 0
    
    try:
        for epoch in range(start_epoch, epochs):
            logger.info(f"\n{model_name} - Epoch {epoch+1}/{epochs}")
            
            # 클래스 가중치 증가
            enhanced_class_weight = {
                0: class_weight_dict[0] if class_weight_dict else 1.0,
                1: class_weight_dict[1] * 1.5 if class_weight_dict else 15.0
            }
            
            epoch_history = model.fit(
                X_train, 
                {'regression_output': y_train_reg, 'spike_output': y_train_cls},
                validation_data=(X_val, {'regression_output': y_val_reg, 'spike_output': y_val_cls}),
                epochs=1,
                batch_size=batch_size,
                class_weight={'spike_output': enhanced_class_weight},
                callbacks=callbacks,
                verbose=1
            )
            
            # 이력 업데이트
            for key in history.keys():
                if key in epoch_history.history:
                    history[key].append(epoch_history.history[key][0])
            
            current_spike_recall = epoch_history.history.get('val_spike_output_recall', [0])[0]
            
            # 최고 성능 모델 저장
            if current_spike_recall > best_spike_recall:
                best_spike_recall = current_spike_recall
                best_weights_path = checkpoint_manager.save_model_weights(model, model_name, epoch)
                patience_counter = 0
                
                spike_precision = epoch_history.history.get('val_spike_output_precision', [0])[0]
                logger.info(f"✓ 최고 급증 예측 성능 갱신!")
                logger.info(f"  - Recall: {best_spike_recall:.3f}")
                logger.info(f"  - Precision: {spike_precision:.3f}")
            else:
                patience_counter += 1
            
            # 조기 종료 조건
            if patience_counter >= 40 and best_spike_recall > 0:
                logger.info(f"조기 종료 - 40에폭 동안 spike recall 개선 없음")
                break
            
            # 매 5에폭마다 체크포인트 저장
            if (epoch + 1) % 5 == 0:
                current_state = checkpoint_manager.load_state() or {}
                current_state['current_model'] = model_name
                current_state['current_epoch'] = epoch + 1
                current_state['best_spike_recall'] = float(best_spike_recall)
                
                if 'model_histories' not in current_state:
                    current_state['model_histories'] = {}
                current_state['model_histories'][model_name] = history
                
                checkpoint_manager.save_state(current_state)
                
    except KeyboardInterrupt:
        logger.warning(f"\n{model_name} 학습이 중단되었습니다.")
        current_state = checkpoint_manager.load_state() or {}
        current_state['current_model'] = model_name
        current_state['current_epoch'] = epoch
        current_state['interrupted'] = True
        
        if 'model_histories' not in current_state:
            current_state['model_histories'] = {}
        current_state['model_histories'][model_name] = history
        
        checkpoint_manager.save_state(current_state)
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
# 6. 메인 학습 프로세스
# ===================================
def main(resume=False):
    checkpoint_manager = CheckpointManager()
    
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
                X_train = saved_data['X_train']
                y_train_reg = saved_data['y_train_reg']
                y_train_cls = saved_data['y_train_cls']
                X_val = saved_data['X_val']
                y_val_reg = saved_data['y_val_reg']
                y_val_cls = saved_data['y_val_cls']
                X_test = saved_data['X_test']
                y_test_reg = saved_data['y_test_reg']
                y_test_cls = saved_data['y_test_cls']
                scaler = saved_data['scaler']
                Modified_Data = saved_data['modified_data']
                input_shape = saved_data['input_shape']
                class_weight_dict = saved_data.get('class_weight_dict', None)
                scaling_columns = saved_data.get('scaling_columns', [])
                input_features = saved_data.get('input_features', [])
                
                logger.info("저장된 데이터를 성공적으로 로드했습니다.")
            else:
                logger.warning("저장된 데이터가 없습니다. 데이터 전처리부터 시작합니다.")
                resume = False
        else:
            logger.info("저장된 학습 상태가 없습니다. 처음부터 시작합니다.")
            resume = False
    
    # 데이터 전처리 (재시작이 아닌 경우)
    if not resume:
        logger.info("="*60)
        logger.info("반도체 물류 예측 모델 v3.2 - 급증 재현율 70% 목표")
        logger.info("="*60)
        
        # 데이터 로드
        Full_data_path = '20240201_TO_202507281705.csv'
        logger.info(f"데이터 로딩 중: {Full_data_path}")
        
        Full_Data = pd.read_csv(Full_data_path)
        logger.info(f"원본 데이터 shape: {Full_Data.shape}")
        
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
        
        # ===== 강화된 특징 엔지니어링 =====
        segment_columns = ['M14AM10A', 'M10AM14A', 'M14AM14B', 'M14BM14A', 'M14AM16', 'M16M14A']
        available_segments = [col for col in segment_columns if col in Modified_Data.columns]
        
        for col in available_segments:
            # 기본 특징
            Modified_Data[f'{col}_ratio'] = Modified_Data[col] / (Modified_Data['TOTALCNT'] + 1e-6)
            Modified_Data[f'{col}_ratio'] = Modified_Data[f'{col}_ratio'].clip(0, 1)
            
            # 변화율
            Modified_Data[f'{col}_change_5'] = Modified_Data[col].pct_change(5).fillna(0)
            Modified_Data[f'{col}_change_10'] = Modified_Data[col].pct_change(10).fillna(0)
            Modified_Data[f'{col}_change_5'] = Modified_Data[f'{col}_change_5'].replace([np.inf, -np.inf], 0).clip(-5, 5)
            Modified_Data[f'{col}_change_10'] = Modified_Data[f'{col}_change_10'].replace([np.inf, -np.inf], 0).clip(-5, 5)
            
            # 이동평균
            Modified_Data[f'{col}_MA3'] = Modified_Data[col].rolling(window=3, min_periods=1).mean()
            Modified_Data[f'{col}_MA5'] = Modified_Data[col].rolling(window=5, min_periods=1).mean()
            Modified_Data[f'{col}_MA10'] = Modified_Data[col].rolling(window=10, min_periods=1).mean()
            
            # 연속 상승 패턴
            col_shifted1 = Modified_Data[col].shift(1).fillna(Modified_Data[col])
            col_shifted2 = Modified_Data[col].shift(2).fillna(Modified_Data[col])
            col_shifted3 = Modified_Data[col].shift(3).fillna(Modified_Data[col])
            
            Modified_Data[f'{col}_rising_3'] = (
                (Modified_Data[col] > col_shifted1) & 
                (col_shifted1 > col_shifted2) & 
                (col_shifted2 > col_shifted3)
            ).astype(int)
            
            # 급격한 변화 감지
            Modified_Data[f'{col}_sharp_rise'] = (Modified_Data[f'{col}_change_10'] > 0.3).astype(int)
            Modified_Data[f'{col}_extreme_rise'] = (Modified_Data[f'{col}_change_5'] > 0.5).astype(int)
            
            # 모멘텀 지표
            Modified_Data[f'{col}_momentum'] = (
                Modified_Data[f'{col}_MA3'] > Modified_Data[f'{col}_MA10']
            ).astype(int)
        
        # 핵심 급증 특징 (더 강화)
        if 'M14AM14B' in Modified_Data.columns:
            Modified_Data['M14AM14B_spike_signal'] = (
                (Modified_Data['M14AM14B_change_10'] > 0.3) | 
                (Modified_Data['M14AM14B_extreme_rise'] == 1)
            ).astype(int)
            
        if 'M16M14A' in Modified_Data.columns:
            Modified_Data['M16M14A_spike_signal'] = (
                (Modified_Data['M16M14A_change_10'] > 0.3) | 
                (Modified_Data['M16M14A_extreme_rise'] == 1)
            ).astype(int)
        
        # 복합 급증 신호 (더 민감하게)
        sharp_rise_cols = [f'{col}_sharp_rise' for col in available_segments if f'{col}_sharp_rise' in Modified_Data.columns]
        extreme_rise_cols = [f'{col}_extreme_rise' for col in available_segments if f'{col}_extreme_rise' in Modified_Data.columns]
        
        if sharp_rise_cols:
            Modified_Data['complex_spike_signal'] = (
                (Modified_Data[sharp_rise_cols].sum(axis=1) >= 2) |
                (Modified_Data[extreme_rise_cols].sum(axis=1) >= 1)
            ).astype(int)
        
        # 급증 강도 지표
        Modified_Data['spike_intensity'] = 0.0
        for col in ['M14AM14B', 'M14BM14A', 'M16M14A']:
            if col in available_segments:
                Modified_Data['spike_intensity'] += Modified_Data[f'{col}_change_10'].clip(0, 2)
        
        # 기존 특징
        Modified_Data['hour'] = Modified_Data.index.hour
        Modified_Data['dayofweek'] = Modified_Data.index.dayofweek
        Modified_Data['is_weekend'] = (Modified_Data.index.dayofweek >= 5).astype(int)
        Modified_Data['MA_5'] = Modified_Data['TOTALCNT'].rolling(window=5, min_periods=1).mean()
        Modified_Data['MA_10'] = Modified_Data['TOTALCNT'].rolling(window=10, min_periods=1).mean()
        Modified_Data['MA_30'] = Modified_Data['TOTALCNT'].rolling(window=30, min_periods=1).mean()
        Modified_Data['STD_5'] = Modified_Data['TOTALCNT'].rolling(window=5, min_periods=1).std().fillna(0)
        Modified_Data['STD_10'] = Modified_Data['TOTALCNT'].rolling(window=10, min_periods=1).std().fillna(0)
        Modified_Data['change_rate'] = Modified_Data['TOTALCNT'].pct_change().fillna(0).replace([np.inf, -np.inf], 0).clip(-2, 2)
        Modified_Data['change_rate_5'] = Modified_Data['TOTALCNT'].pct_change(5).fillna(0).replace([np.inf, -np.inf], 0).clip(-2, 2)
        
        # 결측값 처리
        Modified_Data = Modified_Data.ffill().bfill().fillna(0)
        
        # 무한대 값 제거
        numeric_cols = Modified_Data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Modified_Data[col] = Modified_Data[col].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # 스케일링
        scaling_columns = ['TOTALCNT', 'FUTURE'] + available_segments
        scaling_columns += [col for col in Modified_Data.columns if 'MA' in col or 'STD' in col]
        scaling_columns += [f'{seg}_MA3' for seg in available_segments if f'{seg}_MA3' in Modified_Data.columns]
        scaling_columns += [f'{seg}_MA5' for seg in available_segments if f'{seg}_MA5' in Modified_Data.columns]
        scaling_columns += [f'{seg}_MA10' for seg in available_segments if f'{seg}_MA10' in Modified_Data.columns]
        scaling_columns = list(set(scaling_columns))
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(Modified_Data[scaling_columns])
        
        scaled_columns = [f'scaled_{col}' for col in scaling_columns]
        scaled_df = pd.DataFrame(scaled_data, columns=scaled_columns, index=Modified_Data.index)
        
        # 비율과 신호는 스케일링하지 않음
        non_scaled_features = [col for col in Modified_Data.columns 
                             if ('ratio' in col or 'change' in col or 'signal' in col or 
                                 'rising' in col or 'sharp' in col or 'extreme' in col or
                                 'momentum' in col or 'spike_intensity' in col or
                                 col in ['hour', 'dayofweek', 'is_weekend', 'complex_spike_signal'])]
        
        Scaled_Data = pd.concat([Modified_Data[non_scaled_features + ['future_spike']], scaled_df], axis=1)
        
        # 시퀀스 생성
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
            feature_data = data[feature_cols].values
            target_data_reg = data[target_col_reg].values
            target_data_cls = data[target_col_cls].values
            
            for i in range(len(data) - seq_length):
                X.append(feature_data[i:i+seq_length])
                y_reg.append(target_data_reg[i+seq_length])
                y_cls.append(target_data_cls[i+seq_length])
            
            return np.array(X), np.array(y_reg), np.array(y_cls)
        
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
        
        # 데이터 분할
        train_size = int(0.7 * len(X_seq_all))
        val_size = int(0.15 * len(X_seq_all))
        
        X_train = X_seq_all[:train_size]
        y_train_reg = y_reg_all[:train_size]
        y_train_cls = y_cls_all[:train_size]
        
        X_val = X_seq_all[train_size:train_size+val_size]
        y_val_reg = y_reg_all[train_size:train_size+val_size]
        y_val_cls = y_cls_all[train_size:train_size+val_size]
        
        X_test = X_seq_all[train_size+val_size:]
        y_test_reg = y_reg_all[train_size+val_size:]
        y_test_cls = y_cls_all[train_size+val_size:]
        
        # 개선된 SMOTE 적용
        logger.info("개선된 SMOTE를 사용한 데이터 증강 중...")
        X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
        
        # 데이터 정리
        X_train_reshaped = np.nan_to_num(X_train_reshaped, nan=0.0, posinf=10.0, neginf=-10.0)
        percentile_99 = np.percentile(np.abs(X_train_reshaped), 99)
        X_train_reshaped = np.clip(X_train_reshaped, -percentile_99, percentile_99)
        
        try:
            spike_ratio = y_train_cls.mean()
            logger.info(f"훈련 데이터 급증 비율 (SMOTE 전): {spike_ratio:.2%}")
            
            n_spike_samples = y_train_cls.sum()
            k_neighbors = min(5, n_spike_samples - 1) if n_spike_samples > 1 else 1
            
            if spike_ratio < 0.1 and n_spike_samples > 10:
                # 더 공격적인 오버샘플링
                target_ratio = min(0.4, spike_ratio * 5)  # 최대 5배까지 증가
                
                smote = SMOTE(
                    sampling_strategy=target_ratio,
                    random_state=RANDOM_SEED, 
                    k_neighbors=k_neighbors
                )
                X_train_resampled, y_train_cls_resampled = smote.fit_resample(X_train_reshaped, y_train_cls)
                
                original_size = len(y_train_cls)
                resampled_size = len(y_train_cls_resampled)
                
                if resampled_size > original_size:
                    spike_indices = np.where(y_train_cls == 1)[0]
                    spike_reg_mean = y_train_reg[spike_indices].mean()
                    
                    y_train_reg_extended = np.concatenate([
                        y_train_reg,
                        np.full(resampled_size - original_size, spike_reg_mean)
                    ])
                    y_train_reg = y_train_reg_extended
                
                X_train = X_train_resampled.reshape(-1, SEQ_LENGTH, len(input_features))
                y_train_cls = y_train_cls_resampled
                
                logger.info(f"SMOTE 후 훈련 데이터 급증 비율: {y_train_cls.mean():.2%}")
                logger.info(f"SMOTE 후 데이터 크기: {len(X_train)}")
                
        except Exception as e:
            logger.error(f"SMOTE 적용 실패: {str(e)}")
            logger.warning("SMOTE 없이 원본 데이터로 진행합니다.")
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        # 클래스 가중치 계산 (더 강력하게)
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train_cls),
            y=y_train_cls
        )
        # 급증 클래스에 추가 가중치
        class_weight_dict = {0: class_weights[0], 1: class_weights[1] * 2.0}
        logger.info(f"클래스 가중치: {class_weight_dict}")
        
        logger.info(f"최종 훈련 데이터 급증 비율: {y_train_cls.mean():.2%}")
        logger.info(f"검증 데이터 급증 비율: {y_val_cls.mean():.2%}")
        logger.info(f"테스트 데이터 급증 비율: {y_test_cls.mean():.2%}")
        
        # 데이터 저장
        checkpoint_manager.save_data({
            'X_train': X_train,
            'y_train_reg': y_train_reg,
            'y_train_cls': y_train_cls,
            'X_val': X_val,
            'y_val_reg': y_val_reg,
            'y_val_cls': y_val_cls,
            'X_test': X_test,
            'y_test_reg': y_test_reg,
            'y_test_cls': y_test_cls,
            'scaler': scaler,
            'modified_data': Modified_Data,
            'input_shape': input_shape,
            'scaling_columns': scaling_columns,
            'input_features': input_features,
            'class_weight_dict': class_weight_dict
        })
    
    # 모델 초기화
    enhanced_models = EnhancedDualOutputModels(input_shape)
    
    # 학습 파라미터
    EPOCHS = 300
    BATCH_SIZE = 16  # 더 작은 배치로 변경
    LEARNING_RATE = 0.0003  # 낮은 학습률
    
    # 모델 리스트
    model_configs = [
        ('enhanced_lstm', enhanced_models.build_dual_lstm_model),
        ('enhanced_gru', enhanced_models.build_dual_gru_model),
        ('enhanced_bilstm', enhanced_models.build_dual_bilstm_model)
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
            history = train_enhanced_model(
                model, model_name, X_train, y_train_reg, y_train_cls,
                X_val, y_val_reg, y_val_cls,
                EPOCHS, BATCH_SIZE, checkpoint_manager,
                class_weight_dict=class_weight_dict,
                start_epoch=start_epoch, initial_lr=LEARNING_RATE
            )
            
            enhanced_models.models[model_name] = model
            enhanced_models.histories[model_name] = history
            
        except KeyboardInterrupt:
            logger.warning("\n학습이 중단되었습니다.")
            return
        except Exception as e:
            logger.error(f"\n{model_name} 모델 학습 중 오류 발생: {str(e)}")
            return
    
    logger.info("\n" + "="*60)
    logger.info("모든 모델 학습 완료!")
    logger.info("="*60)
    
    # ===================================
    # 7. 평가 및 앙상블
    # ===================================
    
    logger.info("\n모델 성능 평가 시작...")
    
    def optimized_ensemble_predict(models, X_test, spike_threshold=0.25):  # 더 낮은 임계값
        """최적화된 앙상블 예측"""
        
        regression_preds = {}
        spike_preds = {}
        
        for model_name, model in models.items():
            pred = model.predict(X_test, verbose=0)
            regression_preds[model_name] = pred[0].flatten()
            spike_preds[model_name] = pred[1].flatten()
        
        # 모델별 가중치 (급증 예측 성능 기반)
        weights = {
            'enhanced_lstm': 0.4,
            'enhanced_gru': 0.3,
            'enhanced_bilstm': 0.3
        }
        
        ensemble_regression = np.zeros_like(list(regression_preds.values())[0])
        ensemble_spike = np.zeros_like(list(spike_preds.values())[0])
        
        for model_name in regression_preds:
            weight = weights.get(model_name, 0.33)
            ensemble_regression += weight * regression_preds[model_name]
            ensemble_spike += weight * spike_preds[model_name]
        
        # 급증 확률이 높으면 예측값 상향 조정
        spike_mask = ensemble_spike > spike_threshold
        ensemble_regression[spike_mask] *= 1.2  # 20% 상향
        
        return ensemble_regression, ensemble_spike, regression_preds, spike_preds
    
    # 앙상블 예측
    ensemble_reg, ensemble_spike, reg_preds, spike_preds = optimized_ensemble_predict(
        enhanced_models.models, X_test
    )
    
    # 평가 함수
    def evaluate_model_performance(y_true_reg, y_pred_reg, y_true_cls, y_pred_cls, 
                                 model_name, scaler, scaling_columns, threshold=0.25):
        
        # 회귀 성능
        mse = mean_squared_error(y_true_reg, y_pred_reg)
        mae = mean_absolute_error(y_true_reg, y_pred_reg)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true_reg, y_pred_reg)
        
        # 역변환
        n_features = len(scaling_columns)
        dummy_array = np.zeros((len(y_true_reg), n_features))
        future_idx = scaling_columns.index('FUTURE') if 'FUTURE' in scaling_columns else 0
        
        dummy_array[:, future_idx] = y_true_reg
        y_true_original = scaler.inverse_transform(dummy_array)[:, future_idx]
        
        dummy_array[:, future_idx] = y_pred_reg
        y_pred_original = scaler.inverse_transform(dummy_array)[:, future_idx]
        
        mae_original = mean_absolute_error(y_true_original, y_pred_original)
        
        # 분류 성능 (낮은 임계값)
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
        logger.info(f"    - F1-Score (급증): {cls_report.get('1', {}).get('f1-score', 0):.3f}")
        
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
    
    # 각 모델 평가
    results = {}
    for model_name in reg_preds:
        results[model_name] = evaluate_model_performance(
            y_test_reg, reg_preds[model_name],
            y_test_cls, spike_preds[model_name],
            model_name.upper(), scaler, scaling_columns
        )
    
    # 앙상블 평가
    results['ensemble'] = evaluate_model_performance(
        y_test_reg, ensemble_reg,
        y_test_cls, ensemble_spike,
        "ENSEMBLE", scaler, scaling_columns
    )
    
    # ===================================
    # 8. 결과 저장
    # ===================================
    
    logger.info("\n모델 및 결과 저장 중...")
    
    os.makedirs('model_v32', exist_ok=True)
    os.makedirs('results_v32', exist_ok=True)
    
    # 모델 저장
    for model_name, model in enhanced_models.models.items():
        model_path = f'model_v32/{model_name}_final.keras'
        model.save(model_path)
        logger.info(f"{model_name} 모델 저장: {model_path}")
    
    # 스케일러 저장
    scaler_path = 'model_v32/scaler_v32.pkl'
    joblib.dump(scaler, scaler_path)
    
    # 성능 결과 저장
    results_df = pd.DataFrame(results).T
    results_df.to_csv('results_v32/model_performance.csv')
    
    # 설정 저장
    config = {
        'seq_length': 30,
        'future_minutes': 10,
        'spike_threshold': 1400,
        'decision_threshold': 0.25,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'spike_loss_weight': 30.0,
        'class_weights': class_weight_dict
    }
    
    with open('results_v32/training_config.json', 'w') as f:
        json.dump(config, f, indent=4, default=str)
    
    # ===================================
    # 9. 최종 요약
    # ===================================
    
    logger.info("\n" + "="*60)
    logger.info("학습 완료 요약")
    logger.info("="*60)
    
    best_spike_model = max(results.items(), key=lambda x: x[1]['spike_recall'])
    
    logger.info(f"\n최고 급증 재현율 모델: {best_spike_model[0].upper()}")
    logger.info(f"  - Recall: {best_spike_model[1]['spike_recall']:.3f}")
    logger.info(f"  - Precision: {best_spike_model[1]['spike_precision']:.3f}")
    logger.info(f"  - F1-Score: {best_spike_model[1]['spike_f1']:.3f}")
    
    logger.info(f"\n앙상블 모델 성능:")
    logger.info(f"  - MAE: {results['ensemble']['mae_original']:.2f}")
    logger.info(f"  - 급증 Recall: {results['ensemble']['spike_recall']:.3f}")
    logger.info(f"  - 급증 Precision: {results['ensemble']['spike_precision']:.3f}")
    logger.info(f"  - 급증 F1-Score: {results['ensemble']['spike_f1']:.3f}")
    
    # 목표 달성 여부
    target_recall = 0.7
    if results['ensemble']['spike_recall'] >= target_recall:
        logger.info(f"\n✅ 목표 달성! 급증 예측 재현율: {results['ensemble']['spike_recall']:.1%}")
    else:
        logger.info(f"\n⚠️  목표 미달성. 급증 예측 재현율: {results['ensemble']['spike_recall']:.1%} (목표: {target_recall:.0%})")
    
    logger.info("\n" + "="*60)
    logger.info("모든 작업이 완료되었습니다!")
    logger.info("="*60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='반도체 물류 예측 모델 v3.2')
    parser.add_argument('--resume', action='store_true', help='이전 학습을 이어서 진행')
    parser.add_argument('--reset', action='store_true', help='체크포인트를 삭제하고 처음부터 시작')
    
    args = parser.parse_args()
    
    if args.reset:
        import shutil
        if os.path.exists('checkpoints_v32'):
            shutil.rmtree('checkpoints_v32')
            logger.info("체크포인트가 삭제되었습니다. 처음부터 시작합니다.")
    
    main(resume=args.resume)