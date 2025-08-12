"""
반도체 물류 예측을 위한 하이브리드 딥러닝 모델 v3.0 - 급증 예측 특화
================================================================
본 시스템은 반도체 팹 간 물류 이동량을 예측하고 TOTALCNT > 1400 급증을 
사전에 감지하기 위한 통합 예측 모델입니다.

주요 기능:
1. 개별 구간 데이터를 활용한 급증 패턴 학습
2. 이중 출력 모델 (수치 예측 + 급증 확률 예측)
3. LSTM, RNN, GRU, ARIMA 모델 앙상블
4. 급증 구간에 최적화된 손실 함수
5. 10분 후 물류량 예측 및 급증 감지
6. 학습 중단 시 재시작 기능 지원

개발일: 2024년
버전: 3.0 (급증 예측 특화 버전)
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, Bidirectional, GRU, SimpleRNN, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report, confusion_matrix
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime, timedelta
import joblib
import logging
import warnings
import json
import pickle
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
        logging.FileHandler('training_v3.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===================================
# 2. 체크포인트 관리 클래스
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
        
    def save_model_weights(self, model, model_name, epoch):
        """모델 가중치 저장"""
        weights_path = os.path.join(self.checkpoint_dir, f'{model_name}_weights_epoch_{epoch}.h5')
        model.save_weights(weights_path)
        return weights_path
        
    def load_model_weights(self, model, weights_path):
        """모델 가중치 로드"""
        if os.path.exists(weights_path):
            model.load_weights(weights_path)
            logger.info(f"모델 가중치 로드됨: {weights_path}")
            return True
        return False

# ===================================
# 3. 급증 예측 특화 손실 함수
# ===================================

def spike_aware_loss(y_true, y_pred, spike_weight=5.0, spike_threshold=1400):
    """급증 구간에 가중치를 부여하는 손실 함수"""
    # y_true의 첫 번째 채널은 TOTALCNT 예측값
    totalcnt_true = y_true[:, 0]
    totalcnt_pred = y_pred[:, 0]
    
    # 기본 MSE 손실
    mse_loss = tf.square(totalcnt_true - totalcnt_pred)
    
    # 급증 구간 마스크 (원본 스케일로 역변환 필요)
    # 여기서는 스케일된 값으로 근사적으로 처리
    spike_mask = tf.cast(totalcnt_true > 0.5, tf.float32)  # 스케일된 임계값
    
    # 급증 구간에 가중치 적용
    weighted_loss = tf.where(spike_mask > 0, mse_loss * spike_weight, mse_loss)
    
    # 급증 예측 실패 시 추가 페널티
    false_negative = tf.cast((totalcnt_true > 0.5) & (totalcnt_pred <= 0.5), tf.float32)
    penalty = false_negative * 10.0
    
    return tf.reduce_mean(weighted_loss + penalty)

# ===================================
# 4. 이중 출력 모델 클래스
# ===================================

class DualOutputModels:
    """수치 예측과 급증 확률을 동시에 예측하는 모델"""
    
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.models = {}
        self.histories = {}
        
    def build_dual_lstm_model(self):
        """이중 출력 LSTM 모델"""
        inputs = Input(shape=self.input_shape, name='input')
        
        # 공통 특징 추출 레이어
        x = LSTM(units=128, return_sequences=True)(inputs)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        
        x = LSTM(units=128, return_sequences=True)(x)
        x = Dropout(0.2)(x)
        x = BatchNormalization()(x)
        
        x = LSTM(units=64, return_sequences=False)(x)
        x = Dropout(0.2)(x)
        
        # 공유 특징
        shared_features = Dense(units=32, activation='relu')(x)
        
        # 수치 예측 브랜치
        regression_branch = Dense(units=16, activation='relu')(shared_features)
        regression_output = Dense(units=1, name='regression_output')(regression_branch)
        
        # 급증 분류 브랜치
        classification_branch = Dense(units=16, activation='relu')(shared_features)
        classification_output = Dense(units=1, activation='sigmoid', name='spike_output')(classification_branch)
        
        # 모델 생성
        model = Model(inputs=inputs, outputs=[regression_output, classification_output])
        
        return model
    
    def build_dual_gru_model(self):
        """이중 출력 GRU 모델"""
        inputs = Input(shape=self.input_shape, name='input')
        
        # 공통 특징 추출 레이어
        x = GRU(units=128, return_sequences=True)(inputs)
        x = Dropout(0.2)(x)
        
        x = GRU(units=64, return_sequences=False)(x)
        x = Dropout(0.2)(x)
        
        # 공유 특징
        shared_features = Dense(units=32, activation='relu')(x)
        
        # 수치 예측 브랜치
        regression_output = Dense(units=1, name='regression_output')(shared_features)
        
        # 급증 분류 브랜치
        classification_output = Dense(units=1, activation='sigmoid', name='spike_output')(shared_features)
        
        model = Model(inputs=inputs, outputs=[regression_output, classification_output])
        
        return model
    
    def build_dual_rnn_model(self):
        """이중 출력 RNN 모델"""
        inputs = Input(shape=self.input_shape, name='input')
        
        # 특징 추출
        x = SimpleRNN(units=100, return_sequences=True)(inputs)
        x = Dropout(0.2)(x)
        
        x = SimpleRNN(units=50, return_sequences=False)(x)
        x = Dropout(0.2)(x)
        
        # 공유 특징
        shared_features = Dense(units=32, activation='relu')(x)
        
        # 이중 출력
        regression_output = Dense(units=1, name='regression_output')(shared_features)
        classification_output = Dense(units=1, activation='sigmoid', name='spike_output')(shared_features)
        
        model = Model(inputs=inputs, outputs=[regression_output, classification_output])
        
        return model
    
    def build_dual_bilstm_model(self):
        """이중 출력 양방향 LSTM 모델"""
        inputs = Input(shape=self.input_shape, name='input')
        
        # 양방향 LSTM
        x = Bidirectional(LSTM(units=64, return_sequences=True))(inputs)
        x = Dropout(0.2)(x)
        
        x = Bidirectional(LSTM(units=32, return_sequences=False))(x)
        x = Dropout(0.2)(x)
        
        # 공유 특징
        shared_features = Dense(units=32, activation='relu')(x)
        
        # 이중 출력
        regression_output = Dense(units=1, name='regression_output')(shared_features)
        classification_output = Dense(units=1, activation='sigmoid', name='spike_output')(shared_features)
        
        model = Model(inputs=inputs, outputs=[regression_output, classification_output])
        
        return model

# ===================================
# 5. 개선된 학습 함수
# ===================================

def train_dual_model_with_checkpoint(model, model_name, X_train, y_train_reg, y_train_cls, 
                                    X_val, y_val_reg, y_val_cls, epochs, batch_size, 
                                    checkpoint_manager, start_epoch=0, initial_lr=0.0005):
    """이중 출력 모델 학습 함수"""
    
    # 컴파일
    optimizer = Adam(learning_rate=initial_lr)
    model.compile(
        optimizer=optimizer,
        loss={
            'regression_output': 'mse',
            'spike_output': 'binary_crossentropy'
        },
        loss_weights={
            'regression_output': 1.0,
            'spike_output': 2.0  # 급증 예측에 더 높은 가중치
        },
        metrics={
            'regression_output': 'mae',
            'spike_output': ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        }
    )
    
    # 학습 이력 초기화
    history = {
        'loss': [], 'val_loss': [],
        'regression_output_loss': [], 'val_regression_output_loss': [],
        'spike_output_loss': [], 'val_spike_output_loss': [],
        'spike_output_accuracy': [], 'val_spike_output_accuracy': []
    }
    
    # 기존 이력 로드
    state = checkpoint_manager.load_state()
    if state and model_name in state.get('model_histories', {}):
        history = state['model_histories'][model_name]
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 25
    
    try:
        for epoch in range(start_epoch, epochs):
            logger.info(f"\n{model_name} - Epoch {epoch+1}/{epochs}")
            
            # 에폭별 학습
            epoch_history = model.fit(
                X_train, 
                {'regression_output': y_train_reg, 'spike_output': y_train_cls},
                validation_data=(X_val, {'regression_output': y_val_reg, 'spike_output': y_val_cls}),
                epochs=1,
                batch_size=batch_size,
                verbose=1
            )
            
            # 이력 업데이트
            for key in history.keys():
                if key in epoch_history.history:
                    history[key].append(epoch_history.history[key][0])
            
            # 현재 검증 손실
            current_val_loss = epoch_history.history['val_loss'][0]
            
            # 최고 성능 모델 저장
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_weights_path = checkpoint_manager.save_model_weights(model, model_name, epoch)
                patience_counter = 0
                
                # 급증 예측 정확도 출력
                spike_acc = epoch_history.history.get('val_spike_output_accuracy', [0])[0]
                logger.info(f"최고 성능 갱신! Val Loss: {best_val_loss:.4f}, Spike Acc: {spike_acc:.3f}")
            else:
                patience_counter += 1
            
            # 조기 종료
            if patience_counter >= patience:
                logger.info(f"조기 종료 - {patience}에폭 동안 개선 없음")
                break
            
            # 매 5에폭마다 체크포인트 저장
            if (epoch + 1) % 5 == 0:
                current_state = checkpoint_manager.load_state() or {}
                current_state['current_model'] = model_name
                current_state['current_epoch'] = epoch + 1
                
                if 'model_histories' not in current_state:
                    current_state['model_histories'] = {}
                current_state['model_histories'][model_name] = history
                
                checkpoint_manager.save_state(current_state)
                checkpoint_manager.save_model_weights(model, f"{model_name}_checkpoint", epoch)
                
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
    
    # 최고 성능 가중치 로드
    if 'best_weights_path' in locals():
        checkpoint_manager.load_model_weights(model, best_weights_path)
    
    return history

# ===================================
# 6. 메인 학습 프로세스
# ===================================

def main(resume=False):
    """메인 학습 프로세스"""
    
    checkpoint_manager = CheckpointManager()
    
    # 재시작 모드 확인
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
        logger.info("반도체 물류 예측 모델 v3.0 - 급증 예측 특화")
        logger.info("="*60)
        
        # 데이터 로드
        Full_data_path = 'data/20240201_TO_202507281705.csv'
        logger.info(f"데이터 로딩 중: {Full_data_path}")
        
        Full_Data = pd.read_csv(Full_data_path)
        logger.info(f"원본 데이터 shape: {Full_Data.shape}")
        
        # 시간 컬럼 변환
        Full_Data['CURRTIME'] = pd.to_datetime(Full_Data['CURRTIME'], format='%Y%m%d%H%M')
        Full_Data['TIME'] = pd.to_datetime(Full_Data['TIME'], format='%Y%m%d%H%M')
        
        # SUM 컬럼은 유지 (개별 구간 합계 정보)
        # 필요한 컬럼 선택
        required_columns = ['CURRTIME', 'TOTALCNT', 'M14AM10A', 'M10AM14A', 
                          'M14AM14B', 'M14BM14A', 'M14AM16', 'M16M14A', 'TIME']
        
        # 존재하는 컬럼만 선택
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
        
        # FUTURE 컬럼 생성 (10분 후 TOTALCNT)
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
        
        # 급증 비율 확인
        spike_ratio = Modified_Data['future_spike'].mean()
        logger.info(f"급증 구간 비율: {spike_ratio:.2%}")
        
        # 개별 구간 비율 특징 생성
        segment_columns = ['M14AM10A', 'M10AM14A', 'M14AM14B', 'M14BM14A', 'M14AM16', 'M16M14A']
        available_segments = [col for col in segment_columns if col in Modified_Data.columns]
        
        for col in available_segments:
            # 각 구간의 TOTALCNT 대비 비율
            Modified_Data[f'{col}_ratio'] = Modified_Data[col] / (Modified_Data['TOTALCNT'] + 1e-6)
            
            # 10분 변화율
            Modified_Data[f'{col}_change_10'] = Modified_Data[col].pct_change(10).fillna(0)
            
            # 5분 이동평균
            Modified_Data[f'{col}_MA5'] = Modified_Data[col].rolling(window=5, min_periods=1).mean()
        
        # 급증 관련 핵심 특징 생성
        if 'M14AM14B' in Modified_Data.columns:
            Modified_Data['M14AM14B_spike_signal'] = (Modified_Data['M14AM14B_change_10'] > 0.5).astype(int)
        if 'M16M14A' in Modified_Data.columns:
            Modified_Data['M16M14A_spike_signal'] = (Modified_Data['M16M14A_change_10'] > 0.5).astype(int)
        
        # 기존 특징 추가
        Modified_Data['hour'] = Modified_Data.index.hour
        Modified_Data['dayofweek'] = Modified_Data.index.dayofweek
        Modified_Data['is_weekend'] = (Modified_Data.index.dayofweek >= 5).astype(int)
        Modified_Data['MA_5'] = Modified_Data['TOTALCNT'].rolling(window=5, min_periods=1).mean()
        Modified_Data['MA_10'] = Modified_Data['TOTALCNT'].rolling(window=10, min_periods=1).mean()
        Modified_Data['MA_30'] = Modified_Data['TOTALCNT'].rolling(window=30, min_periods=1).mean()
        Modified_Data['STD_5'] = Modified_Data['TOTALCNT'].rolling(window=5, min_periods=1).std()
        Modified_Data['STD_10'] = Modified_Data['TOTALCNT'].rolling(window=10, min_periods=1).std()
        Modified_Data['change_rate'] = Modified_Data['TOTALCNT'].pct_change()
        Modified_Data['change_rate_5'] = Modified_Data['TOTALCNT'].pct_change(5)
        
        # 결측값 처리
        Modified_Data = Modified_Data.ffill().fillna(0)
        
        # 스케일링할 컬럼 선택
        scaling_columns = ['TOTALCNT', 'FUTURE'] + available_segments
        scaling_columns += [col for col in Modified_Data.columns if 'MA' in col or 'STD' in col]
        scaling_columns += [f'{seg}_MA5' for seg in available_segments if f'{seg}_MA5' in Modified_Data.columns]
        scaling_columns = list(set(scaling_columns))  # 중복 제거
        
        # 스케일링
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(Modified_Data[scaling_columns])
        
        scaled_columns = [f'scaled_{col}' for col in scaling_columns]
        scaled_df = pd.DataFrame(scaled_data, columns=scaled_columns, index=Modified_Data.index)
        
        # 비율과 변화율은 스케일링하지 않음
        non_scaled_features = [col for col in Modified_Data.columns 
                             if ('ratio' in col or 'change' in col or 'signal' in col or 
                                 col in ['hour', 'dayofweek', 'is_weekend'])]
        
        # 최종 데이터 결합
        Scaled_Data = pd.concat([Modified_Data[non_scaled_features + ['future_spike']], scaled_df], axis=1)
        
        # 시퀀스 데이터 생성
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
        
        # 시퀀스 생성
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
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        # 급증 데이터 분포 확인
        logger.info(f"훈련 데이터 급증 비율: {y_train_cls.mean():.2%}")
        logger.info(f"검증 데이터 급증 비율: {y_val_cls.mean():.2%}")
        logger.info(f"테스트 데이터 급증 비율: {y_test_cls.mean():.2%}")
        
        # 전처리된 데이터 저장
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
            'input_features': input_features
        })
    
    # 이중 출력 모델 초기화
    dual_models = DualOutputModels(input_shape)
    
    # 학습 파라미터
    EPOCHS = 250
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0003
    
    # 모델 리스트
    model_configs = [
        ('dual_lstm', dual_models.build_dual_lstm_model),
        ('dual_gru', dual_models.build_dual_gru_model),
        ('dual_rnn', dual_models.build_dual_rnn_model),
        ('dual_bilstm', dual_models.build_dual_bilstm_model)
    ]
    
    # 각 모델 학습
    state = checkpoint_manager.load_state() if resume else {}
    completed_models = state.get('completed_models', [])
    
    for model_name, build_func in model_configs:
        if model_name in completed_models:
            logger.info(f"\n{model_name} 모델은 이미 학습이 완료되었습니다.")
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"{model_name.upper()} 모델 학습 시작")
        logger.info(f"{'='*60}")
        
        # 모델 빌드
        model = build_func()
        
        # 재시작 시 가중치 로드
        start_epoch = 0
        if resume and state.get('current_model') == model_name:
            start_epoch = state.get('current_epoch', 0)
            weights_path = os.path.join(checkpoint_manager.checkpoint_dir, 
                                       f'{model_name}_checkpoint_weights_epoch_{start_epoch-1}.h5')
            if checkpoint_manager.load_model_weights(model, weights_path):
                logger.info(f"{model_name} 모델 가중치 로드 완료. Epoch {start_epoch}부터 재시작")
        
        try:
            # 학습 실행
            history = train_dual_model_with_checkpoint(
                model, model_name, X_train, y_train_reg, y_train_cls,
                X_val, y_val_reg, y_val_cls,
                EPOCHS, BATCH_SIZE, checkpoint_manager,
                start_epoch=start_epoch, initial_lr=LEARNING_RATE
            )
            
            dual_models.models[model_name] = model
            dual_models.histories[model_name] = history
            
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
    # 7. 모델 평가 및 앙상블
    # ===================================
    
    logger.info("\n모델 성능 평가 시작...")
    
    # 개선된 앙상블 예측 함수
    def enhanced_ensemble_predict(models, X_test, spike_threshold=0.7):
        """급증 예측을 반영한 앙상블 예측"""
        
        regression_preds = {}
        spike_preds = {}
        
        # 각 모델별 예측
        for model_name, model in models.items():
            pred = model.predict(X_test, verbose=0)
            regression_preds[model_name] = pred[0].flatten()
            spike_preds[model_name] = pred[1].flatten()
        
        # 가중 평균으로 앙상블
        weights = {
            'dual_lstm': 0.35,
            'dual_gru': 0.25,
            'dual_rnn': 0.15,
            'dual_bilstm': 0.25
        }
        
        ensemble_regression = np.zeros_like(list(regression_preds.values())[0])
        ensemble_spike = np.zeros_like(list(spike_preds.values())[0])
        
        for model_name in regression_preds:
            weight = weights.get(model_name, 0.25)
            ensemble_regression += weight * regression_preds[model_name]
            ensemble_spike += weight * spike_preds[model_name]
        
        # 급증 확률이 높으면 예측값 상향 조정
        spike_mask = ensemble_spike > spike_threshold
        ensemble_regression[spike_mask] *= 1.15  # 15% 상향 조정
        
        return ensemble_regression, ensemble_spike, regression_preds, spike_preds
    
    # 앙상블 예측 수행
    ensemble_reg, ensemble_spike, reg_preds, spike_preds = enhanced_ensemble_predict(
        dual_models.models, X_test
    )
    
    # 성능 평가
    def evaluate_dual_model(y_true_reg, y_pred_reg, y_true_cls, y_pred_cls, model_name, scaler, scaling_columns):
        """이중 출력 모델 평가"""
        
        # 회귀 성능
        mse = mean_squared_error(y_true_reg, y_pred_reg)
        mae = mean_absolute_error(y_true_reg, y_pred_reg)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true_reg, y_pred_reg)
        
        # 역변환을 위한 더미 배열 생성
        n_features = len(scaling_columns)
        dummy_array = np.zeros((len(y_true_reg), n_features))
        
        # FUTURE 컬럼의 인덱스 찾기
        future_idx = scaling_columns.index('FUTURE') if 'FUTURE' in scaling_columns else 0
        
        # 실제값 역변환
        dummy_array[:, future_idx] = y_true_reg
        y_true_original = scaler.inverse_transform(dummy_array)[:, future_idx]
        
        # 예측값 역변환
        dummy_array[:, future_idx] = y_pred_reg
        y_pred_original = scaler.inverse_transform(dummy_array)[:, future_idx]
        
        mae_original = mean_absolute_error(y_true_original, y_pred_original)
        
        # 분류 성능 (이진 변환)
        y_pred_cls_binary = (y_pred_cls > 0.5).astype(int)
        
        # 분류 리포트
        cls_report = classification_report(y_true_cls, y_pred_cls_binary, 
                                         output_dict=True, zero_division=0)
        
        # 혼동 행렬
        cm = confusion_matrix(y_true_cls, y_pred_cls_binary)
        
        logger.info(f"\n{model_name} 모델 성능:")
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
    saved_data = checkpoint_manager.load_data()
    scaling_columns = saved_data['scaling_columns']
    
    results = {}
    for model_name in reg_preds:
        results[model_name] = evaluate_dual_model(
            y_test_reg, reg_preds[model_name],
            y_test_cls, spike_preds[model_name],
            model_name.upper(), scaler, scaling_columns
        )
    
    # 앙상블 평가
    results['ensemble'] = evaluate_dual_model(
        y_test_reg, ensemble_reg,
        y_test_cls, ensemble_spike,
        "ENSEMBLE", scaler, scaling_columns
    )
    
    # ===================================
    # 8. 결과 시각화
    # ===================================
    
    logger.info("\n결과 시각화 생성 중...")
    
    # 1. 급증 예측 성능 비교
    plt.figure(figsize=(15, 10))
    
    # 모델별 급증 예측 지표
    models_list = list(results.keys())
    f1_scores = [results[model]['spike_f1'] for model in models_list]
    recalls = [results[model]['spike_recall'] for model in models_list]
    precisions = [results[model]['spike_precision'] for model in models_list]
    
    x = np.arange(len(models_list))
    width = 0.25
    
    plt.bar(x - width, f1_scores, width, label='F1-Score', color='blue')
    plt.bar(x, recalls, width, label='Recall', color='green')
    plt.bar(x + width, precisions, width, label='Precision', color='red')
    
    plt.xlabel('모델')
    plt.ylabel('점수')
    plt.title('모델별 급증 예측 성능 비교')
    plt.xticks(x, models_list)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 값 표시
    for i in range(len(models_list)):
        plt.text(i - width, f1_scores[i] + 0.01, f'{f1_scores[i]:.2f}', ha='center')
        plt.text(i, recalls[i] + 0.01, f'{recalls[i]:.2f}', ha='center')
        plt.text(i + width, precisions[i] + 0.01, f'{precisions[i]:.2f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('spike_prediction_performance.png', dpi=300)
    plt.close()
    
    # 2. 예측 결과 시각화 (급증 구간 강조)
    plt.figure(figsize=(20, 12))
    
    # 샘플 구간
    sample_size = min(300, len(y_test_reg))
    
    # 실제값 역변환
    dummy_array = np.zeros((sample_size, len(scaling_columns)))
    future_idx = scaling_columns.index('FUTURE') if 'FUTURE' in scaling_columns else 0
    dummy_array[:, future_idx] = y_test_reg[:sample_size]
    y_test_original = scaler.inverse_transform(dummy_array)[:, future_idx]
    
    # 앙상블 예측값 역변환
    dummy_array[:, future_idx] = ensemble_reg[:sample_size]
    ensemble_original = scaler.inverse_transform(dummy_array)[:, future_idx]
    
    # 서브플롯 1: 예측값 vs 실제값
    plt.subplot(2, 1, 1)
    plt.plot(y_test_original, label='실제값', color='black', linewidth=2)
    plt.plot(ensemble_original, label='앙상블 예측', color='darkred', linewidth=1.5, alpha=0.8)
    
    # 급증 구간 표시
    spike_mask = y_test_cls[:sample_size] == 1
    plt.scatter(np.where(spike_mask)[0], y_test_original[spike_mask], 
               color='red', s=50, label='실제 급증', zorder=5)
    
    # 예측된 급증 구간
    pred_spike_mask = (ensemble_spike[:sample_size] > 0.7)
    plt.scatter(np.where(pred_spike_mask)[0], ensemble_original[pred_spike_mask], 
               color='orange', s=30, marker='^', label='예측 급증', zorder=4)
    
    plt.axhline(y=1400, color='red', linestyle='--', alpha=0.5, label='급증 임계값')
    plt.title('물류량 예측 결과 (10분 후)', fontsize=16)
    plt.ylabel('물류량 (TOTALCNT)', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # 서브플롯 2: 급증 확률
    plt.subplot(2, 1, 2)
    plt.plot(ensemble_spike[:sample_size], label='급증 확률', color='red', linewidth=1.5)
    plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='결정 임계값')
    plt.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='조정 임계값')
    plt.fill_between(range(sample_size), 0, ensemble_spike[:sample_size], 
                     where=(ensemble_spike[:sample_size] > 0.7), alpha=0.3, color='red')
    plt.title('급증 확률 예측', fontsize=14)
    plt.xlabel('시간 (분)', fontsize=12)
    plt.ylabel('확률', fontsize=12)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('prediction_results_with_spike.png', dpi=300)
    plt.close()
    
    # 3. 혼동 행렬 시각화
    from sklearn.metrics import ConfusionMatrixDisplay
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for idx, (model_name, result) in enumerate(results.items()):
        if idx < 6:
            cm = result['confusion_matrix']
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                        display_labels=['정상', '급증'])
            disp.plot(ax=axes[idx], cmap='Blues', values_format='d')
            axes[idx].set_title(f'{model_name.upper()} 혼동 행렬')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300)
    plt.close()
    
    # ===================================
    # 9. 모델 및 결과 저장
    # ===================================
    
    logger.info("\n모델 및 결과 저장 중...")
    
    # 저장 디렉토리
    os.makedirs('model_v3', exist_ok=True)
    os.makedirs('scaler_v3', exist_ok=True)
    os.makedirs('results_v3', exist_ok=True)
    
    # 1. 모델 저장
    for model_name, model in dual_models.models.items():
        model_path = f'model_v3/{model_name}_final.keras'
        model.save(model_path)
        logger.info(f"{model_name} 모델 저장: {model_path}")
    
    # 2. 스케일러 저장
    scaler_path = 'scaler_v3/scaler_v3.pkl'
    joblib.dump(scaler, scaler_path)
    logger.info(f"스케일러 저장: {scaler_path}")
    
    # 3. 성능 결과 저장
    results_df = pd.DataFrame(results).T
    results_df.to_csv('results_v3/model_performance.csv')
    logger.info("성능 결과 저장: results_v3/model_performance.csv")
    
    # 4. 설정 저장
    config = {
        'seq_length': 30,
        'future_minutes': 10,
        'spike_threshold': 1400,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'input_features': input_features,
        'scaling_columns': scaling_columns,
        'model_weights': {
            'dual_lstm': 0.35,
            'dual_gru': 0.25,
            'dual_rnn': 0.15,
            'dual_bilstm': 0.25
        }
    }
    
    with open('results_v3/training_config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    # ===================================
    # 10. 최종 요약
    # ===================================
    
    logger.info("\n" + "="*60)
    logger.info("학습 완료 요약")
    logger.info("="*60)
    
    # 최고 급증 예측 성능 모델
    best_spike_model = max(results.items(), 
                          key=lambda x: x[1]['spike_f1'])
    
    logger.info(f"\n최고 급증 예측 모델: {best_spike_model[0].upper()}")
    logger.info(f"  - F1-Score: {best_spike_model[1]['spike_f1']:.3f}")
    logger.info(f"  - Recall: {best_spike_model[1]['spike_recall']:.3f}")
    logger.info(f"  - Precision: {best_spike_model[1]['spike_precision']:.3f}")
    
    # 앙상블 성능
    logger.info(f"\n앙상블 모델 성능:")
    logger.info(f"  - MAE: {results['ensemble']['mae_original']:.2f}")
    logger.info(f"  - 급증 예측 정확도: {results['ensemble']['spike_accuracy']:.3f}")
    logger.info(f"  - 급증 F1-Score: {results['ensemble']['spike_f1']:.3f}")
    
    # 목표 달성 여부
    target_recall = 0.7
    if results['ensemble']['spike_recall'] >= target_recall:
        logger.info(f"\n✓ 목표 달성! 급증 예측 재현율: {results['ensemble']['spike_recall']:.1%} (목표: {target_recall:.0%})")
    else:
        logger.info(f"\n목표 미달성. 급증 예측 재현율: {results['ensemble']['spike_recall']:.1%} (목표: {target_recall:.0%})")
    
    logger.info("\n" + "="*60)
    logger.info("모든 작업이 완료되었습니다!")
    logger.info("="*60)

# ===================================
# 11. 실행
# ===================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='반도체 물류 예측 모델 v3.0')
    parser.add_argument('--resume', action='store_true', 
                       help='이전 학습을 이어서 진행')
    parser.add_argument('--reset', action='store_true',
                       help='체크포인트를 삭제하고 처음부터 시작')
    
    args = parser.parse_args()
    
    if args.reset:
        import shutil
        if os.path.exists('checkpoints_v3'):
            shutil.rmtree('checkpoints_v3')
            logger.info("체크포인트가 삭제되었습니다. 처음부터 시작합니다.")
    
    main(resume=args.resume)