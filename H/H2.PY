"""
반도체 물류 예측을 위한 하이브리드 딥러닝 모델 (재시작 가능 버전)
================================================================
본 시스템은 반도체 팹 간 물류 이동량을 예측하고 병목 구간을 사전에 감지하기 위한 
통합 예측 모델입니다.

주요 기능:
1. LSTM, RNN, GRU, ARIMA 모델을 통합한 앙상블 예측
2. 실시간 병목 구간 감지 및 알림
3. 10분 후 물류량 예측
4. CPU 기반 실행으로 범용성 확보
5. 학습 중단 시 재시작 기능 지원

개발일: 2024년
버전: 2.0 (재시작 가능 버전)
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, Bidirectional, GRU, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime, timedelta
from tensorflow.keras.regularizers import l2
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

# CPU 모드 설정 - GPU가 없는 환경에서도 실행 가능하도록 설정
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
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===================================
# 2. 체크포인트 관리 클래스
# ===================================

class CheckpointManager:
    """학습 상태를 저장하고 복원하는 클래스"""
    
    def __init__(self, checkpoint_dir='checkpoints'):
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
# 3. 재시작 가능한 학습 함수
# ===================================

def train_model_with_checkpoint(model, model_name, X_train, y_train, X_val, y_val, 
                                epochs, batch_size, checkpoint_manager, 
                                start_epoch=0, initial_lr=0.0005):
    """체크포인트를 지원하는 모델 학습 함수"""
    
    # 옵티마이저 설정 (학습률 조정 가능)
    optimizer = Adam(learning_rate=initial_lr)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    # 학습 이력 초기화
    history = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}
    
    # 기존 이력이 있다면 로드
    state = checkpoint_manager.load_state()
    if state and model_name in state.get('model_histories', {}):
        history = state['model_histories'][model_name]
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20
    
    try:
        for epoch in range(start_epoch, epochs):
            logger.info(f"\n{model_name} - Epoch {epoch+1}/{epochs}")
            
            # 에폭별 학습
            epoch_history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=1,
                batch_size=batch_size,
                verbose=1
            )
            
            # 이력 저장
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
                logger.info(f"최고 성능 갱신! Val Loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
            
            # 조기 종료 확인
            if patience_counter >= patience:
                logger.info(f"조기 종료 - {patience}에폭 동안 개선 없음")
                break
            
            # 매 5에폭마다 체크포인트 저장
            if (epoch + 1) % 5 == 0:
                # 현재 상태 저장
                current_state = checkpoint_manager.load_state() or {}
                current_state['current_model'] = model_name
                current_state['current_epoch'] = epoch + 1
                
                if 'model_histories' not in current_state:
                    current_state['model_histories'] = {}
                current_state['model_histories'][model_name] = history
                
                if 'completed_models' not in current_state:
                    current_state['completed_models'] = []
                    
                checkpoint_manager.save_state(current_state)
                checkpoint_manager.save_model_weights(model, f"{model_name}_checkpoint", epoch)
                
    except KeyboardInterrupt:
        logger.warning(f"\n{model_name} 학습이 사용자에 의해 중단되었습니다.")
        # 중단 시점 상태 저장
        current_state = checkpoint_manager.load_state() or {}
        current_state['current_model'] = model_name
        current_state['current_epoch'] = epoch
        current_state['interrupted'] = True
        
        if 'model_histories' not in current_state:
            current_state['model_histories'] = {}
        current_state['model_histories'][model_name] = history
        
        checkpoint_manager.save_state(current_state)
        checkpoint_manager.save_model_weights(model, f"{model_name}_interrupted", epoch)
        raise
        
    except Exception as e:
        logger.error(f"\n{model_name} 학습 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        # 오류 시점 상태 저장
        current_state = checkpoint_manager.load_state() or {}
        current_state['current_model'] = model_name
        current_state['current_epoch'] = epoch
        current_state['error'] = str(e)
        
        if 'model_histories' not in current_state:
            current_state['model_histories'] = {}
        current_state['model_histories'][model_name] = history
        
        checkpoint_manager.save_state(current_state)
        raise
    
    # 학습 완료 상태 저장
    current_state = checkpoint_manager.load_state() or {}
    if 'completed_models' not in current_state:
        current_state['completed_models'] = []
    if model_name not in current_state['completed_models']:
        current_state['completed_models'].append(model_name)
    
    # 'model_histories' 키가 없으면 생성 (수정된 부분)
    if 'model_histories' not in current_state:
        current_state['model_histories'] = {}
        
    current_state['model_histories'][model_name] = history
    checkpoint_manager.save_state(current_state)
    
    # 최고 성능 가중치 로드
    if 'best_weights_path' in locals():
        checkpoint_manager.load_model_weights(model, best_weights_path)
    
    return history

# ===================================
# 4. 메인 학습 프로세스
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
            logger.info(f"완료된 모델: {state.get('completed_models', [])}")
            logger.info("="*60)
            
            # 저장된 데이터 로드
            saved_data = checkpoint_manager.load_data()
            if saved_data:
                X_train = saved_data['X_train']
                y_train = saved_data['y_train']
                X_val = saved_data['X_val']
                y_val = saved_data['y_val']
                X_test = saved_data['X_test']
                y_test = saved_data['y_test']
                sscaler = saved_data['scaler']
                Modified_Data = saved_data['modified_data']
                input_shape = saved_data['input_shape']
                train_size = saved_data.get('train_size', int(0.7 * len(Modified_Data)))
                val_size = saved_data.get('val_size', int(0.15 * len(Modified_Data)))
                scaled_list = saved_data.get('scaled_list', [])
                input_features = saved_data.get('input_features', [])
                
                # ARIMA 모델 로드 시도
                arima_path = 'model/arima_model.pkl'
                if os.path.exists(arima_path):
                    arima_fitted = joblib.load(arima_path)
                    logger.info("저장된 ARIMA 모델을 로드했습니다.")
                else:
                    arima_fitted = None
                    
                logger.info("저장된 데이터를 성공적으로 로드했습니다.")
            else:
                logger.warning("저장된 데이터가 없습니다. 데이터 전처리부터 시작합니다.")
                resume = False
                arima_fitted = None
        else:
            logger.info("저장된 학습 상태가 없습니다. 처음부터 시작합니다.")
            resume = False
            arima_fitted = None
    
    # 데이터 전처리 (재시작이 아닌 경우)
    if not resume:
        logger.info("="*60)
        logger.info("반도체 물류 예측 하이브리드 모델 학습 시작")
        logger.info("="*60)
        
        # ARIMA 모델 초기화
        arima_fitted = None
        
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
        outside_range_filtered_data = outside_range_data[
            (outside_range_data['TOTALCNT'] >= 800) & 
            (outside_range_data['TOTALCNT'] <= 2500)
        ]
        
        Modified_Data = pd.concat([within_range_data, outside_range_filtered_data])
        Modified_Data = Modified_Data.sort_values(by='TIME')
        
        # FUTURE 컬럼 생성
        Modified_Data['FUTURE'] = pd.NA
        future_minutes = 10
        
        for i in Modified_Data.index:
            future_time = i + pd.Timedelta(minutes=future_minutes)
            if (future_time <= Modified_Data.index.max()) & (future_time in Modified_Data.index):
                Modified_Data.loc[i, 'FUTURE'] = Modified_Data.loc[future_time, 'TOTALCNT']
        
        Modified_Data.dropna(subset=['FUTURE'], inplace=True)
        
        # 특징 엔지니어링
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
        Modified_Data = Modified_Data.ffill().fillna(0)
        
        # 데이터 스케일링
        sscaler = StandardScaler()
        scaled_list = ['TOTALCNT', 'FUTURE', 'MA_5', 'MA_10', 'MA_30', 'STD_5', 'STD_10']
        scaled_list = [col for col in scaled_list if col in Modified_Data.columns]
        
        fitted_scaled = sscaler.fit(Modified_Data[scaled_list])
        scaled_list_data = fitted_scaled.transform(Modified_Data[scaled_list])
        
        scaled_columns = [f'scaled_{col}' for col in scaled_list]
        scaled_list_data = pd.DataFrame(scaled_list_data, columns=scaled_columns)
        scaled_list_data.index = Modified_Data.index
        
        Scaled_Data = pd.merge(Modified_Data, scaled_list_data, left_index=True, right_index=True, how='left')
        
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
        def create_sequences(data, feature_cols, target_col, seq_length=30):
            X, y = [], []
            feature_data = data[feature_cols].values
            target_data = data[target_col].values
            
            for i in range(len(data) - seq_length):
                X.append(feature_data[i:i+seq_length])
                y.append(target_data[i+seq_length])
            
            return np.array(X), np.array(y)
        
        SEQ_LENGTH = 30
        input_features = [col for col in Scaled_Data.columns if col.startswith('scaled_') and col != 'scaled_FUTURE']
        
        all_X, all_y = [], []
        for segment in data_segments:
            X_seg, y_seg = create_sequences(segment, input_features, 'scaled_FUTURE', SEQ_LENGTH)
            if len(X_seg) > 0:
                all_X.append(X_seg)
                all_y.append(y_seg)
        
        X_seq_all = np.concatenate(all_X, axis=0)
        y_seq_all = np.concatenate(all_y, axis=0)
        
        # 데이터 분할
        train_size = int(0.7 * len(X_seq_all))
        val_size = int(0.15 * len(X_seq_all))
        
        X_train = X_seq_all[:train_size]
        y_train = y_seq_all[:train_size]
        X_val = X_seq_all[train_size:train_size+val_size]
        y_val = y_seq_all[train_size:train_size+val_size]
        X_test = X_seq_all[train_size+val_size:]
        y_test = y_seq_all[train_size+val_size:]
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        # 전처리된 데이터 저장
        checkpoint_manager.save_data({
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'scaler': sscaler,
            'modified_data': Modified_Data,
            'input_shape': input_shape,
            'scaled_list': scaled_list,
            'train_size': train_size,
            'val_size': val_size,
            'input_features': input_features
        })
    
    # ===================================
    # 하이브리드 모델 클래스 임포트 또는 정의
    # ===================================
    
    # H.py 파일이 있다면 임포트, 없다면 여기서 정의
    try:
        from H import HybridModels
        logger.info("H.py에서 HybridModels 클래스를 임포트했습니다.")
    except ImportError:
        logger.warning("H.py를 찾을 수 없습니다. HybridModels 클래스를 직접 정의합니다.")
        
        class HybridModels:
            """LSTM, RNN, GRU 모델을 포함한 하이브리드 예측 시스템"""
            
            def __init__(self, input_shape):
                self.input_shape = input_shape
                self.models = {}
                self.histories = {}
                
            def build_lstm_model(self):
                """LSTM 모델 구축"""
                model = Sequential([
                    Input(shape=self.input_shape),
                    
                    LSTM(units=100, return_sequences=True),
                    Dropout(0.2),
                    BatchNormalization(),
                    
                    LSTM(units=100, return_sequences=True),
                    Dropout(0.2),
                    BatchNormalization(),
                    
                    LSTM(units=100, return_sequences=False),
                    Dropout(0.2),
                    
                    Dense(units=50, activation='relu'),
                    Dense(units=1)
                ])
                
                return model
            
            def build_gru_model(self):
                """GRU 모델 구축"""
                model = Sequential([
                    Input(shape=self.input_shape),
                    
                    GRU(units=100, return_sequences=True),
                    Dropout(0.2),
                    
                    GRU(units=100, return_sequences=True),
                    Dropout(0.2),
                    
                    GRU(units=50, return_sequences=False),
                    Dropout(0.2),
                    
                    Dense(units=30, activation='relu'),
                    Dense(units=1)
                ])
                
                return model
            
            def build_simple_rnn_model(self):
                """Simple RNN 모델 구축"""
                model = Sequential([
                    Input(shape=self.input_shape),
                    
                    SimpleRNN(units=100, return_sequences=True),
                    Dropout(0.2),
                    
                    SimpleRNN(units=50, return_sequences=False),
                    Dropout(0.2),
                    
                    Dense(units=30, activation='relu'),
                    Dense(units=1)
                ])
                
                return model
            
            def build_bidirectional_lstm_model(self):
                """양방향 LSTM 모델 구축"""
                model = Sequential([
                    Input(shape=self.input_shape),
                    
                    Bidirectional(LSTM(units=50, return_sequences=True)),
                    Dropout(0.2),
                    
                    Bidirectional(LSTM(units=50, return_sequences=False)),
                    Dropout(0.2),
                    
                    Dense(units=30, activation='relu'),
                    Dense(units=1)
                ])
                
                return model
    
    # 하이브리드 모델 초기화
    hybrid_models = HybridModels(input_shape)
    
    # 학습 파라미터
    EPOCHS = 200
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0005
    
    # 모델 리스트
    model_configs = [
        ('lstm', hybrid_models.build_lstm_model),
        ('gru', hybrid_models.build_gru_model),
        ('rnn', hybrid_models.build_simple_rnn_model),
        ('bi_lstm', hybrid_models.build_bidirectional_lstm_model)
    ]
    
    # 재시작 시 완료된 모델 확인
    state = checkpoint_manager.load_state() if resume else {}
    completed_models = state.get('completed_models', [])
    
    # 각 모델 학습
    for model_name, build_func in model_configs:
        if model_name in completed_models:
            logger.info(f"\n{model_name} 모델은 이미 학습이 완료되었습니다. 건너뜁니다.")
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
            # 체크포인트를 지원하는 학습 실행
            history = train_model_with_checkpoint(
                model, model_name, X_train, y_train, X_val, y_val,
                EPOCHS, BATCH_SIZE, checkpoint_manager,
                start_epoch=start_epoch, initial_lr=LEARNING_RATE
            )
            
            hybrid_models.models[model_name] = model
            hybrid_models.histories[model_name] = history
            
        except KeyboardInterrupt:
            logger.warning("\n학습이 중단되었습니다. 현재 상태가 저장되었습니다.")
            logger.info("다시 시작하려면 resume=True로 실행하세요.")
            return
        except Exception as e:
            logger.error(f"\n{model_name} 모델 학습 중 오류 발생: {str(e)}")
            logger.info("다시 시작하려면 resume=True로 실행하세요.")
            return
    
    logger.info("\n" + "="*60)
    logger.info("모든 모델 학습 완료!")
    logger.info("="*60)
    
    # ===================================
    # 16. 모델 평가
    # ===================================
    
    logger.info("\n" + "="*60)
    logger.info("모델 성능 평가")
    logger.info("="*60)
    
    # 앙상블 예측 함수
    def ensemble_predict(models, X_test, weights=None):
        """여러 모델의 예측을 앙상블하여 최종 예측 생성"""
        
        if weights is None:
            weights = {
                'lstm': 0.3,
                'gru': 0.25,
                'rnn': 0.15,
                'bi_lstm': 0.3
            }
        
        individual_preds = {}
        
        # 각 모델별 예측
        for model_name, model in models.items():
            if model_name != 'arima':  # ARIMA는 별도 처리
                pred = model.predict(X_test, verbose=0).flatten()
                individual_preds[model_name] = pred
        
        # 가중 평균 계산
        ensemble_pred = np.zeros_like(list(individual_preds.values())[0])
        
        for model_name, pred in individual_preds.items():
            weight = weights.get(model_name, 0.25)
            ensemble_pred += weight * pred
        
        return ensemble_pred, individual_preds
    
    # 앙상블 예측 수행
    ensemble_pred, individual_preds = ensemble_predict(hybrid_models.models, X_test)
    
    # 성능 지표 계산 함수
    def calculate_metrics(y_true, y_pred, model_name):
        """모델 성능 지표 계산"""
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # 역변환하여 실제 스케일로 변환
        saved_data = checkpoint_manager.load_data()
        scaled_list = saved_data['scaled_list']
        
        y_true_original = sscaler.inverse_transform(
            np.column_stack([y_true] + [np.zeros_like(y_true)] * (len(scaled_list) - 1))
        )[:, 0]
        
        y_pred_original = sscaler.inverse_transform(
            np.column_stack([y_pred] + [np.zeros_like(y_pred)] * (len(scaled_list) - 1))
        )[:, 0]
        
        mae_original = mean_absolute_error(y_true_original, y_pred_original)
        
        logger.info(f"\n{model_name} 모델 성능:")
        logger.info(f"  - MSE: {mse:.4f}")
        logger.info(f"  - MAE: {mae:.4f}")
        logger.info(f"  - RMSE: {rmse:.4f}")
        logger.info(f"  - R²: {r2:.4f}")
        logger.info(f"  - MAE (원본 스케일): {mae_original:.2f}")
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mae_original': mae_original
        }
    
    # 개별 모델 평가
    results = {}
    for model_name, pred in individual_preds.items():
        results[model_name] = calculate_metrics(y_test, pred, model_name.upper())
    
    # 앙상블 모델 평가
    results['ensemble'] = calculate_metrics(y_test, ensemble_pred, "ENSEMBLE")
    
    # ===================================
    # 17. 결과 시각화
    # ===================================
    
    logger.info("\n결과 시각화 생성 중...")
    
    # 1. 학습 곡선 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, (model_name, history) in enumerate(hybrid_models.histories.items()):
        if idx < 4:
            ax = axes[idx]
            
            # 손실 곡선
            if isinstance(history, dict):
                ax.plot(history['loss'], label='Training Loss', color='blue')
                ax.plot(history['val_loss'], label='Validation Loss', color='red')
            else:
                ax.plot(history.history['loss'], label='Training Loss', color='blue')
                ax.plot(history.history['val_loss'], label='Validation Loss', color='red')
            
            ax.set_title(f'{model_name.upper()} 학습 곡선')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 예측 결과 시각화
    plt.figure(figsize=(20, 10))
    
    # 샘플 구간 선택
    sample_size = min(200, len(y_test))
    saved_data = checkpoint_manager.load_data()
    scaled_list = saved_data['scaled_list']
    
    # 실제값을 원본 스케일로 변환
    y_test_original = sscaler.inverse_transform(
        np.column_stack([y_test[:sample_size]] + [np.zeros_like(y_test[:sample_size])] * (len(scaled_list) - 1))
    )[:, 0]
    
    # 실제값 플롯
    plt.plot(y_test_original, label='실제값', color='black', linewidth=2)
    
    # 개별 모델 예측값 플롯
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    for idx, (model_name, pred) in enumerate(individual_preds.items()):
        pred_original = sscaler.inverse_transform(
            np.column_stack([pred[:sample_size]] + [np.zeros_like(pred[:sample_size])] * (len(scaled_list) - 1))
        )[:, 0]
        
        plt.plot(pred_original, label=f'{model_name.upper()} 예측', 
                 color=colors[idx], alpha=0.7, linewidth=1.5)
    
    # 앙상블 예측값 플롯
    ensemble_original = sscaler.inverse_transform(
        np.column_stack([ensemble_pred[:sample_size]] + [np.zeros_like(ensemble_pred[:sample_size])] * (len(scaled_list) - 1))
    )[:, 0]
    
    plt.plot(ensemble_original, label='앙상블 예측', 
             color='darkred', linewidth=2, linestyle='--')
    
    plt.title('반도체 물류량 예측 결과 (10분 후)', fontsize=16)
    plt.xlabel('시간 (분)', fontsize=12)
    plt.ylabel('물류량 (TOTALCNT)', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('prediction_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 모델별 성능 비교 차트
    models_list = list(results.keys())
    mae_values = [results[model]['mae_original'] for model in models_list]
    r2_values = [results[model]['r2'] for model in models_list]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # MAE 비교
    ax1.bar(models_list, mae_values, color=['blue', 'green', 'red', 'orange', 'darkred'])
    ax1.set_title('모델별 MAE 비교 (낮을수록 좋음)', fontsize=14)
    ax1.set_ylabel('MAE (원본 스케일)', fontsize=12)
    ax1.set_xlabel('모델', fontsize=12)
    
    # 값 표시
    for i, v in enumerate(mae_values):
        ax1.text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom')
    
    # R² 비교
    ax2.bar(models_list, r2_values, color=['blue', 'green', 'red', 'orange', 'darkred'])
    ax2.set_title('모델별 R² 비교 (높을수록 좋음)', fontsize=14)
    ax2.set_ylabel('R² Score', fontsize=12)
    ax2.set_xlabel('모델', fontsize=12)
    ax2.set_ylim(0, 1)
    
    # 값 표시
    for i, v in enumerate(r2_values):
        ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ===================================
    # 18. 병목 구간 예측
    # ===================================
    
    logger.info("\n" + "="*60)
    logger.info("병목 구간 예측 분석")
    logger.info("="*60)
    
    # 병목 임계값 설정 (상위 15%)
    bottleneck_threshold = np.percentile(Modified_Data['TOTALCNT'], 85)
    logger.info(f"병목 임계값: {bottleneck_threshold:.0f}")
    
    # 예측값에서 병목 구간 찾기
    ensemble_original_full = sscaler.inverse_transform(
        np.column_stack([ensemble_pred] + [np.zeros_like(ensemble_pred)] * (len(scaled_list) - 1))
    )[:, 0]
    
    bottleneck_predictions = ensemble_original_full > bottleneck_threshold
    bottleneck_count = np.sum(bottleneck_predictions)
    bottleneck_ratio = bottleneck_count / len(bottleneck_predictions) * 100
    
    logger.info(f"예측된 병목 구간: {bottleneck_count}개 ({bottleneck_ratio:.1f}%)")
    
    # 병목 구간 시각화
    plt.figure(figsize=(20, 8))
    
    # 실제값과 예측값
    y_test_original_full = sscaler.inverse_transform(
        np.column_stack([y_test] + [np.zeros_like(y_test)] * (len(scaled_list) - 1))
    )[:, 0]
    
    plt.subplot(2, 1, 1)
    plt.plot(y_test_original_full[:500], label='실제 물류량', color='blue')
    plt.axhline(y=bottleneck_threshold, color='red', linestyle='--', label=f'병목 임계값 ({bottleneck_threshold:.0f})')
    plt.title('실제 물류량 및 병목 구간', fontsize=14)
    plt.ylabel('물류량', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(ensemble_original_full[:500], label='예측 물류량', color='green')
    plt.axhline(y=bottleneck_threshold, color='red', linestyle='--', label=f'병목 임계값 ({bottleneck_threshold:.0f})')
    
    # 병목 구간 하이라이트
    bottleneck_indices = np.where(bottleneck_predictions[:500])[0]
    if len(bottleneck_indices) > 0:
        plt.scatter(bottleneck_indices, ensemble_original_full[bottleneck_indices], 
                    color='red', s=30, label='예측된 병목', zorder=5)
    
    plt.title('예측 물류량 및 병목 구간 감지', fontsize=14)
    plt.xlabel('시간 (분)', fontsize=12)
    plt.ylabel('물류량', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bottleneck_detection.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ===================================
    # 19. 모델 및 결과 저장
    # ===================================
    
    logger.info("\n" + "="*60)
    logger.info("모델 및 결과 저장")
    logger.info("="*60)
    
    # 저장 디렉토리 생성
    os.makedirs('model', exist_ok=True)
    os.makedirs('scaler', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # 1. 딥러닝 모델 저장
    for model_name, model in hybrid_models.models.items():
        model_path = f'model/{model_name}_final_hybrid.keras'
        model.save(model_path)
        logger.info(f"{model_name.upper()} 모델 저장: {model_path}")
    
    # 2. ARIMA 모델 저장
    if arima_fitted is not None:
        arima_path = 'model/arima_model.pkl'
        joblib.dump(arima_fitted, arima_path)
        logger.info(f"ARIMA 모델 저장: {arima_path}")
    
    # 3. 스케일러 저장
    scaler_path = 'scaler/standard_scaler_hybrid.pkl'
    joblib.dump(sscaler, scaler_path)
    logger.info(f"스케일러 저장: {scaler_path}")
    
    # 4. 성능 결과 저장
    results_df = pd.DataFrame(results).T
    results_df.to_csv('results/model_performance.csv')
    logger.info("성능 결과 저장: results/model_performance.csv")
    
    # 5. 학습 설정 저장
    config = {
        'seq_length': 30,
        'future_minutes': 10,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'input_features': input_features,
        'scaled_columns': scaled_list,
        'train_size': train_size,
        'val_size': val_size,
        'test_size': len(X_test),
        'bottleneck_threshold': float(bottleneck_threshold),
        'model_weights': {
            'lstm': 0.3,
            'gru': 0.25,
            'rnn': 0.15,
            'bi_lstm': 0.3
        }
    }
    
    with open('results/training_config.json', 'w') as f:
        json.dump(config, f, indent=4)
    logger.info("학습 설정 저장: results/training_config.json")
    
    # ===================================
    # 20. 최종 요약
    # ===================================
    
    logger.info("\n" + "="*60)
    logger.info("학습 완료 요약")
    logger.info("="*60)
    
    # 최고 성능 모델 찾기
    best_model = min(results.items(), key=lambda x: x[1]['mae_original'])
    logger.info(f"\n최고 성능 모델: {best_model[0].upper()}")
    logger.info(f"  - MAE: {best_model[1]['mae_original']:.2f}")
    logger.info(f"  - R²: {best_model[1]['r2']:.4f}")
    
    # 앙상블 성능
    logger.info(f"\n앙상블 모델 성능:")
    logger.info(f"  - MAE: {results['ensemble']['mae_original']:.2f}")
    logger.info(f"  - R²: {results['ensemble']['r2']:.4f}")
    
    # 병목 예측 정확도
    actual_bottlenecks = y_test_original_full > bottleneck_threshold
    bottleneck_accuracy = np.mean(bottleneck_predictions == actual_bottlenecks) * 100
    logger.info(f"\n병목 구간 예측 정확도: {bottleneck_accuracy:.1f}%")
    
    # 체크포인트 정리
    state = checkpoint_manager.load_state()
    state['training_completed'] = True
    state['completion_time'] = datetime.now().isoformat()
    state['final_results'] = {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()}
    checkpoint_manager.save_state(state)
    
    logger.info("\n" + "="*60)
    logger.info("모든 작업이 완료되었습니다!")
    logger.info("체크포인트는 checkpoints/ 폴더에 저장되어 있습니다.")
    logger.info("="*60)

# ===================================
# 5. 실행
# ===================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='반도체 물류 예측 하이브리드 모델 학습')
    parser.add_argument('--resume', action='store_true', 
                       help='이전 학습을 이어서 진행')
    parser.add_argument('--reset', action='store_true',
                       help='체크포인트를 삭제하고 처음부터 시작')
    
    args = parser.parse_args()
    
    if args.reset:
        import shutil
        if os.path.exists('checkpoints'):
            shutil.rmtree('checkpoints')
            logger.info("체크포인트가 삭제되었습니다. 처음부터 시작합니다.")
    
    main(resume=args.resume)