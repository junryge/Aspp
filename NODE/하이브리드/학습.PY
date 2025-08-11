"""
반도체 물류 예측을 위한 하이브리드 딥러닝 모델 (재시작 가능 버전)
================================================================
* 전체 데이터 학습 및 무작위 분할 적용 버전 *

본 시스템은 반도체 팹 간 물류 이동량을 예측하고 병목 구간을 사전에 감지하기 위한 통합 예측 모델입니다.

주요 기능:
1. LSTM, RNN, GRU, ARIMA 모델을 통합한 앙상블 예측
2. 실시간 병목 구간 감지 및 알림
3. 10분 후 물류량 예측
4. CPU 기반 실행으로 범용성 확보
5. 학습 중단 시 재시작 기능 지원

개발일: 2024년
버전: 3.0 (전체 데이터 학습 버전)
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, RobustScaler # RobustScaler 추가
from sklearn.model_selection import train_test_split # train_test_split 추가
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
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
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
def train_model_with_checkpoint(model, model_name, X_train, y_train, X_val, y_val, epochs, batch_size, checkpoint_manager, start_epoch=0, initial_lr=0.0005):
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
                train_size = saved_data.get('train_size', len(X_train))
                val_size = saved_data.get('val_size', len(X_val))
                scaled_list = saved_data.get('scaled_list', [])
                input_features = saved_data.get('input_features', [])

                logger.info("저장된 데이터를 성공적으로 로드했습니다.")
            else:
                logger.warning("저장된 데이터가 없습니다. 데이터 전처리부터 시작합니다.")
                resume = False
        else:
            logger.info("저장된 학습 상태가 없습니다. 처음부터 시작합니다.")
            resume = False
    else:
        state = {} # 새로운 시작을 위해 state 초기화

    # 데이터 전처리 (재시작이 아닌 경우)
    if not resume:
        logger.info("="*60)
        logger.info("반도체 물류 예측 하이브리드 모델 학습 시작")
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
        
        # ####################################################################
        # ### 중요: 날짜 범위 필터링 제거 ###
        # ####################################################################
        # logger.info("전체 기간 데이터를 사용합니다 (날짜 필터링 없음).")
        # # start_date = pd.to_datetime('2024-02-01 00:00:00')
        # # end_date = pd.to_datetime('2024-07-27 23:59:59')
        # # Full_Data = Full_Data[(Full_Data['TIME'] >= start_date) & (Full_Data['TIME'] <= end_date)].reset_index(drop=True)
        
        # 필요한 컬럼만 선택
        Full_Data = Full_Data[['CURRTIME', 'TOTALCNT', 'TIME']]
        Full_Data.set_index('CURRTIME', inplace=True)

        # 이상치 제거 (범위를 넓게 설정하여 대부분의 데이터를 포함)
        Modified_Data = Full_Data[
            (Full_Data['TOTALCNT'] >= 500) & (Full_Data['TOTALCNT'] <= 3000)
        ].copy()
        Modified_Data = Modified_Data.sort_values(by='TIME')
        logger.info(f"기본 이상치 제거 후 데이터 shape: {Modified_Data.shape}")


        # FUTURE 컬럼 생성
        Modified_Data['FUTURE'] = pd.NA
        future_minutes = 10
        
        # 인덱스 기반으로 빠르게 미래 값 찾기
        time_to_totalcnt = pd.Series(Modified_Data['TOTALCNT'].values, index=Modified_Data.index)
        future_times = Modified_Data.index + pd.Timedelta(minutes=future_minutes)
        future_values = time_to_totalcnt.reindex(future_times).values
        Modified_Data['FUTURE'] = future_values
        
        Modified_Data.dropna(subset=['FUTURE'], inplace=True)
        logger.info(f"FUTURE 컬럼 생성 및 결측치 제거 후 데이터 shape: {Modified_Data.shape}")

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
        Modified_Data = Modified_Data.ffill().fillna(0) # 앞쪽 값으로 채우고 남은건 0으로

        # 데이터 스케일링 (RobustScaler 사용)
        sscaler = RobustScaler()
        scaled_list = ['TOTALCNT', 'FUTURE', 'MA_5', 'MA_10', 'MA_30', 'STD_5', 'STD_10']
        scaled_list = [col for col in scaled_list if col in Modified_Data.columns]
        
        fitted_scaled = sscaler.fit(Modified_Data[scaled_list])
        scaled_list_data = fitted_scaled.transform(Modified_Data[scaled_list])
        
        scaled_columns = [f'scaled_{col}' for col in scaled_list]
        scaled_list_data = pd.DataFrame(scaled_list_data, columns=scaled_columns)
        scaled_list_data.index = Modified_Data.index
        Scaled_Data = pd.merge(Modified_Data, scaled_list_data, left_index=True, right_index=True, how='left')

        # 시퀀스 데이터 생성 (시간 불연속 처리 포함)
        def split_data_by_continuity(data, max_diff_minutes=1):
            time_diff = data.index.to_series().diff()
            split_points = time_diff > pd.Timedelta(minutes=max_diff_minutes)
            segment_ids = split_points.cumsum()
            segments = []
            for segment_id in segment_ids.unique():
                segment = data[segment_ids == segment_id].copy()
                if len(segment) > 30: # 시퀀스 길이보다 길어야 함
                    segments.append(segment)
            logger.info(f"{len(segments)}개의 연속된 데이터 세그먼트 생성됨.")
            return segments

        data_segments = split_data_by_continuity(Scaled_Data)

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

        # ####################################################################
        # ### 중요: 데이터를 무작위로 섞어서 분할 ###
        # ####################################################################
        # 1. 훈련+검증 세트와 테스트 세트로 분할 (85% vs 15%)
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_seq_all, y_seq_all, test_size=0.15, random_state=RANDOM_SEED, shuffle=True
        )

        # 2. 훈련+검증 세트를 다시 훈련 세트와 검증 세트로 분할 (70% vs 15%)
        train_ratio = 0.70 / (1.0 - 0.15) # 0.7 / 0.85
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=(1-train_ratio), random_state=RANDOM_SEED, shuffle=True
        )

        train_size = len(X_train)
        val_size = len(X_val)
        
        logger.info("데이터 분할 완료 (무작위 셔플 적용):")
        logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

        input_shape = (X_train.shape[1], X_train.shape[2])

        # 전처리된 데이터 저장
        checkpoint_manager.save_data({
            'X_train': X_train, 'y_train': y_train, 'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test, 'scaler': sscaler,
            'modified_data': Modified_Data, 'input_shape': input_shape,
            'scaled_list': scaled_list, 'train_size': train_size,
            'val_size': val_size, 'input_features': input_features
        })


    # 하이브리드 모델 클래스 임포트 또는 정의
    try:
        from H import HybridModels
        logger.info("H.py에서 HybridModels 클래스를 임포트했습니다.")
    except ImportError:
        logger.warning("H.py를 찾을 수 없습니다. HybridModels 클래스를 직접 정의합니다.")
        class HybridModels:
            def __init__(self, input_shape):
                self.input_shape = input_shape
                self.models = {}
                self.histories = {}

            def build_lstm_model(self):
                model = Sequential([
                    Input(shape=self.input_shape),
                    LSTM(units=100, return_sequences=True), Dropout(0.2), BatchNormalization(),
                    LSTM(units=100, return_sequences=True), Dropout(0.2), BatchNormalization(),
                    LSTM(units=100, return_sequences=False), Dropout(0.2),
                    Dense(units=50, activation='relu'),
                    Dense(units=1)
                ])
                return model
            
            def build_gru_model(self):
                model = Sequential([
                    Input(shape=self.input_shape),
                    GRU(units=100, return_sequences=True), Dropout(0.2),
                    GRU(units=100, return_sequences=True), Dropout(0.2),
                    GRU(units=50, return_sequences=False), Dropout(0.2),
                    Dense(units=30, activation='relu'),
                    Dense(units=1)
                ])
                return model

            def build_simple_rnn_model(self):
                model = Sequential([
                    Input(shape=self.input_shape),
                    SimpleRNN(units=100, return_sequences=True), Dropout(0.2),
                    SimpleRNN(units=50, return_sequences=False), Dropout(0.2),
                    Dense(units=30, activation='relu'),
                    Dense(units=1)
                ])
                return model

            def build_bidirectional_lstm_model(self):
                model = Sequential([
                    Input(shape=self.input_shape),
                    Bidirectional(LSTM(units=50, return_sequences=True)), Dropout(0.2),
                    Bidirectional(LSTM(units=50, return_sequences=False)), Dropout(0.2),
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
    completed_models = state.get('completed_models', [])

    # 각 모델 학습
    for model_name, build_func in model_configs:
        if model_name in completed_models:
            logger.info(f"\n{model_name} 모델은 이미 학습이 완료되었습니다. 건너뜁니다.")
            # 완료된 모델은 로드해야 평가에서 사용 가능
            try:
                model_path = f'model/{model_name}_final_hybrid.keras'
                hybrid_models.models[model_name] = load_model(model_path)
                hybrid_models.histories[model_name] = state.get('model_histories', {}).get(model_name, {})
                logger.info(f"완료된 모델 {model_name} 로드 완료.")
            except Exception as e:
                logger.error(f"완료된 모델 {model_name} 로드 실패: {e}. 다시 학습합니다.")
                completed_models.remove(model_name) # 로드 실패 시 재학습
                
        if model_name not in completed_models:
            logger.info(f"\n{'='*60}")
            logger.info(f"{model_name.upper()} 모델 학습 시작")
            logger.info(f"{'='*60}")

            model = build_func()
            start_epoch = 0
            if resume and state.get('current_model') == model_name:
                start_epoch = state.get('current_epoch', 0)
                weights_path = os.path.join(checkpoint_manager.checkpoint_dir, f'{model_name}_checkpoint_weights_epoch_{start_epoch-1}.h5')
                if checkpoint_manager.load_model_weights(model, weights_path):
                    logger.info(f"{model_name} 모델 가중치 로드 완료. Epoch {start_epoch}부터 재시작")

            try:
                history = train_model_with_checkpoint(
                    model, model_name, X_train, y_train, X_val, y_val,
                    EPOCHS, BATCH_SIZE, checkpoint_manager,
                    start_epoch=start_epoch, initial_lr=LEARNING_RATE
                )
                hybrid_models.models[model_name] = model
                hybrid_models.histories[model_name] = history

            except (KeyboardInterrupt, Exception) as e:
                logger.warning(f"\n학습이 중단/오류 발생: {e}")
                logger.info("다시 시작하려면 resume=True로 실행하세요.")
                return

    logger.info("\n" + "="*60)
    logger.info("모든 모델 학습 완료!")
    logger.info("="*60)
    
    # --- 이하 평가, 시각화, 저장 코드는 이전과 동일하게 사용 ---
    
    # ===================================
    # 16. 모델 평가
    # ===================================
    logger.info("\n" + "="*60)
    logger.info("모델 성능 평가")
    logger.info("="*60)

    # 앙상블 예측 함수
    def ensemble_predict(models, X_test, weights=None):
        if weights is None:
            weights = {'lstm': 0.3, 'gru': 0.25, 'rnn': 0.15, 'bi_lstm': 0.3}
        
        individual_preds = {}
        for model_name, model in models.items():
            pred = model.predict(X_test, verbose=0).flatten()
            individual_preds[model_name] = pred
            
        ensemble_pred = np.zeros_like(list(individual_preds.values())[0])
        for model_name, pred in individual_preds.items():
            weight = weights.get(model_name, 0)
            ensemble_pred += weight * pred
        
        return ensemble_pred, individual_preds

    # 앙상블 예측 수행
    ensemble_pred, individual_preds = ensemble_predict(hybrid_models.models, X_test)

    # 성능 지표 계산 함수
    def calculate_metrics(y_true, y_pred, model_name):
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        # 역변환하여 실제 스케일로 변환
        saved_data = checkpoint_manager.load_data()
        scaler_obj = saved_data['scaler']
        scaled_list_cols = saved_data['scaled_list']
        
        y_true_reshaped = np.zeros((len(y_true), len(scaled_list_cols)))
        y_true_reshaped[:, scaled_list_cols.index('FUTURE')] = y_true
        y_true_original = scaler_obj.inverse_transform(y_true_reshaped)[:, scaled_list_cols.index('FUTURE')]

        y_pred_reshaped = np.zeros((len(y_pred), len(scaled_list_cols)))
        y_pred_reshaped[:, scaled_list_cols.index('FUTURE')] = y_pred
        y_pred_original = scaler_obj.inverse_transform(y_pred_reshaped)[:, scaled_list_cols.index('FUTURE')]

        mae_original = mean_absolute_error(y_true_original, y_pred_original)
        
        logger.info(f"\n--- {model_name} 모델 성능 ---")
        logger.info(f" - MSE (Scaled): {mse:.4f}")
        logger.info(f" - MAE (Scaled): {mae:.4f}")
        logger.info(f" - RMSE (Scaled): {rmse:.4f}")
        logger.info(f" - R² (Scaled): {r2:.4f}")
        logger.info(f" - MAE (원래 값): {mae_original:.2f}") # 가장 중요한 지표
        
        return {'mse': mse, 'mae': mae, 'rmse': rmse, 'r2': r2, 'mae_original': mae_original}

    # 개별 및 앙상블 모델 평가
    results = {}
    for model_name, pred in individual_preds.items():
        results[model_name] = calculate_metrics(y_test, pred, model_name.upper())
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
            if isinstance(history, dict) and 'loss' in history:
                ax.plot(history['loss'], label='Training Loss', color='blue')
                ax.plot(history['val_loss'], label='Validation Loss', color='red')
            elif hasattr(history, 'history'):
                ax.plot(history.history['loss'], label='Training Loss', color='blue')
                ax.plot(history.history['val_loss'], label='Validation Loss', color='red')

            ax.set_title(f'{model_name.upper()} Training & Validation Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300)
    plt.close()

    # 2. 예측 결과 시각화
    plt.figure(figsize=(20, 10))
    sample_size = min(300, len(y_test)) # 샘플 사이즈
    
    saved_data = checkpoint_manager.load_data()
    scaler_obj = saved_data['scaler']
    scaled_list_cols = saved_data['scaled_list']

    # 스케일 역변환 함수
    def inverse_transform_future(y_scaled):
        y_reshaped = np.zeros((len(y_scaled), len(scaled_list_cols)))
        y_reshaped[:, scaled_list_cols.index('FUTURE')] = y_scaled
        return scaler_obj.inverse_transform(y_reshaped)[:, scaled_list_cols.index('FUTURE')]

    y_test_original = inverse_transform_future(y_test[:sample_size])
    
    # 실제값 플롯
    plt.plot(y_test_original, label='Actual', color='black', linewidth=2, marker='o', markersize=3)
    
    # 개별 모델 예측값 플롯
    colors = ['blue', 'green', 'purple', 'orange']
    for idx, (model_name, pred) in enumerate(individual_preds.items()):
        pred_original = inverse_transform_future(pred[:sample_size])
        plt.plot(pred_original, label=f'{model_name.upper()} Pred.', color=colors[idx], alpha=0.6, linewidth=1.5)

    # 앙상블 예측값 플롯
    ensemble_original = inverse_transform_future(ensemble_pred[:sample_size])
    plt.plot(ensemble_original, label='Ensemble Pred.', color='red', linewidth=2, linestyle='--')

    plt.title('Logistics Volume Prediction (10 Minutes Ahead)', fontsize=16)
    plt.xlabel('Time (Minutes)', fontsize=12)
    plt.ylabel('Volume (TOTALCNT)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('prediction_results.png', dpi=300)
    plt.close()

    # ===================================
    # 18. 병목 구간 예측
    # ===================================
    logger.info("\n" + "="*60)
    logger.info("병목 구간 예측 분석")
    logger.info("="*60)

    bottleneck_threshold = np.percentile(Modified_Data['TOTALCNT'], 85)
    logger.info(f"병목 임계값 (상위 85%): {bottleneck_threshold:.0f}")

    ensemble_original_full = inverse_transform_future(ensemble_pred)
    bottleneck_predictions = ensemble_original_full > bottleneck_threshold
    bottleneck_count = np.sum(bottleneck_predictions)
    bottleneck_ratio = bottleneck_count / len(bottleneck_predictions) * 100
    logger.info(f"예측된 병목 구간: {bottleneck_count}개 ({bottleneck_ratio:.1f}%)")

    # ===================================
    # 19. 모델 및 결과 저장
    # ===================================
    logger.info("\n" + "="*60)
    logger.info("모델 및 결과 저장")
    logger.info("="*60)
    os.makedirs('model', exist_ok=True)
    os.makedirs('scaler', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    for model_name, model in hybrid_models.models.items():
        model_path = f'model/{model_name}_final_hybrid.keras'
        model.save(model_path)
        logger.info(f"{model_name.upper()} 모델 저장: {model_path}")
        
    scaler_path = 'scaler/robust_scaler_hybrid.pkl'
    joblib.dump(sscaler, scaler_path)
    logger.info(f"스케일러 저장: {scaler_path}")
    
    results_df = pd.DataFrame(results).T
    results_df.to_csv('results/model_performance.csv')
    logger.info("성능 결과 저장: results/model_performance.csv")
    
    # ===================================
    # 20. 최종 요약
    # ===================================
    logger.info("\n" + "="*60)
    logger.info("학습 완료 요약")
    logger.info("="*60)
    
    best_model = min(results.items(), key=lambda x: x[1]['mae_original'] if x[0] != 'ensemble' else float('inf'))
    logger.info(f"최고 성능 개별 모델: {best_model[0].upper()}")
    logger.info(f" - MAE: {best_model[1]['mae_original']:.2f}")
    logger.info(f" - R²: {best_model[1]['r2']:.4f}")

    logger.info(f"\n앙상블 모델 성능:")
    logger.info(f" - MAE: {results['ensemble']['mae_original']:.2f}")
    logger.info(f" - R²: {results['ensemble']['r2']:.4f}")

    state = checkpoint_manager.load_state() or {}
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
    parser.add_argument('--resume', action='store_true', help='이전 학습을 이어서 진행')
    parser.add_argument('--reset', action='store_true', help='체크포인트를 삭제하고 처음부터 시작')
    args = parser.parse_args()

    if args.reset:
        import shutil
        if os.path.exists('checkpoints'):
            shutil.rmtree('checkpoints')
        if os.path.exists('model'):
            shutil.rmtree('model')
        if os.path.exists('results'):
            shutil.rmtree('results')
        if os.path.exists('scaler'):
            shutil.rmtree('scaler')
        logger.info("체크포인트와 결과 폴더가 삭제되었습니다. 처음부터 시작합니다.")
    
    main(resume=args.resume and not args.reset)

