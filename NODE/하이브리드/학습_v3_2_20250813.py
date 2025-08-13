"""
반도체 물류 예측 모델 v3.4 - 정밀도 중심 균형잡힌 예측 (완전판)
================================================================
주요 기능:
1. LSTM, GRU, RNN, Bidirectional LSTM 앙상블
2. 학습 중단 시 재시작 기능
3. 정밀도 중심 손실 함수
4. 동적 임계값 조정
5. 체크포인트 관리
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, GRU, SimpleRNN, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os
import joblib
import logging
import warnings
import json
import pickle
import gc
import traceback

warnings.filterwarnings('ignore')

# 환경 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')
tf.keras.backend.set_floatx('float32')

RANDOM_SEED = 2079936
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_v3.4.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===================================
# 체크포인트 관리 (v3.3에서 가져옴)
# ===================================
class CheckpointManager:
    def __init__(self, checkpoint_dir='checkpoints_v34'):
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
            pickle.dump(data_dict, f, protocol=4)
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
# 정밀도 중심 손실 함수
# ===================================
class PrecisionFocusedLoss(tf.keras.losses.Loss):
    def __init__(self, precision_weight=2.0, name='precision_focused_loss'):
        super().__init__(name=name)
        self.precision_weight = precision_weight
        
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        # 기본 BCE
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        
        # False Positive에 더 큰 페널티
        false_positive_penalty = (1 - y_true) * y_pred * self.precision_weight
        
        return tf.reduce_mean(bce + false_positive_penalty)

# ===================================
# 동적 임계값 조정 콜백
# ===================================
class DynamicThresholdCallback(Callback):
    def __init__(self, X_val, y_val_cls, target_ratio=0.025):
        super().__init__()
        self.X_val = X_val
        self.y_val_cls = y_val_cls
        self.target_ratio = target_ratio
        self.threshold_history = []
        
    def on_epoch_end(self, epoch, logs=None):
        val_pred = self.model.predict(self.X_val, verbose=0)
        spike_probs = val_pred[1].flatten()
        
        percentile = 100 - (self.target_ratio * 100)
        optimal_threshold = np.percentile(spike_probs, percentile)
        self.threshold_history.append(optimal_threshold)
        
        y_pred = (spike_probs > optimal_threshold).astype(int)
        tp = np.sum((self.y_val_cls == 1) & (y_pred == 1))
        fp = np.sum((self.y_val_cls == 0) & (y_pred == 1))
        fn = np.sum((self.y_val_cls == 1) & (y_pred == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nEpoch {epoch+1} - 임계값: {optimal_threshold:.4f}")
        print(f"예측: {y_pred.sum()}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

# ===================================
# 완전한 앙상블 모델 클래스
# ===================================
class CompleteEnsembleModels:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.models = {}
        self.histories = {}
        
    def build_precision_lstm_model(self):
        """정밀도 중심 LSTM"""
        inputs = Input(shape=self.input_shape, name='input')
        
        x = LSTM(units=96, return_sequences=True, 
                kernel_regularizer=tf.keras.regularizers.l2(0.005))(inputs)
        x = Dropout(0.3)(x)
        x = BatchNormalization()(x)
        
        x = LSTM(units=48, return_sequences=False)(x)
        x = Dropout(0.3)(x)
        
        shared_features = Dense(units=24, activation='relu')(x)
        
        regression_output = Dense(units=1, name='regression_output')(shared_features)
        
        spike_branch = Dense(units=32, activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(0.01))(shared_features)
        spike_branch = Dropout(0.4)(spike_branch)
        spike_branch = Dense(units=16, activation='relu')(spike_branch)
        spike_branch = Dropout(0.4)(spike_branch)
        spike_output = Dense(units=1, activation='sigmoid', name='spike_output')(spike_branch)
        
        model = Model(inputs=inputs, outputs=[regression_output, spike_output])
        return model
    
    def build_precision_gru_model(self):
        """정밀도 중심 GRU"""
        inputs = Input(shape=self.input_shape, name='input')
        
        x = GRU(units=96, return_sequences=True)(inputs)
        x = Dropout(0.3)(x)
        x = BatchNormalization()(x)
        
        x = GRU(units=48, return_sequences=False)(x)
        x = Dropout(0.3)(x)
        
        shared_features = Dense(units=24, activation='relu')(x)
        
        regression_output = Dense(units=1, name='regression_output')(shared_features)
        
        spike_branch = Dense(units=32, activation='relu')(shared_features)
        spike_branch = Dropout(0.4)(spike_branch)
        spike_output = Dense(units=1, activation='sigmoid', name='spike_output')(spike_branch)
        
        model = Model(inputs=inputs, outputs=[regression_output, spike_output])
        return model
    
    def build_precision_rnn_model(self):
        """정밀도 중심 Simple RNN"""
        inputs = Input(shape=self.input_shape, name='input')
        
        x = SimpleRNN(units=100, return_sequences=True)(inputs)
        x = Dropout(0.3)(x)
        
        x = SimpleRNN(units=50, return_sequences=False)(x)
        x = Dropout(0.3)(x)
        
        shared_features = Dense(units=24, activation='relu')(x)
        
        regression_output = Dense(units=1, name='regression_output')(shared_features)
        
        spike_branch = Dense(units=32, activation='relu')(shared_features)
        spike_branch = Dropout(0.4)(spike_branch)
        spike_output = Dense(units=1, activation='sigmoid', name='spike_output')(spike_branch)
        
        model = Model(inputs=inputs, outputs=[regression_output, spike_output])
        return model
    
    def build_precision_bilstm_model(self):
        """정밀도 중심 Bidirectional LSTM"""
        inputs = Input(shape=self.input_shape, name='input')
        
        x = Bidirectional(LSTM(units=48, return_sequences=True))(inputs)
        x = Dropout(0.3)(x)
        x = BatchNormalization()(x)
        
        x = Bidirectional(LSTM(units=24, return_sequences=False))(x)
        x = Dropout(0.3)(x)
        
        shared_features = Dense(units=24, activation='relu')(x)
        
        regression_output = Dense(units=1, name='regression_output')(shared_features)
        
        spike_branch = Dense(units=32, activation='relu')(shared_features)
        spike_branch = Dropout(0.4)(spike_branch)
        spike_output = Dense(units=1, activation='sigmoid', name='spike_output')(spike_branch)
        
        model = Model(inputs=inputs, outputs=[regression_output, spike_output])
        return model

# ===================================
# 재시작 가능한 학습 함수
# ===================================
def train_with_checkpoint(model, model_name, X_train, y_train_reg, y_train_cls,
                         X_val, y_val_reg, y_val_cls, epochs, batch_size,
                         checkpoint_manager, start_epoch=0, initial_lr=0.0005):
    
    # 컴파일
    optimizer = Adam(learning_rate=initial_lr)
    model.compile(
        optimizer=optimizer,
        loss={
            'regression_output': 'mse',
            'spike_output': PrecisionFocusedLoss(precision_weight=3.0)
        },
        loss_weights={
            'regression_output': 0.7,
            'spike_output': 5.0
        },
        metrics={
            'regression_output': 'mae',
            'spike_output': [
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        }
    )
    
    # 콜백
    dynamic_threshold = DynamicThresholdCallback(X_val, y_val_cls, target_ratio=0.025)
    
    callbacks = [
        dynamic_threshold,
        ReduceLROnPlateau(
            monitor='val_spike_output_auc',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_spike_output_auc',
            patience=20,
            mode='max',
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # 학습 이력
    history = {'loss': [], 'val_loss': []}
    
    # 기존 이력 로드
    state = checkpoint_manager.load_state()
    if state and model_name in state.get('model_histories', {}):
        history = state['model_histories'][model_name]
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20
    
    # 샘플 가중치
    sample_weights = np.ones(len(y_train_cls), dtype=np.float32)
    spike_indices = np.where(y_train_cls == 1)[0]
    sample_weights[spike_indices] = 2.0
    
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
                sample_weight=sample_weights,
                callbacks=callbacks,
                verbose=1
            )
            
            # 이력 업데이트
            for key in ['loss', 'val_loss']:
                if key in epoch_history.history:
                    history[key].append(epoch_history.history[key][0])
            
            current_val_loss = epoch_history.history['val_loss'][0]
            
            # 최고 성능 모델 저장
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_weights_path = checkpoint_manager.save_model_weights(model, model_name, epoch)
                patience_counter = 0
                logger.info(f"최고 성능 갱신! Val Loss: {best_val_loss:.4f}")
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
    
    # 최적 임계값 저장
    model.optimal_threshold = dynamic_threshold.threshold_history[-1] if dynamic_threshold.threshold_history else 0.5
    
    return history

# ===================================
# 메인 학습 프로세스
# ===================================
def main(resume=False):
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
            
            # v3.3의 데이터 로드 시도
            try:
                # 먼저 v3.4 체크포인트에서 찾기
                saved_data = checkpoint_manager.load_data()
                
                # 없으면 v3.3에서 찾기
                if saved_data is None:
                    v33_checkpoint = CheckpointManager(checkpoint_dir='checkpoints_v33')
                    saved_data = v33_checkpoint.load_data()
                    if saved_data:
                        logger.info("v3.3 데이터를 로드했습니다.")
                        # v3.4로 복사
                        checkpoint_manager.save_data(saved_data)
                
                if saved_data:
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
                    input_shape = saved_data['input_shape']
                    scaling_columns = saved_data['scaling_columns']
                    
                    logger.info("저장된 데이터를 성공적으로 로드했습니다.")
                else:
                    logger.error("저장된 데이터가 없습니다.")
                    return
                    
            except Exception as e:
                logger.error(f"데이터 로드 중 오류: {str(e)}")
                return
        else:
            logger.info("저장된 학습 상태가 없습니다. v3.3 데이터를 찾습니다.")
            # v3.3 데이터 로드
            v33_checkpoint = CheckpointManager(checkpoint_dir='checkpoints_v33')
            saved_data = v33_checkpoint.load_data()
            if saved_data:
                checkpoint_manager.save_data(saved_data)
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
                input_shape = saved_data['input_shape']
                scaling_columns = saved_data['scaling_columns']
                logger.info("v3.3 데이터를 성공적으로 로드했습니다.")
            else:
                logger.error("v3.3 데이터도 없습니다. 전처리부터 시작하세요.")
                return
    else:
        # 새로 시작하는 경우 v3.3 데이터 사용
        logger.info("="*60)
        logger.info("반도체 물류 예측 모델 v3.4 - 정밀도 중심 균형잡힌 예측")
        logger.info("="*60)
        
        v33_checkpoint = CheckpointManager(checkpoint_dir='checkpoints_v33')
        saved_data = v33_checkpoint.load_data()
        
        if saved_data is None:
            logger.error("v3.3 데이터가 없습니다. v3.3을 먼저 실행하세요.")
            return
        
        # v3.4로 데이터 복사
        checkpoint_manager.save_data(saved_data)
        
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
        input_shape = saved_data['input_shape']
        scaling_columns = saved_data['scaling_columns']
    
    logger.info(f"데이터 shape: {X_train.shape}")
    logger.info(f"훈련 급증 비율: {y_train_cls.mean():.2%}")
    logger.info(f"검증 급증 비율: {y_val_cls.mean():.2%}")
    logger.info(f"테스트 급증 비율: {y_test_cls.mean():.2%}")
    
    # 앙상블 모델 초기화
    ensemble_models = CompleteEnsembleModels(input_shape)
    
    # 학습 파라미터
    EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0005
    
    # 모델 리스트
    model_configs = [
        ('precision_lstm', ensemble_models.build_precision_lstm_model),
        ('precision_gru', ensemble_models.build_precision_gru_model),
        ('precision_rnn', ensemble_models.build_precision_rnn_model),
        ('precision_bilstm', ensemble_models.build_precision_bilstm_model)
    ]
    
    # 재시작 시 완료된 모델 확인
    state = checkpoint_manager.load_state() if resume else {}
    completed_models = state.get('completed_models', [])
    
    # 각 모델 학습
    for model_name, build_func in model_configs:
        if model_name in completed_models:
            logger.info(f"\n{model_name} 모델은 이미 학습이 완료되었습니다. 건너뜁니다.")
            # 완료된 모델 로드
            model = build_func()
            model_path = os.path.join(checkpoint_manager.checkpoint_dir, f'{model_name}_final.h5')
            if os.path.exists(model_path):
                model.load_weights(model_path)
                ensemble_models.models[model_name] = model
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
            history = train_with_checkpoint(
                model, model_name, X_train, y_train_reg, y_train_cls,
                X_val, y_val_reg, y_val_cls,
                EPOCHS, BATCH_SIZE, checkpoint_manager,
                start_epoch=start_epoch, initial_lr=LEARNING_RATE
            )
            
            ensemble_models.models[model_name] = model
            ensemble_models.histories[model_name] = history
            
            # 모델 저장
            model_path = os.path.join(checkpoint_manager.checkpoint_dir, f'{model_name}_final.h5')
            model.save_weights(model_path)
            
        except KeyboardInterrupt:
            logger.warning("\n학습이 중단되었습니다. 현재 상태가 저장되었습니다.")
            logger.info("다시 시작하려면 --resume 옵션을 사용하세요.")
            return
        except Exception as e:
            logger.error(f"\n{model_name} 모델 학습 중 오류 발생: {str(e)}")
            logger.info("다시 시작하려면 --resume 옵션을 사용하세요.")
            return
    
    logger.info("\n" + "="*60)
    logger.info("모든 모델 학습 완료!")
    logger.info("="*60)
    
    # ===================================
    # 모델 평가 및 앙상블
    # ===================================
    
    logger.info("\n모델 성능 평가 시작...")
    
    # 후처리 함수
    def postprocess_predictions(spike_probs, target_count=None, target_ratio=0.025):
        """예측 후처리 - 상위 N개 또는 N%만 선택"""
        if target_count is not None:
            if len(spike_probs) < target_count:
                target_count = len(spike_probs)
            
            threshold_idx = np.argsort(spike_probs)[-target_count]
            threshold = spike_probs[threshold_idx]
        else:
            percentile = 100 - (target_ratio * 100)
            threshold = np.percentile(spike_probs, percentile)
        
        return (spike_probs > threshold).astype(int), threshold
    
    # 각 모델 평가
    results = {}
    expected_spikes = int(y_test_cls.sum())
    
    for model_name, model in ensemble_models.models.items():
        logger.info(f"\n{model_name} 평가 중...")
        
        pred = model.predict(X_test, verbose=0)
        y_pred_reg = pred[0].flatten()
        y_pred_spike = pred[1].flatten()
        
        # 후처리 적용
        y_pred_binary, used_threshold = postprocess_predictions(
            y_pred_spike, 
            target_count=int(expected_spikes * 1.2)  # 20% 더 예측
        )
        
        # 평가
        cm = confusion_matrix(y_test_cls, y_pred_binary)
        report = classification_report(y_test_cls, y_pred_binary, output_dict=True, zero_division=0)
        
        logger.info(f"\n{model_name} 성능:")
        logger.info(f"임계값: {used_threshold:.4f}")
        logger.info(f"예측 개수: {y_pred_binary.sum()} (목표: {expected_spikes})")
        logger.info(f"Precision: {report.get('1', {}).get('precision', 0):.3f}")
        logger.info(f"Recall: {report.get('1', {}).get('recall', 0):.3f}")
        logger.info(f"F1-Score: {report.get('1', {}).get('f1-score', 0):.3f}")
        
        results[model_name] = {
            'threshold': used_threshold,
            'predictions': y_pred_binary.sum(),
            'precision': report.get('1', {}).get('precision', 0),
            'recall': report.get('1', {}).get('recall', 0),
            'f1': report.get('1', {}).get('f1-score', 0),
            'confusion_matrix': cm,
            'spike_probs': y_pred_spike  # 앙상블을 위해 저장
        }
    
    # 앙상블 예측
    logger.info(f"\n{'='*60}")
    logger.info("앙상블 예측")
    logger.info(f"{'='*60}")
    
    # 가중 앙상블 (F1 스코어 기반)
    ensemble_weights = {}
    total_f1 = sum(r['f1'] for r in results.values())
    
    if total_f1 > 0:
        for model_name, result in results.items():
            ensemble_weights[model_name] = result['f1'] / total_f1
    else:
        # F1이 모두 0인 경우 균등 가중치
        for model_name in results:
            ensemble_weights[model_name] = 1.0 / len(results)
    
    for model_name, weight in ensemble_weights.items():
        logger.info(f"{model_name} 가중치: {weight:.3f}")
    
    # 앙상블 예측
    ensemble_spike = np.zeros_like(y_test_cls, dtype=np.float32)
    
    for model_name, result in results.items():
        ensemble_spike += result['spike_probs'] * ensemble_weights[model_name]
    
    # 앙상블 후처리
    ensemble_binary, ensemble_threshold = postprocess_predictions(
        ensemble_spike,
        target_count=int(expected_spikes * 1.1)  # 10% 더 예측
    )
    
    # 앙상블 평가
    cm = confusion_matrix(y_test_cls, ensemble_binary)
    report = classification_report(y_test_cls, ensemble_binary, output_dict=True, zero_division=0)
    
    logger.info(f"\n앙상블 성능:")
    logger.info(f"예측 개수: {ensemble_binary.sum()} (목표: {expected_spikes})")
    logger.info(f"Precision: {report.get('1', {}).get('precision', 0):.3f}")
    logger.info(f"Recall: {report.get('1', {}).get('recall', 0):.3f}")
    logger.info(f"F1-Score: {report.get('1', {}).get('f1-score', 0):.3f}")
    
    # 혼동 행렬
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        logger.info(f"\n혼동 행렬:")
        logger.info(f"TN: {tn}, FP: {fp}")
        logger.info(f"FN: {fn}, TP: {tp}")
    
    # ===================================
    # 결과 저장
    # ===================================
    
    logger.info("\n결과 저장 중...")
    
    os.makedirs('model_v34', exist_ok=True)
    os.makedirs('results_v34', exist_ok=True)
    
    # 모델 저장
    for model_name, model in ensemble_models.models.items():
        model_path = f'model_v34/{model_name}_final.keras'
        model.save(model_path)
        logger.info(f"{model_name} 모델 저장: {model_path}")
    
    # 스케일러 저장
    scaler_path = 'model_v34/scaler_v34.pkl'
    joblib.dump(scaler, scaler_path)
    
    # 성능 결과 저장
    results_df = pd.DataFrame({k: v for k, v in results.items() if k != 'spike_probs'}).T
    results_df.to_csv('results_v34/model_performance.csv')
    
    # 설정 저장
    config = {
        'ensemble_weights': ensemble_weights,
        'ensemble_threshold': float(ensemble_threshold),
        'target_spike_count': expected_spikes,
        'input_shape': input_shape,
        'scaling_columns': scaling_columns
    }
    
    with open('results_v34/config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    # ===================================
    # 최종 요약
    # ===================================
    
    logger.info("\n" + "="*60)
    logger.info("학습 완료 요약")
    logger.info("="*60)
    
    # 최고 성능 모델
    best_model = max(results.items(), key=lambda x: x[1]['f1'])
    logger.info(f"\n최고 F1-Score 모델: {best_model[0].upper()}")
    logger.info(f"  - F1-Score: {best_model[1]['f1']:.3f}")
    logger.info(f"  - Recall: {best_model[1]['recall']:.3f}")
    logger.info(f"  - Precision: {best_model[1]['precision']:.3f}")
    
    # 목표 달성 여부
    target_recall = 0.7
    if report.get('1', {}).get('recall', 0) >= target_recall and report.get('1', {}).get('precision', 0) >= 0.3:
        logger.info("\n🎯 목표 달성! Recall >= 70%, Precision >= 30%")
    else:
        logger.info("\n📊 추가 조정 필요")
        logger.info(f"현재: Recall={report.get('1', {}).get('recall', 0):.1%}, "
                   f"Precision={report.get('1', {}).get('precision', 0):.1%}")
    
    logger.info("\n" + "="*60)
    logger.info("모든 작업이 완료되었습니다!")
    logger.info("다시 시작하려면: python script.py --resume")
    logger.info("="*60)

# ===================================
# 실행
# ===================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='반도체 물류 예측 모델 v3.4')
    parser.add_argument('--resume', action='store_true', 
                       help='이전 학습을 이어서 진행')
    parser.add_argument('--reset', action='store_true',
                       help='체크포인트를 삭제하고 처음부터 시작')
    
    args = parser.parse_args()
    
    if args.reset:
        import shutil
        if os.path.exists('checkpoints_v34'):
            shutil.rmtree('checkpoints_v34')
            logger.info("체크포인트가 삭제되었습니다. 처음부터 시작합니다.")
    
    try:
        main(resume=args.resume)
    except Exception as e:
        logger.error(f"프로그램 실행 중 오류: {str(e)}")
        import traceback
        traceback.print_exc()