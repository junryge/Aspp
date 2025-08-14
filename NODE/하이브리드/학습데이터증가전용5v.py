"""
반도체 물류 예측 ULTIMATE v5.0 - 1400+ 예측 강화 완전판
========================================================
모든 기능 포함 + 재시작 가능 + 1400+ 예측 개선

핵심 개선:
1. 가중 손실 함수 (1400+에 10배 가중치)
2. 데이터 증강 (1400+ 샘플 3배 증가)
3. 강화된 급변 감지기
4. 동적 앙상블 (최대 20% 부스팅)
5. 특성은 v4와 100% 동일 (캐시 호환)

사용법:
python model_v5_ultimate.py          # 처음 시작
python model_v5_ultimate.py --resume # 이어서 시작
python model_v5_ultimate.py --reset  # 초기화 후 시작
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, RobustScaler
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (Input, LSTM, Dense, Dropout, BatchNormalization,
                                     GRU, Conv1D, MaxPooling1D, GlobalAveragePooling1D,
                                     Bidirectional)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras import backend as K
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
import joblib
import logging
import warnings
import json
import pickle
import traceback
import argparse
import shutil
import time

# 경고 숨기기
warnings.filterwarnings('ignore')

# ===================================
# 1. 환경 설정
# ===================================

# CPU 모드 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

# 랜덤 시드
RANDOM_SEED = 2079936
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_v5.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===================================
# 2. 가중 손실 함수 (1400+ 강화)
# ===================================

def create_weighted_mse(spike_threshold=0.5):
    """1400+ 값에 10배 가중치를 주는 손실 함수"""
    def weighted_mse(y_true, y_pred):
        # 높은 값에 대한 가중치 (스케일된 값 기준)
        weights = tf.where(
            y_true > spike_threshold,
            10.0,  # 1400+ 값에 10배 가중치
            1.0
        )
        
        # 가중 MSE 계산
        squared_diff = tf.square(y_true - y_pred)
        weighted_loss = squared_diff * weights
        
        return tf.reduce_mean(weighted_loss)
    
    return weighted_mse

# ===================================
# 3. 체크포인트 관리자
# ===================================

class UltimateCheckpointManager:
    """완벽한 체크포인트 관리"""
    
    def __init__(self, checkpoint_dir='checkpoints_v5'):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 파일 경로들
        self.state_file = os.path.join(checkpoint_dir, 'training_state.json')
        self.data_file = os.path.join(checkpoint_dir, 'preprocessed_data.pkl')
        self.models_dir = os.path.join(checkpoint_dir, 'models')
        os.makedirs(self.models_dir, exist_ok=True)
        
    def save_state(self, state_dict):
        """학습 상태 저장"""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                existing = json.load(f)
        else:
            existing = {}
        
        existing.update(state_dict)
        
        with open(self.state_file, 'w') as f:
            json.dump(existing, f, indent=4, default=str)
            
    def load_state(self):
        """학습 상태 로드"""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return None
    
    def save_data(self, data_dict):
        """전처리된 데이터 저장"""
        with open(self.data_file, 'wb') as f:
            pickle.dump(data_dict, f)
        logger.info(f"💾 데이터 저장: {self.data_file}")
        
    def load_data(self):
        """전처리된 데이터 로드"""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"📂 데이터 로드: {self.data_file}")
            return data
        return None
    
    def save_model_weights(self, model, model_name, epoch):
        """모델 가중치 저장"""
        path = os.path.join(self.models_dir, f'{model_name}_epoch_{epoch}.h5')
        model.save_weights(path)
        return path
    
    def load_model_weights(self, model, model_name, epoch):
        """모델 가중치 로드"""
        path = os.path.join(self.models_dir, f'{model_name}_epoch_{epoch}.h5')
        if os.path.exists(path):
            model.load_weights(path)
            logger.info(f"✅ 가중치 로드: {path}")
            return True
        return False
    
    def get_latest_epoch(self, model_name):
        """최신 에폭 찾기"""
        state = self.load_state()
        if state and 'model_progress' in state:
            return state['model_progress'].get(model_name, {}).get('last_epoch', 0)
        return 0

# ===================================
# 4. 커스텀 콜백 (진행상황 저장)
# ===================================

class CheckpointCallback(Callback):
    """매 에폭마다 상태 저장"""
    
    def __init__(self, checkpoint_manager, model_name):
        super().__init__()
        self.checkpoint_manager = checkpoint_manager
        self.model_name = model_name
        self.history = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}
        
    def on_epoch_end(self, epoch, logs=None):
        """에폭 종료 시 저장"""
        # 히스토리 업데이트
        for key in self.history.keys():
            if key in logs:
                self.history[key].append(logs[key])
        
        # 5 에폭마다 가중치 저장
        if (epoch + 1) % 5 == 0:
            self.checkpoint_manager.save_model_weights(self.model, self.model_name, epoch + 1)
            
        # 상태 저장
        state = {
            'current_model': self.model_name,
            'model_progress': {
                self.model_name: {
                    'last_epoch': epoch + 1,
                    'history': self.history,
                    'best_val_loss': min(self.history['val_loss']) if self.history['val_loss'] else 999
                }
            },
            'last_update': datetime.now().isoformat()
        }
        self.checkpoint_manager.save_state(state)
        
        # 로그 출력
        logger.info(f"[{self.model_name}] Epoch {epoch+1} - "
                   f"Loss: {logs.get('loss', 0):.4f}, "
                   f"Val Loss: {logs.get('val_loss', 0):.4f}, "
                   f"Val MAE: {logs.get('val_mae', 0):.4f}")

# ===================================
# 5. 데이터 증강 함수 (1400+ 강화)
# ===================================

def augment_high_value_data(X_train, y_train, spike_train):
    """1400+ 샘플을 증강하여 불균형 해결"""
    
    logger.info("🔧 1400+ 데이터 증강 시작...")
    
    # 1400+ 인덱스 찾기
    high_indices = np.where(spike_train == 1)[0]
    normal_indices = np.where(spike_train == 0)[0]
    
    logger.info(f"   원본 1400+ 샘플: {len(high_indices)}개")
    logger.info(f"   원본 일반 샘플: {len(normal_indices)}개")
    
    if len(high_indices) == 0:
        return X_train, y_train, spike_train
    
    # 증강 데이터 리스트
    augmented_X = []
    augmented_y = []
    augmented_spike = []
    
    # 1400+ 샘플 3배 증강
    for idx in high_indices:
        # 원본
        augmented_X.append(X_train[idx])
        augmented_y.append(y_train[idx])
        augmented_spike.append(1)
        
        # 노이즈 추가 버전 (3개)
        for i in range(3):
            noise_level = 0.01 * (i + 1)  # 점진적 노이즈
            noise = np.random.normal(0, noise_level, X_train[idx].shape)
            augmented_sample = X_train[idx] + noise
            
            augmented_X.append(augmented_sample)
            augmented_y.append(y_train[idx] * (1 + np.random.uniform(-0.02, 0.02)))  # 타겟도 약간 변동
            augmented_spike.append(1)
    
    # 원본 데이터와 결합
    X_combined = np.concatenate([X_train, np.array(augmented_X)])
    y_combined = np.concatenate([y_train, np.array(augmented_y)])
    spike_combined = np.concatenate([spike_train, np.array(augmented_spike)])
    
    # 셔플
    indices = np.random.permutation(len(X_combined))
    X_augmented = X_combined[indices]
    y_augmented = y_combined[indices]
    spike_augmented = spike_combined[indices]
    
    logger.info(f"   ✅ 증강 완료! 총 {len(X_augmented)}개 샘플")
    logger.info(f"   ✅ 1400+ 비율: {spike_augmented.mean():.2%}")
    
    return X_augmented, y_augmented, spike_augmented

# ===================================
# 6. 데이터 전처리 (v4와 동일)
# ===================================

def load_and_preprocess_data(data_path, checkpoint_manager, force_reload=False):
    """데이터 로드 및 전처리 (v4와 100% 동일)"""
    
    # 캐시 확인
    if not force_reload:
        cached_data = checkpoint_manager.load_data()
        if cached_data:
            logger.info("✅ 캐시된 데이터 사용")
            return cached_data
    
    logger.info(f"📂 데이터 새로 로딩: {data_path}")
    
    try:
        # 1. 데이터 로드
        logger.info("📊 [1/8] CSV 파일 로딩 중...")
        data = pd.read_csv(data_path)
        logger.info(f"   ✓ 원본 데이터 크기: {data.shape}")
        
        # 2. 시간 변환
        logger.info("🕒 [2/8] 시간 데이터 변환 중...")
        data['CURRTIME'] = pd.to_datetime(data['CURRTIME'], format='%Y%m%d%H%M')
        data['TIME'] = pd.to_datetime(data['TIME'], format='%Y%m%d%H%M')
        
        data = data[['CURRTIME', 'TOTALCNT', 'TIME']]
        data.set_index('CURRTIME', inplace=True)
        
        # 3. 날짜 필터링
        logger.info("📅 [3/8] 날짜 범위 필터링 중...")
        start_date = pd.to_datetime('2024-02-01 00:00:00')
        end_date = pd.to_datetime('2024-07-27 23:59:59')
        data = data[(data['TIME'] >= start_date) & (data['TIME'] <= end_date)]
        logger.info(f"   ✓ 필터링 후 데이터: {data.shape}")
        
        # 4. 이상치 제거
        logger.info("🔍 [4/8] 이상치 제거 중...")
        before_outlier = len(data)
        data = data[(data['TOTALCNT'] >= 800) & (data['TOTALCNT'] <= 2500)]
        logger.info(f"   ✓ 제거된 이상치: {before_outlier - len(data)}개")
        
        # 5. FUTURE 및 라벨 생성
        logger.info("🎯 [5/8] 타겟 변수 생성 중...")
        data['FUTURE'] = data['TOTALCNT'].shift(-10)
        data['spike_label'] = (data['FUTURE'] >= 1400).astype(int)
        data.dropna(inplace=True)
        
        logger.info(f"   ✓ 전처리 완료: {data.shape}")
        logger.info(f"   ✓ 1400+ 비율: {data['spike_label'].mean():.2%}")
        logger.info(f"   ✓ 1400+ 개수: {data['spike_label'].sum()}개")
        
        # 6. 특징 생성
        logger.info("⚙️ [6/8] 특징 엔지니어링 중...")
        
        logger.info("   이동평균 계산...")
        data['MA_10'] = data['TOTALCNT'].rolling(10, min_periods=1).mean()
        data['MA_30'] = data['TOTALCNT'].rolling(30, min_periods=1).mean()
        data['MA_60'] = data['TOTALCNT'].rolling(60, min_periods=1).mean()
        
        logger.info("   표준편차 계산...")
        data['STD_10'] = data['TOTALCNT'].rolling(10, min_periods=1).std()
        data['STD_30'] = data['TOTALCNT'].rolling(30, min_periods=1).std()
        
        logger.info("   변화율 계산...")
        data['change_rate'] = data['TOTALCNT'].pct_change()
        data['change_rate_10'] = data['TOTALCNT'].pct_change(10)
        
        logger.info("   시간 특징 추출...")
        data['hour'] = data.index.hour
        data['dayofweek'] = data.index.dayofweek
        data['is_weekend'] = (data.index.dayofweek >= 5).astype(int)
        data['trend'] = data['MA_10'] - data['MA_30']
        
        data.fillna(method='ffill', inplace=True)
        data.fillna(0, inplace=True)
        
        # 7. 스케일링
        logger.info("📏 [7/8] 데이터 스케일링 중...")
        scaler = RobustScaler()
        feature_cols = ['TOTALCNT', 'MA_10', 'MA_30', 'MA_60', 'STD_10', 'STD_30',
                       'change_rate', 'change_rate_10', 'hour', 'dayofweek', 
                       'is_weekend', 'trend']
        
        data[feature_cols + ['FUTURE']] = scaler.fit_transform(data[feature_cols + ['FUTURE']])
        
        # 8. 시퀀스 생성
        logger.info("🔄 [8/8] 시퀀스 데이터 생성 중...")
        SEQ_LENGTH = 50
        X, y, spike_labels = [], [], []
        
        total_sequences = len(data) - SEQ_LENGTH
        logger.info(f"   총 {total_sequences}개 시퀀스 생성 중...")
        
        for i in range(total_sequences):
            if i % 5000 == 0:
                logger.info(f"   진행률: {(i/total_sequences)*100:.1f}%")
            
            X.append(data[feature_cols].iloc[i:i+SEQ_LENGTH].values)
            y.append(data['FUTURE'].iloc[i+SEQ_LENGTH])
            spike_labels.append(data['spike_label'].iloc[i+SEQ_LENGTH])
        
        X, y, spike_labels = np.array(X), np.array(y), np.array(spike_labels)
        
        # 데이터 분할
        logger.info("📊 데이터셋 분할 중...")
        train_size = int(0.7 * len(X))
        val_size = int(0.15 * len(X))
        
        data_dict = {
            'X_train': X[:train_size],
            'y_train': y[:train_size],
            'spike_train': spike_labels[:train_size],
            'X_val': X[train_size:train_size+val_size],
            'y_val': y[train_size:train_size+val_size],
            'spike_val': spike_labels[train_size:train_size+val_size],
            'X_test': X[train_size+val_size:],
            'y_test': y[train_size+val_size:],
            'spike_test': spike_labels[train_size+val_size:],
            'scaler': scaler,
            'feature_cols': feature_cols,
            'input_shape': (SEQ_LENGTH, len(feature_cols))
        }
        
        # 캐시 저장
        checkpoint_manager.save_data(data_dict)
        
        logger.info("✅ 데이터 전처리 완료!")
        return data_dict
        
    except Exception as e:
        logger.error(f"❌ 데이터 전처리 중 오류: {str(e)}")
        raise

# ===================================
# 7. 모델 정의 (개선된 버전)
# ===================================

def build_improved_lstm(input_shape):
    """개선된 LSTM (1400+ 예측 강화)"""
    model = Sequential([
        Input(shape=input_shape),
        LSTM(128, return_sequences=True, kernel_regularizer=l1_l2(l1=0.005, l2=0.005)),
        Dropout(0.4),
        BatchNormalization(),
        LSTM(64, return_sequences=True),
        Dropout(0.4),
        LSTM(32, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    return model

def build_improved_gru(input_shape):
    """개선된 GRU"""
    model = Sequential([
        Input(shape=input_shape),
        GRU(128, return_sequences=True, kernel_regularizer=l1_l2(l1=0.005, l2=0.005)),
        Dropout(0.4),
        GRU(64, return_sequences=True),
        Dropout(0.4),
        GRU(32, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    return model

def build_improved_cnn_lstm(input_shape):
    """개선된 CNN-LSTM"""
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(64, 3, activation='relu', padding='same'),
        BatchNormalization(),
        Conv1D(64, 3, activation='relu', padding='same'),
        MaxPooling1D(2),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        Dropout(0.4),
        LSTM(32, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    return model

def build_improved_spike_detector(input_shape):
    """강화된 급변 감지기"""
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(64, 3, activation='relu', padding='same'),
        BatchNormalization(),
        Conv1D(64, 3, activation='relu', padding='same'),
        MaxPooling1D(2),
        Dropout(0.3),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.4),
        Bidirectional(LSTM(32, return_sequences=False)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    return model

# ===================================
# 8. 학습 함수 (가중 손실 적용)
# ===================================

def train_model_with_resume(model, model_name, data_dict, checkpoint_manager,
                           epochs=50, batch_size=128, resume=False, use_augmentation=True):
    """재시작 가능한 학습 (가중 손실 함수 사용)"""
    
    # 데이터 언팩
    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    spike_train = data_dict['spike_train']
    X_val = data_dict['X_val']
    y_val = data_dict['y_val']
    
    # 급변 감지기인 경우
    if 'spike' in model_name:
        y_train = spike_train
        y_val = data_dict['spike_val']
        
        # 급변 감지기는 데이터 증강
        if use_augmentation:
            X_train, y_train, _ = augment_high_value_data(X_train, data_dict['y_train'], spike_train)
    else:
        # 일반 모델도 데이터 증강 (선택사항)
        if use_augmentation:
            X_train, y_train, spike_train = augment_high_value_data(X_train, y_train, spike_train)
    
    # 재시작 처리
    start_epoch = 0
    if resume:
        start_epoch = checkpoint_manager.get_latest_epoch(model_name)
        if start_epoch > 0:
            logger.info(f"📂 {model_name} Epoch {start_epoch}부터 재시작")
            checkpoint_manager.load_model_weights(model, model_name, start_epoch)
    
    # 이미 완료된 경우
    if start_epoch >= epochs:
        logger.info(f"✅ {model_name} 이미 완료됨")
        return model
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🚀 {model_name} 학습 시작 (Epoch {start_epoch+1}/{epochs})")
    logger.info(f"{'='*60}")
    
    # 컴파일 (가중 손실 함수 사용)
    if 'spike' in model_name:
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    else:
        # 가중 MSE 사용!
        weighted_loss = create_weighted_mse(spike_threshold=0.5)
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss=weighted_loss,  # 가중 손실 함수
            metrics=['mae']
        )
    
    # 콜백
    callbacks = [
        CheckpointCallback(checkpoint_manager, model_name),
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7, verbose=1),
        ModelCheckpoint(f'{checkpoint_manager.models_dir}/{model_name}_best.h5', 
                       save_best_only=True, verbose=0)
    ]
    
    # 학습
    try:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            initial_epoch=start_epoch,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # 완료 상태 저장
        state = {
            'model_progress': {
                model_name: {
                    'completed': True,
                    'final_epoch': epochs,
                    'completed_time': datetime.now().isoformat()
                }
            }
        }
        checkpoint_manager.save_state(state)
        
    except KeyboardInterrupt:
        logger.warning(f"\n⚠️ {model_name} 학습 중단됨. 재시작 가능!")
        state = {
            'interrupted': True,
            'interrupted_model': model_name,
            'interrupted_time': datetime.now().isoformat()
        }
        checkpoint_manager.save_state(state)
        raise
        
    except Exception as e:
        logger.error(f"❌ {model_name} 학습 중 오류: {str(e)}")
        raise
    
    return model

# ===================================
# 9. 강화된 앙상블 예측
# ===================================

def enhanced_ensemble_predict(models, spike_detector, X_test):
    """1400+ 예측 강화된 앙상블"""
    
    logger.info("🔮 앙상블 예측 시작...")
    
    # 급변 확률
    spike_probs = spike_detector.predict(X_test, verbose=0).flatten()
    
    # 각 모델 예측
    predictions = {}
    for name, model in models.items():
        predictions[name] = model.predict(X_test, verbose=0).flatten()
    
    # 동적 가중치 앙상블
    ensemble_pred = np.zeros(len(X_test))
    
    for i in range(len(X_test)):
        # 급변 확률에 따른 가중치
        if spike_probs[i] > 0.7:  # 높은 확신도
            weights = {'lstm': 0.15, 'gru': 0.15, 'cnn_lstm': 0.7}
            boost_factor = 1.20  # 20% 증폭
        elif spike_probs[i] > 0.5:  # 중간 확신도
            weights = {'lstm': 0.2, 'gru': 0.2, 'cnn_lstm': 0.6}
            boost_factor = 1.15  # 15% 증폭
        elif spike_probs[i] > 0.3:  # 약한 신호
            weights = {'lstm': 0.3, 'gru': 0.3, 'cnn_lstm': 0.4}
            boost_factor = 1.08  # 8% 증폭
        else:  # 정상 범위
            weights = {'lstm': 0.4, 'gru': 0.35, 'cnn_lstm': 0.25}
            boost_factor = 1.0
        
        # 가중 평균
        for name, weight in weights.items():
            if name in predictions:
                ensemble_pred[i] += weight * predictions[name][i]
        
        # 부스팅 적용
        ensemble_pred[i] *= boost_factor
    
    logger.info(f"   ✅ 1400+ 예상 개수: {np.sum(spike_probs > 0.5)}")
    
    return ensemble_pred, predictions, spike_probs

# ===================================
# 10. 메인 실행
# ===================================

def main(resume=False, reset=False):
    """메인 실행"""
    
    # 체크포인트 매니저
    checkpoint_manager = UltimateCheckpointManager()
    
    # 리셋 처리
    if reset:
        if os.path.exists(checkpoint_manager.checkpoint_dir):
            shutil.rmtree(checkpoint_manager.checkpoint_dir)
            logger.info("🔄 체크포인트 초기화됨")
        checkpoint_manager = UltimateCheckpointManager()
    
    # 재시작 상태 확인
    if resume:
        state = checkpoint_manager.load_state()
        if state:
            logger.info("="*60)
            logger.info("📂 이전 학습 재개")
            if 'interrupted_model' in state:
                logger.info(f"   중단된 모델: {state['interrupted_model']}")
            logger.info("="*60)
        else:
            logger.info("⚠️ 저장된 상태 없음. 처음부터 시작")
            resume = False
    
    logger.info("="*60)
    logger.info("🚀 반도체 물류 예측 v5.0 - 1400+ 강화")
    logger.info("="*60)
    
    # 데이터 로드
    data_dict = load_and_preprocess_data(
        'data/20240201_TO_202507281705.csv',
        checkpoint_manager,
        force_reload=not resume
    )
    
    input_shape = data_dict['input_shape']
    logger.info(f"📊 입력 Shape: {input_shape}")
    
    # 모델 정의
    model_configs = [
        ('lstm', build_improved_lstm),
        ('gru', build_improved_gru),
        ('cnn_lstm', build_improved_cnn_lstm),
        ('spike_detector', build_improved_spike_detector)
    ]
    
    models = {}
    spike_model = None
    
    # 각 모델 학습
    for model_name, build_func in model_configs:
        try:
            # 완료 확인
            state = checkpoint_manager.load_state()
            if state and 'model_progress' in state:
                if state['model_progress'].get(model_name, {}).get('completed', False):
                    logger.info(f"✅ {model_name} 이미 완료됨. 건너뜀")
                    # 모델 로드
                    model = build_func(input_shape)
                    model.load_weights(f'{checkpoint_manager.models_dir}/{model_name}_best.h5')
                    if model_name != 'spike_detector':
                        models[model_name] = model
                    else:
                        spike_model = model
                    continue
            
            # 모델 빌드
            model = build_func(input_shape)
            logger.info(f"🔨 {model_name} 모델 생성 완료")
            
            # 학습
            epochs = 40 if model_name == 'spike_detector' else 60
            model = train_model_with_resume(
                model, model_name, data_dict, checkpoint_manager,
                epochs=epochs, batch_size=128, resume=resume,
                use_augmentation=True  # 데이터 증강 사용
            )
            
            if model_name != 'spike_detector':
                models[model_name] = model
            else:
                spike_model = model
                
        except KeyboardInterrupt:
            logger.warning("\n⚠️ 학습 중단. python model_v5_ultimate.py --resume 로 재시작 가능")
            return
        except Exception as e:
            logger.error(f"❌ 오류: {str(e)}")
            logger.info("python model_v5_ultimate.py --resume 로 재시작 시도")
            return
    
    # ===================================
    # 평가
    # ===================================
    
    logger.info("\n" + "="*60)
    logger.info("📊 모델 평가")
    logger.info("="*60)
    
    # 테스트 데이터
    X_test = data_dict['X_test']
    y_test = data_dict['y_test']
    spike_test = data_dict['spike_test']
    scaler = data_dict['scaler']
    feature_cols = data_dict['feature_cols']
    
    # 앙상블 예측
    ensemble_pred, individual_preds, spike_probs = enhanced_ensemble_predict(
        models, spike_model, X_test
    )
    
    # 역변환
    y_test_original = scaler.inverse_transform(
        np.column_stack([np.zeros((len(y_test), len(feature_cols))), y_test])
    )[:, -1]
    
    ensemble_original = scaler.inverse_transform(
        np.column_stack([np.zeros((len(ensemble_pred), len(feature_cols))), ensemble_pred])
    )[:, -1]
    
    # 전체 성능
    mae = mean_absolute_error(y_test_original, ensemble_original)
    rmse = np.sqrt(mean_squared_error(y_test_original, ensemble_original))
    r2 = r2_score(y_test_original, ensemble_original)
    
    logger.info(f"📈 전체 성능:")
    logger.info(f"   MAE: {mae:.2f}")
    logger.info(f"   RMSE: {rmse:.2f}")
    logger.info(f"   R²: {r2:.4f}")
    
    # 1400+ 성능
    high_mask = y_test_original >= 1400
    if high_mask.sum() > 0:
        mae_high = mean_absolute_error(y_test_original[high_mask], ensemble_original[high_mask])
        rmse_high = np.sqrt(mean_squared_error(y_test_original[high_mask], ensemble_original[high_mask]))
        
        logger.info(f"\n🎯 1400+ 성능:")
        logger.info(f"   개수: {high_mask.sum()}개")
        logger.info(f"   MAE: {mae_high:.2f}")
        logger.info(f"   RMSE: {rmse_high:.2f}")
        
        # 예측 성공률
        pred_high = ensemble_original >= 1400
        precision = np.sum((pred_high) & (high_mask)) / np.sum(pred_high) if np.sum(pred_high) > 0 else 0
        recall = np.sum((pred_high) & (high_mask)) / np.sum(high_mask)
        
        logger.info(f"   Precision: {precision:.2%}")
        logger.info(f"   Recall: {recall:.2%}")
    
    # 급변 감지 정확도
    spike_acc = accuracy_score(spike_test, spike_probs > 0.5)
    logger.info(f"\n🔍 급변 감지 정확도: {spike_acc:.2%}")
    
    # ===================================
    # 시각화
    # ===================================
    
    logger.info("\n📊 결과 시각화 생성 중...")
    
    # 1. 예측 결과
    plt.figure(figsize=(20, 10))
    
    # 상단: 전체 예측
    plt.subplot(2, 1, 1)
    plt.plot(y_test_original[:300], label='실제값', color='blue', linewidth=2)
    plt.plot(ensemble_original[:300], label='예측값', color='red', alpha=0.7, linewidth=1.5)
    plt.axhline(y=1400, color='green', linestyle='--', alpha=0.5, label='1400 임계값')
    plt.title(f'예측 결과 (전체 MAE: {mae:.2f}, 1400+ MAE: {mae_high:.2f})', fontsize=14)
    plt.ylabel('물류량', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 하단: 1400+ 구간 집중
    plt.subplot(2, 1, 2)
    high_points = np.where(y_test_original[:300] >= 1400)[0]
    plt.scatter(high_points, y_test_original[high_points], color='blue', s=50, label='실제 1400+', zorder=5)
    plt.scatter(high_points, ensemble_original[high_points], color='red', s=30, label='예측 1400+', zorder=4)
    plt.plot(y_test_original[:300], color='blue', alpha=0.3, linewidth=1)
    plt.plot(ensemble_original[:300], color='red', alpha=0.3, linewidth=1)
    plt.axhline(y=1400, color='green', linestyle='--', alpha=0.5)
    plt.title('1400+ 구간 예측 성능', fontsize=14)
    plt.xlabel('시간 (분)', fontsize=12)
    plt.ylabel('물류량', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results_v5.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. 오차 분포
    plt.figure(figsize=(15, 5))
    
    # 전체 오차
    plt.subplot(1, 3, 1)
    errors = y_test_original - ensemble_original
    plt.hist(errors, bins=50, color='blue', alpha=0.7, edgecolor='black')
    plt.title(f'전체 오차 분포\n평균: {np.mean(errors):.2f}, 표준편차: {np.std(errors):.2f}')
    plt.xlabel('오차')
    plt.ylabel('빈도')
    
    # 1400+ 오차
    plt.subplot(1, 3, 2)
    if high_mask.sum() > 0:
        high_errors = y_test_original[high_mask] - ensemble_original[high_mask]
        plt.hist(high_errors, bins=30, color='red', alpha=0.7, edgecolor='black')
        plt.title(f'1400+ 오차 분포\n평균: {np.mean(high_errors):.2f}')
        plt.xlabel('오차')
        plt.ylabel('빈도')
    
    # 급변 감지 확률
    plt.subplot(1, 3, 3)
    plt.hist(spike_probs, bins=30, color='green', alpha=0.7, edgecolor='black')
    plt.axvline(x=0.5, color='red', linestyle='--', label='임계값')
    plt.title('급변 감지 확률 분포')
    plt.xlabel('확률')
    plt.ylabel('빈도')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('error_distribution_v5.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ===================================
    # 최종 저장
    # ===================================
    
    logger.info("\n💾 모델 저장 중...")
    
    os.makedirs('models_v5', exist_ok=True)
    
    # 모델 저장
    for name, model in models.items():
        path = f'models_v5/{name}_final.h5'
        model.save(path)
        logger.info(f"   ✅ {name}: {path}")
    
    # 급변 감지기 저장
    spike_model.save('models_v5/spike_detector_final.h5')
    logger.info(f"   ✅ spike_detector: models_v5/spike_detector_final.h5")
    
    # 스케일러 저장
    joblib.dump(scaler, 'models_v5/scaler.pkl')
    logger.info(f"   ✅ scaler: models_v5/scaler.pkl")
    
    # 설정 저장
    config = {
        'version': 'v5.0',
        'features': feature_cols,
        'seq_length': 50,
        'models': list(models.keys()),
        'performance': {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'mae_1400+': float(mae_high) if high_mask.sum() > 0 else None,
            'spike_accuracy': float(spike_acc)
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open('models_v5/config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    logger.info("\n" + "="*60)
    logger.info("🎉 완료!")
    logger.info(f"📊 최종 성능:")
    logger.info(f"   전체 MAE: {mae:.2f}")
    logger.info(f"   1400+ MAE: {mae_high:.2f}" if high_mask.sum() > 0 else "   1400+ 데이터 없음")
    logger.info("="*60)

# ===================================
# 11. 실행
# ===================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='반도체 물류 예측 v5.0 - 1400+ 강화')
    parser.add_argument('--resume', action='store_true', help='이전 학습 재개')
    parser.add_argument('--reset', action='store_true', help='초기화 후 시작')
    
    args = parser.parse_args()
    
    try:
        main(resume=args.resume, reset=args.reset)
    except Exception as e:
        logger.error(f"❌ 치명적 오류: {str(e)}")
        logger.error(traceback.format_exc())
        print("\n❌ 오류 발생! --resume 옵션으로 재시작 가능")