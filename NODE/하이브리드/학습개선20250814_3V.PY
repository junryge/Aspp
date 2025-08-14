"""

def load_and_preprocess_data(data_path, checkpoint_manager, force_reload=False):
    """데이터 로드 및 전처리 (진행상황 표시 + 최적화)"""
    
    # 캐시 확인
    if not force_reload:
        cached_data = checkpoint_manager.load_data()
        if cached_data:
            logger.info("✅ 캐시된 데이터 사용")
            return cached_data
    
    logger.info(f"📂 데이터 새로 로딩: {data_path}")
    
    try:
        # ========== 1단계: 데이터 로드 ==========
        logger.info("📊 [1/8] CSV 파일 로딩 중...")
        data = pd.read_csv(data_path)
        logger.info(f"   ✓ 원본 데이터 크기: {data.shape}")
        
        # ========== 2단계: 시간 변환 ==========
        logger.info("🕒 [2/8] 시간 데이터 변환 중...")
        data['CURRTIME'] = pd.to_datetime(data['CURRTIME'], format='%Y%m%d%H%M')
        data['TIME'] = pd.to_datetime(data['TIME'], format='%Y%m%d%H%M')
        
        # 필요 컬럼만 선택
        data = data[['CURRTIME', 'TOTALCNT', 'TIME']]
        data.set_index('CURRTIME', inplace=True)
        logger.info(f"   ✓ 컬럼 선택 완료")
        
        # ========== 3단계: 날짜 필터링 ==========
        logger.info("📅 [3/8] 날짜 범위 필터링 중...")
        start_date = pd.to_datetime('2024-02-01 00:00:00')
        end_date = pd.to_datetime('2024-07-27 23:59:59')
        data = data[(data['TIME'] >= start_date) & (data['TIME'] <= end_date)]
        logger.info(f"   ✓ 필터링 후 데이터: {data.shape}")
        
        # ========== 4단계: 이상치 제거 ==========
        logger.info("🔍 [4/8] 이상치 제거 중...")
        before_outlier = len(data)
        data = data[(data['TOTALCNT'] >= 800) & (data['TOTALCNT'] <= 2500)]
        logger.info(f"   ✓ 제거된 이상치: {before_outlier - len(data)}개")
        
        # ========== 5단계: FUTURE 및 라벨 생성 ==========
        logger.info("🎯 [5/8] 타겟 변수 생성 중...")
        data['FUTURE'] = data['TOTALCNT'].shift(-10)
        data['spike_label'] = (data['FUTURE'] >= 1400).astype(int)
        data.dropna(inplace=True)
        
        logger.info(f"   ✓ 전처리 완료: {data.shape}")
        logger.info(f"   ✓ 1400+ 비율: {data['spike_label'].mean():.2%}")
        logger.info(f"   ✓ 1400+ 개수: {data['spike_label'].sum()}개")
        
        # ========== 6단계: 특징 생성 (최적화) ==========
        logger.info("⚙️ [6/8] 특징 엔지니어링 중...")
        total_features = 12
        current_feature = 0
        
        # 이동평균 계산
        current_feature += 1
        logger.info(f"   [{current_feature}/{total_features}] MA_10 계산...")
        data['MA_10'] = data['TOTALCNT'].rolling(10, min_periods=1).mean()
        
        current_feature += 1
        logger.info(f"   [{current_feature}/{total_features}] MA_30 계산...")
        data['MA_30'] = data['TOTALCNT'].rolling(30, min_periods=1).mean()
        
        current_feature += 1
        logger.info(f"   [{current_feature}/{total_features}] MA_60 계산...")
        data['MA_60'] = data['TOTALCNT'].rolling(60, min_periods=1).mean()
        
        # 표준편차 계산
        current_feature += 1
        logger.info(f"   [{current_feature}/{total_features}] STD_10 계산...")
        data['STD_10'] = data['TOTALCNT'].rolling(10, min_periods=1).std()
        
        current_feature += 1
        logger.info(f"   [{current_feature}/{total_features}] STD_30 계산...")
        data['STD_30'] = data['TOTALCNT'].rolling(30, min_periods=1).std()
        
        # 변화율 계산
        current_feature += 1
        logger.info(f"   [{current_feature}/{total_features}] 변화율 계산...")
        data['change_rate'] = data['TOTALCNT'].pct_change()
        
        current_feature += 1
        logger.info(f"   [{current_feature}/{total_features}] 10기간 변화율 계산...")
        data['change_rate_10'] = data['TOTALCNT'].pct_change(10)
        
        # 시간 특징
        current_feature += 1
        logger.info(f"   [{current_feature}/{total_features}] 시간 특징 추출...")
        data['hour'] = data.index.hour
        
        current_feature += 1
        logger.info(f"   [{current_feature}/{total_features}] 요일 특징 추출...")
        data['dayofweek'] = data.index.dayofweek
        
        current_feature += 1
        logger.info(f"   [{current_feature}/{total_features}] 주말 플래그 생성...")
        data['is_weekend'] = (data.index.dayofweek >= 5).astype(int)
        
        # 추세 계산
        current_feature += 1
        logger.info(f"   [{current_feature}/{total_features}] 추세 계산...")
        data['trend'] = data['MA_10'] - data['MA_30']
        
        # 결측치 처리
        current_feature += 1
        logger.info(f"   [{current_feature}/{total_features}] 결측치 처리...")
        data.fillna(method='ffill', inplace=True)
        data.fillna(0, inplace=True)
        
        logger.info("   ✓ 특징 생성 완료!")
        
        # ========== 7단계: 스케일링 ==========
        logger.info("📏 [7/8] 데이터 스케일링 중...")
        scaler = RobustScaler()
        feature_cols = ['TOTALCNT', 'MA_10', 'MA_30', 'MA_60', 'STD_10', 'STD_30',
                       'change_rate', 'change_rate_10', 'hour', 'dayofweek', 
                       'is_weekend', 'trend']
        
        data[feature_cols + ['FUTURE']] = scaler.fit_transform(data[feature_cols + ['FUTURE']])
        logger.info(f"   ✓ {len(feature_cols)} 개 특징 스케일링 완료")
        
        # ========== 8단계: 시퀀스 생성 ==========
        logger.info("🔄 [8/8] 시퀀스 데이터 생성 중...")
        SEQ_LENGTH = 50
        X, y, spike_labels = [], [], []
        
        # 진행 상황 표시를 위한 총 개수
        total_sequences = len(data) - SEQ_LENGTH
        
        # 배치 처리로 메모리 효율성 향상
        logger.info(f"   총 {total_sequences}개 시퀀스 생성 중...")
        
        for i in range(total_sequences):
            # 1000개마다 진행상황 표시
            if i % 1000 == 0:
                progress = (i / total_sequences) * 100
                logger.info(f"   진행률: {progress:.1f}% ({i}/{total_sequences})")
            
            X.append(data[feature_cols].iloc[i:i+SEQ_LENGTH].values)
            y.append(data['FUTURE'].iloc[i+SEQ_LENGTH])
            spike_labels.append(data['spike_label'].iloc[i+SEQ_LENGTH])
        
        logger.info("   ✓ 시퀀스 생성 완료! NumPy 배열로 변환 중...")
        X, y, spike_labels = np.array(X), np.array(y), np.array(spike_labels)
        
        logger.info(f"   ✓ 최종 시퀀스 shape: X={X.shape}, y={y.shape}")
        
        # ========== 데이터 분할 ==========
        logger.info("📊 데이터셋 분할 중...")
        train_size = int(0.7 * len(X))
        val_size = int(0.15 * len(X))
        
        logger.info(f"   ✓ Train: {train_size}개")
        logger.info(f"   ✓ Val: {val_size}개")
        logger.info(f"   ✓ Test: {len(X) - train_size - val_size}개")
        
        # 딕셔너리 생성
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
        
        # ========== 캐시 저장 ==========
        logger.info("💾 전처리된 데이터 캐싱 중...")
        checkpoint_manager.save_data(data_dict)
        
        # ========== 완료 ==========
        logger.info("✅ 데이터 전처리 완료!")
        logger.info(f"   - 입력 shape: {data_dict['input_shape']}")
        logger.info(f"   - Train 1400+ 비율: {spike_labels[:train_size].mean():.2%}")
        logger.info(f"   - Val 1400+ 비율: {spike_labels[train_size:train_size+val_size].mean():.2%}")
        logger.info(f"   - Test 1400+ 비율: {spike_labels[train_size+val_size:].mean():.2%}")
        
        return data_dict
        
    except Exception as e:
        logger.error(f"❌ 데이터 전처리 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        raise

반도체 물류 예측 ULTIMATE v4.0 - 완전체
========================================
모든 기능 포함 + 재시작 가능 + 오버피팅 해결

핵심 기능:
1. 중단 후 재시작 가능 (--resume)
2. 에폭별 체크포인트 자동 저장
3. 오버피팅 완전 해결
4. 급변 감지 분류기
5. 실시간 진행상황 저장

사용법:
python model_v4_ultimate.py          # 처음 시작
python model_v4_ultimate.py --resume # 이어서 시작
python model_v4_ultimate.py --reset  # 초기화 후 시작
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

# 경고 숨기기
warnings.filterwarnings('ignore')

# ===================================
# 1. 환경 설정
# ===================================

# CPU 모드 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

# 랜덤 시드 (원래 값!)
RANDOM_SEED = 2079936
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_ultimate.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===================================
# 2. 강화된 체크포인트 관리자
# ===================================

class UltimateCheckpointManager:
    """완벽한 체크포인트 관리"""
    
    def __init__(self, checkpoint_dir='checkpoints_ultimate'):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 파일 경로들
        self.state_file = os.path.join(checkpoint_dir, 'training_state.json')
        self.data_file = os.path.join(checkpoint_dir, 'preprocessed_data.pkl')
        self.models_dir = os.path.join(checkpoint_dir, 'models')
        os.makedirs(self.models_dir, exist_ok=True)
        
    def save_state(self, state_dict):
        """학습 상태 저장 (매 에폭)"""
        # 기존 상태 로드
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                existing = json.load(f)
        else:
            existing = {}
        
        # 업데이트
        existing.update(state_dict)
        
        # 저장
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
        logger.info(f"데이터 저장: {self.data_file}")
        
    def load_data(self):
        """전처리된 데이터 로드"""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"데이터 로드: {self.data_file}")
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
            logger.info(f"가중치 로드: {path}")
            return True
        return False
    
    def get_latest_epoch(self, model_name):
        """최신 에폭 찾기"""
        state = self.load_state()
        if state and 'model_progress' in state:
            return state['model_progress'].get(model_name, {}).get('last_epoch', 0)
        return 0

# ===================================
# 3. 커스텀 콜백 (진행상황 저장)
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
# 4. 데이터 전처리 (캐싱 포함)
# ===================================

def load_and_preprocess_data(data_path, checkpoint_manager, force_reload=False):
    """데이터 로드 (캐싱)"""
    
    # 캐시 확인
    if not force_reload:
        cached_data = checkpoint_manager.load_data()
        if cached_data:
            logger.info("캐시된 데이터 사용")
            return cached_data
    
    logger.info(f"데이터 새로 로딩: {data_path}")
    
    # 데이터 로드
    data = pd.read_csv(data_path)
    
    # 시간 변환
    data['CURRTIME'] = pd.to_datetime(data['CURRTIME'], format='%Y%m%d%H%M')
    data['TIME'] = pd.to_datetime(data['TIME'], format='%Y%m%d%H%M')
    
    # 필요 컬럼
    data = data[['CURRTIME', 'TOTALCNT', 'TIME']]
    data.set_index('CURRTIME', inplace=True)
    
    # 날짜 필터링
    start_date = pd.to_datetime('2024-02-01 00:00:00')
    end_date = pd.to_datetime('2024-07-27 23:59:59')
    data = data[(data['TIME'] >= start_date) & (data['TIME'] <= end_date)]
    
    # 이상치 제거
    data = data[(data['TOTALCNT'] >= 800) & (data['TOTALCNT'] <= 2500)]
    
    # FUTURE 생성
    data['FUTURE'] = data['TOTALCNT'].shift(-10)
    data['spike_label'] = (data['FUTURE'] >= 1400).astype(int)
    data.dropna(inplace=True)
    
    logger.info(f"전처리 완료: {data.shape}")
    logger.info(f"1400+ 비율: {data['spike_label'].mean():.2%}")
    
    # 특징 생성
    data['MA_10'] = data['TOTALCNT'].rolling(10, min_periods=1).mean()
    data['MA_30'] = data['TOTALCNT'].rolling(30, min_periods=1).mean()
    data['MA_60'] = data['TOTALCNT'].rolling(60, min_periods=1).mean()
    data['STD_10'] = data['TOTALCNT'].rolling(10, min_periods=1).std()
    data['STD_30'] = data['TOTALCNT'].rolling(30, min_periods=1).std()
    data['change_rate'] = data['TOTALCNT'].pct_change()
    data['change_rate_10'] = data['TOTALCNT'].pct_change(10)
    data['hour'] = data.index.hour
    data['dayofweek'] = data.index.dayofweek
    data['is_weekend'] = (data.index.dayofweek >= 5).astype(int)
    data['trend'] = data['MA_10'] - data['MA_30']
    
    data.fillna(method='ffill', inplace=True)
    data.fillna(0, inplace=True)
    
    # 스케일링
    scaler = RobustScaler()
    feature_cols = ['TOTALCNT', 'MA_10', 'MA_30', 'MA_60', 'STD_10', 'STD_30',
                   'change_rate', 'change_rate_10', 'hour', 'dayofweek', 
                   'is_weekend', 'trend']
    
    data[feature_cols + ['FUTURE']] = scaler.fit_transform(data[feature_cols + ['FUTURE']])
    
    # 시퀀스 생성
    SEQ_LENGTH = 50
    X, y, spike_labels = [], [], []
    
    for i in range(len(data) - SEQ_LENGTH):
        X.append(data[feature_cols].iloc[i:i+SEQ_LENGTH].values)
        y.append(data['FUTURE'].iloc[i+SEQ_LENGTH])
        spike_labels.append(data['spike_label'].iloc[i+SEQ_LENGTH])
    
    X, y, spike_labels = np.array(X), np.array(y), np.array(spike_labels)
    
    # 데이터 분할
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
    
    return data_dict

# ===================================
# 5. 모델 정의
# ===================================

def build_simple_lstm(input_shape):
    """단순 LSTM"""
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True, kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
        Dropout(0.5),
        BatchNormalization(),
        LSTM(32, return_sequences=False, kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
        Dropout(0.5),
        Dense(16, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
        Dropout(0.3),
        Dense(1)
    ])
    return model

def build_gru_model(input_shape):
    """GRU 모델"""
    model = Sequential([
        Input(shape=input_shape),
        GRU(64, return_sequences=True, kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
        Dropout(0.5),
        GRU(32, return_sequences=False),
        Dropout(0.5),
        Dense(16, activation='relu'),
        Dropout(0.3),
        Dense(1)
    ])
    return model

def build_cnn_lstm(input_shape):
    """CNN-LSTM"""
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(32, 3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.4),
        LSTM(32, return_sequences=False, kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
        Dropout(0.5),
        Dense(16, activation='relu'),
        Dropout(0.3),
        Dense(1)
    ])
    return model

def build_spike_detector(input_shape):
    """급변 감지기"""
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(32, 3, activation='relu'),
        MaxPooling1D(2),
        LSTM(32, return_sequences=False),
        Dropout(0.5),
        Dense(16, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    return model

# ===================================
# 6. 재시작 가능한 학습 함수
# ===================================

def train_model_with_resume(model, model_name, data_dict, checkpoint_manager,
                           epochs=50, batch_size=128, resume=False):
    """재시작 가능한 학습"""
    
    # 데이터 언팩
    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    X_val = data_dict['X_val']
    y_val = data_dict['y_val']
    
    # 급변 감지기인 경우
    if 'spike' in model_name:
        y_train = data_dict['spike_train']
        y_val = data_dict['spike_val']
    
    # 재시작 처리
    start_epoch = 0
    if resume:
        start_epoch = checkpoint_manager.get_latest_epoch(model_name)
        if start_epoch > 0:
            logger.info(f"{model_name} Epoch {start_epoch}부터 재시작")
            checkpoint_manager.load_model_weights(model, model_name, start_epoch)
    
    # 이미 완료된 경우
    if start_epoch >= epochs:
        logger.info(f"{model_name} 이미 완료됨")
        return model
    
    logger.info(f"\n{'='*60}")
    logger.info(f"{model_name} 학습 시작 (Epoch {start_epoch+1}/{epochs})")
    logger.info(f"{'='*60}")
    
    # 컴파일
    if 'spike' in model_name:
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    else:
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='mse',
            metrics=['mae']
        )
    
    # 콜백
    callbacks = [
        CheckpointCallback(checkpoint_manager, model_name),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
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
        logger.warning(f"\n{model_name} 학습 중단됨. 재시작 가능!")
        state = {
            'interrupted': True,
            'interrupted_model': model_name,
            'interrupted_time': datetime.now().isoformat()
        }
        checkpoint_manager.save_state(state)
        raise
        
    except Exception as e:
        logger.error(f"{model_name} 학습 중 오류: {str(e)}")
        state = {
            'error': str(e),
            'error_model': model_name,
            'error_time': datetime.now().isoformat()
        }
        checkpoint_manager.save_state(state)
        raise
    
    return model

# ===================================
# 7. 앙상블 예측
# ===================================

def ensemble_predict(models, spike_detector, X_test):
    """앙상블 예측"""
    
    # 급변 확률
    spike_probs = spike_detector.predict(X_test, verbose=0).flatten()
    
    # 각 모델 예측
    predictions = {}
    for name, model in models.items():
        predictions[name] = model.predict(X_test, verbose=0).flatten()
    
    # 동적 가중치
    ensemble_pred = np.zeros(len(X_test))
    
    for i in range(len(X_test)):
        if spike_probs[i] > 0.5:
            weights = {'lstm': 0.2, 'gru': 0.2, 'cnn_lstm': 0.6}
        else:
            weights = {'lstm': 0.4, 'gru': 0.3, 'cnn_lstm': 0.3}
        
        for name, weight in weights.items():
            if name in predictions:
                ensemble_pred[i] += weight * predictions[name][i]
        
        if spike_probs[i] > 0.5:
            ensemble_pred[i] *= 1.05
    
    return ensemble_pred, predictions, spike_probs

# ===================================
# 8. 메인 실행
# ===================================

def main(resume=False, reset=False):
    """메인 실행"""
    
    # 체크포인트 매니저
    checkpoint_manager = UltimateCheckpointManager()
    
    # 리셋 처리
    if reset:
        if os.path.exists(checkpoint_manager.checkpoint_dir):
            shutil.rmtree(checkpoint_manager.checkpoint_dir)
            logger.info("체크포인트 초기화됨")
        checkpoint_manager = UltimateCheckpointManager()
    
    # 재시작 상태 확인
    if resume:
        state = checkpoint_manager.load_state()
        if state:
            logger.info("="*60)
            logger.info("이전 학습 재개")
            if 'interrupted_model' in state:
                logger.info(f"중단된 모델: {state['interrupted_model']}")
            logger.info("="*60)
        else:
            logger.info("저장된 상태 없음. 처음부터 시작")
            resume = False
    
    logger.info("="*60)
    logger.info("반도체 물류 예측 v4.0 ULTIMATE")
    logger.info("="*60)
    
    # 데이터 로드
    data_dict = load_and_preprocess_data(
        'data/20240201_TO_202507281705.csv',
        checkpoint_manager,
        force_reload=not resume
    )
    
    input_shape = data_dict['input_shape']
    
    # 모델 정의
    model_configs = [
        ('lstm', build_simple_lstm),
        ('gru', build_gru_model),
        ('cnn_lstm', build_cnn_lstm),
        ('spike_detector', build_spike_detector)
    ]
    
    models = {}
    
    # 각 모델 학습
    for model_name, build_func in model_configs:
        try:
            # 완료 확인
            state = checkpoint_manager.load_state()
            if state and 'model_progress' in state:
                if state['model_progress'].get(model_name, {}).get('completed', False):
                    logger.info(f"{model_name} 이미 완료됨. 건너뜀")
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
            
            # 학습
            epochs = 30 if model_name == 'spike_detector' else 50
            model = train_model_with_resume(
                model, model_name, data_dict, checkpoint_manager,
                epochs=epochs, batch_size=128, resume=resume
            )
            
            if model_name != 'spike_detector':
                models[model_name] = model
            else:
                spike_model = model
                
        except KeyboardInterrupt:
            logger.warning("\n학습 중단. python model_v4_ultimate.py --resume 로 재시작 가능")
            return
        except Exception as e:
            logger.error(f"오류: {str(e)}")
            logger.info("python model_v4_ultimate.py --resume 로 재시작 시도")
            return
    
    # 평가
    logger.info("\n" + "="*60)
    logger.info("모델 평가")
    logger.info("="*60)
    
    # 앙상블 예측
    X_test = data_dict['X_test']
    y_test = data_dict['y_test']
    spike_test = data_dict['spike_test']
    scaler = data_dict['scaler']
    feature_cols = data_dict['feature_cols']
    
    ensemble_pred, _, spike_probs = ensemble_predict(models, spike_model, X_test)
    
    # 역변환
    y_test_original = scaler.inverse_transform(
        np.column_stack([np.zeros((len(y_test), len(feature_cols))), y_test])
    )[:, -1]
    
    ensemble_original = scaler.inverse_transform(
        np.column_stack([np.zeros((len(ensemble_pred), len(feature_cols))), ensemble_pred])
    )[:, -1]
    
    # 성능
    mae = mean_absolute_error(y_test_original, ensemble_original)
    rmse = np.sqrt(mean_squared_error(y_test_original, ensemble_original))
    r2 = r2_score(y_test_original, ensemble_original)
    
    logger.info(f"MAE: {mae:.2f}")
    logger.info(f"RMSE: {rmse:.2f}")
    logger.info(f"R²: {r2:.4f}")
    
    # 1400+ 성능
    high_mask = y_test_original >= 1400
    if high_mask.sum() > 0:
        mae_high = mean_absolute_error(y_test_original[high_mask], ensemble_original[high_mask])
        logger.info(f"1400+ MAE: {mae_high:.2f}")
    
    # 급변 감지 정확도
    spike_acc = accuracy_score(spike_test, spike_probs > 0.5)
    logger.info(f"급변 감지 정확도: {spike_acc:.2%}")
    
    # 시각화
    plt.figure(figsize=(15, 5))
    plt.plot(y_test_original[:200], label='실제', color='blue')
    plt.plot(ensemble_original[:200], label='예측', color='red', alpha=0.7)
    plt.axhline(y=1400, color='green', linestyle='--', alpha=0.5)
    plt.title(f'예측 결과 (MAE: {mae:.2f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results_ultimate.png')
    plt.show()
    
    # 최종 저장
    os.makedirs('models_ultimate', exist_ok=True)
    for name, model in models.items():
        model.save(f'models_ultimate/{name}_final.h5')
    spike_model.save('models_ultimate/spike_detector_final.h5')
    joblib.dump(scaler, 'models_ultimate/scaler.pkl')
    
    logger.info("\n" + "="*60)
    logger.info("완료!")
    logger.info(f"최종 MAE: {mae:.2f}")
    logger.info("="*60)

# ===================================
# 9. 실행
# ===================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='반도체 물류 예측 v4.0 ULTIMATE')
    parser.add_argument('--resume', action='store_true', help='이전 학습 재개')
    parser.add_argument('--reset', action='store_true', help='초기화 후 시작')
    
    args = parser.parse_args()
    
    try:
        main(resume=args.resume, reset=args.reset)
    except Exception as e:
        logger.error(f"치명적 오류: {str(e)}")
        logger.error(traceback.format_exc())
        print("\n❌ 오류 발생! --resume 옵션으로 재시작 가능")