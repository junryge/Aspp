# -*- coding: utf-8 -*-
"""
HUBROOM 극단값 예측 시스템 - 310+ 특화 개선 버전
- 310+ 극단값 예측 강화
- 다단계 보정 시스템
- 구간별 차별화된 학습
- 중단/재시작 완벽 지원
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle
import os
import json
from tqdm import tqdm
import joblib
from collections import Counter
import signal
import sys

warnings.filterwarnings('ignore')

# TensorFlow 설정
print(f"TensorFlow Version: {tf.__version__}")
tf.random.set_seed(42)
np.random.seed(42)

# GPU 메모리 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# ===========================
# 🔄 체크포인트 관리자 (개선)
# ===========================

class CheckpointManager:
    """학습 중단/재개를 위한 체크포인트 관리"""
    
    def __init__(self, checkpoint_dir='./checkpoints'):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.interrupted = False
        
        # Ctrl+C 인터럽트 핸들러
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        print('\n\n⚠️ 학습 중단 감지! 체크포인트 저장 중...')
        self.interrupted = True
    
    def save_checkpoint(self, model, model_name, epoch, history=None):
        """모델 체크포인트 저장"""
        try:
            # 모델 가중치
            model_path = os.path.join(self.checkpoint_dir, f'{model_name}_checkpoint.h5')
            model.save_weights(model_path)
            
            # 메타데이터
            meta_path = os.path.join(self.checkpoint_dir, f'{model_name}_meta.pkl')
            meta_data = {
                'epoch': epoch,
                'history': history.history if history else None
            }
            with open(meta_path, 'wb') as f:
                pickle.dump(meta_data, f)
            
            print(f"💾 체크포인트 저장: {model_name} (Epoch {epoch})")
            return True
        except Exception as e:
            print(f"❌ 체크포인트 저장 실패: {e}")
            return False
    
    def load_checkpoint(self, model, model_name):
        """모델 체크포인트 로드"""
        try:
            model_path = os.path.join(self.checkpoint_dir, f'{model_name}_checkpoint.h5')
            meta_path = os.path.join(self.checkpoint_dir, f'{model_name}_meta.pkl')
            
            if os.path.exists(model_path) and os.path.exists(meta_path):
                # 가중치 로드
                model.load_weights(model_path)
                
                # 메타데이터 로드
                with open(meta_path, 'rb') as f:
                    meta_data = pickle.load(f)
                
                print(f"📂 체크포인트 로드: {model_name} (Epoch {meta_data['epoch']})")
                return meta_data
            return None
        except Exception as e:
            print(f"⚠️ 체크포인트 로드 실패: {e}")
            return None
    
    def check_resume(self, model_name):
        """재개 여부 확인 (자동 재개)"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f'{model_name}_checkpoint.h5')
        if os.path.exists(checkpoint_path):
            print(f"💾 {model_name}의 이전 체크포인트 발견! 이어서 학습합니다.")
            return True
        return False
    
    def save_training_state(self, state_dict, filename='training_state.pkl'):
        """전체 학습 상태 저장"""
        filepath = os.path.join(self.checkpoint_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(state_dict, f)
        print(f"💾 학습 상태 저장: {filepath}")
    
    def load_training_state(self, filename='training_state.pkl'):
        """전체 학습 상태 로드"""
        filepath = os.path.join(self.checkpoint_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        return None

# ===========================
# 🎯 다단계 극단값 손실함수 (개선)
# ===========================

class MultiLevelExtremeLoss(tf.keras.losses.Loss):
    """310+ 극단값을 위한 다단계 손실함수"""
    
    def __init__(self, name='multi_level_extreme_loss'):
        super().__init__(name=name)
        self.reduction = tf.keras.losses.Reduction.NONE  # 개별 손실값 반환
    
    def call(self, y_true, y_pred):
        # 차원 맞추기
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        
        # 기본 MSE
        mse = tf.square(y_true - y_pred)
        
        # 구간별 가중치 (310+ 특별 강화)
        weight_280 = tf.where(y_true >= 0.6, 2.0, 1.0)  # MinMaxScaler 적용된 값
        weight_300 = tf.where(y_true >= 0.65, 5.0, 1.0)  
        weight_310 = tf.where(y_true >= 0.7, 10.0, 1.0)  # 310+ 강조
        weight_350 = tf.where(y_true >= 0.8, 20.0, 1.0)  # 350+ 더 강조
        
        # 누적 가중치
        total_weight = weight_280 * weight_300 * weight_310 * weight_350
        
        # 가중 MSE (배치의 각 샘플별 손실값 반환)
        weighted_mse = mse * total_weight
        
        return weighted_mse  # 평균하지 않고 반환

# ===========================
# 📊 극단값 특화 데이터 처리 (개선)
# ===========================

class ImprovedExtremeProcessor:
    """310+ 극단값 처리 강화"""
    
    def __init__(self, file_path='data/HUB_0509_TO_0730_DATA.CSV'):
        self.file_path = file_path
        self.target_col = 'CURRENT_M16A_3F_JOB_2'
        
        # 세분화된 임계값
        self.thresholds = {
            'normal': 250,
            'warning': 280,
            'critical': 300,
            'severe': 310,     # 추가
            'extreme': 350,    # 수정
            'disaster': 400
        }
        
        # 스케일러
        self.scaler_X = RobustScaler()
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))
        self.scaler_physics = StandardScaler()
        
        # 스케일러 저장 경로
        self.scaler_dir = './scalers'
        os.makedirs(self.scaler_dir, exist_ok=True)
        
        # 물리적 컬럼
        self.inflow_cols = [
            'M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2',
            'M14A_3F_TO_HUB_JOB2', 'M14B_7F_TO_HUB_JOB2'
        ]
        
        self.outflow_cols = [
            'M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB',
            'M16A_3F_TO_M14A_3F_JOB', 'M16A_3F_TO_M14B_7F_JOB'
        ]
    
    def save_scalers(self):
        """스케일러 저장"""
        joblib.dump(self.scaler_X, os.path.join(self.scaler_dir, 'scaler_X.pkl'))
        joblib.dump(self.scaler_y, os.path.join(self.scaler_dir, 'scaler_y.pkl'))
        joblib.dump(self.scaler_physics, os.path.join(self.scaler_dir, 'scaler_physics.pkl'))
        print(f"✅ 스케일러 저장 완료: {self.scaler_dir}")
    
    def load_scalers(self):
        """스케일러 로드"""
        try:
            self.scaler_X = joblib.load(os.path.join(self.scaler_dir, 'scaler_X.pkl'))
            self.scaler_y = joblib.load(os.path.join(self.scaler_dir, 'scaler_y.pkl'))
            self.scaler_physics = joblib.load(os.path.join(self.scaler_dir, 'scaler_physics.pkl'))
            print(f"✅ 스케일러 로드 완료: {self.scaler_dir}")
            return True
        except:
            print("⚠️ 저장된 스케일러 없음. 새로 학습 필요.")
            return False
    
    def analyze_extremes(self, df):
        """극단값 상세 분석 (310+ 추가)"""
        if self.target_col not in df.columns:
            return
        
        target = df[self.target_col]
        
        print("\n" + "="*60)
        print("🔍 극단값 분석 (HUBROOM 반송량)")
        print("="*60)
        
        print(f"\n📊 기본 통계:")
        print(f"  - 전체 범위: {target.min():.0f} ~ {target.max():.0f}")
        print(f"  - 평균: {target.mean():.2f}")
        print(f"  - 중앙값: {target.median():.0f}")
        print(f"  - 표준편차: {target.std():.2f}")
        
        print(f"\n🚨 임계값별 분포:")
        for name, threshold in self.thresholds.items():
            count = (target >= threshold).sum()
            ratio = count / len(target) * 100
            if ratio > 0:
                print(f"  - {name:8} (≥{threshold:3}): {count:5}개 ({ratio:5.2f}%)")
        
        # 310+ 특별 분석
        print(f"\n💥 310+ 극단값 상세:")
        bins_310 = [310, 320, 330, 340, 350, 400, 500, 1000]
        extreme_310 = target[target >= 310]
        if len(extreme_310) > 0:
            for i in range(len(bins_310)-1):
                mask = (extreme_310 >= bins_310[i]) & (extreme_310 < bins_310[i+1])
                count = mask.sum()
                if count > 0:
                    print(f"    [{bins_310[i]:3}-{bins_310[i+1]:3}): {count:4}개")
    
    def create_enhanced_sequences(self, df, numeric_cols, seq_len=20, pred_len=10):
        """구간별 차별화된 오버샘플링"""
        
        X, y, X_physics, sample_weights = [], [], [], []
        
        available_inflow = [col for col in self.inflow_cols if col in df.columns]
        available_outflow = [col for col in self.outflow_cols if col in df.columns]
        
        total_sequences = len(df) - seq_len - pred_len + 1
        
        # 구간별 인덱스 분류
        indices_by_range = {
            'normal': [],      # < 280
            'warning': [],     # 280-300
            'critical': [],    # 300-310
            'severe': [],      # 310-350
            'extreme': []      # 350+
        }
        
        print("\n📦 구간별 시퀀스 분류 중...")
        
        for i in tqdm(range(total_sequences), desc="시퀀스 분류"):
            target_value = df[self.target_col].iloc[i + seq_len + pred_len - 1]
            
            if target_value >= 350:
                indices_by_range['extreme'].append(i)
            elif target_value >= 310:
                indices_by_range['severe'].append(i)
            elif target_value >= 300:
                indices_by_range['critical'].append(i)
            elif target_value >= 280:
                indices_by_range['warning'].append(i)
            else:
                indices_by_range['normal'].append(i)
        
        # 구간별 통계 출력
        for range_name, indices in indices_by_range.items():
            print(f"  - {range_name:8}: {len(indices):5}개")
        
        # 구간별 차별화된 오버샘플링
        oversample_rates = {
            'normal': 1,
            'warning': 2,
            'critical': 3,
            'severe': 5,      # 310-350은 5배
            'extreme': 10     # 350+는 10배
        }
        
        all_indices = []
        for range_name, indices in indices_by_range.items():
            rate = oversample_rates[range_name]
            all_indices.extend(indices * rate)
            if rate > 1:
                print(f"  - {range_name} 오버샘플링: {len(indices)} → {len(indices) * rate}")
        
        np.random.shuffle(all_indices)
        print(f"\n  - 전체 학습 시퀀스: {len(all_indices)}개")
        
        # 시퀀스 생성
        for i in tqdm(all_indices, desc="시퀀스 생성"):
            # 입력 시퀀스
            X_seq = df[numeric_cols].iloc[i:i+seq_len].values
            X.append(X_seq)
            
            # 타겟 (10분 후)
            y_val = df[self.target_col].iloc[i + seq_len + pred_len - 1]
            y.append(y_val)
            
            # 세분화된 가중치
            if y_val >= 350:
                weight = 20.0
            elif y_val >= 310:
                weight = 15.0
            elif y_val >= 300:
                weight = 10.0
            elif y_val >= 280:
                weight = 5.0
            else:
                weight = 1.0
            sample_weights.append(weight)
            
            # 물리 데이터
            physics_data = np.array([
                df[self.target_col].iloc[i + seq_len - 1],  # 현재 HUBROOM
                df[available_inflow].iloc[i+seq_len:i+seq_len+pred_len].sum().sum() if available_inflow else 0,
                df[available_outflow].iloc[i+seq_len:i+seq_len+pred_len].sum().sum() if available_outflow else 0
            ])
            X_physics.append(physics_data)
        
        return (np.array(X), np.array(y), np.array(X_physics), np.array(sample_weights))

# ===========================
# 🧠 310+ 특화 PatchTST 모델
# ===========================

class EnhancedExtremePatchTST(keras.Model):
    """310+ 극단값 예측 강화 모델"""
    
    def __init__(self, config):
        super(EnhancedExtremePatchTST, self).__init__()
        
        self.seq_len = config['seq_len']
        self.n_features = config['n_features']
        self.patch_len = config['patch_len']
        self.n_patches = self.seq_len // self.patch_len
        
        self.d_model = config['d_model']
        self.n_heads = config['n_heads']
        self.d_ff = config['d_ff']
        self.n_layers = config['n_layers']
        self.dropout_rate = config['dropout']
        
        # 패치 임베딩
        self.patch_embedding = layers.Dense(self.d_model, activation='relu')
        
        # 위치 인코딩
        self.pos_embedding = self.add_weight(
            name='pos_embedding',
            shape=(1, self.n_patches, self.d_model),
            initializer='random_normal',
            trainable=True
        )
        
        # Transformer 인코더
        self.encoder_layers = []
        for _ in range(self.n_layers):
            self.encoder_layers.append(
                layers.MultiHeadAttention(
                    num_heads=self.n_heads,
                    key_dim=self.d_model // self.n_heads,
                    dropout=self.dropout_rate
                )
            )
            self.encoder_layers.append(layers.LayerNormalization())
            self.encoder_layers.append(layers.Dropout(self.dropout_rate))
        
        # 다단계 극단값 감지
        self.extreme_detector_300 = keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')  # 300+ 확률
        ])
        
        self.extreme_detector_310 = keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')  # 310+ 확률
        ])
        
        self.extreme_detector_350 = keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')  # 350+ 확률
        ])
        
        # 값 예측 브랜치
        self.value_predictor = keras.Sequential([
            layers.Dense(256, activation='relu'),  # 더 큰 네트워크
            layers.Dropout(self.dropout_rate),
            layers.Dense(128, activation='relu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])
        
        self.flatten = layers.Flatten()
    
    def create_patches(self, x):
        batch_size = tf.shape(x)[0]
        x_reshaped = tf.reshape(x, [batch_size, self.n_patches, self.patch_len, self.n_features])
        x_patches = tf.reshape(x_reshaped, [batch_size, self.n_patches, self.patch_len * self.n_features])
        return x_patches
    
    def call(self, x, training=False):
        # 패치 생성
        x_patches = self.create_patches(x)
        
        # 패치 임베딩
        x_embed = self.patch_embedding(x_patches)
        x_embed = x_embed + self.pos_embedding
        
        # Transformer 인코더
        for i in range(0, len(self.encoder_layers), 3):
            attention = self.encoder_layers[i](x_embed, x_embed, training=training)
            norm = self.encoder_layers[i+1](attention + x_embed)
            x_embed = self.encoder_layers[i+2](norm, training=training)
        
        # Flatten
        x_flat = self.flatten(x_embed)
        
        # 다단계 극단값 감지
        prob_300 = self.extreme_detector_300(x_flat)
        prob_310 = self.extreme_detector_310(x_flat)
        prob_350 = self.extreme_detector_350(x_flat)
        
        # 값 예측
        value_pred = self.value_predictor(x_flat)
        
        # 다단계 보정 (개선된 로직)
        adjustment_300 = prob_300 * 30.0   # 300+ 기본 보정
        adjustment_310 = prob_310 * 40.0   # 310+ 추가 보정
        adjustment_350 = prob_350 * 50.0   # 350+ 최대 보정
        
        # 누적 보정 (최대값 사용)
        total_adjustment = tf.maximum(
            adjustment_300,
            tf.maximum(adjustment_310, adjustment_350)
        )
        
        # 최종 예측
        adjusted_pred = value_pred + total_adjustment
        
        return tf.squeeze(adjusted_pred, axis=-1)

# ===========================
# 📈 구간별 모니터링 콜백
# ===========================

class DetailedExtremeMonitor(Callback):
    """310+ 구간별 상세 모니터링"""
    
    def __init__(self, X_val, y_val, scaler_y):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.scaler_y = scaler_y
    
    def on_epoch_end(self, epoch, logs=None):
        # 예측
        y_pred_scaled = self.model.predict(self.X_val, verbose=0)
        
        # 역정규화
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_true = self.scaler_y.inverse_transform(self.y_val.reshape(-1, 1)).flatten()
        
        # 구간별 정확도
        ranges = [(300, 310), (310, 350), (350, 1000)]
        
        print(f"\n  🎯 Epoch {epoch+1} - 극단값 구간별 성능:")
        
        for low, high in ranges:
            mask = (y_true >= low) & (y_true < high)
            if mask.sum() > 0:
                pred_correct = ((y_pred >= low) & (y_pred < high))[mask].sum()
                recall = pred_correct / mask.sum()
                mae = np.mean(np.abs(y_true[mask] - y_pred[mask]))
                print(f"     [{low}-{high}): Recall={recall:.1%}, MAE={mae:.1f}, Count={mask.sum()}")

# ===========================
# 🎯 메인 실행
# ===========================

def main():
    print("="*80)
    print("🏭 HUBROOM 극단값 예측 시스템 - 310+ 강화 버전")
    print("🎯 목표: 310 이상 극단값 정확한 예측")
    print("="*80)
    
    # 체크포인트 관리자
    checkpoint_manager = CheckpointManager()
    
    # 이전 학습 상태 확인
    training_state = checkpoint_manager.load_training_state()
    if training_state:
        print("📂 이전 학습 상태 발견! 복원 중...")
        start_phase = training_state.get('phase', 'data_load')
    else:
        start_phase = 'data_load'
        training_state = {}
    
    # 데이터 프로세서
    processor = ImprovedExtremeProcessor()
    
    # Phase 1: 데이터 로드 및 전처리
    if start_phase == 'data_load':
        print("\n[Phase 1/5] 📂 데이터 로드 및 전처리...")
        df = pd.read_csv(processor.file_path)
        print(f"✅ 데이터 로드 완료: {df.shape}")
        
        # 극단값 분석
        processor.analyze_extremes(df)
        
        # 전처리
        time_col = df.columns[0]
        df['timestamp'] = pd.to_datetime(df[time_col], format='%Y%m%d%H%M', errors='coerce')
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = df.fillna(method='ffill').fillna(0)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 시퀀스 생성
        X, y, X_physics, sample_weights = processor.create_enhanced_sequences(
            df, numeric_cols, seq_len=20, pred_len=10
        )
        
        # 상태 저장
        training_state['X'] = X
        training_state['y'] = y
        training_state['X_physics'] = X_physics
        training_state['sample_weights'] = sample_weights
        training_state['phase'] = 'data_split'
        checkpoint_manager.save_training_state(training_state)
    
    # Phase 2: 데이터 분할
    if start_phase in ['data_load', 'data_split']:
        print("\n[Phase 2/5] 🔀 데이터 분할...")
        
        if 'X' not in locals():
            X = training_state['X']
            y = training_state['y']
            X_physics = training_state['X_physics']
            sample_weights = training_state['sample_weights']
        
        indices = np.arange(len(X))
        train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        X_physics_train = X_physics[train_idx]
        X_physics_val = X_physics[val_idx]
        X_physics_test = X_physics[test_idx]
        
        weights_train = sample_weights[train_idx]
        
        print(f"  - Train: {len(X_train)} ({(y_train >= 310).sum()} 310+극단값)")
        print(f"  - Valid: {len(X_val)} ({(y_val >= 310).sum()} 310+극단값)")
        print(f"  - Test: {len(X_test)} ({(y_test >= 310).sum()} 310+극단값)")
        
        training_state['data_split'] = {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'X_physics_train': X_physics_train,
            'X_physics_val': X_physics_val,
            'X_physics_test': X_physics_test,
            'weights_train': weights_train
        }
        training_state['phase'] = 'scaling'
        checkpoint_manager.save_training_state(training_state)
    
    # Phase 3: 스케일링
    if start_phase in ['data_load', 'data_split', 'scaling']:
        print("\n[Phase 3/5] 📏 데이터 정규화...")
        
        if 'X_train' not in locals():
            data_split = training_state['data_split']
            X_train = data_split['X_train']
            X_val = data_split['X_val']
            X_test = data_split['X_test']
            y_train = data_split['y_train']
            y_val = data_split['y_val']
            y_test = data_split['y_test']
            X_physics_train = data_split['X_physics_train']
            X_physics_val = data_split['X_physics_val']
            X_physics_test = data_split['X_physics_test']
            weights_train = data_split['weights_train']
        
        n_samples_train, seq_len, n_features = X_train.shape
        
        # 스케일링
        X_train_scaled = processor.scaler_X.fit_transform(
            X_train.reshape(-1, n_features)
        ).reshape(n_samples_train, seq_len, n_features)
        
        X_val_scaled = processor.scaler_X.transform(
            X_val.reshape(-1, n_features)
        ).reshape(X_val.shape[0], seq_len, n_features)
        
        X_test_scaled = processor.scaler_X.transform(
            X_test.reshape(-1, n_features)
        ).reshape(X_test.shape[0], seq_len, n_features)
        
        y_train_scaled = processor.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = processor.scaler_y.transform(y_val.reshape(-1, 1)).flatten()
        y_test_scaled = processor.scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        
        X_physics_train_scaled = processor.scaler_physics.fit_transform(X_physics_train)
        X_physics_val_scaled = processor.scaler_physics.transform(X_physics_val)
        X_physics_test_scaled = processor.scaler_physics.transform(X_physics_test)
        
        # 스케일러 저장
        processor.save_scalers()
        
        training_state['scaled_data'] = {
            'X_train_scaled': X_train_scaled,
            'X_val_scaled': X_val_scaled,
            'X_test_scaled': X_test_scaled,
            'y_train_scaled': y_train_scaled,
            'y_val_scaled': y_val_scaled,
            'y_test_scaled': y_test_scaled,
            'X_physics_train_scaled': X_physics_train_scaled,
            'X_physics_val_scaled': X_physics_val_scaled,
            'X_physics_test_scaled': X_physics_test_scaled,
            'n_features': n_features
        }
        training_state['phase'] = 'training'
        checkpoint_manager.save_training_state(training_state)
    
    # Phase 4: 모델 학습
    if start_phase in ['data_load', 'data_split', 'scaling', 'training']:
        print("\n[Phase 4/5] 🚀 모델 학습...")
        
        if 'X_train_scaled' not in locals():
            scaled_data = training_state['scaled_data']
            X_train_scaled = scaled_data['X_train_scaled']
            X_val_scaled = scaled_data['X_val_scaled']
            X_test_scaled = scaled_data['X_test_scaled']
            y_train_scaled = scaled_data['y_train_scaled']
            y_val_scaled = scaled_data['y_val_scaled']
            y_test_scaled = scaled_data['y_test_scaled']
            n_features = scaled_data['n_features']
            weights_train = training_state['data_split']['weights_train']
            y_test = training_state['data_split']['y_test']
        
        # 모델 구성
        config = {
            'seq_len': 20,
            'n_features': n_features,
            'patch_len': 5,
            'd_model': 128,
            'n_heads': 8,
            'd_ff': 256,
            'n_layers': 4,
            'dropout': 0.2
        }
        
        # 콜백
        def create_callbacks(model_name):
            return [
                ModelCheckpoint(
                    filepath=os.path.join(checkpoint_manager.checkpoint_dir, f'{model_name}_best.h5'),
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=True,
                    verbose=1
                ),
                EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6),
                DetailedExtremeMonitor(X_val_scaled, y_val_scaled, processor.scaler_y)
            ]
        
        results = {}
        
        # EnhancedExtremePatchTST 학습
        print("\n🤖 EnhancedExtremePatchTST (310+ 특화) 학습")
        model = EnhancedExtremePatchTST(config)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=MultiLevelExtremeLoss(),
            metrics=['mae']
        )
        
        # 체크포인트 확인
        initial_epoch = 0
        if checkpoint_manager.check_resume('EnhancedExtremePatchTST'):
            meta = checkpoint_manager.load_checkpoint(model, 'EnhancedExtremePatchTST')
            if meta:
                initial_epoch = meta['epoch']
        
        try:
            history = model.fit(
                X_train_scaled, y_train_scaled,
                validation_data=(X_val_scaled, y_val_scaled),
                sample_weight=weights_train,
                initial_epoch=initial_epoch,
                epochs=70,  # 더 많은 에폭
                batch_size=32,
                callbacks=create_callbacks('EnhancedExtremePatchTST'),
                verbose=1
            )
            
            # 학습 완료 후 저장
            checkpoint_manager.save_checkpoint(model, 'EnhancedExtremePatchTST', 70, history)
            
        except KeyboardInterrupt:
            print("\n⚠️ 학습 중단됨!")
            current_epoch = len(history.history['loss']) if 'history' in locals() else initial_epoch
            checkpoint_manager.save_checkpoint(model, 'EnhancedExtremePatchTST', current_epoch)
            print("💾 재시작 시 이어서 학습 가능합니다.")
            sys.exit(0)
        
        # 평가
        y_pred_scaled = model.predict(X_test_scaled)
        y_pred = processor.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        results['EnhancedExtremePatchTST'] = y_pred
        
        training_state['results'] = results
        training_state['phase'] = 'evaluation'
        checkpoint_manager.save_training_state(training_state)
    
    # Phase 5: 최종 평가
    print("\n[Phase 5/5] 📊 최종 성능 평가")
    
    if 'results' not in locals():
        results = training_state['results']
        y_test = training_state['data_split']['y_test']
    
    for model_name, y_pred in results.items():
        print(f"\n🔍 {model_name}:")
        
        # 전체 성능
        mae = np.mean(np.abs(y_test - y_pred))
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        print(f"  전체 MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        
        # 구간별 상세 성능
        print(f"\n  📈 구간별 성능:")
        ranges = [
            (280, 300, "경고"),
            (300, 310, "위험"),
            (310, 350, "심각"),
            (350, 1000, "재난")
        ]
        
        for low, high, name in ranges:
            mask = (y_test >= low) & (y_test < high)
            if mask.sum() > 0:
                range_mae = np.mean(np.abs(y_test[mask] - y_pred[mask]))
                pred_in_range = ((y_pred >= low) & (y_pred < high))[mask].sum()
                accuracy = pred_in_range / mask.sum()
                
                print(f"    {name}[{low}-{high}): "
                      f"MAE={range_mae:.1f}, "
                      f"정확도={accuracy:.1%}, "
                      f"개수={mask.sum()}")
        
        # 310+ 특별 분석
        mask_310 = y_test >= 310
        if mask_310.sum() > 0:
            print(f"\n  🎯 310+ 극단값 성능:")
            mae_310 = np.mean(np.abs(y_test[mask_310] - y_pred[mask_310]))
            pred_310 = y_pred >= 310
            tp = (mask_310 & pred_310).sum()
            recall = tp / mask_310.sum()
            
            print(f"    - MAE: {mae_310:.2f}")
            print(f"    - 감지율: {tp}/{mask_310.sum()} ({recall:.1%})")
            
            # 샘플 출력
            print(f"\n  📝 310+ 예측 샘플:")
            indices_310 = np.where(mask_310)[0][:5]
            for idx in indices_310:
                status = "✅" if abs(y_test[idx] - y_pred[idx]) < 30 else "⚠️" if abs(y_test[idx] - y_pred[idx]) < 50 else "❌"
                print(f"    {status} 실제: {y_test[idx]:.0f}, "
                      f"예측: {y_pred[idx]:.0f}, "
                      f"오차: {abs(y_test[idx] - y_pred[idx]):.0f}")
    
    print("\n✅ 모든 작업 완료!")
    print("💾 스케일러: ./scalers/")
    print("💾 체크포인트: ./checkpoints/")
    print("🎯 310+ 극단값 예측 강화 완료")

if __name__ == "__main__":
    main()