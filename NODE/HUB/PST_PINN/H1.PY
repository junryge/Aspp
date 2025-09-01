# -*- coding: utf-8 -*-
"""
HUBROOM 반송량 예측 시스템 - TensorFlow 2.15 버전
- PyTorch 코드를 TensorFlow로 완전 변환
- PatchTST 모델
- PatchTST + PINN 하이브리드 모델
- 진행률 표시 및 체크포인트 기능
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle
import os
import json
from tqdm import tqdm
import time
import signal
import sys

warnings.filterwarnings('ignore')

# TensorFlow 설정
print(f"TensorFlow Version: {tf.__version__}")
tf.random.set_seed(42)

# ===========================
# 🔄 체크포인트 관리
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
    
    def save_checkpoint(self, model, filename):
        """모델 체크포인트 저장"""
        filepath = os.path.join(self.checkpoint_dir, filename)
        model.save_weights(filepath)
        print(f"💾 체크포인트 저장: {filepath}")
    
    def load_checkpoint(self, model, filename):
        """모델 체크포인트 로드"""
        filepath = os.path.join(self.checkpoint_dir, filename)
        if os.path.exists(filepath + '.index'):
            model.load_weights(filepath)
            print(f"📂 체크포인트 로드: {filepath}")
            return True
        return False
    
    def save_data_progress(self, data_dict, filename='data_progress.pkl'):
        """데이터 처리 진행 상황 저장"""
        filepath = os.path.join(self.checkpoint_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(data_dict, f)
        print(f"💾 데이터 진행 상황 저장: {filepath}")
    
    def load_data_progress(self, filename='data_progress.pkl'):
        """데이터 처리 진행 상황 로드"""
        filepath = os.path.join(self.checkpoint_dir, filename)
        if os.path.exists(filepath):
            print(f"📂 데이터 진행 상황 로드: {filepath}")
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        return None

# ===========================
# 📊 개선된 데이터 처리
# ===========================

class ImprovedHUBROOMProcessor:
    """진행률 표시가 추가된 데이터 처리 클래스"""
    
    def __init__(self, file_path='data/HUB_0509_TO_0730_DATA.CSV'):
        self.file_path = file_path
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.target_col = 'CURRENT_M16A_3F_JOB_2'
        self.checkpoint_manager = CheckpointManager()
        
        # 물리적 계산을 위한 컬럼 그룹
        self.inflow_cols = [
            'M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2',
            'M14A_3F_TO_HUB_JOB2', 'M14B_7F_TO_HUB_JOB2'
        ]
        
        self.outflow_cols = [
            'M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB',
            'M16A_3F_TO_M14A_3F_JOB', 'M16A_3F_TO_M14B_7F_JOB'
        ]
    
    def analyze_data(self, df):
        """데이터 특성 상세 분석"""
        print("\n" + "="*60)
        print("📈 데이터 특성 분석")
        print("="*60)
        
        total_steps = 8
        current_step = 0
        
        # 1. 기본 정보
        current_step += 1
        print(f"\n[{current_step}/{total_steps}] 기본 정보 분석... ({current_step/total_steps*100:.0f}%)")
        print(f"  - 전체 행 수: {df.shape[0]:,}")
        print(f"  - 전체 컬럼 수: {df.shape[1]}")
        print(f"  - 메모리 사용량: {df.memory_usage().sum() / 1024**2:.2f} MB")
        
        # 2. 결측치 분석
        current_step += 1
        print(f"\n[{current_step}/{total_steps}] 결측치 분석... ({current_step/total_steps*100:.0f}%)")
        missing_ratio = (df.isnull().sum() / len(df)) * 100
        missing_cols = missing_ratio[missing_ratio > 0]
        if len(missing_cols) > 0:
            print(f"  - 결측치 있는 컬럼: {len(missing_cols)}개")
            for col, ratio in missing_cols.head(5).items():
                print(f"    • {col}: {ratio:.2f}%")
        else:
            print("  - ✅ 결측치 없음")
        
        # 3. 타겟 변수 분석
        current_step += 1
        print(f"\n[{current_step}/{total_steps}] 타겟 변수 분석... ({current_step/total_steps*100:.0f}%)")
        if self.target_col in df.columns:
            target = df[self.target_col]
            print(f"  - 타겟: {self.target_col}")
            print(f"  - 평균: {target.mean():.2f}")
            print(f"  - 표준편차: {target.std():.2f}")
            print(f"  - 최소/최대: {target.min():.0f} / {target.max():.0f}")
            print(f"  - 분포: Q1={target.quantile(0.25):.0f}, Q2={target.median():.0f}, Q3={target.quantile(0.75):.0f}")
        
        # 4. 유입/유출 분석
        current_step += 1
        print(f"\n[{current_step}/{total_steps}] 유입/유출 균형 분석... ({current_step/total_steps*100:.0f}%)")
        available_inflow = [col for col in self.inflow_cols if col in df.columns]
        available_outflow = [col for col in self.outflow_cols if col in df.columns]
        
        if available_inflow and available_outflow:
            total_inflow = df[available_inflow].sum().sum()
            total_outflow = df[available_outflow].sum().sum()
            balance = total_inflow - total_outflow
            print(f"  - 총 유입: {total_inflow:,.0f}")
            print(f"  - 총 유출: {total_outflow:,.0f}")
            print(f"  - 순 변화: {balance:+,.0f} ({balance/total_inflow*100:+.1f}%)")
        
        # 나머지 분석들...
        print("\n" + "="*60)
        
        return {
            'total_rows': df.shape[0],
            'total_cols': df.shape[1],
            'missing_ratio': missing_ratio.mean()
        }
    
    def create_sequences_with_progress(self, df, numeric_cols, seq_len=20, pred_len=10, resume_from=0):
        """진행률 표시와 함께 시퀀스 생성"""
        
        print("\n📦 시퀀스 데이터 생성 중...")
        
        X, y, X_physics = [], [], []
        
        # 체크포인트에서 이전 진행 상황 로드
        saved_progress = self.checkpoint_manager.load_data_progress()
        if saved_progress and resume_from > 0:
            print(f"♻️ 이전 진행 상황 복원: {resume_from}개 완료")
            X = saved_progress['X']
            y = saved_progress['y']
            X_physics = saved_progress['X_physics']
        
        available_inflow = [col for col in self.inflow_cols if col in df.columns]
        available_outflow = [col for col in self.outflow_cols if col in df.columns]
        
        total_sequences = len(df) - seq_len - pred_len + 1
        
        # tqdm으로 진행률 표시
        with tqdm(total=total_sequences, desc="시퀀스 생성", initial=resume_from) as pbar:
            for i in range(resume_from, total_sequences):
                # 입력: 과거 20분
                X_seq = df[numeric_cols].iloc[i:i+seq_len].values
                X.append(X_seq)
                
                # 타겟: 10분 후
                if self.target_col in df.columns:
                    y_val = df[self.target_col].iloc[i+seq_len+pred_len-1]
                else:
                    y_val = df[numeric_cols[0]].iloc[i+seq_len+pred_len-1]
                y.append(y_val)
                
                # 물리 데이터
                physics_data = {
                    'current_hubroom': df[numeric_cols[0]].iloc[i+seq_len-1],
                    'inflow_sum': df[available_inflow].iloc[i+seq_len:i+seq_len+pred_len].sum().sum() if available_inflow else 0,
                    'outflow_sum': df[available_outflow].iloc[i+seq_len:i+seq_len+pred_len].sum().sum() if available_outflow else 0,
                }
                X_physics.append(physics_data)
                
                pbar.update(1)
                
                # 100개마다 체크포인트 저장
                if (i + 1) % 100 == 0:
                    self.checkpoint_manager.save_data_progress({
                        'X': X, 'y': y, 'X_physics': X_physics,
                        'progress': i + 1
                    })
                
                # 인터럽트 확인
                if self.checkpoint_manager.interrupted:
                    print(f"\n💾 중단 지점 저장: {i+1}/{total_sequences}")
                    self.checkpoint_manager.save_data_progress({
                        'X': X, 'y': y, 'X_physics': X_physics,
                        'progress': i + 1
                    })
                    sys.exit(0)
        
        print(f"✅ 시퀀스 생성 완료: {len(X)}개")
        
        return np.array(X), np.array(y), X_physics

# ===========================
# 🧠 PatchTST 모델 (TensorFlow)
# ===========================

class PatchTST(keras.Model):
    """PatchTST: 패치 기반 시계열 Transformer"""
    
    def __init__(self, config):
        super(PatchTST, self).__init__()
        
        self.seq_len = config['seq_len']
        self.n_features = config['n_features']
        self.patch_len = config['patch_len']
        self.n_patches = self.seq_len // self.patch_len
        
        self.d_model = config['d_model']
        self.n_heads = config['n_heads']
        self.d_ff = config['d_ff']
        self.n_layers = config['n_layers']
        self.dropout = config['dropout']
        
        # 패치 임베딩
        self.patch_embedding = layers.Dense(self.d_model)
        
        # 위치 인코딩
        self.pos_embedding = self.add_weight(
            name='pos_embedding',
            shape=(1, self.n_patches, self.d_model),
            initializer='random_normal',
            trainable=True
        )
        
        # Transformer 인코더 레이어들
        self.encoder_layers = []
        for _ in range(self.n_layers):
            self.encoder_layers.append(
                layers.TransformerEncoderLayer(
                    d_model=self.d_model,
                    num_heads=self.n_heads,
                    dff=self.d_ff,
                    dropout=self.dropout
                )
            )
        
        # 출력 레이어
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout_layer = layers.Dropout(self.dropout)
        self.dense2 = layers.Dense(64, activation='relu')
        self.output_layer = layers.Dense(1)
    
    def create_patches(self, x):
        """입력을 패치로 분할"""
        batch_size = tf.shape(x)[0]
        
        # (batch, seq_len, features) -> (batch, n_patches, patch_len, features)
        x_reshaped = tf.reshape(x, [batch_size, self.n_patches, self.patch_len, self.n_features])
        
        # (batch, n_patches, patch_len * features)
        x_patches = tf.reshape(x_reshaped, [batch_size, self.n_patches, self.patch_len * self.n_features])
        
        return x_patches
    
    def call(self, x, training=False):
        # 패치 생성
        x_patches = self.create_patches(x)
        
        # 패치 임베딩
        x_embed = self.patch_embedding(x_patches)
        
        # 위치 인코딩 추가
        x_embed = x_embed + self.pos_embedding
        
        # Transformer 인코더 통과
        for encoder_layer in self.encoder_layers:
            x_embed = encoder_layer(x_embed, training=training)
        
        # 출력 생성
        x_flat = self.flatten(x_embed)
        x = self.dense1(x_flat)
        x = self.dropout_layer(x, training=training)
        x = self.dense2(x)
        output = self.output_layer(x)
        
        return tf.squeeze(output, axis=-1)

# ===========================
# 🔬 PatchTST + PINN 하이브리드 모델
# ===========================

class PatchTST_PINN_Hybrid(keras.Model):
    """PatchTST와 Physics-Informed Neural Network 결합 모델"""
    
    def __init__(self, config):
        super(PatchTST_PINN_Hybrid, self).__init__()
        
        # PatchTST 컴포넌트
        self.patch_tst = PatchTST(config)
        
        # 물리 정보 처리 네트워크
        self.physics_dense1 = layers.Dense(32, activation='relu')
        self.physics_dense2 = layers.Dense(16, activation='relu')
        
        # 결합 레이어
        self.combine_dense1 = layers.Dense(64, activation='relu')
        self.combine_dense2 = layers.Dense(32, activation='relu')
        self.final_output = layers.Dense(1)
        
        self.dropout = layers.Dropout(config['dropout'])
    
    def call(self, inputs, training=False):
        # inputs = [x_seq, x_physics]
        x_seq, x_physics = inputs
        
        # PatchTST 출력
        tst_output = self.patch_tst(x_seq, training=training)
        tst_output = tf.expand_dims(tst_output, axis=-1)
        
        # 물리 정보 처리
        physics_output = self.physics_dense1(x_physics)
        physics_output = self.dropout(physics_output, training=training)
        physics_output = self.physics_dense2(physics_output)
        
        # 결합
        combined = tf.concat([tst_output, physics_output], axis=-1)
        x = self.combine_dense1(combined)
        x = self.dropout(x, training=training)
        x = self.combine_dense2(x)
        output = self.final_output(x)
        
        return tf.squeeze(output, axis=-1)

# ===========================
# 📈 개선된 학습 클래스
# ===========================

class ImprovedTrainer:
    """체크포인트와 진행률 표시가 추가된 학습 클래스"""
    
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
        self.checkpoint_manager = CheckpointManager()
        self.history = None
    
    def create_callbacks(self):
        """콜백 생성"""
        callbacks_list = [
            ModelCheckpoint(
                filepath=os.path.join(self.checkpoint_manager.checkpoint_dir, 
                                     f'{self.model_name}_best.h5'),
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        return callbacks_list
    
    def train_with_checkpoint(self, X_train, y_train, X_val, y_val, 
                             epochs=50, batch_size=32, lr=0.001, resume=False):
        """체크포인트 지원 학습"""
        
        # 모델 컴파일
        self.model.compile(
            optimizer=Adam(learning_rate=lr),
            loss='mse',
            metrics=['mae']
        )
        
        # 체크포인트 복원
        initial_epoch = 0
        if resume:
            if self.checkpoint_manager.load_checkpoint(self.model, f'{self.model_name}_checkpoint'):
                print(f"♻️ 학습 재개")
                # 에포크 정보는 별도 파일에서 로드 필요
                try:
                    with open(os.path.join(self.checkpoint_manager.checkpoint_dir, 
                                          f'{self.model_name}_epoch.txt'), 'r') as f:
                        initial_epoch = int(f.read())
                        print(f"  Epoch {initial_epoch}부터 시작")
                except:
                    pass
        
        print(f"\n🚀 {self.model_name} 학습 시작...")
        print("="*60)
        
        # 학습
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            initial_epoch=initial_epoch,
            batch_size=batch_size,
            callbacks=self.create_callbacks(),
            verbose=1
        )
        
        # 최종 체크포인트 저장
        self.checkpoint_manager.save_checkpoint(self.model, f'{self.model_name}_final')
        
        print("✅ 학습 완료!")
        return self.history
    
    def evaluate(self, X_test, y_test, scaler_y=None):
        """모델 평가"""
        print("\n📊 모델 평가 중...")
        
        # 예측
        y_pred_scaled = self.model.predict(X_test, verbose=0)
        
        # 역정규화
        if scaler_y:
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        else:
            y_pred = y_pred_scaled
            y_true = y_test
        
        # 메트릭 계산
        mae = np.mean(np.abs(y_true - y_pred))
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        
        print(f"  - MAE: {mae:.4f}")
        print(f"  - MSE: {mse:.4f}")
        print(f"  - RMSE: {rmse:.4f}")
        
        return {'mae': mae, 'mse': mse, 'rmse': rmse, 'y_true': y_true, 'y_pred': y_pred}

# ===========================
# 🎯 메인 실행
# ===========================

def main_with_progress():
    """진행률 표시와 체크포인트가 추가된 메인 함수"""
    
    print("=" * 80)
    print("🏭 HUBROOM 반송량 예측 시스템 (TensorFlow 2.15)")
    print("💡 특징: 진행률 표시, 중단/재개 가능")
    print("=" * 80)
    
    # 데이터 처리
    processor = ImprovedHUBROOMProcessor()
    
    try:
        # 1. 데이터 로드
        print("\n📂 데이터 로드 중...")
        df = pd.read_csv(processor.file_path)
        print(f"✅ 데이터 로드 완료: {df.shape[0]:,} 행 × {df.shape[1]} 컬럼")
        
        # 2. 데이터 특성 분석
        data_stats = processor.analyze_data(df)
        
        # 3. 전처리
        print("\n🔧 데이터 전처리 중...")
        time_col = df.columns[0]
        df['timestamp'] = pd.to_datetime(df[time_col], format='%Y%m%d%H%M', errors='coerce')
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = df.fillna(method='ffill').fillna(0)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"✅ 전처리 완료: {len(numeric_cols)}개 숫자형 컬럼")
        
        # 4. 시퀀스 생성
        saved_progress = processor.checkpoint_manager.load_data_progress()
        resume_from = saved_progress['progress'] if saved_progress else 0
        
        X, y, physics_data = processor.create_sequences_with_progress(
            df, numeric_cols, seq_len=20, pred_len=10, resume_from=resume_from
        )
        
        print(f"\n📊 데이터 준비 완료:")
        print(f"  - X shape: {X.shape}")
        print(f"  - y shape: {y.shape}")
        
        # 5. 데이터 분할
        print("\n🔀 데이터 분할 중...")
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        # 물리 데이터도 분할
        physics_train = physics_data[:len(X_train)]
        physics_val = physics_data[len(X_train):len(X_train)+len(X_val)]
        physics_test = physics_data[len(X_train)+len(X_val):]
        
        print(f"  - Train: {len(X_train):,} samples")
        print(f"  - Valid: {len(X_val):,} samples")
        print(f"  - Test: {len(X_test):,} samples")
        
        # 6. 정규화
        print("\n📏 데이터 정규화 중...")
        n_samples_train, seq_len, n_features = X_train.shape
        
        X_train_scaled = processor.scaler_X.fit_transform(X_train.reshape(-1, n_features)).reshape(n_samples_train, seq_len, n_features)
        X_val_scaled = processor.scaler_X.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape[0], seq_len, n_features)
        X_test_scaled = processor.scaler_X.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape[0], seq_len, n_features)
        
        y_train_scaled = processor.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = processor.scaler_y.transform(y_val.reshape(-1, 1)).flatten()
        y_test_scaled = processor.scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        
        # 7. 모델 선택
        print("\n🤖 모델 선택")
        print("1. PatchTST")
        print("2. PatchTST + PINN 하이브리드")
        model_choice = input("선택 (1 또는 2): ")
        
        config = {
            'seq_len': 20,
            'n_features': n_features,
            'patch_len': 5,
            'd_model': 128,
            'n_heads': 8,
            'd_ff': 256,
            'n_layers': 3,
            'dropout': 0.1
        }
        
        if model_choice == '2':
            # 물리 데이터 준비
            X_physics_train = np.array([[p['current_hubroom'], p['inflow_sum'], p['outflow_sum']] 
                                       for p in physics_train])
            X_physics_val = np.array([[p['current_hubroom'], p['inflow_sum'], p['outflow_sum']] 
                                     for p in physics_val])
            X_physics_test = np.array([[p['current_hubroom'], p['inflow_sum'], p['outflow_sum']] 
                                      for p in physics_test])
            
            # 물리 데이터 정규화
            scaler_physics = StandardScaler()
            X_physics_train_scaled = scaler_physics.fit_transform(X_physics_train)
            X_physics_val_scaled = scaler_physics.transform(X_physics_val)
            X_physics_test_scaled = scaler_physics.transform(X_physics_test)
            
            # 하이브리드 모델
            model = PatchTST_PINN_Hybrid(config)
            trainer = ImprovedTrainer(model, "PatchTST_PINN")
            
            # 학습 데이터 준비
            train_data = [X_train_scaled, X_physics_train_scaled]
            val_data = [X_val_scaled, X_physics_val_scaled]
            test_data = [X_test_scaled, X_physics_test_scaled]
        else:
            # 기본 PatchTST 모델
            model = PatchTST(config)
            trainer = ImprovedTrainer(model, "PatchTST")
            
            train_data = X_train_scaled
            val_data = X_val_scaled
            test_data = X_test_scaled
        
        # 8. 학습
        resume_training = input("\n이전 학습을 이어서 하시겠습니까? (y/n): ").lower() == 'y'
        
        history = trainer.train_with_checkpoint(
            train_data, y_train_scaled,
            val_data, y_val_scaled,
            epochs=30,
            batch_size=32,
            lr=0.001,
            resume=resume_training
        )
        
        # 9. 평가
        results = trainer.evaluate(test_data, y_test_scaled, processor.scaler_y)
        
        print("\n✅ 모든 작업 완료!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자 중단 감지")
        print("💾 진행 상황이 저장되었습니다. 다시 실행하면 이어서 진행됩니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        print("💾 진행 상황이 저장되었습니다.")

if __name__ == "__main__":
    main_with_progress()