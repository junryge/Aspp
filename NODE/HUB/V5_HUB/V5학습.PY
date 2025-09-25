#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
🎯 HUBROOM 점프 감지 80% → 81% 시스템 - 완전 통합 학습 코드
================================================================================
모델 구성:
1. 기존 모델들 (ExtraTrees, XGBoost, RandomForest)
2. PatchTST (안정형)
3. PatchTST+PINN (극단형)
4. ExtraTrees + 규칙
5. ExtraTrees + PatchTST 하이브리드
================================================================================
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, ExtraTreesRegressor
from sklearn.preprocessing import RobustScaler, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os
import pickle
import signal
import sys
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🎯 HUBROOM 점프 감지 80% → 81% 시스템")
print("📊 모든 모델 포함: ExtraTrees, PatchTST, 하이브리드")
print("✅ 중단/재개 지원 - Ctrl+C로 언제든 중단 가능!")
print("="*80)

# ==============================================================================
# 💾 체크포인트 관리자
# ==============================================================================

class CheckpointManager:
    """학습 중단/재개를 위한 체크포인트 관리"""
    
    def __init__(self, checkpoint_dir='./checkpoints_jump80'):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.state_file = os.path.join(checkpoint_dir, 'training_state.pkl')
        self.data_file = os.path.join(checkpoint_dir, 'processed_data.pkl')
        self.sequences_file = os.path.join(checkpoint_dir, 'sequences.pkl')
        self.models_dir = os.path.join(checkpoint_dir, 'models')
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.interrupted = False
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """Ctrl+C 시그널 처리"""
        print('\n\n⚠️ 중단 감지! 현재 상태를 저장합니다...')
        self.interrupted = True
    
    def save_state(self, state):
        """현재 상태 저장"""
        with open(self.state_file, 'wb') as f:
            pickle.dump(state, f)
        print(f"💾 상태 저장 완료: Step {state.get('step', 0)}")
    
    def load_state(self):
        """저장된 상태 로드"""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def save_data(self, data_dict):
        """처리된 데이터 저장"""
        with open(self.data_file, 'wb') as f:
            pickle.dump(data_dict, f)
        print("💾 데이터 저장 완료")
    
    def load_data(self):
        """저장된 데이터 로드"""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def save_sequences(self, sequences_dict):
        """시퀀스 데이터 저장"""
        with open(self.sequences_file, 'wb') as f:
            pickle.dump(sequences_dict, f)
        print("💾 시퀀스 저장 완료")
    
    def load_sequences(self):
        """시퀀스 데이터 로드"""
        if os.path.exists(self.sequences_file):
            with open(self.sequences_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def save_model(self, model, model_name):
        """모델 저장"""
        model_path = os.path.join(self.models_dir, f'{model_name}.pkl')
        joblib.dump(model, model_path)
        print(f"💾 {model_name} 모델 저장 완료")
    
    def save_keras_model(self, model, model_name):
        """케라스 모델 저장"""
        model_path = os.path.join(self.models_dir, f'{model_name}.h5')
        model.save_weights(model_path)
        print(f"💾 {model_name} 케라스 모델 저장 완료")
    
    def load_model(self, model_name):
        """모델 로드"""
        model_path = os.path.join(self.models_dir, f'{model_name}.pkl')
        if os.path.exists(model_path):
            return joblib.load(model_path)
        return None
    
    def clear_state(self):
        """상태 초기화"""
        if os.path.exists(self.state_file):
            os.remove(self.state_file)
        if os.path.exists(self.data_file):
            os.remove(self.data_file)
        if os.path.exists(self.sequences_file):
            os.remove(self.sequences_file)
        print("🧹 이전 상태 제거 완료")

# ==============================================================================
# 📊 데이터 처리 클래스
# ==============================================================================

class HubRoomDataProcessor:
    """완전한 데이터 처리 - 모든 특징 포함"""
    
    def __init__(self):
        self.target_col = 'CURRENT_M16A_3F_JOB_2'
        
        # 21개 필수 컬럼
        self.inflow_cols = [
            'M16A_6F_TO_HUB_JOB',
            'M16A_2F_TO_HUB_JOB2',
            'M14A_3F_TO_HUB_JOB2',
            'M14B_7F_TO_HUB_JOB2',
            'M16B_10F_TO_HUB_JOB'
        ]
        
        self.outflow_cols = [
            'M16A_3F_TO_M16A_6F_JOB',
            'M16A_3F_TO_M16A_2F_JOB',
            'M16A_3F_TO_M14A_3F_JOB',
            'M16A_3F_TO_M14B_7F_JOB',
            'M16A_3F_TO_3F_MLUD_JOB'
        ]
        
        self.cmd_cols = [
            'M16A_3F_CMD',
            'M16A_6F_TO_HUB_CMD',
            'M16A_2F_TO_HUB_CMD',
            'M14A_3F_TO_HUB_CMD',
            'M14B_7F_TO_HUB_CMD'
        ]
        
        self.capa_cols = [
            'M16A_6F_LFT_MAXCAPA',
            'M16A_2F_LFT_MAXCAPA'
        ]
        
        self.other_cols = [
            'M16A_3F_STORAGE_UTIL',
            'M14_TO_M16_OFS_CUR',
            'M16_TO_M14_OFS_CUR'
        ]
        
        # 확률 맵 - 매우 중요!
        self.probability_map = {
            0: 0.003, 1: 0.15, 2: 0.25, 3: 0.31, 4: 0.43, 5: 0.43,
            6: 0.35, 7: 0.42, 8: 0.53, 9: 0.49, 10: 0.42,
            11: 0.47, 12: 0.52, 13: 0.60, 14: 0.54, 15: 0.66,
            16: 0.62, 17: 0.71, 18: 0.79, 19: 0.83, 20: 0.987,
            21: 0.99, 22: 0.99, 23: 0.99, 24: 0.99, 25: 0.99,
            26: 0.99, 27: 0.99, 28: 0.99, 29: 0.99, 30: 0.99
        }
    
    def load_and_merge_data(self):
        """데이터 로드 및 BRIDGE_TIME 병합"""
        print("\n[1단계] 데이터 로드")
        
        # 메인 데이터
        df = pd.read_csv('data/HUB_0509_to_0807_DATA.CSV')
        print(f"✅ 메인 데이터: {df.shape}")
        
        # 시간 처리
        time_col = df.columns[0]
        df['datetime'] = pd.to_datetime(df[time_col], format='%Y%m%d%H%M')
        
        # BRIDGE_TIME 데이터
        bridge_df = pd.read_csv('data/BRTIME_0509_TO_0807.CSV')
        print(f"✅ BRIDGE_TIME 데이터: {bridge_df.shape}")
        
        # BRIDGE_TIME 병합 - 시간대 문제 해결
        if 'IDC_VAL' in bridge_df.columns:
            bridge_df['BRIDGE_TIME'] = bridge_df['IDC_VAL']
            bridge_df['datetime'] = pd.to_datetime(bridge_df['CRT_TM'])
            
            # 시간대 정보 제거
            if hasattr(bridge_df['datetime'].dtype, 'tz'):
                bridge_df['datetime'] = bridge_df['datetime'].dt.tz_localize(None)
            if hasattr(df['datetime'].dtype, 'tz'):
                df['datetime'] = df['datetime'].dt.tz_localize(None)
            
            # 분 단위로 반올림
            bridge_df['datetime'] = bridge_df['datetime'].dt.floor('min')
            df['datetime'] = df['datetime'].dt.floor('min')
            
            # 병합
            df = pd.merge(df, bridge_df[['datetime', 'BRIDGE_TIME']], 
                         on='datetime', how='left')
            
            # BRIDGE_TIME 보간 및 결측치 처리
            df['BRIDGE_TIME'] = df['BRIDGE_TIME'].interpolate(method='linear', limit_direction='both')
            df['BRIDGE_TIME'] = df['BRIDGE_TIME'].fillna(3.5)
            
            print(f"✅ BRIDGE_TIME 병합 완료: {df['BRIDGE_TIME'].notna().sum()}/{len(df)} 매칭")
        
        return df
    
    def create_all_features(self, df):
        """완전한 특징 엔지니어링 - 누락 없음!"""
        print("\n[2단계] 특징 엔지니어링")
        
        # 1. 유입/유출 밸런스
        df['flow_balance'] = df[self.inflow_cols].sum(axis=1) - df[self.outflow_cols].sum(axis=1)
        df['flow_ratio'] = df[self.inflow_cols].sum(axis=1) / (df[self.outflow_cols].sum(axis=1) + 1)
        
        # 2. 추세 특징
        df['trend_20min'] = df[self.target_col].diff(20)
        df['trend_10min'] = df[self.target_col].diff(10)
        df['acceleration'] = df['trend_10min'] - df['trend_10min'].shift(10)
        
        # 3. 연속 패턴
        df['consecutive_250+'] = (df[self.target_col] > 250).rolling(10).sum()
        df['consecutive_270+'] = (df[self.target_col] > 270).rolling(10).sum()
        
        # 4. CMD 동기화
        df['cmd_sync_count'] = (df[self.cmd_cols] > 235).sum(axis=1)
        df['cmd_max'] = df[self.cmd_cols].max(axis=1)
        
        # 5. 브릿지타임 변화
        df['bridge_diff'] = df['BRIDGE_TIME'].diff(5)
        df['bridge_high'] = (df['BRIDGE_TIME'] > 4.0).astype(int)
        
        # 6. storage x bridge 상호작용
        df['storage_x_bridge'] = df['M16A_3F_STORAGE_UTIL'] * df['BRIDGE_TIME']
        
        # 7. 연속 300+ 카운트와 확률
        consecutive_300_counts = []
        consecutive_300_probs = []
        
        for i in tqdm(range(len(df)), desc="300+ 패턴 계산"):
            if i < 30:
                count = 0
                prob = 0.003
            else:
                window = df[self.target_col].iloc[i-30:i].values
                count = sum(1 for v in window if v >= 300)
                prob = self.probability_map.get(count, 0.5)
            
            consecutive_300_counts.append(count)
            consecutive_300_probs.append(prob)
        
        df['consecutive_300_count'] = consecutive_300_counts
        df['consecutive_300_prob'] = consecutive_300_probs
        
        # 8. 3구간 분류 (숫자로 직접 변환)
        conditions = [
            df[self.target_col] < 150,
            (df[self.target_col] >= 150) & (df[self.target_col] < 300),
            df[self.target_col] >= 300
        ]
        choices = [0, 1, 2]
        df['range_class'] = np.select(conditions, choices, default=1)
        
        # 9. 점프 여부
        df['past_30min_max'] = df[self.target_col].rolling(30).max()
        df['is_jump'] = ((df['past_30min_max'].shift(10) < 280) & 
                        (df[self.target_col] >= 300)).astype(int)
        
        # 10. 상승/하락 패턴 (숫자로 변환)
        df['change_20min'] = df[self.target_col] - df[self.target_col].shift(20)
        
        trend_conditions = [
            df['change_20min'] < -20,
            (df['change_20min'] >= -20) & (df['change_20min'] < 20),
            (df['change_20min'] >= 20) & (df['change_20min'] < 50),
            df['change_20min'] >= 50
        ]
        trend_choices = [0, 1, 2, 3]  # 0:down, 1:stable, 2:gradual_up, 3:rapid_up
        df['trend_pattern'] = np.select(trend_conditions, trend_choices, default=1)
        
        # 11. 상승률/하락률 (%)
        df['change_rate_10min'] = ((df[self.target_col] - df[self.target_col].shift(10)) / 
                                   (df[self.target_col].shift(10) + 1)) * 100
        df['change_rate_20min'] = ((df[self.target_col] - df[self.target_col].shift(20)) / 
                                   (df[self.target_col].shift(20) + 1)) * 100
        df['change_rate_30min'] = ((df[self.target_col] - df[self.target_col].shift(30)) / 
                                   (df[self.target_col].shift(30) + 1)) * 100
        
        # 12. 변동성
        df['volatility_10min'] = df[self.target_col].rolling(10).std()
        df['volatility_20min'] = df[self.target_col].rolling(20).std()
        df['volatility_30min'] = df[self.target_col].rolling(30).std()
        
        # 13. 극단값 근접도
        df['distance_to_300'] = 300 - df[self.target_col]
        df['near_extreme'] = (df[self.target_col] > 280).astype(int)
        
        # 14. 최근 통계
        df['recent_5min_mean'] = df[self.target_col].rolling(5).mean()
        df['recent_5min_max'] = df[self.target_col].rolling(5).max()
        df['recent_10min_mean'] = df[self.target_col].rolling(10).mean()
        
        # 15. 277 구간 특별 지표
        df['in_jump_zone'] = ((df[self.target_col] >= 275) & (df[self.target_col] <= 279)).astype(int)
        
        # NaN 처리
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(method='ffill').fillna(0)
        
        print(f"✅ 총 {len(df.columns)}개 특징 생성 완료")
        return df
    
    def create_sequences(self, df, seq_len=30, pred_len=10, ckpt=None):
        """시퀀스 데이터 생성 - 중단/재개 지원"""
        print(f"\n[3단계] 시퀀스 생성 ({seq_len}분 → {pred_len}분 후)")
        
        # 특징 컬럼 선택
        feature_cols = [col for col in df.columns 
                       if col not in ['datetime', 'range_class', 'is_jump', 'trend_pattern']]
        
        X, y, y_jump, y_range, y_trend, y_value = [], [], [], [], [], []
        
        total_sequences = len(df) - seq_len - pred_len
        
        for i in tqdm(range(seq_len, len(df) - pred_len), desc="시퀀스 생성"):
            if ckpt and ckpt.interrupted:
                print("\n💾 시퀀스 생성 중단! 현재까지 저장...")
                sequences_dict = {
                    'X': np.array(X),
                    'y': np.array(y),
                    'y_jump': np.array(y_jump),
                    'y_range': np.array(y_range),
                    'y_trend': np.array(y_trend),
                    'y_value': np.array(y_value),
                    'progress': i,
                    'feature_cols': feature_cols
                }
                ckpt.save_sequences(sequences_dict)
                sys.exit(0)
            
            # 입력: 과거 30분
            X.append(df[feature_cols].iloc[i-seq_len:i].values)
            
            # 타겟들
            target_idx = i + pred_len - 1
            y.append(df[self.target_col].iloc[target_idx])
            y_jump.append(df['is_jump'].iloc[target_idx])
            y_range.append(df['range_class'].iloc[target_idx])
            y_trend.append(df['trend_pattern'].iloc[target_idx])
            y_value.append(df[self.target_col].iloc[target_idx])
        
        print(f"✅ {len(X)}개 시퀀스 생성 완료")
        
        return (np.array(X), np.array(y), np.array(y_jump), 
                np.array(y_range), np.array(y_trend), np.array(y_value), feature_cols)

# ==============================================================================
# 🤖 PatchTST 모델 정의
# ==============================================================================

class PatchTSTModel(keras.Model):
    """안정 구간 전문 PatchTST"""
    
    def __init__(self, config):
        super().__init__()
        
        self.seq_len = config['seq_len']
        self.n_features = config['n_features']
        self.patch_len = config['patch_len']
        self.n_patches = self.seq_len // self.patch_len
        
        # 패치 임베딩
        self.patch_embedding = layers.Dense(128, activation='relu')
        
        # Transformer 블록
        self.attention = layers.MultiHeadAttention(num_heads=8, key_dim=16)
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        
        # Feed Forward
        self.ffn = keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(128)
        ])
        
        # 출력층
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout = layers.Dropout(0.3)
        self.dense2 = layers.Dense(64, activation='relu')
        self.output_layer = layers.Dense(1)
    
    def call(self, x, training=False):
        batch_size = tf.shape(x)[0]
        
        # 패치화
        x = tf.reshape(x, [batch_size, self.n_patches, self.patch_len * self.n_features])
        x = self.patch_embedding(x)
        
        # Transformer
        attn = self.attention(x, x, training=training)
        x = self.norm1(x + attn)
        
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        # 출력
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        output = self.output_layer(x)
        
        return tf.squeeze(output, axis=-1)

class PatchTSTPINN(keras.Model):
    """극단값 전문 PatchTST + PINN"""
    
    def __init__(self, config):
        super().__init__()
        
        self.seq_len = config['seq_len']
        self.n_features = config['n_features']
        self.patch_len = config['patch_len']
        self.n_patches = self.seq_len // self.patch_len
        
        # PatchTST 부분
        self.patch_embedding = layers.Dense(128, activation='relu')
        self.attention = layers.MultiHeadAttention(num_heads=8, key_dim=16)
        self.norm = layers.LayerNormalization()
        
        self.flatten = layers.Flatten()
        self.temporal_dense = layers.Dense(64, activation='relu')
        
        # 물리 정보 처리
        self.physics_net = keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu')
        ])
        
        # 융합
        self.fusion = keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])
        
        # 극단값 부스트
        self.extreme_boost = layers.Dense(1, activation='sigmoid')
    
    def call(self, inputs, training=False):
        x_seq, x_physics = inputs
        batch_size = tf.shape(x_seq)[0]
        
        # PatchTST 처리
        x = tf.reshape(x_seq, [batch_size, self.n_patches, self.patch_len * self.n_features])
        x = self.patch_embedding(x)
        
        attn = self.attention(x, x, training=training)
        x = self.norm(x + attn)
        
        x = self.flatten(x)
        x_temporal = self.temporal_dense(x)
        
        # 물리 정보 처리
        x_physics = self.physics_net(x_physics)
        
        # 융합
        x_combined = tf.concat([x_temporal, x_physics], axis=-1)
        output = self.fusion(x_combined)
        
        # 극단값 부스팅
        boost_factor = self.extreme_boost(x_physics)
        output = output * (1 + boost_factor * 0.1)
        
        return tf.squeeze(output, axis=-1)

# ==============================================================================
# 🤖 모델 시스템 (기존 + 신규)
# ==============================================================================

class JumpDetectionSystem:
    """점프 감지 80% 달성 시스템 - 모든 모델 포함"""
    
    def __init__(self):
        # 기존 모델들
        self.model_screening = ExtraTreesClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        self.model_jump = XGBClassifier(
            n_estimators=300,
            max_depth=10,
            learning_rate=0.1,
            scale_pos_weight=100,
            random_state=42
        )
        
        self.model_range = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        self.model_trend = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        self.model_value = ExtraTreesRegressor(
            n_estimators=500,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        
        # ExtraTrees + 규칙 (점프 전용)
        self.model_extra_rules = ExtraTreesClassifier(
            n_estimators=700,
            max_depth=25,
            min_samples_split=3,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # PatchTST 모델들
        self.model_patchtst = None
        self.model_patchtst_pinn = None
        
        # 스케일러
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.scaler_physics = StandardScaler()
        
        self.feature_names = None
        self.feature_indices = {}
    
    def build_patchtst_models(self, n_features):
        """PatchTST 모델 생성"""
        config = {
            'seq_len': 30,
            'n_features': n_features,
            'patch_len': 6  # 5개 패치
        }
        
        # 안정형 모델
        self.model_patchtst = PatchTSTModel(config)
        
        # 극단형 모델
        self.model_patchtst_pinn = PatchTSTPINN(config)
        
        print("✅ PatchTST 모델 생성 완료")
    
    def prepare_features(self, X_seq):
        """시퀀스를 특징으로 변환"""
        # 마지막 시점 특징
        last_features = X_seq[:, -1, :]
        
        # 통계 특징
        mean_features = np.mean(X_seq, axis=1)
        std_features = np.std(X_seq, axis=1)
        max_features = np.max(X_seq, axis=1)
        min_features = np.min(X_seq, axis=1)
        
        # 추세 특징
        trend_features = X_seq[:, -1, :] - X_seq[:, 0, :]
        
        # 모든 특징 결합
        features = np.hstack([
            last_features,
            mean_features,
            std_features,
            max_features,
            min_features,
            trend_features
        ])
        
        return features
    
    def create_physics_features(self, X_seq, feature_cols):
        """물리 특징 생성 (PatchTST-PINN용)"""
        # 특징 인덱스 찾기
        idx_storage = None
        idx_bridge = None
        idx_flow_balance = None
        
        for i, col in enumerate(feature_cols):
            if 'M16A_3F_STORAGE_UTIL' in col:
                idx_storage = i
            elif col == 'BRIDGE_TIME':
                idx_bridge = i
            elif 'flow_balance' in col:
                idx_flow_balance = i
        
        physics_features = []
        
        for seq in X_seq:
            features = []
            
            # 마지막 시점 값들
            if idx_storage is not None:
                features.append(seq[-1, idx_storage])
            else:
                features.append(0)
                
            if idx_bridge is not None:
                features.append(seq[-1, idx_bridge])
            else:
                features.append(3.5)
                
            if idx_flow_balance is not None:
                features.append(seq[-1, idx_flow_balance])
            else:
                features.append(0)
            
            # 추세
            features.append(np.mean(seq[-10:, 0]) - np.mean(seq[:10, 0]))  # target 추세
            features.append(np.std(seq[:, 0]))  # 변동성
            
            physics_features.append(features)
        
        return np.array(physics_features)
    
    def get_feature_indices(self, feature_cols):
        """중요 특징들의 인덱스 찾기"""
        indices = {}
        for i, col in enumerate(feature_cols):
            if 'M16A_3F_STORAGE_UTIL' in col:
                indices['storage_util'] = i
            elif col == 'BRIDGE_TIME':
                indices['bridge_time'] = i
            elif 'flow_balance' in col:
                indices['flow_balance'] = i
            elif 'consecutive_250+' in col:
                indices['consecutive_250'] = i
            elif 'cmd_sync_count' in col:
                indices['cmd_sync'] = i
            elif 'trend_20min' in col:
                indices['trend_20min'] = i
            elif 'flow_ratio' in col:
                indices['flow_ratio'] = i
            elif 'acceleration' in col:
                indices['acceleration'] = i
            elif 'cmd_max' in col:
                indices['cmd_max'] = i
            elif 'bridge_high' in col:
                indices['bridge_high'] = i
            elif 'consecutive_300_prob' in col:
                indices['prob_extreme'] = i
        
        self.feature_indices = indices
        return indices
    
    def train_jump_detector(self, X, y_jump):
        """기존 점프 감지 모델 학습"""
        print("\n🎯 점프 감지 모델 학습 (XGBoost)")
        
        # SMOTE로 불균형 처리
        jump_count = np.sum(y_jump)
        print(f"  점프 케이스: {jump_count}/{len(y_jump)} ({jump_count/len(y_jump)*100:.2f}%)")
        
        smote = SMOTE(sampling_strategy=0.2, random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y_jump)
        
        print(f"  SMOTE 후: {np.sum(y_balanced)}/{len(y_balanced)} ({np.sum(y_balanced)/len(y_balanced)*100:.2f}%)")
        
        # 학습
        self.model_jump.fit(X_balanced, y_balanced)
        
        return self.model_jump
    
    def train_extra_rules_jump(self, X, y_jump):
        """ExtraTrees + 규칙 점프 모델 학습"""
        print("\n🌳 ExtraTrees + 규칙 점프 모델 학습")
        
        # 더 공격적인 SMOTE
        smote = SMOTE(sampling_strategy=0.3, random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y_jump)
        
        # 학습
        self.model_extra_rules.fit(X_balanced, y_balanced)
        
        # 특징 중요도
        importance = pd.DataFrame({
            'feature_idx': range(X.shape[1]),
            'importance': self.model_extra_rules.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n✅ ExtraTrees 상위 10개 특징:")
        for idx, row in importance.head(10).iterrows():
            print(f"  특징 {int(row['feature_idx'])}: {row['importance']:.4f}")
        
        return self.model_extra_rules
    
    def apply_rule_based_boost(self, X, predictions, prob_scores=None):
        """규칙 기반 부스팅 - 80% 달성용"""
        boosted_predictions = predictions.copy()
        
        if not self.feature_indices:
            return boosted_predictions
        
        idx = self.feature_indices
        
        # Phase 1: 강한 신호 (55%)
        if 'storage_util' in idx:
            strong_signal = X[:, idx['storage_util']] > 20
            boosted_predictions[strong_signal] = 1
        
        # Phase 2: 중간 신호 (70%)
        medium_conditions = []
        
        if 'storage_util' in idx:
            medium_conditions.append(X[:, idx['storage_util']] > 10)
        if 'flow_balance' in idx:
            medium_conditions.append(X[:, idx['flow_balance']] > 50)
        if 'bridge_time' in idx:
            medium_conditions.append(X[:, idx['bridge_time']] > 4.0)
        if 'consecutive_250' in idx:
            medium_conditions.append(X[:, idx['consecutive_250']] >= 5)
        if 'cmd_sync' in idx:
            medium_conditions.append(X[:, idx['cmd_sync']] >= 3)
        if 'trend_20min' in idx:
            medium_conditions.append(X[:, idx['trend_20min']] > 30)
        
        if len(medium_conditions) >= 2:
            medium_signal = np.sum(medium_conditions, axis=0) >= 2
            if 'storage_util' in idx:
                storage_medium = X[:, idx['storage_util']] > 10
                final_medium = medium_signal & storage_medium & (boosted_predictions == 0)
                boosted_predictions[final_medium] = 1
        
        # Phase 3: 약한 신호 (80%)
        weak_conditions = []
        
        if 'bridge_time' in idx:
            weak_conditions.append(X[:, idx['bridge_time']] > 3.85)
        if 'flow_ratio' in idx:
            weak_conditions.append(X[:, idx['flow_ratio']] > 1.5)
        if 'acceleration' in idx:
            weak_conditions.append(X[:, idx['acceleration']] > 20)
        if 'cmd_max' in idx:
            weak_conditions.append(X[:, idx['cmd_max']] > 238)
        if 'consecutive_250' in idx:
            weak_conditions.append(X[:, idx['consecutive_250']] >= 3)
        
        if len(weak_conditions) >= 3:
            weak_signal = np.sum(weak_conditions, axis=0) >= 3
            final_weak = weak_signal & (boosted_predictions == 0)
            
            if prob_scores is not None:
                final_weak = final_weak & (prob_scores > 0.3)
            
            boosted_predictions[final_weak] = 1
        
        return boosted_predictions

# ==============================================================================
# 🎯 하이브리드 예측기
# ==============================================================================

class HybridJumpPredictor:
    """ExtraTrees + PatchTST 하이브리드 - 81% 목표"""
    
    def __init__(self, system):
        self.system = system
    
    def predict(self, X_seq, X_features, X_physics=None):
        """하이브리드 예측"""
        predictions = []
        
        # 1. ExtraTrees 확률
        if hasattr(self.system.model_extra_rules, 'predict_proba'):
            et_proba = self.system.model_extra_rules.predict_proba(X_features)[:, 1]
        else:
            et_proba = self.system.model_extra_rules.predict(X_features)
        
        for i in range(len(X_seq)):
            # ExtraTrees 확률 확인
            if et_proba[i] > 0.8:
                # 확실한 점프
                predictions.append(1)
            elif et_proba[i] < 0.2:
                # 확실한 비점프
                predictions.append(0)
            else:
                # 애매한 경우 - PatchTST 사용
                if self.system.model_patchtst_pinn is not None and X_physics is not None:
                    # 극단형 모델로 정밀 예측
                    x_seq_single = X_seq[i:i+1]
                    x_physics_single = X_physics[i:i+1]
                    
                    # 스케일링
                    x_seq_scaled = self.system.scaler_X.transform(
                        x_seq_single.reshape(-1, x_seq_single.shape[-1])
                    ).reshape(x_seq_single.shape)
                    x_physics_scaled = self.system.scaler_physics.transform(x_physics_single)
                    
                    # 예측
                    pred_value = self.system.model_patchtst_pinn.predict(
                        [x_seq_scaled, x_physics_scaled], 
                        verbose=0
                    )[0]
                    
                    # 300+ 예측이면 점프
                    if pred_value > 300:
                        predictions.append(1)
                    else:
                        predictions.append(0)
                else:
                    # PatchTST 없으면 확률 기반
                    predictions.append(1 if et_proba[i] > 0.5 else 0)
        
        return np.array(predictions)

# ==============================================================================
# 📊 평가 함수
# ==============================================================================

def evaluate_jump_detection(y_true, y_pred, model_name="Model"):
    """점프 감지 성능 평가"""
    print(f"\n📊 {model_name} 점프 감지 성능")
    print("-" * 50)
    
    total = len(y_true)
    jump_true = np.sum(y_true)
    jump_pred = np.sum(y_pred)
    
    print(f"전체 샘플: {total}")
    print(f"실제 점프: {jump_true} ({jump_true/total*100:.2f}%)")
    print(f"예측 점프: {jump_pred} ({jump_pred/total*100:.2f}%)")
    
    # 점프 감지율 (Recall)
    if jump_true > 0:
        tp = np.sum((y_true == 1) & (y_pred == 1))
        recall = tp / jump_true
        print(f"\n🎯 점프 감지율: {recall*100:.1f}% ({tp}/{jump_true})")
    else:
        recall = 0
    
    # False Positive Rate
    non_jump_true = total - jump_true
    if non_jump_true > 0:
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fpr = fp / non_jump_true
        print(f"False Positive Rate: {fpr*100:.1f}% ({fp}/{non_jump_true})")
    else:
        fpr = 0
    
    # 정확도
    accuracy = np.sum(y_true == y_pred) / total
    print(f"\n전체 정확도: {accuracy*100:.1f}%")
    
    return {
        'recall': recall,
        'fpr': fpr,
        'accuracy': accuracy
    }

# ==============================================================================
# 🚀 메인 실행
# ==============================================================================

def main():
    """메인 학습 프로세스 - 중단/재개 지원"""
    
    # 체크포인트 매니저 초기화
    ckpt = CheckpointManager()
    
    # 이전 상태 확인
    state = ckpt.load_state()
    if state:
        print(f"\n📂 이전 학습 상태 발견! (Step {state.get('step', 1)}/10)")
        resume = input("이어서 진행하시겠습니까? (y/n): ").lower()
        
        if resume != 'y':
            ckpt.clear_state()
            state = {}
            step = 1
        else:
            step = state.get('step', 1)
            print(f"✅ Step {step}부터 재개합니다.")
    else:
        state = {}
        step = 1
    
    processor = HubRoomDataProcessor()
    system = JumpDetectionSystem()
    
    try:
        # Step 1: 데이터 로드
        if step <= 1:
            print(f"\n[Step 1/10] 데이터 로드 및 병합")
            print("-"*60)
            
            df = processor.load_and_merge_data()
            
            state['step'] = 2
            state['data_shape'] = df.shape
            ckpt.save_state(state)
            
            if ckpt.interrupted:
                print("\n💾 Step 1 저장 후 종료")
                sys.exit(0)
        
        # Step 2: 특징 생성
        if step <= 2:
            print(f"\n[Step 2/10] 특징 엔지니어링")
            print("-"*60)
            
            if step < 2:
                df = processor.load_and_merge_data()
            
            df = processor.create_all_features(df)
            
            # 데이터 저장
            ckpt.save_data({'df': df})
            
            state['step'] = 3
            state['feature_count'] = len(df.columns)
            ckpt.save_state(state)
            
            if ckpt.interrupted:
                print("\n💾 Step 2 저장 후 종료")
                sys.exit(0)
        
        # Step 3: 시퀀스 생성
        if step <= 3:
            print(f"\n[Step 3/10] 시퀀스 생성")
            print("-"*60)
            
            # 데이터 로드
            saved_data = ckpt.load_data()
            if saved_data:
                df = saved_data['df']
            else:
                df = processor.load_and_merge_data()
                df = processor.create_all_features(df)
            
            # 시퀀스 생성
            X, y, y_jump, y_range, y_trend, y_value, feature_cols = processor.create_sequences(df, ckpt=ckpt)
            
            # 시퀀스 저장
            sequences_dict = {
                'X': X,
                'y': y,
                'y_jump': y_jump,
                'y_range': y_range,
                'y_trend': y_trend,
                'y_value': y_value,
                'feature_cols': feature_cols,
                'progress': len(X)
            }
            ckpt.save_sequences(sequences_dict)
            
            # 특징 인덱스 저장
            system.get_feature_indices(feature_cols)
            
            state['step'] = 4
            state['sequence_shape'] = X.shape
            state['n_features'] = X.shape[2]
            ckpt.save_state(state)
            
            if ckpt.interrupted:
                print("\n💾 Step 3 저장 후 종료")
                sys.exit(0)
        
        # Step 4: 데이터 분할
        if step <= 4:
            print(f"\n[Step 4/10] 학습/검증 데이터 분할")
            print("-"*60)
            
            # 시퀀스 로드
            saved_seq = ckpt.load_sequences()
            X = saved_seq['X']
            y = saved_seq['y']
            y_jump = saved_seq['y_jump']
            y_range = saved_seq['y_range']
            y_trend = saved_seq['y_trend']
            y_value = saved_seq['y_value']
            feature_cols = saved_seq['feature_cols']
            
            # 데이터 분할
            split_idx = int(0.8 * len(X))
            
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            y_jump_train, y_jump_val = y_jump[:split_idx], y_jump[split_idx:]
            y_range_train, y_range_val = y_range[:split_idx], y_range[split_idx:]
            y_trend_train, y_trend_val = y_trend[:split_idx], y_trend[split_idx:]
            y_value_train, y_value_val = y_value[:split_idx], y_value[split_idx:]
            
            print(f"✅ 학습 데이터: {len(X_train)}")
            print(f"✅ 검증 데이터: {len(X_val)}")
            
            state['step'] = 5
            state['split_idx'] = split_idx
            ckpt.save_state(state)
            
            if ckpt.interrupted:
                print("\n💾 Step 4 저장 후 종료")
                sys.exit(0)
        
        # Step 5: 특징 준비 및 스케일링
        if step <= 5:
            print(f"\n[Step 5/10] 특징 준비 및 스케일링")
            print("-"*60)
            
            # 데이터 재로드
            saved_seq = ckpt.load_sequences()
            X = saved_seq['X']
            y_value = saved_seq['y_value']
            feature_cols = saved_seq['feature_cols']
            split_idx = state.get('split_idx', int(0.8 * len(X)))
            
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_value_train, y_value_val = y_value[:split_idx], y_value[split_idx:]
            
            # 특징 준비
            X_train_features = system.prepare_features(X_train)
            X_val_features = system.prepare_features(X_val)
            
            # 물리 특징 (PatchTST-PINN용)
            X_physics_train = system.create_physics_features(X_train, feature_cols)
            X_physics_val = system.create_physics_features(X_val, feature_cols)
            
            # 스케일링
            # 시퀀스 데이터
            X_train_scaled = system.scaler_X.fit_transform(
                X_train.reshape(-1, X_train.shape[-1])
            ).reshape(X_train.shape)
            X_val_scaled = system.scaler_X.transform(
                X_val.reshape(-1, X_val.shape[-1])
            ).reshape(X_val.shape)
            
            # 타겟
            y_value_train_scaled = system.scaler_y.fit_transform(y_value_train.reshape(-1, 1)).flatten()
            y_value_val_scaled = system.scaler_y.transform(y_value_val.reshape(-1, 1)).flatten()
            
            # 물리 특징
            X_physics_train_scaled = system.scaler_physics.fit_transform(X_physics_train)
            X_physics_val_scaled = system.scaler_physics.transform(X_physics_val)
            
            print(f"✅ 학습 특징 shape: {X_train_features.shape}")
            print(f"✅ 검증 특징 shape: {X_val_features.shape}")
            print(f"✅ 물리 특징 shape: {X_physics_train.shape}")
            
            # 스케일러 저장
            joblib.dump(system.scaler_X, ckpt.models_dir + '/scaler_X.pkl')
            joblib.dump(system.scaler_y, ckpt.models_dir + '/scaler_y.pkl')
            joblib.dump(system.scaler_physics, ckpt.models_dir + '/scaler_physics.pkl')
            
            # 인덱스 설정
            system.get_feature_indices(feature_cols)
            
            state['step'] = 6
            state['n_features'] = X.shape[2]
            ckpt.save_state(state)
            
            if ckpt.interrupted:
                print("\n💾 Step 5 저장 후 종료")
                sys.exit(0)
        
        # Step 6: 기존 모델 학습
        if step <= 6:
            print(f"\n[Step 6/10] 기존 모델 학습")
            print("-"*60)
            
            # 데이터 재로드
            saved_seq = ckpt.load_sequences()
            X = saved_seq['X']
            y_jump = saved_seq['y_jump']
            y_range = saved_seq['y_range']
            y_trend = saved_seq['y_trend']
            y_value = saved_seq['y_value']
            feature_cols = saved_seq['feature_cols']
            
            split_idx = state.get('split_idx', int(0.8 * len(X)))
            
            X_train = X[:split_idx]
            y_jump_train = y_jump[:split_idx]
            y_range_train = y_range[:split_idx]
            y_trend_train = y_trend[:split_idx]
            y_value_train = y_value[:split_idx]
            
            # 특징 준비
            X_train_features = system.prepare_features(X_train)
            
            # 특징 인덱스 설정
            system.get_feature_indices(feature_cols)
            
            # 6-1. XGBoost 점프 모델
            if not state.get('jump_trained', False):
                system.train_jump_detector(X_train_features, y_jump_train)
                ckpt.save_model(system.model_jump, 'model_jump')
                state['jump_trained'] = True
                ckpt.save_state(state)
                
                if ckpt.interrupted:
                    sys.exit(0)
            
            # 6-2. 3구간 분류
            if not state.get('range_trained', False):
                print("\n📊 3구간 분류 모델 학습")
                system.model_range.fit(X_train_features, y_range_train)
                ckpt.save_model(system.model_range, 'model_range')
                state['range_trained'] = True
                ckpt.save_state(state)
                
                if ckpt.interrupted:
                    sys.exit(0)
            
            # 6-3. 상승/하락 패턴
            if not state.get('trend_trained', False):
                print("\n📈 상승/하락 패턴 모델 학습")
                system.model_trend.fit(X_train_features, y_trend_train)
                ckpt.save_model(system.model_trend, 'model_trend')
                state['trend_trained'] = True
                ckpt.save_state(state)
            
            # 6-4. 값 예측
            if not state.get('value_trained', False):
                print("\n💰 값 예측 모델 학습")
                system.model_value.fit(X_train_features, y_value_train)
                ckpt.save_model(system.model_value, 'model_value')
                state['value_trained'] = True
                ckpt.save_state(state)
            
            state['step'] = 7
            ckpt.save_state(state)
            
            if ckpt.interrupted:
                print("\n💾 기존 모델 저장 후 종료")
                sys.exit(0)
        
        # Step 7: ExtraTrees + 규칙 모델
        if step <= 7:
            print(f"\n[Step 7/10] ExtraTrees + 규칙 모델 학습")
            print("-"*60)
            
            # 데이터 재로드
            saved_seq = ckpt.load_sequences()
            X = saved_seq['X']
            y_jump = saved_seq['y_jump']
            feature_cols = saved_seq['feature_cols']
            
            split_idx = state.get('split_idx', int(0.8 * len(X)))
            X_train = X[:split_idx]
            y_jump_train = y_jump[:split_idx]
            
            # 특징 준비
            X_train_features = system.prepare_features(X_train)
            system.get_feature_indices(feature_cols)
            
            if not state.get('extra_rules_trained', False):
                system.train_extra_rules_jump(X_train_features, y_jump_train)
                ckpt.save_model(system.model_extra_rules, 'model_extra_rules')
                state['extra_rules_trained'] = True
                ckpt.save_state(state)
            
            state['step'] = 8
            ckpt.save_state(state)
            
            if ckpt.interrupted:
                print("\n💾 ExtraTrees 모델 저장 후 종료")
                sys.exit(0)
        
        # Step 8: PatchTST 모델 학습
        if step <= 8:
            print(f"\n[Step 8/10] PatchTST 모델 학습")
            print("-"*60)
            
            # 데이터 재로드
            saved_seq = ckpt.load_sequences()
            X = saved_seq['X']
            y_value = saved_seq['y_value']
            feature_cols = saved_seq['feature_cols']
            n_features = state.get('n_features', X.shape[2])
            
            split_idx = state.get('split_idx', int(0.8 * len(X)))
            
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_value_train, y_value_val = y_value[:split_idx], y_value[split_idx:]
            
            # 스케일러 로드
            system.scaler_X = joblib.load(ckpt.models_dir + '/scaler_X.pkl')
            system.scaler_y = joblib.load(ckpt.models_dir + '/scaler_y.pkl')
            system.scaler_physics = joblib.load(ckpt.models_dir + '/scaler_physics.pkl')
            
            # 스케일링
            X_train_scaled = system.scaler_X.transform(
                X_train.reshape(-1, X_train.shape[-1])
            ).reshape(X_train.shape)
            X_val_scaled = system.scaler_X.transform(
                X_val.reshape(-1, X_val.shape[-1])
            ).reshape(X_val.shape)
            
            y_value_train_scaled = system.scaler_y.transform(y_value_train.reshape(-1, 1)).flatten()
            y_value_val_scaled = system.scaler_y.transform(y_value_val.reshape(-1, 1)).flatten()
            
            # PatchTST 모델 생성
            if system.model_patchtst is None:
                system.build_patchtst_models(n_features)
            
            # 8-1. PatchTST 안정형
            if not state.get('patchtst_trained', False):
                print("\n🔷 PatchTST 안정형 모델 학습")
                
                system.model_patchtst.compile(
                    optimizer=Adam(learning_rate=0.001),
                    loss='mse',
                    metrics=['mae']
                )
                
                history_stable = system.model_patchtst.fit(
                    X_train_scaled, y_value_train_scaled,
                    validation_data=(X_val_scaled, y_value_val_scaled),
                    epochs=30,
                    batch_size=32,
                    callbacks=[
                        EarlyStopping(patience=5, restore_best_weights=True),
                        ReduceLROnPlateau(factor=0.5, patience=3)
                    ],
                    verbose=1
                )
                
                ckpt.save_keras_model(system.model_patchtst, 'model_patchtst')
                state['patchtst_trained'] = True
                ckpt.save_state(state)
                
                if ckpt.interrupted:
                    sys.exit(0)
            
            # 8-2. PatchTST-PINN 극단형
            if not state.get('patchtst_pinn_trained', False):
                print("\n🔶 PatchTST-PINN 극단형 모델 학습")
                
                # 물리 특징
                X_physics_train = system.create_physics_features(X_train, feature_cols)
                X_physics_val = system.create_physics_features(X_val, feature_cols)
                
                X_physics_train_scaled = system.scaler_physics.transform(X_physics_train)
                X_physics_val_scaled = system.scaler_physics.transform(X_physics_val)
                
                system.model_patchtst_pinn.compile(
                    optimizer=Adam(learning_rate=0.001),
                    loss='mse',
                    metrics=['mae']
                )
                
                history_extreme = system.model_patchtst_pinn.fit(
                    [X_train_scaled, X_physics_train_scaled], 
                    y_value_train_scaled,
                    validation_data=(
                        [X_val_scaled, X_physics_val_scaled], 
                        y_value_val_scaled
                    ),
                    epochs=40,
                    batch_size=32,
                    callbacks=[
                        EarlyStopping(patience=7, restore_best_weights=True),
                        ReduceLROnPlateau(factor=0.5, patience=4)
                    ],
                    verbose=1
                )
                
                ckpt.save_keras_model(system.model_patchtst_pinn, 'model_patchtst_pinn')
                state['patchtst_pinn_trained'] = True
                ckpt.save_state(state)
            
            state['step'] = 9
            ckpt.save_state(state)
            
            if ckpt.interrupted:
                print("\n💾 PatchTST 모델 저장 후 종료")
                sys.exit(0)
        
        # Step 9: 평가
        if step <= 9:
            print(f"\n[Step 9/10] 모델 평가")
            print("-"*60)
            
            # 모델 로드
            system.model_jump = ckpt.load_model('model_jump')
            system.model_extra_rules = ckpt.load_model('model_extra_rules')
            
            # 데이터 재로드
            saved_seq = ckpt.load_sequences()
            X = saved_seq['X']
            y_jump = saved_seq['y_jump']
            feature_cols = saved_seq['feature_cols']
            
            split_idx = state.get('split_idx', int(0.8 * len(X)))
            X_val = X[split_idx:]
            y_jump_val = y_jump[split_idx:]
            
            # 특징 준비
            X_val_features = system.prepare_features(X_val)
            system.get_feature_indices(feature_cols)
            
            # 물리 특징
            X_physics_val = system.create_physics_features(X_val, feature_cols)
            
            # 스케일러 로드
            system.scaler_X = joblib.load(ckpt.models_dir + '/scaler_X.pkl')
            system.scaler_physics = joblib.load(ckpt.models_dir + '/scaler_physics.pkl')
            
            print("\n" + "="*60)
            print("📊 모든 모델 성능 평가")
            print("="*60)
            
            # 1. XGBoost 기본
            jump_pred_xgb = system.model_jump.predict(X_val_features)
            metrics_xgb = evaluate_jump_detection(y_jump_val, jump_pred_xgb, "XGBoost 기본")
            
            # 2. XGBoost + 규칙
            jump_pred_xgb_rules = system.apply_rule_based_boost(X_val_features, jump_pred_xgb)
            metrics_xgb_rules = evaluate_jump_detection(y_jump_val, jump_pred_xgb_rules, "XGBoost + 규칙")
            
            # 3. ExtraTrees
            jump_pred_extra = system.model_extra_rules.predict(X_val_features)
            metrics_extra = evaluate_jump_detection(y_jump_val, jump_pred_extra, "ExtraTrees")
            
            # 4. ExtraTrees + 규칙
            jump_pred_extra_rules = system.apply_rule_based_boost(X_val_features, jump_pred_extra)
            metrics_extra_rules = evaluate_jump_detection(y_jump_val, jump_pred_extra_rules, "ExtraTrees + 규칙")
            
            # 5. 하이브리드 (ExtraTrees + PatchTST)
            if state.get('patchtst_pinn_trained', False):
                # PatchTST 모델 재생성
                n_features = state.get('n_features', X_val.shape[2])
                system.build_patchtst_models(n_features)
                
                # 가중치 로드
                system.model_patchtst_pinn.load_weights(ckpt.models_dir + '/model_patchtst_pinn.h5')
                
                # 하이브리드 예측
                hybrid_predictor = HybridJumpPredictor(system)
                jump_pred_hybrid = hybrid_predictor.predict(X_val, X_val_features, X_physics_val)
                metrics_hybrid = evaluate_jump_detection(y_jump_val, jump_pred_hybrid, "ExtraTrees + PatchTST 하이브리드")
            
            state['step'] = 10
            ckpt.save_state(state)
        
        # Step 10: 최종 정리
        if step <= 10:
            print(f"\n[Step 10/10] 최종 정리 및 저장")
            print("-"*60)
            
            # 전체 시스템 저장
            print("\n💾 전체 시스템 저장")
            joblib.dump(system, 'hubroom_jump_detection_system_complete.pkl')
            
            # 최종 결과 요약
            print("\n" + "="*60)
            print("🎉 학습 완료! 최종 성능 요약")
            print("="*60)
            
            print("\n📊 점프 감지율 비교:")
            print(f"  - XGBoost 기본: {metrics_xgb['recall']*100:.1f}%")
            print(f"  - XGBoost + 규칙: {metrics_xgb_rules['recall']*100:.1f}%")
            print(f"  - ExtraTrees: {metrics_extra['recall']*100:.1f}%")
            print(f"  - ExtraTrees + 규칙: {metrics_extra_rules['recall']*100:.1f}%")
            if 'metrics_hybrid' in locals():
                print(f"  - ExtraTrees + PatchTST 하이브리드: {metrics_hybrid['recall']*100:.1f}% ✨")
            
            # 상태 파일 정리
            remove = input("\n학습 완료! 상태 파일을 제거하시겠습니까? (y/n): ")
            if remove.lower() == 'y':
                ckpt.clear_state()
                print("🧹 상태 파일 제거 완료")
            
            print("\n" + "="*60)
            print("✅ 모든 과정 완료!")
            print("="*60)
    
    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자 중단 감지")
        print("💾 진행 상황이 저장되었습니다. 다시 실행하면 이어서 진행됩니다.")
    
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        print("💾 진행 상황이 저장되었습니다.")

if __name__ == "__main__":
    main()