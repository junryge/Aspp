#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
🎯 HUBROOM 점프 감지 시스템 - 전체 모델 평가 코드
================================================================================
평가 데이터: data/20250801_to_20250831.csv
평가 모델:
1. XGBoost 기본
2. XGBoost + 규칙
3. ExtraTrees
4. ExtraTrees + 규칙
5. ExtraTrees + PatchTST 하이브리드
================================================================================
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import os
from datetime import datetime, timedelta
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🎯 HUBROOM 점프 감지 시스템 - 2025년 8월 평가")
print("📊 모든 학습된 모델 평가")
print("="*80)

# ==============================================================================
# 📊 데이터 처리 클래스 (학습 코드와 동일)
# ==============================================================================

class HubRoomDataProcessor:
    """데이터 처리 - 학습 코드와 동일"""
    
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
        
        # 확률 맵
        self.probability_map = {
            0: 0.003, 1: 0.15, 2: 0.25, 3: 0.31, 4: 0.43, 5: 0.43,
            6: 0.35, 7: 0.42, 8: 0.53, 9: 0.49, 10: 0.42,
            11: 0.47, 12: 0.52, 13: 0.60, 14: 0.54, 15: 0.66,
            16: 0.62, 17: 0.71, 18: 0.79, 19: 0.83, 20: 0.987,
            21: 0.99, 22: 0.99, 23: 0.99, 24: 0.99, 25: 0.99,
            26: 0.99, 27: 0.99, 28: 0.99, 29: 0.99, 30: 0.99
        }
    
    def load_evaluation_data(self):
        """8월 평가 데이터 로드"""
        print("\n[1단계] 평가 데이터 로드")
        
        # 8월 데이터
        df = pd.read_csv('data/20250801_to_20250831.csv')
        print(f"✅ 8월 평가 데이터: {df.shape}")
        
        # 시간 처리
        time_col = df.columns[0]
        df['datetime'] = pd.to_datetime(df[time_col], format='%Y%m%d%H%M')
        
        # BRIDGE_TIME 처리 (8월 데이터가 있다면)
        if os.path.exists('data/BRTIME_20250801_to_20250831.csv'):
            bridge_df = pd.read_csv('data/BRTIME_20250801_to_20250831.csv')
            print(f"✅ 8월 BRIDGE_TIME 데이터: {bridge_df.shape}")
            
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
        
        # BRIDGE_TIME 없으면 기본값
        if 'BRIDGE_TIME' not in df.columns:
            print("⚠️ BRIDGE_TIME 데이터 없음 - 기본값 3.5 사용")
            df['BRIDGE_TIME'] = 3.5
        else:
            df['BRIDGE_TIME'] = df['BRIDGE_TIME'].interpolate(method='linear', limit_direction='both')
            df['BRIDGE_TIME'] = df['BRIDGE_TIME'].fillna(3.5)
        
        return df
    
    def create_all_features(self, df):
        """특징 생성 - 학습과 동일"""
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
        
        # 8. 3구간 분류
        conditions = [
            df[self.target_col] < 150,
            (df[self.target_col] >= 150) & (df[self.target_col] < 300),
            df[self.target_col] >= 300
        ]
        choices = [0, 1, 2]
        df['range_class'] = np.select(conditions, choices, default=1)
        
        # 9. 점프 여부 (실제값 기준)
        df['past_30min_max'] = df[self.target_col].rolling(30).max()
        df['is_jump'] = ((df['past_30min_max'].shift(10) < 280) & 
                        (df[self.target_col] >= 300)).astype(int)
        
        # 10. 상승/하락 패턴
        df['change_20min'] = df[self.target_col] - df[self.target_col].shift(20)
        
        trend_conditions = [
            df['change_20min'] < -20,
            (df['change_20min'] >= -20) & (df['change_20min'] < 20),
            (df['change_20min'] >= 20) & (df['change_20min'] < 50),
            df['change_20min'] >= 50
        ]
        trend_choices = [0, 1, 2, 3]
        df['trend_pattern'] = np.select(trend_conditions, trend_choices, default=1)
        
        # 11. 상승률/하락률 
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
    
    def create_sequences(self, df, seq_len=30, pred_len=10):
        """시퀀스 생성"""
        print(f"\n[3단계] 시퀀스 생성 ({seq_len}분 → {pred_len}분 후)")
        
        feature_cols = [col for col in df.columns 
                       if col not in ['datetime', 'range_class', 'is_jump', 'trend_pattern']]
        
        sequences = []
        datetimes = []
        targets = []
        
        for i in tqdm(range(seq_len, len(df) - pred_len), desc="시퀀스 생성"):
            # 입력 시퀀스
            seq_data = {
                'X': df[feature_cols].iloc[i-seq_len:i].values,
                'datetime_start': df['datetime'].iloc[i-seq_len],
                'datetime_end': df['datetime'].iloc[i-1],
                'datetime_current': df['datetime'].iloc[i-1],
                'datetime_target': df['datetime'].iloc[i+pred_len-1],
                'seq_max': df[self.target_col].iloc[i-seq_len:i].max(),
                'seq_min': df[self.target_col].iloc[i-seq_len:i].min(),
                'target_value': df[self.target_col].iloc[i+pred_len-1],
                'is_jump_actual': df['is_jump'].iloc[i+pred_len-1]
            }
            sequences.append(seq_data)
        
        print(f"✅ {len(sequences)}개 시퀀스 생성 완료")
        return sequences, feature_cols

# ==============================================================================
# 🤖 PatchTST 모델 정의 (학습 코드와 동일)
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
# 📊 평가 시스템
# ==============================================================================

class ModelEvaluator:
    """모든 모델 평가"""
    
    def __init__(self, model_dir='./checkpoints_jump80/models'):
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        self.feature_indices = {}
        
    def load_all_models(self, n_features):
        """모든 모델 로드"""
        print("\n📂 모델 로드 중...")
        
        # 1. XGBoost
        try:
            self.models['xgboost'] = joblib.load(os.path.join(self.model_dir, 'model_jump.pkl'))
            print("✅ XGBoost 모델 로드")
        except:
            print("❌ XGBoost 모델 로드 실패")
        
        # 2. ExtraTrees
        try:
            self.models['extra_trees'] = joblib.load(os.path.join(self.model_dir, 'model_extra_rules.pkl'))
            print("✅ ExtraTrees 모델 로드")
        except:
            print("❌ ExtraTrees 모델 로드 실패")
        
        # 3. 스케일러
        try:
            self.scalers['X'] = joblib.load(os.path.join(self.model_dir, 'scaler_X.pkl'))
            self.scalers['y'] = joblib.load(os.path.join(self.model_dir, 'scaler_y.pkl'))
            self.scalers['physics'] = joblib.load(os.path.join(self.model_dir, 'scaler_physics.pkl'))
            print("✅ 스케일러 로드")
        except:
            print("❌ 스케일러 로드 실패")
        
        # 4. PatchTST 모델
        try:
            config = {'seq_len': 30, 'n_features': n_features, 'patch_len': 6}
            
            # PatchTST
            self.models['patchtst'] = PatchTSTModel(config)
            dummy_input = np.zeros((1, 30, n_features))
            self.models['patchtst'](dummy_input)  # 빌드
            self.models['patchtst'].load_weights(os.path.join(self.model_dir, 'model_patchtst.weights.h5'))
            print("✅ PatchTST 모델 로드")
            
            # PatchTST-PINN
            self.models['patchtst_pinn'] = PatchTSTPINN(config)
            dummy_physics = np.zeros((1, 5))
            self.models['patchtst_pinn']([dummy_input, dummy_physics])  # 빌드
            self.models['patchtst_pinn'].load_weights(os.path.join(self.model_dir, 'model_patchtst_pinn.weights.h5'))
            print("✅ PatchTST-PINN 모델 로드")
        except:
            print("❌ PatchTST 모델 로드 실패")
    
    def prepare_features(self, X_seq):
        """특징 준비"""
        last_features = X_seq[:, -1, :]
        mean_features = np.mean(X_seq, axis=1)
        std_features = np.std(X_seq, axis=1)
        max_features = np.max(X_seq, axis=1)
        min_features = np.min(X_seq, axis=1)
        trend_features = X_seq[:, -1, :] - X_seq[:, 0, :]
        
        features = np.hstack([
            last_features, mean_features, std_features,
            max_features, min_features, trend_features
        ])
        
        return features
    
    def create_physics_features(self, X_seq, feature_cols):
        """물리 특징 생성"""
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
            features.append(seq[-1, idx_storage] if idx_storage is not None else 0)
            features.append(seq[-1, idx_bridge] if idx_bridge is not None else 3.5)
            features.append(seq[-1, idx_flow_balance] if idx_flow_balance is not None else 0)
            features.append(np.mean(seq[-10:, 0]) - np.mean(seq[:10, 0]))
            features.append(np.std(seq[:, 0]))
            physics_features.append(features)
        
        return np.array(physics_features)
    
    def get_feature_indices(self, feature_cols):
        """특징 인덱스"""
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
        
        self.feature_indices = indices
        return indices
    
    def apply_rule_based_boost(self, X, predictions):
        """규칙 기반 부스팅"""
        boosted = predictions.copy()
        
        if not self.feature_indices:
            return boosted
        
        idx = self.feature_indices
        
        # 강한 신호
        if 'storage_util' in idx:
            strong_signal = X[:, idx['storage_util']] > 20
            boosted[strong_signal] = 1
        
        # 중간 신호
        medium_conditions = []
        
        if 'storage_util' in idx:
            medium_conditions.append(X[:, idx['storage_util']] > 10)
        if 'flow_balance' in idx:
            medium_conditions.append(X[:, idx['flow_balance']] > 50)
        if 'bridge_time' in idx:
            medium_conditions.append(X[:, idx['bridge_time']] > 4.0)
        
        if len(medium_conditions) >= 2:
            medium_signal = np.sum(medium_conditions, axis=0) >= 2
            if 'storage_util' in idx:
                storage_medium = X[:, idx['storage_util']] > 10
                final_medium = medium_signal & storage_medium & (boosted == 0)
                boosted[final_medium] = 1
        
        return boosted
    
    def evaluate_sequences(self, sequences, feature_cols):
        """모든 모델로 시퀀스 평가"""
        print("\n[4단계] 모델별 예측 수행")
        
        results = []
        
        # 배치 처리를 위한 준비
        all_X = np.array([seq['X'] for seq in sequences])
        all_features = self.prepare_features(all_X)
        all_physics = self.create_physics_features(all_X, feature_cols)
        
        # 특징 인덱스 설정
        self.get_feature_indices(feature_cols)
        
        # 배치 예측
        predictions = {}
        
        # 1. XGBoost
        if 'xgboost' in self.models:
            predictions['xgboost'] = self.models['xgboost'].predict(all_features)
            predictions['xgboost_rules'] = self.apply_rule_based_boost(all_features, predictions['xgboost'])
        
        # 2. ExtraTrees
        if 'extra_trees' in self.models:
            predictions['extra_trees'] = self.models['extra_trees'].predict(all_features)
            predictions['extra_trees_rules'] = self.apply_rule_based_boost(all_features, predictions['extra_trees'])
        
        # 3. 하이브리드 (ExtraTrees + PatchTST)
        if 'extra_trees' in self.models and 'patchtst_pinn' in self.models:
            hybrid_preds = []
            et_proba = self.models['extra_trees'].predict_proba(all_features)[:, 1]
            
            for i in tqdm(range(len(sequences)), desc="하이브리드 예측"):
                if et_proba[i] > 0.8:
                    hybrid_preds.append(1)
                elif et_proba[i] < 0.2:
                    hybrid_preds.append(0)
                else:
                    # PatchTST-PINN으로 정밀 예측
                    x_seq = all_X[i:i+1]
                    x_physics = all_physics[i:i+1]
                    
                    x_seq_scaled = self.scalers['X'].transform(
                        x_seq.reshape(-1, x_seq.shape[-1])
                    ).reshape(x_seq.shape)
                    x_physics_scaled = self.scalers['physics'].transform(x_physics)
                    
                    pred_value = self.models['patchtst_pinn'].predict(
                        [x_seq_scaled, x_physics_scaled], verbose=0
                    )[0]
                    
                    # 역변환
                    pred_value = self.scalers['y'].inverse_transform([[pred_value]])[0][0]
                    
                    hybrid_preds.append(1 if pred_value > 300 else 0)
            
            predictions['hybrid'] = np.array(hybrid_preds)
        
        # 결과 정리
        for i, seq in enumerate(sequences):
            result = {
                '날짜': seq['datetime_current'].strftime('%Y-%m-%d %H:%M'),
                '예측날짜': seq['datetime_target'].strftime('%Y-%m-%d %H:%M'),
                '측정시퀀스MAX': seq['seq_max'],
                '측정시퀀스MIN': seq['seq_min'],
                '시퀀스시작시간': seq['datetime_start'].strftime('%Y-%m-%d %H:%M'),
                '시퀀스완료시간': seq['datetime_end'].strftime('%Y-%m-%d %H:%M'),
                '실제값': seq['target_value'],
                '실제점프케이스300+': seq['is_jump_actual']
            }
            
            # 각 모델 예측 추가
            for model_name, preds in predictions.items():
                result[f'예측_{model_name}'] = preds[i] if i < len(preds) else 0
            
            results.append(result)
        
        return pd.DataFrame(results)

# ==============================================================================
# 🚀 메인 실행
# ==============================================================================

def main():
    """평가 메인 함수"""
    
    # 데이터 처리
    processor = HubRoomDataProcessor()
    df = processor.load_evaluation_data()
    df = processor.create_all_features(df)
    sequences, feature_cols = processor.create_sequences(df)
    
    # 모델 평가
    evaluator = ModelEvaluator()
    n_features = sequences[0]['X'].shape[1]
    evaluator.load_all_models(n_features)
    
    # 평가 수행
    results_df = evaluator.evaluate_sequences(sequences, feature_cols)
    
    # 결과 저장
    output_file = 'hubroom_evaluation_results_august_2025_all_models.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 결과 저장 완료: {output_file}")
    
    # 모델별 성능 요약
    print("\n" + "="*80)
    print("📊 모델별 점프 감지 성능 요약")
    print("="*80)
    
    actual_jumps = results_df['실제점프케이스300+'].sum()
    total_samples = len(results_df)
    
    print(f"\n전체 샘플: {total_samples}")
    print(f"실제 점프 케이스: {actual_jumps} ({actual_jumps/total_samples*100:.2f}%)")
    print("\n모델별 감지율:")
    
    for col in results_df.columns:
        if col.startswith('예측_'):
            model_name = col.replace('예측_', '')
            if col in results_df.columns:
                predicted_jumps = results_df[col].sum()
                tp = ((results_df['실제점프케이스300+'] == 1) & (results_df[col] == 1)).sum()
                recall = tp / actual_jumps if actual_jumps > 0 else 0
                
                print(f"  - {model_name}: {recall*100:.1f}% ({tp}/{actual_jumps})")
    
    print("\n" + "="*80)
    print("✅ 평가 완료!")
    print("="*80)

if __name__ == "__main__":
    main()