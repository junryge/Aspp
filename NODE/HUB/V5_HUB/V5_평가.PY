#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
🎯 HUBROOM V4 Ultimate 평가 시스템
================================================================================
목표:
- 과거 20분 데이터로 10분 후 예측
- Model 1: PatchTST (안정형)
- Model 2: PatchTST+PINN (극단형)
- 평가 데이터: 2025년 8월 데이터
- CSV 결과 출력
================================================================================
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import RobustScaler, StandardScaler
import joblib
import os
import pickle
import warnings
from datetime import datetime, timedelta
from tqdm import tqdm
import h5py

warnings.filterwarnings('ignore')

print("="*80)
print("🎯 HUBROOM V4 Ultimate 평가 시스템")
print("📊 20분 시퀀스 → 10분 후 예측 평가")
print("📅 평가 데이터: 2025년 8월")
print("="*80)

# ==============================================================================
# 📊 데이터 처리 클래스
# ==============================================================================

class HubRoomDataProcessor:
    """V4 Ultimate 데이터 처리"""
    
    def __init__(self):
        self.target_col = 'CURRENT_M16A_3F_JOB_2'
        
        # V4 필수 21개 컬럼
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
        
        self.scaler_X = None
        self.scaler_y = None
        self.scaler_physics = None
    
    def load_and_merge_data(self, data_path, bridge_path=None):
        """데이터 로드 및 BRIDGE_TIME 병합"""
        print("\n📂 데이터 로드 중...")
        
        # 메인 데이터 로드
        df = pd.read_csv(data_path)
        print(f"✅ 평가 데이터 로드: {df.shape}")
        
        # 시간 컬럼 처리
        time_col = df.columns[0]
        if 'datetime' not in df.columns:
            df['datetime'] = pd.to_datetime(df[time_col], format='%Y%m%d%H%M')
        
        # BRIDGE_TIME 데이터 병합 (있다면)
        if bridge_path and os.path.exists(bridge_path):
            print("📂 BRIDGE_TIME 데이터 병합 중...")
            bridge_df = pd.read_csv(bridge_path)
            
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
                
                # 보간
                df['BRIDGE_TIME'] = df['BRIDGE_TIME'].interpolate(method='linear', limit_direction='both')
        
        # BRIDGE_TIME이 없으면 기본값
        if 'BRIDGE_TIME' not in df.columns:
            df['BRIDGE_TIME'] = 3.5
        
        df['BRIDGE_TIME'] = df['BRIDGE_TIME'].fillna(3.5)
        
        return df
    
    def create_features(self, df):
        """특징 엔지니어링"""
        print("\n🔧 특징 생성 중...")
        
        # 1. 유입/유출 밸런스
        df['total_inflow'] = df[self.inflow_cols].sum(axis=1)
        df['total_outflow'] = df[self.outflow_cols].sum(axis=1)
        df['flow_balance'] = df['total_inflow'] - df['total_outflow']
        df['flow_ratio'] = df['total_inflow'] / (df['total_outflow'] + 1)
        
        # 2. 이동 평균
        for window in [5, 10, 20]:
            df[f'target_ma{window}'] = df[self.target_col].rolling(window).mean()
            df[f'target_std{window}'] = df[self.target_col].rolling(window).std()
        
        # 3. 추세 특징
        df['trend_20min'] = df[self.target_col] - df[self.target_col].shift(20)
        df['trend_10min'] = df[self.target_col] - df[self.target_col].shift(10)
        df['acceleration'] = df['trend_10min'] - df['trend_10min'].shift(10)
        
        # 4. CMD 관련
        df['cmd_sync_score'] = (df[self.cmd_cols] > 235).sum(axis=1)
        df['cmd_max'] = df[self.cmd_cols].max(axis=1)
        
        # 5. 극단값 관련 특징
        df['near_extreme'] = (df[self.target_col] > 280).astype(int)
        df['consecutive_high'] = (df[self.target_col] > 270).rolling(10).sum()
        
        # 6. 브릿지타임 관련
        df['bridge_high'] = (df['BRIDGE_TIME'] > 4.0).astype(int)
        df['storage_x_bridge'] = df['M16A_3F_STORAGE_UTIL'] * df['BRIDGE_TIME']
        
        # NaN 처리
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(0)
        
        return df
    
    def prepare_physics_features(self, df):
        """물리 기반 특징 준비"""
        physics_features = pd.DataFrame()
        
        # 1. 유입/유출 총량
        physics_features['total_inflow'] = df[self.inflow_cols].sum(axis=1)
        physics_features['total_outflow'] = df[self.outflow_cols].sum(axis=1)
        
        # 2. 밸런스
        physics_features['flow_balance'] = physics_features['total_inflow'] - physics_features['total_outflow']
        physics_features['flow_ratio'] = physics_features['total_inflow'] / (physics_features['total_outflow'] + 1)
        
        # 3. 용량 제한
        physics_features['capa_mean'] = df[self.capa_cols].mean(axis=1)
        physics_features['storage_util'] = df['M16A_3F_STORAGE_UTIL']
        
        # 4. OFS
        physics_features['ofs_balance'] = df['M14_TO_M16_OFS_CUR'] - df['M16_TO_M14_OFS_CUR']
        
        # 5. CMD 동기화
        physics_features['cmd_sync'] = df['cmd_sync_score']
        
        # 6. 브릿지타임
        physics_features['bridge_time'] = df['BRIDGE_TIME']
        
        return physics_features
    
    def create_sequences_for_evaluation(self, df, seq_len=20, pred_len=10):
        """평가용 시퀀스 생성"""
        print(f"\n📊 시퀀스 생성 중 ({seq_len}분 → {pred_len}분 후)...")
        
        # 특징 컬럼 선택
        feature_cols = [self.target_col] + self.inflow_cols + self.outflow_cols + \
                      self.cmd_cols + self.capa_cols + self.other_cols + ['BRIDGE_TIME']
        
        # 파생 특징들도 추가
        derived_features = [
            'total_inflow', 'total_outflow', 'flow_balance', 'flow_ratio',
            'target_ma5', 'target_ma10', 'target_ma20',
            'target_std5', 'target_std10', 'target_std20',
            'trend_20min', 'trend_10min', 'acceleration',
            'cmd_sync_score', 'cmd_max', 'near_extreme', 'consecutive_high',
            'bridge_high', 'storage_x_bridge'
        ]
        
        # 컬럼 존재 확인 후 추가
        for col in derived_features:
            if col in df.columns and col not in feature_cols:
                feature_cols.append(col)
        
        # 물리 특징 준비
        physics_df = self.prepare_physics_features(df)
        
        sequences = []
        physics_sequences = []
        targets = []
        timestamps = []
        sequence_info = []
        
        # 시퀀스 생성
        for i in tqdm(range(seq_len, len(df) - pred_len), desc="시퀀스 생성"):
            # 입력 시퀀스 (과거 20분)
            seq = df[feature_cols].iloc[i-seq_len:i].values
            physics_seq = physics_df.iloc[i-seq_len:i].values
            
            # 타겟 (10분 후)
            target = df[self.target_col].iloc[i + pred_len - 1]
            
            # 시간 정보
            current_time = df['datetime'].iloc[i-1]
            predict_time = df['datetime'].iloc[i + pred_len - 1]
            seq_start_time = df['datetime'].iloc[i-seq_len]
            seq_end_time = df['datetime'].iloc[i-1]
            
            # 시퀀스 통계
            seq_values = df[self.target_col].iloc[i-seq_len:i].values
            seq_max = np.max(seq_values)
            seq_min = np.min(seq_values)
            
            sequences.append(seq)
            physics_sequences.append(physics_seq)
            targets.append(target)
            timestamps.append(current_time)
            
            sequence_info.append({
                'current_time': current_time,
                'predict_time': predict_time,
                'seq_start_time': seq_start_time,
                'seq_end_time': seq_end_time,
                'seq_max': seq_max,
                'seq_min': seq_min,
                'target_value': target
            })
        
        print(f"✅ 총 {len(sequences)}개 시퀀스 생성 완료")
        
        return (np.array(sequences), np.array(physics_sequences), 
                np.array(targets), timestamps, sequence_info)

# ==============================================================================
# 🤖 모델 정의 (학습 코드와 동일)
# ==============================================================================

class PatchTST(keras.Model):
    """PatchTST Base Model"""
    def __init__(self, config):
        super().__init__()
        
        self.patch_len = config['patch_len']
        self.stride = config['stride']
        self.d_model = config['d_model']
        self.n_heads = config['n_heads']
        self.d_ff = config['d_ff']
        self.n_layers = config['n_layers']
        self.seq_len = config['seq_len']
        self.pred_len = config['pred_len']
        
        # 패치 임베딩
        self.patch_embedding = layers.Dense(self.d_model)
        self.positional_encoding = self._get_positional_encoding()
        
        # Transformer 인코더
        self.encoder_layers = [
            layers.MultiHeadAttention(
                num_heads=self.n_heads,
                key_dim=self.d_model // self.n_heads,
                dropout=0.1
            ) for _ in range(self.n_layers)
        ]
        
        self.ffn_layers = [
            keras.Sequential([
                layers.Dense(self.d_ff, activation='relu'),
                layers.Dropout(0.1),
                layers.Dense(self.d_model),
                layers.Dropout(0.1)
            ]) for _ in range(self.n_layers)
        ]
        
        self.layer_norms = [[layers.LayerNormalization(epsilon=1e-6) 
                            for _ in range(2)] 
                           for _ in range(self.n_layers)]
        
        # 출력층
        self.flatten = layers.Flatten()
        self.output_projection = layers.Dense(self.pred_len)
    
    def _get_positional_encoding(self):
        max_patches = self.seq_len // self.stride
        position = np.arange(max_patches)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * 
                         -(np.log(10000.0) / self.d_model))
        
        pos_encoding = np.zeros((max_patches, self.d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        
        return tf.constant(pos_encoding, dtype=tf.float32)
    
    def create_patches(self, x):
        batch_size = tf.shape(x)[0]
        n_vars = tf.shape(x)[2]
        
        patches = []
        for i in range(0, self.seq_len - self.patch_len + 1, self.stride):
            patch = x[:, i:i+self.patch_len, :]
            patches.append(patch)
        
        patches = tf.stack(patches, axis=1)
        patches = tf.reshape(patches, [batch_size, -1, self.patch_len * n_vars])
        
        return patches
    
    def call(self, inputs, training=False):
        x = inputs
        
        # 패치 생성
        patches = self.create_patches(x)
        
        # 패치 임베딩
        x = self.patch_embedding(patches)
        
        # 위치 인코딩 추가
        seq_len = tf.shape(x)[1]
        x = x + self.positional_encoding[:seq_len]
        
        # Transformer 인코더
        for i in range(self.n_layers):
            # Multi-head attention
            attn_output = self.encoder_layers[i](x, x, training=training)
            x = self.layer_norms[i][0](x + attn_output)
            
            # Feed forward
            ffn_output = self.ffn_layers[i](x, training=training)
            x = self.layer_norms[i][1](x + ffn_output)
        
        # 출력 생성
        x = self.flatten(x)
        output = self.output_projection(x)
        
        return output

class PatchTSTPINN(keras.Model):
    """PatchTST + Physics-Informed Neural Network"""
    def __init__(self, config):
        super().__init__()
        
        self.patchtst = PatchTST(config)
        
        # 물리 네트워크
        self.physics_net = keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(8, activation='relu')
        ])
        
        # 통합 네트워크
        self.fusion_net = keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(config['pred_len'])
        ])
    
    def call(self, inputs, training=False):
        x_seq, x_physics = inputs
        
        # PatchTST 예측
        patchtst_out = self.patchtst(x_seq, training=training)
        
        # 물리 특징 처리
        physics_out = self.physics_net(x_physics[:, -1, :], training=training)
        
        # 통합
        combined = tf.concat([patchtst_out, physics_out], axis=1)
        output = self.fusion_net(combined, training=training)
        
        return output

# ==============================================================================
# 🧮 모델 선택 로직
# ==============================================================================

class ModelSelector:
    """V4 Ultimate 모델 선택 로직"""
    
    @staticmethod
    def select_model(sequence_data, past_max, recent_5min_mean, acceleration):
        """
        모델 선택 로직
        Returns: 'model1' or 'model2'
        """
        # 강한 하락 추세 → Model 1
        if acceleration < -50:
            return 'model1', "강한 하락 추세"
        
        # 277 점프 감지 구간 → Model 2
        if 275 <= past_max <= 279:
            return 'model2', "277 점프 위험 구간"
        
        # 극단값 근접 → Model 2
        if recent_5min_mean > 285:
            return 'model2', "극단값 근접"
        
        # 가속 신호 → Model 2
        if acceleration > 5:
            return 'model2', "가속 신호 감지"
        
        # 안정 구간 → Model 1
        if past_max < 250:
            return 'model1', "안정 구간"
        
        # 기본값
        return 'model1', "기본 선택"
    
    @staticmethod
    def apply_fine_tuning(model_name, prediction, past_max, recent_mean):
        """V4 파인튜닝 로직"""
        if model_name == 'model1':
            # 안정형 조정
            if past_max < 270:
                return prediction * 0.98
            elif prediction > 300 and past_max < 280:
                return 295  # False Positive 방지
        else:
            # 극단형 조정
            if recent_mean > 310:
                return prediction * 1.07
            elif 275 <= past_max <= 279 and recent_mean >= 275:
                return 308  # 277 점프 예측
        
        return prediction

# ==============================================================================
# 📊 평가 및 결과 저장
# ==============================================================================

def evaluate_models():
    """모델 평가 메인 함수"""
    
    print("\n🚀 V4 Ultimate 평가 시작...")
    
    # 1. 체크포인트 로드
    checkpoint_dir = './checkpoints_ultimate'
    if not os.path.exists(checkpoint_dir):
        print("❌ 학습된 모델을 찾을 수 없습니다. 먼저 학습을 진행해주세요.")
        return
    
    # 2. 데이터 처리
    processor = HubRoomDataProcessor()
    
    # 평가 데이터 로드
    eval_data_path = 'data/20250801_to_20250831.csv'
    bridge_data_path = 'data/BRIDGE_TIME_202508.csv'
    
    if not os.path.exists(eval_data_path):
        print(f"❌ 평가 데이터를 찾을 수 없습니다: {eval_data_path}")
        return
    
    df = processor.load_and_merge_data(eval_data_path, bridge_data_path)
    df = processor.create_features(df)
    
    # 3. 시퀀스 생성
    X_seq, X_physics, y_true, timestamps, sequence_info = processor.create_sequences_for_evaluation(df)
    
    print(f"\n📊 평가 데이터 준비 완료")
    print(f"  - 시퀀스 수: {len(X_seq)}")
    print(f"  - 시퀀스 형태: {X_seq.shape}")
    print(f"  - 물리 특징 형태: {X_physics.shape}")
    
    # 4. 스케일러 로드
    scaler_path = os.path.join(checkpoint_dir, 'scalers')
    processor.scaler_X = joblib.load(os.path.join(scaler_path, 'scaler_X.pkl'))
    processor.scaler_y = joblib.load(os.path.join(scaler_path, 'scaler_y.pkl'))
    processor.scaler_physics = joblib.load(os.path.join(scaler_path, 'scaler_physics.pkl'))
    
    # 데이터 정규화
    X_seq_scaled = processor.scaler_X.transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape)
    X_physics_scaled = processor.scaler_physics.transform(X_physics.reshape(-1, X_physics.shape[-1])).reshape(X_physics.shape)
    
    # 5. 모델 로드
    print("\n🤖 모델 로드 중...")
    
    # 모델 설정
    config = {
        'seq_len': 20,
        'pred_len': 10,
        'patch_len': 5,
        'stride': 2,
        'd_model': 128,
        'n_heads': 8,
        'd_ff': 512,
        'n_layers': 3
    }
    
    # Model 1 로드
    model1 = PatchTST(config)
    model1.build([(None, 20, X_seq.shape[-1])])
    model1.load_weights(os.path.join(checkpoint_dir, 'model1_stable.h5'))
    
    # Model 2 로드
    model2 = PatchTSTPINN(config)
    model2.build([(None, 20, X_seq.shape[-1]), (None, 20, 9)])
    model2.load_weights(os.path.join(checkpoint_dir, 'model2_extreme.h5'))
    
    print("✅ 모델 로드 완료")
    
    # 6. 예측 수행
    print("\n🔮 예측 수행 중...")
    
    results = []
    model_selector = ModelSelector()
    
    for i in tqdm(range(len(X_seq)), desc="예측 진행"):
        # 시퀀스 정보
        seq_info = sequence_info[i]
        seq_values = df[processor.target_col].iloc[i:i+20].values
        
        # 통계 계산
        past_max = np.max(seq_values)
        recent_5min_mean = np.mean(seq_values[-5:])
        acceleration = (np.mean(seq_values[-5:]) - np.mean(seq_values[-10:-5])) - \
                      (np.mean(seq_values[-10:-5]) - np.mean(seq_values[-15:-10]))
        
        # 모델 선택
        selected_model, reason = model_selector.select_model(
            seq_values, past_max, recent_5min_mean, acceleration
        )
        
        # 예측
        if selected_model == 'model1':
            pred_scaled = model1.predict(X_seq_scaled[i:i+1], verbose=0)
            pred = processor.scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
        else:
            pred_scaled = model2.predict([X_seq_scaled[i:i+1], X_physics_scaled[i:i+1]], verbose=0)
            pred = processor.scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
        
        # 파인튜닝
        pred_adjusted = model_selector.apply_fine_tuning(
            selected_model, pred, past_max, recent_5min_mean
        )
        
        # 점프 케이스 확인
        is_jump = (past_max < 280) and (seq_info['target_value'] >= 300)
        
        # 결과 저장
        results.append({
            '날짜': seq_info['current_time'].strftime('%Y-%m-%d %H:%M'),
            '예측날짜': seq_info['predict_time'].strftime('%Y-%m-%d %H:%M'),
            '측정시퀀스MAX': round(seq_info['seq_max'], 2),
            '측정시퀀스MIN': round(seq_info['seq_min'], 2),
            '시퀀스시작시간': seq_info['seq_start_time'].strftime('%Y-%m-%d %H:%M'),
            '시퀀스완료시간': seq_info['seq_end_time'].strftime('%Y-%m-%d %H:%M'),
            '실제값': round(seq_info['target_value'], 2),
            '예측값': round(pred_adjusted, 2),
            '오차': round(abs(seq_info['target_value'] - pred_adjusted), 2),
            '선택모델': selected_model,
            '선택이유': reason,
            '점프케이스300+': 'O' if is_jump else 'X'
        })
    
    # 7. 결과 분석
    results_df = pd.DataFrame(results)
    
    print("\n📊 평가 결과 분석")
    print("="*60)
    
    # 전체 성능
    mae = results_df['오차'].mean()
    rmse = np.sqrt((results_df['오차'] ** 2).mean())
    
    print(f"\n📈 전체 성능:")
    print(f"  - MAE: {mae:.2f}")
    print(f"  - RMSE: {rmse:.2f}")
    print(f"  - 총 예측 수: {len(results_df)}")
    
    # 극단값 감지 성능
    extreme_mask = results_df['실제값'] >= 300
    if extreme_mask.sum() > 0:
        extreme_detected = (results_df[extreme_mask]['예측값'] >= 300).sum()
        extreme_recall = extreme_detected / extreme_mask.sum() * 100
        print(f"\n🎯 극단값(300+) 감지:")
        print(f"  - 실제 300+ 케이스: {extreme_mask.sum()}개")
        print(f"  - 감지된 케이스: {extreme_detected}개")
        print(f"  - 감지율: {extreme_recall:.1f}%")
    
    # 점프 케이스 성능
    jump_cases = results_df[results_df['점프케이스300+'] == 'O']
    if len(jump_cases) > 0:
        jump_detected = (jump_cases['예측값'] >= 290).sum()
        jump_recall = jump_detected / len(jump_cases) * 100
        print(f"\n🚀 점프 케이스 (과거<280 → 실제300+):")
        print(f"  - 총 점프 케이스: {len(jump_cases)}개")
        print(f"  - 감지된 케이스: {jump_detected}개")
        print(f"  - 점프 감지율: {jump_recall:.1f}%")
    
    # False Positive
    stable_mask = results_df['실제값'] < 300
    if stable_mask.sum() > 0:
        fp_count = (results_df[stable_mask]['예측값'] >= 300).sum()
        fp_rate = fp_count / stable_mask.sum() * 100
        print(f"\n⚠️ False Positive:")
        print(f"  - 실제 <300 케이스: {stable_mask.sum()}개")
        print(f"  - 잘못 예측된 케이스: {fp_count}개")
        print(f"  - FP Rate: {fp_rate:.1f}%")
    
    # 모델 선택 통계
    print(f"\n🤖 모델 선택 통계:")
    model_stats = results_df['선택모델'].value_counts()
    for model, count in model_stats.items():
        print(f"  - {model}: {count}개 ({count/len(results_df)*100:.1f}%)")
    
    # 8. CSV 파일 저장
    output_path = 'hubroom_v4_evaluation_results.csv'
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 결과 저장 완료: {output_path}")
    
    # 샘플 출력
    print(f"\n📋 샘플 결과 (처음 10개):")
    print(results_df[['날짜', '예측날짜', '실제값', '예측값', '오차', '점프케이스300+']].head(10))
    
    print("\n" + "="*80)
    print("✅ V4 Ultimate 평가 완료!")
    print("="*80)

if __name__ == "__main__":
    evaluate_models()