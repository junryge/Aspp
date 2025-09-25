# -*- coding: utf-8 -*-

def evaluate_sequences(self, df):
    """시퀀스별 평가 - 모델별 예측값 포함"""
    print("\n🔍 시퀀스별 평가 시작...")
    
    results = []
    timestamps = df['timestamp'].values
    df = df.drop('timestamp', axis=1)
    
    # 사용할 컬럼들
    all_cols = self.v4_essential_cols + ['BRIDGE_TIME'] + [
        'consecutive_300_count', 'consecutive_300_prob', 'long_trend',
        'mid_trend', 'acceleration', 'volatility_change', 'jump_risk', 'extreme_risk'
    ]
    
    total = len(df) - 40
    
    for i in tqdm(range(total)):
        # 30분 시퀀스
        X = df[all_cols].iloc[i:i+30].values
        
        # 실제값
        y_true = df[self.target_col].iloc[i+39]
        
        # 시퀀스 정보
        seq_values = df[self.target_col].iloc[i:i+30].values
        
        # 스케일링
        X_scaled = self.scaler_X.transform(X.reshape(-1, X.shape[1]))
        X_scaled = X_scaled.reshape(1, 30, -1)
        
        physics = self.create_physics_features_simple(seq_values)
        physics_scaled = self.scaler_physics.transform(physics.reshape(1, -1))
        
        features = self.create_fine_tuning_features_simple(seq_values)
        features_scaled = self.scaler_features.transform(features.reshape(1, -1))
        
        # 통합 모델 예측
        y_pred_scaled = self.model.predict([X_scaled, physics_scaled, features_scaled], verbose=0)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()[0]
        
        # Model1 (안정형) 예측
        pred1_scaled = self.model.model1.predict(X_scaled, verbose=0)
        pred1 = self.scaler_y.inverse_transform(pred1_scaled.reshape(-1, 1)).flatten()[0]
        
        # Model2 (극단형) 예측  
        pred2_scaled = self.model.model2.predict([X_scaled, physics_scaled], verbose=0)
        pred2 = self.scaler_y.inverse_transform(pred2_scaled.reshape(-1, 1)).flatten()[0]
        
        # Fine-tuning 네트워크 출력
        fine_tune_outputs = self.model.fine_tuning.predict(features_scaled, verbose=0)
        jump_prob = fine_tune_outputs['jump_prob'][0]
        extreme_prob = fine_tune_outputs['extreme_prob'][0]
        
        # Jump Detection 네트워크 출력
        jump_outputs = self.model.jump_detection.predict(features_scaled, verbose=0)
        jump_detect_prob = jump_outputs['jump_prob'][0]
        
        # 어느 모델에 가까운지 판단
        model_selected = "Model1" if abs(y_pred - pred1) < abs(y_pred - pred2) else "Model2"
        
        # numpy.datetime64를 pd.Timestamp로 변환
        target_time = pd.Timestamp(timestamps[i+39])
        start_time = pd.Timestamp(timestamps[i])
        end_time = pd.Timestamp(timestamps[i+29])
        
        # 30분 구간 분석
        first_10 = seq_values[:10]
        middle_10 = seq_values[10:20]
        last_10 = seq_values[20:30]
        
        # 결과 저장
        result = {
            '날짜시간': target_time.strftime('%Y-%m-%d %H:%M'),
            '타겟날짜': target_time.strftime('%Y-%m-%d'),
            '시퀀스시작시간': start_time.strftime('%H:%M'),
            '시퀀스종료시간': end_time.strftime('%H:%M'),
            'JUMP_277구간': any(275 <= v <= 279 for v in seq_values[-5:]),
            '시퀀스MAX': np.max(seq_values),
            '시퀀스MIN': np.min(seq_values),
            '시퀀스평균': np.mean(seq_values),
            '시퀀스표준편차': np.std(seq_values),
            '최근5분MAX': np.max(seq_values[-5:]),
            '최근5분평균': np.mean(seq_values[-5:]),
            '첫10분평균': np.mean(first_10),
            '중간10분평균': np.mean(middle_10),
            '마지막10분평균': np.mean(last_10),
            '장기추세': np.mean(last_10) - np.mean(first_10),
            '가속도': (np.mean(last_10) - np.mean(middle_10)) - (np.mean(middle_10) - np.mean(first_10)),
            '연속300카운트': sum(1 for v in seq_values if v >= 300),
            '실제값': y_true,
            '최종예측값': y_pred,
            'Model1예측': pred1,
            'Model2예측': pred2,
            '선택모델': model_selected,
            'Jump확률': jump_prob,
            'Extreme확률': extreme_prob,
            'JumpDetect확률': jump_detect_prob,
            'MAE': abs(y_true - y_pred),
            'Model1_MAE': abs(y_true - pred1),
            'Model2_MAE': abs(y_true - pred2),
            '300이상여부': y_true >= 300,
            '점프케이스': (np.max(seq_values) < 280) and (y_true >= 300),
            '예측성공': (y_true >= 300 and y_pred >= 300) or (y_true < 300 and y_pred < 300),
            'Model1_성공': (y_true >= 300 and pred1 >= 300) or (y_true < 300 and pred1 < 300),
            'Model2_성공': (y_true >= 300 and pred2 >= 300) or (y_true < 300 and pred2 < 300),
            '예측난이도': self.calculate_difficulty(seq_values, y_true)
        }
        
        results.append(result)
    
    return pd.DataFrame(results)

def calculate_difficulty(self, seq_values, y_true):
    """예측 난이도 계산"""
    difficulty = 0
    
    # 점프 케이스
    if np.max(seq_values) < 280 and y_true >= 300:
        difficulty += 50
    
    # 277 구간
    if any(275 <= v <= 279 for v in seq_values[-5:]):
        difficulty += 30
    
    # 변동성
    if np.std(seq_values) > 50:
        difficulty += 20
    
    # 급변
    if abs(np.mean(seq_values[-5:]) - np.mean(seq_values[:5])) > 30:
        difficulty += 15
    
    return difficulty
"""
================================================================================
📊 V4 ULTIMATE 평가 시스템 - 2025년 8월 데이터 평가 (간단 버전)
================================================================================
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import h5py
import os
from datetime import datetime
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

print("="*80)
print("📊 V4 ULTIMATE 평가 시스템")
print("🎯 2025년 8월 데이터 (학습되지 않은 데이터) 평가")
print("="*80)

# 필요한 모델 클래스들만 정의 (학습된 가중치 로드용)
class PatchTSTModel(keras.Model):
    def __init__(self, config):
        super().__init__()
        self.seq_len = config['seq_len']
        self.n_features = config['n_features']
        self.patch_len = config['patch_len']
        self.n_patches = self.seq_len // self.patch_len
        
        self.patch_embedding = layers.Dense(128, activation='relu')
        self.attention = layers.MultiHeadAttention(num_heads=8, key_dim=16)
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.ffn = keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(128)
        ])
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout = layers.Dropout(0.3)
        self.dense2 = layers.Dense(64, activation='relu')
        self.output_layer = layers.Dense(1)
    
    def call(self, x, training=False):
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, [batch_size, self.n_patches, self.patch_len * self.n_features])
        x = self.patch_embedding(x)
        attn = self.attention(x, x, training=training)
        x = self.norm1(x + attn)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        output = self.output_layer(x)
        return tf.squeeze(output, axis=-1)

class PatchTSTPINN(keras.Model):
    def __init__(self, config):
        super().__init__()
        self.seq_len = config['seq_len']
        self.n_features = config['n_features']
        self.patch_len = config['patch_len']
        self.n_patches = self.seq_len // self.patch_len
        
        self.patch_embedding = layers.Dense(128, activation='relu')
        self.attention = layers.MultiHeadAttention(num_heads=8, key_dim=16)
        self.norm = layers.LayerNormalization()
        self.flatten = layers.Flatten()
        self.temporal_dense = layers.Dense(64, activation='relu')
        self.physics_net = keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu')
        ])
        self.fusion = keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])
        self.extreme_boost = layers.Dense(1, activation='sigmoid')
    
    def call(self, inputs, training=False):
        x_seq, x_physics = inputs
        batch_size = tf.shape(x_seq)[0]
        x = tf.reshape(x_seq, [batch_size, self.n_patches, self.patch_len * self.n_features])
        x = self.patch_embedding(x)
        attn = self.attention(x, x, training=training)
        x = self.norm(x + attn)
        x = self.flatten(x)
        x_temporal = self.temporal_dense(x)
        x_physics = self.physics_net(x_physics)
        x_combined = tf.concat([x_temporal, x_physics], axis=-1)
        output = self.fusion(x_combined)
        boost_factor = self.extreme_boost(x_physics)
        output = output * (1 + boost_factor * 0.1)
        return tf.squeeze(output, axis=-1)

class FineTuningNetwork(keras.Model):
    def __init__(self):
        super().__init__()
        self.feature_net = keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu')
        ])
        self.jump_branch = keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        self.extreme_branch = keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        self.adjustment_branch = keras.Sequential([
            layers.Dense(16, activation='relu'),
            layers.Dense(8, activation='relu'),
            layers.Dense(1, activation='tanh')
        ])
    
    def call(self, features, training=False):
        x = self.feature_net(features, training=training)
        jump_prob = self.jump_branch(x, training=training)
        extreme_prob = self.extreme_branch(x, training=training)
        adjustment = self.adjustment_branch(x, training=training)
        return {
            'jump_prob': tf.squeeze(jump_prob, axis=-1),
            'extreme_prob': tf.squeeze(extreme_prob, axis=-1),
            'adjustment': tf.squeeze(adjustment, axis=-1) * 0.2
        }

class JumpDetectionNetwork(keras.Model):
    def __init__(self):
        super().__init__()
        self.jump_detector = keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        self.jump_magnitude = keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='relu')
        ])
    
    def call(self, features, training=False):
        jump_prob = self.jump_detector(features, training=training)
        jump_size = self.jump_magnitude(features, training=training)
        return {
            'jump_prob': tf.squeeze(jump_prob, axis=-1),
            'jump_size': tf.squeeze(jump_size, axis=-1) * 50 + 250
        }

class DecisionFusionNetwork(keras.Model):
    def __init__(self):
        super().__init__()
        self.model_selector = keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        self.fusion_net = keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(8, activation='relu'),
            layers.Dense(1)
        ])
    
    def call(self, inputs, training=False):
        features, pred1, pred2, fine_tune, jump_detect = inputs
        model_choice_prob = self.model_selector(features, training=training)
        model_choice = tf.squeeze(model_choice_prob, axis=-1)
        combined_features = tf.concat([
            features[:, :8],
            tf.expand_dims(pred1, axis=-1),
            tf.expand_dims(pred2, axis=-1),
            tf.expand_dims(fine_tune['jump_prob'], axis=-1),
            tf.expand_dims(fine_tune['extreme_prob'], axis=-1),
            tf.expand_dims(jump_detect['jump_prob'], axis=-1),
            tf.expand_dims(model_choice, axis=-1)
        ], axis=-1)
        final_pred = self.fusion_net(combined_features, training=training)
        return {
            'model_choice': model_choice,
            'final_prediction': tf.squeeze(final_pred, axis=-1)
        }

class IntegratedV4Model(keras.Model):
    def __init__(self, config):
        super().__init__()
        self.model1 = PatchTSTModel(config)
        self.model2 = PatchTSTPINN(config)
        self.fine_tuning = FineTuningNetwork()
        self.jump_detection = JumpDetectionNetwork()
        self.decision_fusion = DecisionFusionNetwork()
    
    def call(self, inputs, training=False):
        x_seq, x_physics, x_features = inputs
        pred1 = self.model1(x_seq, training=training)
        pred2 = self.model2([x_seq, x_physics], training=training)
        fine_tune_outputs = self.fine_tuning(x_features, training=training)
        jump_outputs = self.jump_detection(x_features, training=training)
        fusion_outputs = self.decision_fusion(
            [x_features, pred1, pred2, fine_tune_outputs, jump_outputs], 
            training=training
        )
        return fusion_outputs['final_prediction']

# 평가 클래스
class V4Evaluator:
    def __init__(self):
        self.target_col = 'CURRENT_M16A_3F_JOB_2'
        self.model = None
        
        # 저장된 스케일러 로드
        print("📂 저장된 스케일러 로드 중...")
        self.scaler_X = joblib.load('./checkpoints_integrated_30min/scalers/scaler_X.pkl')
        self.scaler_y = joblib.load('./checkpoints_integrated_30min/scalers/scaler_y.pkl')
        self.scaler_physics = joblib.load('./checkpoints_integrated_30min/scalers/scaler_physics.pkl')
        self.scaler_features = joblib.load('./checkpoints_integrated_30min/scalers/scaler_features.pkl')
        print("✅ 스케일러 로드 완료")
        
        # 필요한 컬럼 정의 (특징 생성용)
        self.v4_essential_cols = [
            'CURRENT_M16A_3F_JOB_2', 'M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2',
            'M14A_3F_TO_HUB_JOB2', 'M14B_7F_TO_HUB_JOB2', 'M16B_10F_TO_HUB_JOB',
            'M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB', 'M16A_3F_TO_M14A_3F_JOB',
            'M16A_3F_TO_M14B_7F_JOB', 'M16A_3F_TO_3F_MLUD_JOB', 'M16A_3F_CMD',
            'M16A_6F_TO_HUB_CMD', 'M16A_2F_TO_HUB_CMD', 'M14A_3F_TO_HUB_CMD',
            'M14B_7F_TO_HUB_CMD', 'M16A_6F_LFT_MAXCAPA', 'M16A_2F_LFT_MAXCAPA',
            'M16A_3F_STORAGE_UTIL', 'M14_TO_M16_OFS_CUR', 'M16_TO_M14_OFS_CUR'
        ]
    
    def load_august_data(self):
        """8월 데이터만 로드"""
        print("\n📊 8월 평가 데이터 로드 중...")
        
        df_raw = pd.read_csv('data/HUB_0509_TO_0807_DATA.CSV')
        time_col = df_raw.columns[0]
        df_raw['timestamp'] = pd.to_datetime(df_raw[time_col], format='%Y%m%d%H%M', errors='coerce')
        
        # 8월 데이터 필터링
        df_august = df_raw[(df_raw['timestamp'] >= '2025-08-01') & 
                          (df_raw['timestamp'] < '2025-09-01')].copy()
        
        print(f"✅ 8월 데이터: {len(df_august)} 행")
        
        # 필수 컬럼 처리
        for col in self.v4_essential_cols:
            if col not in df_august.columns:
                df_august[col] = 0
        
        # BRIDGE_TIME 및 고급 특징 추가
        df_august['BRIDGE_TIME'] = 3.5
        
        # 고급 특징 간단 생성
        for col in ['consecutive_300_count', 'consecutive_300_prob', 'long_trend', 
                   'mid_trend', 'acceleration', 'volatility_change', 'jump_risk', 'extreme_risk']:
            df_august[col] = 0
        
        return df_august
    
    def load_trained_model(self):
        """학습된 모델 로드"""
        print("\n🤖 학습된 모델 로드 중...")
        
        with h5py.File('./checkpoints_integrated_30min/scaled_integrated_data.h5', 'r') as f:
            n_features = f.attrs['n_features']
        
        config = {
            'seq_len': 30,
            'n_features': n_features,
            'patch_len': 6
        }
        
        self.model = IntegratedV4Model(config)
        
        # 더미 입력으로 모델 빌드
        dummy_input = [
            np.zeros((1, 30, n_features)),
            np.zeros((1, 11)),
            np.zeros((1, 15))
        ]
        _ = self.model(dummy_input, training=False)
        
        # 가중치 로드
        self.model.load_weights('./checkpoints_integrated_30min/models/integrated_model.weights.h5')
        print("✅ 모델 로드 완료")
        
        self.n_features = n_features
    
    def create_physics_features_simple(self, window_30):
        """간단한 물리 특징 생성"""
        physics = np.zeros(11)
        physics[0] = window_30[-1]  # 마지막 값
        physics[8] = np.mean(window_30[-5:])  # 최근 5분 평균
        physics[9] = np.mean(window_30[-10:]) - np.mean(window_30[:10])  # 장기 추세
        physics[10] = 1.0  # 변동성 변화
        return physics
    
    def create_fine_tuning_features_simple(self, window_30):
        """간단한 미세조정 특징 생성"""
        features = np.zeros(15)
        recent_5 = window_30[-5:]
        
        # 277 구간 감지
        features[0] = 1.0 if any(275 <= v <= 279 for v in recent_5) else 0.0
        features[10] = np.max(window_30)  # 최대값
        features[11] = np.mean(recent_5)  # 최근 5분 평균
        
        return features
    
    def evaluate_sequences(self, df):
        """시퀀스별 평가"""
        print("\n🔍 시퀀스별 평가 시작...")
        
        results = []
        timestamps = df['timestamp'].values
        df = df.drop('timestamp', axis=1)
        
        # 사용할 컬럼들
        all_cols = self.v4_essential_cols + ['BRIDGE_TIME'] + [
            'consecutive_300_count', 'consecutive_300_prob', 'long_trend',
            'mid_trend', 'acceleration', 'volatility_change', 'jump_risk', 'extreme_risk'
        ]
        
        total = len(df) - 40
        
        for i in tqdm(range(total)):
            # 30분 시퀀스
            X = df[all_cols].iloc[i:i+30].values
            
            # 실제값
            y_true = df[self.target_col].iloc[i+39]
            
            # 시퀀스 정보
            seq_values = df[self.target_col].iloc[i:i+30].values
            
            # 예측
            X_scaled = self.scaler_X.transform(X.reshape(-1, X.shape[1]))
            X_scaled = X_scaled.reshape(1, 30, -1)
            
            physics = self.create_physics_features_simple(seq_values)
            physics_scaled = self.scaler_physics.transform(physics.reshape(1, -1))
            
            features = self.create_fine_tuning_features_simple(seq_values)
            features_scaled = self.scaler_features.transform(features.reshape(1, -1))
            
            y_pred_scaled = self.model.predict([X_scaled, physics_scaled, features_scaled], verbose=0)
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()[0]
            
            # 결과 저장
            result = {
                '날짜시간': timestamps[i+39].strftime('%Y-%m-%d %H:%M'),
                '타겟날짜': timestamps[i+39].strftime('%Y-%m-%d'),
                '시퀀스시작시간': timestamps[i].strftime('%H:%M'),
                '시퀀스종료시간': timestamps[i+29].strftime('%H:%M'),
                'JUMP_277구간': any(275 <= v <= 279 for v in seq_values[-5:]),
                '시퀀스MAX': np.max(seq_values),
                '시퀀스MIN': np.min(seq_values),
                '시퀀스평균': np.mean(seq_values),
                '시퀀스표준편차': np.std(seq_values),
                '최근5분MAX': np.max(seq_values[-5:]),
                '최근5분평균': np.mean(seq_values[-5:]),
                '실제값': y_true,
                '예측값': y_pred,
                'MAE': abs(y_true - y_pred),
                '300이상여부': y_true >= 300,
                '점프케이스': (np.max(seq_values) < 280) and (y_true >= 300),
                '예측성공': (y_true >= 300 and y_pred >= 300) or (y_true < 300 and y_pred < 300)
            }
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def analyze_and_save(self, df_results):
        """결과 분석 및 저장"""
        print("\n📊 평가 결과 분석")
        print("="*60)
        
        mae = df_results['MAE'].mean()
        print(f"전체 MAE: {mae:.2f}")
        
        extreme_mask = df_results['300이상여부']
        if extreme_mask.sum() > 0:
            extreme_recall = (df_results[extreme_mask]['예측값'] >= 300).sum() / extreme_mask.sum() * 100
            print(f"300+ 감지율: {extreme_recall:.1f}%")
        
        jump_mask = df_results['점프케이스']
        if jump_mask.sum() > 0:
            jump_detection = (df_results[jump_mask]['예측값'] >= 290).sum() / jump_mask.sum() * 100
            print(f"점프 감지율: {jump_detection:.1f}% ({jump_mask.sum()}개)")
        
        # CSV 저장
        output_path = 'evaluation_august_2025.csv'
        df_results.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 평가 결과 저장: {output_path}")
        print(f"   총 {len(df_results)}개 시퀀스 평가 완료")

def main():
    evaluator = V4Evaluator()
    df_august = evaluator.load_august_data()
    evaluator.load_trained_model()
    df_results = evaluator.evaluate_sequences(df_august)
    evaluator.analyze_and_save(df_results)
    print("\n✅ 평가 완료!")

if __name__ == "__main__":
    main()