# -*- coding: utf-8 -*-
"""
HUBROOM 극단값 예측 시스템 V4.0
- V3 대비 개선사항:
  1. BRIDGE_TIME 컬럼 추가 (극단값 물리지표)
  2. 연속 300+ 패턴 분석 강화
  3. False Positive 방지 (300이하를 300이상으로 오탐 방지)
  4. 335+ 극단값 감지율 개선 (목표: 50%+)
- Model 1: PatchTST (전체 구간 균형)
- Model 2: PatchTST + PINN (335+ 극단값 특화, FP 방지)
- 중단/재개 기능 포함
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import os
import pickle
import warnings
from tqdm import tqdm
import joblib
import signal
import sys

warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)
print(f"TensorFlow Version: {tf.__version__}")
print("="*80)
print("🏭 HUBROOM 극단값 예측 시스템 V4.0")
print("🎯 목표: 335+ 극단값 감지율 50% + False Positive 최소화")
print("🚀 V4 핵심: BRIDGE_TIME 활용 + 연속 패턴 분석 + 오탐 방지")
print("✅ 중단/재개 기능 활성화")
print("="*80)

# ========================================
# 체크포인트 관리자 (중단/재시작)
# ========================================

class CheckpointManager:
    def __init__(self, checkpoint_dir='./checkpoints'):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.state_file = os.path.join(checkpoint_dir, 'training_state_v4.pkl')
        self.interrupted = False
        
        # Ctrl+C 핸들러
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        print('\n\n⚠️ 중단 감지! 상태 저장 중...')
        self.interrupted = True
        sys.exit(0)
    
    def save_state(self, state):
        with open(self.state_file, 'wb') as f:
            pickle.dump(state, f)
        print("💾 체크포인트 저장 완료")
    
    def load_state(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, 'rb') as f:
                state = pickle.load(f)
                print("📂 체크포인트 로드 완료")
                return state
        return None
    
    def clear_state(self):
        if os.path.exists(self.state_file):
            os.remove(self.state_file)
            print("🗑️ 체크포인트 제거 완료")

# ========================================
# V4 극단값 손실 함수 (False Positive 방지)
# ========================================

class ExtremeLossV4(keras.losses.Loss):
    def __init__(self, extreme_focus=True):
        super().__init__()
        self.extreme_focus = extreme_focus
        
    def call(self, y_true, y_pred):
        error = y_true - y_pred
        mae = tf.abs(error)
        
        if self.extreme_focus:
            # V4: 335+ 극단값에 더 강한 페널티
            weights = tf.where(y_true >= 0.9, 25.0,    # 335+ (정규화 기준)
                     tf.where(y_true >= 0.8, 15.0,     # 310-335
                     tf.where(y_true >= 0.7, 8.0,      # 300-310
                     tf.where(y_true >= 0.5, 3.0,      # 250-300
                              1.0))))                    # 나머지
            
            # 극단값 미탐지 추가 페널티 (False Negative)
            miss_penalty = tf.where(
                tf.logical_and(y_true >= 0.9, y_pred < 0.85),
                30.0 * mae,  # 335+ 미탐지 시 강력한 페널티
                0.0
            )
            
            # False Positive 방지 페널티 (300 이하를 300 이상으로 예측)
            false_positive_penalty = tf.where(
                tf.logical_and(y_true < 0.7, y_pred >= 0.7),  # 300 이하를 300+ 예측
                20.0 * mae,  # 오탐 페널티
                0.0
            )
            
            # 특히 200-250 구간을 300+로 예측하면 더 강한 페널티
            severe_fp_penalty = tf.where(
                tf.logical_and(y_true < 0.5, y_pred >= 0.7),  # 250 이하를 300+ 예측
                35.0 * mae,  # 심각한 오탐 페널티
                0.0
            )
            
            return tf.reduce_mean(weights * mae + miss_penalty + false_positive_penalty + severe_fp_penalty)
        else:
            # Model 1: 균형잡힌 가중치 + 경미한 오탐 방지
            weights = tf.where(y_true >= 0.7, 5.0,
                     tf.where(y_true >= 0.5, 2.0, 1.0))
            
            # Model 1에도 경미한 오탐 방지
            fp_penalty = tf.where(
                tf.logical_and(y_true < 0.6, y_pred >= 0.7),
                5.0 * mae,
                0.0
            )
            
            return tf.reduce_mean(weights * mae + fp_penalty)

# ========================================
# 극단값 모니터링 콜백 V4
# ========================================

class ExtremeValueCallbackV4(Callback):
    def __init__(self, X_val, y_val, scaler_y, X_physics_val=None):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.scaler_y = scaler_y
        self.X_physics_val = X_physics_val
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:
            # 예측
            if self.X_physics_val is not None:
                y_pred_scaled = self.model.predict([self.X_val, self.X_physics_val], verbose=0)
            else:
                y_pred_scaled = self.model.predict(self.X_val, verbose=0)
            
            # 역스케일링
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_true = self.scaler_y.inverse_transform(self.y_val.reshape(-1, 1)).flatten()
            
            # 335+ 감지율
            mask_335 = y_true >= 335
            if mask_335.sum() > 0:
                detected_335 = (y_pred >= 335)[mask_335].sum()
                recall_335 = detected_335 / mask_335.sum() * 100
                
                # 310+ 감지율
                mask_310 = y_true >= 310
                detected_310 = (y_pred >= 310)[mask_310].sum()
                recall_310 = detected_310 / mask_310.sum() * 100
                
                # False Positive Rate (오탐율)
                mask_under_300 = y_true < 300
                if mask_under_300.sum() > 0:
                    false_positive_300 = (y_pred >= 300)[mask_under_300].sum()
                    fp_rate = false_positive_300 / mask_under_300.sum() * 100
                else:
                    fp_rate = 0
                
                print(f"\n[Epoch {epoch}] 335+: {recall_335:.1f}% | 310+: {recall_310:.1f}% | FP(300): {fp_rate:.1f}%")

# ========================================
# V4 데이터 처리기 (BRIDGE_TIME 포함)
# ========================================

class DataProcessorV4:
    def __init__(self):
        self.target_col = 'CURRENT_M16A_3F_JOB_2'
        self.bridge_time_col = 'BRIDGE_TIME'
        
        # V4 필수 컬럼 (21개 전체 + 선택/권장)
        # 유입 컬럼 (6개)
        self.inflow_cols = [
            'M16A_6F_TO_HUB_JOB',      # 필수 70점
            'M16A_2F_TO_HUB_JOB2',     # 필수 65점
            'M14A_3F_TO_HUB_JOB2',     # 필수 65점
            'M14B_7F_TO_HUB_JOB2',     # 필수 65점
            'M16B_10F_TO_HUB_JOB'      # 선택 25점
        ]
        
        # 유출 컬럼 (5개)
        self.outflow_cols = [
            'M16A_3F_TO_M16A_6F_JOB',  # 필수 60점
            'M16A_3F_TO_M16A_2F_JOB',  # 필수 55점
            'M16A_3F_TO_M14A_3F_JOB',  # 필수 55점
            'M16A_3F_TO_M14B_7F_JOB',  # 필수 55점
            'M16A_3F_TO_3F_MLUD_JOB'   # 선택 35점
        ]
        
        # CMD 컬럼 (6개)
        self.cmd_cols = [
            'M16A_3F_CMD',             # 필수 85점 (내부)
            'M16A_6F_TO_HUB_CMD',      # 권장 55점 (외부)
            'M16A_2F_TO_HUB_CMD',      # 선택 50점 (외부)
            'M14A_3F_TO_HUB_CMD',      # 선택 50점 (외부)
            'M14B_7F_TO_HUB_CMD'       # 선택 50점 (외부)
        ]
        
        # MaxCapa 컬럼 (2개)
        self.maxcapa_cols = [
            'M16A_6F_LFT_MAXCAPA',     # 선택 50점
            'M16A_2F_LFT_MAXCAPA'      # 선택 45점
        ]
        
        # 극단값 지표 컬럼 (2개)
        self.extreme_indicator_cols = [
            'M16A_3F_STORAGE_UTIL',    # 필수 95점
            'BRIDGE_TIME'              # 필수 85점 (물리지표/극단값지표)
        ]
        
        # OFS 컬럼 (2개)
        self.ofs_cols = [
            'M14_TO_M16_OFS_CUR',      # 선택 40점
            'M16_TO_M14_OFS_CUR'       # 선택 40점
        ]
        
        # 모든 V4 사용 컬럼 리스트 (21개)
        self.all_v4_cols = (
            [self.target_col] +
            self.inflow_cols + 
            self.outflow_cols + 
            self.cmd_cols + 
            self.maxcapa_cols + 
            self.extreme_indicator_cols + 
            self.ofs_cols
        )
        
        # 스케일러
        self.scaler_X = RobustScaler()
        self.scaler_y = MinMaxScaler()
        self.scaler_physics = StandardScaler()
        
        # 스케일러 저장 경로
        os.makedirs('./scalers', exist_ok=True)
        
    def analyze_data(self, df):
        """데이터 분석"""
        target = df[self.target_col]
        print("\n📊 데이터 분석:")
        print(f"  범위: {target.min():.0f} ~ {target.max():.0f}")
        print(f"  평균: {target.mean():.1f}")
        print(f"  중앙값: {target.median():.1f}")
        
        print("\n🎯 3구간 분포:")
        print(f"  저구간(<200): {(target < 200).sum():6}개 ({(target < 200).sum()/len(target)*100:5.2f}%)")
        print(f"  정상(200-300): {((target >= 200) & (target < 300)).sum():6}개 ({((target >= 200) & (target < 300)).sum()/len(target)*100:5.2f}%)")
        print(f"  위험(300+): {(target >= 300).sum():6}개 ({(target >= 300).sum()/len(target)*100:5.2f}%)")
        
        print("\n🚨 극단값 세부:")
        print(f"  310+: {(target >= 310).sum():6}개 ({(target >= 310).sum()/len(target)*100:5.2f}%)")
        print(f"  335+: {(target >= 335).sum():6}개 ({(target >= 335).sum()/len(target)*100:5.2f}%)")
        
    def analyze_consecutive_patterns(self, df, seq_len=20):
        """연속 300+ 패턴 분석 (V4 확률 데이터 기반)"""
        target = df[self.target_col]
        
        # V4 추가데이터 기반 확률 매핑
        self.pattern_probability = {
            0: 0.15,   # 0개일 때 300+ 확률 (실제: 0.003)
            1: 0.15,   # 1개일 때
            2: 0.25,   # 2개일 때
            3: 0.31,   # 3개일 때
            4: 0.43,   # 4개일 때
            5: 0.43,   # 5개일 때
            6: 0.35,   # 6개일 때
            7: 0.42,   # 7개일 때
            8: 0.53,   # 8개일 때
            9: 0.49,   # 9개일 때
            10: 0.42,  # 10개일 때
            11: 0.47,  # 11개일 때
            12: 0.52,  # 12개일 때
            13: 0.60,  # 13개일 때
            14: 0.54,  # 14개일 때
            15: 0.66,  # 15개일 때
            16: 0.62,  # 16개일 때
            17: 0.71,  # 17개일 때
            18: 0.79,  # 18개일 때
            19: 0.83,  # 19개일 때
            20: 0.99   # 20개일 때 (실제: 0.987)
        }
        
        consecutive_300_counts = []
        consecutive_300_probs = []  # 확률 추가
        
        for i in range(len(df) - seq_len):
            window = target[i:i+seq_len]
            count_300 = (window >= 300).sum()
            consecutive_300_counts.append(count_300)
            
            # 확률 계산
            prob = self.pattern_probability.get(count_300, 0.5)
            consecutive_300_probs.append(prob)
        
        df['consecutive_300_count'] = 0
        df['consecutive_300_prob'] = 0.5
        df.loc[seq_len:, 'consecutive_300_count'] = consecutive_300_counts
        df.loc[seq_len:, 'consecutive_300_prob'] = consecutive_300_probs
        
        print("\n📊 연속 300+ 패턴 분석 (V4 데이터 기반):")
        print(f"  0개 (99.7% 300이하): {(df['consecutive_300_count'] == 0).sum()}")
        print(f"  1-5개 (15-43% 300+): {((df['consecutive_300_count'] >= 1) & (df['consecutive_300_count'] <= 5)).sum()}")
        print(f"  6-10개 (35-53% 300+): {((df['consecutive_300_count'] >= 6) & (df['consecutive_300_count'] <= 10)).sum()}")
        print(f"  11-15개 (47-66% 300+): {((df['consecutive_300_count'] >= 11) & (df['consecutive_300_count'] <= 15)).sum()}")
        print(f"  16-19개 (62-83% 300+): {((df['consecutive_300_count'] >= 16) & (df['consecutive_300_count'] <= 19)).sum()}")
        print(f"  20개 (98.7% 300+): {(df['consecutive_300_count'] == 20).sum()}")
        
        return df
    
    def create_sequences_v4(self, df, seq_len=20, pred_len=10):
        """V4 시퀀스 생성 (모든 V4 컬럼 포함)"""
        X, y, X_physics, weights = [], [], [], []
        
        # V4 컬럼 존재 여부 확인
        print("\n📋 V4 컬럼 확인:")
        available_v4_cols = []
        missing_cols = []
        
        for col in self.all_v4_cols:
            if col in df.columns:
                available_v4_cols.append(col)
            else:
                missing_cols.append(col)
        
        print(f"✅ 사용 가능 V4 컬럼: {len(available_v4_cols)}/21개")
        if missing_cols:
            print(f"⚠️ 누락 컬럼: {missing_cols}")
        
        # BRIDGE_TIME 확인
        if self.bridge_time_col in df.columns:
            print("✅ BRIDGE_TIME 컬럼 발견 - 극단값 지표로 활용")
            bridge_time_available = True
        else:
            print("⚠️ BRIDGE_TIME 컬럼 없음")
            bridge_time_available = False
        
        # 연속 패턴 분석
        df = self.analyze_consecutive_patterns(df, seq_len)
        
        # V4 컬럼만 사용 (없는 컬럼은 0으로 채움)
        numeric_cols = []
        for col in self.all_v4_cols:
            if col in df.columns:
                numeric_cols.append(col)
            else:
                # 누락된 컬럼은 0으로 채운 컬럼 생성
                df[col] = 0
                numeric_cols.append(col)
        
        # 추가: 연속 패턴 컬럼도 포함
        if 'consecutive_300_count' in df.columns:
            numeric_cols.append('consecutive_300_count')
        if 'consecutive_300_prob' in df.columns:
            numeric_cols.append('consecutive_300_prob')
        
        # 사용 가능한 유입/유출 컬럼
        available_inflow = [col for col in self.inflow_cols if col in df.columns]
        available_outflow = [col for col in self.outflow_cols if col in df.columns]
        available_cmd = [col for col in self.cmd_cols if col in df.columns]
        
        print(f"\n✅ 최종 사용 컬럼: {len(numeric_cols)}개")
        print(f"  유입: {len(available_inflow)}개")
        print(f"  유출: {len(available_outflow)}개")
        print(f"  CMD: {len(available_cmd)}개")
        
        # 구간별 인덱스 수집
        indices = {
            'low': [],      # <200
            'normal': [],   # 200-300
            '300': [],      # 300-310
            '310': [],      # 310-335
            '335': []       # 335+
        }
        
        for i in range(len(df) - seq_len - pred_len):
            target_val = df[self.target_col].iloc[i + seq_len + pred_len - 1]
            consecutive_count = df['consecutive_300_count'].iloc[i + seq_len - 1] if 'consecutive_300_count' in df.columns else 0
            
            # V4: 연속 패턴 기반 분류 (확률 데이터 활용)
            if target_val >= 335:
                indices['335'].append(i)
            elif target_val >= 310:
                indices['310'].append(i)
            elif target_val >= 300:
                indices['300'].append(i)
            elif target_val >= 200:
                # 연속 패턴 확률에 따른 위험도 판단
                prob_300 = df['consecutive_300_prob'].iloc[i + seq_len - 1] if 'consecutive_300_prob' in df.columns else 0.5
                
                if prob_300 >= 0.8:  # 80% 이상 확률
                    indices['300'].append(i)  # 위험구간으로 분류
                elif prob_300 >= 0.5:  # 50-80% 확률
                    if np.random.random() < prob_300:
                        indices['300'].append(i)
                    else:
                        indices['normal'].append(i)
                else:
                    indices['normal'].append(i)
            else:
                indices['low'].append(i)
        
        # V4 오버샘플링 전략 (False Positive 방지)
        all_indices = []
        all_indices.extend(indices['low'] * 3)      # 3배 (FP 방지 강화)
        all_indices.extend(indices['normal'])        # 1배
        all_indices.extend(indices['300'] * 5)       # 5배
        all_indices.extend(indices['310'] * 15)      # 15배
        all_indices.extend(indices['335'] * 25)      # 25배 (최대 강화)
        
        print(f"\n📊 V4 오버샘플링:")
        print(f"  <200: {len(indices['low'])} → {len(indices['low'])*3}")
        print(f"  200-300: {len(indices['normal'])} → {len(indices['normal'])}")
        print(f"  300-310: {len(indices['300'])} → {len(indices['300'])*5}")
        print(f"  310-335: {len(indices['310'])} → {len(indices['310'])*15}")
        print(f"  335+: {len(indices['335'])} → {len(indices['335'])*25}")
        
        np.random.shuffle(all_indices)
        
        # 시퀀스 생성
        for i in tqdm(all_indices, desc="V4 시퀀스 생성"):
            # 시계열 데이터 (V4 컬럼만)
            X.append(df[numeric_cols].iloc[i:i+seq_len].values)
            
            # 타겟
            y_val = df[self.target_col].iloc[i + seq_len + pred_len - 1]
            y.append(y_val)
            
            # V4 물리 데이터 (확장 + 패턴 확률)
            physics = [
                # 기본 물리 데이터
                df[self.target_col].iloc[i + seq_len - 1],  # 현재값
                df[available_inflow].iloc[i+seq_len:i+seq_len+pred_len].sum().sum() if available_inflow else 0,
                df[available_outflow].iloc[i+seq_len:i+seq_len+pred_len].sum().sum() if available_outflow else 0,
                
                # BRIDGE_TIME
                df[self.bridge_time_col].iloc[i + seq_len - 1] if bridge_time_available else 3.5,
                
                # 연속 패턴 수
                df['consecutive_300_count'].iloc[i + seq_len - 1] if 'consecutive_300_count' in df.columns else 0,
                
                # 연속 패턴 확률 (중요!)
                df['consecutive_300_prob'].iloc[i + seq_len - 1] if 'consecutive_300_prob' in df.columns else 0.5,
                
                # STORAGE_UTIL
                df['M16A_3F_STORAGE_UTIL'].iloc[i + seq_len - 1] if 'M16A_3F_STORAGE_UTIL' in df.columns else 0,
                
                # CMD 합계
                df[available_cmd].iloc[i + seq_len - 1].sum() if available_cmd else 0,
                
                # 최근 5개 평균 (추세)
                df[self.target_col].iloc[i+seq_len-5:i+seq_len].mean()
            ]
            
            X_physics.append(physics)
            
            # V4 가중치 (FP 방지 + 패턴 확률 반영)
            prob_300 = df['consecutive_300_prob'].iloc[i + seq_len - 1] if 'consecutive_300_prob' in df.columns else 0.5
            
            if y_val >= 335:
                weights.append(30.0)
            elif y_val >= 310:
                weights.append(20.0)
            elif y_val >= 300:
                weights.append(10.0)
            elif y_val < 200:
                # 저구간 + 낮은 확률 = FP 방지 강화
                if prob_300 < 0.2:  # 20% 미만 확률
                    weights.append(8.0)  # 강한 가중치 (FP 방지)
                elif prob_300 < 0.5:
                    weights.append(5.0)
                else:
                    weights.append(3.0)
            else:  # 200-300 구간
                # 경계 구간: 확률에 따른 가중치
                if prob_300 > 0.7:  # 높은 확률인데 300 미만
                    weights.append(3.0)  # False Negative 가능성
                elif prob_300 < 0.3:  # 낮은 확률이고 300 미만
                    weights.append(2.0)  # 정상
                else:
                    weights.append(1.0)
        
        print(f"\n✅ V4 시퀀스 생성 완료: {len(X)}개")
        print(f"  물리 데이터 차원: {len(physics)}개")
        
        return np.array(X), np.array(y), np.array(X_physics), np.array(weights)
    
    def save_scalers(self):
        """스케일러 저장"""
        joblib.dump(self.scaler_X, './scalers/scaler_X_v4.pkl')
        joblib.dump(self.scaler_y, './scalers/scaler_y_v4.pkl')
        joblib.dump(self.scaler_physics, './scalers/scaler_physics_v4.pkl')
        print("✅ V4 스케일러 저장 완료")
        
    def load_scalers(self):
        """스케일러 로드"""
        try:
            self.scaler_X = joblib.load('./scalers/scaler_X_v4.pkl')
            self.scaler_y = joblib.load('./scalers/scaler_y_v4.pkl')
            self.scaler_physics = joblib.load('./scalers/scaler_physics_v4.pkl')
            print("✅ V4 스케일러 로드 완료")
            return True
        except:
            print("❌ V4 스케일러 로드 실패")
            return False

# ========================================
# Model 1: PatchTST (전체 균형)
# ========================================

class PatchTSTModel(keras.Model):
    def __init__(self, config):
        super().__init__()
        
        self.seq_len = config['seq_len']
        self.n_features = config['n_features']
        self.patch_len = config['patch_len']
        self.n_patches = self.seq_len // self.patch_len
        
        # 패치 임베딩
        self.patch_embedding = layers.Dense(128, activation='relu')
        
        # Transformer
        self.attention = layers.MultiHeadAttention(num_heads=8, key_dim=16)
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        
        self.ffn = keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(128)
        ])
        
        # 출력
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout = layers.Dropout(0.3)
        self.dense2 = layers.Dense(64, activation='relu')
        self.output_layer = layers.Dense(1)
        
    def call(self, x, training=False):
        batch_size = tf.shape(x)[0]
        
        # 패치 생성
        x = tf.reshape(x, [batch_size, self.n_patches, self.patch_len * self.n_features])
        
        # 패치 임베딩
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

# ========================================
# Model 2: PatchTST + PINN (극단값 특화)
# ========================================

class PatchTSTPINN(keras.Model):
    def __init__(self, config):
        super().__init__()
        
        # PatchTST 부분
        self.seq_len = config['seq_len']
        self.n_features = config['n_features']
        self.patch_len = config['patch_len']
        self.n_patches = self.seq_len // self.patch_len
        
        # 패치 임베딩
        self.patch_embedding = layers.Dense(128, activation='relu')
        
        # Transformer
        self.attention = layers.MultiHeadAttention(num_heads=8, key_dim=16)
        self.norm = layers.LayerNormalization()
        
        # 시계열 처리
        self.flatten = layers.Flatten()
        self.temporal_dense = layers.Dense(64, activation='relu')
        
        # 물리 정보 처리 (PINN) - V4 확장
        self.physics_net = keras.Sequential([
            layers.Dense(64, activation='relu'),  # 확장
            layers.BatchNormalization(),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu')
        ])
        
        # 융합 및 극단값 보정
        self.fusion = keras.Sequential([
            layers.Dense(128, activation='relu'),  # 확장
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])
        
        # 극단값 부스팅 레이어
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
        output = output * (1 + boost_factor * 0.2)
        
        return tf.squeeze(output, axis=-1)

# ========================================
# 메인 함수
# ========================================

def main():
    # 체크포인트 관리자
    ckpt = CheckpointManager()
    
    # 이전 상태 확인
    state = ckpt.load_state()
    
    if state:
        print(f"\n📂 이전 학습 상태 발견! (Step {state.get('step', 1)})")
        
        resume = input("이어서 진행하시겠습니까? (y: 이어서, n: 처음부터): ").lower()
        
        if resume != 'y':
            ckpt.clear_state()
            state = None
            step = 1
        else:
            step = state.get('step', 1)
            print(f"✅ Step {step}부터 재개합니다.")
    else:
        state = {}
        step = 1
        print("\n이전 학습 상태가 없습니다.")
        input("Enter를 눌러 시작하세요...")
    
    # 데이터 처리기
    processor = DataProcessorV4()
    
    # Step 1: 데이터 로드 및 전처리
    if step == 1:
        print("\n[Step 1/6] 데이터 로드 및 전처리")
        
        # 메인 데이터 로드
        df = pd.read_csv('data/HUB_0509_TO_0730_DATA.CSV')
        print(f"✅ 메인 데이터 로드: {df.shape}")
        
        # BRIDGE_TIME 데이터 병합 (있는 경우)
        bridge_time_path = 'data/BRIDGE_TIME.CSV'
        if os.path.exists(bridge_time_path):
            print("📊 BRIDGE_TIME 데이터 발견! 병합 중...")
            try:
                bridge_df = pd.read_csv(bridge_time_path)
                # BRIDGE_TIME 컬럼 추출 (IDC_VAL 컬럼)
                if 'IDC_VAL' in bridge_df.columns:
                    bridge_df = bridge_df.rename(columns={'IDC_VAL': 'BRIDGE_TIME'})
                    # 시간 기준으로 병합
                    if 'CRT_TM' in bridge_df.columns:
                        bridge_df['timestamp'] = pd.to_datetime(bridge_df['CRT_TM'])
                        df['timestamp'] = pd.to_datetime(df.iloc[:, 0], format='%Y%m%d%H%M', errors='coerce')
                        df = pd.merge(df, bridge_df[['timestamp', 'BRIDGE_TIME']], 
                                    on='timestamp', how='left')
                        df['BRIDGE_TIME'] = df['BRIDGE_TIME'].fillna(3.5)  # 기본값
                        print("✅ BRIDGE_TIME 데이터 병합 완료")
                else:
                    print("⚠️ BRIDGE_TIME 컬럼 형식이 다릅니다. 기본값 사용")
                    df['BRIDGE_TIME'] = 3.5
            except Exception as e:
                print(f"⚠️ BRIDGE_TIME 병합 실패: {e}")
                df['BRIDGE_TIME'] = 3.5
        else:
            print("ℹ️ BRIDGE_TIME 데이터 없음. 기본값(3.5) 사용")
            df['BRIDGE_TIME'] = 3.5
        
        processor.analyze_data(df)
        
        # 타임스탬프 정리
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.to_datetime(df.iloc[:, 0], format='%Y%m%d%H%M', errors='coerce')
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = df.fillna(method='ffill').fillna(0)
        
        state['df_shape'] = df.shape
        state['step'] = 2
        ckpt.save_state(state)
        
        # 다음 단계를 위해 df 저장
        df.to_pickle('./checkpoints/processed_df_v4.pkl')
        print("💾 전처리된 데이터 저장 완료")
    
    # Step 2: 시퀀스 생성
    if step <= 2:
        print("\n[Step 2/6] V4 시퀀스 생성")
        
        df = pd.read_pickle('./checkpoints/processed_df_v4.pkl')
        X, y, X_physics, weights = processor.create_sequences_v4(df)
        
        state['X'] = X
        state['y'] = y
        state['X_physics'] = X_physics
        state['weights'] = weights
        state['n_features'] = X.shape[2]
        state['step'] = 3
        ckpt.save_state(state)
        print("✅ V4 시퀀스 생성 완료")
    
    # Step 3: 데이터 분할
    if step <= 3:
        print("\n[Step 3/6] 데이터 분할")
        
        X = state.get('X')
        y = state.get('y')
        X_physics = state.get('X_physics')
        weights = state.get('weights')
        
        if X is None or y is None:
            print("❌ 시퀀스 데이터가 없습니다. Step 2부터 다시 실행해주세요.")
            state['step'] = 2
            ckpt.save_state(state)
            return
        
        indices = np.arange(len(X))
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
        val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=42)
        
        state['train_idx'] = train_idx
        state['val_idx'] = val_idx
        state['test_idx'] = test_idx
        state['step'] = 4
        ckpt.save_state(state)
        print(f"✅ 데이터 분할 완료")
        print(f"  Train: {len(train_idx)}")
        print(f"  Valid: {len(val_idx)}")
        print(f"  Test: {len(test_idx)}")
    
    # Step 4: 스케일링
    if step <= 4:
        print("\n[Step 4/6] 데이터 스케일링")
        
        X = state['X']
        y = state['y']
        X_physics = state['X_physics']
        weights = state['weights']
        n_features = state['n_features']
        
        train_idx = state['train_idx']
        val_idx = state['val_idx']
        test_idx = state['test_idx']
        
        # 데이터 분할
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        X_physics_train = X_physics[train_idx]
        X_physics_val = X_physics[val_idx]
        X_physics_test = X_physics[test_idx]
        
        weights_train = weights[train_idx]
        
        # 스케일링
        X_train_flat = X_train.reshape(-1, n_features)
        X_train_scaled = processor.scaler_X.fit_transform(X_train_flat)
        X_train_scaled = X_train_scaled.reshape(len(X_train), 20, n_features)
        
        X_val_scaled = processor.scaler_X.transform(X_val.reshape(-1, n_features)).reshape(len(X_val), 20, n_features)
        X_test_scaled = processor.scaler_X.transform(X_test.reshape(-1, n_features)).reshape(len(X_test), 20, n_features)
        
        y_train_scaled = processor.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = processor.scaler_y.transform(y_val.reshape(-1, 1)).flatten()
        y_test_scaled = processor.scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        
        X_physics_train_scaled = processor.scaler_physics.fit_transform(X_physics_train)
        X_physics_val_scaled = processor.scaler_physics.transform(X_physics_val)
        X_physics_test_scaled = processor.scaler_physics.transform(X_physics_test)
        
        # 스케일러 저장
        processor.save_scalers()
        
        # 상태 저장
        state['X_train_scaled'] = X_train_scaled
        state['X_val_scaled'] = X_val_scaled
        state['X_test_scaled'] = X_test_scaled
        state['y_train_scaled'] = y_train_scaled
        state['y_val_scaled'] = y_val_scaled
        state['y_test'] = y_test
        state['X_physics_train_scaled'] = X_physics_train_scaled
        state['X_physics_val_scaled'] = X_physics_val_scaled
        state['X_physics_test_scaled'] = X_physics_test_scaled
        state['weights_train'] = weights_train
        state['step'] = 5
        ckpt.save_state(state)
        print("✅ 스케일링 완료")
    
    # Step 5: 모델 학습
    if step <= 5:
        print("\n[Step 5/6] 모델 학습")
        print("="*80)
        
        # 데이터 로드
        X_train_scaled = state['X_train_scaled']
        X_val_scaled = state['X_val_scaled']
        y_train_scaled = state['y_train_scaled']
        y_val_scaled = state['y_val_scaled']
        X_physics_train_scaled = state['X_physics_train_scaled']
        X_physics_val_scaled = state['X_physics_val_scaled']
        weights_train = state['weights_train']
        n_features = state.get('n_features', X_train_scaled.shape[2])
        
        config = {
            'seq_len': 20,
            'n_features': n_features,
            'patch_len': 5
        }
        
        # Model 1 학습
        model1_trained = state.get('model1_trained', False)
        
        if not model1_trained:
            print("\n🤖 Model 1: PatchTST 학습")
            model1 = PatchTSTModel(config)
            model1.compile(
                optimizer=Adam(learning_rate=0.001),
                loss=ExtremeLossV4(extreme_focus=False),
                metrics=['mae']
            )
            
            callbacks_model1 = [
                EarlyStopping(patience=20, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6),
                ModelCheckpoint('./checkpoints/model1_v4.h5', save_best_only=True, save_weights_only=True),
                ExtremeValueCallbackV4(X_val_scaled, y_val_scaled, processor.scaler_y)
            ]
            
            history1 = model1.fit(
                X_train_scaled, y_train_scaled,
                validation_data=(X_val_scaled, y_val_scaled),
                sample_weight=weights_train,
                epochs=50,
                batch_size=32,
                callbacks=callbacks_model1,
                verbose=1
            )
            
            state['model1_trained'] = True
            ckpt.save_state(state)
            print("✅ Model 1 학습 완료")
        
        # Model 2 학습
        model2_trained = state.get('model2_trained', False)
        
        if not model2_trained:
            print("\n🤖 Model 2: PatchTST + PINN 학습")
            model2 = PatchTSTPINN(config)
            model2.compile(
                optimizer=Adam(learning_rate=0.0008),
                loss=ExtremeLossV4(extreme_focus=True),
                metrics=['mae']
            )
            
            callbacks_model2 = [
                EarlyStopping(patience=20, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6),
                ModelCheckpoint('./checkpoints/model2_v4.h5', save_best_only=True, save_weights_only=True),
                ExtremeValueCallbackV4((X_val_scaled, X_physics_val_scaled), y_val_scaled, processor.scaler_y, X_physics_val_scaled)
            ]
            
            history2 = model2.fit(
                [X_train_scaled, X_physics_train_scaled], y_train_scaled,
                validation_data=([X_val_scaled, X_physics_val_scaled], y_val_scaled),
                sample_weight=weights_train,
                epochs=60,
                batch_size=32,
                callbacks=callbacks_model2,
                verbose=1
            )
            
            state['model2_trained'] = True
            state['step'] = 6
            ckpt.save_state(state)
            print("✅ Model 2 학습 완료")
    
    # Step 6: 평가
    if step <= 6:
        print("\n[Step 6/6] 모델 평가")
        print("="*80)
        print("📊 V4 최종 평가")
        print("="*80)
        
        # 테스트 데이터 로드
        X_test_scaled = state['X_test_scaled']
        y_test = state['y_test']
        X_physics_test_scaled = state['X_physics_test_scaled']
        n_features = state.get('n_features', X_test_scaled.shape[2])
        
        config = {
            'seq_len': 20,
            'n_features': n_features,
            'patch_len': 5
        }
        
        # 스케일러 로드
        if not processor.load_scalers():
            print("❌ 평가 중단: 스케일러 로드 실패")
            return
        
        # Model 1 평가
        print("\n[Model 1: PatchTST]")
        model1 = PatchTSTModel(config)
        model1.compile(optimizer='adam', loss='mse')
        
        # 더미 데이터로 빌드
        dummy_input = np.zeros((1, 20, n_features))
        _ = model1(dummy_input)
        
        # 가중치 로드
        model1.load_weights('./checkpoints/model1_v4.h5')
        
        y_pred1_scaled = model1.predict(X_test_scaled, verbose=0)
        y_pred1 = processor.scaler_y.inverse_transform(y_pred1_scaled.reshape(-1, 1)).flatten()
        
        mae1 = np.mean(np.abs(y_test - y_pred1))
        rmse1 = np.sqrt(np.mean((y_test - y_pred1)**2))
        print(f"전체 MAE: {mae1:.2f}")
        print(f"전체 RMSE: {rmse1:.2f}")
        
        # 3구간 평가
        print("\n3구간 성능:")
        mask_low = y_test < 200
        mask_normal = (y_test >= 200) & (y_test < 300)
        mask_danger = y_test >= 300
        
        if mask_low.sum() > 0:
            print(f"  저구간(<200): MAE={np.mean(np.abs(y_test[mask_low] - y_pred1[mask_low])):.2f}")
        if mask_normal.sum() > 0:
            print(f"  정상(200-300): MAE={np.mean(np.abs(y_test[mask_normal] - y_pred1[mask_normal])):.2f}")
        if mask_danger.sum() > 0:
            print(f"  위험(300+): MAE={np.mean(np.abs(y_test[mask_danger] - y_pred1[mask_danger])):.2f}")
        
        # 극단값 감지 + False Positive
        print("\n극단값 감지:")
        for threshold in [300, 310, 335]:
            mask = y_test >= threshold
            if mask.sum() > 0:
                detected = (y_pred1 >= threshold)[mask].sum()
                print(f"  {threshold}+: {detected}/{mask.sum()} ({detected/mask.sum()*100:.1f}%)")
        
        # False Positive Rate
        mask_under_300 = y_test < 300
        if mask_under_300.sum() > 0:
            fp_300 = (y_pred1 >= 300)[mask_under_300].sum()
            print(f"\nFalse Positive (300): {fp_300}/{mask_under_300.sum()} ({fp_300/mask_under_300.sum()*100:.1f}%)")
        
        # Model 2 평가
        print("\n[Model 2: PatchTST + PINN]")
        model2 = PatchTSTPINN(config)
        model2.compile(optimizer='adam', loss='mse')
        
        # 더미 데이터로 빌드
        dummy_seq = np.zeros((1, 20, n_features))
        dummy_physics = np.zeros((1, 9))  # V4: 9개 물리 데이터
        _ = model2([dummy_seq, dummy_physics])
        
        # 가중치 로드
        model2.load_weights('./checkpoints/model2_v4.h5')
        
        y_pred2_scaled = model2.predict([X_test_scaled, X_physics_test_scaled], verbose=0)
        y_pred2 = processor.scaler_y.inverse_transform(y_pred2_scaled.reshape(-1, 1)).flatten()
        
        mae2 = np.mean(np.abs(y_test - y_pred2))
        rmse2 = np.sqrt(np.mean((y_test - y_pred2)**2))
        print(f"전체 MAE: {mae2:.2f}")
        print(f"전체 RMSE: {rmse2:.2f}")
        
        # 3구간 평가
        print("\n3구간 성능:")
        if mask_low.sum() > 0:
            print(f"  저구간(<200): MAE={np.mean(np.abs(y_test[mask_low] - y_pred2[mask_low])):.2f}")
        if mask_normal.sum() > 0:
            print(f"  정상(200-300): MAE={np.mean(np.abs(y_test[mask_normal] - y_pred2[mask_normal])):.2f}")
        if mask_danger.sum() > 0:
            print(f"  위험(300+): MAE={np.mean(np.abs(y_test[mask_danger] - y_pred2[mask_danger])):.2f}")
        
        # 극단값 감지 + False Positive
        print("\n극단값 감지:")
        for threshold in [300, 310, 335]:
            mask = y_test >= threshold
            if mask.sum() > 0:
                detected = (y_pred2 >= threshold)[mask].sum()
                print(f"  {threshold}+: {detected}/{mask.sum()} ({detected/mask.sum()*100:.1f}%)")
        
        # False Positive Rate
        if mask_under_300.sum() > 0:
            fp_300 = (y_pred2 >= 300)[mask_under_300].sum()
            print(f"\nFalse Positive (300): {fp_300}/{mask_under_300.sum()} ({fp_300/mask_under_300.sum()*100:.1f}%)")
        
        print("\n" + "="*80)
        print("✅ V4.0 학습 및 평가 완료!")
        print("="*80)
        
        # 완료 후 상태 파일 제거 옵션
        remove = input("\n상태 파일을 제거하시겠습니까? (y/n): ").lower()
        if remove == 'y':
            ckpt.clear_state()
            print("🧹 상태 파일 제거 완료")

if __name__ == "__main__":
    main()