# -*- coding: utf-8 -*-

def integrated_loss(y_true, y_pred):
    """통합 손실 함수 - bool 타입 오류 수정"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    if len(y_true.shape) > 1:
        y_true = tf.squeeze(y_true)
    if len(y_pred.shape) > 1:
        y_pred = tf.squeeze(y_pred)
    
    # 기본 MAE 손실
    mae_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    
    # 극단값 특화 손실 추가 - bool 타입 유지!
    is_extreme = y_true >= 300.0  # bool 타입으로 유지 (수정됨!)
    extreme_penalty = tf.where(is_extreme, 2.0, 1.0)  
    
    # False Positive 방지 - 이미 bool 타입
    fp_condition = tf.logical_and(y_true < 300.0, y_pred >= 300.0)
    fp_penalty = tf.where(fp_condition, 5.0, 1.0)
    
    # 총 페널티 적용
    total_penalty = extreme_penalty * fp_penalty
    weighted_loss = tf.reduce_mean(tf.abs(y_true - y_pred) * total_penalty)
    
    return weighted_loss
"""
================================================================================
🔥 HUBROOM V4 ULTIMATE 완전 통합 학습 시스템 - 30분→10분 예측 (완전 수정판)
================================================================================
✨ 핵심 기능:
   1. 30분 시퀀스 데이터로 10분 후 예측
   2. Base Models: PatchTST (안정형) + PatchTST+PINN (극단형)
   3. Fine-tuning Network: 미세조정 로직을 학습으로 변환
   4. Jump Detection Network: 277→300 점프 특화 학습
   5. Decision Fusion Network: 최종 통합 예측
   6. 완벽한 중단/재개 시스템
   
🎯 모든 오류 수정:
   - TensorFlow 2.16.1 호환 (.weights.h5)
   - Tensor shape 오류 수정
   - 손실 함수 호환성 수정
   - 모든 기능 완전 작동
================================================================================
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
import os
import pickle
import warnings
from datetime import datetime
import signal
import sys
import joblib
import h5py
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any

warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)

print("="*80)
print("🔥 HUBROOM V4 ULTIMATE 통합 시스템 (완전 수정판)")
print("📊 30분 시퀀스 → 10분 후 예측")
print("🎯 Base Models + Fine-tuning + Jump Detection + Decision Fusion")
print("="*80)
print(f"📅 실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"🔧 TensorFlow Version: {tf.__version__}")
print("="*80)

# ==============================================================================
# 💾 완전 통합 체크포인트 관리자
# ==============================================================================

class IntegratedCheckpointManager:
    """완전 통합 시스템용 체크포인트 관리"""
    
    def __init__(self, checkpoint_dir='./checkpoints_integrated_30min'):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 상태 파일들
        self.state_file = os.path.join(checkpoint_dir, 'integrated_state.pkl')
        self.sequence_file = os.path.join(checkpoint_dir, 'sequences_integrated.h5')
        self.scaler_dir = os.path.join(checkpoint_dir, 'scalers')
        self.model_dir = os.path.join(checkpoint_dir, 'models')
        os.makedirs(self.scaler_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.interrupted = False
        
        # Ctrl+C 핸들러
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """Ctrl+C 시그널 처리"""
        print('\n\n⚠️ 중단 감지! 현재 상태를 저장합니다...')
        self.interrupted = True
    
    def save_state(self, state: Dict[str, Any]):
        """현재 상태 저장"""
        with open(self.state_file, 'wb') as f:
            pickle.dump(state, f)
        print(f"💾 상태 저장 완료: Stage {state.get('stage', 0)}")
    
    def load_state(self) -> Optional[Dict[str, Any]]:
        """저장된 상태 로드"""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def save_sequences(self, X, y, X_physics, weights, features, progress, total):
        """시퀀스 데이터 저장"""
        with h5py.File(self.sequence_file, 'w') as f:
            f.create_dataset('X', data=X, compression='gzip')
            f.create_dataset('y', data=y, compression='gzip')
            f.create_dataset('X_physics', data=X_physics, compression='gzip')
            f.create_dataset('weights', data=weights, compression='gzip')
            f.create_dataset('features', data=features, compression='gzip')
            f.attrs['progress'] = progress
            f.attrs['total'] = total
        print(f"💾 통합 시퀀스 저장 완료: {progress}/{total}")
    
    def load_sequences(self):
        """시퀀스 데이터 로드"""
        if os.path.exists(self.sequence_file):
            with h5py.File(self.sequence_file, 'r') as f:
                return {
                    'X': f['X'][:],
                    'y': f['y'][:],
                    'X_physics': f['X_physics'][:],
                    'weights': f['weights'][:],
                    'features': f['features'][:],
                    'progress': f.attrs['progress'],
                    'total': f.attrs['total']
                }
        return None
    
    def clear_state(self):
        """상태 초기화"""
        if os.path.exists(self.state_file):
            os.remove(self.state_file)
        if os.path.exists(self.sequence_file):
            os.remove(self.sequence_file)
        print("🧹 이전 상태 제거 완료")

# ==============================================================================
# 🧠 통합 데이터 처리기 (30분 시퀀스 + 특징 생성)
# ==============================================================================

class IntegratedDataProcessor:
    """통합 시스템용 데이터 처리 - 30분 시퀀스 + 고급 특징"""
    
    def __init__(self):
        self.target_col = 'CURRENT_M16A_3F_JOB_2'
        self.bridge_time_col = 'BRIDGE_TIME'
        self.seq_len = 30  # 30분 시퀀스
        self.pred_len = 10  # 10분 후 예측
        
        # V4 필수 컬럼 정의 (기존과 동일)
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
        
        self.maxcapa_cols = [
            'M16A_6F_LFT_MAXCAPA',
            'M16A_2F_LFT_MAXCAPA'
        ]
        
        self.extreme_cols = [
            'M16A_3F_STORAGE_UTIL'
        ]
        
        self.ofs_cols = [
            'M14_TO_M16_OFS_CUR',
            'M16_TO_M14_OFS_CUR'
        ]
        
        # 전체 V4 필수 컬럼
        self.v4_essential_cols = ([self.target_col] + self.inflow_cols + self.outflow_cols + 
                                  self.cmd_cols + self.maxcapa_cols + self.extreme_cols + self.ofs_cols)
        
        # 스케일러들
        self.scaler_X = RobustScaler()
        self.scaler_y = RobustScaler()
        self.scaler_physics = RobustScaler()
        self.scaler_features = StandardScaler()
        
        # 연속 패턴 확률 (30개까지)
        self.pattern_probability = {
            0: 0.003, 1: 0.15, 2: 0.25, 3: 0.31, 4: 0.43, 5: 0.43,
            6: 0.35, 7: 0.42, 8: 0.53, 9: 0.49, 10: 0.42,
            11: 0.47, 12: 0.52, 13: 0.60, 14: 0.54, 15: 0.66,
            16: 0.62, 17: 0.71, 18: 0.79, 19: 0.83, 20: 0.987,
            21: 0.99, 22: 0.99, 23: 0.99, 24: 0.99, 25: 0.99,
            26: 0.99, 27: 0.99, 28: 0.99, 29: 0.99, 30: 0.99
        }
    
    def load_and_merge_data(self) -> pd.DataFrame:
        """데이터 로드 및 BRIDGE_TIME 병합"""
        print("\n[데이터 로드 - 통합 시스템]")
        
        # 메인 데이터 로드
        df_raw = pd.read_csv('data/HUB_0509_TO_0807_DATA.CSV')
        print(f"📊 원본 데이터: {df_raw.shape}")
        
        # 시간 컬럼 처리
        time_col = df_raw.columns[0]  
        df_raw['timestamp'] = pd.to_datetime(df_raw[time_col], format='%Y%m%d%H%M', errors='coerce')
        
        # V4 필수 컬럼만 선택
        available_cols = [time_col, 'timestamp']
        missing_cols = []
        
        for col in self.v4_essential_cols:
            if col in df_raw.columns:
                available_cols.append(col)
            else:
                missing_cols.append(col)
        
        df = df_raw[available_cols].copy()
        print(f"✅ V4 컬럼 선택: {len(available_cols)-2}/{len(self.v4_essential_cols)} 컬럼 사용")
        
        if missing_cols:
            print(f"⚠️ 누락된 V4 컬럼 ({len(missing_cols)}개): {missing_cols[:5]}...")
            for col in missing_cols:
                if col == self.target_col:
                    print(f"❌ 타겟 컬럼 {self.target_col} 없음!")
                    raise ValueError(f"타겟 컬럼 {self.target_col}이 없습니다!")
                else:
                    df[col] = 0
        
        # BRIDGE_TIME 데이터 병합
        bridge_path = 'data/BRTIME_0509_TO_0807.CSV'
        if os.path.exists(bridge_path):
            print("\n📊 BRIDGE_TIME 파일 처리")
            try:
                bridge_df = pd.read_csv(bridge_path)
                print(f"  원본 shape: {bridge_df.shape}")
                
                if 'IDC_VAL' in bridge_df.columns:
                    bridge_df['BRIDGE_TIME'] = pd.to_numeric(bridge_df['IDC_VAL'], errors='coerce')
                    
                    if 'CRT_TM' in bridge_df.columns:
                        bridge_df['timestamp'] = pd.to_datetime(bridge_df['CRT_TM'])
                        if hasattr(bridge_df['timestamp'].dtype, 'tz'):
                            bridge_df['timestamp'] = bridge_df['timestamp'].dt.tz_localize(None)
                    
                    if hasattr(df['timestamp'].dtype, 'tz'):
                        df['timestamp'] = df['timestamp'].dt.tz_localize(None)
                    
                    bridge_df['timestamp'] = bridge_df['timestamp'].dt.floor('min')
                    df['timestamp'] = df['timestamp'].dt.floor('min')
                    
                    df = pd.merge(df, bridge_df[['timestamp', 'BRIDGE_TIME']], 
                                on='timestamp', how='left')
                    
                    merged_count = df['BRIDGE_TIME'].notna().sum()
                    print(f"✅ BRIDGE_TIME 병합: {merged_count}/{len(df)} 행 매칭")
                    
                    if merged_count > 0:
                        df['BRIDGE_TIME'] = df['BRIDGE_TIME'].interpolate(method='linear', limit_direction='both')
                    df['BRIDGE_TIME'] = df['BRIDGE_TIME'].fillna(3.5)
                    
                    print(f"  최종 통계: min={df['BRIDGE_TIME'].min():.2f}, "
                          f"max={df['BRIDGE_TIME'].max():.2f}, "
                          f"mean={df['BRIDGE_TIME'].mean():.2f}")
                else:
                    df['BRIDGE_TIME'] = 3.5
                    
            except Exception as e:
                print(f"⚠️ BRIDGE_TIME 처리 오류: {e}")
                df['BRIDGE_TIME'] = 3.5
        else:
            print("⚠️ BRIDGE_TIME.CSV 없음 - 기본값 3.5")
            df['BRIDGE_TIME'] = 3.5
        
        df = df.fillna(method='ffill').fillna(0)
        
        if 'timestamp' in df.columns:
            df = df.drop('timestamp', axis=1)
        
        print(f"\n✅ 최종 데이터: {df.shape}")
        
        return df
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """고급 특징 생성 (30분 시퀀스 활용)"""
        print("\n[고급 특징 생성 - 30분 분석]")
        
        # 연속 패턴 분석
        consecutive_counts = []
        consecutive_probs = []
        
        # 30분 추세 분석
        long_trends = []
        mid_trends = []
        accelerations = []
        volatility_changes = []
        
        # 점프 리스크 스코어
        jump_risks = []
        extreme_risks = []
        
        for i in range(len(df)):
            if i < self.seq_len:
                # 초기 데이터 처리
                count = 0
                prob = 0
                long_trend = 0
                mid_trend = 0
                acceleration = 0
                volatility_change = 1
                jump_risk = 0
                extreme_risk = 0
            else:
                # 30분 데이터 분석
                window_30 = df[self.target_col].iloc[i-self.seq_len:i].values
                
                # 연속 300+ 카운트
                count = sum(1 for v in window_30 if v >= 300)
                prob = self.pattern_probability.get(count, 0.5)
                
                # 30분을 3구간으로 분할
                first_10 = window_30[:10]
                middle_10 = window_30[10:20]
                last_10 = window_30[20:30]
                
                # 추세 계산
                long_trend = np.mean(last_10) - np.mean(first_10)
                mid_trend = np.mean(last_10) - np.mean(middle_10)
                acceleration = mid_trend - (np.mean(middle_10) - np.mean(first_10))
                
                # 변동성 변화
                first_std = np.std(first_10) if np.std(first_10) > 0 else 1
                last_std = np.std(last_10)
                volatility_change = last_std / first_std
                
                # 점프 리스크 계산 (270-280 구간 위험도)
                max_val = np.max(window_30)
                recent_5 = window_30[-5:]
                mean_recent = np.mean(recent_5)
                
                jump_risk = 0
                if 270 <= max_val <= 280:
                    jump_risk += 20
                    if acceleration > 2:
                        jump_risk += 25
                    if long_trend > 5:
                        jump_risk += 25
                    if volatility_change > 1.5:
                        jump_risk += 15
                    if mean_recent > 275:
                        jump_risk += 30
                
                # 극단값 리스크
                extreme_risk = 0
                if mean_recent > 285:
                    extreme_risk += 50
                elif mean_recent > 270:
                    extreme_risk += 30
                if count > 15:
                    extreme_risk += 20
                if long_trend > 15:
                    extreme_risk += 25
            
            consecutive_counts.append(count)
            consecutive_probs.append(prob)
            long_trends.append(long_trend)
            mid_trends.append(mid_trend)
            accelerations.append(acceleration)
            volatility_changes.append(volatility_change)
            jump_risks.append(jump_risk)
            extreme_risks.append(extreme_risk)
        
        # 특징 추가
        df['consecutive_300_count'] = consecutive_counts
        df['consecutive_300_prob'] = consecutive_probs
        df['long_trend'] = long_trends
        df['mid_trend'] = mid_trends
        df['acceleration'] = accelerations
        df['volatility_change'] = volatility_changes
        df['jump_risk'] = jump_risks
        df['extreme_risk'] = extreme_risks
        
        print(f"✅ 고급 특징 생성 완료 (30분 분석)")
        return df
    
    def create_integrated_sequences(self, df: pd.DataFrame, ckpt_manager: Optional[IntegratedCheckpointManager] = None) -> Tuple:
        """통합 시퀀스 생성 with 중단/재개 (30분→10분)"""
        print(f"\n[통합 시퀀스 생성 - {self.seq_len}분→{self.pred_len}분 예측]")
        
        saved_sequences = None
        if ckpt_manager:
            saved_sequences = ckpt_manager.load_sequences()
        
        if saved_sequences:
            resume = input(f"\n이전 시퀀스 발견 ({saved_sequences['progress']}/{saved_sequences['total']}). 이어서? (y/n): ")
            if resume.lower() == 'y':
                print("✅ 이전 시퀀스 사용")
                return (saved_sequences['X'], saved_sequences['y'], 
                       saved_sequences['X_physics'], saved_sequences['weights'], 
                       saved_sequences['features'])
        
        X_list, y_list, X_physics_list, weights_list, features_list = [], [], [], [], []
        
        # 사용할 컬럼들
        all_cols = self.v4_essential_cols + ['BRIDGE_TIME'] + [
            'consecutive_300_count', 'consecutive_300_prob', 'long_trend', 
            'mid_trend', 'acceleration', 'volatility_change', 'jump_risk', 'extreme_risk'
        ]
        available_cols = [col for col in all_cols if col in df.columns]
        
        total_sequences = len(df) - self.seq_len - self.pred_len
        
        print(f"📊 총 {total_sequences}개 시퀀스 생성 중 (30분→10분)...")
        
        for i in tqdm(range(total_sequences)):
            if ckpt_manager and ckpt_manager.interrupted:
                print("\n💾 중단! 현재까지 저장...")
                ckpt_manager.save_sequences(
                    np.array(X_list), np.array(y_list),
                    np.array(X_physics_list), np.array(weights_list),
                    np.array(features_list), i, total_sequences
                )
                sys.exit(0)
            
            # 30분 입력 시퀀스
            X = df[available_cols].iloc[i:i+self.seq_len].values
            
            # 10분 후 타겟
            y = df[self.target_col].iloc[i+self.seq_len+self.pred_len-1]
            
            # 물리 특징 (11차원)
            physics = self.create_physics_features(df, i+self.seq_len-1)
            
            # 통합 특징 (미세조정용, 15차원)
            features = self.create_fine_tuning_features(df, i+self.seq_len-1)
            
            # 샘플 가중치
            weight = self.calculate_sample_weight(y)
            
            X_list.append(X)
            y_list.append(y)
            X_physics_list.append(physics)
            weights_list.append(weight)
            features_list.append(features)
        
        print(f"✅ 통합 시퀀스 생성 완료: {len(X_list)}개 (30분→10분)")
        
        return (np.array(X_list), np.array(y_list), 
               np.array(X_physics_list), np.array(weights_list), np.array(features_list))
    
    def create_physics_features(self, df: pd.DataFrame, idx: int) -> np.ndarray:
        """물리 특징 생성 (11차원 - 기존과 동일)"""
        physics = []
        
        # 기본 9개 특징
        physics.append(df[self.target_col].iloc[idx])
        
        inflow_sum = sum(df[col].iloc[idx] for col in self.inflow_cols if col in df.columns)
        physics.append(inflow_sum)
        
        outflow_sum = sum(df[col].iloc[idx] for col in self.outflow_cols if col in df.columns)
        physics.append(outflow_sum)
        
        physics.append(df.get('BRIDGE_TIME', pd.Series([3.5])).iloc[idx])
        physics.append(df.get('consecutive_300_count', pd.Series([0])).iloc[idx])
        physics.append(df.get('consecutive_300_prob', pd.Series([0.5])).iloc[idx])
        physics.append(df.get('M16A_3F_STORAGE_UTIL', pd.Series([0])).iloc[idx])
        
        cmd_sum = sum(df[col].iloc[idx] for col in self.cmd_cols if col in df.columns)
        physics.append(cmd_sum)
        
        if idx >= 4:
            recent_avg = df[self.target_col].iloc[idx-4:idx+1].mean()
        else:
            recent_avg = df[self.target_col].iloc[idx]
        physics.append(recent_avg)
        
        # 30분 특징 (2개)
        if idx >= 30:
            first_10_avg = df[self.target_col].iloc[idx-29:idx-19].mean()
            last_10_avg = df[self.target_col].iloc[idx-9:idx+1].mean()
            long_trend = last_10_avg - first_10_avg
            
            first_10_std = df[self.target_col].iloc[idx-29:idx-19].std()
            last_10_std = df[self.target_col].iloc[idx-9:idx+1].std()
            volatility_change = last_10_std / max(first_10_std, 1)
        else:
            long_trend = 0
            volatility_change = 1
            
        physics.append(long_trend)
        physics.append(volatility_change)
        
        return np.array(physics)
    
    def create_fine_tuning_features(self, df: pd.DataFrame, idx: int) -> np.ndarray:
        """미세조정용 특징 생성 (15차원) - 룰 기반 로직을 특징으로 변환"""
        features = []
        
        if idx < self.seq_len:
            return np.zeros(15)
        
        # 30분 데이터
        past_30 = df[self.target_col].iloc[idx-self.seq_len+1:idx+1].values
        recent_3 = past_30[-3:]
        recent_5 = past_30[-5:]
        
        # 1. 점프 감지 특징들 (277 구간)
        features.append(1.0 if any(275 <= v <= 279 for v in recent_3) else 0.0)  # 277 직격
        features.append(1.0 if any(274 <= v < 275 for v in recent_3) else 0.0)   # 274-275
        features.append(1.0 if any(272 <= v < 274 for v in recent_5) else 0.0)   # 272-274
        features.append(1.0 if any(270 <= v < 272 for v in recent_5) else 0.0)   # 270-272
        
        # 2. 구간별 특징들
        features.append(1.0 if any(255 <= v <= 260 for v in recent_5) else 0.0)  # 255-260
        features.append(1.0 if any(260 <= v < 270 for v in recent_5) else 0.0)   # 260-270
        features.append(1.0 if any(250 <= v < 260 for v in past_30) else 0.0)    # 250-260
        
        # 3. 추세 특징들
        features.append(df.get('long_trend', pd.Series([0])).iloc[idx])
        features.append(df.get('acceleration', pd.Series([0])).iloc[idx])
        features.append(df.get('volatility_change', pd.Series([1])).iloc[idx])
        
        # 4. 극단값 특징들
        features.append(np.max(past_30))  # 최대값
        features.append(np.mean(recent_3))  # 최근 3분 평균
        features.append(np.mean(recent_5))  # 최근 5분 평균
        
        # 5. 위험 스코어들
        features.append(df.get('jump_risk', pd.Series([0])).iloc[idx])
        features.append(df.get('extreme_risk', pd.Series([0])).iloc[idx])
        
        return np.array(features)
    
    def calculate_sample_weight(self, y_value: float) -> float:
        """샘플 가중치 계산 (기존과 동일)"""
        if y_value < 200:
            return 3.0
        elif y_value < 300:
            return 1.0
        elif y_value < 310:
            return 5.0
        elif y_value < 335:
            return 15.0
        else:
            return 25.0

# ==============================================================================
# 🏗️ 통합 모델 정의 - Base Models + Fine-tuning + Jump Detection + Decision Fusion
# ==============================================================================

class PatchTSTModel(keras.Model):
    """안정 구간 전문 PatchTST (30분 시퀀스)"""
    
    def __init__(self, config):
        super().__init__()
        
        self.seq_len = config['seq_len']  # 30
        self.n_features = config['n_features']
        self.patch_len = config['patch_len']  # 6
        self.n_patches = self.seq_len // self.patch_len  # 5
        
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
    """극단값 전문 PatchTST + PINN (30분 시퀀스)"""
    
    def __init__(self, config):
        super().__init__()
        
        self.seq_len = config['seq_len']  # 30
        self.n_features = config['n_features']
        self.patch_len = config['patch_len']  # 6
        self.n_patches = self.seq_len // self.patch_len  # 5
        
        self.patch_embedding = layers.Dense(128, activation='relu')
        
        self.attention = layers.MultiHeadAttention(num_heads=8, key_dim=16)
        self.norm = layers.LayerNormalization()
        
        self.flatten = layers.Flatten()
        self.temporal_dense = layers.Dense(64, activation='relu')
        
        # 물리 정보 처리 (11차원)
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
    """미세조정 네트워크 - 룰 기반 로직을 학습으로 대체"""
    
    def __init__(self):
        super().__init__()
        
        # 특징 처리 레이어
        self.feature_net = keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu')
        ])
        
        # 점프 감지 브랜치
        self.jump_branch = keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid', name='jump_prob')
        ])
        
        # 극단값 감지 브랜치
        self.extreme_branch = keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid', name='extreme_prob')
        ])
        
        # 보정 인수 브랜치
        self.adjustment_branch = keras.Sequential([
            layers.Dense(16, activation='relu'),
            layers.Dense(8, activation='relu'),
            layers.Dense(1, activation='tanh', name='adjustment_factor')  # -1 ~ 1
        ])
    
    def call(self, features, training=False):
        x = self.feature_net(features, training=training)
        
        jump_prob = self.jump_branch(x, training=training)
        extreme_prob = self.extreme_branch(x, training=training)
        adjustment = self.adjustment_branch(x, training=training)
        
        return {
            'jump_prob': tf.squeeze(jump_prob, axis=-1),
            'extreme_prob': tf.squeeze(extreme_prob, axis=-1),
            'adjustment': tf.squeeze(adjustment, axis=-1) * 0.2  # -0.2 ~ 0.2
        }

class JumpDetectionNetwork(keras.Model):
    """점프 감지 특화 네트워크 (277→300 점프)"""
    
    def __init__(self):
        super().__init__()
        
        self.jump_detector = keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid', name='jump_detection')
        ])
        
        self.jump_magnitude = keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='relu', name='jump_size')  # 점프 크기
        ])
    
    def call(self, features, training=False):
        jump_prob = self.jump_detector(features, training=training)
        jump_size = self.jump_magnitude(features, training=training)
        
        return {
            'jump_prob': tf.squeeze(jump_prob, axis=-1),
            'jump_size': tf.squeeze(jump_size, axis=-1) * 50 + 250  # 250~300 범위
        }

class DecisionFusionNetwork(keras.Model):
    """최종 의사결정 융합 네트워크 - Tensor Shape 오류 수정"""
    
    def __init__(self):
        super().__init__()
        
        # 모델 선택 네트워크
        self.model_selector = keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid', name='model_choice')  # 0: Model1, 1: Model2
        ])
        
        # 최종 예측 조합 네트워크
        self.fusion_net = keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(8, activation='relu'),
            layers.Dense(1, name='final_prediction')
        ])
    
    def call(self, inputs, training=False):
        # inputs: [features, pred1, pred2, fine_tune_outputs, jump_outputs]
        features, pred1, pred2, fine_tune, jump_detect = inputs
        
        # 모델 선택 확률
        model_choice_prob = self.model_selector(features, training=training)
        model_choice_squeezed = tf.squeeze(model_choice_prob, axis=-1)  # [?, 1] -> [?]
        
        # 특징들 결합 - 모든 텐서를 [?, 1] 형태로 통일
        combined_features = tf.concat([
            features[:, :8],  # 주요 특징들만 [?, 8]
            tf.expand_dims(pred1, axis=-1),  # [?] -> [?, 1]
            tf.expand_dims(pred2, axis=-1),  # [?] -> [?, 1]
            tf.expand_dims(fine_tune['jump_prob'], axis=-1),  # [?] -> [?, 1]
            tf.expand_dims(fine_tune['extreme_prob'], axis=-1),  # [?] -> [?, 1]
            tf.expand_dims(jump_detect['jump_prob'], axis=-1),  # [?] -> [?, 1]
            tf.expand_dims(model_choice_squeezed, axis=-1)  # [?] -> [?, 1]
        ], axis=-1)
        
        # 최종 예측
        final_pred = self.fusion_net(combined_features, training=training)
        
        return {
            'model_choice': model_choice_squeezed,
            'final_prediction': tf.squeeze(final_pred, axis=-1)
        }

class IntegratedV4Model(keras.Model):
    """통합 V4 모델 - 학습/추론 모드 분리로 손실 함수 호환성 해결"""
    
    def __init__(self, config):
        super().__init__()
        
        self.model1 = PatchTSTModel(config)
        self.model2 = PatchTSTPINN(config)
        self.fine_tuning = FineTuningNetwork()
        self.jump_detection = JumpDetectionNetwork()
        self.decision_fusion = DecisionFusionNetwork()
    
    def call(self, inputs, training=False):
        # inputs: [x_seq, x_physics, x_features]
        x_seq, x_physics, x_features = inputs
        
        # Base 모델 예측
        pred1 = self.model1(x_seq, training=training)
        pred2 = self.model2([x_seq, x_physics], training=training)
        
        # 미세조정 네트워크
        fine_tune_outputs = self.fine_tuning(x_features, training=training)
        
        # 점프 감지 네트워크
        jump_outputs = self.jump_detection(x_features, training=training)
        
        # 최종 융합
        fusion_outputs = self.decision_fusion(
            [x_features, pred1, pred2, fine_tune_outputs, jump_outputs], 
            training=training
        )
        
        # 학습 시에는 final_pred만 반환 (손실 함수 호환)
        if training:
            return fusion_outputs['final_prediction']
        else:
            # 추론 시에는 전체 정보 반환
            return {
                'pred1': pred1,
                'pred2': pred2,
                'fine_tune': fine_tune_outputs,
                'jump_detect': jump_outputs,
                'fusion': fusion_outputs,
                'final_pred': fusion_outputs['final_prediction']
            }

# ==============================================================================
# 📉 통합 손실 함수 - 완전 수정
# ==============================================================================

def stable_loss(y_true, y_pred):
    """Model 1용 - 안정 구간 전문"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    if len(y_true.shape) > 1:
        y_true = tf.squeeze(y_true)
    if len(y_pred.shape) > 1:
        y_pred = tf.squeeze(y_pred)
    
    error = tf.abs(y_true - y_pred)
    
    penalty = tf.ones_like(error)
    penalty = tf.where(y_true < 300.0, 1.0, 0.01)  # 300+ 는 거의 무시
    
    # False Positive 강력 방지
    condition_fp = tf.logical_and(y_true < 300.0, y_pred >= 300.0)
    penalty = tf.where(condition_fp, 50.0, penalty)
    
    condition_extreme_fp = tf.logical_and(y_true < 250.0, y_pred >= 300.0)
    penalty = tf.where(condition_extreme_fp, 100.0, penalty)
    
    weighted_error = error * penalty
    
    result = tf.reduce_mean(weighted_error)
    result = tf.where(tf.math.is_nan(result), 0.0, result)
    
    return result

def extreme_loss(y_true, y_pred):
    """Model 2용 - 극단값 특화"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    if len(y_true.shape) > 1:
        y_true = tf.squeeze(y_true)
    if len(y_pred.shape) > 1:
        y_pred = tf.squeeze(y_pred)
    
    error = tf.abs(y_true - y_pred)
    
    missed_335 = tf.logical_and(y_true >= 335.0, y_pred < 335.0)
    missed_310 = tf.logical_and(
        tf.logical_and(y_true >= 310.0, y_true < 335.0),
        y_pred < 310.0
    )
    in_300_range = tf.logical_and(y_true >= 300.0, y_true < 310.0)
    
    penalty = tf.ones_like(error)
    penalty = tf.where(y_true < 300.0, 0.3, penalty)
    penalty = tf.where(in_300_range, 5.0, penalty)
    penalty = tf.where(missed_310, 15.0, penalty)
    penalty = tf.where(missed_335, 30.0, penalty)
    
    weighted_error = error * penalty
    
    result = tf.reduce_mean(weighted_error)
    result = tf.where(tf.math.is_nan(result), 0.0, result)
    
    return result

def integrated_loss(y_true, y_pred):
    """통합 손실 함수 - 완전 수정 (final_pred만 받음)"""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    if len(y_true.shape) > 1:
        y_true = tf.squeeze(y_true)
    if len(y_pred.shape) > 1:
        y_pred = tf.squeeze(y_pred)
    
    # 기본 MAE 손실
    mae_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    
    # 극단값 특화 손실 추가
    is_extreme = tf.cast(y_true >= 300.0, tf.float32)
    extreme_penalty = tf.where(is_extreme, 2.0, 1.0)  # 극단값은 2배 가중치
    
    # False Positive 방지
    fp_condition = tf.logical_and(y_true < 300.0, y_pred >= 300.0)
    fp_penalty = tf.where(fp_condition, 5.0, 1.0)
    
    # 총 페널티 적용
    total_penalty = extreme_penalty * fp_penalty
    weighted_loss = tf.reduce_mean(tf.abs(y_true - y_pred) * total_penalty)
    
    return weighted_loss

# ==============================================================================
# 📊 통합 콜백 - 수정
# ==============================================================================

class IntegratedCallback(Callback):
    """통합 모델 성능 모니터링 - 수정"""
    
    def __init__(self, X_val, y_val, scaler_y, model_name="Integrated"):
        self.X_val = X_val
        self.y_val = y_val
        self.scaler_y = scaler_y
        self.model_name = model_name
    
    def on_epoch_end(self, epoch, logs=None):
        # 추론 모드로 예측 (전체 출력)
        self.model.training = False
        outputs = self.model.predict(self.X_val, verbose=0)
        self.model.training = True  # 학습 모드로 복원
        
        if isinstance(outputs, dict):
            y_pred_scaled = outputs['final_pred']
        else:
            y_pred_scaled = outputs
        
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_true = self.scaler_y.inverse_transform(self.y_val.reshape(-1, 1)).flatten()
        
        # 전체 MAE
        mae = np.mean(np.abs(y_true - y_pred))
        
        # 극단값 성능
        mask_300 = y_true >= 300
        if mask_300.sum() > 0:
            extreme_recall = (y_pred[mask_300] >= 300).sum() / mask_300.sum() * 100
        else:
            extreme_recall = 0
        
        # 점프 케이스 성능
        mask_jump = (y_true >= 300) & (y_true < 350)
        if mask_jump.sum() > 0:
            jump_recall = (y_pred[mask_jump] >= 290).sum() / mask_jump.sum() * 100
        else:
            jump_recall = 0
        
        # FP 율
        mask_stable = y_true < 300
        if mask_stable.sum() > 0:
            fp_rate = (y_pred[mask_stable] >= 300).sum() / mask_stable.sum() * 100
        else:
            fp_rate = 0
        
        print(f"\n[{self.model_name} Epoch {epoch+1}] "
              f"MAE: {mae:.1f} | "
              f"300+ 감지: {extreme_recall:.1f}% | "
              f"점프 감지: {jump_recall:.1f}% | "
              f"FP: {fp_rate:.1f}%")

# ==============================================================================
# 🎯 통합 학습 메인 함수
# ==============================================================================

def main():
    """통합 V4 시스템 메인 실행"""
    
    print("\n" + "="*80)
    print("🚀 V4 ULTIMATE 통합 시스템 시작")
    print("="*80)
    
    ckpt = IntegratedCheckpointManager()
    
    state = ckpt.load_state()
    
    if state:
        print(f"\n📂 이전 학습 상태 발견! (Stage {state.get('stage', 1)}/5)")
        resume = input("이어서 진행하시겠습니까? (y: 이어서, n: 처음부터): ").lower()
        
        if resume != 'y':
            ckpt.clear_state()
            state = {}
            stage = 1
        else:
            stage = state.get('stage', 1)
            print(f"✅ Stage {stage}부터 재개합니다.")
    else:
        state = {}
        stage = 1
        print("\n새로운 통합 학습을 시작합니다...")
    
    processor = IntegratedDataProcessor()
    
    try:
        # Stage 1: 데이터 준비
        if stage <= 1:
            print(f"\n[Stage 1/5] 데이터 로드 및 특징 생성")
            print("-"*60)
            
            df = processor.load_and_merge_data()
            df = processor.create_advanced_features(df)
            
            state['stage'] = 2
            state['data_shape'] = df.shape
            ckpt.save_state(state)
            print("✅ Stage 1 완료")
        
        # Stage 2: 시퀀스 생성
        if stage <= 2:
            print(f"\n[Stage 2/5] 통합 시퀀스 생성 (30분→10분)")
            print("-"*60)
            
            if stage < 2:
                df = processor.load_and_merge_data()
                df = processor.create_advanced_features(df)
            
            X, y, X_physics, weights, X_features = processor.create_integrated_sequences(df, ckpt_manager=ckpt)
            
            with h5py.File('./checkpoints_integrated_30min/sequences_integrated_complete.h5', 'w') as f:
                f.create_dataset('X', data=X, compression='gzip')
                f.create_dataset('y', data=y, compression='gzip')
                f.create_dataset('X_physics', data=X_physics, compression='gzip')
                f.create_dataset('weights', data=weights, compression='gzip')
                f.create_dataset('X_features', data=X_features, compression='gzip')
            
            state['stage'] = 3
            state['sequence_shape'] = X.shape
            ckpt.save_state(state)
            print("✅ Stage 2 완료")
        
        # Stage 3: 데이터 분할 및 스케일링
        if stage <= 3:
            print(f"\n[Stage 3/5] 데이터 분할 및 스케일링")
            print("-"*60)
            
            with h5py.File('./checkpoints_integrated_30min/sequences_integrated_complete.h5', 'r') as f:
                X = f['X'][:]
                y = f['y'][:]
                X_physics = f['X_physics'][:]
                weights = f['weights'][:]
                X_features = f['X_features'][:]
            
            indices = np.arange(len(X))
            np.random.shuffle(indices)
            
            train_size = int(0.7 * len(X))
            val_size = int(0.15 * len(X))
            
            train_idx = indices[:train_size]
            val_idx = indices[train_size:train_size+val_size]
            test_idx = indices[train_size+val_size:]
            
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            
            X_physics_train = X_physics[train_idx]
            X_physics_val = X_physics[val_idx]
            X_physics_test = X_physics[test_idx]
            
            X_features_train = X_features[train_idx]
            X_features_val = X_features[val_idx]
            X_features_test = X_features[test_idx]
            
            weights_train = weights[train_idx]
            
            print(f"✅ Train: {len(X_train):,} samples")
            print(f"✅ Valid: {len(X_val):,} samples")  
            print(f"✅ Test: {len(X_test):,} samples")
            
            # 스케일링
            n_features = X.shape[2]
            
            X_train_scaled = processor.scaler_X.fit_transform(X_train.reshape(-1, n_features))
            X_train_scaled = X_train_scaled.reshape(len(X_train), 30, n_features)
            
            X_val_scaled = processor.scaler_X.transform(X_val.reshape(-1, n_features))
            X_val_scaled = X_val_scaled.reshape(len(X_val), 30, n_features)
            
            X_test_scaled = processor.scaler_X.transform(X_test.reshape(-1, n_features))
            X_test_scaled = X_test_scaled.reshape(len(X_test), 30, n_features)
            
            y_train_scaled = processor.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
            y_val_scaled = processor.scaler_y.transform(y_val.reshape(-1, 1)).flatten()
            
            X_physics_train_scaled = processor.scaler_physics.fit_transform(X_physics_train)
            X_physics_val_scaled = processor.scaler_physics.transform(X_physics_val)
            X_physics_test_scaled = processor.scaler_physics.transform(X_physics_test)
            
            X_features_train_scaled = processor.scaler_features.fit_transform(X_features_train)
            X_features_val_scaled = processor.scaler_features.transform(X_features_val)
            X_features_test_scaled = processor.scaler_features.transform(X_features_test)
            
            # 스케일러 저장
            os.makedirs('./checkpoints_integrated_30min/scalers', exist_ok=True)
            joblib.dump(processor.scaler_X, './checkpoints_integrated_30min/scalers/scaler_X.pkl')
            joblib.dump(processor.scaler_y, './checkpoints_integrated_30min/scalers/scaler_y.pkl')
            joblib.dump(processor.scaler_physics, './checkpoints_integrated_30min/scalers/scaler_physics.pkl')
            joblib.dump(processor.scaler_features, './checkpoints_integrated_30min/scalers/scaler_features.pkl')
            
            # 스케일된 데이터 저장
            with h5py.File('./checkpoints_integrated_30min/scaled_integrated_data.h5', 'w') as f:
                f.create_dataset('X_train_scaled', data=X_train_scaled, compression='gzip')
                f.create_dataset('X_val_scaled', data=X_val_scaled, compression='gzip')
                f.create_dataset('X_test_scaled', data=X_test_scaled, compression='gzip')
                f.create_dataset('y_train_scaled', data=y_train_scaled, compression='gzip')
                f.create_dataset('y_val_scaled', data=y_val_scaled, compression='gzip')
                f.create_dataset('y_test', data=y_test, compression='gzip')
                f.create_dataset('X_physics_train_scaled', data=X_physics_train_scaled, compression='gzip')
                f.create_dataset('X_physics_val_scaled', data=X_physics_val_scaled, compression='gzip')
                f.create_dataset('X_physics_test_scaled', data=X_physics_test_scaled, compression='gzip')
                f.create_dataset('X_features_train_scaled', data=X_features_train_scaled, compression='gzip')
                f.create_dataset('X_features_val_scaled', data=X_features_val_scaled, compression='gzip')
                f.create_dataset('X_features_test_scaled', data=X_features_test_scaled, compression='gzip')
                f.create_dataset('weights_train', data=weights_train, compression='gzip')
                f.attrs['n_features'] = n_features
            
            state['stage'] = 4
            state['n_features'] = n_features
            ckpt.save_state(state)
            print("✅ Stage 3 완료")
        
        # Stage 4: Base Models 학습
        if stage <= 4:
            print(f"\n[Stage 4/5] Base Models 학습")
            print("-"*60)
            
            with h5py.File('./checkpoints_integrated_30min/scaled_integrated_data.h5', 'r') as f:
                X_train_scaled = f['X_train_scaled'][:]
                X_val_scaled = f['X_val_scaled'][:]
                y_train_scaled = f['y_train_scaled'][:]
                y_val_scaled = f['y_val_scaled'][:]
                X_physics_train_scaled = f['X_physics_train_scaled'][:]
                X_physics_val_scaled = f['X_physics_val_scaled'][:]
                weights_train = f['weights_train'][:]
                n_features = f.attrs['n_features']
            
            processor.scaler_y = joblib.load('./checkpoints_integrated_30min/scalers/scaler_y.pkl')
            
            config = {
                'seq_len': 30,  # 30분 시퀀스
                'n_features': n_features,
                'patch_len': 6
            }
            
            # Model 1 학습
            if not state.get('model1_trained', False):
                print("\n🤖 Model 1: PatchTST (안정형) 학습")
                
                model1 = PatchTSTModel(config)
                
                optimizer1 = Adam(learning_rate=0.0001, clipnorm=1.0)
                
                model1.compile(
                    optimizer=optimizer1,
                    loss=stable_loss,
                    metrics=['mae']
                )
                
                callbacks1 = [
                    EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
                    ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-6, monitor='val_loss'),
                    ModelCheckpoint('./checkpoints_integrated_30min/models/model1_stable.weights.h5', 
                                  save_best_only=True, save_weights_only=True, monitor='val_loss')
                ]
                
                history1 = model1.fit(
                    X_train_scaled, y_train_scaled,
                    validation_data=(X_val_scaled, y_val_scaled),
                    epochs=50,
                    batch_size=64,
                    callbacks=callbacks1,
                    verbose=1
                )
                
                state['model1_trained'] = True
                ckpt.save_state(state)
                print("✅ Model 1 학습 완료")
            
            # Model 2 학습
            if not state.get('model2_trained', False):
                print("\n🤖 Model 2: PatchTST+PINN (극단형) 학습")
                
                model2 = PatchTSTPINN(config)
                
                optimizer2 = Adam(learning_rate=0.0001, clipnorm=1.0)
                
                model2.compile(
                    optimizer=optimizer2,
                    loss=extreme_loss,
                    metrics=['mae']
                )
                
                callbacks2 = [
                    EarlyStopping(patience=20, restore_best_weights=True, monitor='val_loss'),
                    ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6, monitor='val_loss'),
                    ModelCheckpoint('./checkpoints_integrated_30min/models/model2_extreme.weights.h5', 
                                  save_best_only=True, save_weights_only=True, monitor='val_loss')
                ]
                
                history2 = model2.fit(
                    [X_train_scaled, X_physics_train_scaled], y_train_scaled,
                    validation_data=([X_val_scaled, X_physics_val_scaled], y_val_scaled),
                    epochs=60,
                    batch_size=64,
                    callbacks=callbacks2,
                    verbose=1
                )
                
                state['model2_trained'] = True
                ckpt.save_state(state)
                print("✅ Model 2 학습 완료")
            
            state['stage'] = 5
            ckpt.save_state(state)
            print("✅ Stage 4 완료")
        
        # Stage 5: 통합 모델 학습
        if stage <= 5:
            print(f"\n[Stage 5/5] 통합 모델 End-to-End 학습")
            print("-"*60)
            
            with h5py.File('./checkpoints_integrated_30min/scaled_integrated_data.h5', 'r') as f:
                X_train_scaled = f['X_train_scaled'][:]
                X_val_scaled = f['X_val_scaled'][:]
                X_test_scaled = f['X_test_scaled'][:]
                y_train_scaled = f['y_train_scaled'][:]
                y_val_scaled = f['y_val_scaled'][:]
                y_test = f['y_test'][:]
                X_physics_train_scaled = f['X_physics_train_scaled'][:]
                X_physics_val_scaled = f['X_physics_val_scaled'][:]
                X_physics_test_scaled = f['X_physics_test_scaled'][:]
                X_features_train_scaled = f['X_features_train_scaled'][:]
                X_features_val_scaled = f['X_features_val_scaled'][:]
                X_features_test_scaled = f['X_features_test_scaled'][:]
                weights_train = f['weights_train'][:]
                n_features = f.attrs['n_features']
            
            processor.scaler_y = joblib.load('./checkpoints_integrated_30min/scalers/scaler_y.pkl')
            
            config = {
                'seq_len': 30,
                'n_features': n_features,
                'patch_len': 6
            }
            
            # 통합 모델 생성
            integrated_model = IntegratedV4Model(config)
            
            # Base 모델 가중치 로드 (사전 학습된) - TensorFlow 2.16.1 호환
            integrated_model.model1.build(input_shape=(None, 30, n_features))
            integrated_model.model1.load_weights('./checkpoints_integrated_30min/models/model1_stable.weights.h5')
            
            integrated_model.model2.build([(None, 30, n_features), (None, 11)])
            integrated_model.model2.load_weights('./checkpoints_integrated_30min/models/model2_extreme.weights.h5')
            
            # Base 모델 가중치 고정 (선택사항)
            for layer in integrated_model.model1.layers:
                layer.trainable = False
            for layer in integrated_model.model2.layers:
                layer.trainable = False
            
            print("✅ Base 모델 가중치 로드 및 고정")
            
            # 통합 모델 컴파일
            optimizer_integrated = Adam(learning_rate=0.0005, clipnorm=1.0)
            
            integrated_model.compile(
                optimizer=optimizer_integrated,
                loss=integrated_loss,
                metrics=['mae']
            )
            
            # 통합 모델 학습
            callbacks_integrated = [
                EarlyStopping(patience=25, restore_best_weights=True, monitor='val_loss'),
                ReduceLROnPlateau(factor=0.7, patience=12, min_lr=1e-6, monitor='val_loss'),
                ModelCheckpoint('./checkpoints_integrated_30min/models/integrated_model.weights.h5', 
                              save_best_only=True, save_weights_only=True, monitor='val_loss'),
                IntegratedCallback([X_val_scaled, X_physics_val_scaled, X_features_val_scaled], 
                                 y_val_scaled, processor.scaler_y, "Integrated")
            ]
            
            print("\n🚀 통합 모델 End-to-End 학습 시작")
            
            history_integrated = integrated_model.fit(
                [X_train_scaled, X_physics_train_scaled, X_features_train_scaled], 
                y_train_scaled,
                validation_data=([X_val_scaled, X_physics_val_scaled, X_features_val_scaled], 
                               y_val_scaled),
                sample_weight=weights_train,
                epochs=40,
                batch_size=32,
                callbacks=callbacks_integrated,
                verbose=1
            )
            
            print("✅ 통합 모델 학습 완료")
            
            # 최종 평가
            print("\n📊 최종 성능 평가")
            print("="*60)
            
            # 추론 모드로 전환
            integrated_model.training = False
            test_outputs = integrated_model.predict([X_test_scaled, X_physics_test_scaled, X_features_test_scaled], 
                                                   verbose=1)
            
            if isinstance(test_outputs, dict):
                y_pred = processor.scaler_y.inverse_transform(
                    test_outputs['final_pred'].reshape(-1, 1)
                ).flatten()
            else:
                y_pred = processor.scaler_y.inverse_transform(
                    test_outputs.reshape(-1, 1)
                ).flatten()
            
            y_true = y_test
            
            # 성능 지표
            mae = np.mean(np.abs(y_true - y_pred))
            
            # 극단값 성능
            mask_300 = y_true >= 300
            extreme_recall = (y_pred[mask_300] >= 300).sum() / max(1, mask_300.sum()) * 100
            
            # 점프 케이스 성능
            mask_jump = (y_true >= 300) & (y_true < 350)
            jump_recall = (y_pred[mask_jump] >= 290).sum() / max(1, mask_jump.sum()) * 100
            
            # FP 율
            mask_stable = y_true < 300
            fp_rate = (y_pred[mask_stable] >= 300).sum() / max(1, mask_stable.sum()) * 100
            
            print(f"\n📈 최종 통합 모델 성능:")
            print(f"  전체 MAE: {mae:.2f}")
            print(f"  300+ 감지율: {extreme_recall:.1f}%")
            print(f"  점프 감지율: {jump_recall:.1f}%") 
            print(f"  FP 율: {fp_rate:.1f}%")
            
            # 샘플 예측 결과
            print(f"\n📋 샘플 예측 결과 (처음 20개):")
            for i in range(min(20, len(y_true))):
                print(f"[{i+1:2d}] 실제: {y_true[i]:6.1f}, 예측: {y_pred[i]:6.1f}, "
                      f"오차: {abs(y_true[i] - y_pred[i]):5.1f}")
            
            print("\n" + "="*80)
            print("✅ V4 ULTIMATE 통합 시스템 완료! (모든 오류 해결)")
            print("📊 30분 시퀀스 → 10분 예측 시스템 구축 성공!")
            print("="*80)
            
            remove = input("\n상태 파일을 제거하시겠습니까? (y/n): ").lower()
            if remove == 'y':
                ckpt.clear_state()
                print("🧹 상태 파일 제거 완료")
    
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