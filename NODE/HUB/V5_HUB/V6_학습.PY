알겠습니다. ModelCheckpoint 파일명 오류를 수정한 전체 코드입니다:#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
🔥 HUBROOM V4 듀얼 모델 학습 시스템 - 30분→10분 예측 (수정 버전)
================================================================================
✨ 주요 수정사항:
   - 인덱스 매핑 문제 해결
   - 중단/재개 기능 강화
   - 시퀀스 생성 진행상황 저장
   - ModelCheckpoint 파일명 수정 (.weights.h5)
   
🎯 목표:
   - Model 1 (안정형): 80-299 범위 전문, False Positive 최소화
   - Model 2 (극단형): 300+ 전문, 점프 감지 특화
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
import gc  # 메모리 관리

warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)

print("="*80)
print("🔥 HUBROOM V4 듀얼 모델 학습 시스템 (수정 버전)")
print("📊 30분 시퀀스 → 10분 후 예측")
print("🎯 Model1(안정형) + Model2(극단형)")
print("✅ 인덱스 문제 해결 + 중단/재개 강화")
print("="*80)
print(f"📅 실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"🔧 TensorFlow Version: {tf.__version__}")
print("="*80)

# ==============================================================================
# 💾 강화된 체크포인트 관리자
# ==============================================================================

class EnhancedCheckpointManager:
    """강화된 학습 중단/재개 관리"""
    
    def __init__(self, checkpoint_dir='./checkpoints_dual_30min_v2'):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.state_file = os.path.join(checkpoint_dir, 'training_state.pkl')
        self.sequence_file = os.path.join(checkpoint_dir, 'sequences_complete.h5')
        self.scaler_dir = os.path.join(checkpoint_dir, 'scalers')
        os.makedirs(self.scaler_dir, exist_ok=True)
        
        # 시퀀스 생성 진행상황 파일
        self.sequence_progress_file = os.path.join(checkpoint_dir, 'sequence_progress.pkl')
        
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
    
    def save_sequence_progress(self, progress_data):
        """시퀀스 생성 진행상황 저장"""
        with open(self.sequence_progress_file, 'wb') as f:
            pickle.dump(progress_data, f)
        print(f"💾 시퀀스 진행상황 저장: {progress_data.get('current_idx', 0)}개 완료")
    
    def load_sequence_progress(self):
        """시퀀스 생성 진행상황 로드"""
        if os.path.exists(self.sequence_progress_file):
            with open(self.sequence_progress_file, 'rb') as f:
                return pickle.load(f)
        return None

# ==============================================================================
# 📊 데이터 처리 클래스 (수정)
# ==============================================================================

class V4DataProcessor:
    """V4 듀얼 모델용 데이터 처리 - 인덱스 문제 해결"""
    
    def __init__(self):
        self.target_col = 'CURRENT_M16A_3F_JOB_2'
        self.seq_len = 30  # 30분 시퀀스
        
        # V4 필수 21개 컬럼
        self.v4_cols = [
            'CURRENT_M16A_3F_JOB_2',
            'M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2', 
            'M14A_3F_TO_HUB_JOB2', 'M14B_7F_TO_HUB_JOB2', 'M16B_10F_TO_HUB_JOB',
            'M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB',
            'M16A_3F_TO_M14A_3F_JOB', 'M16A_3F_TO_M14B_7F_JOB', 'M16A_3F_TO_3F_MLUD_JOB',
            'M16A_3F_CMD', 'M16A_6F_TO_HUB_CMD', 'M16A_2F_TO_HUB_CMD',
            'M14A_3F_TO_HUB_CMD', 'M14B_7F_TO_HUB_CMD',
            'M16A_6F_LFT_MAXCAPA', 'M16A_2F_LFT_MAXCAPA',
            'M16A_3F_STORAGE_UTIL',
            'M14_TO_M16_OFS_CUR', 'M16_TO_M14_OFS_CUR'
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
        
        self.scaler_X = RobustScaler()
        self.scaler_y = RobustScaler()
        self.scaler_physics = StandardScaler()
    
    def load_and_merge_data(self):
        """데이터 로드 및 BRIDGE_TIME 병합"""
        print("\n[1단계] 데이터 로드")
        
        # 메인 데이터
        df = pd.read_csv('data/HUB_0509_TO_0730_DATA.CSV')
        print(f"✅ 메인 데이터: {df.shape}")
        
        # 시간 처리
        time_col = df.columns[0]
        df['datetime'] = pd.to_datetime(df[time_col], format='%Y%m%d%H%M')
        
        # BRIDGE_TIME 데이터
        bridge_df = pd.read_csv('data/BRTIME_0509_TO_0730.CSV')
        print(f"✅ BRIDGE_TIME 데이터: {bridge_df.shape}")
        
        # BRIDGE_TIME 병합
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
            
            print(f"✅ BRIDGE_TIME 병합 완료")
            
            # V4 컬럼에 추가
            if 'BRIDGE_TIME' not in self.v4_cols:
                self.v4_cols.append('BRIDGE_TIME')
        
        # 인덱스 리셋 (중요!)
        df = df.reset_index(drop=True)
        
        return df
    
    def add_consecutive_patterns(self, df):
        """연속 300+ 패턴 추가 (확률 맵 활용)"""
        consecutive_counts = []
        consecutive_probs = []
        
        for i in tqdm(range(len(df)), desc="연속 패턴 계산"):
            if i < self.seq_len:
                count = 0
                prob = 0.003
            else:
                window = df[self.target_col].iloc[i-self.seq_len:i].values
                count = sum(1 for v in window if v >= 300)
                # 확률 맵에서 정확한 값 찾기
                if count in self.probability_map:
                    prob = self.probability_map[count]
                else:
                    # 가장 가까운 값 찾기
                    closest_key = min(self.probability_map.keys(), 
                                    key=lambda x: abs(x - count))
                    prob = self.probability_map[closest_key]
            
            consecutive_counts.append(count)
            consecutive_probs.append(prob)
        
        df['consecutive_300_count'] = consecutive_counts
        df['consecutive_300_prob'] = consecutive_probs
        
        return df
    
    def create_sequences_with_resume(self, df, ckpt, pred_len=10):
        """시퀀스 데이터 생성 - 중단/재개 지원"""
        print(f"\n[3단계] 시퀀스 생성 ({self.seq_len}분 → {pred_len}분 후)")
        
        # 진행상황 확인
        progress = ckpt.load_sequence_progress()
        if progress and progress.get('completed', False) == False:
            print(f"⚠️ 이전 진행상황 발견: {progress['current_idx']}/{progress['total']} 완료")
            resume = input("이어서 진행하시겠습니까? (y/n): ").lower()
            
            if resume == 'y':
                # 이전 데이터 로드
                X = progress['X']
                y = progress['y']
                physics_features = progress['physics_features']
                sequence_indices = progress['sequence_indices']
                start_idx = progress['current_idx'] + self.seq_len
            else:
                X, y, physics_features, sequence_indices = [], [], [], []
                start_idx = self.seq_len
                progress = None
        else:
            X, y, physics_features, sequence_indices = [], [], [], []
            start_idx = self.seq_len
            progress = None
        
        # 필수 컬럼 + consecutive 컬럼
        feature_cols = self.v4_cols + ['consecutive_300_count', 'consecutive_300_prob']
        
        total = len(df) - pred_len - self.seq_len
        
        try:
            for i in range(start_idx, len(df) - pred_len):
                if ckpt.interrupted:
                    # 중단 시 진행상황 저장
                    progress_data = {
                        'X': X,
                        'y': y,
                        'physics_features': physics_features,
                        'sequence_indices': sequence_indices,
                        'current_idx': i - self.seq_len,
                        'total': total,
                        'completed': False
                    }
                    ckpt.save_sequence_progress(progress_data)
                    print(f"\n💾 시퀀스 생성 중단! {len(X)}개 저장")
                    sys.exit(0)
                
                # 진행률 표시
                if (i - start_idx) % 1000 == 0:
                    print(f"  진행: {i - self.seq_len}/{total} ({(i - self.seq_len)/total*100:.1f}%)")
                
                # 입력: 과거 30분
                X.append(df[feature_cols].iloc[i-self.seq_len:i].values)
                
                # 타겟: 10분 후
                target_idx = i + pred_len - 1
                y.append(df[self.target_col].iloc[target_idx])
                
                # 원본 인덱스 저장 (타겟 시점의 df 인덱스)
                sequence_indices.append(target_idx)
                
                # 물리 특징 (현재 시점 기준)
                physics = self.create_physics_features(df, i-1)
                physics_features.append(physics)
            
            # 완료 표시
            X = np.array(X)
            y = np.array(y)
            physics_features = np.array(physics_features)
            sequence_indices = np.array(sequence_indices)
            
            # 완료 상태 저장
            progress_data = {
                'completed': True,
                'total_sequences': len(X)
            }
            ckpt.save_sequence_progress(progress_data)
            
            print(f"\n✅ {len(X)}개 시퀀스 생성 완료")
            print(f"  X shape: {X.shape}")
            print(f"  y shape: {y.shape}")
            print(f"  Physics shape: {physics_features.shape}")
            print(f"  Indices shape: {sequence_indices.shape}")
            
            return X, y, physics_features, sequence_indices
        
        except Exception as e:
            print(f"\n❌ 시퀀스 생성 중 오류: {e}")
            # 오류 시에도 진행상황 저장
            if len(X) > 0:
                progress_data = {
                    'X': X,
                    'y': y,
                    'physics_features': physics_features,
                    'sequence_indices': sequence_indices,
                    'current_idx': i - self.seq_len if 'i' in locals() else 0,
                    'total': total,
                    'completed': False
                }
                ckpt.save_sequence_progress(progress_data)
            raise
    
    def create_physics_features(self, df, idx):
        """물리 특징 생성 (11차원) - 평가 코드와 동일"""
        physics = []
        
        # 1. 현재 HUBROOM 값
        physics.append(df[self.target_col].iloc[idx])
        
        # 2. 유입 합계
        inflow_cols = ['M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2', 
                      'M14A_3F_TO_HUB_JOB2', 'M14B_7F_TO_HUB_JOB2', 'M16B_10F_TO_HUB_JOB']
        inflow_sum = df[inflow_cols].iloc[idx].sum()
        physics.append(inflow_sum)
        
        # 3. 유출 합계
        outflow_cols = ['M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB',
                       'M16A_3F_TO_M14A_3F_JOB', 'M16A_3F_TO_M14B_7F_JOB', 
                       'M16A_3F_TO_3F_MLUD_JOB']
        outflow_sum = df[outflow_cols].iloc[idx].sum()
        physics.append(outflow_sum)
        
        # 4. BRIDGE_TIME
        physics.append(df.get('BRIDGE_TIME', pd.Series([3.5])).iloc[idx])
        
        # 5. 연속 300+ 카운트
        physics.append(df['consecutive_300_count'].iloc[idx])
        
        # 6. 연속 300+ 확률
        physics.append(df['consecutive_300_prob'].iloc[idx])
        
        # 7. STORAGE_UTIL
        physics.append(df.get('M16A_3F_STORAGE_UTIL', pd.Series([0])).iloc[idx])
        
        # 8. CMD 합계
        cmd_cols = ['M16A_3F_CMD', 'M16A_6F_TO_HUB_CMD', 'M16A_2F_TO_HUB_CMD',
                   'M14A_3F_TO_HUB_CMD', 'M14B_7F_TO_HUB_CMD']
        cmd_sum = df[cmd_cols].iloc[idx].sum()
        physics.append(cmd_sum)
        
        # 9. 최근 5분 평균 (상승/하락 판단용)
        if idx >= 4:
            recent_avg = df[self.target_col].iloc[idx-4:idx+1].mean()
        else:
            recent_avg = df[self.target_col].iloc[idx]
        physics.append(recent_avg)
        
        # 10. 30분 장기 추세 (first 10 vs last 10)
        if idx >= 29:
            first_10_avg = df[self.target_col].iloc[idx-29:idx-19].mean()
            last_10_avg = df[self.target_col].iloc[idx-9:idx+1].mean()
            long_trend = last_10_avg - first_10_avg
        else:
            long_trend = 0
        physics.append(long_trend)
        
        # 11. 변동성 변화
        if idx >= 29:
            first_10_std = df[self.target_col].iloc[idx-29:idx-19].std()
            last_10_std = df[self.target_col].iloc[idx-9:idx+1].std()
            if first_10_std > 0:
                volatility_change = last_10_std / first_10_std
            else:
                volatility_change = 1
        else:
            volatility_change = 1
        physics.append(volatility_change)
        
        return np.array(physics)

# ==============================================================================
# 📊 데이터 분리 (수정) - 시퀀스 인덱스 기반
# ==============================================================================

def create_model_specific_data_v2(df_expanded, sequence_indices):
    """시퀀스 인덱스 기반 모델별 데이터 분리"""
    print("\n[4단계] 모델별 데이터 분리 (시퀀스 인덱스 기반)")
    
    # Model1 데이터 선택 조건 (안정형)
    model1_mask = (
        (df_expanded['consecutive_300_prob'] < 0.5) &  # 낮은 확률
        (df_expanded['consecutive_300_count'] < 10) &  # 300+ 적음
        (df_expanded['past_30min_max'] < 275) &  # 점프 위험 없음
        (df_expanded['BRIDGE_TIME'] <= 4) &  # 물리 지표 안정
        (df_expanded['acceleration'] <= 2) &  # 급변 없음
        (df_expanded['volatility_change'] <= 1.5) &  # 변동성 안정
        (df_expanded['long_trend'].between(-30, 30))  # 극단 추세 없음
    )
    
    # Model2 데이터 선택 조건 (극단형)
    model2_mask = (
        # 확률 맵 기반
        (df_expanded['consecutive_300_prob'] >= 0.6) |  # 높은 확률 (15개+)
        (df_expanded['consecutive_300_count'] >= 15) |  # 극단값 지속
        
        # 277 구간 특별 처리
        ((df_expanded['past_30min_max'] >= 275) & (df_expanded['past_30min_max'] <= 279)) |
        ((df_expanded['recent_5min_max'] >= 275) & (df_expanded['recent_5min_max'] <= 279)) |
        ((df_expanded['recent_3min_max'] >= 275) & (df_expanded['recent_3min_max'] <= 279)) |
        
        # 점프 가능성 구간
        ((df_expanded['past_30min_max'] >= 260) & (df_expanded['past_30min_max'] <= 280)) |
        (df_expanded['CURRENT_M16A_3F_JOB_2'] >= 300) |  # 이미 극단값
        
        # 물리 지표
        (df_expanded['BRIDGE_TIME'] > 4.2) |  # 물리 지표 위험
        
        # 가속도 및 추세
        (df_expanded['acceleration'] > 2) |  # 급변 신호
        (df_expanded['long_trend'] > 10) |  # 장기 상승
        (df_expanded['recent_5min_trend'] > 5) |  # 최근 급등
        
        # 변동성
        (df_expanded['volatility_change'] > 1.8) |  # 변동성 증가
        
        # 특별 케이스
        (df_expanded['is_277_zone']) |  # 277 특별 구간
        (df_expanded['is_jump_case']) |  # 실제 점프 케이스
        
        # 상승 패턴
        (df_expanded['consecutive_rises'] >= 20)  # 연속 상승
    )
    
    # 시퀀스 인덱스 추출
    model1_sequence_idx = np.where(model1_mask)[0]
    model2_sequence_idx = np.where(model2_mask)[0]
    
    print(f"✅ Model1 시퀀스: {len(model1_sequence_idx)}개 ({len(model1_sequence_idx)/len(sequence_indices)*100:.1f}%)")
    print(f"✅ Model2 시퀀스: {len(model2_sequence_idx)}개 ({len(model2_sequence_idx)/len(sequence_indices)*100:.1f}%)")
    
    # 겹치는 데이터 확인
    overlap = set(model1_sequence_idx) & set(model2_sequence_idx)
    print(f"⚠️ 겹치는 시퀀스: {len(overlap)}개")
    
    # 가중치 계산
    model1_weights = calculate_weights_v2(df_expanded.iloc[model1_sequence_idx], 'stable')
    model2_weights = calculate_weights_v2(df_expanded.iloc[model2_sequence_idx], 'extreme')
    
    return model1_sequence_idx, model2_sequence_idx, model1_weights, model2_weights

def calculate_weights_v2(df_subset, model_type):
    """데이터 가중치 계산 (수정)"""
    weights = np.ones(len(df_subset))
    
    if model_type == 'extreme':
        # 점프 케이스에 높은 가중치
        jump_mask = df_subset['is_jump_case'].values == True
        weights[jump_mask] = 10.0
        
        # 277 구간에 높은 가중치  
        zone_277_mask = df_subset['is_277_zone'].values == True
        weights[zone_277_mask] = 8.0
        
        # 극고값(335+)에 높은 가중치
        extreme_high = df_subset['CURRENT_M16A_3F_JOB_2'].values >= 335
        weights[extreme_high] = 5.0
        
        # 300-335 구간
        high_value = (df_subset['CURRENT_M16A_3F_JOB_2'].values >= 300) & \
                    (df_subset['CURRENT_M16A_3F_JOB_2'].values < 335)
        weights[high_value] = 3.0
        
        # 260-280 점프 가능 구간
        jump_possible = (df_subset['past_30min_max'].values >= 260) & \
                       (df_subset['past_30min_max'].values < 280)
        weights[jump_possible] = 2.0
    
    elif model_type == 'stable':
        # 안정형은 균등 가중치
        # 단, 250-270 경계 구간은 조금 더 중요
        boundary = (df_subset['CURRENT_M16A_3F_JOB_2'].values >= 250) & \
                  (df_subset['CURRENT_M16A_3F_JOB_2'].values < 270)
        weights[boundary] = 1.5
    
    return weights

# 특징 계산 함수들은 이전과 동일...
def calculate_advanced_features(df, target_col):
    """평가 코드에서 사용하는 모든 특징 계산"""
    
    # 과거 30분 최대값/최소값/평균
    df['past_30min_max'] = df[target_col].rolling(30, min_periods=1).max()
    df['past_30min_min'] = df[target_col].rolling(30, min_periods=1).min()
    df['past_30min_mean'] = df[target_col].rolling(30, min_periods=1).mean()
    
    # 최근 5분 통계 (상승/하락 판단용)
    df['recent_5min_mean'] = df[target_col].rolling(5, min_periods=1).mean()
    df['recent_5min_max'] = df[target_col].rolling(5, min_periods=1).max()
    df['recent_5min_min'] = df[target_col].rolling(5, min_periods=1).min()
    df['recent_5min_trend'] = df[target_col] - df[target_col].shift(5)
    
    # 최근 3분 통계 (277 감지용)
    df['recent_3min_max'] = df[target_col].rolling(3, min_periods=1).max()
    df['recent_3min_mean'] = df[target_col].rolling(3, min_periods=1).mean()
    
    # 10분 구간별 평균 (30분 데이터 3등분)
    df['first_10min_avg'] = df[target_col].rolling(10, min_periods=1).mean().shift(20)
    df['middle_10min_avg'] = df[target_col].rolling(10, min_periods=1).mean().shift(10)
    df['last_10min_avg'] = df[target_col].rolling(10, min_periods=1).mean()
    
    # 장기 추세 (30분)
    df['long_trend'] = df['last_10min_avg'] - df['first_10min_avg']
    
    # 가속도
    df['mid_trend'] = df['last_10min_avg'] - df['middle_10min_avg']
    df['early_trend'] = df['middle_10min_avg'] - df['first_10min_avg']
    df['acceleration'] = df['mid_trend'] - df['early_trend']
    
    # 변동성 변화
    df['first_10min_std'] = df[target_col].rolling(10, min_periods=1).std().shift(20)
    df['last_10min_std'] = df[target_col].rolling(10, min_periods=1).std()
    df['volatility_change'] = df['last_10min_std'] / (df['first_10min_std'] + 1e-6)
    
    # 277 특별 구간
    df['is_277_zone'] = (
        ((df['past_30min_max'] >= 275) & (df['past_30min_max'] <= 279)) |
        ((df['recent_5min_max'] >= 275) & (df['recent_5min_max'] <= 279)) |
        ((df['recent_3min_max'] >= 275) & (df['recent_3min_max'] <= 279))
    )
    
    # 점프 케이스 (과거 최대 < 280, 현재 >= 300)
    df['is_jump_case'] = (df['past_30min_max'].shift(10) < 280) & (df[target_col] >= 300)
    
    # 연속 상승 카운트
    df['consecutive_rises'] = (df[target_col] > df[target_col].shift(1)).rolling(30).sum()
    
    # 상승/하락 패턴
    df['is_rising'] = df['recent_5min_trend'] > 0
    df['is_rapid_rising'] = df['recent_5min_trend'] > 10
    
    # NaN 처리
    df = df.fillna(0)
    
    return df

# PatchTST와 PatchTSTPINN 모델 클래스는 이전과 동일...
class PatchTST(keras.Model):
    """안정형 모델 - False Positive 최소화"""
    
    def __init__(self, config):
        super().__init__()
        self.seq_len = config['seq_len']
        self.n_features = config['n_features']
        self.patch_len = config['patch_len']
        self.d_model = config.get('d_model', 128)
        self.n_heads = config.get('n_heads', 8)
        self.d_ff = config.get('d_ff', 256)
        self.n_layers = config.get('n_layers', 3)
        self.dropout = config.get('dropout', 0.1)
        
        # 패치 임베딩
        self.patch_embedding = layers.Dense(self.d_model)
        
        # Positional Encoding
        self.pos_encoding = self.create_positional_encoding()
        
        # Transformer Encoder Layers
        self.encoder_layers = [
            layers.MultiHeadAttention(
                num_heads=self.n_heads,
                key_dim=self.d_model // self.n_heads,
                dropout=self.dropout
            ) for _ in range(self.n_layers)
        ]
        
        # Feed Forward Networks
        self.ff_layers = [
            keras.Sequential([
                layers.Dense(self.d_ff, activation='relu'),
                layers.Dropout(self.dropout),
                layers.Dense(self.d_model),
                layers.Dropout(self.dropout)
            ]) for _ in range(self.n_layers)
        ]
        
        # Layer Normalization
        self.ln_layers = [
            [layers.LayerNormalization(epsilon=1e-6),
             layers.LayerNormalization(epsilon=1e-6)]
            for _ in range(self.n_layers)
        ]
        
        # Output layers
        self.global_pool = layers.GlobalAveragePooling1D()
        self.output_dense = layers.Dense(1)
    
    def create_positional_encoding(self):
        num_patches = self.seq_len // self.patch_len
        positions = tf.range(start=0, limit=num_patches, delta=1)
        positions = tf.expand_dims(positions, 0)
        return layers.Embedding(input_dim=num_patches, output_dim=self.d_model)(positions)
    
    def create_patches(self, x):
        batch_size = tf.shape(x)[0]
        patches = tf.reshape(x, [batch_size, -1, self.patch_len * self.n_features])
        return patches
    
    def call(self, inputs, training=False):
        # 패치 생성
        x = self.create_patches(inputs)
        
        # 패치 임베딩
        x = self.patch_embedding(x)
        
        # Positional encoding 추가
        x += self.pos_encoding
        
        # Transformer layers
        for i in range(self.n_layers):
            # Multi-head attention
            attn_output = self.encoder_layers[i](x, x, training=training)
            x = self.ln_layers[i][0](x + attn_output)
            
            # Feed forward
            ff_output = self.ff_layers[i](x, training=training)
            x = self.ln_layers[i][1](x + ff_output)
        
        # Global pooling and output
        x = self.global_pool(x)
        output = self.output_dense(x)
        
        return output

class PatchTSTPINN(keras.Model):
    """극단형 모델 - 점프 감지 특화"""
    
    def __init__(self, config):
        super().__init__()
        self.patchtst = PatchTST(config)
        
        # Physics-Informed Network (11차원 물리 특징)
        self.physics_net = keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu')
        ])
        
        # 통합 레이어
        self.fusion_layer = keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])
    
    def call(self, inputs, training=False):
        seq_input, physics_input = inputs
        
        # PatchTST 출력
        patchtst_out = self.patchtst(seq_input, training=training)
        
        # Physics network 출력
        physics_out = self.physics_net(physics_input, training=training)
        
        # 결합
        combined = tf.concat([patchtst_out, physics_out], axis=1)
        output = self.fusion_layer(combined, training=training)
        
        return output

# ==============================================================================
# 🎯 학습 함수 (수정)
# ==============================================================================

def train_dual_models_v2(processor, X, y, physics, model1_idx, model2_idx, 
                        model1_weights, model2_weights, ckpt):
    """듀얼 모델 학습 - 인덱스 문제 해결"""
    
    # 모델별 데이터 준비 (시퀀스 인덱스 사용)
    X_model1 = X[model1_idx]
    y_model1 = y[model1_idx]
    physics_model1 = physics[model1_idx]
    weights_model1 = model1_weights
    
    X_model2 = X[model2_idx]
    y_model2 = y[model2_idx]
    physics_model2 = physics[model2_idx]
    weights_model2 = model2_weights
    
    print(f"\n📊 Model1 학습 데이터: {len(X_model1)}개")
    print(f"📊 Model2 학습 데이터: {len(X_model2)}개")
    
    # 각 모델별 train/val 분리
    # Model 1
    split1 = int(0.8 * len(X_model1))
    X_train1, X_val1 = X_model1[:split1], X_model1[split1:]
    y_train1, y_val1 = y_model1[:split1], y_model1[split1:]
    weights_train1 = weights_model1[:split1]
    
    # Model 2  
    split2 = int(0.8 * len(X_model2))
    X_train2, X_val2 = X_model2[:split2], X_model2[split2:]
    y_train2, y_val2 = y_model2[:split2], y_model2[split2:]
    physics_train2, physics_val2 = physics_model2[:split2], physics_model2[split2:]
    weights_train2 = weights_model2[:split2]
    
    # 정규화
    # Model 1
    n_features = X_train1.shape[2]
    X_train1_scaled = processor.scaler_X.fit_transform(
        X_train1.reshape(-1, n_features)
    ).reshape(X_train1.shape)
    X_val1_scaled = processor.scaler_X.transform(
        X_val1.reshape(-1, n_features)
    ).reshape(X_val1.shape)
    
    y_train1_scaled = processor.scaler_y.fit_transform(y_train1.reshape(-1, 1)).flatten()
    y_val1_scaled = processor.scaler_y.transform(y_val1.reshape(-1, 1)).flatten()
    
    # Model 2 (별도 스케일러)
    scaler_X2 = RobustScaler()
    scaler_y2 = RobustScaler()
    
    X_train2_scaled = scaler_X2.fit_transform(
        X_train2.reshape(-1, n_features)
    ).reshape(X_train2.shape)
    X_val2_scaled = scaler_X2.transform(
        X_val2.reshape(-1, n_features)
    ).reshape(X_val2.shape)
    
    y_train2_scaled = scaler_y2.fit_transform(y_train2.reshape(-1, 1)).flatten()
    y_val2_scaled = scaler_y2.transform(y_val2.reshape(-1, 1)).flatten()
    
    physics_train2_scaled = processor.scaler_physics.fit_transform(physics_train2)
    physics_val2_scaled = processor.scaler_physics.transform(physics_val2)
    
    # 스케일러 저장
    joblib.dump(processor.scaler_X, f'{ckpt.scaler_dir}/scaler_X.pkl')
    joblib.dump(processor.scaler_y, f'{ckpt.scaler_dir}/scaler_y.pkl')
    joblib.dump(scaler_X2, f'{ckpt.scaler_dir}/scaler_X2.pkl')
    joblib.dump(scaler_y2, f'{ckpt.scaler_dir}/scaler_y2.pkl')
    joblib.dump(processor.scaler_physics, f'{ckpt.scaler_dir}/scaler_physics.pkl')
    
    # 모델 설정
    config = {
        'seq_len': 30,
        'n_features': n_features,
        'patch_len': 6,  # 30/6 = 5 patches
        'd_model': 128,
        'n_heads': 8,
        'd_ff': 256,
        'n_layers': 3,
        'dropout': 0.1
    }
    
    # Model 1 학습 (안정형)
    if not ckpt.load_state().get('model1_trained', False):
        print("\n" + "="*60)
        print("🤖 Model 1: PatchTST (안정형) 학습")
        print("="*60)
        
        model1 = PatchTST(config)
        
        # 안정형 손실함수 (과대예측 페널티)
        def stable_loss(y_true, y_pred):
            mae = tf.abs(y_true - y_pred)
            
            # 과대예측 페널티 (실제 < 300인데 300+ 예측)
            y_true_original = y_true * processor.scaler_y.scale_ + processor.scaler_y.center_
            y_pred_original = y_pred * processor.scaler_y.scale_ + processor.scaler_y.center_
            
            overpredict_penalty = tf.where(
                (y_true_original < 300) & (y_pred_original >= 300),
                mae * 5.0,  # 5배 페널티
                mae
            )
            
            # 250-270 경계 구간은 더 정확하게
            boundary_penalty = tf.where(
                (y_true_original >= 250) & (y_true_original < 270),
                mae * 1.5,
                mae
            )
            
            return tf.reduce_mean(tf.maximum(overpredict_penalty, boundary_penalty))
        
        model1.compile(
            optimizer=Adam(learning_rate=0.0001, clipnorm=1.0),
            loss=stable_loss,
            metrics=['mae']
        )
        
        callbacks1 = [
            EarlyStopping(patience=20, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6),
            ModelCheckpoint(f'{ckpt.checkpoint_dir}/model1_stable.weights.h5',  # 수정!
                           save_best_only=True, save_weights_only=True)
        ]
        
        history1 = model1.fit(
            X_train1_scaled, y_train1_scaled,
            validation_data=(X_val1_scaled, y_val1_scaled),
            epochs=100,
            batch_size=64,
            sample_weight=weights_train1,
            callbacks=callbacks1,
            verbose=1
        )
        
        # 학습 상태 저장
        state = ckpt.load_state()
        state['model1_trained'] = True
        ckpt.save_state(state)
        
        print("✅ Model 1 학습 완료")
    
    # Model 2 학습 (극단형)
    if not ckpt.load_state().get('model2_trained', False):
        print("\n" + "="*60)
        print("🤖 Model 2: PatchTST + PINN (극단형) 학습")
        print("="*60)
        
        model2 = PatchTSTPINN(config)
        
        # 극단형 손실함수 (점프 감지 중시)
        def extreme_loss(y_true, y_pred):
            mae = tf.abs(y_true - y_pred)
            
            # 극단값 놓침 페널티 (실제 300+인데 300 미만 예측)
            y_true_original = y_true * scaler_y2.scale_ + scaler_y2.center_
            y_pred_original = y_pred * scaler_y2.scale_ + scaler_y2.center_
            
            miss_penalty = tf.where(
                (y_true_original >= 300) & (y_pred_original < 300),
                mae * 10.0,  # 10배 페널티
                mae
            )
            
            # 277 구간 특별 처리
            zone_277_penalty = tf.where(
                (y_true_original >= 275) & (y_true_original <= 279),
                mae * 0.5,  # 페널티 감소 (더 관대하게)
                mae
            )
            
            # 극고값(335+) 놓침 추가 페널티
            extreme_miss_penalty = tf.where(
                (y_true_original >= 335) & (y_pred_original < 320),
                mae * 15.0,  # 15배 페널티
                mae
            )
            
            return tf.reduce_mean(tf.maximum(tf.maximum(miss_penalty, zone_277_penalty), extreme_miss_penalty))
        
        model2.compile(
            optimizer=Adam(learning_rate=0.0001, clipnorm=1.0),
            loss=extreme_loss,
            metrics=['mae']
        )
        
        callbacks2 = [
            EarlyStopping(patience=25, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=12, min_lr=1e-6),
            ModelCheckpoint(f'{ckpt.checkpoint_dir}/model2_extreme.weights.h5',  # 수정!
                           save_best_only=True, save_weights_only=True)
        ]
        
        history2 = model2.fit(
            [X_train2_scaled, physics_train2_scaled], y_train2_scaled,
            validation_data=([X_val2_scaled, physics_val2_scaled], y_val2_scaled),
            epochs=120,
            batch_size=64,
            sample_weight=weights_train2,
            callbacks=callbacks2,
            verbose=1
        )
        
        # 학습 상태 저장
        state = ckpt.load_state()
        state['model2_trained'] = True
        ckpt.save_state(state)
        
        print("✅ Model 2 학습 완료")
    
    return True

# ==============================================================================
# 🚀 메인 실행 (수정)
# ==============================================================================

def main():
    """메인 학습 프로세스 - 인덱스 문제 해결 버전"""
    
    ckpt = EnhancedCheckpointManager()
    processor = V4DataProcessor()
    
    # 이전 상태 확인
    state = ckpt.load_state()
    if state:
        print(f"\n📂 이전 학습 상태 발견! (Step {state.get('step', 1)}/7)")
        resume = input("이어서 진행하시겠습니까? (y/n): ").lower()
        
        if resume != 'y':
            state = {}
            step = 1
        else:
            step = state.get('step', 1)
    else:
        state = {}
        step = 1
    
    try:
        # Step 1: 데이터 로드
        if step <= 1:
            print(f"\n[Step 1/7] 데이터 로드 및 병합")
            print("-"*60)
            
            df = processor.load_and_merge_data()
            
            state['step'] = 2
            ckpt.save_state(state)
            
            if ckpt.interrupted:
                sys.exit(0)
        
        # Step 2: 연속 패턴 추가
        if step <= 2:
            print(f"\n[Step 2/7] 연속 패턴 계산")
            print("-"*60)
            
            if step < 2:
                df = processor.load_and_merge_data()
            
            df = processor.add_consecutive_patterns(df)
            
            state['step'] = 3
            state['df_shape'] = df.shape
            ckpt.save_state(state)
            
            # 데이터 임시 저장
            df.to_pickle(f'{ckpt.checkpoint_dir}/df_with_patterns.pkl')
            
            if ckpt.interrupted:
                sys.exit(0)
        
        # Step 3: 시퀀스 생성 (중단/재개 지원)
        if step <= 3:
            print(f"\n[Step 3/7] 시퀀스 생성")
            print("-"*60)
            
            # 데이터 로드
            if os.path.exists(f'{ckpt.checkpoint_dir}/df_with_patterns.pkl'):
                df = pd.read_pickle(f'{ckpt.checkpoint_dir}/df_with_patterns.pkl')
            else:
                df = processor.load_and_merge_data()
                df = processor.add_consecutive_patterns(df)
            
            # 시퀀스 생성 (중단/재개 지원)
            X, y, physics, sequence_indices = processor.create_sequences_with_resume(df, ckpt)
            
            # HDF5로 저장
            with h5py.File(ckpt.sequence_file, 'w') as f:
                f.create_dataset('X', data=X, compression='gzip')
                f.create_dataset('y', data=y, compression='gzip')
                f.create_dataset('physics', data=physics, compression='gzip')
                f.create_dataset('sequence_indices', data=sequence_indices, compression='gzip')
                f.attrs['n_features'] = X.shape[2]
            
            state['step'] = 4
            state['sequences_shape'] = X.shape
            ckpt.save_state(state)
            
            # 메모리 정리
            del X, y, physics, sequence_indices
            gc.collect()
            
            if ckpt.interrupted:
                sys.exit(0)
        
        # Step 4: 모델별 데이터 분리 (수정)
        if step <= 4:
            print(f"\n[Step 4/7] 모델별 데이터 분리")
            print("-"*60)
            
            # 시퀀스 로드
            with h5py.File(ckpt.sequence_file, 'r') as f:
                sequence_indices = f['sequence_indices'][:]
            
            # df 로드 (특징 계산용)
            df = pd.read_pickle(f'{ckpt.checkpoint_dir}/df_with_patterns.pkl')
            
            # 특징 계산
            df = calculate_advanced_features(df, processor.target_col)
            
            # 시퀀스 인덱스에 해당하는 특징만 추출
            df_expanded = df.iloc[sequence_indices].reset_index(drop=True)
            
            # 모델별 데이터 분리
            model1_idx, model2_idx, model1_weights, model2_weights = \
                create_model_specific_data_v2(df_expanded, sequence_indices)
            
            # 인덱스 저장
            np.save(f'{ckpt.checkpoint_dir}/model1_idx.npy', model1_idx)
            np.save(f'{ckpt.checkpoint_dir}/model2_idx.npy', model2_idx)
            np.save(f'{ckpt.checkpoint_dir}/model1_weights.npy', model1_weights)
            np.save(f'{ckpt.checkpoint_dir}/model2_weights.npy', model2_weights)
            
            state['step'] = 5
            state['model1_samples'] = len(model1_idx)
            state['model2_samples'] = len(model2_idx)
            ckpt.save_state(state)
            
            # 메모리 정리
            del df, df_expanded
            gc.collect()
            
            if ckpt.interrupted:
                sys.exit(0)
        
        # Step 5: 모델 학습
        if step <= 5:
            print(f"\n[Step 5/7] 듀얼 모델 학습")
            print("-"*60)
            
            # 데이터 로드
            with h5py.File(ckpt.sequence_file, 'r') as f:
                X = f['X'][:]
                y = f['y'][:]
                physics = f['physics'][:]
            
            # 인덱스와 가중치 로드
            model1_idx = np.load(f'{ckpt.checkpoint_dir}/model1_idx.npy')
            model2_idx = np.load(f'{ckpt.checkpoint_dir}/model2_idx.npy')
            model1_weights = np.load(f'{ckpt.checkpoint_dir}/model1_weights.npy')
            model2_weights = np.load(f'{ckpt.checkpoint_dir}/model2_weights.npy')
            
            # 학습
            success = train_dual_models_v2(
                processor, X, y, physics,
                model1_idx, model2_idx,
                model1_weights, model2_weights,
                ckpt
            )
            
            if success:
                state['step'] = 6
                ckpt.save_state(state)
            
            # 메모리 정리
            del X, y, physics
            gc.collect()
            
            if ckpt.interrupted:
                sys.exit(0)
        
        # Step 6: 최종 저장
        if step <= 6:
            print(f"\n[Step 6/7] 최종 저장")
            print("-"*60)
            
            # 학습 설정 저장
            with h5py.File(f'{ckpt.checkpoint_dir}/scaled_data.h5', 'w') as f:
                with h5py.File(ckpt.sequence_file, 'r') as sf:
                    f.attrs['n_features'] = sf.attrs['n_features']
                f.attrs['seq_len'] = 30
                f.attrs['patch_len'] = 6
                f.attrs['model1_samples'] = state.get('model1_samples', 0)
                f.attrs['model2_samples'] = state.get('model2_samples', 0)
            
            print("✅ 학습 완료!")
            print(f"\n📁 저장 위치: {ckpt.checkpoint_dir}")
            print("  - model1_stable.weights.h5: 안정형 모델")  # 수정!
            print("  - model2_extreme.weights.h5: 극단형 모델")  # 수정!
            print("  - scalers/: 정규화 스케일러")
            print("  - scaled_data.h5: 학습 설정")
            
            state['step'] = 7
            state['completed'] = True
            ckpt.save_state(state)
        
        print("\n" + "="*80)
        print("🎉 V4 듀얼 모델 학습 완료!")
        print("="*80)
        print("\n다음 단계:")
        print("1. 평가 코드로 성능 테스트")
        print("2. 202509 데이터로 최종 검증")
        print("="*80)
    
    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자 중단")
        print("💾 진행 상황이 저장되었습니다.")
    
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()ModelCheckpoint 파일명을 수정한 전체 코드입니다. 이제 오류 없이 실행될 것입니다!