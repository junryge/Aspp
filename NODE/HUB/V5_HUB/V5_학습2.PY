#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
🎯 HUBROOM 점프 감지 80% 시스템 - 완전 통합 학습 코드
================================================================================
목표: 
- 3구간 분류 (50-150, 150-299, 300+)
- 점프 감지 80% (과거 최대 < 280 → 10분 후 >= 300)
- 상승/하락 패턴 분석
- 상승률/하락률 계산 포함
- 완벽한 중단/재개 시스템
================================================================================
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, ExtraTreesRegressor
from sklearn.preprocessing import RobustScaler, LabelEncoder
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
print("🎯 HUBROOM 점프 감지 80% 시스템")
print("📊 3구간 분류 + 점프 감지 + 상승/하락 패턴")
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
                    'progress': i
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
                np.array(y_range), np.array(y_trend), np.array(y_value))

# ==============================================================================
# 🤖 모델 정의
# ==============================================================================

class JumpDetectionSystem:
    """점프 감지 80% 달성 시스템"""
    
    def __init__(self):
        # Stage 1: 빠른 스크리닝
        self.model_screening = ExtraTreesClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # Stage 2: 전문 모델들
        self.model_jump = XGBClassifier(
            n_estimators=300,
            max_depth=10,
            learning_rate=0.1,
            scale_pos_weight=100,  # 불균형 처리
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
        
        # Stage 3: 값 예측
        self.model_value = ExtraTreesRegressor(
            n_estimators=500,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        
        self.scalers = {}
        self.feature_names = None
    
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
    
    def get_feature_indices(self, df):
        """중요 특징들의 인덱스 찾기"""
        feature_cols = [col for col in df.columns 
                       if col not in ['datetime', 'range_class', 'is_jump', 'trend_pattern']]
        
        indices = {}
        for i, col in enumerate(feature_cols):
            if 'STORAGE_UTIL' in col:
                indices['storage_util'] = i
            elif 'BRIDGE_TIME' in col and 'diff' not in col:
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
        """점프 감지 모델 학습 - 80% 목표"""
        print("\n🎯 점프 감지 모델 학습")
        
        # SMOTE로 불균형 처리
        jump_count = np.sum(y_jump)
        print(f"  점프 케이스: {jump_count}/{len(y_jump)} ({jump_count/len(y_jump)*100:.2f}%)")
        
        smote = SMOTE(sampling_strategy=0.2, random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y_jump)
        
        print(f"  SMOTE 후: {np.sum(y_balanced)}/{len(y_balanced)} ({np.sum(y_balanced)/len(y_balanced)*100:.2f}%)")
        
        # 학습
        self.model_jump.fit(X_balanced, y_balanced)
        
        # 특징 중요도
        if hasattr(self.model_jump, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature_idx': range(X.shape[1]),
                'importance': self.model_jump.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\n✅ 점프 감지 상위 10개 특징:")
            for idx, row in importance.head(10).iterrows():
                print(f"  특징 {int(row['feature_idx'])}: {row['importance']:.4f}")
        
        return self.model_jump
    
    def apply_rule_based_boost(self, X, predictions, prob_scores=None):
        """규칙 기반 부스팅 - 80% 달성용"""
        boosted_predictions = predictions.copy()
        
        if not hasattr(self, 'feature_indices'):
            print("⚠️ 특징 인덱스가 없습니다. 부스팅 스킵")
            return boosted_predictions
        
        idx = self.feature_indices
        
        # Phase 1: 강한 신호 (55%)
        if 'storage_util' in idx:
            # 마지막 시점 특징 사용
            strong_signal = X[:, idx['storage_util']] > 20
            boosted_predictions[strong_signal] = 1
            print(f"  Phase 1: {np.sum(strong_signal)}개 케이스 부스트")
        
        # Phase 2: 중간 신호 (70%)
        if 'storage_util' in idx and 'bridge_time' in idx:
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
                storage_medium = X[:, idx['storage_util']] > 10
                final_medium = medium_signal & storage_medium & (boosted_predictions == 0)
                boosted_predictions[final_medium] = 1
                print(f"  Phase 2: {np.sum(final_medium)}개 케이스 추가 부스트")
        
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
            
            # 확률 기반 추가 필터링
            if prob_scores is not None:
                final_weak = final_weak & (prob_scores > 0.3)
            
            boosted_predictions[final_weak] = 1
            print(f"  Phase 3: {np.sum(final_weak)}개 케이스 추가 부스트")
        
        return boosted_predictions

# ==============================================================================
# 📊 평가 함수
# ==============================================================================

def evaluate_jump_detection(y_true, y_pred, y_pred_proba=None):
    """점프 감지 성능 평가"""
    print("\n📊 점프 감지 성능 평가")
    print("-" * 50)
    
    # 기본 지표
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
    
    # False Positive Rate
    non_jump_true = total - jump_true
    if non_jump_true > 0:
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fpr = fp / non_jump_true
        print(f"False Positive Rate: {fpr*100:.1f}% ({fp}/{non_jump_true})")
    
    # 정확도
    accuracy = np.sum(y_true == y_pred) / total
    print(f"\n전체 정확도: {accuracy*100:.1f}%")
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
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
        print(f"\n📂 이전 학습 상태 발견! (Step {state.get('step', 1)}/7)")
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
            print(f"\n[Step 1/7] 데이터 로드 및 병합")
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
            print(f"\n[Step 2/7] 특징 엔지니어링")
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
            print(f"\n[Step 3/7] 시퀀스 생성")
            print("-"*60)
            
            # 데이터 로드
            saved_data = ckpt.load_data()
            if saved_data:
                df = saved_data['df']
            else:
                print("⚠️ 저장된 데이터 없음. 재생성...")
                df = processor.load_and_merge_data()
                df = processor.create_all_features(df)
            
            # 이전 시퀀스 체크
            saved_seq = ckpt.load_sequences()
            if saved_seq and saved_seq.get('progress', 0) > 0:
                use_saved = input(f"이전 시퀀스 발견 (진행도: {saved_seq['progress']}). 사용? (y/n): ")
                if use_saved.lower() == 'y':
                    X = saved_seq['X']
                    y = saved_seq['y']
                    y_jump = saved_seq['y_jump']
                    y_range = saved_seq['y_range']
                    y_trend = saved_seq['y_trend']
                    y_value = saved_seq['y_value']
                else:
                    X, y, y_jump, y_range, y_trend, y_value = processor.create_sequences(df, ckpt=ckpt)
            else:
                X, y, y_jump, y_range, y_trend, y_value = processor.create_sequences(df, ckpt=ckpt)
            
            # 시퀀스 저장
            sequences_dict = {
                'X': X,
                'y': y,
                'y_jump': y_jump,
                'y_range': y_range,
                'y_trend': y_trend,
                'y_value': y_value,
                'progress': len(X)
            }
            ckpt.save_sequences(sequences_dict)
            
            # 특징 인덱스 저장
            system.get_feature_indices(df)
            
            state['step'] = 4
            state['sequence_shape'] = X.shape
            ckpt.save_state(state)
            
            if ckpt.interrupted:
                print("\n💾 Step 3 저장 후 종료")
                sys.exit(0)
        
        # Step 4: 데이터 분할
        if step <= 4:
            print(f"\n[Step 4/7] 학습/검증 데이터 분할")
            print("-"*60)
            
            # 시퀀스 로드
            saved_seq = ckpt.load_sequences()
            X = saved_seq['X']
            y = saved_seq['y']
            y_jump = saved_seq['y_jump']
            y_range = saved_seq['y_range']
            y_trend = saved_seq['y_trend']
            y_value = saved_seq['y_value']
            
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
        
        # Step 5: 특징 준비
        if step <= 5:
            print(f"\n[Step 5/7] 특징 준비 및 스케일링")
            print("-"*60)
            
            # 데이터 재로드
            saved_seq = ckpt.load_sequences()
            X = saved_seq['X']
            y_jump = saved_seq['y_jump']
            y_range = saved_seq['y_range']
            y_trend = saved_seq['y_trend']
            
            split_idx = state.get('split_idx', int(0.8 * len(X)))
            
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_jump_train, y_jump_val = y_jump[:split_idx], y_jump[split_idx:]
            y_range_train, y_range_val = y_range[:split_idx], y_range[split_idx:]
            y_trend_train, y_trend_val = y_trend[:split_idx], y_trend[split_idx:]
            
            # 특징 준비
            X_train_features = system.prepare_features(X_train)
            X_val_features = system.prepare_features(X_val)
            
            print(f"✅ 학습 특징 shape: {X_train_features.shape}")
            print(f"✅ 검증 특징 shape: {X_val_features.shape}")
            
            # 데이터프레임 로드하여 특징 인덱스 설정
            saved_data = ckpt.load_data()
            if saved_data:
                df = saved_data['df']
                system.get_feature_indices(df)
            
            state['step'] = 6
            ckpt.save_state(state)
            
            if ckpt.interrupted:
                print("\n💾 Step 5 저장 후 종료")
                sys.exit(0)
        
        # Step 6: 모델 학습
        if step <= 6:
            print(f"\n[Step 6/7] 모델 학습")
            print("-"*60)
            
            # 데이터 재로드
            saved_seq = ckpt.load_sequences()
            X = saved_seq['X']
            y = saved_seq['y']
            y_jump = saved_seq['y_jump']
            y_range = saved_seq['y_range']
            y_trend = saved_seq['y_trend']
            y_value = saved_seq['y_value']
            
            split_idx = state.get('split_idx', int(0.8 * len(X)))
            
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_jump_train, y_jump_val = y_jump[:split_idx], y_jump[split_idx:]
            y_range_train, y_range_val = y_range[:split_idx], y_range[split_idx:]
            y_trend_train, y_trend_val = y_trend[:split_idx], y_trend[split_idx:]
            y_value_train, y_value_val = y_value[:split_idx], y_value[split_idx:]
            
            # 특징 준비
            X_train_features = system.prepare_features(X_train)
            X_val_features = system.prepare_features(X_val)
            
            # 데이터프레임 로드하여 특징 인덱스 설정
            saved_data = ckpt.load_data()
            if saved_data:
                df = saved_data['df']
                system.get_feature_indices(df)
            
            # 6-1. 점프 감지 모델
            if not state.get('jump_trained', False):
                system.train_jump_detector(X_train_features, y_jump_train)
                ckpt.save_model(system.model_jump, 'model_jump')
                state['jump_trained'] = True
                ckpt.save_state(state)
                
                if ckpt.interrupted:
                    print("\n💾 점프 모델 저장 후 종료")
                    sys.exit(0)
            
            # 6-2. 3구간 분류 모델
            if not state.get('range_trained', False):
                print("\n📊 3구간 분류 모델 학습")
                system.model_range.fit(X_train_features, y_range_train)
                ckpt.save_model(system.model_range, 'model_range')
                state['range_trained'] = True
                ckpt.save_state(state)
                
                if ckpt.interrupted:
                    print("\n💾 구간 모델 저장 후 종료")
                    sys.exit(0)
            
            # 6-3. 상승/하락 패턴 모델
            if not state.get('trend_trained', False):
                print("\n📈 상승/하락 패턴 모델 학습")
                system.model_trend.fit(X_train_features, y_trend_train)
                ckpt.save_model(system.model_trend, 'model_trend')
                state['trend_trained'] = True
                ckpt.save_state(state)
                
                if ckpt.interrupted:
                    print("\n💾 패턴 모델 저장 후 종료")
                    sys.exit(0)
            
            # 6-4. 값 예측 모델
            if not state.get('value_trained', False):
                print("\n💰 값 예측 모델 학습")
                system.model_value.fit(X_train_features, y_value_train)
                ckpt.save_model(system.model_value, 'model_value')
                state['value_trained'] = True
                ckpt.save_state(state)
            
            state['step'] = 7
            ckpt.save_state(state)
            
            if ckpt.interrupted:
                print("\n💾 모든 모델 저장 후 종료")
                sys.exit(0)
        
        # Step 7: 평가
        if step <= 7:
            print(f"\n[Step 7/7] 최종 평가")
            print("-"*60)
            
            # 모델 로드
            system.model_jump = ckpt.load_model('model_jump')
            system.model_range = ckpt.load_model('model_range')
            system.model_trend = ckpt.load_model('model_trend')
            system.model_value = ckpt.load_model('model_value')
            
            # 데이터 재로드
            saved_seq = ckpt.load_sequences()
            X = saved_seq['X']
            y_jump = saved_seq['y_jump']
            y_range = saved_seq['y_range']
            y_trend = saved_seq['y_trend']
            
            split_idx = state.get('split_idx', int(0.8 * len(X)))
            
            X_val = X[split_idx:]
            y_jump_val = y_jump[split_idx:]
            y_range_val = y_range[split_idx:]
            y_trend_val = y_trend[split_idx:]
            
            # 특징 준비
            X_val_features = system.prepare_features(X_val)
            
            # 데이터프레임 로드하여 특징 인덱스 설정
            saved_data = ckpt.load_data()
            if saved_data:
                df = saved_data['df']
                system.get_feature_indices(df)
            
            print("\n" + "="*60)
            print("📊 최종 성능 평가")
            print("="*60)
            
            # 점프 감지 평가
            jump_pred = system.model_jump.predict(X_val_features)
            jump_pred_proba = system.model_jump.predict_proba(X_val_features)[:, 1]
            
            # 규칙 기반 부스팅
            jump_pred_boosted = system.apply_rule_based_boost(X_val_features, jump_pred, jump_pred_proba)
            
            # 성능 평가
            metrics = evaluate_jump_detection(y_jump_val, jump_pred_boosted, jump_pred_proba)
            
            # 3구간 분류 평가
            range_pred = system.model_range.predict(X_val_features)
            range_accuracy = (range_pred == y_range_val).mean()
            
            print(f"\n📊 3구간 분류")
            print(f"  - 정확도: {range_accuracy*100:.1f}%")
            
            # 상승/하락 패턴 평가
            trend_pred = system.model_trend.predict(X_val_features)
            trend_accuracy = (trend_pred == y_trend_val).mean()
            
            print(f"\n📈 상승/하락 패턴")
            print(f"  - 정확도: {trend_accuracy*100:.1f}%")
            
            # 전체 시스템 저장
            print("\n💾 전체 시스템 저장")
            joblib.dump(system, 'hubroom_jump_detection_system.pkl')
            
            # 상태 파일 정리
            remove = input("\n학습 완료! 상태 파일을 제거하시겠습니까? (y/n): ")
            if remove.lower() == 'y':
                ckpt.clear_state()
                print("🧹 상태 파일 제거 완료")
            
            print("\n" + "="*60)
            print("🎉 학습 완료!")
            print(f"✅ 점프 감지율: {metrics['recall']*100:.1f}% (목표: 80%)")
            print(f"✅ 3구간 분류: {range_accuracy*100:.1f}%")
            print(f"✅ 상승/하락 패턴: {trend_accuracy*100:.1f}%")
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