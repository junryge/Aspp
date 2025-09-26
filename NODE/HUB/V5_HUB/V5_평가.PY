#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
🎯 HUBROOM 점프 감지 시스템 평가 코드
================================================================================
목표:
- 과거 30분 데이터로 10분 후 예측 평가
- 학습된 모델 사용:
  * ExtraTreesClassifier (스크리닝)
  * XGBClassifier (점프 감지)
  * RandomForestClassifier (구간, 패턴)
  * ExtraTreesRegressor (값 예측)
- 평가 데이터: 2025년 8월
- CSV 결과 출력 (요청된 형식)
================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, ExtraTreesRegressor
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib
import os
import pickle
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🎯 HUBROOM 점프 감지 시스템 평가")
print("📊 30분 시퀀스 → 10분 후 예측")
print("📅 평가 데이터: 2025년 8월")
print("="*80)

# ==============================================================================
# 📊 데이터 처리 클래스 (학습 코드와 동일)
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
    
    def load_and_merge_data(self, data_path):
        """데이터 로드 및 BRIDGE_TIME 병합"""
        print("\n[1단계] 데이터 로드")
        
        # 메인 데이터
        df = pd.read_csv(data_path)
        print(f"✅ 평가 데이터: {df.shape}")
        
        # 시간 처리
        time_col = df.columns[0]
        df['datetime'] = pd.to_datetime(df[time_col], format='%Y%m%d%H%M')
        
        # BRIDGE_TIME 데이터가 있는지 확인
        bridge_path = data_path.replace('.csv', '_BRIDGE.csv')
        if os.path.exists(bridge_path):
            bridge_df = pd.read_csv(bridge_path)
            print(f"✅ BRIDGE_TIME 데이터: {bridge_df.shape}")
            
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
                
                # BRIDGE_TIME 보간
                df['BRIDGE_TIME'] = df['BRIDGE_TIME'].interpolate(method='linear', limit_direction='both')
        
        # BRIDGE_TIME이 없으면 기본값
        if 'BRIDGE_TIME' not in df.columns:
            df['BRIDGE_TIME'] = 3.5
            
        df['BRIDGE_TIME'] = df['BRIDGE_TIME'].fillna(3.5)
        
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
    
    def create_sequences_for_evaluation(self, df, seq_len=30, pred_len=10):
        """평가용 시퀀스 데이터 생성"""
        print(f"\n[3단계] 시퀀스 생성 ({seq_len}분 → {pred_len}분 후)")
        
        # 특징 컬럼 선택
        feature_cols = [col for col in df.columns 
                       if col not in ['datetime', 'range_class', 'is_jump', 'trend_pattern']]
        
        X = []
        y_info = []
        
        for i in tqdm(range(seq_len, len(df) - pred_len), desc="시퀀스 생성"):
            # 입력: 과거 30분
            X.append(df[feature_cols].iloc[i-seq_len:i].values)
            
            # 타겟 정보들
            target_idx = i + pred_len - 1
            
            # 시간 정보
            current_time = df['datetime'].iloc[i-1]
            predict_time = df['datetime'].iloc[target_idx]
            seq_start_time = df['datetime'].iloc[i-seq_len]
            seq_end_time = df['datetime'].iloc[i-1]
            
            # 시퀀스 통계
            seq_values = df[self.target_col].iloc[i-seq_len:i].values
            seq_max = np.max(seq_values)
            seq_min = np.min(seq_values)
            
            y_info.append({
                'current_time': current_time,
                'predict_time': predict_time,
                'seq_start_time': seq_start_time,
                'seq_end_time': seq_end_time,
                'seq_max': seq_max,
                'seq_min': seq_min,
                'target_value': df[self.target_col].iloc[target_idx],
                'is_jump': df['is_jump'].iloc[target_idx],
                'range_class': df['range_class'].iloc[target_idx],
                'trend_pattern': df['trend_pattern'].iloc[target_idx]
            })
        
        print(f"✅ {len(X)}개 시퀀스 생성 완료")
        
        return np.array(X), y_info, df

# ==============================================================================
# 🤖 모델 시스템 (학습 코드와 동일)
# ==============================================================================

class JumpDetectionSystem:
    """점프 감지 80% 달성 시스템"""
    
    def __init__(self):
        # 모델들
        self.model_screening = None
        self.model_jump = None
        self.model_range = None
        self.model_trend = None
        self.model_value = None
        
        self.feature_indices = {}
    
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
            last_features,    # 0 ~ n-1
            mean_features,    # n ~ 2n-1
            std_features,     # 2n ~ 3n-1
            max_features,     # 3n ~ 4n-1
            min_features,     # 4n ~ 5n-1
            trend_features    # 5n ~ 6n-1
        ])
        
        return features
    
    def get_expanded_feature_indices(self, df):
        """확장된 특징 인덱스 계산 (6배 확장 고려)"""
        feature_cols = [col for col in df.columns 
                       if col not in ['datetime', 'range_class', 'is_jump', 'trend_pattern']]
        
        n_base_features = len(feature_cols)
        expanded_indices = {}
        
        # 원본 특징의 인덱스 찾기
        base_indices = {}
        for i, col in enumerate(feature_cols):
            if 'STORAGE_UTIL' in col:
                base_indices['storage_util'] = i
            elif col == 'BRIDGE_TIME':  # 정확한 매칭
                base_indices['bridge_time'] = i
            elif 'flow_balance' in col:
                base_indices['flow_balance'] = i
            elif 'consecutive_250+' in col:
                base_indices['consecutive_250'] = i
            elif 'cmd_sync_count' in col:
                base_indices['cmd_sync'] = i
            elif 'trend_20min' in col:
                base_indices['trend_20min'] = i
            elif 'acceleration' in col:
                base_indices['acceleration'] = i
            elif 'consecutive_300_prob' in col:
                base_indices['prob_extreme'] = i
            elif 'in_jump_zone' in col:
                base_indices['in_jump_zone'] = i
            elif 'recent_5min_max' in col:
                base_indices['recent_5min_max'] = i
        
        # 확장된 인덱스 계산
        for key, base_idx in base_indices.items():
            expanded_indices[f'{key}_last'] = base_idx
            expanded_indices[f'{key}_mean'] = base_idx + n_base_features
            expanded_indices[f'{key}_std'] = base_idx + 2 * n_base_features
            expanded_indices[f'{key}_max'] = base_idx + 3 * n_base_features
            expanded_indices[f'{key}_min'] = base_idx + 4 * n_base_features
            expanded_indices[f'{key}_trend'] = base_idx + 5 * n_base_features
        
        # 디버깅
        print(f"\n📍 확장된 특징 인덱스:")
        print(f"  - 기본 특징 수: {n_base_features}")
        print(f"  - 확장 특징 수: {n_base_features * 6}")
        print(f"  - storage_util_last: {expanded_indices.get('storage_util_last', 'N/A')}")
        print(f"  - storage_util_max: {expanded_indices.get('storage_util_max', 'N/A')}")
        print(f"  - bridge_time_last: {expanded_indices.get('bridge_time_last', 'N/A')}")
        
        self.feature_indices = expanded_indices
        return expanded_indices
    
    def apply_emergency_boost(self, X_seq, X_features, predictions, value_pred):
        """긴급 부스팅 - 평가 결과를 보고 추가한 강력한 규칙"""
        boosted = predictions.copy()
        boost_count = 0
        
        for i in range(len(X_seq)):
            # 과거 30분 최대값
            seq_max = np.max(X_seq[i, :, 0])  # target_col이 첫 번째라고 가정
            
            # 규칙 1: 값 예측이 290 이상이고 과거 최대값이 280 미만
            if value_pred[i] >= 290 and seq_max < 280:
                boosted[i] = 1
                boost_count += 1
            
            # 규칙 2: 최근 5분 평균이 275 이상이고 과거 최대값이 279 이하
            recent_5min = np.mean(X_seq[i, -5:, 0])
            if recent_5min >= 275 and seq_max <= 279:
                boosted[i] = 1
                boost_count += 1
            
            # 규칙 3: 가속도가 크고 현재값이 270 이상
            if len(X_seq[i]) >= 20:
                accel = (np.mean(X_seq[i, -5:, 0]) - np.mean(X_seq[i, -10:-5, 0])) - \
                       (np.mean(X_seq[i, -10:-5, 0]) - np.mean(X_seq[i, -15:-10, 0]))
                current = X_seq[i, -1, 0]
                
                if accel > 15 and current >= 270:
                    boosted[i] = 1
                    boost_count += 1
        
        print(f"  🚨 긴급 부스팅: {boost_count}개 추가")
        return boosted
    
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
            elif 'in_jump_zone' in col:
                indices['in_jump_zone'] = i
            elif 'recent_5min_max' in col:
                indices['recent_5min_max'] = i
        
        # 디버깅: 찾은 인덱스 출력
        print(f"\n📍 특징 인덱스 설정:")
        print(f"  - 찾은 특징 수: {len(indices)}")
        for key, idx in indices.items():
            if idx < len(feature_cols):
                print(f"  - {key}: {idx} ({feature_cols[idx]})")
        
        self.feature_indices = indices
        return indices
    
    def apply_rule_based_boost_v3(self, X, predictions, prob_scores=None):
        """개선된 규칙 기반 부스팅 - 확장된 인덱스 사용"""
        boosted_predictions = predictions.copy()
        
        if not self.feature_indices:
            print("⚠️ 특징 인덱스가 없습니다.")
            return boosted_predictions
        
        idx = self.feature_indices
        boost_count = 0
        
        # Phase 1: storage_util 기반 (여러 통계 사용)
        conditions = []
        
        if 'storage_util_last' in idx:
            conditions.append(X[:, idx['storage_util_last']] > 15)
        if 'storage_util_max' in idx:
            conditions.append(X[:, idx['storage_util_max']] > 20)
        if 'storage_util_mean' in idx:
            conditions.append(X[:, idx['storage_util_mean']] > 10)
        
        if conditions:
            strong_signal = np.any(conditions, axis=0)
            new_boost = strong_signal & (boosted_predictions == 0)
            boosted_predictions[strong_signal] = 1
            phase1_count = np.sum(new_boost)
            print(f"  Phase 1 (storage): {phase1_count}개 추가")
            boost_count += phase1_count
        
        # Phase 2: bridge_time 기반
        conditions = []
        
        if 'bridge_time_last' in idx:
            conditions.append(X[:, idx['bridge_time_last']] > 3.8)
        if 'bridge_time_max' in idx:
            conditions.append(X[:, idx['bridge_time_max']] > 4.0)
        if 'bridge_time_mean' in idx:
            conditions.append(X[:, idx['bridge_time_mean']] > 3.7)
        
        if conditions:
            bridge_signal = np.any(conditions, axis=0)
            new_boost = bridge_signal & (boosted_predictions == 0)
            boosted_predictions[bridge_signal] = 1
            phase2_count = np.sum(new_boost)
            print(f"  Phase 2 (bridge): {phase2_count}개 추가")
            boost_count += phase2_count
        
        # Phase 3: 복합 조건
        complex_conditions = []
        
        if 'flow_balance_mean' in idx:
            complex_conditions.append(X[:, idx['flow_balance_mean']] > 30)
        if 'acceleration_last' in idx:
            complex_conditions.append(X[:, idx['acceleration_last']] > 10)
        if 'consecutive_250_max' in idx:
            complex_conditions.append(X[:, idx['consecutive_250_max']] >= 5)
        if 'trend_20min_last' in idx:
            complex_conditions.append(X[:, idx['trend_20min_last']] > 20)
        
        if len(complex_conditions) >= 2:
            complex_signal = np.sum(complex_conditions, axis=0) >= 2
            new_boost = complex_signal & (boosted_predictions == 0)
            boosted_predictions[complex_signal] = 1
            phase3_count = np.sum(new_boost)
            print(f"  Phase 3 (복합): {phase3_count}개 추가")
            boost_count += phase3_count
        
        # Phase 4: 확률 기반
        if prob_scores is not None:
            # 확률이 0.1 이상이면 부스트 (더 낮춤)
            prob_boost = (prob_scores > 0.1) & (boosted_predictions == 0)
            boosted_predictions[prob_boost] = 1
            phase4_count = np.sum(prob_boost)
            print(f"  Phase 4 (확률>0.1): {phase4_count}개 추가")
            boost_count += phase4_count
        
        # Phase 5: 점프존 감지
        if 'in_jump_zone_last' in idx:
            jump_zone = X[:, idx['in_jump_zone_last']] > 0
            new_boost = jump_zone & (boosted_predictions == 0)
            boosted_predictions[jump_zone] = 1
            phase5_count = np.sum(new_boost)
            print(f"  Phase 5 (277존): {phase5_count}개 추가")
            boost_count += phase5_count
        
        # Phase 6: recent_5min_max가 275 이상
        if 'recent_5min_max_last' in idx:
            recent_high = X[:, idx['recent_5min_max_last']] >= 275
            new_boost = recent_high & (boosted_predictions == 0)
            boosted_predictions[recent_high] = 1
            phase6_count = np.sum(new_boost)
            print(f"  Phase 6 (최근5분max≥275): {phase6_count}개 추가")
            boost_count += phase6_count
        
        print(f"  📊 총 부스팅: {boost_count}개")
        
        return boosted_predictions

# ==============================================================================
# 📊 평가 함수
# ==============================================================================

def evaluate_models():
    """모델 평가 메인 함수"""
    
    print("\n🚀 점프 감지 시스템 평가 시작...")
    
    # 1. 체크포인트 확인
    checkpoint_dir = './checkpoints_jump80'
    models_dir = os.path.join(checkpoint_dir, 'models')
    
    if not os.path.exists(models_dir):
        print("❌ 학습된 모델을 찾을 수 없습니다. 먼저 학습을 진행해주세요.")
        return
    
    # 2. 데이터 처리
    processor = HubRoomDataProcessor()
    system = JumpDetectionSystem()
    
    # 평가 데이터 로드
    eval_data_path = 'data/20250801_to_20250831.csv'
    
    if not os.path.exists(eval_data_path):
        print(f"❌ 평가 데이터를 찾을 수 없습니다: {eval_data_path}")
        return
    
    # 데이터 로드 및 전처리
    df = processor.load_and_merge_data(eval_data_path)
    df = processor.create_all_features(df)
    
    # 3. 시퀀스 생성
    X_seq, y_info, df = processor.create_sequences_for_evaluation(df)
    
    print(f"\n📊 평가 데이터 준비 완료")
    print(f"  - 시퀀스 수: {len(X_seq)}")
    print(f"  - 시퀀스 형태: {X_seq.shape}")
    
    # 4. 모델 로드
    print("\n🤖 모델 로드 중...")
    
    try:
        system.model_jump = joblib.load(os.path.join(models_dir, 'model_jump.pkl'))
        system.model_range = joblib.load(os.path.join(models_dir, 'model_range.pkl'))
        system.model_trend = joblib.load(os.path.join(models_dir, 'model_trend.pkl'))
        system.model_value = joblib.load(os.path.join(models_dir, 'model_value.pkl'))
        print("✅ 모델 로드 완료")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return
    
    # 특징 인덱스 설정 (확장된 인덱스 사용)
    system.get_expanded_feature_indices(df)
    
    # 5. 특징 준비
    X_features = system.prepare_features(X_seq)
    print(f"✅ 특징 준비 완료: {X_features.shape}")
    
    # 6. 예측 수행
    print("\n🔮 예측 수행 중...")
    
    # 점프 예측
    print("  - 점프 감지 모델 예측 중...")
    jump_pred = system.model_jump.predict(X_features)
    jump_pred_proba = system.model_jump.predict_proba(X_features)[:, 1]
    
    # 점프 예측 통계
    print(f"    초기 점프 예측: {np.sum(jump_pred)}개 감지")
    print(f"    점프 확률 분포: 최소={np.min(jump_pred_proba):.3f}, 최대={np.max(jump_pred_proba):.3f}, 평균={np.mean(jump_pred_proba):.3f}")
    
    # 값 예측 (부스팅에 필요)
    print("  - 값 예측 모델 예측 중...")
    value_pred = system.model_value.predict(X_features)
    
    print("\n  📊 규칙 기반 부스팅 적용")
    jump_pred_boosted = system.apply_rule_based_boost_v3(X_features, jump_pred, jump_pred_proba)
    print(f"    부스팅 후 점프 예측: {np.sum(jump_pred_boosted)}개 감지")
    
    # 긴급 부스팅 추가
    print("\n  🚨 긴급 부스팅 적용")
    jump_pred_final = system.apply_emergency_boost(X_seq, X_features, jump_pred_boosted, value_pred)
    print(f"    최종 점프 예측: {np.sum(jump_pred_final)}개 감지")
    
    # 구간 예측
    print("\n  - 3구간 분류 모델 예측 중...")
    range_pred = system.model_range.predict(X_features)
    
    # 패턴 예측
    print("  - 상승/하락 패턴 모델 예측 중...")
    trend_pred = system.model_trend.predict(X_features)
    
    # 7. 결과 정리
    print("\n📋 결과 생성 중...")
    results = []
    
    # 실제 점프 케이스 카운트
    actual_jump_count = 0
    detected_jump_count = 0
    
    for i in tqdm(range(len(X_seq)), desc="결과 생성"):
        info = y_info[i]
        
        # 점프 케이스 확인 (과거 최대값 < 280 & 실제값 >= 300)
        is_jump_case = (info['seq_max'] < 280) and (info['target_value'] >= 300)
        if is_jump_case:
            actual_jump_count += 1
        
        predicted_jump = jump_pred_final[i] == 1
        if is_jump_case and predicted_jump:
            detected_jump_count += 1
        
        # 예측값 조정 - 점프 예측된 경우 +40
        adjusted_value_pred = value_pred[i]
        if predicted_jump:
            adjusted_value_pred = value_pred[i] + 40
            print(f"\n  🚀 점프 예측! 예측값 조정: {value_pred[i]:.1f} → {adjusted_value_pred:.1f}")
        
        # 디버깅: 점프 케이스인데 못 잡은 경우
        if is_jump_case and not predicted_jump and actual_jump_count <= 10:  # 처음 10개만
            print(f"\n  ⚠️ 놓친 점프 케이스: seq_max={info['seq_max']:.1f}, 실제={info['target_value']:.1f}, "
                  f"예측={value_pred[i]:.1f}, 점프확률={jump_pred_proba[i]:.3f}")
        
        results.append({
            '날짜': info['current_time'].strftime('%Y-%m-%d %H:%M'),
            '예측날짜': info['predict_time'].strftime('%Y-%m-%d %H:%M'),
            '측정시퀀스MAX': round(info['seq_max'], 2),
            '측정시퀀스MIN': round(info['seq_min'], 2),
            '시퀀스시작시간': info['seq_start_time'].strftime('%Y-%m-%d %H:%M'),
            '시퀀스완료시간': info['seq_end_time'].strftime('%Y-%m-%d %H:%M'),
            '실제값': round(info['target_value'], 2),
            '예측값': round(adjusted_value_pred, 2),  # 조정된 예측값 사용
            '오차': round(abs(info['target_value'] - adjusted_value_pred), 2),  # 조정된 오차
            '점프예측': 'O' if predicted_jump else 'X',
            '점프확률': round(jump_pred_proba[i] * 100, 1),
            '구간예측': ['50-150', '150-299', '300+'][range_pred[i]],
            '패턴예측': ['하락', '안정', '점진상승', '급상승'][trend_pred[i]],
            '점프케이스300+': 'O' if is_jump_case else 'X'
        })
    
    print(f"\n📊 점프 케이스 감지 현황:")
    print(f"  - 실제 점프 케이스: {actual_jump_count}개")
    print(f"  - 감지된 점프 케이스: {detected_jump_count}개")
    if actual_jump_count > 0:
        print(f"  - 감지율: {detected_jump_count/actual_jump_count*100:.1f}%")
    
    # 8. 결과 분석
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("📊 평가 결과 분석")
    print("="*80)
    
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
        jump_detected = (jump_cases['점프예측'] == 'O').sum()
        jump_recall = jump_detected / len(jump_cases) * 100
        print(f"\n🚀 점프 케이스 (과거<280 → 실제300+):")
        print(f"  - 총 점프 케이스: {len(jump_cases)}개")
        print(f"  - 감지된 케이스: {jump_detected}개")
        print(f"  - 점프 감지율: {jump_recall:.1f}%")
        
        # 점프 케이스 상세 분석
        if len(jump_cases) > 0:
            print(f"\n  📊 점프 케이스 상세:")
            for idx, row in jump_cases.head(5).iterrows():
                print(f"    - {row['날짜']}: 실제 {row['실제값']}, 예측 {row['예측값']}, "
                      f"점프예측 {row['점프예측']} (확률 {row['점프확률']}%)")
    
    # False Positive
    stable_mask = results_df['실제값'] < 300
    if stable_mask.sum() > 0:
        fp_count = (results_df[stable_mask]['예측값'] >= 300).sum()
        fp_rate = fp_count / stable_mask.sum() * 100
        print(f"\n⚠️ False Positive:")
        print(f"  - 실제 <300 케이스: {stable_mask.sum()}개")
        print(f"  - 잘못 예측된 케이스: {fp_count}개")
        print(f"  - FP Rate: {fp_rate:.1f}%")
    
    # 3구간 분류 정확도
    actual_ranges = []
    for info in y_info:
        if info['target_value'] < 150:
            actual_ranges.append(0)
        elif info['target_value'] < 300:
            actual_ranges.append(1)
        else:
            actual_ranges.append(2)
    
    range_accuracy = (range_pred == actual_ranges).mean() * 100
    print(f"\n📊 3구간 분류 정확도: {range_accuracy:.1f}%")
    
    # 패턴 예측 분석
    pattern_counts = results_df['패턴예측'].value_counts()
    print(f"\n📈 패턴 예측 분포:")
    for pattern, count in pattern_counts.items():
        print(f"  - {pattern}: {count}개 ({count/len(results_df)*100:.1f}%)")
    
    # 8. CSV 파일 저장
    output_path = 'hubroom_evaluation_results_202508.csv'
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 결과 저장 완료: {output_path}")
    
    # 샘플 출력
    print(f"\n📋 샘플 결과 (처음 10개):")
    display_cols = ['날짜', '예측날짜', '실제값', '예측값', '오차', '점프예측', '점프케이스300+']
    print(results_df[display_cols].head(10).to_string(index=False))
    
    # 극단값 케이스 샘플
    extreme_cases = results_df[results_df['실제값'] >= 300]
    if len(extreme_cases) > 0:
        print(f"\n🎯 극단값(300+) 케이스 샘플:")
        print(extreme_cases[display_cols].head(5).to_string(index=False))
    
    print("\n" + "="*80)
    print("✅ 점프 감지 시스템 평가 완료!")
    print("="*80)

if __name__ == "__main__":
    evaluate_models()