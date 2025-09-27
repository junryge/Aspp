#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
🎯 HUBROOM 실시간 예측 시스템
================================================================================
목표:
- 최신 30분 데이터로 10분 후 예측
- 예측값 및 상태(정상/주의/심각) 반환
================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, ExtraTreesRegressor
from sklearn.preprocessing import RobustScaler, LabelEncoder
from xgboost import XGBClassifier
import joblib
import os
import pickle
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

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
        
        # 확률 맵
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
        print("\n📂 데이터 로드 중...")
        
        # 메인 데이터
        df = pd.read_csv(data_path)
        print(f"✅ 데이터 로드: {df.shape}")
        
        # 시간 처리
        time_col = df.columns[0]
        df['datetime'] = pd.to_datetime(df[time_col], format='%Y%m%d%H%M')
        
        # BRIDGE_TIME 데이터 확인
        bridge_path = data_path.replace('.csv', '_BRIDGE.csv')
        if os.path.exists(bridge_path):
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
                
                # BRIDGE_TIME 보간
                df['BRIDGE_TIME'] = df['BRIDGE_TIME'].interpolate(method='linear', limit_direction='both')
        
        # BRIDGE_TIME이 없으면 기본값
        if 'BRIDGE_TIME' not in df.columns:
            df['BRIDGE_TIME'] = 3.5
            
        df['BRIDGE_TIME'] = df['BRIDGE_TIME'].fillna(3.5)
        
        return df
    
    def create_all_features(self, df):
        """완전한 특징 엔지니어링"""
        print("🔧 특징 생성 중...")
        
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
        
        for i in range(len(df)):
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
        
        # 9. 점프 여부
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
        
        print(f"✅ {len(df.columns)}개 특징 생성 완료")
        return df

# ==============================================================================
# 🤖 점프 감지 시스템
# ==============================================================================

class JumpDetectionSystem:
    """점프 감지 시스템"""
    
    def __init__(self):
        self.model_jump = None
        self.model_range = None
        self.model_trend = None
        self.model_value = None
        self.feature_indices = {}
    
    def load_models(self, models_dir):
        """모델 로드"""
        print("\n🤖 모델 로드 중...")
        
        self.model_jump = joblib.load(os.path.join(models_dir, 'model_jump.pkl'))
        self.model_range = joblib.load(os.path.join(models_dir, 'model_range.pkl'))
        self.model_trend = joblib.load(os.path.join(models_dir, 'model_trend.pkl'))
        self.model_value = joblib.load(os.path.join(models_dir, 'model_value.pkl'))
        
        print("✅ 모델 로드 완료")
    
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
    
    def get_expanded_feature_indices(self, df):
        """확장된 특징 인덱스 계산"""
        feature_cols = [col for col in df.columns 
                       if col not in ['datetime', 'range_class', 'is_jump', 'trend_pattern']]
        
        n_base_features = len(feature_cols)
        expanded_indices = {}
        
        # 원본 특징의 인덱스 찾기
        base_indices = {}
        for i, col in enumerate(feature_cols):
            if 'STORAGE_UTIL' in col:
                base_indices['storage_util'] = i
            elif col == 'BRIDGE_TIME':
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
        
        self.feature_indices = expanded_indices
        return expanded_indices
    
    def apply_rule_based_boost_v3(self, X, predictions, prob_scores=None):
        """규칙 기반 부스팅"""
        boosted_predictions = predictions.copy()
        
        if not self.feature_indices:
            return boosted_predictions
        
        idx = self.feature_indices
        
        # Phase 1: storage_util 기반
        conditions = []
        if 'storage_util_last' in idx:
            conditions.append(X[:, idx['storage_util_last']] > 15)
        if 'storage_util_max' in idx:
            conditions.append(X[:, idx['storage_util_max']] > 20)
        if 'storage_util_mean' in idx:
            conditions.append(X[:, idx['storage_util_mean']] > 10)
        
        if conditions:
            strong_signal = np.any(conditions, axis=0)
            boosted_predictions[strong_signal] = 1
        
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
            boosted_predictions[bridge_signal] = 1
        
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
            boosted_predictions[complex_signal] = 1
        
        # Phase 4: 확률 기반
        if prob_scores is not None:
            prob_boost = (prob_scores > 0.1) & (boosted_predictions == 0)
            boosted_predictions[prob_boost] = 1
        
        # Phase 5: 점프존 감지
        if 'in_jump_zone_last' in idx:
            jump_zone = X[:, idx['in_jump_zone_last']] > 0
            boosted_predictions[jump_zone] = 1
        
        # Phase 6: recent_5min_max
        if 'recent_5min_max_last' in idx:
            recent_high = X[:, idx['recent_5min_max_last']] >= 275
            boosted_predictions[recent_high] = 1
        
        return boosted_predictions
    
    def apply_emergency_boost(self, X_seq, X_features, predictions, value_pred):
        """긴급 부스팅"""
        boosted = predictions.copy()
        
        for i in range(len(X_seq)):
            seq_max = np.max(X_seq[i, :, 0])
            
            # 규칙 1: 값 예측이 290 이상이고 과거 최대값이 280 미만
            if value_pred[i] >= 290 and seq_max < 280:
                boosted[i] = 1
            
            # 규칙 2: 최근 5분 평균이 275 이상이고 과거 최대값이 279 이하
            recent_5min = np.mean(X_seq[i, -5:, 0])
            if recent_5min >= 275 and seq_max <= 279:
                boosted[i] = 1
            
            # 규칙 3: 가속도가 크고 현재값이 270 이상
            if len(X_seq[i]) >= 20:
                accel = (np.mean(X_seq[i, -5:, 0]) - np.mean(X_seq[i, -10:-5, 0])) - \
                       (np.mean(X_seq[i, -10:-5, 0]) - np.mean(X_seq[i, -15:-10, 0]))
                current = X_seq[i, -1, 0]
                
                if accel > 15 and current >= 270:
                    boosted[i] = 1
        
        return boosted

# ==============================================================================
# 🔮 실시간 예측 함수
# ==============================================================================

def predict_realtime():
    """실시간 예측 메인 함수"""
    
    print("="*80)
    print("🎯 HUBROOM 실시간 예측 시스템")
    print("📊 최신 30분 데이터 → 10분 후 예측")
    print("="*80)
    
    # 1. 체크포인트 확인
    checkpoint_dir = './checkpoints_jump80'
    models_dir = os.path.join(checkpoint_dir, 'models')
    
    if not os.path.exists(models_dir):
        print("❌ 학습된 모델을 찾을 수 없습니다.")
        return None
    
    # 2. 데이터 처리기 및 시스템 초기화
    processor = HubRoomDataProcessor()
    system = JumpDetectionSystem()
    
    # 3. 모델 로드
    system.load_models(models_dir)
    
    # 4. 데이터 로드
    data_path = 'data/HUBROOM_PIVOT_DATA.csv'
    if not os.path.exists(data_path):
        print(f"❌ 데이터 파일을 찾을 수 없습니다: {data_path}")
        return None
    
    df = processor.load_and_merge_data(data_path)
    df = processor.create_all_features(df)
    
    # 5. 특징 인덱스 설정
    system.get_expanded_feature_indices(df)
    
    # 6. 최신 30분 시퀀스 생성
    seq_len = 30
    
    if len(df) < seq_len:
        print(f"❌ 데이터가 부족합니다. 최소 {seq_len}개 필요 (현재: {len(df)}개)")
        return None
    
    # 특징 컬럼 선택
    feature_cols = [col for col in df.columns 
                   if col not in ['datetime', 'range_class', 'is_jump', 'trend_pattern']]
    
    # 최신 30분 데이터
    X_seq = df[feature_cols].iloc[-seq_len:].values
    X_seq = X_seq.reshape(1, seq_len, -1)  # (1, 30, features)
    
    current_time = df['datetime'].iloc[-1]
    predict_time = current_time + timedelta(minutes=10)
    
    print(f"\n⏰ 현재 시간: {current_time}")
    print(f"🔮 예측 시간: {predict_time} (10분 후)")
    
    # 7. 특징 준비
    X_features = system.prepare_features(X_seq)
    
    # 8. 예측 수행
    print("\n📈 예측 수행 중...")
    
    # 점프 예측
    jump_pred = system.model_jump.predict(X_features)
    jump_pred_proba = system.model_jump.predict_proba(X_features)[:, 1]
    
    # 값 예측
    value_pred = system.model_value.predict(X_features)
    
    # 규칙 기반 부스팅
    jump_pred_boosted = system.apply_rule_based_boost_v3(X_features, jump_pred, jump_pred_proba)
    
    # 긴급 부스팅
    jump_pred_final = system.apply_emergency_boost(X_seq, X_features, jump_pred_boosted, value_pred)
    
    # 구간 예측
    range_pred = system.model_range.predict(X_features)
    
    # 패턴 예측
    trend_pred = system.model_trend.predict(X_features)
    
    # 9. 예측값 조정
    predicted_value = value_pred[0]
    is_jump = jump_pred_final[0] == 1
    
    if is_jump:
        adjusted_value = predicted_value + 40
        print(f"\n🚀 점프 감지! 예측값 조정: {predicted_value:.1f} → {adjusted_value:.1f}")
        predicted_value = adjusted_value
    
    # 10. 상태 판정
    if predicted_value < 250:
        status = "정상"
        status_percent = {"정상": 80, "주의": 15, "심각": 5}
    elif predicted_value < 280:
        status = "주의"
        status_percent = {"정상": 20, "주의": 60, "심각": 20}
    else:
        status = "심각"
        status_percent = {"정상": 5, "주의": 15, "심각": 80}
    
    # 점프 예측이면 심각도 상향
    if is_jump:
        status = "심각"
        status_percent = {"정상": 2, "주의": 8, "심각": 90}
    
    # 11. 결과 생성
    result = {
        "현재시간": current_time.strftime('%Y-%m-%d %H:%M'),
        "예측시간": predict_time.strftime('%Y-%m-%d %H:%M'),
        "예측값": round(predicted_value, 1),
        "상태": status,
        "상태확률": status_percent,
        "점프예측": "예상" if is_jump else "정상",
        "점프확률": round(jump_pred_proba[0] * 100, 1),
        "구간예측": ['50-150', '150-299', '300+'][range_pred[0]],
        "패턴예측": ['하락', '안정', '점진상승', '급상승'][trend_pred[0]]
    }
    
    # 12. 결과 출력
    print("\n" + "="*80)
    print("📊 예측 결과")
    print("="*80)
    print(f"🔮 10분 후 예측값: {result['예측값']}")
    print(f"⚠️ 상태: {result['상태']}")
    print(f"📊 상태 확률: 정상 {result['상태확률']['정상']}% | 주의 {result['상태확률']['주의']}% | 심각 {result['상태확률']['심각']}%")
    print(f"🚀 점프 예측: {result['점프예측']} (확률: {result['점프확률']}%)")
    print(f"📈 패턴: {result['패턴예측']}")
    print(f"📍 구간: {result['구간예측']}")
    print("="*80)
    
    return result

# ==============================================================================
# 🎯 실행
# ==============================================================================

if __name__ == "__main__":
    result = predict_realtime()
    
    if result:
        print("\n✅ 예측 완료!")
        print(f"\n🔄 반환값:")
        print(f"  예측값: {result['예측값']}")
        print(f"  상태확률: {result['상태확률']}")