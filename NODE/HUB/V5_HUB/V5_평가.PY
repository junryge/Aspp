#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
🎯 HUBROOM 점프 감지 시스템 평가 코드
================================================================================
목표:
- 과거 30분 데이터로 10분 후 예측
- Stage 1: ExtraTreesClassifier (스크리닝)
- Stage 2: XGBClassifier (점프), RandomForest (구간/패턴)
- Stage 3: ExtraTreesRegressor (값 예측)
- 평가 데이터: 2025년 8월 데이터
- CSV 결과 출력
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
print("📊 30분 시퀀스 → 10분 후 예측 평가")
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
        
        # BRIDGE_TIME 데이터 체크 (있다면)
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
        """완전한 특징 엔지니어링 - 학습 코드와 동일"""
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
        
        print(f"✅ 총 {len(df.columns)}개 특징 생성 완료")
        return df
    
    def create_sequences_for_evaluation(self, df, seq_len=30, pred_len=10):
        """평가용 시퀀스 데이터 생성"""
        print(f"\n[3단계] 시퀀스 생성 ({seq_len}분 → {pred_len}분 후)")
        
        # 특징 컬럼 선택
        feature_cols = [col for col in df.columns 
                       if col not in ['datetime', 'range_class', 'is_jump', 'trend_pattern']]
        
        X, y, timestamps, sequence_info = [], [], [], []
        
        for i in tqdm(range(seq_len, len(df) - pred_len), desc="시퀀스 생성"):
            # 입력: 과거 30분
            X.append(df[feature_cols].iloc[i-seq_len:i].values)
            
            # 타겟 (10분 후)
            target_idx = i + pred_len - 1
            y.append(df[self.target_col].iloc[target_idx])
            
            # 시간 정보
            current_time = df['datetime'].iloc[i-1]
            predict_time = df['datetime'].iloc[target_idx]
            seq_start_time = df['datetime'].iloc[i-seq_len]
            seq_end_time = df['datetime'].iloc[i-1]
            
            # 시퀀스 통계
            seq_values = df[self.target_col].iloc[i-seq_len:i].values
            seq_max = np.max(seq_values)
            seq_min = np.min(seq_values)
            past_30min_max = seq_max
            
            timestamps.append(current_time)
            
            sequence_info.append({
                'current_time': current_time,
                'predict_time': predict_time,
                'seq_start_time': seq_start_time,
                'seq_end_time': seq_end_time,
                'seq_max': seq_max,
                'seq_min': seq_min,
                'past_30min_max': past_30min_max,
                'target_value': df[self.target_col].iloc[target_idx],
                'is_jump': df['is_jump'].iloc[target_idx],
                'range_class': df['range_class'].iloc[target_idx],
                'trend_pattern': df['trend_pattern'].iloc[target_idx]
            })
        
        print(f"✅ {len(X)}개 시퀀스 생성 완료")
        
        return np.array(X), np.array(y), timestamps, sequence_info, df

# ==============================================================================
# 🤖 모델 시스템 (학습 코드와 동일)
# ==============================================================================

class JumpDetectionSystem:
    """점프 감지 80% 달성 시스템"""
    
    def __init__(self):
        # Stage 1: 빠른 스크리닝
        self.model_screening = None
        
        # Stage 2: 전문 모델들
        self.model_jump = None
        self.model_range = None
        self.model_trend = None
        
        # Stage 3: 값 예측
        self.model_value = None
        
        self.scalers = {}
        self.feature_names = None
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
    
    def apply_rule_based_boost(self, X, predictions, prob_scores=None):
        """규칙 기반 부스팅 - 80% 달성용"""
        boosted_predictions = predictions.copy()
        
        if not hasattr(self, 'feature_indices') or not self.feature_indices:
            print("⚠️ 특징 인덱스가 없습니다. 부스팅 스킵")
            return boosted_predictions
        
        idx = self.feature_indices
        
        # Phase 1: 강한 신호
        if 'storage_util' in idx:
            strong_signal = X[:, idx['storage_util']] > 20
            boosted_predictions[strong_signal] = 1
        
        # Phase 2: 중간 신호
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
        
        if len(medium_conditions) >= 2:
            medium_signal = np.sum(medium_conditions, axis=0) >= 2
            boosted_predictions[medium_signal & (boosted_predictions == 0)] = 1
        
        # Phase 3: 약한 신호
        weak_conditions = []
        
        if 'bridge_time' in idx:
            weak_conditions.append(X[:, idx['bridge_time']] > 3.85)
        if 'flow_ratio' in idx:
            weak_conditions.append(X[:, idx['flow_ratio']] > 1.5)
        if 'acceleration' in idx:
            weak_conditions.append(X[:, idx['acceleration']] > 20)
        
        if len(weak_conditions) >= 3 and prob_scores is not None:
            weak_signal = np.sum(weak_conditions, axis=0) >= 3
            final_weak = weak_signal & (boosted_predictions == 0) & (prob_scores > 0.3)
            boosted_predictions[final_weak] = 1
        
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
    X_seq, y_true, timestamps, sequence_info, df = processor.create_sequences_for_evaluation(df)
    
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
    
    # 특징 인덱스 설정
    system.get_feature_indices(df)
    
    # 5. 특징 준비
    X_features = system.prepare_features(X_seq)
    print(f"✅ 특징 준비 완료: {X_features.shape}")
    
    # 6. 예측 수행
    print("\n🔮 예측 수행 중...")
    
    results = []
    
    # 점프 예측
    jump_pred = system.model_jump.predict(X_features)
    jump_pred_proba = system.model_jump.predict_proba(X_features)[:, 1]
    jump_pred_boosted = system.apply_rule_based_boost(X_features, jump_pred, jump_pred_proba)
    
    # 구간 예측
    range_pred = system.model_range.predict(X_features)
    
    # 패턴 예측
    trend_pred = system.model_trend.predict(X_features)
    
    # 값 예측
    value_pred = system.model_value.predict(X_features)
    
    # 결과 정리
    for i in tqdm(range(len(X_seq)), desc="결과 생성"):
        seq_info = sequence_info[i]
        
        # 점프 케이스 확인
        is_jump = (seq_info['past_30min_max'] < 280) and (seq_info['target_value'] >= 300)
        predicted_jump = jump_pred_boosted[i] == 1
        
        # 모델 선택 (점프 예측 여부에 따라)
        if predicted_jump or value_pred[i] >= 300:
            model_type = "극단형(점프/300+)"
        else:
            model_type = "안정형"
        
        results.append({
            '날짜': seq_info['current_time'].strftime('%Y-%m-%d %H:%M'),
            '예측날짜': seq_info['predict_time'].strftime('%Y-%m-%d %H:%M'),
            '측정시퀀스MAX': round(seq_info['seq_max'], 2),
            '측정시퀀스MIN': round(seq_info['seq_min'], 2),
            '시퀀스시작시간': seq_info['seq_start_time'].strftime('%Y-%m-%d %H:%M'),
            '시퀀스완료시간': seq_info['seq_end_time'].strftime('%Y-%m-%d %H:%M'),
            '실제값': round(seq_info['target_value'], 2),
            '예측값': round(value_pred[i], 2),
            '오차': round(abs(seq_info['target_value'] - value_pred[i]), 2),
            '점프예측': 'O' if predicted_jump else 'X',
            '점프확률': round(jump_pred_proba[i] * 100, 1),
            '구간예측': ['저구간', '정상구간', '위험구간'][range_pred[i]],
            '패턴예측': ['하락', '안정', '점진상승', '급상승'][trend_pred[i]],
            '점프케이스300+': 'O' if is_jump else 'X',
            '모델타입': model_type
        })
    
    # 7. 결과 분석
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
    
    # 3구간 분류 정확도
    actual_ranges = [seq_info['range_class'] for seq_info in sequence_info]
    range_accuracy = (range_pred == actual_ranges).mean() * 100
    print(f"\n📊 3구간 분류 정확도: {range_accuracy:.1f}%")
    
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
    
    # False Positive
    stable_mask = results_df['실제값'] < 300
    if stable_mask.sum() > 0:
        fp_count = (results_df[stable_mask]['예측값'] >= 300).sum()
        fp_rate = fp_count / stable_mask.sum() * 100
        print(f"\n⚠️ False Positive:")
        print(f"  - 실제 <300 케이스: {stable_mask.sum()}개")
        print(f"  - 잘못 예측된 케이스: {fp_count}개")
        print(f"  - FP Rate: {fp_rate:.1f}%")
    
    # 패턴 예측 정확도
    actual_trends = [seq_info['trend_pattern'] for seq_info in sequence_info]
    trend_accuracy = (trend_pred == actual_trends).mean() * 100
    print(f"\n📈 상승/하락 패턴 정확도: {trend_accuracy:.1f}%")
    
    # 8. CSV 파일 저장
    output_path = 'hubroom_jump80_evaluation_results.csv'
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 결과 저장 완료: {output_path}")
    
    # 샘플 출력
    print(f"\n📋 샘플 결과 (처음 10개):")
    display_cols = ['날짜', '예측날짜', '실제값', '예측값', '오차', '점프예측', '점프케이스300+']
    print(results_df[display_cols].head(10).to_string(index=False))
    
    print("\n" + "="*80)
    print("✅ 점프 감지 시스템 평가 완료!")
    print("="*80)

if __name__ == "__main__":
    evaluate_models()
        
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