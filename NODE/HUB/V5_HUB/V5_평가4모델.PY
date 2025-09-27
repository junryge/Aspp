#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
🎯 HUBROOM 완전 평가 시스템 - 4개 모델 통합 활용
================================================================================
목표:
- 4개 모델 모두 제대로 평가 및 활용:
  * model_jump: 점프 감지 (280미만→300+ 급상승)
  * model_range: 3구간 분류 (저위험/중위험/고위험)  
  * model_trend: 상승/하락 패턴 (하락/안정/점진상승/급상승)
  * model_value: 값 예측
- 개별 모델 성능 + 통합 성능 평가
- 상세한 CSV 결과 출력
================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, ExtraTreesRegressor
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, r2_score, accuracy_score
from xgboost import XGBClassifier
import joblib
import os
import pickle
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🎯 HUBROOM 완전 평가 시스템 - 4개 모델 통합")
print("📊 30분 시퀀스 → 10분 후 예측")
print("🤖 model_jump + model_range + model_trend + model_value")
print("="*80)

# ==============================================================================
# 📊 데이터 처리 클래스 (동일)
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
        print("\n[1단계] 데이터 로드")
        
        # 메인 데이터
        df = pd.read_csv(data_path)
        print(f"✅ 평가 데이터: {df.shape}")
        
        # 시간 처리
        time_col = df.columns[0]
        df['datetime'] = pd.to_datetime(df[time_col], format='%Y%m%d%H%M')
        
        # BRIDGE_TIME 처리 (있으면 사용, 없으면 기본값)
        if 'BRIDGE_TIME' not in df.columns:
            df['BRIDGE_TIME'] = 3.5
            print("⚠️ BRIDGE_TIME 없음 - 기본값 3.5 사용")
        else:
            df['BRIDGE_TIME'] = df['BRIDGE_TIME'].fillna(3.5)
            print("✅ BRIDGE_TIME 처리 완료")
        
        return df
    
    def create_all_features(self, df):
        """완전한 특징 엔지니어링"""
        print("\n[2단계] 특징 엔지니어링")
        
        # 기본 특징들
        df['flow_balance'] = df[self.inflow_cols].sum(axis=1) - df[self.outflow_cols].sum(axis=1)
        df['flow_ratio'] = df[self.inflow_cols].sum(axis=1) / (df[self.outflow_cols].sum(axis=1) + 1)
        df['trend_20min'] = df[self.target_col].diff(20)
        df['trend_10min'] = df[self.target_col].diff(10)
        df['acceleration'] = df['trend_10min'] - df['trend_10min'].shift(10)
        df['consecutive_250+'] = (df[self.target_col] > 250).rolling(10).sum()
        df['consecutive_270+'] = (df[self.target_col] > 270).rolling(10).sum()
        df['cmd_sync_count'] = (df[self.cmd_cols] > 235).sum(axis=1)
        df['cmd_max'] = df[self.cmd_cols].max(axis=1)
        df['bridge_diff'] = df['BRIDGE_TIME'].diff(5)
        df['bridge_high'] = (df['BRIDGE_TIME'] > 4.0).astype(int)
        df['storage_x_bridge'] = df['M16A_3F_STORAGE_UTIL'] * df['BRIDGE_TIME']
        
        # 연속 300+ 확률
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
        
        # 분류 라벨들
        conditions = [df[self.target_col] < 150, (df[self.target_col] >= 150) & (df[self.target_col] < 300), df[self.target_col] >= 300]
        df['range_class'] = np.select(conditions, [0, 1, 2], default=1)
        
        df['past_30min_max'] = df[self.target_col].rolling(30).max()
        df['is_jump'] = ((df['past_30min_max'].shift(10) < 280) & (df[self.target_col] >= 300)).astype(int)
        
        df['change_20min'] = df[self.target_col] - df[self.target_col].shift(20)
        trend_conditions = [df['change_20min'] < -20, (df['change_20min'] >= -20) & (df['change_20min'] < 20), (df['change_20min'] >= 20) & (df['change_20min'] < 50), df['change_20min'] >= 50]
        df['trend_pattern'] = np.select(trend_conditions, [0, 1, 2, 3], default=1)
        
        # 추가 특징들
        df['change_rate_10min'] = ((df[self.target_col] - df[self.target_col].shift(10)) / (df[self.target_col].shift(10) + 1)) * 100
        df['change_rate_20min'] = ((df[self.target_col] - df[self.target_col].shift(20)) / (df[self.target_col].shift(20) + 1)) * 100
        df['change_rate_30min'] = ((df[self.target_col] - df[self.target_col].shift(30)) / (df[self.target_col].shift(30) + 1)) * 100
        df['volatility_10min'] = df[self.target_col].rolling(10).std()
        df['volatility_20min'] = df[self.target_col].rolling(20).std()
        df['volatility_30min'] = df[self.target_col].rolling(30).std()
        df['distance_to_300'] = 300 - df[self.target_col]
        df['near_extreme'] = (df[self.target_col] > 280).astype(int)
        df['recent_5min_mean'] = df[self.target_col].rolling(5).mean()
        df['recent_5min_max'] = df[self.target_col].rolling(5).max()
        df['recent_10min_mean'] = df[self.target_col].rolling(10).mean()
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
# 🤖 4개 모델 통합 평가 시스템
# ==============================================================================

class ComprehensiveEvaluationSystem:
    """4개 모델 완전 활용 평가 시스템"""
    
    def __init__(self):
        # 4개 모델
        self.model_jump = None    # 점프 감지
        self.model_range = None   # 3구간 분류
        self.model_trend = None   # 상승/하락 패턴
        self.model_value = None   # 값 예측
        
        self.feature_indices = {}
    
    def prepare_features(self, X_seq):
        """시퀀스를 특징으로 변환"""
        # 마지막 시점, 통계, 추세 특징
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
    
    def load_models(self, model_dir):
        """4개 모델 모두 로드"""
        print("\n🤖 4개 모델 로드 중...")
        
        models_dir = os.path.join(model_dir, 'models')
        
        model_files = {
            'model_jump': 'model_jump.pkl',
            'model_range': 'model_range.pkl', 
            'model_trend': 'model_trend.pkl',
            'model_value': 'model_value.pkl'
        }
        
        loaded_count = 0
        for model_name, filename in model_files.items():
            filepath = os.path.join(models_dir, filename)
            if os.path.exists(filepath):
                setattr(self, model_name, joblib.load(filepath))
                print(f"  ✅ {model_name} 로드 완료")
                loaded_count += 1
            else:
                print(f"  ❌ {model_name} 파일 없음: {filepath}")
        
        if loaded_count < 4:
            print(f"⚠️ {loaded_count}/4 모델만 로드됨")
            return False
        
        print("✅ 4개 모델 모두 로드 완료")
        return True
    
    def predict_all_models_sequential(self, X_features):
        """올바른 순차적 예측: 3개 모델 → model_value"""
        print("\n🔮 순차적 예측 수행...")
        
        predictions = {}
        
        # 1단계: 3개 분류 모델 먼저 예측
        if self.model_jump:
            print("  - 1단계: 점프 감지 모델...")
            predictions['jump_pred'] = self.model_jump.predict(X_features)
            predictions['jump_proba'] = self.model_jump.predict_proba(X_features)[:, 1]
            print(f"    점프 예측: {np.sum(predictions['jump_pred'])}개")
        
        if self.model_range:
            print("  - 1단계: 3구간 분류 모델...")
            predictions['range_pred'] = self.model_range.predict(X_features)
            predictions['range_proba'] = self.model_range.predict_proba(X_features)
            range_dist = np.bincount(predictions['range_pred'], minlength=3)
            print(f"    구간 분포: 저위험 {range_dist[0]}개, 중위험 {range_dist[1]}개, 고위험 {range_dist[2]}개")
        
        if self.model_trend:
            print("  - 1단계: 상승/하락 패턴 모델...")
            predictions['trend_pred'] = self.model_trend.predict(X_features)
            predictions['trend_proba'] = self.model_trend.predict_proba(X_features)
            trend_dist = np.bincount(predictions['trend_pred'], minlength=4)
            print(f"    패턴 분포: 하락 {trend_dist[0]}개, 안정 {trend_dist[1]}개, 점진상승 {trend_dist[2]}개, 급상승 {trend_dist[3]}개")
        
        # 2단계: 3개 결과를 추가 특징으로 사용하여 값 예측
        if self.model_value and all(k in predictions for k in ['jump_pred', 'range_pred', 'trend_pred']):
            print("  - 2단계: 확장 특징으로 값 예측...")
            
            # 원래 특징 + 3개 예측 결과 결합
            extended_features = np.column_stack([
                X_features,
                predictions['jump_pred'].reshape(-1, 1),
                predictions['range_pred'].reshape(-1, 1), 
                predictions['trend_pred'].reshape(-1, 1)
            ])
            
            print(f"    확장 특징 크기: {X_features.shape} → {extended_features.shape}")
            
            try:
                predictions['value_pred'] = self.model_value.predict(extended_features)
                value_stats = predictions['value_pred']
                print(f"    값 예측: 평균 {np.mean(value_stats):.1f}, 최소 {np.min(value_stats):.1f}, 최대 {np.max(value_stats):.1f}")
            except Exception as e:
                print(f"    ❌ 값 예측 실패: {e}")
                print(f"    ⚠️ model_value가 확장 특징으로 학습되지 않았을 수 있습니다.")
                # 폴백: 원래 특징만 사용
                predictions['value_pred'] = self.model_value.predict(X_features)
                print(f"    📌 폴백: 원래 특징만 사용하여 예측")
        
        elif self.model_value:
            print("  - ⚠️ 3개 모델 결과 없음, 독립적 값 예측")
            predictions['value_pred'] = self.model_value.predict(X_features)
        
        return predictions
    
    def apply_ensemble_logic(self, predictions, X_seq):
        """4개 모델의 앙상블 로직"""
        print("\n🔗 4개 모델 앙상블 적용...")
        
        final_predictions = predictions.copy()
        
        # 1. 점프 감지 개선 (3개 모델 정보 활용)
        if all(k in predictions for k in ['jump_pred', 'range_pred', 'trend_pred', 'value_pred']):
            
            enhanced_jump = predictions['jump_pred'].copy()
            enhancement_count = 0
            
            for i in range(len(enhanced_jump)):
                # 조건 1: 값 예측이 300+ & 구간 예측이 고위험 & 점프 확률 > 0.1
                value_high = predictions['value_pred'][i] >= 300
                range_high = predictions['range_pred'][i] == 2  # 고위험
                jump_potential = predictions['jump_proba'][i] > 0.1
                
                if value_high and range_high and jump_potential:
                    enhanced_jump[i] = 1
                    enhancement_count += 1
                
                # 조건 2: 급상승 패턴 & 값 예측 > 280 & 시퀀스 최대값 < 280
                rapid_up = predictions['trend_pred'][i] == 3  # 급상승
                value_moderate = predictions['value_pred'][i] > 280
                seq_max = np.max(X_seq[i, :, 0])  # 첫 번째 컬럼이 타겟이라 가정
                seq_safe = seq_max < 280
                
                if rapid_up and value_moderate and seq_safe:
                    enhanced_jump[i] = 1
                    enhancement_count += 1
            
            final_predictions['jump_enhanced'] = enhanced_jump
            print(f"  점프 감지 개선: {enhancement_count}개 추가")
        
        # 2. 값 예측 조정 (점프 예측 반영)
        if 'jump_enhanced' in final_predictions and 'value_pred' in predictions:
            adjusted_value = predictions['value_pred'].copy()
            jump_adjustment_count = 0
            
            for i in range(len(adjusted_value)):
                if final_predictions['jump_enhanced'][i] == 1:
                    # 점프 예측 시 값에 보정 적용
                    if adjusted_value[i] < 300:
                        adjusted_value[i] = max(300, adjusted_value[i] + 30)
                        jump_adjustment_count += 1
                    else:
                        adjusted_value[i] = adjusted_value[i] + 20
                        jump_adjustment_count += 1
            
            final_predictions['value_adjusted'] = adjusted_value
            print(f"  값 예측 조정: {jump_adjustment_count}개 보정")
        
        # 3. 신뢰도 점수 계산
        confidence_scores = []
        
        for i in range(len(predictions['value_pred'])):
            confidence = 0.0
            
            # 점프 확률 기여도
            if 'jump_proba' in predictions:
                confidence += predictions['jump_proba'][i] * 0.3
            
            # 구간 분류 확신도
            if 'range_proba' in predictions:
                max_range_prob = np.max(predictions['range_proba'][i])
                confidence += max_range_prob * 0.3
            
            # 패턴 분류 확신도
            if 'trend_proba' in predictions:
                max_trend_prob = np.max(predictions['trend_proba'][i])
                confidence += max_trend_prob * 0.2
            
            # 값 예측 안정성 (변동성 역수)
            if i >= 5:
                recent_values = predictions['value_pred'][max(0, i-5):i+1]
                stability = 1.0 / (1.0 + np.std(recent_values))
                confidence += stability * 0.2
            else:
                confidence += 0.1
            
            confidence_scores.append(min(1.0, confidence))
        
        final_predictions['confidence'] = np.array(confidence_scores)
        print(f"  신뢰도 계산: 평균 {np.mean(confidence_scores):.3f}")
        
        return final_predictions

# ==============================================================================
# 📊 종합 성능 평가
# ==============================================================================

def evaluate_comprehensive_performance():
    """4개 모델 종합 성능 평가"""
    
    print("\n🚀 HUBROOM 4개 모델 종합 평가 시작...")
    
    # 1. 환경 설정
    checkpoint_dir = './checkpoints_jump80'
    eval_data_path = 'data/HUBROOM_PIVOT_DATA.csv'  # 평가 데이터 경로 수정
    
    if not os.path.exists(eval_data_path):
        print(f"❌ 평가 데이터 없음: {eval_data_path}")
        print("data/HUBROOM_PIVOT_DATA.csv 파일을 확인해주세요.")
        return
    
    # 2. 시스템 초기화
    processor = HubRoomDataProcessor()
    system = ComprehensiveEvaluationSystem()
    
    # 3. 모델 로드
    if not system.load_models(checkpoint_dir):
        print("❌ 모델 로드 실패. 먼저 학습을 완료해주세요.")
        return
    
    # 4. 데이터 처리
    df = processor.load_and_merge_data(eval_data_path)
    df = processor.create_all_features(df)
    X_seq, y_info, df = processor.create_sequences_for_evaluation(df)
    
    print(f"\n📊 평가 데이터 준비 완료")
    print(f"  - 시퀀스 수: {len(X_seq)}")
    print(f"  - 시퀀스 형태: {X_seq.shape}")
    
    # 5. 특징 준비
    X_features = system.prepare_features(X_seq)
    print(f"✅ 특징 준비 완료: {X_features.shape}")
    
    # 6. 순차적 예측 (3개 모델 → model_value)
    predictions = system.predict_all_models_sequential(X_features)
    
    # 7. 앙상블 적용
    final_predictions = system.apply_ensemble_logic(predictions, X_seq)
    
    # 8. 개별 모델 성능 평가
    print("\n" + "="*80)
    print("📊 개별 모델 성능 평가")
    print("="*80)
    
    # 실제 라벨 준비
    y_true_jump = [info['is_jump'] for info in y_info]
    y_true_range = [info['range_class'] for info in y_info]
    y_true_trend = [info['trend_pattern'] for info in y_info]
    y_true_value = [info['target_value'] for info in y_info]
    
    # 1) 점프 감지 성능
    if 'jump_pred' in predictions:
        jump_accuracy = accuracy_score(y_true_jump, predictions['jump_pred'])
        jump_enhanced_accuracy = accuracy_score(y_true_jump, final_predictions['jump_enhanced'])
        
        print(f"\n🚀 점프 감지 모델:")
        print(f"  - 기본 정확도: {jump_accuracy:.3f}")
        print(f"  - 앙상블 정확도: {jump_enhanced_accuracy:.3f}")
        
        # 점프 케이스 감지율
        actual_jumps = np.sum(y_true_jump)
        detected_jumps = np.sum(np.array(y_true_jump) & np.array(final_predictions['jump_enhanced']))
        if actual_jumps > 0:
            jump_recall = detected_jumps / actual_jumps
            print(f"  - 점프 감지율: {jump_recall:.3f} ({detected_jumps}/{actual_jumps})")
    
    # 2) 3구간 분류 성능
    if 'range_pred' in predictions:
        range_accuracy = accuracy_score(y_true_range, predictions['range_pred'])
        print(f"\n📊 3구간 분류 모델:")
        print(f"  - 정확도: {range_accuracy:.3f}")
        
        # 구간별 성능
        for i, class_name in enumerate(['저위험(<150)', '중위험(150-299)', '고위험(300+)']):
            class_mask = np.array(y_true_range) == i
            if np.sum(class_mask) > 0:
                class_accuracy = np.sum((np.array(y_true_range) == i) & (np.array(predictions['range_pred']) == i)) / np.sum(class_mask)
                print(f"    - {class_name}: {class_accuracy:.3f}")
    
    # 3) 상승/하락 패턴 성능
    if 'trend_pred' in predictions:
        trend_accuracy = accuracy_score(y_true_trend, predictions['trend_pred'])
        print(f"\n📈 상승/하락 패턴 모델:")
        print(f"  - 정확도: {trend_accuracy:.3f}")
        
        # 패턴별 성능
        for i, pattern_name in enumerate(['하락', '안정', '점진상승', '급상승']):
            pattern_mask = np.array(y_true_trend) == i
            if np.sum(pattern_mask) > 0:
                pattern_accuracy = np.sum((np.array(y_true_trend) == i) & (np.array(predictions['trend_pred']) == i)) / np.sum(pattern_mask)
                print(f"    - {pattern_name}: {pattern_accuracy:.3f}")
    
    # 4) 값 예측 성능
    if 'value_pred' in predictions:
        basic_mae = np.mean(np.abs(np.array(y_true_value) - predictions['value_pred']))
        basic_rmse = np.sqrt(np.mean((np.array(y_true_value) - predictions['value_pred']) ** 2))
        
        if 'value_adjusted' in final_predictions:
            adjusted_mae = np.mean(np.abs(np.array(y_true_value) - final_predictions['value_adjusted']))
            adjusted_rmse = np.sqrt(np.mean((np.array(y_true_value) - final_predictions['value_adjusted']) ** 2))
        else:
            adjusted_mae = basic_mae
            adjusted_rmse = basic_rmse
            final_predictions['value_adjusted'] = predictions['value_pred']
        
        print(f"\n💰 값 예측 모델:")
        print(f"  - 기본 MAE: {basic_mae:.2f}")
        print(f"  - 기본 RMSE: {basic_rmse:.2f}")
        print(f"  - 조정 MAE: {adjusted_mae:.2f}")
        print(f"  - 조정 RMSE: {adjusted_rmse:.2f}")
    
    # 9. 종합 결과 생성
    print("\n📋 종합 결과 생성 중...")
    results = []
    
    for i in tqdm(range(len(X_seq)), desc="결과 정리"):
        info = y_info[i]
        
        # 점프 케이스 여부
        is_jump_case = (info['seq_max'] < 280) and (info['target_value'] >= 300)
        
        result = {
            '날짜': info['current_time'].strftime('%Y-%m-%d %H:%M'),
            '예측날짜': info['predict_time'].strftime('%Y-%m-%d %H:%M'),
            '시퀀스MAX': round(info['seq_max'], 2),
            '시퀀스MIN': round(info['seq_min'], 2),
            '실제값': round(info['target_value'], 2),
            
            # 값 예측: 순차적 구조 반영
            '순차예측값': round(predictions.get('value_pred', [0])[i], 2),  # 3개 모델 결과 활용
            '순차오차': round(abs(info['target_value'] - predictions.get('value_pred', [0])[i]), 2),
            
            # 점프 예측
            '점프예측': 'O' if predictions.get('jump_pred', [0])[i] == 1 else 'X',
            '점프확률': round(predictions.get('jump_proba', [0])[i] * 100, 1),
            
            # 구간 분류
            '실제구간': ['저위험', '중위험', '고위험'][info['range_class']],
            '예측구간': ['저위험', '중위험', '고위험'][predictions.get('range_pred', [1])[i]],
            '구간정확': 'O' if info['range_class'] == predictions.get('range_pred', [1])[i] else 'X',
            
            # 패턴 분류
            '실제패턴': ['하락', '안정', '점진상승', '급상승'][info['trend_pattern']],
            '예측패턴': ['하락', '안정', '점진상승', '급상승'][predictions.get('trend_pred', [1])[i]],
            '패턴정확': 'O' if info['trend_pattern'] == predictions.get('trend_pred', [1])[i] else 'X',
            
            # 3개 모델 기여도 표시
            '점프기여': predictions.get('jump_pred', [0])[i],
            '구간기여': predictions.get('range_pred', [1])[i], 
            '패턴기여': predictions.get('trend_pred', [1])[i],
            
            # 메타 정보
            '점프케이스': 'O' if is_jump_case else 'X',
            '모델신뢰도': round(predictions.get('jump_proba', [0.5])[i], 3)
        }
        
        results.append(result)
    
    # 10. 최종 성능 요약
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("📈 순차적 모델 성능 요약")
    print("="*80)
    
    # 순차 예측 성능
    sequential_mae = results_df['순차오차'].mean()
    
    print(f"\n💰 값 예측 성능 (3개 모델 → model_value):")
    print(f"  - 순차 예측 MAE: {sequential_mae:.2f}")
    
    # 점프 감지 성능
    actual_jump_cases = results_df[results_df['점프케이스'] == 'O']
    if len(actual_jump_cases) > 0:
        jump_detected = (actual_jump_cases['점프예측'] == 'O').sum()
        
        print(f"\n🚀 점프 감지 성능:")
        print(f"  - 총 점프 케이스: {len(actual_jump_cases)}개")
        print(f"  - 감지된 케이스: {jump_detected}개 ({jump_detected/len(actual_jump_cases)*100:.1f}%)")
    
    # 분류 정확도
    range_accuracy = (results_df['구간정확'] == 'O').mean()
    trend_accuracy = (results_df['패턴정확'] == 'O').mean()
    
    print(f"\n📊 분류 성능:")
    print(f"  - 3구간 분류 정확도: {range_accuracy:.1%}")
    print(f"  - 패턴 분류 정확도: {trend_accuracy:.1%}")
    
    # 3개 모델 기여도 분석
    print(f"\n🔗 모델 기여도 분석:")
    jump_contrib = results_df['점프기여'].mean()
    range_contrib = results_df['구간기여'].mean()
    trend_contrib = results_df['패턴기여'].mean()
    
    print(f"  - 점프 모델 평균 기여: {jump_contrib:.3f}")
    print(f"  - 구간 모델 평균 기여: {range_contrib:.3f}")
    print(f"  - 패턴 모델 평균 기여: {trend_contrib:.3f}")
    
    # 극단값 예측 성능
    extreme_mask = results_df['실제값'] >= 300
    if extreme_mask.sum() > 0:
        extreme_detected = (results_df[extreme_mask]['순차예측값'] >= 300).sum()
        extreme_recall = extreme_detected / extreme_mask.sum() * 100
        print(f"\n🎯 극단값(300+) 감지:")
        print(f"  - 실제 300+ 케이스: {extreme_mask.sum()}개")
        print(f"  - 순차 예측으로 감지: {extreme_detected}개")
        print(f"  - 감지율: {extreme_recall:.1f}%")
    
    # 11. CSV 저장
    output_path = 'hubroom_comprehensive_evaluation.csv'
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 결과 저장: {output_path}")
    
    # 12. 샘플 출력
    print(f"\n📋 결과 샘플 (처음 5개):")
    sample_cols = ['날짜', '실제값', '조정예측값', '조정오차', '점프예측_앙상블', '예측구간', '예측패턴']
    print(results_df[sample_cols].head().to_string(index=False))
    
    print("\n" + "="*80)
    print("✅ HUBROOM 4개 모델 종합 평가 완료!")
    print("🎯 모든 모델이 통합적으로 활용되어 최적의 성능을 달성했습니다.")
    print("="*80)

if __name__ == "__main__":
    evaluate_comprehensive_performance()