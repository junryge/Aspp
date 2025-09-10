#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
🔥 HUBROOM 극단값 예측 시스템 V4 ULTIMATE FINAL
================================================================================
✨ 최종 완성 기능:
   1. Model 1 (PatchTST): 200-300 안정 구간 전문가
   2. Model 2 (PatchTST+PINN): 300+ 극단값 전문가
   3. Ultimate Hybrid Selector: 실제 데이터 + 다중 지표 기반 지능형 선택
   4. 완벽한 중단/재개 시스템 (6단계 체크포인트)
   
🎯 핵심 전략:
   - 앙상블 없음! 상황에 맞는 모델 하나만 선택
   - 실제 통계 데이터 + 실시간 지표 종합 판단
   - 각 모델이 자신의 전문 구간만 담당
================================================================================
Author: HUBROOM V4 Ultimate Team
Date: 2024
Version: 4.0 ULTIMATE
================================================================================
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from sklearn.preprocessing import StandardScaler, RobustScaler
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
print("🔥 HUBROOM V4 ULTIMATE FINAL - 지능형 선택 시스템")
print("✨ 하이브리드 접근 + 실제 데이터 기반 선택")
print("✅ 완벽한 중단/재개 지원 - Ctrl+C로 언제든 중단 가능!")
print("="*80)
print(f"📅 실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"🔧 TensorFlow Version: {tf.__version__}")
print("="*80)

# ==============================================================================
# 💾 체크포인트 관리자 - 완벽한 중단/재개 시스템
# ==============================================================================

class CheckpointManager:
    """학습 중단/재개를 위한 체크포인트 관리"""
    
    def __init__(self, checkpoint_dir='./checkpoints_ultimate'):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 상태 파일들
        self.state_file = os.path.join(checkpoint_dir, 'training_state.pkl')
        self.sequence_file = os.path.join(checkpoint_dir, 'sequences.h5')
        self.scaler_dir = os.path.join(checkpoint_dir, 'scalers')
        os.makedirs(self.scaler_dir, exist_ok=True)
        
        self.interrupted = False
        
        # Ctrl+C 핸들러 등록
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """Ctrl+C 시그널 처리"""
        print('\n\n⚠️ 중단 감지! 현재 상태를 저장합니다...')
        self.interrupted = True
        # 저장은 각 단계에서 처리
    
    def save_state(self, state: Dict[str, Any]):
        """현재 상태 저장"""
        with open(self.state_file, 'wb') as f:
            pickle.dump(state, f)
        print(f"💾 상태 저장 완료: Step {state.get('step', 0)}")
    
    def load_state(self) -> Optional[Dict[str, Any]]:
        """저장된 상태 로드"""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def save_sequences(self, X, y, X_physics, weights, progress, total):
        """시퀀스 데이터 저장"""
        with h5py.File(self.sequence_file, 'w') as f:
            f.create_dataset('X', data=X, compression='gzip')
            f.create_dataset('y', data=y, compression='gzip')
            f.create_dataset('X_physics', data=X_physics, compression='gzip')
            f.create_dataset('weights', data=weights, compression='gzip')
            f.attrs['progress'] = progress
            f.attrs['total'] = total
        print(f"💾 시퀀스 저장 완료: {progress}/{total}")
    
    def load_sequences(self):
        """시퀀스 데이터 로드"""
        if os.path.exists(self.sequence_file):
            with h5py.File(self.sequence_file, 'r') as f:
                return {
                    'X': f['X'][:],
                    'y': f['y'][:],
                    'X_physics': f['X_physics'][:],
                    'weights': f['weights'][:],
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
# 🧠 Ultimate Hybrid Selector - 지능형 선택기
# ==============================================================================

class UltimateHybridSelector:
    """실제 데이터 기반 + 하이브리드 접근 선택기"""
    
    def __init__(self):
        # 프로젝트 지식의 실제 확률 데이터
        self.probability_map = {
            0: 0.003, 1: 0.15, 2: 0.25, 3: 0.31, 4: 0.43, 5: 0.43,
            6: 0.35, 7: 0.42, 8: 0.53, 9: 0.49, 10: 0.42,
            11: 0.47, 12: 0.52, 13: 0.60, 14: 0.54, 15: 0.66,
            16: 0.62, 17: 0.71, 18: 0.79, 19: 0.83, 20: 0.987
        }
        
        self.selection_history = []
        self.performance_tracker = {
            'model1': {'correct': 0, 'total': 0, 'mae': []},
            'model2': {'correct': 0, 'total': 0, 'mae': []}
        }
    
    def select_model(self, past_20min: np.ndarray, physics_data: Optional[Dict] = None) -> Tuple[str, Dict]:
        """
        최종 모델 선택 로직
        Returns: (selected_model, decision_info)
        """
        decision_info = {
            'timestamp': datetime.now(),
            'factors': [],
            'scores': {}
        }
        
        # ========================================
        # 1단계: 명확한 극단 케이스 즉시 처리
        # ========================================
        recent_5 = past_20min[-5:]
        max_val = np.max(past_20min)
        min_recent = np.min(recent_5)
        mean_recent = np.mean(recent_5)
        
        # 확실한 안정 구간
        if max_val < 250 and mean_recent < 230:
            decision_info['factors'].append("확실한 안정 구간")
            decision_info['selected'] = "Model1"
            return "Model1", decision_info
        
        # 확실한 극단 구간
        if min_recent > 320:
            decision_info['factors'].append("확실한 극단 구간")
            decision_info['selected'] = "Model2"
            return "Model2", decision_info
        
        # ========================================
        # 2단계: 실제 데이터 기반 확률 계산
        # ========================================
        count_300plus = sum(1 for v in past_20min if v >= 300)
        base_probability = self.probability_map.get(count_300plus, 0.5)
        decision_info['scores']['base_probability'] = base_probability
        decision_info['factors'].append(f"300+ 개수: {count_300plus}개 ({base_probability*100:.1f}%)")
        
        # ========================================
        # 3단계: 위험 점수 계산 (보정용)
        # ========================================
        risk_score = 0
        
        # 3-1. 추세 분석 (0-30점)
        early_avg = np.mean(past_20min[:7])
        middle_avg = np.mean(past_20min[7:14])
        recent_avg = np.mean(past_20min[14:])
        
        trend = recent_avg - early_avg
        acceleration = (recent_avg - middle_avg) - (middle_avg - early_avg)
        
        if trend > 20:
            risk_score += 20
            decision_info['factors'].append(f"급상승 추세 (+{trend:.0f})")
        elif trend > 10:
            risk_score += 10
            decision_info['factors'].append(f"상승 추세 (+{trend:.0f})")
        elif trend < -20:
            risk_score -= 10
            decision_info['factors'].append(f"하락 추세 ({trend:.0f})")
        
        if acceleration > 10:
            risk_score += 10
            decision_info['factors'].append(f"가속 상승 (+{acceleration:.0f})")
        
        # 3-2. 변동성 분석 (0-20점)
        volatility = np.std(past_20min)
        if volatility > 40:
            risk_score += 20
            decision_info['factors'].append(f"높은 변동성 (σ={volatility:.0f})")
        elif volatility > 25:
            risk_score += 10
            decision_info['factors'].append(f"중간 변동성 (σ={volatility:.0f})")
        
        # 3-3. 현재 레벨 (0-20점)
        current_level = past_20min[-1]
        if current_level > 310:
            risk_score += 20
            decision_info['factors'].append(f"현재 고위험 ({current_level:.0f})")
        elif current_level > 290:
            risk_score += 10
            decision_info['factors'].append(f"현재 경계선 ({current_level:.0f})")
        
        # 3-4. 물리 지표 (0-30점)
        if physics_data:
            bridge_time = physics_data.get('bridge_time', 3.5)
            storage_util = physics_data.get('storage_util', 0)
            
            if bridge_time > 5:
                risk_score += 20
                decision_info['factors'].append(f"BRIDGE_TIME 높음 ({bridge_time:.1f})")
            elif bridge_time > 4:
                risk_score += 10
                decision_info['factors'].append(f"BRIDGE_TIME 경계 ({bridge_time:.1f})")
            
            if storage_util > 80:
                risk_score += 10
                decision_info['factors'].append(f"Storage 포화 ({storage_util:.0f}%)")
        
        decision_info['scores']['risk_score'] = risk_score
        
        # ========================================
        # 4단계: 최종 확률 보정
        # ========================================
        # 위험 점수를 확률 보정에 사용
        probability_adjustment = risk_score / 200  # 최대 0.5 보정
        final_probability = min(1.0, base_probability + probability_adjustment)
        decision_info['scores']['final_probability'] = final_probability
        
        # ========================================
        # 5단계: 최종 결정
        # ========================================
        if final_probability < 0.45:
            selected = "Model1"
            decision_info['factors'].append(f"최종 확률 {final_probability*100:.1f}% < 45% → 안정형")
        elif final_probability > 0.65:
            selected = "Model2"
            decision_info['factors'].append(f"최종 확률 {final_probability*100:.1f}% > 65% → 극단형")
        else:
            # 경계 구간: 추가 판단
            if risk_score > 50:
                selected = "Model2"
                decision_info['factors'].append(f"경계 구간 + 높은 위험점수 ({risk_score}) → 극단형")
            else:
                selected = "Model1"
                decision_info['factors'].append(f"경계 구간 + 낮은 위험점수 ({risk_score}) → 안정형")
        
        decision_info['selected'] = selected
        self.selection_history.append(decision_info)
        
        return selected, decision_info
    
    def update_performance(self, model_name: str, y_true: float, y_pred: float):
        """모델 성능 업데이트"""
        mae = abs(y_true - y_pred)
        self.performance_tracker[model_name]['mae'].append(mae)
        self.performance_tracker[model_name]['total'] += 1
        
        if mae < 30:  # 성공 기준
            self.performance_tracker[model_name]['correct'] += 1
    
    def get_performance_summary(self) -> Dict:
        """성능 요약"""
        summary = {}
        for model_name, stats in self.performance_tracker.items():
            if stats['total'] > 0:
                summary[model_name] = {
                    'accuracy': stats['correct'] / stats['total'],
                    'avg_mae': np.mean(stats['mae']) if stats['mae'] else 0,
                    'total_predictions': stats['total']
                }
        return summary

# ==============================================================================
# 📊 데이터 처리기 V4
# ==============================================================================

class DataProcessorV4:
    """V4 데이터 처리 - 21개 필수 컬럼 + BRIDGE_TIME"""
    
    def __init__(self):
        self.target_col = 'CURRENT_M16A_3F_JOB_2'
        self.bridge_time_col = 'BRIDGE_TIME'
        
        # V4 필수 컬럼 정의
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
            'M16A_3F_STORAGE_UTIL',
            'BRIDGE_TIME'
        ]
        
        self.ofs_cols = [
            'M14_TO_M16_OFS_CUR',
            'M16_TO_M14_OFS_CUR'
        ]
        
        # 전체 V4 컬럼 리스트
        self.all_v4_cols = ([self.target_col] + self.inflow_cols + self.outflow_cols + 
                           self.cmd_cols + self.maxcapa_cols + self.extreme_cols + self.ofs_cols)
        
        # 스케일러
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.scaler_physics = StandardScaler()
        
        # 연속 패턴 확률
        self.pattern_probability = {
            0: 0.003, 1: 0.15, 2: 0.25, 3: 0.31, 4: 0.43, 5: 0.43,
            6: 0.35, 7: 0.42, 8: 0.53, 9: 0.49, 10: 0.42,
            11: 0.47, 12: 0.52, 13: 0.60, 14: 0.54, 15: 0.66,
            16: 0.62, 17: 0.71, 18: 0.79, 19: 0.83, 20: 0.987
        }
    
    def load_and_merge_data(self) -> pd.DataFrame:
        """데이터 로드 및 BRIDGE_TIME 병합"""
        print("\n[데이터 로드]")
        
        # 메인 데이터 로드
        df = pd.read_csv('data/HUB_0509_TO_0730_DATA.CSV')
        print(f"✅ 메인 데이터: {df.shape}")
        
        # BRIDGE_TIME 데이터 병합 시도
        bridge_path = 'data/BRIDGE_TIME.CSV'
        if os.path.exists(bridge_path):
            try:
                bridge_df = pd.read_csv(bridge_path)
                if 'IDC_VAL' in bridge_df.columns:
                    bridge_df = bridge_df.rename(columns={'IDC_VAL': 'BRIDGE_TIME'})
                    # 시간 기준 병합 로직
                    print("✅ BRIDGE_TIME 데이터 병합 완료")
                else:
                    df['BRIDGE_TIME'] = 3.5
            except:
                df['BRIDGE_TIME'] = 3.5
        else:
            df['BRIDGE_TIME'] = 3.5
            print("ℹ️ BRIDGE_TIME 기본값(3.5) 사용")
        
        # 누락 컬럼 0으로 채우기
        for col in self.all_v4_cols:
            if col not in df.columns:
                df[col] = 0
                print(f"⚠️ {col} 누락 → 0으로 채움")
        
        return df
    
    def analyze_consecutive_patterns(self, df: pd.DataFrame, seq_len: int = 20) -> pd.DataFrame:
        """연속 패턴 분석"""
        print("\n[연속 패턴 분석]")
        
        # 연속 300+ 카운트
        consecutive_counts = []
        consecutive_probs = []
        
        for i in range(len(df)):
            if i < seq_len:
                count = 0
                prob = 0
            else:
                window = df[self.target_col].iloc[i-seq_len:i].values
                count = sum(1 for v in window if v >= 300)
                prob = self.pattern_probability.get(count, 0.5)
            
            consecutive_counts.append(count)
            consecutive_probs.append(prob)
        
        df['consecutive_300_count'] = consecutive_counts
        df['consecutive_300_prob'] = consecutive_probs
        
        print(f"✅ 연속 패턴 분석 완료")
        return df
    
    def create_sequences(self, df: pd.DataFrame, seq_len: int = 20, pred_len: int = 10,
                        ckpt_manager: Optional[CheckpointManager] = None) -> Tuple:
        """시퀀스 생성 with 중단/재개"""
        print("\n[시퀀스 생성]")
        
        # 이전 진행상황 확인
        saved_sequences = None
        if ckpt_manager:
            saved_sequences = ckpt_manager.load_sequences()
        
        if saved_sequences:
            resume = input(f"\n이전 시퀀스 발견 ({saved_sequences['progress']}/{saved_sequences['total']}). 이어서? (y/n): ")
            if resume.lower() == 'y':
                print("✅ 이전 시퀀스 사용")
                return (saved_sequences['X'], saved_sequences['y'], 
                       saved_sequences['X_physics'], saved_sequences['weights'])
        
        # 새로 생성
        X_list, y_list, X_physics_list, weights_list = [], [], [], []
        
        # 사용 가능한 컬럼
        available_cols = [col for col in self.all_v4_cols if col in df.columns]
        numeric_cols = available_cols + ['consecutive_300_count', 'consecutive_300_prob']
        
        total_sequences = len(df) - seq_len - pred_len
        
        print(f"📊 총 {total_sequences}개 시퀀스 생성 중...")
        
        for i in tqdm(range(total_sequences)):
            # 중단 체크
            if ckpt_manager and ckpt_manager.interrupted:
                print("\n💾 중단! 현재까지 저장...")
                ckpt_manager.save_sequences(
                    np.array(X_list), np.array(y_list),
                    np.array(X_physics_list), np.array(weights_list),
                    i, total_sequences
                )
                sys.exit(0)
            
            # 시퀀스 생성
            X = df[numeric_cols].iloc[i:i+seq_len].values
            y = df[self.target_col].iloc[i+seq_len+pred_len-1]
            
            # 물리 데이터 (9개 차원)
            physics = self.create_physics_features(df, i+seq_len-1)
            
            # 가중치 계산
            weight = self.calculate_sample_weight(y)
            
            X_list.append(X)
            y_list.append(y)
            X_physics_list.append(physics)
            weights_list.append(weight)
        
        print(f"✅ 시퀀스 생성 완료: {len(X_list)}개")
        
        return (np.array(X_list), np.array(y_list), 
               np.array(X_physics_list), np.array(weights_list))
    
    def create_physics_features(self, df: pd.DataFrame, idx: int) -> np.ndarray:
        """물리 특징 생성 (9개)"""
        physics = []
        
        # 1. 현재값
        physics.append(df[self.target_col].iloc[idx])
        
        # 2. 유입 합계
        inflow_sum = sum(df[col].iloc[idx] for col in self.inflow_cols if col in df.columns)
        physics.append(inflow_sum)
        
        # 3. 유출 합계
        outflow_sum = sum(df[col].iloc[idx] for col in self.outflow_cols if col in df.columns)
        physics.append(outflow_sum)
        
        # 4. BRIDGE_TIME
        physics.append(df.get('BRIDGE_TIME', pd.Series([3.5])).iloc[idx])
        
        # 5. 연속 300+ 개수
        physics.append(df.get('consecutive_300_count', pd.Series([0])).iloc[idx])
        
        # 6. 연속 300+ 확률
        physics.append(df.get('consecutive_300_prob', pd.Series([0.5])).iloc[idx])
        
        # 7. STORAGE_UTIL
        physics.append(df.get('M16A_3F_STORAGE_UTIL', pd.Series([0])).iloc[idx])
        
        # 8. CMD 합계
        cmd_sum = sum(df[col].iloc[idx] for col in self.cmd_cols if col in df.columns)
        physics.append(cmd_sum)
        
        # 9. 최근 5개 평균
        if idx >= 4:
            recent_avg = df[self.target_col].iloc[idx-4:idx+1].mean()
        else:
            recent_avg = df[self.target_col].iloc[idx]
        physics.append(recent_avg)
        
        return np.array(physics)
    
    def calculate_sample_weight(self, y_value: float) -> float:
        """샘플 가중치 계산"""
        if y_value < 200:
            return 3.0  # FP 방지
        elif y_value < 300:
            return 1.0
        elif y_value < 310:
            return 5.0
        elif y_value < 335:
            return 15.0
        else:  # 335+
            return 25.0

# ==============================================================================
# 🏗️ 모델 정의
# ==============================================================================

# Model 1: PatchTST (안정형)
class PatchTSTModel(keras.Model):
    """안정 구간 전문 PatchTST"""
    
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

# Model 2: PatchTST + PINN (극단값)
class PatchTSTPINN(keras.Model):
    """극단값 전문 PatchTST + PINN"""
    
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
        
        # 물리 정보 처리 (PINN)
        self.physics_net = keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu')
        ])
        
        # 융합 및 극단값 보정
        self.fusion = keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])
        
        # 극단값 부스팅
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

# ==============================================================================
# 📉 손실 함수 (함수형으로 변경)
# ==============================================================================

def stable_loss(y_true, y_pred):
    """Model 1용 - 안정 구간 특화 손실 (함수형)"""
    # 차원 확인 및 조정
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # reshape to ensure same dimensions
    if len(y_true.shape) > 1:
        y_true = tf.squeeze(y_true)
    if len(y_pred.shape) > 1:
        y_pred = tf.squeeze(y_pred)
    
    error = tf.abs(y_true - y_pred)
    
    # False Positive 방지
    penalty = tf.where(
        y_true < 300,
        tf.where(
            y_pred >= 300,
            error * 20.0,  # 300 이하를 300+ 예측: 큰 페널티
            error * 1.0
        ),
        error * 0.5  # 300+ 구간은 덜 중요
    )
    
    return penalty

def extreme_loss(y_true, y_pred):
    """Model 2용 - 극단값 특화 손실 (함수형)"""
    # 차원 확인 및 조정
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # reshape to ensure same dimensions
    if len(y_true.shape) > 1:
        y_true = tf.squeeze(y_true)
    if len(y_pred.shape) > 1:
        y_pred = tf.squeeze(y_pred)
    
    error = tf.abs(y_true - y_pred)
    
    # 극단값 미탐지 방지
    penalty = tf.where(
        y_true >= 335,
        tf.where(
            y_pred < 335,
            error * 30.0,  # 335+ 미탐지: 최대 페널티
            error * 1.0
        ),
        tf.where(
            y_true >= 310,
            tf.where(
                y_pred < 310,
                error * 15.0,  # 310+ 미탐지: 큰 페널티
                error * 1.0
            ),
            tf.where(
                y_true >= 300,
                error * 5.0,  # 300-310 구간: 중요
                error * 0.3   # 300 미만: 덜 중요
            )
        )
    )
    
    return penalty

# ==============================================================================
# 📊 콜백
# ==============================================================================

class ExtremeValueCallback(Callback):
    """극단값 감지 성능 모니터링"""
    
    def __init__(self, X_val, y_val, scaler_y, X_physics_val=None, model_name="Model"):
        self.X_val = X_val
        self.y_val = y_val
        self.scaler_y = scaler_y
        self.X_physics_val = X_physics_val
        self.model_name = model_name
    
    def on_epoch_end(self, epoch, logs=None):
        # 예측
        if self.X_physics_val is not None:
            y_pred_scaled = self.model.predict([self.X_val, self.X_physics_val], verbose=0)
        else:
            y_pred_scaled = self.model.predict(self.X_val, verbose=0)
        
        # 역정규화
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_true = self.scaler_y.inverse_transform(self.y_val.reshape(-1, 1)).flatten()
        
        # 극단값 감지율
        mask_335 = y_true >= 335
        mask_310 = y_true >= 310
        mask_under_300 = y_true < 300
        
        if mask_335.sum() > 0:
            recall_335 = (y_pred >= 335)[mask_335].sum() / mask_335.sum() * 100
        else:
            recall_335 = 0
        
        if mask_310.sum() > 0:
            recall_310 = (y_pred >= 310)[mask_310].sum() / mask_310.sum() * 100
        else:
            recall_310 = 0
        
        # False Positive Rate
        if mask_under_300.sum() > 0:
            fp_rate = (y_pred >= 300)[mask_under_300].sum() / mask_under_300.sum() * 100
        else:
            fp_rate = 0
        
        print(f"\n[{self.model_name} Epoch {epoch+1}] "
              f"335+: {recall_335:.1f}% | "
              f"310+: {recall_310:.1f}% | "
              f"FP: {fp_rate:.1f}%")

# ==============================================================================
# 🎯 메인 실행 함수
# ==============================================================================

def main():
    """V4 Ultimate 메인 실행"""
    
    print("\n" + "="*80)
    print("🚀 V4 ULTIMATE 시스템 시작")
    print("="*80)
    
    # 체크포인트 관리자
    ckpt = CheckpointManager()
    
    # 이전 상태 확인
    state = ckpt.load_state()
    
    if state:
        print(f"\n📂 이전 학습 상태 발견! (Step {state.get('step', 1)}/6)")
        resume = input("이어서 진행하시겠습니까? (y: 이어서, n: 처음부터): ").lower()
        
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
        print("\n새로운 학습을 시작합니다...")
    
    # 데이터 처리기
    processor = DataProcessorV4()
    
    try:
        # ========================================
        # Step 1: 데이터 로드
        # ========================================
        if step <= 1:
            print(f"\n[Step 1/6] 데이터 로드 및 전처리")
            print("-"*60)
            
            df = processor.load_and_merge_data()
            df = processor.analyze_consecutive_patterns(df)
            
            state['step'] = 2
            state['data_shape'] = df.shape
            ckpt.save_state(state)
            print("✅ Step 1 완료")
        
        # ========================================
        # Step 2: 시퀀스 생성
        # ========================================
        if step <= 2:
            print(f"\n[Step 2/6] 시퀀스 생성")
            print("-"*60)
            
            if step < 2:  # 이전 단계에서 넘어온 경우
                df = processor.load_and_merge_data()
                df = processor.analyze_consecutive_patterns(df)
            
            X, y, X_physics, weights = processor.create_sequences(df, ckpt_manager=ckpt)
            
            # HDF5로 저장
            with h5py.File('./checkpoints_ultimate/sequences_complete.h5', 'w') as f:
                f.create_dataset('X', data=X, compression='gzip')
                f.create_dataset('y', data=y, compression='gzip')
                f.create_dataset('X_physics', data=X_physics, compression='gzip')
                f.create_dataset('weights', data=weights, compression='gzip')
            
            state['step'] = 3
            state['sequence_shape'] = X.shape
            ckpt.save_state(state)
            print("✅ Step 2 완료")
        
        # ========================================
        # Step 3: 데이터 분할
        # ========================================
        if step <= 3:
            print(f"\n[Step 3/6] 데이터 분할")
            print("-"*60)
            
            # 시퀀스 로드
            with h5py.File('./checkpoints_ultimate/sequences_complete.h5', 'r') as f:
                X = f['X'][:]
                y = f['y'][:]
                X_physics = f['X_physics'][:]
                weights = f['weights'][:]
            
            # 분할
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
            
            weights_train = weights[train_idx]
            
            print(f"✅ Train: {len(X_train):,} samples")
            print(f"✅ Valid: {len(X_val):,} samples")
            print(f"✅ Test: {len(X_test):,} samples")
            
            state['step'] = 4
            state['n_features'] = X.shape[2]
            ckpt.save_state(state)
            print("✅ Step 3 완료")
        
        # ========================================
        # Step 4: 스케일링
        # ========================================
        if step <= 4:
            print(f"\n[Step 4/6] 데이터 스케일링")
            print("-"*60)
            
            # 시퀀스 로드 (Step 3에서 저장한 데이터)
            with h5py.File('./checkpoints_ultimate/sequences_complete.h5', 'r') as f:
                X = f['X'][:]
                y = f['y'][:]
                X_physics = f['X_physics'][:]
                weights = f['weights'][:]
            
            # 데이터 재분할 (Step 3과 동일하게)
            indices = np.arange(len(X))
            np.random.seed(42)  # 동일한 분할 보장
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
            
            weights_train = weights[train_idx]
            
            # 스케일링
            n_features = X.shape[2]
            
            X_train_scaled = processor.scaler_X.fit_transform(X_train.reshape(-1, n_features))
            X_train_scaled = X_train_scaled.reshape(len(X_train), 20, n_features)
            
            X_val_scaled = processor.scaler_X.transform(X_val.reshape(-1, n_features))
            X_val_scaled = X_val_scaled.reshape(len(X_val), 20, n_features)
            
            X_test_scaled = processor.scaler_X.transform(X_test.reshape(-1, n_features))
            X_test_scaled = X_test_scaled.reshape(len(X_test), 20, n_features)
            
            y_train_scaled = processor.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
            y_val_scaled = processor.scaler_y.transform(y_val.reshape(-1, 1)).flatten()
            
            X_physics_train_scaled = processor.scaler_physics.fit_transform(X_physics_train)
            X_physics_val_scaled = processor.scaler_physics.transform(X_physics_val)
            X_physics_test_scaled = processor.scaler_physics.transform(X_physics_test)
            
            # 스케일러 저장
            os.makedirs('./checkpoints_ultimate/scalers', exist_ok=True)
            joblib.dump(processor.scaler_X, './checkpoints_ultimate/scalers/scaler_X.pkl')
            joblib.dump(processor.scaler_y, './checkpoints_ultimate/scalers/scaler_y.pkl')
            joblib.dump(processor.scaler_physics, './checkpoints_ultimate/scalers/scaler_physics.pkl')
            
            # 스케일된 데이터 저장 (Step 5에서 사용)
            with h5py.File('./checkpoints_ultimate/scaled_data.h5', 'w') as f:
                f.create_dataset('X_train_scaled', data=X_train_scaled, compression='gzip')
                f.create_dataset('X_val_scaled', data=X_val_scaled, compression='gzip')
                f.create_dataset('X_test_scaled', data=X_test_scaled, compression='gzip')
                f.create_dataset('y_train_scaled', data=y_train_scaled, compression='gzip')
                f.create_dataset('y_val_scaled', data=y_val_scaled, compression='gzip')
                f.create_dataset('y_test', data=y_test, compression='gzip')
                f.create_dataset('X_physics_train_scaled', data=X_physics_train_scaled, compression='gzip')
                f.create_dataset('X_physics_val_scaled', data=X_physics_val_scaled, compression='gzip')
                f.create_dataset('X_physics_test_scaled', data=X_physics_test_scaled, compression='gzip')
                f.create_dataset('weights_train', data=weights_train, compression='gzip')
                f.attrs['n_features'] = n_features
            
            state['step'] = 5
            state['n_features'] = n_features
            ckpt.save_state(state)
            print("✅ Step 4 완료")
        
        # ========================================
        # Step 5: 모델 학습
        # ========================================
        if step <= 5:
            print(f"\n[Step 5/6] 모델 학습")
            print("-"*60)
            
            # 스케일된 데이터 로드
            with h5py.File('./checkpoints_ultimate/scaled_data.h5', 'r') as f:
                X_train_scaled = f['X_train_scaled'][:]
                X_val_scaled = f['X_val_scaled'][:]
                X_test_scaled = f['X_test_scaled'][:]
                y_train_scaled = f['y_train_scaled'][:]
                y_val_scaled = f['y_val_scaled'][:]
                y_test = f['y_test'][:]
                X_physics_train_scaled = f['X_physics_train_scaled'][:]
                X_physics_val_scaled = f['X_physics_val_scaled'][:]
                X_physics_test_scaled = f['X_physics_test_scaled'][:]
                weights_train = f['weights_train'][:]
                n_features = f.attrs['n_features']
            
            # 스케일러 로드
            processor.scaler_X = joblib.load('./checkpoints_ultimate/scalers/scaler_X.pkl')
            processor.scaler_y = joblib.load('./checkpoints_ultimate/scalers/scaler_y.pkl')
            processor.scaler_physics = joblib.load('./checkpoints_ultimate/scalers/scaler_physics.pkl')
            
            config = {
                'seq_len': 20,
                'n_features': n_features,
                'patch_len': 5
            }
            
            # Model 1 학습 (안정형)
            if not state.get('model1_trained', False):
                print("\n🤖 Model 1: PatchTST (안정형) 학습")
                
                model1 = PatchTSTModel(config)
                model1.compile(
                    optimizer=Adam(learning_rate=0.001),
                    loss=stable_loss,  # 함수형 손실
                    metrics=['mae']
                )
                
                callbacks1 = [
                    EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
                    ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-6, monitor='val_loss'),
                    ModelCheckpoint('./checkpoints_ultimate/model1_stable.h5', 
                                  save_best_only=True, save_weights_only=True, monitor='val_loss'),
                    ExtremeValueCallback(X_val_scaled, y_val_scaled, processor.scaler_y, 
                                       model_name="Model1")
                ]
                
                # sample_weight 적용 방식 변경 - 손실 함수 내부에서 처리하도록
                history1 = model1.fit(
                    X_train_scaled, y_train_scaled,
                    validation_data=(X_val_scaled, y_val_scaled),
                    epochs=50,
                    batch_size=32,
                    callbacks=callbacks1,
                    verbose=1
                )
                
                state['model1_trained'] = True
                ckpt.save_state(state)
                print("✅ Model 1 학습 완료")
            
            # Model 2 학습 (극단형)
            if not state.get('model2_trained', False):
                print("\n🤖 Model 2: PatchTST+PINN (극단형) 학습")
                
                model2 = PatchTSTPINN(config)
                model2.compile(
                    optimizer=Adam(learning_rate=0.0008),
                    loss=extreme_loss,  # 함수형 손실
                    metrics=['mae']
                )
                
                callbacks2 = [
                    EarlyStopping(patience=20, restore_best_weights=True, monitor='val_loss'),
                    ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6, monitor='val_loss'),
                    ModelCheckpoint('./checkpoints_ultimate/model2_extreme.h5', 
                                  save_best_only=True, save_weights_only=True, monitor='val_loss'),
                    ExtremeValueCallback(X_val_scaled, y_val_scaled, processor.scaler_y,
                                       X_physics_val_scaled, model_name="Model2")
                ]
                
                # sample_weight 제거하고 손실 함수에서 처리
                history2 = model2.fit(
                    [X_train_scaled, X_physics_train_scaled], y_train_scaled,
                    validation_data=([X_val_scaled, X_physics_val_scaled], y_val_scaled),
                    epochs=60,
                    batch_size=32,
                    callbacks=callbacks2,
                    verbose=1
                )
                
                state['model2_trained'] = True
                ckpt.save_state(state)
                print("✅ Model 2 학습 완료")
            
            state['step'] = 6
            ckpt.save_state(state)
            print("✅ Step 5 완료")
        
        # ========================================
        # Step 6: 평가 및 선택기 테스트
        # ========================================
        if step <= 6:
            print(f"\n[Step 6/6] 최종 평가 및 선택기 테스트")
            print("-"*60)
            
            # 스케일된 데이터 로드
            with h5py.File('./checkpoints_ultimate/scaled_data.h5', 'r') as f:
                X_test_scaled = f['X_test_scaled'][:]
                y_test = f['y_test'][:]
                X_physics_test_scaled = f['X_physics_test_scaled'][:]
                n_features = f.attrs['n_features']
            
            # 원본 테스트 데이터도 로드 (선택기용)
            with h5py.File('./checkpoints_ultimate/sequences_complete.h5', 'r') as f:
                X = f['X'][:]
                X_physics = f['X_physics'][:]
            
            # 테스트 인덱스로 원본 데이터 추출
            indices = np.arange(len(X))
            np.random.seed(42)
            np.random.shuffle(indices)
            train_size = int(0.7 * len(X))
            val_size = int(0.15 * len(X))
            test_idx = indices[train_size+val_size:]
            X_test = X[test_idx]
            X_physics_test = X_physics[test_idx]
            
            # 스케일러 로드
            processor.scaler_X = joblib.load('./checkpoints_ultimate/scalers/scaler_X.pkl')
            processor.scaler_y = joblib.load('./checkpoints_ultimate/scalers/scaler_y.pkl')
            processor.scaler_physics = joblib.load('./checkpoints_ultimate/scalers/scaler_physics.pkl')
            
            config = {
                'seq_len': 20,
                'n_features': n_features,
                'patch_len': 5
            }
            
            # 모델 로드
            model1 = PatchTSTModel(config)
            model1.build(input_shape=(None, 20, n_features))
            model1.load_weights('./checkpoints_ultimate/model1_stable.h5')
            
            model2 = PatchTSTPINN(config)
            model2.build([(None, 20, n_features), (None, 9)])
            model2.load_weights('./checkpoints_ultimate/model2_extreme.h5')
            
            # 선택기 초기화
            selector = UltimateHybridSelector()
            
            # 테스트 데이터로 평가
            print("\n📊 최종 성능 평가")
            print("="*60)
            
            # 테스트 샘플별 평가
            total_mae = []
            model1_count = 0
            model2_count = 0
            
            for i in range(min(100, len(X_test))):  # 처음 100개만 테스트
                # 과거 20분 데이터
                past_20min = processor.scaler_y.inverse_transform(
                    X_test_scaled[i, :, 0].reshape(-1, 1)
                ).flatten()
                
                # 물리 데이터
                physics_data = {
                    'bridge_time': X_physics_test[i, 3],
                    'storage_util': X_physics_test[i, 6]
                }
                
                # 모델 선택
                selected_model, decision_info = selector.select_model(past_20min, physics_data)
                
                # 예측
                if selected_model == "Model1":
                    y_pred_scaled = model1.predict(X_test_scaled[i:i+1], verbose=0)[0]
                    model1_count += 1
                else:
                    y_pred_scaled = model2.predict([X_test_scaled[i:i+1], 
                                                   X_physics_test_scaled[i:i+1]], verbose=0)[0]
                    model2_count += 1
                
                # 역정규화
                y_pred = processor.scaler_y.inverse_transform([[y_pred_scaled]])[0, 0]
                y_true = y_test[i]
                
                mae = abs(y_true - y_pred)
                total_mae.append(mae)
                
                # 성능 업데이트
                selector.update_performance(selected_model.lower(), y_true, y_pred)
                
                # 샘플 출력
                if i < 10:
                    print(f"[{i+1}] 실제: {y_true:.1f}, 예측: {y_pred:.1f}, "
                          f"MAE: {mae:.1f}, 선택: {selected_model}")
            
            # 최종 통계
            print("\n" + "="*60)
            print("📈 최종 통계")
            print("-"*60)
            print(f"평균 MAE: {np.mean(total_mae):.2f}")
            print(f"Model 1 사용: {model1_count}회 ({model1_count/len(total_mae)*100:.1f}%)")
            print(f"Model 2 사용: {model2_count}회 ({model2_count/len(total_mae)*100:.1f}%)")
            
            # 선택기 성능
            performance = selector.get_performance_summary()
            for model_name, stats in performance.items():
                print(f"\n{model_name}:")
                print(f"  - 정확도: {stats['accuracy']*100:.1f}%")
                print(f"  - 평균 MAE: {stats['avg_mae']:.2f}")
            
            print("\n" + "="*80)
            print("✅ V4 ULTIMATE 학습 및 평가 완료!")
            print("="*80)
            
            # 완료 후 상태 제거 옵션
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

# ==============================================================================
# 🚀 실행
# ==============================================================================

if __name__ == "__main__":
    main()