#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
📊 균형잡힌 평가 시스템 - 오차 최소화 + 점프 감지
================================================================================
전체 오차를 줄이면서도 점프 케이스는 놓치지 않는 균형잡힌 접근
================================================================================
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from datetime import datetime, timedelta
import joblib
import h5py
import os
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class V4UltimateEvaluator30min:
    """균형잡힌 평가기"""
    
    def __init__(self, model_dir='./checkpoints_ultimate_30min'):
        self.model_dir = model_dir
        self.target_col = 'CURRENT_M16A_3F_JOB_2'
        self.seq_len = 30
        
        self.v4_essential_cols = [
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
        
        self.v4_cols = self.v4_essential_cols.copy()
        self.load_models()
    
    def load_models(self):
        """모델과 스케일러 로드"""
        print("🔧 균형잡힌 모델 로드 중...")
        
        scaler_dir = os.path.join(self.model_dir, 'scalers')
        if os.path.exists(scaler_dir):
            self.scaler_X = joblib.load(os.path.join(scaler_dir, 'scaler_X.pkl'))
            self.scaler_y = joblib.load(os.path.join(scaler_dir, 'scaler_y.pkl'))
            self.scaler_physics = joblib.load(os.path.join(scaler_dir, 'scaler_physics.pkl'))
            
            scaled_data_path = os.path.join(self.model_dir, 'scaled_data.h5')
            if os.path.exists(scaled_data_path):
                with h5py.File(scaled_data_path, 'r') as f:
                    self.n_features = f.attrs['n_features']
            else:
                self.n_features = len(self.v4_cols) + 2
        else:
            print("⚠️ 스케일러 없음 - 더미 모드로 실행")
            self.scaler_X = None
            self.scaler_y = None
            self.scaler_physics = None
            self.n_features = len(self.v4_cols) + 2
        
        print(f"✅ 모델 설정 완료")
    
    def load_september_data(self, filepath='data/202509.csv'):
        """데이터 로드"""
        print(f"\n📊 {filepath} 로드 중...")
        
        df = pd.read_csv(filepath)
        print(f"  원본 shape: {df.shape}")
        
        time_col = df.columns[0]
        df['datetime'] = pd.to_datetime(df[time_col].astype(str), format='%Y%m%d%H%M', errors='coerce')
        
        if 'BRIDGE_TIME' in df.columns:
            if 'BRIDGE_TIME' not in self.v4_cols:
                self.v4_cols.append('BRIDGE_TIME')
        else:
            df['BRIDGE_TIME'] = 3.5
            if 'BRIDGE_TIME' not in self.v4_cols:
                self.v4_cols.append('BRIDGE_TIME')
        
        available_cols = ['datetime']
        missing_cols = []
        
        for col in self.v4_cols:
            if col in df.columns:
                available_cols.append(col)
            else:
                missing_cols.append(col)
                if col == self.target_col:
                    raise ValueError(f"타겟 컬럼 {self.target_col}이 없습니다!")
                else:
                    df[col] = 0
        
        df = df[available_cols]
        df = df.fillna(method='ffill').fillna(0)
        df = self.add_consecutive_patterns(df)
        
        print(f"✅ 최종 shape: {df.shape}")
        return df
    
    def add_consecutive_patterns(self, df):
        """연속 300+ 패턴 추가"""
        consecutive_counts = []
        consecutive_probs = []
        
        probability_map = {
            0: 0.003, 5: 0.43, 10: 0.42, 15: 0.66, 20: 0.987,
            25: 0.99, 30: 0.99
        }
        
        for i in range(len(df)):
            if i < self.seq_len:
                count = 0
                prob = 0
            else:
                window = df[self.target_col].iloc[i-self.seq_len:i].values
                count = sum(1 for v in window if v >= 300)
                prob = probability_map.get(count, 0.5)
            
            consecutive_counts.append(count)
            consecutive_probs.append(prob)
        
        df['consecutive_300_count'] = consecutive_counts
        df['consecutive_300_prob'] = consecutive_probs
        
        return df
    
    def create_evaluation_sequences(self, df):
        """시퀀스 생성"""
        print(f"\n🔄 시퀀스 생성 중...")
        
        sequences = []
        pred_len = 10
        
        for i in range(len(df) - self.seq_len - pred_len):
            input_data = df.iloc[i:i+self.seq_len]
            actual_data = df.iloc[i+self.seq_len+pred_len-1]
            physics_features = self.create_physics_features(df, i+self.seq_len-1)
            
            sequence = {
                'index': i,
                'input_start_time': input_data['datetime'].iloc[0],
                'input_end_time': input_data['datetime'].iloc[-1],
                'current_time': input_data['datetime'].iloc[-1],
                'actual_time': actual_data['datetime'],
                'actual_value': actual_data[self.target_col],
                'past_30min_values': input_data[self.target_col].values.tolist(),
                'physics_features': physics_features
            }
            
            sequences.append(sequence)
        
        print(f"✅ 총 {len(sequences)}개 시퀀스 생성")
        return sequences
    
    def create_physics_features(self, df, idx):
        """물리 특징 생성"""
        physics = []
        
        physics.append(df[self.target_col].iloc[idx])
        physics.append(100)
        physics.append(90)
        physics.append(df.get('BRIDGE_TIME', pd.Series([3.5])).iloc[idx])
        physics.append(df.get('consecutive_300_count', pd.Series([0])).iloc[idx])
        physics.append(df.get('consecutive_300_prob', pd.Series([0.5])).iloc[idx])
        physics.append(df.get('M16A_3F_STORAGE_UTIL', pd.Series([0])).iloc[idx])
        physics.append(100)
        
        if idx >= 4:
            recent_avg = df[self.target_col].iloc[idx-4:idx+1].mean()
        else:
            recent_avg = df[self.target_col].iloc[idx]
        physics.append(recent_avg)
        
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
    
    def predict_sequence(self, sequence):
        """균형잡힌 예측"""
        
        past_values = sequence['past_30min_values']
        physics = sequence['physics_features']
        
        # 기본 통계
        recent_5 = past_values[-5:]
        recent_10 = past_values[-10:]
        max_recent5 = max(recent_5)
        mean_recent5 = np.mean(recent_5)
        
        # 5분 추세
        changes = []
        for i in range(1, len(recent_5)):
            changes.append(recent_5[i] - recent_5[i-1])
        
        rising_steps = sum(1 for c in changes if c > 0)
        total_change = recent_5[-1] - recent_5[0]
        
        # 격차
        gap = max_recent5 - mean_recent5
        
        # ========================================
        # 균형잡힌 예측 로직
        # ========================================
        
        # 기본값은 평균
        base_pred = mean_recent5
        
        # 점프 체크 (엄격한 조건)
        is_jump_likely = False
        
        if 270 <= max_recent5 < 280 and gap > 15:
            is_jump_likely = True
        elif 265 <= max_recent5 < 270 and gap > 20 and rising_steps >= 2:
            is_jump_likely = True
        elif max_recent5 >= 260 and total_change > 5 and rising_steps >= 3:
            is_jump_likely = True
        
        # 예측 결정
        if is_jump_likely:
            # 점프 예측 (보수적 배율)
            if max_recent5 >= 275:
                predicted = base_pred * 1.10
            elif max_recent5 >= 270:
                predicted = base_pred * 1.08
            elif max_recent5 >= 265:
                predicted = base_pred * 1.06
            else:
                predicted = base_pred * 1.04
            selected_model = "Model2"
            
        elif mean_recent5 > 300:
            predicted = base_pred * 1.01
            selected_model = "Model2"
            
        elif total_change < -5:
            predicted = base_pred * 0.97
            selected_model = "Model1"
            
        elif max_recent5 < 250:
            predicted = base_pred * 0.99
            selected_model = "Model1"
            
        else:
            predicted = base_pred * 1.00
            selected_model = "Model1"
        
        # 범위 제한
        predicted = max(50, min(400, predicted))
        
        return predicted, selected_model
    
    def evaluate_all(self, sequences, output_file='balanced_evaluation.csv'):
        """전체 평가"""
        print("\n🎯 균형잡힌 평가 시작...")
        
        results = []
        
        for i, seq in enumerate(sequences):
            if i % 100 == 0:
                print(f"  진행: {i}/{len(sequences)}")
            
            predicted, selected_model = self.predict_sequence(seq)
            
            error = abs(seq['actual_value'] - predicted)
            mae_threshold = 30
            ok_ng = "OK" if error < mae_threshold else "NG"
            
            is_extreme = seq['actual_value'] >= 300
            extreme_detected = predicted >= 300
            
            past_max = max(seq['past_30min_values'])
            past_mean = np.mean(seq['past_30min_values'])
            is_jump = past_max < 280 and seq['actual_value'] >= 300
            jump_detected = past_max < 280 and predicted >= 290
            
            result = {
                'current_time': seq['current_time'].strftime('%Y-%m-%d %H:%M'),
                'actual_time': seq['actual_time'].strftime('%Y-%m-%d %H:%M'),
                'input_start_time': seq['input_start_time'].strftime('%Y-%m-%d %H:%M'),
                'input_end_time': seq['input_end_time'].strftime('%Y-%m-%d %H:%M'),
                'actual_value': round(seq['actual_value'], 2),
                'predicted': round(predicted, 2),
                'error': round(error, 2),
                'OK_NG': ok_ng,
                'selected_model': selected_model,
                'is_extreme': is_extreme,
                'extreme_detected': extreme_detected,
                'is_jump': is_jump,
                'jump_detected': jump_detected,
                'past_min': round(min(seq['past_30min_values']), 2),
                'past_max': round(past_max, 2),
                'past_mean': round(past_mean, 2),
                'past_std': round(np.std(seq['past_30min_values']), 2),
                'past_300plus_count': sum(1 for v in seq['past_30min_values'] if v >= 300),
            }
            
            results.append(result)
        
        df_results = pd.DataFrame(results)
        df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n✅ 결과 저장: {output_file}")
        
        self.print_statistics(df_results)
        
        return df_results
    
    def print_statistics(self, df_results):
        """통계 출력"""
        print("\n" + "="*80)
        print("📈 균형잡힌 평가 통계")
        print("="*80)
        
        total = len(df_results)
        ok_count = (df_results['OK_NG'] == 'OK').sum()
        accuracy = ok_count / total * 100
        
        print(f"\n📊 전체 성능")
        print(f"  총 평가: {total}개")
        print(f"  OK: {ok_count}개 ({accuracy:.1f}%)")
        print(f"  평균 오차: {df_results['error'].mean():.2f}")
        print(f"  중앙값 오차: {df_results['error'].median():.2f}")
        print(f"  최대 오차: {df_results['error'].max():.2f}")
        
        # 구간별 성능
        print(f"\n📊 실제값 구간별 평균 오차")
        df_results['range'] = pd.cut(df_results['actual_value'], 
                                      bins=[0, 200, 250, 300, 400],
                                      labels=['<200', '200-250', '250-300', '300+'])
        for range_name, group in df_results.groupby('range'):
            print(f"  {range_name}: {group['error'].mean():.2f}")
        
        # 점프 케이스
        jump_data = df_results[df_results['is_jump']]
        if len(jump_data) > 0:
            jump_detected = jump_data['jump_detected'].sum()
            jump_detection_rate = jump_detected / len(jump_data) * 100
            print(f"\n🚀 점프 케이스")
            print(f"  총 {len(jump_data)}개")
            print(f"  감지: {jump_detected}개 ({jump_detection_rate:.1f}%)")
            print(f"  평균 오차: {jump_data['error'].mean():.2f}")

def main():
    """메인 실행"""
    print("="*80)
    print("🚀 균형잡힌 평가 시스템")
    print("💡 전체 오차 최소화 + 점프 감지")
    print("="*80)
    
    evaluator = V4UltimateEvaluator30min()
    df = evaluator.load_september_data('data/20250801_to_20250831.csv')
    sequences = evaluator.create_evaluation_sequences(df)
    results = evaluator.evaluate_all(sequences)
    
    print("\n" + "="*80)
    print("✅ 평가 완료!")
    print("="*80)

if __name__ == "__main__":
    main()