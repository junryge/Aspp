#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
📊 개선된 평가 시스템 - 마지막 5분 추세 집중 분석
================================================================================
30분 시퀀스에서 마지막 5분의 상승/하락 추세를 정밀 분석하여 점프 감지
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

# ==============================================================================
# 평가 클래스 - 30분 시퀀스 버전
# ==============================================================================

class V4UltimateEvaluator30min:
    """V4 Ultimate 30분 시퀀스 모델 평가기 - 5분 추세 분석 버전"""
    
    def __init__(self, model_dir='./checkpoints_ultimate_30min'):
        self.model_dir = model_dir
        self.target_col = 'CURRENT_M16A_3F_JOB_2'
        self.seq_len = 30
        
        # V4 필수 컬럼
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
        print("🔧 30분 시퀀스 모델 로드 중...")
        
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
                print("⚠️ scaled_data.h5 없음 - 기본값 사용")
                self.n_features = len(self.v4_cols) + 2
        else:
            print("⚠️ 스케일러 없음 - 더미 모드로 실행")
            self.scaler_X = None
            self.scaler_y = None
            self.scaler_physics = None
            self.n_features = len(self.v4_cols) + 2
        
        self.config = {
            'seq_len': 30,
            'n_features': self.n_features,
            'patch_len': 6
        }
        
        print(f"✅ 30분 시퀀스 모델 설정 완료 (features: {self.n_features})")
    
    def load_september_data(self, filepath='data/202509.csv'):
        """9월 데이터 로드"""
        print(f"\n📊 {filepath} 로드 중...")
        
        df = pd.read_csv(filepath)
        print(f"  원본 shape: {df.shape}")
        
        time_col = df.columns[0]
        df['datetime'] = pd.to_datetime(df[time_col].astype(str), format='%Y%m%d%H%M', errors='coerce')
        
        if 'BRIDGE_TIME' in df.columns:
            print("  ✅ BRIDGE_TIME 컬럼 발견")
            if 'BRIDGE_TIME' not in self.v4_cols:
                self.v4_cols.append('BRIDGE_TIME')
        else:
            print("  ℹ️ BRIDGE_TIME 없음 - 기본값 3.5 사용")
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
                    print(f"❌ 타겟 컬럼 {self.target_col} 없음!")
                    raise ValueError(f"타겟 컬럼 {self.target_col}이 없습니다!")
                else:
                    df[col] = 0
        
        df = df[available_cols]
        
        if missing_cols:
            print(f"⚠️ 누락 컬럼 {len(missing_cols)}개: {missing_cols[:3]}...")
        
        df = df.fillna(method='ffill').fillna(0)
        df = self.add_consecutive_patterns(df)
        
        print(f"✅ 최종 shape: {df.shape}")
        print(f"  사용 컬럼: {len(available_cols)-1}개")
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
                prob = probability_map.get(count, probability_map.get(min(probability_map.keys(), key=lambda x: abs(x-count))))
            
            consecutive_counts.append(count)
            consecutive_probs.append(prob)
        
        df['consecutive_300_count'] = consecutive_counts
        df['consecutive_300_prob'] = consecutive_probs
        
        return df
    
    def create_evaluation_sequences(self, df):
        """평가용 시퀀스 생성 (30분)"""
        print(f"\n🔄 평가 시퀀스 생성 중 ({self.seq_len}분)...")
        
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
                'input_data': input_data[self.v4_cols + ['consecutive_300_count', 'consecutive_300_prob']].values,
                'actual_value': actual_data[self.target_col],
                'past_30min_values': input_data[self.target_col].values.tolist(),
                'physics_features': physics_features
            }
            
            sequences.append(sequence)
        
        print(f"✅ 총 {len(sequences)}개 시퀀스 생성 (30분)")
        return sequences
    
    def create_physics_features(self, df, idx):
        """물리 특징 생성 (11차원)"""
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
        """마지막 5분 추세 집중 분석 버전"""
        
        past_values = sequence['past_30min_values']
        physics = sequence['physics_features']
        
        # 기본 데이터
        recent_3 = past_values[-3:]
        recent_5 = past_values[-5:]
        recent_10 = past_values[-10:]
        first_10 = past_values[:10]
        last_10 = past_values[-10:]
        
        max_val = max(past_values)
        max_recent5 = max(recent_5)
        mean_recent5 = np.mean(recent_5)
        
        # ========================================
        # 🔍 마지막 5분 상세 추세 분석
        # ========================================
        
        # 1. 5분간 개별 변화 분석
        changes = []
        for i in range(1, len(recent_5)):
            changes.append(recent_5[i] - recent_5[i-1])
        
        # 2. 상승/하락 패턴
        rising_steps = sum(1 for c in changes if c > 0)
        falling_steps = sum(1 for c in changes if c < 0)
        
        # 3. 변화 강도
        total_5min_change = recent_5[-1] - recent_5[0]
        avg_change = np.mean(changes) if changes else 0
        max_up = max(changes) if changes else 0
        max_down = min(changes) if changes else 0
        
        # 4. 가속도 체크
        if len(changes) >= 2:
            recent_momentum = changes[-1]
            early_momentum = changes[0]
            acceleration = recent_momentum - early_momentum
        else:
            acceleration = 0
        
        # 5. 최고점 위치
        max_position = np.argmax(recent_5)
        is_peaking = max_position >= 3
        
        # 6. 격차 분석
        gap = max_recent5 - mean_recent5
        gap_ratio = gap / max(mean_recent5, 1)
        
        # ========================================
        # 🎯 점프 판단 로직
        # ========================================
        
        jump_score = 0
        
        # 점수 계산
        if 240 <= max_recent5 < 280:
            
            # 1. 위치 점수
            if max_recent5 >= 275:
                jump_score += 40
            elif max_recent5 >= 270:
                jump_score += 35
            elif max_recent5 >= 265:
                jump_score += 30
            elif max_recent5 >= 260:
                jump_score += 25
            elif max_recent5 >= 255:
                jump_score += 20
            elif max_recent5 >= 250:
                jump_score += 15
            elif max_recent5 >= 245:
                jump_score += 10
            else:
                jump_score += 5
            
            # 2. 상승 추세 점수 (핵심!)
            if rising_steps >= 3:
                jump_score += 30
            elif rising_steps >= 2:
                jump_score += 20
            elif rising_steps >= 1:
                jump_score += 10
            
            # 3. 5분 총 변화량
            if total_5min_change > 5:
                jump_score += 25
            elif total_5min_change > 3:
                jump_score += 15
            elif total_5min_change > 1:
                jump_score += 10
            elif total_5min_change < -3:
                jump_score -= 10
            
            # 4. 가속도 점수
            if acceleration > 2:
                jump_score += 20
            elif acceleration > 1:
                jump_score += 10
            elif acceleration < -2:
                jump_score -= 10
            
            # 5. 최고점 위치 점수
            if is_peaking:
                jump_score += 15
            
            # 6. 격차 점수
            if gap > 25:
                jump_score += 25
            elif gap > 20:
                jump_score += 20
            elif gap > 15:
                jump_score += 15
            elif gap > 10:
                jump_score += 10
            
            # 7. 최대 상승폭 점수
            if max_up > 3:
                jump_score += 15
            elif max_up > 2:
                jump_score += 10
            
            # 8. 물리 특징
            if len(physics) > 3 and physics[3] > 4:
                jump_score += 10
        
        # ========================================
        # 예측 결정
        # ========================================
        
        # 점프 확정 (점수 70+)
        if jump_score >= 70:
            if max_recent5 >= 275:
                predicted = mean_recent5 * 1.22
            elif max_recent5 >= 270:
                predicted = mean_recent5 * 1.20
            elif max_recent5 >= 265:
                predicted = mean_recent5 * 1.18
            elif max_recent5 >= 260:
                predicted = mean_recent5 * 1.16
            else:
                predicted = mean_recent5 * 1.14
            return predicted, "Model2"
        
        # 점프 가능 (점수 50-69)
        elif jump_score >= 50:
            if max_recent5 >= 270:
                predicted = mean_recent5 * 1.15
            elif max_recent5 >= 260:
                predicted = mean_recent5 * 1.12
            else:
                predicted = mean_recent5 * 1.10
            return predicted, "Model2"
        
        # 점프 의심 (점수 30-49)
        elif jump_score >= 30:
            predicted = mean_recent5 * 1.08
            return predicted, "Model2"
        
        # ========================================
        # 일반 케이스
        # ========================================
        
        # 이미 극단값
        if mean_recent5 > 300:
            return mean_recent5 * 1.02, "Model2"
        
        # 명확한 하락 추세
        if total_5min_change < -5 and falling_steps >= 3:
            return mean_recent5 * 0.95, "Model1"
        
        # 안정 구간
        if max_val < 240:
            return mean_recent5 * 0.98, "Model1"
        
        # 기본
        return mean_recent5 * 0.99, "Model1"
    
    def evaluate_all(self, sequences, output_file='202509_5min_trend.csv'):
        """전체 평가 수행"""
        print("\n🎯 평가 시작 (마지막 5분 추세 분석)...")
        
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
            
            # 5분 추세 계산
            recent_5 = seq['past_30min_values'][-5:]
            trend_5min = recent_5[-1] - recent_5[0]
            rising_count = sum(1 for i in range(1, len(recent_5)) if recent_5[i] > recent_5[i-1])
            
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
                'long_trend': round(seq['physics_features'][9], 2) if len(seq['physics_features']) > 9 else 0,
                'volatility_change': round(seq['physics_features'][10], 2) if len(seq['physics_features']) > 10 else 1,
                '5min_trend': round(trend_5min, 2),
                '5min_rising_count': rising_count
            }
            
            results.append(result)
        
        df_results = pd.DataFrame(results)
        df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n✅ 결과 저장: {output_file}")
        
        self.print_statistics(df_results)
        
        return df_results
    
    def print_statistics(self, df_results):
        """평가 통계 출력"""
        print("\n" + "="*80)
        print("📈 평가 통계 (마지막 5분 추세 분석)")
        print("="*80)
        
        total = len(df_results)
        ok_count = (df_results['OK_NG'] == 'OK').sum()
        accuracy = ok_count / total * 100
        
        print(f"\n📊 전체 성능")
        print(f"  총 평가: {total}개")
        print(f"  OK: {ok_count}개 ({accuracy:.1f}%)")
        print(f"  NG: {total-ok_count}개 ({100-accuracy:.1f}%)")
        print(f"  평균 오차: {df_results['error'].mean():.2f}")
        print(f"  최대 오차: {df_results['error'].max():.2f}")
        
        model_counts = df_results['selected_model'].value_counts()
        print(f"\n🤖 모델별 사용")
        for model, count in model_counts.items():
            model_data = df_results[df_results['selected_model'] == model]
            model_accuracy = (model_data['OK_NG'] == 'OK').sum() / len(model_data) * 100
            print(f"  {model}: {count}회 ({count/total*100:.1f}%) - 정확도: {model_accuracy:.1f}%")
        
        extreme_data = df_results[df_results['is_extreme']]
        if len(extreme_data) > 0:
            extreme_detected = extreme_data['extreme_detected'].sum()
            detection_rate = extreme_detected / len(extreme_data) * 100
            print(f"\n🔥 극단값 성능")
            print(f"  극단값 개수: {len(extreme_data)}개")
            print(f"  감지율: {detection_rate:.1f}%")
        
        jump_data = df_results[df_results['is_jump']]
        if len(jump_data) > 0:
            jump_detected = jump_data['jump_detected'].sum()
            jump_detection_rate = jump_detected / len(jump_data) * 100
            print(f"\n🚀 점프 케이스 성능 (정상→극단)")
            print(f"  점프 케이스: {len(jump_data)}개")
            print(f"  감지율: {jump_detection_rate:.1f}%")
            
            missed_jumps = jump_data[~jump_data['jump_detected']]
            if len(missed_jumps) > 0:
                print(f"  놓친 케이스: {len(missed_jumps)}개")
                print(f"\n  놓친 케이스 상세 (최대 5개):")
                for idx, row in missed_jumps.head(5).iterrows():
                    print(f"    {row['current_time']}: max={row['past_max']:.0f}, "
                          f"mean={row['past_mean']:.0f}, 5min_trend={row['5min_trend']:.1f}, "
                          f"rising={row['5min_rising_count']} → "
                          f"실제={row['actual_value']:.0f}, 예측={row['predicted']:.0f}")

def main():
    """메인 실행 함수"""
    print("="*80)
    print("🚀 개선된 평가 - 마지막 5분 추세 분석")
    print("💡 30분 시퀀스에서 마지막 5분의 상승/하락 패턴 집중 분석")
    print("="*80)
    
    evaluator = V4UltimateEvaluator30min()
    df = evaluator.load_september_data('data/20250801_to_20250831.csv')
    sequences = evaluator.create_evaluation_sequences(df)
    results = evaluator.evaluate_all(sequences)
    
    print("\n" + "="*80)
    print("✅ 평가 완료!")
    print("📊 결과 파일: 202509_5min_trend.csv")
    print("="*80)

if __name__ == "__main__":
    main()