# -*- coding: utf-8 -*-
"""
================================================================================
📊 V4 ULTIMATE 평가 시스템 - 2025년 8월 데이터 평가
================================================================================
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import joblib
import h5py
import os
from datetime import datetime, timedelta
from tqdm import tqdm

print("="*80)
print("📊 V4 ULTIMATE 평가 시스템")
print("🎯 2025년 8월 데이터 (학습되지 않은 데이터) 평가")
print("="*80)

# 데이터 프로세서 로드 (기존 코드에서 가져옴)
from your_training_script import IntegratedDataProcessor, IntegratedV4Model

class V4Evaluator:
    """V4 모델 평가 및 결과 생성"""
    
    def __init__(self):
        self.processor = IntegratedDataProcessor()
        self.model = None
        self.evaluation_results = []
        
    def load_august_data(self):
        """8월 데이터만 로드"""
        print("\n📊 8월 평가 데이터 로드 중...")
        
        # 전체 데이터 로드
        df_raw = pd.read_csv('data/HUB_0509_TO_0807_DATA.CSV')
        
        # 시간 처리
        time_col = df_raw.columns[0]
        df_raw['timestamp'] = pd.to_datetime(df_raw[time_col], format='%Y%m%d%H%M', errors='coerce')
        
        # 8월 데이터만 필터링
        df_august = df_raw[(df_raw['timestamp'] >= '2025-08-01') & 
                          (df_raw['timestamp'] < '2025-09-01')].copy()
        
        print(f"✅ 8월 데이터: {len(df_august)} 행")
        
        # 필수 컬럼 처리 (processor 사용)
        available_cols = ['timestamp']
        for col in self.processor.v4_essential_cols:
            if col in df_august.columns:
                available_cols.append(col)
            else:
                df_august[col] = 0
        
        # BRIDGE_TIME 처리
        df_august['BRIDGE_TIME'] = 3.5  # 간단히 기본값 사용
        
        # 고급 특징 생성
        df_august = self.processor.create_advanced_features(df_august)
        
        return df_august
    
    def load_trained_model(self):
        """학습된 모델 로드"""
        print("\n🤖 학습된 모델 로드 중...")
        
        # 스케일러 로드
        self.processor.scaler_X = joblib.load('./checkpoints_integrated_30min/scalers/scaler_X.pkl')
        self.processor.scaler_y = joblib.load('./checkpoints_integrated_30min/scalers/scaler_y.pkl')
        self.processor.scaler_physics = joblib.load('./checkpoints_integrated_30min/scalers/scaler_physics.pkl')
        self.processor.scaler_features = joblib.load('./checkpoints_integrated_30min/scalers/scaler_features.pkl')
        
        # 모델 설정
        with h5py.File('./checkpoints_integrated_30min/scaled_integrated_data.h5', 'r') as f:
            n_features = f.attrs['n_features']
        
        config = {
            'seq_len': 30,
            'n_features': n_features,
            'patch_len': 6
        }
        
        # 통합 모델 로드
        self.model = IntegratedV4Model(config)
        
        # 가중치 로드
        dummy_input = [
            np.zeros((1, 30, n_features)),
            np.zeros((1, 11)),
            np.zeros((1, 15))
        ]
        _ = self.model(dummy_input, training=False)
        
        self.model.load_weights('./checkpoints_integrated_30min/models/integrated_model.weights.h5')
        
        print("✅ 모델 로드 완료")
        
    def evaluate_sequences(self, df):
        """시퀀스별 평가 수행"""
        print("\n🔍 시퀀스별 평가 시작...")
        
        results = []
        
        # 사용할 컬럼들
        all_cols = self.processor.v4_essential_cols + ['BRIDGE_TIME'] + [
            'consecutive_300_count', 'consecutive_300_prob', 'long_trend',
            'mid_trend', 'acceleration', 'volatility_change', 'jump_risk', 'extreme_risk'
        ]
        available_cols = [col for col in all_cols if col in df.columns]
        
        # timestamp 저장
        timestamps = df['timestamp'].values
        
        # timestamp 제거하고 처리
        df_process = df.drop('timestamp', axis=1)
        
        total_sequences = len(df_process) - 30 - 10
        
        for i in tqdm(range(total_sequences)):
            # 30분 시퀀스
            X = df_process[available_cols].iloc[i:i+30].values
            
            # 10분 후 실제값
            y_true = df_process[self.processor.target_col].iloc[i+30+9]
            
            # 시퀀스 정보
            seq_start_time = timestamps[i]
            seq_end_time = timestamps[i+29]
            target_time = timestamps[i+30+9]
            
            # 시퀀스 통계
            seq_values = df_process[self.processor.target_col].iloc[i:i+30].values
            seq_max = np.max(seq_values)
            seq_min = np.min(seq_values)
            seq_mean = np.mean(seq_values)
            seq_std = np.std(seq_values)
            
            # 최근 5분 통계
            recent_5 = seq_values[-5:]
            recent_5_max = np.max(recent_5)
            recent_5_mean = np.mean(recent_5)
            
            # 30분 구간별 분석
            first_10 = seq_values[:10]
            middle_10 = seq_values[10:20]
            last_10 = seq_values[20:30]
            
            first_10_mean = np.mean(first_10)
            middle_10_mean = np.mean(middle_10)
            last_10_mean = np.mean(last_10)
            
            # 특징 추세
            long_trend = last_10_mean - first_10_mean
            acceleration = (last_10_mean - middle_10_mean) - (middle_10_mean - first_10_mean)
            
            # 점프/극단값 판단
            is_jump_case = (seq_max < 280) and (y_true >= 300)
            is_extreme = y_true >= 300
            is_277_zone = 275 <= recent_5_max <= 279
            
            # 예측 난이도 점수
            difficulty_score = 0
            if is_jump_case:
                difficulty_score += 50
            if is_277_zone:
                difficulty_score += 30
            if seq_std > 50:
                difficulty_score += 20
            
            # 모델 예측
            X_scaled = self.processor.scaler_X.transform(X.reshape(-1, X.shape[1]))
            X_scaled = X_scaled.reshape(1, 30, X.shape[1])
            
            # 물리 특징
            physics = self.processor.create_physics_features(df_process, i+29)
            physics_scaled = self.processor.scaler_physics.transform(physics.reshape(1, -1))
            
            # 미세조정 특징
            features = self.processor.create_fine_tuning_features(df_process, i+29)
            features_scaled = self.processor.scaler_features.transform(features.reshape(1, -1))
            
            # 예측
            y_pred_scaled = self.model.predict([X_scaled, physics_scaled, features_scaled], verbose=0)
            y_pred = self.processor.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()[0]
            
            # 오차
            mae = abs(y_true - y_pred)
            
            # Base 모델별 예측 (참고용)
            pred1_scaled = self.model.model1.predict(X_scaled, verbose=0)
            pred1 = self.processor.scaler_y.inverse_transform(pred1_scaled.reshape(-1, 1)).flatten()[0]
            
            pred2_scaled = self.model.model2.predict([X_scaled, physics_scaled], verbose=0)
            pred2 = self.processor.scaler_y.inverse_transform(pred2_scaled.reshape(-1, 1)).flatten()[0]
            
            # 어느 모델이 선택되었나 판단
            model_selected = "Model1" if abs(y_pred - pred1) < abs(y_pred - pred2) else "Model2"
            
            # 연속 300+ 카운트
            consecutive_300 = sum(1 for v in seq_values if v >= 300)
            
            # 결과 저장
            result = {
                '날짜시간': target_time.strftime('%Y-%m-%d %H:%M'),
                '타겟날짜': target_time.strftime('%Y-%m-%d'),
                '시퀀스시작시간': seq_start_time.strftime('%H:%M'),
                '시퀀스종료시간': seq_end_time.strftime('%H:%M'),
                'JUMP_277구간': is_277_zone,
                '시퀀스MAX': seq_max,
                '시퀀스MIN': seq_min,
                '시퀀스평균': seq_mean,
                '시퀀스표준편차': seq_std,
                '최근5분MAX': recent_5_max,
                '최근5분평균': recent_5_mean,
                '첫10분평균': first_10_mean,
                '중간10분평균': middle_10_mean,
                '마지막10분평균': last_10_mean,
                '장기추세': long_trend,
                '가속도': acceleration,
                '연속300카운트': consecutive_300,
                '실제값': y_true,
                '예측값': y_pred,
                'Model1예측': pred1,
                'Model2예측': pred2,
                '선택모델': model_selected,
                'MAE': mae,
                '300이상여부': is_extreme,
                '점프케이스': is_jump_case,
                '예측난이도': difficulty_score,
                '예측성공': (is_extreme and y_pred >= 300) or (not is_extreme and y_pred < 300)
            }
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def analyze_results(self, df_results):
        """결과 분석 및 요약"""
        print("\n📊 평가 결과 분석")
        print("="*60)
        
        # 전체 성능
        mae = df_results['MAE'].mean()
        print(f"전체 MAE: {mae:.2f}")
        
        # 극단값 성능
        extreme_mask = df_results['300이상여부']
        if extreme_mask.sum() > 0:
            extreme_recall = (df_results[extreme_mask]['예측값'] >= 300).sum() / extreme_mask.sum() * 100
            print(f"300+ 감지율: {extreme_recall:.1f}%")
        
        # 점프 케이스 성능
        jump_mask = df_results['점프케이스']
        if jump_mask.sum() > 0:
            jump_detection = (df_results[jump_mask]['예측값'] >= 290).sum() / jump_mask.sum() * 100
            print(f"점프 감지율: {jump_detection:.1f}% ({jump_mask.sum()}개 케이스)")
        
        # 277 구간 성능
        zone_277_mask = df_results['JUMP_277구간']
        if zone_277_mask.sum() > 0:
            zone_277_mae = df_results[zone_277_mask]['MAE'].mean()
            print(f"277구간 MAE: {zone_277_mae:.2f} ({zone_277_mask.sum()}개 케이스)")
        
        # False Positive
        stable_mask = ~df_results['300이상여부']
        if stable_mask.sum() > 0:
            fp_rate = (df_results[stable_mask]['예측값'] >= 300).sum() / stable_mask.sum() * 100
            print(f"False Positive율: {fp_rate:.1f}%")
        
        # 난이도별 성능
        print("\n난이도별 MAE:")
        for difficulty in [0, 20, 50, 80, 100]:
            mask = df_results['예측난이도'] >= difficulty
            if mask.sum() > 0:
                print(f"  난이도 {difficulty}+: {df_results[mask]['MAE'].mean():.2f} ({mask.sum()}개)")
        
        # 모델 선택 분포
        model_dist = df_results['선택모델'].value_counts()
        print(f"\n모델 선택 분포:")
        for model, count in model_dist.items():
            print(f"  {model}: {count} ({count/len(df_results)*100:.1f}%)")
    
    def save_evaluation(self, df_results):
        """평가 결과 저장"""
        output_path = 'evaluation_august_2025.csv'
        df_results.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 평가 결과 저장: {output_path}")
        print(f"   총 {len(df_results)}개 시퀀스 평가 완료")
        
        # 추가 분석 파일 생성
        summary_path = 'evaluation_august_summary.txt'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("V4 ULTIMATE 모델 평가 요약\n")
            f.write("="*60 + "\n")
            f.write(f"평가 기간: 2025년 8월\n")
            f.write(f"총 시퀀스: {len(df_results)}개\n")
            f.write(f"평균 MAE: {df_results['MAE'].mean():.2f}\n")
            f.write(f"최대 오차: {df_results['MAE'].max():.2f}\n")
            f.write(f"최소 오차: {df_results['MAE'].min():.2f}\n")
            
        print(f"   요약 파일: {summary_path}")

def main():
    """메인 실행"""
    evaluator = V4Evaluator()
    
    # 1. 데이터 로드
    df_august = evaluator.load_august_data()
    
    # 2. 모델 로드
    evaluator.load_trained_model()
    
    # 3. 평가 수행
    df_results = evaluator.evaluate_sequences(df_august)
    
    # 4. 결과 분석
    evaluator.analyze_results(df_results)
    
    # 5. 결과 저장
    evaluator.save_evaluation(df_results)
    
    print("\n✅ 평가 완료!")

if __name__ == "__main__":
    main()