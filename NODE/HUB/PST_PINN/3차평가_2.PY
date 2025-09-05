# -*- coding: utf-8 -*-
"""
HUBROOM 예측 모델 평가 시스템
- 2025년 8월 8일 ~ 8월 31일 데이터를 사용한 전체 평가
- 과거 20분 데이터로 10분 후 예측
- 날짜별 예측 성능 추적
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
import os
import pickle
import warnings
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("="*80)
print("🏭 HUBROOM 예측 모델 평가 시스템")
print(f"📅 평가 실행: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("📊 데이터: 2025년 8월 8일 ~ 8월 31일")
print("="*80)

# ========================================
# 모델 정의 (학습 코드와 동일)
# ========================================

class PatchTSTModel(keras.Model):
    """Model 1: PatchTST (전체 균형)"""
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

class PatchTSTPINN(keras.Model):
    """Model 2: PatchTST + PINN (극단값 특화)"""
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
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(16, activation='relu')
        ])
        
        # 융합 및 극단값 보정
        self.fusion = keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])
        
        # 극단값 부스팅 레이어
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
        temporal_features = self.temporal_dense(x)
        
        # 물리 정보 처리
        physics_features = self.physics_net(x_physics)
        
        # 융합
        combined = tf.concat([temporal_features, physics_features], axis=-1)
        output = self.fusion(combined)
        
        # 극단값 부스팅
        boost_factor = self.extreme_boost(combined)
        output = output * (1 + boost_factor * 0.2)
        
        return tf.squeeze(output, axis=-1)

# ========================================
# 데이터 처리 및 예측
# ========================================

class EvaluationSystem:
    def __init__(self):
        self.target_col = 'CURRENT_M16A_3F_JOB_2'
        
        # 물리 법칙용 컬럼
        self.inflow_cols = [
            'M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2',
            'M14A_3F_TO_HUB_JOB2', 'M14B_7F_TO_HUB_JOB2'
        ]
        self.outflow_cols = [
            'M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB',
            'M16A_3F_TO_M14A_3F_JOB', 'M16A_3F_TO_M14B_7F_JOB'
        ]
        
        # 결과 저장용
        self.results = []
        
    def load_scalers(self):
        """스케일러 로드"""
        try:
            self.scaler_X = joblib.load('./scalers/scaler_X.pkl')
            self.scaler_y = joblib.load('./scalers/scaler_y.pkl')
            self.scaler_physics = joblib.load('./scalers/scaler_physics.pkl')
            print("✅ 스케일러 로드 완료")
            return True
        except Exception as e:
            print(f"❌ 스케일러 로드 실패: {e}")
            return False
            
    def load_models(self, n_features):
        """모델 로드"""
        config = {
            'seq_len': 20,
            'n_features': n_features,
            'patch_len': 5
        }
        
        # Model 1 로드
        print("\n📦 Model 1 로드 중...")
        self.model1 = PatchTSTModel(config)
        self.model1.compile(optimizer='adam', loss='mse')
        dummy_input = np.zeros((1, 20, n_features))
        _ = self.model1(dummy_input)
        self.model1.load_weights('./checkpoints/model1_v3.h5')
        print("✅ Model 1 로드 완료")
        
        # Model 2 로드
        print("📦 Model 2 로드 중...")
        self.model2 = PatchTSTPINN(config)
        self.model2.compile(optimizer='adam', loss='mse')
        dummy_seq = np.zeros((1, 20, n_features))
        dummy_physics = np.zeros((1, 3))
        _ = self.model2([dummy_seq, dummy_physics])
        self.model2.load_weights('./checkpoints/model2_v3.h5')
        print("✅ Model 2 로드 완료")
        
    def prepare_data(self, df):
        """데이터 전처리"""
        print("\n📊 데이터 전처리 중...")
        
        # 타임스탬프 처리
        df['timestamp'] = pd.to_datetime(df.iloc[:, 0], format='%Y%m%d%H%M', errors='coerce')
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 결측치 처리
        df = df.fillna(method='ffill').fillna(0)
        
        # 숫자형 컬럼만 선택
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"  전체 데이터: {len(df)} rows")
        print(f"  기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
        print(f"  숫자형 컬럼: {len(numeric_cols)}개")
        
        # 타겟 데이터 분석
        if self.target_col in df.columns:
            target = df[self.target_col]
            print(f"\n📈 타겟({self.target_col}) 분포:")
            print(f"  범위: {target.min():.0f} ~ {target.max():.0f}")
            print(f"  평균: {target.mean():.1f}")
            print(f"  300+: {(target >= 300).sum()}개 ({(target >= 300).sum()/len(target)*100:.2f}%)")
            print(f"  310+: {(target >= 310).sum()}개 ({(target >= 310).sum()/len(target)*100:.2f}%)")
            print(f"  335+: {(target >= 335).sum()}개 ({(target >= 335).sum()/len(target)*100:.2f}%)")
        
        return df, numeric_cols
    
    def create_sequences_for_evaluation(self, df, numeric_cols, seq_len=20, pred_len=10):
        """평가용 시퀀스 생성"""
        sequences = []
        physics_data = []
        timestamps = []
        actual_values = []
        
        # 물리 데이터용 컬럼 확인
        available_inflow = [col for col in self.inflow_cols if col in df.columns]
        available_outflow = [col for col in self.outflow_cols if col in df.columns]
        
        total = len(df) - seq_len - pred_len + 1
        
        if total <= 0:
            print("❌ 데이터가 부족하여 시퀀스를 생성할 수 없습니다.")
            return None, None, None, None
        
        print(f"\n📦 평가용 시퀀스 생성 중... (총 {total}개)")
        
        for i in tqdm(range(total)):
            # 시계열 데이터
            seq = df[numeric_cols].iloc[i:i+seq_len].values
            sequences.append(seq)
            
            # 물리 데이터
            physics = [
                df[self.target_col].iloc[i + seq_len - 1],  # 현재값
                df[available_inflow].iloc[i+seq_len:i+seq_len+pred_len].sum().sum() if available_inflow else 0,
                df[available_outflow].iloc[i+seq_len:i+seq_len+pred_len].sum().sum() if available_outflow else 0
            ]
            physics_data.append(physics)
            
            # 타임스탬프와 실제값
            timestamps.append(df['timestamp'].iloc[i + seq_len + pred_len - 1])
            actual_values.append(df[self.target_col].iloc[i + seq_len + pred_len - 1])
        
        return np.array(sequences), np.array(physics_data), timestamps, np.array(actual_values)
    
    def evaluate_models(self, sequences, physics_data, timestamps, actual_values, numeric_cols):
        """모델 평가"""
        print("\n🤖 모델 예측 중...")
        
        # 데이터 스케일링
        n_features = sequences.shape[2]
        sequences_flat = sequences.reshape(-1, n_features)
        sequences_scaled = self.scaler_X.transform(sequences_flat)
        sequences_scaled = sequences_scaled.reshape(len(sequences), 20, n_features)
        
        physics_scaled = self.scaler_physics.transform(physics_data)
        
        # Model 1 예측
        print("  Model 1 예측 중...")
        pred1_scaled = self.model1.predict(sequences_scaled, verbose=0)
        pred1 = self.scaler_y.inverse_transform(pred1_scaled.reshape(-1, 1)).flatten()
        
        # Model 2 예측
        print("  Model 2 예측 중...")
        pred2_scaled = self.model2.predict([sequences_scaled, physics_scaled], verbose=0)
        pred2 = self.scaler_y.inverse_transform(pred2_scaled.reshape(-1, 1)).flatten()
        
        # 결과 저장
        results_df = pd.DataFrame({
            'timestamp': timestamps,
            'actual': actual_values,
            'pred_model1': pred1,
            'pred_model2': pred2,
            'error_model1': np.abs(actual_values - pred1),
            'error_model2': np.abs(actual_values - pred2),
            'date': pd.to_datetime(timestamps).dt.date,
            'hour': pd.to_datetime(timestamps).dt.hour
        })
        
        # 극단값 플래그 추가
        results_df['is_300_plus'] = results_df['actual'] >= 300
        results_df['is_310_plus'] = results_df['actual'] >= 310
        results_df['is_335_plus'] = results_df['actual'] >= 335
        
        return results_df
    
    def analyze_results(self, results_df):
        """결과 분석 및 출력"""
        print("\n" + "="*80)
        print("📊 평가 결과 분석")
        print("="*80)
        
        # 전체 성능
        print("\n[전체 성능]")
        print(f"데이터 포인트: {len(results_df)}")
        print(f"평가 기간: {results_df['date'].min()} ~ {results_df['date'].max()}")
        print(f"\nModel 1 (PatchTST):")
        print(f"  MAE: {results_df['error_model1'].mean():.2f}")
        print(f"  RMSE: {np.sqrt((results_df['error_model1']**2).mean()):.2f}")
        print(f"\nModel 2 (PatchTST + PINN):")
        print(f"  MAE: {results_df['error_model2'].mean():.2f}")
        print(f"  RMSE: {np.sqrt((results_df['error_model2']**2).mean()):.2f}")
        
        # 구간별 성능
        print("\n[구간별 성능]")
        
        # 저구간 (<200)
        mask_low = results_df['actual'] < 200
        if mask_low.sum() > 0:
            print(f"\n저구간 (<200): {mask_low.sum()}개")
            print(f"  Model 1 MAE: {results_df.loc[mask_low, 'error_model1'].mean():.2f}")
            print(f"  Model 2 MAE: {results_df.loc[mask_low, 'error_model2'].mean():.2f}")
        
        # 정상구간 (200-300)
        mask_normal = (results_df['actual'] >= 200) & (results_df['actual'] < 300)
        if mask_normal.sum() > 0:
            print(f"\n정상구간 (200-300): {mask_normal.sum()}개")
            print(f"  Model 1 MAE: {results_df.loc[mask_normal, 'error_model1'].mean():.2f}")
            print(f"  Model 2 MAE: {results_df.loc[mask_normal, 'error_model2'].mean():.2f}")
        
        # 위험구간 (300+)
        mask_danger = results_df['actual'] >= 300
        if mask_danger.sum() > 0:
            print(f"\n위험구간 (300+): {mask_danger.sum()}개")
            print(f"  Model 1 MAE: {results_df.loc[mask_danger, 'error_model1'].mean():.2f}")
            print(f"  Model 2 MAE: {results_df.loc[mask_danger, 'error_model2'].mean():.2f}")
        
        # 극단값 감지 성능
        print("\n[극단값 감지 성능]")
        
        for threshold in [300, 310, 335]:
            mask = results_df['actual'] >= threshold
            if mask.sum() > 0:
                detected1 = (results_df.loc[mask, 'pred_model1'] >= threshold).sum()
                detected2 = (results_df.loc[mask, 'pred_model2'] >= threshold).sum()
                
                print(f"\n{threshold}+ 감지 ({mask.sum()}개):")
                print(f"  Model 1: {detected1}/{mask.sum()} ({detected1/mask.sum()*100:.1f}%)")
                print(f"  Model 2: {detected2}/{mask.sum()} ({detected2/mask.sum()*100:.1f}%)")
        
        # 날짜별 성능
        print("\n[날짜별 성능]")
        daily_stats = results_df.groupby('date').agg({
            'error_model1': 'mean',
            'error_model2': 'mean',
            'actual': ['min', 'max', 'mean'],
            'is_300_plus': 'sum',
            'is_310_plus': 'sum',
            'is_335_plus': 'sum'
        }).round(2)
        
        print("\n날짜별 MAE:")
        for date in daily_stats.index[:10]:  # 처음 10일만 출력
            mae1 = daily_stats.loc[date, ('error_model1', 'mean')]
            mae2 = daily_stats.loc[date, ('error_model2', 'mean')]
            actual_range = f"{daily_stats.loc[date, ('actual', 'min')]:.0f}-{daily_stats.loc[date, ('actual', 'max')]:.0f}"
            n_extreme = daily_stats.loc[date, ('is_310_plus', 'sum')]
            
            print(f"  {date}: Model1={mae1:.1f}, Model2={mae2:.1f} | 범위:{actual_range} | 310+:{n_extreme:.0f}개")
        
        if len(daily_stats) > 10:
            print(f"  ... (총 {len(daily_stats)}일)")
        
        return results_df
    
    def save_results(self, results_df):
        """결과 저장"""
        print("\n💾 결과 저장 중...")
        
        # CSV 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 상세 결과
        filename_detail = f'evaluation_results_detail_{timestamp}.csv'
        results_df.to_csv(filename_detail, index=False, encoding='utf-8-sig')
        print(f"  상세 결과: {filename_detail}")
        
        # 요약 통계
        summary_stats = pd.DataFrame({
            '지표': ['전체 MAE', '전체 RMSE', '300+ MAE', '310+ MAE', '335+ MAE',
                   '300+ 감지율', '310+ 감지율', '335+ 감지율'],
            'Model1_PatchTST': [
                results_df['error_model1'].mean(),
                np.sqrt((results_df['error_model1']**2).mean()),
                results_df[results_df['is_300_plus']]['error_model1'].mean() if results_df['is_300_plus'].sum() > 0 else None,
                results_df[results_df['is_310_plus']]['error_model1'].mean() if results_df['is_310_plus'].sum() > 0 else None,
                results_df[results_df['is_335_plus']]['error_model1'].mean() if results_df['is_335_plus'].sum() > 0 else None,
                (results_df[results_df['is_300_plus']]['pred_model1'] >= 300).sum() / results_df['is_300_plus'].sum() * 100 if results_df['is_300_plus'].sum() > 0 else None,
                (results_df[results_df['is_310_plus']]['pred_model1'] >= 310).sum() / results_df['is_310_plus'].sum() * 100 if results_df['is_310_plus'].sum() > 0 else None,
                (results_df[results_df['is_335_plus']]['pred_model1'] >= 335).sum() / results_df['is_335_plus'].sum() * 100 if results_df['is_335_plus'].sum() > 0 else None
            ],
            'Model2_PatchTST_PINN': [
                results_df['error_model2'].mean(),
                np.sqrt((results_df['error_model2']**2).mean()),
                results_df[results_df['is_300_plus']]['error_model2'].mean() if results_df['is_300_plus'].sum() > 0 else None,
                results_df[results_df['is_310_plus']]['error_model2'].mean() if results_df['is_310_plus'].sum() > 0 else None,
                results_df[results_df['is_335_plus']]['error_model2'].mean() if results_df['is_335_plus'].sum() > 0 else None,
                (results_df[results_df['is_300_plus']]['pred_model2'] >= 300).sum() / results_df['is_300_plus'].sum() * 100 if results_df['is_300_plus'].sum() > 0 else None,
                (results_df[results_df['is_310_plus']]['pred_model2'] >= 310).sum() / results_df['is_310_plus'].sum() * 100 if results_df['is_310_plus'].sum() > 0 else None,
                (results_df[results_df['is_335_plus']]['pred_model2'] >= 335).sum() / results_df['is_335_plus'].sum() * 100 if results_df['is_335_plus'].sum() > 0 else None
            ]
        })
        
        filename_summary = f'evaluation_results_summary_{timestamp}.csv'
        summary_stats.to_csv(filename_summary, index=False, encoding='utf-8-sig')
        print(f"  요약 통계: {filename_summary}")
        
        # 시각화 저장
        self.create_visualizations(results_df, timestamp)
        
        print("\n✅ 모든 결과 저장 완료!")
        
    def create_visualizations(self, results_df, timestamp):
        """결과 시각화"""
        print("\n📊 시각화 생성 중...")
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # 1. 시계열 예측 비교 (샘플)
        sample_size = min(500, len(results_df))
        sample_df = results_df.iloc[:sample_size]
        
        axes[0, 0].plot(sample_df.index, sample_df['actual'], label='Actual', alpha=0.7, linewidth=1)
        axes[0, 0].plot(sample_df.index, sample_df['pred_model1'], label='Model 1', alpha=0.7, linewidth=1)
        axes[0, 0].plot(sample_df.index, sample_df['pred_model2'], label='Model 2', alpha=0.7, linewidth=1)
        axes[0, 0].axhline(y=310, color='r', linestyle='--', alpha=0.5, label='310 threshold')
        axes[0, 0].set_title(f'Predictions vs Actual (First {sample_size} points)')
        axes[0, 0].set_xlabel('Index')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 산점도 - Model 1
        axes[0, 1].scatter(results_df['actual'], results_df['pred_model1'], alpha=0.5, s=1)
        axes[0, 1].plot([results_df['actual'].min(), results_df['actual'].max()], 
                       [results_df['actual'].min(), results_df['actual'].max()], 
                       'r--', alpha=0.5)
        axes[0, 1].set_title('Model 1: Predicted vs Actual')
        axes[0, 1].set_xlabel('Actual')
        axes[0, 1].set_ylabel('Predicted')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 산점도 - Model 2
        axes[1, 0].scatter(results_df['actual'], results_df['pred_model2'], alpha=0.5, s=1)
        axes[1, 0].plot([results_df['actual'].min(), results_df['actual'].max()], 
                       [results_df['actual'].min(), results_df['actual'].max()], 
                       'r--', alpha=0.5)
        axes[1, 0].set_title('Model 2: Predicted vs Actual')
        axes[1, 0].set_xlabel('Actual')
        axes[1, 0].set_ylabel('Predicted')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 오차 분포
        axes[1, 1].hist(results_df['error_model1'], bins=50, alpha=0.5, label='Model 1', density=True)
        axes[1, 1].hist(results_df['error_model2'], bins=50, alpha=0.5, label='Model 2', density=True)
        axes[1, 1].set_title('Error Distribution')
        axes[1, 1].set_xlabel('Absolute Error')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 5. 날짜별 MAE
        daily_mae = results_df.groupby('date')[['error_model1', 'error_model2']].mean()
        axes[2, 0].plot(daily_mae.index, daily_mae['error_model1'], marker='o', label='Model 1')
        axes[2, 0].plot(daily_mae.index, daily_mae['error_model2'], marker='s', label='Model 2')
        axes[2, 0].set_title('Daily MAE')
        axes[2, 0].set_xlabel('Date')
        axes[2, 0].set_ylabel('MAE')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].tick_params(axis='x', rotation=45)
        
        # 6. 극단값 감지 성능
        thresholds = [300, 310, 335]
        detect_rates_m1 = []
        detect_rates_m2 = []
        
        for threshold in thresholds:
            mask = results_df['actual'] >= threshold
            if mask.sum() > 0:
                rate1 = (results_df.loc[mask, 'pred_model1'] >= threshold).sum() / mask.sum() * 100
                rate2 = (results_df.loc[mask, 'pred_model2'] >= threshold).sum() / mask.sum() * 100
                detect_rates_m1.append(rate1)
                detect_rates_m2.append(rate2)
            else:
                detect_rates_m1.append(0)
                detect_rates_m2.append(0)
        
        x_pos = np.arange(len(thresholds))
        width = 0.35
        
        axes[2, 1].bar(x_pos - width/2, detect_rates_m1, width, label='Model 1')
        axes[2, 1].bar(x_pos + width/2, detect_rates_m2, width, label='Model 2')
        axes[2, 1].set_title('Extreme Value Detection Rate')
        axes[2, 1].set_xlabel('Threshold')
        axes[2, 1].set_ylabel('Detection Rate (%)')
        axes[2, 1].set_xticks(x_pos)
        axes[2, 1].set_xticklabels([f'{t}+' for t in thresholds])
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename_plot = f'evaluation_plots_{timestamp}.png'
        plt.savefig(filename_plot, dpi=150, bbox_inches='tight')
        print(f"  시각화 저장: {filename_plot}")
        plt.close()

# ========================================
# 메인 실행
# ========================================

def main():
    evaluator = EvaluationSystem()
    
    # 1. 스케일러 로드
    if not evaluator.load_scalers():
        print("❌ 평가 중단: 스케일러가 필요합니다.")
        return
    
    # 2. 데이터 로드
    print("\n📂 데이터 로드 중...")
    data_file = '20250808_TO_20250831.csv'
    
    if not os.path.exists(data_file):
        print(f"❌ 파일을 찾을 수 없습니다: {data_file}")
        # 다른 가능한 파일명들 시도
        alternative_files = [
            'data/20250808_TO_20250831.csv',
            './20250808_TO_20250831.CSV',
            'data/20250808_TO_20250831.CSV'
        ]
        
        for alt_file in alternative_files:
            if os.path.exists(alt_file):
                data_file = alt_file
                print(f"✅ 대체 파일 발견: {alt_file}")
                break
        else:
            print("❌ 평가할 데이터 파일을 찾을 수 없습니다.")
            return
    
    df = pd.read_csv(data_file)
    print(f"✅ 데이터 로드 완료: {df.shape}")
    
    # 3. 데이터 전처리
    df, numeric_cols = evaluator.prepare_data(df)
    
    # 4. 모델 로드
    n_features = len(numeric_cols)
    evaluator.load_models(n_features)
    
    # 5. 시퀀스 생성
    sequences, physics_data, timestamps, actual_values = evaluator.create_sequences_for_evaluation(
        df, numeric_cols
    )
    
    if sequences is None:
        print("❌ 시퀀스 생성 실패")
        return
    
    # 6. 모델 평가
    results_df = evaluator.evaluate_models(
        sequences, physics_data, timestamps, actual_values, numeric_cols
    )
    
    # 7. 결과 분석
    results_df = evaluator.analyze_results(results_df)
    
    # 8. 결과 저장
    evaluator.save_results(results_df)
    
    print("\n" + "="*80)
    print("✅ 평가 완료!")
    print("="*80)

if __name__ == "__main__":
    main()