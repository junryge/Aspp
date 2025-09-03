# -*- coding: utf-8 -*-
"""
HUBROOM 극단값 예측 평가 시스템
ExtremePatchTST & ImprovedPINN 모델 평가
2025년 9월 데이터 대상
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("="*80)
print("🏭 HUBROOM 극단값 예측 평가 시스템")
print("🎯 모델: ExtremePatchTST & ImprovedPINN")
print("="*80)

# ========================================
# 모델 재구성
# ========================================

class ExtremePatchTST(keras.Model):
    """극단값 예측 특화 PatchTST"""
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

class ImprovedPINN(keras.Model):
    """물리 법칙 기반 예측 모델"""
    def __init__(self, config):
        super().__init__()
        
        # LSTM for 시계열
        self.lstm = layers.LSTM(64, return_sequences=False)
        
        # 물리 정보 처리
        self.physics_net = keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(16, activation='relu')
        ])
        
        # 융합
        self.fusion = keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        
    def call(self, inputs, training=False):
        x_seq, x_physics = inputs
        
        # 시계열 처리
        seq_features = self.lstm(x_seq)
        
        # 물리 정보 처리
        physics_features = self.physics_net(x_physics)
        
        # 결합
        combined = tf.concat([seq_features, physics_features], axis=-1)
        
        # 출력
        output = self.fusion(combined)
        
        return tf.squeeze(output, axis=-1)

# ========================================
# 평가 클래스
# ========================================

class ExtremePredictionEvaluator:
    def __init__(self, data_path='data/202509.csv'):
        self.data_path = data_path
        self.seq_len = 20
        self.pred_len = 10
        self.target_col = 'CURRENT_M16A_3F_JOB_2'
        
        # 임계값
        self.thresholds = {
            'warning': 300,
            'critical': 310,
            'extreme': 350
        }
        
        # 모델 경로
        self.model1_path = './checkpoints/model1_final.h5'
        self.model2_path = './checkpoints/model2_final.h5'
        
        # 스케일러 경로
        self.scaler_dir = './scalers'
        
        # 물리 컬럼
        self.inflow_cols = [
            'M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2',
            'M14A_3F_TO_HUB_JOB2', 'M14B_7F_TO_HUB_JOB2'
        ]
        self.outflow_cols = [
            'M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB',
            'M16A_3F_TO_M14A_3F_JOB', 'M16A_3F_TO_M14B_7F_JOB'
        ]
        
    def load_scalers(self):
        """스케일러 로드"""
        try:
            self.scaler_X = joblib.load(f'{self.scaler_dir}/scaler_X.pkl')
            self.scaler_y = joblib.load(f'{self.scaler_dir}/scaler_y.pkl')
            self.scaler_physics = joblib.load(f'{self.scaler_dir}/scaler_physics.pkl')
            print("✅ 스케일러 로드 완료")
            return True
        except Exception as e:
            print(f"❌ 스케일러 로드 실패: {e}")
            return False
    
    def prepare_data(self):
        """데이터 준비"""
        print("\n📂 데이터 로드 중...")
        
        # CSV 로드
        df = pd.read_csv(self.data_path)
        print(f"  데이터 크기: {df.shape}")
        
        # 시간 처리
        time_col = df.columns[0]
        df['timestamp'] = pd.to_datetime(df[time_col], format='%Y%m%d%H%M', errors='coerce')
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 전처리
        df = df.fillna(method='ffill').fillna(0)
        
        # 타겟 분석
        target = df[self.target_col]
        print(f"\n📊 타겟 변수 분석:")
        print(f"  범위: {target.min():.0f} ~ {target.max():.0f}")
        print(f"  평균: {target.mean():.1f}")
        print(f"  310+ 비율: {(target >= 310).sum() / len(target) * 100:.2f}%")
        print(f"  350+ 비율: {(target >= 350).sum() / len(target) * 100:.2f}%")
        
        return df
    
    def create_test_sequences(self, df):
        """테스트 시퀀스 생성"""
        print("\n🔄 시퀀스 생성 중...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        n_features = len(numeric_cols)
        
        X, y, X_physics = [], [], []
        timestamps = []
        
        # 물리 컬럼 확인
        available_inflow = [col for col in self.inflow_cols if col in df.columns]
        available_outflow = [col for col in self.outflow_cols if col in df.columns]
        
        total = len(df) - self.seq_len - self.pred_len + 1
        
        for i in range(total):
            # 시계열 데이터
            X.append(df[numeric_cols].iloc[i:i+self.seq_len].values)
            
            # 타겟 (10분 후)
            y_val = df[self.target_col].iloc[i + self.seq_len + self.pred_len - 1]
            y.append(y_val)
            
            # 물리 데이터
            current_val = df[self.target_col].iloc[i + self.seq_len - 1]
            inflow = df[available_inflow].iloc[i+self.seq_len:i+self.seq_len+self.pred_len].sum().sum() if available_inflow else 0
            outflow = df[available_outflow].iloc[i+self.seq_len:i+self.seq_len+self.pred_len].sum().sum() if available_outflow else 0
            
            X_physics.append([current_val, inflow, outflow])
            
            # 타임스탬프
            timestamps.append(df['timestamp'].iloc[i + self.seq_len - 1])
        
        print(f"  생성된 시퀀스: {len(X)}개")
        
        return np.array(X), np.array(y), np.array(X_physics), timestamps, n_features
    
    def load_models(self, n_features):
        """모델 로드"""
        print("\n🤖 모델 로드 중...")
        
        config = {
            'seq_len': 20,
            'n_features': n_features,
            'patch_len': 5
        }
        
        models = {}
        
        # ExtremePatchTST
        if os.path.exists(self.model1_path):
            try:
                model1 = ExtremePatchTST(config)
                dummy = np.zeros((1, 20, n_features))
                _ = model1(dummy)
                model1.load_weights(self.model1_path)
                models['ExtremePatchTST'] = model1
                print("  ✅ ExtremePatchTST 로드 완료")
            except Exception as e:
                print(f"  ❌ ExtremePatchTST 로드 실패: {e}")
        
        # ImprovedPINN
        if os.path.exists(self.model2_path):
            try:
                model2 = ImprovedPINN(config)
                dummy_seq = np.zeros((1, 20, n_features))
                dummy_physics = np.zeros((1, 3))
                _ = model2([dummy_seq, dummy_physics])
                model2.load_weights(self.model2_path)
                models['ImprovedPINN'] = model2
                print("  ✅ ImprovedPINN 로드 완료")
            except Exception as e:
                print(f"  ❌ ImprovedPINN 로드 실패: {e}")
        
        return models
    
    def evaluate_model(self, model, model_name, X, y, X_physics=None):
        """모델 평가"""
        print(f"\n{'='*60}")
        print(f"📊 {model_name} 평가")
        print('='*60)
        
        # 스케일링
        n_samples, seq_len, n_features = X.shape
        X_scaled = self.scaler_X.transform(X.reshape(-1, n_features)).reshape(n_samples, seq_len, n_features)
        y_scaled = self.scaler_y.transform(y.reshape(-1, 1)).flatten()
        
        # 예측
        if model_name == 'ImprovedPINN':
            X_physics_scaled = self.scaler_physics.transform(X_physics)
            y_pred_scaled = model.predict([X_scaled, X_physics_scaled], batch_size=32, verbose=0)
        else:
            y_pred_scaled = model.predict(X_scaled, batch_size=32, verbose=0)
        
        # 역변환
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        # 메트릭 계산
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        
        print(f"\n📈 전체 성능:")
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R²: {r2:.4f}")
        
        # 임계값별 분석
        print(f"\n🎯 임계값별 분석:")
        for name, threshold in self.thresholds.items():
            mask = y >= threshold
            if mask.sum() > 0:
                mae_th = mean_absolute_error(y[mask], y_pred[mask])
                detected = (y_pred >= threshold)[mask].sum()
                detection_rate = detected / mask.sum() * 100
                
                print(f"\n  {name.upper()} ({threshold}+):")
                print(f"    실제: {mask.sum()}개")
                print(f"    감지: {detected}개 ({detection_rate:.1f}%)")
                print(f"    MAE: {mae_th:.2f}")
        
        # 혼동 행렬 (310 기준)
        y_true_binary = (y >= 310).astype(int)
        y_pred_binary = (y_pred >= 310).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n🎯 310+ 이진 분류 성능:")
        print(f"  Precision: {precision:.2%}")
        print(f"  Recall: {recall:.2%}")
        print(f"  F1-Score: {f1:.2%}")
        
        return {
            'y_true': y,
            'y_pred': y_pred,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def visualize_results(self, results, timestamps):
        """결과 시각화"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('HUBROOM 극단값 예측 모델 평가', fontsize=16)
        
        for idx, (model_name, result) in enumerate(results.items()):
            row = idx // 3
            col = idx % 3
            
            if row < 2 and col < 3:
                ax = axes[row, col] if len(results) > 3 else axes[col]
                
                # 시계열 플롯
                sample_size = min(500, len(result['y_true']))
                ax.plot(result['y_true'][:sample_size], label='실제값', alpha=0.7)
                ax.plot(result['y_pred'][:sample_size], label='예측값', alpha=0.7)
                ax.axhline(y=310, color='red', linestyle='--', alpha=0.5, label='임계값(310)')
                ax.set_title(f'{model_name}\nMAE={result["mae"]:.2f}, F1={result["f1"]:.2%}')
                ax.set_xlabel('시간 인덱스')
                ax.set_ylabel('HUBROOM 반송량')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # 모델 비교 (오른쪽 하단)
        ax_comp = axes[1, 2]
        metrics = ['MAE', 'RMSE', 'Precision', 'Recall', 'F1']
        model_names = list(results.keys())
        
        x = np.arange(len(metrics))
        width = 0.35
        
        for i, model_name in enumerate(model_names):
            values = [
                results[model_name]['mae'],
                results[model_name]['rmse'],
                results[model_name]['precision'] * 100,
                results[model_name]['recall'] * 100,
                results[model_name]['f1'] * 100
            ]
            ax_comp.bar(x + i * width, values, width, label=model_name)
        
        ax_comp.set_xlabel('메트릭')
        ax_comp.set_ylabel('값')
        ax_comp.set_title('모델 성능 비교')
        ax_comp.set_xticks(x + width / 2)
        ax_comp.set_xticklabels(metrics)
        ax_comp.legend()
        ax_comp.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('extreme_model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, results, timestamps):
        """결과 저장"""
        print("\n💾 결과 저장 중...")
        
        df_results = pd.DataFrame({'timestamp': timestamps})
        
        for model_name, result in results.items():
            df_results[f'actual'] = result['y_true']
            df_results[f'pred_{model_name}'] = result['y_pred']
            df_results[f'error_{model_name}'] = result['y_pred'] - result['y_true']
        
        df_results['is_extreme'] = df_results['actual'] >= 310
        
        output_path = 'extreme_predictions_202509.csv'
        df_results.to_csv(output_path, index=False)
        print(f"  ✅ 저장 완료: {output_path}")
        
        return df_results

# ========================================
# 메인 실행
# ========================================

def main():
    # 평가기 생성
    evaluator = ExtremePredictionEvaluator()
    
    # 스케일러 로드
    if not evaluator.load_scalers():
        print("스케일러를 먼저 생성하세요!")
        return
    
    # 데이터 준비
    df = evaluator.prepare_data()
    
    # 시퀀스 생성
    X, y, X_physics, timestamps, n_features = evaluator.create_test_sequences(df)
    
    # 모델 로드
    models = evaluator.load_models(n_features)
    
    if not models:
        print("모델을 찾을 수 없습니다!")
        return
    
    # 평가 수행
    results = {}
    
    if 'ExtremePatchTST' in models:
        results['ExtremePatchTST'] = evaluator.evaluate_model(
            models['ExtremePatchTST'], 'ExtremePatchTST', X, y
        )
    
    if 'ImprovedPINN' in models:
        results['ImprovedPINN'] = evaluator.evaluate_model(
            models['ImprovedPINN'], 'ImprovedPINN', X, y, X_physics
        )
    
    # 시각화
    evaluator.visualize_results(results, timestamps)
    
    # 결과 저장
    df_results = evaluator.save_results(results, timestamps)
    
    # 최종 요약
    print("\n" + "="*80)
    print("📊 최종 평가 요약")
    print("="*80)
    
    print(f"\n{'모델':<20} {'MAE':<10} {'310+ 감지율':<15} {'F1-Score':<10}")
    print("-"*60)
    
    for model_name, result in results.items():
        detection_rate = (result['recall'] * 100)
        print(f"{model_name:<20} {result['mae']:<10.2f} {detection_rate:<15.1f}% {result['f1']:<10.2%}")
    
    # 우수 모델 선정
    best_model = min(results.items(), key=lambda x: x[1]['mae'])
    print(f"\n🏆 최우수 모델: {best_model[0]} (MAE: {best_model[1]['mae']:.2f})")
    
    print("\n✅ 평가 완료!")

if __name__ == "__main__":
    main()