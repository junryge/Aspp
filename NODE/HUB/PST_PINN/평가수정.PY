# -*- coding: utf-8 -*-
"""
HUBROOM 전체 데이터 예측 시스템
- 00:20분부터 시작 (20분 과거 데이터 필요)
- 각 시점에서 10분 후 예측
- 전체 기간에 대해 연속적으로 예측
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Flatten, Embedding
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ===========================
# 🔧 모델 클래스 정의
# ===========================

class TransformerEncoderLayer(layers.Layer):
    """커스텀 Transformer Encoder Layer"""
    
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super(TransformerEncoderLayer, self).__init__(**kwargs)
        
        self.mha = MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=d_model // num_heads,
            dropout=dropout_rate
        )
        
        self.ffn = keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])
        
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
    
    def call(self, inputs, training=False):
        attn_output = self.mha(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2

def build_patchtst_model(seq_len=20, n_features=39, patch_len=5, d_model=128, 
                        n_heads=8, d_ff=256, n_layers=3, dropout=0.1):
    """PatchTST 모델 재구성"""
    
    inputs = Input(shape=(seq_len, n_features))
    
    # Patching
    n_patches = seq_len // patch_len
    patches = tf.reshape(inputs, (-1, n_patches, patch_len * n_features))
    
    # Linear projection
    x = Dense(d_model, name='dense')(patches)
    
    # Positional encoding
    positions = tf.range(start=0, limit=n_patches, delta=1)
    pos_embedding = Embedding(input_dim=n_patches, output_dim=d_model, name='pos_embedding')(positions)
    x = x + pos_embedding
    
    # Transformer layers
    x = TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, name='transformer_encoder_layer')(x)
    x = TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, name='transformer_encoder_layer_1')(x)
    x = TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, name='transformer_encoder_layer_2')(x)
    
    # Flatten
    x = Flatten(name='flatten')(x)
    
    # Dense layers
    x = Dense(128, activation='relu', name='dense_7')(x)
    x = Dropout(dropout, name='dropout_6')(x)
    x = Dense(64, activation='relu', name='dense_8')(x)
    outputs = Dense(1, name='dense_9')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='patch_tst')
    return model

def build_patchtst_pinn_model(seq_len=20, n_features=39, patch_len=5, d_model=128,
                             n_heads=8, d_ff=256, n_layers=3, dropout=0.1):
    """PatchTST+PINN 하이브리드 모델 재구성"""
    
    # 두 개의 입력
    seq_input = Input(shape=(seq_len, n_features), name='sequence_input')
    physics_input = Input(shape=(3,), name='physics_input')
    
    # PatchTST 부분
    n_patches = seq_len // patch_len
    patches = tf.reshape(seq_input, (-1, n_patches, patch_len * n_features))
    
    # Linear projection
    x = Dense(d_model, name='dense_10')(patches)
    
    # Positional encoding
    positions = tf.range(start=0, limit=n_patches, delta=1)
    pos_embedding = Embedding(input_dim=n_patches, output_dim=d_model, name='pos_embedding')(positions)
    x = x + pos_embedding
    
    # Transformer layers
    for i in range(3):
        x = TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout,
            name=f'transformer_encoder_layer_{i+3}'
        )(x)
    
    # Global average pooling
    x = GlobalAveragePooling1D()(x)
    
    # Dense layers for TST output
    tst_out = Dense(128, activation='relu', name='dense_17')(x)
    tst_out = Dense(64, activation='relu', name='dense_18')(tst_out)
    tst_out = Dense(1, name='dense_19')(tst_out)
    
    # Physics network
    physics_x = Dense(64, activation='relu', name='dense_20')(physics_input)
    physics_x = Dense(32, activation='relu', name='dense_21')(physics_x)
    
    # Combine
    combined = layers.concatenate([tst_out, physics_x])
    x = Dense(64, activation='relu', name='dense_22')(combined)
    x = Dense(32, activation='relu', name='dense_23')(x)
    outputs = Dense(1, name='dense_24')(x)
    
    # Dropout layer
    outputs = Dropout(dropout, name='dropout_14')(outputs)
    
    model = Model(
        inputs=[seq_input, physics_input],
        outputs=outputs,
        name='patch_tst_pinn_hybrid'
    )
    return model

# ===========================
# 📊 전체 예측 시스템 클래스
# ===========================

class HUBROOMFullPredictor:
    """HUBROOM 전체 시간대 예측 시스템"""
    
    def __init__(self, model_type='PINN'):
        """
        Args:
            model_type: 'PatchTST' 또는 'PINN'
        """
        self.model_type = model_type
        self.seq_len = 20
        self.pred_len = 10
        self.target_col = 'CURRENT_M16A_3F_JOB_2'
        self.critical_threshold = 300
        
        # 모델 경로
        if model_type == 'PINN':
            self.model_weights = './checkpoints/PatchTST_PINN_best.h5'
        else:
            self.model_weights = './checkpoints/PatchTST_best.h5'
        
        # 스케일러 로드
        self.scaler_X = self.load_scaler('scaler_X.pkl')
        self.scaler_y = self.load_scaler('scaler_y.pkl')
        self.scaler_physics = self.load_scaler('scaler_physics.pkl')
        
        # 물리 컬럼
        self.inflow_cols = [
            'M16A_6F_TO_HUB_JOB', 'M16A_2F_TO_HUB_JOB2',
            'M14A_3F_TO_HUB_JOB2', 'M14B_7F_TO_HUB_JOB2'
        ]
        self.outflow_cols = [
            'M16A_3F_TO_M16A_6F_JOB', 'M16A_3F_TO_M16A_2F_JOB',
            'M16A_3F_TO_M14A_3F_JOB', 'M16A_3F_TO_M14B_7F_JOB'
        ]
        
        self.model = None
        self.n_features = None
    
    def load_scaler(self, filename):
        """스케일러 로드"""
        filepath = f'./checkpoints/{filename}'
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        else:
            print(f"⚠️ {filename}이 없습니다!")
            return None
    
    def load_data(self, data_path):
        """데이터 로드 및 전처리"""
        print(f"\n📂 데이터 로드 중: {data_path}")
        df = pd.read_csv(data_path)
        
        # 시간 컬럼 처리
        time_col = df.columns[0]
        df['timestamp'] = pd.to_datetime(df[time_col], format='%Y%m%d%H%M', errors='coerce')
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 결측치 처리
        df = df.fillna(method='ffill').fillna(0)
        
        # 숫자형 컬럼만 선택
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.n_features = len(numeric_cols)
        
        print(f"✅ 데이터 로드 완료: {len(df)} 행")
        print(f"📅 기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
        print(f"📊 특성 수: {self.n_features}개")
        
        return df, numeric_cols
    
    def load_model(self):
        """모델 로드"""
        print(f"\n🤖 {self.model_type} 모델 로드 중...")
        
        if not os.path.exists(self.model_weights):
            print(f"❌ 모델 파일이 없습니다: {self.model_weights}")
            return None
        
        try:
            if self.model_type == 'PINN':
                model = build_patchtst_pinn_model(
                    seq_len=self.seq_len,
                    n_features=self.n_features,
                    patch_len=5,
                    d_model=128,
                    n_heads=8,
                    d_ff=256,
                    n_layers=3,
                    dropout=0.1
                )
                # 더미 데이터로 모델 빌드
                dummy_seq = np.zeros((1, self.seq_len, self.n_features))
                dummy_physics = np.zeros((1, 3))
                _ = model([dummy_seq, dummy_physics])
            else:
                model = build_patchtst_model(
                    seq_len=self.seq_len,
                    n_features=self.n_features,
                    patch_len=5,
                    d_model=128,
                    n_heads=8,
                    d_ff=256,
                    n_layers=3,
                    dropout=0.1
                )
                # 더미 데이터로 모델 빌드
                dummy_input = np.zeros((1, self.seq_len, self.n_features))
                _ = model(dummy_input)
            
            # 가중치 로드
            model.load_weights(self.model_weights, by_name=True, skip_mismatch=True)
            self.model = model
            print(f"✅ {self.model_type} 모델 로드 완료")
            return model
            
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict_all_timestamps(self, df, numeric_cols):
        """전체 시간대에 대해 예측 수행"""
        print(f"\n🔮 전체 예측 시작 (00:20분부터)...")
        
        data = df[numeric_cols].values
        target_idx = numeric_cols.index(self.target_col)
        
        # 사용 가능한 물리 컬럼 확인
        available_inflow = [col for col in self.inflow_cols if col in numeric_cols]
        available_outflow = [col for col in self.outflow_cols if col in numeric_cols]
        
        # 예측 결과 저장
        predictions = []
        actual_values = []
        timestamps = []
        prediction_times = []
        
        # 00:20분부터 시작 (인덱스 19부터)
        start_idx = self.seq_len - 1  # 19
        end_idx = len(data) - self.pred_len
        
        total_predictions = end_idx - start_idx
        print(f"📊 총 예측 수: {total_predictions}개")
        
        # 배치 처리를 위한 데이터 준비
        batch_size = 1000
        n_batches = (total_predictions + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            start = start_idx + batch_idx * batch_size
            end = min(start_idx + (batch_idx + 1) * batch_size, end_idx)
            
            X_batch = []
            physics_batch = []
            timestamps_batch = []
            actual_batch = []
            
            # 배치 데이터 생성
            for i in range(start, end):
                # 입력 시퀀스 (과거 20분)
                X_seq = data[i-self.seq_len+1:i+1]
                
                # 실제값 (10분 후)
                y_actual = data[i+self.pred_len, target_idx]
                
                # 타임스탬프
                current_time = df['timestamp'].iloc[i]
                pred_time = current_time + timedelta(minutes=self.pred_len)
                
                # 물리 데이터 (현재 상태)
                current_state = data[i]
                current_hubroom = current_state[target_idx]
                
                inflow_sum = sum([current_state[numeric_cols.index(col)] 
                                for col in available_inflow])
                outflow_sum = sum([current_state[numeric_cols.index(col)] 
                                 for col in available_outflow])
                
                physics = np.array([current_hubroom, inflow_sum, outflow_sum])
                
                X_batch.append(X_seq)
                physics_batch.append(physics)
                timestamps_batch.append(current_time)
                actual_batch.append(y_actual)
            
            # 배치 예측
            X_batch = np.array(X_batch)
            physics_batch = np.array(physics_batch)
            
            # 정규화
            X_scaled = self.scaler_X.transform(
                X_batch.reshape(-1, self.n_features)
            ).reshape(-1, self.seq_len, self.n_features)
            
            if self.model_type == 'PINN':
                physics_scaled = self.scaler_physics.transform(physics_batch)
                y_pred_scaled = self.model.predict([X_scaled, physics_scaled], 
                                                 batch_size=32, verbose=0)
            else:
                y_pred_scaled = self.model.predict(X_scaled, 
                                                 batch_size=32, verbose=0)
            
            # 역정규화
            y_pred = self.scaler_y.inverse_transform(
                y_pred_scaled.reshape(-1, 1)
            ).flatten()
            
            # 결과 저장
            predictions.extend(y_pred)
            actual_values.extend(actual_batch)
            timestamps.extend(timestamps_batch)
            
            # 진행 상황 출력
            if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
                progress = (batch_idx + 1) / n_batches * 100
                print(f"  진행률: {progress:.1f}% ({len(predictions)}/{total_predictions})")
        
        print(f"\n✅ 예측 완료!")
        
        # 결과 반환
        return {
            'timestamps': np.array(timestamps),
            'predictions': np.array(predictions),
            'actual_values': np.array(actual_values)
        }
    
    def analyze_results(self, results):
        """예측 결과 분석"""
        print("\n📊 예측 결과 분석")
        print("="*60)
        
        y_true = results['actual_values']
        y_pred = results['predictions']
        
        # 전체 성능 메트릭
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        print(f"\n📈 전체 성능:")
        print(f"  - MAE: {mae:.4f}")
        print(f"  - RMSE: {rmse:.4f}")
        print(f"  - R²: {r2:.4f}")
        
        # 300 이상 분석
        over_300_true = np.sum(y_true >= 300)
        over_300_pred = np.sum(y_pred >= 300)
        
        print(f"\n🚨 300 이상 분석:")
        print(f"  - 실제 300 이상: {over_300_true}개 ({over_300_true/len(y_true)*100:.1f}%)")
        print(f"  - 예측 300 이상: {over_300_pred}개 ({over_300_pred/len(y_pred)*100:.1f}%)")
        
        # 300 이상일 때 성능
        mask_300 = y_true >= 300
        if np.sum(mask_300) > 0:
            mae_300 = mean_absolute_error(y_true[mask_300], y_pred[mask_300])
            print(f"  - 300 이상일 때 MAE: {mae_300:.4f}")
        
        # 시간대별 분석
        self.analyze_by_hour(results)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'over_300_true': over_300_true,
            'over_300_pred': over_300_pred
        }
    
    def analyze_by_hour(self, results):
        """시간대별 분석"""
        print("\n⏰ 시간대별 분석:")
        
        # 데이터프레임 생성
        df_results = pd.DataFrame({
            'timestamp': results['timestamps'],
            'actual': results['actual_values'],
            'predicted': results['predictions'],
            'error': results['predictions'] - results['actual_values']
        })
        
        # 시간대 추출
        df_results['hour'] = pd.to_datetime(df_results['timestamp']).dt.hour
        
        # 시간대별 평균 오차
        hourly_mae = df_results.groupby('hour').apply(
            lambda x: mean_absolute_error(x['actual'], x['predicted'])
        )
        
        print("\n시간대별 MAE:")
        for hour, mae in hourly_mae.items():
            print(f"  {hour:02d}시: {mae:.2f}")
    
    def save_results(self, results, output_path='full_predictions.csv'):
        """예측 결과 저장"""
        print(f"\n💾 예측 결과 저장 중...")
        
        # 데이터프레임 생성
        df_results = pd.DataFrame({
            'timestamp': results['timestamps'],
            'actual': results['actual_values'],
            'predicted': results['predictions'],
            'error': results['predictions'] - results['actual_values'],
            'is_critical_actual': results['actual_values'] >= 300,
            'is_critical_pred': results['predictions'] >= 300
        })
        
        # 예측 시간 추가 (10분 후)
        df_results['prediction_for'] = pd.to_datetime(df_results['timestamp']) + timedelta(minutes=10)
        
        # 저장
        df_results.to_csv(output_path, index=False)
        print(f"✅ 저장 완료: {output_path}")
        
        # 요약 출력
        print(f"\n📊 저장된 데이터 요약:")
        print(f"  - 총 예측 수: {len(df_results)}개")
        print(f"  - 기간: {df_results['timestamp'].min()} ~ {df_results['timestamp'].max()}")
        print(f"  - 300 이상 실제: {df_results['is_critical_actual'].sum()}개")
        print(f"  - 300 이상 예측: {df_results['is_critical_pred'].sum()}개")
    
    def visualize_results(self, results, sample_size=1000):
        """결과 시각화"""
        print("\n📊 결과 시각화 생성 중...")
        
        y_true = results['actual_values']
        y_pred = results['predictions']
        timestamps = results['timestamps']
        
        # 큰 데이터셋인 경우 샘플링
        if len(y_true) > sample_size:
            indices = np.linspace(0, len(y_true)-1, sample_size, dtype=int)
            y_true_sample = y_true[indices]
            y_pred_sample = y_pred[indices]
            timestamps_sample = timestamps[indices]
        else:
            y_true_sample = y_true
            y_pred_sample = y_pred
            timestamps_sample = timestamps
        
        # 그래프 생성
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'HUBROOM 전체 예측 결과 ({self.model_type})', fontsize=16)
        
        # 1. 시계열 그래프
        ax1 = axes[0, 0]
        ax1.plot(y_true_sample, label='실제값', alpha=0.7, linewidth=1)
        ax1.plot(y_pred_sample, label='예측값', alpha=0.7, linewidth=1)
        ax1.axhline(y=300, color='red', linestyle=':', label='위험 임계값')
        ax1.set_title('시계열 예측 비교')
        ax1.set_xlabel('시간 인덱스')
        ax1.set_ylabel('HUBROOM 반송량')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 산점도
        ax2 = axes[0, 1]
        ax2.scatter(y_true, y_pred, alpha=0.5, s=1)
        ax2.plot([0, max(y_true.max(), y_pred.max())], 
                 [0, max(y_true.max(), y_pred.max())], 
                 'r--', label='Perfect Prediction')
        ax2.axvline(x=300, color='red', linestyle=':', alpha=0.5)
        ax2.axhline(y=300, color='red', linestyle=':', alpha=0.5)
        ax2.set_title('예측값 vs 실제값')
        ax2.set_xlabel('실제값')
        ax2.set_ylabel('예측값')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 오차 히스토그램
        ax3 = axes[1, 0]
        errors = y_pred - y_true
        ax3.hist(errors, bins=50, alpha=0.7, edgecolor='black')
        ax3.axvline(x=0, color='red', linestyle='--', label='Zero Error')
        ax3.set_title('예측 오차 분포')
        ax3.set_xlabel('예측 오차')
        ax3.set_ylabel('빈도')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 시간대별 평균 오차
        ax4 = axes[1, 1]
        df_temp = pd.DataFrame({
            'timestamp': timestamps,
            'error': np.abs(y_pred - y_true)
        })
        df_temp['hour'] = pd.to_datetime(df_temp['timestamp']).dt.hour
        hourly_mae = df_temp.groupby('hour')['error'].mean()
        
        ax4.bar(hourly_mae.index, hourly_mae.values)
        ax4.set_title('시간대별 평균 절대 오차')
        ax4.set_xlabel('시간 (Hour)')
        ax4.set_ylabel('MAE')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'hubroom_full_prediction_{self.model_type}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ 시각화 완료!")

def main():
    """메인 실행 함수"""
    print("="*80)
    print("🏭 HUBROOM 전체 데이터 예측 시스템")
    print("📅 예측 시작: 00:20분부터 (20분 과거 데이터 필요)")
    print("="*80)
    
    # TensorFlow 로그 레벨 조정
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # 모델 선택
    print("\n모델 선택:")
    print("1. PatchTST (기본 모델)")
    print("2. PatchTST+PINN (물리 법칙 적용 - 권장)")
    
    choice = input("\n선택 (1 또는 2, 기본값 2): ").strip()
    if choice == '1':
        model_type = 'PatchTST'
    else:
        model_type = 'PINN'
    
    # 예측기 생성
    predictor = HUBROOMFullPredictor(model_type=model_type)
    
    # 스케일러 확인
    if predictor.scaler_X is None:
        print("\n❌ 스케일러가 없습니다. 먼저 모델을 학습시켜주세요!")
        return
    
    # 데이터 파일 경로
    data_path = input("\n데이터 파일 경로 (기본값: 20250801_to_20250831.csv): ").strip()
    if not data_path:
        data_path = '20250801_to_20250831.csv'
    
    if not os.path.exists(data_path):
        print(f"❌ 파일이 없습니다: {data_path}")
        return
    
    try:
        # 1. 데이터 로드
        df, numeric_cols = predictor.load_data(data_path)
        
        # 2. 모델 로드
        model = predictor.load_model()
        if model is None:
            return
        
        # 3. 전체 예측 수행
        results = predictor.predict_all_timestamps(df, numeric_cols)
        
        # 4. 결과 분석
        metrics = predictor.analyze_results(results)
        
        # 5. 결과 저장
        output_filename = f'full_predictions_{model_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        predictor.save_results(results, output_filename)
        
        # 6. 시각화
        predictor.visualize_results(results)
        
        print("\n✅ 전체 예측 완료!")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()