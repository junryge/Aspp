# -*- coding: utf-8 -*-
"""
HUBROOM 반송량 예측 평가 시스템 - 10분 단위 결과
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
# 🔧 모델 재구성 (실제 저장된 구조에 맞춤)
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
    """실제 저장된 구조에 맞는 PatchTST 모델 재구성"""
    
    inputs = Input(shape=(seq_len, n_features))
    
    # Patching
    n_patches = seq_len // patch_len
    patches = tf.reshape(inputs, (-1, n_patches, patch_len * n_features))
    
    # Linear projection (dense)
    x = Dense(d_model, name='dense')(patches)
    
    # Positional encoding (임베딩은 별도로 추가)
    positions = tf.range(start=0, limit=n_patches, delta=1)
    pos_embedding = Embedding(input_dim=n_patches, output_dim=d_model, name='pos_embedding')(positions)
    x = x + pos_embedding
    
    # Transformer layers (3개)
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
    """실제 저장된 구조에 맞는 PatchTST+PINN 모델 재구성"""
    
    # 두 개의 입력
    seq_input = Input(shape=(seq_len, n_features), name='sequence_input')
    physics_input = Input(shape=(3,), name='physics_input')
    
    # PatchTST 부분 (patch_tst_1으로 래핑)
    # Patching
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
# 📊 10분 단위 평가 클래스
# ===========================

class HUBROOM10MinEvaluator:
    """HUBROOM 예측 모델 평가 클래스 - 10분 단위"""
    
    def __init__(self, data_path='20250801_to_20250831.csv'):
        self.data_path = data_path
        self.seq_len = 20
        self.pred_len = 10
        self.target_col = 'CURRENT_M16A_3F_JOB_2'
        self.critical_threshold = 300
        
        # 모델 경로
        self.patchtst_weights = './checkpoints/PatchTST_best.h5'
        self.pinn_weights = './checkpoints/PatchTST_PINN_best.h5'
        
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
        
        self.n_features = None  # 데이터 로드 후 설정
    
    def load_scaler(self, filename):
        """스케일러 로드"""
        filepath = f'./checkpoints/{filename}'
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                print(f"✅ {filename} 로드 완료")
                return pickle.load(f)
        else:
            print(f"⚠️ {filename}이 없습니다. save_scalers.py를 먼저 실행하세요!")
            return None
    
    def prepare_data(self):
        """2025년 9월 데이터 준비"""
        print("\n📂 2025년 9월 데이터 로드 중...")
        df = pd.read_csv(self.data_path)
        
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
    
    def create_10min_evaluation_sequences(self, df, numeric_cols):
        """10분 단위로 평가용 시퀀스 생성"""
        results_list = []
        
        data = df[numeric_cols].values
        target_idx = numeric_cols.index(self.target_col)
        
        # 사용 가능한 컬럼 확인
        available_inflow = [col for col in self.inflow_cols if col in numeric_cols]
        available_outflow = [col for col in self.outflow_cols if col in numeric_cols]
        
        print(f"\n📊 10분 단위 시퀀스 생성 중...")
        
        # 10분 간격으로 시퀀스 생성
        i = 0
        while i + self.seq_len + self.pred_len <= len(data):
            # 입력 시퀀스 (과거 20분)
            X_seq = data[i:i+self.seq_len]
            
            # 실제 값 (10분 후 단일 값)
            actual_10min_value = data[i+self.seq_len+self.pred_len-1, target_idx]
            
            # 예측 시점 (10분 후)
            pred_time = df['timestamp'].iloc[i+self.seq_len+self.pred_len-1]
            input_start_time = df['timestamp'].iloc[i]
            input_end_time = df['timestamp'].iloc[i+self.seq_len-1]
            
            # 물리 데이터 (마지막 입력 시점의 상태)
            current_state = data[i+self.seq_len-1]
            current_hubroom = current_state[target_idx]
            
            inflow_sum = sum([current_state[numeric_cols.index(col)] 
                            for col in available_inflow])
            outflow_sum = sum([current_state[numeric_cols.index(col)] 
                             for col in available_outflow])
            
            physics = np.array([current_hubroom, inflow_sum, outflow_sum])
            
            # 결과 저장
            results_list.append({
                'X_seq': X_seq,
                'physics': physics,
                'actual_value': actual_10min_value,
                'pred_time': pred_time,
                'input_period': f"{input_start_time.strftime('%H:%M')}~{input_end_time.strftime('%H:%M')}",
                'input_start': input_start_time,
                'input_end': input_end_time
            })
            
            # 10분 후로 이동
            i += 10
        
        print(f"✅ 10분 단위 시퀀스 생성 완료: {len(results_list)}개")
        
        return results_list
    
    def load_models(self):
        """모델 재구성 및 가중치 로드"""
        print("\n🤖 모델 재구성 및 가중치 로드 중...")
        
        models = {}
        
        # PatchTST 모델
        if os.path.exists(self.patchtst_weights):
            try:
                print("📌 PatchTST 모델 재구성 중...")
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
                model.load_weights(self.patchtst_weights, by_name=True, skip_mismatch=True)
                models['PatchTST'] = model
                print("✅ PatchTST 모델 로드 완료")
                
            except Exception as e:
                print(f"❌ PatchTST 로드 실패: {e}")
        
        # PatchTST+PINN 모델
        if os.path.exists(self.pinn_weights):
            try:
                print("📌 PatchTST+PINN 모델 재구성 중...")
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
                
                # 가중치 로드
                model.load_weights(self.pinn_weights, by_name=True, skip_mismatch=True)
                models['PatchTST_PINN'] = model
                print("✅ PatchTST+PINN 모델 로드 완료")
                
            except Exception as e:
                print(f"❌ PatchTST+PINN 로드 실패: {e}")
        
        return models
    
    def predict_10min_intervals(self, models, results_list):
        """10분 단위로 예측 수행"""
        # 데이터 준비
        X_list = np.array([r['X_seq'] for r in results_list])
        physics_list = np.array([r['physics'] for r in results_list])
        actual_values = np.array([r['actual_value'] for r in results_list])
        
        # 정규화
        n_samples, seq_len, n_features = X_list.shape
        X_scaled = self.scaler_X.transform(X_list.reshape(-1, n_features)).reshape(n_samples, seq_len, n_features)
        physics_scaled = self.scaler_physics.transform(physics_list) if self.scaler_physics else physics_list
        
        predictions = {}
        
        for model_name, model in models.items():
            print(f"\n📊 {model_name} 모델로 예측 중...")
            
            try:
                # 예측
                if model_name == 'PatchTST_PINN':
                    y_pred_scaled = model.predict([X_scaled, physics_scaled], verbose=0, batch_size=32)
                else:
                    y_pred_scaled = model.predict(X_scaled, verbose=0, batch_size=32)
                
                # 역정규화
                y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                predictions[model_name] = y_pred
                
            except Exception as e:
                print(f"❌ {model_name} 예측 실패: {e}")
                predictions[model_name] = np.zeros(len(actual_values))
        
        # 결과 통합
        for i, result in enumerate(results_list):
            for model_name, pred_values in predictions.items():
                result[f'pred_{model_name}'] = pred_values[i]
        
        return results_list
    
    def create_10min_report(self, results_with_predictions):
        """10분 단위 보고서 생성"""
        print("\n" + "="*80)
        print("📊 10분 단위 예측 결과 보고서")
        print("="*80)
        
        # 데이터프레임 생성
        report_data = []
        
        for result in results_with_predictions:
            row = {
                '예측시간': result['pred_time'].strftime('%Y-%m-%d %H:%M'),
                '입력구간': result['input_period'],
                '실제값': f"{result['actual_value']:.1f}",
            }
            
            # 각 모델의 예측값 추가
            for model_name in ['PatchTST', 'PatchTST_PINN']:
                if f'pred_{model_name}' in result:
                    pred_val = result[f'pred_{model_name}']
                    error = pred_val - result['actual_value']
                    row[f'{model_name}_예측'] = f"{pred_val:.1f}"
                    row[f'{model_name}_오차'] = f"{error:+.1f}"
            
            # 위험 여부
            if result['actual_value'] >= 300:
                row['상태'] = '⚠️ 위험'
            else:
                row['상태'] = '✅ 정상'
            
            report_data.append(row)
        
        df_report = pd.DataFrame(report_data)
        
        # 처음 20개 출력
        print("\n📋 처음 20개 예측 결과:")
        print("-"*80)
        print(df_report.head(20).to_string(index=False))
        
        # 통계 요약
        print("\n📈 성능 통계:")
        print("-"*80)
        
        for model_name in ['PatchTST', 'PatchTST_PINN']:
            if f'pred_{model_name}' in results_with_predictions[0]:
                actual = np.array([r['actual_value'] for r in results_with_predictions])
                pred = np.array([r[f'pred_{model_name}'] for r in results_with_predictions])
                
                mae = mean_absolute_error(actual, pred)
                rmse = np.sqrt(mean_squared_error(actual, pred))
                r2 = r2_score(actual, pred)
                
                # 300 이상 정확도
                mask_300 = actual >= 300
                if np.sum(mask_300) > 0:
                    tp = np.sum((pred >= 300) & (actual >= 300))
                    recall = tp / np.sum(mask_300)
                else:
                    recall = 0
                
                print(f"\n{model_name}:")
                print(f"  - MAE: {mae:.2f}")
                print(f"  - RMSE: {rmse:.2f}")
                print(f"  - R²: {r2:.3f}")
                print(f"  - 300 이상 감지율: {recall:.1%}")
        
        # CSV 저장
        df_report.to_csv('hubroom_10min_predictions.csv', index=False, encoding='utf-8-sig')
        print("\n✅ 결과가 'hubroom_10min_predictions.csv'에 저장되었습니다.")
        
        return df_report
    
    def visualize_10min_results(self, results_with_predictions):
        """10분 단위 결과 시각화"""
        # 데이터 추출
        times = [r['pred_time'] for r in results_with_predictions]
        actual = np.array([r['actual_value'] for r in results_with_predictions])
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle('HUBROOM 반송량 10분 단위 예측 결과', fontsize=16)
        
        # 1. 시계열 그래프
        ax1 = axes[0]
        ax1.plot(times, actual, 'k-', label='실제값', linewidth=2)
        
        colors = {'PatchTST': 'blue', 'PatchTST_PINN': 'green'}
        for model_name, color in colors.items():
            if f'pred_{model_name}' in results_with_predictions[0]:
                pred = [r[f'pred_{model_name}'] for r in results_with_predictions]
                ax1.plot(times, pred, '--', color=color, label=f'{model_name} 예측', alpha=0.8)
        
        ax1.axhline(y=300, color='red', linestyle=':', label='위험 임계값')
        ax1.set_xlabel('시간')
        ax1.set_ylabel('HUBROOM 반송량')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title('10분 단위 예측 시계열')
        
        # 2. 산점도
        ax2 = axes[1]
        for model_name, color in colors.items():
            if f'pred_{model_name}' in results_with_predictions[0]:
                pred = np.array([r[f'pred_{model_name}'] for r in results_with_predictions])
                ax2.scatter(actual, pred, alpha=0.6, label=model_name, color=color)
        
        # 대각선
        max_val = max(actual.max(), 400)
        ax2.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Perfect Prediction')
        ax2.axvline(x=300, color='red', linestyle=':', alpha=0.5)
        ax2.axhline(y=300, color='red', linestyle=':', alpha=0.5)
        
        ax2.set_xlabel('실제값')
        ax2.set_ylabel('예측값')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_title('예측값 vs 실제값')
        
        plt.tight_layout()
        plt.savefig('hubroom_10min_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """메인 실행 함수"""
    print("="*80)
    print("🏭 HUBROOM 반송량 10분 단위 예측 평가 시스템")
    print("📅 대상: 2025년 9월 데이터")
    print("="*80)
    
    # TensorFlow 로그 레벨 조정
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # 평가기 생성
    evaluator = HUBROOM10MinEvaluator()
    
    # 스케일러 확인
    if evaluator.scaler_X is None:
        print("\n❌ 스케일러가 없습니다. save_scalers.py를 먼저 실행하세요!")
        return
    
    try:
        # 1. 데이터 준비
        df, numeric_cols = evaluator.prepare_data()
        
        # 2. 10분 단위 시퀀스 생성
        results_list = evaluator.create_10min_evaluation_sequences(df, numeric_cols)
        
        # 3. 모델 로드
        models = evaluator.load_models()
        
        if not models:
            print("\n❌ 로드된 모델이 없습니다!")
            return
        
        # 4. 10분 단위 예측
        results_with_predictions = evaluator.predict_10min_intervals(models, results_list)
        
        # 5. 보고서 생성
        df_report = evaluator.create_10min_report(results_with_predictions)
        
        # 6. 시각화
        evaluator.visualize_10min_results(results_with_predictions)
        
        print("\n✅ 평가 완료!")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()