# -*- coding: utf-8 -*-
"""
HUBROOM 반송량 예측 평가 시스템 - 인덱싱 버그 수정 버전
Created on Mon Sep 1 15:13:16 2025
@author: X0163954
수정: 과거 20분 데이터로 10분 후 예측 - 정확한 인덱싱
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
# 📊 평가 클래스
# ===========================

class HUBROOMEvaluator:
    """HUBROOM 예측 모델 평가 클래스"""
    
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
    
    def create_evaluation_sequences(self, df, numeric_cols):
        """평가용 시퀀스 생성 - 정확한 인덱싱 버전"""
        X_list = []
        y_actual_list = []
        physics_list = []
        
        # 시간 관련 정보 저장
        current_times = []  # 현재 시간 (입력의 마지막 시점)
        predicted_target_times = []  # 예측 타겟 시간 (10분 후)
        input_start_times = []  # 입력 시작 시간
        input_end_times = []  # 입력 끝 시간
        
        # 입력 데이터 통계 정보
        input_max_values = []
        input_min_values = []
        
        # 현재 시점 값 저장 (디버깅용)
        current_values = []
        
        data = df[numeric_cols].values
        target_idx = numeric_cols.index(self.target_col)
        
        # 사용 가능한 컬럼 확인
        available_inflow = [col for col in self.inflow_cols if col in numeric_cols]
        available_outflow = [col for col in self.outflow_cols if col in numeric_cols]
        
        print(f"\n📊 시퀀스 생성 중...")
        print(f"   - 입력: 과거 20분 (seq_len={self.seq_len})")
        print(f"   - 예측: 10분 후 값 (pred_len={self.pred_len})")
        
        # 전체 시퀀스 수 계산
        total_sequences = len(data) - self.seq_len - self.pred_len + 1
        
        # 시퀀스 생성
        for i in range(total_sequences):
            # ===================
            # 1. 입력 시퀀스 (과거 20분: i ~ i+19)
            # ===================
            X_seq = data[i:i+self.seq_len]
            
            # ===================
            # 2. 실제값 (10분 후의 값)
            # ===================
            # 수정된 인덱싱: 현재(i+19)로부터 10분 후 = i+29
            actual_10min_later = data[i+self.seq_len+self.pred_len-1, target_idx]
            
            # ===================
            # 3. 시간 정보
            # ===================
            input_start = df['timestamp'].iloc[i]  # 입력 시작 시간
            input_end = df['timestamp'].iloc[i+self.seq_len-1]  # 입력 끝 시간 (현재 시간)
            current_time = input_end  # 현재 시간은 입력의 마지막 시점
            # 수정된 인덱싱: 10분 후 시간
            target_time = df['timestamp'].iloc[i+self.seq_len+self.pred_len-1]
            
            # ===================
            # 4. 입력 시퀀스의 최대/최소값 (과거 20분 데이터에서)
            # ===================
            input_target_values = X_seq[:, target_idx]  # 과거 20분간의 타겟 컬럼 값들
            input_max = np.max(input_target_values)
            input_min = np.min(input_target_values)
            
            # 현재 시점 값 저장 (디버깅용)
            current_value = X_seq[-1, target_idx]
            current_values.append(current_value)
            
            # ===================
            # 5. 물리 데이터 (현재 상태 = 입력의 마지막 시점)
            # ===================
            current_state = X_seq[-1]  # 입력 시퀀스의 마지막 상태 (현재)
            current_hubroom = current_state[target_idx]
            
            inflow_sum = sum([current_state[numeric_cols.index(col)] 
                            for col in available_inflow])
            outflow_sum = sum([current_state[numeric_cols.index(col)] 
                             for col in available_outflow])
            
            physics = np.array([current_hubroom, inflow_sum, outflow_sum])
            
            # ===================
            # 6. 리스트에 추가
            # ===================
            X_list.append(X_seq)
            y_actual_list.append(actual_10min_later)  # 10분 후 단일 값
            physics_list.append(physics)
            
            current_times.append(current_time)
            predicted_target_times.append(target_time)
            input_start_times.append(input_start)
            input_end_times.append(input_end)
            input_max_values.append(input_max)
            input_min_values.append(input_min)
        
        print(f"✅ 시퀀스 생성 완료: {len(X_list)}개")
        
        # 데이터 검증 출력 (더 자세히)
        print(f"\n📋 데이터 검증 (처음 5개 샘플):")
        for i in range(min(5, len(X_list))):
            print(f"\n  [{i+1}번째 샘플]")
            print(f"    - 입력 기간: {input_start_times[i]} ~ {input_end_times[i]}")
            print(f"    - 현재 시간: {current_times[i]}")
            print(f"    - 현재 HUBROOM 값: {current_values[i]:.1f}")
            print(f"    - 예측 타겟 시간: {predicted_target_times[i]}")
            print(f"    - 입력 최대값: {input_max_values[i]:.1f}")
            print(f"    - 입력 최소값: {input_min_values[i]:.1f}")
            print(f"    - 10분 후 실제값: {y_actual_list[i]:.1f}")
            
            # 시간 차이 검증
            time_diff = predicted_target_times[i] - current_times[i]
            print(f"    - 시간 차이 검증: {time_diff.total_seconds()/60:.0f}분 (10분이어야 함)")
        
        # 인덱싱 검증
        print(f"\n🔍 인덱싱 검증 (첫 번째 샘플):")
        print(f"  - i=0일 때:")
        print(f"  - 입력 인덱스: 0 ~ 19 (20개)")
        print(f"  - 현재 시점 인덱스: 19")
        print(f"  - 10분 후 인덱스: 29 (19 + 10)")
        print(f"  - 실제 사용된 인덱스: {self.seq_len + self.pred_len - 1}")
        
        # 시간 정보 딕셔너리
        time_info = {
            'timestamp': current_times,  # 현재 시간
            'predicted_target_time': predicted_target_times,  # 10분 후 시간
            'input_start': input_start_times,
            'input_end': input_end_times,
            'input_max': input_max_values,
            'input_min': input_min_values,
            'current_value': current_values  # 디버깅용 추가
        }
        
        return (np.array(X_list), np.array(y_actual_list), 
                np.array(physics_list), time_info)
    
    def load_models(self):
        """모델 재구성 및 가중치 로드"""
        print("\n🤖 모델 재구성 및 가중치 로드 중...")
        
        models = {}
        
        # PatchTST 모델 로드
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
                import traceback
                traceback.print_exc()
        
        # PatchTST+PINN 모델 로드
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
                import traceback
                traceback.print_exc()
        
        return models
    
    def predict_and_evaluate(self, models, X, y_actual, physics_data, time_info):
        """모델별 예측 및 평가"""
        results = {}
        
        # 데이터 정규화
        n_samples, seq_len, n_features = X.shape
        
        # 스케일러 확인
        if self.scaler_X is None or self.scaler_y is None:
            print("❌ 스케일러가 로드되지 않았습니다!")
            return results
        
        X_scaled = self.scaler_X.transform(X.reshape(-1, n_features)).reshape(n_samples, seq_len, n_features)
        physics_scaled = self.scaler_physics.transform(physics_data) if self.scaler_physics else physics_data
        
        for model_name, model in models.items():
            print(f"\n{'='*60}")
            print(f"📊 {model_name} 모델 평가")
            print(f"{'='*60}")
            
            try:
                # 예측
                if model_name == 'PatchTST_PINN':
                    # PINN 모델은 물리 데이터도 필요
                    y_pred_scaled = model.predict([X_scaled, physics_scaled], verbose=1, batch_size=32)
                else:
                    # PatchTST는 시퀀스 데이터만
                    y_pred_scaled = model.predict(X_scaled, verbose=1, batch_size=32)
                
                # 모델 출력 형태 확인
                print(f"예측 출력 형태: {y_pred_scaled.shape}")
                
                # 역정규화 (10분 후 예측값)
                y_pred_10min = self.scaler_y.inverse_transform(
                    y_pred_scaled.reshape(-1, 1)
                ).flatten()
                
                # 실제값은 이미 10분 후 단일 값
                y_true_10min = y_actual
                
                # 메트릭 계산
                mae = mean_absolute_error(y_true_10min, y_pred_10min)
                mse = mean_squared_error(y_true_10min, y_pred_10min)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_true_10min, y_pred_10min)
                
                # 300 이상 예측 분석
                over_300_pred = np.sum(y_pred_10min >= 300)
                over_300_true = np.sum(y_true_10min >= 300)
                
                # 300 이상일 때의 정확도
                mask_300 = y_true_10min >= 300
                if np.sum(mask_300) > 0:
                    mae_300 = mean_absolute_error(y_true_10min[mask_300], y_pred_10min[mask_300])
                    acc_300 = np.sum((y_pred_10min >= 300) & (y_true_10min >= 300)) / np.sum(mask_300)
                else:
                    mae_300 = 0
                    acc_300 = 0
                
                # 결과 저장 (시간 정보 포함)
                results[model_name] = {
                    'y_true': y_true_10min,
                    'y_pred': y_pred_10min,
                    'time_info': time_info,  # 시간 정보 추가
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse,
                    'r2': r2,
                    'over_300_pred': over_300_pred,
                    'over_300_true': over_300_true,
                    'mae_300': mae_300,
                    'acc_300': acc_300
                }
                
                # 성능 출력
                print(f"\n📈 전체 성능:")
                print(f"  - MAE: {mae:.4f}")
                print(f"  - RMSE: {rmse:.4f}")
                print(f"  - R²: {r2:.4f}")
                
                print(f"\n🚨 300 이상 예측 분석:")
                print(f"  - 실제 300 이상: {over_300_true}개")
                print(f"  - 예측 300 이상: {over_300_pred}개")
                print(f"  - 300 이상일 때 MAE: {mae_300:.4f}")
                print(f"  - 300 감지 정확도: {acc_300:.2%}")
                
                # 샘플 출력 (개선된 버전)
                print(f"\n📝 예측 샘플 (처음 5개):")
                for i in range(min(5, len(y_pred_10min))):
                    status = "⚠️ 경고" if y_pred_10min[i] >= 300 else "✅ 정상"
                    print(f"\n  [{i+1}] 현재: {time_info['timestamp'][i]}")
                    print(f"       현재 HUBROOM: {time_info['current_value'][i]:.1f}")
                    print(f"       → 10분 후: {time_info['predicted_target_time'][i]}")
                    print(f"       실제: {y_true_10min[i]:.1f}, 예측: {y_pred_10min[i]:.1f} {status}")
                    print(f"       (과거 20분 범위: {time_info['input_min'][i]:.1f} ~ {time_info['input_max'][i]:.1f})")
                
            except Exception as e:
                print(f"❌ {model_name} 예측 중 오류 발생: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return results
    
    def save_predictions(self, results, output_path='predictions_202509.csv'):
        """예측 결과를 요청된 형식의 CSV로 저장"""
        print(f"\n💾 예측 결과 저장 중...")
        
        # PatchTST 모델 결과만 사용
        if 'PatchTST' not in results:
            print("❌ PatchTST 모델 결과가 없습니다!")
            return
        
        patchtst_result = results['PatchTST']
        time_info = patchtst_result['time_info']
        
        # 요청된 컬럼 형식으로 데이터프레임 생성
        df_results = pd.DataFrame({
            'timestamp': time_info['timestamp'],  # 현재 시간 (입력의 마지막 시점)
            'current_value': time_info['current_value'],  # 현재 시점의 실제 HUBROOM 값
            'actual': patchtst_result['y_true'],  # 10분 후 실제값
            'predicted': patchtst_result['y_pred'],  # 10분 후 예측값
            'predicted_Target_time': time_info['predicted_target_time'],  # 예측 타겟 시간 (10분 후)
            'input_start': time_info['input_start'],  # 입력 시작 시간 (20분 전)
            'input_end': time_info['input_end'],  # 입력 종료 시간 (현재)
            'input_max': time_info['input_max'],  # 과거 20분 중 최대값
            'input_min': time_info['input_min'],  # 과거 20분 중 최소값
            'error': patchtst_result['y_true'] - patchtst_result['y_pred'],  # 오차
            'Patchtst_predicted_TIME': time_info['predicted_target_time']  # PatchTST 예측 시간
        })
        
        # CSV 저장
        df_results.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✅ 예측 결과 저장 완료: {output_path}")
        
        # 요약 통계
        print(f"\n📊 저장된 데이터 요약:")
        print(f"  - 전체 예측 수: {len(df_results)}개")
        print(f"  - 300 이상 실제값: {(df_results['actual'] >= 300).sum()}개")
        print(f"  - 300 이상 예측값: {(df_results['predicted'] >= 300).sum()}개")
        print(f"  - 평균 오차: {df_results['error'].mean():.2f}")
        print(f"  - 오차 표준편차: {df_results['error'].std():.2f}")
        print(f"  - 기간: {df_results['timestamp'].min()} ~ {df_results['timestamp'].max()}")
        
        # 데이터 정합성 검증 (더 자세히)
        print(f"\n✅ 데이터 정합성 검증:")
        for sample_idx in [0, len(df_results)//2, len(df_results)-1]:
            if sample_idx < len(df_results):
                print(f"\n  [샘플 {sample_idx+1}]")
                row = df_results.iloc[sample_idx]
                print(f"    - 현재 시간: {row['timestamp']}")
                print(f"    - 현재 HUBROOM 값: {row['current_value']:.1f}")
                print(f"    - 입력 기간: {row['input_start']} ~ {row['input_end']}")
                print(f"    - 과거 20분 범위: {row['input_min']:.1f} ~ {row['input_max']:.1f}")
                print(f"    - 예측 타겟 시간: {row['predicted_Target_time']}")
                print(f"    - 10분 후 실제값: {row['actual']:.1f}")
                print(f"    - 10분 후 예측값: {row['predicted']:.1f}")
                print(f"    - 오차: {row['error']:.1f}")
                
                # 현재값과 과거 범위 관계 확인
                if row['input_min'] <= row['current_value'] <= row['input_max']:
                    print(f"    ✅ 현재값이 과거 20분 범위 내에 있음")
                else:
                    print(f"    ⚠️ 현재값이 과거 20분 범위를 벗어남")
        
        # 처음 5개 행 출력
        print(f"\n📋 저장된 데이터 샘플 (처음 5개):")
        print(df_results.head())
    
    def visualize_results(self, results):
        """결과 시각화"""
        if not results:
            print("시각화할 결과가 없습니다.")
            return
        
        # 모델별 비교 그래프
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('HUBROOM 반송량 예측 모델 비교 (2025년 9월)', fontsize=16)
        
        # 1. 시계열 예측 비교 (처음 100개)
        ax1 = axes[0, 0]
        ax1.plot(results[list(results.keys())[0]]['y_true'][:100], label=f'실제값', alpha=0.7, linewidth=2)
        for model_name, result in results.items():
            ax1.plot(result['y_pred'][:100], label=f'{model_name} 예측', alpha=0.7, linestyle='--')
        ax1.axhline(y=300, color='red', linestyle=':', label='위험 임계값 (300)')
        ax1.set_title('시계열 예측 비교 (처음 100개)')
        ax1.set_xlabel('시간 인덱스')
        ax1.set_ylabel('HUBROOM 반송량')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 산점도
        ax2 = axes[0, 1]
        colors = ['blue', 'green', 'orange']
        for idx, (model_name, result) in enumerate(results.items()):
            ax2.scatter(result['y_true'], result['y_pred'], 
                       alpha=0.5, label=model_name, color=colors[idx % len(colors)])
        ax2.plot([0, 600], [0, 600], 'r--', alpha=0.5, label='Perfect Prediction')
        ax2.axvline(x=300, color='red', linestyle=':', alpha=0.5)
        ax2.axhline(y=300, color='red', linestyle=':', alpha=0.5)
        ax2.set_title('예측값 vs 실제값')
        ax2.set_xlabel('실제값')
        ax2.set_ylabel('예측값')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 오차 분포
        ax3 = axes[1, 0]
        for model_name, result in results.items():
            errors = result['y_pred'] - result['y_true']
            ax3.hist(errors, bins=50, alpha=0.5, label=f'{model_name} (MAE: {result["mae"]:.2f})')
        ax3.set_title('예측 오차 분포')
        ax3.set_xlabel('예측 오차')
        ax3.set_ylabel('빈도')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 성능 메트릭 비교
        ax4 = axes[1, 1]
        metrics = ['MAE', 'RMSE', 'R²', 'MAE@300+']
        model_names = list(results.keys())
        
        x = np.arange(len(metrics))
        width = 0.35
        
        for idx, model_name in enumerate(model_names):
            values = [
                results[model_name]['mae'],
                results[model_name]['rmse'],
                results[model_name]['r2'] * 100,  # R²를 백분율로
                results[model_name]['mae_300']
            ]
            ax4.bar(x + idx * width, values, width, label=model_name)
        
        ax4.set_title('모델 성능 메트릭 비교')
        ax4.set_xlabel('메트릭')
        ax4.set_ylabel('값')
        ax4.set_xticks(x + width / 2)
        ax4.set_xticklabels(metrics)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('hubroom_evaluation_202509.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 300 이상 예측 상세 분석
        self.analyze_critical_predictions(results)
    
    def analyze_critical_predictions(self, results):
        """300 이상 예측 상세 분석"""
        print("\n" + "="*60)
        print("🚨 300 이상 예측 상세 분석")
        print("="*60)
        
        for model_name, result in results.items():
            print(f"\n📊 {model_name} 모델:")
            
            y_true = result['y_true']
            y_pred = result['y_pred']
            
            # 300 이상 케이스 분석
            true_over_300 = y_true >= 300
            pred_over_300 = y_pred >= 300
            
            # 혼동 행렬
            tp = np.sum(true_over_300 & pred_over_300)  # True Positive
            fp = np.sum(~true_over_300 & pred_over_300)  # False Positive
            tn = np.sum(~true_over_300 & ~pred_over_300)  # True Negative
            fn = np.sum(true_over_300 & ~pred_over_300)  # False Negative
            
            # 메트릭 계산
            if tp + fp > 0:
                precision = tp / (tp + fp)
            else:
                precision = 0
                
            if tp + fn > 0:
                recall = tp / (tp + fn)
            else:
                recall = 0
                
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0
            
            print(f"  - True Positive (정확히 예측한 위험): {tp}개")
            print(f"  - False Positive (잘못된 경보): {fp}개")
            print(f"  - True Negative (정확히 예측한 정상): {tn}개")
            print(f"  - False Negative (놓친 위험): {fn}개")
            print(f"  - Precision (정밀도): {precision:.2%}")
            print(f"  - Recall (재현율): {recall:.2%}")
            print(f"  - F1-Score: {f1:.2%}")
            
            # 극단값 분석
            extreme_cases = y_true > 400
            if np.sum(extreme_cases) > 0:
                extreme_mae = mean_absolute_error(y_true[extreme_cases], y_pred[extreme_cases])
                print(f"  - 400 초과 극단값 MAE: {extreme_mae:.2f}")

def main():
    """메인 실행 함수"""
    print("="*80)
    print("🏭 HUBROOM 반송량 예측 평가 시스템")
    print("📅 대상: 2025년 9월 데이터")
    print("🔧 버그 수정: 정확한 인덱싱으로 10분 후 예측")
    print("="*80)
    
    # TensorFlow 로그 레벨 조정
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # 평가기 생성
    evaluator = HUBROOMEvaluator()
    
    # 스케일러 확인
    if evaluator.scaler_X is None:
        print("\n❌ 스케일러가 없습니다. save_scalers.py를 먼저 실행하세요!")
        return
    
    try:
        # 1. 데이터 준비
        df, numeric_cols = evaluator.prepare_data()
        
        # 2. 시퀀스 생성 (수정된 버전)
        X, y_actual, physics_data, time_info = evaluator.create_evaluation_sequences(df, numeric_cols)
        
        # 3. 모델 로드
        models = evaluator.load_models()
        
        if not models:
            print("\n❌ 로드된 모델이 없습니다!")
            print("💡 학습된 모델 파일이 ./checkpoints/ 디렉토리에 있는지 확인하세요:")
            print("   - PatchTST_best.h5")
            print("   - PatchTST_PINN_best.h5")
            return
        
        # 4. 예측 및 평가
        results = evaluator.predict_and_evaluate(models, X, y_actual, physics_data, time_info)
        
        if not results:
            print("\n❌ 예측 결과가 없습니다!")
            return
        
        # 5. 결과 시각화
        evaluator.visualize_results(results)
        
        # 6. 예측 결과 저장 (수정된 형식)
        evaluator.save_predictions(results)
        
        # 7. 최종 요약
        print("\n" + "="*80)
        print("📊 최종 모델 비교 요약")
        print("="*80)
        
        print(f"\n{'모델':<20} {'MAE':<10} {'RMSE':<10} {'R²':<10} {'300+ 정확도':<15}")
        print("-"*65)
        
        for model_name, result in results.items():
            print(f"{model_name:<20} {result['mae']:<10.2f} {result['rmse']:<10.2f} "
                  f"{result['r2']:<10.2f} {result['acc_300']:<15.2%}")
        
        print("\n✅ 평가 완료!")
        print("📁 predictions_202509.csv 파일이 생성되었습니다.")
        print("\n📌 데이터 구조:")
        print("   - timestamp: 현재 시간 (입력의 마지막 시점)")
        print("   - current_value: 현재 시점의 HUBROOM 값")
        print("   - actual: 10분 후 실제값")
        print("   - predicted: 10분 후 예측값")
        print("   - input_max/min: 과거 20분 데이터의 최대/최소값")
        print("   - predicted_Target_time: 예측 타겟 시간 (10분 후)")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()