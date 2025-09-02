# -*- coding: utf-8 -*-
"""
HUBROOM 반송량 예측 평가 시스템 - PatchTST 단일 모델
Created on Mon Sep 1 15:13:16 2025
@author: X0163954
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention, Flatten, Embedding
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
# 🔧 모델 재구성 (PatchTST만)
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
    
    # Linear projection (dense)
    x = Dense(d_model, name='dense')(patches)
    
    # Positional encoding
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

# ===========================
# 📊 평가 클래스
# ===========================

class HUBROOMEvaluator:
    """HUBROOM 예측 모델 평가 클래스 - PatchTST 전용"""
    
    def __init__(self, data_path='20250801_to_20250831.csv'):
        self.data_path = data_path
        self.seq_len = 20
        self.pred_len = 10
        self.target_col = 'CURRENT_M16A_3F_JOB_2'
        self.critical_threshold = 300
        
        # 모델 경로 (PatchTST만)
        self.patchtst_weights = './checkpoints/PatchTST_best.h5'
        
        # 스케일러 로드
        self.scaler_X = self.load_scaler('scaler_X.pkl')
        self.scaler_y = self.load_scaler('scaler_y.pkl')
        
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
        """평가용 시퀀스 생성"""
        X_list = []
        y_actual_list = []
        
        # 시간 관련 정보 저장
        input_start_times = []
        input_end_times = []
        predicted_target_times = []
        
        # 입력 데이터 통계 정보
        input_max_values = []
        input_min_values = []
        
        data = df[numeric_cols].values
        target_idx = numeric_cols.index(self.target_col)
        
        print(f"\n📊 시퀀스 생성 중...")
        total_sequences = len(data) - self.seq_len - self.pred_len + 1
        
        # 시퀀스 생성
        for i in range(total_sequences):
            # 입력 시퀀스 (과거 20분)
            X_seq = data[i:i+self.seq_len]
            
            # 실제 값 (미래 10분)
            y_actual = data[i+self.seq_len:i+self.seq_len+self.pred_len, target_idx]
            
            # 시간 정보
            input_start = df['timestamp'].iloc[i]
            input_end = df['timestamp'].iloc[i+self.seq_len-1]
            target_time = df['timestamp'].iloc[i+self.seq_len+self.pred_len-1]  # 10분 후 시점
            
            # 입력 시퀀스의 타겟 컬럼 최대/최소값
            input_target_values = X_seq[:, target_idx]
            input_max = np.max(input_target_values)
            input_min = np.min(input_target_values)
            
            # 리스트에 추가
            X_list.append(X_seq)
            y_actual_list.append(y_actual)
            
            input_start_times.append(input_start)
            input_end_times.append(input_end)
            predicted_target_times.append(target_time)
            input_max_values.append(input_max)
            input_min_values.append(input_min)
        
        print(f"✅ 시퀀스 생성 완료: {len(X_list)}개")
        
        # 시간 정보와 통계 정보도 함께 반환
        time_info = {
            'input_start': input_start_times,
            'input_end': input_end_times,
            'predicted_target_time': predicted_target_times,
            'input_max': input_max_values,
            'input_min': input_min_values
        }
        
        return np.array(X_list), np.array(y_actual_list), time_info
    
    def load_model(self):
        """PatchTST 모델 재구성 및 가중치 로드"""
        print("\n🤖 PatchTST 모델 재구성 및 가중치 로드 중...")
        
        if not os.path.exists(self.patchtst_weights):
            print(f"❌ 모델 파일이 없습니다: {self.patchtst_weights}")
            return None
        
        try:
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
            print("✅ PatchTST 모델 로드 완료")
            
            return model
            
        except Exception as e:
            print(f"❌ PatchTST 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict_and_evaluate(self, model, X, y_actual, time_info):
        """PatchTST 예측 및 평가"""
        print(f"\n{'='*60}")
        print(f"📊 PatchTST 모델 평가")
        print(f"{'='*60}")
        
        # 데이터 정규화
        n_samples, seq_len, n_features = X.shape
        
        # 스케일러 확인
        if self.scaler_X is None or self.scaler_y is None:
            print("❌ 스케일러가 로드되지 않았습니다!")
            return None
        
        X_scaled = self.scaler_X.transform(X.reshape(-1, n_features)).reshape(n_samples, seq_len, n_features)
        
        try:
            # 예측
            print("예측 수행 중...")
            y_pred_scaled = model.predict(X_scaled, verbose=1, batch_size=32)
            
            # 모델 출력 형태 확인
            print(f"예측 출력 형태: {y_pred_scaled.shape}")
            
            # 역정규화 (10분 후 예측값)
            y_pred_10min = self.scaler_y.inverse_transform(
                y_pred_scaled.reshape(-1, 1)
            ).flatten()
            
            # 실제값은 10분 후 값만 추출
            y_true_10min = y_actual[:, -1]  # 마지막 시점 (10분 후)
            
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
            
            # 결과 저장
            result = {
                'y_true': y_true_10min,
                'y_pred': y_pred_10min,
                'time_info': time_info,
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
            
            # 샘플 출력
            print(f"\n📝 예측 샘플 (처음 10개):")
            for i in range(min(10, len(y_pred_10min))):
                status = "⚠️ 경고" if y_pred_10min[i] >= 300 else "✅ 정상"
                print(f"  [{i+1}] 시간: {time_info['predicted_target_time'][i]}, "
                      f"실제: {y_true_10min[i]:.1f}, 예측: {y_pred_10min[i]:.1f} {status}")
            
            return result
            
        except Exception as e:
            print(f"❌ 예측 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def visualize_results(self, result):
        """PatchTST 결과 시각화"""
        if result is None:
            print("시각화할 결과가 없습니다.")
            return
        
        # 4개 그래프 생성
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('PatchTST HUBROOM 반송량 예측 평가 (2025년 9월)', fontsize=16)
        
        # 1. 시계열 예측 비교 (처음 200개)
        ax1 = axes[0, 0]
        ax1.plot(result['y_true'][:200], label='실제값', alpha=0.7, linewidth=2, color='blue')
        ax1.plot(result['y_pred'][:200], label='PatchTST 예측', alpha=0.7, linestyle='--', color='orange')
        ax1.axhline(y=300, color='red', linestyle=':', label='위험 임계값 (300)')
        ax1.set_title('시계열 예측 비교 (처음 200개)')
        ax1.set_xlabel('시간 인덱스')
        ax1.set_ylabel('HUBROOM 반송량')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 산점도
        ax2 = axes[0, 1]
        ax2.scatter(result['y_true'], result['y_pred'], alpha=0.5, color='blue', s=10)
        ax2.plot([0, 600], [0, 600], 'r--', alpha=0.5, label='Perfect Prediction')
        ax2.axvline(x=300, color='red', linestyle=':', alpha=0.5)
        ax2.axhline(y=300, color='red', linestyle=':', alpha=0.5)
        ax2.set_title(f'예측값 vs 실제값 (R²={result["r2"]:.3f})')
        ax2.set_xlabel('실제값')
        ax2.set_ylabel('예측값')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 오차 분포
        ax3 = axes[1, 0]
        errors = result['y_pred'] - result['y_true']
        ax3.hist(errors, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax3.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax3.set_title(f'예측 오차 분포 (MAE={result["mae"]:.2f})')
        ax3.set_xlabel('예측 오차')
        ax3.set_ylabel('빈도')
        ax3.grid(True, alpha=0.3)
        
        # 오차 통계 텍스트 추가
        ax3.text(0.02, 0.95, f'평균: {np.mean(errors):.2f}\n표준편차: {np.std(errors):.2f}',
                transform=ax3.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 4. 성능 메트릭
        ax4 = axes[1, 1]
        metrics = ['MAE', 'RMSE', 'R²×100', 'MAE@300+']
        values = [
            result['mae'],
            result['rmse'],
            result['r2'] * 100,
            result['mae_300']
        ]
        
        bars = ax4.bar(metrics, values, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
        ax4.set_title('모델 성능 메트릭')
        ax4.set_ylabel('값')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 값 표시
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('patchtst_evaluation_202509.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n✅ 그래프 저장 완료: patchtst_evaluation_202509.png")
    
    def analyze_critical_predictions(self, result):
        """300 이상 예측 상세 분석"""
        print("\n" + "="*60)
        print("🚨 300 이상 예측 상세 분석")
        print("="*60)
        
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
            print(f"  - 400 초과 케이스 수: {np.sum(extreme_cases)}개")
    
    def save_predictions(self, result, output_path='predictions_202509.csv'):
        """예측 결과를 요청된 형식의 CSV로 저장"""
        print(f"\n💾 예측 결과 저장 중...")
        
        if result is None:
            print("❌ 저장할 결과가 없습니다!")
            return
        
        time_info = result['time_info']
        
        # 요청된 컬럼 형식으로 데이터프레임 생성
        df_results = pd.DataFrame({
            'timestamp': time_info['input_end'],  # 현재 시간 (입력 종료 시간)
            'actual': result['y_true'],  # 실제 관측값
            'predicted': result['y_pred'],  # PatchTST 예측값
            'predicted_Target_time': time_info['predicted_target_time'],  # 예측 타켓 시간 (10분 후)
            'input_start': time_info['input_start'],  # 입력 시작 시간
            'input_end': time_info['input_end'],  # 입력 종료 시간
            'input_max': time_info['input_max'],  # 입력 시퀀스 최대값
            'input_min': time_info['input_min'],  # 입력 시퀀스 최소값
            'error': result['y_true'] - result['y_pred'],  # actual - predicted
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
        
        # 처음 5개 행 출력
        print(f"\n📋 저장된 데이터 샘플 (처음 5개):")
        print(df_results.head())

def main():
    """메인 실행 함수"""
    print("="*80)
    print("🏭 HUBROOM 반송량 예측 평가 시스템 - PatchTST")
    print("📅 대상: 2025년 9월 데이터")
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
        
        # 2. 시퀀스 생성
        X, y_actual, time_info = evaluator.create_evaluation_sequences(df, numeric_cols)
        
        # 3. 모델 로드
        model = evaluator.load_model()
        
        if model is None:
            print("\n❌ 모델 로드 실패!")
            print("💡 학습된 모델 파일이 ./checkpoints/PatchTST_best.h5 에 있는지 확인하세요")
            return
        
        # 4. 예측 및 평가
        result = evaluator.predict_and_evaluate(model, X, y_actual, time_info)
        
        if result is None:
            print("\n❌ 예측 실패!")
            return
        
        # 5. 결과 시각화
        evaluator.visualize_results(result)
        
        # 6. 300 이상 상세 분석
        evaluator.analyze_critical_predictions(result)
        
        # 7. 예측 결과 저장
        evaluator.save_predictions(result)
        
        # 8. 최종 요약
        print("\n" + "="*80)
        print("📊 PatchTST 모델 최종 평가 요약")
        print("="*80)
        
        print(f"\n전체 성능:")
        print(f"  - MAE: {result['mae']:.2f}")
        print(f"  - RMSE: {result['rmse']:.2f}")
        print(f"  - R²: {result['r2']:.4f}")
        print(f"  - 300+ 감지 정확도: {result['acc_300']:.2%}")
        
        print("\n✅ 평가 완료!")
        print("📁 생성된 파일:")
        print("  - predictions_202509.csv (예측 결과)")
        print("  - patchtst_evaluation_202509.png (시각화)")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()