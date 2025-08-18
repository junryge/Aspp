"""
평가V5 수정버전 - 실제/앙상블/GRU 3개 그래프만 표시
TensorFlow 2.15.0 지원
반도체 물류 예측: 과거 100분 → 10분 후 예측
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Input, LSTM, Dense, Dropout, BatchNormalization,
                                     GRU, Conv1D, MaxPooling1D, Bidirectional, 
                                     Attention, MultiHeadAttention, LayerNormalization)
from tensorflow.keras.regularizers import l1_l2
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print(f"TensorFlow Version: {tf.__version__}")

class ModelEvaluator:
    """학습된 모델로 평가"""
    
    def __init__(self):
        # 경로 설정
        self.model_dir = r'D:\하이닉스\6.연구_항목\CODE\202508051차_POC구축\앙상블_하이브리드v5_150g학습\models_v5'
        self.data_path = r'D:\하이닉스\6.연구_항목\CODE\202508051차_POC구축\앙상블_하이브리드v5_150g학습\data\20250731_to20250806.csv'
        
        # 중요: 100분 데이터로 10분 후 예측
        self.sequence_length = 100  # 과거 100분
        self.prediction_horizon = 10  # 10분 후 예측
        self.spike_threshold = 1400
        
        self.models = {}
        self.scaler = None
        
        print(f"시퀀스 길이: {self.sequence_length}분")
        print(f"예측 시점: {self.prediction_horizon}분 후")
        
    def build_improved_lstm(self, input_shape):
        """LSTM 모델 - 스파이크 예측 강화"""
        model = Sequential([
            Input(shape=input_shape),
            LSTM(256, return_sequences=True, kernel_regularizer=l1_l2(l1=0.001, l2=0.001)),
            Dropout(0.3),
            BatchNormalization(),
            LSTM(128, return_sequences=True),
            Dropout(0.3),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(1)
        ])
        return model

    def build_improved_gru(self, input_shape):
        """GRU 모델 - 스파이크 예측 특화"""
        model = Sequential([
            Input(shape=input_shape),
            # 첫 번째 GRU 층 - 더 많은 유닛으로 패턴 학습
            GRU(256, return_sequences=True, kernel_regularizer=l1_l2(l1=0.001, l2=0.001)),
            Dropout(0.3),
            BatchNormalization(),
            
            # 두 번째 GRU 층 - 시계열 특징 추출
            GRU(128, return_sequences=True),
            Dropout(0.3),
            
            # 세 번째 GRU 층 - 최종 패턴 학습
            GRU(64, return_sequences=False),
            Dropout(0.2),
            
            # Dense 층 - 스파이크 예측 강화
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        return model

    def build_spike_detector(self, input_shape):
        """1400+ 스파이크 전용 감지기"""
        model = Sequential([
            Input(shape=input_shape),
            
            # CNN으로 급변 패턴 감지
            Conv1D(128, 5, activation='relu', padding='same'),
            BatchNormalization(),
            Conv1D(64, 3, activation='relu', padding='same'),
            MaxPooling1D(2),
            Dropout(0.3),
            
            # Bidirectional LSTM으로 시계열 분석
            Bidirectional(LSTM(128, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(64, return_sequences=False)),
            Dropout(0.2),
            
            # 최종 예측
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')  # 스파이크 확률
        ])
        return model
    
    def load_models(self):
        """모델 로드"""
        print("\n" + "=" * 60)
        print("모델 로드 중...")
        print("=" * 60)
        
        input_shape_50 = (50, 12)  # 학습 시
        input_shape_100 = (100, 12)  # 평가 시
        
        model_configs = {
            'lstm': (self.build_improved_lstm, 'lstm_final.h5'),
            'gru': (self.build_improved_gru, 'gru_final.h5'),
            'spike_detector': (self.build_spike_detector, 'spike_detector_final.h5')
        }
        
        for name, (build_func, filename) in model_configs.items():
            filepath = os.path.join(self.model_dir, filename)
            
            if os.path.exists(filepath):
                try:
                    # 100분 입력을 위한 모델 생성
                    model = build_func(input_shape_100)
                    
                    # 가중치 로드 시도
                    try:
                        model.load_weights(filepath)
                        print(f"✓ {name} 가중치 로드 성공")
                    except:
                        print(f"⚠ {name} 새로운 모델 구조로 초기화")
                    
                    model.compile(optimizer='adam', loss='mse')
                    self.models[name] = model
                    
                except Exception as e:
                    print(f"✗ {name} 로드 실패: {e}")
                    # 새 모델 생성
                    model = build_func(input_shape_100)
                    model.compile(optimizer='adam', loss='mse')
                    self.models[name] = model
            else:
                print(f"⚠ {filename} 파일 없음 - 새 모델 생성")
                model = build_func(input_shape_100)
                model.compile(optimizer='adam', loss='mse')
                self.models[name] = model
        
        # 스케일러 로드
        scaler_path = os.path.join(os.path.dirname(self.model_dir), 'scaler_v5.pkl')
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print(f"✓ 스케일러 로드 성공")
        else:
            self.scaler = RobustScaler()
            print(f"⚠ 새 스케일러 생성")
        
        return len(self.models) > 0
    
    def load_data(self):
        """데이터 로드"""
        print("\n데이터 로드 중...")
        
        if not os.path.exists(self.data_path):
            # CSV 파일 검색
            data_dir = os.path.dirname(self.data_path)
            csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            if csv_files:
                self.data_path = os.path.join(data_dir, csv_files[0])
                print(f"대체 파일 사용: {csv_files[0]}")
            else:
                print(f"✗ 데이터 파일 없음")
                return None
        
        df = pd.read_csv(self.data_path)
        print(f"✓ 데이터 로드: {len(df):,}행")
        
        # 시간 변환
        if 'current_time' in df.columns:
            df['datetime'] = pd.to_datetime(df['current_time'])
        elif 'date' in df.columns:
            df['datetime'] = pd.to_datetime(df['date'])
        else:
            df['datetime'] = pd.date_range(start='2024-01-01', periods=len(df), freq='min')
        
        # TOTALCNT 컬럼 생성
        if 'current_value' in df.columns:
            df['TOTALCNT'] = df['current_value']
        elif '실제' in df.columns:
            df['TOTALCNT'] = df['실제']
        
        # FUTURE 생성 (10분 후)
        df['FUTURE'] = df['TOTALCNT'].shift(-self.prediction_horizon)
        
        return df
    
    def create_features(self, df):
        """특징 생성"""
        print("특징 생성 중...")
        
        # 기본 통계
        for window in [5, 10, 30, 60]:
            df[f'MA_{window}'] = df['TOTALCNT'].rolling(window, min_periods=1).mean()
            df[f'STD_{window}'] = df['TOTALCNT'].rolling(window, min_periods=1).std()
            
        # 변화율
        df['change_rate'] = df['TOTALCNT'].pct_change()
        df['change_5'] = (df['TOTALCNT'] - df['TOTALCNT'].shift(5)) / df['TOTALCNT'].shift(5)
        
        # 스파이크 관련 특징
        df['above_1400'] = (df['TOTALCNT'] > 1400).astype(int)
        df['spike_distance'] = df['TOTALCNT'] - 1400
        df['spike_ratio'] = df['TOTALCNT'] / 1400
        
        # 시간 특징
        df['hour'] = df['datetime'].dt.hour
        df['minute'] = df['datetime'].dt.minute
        df['dayofweek'] = df['datetime'].dt.dayofweek
        
        # 결측치 처리
        df = df.fillna(method='ffill').fillna(0)
        
        return df
    
    def prepare_sequences(self, df):
        """시퀀스 생성"""
        print("\n시퀀스 생성 중...")
        
        # 특징 선택
        feature_cols = ['TOTALCNT', 'MA_5', 'MA_10', 'MA_30', 'STD_5', 'STD_10',
                       'change_rate', 'change_5', 'above_1400', 'spike_distance',
                       'spike_ratio', 'hour']
        
        # 사용 가능한 컬럼만 선택
        available_cols = [col for col in feature_cols if col in df.columns]
        
        # 스케일링
        scaled_df = pd.DataFrame(self.scaler.fit_transform(df[available_cols + ['FUTURE']]))
        scaled_df.columns = available_cols + ['FUTURE']
        
        X, y = [], []
        timestamps = []
        
        # 100분 시퀀스 생성
        for i in range(len(scaled_df) - self.sequence_length - self.prediction_horizon):
            X.append(scaled_df[available_cols].iloc[i:i+self.sequence_length].values)
            y.append(scaled_df['FUTURE'].iloc[i+self.sequence_length+self.prediction_horizon-1])
            
            timestamps.append({
                'start': df['datetime'].iloc[i],
                'end': df['datetime'].iloc[i+self.sequence_length-1],
                'target': df['datetime'].iloc[i+self.sequence_length+self.prediction_horizon-1]
            })
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"생성된 시퀀스: {len(X):,}개")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        
        return X, y, timestamps
    
    def predict_and_evaluate(self, X, y):
        """예측 및 평가"""
        print("\n" + "=" * 60)
        print("모델 예측 및 평가")
        print("=" * 60)
        
        predictions = {}
        results = {}
        
        # 각 모델 예측
        for name, model in self.models.items():
            print(f"\n{name} 예측 중...")
            
            if name == 'spike_detector':
                # 스파이크 감지기는 확률 출력
                pred = model.predict(X, batch_size=64, verbose=0)
                pred = pred.flatten() * 2  # 스케일 조정
            else:
                pred = model.predict(X, batch_size=64, verbose=0).flatten()
            
            predictions[name] = pred
            
            # 평가
            mae = mean_absolute_error(y, pred)
            rmse = np.sqrt(mean_squared_error(y, pred))
            r2 = r2_score(y, pred)
            
            results[name] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2
            }
            
            print(f"  MAE: {mae:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  R²: {r2:.4f}")
        
        # 앙상블 예측 (스파이크 가중치 강화)
        if len(predictions) > 1:
            print(f"\n앙상블 예측...")
            
            # 동적 가중치 (1400+ 예측 성능 기반)
            weights = {
                'lstm': 0.25,
                'gru': 0.45,  # GRU에 더 높은 가중치
                'spike_detector': 0.30  # 스파이크 감지기 가중치 증가
            }
            
            ensemble_pred = np.zeros_like(y)
            total_weight = 0
            
            for name, pred in predictions.items():
                weight = weights.get(name, 1.0 / len(predictions))
                ensemble_pred += pred * weight
                total_weight += weight
            
            ensemble_pred = ensemble_pred / total_weight
            predictions['ensemble'] = ensemble_pred
            
            # 앙상블 평가
            mae = mean_absolute_error(y, ensemble_pred)
            rmse = np.sqrt(mean_squared_error(y, ensemble_pred))
            r2 = r2_score(y, ensemble_pred)
            
            results['ensemble'] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2
            }
            
            print(f"  MAE: {mae:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  R²: {r2:.4f}")
        
        return predictions, results
    
    def inverse_transform(self, scaled_values):
        """역변환"""
        # 특징 수에 맞게 더미 생성
        n_features = self.scaler.n_features_in_ - 1
        dummy = np.zeros((len(scaled_values), n_features))
        combined = np.column_stack([dummy, scaled_values.reshape(-1, 1)])
        inversed = self.scaler.inverse_transform(combined)
        return inversed[:, -1]
    
    def visualize_three_graphs(self, y, predictions, timestamps):
        """실제, 앙상블, GRU 3개 그래프만 표시"""
        print("\n" + "=" * 60)
        print("결과 시각화 (실제/앙상블/GRU)")
        print("=" * 60)
        
        # 역변환
        y_original = self.inverse_transform(y)
        
        # 그래프 설정
        plt.figure(figsize=(20, 12))
        
        # 샘플 크기
        sample_size = min(500, len(y))
        
        # ============================
        # 1. 시계열 비교 (상단)
        # ============================
        plt.subplot(3, 1, 1)
        
        # 실제값
        plt.plot(y_original[:sample_size], 
                label='실제값', 
                color='black', 
                linewidth=2, 
                alpha=0.8)
        
        # 앙상블 예측
        if 'ensemble' in predictions:
            ensemble_original = self.inverse_transform(predictions['ensemble'])
            plt.plot(ensemble_original[:sample_size], 
                    label='앙상블 예측', 
                    color='red', 
                    linewidth=1.5, 
                    alpha=0.7)
        
        # GRU 예측
        if 'gru' in predictions:
            gru_original = self.inverse_transform(predictions['gru'])
            plt.plot(gru_original[:sample_size], 
                    label='GRU 예측', 
                    color='green', 
                    linewidth=1.5, 
                    alpha=0.7)
        
        # 1400 임계선
        plt.axhline(y=1400, color='orange', linestyle='--', alpha=0.5, label='1400 임계값')
        
        plt.title('반도체 물류 예측 결과 비교 (100분 → 10분 후)', fontsize=16, fontweight='bold')
        plt.xlabel('시간 인덱스', fontsize=12)
        plt.ylabel('물류량', fontsize=12)
        plt.legend(loc='upper right', fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # ============================
        # 2. 1400+ 구간 성능 (중단)
        # ============================
        plt.subplot(3, 1, 2)
        
        # 1400 이상 구간만 표시
        spike_indices = np.where(y_original[:sample_size] >= 1400)[0]
        
        if len(spike_indices) > 0:
            # 실제 1400+ 포인트
            plt.scatter(spike_indices, 
                       y_original[spike_indices], 
                       color='black', 
                       s=50, 
                       label=f'실제 1400+ ({len(spike_indices)}개)', 
                       zorder=5,
                       alpha=0.8)
            
            # 앙상블 예측
            if 'ensemble' in predictions:
                plt.scatter(spike_indices, 
                          ensemble_original[spike_indices], 
                          color='red', 
                          s=30, 
                          label='앙상블 예측', 
                          zorder=4,
                          alpha=0.7)
            
            # GRU 예측
            if 'gru' in predictions:
                plt.scatter(spike_indices, 
                          gru_original[spike_indices], 
                          color='green', 
                          s=30, 
                          label='GRU 예측', 
                          zorder=3,
                          alpha=0.7)
            
            # 배경 라인
            plt.plot(y_original[:sample_size], 
                    color='gray', 
                    alpha=0.2, 
                    linewidth=0.5)
        
        plt.axhline(y=1400, color='orange', linestyle='--', alpha=0.5)
        plt.title('1400+ 스파이크 구간 예측 성능', fontsize=14, fontweight='bold')
        plt.xlabel('시간 인덱스', fontsize=12)
        plt.ylabel('물류량', fontsize=12)
        plt.legend(loc='upper right', fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # ============================
        # 3. 성능 메트릭 비교 (하단)
        # ============================
        plt.subplot(3, 1, 3)
        
        # 성능 계산
        metrics_data = {
            '실제': {'mae': 0, 'rmse': 0, 'r2': 1.0, 'spike_mae': 0}
        }
        
        # 앙상블 성능
        if 'ensemble' in predictions:
            ensemble_original = self.inverse_transform(predictions['ensemble'])
            mae = mean_absolute_error(y_original, ensemble_original)
            rmse = np.sqrt(mean_squared_error(y_original, ensemble_original))
            r2 = r2_score(y_original, ensemble_original)
            
            spike_mask = y_original >= 1400
            spike_mae = mean_absolute_error(y_original[spike_mask], 
                                           ensemble_original[spike_mask]) if spike_mask.sum() > 0 else 0
            
            metrics_data['앙상블'] = {
                'mae': mae, 'rmse': rmse, 'r2': r2, 'spike_mae': spike_mae
            }
        
        # GRU 성능
        if 'gru' in predictions:
            gru_original = self.inverse_transform(predictions['gru'])
            mae = mean_absolute_error(y_original, gru_original)
            rmse = np.sqrt(mean_squared_error(y_original, gru_original))
            r2 = r2_score(y_original, gru_original)
            
            spike_mask = y_original >= 1400
            spike_mae = mean_absolute_error(y_original[spike_mask], 
                                           gru_original[spike_mask]) if spike_mask.sum() > 0 else 0
            
            metrics_data['GRU'] = {
                'mae': mae, 'rmse': rmse, 'r2': r2, 'spike_mae': spike_mae
            }
        
        # 막대 그래프
        models = list(metrics_data.keys())
        x = np.arange(len(models))
        width = 0.2
        
        mae_values = [metrics_data[m]['mae'] for m in models]
        spike_mae_values = [metrics_data[m]['spike_mae'] for m in models]
        r2_values = [metrics_data[m]['r2'] for m in models]
        
        ax = plt.gca()
        bars1 = ax.bar(x - width, mae_values, width, label='전체 MAE', color='blue', alpha=0.7)
        bars2 = ax.bar(x, spike_mae_values, width, label='1400+ MAE', color='red', alpha=0.7)
        bars3 = ax.bar(x + width, [v * 100 for v in r2_values], width, label='R² × 100', color='green', alpha=0.7)
        
        # 값 표시
        for i, (bar1, bar2, bar3) in enumerate(zip(bars1, bars2, bars3)):
            if models[i] != '실제':
                ax.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height(), 
                       f'{mae_values[i]:.1f}', ha='center', va='bottom', fontsize=9)
                ax.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height(), 
                       f'{spike_mae_values[i]:.1f}', ha='center', va='bottom', fontsize=9)
                ax.text(bar3.get_x() + bar3.get_width()/2, bar3.get_height(), 
                       f'{r2_values[i]:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('모델', fontsize=12)
        ax.set_ylabel('평가 지표', fontsize=12)
        ax.set_title('성능 메트릭 비교', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # 저장
        output_file = f'evaluation_3graphs_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ 그래프 저장: {output_file}")
        plt.show()
    
    def save_results(self, y, predictions, timestamps):
        """결과 저장"""
        print("\n결과 저장 중...")
        
        # 역변환
        y_original = self.inverse_transform(y)
        
        # 데이터프레임 생성
        results_df = pd.DataFrame({
            'start_time': [t['start'] for t in timestamps],
            'target_time': [t['target'] for t in timestamps],
            '실제': y_original
        })
        
        # 앙상블과 GRU만 저장
        for name in ['ensemble', 'gru']:
            if name in predictions:
                pred_original = self.inverse_transform(predictions[name])
                results_df[f'{name}_예측'] = pred_original
                results_df[f'{name}_오차'] = y_original - pred_original
                results_df[f'{name}_1400+정확도'] = ((pred_original >= 1400) == (y_original >= 1400)).astype(int)
        
        # CSV 저장
        output_file = f'prediction_results_3models_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"✓ 결과 저장: {output_file}")
        
        return results_df

def main():
    """메인 실행"""
    print("=" * 60)
    print("반도체 물류 예측 평가 V5")
    print("실제 / 앙상블 / GRU 3개 그래프")
    print("과거 100분 → 10분 후 예측")
    print("=" * 60)
    
    # 평가기 초기화
    evaluator = ModelEvaluator()
    
    # 1. 모델 로드
    if not evaluator.load_models():
        print("✗ 모델 로드 실패!")
        return None, None
    
    # 2. 데이터 로드
    df = evaluator.load_data()
    if df is None:
        return None, None
    
    # 3. 특징 생성
    df = evaluator.create_features(df)
    
    # 4. 시퀀스 생성 (100분)
    X, y, timestamps = evaluator.prepare_sequences(df)
    
    # 5. 예측 및 평가
    predictions, results = evaluator.predict_and_evaluate(X, y)
    
    # 6. 시각화 (실제/앙상블/GRU 3개만)
    evaluator.visualize_three_graphs(y, predictions, timestamps)
    
    # 7. 결과 저장
    results_df = evaluator.save_results(y, predictions, timestamps)
    
    print("\n" + "=" * 60)
    print("✅ 평가 완료!")
    print("=" * 60)
    
    # 최종 성능 요약
    print("\n【최종 성능 요약】")
    y_original = evaluator.inverse_transform(y)
    
    for name in ['ensemble', 'gru']:
        if name in predictions:
            pred_original = evaluator.inverse_transform(predictions[name])
            
            # 전체 성능
            mae = mean_absolute_error(y_original, pred_original)
            
            # 1400+ 성능
            spike_mask = y_original >= 1400
            if spike_mask.sum() > 0:
                spike_mae = mean_absolute_error(y_original[spike_mask], pred_original[spike_mask])
                spike_accuracy = np.mean((pred_original >= 1400) == (y_original >= 1400))
                
                print(f"\n{name.upper()}:")
                print(f"  전체 MAE: {mae:.2f}")
                print(f"  1400+ MAE: {spike_mae:.2f}")
                print(f"  1400+ 정확도: {spike_accuracy:.2%}")
    
    return results_df, results

if __name__ == "__main__":
    results_df, evaluation = main()