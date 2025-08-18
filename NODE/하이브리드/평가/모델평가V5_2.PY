"""
학습된 모델을 로드하여 평가 수행
20250731_to20250806.csv의 TOTALCNT를 과거 100분으로 10분 후 예측
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Input, LSTM, Dense, Dropout, BatchNormalization,
                                     GRU, Conv1D, MaxPooling1D, Bidirectional)
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
        """LSTM 모델 (학습 코드와 동일한 구조)"""
        model = Sequential([
            Input(shape=input_shape),
            LSTM(128, return_sequences=True, kernel_regularizer=l1_l2(l1=0.005, l2=0.005)),
            Dropout(0.4),
            BatchNormalization(),
            LSTM(64, return_sequences=True),
            Dropout(0.4),
            LSTM(32, return_sequences=False),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        return model

    def build_improved_gru(self, input_shape):
        """GRU 모델"""
        model = Sequential([
            Input(shape=input_shape),
            GRU(128, return_sequences=True, kernel_regularizer=l1_l2(l1=0.005, l2=0.005)),
            Dropout(0.4),
            GRU(64, return_sequences=True),
            Dropout(0.4),
            GRU(32, return_sequences=False),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        return model

    def build_improved_cnn_lstm(self, input_shape):
        """CNN-LSTM 모델"""
        model = Sequential([
            Input(shape=input_shape),
            Conv1D(64, 3, activation='relu', padding='same'),
            BatchNormalization(),
            Conv1D(64, 3, activation='relu', padding='same'),
            MaxPooling1D(2),
            Dropout(0.3),
            LSTM(64, return_sequences=True),
            Dropout(0.4),
            LSTM(32, return_sequences=False),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        return model

    def build_improved_spike_detector(self, input_shape):
        """급변 감지기"""
        model = Sequential([
            Input(shape=input_shape),
            Conv1D(64, 3, activation='relu', padding='same'),
            BatchNormalization(),
            Conv1D(64, 3, activation='relu', padding='same'),
            MaxPooling1D(2),
            Dropout(0.3),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.4),
            Bidirectional(LSTM(32, return_sequences=False)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        return model
    
    def load_models(self):
        """모델 로드"""
        print("\n" + "=" * 60)
        print("모델 로드 중...")
        print("=" * 60)
        
        # 학습 시 사용한 input_shape (50분 시퀀스, 12개 특징)
        # 하지만 평가는 100분으로 할 것임
        input_shape_50 = (50, 12)  # 학습 시
        input_shape_100 = (100, 12)  # 평가 시
        
        model_configs = {
            'lstm': (self.build_improved_lstm, 'lstm_final.h5'),
            'gru': (self.build_improved_gru, 'gru_final.h5'),
            'cnn_lstm': (self.build_improved_cnn_lstm, 'cnn_lstm_final.h5'),
            'spike_detector': (self.build_improved_spike_detector, 'spike_detector_final.h5')
        }
        
        for name, (build_func, filename) in model_configs.items():
            filepath = os.path.join(self.model_dir, filename)
            
            if os.path.exists(filepath):
                try:
                    # 모델 구조 생성 (학습 시와 동일한 50으로)
                    model = build_func(input_shape_50)
                    
                    # 가중치 로드
                    model.load_weights(filepath)
                    
                    # 100분 입력을 위한 새 모델 생성
                    new_model = build_func(input_shape_100)
                    
                    # 가중치 복사 (레이어별로)
                    for i, layer in enumerate(model.layers):
                        if layer.get_weights():
                            new_model.layers[i].set_weights(layer.get_weights())
                    
                    # 컴파일
                    new_model.compile(
                        optimizer='adam',
                        loss='mae',
                        metrics=['mae']
                    )
                    
                    self.models[name] = new_model
                    print(f"✓ {name} 모델 로드 완료")
                    
                except Exception as e:
                    print(f"✗ {name} 모델 로드 실패: {e}")
            else:
                print(f"✗ {name} 모델 파일 없음")
        
        # 스케일러 로드
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            try:
                self.scaler = joblib.load(scaler_path)
                print(f"✓ 스케일러 로드 완료")
            except:
                print(f"⚠ 스케일러 로드 실패, 새로 생성")
                self.scaler = RobustScaler()
        else:
            self.scaler = RobustScaler()
            print(f"⚠ 스케일러 새로 생성")
        
        print()
        return len(self.models) > 0
    
    def load_data(self):
        """데이터 로드"""
        print("데이터 로드 중...")
        
        # CSV 로드
        df = pd.read_csv(self.data_path, encoding='utf-8')
        
        print(f"데이터 shape: {df.shape}")
        print(f"컬럼: {df.columns.tolist()}")
        
        # 시간 처리
        if 'CURRTIME' in df.columns:
            df['datetime'] = pd.to_datetime(df['CURRTIME'], format='%Y%m%d%H%M')
        
        # TOTALCNT 확인
        if 'TOTALCNT' not in df.columns:
            print("✗ TOTALCNT 컬럼이 없습니다!")
            return None
            
        print(f"TOTALCNT 범위: {df['TOTALCNT'].min()} ~ {df['TOTALCNT'].max()}")
        print(f"데이터 기간: {df['datetime'].min()} ~ {df['datetime'].max()}")
        
        return df
    
    def create_features(self, df):
        """특징 생성 (학습 코드와 동일)"""
        print("\n특징 생성 중...")
        
        df = df.copy()
        
        # 시간 특징
        df['hour'] = df['datetime'].dt.hour
        df['dayofweek'] = df['datetime'].dt.dayofweek
        df['is_weekend'] = (df['datetime'].dt.dayofweek >= 5).astype(int)
        
        # 이동평균
        df['MA_10'] = df['TOTALCNT'].rolling(10, min_periods=1).mean()
        df['MA_30'] = df['TOTALCNT'].rolling(30, min_periods=1).mean()
        df['MA_60'] = df['TOTALCNT'].rolling(60, min_periods=1).mean()
        
        # 표준편차
        df['STD_10'] = df['TOTALCNT'].rolling(10, min_periods=1).std().fillna(0)
        df['STD_30'] = df['TOTALCNT'].rolling(30, min_periods=1).std().fillna(0)
        
        # 변화율
        df['change_rate'] = df['TOTALCNT'].pct_change().fillna(0)
        df['change_rate_10'] = df['TOTALCNT'].pct_change(10).fillna(0)
        
        # 트렌드
        df['trend'] = df['MA_10'] - df['MA_30']
        
        # NaN 처리
        df.ffill(inplace=True)
        df.fillna(0, inplace=True)
        
        return df
    
    def prepare_sequences(self, df):
        """100분 시퀀스 생성"""
        print("\n시퀀스 생성 중 (100분 -> 10분 후)...")
        
        # 특징 컬럼 (학습과 동일)
        feature_cols = ['TOTALCNT', 'MA_10', 'MA_30', 'MA_60', 'STD_10', 'STD_30',
                       'change_rate', 'change_rate_10', 'hour', 'dayofweek', 
                       'is_weekend', 'trend']
        
        # 실제값 생성 (10분 후)
        df['FUTURE'] = df['TOTALCNT'].shift(-self.prediction_horizon)
        
        # 스케일링
        scaled_data = self.scaler.fit_transform(df[feature_cols + ['FUTURE']].fillna(0))
        scaled_df = pd.DataFrame(scaled_data, columns=feature_cols + ['FUTURE'])
        
        X, y = [], []
        timestamps = []
        
        # 100분 시퀀스 생성
        for i in range(len(scaled_df) - self.sequence_length - self.prediction_horizon):
            # 과거 100분
            X.append(scaled_df[feature_cols].iloc[i:i+self.sequence_length].values)
            # 10분 후 값
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
                # spike_detector는 sigmoid 출력
                pred = model.predict(X, batch_size=256, verbose=0)
                pred = pred.flatten()
            else:
                pred = model.predict(X, batch_size=256, verbose=0).flatten()
            
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
        
        # 앙상블 예측
        if len(predictions) > 1:
            print(f"\n앙상블 예측...")
            
            # 가중 평균 (GRU 중심)
            weights = {
                'lstm': 0.2,
                'gru': 0.4,
                'cnn_lstm': 0.25,
                'spike_detector': 0.15
            }
            
            ensemble_pred = np.zeros_like(y)
            total_weight = 0
            
            for name, pred in predictions.items():
                if name in weights:
                    weight = weights[name]
                else:
                    weight = 1.0 / len(predictions)
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
        # 12개 특징 + 1개 타겟
        dummy = np.zeros((len(scaled_values), 12))
        combined = np.column_stack([dummy, scaled_values.reshape(-1, 1)])
        inversed = self.scaler.inverse_transform(combined)
        return inversed[:, -1]
    
    def evaluate_final(self, y, predictions):
        """최종 평가 (원본 스케일)"""
        print("\n" + "=" * 60)
        print("최종 평가 (원본 스케일)")
        print("=" * 60)
        
        # 역변환
        y_original = self.inverse_transform(y)
        
        for name, pred in predictions.items():
            pred_original = self.inverse_transform(pred)
            
            # 전체 평가
            mae = mean_absolute_error(y_original, pred_original)
            rmse = np.sqrt(mean_squared_error(y_original, pred_original))
            r2 = r2_score(y_original, pred_original)
            
            # 1400+ 평가
            spike_mask = y_original >= self.spike_threshold
            if spike_mask.sum() > 0:
                spike_mae = mean_absolute_error(y_original[spike_mask], pred_original[spike_mask])
                
                # 급증 감지
                pred_spike = pred_original >= self.spike_threshold
                actual_spike = y_original >= self.spike_threshold
                
                tp = np.sum((pred_spike) & (actual_spike))
                fp = np.sum((pred_spike) & (~actual_spike))
                fn = np.sum((~pred_spike) & (actual_spike))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            else:
                spike_mae = precision = recall = f1 = 0
            
            print(f"\n{name.upper()}:")
            print(f"  전체 MAE: {mae:.2f}")
            print(f"  전체 RMSE: {rmse:.2f}")
            print(f"  R²: {r2:.4f}")
            print(f"  1400+ MAE: {spike_mae:.2f}")
            print(f"  Precision: {precision:.2%}")
            print(f"  Recall: {recall:.2%}")
            print(f"  F1: {f1:.4f}")
            
            # 최고 모델 저장
            if name == 'ensemble':
                self.best_pred = pred_original
                self.best_actual = y_original
    
    def visualize(self, y, predictions, timestamps):
        """시각화 - 실제, 앙상블, GRU 3개만"""
        print("\n결과 시각화 생성 중...")
        
        # 역변환
        y_original = self.inverse_transform(y)
        
        # 그래프 크기 설정
        plt.figure(figsize=(20, 8))
        
        # 샘플 크기
        sample = min(500, len(y))
        
        # 실제값 (검정)
        plt.plot(y_original[:sample], 
                label='실제', 
                color='black', 
                linewidth=2)
        
        # 앙상블 예측 (빨강)
        if 'ensemble' in predictions:
            pred_original = self.inverse_transform(predictions['ensemble'])
            plt.plot(pred_original[:sample], 
                    label='앙상블', 
                    color='red', 
                    linewidth=1.5,
                    alpha=0.8)
        
        # GRU 예측 (초록)
        if 'gru' in predictions:
            pred_original = self.inverse_transform(predictions['gru'])
            plt.plot(pred_original[:sample], 
                    label='GRU', 
                    color='green', 
                    linewidth=1.5,
                    alpha=0.8)
        
        # 1400 임계선
        plt.axhline(y=1400, color='orange', linestyle='--', alpha=0.5, label='1400 임계값')
        
        plt.title('반도체 물류 예측 결과 (100분 → 10분 후)', fontsize=16, fontweight='bold')
        plt.xlabel('시간 인덱스', fontsize=12)
        plt.ylabel('물류량', fontsize=12)
        plt.legend(loc='upper right', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('evaluation_3graphs.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ 시각화 저장: evaluation_3graphs.png")
    
    def save_results(self, y, predictions, timestamps):
        """결과 저장"""
        print("\n결과 저장 중...")
        
        # 역변환
        y_original = self.inverse_transform(y)
        
        # 데이터프레임 생성
        results_df = pd.DataFrame({
            'start_time': [t['start'] for t in timestamps],
            'target_time': [t['target'] for t in timestamps],
            '실제값': y_original
        })
        
        # 예측값 추가
        for name, pred in predictions.items():
            pred_original = self.inverse_transform(pred)
            results_df[f'{name}_예측'] = pred_original
            results_df[f'{name}_오차'] = y_original - pred_original
        
        # CSV 저장
        output_file = f'prediction_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"✓ 결과 저장: {output_file}")
        
        return results_df

def main():
    """메인 실행"""
    print("=" * 60)
    print("반도체 물류 예측 평가")
    print("과거 100분 데이터로 10분 후 예측")
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
    
    # 6. 최종 평가 (원본 스케일)
    evaluator.evaluate_final(y, predictions)
    
    # 7. 시각화
    evaluator.visualize(y, predictions, timestamps)
    
    # 8. 결과 저장
    results_df = evaluator.save_results(y, predictions, timestamps)
    
    print("\n" + "=" * 60)
    print("✅ 평가 완료!")
    print("=" * 60)
    
    return results_df, results

if __name__ == "__main__":
    results_df, evaluation = main()