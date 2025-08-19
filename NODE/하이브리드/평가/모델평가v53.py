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
        """100분 시퀀스 생성 - 수정된 버전"""
        print("\n시퀀스 생성 중 (100분 -> 10분 후)...")
        
        # 특징 컬럼 (학습과 동일)
        feature_cols = ['TOTALCNT', 'MA_10', 'MA_30', 'MA_60', 'STD_10', 'STD_30',
                       'change_rate', 'change_rate_10', 'hour', 'dayofweek', 
                       'is_weekend', 'trend']
        
        # 스케일링 (FUTURE 컬럼 없이)
        features_for_scaling = df[feature_cols].fillna(0)
        scaled_features = self.scaler.fit_transform(features_for_scaling)
        
        X, y = [], []
        timestamps = []
        actuals = []  # 실제값 저장
        
        # 100분 시퀀스 생성
        for i in range(len(df) - self.sequence_length - self.prediction_horizon + 1):
            # 과거 100분 (i부터 i+99까지)
            X.append(scaled_features[i:i+self.sequence_length])
            
            # 10분 후 값 (i+99 시점에서 10분 후 = i+109)
            future_idx = i + self.sequence_length + self.prediction_horizon - 1
            if future_idx < len(df):
                # 실제 TOTALCNT 값을 스케일링
                actual_value = df['TOTALCNT'].iloc[future_idx]
                actuals.append(actual_value)
                
                # 스케일링된 값을 타겟으로
                scaled_value = (actual_value - self.scaler.center_[0]) / self.scaler.scale_[0]
                y.append(scaled_value)
                
                timestamps.append({
                    'start': df['datetime'].iloc[i],
                    'end': df['datetime'].iloc[i+self.sequence_length-1],
                    'target': df['datetime'].iloc[future_idx],
                    'actual_value': actual_value
                })
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"생성된 시퀀스: {len(X):,}개")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        
        # 검증: 시간 차이 확인
        if len(timestamps) > 0:
            time_diff = (timestamps[0]['target'] - timestamps[0]['end']).total_seconds() / 60
            print(f"예측 시간 차이 확인: {time_diff}분 (목표: {self.prediction_horizon}분)")
        
        return X, y, timestamps, np.array(actuals)
    
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
    
    def inverse_transform_single(self, scaled_values):
        """단일 특징 역변환"""
        # RobustScaler는 center_와 scale_ 사용
        return scaled_values * self.scaler.scale_[0] + self.scaler.center_[0]
    
    def evaluate_final(self, y, predictions, actuals):
        """최종 평가 (원본 스케일)"""
        print("\n" + "=" * 60)
        print("최종 평가 (원본 스케일)")
        print("=" * 60)
        
        # y_original은 이미 전달받은 actuals 사용
        y_original = actuals
        
        for name, pred in predictions.items():
            pred_original = self.inverse_transform_single(pred)
            
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
    
    def visualize(self, y, predictions, timestamps, actuals):
        """시각화"""
        print("\n결과 시각화 생성 중...")
        
        # y_original은 이미 전달받은 actuals 사용
        y_original = actuals
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('반도체 물류 예측 결과 (100분 -> 10분 후)', fontsize=14, fontweight='bold')
        
        # 1. 시계열 비교
        ax1 = axes[0, 0]
        sample = min(500, len(y))
        
        ax1.plot(y_original[:sample], label='실제', color='black', linewidth=2)
        
        colors = {'lstm': 'blue', 'gru': 'green', 'cnn_lstm': 'orange', 'ensemble': 'red'}
        for name in ['lstm', 'gru', 'ensemble']:
            if name in predictions:
                pred_original = self.inverse_transform_single(predictions[name])
                ax1.plot(pred_original[:sample], label=name.upper(), alpha=0.7, color=colors.get(name, 'gray'))
        
        ax1.axhline(y=self.spike_threshold, color='red', linestyle='--', alpha=0.3, label='1400 임계값')
        ax1.set_xlabel('시간 인덱스')
        ax1.set_ylabel('물류량')
        ax1.set_title('예측 결과 비교')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 산점도
        ax2 = axes[0, 1]
        if 'ensemble' in predictions:
            pred_original = self.inverse_transform_single(predictions['ensemble'])
            ax2.scatter(y_original, pred_original, alpha=0.5, s=1)
            ax2.plot([y_original.min(), y_original.max()], 
                    [y_original.min(), y_original.max()], 
                    'r--', alpha=0.5)
            ax2.set_xlabel('실제값')
            ax2.set_ylabel('예측값')
            ax2.set_title('앙상블 예측 산점도')
            ax2.grid(True, alpha=0.3)
        
        # 3. 오차 분포
        ax3 = axes[1, 0]
        if 'ensemble' in predictions:
            pred_original = self.inverse_transform_single(predictions['ensemble'])
            errors = y_original - pred_original
            ax3.hist(errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax3.axvline(x=0, color='red', linestyle='--')
            ax3.set_xlabel('예측 오차')
            ax3.set_ylabel('빈도')
            ax3.set_title(f'오차 분포 (평균: {errors.mean():.2f}, 표준편차: {errors.std():.2f})')
            ax3.grid(True, alpha=0.3)
        
        # 4. 1400+ 구간 성능
        ax4 = axes[1, 1]
        if 'ensemble' in predictions:
            pred_original = self.inverse_transform_single(predictions['ensemble'])
            
            # 1400+ 구간만
            spike_mask = y_original >= self.spike_threshold
            if spike_mask.sum() > 0:
                spike_actual = y_original[spike_mask]
                spike_pred = pred_original[spike_mask]
                
                ax4.scatter(spike_actual, spike_pred, alpha=0.6, color='red', s=10)
                ax4.plot([1400, spike_actual.max()], [1400, spike_actual.max()], 'k--', alpha=0.5)
                ax4.set_xlabel('실제값 (1400+)')
                ax4.set_ylabel('예측값')
                ax4.set_title(f'1400+ 구간 예측 (n={spike_mask.sum()})')
                ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('evaluation_results_100min.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ 시각화 저장: evaluation_results_100min.png")
    
    def save_results(self, y, predictions, timestamps, actuals):
        """결과 저장"""
        print("\n결과 저장 중...")
        
        # y_original은 이미 전달받은 actuals 사용
        y_original = actuals
        
        # 데이터프레임 생성
        results_df = pd.DataFrame({
            'start_time': [t['start'] for t in timestamps],
            'end_time': [t['end'] for t in timestamps],
            'target_time': [t['target'] for t in timestamps],
            '실제값': y_original
        })
        
        # 예측값 추가
        for name, pred in predictions.items():
            pred_original = self.inverse_transform_single(pred)
            results_df[f'{name}_예측'] = pred_original
            results_df[f'{name}_오차'] = y_original - pred_original
        
        # 시간 검증 컬럼 추가
        results_df['시간차_분'] = (results_df['target_time'] - results_df['end_time']).dt.total_seconds() / 60
        
        # CSV 저장
        output_file = f'prediction_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"✓ 결과 저장: {output_file}")
        
        # 시간 검증 출력
        print(f"\n시간 검증:")
        print(f"  평균 시간차: {results_df['시간차_분'].mean():.1f}분")
        print(f"  최소 시간차: {results_df['시간차_분'].min():.1f}분")
        print(f"  최대 시간차: {results_df['시간차_분'].max():.1f}분")
        
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
    
    # 4. 시퀀스 생성 (100분) - 수정된 메소드 사용
    X, y, timestamps, actuals = evaluator.prepare_sequences(df)
    
    # 5. 예측 및 평가
    predictions, results = evaluator.predict_and_evaluate(X, y)
    
    # 6. 최종 평가 (원본 스케일) - actuals 전달
    evaluator.evaluate_final(y, predictions, actuals)
    
    # 7. 시각화 - actuals 전달
    evaluator.visualize(y, predictions, timestamps, actuals)
    
    # 8. 결과 저장 - actuals 전달
    results_df = evaluator.save_results(y, predictions, timestamps, actuals)
    
    print("\n" + "=" * 60)
    print("✅ 평가 완료!")
    print("=" * 60)
    
    return results_df, results

if __name__ == "__main__":
    results_df, evaluation = main()