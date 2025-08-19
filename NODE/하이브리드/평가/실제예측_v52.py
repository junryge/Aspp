"""
20250807_DATA.CSV 앙상블 예측
과거 100분 데이터로 10분 후 예측
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Input, LSTM, Dense, Dropout, BatchNormalization,
                                     GRU, Conv1D, MaxPooling1D, Bidirectional)
from tensorflow.keras.regularizers import l1_l2
from sklearn.preprocessing import RobustScaler
import joblib
import os
import warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

class Predictor20250807:
    """20250807 데이터 예측"""
    
    def __init__(self):
        # 모델 경로 (실제 경로로 수정 필요)
        self.model_dir = r'D:\하이닉스\6.연구_항목\CODE\202508051차_POC구축\앙상블_하이브리드v5_150g학습\models_v5'
        self.data_path = r'D:\하이닉스\6.연구_항목\CODE\202508051차_POC구축\앙상블_하이브리드v5_150g학습\data\20250807_DATA.csv'
        
        self.sequence_length = 100  # 과거 100분
        self.prediction_horizon = 10  # 10분 후 예측
        self.spike_threshold = 1400
        
        self.models = {}
        self.scaler = None
        
    def build_improved_lstm(self, input_shape):
        """LSTM 모델"""
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
        """Spike Detector"""
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
        input_shape = (self.sequence_length, 12)
        
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
                    model = build_func(input_shape)
                    model.load_weights(filepath)
                    model.compile(
                        optimizer='adam',
                        loss='mae' if name != 'spike_detector' else 'binary_crossentropy',
                        metrics=['mae'] if name != 'spike_detector' else ['accuracy']
                    )
                    self.models[name] = model
                except Exception as e:
                    pass
        
        # 스케일러 로드
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            try:
                self.scaler = joblib.load(scaler_path)
            except:
                self.scaler = RobustScaler()
        else:
            self.scaler = RobustScaler()
        
        return len(self.models) > 0
    
    def load_data(self):
        """데이터 로드"""
        df = pd.read_csv(self.data_path, encoding='utf-8')
        df['datetime'] = pd.to_datetime(df['CURRTIME'], format='%Y%m%d%H%M')
        return df
    
    def create_features(self, df):
        """특징 생성"""
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
        df = df.fillna(method='ffill').fillna(0)
        
        return df
    
    def prepare_sequences(self, df):
        """100분 시퀀스 생성"""
        # 데이터가 부족한 경우 처리
        if len(df) < self.sequence_length + self.prediction_horizon:
            self.sequence_length = min(50, len(df) - self.prediction_horizon - 1)
        
        # 특징 컬럼
        feature_cols = ['TOTALCNT', 'MA_10', 'MA_30', 'MA_60', 
                       'STD_10', 'STD_30',
                       'change_rate', 'change_rate_10', 
                       'hour', 'dayofweek', 'is_weekend', 'trend']
        
        # 스케일링
        scaled_data = self.scaler.fit_transform(df[feature_cols])
        
        X = []
        timestamps = []
        
        # 시퀀스 생성
        for i in range(self.sequence_length, len(df) - self.prediction_horizon + 1):
            # 과거 데이터
            seq = scaled_data[i-self.sequence_length:i]
            
            # 100분 맞추기 위해 패딩 (필요시)
            if seq.shape[0] < 100:
                padding = np.zeros((100 - seq.shape[0], seq.shape[1]))
                seq = np.vstack([padding, seq])
            
            X.append(seq)
            
            # 시간 정보
            timestamps.append({
                'current_time': df['datetime'].iloc[i-1],
                'predict_time': df['datetime'].iloc[min(i+self.prediction_horizon-1, len(df)-1)],
                'current_value': df['TOTALCNT'].iloc[i-1]
            })
        
        X = np.array(X) if X else np.zeros((0, 100, len(feature_cols)))
        
        return X, timestamps
    
    def predict_ensemble(self, X, timestamps):
        """앙상블 예측"""
        if len(X) == 0:
            return np.array([]), {}
        
        predictions = {}
        
        # 각 모델 예측
        for name, model in self.models.items():
            pred = model.predict(X, batch_size=256, verbose=0)
            predictions[name] = pred.flatten()
        
        # 앙상블 계산
        ensemble_pred = np.zeros(len(X))
        
        for i in range(len(X)):
            # Spike detector 기반 동적 가중치
            spike_prob = predictions.get('spike_detector', [0.5])[i]
            
            if spike_prob > 0.7:  # 높은 스파이크 확률
                weights = {
                    'lstm': 0.15,
                    'gru': 0.20,
                    'cnn_lstm': 0.50,  # CNN-LSTM 강조
                    'spike_detector': 0.15
                }
                boost = 1.10  # 10% 상향
            elif spike_prob > 0.4:  # 중간 확률
                weights = {
                    'lstm': 0.25,
                    'gru': 0.35,
                    'cnn_lstm': 0.30,
                    'spike_detector': 0.10
                }
                boost = 1.05  # 5% 상향
            else:  # 정상 범위
                weights = {
                    'lstm': 0.30,
                    'gru': 0.40,  # GRU 중심
                    'cnn_lstm': 0.25,
                    'spike_detector': 0.05
                }
                boost = 1.0
            
            # 가중 평균
            weighted_sum = 0
            weight_total = 0
            
            for name, weight in weights.items():
                if name in predictions:
                    if name == 'spike_detector':
                        # spike_detector는 확률값이므로 스케일 조정 불필요
                        continue
                    else:
                        value = predictions[name][i]
                    
                    weighted_sum += value * weight
                    weight_total += weight
            
            ensemble_pred[i] = (weighted_sum / weight_total) * boost
        
        # 역변환
        dummy = np.zeros((len(ensemble_pred), 12))
        dummy[:, 0] = ensemble_pred
        ensemble_original = self.scaler.inverse_transform(dummy)[:, 0]
        
        return ensemble_original, predictions

def main():
    """메인 실행"""
    
    predictor = Predictor20250807()
    
    # 1. 모델 로드
    if not predictor.load_models():
        return None
    
    # 2. 데이터 로드
    df = predictor.load_data()
    
    # 3. 특징 생성
    df = predictor.create_features(df)
    
    # 4. 시퀀스 생성 (100분)
    X, timestamps = predictor.prepare_sequences(df)
    
    if len(X) == 0:
        return None
    
    # 5. 앙상블 예측
    ensemble_pred, all_predictions = predictor.predict_ensemble(X, timestamps)
    
    if len(ensemble_pred) == 0:
        return None
    
    # 앙상블 예측 통계만 출력 (정수로 반올림)
    stats_dict = {
        '최소값': int(round(ensemble_pred.min())),
        '최대값': int(round(ensemble_pred.max())),
        '평균값': int(round(ensemble_pred.mean())),
        '표준편차': int(round(ensemble_pred.std()))
    }
    
    print("앙상블 예측 통계:")
    print(stats_dict)
    
    return stats_dict

if __name__ == "__main__":
    statistics = main()