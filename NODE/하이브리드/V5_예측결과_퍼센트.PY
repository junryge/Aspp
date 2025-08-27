"""
20250807_DATA.CSV 앙상블 예측 + 구간별 확률 분석
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

# scipy 추가 (확률 계산용)
try:
    from scipy.stats import norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

class Predictor20250807:
    """20250807 데이터 예측"""
   
    def __init__(self):
        # 모델 경로 (실제 경로로 수정 필요)
        self.model_dir = './models'
        self.data_path = './data/20250807_DATA.csv'
       
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
            'lstm': (self.build_improved_lstm, 'lstm_final.keras'),
            'gru': (self.build_improved_gru, 'gru_final.keras'),
            'cnn_lstm': (self.build_improved_cnn_lstm, 'cnn_lstm_final.keras'),
            'spike_detector': (self.build_improved_spike_detector, 'spike_detector_final.keras')
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
        scaler_path = os.path.join(self.model_dir, './scaler/scaler.pkl')
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

def ensemble_value():
    """메인 실행 - 원본 함수"""
   
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
    ensemble_value = {
        '최소값': int(round(ensemble_pred.min())),
        '최대값': int(round(ensemble_pred.max())),
        '평균값': int(round(ensemble_pred.mean())),
        '표준편차': int(round(ensemble_pred.std()))
    }
   
    print(ensemble_value)
    return ensemble_value

# ========================================
# 🎯 구간별 확률 분석 함수 추가!
# ========================================

def analyze_prediction_probabilities(ensemble_dict):
    """
    ensemble_value 딕셔너리를 받아서 구간별 확률 계산
    
    Args:
        ensemble_dict: {'최소값': 1356, '최대값': 1849, '평균값': 1520, '표준편차': 95}
    
    Returns:
        구간별 확률 딕셔너리
    """
    
    if not ensemble_dict:
        return None
        
    min_val = ensemble_dict.get('최소값', 1200)
    max_val = ensemble_dict.get('최대값', 1600) 
    avg_val = ensemble_dict.get('평균값', 1400)
    std_val = ensemble_dict.get('표준편차', 50)
    
    # scipy 있을 때와 없을 때 다르게 처리
    if SCIPY_AVAILABLE:
        # 정규분포 가정하여 구간별 확률 계산
        mu = avg_val  # 평균
        sigma = max(std_val, 1)  # 표준편차 (0 방지)
        
        # 구간별 확률 계산 (누적분포함수 이용)
        prob_below_1200 = norm.cdf(1200, mu, sigma) * 100
        prob_1200_1400 = (norm.cdf(1400, mu, sigma) - norm.cdf(1200, mu, sigma)) * 100
        prob_1400_1600 = (norm.cdf(1600, mu, sigma) - norm.cdf(1400, mu, sigma)) * 100  
        prob_1600_1700 = (norm.cdf(1700, mu, sigma) - norm.cdf(1600, mu, sigma)) * 100
        prob_above_1700 = (1 - norm.cdf(1700, mu, sigma)) * 100
    else:
        # scipy 없으면 간단한 추정
        if avg_val < 1300:
            prob_below_1200 = 20.0
            prob_1200_1400 = 60.0
            prob_1400_1600 = 15.0
            prob_1600_1700 = 4.0
            prob_above_1700 = 1.0
        elif avg_val < 1500:
            prob_below_1200 = 5.0
            prob_1200_1400 = 40.0
            prob_1400_1600 = 45.0
            prob_1600_1700 = 8.0
            prob_above_1700 = 2.0
        elif avg_val < 1650:
            prob_below_1200 = 1.0
            prob_1200_1400 = 20.0
            prob_1400_1600 = 55.0
            prob_1600_1700 = 20.0
            prob_above_1700 = 4.0
        else:
            prob_below_1200 = 0.0
            prob_1200_1400 = 10.0
            prob_1400_1600 = 30.0
            prob_1600_1700 = 45.0
            prob_above_1700 = 15.0
    
    # 실제 범위에 맞춰 조정
    if min_val >= 1200:
        prob_below_1200 = 0.0
    if max_val <= 1700:
        prob_above_1700 = 0.0
        
    # 총합이 100%가 되도록 정규화
    total_prob = prob_below_1200 + prob_1200_1400 + prob_1400_1600 + prob_1600_1700 + prob_above_1700
    
    if total_prob > 0:
        prob_below_1200 = (prob_below_1200 / total_prob) * 100
        prob_1200_1400 = (prob_1200_1400 / total_prob) * 100
        prob_1400_1600 = (prob_1400_1600 / total_prob) * 100
        prob_1600_1700 = (prob_1600_1700 / total_prob) * 100
        prob_above_1700 = (prob_above_1700 / total_prob) * 100
    
    # 결과 딕셔너리 생성
    result = {
        '평균값': avg_val,
        '최소값': min_val,
        '최대값': max_val,
        '표준편차': std_val,
        
        # 구간별 확률
        '1200미만_퍼센트': round(prob_below_1200, 1),
        '1200-1400양호_퍼센트': round(prob_1200_1400, 1),
        '1400-1600보통_퍼센트': round(prob_1400_1600, 1), 
        '1600이상경고_퍼센트': round(prob_1600_1700, 1),
        '1700주의_퍼센트': round(prob_above_1700, 1)
    }
    
    return result

def show_analysis(ensemble_dict):
    """구간별 확률 분석 결과 출력"""
    
    analysis = analyze_prediction_probabilities(ensemble_dict)
    
    if not analysis:
        print("❌ 분석 실패")
        return None
    
    print("="*60)
    print("🎯 앙상블 예측 구간별 확률 분석")  
    print("="*60)
    print(f"📊 기본 통계:")
    print(f"  평균값: {analysis['평균값']}")
    print(f"  범위: {analysis['최소값']} ~ {analysis['최대값']}")
    print(f"  표준편차: {analysis['표준편차']}")
    print()
    
    print("📋 구간별 예측 확률:")
    print(f"  🔵 1200미만      : {analysis['1200미만_퍼센트']:5.1f}%")
    print(f"  🟢 1200-1400양호 : {analysis['1200-1400양호_퍼센트']:5.1f}%")  
    print(f"  🟡 1400-1600보통 : {analysis['1400-1600보통_퍼센트']:5.1f}%")
    print(f"  🟠 1600이상경고  : {analysis['1600이상경고_퍼센트']:5.1f}%")
    print(f"  🔴 1700주의      : {analysis['1700주의_퍼센트']:5.1f}%")
    print("="*60)
    
    return analysis

# ========================================
# 🚀 사용 예시
# ========================================

if __name__ == "__main__":
    print("🎯 V5 앙상블 모델 실행 + 구간별 확률 분석")
    print("="*60)
    
    # 1. 원본 ensemble_value 함수 실행
    print("1️⃣ 앙상블 예측 실행 중...")
    ensemble_result = ensemble_value()
    
    if ensemble_result:
        print("\n2️⃣ 구간별 확률 분석 중...")
        
        # 2. 구간별 확률 분석
        analysis_result = show_analysis(ensemble_result)
        
        if analysis_result:
            print("\n3️⃣ 개별 값 추출 예시:")
            print("="*40)
            print(f"평균값만:        {analysis_result['평균값']}")
            print(f"양호 확률:       {analysis_result['1200-1400양호_퍼센트']}%")
            print(f"경고 확률:       {analysis_result['1600이상경고_퍼센트']}%")
            print(f"주의 확률:       {analysis_result['1700주의_퍼센트']}%")
            
            print(f"\n📝 사용법:")
            print(f"# 평균값 출력")
            print(f"print(analysis_result['평균값'])  # {analysis_result['평균값']}")
            print(f"# 양호 확률 출력") 
            print(f"print(analysis_result['1200-1400양호_퍼센트'])  # {analysis_result['1200-1400양호_퍼센트']}%")
    else:
        print("❌ 앙상블 예측 실패")

# 간단 사용 함수들
def get_average_only():
    """평균값만 빠르게 가져오기"""
    result = ensemble_value()
    if result:
        return result['평균값']
    return None

def get_full_analysis():
    """전체 분석 한 번에 가져오기"""
    ensemble_result = ensemble_value()
    if ensemble_result:
        return analyze_prediction_probabilities(ensemble_result)
    return None