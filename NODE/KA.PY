import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# CPU 강제 사용 설정
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # GPU 완전 비활성화
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 딥러닝 라이브러리
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, RNN, SimpleRNN, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ARIMA 관련
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# 시각화
import matplotlib.pyplot as plt
import seaborn as sns

# 모델 저장
import pickle
import joblib

# TensorFlow CPU 전용 설정
tf.config.set_visible_devices([], 'GPU')
physical_devices = tf.config.list_physical_devices('CPU')
print(f"사용 가능한 CPU 장치: {len(physical_devices)}개")
print("GPU 사용: 비활성화됨")

class SemiconductorLogisticsDataGenerator:
    """반도체 물류 샘플 데이터 생성 클래스 (10분 단위)"""
    
    def __init__(self, n_samples=600):
        self.n_samples = n_samples
        self.routes = ['A-B', 'B-C', 'C-D', 'D-E', 'A-C', 'B-D', 'C-E']
        self.product_types = ['DRAM', 'NAND', 'CPU', 'GPU', 'ASIC']
        self.priorities = ['High', 'Medium', 'Low']
        
    def generate_data(self):
        """반도체 물류 이동 데이터 생성 (10분 간격)"""
        np.random.seed(42)
        
        data = []
        # 10분 간격으로 600개 = 100시간 (약 4일) 데이터
        start_date = datetime.now() - timedelta(minutes=self.n_samples * 10)
        
        for i in range(self.n_samples):
            timestamp = start_date + timedelta(minutes=i * 10)
            
            # 시간대별 패턴
            hour = timestamp.hour
            if 0 <= hour < 6:  # 새벽
                time_factor = 0.3
            elif 6 <= hour < 9:  # 출근 시간
                time_factor = 1.5
            elif 9 <= hour < 12:  # 오전
                time_factor = 1.2
            elif 12 <= hour < 14:  # 점심시간
                time_factor = 0.8
            elif 14 <= hour < 18:  # 오후
                time_factor = 1.3
            elif 18 <= hour < 21:  # 퇴근 시간
                time_factor = 1.4
            else:  # 야간
                time_factor = 0.6
            
            # 요일별 패턴
            weekday_factor = 0.7 if timestamp.weekday() >= 5 else 1.0
            
            # 기본 물류량 (10분 단위)
            base_volume = 50 + np.sin(i * 0.1) * 20 + np.random.normal(0, 10)
            
            # 경로별 데이터 생성
            for route in self.routes:
                route_factor = np.random.uniform(0.8, 1.2)
                
                volume = base_volume * time_factor * weekday_factor * route_factor
                
                # 병목 현상 시뮬레이션
                bottleneck = 1 if np.random.random() < 0.15 else 0
                if bottleneck:
                    delay_minutes = np.random.uniform(5, 20)  # 5-20분 지연
                else:
                    delay_minutes = np.random.uniform(0, 5)   # 0-5분 정상 지연
                
                # 제품 타입별 특성
                product_type = np.random.choice(self.product_types)
                priority = np.random.choice(self.priorities)
                
                # 처리 시간 (병목 구간에서는 증가)
                process_minutes = np.random.uniform(10, 30) * (1 + bottleneck * 0.5)
                
                # 10분 후 예측을 위한 타겟 값 생성
                future_volume = volume * np.random.uniform(0.9, 1.1)  # ±10% 변동
                
                data.append({
                    'timestamp': timestamp,
                    'route': route,
                    'product_type': product_type,
                    'priority': priority,
                    'current_volume': int(volume),
                    'volume_10min_later': int(future_volume),  # 10분 후 물류량
                    'process_minutes': round(process_minutes, 2),
                    'delay_minutes': round(delay_minutes, 2),
                    'bottleneck': bottleneck,
                    'temperature': np.random.uniform(20, 25),
                    'humidity': np.random.uniform(40, 60),
                    'hour': hour,
                    'weekday': timestamp.weekday()
                })
        
        return pd.DataFrame(data)

class HybridLogisticsPredictor:
    """LSTM, RNN, ARIMA 통합 하이브리드 예측 모델 (10분 후 예측)"""
    
    def __init__(self, lookback=6):  # 6개 = 60분(1시간) 데이터로 10분 후 예측
        self.lookback = lookback
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.lstm_model = None
        self.rnn_model = None
        self.arima_model = None
        self.weights = {'lstm': 0.4, 'rnn': 0.3, 'arima': 0.3}
        self.model_path = 'semiconductor_logistics_models'
        
    def prepare_data(self, df):
        """시계열 데이터 준비 (10분 단위)"""
        # 시간 순으로 정렬
        df = df.sort_values('timestamp')
        
        # 경로별로 분리하여 처리
        all_X, all_y = [], []
        
        for route in df['route'].unique():
            route_data = df[df['route'] == route].copy()
            
            # 특성 선택
            features = ['current_volume', 'process_minutes', 'delay_minutes', 
                       'temperature', 'humidity', 'hour', 'weekday']
            
            X_data = route_data[features].values
            y_data = route_data['volume_10min_later'].values.reshape(-1, 1)
            
            # 정규화
            X_scaled = self.scaler_X.fit_transform(X_data)
            y_scaled = self.scaler_y.fit_transform(y_data)
            
            # 시퀀스 생성 (lookback 시간의 데이터로 10분 후 예측)
            for i in range(self.lookback, len(X_scaled)):
                all_X.append(X_scaled[i-self.lookback:i])
                all_y.append(y_scaled[i, 0])
        
        return np.array(all_X), np.array(all_y)
    
    def build_lstm_model(self, input_shape):
        """LSTM 모델 구축"""
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32, return_sequences=True),
            Dropout(0.2),
            LSTM(16),
            Dropout(0.2),
            Dense(8, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def build_rnn_model(self, input_shape):
        """RNN 모델 구축"""
        model = Sequential([
            SimpleRNN(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            SimpleRNN(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model
    
    def train(self, df, epochs=30, batch_size=32):
        """하이브리드 모델 학습"""
        print("데이터 준비 중...")
        X, y = self.prepare_data(df)
        
        # 훈련/테스트 분할
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"훈련 데이터: {X_train.shape}, 테스트 데이터: {X_test.shape}")
        
        # LSTM 모델 학습
        print("\nLSTM 모델 학습 중...")
        self.lstm_model = self.build_lstm_model((X_train.shape[1], X_train.shape[2]))
        self.lstm_model.fit(X_train, y_train, 
                           epochs=epochs, 
                           batch_size=batch_size, 
                           validation_split=0.1,
                           verbose=1)
        
        # RNN 모델 학습
        print("\nRNN 모델 학습 중...")
        self.rnn_model = self.build_rnn_model((X_train.shape[1], X_train.shape[2]))
        self.rnn_model.fit(X_train, y_train, 
                          epochs=epochs, 
                          batch_size=batch_size, 
                          validation_split=0.1,
                          verbose=1)
        
        # ARIMA 모델은 각 경로별로 별도 학습 (시연용으로 첫 번째 경로만)
        print("\nARIMA 모델 학습 중...")
        first_route = df['route'].unique()[0]
        arima_data = df[df['route'] == first_route]['current_volume'].values[:split_idx]
        try:
            self.arima_model = ARIMA(arima_data, order=(2, 1, 1)).fit()
        except:
            print("ARIMA 모델 학습 실패, 가중치 조정")
            self.weights = {'lstm': 0.5, 'rnn': 0.5, 'arima': 0.0}
        
        # 모델 평가
        self.evaluate_models(X_test, y_test)
        
        return X_test, y_test
    
    def predict_10min_later(self, recent_data):
        """10분 후 물류량 예측"""
        # 입력 데이터 정규화
        X = self.scaler_X.transform(recent_data)
        X = X.reshape(1, X.shape[0], X.shape[1])
        
        # LSTM 예측
        lstm_pred = self.lstm_model.predict(X, verbose=0)[0, 0]
        
        # RNN 예측
        rnn_pred = self.rnn_model.predict(X, verbose=0)[0, 0]
        
        # ARIMA 예측 (단순화)
        if self.arima_model and self.weights['arima'] > 0:
            arima_pred = self.arima_model.forecast(steps=1)[0]
            arima_pred = self.scaler_y.transform([[arima_pred]])[0, 0]
        else:
            arima_pred = 0
        
        # 가중 평균
        hybrid_pred = (
            self.weights['lstm'] * lstm_pred +
            self.weights['rnn'] * rnn_pred +
            self.weights['arima'] * arima_pred
        )
        
        # 역변환
        predicted_volume = self.scaler_y.inverse_transform([[hybrid_pred]])[0, 0]
        
        return {
            'predicted_volume_10min': int(predicted_volume),
            'lstm_prediction': int(self.scaler_y.inverse_transform([[lstm_pred]])[0, 0]),
            'rnn_prediction': int(self.scaler_y.inverse_transform([[rnn_pred]])[0, 0]),
            'confidence': self._calculate_confidence(lstm_pred, rnn_pred)
        }
    
    def _calculate_confidence(self, lstm_pred, rnn_pred):
        """예측 신뢰도 계산"""
        # 두 모델의 예측값 차이를 기반으로 신뢰도 계산
        diff = abs(lstm_pred - rnn_pred)
        confidence = max(0, 1 - diff) * 100
        return round(confidence, 2)
    
    def evaluate_models(self, X_test, y_test):
        """모델 성능 평가"""
        # 예측
        lstm_pred = self.lstm_model.predict(X_test, verbose=0)
        rnn_pred = self.rnn_model.predict(X_test, verbose=0)
        
        # 하이브리드 예측
        hybrid_pred = (
            self.weights['lstm'] * lstm_pred.flatten() +
            self.weights['rnn'] * rnn_pred.flatten()
        ) / (self.weights['lstm'] + self.weights['rnn'])
        
        # 역변환
        y_test_orig = self.scaler_y.inverse_transform(y_test.reshape(-1, 1))
        hybrid_pred_orig = self.scaler_y.inverse_transform(hybrid_pred.reshape(-1, 1))
        lstm_pred_orig = self.scaler_y.inverse_transform(lstm_pred)
        rnn_pred_orig = self.scaler_y.inverse_transform(rnn_pred)
        
        # 성능 지표
        print("\n=== 10분 후 예측 모델 성능 ===")
        print(f"Hybrid MAE: {mean_absolute_error(y_test_orig, hybrid_pred_orig):.2f}")
        print(f"LSTM MAE: {mean_absolute_error(y_test_orig, lstm_pred_orig):.2f}")
        print(f"RNN MAE: {mean_absolute_error(y_test_orig, rnn_pred_orig):.2f}")
        
        # 시각화
        self.plot_predictions(y_test_orig[:100], hybrid_pred_orig[:100], 
                            lstm_pred_orig[:100], rnn_pred_orig[:100])
    
    def plot_predictions(self, y_true, hybrid_pred, lstm_pred, rnn_pred):
        """예측 결과 시각화"""
        plt.figure(figsize=(15, 6))
        
        time_steps = np.arange(len(y_true)) * 10  # 10분 단위
        
        plt.plot(time_steps, y_true, label='실제값', color='black', linewidth=2)
        plt.plot(time_steps, hybrid_pred, label='Hybrid', color='red', linewidth=2, alpha=0.8)
        plt.plot(time_steps, lstm_pred, label='LSTM', color='blue', alpha=0.6)
        plt.plot(time_steps, rnn_pred, label='RNN', color='green', alpha=0.6)
        
        plt.title('10분 후 물류량 예측 결과')
        plt.xlabel('시간 (분)')
        plt.ylabel('물류량')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def save_models(self, path=None):
        """모델 저장"""
        if path is None:
            path = self.model_path
        
        os.makedirs(path, exist_ok=True)
        
        # 딥러닝 모델 저장
        self.lstm_model.save(f'{path}/lstm_model.h5')
        self.rnn_model.save(f'{path}/rnn_model.h5')
        
        # ARIMA 모델과 스케일러 저장
        joblib.dump(self.arima_model, f'{path}/arima_model.pkl')
        joblib.dump(self.scaler_X, f'{path}/scaler_X.pkl')
        joblib.dump(self.scaler_y, f'{path}/scaler_y.pkl')
        joblib.dump(self.weights, f'{path}/weights.pkl')
        
        print(f"\n모델이 '{path}' 디렉토리에 저장되었습니다.")
    
    def load_models(self, path=None):
        """모델 불러오기"""
        if path is None:
            path = self.model_path
        
        try:
            # 딥러닝 모델 불러오기
            self.lstm_model = load_model(f'{path}/lstm_model.h5')
            self.rnn_model = load_model(f'{path}/rnn_model.h5')
            
            # ARIMA 모델과 스케일러 불러오기
            self.arima_model = joblib.load(f'{path}/arima_model.pkl')
            self.scaler_X = joblib.load(f'{path}/scaler_X.pkl')
            self.scaler_y = joblib.load(f'{path}/scaler_y.pkl')
            self.weights = joblib.load(f'{path}/weights.pkl')
            
            print(f"모델이 '{path}' 디렉토리에서 성공적으로 불러와졌습니다.")
            return True
        except Exception as e:
            print(f"모델 불러오기 실패: {e}")
            return False

class BottleneckPredictor:
    """병목 구간 예측 모델"""
    
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        self.model_path = 'semiconductor_logistics_models'
        
    def prepare_features(self, df):
        """특성 엔지니어링"""
        # 원핫 인코딩
        route_dummies = pd.get_dummies(df['route'], prefix='route')
        product_dummies = pd.get_dummies(df['product_type'], prefix='product')
        priority_dummies = pd.get_dummies(df['priority'], prefix='priority')
        
        # 특성 결합
        features = pd.concat([
            df[['current_volume', 'process_minutes', 'temperature', 'humidity', 'hour', 'weekday']],
            route_dummies,
            product_dummies,
            priority_dummies
        ], axis=1)
        
        return features, df['bottleneck']
    
    def build_model(self, input_shape):
        """병목 예측 신경망 모델"""
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_shape,)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='binary_crossentropy', 
                     metrics=['accuracy'])
        return model
    
    def train(self, df, epochs=30, batch_size=32):
        """병목 예측 모델 학습"""
        print("\n병목 구간 예측 모델 학습 중...")
        
        X, y = self.prepare_features(df)
        X_scaled = self.scaler.fit_transform(X)
        
        # 훈련/테스트 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 모델 학습
        self.model = self.build_model(X_train.shape[1])
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1
        )
        
        # 평가
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\n병목 예측 모델 정확도: {test_acc:.4f}")
        
        # 병목 예측 확률
        bottleneck_probs = self.model.predict(X_test, verbose=0)
        bottleneck_preds = (bottleneck_probs > 0.5).astype(int)
        
        # 경로별 병목 위험도 분석
        self.analyze_bottleneck_risk(df, X, y)
        
        return history
    
    def analyze_bottleneck_risk(self, df, X, y):
        """경로별 병목 위험도 분석"""
        df_analysis = df.copy()
        df_analysis['bottleneck_prob'] = self.model.predict(self.scaler.transform(X), verbose=0)
        
        # 경로별 병목 위험도
        route_risk = df_analysis.groupby('route').agg({
            'bottleneck': 'mean',
            'bottleneck_prob': 'mean',
            'delay_minutes': 'mean'
        }).round(3)
        
        print("\n=== 경로별 병목 위험도 분석 ===")
        print(route_risk.sort_values('bottleneck_prob', ascending=False))
    
    def predict_bottleneck(self, features):
        """병목 구간 예측"""
        features_scaled = self.scaler.transform(features)
        prob = self.model.predict(features_scaled, verbose=0)[0, 0]
        
        return {
            'bottleneck_probability': round(prob * 100, 2),
            'is_bottleneck': prob > 0.5,
            'risk_level': 'High' if prob > 0.7 else 'Medium' if prob > 0.3 else 'Low'
        }
    
    def save_model(self, path=None):
        """모델 저장"""
        if path is None:
            path = self.model_path
        
        os.makedirs(path, exist_ok=True)
        
        self.model.save(f'{path}/bottleneck_model.h5')
        joblib.dump(self.scaler, f'{path}/bottleneck_scaler.pkl')
        
        print(f"병목 예측 모델이 저장되었습니다.")
    
    def load_model(self, path=None):
        """모델 불러오기"""
        if path is None:
            path = self.model_path
        
        try:
            self.model = load_model(f'{path}/bottleneck_model.h5')
            self.scaler = joblib.load(f'{path}/bottleneck_scaler.pkl')
            print("병목 예측 모델이 성공적으로 불러와졌습니다.")
            return True
        except Exception as e:
            print(f"병목 모델 불러오기 실패: {e}")
            return False

class RealTimePredictor:
    """실시간 예측 시스템"""
    
    def __init__(self):
        self.volume_predictor = HybridLogisticsPredictor()
        self.bottleneck_predictor = BottleneckPredictor()
        
    def load_all_models(self, path='semiconductor_logistics_models'):
        """모든 모델 불러오기"""
        print("모델을 불러오는 중...")
        volume_loaded = self.volume_predictor.load_models(path)
        bottleneck_loaded = self.bottleneck_predictor.load_model(path)
        
        if volume_loaded and bottleneck_loaded:
            print("모든 모델이 성공적으로 불러와졌습니다!")
            return True
        return False
    
    def predict_next_10min(self, current_data):
        """10분 후 예측 (현재 상태 기반)"""
        # current_data: 최근 1시간(6개) 데이터
        
        # 물류량 예측
        volume_result = self.volume_predictor.predict_10min_later(current_data)
        
        # 병목 예측을 위한 특성 준비
        latest_features = current_data[-1:].reshape(1, -1)
        bottleneck_result = self.bottleneck_predictor.predict_bottleneck(latest_features)
        
        return {
            '10분후_예측': {
                '예상_물류량': volume_result['predicted_volume_10min'],
                '병목_확률': f"{bottleneck_result['bottleneck_probability']}%",
                '위험도': bottleneck_result['risk_level'],
                '예측_신뢰도': f"{volume_result['confidence']}%"
            },
            '모델별_예측': {
                'LSTM': volume_result['lstm_prediction'],
                'RNN': volume_result['rnn_prediction']
            },
            '권장사항': self._generate_recommendations(volume_result, bottleneck_result)
        }
    
    def _generate_recommendations(self, volume_result, bottleneck_result):
        """예측 결과 기반 권장사항 생성"""
        recommendations = []
        
        if bottleneck_result['bottleneck_probability'] > 70:
            recommendations.append("⚠️ 병목 위험 높음: 대체 경로 활용 권장")
            recommendations.append("📦 추가 처리 인력 배치 고려")
        
        if volume_result['predicted_volume_10min'] > 100:
            recommendations.append("📈 높은 물류량 예상: 사전 준비 필요")
        
        if volume_result['confidence'] < 70:
            recommendations.append("🔍 예측 신뢰도 낮음: 실시간 모니터링 강화")
        
        return recommendations if recommendations else ["✅ 정상 운영 예상"]

# 메인 실행 코드
if __name__ == "__main__":
    print("반도체 물류 예측 시스템 (10분 후 예측)")
    print("=" * 60)
    
    # 1. 샘플 데이터 생성
    print("\n1. 샘플 데이터 생성 중...")
    generator = SemiconductorLogisticsDataGenerator(n_samples=600)
    df = generator.generate_data()
    print(f"생성된 데이터: {len(df)}개 레코드")
    print(f"데이터 기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
    print(f"10분 간격 데이터: 총 {(df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 60:.0f}분")
    
    # 2. 물류량 예측 모델 학습
    print("\n2. 하이브리드 물류량 예측 모델 학습 (10분 후 예측)")
    print("=" * 60)
    predictor = HybridLogisticsPredictor(lookback=6)  # 60분 데이터로 10분 후 예측
    X_test, y_test = predictor.train(df, epochs=20, batch_size=32)
    
    # 3. 병목 구간 예측 모델 학습
    print("\n3. 병목 구간 예측 모델 학습")
    print("=" * 60)
    bottleneck_predictor = BottleneckPredictor()
    bottleneck_predictor.train(df, epochs=20, batch_size=32)
    
    # 4. 모델 저장
    print("\n4. 학습된 모델 저장 중...")
    predictor.save_models()
    bottleneck_predictor.save_model()
    
    # 5. 실시간 예측 시뮬레이션
    print("\n5. 실시간 예측 시뮬레이션")
    print("=" * 60)
    
    # 실시간 예측기 생성 및 모델 로드
    realtime = RealTimePredictor()
    if realtime.load_all_models():
        # 최근 1시간 데이터로 10분 후 예측
        recent_data = df.iloc[-6:][['current_volume', 'process_minutes', 'delay_minutes', 
                                    'temperature', 'humidity', 'hour', 'weekday']].values
        
        print("\n현재 상황:")
        print(f"현재 시각: {df.iloc[-1]['timestamp']}")
        print(f"현재 물류량: {df.iloc[-1]['current_volume']}")
        print(f"현재 처리시간: {df.iloc[-1]['process_minutes']:.1f}분")
        
        # 10분 후 예측
        prediction = realtime.predict_next_10min(recent_data)
        
        print("\n=== 10분 후 예측 결과 ===")
        for key, value in prediction['10분후_예측'].items():
            print(f"{key}: {value}")
        
        print("\n모델별 예측:")
        for model, pred in prediction['모델별_예측'].items():
            print(f"  {model}: {pred}")
        
        print("\n권장사항:")
        for rec in prediction['권장사항']:
            print(f"  {rec}")
    
    print("\n시스템 준비 완료! 모델을 사용하여 실시간 예측이 가능합니다.")