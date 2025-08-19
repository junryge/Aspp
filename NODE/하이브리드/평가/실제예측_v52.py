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

print(f"TensorFlow Version: {tf.__version__}")
print("=" * 60)
print("20250807 데이터 앙상블 예측 시스템")
print("과거 100분 → 10분 후 예측")
print("=" * 60)

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
        print("\n모델 로드 중...")
        
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
                    print(f"✓ {name} 모델 로드 완료")
                except Exception as e:
                    print(f"✗ {name} 모델 로드 실패: {e}")
            else:
                print(f"✗ {name} 모델 파일 없음: {filepath}")
        
        # 스케일러 로드
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            try:
                self.scaler = joblib.load(scaler_path)
                print("✓ 스케일러 로드 완료")
            except:
                print("⚠ 스케일러 로드 실패, 새로 생성")
                self.scaler = RobustScaler()
        else:
            self.scaler = RobustScaler()
            print("⚠ 스케일러 새로 생성")
        
        return len(self.models) > 0
    
    def load_data(self):
        """데이터 로드"""
        print(f"\n데이터 로드: {self.data_path}")
        
        # CSV 파일 읽기 - 다양한 형식 시도
        try:
            # 첫 번째 시도: 일반적인 CSV
            df = pd.read_csv(self.data_path, encoding='utf-8')
        except:
            try:
                # 두 번째 시도: 탭 구분
                df = pd.read_csv(self.data_path, encoding='utf-8', sep='\t')
            except:
                # 세 번째 시도: 쉼표 구분, 헤더 없음
                df = pd.read_csv(self.data_path, encoding='utf-8', header=None)
        
        print(f"원본 데이터 shape: {df.shape}")
        print(f"컬럼: {df.columns.tolist()}")
        
        # CURRTIME 컬럼이 없으면 TIME 컬럼 사용
        if 'CURRTIME' in df.columns:
            df['datetime'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M', errors='coerce')
        elif 'TIME' in df.columns:
            df['datetime'] = pd.to_datetime(df['TIME'].astype(str), format='%Y%m%d%H%M', errors='coerce')
        else:
            # 첫 번째 컬럼이 시간일 가능성
            df['datetime'] = pd.to_datetime(df.iloc[:, 0].astype(str), format='%Y%m%d%H%M', errors='coerce')
        
        # NaT 제거
        df = df[df['datetime'].notna()].reset_index(drop=True)
        
        print(f"처리된 데이터 shape: {df.shape}")
        print(f"TOTALCNT 범위: {df['TOTALCNT'].min()} ~ {df['TOTALCNT'].max()}")
        print(f"데이터 기간: {df['datetime'].min()} ~ {df['datetime'].max()}")
        print(f"데이터 개수: {len(df)}개")
        
        # 1400+ 분석
        spikes = df['TOTALCNT'] >= self.spike_threshold
        print(f"1400+ 개수: {spikes.sum()}개 ({spikes.mean()*100:.1f}%)")
        
        return df
    
    def create_features(self, df):
        """특징 생성"""
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
        df = df.fillna(method='ffill').fillna(0)
        
        return df
    
    def prepare_sequences(self, df):
        """100분 시퀀스 생성"""
        print("\n시퀀스 생성 (과거 100분 → 10분 후 예측)...")
        
        # 특징 컬럼
        feature_cols = ['TOTALCNT', 'MA_10', 'MA_30', 'MA_60', 
                       'STD_10', 'STD_30',
                       'change_rate', 'change_rate_10', 
                       'hour', 'dayofweek', 'is_weekend', 'trend']
        
        # 스케일링
        scaled_data = self.scaler.fit_transform(df[feature_cols])
        
        X = []
        timestamps = []
        
        # 데이터가 정확히 100개 이상이면 마지막 100개로 예측
        if len(df) >= self.sequence_length:
            # 마지막 100개 데이터로 10분 후 예측
            X.append(scaled_data[-self.sequence_length:])
            
            # 시간 정보
            timestamps.append({
                'current_time': df['datetime'].iloc[-1],
                'predict_time': df['datetime'].iloc[-1] + pd.Timedelta(minutes=10),
                'current_value': df['TOTALCNT'].iloc[-1]
            })
            
            # 추가로 이전 시점들도 예측 가능하면 추가
            for i in range(self.sequence_length, len(df)):
                X.append(scaled_data[i-self.sequence_length:i])
                
                timestamps.append({
                    'current_time': df['datetime'].iloc[i-1],
                    'predict_time': df['datetime'].iloc[i-1] + pd.Timedelta(minutes=10),
                    'current_value': df['TOTALCNT'].iloc[i-1]
                })
        
        X = np.array(X) if X else np.zeros((0, self.sequence_length, len(feature_cols)))
        
        print(f"생성된 시퀀스: {len(X)}개")
        print(f"입력 shape: {X.shape}")
        
        if len(X) > 0:
            print(f"예측 시점: {timestamps[0]['current_time']} → {timestamps[0]['predict_time']}")
        
        return X, timestamps
    
    def predict_ensemble(self, X, timestamps):
        """앙상블 예측"""
        print("\n앙상블 예측 수행...")
        
        if len(X) == 0:
            print("⚠ 예측할 데이터가 없습니다.")
            return np.array([]), {}
        
        predictions = {}
        
        # 각 모델 예측
        for name, model in self.models.items():
            print(f"  {name} 예측 중...")
            pred = model.predict(X, batch_size=256, verbose=0)
            predictions[name] = pred.flatten()
        
        # 앙상블 계산
        print("\n앙상블 가중치 적용...")
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
        print("\n역변환 수행...")
        dummy = np.zeros((len(ensemble_pred), 12))
        dummy[:, 0] = ensemble_pred
        ensemble_original = self.scaler.inverse_transform(dummy)[:, 0]
        
        return ensemble_original, predictions
    
    def display_results(self, ensemble_pred, timestamps):
        """결과 출력"""
        print("\n" + "="*60)
        print("앙상블 예측 결과")
        print("="*60)
        
        if len(ensemble_pred) == 0:
            print("예측 결과가 없습니다. 데이터를 확인해주세요.")
            return None
        
        # 기본 통계
        print(f"\n예측 통계:")
        print(f"  - 최소값: {ensemble_pred.min():.0f}")
        print(f"  - 최대값: {ensemble_pred.max():.0f}")
        print(f"  - 평균값: {ensemble_pred.mean():.0f}")
        print(f"  - 표준편차: {ensemble_pred.std():.0f}")
        
        # 1400+ 예측
        spike_count = np.sum(ensemble_pred >= self.spike_threshold)
        print(f"\n1400+ 예측: {spike_count}개 ({spike_count/len(ensemble_pred)*100:.1f}%)")
        
        # 상위 20개 예측값
        print(f"\n상위 20개 예측값:")
        top_indices = np.argsort(ensemble_pred)[-min(20, len(ensemble_pred)):][::-1]
        
        for i, idx in enumerate(top_indices, 1):
            ts = timestamps[idx]
            print(f"{i:2}. {ts['current_time'].strftime('%Y-%m-%d %H:%M')} → " +
                  f"{ts['predict_time'].strftime('%H:%M')}: {ensemble_pred[idx]:.0f}")
        
        # 전체 예측 배열 출력
        print("\n" + "="*60)
        print(f"전체 앙상블 예측값 (총 {len(ensemble_pred)}개):")
        print("="*60)
        
        for i in range(min(len(ensemble_pred), len(timestamps))):
            ts = timestamps[i]
            print(f"{ts['current_time'].strftime('%m/%d %H:%M')} → {ts['predict_time'].strftime('%H:%M')}: {ensemble_pred[i]:.0f}")
        
        # CSV 저장
        results_df = pd.DataFrame({
            '현재시간': [t['current_time'] for t in timestamps],
            '예측시간': [t['predict_time'] for t in timestamps],
            '현재값': [t['current_value'] for t in timestamps],
            '앙상블_예측': ensemble_pred.round().astype(int)
        })
        
        output_file = 'ensemble_predictions_20250807.csv'
        results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n결과 저장: {output_file}")
        
        return results_df

def main():
    """메인 실행"""
    
    predictor = Predictor20250807()
    
    # 1. 모델 로드
    if not predictor.load_models():
        print("모델 로드 실패!")
        return None
    
    # 2. 데이터 로드
    df = predictor.load_data()
    
    # 3. 특징 생성
    df = predictor.create_features(df)
    
    # 4. 시퀀스 생성 (100분)
    X, timestamps = predictor.prepare_sequences(df)
    
    if len(X) == 0:
        print("\n" + "="*60)
        print("⚠ 데이터 부족으로 예측을 수행할 수 없습니다.")
        print(f"  현재 데이터: {len(df)}개")
        print(f"  최소 필요: 110개 이상")
        print("="*60)
        return None, None
    
    # 5. 앙상블 예측
    ensemble_pred, all_predictions = predictor.predict_ensemble(X, timestamps)
    
    if len(ensemble_pred) == 0:
        print("예측 실패")
        return None, None
    
    # 6. 결과 출력
    results_df = predictor.display_results(ensemble_pred, timestamps)
    
    print("\n" + "="*60)
    print("예측 완료!")
    print("="*60)
    
    # 앙상블 예측값 반환
    print("\nensemble_values:")
    print(ensemble_pred)
    
    return ensemble_pred, results_df

if __name__ == "__main__":
    ensemble_values, results = main()