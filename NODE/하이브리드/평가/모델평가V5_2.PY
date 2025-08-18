"""
반도체 물류 이동량 예측 하이브리드 모델
- TensorFlow 2.15.0
- GRU, CNN_LSTM, 앙상블 (ARIMA 통합)
- 1400+ 급증 구간 특화
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Input, Dense, Dropout, BatchNormalization,
                                     GRU, Conv1D, MaxPooling1D, Bidirectional, 
                                     Attention, Add, LayerNormalization, Multiply)
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print(f"TensorFlow Version: {tf.__version__}")

class SpikeAwareHybridModel:
    """1400+ 급증 구간 특화 하이브리드 모델"""
    
    def __init__(self, sequence_length=100, prediction_horizon=10):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.spike_threshold = 1400
        self.models = {}
        self.scaler = RobustScaler()
        self.spike_scaler = RobustScaler()  # 급증 구간 전용 스케일러
        
    def create_spike_features(self, df):
        """급증 구간 특화 특징 생성"""
        df = df.copy()
        
        # 기본 특징
        df['hour'] = df['datetime'].dt.hour
        df['dayofweek'] = df['datetime'].dt.dayofweek
        df['is_weekend'] = (df['datetime'].dt.dayofweek >= 5).astype(int)
        
        # 이동평균 (다양한 윈도우)
        for window in [5, 10, 20, 30, 60, 100]:
            df[f'MA_{window}'] = df['TOTALCNT'].rolling(window, min_periods=1).mean()
        
        # 표준편차
        df['STD_10'] = df['TOTALCNT'].rolling(10, min_periods=1).std().fillna(0)
        df['STD_30'] = df['TOTALCNT'].rolling(30, min_periods=1).std().fillna(0)
        df['STD_60'] = df['TOTALCNT'].rolling(60, min_periods=1).std().fillna(0)
        
        # 변화율
        df['change_rate'] = df['TOTALCNT'].pct_change().fillna(0)
        df['change_rate_10'] = df['TOTALCNT'].pct_change(10).fillna(0)
        df['change_rate_30'] = df['TOTALCNT'].pct_change(30).fillna(0)
        
        # 급증 관련 특징 ★
        df['momentum'] = df['TOTALCNT'] - df['TOTALCNT'].shift(10)
        df['acceleration'] = df['momentum'] - df['momentum'].shift(10)
        
        # 볼린저 밴드
        df['BB_upper'] = df['MA_20'] + (2 * df['STD_10'])
        df['BB_lower'] = df['MA_20'] - (2 * df['STD_10'])
        df['BB_position'] = (df['TOTALCNT'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'] + 1e-6)
        
        # RSI (상대강도지수)
        delta = df['TOTALCNT'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-6)))
        
        # 급증 신호
        df['spike_signal'] = (df['TOTALCNT'] > df['MA_30'] * 1.5).astype(int)
        df['pre_spike_pattern'] = (df['momentum'] > df['momentum'].quantile(0.8)).astype(int)
        
        # 트렌드 강도
        df['trend_strength'] = (df['MA_10'] - df['MA_60']) / (df['STD_30'] + 1e-6)
        
        # 변동성 지표
        df['volatility'] = df['STD_30'] / (df['MA_30'] + 1e-6)
        
        # NaN 처리
        df.ffill(inplace=True)
        df.fillna(0, inplace=True)
        
        return df
    
    def build_improved_gru(self, input_shape):
        """향상된 GRU 모델 (급증 구간 최적화)"""
        inputs = Input(shape=input_shape)
        
        # 첫 번째 GRU 블록
        x = GRU(256, return_sequences=True, 
                kernel_regularizer=l1_l2(l1=0.001, l2=0.001))(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Attention 메커니즘 추가
        attention = Attention()([x, x])
        x = Add()([x, attention])
        x = LayerNormalization()(x)
        
        # 두 번째 GRU 블록
        x = GRU(128, return_sequences=True)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # 세 번째 GRU 블록
        x = GRU(64, return_sequences=False)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Dense layers
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.1)(x)
        
        # 출력층
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def build_spike_cnn_lstm(self, input_shape):
        """급증 감지 특화 CNN-LSTM"""
        inputs = Input(shape=input_shape)
        
        # CNN 블록 1 (패턴 감지)
        conv1 = Conv1D(128, 3, activation='relu', padding='same')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv1D(128, 3, activation='relu', padding='same')(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling1D(2)(conv1)
        
        # CNN 블록 2 (급증 패턴)
        conv2 = Conv1D(64, 5, activation='relu', padding='same')(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Conv1D(64, 5, activation='relu', padding='same')(conv2)
        conv2 = BatchNormalization()(conv2)
        
        # Bidirectional GRU
        x = Bidirectional(GRU(128, return_sequences=True))(conv2)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Bidirectional(GRU(64, return_sequences=False))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Dense layers
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.1)(x)
        
        # 출력
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def create_weighted_loss(self):
        """급증 구간 가중 손실 함수"""
        def weighted_mae(y_true, y_pred):
            # 원본 스케일로 역변환 (근사치)
            y_true_scaled = y_true * 1000  # 스케일 복원 근사
            
            # 급증 구간 가중치
            spike_weight = tf.where(y_true_scaled > 1400, 3.0, 1.0)
            
            # 가중 MAE
            mae = tf.abs(y_true - y_pred)
            weighted_mae = mae * spike_weight
            
            return tf.reduce_mean(weighted_mae)
        
        return weighted_mae
    
    def train_arima_component(self, train_data, order=(2,1,1)):
        """ARIMA 컴포넌트 학습"""
        try:
            model = ARIMA(train_data, order=order)
            arima_fit = model.fit()
            return arima_fit
        except:
            # 실패 시 단순 ARIMA
            model = ARIMA(train_data, order=(1,0,1))
            return model.fit()
    
    def create_ensemble_model(self, X_train, y_train, X_val, y_val):
        """앙상블 모델 생성 (GRU + CNN_LSTM + ARIMA)"""
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        # 1. GRU 모델
        print("\n[1/3] GRU 모델 학습...")
        self.models['gru'] = self.build_improved_gru(input_shape)
        self.models['gru'].compile(
            optimizer=Adam(learning_rate=0.001),
            loss=self.create_weighted_loss(),
            metrics=['mae']
        )
        
        # GRU 학습
        history_gru = self.models['gru'].fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=256,
            callbacks=[
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
            ],
            verbose=1
        )
        
        # 2. CNN-LSTM 모델
        print("\n[2/3] CNN-LSTM 모델 학습...")
        self.models['cnn_lstm'] = self.build_spike_cnn_lstm(input_shape)
        self.models['cnn_lstm'].compile(
            optimizer=Adam(learning_rate=0.001),
            loss=self.create_weighted_loss(),
            metrics=['mae']
        )
        
        # CNN-LSTM 학습
        history_cnn = self.models['cnn_lstm'].fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=256,
            callbacks=[
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
            ],
            verbose=1
        )
        
        # 3. ARIMA 컴포넌트
        print("\n[3/3] ARIMA 컴포넌트 학습...")
        # 최근 1000개 데이터로 ARIMA 학습
        recent_data = y_train[-1000:]
        self.arima_model = self.train_arima_component(recent_data)
        
        return history_gru, history_cnn
    
    def predict_ensemble(self, X_test, use_arima=True):
        """앙상블 예측 (급증 구간 강화)"""
        predictions = {}
        
        # GRU 예측
        pred_gru = self.models['gru'].predict(X_test, batch_size=256, verbose=0)
        predictions['gru'] = pred_gru.flatten()
        
        # CNN-LSTM 예측
        pred_cnn = self.models['cnn_lstm'].predict(X_test, batch_size=256, verbose=0)
        predictions['cnn_lstm'] = pred_cnn.flatten()
        
        # ARIMA 보정 (옵션)
        if use_arima and hasattr(self, 'arima_model'):
            arima_forecast = self.arima_model.forecast(steps=len(X_test))
            arima_scaled = self.scaler.transform(arima_forecast.values.reshape(-1, 1)).flatten()
            predictions['arima'] = arima_scaled
        
        # 급증 구간 감지
        # 최근 패턴으로 급증 가능성 평가
        spike_probability = self.detect_spike_probability(X_test)
        
        # 가중 앙상블
        if use_arima and 'arima' in predictions:
            # 급증 구간: CNN-LSTM 가중치 증가
            weights = np.where(spike_probability > 0.5,
                             [0.3, 0.5, 0.2],  # GRU, CNN-LSTM, ARIMA (급증)
                             [0.5, 0.3, 0.2])  # GRU, CNN-LSTM, ARIMA (정상)
            
            ensemble_pred = (predictions['gru'] * weights[:, 0] + 
                           predictions['cnn_lstm'] * weights[:, 1] + 
                           predictions['arima'] * weights[:, 2])
        else:
            # ARIMA 없이
            weights = np.where(spike_probability > 0.5,
                             [0.4, 0.6],  # GRU, CNN-LSTM (급증)
                             [0.6, 0.4])  # GRU, CNN-LSTM (정상)
            
            ensemble_pred = (predictions['gru'] * weights[:, 0] + 
                           predictions['cnn_lstm'] * weights[:, 1])
        
        predictions['ensemble'] = ensemble_pred
        
        return predictions
    
    def detect_spike_probability(self, X):
        """급증 가능성 감지"""
        # 입력 데이터의 마지막 10개 시점 분석
        recent_values = X[:, -10:, 0]  # TOTALCNT
        
        # 상승 추세 감지
        trend = np.mean(np.diff(recent_values, axis=1), axis=1)
        
        # 변동성 감지
        volatility = np.std(recent_values, axis=1)
        
        # 급증 확률 계산
        spike_prob = 1 / (1 + np.exp(-(trend * 0.1 + volatility * 0.05)))
        
        return spike_prob
    
    def evaluate_predictions(self, y_true, predictions, original_scale=True):
        """예측 평가 (급증 구간 특화)"""
        results = {}
        
        for name, y_pred in predictions.items():
            # 기본 메트릭
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            
            # 급증 구간 평가
            if original_scale:
                spike_mask = y_true >= self.spike_threshold
                if spike_mask.sum() > 0:
                    spike_mae = mean_absolute_error(y_true[spike_mask], y_pred[spike_mask])
                    
                    # 급증 감지 성능
                    pred_spike = y_pred >= self.spike_threshold
                    actual_spike = y_true >= self.spike_threshold
                    
                    tp = np.sum((pred_spike) & (actual_spike))
                    fp = np.sum((pred_spike) & (~actual_spike))
                    fn = np.sum((~pred_spike) & (actual_spike))
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                else:
                    spike_mae = precision = recall = f1 = 0
            else:
                spike_mae = precision = recall = f1 = 0
            
            results[name] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'Spike_MAE': spike_mae,
                'Precision': precision,
                'Recall': recall,
                'F1': f1
            }
        
        return results
    
    def visualize_results(self, y_true, predictions, title="예측 결과"):
        """결과 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # 1. 시계열 비교
        ax1 = axes[0, 0]
        sample = min(500, len(y_true))
        
        ax1.plot(y_true[:sample], label='실제', color='black', linewidth=2)
        
        colors = {'gru': 'green', 'cnn_lstm': 'orange', 'ensemble': 'red'}
        for name in ['gru', 'cnn_lstm', 'ensemble']:
            if name in predictions:
                ax1.plot(predictions[name][:sample], label=name.upper(), 
                        alpha=0.7, color=colors[name])
        
        ax1.axhline(y=self.spike_threshold, color='red', linestyle='--', 
                   alpha=0.3, label='1400 임계값')
        ax1.set_xlabel('시간 인덱스')
        ax1.set_ylabel('물류량')
        ax1.set_title('예측 결과 비교')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 급증 구간 포커스
        ax2 = axes[0, 1]
        spike_indices = np.where(y_true >= self.spike_threshold)[0]
        
        if len(spike_indices) > 0:
            for idx in spike_indices[:20]:  # 처음 20개만
                if idx > 10 and idx < len(y_true) - 10:
                    ax2.plot(range(20), y_true[idx-10:idx+10], 'k-', alpha=0.3)
                    if 'ensemble' in predictions:
                        ax2.plot(range(20), predictions['ensemble'][idx-10:idx+10], 
                                'r-', alpha=0.3)
            
            ax2.axhline(y=self.spike_threshold, color='red', linestyle='--', alpha=0.5)
            ax2.set_xlabel('상대 시간')
            ax2.set_ylabel('물류량')
            ax2.set_title('급증 구간 예측')
            ax2.grid(True, alpha=0.3)
        
        # 3. 산점도
        ax3 = axes[1, 0]
        if 'ensemble' in predictions:
            ax3.scatter(y_true, predictions['ensemble'], alpha=0.5, s=1)
            ax3.plot([y_true.min(), y_true.max()], 
                    [y_true.min(), y_true.max()], 'r--', alpha=0.5)
            
            # 급증 구간 강조
            spike_mask = y_true >= self.spike_threshold
            if spike_mask.sum() > 0:
                ax3.scatter(y_true[spike_mask], predictions['ensemble'][spike_mask], 
                          color='red', s=10, alpha=0.7, label='1400+')
            
            ax3.set_xlabel('실제값')
            ax3.set_ylabel('예측값')
            ax3.set_title('앙상블 예측 산점도')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. 오차 분석
        ax4 = axes[1, 1]
        if 'ensemble' in predictions:
            errors = y_true - predictions['ensemble']
            
            # 구간별 오차
            normal_errors = errors[y_true < self.spike_threshold]
            spike_errors = errors[y_true >= self.spike_threshold]
            
            ax4.hist(normal_errors, bins=30, alpha=0.5, color='blue', 
                    label=f'정상 (MAE: {np.mean(np.abs(normal_errors)):.1f})')
            
            if len(spike_errors) > 0:
                ax4.hist(spike_errors, bins=20, alpha=0.5, color='red',
                        label=f'급증 (MAE: {np.mean(np.abs(spike_errors)):.1f})')
            
            ax4.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            ax4.set_xlabel('예측 오차')
            ax4.set_ylabel('빈도')
            ax4.set_title('오차 분포')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

# 사용 예제
def main():
    """메인 실행 함수"""
    print("="*60)
    print("반도체 물류 예측 하이브리드 모델")
    print("GRU + CNN-LSTM + ARIMA 앙상블")
    print("="*60)
    
    # 데이터 로드 (예제)
    # df = pd.read_csv('your_data.csv')
    
    # 모델 초기화
    model = SpikeAwareHybridModel(sequence_length=100, prediction_horizon=10)
    
    # 특징 생성
    # df = model.create_spike_features(df)
    
    # 학습/검증 데이터 분할
    # X_train, y_train, X_val, y_val = prepare_data(df)
    
    # 앙상블 모델 학습
    # history_gru, history_cnn = model.create_ensemble_model(
    #     X_train, y_train, X_val, y_val
    # )
    
    # 예측
    # predictions = model.predict_ensemble(X_test)
    
    # 평가
    # results = model.evaluate_predictions(y_test, predictions)
    
    print("\n모델 구조:")
    print("1. GRU: Attention 메커니즘 + 급증 구간 가중치")
    print("2. CNN-LSTM: 패턴 감지 + Bidirectional GRU")
    print("3. ARIMA: 시계열 트렌드 보정")
    print("\n특징:")
    print("- 1400+ 급증 구간 특화 손실 함수")
    print("- 동적 가중 앙상블")
    print("- 급증 확률 기반 예측 조정")

if __name__ == "__main__":
    main()