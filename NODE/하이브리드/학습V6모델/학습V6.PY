"""
학습V6.py - 반도체 물류 예측 시스템 (전체 구간 + 1400-1700+ 특화)
M14 패턴 통합 하이브리드 모델
TensorFlow 2.15.0
학습 데이터: 20240201_TO_202507281705.CSV
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import json
import os
warnings.filterwarnings('ignore')

print("="*60)
print("🚀 반도체 물류 예측 시스템 V6")
print(f"📦 TensorFlow 버전: {tf.__version__}")
print("="*60)

# ============================================
# 1. 설정 클래스
# ============================================
class Config:
    # 데이터 파일
    DATA_FILE = '20240201_TO_202507281705.CSV'  # 통합 학습 데이터
    
    # 데이터 설정
    LOOKBACK = 100  # 과거 100분
    FORECAST = 1    # 10분 후 예측
    
    # M14 임계값 (발견된 패턴)
    M14B_THRESHOLDS = {
        1400: 320,
        1500: 400,
        1600: 450,
        1700: 500
    }
    
    RATIO_THRESHOLDS = {
        1400: 4,
        1500: 5,
        1600: 6,
        1700: 7
    }
    
    # 학습 설정
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    PATIENCE = 15
    
    # 모델 저장 경로
    MODEL_DIR = './models_v6/'
    
    # 가중치 설정
    SPIKE_WEIGHTS = {
        'normal': 1.0,      # < 1400
        'level_1400': 3.0,  # 1400-1499
        'level_1500': 5.0,  # 1500-1599
        'level_1600': 8.0,  # 1600-1699
        'level_1700': 10.0  # >= 1700
    }

# ============================================
# 2. 데이터 전처리
# ============================================
class DataPreprocessorV6:
    def __init__(self):
        self.value_scaler = RobustScaler()
        self.m14_scaler = StandardScaler()
        self.feature_names = []
        
    def prepare_data(self, file_path=None):
        """통합 데이터 로드 및 전처리"""
        if file_path is None:
            file_path = Config.DATA_FILE
            
        print(f"\n📂 데이터 로딩: {file_path}")
        
        # 데이터 로드
        df = pd.read_csv(file_path)
        print(f"  데이터 크기: {len(df)}행 × {len(df.columns)}열")
        
        # 컬럼명 확인 및 정리
        print(f"  컬럼 확인 중...")
        
        # 필수 컬럼 확인
        required_cols = ['TOTALCNT', 'M14AM10A', 'M14AM14B', 'M14AM16']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"  ⚠️ 경고: 누락된 컬럼 {missing_cols}")
            print(f"  사용 가능한 컬럼: {df.columns.tolist()[:10]}...")
        
        # 시간 컬럼 처리 (CURRTIME 또는 TIME)
        if 'CURRTIME' in df.columns:
            df['TIME'] = df['CURRTIME']
        elif 'TIME' not in df.columns:
            # 시간 컬럼이 없으면 인덱스로 생성
            df['TIME'] = range(len(df))
        
        # TOTALCNT를 현재값으로 사용
        if 'TOTALCNT' in df.columns:
            df['current_value'] = df['TOTALCNT']
        else:
            print("  ⚠️ TOTALCNT 컬럼이 없습니다. 첫 번째 수치 컬럼 사용")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df['current_value'] = df[numeric_cols[0]]
        
        # 타겟 생성 (10분 후 값)
        df['target'] = df['current_value'].shift(-10)  # 10분 후 값
        
        # M14 컬럼 확인
        for col in ['M14AM10A', 'M14AM14B', 'M14AM16']:
            if col not in df.columns:
                print(f"  ⚠️ {col} 컬럼이 없습니다. 0으로 초기화")
                df[col] = 0
        
        # 결측치 제거
        df = df.dropna(subset=['target'])
        
        print(f"  전처리 후: {len(df)}행")
        
        # 특징 생성
        print("\n🔧 특징 생성 중...")
        df = self._create_features(df)
        
        # 타겟 레벨 생성 (다단계)
        df['target_level'] = pd.cut(
            df['target'],
            bins=[-np.inf, 1400, 1500, 1600, 1700, np.inf],
            labels=[0, 1, 2, 3, 4]
        )
        
        # 데이터 통계
        print(f"\n📊 데이터 통계:")
        print(f"  현재값 범위: {df['current_value'].min():.0f} ~ {df['current_value'].max():.0f}")
        print(f"  타겟값 범위: {df['target'].min():.0f} ~ {df['target'].max():.0f}")
        print(f"  1400+ 비율: {(df['target'] >= 1400).mean():.1%}")
        print(f"  1500+ 비율: {(df['target'] >= 1500).mean():.1%}")
        print(f"  1600+ 비율: {(df['target'] >= 1600).mean():.1%}")
        print(f"  1700+ 비율: {(df['target'] >= 1700).mean():.1%}")
        
        return df
    
    def _create_features(self, df):
        """M14 기반 특징 생성"""
        
        # 1. 기본 M14 특징
        df['M14B_norm'] = df['M14AM14B'] / 600  # 정규화
        df['M10A_inverse'] = (100 - df['M14AM10A']) / 100  # 역패턴
        df['M16_norm'] = df['M14AM16'] / 100
        
        # 2. 비율 특징 (핵심!)
        df['ratio_14B_10A'] = df['M14AM14B'] / df['M14AM10A'].clip(lower=1)
        df['ratio_14B_16'] = df['M14AM14B'] / df['M14AM16'].clip(lower=1)
        df['ratio_10A_16'] = df['M14AM10A'] / df['M14AM16'].clip(lower=1)
        
        # 3. 변화량 특징
        for col in ['current_value', 'M14AM14B', 'M14AM10A', 'M14AM16']:
            if col in df.columns:
                df[f'{col}_change_5'] = df[col].diff(5)
                df[f'{col}_change_10'] = df[col].diff(10)
                df[f'{col}_ma_10'] = df[col].rolling(10, min_periods=1).mean()
                df[f'{col}_std_10'] = df[col].rolling(10, min_periods=1).std()
        
        # 4. M14AM10A 역패턴 특징
        df['M10A_drop_5'] = -df['M14AM10A'].diff(5)  # 감소가 양수
        df['M10A_drop_10'] = -df['M14AM10A'].diff(10)
        
        # 5. 급변 신호 특징
        df['signal_1400'] = (df['M14AM14B'] >= Config.M14B_THRESHOLDS[1400]).astype(float)
        df['signal_1500'] = (df['M14AM14B'] >= Config.M14B_THRESHOLDS[1500]).astype(float)
        df['signal_1600'] = (df['M14AM14B'] >= Config.M14B_THRESHOLDS[1600]).astype(float)
        df['signal_1700'] = (df['M14AM14B'] >= Config.M14B_THRESHOLDS[1700]).astype(float)
        
        # 6. 비율 임계값 신호
        df['ratio_signal_1400'] = (df['ratio_14B_10A'] >= Config.RATIO_THRESHOLDS[1400]).astype(float)
        df['ratio_signal_1500'] = (df['ratio_14B_10A'] >= Config.RATIO_THRESHOLDS[1500]).astype(float)
        df['ratio_signal_1600'] = (df['ratio_14B_10A'] >= Config.RATIO_THRESHOLDS[1600]).astype(float)
        df['ratio_signal_1700'] = (df['ratio_14B_10A'] >= Config.RATIO_THRESHOLDS[1700]).astype(float)
        
        # 7. 조합 특징
        df['m14b_high_m10a_low'] = ((df['M14AM14B'] >= 350) & (df['M14AM10A'] < 70)).astype(float)
        df['spike_imminent'] = ((df['M14AM14B'] >= 400) | (df['ratio_14B_10A'] >= 5)).astype(float)
        
        # 8. 통계 특징
        df['current_vs_ma'] = df['current_value'] / df['current_value_ma_10'].clip(lower=1)
        df['m14b_vs_ma'] = df['M14AM14B'] / df['M14AM14B_ma_10'].clip(lower=1)
        
        # 특징 이름 저장
        exclude_cols = ['TIME', 'CURRTIME', 'target', 'target_level', 'TOTALCNT']
        self.feature_names = [col for col in df.columns if col not in exclude_cols]
        
        # 결측치 처리
        df[self.feature_names] = df[self.feature_names].fillna(0)
        
        print(f"  생성된 특징 수: {len(self.feature_names)}개")
        
        return df
    
    def create_sequences(self, df):
        """시계열 시퀀스 생성"""
        print("\n📊 시퀀스 생성 중...")
        
        # 사용할 특징 선택
        feature_cols = [col for col in self.feature_names if col in df.columns]
        
        X, y, m14_features = [], [], []
        
        for i in range(Config.LOOKBACK, len(df) - Config.FORECAST):
            # 시계열 데이터 (과거 100분)
            X.append(df[feature_cols].iloc[i-Config.LOOKBACK:i].values)
            
            # 타겟 (10분 후)
            y.append(df['target'].iloc[i])
            
            # 현재 M14 특징 (앙상블 가중치용)
            m14_features.append([
                df['M14AM14B'].iloc[i],
                df['M14AM10A'].iloc[i],
                df['M14AM16'].iloc[i],
                df['ratio_14B_10A'].iloc[i] if 'ratio_14B_10A' in df.columns else 0
            ])
        
        X = np.array(X)
        y = np.array(y)
        m14_features = np.array(m14_features)
        
        print(f"  시퀀스 shape: X={X.shape}, y={y.shape}")
        print(f"  특징 수: {X.shape[2]}개")
        
        return X, y, m14_features

# ============================================
# 3. 모델 정의
# ============================================
class ModelsV6:
    
    @staticmethod
    def build_enhanced_gru(input_shape):
        """개선된 GRU 모델 (전체 구간 안정적 예측)"""
        inputs = tf.keras.Input(shape=input_shape, name='time_series_input')
        
        # Layer Normalization
        x = tf.keras.layers.LayerNormalization()(inputs)
        
        # Stacked GRU with residual
        gru1 = tf.keras.layers.GRU(128, return_sequences=True, dropout=0.2)(x)
        gru2 = tf.keras.layers.GRU(128, return_sequences=True, dropout=0.2)(gru1)
        
        # Residual connection
        residual = tf.keras.layers.Add()([gru1, gru2])
        
        # Final GRU
        gru3 = tf.keras.layers.GRU(64, dropout=0.2)(residual)
        
        # Dense layers
        dense1 = tf.keras.layers.Dense(128, activation='relu')(gru3)
        dropout = tf.keras.layers.Dropout(0.3)(dense1)
        dense2 = tf.keras.layers.Dense(64, activation='relu')(dropout)
        
        # Output
        output = tf.keras.layers.Dense(1, name='prediction')(dense2)
        
        model = tf.keras.Model(inputs=inputs, outputs=output, name='Enhanced_GRU')
        return model
    
    @staticmethod
    def build_spike_detector(input_shape):
        """M14 기반 급변 감지 CNN-LSTM 모델"""
        inputs = tf.keras.Input(shape=input_shape, name='time_series_input')
        
        # Multi-scale CNN for pattern detection
        conv1 = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
        conv2 = tf.keras.layers.Conv1D(64, 5, activation='relu', padding='same')(inputs)
        conv3 = tf.keras.layers.Conv1D(64, 7, activation='relu', padding='same')(inputs)
        
        # Concatenate multi-scale features
        concat = tf.keras.layers.Concatenate()([conv1, conv2, conv3])
        
        # Batch normalization
        norm = tf.keras.layers.BatchNormalization()(concat)
        
        # Attention mechanism
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=4, 
            key_dim=48,
            dropout=0.2
        )(norm, norm)
        
        # BiLSTM
        lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.2)
        )(attention)
        
        # Global pooling
        pooled = tf.keras.layers.GlobalAveragePooling1D()(lstm)
        
        # Dense layers
        dense1 = tf.keras.layers.Dense(256, activation='relu')(pooled)
        dropout1 = tf.keras.layers.Dropout(0.3)(dense1)
        dense2 = tf.keras.layers.Dense(128, activation='relu')(dropout1)
        dropout2 = tf.keras.layers.Dropout(0.2)(dense2)
        
        # Dual output
        regression_output = tf.keras.layers.Dense(1, name='spike_value')(dropout2)
        classification_output = tf.keras.layers.Dense(1, activation='sigmoid', name='spike_prob')(dropout2)
        
        model = tf.keras.Model(
            inputs=inputs,
            outputs=[regression_output, classification_output],
            name='Spike_Detector'
        )
        return model
    
    @staticmethod
    def build_ensemble_model(gru_model, spike_model, m14_input_shape):
        """동적 앙상블 모델"""
        # 입력
        time_series_input = tf.keras.Input(shape=gru_model.input_shape[1:], name='time_series')
        m14_input = tf.keras.Input(shape=m14_input_shape, name='m14_features')
        
        # 각 모델 예측
        gru_pred = gru_model(time_series_input)
        spike_pred, spike_prob = spike_model(time_series_input)
        
        # M14 기반 가중치 계산
        weight_dense = tf.keras.layers.Dense(16, activation='relu')(m14_input)
        weight_dense = tf.keras.layers.Dense(8, activation='relu')(weight_dense)
        weights = tf.keras.layers.Dense(2, activation='softmax', name='weights')(weight_dense)
        
        # 가중 평균
        gru_weight = tf.keras.layers.Lambda(lambda x: x[:, 0:1])(weights)
        spike_weight = tf.keras.layers.Lambda(lambda x: x[:, 1:2])(weights)
        
        weighted_gru = tf.keras.layers.Multiply()([gru_pred, gru_weight])
        weighted_spike = tf.keras.layers.Multiply()([spike_pred, spike_weight])
        
        # 최종 예측
        final_pred = tf.keras.layers.Add()([weighted_gru, weighted_spike])
        
        # M14 규칙 기반 보정
        final_pred = M14RuleCorrection()([final_pred, m14_input])
        
        model = tf.keras.Model(
            inputs=[time_series_input, m14_input],
            outputs=[final_pred, spike_prob],
            name='Ensemble_Model'
        )
        return model

# ============================================
# 4. 커스텀 레이어 및 손실 함수
# ============================================
class M14RuleCorrection(tf.keras.layers.Layer):
    """M14 규칙 기반 보정 레이어"""
    def __init__(self):
        super().__init__()
        
    def call(self, inputs):
        pred, m14_features = inputs
        
        # M14 특징 분해
        m14b = m14_features[:, 0:1]
        m10a = m14_features[:, 1:2]
        ratio = m14_features[:, 3:4]
        
        # 규칙 기반 보정
        # 1700+ 신호
        condition_1700 = tf.logical_and(
            tf.greater_equal(m14b, 500),
            tf.greater_equal(ratio, 7)
        )
        pred = tf.where(condition_1700, tf.maximum(pred, 1700), pred)
        
        # 1600+ 신호
        condition_1600 = tf.logical_and(
            tf.greater_equal(m14b, 450),
            tf.greater_equal(ratio, 6)
        )
        pred = tf.where(condition_1600, tf.maximum(pred, 1600), pred)
        
        # 1500+ 신호
        condition_1500 = tf.logical_and(
            tf.greater_equal(m14b, 400),
            tf.greater_equal(ratio, 5)
        )
        pred = tf.where(condition_1500, tf.maximum(pred, 1500), pred)
        
        # 1400+ 신호
        condition_1400 = tf.greater_equal(m14b, 320)
        pred = tf.where(condition_1400, tf.maximum(pred, 1400), pred)
        
        # M10A 역패턴 보정
        condition_inverse = tf.logical_and(
            tf.less(m10a, 70),
            tf.greater_equal(m14b, 250)
        )
        pred = tf.where(condition_inverse, pred * 1.08, pred)
        
        return pred

class WeightedLoss(tf.keras.losses.Loss):
    """레벨별 가중 손실 함수"""
    def __init__(self):
        super().__init__()
        self.mae = tf.keras.losses.MeanAbsoluteError()
        
    def call(self, y_true, y_pred):
        # 레벨별 가중치 적용
        weights = tf.where(y_true < 1400, 1.0,
                 tf.where(y_true < 1500, 3.0,
                 tf.where(y_true < 1600, 5.0,
                 tf.where(y_true < 1700, 8.0, 10.0))))
        
        # 가중 MAE
        mae = tf.abs(y_true - y_pred)
        weighted_mae = mae * weights
        
        return tf.reduce_mean(weighted_mae)

# ============================================
# 5. 콜백 및 평가
# ============================================
class SpikePerformanceCallback(tf.keras.callbacks.Callback):
    """1400+ 구간 성능 모니터링"""
    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.best_spike_mae = float('inf')
        
    def on_epoch_end(self, epoch, logs=None):
        # 예측
        y_pred = self.model.predict(self.X_val, verbose=0)
        if isinstance(y_pred, list):
            y_pred = y_pred[0]
        y_pred = y_pred.flatten()
        
        # 1400+ 구간 성능
        spike_mask = self.y_val >= 1400
        if np.any(spike_mask):
            spike_mae = np.mean(np.abs(self.y_val[spike_mask] - y_pred[spike_mask]))
            
            # 레벨별 성능
            level_performance = {}
            for level in [1400, 1500, 1600, 1700]:
                level_mask = self.y_val >= level
                if np.any(level_mask):
                    recall = np.sum((y_pred >= level) & level_mask) / np.sum(level_mask)
                    level_performance[f'{level}+'] = recall
            
            # 출력
            if spike_mae < self.best_spike_mae:
                self.best_spike_mae = spike_mae
                print(f"\n🎯 Epoch {epoch+1} - 1400+ MAE: {spike_mae:.2f} (Best!)")
                for level, recall in level_performance.items():
                    print(f"   {level} Recall: {recall:.2%}")

# ============================================
# 6. 메인 학습 파이프라인
# ============================================
class TrainingPipelineV6:
    def __init__(self):
        self.preprocessor = DataPreprocessorV6()
        self.models = {}
        self.history = {}
        
        # 디렉토리 생성
        os.makedirs(Config.MODEL_DIR, exist_ok=True)
        
    def run(self):
        """전체 학습 파이프라인 실행"""
        print("\n" + "="*60)
        print("🚀 학습 파이프라인 시작")
        print("="*60)
        
        # 1. 데이터 준비
        df = self.preprocessor.prepare_data()
        X, y, m14_features = self.preprocessor.create_sequences(df)
        
        # 2. 학습/검증 분할
        X_train, X_val, y_train, y_val, m14_train, m14_val = train_test_split(
            X, y, m14_features, test_size=0.2, random_state=42
        )
        
        print(f"\n📊 데이터 분할:")
        print(f"  학습: {X_train.shape[0]}개")
        print(f"  검증: {X_val.shape[0]}개")
        
        # 3. 스케일링
        print("\n📏 데이터 스케일링...")
        
        # 각 특징별 스케일링
        n_features = X_train.shape[2]
        for i in range(n_features):
            scaler = RobustScaler()
            X_train[:, :, i] = scaler.fit_transform(X_train[:, :, i].reshape(-1, 1)).reshape(X_train[:, :, i].shape)
            X_val[:, :, i] = scaler.transform(X_val[:, :, i].reshape(-1, 1)).reshape(X_val[:, :, i].shape)
        
        # 4. 모델 학습
        self._train_models(X_train, y_train, X_val, y_val, m14_train, m14_val)
        
        # 5. 평가
        self._evaluate_models(X_val, y_val, m14_val)
        
        # 6. 결과 저장
        self._save_results()
        
        print("\n" + "="*60)
        print("✅ 학습 완료!")
        print("="*60)
    
    def _train_models(self, X_train, y_train, X_val, y_val, m14_train, m14_val):
        """모델 학습"""
        print("\n🏋️ 모델 학습 시작...")
        
        # 1. GRU 모델
        print("\n1️⃣ Enhanced GRU 학습")
        gru_model = ModelsV6.build_enhanced_gru(X_train.shape[1:])
        gru_model.compile(
            optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE),
            loss=WeightedLoss(),
            metrics=['mae']
        )
        
        gru_history = gru_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=Config.EPOCHS,
            batch_size=Config.BATCH_SIZE,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=Config.PATIENCE, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
                SpikePerformanceCallback(X_val, y_val)
            ],
            verbose=1
        )
        
        self.models['gru'] = gru_model
        self.history['gru'] = gru_history
        
        # 2. Spike Detector 모델
        print("\n2️⃣ Spike Detector 학습")
        spike_model = ModelsV6.build_spike_detector(X_train.shape[1:])
        
        # 1400+ 여부 레이블 생성
        y_spike_class = (y_train >= 1400).astype(float)
        y_val_spike_class = (y_val >= 1400).astype(float)
        
        spike_model.compile(
            optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE),
            loss={
                'spike_value': WeightedLoss(),
                'spike_prob': 'binary_crossentropy'
            },
            loss_weights={
                'spike_value': 1.0,
                'spike_prob': 0.5
            },
            metrics={
                'spike_value': 'mae',
                'spike_prob': 'accuracy'
            }
        )
        
        spike_history = spike_model.fit(
            X_train, 
            {'spike_value': y_train, 'spike_prob': y_spike_class},
            validation_data=(
                X_val, 
                {'spike_value': y_val, 'spike_prob': y_val_spike_class}
            ),
            epochs=Config.EPOCHS,
            batch_size=Config.BATCH_SIZE,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=Config.PATIENCE, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
            ],
            verbose=1
        )
        
        self.models['spike'] = spike_model
        self.history['spike'] = spike_history
        
        # 3. 앙상블 모델
        print("\n3️⃣ 앙상블 모델 구성")
        ensemble_model = ModelsV6.build_ensemble_model(
            gru_model, 
            spike_model,
            m14_train.shape[1]
        )
        
        ensemble_model.compile(
            optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE * 0.5),
            loss={
                'tf.math.add': WeightedLoss(),
                'spike_prob': 'binary_crossentropy'
            },
            metrics=['mae']
        )
        
        # 앙상블 파인튜닝
        ensemble_history = ensemble_model.fit(
            [X_train, m14_train],
            [y_train, y_spike_class],
            validation_data=(
                [X_val, m14_val],
                [y_val, y_val_spike_class]
            ),
            epochs=20,
            batch_size=Config.BATCH_SIZE,
            verbose=1
        )
        
        self.models['ensemble'] = ensemble_model
        self.history['ensemble'] = ensemble_history
    
    def _evaluate_models(self, X_val, y_val, m14_val):
        """모델 평가"""
        print("\n📊 모델 평가")
        print("="*60)
        
        results = {}
        
        for name, model in self.models.items():
            if name == 'ensemble':
                pred = model.predict([X_val, m14_val], verbose=0)[0].flatten()
            else:
                pred = model.predict(X_val, verbose=0)
                if isinstance(pred, list):
                    pred = pred[0]
                pred = pred.flatten()
            
            # 전체 성능
            mae = np.mean(np.abs(y_val - pred))
            
            # 구간별 성능
            level_performance = {}
            for level in [1400, 1500, 1600, 1700]:
                mask = y_val >= level
                if np.any(mask):
                    recall = np.sum((pred >= level) & mask) / np.sum(mask)
                    level_mae = np.mean(np.abs(y_val[mask] - pred[mask]))
                    level_performance[level] = {
                        'recall': recall,
                        'mae': level_mae,
                        'count': np.sum(mask)
                    }
            
            results[name] = {
                'overall_mae': mae,
                'levels': level_performance
            }
            
            # 출력
            print(f"\n🎯 {name.upper()} 모델:")
            print(f"  전체 MAE: {mae:.2f}")
            for level, perf in level_performance.items():
                print(f"  {level}+: Recall={perf['recall']:.2%}, MAE={perf['mae']:.1f} (n={perf['count']})")
        
        self.evaluation_results = results
        
        # 최종 선택
        best_model = min(results.keys(), key=lambda x: results[x]['overall_mae'])
        print(f"\n🏆 최고 성능: {best_model.upper()} 모델")
    
    def _save_results(self):
        """모델 및 결과 저장"""
        print("\n💾 모델 저장 중...")
        
        # 모델 저장
        for name, model in self.models.items():
            model.save(f"{Config.MODEL_DIR}{name}_model.h5")
            print(f"  {name}_model.h5 저장 완료")
        
        # 평가 결과 저장
        with open(f"{Config.MODEL_DIR}evaluation_results.json", 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)
        
        # 설정 저장
        config_dict = {k: v for k, v in Config.__dict__.items() if not k.startswith('_')}
        with open(f"{Config.MODEL_DIR}config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print("  결과 파일 저장 완료")
    
    def visualize_results(self):
        """결과 시각화"""
        print("\n📈 결과 시각화 생성 중...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 학습 곡선
        for idx, (name, history) in enumerate(self.history.items()):
            if idx < 3:
                ax = axes[idx//2, idx%2]
                ax.plot(history.history['loss'], label='Train Loss')
                ax.plot(history.history['val_loss'], label='Val Loss')
                ax.set_title(f'{name.upper()} Learning Curve')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.legend()
                ax.grid(True)
        
        # 4. 성능 비교
        ax = axes[1, 1]
        models = list(self.evaluation_results.keys())
        maes = [self.evaluation_results[m]['overall_mae'] for m in models]
        
        bars = ax.bar(models, maes, color=['blue', 'green', 'red'])
        ax.set_title('Model Performance Comparison')
        ax.set_ylabel('MAE')
        ax.set_ylim(0, max(maes) * 1.2)
        
        # 값 표시
        for bar, mae in zip(bars, maes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mae:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{Config.MODEL_DIR}training_results.png", dpi=100)
        print("  training_results.png 저장 완료")
        plt.show()

# ============================================
# 7. 실행
# ============================================
if __name__ == "__main__":
    # GPU 설정
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"🎮 GPU 사용: {len(gpus)}개")
        except RuntimeError as e:
            print(f"⚠️ GPU 설정 오류: {e}")
    else:
        print("💻 CPU 모드로 실행")
    
    # 파이프라인 실행
    pipeline = TrainingPipelineV6()
    pipeline.run()
    pipeline.visualize_results()
    
    print("\n" + "="*60)
    print("🎉 모든 작업 완료!")
    print(f"📁 저장 위치: {Config.MODEL_DIR}")
    print(f"📂 학습 데이터: {Config.DATA_FILE}")
    print("="*60)