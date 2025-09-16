"""
V6 모델 평가 시스템 (모델 로드 개선판)
- 커스텀 레이어/손실함수 완벽 지원
- 모델 로드 오류 해결
- CPU 모드 최적화
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import json
import warnings
from datetime import datetime
import os
warnings.filterwarnings('ignore')

# TensorFlow 경고 억제
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("="*60)
print("📊 V6 모델 평가 시스템 (로드 개선판)")
print(f"📦 TensorFlow: {tf.__version__}")
print("="*60)

# ============================================
# 설정
# ============================================
class Config:
    # 평가 데이터 파일
    EVAL_DATA_FILE = './data/20250731_to20250806.CSV'
    
    # 학습된 모델 경로
    MODEL_DIR = './models_v6_full_train/'
    
    # 시퀀스 설정
    LOOKBACK = 100  # 과거 100분 데이터
    FORECAST = 10   # 10분 후 예측
    
    # 평가 결과 저장 경로
    EVAL_RESULT_DIR = './evaluation_results/'
    
    # 시각화 저장 경로
    PLOT_DIR = './evaluation_plots/'
    
    # CPU 모드 배치 크기
    BATCH_SIZE = 32

# 디렉토리 생성
os.makedirs(Config.EVAL_RESULT_DIR, exist_ok=True)
os.makedirs(Config.PLOT_DIR, exist_ok=True)

# ============================================
# 커스텀 레이어 및 손실 함수 정의
# ============================================

@tf.keras.utils.register_keras_serializable()
class M14RuleCorrection(tf.keras.layers.Layer):
    """M14 규칙 기반 보정 레이어"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs, training=None):
        if isinstance(inputs, list):
            pred, m14_features = inputs
        else:
            # 단일 입력인 경우
            return inputs
        
        pred = tf.cast(pred, tf.float32)
        m14_features = tf.cast(m14_features, tf.float32)
        
        # m14_features shape 확인
        if len(m14_features.shape) == 1:
            m14_features = tf.expand_dims(m14_features, axis=0)
        
        # 특징 추출
        if m14_features.shape[-1] >= 1:
            m14b = m14_features[:, 0:1]
        else:
            m14b = tf.zeros_like(pred)
            
        if m14_features.shape[-1] >= 2:
            m10a = m14_features[:, 1:2]
        else:
            m10a = tf.ones_like(pred)
            
        if m14_features.shape[-1] >= 3:
            m16 = m14_features[:, 2:3]
        else:
            m16 = tf.ones_like(pred)
            
        if m14_features.shape[-1] >= 4:
            ratio = m14_features[:, 3:4]
        else:
            ratio = tf.where(m10a > 0, m14b / (m10a + 1e-7), tf.zeros_like(pred))
        
        # 임계값 규칙 적용
        pred = tf.where(m14b >= 420, tf.maximum(pred, 1550.0), pred)
        pred = tf.where(m14b >= 380, tf.maximum(pred, 1500.0), pred)
        pred = tf.where(m14b >= 350, tf.maximum(pred, 1450.0), pred)
        pred = tf.where(m14b >= 300, tf.maximum(pred, 1400.0), pred)
        
        # 비율 보정
        pred = tf.where(ratio >= 5.5, pred * 1.15, pred)
        pred = tf.where((ratio >= 5.0) & (ratio < 5.5), pred * 1.10, pred)
        pred = tf.where((ratio >= 4.5) & (ratio < 5.0), pred * 1.08, pred)
        pred = tf.where((ratio >= 4.0) & (ratio < 4.5), pred * 1.05, pred)
        
        # 황금 패턴
        golden = (m14b >= 350) & (m10a < 70)
        pred = tf.where(golden, pred * 1.2, pred)
        
        # 범위 제한
        pred = tf.clip_by_value(pred, 1200.0, 2000.0)
        
        return pred
    
    def get_config(self):
        config = super().get_config()
        return config

@tf.keras.utils.register_keras_serializable()
class ImprovedM14RuleCorrection(M14RuleCorrection):
    """개선된 M14 규칙 보정 (호환성용)"""
    pass

@tf.keras.utils.register_keras_serializable()
class WeightedLoss(tf.keras.losses.Loss):
    """가중치 손실 함수"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        mae = tf.abs(y_true - y_pred)
        
        weights = tf.ones_like(y_true)
        weights = tf.where(y_true >= 1550, 30.0, weights)
        weights = tf.where((y_true >= 1500) & (y_true < 1550), 25.0, weights)
        weights = tf.where((y_true >= 1450) & (y_true < 1500), 20.0, weights)
        weights = tf.where((y_true >= 1400) & (y_true < 1450), 15.0, weights)
        weights = tf.where((y_true >= 1350) & (y_true < 1400), 10.0, weights)
        
        large_error = tf.where(mae > 100, mae * 0.2, 0.0)
        
        return tf.reduce_mean(mae * weights + large_error)
    
    def get_config(self):
        return super().get_config()

@tf.keras.utils.register_keras_serializable()
class ImprovedWeightedLoss(WeightedLoss):
    """개선된 가중치 손실 (호환성용)"""
    pass

# ============================================
# 모델 로드 함수 (개선된 버전)
# ============================================
def load_models():
    """학습된 모델 로드 (강화된 버전)"""
    print("\n📦 학습된 모델 로드 중...")
    
    models = {}
    model_names = ['lstm', 'gru', 'cnn_lstm', 'spike', 'rule', 'ensemble']
    
    # 모든 가능한 커스텀 객체 정의
    custom_objects = {
        'M14RuleCorrection': M14RuleCorrection,
        'ImprovedM14RuleCorrection': ImprovedM14RuleCorrection,
        'WeightedLoss': WeightedLoss,
        'ImprovedWeightedLoss': ImprovedWeightedLoss,
        # Lambda 레이어 처리
        'tf': tf,
        'Lambda': tf.keras.layers.Lambda,
    }
    
    for name in model_names:
        print(f"\n  시도 중: {name}")
        
        # 가능한 파일 확장자들 시도
        possible_paths = [
            f"{Config.MODEL_DIR}{name}_final.keras",
            f"{Config.MODEL_DIR}{name}_best.keras",
            f"{Config.MODEL_DIR}{name}_final.h5",
            f"{Config.MODEL_DIR}{name}_best.h5",
        ]
        
        model_loaded = False
        for model_path in possible_paths:
            if os.path.exists(model_path):
                print(f"    파일 발견: {model_path}")
                try:
                    # 방법 1: 일반 로드
                    model = tf.keras.models.load_model(
                        model_path,
                        custom_objects=custom_objects,
                        compile=False
                    )
                    
                    # 재컴파일
                    model.compile(
                        optimizer='adam',
                        loss='mae',
                        metrics=['mae']
                    )
                    
                    models[name] = model
                    model_loaded = True
                    print(f"    ✅ {name} 모델 로드 성공 (방법 1)")
                    break
                    
                except Exception as e1:
                    print(f"    ⚠️ 방법 1 실패: {str(e1)[:100]}...")
                    
                    try:
                        # 방법 2: 가중치만 로드 (구조 재생성)
                        print(f"    방법 2 시도 중 (가중치만 로드)...")
                        model = recreate_model_structure(name)
                        if model:
                            model.load_weights(model_path)
                            models[name] = model
                            model_loaded = True
                            print(f"    ✅ {name} 모델 로드 성공 (방법 2)")
                            break
                    except Exception as e2:
                        print(f"    ⚠️ 방법 2도 실패: {str(e2)[:100]}...")
        
        if not model_loaded:
            print(f"    ❌ {name} 모델 로드 실패 - 스킵")
    
    print(f"\n✅ 총 {len(models)}개 모델 로드 완료")
    return models

def recreate_model_structure(model_name):
    """모델 구조 재생성 (가중치 로드용)"""
    try:
        if model_name == 'lstm':
            return build_lstm_model((100, 59))  # input_shape
        elif model_name == 'gru':
            return build_gru_model((100, 59))
        elif model_name == 'cnn_lstm':
            return build_cnn_lstm((100, 59))
        elif model_name == 'spike':
            return build_spike_detector((100, 59))
        elif model_name == 'rule':
            return build_rule_based_model((100, 59), 4)
        elif model_name == 'ensemble':
            return None  # 앙상블은 복잡해서 스킵
    except:
        return None

# ============================================
# 간단한 모델 구조 정의 (재생성용)
# ============================================
def build_lstm_model(input_shape):
    """LSTM 모델 구조"""
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.2),
        tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.2),
        tf.keras.layers.LSTM(64, dropout=0.2),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ], name='LSTM_Model')
    model.compile(optimizer='adam', loss='mae', metrics=['mae'])
    return model

def build_gru_model(input_shape):
    """GRU 모델 구조"""
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.GRU(256, return_sequences=True, dropout=0.15),
        tf.keras.layers.GRU(128, return_sequences=True, dropout=0.15),
        tf.keras.layers.GRU(64, dropout=0.15),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1)
    ], name='GRU_Model')
    model.compile(optimizer='adam', loss='mae', metrics=['mae'])
    return model

def build_cnn_lstm(input_shape):
    """CNN-LSTM 모델 구조"""
    inputs = tf.keras.Input(shape=input_shape)
    
    convs = []
    for kernel_size in [3, 5, 7, 9]:
        conv = tf.keras.layers.Conv1D(128, kernel_size, activation='relu', padding='same')(inputs)
        convs.append(conv)
    
    concat = tf.keras.layers.Concatenate()(convs)
    lstm1 = tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.15)(concat)
    lstm2 = tf.keras.layers.LSTM(128, dropout=0.15)(lstm1)
    dense1 = tf.keras.layers.Dense(256, activation='relu')(lstm2)
    dense2 = tf.keras.layers.Dense(128, activation='relu')(dense1)
    output = tf.keras.layers.Dense(1)(dense2)
    
    model = tf.keras.Model(inputs=inputs, outputs=output, name='CNN_LSTM_Model')
    model.compile(optimizer='adam', loss='mae', metrics=['mae'])
    return model

def build_spike_detector(input_shape):
    """Spike Detector 구조"""
    inputs = tf.keras.Input(shape=input_shape)
    
    convs = []
    for kernel_size in [3, 5, 7]:
        conv = tf.keras.layers.Conv1D(96, kernel_size, activation='relu', padding='same')(inputs)
        convs.append(conv)
    
    concat = tf.keras.layers.Concatenate()(convs)
    lstm = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.15)
    )(concat)
    
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(lstm)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(lstm)
    pooled = tf.keras.layers.Concatenate()([avg_pool, max_pool])
    
    dense1 = tf.keras.layers.Dense(256, activation='relu')(pooled)
    dense2 = tf.keras.layers.Dense(128, activation='relu')(dense1)
    output = tf.keras.layers.Dense(1, name='spike_value')(dense2)
    
    model = tf.keras.Model(inputs=inputs, outputs=output, name='Spike_Detector')
    model.compile(optimizer='adam', loss='mae', metrics=['mae'])
    return model

def build_rule_based_model(input_shape, m14_shape):
    """Rule-Based 모델 구조"""
    time_input = tf.keras.Input(shape=input_shape, name='time_input')
    m14_input = tf.keras.Input(shape=(m14_shape,), name='m14_input')
    
    lstm1 = tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.15)(time_input)
    lstm2 = tf.keras.layers.LSTM(32, dropout=0.15)(lstm1)
    
    m14_dense = tf.keras.layers.Dense(16, activation='relu')(m14_input)
    
    combined = tf.keras.layers.Concatenate()([lstm2, m14_dense])
    dense1 = tf.keras.layers.Dense(128, activation='relu')(combined)
    dense2 = tf.keras.layers.Dense(64, activation='relu')(dense1)
    prediction = tf.keras.layers.Dense(1)(dense2)
    
    corrected = M14RuleCorrection()([prediction, m14_input])
    
    model = tf.keras.Model(
        inputs=[time_input, m14_input],
        outputs=corrected,
        name='Rule_Based_Model'
    )
    model.compile(optimizer='adam', loss='mae', metrics=['mae'])
    return model

# ============================================
# 데이터 전처리 함수
# ============================================
def prepare_evaluation_data(file_path):
    """평가 데이터 준비"""
    print(f"\n📂 평가 데이터 로드: {file_path}")
    
    # 데이터 로드
    df = pd.read_csv(file_path)
    print(f"  원본 데이터: {len(df)}행")
    
    # 필요한 컬럼 확인
    required_columns = ['M14AM10A', 'M14AM14B', 'M14AM16', 'TOTALCNT']
    
    for col in required_columns:
        if col not in df.columns:
            print(f"  ⚠️ {col} 컬럼 없음 - 0으로 초기화")
            df[col] = 0
    
    # M14AM14BSUM 생성
    if 'M14AM14BSUM' not in df.columns:
        df['M14AM14BSUM'] = df['M14AM14B'] + df['M14AM10A']
    
    # 타겟 생성
    df['target'] = df['TOTALCNT'].shift(-Config.FORECAST)
    
    # 특징 엔지니어링
    print("\n🔧 특징 엔지니어링...")
    
    # 기본 특징
    df['ratio_14B_10A'] = df['M14AM14B'] / (df['M14AM10A'] + 1)
    df['ratio_14B_16'] = df['M14AM14B'] / (df['M14AM16'] + 1)
    df['ratio_10A_16'] = df['M14AM10A'] / (df['M14AM16'] + 1)
    
    # 시계열 특징
    for col in ['TOTALCNT', 'M14AM14B', 'M14AM10A', 'M14AM16']:
        if col in df.columns:
            df[f'{col}_diff_1'] = df[col].diff(1)
            df[f'{col}_diff_5'] = df[col].diff(5)
            df[f'{col}_diff_10'] = df[col].diff(10)
            df[f'{col}_ma_5'] = df[col].rolling(5, min_periods=1).mean()
            df[f'{col}_ma_10'] = df[col].rolling(10, min_periods=1).mean()
            df[f'{col}_ma_20'] = df[col].rolling(20, min_periods=1).mean()
            df[f'{col}_std_5'] = df[col].rolling(5, min_periods=1).std()
            df[f'{col}_std_10'] = df[col].rolling(10, min_periods=1).std()
    
    # 황금 패턴
    df['golden_pattern'] = ((df['M14AM14B'] >= 350) & (df['M14AM10A'] < 70)).astype(float)
    
    # 급증 신호
    for threshold in [250, 300, 350, 400, 450]:
        df[f'signal_{threshold}'] = (df['M14AM14B'] >= threshold).astype(float)
    
    for threshold in [3.5, 4.0, 4.5, 5.0, 5.5]:
        df[f'ratio_signal_{threshold}'] = (df['ratio_14B_10A'] >= threshold).astype(float)
    
    # 결측치 처리
    df = df.fillna(0)
    df = df.dropna(subset=['target'])
    
    print(f"  전처리 완료: {len(df)}행, {len(df.columns)}개 특징")
    
    return df

def create_sequences(df, lookback=100, forecast=10):
    """시퀀스 생성"""
    print("\n⚡ 평가 시퀀스 생성 중...")
    
    X, y = [], []
    data_array = df.values
    totalcnt_idx = df.columns.get_loc('TOTALCNT')
    
    for i in range(len(data_array) - lookback - forecast + 1):
        X.append(data_array[i:i+lookback])
        target_idx = i + lookback + forecast - 1
        if target_idx < len(data_array):
            y.append(data_array[target_idx, totalcnt_idx])
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  y 범위: {y.min():.0f} ~ {y.max():.0f}")
    
    return X, y, df

# ============================================
# 평가 함수
# ============================================
def evaluate_models(models, X_test, y_test, m14_test):
    """모델 평가"""
    print("\n📊 모델 평가 시작...")
    
    results = {}
    predictions = {}
    
    for name, model in models.items():
        print(f"\n  평가 중: {name.upper()}")
        
        try:
            # 예측
            if name in ['ensemble', 'rule']:
                pred = model.predict([X_test, m14_test], batch_size=Config.BATCH_SIZE, verbose=0).flatten()
            else:
                pred = model.predict(X_test, batch_size=Config.BATCH_SIZE, verbose=0).flatten()
            
            predictions[name] = pred
            
            # 성능 지표 계산
            mae = np.mean(np.abs(y_test - pred))
            mse = np.mean((y_test - pred) ** 2)
            rmse = np.sqrt(mse)
            
            non_zero_mask = y_test != 0
            if np.any(non_zero_mask):
                mape = np.mean(np.abs((y_test[non_zero_mask] - pred[non_zero_mask]) / y_test[non_zero_mask])) * 100
            else:
                mape = 0
            
            accuracy_50 = np.mean(np.abs(y_test - pred) <= 50) * 100
            accuracy_100 = np.mean(np.abs(y_test - pred) <= 100) * 100
            
            results[name] = {
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(mape),
                'accuracy_50': float(accuracy_50),
                'accuracy_100': float(accuracy_100),
                'levels': {}
            }
            
            print(f"    MAE: {mae:.2f}")
            print(f"    RMSE: {rmse:.2f}")
            print(f"    MAPE: {mape:.2f}%")
            print(f"    정확도(±50): {accuracy_50:.1f}%")
            print(f"    정확도(±100): {accuracy_100:.1f}%")
            
        except Exception as e:
            print(f"    ❌ 평가 실패: {e}")
    
    return results, predictions

# ============================================
# 메인 실행 함수
# ============================================
def main():
    """메인 평가 프로세스"""
    
    try:
        # 1. 평가 데이터 준비
        df = prepare_evaluation_data(Config.EVAL_DATA_FILE)
        
        # 2. 시퀀스 생성
        X, y, df_processed = create_sequences(df, Config.LOOKBACK, Config.FORECAST)
        
        if len(X) == 0:
            print("\n❌ 시퀀스 생성 실패 - 데이터가 부족합니다.")
            return
        
        # 3. M14 특징 추출
        print("\n📊 M14 특징 추출 중...")
        m14_features = np.zeros((len(X), 4), dtype=np.float32)
        
        for i in range(len(X)):
            idx = i + Config.LOOKBACK
            if idx < len(df_processed):
                m14_features[i] = [
                    df_processed['M14AM14B'].iloc[idx],
                    df_processed['M14AM10A'].iloc[idx],
                    df_processed['M14AM16'].iloc[idx],
                    df_processed['ratio_14B_10A'].iloc[idx]
                ]
        
        # 4. 데이터 스케일링
        print("\n📏 데이터 스케일링...")
        
        X_scaled = np.zeros_like(X)
        for i in range(X.shape[2]):
            scaler = RobustScaler()
            feature = X[:, :, i].reshape(-1, 1)
            X_scaled[:, :, i] = scaler.fit_transform(feature).reshape(X[:, :, i].shape)
        
        m14_scaler = RobustScaler()
        m14_features_scaled = m14_scaler.fit_transform(m14_features)
        
        print("  ✅ 스케일링 완료")
        
        # 5. 모델 로드
        models = load_models()
        
        if not models:
            print("\n❌ 로드된 모델이 없습니다.")
            print("💡 해결 방법:")
            print("  1. 모델 파일 경로 확인: " + Config.MODEL_DIR)
            print("  2. 모델 파일 확장자 확인 (.keras, .h5)")
            print("  3. 학습 코드 먼저 실행하여 모델 생성")
            return
        
        # 6. 모델 평가
        results, predictions = evaluate_models(models, X_scaled, y, m14_features_scaled)
        
        # 7. 결과 출력
        print("\n" + "="*60)
        print("📊 평가 완료!")
        print("="*60)
        
        if results:
            best_model = min(results.keys(), key=lambda x: results[x]['mae'])
            print(f"\n🏆 최고 성능 모델: {best_model.upper()}")
            print(f"  - MAE: {results[best_model]['mae']:.2f}")
            print(f"  - RMSE: {results[best_model]['rmse']:.2f}")
            print(f"  - 정확도(±50): {results[best_model]['accuracy_50']:.1f}%")
            print(f"  - 정확도(±100): {results[best_model]['accuracy_100']:.1f}%")
        
        # 8. 결과 저장
        if results:
            json_path = f"{Config.EVAL_RESULT_DIR}evaluation_results.json"
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n📁 JSON 결과 저장: {json_path}")
        
        print("\n✅ 모든 평가 작업 완료!")
        print("="*60)
        
    except FileNotFoundError:
        print(f"\n❌ 평가 데이터 파일을 찾을 수 없습니다: {Config.EVAL_DATA_FILE}")
    except Exception as e:
        print(f"\n❌ 평가 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()