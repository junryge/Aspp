"""
ensemble_fixed_v6.py - 앙상블 모델 구성 (로딩 오류 수정)
모델 로드 시 인코딩 문제 해결
TensorFlow 2.15.0
"""

import tensorflow as tf
import numpy as np
import json
import os
import gc
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

print("="*60)
print("🎯 앙상블 모델 구성 - 로딩 오류 수정 버전")
print(f"📦 TensorFlow 버전: {tf.__version__}")
print("="*60)

# GPU 메모리 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# ============================================
# 1. 설정
# ============================================
class Config:
    # 데이터 파일
    SEQUENCE_FILE = './sequences_v6.npz'
    
    # 모델 경로
    MODEL_DIR = './models_v6/'
    
    # M14 임계값
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
    
    # 앙상블 학습 설정
    BATCH_SIZE = 16
    ENSEMBLE_EPOCHS = 30
    LEARNING_RATE = 0.0005

# ============================================
# 2. 커스텀 객체 정의 (로드용)
# ============================================
class WeightedLoss(tf.keras.losses.Loss):
    """레벨별 가중 손실 함수"""
    def __init__(self):
        super().__init__()
        
    def call(self, y_true, y_pred):
        # 레벨별 가중치
        weights = tf.where(y_true < 1400, 1.0,
                 tf.where(y_true < 1500, 3.0,
                 tf.where(y_true < 1600, 5.0,
                 tf.where(y_true < 1700, 8.0, 10.0))))
        
        # 가중 MAE
        mae = tf.abs(y_true - y_pred)
        weighted_mae = mae * weights
        
        return tf.reduce_mean(weighted_mae)

class M14RuleCorrection(tf.keras.layers.Layer):
    """M14 규칙 기반 보정 레이어"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def call(self, inputs):
        pred, m14_features = inputs
        
        # M14 특징 분해
        m14b = m14_features[:, 0:1]
        m10a = m14_features[:, 1:2]
        ratio = m14_features[:, 3:4] if m14_features.shape[1] > 3 else tf.ones_like(m14b)
        
        # 규칙 기반 보정
        condition_1700 = tf.logical_and(
            tf.greater_equal(m14b, Config.M14B_THRESHOLDS[1700]),
            tf.greater_equal(ratio, Config.RATIO_THRESHOLDS[1700])
        )
        pred = tf.where(condition_1700, tf.maximum(pred, 1700), pred)
        
        condition_1600 = tf.logical_and(
            tf.greater_equal(m14b, Config.M14B_THRESHOLDS[1600]),
            tf.greater_equal(ratio, Config.RATIO_THRESHOLDS[1600])
        )
        pred = tf.where(condition_1600, tf.maximum(pred, 1600), pred)
        
        condition_1500 = tf.logical_and(
            tf.greater_equal(m14b, Config.M14B_THRESHOLDS[1500]),
            tf.greater_equal(ratio, Config.RATIO_THRESHOLDS[1500])
        )
        pred = tf.where(condition_1500, tf.maximum(pred, 1500), pred)
        
        condition_1400 = tf.greater_equal(m14b, Config.M14B_THRESHOLDS[1400])
        pred = tf.where(condition_1400, tf.maximum(pred, 1400), pred)
        
        condition_inverse = tf.logical_and(
            tf.less(m10a, 70),
            tf.greater_equal(m14b, 250)
        )
        pred = tf.where(condition_inverse, pred * 1.08, pred)
        
        return pred
    
    def get_config(self):
        return super().get_config()

# ============================================
# 3. 안전한 모델 로드 함수
# ============================================
def safe_load_model(model_path, model_name):
    """안전하게 모델을 로드하는 함수"""
    try:
        # 방법 1: 커스텀 객체와 함께 로드
        custom_objects = {
            'WeightedLoss': WeightedLoss,
            'M14RuleCorrection': M14RuleCorrection
        }
        
        model = tf.keras.models.load_model(
            model_path, 
            custom_objects=custom_objects,
            compile=False  # 컴파일 건너뛰기
        )
        print(f"  ✅ {model_name} 모델 로드 완료 (방법 1)")
        return model
        
    except Exception as e1:
        print(f"  ⚠️ {model_name} 로드 방법 1 실패: {str(e1)[:50]}...")
        
        try:
            # 방법 2: weights만 로드
            print(f"  🔄 {model_name} weights 로드 시도...")
            
            # 모델 구조를 알아야 함 - 간단한 추정
            if 'lstm' in model_name.lower():
                model = create_simple_lstm_model()
            elif 'gru' in model_name.lower():
                model = create_simple_gru_model()
            elif 'cnn' in model_name.lower():
                model = create_simple_cnn_model()
            elif 'spike' in model_name.lower():
                model = create_spike_model()
            else:
                raise ValueError(f"Unknown model type: {model_name}")
            
            # weights 파일 경로
            weights_path = model_path.replace('.h5', '_weights.h5')
            if os.path.exists(weights_path):
                model.load_weights(weights_path)
                print(f"  ✅ {model_name} weights 로드 성공")
                return model
            else:
                print(f"  ❌ {model_name} weights 파일도 없음")
                return None
                
        except Exception as e2:
            print(f"  ❌ {model_name} 로드 완전 실패: {str(e2)[:50]}...")
            return None

# ============================================
# 4. 간단한 모델 구조 정의 (weights 로드용)
# ============================================
def create_simple_lstm_model(input_shape=(100, 47)):
    """LSTM 모델 구조"""
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ], name='LSTM_Model')
    return model

def create_simple_gru_model(input_shape=(100, 47)):
    """GRU 모델 구조"""
    model = tf.keras.Sequential([
        tf.keras.layers.GRU(128, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.GRU(128, return_sequences=True),
        tf.keras.layers.GRU(64),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ], name='GRU_Model')
    return model

def create_simple_cnn_model(input_shape=(100, 47)):
    """CNN-LSTM 모델 구조"""
    inputs = tf.keras.Input(shape=input_shape)
    
    conv1 = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
    conv2 = tf.keras.layers.Conv1D(64, 5, activation='relu', padding='same')(inputs)
    conv3 = tf.keras.layers.Conv1D(64, 7, activation='relu', padding='same')(inputs)
    
    concat = tf.keras.layers.Concatenate()([conv1, conv2, conv3])
    norm = tf.keras.layers.BatchNormalization()(concat)
    
    lstm = tf.keras.layers.LSTM(128, return_sequences=True)(norm)
    lstm2 = tf.keras.layers.LSTM(64)(lstm)
    
    dense = tf.keras.layers.Dense(128, activation='relu')(lstm2)
    dropout = tf.keras.layers.Dropout(0.3)(dense)
    output = tf.keras.layers.Dense(1)(dropout)
    
    model = tf.keras.Model(inputs=inputs, outputs=output, name='CNN_LSTM_Model')
    return model

def create_spike_model(input_shape=(100, 47)):
    """Spike Detector 모델 구조"""
    inputs = tf.keras.Input(shape=input_shape)
    
    conv1 = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
    conv2 = tf.keras.layers.Conv1D(64, 5, activation='relu', padding='same')(inputs)
    conv3 = tf.keras.layers.Conv1D(64, 7, activation='relu', padding='same')(inputs)
    
    concat = tf.keras.layers.Concatenate()([conv1, conv2, conv3])
    norm = tf.keras.layers.BatchNormalization()(concat)
    
    # Attention 대신 간단한 처리
    lstm = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True)
    )(norm)
    
    pooled = tf.keras.layers.GlobalAveragePooling1D()(lstm)
    
    dense1 = tf.keras.layers.Dense(256, activation='relu')(pooled)
    dropout1 = tf.keras.layers.Dropout(0.3)(dense1)
    dense2 = tf.keras.layers.Dense(128, activation='relu')(dropout1)
    dropout2 = tf.keras.layers.Dropout(0.2)(dense2)
    
    regression_output = tf.keras.layers.Dense(1, name='spike_value')(dropout2)
    classification_output = tf.keras.layers.Dense(1, activation='sigmoid', name='spike_prob')(dropout2)
    
    model = tf.keras.Model(
        inputs=inputs,
        outputs=[regression_output, classification_output],
        name='Spike_Detector'
    )
    return model

# ============================================
# 5. 데이터 로드
# ============================================
print("\n📂 데이터 로딩 중...")

# 전체 데이터가 아닌 검증용 일부만 로드
data = np.load(Config.SEQUENCE_FILE)
X_val = data['X'][-10000:].astype(np.float32)
y_val = data['y'][-10000:].astype(np.float32)
m14_val = data['m14_features'][-10000:].astype(np.float32)

print(f"  검증 데이터 shape: {X_val.shape}")
print(f"  1400+ 비율: {(y_val >= 1400).mean():.1%}")

# ============================================
# 6. 모델 로드 (수정된 방식)
# ============================================
print("\n📥 학습된 모델 로드 중...")

models = {}

# 모델 리스트
model_files = {
    'lstm': 'lstm_model.h5',
    'gru': 'gru_model.h5',
    'cnn_lstm': 'cnn_lstm_model.h5',
    'spike': 'spike_model.h5'
}

# 각 모델 로드 시도
for model_name, file_name in model_files.items():
    model_path = os.path.join(Config.MODEL_DIR, file_name)
    
    if os.path.exists(model_path):
        model = safe_load_model(model_path, model_name.upper())
        if model is not None:
            models[model_name] = model
    else:
        print(f"  ⚠️ {model_name.upper()} 모델 파일이 없습니다: {model_path}")

print(f"\n💡 로드된 모델 수: {len(models)}개")
print(f"  로드된 모델: {list(models.keys())}")

if len(models) == 0:
    print("\n❌ 로드된 모델이 없습니다.")
    print("💡 해결 방법:")
    print("  1. 모델 파일 경로 확인: ./models_v6/")
    print("  2. 모델 파일명 확인: lstm_model.h5, gru_model.h5 등")
    print("  3. 개별 모델을 먼저 학습해주세요.")
    exit(1)

# ============================================
# 7. 앙상블 모델 구성
# ============================================
print("\n🔧 앙상블 모델 구성 중...")

# 입력 정의
input_shape = X_val.shape[1:]
time_series_input = tf.keras.Input(shape=input_shape, name='ensemble_input')
m14_input = tf.keras.Input(shape=(4,), name='m14_features')

# 각 모델의 예측값 수집
predictions = []
model_names = []

# 각 모델 예측값 수집
for name, model in models.items():
    try:
        if name == 'spike':
            outputs = model(time_series_input)
            if isinstance(outputs, list):
                pred = outputs[0]
                spike_prob = outputs[1]
            else:
                pred = outputs
                spike_prob = None
        else:
            pred = model(time_series_input)
            spike_prob = None
            
        predictions.append(pred)
        model_names.append(name)
        print(f"  ✅ {name} 예측 레이어 추가")
        
    except Exception as e:
        print(f"  ❌ {name} 예측 실패: {str(e)[:50]}...")

print(f"\n  앙상블에 포함된 모델: {model_names}")

# 앙상블 구성 계속...
if len(predictions) == 0:
    print("❌ 앙상블에 사용할 예측값이 없습니다.")
    exit(1)

# 가중 평균 계산
if len(predictions) > 1:
    # 동일 가중치로 시작
    ensemble_pred = tf.keras.layers.Average()(predictions)
    print("  📊 평균 앙상블 사용")
else:
    ensemble_pred = predictions[0]
    print("  📊 단일 모델 사용")

# M14 규칙 보정
final_pred = M14RuleCorrection()([ensemble_pred, m14_input])

# 앙상블 모델 생성
ensemble_model = tf.keras.Model(
    inputs=[time_series_input, m14_input],
    outputs=final_pred,
    name='Ensemble_Model'
)

print("  ✅ 앙상블 모델 구성 완료")

# 모델 구조 출력
print("\n📋 앙상블 모델 구조:")
print(f"  입력: 시계열 {input_shape} + M14 특징 (4,)")
print(f"  포함 모델: {len(model_names)}개")
print(f"  출력: 물류량 예측값")

# ============================================
# 8. 간단한 평가 (컴파일 없이)
# ============================================
print("\n📊 빠른 평가...")

# 100개 샘플만 테스트
test_size = min(100, len(X_val))
X_test = X_val[:test_size]
y_test = y_val[:test_size]
m14_test = m14_val[:test_size]

# 예측
try:
    y_pred = ensemble_model.predict([X_test, m14_test], verbose=0)
    y_pred = y_pred.flatten()
    
    # MAE 계산
    mae = np.mean(np.abs(y_test - y_pred))
    print(f"  테스트 MAE: {mae:.2f}")
    
    # 1400+ Recall
    mask_1400 = y_test >= 1400
    if np.any(mask_1400):
        recall_1400 = np.sum((y_pred >= 1400) & mask_1400) / np.sum(mask_1400)
        print(f"  1400+ Recall: {recall_1400:.2%}")
        
except Exception as e:
    print(f"  ⚠️ 평가 실패: {str(e)[:100]}...")

# ============================================
# 9. 모델 저장
# ============================================
print("\n💾 앙상블 모델 저장 중...")

# 저장 시도
ensemble_path = os.path.join(Config.MODEL_DIR, 'ensemble_model_fixed.h5')

try:
    # 모델 구조와 가중치 저장
    ensemble_model.save(ensemble_path, save_traces=False)
    print(f"  ✅ 앙상블 모델 저장 완료: {ensemble_path}")
except Exception as e:
    print(f"  ❌ 전체 저장 실패: {str(e)[:50]}...")
    
    # weights만 저장
    try:
        weights_path = os.path.join(Config.MODEL_DIR, 'ensemble_weights_fixed.h5')
        ensemble_model.save_weights(weights_path)
        print(f"  ✅ 가중치만 저장 완료: {weights_path}")
    except Exception as e2:
        print(f"  ❌ 가중치 저장도 실패: {str(e2)[:50]}...")

# ============================================
# 10. 요약
# ============================================
print("\n" + "="*60)
print("🎉 앙상블 모델 구성 완료!")
print("="*60)
print(f"📁 저장 위치: {Config.MODEL_DIR}")
print(f"🔧 포함된 모델: {', '.join(model_names)}")
print(f"💾 저장 파일: ensemble_model_fixed.h5")
print("\n💡 다음 단계:")
print("  1. 개별 모델들이 제대로 학습되었는지 확인")
print("  2. 모델 파일이 올바른 위치에 있는지 확인")
print("  3. 필요시 weights 파일도 함께 저장")
print("="*60)