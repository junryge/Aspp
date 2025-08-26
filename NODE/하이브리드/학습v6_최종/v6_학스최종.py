"""
"""
V6_학습_최종본_개선.py - 5개 모델 앙상블 학습 (LSTM 개선 버전)
미리 생성된 시퀀스를 로드하여 LSTM, GRU, CNN-LSTM, Spike Detector, Rule-Based 학습
TensorFlow 2.15.0
"""

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json
import os
import warnings
import pickle
from datetime import datetime
warnings.filterwarnings('ignore')

print("="*60)
print("🚀 반도체 물류 예측 앙상블 학습 V6 (개선판)")
print(f"📦 TensorFlow 버전: {tf.__version__}")
print("="*60)

# ============================================
# 1. 설정
# ============================================
class Config:
   # 시퀀스 파일
   SEQUENCE_FILE = './sequences_v6.npz'
   
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
   
   # 학습 설정 (개선)
   BATCH_SIZE = 64  # 32 → 64
   EPOCHS = 150  # 100 → 150
   LEARNING_RATE = 0.0005  # 0.001 → 0.0005
   PATIENCE = 20  # 15 → 20
   
   # 모델 저장 경로
   MODEL_DIR = './models_v6/'
   CHECKPOINT_DIR = './checkpoints_v6/'
   
   # 가중치 설정
   SPIKE_WEIGHTS = {
       'normal': 1.0,
       'level_1400': 3.0,
       'level_1500': 5.0,
       'level_1600': 8.0,
       'level_1700': 10.0
   }

# 디렉토리 생성
os.makedirs(Config.MODEL_DIR, exist_ok=True)
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

# ============================================
# 체크포인트 관리자
# ============================================
class CheckpointManager:
   def __init__(self):
       self.checkpoint_file = os.path.join(Config.CHECKPOINT_DIR, 'training_state.pkl')
   
   def save_state(self, completed_models, models, history, evaluation_results):
       """학습 상태 저장"""
       state = {
           'completed_models': completed_models,
           'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           'evaluation_results': evaluation_results
       }
       
       # 모델 저장
       for name, model in models.items():
           model.save(os.path.join(Config.CHECKPOINT_DIR, f'{name}_checkpoint.h5'))
       
       # 히스토리 저장
       with open(os.path.join(Config.CHECKPOINT_DIR, 'history.pkl'), 'wb') as f:
           pickle.dump(history, f)
       
       with open(self.checkpoint_file, 'wb') as f:
           pickle.dump(state, f)
       
       print(f"\n💾 체크포인트 저장 완료: {completed_models}")
   
   def load_state(self):
       """저장된 상태 로드"""
       if not os.path.exists(self.checkpoint_file):
           return None, {}, {}, {}
       
       with open(self.checkpoint_file, 'rb') as f:
           state = pickle.load(f)
       
       # 히스토리 로드
       history = {}
       if os.path.exists(os.path.join(Config.CHECKPOINT_DIR, 'history.pkl')):
           with open(os.path.join(Config.CHECKPOINT_DIR, 'history.pkl'), 'rb') as f:
               history = pickle.load(f)
       
       print(f"\n🔄 체크포인트 로드: {state['completed_models']}")
       print(f"   저장 시간: {state['timestamp']}")
       
       return state['completed_models'], {}, history, state.get('evaluation_results', {})

# ============================================
# 2. 커스텀 레이어 및 손실 함수
# ============================================
class WeightedLoss(tf.keras.losses.Loss):
   """물류량 구간별 가중치 손실 함수"""
   def call(self, y_true, y_pred):
       mae = tf.abs(y_true - y_pred)
       
       # 구간별 가중치
       weights = tf.ones_like(y_true)
       weights = tf.where(y_true >= 1700, 10.0, weights)
       weights = tf.where((y_true >= 1600) & (y_true < 1700), 8.0, weights)
       weights = tf.where((y_true >= 1500) & (y_true < 1600), 5.0, weights)
       weights = tf.where((y_true >= 1400) & (y_true < 1500), 3.0, weights)
       
       return tf.reduce_mean(mae * weights)

class M14RuleCorrection(tf.keras.layers.Layer):
   """M14 규칙 기반 보정 레이어"""
   def __init__(self, **kwargs):
       super().__init__(**kwargs)
       
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
       
       # 황금 패턴 보정
       golden_pattern = tf.logical_and(
           tf.greater_equal(m14b, 350),
           tf.less(m10a, 70)
       )
       pred = tf.where(golden_pattern, pred * 1.1, pred)
       
       return pred

class SpikePerformanceCallback(tf.keras.callbacks.Callback):
   """급증 감지 성능 모니터링"""
   def __init__(self, X_val, y_val):
       self.X_val = X_val
       self.y_val = y_val
       
   def on_epoch_end(self, epoch, logs=None):
       if epoch % 10 == 0:
           pred = self.model.predict(self.X_val, verbose=0)
           if isinstance(pred, list):
               pred = pred[0]
           pred = pred.flatten()
           
           # 구간별 Recall
           for level in [1400, 1500, 1600, 1700]:
               mask = self.y_val >= level
               if np.any(mask):
                   recall = np.sum((pred >= level) & mask) / np.sum(mask)
                   print(f"   {level} Recall: {recall:.2%}", end=" ")
           print()

# ============================================
# 3. 모델 정의 (개선)
# ============================================
class ModelsV6:
   
   @staticmethod
   def build_lstm_model(input_shape):
       """1. LSTM 모델 - 개선된 정규화"""
       inputs = tf.keras.Input(shape=input_shape, name='lstm_input')
       
       # Layer Normalization 추가
       x = tf.keras.layers.LayerNormalization()(inputs)
       
       # Stacked LSTM with stronger regularization
       lstm1 = tf.keras.layers.LSTM(
           128, 
           return_sequences=True, 
           dropout=0.3,
           recurrent_dropout=0.3,
           kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)
       )(x)
       
       lstm2 = tf.keras.layers.LSTM(
           128, 
           return_sequences=True, 
           dropout=0.3,
           recurrent_dropout=0.3,
           kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)
       )(lstm1)
       
       lstm3 = tf.keras.layers.LSTM(
           64, 
           dropout=0.3,
           recurrent_dropout=0.3
       )(lstm2)
       
       # BatchNormalization 추가
       norm = tf.keras.layers.BatchNormalization()(lstm3)
       
       # Dense layers with more dropout
       dense1 = tf.keras.layers.Dense(128, activation='relu')(norm)
       dropout1 = tf.keras.layers.Dropout(0.4)(dense1)
       dense2 = tf.keras.layers.Dense(64, activation='relu')(dropout1)
       dropout2 = tf.keras.layers.Dropout(0.3)(dense2)
       
       # Output
       output = tf.keras.layers.Dense(1, name='lstm_output')(dropout2)
       
       model = tf.keras.Model(inputs=inputs, outputs=output, name='LSTM_Model')
       return model
   
   @staticmethod
   def build_enhanced_gru(input_shape):
       """2. GRU 모델 - 단기 변동성 포착"""
       inputs = tf.keras.Input(shape=input_shape, name='gru_input')
       
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
       output = tf.keras.layers.Dense(1, name='gru_output')(dense2)
       
       model = tf.keras.Model(inputs=inputs, outputs=output, name='GRU_Model')
       return model
   
   @staticmethod
   def build_cnn_lstm(input_shape):
       """3. CNN-LSTM 모델 - 복합 패턴 인식"""
       inputs = tf.keras.Input(shape=input_shape, name='cnn_input')
       
       # Multi-scale CNN
       conv1 = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
       conv2 = tf.keras.layers.Conv1D(64, 5, activation='relu', padding='same')(inputs)
       conv3 = tf.keras.layers.Conv1D(64, 7, activation='relu', padding='same')(inputs)
       
       # Concatenate
       concat = tf.keras.layers.Concatenate()([conv1, conv2, conv3])
       
       # Batch normalization
       norm = tf.keras.layers.BatchNormalization()(concat)
       
       # LSTM
       lstm = tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.2)(norm)
       lstm2 = tf.keras.layers.LSTM(64, dropout=0.2)(lstm)
       
       # Dense layers
       dense1 = tf.keras.layers.Dense(128, activation='relu')(lstm2)
       dropout = tf.keras.layers.Dropout(0.3)(dense1)
       
       # Output
       output = tf.keras.layers.Dense(1, name='cnn_lstm_output')(dropout)
       
       model = tf.keras.Model(inputs=inputs, outputs=output, name='CNN_LSTM_Model')
       return model
   
   @staticmethod
   def build_spike_detector(input_shape):
       """4. Spike Detector - 이상치 감지 전문"""
       inputs = tf.keras.Input(shape=input_shape, name='spike_input')
       
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
       
       # Dual output (회귀 + 분류)
       regression_output = tf.keras.layers.Dense(1, name='spike_value')(dropout2)
       classification_output = tf.keras.layers.Dense(1, activation='sigmoid', name='spike_prob')(dropout2)
       
       model = tf.keras.Model(
           inputs=inputs,
           outputs=[regression_output, classification_output],
           name='Spike_Detector'
       )
       return model
   
   @staticmethod
   def build_rule_based_model(input_shape, m14_shape):
       """5. Rule-Based 모델 - 검증된 황금 패턴"""
       # 시계열 입력
       time_input = tf.keras.Input(shape=input_shape, name='time_input')
       # M14 특징 입력
       m14_input = tf.keras.Input(shape=m14_shape, name='m14_input')
       
       # 간단한 시계열 처리
       lstm = tf.keras.layers.LSTM(32, dropout=0.2)(time_input)
       
       # M14 특징 처리
       m14_dense = tf.keras.layers.Dense(16, activation='relu')(m14_input)
       
       # 결합
       combined = tf.keras.layers.Concatenate()([lstm, m14_dense])
       
       # Dense layers
       dense1 = tf.keras.layers.Dense(64, activation='relu')(combined)
       dropout = tf.keras.layers.Dropout(0.2)(dense1)
       dense2 = tf.keras.layers.Dense(32, activation='relu')(dropout)
       
       # 예측
       prediction = tf.keras.layers.Dense(1, name='rule_pred')(dense2)
       
       # M14 규칙 적용
       corrected = M14RuleCorrection()([prediction, m14_input])
       
       model = tf.keras.Model(
           inputs=[time_input, m14_input],
           outputs=corrected,
           name='Rule_Based_Model'
       )
       return model

# ============================================
# 4. 데이터 로드 및 준비
# ============================================
print("\n📂 시퀀스 로딩 중...")

# 시퀀스 로드
data = np.load(Config.SEQUENCE_FILE)
X = data['X']
y = data['y']
m14_features = data['m14_features']

print(f"  ✅ 로드 완료!")
print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")
print(f"  m14_features shape: {m14_features.shape}")

# 학습/검증 분할
X_train, X_val, y_train, y_val, m14_train, m14_val = train_test_split(
   X, y, m14_features, test_size=0.2, random_state=42
)

# 1400+ 여부 레이블 생성
y_spike_class = (y_train >= 1400).astype(float)
y_val_spike_class = (y_val >= 1400).astype(float)

print(f"\n📊 데이터 분할:")
print(f"  학습: {X_train.shape[0]:,}개")
print(f"  검증: {X_val.shape[0]:,}개")
print(f"  1400+ 학습 비율: {y_spike_class.mean():.1%}")
print(f"  1400+ 검증 비율: {y_val_spike_class.mean():.1%}")

# ============================================
# 5. 학습 파이프라인 (체크포인트 지원)
# ============================================
print("\n" + "="*60)
print("🏋️ 5개 모델 학습 시작")
print("="*60)

# 체크포인트 매니저 초기화
checkpoint_manager = CheckpointManager()

# 이전 상태 로드
completed_models, models, history, evaluation_results = checkpoint_manager.load_state()

# 완료되지 않은 경우 초기화
if not completed_models:
   completed_models = []
   models = {}
   history = {}
   evaluation_results = {}

# 모델 리스트
model_list = ['lstm', 'gru', 'cnn_lstm', 'spike', 'rule']

# ============================================
# 5.1 LSTM 모델 (개선)
# ============================================
if 'lstm' not in completed_models:
   print("\n1️⃣ LSTM 모델 학습 (장기 시계열 패턴) - 개선된 버전")
   
   # 체크포인트에서 모델 로드 시도
   checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, 'lstm_checkpoint.h5')
   if os.path.exists(checkpoint_path):
       print("  체크포인트에서 모델 로드 중...")
       lstm_model = tf.keras.models.load_model(checkpoint_path, custom_objects={'WeightedLoss': WeightedLoss})
   else:
       lstm_model = ModelsV6.build_lstm_model(X_train.shape[1:])
       lstm_model.compile(
           optimizer=tf.keras.optimizers.Adam(
               learning_rate=Config.LEARNING_RATE,
               clipnorm=1.0  # gradient clipping 추가
           ),
           loss=WeightedLoss(),
           metrics=['mae']
       )
   
   lstm_history = lstm_model.fit(
       X_train, y_train,
       validation_data=(X_val, y_val),
       epochs=Config.EPOCHS,
       batch_size=Config.BATCH_SIZE,
       callbacks=[
           tf.keras.callbacks.ModelCheckpoint(
               f"{Config.MODEL_DIR}lstm_best.h5",
               save_best_only=True,
               monitor='val_loss',
               verbose=0
           ),
           tf.keras.callbacks.EarlyStopping(
               patience=Config.PATIENCE, 
               restore_best_weights=True,
               min_delta=0.001
           ),
           tf.keras.callbacks.ReduceLROnPlateau(
               patience=7, 
               factor=0.5,
               min_lr=1e-7
           ),
           SpikePerformanceCallback(X_val, y_val)
       ],
       verbose=1
   )
   
   models['lstm'] = lstm_model
   history['lstm'] = lstm_history
   completed_models.append('lstm')
   
   # 체크포인트 저장
   checkpoint_manager.save_state(completed_models, models, history, evaluation_results)
else:
   print("\n1️⃣ LSTM 모델 - 이미 완료 ✓")
   if os.path.exists(f"{Config.MODEL_DIR}lstm_best.h5"):
       models['lstm'] = tf.keras.models.load_model(f"{Config.MODEL_DIR}lstm_best.h5", 
                                                   custom_objects={'WeightedLoss': WeightedLoss})

# ============================================
# 5.2 GRU 모델
# ============================================
if 'gru' not in completed_models:
   print("\n2️⃣ Enhanced GRU 모델 학습 (단기 변동성)")
   
   checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, 'gru_checkpoint.h5')
   if os.path.exists(checkpoint_path):
       print("  체크포인트에서 모델 로드 중...")
       gru_model = tf.keras.models.load_model(checkpoint_path, custom_objects={'WeightedLoss': WeightedLoss})
   else:
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
           tf.keras.callbacks.ModelCheckpoint(
               f"{Config.MODEL_DIR}gru_best.h5",
               save_best_only=True,
               monitor='val_loss',
               verbose=0
           ),
           tf.keras.callbacks.EarlyStopping(patience=Config.PATIENCE, restore_best_weights=True),
           tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
           SpikePerformanceCallback(X_val, y_val)
       ],
       verbose=1
   )
   
   models['gru'] = gru_model
   history['gru'] = gru_history
   completed_models.append('gru')
   
   # 체크포인트 저장
   checkpoint_manager.save_state(completed_models, models, history, evaluation_results)
else:
   print("\n2️⃣ GRU 모델 - 이미 완료 ✓")
   if os.path.exists(f"{Config.MODEL_DIR}gru_best.h5"):
       models['gru'] = tf.keras.models.load_model(f"{Config.MODEL_DIR}gru_best.h5",
                                                  custom_objects={'WeightedLoss': WeightedLoss})

# [이하 CNN-LSTM, Spike Detector, Rule-Based, 앙상블, 평가, 시각화 코드는 동일]
# ... (나머지 코드는 기존과 동일)

print("\n✅ 개선된 V6 모델 학습 완료!")

V6_학습_최종본.py - 5개 모델 앙상블 학습 (체크포인트 지원)
미리 생성된 시퀀스를 로드하여 LSTM, GRU, CNN-LSTM, Spike Detector, Rule-Based 학습
TensorFlow 2.15.0
"""

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json
import os
import warnings
import pickle
from datetime import datetime
warnings.filterwarnings('ignore')

print("="*60)
print("🚀 반도체 물류 예측 앙상블 학습 V6")
print(f"📦 TensorFlow 버전: {tf.__version__}")
print("="*60)

# ============================================
# 1. 설정
# ============================================
class Config:
    # 시퀀스 파일
    SEQUENCE_FILE = './sequences_v6.npz'
    
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
    
    # 학습 설정
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    PATIENCE = 15
    
    # 모델 저장 경로
    MODEL_DIR = './models_v6/'
    CHECKPOINT_DIR = './checkpoints_v6/'
    
    # 가중치 설정
    SPIKE_WEIGHTS = {
        'normal': 1.0,
        'level_1400': 3.0,
        'level_1500': 5.0,
        'level_1600': 8.0,
        'level_1700': 10.0
    }

# 디렉토리 생성
os.makedirs(Config.MODEL_DIR, exist_ok=True)
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

# ============================================
# 체크포인트 관리자
# ============================================
class CheckpointManager:
    def __init__(self):
        self.checkpoint_file = os.path.join(Config.CHECKPOINT_DIR, 'training_state.pkl')
    
    def save_state(self, completed_models, models, history, evaluation_results):
        """학습 상태 저장"""
        state = {
            'completed_models': completed_models,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'evaluation_results': evaluation_results
        }
        
        # 모델 저장
        for name, model in models.items():
            model.save(os.path.join(Config.CHECKPOINT_DIR, f'{name}_checkpoint.h5'))
        
        # 히스토리 저장
        with open(os.path.join(Config.CHECKPOINT_DIR, 'history.pkl'), 'wb') as f:
            pickle.dump(history, f)
        
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"\n💾 체크포인트 저장 완료: {completed_models}")
    
    def load_state(self):
        """저장된 상태 로드"""
        if not os.path.exists(self.checkpoint_file):
            return None, {}, {}, {}
        
        with open(self.checkpoint_file, 'rb') as f:
            state = pickle.load(f)
        
        # 히스토리 로드
        history = {}
        if os.path.exists(os.path.join(Config.CHECKPOINT_DIR, 'history.pkl')):
            with open(os.path.join(Config.CHECKPOINT_DIR, 'history.pkl'), 'rb') as f:
                history = pickle.load(f)
        
        print(f"\n🔄 체크포인트 로드: {state['completed_models']}")
        print(f"   저장 시간: {state['timestamp']}")
        
        return state['completed_models'], {}, history, state.get('evaluation_results', {})

# ============================================
# 2. 커스텀 레이어 및 손실 함수
# ============================================
class WeightedLoss(tf.keras.losses.Loss):
    """물류량 구간별 가중치 손실 함수"""
    def call(self, y_true, y_pred):
        mae = tf.abs(y_true - y_pred)
        
        # 구간별 가중치
        weights = tf.ones_like(y_true)
        weights = tf.where(y_true >= 1700, 10.0, weights)
        weights = tf.where((y_true >= 1600) & (y_true < 1700), 8.0, weights)
        weights = tf.where((y_true >= 1500) & (y_true < 1600), 5.0, weights)
        weights = tf.where((y_true >= 1400) & (y_true < 1500), 3.0, weights)
        
        return tf.reduce_mean(mae * weights)

class M14RuleCorrection(tf.keras.layers.Layer):
    """M14 규칙 기반 보정 레이어"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
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
        
        # 황금 패턴 보정
        golden_pattern = tf.logical_and(
            tf.greater_equal(m14b, 350),
            tf.less(m10a, 70)
        )
        pred = tf.where(golden_pattern, pred * 1.1, pred)
        
        return pred

class SpikePerformanceCallback(tf.keras.callbacks.Callback):
    """급증 감지 성능 모니터링"""
    def __init__(self, X_val, y_val):
        self.X_val = X_val
        self.y_val = y_val
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            pred = self.model.predict(self.X_val, verbose=0)
            if isinstance(pred, list):
                pred = pred[0]
            pred = pred.flatten()
            
            # 구간별 Recall
            for level in [1400, 1500, 1600, 1700]:
                mask = self.y_val >= level
                if np.any(mask):
                    recall = np.sum((pred >= level) & mask) / np.sum(mask)
                    print(f"   {level} Recall: {recall:.2%}", end=" ")
            print()

# ============================================
# 3. 모델 정의
# ============================================
class ModelsV6:
    
    @staticmethod
    def build_lstm_model(input_shape):
        """1. LSTM 모델 - 장기 시계열 패턴 학습"""
        inputs = tf.keras.Input(shape=input_shape, name='lstm_input')
        
        # Stacked LSTM
        lstm1 = tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.2)(inputs)
        lstm2 = tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.2)(lstm1)
        lstm3 = tf.keras.layers.LSTM(64, dropout=0.2)(lstm2)
        
        # Dense layers
        dense1 = tf.keras.layers.Dense(128, activation='relu')(lstm3)
        dropout = tf.keras.layers.Dropout(0.3)(dense1)
        dense2 = tf.keras.layers.Dense(64, activation='relu')(dropout)
        
        # Output
        output = tf.keras.layers.Dense(1, name='lstm_output')(dense2)
        
        model = tf.keras.Model(inputs=inputs, outputs=output, name='LSTM_Model')
        return model
    
    @staticmethod
    def build_enhanced_gru(input_shape):
        """2. GRU 모델 - 단기 변동성 포착"""
        inputs = tf.keras.Input(shape=input_shape, name='gru_input')
        
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
        output = tf.keras.layers.Dense(1, name='gru_output')(dense2)
        
        model = tf.keras.Model(inputs=inputs, outputs=output, name='GRU_Model')
        return model
    
    @staticmethod
    def build_cnn_lstm(input_shape):
        """3. CNN-LSTM 모델 - 복합 패턴 인식"""
        inputs = tf.keras.Input(shape=input_shape, name='cnn_input')
        
        # Multi-scale CNN
        conv1 = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
        conv2 = tf.keras.layers.Conv1D(64, 5, activation='relu', padding='same')(inputs)
        conv3 = tf.keras.layers.Conv1D(64, 7, activation='relu', padding='same')(inputs)
        
        # Concatenate
        concat = tf.keras.layers.Concatenate()([conv1, conv2, conv3])
        
        # Batch normalization
        norm = tf.keras.layers.BatchNormalization()(concat)
        
        # LSTM
        lstm = tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.2)(norm)
        lstm2 = tf.keras.layers.LSTM(64, dropout=0.2)(lstm)
        
        # Dense layers
        dense1 = tf.keras.layers.Dense(128, activation='relu')(lstm2)
        dropout = tf.keras.layers.Dropout(0.3)(dense1)
        
        # Output
        output = tf.keras.layers.Dense(1, name='cnn_lstm_output')(dropout)
        
        model = tf.keras.Model(inputs=inputs, outputs=output, name='CNN_LSTM_Model')
        return model
    
    @staticmethod
    def build_spike_detector(input_shape):
        """4. Spike Detector - 이상치 감지 전문"""
        inputs = tf.keras.Input(shape=input_shape, name='spike_input')
        
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
        
        # Dual output (회귀 + 분류)
        regression_output = tf.keras.layers.Dense(1, name='spike_value')(dropout2)
        classification_output = tf.keras.layers.Dense(1, activation='sigmoid', name='spike_prob')(dropout2)
        
        model = tf.keras.Model(
            inputs=inputs,
            outputs=[regression_output, classification_output],
            name='Spike_Detector'
        )
        return model
    
    @staticmethod
    def build_rule_based_model(input_shape, m14_shape):
        """5. Rule-Based 모델 - 검증된 황금 패턴"""
        # 시계열 입력
        time_input = tf.keras.Input(shape=input_shape, name='time_input')
        # M14 특징 입력
        m14_input = tf.keras.Input(shape=m14_shape, name='m14_input')
        
        # 간단한 시계열 처리
        lstm = tf.keras.layers.LSTM(32, dropout=0.2)(time_input)
        
        # M14 특징 처리
        m14_dense = tf.keras.layers.Dense(16, activation='relu')(m14_input)
        
        # 결합
        combined = tf.keras.layers.Concatenate()([lstm, m14_dense])
        
        # Dense layers
        dense1 = tf.keras.layers.Dense(64, activation='relu')(combined)
        dropout = tf.keras.layers.Dropout(0.2)(dense1)
        dense2 = tf.keras.layers.Dense(32, activation='relu')(dropout)
        
        # 예측
        prediction = tf.keras.layers.Dense(1, name='rule_pred')(dense2)
        
        # M14 규칙 적용
        corrected = M14RuleCorrection()([prediction, m14_input])
        
        model = tf.keras.Model(
            inputs=[time_input, m14_input],
            outputs=corrected,
            name='Rule_Based_Model'
        )
        return model

# ============================================
# 4. 데이터 로드 및 준비
# ============================================
print("\n📂 시퀀스 로딩 중...")

# 시퀀스 로드
data = np.load(Config.SEQUENCE_FILE)
X = data['X']
y = data['y']
m14_features = data['m14_features']

print(f"  ✅ 로드 완료!")
print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")
print(f"  m14_features shape: {m14_features.shape}")

# 학습/검증 분할
X_train, X_val, y_train, y_val, m14_train, m14_val = train_test_split(
    X, y, m14_features, test_size=0.2, random_state=42
)

# 1400+ 여부 레이블 생성
y_spike_class = (y_train >= 1400).astype(float)
y_val_spike_class = (y_val >= 1400).astype(float)

print(f"\n📊 데이터 분할:")
print(f"  학습: {X_train.shape[0]:,}개")
print(f"  검증: {X_val.shape[0]:,}개")
print(f"  1400+ 학습 비율: {y_spike_class.mean():.1%}")
print(f"  1400+ 검증 비율: {y_val_spike_class.mean():.1%}")

# ============================================
# 5. 학습 파이프라인 (체크포인트 지원)
# ============================================
print("\n" + "="*60)
print("🏋️ 5개 모델 학습 시작")
print("="*60)

# 체크포인트 매니저 초기화
checkpoint_manager = CheckpointManager()

# 이전 상태 로드
completed_models, models, history, evaluation_results = checkpoint_manager.load_state()

# 완료되지 않은 경우 초기화
if not completed_models:
    completed_models = []
    models = {}
    history = {}
    evaluation_results = {}

# 모델 리스트
model_list = ['lstm', 'gru', 'cnn_lstm', 'spike', 'rule']

# ============================================
# 5.1 LSTM 모델
# ============================================
if 'lstm' not in completed_models:
    print("\n1️⃣ LSTM 모델 학습 (장기 시계열 패턴)")
    
    # 체크포인트에서 모델 로드 시도
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, 'lstm_checkpoint.h5')
    if os.path.exists(checkpoint_path):
        print("  체크포인트에서 모델 로드 중...")
        lstm_model = tf.keras.models.load_model(checkpoint_path, custom_objects={'WeightedLoss': WeightedLoss})
    else:
        lstm_model = ModelsV6.build_lstm_model(X_train.shape[1:])
        lstm_model.compile(
            optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE),
            loss=WeightedLoss(),
            metrics=['mae']
        )
    
    lstm_history = lstm_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                f"{Config.MODEL_DIR}lstm_best.h5",
                save_best_only=True,
                monitor='val_loss',
                verbose=0
            ),
            tf.keras.callbacks.EarlyStopping(patience=Config.PATIENCE, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
            SpikePerformanceCallback(X_val, y_val)
        ],
        verbose=1
    )
    
    models['lstm'] = lstm_model
    history['lstm'] = lstm_history
    completed_models.append('lstm')
    
    # 체크포인트 저장
    checkpoint_manager.save_state(completed_models, models, history, evaluation_results)
else:
    print("\n1️⃣ LSTM 모델 - 이미 완료 ✓")
    if os.path.exists(f"{Config.MODEL_DIR}lstm_best.h5"):
        models['lstm'] = tf.keras.models.load_model(f"{Config.MODEL_DIR}lstm_best.h5", 
                                                    custom_objects={'WeightedLoss': WeightedLoss})

# ============================================
# 5.2 GRU 모델
# ============================================
if 'gru' not in completed_models:
    print("\n2️⃣ Enhanced GRU 모델 학습 (단기 변동성)")
    
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, 'gru_checkpoint.h5')
    if os.path.exists(checkpoint_path):
        print("  체크포인트에서 모델 로드 중...")
        gru_model = tf.keras.models.load_model(checkpoint_path, custom_objects={'WeightedLoss': WeightedLoss})
    else:
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
            tf.keras.callbacks.ModelCheckpoint(
                f"{Config.MODEL_DIR}gru_best.h5",
                save_best_only=True,
                monitor='val_loss',
                verbose=0
            ),
            tf.keras.callbacks.EarlyStopping(patience=Config.PATIENCE, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
            SpikePerformanceCallback(X_val, y_val)
        ],
        verbose=1
    )
    
    models['gru'] = gru_model
    history['gru'] = gru_history
    completed_models.append('gru')
    
    # 체크포인트 저장
    checkpoint_manager.save_state(completed_models, models, history, evaluation_results)
else:
    print("\n2️⃣ GRU 모델 - 이미 완료 ✓")
    if os.path.exists(f"{Config.MODEL_DIR}gru_best.h5"):
        models['gru'] = tf.keras.models.load_model(f"{Config.MODEL_DIR}gru_best.h5",
                                                   custom_objects={'WeightedLoss': WeightedLoss})

# ============================================
# 5.3 CNN-LSTM 모델
# ============================================
if 'cnn_lstm' not in completed_models:
    print("\n3️⃣ CNN-LSTM 모델 학습 (복합 패턴 인식)")
    
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, 'cnn_lstm_checkpoint.h5')
    if os.path.exists(checkpoint_path):
        print("  체크포인트에서 모델 로드 중...")
        cnn_lstm_model = tf.keras.models.load_model(checkpoint_path, custom_objects={'WeightedLoss': WeightedLoss})
    else:
        cnn_lstm_model = ModelsV6.build_cnn_lstm(X_train.shape[1:])
        cnn_lstm_model.compile(
            optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE),
            loss=WeightedLoss(),
            metrics=['mae']
        )
    
    cnn_lstm_history = cnn_lstm_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                f"{Config.MODEL_DIR}cnn_lstm_best.h5",
                save_best_only=True,
                monitor='val_loss',
                verbose=0
            ),
            tf.keras.callbacks.EarlyStopping(patience=Config.PATIENCE, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
            SpikePerformanceCallback(X_val, y_val)
        ],
        verbose=1
    )
    
    models['cnn_lstm'] = cnn_lstm_model
    history['cnn_lstm'] = cnn_lstm_history
    completed_models.append('cnn_lstm')
    
    # 체크포인트 저장
    checkpoint_manager.save_state(completed_models, models, history, evaluation_results)
else:
    print("\n3️⃣ CNN-LSTM 모델 - 이미 완료 ✓")
    if os.path.exists(f"{Config.MODEL_DIR}cnn_lstm_best.h5"):
        models['cnn_lstm'] = tf.keras.models.load_model(f"{Config.MODEL_DIR}cnn_lstm_best.h5",
                                                        custom_objects={'WeightedLoss': WeightedLoss})

# ============================================
# 5.4 Spike Detector 모델
# ============================================
if 'spike' not in completed_models:
    print("\n4️⃣ Spike Detector 모델 학습 (이상치 감지)")
    
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, 'spike_checkpoint.h5')
    if os.path.exists(checkpoint_path):
        print("  체크포인트에서 모델 로드 중...")
        spike_model = tf.keras.models.load_model(checkpoint_path, custom_objects={'WeightedLoss': WeightedLoss})
    else:
        spike_model = ModelsV6.build_spike_detector(X_train.shape[1:])
        spike_model.compile(
            optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE),
            loss={
                'spike_value': WeightedLoss(),
                'spike_prob': 'binary_crossentropy'
            },
            loss_weights={
                'spike_value': 1.0,
                'spike_prob': 0.3
            },
            metrics=['mae']
        )
    
    spike_history = spike_model.fit(
        X_train, 
        [y_train, y_spike_class],
        validation_data=(X_val, [y_val, y_val_spike_class]),
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                f"{Config.MODEL_DIR}spike_best.h5",
                save_best_only=True,
                monitor='val_spike_value_loss',
                verbose=0
            ),
            tf.keras.callbacks.EarlyStopping(patience=Config.PATIENCE, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
        ],
        verbose=1
    )
    
    models['spike'] = spike_model
    history['spike'] = spike_history
    completed_models.append('spike')
    
    # 체크포인트 저장
    checkpoint_manager.save_state(completed_models, models, history, evaluation_results)
else:
    print("\n4️⃣ Spike Detector 모델 - 이미 완료 ✓")
    if os.path.exists(f"{Config.MODEL_DIR}spike_best.h5"):
        models['spike'] = tf.keras.models.load_model(f"{Config.MODEL_DIR}spike_best.h5",
                                                     custom_objects={'WeightedLoss': WeightedLoss})

# ============================================
# 5.5 Rule-Based 모델
# ============================================
if 'rule' not in completed_models:
    print("\n5️⃣ Rule-Based 모델 학습 (검증된 황금 패턴)")
    
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, 'rule_checkpoint.h5')
    if os.path.exists(checkpoint_path):
        print("  체크포인트에서 모델 로드 중...")
        rule_model = tf.keras.models.load_model(checkpoint_path, 
                                               custom_objects={'WeightedLoss': WeightedLoss, 
                                                             'M14RuleCorrection': M14RuleCorrection})
    else:
        rule_model = ModelsV6.build_rule_based_model(X_train.shape[1:], m14_train.shape[1])
        rule_model.compile(
            optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE * 0.5),
            loss=WeightedLoss(),
            metrics=['mae']
        )
    
    rule_history = rule_model.fit(
        [X_train, m14_train], 
        y_train,
        validation_data=([X_val, m14_val], y_val),
        epochs=50,  # Rule-based는 빠르게 수렴
        batch_size=Config.BATCH_SIZE,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                f"{Config.MODEL_DIR}rule_best.h5",
                save_best_only=True,
                monitor='val_loss',
                verbose=0
            ),
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        ],
        verbose=1
    )
    
    models['rule'] = rule_model
    history['rule'] = rule_history
    completed_models.append('rule')
    
    # 체크포인트 저장
    checkpoint_manager.save_state(completed_models, models, history, evaluation_results)
else:
    print("\n5️⃣ Rule-Based 모델 - 이미 완료 ✓")
    if os.path.exists(f"{Config.MODEL_DIR}rule_best.h5"):
        models['rule'] = tf.keras.models.load_model(f"{Config.MODEL_DIR}rule_best.h5",
                                                    custom_objects={'WeightedLoss': WeightedLoss,
                                                                  'M14RuleCorrection': M14RuleCorrection})

# ============================================
# 6. 최종 앙상블 모델 (수정된 버전)
# ============================================
if 'ensemble' not in completed_models:
    print("\n" + "="*60)
    print("🎯 최종 앙상블 모델 구성")
    print("="*60)
    
    # 입력 정의
    time_series_input = tf.keras.Input(shape=X_train.shape[1:], name='ensemble_time_input')
    m14_input = tf.keras.Input(shape=m14_train.shape[1], name='ensemble_m14_input')
    
    # 각 모델 예측
    lstm_pred = models['lstm'](time_series_input)
    gru_pred = models['gru'](time_series_input)
    cnn_lstm_pred = models['cnn_lstm'](time_series_input)
    spike_pred, spike_prob = models['spike'](time_series_input)
    rule_pred = models['rule']([time_series_input, m14_input])
    
    # M14 기반 동적 가중치 생성
    weight_dense = tf.keras.layers.Dense(32, activation='relu')(m14_input)
    weight_dense = tf.keras.layers.Dense(16, activation='relu')(weight_dense)
    weights = tf.keras.layers.Dense(5, activation='softmax', name='ensemble_weights')(weight_dense)
    
    # 가중치 분리
    w_lstm = tf.keras.layers.Lambda(lambda x: x[:, 0:1])(weights)
    w_gru = tf.keras.layers.Lambda(lambda x: x[:, 1:2])(weights)
    w_cnn = tf.keras.layers.Lambda(lambda x: x[:, 2:3])(weights)
    w_spike = tf.keras.layers.Lambda(lambda x: x[:, 3:4])(weights)
    w_rule = tf.keras.layers.Lambda(lambda x: x[:, 4:5])(weights)
    
    # 가중 평균
    weighted_lstm = tf.keras.layers.Multiply()([lstm_pred, w_lstm])
    weighted_gru = tf.keras.layers.Multiply()([gru_pred, w_gru])
    weighted_cnn = tf.keras.layers.Multiply()([cnn_lstm_pred, w_cnn])
    weighted_spike = tf.keras.layers.Multiply()([spike_pred, w_spike])
    weighted_rule = tf.keras.layers.Multiply()([rule_pred, w_rule])
    
    # 앙상블 예측
    ensemble_pred = tf.keras.layers.Add()([
        weighted_lstm, weighted_gru, weighted_cnn, 
        weighted_spike, weighted_rule
    ])
    
    # 최종 M14 규칙 보정 - name 속성 추가
    final_pred = M14RuleCorrection(name='ensemble_prediction')([ensemble_pred, m14_input])
    
    # spike_prob에도 name 추가
    spike_prob_output = tf.keras.layers.Lambda(lambda x: x, name='spike_probability')(spike_prob)
    
    # 앙상블 모델 정의
    ensemble_model = tf.keras.Model(
        inputs=[time_series_input, m14_input],
        outputs=[final_pred, spike_prob_output],
        name='Final_Ensemble_Model'
    )
    
    # 컴파일 - 출력 이름과 일치하도록 수정
    ensemble_model.compile(
        optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE * 0.5),
        loss={
            'ensemble_prediction': WeightedLoss(),
            'spike_probability': 'binary_crossentropy'
        },
        loss_weights={
            'ensemble_prediction': 1.0,
            'spike_probability': 0.3
        },
        metrics=['mae']
    )
    
    print("\n📊 앙상블 파인튜닝...")
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
    
    models['ensemble'] = ensemble_model
    history['ensemble'] = ensemble_history
    completed_models.append('ensemble')
    
    # 체크포인트 저장
    checkpoint_manager.save_state(completed_models, models, history, evaluation_results)
    
    print("\n✅ 5개 모델 + 앙상블 학습 완료!")
else:
    print("\n🎯 앙상블 모델 - 이미 완료 ✓")
    if os.path.exists(f"{Config.MODEL_DIR}ensemble_model.h5"):
        models['ensemble'] = tf.keras.models.load_model(f"{Config.MODEL_DIR}ensemble_model.h5",
                                                        custom_objects={'WeightedLoss': WeightedLoss,
                                                                      'M14RuleCorrection': M14RuleCorrection})

# ============================================
# 7. 평가
# ============================================
print("\n" + "="*60)
print("📊 모델 평가")
print("="*60)

for name, model in models.items():
    if name == 'ensemble':
        pred = model.predict([X_val, m14_val], verbose=0)[0].flatten()
    elif name == 'spike':
        pred = model.predict(X_val, verbose=0)[0].flatten()
    elif name == 'rule':
        pred = model.predict([X_val, m14_val], verbose=0).flatten()
    else:
        pred = model.predict(X_val, verbose=0).flatten()
    
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
    
    evaluation_results[name] = {
        'overall_mae': mae,
        'levels': level_performance
    }
    
    # 출력
    print(f"\n🎯 {name.upper()} 모델:")
    print(f"  전체 MAE: {mae:.2f}")
    for level, perf in level_performance.items():
        print(f"  {level}+: Recall={perf['recall']:.2%}, MAE={perf['mae']:.1f} (n={perf['count']})")

# 최종 선택
best_model = min(evaluation_results.keys(), key=lambda x: evaluation_results[x]['overall_mae'])
print(f"\n🏆 최고 성능: {best_model.upper()} 모델")
print(f"  MAE: {evaluation_results[best_model]['overall_mae']:.2f}")

# ============================================
# 8. 모델 저장
# ============================================
print("\n💾 모델 저장 중...")

for name, model in models.items():
    model.save(f"{Config.MODEL_DIR}{name}_model.h5")
    print(f"  {name}_model.h5 저장 완료")

# 평가 결과 저장
with open(f"{Config.MODEL_DIR}evaluation_results.json", 'w') as f:
    json.dump(evaluation_results, f, indent=2, default=str)

# 설정 저장
config_dict = {k: v for k, v in Config.__dict__.items() if not k.startswith('_')}
with open(f"{Config.MODEL_DIR}config.json", 'w') as f:
    json.dump(config_dict, f, indent=2)

print("  결과 파일 저장 완료")

# ============================================
# 9. 시각화
# ============================================
print("\n📈 결과 시각화 생성 중...")

fig = plt.figure(figsize=(20, 12))

# 1-5. 각 모델 학습 곡선
for idx, (name, hist) in enumerate(history.items()):
    if idx < 5:  # 개별 모델들
        ax = plt.subplot(3, 4, idx+1)
        
        if hasattr(hist, 'history'):
            if name == 'spike':
                loss = hist.history.get('spike_value_loss', [])
                val_loss = hist.history.get('val_spike_value_loss', [])
            else:
                loss = hist.history.get('loss', [])
                val_loss = hist.history.get('val_loss', [])
            
            if loss and val_loss:
                ax.plot(loss, label='Train Loss', alpha=0.8)
                ax.plot(val_loss, label='Val Loss', alpha=0.8)
        
        ax.set_title(f'{name.upper()} Learning Curve')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

# 6. 앙상블 학습 곡선
ax = plt.subplot(3, 4, 6)
if 'ensemble' in history and hasattr(history['ensemble'], 'history'):
    loss = history['ensemble'].history.get('ensemble_prediction_loss', [])
    val_loss = history['ensemble'].history.get('val_ensemble_prediction_loss', [])
    if loss and val_loss:
        ax.plot(loss, label='Train Loss', alpha=0.8)
        ax.plot(val_loss, label='Val Loss', alpha=0.8)
ax.set_title('ENSEMBLE Learning Curve')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
ax.grid(True, alpha=0.3)

# 7. 모델별 MAE 비교
ax = plt.subplot(3, 4, 7)
model_names = list(evaluation_results.keys())
maes = [evaluation_results[m]['overall_mae'] for m in model_names]
colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown']

bars = ax.bar(model_names, maes, color=colors[:len(model_names)])
ax.set_title('Model MAE Comparison')
ax.set_ylabel('MAE')
ax.set_ylim(0, max(maes) * 1.2)

for bar, mae in zip(bars, maes):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{mae:.1f}', ha='center', va='bottom')

# 8. 1400+ Recall 비교
ax = plt.subplot(3, 4, 8)
recalls_1400 = []
for m in model_names:
    if 1400 in evaluation_results[m]['levels']:
        recalls_1400.append(evaluation_results[m]['levels'][1400]['recall'] * 100)
    else:
        recalls_1400.append(0)

bars = ax.bar(model_names, recalls_1400, color=colors[:len(model_names)])
ax.set_title('1400+ Recall Comparison (%)')
ax.set_ylabel('Recall (%)')
ax.set_ylim(0, 105)

for bar, recall in zip(bars, recalls_1400):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{recall:.1f}%', ha='center', va='bottom')

# 9. 1500+ Recall 비교
ax = plt.subplot(3, 4, 9)
recalls_1500 = []
for m in model_names:
    if 1500 in evaluation_results[m]['levels']:
        recalls_1500.append(evaluation_results[m]['levels'][1500]['recall'] * 100)
    else:
        recalls_1500.append(0)

bars = ax.bar(model_names, recalls_1500, color=colors[:len(model_names)])
ax.set_title('1500+ Recall Comparison (%)')
ax.set_ylabel('Recall (%)')
ax.set_ylim(0, 105)

for bar, recall in zip(bars, recalls_1500):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{recall:.1f}%', ha='center', va='bottom')

# 10. 1600+ Recall 비교
ax = plt.subplot(3, 4, 10)
recalls_1600 = []
for m in model_names:
    if 1600 in evaluation_results[m]['levels']:
        recalls_1600.append(evaluation_results[m]['levels'][1600]['recall'] * 100)
    else:
        recalls_1600.append(0)

bars = ax.bar(model_names, recalls_1600, color=colors[:len(model_names)])
ax.set_title('1600+ Recall Comparison (%)')
ax.set_ylabel('Recall (%)')
ax.set_ylim(0, 105)

for bar, recall in zip(bars, recalls_1600):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{recall:.1f}%', ha='center', va='bottom')

# 11. 1700+ Recall 비교
ax = plt.subplot(3, 4, 11)
recalls_1700 = []
for m in model_names:
    if 1700 in evaluation_results[m]['levels']:
        recalls_1700.append(evaluation_results[m]['levels'][1700]['recall'] * 100)
    else:
        recalls_1700.append(0)

bars = ax.bar(model_names, recalls_1700, color=colors[:len(model_names)])
ax.set_title('1700+ Recall Comparison (%)')
ax.set_ylabel('Recall (%)')
ax.set_ylim(0, 105)

for bar, recall in zip(bars, recalls_1700):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{recall:.1f}%', ha='center', va='bottom')

# 12. 성능 요약
ax = plt.subplot(3, 4, 12)
ax.axis('off')

summary_text = "🏆 Performance Summary\n" + "="*35 + "\n"
summary_text += f"Best Model: {best_model.upper()}\n"
summary_text += f"Overall MAE: {evaluation_results[best_model]['overall_mae']:.2f}\n\n"

summary_text += "Recall by Level:\n"
for level in [1400, 1500, 1600, 1700]:
    if level in evaluation_results[best_model]['levels']:
        recall = evaluation_results[best_model]['levels'][level]['recall']
        mae = evaluation_results[best_model]['levels'][level]['mae']
        summary_text += f"  {level}+: {recall:6.1%} (MAE: {mae:.1f})\n"

summary_text += f"\n5개 모델 앙상블 완성!"

ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
       fontsize=11, verticalalignment='top', fontfamily='monospace')

plt.suptitle('V6 앙상블 모델 성능 분석', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{Config.MODEL_DIR}training_results.png", dpi=100, bbox_inches='tight')
print("  training_results.png 저장 완료")
plt.show()

# ============================================
# 10. 최종 출력 및 정리
# ============================================
print("\n" + "="*60)
print("🎉 모든 작업 완료!")
print("="*60)
print(f"📁 모델 저장 위치: {Config.MODEL_DIR}")
print(f"📂 시퀀스 파일: {Config.SEQUENCE_FILE}")
print(f"📁 체크포인트 위치: {Config.CHECKPOINT_DIR}")
print("\n📊 최종 성능:")
print(f"  최고 모델: {best_model.upper()}")
print(f"  전체 MAE: {evaluation_results[best_model]['overall_mae']:.2f}")
print("\n💡 다음 단계: 실시간 예측 시스템 적용")

# 체크포인트 정리 (옵션)
cleanup = input("\n체크포인트를 삭제하시겠습니까? (y/n): ")
if cleanup.lower() == 'y':
    import shutil
    shutil.rmtree(Config.CHECKPOINT_DIR)
    print("✅ 체크포인트 삭제 완료")

print("="*60)

# GPU 정보 출력
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print(f"\n🎮 GPU 사용: {len(gpus)}개")
    for gpu in gpus:
        print(f"  {gpu}")
else:
    print("\n💻 CPU 모드로 실행됨")