"""
train_v6.py - 학습 전용 코드
미리 생성된 시퀀스를 로드하여 학습
TensorFlow 2.15.0
"""

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json
import os
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("🚀 반도체 물류 예측 학습 V6")
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

# ============================================
# 2. 데이터 로드
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
# 3. 모델 정의
# ============================================
class ModelsV6:
    
    @staticmethod
    def build_lstm_model(input_shape):
        """LSTM 모델 (기본 시계열 예측)"""
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
        """개선된 GRU 모델 (전체 구간 안정적 예측)"""
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
        """CNN-LSTM 모델 (패턴 감지 + 시계열)"""
        inputs = tf.keras.Input(shape=input_shape, name='cnn_lstm_input')
        
        # Multi-scale CNN
        conv1 = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
        conv2 = tf.keras.layers.Conv1D(64, 5, activation='relu', padding='same')(inputs)
        conv3 = tf.keras.layers.Conv1D(64, 7, activation='relu', padding='same')(inputs)
        
        # Concatenate
        concat = tf.keras.layers.Concatenate()([conv1, conv2, conv3])
        pool = tf.keras.layers.MaxPooling1D(pool_size=2)(concat)
        
        # LSTM layers
        lstm1 = tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.2)(pool)
        lstm2 = tf.keras.layers.LSTM(64, dropout=0.2)(lstm1)
        
        # Dense layers
        dense1 = tf.keras.layers.Dense(128, activation='relu')(lstm2)
        dropout = tf.keras.layers.Dropout(0.3)(dense1)
        dense2 = tf.keras.layers.Dense(64, activation='relu')(dropout)
        
        # Output
        output = tf.keras.layers.Dense(1, name='cnn_lstm_output')(dense2)
        
        model = tf.keras.Model(inputs=inputs, outputs=output, name='CNN_LSTM_Model')
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
# 5. 콜백
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
            
            # 개선된 경우만 출력
            if spike_mae < self.best_spike_mae:
                self.best_spike_mae = spike_mae
                
                # 레벨별 성능
                level_performance = {}
                for level in [1400, 1500, 1600, 1700]:
                    level_mask = self.y_val >= level
                    if np.any(level_mask):
                        recall = np.sum((y_pred >= level) & level_mask) / np.sum(level_mask)
                        level_performance[f'{level}+'] = recall
                
                print(f"\n🎯 Epoch {epoch+1} - 1400+ MAE: {spike_mae:.2f} (Best!)")
                for level, recall in level_performance.items():
                    print(f"   {level} Recall: {recall:.2%}")

# ============================================
# 6. 학습 파이프라인
# ============================================
print("\n" + "="*60)
print("🏋️ 모델 학습 시작")
print("="*60)

# 모델 저장용 딕셔너리
models = {}
history = {}
evaluation_results = {}

# ============================================
# 1. LSTM 모델
# ============================================
print("\n1️⃣ LSTM 모델 학습")
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
        tf.keras.callbacks.EarlyStopping(patience=Config.PATIENCE, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
        SpikePerformanceCallback(X_val, y_val)
    ],
    verbose=1
)

models['lstm'] = lstm_model
history['lstm'] = lstm_history

# ============================================
# 2. GRU 모델
# ============================================
print("\n2️⃣ Enhanced GRU 모델 학습")
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

models['gru'] = gru_model
history['gru'] = gru_history

# ============================================
# 3. CNN-LSTM 모델
# ============================================
print("\n3️⃣ CNN-LSTM 모델 학습")
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
        tf.keras.callbacks.EarlyStopping(patience=Config.PATIENCE, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
        SpikePerformanceCallback(X_val, y_val)
    ],
    verbose=1
)

models['cnn_lstm'] = cnn_lstm_model
history['cnn_lstm'] = cnn_lstm_history

# ============================================
# 4. Spike Detector 모델
# ============================================
print("\n4️⃣ Spike Detector 모델 학습 (1400+ 특화)")
spike_model = ModelsV6.build_spike_detector(X_train.shape[1:])

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

models['spike'] = spike_model
history['spike'] = spike_history

# ============================================
# 5. 최종 앙상블 모델
# ============================================
print("\n5️⃣ 최종 앙상블 모델 구성")

# 모든 모델의 예측값을 결합하는 앙상블
time_series_input = tf.keras.Input(shape=X_train.shape[1:], name='ensemble_input')
m14_input = tf.keras.Input(shape=m14_train.shape[1], name='m14_features')

# 각 모델 예측
lstm_pred = lstm_model(time_series_input)
gru_pred = gru_model(time_series_input)
cnn_lstm_pred = cnn_lstm_model(time_series_input)
spike_pred, spike_prob = spike_model(time_series_input)

# M14 기반 동적 가중치
weight_dense = tf.keras.layers.Dense(32, activation='relu')(m14_input)
weight_dense = tf.keras.layers.Dense(16, activation='relu')(weight_dense)
weights = tf.keras.layers.Dense(4, activation='softmax', name='ensemble_weights')(weight_dense)

# 가중 평균
w_lstm = tf.keras.layers.Lambda(lambda x: x[:, 0:1])(weights)
w_gru = tf.keras.layers.Lambda(lambda x: x[:, 1:2])(weights)
w_cnn = tf.keras.layers.Lambda(lambda x: x[:, 2:3])(weights)
w_spike = tf.keras.layers.Lambda(lambda x: x[:, 3:4])(weights)

weighted_lstm = tf.keras.layers.Multiply()([lstm_pred, w_lstm])
weighted_gru = tf.keras.layers.Multiply()([gru_pred, w_gru])
weighted_cnn = tf.keras.layers.Multiply()([cnn_lstm_pred, w_cnn])
weighted_spike = tf.keras.layers.Multiply()([spike_pred, w_spike])

# 최종 예측
ensemble_pred = tf.keras.layers.Add()([weighted_lstm, weighted_gru, weighted_cnn, weighted_spike])

# M14 규칙 보정
final_pred = M14RuleCorrection()([ensemble_pred, m14_input])

ensemble_model = tf.keras.Model(
    inputs=[time_series_input, m14_input],
    outputs=[final_pred, spike_prob],
    name='Final_Ensemble'
)

ensemble_model.compile(
    optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE * 0.5),
    loss={
        'm14_rule_correction': WeightedLoss(),
        'spike_prob': 'binary_crossentropy'
    },
    loss_weights={
        'm14_rule_correction': 1.0,
        'spike_prob': 0.3
    },
    metrics=['mae']
)

# 앙상블 파인튜닝
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

print("\n✅ 모든 모델 학습 완료!")

# ============================================
# 7. 평가
# ============================================
print("\n" + "="*60)
print("📊 모델 평가")
print("="*60)

for name, model in models.items():
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

# ============================================
# 8. 모델 저장
# ============================================
print("\n💾 모델 저장 중...")

for name, model in models.items():
    model.save(f"{Config.MODEL_DIR}{name}_model.h5")
    print(f"  {name}_model.h5 저장 완료")

# 평가 결과 저장
with open(f"{Config.MODEL_DIR}evaluation_results.json", 'w') as f:
    json.dump(evaluation_results, f, indent=2)

# 설정 저장
config_dict = {k: v for k, v in Config.__dict__.items() if not k.startswith('_')}
with open(f"{Config.MODEL_DIR}config.json", 'w') as f:
    json.dump(config_dict, f, indent=2)

print("  결과 파일 저장 완료")

# ============================================
# 9. 시각화
# ============================================
print("\n📈 결과 시각화 생성 중...")

fig = plt.figure(figsize=(18, 12))

# 학습 곡선 (2x3 그리드의 위 5개)
for idx, (name, hist) in enumerate(history.items()):
    ax = plt.subplot(3, 3, idx+1)
    
    # history 객체에서 데이터 추출
    if hasattr(hist, 'history'):
        loss = hist.history.get('loss', hist.history.get('spike_value_loss', []))
        val_loss = hist.history.get('val_loss', hist.history.get('val_spike_value_loss', []))
    else:
        loss = []
        val_loss = []
    
    if loss and val_loss:
        ax.plot(loss, label='Train Loss')
        ax.plot(val_loss, label='Val Loss')
    
    ax.set_title(f'{name.upper()} Learning Curve')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

# 6. 모델별 MAE 비교
ax = plt.subplot(3, 3, 6)
model_names = list(evaluation_results.keys())
maes = [evaluation_results[m]['overall_mae'] for m in model_names]
colors = ['blue', 'green', 'orange', 'red', 'purple']

bars = ax.bar(model_names, maes, color=colors[:len(model_names)])
ax.set_title('Model MAE Comparison')
ax.set_ylabel('MAE')
ax.set_ylim(0, max(maes) * 1.2)

for bar, mae in zip(bars, maes):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{mae:.1f}', ha='center', va='bottom')

# 7. 1400+ Recall 비교
ax = plt.subplot(3, 3, 7)
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

# 8. 1500+ Recall 비교
ax = plt.subplot(3, 3, 8)
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

# 9. 성능 요약
ax = plt.subplot(3, 3, 9)
ax.axis('off')

summary_text = "🏆 Performance Summary\n" + "="*30 + "\n"
summary_text += f"Best Model: {best_model.upper()}\n"
summary_text += f"Overall MAE: {evaluation_results[best_model]['overall_mae']:.2f}\n\n"

summary_text += "Recall by Level:\n"
for level in [1400, 1500, 1600, 1700]:
    if level in evaluation_results[best_model]['levels']:
        recall = evaluation_results[best_model]['levels'][level]['recall']
        summary_text += f"{level}+: {recall:.1%}\n"

ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
       fontsize=11, verticalalignment='top', fontfamily='monospace')

plt.suptitle('학습 V6 모델 성능 분석', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{Config.MODEL_DIR}training_results.png", dpi=100, bbox_inches='tight')
print("  training_results.png 저장 완료")
plt.show()

print("\n" + "="*60)
print("🎉 모든 작업 완료!")
print(f"📁 저장 위치: {Config.MODEL_DIR}")
print(f"📂 시퀀스 파일: {Config.SEQUENCE_FILE}")
print("="*60)

# GPU 정보 출력
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print(f"\n🎮 GPU 사용: {len(gpus)}개")
    for gpu in gpus:
        print(f"  {gpu}")
else:
    print("\n💻 CPU 모드로 실행됨")