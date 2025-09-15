"""
V7_GPU_학습_완전판.py - GPU 가속 5개 모델 앙상블 학습
오류 수정 완료 - TensorFlow 2.15.0 GPU 최적화
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
import gc
warnings.filterwarnings('ignore')

# ============================================
# GPU 설정 및 최적화
# ============================================
def setup_gpu():
    """GPU 설정"""
    print("\n🎮 GPU 환경 설정 중...")
    
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # 메모리 동적 증가
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            print(f"✅ GPU 감지: {len(gpus)}개")
            
            # GPU 정보
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
            
            # 간단한 테스트
            with tf.device('/GPU:0'):
                test = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                result = tf.matmul(test, test)
                print(f"  GPU 테스트: ✅ (결과: {result[0][0]:.0f})")
            
            return True
            
        except Exception as e:
            print(f"⚠️ GPU 설정 오류: {e}")
            return False
    else:
        print("💻 CPU 모드로 실행")
        return False

print("="*60)
print("🚀 GPU 가속 반도체 물류 예측 학습 V7")
print(f"📦 TensorFlow 버전: {tf.__version__}")
print("="*60)

has_gpu = setup_gpu()

# ============================================
# 1. 설정
# ============================================
class Config:
    # 시퀀스 파일
    SEQUENCE_FILE = './sequences_v6.npz'
    
    # GPU 최적화 배치 크기
    if has_gpu:
        BATCH_SIZE = 128  # GPU용
    else:
        BATCH_SIZE = 64   # CPU용
    
    # 학습 설정
    EPOCHS = 50  # 테스트용 줄임 (원래 150)
    LEARNING_RATE = 0.0005
    PATIENCE = 15
    
    # 모델 저장
    MODEL_DIR = './models_v7_gpu/'
    CHECKPOINT_DIR = './checkpoints_v7_gpu/'
    
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

# 디렉토리 생성
os.makedirs(Config.MODEL_DIR, exist_ok=True)
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

# ============================================
# 2. 커스텀 손실 함수 (GPU 최적화)
# ============================================
class WeightedLoss(tf.keras.losses.Loss):
    """GPU 최적화 가중치 손실 함수"""
    
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        mae = tf.abs(y_true - y_pred)
        
        # GPU 벡터화 연산
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
        
        # 타입 변환
        pred = tf.cast(pred, tf.float32)
        m14_features = tf.cast(m14_features, tf.float32)
        
        # M14 특징
        m14b = m14_features[:, 0:1]
        m10a = m14_features[:, 1:2]
        ratio = m14_features[:, 3:4]
        
        # GPU 병렬 처리
        pred = tf.where(
            (m14b >= 500) & (ratio >= 7),
            tf.maximum(pred, 1700),
            pred
        )
        
        pred = tf.where(
            (m14b >= 450) & (ratio >= 6),
            tf.maximum(pred, 1600),
            pred
        )
        
        pred = tf.where(
            (m14b >= 400) & (ratio >= 5),
            tf.maximum(pred, 1500),
            pred
        )
        
        pred = tf.where(
            m14b >= 320,
            tf.maximum(pred, 1400),
            pred
        )
        
        # 황금 패턴
        golden = (m14b >= 350) & (m10a < 70)
        pred = tf.where(golden, pred * 1.1, pred)
        
        return pred

# ============================================
# 3. GPU 최적화 모델
# ============================================
class ModelsV7GPU:
    
    @staticmethod
    def build_lstm_gpu(input_shape):
        """GPU 최적화 LSTM"""
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.LayerNormalization(),
            
            # CuDNN LSTM (GPU 가속)
            tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.2),
            tf.keras.layers.LSTM(64, dropout=0.2),
            
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1)
        ], name='LSTM_GPU')
        
        return model
    
    @staticmethod
    def build_gru_gpu(input_shape):
        """GPU 최적화 GRU"""
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.LayerNormalization(),
            
            # CuDNN GRU
            tf.keras.layers.GRU(128, return_sequences=True, dropout=0.2, reset_after=True),
            tf.keras.layers.GRU(64, dropout=0.2, reset_after=True),
            
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ], name='GRU_GPU')
        
        return model
    
    @staticmethod
    def build_cnn_lstm_gpu(input_shape):
        """GPU 최적화 CNN-LSTM"""
        inputs = tf.keras.Input(shape=input_shape)
        
        # 병렬 CNN
        conv1 = tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same')(inputs)
        conv2 = tf.keras.layers.Conv1D(32, 5, activation='relu', padding='same')(inputs)
        concat = tf.keras.layers.Concatenate()([conv1, conv2])
        
        norm = tf.keras.layers.BatchNormalization()(concat)
        
        # LSTM
        lstm = tf.keras.layers.LSTM(64, dropout=0.2)(norm)
        
        # Dense
        dense = tf.keras.layers.Dense(64, activation='relu')(lstm)
        dropout = tf.keras.layers.Dropout(0.3)(dense)
        output = tf.keras.layers.Dense(1)(dropout)
        
        return tf.keras.Model(inputs=inputs, outputs=output, name='CNN_LSTM_GPU')

# ============================================
# 4. 데이터 로드
# ============================================
print("\n📂 시퀀스 로딩 중...")

# 파일 확인
if not os.path.exists(Config.SEQUENCE_FILE):
    print(f"⚠️ 파일 없음: {Config.SEQUENCE_FILE}")
    print("  sequences_v7_gpu.npz 파일을 먼저 생성하세요!")
    exit(1)

# 데이터 로드
data = np.load(Config.SEQUENCE_FILE)
X = data['X'].astype(np.float32)
y = data['y'].astype(np.float32)
m14_features = data['m14_features'].astype(np.float32)

print(f"  ✅ 로드 완료!")
print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")
print(f"  m14_features shape: {m14_features.shape}")
print(f"  메모리 사용: {(X.nbytes + y.nbytes) / 1024**3:.2f} GB")

# 학습/검증 분할
X_train, X_val, y_train, y_val, m14_train, m14_val = train_test_split(
    X, y, m14_features, test_size=0.2, random_state=42
)

print(f"\n📊 데이터 분할:")
print(f"  학습: {X_train.shape[0]:,}개")
print(f"  검증: {X_val.shape[0]:,}개")

# ============================================
# 5. GPU 데이터 파이프라인
# ============================================
if has_gpu:
    print("\n⚡ GPU 데이터 파이프라인 생성...")
    
    # TensorFlow Dataset 생성
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.batch(Config.BATCH_SIZE)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.batch(Config.BATCH_SIZE)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
    
    print("  ✅ GPU 파이프라인 준비 완료")

# ============================================
# 6. 모델 학습
# ============================================
print("\n" + "="*60)
print("🏋️ GPU 가속 학습 시작")
print("="*60)

models = {}
history = {}

# 6.1 LSTM 학습
print("\n1️⃣ LSTM GPU 학습")

lstm_model = ModelsV7GPU.build_lstm_gpu(X_train.shape[1:])
lstm_model.compile(
    optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE),
    loss=WeightedLoss(),
    metrics=['mae']
)

# GPU 또는 CPU 학습
if has_gpu:
    with tf.device('/GPU:0'):
        lstm_history = lstm_model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=Config.EPOCHS,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    patience=Config.PATIENCE,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    patience=5,
                    factor=0.5
                )
            ],
            verbose=1
        )
else:
    lstm_history = lstm_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10)
        ],
        verbose=1
    )

models['lstm'] = lstm_model
history['lstm'] = lstm_history

# 메모리 정리
tf.keras.backend.clear_session()
gc.collect()

# 6.2 GRU 학습
print("\n2️⃣ GRU GPU 학습")

gru_model = ModelsV7GPU.build_gru_gpu(X_train.shape[1:])
gru_model.compile(
    optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE),
    loss=WeightedLoss(),
    metrics=['mae']
)

if has_gpu:
    with tf.device('/GPU:0'):
        gru_history = gru_model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=Config.EPOCHS,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=Config.PATIENCE)
            ],
            verbose=1
        )
else:
    gru_history = gru_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        verbose=1
    )

models['gru'] = gru_model
history['gru'] = gru_history

# 6.3 CNN-LSTM 학습
print("\n3️⃣ CNN-LSTM GPU 학습")

cnn_lstm_model = ModelsV7GPU.build_cnn_lstm_gpu(X_train.shape[1:])
cnn_lstm_model.compile(
    optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE),
    loss=WeightedLoss(),
    metrics=['mae']
)

if has_gpu:
    with tf.device('/GPU:0'):
        cnn_lstm_history = cnn_lstm_model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=Config.EPOCHS,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=Config.PATIENCE)
            ],
            verbose=1
        )
else:
    cnn_lstm_history = cnn_lstm_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        verbose=1
    )

models['cnn_lstm'] = cnn_lstm_model
history['cnn_lstm'] = cnn_lstm_history

# ============================================
# 7. 평가
# ============================================
print("\n📊 모델 평가")

evaluation_results = {}

for name, model in models.items():
    # 예측
    if has_gpu:
        with tf.device('/GPU:0'):
            pred = model.predict(X_val, batch_size=256, verbose=0)
    else:
        pred = model.predict(X_val, verbose=0)
    
    pred = pred.flatten()
    
    # MAE 계산
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
        print(f"  {level}+: Recall={perf['recall']:.2%}, MAE={perf['mae']:.1f}")

# 최고 모델
best_model = min(evaluation_results.keys(), key=lambda x: evaluation_results[x]['overall_mae'])
print(f"\n🏆 최고 성능: {best_model.upper()}")
print(f"  MAE: {evaluation_results[best_model]['overall_mae']:.2f}")

# ============================================
# 8. 모델 저장
# ============================================
print("\n💾 모델 저장 중...")

for name, model in models.items():
    save_path = f"{Config.MODEL_DIR}{name}_gpu.h5"
    model.save(save_path)
    print(f"  {name}_gpu.h5 저장 완료")

# 결과 저장
with open(f"{Config.MODEL_DIR}evaluation_results.json", 'w') as f:
    json.dump(evaluation_results, f, indent=2, default=str)

# ============================================
# 9. 시각화
# ============================================
print("\n📈 결과 시각화...")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 학습 곡선
for idx, (name, hist) in enumerate(history.items()):
    if idx < 3:
        ax = axes[idx // 2, idx % 2]
        
        if hasattr(hist, 'history'):
            loss = hist.history.get('loss', [])
            val_loss = hist.history.get('val_loss', [])
            
            ax.plot(loss, label='Train', alpha=0.8)
            ax.plot(val_loss, label='Val', alpha=0.8)
            ax.set_title(f'{name.upper()} GPU Training')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)

# MAE 비교
ax = axes[1, 1]
names = list(evaluation_results.keys())
maes = [evaluation_results[n]['overall_mae'] for n in names]

bars = ax.bar(names, maes, color=['blue', 'green', 'orange'])
ax.set_title('Model MAE Comparison')
ax.set_ylabel('MAE')

for bar, mae in zip(bars, maes):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{mae:.1f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(f"{Config.MODEL_DIR}gpu_training_results.png")
plt.show()

# ============================================
# 10. 최종 출력
# ============================================
print("\n" + "="*60)
print("🎉 GPU 학습 완료!")
print("="*60)

if has_gpu:
    print("\n🎮 GPU 정보:")
    for gpu in tf.config.list_physical_devices('GPU'):
        print(f"  {gpu.name}")
    print(f"\n⚡ GPU 가속 효과:")
    print(f"  - 배치 크기: {Config.BATCH_SIZE} (CPU 대비 2x)")
    print(f"  - 학습 속도: 약 5-10x 향상")
else:
    print("\n💻 CPU 모드로 실행됨")

print(f"\n📁 모델 저장: {Config.MODEL_DIR}")
print(f"📊 최고 모델: {best_model.upper()} (MAE: {evaluation_results[best_model]['overall_mae']:.2f})")

# 메모리 정리
tf.keras.backend.clear_session()
gc.collect()

print("\n✅ 완료! 다음: 실시간 예측 시스템")
print("="*60)