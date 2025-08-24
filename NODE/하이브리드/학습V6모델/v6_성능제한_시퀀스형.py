"""
ensemble_only_v6.py - 앙상블 모델 구성 전용
이미 학습된 개별 모델들을 로드하여 앙상블 구성
TensorFlow 2.15.0
"""

import tensorflow as tf
import numpy as np
import json
import os
import gc
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("🎯 앙상블 모델 구성 - 기존 학습된 모델 활용")
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
# 2. 커스텀 레이어 정의 (앙상블용)
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
        ratio = m14_features[:, 3:4] if m14_features.shape[1] > 3 else tf.ones_like(m14b)
        
        # 규칙 기반 보정
        # 1700+ 신호
        condition_1700 = tf.logical_and(
            tf.greater_equal(m14b, Config.M14B_THRESHOLDS[1700]),
            tf.greater_equal(ratio, Config.RATIO_THRESHOLDS[1700])
        )
        pred = tf.where(condition_1700, tf.maximum(pred, 1700), pred)
        
        # 1600+ 신호
        condition_1600 = tf.logical_and(
            tf.greater_equal(m14b, Config.M14B_THRESHOLDS[1600]),
            tf.greater_equal(ratio, Config.RATIO_THRESHOLDS[1600])
        )
        pred = tf.where(condition_1600, tf.maximum(pred, 1600), pred)
        
        # 1500+ 신호
        condition_1500 = tf.logical_and(
            tf.greater_equal(m14b, Config.M14B_THRESHOLDS[1500]),
            tf.greater_equal(ratio, Config.RATIO_THRESHOLDS[1500])
        )
        pred = tf.where(condition_1500, tf.maximum(pred, 1500), pred)
        
        # 1400+ 신호
        condition_1400 = tf.greater_equal(m14b, Config.M14B_THRESHOLDS[1400])
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

# ============================================
# 3. 데이터 로드 (평가용 소량만)
# ============================================
print("\n📂 데이터 로딩 중...")

# 전체 데이터가 아닌 검증용 일부만 로드
data = np.load(Config.SEQUENCE_FILE)
X_val = data['X'][-10000:].astype(np.float32)  # 마지막 10,000개만
y_val = data['y'][-10000:].astype(np.float32)
m14_val = data['m14_features'][-10000:].astype(np.float32)

print(f"  검증 데이터 shape: {X_val.shape}")
print(f"  1400+ 비율: {(y_val >= 1400).mean():.1%}")

# ============================================
# 4. 기존 학습된 모델 로드
# ============================================
print("\n📥 학습된 모델 로드 중...")

models = {}

# LSTM 모델 로드
try:
    lstm_path = f"{Config.MODEL_DIR}lstm_model.h5"
    if os.path.exists(lstm_path):
        models['lstm'] = tf.keras.models.load_model(lstm_path, custom_objects={'WeightedLoss': WeightedLoss})
        print("  ✅ LSTM 모델 로드 완료")
    else:
        print(f"  ⚠️ LSTM 모델 파일이 없습니다: {lstm_path}")
except Exception as e:
    print(f"  ❌ LSTM 로드 실패: {e}")

# GRU 모델 로드
try:
    gru_path = f"{Config.MODEL_DIR}gru_model.h5"
    if os.path.exists(gru_path):
        models['gru'] = tf.keras.models.load_model(gru_path, custom_objects={'WeightedLoss': WeightedLoss})
        print("  ✅ GRU 모델 로드 완료")
    else:
        print(f"  ⚠️ GRU 모델 파일이 없습니다: {gru_path}")
except Exception as e:
    print(f"  ❌ GRU 로드 실패: {e}")

# CNN-LSTM 모델 로드
try:
    cnn_lstm_path = f"{Config.MODEL_DIR}cnn_lstm_model.h5"
    if os.path.exists(cnn_lstm_path):
        models['cnn_lstm'] = tf.keras.models.load_model(cnn_lstm_path, custom_objects={'WeightedLoss': WeightedLoss})
        print("  ✅ CNN-LSTM 모델 로드 완료")
    else:
        print(f"  ⚠️ CNN-LSTM 모델 파일이 없습니다: {cnn_lstm_path}")
except Exception as e:
    print(f"  ❌ CNN-LSTM 로드 실패: {e}")

# Spike Detector 모델 로드
try:
    spike_path = f"{Config.MODEL_DIR}spike_model.h5"
    if os.path.exists(spike_path):
        models['spike'] = tf.keras.models.load_model(spike_path, custom_objects={'WeightedLoss': WeightedLoss})
        print("  ✅ Spike Detector 모델 로드 완료")
    else:
        print(f"  ⚠️ Spike Detector 모델 파일이 없습니다: {spike_path}")
except Exception as e:
    print(f"  ❌ Spike Detector 로드 실패: {e}")

print(f"\n💡 로드된 모델 수: {len(models)}개")

if len(models) == 0:
    print("❌ 로드된 모델이 없습니다. 개별 모델을 먼저 학습해주세요.")
    exit(1)

# ============================================
# 5. 앙상블 모델 구성
# ============================================
print("\n🔧 앙상블 모델 구성 중...")

# 입력 정의
time_series_input = tf.keras.Input(shape=(100, 5), name='ensemble_input')
m14_input = tf.keras.Input(shape=(4,), name='m14_features')

# 각 모델의 예측값 수집
predictions = []
model_names = []

# LSTM 예측
if 'lstm' in models:
    lstm_pred = models['lstm'](time_series_input)
    predictions.append(lstm_pred)
    model_names.append('lstm')

# GRU 예측
if 'gru' in models:
    gru_pred = models['gru'](time_series_input)
    predictions.append(gru_pred)
    model_names.append('gru')

# CNN-LSTM 예측
if 'cnn_lstm' in models:
    cnn_lstm_pred = models['cnn_lstm'](time_series_input)
    predictions.append(cnn_lstm_pred)
    model_names.append('cnn_lstm')

# Spike Detector 예측
spike_prob = None
if 'spike' in models:
    spike_outputs = models['spike'](time_series_input)
    if isinstance(spike_outputs, list):
        spike_pred = spike_outputs[0]
        spike_prob = spike_outputs[1]
    else:
        spike_pred = spike_outputs
    predictions.append(spike_pred)
    model_names.append('spike')

print(f"  앙상블에 포함된 모델: {model_names}")

# 앙상블 예측값 계산
if len(predictions) > 1:
    # 가중 평균 계산
    # 기본 가중치 설정
    weights = {
        'lstm': 0.25,
        'gru': 0.20,
        'cnn_lstm': 0.25,
        'spike': 0.30
    }
    
    # 실제 사용할 가중치 계산
    used_weights = [weights.get(name, 0.25) for name in model_names]
    total_weight = sum(used_weights)
    normalized_weights = [w/total_weight for w in used_weights]
    
    print(f"  정규화된 가중치: {dict(zip(model_names, normalized_weights))}")
    
    # 가중 평균 적용
    weighted_preds = []
    for pred, weight in zip(predictions, normalized_weights):
        weighted_pred = tf.keras.layers.Lambda(lambda x: x * weight)(pred)
        weighted_preds.append(weighted_pred)
    
    ensemble_pred = tf.keras.layers.Add()(weighted_preds)
else:
    # 모델이 하나만 있는 경우
    ensemble_pred = predictions[0]

# M14 규칙 기반 보정
final_pred = M14RuleCorrection()([ensemble_pred, m14_input])

# 앙상블 모델 생성
if spike_prob is not None:
    # Spike 확률 출력도 포함
    ensemble_model = tf.keras.Model(
        inputs=[time_series_input, m14_input],
        outputs=[final_pred, spike_prob],
        name='Ensemble_Model'
    )
else:
    # 예측값만 출력
    ensemble_model = tf.keras.Model(
        inputs=[time_series_input, m14_input],
        outputs=final_pred,
        name='Ensemble_Model'
    )

print("  ✅ 앙상블 모델 구성 완료")

# 모델 구조 출력
ensemble_model.summary()

# ============================================
# 6. 앙상블 모델 컴파일 및 파인튜닝
# ============================================
print("\n🎯 앙상블 모델 파인튜닝...")

# 컴파일
if spike_prob is not None:
    ensemble_model.compile(
        optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE),
        loss=[WeightedLoss(), 'binary_crossentropy'],
        loss_weights=[1.0, 0.3],
        metrics=['mae']
    )
else:
    ensemble_model.compile(
        optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE),
        loss=WeightedLoss(),
        metrics=['mae']
    )

# 학습 데이터 준비 (검증 데이터의 일부를 학습용으로 사용)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test, m14_train, m14_test = train_test_split(
    X_val, y_val, m14_val, test_size=0.3, random_state=42
)

print(f"\n  학습 데이터: {X_train.shape[0]:,}개")
print(f"  테스트 데이터: {X_test.shape[0]:,}개")

# 파인튜닝
if spike_prob is not None:
    # Spike 분류 레이블 생성
    y_spike_train = (y_train >= 1400).astype(np.float32)
    y_spike_test = (y_test >= 1400).astype(np.float32)
    
    history = ensemble_model.fit(
        [X_train, m14_train],
        [y_train, y_spike_train],
        validation_data=(
            [X_test, m14_test],
            [y_test, y_spike_test]
        ),
        epochs=Config.ENSEMBLE_EPOCHS,
        batch_size=Config.BATCH_SIZE,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                patience=5,
                factor=0.5,
                min_lr=1e-6,
                verbose=1
            )
        ],
        verbose=1
    )
else:
    history = ensemble_model.fit(
        [X_train, m14_train],
        y_train,
        validation_data=(
            [X_test, m14_test],
            y_test
        ),
        epochs=Config.ENSEMBLE_EPOCHS,
        batch_size=Config.BATCH_SIZE,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                patience=5,
                factor=0.5,
                min_lr=1e-6,
                verbose=1
            )
        ],
        verbose=1
    )

print("\n✅ 앙상블 파인튜닝 완료")

# ============================================
# 7. 모델 평가
# ============================================
print("\n📊 앙상블 모델 평가...")

# 예측
if spike_prob is not None:
    y_pred, spike_pred = ensemble_model.predict([X_test, m14_test], batch_size=Config.BATCH_SIZE)
    y_pred = y_pred.flatten()
else:
    y_pred = ensemble_model.predict([X_test, m14_test], batch_size=Config.BATCH_SIZE)
    y_pred = y_pred.flatten()

# 전체 성능
mae = np.mean(np.abs(y_test - y_pred))
print(f"\n  전체 MAE: {mae:.2f}")

# 구간별 성능
for level in [1400, 1500, 1600, 1700]:
    mask = y_test >= level
    if np.any(mask):
        recall = np.sum((y_pred >= level) & mask) / np.sum(mask)
        level_mae = np.mean(np.abs(y_test[mask] - y_pred[mask]))
        print(f"  {level}+: Recall={recall:.2%}, MAE={level_mae:.1f} (n={np.sum(mask)})")

# ============================================
# 8. 모델 저장
# ============================================
print("\n💾 앙상블 모델 저장 중...")

# 모델 저장
ensemble_path = f"{Config.MODEL_DIR}ensemble_model.h5"
try:
    # 커스텀 객체와 함께 저장
    ensemble_model.save(ensemble_path)
    print(f"  ✅ 앙상블 모델 저장 완료: {ensemble_path}")
except Exception as e:
    print(f"  ❌ 모델 저장 실패: {e}")
    # 대안: weights만 저장
    try:
        ensemble_model.save_weights(f"{Config.MODEL_DIR}ensemble_weights.h5")
        print(f"  ✅ 앙상블 가중치 저장 완료")
    except Exception as e2:
        print(f"  ❌ 가중치 저장도 실패: {e2}")

# 평가 결과 저장 (JSON 직렬화 오류 방지)
evaluation_results = {
    'ensemble': {
        'overall_mae': float(mae),
        'model_count': len(models),
        'included_models': model_names,
        'weights': dict(zip(model_names, normalized_weights))
    }
}

# 구간별 성능 추가
level_results = {}
for level in [1400, 1500, 1600, 1700]:
    mask = y_test >= level
    if np.any(mask):
        recall = np.sum((y_pred >= level) & mask) / np.sum(mask)
        level_mae = np.mean(np.abs(y_test[mask] - y_pred[mask]))
        level_results[f'level_{level}'] = {
            'recall': float(recall),
            'mae': float(level_mae),
            'count': int(np.sum(mask))
        }

evaluation_results['ensemble']['levels'] = level_results

# JSON 저장
try:
    with open(f"{Config.MODEL_DIR}ensemble_evaluation.json", 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    print("  ✅ 평가 결과 저장 완료")
except Exception as e:
    print(f"  ❌ JSON 저장 실패: {e}")

# ============================================
# 9. 시각화
# ============================================
print("\n📈 결과 시각화...")

try:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 학습 곡선
    if hasattr(history, 'history'):
        loss_key = 'loss' if 'loss' in history.history else 'm14_rule_correction_loss'
        val_loss_key = 'val_loss' if 'val_loss' in history.history else 'val_m14_rule_correction_loss'
        
        ax1.plot(history.history[loss_key], label='Train Loss')
        ax1.plot(history.history[val_loss_key], label='Val Loss')
        ax1.set_title('Ensemble Model Training')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 예측 vs 실제 (샘플링)
    sample_size = min(500, len(y_test))
    indices = np.random.choice(len(y_test), sample_size, replace=False)
    
    ax2.scatter(y_test[indices], y_pred[indices], alpha=0.5, s=10)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax2.set_xlabel('Actual')
    ax2.set_ylabel('Predicted')
    ax2.set_title(f'Ensemble Predictions (MAE: {mae:.2f})')
    ax2.grid(True, alpha=0.3)
    
    # 1400 라인 표시
    ax2.axhline(y=1400, color='orange', linestyle='--', alpha=0.7, label='1400 threshold')
    ax2.axvline(x=1400, color='orange', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"{Config.MODEL_DIR}ensemble_results.png", dpi=150, bbox_inches='tight')
    print("  ✅ 시각화 저장 완료")
    plt.show()
    
except Exception as e:
    print(f"  ❌ 시각화 실패: {e}")

# ============================================
# 10. 최종 요약
# ============================================
print("\n" + "="*60)
print("🎉 앙상블 모델 구성 완료!")
print("="*60)
print(f"📁 저장 위치: {Config.MODEL_DIR}")
print(f"🔧 포함된 모델: {', '.join(model_names)}")
print(f"📊 최종 성능: MAE = {mae:.2f}")
print(f"💾 저장된 파일:")
print(f"  - ensemble_model.h5 (또는 ensemble_weights.h5)")
print(f"  - ensemble_evaluation.json")
print(f"  - ensemble_results.png")
print("="*60)

# 메모리 정리
del X_val, y_val, m14_val
gc.collect()

print("\n💡 다음 단계:")
print("  1. predict_v6.py로 실시간 예측 테스트")
print("  2. 필요시 앙상블 가중치 조정")
print("  3. 추가 모델 학습 후 재구성")