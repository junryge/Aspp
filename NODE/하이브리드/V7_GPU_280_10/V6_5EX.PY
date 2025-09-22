"""
🔥 ExtremeNet V6.5 Final - 구간 재정의 완전판
================================================================
✅ 구간 재정의:
   - 정상: ~1400
   - 주의: 1400~1650
   - 주의관찰(신호): 1651~1699 ← 1700+ 급증 전조!
   - 위험: 1700+
✅ 신호구간(1651~1699) 특별 학습
✅ StandardScaler로 변경
✅ 실시간 예측 시스템 개선
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils import class_weight
import warnings
import os
import pickle
import json
from datetime import datetime

warnings.filterwarnings('ignore')
tf.random.set_seed(42)
np.random.seed(42)

# GPU 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print(f"🔧 GPU {len(gpus)}개 발견!")
    for i, gpu in enumerate(gpus):
        tf.config.experimental.set_memory_growth(gpu, True)
    strategy = tf.distribute.MirroredStrategy()
    DEVICE = 'GPU'
    BATCH_SIZE = 128  # GPU용 배치 크기
else:
    print("⚠️ GPU 없음, CPU 사용")
    strategy = tf.distribute.get_strategy()
    DEVICE = 'CPU'
    BATCH_SIZE = 32

print("="*80)
print("🔥 ExtremeNet V6.5 Final - 구간 재정의 완전판")
print(f"📦 TensorFlow: {tf.__version__}")
print(f"🔧 Device: {DEVICE}")
print(f"📊 배치 크기: {BATCH_SIZE}")
print("="*80)

# ========================================
# 데이터 처리 (구간 재정의)
# ========================================
class DataProcessor:
    def __init__(self):
        self.seq_len = 100
        self.pred_len = 10
        
        # 스케일러
        self.scaler_X = StandardScaler()
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))
        
        # ⭐ 구간 재정의
        self.NORMAL_MAX = 1400      # 정상 상한
        self.CAUTION_MAX = 1650     # 주의 상한
        self.SIGNAL_MIN = 1651      # 신호 시작 (1700+ 전조)
        self.SIGNAL_MAX = 1699      # 신호 끝
        self.DANGER_MIN = 1700      # 위험 시작
        
        self.feature_columns = None
        
    def load_data(self, filepath):
        """데이터 로드"""
        print(f"\n📂 데이터 로딩: {filepath}")
        df = pd.read_csv(filepath)
        
        # 시간 정렬
        df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), 
                                       format='%Y%m%d%H%M', errors='coerce')
        df = df.sort_values('CURRTIME').reset_index(drop=True)
        
        # 0값 제거
        df = df[df['TOTALCNT'] > 0].reset_index(drop=True)
        print(f"✅ 데이터: {len(df):,}개")
        
        # 구간 통계 (재정의)
        normal = (df['TOTALCNT'] < self.NORMAL_MAX).sum()
        caution = ((df['TOTALCNT'] >= self.NORMAL_MAX) & 
                  (df['TOTALCNT'] <= self.CAUTION_MAX)).sum()
        signal = ((df['TOTALCNT'] >= self.SIGNAL_MIN) & 
                 (df['TOTALCNT'] <= self.SIGNAL_MAX)).sum()
        danger = (df['TOTALCNT'] >= self.DANGER_MIN).sum()
        
        print(f"\n📊 TOTALCNT 분포 (재정의):")
        print(f"  🟢 정상(~1400): {normal:,}개 ({normal/len(df)*100:.1f}%)")
        print(f"  🟡 주의(1400~1650): {caution:,}개 ({caution/len(df)*100:.1f}%)")
        print(f"  🟠 신호(1651~1699): {signal:,}개 ({signal/len(df)*100:.1f}%) ← 1700+ 전조!")
        print(f"  🔴 위험(1700+): {danger:,}개 ({danger/len(df)*100:.1f}%)")
        
        return df
    
    def create_features(self, df):
        """특성 생성 (신호구간 강조)"""
        print("\n⚙️ 특성 생성 중...")
        
        # 기본 특성
        df['RATIO'] = df['M14AM14B'] / (df['M14AM10A'] + 1)
        df['GOLDEN'] = ((df['M14AM14B'] > 300) & (df['M14AM10A'] < 80)).astype(int)
        
        # ⭐ 신호구간 특성 강조
        df['SIGNAL_ZONE'] = ((df['TOTALCNT'] >= self.SIGNAL_MIN) & 
                             (df['TOTALCNT'] <= self.SIGNAL_MAX)).astype(int)
        df['PRE_SIGNAL'] = ((df['TOTALCNT'] >= 1600) & 
                            (df['TOTALCNT'] < self.SIGNAL_MIN)).astype(int)
        df['DANGER_ZONE'] = (df['TOTALCNT'] >= self.DANGER_MIN).astype(int)
        
        # 이동평균
        for w in [5, 10, 20]:
            df[f'MA_{w}'] = df['TOTALCNT'].rolling(w, min_periods=1).mean()
            df[f'STD_{w}'] = df['TOTALCNT'].rolling(w, min_periods=1).std().fillna(0)
            df[f'M14B_MA_{w}'] = df['M14AM14B'].rolling(w, min_periods=1).mean()
        
        # 변화율
        for lag in [1, 5, 10]:
            df[f'CHANGE_{lag}'] = df['TOTALCNT'].diff(lag).fillna(0)
            df[f'M14B_CHANGE_{lag}'] = df['M14AM14B'].diff(lag).fillna(0)
        
        # 연속 상승
        df['RISE'] = (df['TOTALCNT'] > df['TOTALCNT'].shift(1)).astype(int)
        df['RISE_COUNT'] = df['RISE'].rolling(10, min_periods=1).sum()
        
        # 패턴 (신호구간 포함)
        df['PATTERN'] = 0
        signal_mask = ((df['TOTALCNT'] >= self.SIGNAL_MIN) & 
                      (df['TOTALCNT'] <= self.SIGNAL_MAX))
        high_m14b = df['M14AM14B'].rolling(100, min_periods=1).mean() > 380
        
        df.loc[signal_mask, 'PATTERN'] = 1  # UU1: 신호
        df.loc[high_m14b, 'PATTERN'] = 2     # UU2: 고위험
        
        # 추세
        df['TREND'] = 0
        for i in range(20, len(df)):
            recent = df['TOTALCNT'].iloc[i-20:i].values
            if len(recent) > 1:
                slope = np.polyfit(range(len(recent)), recent, 1)[0]
                if slope > 5:
                    df.loc[i, 'TREND'] = 2  # 상승
                elif slope < -5:
                    df.loc[i, 'TREND'] = 0  # 하락
                else:
                    df.loc[i, 'TREND'] = 1  # 보합
        
        print(f"✅ 특성 생성 완료")
        return df
    
    def create_sequences(self, df):
        """시퀀스 생성 (상승 추세 신호구간만 오버샘플링)"""
        print("\n🔄 시퀀스 생성 중...")
        
        # 사용할 컬럼
        self.feature_columns = [
            'TOTALCNT', 'M14AM14B', 'M14AM14BSUM', 'M14AM10A', 'M14AM16',
            'RATIO', 'GOLDEN', 
            'SIGNAL_ZONE', 'PRE_SIGNAL', 'DANGER_ZONE',  # 신호구간 특성
            'PATTERN', 'TREND',
            'MA_5', 'MA_10', 'MA_20',
            'STD_5', 'STD_10', 'STD_20',
            'M14B_MA_5', 'M14B_MA_10', 'M14B_MA_20',
            'CHANGE_1', 'CHANGE_5', 'CHANGE_10',
            'M14B_CHANGE_1', 'M14B_CHANGE_5', 'M14B_CHANGE_10',
            'RISE_COUNT'
        ]
        
        print(f"  특성: {len(self.feature_columns)}개")
        
        # 데이터 준비
        X_data = df[self.feature_columns].values
        y_data = df['TOTALCNT'].shift(-self.pred_len).values
        
        # 시퀀스 생성
        X, y = [], []
        signal_X, signal_y = [], []  # 신호구간 따로 저장
        
        for i in range(len(df) - self.seq_len - self.pred_len):
            X.append(X_data[i:i+self.seq_len])
            target = y_data[i+self.seq_len-1]
            current = df['TOTALCNT'].iloc[i+self.seq_len-1]
            y.append(target)
            
            # ⭐ 중요: 상승 추세인지 확인 (현재 < 미래)
            is_rising = target > current  # 10분 후가 현재보다 높으면 상승
            
            # 상승 추세일 때만 위험 신호로 처리
            if is_rising:
                # 신호구간(1651~1699) + 상승 → 오버샘플링 5배
                if self.SIGNAL_MIN <= target <= self.SIGNAL_MAX:
                    for _ in range(5):
                        signal_X.append(X_data[i:i+self.seq_len])
                        signal_y.append(target)
                # 위험구간(1700+) + 상승 → 오버샘플링 10배
                elif target >= self.DANGER_MIN:
                    for _ in range(10):
                        signal_X.append(X_data[i:i+self.seq_len])
                        signal_y.append(target)
        
        # 오버샘플링 데이터 추가
        if signal_X:
            X.extend(signal_X)
            y.extend(signal_y)
            print(f"  🔥 상승 추세 신호/위험구간 {len(signal_X)}개 추가 (오버샘플링)")
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        
        # NaN 제거
        valid_idx = ~np.isnan(y)
        X = X[valid_idx]
        y = y[valid_idx]
        
        print(f"  총 시퀀스: {X.shape[0]}개")
        
        # 셔플
        shuffle_idx = np.arange(len(X))
        np.random.shuffle(shuffle_idx)
        X = X[shuffle_idx]
        y = y[shuffle_idx]
        
        # 데이터 분할
        train_size = int(len(X) * 0.7)
        val_size = int(len(X) * 0.15)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]
        X_test = X[train_size+val_size:]
        y_test = y[train_size+val_size:]
        
        # 스케일링
        print("  스케일링 중...")
        
        # X 스케일링
        X_train_flat = X_train.reshape(-1, X_train.shape[-1])
        self.scaler_X.fit(X_train_flat)
        
        X_train_scaled = self.scaler_X.transform(X_train_flat).reshape(X_train.shape)
        X_val_scaled = self.scaler_X.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        X_test_scaled = self.scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        
        # y 스케일링
        self.scaler_y.fit(y_train.reshape(-1, 1))
        y_train_scaled = self.scaler_y.transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = self.scaler_y.transform(y_val.reshape(-1, 1)).flatten()
        y_test_scaled = self.scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        
        print(f"\n📊 데이터 분할:")
        print(f"  Train: {len(X_train):,}개")
        print(f"  Val: {len(X_val):,}개")
        print(f"  Test: {len(X_test):,}개")
        
        # 신호구간 통계 (상승 추세만 위험으로 간주)
        signal_train = ((y_train >= self.SIGNAL_MIN) & (y_train <= self.SIGNAL_MAX)).sum()
        signal_val = ((y_val >= self.SIGNAL_MIN) & (y_val <= self.SIGNAL_MAX)).sum()
        signal_test = ((y_test >= self.SIGNAL_MIN) & (y_test <= self.SIGNAL_MAX)).sum()
        
        danger_train = (y_train >= self.DANGER_MIN).sum()
        danger_val = (y_val >= self.DANGER_MIN).sum()
        danger_test = (y_test >= self.DANGER_MIN).sum()
        
        print(f"\n🟠 신호구간(1651~1699) 샘플 (상승/하락 포함):")
        print(f"  Train: {signal_train}개 ({signal_train/len(X_train)*100:.1f}%)")
        print(f"  Val: {signal_val}개 ({signal_val/len(X_val)*100:.1f}%)")
        print(f"  Test: {signal_test}개 ({signal_test/len(X_test)*100:.1f}%)")
        print(f"  ※ 상승 추세만 위험 신호로 학습됨")
        
        print(f"\n🔴 위험구간(1700+) 샘플 (상승/하락 포함):")
        print(f"  Train: {danger_train}개 ({danger_train/len(X_train)*100:.1f}%)")
        print(f"  Val: {danger_val}개 ({danger_val/len(X_val)*100:.1f}%)")
        print(f"  Test: {danger_test}개 ({danger_test/len(X_test)*100:.1f}%)")
        print(f"  ※ 상승 추세만 위험으로 학습됨")
        
        # 구간 레이블 (4구간)
        y_train_level = np.zeros(len(y_train), dtype=int)
        y_train_level[(y_train >= self.NORMAL_MAX) & (y_train <= self.CAUTION_MAX)] = 1
        y_train_level[(y_train >= self.SIGNAL_MIN) & (y_train <= self.SIGNAL_MAX)] = 2
        y_train_level[y_train >= self.DANGER_MIN] = 3
        
        y_val_level = np.zeros(len(y_val), dtype=int)
        y_val_level[(y_val >= self.NORMAL_MAX) & (y_val <= self.CAUTION_MAX)] = 1
        y_val_level[(y_val >= self.SIGNAL_MIN) & (y_val <= self.SIGNAL_MAX)] = 2
        y_val_level[y_val >= self.DANGER_MIN] = 3
        
        y_test_level = np.zeros(len(y_test), dtype=int)
        y_test_level[(y_test >= self.NORMAL_MAX) & (y_test <= self.CAUTION_MAX)] = 1
        y_test_level[(y_test >= self.SIGNAL_MIN) & (y_test <= self.SIGNAL_MAX)] = 2
        y_test_level[y_test >= self.DANGER_MIN] = 3
        
        # 신호 이진 레이블
        y_train_signal = ((y_train >= self.SIGNAL_MIN) & (y_train <= self.SIGNAL_MAX)).astype(int)
        y_val_signal = ((y_val >= self.SIGNAL_MIN) & (y_val <= self.SIGNAL_MAX)).astype(int)
        y_test_signal = ((y_test >= self.SIGNAL_MIN) & (y_test <= self.SIGNAL_MAX)).astype(int)
        
        # 스케일러 저장
        os.makedirs('scalers_final', exist_ok=True)
        with open('scalers_final/scaler_X.pkl', 'wb') as f:
            pickle.dump(self.scaler_X, f)
        with open('scalers_final/scaler_y.pkl', 'wb') as f:
            pickle.dump(self.scaler_y, f)
        
        return (X_train_scaled, y_train_scaled, y_train, y_train_level, y_train_signal), \
               (X_val_scaled, y_val_scaled, y_val, y_val_level, y_val_signal), \
               (X_test_scaled, y_test_scaled, y_test, y_test_level, y_test_signal)

# ========================================
# ExtremeNet 모델
# ========================================
def build_extremenet(input_shape):
    """ExtremeNet V6.5 Final"""
    inputs = Input(shape=input_shape)
    
    # LSTM 브랜치 (장기 패턴)
    lstm = LSTM(128, return_sequences=True)(inputs)
    lstm = BatchNormalization()(lstm)
    lstm = LSTM(64)(lstm)
    lstm = Dropout(0.3)(lstm)
    
    # Attention 브랜치 (중요 시점)
    attn = MultiHeadAttention(num_heads=4, key_dim=16)(inputs, inputs)
    attn = GlobalAveragePooling1D()(attn)
    
    # CNN 브랜치 (신호 패턴)
    cnn = Conv1D(64, 5, activation='relu')(inputs)
    cnn = BatchNormalization()(cnn)
    cnn = Conv1D(32, 3, activation='relu')(cnn)
    cnn = GlobalMaxPooling1D()(cnn)
    
    # Recent 브랜치 (최근 급변)
    recent = Lambda(lambda x: x[:, -20:, :])(inputs)
    recent = GRU(32)(recent)
    
    # 통합
    merged = Concatenate()([lstm, attn, cnn, recent])
    merged = BatchNormalization()(merged)
    
    # Dense
    x = Dense(128, activation='relu')(merged)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # 출력
    output_reg = Dense(1, name='regression')(x)  # 값 예측
    output_cls = Dense(4, activation='softmax', name='classification')(x)  # 4구간
    output_signal = Dense(1, activation='sigmoid', name='signal')(x)  # 신호 감지
    
    model = Model(inputs, [output_reg, output_cls, output_signal], name='ExtremeNet_V65_Final')
    return model

# ========================================
# 체크포인트
# ========================================
class CheckpointManager:
    def __init__(self):
        self.checkpoint_dir = 'checkpoints_final/'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.model_path = os.path.join(self.checkpoint_dir, 'extremenet_final.keras')
        self.history_file = os.path.join(self.checkpoint_dir, 'history.json')
        
    def save_checkpoint(self, model, epoch, loss):
        model.save(self.model_path)
        history = {'epoch': epoch, 'loss': float(loss)}
        with open(self.history_file, 'w') as f:
            json.dump(history, f)
        print(f"💾 체크포인트 저장: 에포크 {epoch}")
    
    def load_checkpoint(self):
        if os.path.exists(self.model_path):
            print(f"✅ 체크포인트 발견! 학습 재개...")
            try:
                model = tf.keras.models.load_model(self.model_path, 
                                                  custom_objects={'weighted_mae_signal': weighted_mae_signal})
                with open(self.history_file, 'r') as f:
                    history = json.load(f)
                return model, history.get('epoch', 0)
            except:
                return None, 0
        return None, 0

# ========================================
# 커스텀 콜백 (신호구간 모니터링)
# ========================================
class SignalZoneCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_val, y_val_orig, processor):
        super().__init__()
        self.X_val = X_val
        self.y_val_orig = y_val_orig
        self.processor = processor
        
    def on_epoch_end(self, epoch, logs=None):
        # 신호구간 인덱스
        signal_idx = ((self.y_val_orig >= self.processor.SIGNAL_MIN) & 
                     (self.y_val_orig <= self.processor.SIGNAL_MAX))
        
        if np.sum(signal_idx) > 0:
            X_signal = self.X_val[signal_idx]
            y_signal_true = self.y_val_orig[signal_idx]
            
            # 예측
            preds = self.model.predict(X_signal, verbose=0)
            y_pred_scaled = preds[0].flatten()
            y_pred = self.processor.scaler_y.inverse_transform(
                y_pred_scaled.reshape(-1, 1)).flatten()
            
            mae = mean_absolute_error(y_signal_true, y_pred)
            accuracy = np.mean(np.abs(y_signal_true - y_pred) < 50)
            
            print(f"\n  🟠 신호구간(1651~1699) 성능: MAE={mae:.1f}, 정확도(±50)={accuracy*100:.1f}%")

# ========================================
# 커스텀 손실함수 (신호구간 강조)
# ========================================
def weighted_mae_signal(y_true, y_pred):
    """신호구간에 더 큰 가중치"""
    # MinMaxScaler 0-1 범위에서 신호구간 추정
    # 1651~1699를 0-1로 변환
    signal_min_scaled = 0.65  # 대략적 추정값
    signal_max_scaled = 0.75
    
    # 가중치
    weights = tf.ones_like(y_true)
    
    # 신호구간 마스크
    signal_mask = tf.logical_and(
        y_true >= signal_min_scaled,
        y_true <= signal_max_scaled
    )
    
    # 신호구간 5배 가중치
    weights = tf.where(signal_mask, 5.0, weights)
    
    # 위험구간 10배 가중치
    danger_mask = y_true > signal_max_scaled
    weights = tf.where(danger_mask, 10.0, weights)
    
    # Weighted MAE
    mae = tf.abs(y_true - y_pred)
    weighted_mae = mae * weights
    
    return tf.reduce_mean(weighted_mae)

# ========================================
# 학습 함수
# ========================================
def train_model(train_data, val_data, test_data, processor):
    """모델 학습"""
    X_train, y_train_scaled, y_train_orig, y_train_level, y_train_signal = train_data
    X_val, y_val_scaled, y_val_orig, y_val_level, y_val_signal = val_data
    X_test, y_test_scaled, y_test_orig, y_test_level, y_test_signal = test_data
    
    # 체크포인트
    cp_manager = CheckpointManager()
    model, start_epoch = cp_manager.load_checkpoint()
    
    # 모델 생성
    with strategy.scope():
        if model is None:
            print("새 모델 생성...")
            input_shape = (X_train.shape[1], X_train.shape[2])
            model = build_extremenet(input_shape)
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss={
                    'regression': weighted_mae_signal,
                    'classification': 'sparse_categorical_crossentropy',
                    'signal': 'binary_crossentropy'
                },
                loss_weights={
                    'regression': 0.6,
                    'classification': 0.25,
                    'signal': 0.15
                },
                metrics={
                    'regression': 'mae',
                    'classification': 'accuracy',
                    'signal': 'accuracy'
                }
            )
    
    print("\n" + "="*80)
    print("🔥 ExtremeNet V6.5 Final 학습 시작")
    print(f"🔧 Device: {DEVICE} | Batch: {BATCH_SIZE}")
    print(f"⏰ 시작 에포크: {start_epoch}")
    print("="*80)
    
    # 콜백
    callbacks = [
        ModelCheckpoint(
            'models_final/best_extremenet.keras',
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        ),
        EarlyStopping(
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1
        ),
        SignalZoneCallback(X_val, y_val_orig, processor),
        LambdaCallback(
            on_epoch_end=lambda epoch, logs: 
            cp_manager.save_checkpoint(model, start_epoch + epoch + 1, logs.get('val_loss', 0))
        )
    ]
    
    # 학습
    os.makedirs('models_final', exist_ok=True)
    
    history = model.fit(
        X_train,
        {
            'regression': y_train_scaled,
            'classification': y_train_level,
            'signal': y_train_signal
        },
        validation_data=(
            X_val,
            {
                'regression': y_val_scaled,
                'classification': y_val_level,
                'signal': y_val_signal
            }
        ),
        epochs=100,
        initial_epoch=start_epoch,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # 평가
    print("\n" + "="*80)
    print("📊 최종 평가")
    print("="*80)
    
    # 예측
    preds = model.predict(X_test, verbose=0)
    y_pred_scaled = preds[0].flatten()
    y_pred = processor.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    # 전체 메트릭
    mae = mean_absolute_error(y_test_orig, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
    r2 = r2_score(y_test_orig, y_pred)
    
    print(f"\n📈 전체 성능:")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²: {r2:.4f}")
    
    # 신호구간 평가
    signal_idx = ((y_test_orig >= processor.SIGNAL_MIN) & 
                 (y_test_orig <= processor.SIGNAL_MAX))
    if np.sum(signal_idx) > 0:
        signal_mae = mean_absolute_error(y_test_orig[signal_idx], y_pred[signal_idx])
        print(f"\n🟠 신호구간(1651~1699) 최종:")
        print(f"  샘플: {np.sum(signal_idx)}개")
        print(f"  MAE: {signal_mae:.2f}")
        print(f"  정확도(±50): {np.mean(np.abs(y_test_orig[signal_idx] - y_pred[signal_idx]) < 50)*100:.1f}%")
    
    # 위험구간 평가
    danger_idx = (y_test_orig >= processor.DANGER_MIN)
    if np.sum(danger_idx) > 0:
        danger_mae = mean_absolute_error(y_test_orig[danger_idx], y_pred[danger_idx])
        print(f"\n🔴 위험구간(1700+) 최종:")
        print(f"  샘플: {np.sum(danger_idx)}개")
        print(f"  MAE: {danger_mae:.2f}")
    
    # 모델 저장
    model.save('models_final/extremenet_v65_final.keras')
    print(f"\n💾 최종 모델: models_final/extremenet_v65_final.keras")
    
    return model, {'MAE': mae, 'RMSE': rmse, 'R2': r2}

# ========================================
# 실시간 예측 (상승/하락 추세 구분)
# ========================================
class Predictor:
    def __init__(self, model_path='models_final/extremenet_v65_final.keras'):
        print("\n🚀 예측기 로딩 중...")
        self.model = tf.keras.models.load_model(model_path,
                                               custom_objects={'weighted_mae_signal': weighted_mae_signal})
        with open('scalers_final/scaler_X.pkl', 'rb') as f:
            self.scaler_X = pickle.load(f)
        with open('scalers_final/scaler_y.pkl', 'rb') as f:
            self.scaler_y = pickle.load(f)
        
        # 구간 정의
        self.NORMAL_MAX = 1400
        self.CAUTION_MAX = 1650
        self.SIGNAL_MIN = 1651
        self.SIGNAL_MAX = 1699
        self.DANGER_MIN = 1700
        
        self.feature_columns = [
            'TOTALCNT', 'M14AM14B', 'M14AM14BSUM', 'M14AM10A', 'M14AM16',
            'RATIO', 'GOLDEN', 
            'SIGNAL_ZONE', 'PRE_SIGNAL', 'DANGER_ZONE',
            'PATTERN', 'TREND',
            'MA_5', 'MA_10', 'MA_20',
            'STD_5', 'STD_10', 'STD_20',
            'M14B_MA_5', 'M14B_MA_10', 'M14B_MA_20',
            'CHANGE_1', 'CHANGE_5', 'CHANGE_10',
            'M14B_CHANGE_1', 'M14B_CHANGE_5', 'M14B_CHANGE_10',
            'RISE_COUNT'
        ]
        
        print("✅ 예측기 준비 완료")
    
    def predict(self, data_100min):
        """100분 데이터로 10분 후 예측 (상승/하락 구분)"""
        
        # 현재값
        current_value = data_100min['TOTALCNT'].iloc[-1]
        
        # 추세 판단 (최근 20분)
        recent_20 = data_100min['TOTALCNT'].tail(20).values
        if len(recent_20) > 1:
            slope = np.polyfit(range(len(recent_20)), recent_20, 1)[0]
            if slope > 5:
                trend = "상승"
            elif slope < -5:
                trend = "하락"
            else:
                trend = "보합"
        else:
            trend = "보합"
        
        # 특성 추출
        X = data_100min[self.feature_columns].values
        X_scaled = self.scaler_X.transform(X.reshape(-1, len(self.feature_columns))).reshape(1, 100, len(self.feature_columns))
        
        # 예측
        preds = self.model.predict(X_scaled, verbose=0)
        y_pred_scaled = preds[0][0, 0]
        y_pred = self.scaler_y.inverse_transform([[y_pred_scaled]])[0, 0]
        
        # 변화량
        change = y_pred - current_value
        change_rate = (change / current_value) * 100
        
        # ⭐ 상승/하락에 따른 구간 판정
        if y_pred < self.NORMAL_MAX:
            level = "🟢 정상"
            message = "안정적 상태"
            
        elif y_pred < self.CAUTION_MAX:
            level = "🟡 주의"
            message = "물류량 증가 관찰 필요"
            
        elif y_pred < self.DANGER_MIN:
            # 신호구간(1651~1699)
            if change > 0:  # 상승 중
                level = "🟠 주의관찰 (상승)"
                message = "⚠️ 1700+ 급증 신호! 상승 추세 - 사전 대응 필요"
            else:  # 하락 중
                level = "🟡 주의관찰 (하락)"
                message = "📉 하락 추세 - 안정화 중"
                
        else:  # 1700+
            if change > 0:  # 상승 중
                level = "🔴 위험 (상승)"
                message = "🚨 즉시 대응! 물류 급증 진행 중"
            else:  # 하락 중
                level = "🟠 위험 (하락)"
                message = "📉 피크 지남 - 하락 추세"
        
        return {
            'current': float(current_value),
            'prediction': float(y_pred),
            'change': float(change),
            'change_rate': float(change_rate),
            'trend': trend,
            'level': level,
            'message': message,
            'signal_prob': float(preds[2][0, 0]),
            'is_dangerous': (y_pred >= self.SIGNAL_MIN) and (change > 0)  # 상승 중일 때만 위험
        }

# ========================================
# 메인
# ========================================
def main():
    # 데이터 경로
    data_path = None
    for path in ['data/20240201_TO_202507281705.csv', 
                 '20240201_TO_202507281705.csv']:
        if os.path.exists(path):
            data_path = path
            break
    
    if not data_path:
        print("❌ 데이터 파일 없음!")
        return None, None
    
    # 데이터 처리
    processor = DataProcessor()
    df = processor.load_data(data_path)
    df_features = processor.create_features(df)
    
    # 시퀀스 생성
    train_data, val_data, test_data = processor.create_sequences(df_features)
    
    # 학습
    model, results = train_model(train_data, val_data, test_data, processor)
    
    print("\n✅ 완료!")
    print("🔥 ExtremeNet V6.5 Final 학습 완료!")
    
    return model, results

if __name__ == "__main__":
    model, results = main()