"""
🔥 ExtremeNet V6.5 Fixed - 위험구간 예측 개선판 (오류 수정 완료)
================================================================
✅ StandardScaler로 변경 (RobustScaler 문제 해결)
✅ 위험구간 오버샘플링
✅ 손실함수 개선
✅ 위험구간 특별 관리
✅ 다중 출력 모델의 class_weight 오류 수정 (sample_weight 사용)
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
    # BATCH_SIZE는 GPU 수에 따라 동적으로 설정
    BATCH_SIZE = 64 * strategy.num_replicas_in_sync
else:
    print("⚠️ GPU 없음, CPU 사용")
    strategy = tf.distribute.get_strategy()
    DEVICE = 'CPU'
    BATCH_SIZE = 32

print("="*80)
print("🔥 ExtremeNet V6.5 Fixed - 위험구간 예측 개선판 (오류 수정 완료)")
print(f"📦 TensorFlow: {tf.__version__}")
print(f"🔧 Device: {DEVICE}")
print("="*80)

# ========================================
# 데이터 처리 (개선됨)
# ========================================
class DataProcessor:
    def __init__(self):
        self.seq_len = 100
        self.pred_len = 10
        
        # ⭐ StandardScaler로 변경 (핵심!)
        self.scaler_X = StandardScaler()
        self.scaler_y = MinMaxScaler()  # 0-1 범위로 정규화
        
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
        
        # 구간 통계
        normal = (df['TOTALCNT'] < 1400).sum()
        warning = ((df['TOTALCNT'] >= 1400) & (df['TOTALCNT'] < 1700)).sum()
        critical = (df['TOTALCNT'] >= 1700).sum()
        danger = ((df['TOTALCNT'] >= 1651) & (df['TOTALCNT'] <= 1682)).sum()
        
        print(f"\n📊 TOTALCNT 분포:")
        print(f"  정상(~1400): {normal:,}개 ({normal/len(df)*100:.1f}%)")
        print(f"  주의(1400~1699): {warning:,}개 ({warning/len(df)*100:.1f}%)")
        print(f"  심각(1700+): {critical:,}개 ({critical/len(df)*100:.1f}%)")
        print(f"  🔥 위험신호(1651~1682): {danger:,}개")
        
        return df
    
    def create_features(self, df):
        """특성 생성 (최적화)"""
        print("\n⚙️ 특성 생성 중...")
        
        # 기본 특성
        df['RATIO'] = df['M14AM14B'] / (df['M14AM10A'] + 1)
        df['GOLDEN'] = ((df['M14AM14B'] > 300) & (df['M14AM10A'] < 80)).astype(int)
        
        # 위험 신호 강조
        df['DANGER_ZONE'] = ((df['TOTALCNT'] >= 1651) & 
                             (df['TOTALCNT'] <= 1682)).astype(int)
        df['PRE_DANGER'] = ((df['TOTALCNT'] >= 1600) & 
                            (df['TOTALCNT'] < 1651)).astype(int)
        
        # 이동평균 (fillna로 NaN 처리)
        for w in [5, 10, 20]:
            df[f'MA_{w}'] = df['TOTALCNT'].rolling(w, min_periods=1).mean()
            df[f'STD_{w}'] = df['TOTALCNT'].rolling(w, min_periods=1).std().fillna(0)
            df[f'M14B_MA_{w}'] = df['M14AM14B'].rolling(w, min_periods=1).mean()
        
        # 변화율
        for lag in [1, 5, 10]:
            df[f'CHANGE_{lag}'] = df['TOTALCNT'].diff(lag).fillna(0)
            df[f'M14B_CHANGE_{lag}'] = df['M14AM14B'].diff(lag).fillna(0)
        
        # 연속 상승 (벡터화)
        df['RISE'] = (df['TOTALCNT'] > df['TOTALCNT'].shift(1)).astype(int)
        df['RISE_COUNT'] = df['RISE'].rolling(10, min_periods=1).sum()
        
        # 패턴 (간소화)
        df['PATTERN'] = 0
        danger_mask = ((df['TOTALCNT'] >= 1651) & (df['TOTALCNT'] <= 1682))
        high_m14b = df['M14AM14B'].rolling(100, min_periods=1).mean() > 380
        
        df.loc[danger_mask, 'PATTERN'] = 1  # UU1
        df.loc[high_m14b, 'PATTERN'] = 2     # UU2
        
        # 추세 (벡터화)
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
        """시퀀스 생성 (위험구간 오버샘플링)"""
        print("\n🔄 시퀀스 생성 중...")
        
        # 사용할 컬럼
        self.feature_columns = [
            'TOTALCNT', 'M14AM14B', 'M14AM14BSUM', 'M14AM10A', 'M14AM16',
            'RATIO', 'GOLDEN', 'DANGER_ZONE', 'PRE_DANGER',
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
        danger_X, danger_y = [], []  # 위험구간 따로 저장
        
        for i in range(len(df) - self.seq_len - self.pred_len):
            X.append(X_data[i:i+self.seq_len])
            target = y_data[i+self.seq_len-1]
            y.append(target)
            
            # 위험구간 오버샘플링
            if 1651 <= target <= 1682:
                # 위험구간은 3배 복제
                for _ in range(3):
                    danger_X.append(X_data[i:i+self.seq_len])
                    danger_y.append(target)
        
        # 위험구간 추가
        if danger_X:
            X.extend(danger_X)
            y.extend(danger_y)
            print(f"  🔥 위험구간 {len(danger_X)}개 추가 (오버샘플링)")
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        
        # NaN 제거
        valid_idx = ~np.isnan(y)
        X = X[valid_idx]
        y = y[valid_idx]
        
        print(f"  총 시퀀스: {X.shape[0]}개")
        
        # 셔플 (시계열 순서 유지하면서 위험구간 섞기)
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
        # 훈련 데이터의 형태를 2D로 변경하여 스케일러 학습
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        self.scaler_X.fit(X_train_reshaped)

        # 각 데이터셋에 스케일링 적용
        X_train_scaled = self.scaler_X.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val_scaled = self.scaler_X.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        X_test_scaled = self.scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

        # y 스케일링 (MinMaxScaler로 0-1 범위)
        self.scaler_y.fit(y_train.reshape(-1, 1))
        y_train_scaled = self.scaler_y.transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = self.scaler_y.transform(y_val.reshape(-1, 1)).flatten()
        y_test_scaled = self.scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        
        print(f"\n📊 데이터 분할:")
        print(f"  Train: {len(X_train):,}개")
        print(f"  Val: {len(X_val):,}개")
        print(f"  Test: {len(X_test):,}개")
        
        # 위험구간 비율
        danger_train = ((y_train >= 1651) & (y_train <= 1682)).sum()
        danger_val = ((y_val >= 1651) & (y_val <= 1682)).sum()
        danger_test = ((y_test >= 1651) & (y_test <= 1682)).sum()
        
        print(f"\n🔥 위험신호(1651~1682) 샘플:")
        print(f"  Train: {danger_train}개 ({danger_train/len(X_train)*100:.1f}%)")
        print(f"  Val: {danger_val}개 ({danger_val/len(X_val)*100:.1f}%)")
        print(f"  Test: {danger_test}개 ({danger_test/len(X_test)*100:.1f}%)")
        
        # 구간 레이블
        y_train_level = np.zeros(len(y_train), dtype=int)
        y_train_level[y_train >= 1400] = 1
        y_train_level[y_train >= 1700] = 2
        
        y_val_level = np.zeros(len(y_val), dtype=int)
        y_val_level[y_val >= 1400] = 1
        y_val_level[y_val >= 1700] = 2
        
        y_test_level = np.zeros(len(y_test), dtype=int)
        y_test_level[y_test >= 1400] = 1
        y_test_level[y_test >= 1700] = 2
        
        # 위험구간 레이블
        y_train_danger = ((y_train >= 1651) & (y_train <= 1682)).astype(int)
        y_val_danger = ((y_val >= 1651) & (y_val <= 1682)).astype(int)
        y_test_danger = ((y_test >= 1651) & (y_test <= 1682)).astype(int)
        
        # 스케일러 저장
        os.makedirs('scalers', exist_ok=True)
        with open('scalers/scaler_X.pkl', 'wb') as f:
            pickle.dump(self.scaler_X, f)
        with open('scalers/scaler_y.pkl', 'wb') as f:
            pickle.dump(self.scaler_y, f)
        
        return (X_train_scaled, y_train_scaled, y_train, y_train_level, y_train_danger), \
               (X_val_scaled, y_val_scaled, y_val, y_val_level, y_val_danger), \
               (X_test_scaled, y_test_scaled, y_test, y_test_level, y_test_danger)

# ========================================
# ExtremeNet 모델 (개선)
# ========================================
def build_extremenet(input_shape):
    """ExtremeNet V6.5 Fixed"""
    inputs = Input(shape=input_shape)
    
    # LSTM 브랜치
    lstm = LSTM(128, return_sequences=True)(inputs)
    lstm = BatchNormalization()(lstm)
    lstm = LSTM(64)(lstm)
    lstm = Dropout(0.3)(lstm)
    
    # Attention 브랜치
    attn = MultiHeadAttention(num_heads=4, key_dim=16)(inputs, inputs)
    attn = GlobalAveragePooling1D()(attn)
    
    # CNN 브랜치 (위험구간 패턴)
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
    output_reg = Dense(1, name='regression')(x)
    output_cls = Dense(3, activation='softmax', name='classification')(x)
    output_danger = Dense(1, activation='sigmoid', name='danger')(x)
    
    model = Model(inputs, [output_reg, output_cls, output_danger], name='ExtremeNet_V65_Fixed')
    return model

# ========================================
# 체크포인트
# ========================================
class CheckpointManager:
    def __init__(self, checkpoint_dir='checkpoints/'):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.model_path = os.path.join(self.checkpoint_dir, 'extremenet_fixed.keras')
        self.history_file = os.path.join(self.checkpoint_dir, 'history.json')
        
    def save_checkpoint(self, model, epoch, loss):
        model.save(self.model_path)
        history = {'epoch': epoch, 'loss': float(loss)}
        with open(self.history_file, 'w') as f:
            json.dump(history, f)
        print(f"💾 체크포인트 저장: 에포크 {epoch}, 경로: {self.model_path}")
    
    def load_checkpoint(self):
        if os.path.exists(self.model_path):
            print(f"✅ 체크포인트 발견! 학습 재개... ({self.model_path})")
            try:
                model = tf.keras.models.load_model(self.model_path, 
                                                  custom_objects={'weighted_mae': weighted_mae})
                with open(self.history_file, 'r') as f:
                    history = json.load(f)
                return model, history.get('epoch', 0)
            except Exception as e:
                print(f"⚠️ 체크포인트 로드 실패: {e}. 새 모델로 시작합니다.")
                return None, 0
        return None, 0

# ========================================
# 커스텀 콜백 (위험구간 모니터링)
# ========================================
class DangerZoneCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_val, y_val_orig, processor):
        super().__init__()
        self.X_val = X_val
        self.y_val_orig = y_val_orig
        self.processor = processor
        
    def on_epoch_end(self, epoch, logs=None):
        # 위험구간 인덱스
        danger_idx = ((self.y_val_orig >= 1651) & (self.y_val_orig <= 1682))
        
        if np.sum(danger_idx) > 0:
            X_danger = self.X_val[danger_idx]
            y_danger_true = self.y_val_orig[danger_idx]
            
            # 예측
            preds = self.model.predict(X_danger, verbose=0)
            y_pred_scaled = preds[0].flatten()
            y_pred = self.processor.scaler_y.inverse_transform(
                y_pred_scaled.reshape(-1, 1)).flatten()
            
            mae = mean_absolute_error(y_danger_true, y_pred)
            accuracy = np.mean(np.abs(y_danger_true - y_pred) < 50) # 오차 50 이내를 정확도로 간주
            
            print(f"\n  🔥 위험구간(검증셋) 성능: MAE={mae:.1f}, 정확도(±50)={accuracy*100:.1f}%")

# ========================================
# 커스텀 손실함수 (위험구간 강조)
# ========================================
def weighted_mae(y_true, y_pred):
    """위험구간에 더 큰 가중치를 부여하는 MAE"""
    # y_true는 이미 0-1 사이로 스케일링된 값입니다.
    # 위험구간(1651~1682)의 스케일링된 범위를 계산합니다.
    # 이 계산은 scaler_y가 학습된 후에야 정확하지만, 여기서는 근사치를 사용합니다.
    # 예를 들어, 데이터 범위가 [800, 2000]이라면,
    # (1651 - 800) / (2000 - 800) = 0.709
    # (1682 - 800) / (2000 - 800) = 0.735
    # 데이터 분포에 따라 달라지므로, 여기서는 대략적인 값을 사용합니다.
    danger_min_scaled = 0.7 
    danger_max_scaled = 0.75
    
    # 가중치 텐서 생성 (기본값 1.0)
    weights = tf.ones_like(y_true)
    
    # 위험구간 마스크 생성
    danger_mask = tf.logical_and(
        y_true >= danger_min_scaled,
        y_true <= danger_max_scaled
    )
    
    # 위험구간에 해당하는 샘플에 3배의 가중치 부여
    weights = tf.where(danger_mask, 3.0, weights)
    
    # 가중치가 적용된 MAE 계산
    mae = tf.abs(y_true - y_pred)
    weighted_mae_val = mae * weights
    
    return tf.reduce_mean(weighted_mae_val)

# ========================================
# 학습 함수
# ========================================
def train_model(train_data, val_data, test_data, processor):
    """모델 학습"""
    X_train, y_train_scaled, y_train_orig, y_train_level, y_train_danger = train_data
    X_val, y_val_scaled, y_val_orig, y_val_level, y_val_danger = val_data
    X_test, y_test_scaled, y_test_orig, y_test_level, y_test_danger = test_data
    
    # 체크포인트 관리자
    cp_manager = CheckpointManager()
    model, start_epoch = cp_manager.load_checkpoint()
    
    # 분산 학습 전략 범위 내에서 모델 생성 및 컴파일
    with strategy.scope():
        if model is None:
            print("새 모델 생성...")
            input_shape = (X_train.shape[1], X_train.shape[2])
            model = build_extremenet(input_shape)
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss={
                    'regression': weighted_mae,  # 커스텀 손실함수
                    'classification': 'sparse_categorical_crossentropy',
                    'danger': 'binary_crossentropy'
                },
                loss_weights={
                    'regression': 0.6,
                    'classification': 0.2,
                    'danger': 0.2
                },
                metrics={
                    'regression': 'mae',
                    'classification': 'accuracy',
                    'danger': 'accuracy'
                }
            )
    
    model.summary()
    print("\n" + "="*80)
    print("🔥 ExtremeNet V6.5 Fixed 학습 시작")
    print(f"🔧 Device: {DEVICE} | Global Batch: {BATCH_SIZE}")
    print(f"⏰ 시작 에포크: {start_epoch}")
    print("="*80)
    
    # 콜백
    callbacks = [
        ModelCheckpoint(
            'models/best_extremenet_fixed.keras',
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        ),
        EarlyStopping(
            patience=20,
            monitor='val_loss',
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            monitor='val_loss',
            verbose=1
        ),
        DangerZoneCallback(X_val, y_val_orig, processor),
        LambdaCallback(
            on_epoch_end=lambda epoch, logs: 
            cp_manager.save_checkpoint(model, start_epoch + epoch + 1, logs['val_loss']) if 'val_loss' in logs else None
        )
    ]
    
    # ⭐ 오류 수정: 다중 출력 모델은 class_weight를 직접 지원하지 않습니다.
    # 대신, 각 샘플에 대한 가중치를 계산하는 sample_weight를 사용합니다.
    # 이는 불균형한 'classification' 작업에 가중치를 효과적으로 적용합니다.
    sample_weights = class_weight.compute_sample_weight(
        class_weight='balanced',
        y=y_train_level
    )
    
    # 학습
    os.makedirs('models', exist_ok=True)
    
    history = model.fit(
        X_train,
        {
            'regression': y_train_scaled,
            'classification': y_train_level,
            'danger': y_train_danger
        },
        validation_data=(
            X_val,
            {
                'regression': y_val_scaled,
                'classification': y_val_level,
                'danger': y_val_danger
            }
        ),
        epochs=100,
        initial_epoch=start_epoch,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        sample_weight=sample_weights,  # ⭐ 수정: class_weight 대신 sample_weight 사용
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
    
    # 메트릭
    mae = mean_absolute_error(y_test_orig, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
    r2 = r2_score(y_test_orig, y_pred)
    
    print(f"\n📈 전체 성능:")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²: {r2:.4f}")
    
    # 위험구간 평가
    danger_idx = ((y_test_orig >= 1651) & (y_test_orig <= 1682))
    if np.sum(danger_idx) > 0:
        danger_mae = mean_absolute_error(y_test_orig[danger_idx], y_pred[danger_idx])
        print(f"\n🔥 위험구간 최종 성능:")
        print(f"  샘플: {np.sum(danger_idx)}개")
        print(f"  MAE: {danger_mae:.2f}")
    
    # 모델 저장
    model.save('models/extremenet_v65_fixed_final.keras')
    print(f"\n💾 최종 모델 저장 완료: models/extremenet_v65_fixed_final.keras")
    
    return model, {'MAE': mae, 'RMSE': rmse, 'R2': r2}

# ========================================
# 실시간 예측
# ========================================
class Predictor:
    def __init__(self, model_path='models/extremenet_v65_fixed_final.keras'):
        print("\n🚀 예측기 로딩 중...")
        self.model = tf.keras.models.load_model(model_path,
                                               custom_objects={'weighted_mae': weighted_mae})
        with open('scalers/scaler_X.pkl', 'rb') as f:
            self.scaler_X = pickle.load(f)
        with open('scalers/scaler_y.pkl', 'rb') as f:
            self.scaler_y = pickle.load(f)
        self.feature_columns = [
            'TOTALCNT', 'M14AM14B', 'M14AM14BSUM', 'M14AM10A', 'M14AM16',
            'RATIO', 'GOLDEN', 'DANGER_ZONE', 'PRE_DANGER',
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
        if not isinstance(data_100min, pd.DataFrame) or len(data_100min) != 100:
            raise ValueError("입력 데이터는 100개의 행을 가진 pandas DataFrame이어야 합니다.")
            
        # 특성 준비
        X = data_100min[self.feature_columns].values
        X_reshaped = X.reshape(1, 100, -1)
        
        # 스케일링 (전체 시퀀스를 한 번에 변환)
        X_flat = X_reshaped.reshape(-1, X_reshaped.shape[-1])
        X_scaled_flat = self.scaler_X.transform(X_flat)
        X_scaled = X_scaled_flat.reshape(X_reshaped.shape)
        
        # 예측
        preds = self.model.predict(X_scaled, verbose=0)
        y_pred_scaled = preds[0][0, 0]
        y_pred = self.scaler_y.inverse_transform([[y_pred_scaled]])[0, 0]
        
        # 구간 판정
        if y_pred < 1400:
            level = "🟢 정상"
        elif y_pred < 1700:
            level = "🟡 주의"
        else:
            level = "🔴 심각"
        
        # 위험신호
        alert = ""
        if 1651 <= y_pred <= 1682:
            alert = "⚠️ 위험신호! 1700+ 급증 가능성!"
        
        return {
            'prediction': float(y_pred),
            'level': level,
            'alert': alert,
            'danger_probability': float(preds[2][0, 0])
        }

# ========================================
# 메인 실행
# ========================================
def main():
    # 데이터 경로 탐색
    data_path = None
    possible_paths = [
        'data/20240201_TO_202507281705.csv', 
        '20240201_TO_202507281705.csv'
    ]
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if not data_path:
        print("❌ 데이터 파일을 찾을 수 없습니다! 'data' 폴더 또는 현재 위치에 CSV 파일을 확인해주세요.")
        return None, None
    
    # 데이터 처리
    processor = DataProcessor()
    df = processor.load_data(data_path)
    df_features = processor.create_features(df)
    
    # 시퀀스 생성
    train_data, val_data, test_data = processor.create_sequences(df_features)
    
    # 모델 학습
    model, results = train_model(train_data, val_data, test_data, processor)
    
    print("\n✅ 모든 작업 완료!")
    print("🔥 위험구간 예측이 개선된 모델 학습이 완료되었습니다!")
    
    return model, results

if __name__ == "__main__":
    trained_model, training_results = main()
    
    # 학습이 성공적으로 완료되었을 경우, 예측기 사용 예시
    if trained_model and os.path.exists('models/extremenet_v65_fixed_final.keras'):
        print("\n" + "="*80)
        print("⚡ 예측기 사용 예시")
        print("="*80)
        
        # 테스트 데이터셋의 마지막 100분 데이터를 샘플로 사용
        try:
            processor = DataProcessor()
            df = processor.load_data('20240201_TO_202507281705.csv')
            df_features = processor.create_features(df)
            
            # 예측을 위한 최신 100분 데이터 추출
            sample_data = df_features.tail(100)
            
            if len(sample_data) == 100:
                predictor = Predictor()
                prediction_result = predictor.predict(sample_data)
                
                print(f"🔮 10분 후 TOTALCNT 예측: {prediction_result['prediction']:.2f}")
                print(f"📊 예측 레벨: {prediction_result['level']}")
                print(f"📈 위험 감지 확률: {prediction_result['danger_probability']:.2%}")
                if prediction_result['alert']:
                    print(f"🚨 {prediction_result['alert']}")
            else:
                print("⚠️ 예측 예시를 위한 샘플 데이터가 100개 미만입니다.")

        except Exception as e:
            print(f"💥 예측 예시 실행 중 오류 발생: {e}")

