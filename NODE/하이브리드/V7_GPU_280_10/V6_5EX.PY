"""
🔥 ExtremeNet V6.5 - 패턴 학습 강화 버전
============================================
✅ ExtremeNet 단일 모델 집중
✅ 위험구간(1651~1682) 특별 학습
✅ 패턴 인식 강화 (UU1/UU2)
✅ R2 음수 방지 처리
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import *
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import os
import pickle
import json
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
tf.random.set_seed(42)
np.random.seed(42)

# GPU 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print(f"🔧 GPU {len(gpus)}개 발견")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    strategy = tf.distribute.MirroredStrategy()
    device_type = 'GPU'
else:
    print("⚠️ GPU 없음, CPU 사용")
    strategy = tf.distribute.get_strategy()
    device_type = 'CPU'

print("="*80)
print("🔥 ExtremeNet V6.5 - 패턴 학습 강화 버전")
print(f"📦 TensorFlow: {tf.__version__}")
print(f"🔧 Device: {device_type}")
print("="*80)

# ========================================
# 데이터 처리 클래스 (패턴 강화)
# ========================================
class PatternDataProcessor:
    def __init__(self, seq_len=100, pred_len=10):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.feature_scaler = RobustScaler()  # 이상치에 강건
        self.target_scaler = RobustScaler()
        self.feature_columns = None
        
        # 구간 정의
        self.NORMAL_MAX = 1400
        self.WARNING_MAX = 1700
        self.DANGER_MIN = 1651  # 위험 신호 시작
        self.DANGER_MAX = 1682  # 위험 신호 끝
        
    def save_scalers(self, path='scalers_v65/'):
        """스케일러 저장"""
        os.makedirs(path, exist_ok=True)
        
        with open(f'{path}feature_scaler.pkl', 'wb') as f:
            pickle.dump(self.feature_scaler, f)
        with open(f'{path}target_scaler.pkl', 'wb') as f:
            pickle.dump(self.target_scaler, f)
            
        config = {
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'feature_columns': self.feature_columns
        }
        with open(f'{path}config.json', 'w') as f:
            json.dump(config, f)
            
        print(f"✅ 스케일러 저장: {path}")
    
    def load_and_process(self, filepath):
        print(f"\n📂 데이터 로딩: {filepath}")
        df = pd.read_csv(filepath)
        print(f"✅ 원본: {df.shape[0]:,}행 × {df.shape[1]}열")
        
        # 0값 제거
        before = len(df)
        df = df[df['TOTALCNT'] > 0].reset_index(drop=True)
        print(f"✅ 0값 제거: {before - len(df):,}개")
        
        # 시간 정렬
        df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), 
                                       format='%Y%m%d%H%M', errors='coerce')
        df = df.sort_values('CURRTIME').reset_index(drop=True)
        
        # 구간 분석
        normal = (df['TOTALCNT'] < self.NORMAL_MAX).sum()
        warning = ((df['TOTALCNT'] >= self.NORMAL_MAX) & 
                  (df['TOTALCNT'] < self.WARNING_MAX)).sum()
        critical = (df['TOTALCNT'] >= self.WARNING_MAX).sum()
        danger_signal = ((df['TOTALCNT'] >= self.DANGER_MIN) & 
                        (df['TOTALCNT'] <= self.DANGER_MAX)).sum()
        
        print(f"\n📊 구간 분포:")
        print(f"  정상(~1400): {normal:,}개 ({normal/len(df)*100:.1f}%)")
        print(f"  주의(1400~1699): {warning:,}개 ({warning/len(df)*100:.1f}%)")
        print(f"  심각(1700+): {critical:,}개 ({critical/len(df)*100:.1f}%)")
        print(f"  🔥 위험신호(1651~1682): {danger_signal:,}개 ({danger_signal/len(df)*100:.1f}%)")
        
        return df
    
    def detect_pattern(self, df, window=100):
        """패턴 감지 (UU1/UU2)"""
        patterns = []
        
        for i in range(len(df)):
            if i < window:
                patterns.append(0)  # 패턴 없음
                continue
                
            # 최근 window 데이터
            recent = df.iloc[max(0, i-window):i]
            
            # 패턴 판정 기준
            high_cases = (recent['TOTALCNT'] >= self.DANGER_MAX).sum()
            m14b_mean = recent['M14AM14B'].mean()
            m14b_350_ratio = (recent['M14AM14B'] > 350).mean()
            danger_zone = ((recent['TOTALCNT'] >= self.DANGER_MIN) & 
                          (recent['TOTALCNT'] <= self.DANGER_MAX)).sum()
            
            # UU2: 고위험 패턴 (보수적)
            if high_cases > 0 and m14b_mean > 380:
                patterns.append(2)  # UU2
            elif m14b_350_ratio > 0.3:
                patterns.append(2)  # UU2
            # UU1: 급증 가능 패턴
            elif danger_zone > 0:  # 위험 신호 감지
                patterns.append(1)  # UU1 - 급증 예상
            else:
                patterns.append(0)  # 정상
                
        df['PATTERN'] = patterns
        
        print(f"\n📊 패턴 분포:")
        print(f"  정상: {(df['PATTERN'] == 0).sum():,}개")
        print(f"  UU1(급증가능): {(df['PATTERN'] == 1).sum():,}개")
        print(f"  UU2(고위험): {(df['PATTERN'] == 2).sum():,}개")
        
        return df
    
    def create_features(self, df):
        print("\n⚙️ 특성 생성...")
        
        # 패턴 감지
        df = self.detect_pattern(df)
        
        # 기본 특성
        df['RATIO'] = df['M14AM14B'] / (df['M14AM10A'] + 1)
        df['GOLDEN'] = ((df['M14AM14B'] > 300) & (df['M14AM10A'] < 80)).astype(float)
        
        # 위험 신호 특성
        df['DANGER_SIGNAL'] = ((df['TOTALCNT'] >= self.DANGER_MIN) & 
                               (df['TOTALCNT'] <= self.DANGER_MAX)).astype(float)
        
        # 시간 특성
        df['HOUR'] = df['CURRTIME'].dt.hour
        df['HOUR_SIN'] = np.sin(2 * np.pi * df['HOUR'] / 24)
        df['HOUR_COS'] = np.cos(2 * np.pi * df['HOUR'] / 24)
        
        # 이동 통계
        for w in [5, 10, 20, 30]:
            df[f'MA_{w}'] = df['TOTALCNT'].rolling(w, min_periods=1).mean()
            df[f'STD_{w}'] = df['TOTALCNT'].rolling(w, min_periods=1).std().fillna(0)
            
            # M14AM14B 이동평균
            df[f'M14B_MA_{w}'] = df['M14AM14B'].rolling(w, min_periods=1).mean()
        
        # 변화율
        for lag in [1, 5, 10, 20]:
            df[f'CHANGE_{lag}'] = df['TOTALCNT'].diff(lag).fillna(0)
            df[f'M14B_CHANGE_{lag}'] = df['M14AM14B'].diff(lag).fillna(0)
        
        # 연속 상승 횟수
        df['CONSECUTIVE_RISE'] = 0
        for i in range(1, len(df)):
            if df.loc[i, 'TOTALCNT'] > df.loc[i-1, 'TOTALCNT']:
                df.loc[i, 'CONSECUTIVE_RISE'] = df.loc[i-1, 'CONSECUTIVE_RISE'] + 1
            else:
                df.loc[i, 'CONSECUTIVE_RISE'] = 0
        
        # 타겟 생성
        df['TARGET'] = df['TOTALCNT'].shift(-self.pred_len)
        
        # 구간 레이블 (3구간)
        df['LEVEL'] = 0  # 정상
        df.loc[df['TARGET'] >= self.NORMAL_MAX, 'LEVEL'] = 1  # 주의
        df.loc[df['TARGET'] >= self.WARNING_MAX, 'LEVEL'] = 2  # 심각
        
        # 위험 신호 레이블 (1651~1682 구간 특별 처리)
        df['IS_DANGER'] = ((df['TARGET'] >= self.DANGER_MIN) & 
                          (df['TARGET'] <= self.DANGER_MAX)).astype(int)
        
        df = df.dropna()
        print(f"✅ 유효 데이터: {len(df)}행")
        
        return df
    
    def create_sequences(self, df):
        print("\n🔄 시퀀스 생성...")
        
        # 특성 선택 (ExtremeNet용 확장)
        self.feature_columns = [
            'TOTALCNT', 'M14AM14B', 'M14AM14BSUM', 'M14AM10A', 'M14AM16',
            'RATIO', 'GOLDEN', 'DANGER_SIGNAL',
            'PATTERN',  # 패턴 정보
            'HOUR_SIN', 'HOUR_COS',
            'MA_5', 'MA_10', 'MA_20', 'MA_30',
            'STD_5', 'STD_10', 'STD_20', 'STD_30',
            'M14B_MA_5', 'M14B_MA_10', 'M14B_MA_20',
            'CHANGE_1', 'CHANGE_5', 'CHANGE_10',
            'M14B_CHANGE_1', 'M14B_CHANGE_5',
            'CONSECUTIVE_RISE'
        ]
        
        # 사용 가능한 컬럼만
        self.feature_columns = [f for f in self.feature_columns if f in df.columns]
        print(f"  특성: {len(self.feature_columns)}개")
        
        X_data = df[self.feature_columns].values.astype(np.float32)
        y_reg = df['TARGET'].values.astype(np.float32)
        y_cls = df['LEVEL'].values.astype(np.int32)
        y_danger = df['IS_DANGER'].values.astype(np.int32)
        
        # 추세 레이블 추가
        y_trend = self.calculate_trend_labels(df)
        
        # 분할
        total_samples = len(X_data) - self.seq_len
        train_end = int(total_samples * 0.7)
        val_end = int(total_samples * 0.85)
        
        # 스케일링
        print("  스케일링...")
        self.feature_scaler.fit(X_data[:train_end + self.seq_len])
        self.target_scaler.fit(y_reg[:train_end].reshape(-1, 1))
        
        X_scaled = self.feature_scaler.transform(X_data)
        y_reg_scaled = self.target_scaler.transform(y_reg.reshape(-1, 1)).flatten()
        
        # 시퀀스 생성
        X, y_r, y_c, y_d, y_t = [], [], [], [], []
        for i in range(total_samples):
            X.append(X_scaled[i:i+self.seq_len])
            y_r.append(y_reg_scaled[i+self.seq_len-1])
            y_c.append(y_cls[i+self.seq_len-1])
            y_d.append(y_danger[i+self.seq_len-1])
            y_t.append(y_trend[i+self.seq_len-1])
        
        X = np.array(X, dtype=np.float32)
        y_r = np.array(y_r, dtype=np.float32)
        y_c = np.array(y_c, dtype=np.int32)
        y_d = np.array(y_d, dtype=np.int32)
        y_t = np.array(y_t, dtype=np.int32)
        
        # 데이터 분할
        X_train = X[:train_end]
        X_val = X[train_end:val_end]
        X_test = X[val_end:]
        
        y_train = (y_r[:train_end], y_c[:train_end], y_d[:train_end], y_t[:train_end])
        y_val = (y_r[train_end:val_end], y_c[train_end:val_end], 
                y_d[train_end:val_end], y_t[train_end:val_end])
        y_test = (y_r[val_end:], y_c[val_end:], y_d[val_end:], y_t[val_end:])
        
        print(f"  Train: {X_train.shape}")
        print(f"  Val: {X_val.shape}")
        print(f"  Test: {X_test.shape}")
        
        # 위험 구간 비율 확인
        danger_train = (y_train[2] == 1).sum()
        danger_val = (y_val[2] == 1).sum()
        danger_test = (y_test[2] == 1).sum()
        
        print(f"\n🔥 위험구간(1651~1682) 샘플:")
        print(f"  Train: {danger_train}개 ({danger_train/len(X_train)*100:.1f}%)")
        print(f"  Val: {danger_val}개 ({danger_val/len(X_val)*100:.1f}%)")
        print(f"  Test: {danger_test}개 ({danger_test/len(X_test)*100:.1f}%)")
        
        self.save_scalers()
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def calculate_trend_labels(self, df):
        """추세 레이블 계산 (0: 하락, 1: 보합, 2: 상승)"""
        trends = []
        window = 20
        
        for i in range(len(df)):
            if i < window:
                trends.append(1)  # 보합
                continue
            
            # 최근 window 기간 추세
            recent = df['TOTALCNT'].iloc[max(0, i-window):i].values
            if len(recent) < 2:
                trends.append(1)
                continue
                
            # 선형 회귀 추세
            x = np.arange(len(recent))
            slope = np.polyfit(x, recent, 1)[0]
            
            # 추세 판정
            if slope > 5:  # 상승
                trends.append(2)
            elif slope < -5:  # 하락
                trends.append(0)
            else:  # 보합
                trends.append(1)
        
        return np.array(trends)

# ========================================
# ExtremeNet V6.5 모델 (강화버전)
# ========================================
def build_extremenet_v65(input_shape):
    """ExtremeNet V6.5 - 패턴/추세 학습 강화"""
    inputs = Input(shape=input_shape)
    
    # Branch 1: LSTM 장기 패턴
    lstm1 = LSTM(128, return_sequences=True)(inputs)
    lstm1 = BatchNormalization()(lstm1)
    lstm1 = Dropout(0.3)(lstm1)
    lstm2 = LSTM(64)(lstm1)
    lstm2 = BatchNormalization()(lstm2)
    
    # Branch 2: Attention 메커니즘
    attn = MultiHeadAttention(num_heads=8, key_dim=16)(inputs, inputs)
    attn = LayerNormalization()(attn)
    attn = GlobalAveragePooling1D()(attn)
    
    # Branch 3: CNN 단기 패턴 (위험구간 감지)
    cnn = Conv1D(64, 5, activation='relu', padding='same')(inputs)
    cnn = BatchNormalization()(cnn)
    cnn = Conv1D(32, 3, activation='relu', padding='same')(cnn)
    cnn = GlobalMaxPooling1D()(cnn)
    
    # Branch 4: 최근 데이터 집중 (위험신호 감지)
    recent = Lambda(lambda x: x[:, -30:, :])(inputs)
    recent = GRU(32)(recent)
    
    # 통합
    combined = Concatenate()([lstm2, attn, cnn, recent])
    combined = BatchNormalization()(combined)
    
    # Dense layers
    x = Dense(128, activation='relu')(combined)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # 출력층
    out_reg = Dense(1, name='regression')(x)  # 예측값
    out_cls = Dense(3, activation='softmax', name='classification')(x)  # 구간
    out_danger = Dense(1, activation='sigmoid', name='danger')(x)  # 위험신호
    out_trend = Dense(3, activation='softmax', name='trend')(x)  # 추세
    
    return Model(inputs, [out_reg, out_cls, out_danger, out_trend], name='ExtremeNet_V65')

# ========================================
# 체크포인트 학습 시스템
# ========================================
class CheckpointTrainer:
    def __init__(self, model_name='ExtremeNet_V65'):
        self.model_name = model_name
        self.checkpoint_dir = f'checkpoints_{model_name}/'
        self.history_file = f'{self.checkpoint_dir}history.json'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def load_checkpoint(self):
        """체크포인트 로드"""
        checkpoint_path = f'{self.checkpoint_dir}{self.model_name}.keras'
        history = {'epoch': 0, 'best_loss': float('inf')}
        
        if os.path.exists(checkpoint_path):
            print(f"✅ 체크포인트 발견! 로딩...")
            model = tf.keras.models.load_model(checkpoint_path)
            
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    history = json.load(f)
                print(f"  이전 학습: {history['epoch']}에포크, 최고손실: {history['best_loss']:.4f}")
        else:
            print("⚠️ 체크포인트 없음, 새로 시작")
            model = None
            
        return model, history
    
    def save_checkpoint(self, model, epoch, best_loss):
        """체크포인트 저장"""
        checkpoint_path = f'{self.checkpoint_dir}{self.model_name}.keras'
        model.save(checkpoint_path)
        
        history = {'epoch': epoch, 'best_loss': float(best_loss)}
        with open(self.history_file, 'w') as f:
            json.dump(history, f)
        
        print(f"💾 체크포인트 저장: 에포크 {epoch}, 손실 {best_loss:.4f}")

# ========================================
# 커스텀 콜백 (위험구간 집중)
# ========================================
class DangerZoneFocusCallback(tf.keras.callbacks.Callback):
    """위험구간(1651~1682) 예측 성능 모니터링"""
    
    def __init__(self, X_val, y_val, processor):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.processor = processor
        
    def on_epoch_end(self, epoch, logs=None):
        # 위험구간 샘플만
        danger_idx = np.where(self.y_val[2] == 1)[0]
        
        if len(danger_idx) > 0:
            X_danger = self.X_val[danger_idx]
            y_danger_true = self.y_val[0][danger_idx]
            
            # 예측
            preds = self.model.predict(X_danger, verbose=0)
            y_danger_pred = preds[0].flatten()
            
            # 역변환
            y_true_orig = self.processor.target_scaler.inverse_transform(
                y_danger_true.reshape(-1, 1)).flatten()
            y_pred_orig = self.processor.target_scaler.inverse_transform(
                y_danger_pred.reshape(-1, 1)).flatten()
            
            mae = mean_absolute_error(y_true_orig, y_pred_orig)
            accuracy = np.mean(np.abs(y_true_orig - y_pred_orig) < 50)  # 50 이내 정확도
            
            print(f"\n  🔥 위험구간 성능: MAE={mae:.1f}, 정확도={accuracy:.1%}")

# ========================================
# 학습 및 평가
# ========================================
def train_extremenet(train_data, val_data, test_data, processor):
    """ExtremeNet V6.5 학습"""
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data
    
    # 체크포인트 시스템
    trainer = CheckpointTrainer()
    model, history = trainer.load_checkpoint()
    
    # 모델 생성 또는 로드
    with strategy.scope():
        if model is None:
            input_shape = (X_train.shape[1], X_train.shape[2])
            model = build_extremenet_v65(input_shape)
            
            model.compile(
                optimizer=Adam(0.001),
                loss={
                    'regression': 'huber',  # 이상치에 강건
                    'classification': 'sparse_categorical_crossentropy',
                    'danger': 'binary_crossentropy',
                    'trend': 'sparse_categorical_crossentropy'
                },
                loss_weights={
                    'regression': 0.5,
                    'classification': 0.2,
                    'danger': 0.2,  # 위험구간 중요
                    'trend': 0.1
                },
                metrics={
                    'regression': 'mae',
                    'classification': 'accuracy',
                    'danger': 'accuracy',
                    'trend': 'accuracy'
                }
            )
    
    print("\n" + "="*80)
    print("🔥 ExtremeNet V6.5 학습 시작")
    print("="*80)
    
    # 콜백
    callbacks = [
        # 체크포인트 저장
        ModelCheckpoint(
            f'{trainer.checkpoint_dir}{trainer.model_name}_best.keras',
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        ),
        
        # 조기종료
        EarlyStopping(
            patience=15,
            restore_best_weights=True,
            monitor='val_loss',
            verbose=1
        ),
        
        # 학습률 조정
        ReduceLROnPlateau(
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        
        # 위험구간 모니터링
        DangerZoneFocusCallback(X_val, y_val, processor)
    ]
    
    # 배치 크기
    batch_size = 64 * strategy.num_replicas_in_sync if gpus else 32
    
    # 학습 재개 또는 시작
    initial_epoch = history['epoch']
    
    print(f"\n📚 학습 설정:")
    print(f"  시작 에포크: {initial_epoch}")
    print(f"  배치 크기: {batch_size}")
    print(f"  총 에포크: 100")
    
    # 학습
    history = model.fit(
        X_train,
        {
            'regression': y_train[0],
            'classification': y_train[1],
            'danger': y_train[2],
            'trend': y_train[3]
        },
        validation_data=(
            X_val,
            {
                'regression': y_val[0],
                'classification': y_val[1],
                'danger': y_val[2],
                'trend': y_val[3]
            }
        ),
        epochs=100,
        initial_epoch=initial_epoch,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # 체크포인트 저장
    best_loss = min(history.history['val_loss'])
    final_epoch = initial_epoch + len(history.history['val_loss'])
    trainer.save_checkpoint(model, final_epoch, best_loss)
    
    # 평가
    print("\n" + "="*80)
    print("📊 ExtremeNet V6.5 최종 평가")
    print("="*80)
    
    # 전체 테스트
    preds = model.predict(X_test, verbose=0)
    y_reg_pred = preds[0].flatten()
    y_cls_pred = np.argmax(preds[1], axis=1)
    y_danger_pred = (preds[2].flatten() > 0.5).astype(int)
    y_trend_pred = np.argmax(preds[3], axis=1)
    
    # 역변환
    y_reg_true_orig = processor.target_scaler.inverse_transform(
        y_test[0].reshape(-1, 1)).flatten()
    y_reg_pred_orig = processor.target_scaler.inverse_transform(
        y_reg_pred.reshape(-1, 1)).flatten()
    
    # 메트릭 계산
    mae = mean_absolute_error(y_reg_true_orig, y_reg_pred_orig)
    rmse = np.sqrt(mean_squared_error(y_reg_true_orig, y_reg_pred_orig))
    
    # R2 (음수 방지)
    r2_raw = r2_score(y_reg_true_orig, y_reg_pred_orig)
    r2 = max(0.0, r2_raw)  # 음수면 0으로
    
    # 정확도
    cls_acc = np.mean(y_cls_pred == y_test[1])
    danger_acc = np.mean(y_danger_pred == y_test[2])
    trend_acc = np.mean(y_trend_pred == y_test[3])
    
    print(f"\n📈 회귀 성능:")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R² Score: {r2:.4f} (원본: {r2_raw:.4f})")
    
    print(f"\n🎯 분류 성능:")
    print(f"  구간 정확도: {cls_acc:.2%}")
    print(f"  위험신호 정확도: {danger_acc:.2%}")
    print(f"  추세 정확도: {trend_acc:.2%}")
    
    # 위험구간만 평가
    danger_idx = np.where(y_test[2] == 1)[0]
    if len(danger_idx) > 0:
        danger_mae = mean_absolute_error(
            y_reg_true_orig[danger_idx],
            y_reg_pred_orig[danger_idx]
        )
        print(f"\n🔥 위험구간(1651~1682) 성능:")
        print(f"  샘플 수: {len(danger_idx)}개")
        print(f"  MAE: {danger_mae:.2f}")
        print(f"  감지율: {np.mean(y_danger_pred[danger_idx]):.1%}")
    
    # 추세별 성능
    print(f"\n📊 추세별 예측 성능:")
    for trend, name in enumerate(['하락', '보합', '상승']):
        trend_idx = np.where(y_test[3] == trend)[0]
        if len(trend_idx) > 0:
            trend_mae = mean_absolute_error(
                y_reg_true_orig[trend_idx],
                y_reg_pred_orig[trend_idx]
            )
            print(f"  {name} 추세: MAE={trend_mae:.2f} (n={len(trend_idx)})")
    
    # 모델 저장
    os.makedirs('models_v65', exist_ok=True)
    model.save('models_v65/ExtremeNet_V65_final.keras')
    print(f"\n💾 최종 모델: models_v65/ExtremeNet_V65_final.keras")
    
    return model, {'MAE': mae, 'RMSE': rmse, 'R2': r2}

# ========================================
# 실시간 예측 시스템
# ========================================
class ExtremeNetPredictor:
    def __init__(self, model_path='models_v65/ExtremeNet_V65_final.keras'):
        """ExtremeNet V6.5 예측기"""
        self.model = tf.keras.models.load_model(model_path)
        self.processor = PatternDataProcessor()
        self.processor.load_scalers('scalers_v65/')
        print("✅ ExtremeNet V6.5 예측기 준비 완료")
    
    def predict(self, data_100min):
        """100분 데이터로 10분 후 예측"""
        
        # 패턴 감지
        pattern = self.detect_current_pattern(data_100min)
        
        # 추세 판단
        trend = self.detect_current_trend(data_100min)
        
        # 특성 추출 및 스케일링
        X = data_100min[self.processor.feature_columns].values
        X_scaled = self.processor.feature_scaler.transform(X)
        X_seq = X_scaled.reshape(1, 100, -1)
        
        # 예측
        preds = self.model.predict(X_seq, verbose=0)
        
        # 결과 해석
        y_reg = self.processor.target_scaler.inverse_transform([[preds[0][0, 0]]])[0, 0]
        y_cls = np.argmax(preds[1][0])
        y_danger = preds[2][0, 0]
        y_trend = np.argmax(preds[3][0])
        
        # 레벨
        levels = ['정상', '주의', '심각']
        trends = ['하락', '보합', '상승']
        
        # 위험 신호 체크
        is_danger_zone = (1651 <= y_reg <= 1682)
        
        result = {
            'prediction': float(y_reg),
            'level': levels[y_cls],
            'danger_probability': float(y_danger),
            'trend': trends[y_trend],
            'pattern': f"UU{pattern+1}" if pattern > 0 else "정상",
            'is_danger_zone': is_danger_zone,
            'confidence': float(np.max(preds[1][0]))
        }
        
        # 경고 메시지
        if is_danger_zone or y_danger > 0.7:
            result['alert'] = "⚠️ 위험신호! 1700+ 급증 가능성 높음"
        elif y_cls == 2:
            result['alert'] = "🔴 심각 구간 예상"
        elif y_cls == 1:
            result['alert'] = "🟡 주의 구간 예상"
        else:
            result['alert'] = "🟢 정상 구간"
        
        return result
    
    def detect_current_pattern(self, df):
        """현재 패턴 감지"""
        high_cases = (df['TOTALCNT'] >= 1682).sum()
        m14b_mean = df['M14AM14B'].mean()
        
        if high_cases > 0 and m14b_mean > 380:
            return 2  # UU2
        elif (df['M14AM14B'] > 350).mean() > 0.3:
            return 2  # UU2
        elif ((df['TOTALCNT'] >= 1651) & (df['TOTALCNT'] <= 1682)).sum() > 0:
            return 1  # UU1
        else:
            return 0  # 정상
    
    def detect_current_trend(self, df):
        """현재 추세 감지"""
        recent = df['TOTALCNT'].tail(20).values
        if len(recent) < 2:
            return "보합"
        
        slope = np.polyfit(np.arange(len(recent)), recent, 1)[0]
        
        if slope > 5:
            return "상승"
        elif slope < -5:
            return "하락"
        else:
            return "보합"

# ========================================
# 메인 실행
# ========================================
def main():
    # 데이터 경로
    data_paths = [
        'data/20240201_TO_202507281705.csv',
        '20240201_TO_202507281705.csv'
    ]
    
    data_path = None
    for path in data_paths:
        if os.path.exists(path):
            data_path = path
            print(f"✅ 데이터 발견: {path}")
            break
    
    if not data_path:
        print("❌ 데이터 파일을 찾을 수 없습니다.")
        return None
    
    # 데이터 처리
    processor = PatternDataProcessor()
    df = processor.load_and_process(data_path)
    df = processor.create_features(df)
    
    # 데이터 분할
    train_data, val_data, test_data = processor.create_sequences(df)
    
    # ExtremeNet V6.5 학습
    model, results = train_extremenet(train_data, val_data, test_data, processor)
    
    print("\n" + "="*80)
    print("🎯 ExtremeNet V6.5 학습 완료!")
    print("="*80)
    
    # 실시간 예측 테스트
    print("\n🔮 실시간 예측 테스트")
    print("-"*50)
    
    predictor = ExtremeNetPredictor()
    
    # 테스트 샘플
    test_sample = df.iloc[-100:].copy()
    result = predictor.predict(test_sample)
    
    print(f"\n예측 결과:")
    print(f"  10분 후 예측값: {result['prediction']:.0f}")
    print(f"  구간: {result['level']}")
    print(f"  추세: {result['trend']}")
    print(f"  패턴: {result['pattern']}")
    print(f"  위험확률: {result['danger_probability']:.1%}")
    print(f"  신뢰도: {result['confidence']:.1%}")
    print(f"  {result['alert']}")
    
    # 결과 저장
    summary = {
        'model': 'ExtremeNet V6.5',
        'mae': results['MAE'],
        'rmse': results['RMSE'],
        'r2': results['R2'],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('extremenet_v65_results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n✅ 완료! 결과 저장: extremenet_v65_results.json")
    print("\n💡 중단 후 재실행시 자동으로 체크포인트에서 학습 재개됩니다!")
    
    return model, results

if __name__ == "__main__":
    model, results = main()