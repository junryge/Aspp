"""
🔥 ExtremeNet V6.5 + UU1/UU2 패턴 통합
================================================================
기존 학습 유지 + 패턴 강화로 1700+ 예측 개선
목표: MAE < 40, 위험구간 정확도 > 80%
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
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
    print(f"🔧 GPU {len(gpus)}개 발견!")
    for i, gpu in enumerate(gpus):
        tf.config.experimental.set_memory_growth(gpu, True)
    strategy = tf.distribute.MirroredStrategy()
    DEVICE = 'GPU'
    BATCH_SIZE = 128
else:
    print("⚠️ GPU 없음, CPU 사용")
    strategy = tf.distribute.get_strategy()
    DEVICE = 'CPU'
    BATCH_SIZE = 32

print("="*80)
print("🔥 ExtremeNet V6.5 + 패턴 통합")
print(f"📦 TensorFlow: {tf.__version__}")
print(f"🔧 Device: {DEVICE}")
print(f"📊 배치 크기: {BATCH_SIZE}")
print("="*80)

# ========================================
# UU1/UU2 패턴 감지 (실제 예측 코드에서 가져옴)
# ========================================
def detect_pattern_for_window(df_window):
    """100분 윈도우에서 패턴 감지"""
    totalcnt_max = df_window['TOTALCNT'].max()
    m14b_mean = df_window['M14AM14B'].mean()
    m14b_max = df_window['M14AM14B'].max()
    high_cases = (df_window['TOTALCNT'] >= 1682).sum()
    m14b_350_ratio = (df_window['M14AM14B'] > 350).sum() / len(df_window)
    
    # UU2: 고값 유지
    if high_cases > 0 and m14b_mean > 380:
        return 2  # UU2
    elif m14b_350_ratio > 0.3:
        return 2  # UU2
    # UU1: 급증 가능
    else:
        return 1  # UU1

def calculate_pattern_risk_score(df_window, pattern):
    """패턴 기반 위험도 점수 계산"""
    current_val = df_window['TOTALCNT'].iloc[-1]
    m14b = df_window['M14AM14B'].iloc[-1]
    m14a = df_window['M14AM10A'].iloc[-1]
    ratio = m14b / (m14a + 1)
    
    # 연속 상승
    consecutive_rises = 0
    for i in range(len(df_window)-1, 0, -1):
        if df_window['TOTALCNT'].iloc[i] > df_window['TOTALCNT'].iloc[i-1]:
            consecutive_rises += 1
        else:
            break
    
    # 위험도 점수
    risk_score = 0
    
    if pattern == 1:  # UU1
        if current_val >= 1650:
            risk_score += 40
        if m14b > 300 and m14a < 80:  # 황금 패턴
            risk_score += 30
        if consecutive_rises >= 10:
            risk_score += 20
        if m14b >= 400:
            risk_score += 25
    else:  # UU2
        if m14b >= 450:
            risk_score += 35
        elif m14b >= 400:
            risk_score += 30
    
    return min(risk_score, 100) / 100.0  # 0~1 범위

# ========================================
# 개선된 데이터 처리 (기존 + 패턴)
# ========================================
class DataProcessor:
    def __init__(self):
        self.seq_len = 100
        self.pred_len = 10
        
        # 스케일러
        self.scaler_X = StandardScaler()
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))
        
        # 구간 재정의
        self.NORMAL_MAX = 1400
        self.CAUTION_MAX = 1650
        self.SIGNAL_MIN = 1651
        self.SIGNAL_MAX = 1699
        self.DANGER_MIN = 1700
        
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
        normal = (df['TOTALCNT'] < self.NORMAL_MAX).sum()
        caution = ((df['TOTALCNT'] >= self.NORMAL_MAX) & 
                  (df['TOTALCNT'] <= self.CAUTION_MAX)).sum()
        signal = ((df['TOTALCNT'] >= self.SIGNAL_MIN) & 
                 (df['TOTALCNT'] <= self.SIGNAL_MAX)).sum()
        danger = (df['TOTALCNT'] >= self.DANGER_MIN).sum()
        
        print(f"\n📊 TOTALCNT 분포:")
        print(f"  🟢 정상(~1400): {normal:,}개 ({normal/len(df)*100:.1f}%)")
        print(f"  🟡 주의(1400~1650): {caution:,}개 ({caution/len(df)*100:.1f}%)")
        print(f"  🟠 신호(1651~1699): {signal:,}개 ({signal/len(df)*100:.1f}%)")
        print(f"  🔴 위험(1700+): {danger:,}개 ({danger/len(df)*100:.1f}%)")
        
        return df
    
    def create_features(self, df):
        """특성 생성 (패턴 특성 추가)"""
        print("\n⚙️ 특성 생성 중...")
        
        # ===== 기존 특성 =====
        df['RATIO'] = df['M14AM14B'] / (df['M14AM10A'] + 1)
        df['GOLDEN'] = ((df['M14AM14B'] > 300) & (df['M14AM10A'] < 80)).astype(int)
        
        # 신호구간 특성
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
        
        # ===== 패턴 특성 (새로 추가) =====
        # UU1/UU2 패턴
        df['PATTERN'] = 0
        df['PATTERN_RISK'] = 0.0
        
        # 100분 윈도우로 패턴 감지
        for i in range(100, len(df)):
            window = df.iloc[i-100:i]
            pattern = detect_pattern_for_window(window)
            risk = calculate_pattern_risk_score(window, pattern)
            df.loc[i, 'PATTERN'] = pattern
            df.loc[i, 'PATTERN_RISK'] = risk
        
        # M14B 기반 위험 레벨
        df['M14B_HIGH'] = (df['M14AM14B'] >= 350).astype(int)
        df['M14B_VERY_HIGH'] = (df['M14AM14B'] >= 400).astype(int)
        df['M14B_EXTREME'] = (df['M14AM14B'] >= 450).astype(int)
        
        # 급증 신호
        df['SPIKE_SIGNAL'] = ((df['CHANGE_10'] > 50) & (df['RISE_COUNT'] > 5)).astype(int)
        
        # 추세
        df['TREND'] = 1
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
        """시퀀스 생성 (패턴별 오버샘플링)"""
        print("\n🔄 시퀀스 생성 중...")
        
        # 사용할 컬럼
        self.feature_columns = [
            'TOTALCNT', 'M14AM14B', 'M14AM14BSUM', 'M14AM10A', 'M14AM16',
            'RATIO', 'GOLDEN', 
            'SIGNAL_ZONE', 'PRE_SIGNAL', 'DANGER_ZONE',
            'PATTERN', 'PATTERN_RISK',  # 패턴 특성
            'M14B_HIGH', 'M14B_VERY_HIGH', 'M14B_EXTREME',  # M14B 레벨
            'SPIKE_SIGNAL',  # 급증 신호
            'TREND',
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
        pattern_samples = {'UU1_danger': [], 'UU2_danger': [], 'signal': []}
        
        for i in range(len(df) - self.seq_len - self.pred_len):
            seq_X = X_data[i:i+self.seq_len]
            target = y_data[i+self.seq_len-1]
            
            if np.isnan(target):
                continue
            
            current = df['TOTALCNT'].iloc[i+self.seq_len-1]
            pattern = df['PATTERN'].iloc[i+self.seq_len-1]
            
            X.append(seq_X)
            y.append(target)
            
            # 상승 추세 확인
            is_rising = target > current
            
            # 패턴별 오버샘플링
            if is_rising:
                # UU1 패턴 + 위험구간
                if pattern == 1 and target >= self.SIGNAL_MIN:
                    for _ in range(8 if target >= self.DANGER_MIN else 5):
                        pattern_samples['UU1_danger'].append((seq_X, target))
                
                # UU2 패턴 + 위험구간
                elif pattern == 2 and target >= self.DANGER_MIN:
                    for _ in range(10):
                        pattern_samples['UU2_danger'].append((seq_X, target))
                
                # 신호구간
                elif self.SIGNAL_MIN <= target <= self.SIGNAL_MAX:
                    for _ in range(6):
                        pattern_samples['signal'].append((seq_X, target))
        
        # 오버샘플링 데이터 추가
        for key, samples in pattern_samples.items():
            if samples:
                X_add, y_add = zip(*samples)
                X.extend(X_add)
                y.extend(y_add)
                print(f"  🔥 {key}: {len(samples)}개 추가")
        
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
        
        # 구간별 통계
        signal_train = ((y_train >= self.SIGNAL_MIN) & (y_train <= self.SIGNAL_MAX)).sum()
        signal_val = ((y_val >= self.SIGNAL_MIN) & (y_val <= self.SIGNAL_MAX)).sum()
        signal_test = ((y_test >= self.SIGNAL_MIN) & (y_test <= self.SIGNAL_MAX)).sum()
        
        danger_train = (y_train >= self.DANGER_MIN).sum()
        danger_val = (y_val >= self.DANGER_MIN).sum()
        danger_test = (y_test >= self.DANGER_MIN).sum()
        
        print(f"\n🟠 신호구간(1651~1699):")
        print(f"  Train: {signal_train}개 ({signal_train/len(X_train)*100:.1f}%)")
        print(f"  Val: {signal_val}개 ({signal_val/len(X_val)*100:.1f}%)")
        print(f"  Test: {signal_test}개 ({signal_test/len(X_test)*100:.1f}%)")
        
        print(f"\n🔴 위험구간(1700+):")
        print(f"  Train: {danger_train}개 ({danger_train/len(X_train)*100:.1f}%)")
        print(f"  Val: {danger_val}개 ({danger_val/len(X_val)*100:.1f}%)")
        print(f"  Test: {danger_test}개 ({danger_test/len(X_test)*100:.1f}%)")
        
        # 레이블 생성
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
        
        # 패턴 레이블
        y_train_pattern = []
        y_val_pattern = []
        y_test_pattern = []
        
        for i in range(len(y_train)):
            if i + train_size < len(df):
                y_train_pattern.append(df['PATTERN'].iloc[i + train_size])
            else:
                y_train_pattern.append(1)  # 기본값
        
        for i in range(len(y_val)):
            if i + train_size + val_size < len(df):
                y_val_pattern.append(df['PATTERN'].iloc[i + train_size + val_size])
            else:
                y_val_pattern.append(1)
                
        for i in range(len(y_test)):
            if i + train_size + val_size + len(y_test) < len(df):
                y_test_pattern.append(df['PATTERN'].iloc[i + train_size + val_size + len(y_test)])
            else:
                y_test_pattern.append(1)
        
        y_train_pattern = np.array(y_train_pattern, dtype=int) - 1  # 0,1로 변환
        y_val_pattern = np.array(y_val_pattern, dtype=int) - 1
        y_test_pattern = np.array(y_test_pattern, dtype=int) - 1
        
        # 스케일러 저장
        os.makedirs('scalers_pattern', exist_ok=True)
        with open('scalers_pattern/scaler_X.pkl', 'wb') as f:
            pickle.dump(self.scaler_X, f)
        with open('scalers_pattern/scaler_y.pkl', 'wb') as f:
            pickle.dump(self.scaler_y, f)
        
        return (X_train_scaled, y_train_scaled, y_train, y_train_level, y_train_pattern), \
               (X_val_scaled, y_val_scaled, y_val, y_val_level, y_val_pattern), \
               (X_test_scaled, y_test_scaled, y_test, y_test_level, y_test_pattern)

# ========================================
# 패턴 통합 ExtremeNet 모델
# ========================================
def build_pattern_extremenet(input_shape):
    """패턴 인식 강화 ExtremeNet"""
    inputs = Input(shape=input_shape)
    
    # ===== 기존 브랜치 =====
    # LSTM 브랜치
    lstm = LSTM(128, return_sequences=True)(inputs)
    lstm = BatchNormalization()(lstm)
    lstm = LSTM(64)(lstm)
    lstm = Dropout(0.3)(lstm)
    
    # Attention 브랜치
    attn = MultiHeadAttention(num_heads=4, key_dim=16)(inputs, inputs)
    attn = GlobalAveragePooling1D()(attn)
    
    # CNN 브랜치
    cnn = Conv1D(64, 5, activation='relu')(inputs)
    cnn = BatchNormalization()(cnn)
    cnn = Conv1D(32, 3, activation='relu')(cnn)
    cnn = GlobalMaxPooling1D()(cnn)
    
    # Recent 브랜치
    recent = Lambda(lambda x: x[:, -20:, :])(inputs)
    recent = GRU(32)(recent)
    
    # ===== 패턴 전용 브랜치 (새로 추가) =====
    # 패턴 특성 추출 (PATTERN, PATTERN_RISK 컬럼 위치)
    pattern_features = Lambda(lambda x: x[:, :, 10:12])(inputs)  # 패턴 관련 컬럼
    pattern = Conv1D(16, 3, activation='relu')(pattern_features)
    pattern = GlobalAveragePooling1D()(pattern)
    
    # M14B 레벨 추출
    m14b_features = Lambda(lambda x: x[:, :, 1:2])(inputs)  # M14AM14B 컬럼
    m14b = Conv1D(16, 3, activation='relu')(m14b_features)
    m14b = GlobalMaxPooling1D()(m14b)
    
    # ===== 통합 =====
    merged = Concatenate()([lstm, attn, cnn, recent, pattern, m14b])
    merged = BatchNormalization()(merged)
    
    # Dense
    x = Dense(256, activation='relu')(merged)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # 출력
    output_reg = Dense(1, name='regression')(x)  # 값 예측
    output_cls = Dense(4, activation='softmax', name='classification')(x)  # 4구간
    output_pattern = Dense(2, activation='softmax', name='pattern')(x)  # UU1/UU2
    
    model = Model(inputs, [output_reg, output_cls, output_pattern], 
                  name='Pattern_ExtremeNet')
    return model

# ========================================
# 커스텀 손실함수 (패턴별 가중치)
# ========================================
def weighted_mae_pattern(y_true, y_pred):
    """패턴 기반 가중 MAE"""
    signal_min_scaled = 0.65
    signal_max_scaled = 0.75
    
    weights = tf.ones_like(y_true)
    
    # 신호구간
    signal_mask = tf.logical_and(
        y_true >= signal_min_scaled,
        y_true <= signal_max_scaled
    )
    weights = tf.where(signal_mask, 5.0, weights)
    
    # 위험구간
    danger_mask = y_true > signal_max_scaled
    weights = tf.where(danger_mask, 10.0, weights)
    
    mae = tf.abs(y_true - y_pred)
    weighted_mae = mae * weights
    
    return tf.reduce_mean(weighted_mae)

# ========================================
# 학습 함수
# ========================================
def train_model(train_data, val_data, test_data, processor):
    """모델 학습"""
    X_train, y_train_scaled, y_train_orig, y_train_level, y_train_pattern = train_data
    X_val, y_val_scaled, y_val_orig, y_val_level, y_val_pattern = val_data
    X_test, y_test_scaled, y_test_orig, y_test_level, y_test_pattern = test_data
    
    # 모델 생성
    with strategy.scope():
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = build_pattern_extremenet(input_shape)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={
                'regression': weighted_mae_pattern,
                'classification': 'sparse_categorical_crossentropy',
                'pattern': 'sparse_categorical_crossentropy'
            },
            loss_weights={
                'regression': 0.5,
                'classification': 0.25,
                'pattern': 0.25
            },
            metrics={
                'regression': 'mae',
                'classification': 'accuracy',
                'pattern': 'accuracy'
            }
        )
    
    print("\n" + "="*80)
    print("🔥 패턴 통합 ExtremeNet 학습 시작")
    print(f"🔧 Device: {DEVICE} | Batch: {BATCH_SIZE}")
    print("="*80)
    
    # 콜백
    callbacks = [
        ModelCheckpoint(
            'models_pattern/best_model.keras',
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        ),
        EarlyStopping(
            patience=25,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # 학습
    os.makedirs('models_pattern', exist_ok=True)
    
    history = model.fit(
        X_train,
        {
            'regression': y_train_scaled,
            'classification': y_train_level,
            'pattern': y_train_pattern
        },
        validation_data=(
            X_val,
            {
                'regression': y_val_scaled,
                'classification': y_val_level,
                'pattern': y_val_pattern
            }
        ),
        epochs=100,
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
        print(f"\n🟠 신호구간(1651~1699):")
        print(f"  샘플: {np.sum(signal_idx)}개")
        print(f"  MAE: {signal_mae:.2f}")
        print(f"  정확도(±50): {np.mean(np.abs(y_test_orig[signal_idx] - y_pred[signal_idx]) < 50)*100:.1f}%")
    
    # 위험구간 평가
    danger_idx = (y_test_orig >= processor.DANGER_MIN)
    if np.sum(danger_idx) > 0:
        danger_mae = mean_absolute_error(y_test_orig[danger_idx], y_pred[danger_idx])
        accuracy_30 = np.mean(np.abs(y_test_orig[danger_idx] - y_pred[danger_idx]) < 30)*100
        accuracy_50 = np.mean(np.abs(y_test_orig[danger_idx] - y_pred[danger_idx]) < 50)*100
        
        print(f"\n🔴 위험구간(1700+):")
        print(f"  샘플: {np.sum(danger_idx)}개")
        print(f"  MAE: {danger_mae:.2f}")
        print(f"  정확도(±30): {accuracy_30:.1f}%")
        print(f"  정확도(±50): {accuracy_50:.1f}%")
    
    # 모델 저장
    model.save('models_pattern/pattern_extremenet_final.keras')
    print(f"\n💾 최종 모델: models_pattern/pattern_extremenet_final.keras")
    
    return model, {'MAE': mae, 'RMSE': rmse, 'R2': r2}

# ========================================
# 메인
# ========================================
def main():
    # 데이터 경로
    data_path = None
    for path in ['data/20240201_TO_202507281705.csv', 
                 '20240201_TO_202507281705.csv',
                 'data.csv']:
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
    print("🔥 패턴 통합 ExtremeNet 학습 완료!")
    
    return model, results

if __name__ == "__main__":
    model, results = main()