"""
🔥 ExtremeNet V7.0 - 위험구간(1700+) 예측 강화 버전
================================================================
핵심 개선사항:
1. 전이 패턴 학습: 1651~1699 → 1700+ 전이 특별 학습
2. 시계열 연속성 강화: Transformer + Temporal Convolution
3. 위험구간 전용 브랜치: 1700+ 특화 예측
4. 패턴 마이닝: 위험 전조 패턴 자동 탐지
5. 앙상블 예측: 다중 시점 예측 후 통합
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
from datetime import datetime

warnings.filterwarnings('ignore')
tf.random.set_seed(42)
np.random.seed(42)

# GPU 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    strategy = tf.distribute.MirroredStrategy()
else:
    strategy = tf.distribute.get_strategy()

print("="*80)
print("🔥 ExtremeNet V7.0 - 위험구간 예측 강화")
print(f"📦 TensorFlow: {tf.__version__}")
print("="*80)

# ========================================
# 개선된 데이터 처리
# ========================================
class EnhancedDataProcessor:
    def __init__(self):
        self.seq_len = 100
        self.pred_len = 10
        
        # 다중 스케일러 (RobustScaler 추가 - 이상치에 강함)
        self.scaler_X = StandardScaler()
        self.scaler_y = RobustScaler()  # MinMaxScaler 대신 RobustScaler
        
        # 구간 정의
        self.NORMAL_MAX = 1400
        self.CAUTION_MAX = 1650
        self.SIGNAL_MIN = 1651
        self.SIGNAL_MAX = 1699
        self.DANGER_MIN = 1700
        
    def create_advanced_features(self, df):
        """고급 특성 생성 - 전이 패턴 포착"""
        print("\n⚙️ 고급 특성 생성 중...")
        
        # 기본 특성
        df['RATIO'] = df['M14AM14B'] / (df['M14AM10A'] + 1)
        df['GOLDEN'] = ((df['M14AM14B'] > 300) & (df['M14AM10A'] < 80)).astype(int)
        
        # 구간 특성
        df['SIGNAL_ZONE'] = ((df['TOTALCNT'] >= self.SIGNAL_MIN) & 
                             (df['TOTALCNT'] <= self.SIGNAL_MAX)).astype(int)
        df['PRE_SIGNAL'] = ((df['TOTALCNT'] >= 1600) & 
                            (df['TOTALCNT'] < self.SIGNAL_MIN)).astype(int)
        df['DANGER_ZONE'] = (df['TOTALCNT'] >= self.DANGER_MIN).astype(int)
        
        # ⭐ 전이 패턴 특성 (새로 추가)
        df['NEAR_SIGNAL'] = ((df['TOTALCNT'] >= 1630) & 
                             (df['TOTALCNT'] < self.SIGNAL_MIN)).astype(int)
        df['SIGNAL_TO_DANGER_RISK'] = ((df['TOTALCNT'] >= 1680) & 
                                       (df['TOTALCNT'] < self.DANGER_MIN)).astype(int)
        
        # 신호구간 지속 시간
        df['SIGNAL_DURATION'] = 0
        signal_count = 0
        for i in range(len(df)):
            if df['SIGNAL_ZONE'].iloc[i] == 1:
                signal_count += 1
            else:
                signal_count = 0
            df.loc[i, 'SIGNAL_DURATION'] = signal_count
        
        # 이동평균 (더 많은 윈도우)
        for w in [3, 5, 10, 20, 30]:
            df[f'MA_{w}'] = df['TOTALCNT'].rolling(w, min_periods=1).mean()
            df[f'STD_{w}'] = df['TOTALCNT'].rolling(w, min_periods=1).std().fillna(0)
            df[f'M14B_MA_{w}'] = df['M14AM14B'].rolling(w, min_periods=1).mean()
        
        # 변화율 (더 세밀하게)
        for lag in [1, 3, 5, 10, 15]:
            df[f'CHANGE_{lag}'] = df['TOTALCNT'].diff(lag).fillna(0)
            df[f'M14B_CHANGE_{lag}'] = df['M14AM14B'].diff(lag).fillna(0)
            
            # 변화율의 변화율 (가속도)
            df[f'ACCEL_{lag}'] = df[f'CHANGE_{lag}'].diff(1).fillna(0)
        
        # 연속 상승
        df['RISE'] = (df['TOTALCNT'] > df['TOTALCNT'].shift(1)).astype(int)
        df['RISE_COUNT'] = df['RISE'].rolling(10, min_periods=1).sum()
        df['RISE_STREAK'] = df.groupby((df['RISE'] != df['RISE'].shift()).cumsum())['RISE'].cumsum()
        
        # 급증 신호 (10분에 50 이상 증가)
        df['SPIKE_10'] = (df['CHANGE_10'] > 50).astype(int)
        df['SPIKE_5'] = (df['CHANGE_5'] > 30).astype(int)
        
        # 볼린저 밴드
        for w in [20, 30]:
            ma = df['TOTALCNT'].rolling(w, min_periods=1).mean()
            std = df['TOTALCNT'].rolling(w, min_periods=1).std()
            df[f'BB_UPPER_{w}'] = ma + (2 * std)
            df[f'BB_LOWER_{w}'] = ma - (2 * std)
            df[f'BB_WIDTH_{w}'] = df[f'BB_UPPER_{w}'] - df[f'BB_LOWER_{w}']
            df[f'BB_POSITION_{w}'] = (df['TOTALCNT'] - df[f'BB_LOWER_{w}']) / (df[f'BB_WIDTH_{w}'] + 1)
        
        # RSI (상대강도지수)
        def calculate_rsi(series, period=14):
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
            rs = gain / (loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        df['RSI_14'] = calculate_rsi(df['TOTALCNT'])
        df['RSI_7'] = calculate_rsi(df['TOTALCNT'], 7)
        
        # MACD
        exp1 = df['TOTALCNT'].ewm(span=12, adjust=False).mean()
        exp2 = df['TOTALCNT'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_SIGNAL'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_DIFF'] = df['MACD'] - df['MACD_SIGNAL']
        
        # 패턴 인식
        df['PATTERN'] = 0
        
        # 패턴 1: 신호구간
        signal_mask = ((df['TOTALCNT'] >= self.SIGNAL_MIN) & 
                      (df['TOTALCNT'] <= self.SIGNAL_MAX))
        df.loc[signal_mask, 'PATTERN'] = 1
        
        # 패턴 2: 급상승 중
        rapid_rise = (df['CHANGE_5'] > 30) & (df['RISE_COUNT'] > 7)
        df.loc[rapid_rise, 'PATTERN'] = 2
        
        # 패턴 3: 위험 임박
        danger_imminent = (df['TOTALCNT'] >= 1680) & (df['RSI_14'] > 70)
        df.loc[danger_imminent, 'PATTERN'] = 3
        
        # 추세 (더 정교하게)
        df['TREND'] = 1
        df['TREND_STRENGTH'] = 0
        
        for i in range(30, len(df)):
            recent_30 = df['TOTALCNT'].iloc[i-30:i].values
            recent_10 = df['TOTALCNT'].iloc[i-10:i].values
            
            if len(recent_30) > 1:
                # 장기 추세
                slope_30 = np.polyfit(range(len(recent_30)), recent_30, 1)[0]
                # 단기 추세
                slope_10 = np.polyfit(range(len(recent_10)), recent_10, 1)[0]
                
                # 추세 판단
                if slope_10 > 5:
                    df.loc[i, 'TREND'] = 2  # 상승
                    df.loc[i, 'TREND_STRENGTH'] = slope_10
                elif slope_10 < -5:
                    df.loc[i, 'TREND'] = 0  # 하락
                    df.loc[i, 'TREND_STRENGTH'] = slope_10
                else:
                    df.loc[i, 'TREND'] = 1  # 보합
                    
                # 추세 전환 감지
                if i > 40:
                    prev_slope = np.polyfit(range(10), df['TOTALCNT'].iloc[i-20:i-10].values, 1)[0]
                    if prev_slope < 0 and slope_10 > 5:
                        df.loc[i, 'TREND_REVERSAL'] = 1  # 하락에서 상승 전환
                    elif prev_slope > 0 and slope_10 < -5:
                        df.loc[i, 'TREND_REVERSAL'] = -1  # 상승에서 하락 전환
                    else:
                        df.loc[i, 'TREND_REVERSAL'] = 0
        
        # 특성 컬럼 목록
        self.feature_columns = [
            'TOTALCNT', 'M14AM14B', 'M14AM14BSUM', 'M14AM10A', 'M14AM16',
            'RATIO', 'GOLDEN', 
            'SIGNAL_ZONE', 'PRE_SIGNAL', 'DANGER_ZONE',
            'NEAR_SIGNAL', 'SIGNAL_TO_DANGER_RISK',  # 전이 특성
            'SIGNAL_DURATION',
            'PATTERN', 'TREND', 'TREND_STRENGTH',
            'MA_3', 'MA_5', 'MA_10', 'MA_20', 'MA_30',
            'STD_3', 'STD_5', 'STD_10', 'STD_20', 'STD_30',
            'M14B_MA_3', 'M14B_MA_5', 'M14B_MA_10', 'M14B_MA_20', 'M14B_MA_30',
            'CHANGE_1', 'CHANGE_3', 'CHANGE_5', 'CHANGE_10', 'CHANGE_15',
            'M14B_CHANGE_1', 'M14B_CHANGE_3', 'M14B_CHANGE_5', 'M14B_CHANGE_10', 'M14B_CHANGE_15',
            'ACCEL_1', 'ACCEL_3', 'ACCEL_5', 'ACCEL_10', 'ACCEL_15',
            'RISE_COUNT', 'RISE_STREAK',
            'SPIKE_10', 'SPIKE_5',
            'BB_UPPER_20', 'BB_LOWER_20', 'BB_WIDTH_20', 'BB_POSITION_20',
            'RSI_14', 'RSI_7',
            'MACD', 'MACD_SIGNAL', 'MACD_DIFF'
        ]
        
        # NaN 처리
        df = df.fillna(0)
        
        print(f"✅ 고급 특성 생성 완료: {len(self.feature_columns)}개")
        return df
    
    def create_transition_sequences(self, df):
        """전이 패턴 중심 시퀀스 생성 - 상승/하락 구분"""
        print("\n🔄 전이 패턴 시퀀스 생성 중...")
        
        X_data = df[self.feature_columns].values
        y_data = df['TOTALCNT'].shift(-self.pred_len).values
        
        X, y = [], []
        transition_X, transition_y = [], []  # 전이 패턴
        danger_X, danger_y = [], []  # 위험 패턴
        
        # 상승/하락 구분 카운터
        rise_signal_to_danger = 0
        rise_in_danger = 0
        fall_from_danger = 0
        
        for i in range(len(df) - self.seq_len - self.pred_len):
            seq_X = X_data[i:i+self.seq_len]
            target = y_data[i+self.seq_len-1]
            
            if np.isnan(target):
                continue
                
            current = df['TOTALCNT'].iloc[i+self.seq_len-1]
            
            X.append(seq_X)
            y.append(target)
            
            # ⭐ 상승/하락 구분
            is_rising = target > current  # 상승 중
            change_amount = target - current
            
            # 전이 패턴 탐지 및 오버샘플링
            
            # 1. 신호→위험 상승 전이 (가장 중요!)
            if 1650 <= current <= 1699 and target >= 1700 and is_rising:
                # 상승하여 위험구간 진입 - 20배 오버샘플링
                for _ in range(20):
                    transition_X.append(seq_X)
                    transition_y.append(target)
                rise_signal_to_danger += 1
                print(f"  🚨 상승 전이: {current:.0f} ↗ {target:.0f} (+{change_amount:.0f})")
            
            # 2. 위험구간 내 상승 지속
            elif current >= 1700 and target >= 1700 and is_rising:
                # 위험구간에서 계속 상승 - 15배
                for _ in range(15):
                    danger_X.append(seq_X)
                    danger_y.append(target)
                rise_in_danger += 1
            
            # 3. 위험구간에서 하락 (안정화)
            elif current >= 1700 and target < current:
                # 하락 중 - 3배만 (덜 중요)
                for _ in range(3):
                    transition_X.append(seq_X)
                    transition_y.append(target)
                fall_from_danger += 1
            
            # 4. 급상승 패턴 (50 이상)
            elif change_amount >= 50 and is_rising:
                # 어느 구간이든 급상승 - 12배
                for _ in range(12):
                    transition_X.append(seq_X)
                    transition_y.append(target)
            
            # 5. 신호구간 내 상승
            elif 1651 <= current <= 1699 and is_rising and change_amount >= 20:
                # 신호구간 내 상승 - 8배
                for _ in range(8):
                    transition_X.append(seq_X)
                    transition_y.append(target)
            
            # 6. 1680 이상에서 상승 (위험 임박)
            elif current >= 1680 and is_rising:
                # 위험 임박 상승 - 10배
                for _ in range(10):
                    danger_X.append(seq_X)
                    danger_y.append(target)
        
        # 오버샘플링 데이터 추가
        if transition_X:
            X.extend(transition_X)
            y.extend(transition_y)
            print(f"  🔥 전이 패턴 {len(transition_X)}개 추가")
        
        if danger_X:
            X.extend(danger_X)
            y.extend(danger_y)
            print(f"  🔥 위험 패턴 {len(danger_X)}개 추가")
        
        print(f"\n  📊 패턴 통계:")
        print(f"    신호→위험 상승 전이: {rise_signal_to_danger}건")
        print(f"    위험구간 내 상승: {rise_in_danger}건")
        print(f"    위험구간에서 하락: {fall_from_danger}건")
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        
        print(f"  총 시퀀스: {X.shape[0]}개")
        
        # 셔플
        shuffle_idx = np.arange(len(X))
        np.random.shuffle(shuffle_idx)
        X = X[shuffle_idx]
        y = y[shuffle_idx]
        
        return self.split_and_scale(X, y)
    
    def split_and_scale(self, X, y):
        """데이터 분할 및 스케일링"""
        # 분할
        train_size = int(len(X) * 0.7)
        val_size = int(len(X) * 0.15)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]
        X_test = X[train_size+val_size:]
        y_test = y[train_size+val_size:]
        
        # 스케일링
        X_train_flat = X_train.reshape(-1, X_train.shape[-1])
        self.scaler_X.fit(X_train_flat)
        
        X_train_scaled = self.scaler_X.transform(X_train_flat).reshape(X_train.shape)
        X_val_scaled = self.scaler_X.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        X_test_scaled = self.scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        
        self.scaler_y.fit(y_train.reshape(-1, 1))
        y_train_scaled = self.scaler_y.transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = self.scaler_y.transform(y_val.reshape(-1, 1)).flatten()
        y_test_scaled = self.scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        
        # 구간별 통계
        self._print_statistics(y_train, y_val, y_test)
        
        # 레이블 생성
        y_train_level = self._create_level_labels(y_train)
        y_val_level = self._create_level_labels(y_val)
        y_test_level = self._create_level_labels(y_test)
        
        y_train_danger = (y_train >= self.DANGER_MIN).astype(int)
        y_val_danger = (y_val >= self.DANGER_MIN).astype(int)
        y_test_danger = (y_test >= self.DANGER_MIN).astype(int)
        
        # 스케일러 저장
        os.makedirs('scalers_v7', exist_ok=True)
        with open('scalers_v7/scaler_X.pkl', 'wb') as f:
            pickle.dump(self.scaler_X, f)
        with open('scalers_v7/scaler_y.pkl', 'wb') as f:
            pickle.dump(self.scaler_y, f)
        
        return (X_train_scaled, y_train_scaled, y_train, y_train_level, y_train_danger), \
               (X_val_scaled, y_val_scaled, y_val, y_val_level, y_val_danger), \
               (X_test_scaled, y_test_scaled, y_test, y_test_level, y_test_danger)
    
    def _create_level_labels(self, y):
        """구간 레이블 생성"""
        labels = np.zeros(len(y), dtype=int)
        labels[(y >= self.NORMAL_MAX) & (y <= self.CAUTION_MAX)] = 1
        labels[(y >= self.SIGNAL_MIN) & (y <= self.SIGNAL_MAX)] = 2
        labels[y >= self.DANGER_MIN] = 3
        return labels
    
    def _print_statistics(self, y_train, y_val, y_test):
        """통계 출력"""
        print(f"\n📊 데이터 분할:")
        print(f"  Train: {len(y_train):,}개")
        print(f"  Val: {len(y_val):,}개")
        print(f"  Test: {len(y_test):,}개")
        
        for name, data in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
            signal = ((data >= self.SIGNAL_MIN) & (data <= self.SIGNAL_MAX)).sum()
            danger = (data >= self.DANGER_MIN).sum()
            print(f"\n{name} - 신호: {signal}개 ({signal/len(data)*100:.1f}%), "
                  f"위험: {danger}개 ({danger/len(data)*100:.1f}%)")

# ========================================
# 개선된 모델 아키텍처
# ========================================
class TransformerBlock(layers.Layer):
    """Transformer 블록 - 시계열 장기 의존성 학습"""
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TemporalConvBlock(layers.Layer):
    """Temporal Convolution - 다중 스케일 시간 패턴"""
    def __init__(self, filters, kernel_sizes=[3, 5, 7]):
        super().__init__()
        self.convs = []
        for k in kernel_sizes:
            self.convs.append(Conv1D(filters, k, padding='same', activation='relu'))
        self.batch_norm = BatchNormalization()
        self.dropout = Dropout(0.2)
        
    def call(self, inputs, training):
        outputs = []
        for conv in self.convs:
            outputs.append(conv(inputs))
        x = tf.concat(outputs, axis=-1)
        x = self.batch_norm(x)
        return self.dropout(x, training=training)

def build_extremenet_v7(input_shape, num_features):
    """ExtremeNet V7.0 - 위험구간 예측 강화 모델"""
    
    inputs = Input(shape=input_shape, name='input')
    
    # ========== 1. Feature Extraction ==========
    # Temporal Convolution (다중 스케일 패턴)
    temporal_conv = TemporalConvBlock(64)(inputs)
    
    # ========== 2. 다중 경로 처리 ==========
    
    # 경로 1: Transformer (장기 의존성)
    transformer = TransformerBlock(input_shape[-1], 8, 128)(inputs)
    transformer = TransformerBlock(input_shape[-1], 8, 128)(transformer)
    transformer_pool = GlobalAveragePooling1D()(transformer)
    
    # 경로 2: Bidirectional LSTM (양방향 시계열)
    lstm = Bidirectional(LSTM(128, return_sequences=True))(temporal_conv)
    lstm = Bidirectional(LSTM(64))(lstm)
    lstm = Dropout(0.3)(lstm)
    
    # 경로 3: CNN + Attention (중요 패턴)
    cnn = Conv1D(128, 5, activation='relu')(inputs)
    cnn = BatchNormalization()(cnn)
    cnn = Conv1D(64, 3, activation='relu')(cnn)
    attention = MultiHeadAttention(num_heads=4, key_dim=64)(cnn, cnn)
    cnn_pool = GlobalMaxPooling1D()(attention)
    
    # 경로 4: GRU (최근 패턴)
    recent = Lambda(lambda x: x[:, -30:, :])(inputs)
    gru = GRU(64, return_sequences=True)(recent)
    gru = GRU(32)(gru)
    
    # 경로 5: 통계 특성 (직접 추출)
    stats = Lambda(lambda x: tf.concat([
        tf.reduce_mean(x, axis=1, keepdims=True),
        tf.reduce_max(x, axis=1, keepdims=True),
        tf.reduce_min(x, axis=1, keepdims=True),
        tf.math.reduce_std(x, axis=1, keepdims=True)
    ], axis=1))(inputs)
    stats = Flatten()(stats)
    
    # ========== 3. 특성 통합 ==========
    merged = Concatenate()([transformer_pool, lstm, cnn_pool, gru, stats])
    merged = BatchNormalization()(merged)
    merged = Dropout(0.3)(merged)
    
    # ========== 4. 예측 헤드 ==========
    
    # 공통 특성
    x = Dense(256, activation='relu')(merged)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # 메인 예측 (회귀)
    main_branch = Dense(64, activation='relu')(x)
    output_reg = Dense(1, name='regression')(main_branch)
    
    # 구간 분류 (4구간)
    class_branch = Dense(32, activation='relu')(x)
    output_cls = Dense(4, activation='softmax', name='classification')(class_branch)
    
    # 위험 감지 (이진 분류)
    danger_branch = Dense(32, activation='relu')(x)
    output_danger = Dense(1, activation='sigmoid', name='danger')(danger_branch)
    
    model = Model(inputs, [output_reg, output_cls, output_danger], 
                  name='ExtremeNet_V7')
    
    return model

# ========================================
# 개선된 손실함수
# ========================================
class AdaptiveLoss(tf.keras.losses.Loss):
    """적응형 손실함수 - 구간별 동적 가중치"""
    
    def __init__(self, processor):
        super().__init__()
        self.processor = processor
        
    def call(self, y_true, y_pred):
        # 기본 MAE
        mae = tf.abs(y_true - y_pred)
        
        # 동적 가중치 계산
        weights = tf.ones_like(y_true)
        
        # 값 범위를 실제 스케일로 변환 (역변환 시뮬레이션)
        y_true_rescaled = y_true * 1000 + 1000  # 대략적 변환
        
        # 신호구간 (1651~1699)
        signal_mask = tf.logical_and(
            y_true_rescaled >= 1651,
            y_true_rescaled <= 1699
        )
        weights = tf.where(signal_mask, 5.0, weights)
        
        # 위험구간 (1700+)
        danger_mask = y_true_rescaled >= 1700
        weights = tf.where(danger_mask, 10.0, weights)
        
        # 전이 구간 (1680~1720) 특별 가중치
        transition_mask = tf.logical_and(
            y_true_rescaled >= 1680,
            y_true_rescaled <= 1720
        )
        weights = tf.where(transition_mask, 15.0, weights)
        
        # 오차가 큰 경우 추가 패널티
        large_error_mask = mae > 0.1  # 스케일된 값 기준
        weights = tf.where(large_error_mask, weights * 1.5, weights)
        
        weighted_mae = mae * weights
        return tf.reduce_mean(weighted_mae)

# ========================================
# 학습 관리
# ========================================
class TrainingManager:
    def __init__(self, processor):
        self.processor = processor
        self.best_score = float('inf')
        
    def train(self, train_data, val_data, test_data):
        """모델 학습"""
        X_train, y_train_scaled, y_train, y_train_level, y_train_danger = train_data
        X_val, y_val_scaled, y_val, y_val_level, y_val_danger = val_data
        X_test, y_test_scaled, y_test, y_test_level, y_test_danger = test_data
        
        # 모델 생성
        with strategy.scope():
            model = build_extremenet_v7(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                num_features=X_train.shape[2]
            )
            
            # 다중 옵티마이저 (학습률 스케줄링)
            initial_lr = 0.001
            lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
                initial_lr, 1000, alpha=0.1
            )
            
            model.compile(
                optimizer=Adam(learning_rate=lr_schedule),
                loss={
                    'regression': AdaptiveLoss(self.processor),
                    'classification': 'sparse_categorical_crossentropy',
                    'danger': 'binary_crossentropy'
                },
                loss_weights={
                    'regression': 0.5,
                    'classification': 0.2,
                    'danger': 0.3  # 위험 감지 중요도 증가
                },
                metrics={
                    'regression': 'mae',
                    'classification': 'accuracy',
                    'danger': ['accuracy', tf.keras.metrics.AUC(name='auc')]
                }
            )
        
        print("\n" + "="*80)
        print("🔥 ExtremeNet V7.0 학습 시작")
        print("="*80)
        
        # 콜백
        callbacks = [
            ModelCheckpoint(
                'models_v7/best_model.keras',
                save_best_only=True,
                monitor='val_regression_mae',
                mode='min',
                verbose=1
            ),
            EarlyStopping(
                patience=30,
                restore_best_weights=True,
                monitor='val_regression_mae',
                mode='min',
                verbose=1
            ),
            ReduceLROnPlateau(
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                monitor='val_regression_mae',
                mode='min',
                verbose=1
            ),
            LambdaCallback(
                on_epoch_end=lambda epoch, logs: 
                self._evaluate_danger_zone(model, X_val, y_val, epoch)
            )
        ]
        
        # 학습
        os.makedirs('models_v7', exist_ok=True)
        
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
            epochs=150,  # 더 많은 에포크
            batch_size=64,
            callbacks=callbacks,
            verbose=1
        )
        
        # 최종 평가
        self._final_evaluation(model, X_test, y_test_scaled, y_test)
        
        # 모델 저장
        model.save('models_v7/extremenet_v7_final.keras')
        print(f"\n💾 최종 모델: models_v7/extremenet_v7_final.keras")
        
        return model, history
    
    def _evaluate_danger_zone(self, model, X_val, y_val, epoch):
        """위험구간 평가"""
        if epoch % 5 == 0:
            preds = model.predict(X_val, verbose=0)
            y_pred = self.processor.scaler_y.inverse_transform(
                preds[0].reshape(-1, 1)
            ).flatten()
            
            # 위험구간 평가
            danger_idx = y_val >= 1700
            if danger_idx.sum() > 0:
                danger_mae = mean_absolute_error(y_val[danger_idx], y_pred[danger_idx])
                print(f"\n  🔴 에포크 {epoch} - 위험구간 MAE: {danger_mae:.2f}")
                
                # 전이 패턴 평가
                transition_idx = (y_val >= 1680) & (y_val <= 1720)
                if transition_idx.sum() > 0:
                    trans_mae = mean_absolute_error(y_val[transition_idx], y_pred[transition_idx])
                    print(f"  🎯 전이구간(1680~1720) MAE: {trans_mae:.2f}")
    
    def _final_evaluation(self, model, X_test, y_test_scaled, y_test):
        """최종 평가"""
        print("\n" + "="*80)
        print("📊 최종 평가")
        print("="*80)
        
        preds = model.predict(X_test, verbose=0)
        y_pred = self.processor.scaler_y.inverse_transform(
            preds[0].reshape(-1, 1)
        ).flatten()
        
        # 전체 메트릭
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n📈 전체 성능:")
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R²: {r2:.4f}")
        
        # 구간별 상세 평가
        for name, min_val, max_val in [
            ("신호구간(1651~1699)", 1651, 1699),
            ("전이구간(1680~1720)", 1680, 1720),
            ("위험구간(1700+)", 1700, float('inf'))
        ]:
            if max_val == float('inf'):
                idx = y_test >= min_val
            else:
                idx = (y_test >= min_val) & (y_test <= max_val)
            
            if idx.sum() > 0:
                zone_mae = mean_absolute_error(y_test[idx], y_pred[idx])
                zone_acc = np.mean(np.abs(y_test[idx] - y_pred[idx]) < 30)
                
                print(f"\n{'🟠' if '신호' in name else '🔴'} {name}:")
                print(f"  샘플: {idx.sum()}개")
                print(f"  MAE: {zone_mae:.2f}")
                print(f"  정확도(±30): {zone_acc*100:.1f}%")

# ========================================
# 개선된 실시간 예측
# ========================================
class EnhancedPredictor:
    """상승/하락 구분 강화 예측기"""
    
    def __init__(self, model_path='models_v7/extremenet_v7_final.keras'):
        print("\n🚀 예측기 로딩 중...")
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={'AdaptiveLoss': AdaptiveLoss}
        )
        
        with open('scalers_v7/scaler_X.pkl', 'rb') as f:
            self.scaler_X = pickle.load(f)
        with open('scalers_v7/scaler_y.pkl', 'rb') as f:
            self.scaler_y = pickle.load(f)
        
        self.processor = EnhancedDataProcessor()
        print("✅ 예측기 준비 완료")
    
    def predict_with_trend(self, df_100min):
        """100분 데이터로 10분 후 예측 - 상승/하락 상세 분석"""
        
        # 현재값
        current_value = df_100min['TOTALCNT'].iloc[-1]
        recent_values = df_100min['TOTALCNT'].tail(10).values
        
        # 다중 시간대 추세 분석
        trend_5min = self._calculate_trend(df_100min['TOTALCNT'].tail(5).values)
        trend_10min = self._calculate_trend(df_100min['TOTALCNT'].tail(10).values)
        trend_20min = self._calculate_trend(df_100min['TOTALCNT'].tail(20).values)
        
        # 특성 생성
        df_features = self.processor.create_advanced_features(df_100min)
        X = df_features[self.processor.feature_columns].values
        X_scaled = self.scaler_X.transform(X.reshape(-1, len(self.processor.feature_columns)))
        X_scaled = X_scaled.reshape(1, 100, len(self.processor.feature_columns))
        
        # 예측
        preds = self.model.predict(X_scaled, verbose=0)
        y_pred_scaled = preds[0][0, 0]
        y_pred = self.scaler_y.inverse_transform([[y_pred_scaled]])[0, 0]
        
        # 변화량 계산
        change = y_pred - current_value
        change_rate = (change / current_value) * 100
        
        # 위험도 점수 (0~100)
        danger_prob = float(preds[2][0, 0]) * 100
        
        # 상세 진단
        diagnosis = self._diagnose_situation(
            current_value, y_pred, change, 
            trend_5min, trend_10min, trend_20min,
            danger_prob
        )
        
        return {
            'current': float(current_value),
            'prediction': float(y_pred),
            'change': float(change),
            'change_rate': float(change_rate),
            'trend_5min': trend_5min,
            'trend_10min': trend_10min,
            'trend_20min': trend_20min,
            'danger_probability': danger_prob,
            'level': diagnosis['level'],
            'status': diagnosis['status'],
            'message': diagnosis['message'],
            'action': diagnosis['action'],
            'risk_score': diagnosis['risk_score']
        }
    
    def _calculate_trend(self, values):
        """추세 계산"""
        if len(values) < 2:
            return {'direction': 'flat', 'strength': 0}
        
        slope = np.polyfit(range(len(values)), values, 1)[0]
        
        if slope > 5:
            direction = 'rising'
            strength = min(slope / 10, 10)  # 0~10 스케일
        elif slope < -5:
            direction = 'falling'
            strength = min(abs(slope) / 10, 10)
        else:
            direction = 'flat'
            strength = 0
            
        return {'direction': direction, 'strength': strength}
    
    def _diagnose_situation(self, current, predicted, change, 
                           trend_5, trend_10, trend_20, danger_prob):
        """상황 진단 - 상승/하락 구분"""
        
        diagnosis = {
            'level': '',
            'status': '',
            'message': '',
            'action': '',
            'risk_score': 0
        }
        
        # 현재 구간 판정
        if current < 1400:
            current_zone = 'normal'
        elif current < 1651:
            current_zone = 'caution'
        elif current < 1700:
            current_zone = 'signal'
        else:
            current_zone = 'danger'
        
        # 예측 구간 판정
        if predicted < 1400:
            pred_zone = 'normal'
        elif predicted < 1651:
            pred_zone = 'caution'
        elif predicted < 1700:
            pred_zone = 'signal'
        else:
            pred_zone = 'danger'
        
        # 상승/하락 종합 판단
        is_rising = change > 0
        is_accelerating = (trend_5['strength'] > trend_10['strength'] and 
                          trend_5['direction'] == 'rising')
        
        # ⭐ 핵심 케이스별 진단
        
        # 케이스 1: 신호→위험 상승 전이 (가장 위험!)
        if current_zone == 'signal' and pred_zone == 'danger' and is_rising:
            diagnosis['level'] = '🔴🔴🔴 극도위험'
            diagnosis['status'] = '급증 진행'
            diagnosis['message'] = f'1700+ 돌파 예상! 현재 {current:.0f} → {predicted:.0f} (+{change:.0f})'
            diagnosis['action'] = '🚨 즉시 비상대응! 물류 급증 임박'
            diagnosis['risk_score'] = 100
        
        # 케이스 2: 위험구간 내 계속 상승
        elif current_zone == 'danger' and pred_zone == 'danger' and is_rising:
            diagnosis['level'] = '🔴🔴 매우위험'
            diagnosis['status'] = '위험 상승'
            diagnosis['message'] = f'위험구간 내 상승 지속 {current:.0f} → {predicted:.0f}'
            diagnosis['action'] = '⚠️ 긴급대응 유지'
            diagnosis['risk_score'] = 90
        
        # 케이스 3: 위험구간에서 하락 (안정화)
        elif current_zone == 'danger' and not is_rising:
            diagnosis['level'] = '🟠 위험완화'
            diagnosis['status'] = '하락 전환'
            diagnosis['message'] = f'피크 통과, 하락 중 {current:.0f} → {predicted:.0f} ({change:.0f})'
            diagnosis['action'] = '📉 모니터링 지속, 안정화 확인'
            diagnosis['risk_score'] = 60
        
        # 케이스 4: 신호구간 내 급상승
        elif current_zone == 'signal' and is_rising and change >= 30:
            diagnosis['level'] = '🟠🟠 주의'
            diagnosis['status'] = '급상승 감지'
            diagnosis['message'] = f'신호구간 급상승 {current:.0f} → {predicted:.0f} (+{change:.0f})'
            diagnosis['action'] = '⚠️ 사전 대응 준비'
            diagnosis['risk_score'] = 75
        
        # 케이스 5: 가속 상승 (어느 구간이든)
        elif is_accelerating and change >= 20:
            diagnosis['level'] = '🟡🟡 경계'
            diagnosis['status'] = '가속 상승'
            diagnosis['message'] = f'상승 가속도 증가 중 (+{change:.0f})'
            diagnosis['action'] = '📈 면밀한 모니터링 필요'
            diagnosis['risk_score'] = 65
        
        # 케이스 6: 1680 이상에서 상승 (위험 임박)
        elif current >= 1680 and is_rising:
            diagnosis['level'] = '🟠🟠🟠 고위험'
            diagnosis['status'] = '위험 임박'
            diagnosis['message'] = f'1700 임박! {current:.0f} → {predicted:.0f}'
            diagnosis['action'] = '🔔 대응팀 대기'
            diagnosis['risk_score'] = 85
        
        # 케이스 7: 안정 구간
        else:
            if pred_zone == 'normal':
                diagnosis['level'] = '🟢 정상'
                diagnosis['status'] = '안정'
                diagnosis['message'] = f'정상 범위 유지 {current:.0f} → {predicted:.0f}'
                diagnosis['action'] = '✓ 정상 운영'
                diagnosis['risk_score'] = 20
            else:
                diagnosis['level'] = '🟡 주의'
                diagnosis['status'] = '관찰 필요'
                diagnosis['message'] = f'변동 관찰 {current:.0f} → {predicted:.0f}'
                diagnosis['action'] = '👀 지속 모니터링'
                diagnosis['risk_score'] = 40
        
        # 위험도 점수 보정
        if danger_prob > 70:
            diagnosis['risk_score'] = min(diagnosis['risk_score'] + 10, 100)
        
        return diagnosis
def main():
    # 데이터 로드
    data_path = None
    for path in ['data/20240201_TO_202507281705.csv', 
                 '20240201_TO_202507281705.csv']:
        if os.path.exists(path):
            data_path = path
            break
    
    if not data_path:
        print("❌ 데이터 파일 없음!")
        return None
    
    # 데이터 처리
    processor = EnhancedDataProcessor()
    
    df = pd.read_csv(data_path)
    df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), 
                                   format='%Y%m%d%H%M', errors='coerce')
    df = df.sort_values('CURRTIME').reset_index(drop=True)
    df = df[df['TOTALCNT'] > 0].reset_index(drop=True)
    
    print(f"✅ 데이터: {len(df):,}개 로드")
    
    # 고급 특성 생성
    df = processor.create_advanced_features(df)
    
    # 전이 패턴 시퀀스 생성
    train_data, val_data, test_data = processor.create_transition_sequences(df)
    
    # 학습
    trainer = TrainingManager(processor)
    model, history = trainer.train(train_data, val_data, test_data)
    
    print("\n✅ 학습 완료!")
    print("🔥 ExtremeNet V7.0 - 위험구간 예측 강화 완료!")
    
    return model, history

if __name__ == "__main__":
    model, history = main()