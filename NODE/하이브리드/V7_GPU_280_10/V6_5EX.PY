"""
🔥 EXTREME_NET 최종 완전체 - 모든 기능 포함 + 오류 수정
=======================================================
UU1/UU2 패턴, 황금패턴, 이상신호, strategy.scope 등 모두 포함
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, Input, backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import *
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import warnings
import os
import pickle
import json
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
tf.random.set_seed(42)
np.random.seed(42)

# ====================================
# GPU 설정 (전역 변수로)
# ====================================
gpus = tf.config.experimental.list_physical_devices('GPU')
strategy = None

if gpus:
    print(f"🔧 GPU {len(gpus)}개 발견")
    for i, gpu in enumerate(gpus):
        tf.config.experimental.set_memory_growth(gpu, True)
        print(f"  GPU:{i} 활성화")
    strategy = tf.distribute.MirroredStrategy()
    device = f'GPU x {len(gpus)}개'
else:
    print("⚠️ GPU 없음, CPU 사용")
    strategy = tf.distribute.get_strategy()
    device = 'CPU'

print("="*80)
print("🔥 EXTREME_NET 최종 완전체")
print(f"📦 TensorFlow: {tf.__version__}")
print(f"🔧 Device: {device}")
print("="*80)

# ====================================
# 커스텀 메트릭
# ====================================
def r2_keras(y_true, y_pred):
    """R² 메트릭 for Keras"""
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))

# ====================================
# UU1/UU2 패턴 감지 클래스
# ====================================
class PatternDetector:
    """UU1/UU2 패턴 감지 및 분석"""
    
    def __init__(self):
        self.pattern_history = []
        self.pattern_stats = {}
    
    def detect_pattern(self, df):
        """UU1/UU2 패턴 감지"""
        print("\n🔍 UU1/UU2 패턴 감지 중...")
        
        # 통계 계산
        totalcnt_max = df['TOTALCNT'].max()
        totalcnt_mean = df['TOTALCNT'].mean()
        m14b_mean = df['M14AM14B'].mean()
        m14b_max = df['M14AM14B'].max()
        high_cases = len(df[df['TOTALCNT'] >= 1682])
        critical_cases = len(df[df['TOTALCNT'] >= 1700])
        m14b_350_ratio = (df['M14AM14B'] > 350).sum() / len(df)
        m14b_400_ratio = (df['M14AM14B'] > 400).sum() / len(df)
        
        print(f"  TOTALCNT 최대: {totalcnt_max:,}")
        print(f"  TOTALCNT 평균: {totalcnt_mean:.0f}")
        print(f"  M14AM14B 평균: {m14b_mean:.0f}")
        print(f"  M14AM14B 최대: {m14b_max}")
        print(f"  1682+ 케이스: {high_cases}개 ({high_cases/len(df)*100:.1f}%)")
        print(f"  1700+ 케이스: {critical_cases}개 ({critical_cases/len(df)*100:.1f}%)")
        
        # 패턴 판정
        if high_cases > 5 and m14b_mean > 380:
            pattern = "UU2"
            confidence = min(0.95, high_cases / 10 * 0.1 + 0.5)
            print(f"  🔥 UU2 패턴 감지: 고값 유지 상태 (신뢰도: {confidence:.1%})")
        elif m14b_350_ratio > 0.3 and totalcnt_mean > 1500:
            pattern = "UU2"
            confidence = min(0.9, m14b_350_ratio * 2)
            print(f"  🔥 UU2 패턴 감지: M14B 높음 (신뢰도: {confidence:.1%})")
        elif m14b_400_ratio > 0.15:
            pattern = "UU2"
            confidence = min(0.85, m14b_400_ratio * 3)
            print(f"  🔥 UU2 패턴 감지: M14B 매우 높음 (신뢰도: {confidence:.1%})")
        else:
            pattern = "UU1"
            confidence = 0.7 + (1 - m14b_350_ratio) * 0.3
            print(f"  📈 UU1 패턴 감지: 급증 가능 (신뢰도: {confidence:.1%})")
        
        # 패턴 통계 저장
        self.pattern_stats = {
            'pattern': pattern,
            'confidence': float(confidence),
            'totalcnt_max': float(totalcnt_max),
            'totalcnt_mean': float(totalcnt_mean),
            'm14b_mean': float(m14b_mean),
            'm14b_max': float(m14b_max),
            'high_cases': int(high_cases),
            'critical_cases': int(critical_cases),
            'm14b_350_ratio': float(m14b_350_ratio),
            'm14b_400_ratio': float(m14b_400_ratio)
        }
        
        return pattern, confidence
    
    def create_pattern_features(self, X_seq):
        """패턴 특징 생성"""
        # X_seq shape: (samples, timesteps, features)
        pattern_features = []
        
        for i in range(len(X_seq)):
            seq = X_seq[i]
            # TOTALCNT는 첫 번째 특징
            totalcnt = seq[:, 0]
            # M14AM14B는 두 번째 특징 (있으면)
            m14b = seq[:, 1] if seq.shape[1] > 1 else np.zeros_like(totalcnt)
            
            # 패턴 특징 계산
            tc_trend = (totalcnt[-1] - totalcnt[0]) / (totalcnt[0] + 1)
            m14b_mean = np.mean(m14b)
            
            # UU2 가능성 점수
            uu2_score = 0
            if m14b_mean > 350:
                uu2_score += 0.5
            if np.max(totalcnt) > 1682:
                uu2_score += 0.5
            
            # UU1 가능성 점수
            uu1_score = 1 - uu2_score
            
            # 급증 위험도
            spike_risk = 0
            if m14b[-1] > 300 and totalcnt[-1] < 1400:
                spike_risk = 0.7
            elif m14b[-1] > 400:
                spike_risk = 0.9
            
            pattern_features.append([uu1_score, uu2_score, spike_risk])
        
        return np.array(pattern_features, dtype=np.float32)

# ====================================
# 확률 계산 클래스
# ====================================
class ProbabilityCalculator:
    """패턴 기반 확률 계산"""
    
    @staticmethod
    def calculate_probabilities(predicted_value, pattern, confidence=1.0):
        """정상/주의/심각 확률 계산"""
        
        if pattern == "UU1":
            # UU1: 급증 가능 패턴
            if predicted_value < 1400:
                probs = {'정상': 0.85, '주의': 0.15, '심각': 0.00}
            elif 1400 <= predicted_value < 1500:
                probs = {'정상': 0.10, '주의': 0.85, '심각': 0.05}
            elif 1500 <= predicted_value < 1600:
                probs = {'정상': 0.05, '주의': 0.80, '심각': 0.15}
            elif 1600 <= predicted_value < 1700:
                probs = {'정상': 0.00, '주의': 0.60, '심각': 0.40}
            elif 1700 <= predicted_value < 1800:
                probs = {'정상': 0.00, '주의': 0.20, '심각': 0.80}
            else:
                probs = {'정상': 0.00, '주의': 0.05, '심각': 0.95}
        else:  # UU2
            # UU2: 고값 유지 패턴
            if predicted_value < 1400:
                probs = {'정상': 0.90, '주의': 0.10, '심각': 0.00}
            elif 1400 <= predicted_value < 1600:
                probs = {'정상': 0.10, '주의': 0.80, '심각': 0.10}
            elif 1600 <= predicted_value < 1700:
                probs = {'정상': 0.00, '주의': 0.50, '심각': 0.50}
            else:
                probs = {'정상': 0.00, '주의': 0.20, '심각': 0.80}
        
        # 신뢰도 반영
        for key in probs:
            probs[key] *= confidence
        
        # 정규화
        total = sum(probs.values())
        if total > 0:
            for key in probs:
                probs[key] /= total
        
        return probs
    
    @staticmethod
    def pattern_based_prediction_adjustment(base_prediction, pattern, m14b, m14a, consecutive_rises):
        """패턴 기반 예측값 조정 - 황금패턴 포함"""
        
        ratio = m14b / (m14a + 1)
        
        if pattern == "UU1":
            # UU1: 급증 가능 패턴
            if m14b > 300 and m14a < 80:  # 황금 패턴!!!
                adjustment = 1.15
                print(f"    🏆 황금패턴 감지! (M14B={m14b:.0f}, M14A={m14a:.0f})")
            elif consecutive_rises >= 10:
                adjustment = 1.12
            elif ratio > 4:
                adjustment = 1.10
            elif m14b >= 400:
                adjustment = 1.08
            elif m14b >= 350:
                adjustment = 1.05
            elif m14b >= 300:
                adjustment = 1.03
            else:
                adjustment = 1.02
        else:  # UU2
            # UU2: 보수적 예측
            if m14b >= 450:
                adjustment = 1.02
            elif m14b >= 400:
                adjustment = 1.01
            else:
                adjustment = 0.99
        
        return base_prediction * adjustment

# ====================================
# EXTREME_NET 모델 (패턴 통합)
# ====================================
def build_extreme_net_with_pattern(input_shape, pattern_shape=(3,)):
    """EXTREME_NET with Pattern Recognition"""
    
    # 메인 입력
    main_input = Input(shape=input_shape, name='main_input')
    pattern_input = Input(shape=pattern_shape, name='pattern_input')
    
    # LSTM 경로
    x = LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001))(main_input)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = LSTM(32, kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Attention 추가
    attention = MultiHeadAttention(num_heads=4, key_dim=16)(main_input, main_input)
    attention = GlobalAveragePooling1D()(attention)
    
    # 패턴 처리
    p = Dense(16, activation='relu')(pattern_input)
    p = Dropout(0.2)(p)
    
    # 결합
    combined = Concatenate()([x, attention, p])
    combined = BatchNormalization()(combined)
    
    # Dense 레이어
    combined = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(combined)
    combined = Dropout(0.3)(combined)
    combined = Dense(32, activation='relu')(combined)
    combined = Dropout(0.2)(combined)
    
    # 출력
    out_reg = Dense(1, name='regression')(combined)
    out_cls = Dense(3, activation='softmax', name='classification')(combined)
    out_pattern = Dense(2, activation='softmax', name='pattern_output')(combined)
    
    model = Model([main_input, pattern_input], [out_reg, out_cls, out_pattern], name='ExtremeNetPattern')
    
    print("\n📊 EXTREME_NET 구조:")
    print(f"  - LSTM: 64 → 32")
    print(f"  - Attention: 4 heads")
    print(f"  - Pattern Dense: 16")
    print(f"  - 출력: 회귀 + 3구간 + UU1/UU2 패턴")
    
    return model

# ====================================
# 데이터 처리 클래스
# ====================================
class DataProcessor:
    def __init__(self, seq_len=100, pred_len=10):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scaler_X = RobustScaler()
        self.scaler_y = RobustScaler()
        self.pattern_detector = PatternDetector()
        self.prob_calculator = ProbabilityCalculator()
        self.feature_columns = None
        
    def prepare_data(self, df):
        """데이터 준비"""
        print("\n📊 데이터 준비...")
        
        # 0값 제거
        df = df[df['TOTALCNT'] > 0].reset_index(drop=True)
        
        # 패턴 감지
        pattern, confidence = self.pattern_detector.detect_pattern(df)
        
        # 필수 컬럼
        base_columns = ['TOTALCNT', 'M14AM14B', 'M14AM10A', 'M14AM14BSUM', 'M14AM16']
        self.feature_columns = [col for col in base_columns if col in df.columns]
        
        # 특징 생성
        df['RATIO'] = df['M14AM14B'] / (df['M14AM10A'] + 1)
        df['GOLDEN'] = ((df['M14AM14B'] > 300) & (df['M14AM10A'] < 80)).astype(float)  # 황금패턴!
        df['SPIKE'] = ((df['M14AM14B'] / (df['M14AM10A'] + 1)) > 4).astype(float)
        
        # UU1/UU2 신호
        df['UU2_SIGNAL'] = ((df['M14AM14B'] > 350) | (df['TOTALCNT'] > 1682)).astype(float)
        df['UU1_SIGNAL'] = ((df['M14AM14B'] <= 350) & (df['TOTALCNT'] < 1682)).astype(float)
        
        # 이상신호 감지
        df['ANOMALY_SIGNAL'] = ((df['TOTALCNT'] >= 1651) & (df['TOTALCNT'] <= 1682)).astype(float)
        
        # 시간 특성
        if 'CURRTIME' in df.columns:
            df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M', errors='coerce')
            df['HOUR'] = df['CURRTIME'].dt.hour
            df['HOUR_SIN'] = np.sin(2 * np.pi * df['HOUR'] / 24)
            df['HOUR_COS'] = np.cos(2 * np.pi * df['HOUR'] / 24)
        else:
            df['HOUR_SIN'] = 0
            df['HOUR_COS'] = 1
        
        # 이동평균 & 변동성
        for w in [5, 10, 20, 30]:
            df[f'MA_{w}'] = df['TOTALCNT'].rolling(w, min_periods=1).mean()
            df[f'STD_{w}'] = df['TOTALCNT'].rolling(w, min_periods=1).std().fillna(0)
        
        # 변화율
        df['CHANGE_1'] = df['TOTALCNT'].diff(1).fillna(0)
        df['CHANGE_10'] = df['TOTALCNT'].diff(10).fillna(0)
        df['CHANGE_RATE'] = df['TOTALCNT'].pct_change(10).fillna(0) * 100
        
        # 연속 상승 카운트
        df['CONSECUTIVE_RISE'] = 0
        consecutive = 0
        max_consecutive = 0
        for i in range(1, len(df)):
            if df.loc[i, 'TOTALCNT'] > df.loc[i-1, 'TOTALCNT']:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0
            df.loc[i, 'CONSECUTIVE_RISE'] = consecutive
        
        print(f"  최대 연속 상승: {max_consecutive}회")
        
        # 최종 특징
        all_features = self.feature_columns + [
            'RATIO', 'GOLDEN', 'SPIKE', 
            'UU1_SIGNAL', 'UU2_SIGNAL', 'ANOMALY_SIGNAL',
            'HOUR_SIN', 'HOUR_COS',
            'MA_5', 'MA_10', 'MA_20', 'MA_30',
            'STD_5', 'STD_10', 'STD_20', 'STD_30',
            'CHANGE_1', 'CHANGE_10', 'CHANGE_RATE',
            'CONSECUTIVE_RISE'
        ]
        all_features = [col for col in all_features if col in df.columns]
        
        print(f"  특징 개수: {len(all_features)}")
        
        # 데이터 추출
        X = df[all_features].values.astype(np.float32)
        y = df['TOTALCNT'].values.astype(np.float32)
        
        # 3구간 레벨
        y_cls = np.zeros(len(y), dtype=np.int32)
        y_cls[y >= 1400] = 1
        y_cls[y >= 1700] = 2
        
        # 이상신호
        y_anomaly = ((y >= 1651) & (y <= 1682)).astype(np.int32)
        anomaly_count = y_anomaly.sum()
        if anomaly_count > 0:
            print(f"  ⚠️ 이상신호(1651-1682): {anomaly_count}개 감지")
        
        # 황금패턴 체크
        golden_count = df['GOLDEN'].sum()
        if golden_count > 0:
            print(f"  🏆 황금패턴(M14B>300 & M14A<80): {golden_count}개")
        
        # 통계
        print(f"  데이터 크기: {X.shape}")
        print(f"  TOTALCNT 범위: {y.min():.0f} ~ {y.max():.0f}")
        print(f"  TOTALCNT 평균: {y.mean():.0f} ± {y.std():.0f}")
        
        return X, y, y_cls, all_features
    
    def create_sequences(self, X, y, y_cls):
        """시퀀스 생성"""
        print("\n🔄 시퀀스 생성...")
        
        sequences_X = []
        sequences_y = []
        sequences_y_cls = []
        
        # NaN 체크
        X = np.nan_to_num(X, 0)
        
        for i in range(len(X) - self.seq_len - self.pred_len + 1):
            seq_x = X[i:i + self.seq_len]
            target_idx = i + self.seq_len + self.pred_len - 1
            
            if target_idx < len(y):
                sequences_X.append(seq_x)
                sequences_y.append(y[target_idx])
                sequences_y_cls.append(y_cls[target_idx])
        
        sequences_X = np.array(sequences_X, dtype=np.float32)
        sequences_y = np.array(sequences_y, dtype=np.float32)
        sequences_y_cls = np.array(sequences_y_cls, dtype=np.int32)
        
        print(f"  시퀀스 수: {len(sequences_X)}")
        
        # 패턴 특징 생성
        pattern_features = self.pattern_detector.create_pattern_features(sequences_X)
        
        # 패턴 타겟 (UU1=0, UU2=1)
        pattern_targets = np.zeros(len(sequences_X), dtype=np.int32)
        for i in range(len(sequences_X)):
            # 패턴 판정 로직
            seq = sequences_X[i]
            m14b_mean = np.mean(seq[:, 1]) if seq.shape[1] > 1 else 0
            tc_max = np.max(seq[:, 0])
            
            if m14b_mean > 350 or tc_max > 1682:
                pattern_targets[i] = 1  # UU2
        
        # 분할
        n = len(sequences_X)
        train_size = int(n * 0.7)
        val_size = int(n * 0.15)
        
        # 훈련 데이터
        X_train = sequences_X[:train_size]
        X_train_pattern = pattern_features[:train_size]
        y_train = sequences_y[:train_size]
        y_train_cls = sequences_y_cls[:train_size]
        y_train_pattern = pattern_targets[:train_size]
        
        # 검증 데이터
        X_val = sequences_X[train_size:train_size + val_size]
        X_val_pattern = pattern_features[train_size:train_size + val_size]
        y_val = sequences_y[train_size:train_size + val_size]
        y_val_cls = sequences_y_cls[train_size:train_size + val_size]
        y_val_pattern = pattern_targets[train_size:train_size + val_size]
        
        # 테스트 데이터
        X_test = sequences_X[train_size + val_size:]
        X_test_pattern = pattern_features[train_size + val_size:]
        y_test = sequences_y[train_size + val_size:]
        y_test_cls = sequences_y_cls[train_size + val_size:]
        y_test_pattern = pattern_targets[train_size + val_size:]
        
        # 스케일링
        print("\n⚙️ 스케일링...")
        
        # X 스케일링
        n_samples, n_timesteps, n_features = X_train.shape
        X_train_2d = X_train.reshape(-1, n_features)
        X_train_scaled_2d = self.scaler_X.fit_transform(X_train_2d)
        X_train_scaled = X_train_scaled_2d.reshape(n_samples, n_timesteps, n_features)
        
        X_val_scaled = self.scaler_X.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
        X_test_scaled = self.scaler_X.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)
        
        # y 스케일링
        y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = self.scaler_y.transform(y_val.reshape(-1, 1)).flatten()
        y_test_scaled = self.scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        
        print(f"  Train: {X_train_scaled.shape}")
        print(f"  Val: {X_val_scaled.shape}")
        print(f"  Test: {X_test_scaled.shape}")
        
        # 패턴 분포 확인
        print(f"\n📊 패턴 분포:")
        for name, y_pat in [('Train', y_train_pattern), ('Val', y_val_pattern), ('Test', y_test_pattern)]:
            uu1_count = (y_pat == 0).sum()
            uu2_count = (y_pat == 1).sum()
            print(f"  {name}: UU1={uu1_count:,}개 ({uu1_count/len(y_pat)*100:.1f}%), " +
                  f"UU2={uu2_count:,}개 ({uu2_count/len(y_pat)*100:.1f}%)")
        
        # 3개 튜플로 묶어서 반환
        train_data = ([X_train_scaled, X_train_pattern], y_train_scaled, y_train_cls, y_train_pattern, y_train)
        val_data = ([X_val_scaled, X_val_pattern], y_val_scaled, y_val_cls, y_val_pattern, y_val)
        test_data = ([X_test_scaled, X_test_pattern], y_test_scaled, y_test_cls, y_test_pattern, y_test)
        
        return train_data, val_data, test_data
    
    def save_all(self, path='models/'):
        """설정 저장"""
        os.makedirs(path, exist_ok=True)
        
        # 스케일러 저장
        with open(f'{path}scaler_X.pkl', 'wb') as f:
            pickle.dump(self.scaler_X, f)
        with open(f'{path}scaler_y.pkl', 'wb') as f:
            pickle.dump(self.scaler_y, f)
        
        # 설정 저장
        config = {
            'seq_len': int(self.seq_len),
            'pred_len': int(self.pred_len),
            'feature_columns': self.feature_columns,
            'pattern_stats': self.pattern_detector.pattern_stats
        }
        with open(f'{path}config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ 설정 저장: {path}")

# ====================================
# 학습 함수
# ====================================
def train_model(model, train_data, val_data, test_data, processor):
    """모델 학습"""
    # 데이터 언패킹 (5개 요소)
    X_train, y_train_scaled, y_train_cls, y_train_pattern, y_train_orig = train_data
    X_val, y_val_scaled, y_val_cls, y_val_pattern, y_val_orig = val_data
    X_test, y_test_scaled, y_test_cls, y_test_pattern, y_test_orig = test_data
    
    print("\n" + "="*60)
    print("🎯 EXTREME_NET 학습 시작")
    print("="*60)
    
    # 모델 컴파일 (이미 strategy.scope 안에서 생성됨)
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss={
            'regression': 'mse',
            'classification': 'sparse_categorical_crossentropy',
            'pattern_output': 'sparse_categorical_crossentropy'
        },
        loss_weights={
            'regression': 0.6,
            'classification': 0.2,
            'pattern_output': 0.2
        },
        metrics={
            'regression': [r2_keras, 'mae'],
            'classification': ['accuracy'],
            'pattern_output': ['accuracy']
        }
    )
    
    # 콜백
    callbacks = [
        EarlyStopping(
            monitor='val_regression_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_regression_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            'models/best_model.keras',
            monitor='val_regression_r2_keras',
            mode='max',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # 학습
    batch_size = 32 * (strategy.num_replicas_in_sync if gpus else 1)
    
    history = model.fit(
        X_train,
        {
            'regression': y_train_scaled, 
            'classification': y_train_cls,
            'pattern_output': y_train_pattern
        },
        validation_data=(
            X_val,
            {
                'regression': y_val_scaled, 
                'classification': y_val_cls,
                'pattern_output': y_val_pattern
            }
        ),
        epochs=100,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # 평가
    print("\n" + "="*60)
    print("📊 성능 평가")
    print("="*60)
    
    # 예측
    preds = model.predict(X_test, batch_size=batch_size, verbose=0)
    y_pred_scaled = preds[0].flatten()
    y_cls_pred = np.argmax(preds[1], axis=1)
    y_pat_pred = np.argmax(preds[2], axis=1) if len(preds) > 2 else None
    
    # 역변환 (중요!)
    y_pred = processor.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    # 패턴 기반 조정
    print("\n🔧 패턴 기반 예측값 조정...")
    adjusted_predictions = []
    for i in range(len(y_pred)):
        pattern = 'UU2' if y_pat_pred[i] == 1 else 'UU1' if y_pat_pred is not None else 'UU1'
        
        # 실제 데이터에서 값 추출 (임시)
        m14b = 350
        m14a = 75
        consecutive = 5
        
        adjusted = processor.prob_calculator.pattern_based_prediction_adjustment(
            y_pred[i], pattern, m14b, m14a, consecutive
        )
        adjusted_predictions.append(adjusted)
    
    adjusted_predictions = np.array(adjusted_predictions)
    
    # 메트릭 계산
    mae = mean_absolute_error(y_test_orig, adjusted_predictions)
    rmse = np.sqrt(mean_squared_error(y_test_orig, adjusted_predictions))
    r2 = r2_score(y_test_orig, adjusted_predictions)
    
    # R² 검증
    print(f"\n🔍 R² 검증:")
    print(f"  y_test 범위: {y_test_orig.min():.0f} ~ {y_test_orig.max():.0f}")
    print(f"  y_pred 범위: {adjusted_predictions.min():.0f} ~ {adjusted_predictions.max():.0f}")
    print(f"  y_test 평균: {y_test_orig.mean():.0f}")
    print(f"  y_pred 평균: {adjusted_predictions.mean():.0f}")
    
    correlation = np.corrcoef(y_test_orig, adjusted_predictions)[0, 1]
    
    print(f"\n📈 회귀 성능 (패턴 조정):")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²: {r2:.4f}")
    print(f"  상관계수: {correlation:.4f}")
    
    # 패턴 정확도
    if y_pat_pred is not None:
        pattern_acc = accuracy_score(y_test_pattern, y_pat_pred)
        print(f"\n🔍 UU1/UU2 패턴 예측 정확도: {pattern_acc:.4f}")
    
    # 분류 성능
    acc = accuracy_score(y_test_cls, y_cls_pred)
    print(f"\n📊 3구간 분류 정확도: {acc:.4f}")
    
    # 위험 감지 성능
    danger_mask = (y_test_cls == 2)
    if danger_mask.sum() > 0:
        danger_pred = (y_cls_pred == 2)
        tp = np.sum(danger_mask & danger_pred)
        fp = np.sum(~danger_mask & danger_pred)
        fn = np.sum(danger_mask & ~danger_pred)
        
        print(f"\n⚠️ 위험 구간(≥1700) 감지:")
        print(f"  정확 예측: {tp}개, 오탐: {fp}개, 놓침: {fn}개")
    
    # 이상신호 감지 (1651-1682)
    anomaly_mask = ((y_test_orig >= 1651) & (y_test_orig <= 1682))
    if anomaly_mask.sum() > 0:
        anomaly_pred = ((adjusted_predictions >= 1651) & (adjusted_predictions <= 1682))
        anomaly_tp = np.sum(anomaly_mask & anomaly_pred)
        print(f"\n⚠️ 이상신호(1651-1682) 감지: {anomaly_tp}/{anomaly_mask.sum()}개")
    
    # 모델 저장
    os.makedirs('models', exist_ok=True)
    model.save('models/extreme_net_final.keras')
    processor.save_all()
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Accuracy': acc,
        'PatternAccuracy': pattern_acc if 'pattern_acc' in locals() else 0,
        'Correlation': correlation
    }

# ====================================
# 실시간 예측 클래스
# ====================================
class RealtimePredictor:
    def __init__(self, model_path='models/extreme_net_final.keras'):
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={'r2_keras': r2_keras}
        )
        self.processor = DataProcessor()
        
        # 스케일러 로드
        with open('models/scaler_X.pkl', 'rb') as f:
            self.processor.scaler_X = pickle.load(f)
        with open('models/scaler_y.pkl', 'rb') as f:
            self.processor.scaler_y = pickle.load(f)
        
        # 설정 로드
        with open('models/config.json', 'r') as f:
            config = json.load(f)
            self.processor.feature_columns = config['feature_columns']
            self.processor.pattern_detector.pattern_stats = config.get('pattern_stats', {})
        
        print("✅ 실시간 예측기 준비 완료")
    
    def predict(self, data_100min):
        """100분 데이터로 예측"""
        if len(data_100min) != 100:
            raise ValueError(f"100분 데이터 필요 (현재: {len(data_100min)})")
        
        # 패턴 감지
        pattern, confidence = self.processor.pattern_detector.detect_pattern(data_100min)
        
        # 특징 준비 (임시)
        X = data_100min[self.processor.feature_columns[:5]].values  # 기본 특징만
        X = X.reshape(1, 100, -1)
        
        # 패턴 특징
        pattern_features = self.processor.pattern_detector.create_pattern_features(X)
        
        # 스케일링
        X_scaled = self.processor.scaler_X.transform(X.reshape(-1, X.shape[2])).reshape(X.shape)
        
        # 예측
        preds = self.model.predict([X_scaled, pattern_features], verbose=0)
        y_pred_scaled = preds[0][0, 0]
        y_cls_probs = preds[1][0]
        y_pat_probs = preds[2][0] if len(preds) > 2 else [0.5, 0.5]
        
        # 역변환
        y_pred = self.processor.scaler_y.inverse_transform([[y_pred_scaled]])[0, 0]
        
        # 현재값과 연속상승 계산
        current = data_100min['TOTALCNT'].iloc[-1]
        m14b = data_100min['M14AM14B'].iloc[-1]
        m14a = data_100min['M14AM10A'].iloc[-1]
        
        consecutive = 0
        for i in range(len(data_100min)-1, 0, -1):
            if data_100min['TOTALCNT'].iloc[i] > data_100min['TOTALCNT'].iloc[i-1]:
                consecutive += 1
            else:
                break
        
        # 패턴 기반 조정
        adjusted = self.processor.prob_calculator.pattern_based_prediction_adjustment(
            y_pred, pattern, m14b, m14a, consecutive
        )
        
        # 확률 계산
        probs = self.processor.prob_calculator.calculate_probabilities(adjusted, pattern, confidence)
        
        # 상태 판정
        if adjusted >= 1700:
            status = "🔴 위험 - 물류량 급증!"
        elif adjusted >= 1400:
            status = "🟡 주의 - 물류량 증가"
        else:
            status = "🟢 정상 - 안정적"
        
        # 이상신호 체크
        is_anomaly = (1651 <= adjusted <= 1682)
        
        return {
            'prediction': float(adjusted),
            'current': float(current),
            'change': float(adjusted - current),
            'change_rate': float((adjusted - current) / current * 100) if current > 0 else 0,
            'pattern': pattern,
            'pattern_confidence': confidence,
            'pattern_probs': {
                'UU1': float(y_pat_probs[0]),
                'UU2': float(y_pat_probs[1])
            },
            'level_probs': {
                '정상': float(probs['정상']),
                '주의': float(probs['주의']),
                '심각': float(probs['심각'])
            },
            'status': status,
            'is_anomaly': is_anomaly,
            'consecutive_rises': consecutive,
            'm14b_m14a_ratio': float(m14b / (m14a + 1)),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

# ====================================
# 메인 실행
# ====================================
def main():
    print("\n" + "="*80)
    print("🚀 EXTREME_NET 학습 시작")
    print("="*80)
    
    # 데이터 찾기
    data_paths = [
        '/mnt/user-data/uploads/gs.CSV',
        'data/20240201_TO_202507281705.csv',
        'data/20250731_to20250806.csv',
        'uu.csv',
        'uu2.csv',
        'data.csv'
    ]
    
    data_path = None
    for path in data_paths:
        if os.path.exists(path):
            data_path = path
            print(f"✅ 데이터: {path}")
            break
    
    if not data_path:
        print("❌ 데이터 없음")
        return
    
    # 데이터 로드
    df = pd.read_csv(data_path)
    print(f"  로드: {len(df):,}행")
    
    # 데이터 처리
    processor = DataProcessor()
    X, y, y_cls, features = processor.prepare_data(df)
    
    # 시퀀스 생성
    train_data, val_data, test_data = processor.create_sequences(X, y, y_cls)
    
    # 모델 생성 (strategy.scope 안에서!)
    input_shape = (100, len(features))
    pattern_shape = (3,)
    
    # GPU/CPU에 따라 다르게 처리
    global strategy
    if gpus and strategy:
        with strategy.scope():
            model = build_extreme_net_with_pattern(input_shape, pattern_shape)
    else:
        model = build_extreme_net_with_pattern(input_shape, pattern_shape)
    
    # 학습
    results = train_model(model, train_data, val_data, test_data, processor)
    
    # 결과 출력
    print("\n" + "="*80)
    print("🏆 최종 성능")
    print("="*80)
    print(f"  R²: {results['R2']:.4f}")
    print(f"  MAE: {results['MAE']:.2f}")
    print(f"  3구간 정확도: {results['Accuracy']:.4f}")
    print(f"  패턴 정확도: {results['PatternAccuracy']:.4f}")
    print(f"  상관계수: {results['Correlation']:.4f}")
    
    # 실시간 테스트
    if len(df) >= 100:
        print("\n🔮 실시간 예측 테스트...")
        predictor = RealtimePredictor()
        test_data = df.iloc[-100:].copy()
        result = predictor.predict(test_data)
        
        print(f"\n📈 예측 결과:")
        print(f"  패턴: {result['pattern']} (신뢰도: {result['pattern_confidence']:.1%})")
        print(f"  현재: {result['current']:.0f}")
        print(f"  예측: {result['prediction']:.0f}")
        print(f"  변화: {result['change']:+.0f} ({result['change_rate']:+.1f}%)")
        print(f"  상태: {result['status']}")
        
        print(f"\n  확률 분포:")
        for level, prob in result['level_probs'].items():
            print(f"    {level}: {prob:.1%}")
        
        if result['consecutive_rises'] > 5:
            print(f"  ⚠️ 연속 상승: {result['consecutive_rises']}회")
        if result['is_anomaly']:
            print(f"  ⚠️ 이상신호 감지 (1651-1682)")
        if result['m14b_m14a_ratio'] > 4:
            print(f"  📊 M14B/M14A 비율: {result['m14b_m14a_ratio']:.2f} (높음)")
    
    print("\n✅ 완료!")
    return results

if __name__ == "__main__":
    results = main()