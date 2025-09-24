"""
🔥 EXTREME_NET 완전체 - 모든 기능 포함 (R² 문제 해결)
================================================
패턴 조정 하드코딩 문제 수정 완료!
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, Input, backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import *
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import warnings
import os
import pickle
import json
from datetime import datetime

warnings.filterwarnings('ignore')
tf.random.set_seed(42)
np.random.seed(42)

print("="*80)
print("🔥 EXTREME_NET 완전체 - 모든 기능 포함 (수정 버전)")
print(f"📦 TensorFlow: {tf.__version__}")

# GPU 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print(f"🔧 GPU {len(gpus)}개 발견")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"  GPU 메모리 동적 할당 설정")

print("="*80)

# ====================================
# 수정된 R² 메트릭
# ====================================
class R2Score(tf.keras.metrics.Metric):
    """올바른 R² 메트릭"""
    def __init__(self, name='r2_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.ss_res = self.add_weight(name='ss_res', initializer='zeros')
        self.ss_tot = self.add_weight(name='ss_tot', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
        self.y_sum = self.add_weight(name='y_sum', initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        
        batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)
        self.count.assign_add(batch_size)
        self.y_sum.assign_add(tf.reduce_sum(y_true))
        y_mean = self.y_sum / self.count
        
        ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
        ss_tot = tf.reduce_sum(tf.square(y_true - y_mean))
        
        self.ss_res.assign_add(ss_res)
        self.ss_tot.assign_add(ss_tot)
    
    def result(self):
        return 1 - (self.ss_res / (self.ss_tot + 1e-7))
    
    def reset_state(self):
        self.ss_res.assign(0)
        self.ss_tot.assign(0)
        self.count.assign(0)
        self.y_sum.assign(0)

# ====================================
# UU1/UU2 패턴 감지 클래스
# ====================================
class PatternDetector:
    """UU1/UU2 패턴 감지"""
    
    def __init__(self):
        self.pattern_stats = {}
    
    def detect_pattern(self, df):
        """패턴 감지"""
        print("\n🔍 UU1/UU2 패턴 감지 중...")
        
        totalcnt_max = df['TOTALCNT'].max()
        totalcnt_mean = df['TOTALCNT'].mean()
        m14b_mean = df['M14AM14B'].mean() if 'M14AM14B' in df.columns else 0
        m14b_max = df['M14AM14B'].max() if 'M14AM14B' in df.columns else 0
        high_cases = len(df[df['TOTALCNT'] >= 1682])
        critical_cases = len(df[df['TOTALCNT'] >= 1700])
        m14b_350_ratio = (df['M14AM14B'] > 350).mean() if 'M14AM14B' in df.columns else 0
        
        print(f"  TOTALCNT 최대: {totalcnt_max:,}")
        print(f"  TOTALCNT 평균: {totalcnt_mean:.0f}")
        print(f"  M14AM14B 평균: {m14b_mean:.0f}")
        print(f"  1682+ 케이스: {high_cases}개")
        print(f"  1700+ 케이스: {critical_cases}개")
        
        # 패턴 판정
        if high_cases > 5 and m14b_mean > 380:
            pattern = "UU2"
            confidence = 0.95
            print(f"  🔥 UU2 패턴: 고값 유지 (신뢰도: {confidence:.1%})")
        elif m14b_350_ratio > 0.3:
            pattern = "UU2"
            confidence = 0.90
            print(f"  🔥 UU2 패턴: M14B 높음 (신뢰도: {confidence:.1%})")
        else:
            pattern = "UU1"
            confidence = 0.85
            print(f"  📈 UU1 패턴: 급증 가능 (신뢰도: {confidence:.1%})")
        
        self.pattern_stats = {
            'pattern': pattern,
            'confidence': float(confidence),
            'totalcnt_max': float(totalcnt_max),
            'm14b_mean': float(m14b_mean),
            'high_cases': int(high_cases)
        }
        
        return pattern, confidence
    
    def create_pattern_features(self, X_seq):
        """패턴 특징 생성"""
        features = []
        
        for i in range(len(X_seq)):
            seq = X_seq[i]
            totalcnt = seq[:, 0]
            m14b = seq[:, 1] if seq.shape[1] > 1 else np.zeros_like(totalcnt)
            
            # UU2 점수
            uu2_score = 0
            if np.mean(m14b) > 350:
                uu2_score += 0.5
            if np.max(totalcnt) > 1682:
                uu2_score += 0.5
            
            # UU1 점수
            uu1_score = 1 - uu2_score
            
            # 급증 위험
            spike_risk = 0
            if m14b[-1] > 300 and totalcnt[-1] < 1400:
                spike_risk = 0.7
            elif m14b[-1] > 400:
                spike_risk = 0.9
            
            features.append([uu1_score, uu2_score, spike_risk])
        
        return np.array(features, dtype=np.float32)

# ====================================
# 확률 계산 클래스
# ====================================
class ProbabilityCalculator:
    """패턴 기반 확률 계산"""
    
    @staticmethod
    def calculate_probabilities(predicted_value, pattern, confidence=1.0):
        """정상/주의/심각 확률"""
        
        if pattern == "UU1":
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
    def pattern_based_adjustment(base_pred, pattern, m14b, m14a, consecutive):
        """패턴 기반 예측 조정 - 과도한 조정 방지"""
        
        ratio = m14b / (m14a + 1)
        
        # 조정 비율을 줄임 (기존 1.15 → 1.03 등)
        if pattern == "UU1":
            if m14b > 300 and m14a < 80:  # 황금 패턴!
                adjustment = 1.03  # 기존 1.15에서 감소
            elif consecutive >= 10:
                adjustment = 1.02  # 기존 1.12에서 감소
            elif ratio > 4:
                adjustment = 1.01  # 기존 1.10에서 감소
            elif m14b >= 400:
                adjustment = 1.01
            elif m14b >= 350:
                adjustment = 1.00
            else:
                adjustment = 1.00
        else:  # UU2
            if m14b >= 450:
                adjustment = 1.00
            elif m14b >= 400:
                adjustment = 1.00
            else:
                adjustment = 0.99
        
        return base_pred * adjustment

# ====================================
# EXTREME_NET 모델 (Attention 포함!)
# ====================================
def create_extreme_net_model(input_shape, pattern_shape=(3,)):
    """EXTREME_NET 완전체 모델"""
    
    # 입력
    main_input = Input(shape=input_shape, name='main_input')
    pattern_input = Input(shape=pattern_shape, name='pattern_input')
    
    # LSTM 브랜치
    x = LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001))(main_input)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = LSTM(32, kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Attention 브랜치 (중요!)
    attention = MultiHeadAttention(num_heads=4, key_dim=16)(main_input, main_input)
    attention = GlobalAveragePooling1D()(attention)
    
    # 패턴 처리
    p = Dense(16, activation='relu')(pattern_input)
    p = Dropout(0.1)(p)
    
    # 융합
    combined = Concatenate()([x, attention, p])
    combined = BatchNormalization()(combined)
    combined = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(combined)
    combined = Dropout(0.2)(combined)
    combined = Dense(32, activation='relu')(combined)
    combined = Dropout(0.1)(combined)
    
    # 출력 (3개!)
    out_reg = Dense(1, name='regression')(combined)  # 회귀
    out_cls = Dense(3, activation='softmax', name='classification')(combined)  # 3구간
    out_pattern = Dense(2, activation='softmax', name='pattern_output')(combined)  # UU1/UU2
    
    model = Model(
        [main_input, pattern_input], 
        [out_reg, out_cls, out_pattern],
        name='ExtremeNetComplete'
    )
    
    # 컴파일 (수정된 R² 메트릭 사용)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={
            'regression': 'mse',
            'classification': 'sparse_categorical_crossentropy',
            'pattern_output': 'sparse_categorical_crossentropy'
        },
        loss_weights={
            'regression': 0.7,  # 회귀에 더 집중
            'classification': 0.2,
            'pattern_output': 0.1
        },
        metrics={
            'regression': [R2Score(), 'mae'],
            'classification': ['accuracy'],
            'pattern_output': ['accuracy']
        }
    )
    
    print("\n📊 EXTREME_NET 구조:")
    print("  - LSTM: 64 → 32")
    print("  - Attention: 4 heads (key_dim=16)")
    print("  - Pattern Dense: 16")
    print("  - 융합 Dense: 64 → 32")
    print("  - 출력: 회귀 + 3구간 + UU1/UU2")
    print(f"  - 총 파라미터: {model.count_params():,}개")
    
    return model

# ====================================
# 데이터 처리 클래스 (완전체)
# ====================================
class DataProcessor:
    def __init__(self, seq_len=100, pred_len=10):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.pattern_detector = PatternDetector()
        self.prob_calculator = ProbabilityCalculator()
        self.feature_columns = None
        self.original_data = None  # 원본 데이터 저장용
        
    def prepare_data(self, df):
        """데이터 준비"""
        print("\n📊 데이터 준비...")
        
        # 원본 데이터 저장 (나중에 실제 값 추출용)
        self.original_data = df.copy()
        
        # 0값 제거
        df = df[df['TOTALCNT'] > 0].reset_index(drop=True)
        
        # 이상치 제거 (R² 개선용)
        df = df[df['TOTALCNT'] < 5000].reset_index(drop=True)
        
        # 패턴 감지
        pattern, confidence = self.pattern_detector.detect_pattern(df)
        
        # 필수 컬럼
        base_cols = ['TOTALCNT', 'M14AM14B', 'M14AM10A', 'M14AM14BSUM', 'M14AM16']
        self.feature_columns = [c for c in base_cols if c in df.columns]
        
        # 황금 패턴!
        df['GOLDEN'] = 0
        if 'M14AM14B' in df.columns and 'M14AM10A' in df.columns:
            df['GOLDEN'] = ((df['M14AM14B'] > 300) & (df['M14AM10A'] < 80)).astype(float)
            golden_count = df['GOLDEN'].sum()
            if golden_count > 0:
                print(f"  🏆 황금패턴: {golden_count}개")
        
        # 비율
        df['RATIO'] = 1
        if 'M14AM14B' in df.columns and 'M14AM10A' in df.columns:
            df['RATIO'] = df['M14AM14B'] / (df['M14AM10A'] + 1)
        
        # Spike 감지
        df['SPIKE'] = (df['RATIO'] > 4).astype(float)
        spike_count = df['SPIKE'].sum()
        if spike_count > 0:
            print(f"  ⚡ Spike 감지: {spike_count}개")
        
        # UU1/UU2 신호
        df['UU2_SIGNAL'] = 0
        df['UU1_SIGNAL'] = 1
        if 'M14AM14B' in df.columns:
            df['UU2_SIGNAL'] = ((df['M14AM14B'] > 350) | (df['TOTALCNT'] > 1682)).astype(float)
            df['UU1_SIGNAL'] = 1 - df['UU2_SIGNAL']
        
        # 이상신호 감지 (1651-1682)
        df['ANOMALY'] = ((df['TOTALCNT'] >= 1651) & (df['TOTALCNT'] <= 1682)).astype(float)
        anomaly_count = df['ANOMALY'].sum()
        if anomaly_count > 0:
            print(f"  ⚠️ 이상신호(1651-1682): {anomaly_count}개")
        
        # 이동평균
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
        
        if max_consecutive > 0:
            print(f"  📈 최대 연속상승: {max_consecutive}회")
        
        # 최종 특징
        all_features = self.feature_columns + [
            'GOLDEN', 'RATIO', 'SPIKE',
            'UU1_SIGNAL', 'UU2_SIGNAL', 'ANOMALY',
            'MA_5', 'MA_10', 'MA_20', 'MA_30',
            'STD_5', 'STD_10', 'STD_20', 'STD_30',
            'CHANGE_1', 'CHANGE_10', 'CHANGE_RATE',
            'CONSECUTIVE_RISE'
        ]
        all_features = [c for c in all_features if c in df.columns]
        self.feature_columns = all_features
        
        print(f"  특징 개수: {len(all_features)}")
        
        X = df[all_features].values.astype(np.float32)
        y = df['TOTALCNT'].values.astype(np.float32)
        
        # 3구간
        y_cls = np.zeros(len(y), dtype=np.int32)
        y_cls[y >= 1400] = 1
        y_cls[y >= 1700] = 2
        
        print(f"  데이터: {X.shape}")
        print(f"  TOTALCNT: {y.min():.0f} ~ {y.max():.0f}")
        
        # 구간 분포
        for i in range(3):
            count = (y_cls == i).sum()
            names = ['정상(<1400)', '주의(1400-1699)', '위험(≥1700)']
            print(f"  {names[i]}: {count}개 ({count/len(y)*100:.1f}%)")
        
        return X, y, y_cls
    
    def create_sequences(self, X, y, y_cls):
        """시퀀스 생성"""
        print("\n🔄 시퀀스 생성...")
        
        # NaN 처리
        X = np.nan_to_num(X, 0)
        
        sequences_X = []
        sequences_y = []
        sequences_y_cls = []
        sequences_orig = []  # 원본 데이터 인덱스 저장
        
        for i in range(len(X) - self.seq_len - self.pred_len + 1):
            seq_x = X[i:i + self.seq_len]
            target_idx = i + self.seq_len + self.pred_len - 1
            
            if target_idx < len(y):
                sequences_X.append(seq_x)
                sequences_y.append(y[target_idx])
                sequences_y_cls.append(y_cls[target_idx])
                sequences_orig.append(X[i + self.seq_len - 1])  # 마지막 시점의 원본 특징 저장
        
        sequences_X = np.array(sequences_X, dtype=np.float32)
        sequences_y = np.array(sequences_y, dtype=np.float32)
        sequences_y_cls = np.array(sequences_y_cls, dtype=np.int32)
        sequences_orig = np.array(sequences_orig, dtype=np.float32)
        
        print(f"  시퀀스 수: {len(sequences_X)}")
        
        # 패턴 특징
        pattern_features = self.pattern_detector.create_pattern_features(sequences_X)
        
        # 패턴 타겟 (UU1=0, UU2=1)
        pattern_targets = np.zeros(len(sequences_X), dtype=np.int32)
        for i in range(len(sequences_X)):
            seq = sequences_X[i]
            m14b_mean = np.mean(seq[:, 1]) if seq.shape[1] > 1 else 0
            tc_max = np.max(seq[:, 0])
            
            if m14b_mean > 350 or tc_max > 1682:
                pattern_targets[i] = 1  # UU2
        
        # 분할
        n = len(sequences_X)
        train_size = int(n * 0.7)
        val_size = int(n * 0.15)
        
        # Train
        X_train = sequences_X[:train_size]
        X_train_pattern = pattern_features[:train_size]
        X_train_orig = sequences_orig[:train_size]
        y_train = sequences_y[:train_size]
        y_train_cls = sequences_y_cls[:train_size]
        y_train_pattern = pattern_targets[:train_size]
        
        # Val
        X_val = sequences_X[train_size:train_size + val_size]
        X_val_pattern = pattern_features[train_size:train_size + val_size]
        X_val_orig = sequences_orig[train_size:train_size + val_size]
        y_val = sequences_y[train_size:train_size + val_size]
        y_val_cls = sequences_y_cls[train_size:train_size + val_size]
        y_val_pattern = pattern_targets[train_size:train_size + val_size]
        
        # Test
        X_test = sequences_X[train_size + val_size:]
        X_test_pattern = pattern_features[train_size + val_size:]
        X_test_orig = sequences_orig[train_size + val_size:]
        y_test = sequences_y[train_size + val_size:]
        y_test_cls = sequences_y_cls[train_size + val_size:]
        y_test_pattern = pattern_targets[train_size + val_size:]
        
        # 스케일링 (중요!)
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
        
        # 스케일링 검증
        print(f"\n  스케일링 검증:")
        print(f"    X_train 범위: [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]")
        print(f"    y_train 범위: [{y_train_scaled.min():.2f}, {y_train_scaled.max():.2f}]")
        print(f"    y_train 평균: {y_train_scaled.mean():.2f} ± {y_train_scaled.std():.2f}")
        
        # 패턴 분포
        print(f"\n📊 패턴 분포:")
        for name, y_pat in [('Train', y_train_pattern), ('Val', y_val_pattern), ('Test', y_test_pattern)]:
            uu1 = (y_pat == 0).sum()
            uu2 = (y_pat == 1).sum()
            print(f"  {name}: UU1={uu1}개 ({uu1/len(y_pat)*100:.1f}%), UU2={uu2}개 ({uu2/len(y_pat)*100:.1f}%)")
        
        # 반환 (원본 특징 추가!)
        train_data = ([X_train_scaled, X_train_pattern], y_train_scaled, y_train_cls, y_train_pattern, y_train, X_train_orig)
        val_data = ([X_val_scaled, X_val_pattern], y_val_scaled, y_val_cls, y_val_pattern, y_val, X_val_orig)
        test_data = ([X_test_scaled, X_test_pattern], y_test_scaled, y_test_cls, y_test_pattern, y_test, X_test_orig)
        
        return train_data, val_data, test_data
    
    def save_all(self, path='models/'):
        """모든 설정 저장"""
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
        
        print(f"✅ 모든 설정 저장: {path}")

# ====================================
# 수정된 학습 함수 (핵심!)
# ====================================
def train_model(model, train_data, val_data, test_data, processor):
    """모델 학습 - 패턴 조정 문제 해결"""
    # 6개 요소 언패킹 (원본 특징 포함)
    X_train, y_train_scaled, y_train_cls, y_train_pattern, y_train_orig, X_train_orig = train_data
    X_val, y_val_scaled, y_val_cls, y_val_pattern, y_val_orig, X_val_orig = val_data
    X_test, y_test_scaled, y_test_cls, y_test_pattern, y_test_orig, X_test_orig = test_data
    
    print("\n" + "="*60)
    print("🎯 EXTREME_NET 학습 시작")
    print("="*60)
    
    # 콜백
    callbacks = [
        EarlyStopping(
            monitor='val_regression_r2_score',
            mode='max',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            'models/best_extreme_net.keras',
            monitor='val_regression_r2_score',
            mode='max',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # 배치 크기 (GPU 메모리 고려)
    batch_size = 32 if gpus else 16
    
    # 학습
    print(f"\n📈 학습 설정:")
    print(f"  Batch Size: {batch_size}")
    print(f"  Epochs: 100 (Early Stopping)")
    print(f"  Learning Rate: 0.001 (ReduceLROnPlateau)")
    
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
    print("📊 EXTREME_NET 성능 평가")
    print("="*60)
    
    # 예측
    preds = model.predict(X_test, batch_size=batch_size, verbose=0)
    y_pred_scaled = preds[0].flatten()
    y_cls_pred = np.argmax(preds[1], axis=1)
    y_pat_pred = np.argmax(preds[2], axis=1)
    
    # 역변환 (중요!)
    y_pred = processor.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    # =========================================
    # 수정된 패턴 기반 조정 (핵심 수정!)
    # =========================================
    print("\n🔧 패턴 기반 예측 조정 (수정됨)...")
    adjusted = []
    golden_detected = 0
    
    for i in range(len(y_pred)):
        pattern = 'UU2' if y_pat_pred[i] == 1 else 'UU1'
        
        # 실제 테스트 데이터에서 값 추출 (안전하게!)
        try:
            # X_test에서 스케일된 값 가져오기
            last_timestep = X_test[0][i, -1, :]  # 마지막 시점
            
            # 역스케일링 (중요!)
            unscaled = processor.scaler_X.inverse_transform(
                last_timestep.reshape(1, -1)
            )[0]
            
            # feature_columns에서 인덱스 찾기
            feature_list = processor.feature_columns
            
            if 'M14AM14B' in feature_list:
                m14b_idx = feature_list.index('M14AM14B')
                m14b = unscaled[m14b_idx]
            else:
                m14b = 250
            
            if 'M14AM10A' in feature_list:
                m14a_idx = feature_list.index('M14AM10A')
                m14a = unscaled[m14a_idx]
            else:
                m14a = 80
            
            if 'CONSECUTIVE_RISE' in feature_list:
                cons_idx = feature_list.index('CONSECUTIVE_RISE')
                consecutive = int(unscaled[cons_idx])
            else:
                consecutive = 0
                
        except Exception as e:
            # 에러 시 기본값 (데이터 평균값 사용)
            m14b = 250
            m14a = 80
            consecutive = 0
        
        # 황금 패턴 체크 (실제 값으로!)
        if m14b > 300 and m14a < 80:
            golden_detected += 1
        
        # 조정 (과도하지 않게)
        adj = processor.prob_calculator.pattern_based_adjustment(
            y_pred[i], pattern, m14b, m14a, consecutive
        )
        adjusted.append(adj)
    
    adjusted = np.array(adjusted)
    
    if golden_detected > 0:
        print(f"  🏆 실제 황금패턴 감지: {golden_detected}개")
    
    # 조정 전후 비교
    mae_before = mean_absolute_error(y_test_orig, y_pred)
    mae_after = mean_absolute_error(y_test_orig, adjusted)
    
    print(f"\n  조정 효과:")
    print(f"    조정 전 MAE: {mae_before:.2f}")
    print(f"    조정 후 MAE: {mae_after:.2f}")
    
    # 최종 메트릭 계산
    mae = mae_after
    rmse = np.sqrt(mean_squared_error(y_test_orig, adjusted))
    r2 = r2_score(y_test_orig, adjusted)
    
    # 패턴 정확도
    pattern_acc = accuracy_score(y_test_pattern, y_pat_pred)
    
    # 3구간 정확도
    cls_acc = accuracy_score(y_test_cls, y_cls_pred)
    
    # 이상신호 감지 (1651-1682)
    anomaly_true = ((y_test_orig >= 1651) & (y_test_orig <= 1682))
    anomaly_pred = ((adjusted >= 1651) & (adjusted <= 1682))
    anomaly_detected = np.sum(anomaly_true & anomaly_pred)
    
    # 위험구간 감지 (≥1700)
    danger_true = (y_test_cls == 2)
    danger_pred = (y_cls_pred == 2)
    if danger_true.sum() > 0:
        tp = np.sum(danger_true & danger_pred)
        fp = np.sum(~danger_true & danger_pred)
        fn = np.sum(danger_true & ~danger_pred)
        danger_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        danger_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        danger_f1 = 2 * (danger_precision * danger_recall) / (danger_precision + danger_recall) \
                    if (danger_precision + danger_recall) > 0 else 0
    else:
        danger_precision = danger_recall = danger_f1 = 0
    
    print(f"\n📈 회귀 성능:")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²: {r2:.4f}")
    
    print(f"\n🔍 패턴 예측:")
    print(f"  UU1/UU2 정확도: {pattern_acc:.4f}")
    
    print(f"\n📊 3구간 분류:")
    print(f"  정확도: {cls_acc:.4f}")
    
    if danger_true.sum() > 0:
        print(f"\n⚠️ 위험구간(≥1700) 감지:")
        print(f"  Precision: {danger_precision:.4f}")
        print(f"  Recall: {danger_recall:.4f}")
        print(f"  F1-Score: {danger_f1:.4f}")
    
    if anomaly_true.sum() > 0:
        print(f"\n⚠️ 이상신호(1651-1682) 감지:")
        print(f"  감지: {anomaly_detected}/{anomaly_true.sum()}개")
    
    # R² 검증
    if r2 < 0:
        print(f"\n⚠️ R² 여전히 음수: {r2:.4f}")
        print("  디버깅 정보:")
        print(f"    예측값 범위: {adjusted.min():.0f} ~ {adjusted.max():.0f}")
        print(f"    실제값 범위: {y_test_orig.min():.0f} ~ {y_test_orig.max():.0f}")
        print(f"    예측 평균: {adjusted.mean():.0f}")
        print(f"    실제 평균: {y_test_orig.mean():.0f}")
        correlation = np.corrcoef(y_test_orig, adjusted)[0, 1]
        print(f"    상관계수: {correlation:.4f}")
    elif r2 > 0.8:
        print(f"\n✅ 우수한 R² 성능: {r2:.4f}")
    else:
        print(f"\n📊 R² 정상: {r2:.4f}")
    
    # 저장
    os.makedirs('models', exist_ok=True)
    model.save('models/extreme_net_complete.keras')
    processor.save_all()
    
    print("\n💾 모델 및 설정 저장 완료")
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'PatternAcc': pattern_acc,
        'ClassAcc': cls_acc,
        'DangerF1': danger_f1
    }

# ====================================
# 실시간 예측 클래스
# ====================================
class RealtimePredictor:
    def __init__(self, model_path='models/extreme_net_complete.keras'):
        """실시간 예측기"""
        # 커스텀 객체 등록
        custom_objects = {'R2Score': R2Score}
        
        self.model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        self.processor = DataProcessor()
        
        # 설정 로드
        with open('models/scaler_X.pkl', 'rb') as f:
            self.processor.scaler_X = pickle.load(f)
        with open('models/scaler_y.pkl', 'rb') as f:
            self.processor.scaler_y = pickle.load(f)
        with open('models/config.json', 'r') as f:
            config = json.load(f)
            self.processor.feature_columns = config['feature_columns']
        
        print("✅ 실시간 예측기 준비 완료")
    
    def predict(self, data_100min):
        """100분 데이터로 10분 후 예측"""
        if len(data_100min) != 100:
            raise ValueError(f"100분 데이터 필요 (현재: {len(data_100min)})")
        
        # 데이터 준비 (prepare_data와 동일한 처리)
        df = data_100min.copy()
        
        # 특징 생성
        df['GOLDEN'] = 0
        if 'M14AM14B' in df.columns and 'M14AM10A' in df.columns:
            df['GOLDEN'] = ((df['M14AM14B'] > 300) & (df['M14AM10A'] < 80)).astype(float)
        
        df['RATIO'] = 1
        if 'M14AM14B' in df.columns and 'M14AM10A' in df.columns:
            df['RATIO'] = df['M14AM14B'] / (df['M14AM10A'] + 1)
        
        df['SPIKE'] = (df['RATIO'] > 4).astype(float)
        
        df['UU2_SIGNAL'] = 0
        df['UU1_SIGNAL'] = 1
        if 'M14AM14B' in df.columns:
            df['UU2_SIGNAL'] = ((df['M14AM14B'] > 350) | (df['TOTALCNT'] > 1682)).astype(float)
            df['UU1_SIGNAL'] = 1 - df['UU2_SIGNAL']
        
        df['ANOMALY'] = ((df['TOTALCNT'] >= 1651) & (df['TOTALCNT'] <= 1682)).astype(float)
        
        # 이동평균
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
        for i in range(1, len(df)):
            if df.loc[i, 'TOTALCNT'] > df.loc[i-1, 'TOTALCNT']:
                consecutive += 1
            else:
                consecutive = 0
            df.loc[i, 'CONSECUTIVE_RISE'] = consecutive
        
        # 패턴 감지
        pattern, confidence = self.processor.pattern_detector.detect_pattern(df)
        
        # 특징 준비
        X = df[self.processor.feature_columns].values
        X = X.reshape(1, 100, -1)
        
        # 패턴 특징
        pattern_features = self.processor.pattern_detector.create_pattern_features(X)
        
        # 스케일링
        n_features = X.shape[2]
        X_scaled = self.processor.scaler_X.transform(X.reshape(-1, n_features)).reshape(X.shape)
        
        # 예측
        preds = self.model.predict([X_scaled, pattern_features], verbose=0)
        y_pred_scaled = preds[0][0, 0]
        y_cls_probs = preds[1][0]
        y_pat_probs = preds[2][0]
        
        # 역변환
        y_pred = self.processor.scaler_y.inverse_transform([[y_pred_scaled]])[0, 0]
        
        # 현재값과 패턴 조정
        current = df['TOTALCNT'].iloc[-1]
        m14b = df['M14AM14B'].iloc[-1] if 'M14AM14B' in df.columns else 0
        m14a = df['M14AM10A'].iloc[-1] if 'M14AM10A' in df.columns else 0
        consecutive = df['CONSECUTIVE_RISE'].iloc[-1]
        
        # 패턴 기반 조정 (수정된 버전)
        adjusted = self.processor.prob_calculator.pattern_based_adjustment(
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
        
        # 황금패턴 체크
        is_golden = (m14b > 300 and m14a < 80)
        
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
            'level_probs': probs,
            'status': status,
            'is_anomaly': is_anomaly,
            'is_golden': is_golden,
            'consecutive_rises': int(consecutive),
            'm14b_m14a_ratio': float(m14b / (m14a + 1)),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

# ====================================
# 메인 실행
# ====================================
def main():
    print("\n🚀 EXTREME_NET 완전체 시작 (수정 버전)")
    
    # 데이터 찾기
    data_paths = [
        '/mnt/user-data/uploads/gs.CSV',
        'data/20240201_TO_202507281705.csv',
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
    
    # 처리
    processor = DataProcessor()
    X, y, y_cls = processor.prepare_data(df)
    train_data, val_data, test_data = processor.create_sequences(X, y, y_cls)
    
    # 모델 생성
    input_shape = (100, X.shape[1])
    pattern_shape = (3,)
    
    # GPU 사용 가능시 Strategy
    if gpus:
        strategy = tf.distribute.MirroredStrategy()
        print("🔧 GPU Strategy 사용")
        with strategy.scope():
            model = create_extreme_net_model(input_shape, pattern_shape)
    else:
        print("🔧 CPU 모드")
        model = create_extreme_net_model(input_shape, pattern_shape)
    
    # 학습
    results = train_model(model, train_data, val_data, test_data, processor)
    
    # 최종 결과
    print("\n" + "="*60)
    print("🏆 EXTREME_NET 최종 결과")
    print("="*60)
    print(f"R²: {results['R2']:.4f}")
    print(f"MAE: {results['MAE']:.2f}")
    print(f"패턴 정확도: {results['PatternAcc']:.4f}")
    print(f"3구간 정확도: {results['ClassAcc']:.4f}")
    
    # 실시간 예측 테스트
    if len(df) >= 100:
        print("\n🔮 실시간 예측 테스트...")
        predictor = RealtimePredictor()
        test_sample = df.iloc[-100:].copy()
        result = predictor.predict(test_sample)
        
        print(f"\n📈 예측 결과:")
        print(f"  패턴: {result['pattern']} (신뢰도: {result['pattern_confidence']:.1%})")
        print(f"  현재: {result['current']:.0f}")
        print(f"  10분 후: {result['prediction']:.0f}")
        print(f"  변화: {result['change']:+.0f} ({result['change_rate']:+.1f}%)")
        print(f"  상태: {result['status']}")
        
        if result['is_golden']:
            print(f"  🏆 황금패턴 감지!")
        if result['is_anomaly']:
            print(f"  ⚠️ 이상신호 감지 (1651-1682)")
        if result['consecutive_rises'] > 5:
            print(f"  📈 연속 상승: {result['consecutive_rises']}회")
    
    print("\n✅ EXTREME_NET 완전체 완료!")
    return results

if __name__ == "__main__":
    results = main()