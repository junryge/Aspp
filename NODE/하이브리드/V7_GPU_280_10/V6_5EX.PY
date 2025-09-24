"""
🔥 EXTREME_NET + UU1/UU2 패턴 통합 반도체 물류 예측 시스템
==========================================================
100분 데이터 → 10분 후 예측 + 패턴 감지
TensorFlow 2.16.1 CPU/GPU 지원
UU1: 급증 가능 패턴 / UU2: 고값 유지 패턴
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import *
from tensorflow.keras.losses import Huber
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import warnings
import os
import gc
import pickle
import json
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
tf.random.set_seed(42)
np.random.seed(42)

# ====================================
# GPU 설정
# ====================================
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print(f"🔧 GPU {len(gpus)}개 발견")
    for i, gpu in enumerate(gpus):
        tf.config.experimental.set_memory_growth(gpu, True)
        print(f"  GPU:{i} 메모리 동적 할당 활성화")
    strategy = tf.distribute.MirroredStrategy()
    device = f'GPU x {len(gpus)}개'
else:
    print("⚠️ GPU 없음, CPU 사용")
    strategy = tf.distribute.get_strategy()
    device = 'CPU'

print("="*80)
print("🔥 EXTREME_NET + UU1/UU2 패턴 반도체 물류 예측 시스템")
print(f"📦 TensorFlow: {tf.__version__}")
print(f"🔧 Device: {device}")
print("="*80)

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
        print(f"  M14B>350 비율: {m14b_350_ratio:.1%}")
        print(f"  M14B>400 비율: {m14b_400_ratio:.1%}")
        
        # 패턴 판정 (개선된 로직)
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
            'confidence': confidence,
            'totalcnt_max': totalcnt_max,
            'totalcnt_mean': totalcnt_mean,
            'm14b_mean': m14b_mean,
            'm14b_max': m14b_max,
            'high_cases': high_cases,
            'critical_cases': critical_cases,
            'm14b_350_ratio': m14b_350_ratio,
            'm14b_400_ratio': m14b_400_ratio
        }
        
        return pattern, confidence
    
    def calculate_pattern_features(self, sequence_data):
        """패턴 기반 특징 추출"""
        features = []
        
        for seq in sequence_data:
            # 시퀀스별 패턴 특징
            m14b_values = seq[:, 1]  # M14AM14B 컬럼 (인덱스 1)
            totalcnt_values = seq[:, 0]  # TOTALCNT 컬럼 (인덱스 0)
            
            # UU2 가능성 점수
            uu2_score = 0
            if np.mean(m14b_values) > 380:
                uu2_score += 0.3
            if np.max(totalcnt_values) > 1682:
                uu2_score += 0.3
            if (m14b_values > 350).sum() / len(m14b_values) > 0.3:
                uu2_score += 0.4
            
            # UU1 가능성 점수
            uu1_score = 1 - uu2_score
            
            # 급증 위험도
            spike_risk = 0
            if m14b_values[-1] > 300 and totalcnt_values[-1] < 1400:
                spike_risk = 0.7
            elif m14b_values[-1] > 400:
                spike_risk = 0.9
            
            features.append([uu1_score, uu2_score, spike_risk])
        
        return np.array(features)

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
        """패턴 기반 예측값 조정"""
        
        ratio = m14b / (m14a + 1)
        
        if pattern == "UU1":
            # UU1: 급증 가능 패턴
            if m14b > 300 and m14a < 80:  # 황금 패턴
                adjustment = 1.15
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
# EXTREME_NET 모델 정의 (패턴 인식 강화)
# ====================================
def build_extreme_net_with_pattern(input_shape, pattern_shape=(3,)):
    """
    EXTREME_NET - LSTM + Multi-Head Attention + Pattern Recognition
    """
    # 메인 시퀀스 입력
    main_inputs = Input(shape=input_shape, name='main_inputs')
    # 패턴 특징 입력
    pattern_inputs = Input(shape=pattern_shape, name='pattern_inputs')
    
    # LSTM 브랜치 (시계열 패턴 학습)
    lstm = LSTM(64, return_sequences=True, name='lstm1')(main_inputs)
    lstm = LSTM(32, name='lstm2')(lstm)
    
    # Attention 브랜치 (중요 시점 포착)
    attn = MultiHeadAttention(
        num_heads=4, 
        key_dim=16,
        name='multi_head_attention'
    )(main_inputs, main_inputs)
    attn = GlobalAveragePooling1D(name='global_avg_pool')(attn)
    
    # 패턴 처리
    pattern_dense = Dense(16, activation='relu', name='pattern_dense')(pattern_inputs)
    
    # 융합
    x = Concatenate(name='fusion')([lstm, attn, pattern_dense])
    x = BatchNormalization(name='batch_norm')(x)
    x = Dense(64, activation='relu', name='dense1')(x)
    x = Dropout(0.2, name='dropout1')(x)
    x = Dense(32, activation='relu', name='dense2')(x)
    x = Dropout(0.1, name='dropout2')(x)
    
    # 출력층
    out_reg = Dense(1, name='regression')(x)  # 회귀 (물류량 예측)
    out_cls = Dense(3, activation='softmax', name='classification')(x)  # 분류 (구간)
    out_pattern = Dense(2, activation='softmax', name='pattern_output')(x)  # 패턴 분류 (UU1/UU2)
    
    model = Model([main_inputs, pattern_inputs], [out_reg, out_cls, out_pattern], name='ExtremeNetPattern')
    
    print("\n📊 EXTREME_NET + Pattern 구조:")
    print(f"  - LSTM 레이어: 64 → 32")
    print(f"  - Attention Heads: 4개 (key_dim=16)")
    print(f"  - Pattern Dense: 16")
    print(f"  - 융합 Dense: 64 → 32")
    print(f"  - Dropout: 20% → 10%")
    print(f"  - 출력: 회귀 + 3구간 분류 + UU1/UU2 패턴")
    
    return model

# ====================================
# 데이터 처리 클래스 (패턴 통합)
# ====================================
class ExtremeDataProcessorWithPattern:
    def __init__(self, seq_len=100, pred_len=10):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.feature_scaler = StandardScaler()
        self.target_scaler = RobustScaler()
        self.pattern_detector = PatternDetector()
        self.prob_calculator = ProbabilityCalculator()
        self.feature_columns = None
        self.statistics = {}
        self.pattern_stats = {}
        
    def save_all(self, path='extreme_net_pattern_model/'):
        """모든 설정 저장"""
        os.makedirs(path, exist_ok=True)
        
        # 스케일러 저장
        with open(f'{path}feature_scaler.pkl', 'wb') as f:
            pickle.dump(self.feature_scaler, f)
        with open(f'{path}target_scaler.pkl', 'wb') as f:
            pickle.dump(self.target_scaler, f)
            
        # 설정 저장
        config = {
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'feature_columns': self.feature_columns,
            'statistics': self.statistics,
            'pattern_stats': self.pattern_detector.pattern_stats
        }
        with open(f'{path}config.json', 'w') as f:
            json.dump(config, f, indent=2)
            
        print(f"✅ 모든 설정 저장: {path}")
    
    def load_all(self, path='extreme_net_pattern_model/'):
        """모든 설정 로드"""
        with open(f'{path}feature_scaler.pkl', 'rb') as f:
            self.feature_scaler = pickle.load(f)
        with open(f'{path}target_scaler.pkl', 'rb') as f:
            self.target_scaler = pickle.load(f)
        with open(f'{path}config.json', 'r') as f:
            config = json.load(f)
            self.seq_len = config['seq_len']
            self.pred_len = config['pred_len']
            self.feature_columns = config['feature_columns']
            self.statistics = config['statistics']
            self.pattern_stats = config.get('pattern_stats', {})
        print(f"✅ 모든 설정 로드: {path}")
        
    def load_and_process(self, filepath):
        print(f"\n📂 데이터 로딩: {filepath}")
        df = pd.read_csv(filepath)
        print(f"✅ 원본: {df.shape[0]:,}행 × {df.shape[1]}열")
        
        # 0값 제거
        before = len(df)
        df = df[df['TOTALCNT'] > 0].reset_index(drop=True)
        if before - len(df) > 0:
            print(f"✅ 0값 제거: {before - len(df):,}개")
        
        # 시간 정렬
        if 'CURRTIME' in df.columns:
            df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), 
                                           format='%Y%m%d%H%M', errors='coerce')
            df = df.sort_values('CURRTIME').reset_index(drop=True)
        
        # 전체 데이터 패턴 감지
        pattern, confidence = self.pattern_detector.detect_pattern(df)
        
        # 통계 저장
        self.statistics = {
            'mean': float(df['TOTALCNT'].mean()),
            'std': float(df['TOTALCNT'].std()),
            'min': float(df['TOTALCNT'].min()),
            'max': float(df['TOTALCNT'].max()),
            'median': float(df['TOTALCNT'].median()),
            'pattern': pattern,
            'pattern_confidence': confidence
        }
        
        print(f"\n📊 TOTALCNT 통계:")
        print(f"  평균: {self.statistics['mean']:,.0f}")
        print(f"  표준편차: {self.statistics['std']:,.0f}")
        print(f"  최소: {self.statistics['min']:,}")
        print(f"  최대: {self.statistics['max']:,}")
        print(f"  중앙값: {self.statistics['median']:,}")
        print(f"  주요 패턴: {pattern} (신뢰도: {confidence:.1%})")
        
        # 구간 분포
        normal = (df['TOTALCNT'] < 1400).sum()
        warning = ((df['TOTALCNT'] >= 1400) & (df['TOTALCNT'] < 1700)).sum()
        critical = (df['TOTALCNT'] >= 1700).sum()
        
        print(f"\n📊 3구간 분포:")
        print(f"  Level 0 (정상 < 1400): {normal:,}개 ({normal/len(df)*100:.1f}%)")
        print(f"  Level 1 (주의 1400-1699): {warning:,}개 ({warning/len(df)*100:.1f}%)")
        print(f"  Level 2 (위험 ≥ 1700): {critical:,}개 ({critical/len(df)*100:.1f}%)")
        
        # 급증 패턴 감지
        spike_count = 0
        consecutive_rises = 0
        max_consecutive = 0
        
        for i in range(1, len(df)):
            if df.loc[i, 'TOTALCNT'] > df.loc[i-1, 'TOTALCNT']:
                consecutive_rises += 1
                max_consecutive = max(max_consecutive, consecutive_rises)
            else:
                consecutive_rises = 0
            
            if i >= 10 and df.loc[i, 'TOTALCNT'] - df.loc[i-10, 'TOTALCNT'] > 100:
                spike_count += 1
        
        print(f"\n⚠️ 패턴 분석:")
        print(f"  급증 패턴 (10분간 100+ 증가): {spike_count}회")
        print(f"  최대 연속 상승: {max_consecutive}회")
        
        return df
    
    def create_features(self, df):
        print("\n⚙️ EXTREME_NET + Pattern용 특성 생성...")
        
        # 핵심 특성
        df['RATIO'] = df['M14AM14B'] / (df['M14AM10A'] + 1)
        df['GOLDEN'] = ((df['M14AM14B'] > 300) & (df['M14AM10A'] < 80)).astype(float)
        df['SPIKE'] = ((df['M14AM14B'] / (df['M14AM10A'] + 1)) > 4).astype(float)
        
        # UU1/UU2 패턴 특징
        df['UU2_SIGNAL'] = ((df['M14AM14B'] > 350) | (df['TOTALCNT'] > 1682)).astype(float)
        df['UU1_SIGNAL'] = ((df['M14AM14B'] <= 350) & (df['TOTALCNT'] < 1682)).astype(float)
        df['PATTERN_RISK'] = df['M14AM14B'].rolling(10, min_periods=1).mean() / 350  # 패턴 위험도
        
        # 시간 특성
        if 'CURRTIME' in df.columns:
            df['HOUR'] = df['CURRTIME'].dt.hour
            df['HOUR_SIN'] = np.sin(2 * np.pi * df['HOUR'] / 24)
            df['HOUR_COS'] = np.cos(2 * np.pi * df['HOUR'] / 24)
        else:
            df['HOUR_SIN'] = 0
            df['HOUR_COS'] = 1
        
        # 이동평균 & 변동성
        for w in [5, 10, 20, 30, 50]:
            df[f'MA_{w}'] = df['TOTALCNT'].rolling(w, min_periods=1).mean()
            df[f'STD_{w}'] = df['TOTALCNT'].rolling(w, min_periods=1).std().fillna(0)
        
        # 변화율
        df['CHANGE_1'] = df['TOTALCNT'].diff(1).fillna(0)
        df['CHANGE_5'] = df['TOTALCNT'].diff(5).fillna(0)
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
        
        # 컬럼별 특성
        for col in ['M14AM14B', 'M14AM14BSUM', 'M14AM10A', 'M14AM16']:
            if col in df.columns:
                df[f'{col}_MA10'] = df[col].rolling(10, min_periods=1).mean()
                df[f'{col}_CHANGE'] = df[col].diff(10).fillna(0)
        
        # 타겟 생성
        df['TARGET'] = df['TOTALCNT'].shift(-self.pred_len)
        df['LEVEL'] = 0
        df.loc[df['TARGET'] >= 1400, 'LEVEL'] = 1
        df.loc[df['TARGET'] >= 1700, 'LEVEL'] = 2
        
        # 이상신호 감지
        df['ANOMALY'] = ((df['TARGET'] >= 1651) & (df['TARGET'] <= 1682)).astype(int)
        
        # 패턴 타겟 (UU1=0, UU2=1)
        df['PATTERN_TARGET'] = 0
        # 타겟 시점의 패턴 판정
        for i in range(len(df) - self.pred_len):
            future_window = df.iloc[i:i+self.pred_len+50]
            if len(future_window) > 0:
                future_m14b_mean = future_window['M14AM14B'].mean()
                future_high_ratio = (future_window['TOTALCNT'] > 1682).mean()
                if future_m14b_mean > 380 or future_high_ratio > 0.1:
                    df.loc[i, 'PATTERN_TARGET'] = 1
        
        df = df.dropna()
        print(f"✅ 특성 생성 완료: {len(df.columns)}개 컬럼")
        print(f"✅ 유효 데이터: {len(df):,}행")
        
        return df
    
    def create_sequences(self, df):
        print("\n🔄 EXTREME_NET + Pattern 시퀀스 생성...")
        
        # 특성 선택
        base_features = [
            'TOTALCNT', 'M14AM14B', 'M14AM14BSUM', 'M14AM10A', 'M14AM16',
            'RATIO', 'GOLDEN', 'SPIKE',
            'UU1_SIGNAL', 'UU2_SIGNAL', 'PATTERN_RISK',
            'HOUR_SIN', 'HOUR_COS',
            'CONSECUTIVE_RISE'
        ]
        
        ma_features = [f'MA_{w}' for w in [5, 10, 20, 30, 50]]
        std_features = [f'STD_{w}' for w in [5, 10, 20, 30, 50]]
        change_features = ['CHANGE_1', 'CHANGE_5', 'CHANGE_10', 'CHANGE_RATE']
        
        col_features = []
        for col in ['M14AM14B', 'M14AM14BSUM', 'M14AM10A', 'M14AM16']:
            if f'{col}_MA10' in df.columns:
                col_features.extend([f'{col}_MA10', f'{col}_CHANGE'])
        
        self.feature_columns = base_features + ma_features + std_features + change_features + col_features
        self.feature_columns = [f for f in self.feature_columns if f in df.columns]
        
        print(f"  선택된 특성: {len(self.feature_columns)}개")
        
        X_data = df[self.feature_columns].values.astype(np.float32)
        y_reg = df['TARGET'].values.astype(np.float32)
        y_cls = df['LEVEL'].values.astype(np.int32)
        y_pattern = df['PATTERN_TARGET'].values.astype(np.int32)
        y_anomaly = df['ANOMALY'].values.astype(np.int32)
        
        # 분할 비율
        total_samples = len(X_data) - self.seq_len
        train_end = int(total_samples * 0.7)
        val_end = int(total_samples * 0.85)
        
        # 스케일링
        print("  스케일링 적용...")
        self.feature_scaler.fit(X_data[:train_end + self.seq_len])
        self.target_scaler.fit(y_reg[:train_end].reshape(-1, 1))
        
        X_scaled = self.feature_scaler.transform(X_data)
        y_reg_scaled = self.target_scaler.transform(y_reg.reshape(-1, 1)).flatten()
        
        # 시퀀스 생성
        X, X_pattern, y_r, y_c, y_p, y_a = [], [], [], [], [], []
        for i in range(total_samples):
            seq = X_scaled[i:i+self.seq_len]
            X.append(seq)
            
            # 패턴 특징 계산
            pattern_feat = self.pattern_detector.calculate_pattern_features(seq.reshape(1, self.seq_len, -1))[0]
            X_pattern.append(pattern_feat)
            
            y_r.append(y_reg_scaled[i+self.seq_len-1])
            y_c.append(y_cls[i+self.seq_len-1])
            y_p.append(y_pattern[i+self.seq_len-1])
            y_a.append(y_anomaly[i+self.seq_len-1])
        
        X = np.array(X, dtype=np.float32)
        X_pattern = np.array(X_pattern, dtype=np.float32)
        y_r = np.array(y_r, dtype=np.float32)
        y_c = np.array(y_c, dtype=np.int32)
        y_p = np.array(y_p, dtype=np.int32)
        y_a = np.array(y_a, dtype=np.int32)
        
        # 시계열 순서 유지 분할
        X_train = [X[:train_end], X_pattern[:train_end]]
        X_val = [X[train_end:val_end], X_pattern[train_end:val_end]]
        X_test = [X[val_end:], X_pattern[val_end:]]
        
        y_train = (y_r[:train_end], y_c[:train_end], y_p[:train_end], y_a[:train_end])
        y_val = (y_r[train_end:val_end], y_c[train_end:val_end], y_p[train_end:val_end], y_a[train_end:val_end])
        y_test = (y_r[val_end:], y_c[val_end:], y_p[val_end:], y_a[val_end:])
        
        print(f"\n📊 데이터 분할:")
        print(f"  Train: {X_train[0].shape} (70%)")
        print(f"  Val: {X_val[0].shape} (15%)")
        print(f"  Test: {X_test[0].shape} (15%)")
        
        # 패턴 분포 확인
        print(f"\n📊 패턴 분포:")
        for name, y_pat in [('Train', y_train[2]), ('Val', y_val[2]), ('Test', y_test[2])]:
            uu1_count = (y_pat == 0).sum()
            uu2_count = (y_pat == 1).sum()
            print(f"  {name}: UU1={uu1_count:,}개 ({uu1_count/len(y_pat)*100:.1f}%), " +
                  f"UU2={uu2_count:,}개 ({uu2_count/len(y_pat)*100:.1f}%)")
        
        # 모든 설정 저장
        self.save_all()
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# ====================================
# EXTREME_NET + Pattern 학습 함수
# ====================================
def train_extreme_net_with_pattern(train_data, val_data, test_data, processor):
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data
    
    # 모델 생성
    main_shape = (X_train[0].shape[1], X_train[0].shape[2])
    pattern_shape = (X_train[1].shape[1],)
    
    with strategy.scope():
        model = build_extreme_net_with_pattern(main_shape, pattern_shape)
        
        # 컴파일
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={
                'regression': 'mae',
                'classification': 'sparse_categorical_crossentropy',
                'pattern_output': 'sparse_categorical_crossentropy'
            },
            loss_weights={
                'regression': 0.5,
                'classification': 0.3,
                'pattern_output': 0.2
            },
            metrics={
                'regression': ['mae', 'mse'],
                'classification': ['accuracy'],
                'pattern_output': ['accuracy']
            }
        )
    
    print("\n" + "="*60)
    print("🎯 EXTREME_NET + Pattern 학습 시작")
    print("="*60)
    
    # 콜백 설정
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
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
            'extreme_net_pattern_model/best_model.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # 배치 크기
    batch_size = 64 * strategy.num_replicas_in_sync if gpus else 32
    
    # 학습
    print(f"\n📈 학습 설정:")
    print(f"  Batch Size: {batch_size}")
    print(f"  Epochs: 60 (Early Stopping)")
    print(f"  Learning Rate: 0.001 (ReduceLROnPlateau)")
    
    history = model.fit(
        X_train,
        {
            'regression': y_train[0],
            'classification': y_train[1],
            'pattern_output': y_train[2]
        },
        validation_data=(
            X_val,
            {
                'regression': y_val[0],
                'classification': y_val[1],
                'pattern_output': y_val[2]
            }
        ),
        epochs=60,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # 평가
    print("\n" + "="*60)
    print("📊 EXTREME_NET + Pattern 성능 평가")
    print("="*60)
    
    # 예측
    preds = model.predict(X_test, batch_size=batch_size, verbose=0)
    y_reg_pred = preds[0].flatten()
    y_cls_pred = np.argmax(preds[1], axis=1)
    y_pat_pred = np.argmax(preds[2], axis=1)
    
    # 역변환
    y_reg_true_orig = processor.target_scaler.inverse_transform(
        y_test[0].reshape(-1, 1)
    ).flatten()
    y_reg_pred_orig = processor.target_scaler.inverse_transform(
        y_reg_pred.reshape(-1, 1)
    ).flatten()
    
    # 패턴 기반 조정
    print("\n🔧 패턴 기반 예측값 조정...")
    adjusted_predictions = []
    for i in range(len(y_reg_pred_orig)):
        pattern = 'UU2' if y_pat_pred[i] == 1 else 'UU1'
        # 테스트 데이터에서 M14B, M14A 값 추출 (임시 값 사용)
        m14b = 350  # 실제 구현시 X_test에서 추출
        m14a = 75
        consecutive = 5
        
        adjusted = processor.prob_calculator.pattern_based_prediction_adjustment(
            y_reg_pred_orig[i], pattern, m14b, m14a, consecutive
        )
        adjusted_predictions.append(adjusted)
    
    adjusted_predictions = np.array(adjusted_predictions)
    
    # 조정된 예측 메트릭
    mae_adjusted = mean_absolute_error(y_reg_true_orig, adjusted_predictions)
    rmse_adjusted = np.sqrt(mean_squared_error(y_reg_true_orig, adjusted_predictions))
    r2_adjusted = r2_score(y_reg_true_orig, adjusted_predictions)
    
    # 기본 회귀 메트릭
    mae = mean_absolute_error(y_reg_true_orig, y_reg_pred_orig)
    rmse = np.sqrt(mean_squared_error(y_reg_true_orig, y_reg_pred_orig))
    r2 = r2_score(y_reg_true_orig, y_reg_pred_orig)
    mape = np.mean(np.abs((y_reg_true_orig - y_reg_pred_orig) / y_reg_true_orig)) * 100
    
    print(f"\n📈 회귀 성능 (기본):")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²: {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    
    print(f"\n📈 회귀 성능 (패턴 조정):")
    print(f"  MAE: {mae_adjusted:.2f} ({mae_adjusted-mae:+.2f})")
    print(f"  RMSE: {rmse_adjusted:.2f} ({rmse_adjusted-rmse:+.2f})")
    print(f"  R²: {r2_adjusted:.4f} ({r2_adjusted-r2:+.4f})")
    
    # 패턴 예측 정확도
    pattern_acc = accuracy_score(y_test[2], y_pat_pred)
    print(f"\n🔍 UU1/UU2 패턴 예측 정확도: {pattern_acc:.4f}")
    
    # 패턴별 성능 분석
    for pat_val, pat_name in [(0, 'UU1'), (1, 'UU2')]:
        mask = (y_test[2] == pat_val)
        if mask.sum() > 0:
            pat_mae = mean_absolute_error(y_reg_true_orig[mask], adjusted_predictions[mask])
            print(f"  {pat_name} MAE: {pat_mae:.2f} (n={mask.sum()})")
    
    # 분류 메트릭
    acc = accuracy_score(y_test[1], y_cls_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test[1], y_cls_pred, average='weighted'
    )
    
    print(f"\n📊 3구간 분류 성능:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    
    # 구간별 성능
    cm = confusion_matrix(y_test[1], y_cls_pred)
    print(f"\n📋 Confusion Matrix:")
    print("  실제\\예측  Level0  Level1  Level2")
    for i in range(3):
        print(f"  Level{i}:   {cm[i, 0]:6d}  {cm[i, 1]:6d}  {cm[i, 2]:6d}")
    
    # 위험 감지 성능
    if 2 in y_test[1]:
        danger_true = (y_test[1] == 2)
        danger_pred = (y_cls_pred == 2)
        
        tp = np.sum(danger_true & danger_pred)
        fp = np.sum(~danger_true & danger_pred)
        fn = np.sum(danger_true & ~danger_pred)
        
        danger_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        danger_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        danger_f1 = 2 * (danger_precision * danger_recall) / (danger_precision + danger_recall) \
                    if (danger_precision + danger_recall) > 0 else 0
        
        print(f"\n⚠️ 위험 구간(Level 2) 감지 성능:")
        print(f"  Precision: {danger_precision:.4f}")
        print(f"  Recall: {danger_recall:.4f}")
        print(f"  F1-Score: {danger_f1:.4f}")
        print(f"  정확 예측: {tp}개, 오탐: {fp}개, 놓침: {fn}개")
    
    # 모델 저장
    os.makedirs('extreme_net_pattern_model', exist_ok=True)
    model.save('extreme_net_pattern_model/extreme_net_pattern_final.keras')
    print(f"\n💾 최종 모델 저장: extreme_net_pattern_model/extreme_net_pattern_final.keras")
    
    # 결과 저장
    results = {
        'MAE': mae_adjusted,
        'RMSE': rmse_adjusted,
        'R2': r2_adjusted,
        'MAPE': mape,
        'Pattern_Accuracy': pattern_acc,
        'Classification_Accuracy': acc,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1
    }
    
    # CSV 저장
    pd.DataFrame([results]).to_csv('extreme_net_pattern_results.csv', index=False)
    print(f"💾 결과 저장: extreme_net_pattern_results.csv")
    
    return model, results, history

# ====================================
# 실시간 예측 클래스 (패턴 통합)
# ====================================
class ExtremeNetPatternPredictor:
    def __init__(self, model_path='extreme_net_pattern_model/extreme_net_pattern_final.keras'):
        """EXTREME_NET + Pattern 실시간 예측기"""
        self.model = tf.keras.models.load_model(model_path)
        self.processor = ExtremeDataProcessorWithPattern()
        self.processor.load_all('extreme_net_pattern_model/')
        print(f"✅ EXTREME_NET + Pattern 예측기 준비 완료")
        
    def predict(self, data_100min):
        """100분 데이터로 10분 후 예측"""
        # 데이터 검증
        if len(data_100min) != 100:
            raise ValueError(f"100분 데이터 필요 (현재: {len(data_100min)})")
        
        # 패턴 감지
        pattern, confidence = self.processor.pattern_detector.detect_pattern(data_100min)
        
        # 특성 추출
        X = data_100min[self.processor.feature_columns].values
        
        # 스케일링
        X_scaled = self.processor.feature_scaler.transform(X)
        X_seq = X_scaled.reshape(1, 100, -1).astype(np.float32)
        
        # 패턴 특징
        X_pattern = self.processor.pattern_detector.calculate_pattern_features(X_seq)
        
        # 예측
        preds = self.model.predict([X_seq, X_pattern], verbose=0)
        y_reg_scaled = preds[0][0, 0]
        y_cls_probs = preds[1][0]
        y_pat_probs = preds[2][0]
        
        y_cls = np.argmax(y_cls_probs)
        y_pat = np.argmax(y_pat_probs)
        
        # 역변환
        y_reg = self.processor.target_scaler.inverse_transform([[y_reg_scaled]])[0, 0]
        
        # 패턴 기반 조정
        m14b = data_100min['M14AM14B'].iloc[-1]
        m14a = data_100min['M14AM10A'].iloc[-1]
        
        # 연속 상승 계산
        consecutive = 0
        for i in range(len(data_100min)-1, 0, -1):
            if data_100min['TOTALCNT'].iloc[i] > data_100min['TOTALCNT'].iloc[i-1]:
                consecutive += 1
            else:
                break
        
        # 예측값 조정
        adjusted = self.processor.prob_calculator.pattern_based_prediction_adjustment(
            y_reg, pattern, m14b, m14a, consecutive
        )
        
        # 확률 계산
        probs = self.processor.prob_calculator.calculate_probabilities(adjusted, pattern, confidence)
        
        # 레벨 정의
        levels = {
            0: '정상 (< 1400)',
            1: '주의 (1400-1699)',
            2: '위험 (≥ 1700)'
        }
        
        # 상태 판정
        if adjusted >= 1700:
            status = "🔴 심각 - 물류량 급증!"
        elif adjusted >= 1400:
            status = "🟡 주의 - 물류량 증가 대비"
        else:
            status = "🟢 정상 - 안정적"
        
        # 이상신호 감지
        is_anomaly = (1651 <= adjusted <= 1682)
        
        # 급증 여부
        current = data_100min['TOTALCNT'].iloc[-1]
        is_spike = (adjusted - current > 100)
        
        return {
            'prediction': float(adjusted),
            'prediction_original': float(y_reg),
            'current': float(current),
            'change': float(adjusted - current),
            'change_rate': float((adjusted - current) / current * 100),
            'pattern': pattern,
            'pattern_confidence': confidence,
            'pattern_probs': {
                'UU1': float(y_pat_probs[0]),
                'UU2': float(y_pat_probs[1])
            },
            'level': y_cls,
            'level_name': levels[y_cls],
            'level_probs': {
                '정상': float(probs['정상']),
                '주의': float(probs['주의']),
                '심각': float(probs['심각'])
            },
            'status': status,
            'confidence': float(np.max(y_cls_probs)),
            'is_anomaly': is_anomaly,
            'is_spike': is_spike,
            'consecutive_rises': consecutive,
            'm14b_m14a_ratio': float(m14b / (m14a + 1)),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

# ====================================
# 메인 실행
# ====================================
def main():
    print("\n" + "="*80)
    print("🔥 EXTREME_NET + UU1/UU2 패턴 학습 시작")
    print("="*80)
    
    # 데이터 찾기
    data_paths = [
        '/mnt/user-data/uploads/gs.CSV',
        'data/20240201_TO_202507281705.csv',
        'data/20250731_to20250806.csv',
        'data/20250807_DATA.CSV',
        'uu.csv',
        'uu2.csv',
        'data.csv'
    ]
    
    data_path = None
    for path in data_paths:
        if os.path.exists(path):
            data_path = path
            print(f"✅ 데이터 발견: {path}")
            break
    
    if not data_path:
        print("❌ 데이터 파일을 찾을 수 없습니다!")
        print("다음 경로 중 하나에 데이터를 배치하세요:")
        for p in data_paths:
            print(f"  - {p}")
        return
    
    # 데이터 처리
    processor = ExtremeDataProcessorWithPattern()
    df = processor.load_and_process(data_path)
    df = processor.create_features(df)
    train_data, val_data, test_data = processor.create_sequences(df)
    
    # EXTREME_NET + Pattern 학습
    model, results, history = train_extreme_net_with_pattern(
        train_data, val_data, test_data, processor
    )
    
    # 최종 결과 출력
    print("\n" + "="*80)
    print("🏆 EXTREME_NET + Pattern 최종 성능")
    print("="*80)
    
    print("\n📊 핵심 지표:")
    print(f"  예측 정확도 (R²): {results['R2']:.4f}")
    print(f"  평균 오차 (MAE): {results['MAE']:.2f}")
    print(f"  3구간 정확도: {results['Classification_Accuracy']:.4f}")
    print(f"  패턴 예측 정확도: {results['Pattern_Accuracy']:.4f}")
    
    # 실시간 예측 테스트
    print("\n" + "="*80)
    print("🔮 실시간 예측 테스트 (UU1/UU2 패턴 적용)")
    print("="*80)
    
    predictor = ExtremeNetPatternPredictor()
    
    # 테스트 샘플
    if len(df) >= 100:
        test_sample = df.iloc[-100:].copy()
        result = predictor.predict(test_sample)
        
        print(f"\n📈 예측 결과:")
        print(f"  패턴: {result['pattern']} (신뢰도: {result['pattern_confidence']:.1%})")
        print(f"  현재: {result['current']:.0f}")
        print(f"  10분 후 예측: {result['prediction']:.0f}")
        print(f"  변화량: {result['change']:+.0f} ({result['change_rate']:+.1f}%)")
        print(f"  상태: {result['status']}")
        
        print(f"\n  확률 분포:")
        print(f"    🟢 정상: {result['level_probs']['정상']:.1%}")
        print(f"    🟡 주의: {result['level_probs']['주의']:.1%}")
        print(f"    🔴 심각: {result['level_probs']['심각']:.1%}")
        
        print(f"\n  패턴 확률:")
        print(f"    UU1 (급증 가능): {result['pattern_probs']['UU1']:.1%}")
        print(f"    UU2 (고값 유지): {result['pattern_probs']['UU2']:.1%}")
        
        print(f"\n  추가 정보:")
        print(f"    연속 상승: {result['consecutive_rises']}회")
        print(f"    M14B/M14A 비율: {result['m14b_m14a_ratio']:.2f}")
        
        if result['is_spike']:
            print(f"    ⚠️ 급증 경고!")
        if result['is_anomaly']:
            print(f"    ⚠️ 이상신호 감지 (1651-1682)")
    
    print("\n✅ EXTREME_NET + UU1/UU2 패턴 학습 완료!")
    print("="*80)
    
    # Patch Time Series Transformer 안내
    print("\n" + "="*80)
    print("📌 Patch Time Series Transformer 안내")
    print("="*80)
    print("100만개 데이터가 준비되면 알려주세요!")
    print("PatchTST 모델로 더 높은 성능을 달성할 수 있습니다.")
    print("="*80)
    
    return results

if __name__ == "__main__":
    results = main()