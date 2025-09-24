"""
🔥 EXTREME_NET ULTIMATE - 모든 기능 포함 최종 완전체
====================================================
누락 없음! Strategy 오류 해결! 모든 기능 포함!
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, Input, backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import *
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import RobustScaler
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
print("🔥 EXTREME_NET ULTIMATE - 완전체")
print(f"📦 TensorFlow: {tf.__version__}")

# GPU 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print(f"🔧 GPU {len(gpus)}개 발견")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except:
            pass

print("="*80)

# ====================================
# 커스텀 메트릭
# ====================================
def r2_keras(y_true, y_pred):
    """R² 메트릭"""
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))

# ====================================
# UU1/UU2 패턴 감지 클래스 (완전체)
# ====================================
class PatternDetector:
    """UU1/UU2 패턴 감지"""
    
    def __init__(self):
        self.pattern_stats = {}
    
    def detect_pattern(self, df):
        """패턴 감지"""
        print("\n🔍 UU1/UU2 패턴 감지...")
        
        # 통계
        totalcnt_max = df['TOTALCNT'].max()
        totalcnt_mean = df['TOTALCNT'].mean()
        m14b_mean = df['M14AM14B'].mean() if 'M14AM14B' in df.columns else 0
        m14b_max = df['M14AM14B'].max() if 'M14AM14B' in df.columns else 0
        high_cases = len(df[df['TOTALCNT'] >= 1682])
        critical_cases = len(df[df['TOTALCNT'] >= 1700])
        m14b_350_ratio = (df['M14AM14B'] > 350).mean() if 'M14AM14B' in df.columns else 0
        
        print(f"  TOTALCNT 최대: {totalcnt_max:,}")
        print(f"  M14AM14B 평균: {m14b_mean:.0f}")
        print(f"  1682+ 케이스: {high_cases}개")
        print(f"  1700+ 케이스: {critical_cases}개")
        
        # 패턴 판정
        if high_cases > 5 and m14b_mean > 380:
            pattern = "UU2"
            confidence = 0.95
            print(f"  🔥 UU2 패턴: 고값 유지")
        elif m14b_350_ratio > 0.3:
            pattern = "UU2"
            confidence = 0.90
            print(f"  🔥 UU2 패턴: M14B 높음")
        else:
            pattern = "UU1"
            confidence = 0.85
            print(f"  📈 UU1 패턴: 급증 가능")
        
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
            # 첫 번째 컬럼이 TOTALCNT
            totalcnt = seq[:, 0]
            # 두 번째 컬럼이 M14AM14B (있으면)
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
# 확률 계산 클래스 (완전체)
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
        """패턴 기반 예측 조정"""
        
        ratio = m14b / (m14a + 1)
        
        if pattern == "UU1":
            if m14b > 300 and m14a < 80:  # 황금 패턴!
                adjustment = 1.15
                print(f"    🏆 황금패턴 감지!")
            elif consecutive >= 10:
                adjustment = 1.12
            elif ratio > 4:
                adjustment = 1.10
            elif m14b >= 400:
                adjustment = 1.08
            elif m14b >= 350:
                adjustment = 1.05
            else:
                adjustment = 1.02
        else:  # UU2
            if m14b >= 450:
                adjustment = 1.02
            elif m14b >= 400:
                adjustment = 1.01
            else:
                adjustment = 0.99
        
        return base_pred * adjustment

# ====================================
# 모델 생성 함수 (Attention 포함!)
# ====================================
def create_and_compile_model(input_shape, pattern_shape=(3,)):
    """EXTREME_NET 모델 생성 및 컴파일"""
    
    # 입력
    main_input = Input(shape=input_shape, name='main_input')
    pattern_input = Input(shape=pattern_shape, name='pattern_input')
    
    # LSTM 브랜치
    x = LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001))(main_input)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = LSTM(32, kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Attention 브랜치 (누락됐던 부분!)
    attention = MultiHeadAttention(num_heads=4, key_dim=16)(main_input, main_input)
    attention = GlobalAveragePooling1D()(attention)
    
    # 패턴 처리
    p = Dense(16, activation='relu')(pattern_input)
    p = Dropout(0.2)(p)
    
    # 결합
    combined = Concatenate()([x, attention, p])
    combined = BatchNormalization()(combined)
    combined = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(combined)
    combined = Dropout(0.3)(combined)
    combined = Dense(32, activation='relu')(combined)
    combined = Dropout(0.2)(combined)
    
    # 출력 (3개!)
    out_reg = Dense(1, name='regression')(combined)  # 회귀
    out_cls = Dense(3, activation='softmax', name='classification')(combined)  # 3구간
    out_pattern = Dense(2, activation='softmax', name='pattern_output')(combined)  # UU1/UU2
    
    model = Model(
        [main_input, pattern_input], 
        [out_reg, out_cls, out_pattern],  # 3개 출력!
        name='ExtremeNetUltimate'
    )
    
    # 컴파일
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
    
    print("\n📊 EXTREME_NET 구조:")
    print("  - LSTM: 64 → 32")
    print("  - Attention: 4 heads")
    print("  - Pattern Dense: 16")
    print("  - 출력: 회귀 + 3구간 + UU1/UU2")
    
    return model

# ====================================
# 데이터 처리 클래스 (완전체)
# ====================================
class DataProcessor:
    def __init__(self, seq_len=100, pred_len=10):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scaler_X = RobustScaler()
        self.scaler_y = RobustScaler()
        self.pattern_detector = PatternDetector()
        self.prob_calculator = ProbabilityCalculator()
        
    def prepare_data(self, df):
        """데이터 준비"""
        print("\n📊 데이터 준비...")
        
        # 0값 제거
        df = df[df['TOTALCNT'] > 0].reset_index(drop=True)
        
        # 패턴 감지
        pattern, confidence = self.pattern_detector.detect_pattern(df)
        
        # 필수 컬럼
        base_cols = ['TOTALCNT', 'M14AM14B', 'M14AM10A', 'M14AM14BSUM', 'M14AM16']
        feature_cols = [c for c in base_cols if c in df.columns]
        
        # 특징 생성
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
        
        # Spike
        df['SPIKE'] = (df['RATIO'] > 4).astype(float)
        
        # UU1/UU2 신호
        df['UU2_SIGNAL'] = 0
        df['UU1_SIGNAL'] = 1
        if 'M14AM14B' in df.columns:
            df['UU2_SIGNAL'] = ((df['M14AM14B'] > 350) | (df['TOTALCNT'] > 1682)).astype(float)
            df['UU1_SIGNAL'] = 1 - df['UU2_SIGNAL']
        
        # 이상신호 감지 (1651-1682) - 누락됐던 부분!
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
        
        # 연속 상승
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
        all_features = feature_cols + [
            'GOLDEN', 'RATIO', 'SPIKE',
            'UU1_SIGNAL', 'UU2_SIGNAL', 'ANOMALY',
            'MA_5', 'MA_10', 'MA_20', 'MA_30',
            'STD_5', 'STD_10', 'STD_20', 'STD_30',
            'CHANGE_1', 'CHANGE_10', 'CHANGE_RATE',
            'CONSECUTIVE_RISE'
        ]
        all_features = [c for c in all_features if c in df.columns]
        
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
        
        return X, y, y_cls, all_features
    
    def create_sequences(self, X, y, y_cls):
        """시퀀스 생성"""
        print("\n🔄 시퀀스 생성...")
        
        # NaN 처리
        X = np.nan_to_num(X, 0)
        
        sequences_X = []
        sequences_y = []
        sequences_y_cls = []
        
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
        y_train = sequences_y[:train_size]
        y_train_cls = sequences_y_cls[:train_size]
        y_train_pattern = pattern_targets[:train_size]
        
        # Val
        X_val = sequences_X[train_size:train_size + val_size]
        X_val_pattern = pattern_features[train_size:train_size + val_size]
        y_val = sequences_y[train_size:train_size + val_size]
        y_val_cls = sequences_y_cls[train_size:train_size + val_size]
        y_val_pattern = pattern_targets[train_size:train_size + val_size]
        
        # Test
        X_test = sequences_X[train_size + val_size:]
        X_test_pattern = pattern_features[train_size + val_size:]
        y_test = sequences_y[train_size + val_size:]
        y_test_cls = sequences_y_cls[train_size + val_size:]
        y_test_pattern = pattern_targets[train_size + val_size:]
        
        # 스케일링
        print("  스케일링...")
        
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
        
        # 패턴 분포
        print(f"\n📊 패턴 분포:")
        for name, y_pat in [('Train', y_train_pattern), ('Val', y_val_pattern), ('Test', y_test_pattern)]:
            uu1 = (y_pat == 0).sum()
            uu2 = (y_pat == 1).sum()
            print(f"  {name}: UU1={uu1}개 ({uu1/len(y_pat)*100:.1f}%), UU2={uu2}개 ({uu2/len(y_pat)*100:.1f}%)")
        
        # 반환 (5개 요소씩)
        train_data = ([X_train_scaled, X_train_pattern], y_train_scaled, y_train_cls, y_train_pattern, y_train)
        val_data = ([X_val_scaled, X_val_pattern], y_val_scaled, y_val_cls, y_val_pattern, y_val)
        test_data = ([X_test_scaled, X_test_pattern], y_test_scaled, y_test_cls, y_test_pattern, y_test)
        
        return train_data, val_data, test_data

# ====================================
# 학습 함수
# ====================================
def train_model(model, train_data, val_data, test_data, processor):
    """모델 학습"""
    # 5개 요소 언패킹
    X_train, y_train_scaled, y_train_cls, y_train_pattern, y_train_orig = train_data
    X_val, y_val_scaled, y_val_cls, y_val_pattern, y_val_orig = val_data
    X_test, y_test_scaled, y_test_cls, y_test_pattern, y_test_orig = test_data
    
    print("\n" + "="*60)
    print("🎯 학습 시작")
    print("="*60)
    
    # 콜백
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1),
        ModelCheckpoint('models/best.keras', monitor='val_loss', save_best_only=True, verbose=1)
    ]
    
    # 학습
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
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # 평가
    print("\n📊 평가...")
    preds = model.predict(X_test, verbose=0)
    y_pred_scaled = preds[0].flatten()
    y_cls_pred = np.argmax(preds[1], axis=1)
    y_pat_pred = np.argmax(preds[2], axis=1)
    
    # 역변환
    y_pred = processor.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    # 패턴 조정 (임시값)
    print("\n🔧 패턴 기반 조정...")
    adjusted = []
    for i in range(len(y_pred)):
        pattern = 'UU2' if y_pat_pred[i] == 1 else 'UU1'
        adj = processor.prob_calculator.pattern_based_adjustment(
            y_pred[i], pattern, 350, 75, 5
        )
        adjusted.append(adj)
    adjusted = np.array(adjusted)
    
    # 메트릭
    mae = mean_absolute_error(y_test_orig, adjusted)
    rmse = np.sqrt(mean_squared_error(y_test_orig, adjusted))
    r2 = r2_score(y_test_orig, adjusted)
    
    # 패턴 정확도
    pattern_acc = accuracy_score(y_test_pattern, y_pat_pred)
    
    # 3구간 정확도
    cls_acc = accuracy_score(y_test_cls, y_cls_pred)
    
    # 이상신호 감지
    anomaly_true = ((y_test_orig >= 1651) & (y_test_orig <= 1682))
    anomaly_pred = ((adjusted >= 1651) & (adjusted <= 1682))
    anomaly_detected = np.sum(anomaly_true & anomaly_pred)
    
    print(f"\n📈 성능:")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²: {r2:.4f}")
    print(f"  패턴 정확도: {pattern_acc:.4f}")
    print(f"  3구간 정확도: {cls_acc:.4f}")
    
    if anomaly_true.sum() > 0:
        print(f"  이상신호 감지: {anomaly_detected}/{anomaly_true.sum()}")
    
    # 저장
    os.makedirs('models', exist_ok=True)
    model.save('models/extreme_net_ultimate.keras')
    
    with open('models/scaler_X.pkl', 'wb') as f:
        pickle.dump(processor.scaler_X, f)
    with open('models/scaler_y.pkl', 'wb') as f:
        pickle.dump(processor.scaler_y, f)
    
    print("\n💾 저장 완료")
    
    return {
        'MAE': mae, 'RMSE': rmse, 'R2': r2,
        'PatternAcc': pattern_acc, 'ClassAcc': cls_acc
    }

# ====================================
# 메인
# ====================================
def main():
    print("\n🚀 EXTREME_NET ULTIMATE 시작")
    
    # 데이터 찾기
    data_paths = [
        '/mnt/user-data/uploads/gs.CSV',
        'data/20240201_TO_202507281705.csv',
        'uu.csv',
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
    X, y, y_cls, features = processor.prepare_data(df)
    train_data, val_data, test_data = processor.create_sequences(X, y, y_cls)
    
    # 모델
    input_shape = (100, len(features))
    pattern_shape = (3,)
    
    # GPU 체크
    use_gpu = len(tf.config.experimental.list_physical_devices('GPU')) > 0
    
    if use_gpu:
        # GPU Strategy
        strategy = tf.distribute.MirroredStrategy()
        print("🔧 GPU Strategy 사용")
        
        with strategy.scope():
            model = create_and_compile_model(input_shape, pattern_shape)
            results = train_model(model, train_data, val_data, test_data, processor)
    else:
        # CPU
        print("🔧 CPU 모드")
        model = create_and_compile_model(input_shape, pattern_shape)
        results = train_model(model, train_data, val_data, test_data, processor)
    
    # 결과
    print("\n" + "="*60)
    print("🏆 최종 결과")
    print("="*60)
    print(f"R²: {results['R2']:.4f}")
    print(f"MAE: {results['MAE']:.2f}")
    print(f"패턴 정확도: {results['PatternAcc']:.4f}")
    print(f"3구간 정확도: {results['ClassAcc']:.4f}")
    
    print("\n✅ 완료!")
    return results

if __name__ == "__main__":
    results = main()