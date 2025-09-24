"""
🔥 EXTREME_NET 완전 수정 버전 - R² 문제 해결 + UU1/UU2 패턴
===========================================================
모든 기능 포함, R² 문제 해결
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
from datetime import datetime

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
else:
    print("⚠️ GPU 없음, CPU 사용")
    strategy = tf.distribute.get_strategy()

print("="*80)
print("🔥 EXTREME_NET 완전 수정 버전")
print(f"📦 TensorFlow: {tf.__version__}")
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
# UU1/UU2 패턴 감지
# ====================================
class PatternDetector:
    """UU1/UU2 패턴 감지"""
    
    def detect_pattern(self, df):
        """패턴 감지"""
        totalcnt_max = df['TOTALCNT'].max()
        totalcnt_mean = df['TOTALCNT'].mean()
        m14b_mean = df['M14AM14B'].mean()
        m14b_max = df['M14AM14B'].max()
        high_cases = len(df[df['TOTALCNT'] >= 1682])
        m14b_350_ratio = (df['M14AM14B'] > 350).sum() / len(df)
        
        # 패턴 판정
        if high_cases > 5 and m14b_mean > 380:
            pattern = "UU2"
            confidence = 0.95
        elif m14b_350_ratio > 0.3:
            pattern = "UU2"
            confidence = 0.90
        else:
            pattern = "UU1"
            confidence = 0.85
        
        return pattern, confidence
    
    def create_pattern_features(self, X_seq):
        """패턴 특징 생성"""
        # X_seq shape: (samples, timesteps, features)
        # TOTALCNT가 첫 번째 특징이라고 가정
        totalcnt = X_seq[:, :, 0]
        m14b = X_seq[:, :, 1] if X_seq.shape[2] > 1 else np.zeros_like(totalcnt)
        
        pattern_features = []
        for i in range(len(X_seq)):
            # 각 시퀀스의 패턴 특징
            tc_mean = np.mean(totalcnt[i])
            tc_max = np.max(totalcnt[i])
            tc_trend = (totalcnt[i][-1] - totalcnt[i][0]) / 100
            
            m14b_mean = np.mean(m14b[i])
            m14b_max = np.max(m14b[i])
            
            # UU2 가능성 점수
            uu2_score = 0
            if m14b_mean > 300:
                uu2_score += 0.5
            if tc_max > 1600:
                uu2_score += 0.5
            
            pattern_features.append([tc_trend, uu2_score, m14b_mean/300])
        
        return np.array(pattern_features, dtype=np.float32)

# ====================================
# 수정된 EXTREME_NET 모델
# ====================================
def build_extreme_net_fixed(input_shape, pattern_shape=(3,)):
    """수정된 EXTREME_NET - 더 안정적인 구조"""
    
    # 메인 입력
    main_input = Input(shape=input_shape, name='main_input')
    pattern_input = Input(shape=pattern_shape, name='pattern_input')
    
    # LSTM 경로 (정규화 추가)
    x = LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001))(main_input)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = LSTM(32, kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # 패턴 처리
    p = Dense(8, activation='relu')(pattern_input)
    
    # 결합
    combined = Concatenate()([x, p])
    
    # Dense 레이어
    combined = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(combined)
    combined = BatchNormalization()(combined)
    combined = Dropout(0.3)(combined)
    combined = Dense(16, activation='relu')(combined)
    
    # 출력
    out_reg = Dense(1, name='regression')(combined)
    out_cls = Dense(3, activation='softmax', name='classification')(combined)
    
    model = Model([main_input, pattern_input], [out_reg, out_cls], name='ExtremeNetFixed')
    return model

# ====================================
# 데이터 처리 클래스
# ====================================
class DataProcessor:
    def __init__(self, seq_len=100, pred_len=10):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scaler_X = RobustScaler()  # 이상치에 강건한 스케일러
        self.scaler_y = RobustScaler()
        self.pattern_detector = PatternDetector()
        self.feature_columns = None
        
    def prepare_data(self, df):
        """데이터 준비"""
        print("\n📊 데이터 준비...")
        
        # 0값 제거
        df = df[df['TOTALCNT'] > 0].reset_index(drop=True)
        
        # 패턴 감지
        pattern, confidence = self.pattern_detector.detect_pattern(df)
        print(f"  패턴: {pattern} (신뢰도: {confidence:.1%})")
        
        # 필수 컬럼
        base_columns = ['TOTALCNT', 'M14AM14B', 'M14AM10A', 'M14AM14BSUM', 'M14AM16']
        self.feature_columns = [col for col in base_columns if col in df.columns]
        
        # 특징 생성
        df['RATIO'] = df['M14AM14B'] / (df['M14AM10A'] + 1)
        df['GOLDEN'] = ((df['M14AM14B'] > 300) & (df['M14AM10A'] < 80)).astype(float)
        df['MA_10'] = df['TOTALCNT'].rolling(10, min_periods=1).mean()
        df['MA_30'] = df['TOTALCNT'].rolling(30, min_periods=1).mean()
        df['STD_10'] = df['TOTALCNT'].rolling(10, min_periods=1).std().fillna(0)
        df['CHANGE_1'] = df['TOTALCNT'].diff(1).fillna(0)
        df['CHANGE_10'] = df['TOTALCNT'].diff(10).fillna(0)
        
        # 연속 상승 카운트
        df['CONSECUTIVE_RISE'] = 0
        consecutive = 0
        for i in range(1, len(df)):
            if df.loc[i, 'TOTALCNT'] > df.loc[i-1, 'TOTALCNT']:
                consecutive += 1
            else:
                consecutive = 0
            df.loc[i, 'CONSECUTIVE_RISE'] = consecutive
        
        # UU1/UU2 신호
        df['UU2_SIGNAL'] = ((df['M14AM14B'] > 350) | (df['TOTALCNT'] > 1682)).astype(float)
        
        # 최종 특징
        all_features = self.feature_columns + [
            'RATIO', 'GOLDEN', 'MA_10', 'MA_30', 'STD_10', 
            'CHANGE_1', 'CHANGE_10', 'CONSECUTIVE_RISE', 'UU2_SIGNAL'
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
        
        # 분할
        n = len(sequences_X)
        train_size = int(n * 0.7)
        val_size = int(n * 0.15)
        
        # 훈련 데이터
        X_train = sequences_X[:train_size]
        X_train_pattern = pattern_features[:train_size]
        y_train = sequences_y[:train_size]
        y_train_cls = sequences_y_cls[:train_size]
        
        # 검증 데이터
        X_val = sequences_X[train_size:train_size + val_size]
        X_val_pattern = pattern_features[train_size:train_size + val_size]
        y_val = sequences_y[train_size:train_size + val_size]
        y_val_cls = sequences_y_cls[train_size:train_size + val_size]
        
        # 테스트 데이터
        X_test = sequences_X[train_size + val_size:]
        X_test_pattern = pattern_features[train_size + val_size:]
        y_test = sequences_y[train_size + val_size:]
        y_test_cls = sequences_y_cls[train_size + val_size:]
        
        # 스케일링 (중요!)
        print("\n⚙️ 스케일링...")
        
        # X 스케일링
        n_samples, n_timesteps, n_features = X_train.shape
        X_train_2d = X_train.reshape(-1, n_features)
        X_train_scaled_2d = self.scaler_X.fit_transform(X_train_2d)
        X_train_scaled = X_train_scaled_2d.reshape(n_samples, n_timesteps, n_features)
        
        X_val_scaled = self.scaler_X.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
        X_test_scaled = self.scaler_X.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)
        
        # y 스케일링 (회귀 타겟만)
        y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_val_scaled = self.scaler_y.transform(y_val.reshape(-1, 1)).flatten()
        y_test_scaled = self.scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        
        print(f"  Train: {X_train_scaled.shape}")
        print(f"  Val: {X_val_scaled.shape}")
        print(f"  Test: {X_test_scaled.shape}")
        
        # 스케일링 검증
        print(f"\n  스케일링 검증:")
        print(f"    X_train 범위: [{X_train_scaled.min():.3f}, {X_train_scaled.max():.3f}]")
        print(f"    y_train 범위: [{y_train_scaled.min():.3f}, {y_train_scaled.max():.3f}]")
        
        # 3개의 튜플로 묶어서 반환
        train_data = ([X_train_scaled, X_train_pattern], y_train_scaled, y_train_cls, y_train)
        val_data = ([X_val_scaled, X_val_pattern], y_val_scaled, y_val_cls, y_val)
        test_data = ([X_test_scaled, X_test_pattern], y_test_scaled, y_test_cls, y_test)
        
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
            'feature_columns': self.feature_columns
        }
        with open(f'{path}config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ 설정 저장: {path}")

# ====================================
# 학습 함수
# ====================================
def train_model(model, train_data, val_data, test_data, processor):
    """모델 학습"""
    # 데이터 언패킹
    X_train, y_train_scaled, y_train_cls, y_train_orig = train_data
    X_val, y_val_scaled, y_val_cls, y_val_orig = val_data
    X_test, y_test_scaled, y_test_cls, y_test_orig = test_data
    
    print("\n" + "="*60)
    print("🎯 EXTREME_NET 학습 시작")
    print("="*60)
    
    with strategy.scope():
        # 모델 컴파일
        model.compile(
            optimizer=Adam(learning_rate=0.0005),  # 학습률 낮춤
            loss={
                'regression': 'mse',  # MAE 대신 MSE
                'classification': 'sparse_categorical_crossentropy'
            },
            loss_weights={
                'regression': 0.8,  # 회귀에 더 가중치
                'classification': 0.2
            },
            metrics={
                'regression': [r2_keras, 'mae'],
                'classification': ['accuracy']
            }
        )
    
    # 콜백
    callbacks = [
        EarlyStopping(
            monitor='val_regression_loss',  # 회귀 손실 모니터링
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
            monitor='val_regression_r2_keras',  # R² 모니터링
            mode='max',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # 학습
    batch_size = 32
    
    history = model.fit(
        X_train,
        {'regression': y_train_scaled, 'classification': y_train_cls},
        validation_data=(
            X_val,
            {'regression': y_val_scaled, 'classification': y_val_cls}
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
    
    # 역변환 (중요!)
    y_pred = processor.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    # 메트릭 계산
    mae = mean_absolute_error(y_test_orig, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
    r2 = r2_score(y_test_orig, y_pred)
    
    # R² 디버깅
    print(f"\n🔍 R² 디버깅:")
    print(f"  y_test 범위: {y_test_orig.min():.0f} ~ {y_test_orig.max():.0f}")
    print(f"  y_pred 범위: {y_pred.min():.0f} ~ {y_pred.max():.0f}")
    print(f"  y_test 평균: {y_test_orig.mean():.0f}")
    print(f"  y_pred 평균: {y_pred.mean():.0f}")
    print(f"  y_test 표준편차: {y_test_orig.std():.0f}")
    print(f"  y_pred 표준편차: {y_pred.std():.0f}")
    
    # 상관계수
    correlation = np.corrcoef(y_test_orig, y_pred)[0, 1]
    print(f"  상관계수: {correlation:.4f}")
    
    print(f"\n📈 회귀 성능:")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²: {r2:.4f}")
    
    # 분류 성능
    acc = accuracy_score(y_test_cls, y_cls_pred)
    print(f"\n📊 분류 성능:")
    print(f"  정확도: {acc:.4f}")
    
    # 모델 저장
    os.makedirs('models', exist_ok=True)
    model.save('models/extreme_net_final.keras')
    processor.save_all()
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Accuracy': acc,
        'Correlation': correlation
    }

# ====================================
# 실시간 예측
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
        
        print("✅ 예측기 준비 완료")
    
    def predict(self, data_100min):
        """100분 데이터로 예측"""
        if len(data_100min) != 100:
            raise ValueError(f"100분 데이터 필요 (현재: {len(data_100min)})")
        
        # 패턴 감지
        pattern, confidence = self.processor.pattern_detector.detect_pattern(data_100min)
        
        # 특징 준비
        X = data_100min[self.processor.feature_columns].values
        X = X.reshape(1, 100, -1)
        
        # 패턴 특징
        pattern_features = self.processor.pattern_detector.create_pattern_features(X)
        
        # 스케일링
        X_scaled = self.processor.scaler_X.transform(X.reshape(-1, X.shape[2])).reshape(X.shape)
        
        # 예측
        preds = self.model.predict([X_scaled, pattern_features], verbose=0)
        y_pred_scaled = preds[0][0, 0]
        y_cls_probs = preds[1][0]
        
        # 역변환
        y_pred = self.processor.scaler_y.inverse_transform([[y_pred_scaled]])[0, 0]
        
        # 패턴 기반 조정
        current = data_100min['TOTALCNT'].iloc[-1]
        m14b = data_100min['M14AM14B'].iloc[-1]
        m14a = data_100min['M14AM10A'].iloc[-1]
        
        if pattern == "UU1" and m14b > 300 and m14a < 80:
            y_pred *= 1.1  # 황금 패턴
        elif pattern == "UU2":
            y_pred *= 0.98  # 보수적
        
        return {
            'prediction': float(y_pred),
            'current': float(current),
            'change': float(y_pred - current),
            'pattern': pattern,
            'confidence': confidence,
            'level_probs': y_cls_probs.tolist()
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
    
    # 모델 생성
    input_shape = (100, len(features))
    model = build_extreme_net_fixed(input_shape)
    
    # 학습
    results = train_model(model, train_data, val_data, test_data, processor)
    
    # 결과 출력
    print("\n" + "="*80)
    print("🏆 최종 성능")
    print("="*80)
    print(f"  R²: {results['R2']:.4f}")
    print(f"  MAE: {results['MAE']:.2f}")
    print(f"  정확도: {results['Accuracy']:.4f}")
    print(f"  상관계수: {results['Correlation']:.4f}")
    
    # 실시간 테스트
    if len(df) >= 100:
        print("\n🔮 실시간 예측 테스트...")
        predictor = RealtimePredictor()
        test_data = df.iloc[-100:].copy()
        result = predictor.predict(test_data)
        
        print(f"  패턴: {result['pattern']}")
        print(f"  현재: {result['current']:.0f}")
        print(f"  예측: {result['prediction']:.0f}")
        print(f"  변화: {result['change']:+.0f}")
    
    print("\n✅ 완료!")
    return results

if __name__ == "__main__":
    results = main()