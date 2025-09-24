"""
🚨 EXTREME_NET 긴급 수정 - R² 문제 완전 해결
===========================================
스케일링과 R² 계산 문제 수정
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, Input, backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import *
from sklearn.preprocessing import StandardScaler  # MinMaxScaler 대신!
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import os

warnings.filterwarnings('ignore')
tf.random.set_seed(42)
np.random.seed(42)

print("="*80)
print("🚨 긴급 수정 버전 - R² 문제 해결")
print("="*80)

# ====================================
# 수정된 R² 메트릭
# ====================================
class R2Score(tf.keras.metrics.Metric):
    """올바른 R² 구현"""
    def __init__(self, name='r2_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.ss_res = self.add_weight(name='ss_res', initializer='zeros')
        self.ss_tot = self.add_weight(name='ss_tot', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
        self.y_sum = self.add_weight(name='y_sum', initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        
        # 평균 계산
        batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)
        self.count.assign_add(batch_size)
        self.y_sum.assign_add(tf.reduce_sum(y_true))
        y_mean = self.y_sum / self.count
        
        # SS 계산
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
# 단순화된 모델 (회귀에 집중)
# ====================================
def create_simple_model(input_shape):
    """단순하지만 효과적인 모델"""
    
    inputs = Input(shape=input_shape)
    
    # 단순 LSTM
    x = LSTM(64, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(32)(x)
    x = Dropout(0.2)(x)
    
    # Dense
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    
    # 회귀 출력만
    output = Dense(1, activation='linear')(x)  # linear 중요!
    
    model = Model(inputs, output, name='SimpleExtreme')
    
    # 컴파일 (회귀만 집중)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=[R2Score(), 'mae']
    )
    
    return model

# ====================================
# 데이터 처리 (수정)
# ====================================
class DataProcessor:
    def __init__(self):
        self.scaler_X = StandardScaler()  # MinMax 대신 Standard
        self.scaler_y = StandardScaler()  # 타겟도 Standard
        
    def prepare_data(self, df):
        """데이터 준비"""
        print("\n📊 데이터 준비...")
        
        # 0값과 이상치 제거
        df = df[df['TOTALCNT'] > 0].reset_index(drop=True)
        df = df[df['TOTALCNT'] < 5000].reset_index(drop=True)  # 이상치 제거
        
        # 로그 변환 (분포 정규화)
        df['TOTALCNT_LOG'] = np.log1p(df['TOTALCNT'])
        
        # 기본 특징
        features = []
        if 'M14AM14B' in df.columns:
            features.append('M14AM14B')
            df['M14AM14B_LOG'] = np.log1p(df['M14AM14B'])
            features.append('M14AM14B_LOG')
        if 'M14AM10A' in df.columns:
            features.append('M14AM10A')
        
        # 이동평균
        df['MA_10'] = df['TOTALCNT'].rolling(10, min_periods=1).mean()
        df['MA_30'] = df['TOTALCNT'].rolling(30, min_periods=1).mean()
        features.extend(['MA_10', 'MA_30'])
        
        # 변화율
        df['CHANGE_10'] = df['TOTALCNT'].diff(10).fillna(0)
        features.append('CHANGE_10')
        
        # TOTALCNT도 특징에 포함
        features.insert(0, 'TOTALCNT')
        
        print(f"  특징: {features}")
        
        X = df[features].values
        y = df['TOTALCNT'].values  # 로그 변환 안 함
        
        print(f"  X shape: {X.shape}")
        print(f"  y 범위: {y.min():.0f} ~ {y.max():.0f}")
        print(f"  y 평균: {y.mean():.0f} ± {y.std():.0f}")
        
        return X, y
    
    def create_sequences(self, X, y, seq_len=100, pred_len=10):
        """시퀀스 생성"""
        print("\n🔄 시퀀스 생성...")
        
        # NaN 처리
        X = np.nan_to_num(X, 0)
        
        sequences_X = []
        sequences_y = []
        
        for i in range(len(X) - seq_len - pred_len + 1):
            seq_x = X[i:i + seq_len]
            target = y[i + seq_len + pred_len - 1]
            
            sequences_X.append(seq_x)
            sequences_y.append(target)
        
        sequences_X = np.array(sequences_X)
        sequences_y = np.array(sequences_y)
        
        print(f"  시퀀스: {sequences_X.shape}")
        
        # 분할
        n = len(sequences_X)
        train_size = int(n * 0.7)
        val_size = int(n * 0.15)
        
        X_train = sequences_X[:train_size]
        y_train = sequences_y[:train_size]
        
        X_val = sequences_X[train_size:train_size + val_size]
        y_val = sequences_y[train_size:train_size + val_size]
        
        X_test = sequences_X[train_size + val_size:]
        y_test = sequences_y[train_size + val_size:]
        
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
        
        # 스케일링 검증
        print(f"  X_train 범위: [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]")
        print(f"  y_train 범위: [{y_train_scaled.min():.2f}, {y_train_scaled.max():.2f}]")
        print(f"  y_train 평균: {y_train_scaled.mean():.2f} ± {y_train_scaled.std():.2f}")
        
        return (X_train_scaled, y_train_scaled, y_train), \
               (X_val_scaled, y_val_scaled, y_val), \
               (X_test_scaled, y_test_scaled, y_test)

# ====================================
# 학습
# ====================================
def train_model(model, train_data, val_data, test_data, processor):
    """학습"""
    X_train, y_train_scaled, y_train_orig = train_data
    X_val, y_val_scaled, y_val_orig = val_data
    X_test, y_test_scaled, y_test_orig = test_data
    
    print("\n🎯 학습 시작...")
    
    # 콜백
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]
    
    # 학습
    history = model.fit(
        X_train, y_train_scaled,
        validation_data=(X_val, y_val_scaled),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # 평가
    print("\n📊 평가...")
    
    # 예측
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    
    # 역변환 (중요!)
    y_pred = processor.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    # 메트릭 (원본 스케일에서)
    mae = mean_absolute_error(y_test_orig, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred))
    r2 = r2_score(y_test_orig, y_pred)
    
    print(f"\n📈 최종 성능 (원본 스케일):")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²: {r2:.4f}")
    
    # R² 검증
    if r2 < 0:
        print("\n⚠️ R²가 음수입니다! 확인 필요:")
        print(f"  y_test 평균: {y_test_orig.mean():.0f}")
        print(f"  y_pred 평균: {y_pred.mean():.0f}")
        print(f"  y_test 표준편차: {y_test_orig.std():.0f}")
        print(f"  y_pred 표준편차: {y_pred.std():.0f}")
        
        # 상관계수
        corr = np.corrcoef(y_test_orig, y_pred)[0, 1]
        print(f"  상관계수: {corr:.4f}")
    
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}

# ====================================
# 메인
# ====================================
def main():
    print("\n🚨 긴급 수정 버전 실행")
    
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
    X, y = processor.prepare_data(df)
    train_data, val_data, test_data = processor.create_sequences(X, y)
    
    # 모델 (단순 버전)
    input_shape = (100, X.shape[1])
    model = create_simple_model(input_shape)
    
    # 학습
    results = train_model(model, train_data, val_data, test_data, processor)
    
    # 결과
    print("\n" + "="*60)
    print("🏆 최종 결과")
    print("="*60)
    print(f"R²: {results['R2']:.4f}")
    print(f"MAE: {results['MAE']:.2f}")
    
    if results['R2'] > 0.5:
        print("✅ 성공!")
    else:
        print("❌ 개선 필요")
    
    return results

if __name__ == "__main__":
    results = main()