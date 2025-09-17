"""
🚀 반도체 물류 예측 - 5개 모델 완전판
=====================================
1. StableLSTM
2. PatchTST  
3. SpikeDetector
4. ExtremeNet
5. EnsembleBase

모든 모델 포함, MAE/RMSE/R2/정확도 출력
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import *
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, confusion_matrix
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')
tf.random.set_seed(42)
np.random.seed(42)

print("="*80)
print("🚀 반도체 물류 예측 - 5개 모델 완전판")
print(f"📦 TensorFlow: {tf.__version__}")
print("="*80)

# 데이터 처리
class DataProcessor:
    def __init__(self, seq_len=100, pred_len=10):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.feature_scaler = RobustScaler()
        self.target_scaler = MinMaxScaler()
        
    def load_and_process(self, filepath):
        print(f"\n📂 데이터 로딩: {filepath}")
        df = pd.read_csv(filepath)
        print(f"✅ 데이터: {df.shape[0]:,}행 × {df.shape[1]}열")
        
        df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M', errors='coerce')
        df = df.sort_values('CURRTIME').reset_index(drop=True)
        
        print(f"\n📊 TOTALCNT 분포:")
        print(f"  최소: {df['TOTALCNT'].min():,}")
        print(f"  평균: {df['TOTALCNT'].mean():,.0f}")
        print(f"  최대: {df['TOTALCNT'].max():,}")
        
        normal = (df['TOTALCNT'] < 1400).sum()
        warning = ((df['TOTALCNT'] >= 1400) & (df['TOTALCNT'] < 1700)).sum()
        critical = (df['TOTALCNT'] >= 1700).sum()
        
        print(f"\n📊 구간 분포:")
        print(f"  정상: {normal:,}개 ({normal/len(df)*100:.1f}%)")
        print(f"  주의: {warning:,}개 ({warning/len(df)*100:.1f}%)")
        print(f"  심각: {critical:,}개 ({critical/len(df)*100:.1f}%)")
        
        return df
    
    def create_features(self, df):
        print("\n⚙️ 특성 생성...")
        
        # 황금 패턴
        df['RATIO'] = df['M14AM14B'] / (df['M14AM10A'] + 1)
        df['GOLDEN_1'] = ((df['M14AM14B'] > 300) & (df['M14AM10A'] < 80)).astype(int)
        df['GOLDEN_2'] = (df['M14AM14BSUM'] > 500).astype(int)
        
        # 시간
        df['HOUR'] = df['CURRTIME'].dt.hour
        df['HOUR_SIN'] = np.sin(2 * np.pi * df['HOUR'] / 24)
        df['HOUR_COS'] = np.cos(2 * np.pi * df['HOUR'] / 24)
        
        # 이동평균
        for w in [10, 30]:
            df[f'MA_{w}'] = df['TOTALCNT'].rolling(w, min_periods=1).mean()
            df[f'STD_{w}'] = df['TOTALCNT'].rolling(w, min_periods=1).std().fillna(0)
        
        # 변화율
        df['CHANGE_1'] = df['TOTALCNT'].diff(1).fillna(0)
        df['CHANGE_10'] = df['TOTALCNT'].diff(10).fillna(0)
        
        # 타겟
        df['TARGET'] = df['TOTALCNT'].shift(-self.pred_len)
        df['LEVEL'] = pd.cut(df['TARGET'], bins=[-np.inf, 1400, 1700, np.inf], labels=[0, 1, 2])
        
        return df
    
    def create_sequences(self, df):
        print("\n🔄 시퀀스 생성...")
        df = df.dropna()
        
        features = [
            'TOTALCNT', 'M14AM14B', 'M14AM14BSUM', 'M14AM10A', 'M14AM16',
            'M14AM16SUM', 'M10AM14A', 'M14BM14A', 'M16M14A', 'M14AM10ASUM',
            'RATIO', 'GOLDEN_1', 'GOLDEN_2',
            'HOUR_SIN', 'HOUR_COS',
            'MA_10', 'MA_30', 'STD_10', 'STD_30',
            'CHANGE_1', 'CHANGE_10'
        ]
        
        features = [f for f in features if f in df.columns]
        print(f"  특성: {len(features)}개")
        
        X_data = df[features].values
        y_reg = df['TARGET'].values
        y_cls = df['LEVEL'].astype(int).values
        
        X_scaled = self.feature_scaler.fit_transform(X_data)
        y_reg_scaled = self.target_scaler.fit_transform(y_reg.reshape(-1, 1)).flatten()
        
        X, y_r, y_c = [], [], []
        for i in range(len(X_scaled) - self.seq_len):
            X.append(X_scaled[i:i+self.seq_len])
            y_r.append(y_reg_scaled[i+self.seq_len-1])
            y_c.append(y_cls[i+self.seq_len-1])
        
        return np.array(X, dtype=np.float32), np.array(y_r, dtype=np.float32), np.array(y_c, dtype=np.int32)

# ===== 5개 모델 정의 =====

def build_stable_lstm(input_shape):
    """1. StableLSTM"""
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(64))(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    out_reg = Dense(1, name='regression')(x)
    out_cls = Dense(3, activation='softmax', name='classification')(x)
    return Model(inputs, [out_reg, out_cls], name='StableLSTM')

def build_patch_tst(input_shape):
    """2. PatchTST"""
    inputs = Input(shape=input_shape)
    x = Dense(128)(inputs)
    x = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    out_reg = Dense(1, name='regression')(x)
    out_cls = Dense(3, activation='softmax', name='classification')(x)
    return Model(inputs, [out_reg, out_cls], name='PatchTST')

def build_spike_detector(input_shape):
    """3. SpikeDetector"""
    inputs = Input(shape=input_shape)
    recent = Lambda(lambda x: x[:, -20:, :])(inputs)
    x = Conv1D(64, 3, activation='relu', padding='same')(recent)
    x = MaxPooling1D(2)(x)
    x = GRU(64)(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    out_reg = Dense(1, name='regression')(x)
    out_cls = Dense(3, activation='softmax', name='classification')(x)
    return Model(inputs, [out_reg, out_cls], name='SpikeDetector')

def build_extreme_net(input_shape):
    """4. ExtremeNet"""
    inputs = Input(shape=input_shape)
    normal = LSTM(64, return_sequences=True)(inputs)
    normal = LSTM(32)(normal)
    extreme = MultiHeadAttention(num_heads=4, key_dim=16)(inputs, inputs)
    extreme = GlobalAveragePooling1D()(extreme)
    x = Concatenate()([normal, extreme])
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    out_reg = Dense(1, name='regression')(x)
    out_cls = Dense(3, activation='softmax', name='classification')(x)
    return Model(inputs, [out_reg, out_cls], name='ExtremeNet')

def build_ensemble_base(input_shape):
    """5. EnsembleBase"""
    inputs = Input(shape=input_shape)
    avg_pool = GlobalAveragePooling1D()(inputs)
    max_pool = GlobalMaxPooling1D()(inputs)
    x = Concatenate()([avg_pool, max_pool])
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    out_reg = Dense(1, name='regression')(x)
    out_cls = Dense(3, activation='softmax', name='classification')(x)
    return Model(inputs, [out_reg, out_cls], name='EnsembleBase')

# 학습 및 평가
def train_and_evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test, processor):
    
    model.compile(
        optimizer=Adam(0.001),
        loss=['mse', 'sparse_categorical_crossentropy'],
        loss_weights=[0.3, 0.7],
        metrics={'regression': 'mae', 'classification': 'accuracy'}
    )
    
    print(f"\n{'='*60}")
    print(f"🎯 {model.name} 학습")
    print(f"{'='*60}")
    
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(factor=0.5, patience=3, verbose=0)
    ]
    
    history = model.fit(
        X_train,
        {'regression': y_train[0], 'classification': y_train[1]},
        validation_data=(X_val, {'regression': y_val[0], 'classification': y_val[1]}),
        epochs=20,
        batch_size=256,
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"\n📊 {model.name} 평가 결과:")
    
    preds = model.predict(X_test, verbose=0)
    y_reg_pred = preds[0].flatten()
    y_cls_pred = np.argmax(preds[1], axis=1)
    
    y_reg_true_orig = processor.target_scaler.inverse_transform(y_test[0].reshape(-1, 1)).flatten()
    y_reg_pred_orig = processor.target_scaler.inverse_transform(y_reg_pred.reshape(-1, 1)).flatten()
    
    mae = mean_absolute_error(y_reg_true_orig, y_reg_pred_orig)
    rmse = np.sqrt(mean_squared_error(y_reg_true_orig, y_reg_pred_orig))
    r2 = r2_score(y_reg_true_orig, y_reg_pred_orig)
    acc = accuracy_score(y_test[1], y_cls_pred)
    
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  정확도: {acc:.2%}")
    
    os.makedirs('models', exist_ok=True)
    model.save(f'models/{model.name}.keras')
    print(f"💾 모델 저장: models/{model.name}.keras")
    
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'Accuracy': acc}

# 메인 실행
def main():
    # 데이터 경로
    data_paths = [
        'data/20250731_to20250806.csv',
        '/mnt/user-data/uploads/gs.CSV',
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
        return
    
    # 데이터 처리
    processor = DataProcessor()
    df = processor.load_and_process(data_path)
    df = processor.create_features(df)
    X, y_reg, y_cls = processor.create_sequences(df)
    
    # 데이터 분할
    n_train = int(0.7 * len(X))
    n_val = int(0.15 * len(X))
    
    X_train = X[:n_train]
    X_val = X[n_train:n_train+n_val]
    X_test = X[n_train+n_val:]
    
    y_train = (y_reg[:n_train], y_cls[:n_train])
    y_val = (y_reg[n_train:n_train+n_val], y_cls[n_train:n_train+n_val])
    y_test = (y_reg[n_train+n_val:], y_cls[n_train+n_val:])
    
    print(f"\n📊 데이터 분할:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")
    
    # 5개 모델 생성 및 학습
    input_shape = (X.shape[1], X.shape[2])
    
    models = {
        'StableLSTM': build_stable_lstm(input_shape),
        'PatchTST': build_patch_tst(input_shape),
        'SpikeDetector': build_spike_detector(input_shape),
        'ExtremeNet': build_extreme_net(input_shape),
        'EnsembleBase': build_ensemble_base(input_shape)
    }
    
    print(f"\n✅ 5개 모델 준비 완료:")
    for name in models.keys():
        print(f"  - {name}")
    
    results = {}
    for name, model in models.items():
        results[name] = train_and_evaluate(
            model, X_train, y_train, X_val, y_val, X_test, y_test, processor
        )
    
    # 최종 결과
    print("\n" + "="*80)
    print("📊 최종 성능 비교 (5개 모델)")
    print("="*80)
    
    results_df = pd.DataFrame(results).T
    print("\n", results_df.round(3))
    
    # 최고 성능
    print("\n🏆 최고 성능:")
    print(f"  MAE 최저: {results_df['MAE'].idxmin()} ({results_df['MAE'].min():.2f})")
    print(f"  R² 최고: {results_df['R2'].idxmax()} ({results_df['R2'].max():.4f})")
    print(f"  정확도 최고: {results_df['Accuracy'].idxmax()} ({results_df['Accuracy'].max():.2%})")
    
    results_df.to_csv('model_results.csv')
    print("\n💾 결과 저장: model_results.csv")
    print("\n✅ 5개 모델 학습 완료!")
    
    return results

if __name__ == "__main__":
    results = main()