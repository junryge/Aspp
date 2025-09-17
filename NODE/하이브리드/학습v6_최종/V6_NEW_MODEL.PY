"""
🚀 반도체 물류 예측 시스템 - 78만개 데이터 완전판
================================================
전체 기능 포함 - 학습 중단/재개, 체크포인트, MAE/RMSE/R2
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import *
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, confusion_matrix
import warnings
import os
import pickle
import gc
from datetime import datetime

warnings.filterwarnings('ignore')
tf.random.set_seed(42)
np.random.seed(42)

# GPU 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

print("="*80)
print("🚀 반도체 물류 예측 - 78만개 데이터 완전판")
print(f"📦 TensorFlow: {tf.__version__}")
print(f"🔧 Device: {'GPU' if gpus else 'CPU'}")
print("="*80)

# ========================================
# 데이터 처리 클래스
# ========================================

class DataProcessor:
    def __init__(self, seq_len=100, pred_len=10):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.feature_scaler = RobustScaler()
        self.target_scaler = MinMaxScaler()
        
    def save_scalers(self):
        with open('scalers.pkl', 'wb') as f:
            pickle.dump({
                'feature_scaler': self.feature_scaler,
                'target_scaler': self.target_scaler
            }, f)
        print("💾 스케일러 저장 완료")
        
    def load_and_process(self, filepath):
        print(f"\n📂 데이터 로딩: {filepath}")
        df = pd.read_csv(filepath)
        print(f"✅ 데이터: {df.shape[0]:,}행 × {df.shape[1]}열")
        
        # 타입 최적화
        for col in df.columns:
            if df[col].dtype == 'float64':
                df[col] = df[col].astype('float32')
            elif df[col].dtype == 'int64':
                df[col] = df[col].astype('int32')
        
        df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), 
                                       format='%Y%m%d%H%M', errors='coerce')
        df = df.sort_values('CURRTIME').reset_index(drop=True)
        
        print(f"\n📊 TOTALCNT 분포:")
        print(f"  최소: {df['TOTALCNT'].min():,}")
        print(f"  평균: {df['TOTALCNT'].mean():,.0f}")
        print(f"  최대: {df['TOTALCNT'].max():,}")
        
        normal = (df['TOTALCNT'] < 1400).sum()
        warning = ((df['TOTALCNT'] >= 1400) & (df['TOTALCNT'] < 1700)).sum()
        critical = (df['TOTALCNT'] >= 1700).sum()
        
        print(f"\n📊 구간 분포:")
        print(f"  정상: {normal:,}개 ({normal/len(df)*100:.2f}%)")
        print(f"  주의: {warning:,}개 ({warning/len(df)*100:.2f}%)")
        print(f"  심각: {critical:,}개 ({critical/len(df)*100:.2f}%)")
        
        return df
    
    def create_features(self, df):
        print("\n⚙️ 특성 생성...")
        
        # 황금 패턴
        df['RATIO'] = df['M14AM14B'] / (df['M14AM10A'] + 1)
        df['GOLDEN_1'] = ((df['M14AM14B'] > 300) & (df['M14AM10A'] < 80)).astype('float32')
        df['GOLDEN_2'] = (df['M14AM14BSUM'] > 500).astype('float32')
        
        # 시간
        df['HOUR'] = df['CURRTIME'].dt.hour.fillna(0).astype('float32')
        df['HOUR_SIN'] = np.sin(2 * np.pi * df['HOUR'] / 24).astype('float32')
        df['HOUR_COS'] = np.cos(2 * np.pi * df['HOUR'] / 24).astype('float32')
        
        # 이동평균
        print("  이동평균 계산...")
        for w in [10, 30]:
            df[f'MA_{w}'] = df['TOTALCNT'].rolling(w, min_periods=1).mean().astype('float32')
            df[f'STD_{w}'] = df['TOTALCNT'].rolling(w, min_periods=1).std().fillna(0).astype('float32')
        
        # 변화율
        df['CHANGE_1'] = df['TOTALCNT'].diff(1).fillna(0).astype('float32')
        df['CHANGE_10'] = df['TOTALCNT'].diff(10).fillna(0).astype('float32')
        
        # 타겟 생성
        df['TARGET'] = df['TOTALCNT'].shift(-self.pred_len)
        
        # 마지막 pred_len개 제거 (타겟이 없음)
        df = df[:-self.pred_len].copy()
        
        # 레벨 생성
        df['LEVEL'] = 0
        df.loc[df['TARGET'] >= 1400, 'LEVEL'] = 1
        df.loc[df['TARGET'] >= 1700, 'LEVEL'] = 2
        df['LEVEL'] = df['LEVEL'].astype('int32')
        
        print(f"✅ 특성 생성 완료: {len(df)}행, {len(df.columns)}개 컬럼")
        return df
    
    def create_sequences(self, df, batch_size=10000):
        print("\n🔄 시퀀스 생성...")
        
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
        
        X_data = df[features].values.astype('float32')
        y_reg = df['TARGET'].values.astype('float32')
        y_cls = df['LEVEL'].values.astype('int32')
        
        print(f"  데이터 shape: {X_data.shape}")
        
        # 스케일링
        print("  스케일링...")
        X_scaled = self.feature_scaler.fit_transform(X_data)
        y_reg_scaled = self.target_scaler.fit_transform(y_reg.reshape(-1, 1)).flatten()
        self.save_scalers()
        
        # 시퀀스 생성
        total_sequences = len(X_scaled) - self.seq_len
        print(f"  총 시퀀스: {total_sequences:,}개")
        
        X_list, y_r_list, y_c_list = [], [], []
        
        for start_idx in range(0, total_sequences, batch_size):
            end_idx = min(start_idx + batch_size, total_sequences)
            
            batch_X = []
            batch_y_r = []
            batch_y_c = []
            
            for i in range(start_idx, end_idx):
                batch_X.append(X_scaled[i:i+self.seq_len])
                batch_y_r.append(y_reg_scaled[i+self.seq_len-1])
                batch_y_c.append(y_cls[i+self.seq_len-1])
            
            X_list.append(np.array(batch_X, dtype='float32'))
            y_r_list.append(np.array(batch_y_r, dtype='float32'))
            y_c_list.append(np.array(batch_y_c, dtype='int32'))
            
            if (end_idx % 50000) == 0:
                print(f"    처리: {end_idx:,}/{total_sequences:,}")
        
        # 합치기
        X = np.concatenate(X_list, axis=0)
        y_r = np.concatenate(y_r_list, axis=0)
        y_c = np.concatenate(y_c_list, axis=0)
        
        print(f"✅ 시퀀스 생성 완료: {X.shape}")
        
        del X_list, y_r_list, y_c_list
        gc.collect()
        
        return X, y_r, y_c

# ========================================
# 5개 모델 정의
# ========================================

def build_stable_lstm(input_shape):
    """1. StableLSTM"""
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(64))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    out_reg = Dense(1, name='regression')(x)
    out_cls = Dense(3, activation='softmax', name='classification')(x)
    return Model(inputs, [out_reg, out_cls], name='StableLSTM')

def build_patch_tst(input_shape):
    """2. PatchTST"""
    inputs = Input(shape=input_shape)
    x = Dense(128)(inputs)
    x = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = LayerNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    out_reg = Dense(1, name='regression')(x)
    out_cls = Dense(3, activation='softmax', name='classification')(x)
    return Model(inputs, [out_reg, out_cls], name='PatchTST')

def build_spike_detector(input_shape):
    """3. SpikeDetector"""
    inputs = Input(shape=input_shape)
    recent = Lambda(lambda x: x[:, -20:, :])(inputs)
    x = Conv1D(64, 3, activation='relu', padding='same')(recent)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = GRU(64)(x)
    x = BatchNormalization()(x)
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
    x = BatchNormalization()(x)
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
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    out_reg = Dense(1, name='regression')(x)
    out_cls = Dense(3, activation='softmax', name='classification')(x)
    return Model(inputs, [out_reg, out_cls], name='EnsembleBase')

# ========================================
# 학습 및 평가
# ========================================

def train_and_evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test, processor):
    
    # 가중치 손실 함수
    def weighted_crossentropy(y_true, y_pred):
        weights = tf.constant([1.0, 5.0, 200.0])
        scce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        weight_map = tf.gather(weights, tf.cast(y_true, tf.int32))
        return tf.reduce_mean(scce * weight_map)
    
    model.compile(
        optimizer=Adam(0.001),
        loss=['mse', weighted_crossentropy],
        loss_weights=[0.3, 0.7],
        metrics={'regression': 'mae', 'classification': 'accuracy'}
    )
    
    print(f"\n{'='*60}")
    print(f"🎯 {model.name} 학습")
    print(f"{'='*60}")
    
    # Callbacks
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            f'{checkpoint_dir}/{model.name}_best.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    ]
    
    # 학습
    history = model.fit(
        X_train,
        {'regression': y_train[0], 'classification': y_train[1]},
        validation_data=(X_val, {'regression': y_val[0], 'classification': y_val[1]}),
        epochs=30,
        batch_size=512,
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"\n📊 {model.name} 평가 결과:")
    
    # 예측
    preds = model.predict(X_test, batch_size=1024, verbose=0)
    y_reg_pred = preds[0].flatten()
    y_cls_pred = np.argmax(preds[1], axis=1)
    
    # 역변환
    y_reg_true_orig = processor.target_scaler.inverse_transform(
        y_test[0].reshape(-1, 1)).flatten()
    y_reg_pred_orig = processor.target_scaler.inverse_transform(
        y_reg_pred.reshape(-1, 1)).flatten()
    
    # 메트릭
    mae = mean_absolute_error(y_reg_true_orig, y_reg_pred_orig)
    rmse = np.sqrt(mean_squared_error(y_reg_true_orig, y_reg_pred_orig))
    r2 = r2_score(y_reg_true_orig, y_reg_pred_orig)
    acc = accuracy_score(y_test[1], y_cls_pred)
    
    # 혼동 행렬
    cm = confusion_matrix(y_test[1], y_cls_pred)
    
    print(f"\n[회귀 성능]")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²:   {r2:.4f}")
    
    print(f"\n[분류 성능]")
    print(f"  전체 정확도: {acc:.2%}")
    
    print(f"\n[혼동 행렬]")
    for i in range(min(3, len(cm))):
        if i == 0:
            label = "정상"
        elif i == 1:
            label = "주의"
        else:
            label = "심각"
        if i < len(cm):
            correct = cm[i, i] if i < cm.shape[1] else 0
            total = cm[i].sum()
            print(f"  {label}: {correct}/{total} ({correct/total*100:.1f}%)")
    
    # 모델 저장
    os.makedirs('models', exist_ok=True)
    model.save(f'models/{model.name}_final.keras')
    print(f"\n💾 최종 모델: models/{model.name}_final.keras")
    
    return {
        'MAE': mae, 
        'RMSE': rmse, 
        'R2': r2, 
        'Accuracy': acc
    }

# ========================================
# 메인 실행
# ========================================

def main():
    # 데이터 경로
    data_paths = [
        '/mnt/user-data/uploads/gs.CSV',
        'data/20250731_to20250806.csv',
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
    
    # 디렉토리 생성
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # 데이터 처리
    processor = DataProcessor()
    df = processor.load_and_process(data_path)
    df = processor.create_features(df)
    X, y_reg, y_cls = processor.create_sequences(df)
    
    # 메모리 정리
    del df
    gc.collect()
    
    # 데이터 분할
    n_train = int(0.7 * len(X))
    n_val = int(0.15 * len(X))
    
    print(f"\n📊 데이터 분할:")
    print(f"  전체: {len(X):,}개")
    print(f"  Train: {n_train:,}개 (70%)")
    print(f"  Val: {n_val:,}개 (15%)")
    print(f"  Test: {len(X) - n_train - n_val:,}개 (15%)")
    
    X_train = X[:n_train]
    X_val = X[n_train:n_train+n_val]
    X_test = X[n_train+n_val:]
    
    y_train = (y_reg[:n_train], y_cls[:n_train])
    y_val = (y_reg[n_train:n_train+n_val], y_cls[n_train:n_train+n_val])
    y_test = (y_reg[n_train+n_val:], y_cls[n_train+n_val:])
    
    # 5개 모델 생성
    input_shape = (X.shape[1], X.shape[2])
    
    models = {
        'StableLSTM': build_stable_lstm(input_shape),
        'PatchTST': build_patch_tst(input_shape),
        'SpikeDetector': build_spike_detector(input_shape),
        'ExtremeNet': build_extreme_net(input_shape),
        'EnsembleBase': build_ensemble_base(input_shape)
    }
    
    print(f"\n✅ 5개 모델 준비 완료")
    
    # 학습 및 평가
    results = {}
    for name, model in models.items():
        try:
            results[name] = train_and_evaluate(
                model, X_train, y_train, X_val, y_val, X_test, y_test, processor
            )
            tf.keras.backend.clear_session()
            gc.collect()
        except Exception as e:
            print(f"❌ {name} 학습 실패: {e}")
            continue
    
    # 최종 결과
    print("\n" + "="*80)
    print("📊 최종 성능 비교")
    print("="*80)
    
    results_df = pd.DataFrame(results).T
    print("\n", results_df.round(3))
    
    # 최고 성능
    print("\n🏆 최고 성능:")
    print(f"  MAE 최저: {results_df['MAE'].idxmin()} ({results_df['MAE'].min():.2f})")
    print(f"  RMSE 최저: {results_df['RMSE'].idxmin()} ({results_df['RMSE'].min():.2f})")
    print(f"  R² 최고: {results_df['R2'].idxmax()} ({results_df['R2'].max():.4f})")
    print(f"  정확도 최고: {results_df['Accuracy'].idxmax()} ({results_df['Accuracy'].max():.2%})")
    
    results_df.to_csv('model_results.csv')
    print(f"\n💾 결과 저장: model_results.csv")
    print("\n✅ 학습 완료!")
    
    return results

if __name__ == "__main__":
    results = main()