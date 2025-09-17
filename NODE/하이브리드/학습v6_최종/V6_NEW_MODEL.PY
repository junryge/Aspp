"""
🚀 반도체 물류 예측 - 78만개 데이터 완전판
=========================================
대용량 데이터 처리 최적화
- 배치 처리
- 메모리 효율화
- 체크포인트 저장
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
import gc
import pickle
from datetime import datetime

warnings.filterwarnings('ignore')
tf.random.set_seed(42)
np.random.seed(42)

# GPU 메모리 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

print("="*80)
print("🚀 반도체 물류 예측 - 78만개 데이터 버전")
print(f"📦 TensorFlow: {tf.__version__}")
print(f"🔧 Device: {'GPU' if gpus else 'CPU'}")
print("="*80)

# 데이터 처리 (대용량 최적화)
class DataProcessor:
    def __init__(self, seq_len=100, pred_len=10):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.feature_scaler = RobustScaler()
        self.target_scaler = MinMaxScaler()
        self.scaler_fitted = False
        
    def save_scalers(self):
        """스케일러 저장"""
        with open('scalers.pkl', 'wb') as f:
            pickle.dump({
                'feature_scaler': self.feature_scaler,
                'target_scaler': self.target_scaler
            }, f)
        print("💾 스케일러 저장 완료")
    
    def load_scalers(self):
        """스케일러 로드"""
        if os.path.exists('scalers.pkl'):
            with open('scalers.pkl', 'rb') as f:
                scalers = pickle.load(f)
                self.feature_scaler = scalers['feature_scaler']
                self.target_scaler = scalers['target_scaler']
                self.scaler_fitted = True
            print("📂 스케일러 로드 완료")
            return True
        return False
        
    def load_and_process(self, filepath):
        """대용량 데이터 로드"""
        print(f"\n📂 데이터 로딩: {filepath}")
        
        # 청크 단위로 읽기
        df = pd.read_csv(filepath, low_memory=False)
        print(f"✅ 데이터: {df.shape[0]:,}행 × {df.shape[1]}열")
        
        # 메모리 최적화
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
        """특성 생성 (메모리 효율적)"""
        print("\n⚙️ 특성 생성...")
        
        # 황금 패턴
        df['RATIO'] = (df['M14AM14B'] / (df['M14AM10A'] + 1)).astype('float32')
        df['GOLDEN_1'] = ((df['M14AM14B'] > 300) & (df['M14AM10A'] < 80)).astype('float32')
        df['GOLDEN_2'] = (df['M14AM14BSUM'] > 500).astype('float32')
        
        # 시간 (NaN 처리)
        df['HOUR'] = df['CURRTIME'].dt.hour
        df['HOUR'] = df['HOUR'].fillna(0).astype('float32')  # NaN을 0으로 채우고 float32로
        df['HOUR_SIN'] = np.sin(2 * np.pi * df['HOUR'] / 24).astype('float32')
        df['HOUR_COS'] = np.cos(2 * np.pi * df['HOUR'] / 24).astype('float32')
        
        # 이동평균 (효율적 계산)
        print("  이동평균 계산...")
        for w in [10, 30]:
            df[f'MA_{w}'] = df['TOTALCNT'].rolling(w, min_periods=1).mean().astype('float32')
            df[f'STD_{w}'] = df['TOTALCNT'].rolling(w, min_periods=1).std().fillna(0).astype('float32')
        
        # 변화율
        df['CHANGE_1'] = df['TOTALCNT'].diff(1).fillna(0).astype('float32')
        df['CHANGE_10'] = df['TOTALCNT'].diff(10).fillna(0).astype('float32')
        
        # 타겟
        df['TARGET'] = df['TOTALCNT'].shift(-self.pred_len)
        df['LEVEL'] = pd.cut(df['TARGET'], 
                            bins=[-np.inf, 1400, 1700, np.inf], 
                            labels=[0, 1, 2])
        
        print(f"✅ 특성 생성 완료: {len(df.columns)}개 컬럼")
        return df
    
    def create_sequences(self, df, batch_size=10000):
        """대용량 시퀀스 생성 (배치 처리)"""
        print("\n🔄 시퀀스 생성 (배치 처리)...")
        df = df.dropna()
        
        # 데이터 체크
        if len(df) == 0:
            print("❌ 에러: 데이터가 비어있습니다!")
            return None, None, None
        
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
        y_cls = df['LEVEL'].values  # 타입 변환 제거
        
        print(f"  데이터 크기: {X_data.shape}")
        
        # 스케일링
        print("  스케일러 처리...")
        if not self.scaler_fitted or len(X_data) > 0:
            # 항상 fit_transform 사용 (안전)
            X_scaled = self.feature_scaler.fit_transform(X_data)
            y_reg_scaled = self.target_scaler.fit_transform(y_reg.reshape(-1, 1)).flatten()
            self.scaler_fitted = True
            self.save_scalers()
        else:
            print("❌ 에러: 스케일링할 데이터가 없습니다!")
            return None, None, None
        
        # 배치별 시퀀스 생성
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
                # NaN 체크 후 변환
                if pd.isna(y_cls[i+self.seq_len-1]):
                    batch_y_c.append(0)  # NaN은 0으로 처리
                else:
                    batch_y_c.append(int(y_cls[i+self.seq_len-1]))
            
            X_list.append(np.array(batch_X, dtype='float32'))
            y_r_list.append(np.array(batch_y_r, dtype='float32'))
            y_c_list.append(np.array(batch_y_c, dtype='int32'))
            
            if (end_idx % 50000) == 0:
                print(f"    처리: {end_idx:,}/{total_sequences:,}")
                gc.collect()
        
        # 합치기
        X = np.concatenate(X_list, axis=0)
        y_r = np.concatenate(y_r_list, axis=0)
        y_c = np.concatenate(y_c_list, axis=0)
        
        print(f"✅ 시퀀스 생성 완료: {X.shape}")
        
        # 메모리 정리
        del X_list, y_r_list, y_c_list
        gc.collect()
        
        return X, y_r, y_c

# ===== 5개 모델 정의 (동일) =====

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
    """2. PatchTST (간소화)"""
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

# 학습 및 평가 (체크포인트 포함)
def train_and_evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test, processor):
    
    # 커스텀 손실 함수 (클래스 불균형)
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
    print(f"🎯 {model.name} 학습 (78만개 데이터)")
    print(f"{'='*60}")
    
    # 체크포인트 콜백
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            f'{checkpoint_dir}/{model.name}_best.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            patience=5, 
            restore_best_weights=True, 
            verbose=1
        ),
        ReduceLROnPlateau(
            factor=0.5, 
            patience=3, 
            min_lr=1e-6,
            verbose=1
        ),
        TensorBoard(
            log_dir=f'logs/{model.name}',
            histogram_freq=0
        )
    ]
    
    # 학습 (배치 크기 증가)
    history = model.fit(
        X_train,
        {'regression': y_train[0], 'classification': y_train[1]},
        validation_data=(X_val, {'regression': y_val[0], 'classification': y_val[1]}),
        epochs=30,  # 더 많은 에폭
        batch_size=512,  # 더 큰 배치
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"\n📊 {model.name} 평가 결과:")
    
    # 예측 (배치 처리)
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
    print(f"  정상→정상: {cm[0,0]:,}개")
    print(f"  주의→주의: {cm[1,1]:,}개" if len(cm) > 1 else "")
    print(f"  심각→심각: {cm[2,2]:,}개" if len(cm) > 2 else "")
    
    # 1700+ 감지율
    if 2 in y_test[1]:
        critical_mask = y_test[1] == 2
        critical_recall = (y_cls_pred[critical_mask] == 2).mean()
        print(f"\n[1700+ 감지율]: {critical_recall:.2%}")
    
    # 모델 저장
    os.makedirs('models', exist_ok=True)
    model.save(f'models/{model.name}_final.keras')
    print(f"\n💾 최종 모델 저장: models/{model.name}_final.keras")
    
    # 결과 저장
    results = {
        'MAE': mae, 
        'RMSE': rmse, 
        'R2': r2, 
        'Accuracy': acc,
        'Train_Size': len(X_train),
        'Val_Size': len(X_val),
        'Test_Size': len(X_test)
    }
    
    # 학습 이력 저장
    with open(f'history/{model.name}_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    
    return results

# 메인 실행
def main():
    # 데이터 경로
    data_paths = [
        '/mnt/user-data/uploads/gs.CSV',  # 78만개 데이터
        'data/full_data.csv',
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
    os.makedirs('logs', exist_ok=True)
    os.makedirs('history', exist_ok=True)
    
    # 데이터 처리
    processor = DataProcessor()
    
    # 스케일러는 새로 학습 (78만개 데이터에 맞게)
    # processor.load_scalers()  # 제거
    
    # 데이터 로드
    df = processor.load_and_process(data_path)
    df = processor.create_features(df)
    
    # 시퀀스 생성 (배치 처리)
    X, y_reg, y_cls = processor.create_sequences(df, batch_size=10000)
    
    # 데이터 체크
    if X is None:
        print("❌ 시퀀스 생성 실패!")
        return
    
    # 메모리 정리
    del df
    gc.collect()
    
    # 데이터 분할 (78만개 기준)
    n_train = int(0.7 * len(X))  # 약 54만개
    n_val = int(0.15 * len(X))   # 약 11만개
    # n_test = 나머지             # 약 11만개
    
    print(f"\n📊 데이터 분할 (78만개 기준):")
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
    
    print(f"\n📊 실제 shape:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  X_test:  {X_test.shape}")
    
    # 5개 모델 생성
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
    
    # 학습 및 평가
    results = {}
    for name, model in models.items():
        try:
            results[name] = train_and_evaluate(
                model, X_train, y_train, X_val, y_val, X_test, y_test, processor
            )
            
            # 메모리 정리
            tf.keras.backend.clear_session()
            gc.collect()
            
        except Exception as e:
            print(f"❌ {name} 학습 실패: {e}")
            continue
    
    # 최종 결과
    print("\n" + "="*80)
    print("📊 최종 성능 비교 (78만개 데이터)")
    print("="*80)
    
    results_df = pd.DataFrame(results).T
    print("\n", results_df.round(3))
    
    # 최고 성능
    print("\n🏆 최고 성능:")
    print(f"  MAE 최저: {results_df['MAE'].idxmin()} ({results_df['MAE'].min():.2f})")
    print(f"  RMSE 최저: {results_df['RMSE'].idxmin()} ({results_df['RMSE'].min():.2f})")
    print(f"  R² 최고: {results_df['R2'].idxmax()} ({results_df['R2'].max():.4f})")
    print(f"  정확도 최고: {results_df['Accuracy'].idxmax()} ({results_df['Accuracy'].max():.2%})")
    
    # CSV 저장
    results_df.to_csv('model_results_780k.csv')
    print(f"\n💾 결과 저장: model_results_780k.csv")
    
    print("\n✅ 78만개 데이터 학습 완료!")
    
    return results

if __name__ == "__main__":
    results = main()