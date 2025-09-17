"""
🚀 반도체 물류 예측 - 6개 모델 최종 개선판
=========================================
PatchTST, EnsembleBase 문제 해결
스케일러 저장 및 실시간 예측 지원
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import *
from tensorflow.keras.losses import Huber
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
import warnings
import os
import gc
import pickle
import json

warnings.filterwarnings('ignore')
tf.random.set_seed(42)
np.random.seed(42)

# GPU 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
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
print("🚀 반도체 물류 예측 시스템 v2.0")
print(f"📦 TensorFlow: {tf.__version__}")
print(f"🔧 Device: {device}")
print("="*80)

# ========================================
# 데이터 처리 클래스
# ========================================
class DataProcessor:
    def __init__(self, seq_len=100, pred_len=10):
        self.seq_len = seq_len
        self.pred_len = pred_len
        # 여러 스케일러 테스트
        self.feature_scaler = StandardScaler()  # 특징용
        self.target_scaler = RobustScaler()     # 타겟용 (이상치에 강건)
        self.feature_columns = None
        
    def save_scalers(self, path='scalers/'):
        """스케일러 저장"""
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
            'feature_columns': self.feature_columns
        }
        with open(f'{path}config.json', 'w') as f:
            json.dump(config, f)
            
        print(f"✅ 스케일러 저장: {path}")
    
    def load_scalers(self, path='scalers/'):
        """스케일러 로드"""
        with open(f'{path}feature_scaler.pkl', 'rb') as f:
            self.feature_scaler = pickle.load(f)
        with open(f'{path}target_scaler.pkl', 'rb') as f:
            self.target_scaler = pickle.load(f)
        with open(f'{path}config.json', 'r') as f:
            config = json.load(f)
            self.seq_len = config['seq_len']
            self.pred_len = config['pred_len']
            self.feature_columns = config['feature_columns']
        print(f"✅ 스케일러 로드: {path}")
        
    def load_and_process(self, filepath):
        print(f"\n📂 데이터 로딩: {filepath}")
        df = pd.read_csv(filepath)
        print(f"✅ 원본: {df.shape[0]:,}행 × {df.shape[1]}열")
        
        # 0값 제거
        before = len(df)
        df = df[df['TOTALCNT'] > 0].reset_index(drop=True)
        print(f"✅ 0값 제거: {before - len(df):,}개")
        
        # 시간 정렬
        df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), 
                                       format='%Y%m%d%H%M', errors='coerce')
        df = df.sort_values('CURRTIME').reset_index(drop=True)
        
        # 통계
        print(f"\n📊 TOTALCNT 통계:")
        print(f"  평균: {df['TOTALCNT'].mean():,.0f}")
        print(f"  표준편차: {df['TOTALCNT'].std():,.0f}")
        print(f"  최소: {df['TOTALCNT'].min():,}")
        print(f"  최대: {df['TOTALCNT'].max():,}")
        
        # 구간 분포
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
        
        # 기본 특성
        df['RATIO'] = df['M14AM14B'] / (df['M14AM10A'] + 1)
        df['GOLDEN'] = ((df['M14AM14B'] > 300) & (df['M14AM10A'] < 80)).astype(float)
        
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
        df['LEVEL'] = 0
        df.loc[df['TARGET'] >= 1400, 'LEVEL'] = 1
        df.loc[df['TARGET'] >= 1700, 'LEVEL'] = 2
        
        df = df.dropna()
        print(f"✅ 유효 데이터: {len(df)}행")
        
        return df
    
    def create_sequences(self, df):
        print("\n🔄 시퀀스 생성...")
        
        # 특성 선택
        self.feature_columns = [
            'TOTALCNT', 'M14AM14B', 'M14AM14BSUM', 'M14AM10A', 'M14AM16',
            'RATIO', 'GOLDEN',
            'HOUR_SIN', 'HOUR_COS',
            'MA_10', 'MA_30', 'STD_10', 'STD_30',
            'CHANGE_1', 'CHANGE_10'
        ]
        
        # 사용 가능한 컬럼만
        self.feature_columns = [f for f in self.feature_columns if f in df.columns]
        print(f"  특성: {len(self.feature_columns)}개")
        
        X_data = df[self.feature_columns].values.astype(np.float32)
        y_reg = df['TARGET'].values.astype(np.float32)
        y_cls = df['LEVEL'].values.astype(np.int32)
        
        # 분할
        total_samples = len(X_data) - self.seq_len
        train_end = int(total_samples * 0.7)
        val_end = int(total_samples * 0.85)
        
        # 스케일링 (훈련 데이터로만)
        print("  스케일링...")
        self.feature_scaler.fit(X_data[:train_end + self.seq_len])
        self.target_scaler.fit(y_reg[:train_end].reshape(-1, 1))
        
        # 변환
        X_scaled = self.feature_scaler.transform(X_data)
        y_reg_scaled = self.target_scaler.transform(y_reg.reshape(-1, 1)).flatten()
        
        # 시퀀스 생성
        X, y_r, y_c = [], [], []
        for i in range(total_samples):
            X.append(X_scaled[i:i+self.seq_len])
            y_r.append(y_reg_scaled[i+self.seq_len-1])
            y_c.append(y_cls[i+self.seq_len-1])
        
        X = np.array(X, dtype=np.float32)
        y_r = np.array(y_r, dtype=np.float32)
        y_c = np.array(y_c, dtype=np.int32)
        
        # 시계열 순서 분할
        X_train = X[:train_end]
        X_val = X[train_end:val_end]
        X_test = X[val_end:]
        
        y_train = (y_r[:train_end], y_c[:train_end])
        y_val = (y_r[train_end:val_end], y_c[train_end:val_end])
        y_test = (y_r[val_end:], y_c[val_end:])
        
        print(f"  Train: {X_train.shape}")
        print(f"  Val: {X_val.shape}")
        print(f"  Test: {X_test.shape}")
        
        # 스케일러 저장
        self.save_scalers()
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# ========================================
# 6개 모델 (개선됨)
# ========================================

def build_stable_lstm(input_shape):
    """1. StableLSTM"""
    inputs = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True)(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = LSTM(32)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    out_reg = Dense(1, name='regression')(x)
    out_cls = Dense(3, activation='softmax', name='classification')(x)
    return Model(inputs, [out_reg, out_cls], name='StableLSTM')

def build_patch_tst(input_shape):
    """2. PatchTST - 완전 재설계"""
    inputs = Input(shape=input_shape)
    
    # GRU로 시작 (시계열에 강함)
    x = GRU(64, return_sequences=True)(inputs)
    x = BatchNormalization()(x)
    
    # 가벼운 Attention
    attn = MultiHeadAttention(num_heads=2, key_dim=16)(x, x)
    x = Add()([x, attn])
    x = LayerNormalization()(x)
    
    # 최종 GRU
    x = GRU(32)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    out_reg = Dense(1, name='regression')(x)
    out_cls = Dense(3, activation='softmax', name='classification')(x)
    
    return Model(inputs, [out_reg, out_cls], name='PatchTST')

def build_spike_detector(input_shape):
    """3. SpikeDetector"""
    inputs = Input(shape=input_shape)
    recent = Lambda(lambda x: x[:, -20:, :])(inputs)
    x = Conv1D(32, 3, activation='relu', padding='same')(recent)
    x = MaxPooling1D(2)(x)
    x = GRU(32)(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    out_reg = Dense(1, name='regression')(x)
    out_cls = Dense(3, activation='softmax', name='classification')(x)
    return Model(inputs, [out_reg, out_cls], name='SpikeDetector')

def build_extreme_net(input_shape):
    """4. ExtremeNet"""
    inputs = Input(shape=input_shape)
    lstm = LSTM(48, return_sequences=True)(inputs)
    lstm = LSTM(24)(lstm)
    attn = MultiHeadAttention(num_heads=4, key_dim=12)(inputs, inputs)
    attn = GlobalAveragePooling1D()(attn)
    x = Concatenate()([lstm, attn])
    x = BatchNormalization()(x)
    x = Dense(48, activation='relu')(x)
    x = Dropout(0.2)(x)
    out_reg = Dense(1, name='regression')(x)
    out_cls = Dense(3, activation='softmax', name='classification')(x)
    return Model(inputs, [out_reg, out_cls], name='ExtremeNet')

def build_ensemble_base(input_shape):
    """5. EnsembleBase - 재설계"""
    inputs = Input(shape=input_shape)
    
    # Simple LSTM branch
    lstm_branch = LSTM(32)(inputs)
    
    # Simple Conv branch  
    conv_branch = Conv1D(32, 5, activation='relu')(inputs)
    conv_branch = GlobalAveragePooling1D()(conv_branch)
    
    # Statistics
    mean_pool = Lambda(lambda x: tf.reduce_mean(x, axis=1))(inputs)
    
    # Concatenate
    x = Concatenate()([lstm_branch, conv_branch, mean_pool])
    x = BatchNormalization()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    out_reg = Dense(1, name='regression')(x)
    out_cls = Dense(3, activation='softmax', name='classification')(x)
    
    return Model(inputs, [out_reg, out_cls], name='EnsembleBase')

def build_golden_rule(input_shape):
    """6. GoldenRule"""
    inputs = Input(shape=input_shape)
    last = Lambda(lambda x: x[:, -10:, :])(inputs)
    x = Conv1D(32, 3, activation='relu')(last)
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    out_reg = Dense(1, name='regression')(x)
    out_cls = Dense(3, activation='softmax', name='classification')(x)
    return Model(inputs, [out_reg, out_cls], name='GoldenRule')

# ========================================
# 학습 함수
# ========================================
def train_and_evaluate(model, train_data, val_data, test_data, processor):
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data
    
    with strategy.scope():
        # 모델 재생성
        input_shape = model.input_shape[1:]
        
        if model.name == 'StableLSTM':
            model = build_stable_lstm(input_shape)
        elif model.name == 'PatchTST':
            model = build_patch_tst(input_shape)
        elif model.name == 'SpikeDetector':
            model = build_spike_detector(input_shape)
        elif model.name == 'ExtremeNet':
            model = build_extreme_net(input_shape)
        elif model.name == 'EnsembleBase':
            model = build_ensemble_base(input_shape)
        elif model.name == 'GoldenRule':
            model = build_golden_rule(input_shape)
        
        model.compile(
            optimizer=Adam(0.001),
            loss=['mae', 'sparse_categorical_crossentropy'],
            loss_weights=[0.7, 0.3],
            metrics={'regression': 'mae', 'classification': 'accuracy'}
        )
    
    print(f"\n{'='*60}")
    print(f"🎯 {model.name} 학습")
    print(f"{'='*60}")
    
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(factor=0.5, patience=5, verbose=0)
    ]
    
    batch_size = 64 * strategy.num_replicas_in_sync if gpus else 32
    
    history = model.fit(
        X_train,
        {'regression': y_train[0], 'classification': y_train[1]},
        validation_data=(X_val, {'regression': y_val[0], 'classification': y_val[1]}),
        epochs=30,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # 평가
    print(f"\n📊 {model.name} 평가:")
    
    preds = model.predict(X_test, verbose=0)
    y_reg_pred = preds[0].flatten()
    y_cls_pred = np.argmax(preds[1], axis=1)
    
    # 역변환
    y_reg_true_orig = processor.target_scaler.inverse_transform(
        y_test[0].reshape(-1, 1)).flatten()
    y_reg_pred_orig = processor.target_scaler.inverse_transform(
        y_reg_pred.reshape(-1, 1)).flatten()
    
    mae = mean_absolute_error(y_reg_true_orig, y_reg_pred_orig)
    rmse = np.sqrt(mean_squared_error(y_reg_true_orig, y_reg_pred_orig))
    r2 = r2_score(y_reg_true_orig, y_reg_pred_orig)
    acc = accuracy_score(y_test[1], y_cls_pred)
    
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²: {r2:.4f}")
    print(f"  정확도: {acc:.2%}")
    
    # 모델 저장
    os.makedirs('models', exist_ok=True)
    model.save(f'models/{model.name}.keras')
    print(f"💾 모델: models/{model.name}.keras")
    
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'Accuracy': acc}

# ========================================
# 실시간 예측 클래스
# ========================================
class RealtimePredictor:
    def __init__(self, model_path, scaler_path='scalers/'):
        """실시간 예측기 초기화"""
        self.model = tf.keras.models.load_model(model_path)
        self.processor = DataProcessor()
        self.processor.load_scalers(scaler_path)
        print(f"✅ 실시간 예측기 준비 완료")
        
    def predict(self, data_100min):
        """100분 데이터로 10분 후 예측"""
        # 데이터 검증
        if len(data_100min) != 100:
            raise ValueError(f"100분 데이터 필요 (현재: {len(data_100min)})")
        
        # 특성 추출
        X = data_100min[self.processor.feature_columns].values
        
        # 스케일링
        X_scaled = self.processor.feature_scaler.transform(X)
        X_seq = X_scaled.reshape(1, 100, -1)
        
        # 예측
        preds = self.model.predict(X_seq, verbose=0)
        y_reg_scaled = preds[0][0, 0]
        y_cls = np.argmax(preds[1][0])
        
        # 역변환
        y_reg = self.processor.target_scaler.inverse_transform([[y_reg_scaled]])[0, 0]
        
        # 레벨
        level = ['정상', '주의', '심각'][y_cls]
        
        return {
            'prediction': float(y_reg),
            'level': level,
            'confidence': float(np.max(preds[1][0]))
        }

# ========================================
# 메인 실행
# ========================================
def main():
    # 데이터 찾기
    data_paths = [
        'data/20240201_TO_202507281705.csv',
        'data/20250731_to20250806.csv',
        '/mnt/user-data/uploads/gs.CSV',
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
    
    # 데이터 처리
    processor = DataProcessor()
    df = processor.load_and_process(data_path)
    df = processor.create_features(df)
    train_data, val_data, test_data = processor.create_sequences(df)
    
    # 6개 모델
    input_shape = (train_data[0].shape[1], train_data[0].shape[2])
    
    models = {
        'StableLSTM': build_stable_lstm(input_shape),
        'PatchTST': build_patch_tst(input_shape),
        'SpikeDetector': build_spike_detector(input_shape),
        'ExtremeNet': build_extreme_net(input_shape),
        'EnsembleBase': build_ensemble_base(input_shape),
        'GoldenRule': build_golden_rule(input_shape)
    }
    
    print(f"\n✅ 6개 모델 준비")
    
    # 학습
    results = {}
    for name, model in models.items():
        try:
            results[name] = train_and_evaluate(
                model, train_data, val_data, test_data, processor
            )
        except Exception as e:
            print(f"❌ {name} 실패: {e}")
    
    # 결과
    print("\n" + "="*80)
    print("📊 최종 성능")
    print("="*80)
    
    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values('R2', ascending=False)
    print(results_df.round(3))
    
    print("\n🏆 최고 성능:")
    print(f"  MAE: {results_df['MAE'].idxmin()} ({results_df['MAE'].min():.2f})")
    print(f"  R²: {results_df['R2'].idxmax()} ({results_df['R2'].max():.4f})")
    
    results_df.to_csv('final_results.csv')
    print("\n💾 결과: final_results.csv")
    
    # 실시간 예측 테스트
    print("\n" + "="*80)
    print("🔮 실시간 예측 테스트")
    print("="*80)
    
    best_model = results_df['R2'].idxmax()
    predictor = RealtimePredictor(f'models/{best_model}.keras')
    
    # 테스트 데이터로 예측
    test_sample = df.iloc[-100:].copy()
    result = predictor.predict(test_sample)
    
    print(f"\n예측 결과:")
    print(f"  10분 후 예측: {result['prediction']:.0f}")
    print(f"  레벨: {result['level']}")
    print(f"  신뢰도: {result['confidence']:.2%}")
    
    print("\n✅ 완료!")
    
    return results

if __name__ == "__main__":
    results = main()