"""
🚀 반도체 물류 예측 - 6개 모델
=====================================
1. StableLSTM
2. PatchTST  
3. SpikeDetector
4. ExtremeNet
5. EnsembleBase
6. GoldenRule (황금 규칙)

100분 데이터 → 10분 후 예측
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
import warnings
import os

warnings.filterwarnings('ignore')
tf.random.set_seed(42)
np.random.seed(42)

print("="*80)
print("🚀 반도체 물류 예측 - 6개 모델")
print(f"📦 TensorFlow: {tf.__version__}")
print("="*80)

# 데이터 처리
class DataProcessor:
    def __init__(self, seq_len=100, pred_len=10):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
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
        df['GOLDEN_1'] = ((df['M14AM14B'] > 300) & (df['M14AM10A'] < 80)).astype(float)
        df['GOLDEN_2'] = (df['M14AM14BSUM'] > 500).astype(float)
        
        # 시간
        df['HOUR'] = df['CURRTIME'].dt.hour.fillna(0)
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
        y_cls = df['LEVEL'].values.astype(int)
        
        # 시계열 순서대로 분할 (중요!)
        total_samples = len(X_data) - self.seq_len
        train_end = int(total_samples * 0.7)
        val_end = int(total_samples * 0.85)
        
        # 훈련 데이터로만 fit
        self.feature_scaler.fit(X_data[:train_end + self.seq_len])
        self.target_scaler.fit(y_reg[:train_end].reshape(-1, 1))
        
        # 전체 변환
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
        print(f"  Val:   {X_val.shape}")
        print(f"  Test:  {X_test.shape}")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# ===== 원래 5개 모델 =====

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

def build_golden_rule(input_shape):
    """6. GoldenRule - 황금 규칙 기반"""
    inputs = Input(shape=input_shape)
    # 마지막 시점의 특성만 사용 (규칙 기반)
    last_step = Lambda(lambda x: x[:, -1, :])(inputs)
    x = Dense(64, activation='relu')(last_step)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    out_reg = Dense(1, name='regression')(x)
    out_cls = Dense(3, activation='softmax', name='classification')(x)
    return Model(inputs, [out_reg, out_cls], name='GoldenRule')

# ===== 황금 규칙 기반 예측 =====
def apply_golden_rules(df, processor):
    """황금 규칙 적용"""
    print("\n🏆 황금 규칙 평가...")
    
    # 규칙 1: M14AM14B > 300 & M14AM10A < 80
    rule1 = (df['M14AM14B'] > 300) & (df['M14AM10A'] < 80)
    
    # 규칙 2: M14AM14BSUM > 500
    rule2 = df['M14AM14BSUM'] > 500
    
    # 규칙 3: M14AM14B/M14AM10A > 5
    rule3 = (df['M14AM14B'] / (df['M14AM10A'] + 1)) > 5
    
    # M14AM14B 구간별 예측
    def predict_by_m14b(row):
        m14b = row['M14AM14B']
        if m14b >= 450:
            return 1849
        elif m14b >= 400:
            return 1671
        elif m14b >= 350:
            return 1561
        elif m14b >= 300:
            return 1518
        elif m14b >= 250:
            return 1438
        elif m14b >= 200:
            return 1356
        else:
            return 1200
    
    # 규칙 기반 예측
    df['RULE_PRED'] = df.apply(predict_by_m14b, axis=1)
    
    # 황금 패턴 정확도
    golden_mask = rule1 | rule2 | rule3
    golden_accuracy = (df[golden_mask]['RULE_PRED'] - df[golden_mask]['TARGET']).abs().mean()
    
    print(f"  황금 패턴 발생: {golden_mask.sum()}회 ({golden_mask.sum()/len(df)*100:.1f}%)")
    print(f"  황금 패턴 MAE: {golden_accuracy:.2f}")
    
    # 구간별 정확도
    print(f"\n  M14AM14B 구간별 정확도:")
    for lower, upper, pred_val in [(450, 999999, 1849), (400, 450, 1671), 
                                   (350, 400, 1561), (300, 350, 1518),
                                   (250, 300, 1438), (200, 250, 1356)]:
        mask = (df['M14AM14B'] >= lower) & (df['M14AM14B'] < upper)
        if mask.sum() > 0:
            mae = (df[mask]['TARGET'] - pred_val).abs().mean()
            print(f"    {lower}-{upper}: {mask.sum()}회, MAE={mae:.2f}")
    
    return df['RULE_PRED'].values

# 학습 및 평가
def train_and_evaluate(model, train_data, val_data, test_data, processor):
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data
    
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
        EarlyStopping(patience=10, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(factor=0.5, patience=5, verbose=0)
    ]
    
    history = model.fit(
        X_train,
        {'regression': y_train[0], 'classification': y_train[1]},
        validation_data=(X_val, {'regression': y_val[0], 'classification': y_val[1]}),
        epochs=30,
        batch_size=64,
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
    
    # 황금 규칙 평가
    rule_predictions = apply_golden_rules(df, processor)
    
    train_data, val_data, test_data = processor.create_sequences(df)
    
    # 6개 모델 생성
    input_shape = (train_data[0].shape[1], train_data[0].shape[2])
    
    models = {
        'StableLSTM': build_stable_lstm(input_shape),
        'PatchTST': build_patch_tst(input_shape),
        'SpikeDetector': build_spike_detector(input_shape),
        'ExtremeNet': build_extreme_net(input_shape),
        'EnsembleBase': build_ensemble_base(input_shape),
        'GoldenRule': build_golden_rule(input_shape)
    }
    
    print(f"\n✅ 6개 모델 준비 완료:")
    for name in models.keys():
        print(f"  - {name}")
    
    results = {}
    for name, model in models.items():
        results[name] = train_and_evaluate(
            model, train_data, val_data, test_data, processor
        )
    
    # 최종 결과
    print("\n" + "="*80)
    print("📊 최종 성능 비교 (6개 모델)")
    print("="*80)
    
    results_df = pd.DataFrame(results).T
    print("\n[6개 모델 성능]")
    print(results_df.round(3))
    
    print("\n[황금 규칙 설명]")
    print("  GoldenRule 모델: M14AM14B 구간별 예측")
    print("    - M14AM14B > 300 & M14AM10A < 80 → 급증")
    print("    - M14AM14BSUM > 500 → 누적 경고")
    print("    - M14AM14B/M14AM10A > 5 → 비율 이상")
    
    # 최고 성능
    print("\n🏆 최고 성능:")
    print(f"  MAE 최저: {results_df['MAE'].idxmin()} ({results_df['MAE'].min():.2f})")
    print(f"  R² 최고: {results_df['R2'].idxmax()} ({results_df['R2'].max():.4f})")
    print(f"  정확도 최고: {results_df['Accuracy'].idxmax()} ({results_df['Accuracy'].max():.2%})")
    
    results_df.to_csv('model_results_6models.csv')
    print("\n💾 결과 저장: model_results_6models.csv")
    print("\n✅ 6개 모델 학습 완료!")
    
    return results

if __name__ == "__main__":
    results = main()