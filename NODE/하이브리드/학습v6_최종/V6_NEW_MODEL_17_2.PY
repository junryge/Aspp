"""
TensorFlow 2.16.1 - Lambda 없는 깔끔한 버전
==========================================
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import *
from sklearn.preprocessing import MinMaxScaler
import os

print(f"TensorFlow: {tf.__version__}")
tf.random.set_seed(42)
np.random.seed(42)

# 데이터 처리
class DataProcessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        
    def process(self, filepath):
        df = pd.read_csv(filepath)
        df['CURRTIME'] = pd.to_datetime(df['CURRTIME'], format='%Y%m%d%H%M')
        df = df.sort_values('CURRTIME').reset_index(drop=True)
        
        # 타겟
        df['TARGET'] = df['TOTALCNT'].shift(-10)
        df['IS_1700'] = (df['TARGET'] >= 1700).astype(int)
        
        # 특성
        df['MA_10'] = df['TOTALCNT'].rolling(10, min_periods=1).mean()
        df['MA_30'] = df['TOTALCNT'].rolling(30, min_periods=1).mean()
        df['CHANGE'] = df['TOTALCNT'].diff().fillna(0)
        
        # 비율
        if 'M14AM14B' in df.columns and 'M14AM10A' in df.columns:
            df['RATIO'] = df['M14AM14B'] / (df['M14AM10A'] + 1)
        
        df = df.dropna()
        
        # 시퀀스
        features = [c for c in df.columns if c not in ['CURRTIME', 'TIME', 'TARGET', 'IS_1700']]
        features = [c for c in features if df[c].dtype in [np.float64, np.int64]]
        
        data = self.scaler.fit_transform(df[features])
        targets = self.target_scaler.fit_transform(df[['TARGET']]).flatten()
        is_1700 = df['IS_1700'].values
        
        X, y, y_bin = [], [], []
        for i in range(len(data) - 110):
            X.append(data[i:i+100])
            y.append(targets[i+100])
            y_bin.append(is_1700[i+100])
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        y_bin = np.array(y_bin, dtype=np.float32)
        
        print(f"데이터 shape: {X.shape}")
        print(f"1700+ 비율: {y_bin.mean():.2%}")
        
        return X, y, y_bin

# 15개 모델 (Lambda 없음)
class Models:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.n_features = input_shape[1]
    
    # 1. PatchTST
    def m1_patch_tst(self):
        inputs = Input(shape=self.input_shape)
        x = BatchNormalization()(inputs)
        
        # 패치로 나누기 - Reshape 사용
        x = Reshape((10, 10 * self.n_features))(x)
        x = Dense(128)(x)
        x = LayerNormalization()(x)
        
        x = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
        x = GlobalAveragePooling1D()(x)
        
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs, outputs, name='PatchTST')
    
    # 2. ExtremeValueNet
    def m2_extreme_value(self):
        inputs = Input(shape=self.input_shape)
        x = BatchNormalization()(inputs)
        
        x = LSTM(128, return_sequences=True)(x)
        x = LSTM(64)(x)
        
        # 두 브랜치
        normal = Dense(32, activation='relu')(x)
        normal = Dense(1)(normal)
        
        extreme = Dense(32, activation='relu')(x)
        extreme = Dense(1, activation='sigmoid')(extreme)
        
        # 결합
        outputs = Add()([normal, Multiply()([normal, extreme])])
        outputs = Activation('sigmoid')(outputs)
        
        return Model(inputs, outputs, name='ExtremeValue')
    
    # 3. SpikeDetector
    def m3_spike_detector(self):
        inputs = Input(shape=self.input_shape)
        x = BatchNormalization()(inputs)
        
        x = Conv1D(64, 3, padding='same', activation='relu')(x)
        x = MaxPooling1D(2)(x)
        x = Bidirectional(LSTM(32))(x)
        
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs, outputs, name='SpikeDetector')
    
    # 4. MambaSSM
    def m4_mamba(self):
        inputs = Input(shape=self.input_shape)
        x = BatchNormalization()(inputs)
        
        x = GRU(128, return_sequences=True)(x)
        x = GRU(64, return_sequences=True)(x)
        x = GRU(32)(x)
        
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs, outputs, name='Mamba')
    
    # 5. MTGNN
    def m5_mtgnn(self):
        inputs = Input(shape=self.input_shape)
        x = BatchNormalization()(inputs)
        
        x = Conv1D(64, 3, padding='causal', activation='relu')(x)
        x = Conv1D(64, 3, padding='causal', activation='relu')(x)
        
        x = GlobalAveragePooling1D()(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs, outputs, name='MTGNN')
    
    # 6. NeuralSDE
    def m6_neural_sde(self):
        inputs = Input(shape=self.input_shape)
        x = BatchNormalization()(inputs)
        
        # Drift와 Diffusion
        drift = LSTM(64, return_sequences=True)(x)
        drift = LSTM(32)(drift)
        
        diffusion = GRU(64, return_sequences=True)(x)
        diffusion = GRU(32)(diffusion)
        
        # 결합
        x = Concatenate()([drift, diffusion])
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs, outputs, name='NeuralSDE')
    
    # 7. MixtureOfExperts
    def m7_mixture_experts(self):
        inputs = Input(shape=self.input_shape)
        x = BatchNormalization()(inputs)
        
        shared = LSTM(64, return_sequences=True)(x)
        shared = LSTM(32)(shared)
        
        # 3개 expert
        e1 = Dense(16, activation='relu')(shared)
        e1 = Dense(1, activation='sigmoid')(e1)
        
        e2 = Dense(16, activation='relu')(shared)  
        e2 = Dense(1, activation='sigmoid')(e2)
        
        e3 = Dense(16, activation='relu')(shared)
        e3 = Dense(1, activation='sigmoid')(e3)
        
        # Gate
        gate = Dense(3, activation='softmax')(shared)
        
        # 가중 평균 (간단하게)
        outputs = Average()([e1, e2, e3])  # 단순 평균
        
        return Model(inputs, outputs, name='MixtureExperts')
    
    # 8. DiffusionModel
    def m8_diffusion(self):
        inputs = Input(shape=self.input_shape)
        x = BatchNormalization()(inputs)
        
        x = Conv1D(64, 3, padding='same', activation='relu')(x)
        x = Conv1D(128, 3, padding='same', activation='relu')(x)
        x = MaxPooling1D(2)(x)
        
        x = LSTM(64, return_sequences=True)(x)
        x = LSTM(32)(x)
        
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs, outputs, name='Diffusion')
    
    # 9. Autoformer
    def m9_autoformer(self):
        inputs = Input(shape=self.input_shape)
        x = BatchNormalization()(inputs)
        
        # Trend & Seasonal
        trend = AveragePooling1D(10, strides=1, padding='same')(x)
        seasonal = Subtract()([x, trend])
        
        trend = Conv1D(32, 1, activation='relu')(trend)
        seasonal = Conv1D(32, 1, activation='relu')(seasonal)
        
        x = Concatenate()([trend, seasonal])
        x = GlobalAveragePooling1D()(x)
        
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs, outputs, name='Autoformer')
    
    # 10. TCN
    def m10_tcn(self):
        inputs = Input(shape=self.input_shape)
        x = BatchNormalization()(inputs)
        
        # Dilated Conv
        for dilation in [1, 2, 4, 8]:
            conv = Conv1D(64, 3, dilation_rate=dilation, padding='causal', activation='relu')(x)
            conv = Dropout(0.1)(conv)
            if x.shape[-1] != 64:
                x = Conv1D(64, 1)(x)
            x = Add()([x, conv])
        
        x = GlobalAveragePooling1D()(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs, outputs, name='TCN')
    
    # 11. WaveNet
    def m11_wavenet(self):
        inputs = Input(shape=self.input_shape)
        x = BatchNormalization()(inputs)
        
        for i in range(3):
            dilation = 2 ** i
            tanh = Conv1D(32, 2, dilation_rate=dilation, padding='causal', activation='tanh')(x)
            sigm = Conv1D(32, 2, dilation_rate=dilation, padding='causal', activation='sigmoid')(x)
            gate = Multiply()([tanh, sigm])
            
            if gate.shape[-1] != x.shape[-1]:
                x = Conv1D(x.shape[-1], 1)(gate)
            else:
                x = gate
        
        x = GlobalAveragePooling1D()(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs, outputs, name='WaveNet')
    
    # 12. BiLSTM_Attention  
    def m12_bilstm_attention(self):
        inputs = Input(shape=self.input_shape)
        x = BatchNormalization()(inputs)
        
        x = Bidirectional(LSTM(64, return_sequences=True))(x)
        
        # Simple Attention
        attention = Dense(1, activation='tanh')(x)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(128)(attention)
        attention = Permute([2, 1])(attention)
        
        x = Multiply()([x, attention])
        x = GlobalAveragePooling1D()(x)
        
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs, outputs, name='BiLSTM_Attention')
    
    # 13. Informer
    def m13_informer(self):
        inputs = Input(shape=self.input_shape)
        x = BatchNormalization()(inputs)
        
        x = Dense(64)(x)
        x = MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
        x = LayerNormalization()(x)
        
        x = GlobalAveragePooling1D()(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs, outputs, name='Informer')
    
    # 14. TimesNet (간단 버전)
    def m14_timesnet(self):
        inputs = Input(shape=self.input_shape)
        x = BatchNormalization()(inputs)
        
        # 2D 변환 없이 1D CNN만
        x = Conv1D(64, 3, padding='same', activation='relu')(x)
        x = Conv1D(32, 3, padding='same', activation='relu')(x)
        
        x = GlobalAveragePooling1D()(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs, outputs, name='TimesNet')
    
    # 15. BaselineLSTM
    def m15_baseline_lstm(self):
        inputs = Input(shape=self.input_shape)
        x = BatchNormalization()(inputs)
        
        x = LSTM(128, return_sequences=True)(x)
        x = Dropout(0.2)(x)
        x = LSTM(64)(x)
        x = Dropout(0.2)(x)
        
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs, outputs, name='BaselineLSTM')

# 메인
def main():
    print("\n🚀 시작...")
    
    # 데이터
    processor = DataProcessor()
    
    # 파일 찾기
    for path in ['data.csv', 'gs.CSV', '/mnt/user-data/uploads/gs.CSV']:
        if os.path.exists(path):
            data_path = path
            break
    
    X, y, y_bin = processor.process(data_path)
    
    # 분할
    n_train = int(0.7 * len(X))
    n_val = int(0.15 * len(X))
    
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]
    y_test_bin = y_bin[n_train+n_val:]
    
    print(f"\nTrain: {X_train.shape}")
    print(f"Val: {X_val.shape}")
    print(f"Test: {X_test.shape}")
    
    # 모델 생성
    builder = Models(input_shape=(100, X.shape[2]))
    
    models = {
        '01_PatchTST': builder.m1_patch_tst(),
        '02_ExtremeValue': builder.m2_extreme_value(),
        '03_SpikeDetector': builder.m3_spike_detector(),
        '04_Mamba': builder.m4_mamba(),
        '05_MTGNN': builder.m5_mtgnn(),
        '06_NeuralSDE': builder.m6_neural_sde(),
        '07_MixtureExperts': builder.m7_mixture_experts(),
        '08_Diffusion': builder.m8_diffusion(),
        '09_Autoformer': builder.m9_autoformer(),
        '10_TCN': builder.m10_tcn(),
        '11_WaveNet': builder.m11_wavenet(),
        '12_BiLSTM_Attention': builder.m12_bilstm_attention(),
        '13_Informer': builder.m13_informer(),
        '14_TimesNet': builder.m14_timesnet(),
        '15_BaselineLSTM': builder.m15_baseline_lstm()
    }
    
    print(f"\n✅ {len(models)}개 모델 생성 완료!")
    
    results = {}
    
    # 학습
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"🎯 {name}")
        
        try:
            model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=10,
                batch_size=256,
                callbacks=[
                    EarlyStopping(patience=3, restore_best_weights=True),
                    ReduceLROnPlateau(patience=2)
                ],
                verbose=1
            )
            
            # 평가
            y_pred = model.predict(X_test, verbose=0).flatten()
            
            # 역변환
            y_test_real = processor.target_scaler.inverse_transform(y_test.reshape(-1,1)).flatten()
            y_pred_real = processor.target_scaler.inverse_transform(y_pred.reshape(-1,1)).flatten()
            
            mae = np.mean(np.abs(y_test_real - y_pred_real))
            
            # 1700+ 성능
            actual_1700 = y_test_real >= 1700
            pred_1700 = y_pred_real >= 1700
            
            tp = np.sum(pred_1700 & actual_1700)
            fp = np.sum(pred_1700 & ~actual_1700)
            fn = np.sum(~pred_1700 & actual_1700)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            results[name] = {
                'mae': mae,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            print(f"  MAE: {mae:.2f}")
            print(f"  F1: {f1:.2%}")
            
        except Exception as e:
            print(f"  ❌ 오류: {str(e)}")
    
    # 순위
    print("\n" + "="*60)
    print("🏆 최종 순위")
    print("="*60)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True)
    for i, (name, res) in enumerate(sorted_results, 1):
        print(f"{i:2d}. {name:20s} | F1: {res['f1']:.2%} | MAE: {res['mae']:.2f}")
    
    print("\n✅ 완료!")
    return results

if __name__ == "__main__":
    results = main()