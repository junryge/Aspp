# -*- coding: utf-8 -*-
"""
🔥 ExtremeNet V7.0 - 최종 통합 버전 (학습 + 예측)
================================================================
- 메모리 안정성: tf.data.Dataset 파이프라인으로 대용량 데이터 처리
- 기능 통합: 학습(TrainingManager)과 예측(EnhancedPredictor) 기능 모두 포함
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import *
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
import os
import pickle

warnings.filterwarnings('ignore')
tf.random.set_seed(42)
np.random.seed(42)

# --- GPU 설정 ---
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    strategy = tf.distribute.MirroredStrategy()
else:
    strategy = tf.distribute.get_strategy()

print("="*80)
print("🔥 ExtremeNet V7.0 - 최종 통합 버전")
print(f"📦 TensorFlow: {tf.__version__}")
print(f"🚀 실행 전략: {type(strategy).__name__}")
print("="*80)

# ========================================
# 1. 데이터 처리 클래스 (메모리 효율화)
# ========================================
class EnhancedDataProcessor:
    def __init__(self):
        self.seq_len = 100
        self.pred_len = 10
        self.scaler_X = StandardScaler()
        self.scaler_y = RobustScaler()
        self.NORMAL_MAX, self.SIGNAL_MIN, self.DANGER_MIN = 1400, 1651, 1700
        self.feature_columns = []

    def fit_scalers(self, df):
        print("\n🛠️ 스케일러 학습 중...")
        if not self.feature_columns:
            _ = self.create_advanced_features(df.head())
        X_data = df[self.feature_columns].values
        y_data = df['TOTALCNT'].values.reshape(-1, 1)
        self.scaler_X.fit(X_data)
        self.scaler_y.fit(y_data)
        os.makedirs('scalers_v7', exist_ok=True)
        with open('scalers_v7/scaler_X.pkl', 'wb') as f: pickle.dump(self.scaler_X, f)
        with open('scalers_v7/scaler_y.pkl', 'wb') as f: pickle.dump(self.scaler_y, f)
        print("✅ 스케일러 학습 및 저장 완료")

    def create_advanced_features(self, df):
        print(f"⚙️ {len(df)}개 데이터에 대한 고급 특성 생성 중...")
        df_copy = df.copy()
        df_copy['RATIO'] = df_copy['M14AM14B'] / (df_copy['M14AM10A'] + 1e-6)
        df_copy['SIGNAL_ZONE'] = ((df_copy['TOTALCNT'] >= self.SIGNAL_MIN) & (df_copy['TOTALCNT'] < self.DANGER_MIN)).astype(int)
        df_copy['DANGER_ZONE'] = (df_copy['TOTALCNT'] >= self.DANGER_MIN).astype(int)
        for w in [5, 10, 20, 30]:
            df_copy[f'MA_{w}'] = df_copy['TOTALCNT'].rolling(w, min_periods=1).mean()
            df_copy[f'STD_{w}'] = df_copy['TOTALCNT'].rolling(w, min_periods=1).std()
        for lag in [1, 5, 10, 15]:
            df_copy[f'CHANGE_{lag}'] = df_copy['TOTALCNT'].diff(lag)
        if not self.feature_columns:
            base_cols = ['TOTALCNT', 'M14AM14B', 'M14AM14BSUM', 'M14AM10A', 'M14AM16']
            # df.columns.difference(base_cols) is not reliable, use generated cols
            generated_cols = [c for c in df_copy.columns if c not in df.columns]
            self.feature_columns = base_cols + generated_cols
            if 'CURRTIME' in self.feature_columns: self.feature_columns.remove('CURRTIME')
        df_copy.fillna(method='ffill', inplace=True)
        df_copy.fillna(method='bfill', inplace=True)
        df_copy.fillna(0, inplace=True)
        return df_copy

    def data_generator(self, df, is_training=True):
        featured_df = self.create_advanced_features(df)
        X_data_values = featured_df[self.feature_columns].values
        y_data_series = df['TOTALCNT'].shift(-self.pred_len)
        for i in range(len(df) - self.seq_len - self.pred_len):
            seq_X = X_data_values[i:i+self.seq_len]
            target = y_data_series.iloc[i+self.seq_len-1]
            if np.isnan(target): continue
            level, danger = self._create_labels(target)
            yield (seq_X, (target, level, danger))
            if is_training:
                current = df['TOTALCNT'].iloc[i+self.seq_len-1]
                if 1650 <= current < 1700 and target >= 1700:
                    for _ in range(5): yield (seq_X, (target, level, danger))
                elif current >= 1700 and target > current:
                    for _ in range(4): yield (seq_X, (target, level, danger))
                elif current >= 1680 and target > current:
                    for _ in range(3): yield (seq_X, (target, level, danger))

    def _create_labels(self, target):
        if target >= self.DANGER_MIN: return 3, 1
        if target >= self.SIGNAL_MIN: return 2, 0
        if target > self.NORMAL_MAX: return 1, 0
        return 0, 0

    def get_dataset(self, df, batch_size=64, is_training=True):
        generator = lambda: self.data_generator(df, is_training)
        output_signature = (
            tf.TensorSpec(shape=(self.seq_len, len(self.feature_columns)), dtype=tf.float32),
            (tf.TensorSpec(shape=(), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.int32), tf.TensorSpec(shape=(), dtype=tf.int32))
        )
        dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
        def scale_data(x, y):
            x_scaled = tf.py_function(self.scaler_X.transform, [x], tf.float32)
            y_reg_scaled = tf.py_function(lambda val: self.scaler_y.transform(val.numpy().reshape(-1, 1)), [y[0]], tf.float32)
            x_scaled.set_shape([self.seq_len, len(self.feature_columns)])
            return x_scaled, {'regression': tf.reshape(y_reg_scaled, ()), 'classification': y[1], 'danger': y[2]}
        dataset = dataset.map(scale_data, num_parallel_calls=tf.data.AUTOTUNE)
        if is_training: dataset = dataset.shuffle(10000)
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# ========================================
# 2. 모델 아키텍처 및 손실 함수
# ========================================
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([Dense(ff_dim, activation="relu"), Dense(embed_dim)])
        self.layernorm1, self.layernorm2 = LayerNormalization(epsilon=1e-6), LayerNormalization(epsilon=1e-6)
        self.dropout1, self.dropout2 = Dropout(rate), Dropout(rate)
    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        out1 = self.layernorm1(inputs + self.dropout1(attn_output, training=training))
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + self.dropout2(ffn_output, training=training))

def build_extremenet_v7(input_shape):
    inputs = Input(shape=input_shape)
    transformer = TransformerBlock(input_shape[-1], 8, 128)(inputs)
    transformer_pool = GlobalAveragePooling1D()(transformer)
    lstm = Bidirectional(LSTM(64))(inputs)
    merged = Concatenate()([transformer_pool, lstm])
    x = Dropout(0.3)(BatchNormalization()(merged))
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(BatchNormalization()(x))
    output_reg = Dense(1, name='regression')(Dense(64, activation='relu')(x))
    output_cls = Dense(4, activation='softmax', name='classification')(Dense(32, activation='relu')(x))
    output_danger = Dense(1, activation='sigmoid', name='danger')(Dense(32, activation='relu')(x))
    return Model(inputs, [output_reg, output_cls, output_danger], name='ExtremeNet_V7')

class AdaptiveLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        mae = tf.abs(y_true - y_pred)
        weights = 1.0 + tf.square(y_true) * 5.0
        return tf.reduce_mean(mae * weights)

# ========================================
# 3. 학습 관리 클래스
# ========================================
class TrainingManager:
    def __init__(self, processor):
        self.processor = processor
    def train(self, df_train, df_val):
        self.processor.fit_scalers(df_train)
        train_ds = self.processor.get_dataset(df_train, batch_size=64, is_training=True)
        val_ds = self.processor.get_dataset(df_val, batch_size=256, is_training=False)
        with strategy.scope():
            model = build_extremenet_v7((self.processor.seq_len, len(self.processor.feature_columns)))
            model.compile(
                optimizer=Adam(learning_rate=1e-3),
                loss={'regression': AdaptiveLoss(), 'classification': 'sparse_categorical_crossentropy', 'danger': 'binary_crossentropy'},
                loss_weights={'regression': 0.6, 'classification': 0.1, 'danger': 0.3},
                metrics={'regression': 'mae'}
            )
        print("\n🔥 ExtremeNet V7.0 학습 시작...")
        callbacks = [
            ModelCheckpoint('models_v7/best_model.keras', save_best_only=True, monitor='val_regression_mae', mode='min'),
            EarlyStopping(patience=20, restore_best_weights=True, monitor='val_regression_mae', mode='min'),
            ReduceLROnPlateau(patience=8, factor=0.5, min_lr=1e-7, monitor='val_regression_mae')
        ]
        history = model.fit(train_ds, validation_data=val_ds, epochs=100, callbacks=callbacks)
        model.save('models_v7/extremenet_v7_final.keras')
        print("\n💾 최종 모델 저장 완료: models_v7/extremenet_v7_final.keras")
        return model

# ========================================
# 4. 실시간 예측 클래스
# ========================================
class EnhancedPredictor:
    def __init__(self, model_path='models_v7/extremenet_v7_final.keras'):
        print("\n🚀 예측기 로딩 중...")
        if not os.path.exists(model_path): raise FileNotFoundError(f"모델 파일 없음: {model_path}")
        self.model = tf.keras.models.load_model(model_path, custom_objects={'AdaptiveLoss': AdaptiveLoss})
        with open('scalers_v7/scaler_X.pkl', 'rb') as f: self.scaler_X = pickle.load(f)
        with open('scalers_v7/scaler_y.pkl', 'rb') as f: self.scaler_y = pickle.load(f)
        self.processor = EnhancedDataProcessor()
        self.processor.feature_columns = self.scaler_X.feature_names_in_
        print("✅ 예측기 준비 완료")

    def predict(self, df_100min):
        current_value = df_100min['TOTALCNT'].iloc[-1]
        featured_df = self.processor.create_advanced_features(df_100min)
        X_scaled = self.scaler_X.transform(featured_df[self.processor.feature_columns].values)
        X_scaled = X_scaled.reshape(1, self.processor.seq_len, len(self.processor.feature_columns))
        preds = self.model.predict(X_scaled, verbose=0)
        y_pred = self.scaler_y.inverse_transform(preds[0])[0, 0]
        danger_prob = preds[2][0, 0] * 100
        diagnosis = self._diagnose(current_value, y_pred)
        return {
            'current': f"{current_value:.0f}", 'prediction': f"{y_pred:.0f}",
            'change': f"{y_pred - current_value:+.0f}",
            'danger_probability': f"{danger_prob:.1f}%", 'level': diagnosis['level'],
            'message': diagnosis['message'], 'action': diagnosis['action']
        }

    def _diagnose(self, current, predicted):
        level, msg, act = '🟢 정상', f'안정 범위 유지: {current:.0f} → {predicted:.0f}', '✓ 정상 운영'
        if predicted >= 1700 and current < 1700:
            level, msg, act = '🔴🔴🔴 극도위험', f'1700+ 돌파 예상! {current:.0f} → {predicted:.0f}', '🚨 즉시 비상대응!'
        elif predicted >= 1700 and current >= 1700:
            level, msg, act = '🔴🔴 매우위험', f'위험구간 내 상승 지속: {current:.0f} → {predicted:.0f}', '⚠️ 긴급대응 유지'
        elif current >= 1700 and predicted < 1700:
            level, msg, act = '🟠 위험완화', f'피크 통과, 하락 전환: {current:.0f} → {predicted:.0f}', '📉 안정화 확인'
        elif predicted > 1680:
            level, msg, act = '🟠🟠 주의', f'위험 임박, 급상승 감지: {current:.0f} → {predicted:.0f}', '🔔 대응팀 대기'
        return {'level': level, 'message': msg, 'action': act}

# ========================================
# 5. 메인 실행 블록
# ========================================
def main():
    # --- 학습 실행 ---
    data_path = '20240201_TO_202507281705.csv'
    if not os.path.exists(data_path):
        print(f"❌ 학습 데이터 파일 없음: {data_path}. 예측기만 로드합니다.")
    else:
        df = pd.read_csv(data_path, parse_dates=['CURRTIME'])
        df = df.sort_values('CURRTIME').reset_index(drop=True).dropna(subset=['CURRTIME'])
        df = df[df['TOTALCNT'] > 0]
        print(f"✅ 데이터 {len(df):,}개 로드 완료")
        train_end, val_end = int(len(df) * 0.7), int(len(df) * 0.85)
        df_train, df_val = df.iloc[:train_end], df.iloc[train_end:val_end]
        print(f"📊 데이터 분할: Train({len(df_train)}), Val({len(df_val)})")
        processor = EnhancedDataProcessor()
        trainer = TrainingManager(processor)
        trainer.train(df_train, df_val)
        print("\n✅ 모든 학습 과정 완료!")

    # --- 예측기 사용 예시 ---
    try:
        predictor = EnhancedPredictor()
        if 'df' in locals() and len(df) >= 200:
            # 학습에 사용되지 않은 최신 100분 데이터로 예측 테스트
            sample_data = df.iloc[-100:].reset_index(drop=True)
            print("\n" + "="*80)
            print("🔮 예측기 테스트")
            print(f"입력 데이터: 100개 (시간: {sample_data['CURRTIME'].iloc[0]} ~ {sample_data['CURRTIME'].iloc[-1]})")
            result = predictor.predict(sample_data)
            print("\n--- 예측 결과 ---")
            for key, value in result.items():
                print(f"{key.replace('_', ' ').title():<20}: {value}")
            print("="*80)
        else:
            print("\n🔮 예측 테스트를 위한 데이터가 부족합니다.")
    except Exception as e:
        print(f"\n❌ 예측기 실행 중 오류 발생: {e}")

if __name__ == "__main__":
    main()