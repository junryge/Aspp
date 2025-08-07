"""
CNN-LSTM Multi-Task 모델 즉시 학습 실행
=====================================
이 파일을 실행하면 TO.CSV 데이터로 바로 모델이 학습됩니다!
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (Input, Conv1D, LSTM, Dense, Dropout, 
                                    BatchNormalization, Bidirectional, 
                                    MaxPooling1D, Activation)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta
import joblib
import warnings

warnings.filterwarnings('ignore')

# CPU 모드 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

# 랜덤 시드 고정
RANDOM_SEED = 2079936
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("="*60)
print("🚀 CNN-LSTM Multi-Task 모델 학습 시작!")
print("="*60)

# ===================================
# 데이터 전처리 함수들
# ===================================

def load_and_preprocess_data(data_path):
    """데이터 로드 및 전처리"""
    print(f"\n📁 데이터 로딩: {data_path}")
    
    # 데이터 로드
    data = pd.read_csv(data_path)
    print(f"✓ 데이터 로드 완료 - shape: {data.shape}")
    
    # 시간 컬럼 변환
    data['CURRTIME'] = pd.to_datetime(data['CURRTIME'], format='%Y%m%d%H%M')
    data['TIME'] = pd.to_datetime(data['TIME'], format='%Y%m%d%H%M')
    
    # SUM 컬럼 제거
    columns_to_drop = [col for col in data.columns if 'SUM' in col]
    data = data.drop(columns=columns_to_drop)
    
    # 인덱스를 시간으로 설정
    data.set_index('CURRTIME', inplace=True)
    
    return data

def create_features(data):
    """특징 엔지니어링"""
    print("\n🔧 특징 생성 중...")
    
    features_data = data.copy()
    
    # 시간 특징
    features_data['hour'] = features_data.index.hour
    features_data['dayofweek'] = features_data.index.dayofweek
    features_data['is_weekend'] = (features_data.index.dayofweek >= 5).astype(int)
    
    # 팹 간 불균형 지표
    features_data['imbalance_M14A_M10A'] = features_data['M14AM10A'] - features_data['M10AM14A']
    features_data['imbalance_M14A_M14B'] = features_data['M14AM14B'] - features_data['M14BM14A']
    features_data['imbalance_M14A_M16'] = features_data['M14AM16'] - features_data['M16M14A']
    
    # 이동 평균
    features_data['MA_5'] = features_data['TOTALCNT'].rolling(window=5, min_periods=1).mean()
    features_data['MA_10'] = features_data['TOTALCNT'].rolling(window=10, min_periods=1).mean()
    features_data['MA_30'] = features_data['TOTALCNT'].rolling(window=30, min_periods=1).mean()
    
    # 표준편차
    features_data['STD_5'] = features_data['TOTALCNT'].rolling(window=5, min_periods=1).std()
    features_data['STD_10'] = features_data['TOTALCNT'].rolling(window=10, min_periods=1).std()
    
    # 팹별 부하율
    features_data['load_M14A_out'] = (features_data['M14AM10A'] + features_data['M14AM14B'] + 
                                      features_data['M14AM16']) / features_data['TOTALCNT']
    features_data['load_M14A_in'] = (features_data['M10AM14A'] + features_data['M14BM14A'] + 
                                     features_data['M16M14A']) / features_data['TOTALCNT']
    
    # 변화율
    features_data['change_rate'] = features_data['TOTALCNT'].pct_change()
    
    # 결측값 처리
    features_data = features_data.fillna(method='ffill').fillna(0)
    
    print(f"✓ 특징 생성 완료 - shape: {features_data.shape}")
    
    return features_data

def create_targets(data, future_minutes=10):
    """타겟 변수 생성"""
    print(f"\n🎯 타겟 변수 생성 중 ({future_minutes}분 후 예측)...")
    
    # 1. 물류량 타겟 (회귀)
    data['FUTURE_TOTALCNT'] = pd.NA
    
    for i in data.index:
        future_time = i + pd.Timedelta(minutes=future_minutes)
        if (future_time <= data.index.max()) & (future_time in data.index):
            data.loc[i, 'FUTURE_TOTALCNT'] = data.loc[future_time, 'TOTALCNT']
    
    # 2. 병목 위치 타겟 (분류)
    thresholds = {
        'total': data['TOTALCNT'].quantile(0.85),  # 상위 15%
        'm14a_m10a': 300,
        'm14a_m14b': 350,
        'm14a_m16': 300
    }
    
    data['BOTTLENECK_LOCATION'] = 0  # 0: 병목 없음
    
    for i in data.index:
        future_time = i + pd.Timedelta(minutes=future_minutes)
        if (future_time <= data.index.max()) & (future_time in data.index):
            future_total = data.loc[future_time, 'TOTALCNT']
            
            if future_total > thresholds['total']:
                route_loads = {
                    1: data.loc[future_time, 'M14AM10A'] + data.loc[future_time, 'M10AM14A'],
                    2: data.loc[future_time, 'M14AM14B'] + data.loc[future_time, 'M14BM14A'],
                    3: data.loc[future_time, 'M14AM16'] + data.loc[future_time, 'M16M14A']
                }
                
                max_route = max(route_loads.items(), key=lambda x: x[1])
                if max_route[1] > thresholds[f'm14a_{["m10a", "m14b", "m16"][max_route[0]-1]}']:
                    data.loc[i, 'BOTTLENECK_LOCATION'] = max_route[0]
    
    # NA 제거
    data = data.dropna(subset=['FUTURE_TOTALCNT'])
    
    print(f"✓ 타겟 생성 완료")
    print(f"  - 병목 분포: {data['BOTTLENECK_LOCATION'].value_counts().to_dict()}")
    
    return data

def scale_features(data, feature_columns):
    """특징 스케일링"""
    scaler = StandardScaler()
    
    scale_columns = [col for col in feature_columns if col in data.columns]
    scaled_data = scaler.fit_transform(data[scale_columns])
    
    scaled_df = pd.DataFrame(
        scaled_data, 
        columns=[f'scaled_{col}' for col in scale_columns],
        index=data.index
    )
    
    result = pd.concat([data, scaled_df], axis=1)
    
    return result, scaler

def create_sequences(data, feature_cols, target_cols, seq_length=30):
    """시퀀스 데이터 생성"""
    X, y_regression, y_classification = [], [], []
    
    time_diff = data.index.to_series().diff()
    split_points = time_diff > pd.Timedelta(minutes=1)
    segment_ids = split_points.cumsum()
    
    for segment_id in segment_ids.unique():
        segment = data[segment_ids == segment_id]
        
        if len(segment) > seq_length:
            feature_data = segment[feature_cols].values
            regression_data = segment[target_cols[0]].values
            classification_data = segment[target_cols[1]].values
            
            for i in range(len(segment) - seq_length):
                X.append(feature_data[i:i+seq_length])
                y_regression.append(regression_data[i+seq_length])
                y_classification.append(classification_data[i+seq_length])
    
    return np.array(X), np.array(y_regression), np.array(y_classification)

# ===================================
# CNN-LSTM Multi-Task 모델
# ===================================

def build_cnn_lstm_multitask_model(input_shape, num_classes=4):
    """CNN-LSTM Multi-Task 모델 구축"""
    
    inputs = Input(shape=input_shape, name='input')
    
    # CNN 파트
    x = Conv1D(filters=64, kernel_size=3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    
    x = Conv1D(filters=128, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    
    x = Conv1D(filters=128, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = MaxPooling1D(pool_size=2)(x)
    
    # LSTM 파트
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    
    x = Bidirectional(LSTM(64, return_sequences=False))(x)
    x = Dropout(0.3)(x)
    
    # 공유 Dense 레이어
    shared = Dense(128, activation='relu', name='shared_layer')(x)
    shared = BatchNormalization()(shared)
    shared = Dropout(0.3)(shared)
    
    # Multi-Task 출력
    # Task 1: 물류량 예측 (회귀)
    logistics_output = Dense(64, activation='relu')(shared)
    logistics_output = Dense(1, name='logistics_output')(logistics_output)
    
    # Task 2: 병목 위치 예측 (분류)
    bottleneck_output = Dense(64, activation='relu')(shared)
    bottleneck_output = Dense(num_classes, activation='softmax', name='bottleneck_output')(bottleneck_output)
    
    model = Model(inputs=inputs, outputs=[logistics_output, bottleneck_output])
    
    return model

# ===================================
# 학습 함수
# ===================================

def train_model(model, X_train, y_train_reg, y_train_cls, X_val, y_val_reg, y_val_cls, 
                epochs=50, batch_size=32):  # 에포크를 50으로 줄여서 빠르게 테스트
    """모델 학습"""
    
    losses = {
        'logistics_output': 'mse',
        'bottleneck_output': 'sparse_categorical_crossentropy'
    }
    
    loss_weights = {
        'logistics_output': 0.7,
        'bottleneck_output': 0.3
    }
    
    metrics = {
        'logistics_output': ['mae'],
        'bottleneck_output': ['accuracy']
    }
    
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics
    )
    
    # 디렉토리 생성
    os.makedirs('model', exist_ok=True)
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            'model/cnn_lstm_multitask_best.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    print("\n🚀 모델 학습 시작...")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Train samples: {len(X_train)}")
    print(f"  - Validation samples: {len(X_val)}")
    
    history = model.fit(
        X_train,
        {'logistics_output': y_train_reg, 'bottleneck_output': y_train_cls},
        validation_data=(
            X_val,
            {'logistics_output': y_val_reg, 'bottleneck_output': y_val_cls}
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

# ===================================
# 메인 실행 함수
# ===================================

def main():
    """메인 실행 함수"""
    
    # 1. 데이터 로드 - TO.CSV 파일 자동 탐색
    data_paths = ['TO.CSV', 'data/TO.CSV', './TO.CSV', '../TO.CSV']
    data_loaded = False
    
    for path in data_paths:
        if os.path.exists(path):
            data = load_and_preprocess_data(path)
            data_loaded = True
            break
    
    if not data_loaded:
        print("❌ TO.CSV 파일을 찾을 수 없습니다!")
        print("   현재 디렉토리에 TO.CSV 파일을 넣어주세요.")
        return None, None, None
    
    # 2. 특징 생성
    data = create_features(data)
    
    # 3. 타겟 생성
    data = create_targets(data)
    
    # 4. 특징 스케일링
    scale_features_list = [
        'TOTALCNT', 'M14AM10A', 'M10AM14A', 'M14AM14B', 'M14BM14A', 'M14AM16', 'M16M14A',
        'imbalance_M14A_M10A', 'imbalance_M14A_M14B', 'imbalance_M14A_M16',
        'MA_5', 'MA_10', 'MA_30', 'STD_5', 'STD_10',
        'load_M14A_out', 'load_M14A_in'
    ]
    
    data, scaler = scale_features(data, scale_features_list)
    
    # 5. 시퀀스 생성
    sequence_features = [col for col in data.columns if col.startswith('scaled_')]
    target_features = ['FUTURE_TOTALCNT', 'BOTTLENECK_LOCATION']
    
    X, y_regression, y_classification = create_sequences(
        data, sequence_features, target_features, seq_length=30
    )
    
    print(f"\n📊 시퀀스 데이터 생성 완료:")
    print(f"  - X shape: {X.shape}")
    print(f"  - y_regression shape: {y_regression.shape}")
    print(f"  - y_classification shape: {y_classification.shape}")
    
    # 6. 데이터 분할
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train = X[:train_size]
    X_val = X[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    
    y_train_reg = y_regression[:train_size]
    y_val_reg = y_regression[train_size:train_size+val_size]
    y_test_reg = y_regression[train_size+val_size:]
    
    y_train_cls = y_classification[:train_size]
    y_val_cls = y_classification[train_size:train_size+val_size]
    y_test_cls = y_classification[train_size+val_size:]
    
    # 7. 모델 생성
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(np.unique(y_classification))
    
    print(f"\n🏗️ 모델 구축 중...")
    print(f"  - Input shape: {input_shape}")
    print(f"  - Number of classes: {num_classes}")
    
    model = build_cnn_lstm_multitask_model(input_shape, num_classes)
    model.summary()
    
    # 8. 모델 학습
    history = train_model(
        model, 
        X_train, y_train_reg, y_train_cls,
        X_val, y_val_reg, y_val_cls,
        epochs=50,  # 빠른 테스트를 위해 50 에포크
        batch_size=32
    )
    
    # 9. 모델 평가
    print("\n📈 모델 평가 중...")
    
    predictions = model.predict(X_test)
    pred_logistics = predictions[0].flatten()
    pred_bottleneck = predictions[1]
    
    # 물류량 예측 평가
    mae = mean_absolute_error(y_test_reg, pred_logistics)
    mse = mean_squared_error(y_test_reg, pred_logistics)
    
    print(f"\n📊 물류량 예측 성능:")
    print(f"  - MAE: {mae:.2f}")
    print(f"  - MSE: {mse:.2f}")
    print(f"  - RMSE: {np.sqrt(mse):.2f}")
    
    # 병목 예측 평가
    pred_bottleneck_classes = np.argmax(pred_bottleneck, axis=1)
    accuracy = accuracy_score(y_test_cls, pred_bottleneck_classes)
    
    print(f"\n🚨 병목 위치 예측 성능:")
    print(f"  - Accuracy: {accuracy:.2%}")
    print("\n분류 리포트:")
    print(classification_report(y_test_cls, pred_bottleneck_classes, 
                              target_names=['정상', 'M14A-M10A', 'M14A-M14B', 'M14A-M16']))
    
    # 10. 모델 저장
    print("\n💾 모델 및 스케일러 저장 중...")
    
    os.makedirs('model', exist_ok=True)
    os.makedirs('scaler', exist_ok=True)
    
    model.save('model/cnn_lstm_multitask_final.keras')
    joblib.dump(scaler, 'scaler/multitask_scaler.pkl')
    
    print("✓ 모델 저장 완료: model/cnn_lstm_multitask_final.keras")
    print("✓ 스케일러 저장 완료: scaler/multitask_scaler.pkl")
    
    # 11. 학습 곡선 시각화
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['logistics_output_mae'], label='Train MAE')
    plt.plot(history.history['val_logistics_output_mae'], label='Val MAE')
    plt.title('Logistics MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(history.history['bottleneck_output_accuracy'], label='Train Acc')
    plt.plot(history.history['val_bottleneck_output_accuracy'], label='Val Acc')
    plt.title('Bottleneck Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("\n✓ 학습 곡선 저장: training_history.png")
    
    print("\n" + "="*60)
    print("🎉 학습 완료!")
    print("="*60)
    
    return model, scaler, history

# ===================================
# 실행!!
# ===================================

if __name__ == "__main__":
    print("\n🔥 CNN-LSTM Multi-Task 모델 학습을 시작합니다!")
    print("   TO.CSV 데이터를 사용하여 모델을 생성합니다.\n")
    
    # 학습 실행
    model, scaler, history = main()
    
    if model is not None:
        print("\n✅ 모델 생성 완료!")
        print("\n📁 생성된 파일:")
        print("   - model/cnn_lstm_multitask_final.keras (모델)")
        print("   - scaler/multitask_scaler.pkl (스케일러)")
        print("   - training_history.png (학습 곡선)")
        print("\n이제 이 모델을 사용해서 예측할 수 있습니다!")