"""
CNN-LSTM Multi-Task 기반 반도체 물류 예측 모델 (수정본)
==============================================
전체 물류량 예측과 병목 구간 예측을 동시에 수행하는 하이브리드 모델

주요 수정사항:
1. load_model import 추가
2. 함수 위치 조정으로 접근성 개선
3. 예측 함수들을 클래스 밖으로 이동

개발일: 2024년
버전: 1.1 (수정본)
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model  # load_model 추가
from tensorflow.keras.layers import (Input, Conv1D, LSTM, Dense, Dropout, 
                                    BatchNormalization, Bidirectional, 
                                    MaxPooling1D, Activation, Flatten,
                                    GlobalAveragePooling1D)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta
import joblib
import logging
import warnings

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')

# ===================================
# 1. 환경 설정 및 초기화
# ===================================

# CPU 모드 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

# 랜덤 시드 고정
RANDOM_SEED = 2079936
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===================================
# 2. 데이터 전처리 함수들
# ===================================

def load_and_preprocess_data(data_path):
    """데이터 로드 및 전처리"""
    logger.info("데이터 로딩 중...")
    
    # 데이터 로드
    Full_Data = pd.read_csv(data_path)
    
    # 시간 컬럼 변환
    Full_Data['CURRTIME'] = pd.to_datetime(Full_Data['CURRTIME'], format='%Y%m%d%H%M')
    Full_Data['TIME'] = pd.to_datetime(Full_Data['TIME'], format='%Y%m%d%H%M')
    
    # SUM 컬럼 제거
    columns_to_drop = [col for col in Full_Data.columns if 'SUM' in col]
    Full_Data = Full_Data.drop(columns=columns_to_drop)
    
    # 인덱스를 시간으로 설정
    Full_Data.set_index('CURRTIME', inplace=True)
    
    logger.info(f"원본 데이터 shape: {Full_Data.shape}")
    
    return Full_Data

def create_features(data):
    """특징 엔지니어링"""
    logger.info("특징 생성 중...")
    
    # 기본 특징
    features_data = data.copy()
    
    # 시간 특징
    features_data['hour'] = features_data.index.hour
    features_data['dayofweek'] = features_data.index.dayofweek
    features_data['is_weekend'] = (features_data.index.dayofweek >= 5).astype(int)
    
    # 팹 간 불균형 지표
    if 'M14AM10A' in features_data.columns and 'M10AM14A' in features_data.columns:
        features_data['imbalance_M14A_M10A'] = features_data['M14AM10A'] - features_data['M10AM14A']
    if 'M14AM14B' in features_data.columns and 'M14BM14A' in features_data.columns:
        features_data['imbalance_M14A_M14B'] = features_data['M14AM14B'] - features_data['M14BM14A']
    if 'M14AM16' in features_data.columns and 'M16M14A' in features_data.columns:
        features_data['imbalance_M14A_M16'] = features_data['M14AM16'] - features_data['M16M14A']
    
    # 이동 평균 (전체 물량)
    features_data['MA_5'] = features_data['TOTALCNT'].rolling(window=5, min_periods=1).mean()
    features_data['MA_10'] = features_data['TOTALCNT'].rolling(window=10, min_periods=1).mean()
    features_data['MA_30'] = features_data['TOTALCNT'].rolling(window=30, min_periods=1).mean()
    
    # 표준편차 (변동성)
    features_data['STD_5'] = features_data['TOTALCNT'].rolling(window=5, min_periods=1).std()
    features_data['STD_10'] = features_data['TOTALCNT'].rolling(window=10, min_periods=1).std()
    
    # 팹별 부하율 (컬럼이 있는 경우만)
    if all(col in features_data.columns for col in ['M14AM10A', 'M14AM14B', 'M14AM16']):
        features_data['load_M14A_out'] = (features_data['M14AM10A'] + features_data['M14AM14B'] + 
                                          features_data['M14AM16']) / features_data['TOTALCNT']
    if all(col in features_data.columns for col in ['M10AM14A', 'M14BM14A', 'M16M14A']):
        features_data['load_M14A_in'] = (features_data['M10AM14A'] + features_data['M14BM14A'] + 
                                         features_data['M16M14A']) / features_data['TOTALCNT']
    
    # 변화율
    features_data['change_rate'] = features_data['TOTALCNT'].pct_change()
    
    # 결측값 처리
    features_data = features_data.fillna(method='ffill').fillna(0)
    
    logger.info(f"특징 생성 완료 - shape: {features_data.shape}")
    
    return features_data

def create_targets(data, future_minutes=10):
    """타겟 변수 생성 (물류량 + 병목 위치)"""
    logger.info("타겟 변수 생성 중...")
    
    # 1. 물류량 타겟 (회귀)
    data['FUTURE_TOTALCNT'] = pd.NA
    
    for i in data.index:
        future_time = i + pd.Timedelta(minutes=future_minutes)
        if (future_time <= data.index.max()) & (future_time in data.index):
            data.loc[i, 'FUTURE_TOTALCNT'] = data.loc[future_time, 'TOTALCNT']
    
    # 2. 병목 위치 타겟 (분류)
    # 임계값 설정 (데이터 분포 기반)
    thresholds = {
        'total': data['TOTALCNT'].quantile(0.85),  # 상위 15%
        'm14a_m10a': 300,  # 경로별 임계값
        'm14a_m14b': 350,
        'm14a_m16': 300
    }
    
    data['BOTTLENECK_LOCATION'] = 0  # 0: 병목 없음
    
    # 경로별 컬럼이 있는 경우만 병목 예측
    route_columns_exist = all(col in data.columns for col in ['M14AM10A', 'M10AM14A', 'M14AM14B', 'M14BM14A', 'M14AM16', 'M16M14A'])
    
    if route_columns_exist:
        for i in data.index:
            future_time = i + pd.Timedelta(minutes=future_minutes)
            if (future_time <= data.index.max()) & (future_time in data.index):
                future_total = data.loc[future_time, 'TOTALCNT']
                
                if future_total > thresholds['total']:
                    # 어느 경로가 가장 혼잡한지 확인
                    route_loads = {
                        1: data.loc[future_time, 'M14AM10A'] + data.loc[future_time, 'M10AM14A'],  # M14A-M10A
                        2: data.loc[future_time, 'M14AM14B'] + data.loc[future_time, 'M14BM14A'],  # M14A-M14B
                        3: data.loc[future_time, 'M14AM16'] + data.loc[future_time, 'M16M14A']     # M14A-M16
                    }
                    
                    # 가장 혼잡한 경로를 병목으로 지정
                    max_route = max(route_loads.items(), key=lambda x: x[1])
                    if max_route[1] > thresholds[f'm14a_{["m10a", "m14b", "m16"][max_route[0]-1]}']:
                        data.loc[i, 'BOTTLENECK_LOCATION'] = max_route[0]
    
    # NA 제거
    data = data.dropna(subset=['FUTURE_TOTALCNT'])
    
    logger.info(f"타겟 생성 완료 - 병목 분포: {data['BOTTLENECK_LOCATION'].value_counts()}")
    
    return data

def scale_features(data, feature_columns):
    """특징 스케일링"""
    scaler = StandardScaler()
    
    # 스케일링할 컬럼 선택
    scale_columns = [col for col in feature_columns if col in data.columns]
    
    # 스케일링
    scaled_data = scaler.fit_transform(data[scale_columns])
    
    # 스케일링된 데이터프레임 생성
    scaled_df = pd.DataFrame(
        scaled_data, 
        columns=[f'scaled_{col}' for col in scale_columns],
        index=data.index
    )
    
    # 원본 데이터와 병합
    result = pd.concat([data, scaled_df], axis=1)
    
    return result, scaler

def create_sequences(data, feature_cols, target_cols, seq_length=30):
    """시퀀스 데이터 생성"""
    X, y_regression, y_classification = [], [], []
    
    # 연속성 확인
    time_diff = data.index.to_series().diff()
    split_points = time_diff > pd.Timedelta(minutes=1)
    segment_ids = split_points.cumsum()
    
    for segment_id in segment_ids.unique():
        segment = data[segment_ids == segment_id]
        
        if len(segment) > seq_length:
            feature_data = segment[feature_cols].values
            regression_data = segment[target_cols[0]].values  # FUTURE_TOTALCNT
            classification_data = segment[target_cols[1]].values  # BOTTLENECK_LOCATION
            
            for i in range(len(segment) - seq_length):
                X.append(feature_data[i:i+seq_length])
                y_regression.append(regression_data[i+seq_length])
                y_classification.append(classification_data[i+seq_length])
    
    return np.array(X), np.array(y_regression), np.array(y_classification)

# ===================================
# 3. CNN-LSTM Multi-Task 모델 정의
# ===================================

def build_cnn_lstm_multitask_model(input_shape, num_classes=4):
    """CNN-LSTM Multi-Task 모델 구축"""
    
    # 입력 레이어
    inputs = Input(shape=input_shape, name='input')
    
    # === CNN 파트 ===
    # 첫 번째 Conv1D 블록
    x = Conv1D(filters=64, kernel_size=3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    
    # 두 번째 Conv1D 블록
    x = Conv1D(filters=128, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    
    # 세 번째 Conv1D 블록
    x = Conv1D(filters=128, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Max Pooling (시퀀스 길이 줄이기)
    x = MaxPooling1D(pool_size=2)(x)
    
    # === LSTM 파트 ===
    # Bidirectional LSTM
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    
    x = Bidirectional(LSTM(64, return_sequences=False))(x)
    x = Dropout(0.3)(x)
    
    # === 공유 Dense 레이어 ===
    shared = Dense(128, activation='relu', name='shared_layer')(x)
    shared = BatchNormalization()(shared)
    shared = Dropout(0.3)(shared)
    
    # === Multi-Task 출력 ===
    # Task 1: 물류량 예측 (회귀)
    logistics_output = Dense(64, activation='relu')(shared)
    logistics_output = Dense(1, name='logistics_output')(logistics_output)
    
    # Task 2: 병목 위치 예측 (분류)
    bottleneck_output = Dense(64, activation='relu')(shared)
    bottleneck_output = Dense(num_classes, activation='softmax', name='bottleneck_output')(bottleneck_output)
    
    # 모델 생성
    model = Model(inputs=inputs, outputs=[logistics_output, bottleneck_output])
    
    return model

# ===================================
# 4. 예측 관련 함수들 (클래스 밖으로 이동)
# ===================================

def predict_realtime(model, scaler, recent_data):
    """
    실시간 예측 함수
    recent_data: 최근 30분 데이터 (DataFrame)
    """
    # 특징 생성
    features = create_features(recent_data)
    
    # 스케일링할 특징 선택
    scale_features_list = [
        'TOTALCNT', 'M14AM10A', 'M10AM14A', 'M14AM14B', 'M14BM14A', 'M14AM16', 'M16M14A',
        'imbalance_M14A_M10A', 'imbalance_M14A_M14B', 'imbalance_M14A_M16',
        'MA_5', 'MA_10', 'MA_30', 'STD_5', 'STD_10',
        'load_M14A_out', 'load_M14A_in'
    ]
    
    # 실제 존재하는 컬럼만 선택
    scale_columns = [col for col in scale_features_list if col in features.columns]
    
    # 스케일러가 학습한 컬럼 확인
    if hasattr(scaler, 'feature_names_in_'):
        # 스케일러가 학습한 컬럼만 사용
        scale_columns = [col for col in scale_columns if col in scaler.feature_names_in_]
    
    # 스케일링
    scaled_data = scaler.transform(features[scale_columns])
    
    # 스케일링된 데이터를 DataFrame으로 변환
    scaled_df = pd.DataFrame(
        scaled_data,
        columns=[f'scaled_{col}' for col in scale_columns],
        index=features.index
    )
    
    # 시퀀스용 특징 선택
    sequence_features = [col for col in scaled_df.columns if col.startswith('scaled_')]
    X_input = scaled_df[sequence_features].values
    
    # 모델 입력 형태로 변환 (1, 30, features)
    X_input = X_input.reshape(1, X_input.shape[0], X_input.shape[1])
    
    # 예측
    predictions = model.predict(X_input, verbose=0)
    
    # 결과 추출
    pred_logistics = predictions[0][0][0]  # 물류량 예측값
    pred_bottleneck = predictions[1][0]    # 병목 확률 (4개 클래스)
    pred_bottleneck_class = np.argmax(pred_bottleneck)  # 가장 높은 확률의 클래스
    
    # 병목 위치 매핑
    bottleneck_labels = ['정상', 'M14A-M10A 병목', 'M14A-M14B 병목', 'M14A-M16 병목']
    
    return {
        'logistics_prediction': pred_logistics,
        'bottleneck_location': bottleneck_labels[pred_bottleneck_class],
        'bottleneck_probability': pred_bottleneck[pred_bottleneck_class] * 100,
        'all_probabilities': {
            bottleneck_labels[i]: pred_bottleneck[i] * 100 
            for i in range(len(bottleneck_labels))
        }
    }

def print_prediction_result(result, current_time):
    """예측 결과를 보기 좋게 출력"""
    print(f"\n현재 시간: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"예측 대상 시간: {(current_time + timedelta(minutes=10)).strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 50)
    print(f"📊 10분 후 전체 물류량 예측: {result['logistics_prediction']:.0f}")
    print(f"🚨 병목 예측: {result['bottleneck_location']} (확률: {result['bottleneck_probability']:.1f}%)")
    print("\n📈 각 위치별 병목 확률:")
    for location, prob in result['all_probabilities'].items():
        bar = '█' * int(prob / 5)
        print(f"   {location:15} [{bar:20}] {prob:5.1f}%")
    print("-" * 50)

# ===================================
# 5. 학습 프로세스
# ===================================

def train_model(model, X_train, y_train_reg, y_train_cls, X_val, y_val_reg, y_val_cls, 
                epochs=200, batch_size=64):
    """모델 학습"""
    
    # 손실 함수와 가중치 설정
    losses = {
        'logistics_output': 'mse',
        'bottleneck_output': 'sparse_categorical_crossentropy'
    }
    
    loss_weights = {
        'logistics_output': 0.7,  # 물류량 예측에 더 높은 가중치
        'bottleneck_output': 0.3
    }
    
    metrics = {
        'logistics_output': ['mae'],
        'bottleneck_output': ['accuracy']
    }
    
    # 컴파일
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics
    )
    
    # 콜백 설정
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
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
    
    # 학습
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
# 6. 메인 실행 함수
# ===================================

def main():
    """메인 실행 함수"""
    
    logger.info("="*60)
    logger.info("CNN-LSTM Multi-Task 모델 학습 시작")
    logger.info("="*60)
    
    # 1. 데이터 로드 및 전처리
    data_path = 'data/20240201_TO_202507281705.csv'  # 전체 데이터 경로
    data = load_and_preprocess_data(data_path)
    
    # 2. 특징 생성
    data = create_features(data)
    
    # 3. 타겟 생성
    data = create_targets(data)
    
    # 4. 특징 선택
    # 스케일링할 특징
    scale_features_list = [
        'TOTALCNT', 'M14AM10A', 'M10AM14A', 'M14AM14B', 'M14BM14A', 'M14AM16', 'M16M14A',
        'imbalance_M14A_M10A', 'imbalance_M14A_M14B', 'imbalance_M14A_M16',
        'MA_5', 'MA_10', 'MA_30', 'STD_5', 'STD_10',
        'load_M14A_out', 'load_M14A_in'
    ]
    
    # 실제 존재하는 컬럼만 선택
    scale_features_list = [col for col in scale_features_list if col in data.columns]
    
    # 5. 스케일링
    data, scaler = scale_features(data, scale_features_list)
    
    # 6. 시퀀스용 특징 선택
    sequence_features = [col for col in data.columns if col.startswith('scaled_')]
    target_features = ['FUTURE_TOTALCNT', 'BOTTLENECK_LOCATION']
    
    # 7. 시퀀스 생성
    X, y_regression, y_classification = create_sequences(
        data, 
        sequence_features, 
        target_features,
        seq_length=30
    )
    
    logger.info(f"시퀀스 shape - X: {X.shape}, y_reg: {y_regression.shape}, y_cls: {y_classification.shape}")
    
    # 8. 데이터 분할
    # 시간 순서 유지를 위해 순차적으로 분할
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
    
    # 9. 모델 생성
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(np.unique(y_classification))
    
    model = build_cnn_lstm_multitask_model(input_shape, num_classes)
    model.summary()
    
    # 10. 모델 학습
    logger.info("\n모델 학습 시작...")
    history = train_model(
        model, 
        X_train, y_train_reg, y_train_cls,
        X_val, y_val_reg, y_val_cls,
        epochs=200,
        batch_size=64
    )
    
    # 11. 모델 평가
    logger.info("\n모델 평가 중...")
    
    # 예측
    predictions = model.predict(X_test)
    pred_logistics = predictions[0].flatten()
    pred_bottleneck = predictions[1]
    
    # 물류량 예측 평가
    mae = mean_absolute_error(y_test_reg, pred_logistics)
    mse = mean_squared_error(y_test_reg, pred_logistics)
    
    logger.info(f"\n물류량 예측 성능:")
    logger.info(f"  MAE: {mae:.2f}")
    logger.info(f"  MSE: {mse:.2f}")
    logger.info(f"  RMSE: {np.sqrt(mse):.2f}")
    
    # 병목 예측 평가
    pred_bottleneck_classes = np.argmax(pred_bottleneck, axis=1)
    accuracy = accuracy_score(y_test_cls, pred_bottleneck_classes)
    
    logger.info(f"\n병목 위치 예측 성능:")
    logger.info(f"  Accuracy: {accuracy:.2%}")
    logger.info("\n분류 리포트:")
    print(classification_report(y_test_cls, pred_bottleneck_classes, 
                              target_names=['정상', 'M14A-M10A', 'M14A-M14B', 'M14A-M16']))
    
    # 12. 모델 및 스케일러 저장
    logger.info("\n모델 및 스케일러 저장 중...")
    
    # 디렉토리 생성
    os.makedirs('model', exist_ok=True)
    os.makedirs('scaler', exist_ok=True)
    
    # 모델 저장
    model.save('model/cnn_lstm_multitask_final.keras')
    logger.info("모델 저장 완료: model/cnn_lstm_multitask_final.keras")
    
    # 스케일러 저장
    joblib.dump(scaler, 'scaler/multitask_scaler.pkl')
    logger.info("스케일러 저장 완료: scaler/multitask_scaler.pkl")
    
    # 13. 학습 곡선 시각화
    plot_training_history(history)
    
    # 14. 예측 결과 시각화
    plot_predictions(y_test_reg, pred_logistics, y_test_cls, pred_bottleneck_classes)
    
    logger.info("\n" + "="*60)
    logger.info("학습 완료!")
    logger.info("="*60)
    
    return model, scaler, history

# ===================================
# 7. 시각화 함수
# ===================================

def plot_training_history(history):
    """학습 이력 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 전체 손실
    axes[0, 0].plot(history.history['loss'], label='Train Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 물류량 예측 손실
    axes[0, 1].plot(history.history['logistics_output_loss'], label='Train')
    axes[0, 1].plot(history.history['val_logistics_output_loss'], label='Val')
    axes[0, 1].set_title('Logistics Prediction Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MSE')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 병목 예측 손실
    axes[1, 0].plot(history.history['bottleneck_output_loss'], label='Train')
    axes[1, 0].plot(history.history['val_bottleneck_output_loss'], label='Val')
    axes[1, 0].set_title('Bottleneck Prediction Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Cross Entropy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 병목 예측 정확도
    axes[1, 1].plot(history.history['bottleneck_output_accuracy'], label='Train')
    axes[1, 1].plot(history.history['val_bottleneck_output_accuracy'], label='Val')
    axes[1, 1].set_title('Bottleneck Prediction Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history_multitask.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_predictions(y_true_reg, y_pred_reg, y_true_cls, y_pred_cls):
    """예측 결과 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 물류량 예측 비교
    sample_size = min(200, len(y_true_reg))
    axes[0, 0].plot(y_true_reg[:sample_size], label='Actual', color='blue')
    axes[0, 0].plot(y_pred_reg[:sample_size], label='Predicted', color='red', alpha=0.7)
    axes[0, 0].set_title('Logistics Prediction (First 200 samples)')
    axes[0, 0].set_xlabel('Sample')
    axes[0, 0].set_ylabel('TOTALCNT')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 물류량 예측 산점도
    axes[0, 1].scatter(y_true_reg, y_pred_reg, alpha=0.5)
    axes[0, 1].plot([y_true_reg.min(), y_true_reg.max()], 
                    [y_true_reg.min(), y_true_reg.max()], 
                    'r--', lw=2)
    axes[0, 1].set_title('Logistics Prediction Scatter')
    axes[0, 1].set_xlabel('Actual')
    axes[0, 1].set_ylabel('Predicted')
    axes[0, 1].grid(True)
    
    # 병목 예측 혼동 행렬
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true_cls, y_pred_cls)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
    axes[1, 0].set_title('Bottleneck Prediction Confusion Matrix')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    axes[1, 0].set_xticklabels(['Normal', 'M14A-M10A', 'M14A-M14B', 'M14A-M16'])
    axes[1, 0].set_yticklabels(['Normal', 'M14A-M10A', 'M14A-M14B', 'M14A-M16'])
    
    # 병목 발생 시점 표시
    bottleneck_points = np.where(y_true_cls > 0)[0]
    if len(bottleneck_points) > 0:
        axes[1, 1].scatter(bottleneck_points[:100], 
                          y_true_reg[bottleneck_points[:100]], 
                          color='red', s=50, label='Actual Bottleneck')
    
    predicted_bottleneck = np.where(y_pred_cls > 0)[0]
    if len(predicted_bottleneck) > 0:
        axes[1, 1].scatter(predicted_bottleneck[:100], 
                          y_pred_reg[predicted_bottleneck[:100]], 
                          color='orange', s=30, alpha=0.5, label='Predicted Bottleneck')
    
    axes[1, 1].set_title('Bottleneck Detection')
    axes[1, 1].set_xlabel('Sample')
    axes[1, 1].set_ylabel('TOTALCNT')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('prediction_results_multitask.png', dpi=300, bbox_inches='tight')
    plt.close()

# ===================================
# 8. 독립 실행 가능한 예측 스크립트
# ===================================

def load_and_predict():
    """저장된 모델을 로드하여 예측 수행"""
    print("모델 및 스케일러 로딩 중...")
    
    try:
        # 모델 로드
        model = load_model('model/cnn_lstm_multitask_final.keras')
        scaler = joblib.load('scaler/multitask_scaler.pkl')
        
        print("✓ 모델 로드 완료")
        
        # 예측할 데이터 로드 (예시: 최신 30분 데이터)
        test_data_path = 'data/0730to31.csv'  # 테스트용 데이터
        
        if not os.path.exists(test_data_path):
            print(f"⚠️  테스트 데이터를 찾을 수 없습니다: {test_data_path}")
            print("다른 데이터 경로를 시도합니다...")
            
            # 대체 경로들
            alternative_paths = [
                'data/TO.CSV',
                'data/20240201_TO_202507281705.csv',
                'data/test_data.csv'
            ]
            
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    test_data_path = alt_path
                    print(f"✓ 대체 데이터 사용: {test_data_path}")
                    break
            else:
                print("❌ 사용 가능한 데이터 파일을 찾을 수 없습니다.")
                return None
        
        test_data = pd.read_csv(test_data_path)
        
        # 시간 변환
        test_data['CURRTIME'] = pd.to_datetime(test_data['CURRTIME'], format='%Y%m%d%H%M')
        test_data.set_index('CURRTIME', inplace=True)
        
        # 최소 30개 데이터 확인
        if len(test_data) < 30:
            print(f"⚠️  데이터가 충분하지 않습니다. 최소 30개 필요 (현재: {len(test_data)}개)")
            return None
        
        # 최근 30분 데이터 추출
        recent_30min = test_data.tail(30)
        
        # 예측 수행
        result = predict_realtime(model, scaler, recent_30min)
        
        # 결과 출력
        print_prediction_result(result, recent_30min.index[-1])
        
        return result
        
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# ===================================
# 9. 배치 예측 함수
# ===================================

def batch_predict(model, scaler, data_path, save_results=True):
    """전체 데이터에 대한 배치 예측"""
    print("배치 예측 시작...")
    
    # 데이터 로드
    data = load_and_preprocess_data(data_path)
    data = create_features(data)
    
    # 예측 결과 저장
    predictions_list = []
    
    # 30분 슬라이딩 윈도우로 예측
    for i in range(30, len(data)):
        window_data = data.iloc[i-30:i]
        
        try:
            result = predict_realtime(model, scaler, window_data)
            
            predictions_list.append({
                'time': data.index[i],
                'predict_time': data.index[i] + timedelta(minutes=10),
                'actual_totalcnt': data.iloc[i]['TOTALCNT'],
                'predicted_totalcnt': result['logistics_prediction'],
                'bottleneck_location': result['bottleneck_location'],
                'bottleneck_probability': result['bottleneck_probability']
            })
            
        except Exception as e:
            print(f"예측 오류 (시간: {data.index[i]}): {str(e)}")
            continue
    
    # DataFrame으로 변환
    predictions_df = pd.DataFrame(predictions_list)
    
    if save_results:
        # 결과 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'predictions_batch_{timestamp}.csv'
        predictions_df.to_csv(output_path, index=False)
        print(f"예측 결과 저장: {output_path}")
    
    # 성능 평가 (실제값이 있는 경우)
    evaluate_batch_predictions(predictions_df)
    
    return predictions_df

def evaluate_batch_predictions(predictions_df):
    """배치 예측 결과 평가"""
    # 실제값과 비교 가능한 데이터만 필터링
    valid_predictions = predictions_df.dropna()
    
    if len(valid_predictions) > 0:
        # 물류량 예측 평가
        mae = np.mean(np.abs(valid_predictions['actual_totalcnt'] - valid_predictions['predicted_totalcnt']))
        mape = np.mean(np.abs((valid_predictions['actual_totalcnt'] - valid_predictions['predicted_totalcnt']) / valid_predictions['actual_totalcnt'])) * 100
        
        print("\n배치 예측 성능 평가:")
        print(f"- MAE: {mae:.2f}")
        print(f"- MAPE: {mape:.2f}%")
        
        # 병목 예측 정확도
        bottleneck_accuracy = (valid_predictions['bottleneck_location'] != '정상').sum() / len(valid_predictions) * 100
        print(f"- 병목 감지율: {bottleneck_accuracy:.1f}%")

# ===================================
# 10. 실시간 모니터링 시뮬레이션
# ===================================

def realtime_monitoring_simulation(model, scaler, data_path, interval_seconds=60):
    """실시간 모니터링 시뮬레이션"""
    import time
    
    print("실시간 모니터링 시작 (Ctrl+C로 종료)")
    print("="*60)
    
    # 데이터 로드
    data = load_and_preprocess_data(data_path)
    data = create_features(data)
    
    # 시뮬레이션 시작 인덱스
    current_idx = 30
    
    try:
        while current_idx < len(data):
            # 최근 30분 데이터
            window_data = data.iloc[current_idx-30:current_idx]
            
            # 예측 수행
            result = predict_realtime(model, scaler, window_data)
            
            # 결과 출력
            current_time = data.index[current_idx]
            print_prediction_result(result, current_time)
            
            # 경고 알림
            if result['bottleneck_location'] != '정상' and result['bottleneck_probability'] > 70:
                print("\n⚠️  경고! 병목 발생 예상!")
                print(f"   위치: {result['bottleneck_location']}")
                print(f"   확률: {result['bottleneck_probability']:.1f}%")
                print(f"   예상 물류량: {result['logistics_prediction']:.0f}")
                print("   → 대응 조치 필요\n")
            
            # 다음 시점으로 이동
            current_idx += 1
            
            # 대기 (실제 운영 시에는 실시간 데이터 수신 대기)
            time.sleep(interval_seconds)
            
    except KeyboardInterrupt:
        print("\n모니터링 종료")

# ===================================
# 11. 실행
# ===================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == 'train':
            # 학습 모드
            model, scaler, history = main()
            
        elif mode == 'predict':
            # 예측 모드
            load_and_predict()
            
        elif mode == 'batch':
            # 배치 예측 모드
            if len(sys.argv) > 2:
                data_path = sys.argv[2]
            else:
                data_path = 'data/test_data.csv'
                
            # 모델 로드
            model = load_model('model/cnn_lstm_multitask_final.keras')
            scaler = joblib.load('scaler/multitask_scaler.pkl')
            
            batch_predict(model, scaler, data_path)
            
        elif mode == 'monitor':
            # 실시간 모니터링 모드
            if len(sys.argv) > 2:
                data_path = sys.argv[2]
            else:
                data_path = 'data/test_data.csv'
                
            # 모델 로드
            model = load_model('model/cnn_lstm_multitask_final.keras')
            scaler = joblib.load('scaler/multitask_scaler.pkl')
            
            realtime_monitoring_simulation(model, scaler, data_path, interval_seconds=1)
            
        else:
            print("사용법:")
            print("  python script.py train      # 모델 학습")
            print("  python script.py predict    # 예측 수행")
            print("  python script.py batch [data_path]   # 배치 예측")
            print("  python script.py monitor [data_path]  # 실시간 모니터링")
    else:
        # 기본: 예측 모드
        print("기본 모드: 예측 수행")
        load_and_predict()