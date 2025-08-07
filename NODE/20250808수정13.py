"""
###수정
def main():
    """메인 실행 함수"""
    
    logger.info("="*60)
    logger.info("CNN-LSTM Multi-Task 모델 학습 시작 (실제 데이터)")
    logger.info("="*60)
    
    # 1. 데이터 로드 및 전처리
    data_path = 'data/20240201_TO_202507281705.csv'  # 실제 전체 데이터
    
    # 파일 존재 확인
    if not os.path.exists(data_path):
        logger.error(f"데이터 파일을 찾을 수 없습니다: {data_path}")
        return None, None, None
    
    data = load_and_preprocess_data(data_path)
    
    # 2. 특징 생성
    data = create_features(data)
    
    # 3. 타겟 생성
    data = create_targets(data)
    
    # 4. 특징 선택
    # 스케일링할 특징 (문제가 될 수 있는 비율 특징 제외 가능)
    scale_features_list = [
        'TOTALCNT', 'M14AM10A', 'M10AM14A', 'M14AM14B', 'M14BM14A', 'M14AM16', 'M16M14A',
        'imbalance_M14A_M10A', 'imbalance_M14A_M14B', 'imbalance_M14A_M16',
        'MA_5', 'MA_10', 'MA_30', 'MA_60',
        'STD_5', 'STD_10', 'STD_30',
        'MAX_10', 'MIN_10',
        'load_M14A_out', 'load_M14A_in',
        'ratio_M14A_M10A', 'ratio_M14A_M14B', 'ratio_M14A_M16',
        'change_rate', 'change_rate_5', 'change_rate_10',
        'acceleration'
    ]
    
    # 실제 존재하는 컬럼만 선택
    scale_features_list = [col for col in scale_features_list if col in data.columns]
    logger.info(f"스케일링할 특징 수: {len(scale_features_list)}")
    
    # 5. 스케일링
    data, scaler = scale_features(data, scale_features_list)
    
    # 6. 시퀀스용 특징 선택
    sequence_features = [col for col in data.columns if col.startswith('scaled_')]
    target_features = ['FUTURE_TOTALCNT', 'BOTTLENECK_LOCATION']
    
    # 7. 시퀀스 생성 (60분 시퀀스)
    X, y_regression, y_classification = create_sequences(
        data, 
        sequence_features, 
        target_features,
        seq_length=60  # 1시간 시퀀스
    )
    
    # ==================== 클래스 레이블 재매핑 추가! ====================
    # 클래스 [0, 2, 3]을 [0, 1, 2]로 변경
    unique_classes = np.unique(y_classification)
    class_mapping = {old_class: new_class for new_class, old_class in enumerate(unique_classes)}
    y_classification_mapped = np.array([class_mapping[cls] for cls in y_classification])
    
    logger.info(f"원본 클래스: {unique_classes}")
    logger.info(f"클래스 매핑: {class_mapping}")
    logger.info(f"시퀀스 shape - X: {X.shape}, y_reg: {y_regression.shape}, y_cls: {y_classification_mapped.shape}")
    # ====================================================================
    
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
    
    # 매핑된 클래스 사용
    y_train_cls = y_classification_mapped[:train_size]
    y_val_cls = y_classification_mapped[train_size:train_size+val_size]
    y_test_cls = y_classification_mapped[train_size+val_size:]
    
    logger.info(f"\n데이터 분할:")
    logger.info(f"  - Train: {len(X_train)} samples")
    logger.info(f"  - Validation: {len(X_val)} samples")
    logger.info(f"  - Test: {len(X_test)} samples")
    
    # 9. 모델 생성
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # 실제 클래스 개수 확인 (매핑된 클래스 기준)
    unique_classes_mapped = np.unique(y_classification_mapped)
    num_classes = len(unique_classes_mapped)
    logger.info(f"매핑된 병목 클래스: {unique_classes_mapped}, 총 {num_classes}개")
    
    model = build_cnn_lstm_multitask_model(input_shape, num_classes)
    model.summary()
    
    # 10. 모델 학습
    logger.info("\n모델 학습 시작...")
    history = train_model(
        model, 
        X_train, y_train_reg, y_train_cls,
        X_val, y_val_reg, y_val_cls,
        epochs=200,  # 실제 데이터용
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
    
    # target_names 설정 (원본 클래스 기준)
    if set(unique_classes) == {0, 2, 3}:
        target_names = ['정상', 'M14A-M14B', 'M14A-M16']
    elif num_classes == 4:
        target_names = ['정상', 'M14A-M10A', 'M14A-M14B', 'M14A-M16']
    else:
        target_names = [f'Class_{i}' for i in range(num_classes)]
    
    print(classification_report(y_test_cls, pred_bottleneck_classes, 
                              target_names=target_names))
    
    # 12. 모델 및 스케일러 저장
    logger.info("\n모델 및 스케일러 저장 중...")
    
    # 디렉토리 생성
    os.makedirs('model', exist_ok=True)
    os.makedirs('scaler', exist_ok=True)
    os.makedirs('config', exist_ok=True)
    
    # 모델 저장
    model.save('model/cnn_lstm_multitask_final.keras')
    logger.info("모델 저장 완료: model/cnn_lstm_multitask_final.keras")
    
    # 스케일러 저장
    joblib.dump(scaler, 'scaler/multitask_scaler.pkl')
    logger.info("스케일러 저장 완료: scaler/multitask_scaler.pkl")
    
    # 클래스 매핑 정보 저장
    import json
    with open('config/class_mapping.json', 'w') as f:
        json.dump(class_mapping, f, indent=4)
    logger.info("클래스 매핑 저장 완료: config/class_mapping.json")
    
    # 13. 학습 곡선 시각화
    plot_training_history(history)
    
    # 14. 예측 결과 시각화
    plot_predictions(y_test_reg, pred_logistics, y_test_cls, pred_bottleneck_classes, 
                    num_classes, class_mapping)
    
    logger.info("\n" + "="*60)
    logger.info("학습 완료!")
    logger.info("="*60)
    
    return model, scaler, history
###매인수정끝

###
def plot_predictions(y_true_reg, y_pred_reg, y_true_cls, y_pred_cls, num_classes, class_mapping=None):
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
    
    # 동적 라벨 설정 (클래스 매핑 고려)
    if class_mapping:
        # 원본 클래스로 역매핑
        reverse_mapping = {v: k for k, v in class_mapping.items()}
        original_classes = sorted([reverse_mapping.get(i, i) for i in range(num_classes)])
        
        if set(original_classes) == {0, 2, 3}:
            labels = ['Normal', 'M14A-M14B', 'M14A-M16']
        elif set(original_classes) == {0, 1, 2, 3}:
            labels = ['Normal', 'M14A-M10A', 'M14A-M14B', 'M14A-M16']
        else:
            labels = [f'Class_{i}' for i in range(num_classes)]
    else:
        if num_classes == 3:
            labels = ['Normal', 'Route_1', 'Route_2']
        elif num_classes == 4:
            labels = ['Normal', 'M14A-M10A', 'M14A-M14B', 'M14A-M16']
        else:
            labels = [f'Class_{i}' for i in range(num_classes)]
    
    axes[1, 0].set_xticklabels(labels)
    axes[1, 0].set_yticklabels(labels)
    
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
###수정완료


CNN-LSTM Multi-Task 기반 반도체 물류 예측 모델 - 오류 수정 완료 버전
==================================================================
실제 전체 데이터를 사용하여 모델을 학습시킵니다.
모든 오류가 수정된 버전입니다.

사용 데이터: data/20240201_TO_202507281705.csv
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
# 2. 데이터 전처리 함수
# ===================================

def load_and_preprocess_data(data_path):
    """데이터 로드 및 전처리"""
    logger.info("데이터 로딩 중...")
    
    # 데이터 로드
    Full_Data = pd.read_csv(data_path)
    logger.info(f"원본 데이터 shape: {Full_Data.shape}")
    
    # 시간 컬럼 변환
    Full_Data['CURRTIME'] = pd.to_datetime(Full_Data['CURRTIME'], format='%Y%m%d%H%M')
    Full_Data['TIME'] = pd.to_datetime(Full_Data['TIME'], format='%Y%m%d%H%M')
    
    # SUM 컬럼 제거
    columns_to_drop = [col for col in Full_Data.columns if 'SUM' in col]
    Full_Data = Full_Data.drop(columns=columns_to_drop)
    
    # 특정 날짜 범위만 사용 (옵션)
    start_date = pd.to_datetime('2024-02-01 00:00:00')
    end_date = pd.to_datetime('2025-07-27 23:59:59')
    Full_Data = Full_Data[(Full_Data['TIME'] >= start_date) & (Full_Data['TIME'] <= end_date)]
    
    # 인덱스를 시간으로 설정
    Full_Data.set_index('CURRTIME', inplace=True)
    
    # 이상치 처리 (PM 기간 고려)
    PM_start_date = pd.to_datetime('2024-10-23 00:00:00')
    PM_end_date = pd.to_datetime('2024-10-23 23:59:59')
    
    within_PM = Full_Data[(Full_Data['TIME'] >= PM_start_date) & (Full_Data['TIME'] <= PM_end_date)]
    outside_PM = Full_Data[(Full_Data['TIME'] < PM_start_date) | (Full_Data['TIME'] > PM_end_date)]
    
    # PM 기간 외 데이터는 정상 범위만 사용
    outside_PM_filtered = outside_PM[(outside_PM['TOTALCNT'] >= 800) & (outside_PM['TOTALCNT'] <= 2500)]
    
    # 데이터 합치기
    Full_Data = pd.concat([within_PM, outside_PM_filtered])
    Full_Data = Full_Data.sort_index()
    
    logger.info(f"전처리 후 데이터 shape: {Full_Data.shape}")
    
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
    features_data['month'] = features_data.index.month
    features_data['day'] = features_data.index.day
    
    # 피크 시간대
    features_data['is_peak_hour'] = features_data.index.hour.isin([8, 9, 14, 15, 16, 17]).astype(int)
    
    # 팹 간 불균형 지표
    features_data['imbalance_M14A_M10A'] = features_data['M14AM10A'] - features_data['M10AM14A']
    features_data['imbalance_M14A_M14B'] = features_data['M14AM14B'] - features_data['M14BM14A']
    features_data['imbalance_M14A_M16'] = features_data['M14AM16'] - features_data['M16M14A']
    
    # 이동 평균 (다양한 윈도우)
    for window in [5, 10, 30, 60]:
        features_data[f'MA_{window}'] = features_data['TOTALCNT'].rolling(window=window, min_periods=1).mean()
    
    # 표준편차 (변동성)
    for window in [5, 10, 30]:
        features_data[f'STD_{window}'] = features_data['TOTALCNT'].rolling(window=window, min_periods=1).std()
    
    # 최대/최소값
    features_data['MAX_10'] = features_data['TOTALCNT'].rolling(window=10, min_periods=1).max()
    features_data['MIN_10'] = features_data['TOTALCNT'].rolling(window=10, min_periods=1).min()
    
    # 팹별 부하율 (0으로 나누기 방지)
    total_safe = features_data['TOTALCNT'].replace(0, 1)  # 0을 1로 치환
    features_data['load_M14A_out'] = (features_data['M14AM10A'] + features_data['M14AM14B'] + 
                                      features_data['M14AM16']) / total_safe
    features_data['load_M14A_in'] = (features_data['M10AM14A'] + features_data['M14BM14A'] + 
                                     features_data['M16M14A']) / total_safe
    
    # 경로별 비율 (0으로 나누기 방지)
    features_data['ratio_M14A_M10A'] = (features_data['M14AM10A'] + features_data['M10AM14A']) / total_safe
    features_data['ratio_M14A_M14B'] = (features_data['M14AM14B'] + features_data['M14BM14A']) / total_safe
    features_data['ratio_M14A_M16'] = (features_data['M14AM16'] + features_data['M16M14A']) / total_safe
    
    # 변화율
    features_data['change_rate'] = features_data['TOTALCNT'].pct_change()
    features_data['change_rate_5'] = features_data['TOTALCNT'].pct_change(5)
    features_data['change_rate_10'] = features_data['TOTALCNT'].pct_change(10)
    
    # 가속도 (변화율의 변화)
    features_data['acceleration'] = features_data['change_rate'].diff()
    
    # 결측값 처리
    features_data = features_data.fillna(method='ffill').fillna(0)
    
    # 무한대 값 처리
    features_data = features_data.replace([np.inf, -np.inf], 0)
    
    # 이상치 클리핑 (극단적인 값 제한)
    numeric_columns = features_data.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if col not in ['TIME', 'CURRTIME']:  # 시간 컬럼 제외
            # 99.9 퍼센타일로 클리핑
            upper_limit = features_data[col].quantile(0.999)
            lower_limit = features_data[col].quantile(0.001)
            features_data[col] = features_data[col].clip(lower=lower_limit, upper=upper_limit)
    
    logger.info(f"특징 생성 완료 - shape: {features_data.shape}")
    logger.info(f"무한대 값 체크: {np.isinf(features_data.select_dtypes(include=[np.number])).any().any()}")
    logger.info(f"NaN 값 체크: {features_data.isnull().any().any()}")
    
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
    # 동적 임계값 설정
    thresholds = {
        'total': data['TOTALCNT'].quantile(0.90),  # 상위 10%
        'm14a_m10a': np.percentile(data['M14AM10A'] + data['M10AM14A'], 90),
        'm14a_m14b': np.percentile(data['M14AM14B'] + data['M14BM14A'], 90),
        'm14a_m16': np.percentile(data['M14AM16'] + data['M16M14A'], 90)
    }
    
    logger.info(f"병목 임계값 - 전체: {thresholds['total']:.0f}")
    
    data['BOTTLENECK_LOCATION'] = 0  # 0: 병목 없음
    
    for i in data.index:
        future_time = i + pd.Timedelta(minutes=future_minutes)
        if (future_time <= data.index.max()) & (future_time in data.index):
            future_total = data.loc[future_time, 'TOTALCNT']
            
            if future_total > thresholds['total']:
                # 어느 경로가 가장 혼잡한지 확인
                route_loads = {
                    1: data.loc[future_time, 'M14AM10A'] + data.loc[future_time, 'M10AM14A'],
                    2: data.loc[future_time, 'M14AM14B'] + data.loc[future_time, 'M14BM14A'],
                    3: data.loc[future_time, 'M14AM16'] + data.loc[future_time, 'M16M14A']
                }
                
                # 가장 혼잡한 경로를 병목으로 지정
                max_route = max(route_loads.items(), key=lambda x: x[1])
                if max_route[0] == 1 and max_route[1] > thresholds['m14a_m10a']:
                    data.loc[i, 'BOTTLENECK_LOCATION'] = 1
                elif max_route[0] == 2 and max_route[1] > thresholds['m14a_m14b']:
                    data.loc[i, 'BOTTLENECK_LOCATION'] = 2
                elif max_route[0] == 3 and max_route[1] > thresholds['m14a_m16']:
                    data.loc[i, 'BOTTLENECK_LOCATION'] = 3
    
    # NA 제거
    data = data.dropna(subset=['FUTURE_TOTALCNT'])
    
    logger.info(f"타겟 생성 완료 - 병목 분포: {data['BOTTLENECK_LOCATION'].value_counts()}")
    
    return data

def scale_features(data, feature_columns):
    """특징 스케일링"""
    scaler = StandardScaler()
    
    # 스케일링할 컬럼 선택
    scale_columns = [col for col in feature_columns if col in data.columns]
    
    # 스케일링 전 데이터 검증
    scale_data = data[scale_columns].copy()
    
    # 무한대 값 체크 및 처리
    if np.isinf(scale_data.values).any():
        logger.warning("무한대 값 발견! 처리 중...")
        scale_data = scale_data.replace([np.inf, -np.inf], np.nan)
        scale_data = scale_data.fillna(scale_data.mean())
    
    # NaN 값 체크 및 처리
    if scale_data.isnull().any().any():
        logger.warning("NaN 값 발견! 처리 중...")
        scale_data = scale_data.fillna(scale_data.mean())
    
    # 스케일링
    scaled_data = scaler.fit_transform(scale_data)
    
    # 스케일링된 데이터프레임 생성
    scaled_df = pd.DataFrame(
        scaled_data, 
        columns=[f'scaled_{col}' for col in scale_columns],
        index=data.index
    )
    
    # 원본 데이터와 병합
    result = pd.concat([data, scaled_df], axis=1)
    
    return result, scaler

def create_sequences(data, feature_cols, target_cols, seq_length=60):
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
# 3. CNN-LSTM Multi-Task 모델 (강화된 버전)
# ===================================

def build_cnn_lstm_multitask_model(input_shape, num_classes=4):
    """CNN-LSTM Multi-Task 모델 구축 (실제 데이터용 강화 버전)"""
    
    # 입력 레이어
    inputs = Input(shape=input_shape, name='input')
    
    # === 강화된 CNN 파트 ===
    # 첫 번째 Conv1D 블록
    x = Conv1D(filters=128, kernel_size=5, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    
    # 두 번째 Conv1D 블록
    x = Conv1D(filters=256, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    
    # 세 번째 Conv1D 블록
    x = Conv1D(filters=256, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    
    # 네 번째 Conv1D 블록
    x = Conv1D(filters=128, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Max Pooling
    x = MaxPooling1D(pool_size=2)(x)
    
    # === 강화된 LSTM 파트 ===
    # 첫 번째 Bidirectional LSTM
    x = Bidirectional(LSTM(256, return_sequences=True))(x)
    x = Dropout(0.4)(x)
    
    # 두 번째 Bidirectional LSTM
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.4)(x)
    
    # 세 번째 Bidirectional LSTM
    x = Bidirectional(LSTM(64, return_sequences=False))(x)
    x = Dropout(0.4)(x)
    
    # === 공유 Dense 레이어 ===
    shared = Dense(256, activation='relu', name='shared_layer')(x)
    shared = BatchNormalization()(shared)
    shared = Dropout(0.4)(shared)
    
    shared = Dense(128, activation='relu')(shared)
    shared = BatchNormalization()(shared)
    shared = Dropout(0.3)(shared)
    
    # === Multi-Task 출력 ===
    # Task 1: 물류량 예측 (회귀)
    logistics_branch = Dense(128, activation='relu')(shared)
    logistics_branch = Dropout(0.3)(logistics_branch)
    logistics_branch = Dense(64, activation='relu')(logistics_branch)
    logistics_output = Dense(1, name='logistics_output')(logistics_branch)
    
    # Task 2: 병목 위치 예측 (분류)
    bottleneck_branch = Dense(128, activation='relu')(shared)
    bottleneck_branch = Dropout(0.3)(bottleneck_branch)
    bottleneck_branch = Dense(64, activation='relu')(bottleneck_branch)
    bottleneck_output = Dense(num_classes, activation='softmax', name='bottleneck_output')(bottleneck_branch)
    
    # 모델 생성
    model = Model(inputs=inputs, outputs=[logistics_output, bottleneck_output])
    
    return model

# ===================================
# 4. 학습 프로세스
# ===================================

def train_model(model, X_train, y_train_reg, y_train_cls, X_val, y_val_reg, y_val_cls, 
                epochs=200, batch_size=64):
    """모델 학습 (실제 데이터용 설정)"""
    
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
        optimizer=Adam(learning_rate=0.0001),  # 낮은 학습률
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics
    )
    
    # 콜백 설정
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=30,  # 더 긴 patience
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=15,
            min_lr=1e-7,
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
# 5. 메인 실행 함수
# ===================================

def main():
    """메인 실행 함수"""
    
    logger.info("="*60)
    logger.info("CNN-LSTM Multi-Task 모델 학습 시작 (실제 데이터)")
    logger.info("="*60)
    
    # 1. 데이터 로드 및 전처리
    data_path = 'data/20240201_TO_202507281705.csv'  # 실제 전체 데이터
    
    # 파일 존재 확인
    if not os.path.exists(data_path):
        logger.error(f"데이터 파일을 찾을 수 없습니다: {data_path}")
        return None, None, None
    
    data = load_and_preprocess_data(data_path)
    
    # 2. 특징 생성
    data = create_features(data)
    
    # 3. 타겟 생성
    data = create_targets(data)
    
    # 4. 특징 선택
    # 스케일링할 특징 (문제가 될 수 있는 비율 특징 제외 가능)
    scale_features_list = [
        'TOTALCNT', 'M14AM10A', 'M10AM14A', 'M14AM14B', 'M14BM14A', 'M14AM16', 'M16M14A',
        'imbalance_M14A_M10A', 'imbalance_M14A_M14B', 'imbalance_M14A_M16',
        'MA_5', 'MA_10', 'MA_30', 'MA_60',
        'STD_5', 'STD_10', 'STD_30',
        'MAX_10', 'MIN_10',
        'load_M14A_out', 'load_M14A_in',
        'ratio_M14A_M10A', 'ratio_M14A_M14B', 'ratio_M14A_M16',
        'change_rate', 'change_rate_5', 'change_rate_10',
        'acceleration'
    ]
    
    # 실제 존재하는 컬럼만 선택
    scale_features_list = [col for col in scale_features_list if col in data.columns]
    logger.info(f"스케일링할 특징 수: {len(scale_features_list)}")
    
    # 5. 스케일링
    data, scaler = scale_features(data, scale_features_list)
    
    # 6. 시퀀스용 특징 선택
    sequence_features = [col for col in data.columns if col.startswith('scaled_')]
    target_features = ['FUTURE_TOTALCNT', 'BOTTLENECK_LOCATION']
    
    # 7. 시퀀스 생성 (60분 시퀀스)
    X, y_regression, y_classification = create_sequences(
        data, 
        sequence_features, 
        target_features,
        seq_length=60  # 1시간 시퀀스
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
    
    logger.info(f"\n데이터 분할:")
    logger.info(f"  - Train: {len(X_train)} samples")
    logger.info(f"  - Validation: {len(X_val)} samples")
    logger.info(f"  - Test: {len(X_test)} samples")
    
    # 9. 모델 생성
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # 실제 클래스 개수 확인
    unique_classes = np.unique(y_classification)
    num_classes = len(unique_classes)
    logger.info(f"병목 클래스: {unique_classes}, 총 {num_classes}개")
    
    model = build_cnn_lstm_multitask_model(input_shape, num_classes)
    model.summary()
    
    # 10. 모델 학습
    logger.info("\n모델 학습 시작...")
    history = train_model(
        model, 
        X_train, y_train_reg, y_train_cls,
        X_val, y_val_reg, y_val_cls,
        epochs=200,  # 실제 데이터용
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
    
    # target_names 동적 생성
    if num_classes == 3:
        target_names = ['정상', 'M14A-M10A', 'M14A-M14B']
    elif num_classes == 4:
        target_names = ['정상', 'M14A-M10A', 'M14A-M14B', 'M14A-M16']
    else:
        target_names = [f'Class_{i}' for i in range(num_classes)]
    
    print(classification_report(y_test_cls, pred_bottleneck_classes, 
                              target_names=target_names))
    
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
    plot_predictions(y_test_reg, pred_logistics, y_test_cls, pred_bottleneck_classes, num_classes)
    
    logger.info("\n" + "="*60)
    logger.info("학습 완료!")
    logger.info("="*60)
    
    return model, scaler, history

# ===================================
# 6. 시각화 함수
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

def plot_predictions(y_true_reg, y_pred_reg, y_true_cls, y_pred_cls, num_classes):
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
    
    # 동적 라벨 설정
    if num_classes == 3:
        labels = ['Normal', 'M14A-M10A', 'M14A-M14B']
    elif num_classes == 4:
        labels = ['Normal', 'M14A-M10A', 'M14A-M14B', 'M14A-M16']
    else:
        labels = [f'Class_{i}' for i in range(num_classes)]
    
    axes[1, 0].set_xticklabels(labels)
    axes[1, 0].set_yticklabels(labels)
    
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
# 7. 실행
# ===================================

if __name__ == "__main__":
    # 실제 데이터로 학습 실행!
    model, scaler, history = main()
    
    print("\n" + "="*60)
    print("🎉 실제 데이터 학습 완료!")
    print("="*60)
    print("\n생성된 파일:")
    print("  - model/cnn_lstm_multitask_final.keras")
    print("  - scaler/multitask_scaler.pkl")
    print("  - training_history_multitask.png")
    print("  - prediction_results_multitask.png")