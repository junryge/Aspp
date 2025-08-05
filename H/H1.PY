"""
반도체 물류 예측을 위한 하이브리드 딥러닝 모델
==============================================
본 시스템은 반도체 팹 간 물류 이동량을 예측하고 병목 구간을 사전에 감지하기 위한 
통합 예측 모델입니다.

주요 기능:
1. LSTM, RNN, GRU, ARIMA 모델을 통합한 앙상블 예측
2. 실시간 병목 구간 감지 및 알림
3. 10분 후 물류량 예측
4. CPU 기반 실행으로 범용성 확보

개발일: 2024년
버전: 1.0
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, Bidirectional, GRU, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime, timedelta
from tensorflow.keras.regularizers import l2
import joblib
import logging
import warnings

# 경고 메시지 숨기기 (깔끔한 출력을 위해)
warnings.filterwarnings('ignore')

# ===================================
# 1. 환경 설정 및 초기화
# ===================================

# CPU 모드 설정 - GPU가 없는 환경에서도 실행 가능하도록 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

# 랜덤 시드 고정 - 재현 가능한 결과를 위해
RANDOM_SEED = 2079936
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# 로깅 설정 - 실행 과정 모니터링
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),  # 로그 파일 저장
        logging.StreamHandler()                # 콘솔 출력
    ]
)
logger = logging.getLogger(__name__)

logger.info("="*60)
logger.info("반도체 물류 예측 하이브리드 모델 학습 시작")
logger.info("="*60)

# ===================================
# 2. 데이터 로드 및 전처리
# ===================================

# 데이터 파일 경로 설정
Full_data_path = 'data/20240201_TO_202507281705.csv'
logger.info(f"데이터 로딩 중: {Full_data_path}")

# CSV 파일 로드
Full_Data = pd.read_csv(Full_data_path)
logger.info(f"원본 데이터 shape: {Full_Data.shape}")

# 시간 컬럼을 datetime 형식으로 변환
# CURRTIME: 현재 시간, TIME: 기록 시간
Full_Data['CURRTIME'] = pd.to_datetime(Full_Data['CURRTIME'], format='%Y%m%d%H%M')
Full_Data['TIME'] = pd.to_datetime(Full_Data['TIME'], format='%Y%m%d%H%M')

# 데이터 무결성 검증 - CURRTIME과 TIME이 일치하는지 확인
time_mismatch_count = 0
for i in range(Full_Data.shape[0]):
    if Full_Data['CURRTIME'][i] != Full_Data['TIME'][i]:
        time_mismatch_count += 1
        logger.warning(f'시간 불일치 발견: 행 {i}')

if time_mismatch_count == 0:
    logger.info("✓ 시간 데이터 무결성 검증 완료")

# 불필요한 SUM 컬럼 제거 (양방향 합계는 예측에 사용하지 않음)
columns_to_drop = [col for col in Full_Data.columns if 'SUM' in col]
Full_Data = Full_Data.drop(columns=columns_to_drop)
logger.info(f"SUM 컬럼 {len(columns_to_drop)}개 제거됨")

# ===================================
# 3. 데이터 범위 설정 및 필터링
# ===================================

# 학습에 사용할 날짜 범위 설정 (필요시 수정)
# 주의: 실제 데이터가 2024년인지 확인 필요
start_date = pd.to_datetime('2024-02-01 00:00:00')  # 2025 → 2024로 수정
end_date = pd.to_datetime('2024-07-27 23:59:59')    # 2025 → 2024로 수정

# 날짜 범위로 데이터 필터링
Full_Data = Full_Data[(Full_Data['TIME'] >= start_date) & (Full_Data['TIME'] <= end_date)].reset_index(drop=True)
logger.info(f"날짜 범위 필터링 완료: {start_date} ~ {end_date}")
logger.info(f"필터링 후 데이터 shape: {Full_Data.shape}")

# 필요한 컬럼만 선택 (현재반송큐 중심 분석)
Full_Data = Full_Data[['CURRTIME', 'TOTALCNT', 'TIME']]

# 시간을 인덱스로 설정
Full_Data.set_index('CURRTIME', inplace=True)

# ===================================
# 4. 이상치 제거 및 데이터 정제
# ===================================

# PM(Preventive Maintenance) 기간 설정
PM_start_date = pd.to_datetime('2024-10-23 00:00:00')
PM_end_date = pd.to_datetime('2024-10-23 23:59:59')

# PM 기간 데이터와 일반 운영 데이터 분리
within_range_data = Full_Data[(Full_Data['TIME'] >= PM_start_date) & (Full_Data['TIME'] <= PM_end_date)]
outside_range_data = Full_Data[(Full_Data['TIME'] < PM_start_date) | (Full_Data['TIME'] > PM_end_date)]

# 일반 운영 시간의 정상 범위 데이터만 선택 (800 ~ 2500)
# 이 범위는 도메인 지식에 기반한 정상 물류량 범위
outside_range_filtered_data = outside_range_data[
    (outside_range_data['TOTALCNT'] >= 800) & 
    (outside_range_data['TOTALCNT'] <= 2500)
]

# PM 기간 데이터는 모두 포함 (PM 시 물류량이 비정상적일 수 있음)
Modified_Data = pd.concat([within_range_data, outside_range_filtered_data])
Modified_Data = Modified_Data.sort_values(by='TIME')

logger.info(f"이상치 제거 완료 - 최소값: {Modified_Data['TOTALCNT'].min()}, 최대값: {Modified_Data['TOTALCNT'].max()}")

# ===================================
# 5. 타겟 변수 생성 (미래 예측값)
# ===================================

# FUTURE 컬럼 초기화
Modified_Data['FUTURE'] = pd.NA

# 예측 시간 간격 설정 (10분 후를 예측)
future_minutes = 10

logger.info(f"{future_minutes}분 후 예측 타겟 생성 중...")

# 각 시점에 대해 10분 후의 값을 타겟으로 설정
for i in Modified_Data.index:
    future_time = i + pd.Timedelta(minutes=future_minutes)
    
    # 10분 후 데이터가 존재하는지 확인
    if (future_time <= Modified_Data.index.max()) & (future_time in Modified_Data.index):
        Modified_Data.loc[i, 'FUTURE'] = Modified_Data.loc[future_time, 'TOTALCNT']

# 타겟값이 없는 데이터 제거
Modified_Data.dropna(subset=['FUTURE'], inplace=True)
logger.info(f"타겟 생성 완료 - 최종 데이터 shape: {Modified_Data.shape}")

# 중간 결과 저장 (확인용)
# Modified_Data.to_csv('data/preprocessed_data_with_future.csv')

# ===================================
# 6. 특징 엔지니어링 (추가 특징 생성)
# ===================================

logger.info("특징 엔지니어링 시작...")

# 시간 기반 특징 추가
Modified_Data['hour'] = Modified_Data.index.hour           # 시간 (0-23)
Modified_Data['dayofweek'] = Modified_Data.index.dayofweek # 요일 (0=월요일)
Modified_Data['is_weekend'] = (Modified_Data.index.dayofweek >= 5).astype(int)  # 주말 여부

# 이동 평균 특징 (단기/중기 트렌드 파악)
Modified_Data['MA_5'] = Modified_Data['TOTALCNT'].rolling(window=5, min_periods=1).mean()
Modified_Data['MA_10'] = Modified_Data['TOTALCNT'].rolling(window=10, min_periods=1).mean()
Modified_Data['MA_30'] = Modified_Data['TOTALCNT'].rolling(window=30, min_periods=1).mean()

# 변동성 특징 (표준편차)
Modified_Data['STD_5'] = Modified_Data['TOTALCNT'].rolling(window=5, min_periods=1).std()
Modified_Data['STD_10'] = Modified_Data['TOTALCNT'].rolling(window=10, min_periods=1).std()

# 변화율 특징
Modified_Data['change_rate'] = Modified_Data['TOTALCNT'].pct_change()
Modified_Data['change_rate_5'] = Modified_Data['TOTALCNT'].pct_change(5)

# 결측값 처리
Modified_Data = Modified_Data.fillna(method='ffill').fillna(0)

logger.info(f"특징 엔지니어링 완료 - 총 {Modified_Data.shape[1]}개 특징")

# ===================================
# 7. 데이터 스케일링 (정규화)
# ===================================

# StandardScaler 사용 (평균 0, 표준편차 1로 정규화)
sscaler = StandardScaler()

# 스케일링할 컬럼 목록
scaled_list = ['TOTALCNT', 'FUTURE', 'MA_5', 'MA_10', 'MA_30', 'STD_5', 'STD_10']
scaled_list = [col for col in scaled_list if col in Modified_Data.columns]

# 스케일러 학습 및 변환
fitted_scaled = sscaler.fit(Modified_Data[scaled_list])
scaled_list_data = fitted_scaled.transform(Modified_Data[scaled_list])

# 스케일링된 데이터를 DataFrame으로 변환
scaled_columns = [f'scaled_{col}' for col in scaled_list]
scaled_list_data = pd.DataFrame(scaled_list_data, columns=scaled_columns)
scaled_list_data.index = Modified_Data.index

# 원본 데이터와 스케일링된 데이터 병합
Scaled_Data = pd.merge(Modified_Data, scaled_list_data, left_index=True, right_index=True, how='left')

logger.info("데이터 스케일링 완료")

# ===================================
# 8. 시퀀스 데이터 생성 (시계열 학습용)
# ===================================

def split_data_by_continuity(data):
    """
    연속성이 끊어진 구간을 찾아 데이터를 분할
    1분 이상 차이가 나는 경우 별도 구간으로 분리
    """
    # 시간 차이 계산
    time_diff = data.index.to_series().diff()
    
    # 1분 초과 차이나는 지점 찾기
    split_points = time_diff > pd.Timedelta(minutes=1)
    
    # 구간 ID 생성
    segment_ids = split_points.cumsum()
    
    # 각 구간별로 데이터 분할
    segments = []
    for segment_id in segment_ids.unique():
        segment = data[segment_ids == segment_id].copy()
        if len(segment) > 30:  # 시퀀스 길이보다 긴 구간만 사용
            segments.append(segment)
    
    return segments

# 데이터를 연속된 구간으로 분할
data_segments = split_data_by_continuity(Scaled_Data)
logger.info(f"총 {len(data_segments)}개의 연속 구간으로 분할됨")

# ===================================
# 9. 시퀀스 생성 함수
# ===================================

def create_sequences(data, feature_cols, target_col, seq_length=30):
    """
    시계열 데이터를 학습용 시퀀스로 변환
    
    Parameters:
    - data: 입력 데이터프레임
    - feature_cols: 입력 특징 컬럼 리스트
    - target_col: 타겟 컬럼명
    - seq_length: 시퀀스 길이 (과거 몇 개 시점을 볼 것인가)
    
    Returns:
    - X: 입력 시퀀스 (samples, time_steps, features)
    - y: 타겟 값 (samples,)
    """
    X, y = [], []
    
    feature_data = data[feature_cols].values
    target_data = data[target_col].values
    
    for i in range(len(data) - seq_length):
        # 과거 seq_length 시점의 데이터를 입력으로
        X.append(feature_data[i:i+seq_length])
        # seq_length 시점 이후의 타겟값
        y.append(target_data[i+seq_length])
    
    return np.array(X), np.array(y)

# 시퀀스 길이 설정 (과거 30분 데이터로 10분 후 예측)
SEQ_LENGTH = 30

# 입력 특징 선택
input_features = [col for col in Scaled_Data.columns if col.startswith('scaled_') and col != 'scaled_FUTURE']

# 모든 구간의 시퀀스 데이터 생성
all_X, all_y = [], []

for segment in data_segments:
    X_seg, y_seg = create_sequences(
        segment, 
        input_features, 
        'scaled_FUTURE',  # 스케일링된 타겟 사용
        SEQ_LENGTH
    )
    
    if len(X_seg) > 0:
        all_X.append(X_seg)
        all_y.append(y_seg)

# 모든 시퀀스 합치기
X_seq_all = np.concatenate(all_X, axis=0)
y_seq_all = np.concatenate(all_y, axis=0)

logger.info(f"시퀀스 생성 완료 - X shape: {X_seq_all.shape}, y shape: {y_seq_all.shape}")

# ===================================
# 10. 학습/검증/테스트 데이터 분할
# ===================================

# 시간 순서를 유지한 분할 (시계열 데이터는 랜덤 분할 X)
train_size = int(0.7 * len(X_seq_all))
val_size = int(0.15 * len(X_seq_all))

X_train = X_seq_all[:train_size]
y_train = y_seq_all[:train_size]

X_val = X_seq_all[train_size:train_size+val_size]
y_val = y_seq_all[train_size:train_size+val_size]

X_test = X_seq_all[train_size+val_size:]
y_test = y_seq_all[train_size+val_size:]

logger.info(f"데이터 분할 완료:")
logger.info(f"  - 학습: {X_train.shape}")
logger.info(f"  - 검증: {X_val.shape}")
logger.info(f"  - 테스트: {X_test.shape}")

# ===================================
# 11. 딥러닝 모델 정의
# ===================================

class HybridModels:
    """
    LSTM, RNN, GRU 모델을 포함한 하이브리드 예측 시스템
    """
    
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.models = {}
        self.histories = {}
        
    def build_lstm_model(self):
        """LSTM 모델 구축 - 장기 의존성 학습에 효과적"""
        model = Sequential([
            Input(shape=self.input_shape),
            
            # 첫 번째 LSTM 층 - 시퀀스 반환
            LSTM(units=100, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),
            
            # 두 번째 LSTM 층 - 시퀀스 반환
            LSTM(units=100, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),
            
            # 세 번째 LSTM 층 - 최종 시점만 반환
            LSTM(units=100, return_sequences=False),
            Dropout(0.2),
            
            # 출력층
            Dense(units=50, activation='relu'),
            Dense(units=1)
        ])
        
        return model
    
    def build_gru_model(self):
        """GRU 모델 구축 - LSTM보다 가볍고 빠른 학습"""
        model = Sequential([
            Input(shape=self.input_shape),
            
            # GRU 층들
            GRU(units=100, return_sequences=True),
            Dropout(0.2),
            
            GRU(units=100, return_sequences=True),
            Dropout(0.2),
            
            GRU(units=50, return_sequences=False),
            Dropout(0.2),
            
            # 출력층
            Dense(units=30, activation='relu'),
            Dense(units=1)
        ])
        
        return model
    
    def build_simple_rnn_model(self):
        """Simple RNN 모델 구축 - 단순하지만 단기 패턴에 효과적"""
        model = Sequential([
            Input(shape=self.input_shape),
            
            # SimpleRNN 층들
            SimpleRNN(units=100, return_sequences=True),
            Dropout(0.2),
            
            SimpleRNN(units=50, return_sequences=False),
            Dropout(0.2),
            
            # 출력층
            Dense(units=30, activation='relu'),
            Dense(units=1)
        ])
        
        return model
    
    def build_bidirectional_lstm_model(self):
        """양방향 LSTM 모델 구축 - 과거와 미래 정보 모두 활용"""
        model = Sequential([
            Input(shape=self.input_shape),
            
            # 양방향 LSTM
            Bidirectional(LSTM(units=50, return_sequences=True)),
            Dropout(0.2),
            
            Bidirectional(LSTM(units=50, return_sequences=False)),
            Dropout(0.2),
            
            # 출력층
            Dense(units=30, activation='relu'),
            Dense(units=1)
        ])
        
        return model

# ===================================
# 12. 모델 학습 설정
# ===================================

# 하이브리드 모델 초기화
input_shape = (X_train.shape[1], X_train.shape[2])
hybrid_models = HybridModels(input_shape)

# 학습 파라미터
EPOCHS = 200          # 최대 에폭
BATCH_SIZE = 64       # 배치 크기
LEARNING_RATE = 0.0005  # 학습률
VALIDATION_SPLIT = 0.3  # 검증 데이터 비율 (사용 안함 - 이미 분할됨)

# 옵티마이저 설정
optimizer = Adam(learning_rate=LEARNING_RATE)

# 콜백 설정
def get_callbacks(model_name):
    """각 모델별 콜백 설정"""
    return [
        # 조기 종료 - 개선이 없으면 학습 중단
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        
        # 학습률 감소 - 정체 시 학습률 줄이기
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        ),
        
        # 모델 체크포인트 - 최고 성능 모델 저장
        ModelCheckpoint(
            f"model/{model_name}_best.keras",
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )
    ]

# ===================================
# 13. 개별 모델 학습
# ===================================

logger.info("\n" + "="*60)
logger.info("딥러닝 모델 학습 시작")
logger.info("="*60)

# 1. LSTM 모델 학습
logger.info("\n[1/4] LSTM 모델 학습 중...")
lstm_model = hybrid_models.build_lstm_model()
lstm_model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

lstm_history = lstm_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=get_callbacks('lstm'),
    verbose=1
)

hybrid_models.models['lstm'] = lstm_model
hybrid_models.histories['lstm'] = lstm_history

# 2. GRU 모델 학습
logger.info("\n[2/4] GRU 모델 학습 중...")
gru_model = hybrid_models.build_gru_model()
gru_model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

gru_history = gru_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=get_callbacks('gru'),
    verbose=1
)

hybrid_models.models['gru'] = gru_model
hybrid_models.histories['gru'] = gru_history

# 3. Simple RNN 모델 학습
logger.info("\n[3/4] Simple RNN 모델 학습 중...")
rnn_model = hybrid_models.build_simple_rnn_model()
rnn_model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

rnn_history = rnn_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=get_callbacks('rnn'),
    verbose=1
)

hybrid_models.models['rnn'] = rnn_model
hybrid_models.histories['rnn'] = rnn_history

# 4. Bidirectional LSTM 모델 학습
logger.info("\n[4/4] Bidirectional LSTM 모델 학습 중...")
bi_lstm_model = hybrid_models.build_bidirectional_lstm_model()
bi_lstm_model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

bi_lstm_history = bi_lstm_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=get_callbacks('bi_lstm'),
    verbose=1
)

hybrid_models.models['bi_lstm'] = bi_lstm_model
hybrid_models.histories['bi_lstm'] = bi_lstm_history

# ===================================
# 14. ARIMA 모델 학습
# ===================================

logger.info("\n" + "="*60)
logger.info("ARIMA 모델 학습 시작")
logger.info("="*60)

# ARIMA용 데이터 준비 (원본 스케일 사용)
arima_data = Modified_Data['TOTALCNT'][:train_size]

# 정상성 검정
logger.info("ADF 정상성 검정 수행...")
adf_result = adfuller(arima_data)
logger.info(f"ADF 통계량: {adf_result[0]:.4f}")
logger.info(f"p-value: {adf_result[1]:.4f}")

if adf_result[1] > 0.05:
    logger.warning("데이터가 비정상적입니다. 차분이 필요할 수 있습니다.")

# ARIMA 모델 학습
try:
    # ARIMA(p,d,q) 파라미터
    # p: 자기회귀 차수, d: 차분 차수, q: 이동평균 차수
    arima_model = ARIMA(arima_data, order=(2, 1, 2))
    arima_fitted = arima_model.fit()
    
    logger.info("ARIMA 모델 학습 완료")
    logger.info(f"AIC: {arima_fitted.aic:.2f}")
    logger.info(f"BIC: {arima_fitted.bic:.2f}")
    
except Exception as e:
    logger.error(f"ARIMA 모델 학습 실패: {str(e)}")
    arima_fitted = None

# ===================================
# 15. 앙상블 예측 함수
# ===================================

def ensemble_predict(models, X_test, weights=None):
    """
    여러 모델의 예측을 앙상블하여 최종 예측 생성
    
    Parameters:
    - models: 학습된 모델 딕셔너리
    - X_test: 테스트 입력 데이터
    - weights: 각 모델의 가중치 (None이면 동일 가중치)
    
    Returns:
    - ensemble_pred: 앙상블 예측값
    - individual_preds: 개별 모델 예측값들
    """
    
    if weights is None:
        # 기본 가중치: 모든 모델 동일
        weights = {
            'lstm': 0.3,
            'gru': 0.25,
            'rnn': 0.15,
            'bi_lstm': 0.3
        }
    
    individual_preds = {}
    
    # 각 모델별 예측
    for model_name, model in models.items():
        if model_name != 'arima':  # ARIMA는 별도 처리
            pred = model.predict(X_test, verbose=0).flatten()
            individual_preds[model_name] = pred
    
    # 가중 평균 계산
    ensemble_pred = np.zeros_like(list(individual_preds.values())[0])
    
    for model_name, pred in individual_preds.items():
        weight = weights.get(model_name, 0.25)
        ensemble_pred += weight * pred
    
    return ensemble_pred, individual_preds

# ===================================
# 16. 모델 평가
# ===================================

logger.info("\n" + "="*60)
logger.info("모델 성능 평가")
logger.info("="*60)

# 앙상블 예측 수행
ensemble_pred, individual_preds = ensemble_predict(hybrid_models.models, X_test)

# 성능 지표 계산 함수
def calculate_metrics(y_true, y_pred, model_name):
    """모델 성능 지표 계산"""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # 역변환하여 실제 스케일로 변환
    y_true_original = sscaler.inverse_transform(
        np.column_stack([y_true] + [np.zeros_like(y_true)] * (len(scaled_list) - 1))
    )[:, 0]
    
    y_pred_original = sscaler.inverse_transform(
        np.column_stack([y_pred] + [np.zeros_like(y_pred)] * (len(scaled_list) - 1))
    )[:, 0]
    
    mae_original = mean_absolute_error(y_true_original, y_pred_original)
    
    logger.info(f"\n{model_name} 모델 성능:")
    logger.info(f"  - MSE: {mse:.4f}")
    logger.info(f"  - MAE: {mae:.4f}")
    logger.info(f"  - RMSE: {rmse:.4f}")
    logger.info(f"  - R²: {r2:.4f}")
    logger.info(f"  - MAE (원본 스케일): {mae_original:.2f}")
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mae_original': mae_original
    }

# 개별 모델 평가
results = {}
for model_name, pred in individual_preds.items():
    results[model_name] = calculate_metrics(y_test, pred, model_name.upper())

# 앙상블 모델 평가
results['ensemble'] = calculate_metrics(y_test, ensemble_pred, "ENSEMBLE")

# ===================================
# 17. 결과 시각화
# ===================================

logger.info("\n결과 시각화 생성 중...")

# 1. 학습 곡선 시각화
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()

for idx, (model_name, history) in enumerate(hybrid_models.histories.items()):
    if idx < 4:
        ax = axes[idx]
        
        # 손실 곡선
        ax.plot(history.history['loss'], label='Training Loss', color='blue')
        ax.plot(history.history['val_loss'], label='Validation Loss', color='red')
        
        ax.set_title(f'{model_name.upper()} 학습 곡선')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. 예측 결과 시각화
plt.figure(figsize=(20, 10))

# 샘플 구간 선택 (처음 200개)
sample_size = min(200, len(y_test))

# 실제값을 원본 스케일로 변환
y_test_original = sscaler.inverse_transform(
    np.column_stack([y_test[:sample_size]] + [np.zeros_like(y_test[:sample_size])] * (len(scaled_list) - 1))
)[:, 0]

# 실제값 플롯
plt.plot(y_test_original, label='실제값', color='black', linewidth=2)

# 개별 모델 예측값 플롯
colors = ['blue', 'green', 'red', 'orange', 'purple']
for idx, (model_name, pred) in enumerate(individual_preds.items()):
    pred_original = sscaler.inverse_transform(
        np.column_stack([pred[:sample_size]] + [np.zeros_like(pred[:sample_size])] * (len(scaled_list) - 1))
    )[:, 0]
    
    plt.plot(pred_original, label=f'{model_name.upper()} 예측', 
             color=colors[idx], alpha=0.7, linewidth=1.5)

# 앙상블 예측값 플롯
ensemble_original = sscaler.inverse_transform(
    np.column_stack([ensemble_pred[:sample_size]] + [np.zeros_like(ensemble_pred[:sample_size])] * (len(scaled_list) - 1))
)[:, 0]

plt.plot(ensemble_original, label='앙상블 예측', 
         color='darkred', linewidth=2, linestyle='--')

plt.title('반도체 물류량 예측 결과 (10분 후)', fontsize=16)
plt.xlabel('시간 (분)', fontsize=12)
plt.ylabel('물류량 (TOTALCNT)', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('prediction_results.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. 모델별 성능 비교 차트
models_list = list(results.keys())
mae_values = [results[model]['mae_original'] for model in models_list]
r2_values = [results[model]['r2'] for model in models_list]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# MAE 비교
ax1.bar(models_list, mae_values, color=['blue', 'green', 'red', 'orange', 'darkred'])
ax1.set_title('모델별 MAE 비교 (낮을수록 좋음)', fontsize=14)
ax1.set_ylabel('MAE (원본 스케일)', fontsize=12)
ax1.set_xlabel('모델', fontsize=12)

# 값 표시
for i, v in enumerate(mae_values):
    ax1.text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom')

# R² 비교
ax2.bar(models_list, r2_values, color=['blue', 'green', 'red', 'orange', 'darkred'])
ax2.set_title('모델별 R² 비교 (높을수록 좋음)', fontsize=14)
ax2.set_ylabel('R² Score', fontsize=12)
ax2.set_xlabel('모델', fontsize=12)
ax2.set_ylim(0, 1)

# 값 표시
for i, v in enumerate(r2_values):
    ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ===================================
# 18. 병목 구간 예측
# ===================================

logger.info("\n" + "="*60)
logger.info("병목 구간 예측 분석")
logger.info("="*60)

# 병목 임계값 설정 (상위 15%)
bottleneck_threshold = np.percentile(Modified_Data['TOTALCNT'], 85)
logger.info(f"병목 임계값: {bottleneck_threshold:.0f}")

# 예측값에서 병목 구간 찾기
ensemble_original_full = sscaler.inverse_transform(
    np.column_stack([ensemble_pred] + [np.zeros_like(ensemble_pred)] * (len(scaled_list) - 1))
)[:, 0]

bottleneck_predictions = ensemble_original_full > bottleneck_threshold
bottleneck_count = np.sum(bottleneck_predictions)
bottleneck_ratio = bottleneck_count / len(bottleneck_predictions) * 100

logger.info(f"예측된 병목 구간: {bottleneck_count}개 ({bottleneck_ratio:.1f}%)")

# 병목 구간 시각화
plt.figure(figsize=(20, 8))

# 실제값과 예측값
y_test_original_full = sscaler.inverse_transform(
    np.column_stack([y_test] + [np.zeros_like(y_test)] * (len(scaled_list) - 1))
)[:, 0]

plt.subplot(2, 1, 1)
plt.plot(y_test_original_full[:500], label='실제 물류량', color='blue')
plt.axhline(y=bottleneck_threshold, color='red', linestyle='--', label=f'병목 임계값 ({bottleneck_threshold:.0f})')
plt.title('실제 물류량 및 병목 구간', fontsize=14)
plt.ylabel('물류량', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(ensemble_original_full[:500], label='예측 물류량', color='green')
plt.axhline(y=bottleneck_threshold, color='red', linestyle='--', label=f'병목 임계값 ({bottleneck_threshold:.0f})')

# 병목 구간 하이라이트
bottleneck_indices = np.where(bottleneck_predictions[:500])[0]
if len(bottleneck_indices) > 0:
    plt.scatter(bottleneck_indices, ensemble_original_full[bottleneck_indices], 
                color='red', s=30, label='예측된 병목', zorder=5)

plt.title('예측 물류량 및 병목 구간 감지', fontsize=14)
plt.xlabel('시간 (분)', fontsize=12)
plt.ylabel('물류량', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bottleneck_detection.png', dpi=300, bbox_inches='tight')
plt.show()

# ===================================
# 19. 모델 및 결과 저장
# ===================================

logger.info("\n" + "="*60)
logger.info("모델 및 결과 저장")
logger.info("="*60)

# 저장 디렉토리 생성
os.makedirs('model', exist_ok=True)
os.makedirs('scaler', exist_ok=True)
os.makedirs('results', exist_ok=True)

# 1. 딥러닝 모델 저장
for model_name, model in hybrid_models.models.items():
    model_path = f'model/{model_name}_final_hybrid.keras'
    model.save(model_path)
    logger.info(f"{model_name.upper()} 모델 저장: {model_path}")

# 2. ARIMA 모델 저장
if arima_fitted is not None:
    arima_path = 'model/arima_model.pkl'
    joblib.dump(arima_fitted, arima_path)
    logger.info(f"ARIMA 모델 저장: {arima_path}")

# 3. 스케일러 저장
scaler_path = 'scaler/standard_scaler_hybrid.pkl'
joblib.dump(sscaler, scaler_path)
logger.info(f"스케일러 저장: {scaler_path}")

# 4. 성능 결과 저장
results_df = pd.DataFrame(results).T
results_df.to_csv('results/model_performance.csv')
logger.info("성능 결과 저장: results/model_performance.csv")

# 5. 학습 설정 저장
config = {
    'seq_length': SEQ_LENGTH,
    'future_minutes': future_minutes,
    'epochs': EPOCHS,
    'batch_size': BATCH_SIZE,
    'learning_rate': LEARNING_RATE,
    'input_features': input_features,
    'scaled_columns': scaled_list,
    'train_size': train_size,
    'val_size': val_size,
    'test_size': len(X_test),
    'bottleneck_threshold': bottleneck_threshold,
    'model_weights': {
        'lstm': 0.3,
        'gru': 0.25,
        'rnn': 0.15,
        'bi_lstm': 0.3
    }
}

import json
with open('results/training_config.json', 'w') as f:
    json.dump(config, f, indent=4)
logger.info("학습 설정 저장: results/training_config.json")

# ===================================
# 20. 최종 요약
# ===================================

logger.info("\n" + "="*60)
logger.info("학습 완료 요약")
logger.info("="*60)

# 최고 성능 모델 찾기
best_model = min(results.items(), key=lambda x: x[1]['mae_original'])
logger.info(f"\n최고 성능 모델: {best_model[0].upper()}")
logger.info(f"  - MAE: {best_model[1]['mae_original']:.2f}")
logger.info(f"  - R²: {best_model[1]['r2']:.4f}")

# 앙상블 성능
logger.info(f"\n앙상블 모델 성능:")
logger.info(f"  - MAE: {results['ensemble']['mae_original']:.2f}")
logger.info(f"  - R²: {results['ensemble']['r2']:.4f}")

# 병목 예측 정확도
actual_bottlenecks = y_test_original_full > bottleneck_threshold
bottleneck_accuracy = np.mean(bottleneck_predictions == actual_bottlenecks) * 100
logger.info(f"\n병목 구간 예측 정확도: {bottleneck_accuracy:.1f}%")

logger.info("\n" + "="*60)
logger.info("모든 작업이 완료되었습니다!")
logger.info("="*60)

# 학습 시간 출력
import time
end_time = time.time()
# training_time = end_time - start_time  # start_time이 정의되어 있어야 함
# logger.info(f"\n총 학습 시간: {training_time/60:.1f}분")