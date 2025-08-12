# -*- coding: utf-8 -*-
"""
반도체 물류 예측 - 하이브리드 모델 실시간 예측 (TF 2.15.0 호환)
모델 로드 문제 해결 버전
"""

import os
import sys

# CUDA 경고 완전 제거
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model, Sequential, Model
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Bidirectional, Dense, Dropout, Input, BatchNormalization
from datetime import datetime, timedelta
import joblib
import logging
import warnings
import h5py

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel(logging.ERROR)

# TensorFlow GPU 완전 비활성화
try:
    tf.config.set_visible_devices([], 'GPU')
except:
    pass

# 랜덤 시드 고정
tf.random.set_seed(2079936)

# 스크립트 디렉토리
Script_dir = os.path.dirname(os.path.abspath(__file__))

print("="*70)
print("반도체 물류 예측 시스템 - 모델 로드 (TF 2.15.0)")
print("="*70)

# ===================================
# 모델 구조 정의 함수들
# ===================================

def build_lstm_model(input_shape):
    """LSTM 모델 구조 생성"""
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units=100, return_sequences=True),
        Dropout(0.2),
        BatchNormalization(),
        LSTM(units=100, return_sequences=True),
        Dropout(0.2),
        BatchNormalization(),
        LSTM(units=100, return_sequences=False),
        Dropout(0.2),
        Dense(units=50, activation='relu'),
        Dense(units=1)
    ])
    return model

def build_gru_model(input_shape):
    """GRU 모델 구조 생성"""
    model = Sequential([
        Input(shape=input_shape),
        GRU(units=100, return_sequences=True),
        Dropout(0.2),
        GRU(units=100, return_sequences=True),
        Dropout(0.2),
        GRU(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=30, activation='relu'),
        Dense(units=1)
    ])
    return model

def build_rnn_model(input_shape):
    """RNN 모델 구조 생성"""
    model = Sequential([
        Input(shape=input_shape),
        SimpleRNN(units=100, return_sequences=True),
        Dropout(0.2),
        SimpleRNN(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=30, activation='relu'),
        Dense(units=1)
    ])
    return model

def build_bilstm_model(input_shape):
    """Bidirectional LSTM 모델 구조 생성"""
    model = Sequential([
        Input(shape=input_shape),
        Bidirectional(LSTM(units=50, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(units=50, return_sequences=False)),
        Dropout(0.2),
        Dense(units=30, activation='relu'),
        Dense(units=1)
    ])
    return model

def load_model_safely(model_path, model_type, input_shape):
    """안전하게 모델 로드"""
    model = None
    
    # 방법 1: 직접 로드 시도
    try:
        model = load_model(model_path, compile=False)
        print(f"  ✓ {model_type} 모델 로드 성공 (직접 로드)")
        return model
    except Exception as e:
        print(f"  × 직접 로드 실패: {str(e)[:80]}")
    
    # 방법 2: H5 파일로 변환 시도
    if model_path.endswith('.keras'):
        h5_path = model_path.replace('.keras', '.h5')
        if os.path.exists(h5_path):
            try:
                model = load_model(h5_path, compile=False)
                print(f"  ✓ {model_type} 모델 로드 성공 (H5 파일)")
                return model
            except Exception as e:
                print(f"  × H5 로드 실패: {str(e)[:80]}")
    
    # 방법 3: SavedModel 형식 시도
    saved_model_path = model_path.replace('.keras', '')
    if os.path.exists(saved_model_path) and os.path.isdir(saved_model_path):
        try:
            model = tf.keras.models.load_model(saved_model_path, compile=False)
            print(f"  ✓ {model_type} 모델 로드 성공 (SavedModel)")
            return model
        except Exception as e:
            print(f"  × SavedModel 로드 실패: {str(e)[:80]}")
    
    # 방법 4: 모델 구조 생성 후 가중치만 로드
    try:
        # 모델 구조 생성
        if model_type == 'LSTM':
            model = build_lstm_model(input_shape)
        elif model_type == 'GRU':
            model = build_gru_model(input_shape)
        elif model_type == 'RNN':
            model = build_rnn_model(input_shape)
        elif model_type == 'Bi-LSTM':
            model = build_bilstm_model(input_shape)
        
        # 가중치 로드 시도
        if model_path.endswith('.keras'):
            # .keras 파일에서 가중치 추출
            try:
                with h5py.File(model_path, 'r') as f:
                    if 'model_weights' in f:
                        model.load_weights(model_path)
                        print(f"  ✓ {model_type} 모델 로드 성공 (가중치만)")
                        return model
            except Exception as e:
                print(f"  × 가중치 추출 실패: {str(e)[:80]}")
        
        # weights 파일 찾기
        weights_path = model_path.replace('.keras', '_weights.h5')
        if os.path.exists(weights_path):
            model.load_weights(weights_path)
            print(f"  ✓ {model_type} 모델 로드 성공 (별도 가중치 파일)")
            return model
            
    except Exception as e:
        print(f"  × 모델 구조 생성 실패: {str(e)[:80]}")
    
    return None

# ===================================
# 모델 로드
# ===================================

# 입력 shape 추정 (나중에 데이터에서 확인)
input_shape = (30, 7)  # 기본값, 나중에 조정됨

models = {}

# 1. LSTM 모델 로드
print("\n[1/4] LSTM 모델 로드 중...")
lstm_path = os.path.join(Script_dir, "model", "lee_lstm_final_hybrid.keras")
if os.path.exists(lstm_path):
    print(f"  파일 발견: {os.path.basename(lstm_path)}")
    print(f"  파일 크기: {os.path.getsize(lstm_path):,} bytes")
    lstm_model = load_model_safely(lstm_path, 'LSTM', input_shape)
    if lstm_model is not None:
        models['lstm'] = lstm_model
else:
    print(f"  × 파일 없음: {lstm_path}")

# 2. GRU 모델 로드
print("\n[2/4] GRU 모델 로드 중...")
gru_path = os.path.join(Script_dir, "model", "lee_gru_final_hybrid.keras")
if os.path.exists(gru_path):
    print(f"  파일 발견: {os.path.basename(gru_path)}")
    print(f"  파일 크기: {os.path.getsize(gru_path):,} bytes")
    gru_model = load_model_safely(gru_path, 'GRU', input_shape)
    if gru_model is not None:
        models['gru'] = gru_model
else:
    print(f"  × 파일 없음: {gru_path}")

# 3. RNN 모델 로드
print("\n[3/4] RNN 모델 로드 중...")
rnn_path = os.path.join(Script_dir, "model", "lee_rnn_final_hybrid.keras")
if os.path.exists(rnn_path):
    print(f"  파일 발견: {os.path.basename(rnn_path)}")
    print(f"  파일 크기: {os.path.getsize(rnn_path):,} bytes")
    rnn_model = load_model_safely(rnn_path, 'RNN', input_shape)
    if rnn_model is not None:
        models['rnn'] = rnn_model
else:
    print(f"  × 파일 없음: {rnn_path}")

# 4. Bi-LSTM 모델 로드
print("\n[4/4] Bi-LSTM 모델 로드 중...")
bi_lstm_path = os.path.join(Script_dir, "model", "lee_bi_lstm_final_hybrid.keras")
if os.path.exists(bi_lstm_path):
    print(f"  파일 발견: {os.path.basename(bi_lstm_path)}")
    print(f"  파일 크기: {os.path.getsize(bi_lstm_path):,} bytes")
    bi_lstm_model = load_model_safely(bi_lstm_path, 'Bi-LSTM', input_shape)
    if bi_lstm_model is not None:
        models['bi_lstm'] = bi_lstm_model
else:
    print(f"  × 파일 없음: {bi_lstm_path}")

# ===================================
# 모델 로드 결과 확인
# ===================================
print("\n" + "="*70)
print("모델 로드 결과")
print("="*70)

for model_name in ['lstm', 'gru', 'rnn', 'bi_lstm']:
    if model_name in models:
        print(f"  ✓ {model_name.upper()} 모델 로드됨")
    else:
        print(f"  × {model_name.upper()} 모델 로드 실패")

print(f"\n총 {len(models)}개 모델 로드 성공")

if not models:
    print("\n❌ 로드된 모델이 없습니다!")
    print("\n가능한 해결 방법:")
    print("1. 모델을 H5 형식으로 다시 저장: model.save('model.h5')")
    print("2. SavedModel 형식으로 저장: model.save('model_dir')")
    print("3. 가중치만 별도 저장: model.save_weights('weights.h5')")
    sys.exit(1)

# ===================================
# 데이터 로드 및 전처리
# ===================================
print("\n" + "="*70)
print("데이터 로드 및 전처리")
print("="*70)

# 데이터 파일 경로
Full_data_path = os.path.join(Script_dir, "data", "20250807_DATA.csv")

# 파일 존재 확인
if not os.path.exists(Full_data_path):
    print(f"❌ 파일이 없습니다: {Full_data_path}")
    sys.exit(1)

# 데이터 로드
Full_Data = pd.read_csv(Full_data_path)
print(f"✓ 데이터 로드 완료: {Full_Data.shape}")

# Time 관련된 Columns Datatime으로 타입 변경
Full_Data['CURRTIME'] = pd.to_datetime(Full_Data['CURRTIME'], format='%Y%m%d%H%M')
Full_Data['TIME'] = pd.to_datetime(Full_Data['TIME'], format='%Y%m%d%H%M')

# SUM 컬럼 제거
columns_to_drop = [col for col in Full_Data.columns if 'SUM' in col]
Full_Data = Full_Data.drop(columns=columns_to_drop)

# 현재반송큐만 가지고 처리
Full_Data = Full_Data[['CURRTIME', 'TOTALCNT','TIME']]
Full_Data.set_index('CURRTIME', inplace=True)

Modified_Data = Full_Data.copy()

# 특징 엔지니어링
print("✓ 특징 엔지니어링 수행 중...")
Modified_Data['hour'] = Modified_Data.index.hour
Modified_Data['dayofweek'] = Modified_Data.index.dayofweek
Modified_Data['is_weekend'] = (Modified_Data.index.dayofweek >= 5).astype(int)
Modified_Data['MA_5'] = Modified_Data['TOTALCNT'].rolling(window=5, min_periods=1).mean()
Modified_Data['MA_10'] = Modified_Data['TOTALCNT'].rolling(window=10, min_periods=1).mean()
Modified_Data['MA_30'] = Modified_Data['TOTALCNT'].rolling(window=30, min_periods=1).mean()
Modified_Data['STD_5'] = Modified_Data['TOTALCNT'].rolling(window=5, min_periods=1).std()
Modified_Data['STD_10'] = Modified_Data['TOTALCNT'].rolling(window=10, min_periods=1).std()
Modified_Data['change_rate'] = Modified_Data['TOTALCNT'].pct_change()
Modified_Data['change_rate_5'] = Modified_Data['TOTALCNT'].pct_change(5)
Modified_Data = Modified_Data.fillna(method='ffill').fillna(0)

# ===================================
# 스케일러 로드
# ===================================
print("\n스케일러 로드 중...")
scaler = None
scaler_paths = [
    os.path.join(Script_dir, "scaler", "standard_scaler_hybrid.pkl"),
    os.path.join(Script_dir, "scaler", "StdScaler_s30f10_0731_2079936.save"),
    os.path.join(Script_dir, "scaler", "StdScaler_s30f10_0724_2079936.save")
]

for scaler_path in scaler_paths:
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
            print(f"✓ 스케일러 로드 성공: {os.path.basename(scaler_path)}")
            break
        except Exception as e:
            print(f"× 스케일러 로드 실패: {os.path.basename(scaler_path)}")

if scaler is None:
    print("❌ 스케일러를 찾을 수 없습니다!")
    sys.exit(1)

# 스케일러가 기대하는 특징 확인
if hasattr(scaler, 'feature_names_in_'):
    required_columns = list(scaler.feature_names_in_)
    for col in required_columns:
        if col not in Modified_Data.columns:
            if col == 'FUTURE':
                Modified_Data['FUTURE'] = Modified_Data['TOTALCNT']
    available_columns = [col for col in required_columns if col in Modified_Data.columns]
else:
    if hasattr(scaler, 'n_features_in_'):
        if scaler.n_features_in_ == 2:
            available_columns = ['TOTALCNT', 'FUTURE']
            Modified_Data['FUTURE'] = Modified_Data['TOTALCNT']
        elif scaler.n_features_in_ == 7:
            available_columns = ['TOTALCNT', 'FUTURE', 'MA_5', 'MA_10', 'MA_30', 'STD_5', 'STD_10']
            Modified_Data['FUTURE'] = Modified_Data['TOTALCNT']
        else:
            available_columns = ['TOTALCNT', 'MA_5', 'MA_10', 'MA_30', 'STD_5', 'STD_10']

# 스케일링
scaled_data = scaler.transform(Modified_Data[available_columns])
scaled_df = pd.DataFrame(scaled_data, columns=[f'scaled_{col}' for col in available_columns])
scaled_df.index = Modified_Data.index

Scaled_Data = pd.merge(Modified_Data, scaled_df, left_index=True, right_index=True, how='left')

# 시퀀스 생성
seq_length = 30
input_features = [col for col in Scaled_Data.columns if col.startswith('scaled_') and col != 'scaled_FUTURE']

# 실제 input shape 확인 및 모델 재구성 필요시
actual_input_shape = (seq_length, len(input_features))
if actual_input_shape != input_shape:
    print(f"\n입력 shape 조정: {input_shape} -> {actual_input_shape}")
    # 필요시 모델 재로드...

if len(Scaled_Data) < seq_length:
    print(f"❌ 데이터가 부족합니다. 최소 {seq_length}개 필요")
    sys.exit(1)

X_seq = Scaled_Data[input_features].iloc[-seq_length:].values
X_seq = X_seq.reshape(1, seq_length, len(input_features))

print(f"✓ 입력 시퀀스 shape: {X_seq.shape}")

# ===================================
# 예측 수행
# ===================================
print("\n" + "="*70)
print("예측 수행")
print("="*70)

predictions = {}

# 각 모델별로 예측 수행
for model_name, model in models.items():
    try:
        pred = model.predict(X_seq, verbose=0)
        predictions[model_name] = pred[0][0]
        print(f"  ✓ {model_name.upper()} 예측 완료")
    except Exception as e:
        print(f"  × {model_name.upper()} 예측 실패: {str(e)[:80]}")

print(f"\n예측 성공한 모델 수: {len(predictions)}개")

# 앙상블 예측
if predictions:
    weights = {
        'lstm': 0.3,
        'gru': 0.25,
        'rnn': 0.15,
        'bi_lstm': 0.3
    }
    
    # 사용 가능한 모델에 대해서만 가중치 정규화
    available_weights = {k: v for k, v in weights.items() if k in predictions}
    total_weight = sum(available_weights.values())
    
    if total_weight > 0:
        normalized_weights = {k: v/total_weight for k, v in available_weights.items()}
    else:
        normalized_weights = {k: 1.0/len(predictions) for k in predictions.keys()}
    
    # 앙상블 예측값 계산
    ensemble_pred = sum(predictions[k] * normalized_weights[k] for k in predictions.keys())
else:
    print("❌ 예측 실패!")
    sys.exit(1)

# ===================================
# 역스케일링
# ===================================
n_features = len(available_columns)
dummy_array = np.zeros((1, n_features))

# 예측값 위치 찾기
if 'FUTURE' in available_columns:
    pred_idx = available_columns.index('FUTURE')
elif 'TOTALCNT' in available_columns:
    pred_idx = available_columns.index('TOTALCNT')
else:
    pred_idx = 0

# 앙상블 예측값 역스케일링
dummy_array[0, pred_idx] = ensemble_pred
final_prediction = scaler.inverse_transform(dummy_array)[0, pred_idx]

# 개별 모델 예측값 역스케일링
individual_predictions = {}
for model_name, pred in predictions.items():
    dummy_array[0, pred_idx] = pred
    individual_predictions[model_name] = scaler.inverse_transform(dummy_array)[0, pred_idx]

# ===================================
# 결과 출력
# ===================================
print("\n" + "="*70)
print("예측 결과")
print("="*70)

print("\n개별 모델 예측값:")
for model_name in ['lstm', 'gru', 'rnn', 'bi_lstm']:
    if model_name in individual_predictions:
        print(f"  {model_name.upper():8s} : {int(np.round(individual_predictions[model_name])):4d}")
    else:
        print(f"  {model_name.upper():8s} : N/A")

ensemble_value = int(np.round(final_prediction))
print(f"\n앙상블 예측값: {ensemble_value}")
print("="*70)

# 최종 출력 (기존 형식 유지)
print(ensemble_value)