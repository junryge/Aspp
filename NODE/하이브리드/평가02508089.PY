# -*- coding: utf-8 -*-
"""
반도체 물류 예측 - 하이브리드 모델 실시간 예측
각 모델을 개별적으로 로드하는 버전 (TF 2.18+ 호환)
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Bidirectional, Dense, Dropout, Input
from datetime import datetime, timedelta
import joblib
import logging
import warnings

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
print("반도체 물류 예측 시스템 - 개별 모델 로드 (TF 2.18+ 호환)")
print("="*70)

# -------------------------------------------------------------------
# 중요: 모델의 입력 shape를 여기서 정의합니다.
# 데이터 전처리 후 생성될 X_seq의 shape와 일치해야 합니다.
# 예: seq_length=30, input_features=10 -> Input(shape=(30, 10))
# standard_scaler_hybrid.pkl 스케일러가 10개의 피처를 사용하므로 (30, 10)으로 설정합니다.
# 만약 다른 스케일러나 피처 개수를 사용했다면 이 부분을 수정해야 합니다.
INPUT_SHAPE = (30, 10) 
print(f"모델 입력 Shape 고정: {INPUT_SHAPE}")
# -------------------------------------------------------------------

# ===================================
# 1. LSTM 모델 로드
# ===================================
print("\n[1/4] LSTM 모델 로드 중...")
lstm_model = None
lstm_path = os.path.join(Script_dir, "model", "lee_lstm_final_hybrid.keras")

if os.path.exists(lstm_path):
    try:
        # TF 2.18+ 환경에 맞는 구조를 먼저 정의
        model = Sequential([
            Input(shape=INPUT_SHAPE),
            LSTM(100, return_sequences=True),
            Dropout(0.2),
            LSTM(100, return_sequences=True),
            Dropout(0.2),
            LSTM(100, return_sequences=True),
            Dropout(0.2),
            LSTM(100),
            Dropout(0.2),
            Dense(1)
        ])
        # 구조 위에 가중치만 덮어쓰기
        model.load_weights(lstm_path)
        lstm_model = model
        print(f"  ✓ LSTM 모델 로드 성공 (구조 재구성 + 가중치 로드)")
    except Exception as e:
        print(f"  × LSTM 모델 로드 실패: {str(e)[:120]}")
else:
    print(f"  × 파일 없음: {lstm_path}")


# ===================================
# 2. GRU 모델 로드
# ===================================
print("\n[2/4] GRU 모델 로드 중...")
gru_model = None
gru_path = os.path.join(Script_dir, "model", "lee_gru_final_hybrid.keras")

if os.path.exists(gru_path):
    try:
        model = Sequential([
            Input(shape=INPUT_SHAPE),
            GRU(100, return_sequences=True),
            Dropout(0.2),
            GRU(100, return_sequences=True),
            Dropout(0.2),
            GRU(50, return_sequences=False),
            Dropout(0.2),
            Dense(30, activation='relu'),
            Dense(1)
        ])
        model.load_weights(gru_path)
        gru_model = model
        print(f"  ✓ GRU 모델 로드 성공 (구조 재구성 + 가중치 로드)")
    except Exception as e:
        print(f"  × GRU 모델 로드 실패: {str(e)[:120]}")
else:
    print(f"  × 파일 없음: {gru_path}")


# ===================================
# 3. RNN 모델 로드
# ===================================
print("\n[3/4] RNN 모델 로드 중...")
rnn_model = None
rnn_path = os.path.join(Script_dir, "model", "lee_rnn_final_hybrid.keras")

if os.path.exists(rnn_path):
    try:
        model = Sequential([
            Input(shape=INPUT_SHAPE),
            SimpleRNN(100, return_sequences=True),
            Dropout(0.2),
            SimpleRNN(50, return_sequences=False),
            Dropout(0.2),
            Dense(30, activation='relu'),
            Dense(1)
        ])
        model.load_weights(rnn_path)
        rnn_model = model
        print(f"  ✓ RNN 모델 로드 성공 (구조 재구성 + 가중치 로드)")
    except Exception as e:
        print(f"  × RNN 모델 로드 실패: {str(e)[:120]}")
else:
    print(f"  × 파일 없음: {rnn_path}")


# ===================================
# 4. Bi-LSTM 모델 로드
# ===================================
print("\n[4/4] Bi-LSTM 모델 로드 중...")
bi_lstm_model = None
bi_lstm_path = os.path.join(Script_dir, "model", "lee_bi_lstm_final_hybrid.keras")

if os.path.exists(bi_lstm_path):
    try:
        model = Sequential([
            Input(shape=INPUT_SHAPE),
            Bidirectional(LSTM(50, return_sequences=True)),
            Dropout(0.2),
            Bidirectional(LSTM(50, return_sequences=False)),
            Dropout(0.2),
            Dense(30, activation='relu'),
            Dense(1)
        ])
        model.load_weights(bi_lstm_path)
        bi_lstm_model = model
        print(f"  ✓ Bi-LSTM 모델 로드 성공 (구조 재구성 + 가중치 로드)")
    except Exception as e:
        print(f"  × Bi-LSTM 모델 로드 실패: {str(e)[:120]}")
else:
    print(f"  × 파일 없음: {bi_lstm_path}")

# ===================================
# 모델 로드 결과 확인
# ===================================
print("\n" + "="*70)
print("모델 로드 결과")
print("="*70)

models = {}
if lstm_model is not None:
    models['lstm'] = lstm_model
    print("  ✓ LSTM 모델 로드됨")
else:
    print("  × LSTM 모델 로드 실패")

if gru_model is not None:
    models['gru'] = gru_model
    print("  ✓ GRU 모델 로드됨")
else:
    print("  × GRU 모델 로드 실패")

if rnn_model is not None:
    models['rnn'] = rnn_model
    print("  ✓ RNN 모델 로드됨")
else:
    print("  × RNN 모델 로드 실패")

if bi_lstm_model is not None:
    models['bi_lstm'] = bi_lstm_model
    print("  ✓ Bi-LSTM 모델 로드됨")
else:
    print("  × Bi-LSTM 모델 로드 실패")

print(f"\n총 {len(models)}개 모델 로드 성공")

if not models:
    print("\n❌ 로드된 모델이 없습니다! 프로그램을 종료합니다.")
    sys.exit(1)

# ===================================
# 데이터 로드 및 전처리
# ===================================
print("\n" + "="*70)
print("데이터 로드 및 전처리")
print("="*70)

# 데이터 파일 경로
Full_data_path = os.path.join(Script_dir, "data", "20250807_DATA.CSV")

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
    # FUTURE 컬럼이 스케일러에 필요한 경우, TOTALCNT로 임시 생성
    if 'FUTURE' in required_columns and 'FUTURE' not in Modified_Data.columns:
        Modified_Data['FUTURE'] = Modified_Data['TOTALCNT']
    
    # 스케일러에 필요한 컬럼만 선택
    available_columns = [col for col in required_columns if col in Modified_Data.columns]

else: # 구버전 스케일러 호환
    if hasattr(scaler, 'n_features_in_'):
        if scaler.n_features_in_ == 2:
            available_columns = ['TOTALCNT', 'FUTURE']
            Modified_Data['FUTURE'] = Modified_Data['TOTALCNT']
        elif scaler.n_features_in_ == 10: # hybrid scaler
             available_columns = ['TOTALCNT', 'FUTURE', 'hour', 'dayofweek', 'is_weekend', 'MA_5', 'MA_10', 'MA_30', 'STD_5', 'STD_10']
             Modified_Data['FUTURE'] = Modified_Data['TOTALCNT']
        else: # 기본값
            available_columns = ['TOTALCNT']

# 스케일링
scaled_data = scaler.transform(Modified_Data[available_columns])
scaled_df = pd.DataFrame(scaled_data, columns=[f'scaled_{col}' for col in available_columns])
scaled_df.index = Modified_Data.index

Scaled_Data = pd.merge(Modified_Data, scaled_df, left_index=True, right_index=True, how='left')

# 시퀀스 생성
seq_length = 30
# 스케일링 된 피처들 중 'scaled_FUTURE'는 입력에서 제외
input_features = [col for col in Scaled_Data.columns if col.startswith('scaled_') and 'FUTURE' not in col]


if len(Scaled_Data) < seq_length:
    print(f"❌ 데이터가 부족합니다. 최소 {seq_length}개 필요")
    sys.exit(1)

X_seq = Scaled_Data[input_features].iloc[-seq_length:].values
X_seq = X_seq.reshape(1, seq_length, len(input_features))

print(f"✓ 입력 시퀀스 shape: {X_seq.shape}")

# 입력 shape 검증
if X_seq.shape[1:] != INPUT_SHAPE:
    print(f"❌ 입력 데이터 shape 불일치! 모델은 {INPUT_SHAPE}를 기대하지만, 데이터는 {X_seq.shape[1:]} 입니다.")
    print("  스케일러에 사용된 피처 개수와 모델 정의 시 사용된 피처 개수가 다른 것 같습니다.")
    sys.exit(1)

# ===================================
# 예측 수행
# ===================================
print("\n" + "="*70)
print("예측 수행")
print("="*70)

predictions = {}

# 각 모델별로 예측 수행
if 'lstm' in models:
    try:
        pred = models['lstm'].predict(X_seq, verbose=0)
        predictions['lstm'] = pred[0][0]
        print("  ✓ LSTM 예측 완료")
    except Exception as e:
        print(f"  × LSTM 예측 실패: {str(e)[:80]}")

if 'gru' in models:
    try:
        pred = models['gru'].predict(X_seq, verbose=0)
        predictions['gru'] = pred[0][0]
        print("  ✓ GRU 예측 완료")
    except Exception as e:
        print(f"  × GRU 예측 실패: {str(e)[:80]}")

if 'rnn' in models:
    try:
        pred = models['rnn'].predict(X_seq, verbose=0)
        predictions['rnn'] = pred[0][0]
        print("  ✓ RNN 예측 완료")
    except Exception as e:
        print(f"  × RNN 예측 실패: {str(e)[:80]}")

if 'bi_lstm' in models:
    try:
        pred = models['bi_lstm'].predict(X_seq, verbose=0)
        predictions['bi_lstm'] = pred[0][0]
        print("  ✓ Bi-LSTM 예측 완료")
    except Exception as e:
        print(f"  × Bi-LSTM 예측 실패: {str(e)[:80]}")

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

# 예측값 위치 찾기 (스케일러가 학습한 컬럼 리스트에서 'FUTURE'의 위치)
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
if 'lstm' in individual_predictions:
    print(f"  LSTM      : {int(np.round(individual_predictions['lstm'])):4d}")
else:
    print(f"  LSTM      : N/A")

if 'gru' in individual_predictions:
    print(f"  GRU       : {int(np.round(individual_predictions['gru'])):4d}")
else:
    print(f"  GRU       : N/A")

if 'rnn' in individual_predictions:
    print(f"  RNN       : {int(np.round(individual_predictions['rnn'])):4d}")
else:
    print(f"  RNN       : N/A")

if 'bi_lstm' in individual_predictions:
    print(f"  Bi-LSTM   : {int(np.round(individual_predictions['bi_lstm'])):4d}")
else:
    print(f"  Bi-LSTM   : N/A")

ensemble_value = int(np.round(final_prediction))
print(f"\n앙상블 예측값: {ensemble_value}")
print("="*70)

# 최종 출력 (기존 형식 유지)
print(ensemble_value)
