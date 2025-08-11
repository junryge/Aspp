# -*- coding: utf-8 -*-
"""
반도체 물류 예측 - 하이브리드 모델 실시간 예측 (4개 모델 모두 로드)
TensorFlow 2.18 호환 버전
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
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Bidirectional, Dense, Dropout, Input
from datetime import datetime, timedelta
import joblib
import logging
import warnings
import json
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

# ===================================
# 모델 로드를 위한 커스텀 클래스 정의
# ===================================

# 커스텀 LSTM 클래스 (호환성)
class CompatibleLSTM(LSTM):
    def __init__(self, *args, **kwargs):
        # TF 2.18에서 인식하지 못하는 파라미터 제거
        kwargs.pop('time_major', None)
        kwargs.pop('implementation', None)
        super().__init__(*args, **kwargs)

class CompatibleGRU(GRU):
    def __init__(self, *args, **kwargs):
        kwargs.pop('time_major', None)
        kwargs.pop('implementation', None)
        super().__init__(*args, **kwargs)

class CompatibleSimpleRNN(SimpleRNN):
    def __init__(self, *args, **kwargs):
        kwargs.pop('time_major', None)
        kwargs.pop('implementation', None)
        super().__init__(*args, **kwargs)

# Bidirectional 래퍼
def create_bidirectional(layer_class):
    class CompatibleBidirectional(Bidirectional):
        def __init__(self, layer, **kwargs):
            if hasattr(layer, 'time_major'):
                delattr(layer, 'time_major')
            super().__init__(layer, **kwargs)
    return CompatibleBidirectional

# ===================================
# 모델 구조 재생성 함수 (로드 실패 시 대안)
# ===================================

def create_lstm_model(input_shape):
    """LSTM 모델 구조 생성"""
    model = Sequential([
        Input(shape=input_shape),
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
    return model

def create_gru_model(input_shape):
    """GRU 모델 구조 생성"""
    model = Sequential([
        Input(shape=input_shape),
        GRU(100, return_sequences=True),
        Dropout(0.2),
        GRU(100, return_sequences=True),
        Dropout(0.2),
        GRU(50, return_sequences=False),
        Dropout(0.2),
        Dense(30, activation='relu'),
        Dense(1)
    ])
    return model

def create_rnn_model(input_shape):
    """Simple RNN 모델 구조 생성"""
    model = Sequential([
        Input(shape=input_shape),
        SimpleRNN(100, return_sequences=True),
        Dropout(0.2),
        SimpleRNN(50, return_sequences=False),
        Dropout(0.2),
        Dense(30, activation='relu'),
        Dense(1)
    ])
    return model

def create_bi_lstm_model(input_shape):
    """Bidirectional LSTM 모델 구조 생성"""
    model = Sequential([
        Input(shape=input_shape),
        Bidirectional(LSTM(50, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(50, return_sequences=False)),
        Dropout(0.2),
        Dense(30, activation='relu'),
        Dense(1)
    ])
    return model

# ===================================
# 안전한 모델 로드 함수
# ===================================

def safe_load_model(model_path, model_name, custom_objects=None):
    """다양한 방법으로 모델 로드 시도"""
    print(f"\n{model_name} 모델 로드 시도 중...")
    
    # 방법 1: 기본 로드
    try:
        model = load_model(model_path, custom_objects=custom_objects, compile=False)
        print(f"  ✓ {model_name} - 기본 로드 성공")
        return model
    except Exception as e:
        print(f"  × 기본 로드 실패: {str(e)[:100]}")
    
    # 방법 2: weights_only 로드
    try:
        # 먼저 모델 구조를 생성
        if model_name == 'lstm':
            model = create_lstm_model((30, 1))  # 임시 shape
        elif model_name == 'gru':
            model = create_gru_model((30, 1))
        elif model_name == 'rnn':
            model = create_rnn_model((30, 1))
        elif model_name == 'bi_lstm':
            model = create_bi_lstm_model((30, 1))
        
        # 가중치만 로드
        model.load_weights(model_path)
        print(f"  ✓ {model_name} - weights_only 로드 성공")
        return model
    except Exception as e:
        print(f"  × weights_only 로드 실패: {str(e)[:100]}")
    
    # 방법 3: H5 파일로 변환 시도
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            # keras를 h5로 변환
            temp_model = tf.keras.models.load_model(model_path, compile=False)
            temp_model.save(tmp.name, save_format='h5')
            
            # h5 파일 로드
            model = tf.keras.models.load_model(tmp.name, custom_objects=custom_objects, compile=False)
            os.unlink(tmp.name)
            print(f"  ✓ {model_name} - H5 변환 로드 성공")
            return model
    except Exception as e:
        print(f"  × H5 변환 로드 실패: {str(e)[:100]}")
    
    return None

# ===================================
# 메인 스크립트 시작
# ===================================

# 스크립트 디렉토리
Script_dir = os.path.dirname(os.path.abspath(__file__))

# 데이터 파일 경로
Full_data_path = os.path.join(Script_dir, "data", "20250807_DATA.CSV")

# 파일 존재 확인
if not os.path.exists(Full_data_path):
    print(f"파일이 없습니다: {Full_data_path}")
    sys.exit(1)

print("="*60)
print("반도체 물류 예측 시스템 시작")
print("="*60)

# 데이터 로드
Full_Data = pd.read_csv(Full_data_path)
print(f"데이터 로드 완료: {Full_Data.shape}")

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
print("\n특징 엔지니어링 수행 중...")
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
# 모델 로드 (개선된 버전)
# ===================================

print("\n" + "="*60)
print("모델 로드 시작")
print("="*60)

# 커스텀 객체 정의
custom_objects = {
    'LSTM': CompatibleLSTM,
    'GRU': CompatibleGRU,
    'SimpleRNN': CompatibleSimpleRNN,
    'CompatibleLSTM': CompatibleLSTM,
    'CompatibleGRU': CompatibleGRU,
    'CompatibleSimpleRNN': CompatibleSimpleRNN,
}

# 모델 저장 딕셔너리
models = {}
model_names = ['lstm', 'gru', 'rnn', 'bi_lstm']

# 각 모델별로 로드 시도
for model_name in model_names:
    # 모델 파일 경로들 (여러 가능성 고려)
    model_paths = [
        os.path.join(Script_dir, "model", f"lee_{model_name}_final_hybrid.keras"),
        os.path.join(Script_dir, "model", f"{model_name}_final_hybrid.keras"),
        os.path.join(Script_dir, "model", f"lee_{model_name}_best.keras"),
        os.path.join(Script_dir, "model", f"{model_name}_best.keras"),
    ]
    
    # 실제 존재하는 파일 찾기
    existing_path = None
    for path in model_paths:
        if os.path.exists(path):
            existing_path = path
            print(f"\n파일 발견: {os.path.basename(path)}")
            break
    
    if existing_path:
        # 안전한 로드 함수 사용
        loaded_model = safe_load_model(existing_path, model_name, custom_objects)
        if loaded_model is not None:
            models[model_name] = loaded_model
            print(f"  ✅ {model_name.upper()} 모델 로드 완료!")
        else:
            print(f"  ❌ {model_name.upper()} 모델 로드 실패")
    else:
        print(f"\n⚠️  {model_name} 모델 파일을 찾을 수 없습니다.")
        print(f"   찾은 경로들:")
        for path in model_paths[:2]:  # 처음 2개만 표시
            print(f"   - {path}")

# 로드 결과 요약
print("\n" + "="*60)
print("모델 로드 결과")
print("="*60)
print(f"총 {len(models)}개 모델 로드 성공:")
for name in models.keys():
    print(f"  ✓ {name.upper()}")

missing_models = set(model_names) - set(models.keys())
if missing_models:
    print(f"\n로드 실패한 모델:")
    for name in missing_models:
        print(f"  × {name.upper()}")

# 최소 하나의 모델이라도 로드되었는지 확인
if not models:
    print("\n❌ 로드된 모델이 없습니다!")
    sys.exit(1)

# ===================================
# 스케일러 로드
# ===================================

print("\n" + "="*60)
print("스케일러 로드")
print("="*60)

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
            print(f"  오류: {str(e)[:100]}")

if scaler is None:
    print("❌ 스케일러를 찾을 수 없습니다!")
    sys.exit(1)

# ===================================
# 데이터 전처리 및 예측
# ===================================

print("\n" + "="*60)
print("데이터 전처리 및 예측")
print("="*60)

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

if len(Scaled_Data) < seq_length:
    print(f"❌ 데이터가 부족합니다. 최소 {seq_length}개 필요")
    sys.exit(1)

X_seq = Scaled_Data[input_features].iloc[-seq_length:].values
X_seq = X_seq.reshape(1, seq_length, len(input_features))

print(f"\n입력 시퀀스 shape: {X_seq.shape}")

# ===================================
# 예측 수행
# ===================================

print("\n개별 모델 예측 수행 중...")
predictions = {}

for model_name, model in models.items():
    try:
        pred = model.predict(X_seq, verbose=0)
        predictions[model_name] = pred[0][0]
        print(f"  ✓ {model_name.upper()} 예측 완료")
    except Exception as e:
        print(f"  × {model_name.upper()} 예측 실패: {str(e)[:100]}")

# 앙상블 예측
if predictions:
    # 모델별 가중치
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
        # 모든 모델에 동일한 가중치
        normalized_weights = {k: 1.0/len(predictions) for k in predictions.keys()}
    
    # 앙상블 예측값 계산
    ensemble_pred = sum(predictions[k] * normalized_weights[k] for k in predictions.keys())
    
    print(f"\n앙상블 예측 완료 (사용된 모델: {len(predictions)}개)")
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

print("\n" + "="*60)
print("예측 결과")
print("="*60)

print("\n개별 모델 예측값:")
for model_name in ['lstm', 'gru', 'rnn', 'bi_lstm']:
    if model_name in individual_predictions:
        value = int(np.round(individual_predictions[model_name]))
        print(f"  {model_name.upper():8s}: {value:4d}")
    else:
        print(f"  {model_name.upper():8s}: N/A")

ensemble_value = int(np.round(final_prediction))
print(f"\n앙상블 예측값: {ensemble_value}")
print("="*60)

# 최종 출력 (기존 형식 유지)
print(ensemble_value)