# -*- coding: utf-8 -*-
"""
반도체 물류 예측 - TensorFlow 2.18 호환 버전
모델 파일에서 time_major 파라미터를 제거하고 로드
"""

import os
import sys
import json
import tempfile
import shutil
import zipfile

# CUDA 경고 완전 제거
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Bidirectional, Dense, Dropout, Input, BatchNormalization
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
print("반도체 물류 예측 시스템 - TensorFlow 2.18 호환 버전")
print("="*70)

# ===================================
# 커스텀 레이어 정의 (time_major 무시)
# ===================================

class TF218_LSTM(LSTM):
    def __init__(self, *args, **kwargs):
        # TF 2.18에서 지원하지 않는 파라미터 제거
        kwargs.pop('time_major', None)
        kwargs.pop('implementation', None)
        super().__init__(*args, **kwargs)

class TF218_GRU(GRU):
    def __init__(self, *args, **kwargs):
        kwargs.pop('time_major', None)
        kwargs.pop('implementation', None)
        super().__init__(*args, **kwargs)

class TF218_SimpleRNN(SimpleRNN):
    def __init__(self, *args, **kwargs):
        kwargs.pop('time_major', None)
        kwargs.pop('implementation', None)
        super().__init__(*args, **kwargs)

# ===================================
# 모델 파일 수정 함수
# ===================================

def fix_keras_model_for_tf218(input_path, output_path):
    """
    .keras 파일에서 TF 2.18과 호환되지 않는 파라미터를 제거
    """
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # .keras 파일 압축 해제
            with zipfile.ZipFile(input_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # config.json 파일 읽기 및 수정
            config_path = os.path.join(temp_dir, 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # time_major 파라미터 제거 (재귀적으로)
                def remove_incompatible_params(obj):
                    if isinstance(obj, dict):
                        # config 딕셔너리에서 호환되지 않는 파라미터 제거
                        if 'config' in obj and isinstance(obj['config'], dict):
                            obj['config'].pop('time_major', None)
                            obj['config'].pop('implementation', None)
                        
                        # class_name을 커스텀 클래스로 변경
                        if 'class_name' in obj:
                            if obj['class_name'] == 'LSTM':
                                obj['class_name'] = 'TF218_LSTM'
                            elif obj['class_name'] == 'GRU':
                                obj['class_name'] = 'TF218_GRU'
                            elif obj['class_name'] == 'SimpleRNN':
                                obj['class_name'] = 'TF218_SimpleRNN'
                        
                        # 모든 하위 요소에 대해 재귀 호출
                        for key, value in obj.items():
                            remove_incompatible_params(value)
                    elif isinstance(obj, list):
                        for item in obj:
                            remove_incompatible_params(item)
                
                remove_incompatible_params(config)
                
                # 수정된 config 저장
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2)
            
            # 수정된 파일로 다시 압축
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, temp_dir)
                        zip_ref.write(file_path, arcname)
        
        return True
    except Exception as e:
        print(f"  모델 파일 수정 실패: {str(e)}")
        return False

# ===================================
# 모델별 구조 정의 (백업용)
# ===================================

def create_model_architectures():
    """각 모델의 구조를 미리 정의"""
    architectures = {}
    
    # LSTM 모델
    architectures['lstm'] = Sequential([
        Input(shape=(30, 6)),  # scaled features 6개
        LSTM(100, return_sequences=True),
        Dropout(0.2),
        BatchNormalization(),
        LSTM(100, return_sequences=True),
        Dropout(0.2),
        BatchNormalization(),
        LSTM(100, return_sequences=True),
        Dropout(0.2),
        LSTM(100),
        Dropout(0.2),
        Dense(1)
    ])
    
    # GRU 모델
    architectures['gru'] = Sequential([
        Input(shape=(30, 6)),
        GRU(100, return_sequences=True),
        Dropout(0.2),
        GRU(100, return_sequences=True),
        Dropout(0.2),
        GRU(50, return_sequences=False),
        Dropout(0.2),
        Dense(30, activation='relu'),
        Dense(1)
    ])
    
    # RNN 모델
    architectures['rnn'] = Sequential([
        Input(shape=(30, 6)),
        SimpleRNN(100, return_sequences=True),
        Dropout(0.2),
        SimpleRNN(50, return_sequences=False),
        Dropout(0.2),
        Dense(30, activation='relu'),
        Dense(1)
    ])
    
    # Bi-LSTM 모델
    architectures['bi_lstm'] = Sequential([
        Input(shape=(30, 6)),
        Bidirectional(LSTM(50, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(50, return_sequences=False)),
        Dropout(0.2),
        Dense(30, activation='relu'),
        Dense(1)
    ])
    
    return architectures

# ===================================
# 메인 모델 로드 함수
# ===================================

# 커스텀 객체 정의
custom_objects = {
    'TF218_LSTM': TF218_LSTM,
    'TF218_GRU': TF218_GRU,
    'TF218_SimpleRNN': TF218_SimpleRNN,
    'LSTM': TF218_LSTM,
    'GRU': TF218_GRU,
    'SimpleRNN': TF218_SimpleRNN,
}

# 모델 아키텍처 생성
model_architectures = create_model_architectures()

# 각 모델 로드
models = {}
model_names = ['lstm', 'gru', 'rnn', 'bi_lstm']

for idx, model_name in enumerate(model_names, 1):
    print(f"\n[{idx}/4] {model_name.upper()} 모델 로드 중...")
    
    original_path = os.path.join(Script_dir, "model", f"lee_{model_name}_final_hybrid.keras")
    fixed_path = os.path.join(Script_dir, "model", f"lee_{model_name}_final_hybrid_tf218.keras")
    
    if os.path.exists(original_path):
        print(f"  파일 발견: {os.path.basename(original_path)}")
        print(f"  파일 크기: {os.path.getsize(original_path):,} bytes")
        
        # 방법 1: 수정된 파일이 이미 있으면 로드
        if os.path.exists(fixed_path):
            try:
                models[model_name] = load_model(fixed_path, custom_objects=custom_objects, compile=False)
                print(f"  ✓ {model_name.upper()} 모델 로드 성공 (수정된 파일)")
                continue
            except:
                pass
        
        # 방법 2: 원본 파일 수정 후 로드
        print(f"  모델 파일 수정 중...")
        if fix_keras_model_for_tf218(original_path, fixed_path):
            try:
                models[model_name] = load_model(fixed_path, custom_objects=custom_objects, compile=False)
                print(f"  ✓ {model_name.upper()} 모델 로드 성공 (파일 수정 후)")
                continue
            except Exception as e:
                print(f"  × 수정된 파일 로드 실패: {str(e)[:80]}")
        
        # 방법 3: 커스텀 객체로 직접 로드 시도
        try:
            models[model_name] = load_model(original_path, custom_objects=custom_objects, compile=False)
            print(f"  ✓ {model_name.upper()} 모델 로드 성공 (커스텀 객체)")
            continue
        except:
            pass
        
        # 방법 4: 구조 생성 후 weights만 로드
        try:
            model = model_architectures[model_name]
            model.load_weights(original_path)
            models[model_name] = model
            print(f"  ✓ {model_name.upper()} 모델 로드 성공 (weights only)")
        except Exception as e:
            print(f"  × 모든 방법 실패: {str(e)[:80]}")
    else:
        print(f"  × 파일 없음: {original_path}")

# ===================================
# 모델 로드 결과
# ===================================
print("\n" + "="*70)
print("모델 로드 결과")
print("="*70)

for model_name in model_names:
    if model_name in models:
        print(f"  ✓ {model_name.upper()} 모델 로드됨")
    else:
        print(f"  × {model_name.upper()} 모델 로드 실패")

print(f"\n총 {len(models)}개 모델 로드 성공")

if not models:
    print("\n❌ 로드된 모델이 없습니다!")
    print("\n가능한 해결 방법:")
    print("1. TensorFlow 버전을 2.14로 다운그레이드: pip install tensorflow==2.14")
    print("2. 모델을 TensorFlow 2.18에서 다시 학습")
    print("3. 수동으로 모델 파일을 수정")
    sys.exit(1)

# ===================================
# 데이터 로드 및 전처리 (기존 코드와 동일)
# ===================================
print("\n" + "="*70)
print("데이터 로드 및 전처리")
print("="*70)

# 데이터 파일 경로
Full_data_path = os.path.join(Script_dir, "data", "20250807_DATA.CSV")

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

# 스케일러 로드
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
        except:
            pass

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
    weights = {'lstm': 0.3, 'gru': 0.25, 'rnn': 0.15, 'bi_lstm': 0.3}
    available_weights = {k: v for k, v in weights.items() if k in predictions}
    total_weight = sum(available_weights.values())
    
    if total_weight > 0:
        normalized_weights = {k: v/total_weight for k, v in available_weights.items()}
    else:
        normalized_weights = {k: 1.0/len(predictions) for k in predictions.keys()}
    
    ensemble_pred = sum(predictions[k] * normalized_weights[k] for k in predictions.keys())
else:
    print("❌ 예측 실패!")
    sys.exit(1)

# 역스케일링
n_features = len(available_columns)
dummy_array = np.zeros((1, n_features))

if 'FUTURE' in available_columns:
    pred_idx = available_columns.index('FUTURE')
elif 'TOTALCNT' in available_columns:
    pred_idx = available_columns.index('TOTALCNT')
else:
    pred_idx = 0

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
for model_name in model_names:
    if model_name in individual_predictions:
        print(f"  {model_name.upper():8s}: {int(np.round(individual_predictions[model_name])):4d}")
    else:
        print(f"  {model_name.upper():8s}: N/A")

ensemble_value = int(np.round(final_prediction))
print(f"\n앙상블 예측값: {ensemble_value}")
print("="*70)

# 최종 출력
print(ensemble_value)