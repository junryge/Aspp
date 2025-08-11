# -*- coding: utf-8 -*-
"""
TensorFlow 2.18 강제 실행 솔루션
모델 파일을 직접 수정하여 로드
"""

import os
import sys
import json
import zipfile
import tempfile
import shutil

# CUDA 경고 제거
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
import warnings

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

print("="*80)
print("TensorFlow 2.18 강제 실행 솔루션")
print(f"TensorFlow 버전: {tf.__version__}")
print("="*80)

# ===================================
# 1단계: 모델 파일 강제 수정 함수
# ===================================

def force_fix_keras_model(input_path, output_path):
    """
    .keras 파일을 강제로 수정하여 TF 2.18 호환 만들기
    """
    print(f"\n파일 수정 중: {os.path.basename(input_path)}")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # 1. keras 파일 압축 해제
            with zipfile.ZipFile(input_path, 'r') as zip_file:
                zip_file.extractall(temp_dir)
            
            # 2. config.json 수정
            config_path = os.path.join(temp_dir, 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # 문제가 되는 모든 파라미터 제거
                def clean_config(obj):
                    if isinstance(obj, dict):
                        # 제거할 키들
                        keys_to_remove = ['time_major', 'implementation']
                        
                        # config 내부 정리
                        if 'config' in obj and isinstance(obj['config'], dict):
                            for key in keys_to_remove:
                                obj['config'].pop(key, None)
                        
                        # 재귀적으로 정리
                        for key in list(obj.keys()):
                            if key in keys_to_remove:
                                obj.pop(key, None)
                            else:
                                clean_config(obj[key])
                    
                    elif isinstance(obj, list):
                        for item in obj:
                            clean_config(item)
                    
                    return obj
                
                config = clean_config(config)
                
                # 3. 수정된 config 저장
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2)
                
                print("  ✓ config.json 수정 완료")
            
            # 4. 다시 압축
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, temp_dir)
                        zip_file.write(file_path, arcname)
            
            print("  ✓ 수정된 파일 저장 완료")
            return True
            
    except Exception as e:
        print(f"  × 파일 수정 실패: {str(e)}")
        return False

# ===================================
# 2단계: 모델별 정확한 아키텍처 정의
# ===================================

def build_exact_models():
    """학습 시 사용된 정확한 모델 구조"""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        LSTM, GRU, SimpleRNN, Dense, Dropout, 
        Input, BatchNormalization, Bidirectional
    )
    
    models = {}
    
    # LSTM 모델 (학습_수정후.PY 참조)
    models['lstm'] = Sequential([
        Input(shape=(30, 6)),  # 6개 특징
        LSTM(100, return_sequences=True),
        Dropout(0.2),
        BatchNormalization(),
        LSTM(100, return_sequences=True),
        Dropout(0.2),
        BatchNormalization(),
        LSTM(100, return_sequences=True),
        Dropout(0.2),
        LSTM(100, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    
    # GRU 모델
    models['gru'] = Sequential([
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
    models['rnn'] = Sequential([
        Input(shape=(30, 6)),
        SimpleRNN(100, return_sequences=True),
        Dropout(0.2),
        SimpleRNN(50, return_sequences=False),
        Dropout(0.2),
        Dense(30, activation='relu'),
        Dense(1)
    ])
    
    # Bi-LSTM 모델
    models['bi_lstm'] = Sequential([
        Input(shape=(30, 6)),
        Bidirectional(LSTM(50, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(50, return_sequences=False)),
        Dropout(0.2),
        Dense(30, activation='relu'),
        Dense(1)
    ])
    
    return models

# ===================================
# 3단계: 스크립트 디렉토리 설정
# ===================================

Script_dir = os.path.dirname(os.path.abspath(__file__))

# ===================================
# 4단계: 모델 로드 프로세스
# ===================================

print("\n모델 파일 수정 및 로드 시작...")

# 수정된 파일을 저장할 디렉토리 생성
fixed_dir = os.path.join(Script_dir, "model", "tf218_fixed")
os.makedirs(fixed_dir, exist_ok=True)

# 모델 아키텍처 생성
model_architectures = build_exact_models()

# 각 모델 처리
models = {}
model_names = ['lstm', 'gru', 'rnn', 'bi_lstm']

for idx, model_name in enumerate(model_names, 1):
    print(f"\n[{idx}/4] {model_name.upper()} 모델 처리 중...")
    
    original_path = os.path.join(Script_dir, "model", f"lee_{model_name}_final_hybrid.keras")
    fixed_path = os.path.join(fixed_dir, f"{model_name}_fixed.keras")
    
    if not os.path.exists(original_path):
        print(f"  × 원본 파일 없음: {original_path}")
        continue
    
    print(f"  파일 크기: {os.path.getsize(original_path):,} bytes")
    
    # 방법 1: 수정된 파일이 이미 있으면 사용
    if os.path.exists(fixed_path):
        try:
            models[model_name] = tf.keras.models.load_model(fixed_path, compile=False)
            print(f"  ✓ 기존 수정 파일 로드 성공")
            continue
        except:
            pass
    
    # 방법 2: 파일 수정 후 로드
    if force_fix_keras_model(original_path, fixed_path):
        try:
            models[model_name] = tf.keras.models.load_model(fixed_path, compile=False)
            print(f"  ✓ 수정된 파일 로드 성공")
            continue
        except Exception as e:
            print(f"  × 수정 파일 로드 실패: {str(e)[:60]}")
    
    # 방법 3: 구조 생성 후 weights만 추출
    print("  가중치 추출 시도 중...")
    try:
        # 임시로 원본 파일에서 weights 추출
        with tempfile.TemporaryDirectory() as temp_dir:
            # keras 파일 압축 해제
            with zipfile.ZipFile(original_path, 'r') as zip_file:
                zip_file.extractall(temp_dir)
            
            # model.weights.h5 파일 찾기
            weights_path = os.path.join(temp_dir, 'model.weights.h5')
            if os.path.exists(weights_path):
                # 새 모델에 weights 로드
                model = model_architectures[model_name]
                model.load_weights(weights_path)
                models[model_name] = model
                print(f"  ✓ 가중치 직접 로드 성공")
                continue
    except Exception as e:
        print(f"  × 가중치 추출 실패: {str(e)[:60]}")
    
    # 방법 4: 마지막 시도 - 전체 weights 로드
    try:
        model = model_architectures[model_name]
        model.build((None, 30, 6))  # 모델 빌드
        model.load_weights(original_path)
        models[model_name] = model
        print(f"  ✓ 전체 가중치 로드 성공")
    except Exception as e:
        print(f"  × 모든 방법 실패: {str(e)[:60]}")

# ===================================
# 5단계: 로드 결과 확인
# ===================================

print("\n" + "="*80)
print("모델 로드 결과")
print("="*80)

for model_name in model_names:
    if model_name in models:
        print(f"  ✓ {model_name.upper()} 모델 로드 성공")
    else:
        print(f"  × {model_name.upper()} 모델 로드 실패")

print(f"\n총 {len(models)}개 모델 로드 완료")

if not models:
    print("\n❌ 모델 로드 실패!")
    print("\n추가 시도할 수 있는 방법:")
    print("1. sudo pip install tensorflow==2.14.0 --force-reinstall")
    print("2. Docker 컨테이너에서 TF 2.14 실행")
    print("3. 모델을 처음부터 다시 학습 (TF 2.18에서)")
    sys.exit(1)

# ===================================
# 6단계: 데이터 처리 및 예측
# ===================================

print("\n" + "="*80)
print("데이터 로드 및 예측")
print("="*80)

# 데이터 로드
data_path = os.path.join(Script_dir, "data", "20250807_DATA.CSV")
if not os.path.exists(data_path):
    print(f"❌ 데이터 파일 없음: {data_path}")
    sys.exit(1)

data = pd.read_csv(data_path)
data['CURRTIME'] = pd.to_datetime(data['CURRTIME'], format='%Y%m%d%H%M')
data['TIME'] = pd.to_datetime(data['TIME'], format='%Y%m%d%H%M')

# SUM 컬럼 제거
data = data.drop(columns=[col for col in data.columns if 'SUM' in col])
data = data[['CURRTIME', 'TOTALCNT', 'TIME']]
data.set_index('CURRTIME', inplace=True)

# 특징 생성
print("특징 엔지니어링...")
data['hour'] = data.index.hour
data['dayofweek'] = data.index.dayofweek
data['is_weekend'] = (data.index.dayofweek >= 5).astype(int)
data['MA_5'] = data['TOTALCNT'].rolling(window=5, min_periods=1).mean()
data['MA_10'] = data['TOTALCNT'].rolling(window=10, min_periods=1).mean()
data['MA_30'] = data['TOTALCNT'].rolling(window=30, min_periods=1).mean()
data['STD_5'] = data['TOTALCNT'].rolling(window=5, min_periods=1).std()
data['STD_10'] = data['TOTALCNT'].rolling(window=10, min_periods=1).std()
data['change_rate'] = data['TOTALCNT'].pct_change()
data['change_rate_5'] = data['TOTALCNT'].pct_change(5)
data = data.fillna(method='ffill').fillna(0)

# 스케일러 로드
scaler = None
scaler_paths = [
    os.path.join(Script_dir, "scaler", "StdScaler_s30f10_0731_2079936.save"),
    os.path.join(Script_dir, "scaler", "standard_scaler_hybrid.pkl")
]

for path in scaler_paths:
    if os.path.exists(path):
        try:
            scaler = joblib.load(path)
            print(f"스케일러 로드: {os.path.basename(path)}")
            break
        except:
            pass

if scaler is None:
    print("❌ 스케일러 없음!")
    sys.exit(1)

# 스케일링
if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ == 2:
    data['FUTURE'] = data['TOTALCNT']
    scale_cols = ['TOTALCNT', 'FUTURE']
elif hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ == 7:
    data['FUTURE'] = data['TOTALCNT']
    scale_cols = ['TOTALCNT', 'FUTURE', 'MA_5', 'MA_10', 'MA_30', 'STD_5', 'STD_10']
else:
    scale_cols = ['TOTALCNT', 'MA_5', 'MA_10', 'MA_30', 'STD_5', 'STD_10']

scaled = scaler.transform(data[scale_cols])
scaled_df = pd.DataFrame(scaled, columns=[f'scaled_{c}' for c in scale_cols], index=data.index)
data = pd.concat([data, scaled_df], axis=1)

# 시퀀스 생성
input_cols = [c for c in data.columns if c.startswith('scaled_') and c != 'scaled_FUTURE']
X = data[input_cols].values[-30:].reshape(1, 30, -1)

print(f"입력 shape: {X.shape}")

# 예측
predictions = {}
for name, model in models.items():
    try:
        pred = model.predict(X, verbose=0)[0][0]
        predictions[name] = pred
        print(f"✓ {name.upper()} 예측 완료")
    except:
        pass

# 앙상블
if predictions:
    weights = {'lstm': 0.3, 'gru': 0.25, 'rnn': 0.15, 'bi_lstm': 0.3}
    total_w = sum(weights.get(k, 0.25) for k in predictions)
    ensemble = sum(predictions[k] * weights.get(k, 0.25) / total_w for k in predictions)
    
    # 역스케일링
    dummy = np.zeros((1, len(scale_cols)))
    idx = 1 if 'FUTURE' in scale_cols else 0
    dummy[0, idx] = ensemble
    result = scaler.inverse_transform(dummy)[0, idx]
    
    print(f"\n예측 결과: {int(np.round(result))}")
    print(int(np.round(result)))
else:
    print("❌ 예측 실패!")