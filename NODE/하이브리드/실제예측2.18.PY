# -*- coding: utf-8 -*-
"""
반도체 물류 예측 - 하이브리드 모델 실시간 예측 (TensorFlow 2.18 호환 버전)
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
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Bidirectional
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

# TensorFlow 버전 호환성을 위한 커스텀 레이어 정의
class CompatibleLSTM(LSTM):
    def __init__(self, *args, **kwargs):
        # 인식되지 않는 파라미터 제거
        kwargs.pop('time_major', None)
        super().__init__(*args, **kwargs)

class CompatibleGRU(GRU):
    def __init__(self, *args, **kwargs):
        kwargs.pop('time_major', None)
        super().__init__(*args, **kwargs)

class CompatibleSimpleRNN(SimpleRNN):
    def __init__(self, *args, **kwargs):
        kwargs.pop('time_major', None)
        super().__init__(*args, **kwargs)

# 스크립트 디렉토리
Script_dir = os.path.dirname(os.path.abspath(__file__))

# 데이터 파일 경로 직접 지정
Full_data_path = os.path.join(Script_dir, "data", "20250807_DATA.CSV")

# 파일 존재 확인
if not os.path.exists(Full_data_path):
    print(f"파일이 없습니다: {Full_data_path}")
    print(f"Script_dir: {Script_dir}")
    sys.exit(1)

# 데이터 로드
Full_Data = pd.read_csv(Full_data_path)

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

# 특징 엔지니어링 추가 (학습 시와 동일하게)
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

# 커스텀 객체 정의 (TensorFlow 버전 호환성)
custom_objects = {
    'LSTM': CompatibleLSTM,
    'GRU': CompatibleGRU,
    'SimpleRNN': CompatibleSimpleRNN,
    'time_major': None
}

# 모델 로드 - 여러 경로 시도
models = {}
model_names = ['lstm', 'gru', 'rnn', 'bi_lstm']

for model_name in model_names:
    model_paths = [
        os.path.join(Script_dir, "model", f"{model_name}_final_hybrid.keras"),
        os.path.join(Script_dir, "model", f"BM_s30f10_0731_2079936.keras"),
        os.path.join(Script_dir, "model", f"Model_s30f10_0731_2079936.keras"),
        os.path.join(Script_dir, "model", f"Model_s30f10_0724_2079936.keras")
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                # 커스텀 객체와 함께 로드
                models[model_name] = load_model(model_path, 
                                                custom_objects=custom_objects, 
                                                compile=False)
                print(f"{model_name} 모델 로드 성공")
                break
            except Exception as e:
                # 오류 발생 시 다른 방법 시도
                try:
                    # JSON 구조 수정 방식
                    import json
                    import tempfile
                    
                    # 모델 파일 읽기
                    with open(model_path, 'rb') as f:
                        model_data = f.read()
                    
                    # 임시 파일로 저장하고 로드
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.keras') as tmp:
                        tmp.write(model_data)
                        tmp_path = tmp.name
                    
                    # 수정된 커스텀 객체로 재시도
                    models[model_name] = tf.keras.models.load_model(
                        tmp_path,
                        custom_objects=custom_objects,
                        compile=False
                    )
                    os.unlink(tmp_path)
                    print(f"{model_name} 모델 로드 성공 (대체 방법)")
                    break
                except:
                    continue

# 최소 하나의 모델이라도 로드되었는지 확인
if not models:
    print("로드된 모델이 없습니다!")
    sys.exit(1)
else:
    print(f"로드된 모델 수: {len(models)}")

# 스케일러 로드 - 여러 경로 시도
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
            print("스케일러 로드 성공")
            break
        except:
            continue

if scaler is None:
    print("스케일러를 찾을 수 없습니다!")
    sys.exit(1)

# 스케일러가 기대하는 특징 확인 및 처리
if hasattr(scaler, 'feature_names_in_'):
    # 스케일러가 기대하는 정확한 컬럼 사용
    required_columns = list(scaler.feature_names_in_)
    
    # 필요한 컬럼이 없으면 생성
    for col in required_columns:
        if col not in Modified_Data.columns:
            if col == 'FUTURE':
                Modified_Data['FUTURE'] = Modified_Data['TOTALCNT']
    
    # 스케일러가 기대하는 순서대로 컬럼 선택
    available_columns = [col for col in required_columns if col in Modified_Data.columns]
else:
    # feature_names_in_이 없는 경우 (구버전 sklearn)
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
try:
    scaled_data = scaler.transform(Modified_Data[available_columns])
    scaled_df = pd.DataFrame(scaled_data, columns=[f'scaled_{col}' for col in available_columns])
    scaled_df.index = Modified_Data.index
except Exception as e:
    print(f"스케일링 오류: {e}")
    sys.exit(1)

Scaled_Data = pd.merge(Modified_Data, scaled_df, left_index=True, right_index=True, how='left')

# 시퀀스 생성 (마지막 30개 데이터)
seq_length = 30
input_features = [col for col in Scaled_Data.columns if col.startswith('scaled_') and col != 'scaled_FUTURE']

if len(Scaled_Data) < seq_length:
    print(f"데이터가 부족합니다. 최소 {seq_length}개 필요")
    sys.exit(1)

X_seq = Scaled_Data[input_features].iloc[-seq_length:].values
X_seq = X_seq.reshape(1, seq_length, len(input_features))

# 예측 수행
predictions = {}
for model_name, model in models.items():
    try:
        pred = model.predict(X_seq, verbose=0)
        predictions[model_name] = pred[0][0]
    except Exception as e:
        print(f"{model_name} 예측 실패: {e}")
        pass

# 앙상블 예측 (가중 평균)
if predictions:
    weights = {'lstm': 0.3, 'gru': 0.25, 'rnn': 0.15, 'bi_lstm': 0.3}
    ensemble_pred = 0
    total_weight = 0
    
    for model_name, pred in predictions.items():
        weight = weights.get(model_name, 0.25)
        ensemble_pred += pred * weight
        total_weight += weight
    
    if total_weight > 0:
        ensemble_pred = ensemble_pred / total_weight
else:
    print("예측 실패!")
    sys.exit(1)

# 역스케일링
n_features = len(available_columns)
dummy_array = np.zeros((1, n_features))

# FUTURE 또는 TOTALCNT 위치 찾기
if 'FUTURE' in available_columns:
    pred_idx = available_columns.index('FUTURE')
elif 'TOTALCNT' in available_columns:
    pred_idx = available_columns.index('TOTALCNT')
else:
    pred_idx = 0

dummy_array[0, pred_idx] = ensemble_pred
final_prediction = scaler.inverse_transform(dummy_array)[0, pred_idx]

# 앙상블 예측값만 출력
ensemble_value = int(np.round(final_prediction))
print(ensemble_value)