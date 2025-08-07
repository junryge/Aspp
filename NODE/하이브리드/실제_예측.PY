# -*- coding: utf-8 -*-



# 파일 찾기 부분만 수정
import sys
import glob

# 스크립트 디렉토리
Script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"현재 디렉토리: {Script_dir}")

# CSV 파일 찾기 - 대소문자 구분 없이 검색
Full_data_path = None
target_filename = '20250807_DATA.csv'

# 현재 디렉토리와 data 하위 디렉토리에서 검색
search_dirs = [
    Script_dir,
    os.path.join(Script_dir, "data"),
    os.path.join(Script_dir, "DATA"),
    os.getcwd()  # 현재 작업 디렉토리도 추가
]

for search_dir in search_dirs:
    if os.path.exists(search_dir):
        # 대소문자 구분 없이 파일 검색
        files = glob.glob(os.path.join(search_dir, "*"))
        for file in files:
            if os.path.basename(file).lower() == target_filename.lower():
                Full_data_path = file
                print(f"파일 찾음: {Full_data_path}")
                break
    if Full_data_path:
        break

# 파일을 못 찾은 경우
if Full_data_path is None:
    print(f"\n'{target_filename}' 파일을 찾을 수 없습니다.")
    print("검색한 디렉토리:")
    for d in search_dirs:
        print(f"  - {d}")
    print("\n현재 디렉토리의 CSV 파일 목록:")
    for f in glob.glob(os.path.join(Script_dir, "*.csv")) + glob.glob(os.path.join(Script_dir, "*.CSV")):
        print(f"  - {os.path.basename(f)}")
    sys.exit(1)  # exit() 대신 sys.exit() 사용

# 데이터 로드
Full_Data = pd.read_csv(Full_data_path)
print(f"데이터 로드 성공: {Full_Data.shape}")



"""
반도체 물류 예측 - 하이브리드 모델 실시간 예측
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import os
from datetime import datetime, timedelta
import joblib
import logging
import warnings

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel(logging.ERROR)

# CPU 모드 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

# 랜덤 시드 고정
tf.random.set_seed(2079936)

# 스크립트 디렉토리
Script_dir = os.path.dirname(os.path.abspath(__file__))

# 데이터 파일 찾기 - 여러 경로 시도
Full_data_name = '20250807_DATA.csv'
possible_paths = [
    os.path.join(Script_dir, Full_data_name),  # 스크립트와 같은 폴더
    os.path.join(Script_dir, "data", Full_data_name),  # data 폴더
    Full_data_name  # 현재 디렉토리
]

Full_data_path = None
for path in possible_paths:
    if os.path.exists(path):
        Full_data_path = path
        break

if Full_data_path is None:
    print("CSV 파일을 찾을 수 없습니다. 다음 위치에 파일을 놓아주세요:")
    for path in possible_paths:
        print(f"  - {path}")
    exit(1)

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
                models[model_name] = load_model(model_path, compile=False)
                break
            except:
                continue

# 최소 하나의 모델이라도 로드되었는지 확인
if not models:
    print("로드된 모델이 없습니다!")
    exit(1)

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
            break
        except:
            continue

if scaler is None:
    print("스케일러를 찾을 수 없습니다!")
    exit(1)

# 스케일링할 컬럼
scale_columns = ['TOTALCNT', 'MA_5', 'MA_10', 'MA_30', 'STD_5', 'STD_10']
available_columns = [col for col in scale_columns if col in Modified_Data.columns]

# FUTURE 컬럼 더미로 추가 (스케일러가 기대하는 경우)
if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ == 7:
    Modified_Data['FUTURE'] = Modified_Data['TOTALCNT']
    available_columns = ['TOTALCNT', 'FUTURE', 'MA_5', 'MA_10', 'MA_30', 'STD_5', 'STD_10']
    available_columns = [col for col in available_columns if col in Modified_Data.columns]

# 스케일링
scaled_data = scaler.transform(Modified_Data[available_columns])
scaled_df = pd.DataFrame(scaled_data, columns=[f'scaled_{col}' for col in available_columns])
scaled_df.index = Modified_Data.index

Scaled_Data = pd.merge(Modified_Data, scaled_df, left_index=True, right_index=True, how='left')

# 시퀀스 생성 (마지막 30개 데이터)
seq_length = 30
input_features = [col for col in Scaled_Data.columns if col.startswith('scaled_') and col != 'scaled_FUTURE']

if len(Scaled_Data) < seq_length:
    print(f"데이터가 부족합니다. 최소 {seq_length}개 필요")
    exit(1)

X_seq = Scaled_Data[input_features].iloc[-seq_length:].values
X_seq = X_seq.reshape(1, seq_length, len(input_features))

# 예측 수행
predictions = {}
for model_name, model in models.items():
    try:
        pred = model.predict(X_seq, verbose=0)
        predictions[model_name] = pred[0][0]
    except:
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
    exit(1)

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

# 개별 모델 예측값도 역스케일링
individual_predictions = {}
for model_name, pred in predictions.items():
    dummy_array[0, pred_idx] = pred
    individual_predictions[model_name] = scaler.inverse_transform(dummy_array)[0, pred_idx]

# 결과 출력 (기존 형식과 동일하게)
predictions = np.round(final_prediction).astype(int)

# 딕셔너리 형태로 출력
predicted_dict = {
    "LSTM": int(round(individual_predictions.get('lstm', predictions))),
    "GRU": int(round(individual_predictions.get('gru', predictions))),
    "RNN": int(round(individual_predictions.get('rnn', predictions))),
    "Bi-LSTM": int(round(individual_predictions.get('bi_lstm', predictions))),
    "ENSEMBLE": int(predictions)
}

print([predicted_dict])