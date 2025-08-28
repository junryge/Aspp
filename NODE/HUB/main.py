# -*- coding: utf-8 -*- 
# 기본 실행
python MAIN_LSTM_SINGLE_20250709_CPU.py

# 파라미터와 함께 실행
python MAIN_LSTM_SINGLE_20250709_CPU.py 0 5

# VHL OFF 상황 테스트
python MAIN_LSTM_SINGLE_20250709_CPU.py 1 10
#MAIN LSTM_SINGLE_20250709 - CPU 최적화 버전
import numpy as np
import pandas as pd

# TensorFlow 2.15.0 CPU 호환성 설정
import warnings
warnings.filterwarnings('ignore')
import os

# CPU 전용 설정
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # GPU 완전 비활성화
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'

import tensorflow as tf

# CPU만 사용하도록 강제 설정
tf.config.set_visible_devices([], 'GPU')

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import sys
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error

# preproc 폴더의 상대 경로 추가
preproc_path = os.path.join(os.path.dirname(__file__), 'preproc')
sys.path.append(preproc_path)

from datetime import datetime
from EMPTY_ROW_SPLIT import EmptyRowSplit
#from ADD_OHT_ERROR import process_data_in_chunks 
from MAKE_UTIL import process_data_in_chunks as make_util_process_data_in_chunks
from MAKE_FUTURE import process_data_in_chunks as make_future_process_data_in_chunks
import joblib
import logging

tf.get_logger().setLevel(logging.ERROR)


def parse_command_arguments():
    """명령행 인수를 파싱하는 함수"""
    if len(sys.argv) > 1:
        param1 = sys.argv[1] 
        param2 = sys.argv[2]
        priority = int(param1)  # Convert to integer
        add_argument = int(param2)
        print(f"파라미터 받음: priority={priority}, add_argument={add_argument}")
    else:   #argument 전달 못받은 경우 기본값 0으로 적용
        priority = 0
        add_argument = 0
        print("기본 파라미터 사용: priority=0, add_argument=0")
    
    return priority, add_argument


def preprocess_data(input_file_path):
    """데이터 전처리를 수행하는 함수"""
    print("=== 데이터 전처리 시작 ===")
    
    # 1. EMPTY_ROW_SPLIT 처리
    print("1. EMPTY_ROW_SPLIT 처리 중...")
    empty_row_split = EmptyRowSplit(input_file_path, "HUBROOM")
    datalist = empty_row_split.process_data()
    print(f"   데이터 청크 개수: {len(datalist)}")

    # 2. OHT ERROR 컬럼 추가 처리 (현재 주석 처리됨)
    #processed_data = process_data_in_chunks(datalist,'OHT_ERROR_0624_TO_0630.csv')

    # 3. 유틸리티 컬럼 추가 처리
    print("2. MAKE_UTIL 처리 중...")
    utility_data = make_util_process_data_in_chunks(datalist)
    print("   유틸리티 컬럼 추가 완료")

    # 4. future 컬럼 추가 처리 
    print("3. MAKE_FUTURE 처리 중...")
    future_data = make_future_process_data_in_chunks(utility_data, 'CURRENT_M16A_3F_JOB_2', 'HUBROOM')
    print("   Future 컬럼 추가 완료")

    # 5. 결측치 제거 및 유효 데이터 필터링
    result_list = {}
    for key, data_chunk in future_data.items():
        df = data_chunk.copy()
        # future 컬럼이 NaN인 행 제거
        df.dropna(subset=['future'], inplace=True)
        if(df.size != 0): #데이터가 없는 chunk 제외
            result_list[key] = df
            print(f"   유효한 청크 {key}: {len(df)}개 행")
    
    print(f"=== 전처리 완료: 총 {len(result_list)}개 청크 ===")
    return result_list


def load_model_and_scaler():
    """모델과 스케일러를 로드하는 함수"""
    print("=== 모델 및 스케일러 로딩 ===")
    
    # 프로젝트 루트 디렉토리 찾기
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while not os.path.exists(os.path.join(current_dir, 'data')):
        current_dir = os.path.dirname(current_dir)

    # 모델 로딩 (TF 2.15.0 호환성 처리)
    model_name = 'LSTM_TEST_070714.keras'
    model_path = os.path.join(current_dir, "model", model_name)
    
    print(f"모델 경로: {model_path}")
    print(f"모델 파일 존재: {os.path.exists(model_path)}")
    
    # 다중 시도 모델 로딩
    model = None
    try:
        # 방법 1: 기본 로딩
        print("방법 1: 기본 로딩 시도...")
        model = load_model(model_path, compile=False)
        print("✅ 기본 방법으로 모델 로딩 성공!")
    except Exception as e1:
        print(f"❌ 기본 로딩 실패: {str(e1)[:100]}...")
        try:
            # 방법 2: safe_mode=False 사용
            print("방법 2: safe_mode=False 시도...")
            model = tf.keras.models.load_model(
                model_path, 
                compile=False,
                safe_mode=False
            )
            print("✅ safe_mode=False로 모델 로딩 성공!")
        except Exception as e2:
            print(f"❌ safe_mode 로딩도 실패: {str(e2)[:100]}...")
            try:
                # 방법 3: CPU 강제 사용
                print("방법 3: CPU 강제 사용 시도...")
                with tf.device('/CPU:0'):
                    model = tf.keras.models.load_model(model_path, compile=False)
                print("✅ CPU 디바이스로 모델 로딩 성공!")
            except Exception as e3:
                print(f"❌ 모든 방법 실패: {e3}")
                print("모델 파일을 다시 확인해주세요.")
                raise e3

    # 스케일러 로딩
    scaler_name = 'LSTM_TEST_scaler_070714.save'
    scaler_path = os.path.join(current_dir, "scaler", scaler_name)
    
    print(f"스케일러 경로: {scaler_path}")
    print(f"스케일러 파일 존재: {os.path.exists(scaler_path)}")
    
    scaler = joblib.load(scaler_path)
    print("✅ 스케일러 로딩 완료")
    
    return model, scaler


def prepare_prediction_data(result_list, scaler):
    """예측을 위한 데이터 준비 함수"""
    print("=== 예측 데이터 준비 ===")
    
    # 초기화
    X_not_scaled_all = []
    X_seq_all = []
    y_seq_all = []
    future_y_seq_all = []
    time_seq_all = []

    # 각 데이터 청크에 대해 전처리
    for key, data_chunk in result_list.items():
        print(f"청크 {key} 처리 중...")
        df = data_chunk.copy()
        
        # STAT_DT를 datetime 형식으로 변환
        df['STAT_DT'] = pd.to_datetime(df['STAT_DT'], format='%Y-%m-%d %H:%M:%S')
        
        y = df['future']  # future 값을 예측하기 위해 타겟으로 설정
        future_values = df['future'].values  # future 값 별도로 추출
        time_data = df['STAT_DT'].values  # 시간 데이터 별도로 추출
        
        #oht_error_data = df['M16A_3F_OHT_ERROR'].values
        X = df.drop(columns=['STAT_DT', 'future'])
        X_scaled = scaler.transform(X)  # fit_transform -> transform으로 수정

        #oht_error_data = oht_error_data.reshape(-1,1)
        #X_scaled=np.hstack((X_scaled, oht_error_data))
        #X['M16A_3F_OHT_ERROR'] = df['M16A_3F_OHT_ERROR'].values

        # 결과 누적
        X_not_scaled_all.extend(X.values)
        X_seq_all.extend(X_scaled)
        y_seq_all.extend(y.values)
        future_y_seq_all.extend(future_values)
        time_seq_all.extend(time_data)

    # 최종 시퀀스 데이터 생성
    X_not_scaled_all = np.array(X_not_scaled_all)
    X_seq_all = np.array(X_seq_all)
    y_seq_all = np.array(y_seq_all)
    future_y_seq_all = np.array(future_y_seq_all)
    time_seq_all = np.array(time_seq_all)

    print(f"최종 데이터 형태: X_seq_all.shape = {X_seq_all.shape}")
    
    return X_seq_all, y_seq_all, future_y_seq_all, time_seq_all


def make_prediction(model, X_seq_all, add_argument):
    """예측을 수행하는 함수"""
    print("=== AI 예측 수행 ===")
    
    # CPU에서 예측 수행
    with tf.device('/CPU:0'):
        # 시퀀스 데이터 reshape
        X_seq_all = np.array(X_seq_all)
        X_seq_all = X_seq_all.reshape((1, X_seq_all.shape[0], X_seq_all.shape[1]))
        
        print(f"예측 입력 데이터 형태: {X_seq_all.shape}")
        
        # 예측 수행
        predictions = np.round(model.predict(X_seq_all, verbose=0).flatten()).astype(int)
        classification_label = int(predictions >= 0.7) 
        predictions = round(predictions[0]) 
        predictions_new = predictions + add_argument

        print(f"원본 예측값: {predictions}")
        print(f"보정 후 예측값: {predictions_new}")
        print(f"위험도 판단: {classification_label} ({'위험' if classification_label == 1 else '정상'})")

    return predictions_new, classification_label


def main():
    """메인 실행 함수"""
    print("🚀 HUBROOM 반송량 예측 시스템 시작")
    print(f"TensorFlow 버전: {tf.__version__}")
    print(f"사용 디바이스: {tf.config.list_physical_devices()}")
    
    try:
        # 1. 명령행 인수 파싱
        priority, add_argument = parse_command_arguments()
        
        # 2. 데이터 전처리
        input_file_path = "HUBROOM_PIVOT_DATA.csv"
        result_list = preprocess_data(input_file_path)
        
        if not result_list:
            print("❌ 처리할 데이터가 없습니다.")
            return
        
        # 3. 모델과 스케일러 로드
        model, scaler = load_model_and_scaler()
        
        # 4. 예측 데이터 준비
        X_seq_all, y_seq_all, future_y_seq_all, time_seq_all = prepare_prediction_data(result_list, scaler)
        
        # 5. 예측 수행
        predictions_new, classification_label = make_prediction(model, X_seq_all, add_argument)
        
        # 6. 최종 결과 출력
        row = [{
            "PREDICTVAL": predictions_new,
            "JUDGEVAL": classification_label,
            "MODEL": 'LSTM_070714',
        }]
        
        print("=== 최종 예측 결과 ===")
        print(row)
        
        return row
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()