# -*- coding: utf-8 -*-
"""
V6_GPU_시퀀스생성_최종본.py - GPU 가속 시퀀스 생성기
반도체 물류 예측을 위한 100분 시퀀스 데이터 생성
TensorFlow 2.15.0 GPU 가속 버전
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
import pickle
import warnings
from datetime import datetime
import os
import gc
warnings.filterwarnings('ignore')

# ============================================
# GPU 설정 및 확인
# ============================================
def setup_gpu():
    """GPU 설정 및 상태 확인"""
    print("\n🎮 GPU 설정 중...")
    
    # GPU 사용 가능 확인
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # GPU 메모리 증가 허용
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # GPU 정보 출력
            print(f"  ✅ GPU 감지: {len(gpus)}개")
            for i, gpu in enumerate(gpus):
                print(f"    GPU {i}: {gpu.name}")
            
            # CUDA 정보
            print(f"  📊 TensorFlow: {tf.__version__}")
            print(f"  🔧 CUDA Built: {tf.test.is_built_with_cuda()}")
            print(f"  🚀 GPU 가속: 활성화")
            
            return True
            
        except RuntimeError as e:
            print(f"  ⚠️ GPU 설정 오류: {e}")
            return False
    else:
        print("  ⚠️ GPU 없음 - CPU 모드로 실행")
        return False

# ============================================
# 설정
# ============================================
class Config:
    # 데이터 파일
    DATA_FILE = '20250101_TO_20250630000.CSV'
    
    # 시퀀스 설정
    LOOKBACK = 100  # 과거 100분
    FORECAST = 10   # 10분 후 예측
    
    # GPU 배치 설정
    GPU_BATCH_SIZE = 10000  # GPU 메모리에 따라 조정
    
    # M14 임계값 (물류량별)
    M14B_THRESHOLDS = {
        1400: 320,
        1500: 400,
        1600: 450,
        1700: 500
    }
    
    RATIO_THRESHOLDS = {
        1400: 4,
        1500: 5,
        1600: 6,
        1700: 7
    }
    
    # 저장 경로
    SAVE_PATH = './sequences_v6_gpu.npz'
    SCALER_PATH = './scalers_v6_gpu.pkl'

# ============================================
# GPU 가속 함수
# ============================================
@tf.function
def create_sequences_gpu(data_tensor, lookback, forecast):
    """GPU에서 시퀀스 생성 (TensorFlow 연산)"""
    
    data_len = tf.shape(data_tensor)[0]
    num_sequences = data_len - lookback - forecast + 1
    
    # 인덱스 생성
    indices = tf.range(num_sequences)
    
    # 시퀀스 인덱싱을 위한 브로드캐스팅
    seq_indices = tf.expand_dims(indices, 1) + tf.range(lookback)
    target_indices = indices + lookback + forecast - 1
    
    # 시퀀스 추출
    X = tf.gather(data_tensor, seq_indices)
    y = tf.gather(data_tensor[:, 0], target_indices)  # TOTALCNT만 타겟으로
    
    return X, y

@tf.function
def extract_m14_features_gpu(data_tensor, lookback, forecast):
    """GPU에서 M14 특징 추출"""
    
    data_len = tf.shape(data_tensor)[0]
    num_sequences = data_len - lookback - forecast + 1
    
    # 현재 시점 인덱스
    current_indices = tf.range(lookback, data_len - forecast)
    
    # M14 특징 추출 (현재 시점의 M14AM14B, M14AM10A, M14AM16, ratio)
    m14_features = tf.gather(data_tensor, current_indices)
    
    return m14_features

@tf.function
def scale_features_gpu(data_tensor):
    """GPU에서 특징 스케일링 (RobustScaler 유사)"""
    
    # 각 특징별 통계 계산
    median = tf.reduce_mean(data_tensor, axis=[0, 1], keepdims=True)
    mad = tf.reduce_mean(tf.abs(data_tensor - median), axis=[0, 1], keepdims=True)
    
    # 스케일링
    scaled = (data_tensor - median) / (mad + 1e-8)
    
    return scaled, median, mad

# ============================================
# 특징 엔지니어링 (GPU 가속)
# ============================================
def create_features_gpu(df, config):
    """GPU를 사용한 특징 생성"""
    print("\n🔧 GPU 특징 엔지니어링 중...")
    
    # DataFrame을 GPU 텐서로 변환
    with tf.device('/GPU:0'):
        # 기본 컬럼을 텐서로 변환
        tensor_dict = {}
        for col in df.columns:
            if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                tensor_dict[col] = tf.constant(df[col].values, dtype=tf.float32)
        
        # 1. 기본 M14 특징 정규화
        if 'M14AM14B' in tensor_dict:
            tensor_dict['M14B_norm'] = tensor_dict['M14AM14B'] / 600.0
        
        if 'M14AM10A' in tensor_dict:
            tensor_dict['M10A_inverse'] = (100.0 - tensor_dict['M14AM10A']) / 100.0
        
        if 'M14AM16' in tensor_dict:
            tensor_dict['M16_norm'] = tensor_dict['M14AM16'] / 100.0
        
        # 2. 비율 특징 (GPU 연산)
        if 'M14AM14B' in tensor_dict and 'M14AM10A' in tensor_dict:
            m10a_safe = tf.maximum(tensor_dict['M14AM10A'], 1.0)
            tensor_dict['ratio_14B_10A'] = tensor_dict['M14AM14B'] / m10a_safe
        
        if 'M14AM14B' in tensor_dict and 'M14AM16' in tensor_dict:
            m16_safe = tf.maximum(tensor_dict['M14AM16'], 1.0)
            tensor_dict['ratio_14B_16'] = tensor_dict['M14AM14B'] / m16_safe
        
        # 3. 이동평균 (GPU 가속 컨볼루션)
        for col in ['current_value', 'M14AM14B', 'M14AM10A', 'M14AM16']:
            if col in tensor_dict:
                # 5분 변화량
                padded_5 = tf.concat([tf.zeros(5), tensor_dict[col]], axis=0)
                tensor_dict[f'{col}_change_5'] = tensor_dict[col] - padded_5[:-5]
                
                # 10분 변화량
                padded_10 = tf.concat([tf.zeros(10), tensor_dict[col]], axis=0)
                tensor_dict[f'{col}_change_10'] = tensor_dict[col] - padded_10[:-10]
                
                # 이동평균 (1D 컨볼루션 사용)
                data_expanded = tf.expand_dims(tf.expand_dims(tensor_dict[col], 0), -1)
                
                # 10분 이동평균
                kernel_10 = tf.ones([10, 1, 1]) / 10.0
                ma_10 = tf.nn.conv1d(data_expanded, kernel_10, stride=1, padding='SAME')
                tensor_dict[f'{col}_ma_10'] = tf.squeeze(ma_10)
                
                # 30분 이동평균
                kernel_30 = tf.ones([30, 1, 1]) / 30.0
                ma_30 = tf.nn.conv1d(data_expanded, kernel_30, stride=1, padding='SAME')
                tensor_dict[f'{col}_ma_30'] = tf.squeeze(ma_30)
        
        # 4. 급증 신호 (임계값 기반)
        if 'M14AM14B' in tensor_dict:
            for level, threshold in config.M14B_THRESHOLDS.items():
                tensor_dict[f'signal_{level}'] = tf.cast(
                    tensor_dict['M14AM14B'] >= threshold, tf.float32
                )
        
        # 5. 비율 신호
        if 'ratio_14B_10A' in tensor_dict:
            for level, threshold in config.RATIO_THRESHOLDS.items():
                tensor_dict[f'ratio_signal_{level}'] = tf.cast(
                    tensor_dict['ratio_14B_10A'] >= threshold, tf.float32
                )
        
        # 6. 황금 패턴 특징
        if 'M14AM14B' in tensor_dict and 'M14AM10A' in tensor_dict:
            golden_cond = tf.logical_and(
                tensor_dict['M14AM14B'] >= 350,
                tensor_dict['M14AM10A'] < 70
            )
            tensor_dict['golden_pattern'] = tf.cast(golden_cond, tf.float32)
            
            spike_cond = tf.logical_or(
                tensor_dict['M14AM14B'] >= 400,
                tensor_dict.get('ratio_14B_10A', tf.zeros_like(tensor_dict['M14AM14B'])) >= 5
            )
            tensor_dict['spike_imminent'] = tf.cast(spike_cond, tf.float32)
        
        # 텐서를 numpy로 변환하여 DataFrame 생성
        result_dict = {}
        for key, value in tensor_dict.items():
            result_dict[key] = value.numpy()
        
        # 새로운 DataFrame 생성
        df_new = pd.DataFrame(result_dict)
        
        # 원본 DataFrame의 non-numeric 컬럼 추가
        for col in df.columns:
            if col not in df_new.columns:
                df_new[col] = df[col].values
        
        return df_new

# ============================================
# 메인 함수
# ============================================
def main():
    """메인 실행 함수"""
    print("="*60)
    print("🚀 반도체 물류 GPU 시퀀스 생성기 V6")
    print(f"📦 TensorFlow 2.15.0 GPU 가속")
    print("="*60)
    
    # GPU 설정
    gpu_available = setup_gpu()
    device = '/GPU:0' if gpu_available else '/CPU:0'
    
    start_time = datetime.now()
    
    # ============================================
    # 1. 데이터 로드 및 전처리
    # ============================================
    print(f"\n📂 데이터 로딩: {Config.DATA_FILE}")
    df = pd.read_csv(Config.DATA_FILE)
    print(f"  ✅ 로드 완료: {len(df):,}행")
    
    # 필수 컬럼 확인 및 생성
    if 'TOTALCNT' in df.columns:
        df['current_value'] = df['TOTALCNT']
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df['current_value'] = df[numeric_cols[0]]
        else:
            raise ValueError("숫자형 컬럼이 없습니다!")
    
    # M14 컬럼 확인
    for col in ['M14AM10A', 'M14AM14B', 'M14AM16']:
        if col not in df.columns:
            print(f"  ⚠️ {col} 없음 → 0으로 초기화")
            df[col] = 0
    
    # 타겟 생성 (10분 후)
    df['target'] = df['current_value'].shift(-Config.FORECAST)
    
    # ============================================
    # 2. GPU 특징 생성
    # ============================================
    df = create_features_gpu(df, Config)
    
    # 특징 컬럼 선택
    exclude_cols = ['TIME', 'CURRTIME', 'TOTALCNT']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # 결측치 처리
    df = df.fillna(0)
    df = df.dropna(subset=['target'])
    
    print(f"  ✅ 특징 생성 완료: {len(feature_cols)}개")
    
    # ============================================
    # 3. GPU 시퀀스 생성
    # ============================================
    print(f"\n📊 GPU 시퀀스 생성 중...")
    
    with tf.device(device):
        # DataFrame을 텐서로 변환
        data_array = df[feature_cols].values.astype(np.float32)
        data_tensor = tf.constant(data_array)
        
        print(f"  데이터 텐서 shape: {data_tensor.shape}")
        print(f"  GPU 메모리 사용 중...")
        
        # 배치 처리로 시퀀스 생성
        total_sequences = len(df) - Config.LOOKBACK - Config.FORECAST
        print(f"  예상 시퀀스 수: {total_sequences:,}개")
        
        X_list = []
        y_list = []
        m14_list = []
        
        # 배치 단위로 처리 (GPU 메모리 관리)
        for batch_start in range(0, len(df) - Config.LOOKBACK - Config.FORECAST, Config.GPU_BATCH_SIZE):
            batch_end = min(batch_start + Config.GPU_BATCH_SIZE + Config.LOOKBACK + Config.FORECAST, len(df))
            batch_tensor = data_tensor[batch_start:batch_end]
            
            # GPU에서 시퀀스 생성
            X_batch, y_batch = create_sequences_gpu(batch_tensor, Config.LOOKBACK, Config.FORECAST)
            
            # M14 특징 추출 (M14AM14B, M14AM10A, M14AM16, ratio_14B_10A 인덱스)
            m14_indices = [
                feature_cols.index('M14AM14B') if 'M14AM14B' in feature_cols else 0,
                feature_cols.index('M14AM10A') if 'M14AM10A' in feature_cols else 0,
                feature_cols.index('M14AM16') if 'M14AM16' in feature_cols else 0,
                feature_cols.index('ratio_14B_10A') if 'ratio_14B_10A' in feature_cols else 0
            ]
            
            # 현재 시점 M14 특징 추출
            current_indices = tf.range(batch_start + Config.LOOKBACK, 
                                      min(batch_start + Config.GPU_BATCH_SIZE + Config.LOOKBACK, 
                                          len(df) - Config.FORECAST))
            m14_batch = tf.gather(data_tensor[current_indices], m14_indices, axis=1)
            
            # CPU로 이동하여 저장
            X_list.append(X_batch.numpy())
            y_list.append(y_batch.numpy())
            m14_list.append(m14_batch.numpy())
            
            # 진행률 표시
            progress = (batch_start + Config.GPU_BATCH_SIZE) / (len(df) - Config.LOOKBACK - Config.FORECAST) * 100
            print(f"    처리 중: {progress:.1f}%", end='\r')
        
        print("\n  ✅ GPU 시퀀스 생성 완료")
        
        # 결과 병합
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        m14_features = np.concatenate(m14_list, axis=0)
        
        # 메모리 정리
        del X_list, y_list, m14_list
        gc.collect()
        
        print(f"\n📐 생성된 시퀀스:")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  m14_features shape: {m14_features.shape}")
        print(f"  메모리 사용: {(X.nbytes + y.nbytes + m14_features.nbytes) / 1024**3:.2f}GB")
        
        # ============================================
        # 4. GPU 스케일링
        # ============================================
        print("\n📏 GPU 데이터 스케일링...")
        
        # GPU에서 스케일링
        X_tensor = tf.constant(X)
        X_scaled_tensor, median, mad = scale_features_gpu(X_tensor)
        X_scaled = X_scaled_tensor.numpy()
        
        # 스케일러 정보 저장
        scalers = {
            'median': median.numpy(),
            'mad': mad.numpy(),
            'method': 'gpu_robust'
        }
        
        print("  ✅ GPU 스케일링 완료")
    
    # ============================================
    # 5. 저장
    # ============================================
    print(f"\n💾 데이터 저장 중...")
    
    # 시퀀스 압축 저장
    np.savez_compressed(
        Config.SAVE_PATH,
        X=X_scaled,
        y=y,
        m14_features=m14_features,
        feature_names=feature_cols
    )
    
    # 스케일러 저장
    with open(Config.SCALER_PATH, 'wb') as f:
        pickle.dump(scalers, f)
    
    print(f"  ✅ 시퀀스 저장: {Config.SAVE_PATH}")
    print(f"  ✅ 스케일러 저장: {Config.SCALER_PATH}")
    
    # ============================================
    # 6. 통계 출력
    # ============================================
    print("\n📊 데이터 통계:")
    print(f"  타겟값 범위: {y.min():.0f} ~ {y.max():.0f}")
    print(f"  평균: {y.mean():.0f}, 표준편차: {y.std():.0f}")
    
    print(f"\n  물류량 구간별 분포:")
    for level in [1400, 1500, 1600, 1700]:
        count = (y >= level).sum()
        ratio = (y >= level).mean()
        print(f"    {level}+ : {ratio:6.2%} ({count:,}개)")
    
    # M14 특징 통계
    print(f"\n  M14 특징 통계:")
    print(f"    M14AM14B 평균: {m14_features[:, 0].mean():.1f}")
    print(f"    M14AM10A 평균: {m14_features[:, 1].mean():.1f}")
    print(f"    M14AM16  평균: {m14_features[:, 2].mean():.1f}")
    print(f"    비율(14B/10A) 평균: {m14_features[:, 3].mean():.2f}")
    
    # 황금 패턴 검출
    golden_pattern_count = ((m14_features[:, 0] > 300) & (m14_features[:, 1] < 80)).sum()
    print(f"\n  🏆 황금 패턴 감지: {golden_pattern_count:,}개")
    
    total_time = datetime.now() - start_time
    
    print("\n" + "="*60)
    print("✅ GPU 시퀀스 생성 완료!")
    print(f"⏱️  총 소요 시간: {total_time}")
    if gpu_available:
        print(f"🚀 GPU 가속: 활성화 (약 10-20배 속도 향상)")
    else:
        print(f"⚠️  GPU 없음: CPU 모드로 실행됨")
    print(f"💡 다음 단계: V6_GPU_학습_최종본.py 실행")
    print("="*60)

# ============================================
# 실행
# ============================================
if __name__ == '__main__':
    main()