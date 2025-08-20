"""
sequence_generator_v6_parallel.py - 병렬처리 시퀀스 생성
멀티프로세싱으로 시퀀스 생성 속도 대폭 향상
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import pickle
import warnings
from datetime import datetime
import os
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
import gc
warnings.filterwarnings('ignore')

print("="*60)
print("🚀 반도체 물류 시퀀스 생성기 V6 (병렬처리)")
print(f"💻 사용 가능한 CPU 코어: {cpu_count()}개")
print("="*60)

# ============================================
# 설정
# ============================================
# 데이터 파일
DATA_FILE = '20240201_TO_202507281705.CSV'

# 데이터 설정
LOOKBACK = 100  # 과거 100분
FORECAST = 10   # 10분 후 예측

# M14 임계값
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
SAVE_PATH = './sequences_v6.npz'

# 병렬처리 설정
N_WORKERS = min(cpu_count() - 1, 8)  # 최대 8개 코어 사용
CHUNK_SIZE = 5000  # 각 프로세스가 처리할 시퀀스 수

# ============================================
# 병렬처리용 함수
# ============================================
def process_chunk(args):
    """청크 단위로 시퀀스 생성 (병렬처리용)"""
    start_idx, end_idx, df_values, feature_cols, lookback, forecast = args
    
    X_chunk = []
    y_chunk = []
    m14_chunk = []
    
    # DataFrame 재구성 (values에서)
    df = pd.DataFrame(df_values, columns=feature_cols)
    
    # 시퀀스 생성
    for i in range(start_idx, min(end_idx, len(df) - forecast)):
        if i >= lookback:
            # 시계열 데이터
            X_chunk.append(df.iloc[i-lookback:i].values)
            
            # 타겟
            y_chunk.append(df['target'].iloc[i+forecast-1])
            
            # M14 특징
            m14_chunk.append([
                df['M14AM14B'].iloc[i],
                df['M14AM10A'].iloc[i],
                df['M14AM16'].iloc[i],
                df['ratio_14B_10A'].iloc[i] if 'ratio_14B_10A' in df else 0
            ])
    
    return np.array(X_chunk, dtype=np.float32), \
           np.array(y_chunk, dtype=np.float32), \
           np.array(m14_chunk, dtype=np.float32)

# ============================================
# 데이터 전처리 및 특징 생성
# ============================================
print(f"\n📂 데이터 로딩: {DATA_FILE}")

# 데이터 로드
df = pd.read_csv(DATA_FILE)
print(f"  데이터 크기: {len(df)}행")

# 필수 컬럼 확인
if 'TOTALCNT' in df.columns:
    df['current_value'] = df['TOTALCNT']
else:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df['current_value'] = df[numeric_cols[0]]

# M14 컬럼 확인
for col in ['M14AM10A', 'M14AM14B', 'M14AM16']:
    if col not in df.columns:
        print(f"  ⚠️ {col} 없음 → 0으로 초기화")
        df[col] = 0

# 타겟 생성 (10분 후)
df['target'] = df['current_value'].shift(-10)

print("\n🔧 특징 생성 중...")

# 1. 기본 M14 특징
df['M14B_norm'] = df['M14AM14B'] / 600
df['M10A_inverse'] = (100 - df['M14AM10A']) / 100
df['M16_norm'] = df['M14AM16'] / 100

# 2. 비율 특징
df['ratio_14B_10A'] = df['M14AM14B'] / df['M14AM10A'].clip(lower=1)
df['ratio_14B_16'] = df['M14AM14B'] / df['M14AM16'].clip(lower=1)
df['ratio_10A_16'] = df['M14AM10A'] / df['M14AM16'].clip(lower=1)

# 3. 변화량 특징 (벡터화로 빠르게)
print("  변화량 특징 생성 중...")
for col in ['current_value', 'M14AM14B', 'M14AM10A', 'M14AM16']:
    if col in df.columns:
        df[f'{col}_change_5'] = df[col].diff(5)
        df[f'{col}_change_10'] = df[col].diff(10)
        df[f'{col}_ma_10'] = df[col].rolling(10, min_periods=1).mean()
        df[f'{col}_std_10'] = df[col].rolling(10, min_periods=1).std()

# 4. M14AM10A 역패턴
df['M10A_drop_5'] = -df['M14AM10A'].diff(5)
df['M10A_drop_10'] = -df['M14AM10A'].diff(10)

# 5. 급변 신호 (벡터화)
print("  신호 특징 생성 중...")
df['signal_1400'] = (df['M14AM14B'] >= M14B_THRESHOLDS[1400]).astype(float)
df['signal_1500'] = (df['M14AM14B'] >= M14B_THRESHOLDS[1500]).astype(float)
df['signal_1600'] = (df['M14AM14B'] >= M14B_THRESHOLDS[1600]).astype(float)
df['signal_1700'] = (df['M14AM14B'] >= M14B_THRESHOLDS[1700]).astype(float)

# 6. 비율 신호
df['ratio_signal_1400'] = (df['ratio_14B_10A'] >= RATIO_THRESHOLDS[1400]).astype(float)
df['ratio_signal_1500'] = (df['ratio_14B_10A'] >= RATIO_THRESHOLDS[1500]).astype(float)
df['ratio_signal_1600'] = (df['ratio_14B_10A'] >= RATIO_THRESHOLDS[1600]).astype(float)
df['ratio_signal_1700'] = (df['ratio_14B_10A'] >= RATIO_THRESHOLDS[1700]).astype(float)

# 7. 조합 특징
df['m14b_high_m10a_low'] = ((df['M14AM14B'] >= 350) & (df['M14AM10A'] < 70)).astype(float)
df['spike_imminent'] = ((df['M14AM14B'] >= 400) | (df['ratio_14B_10A'] >= 5)).astype(float)

# 8. 통계 특징
df['current_vs_ma'] = df['current_value'] / df['current_value_ma_10'].clip(lower=1)
df['m14b_vs_ma'] = df['M14AM14B'] / df['M14AM14B_ma_10'].clip(lower=1)

# 특징 컬럼 선택
exclude_cols = ['TIME', 'CURRTIME', 'TOTALCNT']
feature_cols = [col for col in df.columns if col not in exclude_cols]

# 결측치 처리
df = df.fillna(0)
df = df.dropna(subset=['target'])

print(f"  생성된 특징 수: {len(feature_cols)}개")

# ============================================
# 병렬 시퀀스 생성
# ============================================
print(f"\n📊 병렬 시퀀스 생성 중... (워커: {N_WORKERS}개)")

total_sequences = len(df) - LOOKBACK - FORECAST
print(f"  총 {total_sequences:,}개 시퀀스 생성 예정")

# 청크 인덱스 생성
chunk_indices = []
for i in range(LOOKBACK, len(df) - FORECAST, CHUNK_SIZE):
    chunk_indices.append((
        i, 
        min(i + CHUNK_SIZE, len(df) - FORECAST)
    ))

print(f"  청크 수: {len(chunk_indices)}개")

# DataFrame을 numpy array로 변환 (메모리 효율)
df_values = df[feature_cols].values

# 병렬처리 인자 준비
process_args = [
    (start, end, df_values, feature_cols, LOOKBACK, FORECAST)
    for start, end in chunk_indices
]

# 병렬 처리 실행
print("\n⚡ 병렬 처리 시작...")
start_time = datetime.now()

with Pool(processes=N_WORKERS) as pool:
    results = pool.map(process_chunk, process_args)

# 결과 합치기
print("\n📦 결과 병합 중...")
X_list = [r[0] for r in results if len(r[0]) > 0]
y_list = [r[1] for r in results if len(r[1]) > 0]
m14_list = [r[2] for r in results if len(r[2]) > 0]

X = np.concatenate(X_list, axis=0)
y = np.concatenate(y_list, axis=0)
m14_features = np.concatenate(m14_list, axis=0)

# 메모리 정리
del X_list, y_list, m14_list, results
gc.collect()

elapsed_time = datetime.now() - start_time
print(f"  병렬 처리 시간: {elapsed_time}")

print(f"\n✅ 시퀀스 생성 완료!")
print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")
print(f"  m14_features shape: {m14_features.shape}")
print(f"  메모리 사용: {(X.nbytes + y.nbytes + m14_features.nbytes) / 1024**3:.2f}GB")

# ============================================
# 스케일링 (병렬처리)
# ============================================
print("\n📏 데이터 스케일링 (병렬)...")

def scale_feature(args):
    """특징 하나를 스케일링 (병렬처리용)"""
    feature_idx, feature_data = args
    scaler = RobustScaler()
    feature_flat = feature_data.reshape(-1, 1)
    scaler.fit(feature_flat)
    scaled = scaler.transform(feature_flat).reshape(feature_data.shape)
    return feature_idx, scaled, scaler

# 병렬 스케일링
X_scaled = X.copy()
scalers = {}

# 특징별로 분리
feature_data_list = [(i, X[:, :, i]) for i in range(X.shape[2])]

# 병렬 처리
with Pool(processes=N_WORKERS) as pool:
    scaling_results = pool.map(scale_feature, feature_data_list)

# 결과 적용
for feature_idx, scaled_data, scaler in scaling_results:
    X_scaled[:, :, feature_idx] = scaled_data
    scalers[f'feature_{feature_idx}'] = scaler
    
    if (feature_idx + 1) % 10 == 0:
        print(f"  {feature_idx+1}/{X.shape[2]} 특징 스케일링 완료")

print("  ✅ 스케일링 완료")

# ============================================
# 저장
# ============================================
print(f"\n💾 시퀀스 저장 중...")

# 압축 저장
np.savez_compressed(
    SAVE_PATH,
    X=X_scaled,
    y=y,
    m14_features=m14_features,
    feature_names=feature_cols
)

# 스케일러 저장
with open('./scalers_v6.pkl', 'wb') as f:
    pickle.dump(scalers, f)

print(f"  ✅ 저장 완료: {SAVE_PATH}")
print(f"  ✅ 스케일러 저장: ./scalers_v6.pkl")

# ============================================
# 데이터 통계
# ============================================
print("\n📊 데이터 통계:")
print(f"  타겟값 범위: {y.min():.0f} ~ {y.max():.0f}")
print(f"  1400+ 비율: {(y >= 1400).mean():.1%} ({(y >= 1400).sum():,}개)")
print(f"  1500+ 비율: {(y >= 1500).mean():.1%} ({(y >= 1500).sum():,}개)")
print(f"  1600+ 비율: {(y >= 1600).mean():.1%} ({(y >= 1600).sum():,}개)")
print(f"  1700+ 비율: {(y >= 1700).mean():.1%} ({(y >= 1700).sum():,}개)")

total_time = datetime.now() - start_time
print("\n" + "="*60)
print("✅ 시퀀스 생성 완료!")
print(f"⏱️ 총 소요 시간: {total_time}")
print(f"🚀 속도 향상: 약 {N_WORKERS}배 빠름")
print("💡 다음 단계: train_v6.py 실행")
print("="*60)