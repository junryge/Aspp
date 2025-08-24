"""
V6_시퀀스생성_최종본.py - 병렬처리 시퀀스 생성기
반도체 물류 예측을 위한 100분 시퀀스 데이터 생성
TensorFlow 2.15.0 호환
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import pickle
import warnings
from datetime import datetime
import os
from multiprocessing import Pool, cpu_count
import gc
warnings.filterwarnings('ignore')

# ============================================
# 설정
# ============================================
class Config:
    # 데이터 파일
    DATA_FILE = '20240201_TO_202507281705.CSV'
    
    # 시퀀스 설정
    LOOKBACK = 100  # 과거 100분
    FORECAST = 10   # 10분 후 예측
    
    # M14 임계값 (물류량별)
    M14B_THRESHOLDS = {
        1400: 320,  # 1400개 예측시 M14AM14B 임계값
        1500: 400,
        1600: 450,
        1700: 500
    }
    
    RATIO_THRESHOLDS = {
        1400: 4,  # M14B/M10A 비율 임계값
        1500: 5,
        1600: 6,
        1700: 7
    }
    
    # 저장 경로
    SAVE_PATH = './sequences_v6.npz'
    SCALER_PATH = './scalers_v6.pkl'
    
    # 병렬처리 설정
    N_WORKERS = min(cpu_count() - 1, 8)  # 최대 8개 코어
    CHUNK_SIZE = 5000  # 청크당 시퀀스 수

# ============================================
# 병렬처리 함수
# ============================================
def process_chunk(args):
    """청크 단위로 시퀀스 생성 (병렬처리용)"""
    start_idx, end_idx, df_values, feature_cols, lookback, forecast = args
    
    X_chunk = []
    y_chunk = []
    m14_chunk = []
    
    # DataFrame 재구성
    df = pd.DataFrame(df_values, columns=feature_cols)
    
    # 시퀀스 생성
    for i in range(start_idx, min(end_idx, len(df) - forecast)):
        if i >= lookback:
            # 시계열 데이터 (100분)
            X_chunk.append(df.iloc[i-lookback:i].values)
            
            # 타겟 (10분 후 TOTALCNT)
            y_chunk.append(df['target'].iloc[i+forecast-1])
            
            # M14 특징 (현재 시점)
            m14_chunk.append([
                df['M14AM14B'].iloc[i],
                df['M14AM10A'].iloc[i],
                df['M14AM16'].iloc[i],
                df['ratio_14B_10A'].iloc[i] if 'ratio_14B_10A' in df else 0
            ])
    
    return (np.array(X_chunk, dtype=np.float32),
            np.array(y_chunk, dtype=np.float32),
            np.array(m14_chunk, dtype=np.float32))

def scale_feature(args):
    """특징별 스케일링 (병렬처리용)"""
    feature_idx, feature_data = args
    scaler = RobustScaler()
    feature_flat = feature_data.reshape(-1, 1)
    scaler.fit(feature_flat)
    scaled = scaler.transform(feature_flat).reshape(feature_data.shape)
    return feature_idx, scaled, scaler

# ============================================
# 특징 엔지니어링
# ============================================
def create_features(df, config):
    """다양한 특징 생성"""
    print("\n🔧 특징 엔지니어링 중...")
    
    # 1. 기본 M14 특징 정규화
    df['M14B_norm'] = df['M14AM14B'] / 600
    df['M10A_inverse'] = (100 - df['M14AM10A']) / 100
    df['M16_norm'] = df['M14AM16'] / 100
    
    # 2. 비율 특징
    df['ratio_14B_10A'] = df['M14AM14B'] / df['M14AM10A'].clip(lower=1)
    df['ratio_14B_16'] = df['M14AM14B'] / df['M14AM16'].clip(lower=1)
    df['ratio_10A_16'] = df['M14AM10A'] / df['M14AM16'].clip(lower=1)
    
    # 3. 시계열 특징 (변화량, 이동평균, 표준편차)
    for col in ['current_value', 'M14AM14B', 'M14AM10A', 'M14AM16']:
        if col in df.columns:
            # 변화량
            df[f'{col}_change_5'] = df[col].diff(5)
            df[f'{col}_change_10'] = df[col].diff(10)
            
            # 이동평균
            df[f'{col}_ma_10'] = df[col].rolling(10, min_periods=1).mean()
            df[f'{col}_ma_30'] = df[col].rolling(30, min_periods=1).mean()
            
            # 표준편차
            df[f'{col}_std_10'] = df[col].rolling(10, min_periods=1).std()
    
    # 4. M14AM10A 역패턴 (하락 감지)
    df['M10A_drop_5'] = -df['M14AM10A'].diff(5)
    df['M10A_drop_10'] = -df['M14AM10A'].diff(10)
    
    # 5. 급증 신호 (임계값 기반)
    for level, threshold in config.M14B_THRESHOLDS.items():
        df[f'signal_{level}'] = (df['M14AM14B'] >= threshold).astype(float)
    
    # 6. 비율 신호
    for level, threshold in config.RATIO_THRESHOLDS.items():
        df[f'ratio_signal_{level}'] = (df['ratio_14B_10A'] >= threshold).astype(float)
    
    # 7. 황금 패턴 특징
    df['golden_pattern'] = ((df['M14AM14B'] >= 350) & (df['M14AM10A'] < 70)).astype(float)
    df['spike_imminent'] = ((df['M14AM14B'] >= 400) | (df['ratio_14B_10A'] >= 5)).astype(float)
    
    # 8. 통계적 특징
    df['current_vs_ma'] = df['current_value'] / df['current_value_ma_10'].clip(lower=1)
    df['m14b_vs_ma'] = df['M14AM14B'] / df['M14AM14B_ma_10'].clip(lower=1)
    
    # 9. 시간 특징 (옵션)
    if 'TIME' in df.columns:
        try:
            df['hour'] = pd.to_datetime(df['TIME']).dt.hour
            df['minute'] = pd.to_datetime(df['TIME']).dt.minute
            df['is_peak'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(float)
        except:
            pass
    
    return df

# ============================================
# 메인 함수
# ============================================
def main():
    """메인 실행 함수"""
    print("="*60)
    print("🚀 반도체 물류 시퀀스 생성기 V6 최종본")
    print(f"📦 TensorFlow 2.15.0 호환")
    print(f"💻 CPU 코어: {cpu_count()}개 (사용: {Config.N_WORKERS}개)")
    print("="*60)
    
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
    # 2. 특징 생성
    # ============================================
    df = create_features(df, Config)
    
    # 특징 컬럼 선택
    exclude_cols = ['TIME', 'CURRTIME', 'TOTALCNT']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # 결측치 처리
    df = df.fillna(0)
    df = df.dropna(subset=['target'])
    
    print(f"  ✅ 특징 생성 완료: {len(feature_cols)}개")
    
    # ============================================
    # 3. 병렬 시퀀스 생성
    # ============================================
    print(f"\n📊 병렬 시퀀스 생성 중...")
    
    total_sequences = len(df) - Config.LOOKBACK - Config.FORECAST
    print(f"  예상 시퀀스 수: {total_sequences:,}개")
    
    # 청크 인덱스 생성
    chunk_indices = []
    for i in range(Config.LOOKBACK, len(df) - Config.FORECAST, Config.CHUNK_SIZE):
        chunk_indices.append((
            i, 
            min(i + Config.CHUNK_SIZE, len(df) - Config.FORECAST)
        ))
    
    print(f"  청크 수: {len(chunk_indices)}개")
    
    # DataFrame을 numpy array로 변환
    df_values = df[feature_cols].values
    
    # 병렬처리 인자 준비
    process_args = [
        (start, end, df_values, feature_cols, Config.LOOKBACK, Config.FORECAST)
        for start, end in chunk_indices
    ]
    
    # 병렬 처리 실행
    print("\n⚡ 병렬 처리 시작...")
    chunk_start = datetime.now()
    
    with Pool(processes=Config.N_WORKERS) as pool:
        results = pool.map(process_chunk, process_args)
    
    # 결과 병합
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
    
    chunk_time = datetime.now() - chunk_start
    print(f"  ✅ 병렬 처리 완료: {chunk_time}")
    
    print(f"\n📐 생성된 시퀀스:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  m14_features shape: {m14_features.shape}")
    print(f"  메모리 사용: {(X.nbytes + y.nbytes + m14_features.nbytes) / 1024**3:.2f}GB")
    
    # ============================================
    # 4. 스케일링 (병렬)
    # ============================================
    print("\n📏 데이터 스케일링 (병렬)...")
    
    X_scaled = X.copy()
    scalers = {}
    
    # 특징별로 분리
    feature_data_list = [(i, X[:, :, i]) for i in range(X.shape[2])]
    
    # 병렬 스케일링
    with Pool(processes=Config.N_WORKERS) as pool:
        scaling_results = pool.map(scale_feature, feature_data_list)
    
    # 결과 적용
    for feature_idx, scaled_data, scaler in scaling_results:
        X_scaled[:, :, feature_idx] = scaled_data
        scalers[f'feature_{feature_idx}'] = scaler
    
    print("  ✅ 스케일링 완료")
    
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
    print("✅ 시퀀스 생성 완료!")
    print(f"⏱️  총 소요 시간: {total_time}")
    print(f"🚀 속도 향상: 약 {Config.N_WORKERS}배")
    print(f"💡 다음 단계: V6_학습_최종본.py 실행")
    print("="*60)

# ============================================
# Windows 호환 메인 가드
# ============================================
if __name__ == '__main__':
    # Windows에서 multiprocessing 사용시 필수
    main()