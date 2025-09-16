"""
sequence_aware_data_reducer.py - 시퀀스 학습용 데이터 축소
100분 연속성을 유지하면서 효율적으로 데이터 축소
"""

import pandas as pd
import numpy as np
from datetime import datetime

# ============================================
# 설정 (조정 가능)
# ============================================
# 파일 경로
INPUT_FILE = './data/20240201_TO_202507281705.CSV'
OUTPUT_FILE = './data/reduced_for_sequence.csv'

# 시퀀스 설정 (V6과 동일)
LOOKBACK = 100  # 과거 100분
FORECAST = 10   # 10분 후 예측

# 데이터 축소 설정
REDUCTION_CONFIG = {
    'target_sequences': 50000,  # 목표 시퀀스 수 (약 5만개)
    'min_segment_length': 200,  # 최소 연속 구간 (100+10+여유)
    'keep_important_patterns': True,  # 중요 패턴 우선 보존
    'sampling_method': 'segment'  # 'segment' or 'sliding'
}

# 중요 패턴 (물류 급증 구간)
IMPORTANT_THRESHOLDS = {
    'high_totalcnt': 1500,  # 높은 물류량
    'high_m14b': 400,       # M14AM14B 높은 값
    'spike_change': 100     # 급격한 변화
}

# ============================================
# 연속 구간 찾기
# ============================================
def find_continuous_segments(df):
    """연속된 데이터 구간 찾기"""
    print("\n🔍 연속 구간 분석 중...")
    
    # 시간 컬럼 처리
    time_col = 'TIME' if 'TIME' in df.columns else 'CURRTIME'
    df['datetime'] = pd.to_datetime(df[time_col], format='%Y%m%d%H%M', errors='coerce')
    
    # 시간 차이 계산 (분 단위)
    df['time_diff'] = df['datetime'].diff().dt.total_seconds() / 60
    
    # 연속 구간 찾기 (5분 이상 차이나면 끊김)
    df['segment_id'] = (df['time_diff'] > 5).cumsum()
    
    # 각 구간의 크기
    segment_sizes = df.groupby('segment_id').size()
    valid_segments = segment_sizes[segment_sizes >= REDUCTION_CONFIG['min_segment_length']]
    
    print(f"  • 전체 구간 수: {len(segment_sizes)}")
    print(f"  • 유효 구간 수 (>={REDUCTION_CONFIG['min_segment_length']}분): {len(valid_segments)}")
    print(f"  • 최대 구간 크기: {segment_sizes.max()}분")
    print(f"  • 평균 구간 크기: {segment_sizes.mean():.0f}분")
    
    return df, valid_segments

# ============================================
# 중요 패턴 점수 계산
# ============================================
def calculate_importance_score(segment_df):
    """각 구간의 중요도 점수 계산"""
    score = 0
    
    # 1. 높은 TOTALCNT
    if 'TOTALCNT' in segment_df.columns:
        high_total = (segment_df['TOTALCNT'] >= IMPORTANT_THRESHOLDS['high_totalcnt']).sum()
        score += high_total * 10
        
        # 평균 TOTALCNT도 고려
        avg_total = segment_df['TOTALCNT'].mean()
        score += (avg_total / 100)
    
    # 2. 높은 M14AM14B
    if 'M14AM14B' in segment_df.columns:
        high_m14b = (segment_df['M14AM14B'] >= IMPORTANT_THRESHOLDS['high_m14b']).sum()
        score += high_m14b * 5
        
        # M14B/M14A 비율
        if 'M14AM10A' in segment_df.columns:
            segment_df['temp_ratio'] = segment_df['M14AM14B'] / segment_df['M14AM10A'].clip(lower=1)
            high_ratio = (segment_df['temp_ratio'] >= 5).sum()
            score += high_ratio * 8
    
    # 3. 변동성 (급격한 변화)
    if 'TOTALCNT' in segment_df.columns:
        changes = segment_df['TOTALCNT'].diff().abs()
        spikes = (changes >= IMPORTANT_THRESHOLDS['spike_change']).sum()
        score += spikes * 3
        
        # 표준편차 (변동성)
        std_value = segment_df['TOTALCNT'].std()
        score += (std_value / 10)
    
    return score

# ============================================
# 세그먼트 기반 샘플링
# ============================================
def segment_based_sampling(df, valid_segments):
    """연속 구간 단위로 샘플링"""
    print("\n📊 세그먼트 기반 샘플링 중...")
    
    target_sequences = REDUCTION_CONFIG['target_sequences']
    segments_needed = target_sequences // 100  # 대략적인 필요 세그먼트 수
    
    segment_scores = {}
    
    # 각 세그먼트의 중요도 계산
    for seg_id in valid_segments.index:
        segment_df = df[df['segment_id'] == seg_id]
        
        # 중요 패턴 보존 옵션
        if REDUCTION_CONFIG['keep_important_patterns']:
            score = calculate_importance_score(segment_df)
        else:
            score = len(segment_df)  # 단순히 길이만
        
        segment_scores[seg_id] = score
    
    # 중요도 순으로 정렬
    sorted_segments = sorted(segment_scores.items(), key=lambda x: x[1], reverse=True)
    
    # 상위 세그먼트 선택
    selected_segments = []
    total_rows = 0
    
    for seg_id, score in sorted_segments:
        segment_size = valid_segments[seg_id]
        
        # 목표 크기에 도달하면 중단
        if total_rows + segment_size > target_sequences * 1.5:  # 여유 있게
            break
        
        selected_segments.append(seg_id)
        total_rows += segment_size
        
        if len(selected_segments) >= segments_needed:
            break
    
    print(f"  • 선택된 세그먼트: {len(selected_segments)}개")
    print(f"  • 예상 시퀀스 수: ~{total_rows - LOOKBACK - FORECAST:,}개")
    
    # 선택된 세그먼트의 데이터만 추출
    reduced_df = df[df['segment_id'].isin(selected_segments)].copy()
    
    return reduced_df

# ============================================
# 슬라이딩 윈도우 샘플링
# ============================================
def sliding_window_sampling(df, valid_segments):
    """슬라이딩 윈도우 방식으로 샘플링"""
    print("\n📊 슬라이딩 윈도우 샘플링 중...")
    
    target_sequences = REDUCTION_CONFIG['target_sequences']
    
    # 각 세그먼트에서 균등하게 샘플링
    sampled_indices = []
    sequences_per_segment = target_sequences // len(valid_segments)
    
    for seg_id in valid_segments.index:
        segment_df = df[df['segment_id'] == seg_id]
        segment_indices = segment_df.index.tolist()
        
        # 이 세그먼트에서 가능한 시퀀스 수
        max_sequences = len(segment_indices) - LOOKBACK - FORECAST
        
        if max_sequences <= 0:
            continue
        
        # 샘플링 간격 계산
        n_samples = min(sequences_per_segment, max_sequences)
        step = max(1, max_sequences // n_samples)
        
        # 시작 지점들 선택
        for i in range(LOOKBACK, len(segment_indices) - FORECAST, step):
            # 시퀀스에 필요한 인덱스들 (100분 + 10분)
            sequence_indices = segment_indices[i-LOOKBACK:i+FORECAST]
            sampled_indices.extend(sequence_indices)
            
            if len(set(sampled_indices)) // (LOOKBACK + FORECAST) >= target_sequences:
                break
        
        if len(set(sampled_indices)) // (LOOKBACK + FORECAST) >= target_sequences:
            break
    
    # 중복 제거하고 정렬
    sampled_indices = sorted(list(set(sampled_indices)))
    reduced_df = df.loc[sampled_indices].copy()
    
    print(f"  • 샘플링된 데이터: {len(reduced_df):,}행")
    print(f"  • 예상 시퀀스 수: ~{(len(reduced_df) - LOOKBACK - FORECAST):,}개")
    
    return reduced_df

# ============================================
# 데이터 정리 및 검증
# ============================================
def clean_and_validate(df):
    """데이터 정리 및 검증"""
    print("\n✅ 데이터 정리 및 검증 중...")
    
    # 임시 컬럼 제거
    temp_cols = ['datetime', 'time_diff', 'segment_id', 'temp_ratio']
    for col in temp_cols:
        if col in df.columns:
            df = df.drop(col, axis=1)
    
    # 인덱스 리셋
    df = df.reset_index(drop=True)
    
    # 검증
    print(f"  • 최종 행 수: {len(df):,}")
    print(f"  • 컬럼 수: {len(df.columns)}")
    
    # 시퀀스 생성 가능 확인
    if len(df) >= LOOKBACK + FORECAST:
        max_sequences = len(df) - LOOKBACK - FORECAST
        print(f"  • 생성 가능한 시퀀스: {max_sequences:,}개")
    else:
        print(f"  ⚠️ 데이터가 너무 적음! 최소 {LOOKBACK + FORECAST}행 필요")
    
    return df

# ============================================
# 통계 비교
# ============================================
def compare_statistics(df_original, df_reduced):
    """원본과 축소 데이터 통계 비교"""
    print("\n📊 통계 비교")
    print("="*60)
    
    print(f"크기: {len(df_original):,} → {len(df_reduced):,} ({len(df_reduced)/len(df_original)*100:.1f}%)")
    
    if 'TOTALCNT' in df_original.columns:
        print(f"\nTOTALCNT 통계:")
        print(f"  • 원본 평균: {df_original['TOTALCNT'].mean():.0f}")
        print(f"  • 축소 평균: {df_reduced['TOTALCNT'].mean():.0f}")
        print(f"  • 원본 1500+: {(df_original['TOTALCNT'] >= 1500).mean():.1%}")
        print(f"  • 축소 1500+: {(df_reduced['TOTALCNT'] >= 1500).mean():.1%}")
    
    if 'M14AM14B' in df_original.columns:
        print(f"\nM14AM14B 통계:")
        print(f"  • 원본 평균: {df_original['M14AM14B'].mean():.0f}")
        print(f"  • 축소 평균: {df_reduced['M14AM14B'].mean():.0f}")
        print(f"  • 원본 400+: {(df_original['M14AM14B'] >= 400).mean():.1%}")
        print(f"  • 축소 400+: {(df_reduced['M14AM14B'] >= 400).mean():.1%}")

# ============================================
# 메인 함수
# ============================================
def main():
    print("="*60)
    print("🚀 시퀀스 학습용 데이터 축소 도구")
    print(f"📍 설정: {LOOKBACK}분 과거 → {FORECAST}분 미래 예측")
    print("="*60)
    
    # 1. 데이터 로드
    print(f"\n📂 데이터 로딩: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    df_original = df.copy()
    print(f"✅ 원본 크기: {len(df):,}행")
    
    # 2. 연속 구간 찾기
    df, valid_segments = find_continuous_segments(df)
    
    if len(valid_segments) == 0:
        print("❌ 유효한 연속 구간이 없습니다!")
        return None
    
    # 3. 샘플링 수행
    if REDUCTION_CONFIG['sampling_method'] == 'segment':
        df_reduced = segment_based_sampling(df, valid_segments)
    else:
        df_reduced = sliding_window_sampling(df, valid_segments)
    
    # 4. 데이터 정리
    df_reduced = clean_and_validate(df_reduced)
    
    # 5. 통계 비교
    compare_statistics(df_original, df_reduced)
    
    # 6. 저장
    print(f"\n💾 저장 중: {OUTPUT_FILE}")
    df_reduced.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ 저장 완료!")
    
    # 7. 최종 안내
    print("\n" + "="*60)
    print("✨ 완료! 이제 기존 V6 시퀀스 생성기 사용 가능")
    print("="*60)
    print(f"1. 데이터 파일 경로를 변경하세요:")
    print(f"   DATA_FILE = '{OUTPUT_FILE}'")
    print(f"2. 그대로 실행하면 됩니다:")
    print(f"   python V6_시퀀스_END.PY")
    print("\n💡 팁: 더 줄이려면 target_sequences를 30000으로 설정")
    
    return df_reduced

if __name__ == '__main__':
    result = main()