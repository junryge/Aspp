"""
data_reducer_for_v6.py - V6 시퀀스 생성기 전용 데이터 축소
기존 코드 그대로 사용 가능하도록 최적화
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 설정 (V6와 동일하게 유지)
# ============================================
# 파일 경로
INPUT_FILE = './data/20240201_TO_202507281705.CSV'  # 원본 (V6와 동일)
OUTPUT_FILE = './data/20240201_TO_202507281705_reduced.CSV'  # 축소 버전

# V6 설정값 그대로
LOOKBACK = 100  # 과거 100분
FORECAST = 10   # 10분 후 예측

# M14 임계값 (V6와 동일)
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

# 축소 설정
REDUCTION_CONFIG = {
    'target_rows': 100000,      # 목표: 10만행 (원본의 약 13%)
    'method': 'smart',           # 중요 패턴 우선 보존
    'keep_critical_patterns': True,  # 핵심 패턴 100% 보존
}

# ============================================
# 메인 처리
# ============================================
def main():
    print("="*60)
    print("🚀 V6 시퀀스 생성기용 데이터 축소")
    print("="*60)
    
    # 1. 데이터 로드
    print(f"\n📂 원본 데이터 로딩: {INPUT_FILE}")
    try:
        df = pd.read_csv(INPUT_FILE)
    except:
        print("❌ 파일을 찾을 수 없습니다. 경로를 확인하세요.")
        return None
    
    original_size = len(df)
    print(f"✅ 원본 크기: {original_size:,}행")
    
    # 필수 컬럼 확인 (V6와 동일)
    required_cols = ['TOTALCNT'] if 'TOTALCNT' in df.columns else []
    m14_cols = ['M14AM10A', 'M14AM14B', 'M14AM16']
    
    for col in m14_cols:
        if col not in df.columns:
            print(f"  ⚠️ {col} 없음 → 0으로 초기화")
            df[col] = 0
    
    # 2. 시간 기반 연속성 확인
    print("\n🔍 연속 구간 분석 중...")
    
    # TIME 또는 CURRTIME 컬럼 확인
    time_col = None
    if 'TIME' in df.columns:
        time_col = 'TIME'
    elif 'CURRTIME' in df.columns:
        time_col = 'CURRTIME'
    
    if time_col:
        # 시간 순서로 정렬
        df = df.sort_values(time_col).reset_index(drop=True)
        
        # datetime 변환
        df['datetime'] = pd.to_datetime(df[time_col].astype(str), 
                                       format='%Y%m%d%H%M', 
                                       errors='coerce')
        
        # 시간 차이 계산
        df['time_diff'] = df['datetime'].diff().dt.total_seconds() / 60
        
        # 5분 이상 끊어진 구간 찾기
        df['segment_id'] = (df['time_diff'] > 5).cumsum()
        
        # 세그먼트 크기
        segment_sizes = df.groupby('segment_id').size()
        min_size = LOOKBACK + FORECAST + 10  # 여유 있게
        valid_segments = segment_sizes[segment_sizes >= min_size]
        
        print(f"  • 전체 세그먼트: {len(segment_sizes)}개")
        print(f"  • 유효 세그먼트 (>={min_size}분): {len(valid_segments)}개")
        print(f"  • 최대 세그먼트: {segment_sizes.max():,}분")
    else:
        # 시간 컬럼이 없으면 전체를 하나로
        df['segment_id'] = 0
        valid_segments = pd.Series([len(df)], index=[0])
        print("  • TIME 컬럼 없음 - 전체를 하나의 세그먼트로 처리")
    
    # 3. 중요도 점수 계산
    print("\n📊 중요 패턴 분석 중...")
    
    segment_scores = {}
    critical_indices = []  # 반드시 보존할 인덱스
    
    for seg_id in valid_segments.index:
        seg_df = df[df['segment_id'] == seg_id]
        score = 0
        
        # TOTALCNT 기반 점수
        if 'TOTALCNT' in seg_df.columns:
            # 1500 이상인 행들은 반드시 보존
            critical_mask = seg_df['TOTALCNT'] >= 1500
            critical_indices.extend(seg_df[critical_mask].index.tolist())
            
            # 평균값 점수
            avg_total = seg_df['TOTALCNT'].mean()
            score += avg_total / 10
            
            # 변동성 점수
            std_total = seg_df['TOTALCNT'].std()
            score += std_total / 20
        
        # M14AM14B 기반 점수
        if 'M14AM14B' in seg_df.columns:
            # 400 이상인 행들은 반드시 보존
            critical_m14b = seg_df['M14AM14B'] >= M14B_THRESHOLDS[1500]
            critical_indices.extend(seg_df[critical_m14b].index.tolist())
            
            # 임계값별 가중치
            for threshold_val, m14b_threshold in M14B_THRESHOLDS.items():
                count = (seg_df['M14AM14B'] >= m14b_threshold).sum()
                score += count * (threshold_val / 1000)
        
        # 황금 패턴 (M14B 높고 M10A 낮음)
        if 'M14AM14B' in seg_df.columns and 'M14AM10A' in seg_df.columns:
            golden_pattern = (seg_df['M14AM14B'] >= 300) & (seg_df['M14AM10A'] < 80)
            critical_indices.extend(seg_df[golden_pattern].index.tolist())
            score += golden_pattern.sum() * 20
            
            # 비율 패턴
            seg_df['temp_ratio'] = seg_df['M14AM14B'] / seg_df['M14AM10A'].clip(lower=1)
            for threshold_val, ratio_threshold in RATIO_THRESHOLDS.items():
                ratio_count = (seg_df['temp_ratio'] >= ratio_threshold).sum()
                score += ratio_count * (threshold_val / 1000)
        
        # 세그먼트 길이 보너스
        score += len(seg_df) / 1000
        
        segment_scores[seg_id] = score
    
    # 중복 제거
    critical_indices = list(set(critical_indices))
    print(f"  • 필수 보존 행: {len(critical_indices):,}개")
    
    # 4. 스마트 샘플링
    print("\n🎯 스마트 샘플링 시작...")
    
    target_rows = REDUCTION_CONFIG['target_rows']
    
    # 필수 보존 데이터
    if REDUCTION_CONFIG['keep_critical_patterns']:
        sampled_indices = critical_indices.copy()
        remaining_quota = target_rows - len(sampled_indices)
        print(f"  • 필수 패턴 보존: {len(sampled_indices):,}개")
    else:
        sampled_indices = []
        remaining_quota = target_rows
    
    # 세그먼트별로 추가 샘플링
    if remaining_quota > 0:
        # 점수 순으로 정렬
        sorted_segments = sorted(segment_scores.items(), key=lambda x: x[1], reverse=True)
        
        for seg_id, score in sorted_segments:
            if remaining_quota <= 0:
                break
            
            seg_df = df[df['segment_id'] == seg_id]
            
            # 이미 선택된 것 제외
            unselected = seg_df[~seg_df.index.isin(sampled_indices)]
            
            if len(unselected) == 0:
                continue
            
            # 이 세그먼트에서 샘플링할 개수
            seg_quota = min(remaining_quota, len(unselected) // 2)
            
            if seg_quota > 0:
                # 균등 간격 샘플링
                step = max(1, len(unselected) // seg_quota)
                seg_samples = unselected.iloc[::step].index.tolist()
                sampled_indices.extend(seg_samples[:seg_quota])
                remaining_quota -= len(seg_samples[:seg_quota])
        
        print(f"  • 추가 샘플링: {target_rows - remaining_quota:,}개")
    
    # 5. 최종 데이터 생성
    print("\n📦 최종 데이터 생성 중...")
    
    # 선택된 인덱스로 데이터 추출
    sampled_indices = sorted(list(set(sampled_indices)))
    df_reduced = df.loc[sampled_indices].copy()
    
    # 임시 컬럼 제거
    temp_cols = ['datetime', 'time_diff', 'segment_id', 'temp_ratio']
    for col in temp_cols:
        if col in df_reduced.columns:
            df_reduced = df_reduced.drop(col, axis=1)
    
    # 인덱스 리셋
    df_reduced = df_reduced.reset_index(drop=True)
    
    # 6. 검증
    print("\n✅ 결과 검증")
    print("="*60)
    
    print(f"축소 결과:")
    print(f"  • 원본: {original_size:,}행")
    print(f"  • 축소: {len(df_reduced):,}행")
    print(f"  • 축소율: {(1 - len(df_reduced)/original_size)*100:.1f}%")
    
    # 시퀀스 생성 가능 확인
    max_sequences = max(0, len(df_reduced) - LOOKBACK - FORECAST)
    print(f"\n생성 가능 시퀀스: {max_sequences:,}개")
    
    # 중요 패턴 보존률
    if 'TOTALCNT' in df_reduced.columns:
        orig_1500 = (df['TOTALCNT'] >= 1500).sum()
        redu_1500 = (df_reduced['TOTALCNT'] >= 1500).sum()
        if orig_1500 > 0:
            print(f"\nTOTALCNT ≥ 1500 보존률: {redu_1500/orig_1500*100:.1f}%")
    
    if 'M14AM14B' in df_reduced.columns:
        orig_400 = (df['M14AM14B'] >= 400).sum()
        redu_400 = (df_reduced['M14AM14B'] >= 400).sum()
        if orig_400 > 0:
            print(f"M14AM14B ≥ 400 보존률: {redu_400/orig_400*100:.1f}%")
    
    # 7. 저장
    print(f"\n💾 저장 중: {OUTPUT_FILE}")
    df_reduced.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ 저장 완료!")
    
    # 8. 사용 안내
    print("\n" + "="*60)
    print("✨ 완료! V6 시퀀스 생성기 사용 방법:")
    print("="*60)
    print("\n1. V6 코드에서 경로만 변경:")
    print(f"   DATA_FILE = '{OUTPUT_FILE}'")
    print("\n2. 나머지는 그대로 실행:")
    print("   python sequence_generator_v6_parallel.py")
    print("\n💡 추가 옵션:")
    print("  • 더 작게: target_rows = 50000")
    print("  • 더 크게: target_rows = 200000")
    print("  • 모든 패턴: keep_critical_patterns = True (권장)")
    
    return df_reduced

if __name__ == '__main__':
    result = main()