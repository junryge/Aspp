"""
optimize_csv_for_v6.py - V6 시퀀스 생성기용 CSV 최적화
중요 패턴은 보존하면서 효율적으로 데이터 축소
"""

import pandas as pd
import numpy as np
from datetime import datetime
import gc

# ============================================
# 설정 (V6와 동일한 임계값)
# ============================================
# 입출력 파일
INPUT_FILE = './data/balanced_train_data.csv'  # 원본 CSV
OUTPUT_FILE = './data/optimized_for_v6.csv'    # 최적화된 CSV

# V6 설정값
LOOKBACK = 100
FORECAST = 10

# V6 임계값 (동일하게 유지)
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

# 최적화 설정
OPTIMIZATION_CONFIG = {
    'target_size': 50000,        # 목표 크기 (5만행)
    'min_segment_length': 200,   # 최소 연속 구간 (LOOKBACK + FORECAST + 여유)
    
    # 보존 우선순위
    'preserve_priority': {
        'critical': 1.0,    # 100% 보존 - TOTALCNT >= 1500 또는 M14B >= 400
        'important': 0.8,   # 80% 보존 - TOTALCNT >= 1400 또는 M14B >= 350
        'medium': 0.3,      # 30% 보존 - TOTALCNT >= 1350
        'normal': 0.1       # 10% 보존 - 나머지
    }
}

# ============================================
# CSV 최적화 함수
# ============================================
def optimize_csv_for_sequences():
    """CSV 데이터 최적화"""
    print("="*60)
    print("🚀 V6 시퀀스용 CSV 최적화")
    print("="*60)
    
    # 1. 데이터 로드
    print(f"\n📂 원본 CSV 로딩: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    original_size = len(df)
    print(f"✅ 원본 크기: {original_size:,}행")
    
    # 2. 연속성 확인
    print("\n🔍 연속 구간 분석...")
    
    # 시간 컬럼 확인
    time_col = 'TIME' if 'TIME' in df.columns else 'CURRTIME'
    if time_col in df.columns:
        df['datetime'] = pd.to_datetime(df[time_col].astype(str), 
                                       format='%Y%m%d%H%M', 
                                       errors='coerce')
        df['time_diff'] = df['datetime'].diff().dt.total_seconds() / 60
        df['segment_id'] = (df['time_diff'] > 5).cumsum()
        
        segment_sizes = df.groupby('segment_id').size()
        valid_segments = segment_sizes[segment_sizes >= OPTIMIZATION_CONFIG['min_segment_length']]
        print(f"  • 유효 세그먼트: {len(valid_segments)}개")
    else:
        df['segment_id'] = 0
        valid_segments = pd.Series([len(df)], index=[0])
    
    # 3. 중요도 계산
    print("\n📊 중요도 기반 샘플링...")
    
    # 중요도 레벨 할당
    df['importance'] = 'normal'
    
    # Critical: TOTALCNT >= 1500 또는 M14B >= 400
    critical_mask = (df['TOTALCNT'] >= 1500) | (df['M14AM14B'] >= M14B_THRESHOLDS[1500])
    df.loc[critical_mask, 'importance'] = 'critical'
    
    # Important: TOTALCNT >= 1400 또는 M14B >= 350
    important_mask = ((df['TOTALCNT'] >= 1400) | (df['M14AM14B'] >= 350)) & ~critical_mask
    df.loc[important_mask, 'importance'] = 'important'
    
    # Medium: TOTALCNT >= 1350
    medium_mask = (df['TOTALCNT'] >= 1350) & ~critical_mask & ~important_mask
    df.loc[medium_mask, 'importance'] = 'medium'
    
    # 황금 패턴 (M14B >= 300 & M10A < 80)은 무조건 critical
    golden_mask = (df['M14AM14B'] >= 300) & (df['M14AM10A'] < 80)
    df.loc[golden_mask, 'importance'] = 'critical'
    
    # 통계 출력
    importance_counts = df['importance'].value_counts()
    print(f"  • Critical: {importance_counts.get('critical', 0):,}개")
    print(f"  • Important: {importance_counts.get('important', 0):,}개")
    print(f"  • Medium: {importance_counts.get('medium', 0):,}개")
    print(f"  • Normal: {importance_counts.get('normal', 0):,}개")
    
    # 4. 샘플링
    sampled_indices = []
    
    for importance_level, preserve_ratio in OPTIMIZATION_CONFIG['preserve_priority'].items():
        level_df = df[df['importance'] == importance_level]
        
        if len(level_df) > 0:
            if preserve_ratio == 1.0:
                # 100% 보존
                sampled_indices.extend(level_df.index.tolist())
            else:
                # 비율만큼 샘플링
                n_samples = int(len(level_df) * preserve_ratio)
                if n_samples > 0:
                    samples = level_df.sample(n=n_samples, random_state=42)
                    sampled_indices.extend(samples.index.tolist())
    
    # 5. 최종 데이터 생성
    print(f"\n📦 최적화된 데이터 생성...")
    
    # 중복 제거 및 정렬
    sampled_indices = sorted(list(set(sampled_indices)))
    
    # 목표 크기 조정
    if len(sampled_indices) > OPTIMIZATION_CONFIG['target_size']:
        # 중요도 낮은 것부터 제거
        df_sampled = df.loc[sampled_indices]
        
        # importance 순서대로 정렬
        importance_order = {'critical': 0, 'important': 1, 'medium': 2, 'normal': 3}
        df_sampled['importance_order'] = df_sampled['importance'].map(importance_order)
        df_sampled = df_sampled.sort_values('importance_order')
        
        # 목표 크기만큼만 유지
        df_sampled = df_sampled.head(OPTIMIZATION_CONFIG['target_size'])
        df_optimized = df_sampled.sort_index()
    else:
        df_optimized = df.loc[sampled_indices]
    
    # 임시 컬럼 제거
    columns_to_drop = ['datetime', 'time_diff', 'segment_id', 'importance', 'importance_order']
    for col in columns_to_drop:
        if col in df_optimized.columns:
            df_optimized = df_optimized.drop(col, axis=1)
    
    # 인덱스 리셋
    df_optimized = df_optimized.reset_index(drop=True)
    
    # 6. 저장
    print(f"\n💾 최적화된 CSV 저장: {OUTPUT_FILE}")
    df_optimized.to_csv(OUTPUT_FILE, index=False)
    
    # 7. 결과 통계
    print("\n✅ 최적화 완료!")
    print("="*60)
    print(f"원본: {original_size:,}행 → 최적화: {len(df_optimized):,}행")
    print(f"축소율: {(1 - len(df_optimized)/original_size)*100:.1f}%")
    
    if 'TOTALCNT' in df_optimized.columns:
        print(f"\nTOTALCNT 분포:")
        print(f"  • 평균: {df_optimized['TOTALCNT'].mean():.0f}")
        print(f"  • 1400+: {(df_optimized['TOTALCNT'] >= 1400).sum():,}개")
        print(f"  • 1500+: {(df_optimized['TOTALCNT'] >= 1500).sum():,}개")
    
    if 'M14AM14B' in df_optimized.columns:
        print(f"\nM14AM14B 분포:")
        print(f"  • 평균: {df_optimized['M14AM14B'].mean():.0f}")
        print(f"  • 400+: {(df_optimized['M14AM14B'] >= 400).sum():,}개")
        
        # 황금 패턴
        golden = (df_optimized['M14AM14B'] >= 300) & (df_optimized['M14AM10A'] < 80)
        print(f"  • 황금 패턴: {golden.sum():,}개")
    
    # 시퀀스 생성 가능 수
    max_sequences = max(0, len(df_optimized) - LOOKBACK - FORECAST)
    print(f"\n생성 가능 시퀀스: {max_sequences:,}개")
    
    print("\n" + "="*60)
    print("💡 V6 사용법:")
    print(f"DATA_FILE = '{OUTPUT_FILE}'")
    print("나머지는 그대로 실행하면 됩니다!")
    print("="*60)
    
    return df_optimized

# ============================================
# 메인 실행
# ============================================
if __name__ == '__main__':
    df_optimized = optimize_csv_for_sequences()