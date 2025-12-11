#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔬 시퀀스 길이 최적화 검증
280분 vs 100분 vs 60분 vs 30분
"""

import pandas as pd
import numpy as np

CSV_FILE = "YOUR_CSV_FILE.csv"  # ← 실제 파일 경로!

print("="*70)
print("🔬 시퀀스 길이 최적화 검증")
print("="*70)

df = pd.read_csv(CSV_FILE, on_bad_lines='skip')
print(f"총 {len(df):,}행")

# 컬럼 지정
totalcnt_col = 'TOTALCNT'
m14b_col = 'M14AM14B'
m14bsum_col = 'M14AM14BSUM'
qcreated_col = 'M14.QUE.ALL.CURRENTQCREATED'
qcompleted_col = 'M14.QUE.ALL.CURRENTQCOMPLETED'

# queue_gap 계산
df['queue_gap'] = df[qcreated_col] - df[qcompleted_col]

# 10분 후 타겟
PRED_MINUTES = 10
df['future_totalcnt'] = df[totalcnt_col].shift(-PRED_MINUTES)
df['future_1700'] = df['future_totalcnt'] >= 1700

# ============================================
# 시퀀스 길이별 Feature 생성 함수
# ============================================
def create_features(df, idx, seq_len):
    """간단한 Feature 생성 (시퀀스 길이별)"""
    if idx < seq_len:
        return None
    
    seq_total = df[totalcnt_col].iloc[idx-seq_len:idx].values
    seq_m14b = df[m14b_col].iloc[idx-seq_len:idx].values
    seq_m14bsum = df[m14bsum_col].iloc[idx-seq_len:idx].values
    seq_gap = df['queue_gap'].iloc[idx-seq_len:idx].values
    
    features = {
        'total_current': seq_total[-1],
        'total_mean': np.mean(seq_total),
        'total_max': np.max(seq_total),
        'total_slope': np.polyfit(np.arange(seq_len), seq_total, 1)[0],
        'm14b_current': seq_m14b[-1],
        'm14b_mean': np.mean(seq_m14b),
        'm14b_max': np.max(seq_m14b),
        'm14bsum_current': seq_m14bsum[-1],
        'm14bsum_mean': np.mean(seq_m14bsum),
        'gap_current': seq_gap[-1],
        'gap_mean': np.mean(seq_gap),
        'gap_max': np.max(seq_gap),
        'gap_slope': np.polyfit(np.arange(seq_len), seq_gap, 1)[0],
    }
    return features

# ============================================
# 시퀀스 길이별 테스트
# ============================================
print("\n" + "="*70)
print("📊 시퀀스 길이별 패턴 감지 테스트")
print("="*70)

seq_lengths = [30, 60, 100, 150, 200, 280]

for seq_len in seq_lengths:
    print(f"\n🔍 시퀀스 길이: {seq_len}분")
    
    # 유효 데이터 범위
    valid_start = seq_len
    valid_end = len(df) - PRED_MINUTES
    
    total_samples = valid_end - valid_start
    actual_1700 = df['future_1700'].iloc[valid_start:valid_end].sum()
    
    # 패턴 체크 (현재 시점 기준 - 시퀀스 마지막 값)
    triple_check_count = 0
    triple_check_hit = 0
    
    for idx in range(valid_start, valid_end):
        # 현재값 (시퀀스 끝)
        current_m14b = df[m14b_col].iloc[idx-1]
        current_m14bsum = df[m14bsum_col].iloc[idx-1]
        current_gap = df['queue_gap'].iloc[idx-1]
        future_1700 = df['future_1700'].iloc[idx]
        
        # Triple Check
        if current_m14b > 520 and current_m14bsum > 600 and current_gap > 250:
            triple_check_count += 1
            if future_1700:
                triple_check_hit += 1
    
    precision = triple_check_hit / triple_check_count * 100 if triple_check_count > 0 else 0
    recall = triple_check_hit / actual_1700 * 100 if actual_1700 > 0 else 0
    
    print(f"  유효 샘플: {total_samples:,}개")
    print(f"  실제 1700+: {actual_1700:,}개")
    print(f"  Triple Check 감지: {triple_check_count:,}개")
    print(f"  정밀도: {precision:.1f}%")
    print(f"  재현율: {recall:.1f}%")

# ============================================
# slope 비교 (시퀀스 길이가 영향 주는 부분)
# ============================================
print("\n" + "="*70)
print("📊 slope 비교 (시퀀스 길이 영향)")
print("="*70)

# 샘플 1000개로 테스트
sample_indices = np.random.choice(range(280, len(df) - PRED_MINUTES), size=min(1000, len(df)-280-PRED_MINUTES), replace=False)

results = {seq_len: {'slopes': [], 'targets': []} for seq_len in seq_lengths}

for idx in sample_indices:
    target = df['future_totalcnt'].iloc[idx]
    if pd.isna(target):
        continue
    
    for seq_len in seq_lengths:
        if idx >= seq_len:
            seq_total = df[totalcnt_col].iloc[idx-seq_len:idx].values
            slope = np.polyfit(np.arange(seq_len), seq_total, 1)[0]
            results[seq_len]['slopes'].append(slope)
            results[seq_len]['targets'].append(target)

print(f"\n{'시퀀스':<10} {'slope 평균':>12} {'slope std':>12} {'타겟 상관':>12}")
print("-"*50)

for seq_len in seq_lengths:
    slopes = results[seq_len]['slopes']
    targets = results[seq_len]['targets']
    
    if len(slopes) > 0:
        mean_slope = np.mean(slopes)
        std_slope = np.std(slopes)
        corr = np.corrcoef(slopes, targets)[0, 1]
        print(f"{seq_len}분{'':<6} {mean_slope:>12.4f} {std_slope:>12.4f} {corr:>12.4f}")

# ============================================
# 추천
# ============================================
print("\n" + "="*70)
print("💡 결론")
print("="*70)
print("""
패턴 기반 예측은 "현재 시점 값"만 사용:
  - M14B[-1], SUM[-1], gap[-1], TOTALCNT[-1]
  → 시퀀스 길이가 패턴 정확도에 영향 없음!

시퀀스가 영향 주는 것:
  - mean, max, slope 같은 통계 Feature
  - ML 모델 성능

권장:
  - 100분: 충분한 트렌드 + 빠른 처리
  - 280분: 과도하게 김 (4시간 40분 전 데이터?)
""")

print("\n✅ 검증 완료!")