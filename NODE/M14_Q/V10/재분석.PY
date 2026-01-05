# -*- coding: utf-8 -*-
"""
================================================================================
급증 패턴 분석 - 새로운 피처 발굴
================================================================================

분석 내용:
1. Hub/CMD 선행 지표 - TOTALCNT보다 먼저 급증하는지
2. 가속도 (acceleration) - 기울기의 변화율
3. 연속 상승 (consecutive_rise) - 최근 N분 중 상승한 분 수
4. 급등 횟수 (spike_count) - 순간 점프 카운트

사용법:
    python surge_pattern_analysis.py
"""

import pandas as pd
import numpy as np
from datetime import datetime

# ============================================================================
# 설정
# ============================================================================
TRAIN_FILE = 'M14_학습_20251101_20251231_C_109.CSV'  # 학습 파일명 수정 필요
TARGET_COL = 'TOTALCNT'
LIMIT_VALUE = 1700

# Important 그룹 컬럼 (선행 지표 분석용)
HUB_COLS = [
    'M16HUB.QUE.M16TOM14B.CURRENTQCREATED',
    'M16HUB.QUE.M14BTOM16.CURRENTQCREATED',
    'M16HUB.QUE.M14TOM16.CURRENTQCREATED',
    'M16HUB.QUE.M16TOM14.CURRENTQCREATED',
]

CMD_COLS = [
    'M14.QUE.ALL.CURRENTQCOMPLETED',
    'M14.QUE.ALL.CURRENTQCREATED',
    'M14.QUE.ALL.TRANSPORT4MINOVERCNT',
]

LOAD_COLS = [
    'M14.QUE.LOAD.CURRENTLOADQCNT',
    'M14.QUE.LOAD.AVGLOADTIME',
]

# ============================================================================
# 데이터 로드
# ============================================================================
print("=" * 80)
print("급증 패턴 분석 시작")
print("=" * 80)

print("\n[1/5] 데이터 로드 중...")

try:
    df = pd.read_csv(TRAIN_FILE, encoding='utf-8')
except:
    try:
        df = pd.read_csv(TRAIN_FILE, encoding='cp949')
    except:
        df = pd.read_csv(TRAIN_FILE, encoding='euc-kr')

print(f"  - 원본 데이터: {len(df):,}행")

# CURRTIME을 datetime으로 변환
df['CURRTIME'] = pd.to_datetime(df['CURRTIME'], format='%Y%m%d%H%M')
df = df.sort_values('CURRTIME').reset_index(drop=True)

# 숫자형 변환
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce').fillna(0)

# ============================================================================
# 급증 케이스 식별
# ============================================================================
print("\n[2/5] 급증 케이스 식별 중...")

# 30분 후 최대값 계산
df['future_max_30'] = df[TARGET_COL].rolling(window=30, min_periods=1).max().shift(-30)
df['is_surge'] = ((df[TARGET_COL] < 1600) & (df['future_max_30'] >= LIMIT_VALUE)).astype(int)

surge_count = df['is_surge'].sum()
print(f"  - 급증 케이스 (현재<1600 → 30분 후≥1700): {surge_count:,}건")

# ============================================================================
# 분석 1: Hub/CMD 선행 지표
# ============================================================================
print("\n[3/5] Hub/CMD 선행 지표 분석...")

def analyze_leading_indicator(df, indicator_cols, target_col, name):
    """선행 지표 분석: 지표가 타겟보다 먼저 급증하는지"""
    results = []
    
    for col in indicator_cols:
        if col not in df.columns:
            continue
            
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # 각 컬럼의 변화량 계산
        df[f'{col}_change_10'] = df[col].diff(10)
        df[f'{target_col}_change_10'] = df[target_col].diff(10)
        
        # 급증 케이스에서 지표가 먼저 상승한 비율
        surge_df = df[df['is_surge'] == 1].copy()
        if len(surge_df) == 0:
            continue
            
        # 10분 전 지표 변화 > 타겟 변화인 비율
        leading = (surge_df[f'{col}_change_10'] > surge_df[f'{target_col}_change_10']).mean()
        
        # 상관관계
        corr = df[col].corr(df[target_col])
        
        results.append({
            'column': col.split('.')[-1][:30],
            'leading_ratio': leading,
            'correlation': corr
        })
    
    if results:
        print(f"\n  [{name}]")
        print(f"  {'컬럼':<35} {'선행비율':>10} {'상관관계':>10}")
        print("  " + "-" * 55)
        for r in results:
            print(f"  {r['column']:<35} {r['leading_ratio']:>10.2%} {r['correlation']:>10.3f}")

analyze_leading_indicator(df, HUB_COLS, TARGET_COL, "Hub 지표")
analyze_leading_indicator(df, CMD_COLS, TARGET_COL, "CMD 지표")
analyze_leading_indicator(df, LOAD_COLS, TARGET_COL, "Load 지표")

# ============================================================================
# 분석 2: 가속도 (Acceleration) 분석
# ============================================================================
print("\n[4/5] 가속도 패턴 분석...")

# 기울기 계산
df['slope_5'] = df[TARGET_COL].diff(5) / 5
df['slope_10'] = df[TARGET_COL].diff(10) / 10

# 가속도 = 기울기의 변화
df['acceleration_5'] = df['slope_5'].diff(5)
df['acceleration_10'] = df['slope_10'].diff(10)

# 급증 케이스 vs 일반 케이스 비교
surge_df = df[df['is_surge'] == 1]
normal_df = df[df['is_surge'] == 0]

print(f"\n  [가속도 비교]")
print(f"  {'지표':<20} {'급증케이스':>15} {'일반케이스':>15} {'차이':>10}")
print("  " + "-" * 60)

for col in ['slope_5', 'slope_10', 'acceleration_5', 'acceleration_10']:
    surge_mean = surge_df[col].mean()
    normal_mean = normal_df[col].mean()
    diff = surge_mean - normal_mean
    print(f"  {col:<20} {surge_mean:>15.3f} {normal_mean:>15.3f} {diff:>+10.3f}")

# ============================================================================
# 분석 3: 연속 상승 패턴
# ============================================================================
print("\n  [연속 상승 패턴]")

# 연속 상승 분 수 계산
df['is_rising'] = (df[TARGET_COL].diff() > 0).astype(int)

for window in [5, 10, 15, 20]:
    df[f'consecutive_rise_{window}'] = df['is_rising'].rolling(window=window, min_periods=1).sum()
    
    surge_mean = surge_df[f'consecutive_rise_{window}'].mean() if f'consecutive_rise_{window}' in surge_df.columns else df[df['is_surge']==1][f'consecutive_rise_{window}'].mean()
    normal_mean = normal_df[f'consecutive_rise_{window}'].mean() if f'consecutive_rise_{window}' in normal_df.columns else df[df['is_surge']==0][f'consecutive_rise_{window}'].mean()
    
    # 다시 계산
    surge_df = df[df['is_surge'] == 1]
    normal_df = df[df['is_surge'] == 0]
    
    surge_mean = surge_df[f'consecutive_rise_{window}'].mean()
    normal_mean = normal_df[f'consecutive_rise_{window}'].mean()
    diff = surge_mean - normal_mean
    
    print(f"  최근{window:>2}분 상승횟수      {surge_mean:>15.2f} {normal_mean:>15.2f} {diff:>+10.2f}")

# ============================================================================
# 분석 4: 급등 횟수 (Spike Count)
# ============================================================================
print("\n  [급등 횟수 (1분 변화량 > N)]")

df['change_1min'] = df[TARGET_COL].diff()

for threshold in [10, 20, 30, 50]:
    df[f'spike_{threshold}'] = (df['change_1min'] > threshold).astype(int)
    
    for window in [10, 30]:
        col_name = f'spike_{threshold}_last{window}'
        df[col_name] = df[f'spike_{threshold}'].rolling(window=window, min_periods=1).sum()
        
        surge_df = df[df['is_surge'] == 1]
        normal_df = df[df['is_surge'] == 0]
        
        surge_mean = surge_df[col_name].mean()
        normal_mean = normal_df[col_name].mean()
        diff = surge_mean - normal_mean
        
        if window == 30:
            print(f"  최근{window}분 +{threshold:>2} 점프횟수  {surge_mean:>15.2f} {normal_mean:>15.2f} {diff:>+10.2f}")

# ============================================================================
# 분석 5: 추천 피처 요약
# ============================================================================
print("\n" + "=" * 80)
print("[5/5] 추천 새 피처 요약")
print("=" * 80)

print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│ 추천 피처                                                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. acceleration_5/10    : 기울기의 변화율 (급증 직전 급가속 감지)              │
│ 2. consecutive_rise_10  : 최근 10분 중 상승한 분 수                           │
│ 3. spike_20_last30      : 최근 30분 내 +20 이상 점프 횟수                      │
│ 4. hub_change_ratio     : Hub 변화량 / Target 변화량 (선행 비율)              │
│ 5. momentum             : slope_5 - slope_10 (단기-장기 기울기 차이)          │
└─────────────────────────────────────────────────────────────────────────────┘
""")

print("\n분석 완료!")
print("=" * 80)