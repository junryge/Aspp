# -*- coding: utf-8 -*-
"""
================================================================================
미사용 컬럼 분석 - TOTALCNT와의 상관관계 및 선행 지표 분석
================================================================================
"""

import pandas as pd
import numpy as np

# ============================================================================
# 설정
# ============================================================================
TRAIN_FILE = 'M14_학습_20251101_20251231_C_109.CSV'  # 파일명 수정 필요
TARGET_COL = 'TOTALCNT'
LIMIT_VALUE = 1700

# ============================================================================
# 데이터 로드
# ============================================================================
print("=" * 80)
print("미사용 컬럼 분석 시작")
print("=" * 80)

print("\n[1/4] 데이터 로드 중...")

try:
    df = pd.read_csv(TRAIN_FILE, encoding='utf-8')
except:
    try:
        df = pd.read_csv(TRAIN_FILE, encoding='cp949')
    except:
        df = pd.read_csv(TRAIN_FILE, encoding='euc-kr')

print(f"  - 원본 데이터: {len(df):,}행, {len(df.columns)}컬럼")

# CURRTIME을 datetime으로 변환
df['CURRTIME'] = pd.to_datetime(df['CURRTIME'], format='%Y%m%d%H%M')
df = df.sort_values('CURRTIME').reset_index(drop=True)

# 숫자형 변환
for col in df.columns:
    if col != 'CURRTIME':
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# ============================================================================
# 급증 케이스 식별
# ============================================================================
print("\n[2/4] 급증 케이스 식별 중...")

df['future_max_30'] = df[TARGET_COL].rolling(window=30, min_periods=1).max().shift(-30)
df['is_surge'] = ((df[TARGET_COL] < 1600) & (df['future_max_30'] >= LIMIT_VALUE)).astype(int)

surge_count = df['is_surge'].sum()
print(f"  - 급증 케이스: {surge_count:,}건")

# ============================================================================
# 상관관계 분석
# ============================================================================
print("\n[3/4] 상관관계 분석...")

# 분석할 컬럼 (CURRTIME, 이미 분석한 컬럼 제외)
exclude_cols = ['CURRTIME', 'future_max_30', 'is_surge']
analyze_cols = [c for c in df.columns if c not in exclude_cols]

results = []

for col in analyze_cols:
    if col == TARGET_COL:
        continue
    
    # 상관관계
    corr = df[col].corr(df[TARGET_COL])
    
    # 변화량 상관관계 (10분)
    col_change = df[col].diff(10)
    target_change = df[TARGET_COL].diff(10)
    change_corr = col_change.corr(target_change)
    
    # 급증 케이스에서 평균 vs 일반 케이스 평균
    surge_mean = df[df['is_surge'] == 1][col].mean()
    normal_mean = df[df['is_surge'] == 0][col].mean()
    diff_ratio = (surge_mean - normal_mean) / (normal_mean + 0.001) * 100
    
    # 선행 지표 분석: 10분 전 값이 타겟보다 먼저 상승하는 비율
    col_lead = df[col].shift(10)  # 10분 전 값
    lead_corr = col_lead.corr(df[TARGET_COL])
    
    results.append({
        'column': col,
        'corr': corr,
        'change_corr': change_corr,
        'lead_corr': lead_corr,
        'surge_mean': surge_mean,
        'normal_mean': normal_mean,
        'diff_ratio': diff_ratio
    })

# 결과 정렬 (상관관계 절대값 기준)
results_df = pd.DataFrame(results)
results_df['abs_corr'] = results_df['corr'].abs()
results_df = results_df.sort_values('abs_corr', ascending=False)

# ============================================================================
# 결과 출력
# ============================================================================
print("\n[4/4] 분석 결과")
print("=" * 120)

print("\n[상관관계 TOP 30]")
print(f"{'컬럼':<50} {'상관관계':>10} {'변화상관':>10} {'선행상관':>10} {'급증차이%':>10}")
print("-" * 90)

for _, row in results_df.head(30).iterrows():
    col_short = row['column'][-50:] if len(row['column']) > 50 else row['column']
    print(f"{col_short:<50} {row['corr']:>10.3f} {row['change_corr']:>10.3f} {row['lead_corr']:>10.3f} {row['diff_ratio']:>+10.1f}%")

# 선행 지표 분석 (lead_corr 기준)
print("\n" + "=" * 120)
print("\n[선행 지표 TOP 20] (10분 전 값과 현재 TOTALCNT 상관관계)")
results_df_lead = results_df.copy()
results_df_lead['abs_lead'] = results_df_lead['lead_corr'].abs()
results_df_lead = results_df_lead.sort_values('abs_lead', ascending=False)

print(f"{'컬럼':<50} {'선행상관':>10} {'일반상관':>10} {'차이':>10}")
print("-" * 80)

for _, row in results_df_lead.head(20).iterrows():
    col_short = row['column'][-50:] if len(row['column']) > 50 else row['column']
    diff = row['lead_corr'] - row['corr']
    print(f"{col_short:<50} {row['lead_corr']:>10.3f} {row['corr']:>10.3f} {diff:>+10.3f}")

# 급증 케이스 차이 분석
print("\n" + "=" * 120)
print("\n[급증 케이스 차이 TOP 20] (급증 시 평균 vs 일반 시 평균)")
results_df_surge = results_df.copy()
results_df_surge['abs_diff'] = results_df_surge['diff_ratio'].abs()
results_df_surge = results_df_surge.sort_values('abs_diff', ascending=False)

print(f"{'컬럼':<50} {'급증평균':>12} {'일반평균':>12} {'차이%':>10}")
print("-" * 84)

for _, row in results_df_surge.head(20).iterrows():
    col_short = row['column'][-50:] if len(row['column']) > 50 else row['column']
    print(f"{col_short:<50} {row['surge_mean']:>12.1f} {row['normal_mean']:>12.1f} {row['diff_ratio']:>+10.1f}%")

print("\n" + "=" * 120)
print("분석 완료!")
print("=" * 120)