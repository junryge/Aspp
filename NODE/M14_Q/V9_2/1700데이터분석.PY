#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔥 AMHS 전체 컬럼 분석 코드
109개 컬럼 → 1700+ 급증/하락 예측 핵심 컬럼 발굴
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 데이터 로드
# ============================================================
print("="*80)
print("🔥 AMHS 전체 컬럼 분석 (109개 컬럼)")
print("="*80)

# CSV 경로 설정 (필요시 수정)
CSV_PATH = '/mnt/user-data/uploads/OP22.CSV'

df = pd.read_csv(CSV_PATH, on_bad_lines='skip')
print(f"\n📊 데이터: {len(df):,}행 × {df.shape[1]}열")

# 시간 파싱
if 'CURRTIME' in df.columns:
    df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M', errors='coerce')
    print(f"📅 기간: {df['CURRTIME'].min()} ~ {df['CURRTIME'].max()}")

# ============================================================
# 2. TOTALCNT 분포 분석
# ============================================================
print("\n" + "="*80)
print("📊 TOTALCNT 분포")
print("="*80)

print(f"  범위: {df['TOTALCNT'].min():.0f} ~ {df['TOTALCNT'].max():.0f}")
print(f"  평균: {df['TOTALCNT'].mean():.1f}")
print(f"  표준편차: {df['TOTALCNT'].std():.1f}")

# 구간별 분포
thresholds = [1400, 1500, 1600, 1650, 1700, 1750, 1800]
for t in thresholds:
    cnt = (df['TOTALCNT'] >= t).sum()
    pct = cnt / len(df) * 100
    print(f"  {t}+: {cnt:,}개 ({pct:.2f}%)")

# ============================================================
# 3. 전체 컬럼 상관관계 분석
# ============================================================
print("\n" + "="*80)
print("📈 전체 컬럼 TOTALCNT 상관관계")
print("="*80)

# 기존 사용 컬럼
KNOWN_COLS = [
    'CURRTIME', 'TOTALCNT', 'M14AM14B', 'M14AM14BSUM', 'M10AM14A',
    'M14.QUE.ALL.TRANSPORT4MINOVERCNT', 'M14.QUE.OHT.OHTUTIL',
    'M14.QUE.ALL.CURRENTQCREATED', 'M14.QUE.ALL.CURRENTQCOMPLETED'
]

correlations = []
numeric_cols = df.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    if col == 'TOTALCNT':
        continue
    
    valid = df[[col, 'TOTALCNT']].dropna()
    if len(valid) < 10:
        continue
    
    corr = valid[col].corr(valid['TOTALCNT'])
    
    # 통계
    col_mean = valid[col].mean()
    col_std = valid[col].std()
    col_min = valid[col].min()
    col_max = valid[col].max()
    na_count = df[col].isna().sum()
    
    correlations.append({
        'column': col,
        'correlation': corr,
        'abs_corr': abs(corr),
        'mean': col_mean,
        'std': col_std,
        'min': col_min,
        'max': col_max,
        'na_count': na_count,
        'is_new': col not in KNOWN_COLS
    })

corr_df = pd.DataFrame(correlations).sort_values('abs_corr', ascending=False)

# TOP 30 출력
print(f"\n🏆 상관관계 TOP 30:")
print("-"*100)
print(f"{'순위':>3} {'컬럼':<50} {'상관':>8} {'평균':>10} {'범위':>20} {'신규':>4}")
print("-"*100)

for idx, row in corr_df.head(30).iterrows():
    rank = corr_df.index.get_loc(idx) + 1
    new_flag = "🆕" if row['is_new'] else ""
    range_str = f"{row['min']:.0f}~{row['max']:.0f}"
    print(f"{rank:>3} {row['column']:<50} {row['correlation']:>8.4f} {row['mean']:>10.1f} {range_str:>20} {new_flag:>4}")

# ============================================================
# 4. 신규 컬럼만 분석
# ============================================================
print("\n" + "="*80)
print("🆕 신규 컬럼 TOP 20 (기존 미사용)")
print("="*80)

new_cols = corr_df[corr_df['is_new']].head(20)
print("-"*90)
for idx, row in new_cols.iterrows():
    rank = new_cols.index.get_loc(idx) + 1
    print(f"{rank:>3}. {row['column']:<50} 상관: {row['correlation']:>8.4f}")

# ============================================================
# 5. 파생변수 생성 및 분석
# ============================================================
print("\n" + "="*80)
print("🔧 파생변수 분석")
print("="*80)

derived = {}

# queue_gap (기존)
if 'M14.QUE.ALL.CURRENTQCREATED' in df.columns and 'M14.QUE.ALL.CURRENTQCOMPLETED' in df.columns:
    df['queue_gap'] = df['M14.QUE.ALL.CURRENTQCREATED'] - df['M14.QUE.ALL.CURRENTQCOMPLETED']
    derived['queue_gap'] = df['queue_gap'].corr(df['TOTALCNT'])

# M14B queue_gap
if 'M14B.QUE.ALL.CURRENTQCREATED' in df.columns and 'M14B.QUE.ALL.CURRENTQCOMPLETED' in df.columns:
    df['queue_gap_M14B'] = df['M14B.QUE.ALL.CURRENTQCREATED'] - df['M14B.QUE.ALL.CURRENTQCOMPLETED']
    derived['queue_gap_M14B'] = df['queue_gap_M14B'].corr(df['TOTALCNT'])

# m14b_ratio
if 'M14B.QUE.ALL.CURRENTQCNT' in df.columns and 'M14.QUE.ALL.CURRENTQCNT' in df.columns:
    df['m14b_ratio'] = df['M14B.QUE.ALL.CURRENTQCNT'] / (df['M14.QUE.ALL.CURRENTQCNT'] + 1)
    derived['m14b_ratio'] = df['m14b_ratio'].corr(df['TOTALCNT'])

# OHT 합계
oht_cols = ['M14.QUE.OHT.CURRENTOHTQCNT', 'M14B.QUE.OHT.CURRENTOHTQCNT']
if all(c in df.columns for c in oht_cols):
    df['oht_total'] = df[oht_cols].sum(axis=1)
    derived['oht_total'] = df['oht_total'].corr(df['TOTALCNT'])

# Transport ratio
if 'M14.QUE.ALL.TRANSPORT4MINOVERCNT' in df.columns:
    df['trans_ratio'] = df['M14.QUE.ALL.TRANSPORT4MINOVERCNT'] / (df['TOTALCNT'] + 1)
    derived['trans_ratio'] = df['trans_ratio'].corr(df['TOTALCNT'])

# M14+M14B 합계
if 'M14AM14B' in df.columns and 'M14AM14BSUM' in df.columns:
    df['m14_combined'] = df['M14AM14B'] + df['M14AM14BSUM']
    derived['m14_combined'] = df['m14_combined'].corr(df['TOTALCNT'])

# CNV 시간 합계
cnv_time_cols = ['M14.QUE.CNV.NORTHCNVTOM14TIME', 'M14.QUE.CNV.SOUTHCNVTOM14TIME']
if all(c in df.columns for c in cnv_time_cols):
    df['cnv_time_total'] = df[cnv_time_cols].sum(axis=1)
    derived['cnv_time_total'] = df['cnv_time_total'].corr(df['TOTALCNT'])

print("\n📊 파생변수 상관관계:")
for name, corr in sorted(derived.items(), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {name:<25}: {corr:>8.4f}")

# ============================================================
# 6. 컬럼 그룹별 분석
# ============================================================
print("\n" + "="*80)
print("📦 컬럼 그룹별 TOP 3")
print("="*80)

groups = {
    'M14.QUE': [c for c in df.columns if c.startswith('M14.QUE')],
    'M14B.QUE': [c for c in df.columns if c.startswith('M14B.QUE')],
    'M16HUB': [c for c in df.columns if c.startswith('M16HUB')],
    'M16A/E': [c for c in df.columns if c.startswith('M16A') or c.startswith('M16E')],
    'M14AM': [c for c in df.columns if c.startswith('M14AM') or c.startswith('M14BM')],
}

for group_name, cols in groups.items():
    if not cols:
        continue
    group_corr = corr_df[corr_df['column'].isin(cols)].head(3)
    print(f"\n🔹 {group_name} ({len(cols)}개)")
    for idx, row in group_corr.iterrows():
        print(f"   - {row['column']}: {row['correlation']:.4f}")

# ============================================================
# 7. 시간 지연 분석 (10분 후 예측력)
# ============================================================
print("\n" + "="*80)
print("⏰ 시간 지연 분석 (10분 후 TOTALCNT)")
print("="*80)

df['TOTALCNT_10min'] = df['TOTALCNT'].shift(-10)
df['delta_10min'] = df['TOTALCNT_10min'] - df['TOTALCNT']

# 각 컬럼의 10분 후 예측력
future_corr = []
for col in numeric_cols:
    if col == 'TOTALCNT':
        continue
    valid = df[[col, 'TOTALCNT_10min']].dropna()
    if len(valid) < 10:
        continue
    corr = valid[col].corr(valid['TOTALCNT_10min'])
    future_corr.append({'column': col, 'corr_future': corr})

future_df = pd.DataFrame(future_corr)
future_df['abs_corr'] = future_df['corr_future'].abs()
future_df = future_df.sort_values('abs_corr', ascending=False)

print(f"\n🎯 10분 후 예측력 TOP 15:")
for idx, row in future_df.head(15).iterrows():
    print(f"  {row['column']:<50}: {row['corr_future']:.4f}")

# ============================================================
# 8. 급증/급락 패턴 분석
# ============================================================
print("\n" + "="*80)
print("📈 급증/급락 패턴 분석")
print("="*80)

# 10분간 변화량
df['change_10min'] = df['TOTALCNT'].diff(10)

# 급증: 10분간 +100 이상
surge_mask = df['change_10min'] >= 100
# 급락: 10분간 -100 이상
drop_mask = df['change_10min'] <= -100

print(f"급증(+100 이상): {surge_mask.sum()}건")
print(f"급락(-100 이상): {drop_mask.sum()}건")

if surge_mask.sum() > 0:
    print(f"\n🔺 급증 시점 컬럼 평균:")
    surge_data = df[surge_mask]
    normal_data = df[~surge_mask & ~drop_mask]
    
    # 급증 vs 정상 비교
    compare_cols = ['M14AM14B', 'M14AM14BSUM', 'queue_gap', 'M14.QUE.OHT.CURRENTOHTQCNT']
    for col in compare_cols:
        if col in df.columns:
            surge_mean = surge_data[col].mean()
            normal_mean = normal_data[col].mean()
            diff = surge_mean - normal_mean
            print(f"  {col:<40}: 급증={surge_mean:.1f}, 정상={normal_mean:.1f}, 차이={diff:+.1f}")

# ============================================================
# 9. 데이터 누출 위험 컬럼 체크
# ============================================================
print("\n" + "="*80)
print("⚠️ 데이터 누출 위험 체크 (상관 > 0.95)")
print("="*80)

leak_risk = corr_df[corr_df['abs_corr'] > 0.95]
if len(leak_risk) > 0:
    print("🚨 사용 금지 컬럼:")
    for idx, row in leak_risk.iterrows():
        print(f"  - {row['column']}: {row['correlation']:.4f}")
else:
    print("✅ 데이터 누출 위험 컬럼 없음")

# ============================================================
# 10. 결측값 분석
# ============================================================
print("\n" + "="*80)
print("🔍 결측값 분석")
print("="*80)

na_info = df.isna().sum()
na_cols = na_info[na_info > 0]
if len(na_cols) > 0:
    print(f"결측값 있는 컬럼: {len(na_cols)}개")
    for col, cnt in na_cols.items():
        pct = cnt / len(df) * 100
        if pct > 1:  # 1% 이상만 출력
            print(f"  {col}: {cnt}개 ({pct:.1f}%)")
else:
    print("✅ 결측값 없음")

# ============================================================
# 11. 최종 추천
# ============================================================
print("\n" + "="*80)
print("⭐ 최종 추천 컬럼")
print("="*80)

print("\n🔥 강력 추천 (상관 0.4 이상 신규):")
strong = corr_df[(corr_df['is_new']) & (corr_df['abs_corr'] >= 0.4)]
for idx, row in strong.iterrows():
    print(f"  - {row['column']}: {row['correlation']:.4f}")

print("\n📈 추천 (상관 0.3~0.4 신규):")
medium = corr_df[(corr_df['is_new']) & (corr_df['abs_corr'] >= 0.3) & (corr_df['abs_corr'] < 0.4)]
for idx, row in medium.iterrows():
    print(f"  - {row['column']}: {row['correlation']:.4f}")

# ============================================================
# 12. 결과 저장
# ============================================================
output_path = '/mnt/user-data/outputs/column_analysis_result.csv'
corr_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n💾 결과 저장: {output_path}")

print("\n" + "="*80)
print("✅ 분석 완료!")
print("="*80)