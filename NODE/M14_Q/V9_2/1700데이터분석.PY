#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔥 1700+ 급증/하락 예측 컬럼 분석
목표: 10분 후 1700+ 급증 & 안정화 하락 잡는 컬럼 발굴
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 설정 - CSV 경로 수정하세요!
# ============================================================
CSV_PATH = '/mnt/user-data/uploads/YOUR_3MONTH_DATA.CSV'  # ← 여기 수정!

print("="*80)
print("🔥 1700+ 급증/하락 예측 컬럼 분석")
print("="*80)

# ============================================================
# 1. 데이터 로드
# ============================================================
df = pd.read_csv(CSV_PATH, on_bad_lines='skip')
print(f"\n📊 데이터: {len(df):,}행 × {df.shape[1]}열")

# 시간 파싱
if 'CURRTIME' in df.columns:
    df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M', errors='coerce')
    print(f"📅 기간: {df['CURRTIME'].min()} ~ {df['CURRTIME'].max()}")

# ============================================================
# 2. TOTALCNT 분포
# ============================================================
print("\n" + "="*80)
print("📊 TOTALCNT 분포")
print("="*80)

print(f"  범위: {df['TOTALCNT'].min():.0f} ~ {df['TOTALCNT'].max():.0f}")
print(f"  평균: {df['TOTALCNT'].mean():.1f}")

for t in [1600, 1650, 1700, 1750, 1800]:
    cnt = (df['TOTALCNT'] >= t).sum()
    pct = cnt / len(df) * 100
    print(f"  {t}+: {cnt:,}개 ({pct:.2f}%)")

# ============================================================
# 3. 핵심 파생변수 생성
# ============================================================
print("\n" + "="*80)
print("🔧 파생변수 생성")
print("="*80)

# queue_gap (가장 중요!)
if 'M14.QUE.ALL.CURRENTQCREATED' in df.columns and 'M14.QUE.ALL.CURRENTQCOMPLETED' in df.columns:
    df['queue_gap'] = df['M14.QUE.ALL.CURRENTQCREATED'] - df['M14.QUE.ALL.CURRENTQCOMPLETED']
    print("✅ queue_gap 생성")

# M14B queue_gap
if 'M14B.QUE.ALL.CURRENTQCREATED' in df.columns and 'M14B.QUE.ALL.CURRENTQCOMPLETED' in df.columns:
    df['queue_gap_M14B'] = df['M14B.QUE.ALL.CURRENTQCREATED'] - df['M14B.QUE.ALL.CURRENTQCOMPLETED']
    print("✅ queue_gap_M14B 생성")

# m14_combined
if 'M14AM14B' in df.columns and 'M14AM14BSUM' in df.columns:
    df['m14_combined'] = df['M14AM14B'] + df['M14AM14BSUM']
    print("✅ m14_combined 생성")

# ============================================================
# 4. 타겟 변수 생성 (핵심!)
# ============================================================
print("\n" + "="*80)
print("🎯 타겟 변수 생성")
print("="*80)

# 10분 후 TOTALCNT
df['TOTAL_10min'] = df['TOTALCNT'].shift(-10)

# 타겟 1: 10분 후 1700+ 여부
df['target_1700'] = (df['TOTAL_10min'] >= 1700).astype(int)

# 타겟 2: 급증 (현재 <1650 → 10분후 >=1700)
df['target_surge'] = ((df['TOTALCNT'] < 1650) & (df['TOTAL_10min'] >= 1700)).astype(int)

# 타겟 3: 하락 (현재 >=1700 → 10분후 <1650)
df['target_drop'] = ((df['TOTALCNT'] >= 1700) & (df['TOTAL_10min'] < 1650)).astype(int)

# 타겟 4: 위험 진입 (현재 <1700 → 10분후 >=1700)
df['target_enter'] = ((df['TOTALCNT'] < 1700) & (df['TOTAL_10min'] >= 1700)).astype(int)

# 타겟 5: 안정화 (현재 >=1700 → 10분후 <1700)
df['target_stable'] = ((df['TOTALCNT'] >= 1700) & (df['TOTAL_10min'] < 1700)).astype(int)

print(f"📊 타겟 분포:")
print(f"  10분후 1700+: {df['target_1700'].sum():,}건 ({df['target_1700'].mean()*100:.2f}%)")
print(f"  급증(<1650→1700+): {df['target_surge'].sum():,}건 ({df['target_surge'].mean()*100:.3f}%)")
print(f"  위험진입(<1700→1700+): {df['target_enter'].sum():,}건 ({df['target_enter'].mean()*100:.2f}%)")
print(f"  안정화(1700+→<1700): {df['target_stable'].sum():,}건 ({df['target_stable'].mean()*100:.2f}%)")
print(f"  급락(1700+→<1650): {df['target_drop'].sum():,}건 ({df['target_drop'].mean()*100:.3f}%)")

# ============================================================
# 5. 각 컬럼의 예측력 분석
# ============================================================
print("\n" + "="*80)
print("📈 컬럼별 예측력 분석")
print("="*80)

numeric_cols = df.select_dtypes(include=[np.number]).columns
exclude_cols = ['TOTALCNT', 'TOTAL_10min', 'target_1700', 'target_surge', 
                'target_drop', 'target_enter', 'target_stable']

results = []

for col in numeric_cols:
    if col in exclude_cols:
        continue
    
    valid = df[[col, 'target_1700', 'target_surge', 'target_enter', 'target_stable']].dropna()
    if len(valid) < 100:
        continue
    
    # 각 타겟과의 상관관계
    corr_1700 = valid[col].corr(valid['target_1700'])
    corr_surge = valid[col].corr(valid['target_surge'])
    corr_enter = valid[col].corr(valid['target_enter'])
    corr_stable = valid[col].corr(valid['target_stable'])
    
    # 급증 시점 vs 정상 시점 평균 비교
    surge_mask = valid['target_surge'] == 1
    normal_mask = valid['target_surge'] == 0
    
    surge_mean = valid[surge_mask][col].mean() if surge_mask.sum() > 0 else np.nan
    normal_mean = valid[normal_mask][col].mean() if normal_mask.sum() > 0 else np.nan
    
    # 구분도 (급증시/정상시 비율)
    if pd.notna(surge_mean) and pd.notna(normal_mean) and normal_mean != 0:
        discriminability = surge_mean / normal_mean
    else:
        discriminability = np.nan
    
    results.append({
        'column': col,
        'corr_1700': corr_1700,
        'corr_surge': corr_surge,
        'corr_enter': corr_enter,
        'corr_stable': corr_stable,
        'surge_mean': surge_mean,
        'normal_mean': normal_mean,
        'discriminability': discriminability
    })

result_df = pd.DataFrame(results)

# ============================================================
# 6. 10분 후 1700+ 예측력 TOP 30
# ============================================================
print("\n" + "="*80)
print("🎯 10분 후 1700+ 예측력 TOP 30")
print("="*80)

top_1700 = result_df.copy()
top_1700['abs_corr'] = top_1700['corr_1700'].abs()
top_1700 = top_1700.sort_values('abs_corr', ascending=False).head(30)

print("-"*90)
print(f"{'순위':>3} {'컬럼':<45} {'1700+상관':>10} {'급증상관':>10} {'진입상관':>10}")
print("-"*90)

for idx, row in top_1700.iterrows():
    rank = top_1700.index.get_loc(idx) + 1
    print(f"{rank:>3} {row['column']:<45} {row['corr_1700']:>10.4f} {row['corr_surge']:>10.4f} {row['corr_enter']:>10.4f}")

# ============================================================
# 7. 급증 예측력 TOP 20 (핵심!)
# ============================================================
print("\n" + "="*80)
print("🔥 급증(<1650→1700+) 예측력 TOP 20")
print("="*80)

top_surge = result_df.copy()
top_surge['abs_corr'] = top_surge['corr_surge'].abs()
top_surge = top_surge.sort_values('abs_corr', ascending=False).head(20)

print("-"*100)
print(f"{'순위':>3} {'컬럼':<45} {'급증상관':>10} {'급증시평균':>12} {'정상시평균':>12} {'구분도':>8}")
print("-"*100)

for idx, row in top_surge.iterrows():
    rank = top_surge.index.get_loc(idx) + 1
    disc = f"{row['discriminability']:.2f}" if pd.notna(row['discriminability']) else "N/A"
    surge_m = f"{row['surge_mean']:.1f}" if pd.notna(row['surge_mean']) else "N/A"
    normal_m = f"{row['normal_mean']:.1f}" if pd.notna(row['normal_mean']) else "N/A"
    print(f"{rank:>3} {row['column']:<45} {row['corr_surge']:>10.4f} {surge_m:>12} {normal_m:>12} {disc:>8}")

# ============================================================
# 8. 위험 진입 예측력 TOP 20
# ============================================================
print("\n" + "="*80)
print("⚠️ 위험 진입(<1700→1700+) 예측력 TOP 20")
print("="*80)

top_enter = result_df.copy()
top_enter['abs_corr'] = top_enter['corr_enter'].abs()
top_enter = top_enter.sort_values('abs_corr', ascending=False).head(20)

print("-"*90)
print(f"{'순위':>3} {'컬럼':<45} {'진입상관':>10} {'1700+상관':>10}")
print("-"*90)

for idx, row in top_enter.iterrows():
    rank = top_enter.index.get_loc(idx) + 1
    print(f"{rank:>3} {row['column']:<45} {row['corr_enter']:>10.4f} {row['corr_1700']:>10.4f}")

# ============================================================
# 9. 안정화(하락) 예측력 TOP 20
# ============================================================
print("\n" + "="*80)
print("📉 안정화(1700+→<1700) 예측력 TOP 20")
print("="*80)

top_stable = result_df.copy()
top_stable['abs_corr'] = top_stable['corr_stable'].abs()
top_stable = top_stable.sort_values('abs_corr', ascending=False).head(20)

print("-"*90)
print(f"{'순위':>3} {'컬럼':<45} {'안정화상관':>10} {'방향':>6}")
print("-"*90)

for idx, row in top_stable.iterrows():
    rank = top_stable.index.get_loc(idx) + 1
    direction = "↓낮을수록" if row['corr_stable'] < 0 else "↑높을수록"
    print(f"{rank:>3} {row['column']:<45} {row['corr_stable']:>10.4f} {direction:>6}")

# ============================================================
# 10. queue_gap 구간별 급증 확률
# ============================================================
if 'queue_gap' in df.columns:
    print("\n" + "="*80)
    print("🔥 queue_gap 구간별 급증/1700+ 확률")
    print("="*80)
    
    gap_bins = [0, 100, 150, 200, 250, 300, 350, 400, 500, 1000]
    df['gap_bin'] = pd.cut(df['queue_gap'], bins=gap_bins)
    
    print(f"\n{'구간':<15} {'건수':>8} {'1700+확률':>10} {'급증확률':>10} {'진입확률':>10}")
    print("-"*60)
    
    for gap_range in df['gap_bin'].dropna().unique():
        mask = df['gap_bin'] == gap_range
        cnt = mask.sum()
        prob_1700 = df[mask]['target_1700'].mean() * 100
        prob_surge = df[mask]['target_surge'].mean() * 100
        prob_enter = df[mask]['target_enter'].mean() * 100
        print(f"{str(gap_range):<15} {cnt:>8,} {prob_1700:>9.1f}% {prob_surge:>9.2f}% {prob_enter:>9.1f}%")

# ============================================================
# 11. 황금패턴 분석
# ============================================================
if 'M14AM14B' in df.columns and 'M14AM14BSUM' in df.columns:
    print("\n" + "="*80)
    print("🏆 황금패턴 급증 확률")
    print("="*80)
    
    patterns = [
        ('M14B>520 & SUM>600', (df['M14AM14B'] > 520) & (df['M14AM14BSUM'] > 600)),
        ('M14B>540 & SUM>620', (df['M14AM14B'] > 540) & (df['M14AM14BSUM'] > 620)),
        ('M14B>500 & SUM>580', (df['M14AM14B'] > 500) & (df['M14AM14BSUM'] > 580)),
        ('M14B>450', df['M14AM14B'] > 450),
        ('M14B>400', df['M14AM14B'] > 400),
    ]
    
    if 'queue_gap' in df.columns:
        patterns.extend([
            ('queue_gap>250', df['queue_gap'] > 250),
            ('queue_gap>300', df['queue_gap'] > 300),
            ('queue_gap>350', df['queue_gap'] > 350),
            ('M14B>520 & gap>250', (df['M14AM14B'] > 520) & (df['queue_gap'] > 250)),
        ])
    
    print(f"\n{'패턴':<30} {'건수':>8} {'1700+확률':>10} {'급증확률':>10} {'진입확률':>10}")
    print("-"*75)
    
    for name, mask in patterns:
        cnt = mask.sum()
        if cnt > 0:
            prob_1700 = df[mask]['target_1700'].mean() * 100
            prob_surge = df[mask]['target_surge'].mean() * 100
            prob_enter = df[mask]['target_enter'].mean() * 100
            print(f"{name:<30} {cnt:>8,} {prob_1700:>9.1f}% {prob_surge:>9.2f}% {prob_enter:>9.1f}%")

# ============================================================
# 12. 결과 저장
# ============================================================
output_path = '/mnt/user-data/outputs/surge_prediction_analysis.csv'
result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n💾 결과 저장: {output_path}")

# ============================================================
# 13. 최종 추천
# ============================================================
print("\n" + "="*80)
print("⭐ 최종 추천: 급증/하락 예측 핵심 컬럼")
print("="*80)

print("\n🔥 급증 예측 TOP 5:")
for idx, row in top_surge.head(5).iterrows():
    print(f"  - {row['column']}: {row['corr_surge']:.4f}")

print("\n📉 안정화 예측 TOP 5:")
for idx, row in top_stable.head(5).iterrows():
    print(f"  - {row['column']}: {row['corr_stable']:.4f}")

print("\n" + "="*80)
print("✅ 분석 완료!")
print("="*80)