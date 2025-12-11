#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔬 황금패턴 이론 검증 - 수정판
컬럼 직접 지정!
"""

import pandas as pd
import numpy as np

CSV_FILE = "YOUR_CSV_FILE.csv"  # ← 실제 파일 경로!

print("="*70)
print("📂 데이터 로드")
print("="*70)

df = pd.read_csv(CSV_FILE, on_bad_lines='skip')
print(f"총 {len(df):,}행 × {df.shape[1]}열")

# ============================================
# 컬럼 직접 지정! (자동탐지 X)
# ============================================
totalcnt_col = 'TOTALCNT'
m14b_col = 'M14AM14B'
m14bsum_col = 'M14AM14BSUM'
qcreated_col = 'M14.QUE.ALL.CURRENTQCREATED'      # ← 수정!
qcompleted_col = 'M14.QUE.ALL.CURRENTQCOMPLETED'  # ← 수정!

print(f"\n✅ 사용 컬럼:")
print(f"  TOTALCNT: {totalcnt_col}")
print(f"  M14AM14B: {m14b_col}")
print(f"  M14AM14BSUM: {m14bsum_col}")
print(f"  Q_CREATED: {qcreated_col}")
print(f"  Q_COMPLETED: {qcompleted_col}")

# ============================================
# queue_gap 계산
# ============================================
df['queue_gap'] = df[qcreated_col] - df[qcompleted_col]

print(f"\n📊 queue_gap 통계:")
print(f"  min: {df['queue_gap'].min():.0f}")
print(f"  max: {df['queue_gap'].max():.0f}")
print(f"  mean: {df['queue_gap'].mean():.1f}")

# ============================================
# 황금패턴 이론 검증
# ============================================
print("\n" + "="*70)
print("🔬 황금패턴 이론 검증")
print("="*70)

df['is_1700'] = df[totalcnt_col] >= 1700
df['gap_over_300'] = df['queue_gap'] > 300
df['gap_over_250'] = df['queue_gap'] > 250
df['gold_pattern'] = (df[m14b_col] > 520) & (df[m14bsum_col] > 600)
df['triple_check'] = (df[m14b_col] > 520) & (df[m14bsum_col] > 600) & (df['queue_gap'] > 250)

total = len(df)
cnt_1700 = df['is_1700'].sum()

print(f"\n📊 기본 통계:")
print(f"  전체 데이터: {total:,}개")
print(f"  1700+ 이벤트: {cnt_1700:,}개 ({cnt_1700/total*100:.2f}%)")

# 이론 1: queue_gap > 300 → 위험
print(f"\n🔥 이론 1: queue_gap > 300 → 1700+")
gap_over = df['gap_over_300'].sum()
gap_and_1700 = (df['gap_over_300'] & df['is_1700']).sum()
print(f"  gap > 300: {gap_over:,}개")
if gap_over > 0:
    print(f"  그 중 1700+: {gap_and_1700:,}개 ({gap_and_1700/gap_over*100:.1f}%)")

# 이론 2: 황금패턴 → 1700+
print(f"\n🏆 이론 2: M14B>520 & SUM>600 → 1700+")
gold = df['gold_pattern'].sum()
gold_and_1700 = (df['gold_pattern'] & df['is_1700']).sum()
print(f"  황금패턴: {gold:,}개")
if gold > 0:
    print(f"  그 중 1700+: {gold_and_1700:,}개 ({gold_and_1700/gold*100:.1f}%)")

# 이론 3: Triple Check → 96.5%
print(f"\n⚡ 이론 3: M14B>520 & SUM>600 & gap>250 → 96.5%")
triple = df['triple_check'].sum()
triple_and_1700 = (df['triple_check'] & df['is_1700']).sum()
print(f"  Triple Check: {triple:,}개")
if triple > 0:
    print(f"  그 중 1700+: {triple_and_1700:,}개 ({triple_and_1700/triple*100:.1f}%)")

# 1700+ 중 패턴 미충족
print(f"\n⚠️ 1700+ 중 패턴 미충족 케이스:")
no_pattern = df['is_1700'] & ~df['gold_pattern'] & ~df['gap_over_300']
print(f"  황금패턴X & gap<=300 인데 1700+: {no_pattern.sum():,}개")

# ============================================
# 상세 임계값 분석
# ============================================
print("\n" + "="*70)
print("📊 queue_gap 임계값별 분석")
print("="*70)

for gap_th in [100, 150, 200, 250, 300, 350, 400, 500]:
    cond = df['queue_gap'] > gap_th
    cnt = cond.sum()
    if cnt > 0:
        hit = (cond & df['is_1700']).sum()
        print(f"  gap > {gap_th}: {cnt:,}개 중 1700+: {hit:,}개 ({hit/cnt*100:.1f}%)")

print("\n" + "="*70)
print("📊 M14B 임계값별 분석")
print("="*70)

for m14b_th in [400, 450, 480, 500, 520, 540, 560]:
    cond = df[m14b_col] > m14b_th
    cnt = cond.sum()
    if cnt > 0:
        hit = (cond & df['is_1700']).sum()
        print(f"  M14B > {m14b_th}: {cnt:,}개 중 1700+: {hit:,}개 ({hit/cnt*100:.1f}%)")

print("\n" + "="*70)
print("📊 복합 조건 분석")
print("="*70)

# 다양한 조합 테스트
combos = [
    ("M14B>520 & SUM>600", (df[m14b_col] > 520) & (df[m14bsum_col] > 600)),
    ("M14B>520 & gap>250", (df[m14b_col] > 520) & (df['queue_gap'] > 250)),
    ("SUM>600 & gap>250", (df[m14bsum_col] > 600) & (df['queue_gap'] > 250)),
    ("M14B>520 & SUM>600 & gap>200", (df[m14b_col] > 520) & (df[m14bsum_col] > 600) & (df['queue_gap'] > 200)),
    ("M14B>520 & SUM>600 & gap>250", (df[m14b_col] > 520) & (df[m14bsum_col] > 600) & (df['queue_gap'] > 250)),
    ("M14B>520 & SUM>600 & gap>300", (df[m14b_col] > 520) & (df[m14bsum_col] > 600) & (df['queue_gap'] > 300)),
    ("gap>300 alone", df['queue_gap'] > 300),
    ("gap>400 alone", df['queue_gap'] > 400),
    ("gap>500 alone", df['queue_gap'] > 500),
]

for name, cond in combos:
    cnt = cond.sum()
    if cnt > 0:
        hit = (cond & df['is_1700']).sum()
        print(f"  {name}: {cnt:,}개 중 1700+: {hit:,}개 ({hit/cnt*100:.1f}%)")
    else:
        print(f"  {name}: 0개")

print("\n✅ 검증 완료!")