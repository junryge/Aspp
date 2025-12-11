#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔬 황금패턴 이론 검증 코드
- queue_gap > 300 → 위험
- M14B > 520 & SUM > 600 → 황금패턴
- 둘 다 만족 → 96.5% 확률
"""

import pandas as pd
import numpy as np

# ============================================
# 1단계: CSV 파일 경로 설정
# ============================================
CSV_FILE = "YOUR_CSV_FILE.csv"  # ← 실제 파일 경로로 변경!

# ============================================
# 2단계: 데이터 로드 및 컬럼 확인
# ============================================
print("="*70)
print("📂 데이터 로드")
print("="*70)

df = pd.read_csv(CSV_FILE, on_bad_lines='skip')
print(f"총 {len(df):,}행 × {df.shape[1]}열")

print("\n📋 전체 컬럼 목록:")
for i, col in enumerate(df.columns):
    print(f"  [{i:3d}] {col}")

# ============================================
# 3단계: 핵심 컬럼 자동 탐지
# ============================================
print("\n" + "="*70)
print("🔍 핵심 컬럼 자동 탐지")
print("="*70)

# TOTALCNT 찾기
totalcnt_col = None
for col in df.columns:
    if 'TOTALCNT' in col.upper():
        totalcnt_col = col
        print(f"✅ TOTALCNT: {col}")
        break

# M14AM14B 찾기
m14b_col = None
for col in df.columns:
    if 'M14AM14B' in col.upper() and 'SUM' not in col.upper():
        m14b_col = col
        print(f"✅ M14AM14B: {col}")
        break

# M14AM14BSUM 찾기
m14bsum_col = None
for col in df.columns:
    if 'M14AM14BSUM' in col.upper() or ('M14AM14B' in col.upper() and 'SUM' in col.upper()):
        m14bsum_col = col
        print(f"✅ M14AM14BSUM: {col}")
        break

# CURRENTQCREATED 찾기
qcreated_col = None
for col in df.columns:
    if 'CURRENTQCREATED' in col.upper() or 'QCREATED' in col.upper():
        qcreated_col = col
        print(f"✅ Q_CREATED: {col}")
        break

# CURRENTQCOMPLETED 찾기
qcompleted_col = None
for col in df.columns:
    if 'CURRENTQCOMPLETED' in col.upper() or 'QCOMPLETED' in col.upper():
        qcompleted_col = col
        print(f"✅ Q_COMPLETED: {col}")
        break

# ============================================
# 4단계: 컬럼 확인 요청
# ============================================
print("\n" + "="*70)
print("❓ 컬럼 확인 필요")
print("="*70)

missing = []
if not totalcnt_col:
    missing.append("TOTALCNT")
if not m14b_col:
    missing.append("M14AM14B")
if not m14bsum_col:
    missing.append("M14AM14BSUM")
if not qcreated_col:
    missing.append("CURRENTQCREATED")
if not qcompleted_col:
    missing.append("CURRENTQCOMPLETED")

if missing:
    print(f"⚠️ 자동 탐지 실패: {missing}")
    print("\n위 컬럼 목록에서 해당 컬럼명을 찾아서 알려주세요!")
else:
    print("✅ 모든 핵심 컬럼 탐지 완료!")
    
    # ============================================
    # 5단계: 황금패턴 이론 검증
    # ============================================
    print("\n" + "="*70)
    print("🔬 황금패턴 이론 검증")
    print("="*70)
    
    # queue_gap 계산
    df['queue_gap'] = df[qcreated_col] - df[qcompleted_col]
    
    # 조건들
    df['is_1700'] = df[totalcnt_col] >= 1700
    df['gap_over_300'] = df['queue_gap'] > 300
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
    if gap_over > 0:
        print(f"  gap > 300: {gap_over:,}개")
        print(f"  그 중 1700+: {gap_and_1700:,}개 ({gap_and_1700/gap_over*100:.1f}%)")
    else:
        print(f"  gap > 300: 0개 (데이터 없음)")
    
    # 이론 2: 황금패턴 → 1700+
    print(f"\n🏆 이론 2: M14B>520 & SUM>600 → 1700+")
    gold = df['gold_pattern'].sum()
    gold_and_1700 = (df['gold_pattern'] & df['is_1700']).sum()
    if gold > 0:
        print(f"  황금패턴: {gold:,}개")
        print(f"  그 중 1700+: {gold_and_1700:,}개 ({gold_and_1700/gold*100:.1f}%)")
    else:
        print(f"  황금패턴: 0개 (데이터 없음)")
    
    # 이론 3: Triple Check → 96.5%
    print(f"\n⚡ 이론 3: M14B>520 & SUM>600 & gap>250 → 96.5%")
    triple = df['triple_check'].sum()
    triple_and_1700 = (df['triple_check'] & df['is_1700']).sum()
    if triple > 0:
        print(f"  Triple Check: {triple:,}개")
        print(f"  그 중 1700+: {triple_and_1700:,}개 ({triple_and_1700/triple*100:.1f}%)")
    else:
        print(f"  Triple Check: 0개 (데이터 없음)")
    
    # 추가: 1700+ 중 패턴 미충족 케이스
    print(f"\n⚠️ 1700+ 중 패턴 미충족 케이스:")
    no_pattern_1700 = df['is_1700'] & ~df['gold_pattern'] & ~df['gap_over_300']
    print(f"  황금패턴X & gap<=300 인데 1700+: {no_pattern_1700.sum():,}개")
    
    # 상세 임계값 분석
    print("\n" + "="*70)
    print("📊 임계값별 상세 분석")
    print("="*70)
    
    for gap_th in [200, 250, 300, 350, 400, 500]:
        cond = df['queue_gap'] > gap_th
        cnt = cond.sum()
        if cnt > 0:
            hit = (cond & df['is_1700']).sum()
            print(f"  gap > {gap_th}: {cnt:,}개 중 1700+: {hit:,}개 ({hit/cnt*100:.1f}%)")
    
    print()
    for m14b_th in [450, 480, 500, 520, 540]:
        cond = df[m14b_col] > m14b_th
        cnt = cond.sum()
        if cnt > 0:
            hit = (cond & df['is_1700']).sum()
            print(f"  M14B > {m14b_th}: {cnt:,}개 중 1700+: {hit:,}개 ({hit/cnt*100:.1f}%)")

print("\n✅ 검증 완료!")