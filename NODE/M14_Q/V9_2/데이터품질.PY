#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 데이터 품질 체크 & 정제
1. 중복 데이터 확인/제거
2. 1700+ 데이터 확인
3. 정제된 데이터 저장
"""

import pandas as pd
import numpy as np
import glob

print("="*70)
print("📊 데이터 품질 체크")
print("="*70)

# ========================================
# 1. 데이터 로드
# ========================================
print("\n📂 데이터 로딩...")

# 파일 경로 수정 필요
DATA_FILE = 'data/M14_학습데이터.csv'  # ← 실제 파일 경로로 변경

df = pd.read_csv(DATA_FILE, on_bad_lines='skip')
print(f"✅ 원본 데이터: {len(df):,}행 × {df.shape[1]}열")

# ========================================
# 2. 필수 컬럼 확인
# ========================================
required = [
    'CURRTIME',
    'TOTALCNT', 
    'M14AM14B', 
    'M14AM14BSUM', 
    'M10AM14A',
    'M14.QUE.ALL.TRANSPORT4MINOVERCNT', 
    'M14.QUE.OHT.OHTUTIL',
    'M14.QUE.ALL.CURRENTQCREATED', 
    'M14.QUE.ALL.CURRENTQCOMPLETED'
]

missing = [c for c in required if c not in df.columns]
if missing:
    print(f"❌ 누락 컬럼: {missing}")
else:
    print(f"✅ 필수 컬럼 9개 확인")

# ========================================
# 3. 중복 데이터 확인
# ========================================
print("\n" + "="*70)
print("🔍 중복 데이터 확인")
print("="*70)

# 3-1. 완전 중복 (모든 컬럼 동일)
full_dup = df.duplicated().sum()
print(f"\n완전 중복 (모든 컬럼 동일): {full_dup:,}개")

# 3-2. CURRTIME 중복 (같은 시간에 여러 행)
time_dup = df.duplicated(subset=['CURRTIME']).sum()
print(f"CURRTIME 중복: {time_dup:,}개")

# 3-3. 연속 동일값 (시간만 다르고 나머지 동일)
check_cols = ['TOTALCNT', 'M14AM14B', 'M14AM14BSUM', 'M10AM14A']
df['연속중복'] = df[check_cols].eq(df[check_cols].shift()).all(axis=1)
consecutive_dup = df['연속중복'].sum()
print(f"연속 동일값 (시간 제외 동일): {consecutive_dup:,}개")

# 중복 샘플 보기
if consecutive_dup > 0:
    print("\n[연속 중복 샘플]")
    dup_idx = df[df['연속중복']].index[:5]
    for idx in dup_idx:
        print(f"  행 {idx-1} → {idx}: TOTALCNT={df.loc[idx, 'TOTALCNT']}")

# ========================================
# 4. 1700+ 데이터 확인
# ========================================
print("\n" + "="*70)
print("🔥 1700+ 데이터 확인")
print("="*70)

total_rows = len(df)
high_1700 = (df['TOTALCNT'] >= 1700).sum()
high_1600 = ((df['TOTALCNT'] >= 1600) & (df['TOTALCNT'] < 1700)).sum()
high_1400 = ((df['TOTALCNT'] >= 1400) & (df['TOTALCNT'] < 1600)).sum()
normal = (df['TOTALCNT'] < 1400).sum()

print(f"\n구간별 분포:")
print(f"  0~1400:    {normal:>7,}개 ({normal/total_rows*100:5.2f}%)")
print(f"  1400~1600: {high_1400:>7,}개 ({high_1400/total_rows*100:5.2f}%)")
print(f"  1600~1700: {high_1600:>7,}개 ({high_1600/total_rows*100:5.2f}%)")
print(f"  1700+:     {high_1700:>7,}개 ({high_1700/total_rows*100:5.2f}%) ← 핵심!")

# 1700+ 상세
if high_1700 > 0:
    df_high = df[df['TOTALCNT'] >= 1700]
    print(f"\n1700+ 상세:")
    print(f"  최소: {df_high['TOTALCNT'].min():.0f}")
    print(f"  최대: {df_high['TOTALCNT'].max():.0f}")
    print(f"  평균: {df_high['TOTALCNT'].mean():.0f}")

# ========================================
# 5. 결측치 확인
# ========================================
print("\n" + "="*70)
print("🔍 결측치 확인")
print("="*70)

na_counts = df[required].isna().sum()
print(f"\n컬럼별 결측치:")
for col in required:
    na = df[col].isna().sum()
    if na > 0:
        print(f"  {col}: {na:,}개")
if na_counts.sum() == 0:
    print("  결측치 없음 ✅")

# ========================================
# 6. 중복 제거 옵션
# ========================================
print("\n" + "="*70)
print("🧹 데이터 정제 옵션")
print("="*70)

# 옵션1: 완전 중복만 제거
df_clean1 = df.drop_duplicates()
print(f"\n옵션1 - 완전 중복 제거: {len(df):,} → {len(df_clean1):,}행")

# 옵션2: CURRTIME 중복 제거 (첫번째만 유지)
df_clean2 = df.drop_duplicates(subset=['CURRTIME'], keep='first')
print(f"옵션2 - CURRTIME 중복 제거: {len(df):,} → {len(df_clean2):,}행")

# 옵션3: 연속 동일값 제거
df_clean3 = df[~df['연속중복']].copy()
print(f"옵션3 - 연속 동일값 제거: {len(df):,} → {len(df_clean3):,}행")

# ========================================
# 7. 정제 후 1700+ 비율 확인
# ========================================
print("\n" + "="*70)
print("📊 정제 후 1700+ 비율")
print("="*70)

for name, df_c in [('원본', df), ('옵션1', df_clean1), ('옵션2', df_clean2), ('옵션3', df_clean3)]:
    cnt = (df_c['TOTALCNT'] >= 1700).sum()
    ratio = cnt / len(df_c) * 100
    print(f"  {name}: {cnt:,}개 / {len(df_c):,}행 ({ratio:.2f}%)")

# ========================================
# 8. 정제 데이터 저장
# ========================================
print("\n" + "="*70)
print("💾 저장")
print("="*70)

# 원하는 옵션 선택해서 저장
# df_clean = df_clean1  # 옵션1
# df_clean = df_clean2  # 옵션2
df_clean = df_clean3  # 옵션3 (연속 동일값 제거)

# 임시 컬럼 제거
if '연속중복' in df_clean.columns:
    df_clean = df_clean.drop(columns=['연속중복'])

output_file = 'data/M14_정제완료.csv'
df_clean.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"✅ 저장: {output_file}")
print(f"   {len(df_clean):,}행")

# 1700+ 전용 파일 저장
df_high = df_clean[df_clean['TOTALCNT'] >= 1700]
df_high.to_csv('data/M14_1700plus.csv', index=False, encoding='utf-8-sig')
print(f"✅ 1700+ 저장: data/M14_1700plus.csv ({len(df_high):,}행)")

print("\n✅ 데이터 품질 체크 완료!")