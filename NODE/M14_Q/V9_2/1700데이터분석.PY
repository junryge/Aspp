#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔬 나머지 컬럼 가치 분석
- 핵심 5개 외 다른 컬럼이 미감지 70%를 잡을 수 있나?
"""

import pandas as pd
import numpy as np

CSV_FILE = "YOUR_CSV_FILE.csv"  # ← 실제 파일 경로!

print("="*70)
print("🔬 컬럼 가치 분석")
print("="*70)

df = pd.read_csv(CSV_FILE, on_bad_lines='skip')
print(f"총 {len(df):,}행 × {df.shape[1]}열")

# 핵심 컬럼
totalcnt_col = 'TOTALCNT'
m14b_col = 'M14AM14B'
m14bsum_col = 'M14AM14BSUM'
qcreated_col = 'M14.QUE.ALL.CURRENTQCREATED'
qcompleted_col = 'M14.QUE.ALL.CURRENTQCOMPLETED'

# queue_gap 계산
df['queue_gap'] = df[qcreated_col] - df[qcompleted_col]

# 10분 후 TOTALCNT
df['future_totalcnt'] = df[totalcnt_col].shift(-10)
df['future_1700'] = df['future_totalcnt'] >= 1700
df = df.dropna(subset=['future_totalcnt']).copy()

# 현재 조건
df['triple_check'] = (df[m14b_col] > 520) & (df[m14bsum_col] > 600) & (df['queue_gap'] > 250)

# 미감지 케이스 (Triple Check로 못 잡은 1700+)
df['missed'] = df['future_1700'] & ~df['triple_check']

total_1700 = df['future_1700'].sum()
missed_count = df['missed'].sum()
print(f"\n📊 기본 통계:")
print(f"  10분 후 1700+ 전체: {total_1700:,}개")
print(f"  Triple Check 감지: {total_1700 - missed_count:,}개")
print(f"  미감지: {missed_count:,}개 ({missed_count/total_1700*100:.1f}%)")

# ============================================
# V8.3 사용 컬럼들 분석
# ============================================
print("\n" + "="*70)
print("📊 V8.3 사용 컬럼 분석")
print("="*70)

v83_cols = [
    'M10AM14A',
    'M14.QUE.ALL.TRANSPORT4MINOVERCNT',
    'M14.QUE.OHT.OHTUTIL',
]

for col in v83_cols:
    if col in df.columns:
        # 1700+ vs 정상 비교
        val_1700 = df[df['future_1700']][col].mean()
        val_normal = df[~df['future_1700']][col].mean()
        val_missed = df[df['missed']][col].mean()
        
        print(f"\n🔍 {col}")
        print(f"  정상 평균: {val_normal:.1f}")
        print(f"  1700+ 평균: {val_1700:.1f} (차이: {val_1700-val_normal:+.1f})")
        print(f"  미감지 평균: {val_missed:.1f}")
        
        # 상관관계
        corr = df[col].corr(df['future_totalcnt'])
        print(f"  10분 후 TOTALCNT 상관계수: {corr:.3f}")

# ============================================
# 전체 컬럼 상관관계 분석
# ============================================
print("\n" + "="*70)
print("📊 전체 컬럼 상관관계 TOP 20")
print("="*70)

correlations = []
for col in df.columns:
    if col in ['CURRTIME', 'future_totalcnt', 'future_1700', 'triple_check', 'missed', 'queue_gap']:
        continue
    try:
        corr = df[col].corr(df['future_totalcnt'])
        if not np.isnan(corr):
            correlations.append((col, corr))
    except:
        pass

correlations.sort(key=lambda x: abs(x[1]), reverse=True)

print(f"\n{'컬럼':<50} {'상관계수':>10}")
print("-"*62)
for col, corr in correlations[:20]:
    marker = "⭐" if col in [totalcnt_col, m14b_col, m14bsum_col, qcreated_col, qcompleted_col] else ""
    print(f"{col:<50} {corr:>10.4f} {marker}")

# ============================================
# 미감지 케이스 특성 분석
# ============================================
print("\n" + "="*70)
print("🔍 미감지 케이스 vs 감지 케이스 비교")
print("="*70)

detected = df['future_1700'] & df['triple_check']

key_cols = [totalcnt_col, m14b_col, m14bsum_col, 'queue_gap']
if 'M10AM14A' in df.columns:
    key_cols.append('M10AM14A')
if 'M14.QUE.ALL.TRANSPORT4MINOVERCNT' in df.columns:
    key_cols.append('M14.QUE.ALL.TRANSPORT4MINOVERCNT')
if 'M14.QUE.OHT.OHTUTIL' in df.columns:
    key_cols.append('M14.QUE.OHT.OHTUTIL')
if 'M14.QUE.ALL.CURRENTQCNT' in df.columns:
    key_cols.append('M14.QUE.ALL.CURRENTQCNT')

print(f"\n{'컬럼':<45} {'감지 평균':>12} {'미감지 평균':>12} {'차이':>10}")
print("-"*82)
for col in key_cols:
    if col in df.columns:
        det_mean = df[detected][col].mean()
        mis_mean = df[df['missed']][col].mean()
        diff = mis_mean - det_mean
        print(f"{col:<45} {det_mean:>12.1f} {mis_mean:>12.1f} {diff:>+10.1f}")

# ============================================
# 미감지 케이스 보완 가능성
# ============================================
print("\n" + "="*70)
print("🎯 미감지 보완 가능성 탐색")
print("="*70)

# 미감지 케이스에서 다른 조건으로 잡을 수 있는지
missed_df = df[df['missed']]

print(f"\n미감지 {missed_count:,}개 중:")

# 현재 TOTALCNT가 이미 높은 경우
high_current = (missed_df[totalcnt_col] >= 1600).sum()
print(f"  현재 TOTALCNT >= 1600: {high_current:,}개 ({high_current/missed_count*100:.1f}%)")

high_current2 = (missed_df[totalcnt_col] >= 1650).sum()
print(f"  현재 TOTALCNT >= 1650: {high_current2:,}개 ({high_current2/missed_count*100:.1f}%)")

# M14B만 높은 경우
m14b_high = (missed_df[m14b_col] > 450).sum()
print(f"  M14B > 450: {m14b_high:,}개 ({m14b_high/missed_count*100:.1f}%)")

# gap > 150
gap_150 = (missed_df['queue_gap'] > 150).sum()
print(f"  gap > 150: {gap_150:,}개 ({gap_150/missed_count*100:.1f}%)")

gap_100 = (missed_df['queue_gap'] > 100).sum()
print(f"  gap > 100: {gap_100:,}개 ({gap_100/missed_count*100:.1f}%)")

# TRANSPORT 높은 경우
if 'M14.QUE.ALL.TRANSPORT4MINOVERCNT' in df.columns:
    trans_col = 'M14.QUE.ALL.TRANSPORT4MINOVERCNT'
    trans_high = (missed_df[trans_col] > 100).sum()
    print(f"  TRANSPORT > 100: {trans_high:,}개 ({trans_high/missed_count*100:.1f}%)")

# ============================================
# 보완 조건 테스트
# ============================================
print("\n" + "="*70)
print("🧪 보완 조건 테스트")
print("="*70)

# 기존 Triple Check + 추가 조건
base_cond = df['triple_check']

# 추가 조건 후보
additional_conditions = [
    ("현재 TOTALCNT >= 1650", df[totalcnt_col] >= 1650),
    ("현재 TOTALCNT >= 1600", df[totalcnt_col] >= 1600),
    ("M14B > 450 & gap > 150", (df[m14b_col] > 450) & (df['queue_gap'] > 150)),
    ("M14B > 400 & gap > 200", (df[m14b_col] > 400) & (df['queue_gap'] > 200)),
    ("gap > 200 only", df['queue_gap'] > 200),
]

print(f"\n{'조건':<40} {'정밀도':>10} {'재현율':>10} {'감지수':>10}")
print("-"*72)

# 기본 Triple Check
tc_pred = base_cond.sum()
tc_hit = (base_cond & df['future_1700']).sum()
tc_prec = tc_hit / tc_pred * 100 if tc_pred > 0 else 0
tc_rec = tc_hit / total_1700 * 100
print(f"{'Triple Check (기본)':<40} {tc_prec:>9.1f}% {tc_rec:>9.1f}% {tc_hit:>10,}")

for name, add_cond in additional_conditions:
    combined = base_cond | add_cond
    pred = combined.sum()
    hit = (combined & df['future_1700']).sum()
    prec = hit / pred * 100 if pred > 0 else 0
    rec = hit / total_1700 * 100
    print(f"{'Triple + ' + name:<40} {prec:>9.1f}% {rec:>9.1f}% {hit:>10,}")

print("\n✅ 분석 완료!")