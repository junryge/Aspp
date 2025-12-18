#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 오탐/미감지 분석
"""

import pandas as pd
import numpy as np

df = pd.read_csv('evaluation_V8.3_10min.csv')

actual_danger = df['실제위험(1700+)'] == 'O'
pred_danger = df['예측위험(1700+)'] == 'O'

# ============================================
# 1. 놓친 케이스 (FN: 실제O, 예측X)
# ============================================
fn = actual_danger & ~pred_danger
print("="*70)
print(f"❌ 놓친 케이스 (FN): {fn.sum()}개")
print("="*70)
if fn.sum() > 0:
    cols = ['현재시간', '현재TOTALCNT', '12분변화량', '실제값', '보정예측', '추세예측', '최종예측', 'M14AM14B', 'M14AM14BSUM', '>=1580', '추세적용']
    print(df[fn][cols].to_string(index=False))
    
    print(f"\n📌 놓친 케이스 특징:")
    print(f"  현재TOTALCNT 평균: {df[fn]['현재TOTALCNT'].mean():.1f}")
    print(f"  12분변화량 평균: {df[fn]['12분변화량'].mean():.1f}")
    print(f"  M14AM14B 평균: {df[fn]['M14AM14B'].mean():.1f}")
    print(f"  >=1580 비율: {(df[fn]['>=1580']=='O').sum()}/{fn.sum()}")

# ============================================
# 2. 오탐 케이스 (FP: 실제X, 예측O)
# ============================================
fp = ~actual_danger & pred_danger
print("\n" + "="*70)
print(f"⚠️ 오탐 케이스 (FP): {fp.sum()}개")
print("="*70)
if fp.sum() > 0:
    cols = ['현재시간', '현재TOTALCNT', '12분변화량', '실제값', '보정예측', '추세예측', '최종예측', 'M14AM14B', '>=1580', '추세적용']
    print(df[fp][cols].head(20).to_string(index=False))
    
    print(f"\n📌 오탐 케이스 특징:")
    print(f"  현재TOTALCNT 평균: {df[fp]['현재TOTALCNT'].mean():.1f}")
    print(f"  12분변화량 평균: {df[fp]['12분변화량'].mean():.1f}")
    print(f"  M14AM14B 평균: {df[fp]['M14AM14B'].mean():.1f}")
    print(f"  실제값 평균: {df[fp]['실제값'].mean():.1f}")
    print(f"  최종예측 평균: {df[fp]['최종예측'].mean():.1f}")
    print(f"  과대예측(최종-실제) 평균: {(df[fp]['최종예측'] - df[fp]['실제값']).mean():.1f}")
    
    # 오탐 원인 분류
    fp_by_trend = (df[fp]['추세적용'] == 'O').sum()
    fp_by_model = fp.sum() - fp_by_trend
    print(f"\n📌 오탐 원인:")
    print(f"  추세 적용으로 인한 오탐: {fp_by_trend}개")
    print(f"  보정예측 자체의 오탐: {fp_by_model}개")

# ============================================
# 3. 추세 적용 효과 분석
# ============================================
print("\n" + "="*70)
print("📊 추세 적용 효과 상세")
print("="*70)
trend_applied = df['추세적용'] == 'O'
print(f"추세 적용 케이스: {trend_applied.sum()}개")

if trend_applied.sum() > 0:
    # 추세 적용 중 실제 1700+
    trend_tp = (trend_applied & actual_danger).sum()
    trend_fp = (trend_applied & ~actual_danger).sum()
    print(f"  → 실제 1700+ (TP): {trend_tp}개")
    print(f"  → 오탐 (FP): {trend_fp}개")
    print(f"  → 정밀도: {trend_tp/(trend_tp+trend_fp)*100:.1f}%")

# ============================================
# 4. 임계값별 성능
# ============================================
print("\n" + "="*70)
print("📊 조건별 성능 비교")
print("="*70)

conditions = [
    ('보정예측>=1700', df['보정예측'] >= 1700),
    ('최종예측>=1700', df['최종예측'] >= 1700),
    ('>=1580', df['현재TOTALCNT'] >= 1580),
    ('>=1580 & M14B>350', (df['현재TOTALCNT'] >= 1580) & (df['M14AM14B'] > 350)),
    ('>=1580 & M14B>400', (df['현재TOTALCNT'] >= 1580) & (df['M14AM14B'] > 400)),
    ('>=1600', df['현재TOTALCNT'] >= 1600),
    ('>=1600 & M14B>350', (df['현재TOTALCNT'] >= 1600) & (df['M14AM14B'] > 350)),
]

print(f"\n{'조건':<30} {'발생':>6} {'TP':>5} {'FP':>5} {'정밀도':>8} {'감지율':>8}")
print("-"*70)

actual_count = actual_danger.sum()
for name, cond in conditions:
    triggered = cond.sum()
    tp = (cond & actual_danger).sum()
    fp = triggered - tp
    precision = tp / triggered * 100 if triggered > 0 else 0
    recall = tp / actual_count * 100 if actual_count > 0 else 0
    
    flag = "✅" if recall == 100 else ("🔥" if recall >= 90 else "")
    print(f"{name:<30} {triggered:>6} {tp:>5} {fp:>5} {precision:>7.1f}% {recall:>7.1f}% {flag}")

print("\n✅ 분석 완료!")