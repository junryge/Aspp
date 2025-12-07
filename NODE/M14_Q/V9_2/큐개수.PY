#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 현재 큐개수별 10분 후 증감량 분석 (날짜별)
"""

import pandas as pd
import numpy as np

# 데이터 로드
DATA_PATH = 'your_data.csv'  # ← 실제 파일 경로!
df = pd.read_csv(DATA_PATH)

print("="*70)
print("📊 현재 큐개수별 10분 후 증감량 분석")
print("="*70)

# CURRTIME 파싱
df['CURRTIME'] = df['CURRTIME'].astype(str)
df['날짜'] = df['CURRTIME'].str[:8]  # YYYYMMDD
df['시간'] = df['CURRTIME'].str[8:]  # HHMM

# 10분 후 값과 변화량 계산
df['10분후_TOTALCNT'] = df['TOTALCNT'].shift(-10)
df['10분후_증감'] = df['10분후_TOTALCNT'] - df['TOTALCNT']

# 임계값별 분석
thresholds = [1200, 1300, 1400, 1500, 1600, 1700, 1800]

for th in thresholds:
    print(f"\n{'='*70}")
    print(f"🔹 현재 큐개수 >= {th}")
    print(f"{'='*70}")
    
    subset = df[df['TOTALCNT'] >= th].copy()
    subset = subset.dropna(subset=['10분후_증감'])
    
    if len(subset) == 0:
        print("  해당 데이터 없음")
        continue
    
    # 날짜별 집계
    daily = subset.groupby('날짜').agg(
        건수=('TOTALCNT', 'count'),
        현재큐_평균=('TOTALCNT', 'mean'),
        현재큐_최대=('TOTALCNT', 'max'),
        _10분후_평균=('10분후_TOTALCNT', 'mean'),
        증감_평균=('10분후_증감', 'mean'),
        증감_최소=('10분후_증감', 'min'),
        증감_최대=('10분후_증감', 'max'),
        증가건수=('10분후_증감', lambda x: (x > 0).sum()),
        감소건수=('10분후_증감', lambda x: (x < 0).sum())
    ).round(1)
    
    daily['증가비율(%)'] = (daily['증가건수'] / daily['건수'] * 100).round(1)
    daily['감소비율(%)'] = (daily['감소건수'] / daily['건수'] * 100).round(1)
    
    print(daily.to_string())
    
    # CSV 저장
    daily.to_csv(f'queue_{th}_daily_analysis.csv', encoding='utf-8-sig')
    print(f"\n💾 queue_{th}_daily_analysis.csv 저장")

# 전체 요약
print("\n" + "="*70)
print("📋 전체 요약")
print("="*70)

summary = []
for th in thresholds:
    subset = df[df['TOTALCNT'] >= th].dropna(subset=['10분후_증감'])
    if len(subset) > 0:
        summary.append({
            '임계값': f'>={th}',
            '총건수': len(subset),
            '현재큐_평균': round(subset['TOTALCNT'].mean(), 1),
            '10분후_평균': round(subset['10분후_TOTALCNT'].mean(), 1),
            '증감_평균': round(subset['10분후_증감'].mean(), 1),
            '증가건수': (subset['10분후_증감'] > 0).sum(),
            '감소건수': (subset['10분후_증감'] < 0).sum(),
            '증가비율(%)': round((subset['10분후_증감'] > 0).sum() / len(subset) * 100, 1)
        })

summary_df = pd.DataFrame(summary)
print(summary_df.to_string(index=False))
summary_df.to_csv('queue_threshold_summary.csv', index=False, encoding='utf-8-sig')
print("\n💾 queue_threshold_summary.csv 저장")

print("\n✅ 분석 완료!")