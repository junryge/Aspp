#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 큐 개수 구간별 10분 후 증감량 분석
"""

import pandas as pd
import numpy as np

# 데이터 로드
DATA_PATH = 'your_data.csv'  # ← 실제 파일 경로!
df = pd.read_csv(DATA_PATH)

print("="*70)
print("📊 큐 개수 구간별 10분 후 증감량 분석")
print("="*70)

# 10분 후 값과 변화량 계산
df['TOTALCNT_10min_later'] = df['TOTALCNT'].shift(-10)
df['변화량'] = df['TOTALCNT_10min_later'] - df['TOTALCNT']

# 변화량 구간 정의
bins = [-np.inf, -200, -150, -100, -50, 0, 50, 100, 150, 200, np.inf]
labels = ['<-200', '-200~-150', '-150~-100', '-100~-50', '-50~0', 
          '0~50', '50~100', '100~150', '150~200', '>200']

# 임계값별 분석
thresholds = [1200, 1300, 1400, 1500, 1600, 1700, 1800]

results = {}
for th in thresholds:
    subset = df[df['TOTALCNT'] >= th].copy()
    subset = subset.dropna(subset=['변화량'])
    
    subset['구간'] = pd.cut(subset['변화량'], bins=bins, labels=labels)
    counts = subset['구간'].value_counts().sort_index()
    
    results[f'>={th}'] = counts
    
    print(f"\n{'='*50}")
    print(f"🔹 현재 큐개수 >= {th} (총 {len(subset):,}건)")
    print(f"{'='*50}")
    print(counts.to_string())
    print(f"\n  평균 변화량: {subset['변화량'].mean():.1f}")
    print(f"  증가 비율: {(subset['변화량'] > 0).sum()/len(subset)*100:.1f}%")
    print(f"  감소 비율: {(subset['변화량'] < 0).sum()/len(subset)*100:.1f}%")

# 결과 테이블로 정리
result_df = pd.DataFrame(results)
result_df.to_csv('queue_change_analysis.csv', encoding='utf-8-sig')
print(f"\n💾 queue_change_analysis.csv 저장 완료!")
print("\n" + result_df.to_string())