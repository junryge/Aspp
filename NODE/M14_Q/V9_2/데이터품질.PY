#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 V9.3 데이터 품질 - 균형 잡힌 학습 데이터 생성

핵심:
1. TOTALCNT 구간별 균형 샘플링
2. 정상 데이터 다운샘플링
3. 전이 구간(1600~1700) 비중 높이기
4. 1700+ 전부 포함
5. 다양한 시간대 패턴 포함
"""

import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🔧 V9.3 데이터 품질 - 균형 잡힌 학습 데이터 생성")
print("="*70)

# ============================================
# 설정
# ============================================
INPUT_FILE = "YOUR_CSV_FILE.csv"  # ← 원본 데이터!
OUTPUT_FILE = "train_balanced.csv"  # 균형 데이터 출력

# 목표 비율 설정
TARGET_RATIO = {
    'low': 0.20,       # ~1400: 20%
    'normal': 0.20,    # 1400~1550: 20%
    'rising': 0.20,    # 1550~1650: 20%
    'warning': 0.20,   # 1650~1700: 20%
    'danger': 0.20,    # 1700+: 20%
}

# ============================================
# 데이터 로드
# ============================================
print(f"\n📂 데이터 로딩: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE, on_bad_lines='skip')
print(f"✅ 원본: {len(df):,}행")

# 필수 컬럼 확인
required = ['TOTALCNT', 'M14AM14B', 'M14AM14BSUM', 
            'M14.QUE.ALL.CURRENTQCREATED', 'M14.QUE.ALL.CURRENTQCOMPLETED']
missing = [c for c in required if c not in df.columns]
if missing:
    print(f"❌ 누락 컬럼: {missing}")
    exit(1)

# NaN 제거
df = df.dropna(subset=required)
print(f"✅ NaN 제거 후: {len(df):,}행")

# queue_gap 계산
df['queue_gap'] = df['M14.QUE.ALL.CURRENTQCREATED'] - df['M14.QUE.ALL.CURRENTQCOMPLETED']

# ============================================
# 원본 분포 확인
# ============================================
print(f"\n📊 원본 TOTALCNT 분포:")

def get_segment(val):
    if val < 1400:
        return 'low'
    elif val < 1550:
        return 'normal'
    elif val < 1650:
        return 'rising'
    elif val < 1700:
        return 'warning'
    else:
        return 'danger'

df['segment'] = df['TOTALCNT'].apply(get_segment)
orig_counts = df['segment'].value_counts()

for seg in ['low', 'normal', 'rising', 'warning', 'danger']:
    cnt = orig_counts.get(seg, 0)
    pct = cnt / len(df) * 100
    print(f"  {seg:10}: {cnt:7,}개 ({pct:5.1f}%)")

# ============================================
# 균형 샘플링
# ============================================
print(f"\n🔧 균형 샘플링 시작...")

# 1700+ 전체 개수 기준으로 목표 설정
danger_count = len(df[df['segment'] == 'danger'])
print(f"  1700+ (danger): {danger_count:,}개 (전부 포함)")

# 목표 총 개수 계산 (danger 기준)
target_total = int(danger_count / TARGET_RATIO['danger'])
print(f"  목표 총 개수: {target_total:,}개")

# 세그먼트별 목표 개수
target_counts = {seg: int(target_total * ratio) for seg, ratio in TARGET_RATIO.items()}
target_counts['danger'] = danger_count  # danger는 전부!

print(f"\n📊 세그먼트별 목표:")
for seg, cnt in target_counts.items():
    print(f"  {seg:10}: {cnt:,}개")

# ============================================
# 다양한 패턴 샘플링 함수
# ============================================
def diverse_sampling(df_seg, target_n, segment_name):
    """시간대/패턴 다양하게 샘플링"""
    
    if len(df_seg) <= target_n:
        print(f"  {segment_name}: 전체 포함 ({len(df_seg):,}개)")
        return df_seg
    
    sampled_indices = []
    
    # 1. 시간대별 분산 (CURRTIME 있으면)
    if 'CURRTIME' in df_seg.columns:
        try:
            df_seg = df_seg.copy()
            df_seg['hour'] = pd.to_datetime(df_seg['CURRTIME'].astype(str), format='%Y%m%d%H%M').dt.hour
            
            # 시간대별로 골고루
            hours = df_seg['hour'].unique()
            per_hour = target_n // len(hours)
            
            for hour in hours:
                hour_df = df_seg[df_seg['hour'] == hour]
                if len(hour_df) > 0:
                    n_sample = min(per_hour, len(hour_df))
                    sampled_indices.extend(hour_df.sample(n=n_sample, random_state=42).index.tolist())
            
            # 부족한 만큼 추가 샘플링
            remaining = target_n - len(sampled_indices)
            if remaining > 0:
                available = df_seg[~df_seg.index.isin(sampled_indices)]
                if len(available) > 0:
                    extra = min(remaining, len(available))
                    sampled_indices.extend(available.sample(n=extra, random_state=42).index.tolist())
        except:
            # 시간 파싱 실패 시 랜덤 샘플링
            sampled_indices = df_seg.sample(n=target_n, random_state=42).index.tolist()
    else:
        # CURRTIME 없으면 랜덤 샘플링
        sampled_indices = df_seg.sample(n=target_n, random_state=42).index.tolist()
    
    result = df_seg.loc[sampled_indices[:target_n]]
    print(f"  {segment_name}: {len(df_seg):,} → {len(result):,}개")
    return result

# ============================================
# 세그먼트별 샘플링 실행
# ============================================
sampled_dfs = []

for seg in ['low', 'normal', 'rising', 'warning', 'danger']:
    df_seg = df[df['segment'] == seg]
    target_n = target_counts[seg]
    
    if seg == 'danger':
        # danger는 전부 포함!
        sampled_dfs.append(df_seg)
        print(f"  {seg}: 전체 포함 ({len(df_seg):,}개)")
    else:
        sampled = diverse_sampling(df_seg, target_n, seg)
        sampled_dfs.append(sampled)

# 합치기
df_balanced = pd.concat(sampled_dfs, ignore_index=True)

# 셔플
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# ============================================
# 결과 확인
# ============================================
print(f"\n📊 균형 데이터 분포:")
balanced_counts = df_balanced['segment'].value_counts()

for seg in ['low', 'normal', 'rising', 'warning', 'danger']:
    cnt = balanced_counts.get(seg, 0)
    pct = cnt / len(df_balanced) * 100
    marker = "✅" if 15 <= pct <= 25 else "⚠️"
    print(f"  {seg:10}: {cnt:7,}개 ({pct:5.1f}%) {marker}")

# ============================================
# 패턴 다양성 확인
# ============================================
print(f"\n📊 패턴 다양성 확인:")

# 황금패턴
gold = (df_balanced['M14AM14B'] > 520) & (df_balanced['M14AM14BSUM'] > 600)
print(f"  황금패턴: {gold.sum():,}개 ({gold.sum()/len(df_balanced)*100:.1f}%)")

# gap 분포
print(f"  queue_gap 분포:")
print(f"    < 100: {(df_balanced['queue_gap'] < 100).sum():,}개")
print(f"    100~200: {((df_balanced['queue_gap'] >= 100) & (df_balanced['queue_gap'] < 200)).sum():,}개")
print(f"    200~300: {((df_balanced['queue_gap'] >= 200) & (df_balanced['queue_gap'] < 300)).sum():,}개")
print(f"    300+: {(df_balanced['queue_gap'] >= 300).sum():,}개")

# 1700+ 세부
danger_df = df_balanced[df_balanced['segment'] == 'danger']
print(f"\n📊 1700+ 세부 분석:")
print(f"  1700~1750: {((danger_df['TOTALCNT'] >= 1700) & (danger_df['TOTALCNT'] < 1750)).sum()}개")
print(f"  1750~1800: {((danger_df['TOTALCNT'] >= 1750) & (danger_df['TOTALCNT'] < 1800)).sum()}개")
print(f"  1800+: {(danger_df['TOTALCNT'] >= 1800).sum()}개")

# ============================================
# 저장
# ============================================
# segment 컬럼 제거 후 저장
df_balanced = df_balanced.drop(columns=['segment', 'hour'], errors='ignore')
df_balanced.to_csv(OUTPUT_FILE, index=False)

print(f"\n💾 저장: {OUTPUT_FILE}")
print(f"  행: {len(df_balanced):,}")
print(f"  열: {df_balanced.shape[1]}")

# ============================================
# 요약
# ============================================
print(f"\n" + "="*70)
print("✅ 데이터 품질 처리 완료!")
print("="*70)
print(f"""
원본: {len(df):,}개
  → 균형: {len(df_balanced):,}개

변화:
  low (~1400):     {orig_counts.get('low',0)/len(df)*100:.1f}% → {balanced_counts.get('low',0)/len(df_balanced)*100:.1f}%
  normal:          {orig_counts.get('normal',0)/len(df)*100:.1f}% → {balanced_counts.get('normal',0)/len(df_balanced)*100:.1f}%
  rising:          {orig_counts.get('rising',0)/len(df)*100:.1f}% → {balanced_counts.get('rising',0)/len(df_balanced)*100:.1f}%
  warning:         {orig_counts.get('warning',0)/len(df)*100:.1f}% → {balanced_counts.get('warning',0)/len(df_balanced)*100:.1f}%
  danger (1700+):  {orig_counts.get('danger',0)/len(df)*100:.1f}% → {balanced_counts.get('danger',0)/len(df_balanced)*100:.1f}%

다음 단계: V93_학습 실행!
""")