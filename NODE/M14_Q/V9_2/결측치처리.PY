#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 AMHS 결측치 처리 (시계열 특성 고려)

처리 방법:
1. 짧은 결측 (1~3개) → 선형보간 (앞뒤 값 사이 직선)
2. 긴 결측 (4개 이상) → 이동평균 (주변 5분 평균)
3. 처음/끝 결측 → 가장 가까운 유효값으로 채움
"""

import pandas as pd
import numpy as np

# ========================================
# 설정
# ========================================
INPUT_FILE = 'aps.csv'
OUTPUT_FILE = 'aps_cleaned.csv'

# ========================================
# 함수
# ========================================
def smart_interpolate(series):
    """
    스마트 보간:
    - 짧은 결측: 선형보간
    - 긴 결측: 이동평균
    """
    result = series.copy()
    
    # 1단계: 선형보간 (짧은 갭은 이게 제일 자연스러움)
    result = result.interpolate(method='linear', limit=3, limit_direction='both')
    
    # 2단계: 남은 결측은 이동평균 (5분 윈도우)
    if result.isna().any():
        rolling_mean = series.rolling(window=5, min_periods=1, center=True).mean()
        result = result.fillna(rolling_mean)
    
    # 3단계: 처음/끝 남은건 앞뒤 값으로
    result = result.ffill().bfill()
    
    return result

# ========================================
# 실행
# ========================================
print("="*60)
print("📊 AMHS 결측치 처리 (시계열 특성 고려)")
print("="*60)

# 1. 로드
try:
    df = pd.read_csv(INPUT_FILE, encoding='utf-8')
except:
    df = pd.read_csv(INPUT_FILE, encoding='cp949')

print(f"✅ 로드: {len(df):,}행 × {len(df.columns)}열")

# 2. 결측치 현황
print(f"\n📌 결측치 현황:")
missing_cols = []
for col in df.columns:
    if col == 'CURRTIME':
        continue
    missing = df[col].isna().sum() + (df[col] == '').sum()
    if missing > 0:
        missing_cols.append((col, missing))

if missing_cols:
    missing_cols.sort(key=lambda x: -x[1])
    for col, cnt in missing_cols[:10]:
        pct = cnt / len(df) * 100
        print(f"   {col[:45]:<45} : {cnt:>4}개 ({pct:.1f}%)")
    if len(missing_cols) > 10:
        print(f"   ... 외 {len(missing_cols)-10}개 컬럼")
else:
    print("   결측치 없음!")

# 3. 처리
print(f"\n🔄 처리 중...")
numeric_cols = [col for col in df.columns if col != 'CURRTIME']

for col in numeric_cols:
    # 숫자형 변환 (빈 문자열, 공백 등 → NaN)
    df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 스마트 보간 적용
    df[col] = smart_interpolate(df[col])

# 4. 결과 확인
missing_after = df.isna().sum().sum()
print(f"✅ 처리 완료! 남은 결측치: {missing_after}개")

# 5. 주요 컬럼 통계
print(f"\n📈 주요 컬럼 통계:")
key_cols = ['TOTALCNT', 'M14AM14B', 'M14AM14BSUM', 'M10AM14A']
for col in key_cols:
    if col in df.columns:
        print(f"   {col}: {df[col].min():.0f} ~ {df[col].max():.0f} (평균: {df[col].mean():.1f})")

# 6. 저장
df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
print(f"\n💾 저장: {OUTPUT_FILE}")
print("="*60)