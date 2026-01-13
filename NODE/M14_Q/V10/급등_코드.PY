# -*- coding: utf-8 -*-
"""
================================================================================
🚀 급상승 구간 원본 데이터 추출기
- 각 범위별 1개 이벤트 선택 → 전후 30개 원본 데이터 그대로 저장
================================================================================
"""

import pandas as pd
from datetime import datetime

# ============================================================================
# 설정
# ============================================================================
INPUT_FILE = 'M14_평가.CSV'
TARGET_COL = 'TOTALCNT'
THRESHOLDS = [3000, 2500, 2200, 2000, 1700]
DATA_COUNT = 30

print("=" * 70)
print("🚀 급상승 구간 원본 데이터 추출기")
print("=" * 70)

# ============================================================================
# 데이터 로드
# ============================================================================
print(f"\n[1/2] 데이터 로드 중...")

for enc in ['utf-8', 'cp949', 'euc-kr']:
    try:
        df = pd.read_csv(INPUT_FILE, encoding=enc, on_bad_lines='skip')
        break
    except:
        continue

print(f"  → 총 {len(df):,}행, {len(df.columns)}개 컬럼")

# 시간순 정렬
if 'CURRTIME' in df.columns:
    df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M', errors='coerce')
    df = df.dropna(subset=['CURRTIME']).sort_values('CURRTIME').reset_index(drop=True)

df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce').fillna(0)

# ============================================================================
# 범위별 추출
# ============================================================================
print(f"\n[2/2] 범위별 추출 중...")

for threshold in THRESHOLDS:
    print(f"\n  🔍 {threshold}대...")
    
    # 해당 범위 진입 시점 찾기
    prev_val = df[TARGET_COL].shift(1)
    
    if threshold == 3000:
        mask = (df[TARGET_COL] >= 3000) & (prev_val < 3000)
    elif threshold == 2500:
        mask = (df[TARGET_COL] >= 2500) & (prev_val < 2500)
    elif threshold == 2200:
        mask = (df[TARGET_COL] >= 2200) & (prev_val < 2200)
    elif threshold == 2000:
        mask = (df[TARGET_COL] >= 2000) & (prev_val < 2000)
    else:
        mask = (df[TARGET_COL] >= 1700) & (prev_val < 1700)
    
    candidates = df[mask]
    
    if len(candidates) == 0:
        # 대안: 해당 범위 최대값
        if threshold == 3000:
            range_mask = df[TARGET_COL] >= 3000
        elif threshold == 2500:
            range_mask = (df[TARGET_COL] >= 2500) & (df[TARGET_COL] < 3000)
        elif threshold == 2200:
            range_mask = (df[TARGET_COL] >= 2200) & (df[TARGET_COL] < 2500)
        elif threshold == 2000:
            range_mask = (df[TARGET_COL] >= 2000) & (df[TARGET_COL] < 2200)
        else:
            range_mask = (df[TARGET_COL] >= 1700) & (df[TARGET_COL] < 2000)
        
        range_df = df[range_mask]
        if len(range_df) == 0:
            print(f"     ❌ 데이터 없음")
            continue
        peak_idx = range_df[TARGET_COL].idxmax()
    else:
        # 변화량 가장 큰 이벤트
        changes = df.loc[candidates.index, TARGET_COL] - prev_val.loc[candidates.index]
        peak_idx = changes.idxmax()
    
    peak_val = df.loc[peak_idx, TARGET_COL]
    print(f"     → 이벤트: idx={peak_idx}, TOTALCNT={peak_val:.0f}")
    
    # 전후 30개 (15개 전 ~ 14개 후)
    start = max(0, peak_idx - 15)
    end = start + DATA_COUNT
    if end > len(df):
        end = len(df)
        start = max(0, end - DATA_COUNT)
    
    # 원본 데이터 그대로 추출
    output_df = df.iloc[start:end].copy()
    
    # 저장
    filename = f"급상승_{threshold}대_30개.csv"
    output_df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"     ✅ 저장: {filename} ({len(output_df)}행 x {len(output_df.columns)}컬럼)")

print("\n" + "=" * 70)
print("✅ 완료!")
print("=" * 70)