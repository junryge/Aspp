# -*- coding: utf-8 -*-
"""
================================================================================
🚀 급상승 구간 데이터 추출기
- 3000대, 2500대, 2200대, 2000대, 1700대 각각 1개 이벤트 선택
- 해당 이벤트 전후 30개 데이터 추출 → 범위별 CSV 저장
================================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

# ============================================================================
# 설정
# ============================================================================
CONFIG = {
    'input_file': 'M14_평가.CSV',
    'target_column': 'TOTALCNT',
    'thresholds': [3000, 2500, 2200, 2000, 1700],
    'data_range': 30,  # 전후 30개 데이터 (총 30개)
}

print("=" * 70)
print("🚀 급상승 구간 데이터 추출기")
print("   각 범위별 1개 이벤트 → 전후 30개 데이터 CSV 저장")
print("=" * 70)

# ============================================================================
# 데이터 로드
# ============================================================================
print(f"\n[1/3] 데이터 로드 중...")

encodings = ['utf-8', 'cp949', 'euc-kr']
df = None
for enc in encodings:
    try:
        df = pd.read_csv(CONFIG['input_file'], encoding=enc, on_bad_lines='skip')
        break
    except:
        continue

if df is None:
    raise ValueError("파일을 읽을 수 없습니다!")

print(f"  → 총 데이터: {len(df):,}행")

# 시간 파싱
if 'CURRTIME' in df.columns:
    df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M', errors='coerce')
    df = df.dropna(subset=['CURRTIME']).sort_values('CURRTIME').reset_index(drop=True)

df[CONFIG['target_column']] = pd.to_numeric(df[CONFIG['target_column']], errors='coerce').fillna(0)
df['이전값'] = df[CONFIG['target_column']].shift(1)
df['변화량'] = df[CONFIG['target_column']] - df['이전값']

# ============================================================================
# 범위별 추출
# ============================================================================
print("\n[2/3] 범위별 급상승 이벤트 탐색...")

for threshold in CONFIG['thresholds']:
    range_name = f"{threshold}대"
    print(f"\n  🔍 {range_name} 처리 중...")
    
    # 해당 범위로 처음 진입하는 순간 찾기
    if threshold == 3000:
        mask = (df[CONFIG['target_column']] >= 3000)
    elif threshold == 2500:
        mask = (df[CONFIG['target_column']] >= 2500) & (df[CONFIG['target_column']] < 3000)
    elif threshold == 2200:
        mask = (df[CONFIG['target_column']] >= 2200) & (df[CONFIG['target_column']] < 2500)
    elif threshold == 2000:
        mask = (df[CONFIG['target_column']] >= 2000) & (df[CONFIG['target_column']] < 2200)
    else:  # 1700
        mask = (df[CONFIG['target_column']] >= 1700) & (df[CONFIG['target_column']] < 2000)
    
    # 이전값이 해당 범위 미만인 경우 (진입 순간)
    entry_mask = mask & (df['이전값'] < threshold)
    candidates = df[entry_mask]
    
    if len(candidates) == 0:
        print(f"     ⚠️ 진입 이벤트 없음, 최대값 구간으로 대체")
        # 대안: 해당 범위 내 최대값 시점
        range_df = df[mask]
        if len(range_df) == 0:
            print(f"     ❌ {range_name} 데이터 없음")
            continue
        peak_idx = range_df[CONFIG['target_column']].idxmax()
    else:
        # 변화량이 가장 큰 진입 이벤트 선택
        peak_idx = candidates['변화량'].idxmax()
    
    peak_value = df.loc[peak_idx, CONFIG['target_column']]
    peak_time = df.loc[peak_idx, 'CURRTIME'] if 'CURRTIME' in df.columns else ''
    
    print(f"     → 선택된 이벤트: {peak_time} | TOTALCNT: {peak_value:.0f}")
    
    # 전후 30개 데이터 추출 (15개 전 ~ 14개 후 = 30개)
    start_idx = max(0, peak_idx - 15)
    end_idx = min(len(df), peak_idx + 15)
    
    # 정확히 30개 맞추기
    if end_idx - start_idx < CONFIG['data_range']:
        if start_idx == 0:
            end_idx = min(len(df), CONFIG['data_range'])
        else:
            start_idx = max(0, end_idx - CONFIG['data_range'])
    
    range_data = df.iloc[start_idx:end_idx].copy()
    
    # 피크 지점 표시
    range_data['피크여부'] = ''
    range_data.loc[peak_idx, '피크여부'] = '★'
    
    # 필요한 컬럼만 선택
    output_cols = ['CURRTIME', CONFIG['target_column'], '이전값', '변화량', '피크여부']
    
    # 추가 컬럼 (있으면)
    extra_cols = ['M14AM14B', 'M14AM14BSUM', 'M14BM14A', 'M14AM10ASUM',
                  'M14.QUE.ALL.CURRENTQCREATED', 'M14.QUE.ALL.CURRENTQCOMPLETED',
                  'M14.QUE.OHT.OHTUTIL', 'M14.QUE.ALL.TRANSPORT4MINOVERCNT']
    
    for col in extra_cols:
        if col in df.columns:
            output_cols.append(col)
    
    # 존재하는 컬럼만 필터
    output_cols = [c for c in output_cols if c in range_data.columns]
    output_df = range_data[output_cols].copy()
    
    # 시간 포맷
    if 'CURRTIME' in output_df.columns:
        output_df['CURRTIME'] = output_df['CURRTIME'].dt.strftime('%Y-%m-%d %H:%M')
    
    # CSV 저장
    filename = f"급상승_{threshold}대_30개.csv"
    output_df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"     ✅ 저장: {filename} ({len(output_df)}개 데이터)")

# ============================================================================
# 완료
# ============================================================================
print("\n" + "=" * 70)
print("✅ 완료! 생성된 파일:")
for threshold in CONFIG['thresholds']:
    print(f"   → 급상승_{threshold}대_30개.csv")
print("=" * 70)