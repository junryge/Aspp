# -*- coding: utf-8 -*-
"""
================================================================================
🚀 "갑자기 튀는" 데이터 추출기
- 각 범위(3000, 2500, 2200, 2000, 1700대)로 처음 진입하는 순간 30개씩
- 원본 데이터 형식 그대로 유지
- 범위별 별도 CSV 저장
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
    'input_file': 'M14_평가.CSV',  # 데이터 파일
    'target_column': 'TOTALCNT',
    
    # 범위 설정: 해당 값 이상으로 "튀어오르는" 순간 추출
    'thresholds': [3000, 2500, 2200, 2000, 1700],
    'samples_per_range': 30,
}

print("=" * 70)
print("🚀 '갑자기 튀는' 데이터 추출기")
print("   각 범위로 진입하는 순간 추출 → 원본 형식 그대로 저장")
print("=" * 70)

# ============================================================================
# 데이터 로드
# ============================================================================
print(f"\n[1/3] 데이터 로드 중... ({CONFIG['input_file']})")

encodings = ['utf-8', 'cp949', 'euc-kr']
df = None
for enc in encodings:
    try:
        df = pd.read_csv(CONFIG['input_file'], encoding=enc, on_bad_lines='skip')
        print(f"  ✅ 인코딩: {enc}")
        break
    except:
        continue

if df is None:
    raise ValueError("파일을 읽을 수 없습니다!")

print(f"  → 총 데이터: {len(df):,}행, {len(df.columns)}개 컬럼")

# 원본 컬럼 저장
original_columns = df.columns.tolist()

# 시간 정렬 (CURRTIME 있으면)
if 'CURRTIME' in df.columns:
    df['_CURRTIME_DT'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M', errors='coerce')
    df = df.dropna(subset=['_CURRTIME_DT']).sort_values('_CURRTIME_DT').reset_index(drop=True)

# 숫자 변환
df[CONFIG['target_column']] = pd.to_numeric(df[CONFIG['target_column']], errors='coerce').fillna(0)

# 이전 값 계산 (추출용, 저장 안함)
df['_이전값'] = df[CONFIG['target_column']].shift(1)
df['_변화량'] = df[CONFIG['target_column']] - df['_이전값']

# ============================================================================
# 범위별 "갑자기 튀는" 구간 추출
# ============================================================================
print("\n[2/3] 범위별 급상승 진입 구간 추출 중...")

output_files = []

for threshold in CONFIG['thresholds']:
    range_name = f"{threshold}대"
    output_file = f"급상승_{threshold}대_30개.csv"
    
    print(f"\n  🔍 {range_name} 추출 중...")
    
    # "갑자기 튀는" 조건:
    # - 현재값 >= threshold
    # - 이전값 < threshold (처음 진입)
    mask = (df[CONFIG['target_column']] >= threshold) & (df['_이전값'] < threshold)
    
    jump_df = df[mask].copy()
    
    if len(jump_df) == 0:
        print(f"     ⚠️ {range_name}: 진입 구간 없음")
        continue
    
    print(f"     → 전체 진입 구간: {len(jump_df)}개")
    
    # 변화량이 큰 순으로 정렬 (더 "확" 튀는 순)
    jump_df = jump_df.sort_values('_변화량', ascending=False)
    
    # 상위 N개 추출 (중복 시간대 제거)
    selected_indices = []
    used_times = set()
    
    for idx, row in jump_df.iterrows():
        if len(selected_indices) >= CONFIG['samples_per_range']:
            break
        
        # 30분 이내 중복 제거
        is_duplicate = False
        if '_CURRTIME_DT' in row and pd.notna(row['_CURRTIME_DT']):
            current_time = row['_CURRTIME_DT']
            for ut in used_times:
                if abs((current_time - ut).total_seconds()) < 1800:  # 30분
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                selected_indices.append(idx)
                used_times.add(current_time)
        else:
            selected_indices.append(idx)
    
    print(f"     ✅ 추출: {len(selected_indices)}개")
    
    if len(selected_indices) == 0:
        continue
    
    # 원본 컬럼만 추출하여 저장
    result_df = df.loc[selected_indices, original_columns].copy()
    
    # 시간순 정렬
    if 'CURRTIME' in result_df.columns:
        result_df = result_df.sort_values('CURRTIME')
    
    # CSV 저장
    result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    output_files.append((range_name, output_file, len(result_df)))
    print(f"     💾 저장: {output_file}")

# ============================================================================
# 결과 요약
# ============================================================================
print("\n" + "=" * 70)
print("📊 추출 결과 요약")
print("=" * 70)

print(f"\n{'범위':<12} {'파일명':<30} {'개수':>6}")
print("-" * 50)

total_count = 0
for range_name, filename, count in output_files:
    print(f"{range_name:<12} {filename:<30} {count:>6}개")
    total_count += count

print("-" * 50)
print(f"{'총계':<12} {'':<30} {total_count:>6}개")

print(f"\n✅ 완료! {len(output_files)}개 파일 생성")
print("=" * 70)