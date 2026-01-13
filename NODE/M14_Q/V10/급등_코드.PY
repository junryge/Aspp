# -*- coding: utf-8 -*-
"""
================================================================================
🚀 "확 튀는" 데이터 추출기
- 범위별 (3000, 2500, 2200, 2000, 1700대) 급상승 구간 30개씩 추출
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
    'input_file': 'M14_평가.CSV',  # 또는 다른 데이터 파일
    'output_file': f'급상승_데이터_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
    'target_column': 'TOTALCNT',
    
    # 범위 설정 (각 범위에서 30개씩)
    'ranges': [
        {'name': '3000대', 'min': 3000, 'max': 3999},
        {'name': '2500대', 'min': 2500, 'max': 2999},
        {'name': '2200대', 'min': 2200, 'max': 2499},
        {'name': '2000대', 'min': 2000, 'max': 2199},
        {'name': '1700대', 'min': 1700, 'max': 1999},
    ],
    'samples_per_range': 30,
    
    # "확 튀는" 기준 (이전 대비 증가량)
    'min_jump': 50,  # 최소 50 이상 상승
}

print("=" * 70)
print("🚀 '확 튀는' 데이터 추출기")
print("   범위별 급상승 구간 추출")
print("=" * 70)

# ============================================================================
# 데이터 로드
# ============================================================================
print(f"\n[1/4] 데이터 로드 중... ({CONFIG['input_file']})")

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

print(f"  → 총 데이터: {len(df):,}행")

# 시간 파싱
if 'CURRTIME' in df.columns:
    df['CURRTIME'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M', errors='coerce')
    df = df.dropna(subset=['CURRTIME']).sort_values('CURRTIME').reset_index(drop=True)

# 숫자 변환
df[CONFIG['target_column']] = pd.to_numeric(df[CONFIG['target_column']], errors='coerce').fillna(0)

# ============================================================================
# "확 튀는" 구간 계산
# ============================================================================
print("\n[2/4] 급상승 구간 계산 중...")

# 이전 대비 변화량
df['변화량_1분'] = df[CONFIG['target_column']].diff(1)
df['변화량_5분'] = df[CONFIG['target_column']].diff(5)
df['변화량_10분'] = df[CONFIG['target_column']].diff(10)

# 5분 전 값
df['5분전값'] = df[CONFIG['target_column']].shift(5)
df['10분전값'] = df[CONFIG['target_column']].shift(10)

# 급상승 점수 (변화량 합산)
df['급상승점수'] = (
    df['변화량_1분'].fillna(0).clip(lower=0) * 3 +  # 최근 1분 가중치
    df['변화량_5분'].fillna(0).clip(lower=0) * 2 +  # 5분 가중치
    df['변화량_10분'].fillna(0).clip(lower=0) * 1    # 10분 가중치
)

print(f"  → 급상승점수 최대: {df['급상승점수'].max():.1f}")

# ============================================================================
# 범위별 추출
# ============================================================================
print("\n[3/4] 범위별 데이터 추출 중...")

results = []

for r in CONFIG['ranges']:
    range_name = r['name']
    min_val = r['min']
    max_val = r['max']
    
    # 해당 범위 필터링
    mask = (df[CONFIG['target_column']] >= min_val) & (df[CONFIG['target_column']] <= max_val)
    range_df = df[mask].copy()
    
    if len(range_df) == 0:
        print(f"  ⚠️ {range_name}: 데이터 없음")
        continue
    
    # 급상승점수가 높은 순으로 정렬
    range_df = range_df.sort_values('급상승점수', ascending=False)
    
    # 상위 N개 추출 (중복 시간대 제거를 위해 여유있게 추출 후 필터링)
    selected = []
    used_times = set()
    
    for idx, row in range_df.iterrows():
        if len(selected) >= CONFIG['samples_per_range']:
            break
        
        # 10분 이내 중복 제거
        current_time = row['CURRTIME'] if 'CURRTIME' in row else idx
        
        # 시간 기반 중복 체크 (비슷한 시간대 제외)
        is_duplicate = False
        if isinstance(current_time, pd.Timestamp):
            for ut in used_times:
                if abs((current_time - ut).total_seconds()) < 600:  # 10분
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            selected.append(row)
            if isinstance(current_time, pd.Timestamp):
                used_times.add(current_time)
    
    print(f"  ✅ {range_name}: {len(selected)}개 추출 (전체 {len(range_df):,}개 중)")
    
    # 결과에 범위 정보 추가
    for row in selected:
        result_row = {
            '범위': range_name,
            '시간': row['CURRTIME'].strftime('%Y-%m-%d %H:%M') if 'CURRTIME' in row and pd.notna(row['CURRTIME']) else '',
            'TOTALCNT': round(row[CONFIG['target_column']], 1),
            '5분전값': round(row['5분전값'], 1) if pd.notna(row['5분전값']) else '',
            '10분전값': round(row['10분전값'], 1) if pd.notna(row['10분전값']) else '',
            '5분변화': round(row['변화량_5분'], 1) if pd.notna(row['변화량_5분']) else '',
            '10분변화': round(row['변화량_10분'], 1) if pd.notna(row['변화량_10분']) else '',
            '급상승점수': round(row['급상승점수'], 1),
        }
        
        # 주요 컬럼 추가 (있으면)
        for col in ['M14AM14B', 'M14AM14BSUM', 'QUEUE_GAP', 
                    'M14.QUE.ALL.CURRENTQCREATED', 'M14.QUE.ALL.CURRENTQCOMPLETED',
                    'M14.QUE.OHT.OHTUTIL', 'M14.QUE.ALL.TRANSPORT4MINOVERCNT']:
            if col in row:
                result_row[col] = round(row[col], 1) if pd.notna(row[col]) else ''
        
        results.append(result_row)

# ============================================================================
# 결과 저장
# ============================================================================
print("\n[4/4] 결과 저장...")

df_result = pd.DataFrame(results)

# 범위별 정렬
range_order = {r['name']: i for i, r in enumerate(CONFIG['ranges'])}
df_result['범위순서'] = df_result['범위'].map(range_order)
df_result = df_result.sort_values(['범위순서', '급상승점수'], ascending=[True, False])
df_result = df_result.drop(columns=['범위순서'])

df_result.to_csv(CONFIG['output_file'], index=False, encoding='utf-8-sig')
print(f"  → 저장: {CONFIG['output_file']}")

# ============================================================================
# 요약 출력
# ============================================================================
print("\n" + "=" * 70)
print("📊 추출 결과 요약")
print("=" * 70)

summary = df_result.groupby('범위').agg({
    'TOTALCNT': ['count', 'min', 'max', 'mean'],
    '급상승점수': ['min', 'max', 'mean']
}).round(1)

print(f"\n{'범위':<10} {'개수':>6} {'최소값':>8} {'최대값':>8} {'평균':>8} {'급상승점수':>10}")
print("-" * 60)

for range_name in [r['name'] for r in CONFIG['ranges']]:
    range_data = df_result[df_result['범위'] == range_name]
    if len(range_data) > 0:
        cnt = len(range_data)
        min_v = range_data['TOTALCNT'].min()
        max_v = range_data['TOTALCNT'].max()
        avg_v = range_data['TOTALCNT'].mean()
        avg_score = range_data['급상승점수'].mean()
        print(f"{range_name:<10} {cnt:>6} {min_v:>8.0f} {max_v:>8.0f} {avg_v:>8.0f} {avg_score:>10.1f}")

print(f"\n총 추출 데이터: {len(df_result)}개")
print(f"\n✅ 완료! → {CONFIG['output_file']}")
print("=" * 70)

# ============================================================================
# 샘플 출력
# ============================================================================
print("\n📋 샘플 데이터 (각 범위 상위 3개):")
print("-" * 70)

for range_name in [r['name'] for r in CONFIG['ranges']]:
    range_data = df_result[df_result['범위'] == range_name].head(3)
    if len(range_data) > 0:
        print(f"\n🔥 {range_name}:")
        for _, row in range_data.iterrows():
            print(f"   {row['시간']} | TOTALCNT: {row['TOTALCNT']:.0f} | "
                  f"5분변화: +{row['5분변화']:.0f} | 급상승점수: {row['급상승점수']:.0f}")