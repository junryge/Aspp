# -*- coding: utf-8 -*-
"""
V13 PDT 데이터 - 누락 시간 0으로 채우기
"""
import pandas as pd
import numpy as np

# ============================================================
# 설정
# ============================================================
PDT_FILE = 'V13_PDT_20250909_20251231.csv'
OUTPUT_FILE = 'V13_PDT_FILLED_20250909_20251231.csv'

# ============================================================
# 데이터 로드
# ============================================================
print("데이터 로드...")
try:
    df = pd.read_csv(PDT_FILE, encoding='utf-8')
except:
    try:
        df = pd.read_csv(PDT_FILE, encoding='cp949')
    except:
        df = pd.read_csv(PDT_FILE, encoding='euc-kr')

print(f"원본: {len(df):,}행")

# CURRTIME → datetime
df['CURRTIME'] = pd.to_datetime(df['CURRTIME'], format='%Y%m%d%H%M')
df = df.drop_duplicates(subset='CURRTIME', keep='last')

# 전체 시간 범위 생성 (1분 간격)
full_range = pd.date_range(start='2025-09-09 00:00', end='2025-12-31 23:59', freq='1min')
df_full = pd.DataFrame({'CURRTIME': full_range})

print(f"전체 시간: {len(df_full):,}행")

# 병합
df_merged = df_full.merge(df, on='CURRTIME', how='left')

# 누락값 0으로
for col in df_merged.columns:
    if col != 'CURRTIME':
        df_merged[col] = df_merged[col].fillna(0)

# CURRTIME 형식 변환 (YYYYMMDDHHMM)
df_merged['CURRTIME'] = df_merged['CURRTIME'].dt.strftime('%Y%m%d%H%M')

print(f"결과: {len(df_merged):,}행")

# 저장
df_merged.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
print(f"저장: {OUTPUT_FILE}")