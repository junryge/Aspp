# -*- coding: utf-8 -*-
import pandas as pd

# === 파일 경로 ===
FILE1 = 'M14_Q_20241201_2025120212_S2.csv'
FILE2 = 'M14_Q_20250909_20251205_S1.csv'
OUTPUT = 'merged_output.csv'

# === 로딩 ===
def load_csv(path):
    for enc in ['utf-8', 'cp949', 'euc-kr']:
        try:
            return pd.read_csv(path, encoding=enc, on_bad_lines='skip', low_memory=False)
        except:
            continue
    raise Exception(f"로딩 실패: {path}")

df1 = load_csv(FILE1)
df2 = load_csv(FILE2)

# === 날짜 파싱 ===
df1['CURRTIME'] = df1['CURRTIME'].astype(str).str.strip()
df1['datetime'] = pd.to_datetime(df1['CURRTIME'], format='%Y%m%d%H%M', errors='coerce')

df2['CRT_TM'] = df2['CRT_TM'].astype(str).str.strip()
df2['datetime'] = pd.to_datetime(df2['CRT_TM'], format='%Y-%m-%d %H:%M', errors='coerce')

print(f"FILE1 파싱: {df1['datetime'].notna().sum()}/{len(df1)}")
print(f"FILE2 파싱: {df2['datetime'].notna().sum()}/{len(df2)}")

# === 전체 시간 범위 ===
all_times = pd.concat([df1['datetime'].dropna(), df2['datetime'].dropna()])
full_range = pd.date_range(start=all_times.min(), end=all_times.max(), freq='1min')
df_base = pd.DataFrame({'datetime': full_range})
print(f"범위: {all_times.min()} ~ {all_times.max()}")
print(f"전체 행: {len(df_base)}")

# === 병합 ===
df1_cols = [c for c in df1.columns if c not in ['datetime', 'CURRTIME']]
df2_cols = [c for c in df2.columns if c not in ['datetime', 'CRT_TM']]

df_merged = df_base.merge(df1[['datetime'] + df1_cols], on='datetime', how='left')
df_merged = df_merged.merge(df2[['datetime'] + df2_cols], on='datetime', how='left')

# === 누락 데이터 보간 (핵심 추가!) ===
data_cols = df1_cols + df2_cols

# 숫자 컬럼만 보간
for col in data_cols:
    df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')

print(f"\n=== 보간 전 누락 ===")
before_missing = df_merged[data_cols].isna().sum().sum()
print(f"총 누락: {before_missing}")

# 1) 선형 보간 (최대 10분 갭까지만)
df_merged[data_cols] = df_merged[data_cols].interpolate(method='linear', limit=10, limit_direction='both')

# 2) 남은 건 앞뒤값으로 채우기
df_merged[data_cols] = df_merged[data_cols].ffill(limit=5)
df_merged[data_cols] = df_merged[data_cols].bfill(limit=5)

print(f"\n=== 보간 후 누락 ===")
after_missing = df_merged[data_cols].isna().sum().sum()
print(f"총 누락: {after_missing}")
print(f"채운 개수: {before_missing - after_missing}")

# === CURRTIME 복원 ===
df_merged['CURRTIME'] = df_merged['datetime'].dt.strftime('%Y%m%d%H%M')

# === 컬럼 정리 ===
cols = ['CURRTIME'] + data_cols
df_merged = df_merged[cols]

# === 결과 ===
print(f"\n=== 최종 결과 ===")
print(f"총 행: {len(df_merged)}")
print(f"FILE1 컬럼 채움률: {df_merged[df1_cols[0]].notna().sum()}/{len(df_merged)}")
print(f"FILE2 컬럼 채움률: {df_merged[df2_cols[0]].notna().sum()}/{len(df_merged)}")

df_merged.to_csv(OUTPUT, index=False, encoding='utf-8-sig')
print(f"\n저장: {OUTPUT}")