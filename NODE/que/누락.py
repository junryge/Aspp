# -*- coding: utf-8 -*-
import pandas as pd

# === 파일 경로 ===
FILE1 = 'M14_Q_20241201_2025120212_S2.csv'  # CURRTIME (yyyyMMddHHmm)
FILE2 = 'M14_Q_20250909_20251205_S1.csv'    # CRT_TM (yyyy-MM-dd H:mm)
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

# === 날짜 파싱 (핵심 수정!) ===
# FILE1: 202509091003 -> 공백 없음!
df1['CURRTIME'] = df1['CURRTIME'].astype(str).str.strip()
df1['datetime'] = pd.to_datetime(df1['CURRTIME'], format='%Y%m%d%H%M', errors='coerce')

# FILE2: 2025-09-09 10:03
df2['CRT_TM'] = df2['CRT_TM'].astype(str).str.strip()
df2['datetime'] = pd.to_datetime(df2['CRT_TM'], format='%Y-%m-%d %H:%M', errors='coerce')

# === 파싱 결과 확인 ===
print(f"FILE1 파싱 성공: {df1['datetime'].notna().sum()}/{len(df1)}")
print(f"FILE2 파싱 성공: {df2['datetime'].notna().sum()}/{len(df2)}")

# === 전체 시간 범위 ===
all_times = pd.concat([df1['datetime'].dropna(), df2['datetime'].dropna()])
start_time = all_times.min()
end_time = all_times.max()
print(f"범위: {start_time} ~ {end_time}")

full_range = pd.date_range(start=start_time, end=end_time, freq='1min')
df_base = pd.DataFrame({'datetime': full_range})
print(f"전체 행: {len(df_base)}")

# === 병합 ===
df1_cols = [c for c in df1.columns if c not in ['datetime', 'CURRTIME']]
df2_cols = [c for c in df2.columns if c not in ['datetime', 'CRT_TM']]

df_merged = df_base.merge(df1[['datetime'] + df1_cols], on='datetime', how='left')
df_merged = df_merged.merge(df2[['datetime'] + df2_cols], on='datetime', how='left')

# === CURRTIME 복원 ===
df_merged['CURRTIME'] = df_merged['datetime'].dt.strftime('%Y%m%d%H%M')

# === 컬럼 정리 ===
cols = ['CURRTIME'] + df1_cols + df2_cols
df_merged = df_merged[cols]

# === 결과 ===
print(f"\n=== 결과 ===")
print(f"총 행: {len(df_merged)}")
print(f"FILE1 데이터: {df_merged[df1_cols[0]].notna().sum()}")
print(f"FILE2 데이터: {df_merged[df2_cols[0]].notna().sum()}")

df_merged.to_csv(OUTPUT, index=False, encoding='utf-8-sig')
print(f"저장: {OUTPUT}")