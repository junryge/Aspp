import pandas as pd
from datetime import datetime, timedelta

# === 파일 경로 설정 ===
FILE1 = 'file1.csv'
FILE2 = 'file2.csv'
OUTPUT_MERGED = 'merged_output.csv'
OUTPUT_MISSING = 'missing_data.csv'  # 누락 데이터 하나로

# === 파일 로딩 함수 ===
def load_csv(path):
    for enc in ['utf-8', 'cp949', 'euc-kr']:
        try:
            return pd.read_csv(path, encoding=enc, on_bad_lines='skip', low_memory=False)
        except:
            continue
    raise Exception(f"파일 로딩 실패: {path}")

# === 1. 파일 로딩 ===
df1 = load_csv(FILE1)
df2 = load_csv(FILE2)

# === 2. 날짜 컬럼 통일 ===
df1['CURRTIME'] = df1['CURRTIME'].astype(str).str.replace(' ', '').str.strip()
df1['datetime'] = pd.to_datetime(df1['CURRTIME'], format='%Y%m%d%H%M', errors='coerce')

df2['CRT_TM'] = df2['CRT_TM'].astype(str).str.strip()
df2['datetime'] = pd.to_datetime(df2['CRT_TM'], errors='coerce')

# === 3. 전체 시간 범위 생성 ===
all_times = pd.concat([df1['datetime'].dropna(), df2['datetime'].dropna()])
start_time = all_times.min()
end_time = all_times.max()

full_range = pd.date_range(start=start_time, end=end_time, freq='1min')
df_base = pd.DataFrame({'datetime': full_range})

# === 4. 병합 ===
df1_cols = [c for c in df1.columns if c not in ['datetime', 'CURRTIME']]
df2_cols = [c for c in df2.columns if c not in ['datetime', 'CRT_TM']]

df_merged = df_base.merge(df1[['datetime'] + df1_cols], on='datetime', how='left')
df_merged = df_merged.merge(df2[['datetime'] + df2_cols], on='datetime', how='left')
df_merged['CURRTIME'] = df_merged['datetime'].dt.strftime('%Y%m%d%H%M')

# === 5. 누락 데이터 추출 (하나로) ===
file1_missing = df_merged[df1_cols[0]].isna()
file2_missing = df_merged[df2_cols[0]].isna()
any_missing = file1_missing | file2_missing

df_missing = df_merged[any_missing][['CURRTIME', 'datetime']].copy()
df_missing['FILE1_MISSING'] = file1_missing[any_missing].astype(int)
df_missing['FILE2_MISSING'] = file2_missing[any_missing].astype(int)

print(f"누락 행 수: {len(df_missing)}")
df_missing.to_csv(OUTPUT_MISSING, index=False, encoding='utf-8-sig')
print(f"누락 데이터 저장: {OUTPUT_MISSING}")

# === 6. 선형 보간 ===
numeric_cols = df1_cols + df2_cols
for col in numeric_cols:
    df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')

df_merged[numeric_cols] = df_merged[numeric_cols].interpolate(method='linear', limit=30)
df_merged[numeric_cols] = df_merged[numeric_cols].ffill(limit=10)
df_merged[numeric_cols] = df_merged[numeric_cols].bfill(limit=10)

# === 7. 저장 ===
cols = ['CURRTIME', 'datetime'] + df1_cols + df2_cols
df_merged = df_merged[cols]
df_merged.to_csv(OUTPUT_MERGED, index=False, encoding='utf-8-sig')
print(f"통합 파일 저장: {OUTPUT_MERGED}")