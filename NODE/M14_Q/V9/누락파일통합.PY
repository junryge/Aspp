import pandas as pd
from datetime import datetime, timedelta

# === 파일 경로 설정 ===
FILE1 = 'file1.csv'  # CURRTIME 컬럼 있는 파일
FILE2 = 'file2.csv'  # CRT_TM 컬럼 있는 파일
OUTPUT = 'merged_output.csv'

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

print(f"파일1 컬럼: {df1.columns.tolist()}")
print(f"파일2 컬럼: {df2.columns.tolist()}")

# === 2. 날짜 컬럼 통일 (CURRTIME 기준) ===
# 파일1: CURRTIME (yyyyMMdd HHmm) -> datetime
df1['CURRTIME'] = df1['CURRTIME'].astype(str).str.strip()
df1['datetime'] = pd.to_datetime(df1['CURRTIME'], format='%Y%m%d %H%M', errors='coerce')

# 파일2: CRT_TM (yyyy-MM-dd H:mm) -> datetime
df2['CRT_TM'] = df2['CRT_TM'].astype(str).str.strip()
df2['datetime'] = pd.to_datetime(df2['CRT_TM'], format='%Y-%m-%d %H:%M', errors='coerce')

# === 3. 전체 시간 범위 생성 (1분 단위) ===
all_times = pd.concat([df1['datetime'].dropna(), df2['datetime'].dropna()])
start_time = all_times.min()
end_time = all_times.max()

print(f"시간 범위: {start_time} ~ {end_time}")

# 전체 시간 인덱스 생성
full_range = pd.date_range(start=start_time, end=end_time, freq='1min')
df_base = pd.DataFrame({'datetime': full_range})

print(f"전체 행 수: {len(df_base)}")

# === 4. 파일1, 파일2 병합 ===
# 파일1 병합
df1_cols = [c for c in df1.columns if c not in ['datetime', 'CURRTIME']]
df_merged = df_base.merge(df1[['datetime'] + df1_cols], on='datetime', how='left')

# 파일2 병합
df2_cols = [c for c in df2.columns if c not in ['datetime', 'CRT_TM']]
df_merged = df_merged.merge(df2[['datetime'] + df2_cols], on='datetime', how='left')

# === 5. CURRTIME 형식 컬럼 추가 ===
df_merged['CURRTIME'] = df_merged['datetime'].dt.strftime('%Y%m%d %H%M')

# 컬럼 순서 정리
cols = ['CURRTIME', 'datetime'] + df1_cols + df2_cols
df_merged = df_merged[cols]

# === 6. 결과 확인 및 저장 ===
print(f"\n=== 통합 결과 ===")
print(f"총 행 수: {len(df_merged)}")
print(f"파일1 데이터 있는 행: {df_merged[df1_cols[0]].notna().sum()}")
print(f"파일2 데이터 있는 행: {df_merged[df2_cols[0]].notna().sum()}")
print(f"둘 다 있는 행: {(df_merged[df1_cols[0]].notna() & df_merged[df2_cols[0]].notna()).sum()}")

# 누락 통계
print(f"\n=== 누락 통계 ===")
for col in df1_cols + df2_cols:
    missing = df_merged[col].isna().sum()
    pct = missing / len(df_merged) * 100
    print(f"{col}: {missing}개 누락 ({pct:.1f}%)")

# 저장
df_merged.to_csv(OUTPUT, index=False, encoding='utf-8-sig')
print(f"\n저장 완료: {OUTPUT}")