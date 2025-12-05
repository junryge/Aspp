# -*- coding: utf-8 -*-
import pandas as pd

# === 파일 경로 ===
FILE1 = 'M14_Q_20241201_2025120212_S2.csv'
FILE2 = 'M14_Q_20250909_20251205_S1.csv'
OUTPUT = 'merged_output.csv'
REPORT = 'missing_report.csv'  # 누락 리포트

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

data_cols = df1_cols + df2_cols
for col in data_cols:
    df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')

# ==============================
# 누락 분석 (보간 전)
# ==============================
print(f"\n{'='*50}")
print("누락 분석 (보간 전)")
print('='*50)

# 1) 컬럼별 누락 통계
missing_stats = []
for col in data_cols:
    missing_cnt = df_merged[col].isna().sum()
    total = len(df_merged)
    pct = missing_cnt / total * 100
    source = 'FILE1' if col in df1_cols else 'FILE2'
    missing_stats.append({
        'column': col,
        'source': source,
        'missing_count': missing_cnt,
        'total': total,
        'missing_pct': round(pct, 2)
    })

df_missing = pd.DataFrame(missing_stats)
df_missing = df_missing.sort_values('missing_pct', ascending=False)

print(f"\n[컬럼별 누락률 TOP 20]")
print(df_missing.head(20).to_string(index=False))

# 2) 시간대별 누락 (일별)
df_merged['date'] = df_merged['datetime'].dt.date
daily_missing = df_merged.groupby('date')[data_cols].apply(lambda x: x.isna().sum().sum())
daily_total = df_merged.groupby('date')[data_cols].apply(lambda x: x.size)
daily_pct = (daily_missing / daily_total * 100).round(2)

print(f"\n[일별 누락률]")
for dt, pct in daily_pct.items():
    if pct > 0:
        print(f"  {dt}: {pct:.2f}%")

# 3) FILE1 vs FILE2 비교
file1_missing = df_merged[df1_cols].isna().sum().sum()
file1_total = len(df_merged) * len(df1_cols)
file2_missing = df_merged[df2_cols].isna().sum().sum()
file2_total = len(df_merged) * len(df2_cols)

print(f"\n[파일별 누락 요약]")
print(f"  FILE1: {file1_missing:,}/{file1_total:,} ({file1_missing/file1_total*100:.2f}%)")
print(f"  FILE2: {file2_missing:,}/{file2_total:,} ({file2_missing/file2_total*100:.2f}%)")

# 4) 완전 누락 시간대 찾기
df_merged['all_missing'] = df_merged[data_cols].isna().all(axis=1)
missing_times = df_merged[df_merged['all_missing']]['datetime']
print(f"\n[완전 누락 시간대]: {len(missing_times)}개")
if len(missing_times) > 0 and len(missing_times) <= 20:
    for t in missing_times:
        print(f"  {t}")
elif len(missing_times) > 20:
    print(f"  처음 5개: {missing_times.head().tolist()}")
    print(f"  마지막 5개: {missing_times.tail().tolist()}")

# ==============================
# 보간
# ==============================
print(f"\n{'='*50}")
print("보간 수행")
print('='*50)

before_missing = df_merged[data_cols].isna().sum().sum()

df_merged[data_cols] = df_merged[data_cols].interpolate(method='linear', limit=10, limit_direction='both')
df_merged[data_cols] = df_merged[data_cols].ffill(limit=5)
df_merged[data_cols] = df_merged[data_cols].bfill(limit=5)

after_missing = df_merged[data_cols].isna().sum().sum()
print(f"보간 전: {before_missing:,}개 누락")
print(f"보간 후: {after_missing:,}개 누락")
print(f"채운 개수: {before_missing - after_missing:,}개")

# ==============================
# 보간 후 누락 분석
# ==============================
print(f"\n[보간 후에도 남은 누락 컬럼]")
for col in data_cols:
    remaining = df_merged[col].isna().sum()
    if remaining > 0:
        pct = remaining / len(df_merged) * 100
        print(f"  {col}: {remaining}개 ({pct:.2f}%)")

# === CURRTIME 복원 ===
df_merged['CURRTIME'] = df_merged['datetime'].dt.strftime('%Y%m%d%H%M')

# === 저장 ===
# 1) 메인 데이터
cols = ['CURRTIME'] + data_cols
df_merged[cols].to_csv(OUTPUT, index=False, encoding='utf-8-sig')
print(f"\n데이터 저장: {OUTPUT}")

# 2) 누락 리포트
df_missing.to_csv(REPORT, index=False, encoding='utf-8-sig')
print(f"리포트 저장: {REPORT}")

print(f"\n{'='*50}")
print("완료!")
print('='*50)