import pandas as pd

# CSV 로드
df = pd.read_csv('ahso_alarm.csv', encoding='utf-8-sig')

# 기존 컬럼 확인
print("=== 기존 컬럼 ===")
print(df.columns.tolist())

# AHSO 컬럼 문자열 변환
ahso_col = [col for col in df.columns if 'AHSO' in col][0]
df[ahso_col] = df[ahso_col].astype(str)

# 날짜 추출
df['DATETIME'] = pd.to_datetime(
    df[ahso_col].str.extract(r'(\d{12})')[0],
    format='%Y%m%d%H%M',
    errors='coerce'
)

# ========== 기간 설정 ==========
START_DATE = '202601010001'  # 시작: 2026-01-01 00:01
END_DATE = '202601312359'    # 종료: 2026-01-31 23:59

start_dt = pd.to_datetime(START_DATE, format='%Y%m%d%H%M')
end_dt = pd.to_datetime(END_DATE, format='%Y%m%d%H%M')

# 1분 단위 전체 시간 범위 생성
full_range = pd.date_range(start=start_dt, end=end_dt, freq='1min')
full_df = pd.DataFrame({'DATETIME': full_range})

# 원본 데이터와 병합
df = full_df.merge(df, on='DATETIME', how='left')

# ========== 기존 컬럼 유지하면서 결측치만 채우기 ==========
for col in df.columns:
    if col == 'DATETIME':
        continue
    elif col == ahso_col:
        # AHSO 컬럼: 빈 값은 AHSO + 날짜로 채우기
        mask = df[col].isna() | (df[col] == 'nan')
        df.loc[mask, col] = 'AHSO' + df.loc[mask, 'DATETIME'].dt.strftime('%Y%m%d%H%M')
    elif df[col].dtype in ['float64', 'int64'] or col in ['AHTO+AHSO', 'REWORK']:
        # 숫자 컬럼: 0으로
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    else:
        # 문자 컬럼: 빈 문자열
        df[col] = df[col].fillna('')

# 정렬
df = df.sort_values('DATETIME').reset_index(drop=True)

# DATETIME 컬럼 위치 맨 뒤로 (기존 컬럼 순서 유지)
cols = [c for c in df.columns if c != 'DATETIME'] + ['DATETIME']
df = df[cols]

# 결과 확인
print(f"\n=== 최종 컬럼 ===")
print(df.columns.tolist())
print(f"\n총 {len(df)}건")
print(df.head(20))

# 저장
df.to_csv('ahso_alarm_full_range.csv', index=False, encoding='utf-8-sig')