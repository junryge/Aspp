import pandas as pd

# CSV 로드
df = pd.read_csv('ahso_alarm.csv', encoding='utf-8-sig')

# 기존 컬럼 저장
original_cols = df.columns.tolist()
print("=== 기존 컬럼 ===")
print(original_cols)

# AHSO 컬럼 찾기 & 문자열 변환
ahso_col = [col for col in df.columns if 'AHSO' in col][0]
df[ahso_col] = df[ahso_col].astype(str)

# 날짜 추출 (임시)
df['DATETIME'] = pd.to_datetime(
    df[ahso_col].str.extract(r'(\d{12})')[0],
    format='%Y%m%d%H%M',
    errors='coerce'
)

# ========== 기간 설정 ==========
START_DATE = '202601010001'
END_DATE = '202601312359'

start_dt = pd.to_datetime(START_DATE, format='%Y%m%d%H%M')
end_dt = pd.to_datetime(END_DATE, format='%Y%m%d%H%M')

# 전체 시간 범위 (1분 단위)
full_range = pd.date_range(start=start_dt, end=end_dt, freq='1min')

# 기존 데이터에 있는 시간
existing_times = set(df['DATETIME'].dropna())

# 빠진 시간만 찾기
missing_times = [t for t in full_range if t not in existing_times]
print(f"\n빠진 시간: {len(missing_times)}건")

# 빠진 시간대 행 생성 (기존 컬럼 구조 유지)
new_rows = []
for t in missing_times:
    row = {}
    for col in original_cols:
        if col == ahso_col:
            row[col] = 'AHSO' + t.strftime('%Y%m%d%H%M')
        elif col == '분류':
            row[col] = ''
        else:
            row[col] = 0  # 숫자 컬럼은 0
    row['DATETIME'] = t
    new_rows.append(row)

# 새 행 추가
new_df = pd.DataFrame(new_rows)
df = pd.concat([df, new_df], ignore_index=True)

# 정렬
df = df.sort_values('DATETIME').reset_index(drop=True)

# DATETIME 컬럼 삭제 or 유지 선택
# df = df.drop(columns=['DATETIME'])  # 삭제하려면 주석 해제

# 기존 컬럼 순서 + DATETIME
df = df[original_cols + ['DATETIME']]

print(f"\n=== 최종 결과 ===")
print(f"총 {len(df)}건")
print(df.head(20))

# 저장
df.to_csv('ahso_alarm_full_range.csv', index=False, encoding='utf-8-sig')