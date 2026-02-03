import pandas as pd

# CSV 로드
df = pd.read_csv('ahso_alarm.csv', encoding='utf-8-sig')

# AHSO 컬럼 찾기 & 문자열 변환
ahso_col = [col for col in df.columns if 'AHSO' in str(col)][0]
df[ahso_col] = df[ahso_col].astype(str)

# 기존 시간 추출
df['_TIME'] = df[ahso_col].str.extract(r'(\d{12})')[0]
existing_times = set(df['_TIME'].dropna())

# ========== 기간 설정 (직접 입력) ==========
START_DATE = '202601010001'  # 여기 수정
END_DATE = '202601312359'    # 여기 수정

start_dt = pd.to_datetime(START_DATE, format='%Y%m%d%H%M')
end_dt = pd.to_datetime(END_DATE, format='%Y%m%d%H%M')

print(f"설정 범위: {START_DATE} ~ {END_DATE}")

# 전체 시간 범위 (1분 단위)
full_range = pd.date_range(start=start_dt, end=end_dt, freq='1min')
all_times = [t.strftime('%Y%m%d%H%M') for t in full_range]

# 빠진 시간 찾기
missing_times = [t for t in all_times if t not in existing_times]
print(f"빠진 시간: {len(missing_times)}개")

# 빠진 시간 행 생성
new_rows = []
for t in missing_times:
    row = {
        '분류': '',
        'AHTO+AHSO': 0,
        'REWORK': 0,
        ahso_col: f'AHSO{t} ',
    }
    new_rows.append(row)

# 새 행 추가
new_df = pd.DataFrame(new_rows)
df = pd.concat([df, new_df], ignore_index=True)

# 정렬
df['_TIME'] = df[ahso_col].str.extract(r'(\d{12})')[0]
df = df.sort_values('_TIME').reset_index(drop=True)
df = df.drop(columns=['_TIME'])

print(f"\n총 {len(df)}행")
print(df.head(30))

# 저장
df.to_csv('ahso_alarm_full_range.csv', index=False, encoding='utf-8-sig')