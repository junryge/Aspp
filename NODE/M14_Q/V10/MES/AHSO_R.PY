import pandas as pd

# CSV 로드
df = pd.read_csv('ahso_alarm.csv', encoding='utf-8-sig')

# 분류 컬럼 문자열 변환
df['분류'] = df['분류'].astype(str).str.strip()

# 기존 시간 추출
existing_times = set(df['분류'].tolist())
print(f"기존 데이터: {len(existing_times)}개")

# ========== 기간 설정 ==========
START_DATE = '202601010007'
END_DATE = '202601312359'

start_dt = pd.to_datetime(START_DATE, format='%Y%m%d%H%M')
end_dt = pd.to_datetime(END_DATE, format='%Y%m%d%H%M')

# 전체 시간 범위 (1분 단위)
full_range = pd.date_range(start=start_dt, end=end_dt, freq='1min')
all_times = [t.strftime('%Y%m%d%H%M') for t in full_range]

# 빠진 시간만 찾기 (기존에 없는 것만!)
missing_times = [t for t in all_times if t not in existing_times]
print(f"빠진 시간: {len(missing_times)}개")

# 빠진 시간 행 생성
new_rows = []
for t in missing_times:
    row = {'분류': t}
    for col in df.columns:
        if col != '분류':
            row[col] = 0
    new_rows.append(row)

# 기존 데이터 + 새 행 합치기
new_df = pd.DataFrame(new_rows)
df = pd.concat([df, new_df], ignore_index=True)

# 혹시 모를 중복 제거 (분류 기준)
df = df.drop_duplicates(subset=['분류'], keep='first')

# 정렬
df = df.sort_values('분류').reset_index(drop=True)

print(f"\n총 {len(df)}행 (중복 제거 완료)")
print(df.head(30))

# 저장
df.to_csv('ahso_alarm_full_range.csv', index=False, encoding='utf-8-sig')