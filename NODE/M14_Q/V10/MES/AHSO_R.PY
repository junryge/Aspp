import pandas as pd

# CSV 로드
df = pd.read_csv('ahso_alarm.csv', encoding='utf-8-sig')

# ========== 기간 설정 ==========
START_DATE = '202511010000'  # 시작
END_DATE = '202511012359'    # 종료 (데이터 범위에 맞게 수정)

start_dt = pd.to_datetime(START_DATE, format='%Y%m%d%H%M')
end_dt = pd.to_datetime(END_DATE, format='%Y%m%d%H%M')

# 전체 시간 범위 (1분 단위)
full_range = pd.date_range(start=start_dt, end=end_dt, freq='1min')
all_cols = ['AHSO' + t.strftime('%Y%m%d%H%M') for t in full_range]

# 기존 AHSO 컬럼
existing_ahso = [col for col in df.columns if col.startswith('AHSO')]
print(f"기존 AHSO 컬럼: {len(existing_ahso)}개")

# 빠진 컬럼
missing_cols = [col for col in all_cols if col not in df.columns]
print(f"빠진 AHSO 컬럼: {len(missing_cols)}개")

# 빠진 컬럼 추가 (0으로)
for col in missing_cols:
    df[col] = 0

# 고정 컬럼 + AHSO 컬럼 정렬
fixed_cols = [col for col in df.columns if not col.startswith('AHSO')]
ahso_cols = sorted([col for col in df.columns if col.startswith('AHSO')])
df = df[fixed_cols + ahso_cols]

print(f"\n최종 컬럼: {len(df.columns)}개")

# 저장
df.to_csv('ahso_alarm_full_range.csv', index=False, encoding='utf-8-sig')
print("저장 완료!")