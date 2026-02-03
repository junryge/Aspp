import pandas as pd
import re

# CSV 로드
df = pd.read_csv('ahso_alarm.csv', encoding='utf-8-sig')

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

# 원본 데이터와 병합 (없는 시간대는 NaN)
df = full_df.merge(df, on='DATETIME', how='left')

# 빈 데이터 채우기
# 분류 컬럼
if '분류' in df.columns:
    df['분류'] = df['분류'].fillna('')

# AHTO+AHSO, REWORK 숫자 컬럼 0으로
numeric_cols = ['AHTO+AHSO', 'REWORK']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

# AHSO 컬럼 - 날짜 기반으로 재생성
df[ahso_col] = 'AHSO' + df['DATETIME'].dt.strftime('%Y%m%d%H%M')

# 정렬
df = df.sort_values('DATETIME').reset_index(drop=True)

# 결과 확인
print(f"=== 기간: {start_dt} ~ {end_dt} ===")
print(f"총 {len(df)}건 (1분 단위)")
print(f"\n=== 결측치 현황 ===")
print(df.isnull().sum())
print(f"\n=== 샘플 데이터 ===")
print(df.head(20))

# 저장
df.to_csv('ahso_alarm_full_range.csv', index=False, encoding='utf-8-sig')
print("\n저장 완료: ahso_alarm_full_range.csv")