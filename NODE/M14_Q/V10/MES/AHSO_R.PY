import pandas as pd
import re

# CSV 로드
df = pd.read_csv('ahso_alarm.csv', encoding='utf-8-sig')

# AHSO 컬럼 문자열 변환 (오류 해결)
ahso_col = [col for col in df.columns if 'AHSO' in col][0]
df[ahso_col] = df[ahso_col].astype(str)

# 날짜 추출
df['DATETIME'] = pd.to_datetime(
    df[ahso_col].str.extract(r'(\d{12})')[0],
    format='%Y%m%d%H%M',
    errors='coerce'
)

# 빈 데이터(결측치) 채우기
# 숫자 컬럼 0으로 채우기
numeric_cols = ['AHTO+AHSO', 'REWORK']  # 실제 컬럼명에 맞게 수정
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

# 문자 컬럼 빈 문자열로 채우기
df['분류'] = df['분류'].fillna('')

# 날짜 결측치 확인
print("=== 결측치 현황 ===")
print(df.isnull().sum())

# 저장
df.to_csv('ahso_alarm_with_datetime.csv', index=False, encoding='utf-8-sig')
print(df.head(10))