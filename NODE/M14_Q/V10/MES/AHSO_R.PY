import pandas as pd
import re

# CSV 로드 (실제 파일 경로로 변경)
df = pd.read_csv('ahso_alarm.csv', encoding='utf-8-sig')

# 원본 컬럼 구조: 분류, AHTO+AHSO, REWORK, AHSO(알람ID)
# AHSO 컬럼에서 날짜 추출해서 DATETIME 컬럼만 추가

# AHSO 컬럼명 찾기 (AHSO로 시작하는 컬럼 또는 알람ID 컬럼)
ahso_col = [col for col in df.columns if 'AHSO' in col or '알람' in col.lower()][0]

# 날짜 추출 (AHSO202511010006 → 2025-11-01 00:06)
df['DATETIME'] = pd.to_datetime(
    df[ahso_col].str.extract(r'(\d{12})')[0],
    format='%Y%m%d%H%M',
    errors='coerce'
)

# 컬럼 순서: 분류, AHTO+AHSO, REWORK, AHSO, DATETIME (맨 뒤에 추가)
# 기존 컬럼 유지 + DATETIME만 추가

# 결측치 확인
print("=== 결측치 현황 ===")
print(df.isnull().sum())

# 저장
df.to_csv('ahso_alarm_with_datetime.csv', index=False, encoding='utf-8-sig')
print(df.head())