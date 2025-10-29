import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# CSV 파일 읽기
input_file = 'input.csv'  # 입력 파일명 변경하세요
output_file = 'output_filled.csv'  # 출력 파일명

# CSV 파일 읽기 (한글 인코딩 처리)
try:
    df = pd.read_csv(input_file, encoding='cp949')
except:
    try:
        df = pd.read_csv(input_file, encoding='euc-kr')
    except:
        df = pd.read_csv(input_file, encoding='utf-8')

# CRT_TM 컬럼을 datetime 타입으로 변환
df['CRT_TM'] = pd.to_datetime(df['CRT_TM'])

# 전체 날짜 범위 생성 (2025-09-05 00:00 ~ 2025-10-25 23:59, 1분 간격)
start_date = datetime(2025, 9, 5, 0, 0)
end_date = datetime(2025, 10, 25, 23, 59)

# 1분 간격으로 전체 datetime 범위 생성
full_datetime_range = pd.date_range(start=start_date, end=end_date, freq='min')

# 전체 datetime 범위로 DataFrame 생성
full_df = pd.DataFrame({'CRT_TM': full_datetime_range})

# 원본 데이터와 병합 (left join)
# CRT_TM 기준으로 병합하면 누락된 시간대는 자동으로 NaN(NULL) 처리됨
result_df = full_df.merge(df, on='CRT_TM', how='left')

# IDC_VAL이 NULL인 행 확인 (누락된 데이터)
null_count = result_df['IDC_VAL'].isna().sum()

# CSV로 저장 (NULL은 빈 칸으로 저장됨)
result_df.to_csv(output_file, index=False)

print(f"완료!")
print(f"원본 데이터: {len(df)}행")
print(f"전체 datetime 범위: {len(full_df)}행 (2025-09-05 00:00 ~ 2025-10-25 23:59, 1분 간격)")
print(f"결과 파일: {len(result_df)}행")
print(f"누락된 데이터 (IDC_VAL NULL): {null_count}행")
print(f"\n결과 파일 저장: {output_file}")



import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# CSV 파일 읽기
input_file = 'input.csv'  # 입력 파일명 변경하세요
output_file = 'output_filled.csv'  # 출력 파일명

# CSV 파일 읽기 (한글 인코딩 처리)
try:
    df = pd.read_csv(input_file, encoding='cp949')
except:
    try:
        df = pd.read_csv(input_file, encoding='euc-kr')
    except:
        df = pd.read_csv(input_file, encoding='utf-8')

print(f"원본 데이터: {len(df)}행")
print(f"컬럼: {list(df.columns)}")

# STAT_DT를 datetime으로 변환 (YYYYMMDDHHMM 형태)
df['STAT_DT_DATETIME'] = pd.to_datetime(df['STAT_DT'], format='%Y%m%d%H%M')

# 전체 날짜 범위 생성 (2025-09-05 00:00 ~ 2025-10-25 23:59, 1분 간격)
start_date = datetime(2025, 9, 5, 0, 0)
end_date = datetime(2025, 10, 25, 23, 59)

# 1분 간격으로 전체 datetime 범위 생성
full_datetime_range = pd.date_range(start=start_date, end=end_date, freq='min')

# STAT_DT 형식으로 변환 (YYYYMMDDHHMM)
full_df = pd.DataFrame({
    'STAT_DT_DATETIME': full_datetime_range,
    'STAT_DT': full_datetime_range.strftime('%Y%m%d%H%M')
})

# 원본 데이터와 병합 (left join)
result_df = full_df.merge(df, on='STAT_DT', how='left', suffixes=('', '_old'))

# 불필요한 컬럼 제거
if 'STAT_DT_DATETIME_old' in result_df.columns:
    result_df = result_df.drop(columns=['STAT_DT_DATETIME_old'])

# STAT_DT_DATETIME 컬럼 제거 (최종 결과에서)
result_df = result_df.drop(columns=['STAT_DT_DATETIME'])

# 누락된 데이터 확인
null_count = result_df.isnull().any(axis=1).sum()

# CSV로 저장 (NULL은 빈 칸으로 저장됨)
result_df.to_csv(output_file, index=False, encoding='cp949')

print(f"\n완료!")
print(f"전체 datetime 범위: {len(full_df)}행 (2025-09-05 00:00 ~ 2025-10-25 23:59, 1분 간격)")
print(f"결과 파일: {len(result_df)}행")
print(f"누락된 데이터 (NULL 포함): {null_count}행")
print(f"\n결과 파일 저장: {output_file}")