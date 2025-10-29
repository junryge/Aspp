import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# CSV 파일 읽기 (파일명을 실제 파일명으로 변경하세요)
input_file = 'input.csv'  # 입력 파일명
output_file = 'output_filled.csv'  # 출력 파일명

# CSV 파일 읽기
df = pd.read_csv(input_file)

# datetime 컬럼명 (실제 컬럼명으로 변경하세요)
datetime_column = 'datetime'  # 예: 'timestamp', 'date', 'datetime' 등

# datetime 컬럼을 datetime 타입으로 변환
df[datetime_column] = pd.to_datetime(df[datetime_column])

# 전체 날짜 범위 생성 (2025-09-05 00:00 ~ 2025-10-25 23:59, 1분 간격)
start_date = datetime(2025, 9, 5, 0, 0)
end_date = datetime(2025, 10, 25, 23, 59)

# 1분 간격으로 전체 datetime 범위 생성
full_datetime_range = pd.date_range(start=start_date, end=end_date, freq='min')

# 전체 datetime 범위로 DataFrame 생성
full_df = pd.DataFrame({datetime_column: full_datetime_range})

# 원본 데이터와 병합 (left join)
result_df = full_df.merge(df, on=datetime_column, how='left')

# 누락된 데이터는 이미 NaN(NULL)으로 처리됨

# CSV로 저장
result_df.to_csv(output_file, index=False)

print(f"완료! 원본 데이터: {len(df)}행")
print(f"전체 datetime 범위: {len(full_df)}행")
print(f"결과 파일: {len(result_df)}행")
print(f"누락된 데이터 (NULL): {len(result_df) - len(df)}행")
print(f"\n결과 파일: {output_file}")