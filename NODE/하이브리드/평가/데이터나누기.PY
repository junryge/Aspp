import pandas as pd
from datetime import datetime

# 데이터 읽기 (탭으로 구분된 것으로 보임)
# 파일명을 실제 파일 경로로 변경하세요
df = pd.read_csv('paste.txt', sep='\t')

# start_time을 datetime 형식으로 변환
df['start_time'] = pd.to_datetime(df['start_time'])

# 분(minute)이 0, 10, 20, 30, 40, 50인 데이터만 필터링
df_filtered = df[df['start_time'].dt.minute % 10 == 0]

# 또는 더 명시적으로 특정 분만 선택하려면:
# valid_minutes = [0, 10, 20, 30, 40, 50]
# df_filtered = df[df['start_time'].dt.minute.isin(valid_minutes)]

# 결과 출력
print("원본 데이터 개수:", len(df))
print("필터링된 데이터 개수:", len(df_filtered))
print("\n필터링된 데이터 미리보기:")
print(df_filtered.head(10))

# 필터링된 데이터를 새 파일로 저장
df_filtered.to_csv('filtered_10min_data.csv', index=False)
print("\n결과가 'filtered_10min_data.csv' 파일로 저장되었습니다.")

# start_time의 분 확인 (검증용)
print("\n필터링된 데이터의 시간 패턴:")
print(df_filtered['start_time'].dt.strftime('%H:%M').value_counts().sort_index().head(20))