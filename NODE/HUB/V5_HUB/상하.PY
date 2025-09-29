import pandas as pd

# CSV 읽기
df = pd.read_csv('RESULT_2025_with_queue_analysis.csv')

# 200-250 구간 & 5분 변화량 0~20 필터
filtered = df[(df['현재값구간'] == '200-250') & 
              (df['변화구간_5분'] == '0~20')]

# 상승/하락 분리
상승 = filtered[filtered['변화방향_5분'] == '상승']
하락 = filtered[filtered['변화방향_5분'] == '하락']

print(f"200-250 구간, 5분 변화 0~20:")
print(f"- 전체: {len(filtered)}개")
print(f"- 상승: {len(상승)}개 ({len(상승)/len(filtered)*100:.1f}%)")
print(f"- 하락: {len(하락)}개 ({len(하락)/len(filtered)*100:.1f}%)")

# 결과를 새 CSV로 저장
filtered.to_csv('filtered_200_250_변화0_20.csv', index=False)