import pandas as pd

# 한글 CSV 안전하게 읽기
try:
    # 먼저 utf-8-sig 시도 (BOM 처리)
    df = pd.read_csv('RESULT_2025_with_queue_analysis.csv', encoding='utf-8-sig')
except:
    try:
        # cp949 시도 (Windows 한글)
        df = pd.read_csv('RESULT_2025_with_queue_analysis.csv', encoding='cp949')
    except:
        # euc-kr 시도
        df = pd.read_csv('RESULT_2025_with_queue_analysis.csv', encoding='euc-kr')

# 필터링
filtered = df[(df['현재값구간'] == '200-250') & 
              (df['변화구간_5분'] == '0~20')]

상승 = filtered[filtered['변화방향_5분'] == '상승']
하락 = filtered[filtered['변화방향_5분'] == '하락']

print(f"200-250 구간, 5분 변화 0~20:")
print(f"- 전체: {len(filtered)}개")
print(f"- 상승: {len(상승)}개")
print(f"- 하락: {len(하락)}개")