import pandas as pd

# 1. 데이터 불러오기
df = pd.read_csv('M14_Q_20241201_20251201610.CSV')

# 2. 구간 분류 함수 정의
def get_load_category(cnt):
    if cnt <= 1200:
        return 'Under_1200'      # 1200 이하
    elif 1200 < cnt <= 1300:
        return '1200_1300'       # 1200 ~ 1300
    elif 1300 < cnt <= 1400:
        return '1300_1400'       # 1300 ~ 1400
    elif 1400 < cnt <= 1500:
        return '1400_1500'       # 1400 ~ 1500
    elif 1500 < cnt <= 1600:
        return '1500_1600'       # 1500 ~ 1600
    elif 1600 < cnt < 1700:
        return '1600_1700'       # 1600 ~ 1700 (누락 구간 보완)
    else: # cnt >= 1700
        return 'Over_1700'       # 1700 이상

# 3. 데이터에 구간 정보('Load_Category') 추가
df['Load_Category'] = df['TOTALCNT'].apply(get_load_category)

# 4. 구간별 데이터 개수 확인
print(df['Load_Category'].value_counts())

# 5. 하나의 통합 파일로 저장
df.to_csv('M14_Load_Categorized.csv', index=False)
print("통합 파일 저장 완료: M14_Load_Categorized.csv")
