import pandas as pd

# 1. 데이터 불러오기
df = pd.read_csv('M14_Q_20241201_20251201610.CSV')

# 2. 구간 분류 (이전과 동일)
def get_load_category(cnt):
    if cnt <= 1200: return 'Under_1200'
    elif 1200 < cnt <= 1300: return '1200_1300'
    elif 1300 < cnt <= 1400: return '1300_1400'
    elif 1400 < cnt <= 1500: return '1400_1500'
    elif 1500 < cnt <= 1600: return '1500_1600'
    elif 1600 < cnt < 1700: return '1600_1700'
    else: return 'Over_1700'

df['Load_Category'] = df['TOTALCNT'].apply(get_load_category)

# 3. 데이터 밸런싱 (Under-sampling)
# 가장 데이터가 적은 구간의 개수를 찾습니다. (Over_1700: 6517개)
min_count = df['Load_Category'].value_counts().min()

# 각 그룹별로 min_count만큼만 랜덤하게 뽑습니다.
balanced_df = df.groupby('Load_Category', group_keys=False).apply(
    lambda x: x.sample(min_count, random_state=42)
)

# 데이터 순서를 섞어줍니다.
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# 4. 저장
balanced_df.to_csv('M14_Balanced_Learning_Data.csv', index=False)
print(f"학습용 균형 데이터 저장 완료: {len(balanced_df)}건")
