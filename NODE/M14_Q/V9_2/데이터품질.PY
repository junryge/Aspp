import pandas as pd
import numpy as np
import glob
import warnings
warnings.filterwarnings('ignore')

print("🚀 데이터 비율 조정 및 CSV 저장 시작...")

# 1. 원본 데이터 로드
files = glob.glob('M14_Q_*.CSV')
if not files:
    print("❌ M14_Q_*.CSV 파일이 없습니다.")
    exit()

dfs = []
for f in sorted(files):
    try: dfs.append(pd.read_csv(f, on_bad_lines='skip', encoding='utf-8'))
    except: dfs.append(pd.read_csv(f, on_bad_lines='skip', encoding='cp949'))
df = pd.concat(dfs, ignore_index=True)

# 2. 전처리 (숫자 변환)
cols = ['TOTALCNT', 'M14AM14B', 'M14AM14BSUM', 'M10AM14A', 
        'M14.QUE.ALL.TRANSPORT4MINOVERCNT', 'M14.QUE.OHT.OHTUTIL',
        'M14.QUE.ALL.CURRENTQCREATED', 'M14.QUE.ALL.CURRENTQCOMPLETED']
for c in cols:
    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

# 3. 타겟 생성 (1700 이상인지 판단하기 위해 필수)
# 10분 뒤 예측이므로 -10 shift
df['Target_Total_Future'] = df['TOTALCNT'].shift(-10)

# 4. 12월 11일(평가용) 데이터 제거
df['datetime'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M', errors='coerce')
df_train = df[df['datetime'].dt.date.astype(str) != '2025-12-11'].copy()
df_train = df_train.dropna(subset=['Target_Total_Future']).reset_index(drop=True)

# 5. 비율 조정 (핵심!)
# 1700 이상(High) vs 미만(Normal)
df_high = df_train[df_train['Target_Total_Future'] >= 1700].copy()
df_normal = df_train[df_train['Target_Total_Future'] < 1700].copy()

count_high = len(df_high)
count_normal = len(df_normal)

print(f" - 정상 데이터: {count_normal:,}개")
print(f" - 1700+ 데이터: {count_high:,}개")

if count_high > 0:
    # 비율 맞추기 (정상 데이터 개수만큼 1700 데이터 복사)
    multiplier = int(count_normal / count_high)
    print(f"🔥 1700+ 데이터를 {multiplier}배로 뻥튀기합니다.")
    
    df_high_augmented = pd.concat([df_high] * multiplier, ignore_index=True)
    
    # 합치기
    df_balanced = pd.concat([df_normal, df_high_augmented], ignore_index=True)
    
    # 순서 섞기
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 6. CSV 저장
    filename = 'Balanced_Data.csv'
    df_balanced.to_csv(filename, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ 저장 완료: {filename}")
    print(f"   총 데이터: {len(df_balanced):,}개 (1700+ 비율 약 50% 맞춤)")
else:
    print("❌ 1700 이상 데이터가 없어서 증폭할 수 없습니다.")