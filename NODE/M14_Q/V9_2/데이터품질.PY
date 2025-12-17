#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🛠️ 학습용 고부하(Over 1700) 데이터 추출기
중요: 평가하려는 날짜(12월 11일)는 절대 포함하지 않습니다. (데이터 누수 방지)
"""
import pandas as pd
import glob
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🛡️ 학습용 고부하 데이터 추출 (평가일 제외)")
print("="*70)

# 🔴 평가할 날짜 (이 날짜는 학습 데이터에서 무조건 뺍니다)
TEST_DATE = '2025-12-11'
PRED_OFFSET = 10

def load_data(path_pattern):
    files = glob.glob(path_pattern)
    if not files: files = glob.glob('M14_Q_*.CSV')
    dfs = []
    for f in sorted(files):
        try: dfs.append(pd.read_csv(f, on_bad_lines='skip', encoding='utf-8'))
        except: dfs.append(pd.read_csv(f, on_bad_lines='skip', encoding='cp949'))
    return pd.concat(dfs, ignore_index=True)

# 1. 전체 데이터 로드
df = load_data('M14_Q_*.CSV')
df['CURRTIME_OBJ'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M', errors='coerce')

# 2. 피처 생성 (학습 때 쓰는 것과 동일하게 280 시퀀스 등)
# (간단하게 타겟만 확인하기 위해 로직 최소화, 실제론 학습 코드의 피처 생성 함수 쓰면 됨)
df['Target_Total_Future'] = df['TOTALCNT'].shift(-PRED_OFFSET)

# 3. [중요] 평가일(12월 11일) 데이터 제거
mask_not_test = df['CURRTIME_OBJ'].dt.date.astype(str) != TEST_DATE
df_train_pool = df[mask_not_test].copy()

print(f"\n📉 데이터 필터링:")
print(f" - 전체 데이터: {len(df):,}개")
print(f" - 평가일({TEST_DATE}) 제외 후: {len(df_train_pool):,}개")

# 4. 1700 이상인 데이터만 추출
df_high_load = df_train_pool[df_train_pool['Target_Total_Future'] >= 1700].copy()

print(f"\n🔥 추출된 학습용 고부하 데이터:")
print(f" - 개수: {len(df_high_load):,}개")

if len(df_high_load) > 0:
    # 저장
    filename = 'High_Load_Train_Only.csv'
    df_high_load.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"\n💾 저장 완료: {filename}")
    print("👉 학습 코드에서 이 파일을 불러와서 50~100배 복사(pd.concat)해서 쓰세요.")
else:
    print("\n⚠ 12월 11일을 뺐더니 1700 넘는 게 하나도 없습니다. 데이터 기간을 확인하세요.")
