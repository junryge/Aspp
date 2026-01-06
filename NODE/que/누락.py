# -*- coding: utf-8 -*-
"""
V13 데이터 병합 - 신규 컬럼 추가
"""
import pandas as pd
import numpy as np

# ============================================================
# 설정
# ============================================================
TRAIN_FILE = 'M14_학습_20250909_20251231_C_109.CSV'  # 기존 학습 데이터
PDT_FILE = 'V13_PDT_20250909_20251231.csv'           # 신규 PDT 데이터
OUTPUT_FILE = 'M14_학습_V13_20250909_20251231.CSV'   # 병합 결과

# ============================================================
# 데이터 로드
# ============================================================
print("=" * 60)
print("V13 데이터 병합")
print("=" * 60)

# 기존 학습 데이터
print("\n[1/4] 기존 학습 데이터 로드...")
try:
    df_train = pd.read_csv(TRAIN_FILE, encoding='utf-8')
except:
    try:
        df_train = pd.read_csv(TRAIN_FILE, encoding='cp949')
    except:
        df_train = pd.read_csv(TRAIN_FILE, encoding='euc-kr')

print(f"  - 기존 데이터: {len(df_train):,}행, {len(df_train.columns)}컬럼")

# 신규 PDT 데이터
print("\n[2/4] 신규 PDT 데이터 로드...")
try:
    df_pdt = pd.read_csv(PDT_FILE, encoding='utf-8')
except:
    try:
        df_pdt = pd.read_csv(PDT_FILE, encoding='cp949')
    except:
        df_pdt = pd.read_csv(PDT_FILE, encoding='euc-kr')

print(f"  - PDT 데이터: {len(df_pdt):,}행, {len(df_pdt.columns)}컬럼")

# ============================================================
# CURRTIME 형식 맞추기
# ============================================================
print("\n[3/4] CURRTIME 형식 맞추기...")

# 기존 데이터 CURRTIME 형식 확인 및 변환
df_train['CURRTIME'] = df_train['CURRTIME'].astype(str).str.replace('-', '').str.replace(':', '').str.replace(' ', '').str[:12]
df_pdt['CURRTIME'] = df_pdt['CURRTIME'].astype(str).str.replace('-', '').str.replace(':', '').str.replace(' ', '').str[:12]

print(f"  - 기존 CURRTIME 샘플: {df_train['CURRTIME'].iloc[0]}")
print(f"  - PDT CURRTIME 샘플: {df_pdt['CURRTIME'].iloc[0]}")

# 중복 제거 (PDT)
df_pdt = df_pdt.drop_duplicates(subset='CURRTIME', keep='last')
print(f"  - PDT 중복 제거 후: {len(df_pdt):,}행")

# ============================================================
# 병합
# ============================================================
print("\n[4/4] 데이터 병합...")

df_merged = df_train.merge(df_pdt, on='CURRTIME', how='left')

# 누락값 0으로 채우기
new_cols = [c for c in df_pdt.columns if c != 'CURRTIME']
for col in new_cols:
    df_merged[col] = df_merged[col].fillna(0)
    
print(f"  - 병합 결과: {len(df_merged):,}행, {len(df_merged.columns)}컬럼")
print(f"  - 신규 컬럼: {new_cols}")

# 누락 확인
for col in new_cols:
    null_cnt = (df_merged[col] == 0).sum()
    print(f"  - {col}: 0값 {null_cnt:,}개 ({100*null_cnt/len(df_merged):.1f}%)")

# ============================================================
# 저장
# ============================================================
df_merged.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
print(f"\n저장 완료: {OUTPUT_FILE}")
print("=" * 60)