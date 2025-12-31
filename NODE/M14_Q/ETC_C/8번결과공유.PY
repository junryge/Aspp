#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
8단계 결과 확인 코드
"""

import pandas as pd

# 결과 파일 로드
df = pd.read_csv('step8_features_10min.csv', nrows=100)  # 100행만 먼저

print("="*80)
print("📊 8단계 Feature Engineering 결과")
print("="*80)

# 1. 기본 정보
print(f"\n1️⃣ 기본 정보")
print(f"   총 Feature 수: {len(df.columns)}개")
print(f"   샘플 수 (미리보기): {len(df)}개")

# 2. Feature 목록
print(f"\n2️⃣ Feature 목록 (전체 {len(df.columns)}개)")
for i, col in enumerate(df.columns, 1):
    print(f"   {i:3d}. {col}")

# 3. 전체 파일 통계 (따로)
print("\n" + "="*80)
print("📈 전체 파일 통계 (시간 걸릴 수 있음)")
print("="*80)

try:
    # 전체 파일 행 수 세기
    total_rows = sum(1 for _ in open('step8_features_10min.csv')) - 1  # 헤더 제외
    print(f"   총 샘플 수: {total_rows:,}개")
    
    # 타겟 컬럼만 로드해서 통계
    df_target = pd.read_csv('step8_features_10min.csv', usecols=['target_TOTALCNT', 'is_danger'])
    print(f"   1700+ 샘플 수: {df_target['is_danger'].sum():,}개 ({df_target['is_danger'].mean()*100:.2f}%)")
    print(f"   타겟 평균: {df_target['target_TOTALCNT'].mean():.2f}")
    print(f"   타겟 최대: {df_target['target_TOTALCNT'].max():.2f}")
    print(f"   타겟 최소: {df_target['target_TOTALCNT'].min():.2f}")
except Exception as e:
    print(f"   ⚠️ 전체 통계 오류: {e}")

print("\n✅ 확인 완료!")