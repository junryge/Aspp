#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 현재 큐개수별 10분 후 증감량 분석 (실제값, 한 시트)
"""

import pandas as pd
import numpy as np

# 데이터 로드
DATA_PATH = 'your_data.csv'  # ← 실제 파일 경로!
df = pd.read_csv(DATA_PATH)

print("="*70)
print("📊 현재 큐개수별 10분 후 증감량 분석")
print("="*70)

# 10분 후 값과 변화량 계산
df['10분후_TOTALCNT'] = df['TOTALCNT'].shift(-10)
df['10분후_증감'] = df['10분후_TOTALCNT'] - df['TOTALCNT']

# 현재큐 구간
df['구간'] = pd.cut(df['TOTALCNT'], 
                    bins=[0, 1200, 1300, 1400, 1500, 1600, 1700, 1800, np.inf],
                    labels=['<1200', '1200~1299', '1300~1399', '1400~1499', 
                            '1500~1599', '1600~1699', '1700~1799', '>=1800'])

# 결과 정리 (실제값)
result = df[['CURRTIME', 'TOTALCNT', '10분후_TOTALCNT', '10분후_증감', '구간']].copy()
result.columns = ['시간', '현재큐', '10분후큐', '10분후증감', '구간']
result = result.dropna()

# 저장
result.to_csv('queue_10min_change.csv', index=False, encoding='utf-8-sig')
print(f"✅ 총 {len(result):,}건")
print(f"💾 queue_10min_change.csv 저장 완료!")

print("\n샘플 데이터:")
print(result.head(20).to_string(index=False))