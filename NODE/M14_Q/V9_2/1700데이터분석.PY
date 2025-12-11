#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔬 10분 후 예측 정확도 검증
- 현재 gap, M14B, SUM으로 → 10분 후 TOTALCNT 예측
"""

import pandas as pd
import numpy as np

CSV_FILE = "YOUR_CSV_FILE.csv"  # ← 실제 파일 경로!

print("="*70)
print("🔮 10분 후 예측 검증")
print("="*70)

df = pd.read_csv(CSV_FILE, on_bad_lines='skip')
print(f"총 {len(df):,}행")

# 컬럼 지정
totalcnt_col = 'TOTALCNT'
m14b_col = 'M14AM14B'
m14bsum_col = 'M14AM14BSUM'
qcreated_col = 'M14.QUE.ALL.CURRENTQCREATED'
qcompleted_col = 'M14.QUE.ALL.CURRENTQCOMPLETED'

# queue_gap 계산
df['queue_gap'] = df[qcreated_col] - df[qcompleted_col]

# ============================================
# 10분 후 TOTALCNT (시프트!)
# ============================================
PRED_MINUTES = 10
df['future_totalcnt'] = df[totalcnt_col].shift(-PRED_MINUTES)
df['future_1700'] = df['future_totalcnt'] >= 1700

# NaN 제거 (마지막 10행)
df_valid = df.dropna(subset=['future_totalcnt']).copy()
print(f"유효 데이터: {len(df_valid):,}행 (마지막 {PRED_MINUTES}행 제외)")

# ============================================
# 현재 조건들
# ============================================
df_valid['gap_over_300'] = df_valid['queue_gap'] > 300
df_valid['gap_over_400'] = df_valid['queue_gap'] > 400
df_valid['gap_over_500'] = df_valid['queue_gap'] > 500
df_valid['gold_pattern'] = (df_valid[m14b_col] > 520) & (df_valid[m14bsum_col] > 600)
df_valid['triple_check'] = (df_valid[m14b_col] > 520) & (df_valid[m14bsum_col] > 600) & (df_valid['queue_gap'] > 250)

# ============================================
# 기본 통계
# ============================================
total = len(df_valid)
actual_1700 = df_valid['future_1700'].sum()

print(f"\n📊 기본 통계:")
print(f"  전체: {total:,}개")
print(f"  10분 후 1700+ 실제 발생: {actual_1700:,}개 ({actual_1700/total*100:.2f}%)")

# ============================================
# 예측 정확도 검증
# ============================================
print("\n" + "="*70)
print("🎯 10분 후 예측 정확도")
print("="*70)

def calc_metrics(condition, actual, name):
    """정밀도, 재현율, 감지 개수 계산"""
    pred_positive = condition.sum()
    true_positive = (condition & actual).sum()
    actual_positive = actual.sum()
    
    precision = true_positive / pred_positive * 100 if pred_positive > 0 else 0
    recall = true_positive / actual_positive * 100 if actual_positive > 0 else 0
    
    print(f"\n🔥 {name}")
    print(f"  조건 충족: {pred_positive:,}개")
    print(f"  실제 1700+ 감지: {true_positive:,}개 / {actual_positive:,}개")
    print(f"  정밀도(Precision): {precision:.1f}% (경보 중 실제 위험)")
    print(f"  재현율(Recall): {recall:.1f}% (실제 위험 중 감지)")
    
    return pred_positive, true_positive, precision, recall

# 각 조건별 테스트
print("\n" + "-"*50)
print("📊 queue_gap 단독")
print("-"*50)

for gap_th in [200, 250, 300, 350, 400, 500]:
    cond = df_valid['queue_gap'] > gap_th
    calc_metrics(cond, df_valid['future_1700'], f"gap > {gap_th}")

print("\n" + "-"*50)
print("📊 황금패턴")
print("-"*50)

calc_metrics(df_valid['gold_pattern'], df_valid['future_1700'], "M14B>520 & SUM>600")

print("\n" + "-"*50)
print("📊 복합 조건 (Triple Check)")
print("-"*50)

calc_metrics(df_valid['triple_check'], df_valid['future_1700'], "M14B>520 & SUM>600 & gap>250")

# 추가 복합 조건
for gap_th in [200, 300, 400]:
    cond = (df_valid[m14b_col] > 520) & (df_valid[m14bsum_col] > 600) & (df_valid['queue_gap'] > gap_th)
    calc_metrics(cond, df_valid['future_1700'], f"황금패턴 & gap>{gap_th}")

# ============================================
# 미감지 분석
# ============================================
print("\n" + "="*70)
print("⚠️ 미감지 분석")
print("="*70)

# Triple Check로 미감지된 케이스
missed = df_valid['future_1700'] & ~df_valid['triple_check']
print(f"\nTriple Check 미감지: {missed.sum():,}개 / {actual_1700:,}개")

# gap > 400 미감지
missed_gap400 = df_valid['future_1700'] & ~(df_valid['queue_gap'] > 400)
print(f"gap > 400 미감지: {missed_gap400.sum():,}개 / {actual_1700:,}개")

# 최적 조건 조합
print("\n" + "="*70)
print("🏆 최적 조건 탐색")
print("="*70)

best_f1 = 0
best_cond = ""

conditions = [
    ("gap>300", df_valid['queue_gap'] > 300),
    ("gap>350", df_valid['queue_gap'] > 350),
    ("gap>400", df_valid['queue_gap'] > 400),
    ("gap>500", df_valid['queue_gap'] > 500),
    ("Triple Check", df_valid['triple_check']),
    ("황금패턴 & gap>300", (df_valid[m14b_col] > 520) & (df_valid[m14bsum_col] > 600) & (df_valid['queue_gap'] > 300)),
    ("gap>300 OR 황금패턴", (df_valid['queue_gap'] > 300) | df_valid['gold_pattern']),
    ("gap>250 OR 황금패턴", (df_valid['queue_gap'] > 250) | df_valid['gold_pattern']),
]

print(f"\n{'조건':<30} {'정밀도':>8} {'재현율':>8} {'F1':>8} {'감지수':>8}")
print("-"*70)

for name, cond in conditions:
    pred_pos = cond.sum()
    true_pos = (cond & df_valid['future_1700']).sum()
    actual_pos = df_valid['future_1700'].sum()
    
    prec = true_pos / pred_pos * 100 if pred_pos > 0 else 0
    rec = true_pos / actual_pos * 100 if actual_pos > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    
    print(f"{name:<30} {prec:>7.1f}% {rec:>7.1f}% {f1:>7.1f} {true_pos:>7,}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_cond = name

print(f"\n🏆 최적 조건: {best_cond} (F1: {best_f1:.1f})")

print("\n✅ 10분 후 예측 검증 완료!")