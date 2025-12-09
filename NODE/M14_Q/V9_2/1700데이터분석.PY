#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔬 AMHS 심층 분석 - Leakage 제거 + 1700+ 집중 분석
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🔬 AMHS 심층 분석 - Leakage 제거 + 1700+ 집중 분석")
print("="*80)

# ========================================
# 📌 파일 경로 수정!
# ========================================
DATA_FILE = "data/M14_Q_3MONTH.csv"  # 여기 수정!

# 데이터 로드
df = pd.read_csv(DATA_FILE, on_bad_lines='skip')
print(f"✅ 데이터: {len(df):,}행 x {df.shape[1]}열")

# queue_gap 생성
if 'M14.QUE.ALL.CURRENTQCREATED' in df.columns and 'M14.QUE.ALL.CURRENTQCOMPLETED' in df.columns:
    df['queue_gap'] = df['M14.QUE.ALL.CURRENTQCREATED'] - df['M14.QUE.ALL.CURRENTQCOMPLETED']

# ============================================================
# 1. 기본 통계
# ============================================================
print("\n" + "="*60)
print("📊 1. 기본 통계")
print("="*60)

target = 'TOTALCNT'
print(f"TOTALCNT 평균: {df[target].mean():.2f}")
print(f"TOTALCNT 중앙값: {df[target].median():.2f}")
print(f"TOTALCNT 범위: {df[target].min()} ~ {df[target].max()}")

print(f"\n📌 위험 구간 분포:")
for t in [1500, 1600, 1650, 1700, 1750, 1800, 1900, 2000]:
    cnt = (df[target] >= t).sum()
    print(f"  {t}+ : {cnt:,}개 ({cnt/len(df)*100:.3f}%)")

# ============================================================
# 2. Leakage 변수 확인
# ============================================================
print("\n" + "="*60)
print("⚠️ 2. Leakage 의심 변수 확인")
print("="*60)

leakage_cols = ['M14.QUE.ALL.CURRENTQCNT', 'M14B.QUE.ALL.CURRENTQCNT']
for col in leakage_cols:
    if col in df.columns:
        corr = df[col].corr(df[target])
        print(f"  {col} ↔ TOTALCNT: {corr:.4f}")

# ============================================================
# 3. Leakage 제거 후 Feature Importance
# ============================================================
print("\n" + "="*60)
print("🎯 3. Leakage 제거 후 Feature Importance")
print("="*60)

try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score
    
    # 제거할 컬럼
    remove_cols = [
        'CURRTIME', 'TOTALCNT',
        # Leakage 의심
        'M14.QUE.ALL.CURRENTQCNT',
        'M14B.QUE.ALL.CURRENTQCNT',
        # 결측치 100%
        'M14.QUE.SFAB.SENDTOR3', 'M14.QUE.SFAB.SENDTOM16', 'M14.QUE.SFAB.SENDTOM10',
        'M14.QUE.SFAB.SENDQUEUETOTAL', 'M14.QUE.SFAB.RECEIVETOR3', 'M14.QUE.SFAB.RECEIVETOM16',
        'M14.QUE.SFAB.RECEIVETOM10', 'M14.QUE.SFAB.RECEIVEQUEUETOTAL',
        'M14.QUE.SFAB.RETURNTOR3', 'M14.QUE.SFAB.RETURNTOM16', 'M14.QUE.SFAB.RETURNTOM10',
        'M14.QUE.SFAB.RETURNQUEUETOTAL', 'M14.QUE.SFAB.COMPLETETOR3', 'M14.QUE.SFAB.COMPLETETOM16',
        'M14.QUE.SFAB.COMPLETETOM10', 'M14.QUE.SFAB.COMPLETEQUEUETOTAL',
        'M14B.QUE.ABN.QUETIMEDELAY',
        'M16.QUE.SFAB.SENDTOM14', 'M16.QUE.SFAB.RECEIVETOM14',
        'M16.QUE.SFAB.RETURNTOM14', 'M16.QUE.SFAB.COMPLETETOM14',
    ]
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in remove_cols]
    
    print(f"✅ 사용 Feature: {len(feature_cols)}개")
    
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor(max_depth=6, learning_rate=0.1, n_estimators=100, 
                              random_state=42, n_jobs=-1)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\n📌 모델 성능: MAE={mae:.2f}, R²={r2:.4f}")
    
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n📌 Feature Importance TOP 30 (Leakage 제거 후):")
    for i, row in importance.head(30).iterrows():
        print(f"  {importance.index.get_loc(i)+1:2d}. {row['feature']}: {row['importance']:.4f}")
    
    # CSV 저장
    importance.to_csv('importance_no_leakage.csv', index=False, encoding='utf-8-sig')
    print(f"\n✅ 저장: importance_no_leakage.csv")
    
except ImportError:
    print("❌ xgboost 필요")

# ============================================================
# 4. queue_gap 심층 분석
# ============================================================
print("\n" + "="*60)
print("🔥 4. queue_gap 심층 분석")
print("="*60)

if 'queue_gap' in df.columns:
    print(f"queue_gap 평균: {df['queue_gap'].mean():.2f}")
    print(f"queue_gap 표준편차: {df['queue_gap'].std():.2f}")
    print(f"queue_gap 범위: {df['queue_gap'].min():.0f} ~ {df['queue_gap'].max():.0f}")
    
    # 전체 상관계수
    corr_all = df['queue_gap'].corr(df[target])
    print(f"\n📌 전체 데이터 상관계수: {corr_all:.4f}")
    
    # 1600+ 구간만
    high_mask = df[target] >= 1600
    if high_mask.sum() > 0:
        corr_high = df[high_mask]['queue_gap'].corr(df[high_mask][target])
        print(f"📌 1600+ 구간 상관계수: {corr_high:.4f}")
    
    # 1700+ 구간만
    danger_mask = df[target] >= 1700
    if danger_mask.sum() > 0:
        corr_danger = df[danger_mask]['queue_gap'].corr(df[danger_mask][target])
        print(f"📌 1700+ 구간 상관계수: {corr_danger:.4f}")
    
    # 구간별 분석
    print(f"\n📌 queue_gap 구간별 1700+ 확률:")
    bins = [-9999, 50, 100, 150, 200, 250, 300, 350, 400, 500, 9999]
    labels = ['<50', '50-100', '100-150', '150-200', '200-250', '250-300', '300-350', '350-400', '400-500', '500+']
    df['gap_bin'] = pd.cut(df['queue_gap'], bins=bins, labels=labels)
    
    for label in labels:
        mask = df['gap_bin'] == label
        if mask.sum() > 0:
            cnt = mask.sum()
            avg = df[mask][target].mean()
            rate_1700 = (df[mask][target] >= 1700).sum() / cnt * 100
            print(f"    {label:10s}: {cnt:6,}개, 평균 {avg:6.0f}, 1700+ {rate_1700:5.2f}%")
    
    df.drop('gap_bin', axis=1, inplace=True)

# ============================================================
# 5. 황금 패턴 분석
# ============================================================
print("\n" + "="*60)
print("✨ 5. 황금 패턴 분석")
print("="*60)

if 'M14AM14B' in df.columns and 'M14AM14BSUM' in df.columns:
    print(f"M14AM14B 범위: {df['M14AM14B'].min()} ~ {df['M14AM14B'].max()}")
    print(f"M14AM14BSUM 범위: {df['M14AM14BSUM'].min()} ~ {df['M14AM14BSUM'].max()}")
    
    # 상관계수
    corr_m14b = df['M14AM14B'].corr(df[target])
    corr_m14bsum = df['M14AM14BSUM'].corr(df[target])
    print(f"\n📌 TOTALCNT 상관계수:")
    print(f"  M14AM14B: {corr_m14b:.4f}")
    print(f"  M14AM14BSUM: {corr_m14bsum:.4f}")
    
    # 다양한 임계값 테스트
    print(f"\n📌 황금 패턴 임계값별 1700+ 확률:")
    thresholds = [
        (500, 580), (510, 590), (520, 600), (530, 610), (540, 620),
        (550, 630), (560, 640), (580, 660), (600, 680)
    ]
    
    for m14b_th, sum_th in thresholds:
        mask = (df['M14AM14B'] > m14b_th) & (df['M14AM14BSUM'] > sum_th)
        if mask.sum() > 0:
            cnt = mask.sum()
            rate_1700 = (df[mask][target] >= 1700).sum() / cnt * 100
            avg = df[mask][target].mean()
            print(f"    M14B>{m14b_th} & SUM>{sum_th}: {cnt:5,}개, 평균 {avg:.0f}, 1700+ {rate_1700:5.1f}%")

# ============================================================
# 6. 1700+ 구간 특성 분석
# ============================================================
print("\n" + "="*60)
print("🚨 6. 1700+ 구간 특성 분석")
print("="*60)

danger_mask = df[target] >= 1700
normal_mask = df[target] < 1600

print(f"1700+ 데이터: {danger_mask.sum():,}개")
print(f"정상(<1600) 데이터: {normal_mask.sum():,}개")

# 주요 변수 비교
key_vars = ['queue_gap', 'M14AM14B', 'M14AM14BSUM', 'M10AM14A', 
            'M14.QUE.ALL.TRANSPORT4MINOVERCNT', 'M14.QUE.OHT.OHTUTIL',
            'M14.QUE.LOAD.CURRENTLOADQCNT', 'M14.QUE.SENDFAB.CURRENTSENDFABCOMPLETED']

print(f"\n📌 주요 변수 비교 (정상 vs 위험):")
print(f"{'변수':<45} {'정상평균':>10} {'위험평균':>10} {'배율':>8}")
print("-" * 75)

for var in key_vars:
    if var in df.columns:
        normal_avg = df[normal_mask][var].mean()
        danger_avg = df[danger_mask][var].mean()
        ratio = danger_avg / normal_avg if normal_avg != 0 else 0
        print(f"{var:<45} {normal_avg:>10.1f} {danger_avg:>10.1f} {ratio:>8.2f}x")

# ============================================================
# 7. 복합 조건 분석 (Triple Check)
# ============================================================
print("\n" + "="*60)
print("🔥 7. 복합 조건 분석")
print("="*60)

if 'queue_gap' in df.columns and 'M14AM14B' in df.columns:
    conditions = [
        ("M14B>520 & SUM>600", (df['M14AM14B'] > 520) & (df['M14AM14BSUM'] > 600)),
        ("M14B>520 & SUM>600 & gap>200", (df['M14AM14B'] > 520) & (df['M14AM14BSUM'] > 600) & (df['queue_gap'] > 200)),
        ("M14B>520 & SUM>600 & gap>250", (df['M14AM14B'] > 520) & (df['M14AM14BSUM'] > 600) & (df['queue_gap'] > 250)),
        ("M14B>520 & SUM>600 & gap>300", (df['M14AM14B'] > 520) & (df['M14AM14BSUM'] > 600) & (df['queue_gap'] > 300)),
        ("gap>300 만", df['queue_gap'] > 300),
        ("gap>350 만", df['queue_gap'] > 350),
        ("gap>400 만", df['queue_gap'] > 400),
    ]
    
    print(f"{'조건':<40} {'발생':>8} {'1700+':>8} {'확률':>8}")
    print("-" * 70)
    
    for name, mask in conditions:
        if mask.sum() > 0:
            cnt = mask.sum()
            danger_cnt = (df[mask][target] >= 1700).sum()
            rate = danger_cnt / cnt * 100
            print(f"{name:<40} {cnt:>8,} {danger_cnt:>8,} {rate:>7.1f}%")

# ============================================================
# 8. 결과 요약 저장
# ============================================================
print("\n" + "="*60)
print("💾 8. 결과 요약")
print("="*60)

summary = []
summary.append(f"총 데이터: {len(df):,}행")
summary.append(f"1700+ 데이터: {(df[target]>=1700).sum():,}개 ({(df[target]>=1700).sum()/len(df)*100:.2f}%)")
summary.append(f"TOTALCNT 범위: {df[target].min()} ~ {df[target].max()}")

if 'queue_gap' in df.columns:
    summary.append(f"queue_gap 범위: {df['queue_gap'].min():.0f} ~ {df['queue_gap'].max():.0f}")
    summary.append(f"queue_gap-TOTALCNT 상관: {df['queue_gap'].corr(df[target]):.4f}")

with open('analysis_summary.txt', 'w', encoding='utf-8') as f:
    f.write("\n".join(summary))
print("✅ 저장: analysis_summary.txt")

print("\n" + "="*80)
print("✅ 분석 완료!")
print("="*80)