#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏭 AMHS 장애예측 판단 데이터 EDA
목적: 반도체 공장 물류량(TOTALCNT) 예측을 위한 탐색적 데이터 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("🏭 AMHS 장애예측 EDA 분석")
print("="*70)

# ============================================
# 1. 데이터 로드
# ============================================
DATA_PATH = 'your_data.csv'  # ← 실제 파일 경로로 변경!

df = pd.read_csv(DATA_PATH, on_bad_lines='skip')
print(f"\n✅ 데이터 로드: {df.shape[0]:,}행 × {df.shape[1]}열")

# 기본 정보
print("\n" + "="*70)
print("📌 1. 기본 정보")
print("="*70)
print(df.info())
print("\n기술통계:")
print(df.describe())

# ============================================
# 2. 변수 타입 정의 (카테고리 vs 연속형)
# ============================================
print("\n" + "="*70)
print("📌 2. 변수 타입 분석")
print("="*70)

def analyze_variable_types(df, unique_threshold=20):
    results = []
    for col in df.columns:
        unique_count = df[col].nunique()
        dtype = str(df[col].dtype)
        
        if col == 'CURRTIME':
            var_type = 'DATETIME'
        elif dtype == 'object':
            var_type = 'CATEGORY'
        elif unique_count <= unique_threshold:
            var_type = 'CATEGORY(후보)'
        else:
            var_type = 'CONTINUOUS'
        
        results.append({
            '컬럼명': col,
            '데이터타입': dtype,
            '고유값수': unique_count,
            '변수타입': var_type,
            '결측치': df[col].isna().sum(),
            '결측률(%)': round(df[col].isna().sum() / len(df) * 100, 2)
        })
    return pd.DataFrame(results)

var_types = analyze_variable_types(df)
print(var_types.to_string())
print("\n📊 변수 타입별 개수:")
print(var_types['변수타입'].value_counts())

# 카테고리 세그먼트 정의
print("\n📊 카테고리 변수 세그먼트:")
cat_cols = var_types[var_types['변수타입'].str.contains('CATEGORY')]['컬럼명'].tolist()
for col in cat_cols[:10]:  # 상위 10개만
    if col in df.columns:
        print(f"\n▶ {col}")
        vc = df[col].value_counts()
        for val, cnt in vc.head(5).items():
            print(f"  '{val}': {cnt:,}개 ({cnt/len(df)*100:.1f}%)")

# ============================================
# 3. 결측치 분석
# ============================================
print("\n" + "="*70)
print("📌 3. 결측치 분석")
print("="*70)

missing_df = pd.DataFrame({
    '결측치수': df.isna().sum(),
    '결측률(%)': round(df.isna().sum() / len(df) * 100, 2)
}).sort_values('결측치수', ascending=False)

missing_df = missing_df[missing_df['결측치수'] > 0]

if len(missing_df) > 0:
    print(f"결측치 있는 컬럼: {len(missing_df)}개")
    print(missing_df.head(20))
else:
    print("✅ 결측치 없음!")

# missingno 시각화 (설치된 경우)
try:
    import missingno as msno
    if df.isna().sum().sum() > 0:
        fig, ax = plt.subplots(figsize=(16, 8))
        msno.matrix(df, ax=ax, sparkline=False)
        plt.title('결측치 Matrix')
        plt.tight_layout()
        plt.savefig('missing_analysis.png', dpi=150)
        plt.close()
        print("💾 missing_analysis.png 저장")
except ImportError:
    print("⚠️ missingno 미설치 (pip install missingno)")

# ============================================
# 4. 이상치 분석
# ============================================
print("\n" + "="*70)
print("📌 4. 이상치 분석 (IQR)")
print("="*70)

def detect_outliers_iqr(series, multiplier=1.5):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR
    outliers = (series < lower) | (series > upper)
    return outliers.sum(), lower, upper, Q1, Q3

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'CURRTIME' in numeric_cols:
    numeric_cols.remove('CURRTIME')

outlier_results = []
for col in numeric_cols:
    if df[col].notna().sum() > 0:
        count, lower, upper, q1, q3 = detect_outliers_iqr(df[col].dropna())
        outlier_results.append({
            '컬럼': col,
            'Q1': round(q1, 2),
            'Q3': round(q3, 2),
            '하한': round(lower, 2),
            '상한': round(upper, 2),
            '이상치수': count,
            '이상치율(%)': round(count / len(df) * 100, 2)
        })

outlier_df = pd.DataFrame(outlier_results).sort_values('이상치율(%)', ascending=False)
print(outlier_df.head(20).to_string(index=False))

# Boxplot
key_cols = ['TOTALCNT', 'M14AM14B', 'M14AM14BSUM', 'M10AM14A']
key_cols = [c for c in key_cols if c in df.columns]

if key_cols:
    fig, axes = plt.subplots(1, len(key_cols), figsize=(4*len(key_cols), 5))
    if len(key_cols) == 1:
        axes = [axes]
    for i, col in enumerate(key_cols):
        axes[i].boxplot(df[col].dropna())
        axes[i].set_title(col)
    plt.suptitle('주요 변수 Boxplot')
    plt.tight_layout()
    plt.savefig('boxplot_analysis.png', dpi=150)
    plt.close()
    print("💾 boxplot_analysis.png 저장")

# TOTALCNT 분포
if 'TOTALCNT' in df.columns:
    print("\n📊 TOTALCNT 구간별 분포:")
    print(f"  < 900 (LOW): {(df['TOTALCNT'] < 900).sum():,}개 ({(df['TOTALCNT'] < 900).sum()/len(df)*100:.2f}%)")
    print(f"  900~1599 (NORMAL): {((df['TOTALCNT'] >= 900) & (df['TOTALCNT'] < 1600)).sum():,}개")
    print(f"  1600~1699 (CAUTION): {((df['TOTALCNT'] >= 1600) & (df['TOTALCNT'] < 1700)).sum():,}개")
    print(f"  >= 1700 (CRITICAL): {(df['TOTALCNT'] >= 1700).sum():,}개 ({(df['TOTALCNT'] >= 1700).sum()/len(df)*100:.2f}%)")

# ============================================
# 5. 중복값 분석
# ============================================
print("\n" + "="*70)
print("📌 5. 중복값 분석")
print("="*70)

# 완전 중복
full_dup = df.duplicated().sum()
print(f"1️⃣ 완전 중복: {full_dup:,}개 ({full_dup/len(df)*100:.2f}%)")

# CURRTIME 제외 중복
cols_except_time = [c for c in df.columns if c != 'CURRTIME']
if cols_except_time:
    time_diff_dup = df.duplicated(subset=cols_except_time).sum()
    print(f"2️⃣ CURRTIME 제외 중복: {time_diff_dup:,}개 ({time_diff_dup/len(df)*100:.2f}%)")

# 연속 중복
consecutive_dup = 0
for i in range(1, len(df)):
    if df[cols_except_time].iloc[i].equals(df[cols_except_time].iloc[i-1]):
        consecutive_dup += 1
print(f"3️⃣ 연속 중복 (직전과 동일): {consecutive_dup:,}개")

# ============================================
# 6. 상관관계 분석
# ============================================
print("\n" + "="*70)
print("📌 6. 상관관계 분석")
print("="*70)

numeric_df = df.select_dtypes(include=[np.number])
if 'CURRTIME' in numeric_df.columns:
    numeric_df = numeric_df.drop(columns=['CURRTIME'])

corr_matrix = numeric_df.corr()

# TOTALCNT 상관관계
if 'TOTALCNT' in corr_matrix.columns:
    totalcnt_corr = corr_matrix['TOTALCNT'].drop('TOTALCNT').sort_values(key=abs, ascending=False)
    print("\n🏆 TOTALCNT 상관관계 TOP 20:")
    print(totalcnt_corr.head(20))

# 높은 상관관계 쌍 (0.95 이상)
print("\n📊 높은 상관관계 쌍 (|r| >= 0.95):")
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
        corr_val = corr_matrix.iloc[i, j]
        if abs(corr_val) >= 0.95:
            high_corr_pairs.append({'변수1': col1, '변수2': col2, '상관계수': round(corr_val, 4)})

if high_corr_pairs:
    high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('상관계수', key=abs, ascending=False)
    print(f"발견: {len(high_corr_df)}쌍")
    print(high_corr_df.to_string(index=False))
else:
    print("✅ 0.95 이상 없음")

# 히트맵
if 'TOTALCNT' in corr_matrix.columns:
    top_cols = ['TOTALCNT'] + totalcnt_corr.head(15).index.tolist()
    top_cols = [c for c in top_cols if c in numeric_df.columns]
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(numeric_df[top_cols].corr(), annot=True, fmt='.2f', cmap='RdBu_r', center=0)
    plt.title('주요 변수 상관관계')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=150)
    plt.close()
    print("💾 correlation_heatmap.png 저장")

# ============================================
# 7. 파생 변수 (queue_gap)
# ============================================
print("\n" + "="*70)
print("📌 7. 파생 변수 분석")
print("="*70)

# M14 queue_gap
if 'M14.QUE.ALL.CURRENTQCREATED' in df.columns and 'M14.QUE.ALL.CURRENTQCOMPLETED' in df.columns:
    df['M14_queue_gap'] = df['M14.QUE.ALL.CURRENTQCREATED'] - df['M14.QUE.ALL.CURRENTQCOMPLETED']
    print(f"\n✅ M14_queue_gap = CURRENTQCREATED - CURRENTQCOMPLETED")
    print(f"   범위: {df['M14_queue_gap'].min():.0f} ~ {df['M14_queue_gap'].max():.0f}")
    print(f"   평균: {df['M14_queue_gap'].mean():.2f}")
    if 'TOTALCNT' in df.columns:
        gap_corr = df['M14_queue_gap'].corr(df['TOTALCNT'])
        print(f"   🔥 TOTALCNT 상관계수: {gap_corr:.4f}")

# M14B queue_gap
if 'M14B.QUE.ALL.CURRENTQCREATED' in df.columns and 'M14B.QUE.ALL.CURRENTQCOMPLETED' in df.columns:
    df['M14B_queue_gap'] = df['M14B.QUE.ALL.CURRENTQCREATED'] - df['M14B.QUE.ALL.CURRENTQCOMPLETED']
    print(f"\n✅ M14B_queue_gap")
    print(f"   범위: {df['M14B_queue_gap'].min():.0f} ~ {df['M14B_queue_gap'].max():.0f}")
    if 'TOTALCNT' in df.columns:
        gap_corr = df['M14B_queue_gap'].corr(df['TOTALCNT'])
        print(f"   TOTALCNT 상관계수: {gap_corr:.4f}")

# ============================================
# 8. Feature Importance (XGBoost)
# ============================================
print("\n" + "="*70)
print("📌 8. Feature Importance (XGBoost)")
print("="*70)

try:
    from sklearn.model_selection import train_test_split
    import xgboost as xgb
    
    feature_cols = [c for c in numeric_df.columns if c != 'TOTALCNT']
    X = numeric_df[feature_cols].fillna(0)
    y = numeric_df['TOTALCNT'] if 'TOTALCNT' in numeric_df.columns else None
    
    if y is not None and len(X) > 100:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = xgb.XGBRegressor(max_depth=6, learning_rate=0.1, n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🏆 Feature Importance TOP 30:")
        print(importance_df.head(30).to_string(index=False))
        
        # 시각화
        plt.figure(figsize=(12, 10))
        plt.barh(importance_df['feature'].head(30)[::-1], importance_df['importance'].head(30)[::-1])
        plt.xlabel('Importance')
        plt.title('XGBoost Feature Importance TOP 30')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=150)
        plt.close()
        print("💾 feature_importance.png 저장")
        
        importance_df.to_csv('feature_importance.csv', index=False, encoding='utf-8-sig')
except ImportError:
    print("⚠️ xgboost 미설치 (pip install xgboost)")

# ============================================
# 9. 결과 저장
# ============================================
print("\n" + "="*70)
print("📌 9. 결과 저장")
print("="*70)

var_types.to_csv('variable_types.csv', index=False, encoding='utf-8-sig')
print("💾 variable_types.csv")

outlier_df.to_csv('outlier_analysis.csv', index=False, encoding='utf-8-sig')
print("💾 outlier_analysis.csv")

if high_corr_pairs:
    high_corr_df.to_csv('high_correlation_pairs.csv', index=False, encoding='utf-8-sig')
    print("💾 high_correlation_pairs.csv")

# ============================================
# 요약 리포트
# ============================================
print("\n" + "="*70)
print("📋 EDA 요약")
print("="*70)
print(f"""
데이터: {len(df):,}행 × {len(df.columns)}열
수치형: {len(numeric_cols)}개
결측치 컬럼: {len(missing_df)}개
1700+ 비율: {(df['TOTALCNT'] >= 1700).sum()/len(df)*100:.2f}%
높은상관(0.95+): {len(high_corr_pairs)}쌍

📌 1차 회의 안건:
□ 카테고리/연속형 구분 확정
□ 상관관계 기반 변수 제거 (도메인 지식)
□ 결측치/이상치 처리 전략
□ 중복값 처리
""")

print("\n✅ EDA 완료!")