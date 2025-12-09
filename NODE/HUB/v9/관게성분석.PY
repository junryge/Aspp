import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🎯 전체 컬럼 관계성 분석")
print("="*80)

# ========== 데이터 로드 ==========
DATA_FILE = '10월_데이터.csv'  # ← 파일명 수정!

try:
    df = pd.read_csv(DATA_FILE, encoding='utf-8', low_memory=False)
except:
    try:
        df = pd.read_csv(DATA_FILE, encoding='cp949', low_memory=False)
    except:
        df = pd.read_csv(DATA_FILE, encoding='euc-kr', low_memory=False)

print(f"\n✅ 데이터: {len(df)}행 x {len(df.columns)}컬럼")

TARGET_COL = 'CURRENT_M16A_3F_JOB_2'

# 타겟 전처리
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')
df = df.dropna(subset=[TARGET_COL])

# 10분 후 타겟값
df['TARGET_FUTURE_10'] = df[TARGET_COL].shift(-10)
df['TARGET_CHANGE_10'] = df['TARGET_FUTURE_10'] - df[TARGET_COL]

print(f"✅ 유효 데이터: {len(df)}행")

# ========== 상관관계 분석 ==========
print("\n📊 상관관계 계산 중...")

results = []

for col in df.columns:
    if col in ['STAT_DT', TARGET_COL, 'TARGET_FUTURE_10', 'TARGET_CHANGE_10']:
        continue
    
    df[col] = pd.to_numeric(df[col], errors='coerce')
    
    if df[col].std() == 0 or df[col].isna().all():
        continue
    
    corr_current = df[col].corr(df[TARGET_COL])
    corr_future = df[col].corr(df['TARGET_FUTURE_10'])
    corr_change = df[col].corr(df['TARGET_CHANGE_10'])
    
    results.append({
        '컬럼명': col,
        '현재_상관': round(corr_current, 4) if not np.isnan(corr_current) else 0,
        '10분후_상관': round(corr_future, 4) if not np.isnan(corr_future) else 0,
        '변화량_상관': round(corr_change, 4) if not np.isnan(corr_change) else 0,
        '평균': round(df[col].mean(), 2),
        '표준편차': round(df[col].std(), 2),
        '최소': round(df[col].min(), 2),
        '최대': round(df[col].max(), 2)
    })

results_df = pd.DataFrame(results)

# ========== 결과 출력 ==========
print("\n" + "="*80)
print("🔥 현재 타겟 상관관계 Top 30")
print("="*80)
top_current = results_df.reindex(results_df['현재_상관'].abs().sort_values(ascending=False).index).head(30)
for _, row in top_current.iterrows():
    d = "↑" if row['현재_상관'] > 0 else "↓"
    print(f"  {row['컬럼명']:55s} {row['현재_상관']:+.4f} {d}")

print("\n" + "="*80)
print("🎯 10분 후 예측력 Top 30 ⭐")
print("="*80)
top_future = results_df.reindex(results_df['10분후_상관'].abs().sort_values(ascending=False).index).head(30)
for _, row in top_future.iterrows():
    d = "↑" if row['10분후_상관'] > 0 else "↓"
    print(f"  {row['컬럼명']:55s} {row['10분후_상관']:+.4f} {d}")

print("\n" + "="*80)
print("📈 변화량 예측력 Top 30 ⭐⭐")
print("="*80)
top_change = results_df.reindex(results_df['변화량_상관'].abs().sort_values(ascending=False).index).head(30)
for _, row in top_change.iterrows():
    d = "↑" if row['변화량_상관'] > 0 else "↓"
    print(f"  {row['컬럼명']:55s} {row['변화량_상관']:+.4f} {d}")

# CSV 저장
output_file = '관계성_분석_결과.csv'
results_df.sort_values('변화량_상관', key=abs, ascending=False).to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n✅ 결과 저장: {output_file}")

# 요약
print("\n" + "="*80)
print("📋 요약")
print("="*80)
print(f"분석 컬럼: {len(results)}개")
print(f"강한 현재 상관 (|r|≥0.5): {len(results_df[results_df['현재_상관'].abs() >= 0.5])}개")
print(f"강한 미래 상관 (|r|≥0.5): {len(results_df[results_df['10분후_상관'].abs() >= 0.5])}개")
print(f"변화량 상관 (|r|≥0.1): {len(results_df[results_df['변화량_상관'].abs() >= 0.1])}개")

print("\n🎉 완료!")