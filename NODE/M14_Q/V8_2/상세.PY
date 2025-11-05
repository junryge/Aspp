#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 전체 컬럼 상세 분석 코드 (카테고리 분류 포함)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ========================================
# CSV 파일 경로 (수정하세요!)
# ========================================
CSV_FILE = "당신의파일.csv"

print("="*120)
print(" "*40 + "📊 전체 컬럼 상세 분석")
print("="*120)

# 데이터 로드
print(f"\n📂 파일 로딩: {CSV_FILE}")
df = pd.read_csv(CSV_FILE)
print(f"✅ {len(df):,}행 × {df.shape[1]}열\n")

# 기본 정보
if 'CURRTIME' in df.columns:
    print(f"기간: {df['CURRTIME'].min()} ~ {df['CURRTIME'].max()}")
if 'TOTALCNT' in df.columns:
    print(f"TOTALCNT: {df['TOTALCNT'].min():.0f} ~ {df['TOTALCNT'].max():.0f}")
    print(f"1700+ 구간: {(df['TOTALCNT']>=1700).sum():,}개 ({(df['TOTALCNT']>=1700).sum()/len(df)*100:.3f}%)")

# ========================================
# 컬럼 통계 계산
# ========================================
print(f"\n" + "="*120)
print("🔍 컬럼별 통계 계산 중...")
print("="*120)

stats_list = []

for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        mean_val = df[col].mean()
        std_val = df[col].std()
        
        stat = {
            '컬럼명': col,
            '평균': mean_val,
            '표준편차': std_val,
            '최소': df[col].min(),
            '최대': df[col].max(),
            '결측': df[col].isna().sum(),
            '고유값': df[col].nunique(),
            '변동계수': std_val / (abs(mean_val) + 0.0001),
        }
        
        # TOTALCNT 상관관계
        if 'TOTALCNT' in df.columns and col != 'TOTALCNT':
            try:
                corr = df[[col, 'TOTALCNT']].corr().iloc[0,1]
                stat['상관'] = corr if not np.isnan(corr) else 0
            except:
                stat['상관'] = 0
        else:
            stat['상관'] = 1.0 if col == 'TOTALCNT' else 0
        
        stats_list.append(stat)

stats_df = pd.DataFrame(stats_list)

# ========================================
# TOP 컬럼 출력
# ========================================
print(f"\n" + "="*120)
print("🎯 TOTALCNT 상관관계 TOP 40")
print("="*120)
print(f"{'순위':<4} {'컬럼명':<55} {'상관':>8} {'변동':>8} {'평균':>12} {'범위':>18} {'중요도':>6}")
print("-"*120)

sorted_stats = stats_df.sort_values('상관', key=lambda x: abs(x), ascending=False)

for idx, (_, row) in enumerate(sorted_stats.head(40).iterrows(), 1):
    marker = "⭐⭐⭐" if abs(row['상관']) > 0.5 else ("⭐⭐" if abs(row['상관']) > 0.3 else ("⭐" if abs(row['상관']) > 0.15 else ""))
    range_str = f"{row['최소']:.0f}~{row['최대']:.0f}"
    print(f"{idx:2d}. {row['컬럼명']:<53s} {row['상관']:>+8.3f} {row['변동계수']:>8.3f} {row['평균']:>12.1f} {range_str:>18s} {marker:>6s}")

# ========================================
# 카테고리 분류
# ========================================
print(f"\n" + "="*120)
print("📂 컬럼 카테고리별 분류")
print("="*120)

categories = {}

for col in df.columns:
    if 'TIME' in col and col != 'TRANSPORT':
        cat = '⏰ 시간'
    elif col == 'TOTALCNT':
        cat = '🎯 타겟'
    elif 'M14AM14B' in col or 'M14AM10A' in col or 'M14AM16' in col:
        cat = '📤 물류_순방향'
    elif 'M14BM14A' in col or 'M10AM14A' in col or 'M16M14A' in col:
        cat = '📥 물류_역방향'
    elif 'SUM' in col:
        cat = '📊 물류_합계'
    elif 'M10A.QUE' in col:
        cat = '🏭 M10A큐정보'
    elif 'M14.QUE' in col or 'M14B.QUE' in col:
        cat = '🔧 M14큐관리'
    elif 'ALLALL_CD' in col:
        cat = '💾 저장소_CD'
    elif 'M14M16_CD' in col or 'M10M16_CD' in col:
        cat = '🔀 M14M16_CD'
    elif 'M16M14_CD' in col or 'M16M10_CD' in col:
        cat = '🔁 M16M14_CD'
    else:
        cat = '❓ 기타'
    
    if cat not in categories:
        categories[cat] = []
    categories[cat].append(col)

for cat_name in sorted(categories.keys()):
    cols = categories[cat_name]
    print(f"\n{cat_name} ({len(cols)}개)")
    for col in cols:
        if col in stats_df['컬럼명'].values:
            row = stats_df[stats_df['컬럼명']==col].iloc[0]
            print(f"  • {col:<55s} 상관:{row['상관']:>+7.3f}  변동:{row['변동계수']:>6.3f}")
        else:
            print(f"  • {col}")

# ========================================
# 1700+ 선행 지표
# ========================================
if 'TOTALCNT' in df.columns:
    print(f"\n" + "="*120)
    print("🔥 1700+ 위험 구간 선행 지표 (30분 전 평균 비교)")
    print("="*120)
    
    high_idx = df[df['TOTALCNT'] >= 1700].index.tolist()
    print(f"1700+ 발생: {len(high_idx):,}회\n")
    
    if len(high_idx) > 0:
        print(f"{'컬럼명':<55} {'1700+직전':>12} {'전체평균':>12} {'배율':>8} {'상관':>8}")
        print("-"*120)
        
        lead_list = []
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64'] and col not in ['CURRTIME', 'CURRTIME.1', 'TIME', 'TOTALCNT']:
                vals_30min = []
                for idx in high_idx:
                    if idx >= 30:
                        vals_30min.append(df[col].iloc[idx-30:idx].mean())
                
                if len(vals_30min) > 0 and df[col].mean() != 0:
                    avg_30 = np.mean(vals_30min)
                    avg_all = df[col].mean()
                    ratio = avg_30 / avg_all
                    corr_val = stats_df[stats_df['컬럼명']==col]['상관'].values[0] if col in stats_df['컬럼명'].values else 0
                    
                    lead_list.append({
                        'col': col,
                        'lead': avg_30,
                        'avg': avg_all,
                        'ratio': ratio,
                        'corr': abs(corr_val)
                    })
        
        lead_df = pd.DataFrame(lead_list).sort_values('ratio', ascending=False)
        
        for i, r in lead_df.head(25).iterrows():
            mark = "🔥" if r['ratio'] > 1.5 else ("⚠️" if r['ratio'] > 1.2 else "")
            print(f"{r['col']:<55s} {r['lead']:>12.1f} {r['avg']:>12.1f} {r['ratio']:>8.2f}x {r['corr']:>7.3f} {mark}")

# ========================================
# 제외 권장
# ========================================
print(f"\n" + "="*120)
print("❌ 제외 권장 컬럼")
print("="*120)

print("\n【상수】")
for _, r in stats_df[stats_df['고유값']==1].iterrows():
    print(f"  • {r['컬럼명']:<55s} (값: {r['평균']:.1f})")

print("\n【중복】")
if 'CURRTIME' in df.columns and 'CURRTIME.1' in df.columns:
    if (df['CURRTIME']==df['CURRTIME.1']).all():
        print("  • CURRTIME.1")
if 'TIME' in df.columns:
    print("  • TIME")

print("\n【저변동/저상관】(CV<0.1 AND |r|<0.1)")
low = stats_df[(stats_df['변동계수']<0.1) & (abs(stats_df['상관'])<0.1)]
low = low[~low['컬럼명'].isin(['CURRTIME', 'CURRTIME.1', 'TIME', 'TOTALCNT'])]
for _, r in low.iterrows():
    print(f"  • {r['컬럼명']}")

# ========================================
# 추천 컬럼
# ========================================
print(f"\n" + "="*120)
print("✅ 추천 사용 컬럼")
print("="*120)

rec = stats_df[abs(stats_df['상관']) >= 0.15]
rec = rec[~rec['컬럼명'].isin(['CURRTIME.1', 'TIME', 'TOTALCNT'])]

print(f"\n【높은 상관 컬럼】 (|r| ≥ 0.15) - {len(rec)}개")
for i, (_, r) in enumerate(rec.iterrows(), 1):
    print(f"  {i:2d}. {r['컬럼명']:<55s} {r['상관']:>+7.3f}")

# CSV 저장
out_file = "컬럼분석_상세.csv"
stats_df.to_csv(out_file, index=False, encoding='utf-8-sig')
print(f"\n💾 저장: {out_file}")

print(f"\n" + "="*120)
print("✅ 분석 완료!")
print("="*120)