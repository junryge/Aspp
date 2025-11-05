#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 전체 컬럼 분석 코드
당신의 CSV 파일을 분석해서 어떤 컬럼을 사용할지 결정합니다
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ========================================
# 1. CSV 파일 경로 설정 (여기를 수정하세요!)
# ========================================
CSV_FILE = "당신의파일.csv"  # ← 여기에 파일 경로를 넣으세요!

print("="*100)
print(" "*30 + "📊 전체 컬럼 분석 시작")
print("="*100)

# ========================================
# 2. 데이터 로드
# ========================================
print(f"\n📂 파일 로딩: {CSV_FILE}")
try:
    df = pd.read_csv(CSV_FILE)
    print(f"✅ 로드 성공: {len(df):,}행 × {df.shape[1]}열")
except Exception as e:
    print(f"❌ 에러: {e}")
    exit(1)

# ========================================
# 3. 기본 정보
# ========================================
print(f"\n" + "="*100)
print("📋 기본 정보")
print("="*100)
print(f"데이터 크기: {len(df):,}행 × {df.shape[1]}열")
if 'CURRTIME' in df.columns:
    print(f"데이터 기간: {df['CURRTIME'].min()} ~ {df['CURRTIME'].max()}")
if 'TOTALCNT' in df.columns:
    print(f"TOTALCNT 범위: {df['TOTALCNT'].min():.0f} ~ {df['TOTALCNT'].max():.0f}")
    print(f"1700+ 구간: {(df['TOTALCNT']>=1700).sum():,}개 ({(df['TOTALCNT']>=1700).sum()/len(df)*100:.2f}%)")

# ========================================
# 4. 전체 컬럼 목록
# ========================================
print(f"\n" + "="*100)
print(f"📋 전체 컬럼 목록 ({len(df.columns)}개)")
print("="*100)
for i, col in enumerate(df.columns, 1):
    dtype = df[col].dtype
    null_count = df[col].isna().sum()
    null_pct = null_count / len(df) * 100
    print(f"{i:3d}. {col:<55s} {str(dtype):<10s} (결측:{null_count:4d}, {null_pct:5.1f}%)")

# ========================================
# 5. 컬럼별 상세 통계
# ========================================
print(f"\n" + "="*100)
print("📊 컬럼별 상세 통계 (숫자형 컬럼만)")
print("="*100)

column_stats = []

for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        stats = {
            '컬럼명': col,
            '평균': df[col].mean(),
            '표준편차': df[col].std(),
            '최소': df[col].min(),
            '최대': df[col].max(),
            '결측': df[col].isna().sum(),
            '고유값': df[col].nunique(),
        }
        
        # 변동계수 (CV)
        if stats['평균'] != 0:
            stats['변동계수'] = stats['표준편차'] / abs(stats['평균'])
        else:
            stats['변동계수'] = 0
        
        # TOTALCNT와 상관관계
        if 'TOTALCNT' in df.columns and col != 'TOTALCNT':
            try:
                corr = df[[col, 'TOTALCNT']].corr().iloc[0,1]
                stats['TOTALCNT상관'] = corr if not np.isnan(corr) else 0
            except:
                stats['TOTALCNT상관'] = 0
        else:
            stats['TOTALCNT상관'] = 1.0 if col == 'TOTALCNT' else 0
        
        column_stats.append(stats)

stats_df = pd.DataFrame(column_stats)

# 상관관계 순으로 정렬
stats_df = stats_df.sort_values('TOTALCNT상관', key=lambda x: abs(x), ascending=False)

print(f"\n{'순위':<4} {'컬럼명':<50} {'상관계수':>10} {'변동계수':>10} {'평균':>15} {'범위':>20}")
print("-"*100)

for idx, (_, row) in enumerate(stats_df.head(40).iterrows(), 1):
    col_name = row['컬럼명']
    corr = row['TOTALCNT상관']
    cv = row['변동계수']
    mean = row['평균']
    range_str = f"{row['최소']:.0f}~{row['최대']:.0f}"
    
    # 중요도 표시
    if abs(corr) > 0.5:
        marker = "⭐⭐⭐"
    elif abs(corr) > 0.3:
        marker = "⭐⭐"
    elif abs(corr) > 0.15:
        marker = "⭐"
    else:
        marker = ""
    
    print(f"{idx:2d}. {col_name:<48s} {corr:>+10.4f} {cv:>10.3f} {mean:>15.1f} {range_str:>20s} {marker}")

# ========================================
# 6. 1700+ 구간 선행 지표 분석
# ========================================
if 'TOTALCNT' in df.columns:
    print(f"\n" + "="*100)
    print("🔥 1700+ 위험 구간 선행 지표 분석 (30분 전)")
    print("="*100)
    
    high_risk = df[df['TOTALCNT'] >= 1700].index.tolist()
    print(f"1700+ 구간: {len(high_risk):,}개 ({len(high_risk)/len(df)*100:.2f}%)\n")
    
    if len(high_risk) > 0:
        print(f"{'컬럼명':<50} {'1700+직전':>12} {'전체평균':>12} {'배율':>8} {'상관':>8}")
        print("-"*100)
        
        lead_results = []
        
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64'] and col not in ['CURRTIME', 'CURRTIME.1', 'TIME', 'TOTALCNT']:
                lead_vals = []
                for idx in high_risk:
                    if idx >= 30:
                        lead_vals.append(df[col].iloc[idx-30:idx].mean())
                
                if len(lead_vals) > 0 and df[col].mean() != 0:
                    lead_avg = np.mean(lead_vals)
                    overall_avg = df[col].mean()
                    ratio = lead_avg / overall_avg
                    corr = abs(stats_df[stats_df['컬럼명']==col]['TOTALCNT상관'].values[0]) if col in stats_df['컬럼명'].values else 0
                    
                    lead_results.append({
                        '컬럼': col,
                        '1700+직전': lead_avg,
                        '전체평균': overall_avg,
                        '배율': ratio,
                        '상관': corr
                    })
        
        lead_df = pd.DataFrame(lead_results).sort_values('배율', ascending=False)
        
        for idx, row in lead_df.head(30).iterrows():
            marker = "🔥" if row['배율'] > 1.5 else ("⚠️" if row['배율'] > 1.2 else "")
            print(f"{row['컬럼']:<50s} {row['1700+직전']:>12.1f} {row['전체평균']:>12.1f} {row['배율']:>8.2f}x {row['상관']:>7.3f} {marker}")

# ========================================
# 7. 제외 권장 컬럼
# ========================================
print(f"\n" + "="*100)
print("❌ 제외 권장 컬럼")
print("="*100)

# 상수 컬럼
print("\n【상수 컬럼】 (변동 없음)")
constant_cols = stats_df[stats_df['고유값'] == 1]
if len(constant_cols) > 0:
    for _, row in constant_cols.iterrows():
        print(f"  • {row['컬럼명']:<50s} (값: {row['평균']:.1f})")
else:
    print("  없음")

# 중복 컬럼
print("\n【중복 컬럼】")
if 'CURRTIME' in df.columns and 'CURRTIME.1' in df.columns:
    if (df['CURRTIME'] == df['CURRTIME.1']).all():
        print("  • CURRTIME.1 (CURRTIME과 100% 동일)")
if 'TIME' in df.columns:
    print("  • TIME (CURRTIME과 유사)")

# 저변동/저상관
print("\n【저변동/저상관 컬럼】 (변동계수<0.1 AND 상관<0.1)")
low_cols = stats_df[(stats_df['변동계수'] < 0.1) & (abs(stats_df['TOTALCNT상관']) < 0.1)]
low_cols = low_cols[~low_cols['컬럼명'].isin(['CURRTIME', 'CURRTIME.1', 'TIME', 'TOTALCNT'])]
if len(low_cols) > 0:
    for _, row in low_cols.iterrows():
        print(f"  • {row['컬럼명']}")
else:
    print("  없음")

# ========================================
# 8. 추천 컬럼 세트
# ========================================
print(f"\n" + "="*100)
print("✅ 추천 컬럼 세트")
print("="*100)

# 상관관계 0.15 이상
high_corr = stats_df[abs(stats_df['TOTALCNT상관']) >= 0.15]
high_corr = high_corr[~high_corr['컬럼명'].isin(['CURRTIME.1', 'TIME', 'TOTALCNT'])]

print(f"\n【높은 상관관계 컬럼】 (|상관| ≥ 0.15) - {len(high_corr)}개")
for idx, (_, row) in enumerate(high_corr.iterrows(), 1):
    print(f"  {idx:2d}. {row['컬럼명']:<50s} 상관:{row['TOTALCNT상관']:>+7.3f}")

# ========================================
# 9. CSV 저장
# ========================================
output_file = "컬럼분석결과.csv"
stats_df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n💾 결과 저장: {output_file}")

# ========================================
# 10. 최종 요약
# ========================================
print(f"\n" + "="*100)
print("📊 최종 요약")
print("="*100)
print(f"총 컬럼 수: {len(df.columns)}개")
print(f"숫자형 컬럼: {len(stats_df)}개")
print(f"상수 컬럼: {len(constant_cols)}개")
print(f"높은 상관 컬럼 (|r|≥0.15): {len(high_corr)}개")
if 'TOTALCNT' in df.columns:
    print(f"1700+ 구간: {(df['TOTALCNT']>=1700).sum():,}개")
print(f"\n✅ 분석 완료!")
print("="*100)