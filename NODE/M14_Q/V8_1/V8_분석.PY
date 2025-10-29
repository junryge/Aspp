# -*- coding: utf-8 -*-
"""
1년치 CSV 데이터로 1550+ → 10분 후 1700+ 증가 패턴 분석
메모리 최적화 + 상세 통계 + 시각화
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import gc
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_large_csv(filepath):
    """대용량 CSV 로드 (메모리 최적화)"""
    print("\n" + "="*80)
    print("📂 데이터 로딩 중...")
    print("="*80)
    
    # 먼저 샘플로 컬럼 확인
    sample_df = pd.read_csv(filepath, nrows=5)
    print(f"컬럼: {list(sample_df.columns)}")
    
    # 청크 단위로 읽기
    chunk_size = 100000
    chunks = []
    
    for i, chunk in enumerate(pd.read_csv(filepath, chunksize=chunk_size)):
        chunks.append(chunk)
        if (i + 1) % 5 == 0:
            print(f"  {(i+1)*chunk_size:,}개 로딩...")
    
    df = pd.concat(chunks, ignore_index=True)
    print(f"✅ 로딩 완료: {len(df):,}개 행")
    
    # CURRTIME 변환
    if 'CURRTIME' in df.columns:
        df['CURRTIME'] = pd.to_datetime(df['CURRTIME'], format='%Y%m%d%H%M', errors='coerce')
        print(f"기간: {df['CURRTIME'].min()} ~ {df['CURRTIME'].max()}")
    
    return df

def analyze_1550_to_1700_pattern(df):
    """1550+ → 1700+ 패턴 분석"""
    
    print("\n" + "="*80)
    print("📊 1550+ → 10분 후 1700+ 증가 패턴 분석")
    print("="*80)
    print(f"전체 데이터: {len(df):,}개")
    
    # ===== 1. 1550 이상 구간 찾기 =====
    print("\n" + "="*80)
    print("1️⃣ TOTALCNT 1550+ 구간 분석")
    print("="*80)
    
    mask_1550 = df['TOTALCNT'] >= 1550
    count_1550 = mask_1550.sum()
    
    print(f"1550+ 구간: {count_1550:,}개 ({count_1550/len(df)*100:.2f}%)")
    
    # 분포 확인
    print(f"\nTOTALCNT 분포:")
    print(f"  최소값: {df['TOTALCNT'].min()}")
    print(f"  평균값: {df['TOTALCNT'].mean():.2f}")
    print(f"  중앙값: {df['TOTALCNT'].median():.2f}")
    print(f"  최대값: {df['TOTALCNT'].max()}")
    print(f"  1400+: {(df['TOTALCNT'] >= 1400).sum():,}개")
    print(f"  1500+: {(df['TOTALCNT'] >= 1500).sum():,}개")
    print(f"  1550+: {count_1550:,}개")
    print(f"  1600+: {(df['TOTALCNT'] >= 1600).sum():,}개")
    print(f"  1700+: {(df['TOTALCNT'] >= 1700).sum():,}개")
    
    if count_1550 < 100:
        print(f"\n⚠️ 1550+ 데이터가 {count_1550}개로 적습니다.")
        threshold = 1500
        print(f"🔄 임계값을 {threshold}으로 조정하여 분석합니다.")
    else:
        threshold = 1550
    
    # ===== 2. 10분 후 비교 데이터 생성 =====
    print("\n" + "="*80)
    print("2️⃣ 10분 후 1700+ 도달 여부 분석")
    print("="*80)
    
    results = []
    
    # 인덱스 생성
    valid_indices = df[df['TOTALCNT'] >= threshold].index
    total_valid = len(valid_indices)
    
    print(f"분석 대상: {total_valid:,}개")
    print("10분 후 비교 데이터 생성 중...")
    
    for count, idx in enumerate(valid_indices):
        if idx + 10 >= len(df):
            continue
        
        # 진행상황 출력
        if count % 10000 == 0 and count > 0:
            print(f"  {count:,}/{total_valid:,} ({count/total_valid*100:.1f}%)")
        
        current_totalcnt = df.loc[idx, 'TOTALCNT']
        future_totalcnt = df.loc[idx + 10, 'TOTALCNT']
        
        # 증가 여부
        increase_to_1700 = (future_totalcnt >= 1700)
        increase_amount = future_totalcnt - current_totalcnt
        
        # 현재 시점 컬럼값
        result = {
            'index': idx,
            'currtime': df.loc[idx, 'CURRTIME'] if 'CURRTIME' in df.columns else idx,
            'current_totalcnt': current_totalcnt,
            'future_totalcnt': future_totalcnt,
            'increase': increase_amount,
            'reach_1700': increase_to_1700,
            'M14AM10A': df.loc[idx, 'M14AM10A'],
            'M10AM14A': df.loc[idx, 'M10AM14A'],
            'M14AM10ASUM': df.loc[idx, 'M14AM10ASUM'],
            'M14AM14B': df.loc[idx, 'M14AM14B'],
            'M14BM14A': df.loc[idx, 'M14BM14A'],
            'M14AM14BSUM': df.loc[idx, 'M14AM14BSUM'],
            'M14AM16': df.loc[idx, 'M14AM16'],
            'M16M14A': df.loc[idx, 'M16M14A'],
            'M14AM16SUM': df.loc[idx, 'M14AM16SUM']
        }
        
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    # 메모리 정리
    del valid_indices
    gc.collect()
    
    # 통계
    total_cases = len(results_df)
    reach_1700_cases = results_df['reach_1700'].sum()
    
    print(f"\n✅ 분석 완료!")
    print(f"분석 케이스: {total_cases:,}개")
    print(f"10분 후 1700+ 도달: {reach_1700_cases:,}개 ({reach_1700_cases/total_cases*100:.2f}%)")
    print(f"10분 후 1700 미달: {total_cases - reach_1700_cases:,}개 ({(total_cases-reach_1700_cases)/total_cases*100:.2f}%)")
    
    # ===== 3. 그룹 비교 =====
    print("\n" + "="*80)
    print("3️⃣ 1700+ 도달 vs 미달 그룹 비교")
    print("="*80)
    
    group_reach = results_df[results_df['reach_1700'] == True]
    group_not_reach = results_df[results_df['reach_1700'] == False]
    
    print(f"\n【1700+ 도달 그룹】 {len(group_reach):,}개")
    print(f"【1700 미달 그룹】 {len(group_not_reach):,}개")
    
    # 컬럼별 평균 비교
    comparison_cols = ['M14AM10A', 'M10AM14A', 'M14AM10ASUM', 
                       'M14AM14B', 'M14BM14A', 'M14AM14BSUM',
                       'M14AM16', 'M16M14A', 'M14AM16SUM']
    
    print("\n" + "-"*80)
    print(f"{'컬럼':<15} {'도달 평균':>12} {'미달 평균':>12} {'차이':>12} {'차이율':>10}")
    print("-"*80)
    
    comparison_results = []
    
    for col in comparison_cols:
        if len(group_reach) > 0 and len(group_not_reach) > 0:
            reach_mean = group_reach[col].mean()
            not_reach_mean = group_not_reach[col].mean()
            diff = reach_mean - not_reach_mean
            diff_rate = (diff / not_reach_mean * 100) if not_reach_mean != 0 else 0
            
            print(f"{col:<15} {reach_mean:>12.2f} {not_reach_mean:>12.2f} {diff:>12.2f} {diff_rate:>9.2f}%")
            
            comparison_results.append({
                'column': col,
                'reach_mean': reach_mean,
                'reach_std': group_reach[col].std(),
                'reach_median': group_reach[col].median(),
                'not_reach_mean': not_reach_mean,
                'not_reach_std': group_not_reach[col].std(),
                'not_reach_median': group_not_reach[col].median(),
                'diff': diff,
                'diff_rate': diff_rate
            })
    
    comparison_df = pd.DataFrame(comparison_results)
    
    # ===== 4. 중요 패턴 발견 =====
    print("\n" + "="*80)
    print("4️⃣ 🔥 증가 예측 핵심 패턴")
    print("="*80)
    
    # 차이율 기준 정렬
    comparison_df_sorted = comparison_df.sort_values('diff_rate', ascending=False)
    
    print("\n【가장 큰 차이를 보이는 컬럼 TOP 5】")
    for idx, row in comparison_df_sorted.head(5).iterrows():
        direction = "높음" if row['diff_rate'] > 0 else "낮음"
        print(f"  {idx+1}. {row['column']}: 도달그룹이 {abs(row['diff_rate']):.1f}% 더 {direction}")
        print(f"     도달: {row['reach_mean']:.2f} (±{row['reach_std']:.2f}) vs 미달: {row['not_reach_mean']:.2f} (±{row['not_reach_std']:.2f})")
    
    # ===== 5. 임계값 분석 =====
    print("\n" + "="*80)
    print("5️⃣ 임계값 기반 예측 규칙")
    print("="*80)
    
    if len(group_reach) > 0:
        print("\n【1700+ 도달 그룹의 컬럼 분포】")
        
        for col in comparison_cols:
            reach_values = group_reach[col]
            
            percentile_10 = reach_values.quantile(0.10)
            percentile_25 = reach_values.quantile(0.25)
            percentile_50 = reach_values.median()
            percentile_75 = reach_values.quantile(0.75)
            percentile_90 = reach_values.quantile(0.90)
            
            print(f"\n{col}:")
            print(f"  최소: {reach_values.min():.2f}")
            print(f"  10%: {percentile_10:.2f}")
            print(f"  25%: {percentile_25:.2f}")
            print(f"  중앙: {percentile_50:.2f}")
            print(f"  75%: {percentile_75:.2f}")
            print(f"  90%: {percentile_90:.2f}")
            print(f"  최대: {reach_values.max():.2f}")
    
    # ===== 6. 예측 규칙 생성 =====
    print("\n" + "="*80)
    print("6️⃣ 💡 예측 규칙 제안")
    print("="*80)
    
    if len(group_reach) > 0:
        # 도달 그룹의 중앙값을 기준으로 규칙 생성
        rules = []
        
        for col in comparison_cols:
            reach_median = group_reach[col].median()
            not_reach_median = group_not_reach[col].median() if len(group_not_reach) > 0 else 0
            
            if reach_median > not_reach_median:
                rules.append({
                    'column': col,
                    'threshold': reach_median,
                    'direction': '>=',
                    'diff': reach_median - not_reach_median
                })
            elif reach_median < not_reach_median:
                rules.append({
                    'column': col,
                    'threshold': reach_median,
                    'direction': '<=',
                    'diff': not_reach_median - reach_median
                })
        
        # 차이가 큰 순서로 정렬
        rules_sorted = sorted(rules, key=lambda x: abs(x['diff']), reverse=True)
        
        print(f"\n【제안 규칙】 현재 {threshold}+ 일 때, 10분 후 1700+ 예측:")
        print("\nIF (다음 조건 중 3개 이상 만족):")
        
        for i, rule in enumerate(rules_sorted[:6], 1):
            print(f"  {i}. {rule['column']} {rule['direction']} {rule['threshold']:.2f}")
        
        print("\nTHEN: 10분 후 1700+ 도달 가능성 높음")
        
        # ===== 7. 규칙 검증 =====
        print("\n" + "="*80)
        print("7️⃣ 규칙 검증")
        print("="*80)
        
        if len(rules_sorted) >= 3:
            # 상위 5개 규칙 적용
            top_rules = rules_sorted[:5]
            
            # 규칙 적용
            conditions = []
            for rule in top_rules:
                col = rule['column']
                threshold_val = rule['threshold']
                direction = rule['direction']
                
                if direction == '>=':
                    condition = results_df[col] >= threshold_val
                else:
                    condition = results_df[col] <= threshold_val
                
                conditions.append(condition)
            
            # 3개 이상 만족
            rule_match_3 = sum(conditions) >= 3
            
            # 검증
            true_positive = (rule_match_3 & results_df['reach_1700']).sum()
            false_positive = (rule_match_3 & ~results_df['reach_1700']).sum()
            true_negative = (~rule_match_3 & ~results_df['reach_1700']).sum()
            false_negative = (~rule_match_3 & results_df['reach_1700']).sum()
            
            total = len(results_df)
            
            print(f"\n【3개 이상 규칙 만족 시】")
            print(f"  True Positive (규칙O, 실제O): {true_positive:,}개")
            print(f"  False Positive (규칙O, 실제X): {false_positive:,}개")
            print(f"  True Negative (규칙X, 실제X): {true_negative:,}개")
            print(f"  False Negative (규칙X, 실제O): {false_negative:,}개")
            
            if true_positive + false_positive > 0:
                precision = true_positive / (true_positive + false_positive)
                print(f"\n  정확도 (Precision): {precision*100:.2f}%")
            
            if true_positive + false_negative > 0:
                recall = true_positive / (true_positive + false_negative)
                print(f"  재현율 (Recall): {recall*100:.2f}%")
            
            if total > 0:
                accuracy = (true_positive + true_negative) / total
                print(f"  전체 정확도: {accuracy*100:.2f}%")
            
            if precision > 0 and recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
                print(f"  F1 Score: {f1*100:.2f}%")
    
    # ===== 8. 증가량 분석 =====
    print("\n" + "="*80)
    print("8️⃣ 증가량 상세 분석")
    print("="*80)
    
    print(f"\n증가량 통계:")
    print(f"  평균 증가: {results_df['increase'].mean():.2f}")
    print(f"  중앙 증가: {results_df['increase'].median():.2f}")
    print(f"  최대 증가: {results_df['increase'].max():.2f}")
    print(f"  최소 증가: {results_df['increase'].min():.2f}")
    
    print(f"\n도달 그룹 증가량:")
    if len(group_reach) > 0:
        print(f"  평균: {group_reach['increase'].mean():.2f}")
        print(f"  중앙: {group_reach['increase'].median():.2f}")
        print(f"  최대: {group_reach['increase'].max():.2f}")
    
    print(f"\n미달 그룹 증가량:")
    if len(group_not_reach) > 0:
        print(f"  평균: {group_not_reach['increase'].mean():.2f}")
        print(f"  중앙: {group_not_reach['increase'].median():.2f}")
        print(f"  최소: {group_not_reach['increase'].min():.2f}")
    
    # ===== 9. CSV 저장 =====
    print("\n" + "="*80)
    print("9️⃣ 결과 저장")
    print("="*80)
    
    # 분석 결과
    output_csv = 'pattern_analysis_results_1year.csv'
    results_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"✅ 분석 결과: {output_csv}")
    
    # 비교 통계
    comparison_output = 'pattern_comparison_stats_1year.csv'
    comparison_df.to_csv(comparison_output, index=False, encoding='utf-8-sig')
    print(f"✅ 비교 통계: {comparison_output}")
    
    # ===== 10. 시각화 =====
    print("\n" + "="*80)
    print("🔟 그래프 생성 중...")
    print("="*80)
    
    create_analysis_graphs(results_df, comparison_df_sorted, group_reach, group_not_reach, threshold)
    
    return results_df, comparison_df

def create_analysis_graphs(results_df, comparison_df, group_reach, group_not_reach, threshold):
    """분석 그래프 생성"""
    
    fig = plt.figure(figsize=(20, 14))
    
    # 1. 컬럼별 평균 비교
    ax1 = plt.subplot(3, 3, 1)
    x = range(len(comparison_df))
    reach_means = comparison_df['reach_mean']
    not_reach_means = comparison_df['not_reach_mean']
    
    width = 0.35
    ax1.bar([i - width/2 for i in x], reach_means, width, label='1700+ Reach', alpha=0.7, color='#2E86AB')
    ax1.bar([i + width/2 for i in x], not_reach_means, width, label='1700 Not Reach', alpha=0.7, color='#F18F01')
    
    ax1.set_xlabel('Columns', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Average Value', fontsize=11, fontweight='bold')
    ax1.set_title('1) Column Averages Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(comparison_df['column'], rotation=45, ha='right', fontsize=9)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. 차이율 가로 막대
    ax2 = plt.subplot(3, 3, 2)
    colors = ['#2E86AB' if v > 0 else '#F18F01' for v in comparison_df['diff_rate']]
    ax2.barh(range(len(comparison_df)), comparison_df['diff_rate'], 
            color=colors, alpha=0.7, edgecolor='black')
    ax2.set_yticks(range(len(comparison_df)))
    ax2.set_yticklabels(comparison_df['column'], fontsize=9)
    ax2.set_xlabel('Difference Rate (%)', fontsize=11, fontweight='bold')
    ax2.set_title('2) Difference Rate by Column', fontsize=12, fontweight='bold')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. 증가량 분포
    ax3 = plt.subplot(3, 3, 3)
    ax3.hist(results_df['increase'], bins=50, alpha=0.7, color='#A23B72', edgecolor='black')
    ax3.axvline(x=results_df['increase'].mean(), color='red', linestyle='--', linewidth=2, 
              label=f'Mean: {results_df["increase"].mean():.1f}')
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax3.set_xlabel('Increase Amount (10min)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax3.set_title('3) TOTALCNT Increase Distribution', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 현재값 vs 10분 후
    ax4 = plt.subplot(3, 3, 4)
    if len(group_reach) > 0:
        ax4.scatter(group_reach['current_totalcnt'], group_reach['future_totalcnt'], 
                  alpha=0.5, s=20, color='#2E86AB', label='1700+ Reach')
    if len(group_not_reach) > 0:
        ax4.scatter(group_not_reach['current_totalcnt'], group_not_reach['future_totalcnt'], 
                  alpha=0.5, s=20, color='#F18F01', label='1700 Not Reach')
    
    ax4.axhline(y=1700, color='red', linestyle='--', linewidth=2, label='1700 Threshold')
    ax4.plot([threshold, 2000], [threshold, 2000], 'k--', alpha=0.3, label='No Change')
    
    ax4.set_xlabel('Current TOTALCNT', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Future TOTALCNT (10min)', fontsize=11, fontweight='bold')
    ax4.set_title('4) Current vs Future TOTALCNT', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # 5-9. 주요 컬럼 산점도
    key_columns = comparison_df.head(5)['column'].tolist()
    
    for idx, col in enumerate(key_columns):
        row = (idx + 4) // 3
        col_idx = (idx + 4) % 3
        ax = plt.subplot(3, 3, idx + 5)
        
        if len(group_reach) > 0:
            ax.scatter(group_reach[col], group_reach['future_totalcnt'], 
                      alpha=0.5, s=15, color='#2E86AB', label='1700+ Reach')
        
        if len(group_not_reach) > 0:
            ax.scatter(group_not_reach[col], group_not_reach['future_totalcnt'], 
                      alpha=0.5, s=15, color='#F18F01', label='1700 Not Reach')
        
        ax.axhline(y=1700, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        ax.set_xlabel(col, fontsize=10, fontweight='bold')
        ax.set_ylabel('Future TOTALCNT', fontsize=10, fontweight='bold')
        ax.set_title(f'{idx+5}) {col} vs Future', fontsize=11, fontweight='bold')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Pattern Analysis: {threshold}+ → 10min later 1700+ (1 Year Data)', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    output_file = 'pattern_analysis_1550_to_1700_1year.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ 그래프 저장: {output_file}")
    
    plt.close()

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 1년치 데이터로 1550+ → 1700+ 패턴 분석")
    print("="*80)
    
    # CSV 파일 경로 (수정 필요)
    csv_file = 'uu.csv'  # 또는 'uu2.csv'
    
    # 파일 존재 확인
    import os
    if not os.path.exists(csv_file):
        print(f"\n❌ 파일이 없습니다: {csv_file}")
        print("\n사용 가능한 파일:")
        for f in ['uu.csv', 'uu2.csv', 'A.TXT']:
            if os.path.exists(f):
                print(f"  ✅ {f}")
        print("\n코드 상단의 csv_file 변수를 수정하세요.")
        exit(1)
    
    # 데이터 로드
    df = load_large_csv(csv_file)
    
    # 패턴 분석
    results_df, comparison_df = analyze_1550_to_1700_pattern(df)
    
    print("\n" + "="*80)
    print("✅ 전체 분석 완료!")
    print("="*80)
    print("\n생성된 파일:")
    print("  1. pattern_analysis_results_1year.csv - 전체 분석 결과")
    print("  2. pattern_comparison_stats_1year.csv - 컬럼별 비교 통계")
    print("  3. pattern_analysis_1550_to_1700_1year.png - 9개 그래프")
    print("\n" + "="*80)