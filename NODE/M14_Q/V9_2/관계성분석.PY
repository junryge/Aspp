#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 AMHS 전체 컬럼 vs TOTALCNT 관계성 분석
- 상관계수 (Pearson, Spearman)
- 기본 통계량
- 결측치/이상치 분석
- 시각화
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("="*80)
print("📊 AMHS 전체 컬럼 vs TOTALCNT 관계성 분석")
print("="*80)

# ========================================
# 1. 데이터 로드
# ========================================
def load_data(file_path):
    """CSV 파일 로드"""
    print(f"\n📂 데이터 로드: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ 파일 없음: {file_path}")
        return None
    
    df = pd.read_csv(file_path, on_bad_lines='skip')
    print(f"✅ {len(df):,}행 × {df.shape[1]}열")
    
    return df

# ========================================
# 2. 기본 정보 분석
# ========================================
def basic_info(df):
    """기본 정보 출력"""
    print("\n" + "="*80)
    print("📋 기본 정보")
    print("="*80)
    
    print(f"\n총 컬럼 수: {df.shape[1]}")
    print(f"총 행 수: {len(df):,}")
    
    # 컬럼 타입별 개수
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    print(f"\n수치형 컬럼: {len(numeric_cols)}개")
    print(f"문자형 컬럼: {len(object_cols)}개")
    
    if 'CURRTIME' in object_cols:
        object_cols.remove('CURRTIME')
    
    return numeric_cols, object_cols

# ========================================
# 3. 결측치 분석
# ========================================
def missing_analysis(df, numeric_cols):
    """결측치 분석"""
    print("\n" + "="*80)
    print("🔍 결측치 분석")
    print("="*80)
    
    missing_info = []
    
    for col in numeric_cols:
        missing_count = df[col].isna().sum()
        missing_pct = missing_count / len(df) * 100
        zero_count = (df[col] == 0).sum()
        zero_pct = zero_count / len(df) * 100
        
        missing_info.append({
            '컬럼': col,
            '결측수': missing_count,
            '결측률(%)': round(missing_pct, 2),
            '0값수': zero_count,
            '0값비율(%)': round(zero_pct, 2)
        })
    
    df_missing = pd.DataFrame(missing_info)
    df_missing = df_missing.sort_values('결측률(%)', ascending=False)
    
    # 결측치 있는 컬럼만 출력
    has_missing = df_missing[df_missing['결측률(%)'] > 0]
    if len(has_missing) > 0:
        print(f"\n⚠️ 결측치 있는 컬럼: {len(has_missing)}개")
        print(has_missing.to_string(index=False))
    else:
        print("\n✅ 결측치 없음!")
    
    # 0값 비율 높은 컬럼
    high_zero = df_missing[df_missing['0값비율(%)'] > 50]
    if len(high_zero) > 0:
        print(f"\n⚠️ 0값 50% 이상 컬럼: {len(high_zero)}개")
        print(high_zero[['컬럼', '0값수', '0값비율(%)']].head(20).to_string(index=False))
    
    return df_missing

# ========================================
# 4. 상관관계 분석 (핵심!)
# ========================================
def correlation_analysis(df, numeric_cols, target='TOTALCNT'):
    """TOTALCNT와 전체 컬럼 상관관계 분석"""
    print("\n" + "="*80)
    print(f"📈 {target}와 전체 컬럼 상관관계 분석")
    print("="*80)
    
    if target not in df.columns:
        print(f"❌ 타겟 컬럼 없음: {target}")
        return None
    
    # 분석할 컬럼 (타겟 제외)
    analysis_cols = [c for c in numeric_cols if c != target]
    
    correlation_results = []
    
    for col in analysis_cols:
        try:
            # NaN 제거
            valid_mask = ~(df[col].isna() | df[target].isna())
            x = df.loc[valid_mask, col]
            y = df.loc[valid_mask, target]
            
            if len(x) < 10:
                continue
            
            # Pearson 상관계수
            pearson_corr, pearson_p = stats.pearsonr(x, y)
            
            # Spearman 상관계수 (비선형 관계 포착)
            spearman_corr, spearman_p = stats.spearmanr(x, y)
            
            # 기본 통계
            col_mean = x.mean()
            col_std = x.std()
            col_min = x.min()
            col_max = x.max()
            
            correlation_results.append({
                '컬럼': col,
                'Pearson': round(pearson_corr, 4),
                'Pearson_P': round(pearson_p, 6),
                'Spearman': round(spearman_corr, 4),
                'Spearman_P': round(spearman_p, 6),
                '평균': round(col_mean, 2),
                '표준편차': round(col_std, 2),
                '최소': round(col_min, 2),
                '최대': round(col_max, 2),
                '유효데이터': len(x)
            })
            
        except Exception as e:
            print(f"  ⚠️ {col}: {e}")
    
    df_corr = pd.DataFrame(correlation_results)
    
    # 절대값 기준 정렬
    df_corr['Pearson_abs'] = df_corr['Pearson'].abs()
    df_corr['Spearman_abs'] = df_corr['Spearman'].abs()
    df_corr = df_corr.sort_values('Pearson_abs', ascending=False)
    
    print(f"\n📊 분석 완료: {len(df_corr)}개 컬럼")
    
    return df_corr

# ========================================
# 5. 상관관계 등급 분류
# ========================================
def classify_correlation(df_corr):
    """상관관계 강도별 분류"""
    print("\n" + "="*80)
    print("🏆 상관관계 강도별 분류")
    print("="*80)
    
    # 등급 분류
    very_strong = df_corr[df_corr['Pearson_abs'] >= 0.8]
    strong = df_corr[(df_corr['Pearson_abs'] >= 0.6) & (df_corr['Pearson_abs'] < 0.8)]
    moderate = df_corr[(df_corr['Pearson_abs'] >= 0.4) & (df_corr['Pearson_abs'] < 0.6)]
    weak = df_corr[(df_corr['Pearson_abs'] >= 0.2) & (df_corr['Pearson_abs'] < 0.4)]
    very_weak = df_corr[df_corr['Pearson_abs'] < 0.2]
    
    print(f"\n🔴 매우 강함 (|r| ≥ 0.8): {len(very_strong)}개")
    if len(very_strong) > 0:
        print(very_strong[['컬럼', 'Pearson', 'Spearman']].to_string(index=False))
    
    print(f"\n🟠 강함 (0.6 ≤ |r| < 0.8): {len(strong)}개")
    if len(strong) > 0:
        print(strong[['컬럼', 'Pearson', 'Spearman']].to_string(index=False))
    
    print(f"\n🟡 중간 (0.4 ≤ |r| < 0.6): {len(moderate)}개")
    if len(moderate) > 0:
        print(moderate[['컬럼', 'Pearson', 'Spearman']].to_string(index=False))
    
    print(f"\n🟢 약함 (0.2 ≤ |r| < 0.4): {len(weak)}개")
    if len(weak) > 0:
        print(weak[['컬럼', 'Pearson', 'Spearman']].head(20).to_string(index=False))
    
    print(f"\n⚪ 매우 약함 (|r| < 0.2): {len(very_weak)}개")
    
    return {
        'very_strong': very_strong,
        'strong': strong,
        'moderate': moderate,
        'weak': weak,
        'very_weak': very_weak
    }

# ========================================
# 6. 데이터 누수 위험 컬럼 탐지
# ========================================
def detect_leakage_risk(df_corr, threshold=0.95):
    """데이터 누수 위험 컬럼 탐지"""
    print("\n" + "="*80)
    print(f"⚠️ 데이터 누수 위험 탐지 (|r| ≥ {threshold})")
    print("="*80)
    
    leakage_risk = df_corr[df_corr['Pearson_abs'] >= threshold]
    
    if len(leakage_risk) > 0:
        print(f"\n🚨 위험 컬럼: {len(leakage_risk)}개")
        print("이 컬럼들은 TOTALCNT와 거의 동일한 정보를 담고 있어 누수 위험!")
        print(leakage_risk[['컬럼', 'Pearson', 'Spearman']].to_string(index=False))
    else:
        print("\n✅ 데이터 누수 위험 컬럼 없음")
    
    return leakage_risk

# ========================================
# 7. 파생변수 후보 탐지
# ========================================
def detect_derived_candidates(df, numeric_cols, target='TOTALCNT'):
    """파생변수 후보 탐지 (차이, 비율 등)"""
    print("\n" + "="*80)
    print("🔧 파생변수 후보 탐지")
    print("="*80)
    
    # CREATED - COMPLETED 패턴 탐지
    created_cols = [c for c in numeric_cols if 'CREATED' in c.upper()]
    completed_cols = [c for c in numeric_cols if 'COMPLETED' in c.upper()]
    
    print(f"\nCREATED 컬럼: {len(created_cols)}개")
    print(f"COMPLETED 컬럼: {len(completed_cols)}개")
    
    derived_results = []
    
    # 짝 찾기
    for cr in created_cols:
        base_name = cr.replace('CREATED', '').replace('CURRENTQ', '')
        for cp in completed_cols:
            if base_name in cp or cp.replace('COMPLETED', '').replace('CURRENTQ', '') == base_name:
                try:
                    gap = df[cr] - df[cp]
                    valid_mask = ~(gap.isna() | df[target].isna())
                    
                    if valid_mask.sum() > 10:
                        corr, _ = stats.pearsonr(gap[valid_mask], df.loc[valid_mask, target])
                        
                        derived_results.append({
                            'CREATED': cr,
                            'COMPLETED': cp,
                            'GAP_상관계수': round(corr, 4),
                            'GAP_평균': round(gap.mean(), 2),
                            'GAP_최대': round(gap.max(), 2)
                        })
                except:
                    pass
    
    if derived_results:
        df_derived = pd.DataFrame(derived_results)
        df_derived = df_derived.sort_values('GAP_상관계수', ascending=False)
        print("\n📊 GAP(CREATED-COMPLETED) 상관관계:")
        print(df_derived.to_string(index=False))
        return df_derived
    
    return None

# ========================================
# 8. 시각화
# ========================================
def visualize_correlations(df_corr, output_dir='correlation_analysis'):
    """상관관계 시각화"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("📊 시각화 생성 중...")
    print("="*80)
    
    # 1. TOP 30 상관계수 바 차트
    fig, ax = plt.subplots(figsize=(14, 10))
    top30 = df_corr.head(30)
    colors = ['red' if x >= 0.8 else 'orange' if x >= 0.6 else 'yellow' if x >= 0.4 else 'green' 
              for x in top30['Pearson_abs']]
    
    bars = ax.barh(range(len(top30)), top30['Pearson'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(top30)))
    ax.set_yticklabels(top30['컬럼'], fontsize=8)
    ax.set_xlabel('Pearson Correlation')
    ax.set_title('TOP 30 Correlation with TOTALCNT')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.axvline(x=0.8, color='red', linestyle='--', linewidth=0.5, label='Very Strong (0.8)')
    ax.axvline(x=0.6, color='orange', linestyle='--', linewidth=0.5, label='Strong (0.6)')
    ax.axvline(x=-0.8, color='red', linestyle='--', linewidth=0.5)
    ax.axvline(x=-0.6, color='orange', linestyle='--', linewidth=0.5)
    ax.legend(loc='lower right')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_top30.png', dpi=150)
    plt.close()
    print(f"  ✅ {output_dir}/correlation_top30.png")
    
    # 2. 상관계수 분포 히스토그램
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(df_corr['Pearson'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].axvline(x=0.8, color='red', linestyle='--', label='±0.8')
    axes[0].axvline(x=-0.8, color='red', linestyle='--')
    axes[0].axvline(x=0.6, color='orange', linestyle='--', label='±0.6')
    axes[0].axvline(x=-0.6, color='orange', linestyle='--')
    axes[0].set_xlabel('Pearson Correlation')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Pearson Correlation Distribution')
    axes[0].legend()
    
    axes[1].hist(df_corr['Spearman'], bins=50, color='darkgreen', alpha=0.7, edgecolor='black')
    axes[1].axvline(x=0.8, color='red', linestyle='--', label='±0.8')
    axes[1].axvline(x=-0.8, color='red', linestyle='--')
    axes[1].axvline(x=0.6, color='orange', linestyle='--', label='±0.6')
    axes[1].axvline(x=-0.6, color='orange', linestyle='--')
    axes[1].set_xlabel('Spearman Correlation')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Spearman Correlation Distribution')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_distribution.png', dpi=150)
    plt.close()
    print(f"  ✅ {output_dir}/correlation_distribution.png")
    
    # 3. Pearson vs Spearman 비교
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(df_corr['Pearson'], df_corr['Spearman'], alpha=0.5, c='steelblue')
    ax.plot([-1, 1], [-1, 1], 'r--', label='y=x')
    ax.set_xlabel('Pearson Correlation')
    ax.set_ylabel('Spearman Correlation')
    ax.set_title('Pearson vs Spearman Correlation')
    ax.legend()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pearson_vs_spearman.png', dpi=150)
    plt.close()
    print(f"  ✅ {output_dir}/pearson_vs_spearman.png")

# ========================================
# 9. 상위 컬럼 산점도
# ========================================
def plot_scatter_top_columns(df, df_corr, target='TOTALCNT', top_n=12, output_dir='correlation_analysis'):
    """상위 상관관계 컬럼 산점도"""
    os.makedirs(output_dir, exist_ok=True)
    
    top_cols = df_corr.head(top_n)['컬럼'].tolist()
    
    n_cols = 4
    n_rows = (len(top_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten()
    
    for idx, col in enumerate(top_cols):
        ax = axes[idx]
        valid_mask = ~(df[col].isna() | df[target].isna())
        
        # 샘플링 (너무 많으면)
        if valid_mask.sum() > 5000:
            sample_idx = df[valid_mask].sample(5000, random_state=42).index
            x = df.loc[sample_idx, col]
            y = df.loc[sample_idx, target]
        else:
            x = df.loc[valid_mask, col]
            y = df.loc[valid_mask, target]
        
        ax.scatter(x, y, alpha=0.3, s=5, c='steelblue')
        ax.axhline(y=1700, color='red', linestyle='--', alpha=0.7, label='1700')
        ax.axhline(y=1600, color='orange', linestyle='--', alpha=0.5, label='1600')
        
        corr_val = df_corr[df_corr['컬럼'] == col]['Pearson'].values[0]
        ax.set_title(f'{col[:30]}\nr={corr_val:.3f}', fontsize=9)
        ax.set_xlabel(col[:20], fontsize=8)
        ax.set_ylabel(target, fontsize=8)
    
    # 빈 subplot 제거
    for idx in range(len(top_cols), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/scatter_top{top_n}.png', dpi=150)
    plt.close()
    print(f"  ✅ {output_dir}/scatter_top{top_n}.png")

# ========================================
# 10. 결과 저장
# ========================================
def save_results(df_corr, df_missing, output_dir='correlation_analysis'):
    """결과 CSV 저장"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 상관관계 결과
    df_corr.to_csv(f'{output_dir}/correlation_all_columns.csv', index=False, encoding='utf-8-sig')
    print(f"  ✅ {output_dir}/correlation_all_columns.csv")
    
    # 결측치 결과
    df_missing.to_csv(f'{output_dir}/missing_analysis.csv', index=False, encoding='utf-8-sig')
    print(f"  ✅ {output_dir}/missing_analysis.csv")
    
    # 요약 리포트
    summary = df_corr[['컬럼', 'Pearson', 'Spearman', '평균', '표준편차', '최소', '최대']].copy()
    summary.to_csv(f'{output_dir}/correlation_summary.csv', index=False, encoding='utf-8-sig')
    print(f"  ✅ {output_dir}/correlation_summary.csv")

# ========================================
# 11. HTML 리포트 생성
# ========================================
def generate_html_report(df_corr, classified, output_dir='correlation_analysis'):
    """HTML 리포트 생성"""
    os.makedirs(output_dir, exist_ok=True)
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>AMHS Column Correlation Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .summary {{ display: flex; gap: 20px; flex-wrap: wrap; margin: 20px 0; }}
        .card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; min-width: 150px; text-align: center; }}
        .card.danger {{ background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); }}
        .card.warning {{ background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%); }}
        .card.success {{ background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%); }}
        .card h3 {{ margin: 0; font-size: 2em; }}
        .card p {{ margin: 5px 0 0 0; opacity: 0.9; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
        th {{ background: #3498db; color: white; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        tr:hover {{ background: #e8f4f8; }}
        .high {{ color: #e74c3c; font-weight: bold; }}
        .medium {{ color: #f39c12; font-weight: bold; }}
        .low {{ color: #27ae60; }}
        img {{ max-width: 100%; height: auto; margin: 10px 0; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.2); }}
        .img-container {{ display: flex; gap: 20px; flex-wrap: wrap; justify-content: center; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 AMHS Column Correlation Analysis Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="summary">
            <div class="card danger">
                <h3>{len(classified['very_strong'])}</h3>
                <p>Very Strong (≥0.8)</p>
            </div>
            <div class="card warning">
                <h3>{len(classified['strong'])}</h3>
                <p>Strong (0.6-0.8)</p>
            </div>
            <div class="card">
                <h3>{len(classified['moderate'])}</h3>
                <p>Moderate (0.4-0.6)</p>
            </div>
            <div class="card success">
                <h3>{len(classified['weak']) + len(classified['very_weak'])}</h3>
                <p>Weak (<0.4)</p>
            </div>
        </div>
        
        <h2>🔴 Very Strong Correlation (|r| ≥ 0.8)</h2>
        <p>⚠️ These columns may cause data leakage!</p>
        <table>
            <tr><th>Column</th><th>Pearson</th><th>Spearman</th><th>Mean</th><th>Std</th></tr>
"""
    
    for _, row in classified['very_strong'].iterrows():
        html += f"<tr><td>{row['컬럼']}</td><td class='high'>{row['Pearson']:.4f}</td><td>{row['Spearman']:.4f}</td><td>{row['평균']:.2f}</td><td>{row['표준편차']:.2f}</td></tr>\n"
    
    html += """
        </table>
        
        <h2>🟠 Strong Correlation (0.6 ≤ |r| < 0.8)</h2>
        <table>
            <tr><th>Column</th><th>Pearson</th><th>Spearman</th><th>Mean</th><th>Std</th></tr>
"""
    
    for _, row in classified['strong'].iterrows():
        html += f"<tr><td>{row['컬럼']}</td><td class='medium'>{row['Pearson']:.4f}</td><td>{row['Spearman']:.4f}</td><td>{row['평균']:.2f}</td><td>{row['표준편차']:.2f}</td></tr>\n"
    
    html += """
        </table>
        
        <h2>🟡 Moderate Correlation (0.4 ≤ |r| < 0.6)</h2>
        <table>
            <tr><th>Column</th><th>Pearson</th><th>Spearman</th><th>Mean</th><th>Std</th></tr>
"""
    
    for _, row in classified['moderate'].iterrows():
        html += f"<tr><td>{row['컬럼']}</td><td>{row['Pearson']:.4f}</td><td>{row['Spearman']:.4f}</td><td>{row['평균']:.2f}</td><td>{row['표준편차']:.2f}</td></tr>\n"
    
    html += """
        </table>
        
        <h2>📊 TOP 30 Correlation Chart</h2>
        <div class="img-container">
            <img src="correlation_top30.png" alt="Top 30 Correlation">
        </div>
        
        <h2>📈 Correlation Distribution</h2>
        <div class="img-container">
            <img src="correlation_distribution.png" alt="Correlation Distribution">
        </div>
        
        <h2>🔗 Pearson vs Spearman</h2>
        <div class="img-container">
            <img src="pearson_vs_spearman.png" alt="Pearson vs Spearman">
        </div>
        
        <h2>📉 Top Columns Scatter Plots</h2>
        <div class="img-container">
            <img src="scatter_top12.png" alt="Scatter Plots">
        </div>
        
        <h2>📋 All Columns (Sorted by |Pearson|)</h2>
        <table>
            <tr><th>#</th><th>Column</th><th>Pearson</th><th>Spearman</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th></tr>
"""
    
    for idx, (_, row) in enumerate(df_corr.iterrows(), 1):
        corr_class = 'high' if row['Pearson_abs'] >= 0.8 else 'medium' if row['Pearson_abs'] >= 0.6 else 'low'
        html += f"<tr><td>{idx}</td><td>{row['컬럼']}</td><td class='{corr_class}'>{row['Pearson']:.4f}</td><td>{row['Spearman']:.4f}</td><td>{row['평균']:.2f}</td><td>{row['표준편차']:.2f}</td><td>{row['최소']:.2f}</td><td>{row['최대']:.2f}</td></tr>\n"
    
    html += """
        </table>
    </div>
</body>
</html>
"""
    
    with open(f'{output_dir}/correlation_report.html', 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"  ✅ {output_dir}/correlation_report.html")

# ========================================
# 메인 실행
# ========================================
def main(csv_file):
    """메인 분석 실행"""
    
    # 출력 디렉토리
    output_dir = 'correlation_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 데이터 로드
    df = load_data(csv_file)
    if df is None:
        return
    
    # 2. 기본 정보
    numeric_cols, object_cols = basic_info(df)
    
    # 3. 결측치 분석
    df_missing = missing_analysis(df, numeric_cols)
    
    # 4. 상관관계 분석
    df_corr = correlation_analysis(df, numeric_cols, target='TOTALCNT')
    if df_corr is None:
        return
    
    # 5. 등급 분류
    classified = classify_correlation(df_corr)
    
    # 6. 데이터 누수 탐지
    leakage = detect_leakage_risk(df_corr)
    
    # 7. 파생변수 후보
    derived = detect_derived_candidates(df, numeric_cols)
    
    # 8. 시각화
    visualize_correlations(df_corr, output_dir)
    
    # 9. 산점도
    plot_scatter_top_columns(df, df_corr, top_n=12, output_dir=output_dir)
    
    # 10. 결과 저장
    print("\n" + "="*80)
    print("💾 결과 저장")
    print("="*80)
    save_results(df_corr, df_missing, output_dir)
    
    # 11. HTML 리포트
    generate_html_report(df_corr, classified, output_dir)
    
    print("\n" + "="*80)
    print("✅ 분석 완료!")
    print("="*80)
    print(f"\n📁 결과 폴더: {output_dir}/")
    print(f"   - correlation_report.html (메인 리포트)")
    print(f"   - correlation_all_columns.csv (전체 상관관계)")
    print(f"   - correlation_summary.csv (요약)")
    print(f"   - missing_analysis.csv (결측치)")
    print(f"   - *.png (차트들)")
    
    return df_corr, classified

if __name__ == '__main__':
    import sys
    
    # 파일 경로 지정 (수정 필요!)
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        # 기본 파일 경로들 시도
        candidates = [
            'data/M14_Q_20250802_TO_20251015.csv',
            'data/2025_DATA.CSV',
            'M14_Q_20250802_TO_20251015.csv',
            '2025_DATA.CSV'
        ]
        csv_file = None
        for c in candidates:
            if os.path.exists(c):
                csv_file = c
                break
        
        if csv_file is None:
            print("❌ CSV 파일을 찾을 수 없습니다!")
            print("\n사용법: python column_correlation_analysis.py [CSV파일경로]")
            print("\n예시:")
            print("  python column_correlation_analysis.py data/mydata.csv")
            exit(1)
    
    main(csv_file)