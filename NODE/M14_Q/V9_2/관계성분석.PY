#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📊 AMHS 전체 컬럼 vs TOTALCNT 관계성 분석 (한글판)
실행: python column_correlation_korean.py 데이터파일.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')
plt.rcParams['axes.unicode_minus'] = False

print("="*80)
print("📊 AMHS 전체 컬럼 vs TOTALCNT 관계성 분석")
print("="*80)

# ========================================
# 1. 데이터 로드
# ========================================
def load_data(file_path):
    print(f"\n📂 데이터 로드: {file_path}")
    if not os.path.exists(file_path):
        print(f"❌ 파일 없음: {file_path}")
        return None
    df = pd.read_csv(file_path, on_bad_lines='skip')
    print(f"✅ {len(df):,}행 × {df.shape[1]}열")
    return df

# ========================================
# 2. 기본 정보
# ========================================
def basic_info(df):
    print("\n" + "="*80)
    print("📋 기본 정보")
    print("="*80)
    print(f"총 컬럼 수: {df.shape[1]}")
    print(f"총 행 수: {len(df):,}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"수치형 컬럼: {len(numeric_cols)}개")
    
    return numeric_cols, object_cols

# ========================================
# 3. 결측치 분석
# ========================================
def missing_analysis(df, numeric_cols):
    print("\n🔍 결측치 분석 중...")
    missing_info = []
    for col in numeric_cols:
        missing_count = df[col].isna().sum()
        missing_pct = missing_count / len(df) * 100
        zero_count = (df[col] == 0).sum()
        zero_pct = zero_count / len(df) * 100
        missing_info.append({
            '컬럼': col, '결측수': missing_count, '결측률': round(missing_pct, 2),
            '0값수': zero_count, '0값비율': round(zero_pct, 2)
        })
    df_missing = pd.DataFrame(missing_info).sort_values('결측률', ascending=False)
    return df_missing

# ========================================
# 4. 상관관계 분석
# ========================================
def correlation_analysis(df, numeric_cols, target='TOTALCNT'):
    print(f"\n📈 {target}와 전체 컬럼 상관관계 분석 중...")
    if target not in df.columns:
        print(f"❌ 타겟 컬럼 없음: {target}")
        return None
    
    analysis_cols = [c for c in numeric_cols if c != target]
    results = []
    
    for col in analysis_cols:
        try:
            valid = ~(df[col].isna() | df[target].isna())
            x, y = df.loc[valid, col], df.loc[valid, target]
            if len(x) < 10: continue
            
            pearson, p_p = stats.pearsonr(x, y)
            spearman, s_p = stats.spearmanr(x, y)
            
            results.append({
                '컬럼': col, 'Pearson': round(pearson, 4), 'Spearman': round(spearman, 4),
                '평균': round(x.mean(), 2), '표준편차': round(x.std(), 2),
                '최소': round(x.min(), 2), '최대': round(x.max(), 2)
            })
        except: pass
    
    df_corr = pd.DataFrame(results)
    df_corr['Pearson_abs'] = df_corr['Pearson'].abs()
    df_corr = df_corr.sort_values('Pearson_abs', ascending=False)
    print(f"✅ 분석 완료: {len(df_corr)}개 컬럼")
    return df_corr

# ========================================
# 5. 등급 분류
# ========================================
def classify_correlation(df_corr):
    return {
        'very_strong': df_corr[df_corr['Pearson_abs'] >= 0.8],
        'strong': df_corr[(df_corr['Pearson_abs'] >= 0.6) & (df_corr['Pearson_abs'] < 0.8)],
        'moderate': df_corr[(df_corr['Pearson_abs'] >= 0.4) & (df_corr['Pearson_abs'] < 0.6)],
        'weak': df_corr[(df_corr['Pearson_abs'] >= 0.2) & (df_corr['Pearson_abs'] < 0.4)],
        'very_weak': df_corr[df_corr['Pearson_abs'] < 0.2]
    }

# ========================================
# 6. 시각화
# ========================================
def visualize_correlations(df_corr, output_dir):
    print("\n📊 그래프 생성 중...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. TOP 30 바 차트
    fig, ax = plt.subplots(figsize=(14, 10))
    top30 = df_corr.head(30)
    colors = ['red' if x >= 0.8 else 'orange' if x >= 0.6 else 'gold' if x >= 0.4 else 'green' 
              for x in top30['Pearson_abs']]
    
    ax.barh(range(len(top30)), top30['Pearson'], color=colors, alpha=0.8)
    ax.set_yticks(range(len(top30)))
    ax.set_yticklabels(top30['컬럼'], fontsize=8)
    ax.set_xlabel('Pearson Correlation', fontsize=12)
    ax.set_title('TOP 30 Correlation with TOTALCNT\n(Red>=0.8, Orange>=0.6, Yellow>=0.4, Green<0.4)', fontsize=14)
    ax.axvline(x=0.8, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(x=0.6, color='orange', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(x=0.4, color='gold', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_top30.png', dpi=150)
    plt.close()
    print(f"  ✅ correlation_top30.png")
    
    # 2. 분포 히스토그램
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df_corr['Pearson'], bins=40, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].axvline(x=0.8, color='red', linestyle='--', label='0.8')
    axes[0].axvline(x=0.6, color='orange', linestyle='--', label='0.6')
    axes[0].axvline(x=0.4, color='gold', linestyle='--', label='0.4')
    axes[0].set_xlabel('Pearson Correlation')
    axes[0].set_title('Pearson Distribution')
    axes[0].legend()
    
    axes[1].hist(df_corr['Spearman'], bins=40, color='darkgreen', alpha=0.7, edgecolor='black')
    axes[1].axvline(x=0.8, color='red', linestyle='--', label='0.8')
    axes[1].axvline(x=0.6, color='orange', linestyle='--', label='0.6')
    axes[1].set_xlabel('Spearman Correlation')
    axes[1].set_title('Spearman Distribution')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_distribution.png', dpi=150)
    plt.close()
    print(f"  ✅ correlation_distribution.png")
    
    # 3. Pearson vs Spearman
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(df_corr['Pearson'], df_corr['Spearman'], alpha=0.5, c='steelblue', s=50)
    ax.plot([-1, 1], [-1, 1], 'r--', label='y=x')
    ax.set_xlabel('Pearson')
    ax.set_ylabel('Spearman')
    ax.set_title('Pearson vs Spearman')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pearson_vs_spearman.png', dpi=150)
    plt.close()
    print(f"  ✅ pearson_vs_spearman.png")

# ========================================
# 7. 산점도
# ========================================
def plot_scatter(df, df_corr, target='TOTALCNT', output_dir='output'):
    print("\n📉 산점도 생성 중...")
    top_cols = df_corr.head(12)['컬럼'].tolist()
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, col in enumerate(top_cols):
        ax = axes[idx]
        valid = ~(df[col].isna() | df[target].isna())
        x, y = df.loc[valid, col], df.loc[valid, target]
        
        if len(x) > 3000:
            sample = np.random.choice(len(x), 3000, replace=False)
            x, y = x.iloc[sample], y.iloc[sample]
        
        ax.scatter(x, y, alpha=0.3, s=5, c='steelblue')
        ax.axhline(y=1700, color='red', linestyle='--', alpha=0.7)
        ax.axhline(y=1600, color='orange', linestyle='--', alpha=0.5)
        
        corr = df_corr[df_corr['컬럼'] == col]['Pearson'].values[0]
        ax.set_title(f'{col[:35]}\nr={corr:.3f}', fontsize=9)
        ax.set_xlabel('')
        ax.set_ylabel('TOTALCNT' if idx % 4 == 0 else '')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/scatter_top12.png', dpi=150)
    plt.close()
    print(f"  ✅ scatter_top12.png")

# ========================================
# 8. HTML 리포트 (한글!)
# ========================================
def generate_html_report(df_corr, classified, output_dir):
    print("\n📝 HTML 리포트 생성 중...")
    
    # 데이터 누수 위험 컬럼
    leakage = df_corr[df_corr['Pearson_abs'] >= 0.95]
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>AMHS 컬럼 상관관계 분석 리포트</title>
    <style>
        body {{ font-family: 'Malgun Gothic', Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 25px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; border-left: 4px solid #3498db; padding-left: 10px; }}
        h3 {{ color: #2980b9; }}
        .summary {{ display: flex; gap: 20px; flex-wrap: wrap; margin: 20px 0; }}
        .card {{ color: white; padding: 20px; border-radius: 10px; min-width: 150px; text-align: center; }}
        .card.danger {{ background: linear-gradient(135deg, #e74c3c, #c0392b); }}
        .card.warning {{ background: linear-gradient(135deg, #f39c12, #e67e22); }}
        .card.info {{ background: linear-gradient(135deg, #3498db, #2980b9); }}
        .card.success {{ background: linear-gradient(135deg, #27ae60, #2ecc71); }}
        .card h3 {{ margin: 0; font-size: 2.2em; color: white; }}
        .card p {{ margin: 5px 0 0 0; opacity: 0.9; font-size: 0.95em; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
        th {{ background: #3498db; color: white; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        tr:hover {{ background: #e8f4f8; }}
        .high {{ color: #e74c3c; font-weight: bold; }}
        .medium {{ color: #f39c12; font-weight: bold; }}
        .low {{ color: #27ae60; }}
        .alert {{ background: #fff3cd; border: 2px solid #ffc107; border-radius: 10px; padding: 15px; margin: 20px 0; }}
        .alert.danger {{ background: #f8d7da; border-color: #f5c6cb; }}
        .alert.info {{ background: #d1ecf1; border-color: #bee5eb; }}
        .alert.success {{ background: #d4edda; border-color: #c3e6cb; }}
        img {{ max-width: 100%; height: auto; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.15); }}
        .img-box {{ background: #f8f9fa; padding: 15px; border-radius: 10px; margin: 15px 0; }}
        .conclusion {{ background: linear-gradient(135deg, #1a1a2e, #16213e); color: white; padding: 25px; border-radius: 15px; margin-top: 30px; }}
        .conclusion h2 {{ color: #00d4ff; border-left: 4px solid #00d4ff; }}
        .conclusion ul {{ line-height: 2; }}
        .tag {{ display: inline-block; padding: 3px 10px; border-radius: 15px; font-size: 0.85em; margin: 2px; }}
        .tag.red {{ background: #e74c3c; color: white; }}
        .tag.green {{ background: #27ae60; color: white; }}
        .tag.blue {{ background: #3498db; color: white; }}
        .tag.yellow {{ background: #f1c40f; color: black; }}
    </style>
</head>
<body>
<div class="container">
    <h1>📊 AMHS 컬럼 vs TOTALCNT 상관관계 분석 리포트</h1>
    <p>생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p><strong>목적:</strong> TOTALCNT(전체 물류량) 예측을 위한 입력 컬럼 선정 및 데이터 누수 탐지</p>
    
    <h2>📋 분석 요약</h2>
    <div class="summary">
        <div class="card danger">
            <h3>{len(classified['very_strong'])}</h3>
            <p>🔴 매우 강함<br>(|r| ≥ 0.8)</p>
        </div>
        <div class="card warning">
            <h3>{len(classified['strong'])}</h3>
            <p>🟠 강함<br>(0.6~0.8)</p>
        </div>
        <div class="card info">
            <h3>{len(classified['moderate'])}</h3>
            <p>🟡 중간<br>(0.4~0.6)</p>
        </div>
        <div class="card success">
            <h3>{len(classified['weak']) + len(classified['very_weak'])}</h3>
            <p>🟢 약함<br>(|r| < 0.4)</p>
        </div>
    </div>
"""
    
    # 데이터 누수 경고
    if len(leakage) > 0:
        html += """
    <div class="alert danger">
        <h3>🚨 데이터 누수 경고!</h3>
        <p>아래 컬럼들은 TOTALCNT와 <strong>95% 이상</strong> 동일한 값입니다. <strong>학습에서 반드시 제외</strong>하세요!</p>
        <ul>
"""
        for _, row in leakage.iterrows():
            html += f"<li><strong>{row['컬럼']}</strong> (r = {row['Pearson']:.4f})</li>\n"
        html += """
        </ul>
        <p>→ 이 컬럼을 넣으면 가짜 100% 정확도가 나옵니다. 실제로는 쓸모없음!</p>
    </div>
"""
    
    # 상관관계 해석 가이드
    html += """
    <div class="alert info">
        <h3>📖 상관계수 해석 가이드</h3>
        <table>
            <tr><th>Pearson 값</th><th>의미</th><th>Feature로 사용?</th></tr>
            <tr><td>0.0 ~ 0.2</td><td>TOTALCNT와 관계 거의 없음</td><td>❌ 쓸모없음</td></tr>
            <tr><td>0.2 ~ 0.4</td><td>약한 관계</td><td>△ 보조용</td></tr>
            <tr><td class="medium">0.4 ~ 0.8</td><td>적당히 강한 관계</td><td>✅ <strong>BEST! 이거 쓰기!</strong></td></tr>
            <tr><td class="high">0.95 이상</td><td>거의 똑같은 값</td><td>❌ 데이터 누수 위험!</td></tr>
        </table>
    </div>
"""
    
    # 매우 강한 상관관계
    html += """
    <h2>🔴 매우 강한 상관관계 (|r| ≥ 0.8)</h2>
    <p>⚠️ 0.95 이상은 데이터 누수 위험! 0.8~0.95는 핵심 Feature 후보</p>
    <table>
        <tr><th>컬럼명</th><th>Pearson</th><th>Spearman</th><th>평균</th><th>표준편차</th><th>판정</th></tr>
"""
    for _, row in classified['very_strong'].iterrows():
        if row['Pearson_abs'] >= 0.95:
            tag = '<span class="tag red">제외 필수</span>'
        else:
            tag = '<span class="tag green">핵심 Feature</span>'
        html += f"<tr><td>{row['컬럼']}</td><td class='high'>{row['Pearson']:.4f}</td><td>{row['Spearman']:.4f}</td><td>{row['평균']:.2f}</td><td>{row['표준편차']:.2f}</td><td>{tag}</td></tr>\n"
    html += "</table>\n"
    
    # 강한 상관관계
    html += """
    <h2>🟠 강한 상관관계 (0.6 ≤ |r| < 0.8)</h2>
    <p>✅ 좋은 Feature 후보들!</p>
    <table>
        <tr><th>컬럼명</th><th>Pearson</th><th>Spearman</th><th>평균</th><th>표준편차</th><th>판정</th></tr>
"""
    for _, row in classified['strong'].iterrows():
        html += f"<tr><td>{row['컬럼']}</td><td class='medium'>{row['Pearson']:.4f}</td><td>{row['Spearman']:.4f}</td><td>{row['평균']:.2f}</td><td>{row['표준편차']:.2f}</td><td><span class='tag green'>추천</span></td></tr>\n"
    html += "</table>\n"
    
    # 중간 상관관계
    html += """
    <h2>🟡 중간 상관관계 (0.4 ≤ |r| < 0.6)</h2>
    <p>✅ 적절한 Feature들 - 현재 V8.3에서 많이 사용 중</p>
    <table>
        <tr><th>컬럼명</th><th>Pearson</th><th>Spearman</th><th>평균</th><th>표준편차</th><th>판정</th></tr>
"""
    for _, row in classified['moderate'].iterrows():
        html += f"<tr><td>{row['컬럼']}</td><td>{row['Pearson']:.4f}</td><td>{row['Spearman']:.4f}</td><td>{row['평균']:.2f}</td><td>{row['표준편차']:.2f}</td><td><span class='tag blue'>검토</span></td></tr>\n"
    html += "</table>\n"
    
    # 그래프 섹션
    html += """
    <h2>📊 TOP 30 상관관계 차트</h2>
    <p>상관계수가 높은 상위 30개 컬럼 (빨강≥0.8, 주황≥0.6, 노랑≥0.4, 초록&lt;0.4)</p>
    <div class="img-box">
        <img src="correlation_top30.png" alt="TOP 30 상관관계">
    </div>
    
    <h2>📈 상관계수 분포</h2>
    <p>전체 컬럼의 Pearson/Spearman 상관계수 분포</p>
    <div class="img-box">
        <img src="correlation_distribution.png" alt="상관계수 분포">
    </div>
    
    <h2>🔗 Pearson vs Spearman 비교</h2>
    <p>두 값이 비슷하면 선형관계, 다르면 비선형관계 가능성</p>
    <div class="img-box">
        <img src="pearson_vs_spearman.png" alt="Pearson vs Spearman">
    </div>
    
    <h2>📉 상위 12개 컬럼 산점도</h2>
    <p>TOTALCNT와 상관관계 높은 컬럼들의 실제 분포 (빨간선=1700, 주황선=1600)</p>
    <div class="img-box">
        <img src="scatter_top12.png" alt="산점도">
    </div>
"""
    
    # 결론 섹션
    best_features = classified['strong']['컬럼'].tolist()[:3] + classified['moderate']['컬럼'].tolist()[:5]
    exclude_features = leakage['컬럼'].tolist() if len(leakage) > 0 else []
    
    html += f"""
    <div class="conclusion">
        <h2>📝 분석 결과 및 권장사항</h2>
        
        <h3>1️⃣ 핵심 발견사항</h3>
        <ul>
            <li>총 <strong>{len(df_corr)}개</strong> 컬럼 분석 완료</li>
            <li>매우 강한 상관관계 (|r|≥0.8): <strong>{len(classified['very_strong'])}개</strong></li>
            <li>적절한 범위 (0.4~0.8): <strong>{len(classified['strong']) + len(classified['moderate'])}개</strong> → Feature 후보!</li>
            <li>데이터 누수 위험 (|r|≥0.95): <strong>{len(leakage)}개</strong> → 제외 필수!</li>
        </ul>
        
        <h3>2️⃣ 제외해야 할 컬럼 (데이터 누수)</h3>
        <ul>
"""
    if exclude_features:
        for col in exclude_features:
            html += f"<li><span class='tag red'>❌ {col}</span></li>\n"
    else:
        html += "<li>없음</li>\n"
    
    html += """
        </ul>
        
        <h3>3️⃣ 추천 Feature (0.4 ~ 0.8 범위)</h3>
        <ul>
"""
    for col in best_features[:8]:
        corr = df_corr[df_corr['컬럼'] == col]['Pearson'].values
        if len(corr) > 0:
            html += f"<li><span class='tag green'>✅ {col}</span> (r={corr[0]:.3f})</li>\n"
    
    html += """
        </ul>
        
        <h3>4️⃣ 상관계수 해석 요약</h3>
        <p style="font-size: 1.1em; line-height: 1.8;">
            • <strong>0.4 ~ 0.8</strong> 범위가 예측 모델에 가장 적합<br>
            • 너무 낮으면 (0.2 이하) → 예측에 도움 안됨<br>
            • 너무 높으면 (0.95 이상) → 데이터 누수, 가짜 성능<br>
            • 파생변수 (예: queue_gap = CREATED - COMPLETED)도 고려하세요!
        </p>
    </div>
"""
    
    # 전체 컬럼 테이블
    html += """
    <h2>📋 전체 컬럼 상관관계 (Pearson 절대값 순)</h2>
    <table>
        <tr><th>#</th><th>컬럼명</th><th>Pearson</th><th>Spearman</th><th>평균</th><th>표준편차</th><th>최소</th><th>최대</th></tr>
"""
    for idx, (_, row) in enumerate(df_corr.iterrows(), 1):
        cls = 'high' if row['Pearson_abs'] >= 0.8 else 'medium' if row['Pearson_abs'] >= 0.6 else 'low'
        html += f"<tr><td>{idx}</td><td>{row['컬럼']}</td><td class='{cls}'>{row['Pearson']:.4f}</td><td>{row['Spearman']:.4f}</td><td>{row['평균']:.2f}</td><td>{row['표준편차']:.2f}</td><td>{row['최소']:.2f}</td><td>{row['최대']:.2f}</td></tr>\n"
    
    html += """
    </table>
</div>
</body>
</html>
"""
    
    with open(f'{output_dir}/correlation_report_korean.html', 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  ✅ correlation_report_korean.html")

# ========================================
# 메인 실행
# ========================================
def main(csv_file):
    output_dir = 'correlation_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 데이터 로드
    df = load_data(csv_file)
    if df is None: return
    
    # 2. 기본 정보
    numeric_cols, _ = basic_info(df)
    
    # 3. 결측치 분석
    df_missing = missing_analysis(df, numeric_cols)
    
    # 4. 상관관계 분석
    df_corr = correlation_analysis(df, numeric_cols)
    if df_corr is None: return
    
    # 5. 등급 분류
    classified = classify_correlation(df_corr)
    
    # 6. 시각화
    visualize_correlations(df_corr, output_dir)
    
    # 7. 산점도
    plot_scatter(df, df_corr, output_dir=output_dir)
    
    # 8. CSV 저장
    df_corr.to_csv(f'{output_dir}/correlation_all.csv', index=False, encoding='utf-8-sig')
    df_missing.to_csv(f'{output_dir}/missing_analysis.csv', index=False, encoding='utf-8-sig')
    print(f"\n💾 CSV 저장 완료")
    
    # 9. HTML 리포트
    generate_html_report(df_corr, classified, output_dir)
    
    print("\n" + "="*80)
    print("✅ 분석 완료!")
    print("="*80)
    print(f"\n📁 결과 폴더: {output_dir}/")
    print(f"   - correlation_report_korean.html (메인 리포트)")
    print(f"   - correlation_all.csv (전체 상관관계)")
    print(f"   - correlation_top30.png")
    print(f"   - correlation_distribution.png")
    print(f"   - pearson_vs_spearman.png")
    print(f"   - scatter_top12.png")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        # 기본 파일 찾기
        candidates = ['M14_Q_*.CSV', '*.csv', 'data/*.csv']
        import glob
        csv_file = None
        for pattern in candidates:
            files = glob.glob(pattern)
            if files:
                csv_file = files[0]
                break
        
        if csv_file is None:
            print("❌ CSV 파일을 찾을 수 없습니다!")
            print("\n사용법: python column_correlation_korean.py 데이터파일.csv")
            exit(1)
    
    main(csv_file)