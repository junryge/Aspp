# -*- coding: utf-8 -*-
"""
================================================================================
V10_4 신규 컬럼 분석 코드 (HTML 결과 출력)
추가된 컬럼: AHTO+AHSO, REWORK, AHSO

[컬럼 설명]
- AHTO+AHSO: Q-Time 위반 알람 합계 (늦음+빠름)
- REWORK: Q-Time 위반 후 재작업 처분된 Lot 수
- AHSO: Q-Time 하한 미달 알람 (너무 빨리 이동)
================================================================================
"""

import os
import warnings
import glob
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# 설정
# ============================================================================
CONFIG = {
    'limit_value': 1700,
    'target_column': 'TOTALCNT',
}

NEW_COLUMNS = ['AHTO+AHSO', 'REWORK', 'AHSO']

print("=" * 70)
print("🔍 V10_4 신규 컬럼 분석 (Q-Time Interlock 알람)")
print("=" * 70)

# ============================================================================
# 데이터 로드
# ============================================================================
def load_csv(filepath):
    encodings = ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr']
    for enc in encodings:
        try:
            return pd.read_csv(filepath, encoding=enc, on_bad_lines='skip')
        except:
            continue
    raise ValueError(f"파일 인코딩 오류: {filepath}")

def load_data(path_input):
    print(f"\n[1/6] 데이터 로드 중...")
    if '*' in path_input:
        files = sorted(glob.glob(path_input))
    elif os.path.exists(path_input):
        files = [path_input]
    else:
        raise ValueError(f"파일 없음: {path_input}")
    
    dfs = []
    for f in files:
        df = load_csv(f)
        print(f"  - {os.path.basename(f)}: {len(df):,}행")
        dfs.append(df)
    
    result = pd.concat(dfs, ignore_index=True)
    print(f"  → 총: {len(result):,}행")
    return result

# ============================================================================
# 분석 함수들
# ============================================================================
def analyze_new_columns(df):
    print("\n[2/6] 기본 통계...")
    results = {}
    for col in NEW_COLUMNS:
        if col in df.columns:
            data = pd.to_numeric(df[col], errors='coerce')
            non_zero = data[data > 0]
            results[col] = {
                'mean': data.mean(),
                'std': data.std(),
                'min': data.min(),
                'max': data.max(),
                'median': data.median(),
                'zero_pct': (data == 0).sum() / len(data) * 100,
                'nonzero_mean': non_zero.mean() if len(non_zero) > 0 else 0,
                'nonzero_count': len(non_zero),
                'total_count': len(data),
            }
    return results

def analyze_correlation(df):
    print("[3/6] 상관관계...")
    target = pd.to_numeric(df[CONFIG['target_column']], errors='coerce')
    correlations = {}
    for col in NEW_COLUMNS:
        if col in df.columns:
            data = pd.to_numeric(df[col], errors='coerce')
            correlations[col] = target.corr(data)
    return correlations

def analyze_danger_zone(df):
    print("[4/6] 위험구간...")
    target = pd.to_numeric(df[CONFIG['target_column']], errors='coerce')
    danger = target >= CONFIG['limit_value']
    safe = target < CONFIG['limit_value']
    
    analysis = {'danger_count': danger.sum(), 'safe_count': safe.sum()}
    for col in NEW_COLUMNS:
        if col in df.columns:
            data = pd.to_numeric(df[col], errors='coerce')
            d_mean = data[danger].mean() if danger.sum() > 0 else 0
            s_mean = data[safe].mean() if safe.sum() > 0 else 0
            diff_pct = ((d_mean - s_mean) / s_mean * 100) if s_mean != 0 else 0
            analysis[col] = {'danger_mean': d_mean, 'safe_mean': s_mean, 'diff_pct': diff_pct}
    return analysis

def analyze_lead_indicator(df, look_ahead=10):
    print("[5/6] 선행지표...")
    target = pd.to_numeric(df[CONFIG['target_column']], errors='coerce')
    future_danger = (target.shift(-look_ahead) >= CONFIG['limit_value']).astype(int)
    
    lead = {}
    for col in NEW_COLUMNS:
        if col in df.columns:
            data = pd.to_numeric(df[col], errors='coerce')
            valid = ~(data.isna() | future_danger.isna())
            if valid.sum() > 100:
                corr = data[valid].corr(future_danger[valid])
                # 임계값별 위험확률
                thresh_analysis = []
                for pct in [50, 75, 90]:
                    thresh = data.quantile(pct/100)
                    mask = (data >= thresh) & valid
                    if mask.sum() > 0:
                        rate = future_danger[mask].mean() * 100
                        thresh_analysis.append({'pct': pct, 'thresh': thresh, 'danger_rate': rate})
                lead[col] = {'corr': corr, 'thresholds': thresh_analysis}
    return lead

def analyze_time_pattern(df):
    print("[6/6] 시간대별 패턴...")
    if 'CURRTIME' not in df.columns:
        return None
    
    df['HOUR'] = pd.to_datetime(df['CURRTIME'].astype(str), format='%Y%m%d%H%M', errors='coerce').dt.hour
    
    patterns = {}
    for col in NEW_COLUMNS:
        if col in df.columns:
            hourly = df.groupby('HOUR')[col].agg(['mean', 'sum']).reset_index()
            patterns[col] = hourly.to_dict('records')
    return patterns

# ============================================================================
# HTML 생성
# ============================================================================
def generate_html(df, stats, correlations, danger, lead, time_patterns, output_path):
    print("\n📊 HTML 대시보드 생성 중...")
    
    total_rows = len(df)
    danger_count = danger.get('danger_count', 0)
    danger_pct = danger_count / total_rows * 100 if total_rows > 0 else 0
    
    # 권장사항 계산
    recommendations = []
    for col in NEW_COLUMNS:
        if col in correlations:
            corr = correlations[col]
            diff = danger.get(col, {}).get('diff_pct', 0)
            score = abs(corr) * 50 + abs(diff) * 0.5
            if score >= 15:
                level = 'important'
            elif score >= 5:
                level = 'auxiliary'
            else:
                level = 'low'
            recommendations.append({'col': col, 'corr': corr, 'diff': diff, 'score': score, 'level': level})
    
    html = f'''<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>V10_4 신규 컬럼 분석 결과</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'Segoe UI', sans-serif; background: linear-gradient(135deg, #0a0a1a, #1a1a3e); min-height: 100vh; padding: 30px; color: #fff; }}
.container {{ max-width: 1400px; margin: 0 auto; }}
h1 {{ text-align: center; color: #00d4ff; font-size: 2rem; margin-bottom: 10px; }}
.subtitle {{ text-align: center; color: #888; margin-bottom: 30px; }}
.grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 30px; }}
.card {{ background: rgba(255,255,255,0.05); border-radius: 16px; padding: 25px; border: 1px solid rgba(255,255,255,0.1); }}
.card h3 {{ color: #00d4ff; margin-bottom: 15px; font-size: 1.1rem; }}
.stat-row {{ display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05); }}
.stat-label {{ color: #888; }}
.stat-value {{ font-weight: bold; }}
.stat-value.green {{ color: #00ff88; }}
.stat-value.red {{ color: #ff4466; }}
.stat-value.yellow {{ color: #ffcc00; }}
.stat-value.cyan {{ color: #00d4ff; }}
.section {{ background: rgba(255,255,255,0.03); border-radius: 16px; padding: 25px; margin-bottom: 25px; border: 1px solid rgba(255,255,255,0.1); }}
.section h2 {{ color: #00d4ff; margin-bottom: 20px; }}
.chart-container {{ height: 300px; margin: 20px 0; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.1); }}
th {{ background: rgba(0,212,255,0.1); color: #00d4ff; }}
.badge {{ display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; }}
.badge-green {{ background: rgba(0,255,136,0.2); color: #00ff88; }}
.badge-yellow {{ background: rgba(255,204,0,0.2); color: #ffcc00; }}
.badge-gray {{ background: rgba(136,136,136,0.2); color: #888; }}
.recommendation {{ padding: 15px; margin: 10px 0; border-radius: 10px; }}
.rec-important {{ background: rgba(0,255,136,0.1); border-left: 4px solid #00ff88; }}
.rec-auxiliary {{ background: rgba(0,212,255,0.1); border-left: 4px solid #00d4ff; }}
.rec-low {{ background: rgba(136,136,136,0.1); border-left: 4px solid #888; }}
code {{ background: #1a1a2e; padding: 2px 8px; border-radius: 4px; font-family: monospace; }}
.code-block {{ background: #0d1117; padding: 15px; border-radius: 8px; font-family: monospace; white-space: pre; overflow-x: auto; margin-top: 15px; }}
</style>
</head>
<body>
<div class="container">
    <h1>🔍 V10_4 신규 컬럼 분석 결과</h1>
    <p class="subtitle">Q-Time Interlock 알람 변수 (AHTO+AHSO, REWORK, AHSO)</p>
    
    <!-- 요약 통계 -->
    <div class="grid">
        <div class="card">
            <h3>📊 데이터 개요</h3>
            <div class="stat-row"><span class="stat-label">총 데이터</span><span class="stat-value cyan">{total_rows:,}행</span></div>
            <div class="stat-row"><span class="stat-label">위험구간(1700+)</span><span class="stat-value red">{danger_count:,}개 ({danger_pct:.2f}%)</span></div>
            <div class="stat-row"><span class="stat-label">안전구간</span><span class="stat-value green">{danger.get('safe_count', 0):,}개</span></div>
        </div>
'''
    
    # 각 컬럼별 통계 카드
    for col in NEW_COLUMNS:
        if col in stats:
            s = stats[col]
            corr = correlations.get(col, 0)
            corr_class = 'green' if abs(corr) >= 0.2 else 'yellow' if abs(corr) >= 0.1 else 'red'
            html += f'''
        <div class="card">
            <h3>📈 {col}</h3>
            <div class="stat-row"><span class="stat-label">평균</span><span class="stat-value">{s['mean']:.3f}</span></div>
            <div class="stat-row"><span class="stat-label">범위</span><span class="stat-value">{s['min']:.0f} ~ {s['max']:.0f}</span></div>
            <div class="stat-row"><span class="stat-label">0값 비율</span><span class="stat-value">{s['zero_pct']:.1f}%</span></div>
            <div class="stat-row"><span class="stat-label">양수일때 평균</span><span class="stat-value yellow">{s['nonzero_mean']:.2f}</span></div>
            <div class="stat-row"><span class="stat-label">TOTALCNT 상관</span><span class="stat-value {corr_class}">{corr:.4f}</span></div>
        </div>
'''
    
    html += '''
    </div>
    
    <!-- 위험구간 분석 -->
    <div class="section">
        <h2>⚠️ 위험구간(1700+) vs 안전구간 비교</h2>
        <table>
            <tr><th>컬럼</th><th>위험구간 평균</th><th>안전구간 평균</th><th>차이</th><th>판정</th></tr>
'''
    
    for col in NEW_COLUMNS:
        if col in danger and isinstance(danger[col], dict):
            d = danger[col]
            diff = d.get('diff_pct', 0)
            if abs(diff) >= 30:
                badge = '<span class="badge badge-green">🔥 유의미</span>'
            elif abs(diff) >= 10:
                badge = '<span class="badge badge-yellow">⚡ 주목</span>'
            else:
                badge = '<span class="badge badge-gray">💤 미미</span>'
            html += f'''
            <tr>
                <td><strong>{col}</strong></td>
                <td>{d.get('danger_mean', 0):.3f}</td>
                <td>{d.get('safe_mean', 0):.3f}</td>
                <td style="color: {'#00ff88' if diff > 0 else '#ff4466'}">{diff:+.1f}%</td>
                <td>{badge}</td>
            </tr>
'''
    
    html += '''
        </table>
    </div>
    
    <!-- 선행지표 분석 -->
    <div class="section">
        <h2>🎯 선행 지표 분석 (10분 후 1700+ 예측력)</h2>
        <table>
            <tr><th>컬럼</th><th>10분후 위험 상관</th><th>상위 10% 임계값</th><th>해당시 위험확률</th></tr>
'''
    
    for col in NEW_COLUMNS:
        if col in lead:
            l = lead[col]
            corr = l.get('corr', 0)
            thresh_90 = next((t for t in l.get('thresholds', []) if t['pct'] == 90), None)
            if thresh_90:
                html += f'''
            <tr>
                <td><strong>{col}</strong></td>
                <td style="color: {'#00ff88' if abs(corr) >= 0.1 else '#888'}">{corr:.4f}</td>
                <td>{thresh_90['thresh']:.1f}</td>
                <td style="color: #ffcc00">{thresh_90['danger_rate']:.2f}%</td>
            </tr>
'''
    
    html += '''
        </table>
    </div>
    
    <!-- 시간대별 패턴 차트 -->
    <div class="section">
        <h2>🕐 시간대별 알람 패턴</h2>
        <div class="chart-container"><canvas id="hourlyChart"></canvas></div>
    </div>
    
    <!-- 권장사항 -->
    <div class="section">
        <h2>📋 V10_4 모델 통합 권장사항</h2>
'''
    
    for rec in sorted(recommendations, key=lambda x: -x['score']):
        level_class = f"rec-{rec['level']}"
        level_icon = '✅' if rec['level'] == 'important' else '⚡' if rec['level'] == 'auxiliary' else '💤'
        level_text = '강력 추천 (important 그룹)' if rec['level'] == 'important' else '보조 변수 (auxiliary 그룹)' if rec['level'] == 'auxiliary' else '낮은 우선순위'
        html += f'''
        <div class="recommendation {level_class}">
            <strong>{level_icon} {rec['col']}</strong>: {level_text}<br>
            <small>상관계수: {rec['corr']:.4f} | 위험구간 차이: {rec['diff']:+.1f}% | 점수: {rec['score']:.1f}</small>
        </div>
'''
    
    # FEATURE_GROUPS 코드 생성
    important_cols = [r['col'] for r in recommendations if r['level'] == 'important']
    auxiliary_cols = [r['col'] for r in recommendations if r['level'] in ['auxiliary', 'low']]
    
    code_block = "# V10_4 FEATURE_GROUPS 추가 코드\n\n"
    if important_cols:
        code_block += "# important 그룹에 추가:\n"
        for c in important_cols:
            code_block += f"    '{c}',\n"
    if auxiliary_cols:
        code_block += "\n# auxiliary 또는 pdt_new 그룹에 추가:\n"
        for c in auxiliary_cols:
            code_block += f"    '{c}',\n"
    
    html += f'''
        <div class="code-block">{code_block}</div>
    </div>
</div>

<script>
'''
    
    # 시간대별 차트 데이터
    if time_patterns:
        for col in NEW_COLUMNS:
            if col in time_patterns:
                hours = [p['HOUR'] for p in time_patterns[col]]
                sums = [p['sum'] for p in time_patterns[col]]
                break
        else:
            hours = list(range(24))
            sums = [0] * 24
    else:
        hours = list(range(24))
        sums = [0] * 24
    
    html += f'''
const ctx = document.getElementById('hourlyChart').getContext('2d');
new Chart(ctx, {{
    type: 'bar',
    data: {{
        labels: {hours},
        datasets: [{{
            label: 'Q-Time 알람 합계',
            data: {sums},
            backgroundColor: 'rgba(0, 212, 255, 0.6)',
            borderColor: '#00d4ff',
            borderWidth: 1
        }}]
    }},
    options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{
            legend: {{ labels: {{ color: '#fff' }} }}
        }},
        scales: {{
            x: {{ 
                ticks: {{ color: '#888' }},
                grid: {{ color: 'rgba(255,255,255,0.1)' }},
                title: {{ display: true, text: '시간 (Hour)', color: '#888' }}
            }},
            y: {{ 
                ticks: {{ color: '#888' }},
                grid: {{ color: 'rgba(255,255,255,0.1)' }},
                title: {{ display: true, text: '알람 건수', color: '#888' }}
            }}
        }}
    }}
}});
</script>
</body>
</html>
'''
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ 저장 완료: {output_path}")
    return output_path

# ============================================================================
# 메인
# ============================================================================
def main():
    print("\n" + "=" * 70)
    csv_path = input("📂 CSV 파일 경로 입력: ").strip()
    
    if not csv_path:
        print("❌ 파일 경로 필요")
        return None
    
    try:
        df = load_data(csv_path)
    except Exception as e:
        print(f"❌ 오류: {e}")
        return None
    
    # 컬럼 확인
    found = [c for c in NEW_COLUMNS if c in df.columns]
    print(f"\n[컬럼 확인] {len(found)}/{len(NEW_COLUMNS)}개 발견: {found}")
    
    if not found:
        print("❌ 분석할 컬럼 없음")
        return None
    
    # 분석
    stats = analyze_new_columns(df)
    correlations = analyze_correlation(df)
    danger = analyze_danger_zone(df)
    lead = analyze_lead_indicator(df)
    time_patterns = analyze_time_pattern(df)
    
    # HTML 생성
    output_path = f"V10_4_신규컬럼_분석결과_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    generate_html(df, stats, correlations, danger, lead, time_patterns, output_path)
    
    print("\n" + "=" * 70)
    print("✅ 분석 완료!")
    print("=" * 70)
    
    return output_path

if __name__ == "__main__":
    main()