import streamlit as st
import pandas as pd
import os
from datetime import datetime

st.set_page_config(page_title="급증 감지 분석 시스템 V8", page_icon="📊", layout="wide")

# CSS 스타일
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #00d4ff, #ff6b35);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 10px;
    }
    .sub-header {
        text-align: center;
        color: #888;
        margin-bottom: 30px;
    }
    .metric-card {
        background: #1a1a2e;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
    }
    .success { color: #00ff88; }
    .warning { color: #ffcc00; }
    .danger { color: #ff4466; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">📊 급증 감지 분석 시스템</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">HUBROOM V8.0 | M14_Q V8.3.1</p>', unsafe_allow_html=True)


def analyze_hubroom(df, threshold=350):
    """HUBROOM 분석"""
    drop_cols = ['Unnamed: 22', 'Unnamed: 23']
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    df['사전감지OK_NG'] = ''
    surge_condition = (df['실제값'] >= threshold) & (df['현재값'] <= threshold)
    df.loc[surge_condition, '사전감지OK_NG'] = 'Y'
    
    df['사전감지결과'] = ''
    ok_condition = surge_condition & (df['예측값_MAX'] >= threshold)
    df.loc[ok_condition, '사전감지결과'] = 'OK'
    
    ng_condition = surge_condition & (df['예측값_MAX'] < threshold)
    ng_indices = df[ng_condition].index.tolist()
    
    for idx in ng_indices:
        found = False
        for back in range(1, 6):
            prev_idx = idx - back
            if prev_idx >= 0:
                prev_pred = df.loc[prev_idx, '예측값_MAX']
                if prev_pred >= threshold:
                    df.loc[idx, '사전감지결과'] = f'사전감지({back}분전)'
                    found = True
                    break
        if not found:
            df.loc[idx, '사전감지결과'] = 'NG'
    
    df['오탐/사전감지'] = ''
    fp_condition = (df['실제값'] < threshold) & (df['예측값_MAX'] >= threshold)
    fp_indices = df[fp_condition].index.tolist()
    
    for idx in fp_indices:
        found = False
        for offset in range(-5, 6):
            if offset == 0:
                continue
            check_idx = idx + offset
            if 0 <= check_idx < len(df):
                if df.loc[check_idx, '실제값'] >= threshold:
                    df.loc[idx, '오탐/사전감지'] = '오탐사전감지'
                    found = True
                    break
        if not found:
            df.loc[idx, '오탐/사전감지'] = '오탐'
    
    cols = df.columns.tolist()
    cols.remove('사전감지OK_NG')
    cols.remove('사전감지결과')
    cols.remove('오탐/사전감지')
    actual_idx = cols.index('실제값')
    new_cols = cols[:actual_idx+1] + ['사전감지OK_NG', '사전감지결과', '오탐/사전감지'] + cols[actual_idx+1:]
    df = df[new_cols]
    
    total_surge = len(df[surge_condition])
    ok_cnt = len(df[df['사전감지결과'] == 'OK'])
    pre_cnt = len(df[df['사전감지결과'].str.contains('사전감지', na=False)])
    ng_cnt = len(df[df['사전감지결과'] == 'NG'])
    fp_total = len(fp_indices)
    fp_pre = len(df[df['오탐/사전감지'] == '오탐사전감지'])
    fp_real = len(df[df['오탐/사전감지'] == '오탐'])
    
    return df, {
        'total': total_surge, 'ok': ok_cnt, 'pre': pre_cnt, 'ng': ng_cnt,
        'fp_total': fp_total, 'fp_pre': fp_pre, 'fp_real': fp_real
    }


def analyze_m14(df, threshold=1700):
    """M14_Q 분석"""
    drop_cols = ['12분전TOTALCNT', '12변화량', 'Unnamed: 22', 'Unnamed: 23']
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    df['사전감지OK_NG'] = ''
    surge_condition = (df['실제값'] >= threshold) & (df['현재TOTALCNT'] <= threshold)
    df.loc[surge_condition, '사전감지OK_NG'] = 'Y'
    
    df['사전감지결과'] = ''
    ok_condition = surge_condition & (df['최종예측'] >= threshold)
    df.loc[ok_condition, '사전감지결과'] = 'OK'
    
    ng_condition = surge_condition & (df['최종예측'] < threshold)
    ng_indices = df[ng_condition].index.tolist()
    
    for idx in ng_indices:
        found = False
        for back in range(1, 6):
            prev_idx = idx - back
            if prev_idx >= 0:
                prev_pred = df.loc[prev_idx, '최종예측']
                if prev_pred >= threshold:
                    df.loc[idx, '사전감지결과'] = f'사전감지({back}분전)'
                    found = True
                    break
        if not found:
            df.loc[idx, '사전감지결과'] = 'NG'
    
    df['오탐/사전감지'] = ''
    fp_condition = (df['실제값'] < threshold) & (df['최종예측'] >= threshold)
    fp_indices = df[fp_condition].index.tolist()
    
    for idx in fp_indices:
        found = False
        for offset in range(-5, 6):
            if offset == 0:
                continue
            check_idx = idx + offset
            if 0 <= check_idx < len(df):
                if df.loc[check_idx, '실제값'] >= threshold:
                    df.loc[idx, '오탐/사전감지'] = '오탐사전감지'
                    found = True
                    break
        if not found:
            df.loc[idx, '오탐/사전감지'] = '오탐'
    
    cols = df.columns.tolist()
    cols.remove('사전감지OK_NG')
    cols.remove('사전감지결과')
    cols.remove('오탐/사전감지')
    actual_idx = cols.index('실제값')
    new_cols = cols[:actual_idx+1] + ['사전감지OK_NG', '사전감지결과', '오탐/사전감지'] + cols[actual_idx+1:]
    df = df[new_cols]
    
    total_surge = len(df[surge_condition])
    ok_cnt = len(df[df['사전감지결과'] == 'OK'])
    pre_cnt = len(df[df['사전감지결과'].str.contains('사전감지', na=False)])
    ng_cnt = len(df[df['사전감지결과'] == 'NG'])
    fp_total = len(fp_indices)
    fp_pre = len(df[df['오탐/사전감지'] == '오탐사전감지'])
    fp_real = len(df[df['오탐/사전감지'] == '오탐'])
    
    return df, {
        'total': total_surge, 'ok': ok_cnt, 'pre': pre_cnt, 'ng': ng_cnt,
        'fp_total': fp_total, 'fp_pre': fp_pre, 'fp_real': fp_real
    }


def generate_html_report(hub_stats, m14_stats, hub_df, m14_df):
    """HTML 리포트 생성"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def calc_rates(stats):
        total = stats['total'] if stats['total'] > 0 else 1
        fp_total = stats['fp_total'] if stats['fp_total'] > 0 else 1
        return {
            'ok_rate': stats['ok'] / total * 100,
            'pre_rate': stats['pre'] / total * 100,
            'ng_rate': stats['ng'] / total * 100,
            'detect_rate': (stats['ok'] + stats['pre']) / total * 100,
            'fp_pre_rate': stats['fp_pre'] / fp_total * 100,
            'fp_real_rate': stats['fp_real'] / fp_total * 100
        }
    
    hub_rates = calc_rates(hub_stats) if hub_stats else None
    m14_rates = calc_rates(m14_stats) if m14_stats else None
    
    html = f'''<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>급증 감지 분석 리포트</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700;900&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-dark: #0a0a0f;
            --bg-card: #12121a;
            --accent-hub: #00d4ff;
            --accent-m14: #ff6b35;
            --success: #00ff88;
            --warning: #ffcc00;
            --danger: #ff4466;
            --text-primary: #ffffff;
            --text-secondary: #8888aa;
            --border: #2a2a3a;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Noto Sans KR', sans-serif;
            background: var(--bg-dark);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 40px 20px;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{
            font-size: 2.5rem;
            font-weight: 900;
            background: linear-gradient(135deg, var(--accent-hub), var(--accent-m14));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 10px;
        }}
        .subtitle {{ text-align: center; color: var(--text-secondary); margin-bottom: 40px; }}
        .section {{ margin-bottom: 50px; }}
        .section-title {{
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 25px;
            padding-bottom: 10px;
            border-bottom: 2px solid var(--border);
        }}
        .section-title.hub {{ color: var(--accent-hub); border-color: var(--accent-hub); }}
        .section-title.m14 {{ color: var(--accent-m14); border-color: var(--accent-m14); }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .card {{
            background: var(--bg-card);
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            border: 1px solid var(--border);
        }}
        .card-label {{ font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 10px; text-transform: uppercase; }}
        .card-value {{ font-size: 2.5rem; font-weight: 900; font-family: 'JetBrains Mono', monospace; }}
        .card-sub {{ font-size: 0.9rem; color: var(--text-secondary); margin-top: 5px; }}
        .success {{ color: var(--success); }}
        .warning {{ color: var(--warning); }}
        .danger {{ color: var(--danger); }}
        .hub-color {{ color: var(--accent-hub); }}
        .m14-color {{ color: var(--accent-m14); }}
        .chart-container {{ display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-bottom: 30px; }}
        .chart-box {{ background: var(--bg-card); border-radius: 15px; padding: 25px; border: 1px solid var(--border); }}
        .chart-title {{ font-size: 1.1rem; font-weight: 600; margin-bottom: 20px; }}
        .chart-wrapper {{ height: 280px; position: relative; }}
        .summary-box {{
            background: var(--bg-card);
            border-radius: 15px;
            padding: 25px;
            border: 1px solid var(--border);
        }}
        .summary-title {{ font-size: 1.2rem; font-weight: 600; margin-bottom: 15px; }}
        .summary-content {{
            font-family: 'JetBrains Mono', monospace;
            white-space: pre-wrap;
            line-height: 1.8;
            color: var(--text-secondary);
            font-size: 0.9rem;
        }}
        .footer {{ text-align: center; color: var(--text-secondary); margin-top: 50px; font-size: 0.85rem; }}
        @media (max-width: 768px) {{
            .chart-container {{ grid-template-columns: 1fr; }}
            h1 {{ font-size: 1.8rem; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 급증 감지 분석 리포트</h1>
        <p class="subtitle">생성일시: {now}</p>
'''
    
    # HUBROOM Section
    if hub_stats and hub_stats['total'] > 0:
        html += f'''
        <div class="section">
            <h2 class="section-title hub">🏠 HUBROOM V8.0 분석 결과</h2>
            <div class="grid">
                <div class="card">
                    <div class="card-label">급증 케이스</div>
                    <div class="card-value hub-color">{hub_stats['total']}</div>
                    <div class="card-sub">기준: 350</div>
                </div>
                <div class="card">
                    <div class="card-label">총 감지율</div>
                    <div class="card-value success">{hub_rates['detect_rate']:.1f}%</div>
                    <div class="card-sub">OK + 사전감지</div>
                </div>
                <div class="card">
                    <div class="card-label">OK (즉시감지)</div>
                    <div class="card-value success">{hub_stats['ok']}</div>
                    <div class="card-sub">{hub_rates['ok_rate']:.1f}%</div>
                </div>
                <div class="card">
                    <div class="card-label">사전감지</div>
                    <div class="card-value warning">{hub_stats['pre']}</div>
                    <div class="card-sub">{hub_rates['pre_rate']:.1f}%</div>
                </div>
                <div class="card">
                    <div class="card-label">NG (미감지)</div>
                    <div class="card-value danger">{hub_stats['ng']}</div>
                    <div class="card-sub">{hub_rates['ng_rate']:.1f}%</div>
                </div>
                <div class="card">
                    <div class="card-label">오탐</div>
                    <div class="card-value danger">{hub_stats['fp_real']}</div>
                    <div class="card-sub">{hub_rates['fp_real_rate']:.1f}% (전체 {hub_stats['fp_total']}개)</div>
                </div>
            </div>
            <div class="chart-container">
                <div class="chart-box">
                    <div class="chart-title">📈 감지 결과 분포</div>
                    <div class="chart-wrapper"><canvas id="hubPieChart"></canvas></div>
                </div>
                <div class="chart-box">
                    <div class="chart-title">📉 오탐 분석</div>
                    <div class="chart-wrapper"><canvas id="hubFpChart"></canvas></div>
                </div>
            </div>
            <div class="summary-box">
                <div class="summary-title">📋 HUBROOM 요약</div>
                <pre class="summary-content">=== HUBROOM V8.0 최종 요약 (기준: 350) ===

[ 급증 감지 결과 ]
급증 케이스: {hub_stats['total']}개 (100%)
OK: {hub_stats['ok']}개 ({hub_rates['ok_rate']:.1f}%)
사전감지: {hub_stats['pre']}개 ({hub_rates['pre_rate']:.1f}%)
NG: {hub_stats['ng']}개 ({hub_rates['ng_rate']:.1f}%)
총 감지율: {hub_rates['detect_rate']:.1f}%

[ 오탐 분석 ]
오탐 후보: {hub_stats['fp_total']}개
오탐사전감지: {hub_stats['fp_pre']}개 ({hub_rates['fp_pre_rate']:.1f}%)
오탐: {hub_stats['fp_real']}개 ({hub_rates['fp_real_rate']:.1f}%)</pre>
            </div>
        </div>
'''
    
    # M14 Section
    if m14_stats and m14_stats['total'] > 0:
        html += f'''
        <div class="section">
            <h2 class="section-title m14">🏭 M14_Q V8.3.1 분석 결과</h2>
            <div class="grid">
                <div class="card">
                    <div class="card-label">급증 케이스</div>
                    <div class="card-value m14-color">{m14_stats['total']}</div>
                    <div class="card-sub">기준: 1700</div>
                </div>
                <div class="card">
                    <div class="card-label">총 감지율</div>
                    <div class="card-value success">{m14_rates['detect_rate']:.1f}%</div>
                    <div class="card-sub">OK + 사전감지</div>
                </div>
                <div class="card">
                    <div class="card-label">OK (즉시감지)</div>
                    <div class="card-value success">{m14_stats['ok']}</div>
                    <div class="card-sub">{m14_rates['ok_rate']:.1f}%</div>
                </div>
                <div class="card">
                    <div class="card-label">사전감지</div>
                    <div class="card-value warning">{m14_stats['pre']}</div>
                    <div class="card-sub">{m14_rates['pre_rate']:.1f}%</div>
                </div>
                <div class="card">
                    <div class="card-label">NG (미감지)</div>
                    <div class="card-value danger">{m14_stats['ng']}</div>
                    <div class="card-sub">{m14_rates['ng_rate']:.1f}%</div>
                </div>
                <div class="card">
                    <div class="card-label">오탐</div>
                    <div class="card-value danger">{m14_stats['fp_real']}</div>
                    <div class="card-sub">{m14_rates['fp_real_rate']:.1f}% (전체 {m14_stats['fp_total']}개)</div>
                </div>
            </div>
            <div class="chart-container">
                <div class="chart-box">
                    <div class="chart-title">📈 감지 결과 분포</div>
                    <div class="chart-wrapper"><canvas id="m14PieChart"></canvas></div>
                </div>
                <div class="chart-box">
                    <div class="chart-title">📉 오탐 분석</div>
                    <div class="chart-wrapper"><canvas id="m14FpChart"></canvas></div>
                </div>
            </div>
            <div class="summary-box">
                <div class="summary-title">📋 M14_Q 요약</div>
                <pre class="summary-content">=== M14_Q V8.3.1 최종 요약 (기준: 1700) ===

[ 급증 감지 결과 ]
급증 케이스: {m14_stats['total']}개 (100%)
OK: {m14_stats['ok']}개 ({m14_rates['ok_rate']:.1f}%)
사전감지: {m14_stats['pre']}개 ({m14_rates['pre_rate']:.1f}%)
NG: {m14_stats['ng']}개 ({m14_rates['ng_rate']:.1f}%)
총 감지율: {m14_rates['detect_rate']:.1f}%

[ 오탐 분석 ]
오탐 후보: {m14_stats['fp_total']}개
오탐사전감지: {m14_stats['fp_pre']}개 ({m14_rates['fp_pre_rate']:.1f}%)
오탐: {m14_stats['fp_real']}개 ({m14_rates['fp_real_rate']:.1f}%)</pre>
            </div>
        </div>
'''
    
    # Chart Scripts
    html += '''
        <div class="footer">급증 감지 분석 시스템 V8 | AMHS 물류 예측 모델</div>
    </div>
    <script>
        const chartOptions = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom', labels: { color: '#ffffff', font: { size: 11 } } }
            }
        };
'''
    
    if hub_stats and hub_stats['total'] > 0:
        html += f'''
        new Chart(document.getElementById('hubPieChart'), {{
            type: 'doughnut',
            data: {{
                labels: ['OK', '사전감지', 'NG'],
                datasets: [{{ data: [{hub_stats['ok']}, {hub_stats['pre']}, {hub_stats['ng']}], backgroundColor: ['#00ff88', '#ffcc00', '#ff4466'], borderWidth: 0 }}]
            }},
            options: chartOptions
        }});
        new Chart(document.getElementById('hubFpChart'), {{
            type: 'doughnut',
            data: {{
                labels: ['오탐사전감지', '오탐'],
                datasets: [{{ data: [{hub_stats['fp_pre']}, {hub_stats['fp_real']}], backgroundColor: ['#ffcc00', '#ff4466'], borderWidth: 0 }}]
            }},
            options: chartOptions
        }});
'''
    
    if m14_stats and m14_stats['total'] > 0:
        html += f'''
        new Chart(document.getElementById('m14PieChart'), {{
            type: 'doughnut',
            data: {{
                labels: ['OK', '사전감지', 'NG'],
                datasets: [{{ data: [{m14_stats['ok']}, {m14_stats['pre']}, {m14_stats['ng']}], backgroundColor: ['#00ff88', '#ffcc00', '#ff4466'], borderWidth: 0 }}]
            }},
            options: chartOptions
        }});
        new Chart(document.getElementById('m14FpChart'), {{
            type: 'doughnut',
            data: {{
                labels: ['오탐사전감지', '오탐'],
                datasets: [{{ data: [{m14_stats['fp_pre']}, {m14_stats['fp_real']}], backgroundColor: ['#ffcc00', '#ff4466'], borderWidth: 0 }}]
            }},
            options: chartOptions
        }});
'''
    
    html += '''
    </script>
</body>
</html>'''
    
    return html


# ============================================
# Streamlit UI
# ============================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏠 HUBROOM V8.0")
    st.caption("기준값: 350 | 컬럼: 현재값, 실제값, 예측값_MAX")
    hub_file = st.file_uploader("HUBROOM CSV 업로드", type=['csv'], key='hub')

with col2:
    st.subheader("🏭 M14_Q V8.3.1")
    st.caption("기준값: 1700 | 컬럼: 현재TOTALCNT, 실제값, 최종예측")
    m14_file = st.file_uploader("M14_Q CSV 업로드", type=['csv'], key='m14')

st.markdown("---")

if st.button("🔍 분석 시작", type="primary", use_container_width=True):
    hub_stats = None
    m14_stats = None
    hub_df = None
    m14_df = None
    
    if hub_file:
        with st.spinner("HUBROOM 분석 중..."):
            df = pd.read_csv(hub_file)
            hub_df, hub_stats = analyze_hubroom(df.copy(), 350)
        
        st.success(f"✅ HUBROOM 분석 완료! 급증 케이스: {hub_stats['total']}개")
        
        col1, col2, col3, col4 = st.columns(4)
        total = hub_stats['total'] if hub_stats['total'] > 0 else 1
        col1.metric("OK", f"{hub_stats['ok']}개", f"{hub_stats['ok']/total*100:.1f}%")
        col2.metric("사전감지", f"{hub_stats['pre']}개", f"{hub_stats['pre']/total*100:.1f}%")
        col3.metric("NG", f"{hub_stats['ng']}개", f"{hub_stats['ng']/total*100:.1f}%")
        col4.metric("총 감지율", f"{(hub_stats['ok']+hub_stats['pre'])/total*100:.1f}%")
        
        st.download_button("📥 HUBROOM CSV 다운로드", hub_df.to_csv(index=False, encoding='utf-8-sig'), "HUBROOM_결과.csv", "text/csv")
    
    if m14_file:
        with st.spinner("M14_Q 분석 중..."):
            df = pd.read_csv(m14_file)
            m14_df, m14_stats = analyze_m14(df.copy(), 1700)
        
        st.success(f"✅ M14_Q 분석 완료! 급증 케이스: {m14_stats['total']}개")
        
        col1, col2, col3, col4 = st.columns(4)
        total = m14_stats['total'] if m14_stats['total'] > 0 else 1
        col1.metric("OK", f"{m14_stats['ok']}개", f"{m14_stats['ok']/total*100:.1f}%")
        col2.metric("사전감지", f"{m14_stats['pre']}개", f"{m14_stats['pre']/total*100:.1f}%")
        col3.metric("NG", f"{m14_stats['ng']}개", f"{m14_stats['ng']/total*100:.1f}%")
        col4.metric("총 감지율", f"{(m14_stats['ok']+m14_stats['pre'])/total*100:.1f}%")
        
        st.download_button("📥 M14_Q CSV 다운로드", m14_df.to_csv(index=False, encoding='utf-8-sig'), "M14_결과.csv", "text/csv")
    
    # HTML 리포트 생성
    if hub_stats or m14_stats:
        st.markdown("---")
        st.subheader("📄 HTML 리포트 다운로드")
        html_report = generate_html_report(hub_stats, m14_stats, hub_df, m14_df)
        st.download_button(
            "📥 HTML 리포트 다운로드",
            html_report,
            f"급증감지_분석리포트_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            "text/html",
            use_container_width=True
        )