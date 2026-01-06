import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# 1. CSV 읽기
df = pd.read_csv('RAL', encoding='utf-8')
df = df.dropna(subset=['CRT_TM'])
df['CRT_TM'] = pd.to_datetime(df['CRT_TM'])
df['CRT_TM_STR'] = df['CRT_TM'].dt.strftime('%Y%m%d%H%M')

# 2. 파생 컬럼
df['6ECMB101_ratio'] = (df['6ECMB101_WELL'] / df['RAIL-TRANSFERPAUSED_WELL'] * 100).fillna(0)
df['date_str'] = df['CRT_TM'].dt.strftime('%Y%m%d')
df['hour'] = df['CRT_TM'].dt.hour
df['is_anomaly'] = ((df['CURRENT_M16A_3F_JOB_2_max'] >= 400) | (df['6ECMB101_ratio'] >= 70))

# 3. 집계
daily = df.groupby('date_str').agg({
    'RAIL-TRANSFERPAUSED_WELL': 'sum',
    '6ECMB101_WELL': 'sum',
    'CURRENT_M16A_3F_JOB_2_max': 'max',
    'M16HUB_AVGTOTALTIME1MIN_max': 'max'
}).reset_index()
daily['ratio'] = daily['6ECMB101_WELL'] / daily['RAIL-TRANSFERPAUSED_WELL'] * 100

hourly = df.groupby('hour').agg({
    'RAIL-TRANSFERPAUSED_WELL': 'mean',
    '6ECMB101_WELL': 'mean',
    'CURRENT_M16A_3F_JOB_2_mean': 'mean'
}).reset_index()

corr_cols = ['RAIL-TRANSFERPAUSED_WELL', '6ECMB101_WELL', 'CURRENT_M16A_3F_JOB_2_mean', 'M16HUB_AVGTOTALTIME1MIN_mean']
corr_matrix = df[corr_cols].corr()

anomaly_df = df[df['is_anomaly']]

# 통계
total_pause = df['RAIL-TRANSFERPAUSED_WELL'].sum()
total_6ecmb = df['6ECMB101_WELL'].sum()
ratio_6ecmb = total_6ecmb / total_pause * 100
max_job2 = df['CURRENT_M16A_3F_JOB_2_max'].max()
max_avgtime = df['M16HUB_AVGTOTALTIME1MIN_max'].max()

# ========== HTML 생성 ==========
html_content = f'''
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>RAIL-TRANSFERPAUSED 대시보드</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Malgun Gothic', sans-serif; background: #0f172a; color: white; }}
        .header {{ background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); padding: 20px 30px; }}
        .header h1 {{ font-size: 24px; }}
        .header p {{ opacity: 0.8; margin-top: 5px; }}
        .cards {{ display: flex; gap: 15px; padding: 20px 30px; flex-wrap: wrap; }}
        .card {{ background: #1e293b; padding: 20px; border-radius: 10px; flex: 1; min-width: 150px; text-align: center; }}
        .card h3 {{ font-size: 12px; color: #94a3b8; margin-bottom: 8px; }}
        .card .value {{ font-size: 28px; font-weight: bold; color: #3b82f6; }}
        .card .sub {{ font-size: 11px; color: #64748b; margin-top: 5px; }}
        .card.warning .value {{ color: #f59e0b; }}
        .card.danger .value {{ color: #ef4444; }}
        .tabs {{ display: flex; background: #1e293b; padding: 0 30px; }}
        .tab {{ padding: 15px 25px; cursor: pointer; border-bottom: 3px solid transparent; color: #94a3b8; }}
        .tab:hover {{ color: white; }}
        .tab.active {{ border-bottom-color: #3b82f6; color: white; font-weight: bold; }}
        .tab-content {{ display: none; padding: 20px 30px; }}
        .tab-content.active {{ display: block; }}
        .chart {{ background: #1e293b; border-radius: 10px; padding: 15px; margin-bottom: 15px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚦 RAIL-TRANSFERPAUSED 분석 대시보드</h1>
        <p>📅 202511250000 ~ 202601060700 | 📊 10분 단위 {len(df):,}건</p>
    </div>

    <div class="cards">
        <div class="card">
            <h3>📊 총 PAUSE</h3>
            <div class="value">{total_pause:,.0f}</div>
        </div>
        <div class="card">
            <h3>🔴 6ECMB101</h3>
            <div class="value">{total_6ecmb:,.0f}</div>
            <div class="sub">{ratio_6ecmb:.1f}%</div>
        </div>
        <div class="card {'warning' if max_job2 >= 400 else ''}">
            <h3>📈 JOB_2 최대</h3>
            <div class="value">{max_job2:.0f}</div>
        </div>
        <div class="card {'danger' if max_avgtime >= 20 else 'warning' if max_avgtime >= 10 else ''}">
            <h3>⏱️ AvgTime 최대</h3>
            <div class="value">{max_avgtime:.1f}분</div>
        </div>
        <div class="card">
            <h3>⚠️ 이상치</h3>
            <div class="value">{len(anomaly_df):,}건</div>
        </div>
    </div>

    <div class="tabs">
        <div class="tab active" onclick="showTab('trend')">📈 트렌드</div>
        <div class="tab" onclick="showTab('daily')">📊 일별</div>
        <div class="tab" onclick="showTab('hourly')">🕐 시간대별</div>
        <div class="tab" onclick="showTab('corr')">🔗 상관관계</div>
        <div class="tab" onclick="showTab('scatter')">🔍 산점도</div>
    </div>

    <div id="trend" class="tab-content active">
        <div class="chart" id="chart_trend"></div>
    </div>
    <div id="daily" class="tab-content">
        <div class="chart" id="chart_daily"></div>
    </div>
    <div id="hourly" class="tab-content">
        <div class="chart" id="chart_hourly"></div>
    </div>
    <div id="corr" class="tab-content">
        <div class="chart" id="chart_corr"></div>
    </div>
    <div id="scatter" class="tab-content">
        <div class="chart" id="chart_scatter"></div>
    </div>

    <script>
        function showTab(tabId) {{
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.querySelector(`.tab[onclick="showTab('${{tabId}}')"]`).classList.add('active');
            document.getElementById(tabId).classList.add('active');
            window.dispatchEvent(new Event('resize'));
        }}

        var dark = {{ paper_bgcolor: '#1e293b', plot_bgcolor: '#1e293b', font: {{ color: '#e2e8f0' }}, margin: {{ t: 50, b: 50, l: 60, r: 30 }} }};

        // 트렌드
        var trend = {{
            data: [
                {{ x: {df['CRT_TM_STR'].tolist()}, y: {df['RAIL-TRANSFERPAUSED_WELL'].tolist()}, name: '전체 PAUSE', type: 'scatter', line: {{ color: '#3b82f6' }} }},
                {{ x: {df['CRT_TM_STR'].tolist()}, y: {df['6ECMB101_WELL'].tolist()}, name: '6ECMB101', type: 'scatter', line: {{ color: '#ef4444' }} }},
                {{ x: {df['CRT_TM_STR'].tolist()}, y: {df['CURRENT_M16A_3F_JOB_2_mean'].tolist()}, name: 'JOB_2', type: 'scatter', yaxis: 'y2', line: {{ color: '#22c55e' }} }}
            ],
            layout: Object.assign({{ 
                height: 500, 
                title: 'PAUSE 건수 & JOB_2 트렌드',
                yaxis: {{ title: 'PAUSE 건수' }},
                yaxis2: {{ title: 'JOB_2', overlaying: 'y', side: 'right' }},
                xaxis: {{ rangeslider: {{ visible: true }} }},
                shapes: [{{ type: 'line', y0: 300, y1: 300, x0: 0, x1: 1, xref: 'paper', yref: 'y2', line: {{ color: 'red', dash: 'dash' }} }}]
            }}, dark)
        }};
        Plotly.newPlot('chart_trend', trend.data, trend.layout, {{responsive: true}});

        // 일별
        var daily = {{
            data: [
                {{ x: {daily['date_str'].tolist()}, y: {daily['RAIL-TRANSFERPAUSED_WELL'].tolist()}, name: '전체', type: 'bar', marker: {{ color: '#3b82f6' }} }},
                {{ x: {daily['date_str'].tolist()}, y: {daily['6ECMB101_WELL'].tolist()}, name: '6ECMB101', type: 'bar', marker: {{ color: '#ef4444' }} }}
            ],
            layout: Object.assign({{ height: 500, title: '일별 PAUSE 건수', barmode: 'group' }}, dark)
        }};
        Plotly.newPlot('chart_daily', daily.data, daily.layout, {{responsive: true}});

        // 시간대별
        var hourly = {{
            data: [
                {{ x: {hourly['hour'].tolist()}, y: {hourly['RAIL-TRANSFERPAUSED_WELL'].tolist()}, name: '전체', type: 'bar', marker: {{ color: '#3b82f6' }} }},
                {{ x: {hourly['hour'].tolist()}, y: {hourly['6ECMB101_WELL'].tolist()}, name: '6ECMB101', type: 'bar', marker: {{ color: '#ef4444' }} }}
            ],
            layout: Object.assign({{ height: 500, title: '시간대별 평균 PAUSE 건수', barmode: 'group', xaxis: {{ title: '시간 (0~23시)', dtick: 1 }} }}, dark)
        }};
        Plotly.newPlot('chart_hourly', hourly.data, hourly.layout, {{responsive: true}});

        // 상관관계
        var corr = {{
            data: [{{
                z: {corr_matrix.values.tolist()},
                x: ['전체PAUSE', '6ECMB101', 'JOB_2', 'AvgTime'],
                y: ['전체PAUSE', '6ECMB101', 'JOB_2', 'AvgTime'],
                type: 'heatmap',
                colorscale: 'RdBu',
                zmid: 0,
                text: {np.round(corr_matrix.values, 2).tolist()},
                texttemplate: '%{{text}}',
                hovertemplate: '%{{x}} vs %{{y}}: %{{z:.3f}}<extra></extra>'
            }}],
            layout: Object.assign({{ height: 500, title: '상관관계 분석' }}, dark)
        }};
        Plotly.newPlot('chart_corr', corr.data, corr.layout, {{responsive: true}});

        // 산점도
        var scatter = {{
            data: [{{
                x: {df['CURRENT_M16A_3F_JOB_2_mean'].tolist()},
                y: {df['6ECMB101_WELL'].tolist()},
                mode: 'markers',
                type: 'scatter',
                marker: {{ 
                    size: 5, 
                    color: {df['M16HUB_AVGTOTALTIME1MIN_mean'].fillna(0).tolist()}, 
                    colorscale: 'Viridis', 
                    showscale: true,
                    colorbar: {{ title: 'AvgTime' }}
                }},
                text: {df['CRT_TM_STR'].tolist()},
                hovertemplate: '시간: %{{text}}<br>JOB_2: %{{x:.1f}}<br>6ECMB101: %{{y}}<extra></extra>'
            }}],
            layout: Object.assign({{ 
                height: 500, 
                title: 'JOB_2 vs 6ECMB101 (색상: AvgTime)',
                xaxis: {{ title: 'JOB_2' }},
                yaxis: {{ title: '6ECMB101' }},
                shapes: [
                    {{ type: 'line', x0: 300, x1: 300, y0: 0, y1: 1, yref: 'paper', line: {{ color: 'red', dash: 'dash' }} }},
                    {{ type: 'line', y0: 50, y1: 50, x0: 0, x1: 1, xref: 'paper', line: {{ color: 'orange', dash: 'dash' }} }}
                ]
            }}, dark)
        }};
        Plotly.newPlot('chart_scatter', scatter.data, scatter.layout, {{responsive: true}});
    </script>
</body>
</html>
'''

with open('RAIL_대시보드.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("저장 완료: RAIL_대시보드.html")