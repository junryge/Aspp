import pandas as pd
import plotly.graph_objects as go
import numpy as np
import json

# 1. CSV 읽기
df = pd.read_csv('RAL', encoding='utf-8')
df = df.dropna(subset=['CRT_TM'])
df['CRT_TM'] = pd.to_datetime(df['CRT_TM'])
df['CRT_TM_STR'] = df['CRT_TM'].dt.strftime('%Y-%m-%d %H:%M')

# 2. 파생 컬럼
df['6ECMB101_ratio'] = (df['6ECMB101_WELL'] / df['RAIL-TRANSFERPAUSED_WELL'] * 100).fillna(0)
df['date_str'] = df['CRT_TM'].dt.strftime('%Y-%m-%d')
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

# JSON 데이터
trend_time = df['CRT_TM_STR'].tolist()
trend_pause = df['RAIL-TRANSFERPAUSED_WELL'].tolist()
trend_6ecmb = df['6ECMB101_WELL'].tolist()
trend_job2_mean = df['CURRENT_M16A_3F_JOB_2_mean'].tolist()
trend_job2_max = df['CURRENT_M16A_3F_JOB_2_max'].tolist()
trend_avgtime_mean = df['M16HUB_AVGTOTALTIME1MIN_mean'].fillna(0).tolist()
trend_avgtime_max = df['M16HUB_AVGTOTALTIME1MIN_max'].fillna(0).tolist()

daily_date = daily['date_str'].tolist()
daily_pause = daily['RAIL-TRANSFERPAUSED_WELL'].tolist()
daily_6ecmb = daily['6ECMB101_WELL'].tolist()
daily_job2 = daily['CURRENT_M16A_3F_JOB_2_max'].tolist()
daily_avgtime = daily['M16HUB_AVGTOTALTIME1MIN_max'].tolist()

hourly_hour = hourly['hour'].tolist()
hourly_pause = hourly['RAIL-TRANSFERPAUSED_WELL'].tolist()
hourly_6ecmb = hourly['6ECMB101_WELL'].tolist()

corr_z = corr_matrix.values.tolist()
corr_text = np.round(corr_matrix.values, 2).tolist()

scatter_x = df['CURRENT_M16A_3F_JOB_2_mean'].tolist()
scatter_y = df['6ECMB101_WELL'].tolist()
scatter_color = df['M16HUB_AVGTOTALTIME1MIN_mean'].fillna(0).tolist()
scatter_text = df['CRT_TM_STR'].tolist()

# HTML 생성
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
        .tabs {{ display: flex; background: #1e293b; padding: 0 30px; flex-wrap: wrap; }}
        .tab {{ padding: 15px 20px; cursor: pointer; border-bottom: 3px solid transparent; color: #94a3b8; font-size: 14px; }}
        .tab:hover {{ color: white; }}
        .tab.active {{ border-bottom-color: #3b82f6; color: white; font-weight: bold; }}
        .tab-content {{ display: none; padding: 20px 30px; }}
        .tab-content.active {{ display: block; }}
        .chart {{ background: #1e293b; border-radius: 10px; padding: 15px; }}
        .chart-title {{ font-size: 18px; font-weight: bold; margin-bottom: 15px; color: #e2e8f0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚦 RAIL-TRANSFERPAUSED 분석 대시보드</h1>
        <p>📅 2025-11-25 00:00 ~ 2026-01-06 07:00 | 📊 10분 단위 {len(df):,}건</p>
    </div>

    <div class="cards">
        <div class="card"><h3>📊 RAIL-TRANSFERPAUSED</h3><div class="value">{total_pause:,.0f}</div></div>
        <div class="card"><h3>🔴 6ECMB101</h3><div class="value">{total_6ecmb:,.0f}</div><div class="sub">{ratio_6ecmb:.1f}%</div></div>
        <div class="card {'warning' if max_job2 >= 400 else ''}"><h3>📈 JOB_2 최대</h3><div class="value">{max_job2:.0f}</div></div>
        <div class="card {'danger' if max_avgtime >= 20 else 'warning' if max_avgtime >= 10 else ''}"><h3>⏱️ AvgTime 최대</h3><div class="value">{max_avgtime:.1f}분</div></div>
        <div class="card"><h3>⚠️ 이상치</h3><div class="value">{len(anomaly_df):,}건</div></div>
    </div>

    <div class="tabs">
        <div class="tab active" onclick="showTab('trend')">📈 트렌드</div>
        <div class="tab" onclick="showTab('daily')">📊 일별</div>
        <div class="tab" onclick="showTab('hourly')">🕐 시간대별</div>
        <div class="tab" onclick="showTab('corr')">🔗 상관관계</div>
        <div class="tab" onclick="showTab('scatter')">🔍 산점도</div>
    </div>

    <div id="trend" class="tab-content active">
        <div class="chart">
            <div class="chart-title">📈 RAIL-TRANSFERPAUSED 전체 트렌드 (클릭하여 ON/OFF)</div>
            <div id="chart_trend"></div>
        </div>
    </div>
    <div id="daily" class="tab-content">
        <div class="chart">
            <div class="chart-title">📊 일별 요약 (클릭하여 ON/OFF)</div>
            <div id="chart_daily"></div>
        </div>
    </div>
    <div id="hourly" class="tab-content">
        <div class="chart">
            <div class="chart-title">🕐 시간대별 평균 PAUSE 건수</div>
            <div id="chart_hourly"></div>
        </div>
    </div>
    <div id="corr" class="tab-content">
        <div class="chart">
            <div class="chart-title">🔗 상관관계 분석 (1: 양의 상관, -1: 음의 상관)</div>
            <div id="chart_corr"></div>
        </div>
    </div>
    <div id="scatter" class="tab-content">
        <div class="chart">
            <div class="chart-title">🔍 JOB_2 vs 6ECMB101 산점도 (색상: AvgTime)</div>
            <div id="chart_scatter"></div>
        </div>
    </div>

    <script>
        function showTab(tabId) {{
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById(tabId).classList.add('active');
            window.dispatchEvent(new Event('resize'));
        }}

        var darkLayout = {{
            paper_bgcolor: '#1e293b',
            plot_bgcolor: '#1e293b',
            font: {{ color: '#e2e8f0' }},
            hovermode: 'x unified'
        }};

        // 트렌드
        var trendData = [
            {{ x: {json.dumps(trend_time)}, y: {json.dumps(trend_pause)}, name: 'RAIL-TRANSFERPAUSED', type: 'scatter', line: {{ color: '#3b82f6', width: 1.5 }} }},
            {{ x: {json.dumps(trend_time)}, y: {json.dumps(trend_6ecmb)}, name: '6ECMB101', type: 'scatter', line: {{ color: '#ef4444', width: 1.5 }} }},
            {{ x: {json.dumps(trend_time)}, y: {json.dumps(trend_job2_mean)}, name: 'JOB_2 평균', type: 'scatter', line: {{ color: '#22c55e', width: 1.5 }}, yaxis: 'y2' }},
            {{ x: {json.dumps(trend_time)}, y: {json.dumps(trend_job2_max)}, name: 'JOB_2 최대', type: 'scatter', line: {{ color: '#f97316', width: 1.5, dash: 'dot' }}, yaxis: 'y2' }},
            {{ x: {json.dumps(trend_time)}, y: {json.dumps(trend_avgtime_mean)}, name: 'AvgTime 평균', type: 'scatter', line: {{ color: '#a855f7', width: 1.5 }}, yaxis: 'y3' }},
            {{ x: {json.dumps(trend_time)}, y: {json.dumps(trend_avgtime_max)}, name: 'AvgTime 최대', type: 'scatter', line: {{ color: '#ec4899', width: 1.5, dash: 'dot' }}, yaxis: 'y3' }}
        ];
        var trendLayout = Object.assign({{}}, darkLayout, {{
            height: 650,
            margin: {{ t: 10, b: 100, l: 70, r: 120 }},
            xaxis: {{ title: '시간', rangeslider: {{ visible: true }} }},
            yaxis: {{ title: 'PAUSE 건수', side: 'left', color: '#3b82f6' }},
            yaxis2: {{ title: 'JOB_2', side: 'right', overlaying: 'y', color: '#22c55e' }},
            yaxis3: {{ title: 'AvgTime (분)', side: 'right', overlaying: 'y', position: 0.92, color: '#a855f7' }},
            legend: {{ orientation: 'h', y: 1.15, x: 0.5, xanchor: 'center', bgcolor: 'rgba(30,41,59,0.8)' }},
            shapes: [
                {{ type: 'line', y0: 300, y1: 300, x0: 0, x1: 1, xref: 'paper', yref: 'y2', line: {{ color: 'red', width: 2 }} }},
                {{ type: 'line', y0: 10, y1: 10, x0: 0, x1: 1, xref: 'paper', yref: 'y3', line: {{ color: 'orange', width: 2, dash: 'dash' }} }}
            ]
        }});
        Plotly.newPlot('chart_trend', trendData, trendLayout, {{responsive: true}});

        // 일별
        var dailyData = [
            {{ x: {json.dumps(daily_date)}, y: {json.dumps(daily_pause)}, name: 'RAIL-TRANSFERPAUSED', type: 'bar', marker: {{ color: '#3b82f6' }} }},
            {{ x: {json.dumps(daily_date)}, y: {json.dumps(daily_6ecmb)}, name: '6ECMB101', type: 'bar', marker: {{ color: '#ef4444' }} }},
            {{ x: {json.dumps(daily_date)}, y: {json.dumps(daily_job2)}, name: 'JOB_2 Max', type: 'scatter', mode: 'lines+markers', line: {{ color: '#22c55e', width: 2 }}, yaxis: 'y2' }},
            {{ x: {json.dumps(daily_date)}, y: {json.dumps(daily_avgtime)}, name: 'AvgTime Max', type: 'scatter', mode: 'lines+markers', line: {{ color: '#a855f7', width: 2 }}, yaxis: 'y3' }}
        ];
        var dailyLayout = Object.assign({{}}, darkLayout, {{
            height: 600,
            margin: {{ t: 10, b: 80, l: 70, r: 120 }},
            barmode: 'group',
            xaxis: {{ title: '날짜' }},
            yaxis: {{ title: 'PAUSE 건수', side: 'left' }},
            yaxis2: {{ title: 'JOB_2 Max', side: 'right', overlaying: 'y' }},
            yaxis3: {{ title: 'AvgTime Max', side: 'right', overlaying: 'y', position: 0.92 }},
            legend: {{ orientation: 'h', y: 1.1, x: 0.5, xanchor: 'center', bgcolor: 'rgba(30,41,59,0.8)' }},
            shapes: [
                {{ type: 'line', y0: 300, y1: 300, x0: 0, x1: 1, xref: 'paper', yref: 'y2', line: {{ color: 'red', width: 2 }} }},
                {{ type: 'line', y0: 10, y1: 10, x0: 0, x1: 1, xref: 'paper', yref: 'y3', line: {{ color: 'orange', width: 2, dash: 'dash' }} }}
            ]
        }});
        Plotly.newPlot('chart_daily', dailyData, dailyLayout, {{responsive: true}});

        // 시간대별
        var hourlyData = [
            {{ x: {json.dumps(hourly_hour)}, y: {json.dumps(hourly_pause)}, name: 'RAIL-TRANSFERPAUSED', type: 'bar', marker: {{ color: '#3b82f6' }} }},
            {{ x: {json.dumps(hourly_hour)}, y: {json.dumps(hourly_6ecmb)}, name: '6ECMB101', type: 'bar', marker: {{ color: '#ef4444' }} }}
        ];
        var hourlyLayout = Object.assign({{}}, darkLayout, {{
            height: 500,
            margin: {{ t: 10, b: 60, l: 60, r: 40 }},
            barmode: 'group',
            xaxis: {{ title: '시간 (0~23시)', dtick: 1 }},
            yaxis: {{ title: '평균 건수' }},
            legend: {{ orientation: 'h', y: 1.1, x: 0.5, xanchor: 'center', bgcolor: 'rgba(30,41,59,0.8)' }}
        }});
        Plotly.newPlot('chart_hourly', hourlyData, hourlyLayout, {{responsive: true}});

        // 상관관계
        var corrData = [{{
            z: {json.dumps(corr_z)},
            x: ['RAIL-TRANSFERPAUSED', '6ECMB101', 'JOB_2', 'AvgTime'],
            y: ['RAIL-TRANSFERPAUSED', '6ECMB101', 'JOB_2', 'AvgTime'],
            type: 'heatmap',
            colorscale: 'RdBu',
            zmid: 0,
            text: {json.dumps(corr_text)},
            texttemplate: '%{{text}}',
            hovertemplate: '%{{x}} vs %{{y}}: %{{z:.3f}}<extra></extra>'
        }}];
        var corrLayout = Object.assign({{}}, darkLayout, {{
            height: 500,
            margin: {{ t: 10, b: 60, l: 150, r: 40 }}
        }});
        Plotly.newPlot('chart_corr', corrData, corrLayout, {{responsive: true}});

        // 산점도
        var scatterData = [{{
            x: {json.dumps(scatter_x)},
            y: {json.dumps(scatter_y)},
            mode: 'markers',
            type: 'scatter',
            marker: {{
                size: 5,
                color: {json.dumps(scatter_color)},
                colorscale: 'Viridis',
                showscale: true,
                colorbar: {{ title: 'AvgTime' }}
            }},
            text: {json.dumps(scatter_text)},
            hovertemplate: '시간: %{{text}}<br>JOB_2: %{{x:.1f}}<br>6ECMB101: %{{y}}<extra></extra>'
        }}];
        var scatterLayout = Object.assign({{}}, darkLayout, {{
            height: 550,
            margin: {{ t: 10, b: 60, l: 70, r: 40 }},
            xaxis: {{ title: 'CURRENT_M16A_3F_JOB_2 평균' }},
            yaxis: {{ title: '6ECMB101 PAUSE 건수' }},
            shapes: [
                {{ type: 'line', x0: 300, x1: 300, y0: 0, y1: 1, yref: 'paper', line: {{ color: 'red', width: 2 }} }},
                {{ type: 'line', y0: 50, y1: 50, x0: 0, x1: 1, xref: 'paper', line: {{ color: 'orange', width: 2, dash: 'dash' }} }}
            ]
        }});
        Plotly.newPlot('chart_scatter', scatterData, scatterLayout, {{responsive: true}});
    </script>
</body>
</html>
'''

with open('RAIL_대시보드.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("저장 완료: RAIL_대시보드.html")