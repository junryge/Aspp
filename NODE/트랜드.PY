import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json

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

# ========== 차트 생성 ==========

# 1) 트렌드 차트 (subplot 4행)
fig_trend = make_subplots(
    rows=4, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=(
        'PAUSE 건수 (전체 vs 6ECMB101)',
        'CURRENT_M16A_3F_JOB_2 (평균/최대)',
        'M16HUB_AVGTOTALTIME1MIN (평균/최대)',
        '6ECMB101 비율 (%)'
    ),
    row_heights=[0.25, 0.25, 0.25, 0.25]
)

fig_trend.add_trace(go.Scatter(x=df['CRT_TM_STR'], y=df['RAIL-TRANSFERPAUSED_WELL'], name='전체 PAUSE', line=dict(color='#3b82f6', width=1)), row=1, col=1)
fig_trend.add_trace(go.Scatter(x=df['CRT_TM_STR'], y=df['6ECMB101_WELL'], name='6ECMB101', line=dict(color='#ef4444', width=1)), row=1, col=1)

fig_trend.add_trace(go.Scatter(x=df['CRT_TM_STR'], y=df['CURRENT_M16A_3F_JOB_2_mean'], name='JOB_2 평균', line=dict(color='#22c55e', width=1)), row=2, col=1)
fig_trend.add_trace(go.Scatter(x=df['CRT_TM_STR'], y=df['CURRENT_M16A_3F_JOB_2_max'], name='JOB_2 최대', line=dict(color='#f97316', width=1)), row=2, col=1)
fig_trend.add_hline(y=300, line_dash="dash", line_color="red", row=2, col=1)
fig_trend.add_hline(y=400, line_dash="dash", line_color="darkred", row=2, col=1)

fig_trend.add_trace(go.Scatter(x=df['CRT_TM_STR'], y=df['M16HUB_AVGTOTALTIME1MIN_mean'], name='AvgTime 평균', line=dict(color='#a855f7', width=1)), row=3, col=1)
fig_trend.add_trace(go.Scatter(x=df['CRT_TM_STR'], y=df['M16HUB_AVGTOTALTIME1MIN_max'], name='AvgTime 최대', line=dict(color='#ec4899', width=1)), row=3, col=1)
fig_trend.add_hline(y=10, line_dash="dash", line_color="orange", row=3, col=1)
fig_trend.add_hline(y=20, line_dash="dash", line_color="red", row=3, col=1)

fig_trend.add_trace(go.Scatter(x=df['CRT_TM_STR'], y=df['6ECMB101_ratio'], name='6ECMB101 비율', line=dict(color='#78716c', width=1)), row=4, col=1)
fig_trend.add_hline(y=50, line_dash="dash", line_color="red", row=4, col=1)

fig_trend.update_layout(
    height=900,
    template='plotly_dark',
    hovermode='x unified',
    showlegend=True,
    legend=dict(orientation='h', yanchor='bottom', y=1.02),
    xaxis4=dict(rangeslider=dict(visible=True))
)

# 2) 일별 차트 (subplot 2x2)
fig_daily = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        '일별 PAUSE 건수',
        '일별 JOB_2 최대값',
        '일별 6ECMB101 비율 (%)',
        '일별 AvgTime 최대값'
    ),
    vertical_spacing=0.15,
    horizontal_spacing=0.1
)

fig_daily.add_trace(go.Bar(x=daily['date_str'], y=daily['RAIL-TRANSFERPAUSED_WELL'], name='전체', marker_color='#3b82f6'), row=1, col=1)
fig_daily.add_trace(go.Bar(x=daily['date_str'], y=daily['6ECMB101_WELL'], name='6ECMB101', marker_color='#ef4444'), row=1, col=1)

colors_job = ['#ef4444' if v >= 400 else '#f97316' if v >= 300 else '#22c55e' for v in daily['CURRENT_M16A_3F_JOB_2_max']]
fig_daily.add_trace(go.Bar(x=daily['date_str'], y=daily['CURRENT_M16A_3F_JOB_2_max'], name='JOB_2 Max', marker_color=colors_job, showlegend=False), row=1, col=2)
fig_daily.add_hline(y=300, line_dash="dash", line_color="orange", row=1, col=2)
fig_daily.add_hline(y=400, line_dash="dash", line_color="red", row=1, col=2)

colors_ratio = ['#ef4444' if v >= 70 else '#f97316' if v >= 50 else '#22c55e' for v in daily['ratio']]
fig_daily.add_trace(go.Bar(x=daily['date_str'], y=daily['ratio'], name='비율', marker_color=colors_ratio, showlegend=False), row=2, col=1)
fig_daily.add_hline(y=50, line_dash="dash", line_color="red", row=2, col=1)

colors_time = ['#ef4444' if v >= 20 else '#f97316' if v >= 10 else '#a855f7' for v in daily['M16HUB_AVGTOTALTIME1MIN_max']]
fig_daily.add_trace(go.Bar(x=daily['date_str'], y=daily['M16HUB_AVGTOTALTIME1MIN_max'], name='AvgTime Max', marker_color=colors_time, showlegend=False), row=2, col=2)
fig_daily.add_hline(y=10, line_dash="dash", line_color="orange", row=2, col=2)
fig_daily.add_hline(y=20, line_dash="dash", line_color="red", row=2, col=2)

fig_daily.update_layout(
    height=700,
    template='plotly_dark',
    barmode='group',
    showlegend=True,
    legend=dict(orientation='h', yanchor='bottom', y=1.02)
)

# 3) 시간대별 차트
fig_hourly = make_subplots(
    rows=1, cols=2,
    subplot_titles=('시간대별 PAUSE 건수', '시간대별 JOB_2 평균'),
    horizontal_spacing=0.1
)

fig_hourly.add_trace(go.Bar(x=hourly['hour'], y=hourly['RAIL-TRANSFERPAUSED_WELL'], name='전체', marker_color='#3b82f6'), row=1, col=1)
fig_hourly.add_trace(go.Bar(x=hourly['hour'], y=hourly['6ECMB101_WELL'], name='6ECMB101', marker_color='#ef4444'), row=1, col=1)
fig_hourly.add_trace(go.Bar(x=hourly['hour'], y=hourly['CURRENT_M16A_3F_JOB_2_mean'], name='JOB_2', marker_color='#22c55e', showlegend=False), row=1, col=2)
fig_hourly.add_hline(y=300, line_dash="dash", line_color="red", row=1, col=2)

fig_hourly.update_layout(
    height=400,
    template='plotly_dark',
    barmode='group',
    xaxis=dict(dtick=1),
    xaxis2=dict(dtick=1)
)

# 4) 상관관계 히트맵
fig_corr = go.Figure(data=go.Heatmap(
    z=corr_matrix.values,
    x=['전체PAUSE', '6ECMB101', 'JOB_2', 'AvgTime'],
    y=['전체PAUSE', '6ECMB101', 'JOB_2', 'AvgTime'],
    colorscale='RdBu',
    zmid=0,
    text=np.round(corr_matrix.values, 2),
    texttemplate='%{text}',
    textfont={"size": 16}
))
fig_corr.update_layout(height=450, template='plotly_dark', title='상관관계 분석')

# 5) 산점도
fig_scatter = go.Figure()
fig_scatter.add_trace(go.Scatter(
    x=df['CURRENT_M16A_3F_JOB_2_mean'],
    y=df['6ECMB101_WELL'],
    mode='markers',
    marker=dict(
        size=5,
        color=df['M16HUB_AVGTOTALTIME1MIN_mean'].fillna(0),
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title='AvgTime')
    ),
    text=df['CRT_TM_STR'],
    hovertemplate='시간: %{text}<br>JOB_2: %{x:.1f}<br>6ECMB101: %{y}<extra></extra>'
))
fig_scatter.add_vline(x=300, line_dash="dash", line_color="red")
fig_scatter.add_hline(y=50, line_dash="dash", line_color="orange")
fig_scatter.update_layout(
    height=500,
    template='plotly_dark',
    title='JOB_2 vs 6ECMB101 (색상: AvgTime)',
    xaxis_title='JOB_2',
    yaxis_title='6ECMB101'
)

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
        .chart {{ background: #1e293b; border-radius: 10px; padding: 15px; }}
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

    <div id="trend" class="tab-content active"><div class="chart" id="chart_trend"></div></div>
    <div id="daily" class="tab-content"><div class="chart" id="chart_daily"></div></div>
    <div id="hourly" class="tab-content"><div class="chart" id="chart_hourly"></div></div>
    <div id="corr" class="tab-content"><div class="chart" id="chart_corr"></div></div>
    <div id="scatter" class="tab-content"><div class="chart" id="chart_scatter"></div></div>

    <script>
        function showTab(tabId) {{
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById(tabId).classList.add('active');
            window.dispatchEvent(new Event('resize'));
        }}

        Plotly.newPlot('chart_trend', {fig_trend.to_json()}.data, {fig_trend.to_json()}.layout, {{responsive: true}});
        Plotly.newPlot('chart_daily', {fig_daily.to_json()}.data, {fig_daily.to_json()}.layout, {{responsive: true}});
        Plotly.newPlot('chart_hourly', {fig_hourly.to_json()}.data, {fig_hourly.to_json()}.layout, {{responsive: true}});
        Plotly.newPlot('chart_corr', {fig_corr.to_json()}.data, {fig_corr.to_json()}.layout, {{responsive: true}});
        Plotly.newPlot('chart_scatter', {fig_scatter.to_json()}.data, {fig_scatter.to_json()}.layout, {{responsive: true}});
    </script>
</body>
</html>
'''

with open('RAIL_대시보드.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("저장 완료: RAIL_대시보드.html")