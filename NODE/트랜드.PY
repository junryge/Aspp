import pandas as pd
import plotly.graph_objects as go
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

# ========== 차트 생성 ==========

# 1) 트렌드 - PAUSE 건수 (한 화면)
fig_trend1 = go.Figure()
fig_trend1.add_trace(go.Scatter(x=df['CRT_TM'], y=df['RAIL-TRANSFERPAUSED_WELL'], name='전체 PAUSE', line=dict(color='#3b82f6', width=1.5)))
fig_trend1.add_trace(go.Scatter(x=df['CRT_TM'], y=df['6ECMB101_WELL'], name='6ECMB101', line=dict(color='#ef4444', width=1.5)))
fig_trend1.add_hline(y=100, line_dash="solid", line_color="red", annotation_text="경고선 (100)")
fig_trend1.update_layout(
    height=500,
    template='plotly_dark',
    title='PAUSE 건수 트렌드 (전체 vs 6ECMB101)',
    xaxis_title='시간',
    yaxis_title='건수',
    hovermode='x unified',
    xaxis=dict(rangeslider=dict(visible=True))
)

# 2) 트렌드 - JOB_2 (한 화면)
fig_trend2 = go.Figure()
fig_trend2.add_trace(go.Scatter(x=df['CRT_TM'], y=df['CURRENT_M16A_3F_JOB_2_mean'], name='JOB_2 평균', line=dict(color='#22c55e', width=1.5)))
fig_trend2.add_trace(go.Scatter(x=df['CRT_TM'], y=df['CURRENT_M16A_3F_JOB_2_max'], name='JOB_2 최대', line=dict(color='#f97316', width=1.5)))
fig_trend2.add_hline(y=300, line_dash="solid", line_color="red", annotation_text="급증 임계값 (300)")
fig_trend2.add_hline(y=400, line_dash="solid", line_color="darkred", annotation_text="위험 (400)")
fig_trend2.update_layout(
    height=500,
    template='plotly_dark',
    title='CURRENT_M16A_3F_JOB_2 트렌드',
    xaxis_title='시간',
    yaxis_title='JOB 수',
    hovermode='x unified',
    xaxis=dict(rangeslider=dict(visible=True))
)

# 3) 트렌드 - AvgTime (한 화면)
fig_trend3 = go.Figure()
fig_trend3.add_trace(go.Scatter(x=df['CRT_TM'], y=df['M16HUB_AVGTOTALTIME1MIN_mean'], name='AvgTime 평균', line=dict(color='#a855f7', width=1.5)))
fig_trend3.add_trace(go.Scatter(x=df['CRT_TM'], y=df['M16HUB_AVGTOTALTIME1MIN_max'], name='AvgTime 최대', line=dict(color='#ec4899', width=1.5)))
fig_trend3.add_hline(y=10, line_dash="solid", line_color="orange", annotation_text="경고 (10분)")
fig_trend3.add_hline(y=20, line_dash="solid", line_color="red", annotation_text="위험 (20분)")
fig_trend3.update_layout(
    height=500,
    template='plotly_dark',
    title='M16HUB_AVGTOTALTIME1MIN 트렌드',
    xaxis_title='시간',
    yaxis_title='시간 (분)',
    hovermode='x unified',
    xaxis=dict(rangeslider=dict(visible=True))
)

# 4) 트렌드 - 비율 (한 화면)
fig_trend4 = go.Figure()
fig_trend4.add_trace(go.Scatter(x=df['CRT_TM'], y=df['6ECMB101_ratio'], name='6ECMB101 비율', line=dict(color='#78716c', width=1.5)))
fig_trend4.add_hline(y=50, line_dash="solid", line_color="red", annotation_text="50%")
fig_trend4.update_layout(
    height=500,
    template='plotly_dark',
    title='6ECMB101 비율 (%) 트렌드',
    xaxis_title='시간',
    yaxis_title='비율 (%)',
    hovermode='x unified',
    xaxis=dict(rangeslider=dict(visible=True))
)

# 5) 일별 차트 - PAUSE
fig_daily1 = go.Figure()
fig_daily1.add_trace(go.Bar(x=daily['date_str'], y=daily['RAIL-TRANSFERPAUSED_WELL'], name='전체', marker_color='#3b82f6'))
fig_daily1.add_trace(go.Bar(x=daily['date_str'], y=daily['6ECMB101_WELL'], name='6ECMB101', marker_color='#ef4444'))
fig_daily1.update_layout(height=500, template='plotly_dark', title='일별 PAUSE 건수', barmode='group')

# 6) 일별 차트 - JOB_2
colors_job = ['#ef4444' if v >= 400 else '#f97316' if v >= 300 else '#22c55e' for v in daily['CURRENT_M16A_3F_JOB_2_max']]
fig_daily2 = go.Figure()
fig_daily2.add_trace(go.Bar(x=daily['date_str'], y=daily['CURRENT_M16A_3F_JOB_2_max'], name='JOB_2 Max', marker_color=colors_job))
fig_daily2.add_hline(y=300, line_dash="solid", line_color="orange")
fig_daily2.add_hline(y=400, line_dash="solid", line_color="red")
fig_daily2.update_layout(height=500, template='plotly_dark', title='일별 JOB_2 최대값')

# 7) 시간대별
fig_hourly = go.Figure()
fig_hourly.add_trace(go.Bar(x=hourly['hour'], y=hourly['RAIL-TRANSFERPAUSED_WELL'], name='전체', marker_color='#3b82f6'))
fig_hourly.add_trace(go.Bar(x=hourly['hour'], y=hourly['6ECMB101_WELL'], name='6ECMB101', marker_color='#ef4444'))
fig_hourly.update_layout(height=500, template='plotly_dark', title='시간대별 평균 PAUSE 건수', barmode='group', xaxis=dict(dtick=1))

# 8) 상관관계
fig_corr = go.Figure(data=go.Heatmap(
    z=corr_matrix.values,
    x=['전체PAUSE', '6ECMB101', 'JOB_2', 'AvgTime'],
    y=['전체PAUSE', '6ECMB101', 'JOB_2', 'AvgTime'],
    colorscale='RdBu', zmid=0,
    text=np.round(corr_matrix.values, 2), texttemplate='%{text}', textfont={"size": 16}
))
fig_corr.update_layout(height=500, template='plotly_dark', title='상관관계 분석')

# 9) 산점도
fig_scatter = go.Figure()
fig_scatter.add_trace(go.Scatter(
    x=df['CURRENT_M16A_3F_JOB_2_mean'], y=df['6ECMB101_WELL'], mode='markers',
    marker=dict(size=5, color=df['M16HUB_AVGTOTALTIME1MIN_mean'].fillna(0), colorscale='Viridis', showscale=True, colorbar=dict(title='AvgTime')),
    text=df['CRT_TM_STR'], hovertemplate='시간: %{text}<br>JOB_2: %{x:.1f}<br>6ECMB101: %{y}<extra></extra>'
))
fig_scatter.add_vline(x=300, line_dash="solid", line_color="red")
fig_scatter.add_hline(y=50, line_dash="solid", line_color="orange")
fig_scatter.update_layout(height=500, template='plotly_dark', title='JOB_2 vs 6ECMB101 (색상: AvgTime)', xaxis_title='JOB_2', yaxis_title='6ECMB101')

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
        .tabs {{ display: flex; background: #1e293b; padding: 0 30px; flex-wrap: wrap; }}
        .tab {{ padding: 15px 20px; cursor: pointer; border-bottom: 3px solid transparent; color: #94a3b8; font-size: 14px; }}
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
        <div class="card"><h3>📊 총 PAUSE</h3><div class="value">{total_pause:,.0f}</div></div>
        <div class="card"><h3>🔴 6ECMB101</h3><div class="value">{total_6ecmb:,.0f}</div><div class="sub">{ratio_6ecmb:.1f}%</div></div>
        <div class="card {'warning' if max_job2 >= 400 else ''}"><h3>📈 JOB_2 최대</h3><div class="value">{max_job2:.0f}</div></div>
        <div class="card {'danger' if max_avgtime >= 20 else 'warning' if max_avgtime >= 10 else ''}"><h3>⏱️ AvgTime 최대</h3><div class="value">{max_avgtime:.1f}분</div></div>
        <div class="card"><h3>⚠️ 이상치</h3><div class="value">{len(anomaly_df):,}건</div></div>
    </div>

    <div class="tabs">
        <div class="tab active" onclick="showTab('t1')">📈 PAUSE 트렌드</div>
        <div class="tab" onclick="showTab('t2')">📈 JOB_2 트렌드</div>
        <div class="tab" onclick="showTab('t3')">📈 AvgTime 트렌드</div>
        <div class="tab" onclick="showTab('t4')">📈 비율 트렌드</div>
        <div class="tab" onclick="showTab('d1')">📊 일별 PAUSE</div>
        <div class="tab" onclick="showTab('d2')">📊 일별 JOB_2</div>
        <div class="tab" onclick="showTab('h1')">🕐 시간대별</div>
        <div class="tab" onclick="showTab('corr')">🔗 상관관계</div>
        <div class="tab" onclick="showTab('scatter')">🔍 산점도</div>
    </div>

    <div id="t1" class="tab-content active"><div class="chart" id="chart_t1"></div></div>
    <div id="t2" class="tab-content"><div class="chart" id="chart_t2"></div></div>
    <div id="t3" class="tab-content"><div class="chart" id="chart_t3"></div></div>
    <div id="t4" class="tab-content"><div class="chart" id="chart_t4"></div></div>
    <div id="d1" class="tab-content"><div class="chart" id="chart_d1"></div></div>
    <div id="d2" class="tab-content"><div class="chart" id="chart_d2"></div></div>
    <div id="h1" class="tab-content"><div class="chart" id="chart_h1"></div></div>
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

        Plotly.newPlot('chart_t1', {fig_trend1.to_json()}.data, {fig_trend1.to_json()}.layout, {{responsive: true}});
        Plotly.newPlot('chart_t2', {fig_trend2.to_json()}.data, {fig_trend2.to_json()}.layout, {{responsive: true}});
        Plotly.newPlot('chart_t3', {fig_trend3.to_json()}.data, {fig_trend3.to_json()}.layout, {{responsive: true}});
        Plotly.newPlot('chart_t4', {fig_trend4.to_json()}.data, {fig_trend4.to_json()}.layout, {{responsive: true}});
        Plotly.newPlot('chart_d1', {fig_daily1.to_json()}.data, {fig_daily1.to_json()}.layout, {{responsive: true}});
        Plotly.newPlot('chart_d2', {fig_daily2.to_json()}.data, {fig_daily2.to_json()}.layout, {{responsive: true}});
        Plotly.newPlot('chart_h1', {fig_hourly.to_json()}.data, {fig_hourly.to_json()}.layout, {{responsive: true}});
        Plotly.newPlot('chart_corr', {fig_corr.to_json()}.data, {fig_corr.to_json()}.layout, {{responsive: true}});
        Plotly.newPlot('chart_scatter', {fig_scatter.to_json()}.data, {fig_scatter.to_json()}.layout, {{responsive: true}});
    </script>
</body>
</html>
'''

with open('RAIL_대시보드.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("저장 완료: RAIL_대시보드.html")