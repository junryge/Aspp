import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# 1. CSV 읽기
df = pd.read_csv('RAL', encoding='utf-8')
df = df.dropna(subset=['CRT_TM'])
df['CRT_TM'] = pd.to_datetime(df['CRT_TM'])

# 날짜 포맷 변환 (202601010000 형식)
df['CRT_TM_STR'] = df['CRT_TM'].dt.strftime('%Y%m%d%H%M')

# 2. 파생 컬럼 생성
df['6ECMB101_ratio'] = (df['6ECMB101_WELL'] / df['RAIL-TRANSFERPAUSED_WELL'] * 100).fillna(0)
df['date'] = df['CRT_TM'].dt.date
df['date_str'] = df['CRT_TM'].dt.strftime('%Y%m%d')
df['hour'] = df['CRT_TM'].dt.hour
df['weekday'] = df['CRT_TM'].dt.dayofweek

# 3. 이상치 탐지 (JOB_2 >= 400 또는 6ECMB101 비율 >= 70%)
df['is_anomaly'] = ((df['CURRENT_M16A_3F_JOB_2_max'] >= 400) | (df['6ECMB101_ratio'] >= 70))

# 4. 일별 집계
daily = df.groupby(['date', 'date_str']).agg({
    'RAIL-TRANSFERPAUSED_WELL': 'sum',
    '6ECMB101_WELL': 'sum',
    'CURRENT_M16A_3F_JOB_2_max': 'max',
    'CURRENT_M16A_3F_JOB_2_mean': 'mean',
    'M16HUB_AVGTOTALTIME1MIN_max': 'max',
    'M16HUB_AVGTOTALTIME1MIN_mean': 'mean'
}).reset_index()
daily['ratio'] = daily['6ECMB101_WELL'] / daily['RAIL-TRANSFERPAUSED_WELL'] * 100

# 5. 시간대별 집계
hourly = df.groupby('hour').agg({
    'RAIL-TRANSFERPAUSED_WELL': 'mean',
    '6ECMB101_WELL': 'mean',
    'CURRENT_M16A_3F_JOB_2_mean': 'mean'
}).reset_index()

# 6. 요일별 집계
weekday_names = ['월', '화', '수', '목', '금', '토', '일']
weekday_agg = df.groupby('weekday').agg({
    'RAIL-TRANSFERPAUSED_WELL': 'mean',
    '6ECMB101_WELL': 'mean'
}).reset_index()
weekday_agg['weekday_name'] = weekday_agg['weekday'].apply(lambda x: weekday_names[x])

# 7. 상관관계 계산
corr_cols = ['RAIL-TRANSFERPAUSED_WELL', '6ECMB101_WELL', 'CURRENT_M16A_3F_JOB_2_mean', 'M16HUB_AVGTOTALTIME1MIN_mean']
corr_matrix = df[corr_cols].corr()

# ========== HTML 대시보드 생성 ==========

# 메인 트렌드 그래프 (4개 subplot)
fig1 = make_subplots(
    rows=4, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    subplot_titles=(
        '📊 RAIL-TRANSFERPAUSED 발생 건수 (전체 vs 6ECMB101)',
        '📈 CURRENT_M16A_3F_JOB_2 (평균/최대) - 급증 임계값: 300',
        '⏱️ M16HUB_AVGTOTALTIME1MIN (평균/최대)',
        '📉 6ECMB101 비율 (%) - 전체 PAUSE 대비'
    ),
    row_heights=[0.25, 0.25, 0.25, 0.25]
)

# 1) PAUSE 건수
fig1.add_trace(go.Scatter(x=df['CRT_TM_STR'], y=df['RAIL-TRANSFERPAUSED_WELL'], name='전체 PAUSE', line=dict(color='blue', width=1), opacity=0.7), row=1, col=1)
fig1.add_trace(go.Scatter(x=df['CRT_TM_STR'], y=df['6ECMB101_WELL'], name='6ECMB101', line=dict(color='red', width=1), opacity=0.7), row=1, col=1)

# 2) JOB_2
fig1.add_trace(go.Scatter(x=df['CRT_TM_STR'], y=df['CURRENT_M16A_3F_JOB_2_mean'], name='JOB_2 평균', line=dict(color='green', width=1), opacity=0.7), row=2, col=1)
fig1.add_trace(go.Scatter(x=df['CRT_TM_STR'], y=df['CURRENT_M16A_3F_JOB_2_max'], name='JOB_2 최대', line=dict(color='orange', width=1), opacity=0.5), row=2, col=1)
fig1.add_hline(y=300, line_dash="dash", line_color="red", annotation_text="급증 임계값 (300)", row=2, col=1)
fig1.add_hline(y=400, line_dash="dash", line_color="darkred", annotation_text="위험 (400)", row=2, col=1)

# 3) AvgTime
fig1.add_trace(go.Scatter(x=df['CRT_TM_STR'], y=df['M16HUB_AVGTOTALTIME1MIN_mean'], name='AvgTime 평균', line=dict(color='purple', width=1), opacity=0.7), row=3, col=1)
fig1.add_trace(go.Scatter(x=df['CRT_TM_STR'], y=df['M16HUB_AVGTOTALTIME1MIN_max'], name='AvgTime 최대', line=dict(color='magenta', width=1), opacity=0.5), row=3, col=1)
fig1.add_hline(y=10, line_dash="dash", line_color="orange", annotation_text="경고 (10분)", row=3, col=1)
fig1.add_hline(y=20, line_dash="dash", line_color="red", annotation_text="위험 (20분)", row=3, col=1)

# 4) 비율
fig1.add_trace(go.Scatter(x=df['CRT_TM_STR'], y=df['6ECMB101_ratio'], name='6ECMB101 비율', line=dict(color='brown', width=1), opacity=0.7), row=4, col=1)
fig1.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="50%", row=4, col=1)

# 이상치 마커
anomaly_df = df[df['is_anomaly']]
fig1.add_trace(go.Scatter(x=anomaly_df['CRT_TM_STR'], y=anomaly_df['CURRENT_M16A_3F_JOB_2_max'], mode='markers', name='⚠️ 이상치', marker=dict(color='red', size=8, symbol='x')), row=2, col=1)

fig1.update_layout(
    height=1200,
    title_text='<b>RAIL-TRANSFERPAUSED 트렌드 분석 대시보드</b><br><sup>기간: 202511250000 ~ 202601060700 (10분 단위)</sup>',
    showlegend=True,
    hovermode='x unified',
    xaxis4=dict(
        rangeslider=dict(visible=True),
        type='category'
    )
)

# ========== 일별 요약 그래프 ==========
fig2 = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        '📊 일별 PAUSE 건수', 
        '📈 일별 JOB_2 최대값',
        '📉 일별 6ECMB101 비율 (%)',
        '⏱️ 일별 AvgTime 최대값'
    ),
    vertical_spacing=0.12,
    horizontal_spacing=0.08
)

# 1) 일별 PAUSE
fig2.add_trace(go.Bar(x=daily['date_str'], y=daily['RAIL-TRANSFERPAUSED_WELL'], name='전체', marker_color='blue', opacity=0.7), row=1, col=1)
fig2.add_trace(go.Bar(x=daily['date_str'], y=daily['6ECMB101_WELL'], name='6ECMB101', marker_color='red', opacity=0.7), row=1, col=1)

# 2) 일별 JOB_2 Max
colors_job = ['red' if v >= 400 else 'orange' if v >= 300 else 'green' for v in daily['CURRENT_M16A_3F_JOB_2_max']]
fig2.add_trace(go.Bar(x=daily['date_str'], y=daily['CURRENT_M16A_3F_JOB_2_max'], name='JOB_2 Max', marker_color=colors_job), row=1, col=2)
fig2.add_hline(y=300, line_dash="dash", line_color="orange", row=1, col=2)
fig2.add_hline(y=400, line_dash="dash", line_color="red", row=1, col=2)

# 3) 일별 비율
colors_ratio = ['red' if v >= 70 else 'orange' if v >= 50 else 'green' for v in daily['ratio']]
fig2.add_trace(go.Bar(x=daily['date_str'], y=daily['ratio'], name='비율', marker_color=colors_ratio), row=2, col=1)
fig2.add_hline(y=50, line_dash="dash", line_color="red", row=2, col=1)

# 4) 일별 AvgTime Max
colors_time = ['red' if v >= 20 else 'orange' if v >= 10 else 'purple' for v in daily['M16HUB_AVGTOTALTIME1MIN_max']]
fig2.add_trace(go.Bar(x=daily['date_str'], y=daily['M16HUB_AVGTOTALTIME1MIN_max'], name='AvgTime Max', marker_color=colors_time), row=2, col=2)
fig2.add_hline(y=10, line_dash="dash", line_color="orange", row=2, col=2)
fig2.add_hline(y=20, line_dash="dash", line_color="red", row=2, col=2)

fig2.update_layout(
    height=800,
    title_text='<b>일별 요약 대시보드</b>',
    showlegend=True,
    barmode='group'
)

# ========== 시간대별/요일별 패턴 ==========
fig3 = make_subplots(
    rows=1, cols=3,
    subplot_titles=(
        '🕐 시간대별 PAUSE 패턴',
        '🕐 시간대별 JOB_2 패턴',
        '📅 요일별 6ECMB101 패턴'
    ),
    horizontal_spacing=0.08
)

# 시간대별 PAUSE
fig3.add_trace(go.Bar(x=hourly['hour'], y=hourly['RAIL-TRANSFERPAUSED_WELL'], name='전체', marker_color='blue', opacity=0.7), row=1, col=1)
fig3.add_trace(go.Bar(x=hourly['hour'], y=hourly['6ECMB101_WELL'], name='6ECMB101', marker_color='red', opacity=0.7), row=1, col=1)

# 시간대별 JOB_2
fig3.add_trace(go.Bar(x=hourly['hour'], y=hourly['CURRENT_M16A_3F_JOB_2_mean'], name='JOB_2 평균', marker_color='green', opacity=0.7), row=1, col=2)
fig3.add_hline(y=300, line_dash="dash", line_color="red", row=1, col=2)

# 요일별
fig3.add_trace(go.Bar(x=weekday_agg['weekday_name'], y=weekday_agg['RAIL-TRANSFERPAUSED_WELL'], name='전체', marker_color='blue', opacity=0.7), row=1, col=3)
fig3.add_trace(go.Bar(x=weekday_agg['weekday_name'], y=weekday_agg['6ECMB101_WELL'], name='6ECMB101', marker_color='red', opacity=0.7), row=1, col=3)

fig3.update_layout(
    height=500,
    title_text='<b>시간대별/요일별 패턴 분석</b>',
    showlegend=True,
    barmode='group'
)

# ========== 상관관계 히트맵 ==========
fig4 = go.Figure(data=go.Heatmap(
    z=corr_matrix.values,
    x=['전체PAUSE', '6ECMB101', 'JOB_2평균', 'AvgTime평균'],
    y=['전체PAUSE', '6ECMB101', 'JOB_2평균', 'AvgTime평균'],
    colorscale='RdBu',
    zmid=0,
    text=np.round(corr_matrix.values, 2),
    texttemplate='%{text}',
    textfont={"size": 14},
    hovertemplate='%{x} vs %{y}: %{z:.3f}<extra></extra>'
))

fig4.update_layout(
    height=500,
    title_text='<b>상관관계 분석</b><br><sup>1에 가까울수록 양의 상관관계, -1에 가까울수록 음의 상관관계</sup>'
)

# ========== 산점도 (JOB_2 vs PAUSE) ==========
fig5 = go.Figure()
fig5.add_trace(go.Scatter(
    x=df['CURRENT_M16A_3F_JOB_2_mean'],
    y=df['6ECMB101_WELL'],
    mode='markers',
    marker=dict(
        size=5,
        color=df['M16HUB_AVGTOTALTIME1MIN_mean'],
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title='AvgTime')
    ),
    text=df['CRT_TM_STR'],
    hovertemplate='시간: %{text}<br>JOB_2: %{x:.1f}<br>6ECMB101: %{y}<br>AvgTime: %{marker.color:.1f}<extra></extra>'
))

fig5.add_vline(x=300, line_dash="dash", line_color="red", annotation_text="JOB_2 급증 임계값")
fig5.add_hline(y=50, line_dash="dash", line_color="orange", annotation_text="6ECMB101 경고")

fig5.update_layout(
    height=600,
    title_text='<b>JOB_2 vs 6ECMB101 산점도</b><br><sup>색상: AvgTime (진할수록 지연 심함)</sup>',
    xaxis_title='CURRENT_M16A_3F_JOB_2 평균',
    yaxis_title='6ECMB101 PAUSE 건수'
)

# ========== HTML 파일 생성 ==========
html_content = f'''
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAIL-TRANSFERPAUSED 분석 대시보드</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Malgun Gothic', sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 28px;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .card h3 {{
            margin: 0 0 10px 0;
            color: #666;
            font-size: 14px;
        }}
        .card .value {{
            font-size: 32px;
            font-weight: bold;
            color: #1e3c72;
        }}
        .card .sub {{
            font-size: 12px;
            color: #999;
            margin-top: 5px;
        }}
        .card.warning .value {{
            color: #ff9800;
        }}
        .card.danger .value {{
            color: #f44336;
        }}
        .chart-container {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .section-title {{
            font-size: 18px;
            font-weight: bold;
            color: #333;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #1e3c72;
        }}
        .insight-box {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
            border-radius: 0 10px 10px 0;
        }}
        .insight-box.danger {{
            background: #f8d7da;
            border-left-color: #dc3545;
        }}
        .insight-box h4 {{
            margin: 0 0 10px 0;
            color: #856404;
        }}
        .insight-box.danger h4 {{
            color: #721c24;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚦 RAIL-TRANSFERPAUSED 분석 대시보드</h1>
        <p>📅 분석 기간: 202511250000 ~ 202601060700 | 📊 데이터: 10분 단위 {len(df):,}건</p>
    </div>

    <div class="summary-cards">
        <div class="card">
            <h3>📊 총 PAUSE 건수</h3>
            <div class="value">{df['RAIL-TRANSFERPAUSED_WELL'].sum():,.0f}</div>
            <div class="sub">전체 기간 합계</div>
        </div>
        <div class="card">
            <h3>🔴 6ECMB101 건수</h3>
            <div class="value">{df['6ECMB101_WELL'].sum():,.0f}</div>
            <div class="sub">전체 대비 {df['6ECMB101_WELL'].sum()/df['RAIL-TRANSFERPAUSED_WELL'].sum()*100:.1f}%</div>
        </div>
        <div class="card {'warning' if df['CURRENT_M16A_3F_JOB_2_max'].max() >= 400 else ''}">
            <h3>📈 JOB_2 최대값</h3>
            <div class="value">{df['CURRENT_M16A_3F_JOB_2_max'].max():.0f}</div>
            <div class="sub">급증 임계값: 300</div>
        </div>
        <div class="card {'danger' if df['M16HUB_AVGTOTALTIME1MIN_max'].max() >= 20 else 'warning' if df['M16HUB_AVGTOTALTIME1MIN_max'].max() >= 10 else ''}">
            <h3>⏱️ AvgTime 최대값</h3>
            <div class="value">{df['M16HUB_AVGTOTALTIME1MIN_max'].max():.1f}분</div>
            <div class="sub">경고: 10분 / 위험: 20분</div>
        </div>
        <div class="card">
            <h3>⚠️ 이상치 발생</h3>
            <div class="value">{len(anomaly_df):,}건</div>
            <div class="sub">JOB≥400 또는 비율≥70%</div>
        </div>
        <div class="card">
            <h3>📅 분석 일수</h3>
            <div class="value">{len(daily)}일</div>
            <div class="sub">10분 단위 {len(df):,}건</div>
        </div>
    </div>

    <div class="insight-box danger">
        <h4>🚨 주요 발견 사항</h4>
        <ul>
            <li>6ECMB101 장비가 전체 PAUSE의 <b>{df['6ECMB101_WELL'].sum()/df['RAIL-TRANSFERPAUSED_WELL'].sum()*100:.1f}%</b>를 차지 → 집중 점검 필요</li>
            <li>JOB_2 최대값 <b>{df['CURRENT_M16A_3F_JOB_2_max'].max():.0f}</b> 기록 (급증 임계값 300 초과)</li>
            <li>AvgTime 최대 <b>{df['M16HUB_AVGTOTALTIME1MIN_max'].max():.1f}분</b> → 심각한 지연 발생 이력</li>
        </ul>
    </div>

    <div class="chart-container">
        <div class="section-title">📈 전체 기간 트렌드 (드래그하여 줌, 더블클릭하여 리셋)</div>
        <div id="chart1"></div>
    </div>

    <div class="chart-container">
        <div class="section-title">📊 일별 요약</div>
        <div id="chart2"></div>
    </div>

    <div class="chart-container">
        <div class="section-title">🕐 시간대별/요일별 패턴</div>
        <div id="chart3"></div>
    </div>

    <div class="chart-container">
        <div class="section-title">🔗 상관관계 분석</div>
        <div id="chart4"></div>
    </div>

    <div class="chart-container">
        <div class="section-title">🔍 JOB_2 vs 6ECMB101 산점도 (색상: AvgTime)</div>
        <div id="chart5"></div>
    </div>

    <script>
        var chart1 = {fig1.to_json()};
        var chart2 = {fig2.to_json()};
        var chart3 = {fig3.to_json()};
        var chart4 = {fig4.to_json()};
        var chart5 = {fig5.to_json()};
        
        Plotly.newPlot('chart1', chart1.data, chart1.layout, {{responsive: true}});
        Plotly.newPlot('chart2', chart2.data, chart2.layout, {{responsive: true}});
        Plotly.newPlot('chart3', chart3.data, chart3.layout, {{responsive: true}});
        Plotly.newPlot('chart4', chart4.data, chart4.layout, {{responsive: true}});
        Plotly.newPlot('chart5', chart5.data, chart5.layout, {{responsive: true}});
    </script>
</body>
</html>
'''

# HTML 파일 저장
with open('RAIL_TRANSFERPAUSED_대시보드.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("저장 완료: RAIL_TRANSFERPAUSED_대시보드.html")