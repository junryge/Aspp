# -*- coding: utf-8 -*-
"""
V12 평가 결과 대시보드 생성기
CSV 파일을 로드하여 HTML 대시보드 생성
"""

import pandas as pd
import json
import os
from datetime import datetime

def load_csv(filepath):
    """CSV 파일 로드 (여러 인코딩 시도)"""
    encodings = ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr']
    for enc in encodings:
        try:
            return pd.read_csv(filepath, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"파일 인코딩을 알 수 없습니다: {filepath}")

def analyze_results(df):
    """평가 결과 통계 분석"""
    stats = {
        'total': len(df),
        'actual_breach': len(df[df['실제_돌파여부'] == 1]),
        'pred_breach': len(df[df['최종예측_돌파여부'] == 1]),
    }
    
    # 예측상태별 집계
    status_counts = df['예측상태'].value_counts().to_dict()
    stats['TP'] = status_counts.get('정상예측_TP', 0)
    stats['TN'] = status_counts.get('정상예측_TN', 0)
    stats['FN_10min'] = status_counts.get('FN_10분전예측', 0)
    stats['FN_miss'] = status_counts.get('FN_완전놓침', 0)
    stats['FP_10min'] = status_counts.get('FP_10분후돌파', 0)
    stats['FP_false'] = status_counts.get('FP_잘못된경고', 0)
    
    # 성능 지표
    total_positive = stats['TP'] + stats['FN_10min'] + stats['FN_miss']
    stats['recall'] = stats['TP'] / total_positive * 100 if total_positive > 0 else 0
    stats['precision'] = stats['TP'] / (stats['TP'] + stats['FP_10min'] + stats['FP_false']) * 100 if (stats['TP'] + stats['FP_10min'] + stats['FP_false']) > 0 else 0
    
    return stats

def generate_dashboard_html(df, stats, output_path, title="V12 TOTALCNT 예측 대시보드"):
    """HTML 대시보드 생성"""
    
    # 데이터를 JSON으로 변환
    data_json = []
    for idx, row in df.iterrows():
        data_json.append({
            'idx': idx,
            'time': str(row.get('현재시간', '')),
            'pred_range': str(row.get('예측범위', '')),
            'current': float(row.get('현재_TOTALCNT', 0)),
            'actual_max': float(row.get('실제_최대값', 0)),
            'actual_breach': int(row.get('실제_돌파여부', 0)),
            'xgb_target': float(row.get('XGB_타겟_예측값', 0)),
            'xgb_important': float(row.get('XGB_중요_예측값', 0)),
            'xgb_auxiliary': float(row.get('XGB_보조_예측값', 0)),
            'lgbm_target_prob': float(row.get('LGBM_타겟_확률', 0)),
            'lgbm_important_prob': float(row.get('LGBM_중요_확률', 0)),
            'lgbm_auxiliary_prob': float(row.get('LGBM_보조_확률', 0)),
            'votes': int(row.get('투표수_6개중', 0)),
            'pred_breach': int(row.get('최종예측_돌파여부', 0)),
            'status': str(row.get('예측상태', ''))
        })
    
    # 기간 추출
    period_start = df['현재시간'].iloc[0] if len(df) > 0 else ''
    period_end = df['현재시간'].iloc[-1] if len(df) > 0 else ''
    
    html = f'''<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation"></script>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&family=Orbitron:wght@400;700&display=swap" rel="stylesheet">
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'Noto Sans KR', sans-serif; background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3e 50%, #0f2840 100%); min-height: 100vh; padding: 30px; color: #fff; }}
.container {{ max-width: 1800px; margin: 0 auto; }}

/* 헤더 */
.header {{ text-align: center; margin-bottom: 40px; padding: 30px; background: rgba(255,255,255,0.03); border-radius: 20px; border: 1px solid rgba(0,212,255,0.2); }}
.header h1 {{ font-family: 'Orbitron', sans-serif; font-size: 2.2rem; font-weight: 700; background: linear-gradient(90deg, #00d4ff, #00ff88); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 10px; }}
.header .subtitle {{ font-size: 1rem; color: #8892b0; }}
.header .period {{ margin-top: 15px; padding: 10px 25px; background: rgba(0,212,255,0.1); border-radius: 30px; display: inline-block; font-size: 0.9rem; color: #00d4ff; border: 1px solid rgba(0,212,255,0.3); }}

/* 성능 카드 */
.stats-grid {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 20px; margin-bottom: 30px; }}
.stat-card {{ background: rgba(255,255,255,0.05); border-radius: 16px; padding: 25px 20px; text-align: center; border: 1px solid rgba(255,255,255,0.1); transition: all 0.3s ease; cursor: pointer; }}
.stat-card:hover {{ transform: translateY(-5px); border-color: rgba(0,212,255,0.5); box-shadow: 0 10px 40px rgba(0,212,255,0.2); }}
.stat-card.highlight {{ background: linear-gradient(135deg, rgba(0,255,136,0.15), rgba(0,212,255,0.15)); border-color: rgba(0,255,136,0.4); }}
.stat-card.warning {{ background: linear-gradient(135deg, rgba(255,107,107,0.15), rgba(249,115,22,0.15)); border-color: rgba(255,107,107,0.4); }}
.stat-value {{ font-family: 'Orbitron', sans-serif; font-size: 2.2rem; font-weight: 700; margin-bottom: 8px; }}
.stat-value.green {{ color: #00ff88; }}
.stat-value.red {{ color: #ff4466; }}
.stat-value.yellow {{ color: #ffcc00; }}
.stat-value.gray {{ color: #888; }}
.stat-label {{ font-size: 0.9rem; color: #8892b0; margin-bottom: 5px; }}
.stat-sub {{ font-size: 0.8rem; color: #666; }}

/* 차트 영역 */
.chart-section {{ background: rgba(255,255,255,0.03); border-radius: 20px; padding: 25px; margin-bottom: 30px; border: 1px solid rgba(255,255,255,0.1); }}
.chart-title {{ font-size: 1.2rem; font-weight: 600; color: #00d4ff; margin-bottom: 20px; display: flex; align-items: center; gap: 10px; }}
.chart-container {{ height: 400px; }}
.chart-controls {{ display: flex; gap: 15px; margin-bottom: 20px; flex-wrap: wrap; align-items: center; }}
.chart-controls label {{ color: #888; font-size: 0.9rem; }}
.chart-controls select, .chart-controls input {{ background: #1a1a2e; border: 1px solid #333; color: #fff; padding: 8px 12px; border-radius: 8px; }}
.chart-controls button {{ background: linear-gradient(135deg, #00d4ff, #0099cc); border: none; color: #000; padding: 10px 20px; border-radius: 8px; cursor: pointer; font-weight: 600; transition: all 0.2s; }}
.chart-controls button:hover {{ transform: scale(1.05); box-shadow: 0 0 20px rgba(0,212,255,0.4); }}

/* 이벤트 리스트 */
.event-section {{ background: rgba(255,255,255,0.03); border-radius: 20px; padding: 25px; border: 1px solid rgba(255,255,255,0.1); }}
.event-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }}
.event-title {{ font-size: 1.2rem; font-weight: 600; color: #ff6b35; }}
.event-count {{ color: #00ff88; font-size: 1rem; }}
.event-list {{ max-height: 500px; overflow-y: auto; }}
.event-item {{ display: grid; grid-template-columns: 60px 140px 100px 100px 100px 80px 120px; gap: 10px; padding: 12px 15px; border-bottom: 1px solid #222; cursor: pointer; transition: all 0.2s; font-size: 0.85rem; align-items: center; }}
.event-item:hover {{ background: rgba(0,212,255,0.1); }}
.event-item.header {{ background: #15151f; font-weight: bold; color: #888; cursor: default; position: sticky; top: 0; }}
.event-item.header:hover {{ background: #15151f; }}
.event-num {{ color: #00d4ff; }}
.event-time {{ color: #aaa; }}
.event-val {{ text-align: right; }}
.event-val.current {{ color: #00ff88; }}
.event-val.actual {{ color: #00d4ff; }}
.event-val.pred {{ color: #ff6b35; }}
.event-status {{ text-align: center; padding: 4px 8px; border-radius: 6px; font-size: 0.75rem; }}
.event-status.tp {{ background: #1a3a1a; color: #00ff88; }}
.event-status.tn {{ background: #1a1a3a; color: #3b82f6; }}
.event-status.fn {{ background: #3a1a1a; color: #ff4466; }}
.event-status.fp {{ background: #3a2a1a; color: #ffcc00; }}

/* 모달 */
.modal {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.9); z-index: 1000; justify-content: center; align-items: center; }}
.modal.show {{ display: flex; }}
.modal-content {{ background: #12121a; border-radius: 20px; padding: 30px; width: 95%; max-width: 1200px; max-height: 90vh; overflow-y: auto; border: 1px solid #333; }}
.modal-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; padding-bottom: 15px; border-bottom: 1px solid #333; }}
.modal-title {{ font-size: 1.3rem; font-weight: bold; color: #00d4ff; }}
.modal-close {{ background: none; border: none; color: #888; font-size: 2rem; cursor: pointer; }}
.modal-close:hover {{ color: #fff; }}
.modal-chart {{ height: 450px; margin-bottom: 20px; }}
.modal-info {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 15px; }}
.modal-stat {{ background: #0a0a12; padding: 15px; border-radius: 10px; text-align: center; }}
.modal-stat-label {{ font-size: 0.8rem; color: #666; margin-bottom: 5px; }}
.modal-stat-value {{ font-size: 1.3rem; font-weight: bold; }}

/* 필터 탭 */
.filter-tabs {{ display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }}
.filter-tab {{ padding: 10px 20px; border-radius: 8px; border: 2px solid #333; background: #1a1a2e; color: #888; cursor: pointer; transition: all 0.2s; font-weight: 500; }}
.filter-tab:hover {{ border-color: #00d4ff; color: #fff; }}
.filter-tab.active {{ background: linear-gradient(135deg, #00d4ff, #0099cc); border-color: #00d4ff; color: #000; }}
.filter-tab.active.warn {{ background: linear-gradient(135deg, #ff4466, #cc2244); border-color: #ff4466; color: #fff; }}

/* 범례 */
.legend {{ display: flex; gap: 20px; margin-top: 15px; justify-content: center; flex-wrap: wrap; }}
.legend-item {{ display: flex; align-items: center; gap: 8px; font-size: 0.85rem; color: #888; }}
.legend-color {{ width: 12px; height: 12px; border-radius: 3px; }}

@media (max-width: 1200px) {{ .stats-grid {{ grid-template-columns: repeat(3, 1fr); }} }}
@media (max-width: 800px) {{ .stats-grid {{ grid-template-columns: repeat(2, 1fr); }} .event-item {{ font-size: 0.75rem; }} }}
</style>
</head>
<body>
<div class="container">
    <!-- 헤더 -->
    <div class="header">
        <h1>📊 {title}</h1>
        <p class="subtitle">M14 TOTALCNT 30분 내 리미트(1700) 초과 예측 시스템</p>
        <div class="period">📅 {period_start} ~ {period_end}</div>
    </div>
    
    <!-- 성능 요약 카드 -->
    <div class="stats-grid">
        <div class="stat-card highlight" onclick="filterEvents('TP')">
            <div class="stat-label">✅ 정상예측 TP</div>
            <div class="stat-value green">{stats['TP']}</div>
            <div class="stat-sub">정확한 돌파 예측</div>
        </div>
        <div class="stat-card" onclick="filterEvents('TN')">
            <div class="stat-label">✅ 정상예측 TN</div>
            <div class="stat-value gray">{stats['TN']}</div>
            <div class="stat-sub">정확한 안전 예측</div>
        </div>
        <div class="stat-card" onclick="filterEvents('FN_10min')">
            <div class="stat-label">⚠️ FN 10분전예측</div>
            <div class="stat-value yellow">{stats['FN_10min']}</div>
            <div class="stat-sub">조기 감지됨</div>
        </div>
        <div class="stat-card warning" onclick="filterEvents('FN_miss')">
            <div class="stat-label">❌ FN 완전놓침</div>
            <div class="stat-value red">{stats['FN_miss']}</div>
            <div class="stat-sub">실질 놓침</div>
        </div>
        <div class="stat-card" onclick="filterEvents('FP_10min')">
            <div class="stat-label">⚠️ FP 10분후돌파</div>
            <div class="stat-value yellow">{stats['FP_10min']}</div>
            <div class="stat-sub">유효 조기 경고</div>
        </div>
        <div class="stat-card warning" onclick="filterEvents('FP_false')">
            <div class="stat-label">❌ FP 잘못된경고</div>
            <div class="stat-value red">{stats['FP_false']}</div>
            <div class="stat-sub">실질 오탐</div>
        </div>
    </div>
    
    <!-- 핵심 성능 지표 -->
    <div class="stats-grid" style="grid-template-columns: repeat(4, 1fr); margin-bottom: 30px;">
        <div class="stat-card highlight">
            <div class="stat-label">🎯 Recall (감지율)</div>
            <div class="stat-value green">{stats['recall']:.1f}%</div>
            <div class="stat-sub">돌파 감지 성공률</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">🎯 Precision (정밀도)</div>
            <div class="stat-value" style="color:#00d4ff">{stats['precision']:.1f}%</div>
            <div class="stat-sub">경고 정확도</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">📊 실제 돌파</div>
            <div class="stat-value" style="color:#ff6b35">{stats['actual_breach']}</div>
            <div class="stat-sub">1700+ 케이스</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">📊 총 예측</div>
            <div class="stat-value gray">{stats['total']}</div>
            <div class="stat-sub">전체 데이터</div>
        </div>
    </div>
    
    <!-- 트렌드 차트 -->
    <div class="chart-section">
        <div class="chart-title">📈 TOTALCNT 트렌드 & 1700 알람 라인</div>
        <div class="chart-controls">
            <label>시작:</label>
            <input type="number" id="chartStart" value="0" min="0" style="width:80px">
            <label>개수:</label>
            <input type="number" id="chartCount" value="200" min="50" max="500" style="width:80px">
            <button onclick="updateChart()">차트 갱신</button>
            <button onclick="showFullChart()" style="background:linear-gradient(135deg,#ff6b35,#cc4422)">전체 보기</button>
        </div>
        <div class="chart-container">
            <canvas id="trendChart"></canvas>
        </div>
        <div class="legend">
            <div class="legend-item"><div class="legend-color" style="background:#00d4ff"></div>현재 TOTALCNT</div>
            <div class="legend-item"><div class="legend-color" style="background:#00ff88"></div>실제 최대값</div>
            <div class="legend-item"><div class="legend-color" style="background:#ff6b35"></div>XGB 중요 예측</div>
            <div class="legend-item"><div class="legend-color" style="background:#ff4466;height:2px;width:20px"></div>1700 리미트</div>
        </div>
    </div>
    
    <!-- 이벤트 리스트 -->
    <div class="event-section">
        <div class="event-header">
            <div class="event-title">📋 예측 이벤트 목록</div>
            <div class="event-count" id="eventCount">전체: {stats['total']}개</div>
        </div>
        <div class="filter-tabs">
            <div class="filter-tab active" onclick="filterEvents('all')">전체</div>
            <div class="filter-tab" onclick="filterEvents('TP')">✅ TP ({stats['TP']})</div>
            <div class="filter-tab" onclick="filterEvents('TN')">✅ TN ({stats['TN']})</div>
            <div class="filter-tab" onclick="filterEvents('FN')">❌ FN ({stats['FN_10min'] + stats['FN_miss']})</div>
            <div class="filter-tab" onclick="filterEvents('FP')">⚠️ FP ({stats['FP_10min'] + stats['FP_false']})</div>
            <div class="filter-tab" onclick="filterEvents('breach')">🔥 돌파 ({stats['actual_breach']})</div>
        </div>
        <div class="event-list" id="eventList">
            <div class="event-item header">
                <div>No.</div>
                <div>현재시간</div>
                <div>현재값</div>
                <div>실제최대</div>
                <div>예측값</div>
                <div>투표</div>
                <div>상태</div>
            </div>
        </div>
    </div>
</div>

<!-- 상세 모달 -->
<div id="detailModal" class="modal">
    <div class="modal-content">
        <div class="modal-header">
            <div class="modal-title" id="modalTitle">📈 이벤트 상세</div>
            <button class="modal-close" onclick="closeModal()">&times;</button>
        </div>
        <div id="modalInfo" style="text-align:center;color:#888;margin-bottom:15px"></div>
        <div class="modal-chart">
            <canvas id="modalChart"></canvas>
        </div>
        <div class="modal-info" id="modalStats"></div>
    </div>
</div>

<script>
const allData = {json.dumps(data_json, ensure_ascii=False)};
let trendChart = null;
let modalChart = null;
let currentFilter = 'all';

// 초기화
document.addEventListener('DOMContentLoaded', function() {{
    renderEventList(allData);
    initTrendChart();
}});

function initTrendChart() {{
    const start = parseInt(document.getElementById('chartStart').value);
    const count = parseInt(document.getElementById('chartCount').value);
    const data = allData.slice(start, start + count);
    
    const ctx = document.getElementById('trendChart').getContext('2d');
    if (trendChart) trendChart.destroy();
    
    trendChart = new Chart(ctx, {{
        type: 'line',
        data: {{
            labels: data.map(d => d.time.split(' ')[1] || d.time),
            datasets: [
                {{
                    label: '현재 TOTALCNT',
                    data: data.map(d => d.current),
                    borderColor: '#00d4ff',
                    backgroundColor: 'rgba(0,212,255,0.1)',
                    fill: false,
                    tension: 0.3,
                    pointRadius: 1,
                    borderWidth: 2
                }},
                {{
                    label: '실제 최대값',
                    data: data.map(d => d.actual_max),
                    borderColor: '#00ff88',
                    backgroundColor: 'rgba(0,255,136,0.1)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 1,
                    borderWidth: 2
                }},
                {{
                    label: 'XGB 중요 예측',
                    data: data.map(d => d.xgb_important),
                    borderColor: '#ff6b35',
                    backgroundColor: 'rgba(255,107,53,0.1)',
                    fill: false,
                    tension: 0.3,
                    pointRadius: 1,
                    borderWidth: 1.5,
                    borderDash: [5, 3]
                }},
                {{
                    label: '1700 리미트',
                    data: data.map(() => 1700),
                    borderColor: '#ff4466',
                    borderDash: [10, 5],
                    borderWidth: 2,
                    fill: false,
                    pointRadius: 0
                }}
            ]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            interaction: {{ mode: 'index', intersect: false }},
            plugins: {{
                legend: {{ position: 'top', labels: {{ color: '#fff' }} }},
                tooltip: {{
                    backgroundColor: 'rgba(0,0,0,0.9)',
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    callbacks: {{
                        afterBody: function(context) {{
                            const idx = context[0].dataIndex;
                            const d = data[idx];
                            return ['', '상태: ' + d.status, '투표: ' + d.votes + '/6'];
                        }}
                    }}
                }}
            }},
            scales: {{
                x: {{ ticks: {{ color: '#888', maxRotation: 45 }}, grid: {{ color: '#333' }} }},
                y: {{ ticks: {{ color: '#888' }}, grid: {{ color: '#333' }}, min: 1300 }}
            }},
            onClick: function(e, elements) {{
                if (elements.length > 0) {{
                    const idx = elements[0].index;
                    showDetail(start + idx);
                }}
            }}
        }}
    }});
}}

function updateChart() {{
    initTrendChart();
}}

function showFullChart() {{
    document.getElementById('chartStart').value = 0;
    document.getElementById('chartCount').value = Math.min(allData.length, 500);
    initTrendChart();
}}

function renderEventList(data) {{
    const container = document.getElementById('eventList');
    let html = '<div class="event-item header"><div>No.</div><div>현재시간</div><div>현재값</div><div>실제최대</div><div>예측값</div><div>투표</div><div>상태</div></div>';
    
    data.forEach((d, i) => {{
        let statusClass = 'tn';
        if (d.status.includes('TP')) statusClass = 'tp';
        else if (d.status.includes('FN')) statusClass = 'fn';
        else if (d.status.includes('FP')) statusClass = 'fp';
        
        const timeShort = d.time.split(' ')[1] || d.time;
        html += `<div class="event-item" onclick="showDetail(${{d.idx}})">
            <div class="event-num">${{i + 1}}</div>
            <div class="event-time">${{timeShort}}</div>
            <div class="event-val current">${{d.current.toFixed(0)}}</div>
            <div class="event-val actual">${{d.actual_max.toFixed(0)}}</div>
            <div class="event-val pred">${{d.xgb_important.toFixed(0)}}</div>
            <div style="text-align:center">${{d.votes}}/6</div>
            <div class="event-status ${{statusClass}}">${{d.status.substring(0,12)}}</div>
        </div>`;
    }});
    
    container.innerHTML = html;
    document.getElementById('eventCount').textContent = `필터: ${{data.length}}개`;
}}

function filterEvents(type) {{
    document.querySelectorAll('.filter-tab').forEach(t => t.classList.remove('active'));
    event.target.classList.add('active');
    
    let filtered = allData;
    if (type === 'TP') filtered = allData.filter(d => d.status === '정상예측_TP');
    else if (type === 'TN') filtered = allData.filter(d => d.status === '정상예측_TN');
    else if (type === 'FN') filtered = allData.filter(d => d.status.includes('FN'));
    else if (type === 'FN_10min') filtered = allData.filter(d => d.status === 'FN_10분전예측');
    else if (type === 'FN_miss') filtered = allData.filter(d => d.status === 'FN_완전놓침');
    else if (type === 'FP') filtered = allData.filter(d => d.status.includes('FP'));
    else if (type === 'FP_10min') filtered = allData.filter(d => d.status === 'FP_10분후돌파');
    else if (type === 'FP_false') filtered = allData.filter(d => d.status === 'FP_잘못된경고');
    else if (type === 'breach') filtered = allData.filter(d => d.actual_breach === 1);
    
    renderEventList(filtered);
}}

function showDetail(idx) {{
    const item = allData[idx];
    const start = Math.max(0, idx - 30);
    const end = Math.min(allData.length, idx + 31);
    const rangeData = allData.slice(start, end);
    const centerPos = idx - start;
    
    document.getElementById('modalTitle').textContent = `📈 ${{item.time}} - ${{item.status}}`;
    document.getElementById('modalInfo').textContent = `현재값: ${{item.current.toFixed(0)}} | 실제최대: ${{item.actual_max.toFixed(0)}} | XGB예측: ${{item.xgb_important.toFixed(1)}} | 투표: ${{item.votes}}/6`;
    
    const ctx = document.getElementById('modalChart').getContext('2d');
    if (modalChart) modalChart.destroy();
    
    // 포인트 색상 (상태별)
    const pointColors = rangeData.map((d, i) => {{
        if (i === centerPos) return '#ffcc00';
        if (d.actual_breach === 1) return '#ff4466';
        if (d.pred_breach === 1) return '#ff6b35';
        return '#00d4ff';
    }});
    
    modalChart = new Chart(ctx, {{
        type: 'line',
        data: {{
            labels: rangeData.map(d => (d.time.split(' ')[1] || d.time).substring(0,5)),
            datasets: [
                {{
                    label: '현재 TOTALCNT',
                    data: rangeData.map(d => d.current),
                    borderColor: '#00d4ff',
                    backgroundColor: 'rgba(0,212,255,0.1)',
                    fill: false,
                    tension: 0.3,
                    pointRadius: 4,
                    pointBackgroundColor: pointColors
                }},
                {{
                    label: '실제 최대값',
                    data: rangeData.map(d => d.actual_max),
                    borderColor: '#00ff88',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 3
                }},
                {{
                    label: 'XGB 중요 예측',
                    data: rangeData.map(d => d.xgb_important),
                    borderColor: '#ff6b35',
                    borderDash: [5, 3],
                    fill: false,
                    tension: 0.3,
                    pointRadius: 3
                }},
                {{
                    label: '1700 리미트',
                    data: rangeData.map(() => 1700),
                    borderColor: '#ff4466',
                    borderDash: [10, 5],
                    borderWidth: 2,
                    fill: false,
                    pointRadius: 0
                }}
            ]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {{
                legend: {{ position: 'top', labels: {{ color: '#fff' }} }},
                tooltip: {{
                    backgroundColor: 'rgba(0,0,0,0.9)',
                    callbacks: {{
                        afterBody: function(context) {{
                            const i = context[0].dataIndex;
                            const d = rangeData[i];
                            return ['', '상태: ' + d.status, 'LGBM확률: ' + (d.lgbm_important_prob * 100).toFixed(1) + '%'];
                        }}
                    }}
                }}
            }},
            scales: {{
                x: {{ ticks: {{ color: '#888' }}, grid: {{ color: '#333' }} }},
                y: {{ ticks: {{ color: '#888' }}, grid: {{ color: '#333' }} }}
            }}
        }}
    }});
    
    // 통계
    const actuals = rangeData.map(d => d.actual_max);
    const preds = rangeData.map(d => d.xgb_important);
    document.getElementById('modalStats').innerHTML = `
        <div class="modal-stat"><div class="modal-stat-label">범위</div><div class="modal-stat-value" style="color:#00d4ff">${{rangeData.length}}분</div></div>
        <div class="modal-stat"><div class="modal-stat-label">현재값</div><div class="modal-stat-value" style="color:#00ff88">${{item.current.toFixed(0)}}</div></div>
        <div class="modal-stat"><div class="modal-stat-label">실제 최대</div><div class="modal-stat-value" style="color:#00d4ff">${{Math.max(...actuals).toFixed(0)}}</div></div>
        <div class="modal-stat"><div class="modal-stat-label">예측 평균</div><div class="modal-stat-value" style="color:#ff6b35">${{(preds.reduce((a,b)=>a+b,0)/preds.length).toFixed(1)}}</div></div>
        <div class="modal-stat"><div class="modal-stat-label">LGBM 확률</div><div class="modal-stat-value" style="color:#ffcc00">${{(item.lgbm_important_prob * 100).toFixed(1)}}%</div></div>
    `;
    
    document.getElementById('detailModal').classList.add('show');
}}

function closeModal() {{
    document.getElementById('detailModal').classList.remove('show');
    if (modalChart) {{ modalChart.destroy(); modalChart = null; }}
}}

document.getElementById('detailModal').addEventListener('click', function(e) {{
    if (e.target === this) closeModal();
}});
</script>
</body>
</html>'''
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    return output_path

def main():
    """메인 실행"""
    print("="*60)
    print("📊 V12 평가 결과 대시보드 생성기")
    print("="*60)
    
    # CSV 파일 경로 설정 (수정 가능)
    csv_path = input("CSV 파일 경로 입력 (Enter=YUYS): ").strip()
    if not csv_path:
        csv_path = "YUYS"
    
    if not os.path.exists(csv_path):
        print(f"❌ 파일을 찾을 수 없습니다: {csv_path}")
        return
    
    # CSV 로드
    print(f"\n📂 CSV 로드 중: {csv_path}")
    df = load_csv(csv_path)
    print(f"✅ 로드 완료: {len(df):,}개 레코드")
    
    # 통계 분석
    print("\n📈 통계 분석 중...")
    stats = analyze_results(df)
    print(f"  - 총 예측: {stats['total']:,}개")
    print(f"  - 실제 돌파: {stats['actual_breach']:,}개")
    print(f"  - Recall: {stats['recall']:.1f}%")
    print(f"  - 실질 FN: {stats['FN_miss']}개")
    
    # HTML 생성
    output_path = f"v12_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    print(f"\n🔧 HTML 대시보드 생성 중...")
    generate_dashboard_html(df, stats, output_path)
    print(f"✅ 저장 완료: {output_path}")
    
    # 브라우저 열기
    try:
        import webbrowser
        webbrowser.open(output_path)
        print("🌐 브라우저에서 열기 완료!")
    except:
        print("💡 브라우저에서 직접 열어주세요.")
    
    print("\n" + "="*60)
    print("완료!")
    print("="*60)

if __name__ == "__main__":
    main()