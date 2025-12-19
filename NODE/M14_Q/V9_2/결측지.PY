#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 AMHS 결측치 분석 대시보드
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
실행방법:
    python missing_dashboard.py

출력파일:
    - missing_dashboard.html  (인터랙티브 대시보드)
    - missing_by_column.csv   (컬럼별 결측치)
    - missing_by_date.csv     (날짜별 결측치)
    - missing_by_hour.csv     (시간대별 결측치)
    - key_columns_status.csv  (핵심 8개 컬럼 상태)
    - collection_gaps.csv     (미수집 구간)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# ⚙️ 설정 - 여기만 수정하면 됨!
# ============================================================
CONFIG = {
    'input_file': 'data/M14_Q_20250802_TO_20251015.csv',  # 📌 분석할 CSV 파일 경로
    'output_dir': 'missing_results',                       # 📌 결과 저장 폴더
    'time_column': 'CURRTIME',                             # 시간 컬럼명
    'time_format': '%Y%m%d%H%M',                           # 시간 포맷 (YYYYMMDDHHMM)
}

# V8.3 핵심 컬럼 (수정 가능)
KEY_COLUMNS = [
    ('M14AM14B', 'M14AM14B'),
    ('M14AM14BSUM', 'M14AM14BSUM'),
    ('M10AM14A', 'M10AM14A'),
    ('TOTALCNT', 'TOTALCNT'),
    ('M14.QUE.ALL.TRANSPORT4MINOVERCNT', 'TRANSPORT'),
    ('M14.QUE.OHT.OHTUTIL', 'OHT_UTIL'),
    ('M14.QUE.ALL.CURRENTQCREATED', 'Q_CREATED'),
    ('M14.QUE.ALL.CURRENTQCOMPLETED', 'Q_COMPLETED'),
]
# ============================================================


class MissingAnalyzer:
    def __init__(self, config):
        self.config = config
        self.df = None
        self.analysis = {}
        
    def load_data(self):
        """데이터 로드"""
        file_path = self.config['input_file']
        print(f"\n📂 데이터 로딩: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
            print("💡 CONFIG의 'input_file' 경로를 확인하세요!")
            return False
        
        self.df = pd.read_csv(file_path, on_bad_lines='skip')
        print(f"✅ {len(self.df):,}행 × {self.df.shape[1]}열")
        
        # 시간 컬럼 파싱
        time_col = self.config['time_column']
        if time_col in self.df.columns:
            self.df[time_col] = self.df[time_col].astype(str).str.strip()
            self.df[time_col] = pd.to_datetime(
                self.df[time_col], 
                format=self.config['time_format'], 
                errors='coerce'
            )
            self.df['DATE'] = self.df[time_col].dt.date
            self.df['HOUR'] = self.df[time_col].dt.hour
            
            valid_times = self.df[time_col].dropna()
            if len(valid_times) > 0:
                print(f"📅 기간: {valid_times.min()} ~ {valid_times.max()}")
        
        return True
    
    def analyze(self):
        """전체 분석 실행"""
        print("\n🔍 분석 중...")
        df = self.df
        total = len(df)
        time_col = self.config['time_column']
        
        # 1. 기본 정보
        date_start = '-'
        date_end = '-'
        if time_col in df.columns:
            valid_times = df[time_col].dropna()
            if len(valid_times) > 0:
                date_start = valid_times.min().strftime('%Y-%m-%d %H:%M')
                date_end = valid_times.max().strftime('%Y-%m-%d %H:%M')
        
        self.analysis['info'] = {
            'file': os.path.basename(self.config['input_file']),
            'total_rows': total,
            'total_cols': len(df.columns),
            'date_start': date_start,
            'date_end': date_end,
            'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 2. 컬럼별 결측치
        exclude_cols = ['DATE', 'HOUR']
        analysis_cols = [c for c in df.columns if c not in exclude_cols]
        
        missing_counts = df[analysis_cols].isnull().sum()
        missing_pct = (missing_counts / total * 100).round(2)
        
        col_summary = []
        for col in analysis_cols:
            col_summary.append({
                'column': col,
                'missing': int(missing_counts[col]),
                'pct': float(missing_pct[col]),
                'valid': int(total - missing_counts[col]),
                'dtype': str(df[col].dtype)
            })
        
        col_summary = sorted(col_summary, key=lambda x: x['pct'], reverse=True)
        self.analysis['columns'] = col_summary
        self.analysis['missing_columns'] = [c for c in col_summary if c['missing'] > 0]
        
        # 3. 날짜별 결측치
        daily_missing = []
        if 'DATE' in df.columns:
            dates = sorted(df['DATE'].dropna().unique())
            num_cols = len(analysis_cols)
            for date in dates:
                day_df = df[df['DATE'] == date]
                day_total = len(day_df)
                day_missing = int(day_df[analysis_cols].isnull().sum().sum())
                day_pct = round((day_missing / (day_total * num_cols)) * 100, 2) if day_total > 0 else 0
                daily_missing.append({
                    'date': date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date),
                    'rows': day_total,
                    'missing_cells': day_missing,
                    'pct': day_pct
                })
        self.analysis['daily'] = daily_missing
        
        # 4. 시간대별 결측치
        hourly_missing = []
        if 'HOUR' in df.columns:
            num_cols = len(analysis_cols)
            for hour in range(24):
                hour_df = df[df['HOUR'] == hour]
                if len(hour_df) > 0:
                    hour_missing = int(hour_df[analysis_cols].isnull().sum().sum())
                    hour_pct = round((hour_missing / (len(hour_df) * num_cols)) * 100, 2)
                    hourly_missing.append({
                        'hour': f'{hour:02d}:00',
                        'rows': len(hour_df),
                        'missing_cells': hour_missing,
                        'pct': hour_pct
                    })
        self.analysis['hourly'] = hourly_missing
        
        # 5. 핵심 컬럼 상태
        key_status = []
        for col, short in KEY_COLUMNS:
            if col in df.columns:
                m = int(df[col].isnull().sum())
                p = round(m / total * 100, 2)
                status = '✅' if p == 0 else '⚠️' if p < 1 else '❌'
                key_status.append({
                    'column': col, 'short': short, 'missing': m, 
                    'pct': p, 'status': status, 'exists': True
                })
            else:
                key_status.append({
                    'column': col, 'short': short, 'missing': total, 
                    'pct': 100, 'status': '❌', 'exists': False
                })
        self.analysis['key_columns'] = key_status
        
        # 6. 수집 간격 분석
        gaps = []
        gap_stats = {'total_gaps': 0, 'max_gap': 0, 'total_missing_min': 0}
        
        if time_col in df.columns:
            df_sorted = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
            if len(df_sorted) > 1:
                df_sorted['time_diff'] = df_sorted[time_col].diff().dt.total_seconds() / 60
                gap_df = df_sorted[df_sorted['time_diff'] > 1.5].copy()
                
                for idx, row in gap_df.head(50).iterrows():
                    prev_idx = idx - 1
                    if prev_idx >= 0 and prev_idx in df_sorted.index:
                        start_time = df_sorted.loc[prev_idx, time_col]
                        gaps.append({
                            'start': start_time.strftime('%Y-%m-%d %H:%M') if pd.notna(start_time) else '-',
                            'end': row[time_col].strftime('%Y-%m-%d %H:%M') if pd.notna(row[time_col]) else '-',
                            'gap_min': round(row['time_diff'], 1)
                        })
                
                if len(gap_df) > 0:
                    gap_stats = {
                        'total_gaps': len(gap_df),
                        'max_gap': round(gap_df['time_diff'].max(), 1),
                        'total_missing_min': round(gap_df['time_diff'].sum(), 0)
                    }
        
        self.analysis['gaps'] = gaps
        self.analysis['gap_stats'] = gap_stats
        
        # 7. 중복 체크
        self.analysis['duplicates'] = {
            'full': int(df.duplicated().sum()),
            'time': int(df[time_col].duplicated().sum()) if time_col in df.columns else 0
        }
        
        # 8. 요약
        num_cols = len(analysis_cols)
        total_cells = total * num_cols
        total_missing = df[analysis_cols].isnull().sum().sum()
        
        self.analysis['summary'] = {
            'total_cells': int(total_cells),
            'missing_cells': int(total_missing),
            'missing_pct': round(total_missing / total_cells * 100, 2) if total_cells > 0 else 0,
            'columns_with_missing': len(self.analysis['missing_columns']),
            'columns_complete': len([c for c in col_summary if c['missing'] == 0])
        }
        
        print("✅ 분석 완료!")
        return self
    
    def save_csv(self):
        """CSV 결과 저장"""
        output_dir = self.config['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        # 컬럼별
        pd.DataFrame(self.analysis['columns']).to_csv(
            f'{output_dir}/missing_by_column.csv', index=False, encoding='utf-8-sig')
        
        # 날짜별
        if self.analysis['daily']:
            pd.DataFrame(self.analysis['daily']).to_csv(
                f'{output_dir}/missing_by_date.csv', index=False, encoding='utf-8-sig')
        
        # 시간대별
        if self.analysis['hourly']:
            pd.DataFrame(self.analysis['hourly']).to_csv(
                f'{output_dir}/missing_by_hour.csv', index=False, encoding='utf-8-sig')
        
        # 핵심 컬럼
        pd.DataFrame(self.analysis['key_columns']).to_csv(
            f'{output_dir}/key_columns_status.csv', index=False, encoding='utf-8-sig')
        
        # 미수집 구간
        if self.analysis['gaps']:
            pd.DataFrame(self.analysis['gaps']).to_csv(
                f'{output_dir}/collection_gaps.csv', index=False, encoding='utf-8-sig')
        
        print(f"\n💾 CSV 저장 완료: {output_dir}/")
        return self
    
    def generate_html(self):
        """HTML 대시보드 생성"""
        output_dir = self.config['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        a = self.analysis
        
        # 차트 데이터
        daily_labels = json.dumps([d['date'] for d in a['daily']])
        daily_data = json.dumps([d['pct'] for d in a['daily']])
        hourly_labels = json.dumps([h['hour'] for h in a['hourly']])
        hourly_data = json.dumps([h['pct'] for h in a['hourly']])
        
        top_missing = a['missing_columns'][:20]
        top_labels = json.dumps([c['column'][:35] for c in top_missing])
        top_data = json.dumps([c['pct'] for c in top_missing])
        
        key_labels = json.dumps([k['short'] for k in a['key_columns']])
        key_data = json.dumps([k['pct'] for k in a['key_columns']])
        key_colors = json.dumps(['#22c55e' if k['pct']==0 else '#f59e0b' if k['pct']<1 else '#ef4444' for k in a['key_columns']])
        
        html = f'''<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🔍 AMHS 결측치 분석</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', sans-serif; background: linear-gradient(135deg, #1a1a2e, #16213e); color: #e4e4e7; min-height: 100vh; padding: 20px; }}
        .container {{ max-width: 1600px; margin: 0 auto; }}
        header {{ text-align: center; padding: 30px; margin-bottom: 30px; background: rgba(255,255,255,0.05); border-radius: 16px; border: 1px solid rgba(255,255,255,0.1); }}
        header h1 {{ font-size: 2.2rem; margin-bottom: 10px; }}
        header p {{ color: #a1a1aa; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 24px; }}
        .stat-card {{ background: rgba(255,255,255,0.05); border-radius: 12px; padding: 20px; text-align: center; border: 1px solid rgba(255,255,255,0.1); }}
        .stat-card:hover {{ transform: translateY(-3px); transition: 0.2s; }}
        .stat-card .icon {{ font-size: 1.8rem; margin-bottom: 8px; }}
        .stat-card .value {{ font-size: 1.8rem; font-weight: bold; color: #60a5fa; }}
        .stat-card .label {{ color: #a1a1aa; margin-top: 4px; font-size: 0.9rem; }}
        .stat-card.warning .value {{ color: #fbbf24; }}
        .stat-card.danger .value {{ color: #f87171; }}
        .stat-card.success .value {{ color: #4ade80; }}
        .section {{ background: rgba(255,255,255,0.05); border-radius: 16px; padding: 24px; margin-bottom: 24px; border: 1px solid rgba(255,255,255,0.1); }}
        .section h2 {{ font-size: 1.3rem; margin-bottom: 16px; padding-bottom: 10px; border-bottom: 1px solid rgba(255,255,255,0.1); }}
        .chart-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 20px; }}
        .chart-box {{ background: rgba(0,0,0,0.2); border-radius: 12px; padding: 16px; min-height: 320px; }}
        .chart-box h3 {{ margin-bottom: 12px; color: #d4d4d8; font-size: 1rem; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
        th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.1); }}
        th {{ background: rgba(0,0,0,0.3); font-weight: 600; color: #a1a1aa; text-transform: uppercase; font-size: 0.75rem; }}
        tr:hover {{ background: rgba(255,255,255,0.05); }}
        .badge {{ padding: 3px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }}
        .badge-ok {{ background: #166534; color: #4ade80; }}
        .badge-warn {{ background: #854d0e; color: #fbbf24; }}
        .badge-error {{ background: #991b1b; color: #fca5a5; }}
        .progress {{ width: 80px; height: 6px; background: rgba(255,255,255,0.1); border-radius: 3px; overflow: hidden; display: inline-block; vertical-align: middle; margin-left: 8px; }}
        .progress-fill {{ height: 100%; border-radius: 3px; }}
        .progress-ok {{ background: #22c55e; }}
        .progress-warn {{ background: #f59e0b; }}
        .progress-error {{ background: #ef4444; }}
        .key-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }}
        .key-card {{ background: rgba(0,0,0,0.2); border-radius: 10px; padding: 14px; display: flex; align-items: center; gap: 12px; }}
        .key-card .icon {{ font-size: 1.6rem; width: 44px; height: 44px; display: flex; align-items: center; justify-content: center; border-radius: 8px; }}
        .key-card .icon.ok {{ background: rgba(34,197,94,0.2); }}
        .key-card .icon.warn {{ background: rgba(245,158,11,0.2); }}
        .key-card .icon.error {{ background: rgba(239,68,68,0.2); }}
        .key-card .info {{ flex: 1; }}
        .key-card .name {{ font-weight: 600; font-size: 0.85rem; }}
        .key-card .stat {{ color: #a1a1aa; font-size: 0.8rem; margin-top: 2px; }}
        .table-wrap {{ max-height: 400px; overflow-y: auto; }}
        .table-wrap::-webkit-scrollbar {{ width: 6px; }}
        .table-wrap::-webkit-scrollbar-thumb {{ background: rgba(255,255,255,0.2); border-radius: 3px; }}
        .info-box {{ background: rgba(59,130,246,0.1); border: 1px solid rgba(59,130,246,0.3); border-radius: 10px; padding: 14px; margin-bottom: 16px; }}
        .info-box strong {{ color: #60a5fa; }}
        footer {{ text-align: center; padding: 24px; color: #71717a; font-size: 0.85rem; }}
        @media (max-width: 768px) {{ .chart-grid {{ grid-template-columns: 1fr; }} .stats-grid {{ grid-template-columns: repeat(2, 1fr); }} }}
    </style>
</head>
<body>
<div class="container">
    <header>
        <h1>🔍 AMHS 결측치 분석 대시보드</h1>
        <p>📁 {a['info']['file']} | ⏰ {a['info']['analysis_time']}</p>
    </header>
    
    <div class="info-box">
        <p>📅 <strong>분석 기간:</strong> {a['info']['date_start']} ~ {a['info']['date_end']}</p>
        <p>📊 <strong>데이터:</strong> {a['info']['total_rows']:,}행 × {a['info']['total_cols']}열 = {a['summary']['total_cells']:,}셀</p>
    </div>
    
    <div class="stats-grid">
        <div class="stat-card"><div class="icon">📊</div><div class="value">{a['info']['total_rows']:,}</div><div class="label">전체 행</div></div>
        <div class="stat-card"><div class="icon">📋</div><div class="value">{a['info']['total_cols']}</div><div class="label">전체 컬럼</div></div>
        <div class="stat-card {'warning' if a['summary']['missing_pct']>1 else 'success'}"><div class="icon">⚠️</div><div class="value">{a['summary']['missing_pct']}%</div><div class="label">전체 결측률</div></div>
        <div class="stat-card {'danger' if a['summary']['columns_with_missing']>10 else 'warning' if a['summary']['columns_with_missing']>0 else 'success'}"><div class="icon">🔴</div><div class="value">{a['summary']['columns_with_missing']}</div><div class="label">결측 컬럼</div></div>
        <div class="stat-card success"><div class="icon">✅</div><div class="value">{a['summary']['columns_complete']}</div><div class="label">완전 컬럼</div></div>
        <div class="stat-card {'danger' if a['gap_stats']['total_gaps']>10 else 'warning' if a['gap_stats']['total_gaps']>0 else 'success'}"><div class="icon">⏱️</div><div class="value">{a['gap_stats']['total_gaps']}</div><div class="label">미수집 구간</div></div>
    </div>
    
    <div class="section">
        <h2>🔑 핵심 컬럼 상태 (V8.3)</h2>
        <div class="key-grid">
            {''.join([f'<div class="key-card"><div class="icon {"ok" if k["pct"]==0 else "warn" if k["pct"]<1 else "error"}">{k["status"]}</div><div class="info"><div class="name">{k["short"]}</div><div class="stat">{"컬럼 없음!" if not k["exists"] else f"결측: {k["missing"]:,}개 ({k["pct"]}%)"}</div></div></div>' for k in a['key_columns']])}
        </div>
    </div>
    
    <div class="chart-grid">
        <div class="chart-box"><h3>📅 날짜별 결측률</h3><canvas id="dailyChart"></canvas></div>
        <div class="chart-box"><h3>⏰ 시간대별 결측률</h3><canvas id="hourlyChart"></canvas></div>
        <div class="chart-box"><h3>📊 결측 컬럼 TOP 20</h3><canvas id="topChart"></canvas></div>
        <div class="chart-box"><h3>🔑 핵심 컬럼 결측률</h3><canvas id="keyChart"></canvas></div>
    </div>
    
    <div class="section">
        <h2>📋 전체 컬럼 현황 ({len(a['columns'])}개) - 결측 {len(a['missing_columns'])}개 / 완전 {a['summary']['columns_complete']}개</h2>
        <div class="table-wrap" style="max-height:600px;">
            <table>
                <thead><tr><th>#</th><th>컬럼명</th><th>데이터타입</th><th>결측</th><th>결측률</th><th>유효</th><th>상태</th></tr></thead>
                <tbody>
                    {''.join([f'<tr><td>{i+1}</td><td style="max-width:280px;word-break:break-all;">{c["column"]}</td><td>{c["dtype"]}</td><td>{c["missing"]:,}</td><td>{c["pct"]}%<div class="progress"><div class="progress-fill {"progress-error" if c["pct"]>10 else "progress-warn" if c["pct"]>1 else "progress-ok"}" style="width:{min(c["pct"],100)}%"></div></div></td><td>{c["valid"]:,}</td><td><span class="badge {"badge-error" if c["pct"]>10 else "badge-warn" if c["pct"]>1 else "badge-ok"}">{"심각" if c["pct"]>10 else "주의" if c["pct"]>1 else "양호"}</span></td></tr>' for i, c in enumerate(a['columns'])])}
                </tbody>
            </table>
        </div>
    </div>
    
    <div class="section">
        <h2>📅 날짜별 현황 ({len(a['daily'])}일)</h2>
        <div class="table-wrap">
            <table>
                <thead><tr><th>날짜</th><th>데이터</th><th>결측 셀</th><th>결측률</th><th>상태</th></tr></thead>
                <tbody>
                    {''.join([f'<tr><td>{d["date"]}</td><td>{d["rows"]:,}</td><td>{d["missing_cells"]:,}</td><td>{d["pct"]}%<div class="progress"><div class="progress-fill {"progress-error" if d["pct"]>5 else "progress-warn" if d["pct"]>1 else "progress-ok"}" style="width:{min(d["pct"]*10,100)}%"></div></div></td><td><span class="badge {"badge-error" if d["pct"]>5 else "badge-warn" if d["pct"]>1 else "badge-ok"}">{"심각" if d["pct"]>5 else "주의" if d["pct"]>1 else "양호"}</span></td></tr>' for d in a['daily']])}
                </tbody>
            </table>
        </div>
    </div>
    
    {f'''<div class="section">
        <h2>⏱️ 미수집 구간 ({a['gap_stats']['total_gaps']}건, 총 {a['gap_stats']['total_missing_min']:.0f}분)</h2>
        <div class="info-box"><p>📊 <strong>최대 간격:</strong> {a['gap_stats']['max_gap']}분 | 1.5분 초과 구간만 표시</p></div>
        <div class="table-wrap">
            <table>
                <thead><tr><th>#</th><th>시작</th><th>종료</th><th>간격(분)</th><th>상태</th></tr></thead>
                <tbody>{''.join([f"<tr><td>{i+1}</td><td>{g['start']}</td><td>{g['end']}</td><td>{g['gap_min']}</td><td><span class='badge {'badge-error' if g['gap_min']>10 else 'badge-warn'}'>{'심각' if g['gap_min']>10 else '주의'}</span></td></tr>" for i, g in enumerate(a['gaps'])])}</tbody>
            </table>
        </div>
    </div>''' if a['gaps'] else ''}
    
    <div class="section">
        <h2>🔄 중복 데이터</h2>
        <div class="stats-grid" style="max-width:400px;">
            <div class="stat-card {'danger' if a['duplicates']['full']>0 else 'success'}"><div class="value">{a['duplicates']['full']:,}</div><div class="label">완전 중복</div></div>
            <div class="stat-card {'warning' if a['duplicates']['time']>0 else 'success'}"><div class="value">{a['duplicates']['time']:,}</div><div class="label">시간 중복</div></div>
        </div>
    </div>
    
    <footer>🔍 AMHS 결측치 분석 대시보드 | V8.3</footer>
</div>

<script>
const colors = {{ primary: 'rgba(96,165,250,1)', primaryBg: 'rgba(96,165,250,0.3)', grid: 'rgba(255,255,255,0.1)', text: '#a1a1aa' }};
Chart.defaults.color = colors.text;
Chart.defaults.borderColor = colors.grid;

new Chart(document.getElementById('dailyChart'), {{
    type: 'line',
    data: {{ labels: {daily_labels}, datasets: [{{ label: '결측률(%)', data: {daily_data}, borderColor: colors.primary, backgroundColor: colors.primaryBg, fill: true, tension: 0.3 }}] }},
    options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ display: false }} }}, scales: {{ y: {{ beginAtZero: true }}, x: {{ display: {str(len(a['daily']) <= 31).lower()} }} }} }}
}});

new Chart(document.getElementById('hourlyChart'), {{
    type: 'bar',
    data: {{ labels: {hourly_labels}, datasets: [{{ label: '결측률(%)', data: {hourly_data}, backgroundColor: colors.primaryBg, borderColor: colors.primary, borderWidth: 1 }}] }},
    options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ display: false }} }}, scales: {{ y: {{ beginAtZero: true }} }} }}
}});

new Chart(document.getElementById('topChart'), {{
    type: 'bar',
    data: {{ labels: {top_labels}, datasets: [{{ data: {top_data}, backgroundColor: {top_data}.map(v => v > 10 ? '#ef4444' : v > 1 ? '#f59e0b' : '#22c55e') }}] }},
    options: {{ responsive: true, maintainAspectRatio: false, indexAxis: 'y', plugins: {{ legend: {{ display: false }} }}, scales: {{ x: {{ beginAtZero: true }} }} }}
}});

new Chart(document.getElementById('keyChart'), {{
    type: 'bar',
    data: {{ labels: {key_labels}, datasets: [{{ data: {key_data}, backgroundColor: {key_colors} }}] }},
    options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ display: false }} }}, scales: {{ y: {{ beginAtZero: true }} }} }}
}});
</script>
</body>
</html>'''
        
        output_path = f'{output_dir}/missing_dashboard.html'
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"🎨 HTML 대시보드 저장: {output_path}")
        return self


def main():
    print("=" * 60)
    print("🔍 AMHS 결측치 분석 대시보드")
    print("=" * 60)
    
    analyzer = MissingAnalyzer(CONFIG)
    
    if not analyzer.load_data():
        return
    
    analyzer.analyze()
    analyzer.save_csv()
    analyzer.generate_html()
    
    output_dir = CONFIG['output_dir']
    print(f"\n{'=' * 60}")
    print("✅ 분석 완료!")
    print(f"📁 결과 폴더: {output_dir}/")
    print("   📊 missing_dashboard.html  ← 브라우저로 열기!")
    print("   📋 missing_by_column.csv")
    print("   📅 missing_by_date.csv")
    print("   ⏰ missing_by_hour.csv")
    print("   🔑 key_columns_status.csv")
    print("   ⏱️ collection_gaps.csv")
    print("=" * 60)


if __name__ == '__main__':
    main()