#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AMHS 결측치 분석 UI (한글판)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import webbrowser
import threading
import warnings
warnings.filterwarnings('ignore')


class MissingAnalyzerUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AMHS 결측치 분석 도구")
        self.root.geometry("1000x700")
        self.root.configure(bg='#1a1a2e')
        
        self.df = None
        self.file_path = None
        self.output_dir = "missing_results"
        
        self.setup_styles()
        self.create_widgets()
    
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('TFrame', background='#1a1a2e')
        style.configure('TLabel', background='#1a1a2e', foreground='#e4e4e7', font=('Segoe UI', 10))
        style.configure('Header.TLabel', font=('Segoe UI', 24, 'bold'), foreground='#60a5fa')
        style.configure('Sub.TLabel', font=('Segoe UI', 11), foreground='#a1a1aa')
        style.configure('Status.TLabel', font=('Segoe UI', 10), foreground='#4ade80')
        style.configure('TButton', font=('Segoe UI', 11, 'bold'), padding=10)
        
        style.configure('Treeview', 
                       background='#16213e', 
                       foreground='#e4e4e7', 
                       fieldbackground='#16213e',
                       font=('Consolas', 9),
                       rowheight=25)
        style.configure('Treeview.Heading', 
                       font=('Segoe UI', 10, 'bold'),
                       background='#0f172a',
                       foreground='#60a5fa')
        style.map('Treeview', background=[('selected', '#3b82f6')])
    
    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        ttk.Label(header_frame, text="[AMHS] 결측치 분석 도구", style='Header.TLabel').pack(side=tk.LEFT)
        
        file_frame = ttk.Frame(main_frame)
        file_frame.pack(fill=tk.X, pady=(0, 15))
        ttk.Label(file_frame, text="CSV 파일:", style='Sub.TLabel').pack(side=tk.LEFT)
        self.file_label = ttk.Label(file_frame, text="파일을 선택하세요", style='Sub.TLabel')
        self.file_label.pack(side=tk.LEFT, padx=10)
        ttk.Button(file_frame, text="파일 선택", command=self.select_file).pack(side=tk.RIGHT)
        
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=(0, 15))
        self.info_label = ttk.Label(info_frame, text="", style='Status.TLabel')
        self.info_label.pack(side=tk.LEFT)
        
        columns_frame = ttk.Frame(main_frame)
        columns_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        ttk.Label(columns_frame, text="컬럼 목록", style='Sub.TLabel').pack(anchor=tk.W)
        
        tree_container = ttk.Frame(columns_frame)
        tree_container.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.tree = ttk.Treeview(tree_container, columns=('idx', 'column', 'dtype', 'non_null', 'null', 'null_pct'), show='headings', height=15)
        self.tree.heading('idx', text='#')
        self.tree.heading('column', text='컬럼명')
        self.tree.heading('dtype', text='타입')
        self.tree.heading('non_null', text='유효')
        self.tree.heading('null', text='결측')
        self.tree.heading('null_pct', text='결측률')
        
        self.tree.column('idx', width=50, anchor='center')
        self.tree.column('column', width=350, anchor='w')
        self.tree.column('dtype', width=100, anchor='center')
        self.tree.column('non_null', width=100, anchor='center')
        self.tree.column('null', width=100, anchor='center')
        self.tree.column('null_pct', width=100, anchor='center')
        
        scrollbar_y = ttk.Scrollbar(tree_container, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar_x = ttk.Scrollbar(tree_container, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.analyze_btn = ttk.Button(button_frame, text="분석 실행", command=self.run_analysis, state=tk.DISABLED)
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        self.open_html_btn = ttk.Button(button_frame, text="HTML 열기", command=self.open_html, state=tk.DISABLED)
        self.open_html_btn.pack(side=tk.LEFT, padx=5)
        self.open_folder_btn = ttk.Button(button_frame, text="결과 폴더", command=self.open_folder, state=tk.DISABLED)
        self.open_folder_btn.pack(side=tk.LEFT, padx=5)
        
        self.progress_var = tk.StringVar(value="")
        self.progress_label = ttk.Label(button_frame, textvariable=self.progress_var, style='Status.TLabel')
        self.progress_label.pack(side=tk.RIGHT, padx=10)
        
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=(10, 0))
    
    def select_file(self):
        file_path = filedialog.askopenfilename(
            title="CSV 파일 선택",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            self.file_path = file_path
            self.file_label.config(text=os.path.basename(file_path))
            self.load_csv()
    
    def load_csv(self):
        try:
            self.progress_var.set("파일 로딩 중...")
            self.progress.start()
            self.root.update()
            
            self.df = pd.read_csv(self.file_path, on_bad_lines='skip')
            
            if 'CURRTIME' in self.df.columns:
                self.df['CURRTIME'] = pd.to_datetime(
                    self.df['CURRTIME'].astype(str).str.strip(),
                    format='%Y%m%d%H%M',
                    errors='coerce'
                )
            
            rows, cols = self.df.shape
            self.info_label.config(text=f"{rows:,}행 x {cols}열 로드 완료")
            self.update_column_list()
            self.analyze_btn.config(state=tk.NORMAL)
            self.progress.stop()
            self.progress_var.set("로드 완료! [분석 실행] 클릭")
            
        except Exception as e:
            self.progress.stop()
            self.progress_var.set("")
            messagebox.showerror("오류", f"파일 로드 실패:\n{str(e)}")
    
    def update_column_list(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        total = len(self.df)
        for i, col in enumerate(self.df.columns):
            non_null = self.df[col].notna().sum()
            null = self.df[col].isna().sum()
            null_pct = f"{null / total * 100:.2f}%"
            dtype = str(self.df[col].dtype)
            tag = 'error' if null / total > 0.1 else 'warn' if null > 0 else 'ok'
            self.tree.insert('', 'end', values=(i+1, col, dtype, f"{non_null:,}", f"{null:,}", null_pct), tags=(tag,))
        
        self.tree.tag_configure('error', foreground='#f87171')
        self.tree.tag_configure('warn', foreground='#fbbf24')
        self.tree.tag_configure('ok', foreground='#4ade80')
    
    def run_analysis(self):
        self.analyze_btn.config(state=tk.DISABLED)
        self.progress.start()
        self.progress_var.set("분석 중...")
        thread = threading.Thread(target=self._analyze_thread)
        thread.start()
    
    def _analyze_thread(self):
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            analysis = self.analyze_data()
            self.root.after(0, lambda: self.progress_var.set("CSV 저장 중..."))
            self.save_csv(analysis)
            self.root.after(0, lambda: self.progress_var.set("HTML 생성 중..."))
            self.generate_html(analysis)
            self.root.after(0, self._analysis_complete)
        except Exception as e:
            self.root.after(0, lambda: self._analysis_error(str(e)))
    
    def _analysis_complete(self):
        self.progress.stop()
        self.progress_var.set(f"분석 완료! 결과: {self.output_dir}/")
        self.analyze_btn.config(state=tk.NORMAL)
        self.open_html_btn.config(state=tk.NORMAL)
        self.open_folder_btn.config(state=tk.NORMAL)
        if messagebox.askyesno("완료", "분석 완료!\n\nHTML 대시보드를 열까요?"):
            self.open_html()
    
    def _analysis_error(self, error):
        self.progress.stop()
        self.progress_var.set("")
        self.analyze_btn.config(state=tk.NORMAL)
        messagebox.showerror("오류", f"분석 실패:\n{error}")
    
    def analyze_data(self):
        df = self.df.copy()
        total = len(df)
        
        exclude_cols = ['DATE', 'HOUR']
        if 'CURRTIME' in df.columns:
            df['DATE'] = df['CURRTIME'].dt.date
            df['HOUR'] = df['CURRTIME'].dt.hour
        
        analysis_cols = [c for c in df.columns if c not in exclude_cols]
        
        date_start = '-'
        date_end = '-'
        if 'CURRTIME' in df.columns:
            valid_times = df['CURRTIME'].dropna()
            if len(valid_times) > 0:
                date_start = valid_times.min().strftime('%Y-%m-%d %H:%M')
                date_end = valid_times.max().strftime('%Y-%m-%d %H:%M')
        
        analysis = {
            'info': {
                'file': os.path.basename(self.file_path),
                'total_rows': total,
                'total_cols': len(df.columns),
                'date_start': date_start,
                'date_end': date_end,
                'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        # 컬럼별 결측치
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
        analysis['columns'] = col_summary
        analysis['missing_columns'] = [c for c in col_summary if c['missing'] > 0]
        
        # 날짜별 결측치 (빈 날짜도 포함!)
        daily_missing = []
        if 'CURRTIME' in df.columns:
            valid_times = df['CURRTIME'].dropna()
            if len(valid_times) > 0:
                min_date = valid_times.min().date()
                max_date = valid_times.max().date()
                
                # 모든 날짜 생성
                all_dates = []
                current = min_date
                while current <= max_date:
                    all_dates.append(current)
                    current += timedelta(days=1)
                
                num_cols = len(analysis_cols)
                for date in all_dates:
                    day_df = df[df['DATE'] == date]
                    day_total = len(day_df)
                    
                    if day_total == 0:
                        # 데이터 없는 날
                        daily_missing.append({
                            'date': date.strftime('%Y-%m-%d'),
                            'rows': 0,
                            'missing_cells': 0,
                            'pct': 100.0,  # 전체 결측으로 표시
                            'no_data': True
                        })
                    else:
                        day_missing = int(day_df[analysis_cols].isnull().sum().sum())
                        day_pct = round((day_missing / (day_total * num_cols)) * 100, 2)
                        daily_missing.append({
                            'date': date.strftime('%Y-%m-%d'),
                            'rows': day_total,
                            'missing_cells': day_missing,
                            'pct': day_pct,
                            'no_data': False
                        })
        analysis['daily'] = daily_missing
        
        # 시간대별 결측치
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
                else:
                    hourly_missing.append({
                        'hour': f'{hour:02d}:00',
                        'rows': 0,
                        'missing_cells': 0,
                        'pct': 0
                    })
        analysis['hourly'] = hourly_missing
        
        # 핵심 컬럼
        key_cols = [
            ('M14AM14B', 'M14AM14B'),
            ('M14AM14BSUM', 'M14AM14BSUM'),
            ('M10AM14A', 'M10AM14A'),
            ('TOTALCNT', 'TOTALCNT'),
            ('M14.QUE.ALL.TRANSPORT4MINOVERCNT', 'TRANSPORT'),
            ('M14.QUE.OHT.OHTUTIL', 'OHT_UTIL'),
            ('M14.QUE.ALL.CURRENTQCREATED', 'Q_CREATED'),
            ('M14.QUE.ALL.CURRENTQCOMPLETED', 'Q_COMPLETED'),
        ]
        
        key_status = []
        for col, short in key_cols:
            if col in df.columns:
                m = int(df[col].isnull().sum())
                p = round(m / total * 100, 2)
                status = 'OK' if p == 0 else 'WARN' if p < 1 else 'ERROR'
                key_status.append({'column': col, 'short': short, 'missing': m, 'pct': p, 'status': status, 'exists': True})
            else:
                key_status.append({'column': col, 'short': short, 'missing': total, 'pct': 100, 'status': 'ERROR', 'exists': False})
        analysis['key_columns'] = key_status
        
        # 수집 간격
        gaps = []
        gap_stats = {'total_gaps': 0, 'max_gap': 0, 'total_missing_min': 0}
        
        if 'CURRTIME' in df.columns:
            df_sorted = df.dropna(subset=['CURRTIME']).sort_values('CURRTIME').reset_index(drop=True)
            if len(df_sorted) > 1:
                df_sorted['time_diff'] = df_sorted['CURRTIME'].diff().dt.total_seconds() / 60
                gap_df = df_sorted[df_sorted['time_diff'] > 1.5].copy()
                
                for idx, row in gap_df.head(50).iterrows():
                    prev_idx = idx - 1
                    if prev_idx >= 0 and prev_idx in df_sorted.index:
                        start_time = df_sorted.loc[prev_idx, 'CURRTIME']
                        gaps.append({
                            'start': start_time.strftime('%Y-%m-%d %H:%M') if pd.notna(start_time) else '-',
                            'end': row['CURRTIME'].strftime('%Y-%m-%d %H:%M') if pd.notna(row['CURRTIME']) else '-',
                            'gap_min': round(row['time_diff'], 1)
                        })
                
                if len(gap_df) > 0:
                    gap_stats = {
                        'total_gaps': len(gap_df),
                        'max_gap': round(gap_df['time_diff'].max(), 1),
                        'total_missing_min': round(gap_df['time_diff'].sum(), 0)
                    }
        
        analysis['gaps'] = gaps
        analysis['gap_stats'] = gap_stats
        
        analysis['duplicates'] = {
            'full': int(df.duplicated().sum()),
            'time': int(df['CURRTIME'].duplicated().sum()) if 'CURRTIME' in df.columns else 0
        }
        
        num_cols = len(analysis_cols)
        total_cells = total * num_cols
        total_missing = df[analysis_cols].isnull().sum().sum()
        
        analysis['summary'] = {
            'total_cells': int(total_cells),
            'missing_cells': int(total_missing),
            'missing_pct': round(total_missing / total_cells * 100, 2) if total_cells > 0 else 0,
            'columns_with_missing': len(analysis['missing_columns']),
            'columns_complete': len([c for c in col_summary if c['missing'] == 0])
        }
        
        return analysis
    
    def save_csv(self, analysis):
        pd.DataFrame(analysis['columns']).to_csv(
            f'{self.output_dir}/missing_by_column.csv', index=False, encoding='utf-8-sig')
        
        if analysis['daily']:
            pd.DataFrame(analysis['daily']).to_csv(
                f'{self.output_dir}/missing_by_date.csv', index=False, encoding='utf-8-sig')
        
        if analysis['hourly']:
            pd.DataFrame(analysis['hourly']).to_csv(
                f'{self.output_dir}/missing_by_hour.csv', index=False, encoding='utf-8-sig')
        
        pd.DataFrame(analysis['key_columns']).to_csv(
            f'{self.output_dir}/key_columns_status.csv', index=False, encoding='utf-8-sig')
        
        if analysis['gaps']:
            pd.DataFrame(analysis['gaps']).to_csv(
                f'{self.output_dir}/collection_gaps.csv', index=False, encoding='utf-8-sig')
    
    def generate_html(self, analysis):
        a = analysis
        
        # 차트 데이터
        daily_labels = [d['date'] for d in a['daily']]
        daily_pcts = [d['pct'] for d in a['daily']]
        daily_rows_data = [d['rows'] for d in a['daily']]
        daily_no_data = [d.get('no_data', False) for d in a['daily']]
        
        # 데이터 없는 날 표시용 색상
        daily_colors = []
        for d in a['daily']:
            if d.get('no_data', False):
                daily_colors.append('#ef4444')  # 빨강 (데이터 없음)
            elif d['pct'] > 5:
                daily_colors.append('#f59e0b')  # 주황
            elif d['pct'] > 0:
                daily_colors.append('#fbbf24')  # 노랑
            else:
                daily_colors.append('#22c55e')  # 초록
        
        hourly_labels = [h['hour'] for h in a['hourly']]
        hourly_data = [h['rows'] for h in a['hourly']]
        
        # 결측 컬럼 TOP 30
        top_missing = a['missing_columns'][:30]
        top_labels = [c['column'][:40] for c in top_missing]
        top_data = [c['pct'] for c in top_missing]
        top_colors = ['#ef4444' if c['pct'] > 10 else '#f59e0b' if c['pct'] > 1 else '#22c55e' for c in top_missing]
        
        key_labels = [k['short'] for k in a['key_columns']]
        key_data = [k['pct'] for k in a['key_columns']]
        key_colors = ['#22c55e' if k['pct']==0 else '#f59e0b' if k['pct']<1 else '#ef4444' for k in a['key_columns']]
        
        # 핵심 컬럼 카드
        key_cards = ""
        for k in a['key_columns']:
            icon_class = "ok" if k['pct'] == 0 else "warn" if k['pct'] < 1 else "error"
            status_icon = "[OK]" if k['pct'] == 0 else "[!]" if k['pct'] < 1 else "[X]"
            if not k['exists']:
                stat_text = "컬럼 없음!"
            else:
                stat_text = "결측: {:,}개 ({:.2f}%)".format(k['missing'], k['pct'])
            key_cards += '<div class="key-card"><div class="icon {}">{}</div><div class="info"><div class="name">{}</div><div class="stat">{}</div></div></div>'.format(
                icon_class, status_icon, k['short'], stat_text)
        
        # 컬럼 테이블
        col_rows = ""
        for i, c in enumerate(a['columns']):
            pclass = "progress-error" if c['pct'] > 10 else "progress-warn" if c['pct'] > 1 else "progress-ok"
            bclass = "badge-error" if c['pct'] > 10 else "badge-warn" if c['pct'] > 1 else "badge-ok"
            btext = "심각" if c['pct'] > 10 else "주의" if c['pct'] > 1 else "양호"
            pwidth = min(c['pct'], 100)
            col_rows += '<tr><td>{}</td><td style="max-width:300px;word-break:break-all;">{}</td><td>{}</td><td>{:,}</td><td>{}%<div class="progress"><div class="progress-fill {}" style="width:{}%"></div></div></td><td>{:,}</td><td><span class="badge {}">{}</span></td></tr>'.format(
                i+1, c['column'], c['dtype'], c['missing'], c['pct'], pclass, pwidth, c['valid'], bclass, btext)
        
        # 날짜별 테이블
        daily_rows = ""
        for d in a['daily']:
            if d.get('no_data', False):
                daily_rows += '<tr style="background:rgba(239,68,68,0.2);"><td>{}</td><td style="color:#ef4444;font-weight:bold;">데이터 없음</td><td>-</td><td style="color:#ef4444;">100%</td><td><span class="badge badge-error">미수집</span></td></tr>'.format(d['date'])
            else:
                pclass = "progress-error" if d['pct'] > 5 else "progress-warn" if d['pct'] > 1 else "progress-ok"
                bclass = "badge-error" if d['pct'] > 5 else "badge-warn" if d['pct'] > 1 else "badge-ok"
                btext = "심각" if d['pct'] > 5 else "주의" if d['pct'] > 1 else "양호"
                pwidth = min(d['pct'] * 10, 100)
                daily_rows += '<tr><td>{}</td><td>{:,}</td><td>{:,}</td><td>{}%<div class="progress"><div class="progress-fill {}" style="width:{}%"></div></div></td><td><span class="badge {}">{}</span></td></tr>'.format(
                    d['date'], d['rows'], d['missing_cells'], d['pct'], pclass, pwidth, bclass, btext)
        
        # 미수집 구간 테이블
        gap_rows = ""
        for i, g in enumerate(a['gaps']):
            bclass = "badge-error" if g['gap_min'] > 10 else "badge-warn"
            btext = "심각" if g['gap_min'] > 10 else "주의"
            gap_rows += '<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td><span class="badge {}">{}</span></td></tr>'.format(
                i+1, g['start'], g['end'], g['gap_min'], bclass, btext)
        
        gap_section = ""
        if a['gaps']:
            gap_section = '''
    <div class="section">
        <h2>미수집 구간 ({}건, 총 {:.0f}분)</h2>
        <div class="info-box"><p><strong>최대 간격:</strong> {}분 | 1.5분 초과 구간만 표시</p></div>
        <div class="table-wrap">
            <table>
                <thead><tr><th>#</th><th>시작</th><th>종료</th><th>간격(분)</th><th>상태</th></tr></thead>
                <tbody>{}</tbody>
            </table>
        </div>
    </div>'''.format(a['gap_stats']['total_gaps'], a['gap_stats']['total_missing_min'], a['gap_stats']['max_gap'], gap_rows)
        
        # 상태 클래스
        missing_pct_class = "warning" if a['summary']['missing_pct'] > 1 else "success"
        missing_col_class = "danger" if a['summary']['columns_with_missing'] > 10 else "warning" if a['summary']['columns_with_missing'] > 0 else "success"
        gap_class = "danger" if a['gap_stats']['total_gaps'] > 10 else "warning" if a['gap_stats']['total_gaps'] > 0 else "success"
        dup_full_class = "danger" if a['duplicates']['full'] > 0 else "success"
        dup_time_class = "warning" if a['duplicates']['time'] > 0 else "success"
        
        # 데이터 없는 날 수
        no_data_days = sum(1 for d in a['daily'] if d.get('no_data', False))
        
        html = '''<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AMHS 결측치 분석</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Malgun Gothic', 'Segoe UI', sans-serif; background: linear-gradient(135deg, #1a1a2e, #16213e); color: #e4e4e7; min-height: 100vh; padding: 20px; }
        .container { max-width: 1800px; margin: 0 auto; }
        header { text-align: center; padding: 30px; margin-bottom: 30px; background: rgba(255,255,255,0.05); border-radius: 16px; border: 1px solid rgba(255,255,255,0.1); }
        header h1 { font-size: 2rem; margin-bottom: 10px; }
        header p { color: #a1a1aa; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 16px; margin-bottom: 24px; }
        .stat-card { background: rgba(255,255,255,0.05); border-radius: 12px; padding: 18px; text-align: center; border: 1px solid rgba(255,255,255,0.1); }
        .stat-card:hover { transform: translateY(-3px); transition: 0.2s; }
        .stat-card .value { font-size: 1.6rem; font-weight: bold; color: #60a5fa; }
        .stat-card .label { color: #a1a1aa; margin-top: 4px; font-size: 0.85rem; }
        .stat-card.warning .value { color: #fbbf24; }
        .stat-card.danger .value { color: #f87171; }
        .stat-card.success .value { color: #4ade80; }
        .section { background: rgba(255,255,255,0.05); border-radius: 16px; padding: 24px; margin-bottom: 24px; border: 1px solid rgba(255,255,255,0.1); }
        .section h2 { font-size: 1.2rem; margin-bottom: 16px; padding-bottom: 10px; border-bottom: 1px solid rgba(255,255,255,0.1); }
        .chart-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-bottom: 24px; }
        .chart-box { background: rgba(255,255,255,0.05); border-radius: 16px; padding: 20px; border: 1px solid rgba(255,255,255,0.1); }
        .chart-box.full { grid-column: 1 / -1; }
        .chart-box h3 { margin-bottom: 16px; color: #e4e4e7; font-size: 1.1rem; }
        .chart-container { position: relative; height: 300px; }
        .chart-container.tall { height: 400px; }
        table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
        th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.1); }
        th { background: rgba(0,0,0,0.3); font-weight: 600; color: #a1a1aa; font-size: 0.75rem; position: sticky; top: 0; }
        tr:hover { background: rgba(255,255,255,0.05); }
        .badge { padding: 3px 10px; border-radius: 12px; font-size: 0.7rem; font-weight: 600; }
        .badge-ok { background: #166534; color: #4ade80; }
        .badge-warn { background: #854d0e; color: #fbbf24; }
        .badge-error { background: #991b1b; color: #fca5a5; }
        .progress { width: 60px; height: 5px; background: rgba(255,255,255,0.1); border-radius: 3px; overflow: hidden; display: inline-block; vertical-align: middle; margin-left: 6px; }
        .progress-fill { height: 100%; border-radius: 3px; }
        .progress-ok { background: #22c55e; }
        .progress-warn { background: #f59e0b; }
        .progress-error { background: #ef4444; }
        .key-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; }
        .key-card { background: rgba(0,0,0,0.2); border-radius: 10px; padding: 14px; display: flex; align-items: center; gap: 12px; }
        .key-card .icon { font-size: 1rem; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; border-radius: 8px; font-weight: bold; }
        .key-card .icon.ok { background: rgba(34,197,94,0.2); color: #4ade80; }
        .key-card .icon.warn { background: rgba(245,158,11,0.2); color: #fbbf24; }
        .key-card .icon.error { background: rgba(239,68,68,0.2); color: #f87171; }
        .key-card .info { flex: 1; }
        .key-card .name { font-weight: 600; font-size: 0.85rem; }
        .key-card .stat { color: #a1a1aa; font-size: 0.75rem; margin-top: 2px; }
        .table-wrap { max-height: 450px; overflow-y: auto; }
        .table-wrap::-webkit-scrollbar { width: 6px; }
        .table-wrap::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.2); border-radius: 3px; }
        .info-box { background: rgba(59,130,246,0.1); border: 1px solid rgba(59,130,246,0.3); border-radius: 10px; padding: 12px; margin-bottom: 16px; font-size: 0.9rem; }
        .info-box strong { color: #60a5fa; }
        .legend { display: flex; gap: 16px; margin-top: 12px; justify-content: center; flex-wrap: wrap; }
        .legend-item { display: flex; align-items: center; gap: 6px; font-size: 0.8rem; color: #a1a1aa; }
        .legend-color { width: 14px; height: 14px; border-radius: 3px; }
        footer { text-align: center; padding: 24px; color: #71717a; font-size: 0.85rem; }
        @media (max-width: 1024px) { .chart-grid { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
<div class="container">
    <header>
        <h1>AMHS 결측치 분석 대시보드</h1>
        <p>파일: ''' + a['info']['file'] + ''' | 분석시간: ''' + a['info']['analysis_time'] + '''</p>
    </header>
    
    <div class="info-box">
        <p><strong>분석 기간:</strong> ''' + a['info']['date_start'] + ''' ~ ''' + a['info']['date_end'] + '''</p>
        <p><strong>데이터:</strong> ''' + "{:,}".format(a['info']['total_rows']) + '''행 x ''' + str(a['info']['total_cols']) + '''열 = ''' + "{:,}".format(a['summary']['total_cells']) + '''셀</p>
    </div>
    
    <div class="stats-grid">
        <div class="stat-card"><div class="value">''' + "{:,}".format(a['info']['total_rows']) + '''</div><div class="label">전체 행</div></div>
        <div class="stat-card"><div class="value">''' + str(a['info']['total_cols']) + '''</div><div class="label">전체 컬럼</div></div>
        <div class="stat-card ''' + missing_pct_class + '''"><div class="value">''' + str(a['summary']['missing_pct']) + '''%</div><div class="label">전체 결측률</div></div>
        <div class="stat-card ''' + missing_col_class + '''"><div class="value">''' + str(a['summary']['columns_with_missing']) + '''</div><div class="label">결측 컬럼</div></div>
        <div class="stat-card success"><div class="value">''' + str(a['summary']['columns_complete']) + '''</div><div class="label">완전 컬럼</div></div>
        <div class="stat-card ''' + gap_class + '''"><div class="value">''' + str(a['gap_stats']['total_gaps']) + '''</div><div class="label">미수집 구간</div></div>
        <div class="stat-card ''' + ("danger" if no_data_days > 0 else "success") + '''"><div class="value">''' + str(no_data_days) + '''</div><div class="label">미수집일</div></div>
    </div>
    
    <div class="section">
        <h2>핵심 컬럼 상태 (V8.3)</h2>
        <div class="key-grid">''' + key_cards + '''</div>
    </div>
    
    <div class="chart-grid">
        <div class="chart-box full">
            <h3>날짜별 데이터 수집 현황</h3>
            <div class="chart-container tall"><canvas id="dailyChart"></canvas></div>
            <div class="legend">
                <div class="legend-item"><div class="legend-color" style="background:#22c55e;"></div>정상</div>
                <div class="legend-item"><div class="legend-color" style="background:#fbbf24;"></div>일부 결측</div>
                <div class="legend-item"><div class="legend-color" style="background:#f59e0b;"></div>결측 많음</div>
                <div class="legend-item"><div class="legend-color" style="background:#ef4444;"></div>데이터 없음</div>
            </div>
        </div>
        <div class="chart-box">
            <h3>시간대별 데이터 수집량</h3>
            <div class="chart-container"><canvas id="hourlyChart"></canvas></div>
        </div>
        <div class="chart-box">
            <h3>핵심 컬럼 결측률</h3>
            <div class="chart-container"><canvas id="keyChart"></canvas></div>
        </div>
    </div>
    
    <div class="chart-grid">
        <div class="chart-box full">
            <h3>결측 컬럼 TOP 30</h3>
            <div class="chart-container tall"><canvas id="topChart"></canvas></div>
        </div>
    </div>
    
    <div class="section">
        <h2>전체 컬럼 현황 (''' + str(len(a['columns'])) + '''개) - 결측: ''' + str(len(a['missing_columns'])) + '''개 / 완전: ''' + str(a['summary']['columns_complete']) + '''개</h2>
        <div class="table-wrap">
            <table>
                <thead><tr><th>#</th><th>컬럼명</th><th>타입</th><th>결측</th><th>결측률</th><th>유효</th><th>상태</th></tr></thead>
                <tbody>''' + col_rows + '''</tbody>
            </table>
        </div>
    </div>
    
    <div class="section">
        <h2>날짜별 현황 (''' + str(len(a['daily'])) + '''일)</h2>
        <div class="table-wrap">
            <table>
                <thead><tr><th>날짜</th><th>데이터</th><th>결측 셀</th><th>결측률</th><th>상태</th></tr></thead>
                <tbody>''' + daily_rows + '''</tbody>
            </table>
        </div>
    </div>
    
    ''' + gap_section + '''
    
    <div class="section">
        <h2>중복 데이터</h2>
        <div class="stats-grid" style="max-width:400px;">
            <div class="stat-card ''' + dup_full_class + '''"><div class="value">''' + "{:,}".format(a['duplicates']['full']) + '''</div><div class="label">완전 중복</div></div>
            <div class="stat-card ''' + dup_time_class + '''"><div class="value">''' + "{:,}".format(a['duplicates']['time']) + '''</div><div class="label">시간 중복</div></div>
        </div>
    </div>
    
    <footer>AMHS 결측치 분석 대시보드 | V8.3</footer>
</div>

<script>
const colors = { primary: 'rgba(96,165,250,1)', primaryBg: 'rgba(96,165,250,0.3)', grid: 'rgba(255,255,255,0.1)', text: '#a1a1aa' };
Chart.defaults.color = colors.text;
Chart.defaults.borderColor = colors.grid;

// 날짜별 차트 (막대 - 데이터 수)
new Chart(document.getElementById('dailyChart'), {
    type: 'bar',
    data: {
        labels: ''' + json.dumps(daily_labels) + ''',
        datasets: [{
            label: '데이터 수',
            data: ''' + json.dumps(daily_rows_data) + ''',
            backgroundColor: ''' + json.dumps(daily_colors) + ''',
            borderRadius: 4
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { display: false },
            tooltip: {
                callbacks: {
                    label: function(ctx) {
                        const idx = ctx.dataIndex;
                        const noData = ''' + json.dumps(daily_no_data) + '''[idx];
                        if (noData) return '데이터 없음 (미수집)';
                        return '데이터: ' + ctx.raw.toLocaleString() + '개';
                    }
                }
            }
        },
        scales: {
            y: { beginAtZero: true, title: { display: true, text: '데이터 수' } },
            x: { 
                ticks: { 
                    maxRotation: 45, 
                    minRotation: 45,
                    autoSkip: true,
                    maxTicksLimit: 30
                }
            }
        }
    }
});

// 시간대별 차트
new Chart(document.getElementById('hourlyChart'), {
    type: 'bar',
    data: {
        labels: ''' + json.dumps(hourly_labels) + ''',
        datasets: [{
            label: '데이터 수',
            data: ''' + json.dumps(hourly_data) + ''',
            backgroundColor: colors.primaryBg,
            borderColor: colors.primary,
            borderWidth: 1,
            borderRadius: 4
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: { y: { beginAtZero: true } }
    }
});

// 핵심 컬럼 차트
new Chart(document.getElementById('keyChart'), {
    type: 'bar',
    data: {
        labels: ''' + json.dumps(key_labels) + ''',
        datasets: [{
            label: '결측률(%)',
            data: ''' + json.dumps(key_data) + ''',
            backgroundColor: ''' + json.dumps(key_colors) + ''',
            borderRadius: 4
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: { y: { beginAtZero: true, max: Math.max(10, Math.max(...''' + json.dumps(key_data) + ''') * 1.2) } }
    }
});

// TOP 결측 컬럼 차트
new Chart(document.getElementById('topChart'), {
    type: 'bar',
    data: {
        labels: ''' + json.dumps(top_labels) + ''',
        datasets: [{
            label: '결측률(%)',
            data: ''' + json.dumps(top_data) + ''',
            backgroundColor: ''' + json.dumps(top_colors) + ''',
            borderRadius: 4
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: 'y',
        plugins: { legend: { display: false } },
        scales: { x: { beginAtZero: true } }
    }
});
</script>
</body>
</html>'''
        
        output_path = f'{self.output_dir}/missing_dashboard.html'
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
    
    def open_html(self):
        html_path = os.path.abspath(f'{self.output_dir}/missing_dashboard.html')
        if os.path.exists(html_path):
            webbrowser.open(f'file://{html_path}')
        else:
            messagebox.showwarning("경고", "HTML 파일이 없습니다. 분석을 먼저 실행하세요.")
    
    def open_folder(self):
        folder_path = os.path.abspath(self.output_dir)
        if os.path.exists(folder_path):
            if os.name == 'nt':
                os.startfile(folder_path)
            elif os.name == 'posix':
                import subprocess
                subprocess.run(['open' if os.uname().sysname == 'Darwin' else 'xdg-open', folder_path])
        else:
            messagebox.showwarning("경고", "결과 폴더가 없습니다. 분석을 먼저 실행하세요.")


def main():
    root = tk.Tk()
    app = MissingAnalyzerUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()