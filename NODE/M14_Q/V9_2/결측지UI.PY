#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AMHS 결측치 분석 UI (v8)
- v5 기능 그대로 유지
- 날짜/시간:분/컬럼별 결측 추가 (날짜별 10개씩 표시)
- CSV: date_time_column_missing.csv 추가
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
        self.root.title("AMHS 결측치 분석 도구 v8")
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
        style.configure('Treeview', background='#16213e', foreground='#e4e4e7', fieldbackground='#16213e', font=('Consolas', 9), rowheight=25)
        style.configure('Treeview.Heading', font=('Segoe UI', 10, 'bold'), background='#0f172a', foreground='#60a5fa')
        style.map('Treeview', background=[('selected', '#3b82f6')])
    
    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(main_frame, text="[AMHS] 결측치 분석 도구", style='Header.TLabel').pack(anchor=tk.W, pady=(0, 15))
        
        file_frame = ttk.Frame(main_frame)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(file_frame, text="CSV:", style='Sub.TLabel').pack(side=tk.LEFT)
        self.file_label = ttk.Label(file_frame, text="파일을 선택하세요", style='Sub.TLabel')
        self.file_label.pack(side=tk.LEFT, padx=10)
        ttk.Button(file_frame, text="파일 선택", command=self.select_file).pack(side=tk.RIGHT)
        
        self.info_label = ttk.Label(main_frame, text="", style='Status.TLabel')
        self.info_label.pack(anchor=tk.W, pady=(0, 10))
        
        tree_frame = ttk.Frame(main_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.tree = ttk.Treeview(tree_frame, columns=('idx', 'column', 'dtype', 'null', 'null_pct'), show='headings', height=12)
        self.tree.heading('idx', text='#')
        self.tree.heading('column', text='컬럼명')
        self.tree.heading('dtype', text='타입')
        self.tree.heading('null', text='결측')
        self.tree.heading('null_pct', text='결측률')
        self.tree.column('idx', width=40, anchor='center')
        self.tree.column('column', width=400, anchor='w')
        self.tree.column('dtype', width=80, anchor='center')
        self.tree.column('null', width=80, anchor='center')
        self.tree.column('null_pct', width=80, anchor='center')
        
        sb = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X)
        self.analyze_btn = ttk.Button(btn_frame, text="분석 실행", command=self.run_analysis, state=tk.DISABLED)
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        self.open_html_btn = ttk.Button(btn_frame, text="HTML 열기", command=self.open_html, state=tk.DISABLED)
        self.open_html_btn.pack(side=tk.LEFT, padx=5)
        self.open_folder_btn = ttk.Button(btn_frame, text="결과 폴더", command=self.open_folder, state=tk.DISABLED)
        self.open_folder_btn.pack(side=tk.LEFT, padx=5)
        
        self.progress_var = tk.StringVar(value="")
        ttk.Label(btn_frame, textvariable=self.progress_var, style='Status.TLabel').pack(side=tk.RIGHT, padx=10)
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=(10, 0))
    
    def select_file(self):
        fp = filedialog.askopenfilename(title="CSV 파일 선택", filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if fp:
            self.file_path = fp
            self.file_label.config(text=os.path.basename(fp))
            self.load_csv()
    
    def load_csv(self):
        try:
            self.progress_var.set("로딩...")
            self.progress.start()
            self.root.update()
            self.df = pd.read_csv(self.file_path, on_bad_lines='skip')
            if 'CURRTIME' in self.df.columns:
                self.df['CURRTIME'] = pd.to_datetime(self.df['CURRTIME'].astype(str).str.strip(), format='%Y%m%d%H%M', errors='coerce')
            self.info_label.config(text=f"{len(self.df):,}행 x {self.df.shape[1]}열")
            self.update_tree()
            self.analyze_btn.config(state=tk.NORMAL)
            self.progress.stop()
            self.progress_var.set("로드 완료!")
        except Exception as e:
            self.progress.stop()
            messagebox.showerror("오류", str(e))
    
    def update_tree(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        total = len(self.df)
        for i, col in enumerate(self.df.columns):
            null = self.df[col].isna().sum()
            pct = f"{null/total*100:.2f}%"
            tag = 'err' if null/total > 0.1 else 'warn' if null > 0 else 'ok'
            self.tree.insert('', 'end', values=(i+1, col, str(self.df[col].dtype), f"{null:,}", pct), tags=(tag,))
        self.tree.tag_configure('err', foreground='#f87171')
        self.tree.tag_configure('warn', foreground='#fbbf24')
        self.tree.tag_configure('ok', foreground='#4ade80')
    
    def run_analysis(self):
        self.analyze_btn.config(state=tk.DISABLED)
        self.progress.start()
        self.progress_var.set("분석 중...")
        threading.Thread(target=self._analyze).start()
    
    def _analyze(self):
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            analysis = self.analyze_data()
            self.root.after(0, lambda: self.progress_var.set("저장 중..."))
            self.save_csv(analysis)
            self.generate_html(analysis)
            self.root.after(0, self._done)
        except Exception as e:
            self.root.after(0, lambda: self._error(str(e)))
    
    def _done(self):
        self.progress.stop()
        self.progress_var.set("완료!")
        self.analyze_btn.config(state=tk.NORMAL)
        self.open_html_btn.config(state=tk.NORMAL)
        self.open_folder_btn.config(state=tk.NORMAL)
        if messagebox.askyesno("완료", "HTML 열까요?"):
            self.open_html()
    
    def _error(self, e):
        self.progress.stop()
        self.analyze_btn.config(state=tk.NORMAL)
        messagebox.showerror("오류", e)
    
    def analyze_data(self):
        df = self.df.copy()
        total = len(df)
        
        exclude = ['DATE', 'HOUR']
        if 'CURRTIME' in df.columns:
            df['DATE'] = df['CURRTIME'].dt.date
            df['HOUR'] = df['CURRTIME'].dt.hour
        
        cols = [c for c in df.columns if c not in exclude]
        
        ds, de = '-', '-'
        if 'CURRTIME' in df.columns:
            vt = df['CURRTIME'].dropna()
            if len(vt) > 0:
                ds = vt.min().strftime('%Y-%m-%d %H:%M')
                de = vt.max().strftime('%Y-%m-%d %H:%M')
        
        a = {'info': {'file': os.path.basename(self.file_path), 'total_rows': total, 'total_cols': len(cols), 'date_start': ds, 'date_end': de, 'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}
        
        # 컬럼별
        mc = df[cols].isnull().sum()
        mp = (mc / total * 100).round(2)
        
        col_sum = []
        col_daily = {}
        for c in cols:
            col_sum.append({'column': c, 'missing': int(mc[c]), 'pct': float(mp[c]), 'valid': int(total - mc[c]), 'dtype': str(df[c].dtype)})
            if mc[c] > 0 and 'DATE' in df.columns:
                dd = []
                for date in sorted(df['DATE'].dropna().unique()):
                    ddf = df[df['DATE'] == date]
                    dm = int(ddf[c].isnull().sum())
                    if dm > 0:
                        dd.append({'date': date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date), 'missing': dm, 'total': len(ddf), 'pct': round(dm/len(ddf)*100, 2)})
                col_daily[c] = dd
        
        col_sum = sorted(col_sum, key=lambda x: x['pct'], reverse=True)
        a['columns'] = col_sum
        a['missing_columns'] = [c for c in col_sum if c['missing'] > 0]
        a['col_daily'] = col_daily
        a['cols'] = cols
        
        # 날짜별 + 매트릭스
        daily = []
        matrix = {}
        date_col_detail = {}
        
        if 'CURRTIME' in df.columns:
            vt = df['CURRTIME'].dropna()
            if len(vt) > 0:
                mind, maxd = vt.min().date(), vt.max().date()
                alld = []
                cur = mind
                while cur <= maxd:
                    alld.append(cur)
                    cur += timedelta(days=1)
                
                nc = len(cols)
                for date in alld:
                    dstr = date.strftime('%Y-%m-%d')
                    ddf = df[df['DATE'] == date]
                    dt = len(ddf)
                    
                    row = {'날짜': dstr, '데이터수': dt}
                    col_missing_list = []
                    
                    if dt == 0:
                        daily.append({'date': dstr, 'rows': 0, 'cells': 0, 'missing': 0, 'pct': 100.0, 'no_data': True, 'mcols': nc})
                        for c in cols:
                            row[c] = '-'
                            col_missing_list.append({'column': c, 'missing': 0, 'total': 0, 'pct': 100})
                    else:
                        tc = dt * nc
                        tmiss = int(ddf[cols].isnull().sum().sum())
                        mcnt = int((ddf[cols].isnull().sum() > 0).sum())
                        dp = round((tmiss / tc) * 100, 2) if tc > 0 else 0
                        daily.append({'date': dstr, 'rows': dt, 'cells': tc, 'missing': tmiss, 'pct': dp, 'no_data': False, 'mcols': mcnt})
                        
                        for c in cols:
                            m = int(ddf[c].isnull().sum())
                            row[c] = m if m > 0 else 0
                            if m > 0:
                                col_missing_list.append({'column': c, 'missing': m, 'total': dt, 'pct': round(m/dt*100, 2)})
                    
                    matrix[dstr] = row
                    date_col_detail[dstr] = col_missing_list
        
        a['daily'] = daily
        a['matrix'] = matrix
        a['date_col_detail'] = date_col_detail
        
        # 간격
        gaps = []
        gs = {'cnt': 0, 'max': 0, 'sum': 0}
        if 'CURRTIME' in df.columns:
            ds_df = df.dropna(subset=['CURRTIME']).sort_values('CURRTIME').reset_index(drop=True)
            if len(ds_df) > 1:
                ds_df['td'] = ds_df['CURRTIME'].diff().dt.total_seconds() / 60
                gd = ds_df[ds_df['td'] > 1.5]
                for idx, row in gd.head(50).iterrows():
                    pi = idx - 1
                    if pi >= 0 and pi in ds_df.index:
                        st = ds_df.loc[pi, 'CURRTIME']
                        gaps.append({'start': st.strftime('%Y-%m-%d %H:%M') if pd.notna(st) else '-', 'end': row['CURRTIME'].strftime('%Y-%m-%d %H:%M') if pd.notna(row['CURRTIME']) else '-', 'gap': round(row['td'], 1)})
                if len(gd) > 0:
                    gs = {'cnt': len(gd), 'max': round(gd['td'].max(), 1), 'sum': round(gd['td'].sum(), 0)}
        a['gaps'] = gaps
        a['gap_stats'] = gs
        
        a['dup'] = {'full': int(df.duplicated().sum()), 'time': int(df['CURRTIME'].duplicated().sum()) if 'CURRTIME' in df.columns else 0}
        
        # 🔥 날짜/시간:분/컬럼별 결측 상세
        datetime_col_missing = []
        if 'CURRTIME' in df.columns:
            for idx, row in df.iterrows():
                if pd.isna(row['CURRTIME']):
                    continue
                date_str = row['CURRTIME'].strftime('%Y-%m-%d')
                time_str = row['CURRTIME'].strftime('%H:%M')
                
                for c in cols:
                    if pd.isna(row[c]):
                        datetime_col_missing.append({
                            'date': date_str,
                            'time': time_str,
                            'column': c
                        })
        
        a['datetime_col_missing'] = datetime_col_missing
        
        tc = total * len(cols)
        tm = df[cols].isnull().sum().sum()
        a['summary'] = {'cells': int(tc), 'missing': int(tm), 'pct': round(tm/tc*100, 2) if tc > 0 else 0, 'mcols': len(a['missing_columns']), 'ccols': len([c for c in col_sum if c['missing'] == 0])}
        
        return a
    
    def save_csv(self, a):
        pd.DataFrame(a['columns']).to_csv(f'{self.output_dir}/missing_by_column.csv', index=False, encoding='utf-8-sig')
        if a['daily']:
            pd.DataFrame(a['daily']).to_csv(f'{self.output_dir}/missing_by_date.csv', index=False, encoding='utf-8-sig')
        if a['matrix']:
            rows = list(a['matrix'].values())
            dfm = pd.DataFrame(rows)
            order = ['날짜', '데이터수'] + a['cols']
            order = [c for c in order if c in dfm.columns]
            dfm[order].to_csv(f'{self.output_dir}/date_column_matrix.csv', index=False, encoding='utf-8-sig')
        if a['gaps']:
            pd.DataFrame(a['gaps']).to_csv(f'{self.output_dir}/collection_gaps.csv', index=False, encoding='utf-8-sig')
        
        dr = []
        for col, details in a['col_daily'].items():
            for d in details:
                dr.append({'column': col, 'date': d['date'], 'missing': d['missing'], 'total': d['total'], 'pct': d['pct']})
        if dr:
            pd.DataFrame(dr).to_csv(f'{self.output_dir}/column_daily_detail.csv', index=False, encoding='utf-8-sig')
        
        # 🔥 날짜/시간:분/컬럼별 결측 CSV
        if a['datetime_col_missing']:
            pd.DataFrame(a['datetime_col_missing']).to_csv(f'{self.output_dir}/date_time_column_missing.csv', index=False, encoding='utf-8-sig')
    
    def generate_datetime_col_table(self, dtc_data, daily):
        """날짜별로 그룹핑해서 10개씩 테이블 생성"""
        if not dtc_data:
            return '<p style="color:#4ade80;padding:16px">결측 없음</p>'
        
        # 날짜별 그룹핑
        by_date = {}
        for x in dtc_data:
            d = x['date']
            if d not in by_date:
                by_date[d] = []
            by_date[d].append(x)
        
        html = '<div class="tw" style="max-height:500px">'
        for date in sorted(by_date.keys()):
            items = by_date[date]
            total_cnt = len(items)
            show_items = items[:10]
            
            html += '<div style="margin-bottom:16px;background:rgba(0,0,0,.2);padding:12px;border-radius:8px">'
            html += '<div style="font-weight:bold;color:#60a5fa;margin-bottom:8px">{} <span style="color:#a1a1aa;font-weight:normal">({}건{})</span></div>'.format(
                date, total_cnt, ' - 10건만 표시' if total_cnt > 10 else '')
            html += '<table><thead><tr><th>시간</th><th>결측 컬럼</th></tr></thead><tbody>'
            for x in show_items:
                html += '<tr><td>{}</td><td class="cn">{}</td></tr>'.format(x['time'], x['column'])
            html += '</tbody></table>'
            html += '</div>'
        
        html += '</div>'
        return html
    
    def generate_html(self, a):
        # 날짜별 차트 데이터
        dl = [d['date'] for d in a['daily']]
        dp = [d['pct'] for d in a['daily']]
        dmc = [d['mcols'] for d in a['daily']]
        dc = ['#ef4444' if d.get('no_data') else '#f59e0b' if d['pct']>5 else '#fbbf24' if d['pct']>0 else '#22c55e' for d in a['daily']]
        
        # 전체 컬럼 차트 데이터 (결측률 순 정렬)
        all_cols = a['columns']  # 이미 결측률 순 정렬됨
        all_labels = [c['column'][:40] for c in all_cols]
        all_data = [c['pct'] for c in all_cols]
        all_colors = ['#ef4444' if c['pct']>10 else '#f59e0b' if c['pct']>1 else '#22c55e' if c['pct']>0 else '#3b82f6' for c in all_cols]
        all_col_names = [c['column'] for c in all_cols]
        
        # JSON
        col_daily_json = json.dumps(a['col_daily'], ensure_ascii=False)
        date_col_json = json.dumps(a['date_col_detail'], ensure_ascii=False)
        col_data_json = json.dumps({c['column']: c for c in a['columns']}, ensure_ascii=False)
        daily_data_json = json.dumps({d['date']: d for d in a['daily']}, ensure_ascii=False)
        
        # 결측 컬럼 카드
        cards = ""
        for c in a['missing_columns'][:16]:
            ic = "error" if c['pct']>10 else "warn"
            nm = c['column'][:22] + ('...' if len(c['column'])>22 else '')
            cards += '<div class="card" onclick="showColDetail(\'{}\')"><div class="ic {}">!</div><div class="inf"><div class="nm" title="{}">{}</div><div class="st">{:,}개 ({}%)</div></div></div>'.format(c['column'].replace("'","\\'"), ic, c['column'], nm, c['missing'], c['pct'])
        if len(a['missing_columns']) > 16:
            cards += '<div class="card"><div class="ic ok">+</div><div class="inf"><div class="nm">외 {}개</div><div class="st">테이블 확인</div></div></div>'.format(len(a['missing_columns'])-16)
        
        # 컬럼 테이블
        cr = ""
        for i, c in enumerate(a['columns']):
            pc = "pe" if c['pct']>10 else "pw" if c['pct']>1 else "po"
            bc = "be" if c['pct']>10 else "bw" if c['pct']>1 else "bo"
            bt = "심각" if c['pct']>10 else "주의" if c['pct']>1 else "양호"
            cl = 'onclick="showColDetail(\'{}\')" class="clk"'.format(c['column'].replace("'","\\'")) if c['missing']>0 else ''
            cr += '<tr {}><td>{}</td><td class="cn">{}</td><td>{:,}</td><td>{}%<div class="pg"><div class="{}" style="width:{}%"></div></div></td><td><span class="{}">{}</span></td></tr>'.format(cl, i+1, c['column'], c['missing'], c['pct'], pc, min(c['pct'],100), bc, bt)
        
        # 날짜 테이블
        drw = ""
        for d in a['daily']:
            if d.get('no_data'):
                drw += '<tr class="nd clk" onclick="showDateDetail(\'{}\')"><td>{}</td><td>0</td><td>-</td><td>100%</td><td>{}</td><td><span class="be">미수집</span></td></tr>'.format(d['date'], d['date'], d['mcols'])
            else:
                pc = "pe" if d['pct']>5 else "pw" if d['pct']>1 else "po"
                bc = "be" if d['pct']>5 else "bw" if d['pct']>1 else "bo"
                bt = "심각" if d['pct']>5 else "주의" if d['pct']>1 else "양호"
                cl = 'onclick="showDateDetail(\'{}\')" class="clk"'.format(d['date']) if d['mcols']>0 else ''
                drw += '<tr {}><td>{}</td><td>{:,}</td><td>{:,}</td><td>{}%<div class="pg"><div class="{}" style="width:{}%"></div></div></td><td>{}</td><td><span class="{}">{}</span></td></tr>'.format(cl, d['date'], d['rows'], d['missing'], d['pct'], pc, min(d['pct']*10,100), d['mcols'], bc, bt)
        
        # 간격 테이블
        gr = ""
        for i, g in enumerate(a['gaps']):
            bc = "be" if g['gap']>10 else "bw"
            bt = "심각" if g['gap']>10 else "주의"
            gr += '<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td><span class="{}">{}</span></td></tr>'.format(i+1, g['start'], g['end'], g['gap'], bc, bt)
        
        gs = ""
        if a['gaps']:
            gs = '<div class="sec"><h2>미수집 구간 ({}건, {:.0f}분)</h2><div class="ib">최대: {}분</div><div class="tw"><table><thead><tr><th>#</th><th>시작</th><th>종료</th><th>간격</th><th>상태</th></tr></thead><tbody>{}</tbody></table></div></div>'.format(a['gap_stats']['cnt'], a['gap_stats']['sum'], a['gap_stats']['max'], gr)
        
        # 상태
        mpc = "warn" if a['summary']['pct']>1 else "ok"
        mcc = "err" if a['summary']['mcols']>10 else "warn" if a['summary']['mcols']>0 else "ok"
        gc = "err" if a['gap_stats']['cnt']>10 else "warn" if a['gap_stats']['cnt']>0 else "ok"
        ndd = sum(1 for d in a['daily'] if d.get('no_data'))
        ndc = "err" if ndd>0 else "ok"
        
        # 차트 높이 계산 (전체 컬럼 수에 따라)
        chart_height = max(400, len(all_cols) * 18)
        
        html = '''<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AMHS 결측치 분석</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Malgun Gothic',sans-serif;background:linear-gradient(135deg,#1a1a2e,#16213e);color:#e4e4e7;min-height:100vh;padding:16px}
.ct{max-width:1700px;margin:0 auto}
header{text-align:center;padding:20px;margin-bottom:20px;background:rgba(255,255,255,.05);border-radius:12px;border:1px solid rgba(255,255,255,.1)}
header h1{font-size:1.6rem;margin-bottom:6px}
header p{color:#a1a1aa;font-size:.85rem}
.sg{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px;margin-bottom:18px}
.sc{background:rgba(255,255,255,.05);border-radius:10px;padding:12px;text-align:center;border:1px solid rgba(255,255,255,.1)}
.sc .v{font-size:1.2rem;font-weight:bold;color:#60a5fa}
.sc .l{color:#a1a1aa;font-size:.7rem;margin-top:2px}
.sc.warn .v{color:#fbbf24}.sc.err .v{color:#f87171}.sc.ok .v{color:#4ade80}
.sec{background:rgba(255,255,255,.05);border-radius:12px;padding:18px;margin-bottom:18px;border:1px solid rgba(255,255,255,.1)}
.sec h2{font-size:.95rem;margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid rgba(255,255,255,.1)}
.cg{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:8px}
.card{background:rgba(0,0,0,.2);border-radius:8px;padding:10px;display:flex;align-items:center;gap:8px;cursor:pointer}
.card:hover{background:rgba(59,130,246,.2)}
.card .ic{width:28px;height:28px;display:flex;align-items:center;justify-content:center;border-radius:6px;font-weight:bold;font-size:.75rem}
.card .ic.error{background:rgba(239,68,68,.2);color:#f87171}
.card .ic.warn{background:rgba(245,158,11,.2);color:#fbbf24}
.card .ic.ok{background:rgba(34,197,94,.2);color:#4ade80}
.card .inf{flex:1;overflow:hidden}
.card .nm{font-weight:600;font-size:.72rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.card .st{color:#a1a1aa;font-size:.65rem}
.chb{background:rgba(255,255,255,.05);border-radius:12px;padding:16px;border:1px solid rgba(255,255,255,.1);margin-bottom:18px}
.chb h3{margin-bottom:10px;font-size:.9rem}
.chc{position:relative}
table{width:100%;border-collapse:collapse;font-size:.75rem}
th,td{padding:6px 8px;text-align:left;border-bottom:1px solid rgba(255,255,255,.1)}
th{background:rgba(0,0,0,.3);font-weight:600;color:#a1a1aa;font-size:.65rem;position:sticky;top:0}
tr:hover{background:rgba(255,255,255,.05)}
tr.clk{cursor:pointer}
tr.clk:hover{background:rgba(59,130,246,.2)}
tr.nd{background:rgba(239,68,68,.15)}
.cn{max-width:280px;word-break:break-all}
.be{background:#991b1b;color:#fca5a5;padding:2px 6px;border-radius:6px;font-size:.6rem;font-weight:600}
.bw{background:#854d0e;color:#fbbf24;padding:2px 6px;border-radius:6px;font-size:.6rem;font-weight:600}
.bo{background:#166534;color:#4ade80;padding:2px 6px;border-radius:6px;font-size:.6rem;font-weight:600}
.pg{width:40px;height:4px;background:rgba(255,255,255,.1);border-radius:2px;display:inline-block;vertical-align:middle;margin-left:4px;overflow:hidden}
.pe{height:100%;background:#ef4444;border-radius:2px}
.pw{height:100%;background:#f59e0b;border-radius:2px}
.po{height:100%;background:#22c55e;border-radius:2px}
.tw{max-height:350px;overflow-y:auto}
.tw::-webkit-scrollbar{width:5px}
.tw::-webkit-scrollbar-thumb{background:rgba(255,255,255,.2);border-radius:3px}
.ib{background:rgba(59,130,246,.1);border:1px solid rgba(59,130,246,.3);border-radius:8px;padding:8px;margin-bottom:10px;font-size:.8rem}
.ib strong{color:#60a5fa}
.lg{display:flex;gap:12px;margin-top:8px;justify-content:center;flex-wrap:wrap}
.lg span{display:flex;align-items:center;gap:4px;font-size:.7rem;color:#a1a1aa}
.lg i{width:10px;height:10px;border-radius:2px;display:inline-block}
footer{text-align:center;padding:16px;color:#71717a;font-size:.75rem}
.modal{display:none;position:fixed;z-index:1000;left:0;top:0;width:100%;height:100%;background:rgba(0,0,0,.8)}
.mc{background:#1a1a2e;margin:5% auto;padding:18px;border-radius:12px;width:90%;max-width:650px;max-height:80vh;overflow-y:auto;border:1px solid rgba(255,255,255,.1)}
.mh{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;padding-bottom:10px;border-bottom:1px solid rgba(255,255,255,.1)}
.mh h3{font-size:.95rem;color:#60a5fa}
.mx{background:none;border:none;color:#a1a1aa;font-size:1.3rem;cursor:pointer}
.mx:hover{color:#fff}
.ds{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:12px}
.ds div{background:rgba(0,0,0,.2);padding:10px;border-radius:6px;text-align:center}
.ds .dv{font-size:1.1rem;font-weight:bold;color:#60a5fa}
.ds .dl{font-size:.7rem;color:#a1a1aa;margin-top:2px}
.all-chart-wrap{max-height:''' + str(chart_height) + '''px;overflow-y:auto}
</style>
</head>
<body>
<div class="ct">
<header>
<h1>AMHS 결측치 분석 대시보드</h1>
<p>파일: ''' + a['info']['file'] + ''' | 분석: ''' + a['info']['time'] + '''</p>
</header>

<div class="ib"><strong>기간:</strong> ''' + a['info']['date_start'] + ''' ~ ''' + a['info']['date_end'] + ''' | <strong>데이터:</strong> ''' + "{:,}".format(a['info']['total_rows']) + '''행 x ''' + str(a['info']['total_cols']) + '''열</div>

<div class="sg">
<div class="sc"><div class="v">''' + "{:,}".format(a['info']['total_rows']) + '''</div><div class="l">전체 행</div></div>
<div class="sc"><div class="v">''' + str(a['info']['total_cols']) + '''</div><div class="l">전체 컬럼</div></div>
<div class="sc ''' + mpc + '''"><div class="v">''' + str(a['summary']['pct']) + '''%</div><div class="l">전체 결측률</div></div>
<div class="sc ''' + mcc + '''"><div class="v">''' + str(a['summary']['mcols']) + '''</div><div class="l">결측 컬럼</div></div>
<div class="sc ok"><div class="v">''' + str(a['summary']['ccols']) + '''</div><div class="l">완전 컬럼</div></div>
<div class="sc ''' + gc + '''"><div class="v">''' + str(a['gap_stats']['cnt']) + '''</div><div class="l">미수집 구간</div></div>
<div class="sc ''' + ndc + '''"><div class="v">''' + str(ndd) + '''</div><div class="l">미수집일</div></div>
</div>

<div class="sec">
<h2>결측 컬럼 현황 (클릭시 상세)</h2>
<div class="cg">''' + cards + '''</div>
</div>

<div class="chb">
<h3>날짜별 전체 컬럼 결측률 (%) - 막대 클릭시 상세</h3>
<div class="chc" style="height:320px"><canvas id="dc"></canvas></div>
<div class="lg"><span><i style="background:#22c55e"></i>0%</span><span><i style="background:#fbbf24"></i>0~5%</span><span><i style="background:#f59e0b"></i>5%+</span><span><i style="background:#ef4444"></i>미수집</span></div>
</div>

<div class="chb">
<h3>전체 컬럼 결측률 (%) - ''' + str(len(all_cols)) + '''개 컬럼 (막대 클릭시 상세)</h3>
<div class="all-chart-wrap">
<div class="chc" style="height:''' + str(chart_height) + '''px"><canvas id="ac"></canvas></div>
</div>
<div class="lg"><span><i style="background:#ef4444"></i>10%+</span><span><i style="background:#f59e0b"></i>1~10%</span><span><i style="background:#22c55e"></i>0~1%</span><span><i style="background:#3b82f6"></i>0%</span></div>
</div>

<div class="sec">
<h2>전체 컬럼 현황 (''' + str(len(a['columns'])) + '''개) <span style="font-size:.75rem;color:#a1a1aa">(결측 컬럼 클릭시 상세)</span></h2>
<div class="tw"><table><thead><tr><th>#</th><th>컬럼명</th><th>결측</th><th>결측률</th><th>상태</th></tr></thead><tbody>''' + cr + '''</tbody></table></div>
</div>

<div class="sec">
<h2>날짜별 현황 (''' + str(len(a['daily'])) + '''일) <span style="font-size:.75rem;color:#a1a1aa">(결측 날짜 클릭시 상세)</span></h2>
<div class="tw"><table><thead><tr><th>날짜</th><th>데이터</th><th>결측셀</th><th>결측률</th><th>결측컬럼</th><th>상태</th></tr></thead><tbody>''' + drw + '''</tbody></table></div>
</div>

''' + gs + '''

<div class="sec">
<h2>날짜/시간:분/컬럼별 결측 (날짜별 최대 10건) <span style="font-size:.75rem;color:#a1a1aa">전체: ''' + "{:,}".format(len(a['datetime_col_missing'])) + '''건 → CSV 확인</span></h2>
''' + self.generate_datetime_col_table(a['datetime_col_missing'], a['daily']) + '''
</div>

<div class="sec">
<h2>중복 데이터</h2>
<div class="sg" style="max-width:300px">
<div class="sc ''' + ("err" if a['dup']['full']>0 else "ok") + '''"><div class="v">''' + "{:,}".format(a['dup']['full']) + '''</div><div class="l">완전 중복</div></div>
<div class="sc ''' + ("warn" if a['dup']['time']>0 else "ok") + '''"><div class="v">''' + "{:,}".format(a['dup']['time']) + '''</div><div class="l">시간 중복</div></div>
</div>
</div>

<footer>AMHS 결측치 분석 | V8</footer>
</div>

<div id="modal" class="modal">
<div class="mc">
<div class="mh"><h3 id="mtitle">상세</h3><button class="mx" onclick="closeModal()">&times;</button></div>
<div class="ds" id="msum"></div>
<div class="tw" id="mtable"></div>
</div>
</div>

<script>
const colDaily = ''' + col_daily_json + ''';
const dateCol = ''' + date_col_json + ''';
const colData = ''' + col_data_json + ''';
const dailyData = ''' + daily_data_json + ''';
const dateList = ''' + json.dumps(dl) + ''';
const allColNames = ''' + json.dumps(all_col_names) + ''';

function showColDetail(col) {
    const c = colData[col];
    const d = colDaily[col] || [];
    document.getElementById('mtitle').textContent = col + ' 결측 상세';
    document.getElementById('msum').innerHTML = '<div><div class="dv">' + c.missing.toLocaleString() + '</div><div class="dl">총 결측</div></div><div><div class="dv">' + c.pct + '%</div><div class="dl">결측률</div></div><div><div class="dv">' + d.length + '</div><div class="dl">결측 발생일</div></div>';
    if (d.length === 0) {
        document.getElementById('mtable').innerHTML = '<p style="color:#a1a1aa;padding:16px">상세 없음</p>';
    } else {
        let h = '<table><thead><tr><th>날짜</th><th>결측</th><th>전체</th><th>결측률</th></tr></thead><tbody>';
        d.forEach(x => { const bc = x.pct > 50 ? 'be' : x.pct > 10 ? 'bw' : 'bo'; h += '<tr><td>' + x.date + '</td><td>' + x.missing.toLocaleString() + '</td><td>' + x.total.toLocaleString() + '</td><td><span class="' + bc + '">' + x.pct + '%</span></td></tr>'; });
        h += '</tbody></table>';
        document.getElementById('mtable').innerHTML = h;
    }
    document.getElementById('modal').style.display = 'block';
}

function showDateDetail(date) {
    const d = dailyData[date];
    const cols = dateCol[date] || [];
    document.getElementById('mtitle').textContent = date + ' 결측 상세';
    const noData = d.no_data;
    if (noData) {
        document.getElementById('msum').innerHTML = '<div><div class="dv" style="color:#ef4444">미수집</div><div class="dl">상태</div></div><div><div class="dv">0</div><div class="dl">데이터</div></div><div><div class="dv">' + d.mcols + '</div><div class="dl">결측 컬럼</div></div>';
        document.getElementById('mtable').innerHTML = '<p style="color:#ef4444;padding:16px">이 날짜는 데이터가 수집되지 않았습니다.</p>';
    } else {
        document.getElementById('msum').innerHTML = '<div><div class="dv">' + d.rows.toLocaleString() + '</div><div class="dl">데이터</div></div><div><div class="dv">' + d.pct + '%</div><div class="dl">결측률</div></div><div><div class="dv">' + d.mcols + '</div><div class="dl">결측 컬럼</div></div>';
        if (cols.length === 0) {
            document.getElementById('mtable').innerHTML = '<p style="color:#4ade80;padding:16px">결측 컬럼 없음 (완전)</p>';
        } else {
            let h = '<table><thead><tr><th>컬럼명</th><th>결측</th><th>전체</th><th>결측률</th></tr></thead><tbody>';
            cols.forEach(x => { const bc = x.pct > 50 ? 'be' : x.pct > 10 ? 'bw' : 'bo'; h += '<tr><td class="cn">' + x.column + '</td><td>' + x.missing.toLocaleString() + '</td><td>' + x.total.toLocaleString() + '</td><td><span class="' + bc + '">' + x.pct + '%</span></td></tr>'; });
            h += '</tbody></table>';
            document.getElementById('mtable').innerHTML = h;
        }
    }
    document.getElementById('modal').style.display = 'block';
}

function closeModal() { document.getElementById('modal').style.display = 'none'; }
window.onclick = function(e) { if (e.target === document.getElementById('modal')) closeModal(); }

Chart.defaults.color = '#a1a1aa';
Chart.defaults.borderColor = 'rgba(255,255,255,.1)';

// 날짜별 차트 (클릭 가능!)
new Chart(document.getElementById('dc'), {
    type: 'bar',
    data: { labels: ''' + json.dumps(dl) + ''', datasets: [{ label: '결측률(%)', data: ''' + json.dumps(dp) + ''', backgroundColor: ''' + json.dumps(dc) + ''', borderRadius: 3 }] },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false }, tooltip: { callbacks: { label: ctx => ['결측률: ' + ctx.raw + '%', '결측컬럼: ' + ''' + json.dumps(dmc) + '''[ctx.dataIndex] + '개'] } } },
        scales: { y: { beginAtZero: true, title: { display: true, text: '결측률(%)' } }, x: { ticks: { maxRotation: 45, minRotation: 45, autoSkip: true, maxTicksLimit: 35 } } },
        onClick: (evt, el) => { if (el.length > 0) showDateDetail(dateList[el[0].index]); }
    }
});

// 전체 컬럼 차트 (클릭 가능!)
new Chart(document.getElementById('ac'), {
    type: 'bar',
    data: { labels: ''' + json.dumps(all_labels) + ''', datasets: [{ label: '결측률(%)', data: ''' + json.dumps(all_data) + ''', backgroundColor: ''' + json.dumps(all_colors) + ''', borderRadius: 3 }] },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: 'y',
        plugins: { legend: { display: false } },
        scales: { x: { beginAtZero: true } },
        onClick: (evt, el) => { if (el.length > 0) { const col = allColNames[el[0].index]; if (colData[col].missing > 0) showColDetail(col); } }
    }
});
</script>
</body>
</html>'''
        
        with open(f'{self.output_dir}/missing_dashboard.html', 'w', encoding='utf-8') as f:
            f.write(html)
    
    def open_html(self):
        p = os.path.abspath(f'{self.output_dir}/missing_dashboard.html')
        if os.path.exists(p):
            webbrowser.open(f'file://{p}')
    
    def open_folder(self):
        p = os.path.abspath(self.output_dir)
        if os.path.exists(p):
            if os.name == 'nt':
                os.startfile(p)
            else:
                import subprocess
                subprocess.run(['open' if os.uname().sysname == 'Darwin' else 'xdg-open', p])


if __name__ == '__main__':
    root = tk.Tk()
    MissingAnalyzerUI(root)
    root.mainloop()