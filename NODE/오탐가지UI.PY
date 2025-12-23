import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import pandas as pd
from datetime import datetime
import webbrowser
import os

class SurgeAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("ML급증 감지 분석 시스템 V8")
        self.root.geometry("900x800")
        self.root.configure(bg='#1a1a2e')
        
        self.hub_file = None
        self.m14_file = None
        self.hub_df = None
        self.m14_df = None
        self.hub_stats = None
        self.m14_stats = None
        
        self.setup_ui()
    
    def setup_ui(self):
        main_frame = tk.Frame(self.root, bg='#1a1a2e')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)
        
        # 타이틀
        tk.Label(main_frame, text="📊 ML급증 감지 분석 시스템", font=('맑은 고딕', 24, 'bold'),
                 bg='#1a1a2e', fg='#00d4ff').pack(pady=(0, 5))
        tk.Label(main_frame, text="HUBROOM V8.0 | M14_Q V8.3.1", font=('맑은 고딕', 12),
                 bg='#1a1a2e', fg='#888888').pack(pady=(0, 15))
        
        # 분석 날짜 입력
        date_frame = tk.Frame(main_frame, bg='#252540', relief='ridge', bd=2)
        date_frame.pack(fill=tk.X, pady=8, ipady=8, ipadx=15)
        
        tk.Label(date_frame, text="📅 분석 날짜", font=('맑은 고딕', 12, 'bold'),
                 bg='#252540', fg='#00ff88').pack(side=tk.LEFT, padx=(10,15), pady=8)
        
        today = datetime.now()
        
        self.year_var = tk.StringVar(value=str(today.year))
        self.month_var = tk.StringVar(value=str(today.month))
        self.day_var = tk.StringVar(value=str(today.day))
        
        tk.Entry(date_frame, textvariable=self.year_var, font=('맑은 고딕', 11), width=6,
                 bg='#0d0d15', fg='white', insertbackground='white', justify='center').pack(side=tk.LEFT)
        tk.Label(date_frame, text="년", font=('맑은 고딕', 11), bg='#252540', fg='white').pack(side=tk.LEFT, padx=(3,10))
        
        tk.Entry(date_frame, textvariable=self.month_var, font=('맑은 고딕', 11), width=4,
                 bg='#0d0d15', fg='white', insertbackground='white', justify='center').pack(side=tk.LEFT)
        tk.Label(date_frame, text="월", font=('맑은 고딕', 11), bg='#252540', fg='white').pack(side=tk.LEFT, padx=(3,10))
        
        tk.Entry(date_frame, textvariable=self.day_var, font=('맑은 고딕', 11), width=4,
                 bg='#0d0d15', fg='white', insertbackground='white', justify='center').pack(side=tk.LEFT)
        tk.Label(date_frame, text="일", font=('맑은 고딕', 11), bg='#252540', fg='white').pack(side=tk.LEFT, padx=(3,10))
        
        # HUBROOM 프레임
        hub_frame = tk.Frame(main_frame, bg='#252540', relief='ridge', bd=2)
        hub_frame.pack(fill=tk.X, pady=8, ipady=10, ipadx=15)
        
        tk.Label(hub_frame, text="🏠 HUBROOM V8.0", font=('맑은 고딕', 13, 'bold'), 
                 bg='#252540', fg='#00d4ff').pack(anchor='w', padx=10, pady=(8,3))
        tk.Label(hub_frame, text="기준값: 350 | 컬럼: 현재값, 실제값, 예측값_MAX", 
                 font=('맑은 고딕', 9), bg='#252540', fg='#888888').pack(anchor='w', padx=10)
        
        hub_btn_frame = tk.Frame(hub_frame, bg='#252540')
        hub_btn_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Button(hub_btn_frame, text="📁 파일 선택", command=self.select_hub_file,
                  font=('맑은 고딕', 10), bg='#00d4ff', fg='black', width=12).pack(side=tk.LEFT)
        self.hub_label = tk.Label(hub_btn_frame, text="파일 선택 안됨", 
                                   font=('맑은 고딕', 9), bg='#252540', fg='#888888')
        self.hub_label.pack(side=tk.LEFT, padx=10)
        
        tk.Label(hub_frame, text="📝 코멘트:", font=('맑은 고딕', 9), bg='#252540', fg='#00d4ff').pack(anchor='w', padx=10, pady=(5,0))
        self.hub_comment = tk.Text(hub_frame, font=('맑은 고딕', 9), bg='#0d0d15', fg='white', 
                                    height=2, relief='flat', padx=8, pady=5, insertbackground='white')
        self.hub_comment.pack(fill=tk.X, padx=10, pady=(0,8))
        
        # M14_Q 프레임
        m14_frame = tk.Frame(main_frame, bg='#252540', relief='ridge', bd=2)
        m14_frame.pack(fill=tk.X, pady=8, ipady=10, ipadx=15)
        
        tk.Label(m14_frame, text="🏭 M14_Q V8.3.1", font=('맑은 고딕', 13, 'bold'), 
                 bg='#252540', fg='#ff6b35').pack(anchor='w', padx=10, pady=(8,3))
        tk.Label(m14_frame, text="기준값: 1700 | 컬럼: 현재TOTALCNT, 실제값, 최종예측", 
                 font=('맑은 고딕', 9), bg='#252540', fg='#888888').pack(anchor='w', padx=10)
        
        m14_btn_frame = tk.Frame(m14_frame, bg='#252540')
        m14_btn_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Button(m14_btn_frame, text="📁 파일 선택", command=self.select_m14_file,
                  font=('맑은 고딕', 10), bg='#ff6b35', fg='black', width=12).pack(side=tk.LEFT)
        self.m14_label = tk.Label(m14_btn_frame, text="파일 선택 안됨", 
                                   font=('맑은 고딕', 9), bg='#252540', fg='#888888')
        self.m14_label.pack(side=tk.LEFT, padx=10)
        
        tk.Label(m14_frame, text="📝 코멘트:", font=('맑은 고딕', 9), bg='#252540', fg='#ff6b35').pack(anchor='w', padx=10, pady=(5,0))
        self.m14_comment = tk.Text(m14_frame, font=('맑은 고딕', 9), bg='#0d0d15', fg='white', 
                                    height=2, relief='flat', padx=8, pady=5, insertbackground='white')
        self.m14_comment.pack(fill=tk.X, padx=10, pady=(0,8))
        
        # 분석 버튼
        tk.Button(main_frame, text="🔍 분석 시작 및 저장", command=self.run_analysis,
                  font=('맑은 고딕', 14, 'bold'), bg='#00ff88', fg='black',
                  width=25, height=2).pack(pady=20)
        
        # 결과 영역
        tk.Label(main_frame, text="📋 분석 결과", font=('맑은 고딕', 11, 'bold'),
                 bg='#1a1a2e', fg='white').pack(anchor='w')
        self.result_text = tk.Text(main_frame, font=('Consolas', 10), bg='#0d0d15', fg='white', 
                                    height=12, relief='flat', padx=15, pady=15)
        self.result_text.pack(fill=tk.BOTH, expand=True, pady=(5,0))
        self.result_text.insert(tk.END, "CSV 파일을 선택하고 분석을 시작하세요...")
        self.result_text.config(state=tk.DISABLED)
    
    def select_hub_file(self):
        file = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file:
            self.hub_file = file
            self.hub_label.config(text=f"✅ {os.path.basename(file)}", fg='#00ff88')
    
    def select_m14_file(self):
        file = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file:
            self.m14_file = file
            self.m14_label.config(text=f"✅ {os.path.basename(file)}", fg='#00ff88')
    
    def analyze_hubroom(self, df, threshold=350):
        for col in ['Unnamed: 22', 'Unnamed: 23']:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        df['사전감지OK_NG'] = ''
        surge_cond = (df['실제값'] >= threshold) & (df['현재값'] <= threshold)
        df.loc[surge_cond, '사전감지OK_NG'] = 'Y'
        
        df['사전감지결과'] = ''
        df.loc[surge_cond & (df['예측값_MAX'] >= threshold), '사전감지결과'] = 'OK'
        
        ng_idx = df[surge_cond & (df['예측값_MAX'] < threshold)].index.tolist()
        for idx in ng_idx:
            found = False
            for back in range(1, 6):
                if idx - back >= 0 and df.loc[idx - back, '예측값_MAX'] >= threshold:
                    df.loc[idx, '사전감지결과'] = f'사전감지({back}분전)'
                    found = True
                    break
            if not found:
                df.loc[idx, '사전감지결과'] = 'NG'
        
        df['오탐/사전감지'] = ''
        fp_cond = (df['실제값'] < threshold) & (df['예측값_MAX'] >= threshold)
        for idx in df[fp_cond].index.tolist():
            found = False
            for off in range(-5, 6):
                if off != 0 and 0 <= idx + off < len(df) and df.loc[idx + off, '실제값'] >= threshold:
                    df.loc[idx, '오탐/사전감지'] = '오탐사전감지'
                    found = True
                    break
            if not found:
                df.loc[idx, '오탐/사전감지'] = '오탐'
        
        cols = df.columns.tolist()
        for c in ['사전감지OK_NG', '사전감지결과', '오탐/사전감지']:
            cols.remove(c)
        idx = cols.index('실제값')
        df = df[cols[:idx+1] + ['사전감지OK_NG', '사전감지결과', '오탐/사전감지'] + cols[idx+1:]]
        
        normal_correct = len(df[(df['실제값'] < threshold) & (df['예측값_MAX'] < threshold)])
        
        return df, {
            'total_rows': len(df),
            'total': len(df[surge_cond]), 
            'normal': len(df) - len(df[surge_cond]),
            'normal_correct': normal_correct,
            'ok': len(df[df['사전감지결과'] == 'OK']),
            'pre': len(df[df['사전감지결과'].str.contains('사전감지', na=False)]),
            'ng': len(df[df['사전감지결과'] == 'NG']),
            'fp_total': len(df[fp_cond]),
            'fp_pre': len(df[df['오탐/사전감지'] == '오탐사전감지']),
            'fp_real': len(df[df['오탐/사전감지'] == '오탐'])
        }
    
    def analyze_m14(self, df, threshold=1700):
        for col in ['12분전TOTALCNT', '12변화량', 'Unnamed: 22', 'Unnamed: 23']:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        df['사전감지OK_NG'] = ''
        surge_cond = (df['실제값'] >= threshold) & (df['현재TOTALCNT'] <= threshold)
        df.loc[surge_cond, '사전감지OK_NG'] = 'Y'
        
        df['사전감지결과'] = ''
        df.loc[surge_cond & (df['최종예측'] >= threshold), '사전감지결과'] = 'OK'
        
        ng_idx = df[surge_cond & (df['최종예측'] < threshold)].index.tolist()
        for idx in ng_idx:
            found = False
            for back in range(1, 6):
                if idx - back >= 0 and df.loc[idx - back, '최종예측'] >= threshold:
                    df.loc[idx, '사전감지결과'] = f'사전감지({back}분전)'
                    found = True
                    break
            if not found:
                df.loc[idx, '사전감지결과'] = 'NG'
        
        df['오탐/사전감지'] = ''
        fp_cond = (df['실제값'] < threshold) & (df['최종예측'] >= threshold)
        for idx in df[fp_cond].index.tolist():
            found = False
            for off in range(-5, 6):
                if off != 0 and 0 <= idx + off < len(df) and df.loc[idx + off, '실제값'] >= threshold:
                    df.loc[idx, '오탐/사전감지'] = '오탐사전감지'
                    found = True
                    break
            if not found:
                df.loc[idx, '오탐/사전감지'] = '오탐'
        
        cols = df.columns.tolist()
        for c in ['사전감지OK_NG', '사전감지결과', '오탐/사전감지']:
            cols.remove(c)
        idx = cols.index('실제값')
        df = df[cols[:idx+1] + ['사전감지OK_NG', '사전감지결과', '오탐/사전감지'] + cols[idx+1:]]
        
        normal_correct = len(df[(df['실제값'] < threshold) & (df['최종예측'] < threshold)])
        
        return df, {
            'total_rows': len(df),
            'total': len(df[surge_cond]), 
            'normal': len(df) - len(df[surge_cond]),
            'normal_correct': normal_correct,
            'ok': len(df[df['사전감지결과'] == 'OK']),
            'pre': len(df[df['사전감지결과'].str.contains('사전감지', na=False)]),
            'ng': len(df[df['사전감지결과'] == 'NG']),
            'fp_total': len(df[fp_cond]),
            'fp_pre': len(df[df['오탐/사전감지'] == '오탐사전감지']),
            'fp_real': len(df[df['오탐/사전감지'] == '오탐'])
        }
    
    def run_analysis(self):
        if not self.hub_file and not self.m14_file:
            messagebox.showwarning("경고", "CSV 파일을 선택하세요!")
            return
        
        hub_cmt = self.hub_comment.get("1.0", tk.END).strip()
        m14_cmt = self.m14_comment.get("1.0", tk.END).strip()
        
        # 분석 날짜 가져오기
        try:
            anal_date = f"{self.year_var.get()}년 {self.month_var.get()}월 {self.day_var.get()}일"
        except:
            anal_date = datetime.now().strftime("%Y년 %m월 %d일")
        
        result_text = ""
        
        try:
            if self.hub_file:
                df = pd.read_csv(self.hub_file, encoding='utf-8-sig')
                self.hub_df, self.hub_stats = self.analyze_hubroom(df.copy(), 350)
                result_text += self.fmt("HUBROOM V8.0", self.hub_stats, 350)
            
            if self.m14_file:
                df = pd.read_csv(self.m14_file, encoding='utf-8-sig')
                self.m14_df, self.m14_stats = self.analyze_m14(df.copy(), 1700)
                result_text += self.fmt("M14_Q V8.3.1", self.m14_stats, 1700)
            
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, result_text)
            self.result_text.config(state=tk.DISABLED)
            
            save_dir = filedialog.askdirectory(title="결과 저장할 폴더 선택")
            if save_dir:
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                files = []
                
                if self.hub_df is not None:
                    f = os.path.join(save_dir, f"HUBROOM_결과_{ts}.csv")
                    self.hub_df.to_csv(f, index=False, encoding='utf-8-sig')
                    files.append(f)
                
                if self.m14_df is not None:
                    f = os.path.join(save_dir, f"M14_결과_{ts}.csv")
                    self.m14_df.to_csv(f, index=False, encoding='utf-8-sig')
                    files.append(f)
                
                html_f = os.path.join(save_dir, f"ML급증감지_리포트_{ts}.html")
                with open(html_f, 'w', encoding='utf-8') as f:
                    f.write(self.gen_html(hub_cmt, m14_cmt, anal_date))
                files.append(html_f)
                
                webbrowser.open(html_f)
                messagebox.showinfo("완료", "저장 완료!\n\n" + "\n".join([os.path.basename(x) for x in files]))
            
        except Exception as e:
            messagebox.showerror("오류", str(e))
    
    def fmt(self, name, s, th):
        t = s['total'] or 1
        f = s['fp_total'] or 1
        n = s['normal'] or 1
        return f"""
{'='*50}
  {name} (기준: {th})
{'='*50}
전체: {s['total_rows']}개 | 정상: {s['normal']}개 | 급증: {s['total']}개
정상예측: {s['normal_correct']}개({s['normal_correct']/n*100:.1f}%)
급증감지 - OK: {s['ok']}({s['ok']/t*100:.1f}%) | 사전감지: {s['pre']}({s['pre']/t*100:.1f}%) | NG: {s['ng']}({s['ng']/t*100:.1f}%)
감지율: {(s['ok']+s['pre'])/t*100:.1f}% | 오탐: {s['fp_real']}/{s['fp_total']}({s['fp_real']/f*100:.1f}%)
"""
    
    def gen_html(self, hub_cmt, m14_cmt, anal_date):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        def r(s):
            t = s['total'] or 1
            f = s['fp_total'] or 1
            n = s['normal'] or 1
            return {
                'ok': s['ok']/t*100, 'pre': s['pre']/t*100, 'ng': s['ng']/t*100,
                'det': (s['ok']+s['pre'])/t*100, 'fp_pre': s['fp_pre']/f*100, 'fp': s['fp_real']/f*100,
                'norm': s['normal_correct']/n*100
            }
        
        html = f'''<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>ML급증 감지 분석 리포트</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'맑은 고딕',sans-serif;background:#0a0a0f;color:#fff;padding:40px}}
.container{{max-width:1200px;margin:0 auto}}
h1{{font-size:2.5rem;text-align:center;background:linear-gradient(135deg,#00d4ff,#ff6b35);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:10px}}
.sub{{text-align:center;color:#888;margin-bottom:30px}}
.section{{margin-bottom:50px}}
.stitle{{font-size:1.5rem;margin-bottom:20px;padding-bottom:10px;border-bottom:2px solid #333}}
.stitle.hub{{color:#00d4ff;border-color:#00d4ff}}
.stitle.m14{{color:#ff6b35;border-color:#ff6b35}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:15px;margin-bottom:25px}}
.card{{background:#12121a;padding:18px;border-radius:12px;text-align:center;border:1px solid #2a2a3a}}
.card-label{{font-size:0.8rem;color:#888;margin-bottom:8px}}
.card-value{{font-size:1.8rem;font-weight:bold}}
.card-sub{{font-size:0.85rem;color:#888;margin-top:5px}}
.ok{{color:#00ff88}}.warn{{color:#ffcc00}}.bad{{color:#ff4466}}.hub-c{{color:#00d4ff}}.m14-c{{color:#ff6b35}}
.charts{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:25px}}
.cbox{{background:#12121a;padding:20px;border-radius:12px;border:1px solid #2a2a3a}}
.ctitle{{font-size:1rem;margin-bottom:15px}}
.cwrap{{height:260px}}
.summary{{background:#12121a;padding:20px;border-radius:12px;border:1px solid #2a2a3a;margin-bottom:15px}}
.summary pre{{font-family:Consolas,monospace;white-space:pre-wrap;color:#aaa;line-height:1.6}}
.comment{{background:#1a2a1a;border:2px solid #00ff88;border-radius:12px;padding:18px;margin-top:15px}}
.comment.m14{{background:#2a1a1a;border-color:#ff6b35}}
.comment-title{{font-weight:bold;margin-bottom:8px}}
.comment-title.hub{{color:#00d4ff}}
.comment-title.m14{{color:#ff6b35}}
.comment-content{{color:#ccc;line-height:1.7}}
.footer{{text-align:center;color:#555;margin-top:50px;font-size:0.85rem}}
@media(max-width:900px){{.grid{{grid-template-columns:repeat(2,1fr)}}.charts{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<div class="container">
<h1>📊 ML급증 감지 분석 리포트</h1>
<p class="sub">📅 분석 날짜: {anal_date} | 생성: {now}</p>
'''
        
        if self.hub_stats and self.hub_stats['total_rows'] > 0:
            s = self.hub_stats
            rt = r(s)
            cmt_html = ""
            if hub_cmt:
                cmt_html = f'<div class="comment"><div class="comment-title hub">📝 HUBROOM 분석 코멘트</div><div class="comment-content">{hub_cmt.replace(chr(10),"<br>")}</div></div>'
            
            html += f'''
<div class="section">
<h2 class="stitle hub">🏠 HUBROOM V8.0</h2>
<div class="grid">
<div class="card"><div class="card-label">전체 데이터</div><div class="card-value">{s['total_rows']}</div></div>
<div class="card"><div class="card-label">정상</div><div class="card-value ok">{s['normal']}</div><div class="card-sub">정상예측 {rt['norm']:.1f}%</div></div>
<div class="card"><div class="card-label">급증 케이스</div><div class="card-value hub-c">{s['total']}</div></div>
<div class="card"><div class="card-label">총 감지율</div><div class="card-value ok">{rt['det']:.1f}%</div></div>
<div class="card"><div class="card-label">OK</div><div class="card-value ok">{s['ok']}</div><div class="card-sub">{rt['ok']:.1f}%</div></div>
<div class="card"><div class="card-label">사전감지</div><div class="card-value warn">{s['pre']}</div><div class="card-sub">{rt['pre']:.1f}%</div></div>
<div class="card"><div class="card-label">NG</div><div class="card-value bad">{s['ng']}</div><div class="card-sub">{rt['ng']:.1f}%</div></div>
<div class="card"><div class="card-label">오탐</div><div class="card-value bad">{s['fp_real']}</div><div class="card-sub">{rt['fp']:.1f}%</div></div>
</div>
<div class="charts">
<div class="cbox"><div class="ctitle">📈 감지 분포</div><div class="cwrap"><canvas id="hubPie"></canvas></div></div>
<div class="cbox"><div class="ctitle">📉 오탐 분석</div><div class="cwrap"><canvas id="hubFp"></canvas></div></div>
</div>
<div class="summary"><pre>=== HUBROOM V8.0 요약 (기준: 350) ===
전체: {s['total_rows']}개 | 정상: {s['normal']}개 (정상예측 {s['normal_correct']}개, {rt['norm']:.1f}%) | 급증: {s['total']}개
급증감지 - OK: {s['ok']}개({rt['ok']:.1f}%) | 사전감지: {s['pre']}개({rt['pre']:.1f}%) | NG: {s['ng']}개({rt['ng']:.1f}%)
총 감지율: {rt['det']:.1f}% | 오탐: {s['fp_real']}/{s['fp_total']}개({rt['fp']:.1f}%)</pre></div>
{cmt_html}
</div>
'''
        
        if self.m14_stats and self.m14_stats['total_rows'] > 0:
            s = self.m14_stats
            rt = r(s)
            cmt_html = ""
            if m14_cmt:
                cmt_html = f'<div class="comment m14"><div class="comment-title m14">📝 M14_Q 분석 코멘트</div><div class="comment-content">{m14_cmt.replace(chr(10),"<br>")}</div></div>'
            
            html += f'''
<div class="section">
<h2 class="stitle m14">🏭 M14_Q V8.3.1</h2>
<div class="grid">
<div class="card"><div class="card-label">전체 데이터</div><div class="card-value">{s['total_rows']}</div></div>
<div class="card"><div class="card-label">정상</div><div class="card-value ok">{s['normal']}</div><div class="card-sub">정상예측 {rt['norm']:.1f}%</div></div>
<div class="card"><div class="card-label">급증 케이스</div><div class="card-value m14-c">{s['total']}</div></div>
<div class="card"><div class="card-label">총 감지율</div><div class="card-value ok">{rt['det']:.1f}%</div></div>
<div class="card"><div class="card-label">OK</div><div class="card-value ok">{s['ok']}</div><div class="card-sub">{rt['ok']:.1f}%</div></div>
<div class="card"><div class="card-label">사전감지</div><div class="card-value warn">{s['pre']}</div><div class="card-sub">{rt['pre']:.1f}%</div></div>
<div class="card"><div class="card-label">NG</div><div class="card-value bad">{s['ng']}</div><div class="card-sub">{rt['ng']:.1f}%</div></div>
<div class="card"><div class="card-label">오탐</div><div class="card-value bad">{s['fp_real']}</div><div class="card-sub">{rt['fp']:.1f}%</div></div>
</div>
<div class="charts">
<div class="cbox"><div class="ctitle">📈 감지 분포</div><div class="cwrap"><canvas id="m14Pie"></canvas></div></div>
<div class="cbox"><div class="ctitle">📉 오탐 분석</div><div class="cwrap"><canvas id="m14Fp"></canvas></div></div>
</div>
<div class="summary"><pre>=== M14_Q V8.3.1 요약 (기준: 1700) ===
전체: {s['total_rows']}개 | 정상: {s['normal']}개 (정상예측 {s['normal_correct']}개, {rt['norm']:.1f}%) | 급증: {s['total']}개
급증감지 - OK: {s['ok']}개({rt['ok']:.1f}%) | 사전감지: {s['pre']}개({rt['pre']:.1f}%) | NG: {s['ng']}개({rt['ng']:.1f}%)
총 감지율: {rt['det']:.1f}% | 오탐: {s['fp_real']}/{s['fp_total']}개({rt['fp']:.1f}%)</pre></div>
{cmt_html}
</div>
'''
        
        html += '''
<div class="footer">ML급증 감지 분석 시스템 V8 | AMHS 물류 예측 모델</div>
</div>
<script>
Chart.register(ChartDataLabels);
const opt={responsive:true,maintainAspectRatio:false,plugins:{legend:{position:'bottom',labels:{color:'#fff',font:{size:11}}},datalabels:{color:'#fff',font:{weight:'bold',size:13},formatter:(v,ctx)=>{let sum=ctx.chart.data.datasets[0].data.reduce((a,b)=>a+b,0);return v>0?(v/sum*100).toFixed(1)+'%':''}}}};
'''
        
        if self.hub_stats and self.hub_stats['total_rows'] > 0:
            s = self.hub_stats
            html += f'''
new Chart(document.getElementById('hubPie'),{{type:'doughnut',data:{{labels:['OK ({s['ok']}개)','사전감지 ({s['pre']}개)','NG ({s['ng']}개)'],datasets:[{{data:[{s['ok']},{s['pre']},{s['ng']}],backgroundColor:['#00ff88','#ffcc00','#ff4466']}}]}},options:opt}});
new Chart(document.getElementById('hubFp'),{{type:'doughnut',data:{{labels:['오탐사전감지 ({s['fp_pre']}개)','오탐 ({s['fp_real']}개)'],datasets:[{{data:[{s['fp_pre']},{s['fp_real']}],backgroundColor:['#ffcc00','#ff4466']}}]}},options:opt}});
'''
        
        if self.m14_stats and self.m14_stats['total_rows'] > 0:
            s = self.m14_stats
            html += f'''
new Chart(document.getElementById('m14Pie'),{{type:'doughnut',data:{{labels:['OK ({s['ok']}개)','사전감지 ({s['pre']}개)','NG ({s['ng']}개)'],datasets:[{{data:[{s['ok']},{s['pre']},{s['ng']}],backgroundColor:['#00ff88','#ffcc00','#ff4466']}}]}},options:opt}});
new Chart(document.getElementById('m14Fp'),{{type:'doughnut',data:{{labels:['오탐사전감지 ({s['fp_pre']}개)','오탐 ({s['fp_real']}개)'],datasets:[{{data:[{s['fp_pre']},{s['fp_real']}],backgroundColor:['#ffcc00','#ff4466']}}]}},options:opt}});
'''
        
        html += '</script></body></html>'
        return html


if __name__ == "__main__":
    root = tk.Tk()
    app = SurgeAnalyzer(root)
    root.mainloop()