import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
from datetime import datetime
import webbrowser
import os

class SurgeAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("ML급증 감지 분석 시스템 V9")
        self.root.geometry("1200x900")
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
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)
        
        top_frame = tk.Frame(main_frame, bg='#1a1a2e')
        top_frame.pack(fill=tk.X)
        
        # 타이틀
        tk.Label(top_frame, text="📊 ML급증 감지 분석 시스템 V9", font=('맑은 고딕', 20, 'bold'),
                 bg='#1a1a2e', fg='#00d4ff').pack(pady=(0, 3))
        tk.Label(top_frame, text="이상탐지 / 급증사전감지 / 정상예측 / 미탐 / 정상사전감지 / 오탐 / 정상", font=('맑은 고딕', 10),
                 bg='#1a1a2e', fg='#888888').pack(pady=(0, 10))
        
        # 분석 날짜
        date_frame = tk.Frame(top_frame, bg='#252540', relief='ridge', bd=2)
        date_frame.pack(fill=tk.X, pady=5, ipady=5, ipadx=10)
        
        tk.Label(date_frame, text="📅 분석 날짜", font=('맑은 고딕', 11, 'bold'),
                 bg='#252540', fg='#00ff88').pack(side=tk.LEFT, padx=(10,15), pady=5)
        
        today = datetime.now()
        self.year_var = tk.StringVar(value=str(today.year))
        self.month_var = tk.StringVar(value=str(today.month))
        self.day_var = tk.StringVar(value=str(today.day))
        
        tk.Entry(date_frame, textvariable=self.year_var, font=('맑은 고딕', 10), width=5,
                 bg='#0d0d15', fg='white', insertbackground='white', justify='center').pack(side=tk.LEFT)
        tk.Label(date_frame, text="년", font=('맑은 고딕', 10), bg='#252540', fg='white').pack(side=tk.LEFT, padx=(2,8))
        tk.Entry(date_frame, textvariable=self.month_var, font=('맑은 고딕', 10), width=3,
                 bg='#0d0d15', fg='white', insertbackground='white', justify='center').pack(side=tk.LEFT)
        tk.Label(date_frame, text="월", font=('맑은 고딕', 10), bg='#252540', fg='white').pack(side=tk.LEFT, padx=(2,8))
        tk.Entry(date_frame, textvariable=self.day_var, font=('맑은 고딕', 10), width=3,
                 bg='#0d0d15', fg='white', insertbackground='white', justify='center').pack(side=tk.LEFT)
        tk.Label(date_frame, text="일", font=('맑은 고딕', 10), bg='#252540', fg='white').pack(side=tk.LEFT, padx=(2,8))
        
        # 파일 선택
        file_frame = tk.Frame(top_frame, bg='#1a1a2e')
        file_frame.pack(fill=tk.X, pady=5)
        
        # HUBROOM
        hub_frame = tk.Frame(file_frame, bg='#252540', relief='ridge', bd=2)
        hub_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,5), ipady=5, ipadx=10)
        
        tk.Label(hub_frame, text="🏠 HUBROOM (350)", font=('맑은 고딕', 11, 'bold'), 
                 bg='#252540', fg='#00d4ff').pack(anchor='w', padx=10, pady=(5,2))
        
        hub_btn_frame = tk.Frame(hub_frame, bg='#252540')
        hub_btn_frame.pack(fill=tk.X, padx=10, pady=3)
        tk.Button(hub_btn_frame, text="📁 파일", command=self.select_hub_file,
                  font=('맑은 고딕', 9), bg='#00d4ff', fg='black', width=8).pack(side=tk.LEFT)
        self.hub_label = tk.Label(hub_btn_frame, text="선택 안됨", font=('맑은 고딕', 8), bg='#252540', fg='#888')
        self.hub_label.pack(side=tk.LEFT, padx=5)
        
        self.hub_comment = tk.Text(hub_frame, font=('맑은 고딕', 9), bg='#0d0d15', fg='white', 
                                    height=2, relief='flat', padx=5, pady=3, insertbackground='white')
        self.hub_comment.pack(fill=tk.X, padx=10, pady=(0,5))
        
        # M14_Q
        m14_frame = tk.Frame(file_frame, bg='#252540', relief='ridge', bd=2)
        m14_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5,0), ipady=5, ipadx=10)
        
        tk.Label(m14_frame, text="🏭 M14_Q (1700)", font=('맑은 고딕', 11, 'bold'), 
                 bg='#252540', fg='#ff6b35').pack(anchor='w', padx=10, pady=(5,2))
        
        m14_btn_frame = tk.Frame(m14_frame, bg='#252540')
        m14_btn_frame.pack(fill=tk.X, padx=10, pady=3)
        tk.Button(m14_btn_frame, text="📁 파일", command=self.select_m14_file,
                  font=('맑은 고딕', 9), bg='#ff6b35', fg='black', width=8).pack(side=tk.LEFT)
        self.m14_label = tk.Label(m14_btn_frame, text="선택 안됨", font=('맑은 고딕', 8), bg='#252540', fg='#888')
        self.m14_label.pack(side=tk.LEFT, padx=5)
        
        self.m14_comment = tk.Text(m14_frame, font=('맑은 고딕', 9), bg='#0d0d15', fg='white', 
                                    height=2, relief='flat', padx=5, pady=3, insertbackground='white')
        self.m14_comment.pack(fill=tk.X, padx=10, pady=(0,5))
        
        # 분석 버튼
        tk.Button(top_frame, text="🔍 분석 시작 및 저장", command=self.run_analysis,
                  font=('맑은 고딕', 12, 'bold'), bg='#00ff88', fg='black',
                  width=20, height=1).pack(pady=10)
        
        # 탭
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(5,0))
        
        style = ttk.Style()
        style.configure('TNotebook', background='#1a1a2e')
        style.configure('TNotebook.Tab', font=('맑은 고딕', 10, 'bold'), padding=[10, 5])
        
        result_frame = tk.Frame(self.notebook, bg='#0d0d15')
        self.notebook.add(result_frame, text='📋 분석 결과')
        
        self.result_text = tk.Text(result_frame, font=('Consolas', 10), bg='#0d0d15', fg='white', 
                                    relief='flat', padx=15, pady=15)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        self.result_text.insert(tk.END, "CSV 파일을 선택하고 분석을 시작하세요...")
        
        self.hub_data_frame = tk.Frame(self.notebook, bg='#0d0d15')
        self.notebook.add(self.hub_data_frame, text='🏠 HUBROOM 데이터')
        
        self.m14_data_frame = tk.Frame(self.notebook, bg='#0d0d15')
        self.notebook.add(self.m14_data_frame, text='🏭 M14_Q 데이터')
    
    def create_data_view(self, parent, df):
        for widget in parent.winfo_children():
            widget.destroy()
        
        if df is None or len(df) == 0:
            tk.Label(parent, text="데이터 없음", font=('맑은 고딕', 12), bg='#0d0d15', fg='#888').pack(pady=50)
            return
        
        tree_frame = tk.Frame(parent, bg='#0d0d15')
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        y_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL)
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        x_scroll = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL)
        x_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        
        cols = list(df.columns)
        tree = ttk.Treeview(tree_frame, columns=cols, show='headings',
                            yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        
        y_scroll.config(command=tree.yview)
        x_scroll.config(command=tree.xview)
        
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=100, minwidth=50)
        
        for idx, row in df.iterrows():
            values = [str(v) if pd.notna(v) else '' for v in row.values]
            tag = ''
            if '분류' in df.columns:
                분류 = str(row.get('분류', ''))
                if 분류 == '이상탐지':
                    tag = 'detect'
                elif '급증사전감지' in 분류:
                    tag = 'pre'
                elif '정상예측' in 분류:
                    tag = 'normal_pred'
                elif 분류 == '미탐':
                    tag = 'miss'
                elif '정상사전감지' in 분류:
                    tag = 'pre_future'
                elif '오탐' in 분류:
                    tag = 'false'
            tree.insert('', tk.END, values=values, tags=(tag,))
        
        tree.tag_configure('detect', background='#1a3a1a')
        tree.tag_configure('pre', background='#2a3a1a')
        tree.tag_configure('normal_pred', background='#1a2a3a')
        tree.tag_configure('miss', background='#3a1a1a')
        tree.tag_configure('pre_future', background='#2a2a3a')
        tree.tag_configure('false', background='#3a1a2a')
        
        tree.pack(fill=tk.BOTH, expand=True)
    
    def select_hub_file(self):
        file = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file:
            self.hub_file = file
            self.hub_label.config(text=f"✅ {os.path.basename(file)[:25]}", fg='#00ff88')
    
    def select_m14_file(self):
        file = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file:
            self.m14_file = file
            self.m14_label.config(text=f"✅ {os.path.basename(file)[:25]}", fg='#00ff88')
    
    def analyze_v9(self, df, threshold, pred_col, curr_col, actual_col):
        """
        V9 분석 로직
        급증(실제>=threshold):
          - 현재 예측>=threshold → 이상탐지
          - 과거 10분내 예측>=threshold → 급증사전감지
          - 미래 10분내 예측>=threshold → 정상예측
          - 다 아니면 → 미탐
        정상(실제<threshold):
          - 예측>=threshold + 10분내 급증 → 정상사전감지
          - 예측>=threshold + 10분내 급증 없음 → 오탐
          - 예측<threshold → 정상
        """
        df = df.copy()
        
        # 불필요 컬럼 제거
        drop_cols = ['Unnamed: 22', 'Unnamed: 23', '12분전TOTALCNT', '12변화량']
        for col in drop_cols:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        df['분류'] = ''
        
        for idx in range(len(df)):
            actual = df.loc[idx, actual_col]
            pred = df.loc[idx, pred_col]
            
            if actual >= threshold:
                # 급증 케이스
                if pred >= threshold:
                    df.loc[idx, '분류'] = '이상탐지'
                else:
                    # 1) 과거 10분내 예측 >= threshold
                    found_past = False
                    for back in range(1, 11):
                        if idx - back >= 0 and df.loc[idx - back, pred_col] >= threshold:
                            df.loc[idx, '분류'] = f'급증사전감지({back}분전)'
                            found_past = True
                            break
                    
                    if not found_past:
                        # 2) 미래 10분내 예측 >= threshold
                        found_future = False
                        for fwd in range(1, 11):
                            if idx + fwd < len(df) and df.loc[idx + fwd, pred_col] >= threshold:
                                df.loc[idx, '분류'] = f'정상예측({fwd}분후)'
                                found_future = True
                                break
                        
                        if not found_future:
                            df.loc[idx, '분류'] = '미탐'
            else:
                # 정상 케이스
                if pred >= threshold:
                    # 10분내 급증 있는지 체크
                    is_pre = False
                    for fwd in range(1, 11):
                        if idx + fwd < len(df) and df.loc[idx + fwd, actual_col] >= threshold:
                            is_pre = True
                            df.loc[idx, '분류'] = f'정상사전감지({fwd}분후)'
                            break
                    if not is_pre:
                        df.loc[idx, '분류'] = '오탐'
                else:
                    df.loc[idx, '분류'] = '정상'
        
        # 통계
        total = len(df)
        surge_total = len(df[df[actual_col] >= threshold])
        detect = len(df[df['분류'] == '이상탐지'])
        pre_past = len(df[df['분류'].str.contains('급증사전감지', na=False)])  # 급증케이스 사전감지
        pre_future = len(df[df['분류'].str.contains('정상사전감지', na=False)])  # 정상케이스 사전감지
        normal_pred = len(df[df['분류'].str.contains('정상예측', na=False)])
        miss = len(df[df['분류'] == '미탐'])
        false_alarm = len(df[df['분류'] == '오탐'])
        normal = len(df[df['분류'] == '정상'])
        
        stats = {
            'total_rows': total,
            'surge_total': surge_total,
            'detect': detect,
            'pre_past': pre_past,        # 급증 케이스 사전감지 (N분전)
            'pre_future': pre_future,    # 정상 케이스 사전감지 (N분후)
            'normal_pred': normal_pred,
            'miss': miss,
            'false': false_alarm,
            'normal': normal
        }
        
        # 컬럼 순서 정리
        cols = df.columns.tolist()
        if '분류' in cols:
            cols.remove('분류')
            actual_idx = cols.index(actual_col)
            df = df[cols[:actual_idx+1] + ['분류'] + cols[actual_idx+1:]]
        
        return df, stats
    
    def run_analysis(self):
        if not self.hub_file and not self.m14_file:
            messagebox.showwarning("경고", "CSV 파일을 선택하세요!")
            return
        
        hub_cmt = self.hub_comment.get("1.0", tk.END).strip()
        m14_cmt = self.m14_comment.get("1.0", tk.END).strip()
        
        try:
            anal_date = f"{self.year_var.get()}년 {self.month_var.get()}월 {self.day_var.get()}일"
        except:
            anal_date = datetime.now().strftime("%Y년 %m월 %d일")
        
        result_text = ""
        
        try:
            if self.hub_file:
                df = pd.read_csv(self.hub_file, encoding='utf-8-sig')
                self.hub_df, self.hub_stats = self.analyze_v9(df.copy(), 350, '예측값_MAX', '현재값', '실제값')
                result_text += self.fmt("HUBROOM V9", self.hub_stats, 350)
                self.create_data_view(self.hub_data_frame, self.hub_df)
            
            if self.m14_file:
                df = pd.read_csv(self.m14_file, encoding='utf-8-sig')
                self.m14_df, self.m14_stats = self.analyze_v9(df.copy(), 1700, '최종예측', '현재TOTALCNT', '실제값')
                result_text += self.fmt("M14_Q V9", self.m14_stats, 1700)
                self.create_data_view(self.m14_data_frame, self.m14_df)
            
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, result_text)
            self.result_text.config(state=tk.DISABLED)
            
            save_dir = filedialog.askdirectory(title="결과 저장할 폴더 선택")
            if save_dir:
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                files = []
                
                if self.hub_df is not None:
                    f = os.path.join(save_dir, f"HUBROOM_V9_결과_{ts}.csv")
                    self.hub_df.to_csv(f, index=False, encoding='utf-8-sig')
                    files.append(f)
                
                if self.m14_df is not None:
                    f = os.path.join(save_dir, f"M14_V9_결과_{ts}.csv")
                    self.m14_df.to_csv(f, index=False, encoding='utf-8-sig')
                    files.append(f)
                
                html_f = os.path.join(save_dir, f"ML급증감지_V9_리포트_{ts}.html")
                with open(html_f, 'w', encoding='utf-8') as f:
                    f.write(self.gen_html(hub_cmt, m14_cmt, anal_date))
                files.append(html_f)
                
                webbrowser.open(html_f)
                messagebox.showinfo("완료", "저장 완료!\n\n" + "\n".join([os.path.basename(x) for x in files]))
            
        except Exception as e:
            import traceback
            messagebox.showerror("오류", f"{str(e)}\n\n{traceback.format_exc()}")
    
    def fmt(self, name, s, th):
        t = s['total_rows'] or 1
        surge = s['surge_total'] or 1
        normal_case = t - surge if t - surge > 0 else 1
        detect_rate = (s['detect'] + s['pre_past'] + s['normal_pred']) / surge * 100 if surge > 0 else 0
        
        return f"""
{'='*60}
  {name} (기준: {th})
{'='*60}
전체: {s['total_rows']}개

[ 급증 케이스: {s['surge_total']}개 ]
  ✅ 이상탐지: {s['detect']}개 ({s['detect']/surge*100:.1f}%)
  ⏰ 급증사전감지: {s['pre_past']}개 ({s['pre_past']/surge*100:.1f}%)
  ✅ 정상예측: {s['normal_pred']}개 ({s['normal_pred']/surge*100:.1f}%)
  ❌ 미탐: {s['miss']}개 ({s['miss']/surge*100:.1f}%)
  → 탐지율: {detect_rate:.1f}%

[ 정상 케이스: {t - surge}개 ]
  ✔️ 정상: {s['normal']}개 ({s['normal']/normal_case*100:.1f}%)
  ⏰ 정상사전감지: {s['pre_future']}개 ({s['pre_future']/normal_case*100:.1f}%)
  ⚠️ 오탐: {s['false']}개 ({s['false']/normal_case*100:.1f}%)
"""
    
    def gen_html(self, hub_cmt, m14_cmt, anal_date):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html = f'''<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>ML급증 감지 분석 리포트 V9</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'맑은 고딕',sans-serif;background:#0a0a0f;color:#fff;padding:40px}}
.container{{max-width:1400px;margin:0 auto}}
h1{{font-size:2.2rem;text-align:center;background:linear-gradient(135deg,#00d4ff,#ff6b35);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:10px}}
.sub{{text-align:center;color:#888;margin-bottom:25px}}
.section{{margin-bottom:50px}}
.stitle{{font-size:1.4rem;margin-bottom:20px;padding-bottom:10px;border-bottom:2px solid #333}}
.stitle.hub{{color:#00d4ff;border-color:#00d4ff}}
.stitle.m14{{color:#ff6b35;border-color:#ff6b35}}
.grid{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:20px}}
.card{{background:#12121a;padding:15px;border-radius:10px;text-align:center;border:1px solid #2a2a3a}}
.card-label{{font-size:0.75rem;color:#888;margin-bottom:6px}}
.card-value{{font-size:1.5rem;font-weight:bold}}
.card-sub{{font-size:0.8rem;color:#888;margin-top:4px}}
.ok{{color:#00ff88}}.warn{{color:#ffcc00}}.bad{{color:#ff4466}}.blue{{color:#00d4ff}}.orange{{color:#ff6b35}}.gray{{color:#aaa}}
.charts{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:25px}}
.cbox{{background:#12121a;padding:20px;border-radius:12px;border:1px solid #2a2a3a}}
.ctitle{{font-size:1rem;margin-bottom:15px;text-align:center}}
.cwrap{{height:280px}}
.summary{{background:#12121a;padding:20px;border-radius:12px;border:1px solid #2a2a3a;margin-bottom:15px}}
.summary pre{{font-family:Consolas,monospace;white-space:pre-wrap;color:#aaa;line-height:1.8;font-size:0.9rem}}
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
<h1>📊 ML급증 감지 분석 리포트 V9</h1>
<p class="sub">📅 분석 날짜: {anal_date} | 생성: {now}</p>
'''
        
        if self.hub_stats and self.hub_stats['total_rows'] > 0:
            s = self.hub_stats
            t = s['total_rows']
            surge = s['surge_total'] or 1
            det_rate = (s['detect']+s['pre_past']+s['normal_pred'])/surge*100
            
            cmt_html = ""
            if hub_cmt:
                cmt_html = f'<div class="comment"><div class="comment-title hub">📝 HUBROOM 코멘트</div><div class="comment-content">{hub_cmt.replace(chr(10),"<br>")}</div></div>'
            
            html += f'''
<div class="section">
<h2 class="stitle hub">🏠 HUBROOM V9 (기준: 350)</h2>
<div class="grid">
<div class="card"><div class="card-label">전체 데이터</div><div class="card-value">{t}</div></div>
<div class="card"><div class="card-label">급증 케이스</div><div class="card-value blue">{s['surge_total']}</div></div>
<div class="card"><div class="card-label">정상 케이스</div><div class="card-value gray">{t-s['surge_total']}</div></div>
<div class="card"><div class="card-label">탐지율</div><div class="card-value ok">{det_rate:.1f}%</div></div>
<div class="card"><div class="card-label">미탐</div><div class="card-value bad">{s['miss']}</div><div class="card-sub">{s['miss']/surge*100:.1f}%</div></div>
</div>
<div class="grid">
<div class="card"><div class="card-label">이상탐지</div><div class="card-value ok">{s['detect']}</div><div class="card-sub">{s['detect']/surge*100:.1f}%</div></div>
<div class="card"><div class="card-label">급증사전감지</div><div class="card-value warn">{s['pre_past']}</div><div class="card-sub">{s['pre_past']/surge*100:.1f}%</div></div>
<div class="card"><div class="card-label">정상예측</div><div class="card-value ok">{s['normal_pred']}</div><div class="card-sub">{s['normal_pred']/surge*100:.1f}%</div></div>
<div class="card"><div class="card-label">오탐</div><div class="card-value bad">{s['false']}</div><div class="card-sub">{s['false']/(t-surge)*100:.1f}%</div></div>
<div class="card"><div class="card-label">정상</div><div class="card-value gray">{s['normal']}</div><div class="card-sub">{s['normal']/(t-surge)*100:.1f}%</div></div>
</div>
<div class="charts">
<div class="cbox"><div class="ctitle">📈 급증 탐지 분포</div><div class="cwrap"><canvas id="hubSurge"></canvas></div></div>
<div class="cbox"><div class="ctitle">📊 정상 케이스 분포</div><div class="cwrap"><canvas id="hubNormal"></canvas></div></div>
</div>
<div class="summary"><pre>=== HUBROOM V9 요약 (기준: 350) ===
전체: {t}개 | 급증: {s['surge_total']}개 | 정상케이스: {t-s['surge_total']}개

[급증 탐지]
이상탐지: {s['detect']}개({s['detect']/surge*100:.1f}%) | 급증사전감지: {s['pre_past']}개({s['pre_past']/surge*100:.1f}%) | 정상예측: {s['normal_pred']}개({s['normal_pred']/surge*100:.1f}%) | 미탐: {s['miss']}개({s['miss']/surge*100:.1f}%)
→ 탐지율: {det_rate:.1f}%

[정상 케이스]
정상: {s['normal']}개 | 정상사전감지: {s['pre_future']}개 | 오탐: {s['false']}개</pre></div>
{cmt_html}
</div>
'''
        
        if self.m14_stats and self.m14_stats['total_rows'] > 0:
            s = self.m14_stats
            t = s['total_rows']
            surge = s['surge_total'] or 1
            det_rate = (s['detect']+s['pre_past']+s['normal_pred'])/surge*100
            normal_case = t - surge if t - surge > 0 else 1
            
            cmt_html = ""
            if m14_cmt:
                cmt_html = f'<div class="comment m14"><div class="comment-title m14">📝 M14_Q 코멘트</div><div class="comment-content">{m14_cmt.replace(chr(10),"<br>")}</div></div>'
            
            html += f'''
<div class="section">
<h2 class="stitle m14">🏭 M14_Q V9 (기준: 1700)</h2>
<div class="grid">
<div class="card"><div class="card-label">전체 데이터</div><div class="card-value">{t}</div></div>
<div class="card"><div class="card-label">급증 케이스</div><div class="card-value orange">{s['surge_total']}</div></div>
<div class="card"><div class="card-label">정상 케이스</div><div class="card-value gray">{t-s['surge_total']}</div></div>
<div class="card"><div class="card-label">탐지율</div><div class="card-value ok">{det_rate:.1f}%</div></div>
<div class="card"><div class="card-label">미탐</div><div class="card-value bad">{s['miss']}</div><div class="card-sub">{s['miss']/surge*100:.1f}%</div></div>
</div>
<div class="grid">
<div class="card"><div class="card-label">이상탐지</div><div class="card-value ok">{s['detect']}</div><div class="card-sub">{s['detect']/surge*100:.1f}%</div></div>
<div class="card"><div class="card-label">급증사전감지</div><div class="card-value warn">{s['pre_past']}</div><div class="card-sub">{s['pre_past']/surge*100:.1f}%</div></div>
<div class="card"><div class="card-label">정상예측</div><div class="card-value ok">{s['normal_pred']}</div><div class="card-sub">{s['normal_pred']/surge*100:.1f}%</div></div>
<div class="card"><div class="card-label">오탐</div><div class="card-value bad">{s['false']}</div><div class="card-sub">{s['false']/normal_case*100:.1f}%</div></div>
<div class="card"><div class="card-label">정상</div><div class="card-value gray">{s['normal']}</div><div class="card-sub">{s['normal']/normal_case*100:.1f}%</div></div>
</div>
<div class="charts">
<div class="cbox"><div class="ctitle">📈 급증 탐지 분포</div><div class="cwrap"><canvas id="m14Surge"></canvas></div></div>
<div class="cbox"><div class="ctitle">📊 정상 케이스 분포</div><div class="cwrap"><canvas id="m14Normal"></canvas></div></div>
</div>
<div class="summary"><pre>=== M14_Q V9 요약 (기준: 1700) ===
전체: {t}개 | 급증: {s['surge_total']}개 | 정상케이스: {t-s['surge_total']}개

[급증 탐지]
이상탐지: {s['detect']}개({s['detect']/surge*100:.1f}%) | 급증사전감지: {s['pre_past']}개({s['pre_past']/surge*100:.1f}%) | 정상예측: {s['normal_pred']}개({s['normal_pred']/surge*100:.1f}%) | 미탐: {s['miss']}개({s['miss']/surge*100:.1f}%)
→ 탐지율: {det_rate:.1f}%

[정상 케이스]
정상: {s['normal']}개 | 정상사전감지: {s['pre_future']}개 | 오탐: {s['false']}개</pre></div>
{cmt_html}
</div>
'''
        
        html += '''
<div class="footer">ML급증 감지 분석 시스템 V9 | AMHS 물류 예측 모델</div>
</div>
<script>
Chart.register(ChartDataLabels);
const opt={responsive:true,maintainAspectRatio:false,plugins:{legend:{position:'bottom',labels:{color:'#fff',font:{size:11}}},datalabels:{color:'#fff',font:{weight:'bold',size:12},formatter:(v,ctx)=>{let sum=ctx.chart.data.datasets[0].data.reduce((a,b)=>a+b,0);return v>0?(v/sum*100).toFixed(1)+'%':''}}}};
'''
        
        if self.hub_stats and self.hub_stats['total_rows'] > 0:
            s = self.hub_stats
            html += f'''
new Chart(document.getElementById('hubSurge'),{{type:'doughnut',data:{{labels:['이상탐지 ({s['detect']}개)','급증사전감지 ({s['pre_past']}개)','정상예측 ({s['normal_pred']}개)','미탐 ({s['miss']}개)'],datasets:[{{data:[{s['detect']},{s['pre_past']},{s['normal_pred']},{s['miss']}],backgroundColor:['#00ff88','#ffcc00','#00d4ff','#ff4466']}}]}},options:opt}});
new Chart(document.getElementById('hubNormal'),{{type:'doughnut',data:{{labels:['정상 ({s['normal']}개)','정상사전감지 ({s['pre_future']}개)','오탐 ({s['false']}개)'],datasets:[{{data:[{s['normal']},{s['pre_future']},{s['false']}],backgroundColor:['#666666','#ffcc00','#ff4466']}}]}},options:opt}});
'''
        
        if self.m14_stats and self.m14_stats['total_rows'] > 0:
            s = self.m14_stats
            html += f'''
new Chart(document.getElementById('m14Surge'),{{type:'doughnut',data:{{labels:['이상탐지 ({s['detect']}개)','급증사전감지 ({s['pre_past']}개)','정상예측 ({s['normal_pred']}개)','미탐 ({s['miss']}개)'],datasets:[{{data:[{s['detect']},{s['pre_past']},{s['normal_pred']},{s['miss']}],backgroundColor:['#00ff88','#ffcc00','#00d4ff','#ff4466']}}]}},options:opt}});
new Chart(document.getElementById('m14Normal'),{{type:'doughnut',data:{{labels:['정상 ({s['normal']}개)','정상사전감지 ({s['pre_future']}개)','오탐 ({s['false']}개)'],datasets:[{{data:[{s['normal']},{s['pre_future']},{s['false']}],backgroundColor:['#666666','#ffcc00','#ff4466']}}]}},options:opt}});
'''
        
        html += '</script></body></html>'
        return html


if __name__ == "__main__":
    root = tk.Tk()
    app = SurgeAnalyzer(root)
    root.mainloop()