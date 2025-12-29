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
                elif '급증연속' in 분류:
                    tag = 'surge_cont'
                elif '정상사전감지' in 분류:
                    tag = 'pre_future'
                elif '오탐' in 분류:
                    tag = 'false'
            tree.insert('', tk.END, values=values, tags=(tag,))
        
        tree.tag_configure('detect', background='#1a3a1a')
        tree.tag_configure('pre', background='#2a3a1a')
        tree.tag_configure('normal_pred', background='#1a2a3a')
        tree.tag_configure('miss', background='#3a1a1a')
        tree.tag_configure('surge_cont', background='#2a2a1a')
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
                # 정상 케이스 (실제값 < threshold)
                if pred >= threshold:
                    # 1) 10분 전에 급증 있는지 체크 → 급증연속
                    found_past = False
                    for back in range(1, 11):
                        if idx - back >= 0 and df.loc[idx - back, actual_col] >= threshold:
                            found_past = True
                            df.loc[idx, '분류'] = f'급증연속({back}분전)'
                            break
                    
                    if not found_past:
                        # 2) 10분 후에 급증 있는지 체크 → 정상사전감지
                        found_future = False
                        for fwd in range(1, 11):
                            if idx + fwd < len(df) and df.loc[idx + fwd, actual_col] >= threshold:
                                found_future = True
                                df.loc[idx, '분류'] = f'정상사전감지({fwd}분후)'
                                break
                        
                        if not found_future:
                            # 3) 둘 다 없으면 진짜 오탐
                            df.loc[idx, '분류'] = '오탐'
                else:
                    df.loc[idx, '분류'] = '정상'
        
        # 통계
        total = len(df)
        surge_total = len(df[df[actual_col] >= threshold])
        detect = len(df[df['분류'] == '이상탐지'])
        pre_past = len(df[df['분류'].str.contains('급증사전감지', na=False)])  # 급증케이스 사전감지
        pre_future = len(df[df['분류'].str.contains('정상사전감지', na=False)])  # 정상케이스 사전감지
        surge_cont = len(df[df['분류'].str.contains('급증연속', na=False)])  # 급증연속
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
            'surge_cont': surge_cont,    # 급증연속
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
        
        # 요약 계산
        surge_ok = s['detect'] + s['pre_past'] + s['normal_pred']
        surge_ng = s['miss']
        normal_ok = s['normal'] + s['pre_future'] + s['surge_cont']
        normal_ng = s['false']
        total_ok = surge_ok + normal_ok
        total_ng = surge_ng + normal_ng
        
        return f"""
{'='*60}
  {name} (기준: {th})
{'='*60}
전체: {s['total_rows']}개

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ 전체 예측OK: {total_ok}개 ({total_ok/t*100:.1f}%)
❌ 전체 예측NG: {total_ng}개 ({total_ng/t*100:.1f}%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[ 급증 케이스: {s['surge_total']}개 ]
  ✅ 이상탐지: {s['detect']}개 ({s['detect']/surge*100:.1f}%)
  ⏰ 급증사전감지: {s['pre_past']}개 ({s['pre_past']/surge*100:.1f}%)
  ✅ 정상예측: {s['normal_pred']}개 ({s['normal_pred']/surge*100:.1f}%)
  ❌ 미탐: {s['miss']}개 ({s['miss']/surge*100:.1f}%)
  ─────────────────────────────
  📊 이상탐지(±10분): {surge_ok}개 ({surge_ok/surge*100:.1f}%)
  📊 미탐: {surge_ng}개 ({surge_ng/surge*100:.1f}%)
  ─────────────────────────────
  ✅ 예측OK: {surge_ok}개 ({surge_ok/surge*100:.1f}%)
  ❌ 예측NG: {surge_ng}개 ({surge_ng/surge*100:.1f}%)

[ 정상 케이스: {t - surge}개 ]
  ✔️ 정상: {s['normal']}개 ({s['normal']/normal_case*100:.1f}%)
  🔄 급증연속: {s['surge_cont']}개 ({s['surge_cont']/normal_case*100:.1f}%)
  ⏰ 정상사전감지: {s['pre_future']}개 ({s['pre_future']/normal_case*100:.1f}%)
  ⚠️ 오탐: {s['false']}개 ({s['false']/normal_case*100:.1f}%)
  ─────────────────────────────
  📊 정상탐지: {normal_ok}개 ({normal_ok/normal_case*100:.1f}%)
  📊 오탐: {normal_ng}개 ({normal_ng/normal_case*100:.1f}%)
  ─────────────────────────────
  ✅ 예측OK: {normal_ok}개 ({normal_ok/normal_case*100:.1f}%)
  ❌ 예측NG: {normal_ng}개 ({normal_ng/normal_case*100:.1f}%)
"""
    
    def gen_html(self, hub_cmt, m14_cmt, anal_date):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 데이터를 JSON으로 변환 (그래프용)
        hub_json = "[]"
        m14_json = "[]"
        
        if self.hub_df is not None and len(self.hub_df) > 0:
            hub_data = []
            for _, row in self.hub_df.iterrows():
                hub_data.append({
                    'time': str(row.get('현재시간', '')),
                    'actual': float(row.get('실제값', 0)),
                    'pred': float(row.get('예측값_MAX', 0)),
                    'cls': str(row.get('분류', ''))
                })
            hub_json = str(hub_data).replace("'", '"')
        
        if self.m14_df is not None and len(self.m14_df) > 0:
            m14_data = []
            for _, row in self.m14_df.iterrows():
                m14_data.append({
                    'time': str(row.get('현재시간', '')),
                    'actual': float(row.get('실제값', 0)),
                    'pred': float(row.get('예측값_MAX', 0)),
                    'cls': str(row.get('분류', ''))
                })
            m14_json = str(m14_data).replace("'", '"')
        
        html = f'''<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>ML급증 감지 분석 리포트 V9</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'맑은 고딕',sans-serif;background:#0a0a0f;color:#fff;padding:30px}}
.container{{max-width:1200px;margin:0 auto}}
h1{{font-size:2rem;text-align:center;background:linear-gradient(135deg,#00d4ff,#ff6b35);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:8px}}
.sub{{text-align:center;color:#888;margin-bottom:30px;font-size:0.9rem}}
.system{{background:#12121a;border-radius:16px;padding:25px;margin-bottom:30px;border:1px solid #2a2a3a}}
.sys-header{{display:flex;justify-content:space-between;align-items:center;margin-bottom:20px;padding-bottom:15px;border-bottom:2px solid #333}}
.sys-title{{font-size:1.3rem;font-weight:bold}}
.sys-title.hub{{color:#00d4ff}}
.sys-title.m14{{color:#ff6b35}}
.sys-stats{{display:flex;gap:20px;font-size:0.95rem}}
.sys-stats span{{color:#aaa}}
.sys-stats strong{{margin-left:5px}}
.case-row{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px}}
.case-box{{background:#0a0a12;border-radius:12px;padding:20px;border:1px solid #333}}
.case-title{{font-size:1rem;font-weight:bold;margin-bottom:15px;display:flex;justify-content:space-between;align-items:center}}
.case-title.surge{{color:#00d4ff}}
.case-title.normal{{color:#888}}
.rate{{font-size:0.85rem;padding:4px 10px;border-radius:20px}}
.rate.good{{background:#1a3a1a;color:#00ff88}}
.rate.bad{{background:#3a1a1a;color:#ff4466}}
.items{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}}
.item{{background:#15151f;padding:12px;border-radius:8px;text-align:center;cursor:pointer;transition:all 0.2s}}
.item:hover{{background:#1a1a2a;transform:scale(1.02);box-shadow:0 0 10px rgba(0,212,255,0.3)}}
.item-label{{font-size:0.7rem;color:#666;margin-bottom:4px}}
.item-value{{font-size:1.3rem;font-weight:bold}}
.item-sub{{font-size:0.75rem;color:#888;margin-top:3px}}
.ok{{color:#00ff88}}.warn{{color:#ffcc00}}.bad{{color:#ff4466}}.gray{{color:#aaa}}
.summary-row{{display:flex;gap:10px;margin-top:15px;padding-top:12px;border-top:1px solid #333}}
.sum-ok,.sum-ng{{flex:1;padding:10px 12px;border-radius:8px;font-size:0.85rem}}
.sum-ok{{background:#1a2a1a;color:#00ff88}}
.sum-ng{{background:#2a1a1a;color:#ff4466}}
.sum-ok strong,.sum-ng strong{{font-size:1.1rem;margin:0 5px}}
.sum-ok span,.sum-ng span{{opacity:0.8}}
.mid-summary{{display:flex;gap:10px;margin-top:12px;padding:10px;background:#0a0a12;border-radius:8px}}
.mid-item{{flex:1;display:flex;align-items:center;justify-content:center;gap:8px;padding:8px 10px;border-radius:6px;cursor:pointer;transition:all 0.2s}}
.mid-item:hover{{transform:scale(1.02)}}
.mid-item.ok-box{{background:#1a2a1a;border:1px solid #00ff8855}}
.mid-item.ng-box{{background:#2a1a1a;border:1px solid #ff446655}}
.mid-label{{font-size:0.75rem;color:#aaa}}
.mid-item strong{{font-size:1.1rem;color:#fff}}
.mid-pct{{font-size:0.8rem;color:#888}}
.mid-item.ok-box strong{{color:#00ff88}}
.mid-item.ng-box strong{{color:#ff4466}}
.items-3{{grid-template-columns:repeat(3,1fr)}}
.total-summary{{display:flex;gap:15px;margin-bottom:15px;padding:15px;background:#0a0a12;border-radius:10px;border:1px solid #333}}
.total-ok,.total-ng{{flex:1;padding:12px 15px;border-radius:8px;font-size:0.95rem;text-align:center}}
.total-ok{{background:#1a3a1a;color:#00ff88;border:1px solid #00ff88}}
.total-ng{{background:#3a1a1a;color:#ff4466;border:1px solid #ff4466}}
.total-ok strong,.total-ng strong{{font-size:1.3rem;margin:0 5px}}
.total-ok span,.total-ng span{{opacity:0.8}}
.charts{{display:grid;grid-template-columns:1fr 1fr;gap:15px}}
.chart-box{{background:#0a0a12;border-radius:12px;padding:15px;border:1px solid #333}}
.chart-title{{font-size:0.85rem;color:#888;text-align:center;margin-bottom:10px}}
.chart-wrap{{height:200px}}
.comment{{background:#1a2a1a;border:2px solid #00ff88;border-radius:10px;padding:15px;margin-top:15px}}
.comment.m14{{background:#2a1a1a;border-color:#ff6b35}}
.comment-title{{font-weight:bold;margin-bottom:6px;font-size:0.9rem}}
.comment-title.hub{{color:#00d4ff}}
.comment-title.m14{{color:#ff6b35}}
.comment-content{{color:#ccc;line-height:1.6;font-size:0.9rem}}
.footer{{text-align:center;color:#555;margin-top:40px;font-size:0.8rem}}
.modal{{display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.85);z-index:1000;justify-content:center;align-items:center}}
.modal.show{{display:flex}}
.modal-content{{background:#12121a;border-radius:16px;padding:25px;width:90%;max-width:1000px;max-height:85vh;overflow-y:auto;border:1px solid #333}}
.modal-header{{display:flex;justify-content:space-between;align-items:center;margin-bottom:20px;padding-bottom:15px;border-bottom:1px solid #333}}
.modal-title{{font-size:1.2rem;font-weight:bold}}
.modal-close{{background:none;border:none;color:#888;font-size:1.5rem;cursor:pointer}}
.modal-close:hover{{color:#fff}}
.modal-chart{{height:400px;margin-bottom:20px}}
.modal-info{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}}
.modal-stat{{background:#0a0a12;padding:12px;border-radius:8px;text-align:center}}
.modal-stat-label{{font-size:0.75rem;color:#666;margin-bottom:4px}}
.modal-stat-value{{font-size:1.1rem;font-weight:bold}}
@media(max-width:800px){{.case-row,.charts,.items,.modal-info{{grid-template-columns:1fr}}.sys-header{{flex-direction:column;gap:10px}}.sys-stats{{flex-wrap:wrap}}}}
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
            normal_case = t - surge if t - surge > 0 else 1
            det_rate = (s['detect']+s['pre_past']+s['normal_pred'])/surge*100
            
            # 요약 계산
            surge_ok = s['detect'] + s['pre_past'] + s['normal_pred']
            surge_ng = s['miss']
            normal_ok = s['normal'] + s['pre_future'] + s['surge_cont']
            normal_ng = s['false']
            total_ok = surge_ok + normal_ok
            total_ng = surge_ng + normal_ng
            
            cmt_html = ""
            if hub_cmt:
                cmt_html = f'<div class="comment"><div class="comment-title hub">📝 코멘트</div><div class="comment-content">{hub_cmt.replace(chr(10),"<br>")}</div></div>'
            
            html += f'''
<div class="system">
<div class="sys-header">
<div class="sys-title hub">🏠 HUBROOM V9 (기준: 350)</div>
<div class="sys-stats">
<span>전체 <strong>{t}</strong>개</span>
<span>급증 <strong style="color:#00d4ff">{s['surge_total']}</strong>개</span>
<span>정상 <strong style="color:#aaa">{t-s['surge_total']}</strong>개</span>
</div>
</div>
<div class="total-summary">
<div class="total-ok">✅ 전체 예측OK <strong>{total_ok}</strong>개 <span>({total_ok/t*100:.1f}%)</span></div>
<div class="total-ng">❌ 전체 예측NG <strong>{total_ng}</strong>개 <span>({total_ng/t*100:.1f}%)</span></div>
</div>
<div class="case-row">
<div class="case-box">
<div class="case-title surge">📈 급증 케이스 ({s['surge_total']}개)</div>
<div class="items">
<div class="item" onclick="showChart('hub','이상탐지',350)"><div class="item-label">이상탐지</div><div class="item-value ok">{s['detect']}</div><div class="item-sub">{s['detect']/surge*100:.1f}%</div></div>
<div class="item" onclick="showChart('hub','급증사전감지',350)"><div class="item-label">급증사전감지</div><div class="item-value warn">{s['pre_past']}</div><div class="item-sub">{s['pre_past']/surge*100:.1f}%</div></div>
<div class="item" onclick="showChart('hub','정상예측',350)"><div class="item-label">정상예측</div><div class="item-value ok">{s['normal_pred']}</div><div class="item-sub">{s['normal_pred']/surge*100:.1f}%</div></div>
<div class="item" onclick="showChart('hub','미탐',350)"><div class="item-label">미탐</div><div class="item-value bad">{s['miss']}</div><div class="item-sub">{s['miss']/surge*100:.1f}%</div></div>
</div>
<div class="mid-summary">
<div class="mid-item ok-box" onclick="showChart('hub','이상탐지_ALL',350)"><span class="mid-label">이상탐지(±10분)</span><strong>{surge_ok}</strong><span class="mid-pct">{surge_ok/surge*100:.1f}%</span></div>
<div class="mid-item ng-box" onclick="showChart('hub','미탐',350)"><span class="mid-label">미탐</span><strong>{surge_ng}</strong><span class="mid-pct">{surge_ng/surge*100:.1f}%</span></div>
</div>
<div class="summary-row">
<div class="sum-ok">✅ 예측OK <strong>{surge_ok}</strong> <span>({surge_ok/surge*100:.1f}%)</span></div>
<div class="sum-ng">❌ 예측NG <strong>{surge_ng}</strong> <span>({surge_ng/surge*100:.1f}%)</span></div>
</div>
</div>
<div class="case-box">
<div class="case-title normal">📊 정상 케이스 ({t-s['surge_total']}개)</div>
<div class="items">
<div class="item" onclick="showChart('hub','정상',350)"><div class="item-label">정상</div><div class="item-value gray">{s['normal']}</div><div class="item-sub">{s['normal']/normal_case*100:.1f}%</div></div>
<div class="item" onclick="showChart('hub','급증연속',350)"><div class="item-label">급증연속</div><div class="item-value warn">{s['surge_cont']}</div><div class="item-sub">{s['surge_cont']/normal_case*100:.1f}%</div></div>
<div class="item" onclick="showChart('hub','정상사전감지',350)"><div class="item-label">정상사전감지</div><div class="item-value warn">{s['pre_future']}</div><div class="item-sub">{s['pre_future']/normal_case*100:.1f}%</div></div>
<div class="item" onclick="showChart('hub','오탐',350)"><div class="item-label">오탐</div><div class="item-value bad">{s['false']}</div><div class="item-sub">{s['false']/normal_case*100:.1f}%</div></div>
</div>
<div class="mid-summary">
<div class="mid-item ok-box" onclick="showChart('hub','정상탐지_ALL',350)"><span class="mid-label">정상탐지</span><strong>{normal_ok}</strong><span class="mid-pct">{normal_ok/normal_case*100:.1f}%</span></div>
<div class="mid-item ng-box" onclick="showChart('hub','오탐',350)"><span class="mid-label">오탐</span><strong>{normal_ng}</strong><span class="mid-pct">{normal_ng/normal_case*100:.1f}%</span></div>
</div>
<div class="summary-row">
<div class="sum-ok">✅ 예측OK <strong>{normal_ok}</strong> <span>({normal_ok/normal_case*100:.1f}%)</span></div>
<div class="sum-ng">❌ 예측NG <strong>{normal_ng}</strong> <span>({normal_ng/normal_case*100:.1f}%)</span></div>
</div>
</div>
</div>
<div class="charts">
<div class="chart-box"><div class="chart-title">급증 탐지 분포</div><div class="chart-wrap"><canvas id="hubSurge"></canvas></div></div>
<div class="chart-box"><div class="chart-title">정상 케이스 분포</div><div class="chart-wrap"><canvas id="hubNormal"></canvas></div></div>
</div>
{cmt_html}
</div>
'''
        
        if self.m14_stats and self.m14_stats['total_rows'] > 0:
            s = self.m14_stats
            t = s['total_rows']
            surge = s['surge_total'] or 1
            det_rate = (s['detect']+s['pre_past']+s['normal_pred'])/surge*100
            normal_case = t - surge if t - surge > 0 else 1
            
            # 요약 계산
            surge_ok = s['detect'] + s['pre_past'] + s['normal_pred']
            surge_ng = s['miss']
            normal_ok = s['normal'] + s['pre_future'] + s['surge_cont']
            normal_ng = s['false']
            total_ok = surge_ok + normal_ok
            total_ng = surge_ng + normal_ng
            
            cmt_html = ""
            if m14_cmt:
                cmt_html = f'<div class="comment m14"><div class="comment-title m14">📝 코멘트</div><div class="comment-content">{m14_cmt.replace(chr(10),"<br>")}</div></div>'
            
            html += f'''
<div class="system">
<div class="sys-header">
<div class="sys-title m14">🏭 M14_Q V9 (기준: 1700)</div>
<div class="sys-stats">
<span>전체 <strong>{t}</strong>개</span>
<span>급증 <strong style="color:#ff6b35">{s['surge_total']}</strong>개</span>
<span>정상 <strong style="color:#aaa">{t-s['surge_total']}</strong>개</span>
</div>
</div>
<div class="total-summary">
<div class="total-ok">✅ 전체 예측OK <strong>{total_ok}</strong>개 <span>({total_ok/t*100:.1f}%)</span></div>
<div class="total-ng">❌ 전체 예측NG <strong>{total_ng}</strong>개 <span>({total_ng/t*100:.1f}%)</span></div>
</div>
<div class="case-row">
<div class="case-box">
<div class="case-title surge" style="color:#ff6b35">📈 급증 케이스 ({s['surge_total']}개)</div>
<div class="items">
<div class="item" onclick="showChart('m14','이상탐지',1700)"><div class="item-label">이상탐지</div><div class="item-value ok">{s['detect']}</div><div class="item-sub">{s['detect']/surge*100:.1f}%</div></div>
<div class="item" onclick="showChart('m14','급증사전감지',1700)"><div class="item-label">급증사전감지</div><div class="item-value warn">{s['pre_past']}</div><div class="item-sub">{s['pre_past']/surge*100:.1f}%</div></div>
<div class="item" onclick="showChart('m14','정상예측',1700)"><div class="item-label">정상예측</div><div class="item-value ok">{s['normal_pred']}</div><div class="item-sub">{s['normal_pred']/surge*100:.1f}%</div></div>
<div class="item" onclick="showChart('m14','미탐',1700)"><div class="item-label">미탐</div><div class="item-value bad">{s['miss']}</div><div class="item-sub">{s['miss']/surge*100:.1f}%</div></div>
</div>
<div class="mid-summary">
<div class="mid-item ok-box" onclick="showChart('m14','이상탐지_ALL',1700)"><span class="mid-label">이상탐지(±10분)</span><strong>{surge_ok}</strong><span class="mid-pct">{surge_ok/surge*100:.1f}%</span></div>
<div class="mid-item ng-box" onclick="showChart('m14','미탐',1700)"><span class="mid-label">미탐</span><strong>{surge_ng}</strong><span class="mid-pct">{surge_ng/surge*100:.1f}%</span></div>
</div>
<div class="summary-row">
<div class="sum-ok">✅ 예측OK <strong>{surge_ok}</strong> <span>({surge_ok/surge*100:.1f}%)</span></div>
<div class="sum-ng">❌ 예측NG <strong>{surge_ng}</strong> <span>({surge_ng/surge*100:.1f}%)</span></div>
</div>
</div>
<div class="case-box">
<div class="case-title normal">📊 정상 케이스 ({t-s['surge_total']}개)</div>
<div class="items">
<div class="item" onclick="showChart('m14','정상',1700)"><div class="item-label">정상</div><div class="item-value gray">{s['normal']}</div><div class="item-sub">{s['normal']/normal_case*100:.1f}%</div></div>
<div class="item" onclick="showChart('m14','급증연속',1700)"><div class="item-label">급증연속</div><div class="item-value warn">{s['surge_cont']}</div><div class="item-sub">{s['surge_cont']/normal_case*100:.1f}%</div></div>
<div class="item" onclick="showChart('m14','정상사전감지',1700)"><div class="item-label">정상사전감지</div><div class="item-value warn">{s['pre_future']}</div><div class="item-sub">{s['pre_future']/normal_case*100:.1f}%</div></div>
<div class="item" onclick="showChart('m14','오탐',1700)"><div class="item-label">오탐</div><div class="item-value bad">{s['false']}</div><div class="item-sub">{s['false']/normal_case*100:.1f}%</div></div>
</div>
<div class="mid-summary">
<div class="mid-item ok-box" onclick="showChart('m14','정상탐지_ALL',1700)"><span class="mid-label">정상탐지</span><strong>{normal_ok}</strong><span class="mid-pct">{normal_ok/normal_case*100:.1f}%</span></div>
<div class="mid-item ng-box" onclick="showChart('m14','오탐',1700)"><span class="mid-label">오탐</span><strong>{normal_ng}</strong><span class="mid-pct">{normal_ng/normal_case*100:.1f}%</span></div>
</div>
<div class="summary-row">
<div class="sum-ok">✅ 예측OK <strong>{normal_ok}</strong> <span>({normal_ok/normal_case*100:.1f}%)</span></div>
<div class="sum-ng">❌ 예측NG <strong>{normal_ng}</strong> <span>({normal_ng/normal_case*100:.1f}%)</span></div>
</div>
</div>
</div>
<div class="charts">
<div class="chart-box"><div class="chart-title">급증 탐지 분포</div><div class="chart-wrap"><canvas id="m14Surge"></canvas></div></div>
<div class="chart-box"><div class="chart-title">정상 케이스 분포</div><div class="chart-wrap"><canvas id="m14Normal"></canvas></div></div>
</div>
{cmt_html}
</div>
'''
        
        html += '''
<div class="footer">ML급증 감지 분석 시스템 V9 | AMHS 물류 예측 모델</div>
</div>
<script>
Chart.register(ChartDataLabels);
const opt={responsive:true,maintainAspectRatio:false,plugins:{legend:{position:'bottom',labels:{color:'#fff',font:{size:10},padding:8}},datalabels:{color:'#fff',font:{weight:'bold',size:11},formatter:(v,ctx)=>{let sum=ctx.chart.data.datasets[0].data.reduce((a,b)=>a+b,0);return v>0?(v/sum*100).toFixed(1)+'%':''}}}};
'''
        
        if self.hub_stats and self.hub_stats['total_rows'] > 0:
            s = self.hub_stats
            html += f'''
new Chart(document.getElementById('hubSurge'),{{type:'doughnut',data:{{labels:['이상탐지 ({s['detect']}개)','급증사전감지 ({s['pre_past']}개)','정상예측 ({s['normal_pred']}개)','미탐 ({s['miss']}개)'],datasets:[{{data:[{s['detect']},{s['pre_past']},{s['normal_pred']},{s['miss']}],backgroundColor:['#00ff88','#ffcc00','#00d4ff','#ff4466']}}]}},options:opt}});
new Chart(document.getElementById('hubNormal'),{{type:'doughnut',data:{{labels:['정상 ({s['normal']}개)','급증연속 ({s['surge_cont']}개)','정상사전감지 ({s['pre_future']}개)','오탐 ({s['false']}개)'],datasets:[{{data:[{s['normal']},{s['surge_cont']},{s['pre_future']},{s['false']}],backgroundColor:['#666666','#ff9500','#ffcc00','#ff4466']}}]}},options:opt}});
'''
        
        if self.m14_stats and self.m14_stats['total_rows'] > 0:
            s = self.m14_stats
            html += f'''
new Chart(document.getElementById('m14Surge'),{{type:'doughnut',data:{{labels:['이상탐지 ({s['detect']}개)','급증사전감지 ({s['pre_past']}개)','정상예측 ({s['normal_pred']}개)','미탐 ({s['miss']}개)'],datasets:[{{data:[{s['detect']},{s['pre_past']},{s['normal_pred']},{s['miss']}],backgroundColor:['#00ff88','#ffcc00','#00d4ff','#ff4466']}}]}},options:opt}});
new Chart(document.getElementById('m14Normal'),{{type:'doughnut',data:{{labels:['정상 ({s['normal']}개)','급증연속 ({s['surge_cont']}개)','정상사전감지 ({s['pre_future']}개)','오탐 ({s['false']}개)'],datasets:[{{data:[{s['normal']},{s['surge_cont']},{s['pre_future']},{s['false']}],backgroundColor:['#666666','#ff9500','#ffcc00','#ff4466']}}]}},options:opt}});
'''
        
        # 모달과 그래프 스크립트 추가
        html += f'''
const hubData = {hub_json};
const m14Data = {m14_json};
let modalChart = null;

function showChart(system, cls, threshold) {{
    const data = system === 'hub' ? hubData : m14Data;
    const sysName = system === 'hub' ? 'HUBROOM' : 'M14_Q';
    
    let filtered = [];
    if (cls === '이상탐지_ALL') {{
        filtered = data.filter(d => d.cls === '이상탐지' || d.cls.includes('급증사전감지') || d.cls.includes('정상예측'));
    }} else if (cls === '정상탐지_ALL') {{
        filtered = data.filter(d => d.cls === '정상' || d.cls.includes('급증연속') || d.cls.includes('정상사전감지'));
    }} else {{
        filtered = data.filter(d => d.cls.includes(cls));
    }}
    
    if (filtered.length === 0) {{
        alert('해당 분류의 데이터가 없습니다.');
        return;
    }}
    
    // 최대 500개만 샘플링 (성능)
    if (filtered.length > 500) {{
        const step = Math.ceil(filtered.length / 500);
        filtered = filtered.filter((_, i) => i % step === 0);
    }}
    
    document.getElementById('modalTitle').textContent = `${{sysName}} - ${{cls}} (${{filtered.length}}개)`;
    document.getElementById('chartModal').classList.add('show');
    
    const ctx = document.getElementById('modalLineChart').getContext('2d');
    if (modalChart) modalChart.destroy();
    
    modalChart = new Chart(ctx, {{
        type: 'line',
        data: {{
            labels: filtered.map(d => d.time.split(' ')[1] || d.time),
            datasets: [
                {{
                    label: '실제값',
                    data: filtered.map(d => d.actual),
                    borderColor: '#00d4ff',
                    backgroundColor: 'rgba(0,212,255,0.1)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 2
                }},
                {{
                    label: '예측값',
                    data: filtered.map(d => d.pred),
                    borderColor: '#ff6b35',
                    backgroundColor: 'rgba(255,107,53,0.1)',
                    fill: false,
                    tension: 0.3,
                    pointRadius: 2
                }},
                {{
                    label: '임계값',
                    data: filtered.map(() => threshold),
                    borderColor: '#ff4466',
                    borderDash: [5, 5],
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
                datalabels: {{ display: false }}
            }},
            scales: {{
                x: {{ ticks: {{ color: '#888', maxTicksLimit: 20 }}, grid: {{ color: '#333' }} }},
                y: {{ ticks: {{ color: '#888' }}, grid: {{ color: '#333' }} }}
            }}
        }}
    }});
    
    // 통계
    const actuals = filtered.map(d => d.actual);
    const preds = filtered.map(d => d.pred);
    document.getElementById('statCount').textContent = filtered.length;
    document.getElementById('statActualAvg').textContent = (actuals.reduce((a,b)=>a+b,0)/actuals.length).toFixed(1);
    document.getElementById('statPredAvg').textContent = (preds.reduce((a,b)=>a+b,0)/preds.length).toFixed(1);
    document.getElementById('statActualMax').textContent = Math.max(...actuals).toFixed(1);
}}

function closeModal() {{
    document.getElementById('chartModal').classList.remove('show');
    if (modalChart) {{ modalChart.destroy(); modalChart = null; }}
}}

document.getElementById('chartModal').addEventListener('click', function(e) {{
    if (e.target === this) closeModal();
}});
</script>

<div id="chartModal" class="modal">
<div class="modal-content">
<div class="modal-header">
<div id="modalTitle" class="modal-title">그래프</div>
<button class="modal-close" onclick="closeModal()">&times;</button>
</div>
<div class="modal-chart"><canvas id="modalLineChart"></canvas></div>
<div class="modal-info">
<div class="modal-stat"><div class="modal-stat-label">데이터 수</div><div id="statCount" class="modal-stat-value ok">0</div></div>
<div class="modal-stat"><div class="modal-stat-label">실제값 평균</div><div id="statActualAvg" class="modal-stat-value" style="color:#00d4ff">0</div></div>
<div class="modal-stat"><div class="modal-stat-label">예측값 평균</div><div id="statPredAvg" class="modal-stat-value" style="color:#ff6b35">0</div></div>
<div class="modal-stat"><div class="modal-stat-label">실제값 최대</div><div id="statActualMax" class="modal-stat-value bad">0</div></div>
</div>
</div>
</div>
</body></html>'''
        return html


if __name__ == "__main__":
    root = tk.Tk()
    app = SurgeAnalyzer(root)
    root.mainloop()