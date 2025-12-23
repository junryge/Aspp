import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
from datetime import datetime
import webbrowser
import os

class SurgeAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("급증 감지 분석 시스템 V8")
        self.root.geometry("800x600")
        self.root.configure(bg='#1a1a2e')
        
        self.hub_file = None
        self.m14_file = None
        self.hub_df = None
        self.m14_df = None
        self.hub_stats = None
        self.m14_stats = None
        
        self.setup_ui()
    
    def setup_ui(self):
        # 스타일 설정
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#1a1a2e')
        style.configure('TLabel', background='#1a1a2e', foreground='white', font=('맑은 고딕', 10))
        style.configure('Title.TLabel', font=('맑은 고딕', 24, 'bold'), foreground='#00d4ff')
        style.configure('Sub.TLabel', font=('맑은 고딕', 12), foreground='#888888')
        style.configure('TButton', font=('맑은 고딕', 11, 'bold'), padding=10)
        
        # 메인 프레임
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)
        
        # 타이틀
        ttk.Label(main_frame, text="📊 급증 감지 분석 시스템", style='Title.TLabel').pack(pady=(0, 5))
        ttk.Label(main_frame, text="HUBROOM V8.0 | M14_Q V8.3.1", style='Sub.TLabel').pack(pady=(0, 30))
        
        # 파일 업로드 프레임
        upload_frame = ttk.Frame(main_frame)
        upload_frame.pack(fill=tk.X, pady=10)
        
        # HUBROOM
        hub_frame = tk.Frame(upload_frame, bg='#252540', relief='ridge', bd=2)
        hub_frame.pack(fill=tk.X, pady=10, ipady=15, ipadx=15)
        
        tk.Label(hub_frame, text="🏠 HUBROOM V8.0", font=('맑은 고딕', 14, 'bold'), 
                 bg='#252540', fg='#00d4ff').pack(anchor='w', padx=10, pady=(10,5))
        tk.Label(hub_frame, text="기준값: 350 | 컬럼: 현재값, 실제값, 예측값_MAX", 
                 font=('맑은 고딕', 9), bg='#252540', fg='#888888').pack(anchor='w', padx=10)
        
        hub_btn_frame = tk.Frame(hub_frame, bg='#252540')
        hub_btn_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Button(hub_btn_frame, text="📁 파일 선택", command=self.select_hub_file,
                  font=('맑은 고딕', 10), bg='#00d4ff', fg='black', width=15).pack(side=tk.LEFT)
        self.hub_label = tk.Label(hub_btn_frame, text="파일을 선택하세요", 
                                   font=('맑은 고딕', 9), bg='#252540', fg='#888888')
        self.hub_label.pack(side=tk.LEFT, padx=15)
        
        # M14_Q
        m14_frame = tk.Frame(upload_frame, bg='#252540', relief='ridge', bd=2)
        m14_frame.pack(fill=tk.X, pady=10, ipady=15, ipadx=15)
        
        tk.Label(m14_frame, text="🏭 M14_Q V8.3.1", font=('맑은 고딕', 14, 'bold'), 
                 bg='#252540', fg='#ff6b35').pack(anchor='w', padx=10, pady=(10,5))
        tk.Label(m14_frame, text="기준값: 1700 | 컬럼: 현재TOTALCNT, 실제값, 최종예측", 
                 font=('맑은 고딕', 9), bg='#252540', fg='#888888').pack(anchor='w', padx=10)
        
        m14_btn_frame = tk.Frame(m14_frame, bg='#252540')
        m14_btn_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Button(m14_btn_frame, text="📁 파일 선택", command=self.select_m14_file,
                  font=('맑은 고딕', 10), bg='#ff6b35', fg='black', width=15).pack(side=tk.LEFT)
        self.m14_label = tk.Label(m14_btn_frame, text="파일을 선택하세요", 
                                   font=('맑은 고딕', 9), bg='#252540', fg='#888888')
        self.m14_label.pack(side=tk.LEFT, padx=15)
        
        # 분석 버튼
        tk.Button(main_frame, text="🔍 분석 시작", command=self.run_analysis,
                  font=('맑은 고딕', 14, 'bold'), bg='#00ff88', fg='black',
                  width=30, height=2).pack(pady=30)
        
        # 결과 영역
        self.result_frame = tk.Frame(main_frame, bg='#1a1a2e')
        self.result_frame.pack(fill=tk.BOTH, expand=True)
        
        self.result_text = tk.Text(self.result_frame, font=('Consolas', 10), 
                                    bg='#0d0d15', fg='white', height=12,
                                    relief='flat', padx=15, pady=15)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        self.result_text.insert(tk.END, "CSV 파일을 선택하고 분석을 시작하세요...")
        self.result_text.config(state=tk.DISABLED)
        
        # 다운로드 버튼 프레임
        self.download_frame = tk.Frame(main_frame, bg='#1a1a2e')
        self.download_frame.pack(fill=tk.X, pady=15)
    
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
        drop_cols = ['Unnamed: 22', 'Unnamed: 23']
        for col in drop_cols:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        df['사전감지OK_NG'] = ''
        surge_condition = (df['실제값'] >= threshold) & (df['현재값'] <= threshold)
        df.loc[surge_condition, '사전감지OK_NG'] = 'Y'
        
        df['사전감지결과'] = ''
        ok_condition = surge_condition & (df['예측값_MAX'] >= threshold)
        df.loc[ok_condition, '사전감지결과'] = 'OK'
        
        ng_condition = surge_condition & (df['예측값_MAX'] < threshold)
        ng_indices = df[ng_condition].index.tolist()
        
        for idx in ng_indices:
            found = False
            for back in range(1, 6):
                prev_idx = idx - back
                if prev_idx >= 0:
                    prev_pred = df.loc[prev_idx, '예측값_MAX']
                    if prev_pred >= threshold:
                        df.loc[idx, '사전감지결과'] = f'사전감지({back}분전)'
                        found = True
                        break
            if not found:
                df.loc[idx, '사전감지결과'] = 'NG'
        
        df['오탐/사전감지'] = ''
        fp_condition = (df['실제값'] < threshold) & (df['예측값_MAX'] >= threshold)
        fp_indices = df[fp_condition].index.tolist()
        
        for idx in fp_indices:
            found = False
            for offset in range(-5, 6):
                if offset == 0:
                    continue
                check_idx = idx + offset
                if 0 <= check_idx < len(df):
                    if df.loc[check_idx, '실제값'] >= threshold:
                        df.loc[idx, '오탐/사전감지'] = '오탐사전감지'
                        found = True
                        break
            if not found:
                df.loc[idx, '오탐/사전감지'] = '오탐'
        
        cols = df.columns.tolist()
        cols.remove('사전감지OK_NG')
        cols.remove('사전감지결과')
        cols.remove('오탐/사전감지')
        actual_idx = cols.index('실제값')
        new_cols = cols[:actual_idx+1] + ['사전감지OK_NG', '사전감지결과', '오탐/사전감지'] + cols[actual_idx+1:]
        df = df[new_cols]
        
        total_surge = len(df[surge_condition])
        ok_cnt = len(df[df['사전감지결과'] == 'OK'])
        pre_cnt = len(df[df['사전감지결과'].str.contains('사전감지', na=False)])
        ng_cnt = len(df[df['사전감지결과'] == 'NG'])
        fp_total = len(fp_indices)
        fp_pre = len(df[df['오탐/사전감지'] == '오탐사전감지'])
        fp_real = len(df[df['오탐/사전감지'] == '오탐'])
        
        return df, {
            'total': total_surge, 'ok': ok_cnt, 'pre': pre_cnt, 'ng': ng_cnt,
            'fp_total': fp_total, 'fp_pre': fp_pre, 'fp_real': fp_real
        }
    
    def analyze_m14(self, df, threshold=1700):
        drop_cols = ['12분전TOTALCNT', '12변화량', 'Unnamed: 22', 'Unnamed: 23']
        for col in drop_cols:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        df['사전감지OK_NG'] = ''
        surge_condition = (df['실제값'] >= threshold) & (df['현재TOTALCNT'] <= threshold)
        df.loc[surge_condition, '사전감지OK_NG'] = 'Y'
        
        df['사전감지결과'] = ''
        ok_condition = surge_condition & (df['최종예측'] >= threshold)
        df.loc[ok_condition, '사전감지결과'] = 'OK'
        
        ng_condition = surge_condition & (df['최종예측'] < threshold)
        ng_indices = df[ng_condition].index.tolist()
        
        for idx in ng_indices:
            found = False
            for back in range(1, 6):
                prev_idx = idx - back
                if prev_idx >= 0:
                    prev_pred = df.loc[prev_idx, '최종예측']
                    if prev_pred >= threshold:
                        df.loc[idx, '사전감지결과'] = f'사전감지({back}분전)'
                        found = True
                        break
            if not found:
                df.loc[idx, '사전감지결과'] = 'NG'
        
        df['오탐/사전감지'] = ''
        fp_condition = (df['실제값'] < threshold) & (df['최종예측'] >= threshold)
        fp_indices = df[fp_condition].index.tolist()
        
        for idx in fp_indices:
            found = False
            for offset in range(-5, 6):
                if offset == 0:
                    continue
                check_idx = idx + offset
                if 0 <= check_idx < len(df):
                    if df.loc[check_idx, '실제값'] >= threshold:
                        df.loc[idx, '오탐/사전감지'] = '오탐사전감지'
                        found = True
                        break
            if not found:
                df.loc[idx, '오탐/사전감지'] = '오탐'
        
        cols = df.columns.tolist()
        cols.remove('사전감지OK_NG')
        cols.remove('사전감지결과')
        cols.remove('오탐/사전감지')
        actual_idx = cols.index('실제값')
        new_cols = cols[:actual_idx+1] + ['사전감지OK_NG', '사전감지결과', '오탐/사전감지'] + cols[actual_idx+1:]
        df = df[new_cols]
        
        total_surge = len(df[surge_condition])
        ok_cnt = len(df[df['사전감지결과'] == 'OK'])
        pre_cnt = len(df[df['사전감지결과'].str.contains('사전감지', na=False)])
        ng_cnt = len(df[df['사전감지결과'] == 'NG'])
        fp_total = len(fp_indices)
        fp_pre = len(df[df['오탐/사전감지'] == '오탐사전감지'])
        fp_real = len(df[df['오탐/사전감지'] == '오탐'])
        
        return df, {
            'total': total_surge, 'ok': ok_cnt, 'pre': pre_cnt, 'ng': ng_cnt,
            'fp_total': fp_total, 'fp_pre': fp_pre, 'fp_real': fp_real
        }
    
    def run_analysis(self):
        if not self.hub_file and not self.m14_file:
            messagebox.showwarning("경고", "CSV 파일을 선택하세요!")
            return
        
        result_text = ""
        
        try:
            if self.hub_file:
                df = pd.read_csv(self.hub_file, encoding='utf-8-sig')
                self.hub_df, self.hub_stats = self.analyze_hubroom(df.copy(), 350)
                result_text += self.format_result("HUBROOM V8.0", self.hub_stats, 350)
            
            if self.m14_file:
                df = pd.read_csv(self.m14_file, encoding='utf-8-sig')
                self.m14_df, self.m14_stats = self.analyze_m14(df.copy(), 1700)
                result_text += self.format_result("M14_Q V8.3.1", self.m14_stats, 1700)
            
            # 결과 표시
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, result_text)
            self.result_text.config(state=tk.DISABLED)
            
            # 다운로드 버튼 추가
            for widget in self.download_frame.winfo_children():
                widget.destroy()
            
            if self.hub_df is not None:
                tk.Button(self.download_frame, text="📥 HUBROOM CSV", command=self.save_hub_csv,
                          font=('맑은 고딕', 10), bg='#00d4ff', fg='black').pack(side=tk.LEFT, padx=5)
            
            if self.m14_df is not None:
                tk.Button(self.download_frame, text="📥 M14_Q CSV", command=self.save_m14_csv,
                          font=('맑은 고딕', 10), bg='#ff6b35', fg='black').pack(side=tk.LEFT, padx=5)
            
            tk.Button(self.download_frame, text="📄 HTML 리포트", command=self.save_html,
                      font=('맑은 고딕', 10, 'bold'), bg='#00ff88', fg='black').pack(side=tk.LEFT, padx=5)
            
            # 자동 저장
            save_dir = filedialog.askdirectory(title="결과 저장할 폴더 선택")
            if save_dir:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                saved_files = []
                
                if self.hub_df is not None:
                    hub_csv = os.path.join(save_dir, f"HUBROOM_결과_{timestamp}.csv")
                    self.hub_df.to_csv(hub_csv, index=False, encoding='utf-8-sig')
                    saved_files.append(hub_csv)
                
                if self.m14_df is not None:
                    m14_csv = os.path.join(save_dir, f"M14_결과_{timestamp}.csv")
                    self.m14_df.to_csv(m14_csv, index=False, encoding='utf-8-sig')
                    saved_files.append(m14_csv)
                
                # HTML 리포트 저장
                html_file = os.path.join(save_dir, f"급증감지_리포트_{timestamp}.html")
                html = self.generate_html()
                with open(html_file, 'w', encoding='utf-8') as f:
                    f.write(html)
                saved_files.append(html_file)
                
                # 브라우저에서 HTML 열기
                webbrowser.open(html_file)
                
                messagebox.showinfo("완료", f"분석 완료!\n\n저장된 파일:\n" + "\n".join(saved_files))
            else:
                messagebox.showinfo("완료", "분석이 완료되었습니다!\n(저장 취소됨)")
            
        except Exception as e:
            messagebox.showerror("오류", f"분석 중 오류 발생:\n{str(e)}")
    
    def format_result(self, name, stats, threshold):
        total = stats['total'] if stats['total'] > 0 else 1
        fp_total = stats['fp_total'] if stats['fp_total'] > 0 else 1
        
        return f"""
{'='*50}
  {name} (기준: {threshold})
{'='*50}

[ 급증 감지 결과 ]
  급증 케이스: {stats['total']}개
  OK: {stats['ok']}개 ({stats['ok']/total*100:.1f}%)
  사전감지: {stats['pre']}개 ({stats['pre']/total*100:.1f}%)
  NG: {stats['ng']}개 ({stats['ng']/total*100:.1f}%)
  총 감지율: {(stats['ok']+stats['pre'])/total*100:.1f}%

[ 오탐 분석 ]
  오탐 후보: {stats['fp_total']}개
  오탐사전감지: {stats['fp_pre']}개 ({stats['fp_pre']/fp_total*100:.1f}%)
  오탐: {stats['fp_real']}개 ({stats['fp_real']/fp_total*100:.1f}%)

"""
    
    def save_hub_csv(self):
        file = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="HUBROOM_결과.csv",
                                             filetypes=[("CSV files", "*.csv")])
        if file:
            self.hub_df.to_csv(file, index=False, encoding='utf-8-sig')
            messagebox.showinfo("저장 완료", f"저장됨: {file}")
    
    def save_m14_csv(self):
        file = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="M14_결과.csv",
                                             filetypes=[("CSV files", "*.csv")])
        if file:
            self.m14_df.to_csv(file, index=False, encoding='utf-8-sig')
            messagebox.showinfo("저장 완료", f"저장됨: {file}")
    
    def save_html(self):
        file = filedialog.asksaveasfilename(
            defaultextension=".html", 
            initialfile=f"급증감지_리포트_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            filetypes=[("HTML files", "*.html")]
        )
        if file:
            html = self.generate_html()
            with open(file, 'w', encoding='utf-8') as f:
                f.write(html)
            messagebox.showinfo("저장 완료", f"저장됨: {file}")
            webbrowser.open(file)
    
    def generate_html(self):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        def calc_rates(stats):
            total = stats['total'] if stats['total'] > 0 else 1
            fp_total = stats['fp_total'] if stats['fp_total'] > 0 else 1
            return {
                'ok_rate': stats['ok'] / total * 100,
                'pre_rate': stats['pre'] / total * 100,
                'ng_rate': stats['ng'] / total * 100,
                'detect_rate': (stats['ok'] + stats['pre']) / total * 100,
                'fp_pre_rate': stats['fp_pre'] / fp_total * 100,
                'fp_real_rate': stats['fp_real'] / fp_total * 100
            }
        
        html = f'''<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>급증 감지 분석 리포트</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: '맑은 고딕', sans-serif; background: #0a0a0f; color: white; padding: 40px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ font-size: 2.5rem; text-align: center; background: linear-gradient(135deg, #00d4ff, #ff6b35); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 10px; }}
        .subtitle {{ text-align: center; color: #888; margin-bottom: 40px; }}
        .section {{ margin-bottom: 50px; }}
        .section-title {{ font-size: 1.5rem; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 2px solid #333; }}
        .section-title.hub {{ color: #00d4ff; border-color: #00d4ff; }}
        .section-title.m14 {{ color: #ff6b35; border-color: #ff6b35; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-bottom: 30px; }}
        .card {{ background: #12121a; padding: 20px; border-radius: 12px; text-align: center; border: 1px solid #2a2a3a; }}
        .card-label {{ font-size: 0.8rem; color: #888; margin-bottom: 8px; }}
        .card-value {{ font-size: 2rem; font-weight: bold; }}
        .card-sub {{ font-size: 0.85rem; color: #888; margin-top: 5px; }}
        .success {{ color: #00ff88; }}
        .warning {{ color: #ffcc00; }}
        .danger {{ color: #ff4466; }}
        .hub-color {{ color: #00d4ff; }}
        .m14-color {{ color: #ff6b35; }}
        .chart-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px; }}
        .chart-box {{ background: #12121a; padding: 20px; border-radius: 12px; border: 1px solid #2a2a3a; }}
        .chart-title {{ font-size: 1rem; margin-bottom: 15px; }}
        .chart-wrapper {{ height: 250px; }}
        .summary {{ background: #12121a; padding: 20px; border-radius: 12px; border: 1px solid #2a2a3a; }}
        .summary pre {{ font-family: Consolas, monospace; white-space: pre-wrap; color: #aaa; line-height: 1.6; }}
        @media (max-width: 768px) {{ .chart-grid {{ grid-template-columns: 1fr; }} }}
    </style>
</head>
<body>
<div class="container">
    <h1>📊 급증 감지 분석 리포트</h1>
    <p class="subtitle">생성: {now}</p>
'''
        
        if self.hub_stats and self.hub_stats['total'] > 0:
            r = calc_rates(self.hub_stats)
            s = self.hub_stats
            html += f'''
    <div class="section">
        <h2 class="section-title hub">🏠 HUBROOM V8.0</h2>
        <div class="grid">
            <div class="card"><div class="card-label">급증 케이스</div><div class="card-value hub-color">{s['total']}</div></div>
            <div class="card"><div class="card-label">총 감지율</div><div class="card-value success">{r['detect_rate']:.1f}%</div></div>
            <div class="card"><div class="card-label">OK</div><div class="card-value success">{s['ok']}</div><div class="card-sub">{r['ok_rate']:.1f}%</div></div>
            <div class="card"><div class="card-label">사전감지</div><div class="card-value warning">{s['pre']}</div><div class="card-sub">{r['pre_rate']:.1f}%</div></div>
            <div class="card"><div class="card-label">NG</div><div class="card-value danger">{s['ng']}</div><div class="card-sub">{r['ng_rate']:.1f}%</div></div>
            <div class="card"><div class="card-label">오탐</div><div class="card-value danger">{s['fp_real']}</div><div class="card-sub">{r['fp_real_rate']:.1f}%</div></div>
        </div>
        <div class="chart-grid">
            <div class="chart-box"><div class="chart-title">📈 감지 분포</div><div class="chart-wrapper"><canvas id="hubPie"></canvas></div></div>
            <div class="chart-box"><div class="chart-title">📉 오탐 분석</div><div class="chart-wrapper"><canvas id="hubFp"></canvas></div></div>
        </div>
        <div class="summary"><pre>=== HUBROOM V8.0 요약 (기준: 350) ===
급증 케이스: {s['total']}개 | OK: {s['ok']}개 ({r['ok_rate']:.1f}%) | 사전감지: {s['pre']}개 ({r['pre_rate']:.1f}%) | NG: {s['ng']}개 ({r['ng_rate']:.1f}%)
총 감지율: {r['detect_rate']:.1f}% | 오탐: {s['fp_real']}개/{s['fp_total']}개 ({r['fp_real_rate']:.1f}%)</pre></div>
    </div>
'''
        
        if self.m14_stats and self.m14_stats['total'] > 0:
            r = calc_rates(self.m14_stats)
            s = self.m14_stats
            html += f'''
    <div class="section">
        <h2 class="section-title m14">🏭 M14_Q V8.3.1</h2>
        <div class="grid">
            <div class="card"><div class="card-label">급증 케이스</div><div class="card-value m14-color">{s['total']}</div></div>
            <div class="card"><div class="card-label">총 감지율</div><div class="card-value success">{r['detect_rate']:.1f}%</div></div>
            <div class="card"><div class="card-label">OK</div><div class="card-value success">{s['ok']}</div><div class="card-sub">{r['ok_rate']:.1f}%</div></div>
            <div class="card"><div class="card-label">사전감지</div><div class="card-value warning">{s['pre']}</div><div class="card-sub">{r['pre_rate']:.1f}%</div></div>
            <div class="card"><div class="card-label">NG</div><div class="card-value danger">{s['ng']}</div><div class="card-sub">{r['ng_rate']:.1f}%</div></div>
            <div class="card"><div class="card-label">오탐</div><div class="card-value danger">{s['fp_real']}</div><div class="card-sub">{r['fp_real_rate']:.1f}%</div></div>
        </div>
        <div class="chart-grid">
            <div class="chart-box"><div class="chart-title">📈 감지 분포</div><div class="chart-wrapper"><canvas id="m14Pie"></canvas></div></div>
            <div class="chart-box"><div class="chart-title">📉 오탐 분석</div><div class="chart-wrapper"><canvas id="m14Fp"></canvas></div></div>
        </div>
        <div class="summary"><pre>=== M14_Q V8.3.1 요약 (기준: 1700) ===
급증 케이스: {s['total']}개 | OK: {s['ok']}개 ({r['ok_rate']:.1f}%) | 사전감지: {s['pre']}개 ({r['pre_rate']:.1f}%) | NG: {s['ng']}개 ({r['ng_rate']:.1f}%)
총 감지율: {r['detect_rate']:.1f}% | 오탐: {s['fp_real']}개/{s['fp_total']}개 ({r['fp_real_rate']:.1f}%)</pre></div>
    </div>
'''
        
        html += '''
</div>
<script>
const opts = { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom', labels: { color: '#fff' } } } };
'''
        
        if self.hub_stats and self.hub_stats['total'] > 0:
            s = self.hub_stats
            html += f'''
new Chart(document.getElementById('hubPie'), {{ type: 'doughnut', data: {{ labels: ['OK','사전감지','NG'], datasets: [{{ data: [{s['ok']},{s['pre']},{s['ng']}], backgroundColor: ['#00ff88','#ffcc00','#ff4466'] }}] }}, options: opts }});
new Chart(document.getElementById('hubFp'), {{ type: 'doughnut', data: {{ labels: ['오탐사전감지','오탐'], datasets: [{{ data: [{s['fp_pre']},{s['fp_real']}], backgroundColor: ['#ffcc00','#ff4466'] }}] }}, options: opts }});
'''
        
        if self.m14_stats and self.m14_stats['total'] > 0:
            s = self.m14_stats
            html += f'''
new Chart(document.getElementById('m14Pie'), {{ type: 'doughnut', data: {{ labels: ['OK','사전감지','NG'], datasets: [{{ data: [{s['ok']},{s['pre']},{s['ng']}], backgroundColor: ['#00ff88','#ffcc00','#ff4466'] }}] }}, options: opts }});
new Chart(document.getElementById('m14Fp'), {{ type: 'doughnut', data: {{ labels: ['오탐사전감지','오탐'], datasets: [{{ data: [{s['fp_pre']},{s['fp_real']}], backgroundColor: ['#ffcc00','#ff4466'] }}] }}, options: opts }});
'''
        
        html += '''
</script>
</body>
</html>'''
        
        return html


if __name__ == "__main__":
    root = tk.Tk()
    app = SurgeAnalyzer(root)
    root.mainloop()