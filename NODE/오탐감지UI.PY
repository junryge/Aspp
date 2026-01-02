# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 2025
V8.1 - 분류 체계 단순화
- 정상예측 → 이상(정상(-N분))으로 이상탐지에 포함
- 급증연속 → 정상(급증(-N분))으로 정상에 포함
"""
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
from datetime import datetime
import webbrowser
import os

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

class SurgeAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("ML급증 감지 분석 시스템 V8.2 (알람기반)")
        self.root.geometry("1200x900")
        self.root.configure(bg='#1a1a2e')
        
        self.hub_file = None
        self.m14_file = None
        self.hub_alarm_file = None
        self.m14_alarm_file = None
        self.hub_df = None
        self.m14_df = None
        self.hub_alarm_df = None
        self.m14_alarm_df = None
        self.hub_stats = None
        self.m14_stats = None
        
        self.setup_ui()
    
    def setup_ui(self):
        main_frame = tk.Frame(self.root, bg='#1a1a2e')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)
        
        top_frame = tk.Frame(main_frame, bg='#1a1a2e')
        top_frame.pack(fill=tk.X)
        
        # 타이틀
        tk.Label(top_frame, text="📊 ML급증 감지 분석 시스템 V8.2", font=('맑은 고딕', 20, 'bold'),
                 bg='#1a1a2e', fg='#00d4ff').pack(pady=(0, 3))
        tk.Label(top_frame, text="알람 기반: 이상탐지 / 오탐 / 미탐 / 정상", font=('맑은 고딕', 10),
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
        
        hub_alarm_frame = tk.Frame(hub_frame, bg='#252540')
        hub_alarm_frame.pack(fill=tk.X, padx=10, pady=2)
        tk.Button(hub_alarm_frame, text="🚨 알람", command=self.select_hub_alarm_file,
                  font=('맑은 고딕', 9), bg='#ffcc00', fg='black', width=8).pack(side=tk.LEFT)
        self.hub_alarm_label = tk.Label(hub_alarm_frame, text="(선택)", font=('맑은 고딕', 8), bg='#252540', fg='#666')
        self.hub_alarm_label.pack(side=tk.LEFT, padx=5)
        
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
        
        m14_alarm_frame = tk.Frame(m14_frame, bg='#252540')
        m14_alarm_frame.pack(fill=tk.X, padx=10, pady=2)
        tk.Button(m14_alarm_frame, text="🚨 알람", command=self.select_m14_alarm_file,
                  font=('맑은 고딕', 9), bg='#ffcc00', fg='black', width=8).pack(side=tk.LEFT)
        self.m14_alarm_label = tk.Label(m14_alarm_frame, text="(선택)", font=('맑은 고딕', 8), bg='#252540', fg='#666')
        self.m14_alarm_label.pack(side=tk.LEFT, padx=5)
        
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
        
        # 이벤트 뷰어 탭 추가
        self.event_frame = tk.Frame(self.notebook, bg='#0d0d15')
        self.notebook.add(self.event_frame, text='📈 이벤트 그래프')
        self.setup_event_viewer()
    
    def read_csv_auto(self, filepath):
        """여러 인코딩 시도해서 CSV 읽기"""
        encodings = ['utf-8-sig', 'utf-8', 'cp949', 'euc-kr']
        for enc in encodings:
            try:
                return pd.read_csv(filepath, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"파일 인코딩을 알 수 없습니다: {filepath}")
    
    def setup_event_viewer(self):
        """이벤트 뷰어 탭 설정"""
        # 상단 컨트롤
        ctrl_frame = tk.Frame(self.event_frame, bg='#252540')
        ctrl_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(ctrl_frame, text="시스템:", font=('맑은 고딕', 10, 'bold'),
                 bg='#252540', fg='white').pack(side=tk.LEFT, padx=(10,5))
        self.ev_system = ttk.Combobox(ctrl_frame, values=['HUBROOM', 'M14_Q'], width=12, state='readonly')
        self.ev_system.set('HUBROOM')
        self.ev_system.pack(side=tk.LEFT, padx=5)
        self.ev_system.bind('<<ComboboxSelected>>', lambda e: self.refresh_event_list())
        
        tk.Label(ctrl_frame, text="분류:", font=('맑은 고딕', 10, 'bold'),
                 bg='#252540', fg='white').pack(side=tk.LEFT, padx=(20,5))
        self.ev_class = ttk.Combobox(ctrl_frame, values=[
            '전체', '이상탐지', '미탐', '정상', '오탐'
        ], width=15, state='readonly')
        self.ev_class.set('이상탐지')
        self.ev_class.pack(side=tk.LEFT, padx=5)
        self.ev_class.bind('<<ComboboxSelected>>', lambda e: self.refresh_event_list())
        
        tk.Button(ctrl_frame, text="🔄 새로고침", command=self.refresh_event_list,
                  font=('맑은 고딕', 9), bg='#00d4ff', fg='black').pack(side=tk.LEFT, padx=20)
        
        self.ev_count_label = tk.Label(ctrl_frame, text="이벤트: 0개", font=('맑은 고딕', 10),
                                        bg='#252540', fg='#00ff88')
        self.ev_count_label.pack(side=tk.RIGHT, padx=10)
        
        # 메인 영역 (리스트 + 그래프)
        main_pane = tk.PanedWindow(self.event_frame, orient=tk.HORIZONTAL, bg='#1a1a2e', sashwidth=5)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0,10))
        
        # 왼쪽: 이벤트 리스트
        list_frame = tk.Frame(main_pane, bg='#0d0d15')
        main_pane.add(list_frame, width=400)
        
        tk.Label(list_frame, text="📋 이벤트 목록 (클릭 → 그래프)", font=('맑은 고딕', 10, 'bold'),
                 bg='#0d0d15', fg='#ffcc00').pack(pady=(10,5))
        
        # 리스트박스
        list_container = tk.Frame(list_frame, bg='#0d0d15')
        list_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        y_scroll = ttk.Scrollbar(list_container, orient=tk.VERTICAL)
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.event_listbox = tk.Listbox(list_container, font=('Consolas', 9), bg='#15151f', fg='white',
                                         selectmode=tk.SINGLE, activestyle='none',
                                         yscrollcommand=y_scroll.set, selectbackground='#00d4ff',
                                         selectforeground='black')
        self.event_listbox.pack(fill=tk.BOTH, expand=True)
        y_scroll.config(command=self.event_listbox.yview)
        self.event_listbox.bind('<<ListboxSelect>>', self.on_event_select)
        
        # 오른쪽: 그래프
        graph_frame = tk.Frame(main_pane, bg='#0d0d15')
        main_pane.add(graph_frame, width=700)
        
        self.graph_title = tk.Label(graph_frame, text="📈 이벤트 선택 시 그래프 표시", 
                                     font=('맑은 고딕', 11, 'bold'), bg='#0d0d15', fg='#00d4ff')
        self.graph_title.pack(pady=(10,5))
        
        # 그래프 정보
        self.graph_info = tk.Label(graph_frame, text="", font=('맑은 고딕', 9), bg='#0d0d15', fg='#888')
        self.graph_info.pack()
        
        # matplotlib 캔버스
        self.graph_container = tk.Frame(graph_frame, bg='#0d0d15')
        self.graph_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        if HAS_MATPLOTLIB:
            plt.style.use('dark_background')
            self.fig = Figure(figsize=(8, 5), dpi=100, facecolor='#0d0d15')
            self.ax = self.fig.add_subplot(111)
            self.ax.set_facecolor('#15151f')
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_container)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            tk.Label(self.graph_container, text="matplotlib 설치 필요\npip install matplotlib",
                     font=('맑은 고딕', 12), bg='#0d0d15', fg='#ff4466').pack(pady=50)
        
        self.event_data = []  # 현재 표시중인 이벤트 리스트
    
    def refresh_event_list(self):
        """이벤트 리스트 새로고침"""
        self.event_listbox.delete(0, tk.END)
        self.event_data = []
        
        system = self.ev_system.get()
        cls_filter = self.ev_class.get()
        
        df = self.hub_df if system == 'HUBROOM' else self.m14_df
        
        if df is None or len(df) == 0 or '분류' not in df.columns:
            self.ev_count_label.config(text="이벤트: 0개")
            return
        
        # 필터링 - includes 사용 + 쿨다운 제외
        if cls_filter == '전체':
            filtered = df
        elif cls_filter == '이상탐지':
            filtered = df[df['분류'].str.contains('이상', na=False) & ~df['분류'].str.contains('쿨다운', na=False)]
        elif cls_filter == '미탐':
            filtered = df[df['분류'].str.contains('미탐', na=False) & ~df['분류'].str.contains('쿨다운', na=False)]
        elif cls_filter == '정상':
            filtered = df[df['분류'].str.contains('정상', na=False) & ~df['분류'].str.contains('이상', na=False)]
        elif cls_filter == '오탐':
            filtered = df[df['분류'] == '오탐']
        else:
            filtered = df[df['분류'].str.contains(cls_filter, na=False)]
        
        self.ev_count_label.config(text=f"이벤트: {len(filtered)}개")
        
        # 리스트에 추가
        for i, (idx, row) in enumerate(filtered.iterrows()):
            time_str = str(row.get('현재시간', ''))[-11:]
            pred_time = str(row.get('예측시점', ''))[-11:]
            actual = row.get('실제값', 0)
            pred = row.get('예측값_MAX', 0)
            cls = str(row.get('분류', ''))[:12]
            
            display = f"{i+1:4d} | {time_str} → {pred_time} | 실:{actual:6.0f} 예:{pred:6.0f} | {cls}"
            self.event_listbox.insert(tk.END, display)
            self.event_data.append(idx)
    
    def on_event_select(self, event):
        """이벤트 선택 시 그래프 표시"""
        if not HAS_MATPLOTLIB:
            return
        
        selection = self.event_listbox.curselection()
        if not selection:
            return
        
        list_idx = selection[0]
        if list_idx >= len(self.event_data):
            return
        
        df_idx = self.event_data[list_idx]
        system = self.ev_system.get()
        df = self.hub_df if system == 'HUBROOM' else self.m14_df
        threshold = 350 if system == 'HUBROOM' else 1700
        
        if df is None:
            return
        
        row = df.loc[df_idx]
        center_time = str(row.get('현재시간', ''))
        cls = str(row.get('분류', ''))
        actual_val = row.get('실제값', 0)
        pred_val = row.get('예측값_MAX', 0)
        
        row_pos = df.index.get_loc(df_idx)
        start_pos = max(0, row_pos - 30)
        end_pos = min(len(df), row_pos + 31)
        
        range_df = df.iloc[start_pos:end_pos].copy()
        
        self.ax.clear()
        self.ax.set_facecolor('#15151f')
        
        x_labels = [str(t)[-8:] for t in range_df['현재시간'].values]
        x_pos = list(range(len(range_df)))
        
        # 예측값 포인트 색상/크기 설정 (알람 상태에 따라)
        pred_colors = []
        pred_sizes = []
        alarm_count = 0
        cooldown_count = 0
        miss_count = 0
        
        for i, (idx, r) in enumerate(range_df.iterrows()):
            if '알람' in range_df.columns and pd.notna(r.get('알람')) and str(r.get('알람', '')).strip():
                pred_colors.append('#ff4466')  # 알람: 빨간색
                pred_sizes.append(80)
                alarm_count += 1
            elif '알람쿨다운' in range_df.columns and pd.notna(r.get('알람쿨다운')) and str(r.get('알람쿨다운', '')).strip():
                pred_colors.append('#4488ff')  # 쿨다운: 파란색
                pred_sizes.append(30)
                cooldown_count += 1
            elif '알람미발생' in range_df.columns and pd.notna(r.get('알람미발생')) and str(r.get('알람미발생', '')).strip():
                pred_colors.append('#aa44ff')  # 미발생: 보라색
                pred_sizes.append(80)
                miss_count += 1
            else:
                pred_colors.append('#ff6b35')  # 기본: 주황색
                pred_sizes.append(25)
        
        # 실제값 라인
        self.ax.plot(x_pos, range_df['실제값'].values, 'o-', color='#00d4ff', 
                     linewidth=2, markersize=6, label='실제값')
        
        # 예측값 라인 (기본)
        self.ax.plot(x_pos, range_df['예측값_MAX'].values, '--', color='#ff6b35', 
                     linewidth=2, label='예측값')
        
        # 예측값 포인트 (알람 색상 적용)
        self.ax.scatter(x_pos, range_df['예측값_MAX'].values, c=pred_colors, s=pred_sizes, 
                        marker='s', zorder=4, edgecolors='white', linewidths=0.5)
        
        self.ax.axhline(y=threshold, color='#ff4466', linestyle=':', linewidth=2, label=f'리미트값 ({threshold})')
        
        center_pos = row_pos - start_pos
        self.ax.axvline(x=center_pos, color='#ffcc00', linestyle='-', linewidth=2, alpha=0.5)
        self.ax.scatter([center_pos], [actual_val], color='#ffcc00', s=150, zorder=5, marker='*')
        
        self.ax.set_xticks(x_pos)
        self.ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
        self.ax.set_xlabel('시간', fontsize=10, color='white')
        self.ax.set_ylabel('값', fontsize=10, color='white')
        self.ax.legend(loc='upper right', fontsize=9)
        self.ax.grid(True, alpha=0.3, color='#333')
        self.ax.tick_params(colors='white')
        
        self.fig.tight_layout()
        self.canvas.draw()
        
        # 상태 텍스트
        status_text = ""
        if alarm_count > 0:
            status_text += f" | 🔴알람 {alarm_count}"
        if cooldown_count > 0:
            status_text += f" | 🔵쿨다운 {cooldown_count}"
        if miss_count > 0:
            status_text += f" | 🟣미발생 {miss_count}"
        
        self.graph_title.config(text=f"📈 {system} - {cls}")
        self.graph_info.config(text=f"시점: {center_time} | 실제값: {actual_val:.0f} | 예측값: {pred_val:.1f} | ±30분 범위{status_text}")
    
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
                if '이상' in 분류:
                    tag = 'detect'
                elif '미탐' in 분류:
                    tag = 'miss'
                elif '정상' in 분류:
                    tag = 'normal'
                elif '오탐' in 분류:
                    tag = 'false'
            tree.insert('', tk.END, values=values, tags=(tag,))
        
        tree.tag_configure('detect', background='#1a3a1a')
        tree.tag_configure('miss', background='#3a1a1a')
        tree.tag_configure('normal', background='#1a1a2a')
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
    
    def select_hub_alarm_file(self):
        file = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if file:
            self.hub_alarm_file = file
            self.hub_alarm_label.config(text=f"✅ {os.path.basename(file)[:20]}", fg='#ffcc00')
    
    def select_m14_alarm_file(self):
        file = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if file:
            self.m14_alarm_file = file
            self.m14_alarm_label.config(text=f"✅ {os.path.basename(file)[:20]}", fg='#ffcc00')
    
    def load_alarm_data(self, alarm_file):
        """알람 CSV 로드 및 시간 형식 변환"""
        if not alarm_file:
            return None
        try:
            df = self.read_csv_auto(alarm_file)
            if 'MEAS_TM' in df.columns:
                # MEAS_TM을 문자열로 변환 (202512291354 형식)
                df['MEAS_TM'] = df['MEAS_TM'].astype(str)
            if 'LSTM_FCAST_TM' in df.columns:
                # LSTM_FCAST_TM을 문자열로 변환 (예측시점과 매칭용)
                df['LSTM_FCAST_TM'] = df['LSTM_FCAST_TM'].astype(str)
            return df
        except Exception as e:
            print(f"알람 파일 로드 실패: {e}")
            return None
    
    def match_alarm(self, df, alarm_df, pred_time_col='예측시점', pred_col='예측값_MAX', threshold=1700):
        """데이터프레임에 알람 매칭 (최초 1개 + 60분 쿨다운 방식)
        - 알람: 최초 알람 발생 (60분 내 재발생은 쿨다운)
        - 알람쿨다운: 알람 발생 후 60분 이내
        - 알람미발생: 예측>=threshold인데 알람도 쿨다운도 아님 (전부 표시)
        """
        df = df.copy()
        df = df.reset_index(drop=True)  # 인덱스 0부터 리셋!
        df['알람'] = ''
        df['알람쿨다운'] = ''
        df['알람미발생'] = ''
        df['알람미발생쿨다운'] = ''
        
        # 예측 컬럼 확인 (예측값_MAX 또는 최종예측)
        actual_pred_col = None
        if pred_col in df.columns:
            actual_pred_col = pred_col
        elif '예측값_MAX' in df.columns:
            actual_pred_col = '예측값_MAX'
        elif '최종예측' in df.columns:
            actual_pred_col = '최종예측'
        
        if actual_pred_col is None:
            print(f"경고: 예측 컬럼을 찾을 수 없습니다. 컬럼 목록: {df.columns.tolist()}")
            return df
        
        if alarm_df is None or len(alarm_df) == 0:
            # 알람 데이터 없으면 예측값>=threshold인 곳 전부 미발생 처리
            for idx in range(len(df)):
                pred_val = df.loc[idx, actual_pred_col]
                if pred_val >= threshold:
                    df.loc[idx, '알람미발생'] = '알람데이터없음'
            return df
        
        # 1단계: 알람 발생 시점 찾기 (최초만 기록, 60분 내 재발생은 쿨다운)
        last_alarm_idx = -9999  # 마지막 최초 알람 인덱스 (아직 없음)
        last_alarm_at = {}  # 각 idx 시점에서의 last_alarm_idx 저장
        
        for idx in range(len(df)):
            pred_time_str = str(df.loc[idx, pred_time_col])
            time_key = pred_time_str.replace('-', '').replace(':', '').replace(' ', '')[:12]
            
            if 'LSTM_FCAST_TM' in alarm_df.columns:
                matched = alarm_df[alarm_df['LSTM_FCAST_TM'].str[:12] == time_key]
            else:
                matched = alarm_df[alarm_df['MEAS_TM'].str[:12] == time_key]
            
            if len(matched) > 0:
                # 알람이 있는 시점
                alarm_texts = []
                for _, row in matched.head(2).iterrows():
                    meas_tm = str(row.get('MEAS_TM', ''))
                    if len(meas_tm) >= 12:
                        meas_formatted = f"{meas_tm[4:6]}-{meas_tm[6:8]} {meas_tm[8:10]}:{meas_tm[10:12]}"
                    else:
                        meas_formatted = meas_tm
                    alarm_texts.append(f"[{meas_formatted}] Y 예측값{threshold}초과")
                alarm_text = ' / '.join(alarm_texts)
                
                if last_alarm_idx < 0 or idx - last_alarm_idx > 60:
                    # 60분 지났으니 새 알람 (최초)
                    df.loc[idx, '알람'] = alarm_text
                    last_alarm_idx = idx
                else:
                    # 60분 안이니 쿨다운 - 몇 분 전 알람인지 표시
                    minutes_ago = idx - last_alarm_idx
                    df.loc[idx, '알람쿨다운'] = f'{minutes_ago}분전'
            
            # 현재 idx 시점의 last_alarm_idx 저장
            last_alarm_at[idx] = last_alarm_idx
        
        # 2단계: 예측값>=threshold인데 알람 아닌 곳 처리
        # - 60분 이내: 알람쿨다운 (알람 CSV에 없어도!)
        # - 60분 초과: 알람미발생쿨다운
        for idx in range(len(df)):
            # 이미 알람이면 스킵
            if df.loc[idx, '알람'] != '':
                continue
            
            # 예측값이 threshold 미만이면 스킵
            pred_val = df.loc[idx, actual_pred_col]
            if pred_val < threshold:
                continue
            
            # 해당 시점까지의 마지막 알람 인덱스
            my_last_alarm = last_alarm_at.get(idx, -9999)
            
            # 예측값 >= threshold인데...
            if my_last_alarm < 0:
                # 아직 알람 발생 전 → 알람미발생 (큰 크기)
                df.loc[idx, '알람미발생'] = '이전알람없음'
            else:
                minutes_ago = idx - my_last_alarm
                if minutes_ago <= 60:
                    # 60분 이내 → 알람쿨다운
                    if df.loc[idx, '알람쿨다운'] == '':
                        df.loc[idx, '알람쿨다운'] = f'{minutes_ago}분전'
                else:
                    # 60분 초과 → 알람미발생쿨다운
                    df.loc[idx, '알람미발생쿨다운'] = f'{minutes_ago}분전'
        
        return df
    
    def analyze_alarm_based(self, df, alarm_df, threshold, pred_col, curr_col, actual_col, pred_time_col):
        """
        V8.2 알람 기반 분류 로직
        1) 실제값 >= threshold + 알람발생 → 이상탐지
        2) 실제값 >= threshold + 알람미발생 → 미탐
        3) 실제값 < threshold + 알람발생 → 오탐
        4) 실제값 < threshold + 알람미발생 → 정상
        """
        df = df.copy()
        df = df.reset_index(drop=True)
        
        # 불필요 컬럼 제거
        drop_cols = ['Unnamed: 22', 'Unnamed: 23', '12분전TOTALCNT', '12변화량']
        for col in drop_cols:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        # 알람 컬럼 초기화
        df['알람'] = ''
        df['알람쿨다운'] = ''
        df['알람미발생'] = ''
        df['알람미발생쿨다운'] = ''
        df['분류'] = ''
        
        # 알람 매칭 (60분 쿨다운) - 예측값 >= threshold인 경우에만 표시!
        last_alarm_idx = -9999
        
        if alarm_df is not None and len(alarm_df) > 0:
            for idx in range(len(df)):
                pred = df.loc[idx, pred_col]
                if pred < threshold:
                    continue  # 예측값 < threshold면 알람 표시 안함
                
                pred_time_str = str(df.loc[idx, pred_time_col])
                time_key = pred_time_str.replace('-', '').replace(':', '').replace(' ', '')[:12]
                
                if 'LSTM_FCAST_TM' in alarm_df.columns:
                    matched = alarm_df[alarm_df['LSTM_FCAST_TM'].str[:12] == time_key]
                else:
                    matched = alarm_df[alarm_df['MEAS_TM'].str[:12] == time_key]
                
                if len(matched) > 0:
                    alarm_texts = []
                    for _, row in matched.head(2).iterrows():
                        meas_tm = str(row.get('MEAS_TM', ''))
                        if len(meas_tm) >= 12:
                            meas_formatted = f"{meas_tm[4:6]}-{meas_tm[6:8]} {meas_tm[8:10]}:{meas_tm[10:12]}"
                        else:
                            meas_formatted = meas_tm
                        alarm_texts.append(f"[{meas_formatted}]")
                    alarm_text = ' / '.join(alarm_texts)
                    
                    if last_alarm_idx < 0 or idx - last_alarm_idx > 60:
                        df.loc[idx, '알람'] = alarm_text
                        last_alarm_idx = idx
                    else:
                        minutes_ago = idx - last_alarm_idx
                        df.loc[idx, '알람쿨다운'] = f'{minutes_ago}분전'
        
        # 알람미발생 처리 (예측값 >= threshold인데 알람도 쿨다운도 없는 경우) + 60분 쿨다운
        # 단, 알람 발생 후 60분간은 알람미발생 체크 안함!
        last_miss_idx = -9999
        for idx in range(len(df)):
            # 이 행에 알람이나 쿨다운 있으면 스킵
            if df.loc[idx, '알람'] or df.loc[idx, '알람쿨다운']:
                continue
            
            pred = df.loc[idx, pred_col]
            if pred < threshold:
                continue
            
            # 최근 60분 내 알람 발생했으면 → 알람쿨다운으로 표시!
            alarm_cooldown_back = 0
            for back in range(1, 61):
                if idx - back >= 0 and df.loc[idx - back, '알람']:
                    alarm_cooldown_back = back
                    break
            
            if alarm_cooldown_back > 0:
                # 알람 쿨다운 기간이라 알람 안 울림 → 알람쿨다운 표시
                df.loc[idx, '알람쿨다운'] = f'쿨다운중({alarm_cooldown_back}분전)'
            else:
                # 60분 내 알람 없음 → 알람미발생
                if last_miss_idx < 0 or idx - last_miss_idx > 60:
                    df.loc[idx, '알람미발생'] = '미발생'
                    last_miss_idx = idx
                else:
                    minutes_ago = idx - last_miss_idx
                    df.loc[idx, '알람미발생쿨다운'] = f'{minutes_ago}분전'
        
        # 분류 (쿨다운 구분) - 60분 이내 알람/알람미발생 있으면 쿨다운
        for idx in range(len(df)):
            actual = df.loc[idx, actual_col]
            
            # 현재 행의 상태
            has_alarm = df.loc[idx, '알람'] != ''
            has_alarm_cd = df.loc[idx, '알람쿨다운'] != ''
            has_miss = df.loc[idx, '알람미발생'] != ''
            has_miss_cd = df.loc[idx, '알람미발생쿨다운'] != ''
            
            # 60분 이내 알람/알람미발생 있었는지 체크
            in_alarm_cooldown = False
            in_miss_cooldown = False
            for back in range(1, 61):
                if idx - back >= 0:
                    if df.loc[idx - back, '알람'] != '':
                        in_alarm_cooldown = True
                        break
            for back in range(1, 61):
                if idx - back >= 0:
                    if df.loc[idx - back, '알람미발생'] != '':
                        in_miss_cooldown = True
                        break
            
            if actual >= threshold:
                # 급증 케이스
                if has_alarm:
                    df.loc[idx, '분류'] = '이상탐지'
                elif has_alarm_cd or in_alarm_cooldown:
                    df.loc[idx, '분류'] = '이상탐지(쿨다운)'
                elif has_miss:
                    df.loc[idx, '분류'] = '미탐'
                elif has_miss_cd or in_miss_cooldown:
                    df.loc[idx, '분류'] = '미탐(쿨다운)'
                else:
                    df.loc[idx, '분류'] = '미탐'
            else:
                # 정상 케이스
                if has_alarm:
                    df.loc[idx, '분류'] = '오탐'
                elif has_alarm_cd or in_alarm_cooldown:
                    df.loc[idx, '분류'] = '오탐(쿨다운)'
                else:
                    df.loc[idx, '분류'] = '정상'
        
        # 통계 (쿨다운 제외한 순수 개수)
        total = len(df)
        surge_total = len(df[df[actual_col] >= threshold])  # 급증 건수 전체
        detect = len(df[df['분류'] == '이상탐지'])  # 쿨다운 제외
        miss = len(df[df['분류'] == '미탐'])  # 쿨다운 제외
        normal = len(df[df['분류'] == '정상'])
        false_alarm = len(df[df['분류'] == '오탐'])  # 쿨다운 제외
        alarm_total = detect + false_alarm  # 쿨다운 제외한 알람발생
        surge_base = detect + miss  # 쿨다운 제외한 급증 케이스 (탐지율/미탐율 기준)
        
        stats = {
            'total_rows': total,
            'surge_total': surge_total,
            'detect': detect,
            'miss': miss,
            'normal': normal,
            'false': false_alarm,
            'alarm_total': alarm_total,  # 오탐율 계산용
            'surge_base': surge_base     # 탐지율/미탐율 계산용
        }
        
        # 컬럼 순서 정리
        cols = df.columns.tolist()
        if '분류' in cols:
            cols.remove('분류')
            actual_idx = cols.index(actual_col)
            df = df[cols[:actual_idx+1] + ['분류'] + cols[actual_idx+1:]]
        
        return df, stats
    
    def analyze_v81(self, df, threshold, pred_col, curr_col, actual_col):
        """
        V8.1 분석 로직 (단순화)
        급증(실제>=threshold):
          - 현재 예측>=threshold → 이상탐지
          - 과거 10분내 예측>=threshold → 이상(정상(-N분))
          - 그 외 → 미탐 또는 미탐(N분후)
        정상(실제<threshold):
          - 예측<threshold → 정상
          - 예측>=threshold + 과거10분내 급증 → 정상(급증(-N분))
          - 예측>=threshold + 과거 급증 없음 → 오탐
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
                    # 1) 과거 10분내 예측 >= threshold → 이상(정상(-N분))
                    found_past = False
                    for back in range(1, 11):
                        if idx - back >= 0 and df.loc[idx - back, pred_col] >= threshold:
                            df.loc[idx, '분류'] = f'이상(정상(-{back}분))'
                            found_past = True
                            break
                    
                    if not found_past:
                        # 2) 미래 10분내 예측 >= threshold → 미탐(N분후)
                        found_future = False
                        for fwd in range(1, 11):
                            if idx + fwd < len(df) and df.loc[idx + fwd, pred_col] >= threshold:
                                df.loc[idx, '분류'] = f'미탐({fwd}분후)'
                                found_future = True
                                break
                        
                        if not found_future:
                            df.loc[idx, '분류'] = '미탐'
            else:
                # 정상 케이스 (실제값 < threshold)
                if pred >= threshold:
                    # 10분 전에 급증 있는지 체크 → 정상(급증(-N분))
                    found_past = False
                    for back in range(1, 11):
                        if idx - back >= 0 and df.loc[idx - back, actual_col] >= threshold:
                            found_past = True
                            df.loc[idx, '분류'] = f'정상(급증(-{back}분))'
                            break
                    
                    if not found_past:
                        # 과거 급증 없으면 오탐
                        df.loc[idx, '분류'] = '오탐'
                else:
                    df.loc[idx, '분류'] = '정상'
        
        # 통계 - 단순화된 4개 분류
        total = len(df)
        surge_total = len(df[df[actual_col] >= threshold])
        
        # 이상탐지 = 이상탐지 + 이상(정상(-N분))
        detect = len(df[df['분류'].str.contains('이상', na=False)])
        # 미탐 = 미탐 + 미탐(N분후)
        miss = len(df[df['분류'].str.contains('미탐', na=False)])
        # 정상 = 정상 + 정상(급증(-N분))
        normal = len(df[df['분류'].str.contains('정상', na=False) & ~df['분류'].str.contains('이상', na=False)])
        # 오탐
        false_alarm = len(df[df['분류'] == '오탐'])
        
        surge_base = detect + miss  # 탐지율/미탐율 기준
        alarm_total = detect + false_alarm  # 오탐율 기준
        
        stats = {
            'total_rows': total,
            'surge_total': surge_total,
            'detect': detect,      # 이상탐지 전체
            'miss': miss,          # 미탐 전체
            'normal': normal,      # 정상 전체
            'false': false_alarm,  # 오탐
            'surge_base': surge_base,
            'alarm_total': alarm_total
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
            # 알람 데이터 로드
            self.hub_alarm_df = self.load_alarm_data(self.hub_alarm_file)
            self.m14_alarm_df = self.load_alarm_data(self.m14_alarm_file)
            
            if self.hub_file:
                df = self.read_csv_auto(self.hub_file)
                # 알람 파일 있으면 알람 기반, 없으면 예측값 기준
                if self.hub_alarm_df is not None and len(self.hub_alarm_df) > 0:
                    self.hub_df, self.hub_stats = self.analyze_alarm_based(
                        df.copy(), self.hub_alarm_df, 350, '예측값_MAX', '현재값', '실제값', '예측시점')
                    result_text += self.fmt("HUBROOM V8.2 (알람기반)", self.hub_stats, 350)
                else:
                    self.hub_df, self.hub_stats = self.analyze_v81(
                        df.copy(), 350, '예측값_MAX', '현재값', '실제값')
                    result_text += self.fmt_pred("HUBROOM V8.1 (예측기반)", self.hub_stats, 350)
                self.create_data_view(self.hub_data_frame, self.hub_df)
            
            if self.m14_file:
                df = self.read_csv_auto(self.m14_file)
                # 알람 파일 있으면 알람 기반, 없으면 예측값 기준
                if self.m14_alarm_df is not None and len(self.m14_alarm_df) > 0:
                    self.m14_df, self.m14_stats = self.analyze_alarm_based(
                        df.copy(), self.m14_alarm_df, 1700, '최종예측', '현재TOTALCNT', '실제값', '예측시점')
                    result_text += self.fmt("M14_Q V8.2 (알람기반)", self.m14_stats, 1700)
                else:
                    self.m14_df, self.m14_stats = self.analyze_v81(
                        df.copy(), 1700, '최종예측', '현재TOTALCNT', '실제값')
                    result_text += self.fmt_pred("M14_Q V8.1 (예측기반)", self.m14_stats, 1700)
                self.create_data_view(self.m14_data_frame, self.m14_df)
            
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, result_text)
            self.result_text.config(state=tk.DISABLED)
            
            self.refresh_event_list()
            
            save_dir = filedialog.askdirectory(title="결과 저장할 폴더 선택")
            if save_dir:
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                files = []
                
                if self.hub_df is not None:
                    f = os.path.join(save_dir, f"HUB수치형_V8.2_평가결과_{ts}.csv")
                    self.hub_df.to_csv(f, index=False, encoding='utf-8-sig')
                    files.append(f)
                
                if self.m14_df is not None:
                    f = os.path.join(save_dir, f"M14수치형_V8.2_평가결과_{ts}.csv")
                    self.m14_df.to_csv(f, index=False, encoding='utf-8-sig')
                    files.append(f)
                
                html_f = os.path.join(save_dir, f"ML급증감지_V8.2_리포트_{ts}.html")
                with open(html_f, 'w', encoding='utf-8') as f:
                    f.write(self.gen_html(hub_cmt, m14_cmt, anal_date))
                files.append(html_f)
                
                webbrowser.open(html_f)
                messagebox.showinfo("완료", "저장 완료!\n\n" + "\n".join([os.path.basename(x) for x in files]))
            
            self.refresh_event_list()
            
        except Exception as e:
            import traceback
            messagebox.showerror("오류", f"{str(e)}\n\n{traceback.format_exc()}")
    
    def fmt(self, name, s, th):
        """알람 기반 포맷 - 오탐율 = 오탐/전체알람발생"""
        t = s['total_rows'] or 1
        surge = s['surge_total'] or 1
        alarm_total = s.get('alarm_total', 1) or 1
        surge_base = s.get('surge_base', 1) or 1  # 이상탐지 + 미탐
        
        return f"""
{'='*60}
  {name} (기준: {th})
{'='*60}
전체: {s['total_rows']}개 | 급증: {s['surge_total']}개 | 알람발생: {s.get('alarm_total', 0)}개

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ 이상탐지: {s['detect']}개 (탐지율: {s['detect']/surge_base*100:.1f}%)
❌ 오탐: {s['false']}개 (오탐율: {s['false']/alarm_total*100:.1f}%)
❌ 미탐: {s['miss']}개 (미탐율: {s['miss']/surge_base*100:.1f}%)
✅ 정상: {s['normal']}개
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    def fmt_pred(self, name, s, th):
        """예측값 기준 포맷"""
        t = s['total_rows'] or 1
        surge = s['surge_total'] or 1
        surge_base = s.get('surge_base', 1) or 1
        alarm_total = s.get('alarm_total', 1) or 1
        
        return f"""
{'='*60}
  {name} (기준: {th})
{'='*60}
전체: {s['total_rows']}개 | 급증: {s['surge_total']}개 | 정상: {t - surge}개

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ 이상탐지: {s['detect']}개 (탐지율: {s['detect']/surge_base*100:.1f}%)
❌ 오탐: {s['false']}개 (오탐율: {s['false']/alarm_total*100:.1f}%)
❌ 미탐: {s['miss']}개 (미탐율: {s['miss']/surge_base*100:.1f}%)
✅ 정상: {s['normal']}개
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    def gen_html(self, hub_cmt, m14_cmt, anal_date):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 데이터를 JSON으로 변환
        hub_json = "[]"
        m14_json = "[]"
        hub_stats_json = "{}"
        m14_stats_json = "{}"
        
        if self.hub_df is not None and len(self.hub_df) > 0:
            hub_data = []
            for i, (idx, row) in enumerate(self.hub_df.iterrows()):
                alarm_val = str(row.get('알람', '')) if pd.notna(row.get('알람')) else ''
                cooldown_val = str(row.get('알람쿨다운', '')) if pd.notna(row.get('알람쿨다운')) else ''
                miss_val = str(row.get('알람미발생', '')) if pd.notna(row.get('알람미발생')) else ''
                miss_cd_val = str(row.get('알람미발생쿨다운', '')) if pd.notna(row.get('알람미발생쿨다운')) else ''
                hub_data.append({
                    'idx': i,
                    'time': str(row.get('현재시간', '')),
                    'pred_time': str(row.get('예측시점', '')),
                    'current': float(row.get('현재값', 0)),
                    'actual': float(row.get('실제값', 0)),
                    'pred': float(row.get('예측값_MAX', 0)),
                    'cls': str(row.get('분류', '')),
                    'alarm': alarm_val,
                    'cooldown': cooldown_val,
                    'alarm_miss': miss_val,
                    'alarm_miss_cd': miss_cd_val
                })
            hub_json = str(hub_data).replace("'", '"')
        
        if self.m14_df is not None and len(self.m14_df) > 0:
            m14_data = []
            for i, (idx, row) in enumerate(self.m14_df.iterrows()):
                alarm_val = str(row.get('알람', '')) if pd.notna(row.get('알람')) else ''
                cooldown_val = str(row.get('알람쿨다운', '')) if pd.notna(row.get('알람쿨다운')) else ''
                miss_val = str(row.get('알람미발생', '')) if pd.notna(row.get('알람미발생')) else ''
                miss_cd_val = str(row.get('알람미발생쿨다운', '')) if pd.notna(row.get('알람미발생쿨다운')) else ''
                m14_data.append({
                    'idx': i,
                    'time': str(row.get('현재시간', '')),
                    'pred_time': str(row.get('예측시점', '')),
                    'current': float(row.get('현재TOTALCNT', 0)),
                    'actual': float(row.get('실제값', 0)),
                    'pred': float(row.get('예측값_MAX', row.get('최종예측', 0))),
                    'cls': str(row.get('분류', '')),
                    'alarm': alarm_val,
                    'cooldown': cooldown_val,
                    'alarm_miss': miss_val,
                    'alarm_miss_cd': miss_cd_val
                })
            m14_json = str(m14_data).replace("'", '"')
        
        if self.hub_stats:
            s = self.hub_stats
            hub_stats_json = str({
                'total': s['total_rows'],
                'surge_total': s['surge_total'],
                'detect': s['detect'],
                'miss': s['miss'],
                'normal': s['normal'],
                'false': s['false'],
                'alarm_total': s.get('alarm_total', 0)
            }).replace("'", '"')
        
        if self.m14_stats:
            s = self.m14_stats
            m14_stats_json = str({
                'total': s['total_rows'],
                'surge_total': s['surge_total'],
                'detect': s['detect'],
                'miss': s['miss'],
                'normal': s['normal'],
                'false': s['false'],
                'alarm_total': s.get('alarm_total', 0)
            }).replace("'", '"')
        
        html = f'''<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>ML급증 감지 분석 리포트 V8.2</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'맑은 고딕',sans-serif;background:#0a0a0f;color:#fff;padding:25px;font-size:0.9rem}}
.container{{max-width:1400px;margin:0 auto}}
h1{{font-size:2rem;text-align:center;background:linear-gradient(135deg,#00d4ff,#ff6b35);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:8px}}
.sub{{text-align:center;color:#888;margin-bottom:20px;font-size:0.9rem}}
.view-toggle{{display:flex;justify-content:center;gap:15px;margin-bottom:25px;padding:15px;background:#12121a;border-radius:12px;border:1px solid #2a2a3a}}
.toggle-btn{{padding:12px 28px;border-radius:8px;border:2px solid #444;background:#1a1a2a;color:#888;cursor:pointer;font-size:0.95rem;font-weight:bold;transition:all 0.2s}}
.toggle-btn:hover{{border-color:#00d4ff;color:#fff;background:#252540}}
.toggle-btn.active{{background:linear-gradient(135deg,#00d4ff,#0099cc);border-color:#00d4ff;color:#000;box-shadow:0 0 15px rgba(0,212,255,0.4)}}
.toggle-btn.active-warn{{background:linear-gradient(135deg,#ff4466,#cc2244);border-color:#ff4466;color:#fff;box-shadow:0 0 15px rgba(255,68,102,0.4)}}
.system{{background:#12121a;border-radius:16px;padding:25px;margin-bottom:30px;border:1px solid #2a2a3a}}
.sys-header{{display:flex;justify-content:space-between;align-items:center;margin-bottom:20px;padding-bottom:15px;border-bottom:2px solid #333}}
.sys-title{{font-size:1.3rem;font-weight:bold}}
.sys-title.hub{{color:#00d4ff}}
.sys-title.m14{{color:#ff6b35}}
.sys-stats{{display:flex;gap:20px;font-size:0.95rem}}
.sys-stats span{{color:#aaa}}
.sys-stats strong{{margin-left:5px}}
.case-row{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px}}
.case-row.single{{grid-template-columns:1fr;max-width:600px;margin:0 auto 20px auto}}
.case-box{{background:#0a0a12;border-radius:12px;padding:20px;border:1px solid #333}}
.case-title{{font-size:1rem;font-weight:bold;margin-bottom:15px;display:flex;justify-content:space-between;align-items:center}}
.case-title.surge{{color:#00d4ff}}
.case-title.normal{{color:#888}}
.items{{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}}
.item{{background:#15151f;padding:12px;border-radius:8px;text-align:center;cursor:pointer;transition:all 0.2s}}
.item:hover{{background:#1a1a2a;transform:scale(1.02);box-shadow:0 0 10px rgba(0,212,255,0.3)}}
.item-label{{font-size:0.7rem;color:#666;margin-bottom:4px}}
.item-value{{font-size:1.3rem;font-weight:bold}}
.item-sub{{font-size:0.75rem;color:#888;margin-top:3px}}
.ok{{color:#00ff88}}.warn{{color:#ffcc00}}.bad{{color:#ff4466}}.gray{{color:#aaa}}
.stat-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:15px;margin-bottom:15px}}
.stat-box{{background:#0a0a12;border-radius:12px;padding:20px;text-align:center;cursor:pointer;transition:all 0.2s;border:1px solid #333}}
.stat-box:hover{{background:#15151f;transform:scale(1.02);box-shadow:0 0 15px rgba(0,212,255,0.3)}}
.stat-label{{font-size:0.9rem;margin-bottom:8px;font-weight:bold}}
.stat-value{{font-size:2rem;font-weight:bold}}
.stat-sub{{font-size:0.75rem;color:#888;margin-top:5px}}
.total-summary{{display:flex;gap:15px;margin-bottom:15px;padding:15px;background:#0a0a12;border-radius:10px;border:1px solid #333}}
.total-ok,.total-ng{{flex:1;padding:12px 15px;border-radius:8px;font-size:0.95rem;text-align:center}}
.total-ok{{background:#1a3a1a;color:#00ff88;border:1px solid #00ff88}}
.total-ng{{background:#3a1a1a;color:#ff4466;border:1px solid #ff4466}}
.total-ok strong,.total-ng strong{{font-size:1.3rem;margin:0 5px}}
.total-ok span,.total-ng span{{opacity:0.8}}
.charts{{display:grid;grid-template-columns:1fr 1fr;gap:15px;margin-bottom:20px}}
.charts.single{{grid-template-columns:1fr}}
.chart-box{{background:#0a0a12;border-radius:12px;padding:15px;border:1px solid #333}}
.chart-box.full{{max-width:500px;margin:0 auto}}
.chart-title{{font-size:0.85rem;color:#888;text-align:center;margin-bottom:10px}}
.chart-wrap{{height:200px;cursor:pointer}}
.chart-wrap.tall{{height:280px}}
.normal-section{{transition:opacity 0.3s ease}}
.normal-section.hidden{{display:none}}
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
.modal-info{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}}
.modal-stat{{background:#0a0a12;padding:12px;border-radius:8px;text-align:center}}
.modal-stat-label{{font-size:0.75rem;color:#666;margin-bottom:4px}}
.modal-stat-value{{font-size:1.1rem;font-weight:bold}}
.event-list{{max-height:60vh;overflow-y:auto;background:#0a0a12;border-radius:8px;padding:10px}}
.event-item{{display:grid;grid-template-columns:50px 1fr 1fr 80px 80px 80px 120px;gap:10px;padding:10px 15px;border-bottom:1px solid #222;cursor:pointer;transition:all 0.2s;font-size:0.85rem}}
.event-item:hover{{background:#1a1a2a}}
.event-item.header{{background:#15151f;font-weight:bold;color:#888;cursor:default}}
.event-item.header:hover{{background:#15151f}}
.event-num{{color:#00d4ff}}
.event-time{{color:#aaa}}
.event-current{{color:#00ff88;text-align:right}}
.event-actual{{color:#00d4ff;text-align:right}}
.event-pred{{color:#ff6b35;text-align:right}}
.event-cls{{text-align:center;padding:2px 8px;border-radius:4px;font-size:0.75rem}}
.event-cls.detect{{background:#1a3a1a;color:#00ff88}}
.event-cls.miss{{background:#3a1a1a;color:#ff4466}}
.event-cls.normal{{background:#1a1a2a;color:#888}}
.event-cls.false{{background:#3a1a2a;color:#ff4466}}
.back-btn{{background:#333;color:#fff;border:none;padding:8px 15px;border-radius:6px;cursor:pointer;margin-right:10px}}
.back-btn:hover{{background:#444}}
.mode-desc{{text-align:center;color:#666;font-size:0.8rem;margin-top:8px}}
.help-box{{position:fixed;right:15px;top:50%;transform:translateY(-50%);width:220px;background:#1a1a2a;border:2px solid #00d4ff;border-radius:12px;padding:12px;z-index:100;box-shadow:0 0 20px rgba(0,212,255,0.3);max-height:85vh;overflow-y:auto}}
.help-box h3{{font-size:0.8rem;color:#00d4ff;margin-bottom:10px;text-align:center;border-bottom:1px solid #333;padding-bottom:6px}}
.help-box .close-btn{{position:absolute;top:8px;right:10px;background:none;border:none;color:#666;font-size:1rem;cursor:pointer}}
.help-box .close-btn:hover{{color:#fff}}
.help-item{{margin-bottom:6px;font-size:0.7rem;line-height:1.3}}
.help-item:last-child{{margin-bottom:0}}
.help-label{{font-weight:bold;margin-bottom:2px}}
.help-label.ok{{color:#00ff88}}
.help-label.bad{{color:#ff4466}}
.help-label.gray{{color:#aaa}}
.help-desc{{color:#888}}
.help-toggle{{position:fixed;right:15px;top:50%;transform:translateY(-50%);width:36px;height:36px;background:#00d4ff;border:none;border-radius:50%;cursor:pointer;font-size:1.2rem;z-index:101;display:none}}
.help-box.hidden{{display:none}}
.help-box.hidden+.help-toggle{{display:block}}
@media(max-width:800px){{.stat-row{{grid-template-columns:repeat(2,1fr)}}.sys-header{{flex-direction:column;gap:10px}}.sys-stats{{flex-wrap:wrap}}.modal-info{{grid-template-columns:1fr}}.help-box{{display:none}}.help-toggle{{display:block}}}}
</style>
</head>
<body>
<div class="container">
<h1>📊 ML급증 감지 분석 리포트 V8.2 (알람기반)</h1>
<p class="sub">📅 분석 날짜: {anal_date} | 생성: {now}</p>
'''
        
        if self.hub_stats and self.hub_stats['total_rows'] > 0:
            s = self.hub_stats
            t = s['total_rows']
            surge = s['surge_total'] or 1
            normal_case = t - surge if t - surge > 0 else 1
            
            # 알람파일 유무 체크
            hub_has_alarm = self.hub_alarm_df is not None and len(self.hub_alarm_df) > 0
            alarm_total = s.get('alarm_total', 0) or 1
            surge_base = s.get('surge_base', 1) or 1  # 이상탐지 + 미탐
            
            if hub_has_alarm:
                mode_text = "알람기반"
                mode_color = "#00ff88"
                stats_html = f'<span>분석대상 <strong>{t}</strong>개</span><span style="color:{mode_color};font-weight:bold">📡 {mode_text}</span>'
            else:
                mode_text = "예측기반"
                mode_color = "#ffcc00"
                stats_html = f'<span>분석대상 <strong>{t}</strong>개</span><span style="color:{mode_color};font-weight:bold">📊 {mode_text}</span>'
            
            # 퍼센트 계산 (알람/예측 기반 동일)
            detect_rate = s['detect']/surge_base*100
            miss_rate = s['miss']/surge_base*100
            ohtam_rate = s['false']/alarm_total*100
            
            cmt_html = ""
            if hub_cmt:
                cmt_html = f'<div class="comment"><div class="comment-title hub">📝 코멘트</div><div class="comment-content">{hub_cmt.replace(chr(10),"<br>")}</div></div>'
            
            html += f'''
<div class="system" id="hubSystem">
<div class="sys-header">
<div class="sys-title hub">🏠 HUBROOM (기준: 350)</div>
<div class="sys-stats">
{stats_html}
</div>
</div>
<div class="stat-row">
<div class="stat-box" onclick="showList('hub','이상탐지',350)"><div class="stat-label ok">이상탐지</div><div class="stat-value ok">{s['detect']}</div><div class="stat-sub">탐지율 {detect_rate:.1f}%</div></div>
<div class="stat-box" onclick="showList('hub','오탐',350)"><div class="stat-label bad">오탐</div><div class="stat-value bad">{s['false']}</div><div class="stat-sub">오탐율 {ohtam_rate:.1f}%</div></div>
<div class="stat-box" onclick="showList('hub','미탐',350)"><div class="stat-label bad">미탐</div><div class="stat-value bad">{s['miss']}</div><div class="stat-sub">미탐율 {miss_rate:.1f}%</div></div>
<div class="stat-box" onclick="showList('hub','정상',350)"><div class="stat-label gray">정상</div><div class="stat-value gray">{s['normal']}</div><div class="stat-sub">&nbsp;</div></div>
</div>
{cmt_html}
</div>
'''
        
        if self.m14_stats and self.m14_stats['total_rows'] > 0:
            s = self.m14_stats
            t = s['total_rows']
            surge = s['surge_total'] or 1
            normal_case = t - surge if t - surge > 0 else 1
            
            # 알람파일 유무 체크
            m14_has_alarm = self.m14_alarm_df is not None and len(self.m14_alarm_df) > 0
            alarm_total = s.get('alarm_total', 0) or 1
            surge_base = s.get('surge_base', 1) or 1  # 이상탐지 + 미탐
            
            if m14_has_alarm:
                mode_text = "알람기반"
                mode_color = "#00ff88"
                stats_html = f'<span>분석대상 <strong>{t}</strong>개</span><span style="color:{mode_color};font-weight:bold">📡 {mode_text}</span>'
            else:
                mode_text = "예측기반"
                mode_color = "#ffcc00"
                stats_html = f'<span>분석대상 <strong>{t}</strong>개</span><span style="color:{mode_color};font-weight:bold">📊 {mode_text}</span>'
            
            # 퍼센트 계산 (알람/예측 기반 동일)
            detect_rate = s['detect']/surge_base*100
            miss_rate = s['miss']/surge_base*100
            ohtam_rate = s['false']/alarm_total*100
            
            cmt_html = ""
            if m14_cmt:
                cmt_html = f'<div class="comment m14"><div class="comment-title m14">📝 코멘트</div><div class="comment-content">{m14_cmt.replace(chr(10),"<br>")}</div></div>'
            
            html += f'''
<div class="system" id="m14System">
<div class="sys-header">
<div class="sys-title m14">🏭 M14_Q (기준: 1700)</div>
<div class="sys-stats">
{stats_html}
</div>
</div>
<div class="stat-row">
<div class="stat-box" onclick="showList('m14','이상탐지',1700)"><div class="stat-label ok">이상탐지</div><div class="stat-value ok">{s['detect']}</div><div class="stat-sub">탐지율 {detect_rate:.1f}%</div></div>
<div class="stat-box" onclick="showList('m14','오탐',1700)"><div class="stat-label bad">오탐</div><div class="stat-value bad">{s['false']}</div><div class="stat-sub">오탐율 {ohtam_rate:.1f}%</div></div>
<div class="stat-box" onclick="showList('m14','미탐',1700)"><div class="stat-label bad">미탐</div><div class="stat-value bad">{s['miss']}</div><div class="stat-sub">미탐율 {miss_rate:.1f}%</div></div>
<div class="stat-box" onclick="showList('m14','정상',1700)"><div class="stat-label gray">정상</div><div class="stat-value gray">{s['normal']}</div><div class="stat-sub">&nbsp;</div></div>
</div>
{cmt_html}
</div>
'''
        
        html += f'''
<div class="help-box" id="helpBox">
<button class="close-btn" onclick="this.parentElement.classList.add('hidden')">&times;</button>
<h3>📖 분류 용어 설명</h3>
<div class="help-item"><div class="help-label ok">✅ 이상탐지</div><div class="help-desc">실제 초과 + 알람 발생 (최초)</div></div>
<div class="help-item"><div class="help-label bad">❌ 오탐</div><div class="help-desc">실제 미만 + 알람 발생 (최초, 정상인데 알람)</div></div>
<div class="help-item"><div class="help-label bad">❌ 미탐</div><div class="help-desc">실제 초과 + 알람미발생 (최초, 급증인데 못잡음)</div></div>
<div class="help-item"><div class="help-label gray">✅ 정상</div><div class="help-desc">실제 미만 + 알람/알람미발생 없음</div></div>
<hr style="border-color:#333;margin:12px 0">
<h3 style="margin-bottom:8px">🕐 쿨다운 분류 (60분 이내, 통계 제외)</h3>
<div class="help-item"><div class="help-label" style="color:#888">이상탐지(쿨다운)</div><div class="help-desc">60분내 알람 있었음</div></div>
<div class="help-item"><div class="help-label" style="color:#888">오탐(쿨다운)</div><div class="help-desc">60분내 알람 있었음</div></div>
<div class="help-item"><div class="help-label" style="color:#888">미탐(쿨다운)</div><div class="help-desc">60분내 알람미발생 있었음</div></div>
<hr style="border-color:#333;margin:12px 0">
<h3 style="margin-bottom:8px">📊 퍼센트 기준 (쿨다운 제외)</h3>
<div class="help-item"><div class="help-label">탐지율</div><div class="help-desc">이상탐지 / (이상탐지 + 미탐)</div></div>
<div class="help-item"><div class="help-label">오탐율</div><div class="help-desc">오탐 / (이상탐지 + 오탐)</div></div>
<div class="help-item"><div class="help-label">미탐율</div><div class="help-desc">미탐 / (이상탐지 + 미탐)</div></div>
</div>
<button class="help-toggle" onclick="document.getElementById('helpBox').classList.remove('hidden')">❓</button>
<div class="footer">ML급증 감지 분석 시스템 V8.2 (알람기반) | AMHS 물류 예측 모델</div>
</div>
<script>
const hubStats = {hub_stats_json};
const m14Stats = {m14_stats_json};
'''
        
        html += f'''
const hubData = {hub_json};
const m14Data = {m14_json};
let modalChart = null;
let currentSystem = '';
let currentThreshold = 350;
let currentFiltered = [];

function showList(system, cls, threshold) {{
    const data = system === 'hub' ? hubData : m14Data;
    const sysName = system === 'hub' ? 'HUBROOM' : 'M14_Q';
    currentSystem = system;
    currentThreshold = threshold;
    
    let filtered = [];
    if (cls === '이상탐지') {{
        filtered = data.filter(d => d.cls.includes('이상') && !d.cls.includes('쿨다운'));
    }} else if (cls === '미탐') {{
        filtered = data.filter(d => d.cls.includes('미탐') && !d.cls.includes('쿨다운'));
    }} else if (cls === '정상') {{
        filtered = data.filter(d => d.cls.includes('정상') && !d.cls.includes('이상'));
    }} else if (cls === '오탐') {{
        filtered = data.filter(d => d.cls === '오탐');
    }} else {{
        filtered = data.filter(d => d.cls.includes(cls));
    }}
    
    if (filtered.length === 0) {{
        alert('해당 분류의 데이터가 없습니다.');
        return;
    }}
    
    currentFiltered = filtered;
    document.getElementById('listTitle').textContent = `${{sysName}} - ${{cls}} (${{filtered.length}}개)`;
    
    let html = '<div class="event-item header"><div>No.</div><div>현재시간</div><div>예측시점</div><div>현재값</div><div>실제값</div><div>예측값</div><div>분류</div></div>';
    
    filtered.forEach((d, i) => {{
        let clsClass = 'normal';
        if (d.cls.includes('이상')) clsClass = 'detect';
        else if (d.cls.includes('미탐')) clsClass = 'miss';
        else if (d.cls === '오탐') clsClass = 'false';
        
        const timeShort = d.time.split(' ')[1] || d.time;
        const predTimeShort = d.pred_time.split(' ')[1] || d.pred_time;
        
        html += `<div class="event-item" onclick="showGraph(${{i}})">
            <div class="event-num">${{i+1}}</div>
            <div class="event-time">${{timeShort}}</div>
            <div class="event-time">${{predTimeShort}}</div>
            <div class="event-current">${{d.current.toFixed(0)}}</div>
            <div class="event-actual">${{d.actual.toFixed(0)}}</div>
            <div class="event-pred">${{d.pred.toFixed(1)}}</div>
            <div class="event-cls ${{clsClass}}">${{d.cls.substring(0,12)}}</div>
        </div>`;
    }});
    
    document.getElementById('eventListContainer').innerHTML = html;
    document.getElementById('listModal').classList.add('show');
}}

function showGraph(filteredIdx) {{
    const item = currentFiltered[filteredIdx];
    const data = currentSystem === 'hub' ? hubData : m14Data;
    const sysName = currentSystem === 'hub' ? 'HUBROOM' : 'M14_Q';
    const centerIdx = item.idx;
    
    const startIdx = Math.max(0, centerIdx - 30);
    const endIdx = Math.min(data.length, centerIdx + 31);
    const rangeData = data.slice(startIdx, endIdx);
    const centerPos = centerIdx - startIdx;
    
    // 알람/쿨다운/미발생 개수 세기
    const alarmCount = rangeData.filter(d => d.alarm && d.alarm.length > 0).length;
    const cooldownCount = rangeData.filter(d => d.cooldown && d.cooldown.length > 0).length;
    const missCount = rangeData.filter(d => d.alarm_miss && d.alarm_miss.length > 0).length;
    let statusText = '';
    if (alarmCount > 0) statusText += ` | 🔴알람 ${{alarmCount}}`;
    if (cooldownCount > 0) statusText += ` | 🔵쿨다운 ${{cooldownCount}}`;
    if (missCount > 0) statusText += ` | 🟣미발생 ${{missCount}}`;
    
    document.getElementById('chartTitle').innerHTML = `<button class="back-btn" onclick="backToList()">← 목록</button>${{sysName}} - ${{item.cls}}`;
    const t1 = (item.time.split(' ')[1] || item.time).substring(0,5);
    const t2 = (item.pred_time.split(' ')[1] || item.pred_time).substring(0,5);
    document.getElementById('chartInfo').textContent = `${{t1}}(현재)->${{t2}}(예측) | 현재값: ${{item.current.toFixed(0)}} | 실제값: ${{item.actual.toFixed(0)}} | 예측값: ${{item.pred.toFixed(1)}}${{statusText}}`;
    
    document.getElementById('listModal').classList.remove('show');
    document.getElementById('chartModal').classList.add('show');
    
    const ctx = document.getElementById('modalLineChart').getContext('2d');
    if (modalChart) modalChart.destroy();
    
    // 알람/쿨다운/미발생 포인트 색상 (예측값에 적용!)
    const predPointColors = rangeData.map((d, i) => {{
        if (i === centerPos) return '#ffcc00';  // 현재 선택 포인트: 노란색
        if (d.alarm && d.alarm.length > 0) return '#ff4466';  // 알람: 빨간색
        if (d.cooldown && d.cooldown.length > 0) return '#4488ff';  // 쿨다운: 파란색
        if (d.alarm_miss && d.alarm_miss.length > 0) return '#aa44ff';  // 미발생: 보라색
        return '#ff6b35';  // 기본 주황색
    }});
    const predPointSizes = rangeData.map((d, i) => {{
        if (i === centerPos) return 12;  // 현재 선택 포인트: 크게
        if (d.alarm && d.alarm.length > 0) return 8;  // 알람
        if (d.cooldown && d.cooldown.length > 0) return 5;  // 쿨다운: 작게
        if (d.alarm_miss && d.alarm_miss.length > 0) return 8;  // 미발생
        return 4;
    }});
    
    const labelsShort = rangeData.map(d => (d.pred_time.split(' ')[1] || d.pred_time).substring(0,5) + '(예측)');
    const labelsFull = rangeData.map(d => {{
        const t1 = (d.time.split(' ')[1] || d.time).substring(0,5);
        const t2 = (d.pred_time.split(' ')[1] || d.pred_time).substring(0,5);
        return t1 + '(현재)->' + t2 + '(예측)';
    }});
    
    modalChart = new Chart(ctx, {{
        type: 'line',
        data: {{
            labels: labelsShort,
            datasets: [
                {{
                    label: '현재값 (현재시간)',
                    data: rangeData.map(d => d.current),
                    borderColor: '#00ff88',
                    backgroundColor: 'rgba(0,255,136,0.1)',
                    fill: false,
                    tension: 0.3,
                    pointRadius: 3,
                    borderWidth: 2,
                    hidden: true
                }},
                {{
                    label: '실제값 (예측시점)',
                    data: rangeData.map(d => d.actual),
                    borderColor: '#00d4ff',
                    backgroundColor: 'rgba(0,212,255,0.1)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 4
                }},
                {{
                    label: '예측값 (예측시점)',
                    data: rangeData.map(d => d.pred),
                    borderColor: '#ff6b35',
                    backgroundColor: 'rgba(255,107,53,0.1)',
                    fill: false,
                    tension: 0.3,
                    pointBackgroundColor: predPointColors,
                    pointRadius: predPointSizes,
                    pointBorderColor: predPointColors
                }},
                {{
                    label: '리미트값 (' + currentThreshold + ')',
                    data: rangeData.map(() => currentThreshold),
                    borderColor: '#ff4466',
                    borderDash: [5, 5],
                    borderWidth: 2,
                    fill: false,
                    pointRadius: 0,
                    pointHoverRadius: 0,
                    pointHitRadius: 0,
                    tension: 0,
                    order: 999
                }}
            ]
        }},
        options: {{
            responsive: true,
            maintainAspectRatio: false,
            events: ['click'],
            interaction: {{mode: 'nearest', intersect: true}},
            plugins: {{
                legend: {{ 
                    position: 'top', 
                    labels: {{ color: '#fff' }},
                    onClick: function(e, legendItem, legend) {{
                        const index = legendItem.datasetIndex;
                        const ci = legend.chart;
                        const meta = ci.getDatasetMeta(index);
                        meta.hidden = meta.hidden === null ? !ci.data.datasets[index].hidden : null;
                        if (index === 0) {{
                            ci.data.labels = meta.hidden ? labelsShort : labelsFull;
                        }}
                        ci.update();
                    }}
                }},
                datalabels: {{ display: false }},
                tooltip: {{
                    enabled: true,
                    backgroundColor: 'rgba(0,0,0,0.9)',
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    borderColor: '#444',
                    borderWidth: 1,
                    padding: 8,
                    displayColors: true,
                    titleFont: {{ size: 11 }},
                    bodyFont: {{ size: 10 }},
                    filter: function(tooltipItem) {{
                        return tooltipItem.datasetIndex !== 3;
                    }},
                    callbacks: {{
                        title: function(context) {{ return context[0].label; }},
                        label: function(context) {{
                            const idx = context.dataIndex;
                            const d = rangeData[idx];
                            if (context.datasetIndex === 0) return '🟢 현재값: ' + d.current.toFixed(0);
                            else if (context.datasetIndex === 1) return '🔵 실제값: ' + d.actual.toFixed(0);
                            else if (context.datasetIndex === 2) return '🟠 예측값: ' + d.pred.toFixed(1);
                            else return null;
                        }},
                        afterBody: function(context) {{
                            const idx = context[0].dataIndex;
                            const d = rangeData[idx];
                            const diff = d.pred - d.actual;
                            const timeStr = (d.pred_time.split(' ')[1] || d.pred_time).substring(0,8);
                            let result = [];
                            
                            // 알람 상태 표시
                            if (d.alarm && d.alarm.length > 0) {{
                                result.push('🔴 알람발생 [' + timeStr + ']');
                            }} else if (d.cooldown && d.cooldown.length > 0) {{
                                result.push('🔵 알람쿨다운(' + d.cooldown + ')');
                            }} else if (d.alarm_miss && d.alarm_miss.length > 0) {{
                                result.push('🟣 알람미발생 (' + d.alarm_miss + ')');
                            }} else if (d.alarm_miss_cd && d.alarm_miss_cd.length > 0) {{
                                result.push('🟣 알람미발생쿨다운(' + d.alarm_miss_cd + ')');
                            }}
                            
                            result.push('');
                            result.push('예측 오차: ' + (diff >= 0 ? '+' : '') + diff.toFixed(1));
                            return result;
                        }}
                    }}
                }}
            }},
            scales: {{
                x: {{ ticks: {{ color: '#888', maxRotation: 60, font: {{ size: 8 }} }}, grid: {{ color: '#333' }} }},
                y: {{ ticks: {{ color: '#888' }}, grid: {{ color: '#333' }} }}
            }}
        }}
    }});
    
    const actuals = rangeData.map(d => d.actual);
    const preds = rangeData.map(d => d.pred);
    document.getElementById('statCount').textContent = rangeData.length + '분';
    document.getElementById('statCurrent').textContent = item.current.toFixed(0);
    document.getElementById('statActualAvg').textContent = (actuals.reduce((a,b)=>a+b,0)/actuals.length).toFixed(1);
    document.getElementById('statPredAvg').textContent = (preds.reduce((a,b)=>a+b,0)/preds.length).toFixed(1);
    document.getElementById('statActualMax').textContent = Math.max(...actuals).toFixed(1);
}}

function backToList() {{
    document.getElementById('chartModal').classList.remove('show');
    document.getElementById('listModal').classList.add('show');
    if (modalChart) {{ modalChart.destroy(); modalChart = null; }}
}}

function closeListModal() {{ document.getElementById('listModal').classList.remove('show'); }}
function closeChartModal() {{
    document.getElementById('chartModal').classList.remove('show');
    if (modalChart) {{ modalChart.destroy(); modalChart = null; }}
}}

document.getElementById('listModal').addEventListener('click', function(e) {{ if (e.target === this) closeListModal(); }});
document.getElementById('chartModal').addEventListener('click', function(e) {{ if (e.target === this) closeChartModal(); }});
</script>

<div id="listModal" class="modal">
<div class="modal-content">
<div class="modal-header">
<div id="listTitle" class="modal-title">이벤트 목록</div>
<button class="modal-close" onclick="closeListModal()">&times;</button>
</div>
<div id="eventListContainer" class="event-list"></div>
</div>
</div>

<div id="chartModal" class="modal">
<div class="modal-content">
<div class="modal-header">
<div id="chartTitle" class="modal-title">그래프</div>
<button class="modal-close" onclick="closeChartModal()">&times;</button>
</div>
<div id="chartInfo" style="text-align:center;color:#888;margin-bottom:15px;font-size:0.9rem"></div>
<div class="modal-chart"><canvas id="modalLineChart"></canvas></div>
<div class="modal-info">
<div class="modal-stat"><div class="modal-stat-label">범위</div><div id="statCount" class="modal-stat-value ok">0</div></div>
<div class="modal-stat"><div class="modal-stat-label">현재값</div><div id="statCurrent" class="modal-stat-value" style="color:#00ff88">0</div></div>
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