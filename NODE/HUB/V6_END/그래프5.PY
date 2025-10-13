import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser, scrolledtext, ttk
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
import os

# --------------------------------------------------------------------------
# 인코딩 자동 감지 함수 추가
# --------------------------------------------------------------------------
def read_csv_safe(filepath):
    """여러 인코딩을 시도하여 CSV 파일을 안전하게 읽습니다."""
    encodings = ['utf-8', 'cp949', 'euc-kr', 'ms949', 'latin-1']
    
    for encoding in encodings:
        try:
            return pd.read_csv(filepath, encoding=encoding)
        except UnicodeDecodeError:
            continue
    
    return pd.read_csv(filepath, encoding='utf-8', errors='ignore')

# --------------------------------------------------------------------------
# 성능 정보 창 클래스
# --------------------------------------------------------------------------
class PerformanceWindow:
    def __init__(self, parent):
        self.window = tk.Toplevel(parent)
        self.window.title("평가 결과 분석 입력")
        self.window.geometry("600x700")
        
        self.values = None
        
        main_frame = tk.Frame(self.window)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        top_frame = tk.LabelFrame(main_frame, text="📊 전체 성능 지표 입력", padx=10, pady=10)
        top_frame.pack(fill='x', pady=(0, 10))
        
        input_frame = tk.Frame(top_frame)
        input_frame.pack(pady=10)
        
        tk.Label(input_frame, text="MAE:", font=('Arial', 10)).grid(row=0, column=0, sticky='e', padx=5, pady=5)
        self.mae_var = tk.StringVar(value="30.23")
        tk.Entry(input_frame, textvariable=self.mae_var, width=15).grid(row=0, column=1, padx=5, pady=5)
        
        tk.Label(input_frame, text="RMSE:", font=('Arial', 10)).grid(row=1, column=0, sticky='e', padx=5, pady=5)
        self.rmse_var = tk.StringVar(value="36.02")
        tk.Entry(input_frame, textvariable=self.rmse_var, width=15).grid(row=1, column=1, padx=5, pady=5)
        
        tk.Label(input_frame, text="R²:", font=('Arial', 10)).grid(row=2, column=0, sticky='e', padx=5, pady=5)
        self.r2_var = tk.StringVar(value="0.72")
        tk.Entry(input_frame, textvariable=self.r2_var, width=15).grid(row=2, column=1, padx=5, pady=5)
        
        bottom_frame = tk.LabelFrame(main_frame, text="📝 기타 정보 (붙여넣기 가능)", padx=10, pady=10)
        bottom_frame.pack(fill='both', expand=True)
        
        self.text_widget = scrolledtext.ScrolledText(bottom_frame, height=20, width=70, wrap=tk.WORD)
        self.text_widget.pack(fill='both', expand=True)
        
        self.text_widget.insert('1.0', """📊 평가 결과 분석""")
        
        tk.Button(main_frame, text="확인", 
                 command=self.save_and_close,
                 bg='#3498DB', fg='white',
                 font=('Arial', 10, 'bold')).pack(pady=10)
        
    def save_and_close(self):
        self.values = {
            'mae': self.mae_var.get(),
            'rmse': self.rmse_var.get(),
            'r2': self.r2_var.get(),
            'other_info': self.text_widget.get('1.0', 'end-1c')
        }
        self.window.destroy()

# --------------------------------------------------------------------------
# 그래프 생성 로직
# --------------------------------------------------------------------------
def create_graph(params, perf_values=None):
    try:
        df = read_csv_safe(params['file_path'])
        
        actual_x_col = params['actual_x']
        actual_y_col = params['actual_y']
        hubroom_x_col = params['hubroom_x']
        hubroom_y_col = params['hubroom_y']
        predicted1_x_col = params['predicted1_x']
        predicted1_y_col = params['predicted1_y']
        predicted2_x_col = params['predicted2_x']
        predicted2_y_col = params['predicted2_y']
        predicted3_x_col = params['predicted3_x']
        predicted3_y_col = params['predicted3_y']
        
        # Y값을 숫자로 변환
        df[actual_y_col] = pd.to_numeric(df[actual_y_col], errors='coerce')
        df[hubroom_y_col] = pd.to_numeric(df[hubroom_y_col], errors='coerce')
        df[predicted1_y_col] = pd.to_numeric(df[predicted1_y_col], errors='coerce')
        df[predicted2_y_col] = pd.to_numeric(df[predicted2_y_col], errors='coerce')
        df[predicted3_y_col] = pd.to_numeric(df[predicted3_y_col], errors='coerce')
        
        # 트렌드 데이터 처리
        trend_data = params.get('trend_data', [])
        for trend_info in trend_data:
            x_col = trend_info['x_col']
            y_col = trend_info['y_col']
            if x_col in df.columns and y_col in df.columns:
                df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
                try:
                    df[x_col] = pd.to_datetime(df[x_col])
                except:
                    pass
        
        # 필수 컬럼만 NaN 체크
        required_cols = [actual_y_col, hubroom_y_col, predicted1_y_col, predicted2_y_col, predicted3_y_col]
        df = df.dropna(subset=required_cols)
        
        # 날짜 변환
        try:
            df[actual_x_col] = pd.to_datetime(df[actual_x_col])
        except:
            pass
            
        try:
            df[hubroom_x_col] = pd.to_datetime(df[hubroom_x_col])
        except:
            pass
            
        try:
            df[predicted1_x_col] = pd.to_datetime(df[predicted1_x_col])
        except:
            pass
            
        try:
            df[predicted2_x_col] = pd.to_datetime(df[predicted2_x_col])
        except:
            pass
            
        try:
            df[predicted3_x_col] = pd.to_datetime(df[predicted3_x_col])
        except:
            pass
        
        # 실제값의 시간을 공통 시간축으로 사용
        df['common_time'] = df[actual_x_col]

        fig = go.Figure()

        # 기본 호버 템플릿
        base_hover = '<b style="color: #2E86C1; font-size: 14px;">📊 INFO</b><br>' + \
                     '<b>시간:</b> %{x}<br>' + \
                     '<b>값:</b> %{y:.2f}' + \
                     '<extra></extra>'

        # 실제값 라인
        fig.add_trace(go.Scattergl(
            x=df['common_time'],
            y=df[actual_y_col], 
            mode='lines+markers',
            name='실제값 (Actual)',
            line=dict(color=params['actual_color'], 
                     dash=None if params['actual_style'] == 'Solid' else params['actual_style'].lower(), 
                     width=3),
            marker=dict(size=6),
            hovertemplate=base_hover
        ))
        
        # HUBROOM 라인
        fig.add_trace(go.Scattergl(
            x=df['common_time'],
            y=df[hubroom_y_col], 
            mode='lines+markers',
            name='HUBROOM',
            line=dict(color=params['hubroom_color'], 
                     dash=None if params['hubroom_style'] == 'Solid' else params['hubroom_style'].lower(), 
                     width=3),
            marker=dict(size=6),
            hovertemplate=base_hover
        ))
        
        # 예측값1 라인
        fig.add_trace(go.Scattergl(
            x=df['common_time'],
            y=df[predicted1_y_col], 
            mode='lines+markers',
            name='예측값1 (Predicted1)',
            line=dict(color=params['predicted1_color'], 
                     dash=None if params['predicted1_style'] == 'Solid' else params['predicted1_style'].lower(), 
                     width=3),
            marker=dict(size=6),
            hovertemplate=base_hover
        ))
        
        # 예측값2 라인
        fig.add_trace(go.Scattergl(
            x=df['common_time'],
            y=df[predicted2_y_col], 
            mode='lines+markers',
            name='예측값2 (Predicted2)',
            line=dict(color=params['predicted2_color'], 
                     dash=None if params['predicted2_style'] == 'Solid' else params['predicted2_style'].lower(), 
                     width=3),
            marker=dict(size=6),
            hovertemplate=base_hover
        ))
        
        # 예측값3 라인 (새로 추가!)
        fig.add_trace(go.Scattergl(
            x=df['common_time'],
            y=df[predicted3_y_col], 
            mode='lines+markers',
            name='예측값3 (Predicted3)',
            line=dict(color=params['predicted3_color'], 
                     dash=None if params['predicted3_style'] == 'Solid' else params['predicted3_style'].lower(), 
                     width=3),
            marker=dict(size=6),
            hovertemplate=base_hover
        ))
        
        # 트렌드 데이터 라인들
        trend_colors = ['#808080', '#A9A9A9', '#696969', '#778899', '#708090', 
                       '#2F4F4F', '#556B2F', '#8B4513', '#A0522D', '#B8860B',
                       '#BDB76B', '#8B008B', '#9932CC', '#8B0000', '#DC143C',
                       '#FF6347', '#FF4500']
        
        for idx, trend_info in enumerate(trend_data):
            x_col = trend_info['x_col']
            y_col = trend_info['y_col']
            name = trend_info['name']
            
            if x_col in df.columns and y_col in df.columns:
                color = trend_colors[idx % len(trend_colors)]
                fig.add_trace(go.Scattergl(
                    x=df[x_col],  # 각 트렌드 데이터의 고유 X축 사용
                    y=df[y_col],
                    mode='lines',
                    name=f'Trend: {name}',
                    line=dict(color=color, dash='dot', width=1.5),
                    opacity=0.7,
                    hovertemplate=f'<b>{name}</b><br>' + 
                                 '시간: %{x}<br>' + 
                                 '값: %{y:.2f}' +
                                 '<extra></extra>'
                ))
        
        # 리미트선
        try:
            limit_value = float(params.get('limit_value', 300))
            all_x = df['common_time'].sort_values()
            limit_x = pd.date_range(start=all_x.iloc[0], end=all_x.iloc[-1], periods=100)
            limit_y = [limit_value] * 100
            
            fig.add_trace(go.Scatter(
                x=limit_x,
                y=limit_y,
                mode='lines',
                name=f'리미트선 ({limit_value})',
                line=dict(color='red', width=2.5),
                hovertemplate='<b style="color: red;">⚠️ 리미트선</b><br>' +
                             f'<span style="color: red;">설정값: {limit_value:.2f}</span><br>' +
                             '<extra></extra>'
            ))
        except:
            pass
        
        # 레이아웃
        fig.update_layout(
            title=params['title'], 
            xaxis_title='시간', 
            yaxis_title='값', 
            hovermode='closest',
            plot_bgcolor='white',
            xaxis=dict(showgrid=True, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridcolor='lightgray'),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial",
                bordercolor="#2E86C1"
            )
        )

        output_filename = "final_guided_graph.html"
        html_content = fig.to_html(include_plotlyjs='cdn')

        if perf_values:
            other_info = perf_values.get('other_info', '정보가 입력되지 않았습니다.')
            other_info_escaped = other_info.replace('<', '&lt;').replace('>', '&gt;')
            
            performance_section = f"""
            <div style="margin: 30px auto; max-width: 1200px; padding: 0 20px;">
                <div style="background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                    <h2 style="color: #2E86C1; text-align: center; margin-bottom: 30px; border-bottom: 3px solid #2E86C1; padding-bottom: 15px;">
                        📊 평가 결과 분석 - 기타 정보
                    </h2>
                    <div style="white-space: pre-wrap; line-height: 1.8; font-size: 14px; color: #333; background-color: #f8f9fa; padding: 25px; border-radius: 8px; border: 2px solid #dee2e6; font-family: 'Courier New', monospace;">
{other_info_escaped}
                    </div>
                </div>
            </div>
            """
        else:
            performance_section = ""

        html_content = html_content.replace('</body>', performance_section + '</body>')

        style_section = """
        <style>
            body {
                background-color: #f8f9fa;
                margin: 0;
                padding: 20px 0;
            }
            .plotly-graph-div {
                margin: 0 auto;
                max-width: 1400px;
            }
        </style>
        """
        html_content = html_content.replace('</head>', style_section + '</head>')

        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(html_content)

        webbrowser.open('file://' + os.path.realpath(output_filename))

        trend_count = len(trend_data)
        messagebox.showinfo("성공", 
            f"'{output_filename}' 파일이 생성되었습니다.\n\n"
            f"✅ 실제값 + HUBROOM + 예측값3개 (총 5개 메인 라인)\n"
            f"✅ 트렌드 {trend_count}개 (각각 고유 X축 사용)\n"
            f"✅ 범례 클릭으로 표시/숨김 가능")
            
    except Exception as e:
        messagebox.showerror("오류 발생", f"그래프 생성 중 오류:\n{e}")

# --------------------------------------------------------------------------
# GUI 애플리케이션
# --------------------------------------------------------------------------
class GraphApp:
    def __init__(self, root):
        self.root = root
        self.root.title("단계별 그래프 생성기 v4.0 (예측값3개 + 트렌드 X/Y축)")
        self.root.geometry("1100x900")
        
        self.file_path = ""
        self.df_columns = []
        self.perf_values = None
        self.trend_widgets = []

        # 전체 스크롤 가능한 메인 컨테이너
        main_canvas = tk.Canvas(root)
        scrollbar = tk.Scrollbar(root, orient="vertical", command=main_canvas.yview)
        self.scrollable_frame = tk.Frame(main_canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        )
        
        main_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=scrollbar.set)
        
        # 마우스 휠 스크롤 지원
        def _on_mousewheel(event):
            main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        main_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        main_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 상단 버튼 프레임
        top_button_frame = tk.Frame(self.scrollable_frame)
        top_button_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Button(top_button_frame, text="📊 내용정보 입력", 
                 command=self.open_performance_window,
                 font=('Arial', 10, 'bold'),
                 bg='#2ECC71', fg='white',
                 width=15, height=2).pack(side='right', padx=5)

        # 1단계: 파일 선택
        self.step1_frame = tk.LabelFrame(self.scrollable_frame, text="✅ 1단계: 파일 선택", padx=10, pady=10)
        self.step1_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Button(self.step1_frame, text="CSV 파일 열기", command=self.select_file, 
                 font=('Helvetica', 10, 'bold'), bg='#e8f4f8').pack(side='left')
        self.file_label = tk.Label(self.step1_frame, text="선택된 파일이 없습니다.", fg="blue")
        self.file_label.pack(side='left', padx=10)
        
        # 2단계: 데이터 컬럼 선택
        self.step2_frame = tk.LabelFrame(self.scrollable_frame, text="🔒 2단계: 메인 데이터 컬럼 선택", padx=10, pady=10)
        self.step2_frame.pack(fill='x', padx=10, pady=5)
        self.create_step2_widgets()
        
        # 2.5단계: 트렌드 데이터 선택
        self.step2_5_frame = tk.LabelFrame(self.scrollable_frame, text="🔒 2.5단계: 트렌드 데이터 선택 (각각 X축/Y축 지정)", padx=10, pady=10)
        self.step2_5_frame.pack(fill='x', padx=10, pady=5)
        self.create_step2_5_widgets()

        # 3단계: 그래프 스타일 설정
        self.step3_frame = tk.LabelFrame(self.scrollable_frame, text="🔒 3단계: 그래프 스타일 설정", padx=10, pady=10)
        self.step3_frame.pack(fill='x', padx=10, pady=5)
        self.create_step3_widgets()
        
        # 4단계: 그래프 생성
        self.step4_frame = tk.LabelFrame(self.scrollable_frame, text="🔒 4단계: 그래프 생성", padx=10, pady=10)
        self.step4_frame.pack(fill='x', padx=10, pady=5)
        self.create_step4_widgets()
        
        # 초기 비활성화
        self.toggle_widgets_state(self.step2_frame, 'disabled')
        self.toggle_widgets_state(self.step2_5_frame, 'disabled')
        self.toggle_widgets_state(self.step3_frame, 'disabled')
        self.toggle_widgets_state(self.step4_frame, 'disabled')
    
    def open_performance_window(self):
        perf_window = PerformanceWindow(self.root)
        self.root.wait_window(perf_window.window)
        self.perf_values = perf_window.values
    
    def toggle_widgets_state(self, frame, state):
        for child in frame.winfo_children():
            try:
                child.config(state=state)
            except tk.TclError:
                pass

    def create_step2_widgets(self):
        # 실제값 + HUBROOM + 예측값3개 = 총 10개 컬럼
        labels = [
            "실제값 X축 (시간):", 
            "실제값 Y축 (값):", 
            "HUBROOM X축 (시간):",
            "HUBROOM Y축 (값):",
            "예측값1 X축 (시간):", 
            "예측값1 Y축 (값):",
            "예측값2 X축 (시간):",
            "예측값2 Y축 (값):",
            "예측값3 X축 (시간):",  # 새로 추가
            "예측값3 Y축 (값):"      # 새로 추가
        ]
        self.column_vars = [tk.StringVar() for _ in labels]
        self.column_menus = []

        for i, label_text in enumerate(labels):
            tk.Label(self.step2_frame, text=label_text).grid(row=i, column=0, sticky='w', padx=5, pady=2)
            menu = tk.OptionMenu(self.step2_frame, self.column_vars[i], "")
            menu.config(width=30)
            menu.grid(row=i, column=1, sticky='ew', padx=5, pady=2)
            self.column_menus.append(menu)
    
    def create_step2_5_widgets(self):
        """트렌드 데이터 선택 위젯 - 각각 X축/Y축 지정 가능"""
        # 트렌드 데이터 목록
        trend_list = [
            'M14A_3F_CNV_MAXCAP',
            'M14A_3F_TO_HUB_JOB2',
            'M14A_3F_TO_HUB_JOB_ALT',
            'M14B_7F_LFT_MAXCAP',
            'M14B_7F_TO_HUB_JOB2',
            'M14B_7F_TO_HUB_JOB_ALT',
            'M16A_2F_LFT_MAXCAP',
            'M16A_2F_TO_HUB_JOB2',
            'M16A_2F_TO_HUB_JOB_ALT',
            'M16A_3F_CNV_MAXCAP',
            'M16A_3F_LFT_MAXCAP',
            'M16A_3F_M14BLFT_MAXCAP',
            'M16A_3F_STORAGE_UTIL',
            'M16A_6F_LFT_MAXCAP',
            'M16A_6F_TO_HUB_JOB',
            'M16A_6F_TO_HUB_JOB_ALT',
            'M16B_10F_TO_HUB_JOB'
        ]
        
        # 스크롤 가능한 프레임
        canvas = tk.Canvas(self.step2_5_frame, height=300)
        scrollbar_trend = tk.Scrollbar(self.step2_5_frame, orient="vertical", command=canvas.yview)
        scrollable_trend_frame = tk.Frame(canvas)
        
        scrollable_trend_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_trend_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar_trend.set)
        
        # 헤더
        tk.Label(scrollable_trend_frame, text="사용", font=('Arial', 9, 'bold')).grid(row=0, column=0, padx=5, pady=5)
        tk.Label(scrollable_trend_frame, text="트렌드 이름", font=('Arial', 9, 'bold')).grid(row=0, column=1, padx=5, pady=5, sticky='w')
        tk.Label(scrollable_trend_frame, text="X축 (시간)", font=('Arial', 9, 'bold')).grid(row=0, column=2, padx=5, pady=5)
        tk.Label(scrollable_trend_frame, text="Y축 (값)", font=('Arial', 9, 'bold')).grid(row=0, column=3, padx=5, pady=5)
        
        # 체크박스와 드롭다운 생성
        self.trend_widgets = []
        
        for idx, trend_name in enumerate(trend_list):
            row = idx + 1
            
            # 체크박스
            var = tk.BooleanVar(value=False)
            cb = tk.Checkbutton(scrollable_trend_frame, variable=var)
            cb.grid(row=row, column=0, padx=5, pady=2)
            
            # 이름 라벨
            tk.Label(scrollable_trend_frame, text=trend_name, font=('Arial', 8)).grid(row=row, column=1, sticky='w', padx=5, pady=2)
            
            # X축 선택
            x_var = tk.StringVar()
            x_menu = tk.OptionMenu(scrollable_trend_frame, x_var, "")
            x_menu.config(width=20)
            x_menu.grid(row=row, column=2, padx=5, pady=2)
            
            # Y축 선택
            y_var = tk.StringVar()
            y_menu = tk.OptionMenu(scrollable_trend_frame, y_var, "")
            y_menu.config(width=20)
            y_menu.grid(row=row, column=3, padx=5, pady=2)
            
            self.trend_widgets.append({
                'name': trend_name,
                'enabled': var,
                'x_var': x_var,
                'x_menu': x_menu,
                'y_var': y_var,
                'y_menu': y_menu
            })
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar_trend.pack(side="right", fill="y")
            
    def create_step3_widgets(self):
        tk.Label(self.step3_frame, text="그래프 제목:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.title_var = tk.StringVar(value="사용자 정의 그래프")
        tk.Entry(self.step3_frame, textvariable=self.title_var, width=50).grid(row=0, column=1, columnspan=3, sticky='ew', padx=5, pady=2)
        
        # 실제값 라인
        tk.Label(self.step3_frame, text="[실제값 라인]").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.actual_color_var = tk.StringVar(value='#1f77b4')
        self.actual_color_btn = tk.Button(self.step3_frame, text="색상", 
                                         command=lambda: self.choose_color(self.actual_color_var, self.actual_color_btn), 
                                         bg=self.actual_color_var.get())
        self.actual_color_btn.grid(row=1, column=1, padx=5)
        self.actual_style_var = tk.StringVar(value='Solid')
        tk.OptionMenu(self.step3_frame, self.actual_style_var, 'Solid', 'Dash', 'Dot').grid(row=1, column=2, padx=5)

        # HUBROOM 라인
        tk.Label(self.step3_frame, text="[HUBROOM 라인]").grid(row=2, column=0, sticky='w', padx=5, pady=5)
        self.hubroom_color_var = tk.StringVar(value='#9467bd')
        self.hubroom_color_btn = tk.Button(self.step3_frame, text="색상", 
                                         command=lambda: self.choose_color(self.hubroom_color_var, self.hubroom_color_btn), 
                                         bg=self.hubroom_color_var.get())
        self.hubroom_color_btn.grid(row=2, column=1, padx=5)
        self.hubroom_style_var = tk.StringVar(value='Solid')
        tk.OptionMenu(self.step3_frame, self.hubroom_style_var, 'Solid', 'Dash', 'Dot').grid(row=2, column=2, padx=5)

        # 예측값1 라인
        tk.Label(self.step3_frame, text="[예측값1 라인]").grid(row=3, column=0, sticky='w', padx=5, pady=5)
        self.predicted1_color_var = tk.StringVar(value='#ff7f0e')
        self.predicted1_color_btn = tk.Button(self.step3_frame, text="색상", 
                                           command=lambda: self.choose_color(self.predicted1_color_var, self.predicted1_color_btn), 
                                           bg=self.predicted1_color_var.get())
        self.predicted1_color_btn.grid(row=3, column=1, padx=5)
        self.predicted1_style_var = tk.StringVar(value='Dash')
        tk.OptionMenu(self.step3_frame, self.predicted1_style_var, 'Solid', 'Dash', 'Dot').grid(row=3, column=2, padx=5)
        
        # 예측값2 라인
        tk.Label(self.step3_frame, text="[예측값2 라인]").grid(row=4, column=0, sticky='w', padx=5, pady=5)
        self.predicted2_color_var = tk.StringVar(value='#2ca02c')
        self.predicted2_color_btn = tk.Button(self.step3_frame, text="색상", 
                                           command=lambda: self.choose_color(self.predicted2_color_var, self.predicted2_color_btn), 
                                           bg=self.predicted2_color_var.get())
        self.predicted2_color_btn.grid(row=4, column=1, padx=5)
        self.predicted2_style_var = tk.StringVar(value='Dot')
        tk.OptionMenu(self.step3_frame, self.predicted2_style_var, 'Solid', 'Dash', 'Dot').grid(row=4, column=2, padx=5)
        
        # 예측값3 라인 (새로 추가!)
        tk.Label(self.step3_frame, text="[예측값3 라인]").grid(row=5, column=0, sticky='w', padx=5, pady=5)
        self.predicted3_color_var = tk.StringVar(value='#d62728')
        self.predicted3_color_btn = tk.Button(self.step3_frame, text="색상", 
                                           command=lambda: self.choose_color(self.predicted3_color_var, self.predicted3_color_btn), 
                                           bg=self.predicted3_color_var.get())
        self.predicted3_color_btn.grid(row=5, column=1, padx=5)
        self.predicted3_style_var = tk.StringVar(value='Dash')
        tk.OptionMenu(self.step3_frame, self.predicted3_style_var, 'Solid', 'Dash', 'Dot').grid(row=5, column=2, padx=5)
        
        # 리미트선 설정
        tk.Label(self.step3_frame, text="[🔴 리미트선]").grid(row=6, column=0, sticky='w', padx=5, pady=5)
        tk.Label(self.step3_frame, text="값:").grid(row=6, column=1, sticky='e', padx=(0, 5))
        self.limit_value_var = tk.StringVar(value="300")
        tk.Entry(self.step3_frame, textvariable=self.limit_value_var, width=10).grid(row=6, column=2, sticky='w', padx=5)

    def create_step4_widgets(self):
        self.generate_button = tk.Button(self.step4_frame, text="그래프 생성 실행", 
                                        command=self.generate_graph, 
                                        font=('Helvetica', 12, 'bold'), 
                                        bg='#d3ffd3')
        self.generate_button.pack(fill='x')

    def choose_color(self, color_var, button):
        color_code = colorchooser.askcolor(title="색상 선택")[1]
        if color_code:
            color_var.set(color_code)
            button.config(bg=color_code)

    def select_file(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not path: return
        
        self.file_path = path
        filename = os.path.basename(path)
        self.file_label.config(text=f"선택됨: {filename}")
        
        try:
            df = read_csv_safe(self.file_path)
            self.df_columns = df.columns.tolist()
            
            # 메인 데이터 컬럼 메뉴 업데이트
            for i, menu in enumerate(self.column_menus):
                menu['menu'].delete(0, 'end')
                for col in self.df_columns:
                    menu['menu'].add_command(label=col, command=tk._setit(self.column_vars[i], col))
            
            # 트렌드 데이터 컬럼 메뉴 업데이트
            for widget_dict in self.trend_widgets:
                # X축 메뉴 업데이트
                widget_dict['x_menu']['menu'].delete(0, 'end')
                for col in self.df_columns:
                    widget_dict['x_menu']['menu'].add_command(label=col, command=tk._setit(widget_dict['x_var'], col))
                
                # Y축 메뉴 업데이트
                widget_dict['y_menu']['menu'].delete(0, 'end')
                for col in self.df_columns:
                    widget_dict['y_menu']['menu'].add_command(label=col, command=tk._setit(widget_dict['y_var'], col))
            
            # 컬럼 자동 추천
            patterns = [
                ['날짜', 'date', 'time'],
                ['실제값', 'actual', 'real'],
                ['hubroom', 'hub'],
                ['hubroom', 'hub'],
                ['예측날짜', 'pred_date', 'forecast'],
                ['예측값', 'predicted', 'pred'],
                ['예측날짜', 'pred_date', 'forecast', '2'],
                ['예측값', 'predicted', 'pred', '2'],
                ['예측날짜', 'pred_date', 'forecast', '3'],  # 예측값3
                ['예측값', 'predicted', 'pred', '3']          # 예측값3
            ]
            
            for idx, pattern_list in enumerate(patterns):
                for col in self.df_columns:
                    col_lower = col.lower()
                    if any(p in col_lower for p in pattern_list):
                        self.column_vars[idx].set(col)
                        break

            self.step2_frame.config(text="✅ 2단계: 메인 데이터 컬럼 선택")
            self.toggle_widgets_state(self.step2_frame, 'normal')
            self.step2_5_frame.config(text="✅ 2.5단계: 트렌드 데이터 선택")
            self.toggle_widgets_state(self.step2_5_frame, 'normal')
            self.step3_frame.config(text="✅ 3단계: 그래프 스타일 설정")
            self.toggle_widgets_state(self.step3_frame, 'normal')
            self.step4_frame.config(text="✅ 4단계: 그래프 생성")
            self.toggle_widgets_state(self.step4_frame, 'normal')

        except Exception as e:
            messagebox.showerror("오류", f"CSV 파일을 읽는 데 실패했습니다:\n{e}")

    def generate_graph(self):
        # 선택된 트렌드 데이터 수집
        selected_trends = []
        for widget_dict in self.trend_widgets:
            if widget_dict['enabled'].get():
                x_col = widget_dict['x_var'].get()
                y_col = widget_dict['y_var'].get()
                if x_col and y_col:
                    selected_trends.append({
                        'name': widget_dict['name'],
                        'x_col': x_col,
                        'y_col': y_col
                    })
        
        params = {
            'file_path': self.file_path, 
            'title': self.title_var.get(),
            'actual_x': self.column_vars[0].get(), 
            'actual_y': self.column_vars[1].get(),
            'hubroom_x': self.column_vars[2].get(),
            'hubroom_y': self.column_vars[3].get(),
            'predicted1_x': self.column_vars[4].get(), 
            'predicted1_y': self.column_vars[5].get(),
            'predicted2_x': self.column_vars[6].get(),
            'predicted2_y': self.column_vars[7].get(),
            'predicted3_x': self.column_vars[8].get(),  # 예측값3 추가
            'predicted3_y': self.column_vars[9].get(),  # 예측값3 추가
            'actual_color': self.actual_color_var.get(), 
            'actual_style': self.actual_style_var.get(),
            'hubroom_color': self.hubroom_color_var.get(),
            'hubroom_style': self.hubroom_style_var.get(),
            'predicted1_color': self.predicted1_color_var.get(), 
            'predicted1_style': self.predicted1_style_var.get(),
            'predicted2_color': self.predicted2_color_var.get(),
            'predicted2_style': self.predicted2_style_var.get(),
            'predicted3_color': self.predicted3_color_var.get(),  # 예측값3 추가
            'predicted3_style': self.predicted3_style_var.get(),  # 예측값3 추가
            'limit_value': self.limit_value_var.get(),
            'trend_data': selected_trends
        }
        
        if not all([params['actual_x'], params['actual_y'], 
                   params['hubroom_x'], params['hubroom_y'],
                   params['predicted1_x'], params['predicted1_y'],
                   params['predicted2_x'], params['predicted2_y'],
                   params['predicted3_x'], params['predicted3_y']]):
            messagebox.showwarning("경고", "2단계에서 모든 메인 데이터 축(X, Y)의 컬럼을 선택해주세요.")
            return
        
        if not self.perf_values:
            self.perf_values = {
                'mae': '30.23',
                'rmse': '36.02',
                'r2': '0.72',
                'other_info': '기타 정보가 입력되지 않았습니다.'
            }
            
        create_graph(params, self.perf_values)

if __name__ == "__main__":
    root = tk.Tk()
    app = GraphApp(root)
    root.mainloop()