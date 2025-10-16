import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser, scrolledtext, ttk
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
import os

# 현대적인 색상 테마
COLORS = {
    'primary': '#2C3E50',      # 진한 네이비
    'secondary': '#3498DB',    # 밝은 블루
    'success': '#27AE60',      # 그린
    'danger': '#E74C3C',       # 레드
    'warning': '#F39C12',      # 오렌지
    'info': '#16A085',         # 틸
    'light': '#ECF0F1',        # 연한 회색
    'dark': '#2C3E50',         # 다크
    'white': '#FFFFFF',
    'bg': '#F8F9FA',           # 배경
    'border': '#BDC3C7',       # 테두리
    'accent': '#9B59B6'        # 보라
}

# --------------------------------------------------------------------------
# 인코딩 자동 감지 함수
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
# 커스텀 스타일 버튼
# --------------------------------------------------------------------------
class StyledButton(tk.Button):
    def __init__(self, parent, **kwargs):
        style = kwargs.pop('style', 'primary')
        
        colors = {
            'primary': (COLORS['secondary'], COLORS['white']),
            'success': (COLORS['success'], COLORS['white']),
            'danger': (COLORS['danger'], COLORS['white']),
            'warning': (COLORS['warning'], COLORS['white']),
            'info': (COLORS['info'], COLORS['white']),
            'accent': (COLORS['accent'], COLORS['white'])
        }
        
        bg, fg = colors.get(style, colors['primary'])
        
        super().__init__(
            parent,
            bg=bg,
            fg=fg,
            activebackground=self._darken_color(bg),
            activeforeground=fg,
            relief='flat',
            borderwidth=0,
            cursor='hand2',
            font=('Segoe UI', 10, 'bold'),
            padx=20,
            pady=10,
            **kwargs
        )
        
        self.bind('<Enter>', lambda e: self.config(bg=self._darken_color(bg)))
        self.bind('<Leave>', lambda e: self.config(bg=bg))
    
    def _darken_color(self, hex_color):
        """색상을 약간 어둡게 만듭니다."""
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        r, g, b = max(0, r-20), max(0, g-20), max(0, b-20)
        return f'#{r:02x}{g:02x}{b:02x}'

# --------------------------------------------------------------------------
# 성능 정보 창 클래스
# --------------------------------------------------------------------------
class PerformanceWindow:
    def __init__(self, parent):
        self.window = tk.Toplevel(parent)
        self.window.title("📊 평가 결과 분석 입력")
        self.window.geometry("700x750")
        self.window.configure(bg=COLORS['bg'])
        
        # 창 중앙 배치
        self.window.update_idletasks()
        x = (self.window.winfo_screenwidth() // 2) - (700 // 2)
        y = (self.window.winfo_screenheight() // 2) - (750 // 2)
        self.window.geometry(f'700x750+{x}+{y}')
        
        self.values = None
        
        # 헤더
        header = tk.Frame(self.window, bg=COLORS['primary'], height=60)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        tk.Label(
            header,
            text="📊 평가 결과 분석",
            font=('Segoe UI', 16, 'bold'),
            bg=COLORS['primary'],
            fg=COLORS['white']
        ).pack(pady=15)
        
        main_frame = tk.Frame(self.window, bg=COLORS['bg'])
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # 성능 지표 입력 섹션
        top_frame = tk.LabelFrame(
            main_frame,
            text="  성능 지표  ",
            font=('Segoe UI', 12, 'bold'),
            bg=COLORS['white'],
            fg=COLORS['primary'],
            relief='flat',
            borderwidth=2,
            padx=20,
            pady=20
        )
        top_frame.pack(fill='x', pady=(0, 20))
        
        input_frame = tk.Frame(top_frame, bg=COLORS['white'])
        input_frame.pack(pady=10, fill='x')
        
        metrics = [
            ("MAE:", "30.23", COLORS['info']),
            ("RMSE:", "36.02", COLORS['warning']),
            ("R²:", "0.72", COLORS['success'])
        ]
        
        self.metric_vars = {}
        for idx, (label, default, color) in enumerate(metrics):
            # 라벨
            label_widget = tk.Label(
                input_frame,
                text=label,
                font=('Segoe UI', 11, 'bold'),
                bg=COLORS['white'],
                fg=color,
                width=8,
                anchor='e'
            )
            label_widget.grid(row=idx, column=0, sticky='e', padx=(0, 10), pady=8)
            
            # 입력 필드
            var = tk.StringVar(value=default)
            self.metric_vars[label.rstrip(':')] = var
            
            entry = tk.Entry(
                input_frame,
                textvariable=var,
                font=('Segoe UI', 11),
                width=20,
                relief='flat',
                borderwidth=2,
                bg=COLORS['light']
            )
            entry.grid(row=idx, column=1, padx=5, pady=8, sticky='ew')
        
        input_frame.columnconfigure(1, weight=1)
        
        # 기타 정보 섹션
        bottom_frame = tk.LabelFrame(
            main_frame,
            text="  기타 정보 (붙여넣기 가능)  ",
            font=('Segoe UI', 12, 'bold'),
            bg=COLORS['white'],
            fg=COLORS['primary'],
            relief='flat',
            borderwidth=2,
            padx=20,
            pady=20
        )
        bottom_frame.pack(fill='both', expand=True)
        
        self.text_widget = scrolledtext.ScrolledText(
            bottom_frame,
            height=20,
            width=70,
            wrap=tk.WORD,
            font=('Consolas', 10),
            relief='flat',
            borderwidth=2,
            bg=COLORS['light']
        )
        self.text_widget.pack(fill='both', expand=True)
        
        self.text_widget.insert('1.0', """📊 평가 결과 분석

여기에 평가 결과와 분석 내용을 입력하세요...
""")
        
        # 버튼
        btn_frame = tk.Frame(main_frame, bg=COLORS['bg'])
        btn_frame.pack(pady=20)
        
        StyledButton(
            btn_frame,
            text="✓ 확인",
            command=self.save_and_close,
            style='success',
            width=15
        ).pack(side='left', padx=5)
        
        StyledButton(
            btn_frame,
            text="✗ 취소",
            command=self.window.destroy,
            style='danger',
            width=15
        ).pack(side='left', padx=5)
        
    def save_and_close(self):
        self.values = {
            'mae': self.metric_vars['MAE'].get(),
            'rmse': self.metric_vars['RMSE'].get(),
            'r2': self.metric_vars['R²'].get(),
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
        
        selected_lines = {
            'actual': params.get('enable_actual', True),
            'hubroom': params.get('enable_hubroom', True),
            'predicted1': params.get('enable_predicted1', True),
            'predicted2': params.get('enable_predicted2', True),
            'predicted3': params.get('enable_predicted3', True)
        }
        
        if selected_lines['actual'] and actual_y_col and actual_y_col in df.columns:
            df[actual_y_col] = pd.to_numeric(df[actual_y_col], errors='coerce')
        else:
            selected_lines['actual'] = False
            
        if selected_lines['hubroom'] and hubroom_y_col and hubroom_y_col in df.columns:
            df[hubroom_y_col] = pd.to_numeric(df[hubroom_y_col], errors='coerce')
        else:
            selected_lines['hubroom'] = False
            
        if selected_lines['predicted1'] and predicted1_y_col and predicted1_y_col in df.columns:
            df[predicted1_y_col] = pd.to_numeric(df[predicted1_y_col], errors='coerce')
        else:
            selected_lines['predicted1'] = False
            
        if selected_lines['predicted2'] and predicted2_y_col and predicted2_y_col in df.columns:
            df[predicted2_y_col] = pd.to_numeric(df[predicted2_y_col], errors='coerce')
        else:
            selected_lines['predicted2'] = False
            
        if selected_lines['predicted3'] and predicted3_y_col and predicted3_y_col in df.columns:
            df[predicted3_y_col] = pd.to_numeric(df[predicted3_y_col], errors='coerce')
        else:
            selected_lines['predicted3'] = False
        
        if not any(selected_lines.values()):
            messagebox.showerror("오류", "최소 1개 이상의 데이터 라인을 선택해야 합니다.")
            return
        
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
        
        dropna_cols = []
        if selected_lines['actual']:
            dropna_cols.append(actual_y_col)
        if selected_lines['hubroom']:
            dropna_cols.append(hubroom_y_col)
        if selected_lines['predicted1']:
            dropna_cols.append(predicted1_y_col)
        if selected_lines['predicted2']:
            dropna_cols.append(predicted2_y_col)
        if selected_lines['predicted3']:
            dropna_cols.append(predicted3_y_col)
            
        if dropna_cols:
            df = df.dropna(subset=dropna_cols)
        
        jump_col = None
        for col in df.columns:
            if '점프예측' in col or 'jump' in col.lower():
                jump_col = col
                break
        
        pattern_col = None
        for col in df.columns:
            if '패턴예측' in col or 'pattern' in col.lower():
                pattern_col = col
                break
        
        if selected_lines['actual']:
            try:
                df[actual_x_col] = pd.to_datetime(df[actual_x_col])
                df['actual_time_str'] = df[actual_x_col].dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                df['actual_time_str'] = df[actual_x_col].astype(str)
        else:
            df['actual_time_str'] = ''
                
        if selected_lines['hubroom']:
            try:
                df[hubroom_x_col] = pd.to_datetime(df[hubroom_x_col])
                df['hubroom_time_str'] = df[hubroom_x_col].dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                df['hubroom_time_str'] = df[hubroom_x_col].astype(str)
        else:
            df['hubroom_time_str'] = ''
                
        if selected_lines['predicted1']:
            try:
                df[predicted1_x_col] = pd.to_datetime(df[predicted1_x_col])
                df['predicted1_time_str'] = df[predicted1_x_col].dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                df['predicted1_time_str'] = df[predicted1_x_col].astype(str)
        else:
            df['predicted1_time_str'] = ''
                
        if selected_lines['predicted2']:
            try:
                df[predicted2_x_col] = pd.to_datetime(df[predicted2_x_col])
                df['predicted2_time_str'] = df[predicted2_x_col].dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                df['predicted2_time_str'] = df[predicted2_x_col].astype(str)
        else:
            df['predicted2_time_str'] = ''
                
        if selected_lines['predicted3']:
            try:
                df[predicted3_x_col] = pd.to_datetime(df[predicted3_x_col])
                df['predicted3_time_str'] = df[predicted3_x_col].dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                df['predicted3_time_str'] = df[predicted3_x_col].astype(str)
        else:
            df['predicted3_time_str'] = ''
        
        if selected_lines['actual']:
            df['common_time'] = df[actual_x_col]
        elif selected_lines['hubroom']:
            df['common_time'] = df[hubroom_x_col]
        elif selected_lines['predicted1']:
            df['common_time'] = df[predicted1_x_col]
        elif selected_lines['predicted2']:
            df['common_time'] = df[predicted2_x_col]
        elif selected_lines['predicted3']:
            df['common_time'] = df[predicted3_x_col]
        
        if jump_col:
            df['jump_value'] = df[jump_col].astype(str)
            df['jump_color'] = df[jump_col].apply(
                lambda x: '#27AE60' if str(x).upper() == 'O' else '#E74C3C' if str(x).upper() == 'X' else '#666'
            )
        else:
            df['jump_value'] = 'N/A'
            df['jump_color'] = '#666'
        
        if pattern_col:
            df['pattern_value'] = df[pattern_col].astype(str)
            def get_pattern_color(val):
                val_str = str(val).strip()
                if '상승' in val_str or '증가' in val_str:
                    return '#27AE60'
                elif '하락' in val_str or '감소' in val_str:
                    return '#E74C3C'
                elif '안정' in val_str or '유지' in val_str:
                    return '#3498DB'
                else:
                    return '#8E44AD'
            df['pattern_color'] = df[pattern_col].apply(get_pattern_color)
        else:
            df['pattern_value'] = 'N/A'
            df['pattern_color'] = '#95A5A6'
        
        for trend_info in trend_data:
            y_col = trend_info.get('y_col', '')
            if y_col and y_col in df.columns:
                try:
                    df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
                except:
                    pass

        fig = go.Figure()

        if selected_lines['actual']:
            fig.add_trace(go.Scattergl(
                x=df['common_time'],
                y=df[actual_y_col], 
                mode='lines+markers',
                name='실제값 (Actual)',
                line=dict(color=params['actual_color'], 
                         dash=None if params['actual_style'] == 'Solid' else params['actual_style'].lower(), 
                         width=3),
                marker=dict(size=6),
                customdata=df[['actual_time_str']].values,
                hovertemplate='<b style="color: #2E86C1; font-size: 14px;">📊 INFO 정보</b><br>' +
                             '<span style="color: #85C1E2;">═══════════════════</span><br>' +
                             '<b style="color: #1f77b4;">🔵 실제값</b><br>' +
                             '<span style="color: #666;">시간:</span> %{customdata[0]}<br>' +
                             '<span style="color: #666;">값:</span> <b>%{y:.2f}</b>' +
                             '<extra></extra>'
            ))
        
        if selected_lines['hubroom']:
            fig.add_trace(go.Scattergl(
                x=df['common_time'],
                y=df[hubroom_y_col], 
                mode='lines+markers',
                name='HUBROOM',
                line=dict(color=params['hubroom_color'], 
                         dash=None if params['hubroom_style'] == 'Solid' else params['hubroom_style'].lower(), 
                         width=3),
                marker=dict(size=6),
                customdata=df[['hubroom_time_str']].values,
                hovertemplate='<b style="color: #9467bd;">🟣 HUBROOM</b><br>' +
                             '<span style="color: #666;">시간:</span> %{customdata[0]}<br>' +
                             '<span style="color: #666;">값:</span> <b>%{y:.2f}</b>' +
                             '<extra></extra>'
            ))
        
        if selected_lines['predicted1']:
            fig.add_trace(go.Scattergl(
                x=df['common_time'],
                y=df[predicted1_y_col], 
                mode='lines+markers',
                name='예측값1 (Predicted1)',
                line=dict(color=params['predicted1_color'], 
                         dash=None if params['predicted1_style'] == 'Solid' else params['predicted1_style'].lower(), 
                         width=3),
                marker=dict(size=6),
                customdata=df[['predicted1_time_str']].values,
                hovertemplate='<b style="color: #ff7f0e;">🔶 예측값1</b><br>' +
                             '<span style="color: #666;">시간:</span> %{customdata[0]}<br>' +
                             '<span style="color: #666;">값:</span> <b>%{y:.2f}</b>' +
                             '<extra></extra>'
            ))
        
        if selected_lines['predicted2']:
            fig.add_trace(go.Scattergl(
                x=df['common_time'],
                y=df[predicted2_y_col], 
                mode='lines+markers',
                name='예측값2 (Predicted2)',
                line=dict(color=params['predicted2_color'], 
                         dash=None if params['predicted2_style'] == 'Solid' else params['predicted2_style'].lower(), 
                         width=3),
                marker=dict(size=6),
                customdata=df[['predicted2_time_str']].values,
                hovertemplate='<b style="color: #2ca02c;">🔷 예측값2</b><br>' +
                             '<span style="color: #666;">시간:</span> %{customdata[0]}<br>' +
                             '<span style="color: #666;">값:</span> <b>%{y:.2f}</b>' +
                             '<extra></extra>'
            ))
        
        if selected_lines['predicted3']:
            fig.add_trace(go.Scattergl(
                x=df['common_time'],
                y=df[predicted3_y_col], 
                mode='lines+markers',
                name='예측값3 (Predicted3)',
                line=dict(color=params['predicted3_color'], 
                         dash=None if params['predicted3_style'] == 'Solid' else params['predicted3_style'].lower(), 
                         width=3),
                marker=dict(size=6),
                customdata=df[['predicted3_time_str']].values,
                hovertemplate='<b style="color: #d62728;">🔸 예측값3</b><br>' +
                             '<span style="color: #666;">시간:</span> %{customdata[0]}<br>' +
                             '<span style="color: #666;">값:</span> <b>%{y:.2f}</b>' +
                             '<extra></extra>'
            ))
        
        trend_colors = ['#808080', '#A9A9A9', '#696969', '#778899', '#708090', 
                       '#2F4F4F', '#556B2F', '#8B4513', '#A0522D', '#B8860B',
                       '#BDB76B', '#8B008B', '#9932CC', '#8B0000', '#DC143C',
                       '#FF6347', '#FF4500']
        
        for idx, trend_info in enumerate(trend_data):
            x_col = trend_info.get('x_col', '')
            y_col = trend_info.get('y_col', '')
            name = trend_info.get('name', '')
            
            if not x_col or not y_col or not name:
                continue
            
            if x_col in df.columns and y_col in df.columns:
                color = trend_colors[idx % len(trend_colors)]
                
                fig.add_trace(go.Scattergl(
                    x=df[x_col],
                    y=df[y_col],
                    mode='lines',
                    name=f'{name}',
                    line=dict(color=color, dash='dot', width=1.5),
                    opacity=0.7,
                    hovertemplate=f'<b>{name}:</b> %{{y:.2f}}<extra></extra>'
                ))
        
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
                hovertemplate=f'<b style="color: red;">⚠️ 리미트선: {limit_value:.2f}</b><extra></extra>'
            ))
        except Exception as e:
            print(f"리미트선 생성 중 오류: {e}")
            pass
        
        fig.update_layout(
            title=params['title'], 
            xaxis_title='시간', 
            yaxis_title='값', 
            hovermode='x unified',
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
                        📊 평가 결과 분석
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
            f"✅ 메인 5개 라인 + 트렌드 {trend_count}개\n"
            f"✅ 실제값 라인에 전체 INFO 정보 포함\n"
            f"✅ 범례 클릭으로 표시/숨김 가능")
            
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        messagebox.showerror("오류 발생", f"그래프 생성 중 오류:\n{e}\n\n상세:\n{error_detail}")

# --------------------------------------------------------------------------
# GUI 애플리케이션
# --------------------------------------------------------------------------
class GraphApp:
    def __init__(self, root):
        self.root = root
        self.root.title("📈 그래프 생성기 Pro v5.0")
        self.root.geometry("1300x900")
        self.root.configure(bg=COLORS['bg'])
        
        self.file_path = ""
        self.df_columns = []
        self.perf_values = None
        self.trend_widgets = {}

        # 헤더
        self.create_header()
        
        # 메인 컨테이너 (스크롤 가능)
        container = tk.Frame(root, bg=COLORS['bg'])
        container.pack(fill='both', expand=True)
        
        main_canvas = tk.Canvas(container, bg=COLORS['bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=main_canvas.yview)
        self.scrollable_frame = tk.Frame(main_canvas, bg=COLORS['bg'])
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        )
        
        main_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=scrollbar.set)
        
        def _on_mousewheel(event):
            main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        main_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        main_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 각 단계별 섹션
        self.create_all_sections()
        
    def create_header(self):
        """상단 헤더 생성"""
        header = tk.Frame(self.root, bg=COLORS['primary'], height=80)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        # 제목
        title_label = tk.Label(
            header,
            text="📈 그래프 생성기 Pro",
            font=('Segoe UI', 20, 'bold'),
            bg=COLORS['primary'],
            fg=COLORS['white']
        )
        title_label.pack(side='left', padx=30, pady=20)
        
        # 버전
        version_label = tk.Label(
            header,
            text="v5.0",
            font=('Segoe UI', 10),
            bg=COLORS['primary'],
            fg=COLORS['light']
        )
        version_label.pack(side='left', pady=20)
        
        # 평가 결과 버튼
        StyledButton(
            header,
            text="📊 내용정보 입력",
            command=self.open_performance_window,
            style='accent',
            width=15
        ).pack(side='right', padx=30, pady=20)
    
    def create_section_frame(self, title, emoji, state='locked'):
        """섹션 프레임 생성"""
        # 외부 프레임
        outer_frame = tk.Frame(self.scrollable_frame, bg=COLORS['bg'])
        outer_frame.pack(fill='x', padx=20, pady=10)
        
        # 카드 스타일 프레임
        card = tk.Frame(
            outer_frame,
            bg=COLORS['white'],
            relief='flat',
            borderwidth=0
        )
        card.pack(fill='x')
        
        # 그림자 효과 (간단히)
        shadow = tk.Frame(outer_frame, bg=COLORS['border'], height=2)
        shadow.place(relx=0.01, rely=1, relwidth=0.98)
        
        # 헤더
        header_frame = tk.Frame(card, bg=COLORS['secondary'] if state == 'active' else COLORS['border'], height=50)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        
        status_emoji = "✓" if state == 'active' else "🔒" if state == 'locked' else emoji
        
        tk.Label(
            header_frame,
            text=f"{status_emoji} {title}",
            font=('Segoe UI', 12, 'bold'),
            bg=COLORS['secondary'] if state == 'active' else COLORS['border'],
            fg=COLORS['white']
        ).pack(side='left', padx=20, pady=10)
        
        # 내용 프레임
        content_frame = tk.Frame(card, bg=COLORS['white'], padx=20, pady=20)
        content_frame.pack(fill='both', expand=True)
        
        return content_frame, header_frame
    
    def create_all_sections(self):
        """모든 섹션 생성"""
        # 1단계: 파일 선택
        self.step1_content, self.step1_header = self.create_section_frame("1단계: 파일 선택", "📁", 'active')
        self.create_step1_widgets()
        
        # 2단계: 메인 데이터
        self.step2_content, self.step2_header = self.create_section_frame("2단계: 메인 데이터 컬럼 선택", "📊", 'locked')
        self.create_step2_widgets()
        
        # 2.5단계: 트렌드 데이터
        self.step2_5_content, self.step2_5_header = self.create_section_frame("2.5단계: 트렌드 데이터 선택", "📈", 'locked')
        self.create_step2_5_widgets()
        
        # 3단계: 스타일 설정
        self.step3_content, self.step3_header = self.create_section_frame("3단계: 그래프 스타일 설정", "🎨", 'locked')
        self.create_step3_widgets()
        
        # 4단계: 생성
        self.step4_content, self.step4_header = self.create_section_frame("4단계: 그래프 생성", "🚀", 'locked')
        self.create_step4_widgets()
    
    def create_step1_widgets(self):
        btn_frame = tk.Frame(self.step1_content, bg=COLORS['white'])
        btn_frame.pack(fill='x')
        
        StyledButton(
            btn_frame,
            text="📂 CSV 파일 선택",
            command=self.select_file,
            style='primary',
            width=20
        ).pack(side='left', padx=5)
        
        self.file_label = tk.Label(
            btn_frame,
            text="선택된 파일이 없습니다.",
            fg=COLORS['border'],
            bg=COLORS['white'],
            font=('Segoe UI', 10)
        )
        self.file_label.pack(side='left', padx=20)
    
    def create_step2_widgets(self):
        """메인 데이터 선택 위젯"""
        labels = [
            "실제값 X축 (시간):", "실제값 Y축 (값):",
            "HUBROOM X축 (시간):", "HUBROOM Y축 (값):",
            "예측값1 X축 (시간):", "예측값1 Y축 (값):",
            "예측값2 X축 (시간):", "예측값2 Y축 (값):",
            "예측값3 X축 (시간):", "예측값3 Y축 (값):"
        ]
        
        self.enable_actual = tk.BooleanVar(value=True)
        self.enable_hubroom = tk.BooleanVar(value=True)
        self.enable_predicted1 = tk.BooleanVar(value=True)
        self.enable_predicted2 = tk.BooleanVar(value=True)
        self.enable_predicted3 = tk.BooleanVar(value=True)
        
        self.column_vars = [tk.StringVar() for _ in labels]
        self.column_menus = []

        enable_vars = [
            (self.enable_actual, "✓ 실제값 사용"),
            (self.enable_actual, ""),
            (self.enable_hubroom, "✓ HUBROOM 사용"),
            (self.enable_hubroom, ""),
            (self.enable_predicted1, "✓ 예측값1 사용"),
            (self.enable_predicted1, ""),
            (self.enable_predicted2, "✓ 예측값2 사용"),
            (self.enable_predicted2, ""),
            (self.enable_predicted3, "✓ 예측값3 사용"),
            (self.enable_predicted3, "")
        ]

        for i, label_text in enumerate(labels):
            row_frame = tk.Frame(self.step2_content, bg=COLORS['white'])
            row_frame.pack(fill='x', pady=5)
            
            if i % 2 == 0:
                enable_var, checkbox_text = enable_vars[i]
                cb = tk.Checkbutton(
                    row_frame,
                    text=checkbox_text,
                    variable=enable_var,
                    font=('Segoe UI', 10, 'bold'),
                    bg=COLORS['white'],
                    activebackground=COLORS['white'],
                    selectcolor=COLORS['light']
                )
                cb.pack(side='left', padx=(0, 20))
            
            tk.Label(
                row_frame,
                text=label_text,
                font=('Segoe UI', 10),
                bg=COLORS['white'],
                width=20,
                anchor='w'
            ).pack(side='left', padx=5)
            
            menu = ttk.Combobox(
                row_frame,
                textvariable=self.column_vars[i],
                state='readonly',
                width=40,
                font=('Segoe UI', 9)
            )
            menu.pack(side='left', padx=5, fill='x', expand=True)
            self.column_menus.append(menu)
    
    def create_step2_5_widgets(self):
        """트렌드 데이터 선택 위젯"""
        # 트렌드 그룹
        trend_groups = {
            '📦 MAXCAP 그룹': [
                'M14A_3F_CNV_MAXCAP', 'M14B_7F_LFT_MAXCAP', 'M16A_2F_LFT_MAXCAP',
                'M16A_3F_CNV_MAXCAP', 'M16A_3F_LFT_MAXCAP', 'M16A_3F_M14BLFT_MAXCAP',
                'M16A_6F_LFT_MAXCAP'
            ],
            '📊 UTIL 그룹': ['M16A_3F_STORAGE_UTIL'],
            '🔄 JOB 그룹': [
                'M14A_3F_TO_HUB_JOB2', 'M14A_3F_TO_HUB_JOB_ALT', 'M14B_7F_TO_HUB_JOB2',
                'M14B_7F_TO_HUB_JOB_ALT', 'M16A_2F_TO_HUB_JOB2', 'M16A_2F_TO_HUB_JOB_ALT',
                'M16A_6F_TO_HUB_JOB', 'M16A_6F_TO_HUB_JOB_ALT', 'M16B_10F_TO_HUB_JOB'
            ]
        }
        
        # 스크롤 가능 영역
        canvas = tk.Canvas(self.step2_5_content, height=450, bg=COLORS['white'], highlightthickness=0)
        scrollbar_v = ttk.Scrollbar(self.step2_5_content, orient="vertical", command=canvas.yview)
        scrollbar_h = ttk.Scrollbar(self.step2_5_content, orient="horizontal", command=canvas.xview)
        scrollable_frame = tk.Frame(canvas, bg=COLORS['white'])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar_v.set, xscrollcommand=scrollbar_h.set)
        
        for group_name, trend_list in trend_groups.items():
            # 그룹 프레임
            group_card = tk.Frame(
                scrollable_frame,
                bg=COLORS['light'],
                relief='flat',
                borderwidth=2
            )
            group_card.pack(fill='x', padx=10, pady=10)
            
            # 그룹 헤더
            header = tk.Frame(group_card, bg=COLORS['info'], height=40)
            header.pack(fill='x')
            header.pack_propagate(False)
            
            tk.Label(
                header,
                text=group_name,
                font=('Segoe UI', 11, 'bold'),
                bg=COLORS['info'],
                fg=COLORS['white']
            ).pack(side='left', padx=15, pady=8)
            
            # 그룹 버튼들
            btn_container = tk.Frame(header, bg=COLORS['info'])
            btn_container.pack(side='right', padx=10)
            
            group_vars = []
            
            def make_toggle_func(vars_list, value):
                return lambda: [v.set(value) for v in vars_list]
            
            def make_auto_match_func(trend_names, group_title):
                def auto_match():
                    if not self.df_columns:
                        messagebox.showwarning("경고", "먼저 CSV 파일을 선택해주세요!")
                        return
                    
                    select_window = tk.Toplevel(self.root)
                    select_window.title(f"{group_title} X축 선택")
                    select_window.geometry("500x250")
                    select_window.configure(bg=COLORS['bg'])
                    select_window.transient(self.root)
                    select_window.grab_set()
                    
                    select_window.update_idletasks()
                    x = (select_window.winfo_screenwidth() // 2) - (500 // 2)
                    y = (select_window.winfo_screenheight() // 2) - (250 // 2)
                    select_window.geometry(f'500x250+{x}+{y}')
                    
                    selected_col = tk.StringVar()
                    
                    frame = tk.Frame(select_window, bg=COLORS['white'])
                    frame.pack(fill='both', expand=True, padx=30, pady=30)
                    
                    tk.Label(
                        frame,
                        text=f"{group_title}의 X축(시간) 컬럼 선택",
                        font=('Segoe UI', 12, 'bold'),
                        bg=COLORS['white'],
                        fg=COLORS['primary']
                    ).pack(pady=15)
                    
                    col_menu = ttk.Combobox(
                        frame,
                        textvariable=selected_col,
                        values=self.df_columns,
                        font=('Segoe UI', 10),
                        width=50,
                        state='readonly'
                    )
                    col_menu.pack(pady=10)
                    
                    time_keywords = ['날짜', 'date', 'time', '시간', 'datetime', 'timestamp']
                    for col in self.df_columns:
                        if any(kw in col.lower() for kw in time_keywords):
                            selected_col.set(col)
                            break
                    
                    def confirm():
                        time_col = selected_col.get()
                        if not time_col:
                            messagebox.showwarning("경고", "X축 컬럼을 선택해주세요!")
                            return
                        
                        select_window.destroy()
                        matched = 0
                        for tn in trend_names:
                            wd = self.trend_widgets[tn]
                            y_col = None
                            for col in self.df_columns:
                                if tn in col or col in tn:
                                    y_col = col
                                    break
                            if y_col:
                                wd['x_var'].set(time_col)
                                wd['y_var'].set(y_col)
                                wd['enabled'].set(True)
                                matched += 1
                        messagebox.showinfo("완료", f"{group_title}: {matched}개 매칭")
                    
                    btn_frame = tk.Frame(frame, bg=COLORS['white'])
                    btn_frame.pack(pady=20)
                    
                    StyledButton(btn_frame, text="확인", command=confirm, style='success', width=12).pack(side='left', padx=5)
                    StyledButton(btn_frame, text="취소", command=select_window.destroy, style='danger', width=12).pack(side='left', padx=5)
                
                return auto_match
            
            small_btn_style = {'font': ('Segoe UI', 8), 'padx': 10, 'pady': 5}
            
            btn1 = tk.Button(btn_container, text="전체선택", bg=COLORS['success'], fg=COLORS['white'],
                            relief='flat', cursor='hand2', **small_btn_style)
            btn1.pack(side='left', padx=2)
            
            btn2 = tk.Button(btn_container, text="전체해제", bg=COLORS['danger'], fg=COLORS['white'],
                            relief='flat', cursor='hand2', **small_btn_style)
            btn2.pack(side='left', padx=2)
            
            btn3 = tk.Button(btn_container, text="자동매칭", bg=COLORS['accent'], fg=COLORS['white'],
                            relief='flat', cursor='hand2', **small_btn_style)
            btn3.pack(side='left', padx=2)
            
            # 내용 프레임
            content = tk.Frame(group_card, bg=COLORS['white'], padx=15, pady=15)
            content.pack(fill='both', expand=True)
            
            # 헤더 행
            tk.Label(content, text="사용", font=('Segoe UI', 9, 'bold'), bg=COLORS['white'], width=6).grid(row=0, column=0, padx=5, pady=8)
            tk.Label(content, text="트렌드 이름", font=('Segoe UI', 9, 'bold'), bg=COLORS['white'], width=32, anchor='w').grid(row=0, column=1, sticky='w', padx=5, pady=8)
            tk.Label(content, text="X축 (시간)", font=('Segoe UI', 9, 'bold'), bg=COLORS['white'], width=32, anchor='w').grid(row=0, column=2, padx=5, pady=8)
            tk.Label(content, text="Y축 (값)", font=('Segoe UI', 9, 'bold'), bg=COLORS['white'], width=32, anchor='w').grid(row=0, column=3, padx=5, pady=8)
            
            # 각 트렌드
            for idx, trend_name in enumerate(trend_list):
                row = idx + 1
                
                var = tk.BooleanVar(value=False)
                cb = tk.Checkbutton(content, variable=var, bg=COLORS['white'], activebackground=COLORS['white'])
                cb.grid(row=row, column=0, padx=5, pady=5)
                group_vars.append(var)
                
                tk.Label(content, text=trend_name, font=('Segoe UI', 9), bg=COLORS['white'], anchor='w', width=32).grid(row=row, column=1, sticky='w', padx=5, pady=5)
                
                x_var = tk.StringVar()
                x_menu = ttk.Combobox(content, textvariable=x_var, state='readonly', width=30, font=('Segoe UI', 9))
                x_menu.grid(row=row, column=2, padx=5, pady=5, sticky='ew')
                
                y_var = tk.StringVar()
                y_menu = ttk.Combobox(content, textvariable=y_var, state='readonly', width=30, font=('Segoe UI', 9))
                y_menu.grid(row=row, column=3, padx=5, pady=5, sticky='ew')
                
                self.trend_widgets[trend_name] = {
                    'name': trend_name,
                    'enabled': var,
                    'x_var': x_var,
                    'x_menu': x_menu,
                    'y_var': y_var,
                    'y_menu': y_menu
                }
            
            content.grid_columnconfigure(2, weight=1)
            content.grid_columnconfigure(3, weight=1)
            
            btn1.config(command=make_toggle_func(group_vars, True))
            btn2.config(command=make_toggle_func(group_vars, False))
            btn3.config(command=make_auto_match_func(trend_list, group_name))
        
        canvas.pack(side='top', fill='both', expand=True)
        scrollbar_v.pack(side='right', fill='y')
        scrollbar_h.pack(side='bottom', fill='x')
        
        # 전체 자동 매칭 버튼
        auto_all_frame = tk.Frame(self.step2_5_content, bg=COLORS['white'])
        auto_all_frame.pack(fill='x', pady=15)
        
        StyledButton(
            auto_all_frame,
            text="🔄 전체 자동 매칭",
            command=self.auto_match_all_trends,
            style='accent',
            width=20
        ).pack()
        
        tk.Label(
            auto_all_frame,
            text="※ 트렌드 이름과 일치하는 Y축 + 날짜/시간 X축을 자동 선택",
            font=('Segoe UI', 9),
            fg=COLORS['border'],
            bg=COLORS['white']
        ).pack(pady=5)
    
    def create_step3_widgets(self):
        """그래프 스타일 설정 위젯"""
        # 제목
        title_frame = tk.Frame(self.step3_content, bg=COLORS['white'])
        title_frame.pack(fill='x', pady=10)
        
        tk.Label(
            title_frame,
            text="그래프 제목:",
            font=('Segoe UI', 10, 'bold'),
            bg=COLORS['white'],
            width=15,
            anchor='w'
        ).pack(side='left', padx=5)
        
        self.title_var = tk.StringVar(value="사용자 정의 그래프")
        title_entry = tk.Entry(
            title_frame,
            textvariable=self.title_var,
            font=('Segoe UI', 10),
            relief='flat',
            bg=COLORS['light']
        )
        title_entry.pack(side='left', fill='x', expand=True, padx=5)
        
        # 라인 스타일들
        styles = [
            ("실제값", 'actual', '#1f77b4', 'Solid'),
            ("HUBROOM", 'hubroom', '#9467bd', 'Solid'),
            ("예측값1", 'predicted1', '#ff7f0e', 'Dash'),
            ("예측값2", 'predicted2', '#2ca02c', 'Dot'),
            ("예측값3", 'predicted3', '#d62728', 'Dash')
        ]
        
        for label, key, default_color, default_style in styles:
            self.create_style_row(label, key, default_color, default_style)
        
        # 리미트선
        limit_frame = tk.Frame(self.step3_content, bg=COLORS['white'])
        limit_frame.pack(fill='x', pady=10)
        
        tk.Label(
            limit_frame,
            text="🔴 리미트선:",
            font=('Segoe UI', 10, 'bold'),
            bg=COLORS['white'],
            fg=COLORS['danger'],
            width=15,
            anchor='w'
        ).pack(side='left', padx=5)
        
        tk.Label(
            limit_frame,
            text="값:",
            font=('Segoe UI', 10),
            bg=COLORS['white']
        ).pack(side='left', padx=5)
        
        self.limit_value_var = tk.StringVar(value="300")
        tk.Entry(
            limit_frame,
            textvariable=self.limit_value_var,
            font=('Segoe UI', 10),
            width=15,
            relief='flat',
            bg=COLORS['light']
        ).pack(side='left', padx=5)
    
    def create_style_row(self, label, key, default_color, default_style):
        """스타일 설정 행 생성"""
        row_frame = tk.Frame(self.step3_content, bg=COLORS['white'])
        row_frame.pack(fill='x', pady=8)
        
        tk.Label(
            row_frame,
            text=f"{label}:",
            font=('Segoe UI', 10, 'bold'),
            bg=COLORS['white'],
            width=15,
            anchor='w'
        ).pack(side='left', padx=5)
        
        # 색상 변수와 버튼
        color_var = tk.StringVar(value=default_color)
        setattr(self, f'{key}_color_var', color_var)
        
        color_btn = tk.Button(
            row_frame,
            text="  색상 선택  ",
            command=lambda: self.choose_color(color_var, color_btn),
            bg=default_color,
            fg=COLORS['white'],
            font=('Segoe UI', 9, 'bold'),
            relief='flat',
            cursor='hand2',
            padx=15,
            pady=5
        )
        color_btn.pack(side='left', padx=5)
        setattr(self, f'{key}_color_btn', color_btn)
        
        # 스타일 선택
        style_var = tk.StringVar(value=default_style)
        setattr(self, f'{key}_style_var', style_var)
        
        style_menu = ttk.Combobox(
            row_frame,
            textvariable=style_var,
            values=['Solid', 'Dash', 'Dot'],
            state='readonly',
            width=15,
            font=('Segoe UI', 9)
        )
        style_menu.pack(side='left', padx=5)
    
    def create_step4_widgets(self):
        """그래프 생성 버튼"""
        StyledButton(
            self.step4_content,
            text="🚀 그래프 생성 실행",
            command=self.generate_graph,
            style='success',
            font=('Segoe UI', 14, 'bold'),
            width=30
        ).pack(pady=20)
        
        tk.Label(
            self.step4_content,
            text="모든 설정을 완료한 후 버튼을 클릭하세요",
            font=('Segoe UI', 10),
            fg=COLORS['border'],
            bg=COLORS['white']
        ).pack()
    
    def open_performance_window(self):
        perf_window = PerformanceWindow(self.root)
        self.root.wait_window(perf_window.window)
        self.perf_values = perf_window.values
    
    def activate_section(self, section_num):
        """섹션 활성화"""
        sections = [
            (self.step2_header, self.step2_content),
            (self.step2_5_header, self.step2_5_content),
            (self.step3_header, self.step3_content),
            (self.step4_header, self.step4_content)
        ]
        
        if section_num <= len(sections):
            header, content = sections[section_num - 1]
            header.config(bg=COLORS['secondary'])
            for child in header.winfo_children():
                if isinstance(child, tk.Label):
                    child.config(bg=COLORS['secondary'])

    def choose_color(self, color_var, button):
        color_code = colorchooser.askcolor(title="색상 선택")[1]
        if color_code:
            color_var.set(color_code)
            button.config(bg=color_code)
    
    def auto_match_all_trends(self):
        """전체 트렌드 자동 매칭"""
        if not self.df_columns:
            messagebox.showwarning("경고", "먼저 CSV 파일을 선택해주세요!")
            return
        
        select_window = tk.Toplevel(self.root)
        select_window.title("X축(시간) 컬럼 선택")
        select_window.geometry("500x250")
        select_window.configure(bg=COLORS['bg'])
        select_window.transient(self.root)
        select_window.grab_set()
        
        select_window.update_idletasks()
        x = (select_window.winfo_screenwidth() // 2) - (500 // 2)
        y = (select_window.winfo_screenheight() // 2) - (250 // 2)
        select_window.geometry(f'500x250+{x}+{y}')
        
        selected_col = tk.StringVar()
        
        frame = tk.Frame(select_window, bg=COLORS['white'])
        frame.pack(fill='both', expand=True, padx=30, pady=30)
        
        tk.Label(
            frame,
            text="모든 트렌드의 X축(시간) 컬럼 선택",
            font=('Segoe UI', 12, 'bold'),
            bg=COLORS['white'],
            fg=COLORS['primary']
        ).pack(pady=15)
        
        col_menu = ttk.Combobox(
            frame,
            textvariable=selected_col,
            values=self.df_columns,
            font=('Segoe UI', 10),
            width=50,
            state='readonly'
        )
        col_menu.pack(pady=10)
        
        time_keywords = ['날짜', 'date', 'time', '시간', 'datetime', 'timestamp']
        for col in self.df_columns:
            if any(kw in col.lower() for kw in time_keywords):
                selected_col.set(col)
                break
        
        def confirm():
            time_col = selected_col.get()
            if not time_col:
                messagebox.showwarning("경고", "X축 컬럼을 선택해주세요!")
                return
            
            select_window.destroy()
            matched = 0
            for trend_name, wd in self.trend_widgets.items():
                y_col = None
                for col in self.df_columns:
                    if trend_name in col or col in trend_name:
                        y_col = col
                        break
                if y_col:
                    wd['x_var'].set(time_col)
                    wd['y_var'].set(y_col)
                    wd['enabled'].set(True)
                    matched += 1
            messagebox.showinfo("완료", f"총 {matched}개 트렌드 자동 매칭 완료\nX축: {time_col}")
        
        btn_frame = tk.Frame(frame, bg=COLORS['white'])
        btn_frame.pack(pady=20)
        
        StyledButton(btn_frame, text="확인", command=confirm, style='success', width=12).pack(side='left', padx=5)
        StyledButton(btn_frame, text="취소", command=select_window.destroy, style='danger', width=12).pack(side='left', padx=5)

    def select_file(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not path:
            return
        
        self.file_path = path
        filename = os.path.basename(path)
        self.file_label.config(text=f"✓ {filename}", fg=COLORS['success'])
        
        try:
            df = read_csv_safe(self.file_path)
            self.df_columns = df.columns.tolist()
            
            # 메인 데이터 컬럼 업데이트
            for i, menu in enumerate(self.column_menus):
                menu['values'] = self.df_columns
            
            # 트렌드 데이터 컬럼 업데이트
            for wd in self.trend_widgets.values():
                wd['x_menu']['values'] = self.df_columns
                wd['y_menu']['values'] = self.df_columns
            
            # 자동 추천
            patterns = [
                ['날짜', 'date', 'time'],
                ['실제값', 'actual', 'real'],
                ['hubroom', 'hub'],
                ['hubroom', 'hub'],
                ['예측날짜', 'pred_date', 'forecast'],
                ['예측값', 'predicted', 'pred'],
                ['예측날짜', 'pred_date', 'forecast', '2'],
                ['예측값', 'predicted', 'pred', '2'],
                ['예측날짜', 'pred_date', 'forecast', '3'],
                ['예측값', 'predicted', 'pred', '3']
            ]
            
            for idx, pattern_list in enumerate(patterns):
                for col in self.df_columns:
                    if any(p in col.lower() for p in pattern_list):
                        self.column_vars[idx].set(col)
                        break

            # 섹션 활성화
            for i in range(1, 5):
                self.activate_section(i)
            
            messagebox.showinfo("성공", f"파일이 로드되었습니다!\n\n컬럼 수: {len(self.df_columns)}")

        except Exception as e:
            messagebox.showerror("오류", f"CSV 파일 읽기 실패:\n{e}")

    def generate_graph(self):
        selected_trends = []
        for wd in self.trend_widgets.values():
            if wd['enabled'].get():
                x_col = wd['x_var'].get()
                y_col = wd['y_var'].get()
                if x_col and y_col:
                    selected_trends.append({
                        'name': wd['name'],
                        'x_col': x_col,
                        'y_col': y_col
                    })
        
        params = {
            'file_path': self.file_path,
            'title': self.title_var.get(),
            'actual_x': self.column_vars[0].get() if self.enable_actual.get() else '',
            'actual_y': self.column_vars[1].get() if self.enable_actual.get() else '',
            'hubroom_x': self.column_vars[2].get() if self.enable_hubroom.get() else '',
            'hubroom_y': self.column_vars[3].get() if self.enable_hubroom.get() else '',
            'predicted1_x': self.column_vars[4].get() if self.enable_predicted1.get() else '',
            'predicted1_y': self.column_vars[5].get() if self.enable_predicted1.get() else '',
            'predicted2_x': self.column_vars[6].get() if self.enable_predicted2.get() else '',
            'predicted2_y': self.column_vars[7].get() if self.enable_predicted2.get() else '',
            'predicted3_x': self.column_vars[8].get() if self.enable_predicted3.get() else '',
            'predicted3_y': self.column_vars[9].get() if self.enable_predicted3.get() else '',
            'actual_color': self.actual_color_var.get(),
            'actual_style': self.actual_style_var.get(),
            'hubroom_color': self.hubroom_color_var.get(),
            'hubroom_style': self.hubroom_style_var.get(),
            'predicted1_color': self.predicted1_color_var.get(),
            'predicted1_style': self.predicted1_style_var.get(),
            'predicted2_color': self.predicted2_color_var.get(),
            'predicted2_style': self.predicted2_style_var.get(),
            'predicted3_color': self.predicted3_color_var.get(),
            'predicted3_style': self.predicted3_style_var.get(),
            'limit_value': self.limit_value_var.get(),
            'trend_data': selected_trends,
            'enable_actual': self.enable_actual.get(),
            'enable_hubroom': self.enable_hubroom.get(),
            'enable_predicted1': self.enable_predicted1.get(),
            'enable_predicted2': self.enable_predicted2.get(),
            'enable_predicted3': self.enable_predicted3.get()
        }
        
        enabled_count = sum([
            self.enable_actual.get(),
            self.enable_hubroom.get(),
            self.enable_predicted1.get(),
            self.enable_predicted2.get(),
            self.enable_predicted3.get()
        ])
        
        if enabled_count == 0:
            messagebox.showwarning("경고", "최소 1개 이상의 메인 데이터를 선택해주세요.")
            return
        
        missing = []
        if self.enable_actual.get() and (not params['actual_x'] or not params['actual_y']):
            missing.append("실제값")
        if self.enable_hubroom.get() and (not params['hubroom_x'] or not params['hubroom_y']):
            missing.append("HUBROOM")
        if self.enable_predicted1.get() and (not params['predicted1_x'] or not params['predicted1_y']):
            missing.append("예측값1")
        if self.enable_predicted2.get() and (not params['predicted2_x'] or not params['predicted2_y']):
            missing.append("예측값2")
        if self.enable_predicted3.get() and (not params['predicted3_x'] or not params['predicted3_y']):
            missing.append("예측값3")
        
        if missing:
            messagebox.showwarning("경고", f"다음 항목의 컬럼을 선택해주세요:\n{', '.join(missing)}")
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