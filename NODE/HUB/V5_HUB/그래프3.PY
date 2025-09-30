import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser, scrolledtext
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
        
        # 값들을 저장할 변수 초기화
        self.values = None
        
        # 메인 프레임
        main_frame = tk.Frame(self.window)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # 상단 프레임 - 성능 지표 입력
        top_frame = tk.LabelFrame(main_frame, text="📊 전체 성능 지표 입력", padx=10, pady=10)
        top_frame.pack(fill='x', pady=(0, 10))
        
        # 성능 지표 입력 필드들
        input_frame = tk.Frame(top_frame)
        input_frame.pack(pady=10)
        
        # MAE 입력
        tk.Label(input_frame, text="MAE:", font=('Arial', 10)).grid(row=0, column=0, sticky='e', padx=5, pady=5)
        self.mae_var = tk.StringVar(value="30.23")
        tk.Entry(input_frame, textvariable=self.mae_var, width=15).grid(row=0, column=1, padx=5, pady=5)
        
        # RMSE 입력
        tk.Label(input_frame, text="RMSE:", font=('Arial', 10)).grid(row=1, column=0, sticky='e', padx=5, pady=5)
        self.rmse_var = tk.StringVar(value="36.02")
        tk.Entry(input_frame, textvariable=self.rmse_var, width=15).grid(row=1, column=1, padx=5, pady=5)
        
        # R² 입력
        tk.Label(input_frame, text="R²:", font=('Arial', 10)).grid(row=2, column=0, sticky='e', padx=5, pady=5)
        self.r2_var = tk.StringVar(value="0.72")
        tk.Entry(input_frame, textvariable=self.r2_var, width=15).grid(row=2, column=1, padx=5, pady=5)
        
        # 하단 프레임 - 기타 정보
        bottom_frame = tk.LabelFrame(main_frame, text="📝 기타 정보 (붙여넣기 가능)", padx=10, pady=10)
        bottom_frame.pack(fill='both', expand=True)
        
        # 스크롤 가능한 텍스트 위젯
        self.text_widget = scrolledtext.ScrolledText(bottom_frame, height=20, width=70, wrap=tk.WORD)
        self.text_widget.pack(fill='both', expand=True)
        
        # 샘플 텍스트
        self.text_widget.insert('1.0', """📊 점프 케이스 감지 현황:
  - 실제 점프 케이스: 66개
  - 감지된 점프 케이스: 51개
  - 감지율: 77.3%

================================================================================
📊 평가 결과 분석
================================================================================

📈 전체 성능:
  - MAE: 30.23
  - RMSE: 36.02
  - R²: 0.72
  - 총 예측 수: 34518

🎯 극단값(300+) 감지:
  - 실제 300+ 케이스: 7318개
  - 감지된 케이스: 7222개
  - 감지율: 98.7%

🚀 점프 케이스 (과거<280 → 실제300+):
  - 총 점프 케이스: 66개
  - 감지된 케이스: 51개
  - 점프 감지율: 77.3%
  
📊 점프 케이스 상세:
  - 2025-08-08 09:33: 실제 303, 예측 307.19, 점프예측 O (확률 0.2%)
  - 2025-08-08 09:36: 실제 316, 예측 309.0, 점프예측 O (확률 0.1%)
  - 2025-08-08 09:37: 실제 307, 예측 313.85, 점프예측 O (확률 0.3%)
  - 2025-08-08 09:38: 실제 308, 예측 317.05, 점프예측 O (확률 0.2%)
  - 2025-08-08 09:39: 실제 306, 예측 315.86, 점프예측 O (확률 0.1%)

⚠️ False Positive:
  - 실제 <300 케이스: 27200개
  - 잘못 예측된 케이스: 5082개
  - FP Rate: 18.7%

📊 3구간 분류 정확도: 94.5%

📈 패턴 예측 분포:
  - 하락: 15301개 (44.3%)
  - 안정: 10594개 (30.7%)
  - 점진상승: 7562개 (21.9%)
  - 급상승: 1061개 (3.1%)""")
        
        # 확인 버튼
        tk.Button(main_frame, text="확인", 
                 command=self.save_and_close,
                 bg='#3498DB', fg='white',
                 font=('Arial', 10, 'bold')).pack(pady=10)
        
    def save_and_close(self):
        """값을 저장하고 창을 닫기"""
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
        # 인코딩 안전하게 읽기
        df = read_csv_safe(params['file_path'])
        
        actual_x_col = params['actual_x']
        actual_y_col = params['actual_y']
        predicted_x_col = params['predicted_x']
        predicted_y_col = params['predicted_y']
        
        # Y값을 숫자로 변환
        df[actual_y_col] = pd.to_numeric(df[actual_y_col], errors='coerce')
        df[predicted_y_col] = pd.to_numeric(df[predicted_y_col], errors='coerce')
        
        # NaN 제거
        df = df.dropna(subset=[actual_y_col, predicted_y_col])
        
        # 점프예측 컬럼 확인
        jump_col = None
        for col in df.columns:
            if '점프예측' in col or 'jump' in col.lower():
                jump_col = col
                break
        
        # 패턴예측 컬럼 확인
        pattern_col = None
        for col in df.columns:
            if '패턴예측' in col or 'pattern' in col.lower():
                pattern_col = col
                break

        # 날짜 변환 및 포맷팅
        try:
            df[actual_x_col] = pd.to_datetime(df[actual_x_col])
            df['actual_time_str'] = df[actual_x_col].dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            df['actual_time_str'] = df[actual_x_col].astype(str)
            
        try:
            df[predicted_x_col] = pd.to_datetime(df[predicted_x_col])
            df['predicted_time_str'] = df[predicted_x_col].dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            df['predicted_time_str'] = df[predicted_x_col].astype(str)
        
        # ⭐ 핵심 변경: 실제값과 예측값을 같은 X축(시간)에 표시
        # 실제값의 시간을 공통 시간축으로 사용
        df['common_time'] = df[actual_x_col]
        
        # 점프예측 값 준비 및 색상 결정
        if jump_col:
            df['jump_value'] = df[jump_col].astype(str)
            df['jump_color'] = df[jump_col].apply(
                lambda x: '#27AE60' if str(x).upper() == 'O' else '#E74C3C' if str(x).upper() == 'X' else '#666'
            )
        else:
            df['jump_value'] = 'N/A'
            df['jump_color'] = '#666'
        
        # 패턴예측 값 준비 및 색상 결정
        if pattern_col:
            df['pattern_value'] = df[pattern_col].astype(str)
            # 패턴예측 값에 따른 색상 설정
            def get_pattern_color(val):
                val_str = str(val).strip()
                if '상승' in val_str or '증가' in val_str:
                    return '#27AE60'  # 초록색 - 상승
                elif '하락' in val_str or '감소' in val_str:
                    return '#E74C3C'  # 빨간색 - 하락
                elif '안정' in val_str or '유지' in val_str:
                    return '#3498DB'  # 파란색 - 안정
                else:
                    return '#8E44AD'  # 보라색 - 기타
            
            df['pattern_color'] = df[pattern_col].apply(get_pattern_color)
        else:
            df['pattern_value'] = 'N/A'
            df['pattern_color'] = '#95A5A6'

        fig = go.Figure()

        # ⭐ 실제값 라인 - 같은 시간축(common_time) 사용
        fig.add_trace(go.Scattergl(
            x=df['common_time'],  # 공통 시간축 사용
            y=df[actual_y_col], 
            mode='lines+markers',
            name='실제값 (Actual)',
            line=dict(color=params['actual_color'], 
                     dash=None if params['actual_style'] == 'Solid' else params['actual_style'].lower(), 
                     width=2),
            marker=dict(size=5),
            customdata=df[[actual_y_col, 'actual_time_str', predicted_y_col, 'predicted_time_str', 
                          'jump_value', 'jump_color', 'pattern_value', 'pattern_color']].values,
            hovertemplate='<b style="color: #2E86C1; font-size: 14px;">📊 INFO 정보</b><br>' +
                         '<span style="color: #85C1E2;">═══════════════════</span><br>' +
                         '<b style="color: #1f77b4;">🔵 실제값</b><br>' +
                         '<span style="color: #666;">날짜:</span> <span style="color: #000;">%{customdata[1]}</span><br>' +
                         '<span style="color: #666;">실제값:</span> <b style="color: #1f77b4;">%{customdata[0]:.2f}</b><br>' +
                         '<span style="color: #85C1E2;">═══════════════════</span><br>' +
                         '<b style="color: #ff7f0e;">🔶 예측값</b><br>' + 
                         '<span style="color: #666;">예측날짜:</span> <span style="color: #000;">%{customdata[3]}</span><br>' +
                         '<span style="color: #666;">예측값:</span> <b style="color: #ff7f0e;">%{customdata[2]:.2f}</b><br>' +
                         '<span style="color: #85C1E2;">═══════════════════</span><br>' +
                         '<span style="color: #666;">점프예측:</span> <b style="color: %{customdata[5]};">%{customdata[4]}</b><br>' +
                         '<span style="color: #666;">패턴예측:</span> <b style="color: %{customdata[7]};">%{customdata[6]}</b>' +
                         '<extra></extra>'
        ))
        
        # ⭐ 예측값 라인 - 같은 시간축(common_time) 사용
        fig.add_trace(go.Scattergl(
            x=df['common_time'],  # 공통 시간축 사용
            y=df[predicted_y_col], 
            mode='lines+markers',
            name='예측값 (Predicted)',
            line=dict(color=params['predicted_color'], 
                     dash=None if params['predicted_style'] == 'Solid' else params['predicted_style'].lower(), 
                     width=2),
            marker=dict(size=5),
            customdata=df[[actual_y_col, 'actual_time_str', predicted_y_col, 'predicted_time_str', 
                          'jump_value', 'jump_color', 'pattern_value', 'pattern_color']].values,
            hovertemplate='<b style="color: #2E86C1; font-size: 14px;">📊 INFO 정보</b><br>' +
                         '<span style="color: #85C1E2;">═══════════════════</span><br>' +
                         '<b style="color: #1f77b4;">🔵 실제값</b><br>' +
                         '<span style="color: #666;">날짜:</span> <span style="color: #000;">%{customdata[1]}</span><br>' +
                         '<span style="color: #666;">실제값:</span> <b style="color: #1f77b4;">%{customdata[0]:.2f}</b><br>' +
                         '<span style="color: #85C1E2;">═══════════════════</span><br>' +
                         '<b style="color: #ff7f0e;">🔶 예측값</b><br>' + 
                         '<span style="color: #666;">예측날짜:</span> <span style="color: #000;">%{customdata[3]}</span><br>' +
                         '<span style="color: #666;">예측값:</span> <b style="color: #ff7f0e;">%{customdata[2]:.2f}</b><br>' +
                         '<span style="color: #85C1E2;">═══════════════════</span><br>' +
                         '<span style="color: #666;">점프예측:</span> <b style="color: %{customdata[5]};">%{customdata[4]}</b><br>' +
                         '<span style="color: #666;">패턴예측:</span> <b style="color: %{customdata[7]};">%{customdata[6]}</b>' +
                         '<extra></extra>'
        ))
        
        # 리미트선 추가
        try:
            limit_value = float(params.get('limit_value', 300))
            # X축의 전체 범위 구하기 (common_time 기준)
            all_x = df['common_time'].sort_values()
            
            # 리미트선을 여러 포인트로 생성 (호버 기능 향상)
            limit_x = pd.date_range(start=all_x.iloc[0], end=all_x.iloc[-1], periods=100)
            limit_y = [limit_value] * 100
            
            fig.add_trace(go.Scatter(
                x=limit_x,
                y=limit_y,
                mode='lines',
                name=f'리미트선 ({limit_value})',
                line=dict(
                    color='red',
                    width=2.5
                ),
                hovertemplate='<b style="color: red;">⚠️ 리미트선</b><br>' +
                             f'<span style="color: red;">설정값: {limit_value:.2f}</span><br>' +
                             '<span style="color: #666;">시간: %{x}</span>' +
                             '<extra></extra>'
            ))
        except:
            pass  # 리미트값이 올바르지 않으면 무시
        
        # 레이아웃 업데이트
        fig.update_layout(
            title=params['title'], 
            xaxis_title='시간', 
            yaxis_title='값', 
            hovermode='closest',
            plot_bgcolor='white',
            xaxis=dict(showgrid=True, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridcolor='lightgray'),
            showlegend=True,
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial",
                bordercolor="#2E86C1"
            )
        )

        # 메인 그래프 HTML 생성
        output_filename = "final_guided_graph.html"
        html_content = fig.to_html(include_plotlyjs='cdn')

        # 성능 정보 HTML 섹션 생성
        if perf_values:
            other_info = perf_values.get('other_info', '정보가 입력되지 않았습니다.')
            # HTML 특수문자 이스케이프 방지를 위해 그대로 사용
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
            performance_section = """
            <div style="margin: 30px auto; max-width: 1200px; padding: 0 20px;">
                <div style="background-color: #fff3cd; padding: 20px; border-radius: 10px; border: 2px solid #ffc107;">
                    <p style="text-align: center; color: #856404; margin: 0;">
                        ⚠️ 기타 정보가 입력되지 않았습니다. '내용정보 입력' 버튼을 클릭하여 정보를 입력해주세요.
                    </p>
                </div>
            </div>
            """

        # 그래프와 성능 정보를 하나로 통합
        html_content = html_content.replace('</body>', performance_section + '</body>')

        # 스타일 추가 (전체 페이지 배경)
        style_section = """
        <style>
            body {
                background-color: #f8f9fa;
                margin: 0;
                padding: 20px 0;
            }
            .plotly-graph-div {
                margin: 0 auto;
                max-width: 1200px;
            }
        </style>
        """
        html_content = html_content.replace('</head>', style_section + '</head>')

        with open(output_filename, 'w', encoding='utf-8')as f:
            f.write(html_content)

        webbrowser.open('file://' + os.path.realpath(output_filename))

        messagebox.showinfo("성공", 
            f"'{output_filename}' 파일이 생성되었습니다.\n\n"
            f"✅ 실제값과 예측값이 같은 시간축에 겹쳐서 표시됩니다!\n"
            f"✅ 그래프와 성능 정보가 하나의 화면에 통합\n"
            f"✅ 색상이 추가된 INFO 정보창\n"
            f"✅ 점프예측 정보 표시\n"
            f"✅ 패턴예측 정보 표시\n"
            f"✅ 빨간색 리미트선 표시")
            
    except Exception as e:
        messagebox.showerror("오류 발생", f"그래프 생성 중 오류가 발생했습니다:\n{e}")

# --------------------------------------------------------------------------
# GUI 애플리케이션 로직
# --------------------------------------------------------------------------
class GraphApp:
    def __init__(self, root):
        self.root = root
        self.root.title("단계별 그래프 생성기 v3.0 (실제값/예측값 겹침 표시)")
        self.file_path = ""
        self.df_columns = []
        self.perf_values = None  # 성능 값 저장

        # 전체 레이아웃을 위한 메인 프레임
        main_container = tk.Frame(root)
        main_container.pack(fill='both', expand=True)
        
        # 왼쪽 프레임 (기존 내용)
        left_frame = tk.Frame(main_container)
        left_frame.pack(side='left', fill='both', expand=True, padx=(10, 5), pady=10)
        
        # 오른쪽 프레임 (내용정보 버튼)
        right_frame = tk.Frame(main_container)
        right_frame.pack(side='right', fill='y', padx=(5, 10), pady=10)
        
        # 내용정보 버튼
        tk.Button(right_frame, text="📊 내용정보\n입력", 
                 command=self.open_performance_window,
                 font=('Arial', 12, 'bold'),
                 bg='#2ECC71', fg='white',
                 width=12, height=3).pack(pady=10)

        # -- 단계별 프레임 생성 (왼쪽 프레임에) --
        self.step1_frame = tk.LabelFrame(left_frame, text="✅ 1단계: 파일 선택", padx=10, pady=10)
        self.step1_frame.pack(fill='x', padx=10, pady=5)
        
        self.step2_frame = tk.LabelFrame(left_frame, text="🔒 2단계: 데이터 컬럼 선택", padx=10, pady=10)
        self.step2_frame.pack(fill='x', padx=10, pady=5)

        self.step3_frame = tk.LabelFrame(left_frame, text="🔒 3단계: 그래프 스타일 설정", padx=10, pady=10)
        self.step3_frame.pack(fill='x', padx=10, pady=5)
        
        self.step4_frame = tk.LabelFrame(left_frame, text="🔒 4단계: 그래프 생성", padx=10, pady=10)
        self.step4_frame.pack(fill='x', padx=10, pady=5)

        # -- 1단계 위젯 --
        tk.Button(self.step1_frame, text="CSV 파일 열기", command=self.select_file, 
                 font=('Helvetica', 10, 'bold'), bg='#e8f4f8').pack(side='left')
        self.file_label = tk.Label(self.step1_frame, text="선택된 파일이 없습니다.", fg="blue")
        self.file_label.pack(side='left', padx=10)
        
        # -- 2, 3, 4단계 위젯 미리 생성 --
        self.create_step2_widgets()
        self.create_step3_widgets()
        self.create_step4_widgets()
        
        # -- 초기 상태: 2, 3, 4단계 비활성화 --
        self.toggle_widgets_state(self.step2_frame, 'disabled')
        self.toggle_widgets_state(self.step3_frame, 'disabled')
        self.toggle_widgets_state(self.step4_frame, 'disabled')
    
    def open_performance_window(self):
        """성능 정보 창 열기"""
        perf_window = PerformanceWindow(self.root)
        self.root.wait_window(perf_window.window)  # 창이 닫힐 때까지 대기
        self.perf_values = perf_window.values  # values 직접 사용
    
    def toggle_widgets_state(self, frame, state):
        for child in frame.winfo_children():
            try:
                child.config(state=state)
            except tk.TclError:
                pass

    def create_step2_widgets(self):
        labels = ["실제값 X축 (시간):", "실제값 Y축 (값):", "예측값 X축 (시간):", "예측값 Y축 (값):"]
        self.column_vars = [tk.StringVar() for _ in labels]
        self.column_menus = []

        for i, label_text in enumerate(labels):
            tk.Label(self.step2_frame, text=label_text).grid(row=i, column=0, sticky='w', padx=5, pady=2)
            menu = tk.OptionMenu(self.step2_frame, self.column_vars[i], "")
            menu.config(width=30)
            menu.grid(row=i, column=1, sticky='ew', padx=5, pady=2)
            self.column_menus.append(menu)
            
    def create_step3_widgets(self):
        tk.Label(self.step3_frame, text="그래프 제목:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.title_var = tk.StringVar(value="사용자 정의 그래프")
        tk.Entry(self.step3_frame, textvariable=self.title_var, width=50).grid(row=0, column=1, columnspan=3, sticky='ew', padx=5, pady=2)
        
        tk.Label(self.step3_frame, text="[실제값 라인]").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.actual_color_var = tk.StringVar(value='#1f77b4')
        self.actual_color_btn = tk.Button(self.step3_frame, text="색상", 
                                         command=lambda: self.choose_color(self.actual_color_var, self.actual_color_btn), 
                                         bg=self.actual_color_var.get())
        self.actual_color_btn.grid(row=1, column=1, padx=5)
        self.actual_style_var = tk.StringVar(value='Solid')
        tk.OptionMenu(self.step3_frame, self.actual_style_var, 'Solid', 'Dash', 'Dot').grid(row=1, column=2, padx=5)

        tk.Label(self.step3_frame, text="[예측값 라인]").grid(row=2, column=0, sticky='w', padx=5, pady=5)
        self.predicted_color_var = tk.StringVar(value='#ff7f0e')
        self.predicted_color_btn = tk.Button(self.step3_frame, text="색상", 
                                           command=lambda: self.choose_color(self.predicted_color_var, self.predicted_color_btn), 
                                           bg=self.predicted_color_var.get())
        self.predicted_color_btn.grid(row=2, column=1, padx=5)
        self.predicted_style_var = tk.StringVar(value='Dash')
        tk.OptionMenu(self.step3_frame, self.predicted_style_var, 'Solid', 'Dash', 'Dot').grid(row=2, column=2, padx=5)
        
        # 리미트선 설정 추가
        tk.Label(self.step3_frame, text="[🔴 리미트선]").grid(row=3, column=0, sticky='w', padx=5, pady=5)
        tk.Label(self.step3_frame, text="값:").grid(row=3, column=1, sticky='e', padx=(0, 5))
        self.limit_value_var = tk.StringVar(value="300")
        tk.Entry(self.step3_frame, textvariable=self.limit_value_var, width=10).grid(row=3, column=2, sticky='w', padx=5)
        tk.Label(self.step3_frame, text="(빨간색 수평선)").grid(row=3, column=3, sticky='w', padx=5)

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
            # 인코딩 안전하게 읽기
            df = read_csv_safe(self.file_path)
            self.df_columns = df.columns.tolist()
            
            # 컬럼 메뉴 업데이트
            for i, menu in enumerate(self.column_menus):
                menu['menu'].delete(0, 'end')
                for col in self.df_columns:
                    menu['menu'].add_command(label=col, command=tk._setit(self.column_vars[i], col))
            
            # 컬럼 자동 추천 (개선된 패턴 매칭)
            patterns = [
                ['날짜', 'date', 'time'],
                ['실제값', 'actual', 'real'],
                ['예측날짜', 'pred_date', 'forecast'],
                ['예측값', 'predicted', 'pred']
            ]
            
            for idx, pattern_list in enumerate(patterns):
                for col in self.df_columns:
                    col_lower = col.lower()
                    if any(p in col_lower for p in pattern_list):
                        self.column_vars[idx].set(col)
                        break

            # 2, 3, 4단계 활성화
            self.step2_frame.config(text="✅ 2단계: 데이터 컬럼 선택")
            self.toggle_widgets_state(self.step2_frame, 'normal')
            self.step3_frame.config(text="✅ 3단계: 그래프 스타일 설정")
            self.toggle_widgets_state(self.step3_frame, 'normal')
            self.step4_frame.config(text="✅ 4단계: 그래프 생성")
            self.toggle_widgets_state(self.step4_frame, 'normal')

        except Exception as e:
            messagebox.showerror("오류", f"CSV 파일을 읽는 데 실패했습니다:\n{e}")

    def generate_graph(self):
        params = {
            'file_path': self.file_path, 
            'title': self.title_var.get(),
            'actual_x': self.column_vars[0].get(), 
            'actual_y': self.column_vars[1].get(),
            'predicted_x': self.column_vars[2].get(), 
            'predicted_y': self.column_vars[3].get(),
            'actual_color': self.actual_color_var.get(), 
            'actual_style': self.actual_style_var.get(),
            'predicted_color': self.predicted_color_var.get(), 
            'predicted_style': self.predicted_style_var.get(),
            'limit_value': self.limit_value_var.get()  # 리미트값 추가
        }
        
        if not all([params['actual_x'], params['actual_y'], params['predicted_x'], params['predicted_y']]):
            messagebox.showwarning("경고", "2단계에서 모든 데이터 축(X, Y)의 컬럼을 선택해주세요.")
            return
        
        # 성능 값이 없으면 기본값으로 생성
        if not self.perf_values:
            self.perf_values = {
                'mae': '30.23',
                'rmse': '36.02',
                'r2': '0.72',
                'other_info': '기타 정보가 입력되지 않았습니다.\n내용정보 입력 버튼을 클릭하여 정보를 입력해주세요.'
            }
            
        create_graph(params, self.perf_values)

if __name__ == "__main__":
    root = tk.Tk()
    app = GraphApp(root)
    root.mainloop()