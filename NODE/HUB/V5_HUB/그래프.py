import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser
import pandas as pd
import plotly.graph_objects as go
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
# 그래프 생성 로직 (INFO 정보창, 리미트선, 점프예측, 패턴예측 포함)
# --------------------------------------------------------------------------
def create_graph(params):
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

        # 실제값 라인 - 점프예측, 패턴예측 정보 포함
        fig.add_trace(go.Scattergl(
            x=df[actual_x_col], 
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
        
        # 예측값 라인 - 점프예측, 패턴예측 정보 포함
        fig.add_trace(go.Scattergl(
            x=df[predicted_x_col], 
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
            # X축의 전체 범위 구하기
            all_x = pd.concat([df[actual_x_col], df[predicted_x_col]]).sort_values()
            
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

        output_filename = "final_guided_graph.html"
        fig.write_html(output_filename, include_plotlyjs='cdn')
        webbrowser.open('file://' + os.path.realpath(output_filename))
        
        messagebox.showinfo("성공", 
            f"'{output_filename}' 파일이 생성되었습니다.\n\n"
            f"✅ 색상이 추가된 INFO 정보창\n"
            f"✅ 점프예측 정보 표시\n"
            f"✅ 패턴예측 정보 표시\n"
            f"✅ 빨간색 리미트선 표시\n"
            f"✅ 격자선 유지\n"
            f"✅ 인코딩 자동 감지")
            
    except Exception as e:
        messagebox.showerror("오류 발생", f"그래프 생성 중 오류가 발생했습니다:\n{e}")

# --------------------------------------------------------------------------
# GUI 애플리케이션 로직 (리미트선 설정 추가)
# --------------------------------------------------------------------------
class GraphApp:
    def __init__(self, root):
        self.root = root
        self.root.title("단계별 그래프 생성기 v3.0 (패턴예측 포함)")
        self.file_path = ""
        self.df_columns = []

        # -- 단계별 프레임 생성 --
        self.step1_frame = tk.LabelFrame(root, text="✅ 1단계: 파일 선택", padx=10, pady=10)
        self.step1_frame.pack(fill='x', padx=10, pady=5)
        
        self.step2_frame = tk.LabelFrame(root, text="🔒 2단계: 데이터 컬럼 선택", padx=10, pady=10)
        self.step2_frame.pack(fill='x', padx=10, pady=5)

        self.step3_frame = tk.LabelFrame(root, text="🔒 3단계: 그래프 스타일 설정", padx=10, pady=10)
        self.step3_frame.pack(fill='x', padx=10, pady=5)
        
        self.step4_frame = tk.LabelFrame(root, text="🔒 4단계: 그래프 생성", padx=10, pady=10)
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
            
        create_graph(params)

if __name__ == "__main__":
    root = tk.Tk()
    app = GraphApp(root)
    root.mainloop()