import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser
import pandas as pd
import plotly.graph_objects as go
import webbrowser
import os

# --------------------------------------------------------------------------
# 그래프 생성 로직 (이전 버전과 동일)
# --------------------------------------------------------------------------
def create_graph(params):
    try:
        df = pd.read_csv(params['file_path'])
        
        actual_x_col = params['actual_x']
        actual_y_col = params['actual_y']
        predicted_x_col = params['predicted_x']
        predicted_y_col = params['predicted_y']

        try:
            df[actual_x_col] = pd.to_datetime(df[actual_x_col])
            df[predicted_x_col] = pd.to_datetime(df[predicted_x_col])
            df['hover_actual_datetime'] = df[actual_x_col].dt.strftime('%Y-%m-%d %H:%M')
            df['hover_predicted_datetime'] = df[predicted_x_col].dt.strftime('%Y-%m-%d %H:%M')
        except Exception:
            df['hover_actual_datetime'] = df[actual_x_col]
            df['hover_predicted_datetime'] = df[predicted_x_col]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df[actual_x_col], y=df[actual_y_col], mode='lines', name='실제값 (Actual)',
            line=dict(color=params['actual_color'], dash=params['actual_style'].lower()),
            customdata=df[['hover_actual_datetime']],
            hovertemplate=f"<b>{actual_y_col}:</b> %{{y}}<br><b>시간:</b> %{{customdata[0]}}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=df[predicted_x_col], y=df[predicted_y_col], mode='lines', name='예측값 (Predicted)',
            line=dict(color=params['predicted_color'], dash=params['predicted_style'].lower()),
            customdata=df[['hover_predicted_datetime']],
            hovertemplate=f"<b>{predicted_y_col}:</b> %{{y}}<br><b>시간:</b> %{{customdata[0]}}<extra></extra>"
        ))
        
        fig.update_layout(title=params['title'], xaxis_title='X-Axis', yaxis_title='Y-Axis', hovermode='x unified')

        output_filename = "final_guided_graph.html"
        fig.write_html(output_filename)
        webbrowser.open('file://' + os.path.realpath(output_filename))
        messagebox.showinfo("성공", f"'{output_filename}' 파일이 생성되었습니다.")
    except Exception as e:
        messagebox.showerror("오류 발생", f"그래프 생성 중 오류가 발생했습니다:\n{e}")

# --------------------------------------------------------------------------
# GUI 애플리케이션 로직 (✨ 단계별로 비활성화/활성화되도록 수정)
# --------------------------------------------------------------------------
class GraphApp:
    def __init__(self, root):
        self.root = root
        self.root.title("단계별 그래프 생성기 v3.0")
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
        tk.Button(self.step1_frame, text="CSV 파일 열기", command=self.select_file, font=('Helvetica', 10, 'bold')).pack(side='left')
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
                pass # 일부 위젯(Label 등)은 state 속성이 없어 오류 발생 가능

    def create_step2_widgets(self):
        labels = ["실제값 X축 (시간):", "실제값 Y축 (값):", "예측값 X축 (시간):", "예측값 Y축 (값):"]
        self.column_vars = [tk.StringVar() for _ in labels]
        self.column_menus = []

        for i, label_text in enumerate(labels):
            tk.Label(self.step2_frame, text=label_text).grid(row=i, column=0, sticky='w', padx=5, pady=2)
            menu = tk.OptionMenu(self.step2_frame, self.column_vars[i], "")
            menu.grid(row=i, column=1, sticky='ew', padx=5, pady=2)
            self.column_menus.append(menu)
            
    def create_step3_widgets(self):
        tk.Label(self.step3_frame, text="그래프 제목:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.title_var = tk.StringVar(value="사용자 정의 그래프")
        tk.Entry(self.step3_frame, textvariable=self.title_var, width=50).grid(row=0, column=1, columnspan=3, sticky='ew', padx=5, pady=2)
        
        tk.Label(self.step3_frame, text="[실제값 라인]").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.actual_color_var = tk.StringVar(value='#1f77b4')
        self.actual_color_btn = tk.Button(self.step3_frame, text="색상", command=lambda: self.choose_color(self.actual_color_var, self.actual_color_btn), bg=self.actual_color_var.get())
        self.actual_color_btn.grid(row=1, column=1, padx=5)
        self.actual_style_var = tk.StringVar(value='Solid')
        tk.OptionMenu(self.step3_frame, self.actual_style_var, 'Solid', 'Dash', 'Dot').grid(row=1, column=2, padx=5)

        tk.Label(self.step3_frame, text="[예측값 라인]").grid(row=2, column=0, sticky='w', padx=5, pady=5)
        self.predicted_color_var = tk.StringVar(value='#ff7f0e')
        self.predicted_color_btn = tk.Button(self.step3_frame, text="색상", command=lambda: self.choose_color(self.predicted_color_var, self.predicted_color_btn), bg=self.predicted_color_var.get())
        self.predicted_color_btn.grid(row=2, column=1, padx=5)
        self.predicted_style_var = tk.StringVar(value='Dash')
        tk.OptionMenu(self.step3_frame, self.predicted_style_var, 'Solid', 'Dash', 'Dot').grid(row=2, column=2, padx=5)

    def create_step4_widgets(self):
        self.generate_button = tk.Button(self.step4_frame, text="그래프 생성 실행", command=self.generate_graph, font=('Helvetica', 12, 'bold'), bg='#d3ffd3')
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
            self.df_columns = pd.read_csv(self.file_path, nrows=0).columns.tolist()
            
            # 컬럼 메뉴 업데이트
            for i, menu in enumerate(self.column_menus):
                menu['menu'].delete(0, 'end')
                for col in self.df_columns:
                    menu['menu'].add_command(label=col, command=tk._setit(self.column_vars[i], col))
            
            # 컬럼 자동 추천
            for i, label_text in enumerate([l.cget("text") for l in self.step2_frame.winfo_children() if isinstance(l, tk.Label)]):
                for col in self.df_columns:
                    if label_text.startswith('실제값 X') and col == '날짜': self.column_vars[i].set(col)
                    if label_text.startswith('실제값 Y') and col == '실제값': self.column_vars[i].set(col)
                    if label_text.startswith('예측값 X') and col == '예측날짜': self.column_vars[i].set(col)
                    if label_text.startswith('예측값 Y') and col == '예측값': self.column_vars[i].set(col)

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
            'file_path': self.file_path, 'title': self.title_var.get(),
            'actual_x': self.column_vars[0].get(), 'actual_y': self.column_vars[1].get(),
            'predicted_x': self.column_vars[2].get(), 'predicted_y': self.column_vars[3].get(),
            'actual_color': self.actual_color_var.get(), 'actual_style': self.actual_style_var.get(),
            'predicted_color': self.predicted_color_var.get(), 'predicted_style': self.predicted_style_var.get()
        }
        
        if not all([params['actual_x'], params['actual_y'], params['predicted_x'], params['predicted_y']]):
            messagebox.showwarning("경고", "2단계에서 모든 데이터 축(X, Y)의 컬럼을 선택해주세요.")
            return
            
        create_graph(params)

if __name__ == "__main__":
    root = tk.Tk()
    app = GraphApp(root)
    root.mainloop()