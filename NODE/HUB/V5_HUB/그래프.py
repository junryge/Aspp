import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser
import pandas as pd
import plotly.graph_objects as go
import webbrowser
import os

# --------------------------------------------------------------------------
# 그래프 생성 로직
# ✨ Hovertemplate에서 customdata를 사용하여 시간 형식을 'HH:MM'으로 고정
# --------------------------------------------------------------------------
def create_graph(params):
    try:
        df = pd.read_csv(params['file_path'])
        
        # UI에서 선택한 컬럼 이름 변수 할당
        actual_x_col = params['actual_x']
        actual_y_col = params['actual_y']
        predicted_x_col = params['predicted_x']
        predicted_y_col = params['predicted_y']

        # 날짜 형식 변환 시도
        try:
            df[actual_x_col] = pd.to_datetime(df[actual_x_col])
            df[predicted_x_col] = pd.to_datetime(df[predicted_x_col])
            
            # ✨ 정보창에 표시할 날짜/시간 문자열 컬럼 생성 ('초' 제외)
            df['hover_actual_date'] = df[actual_x_col].dt.strftime('%Y-%m-%d')
            df['hover_actual_time'] = df[actual_x_col].dt.strftime('%H:%M')
            df['hover_predicted_date'] = df[predicted_x_col].dt.strftime('%Y-%m-%d')
            df['hover_predicted_time'] = df[predicted_x_col].dt.strftime('%H:%M')

        except Exception as e:
            messagebox.showwarning("날짜 변환 경고", f"시간 축으로 선택된 컬럼을 날짜 형식으로 변환하는 데 실패했습니다. 일반 데이터로 처리합니다.\n({e})")
            df['hover_actual_date'] = df[actual_x_col]
            df['hover_actual_time'] = ""
            df['hover_predicted_date'] = df[predicted_x_col]
            df['hover_predicted_time'] = ""


        fig = go.Figure()

        # 실제값(Actual) 라인 추가
        fig.add_trace(go.Scatter(
            x=df[actual_x_col],
            y=df[actual_y_col],
            mode='lines',
            name='실제값 (Actual)',
            line=dict(color=params['actual_color'], dash=params['actual_style'].lower()),
            customdata=df[['hover_actual_date', 'hover_actual_time']],
            hovertemplate=(
                f"<b>{actual_y_col}:</b> %{{y}}<br>"
                "------------------<br>"
                "<b>날짜:</b> %{customdata[0]}<br>"
                "<b>시간:</b> %{customdata[1]}"
                "<extra></extra>"
            )
        ))

        # 예측값(Predicted) 라인 추가
        fig.add_trace(go.Scatter(
            x=df[predicted_x_col],
            y=df[predicted_y_col],
            mode='lines',
            name='예측값 (Predicted)',
            line=dict(color=params['predicted_color'], dash=params['predicted_style'].lower()),
            customdata=df[['hover_predicted_date', 'hover_predicted_time']],
            hovertemplate=(
                f"<b>{predicted_y_col}:</b> %{{y}}<br>"
                "------------------<br>"
                "<b>날짜:</b> %{customdata[0]}<br>"
                "<b>시간:</b> %{customdata[1]}"
                "<extra></extra>"
            )
        ))
        
        fig.update_layout(
            title=params['title'],
            xaxis_title='X-Axis',
            yaxis_title='Y-Axis',
            hovermode='x unified'
        )

        output_filename = "final_custom_graph.html"
        fig.write_html(output_filename)
        webbrowser.open('file://' + os.path.realpath(output_filename))
        messagebox.showinfo("성공", f"'{output_filename}' 파일이 생성되었으며, 웹 브라우저에서 자동으로 열립니다.")

    except Exception as e:
        messagebox.showerror("오류 발생", f"그래프 생성 중 오류가 발생했습니다:\n{e}")

# --------------------------------------------------------------------------
# GUI 애플리케이션 로직 (이전과 동일)
# --------------------------------------------------------------------------
class GraphApp:
    def __init__(self, root):
        self.root = root
        self.root.title("맞춤형 그래프 생성기 v2.1 (시간수정)")
        self.file_path = ""
        self.column_widgets = []
        self.df_columns = []

        top_frame = tk.Frame(root, padx=10, pady=5)
        top_frame.pack(fill='x')
        
        columns_frame = tk.LabelFrame(root, text="데이터 선택 (CSV 선택 후 활성화)", padx=10, pady=10)
        columns_frame.pack(fill='x', padx=10, pady=5)

        style_frame = tk.LabelFrame(root, text="그래프 스타일링", padx=10, pady=10)
        style_frame.pack(fill='x', padx=10, pady=5)
        
        bottom_frame = tk.Frame(root, pady=10)
        bottom_frame.pack()

        tk.Button(top_frame, text="1. CSV 파일 열기", command=self.select_file).pack(side='left')
        self.file_label = tk.Label(top_frame, text="선택된 파일이 없습니다.", fg="blue")
        self.file_label.pack(side='left', padx=10)
        
        tk.Label(style_frame, text="그래프 제목:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.title_var = tk.StringVar(value="사용자 정의 그래프")
        tk.Entry(style_frame, textvariable=self.title_var, width=50).grid(row=0, column=1, columnspan=3, sticky='ew', padx=5, pady=2)
        
        self.columns_frame = columns_frame
        
        tk.Label(style_frame, text="[실제값 라인]").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.actual_color_var = tk.StringVar(value='#1f77b4')
        self.actual_color_btn = tk.Button(style_frame, text="색상 선택", command=lambda: self.choose_color(self.actual_color_var, self.actual_color_btn), bg=self.actual_color_var.get())
        self.actual_color_btn.grid(row=1, column=1, padx=5)

        self.actual_style_var = tk.StringVar(value='Solid')
        tk.OptionMenu(style_frame, self.actual_style_var, 'Solid', 'Dash', 'Dot').grid(row=1, column=2, padx=5)

        tk.Label(style_frame, text="[예측값 라인]").grid(row=2, column=0, sticky='w', padx=5, pady=5)
        self.predicted_color_var = tk.StringVar(value='#ff7f0e')
        self.predicted_color_btn = tk.Button(style_frame, text="색상 선택", command=lambda: self.choose_color(self.predicted_color_var, self.predicted_color_btn), bg=self.predicted_color_var.get())
        self.predicted_color_btn.grid(row=2, column=1, padx=5)

        self.predicted_style_var = tk.StringVar(value='Dash')
        tk.OptionMenu(style_frame, self.predicted_style_var, 'Solid', 'Dash', 'Dot').grid(row=2, column=2, padx=5)

        tk.Button(bottom_frame, text="2. 그래프 생성 실행", command=self.generate_graph, font=('Helvetica', 12, 'bold'), bg='#d3ffd3').pack()
    
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
            self.update_column_widgets()
        except Exception as e:
            messagebox.showerror("오류", f"CSV 파일을 읽는 데 실패했습니다:\n{e}")

    def update_column_widgets(self):
        for widget in self.column_widgets: widget.destroy()
        self.column_widgets = []

        labels = ["실제값 X축 (시간):", "실제값 Y축 (값):", "예측값 X축 (시간):", "예측값 Y축 (값):"]
        self.column_vars = [tk.StringVar() for _ in labels]

        for i, label_text in enumerate(labels):
            label = tk.Label(self.columns_frame, text=label_text)
            label.grid(row=i, column=0, sticky='w', padx=5, pady=2)
            self.column_widgets.append(label)

            menu = tk.OptionMenu(self.columns_frame, self.column_vars[i], *self.df_columns)
            menu.grid(row=i, column=1, sticky='ew', padx=5, pady=2)
            self.column_widgets.append(menu)
            
            for col in self.df_columns:
                if label_text.startswith('실제값 X') and col == '날짜': self.column_vars[i].set(col)
                if label_text.startswith('실제값 Y') and col == '실제값': self.column_vars[i].set(col)
                if label_text.startswith('예측값 X') and col == '예측날짜': self.column_vars[i].set(col)
                if label_text.startswith('예측값 Y') and col == '예측값': self.column_vars[i].set(col)

    def generate_graph(self):
        if not self.file_path:
            messagebox.showwarning("경고", "먼저 CSV 파일을 선택해주세요.")
            return

        params = {
            'file_path': self.file_path, 'title': self.title_var.get(),
            'actual_x': self.column_vars[0].get(), 'actual_y': self.column_vars[1].get(),
            'predicted_x': self.column_vars[2].get(), 'predicted_y': self.column_vars[3].get(),
            'actual_color': self.actual_color_var.get(), 'actual_style': self.actual_style_var.get(),
            'predicted_color': self.predicted_color_var.get(), 'predicted_style': self.predicted_style_var.get()
        }
        
        if not all([params['actual_x'], params['actual_y'], params['predicted_x'], params['predicted_y']]):
            messagebox.showwarning("경고", "모든 데이터 축(X, Y)의 컬럼을 선택해주세요.")
            return
            
        create_graph(params)

if __name__ == "__main__":
    root = tk.Tk()
    app = GraphApp(root)
    root.mainloop()