import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import pandas as pd
import plotly.graph_objects as go
import webbrowser
import os

# --------------------------------------------------------------------------
# 이전 단계에서 완성된 그래프 생성 로직을 함수 안에 넣습니다.
# 이 함수는 파일 경로와 그래프 제목을 인자로 받습니다.
# --------------------------------------------------------------------------
def create_graph(file_path, graph_title):
    try:
        # 1. CSV 파일 불러오기
        df = pd.read_csv(file_path)

        # 2. 필수 컬럼 확인
        required_columns = ['날짜', '실제값', '예측날짜', '예측값', '점프예측']
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            messagebox.showerror("오류", f"CSV 파일에 필수 컬럼이 없습니다: {', '.join(missing_cols)}")
            return

        # 3. 날짜 형식 변환
        df['날짜'] = pd.to_datetime(df['날짜'])
        df['예측날짜'] = pd.to_datetime(df['예측날짜'])

        # 4. 정보창 표시용 데이터 준비 ('초' 제외)
        df['표시_날짜'] = df['날짜'].dt.strftime('%Y-%m-%d')
        df['표시_시간'] = df['날짜'].dt.strftime('%H:%M')
        df['표시_예측날짜'] = df['예측날짜'].dt.strftime('%Y-%m-%d')
        df['표시_예측시간'] = df['예측날짜'].dt.strftime('%H:%M')

        # 5. 그래프 객체 생성
        fig = go.Figure()

        # 6. '실제값' 라인 추가
        fig.add_trace(go.Scatter(
            x=df['날짜'],
            y=df['실제값'],
            mode='lines',
            name='실제값 (Actual)',
            customdata=df[['표시_날짜', '표시_시간', '예측값', '표시_예측날짜', '표시_예측시간', '점프예측']],
            hovertemplate=(
                '<b>--- 측정 시점 (기준) ---</b><br>'
                '<b>시간:</b> %{customdata[0]} %{customdata[1]}<br>'
                '<b>실제값:</b> %{y}<br>'
                '<b>점프예측:</b> %{customdata[5]}<br>'
                '<br><b>--- 해당 시점의 예측 정보 ---</b><br>'
                '<b>예측 대상 시간:</b> %{customdata[3]} %{customdata[4]}<br>'
                '<b>예측값:</b> %{customdata[2]}'
                '<extra></extra>'
            )
        ))

        # 7. '예측값' 라인 추가
        fig.add_trace(go.Scatter(
            x=df['예측날짜'],
            y=df['예측값'],
            mode='lines',
            name='예측값 (Predicted)',
            line=dict(dash='dot'),
            hovertemplate='<b>예측값:</b> %{y}<extra></extra>'
        ))

        # 8. 그래프 레이아웃 설정
        fig.update_layout(
            title=graph_title, # UI에서 입력받은 제목 사용
            xaxis_title='타임라인',
            yaxis_title='값',
            hovermode='x unified'
        )

        # 9. 그래프를 HTML 파일로 저장하고 자동으로 열기
        output_filename = "interactive_graph.html"
        fig.write_html(output_filename)
        webbrowser.open('file://' + os.path.realpath(output_filename))
        messagebox.showinfo("성공", f"'{output_filename}' 파일이 생성되었으며, 웹 브라우저에서 자동으로 열립니다.")

    except Exception as e:
        messagebox.showerror("오류 발생", f"그래프 생성 중 오류가 발생했습니다:\n{e}")

# --------------------------------------------------------------------------
# GUI 애플리케이션 로직
# --------------------------------------------------------------------------
class GraphApp:
    def __init__(self, root):
        self.root = root
        self.root.title("대화형 그래프 생성기")
        self.root.geometry("400x200") # 창 크기 조절

        self.file_path = ""

        # 제목 입력 위젯
        self.title_label = tk.Label(root, text="그래프 제목:")
        self.title_label.pack(pady=5)
        self.title_entry = tk.Entry(root, width=50)
        self.title_entry.pack(pady=5)
        self.title_entry.insert(0, "실제값 vs 예측값 시계열 분석") # 기본 제목 설정

        # 파일 선택 위젯
        self.file_button = tk.Button(root, text="CSV 파일 선택", command=self.select_file)
        self.file_button.pack(pady=10)
        self.file_label = tk.Label(root, text="선택된 파일이 없습니다.", fg="blue")
        self.file_label.pack()

        # 그래프 생성 버튼
        self.generate_button = tk.Button(root, text="그래프 생성", command=self.generate_graph, font=('Helvetica', 10, 'bold'))
        self.generate_button.pack(pady=20)

    def select_file(self):
        # 파일 탐색기 열기
        path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv")]
        )
        if path:
            self.file_path = path
            # os.path.basename을 사용하여 파일 이름만 추출
            filename = os.path.basename(path)
            self.file_label.config(text=f"선택된 파일: {filename}")

    def generate_graph(self):
        graph_title = self.title_entry.get()
        if not self.file_path:
            messagebox.showwarning("경고", "먼저 CSV 파일을 선택해주세요.")
            return
        if not graph_title:
            messagebox.showwarning("경고", "그래프 제목을 입력해주세요.")
            return
        
        # 그래프 생성 함수 호출
        create_graph(self.file_path, graph_title)

if __name__ == "__main__":
    root = tk.Tk()
    app = GraphApp(root)
    root.mainloop()