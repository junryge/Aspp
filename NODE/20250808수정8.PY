"""
반도체 물류 예측 하이브리드 딥러닝 모델 GUI 애플리케이션
=======================================================
LSTM, RNN, GRU, ARIMA를 통합한 하이브리드 모델을
사용하기 쉬운 GUI로 제공합니다.

주요 기능:
1. 데이터 파일 선택 및 로드
2. 학습 파라미터 설정
3. 실시간 학습 진행 모니터링
4. 모델 예측 및 평가
5. 결과 시각화 및 저장

개발일: 2024년
버전: 2.1 GUI
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import queue
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, GRU, SimpleRNN, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.font_manager as fm
import os
import platform
import json
import joblib
import warnings
from datetime import datetime, timedelta
import sys

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')

# ===================================
# 한글 폰트 설정
# ===================================
def setup_korean_font():
    """운영체제별 한글 폰트 자동 설정"""
    system = platform.system()

    # 폰트 설정 시도
    if system == 'Windows':
        # 윈도우
        font_candidates = ['Malgun Gothic', 'NanumGothic', '맑은 고딕', '나눔고딕']
    elif system == 'Darwin':
        # 맥OS
        font_candidates = ['AppleGothic', 'NanumGothic', 'Apple SD Gothic Neo']
    else:
        # 리눅스
        font_candidates = ['NanumGothic', 'DejaVu Sans']

    # 사용 가능한 폰트 찾기
    font_set = False
    for font_name in font_candidates:
        if font_name in [f.name for f in fm.fontManager.ttflist]:
            plt.rcParams['font.family'] = font_name
            font_set = True
            print(f"✓ 한글 폰트 설정: {font_name}")
            break

    # 폰트를 찾지 못한 경우 직접 경로 설정 시도
    if not font_set:
        if system == 'Windows':
            font_path = 'C:/Windows/Fonts/malgun.ttf'
            if os.path.exists(font_path):
                font_prop = fm.FontProperties(fname=font_path)
                plt.rcParams['font.family'] = font_prop.get_name()
                font_set = True
        elif system == 'Darwin':
            font_path = '/System/Library/Fonts/Supplemental/AppleGothic.ttf'
            if os.path.exists(font_path):
                font_prop = fm.FontProperties(fname=font_path)
                plt.rcParams['font.family'] = font_prop.get_name()
                font_set = True

    # 마이너스 기호 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False

    # 폰트 크기 설정
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9

    return font_set

# 한글 폰트 설정 실행
font_initialized = setup_korean_font()
if not font_initialized:
    print("⚠ 한글 폰트 설정 실패 - 영문으로 표시됩니다.")

# CPU 모드 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

# 랜덤 시드 고정
RANDOM_SEED = 2079936
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class HybridModelGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🚀 반도체 물류 예측 하이브리드 AI 시스템")

        # 한글 폰트 확인 및 텍스트 설정
        if not font_initialized:
            print("⚠ 한글 폰트가 설정되지 않았습니다. matplotlib 그래프에서 한글이 깨질 수 있습니다.")
            print("  해결방법: 나눔고딕 또는 맑은 고딕 폰트를 설치하세요.")
            self.use_korean = False
            # 영문 텍스트 설정
            self.text_labels = {
                'training_progress': 'Training Progress',
                'epoch': 'Epoch',
                'loss': 'Loss',
                'prediction_comparison': 'Prediction vs Actual',
                'time_index': 'Time Index',
                'logistics_volume': 'Logistics Volume',
                'actual': 'Actual',
                'predicted': 'Predicted'
            }
        else:
            self.use_korean = True
            # 한글 텍스트 설정
            self.text_labels = {
                'training_progress': '학습 진행 상황',
                'epoch': '에폭',
                'loss': '손실값',
                'prediction_comparison': '예측 결과 비교',
                'time_index': '시간 인덱스',
                'logistics_volume': '물류량',
                'actual': '실제값',
                'predicted': '예측값'
            }

        # 다크 테마 스타일 설정
        self.setup_styles()

        # 변수 초기화
        self.data_path = tk.StringVar()
        self.models = {}
        self.scaler = None
        self.data = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.is_training = False
        self.log_queue = queue.Queue()

        # UI 구성
        self.create_widgets()

        # 로그 업데이트 시작
        self.update_logs()

    def setup_styles(self):
        """GUI 스타일 설정"""
        style = ttk.Style()
        style.theme_use('clam')

        # 다크 테마 색상
        bg_color = "#2b2b2b"
        fg_color = "#ffffff"
        button_bg = "#4CAF50"
        button_hover = "#45a049"

        self.root.configure(bg=bg_color)

        style.configure("Title.TLabel", background=bg_color, foreground=fg_color,
                       font=('Arial', 16, 'bold'))
        style.configure("Heading.TLabel", background=bg_color, foreground=fg_color,
                       font=('Arial', 12, 'bold'))
        style.configure("Dark.TFrame", background=bg_color)
        style.configure("Success.TButton", font=('Arial', 10, 'bold'))

    def create_widgets(self):
        """위젯 생성 및 배치"""
        # 메인 컨테이너
        main_container = ttk.Frame(self.root, style="Dark.TFrame")
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 왼쪽 패널 - 데이터 및 설정
        left_panel = ttk.Frame(main_container, style="Dark.TFrame")
        left_panel.grid(row=0, column=0, sticky="nsew", padx=5)

        # 데이터 선택 섹션
        self.create_data_section(left_panel)

        # 파라미터 설정 섹션
        self.create_parameter_section(left_panel)

        # 컨트롤 버튼 섹션
        self.create_control_section(left_panel)

        # 중앙 패널 - 진행 상황 및 로그
        center_panel = ttk.Frame(main_container, style="Dark.TFrame")
        center_panel.grid(row=0, column=1, sticky="nsew", padx=5)

        # 진행 상황 섹션
        self.create_progress_section(center_panel)

        # 로그 섹션
        self.create_log_section(center_panel)

        # 오른쪽 패널 - 결과 및 시각화
        right_panel = ttk.Frame(main_container, style="Dark.TFrame")
        right_panel.grid(row=0, column=2, sticky="nsew", padx=5)

        # 결과 섹션
        self.create_results_section(right_panel)

        # 시각화 섹션
        self.create_visualization_section(right_panel)

        # 그리드 가중치 설정
        main_container.grid_rowconfigure(0, weight=1)
        main_container.grid_columnconfigure(0, weight=2)
        main_container.grid_columnconfigure(1, weight=3)
        main_container.grid_columnconfigure(2, weight=3)

    def create_data_section(self, parent):
        """데이터 선택 섹션"""
        frame = ttk.LabelFrame(parent, text="📁 데이터 설정", padding=10)
        frame.pack(fill=tk.X, pady=5)

        # 파일 경로 표시
        ttk.Label(frame, text="데이터 파일:").pack(anchor=tk.W)
        path_frame = ttk.Frame(frame)
        path_frame.pack(fill=tk.X, pady=5)

        self.path_entry = ttk.Entry(path_frame, textvariable=self.data_path, width=30)
        self.path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Button(path_frame, text="찾아보기",
                  command=self.browse_file).pack(side=tk.LEFT, padx=5)

        # 데이터 정보
        self.data_info = tk.Text(frame, height=4, width=40, bg="#3c3c3c", fg="white")
        self.data_info.pack(fill=tk.X, pady=5)
        self.data_info.insert(tk.END, "데이터를 선택하세요...")
        self.data_info.config(state=tk.DISABLED)

    def create_parameter_section(self, parent):
        """파라미터 설정 섹션"""
        frame = ttk.LabelFrame(parent, text="⚙️ 학습 파라미터", padding=10)
        frame.pack(fill=tk.X, pady=5)

        # 에폭 수
        ttk.Label(frame, text="에폭 수:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.epochs_var = tk.IntVar(value=100)
        ttk.Spinbox(frame, from_=10, to=500, textvariable=self.epochs_var,
                   width=15).grid(row=0, column=1, pady=2)

        # 배치 크기
        ttk.Label(frame, text="배치 크기:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.batch_var = tk.IntVar(value=64)
        ttk.Spinbox(frame, from_=16, to=256, increment=16, textvariable=self.batch_var,
                   width=15).grid(row=1, column=1, pady=2)

        # 학습률
        ttk.Label(frame, text="학습률:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.lr_var = tk.DoubleVar(value=0.001)
        ttk.Entry(frame, textvariable=self.lr_var, width=15).grid(row=2, column=1, pady=2)

        # 시퀀스 길이
        ttk.Label(frame, text="시퀀스 길이:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.seq_var = tk.IntVar(value=30)
        ttk.Spinbox(frame, from_=10, to=100, textvariable=self.seq_var,
                   width=15).grid(row=3, column=1, pady=2)

        # 예측 시간
        ttk.Label(frame, text="예측 시간(분):").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.future_var = tk.IntVar(value=10)
        ttk.Spinbox(frame, from_=5, to=60, increment=5, textvariable=self.future_var,
                   width=15).grid(row=4, column=1, pady=2)

        # 모델 선택
        ttk.Label(frame, text="모델 선택:").grid(row=5, column=0, sticky=tk.W, pady=2)
        model_frame = ttk.Frame(frame)
        model_frame.grid(row=5, column=1, pady=2)

        self.model_vars = {
            'LSTM': tk.BooleanVar(value=True),
            'GRU': tk.BooleanVar(value=True),
            'RNN': tk.BooleanVar(value=True),
            'Bi-LSTM': tk.BooleanVar(value=True)
        }

        for i, (name, var) in enumerate(self.model_vars.items()):
            ttk.Checkbutton(model_frame, text=name, variable=var).grid(row=i//2, column=i%2)

    def create_control_section(self, parent):
        """컨트롤 버튼 섹션"""
        frame = ttk.LabelFrame(parent, text="🎮 컨트롤", padding=10)
        frame.pack(fill=tk.X, pady=5)

        # 버튼들
        self.load_btn = ttk.Button(frame, text="📊 데이터 로드",
                                  command=self.load_data, style="Success.TButton")
        self.load_btn.pack(fill=tk.X, pady=2)

        self.train_btn = ttk.Button(frame, text="🚀 학습 시작",
                                   command=self.start_training, style="Success.TButton")
        self.train_btn.pack(fill=tk.X, pady=2)
        self.train_btn.config(state=tk.DISABLED)

        self.stop_btn = ttk.Button(frame, text="⏹️ 학습 중지",
                                  command=self.stop_training)
        self.stop_btn.pack(fill=tk.X, pady=2)
        self.stop_btn.config(state=tk.DISABLED)

        self.predict_btn = ttk.Button(frame, text="🔮 예측 실행",
                                     command=self.run_prediction)
        self.predict_btn.pack(fill=tk.X, pady=2)
        self.predict_btn.config(state=tk.DISABLED)

        self.save_btn = ttk.Button(frame, text="💾 모델 저장",
                                  command=self.save_models)
        self.save_btn.pack(fill=tk.X, pady=2)
        self.save_btn.config(state=tk.DISABLED)

        ttk.Separator(frame, orient='horizontal').pack(fill=tk.X, pady=5)

        self.load_model_btn = ttk.Button(frame, text="📂 모델 불러오기",
                                        command=self.load_saved_models)
        self.load_model_btn.pack(fill=tk.X, pady=2)

    def create_progress_section(self, parent):
        """진행 상황 섹션"""
        frame = ttk.LabelFrame(parent, text="📈 학습 진행 상황", padding=10)
        frame.pack(fill=tk.X, pady=5)

        # 전체 진행률
        ttk.Label(frame, text="전체 진행률:").pack(anchor=tk.W)
        self.overall_progress = ttk.Progressbar(frame, mode='determinate')
        self.overall_progress.pack(fill=tk.X, pady=2)

        # 현재 모델
        self.current_model_label = ttk.Label(frame, text="대기 중...")
        self.current_model_label.pack(anchor=tk.W, pady=2)

        # 에폭 진행률
        ttk.Label(frame, text="에폭 진행률:").pack(anchor=tk.W)
        self.epoch_progress = ttk.Progressbar(frame, mode='determinate')
        self.epoch_progress.pack(fill=tk.X, pady=2)

        # 현재 에폭
        self.current_epoch_label = ttk.Label(frame, text="")
        self.current_epoch_label.pack(anchor=tk.W, pady=2)

        # 성능 지표
        self.metrics_frame = ttk.Frame(frame)
        self.metrics_frame.pack(fill=tk.X, pady=5)

        metrics_labels = ["Loss:", "Val Loss:", "MAE:", "Val MAE:"]
        self.metric_labels = {}

        for i, label in enumerate(metrics_labels):
            ttk.Label(self.metrics_frame, text=label).grid(row=i//2, column=(i%2)*2, sticky=tk.W)
            metric_label = ttk.Label(self.metrics_frame, text="--")
            metric_label.grid(row=i//2, column=(i%2)*2+1, sticky=tk.W, padx=5)
            self.metric_labels[label] = metric_label

    def create_log_section(self, parent):
        """로그 섹션"""
        frame = ttk.LabelFrame(parent, text="📝 로그", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # 로그 텍스트
        self.log_text = scrolledtext.ScrolledText(frame, height=15, width=50,
                                                  bg="#1e1e1e", fg="#00ff00",
                                                  font=('Consolas', 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def create_results_section(self, parent):
        """결과 섹션"""
        frame = ttk.LabelFrame(parent, text="📊 예측 결과", padding=10)
        frame.pack(fill=tk.X, pady=5)

        # 결과 텍스트
        self.results_text = tk.Text(frame, height=10, width=40,
                                   bg="#3c3c3c", fg="white", font=('Arial', 9))
        self.results_text.pack(fill=tk.X)
        self.results_text.insert(tk.END, "학습 후 결과가 여기에 표시됩니다...")
        self.results_text.config(state=tk.DISABLED)

    def create_visualization_section(self, parent):
        """시각화 섹션"""
        frame = ttk.LabelFrame(parent, text="📈 시각화", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # 그래프 캔버스 (한글/영문 자동 선택)
        self.fig, self.ax = plt.subplots(figsize=(6, 4), facecolor='#2b2b2b')
        self.ax.set_facecolor('#3c3c3c')

        # 언어에 따른 제목과 라벨 설정
        self.ax.set_title(self.text_labels['training_progress'], color='white', fontsize=12, pad=10)
        self.ax.set_xlabel(self.text_labels['epoch'], color='white', fontsize=10)
        self.ax.set_ylabel(self.text_labels['loss'], color='white', fontsize=10)
        self.ax.tick_params(colors='white', labelsize=9)
        self.ax.grid(True, alpha=0.3)

        # 캔버스 생성 및 폰트 적용
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def browse_file(self):
        """파일 선택 다이얼로그"""
        filename = filedialog.askopenfilename(
            title="데이터 파일 선택",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.data_path.set(filename)
            self.log("파일 선택: " + os.path.basename(filename))

    def log(self, message):
        """로그 메시지 추가"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        self.log_queue.put(log_message)

    def update_logs(self):
        """로그 업데이트"""
        try:
            while True:
                message = self.log_queue.get_nowait()
                self.log_text.insert(tk.END, message)
                self.log_text.see(tk.END)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.update_logs)

    def load_data(self):
        """데이터 로드 및 전처리"""
        if not self.data_path.get():
            messagebox.showerror("오류", "데이터 파일을 선택하세요!")
            return

        try:
            self.log("데이터 로딩 시작...")

            # 데이터 로드
            self.data = pd.read_csv(self.data_path.get())

            # 데이터 정보 업데이트
            self.data_info.config(state=tk.NORMAL)
            self.data_info.delete(1.0, tk.END)
            info_text = f"📊 데이터 정보\n"
            info_text += f"행 수: {len(self.data):,}\n"
            info_text += f"열 수: {len(self.data.columns)}\n"
            info_text += f"컬럼: {', '.join(self.data.columns[:5])}..."
            self.data_info.insert(tk.END, info_text)
            self.data_info.config(state=tk.DISABLED)

            # 전처리
            self.preprocess_data()

            self.log("✅ 데이터 로드 완료!")
            self.train_btn.config(state=tk.NORMAL)
            
            # 모델이 미리 로드된 상태였다면, 데이터가 로드되었으므로 예측 버튼 활성화
            if self.models:
                self.predict_btn.config(state=tk.NORMAL)

            messagebox.showinfo("성공", "데이터 로드 및 전처리 완료!")

        except Exception as e:
            self.log(f"❌ 데이터 로드 실패: {str(e)}")
            messagebox.showerror("오류", f"데이터 로드 실패:\n{str(e)}")

    def preprocess_data(self):
        """데이터 전처리"""
        self.log("데이터 전처리 중...")

        # 시간 컬럼 변환
        self.data['CURRTIME'] = pd.to_datetime(self.data['CURRTIME'], format='%Y%m%d%H%M')
        self.data['TIME'] = pd.to_datetime(self.data['TIME'], format='%Y%m%d%H%M')

        # SUM 컬럼 제거
        columns_to_drop = [col for col in self.data.columns if 'SUM' in col]
        self.data = self.data.drop(columns=columns_to_drop)

        # 필요한 컬럼만 선택
        self.data = self.data[['CURRTIME', 'TOTALCNT', 'TIME']]
        self.data.set_index('CURRTIME', inplace=True)

        # FUTURE 컬럼 생성
        self.data['FUTURE'] = pd.NA
        future_minutes = self.future_var.get()

        for i in self.data.index:
            future_time = i + pd.Timedelta(minutes=future_minutes)
            if (future_time <= self.data.index.max()) & (future_time in self.data.index):
                self.data.loc[i, 'FUTURE'] = self.data.loc[future_time, 'TOTALCNT']

        self.data.dropna(subset=['FUTURE'], inplace=True)

        # 특징 엔지니어링
        self.data['hour'] = self.data.index.hour
        self.data['dayofweek'] = self.data.index.dayofweek
        self.data['is_weekend'] = (self.data.index.dayofweek >= 5).astype(int)
        self.data['MA_5'] = self.data['TOTALCNT'].rolling(window=5, min_periods=1).mean()
        self.data['MA_10'] = self.data['TOTALCNT'].rolling(window=10, min_periods=1).mean()
        self.data['MA_30'] = self.data['TOTALCNT'].rolling(window=30, min_periods=1).mean()
        self.data['STD_5'] = self.data['TOTALCNT'].rolling(window=5, min_periods=1).std()
        self.data['STD_10'] = self.data['TOTALCNT'].rolling(window=10, min_periods=1).std()
        self.data['change_rate'] = self.data['TOTALCNT'].pct_change()
        self.data['change_rate_5'] = self.data['TOTALCNT'].pct_change(5)
        self.data = self.data.ffill().fillna(0)

        # 스케일링
        self.scaler = StandardScaler()
        scaled_list = ['TOTALCNT', 'FUTURE', 'MA_5', 'MA_10', 'MA_30', 'STD_5', 'STD_10']
        scaled_list = [col for col in scaled_list if col in self.data.columns]

        scaled_data = self.scaler.fit_transform(self.data[scaled_list])
        scaled_columns = [f'scaled_{col}' for col in scaled_list]
        scaled_df = pd.DataFrame(scaled_data, columns=scaled_columns, index=self.data.index)

        self.data = pd.merge(self.data, scaled_df, left_index=True, right_index=True, how='left')

        # 시퀀스 생성
        self.create_sequences()

        self.log(f"전처리 완료 - 훈련: {len(self.X_train)}, 검증: {len(self.X_val)}, 테스트: {len(self.X_test)}")

    def create_sequences(self):
        """시퀀스 데이터 생성"""
        seq_length = self.seq_var.get()

        # 연속성 확인하여 데이터 분할
        time_diff = self.data.index.to_series().diff()
        split_points = time_diff > pd.Timedelta(minutes=1)
        segment_ids = split_points.cumsum()

        input_features = [col for col in self.data.columns if col.startswith('scaled_') and col != 'scaled_FUTURE']

        all_X, all_y = [], []

        for segment_id in segment_ids.unique():
            segment = self.data[segment_ids == segment_id]

            if len(segment) > seq_length:
                X_data = segment[input_features].values
                y_data = segment['scaled_FUTURE'].values

                for i in range(len(segment) - seq_length):
                    all_X.append(X_data[i:i+seq_length])
                    all_y.append(y_data[i+seq_length])

        X = np.array(all_X)
        y = np.array(all_y)

        # 데이터 분할
        train_size = int(0.7 * len(X))
        val_size = int(0.15 * len(X))

        self.X_train = X[:train_size]
        self.y_train = y[:train_size]
        self.X_val = X[train_size:train_size+val_size]
        self.y_val = y[train_size:train_size+val_size]
        self.X_test = X[train_size+val_size:]
        self.y_test = y[train_size+val_size:]

    def start_training(self):
        """학습 시작"""
        if self.is_training:
            messagebox.showwarning("경고", "이미 학습이 진행 중입니다!")
            return

        self.is_training = True
        self.train_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.load_model_btn.config(state=tk.DISABLED) # 학습 중에는 모델 로드 불가

        # 학습 스레드 시작
        self.training_thread = threading.Thread(target=self.train_models)
        self.training_thread.start()

    def stop_training(self):
        """학습 중지"""
        self.is_training = False
        self.log("⏹️ 학습 중지 요청...")

    def train_models(self):
        """모델 학습"""
        try:
            self.log("🚀 하이브리드 모델 학습 시작")
            self.models.clear() # 학습 시작 시 기존 모델 초기화

            input_shape = (self.X_train.shape[1], self.X_train.shape[2])
            selected_models = []

            # 선택된 모델 확인
            if self.model_vars['LSTM'].get():
                selected_models.append(('lstm', self.build_lstm_model(input_shape)))
            if self.model_vars['GRU'].get():
                selected_models.append(('gru', self.build_gru_model(input_shape)))
            if self.model_vars['RNN'].get():
                selected_models.append(('rnn', self.build_rnn_model(input_shape)))
            if self.model_vars['Bi-LSTM'].get():
                selected_models.append(('bi_lstm', self.build_bilstm_model(input_shape)))

            total_models = len(selected_models)

            for idx, (model_name, model) in enumerate(selected_models):
                if not self.is_training:
                    break

                self.log(f"\n📊 {model_name.upper()} 모델 학습 시작")
                self.root.after(0, self.update_current_model, f"{model_name.upper()} 모델 학습 중...")

                # 전체 진행률 업데이트
                overall_progress = (idx / total_models) * 100
                self.root.after(0, self.update_overall_progress, overall_progress)

                # 모델 컴파일
                optimizer = Adam(learning_rate=self.lr_var.get())
                model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

                # 콜백 설정
                early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
                reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

                # 학습
                epochs = self.epochs_var.get()
                batch_size = self.batch_var.get()

                for epoch in range(epochs):
                    if not self.is_training:
                        break

                    history = model.fit(
                        self.X_train, self.y_train,
                        validation_data=(self.X_val, self.y_val),
                        epochs=1,
                        batch_size=batch_size,
                        verbose=0,
                        callbacks=[early_stop, reduce_lr]
                    )

                    # 진행률 업데이트
                    epoch_progress = ((epoch + 1) / epochs) * 100
                    self.root.after(0, self.update_epoch_progress, epoch_progress)
                    self.root.after(0, self.update_current_epoch, f"에폭: {epoch+1}/{epochs}")

                    # 메트릭 업데이트
                    loss = history.history['loss'][0]
                    val_loss = history.history['val_loss'][0]
                    mae = history.history['mae'][0]
                    val_mae = history.history['val_mae'][0]

                    self.root.after(0, self.update_metrics, loss, val_loss, mae, val_mae)

                    # 그래프 업데이트
                    if epoch % 5 == 0:
                        self.log(f"  에폭 {epoch+1}: Loss={loss:.4f}, Val_Loss={val_loss:.4f}")

                    # Early stopping 체크
                    if early_stop.stopped_epoch > 0:
                        self.log(f"  조기 종료: 에폭 {epoch+1}")
                        break

                self.models[model_name] = model
                self.log(f"✅ {model_name.upper()} 모델 학습 완료")

            if self.is_training:
                self.log("\n🎉 모든 모델 학습 완료!")
                if self.models: # 학습된 모델이 있을 경우에만 평가 진행
                    self.evaluate_models()
                    # 버튼 상태 업데이트
                    self.root.after(0, self.training_completed)
            else:
                self.log("⏹️ 학습이 중지되었습니다.")

        except Exception as e:
            self.log(f"❌ 학습 중 오류 발생: {str(e)}")
            messagebox.showerror("오류", f"학습 중 오류 발생:\n{str(e)}")
        finally:
            self.is_training = False
            self.root.after(0, self.reset_buttons)

    def build_lstm_model(self, input_shape):
        """LSTM 모델 구축"""
        model = Sequential([
            Input(shape=input_shape),
            LSTM(units=100, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),
            LSTM(units=100, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),
            LSTM(units=100, return_sequences=False),
            Dropout(0.2),
            Dense(units=50, activation='relu'),
            Dense(units=1)
        ])
        return model

    def build_gru_model(self, input_shape):
        """GRU 모델 구축"""
        model = Sequential([
            Input(shape=input_shape),
            GRU(units=100, return_sequences=True),
            Dropout(0.2),
            GRU(units=100, return_sequences=True),
            Dropout(0.2),
            GRU(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=30, activation='relu'),
            Dense(units=1)
        ])
        return model

    def build_rnn_model(self, input_shape):
        """RNN 모델 구축"""
        model = Sequential([
            Input(shape=input_shape),
            SimpleRNN(units=100, return_sequences=True),
            Dropout(0.2),
            SimpleRNN(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=30, activation='relu'),
            Dense(units=1)
        ])
        return model

    def build_bilstm_model(self, input_shape):
        """Bidirectional LSTM 모델 구축"""
        model = Sequential([
            Input(shape=input_shape),
            Bidirectional(LSTM(units=50, return_sequences=True)),
            Dropout(0.2),
            Bidirectional(LSTM(units=50, return_sequences=False)),
            Dropout(0.2),
            Dense(units=30, activation='relu'),
            Dense(units=1)
        ])
        return model

    def evaluate_models(self):
        """모델 평가"""
        self.log("\n📊 모델 평가 시작...")

        results_text = "=" * 50 + "\n"
        results_text += "📊 모델 성능 평가 결과\n"
        results_text += "=" * 50 + "\n\n"
        
        y_test_original = self.inverse_scale(self.y_test)

        # 개별 모델 평가
        for model_name, model in self.models.items():
            y_pred = model.predict(self.X_test, verbose=0).flatten()
            y_pred_original = self.inverse_scale(y_pred)
            
            mae = mean_absolute_error(y_test_original, y_pred_original)
            mse = mean_squared_error(y_test_original, y_pred_original)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test_original, y_pred_original)
            
            results_text += f"🔹 {model_name.upper()} 모델\n"
            results_text += f"  MAE: {mae:.2f}\n"
            results_text += f"  RMSE: {rmse:.2f}\n"
            results_text += f"  R²: {r2:.4f}\n\n"
            self.log(f"{model_name.upper()} - MAE: {mae:.2f}, R²: {r2:.4f}")

        # 앙상블 예측 (모델이 2개 이상일 때만 의미 있음)
        if len(self.models) > 1:
            ensemble_pred = self.ensemble_predict(self.X_test)
            ensemble_pred_original = self.inverse_scale(ensemble_pred)
            
            mae = mean_absolute_error(y_test_original, ensemble_pred_original)
            mse = mean_squared_error(y_test_original, ensemble_pred_original)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test_original, ensemble_pred_original)
            
            results_text += f"🔸 앙상블 모델\n"
            results_text += f"  MAE: {mae:.2f}\n"
            results_text += f"  RMSE: {rmse:.2f}\n"
            results_text += f"  R²: {r2:.4f}\n"
            self.log(f"앙상블 - MAE: {mae:.2f}, R²: {r2:.4f}")
            # 시각화는 앙상블 결과로 업데이트
            self.update_visualization(y_test_original, ensemble_pred_original)
        else: # 모델이 하나일 경우
            # 마지막으로 평가된 모델의 예측 결과로 시각화
            self.update_visualization(y_test_original, y_pred_original)


        # 결과 텍스트 업데이트
        self.root.after(0, self.update_results, results_text)

    def ensemble_predict(self, X):
        """앙상블 예측"""
        if not self.models:
            return np.array([])
        predictions = [model.predict(X, verbose=0).flatten() for model in self.models.values()]
        return np.mean(predictions, axis=0)

    def inverse_scale(self, scaled_data):
        """역스케일링"""
        n_features = self.scaler.n_features_in_
        dummy = np.zeros((len(scaled_data), n_features))
        # 'FUTURE'가 스케일링 대상 컬럼 중 2번째였으므로 해당 위치에 데이터 삽입
        # 주의: preprocess_data의 scaled_list 순서에 의존적
        # ['TOTALCNT', 'FUTURE', 'MA_5', ...] -> FUTURE는 1번 인덱스
        dummy[:, 1] = scaled_data # scaled_list에서 'FUTURE'가 두번째(인덱스 1)였음을 가정
        return self.scaler.inverse_transform(dummy)[:, 1]

    def run_prediction(self):
        """예측 실행"""
        if not self.models:
            messagebox.showerror("오류", "먼저 모델을 학습하거나 불러오세요!")
            return
        if self.X_test is None:
            messagebox.showerror("오류", "테스트 데이터가 없습니다. 먼저 데이터를 로드하세요!")
            return

        try:
            self.log("🔮 예측 실행 중...")

            # 예측 수행
            ensemble_pred = self.ensemble_predict(self.X_test[-100:])
            ensemble_pred_original = self.inverse_scale(ensemble_pred)
            y_test_original = self.inverse_scale(self.y_test[-100:])

            # 병목 구간 감지
            threshold = np.percentile(y_test_original, 85)
            bottlenecks = ensemble_pred_original > threshold

            # 결과 출력
            result_text = "=" * 50 + "\n"
            result_text += "🔮 예측 결과 (최근 10개)\n"
            result_text += "=" * 50 + "\n\n"

            for i in range(-10, 0):
                actual = y_test_original[i]
                predicted = ensemble_pred_original[i]
                error = abs(actual - predicted)
                is_bottleneck = "⚠️ 병목" if bottlenecks[i] else "정상"

                result_text += f"실제: {actual:.0f} | 예측: {predicted:.0f}\n"
                result_text += f"오차: {error:.1f} | 상태: {is_bottleneck}\n"
                result_text += "-" * 30 + "\n"

            # 병목 예측 통계
            n_bottlenecks = np.sum(bottlenecks)
            bottleneck_ratio = n_bottlenecks / len(bottlenecks) * 100

            result_text += f"\n📊 병목 예측 통계\n"
            result_text += f"병목 구간: {n_bottlenecks}개 ({bottleneck_ratio:.1f}%)\n"
            result_text += f"임계값: {threshold:.0f}\n"

            self.root.after(0, self.update_results, result_text)
            self.log("✅ 예측 완료!")

            # 시각화 업데이트
            self.update_visualization(y_test_original, ensemble_pred_original)

        except Exception as e:
            self.log(f"❌ 예측 실패: {str(e)}")
            messagebox.showerror("오류", f"예측 실패:\n{str(e)}")

    def save_models(self):
        """모델 저장"""
        if not self.models:
            messagebox.showerror("오류", "저장할 모델이 없습니다!")
            return

        try:
            # 저장 디렉토리 생성
            save_dir = f"saved_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(save_dir, exist_ok=True)

            # 각 모델 저장
            for model_name, model in self.models.items():
                model_path = os.path.join(save_dir, f'{model_name}_model.keras')
                model.save(model_path)
                self.log(f"💾 {model_name} 모델 저장: {model_path}")

            # 스케일러 저장
            scaler_path = os.path.join(save_dir, 'scaler.pkl')
            joblib.dump(self.scaler, scaler_path)
            self.log(f"💾 스케일러 저장: {scaler_path}")

            # 설정 저장
            config = {
                'seq_length': self.seq_var.get(),
                'future_minutes': self.future_var.get(),
                'input_shape': (self.X_train.shape[1], self.X_train.shape[2])
            }

            with open(os.path.join(save_dir, 'config.json'), 'w') as f:
                json.dump(config, f)

            self.log(f"✅ 모든 모델 및 설정 저장 완료! (폴더: {save_dir})")
            messagebox.showinfo("성공", f"모델이 '{save_dir}' 폴더에 성공적으로 저장되었습니다!")

        except Exception as e:
            self.log(f"❌ 모델 저장 실패: {str(e)}")
            messagebox.showerror("오류", f"모델 저장 실패:\n{str(e)}")

    def load_saved_models(self):
        """저장된 모델을 사용자가 선택하여 불러오기"""
        try:
            # 파일 선택 다이얼로그 열기
            initial_dir = 'saved_models' if os.path.exists('saved_models') else '.'
            model_path = filedialog.askopenfilename(
                title="모델 파일 선택",
                initialdir=os.path.abspath(initial_dir), # 시작 디렉토리 설정
                filetypes=[("Keras 모델", "*.keras"), ("모든 파일", "*.*")]
            )

            # 사용자가 파일을 선택한 경우
            if model_path:
                self.log("📂 선택한 모델 불러오는 중...")
                self.models.clear() # 기존 모델 초기화

                model_dir = os.path.dirname(model_path)
                model_filename = os.path.basename(model_path)
                model_name = model_filename.replace('_model.keras', '')

                # 모델 로드
                self.models[model_name] = load_model(model_path, compile=False)
                self.log(f"✅ {model_name} 모델 로드 완료: {model_filename}")

                # 스케일러 로드 (모델과 동일한 디렉토리에서 찾기)
                scaler_path = os.path.join(model_dir, 'scaler.pkl')
                if os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)
                    self.log("✅ 스케일러 로드 완료")
                else:
                    self.scaler = None # 스케일러가 없으면 None으로 설정
                    self.log("⚠️ 스케일러 파일(scaler.pkl)을 찾을 수 없습니다.")
                    messagebox.showwarning("경고", "모델과 함께 저장된 스케일러(scaler.pkl)를 찾을 수 없습니다.\n정확한 예측을 위해선 반드시 데이터 로드를 다시 진행해야 합니다.")

                # 설정 로드 (모델과 동일한 디렉토리에서 찾기)
                config_path = os.path.join(model_dir, 'config.json')
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    self.seq_var.set(config.get('seq_length', 30))
                    self.future_var.set(config.get('future_minutes', 10))
                    self.log("✅ 설정 파일 로드 완료")

                # 버튼 상태 업데이트
                self.save_btn.config(state=tk.NORMAL) # 불러온 모델은 다시 저장할 수 있도록 활성화
                if self.data is not None and self.scaler is not None:
                     self.predict_btn.config(state=tk.NORMAL)
                else:
                     self.predict_btn.config(state=tk.DISABLED)

                messagebox.showinfo("성공", f"'{model_name}' 모델을 성공적으로 불러왔습니다!")

        except Exception as e:
            self.log(f"❌ 모델 불러오기 실패: {str(e)}")
            messagebox.showerror("오류", f"모델 불러오기 실패:\n{str(e)}")

    def update_current_model(self, text):
        """현재 모델 라벨 업데이트"""
        self.current_model_label.config(text=text)

    def update_overall_progress(self, value):
        """전체 진행률 업데이트"""
        self.overall_progress['value'] = value

    def update_epoch_progress(self, value):
        """에폭 진행률 업데이트"""
        self.epoch_progress['value'] = value

    def update_current_epoch(self, text):
        """현재 에폭 라벨 업데이트"""
        self.current_epoch_label.config(text=text)

    def update_metrics(self, loss, val_loss, mae, val_mae):
        """메트릭 업데이트"""
        self.metric_labels["Loss:"].config(text=f"{loss:.4f}")
        self.metric_labels["Val Loss:"].config(text=f"{val_loss:.4f}")
        self.metric_labels["MAE:"].config(text=f"{mae:.4f}")
        self.metric_labels["Val MAE:"].config(text=f"{val_mae:.4f}")

    def update_results(self, text):
        """결과 텍스트 업데이트"""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, text)
        self.results_text.config(state=tk.DISABLED)

    def update_visualization(self, y_true, y_pred):
        """시각화 업데이트"""
        self.ax.clear()
        self.ax.set_facecolor('#3c3c3c')

        # 예측 결과 플롯
        self.ax.plot(y_true[:100], label=self.text_labels['actual'], color='#00ff00', linewidth=2)
        self.ax.plot(y_pred[:100], label=self.text_labels['predicted'], color='#ff6b6b', linewidth=1.5, alpha=0.8)

        # 언어에 따른 제목과 라벨 설정
        self.ax.set_title(self.text_labels['prediction_comparison'], color='white', fontsize=12, pad=10)
        self.ax.set_xlabel(self.text_labels['time_index'], color='white', fontsize=10)
        self.ax.set_ylabel(self.text_labels['logistics_volume'], color='white', fontsize=10)
        self.ax.tick_params(colors='white', labelsize=9)
        self.ax.legend(loc='upper right', framealpha=0.9)
        self.ax.grid(True, alpha=0.3)

        # tight_layout으로 레이아웃 조정
        self.fig.tight_layout()

        self.canvas.draw()

    def training_completed(self):
        """학습 완료 처리"""
        self.train_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)
        self.predict_btn.config(state=tk.NORMAL)
        self.save_btn.config(state=tk.NORMAL)
        self.load_model_btn.config(state=tk.NORMAL)

        messagebox.showinfo("완료", "모든 모델 학습이 완료되었습니다!")

    def reset_buttons(self):
        """버튼 상태 리셋"""
        self.train_btn.config(state=tk.NORMAL if self.data is not None else tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)
        self.load_model_btn.config(state=tk.NORMAL)

def main():
    """메인 함수"""
    print("="*60)
    print("반도체 물류 예측 하이브리드 AI 시스템 시작")
    print("="*60)

    # 한글 폰트 체크 및 안내
    if not font_initialized:
        print("\n⚠️  한글 폰트 설정 안내:")
        print("━"*50)

        system = platform.system()
        if system == 'Windows':
            print("Windows 사용자:")
            print("  1. 나눔고딕 폰트 설치: https://hangeul.naver.com/font")
            print("  2. 또는 이미 설치된 '맑은 고딕' 사용")
        elif system == 'Darwin':
            print("macOS 사용자:")
            print("  1. 나눔고딕 폰트 설치: https://hangeul.naver.com/font")
            print("  2. 설치 후 프로그램 재시작")
        else:
            print("Linux 사용자:")
            print("  1. sudo apt-get install fonts-nanum")
            print("  2. fc-cache -fv")

        print("\n현재는 영문으로 표시됩니다.")
        print("━"*50)
    else:
        print("✓ 한글 폰트 설정 완료")

    print("\n프로그램을 시작합니다...\n")

    root = tk.Tk()
    app = HybridModelGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
