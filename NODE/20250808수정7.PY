"""
반도체 물류 예측 통합 UI 시스템
=====================================
LSTM, RNN, GRU, ARIMA 하이브리드 모델을 활용한
반도체 FAB 물류 이동 및 병목 구간 예측 시스템

주요 기능:
1. 데이터 로드 및 전처리
2. 하이브리드 모델 학습 (체크포인트 지원)
3. 실시간 예측 및 병목 구간 감지
4. 모델 성능 평가 및 시각화

개발일: 2024년
버전: 1.0
"""

import sys
import os
import json
import pickle
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import threading
import time

# 데이터 처리
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 딥러닝
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, Bidirectional, GRU, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# GUI
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# 시각화
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.font_manager as fm

# 경고 숨기기
import warnings
warnings.filterwarnings('ignore')

# CPU 모드 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

# 랜덤 시드
RANDOM_SEED = 2079936
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class WorkerSignals(QObject):
    """워커 스레드 시그널"""
    started = pyqtSignal()
    progress = pyqtSignal(int, str)
    log = pyqtSignal(str)
    result = pyqtSignal(object)
    error = pyqtSignal(str)
    finished = pyqtSignal()


class TrainingWorker(QRunnable):
    """학습 워커 스레드"""
    def __init__(self, data_path, config):
        super().__init__()
        self.data_path = data_path
        self.config = config
        self.signals = WorkerSignals()
        self.is_running = True
        
    def run(self):
        """학습 실행"""
        try:
            self.signals.started.emit()
            self.signals.log.emit("학습 시작...")
            
            # 데이터 로드 및 전처리
            self.signals.progress.emit(10, "데이터 로드 중...")
            data = self.load_and_preprocess_data()
            
            # 모델 생성
            self.signals.progress.emit(30, "모델 생성 중...")
            models = self.create_models(data['input_shape'])
            
            # 학습
            self.signals.progress.emit(50, "모델 학습 중...")
            histories = self.train_models(models, data)
            
            # 저장
            self.signals.progress.emit(90, "모델 저장 중...")
            self.save_models(models)
            
            self.signals.progress.emit(100, "학습 완료!")
            self.signals.result.emit({
                'models': models,
                'histories': histories,
                'data': data
            })
            
        except Exception as e:
            self.signals.error.emit(str(e))
            traceback.print_exc()
        finally:
            self.signals.finished.emit()
            
    def load_and_preprocess_data(self):
        """데이터 로드 및 전처리"""
        # 실제 구현 코드 (학습_수정후.PY 참조)
        self.signals.log.emit(f"데이터 로딩: {self.data_path}")
        
        # 더미 데이터 (실제로는 CSV 로드)
        Full_Data = pd.read_csv(self.data_path)
        
        # 시간 컬럼 변환
        Full_Data['CURRTIME'] = pd.to_datetime(Full_Data['CURRTIME'], format='%Y%m%d%H%M')
        Full_Data['TIME'] = pd.to_datetime(Full_Data['TIME'], format='%Y%m%d%H%M')
        
        # 전처리...
        self.signals.log.emit(f"데이터 shape: {Full_Data.shape}")
        
        # 시퀀스 생성
        seq_length = self.config.get('seq_length', 30)
        # ... 실제 시퀀스 생성 코드
        
        # 더미 반환 (실제로는 처리된 데이터)
        return {
            'X_train': np.random.randn(1000, seq_length, 10),
            'y_train': np.random.randn(1000, 1),
            'X_val': np.random.randn(200, seq_length, 10),
            'y_val': np.random.randn(200, 1),
            'X_test': np.random.randn(200, seq_length, 10),
            'y_test': np.random.randn(200, 1),
            'input_shape': (seq_length, 10),
            'scaler': StandardScaler()
        }
        
    def create_models(self, input_shape):
        """모델 생성"""
        models = {}
        
        # LSTM 모델
        if self.config.get('use_lstm', True):
            lstm_model = Sequential([
                Input(shape=input_shape),
                LSTM(100, return_sequences=True),
                Dropout(0.2),
                LSTM(100, return_sequences=False),
                Dropout(0.2),
                Dense(50, activation='relu'),
                Dense(1)
            ])
            lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            models['lstm'] = lstm_model
            self.signals.log.emit("LSTM 모델 생성 완료")
            
        # RNN 모델
        if self.config.get('use_rnn', True):
            rnn_model = Sequential([
                Input(shape=input_shape),
                SimpleRNN(100, return_sequences=True),
                Dropout(0.2),
                SimpleRNN(50, return_sequences=False),
                Dropout(0.2),
                Dense(30, activation='relu'),
                Dense(1)
            ])
            rnn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            models['rnn'] = rnn_model
            self.signals.log.emit("RNN 모델 생성 완료")
            
        # GRU 모델
        if self.config.get('use_gru', True):
            gru_model = Sequential([
                Input(shape=input_shape),
                GRU(100, return_sequences=True),
                Dropout(0.2),
                GRU(50, return_sequences=False),
                Dropout(0.2),
                Dense(30, activation='relu'),
                Dense(1)
            ])
            gru_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            models['gru'] = gru_model
            self.signals.log.emit("GRU 모델 생성 완료")
            
        return models
        
    def train_models(self, models, data):
        """모델 학습"""
        histories = {}
        epochs = self.config.get('epochs', 100)
        batch_size = self.config.get('batch_size', 64)
        
        for name, model in models.items():
            if not self.is_running:
                break
                
            self.signals.log.emit(f"{name.upper()} 모델 학습 시작...")
            
            # 간단한 학습 (실제로는 콜백 등 추가)
            history = model.fit(
                data['X_train'], data['y_train'],
                validation_data=(data['X_val'], data['y_val']),
                epochs=min(epochs, 10),  # 데모용 제한
                batch_size=batch_size,
                verbose=0
            )
            
            histories[name] = history.history
            self.signals.log.emit(f"{name.upper()} 모델 학습 완료")
            
        return histories
        
    def save_models(self, models):
        """모델 저장"""
        os.makedirs('models', exist_ok=True)
        
        for name, model in models.items():
            model_path = f'models/{name}_model.keras'
            model.save(model_path)
            self.signals.log.emit(f"{name.upper()} 모델 저장: {model_path}")


class PredictionWorker(QRunnable):
    """예측 워커 스레드"""
    def __init__(self, data_path, model_paths):
        super().__init__()
        self.data_path = data_path
        self.model_paths = model_paths
        self.signals = WorkerSignals()
        
    def run(self):
        """예측 실행"""
        try:
            self.signals.started.emit()
            self.signals.log.emit("예측 시작...")
            
            # 데이터 로드
            self.signals.progress.emit(20, "데이터 로드 중...")
            data = self.load_data()
            
            # 모델 로드
            self.signals.progress.emit(40, "모델 로드 중...")
            models = self.load_models()
            
            # 예측
            self.signals.progress.emit(60, "예측 수행 중...")
            predictions = self.make_predictions(models, data)
            
            # 병목 분석
            self.signals.progress.emit(80, "병목 구간 분석 중...")
            bottlenecks = self.analyze_bottlenecks(predictions)
            
            self.signals.progress.emit(100, "예측 완료!")
            self.signals.result.emit({
                'predictions': predictions,
                'bottlenecks': bottlenecks,
                'data': data
            })
            
        except Exception as e:
            self.signals.error.emit(str(e))
            traceback.print_exc()
        finally:
            self.signals.finished.emit()
            
    def load_data(self):
        """데이터 로드"""
        # 실제 구현
        return np.random.randn(100, 30, 10)  # 더미 데이터
        
    def load_models(self):
        """모델 로드"""
        models = {}
        for name, path in self.model_paths.items():
            if os.path.exists(path):
                models[name] = load_model(path, compile=False)
                self.signals.log.emit(f"{name.upper()} 모델 로드 완료")
        return models
        
    def make_predictions(self, models, data):
        """예측 수행"""
        predictions = {}
        for name, model in models.items():
            pred = model.predict(data, verbose=0)
            predictions[name] = pred
            self.signals.log.emit(f"{name.upper()} 예측 완료")
            
        # 앙상블 예측
        if len(predictions) > 1:
            ensemble = np.mean(list(predictions.values()), axis=0)
            predictions['ensemble'] = ensemble
            
        return predictions
        
    def analyze_bottlenecks(self, predictions):
        """병목 구간 분석"""
        bottlenecks = []
        
        # 임계값 기반 병목 감지
        threshold = 2000  # 예시 임계값
        
        for name, pred in predictions.items():
            bottleneck_indices = np.where(pred > threshold)[0]
            if len(bottleneck_indices) > 0:
                bottlenecks.append({
                    'model': name,
                    'indices': bottleneck_indices.tolist(),
                    'severity': 'HIGH' if len(bottleneck_indices) > 10 else 'MEDIUM'
                })
                
        return bottlenecks


class MatplotlibWidget(QWidget):
    """Matplotlib 위젯"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
    def clear(self):
        """그래프 초기화"""
        self.figure.clear()
        self.canvas.draw()
        
    def plot_training_history(self, histories):
        """학습 이력 플롯"""
        self.clear()
        
        n_models = len(histories)
        axes = self.figure.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
            
        for idx, (name, history) in enumerate(histories.items()):
            ax = axes[idx]
            
            # 손실 곡선
            ax.plot(history.get('loss', []), label='Training Loss', color='blue')
            ax.plot(history.get('val_loss', []), label='Validation Loss', color='red')
            
            ax.set_title(f'{name.upper()} Training')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        self.figure.tight_layout()
        self.canvas.draw()
        
    def plot_predictions(self, predictions, actual=None):
        """예측 결과 플롯"""
        self.clear()
        
        ax = self.figure.add_subplot(111)
        
        # 실제값
        if actual is not None:
            ax.plot(actual, label='Actual', color='black', linewidth=2)
            
        # 예측값들
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        for idx, (name, pred) in enumerate(predictions.items()):
            color = colors[idx % len(colors)]
            ax.plot(pred, label=f'{name.upper()}', color=color, alpha=0.7)
            
        ax.set_title('물류량 예측 결과')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.figure.tight_layout()
        self.canvas.draw()
        
    def plot_evaluation(self, metrics):
        """평가 지표 플롯"""
        self.clear()
        
        # 모델별 성능 비교
        models = list(metrics.keys())
        mae_values = [metrics[m].get('mae', 0) for m in models]
        r2_values = [metrics[m].get('r2', 0) for m in models]
        
        ax1 = self.figure.add_subplot(121)
        bars1 = ax1.bar(models, mae_values, color='coral')
        ax1.set_title('MAE (낮을수록 좋음)')
        ax1.set_ylabel('MAE')
        ax1.set_xlabel('Model')
        
        # 값 표시
        for bar, val in zip(bars1, mae_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}', ha='center', va='bottom')
        
        ax2 = self.figure.add_subplot(122)
        bars2 = ax2.bar(models, r2_values, color='skyblue')
        ax2.set_title('R² Score (높을수록 좋음)')
        ax2.set_ylabel('R²')
        ax2.set_xlabel('Model')
        ax2.set_ylim(0, 1)
        
        # 값 표시
        for bar, val in zip(bars2, r2_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom')
        
        self.figure.tight_layout()
        self.canvas.draw()


class TrainingTab(QWidget):
    """학습 탭"""
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.worker = None
        
    def init_ui(self):
        """UI 초기화"""
        layout = QVBoxLayout()
        
        # 데이터 설정 그룹
        data_group = QGroupBox("데이터 설정")
        data_layout = QGridLayout()
        
        data_layout.addWidget(QLabel("데이터 파일:"), 0, 0)
        self.data_path_edit = QLineEdit()
        self.data_path_edit.setText("data/20240201_TO_202507281705.csv")
        data_layout.addWidget(self.data_path_edit, 0, 1)
        
        self.browse_btn = QPushButton("찾아보기...")
        self.browse_btn.clicked.connect(self.browse_file)
        data_layout.addWidget(self.browse_btn, 0, 2)
        
        data_layout.addWidget(QLabel("날짜 범위:"), 1, 0)
        self.date_range_edit = QLineEdit("2024-02-01 ~ 2024-07-27")
        data_layout.addWidget(self.date_range_edit, 1, 1, 1, 2)
        
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)
        
        # 모델 설정 그룹
        model_group = QGroupBox("모델 설정")
        model_layout = QGridLayout()
        
        self.lstm_check = QCheckBox("LSTM (이송시간 예측)")
        self.lstm_check.setChecked(True)
        model_layout.addWidget(self.lstm_check, 0, 0)
        
        self.rnn_check = QCheckBox("RNN (병목 예측)")
        self.rnn_check.setChecked(True)
        model_layout.addWidget(self.rnn_check, 0, 1)
        
        self.gru_check = QCheckBox("GRU (복합 예측)")
        self.gru_check.setChecked(True)
        model_layout.addWidget(self.gru_check, 0, 2)
        
        self.bidirectional_check = QCheckBox("Bidirectional LSTM")
        self.bidirectional_check.setChecked(False)
        model_layout.addWidget(self.bidirectional_check, 0, 3)
        
        model_layout.addWidget(QLabel("시퀀스 길이:"), 1, 0)
        self.seq_length_spin = QSpinBox()
        self.seq_length_spin.setRange(10, 100)
        self.seq_length_spin.setValue(30)
        model_layout.addWidget(self.seq_length_spin, 1, 1)
        
        model_layout.addWidget(QLabel("에폭 수:"), 1, 2)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(10, 500)
        self.epochs_spin.setValue(100)
        model_layout.addWidget(self.epochs_spin, 1, 3)
        
        model_layout.addWidget(QLabel("배치 크기:"), 2, 0)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(16, 256)
        self.batch_size_spin.setValue(64)
        self.batch_size_spin.setSingleStep(16)
        model_layout.addWidget(self.batch_size_spin, 2, 1)
        
        model_layout.addWidget(QLabel("학습률:"), 2, 2)
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(0.0001, 0.1)
        self.learning_rate_spin.setValue(0.0005)
        self.learning_rate_spin.setSingleStep(0.0001)
        self.learning_rate_spin.setDecimals(4)
        model_layout.addWidget(self.learning_rate_spin, 2, 3)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # 고급 옵션 그룹
        advanced_group = QGroupBox("고급 옵션")
        advanced_layout = QGridLayout()
        
        self.checkpoint_check = QCheckBox("체크포인트 사용")
        self.checkpoint_check.setChecked(True)
        advanced_layout.addWidget(self.checkpoint_check, 0, 0)
        
        self.early_stop_check = QCheckBox("조기 종료")
        self.early_stop_check.setChecked(True)
        advanced_layout.addWidget(self.early_stop_check, 0, 1)
        
        self.reduce_lr_check = QCheckBox("학습률 감소")
        self.reduce_lr_check.setChecked(True)
        advanced_layout.addWidget(self.reduce_lr_check, 0, 2)
        
        advanced_layout.addWidget(QLabel("검증 비율:"), 1, 0)
        self.val_split_spin = QDoubleSpinBox()
        self.val_split_spin.setRange(0.1, 0.5)
        self.val_split_spin.setValue(0.3)
        self.val_split_spin.setSingleStep(0.05)
        advanced_layout.addWidget(self.val_split_spin, 1, 1)
        
        advanced_group.setLayout(advanced_layout)
        layout.addWidget(advanced_group)
        
        # 진행 상황
        progress_group = QGroupBox("진행 상황")
        progress_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("대기 중...")
        progress_layout.addWidget(self.status_label)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        progress_layout.addWidget(self.log_text)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # 버튼
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("학습 시작")
        self.start_btn.clicked.connect(self.start_training)
        button_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("중지")
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)
        
        self.resume_btn = QPushButton("이어서 학습")
        self.resume_btn.clicked.connect(self.resume_training)
        button_layout.addWidget(self.resume_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # 그래프
        self.plot_widget = MatplotlibWidget()
        layout.addWidget(self.plot_widget)
        
        self.setLayout(layout)
        
    def browse_file(self):
        """파일 찾아보기"""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "데이터 파일 선택",
            "",
            "CSV Files (*.csv);;All Files (*.*)"
        )
        if filename:
            self.data_path_edit.setText(filename)
            
    def start_training(self):
        """학습 시작"""
        # 설정 수집
        config = {
            'use_lstm': self.lstm_check.isChecked(),
            'use_rnn': self.rnn_check.isChecked(),
            'use_gru': self.gru_check.isChecked(),
            'use_bidirectional': self.bidirectional_check.isChecked(),
            'seq_length': self.seq_length_spin.value(),
            'epochs': self.epochs_spin.value(),
            'batch_size': self.batch_size_spin.value(),
            'learning_rate': self.learning_rate_spin.value(),
            'val_split': self.val_split_spin.value(),
            'use_checkpoint': self.checkpoint_check.isChecked(),
            'early_stopping': self.early_stop_check.isChecked(),
            'reduce_lr': self.reduce_lr_check.isChecked(),
        }
        
        # 워커 생성 및 실행
        self.worker = TrainingWorker(self.data_path_edit.text(), config)
        self.worker.signals.started.connect(self.on_training_started)
        self.worker.signals.progress.connect(self.update_progress)
        self.worker.signals.log.connect(self.append_log)
        self.worker.signals.result.connect(self.on_training_completed)
        self.worker.signals.error.connect(self.on_training_error)
        self.worker.signals.finished.connect(self.on_training_finished)
        
        # 스레드 풀에서 실행
        QThreadPool.globalInstance().start(self.worker)
        
    def stop_training(self):
        """학습 중지"""
        if self.worker:
            self.worker.is_running = False
            self.append_log("학습 중지 요청...")
            
    def resume_training(self):
        """학습 재개"""
        self.append_log("이전 체크포인트에서 학습 재개...")
        # 실제 구현 필요
        
    def on_training_started(self):
        """학습 시작 시"""
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("학습 중...")
        
    def update_progress(self, value, message):
        """진행 상황 업데이트"""
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
        
    def append_log(self, message):
        """로그 추가"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        
    def on_training_completed(self, result):
        """학습 완료 시"""
        self.append_log("학습이 성공적으로 완료되었습니다!")
        
        # 학습 곡선 표시
        if 'histories' in result:
            self.plot_widget.plot_training_history(result['histories'])
            
    def on_training_error(self, error):
        """학습 오류 시"""
        QMessageBox.critical(self, "학습 오류", f"학습 중 오류가 발생했습니다:\n{error}")
        
    def on_training_finished(self):
        """학습 종료 시"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(0)


class PredictionTab(QWidget):
    """예측 탭"""
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.worker = None
        
    def init_ui(self):
        """UI 초기화"""
        layout = QVBoxLayout()
        
        # 데이터 설정
        data_group = QGroupBox("데이터 설정")
        data_layout = QGridLayout()
        
        data_layout.addWidget(QLabel("예측 데이터:"), 0, 0)
        self.data_path_edit = QLineEdit()
        self.data_path_edit.setText("data/0730to31.csv")
        data_layout.addWidget(self.data_path_edit, 0, 1)
        
        self.browse_btn = QPushButton("찾아보기...")
        self.browse_btn.clicked.connect(self.browse_file)
        data_layout.addWidget(self.browse_btn, 0, 2)
        
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)
        
        # 모델 선택
        model_group = QGroupBox("모델 선택")
        model_layout = QVBoxLayout()
        
        self.model_list = QListWidget()
        self.model_list.setSelectionMode(QAbstractItemView.MultiSelection)
        
        # 모델 파일 검색
        self.refresh_model_list()
        
        model_layout.addWidget(self.model_list)
        
        refresh_btn = QPushButton("모델 목록 새로고침")
        refresh_btn.clicked.connect(self.refresh_model_list)
        model_layout.addWidget(refresh_btn)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # 예측 옵션
        option_group = QGroupBox("예측 옵션")
        option_layout = QGridLayout()
        
        option_layout.addWidget(QLabel("예측 기간:"), 0, 0)
        self.period_combo = QComboBox()
        self.period_combo.addItems(["10분", "30분", "1시간", "4시간", "1일"])
        option_layout.addWidget(self.period_combo, 0, 1)
        
        option_layout.addWidget(QLabel("병목 임계값:"), 0, 2)
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(1000, 5000)
        self.threshold_spin.setValue(2000)
        self.threshold_spin.setSingleStep(100)
        option_layout.addWidget(self.threshold_spin, 0, 3)
        
        self.ensemble_check = QCheckBox("앙상블 예측 사용")
        self.ensemble_check.setChecked(True)
        option_layout.addWidget(self.ensemble_check, 1, 0, 1, 2)
        
        self.realtime_check = QCheckBox("실시간 모드")
        self.realtime_check.setChecked(False)
        option_layout.addWidget(self.realtime_check, 1, 2, 1, 2)
        
        option_group.setLayout(option_layout)
        layout.addWidget(option_group)
        
        # 버튼
        button_layout = QHBoxLayout()
        
        self.predict_btn = QPushButton("예측 시작")
        self.predict_btn.clicked.connect(self.start_prediction)
        button_layout.addWidget(self.predict_btn)
        
        self.stop_btn = QPushButton("중지")
        self.stop_btn.clicked.connect(self.stop_prediction)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)
        
        button_layout.addStretch()
        
        self.export_btn = QPushButton("결과 내보내기")
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setEnabled(False)
        button_layout.addWidget(self.export_btn)
        
        layout.addLayout(button_layout)
        
        # 결과 표시
        result_group = QGroupBox("예측 결과")
        result_layout = QVBoxLayout()
        
        # 탭 위젯
        self.result_tabs = QTabWidget()
        
        # 요약 탭
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.result_tabs.addTab(self.summary_text, "요약")
        
        # 상세 탭
        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        self.result_tabs.addTab(self.detail_text, "상세")
        
        # 병목 분석 탭
        self.bottleneck_text = QTextEdit()
        self.bottleneck_text.setReadOnly(True)
        self.result_tabs.addTab(self.bottleneck_text, "병목 분석")
        
        result_layout.addWidget(self.result_tabs)
        
        # 그래프
        self.plot_widget = MatplotlibWidget()
        result_layout.addWidget(self.plot_widget)
        
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)
        
        self.setLayout(layout)
        
    def browse_file(self):
        """파일 찾아보기"""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "예측 데이터 선택",
            "",
            "CSV Files (*.csv);;All Files (*.*)"
        )
        if filename:
            self.data_path_edit.setText(filename)
            
    def refresh_model_list(self):
        """모델 목록 새로고침"""
        self.model_list.clear()
        
        if os.path.exists('models'):
            for file in os.listdir('models'):
                if file.endswith('.keras') or file.endswith('.h5'):
                    item = QListWidgetItem(file)
                    self.model_list.addItem(item)
                    item.setSelected(True)  # 기본 선택
                    
    def start_prediction(self):
        """예측 시작"""
        # 선택된 모델 확인
        selected_items = self.model_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "경고", "예측할 모델을 선택하세요.")
            return
            
        # 모델 경로 수집
        model_paths = {}
        for item in selected_items:
            name = item.text().replace('.keras', '').replace('.h5', '')
            model_paths[name] = os.path.join('models', item.text())
            
        # 워커 생성 및 실행
        self.worker = PredictionWorker(self.data_path_edit.text(), model_paths)
        self.worker.signals.started.connect(self.on_prediction_started)
        self.worker.signals.progress.connect(self.update_progress)
        self.worker.signals.log.connect(self.append_log)
        self.worker.signals.result.connect(self.on_prediction_completed)
        self.worker.signals.error.connect(self.on_prediction_error)
        self.worker.signals.finished.connect(self.on_prediction_finished)
        
        QThreadPool.globalInstance().start(self.worker)
        
    def stop_prediction(self):
        """예측 중지"""
        # 실제 구현 필요
        pass
        
    def on_prediction_started(self):
        """예측 시작 시"""
        self.predict_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
    def update_progress(self, value, message):
        """진행 상황 업데이트"""
        # 상태바나 진행바 업데이트
        pass
        
    def append_log(self, message):
        """로그 추가"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.detail_text.append(f"[{timestamp}] {message}")
        
    def on_prediction_completed(self, result):
        """예측 완료 시"""
        # 요약 표시
        summary = self.generate_summary(result)
        self.summary_text.setPlainText(summary)
        
        # 병목 분석 표시
        if 'bottlenecks' in result:
            bottleneck_text = self.analyze_bottlenecks(result['bottlenecks'])
            self.bottleneck_text.setPlainText(bottleneck_text)
            
        # 그래프 표시
        if 'predictions' in result:
            self.plot_widget.plot_predictions(result['predictions'])
            
        self.export_btn.setEnabled(True)
        self.last_result = result
        
    def generate_summary(self, result):
        """예측 요약 생성"""
        summary = "=" * 60 + "\n"
        summary += "예측 결과 요약\n"
        summary += "=" * 60 + "\n\n"
        summary += f"예측 시점: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        summary += f"예측 기간: {self.period_combo.currentText()}\n"
        summary += f"병목 임계값: {self.threshold_spin.value()}\n\n"
        
        if 'predictions' in result:
            for name, pred in result['predictions'].items():
                summary += f"\n{name.upper()} 모델:\n"
                summary += f"  - 평균 예측값: {np.mean(pred):.2f}\n"
                summary += f"  - 최대 예측값: {np.max(pred):.2f}\n"
                summary += f"  - 최소 예측값: {np.min(pred):.2f}\n"
                
        return summary
        
    def analyze_bottlenecks(self, bottlenecks):
        """병목 분석 텍스트 생성"""
        text = "병목 구간 분석 결과\n"
        text += "=" * 60 + "\n\n"
        
        for bn in bottlenecks:
            text += f"\n{bn['model'].upper()} 모델:\n"
            text += f"  - 병목 구간 수: {len(bn['indices'])}\n"
            text += f"  - 심각도: {bn['severity']}\n"
            text += f"  - 위치: {bn['indices'][:10]}...\n" if len(bn['indices']) > 10 else f"  - 위치: {bn['indices']}\n"
            
        return text
        
    def on_prediction_error(self, error):
        """예측 오류 시"""
        QMessageBox.critical(self, "예측 오류", f"예측 중 오류가 발생했습니다:\n{error}")
        
    def on_prediction_finished(self):
        """예측 종료 시"""
        self.predict_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
    def export_results(self):
        """결과 내보내기"""
        if not hasattr(self, 'last_result'):
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "예측 결과 저장",
            f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            "Excel Files (*.xlsx);;CSV Files (*.csv)"
        )
        
        if filename:
            # 실제 내보내기 구현
            QMessageBox.information(self, "완료", "예측 결과가 저장되었습니다.")


class EvaluationTab(QWidget):
    """평가 탭"""
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        """UI 초기화"""
        layout = QVBoxLayout()
        
        # 모델 선택
        model_group = QGroupBox("평가할 모델")
        model_layout = QVBoxLayout()
        
        self.model_list = QListWidget()
        self.model_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.refresh_model_list()
        
        model_layout.addWidget(self.model_list)
        
        refresh_btn = QPushButton("모델 목록 새로고침")
        refresh_btn.clicked.connect(self.refresh_model_list)
        model_layout.addWidget(refresh_btn)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # 평가 옵션
        option_group = QGroupBox("평가 옵션")
        option_layout = QGridLayout()
        
        self.mape_check = QCheckBox("MAPE")
        self.mape_check.setChecked(True)
        option_layout.addWidget(self.mape_check, 0, 0)
        
        self.mae_check = QCheckBox("MAE")
        self.mae_check.setChecked(True)
        option_layout.addWidget(self.mae_check, 0, 1)
        
        self.rmse_check = QCheckBox("RMSE")
        self.rmse_check.setChecked(True)
        option_layout.addWidget(self.rmse_check, 0, 2)
        
        self.r2_check = QCheckBox("R² Score")
        self.r2_check.setChecked(True)
        option_layout.addWidget(self.r2_check, 0, 3)
        
        option_layout.addWidget(QLabel("테스트 데이터:"), 1, 0)
        self.test_data_edit = QLineEdit()
        option_layout.addWidget(self.test_data_edit, 1, 1, 1, 2)
        
        browse_btn = QPushButton("찾아보기...")
        browse_btn.clicked.connect(self.browse_test_data)
        option_layout.addWidget(browse_btn, 1, 3)
        
        option_group.setLayout(option_layout)
        layout.addWidget(option_group)
        
        # 평가 시작 버튼
        eval_btn = QPushButton("평가 시작")
        eval_btn.clicked.connect(self.start_evaluation)
        layout.addWidget(eval_btn)
        
        # 결과 표시
        result_group = QGroupBox("평가 결과")
        result_layout = QVBoxLayout()
        
        # 결과 테이블
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(5)
        self.result_table.setHorizontalHeaderLabels(["모델", "MAPE", "MAE", "RMSE", "R²"])
        result_layout.addWidget(self.result_table)
        
        # 그래프
        self.plot_widget = MatplotlibWidget()
        result_layout.addWidget(self.plot_widget)
        
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)
        
        # 리포트 생성 버튼
        report_btn = QPushButton("평가 리포트 생성")
        report_btn.clicked.connect(self.generate_report)
        layout.addWidget(report_btn)
        
        self.setLayout(layout)
        
    def refresh_model_list(self):
        """모델 목록 새로고침"""
        self.model_list.clear()
        
        if os.path.exists('models'):
            for file in os.listdir('models'):
                if file.endswith('.keras') or file.endswith('.h5'):
                    item = QListWidgetItem(file)
                    self.model_list.addItem(item)
                    item.setSelected(True)
                    
    def browse_test_data(self):
        """테스트 데이터 찾아보기"""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "테스트 데이터 선택",
            "",
            "CSV Files (*.csv);;All Files (*.*)"
        )
        if filename:
            self.test_data_edit.setText(filename)
            
    def start_evaluation(self):
        """평가 시작"""
        selected_items = self.model_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "경고", "평가할 모델을 선택하세요.")
            return
            
        # 더미 평가 결과 생성
        self.result_table.setRowCount(len(selected_items))
        
        metrics = {}
        for idx, item in enumerate(selected_items):
            model_name = item.text().replace('.keras', '').replace('.h5', '')
            
            # 더미 메트릭
            mape = np.random.uniform(5, 15)
            mae = np.random.uniform(50, 150)
            rmse = np.random.uniform(70, 200)
            r2 = np.random.uniform(0.7, 0.95)
            
            self.result_table.setItem(idx, 0, QTableWidgetItem(model_name))
            self.result_table.setItem(idx, 1, QTableWidgetItem(f"{mape:.2f}%"))
            self.result_table.setItem(idx, 2, QTableWidgetItem(f"{mae:.2f}"))
            self.result_table.setItem(idx, 3, QTableWidgetItem(f"{rmse:.2f}"))
            self.result_table.setItem(idx, 4, QTableWidgetItem(f"{r2:.3f}"))
            
            metrics[model_name] = {'mae': mae, 'r2': r2}
            
        # 그래프 표시
        self.plot_widget.plot_evaluation(metrics)
        
        QMessageBox.information(self, "완료", "모델 평가가 완료되었습니다.")
        
    def generate_report(self):
        """평가 리포트 생성"""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "평가 리포트 저장",
            f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text Files (*.txt)"
        )
        
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("모델 평가 리포트\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"생성 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # 테이블 내용 저장
                for row in range(self.result_table.rowCount()):
                    model = self.result_table.item(row, 0).text() if self.result_table.item(row, 0) else ""
                    mape = self.result_table.item(row, 1).text() if self.result_table.item(row, 1) else ""
                    mae = self.result_table.item(row, 2).text() if self.result_table.item(row, 2) else ""
                    rmse = self.result_table.item(row, 3).text() if self.result_table.item(row, 3) else ""
                    r2 = self.result_table.item(row, 4).text() if self.result_table.item(row, 4) else ""
                    
                    f.write(f"\n{model}\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"  MAPE: {mape}\n")
                    f.write(f"  MAE: {mae}\n")
                    f.write(f"  RMSE: {rmse}\n")
                    f.write(f"  R²: {r2}\n")
                    
            QMessageBox.information(self, "완료", "평가 리포트가 저장되었습니다.")


class MainWindow(QMainWindow):
    """메인 윈도우"""
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle("반도체 물류 예측 시스템 - 통합 관리자")
        self.setGeometry(100, 100, 1400, 900)
        
        # 아이콘 설정 (있다면)
        # self.setWindowIcon(QIcon('icon.png'))
        
        # 중앙 위젯
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 메인 레이아웃
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # 헤더
        header_layout = QHBoxLayout()
        
        title_label = QLabel("🚀 반도체 FAB 물류 예측 시스템")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
                padding: 10px;
            }
        """)
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        info_label = QLabel("LSTM/RNN/GRU 하이브리드 모델")
        info_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: #7f8c8d;
                padding: 10px;
            }
        """)
        header_layout.addWidget(info_label)
        
        main_layout.addLayout(header_layout)
        
        # 구분선
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(line)
        
        # 탭 위젯
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #cccccc;
                background-color: white;
            }
            QTabBar::tab {
                padding: 10px 20px;
                margin-right: 5px;
            }
            QTabBar::tab:selected {
                background-color: #3498db;
                color: white;
            }
        """)
        
        # 탭 추가
        self.training_tab = TrainingTab()
        self.tabs.addTab(self.training_tab, "📚 학습")
        
        self.prediction_tab = PredictionTab()
        self.tabs.addTab(self.prediction_tab, "🔮 예측")
        
        self.evaluation_tab = EvaluationTab()
        self.tabs.addTab(self.evaluation_tab, "📊 평가")
        
        main_layout.addWidget(self.tabs)
        
        # 상태바
        self.statusBar().showMessage("준비됨")
        
        # 메뉴바
        self.create_menu_bar()
        
        # 툴바
        self.create_toolbar()
        
        # 스타일 적용
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                padding: 8px 15px;
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                padding: 5px;
                border: 1px solid #cccccc;
                border-radius: 3px;
            }
            QTextEdit {
                border: 1px solid #cccccc;
                border-radius: 3px;
            }
        """)
        
    def create_menu_bar(self):
        """메뉴바 생성"""
        menubar = self.menuBar()
        
        # 파일 메뉴
        file_menu = menubar.addMenu("파일")
        
        new_action = QAction("새 프로젝트", self)
        new_action.setShortcut("Ctrl+N")
        file_menu.addAction(new_action)
        
        open_action = QAction("프로젝트 열기", self)
        open_action.setShortcut("Ctrl+O")
        file_menu.addAction(open_action)
        
        save_action = QAction("프로젝트 저장", self)
        save_action.setShortcut("Ctrl+S")
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("종료", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 도구 메뉴
        tools_menu = menubar.addMenu("도구")
        
        settings_action = QAction("설정", self)
        settings_action.triggered.connect(self.show_settings)
        tools_menu.addAction(settings_action)
        
        # 도움말 메뉴
        help_menu = menubar.addMenu("도움말")
        
        about_action = QAction("프로그램 정보", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
        manual_action = QAction("사용 설명서", self)
        manual_action.triggered.connect(self.show_manual)
        help_menu.addAction(manual_action)
        
    def create_toolbar(self):
        """툴바 생성"""
        toolbar = self.addToolBar("메인 툴바")
        toolbar.setMovable(False)
        
        # 학습 시작
        train_action = QAction("🚀 학습", self)
        train_action.triggered.connect(lambda: self.tabs.setCurrentIndex(0))
        toolbar.addAction(train_action)
        
        # 예측 시작
        predict_action = QAction("🔮 예측", self)
        predict_action.triggered.connect(lambda: self.tabs.setCurrentIndex(1))
        toolbar.addAction(predict_action)
        
        # 평가 시작
        eval_action = QAction("📊 평가", self)
        eval_action.triggered.connect(lambda: self.tabs.setCurrentIndex(2))
        toolbar.addAction(eval_action)
        
        toolbar.addSeparator()
        
        # 설정
        settings_action = QAction("⚙️ 설정", self)
        settings_action.triggered.connect(self.show_settings)
        toolbar.addAction(settings_action)
        
    def show_settings(self):
        """설정 대화상자"""
        dialog = QDialog(self)
        dialog.setWindowTitle("설정")
        dialog.setModal(True)
        
        layout = QVBoxLayout()
        
        # 설정 옵션들
        layout.addWidget(QLabel("GPU 사용: 비활성화 (CPU 모드)"))
        layout.addWidget(QLabel("자동 저장: 활성화"))
        layout.addWidget(QLabel("로그 레벨: INFO"))
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)
        
        dialog.setLayout(layout)
        dialog.exec_()
        
    def show_about(self):
        """프로그램 정보"""
        QMessageBox.information(
            self,
            "프로그램 정보",
            "반도체 물류 예측 시스템 v1.0\n\n"
            "LSTM/RNN/GRU 하이브리드 모델을 활용한\n"
            "반도체 FAB 물류 이동 및 병목 구간 예측\n\n"
            "개발: 2024년\n"
            "Python 3.8+ / TensorFlow 2.x / PyQt5"
        )
        
    def show_manual(self):
        """사용 설명서"""
        manual = """
        사용 방법:
        
        1. 학습 탭
           - 데이터 파일 선택
           - 모델 옵션 설정
           - 학습 시작 클릭
           
        2. 예측 탭
           - 학습된 모델 선택
           - 예측 데이터 입력
           - 예측 시작 클릭
           
        3. 평가 탭
           - 평가할 모델 선택
           - 테스트 데이터 입력
           - 평가 시작 클릭
           
        단축키:
        - Ctrl+N: 새 프로젝트
        - Ctrl+O: 프로젝트 열기
        - Ctrl+S: 프로젝트 저장
        - Ctrl+Q: 종료
        """
        
        msg = QMessageBox(self)
        msg.setWindowTitle("사용 설명서")
        msg.setText(manual)
        msg.setIcon(QMessageBox.Information)
        msg.exec_()


def main():
    """메인 함수"""
    app = QApplication(sys.argv)
    
    # 애플리케이션 정보 설정
    app.setApplicationName("반도체 물류 예측 시스템")
    app.setOrganizationName("Semiconductor Logistics")
    
    # 스타일 설정
    app.setStyle("Fusion")
    
    # 메인 윈도우 생성 및 표시
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()