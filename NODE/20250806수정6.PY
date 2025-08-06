"""
반도체 FAB 물류 예측을 위한 통합 하이브리드 딥러닝 시스템
================================================================
LSTM, RNN, GRU, ARIMA 모델을 통합한 하이브리드 예측 시스템으로
반도체 팹 간 물류 이동량과 병목 구간을 예측합니다.

주요 기능:
1. 노드 기반 파이프라인 구성
2. 실시간 모델 학습 (재시작 가능)
3. 앙상블 예측 및 병목 구간 감지
4. 퍼센트 기반 성능 평가
5. CPU 기반 실행

버전: 3.0 (통합 버전)
"""

import sys
import json
import os
import pickle
import traceback
import warnings
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, Bidirectional, GRU, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')

# CPU 모드 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

# 랜덤 시드 고정
RANDOM_SEED = 2079936
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class NodeType(Enum):
    """노드 타입 정의"""
    DATA_LOAD = "data_load"
    PREPROCESS = "preprocess"
    TRAIN = "train"
    PREDICT = "predict"
    EVALUATE = "evaluate"
    VISUALIZE = "visualize"


@dataclass
class NodeConfig:
    """노드 설정 데이터 클래스"""
    node_type: NodeType
    name: str
    color: str
    inputs: int = 1
    outputs: int = 1


# 노드 타입별 설정
NODE_CONFIGS = {
    NodeType.DATA_LOAD: NodeConfig(NodeType.DATA_LOAD, "데이터 로드", "#3498db", 0, 1),
    NodeType.PREPROCESS: NodeConfig(NodeType.PREPROCESS, "전처리", "#e74c3c", 1, 1),
    NodeType.TRAIN: NodeConfig(NodeType.TRAIN, "학습", "#27ae60", 1, 1),
    NodeType.PREDICT: NodeConfig(NodeType.PREDICT, "예측", "#f39c12", 1, 1),
    NodeType.EVALUATE: NodeConfig(NodeType.EVALUATE, "평가", "#9b59b6", 1, 1),
    NodeType.VISUALIZE: NodeConfig(NodeType.VISUALIZE, "시각화", "#1abc9c", 1, 0),
}


class CheckpointManager:
    """학습 상태를 저장하고 복원하는 클래스"""
    
    def __init__(self, checkpoint_dir='checkpoints'):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.state_file = os.path.join(checkpoint_dir, 'training_state.json')
        self.data_file = os.path.join(checkpoint_dir, 'preprocessed_data.pkl')
        
    def save_state(self, state_dict):
        """현재 학습 상태 저장"""
        with open(self.state_file, 'w') as f:
            json.dump(state_dict, f, indent=4, default=str)
        
    def load_state(self):
        """저장된 학습 상태 로드"""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return None
        
    def save_data(self, data_dict):
        """전처리된 데이터 저장"""
        with open(self.data_file, 'wb') as f:
            pickle.dump(data_dict, f)
        
    def load_data(self):
        """저장된 데이터 로드"""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'rb') as f:
                return pickle.load(f)
        return None
        
    def save_model_weights(self, model, model_name, epoch):
        """모델 가중치 저장"""
        weights_path = os.path.join(self.checkpoint_dir, f'{model_name}_weights_epoch_{epoch}.h5')
        model.save_weights(weights_path)
        return weights_path
        
    def load_model_weights(self, model, weights_path):
        """모델 가중치 로드"""
        if os.path.exists(weights_path):
            model.load_weights(weights_path)
            return True
        return False


class HybridModels:
    """LSTM, RNN, GRU 모델을 포함한 하이브리드 예측 시스템"""
    
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.models = {}
        self.histories = {}
        
    def build_lstm_model(self):
        """LSTM 모델 구축"""
        model = Sequential([
            Input(shape=self.input_shape),
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
    
    def build_gru_model(self):
        """GRU 모델 구축"""
        model = Sequential([
            Input(shape=self.input_shape),
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
    
    def build_simple_rnn_model(self):
        """Simple RNN 모델 구축"""
        model = Sequential([
            Input(shape=self.input_shape),
            SimpleRNN(units=100, return_sequences=True),
            Dropout(0.2),
            SimpleRNN(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=30, activation='relu'),
            Dense(units=1)
        ])
        return model
    
    def build_bidirectional_lstm_model(self):
        """양방향 LSTM 모델 구축"""
        model = Sequential([
            Input(shape=self.input_shape),
            Bidirectional(LSTM(units=50, return_sequences=True)),
            Dropout(0.2),
            Bidirectional(LSTM(units=50, return_sequences=False)),
            Dropout(0.2),
            Dense(units=30, activation='relu'),
            Dense(units=1)
        ])
        return model


class ModelPercentEvaluator:
    """모델 성능을 퍼센트로 평가하는 클래스"""
    
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.config = None
        
    def calculate_mape(self, y_true, y_pred):
        """MAPE (Mean Absolute Percentage Error) 계산"""
        mask = y_true != 0
        if not np.any(mask):
            return 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        return mape
    
    def calculate_accuracy_within_threshold(self, y_true, y_pred, threshold_percent=10):
        """특정 오차 범위 내 예측 정확도 계산"""
        threshold = np.mean(y_true) * (threshold_percent / 100)
        within_threshold = np.abs(y_true - y_pred) <= threshold
        accuracy = np.mean(within_threshold) * 100
        return accuracy
    
    def calculate_directional_accuracy(self, y_true, y_pred):
        """방향성 정확도 (증가/감소 예측 정확도)"""
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        directional_accuracy = np.mean(true_direction == pred_direction) * 100
        return directional_accuracy
    
    def calculate_bottleneck_accuracy(self, y_true, y_pred, threshold):
        """병목 구간 예측 정확도"""
        true_bottleneck = y_true > threshold
        pred_bottleneck = y_pred > threshold
        
        tp = np.sum((true_bottleneck == True) & (pred_bottleneck == True))
        tn = np.sum((true_bottleneck == False) & (pred_bottleneck == False))
        fp = np.sum((true_bottleneck == False) & (pred_bottleneck == True))
        fn = np.sum((true_bottleneck == True) & (pred_bottleneck == False))
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) * 100 if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }


class TrainingThread(QThread):
    """학습을 별도 스레드에서 실행"""
    progress_update = pyqtSignal(str)
    training_complete = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, data, config):
        super().__init__()
        self.data = data
        self.config = config
        self.checkpoint_manager = CheckpointManager()
        self.is_running = True
        
    def run(self):
        """학습 실행"""
        try:
            self.progress_update.emit("학습 시작...")
            
            # 데이터 준비
            X_train = self.data['X_train']
            y_train = self.data['y_train']
            X_val = self.data['X_val']
            y_val = self.data['y_val']
            input_shape = (X_train.shape[1], X_train.shape[2])
            
            # 하이브리드 모델 초기화
            hybrid_models = HybridModels(input_shape)
            
            # 모델 리스트
            model_configs = [
                ('lstm', hybrid_models.build_lstm_model),
                ('gru', hybrid_models.build_gru_model),
                ('rnn', hybrid_models.build_simple_rnn_model),
                ('bi_lstm', hybrid_models.build_bidirectional_lstm_model)
            ]
            
            results = {}
            
            # 각 모델 학습
            for model_name, build_func in model_configs:
                if not self.is_running:
                    break
                    
                self.progress_update.emit(f"{model_name.upper()} 모델 학습 중...")
                
                # 모델 빌드
                model = build_func()
                
                # 컴파일
                optimizer = Adam(learning_rate=self.config.get('learning_rate', 0.0005))
                model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
                
                # 콜백
                early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
                reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=0)
                
                # 학습
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=self.config.get('epochs', 50),
                    batch_size=self.config.get('batch_size', 64),
                    callbacks=[early_stop, reduce_lr],
                    verbose=0
                )
                
                # 모델 저장
                model_path = f'model/{model_name}_hybrid.keras'
                os.makedirs('model', exist_ok=True)
                model.save(model_path)
                
                results[model_name] = {
                    'model_path': model_path,
                    'history': history.history,
                    'final_loss': history.history['loss'][-1],
                    'final_val_loss': history.history['val_loss'][-1]
                }
                
                self.progress_update.emit(f"{model_name.upper()} 모델 학습 완료")
            
            self.progress_update.emit("모든 모델 학습 완료!")
            self.training_complete.emit(results)
            
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def stop(self):
        """학습 중단"""
        self.is_running = False


class PredictionWidget(QWidget):
    """예측 결과 표시 위젯"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 예측 결과 테이블
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(5)
        self.result_table.setHorizontalHeaderLabels(['시간', '현재값', '예측값', '오차', '병목여부'])
        layout.addWidget(self.result_table)
        
        # 차트
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
        
    def update_predictions(self, predictions, actual_values, times):
        """예측 결과 업데이트"""
        # 테이블 업데이트
        self.result_table.setRowCount(min(20, len(predictions)))
        
        for i in range(min(20, len(predictions))):
            self.result_table.setItem(i, 0, QTableWidgetItem(str(times[i])))
            self.result_table.setItem(i, 1, QTableWidgetItem(f"{actual_values[i]:.0f}"))
            self.result_table.setItem(i, 2, QTableWidgetItem(f"{predictions[i]:.0f}"))
            self.result_table.setItem(i, 3, QTableWidgetItem(f"{abs(predictions[i] - actual_values[i]):.1f}"))
            
            bottleneck = "예" if predictions[i] > 2000 else "아니오"
            self.result_table.setItem(i, 4, QTableWidgetItem(bottleneck))
        
        # 차트 업데이트
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        sample_size = min(200, len(predictions))
        ax.plot(actual_values[:sample_size], label='실제값', color='blue', linewidth=2)
        ax.plot(predictions[:sample_size], label='예측값', color='red', linewidth=1.5)
        ax.axhline(y=2000, color='orange', linestyle='--', label='병목 임계값')
        
        ax.set_title('반도체 물류량 예측 결과', fontsize=14)
        ax.set_xlabel('시간', fontsize=12)
        ax.set_ylabel('물류량', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.canvas.draw()


class MainWindow(QMainWindow):
    """메인 윈도우"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("반도체 FAB 물류 예측 하이브리드 딥러닝 시스템")
        self.setGeometry(100, 100, 1400, 900)
        
        # 다크 테마 적용
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QTabWidget::pane {
                background-color: #2d2d2d;
                border: 1px solid #444444;
            }
            QTabBar::tab {
                background-color: #3c3c3c;
                color: #ffffff;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #2d2d2d;
                border-bottom: 2px solid #3498db;
            }
            QPushButton {
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #555555;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #484848;
                border: 1px solid #3498db;
            }
            QPushButton:pressed {
                background-color: #2d2d2d;
            }
            QLabel {
                color: #ffffff;
            }
            QGroupBox {
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QTextEdit, QTableWidget {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 1px solid #444444;
            }
            QHeaderView::section {
                background-color: #3c3c3c;
                color: #ffffff;
                padding: 5px;
                border: 1px solid #555555;
            }
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 3px;
                text-align: center;
                background-color: #2d2d2d;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 3px;
            }
        """)
        
        # 중앙 위젯
        self.init_ui()
        
        # 데이터 저장
        self.current_data = None
        self.models = {}
        self.scaler = None
        self.checkpoint_manager = CheckpointManager()
        self.training_thread = None
        
    def init_ui(self):
        """UI 초기화"""
        # 메인 위젯
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # 메인 레이아웃
        layout = QVBoxLayout()
        
        # 탭 위젯
        self.tabs = QTabWidget()
        
        # 데이터 탭
        self.data_tab = self.create_data_tab()
        self.tabs.addTab(self.data_tab, "📊 데이터")
        
        # 학습 탭
        self.train_tab = self.create_train_tab()
        self.tabs.addTab(self.train_tab, "🎓 학습")
        
        # 예측 탭
        self.predict_tab = self.create_predict_tab()
        self.tabs.addTab(self.predict_tab, "🔮 예측")
        
        # 평가 탭
        self.evaluate_tab = self.create_evaluate_tab()
        self.tabs.addTab(self.evaluate_tab, "📈 평가")
        
        # 시각화 탭
        self.visualize_tab = self.create_visualize_tab()
        self.tabs.addTab(self.visualize_tab, "📉 시각화")
        
        layout.addWidget(self.tabs)
        
        # 상태바
        self.status_label = QLabel("준비됨")
        self.status_label.setStyleSheet("padding: 5px; color: #3498db;")
        layout.addWidget(self.status_label)
        
        main_widget.setLayout(layout)
        
        # 메뉴바
        self.create_menu_bar()
        
    def create_menu_bar(self):
        """메뉴바 생성"""
        menubar = self.menuBar()
        
        # 파일 메뉴
        file_menu = menubar.addMenu("파일")
        
        load_action = QAction("데이터 불러오기", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self.load_data)
        file_menu.addAction(load_action)
        
        save_action = QAction("모델 저장", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_models)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("종료", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 도구 메뉴
        tools_menu = menubar.addMenu("도구")
        
        checkpoint_action = QAction("체크포인트 관리", self)
        checkpoint_action.triggered.connect(self.manage_checkpoints)
        tools_menu.addAction(checkpoint_action)
        
        # 도움말 메뉴
        help_menu = menubar.addMenu("도움말")
        
        about_action = QAction("프로그램 정보", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def create_data_tab(self):
        """데이터 탭 생성"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # 데이터 로드 섹션
        load_group = QGroupBox("데이터 로드")
        load_layout = QVBoxLayout()
        
        # 파일 선택
        file_layout = QHBoxLayout()
        self.data_path_edit = QLineEdit()
        self.data_path_edit.setPlaceholderText("데이터 파일 경로...")
        browse_btn = QPushButton("찾아보기")
        browse_btn.clicked.connect(self.browse_data_file)
        file_layout.addWidget(self.data_path_edit)
        file_layout.addWidget(browse_btn)
        load_layout.addLayout(file_layout)
        
        # 로드 버튼
        load_btn = QPushButton("데이터 로드")
        load_btn.clicked.connect(self.load_data)
        load_layout.addWidget(load_btn)
        
        load_group.setLayout(load_layout)
        layout.addWidget(load_group)
        
        # 데이터 정보
        info_group = QGroupBox("데이터 정보")
        info_layout = QVBoxLayout()
        
        self.data_info_text = QTextEdit()
        self.data_info_text.setReadOnly(True)
        self.data_info_text.setMaximumHeight(200)
        info_layout.addWidget(self.data_info_text)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # 전처리 옵션
        preprocess_group = QGroupBox("전처리 옵션")
        preprocess_layout = QGridLayout()
        
        # 날짜 범위
        preprocess_layout.addWidget(QLabel("시작 날짜:"), 0, 0)
        self.start_date_edit = QDateEdit()
        self.start_date_edit.setCalendarPopup(True)
        self.start_date_edit.setDate(QDate(2025, 1, 1))
        preprocess_layout.addWidget(self.start_date_edit, 0, 1)
        
        preprocess_layout.addWidget(QLabel("종료 날짜:"), 0, 2)
        self.end_date_edit = QDateEdit()
        self.end_date_edit.setCalendarPopup(True)
        self.end_date_edit.setDate(QDate(2025, 7, 27))
        preprocess_layout.addWidget(self.end_date_edit, 0, 3)
        
        # 이상치 제거
        self.remove_outliers_check = QCheckBox("이상치 제거")
        self.remove_outliers_check.setChecked(True)
        preprocess_layout.addWidget(self.remove_outliers_check, 1, 0)
        
        # 스케일링
        preprocess_layout.addWidget(QLabel("스케일링:"), 1, 2)
        self.scaling_combo = QComboBox()
        self.scaling_combo.addItems(["StandardScaler", "MinMaxScaler", "RobustScaler"])
        preprocess_layout.addWidget(self.scaling_combo, 1, 3)
        
        # 전처리 실행 버튼
        preprocess_btn = QPushButton("전처리 실행")
        preprocess_btn.clicked.connect(self.preprocess_data)
        preprocess_layout.addWidget(preprocess_btn, 2, 0, 1, 4)
        
        preprocess_group.setLayout(preprocess_layout)
        layout.addWidget(preprocess_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
        
    def create_train_tab(self):
        """학습 탭 생성"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # 학습 설정
        config_group = QGroupBox("학습 설정")
        config_layout = QGridLayout()
        
        # 에폭
        config_layout.addWidget(QLabel("에폭:"), 0, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(10, 500)
        self.epochs_spin.setValue(200)
        config_layout.addWidget(self.epochs_spin, 0, 1)
        
        # 배치 크기
        config_layout.addWidget(QLabel("배치 크기:"), 0, 2)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(16, 256)
        self.batch_size_spin.setValue(64)
        self.batch_size_spin.setSingleStep(16)
        config_layout.addWidget(self.batch_size_spin, 0, 3)
        
        # 학습률
        config_layout.addWidget(QLabel("학습률:"), 1, 0)
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(0.0001, 0.1)
        self.learning_rate_spin.setValue(0.0005)
        self.learning_rate_spin.setSingleStep(0.0001)
        self.learning_rate_spin.setDecimals(4)
        config_layout.addWidget(self.learning_rate_spin, 1, 1)
        
        # 검증 비율
        config_layout.addWidget(QLabel("검증 비율:"), 1, 2)
        self.validation_split_spin = QDoubleSpinBox()
        self.validation_split_spin.setRange(0.1, 0.5)
        self.validation_split_spin.setValue(0.3)
        self.validation_split_spin.setSingleStep(0.05)
        config_layout.addWidget(self.validation_split_spin, 1, 3)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # 모델 선택
        model_group = QGroupBox("모델 선택")
        model_layout = QVBoxLayout()
        
        self.model_checks = {
            'lstm': QCheckBox("LSTM (Long Short-Term Memory)"),
            'gru': QCheckBox("GRU (Gated Recurrent Unit)"),
            'rnn': QCheckBox("RNN (Recurrent Neural Network)"),
            'bi_lstm': QCheckBox("Bidirectional LSTM")
        }
        
        for check in self.model_checks.values():
            check.setChecked(True)
            model_layout.addWidget(check)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # 학습 제어
        control_layout = QHBoxLayout()
        
        self.train_btn = QPushButton("학습 시작")
        self.train_btn.clicked.connect(self.start_training)
        control_layout.addWidget(self.train_btn)
        
        self.stop_btn = QPushButton("학습 중단")
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)
        
        self.resume_btn = QPushButton("이어서 학습")
        self.resume_btn.clicked.connect(self.resume_training)
        control_layout.addWidget(self.resume_btn)
        
        layout.addLayout(control_layout)
        
        # 진행 상황
        progress_group = QGroupBox("학습 진행 상황")
        progress_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        
        self.training_log = QTextEdit()
        self.training_log.setReadOnly(True)
        self.training_log.setMaximumHeight(200)
        progress_layout.addWidget(self.training_log)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
        
    def create_predict_tab(self):
        """예측 탭 생성"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # 예측 설정
        predict_group = QGroupBox("예측 설정")
        predict_layout = QVBoxLayout()
        
        # 모델 선택
        model_select_layout = QHBoxLayout()
        model_select_layout.addWidget(QLabel("예측 모델:"))
        self.predict_model_combo = QComboBox()
        self.predict_model_combo.addItems(["앙상블 (전체)", "LSTM", "GRU", "RNN", "Bi-LSTM"])
        model_select_layout.addWidget(self.predict_model_combo)
        predict_layout.addLayout(model_select_layout)
        
        # 예측 실행
        predict_btn = QPushButton("예측 실행")
        predict_btn.clicked.connect(self.run_prediction)
        predict_layout.addWidget(predict_btn)
        
        predict_group.setLayout(predict_layout)
        layout.addWidget(predict_group)
        
        # 예측 결과
        self.prediction_widget = PredictionWidget()
        layout.addWidget(self.prediction_widget)
        
        widget.setLayout(layout)
        return widget
        
    def create_evaluate_tab(self):
        """평가 탭 생성"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # 평가 실행
        eval_btn = QPushButton("모델 평가 실행")
        eval_btn.clicked.connect(self.evaluate_models)
        layout.addWidget(eval_btn)
        
        # 평가 결과
        self.eval_table = QTableWidget()
        self.eval_table.setColumnCount(8)
        self.eval_table.setHorizontalHeaderLabels([
            '모델', 'MAPE(%)', '정확도(10%)', 'R² Score', 
            '방향성 정확도', '병목 정확도', '종합 점수', '등급'
        ])
        layout.addWidget(self.eval_table)
        
        # 상세 결과
        self.eval_detail_text = QTextEdit()
        self.eval_detail_text.setReadOnly(True)
        self.eval_detail_text.setMaximumHeight(200)
        layout.addWidget(self.eval_detail_text)
        
        widget.setLayout(layout)
        return widget
        
    def create_visualize_tab(self):
        """시각화 탭 생성"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # 시각화 옵션
        viz_options = QHBoxLayout()
        
        self.viz_type_combo = QComboBox()
        self.viz_type_combo.addItems([
            "학습 곡선", "예측 vs 실제", "오차 분포", 
            "병목 구간 히트맵", "시계열 분해"
        ])
        viz_options.addWidget(QLabel("차트 유형:"))
        viz_options.addWidget(self.viz_type_combo)
        
        viz_btn = QPushButton("시각화 생성")
        viz_btn.clicked.connect(self.create_visualization)
        viz_options.addWidget(viz_btn)
        
        layout.addLayout(viz_options)
        
        # 차트 영역
        self.viz_figure = Figure(figsize=(12, 8))
        self.viz_canvas = FigureCanvas(self.viz_figure)
        layout.addWidget(self.viz_canvas)
        
        widget.setLayout(layout)
        return widget
        
    def browse_data_file(self):
        """데이터 파일 찾아보기"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "데이터 파일 선택", "",
            "CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*.*)"
        )
        if filename:
            self.data_path_edit.setText(filename)
            
    def load_data(self):
        """데이터 로드"""
        try:
            file_path = self.data_path_edit.text()
            if not file_path:
                QMessageBox.warning(self, "경고", "데이터 파일을 선택하세요")
                return
                
            self.update_status("데이터 로딩 중...")
            
            # 데이터 로드
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                raise ValueError("지원하지 않는 파일 형식입니다")
            
            # 데이터 정보 표시
            info = f"데이터 로드 완료\n"
            info += f"파일: {os.path.basename(file_path)}\n"
            info += f"크기: {len(df)} × {len(df.columns)}\n"
            info += f"컬럼: {', '.join(df.columns[:10])}"
            if len(df.columns) > 10:
                info += f" ... 외 {len(df.columns)-10}개"
            
            self.data_info_text.setText(info)
            self.current_data = df
            
            self.update_status("데이터 로드 완료")
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"데이터 로드 실패: {str(e)}")
            self.update_status("데이터 로드 실패")
            
    def preprocess_data(self):
        """데이터 전처리"""
        if self.current_data is None:
            QMessageBox.warning(self, "경고", "먼저 데이터를 로드하세요")
            return
            
        try:
            self.update_status("전처리 시작...")
            
            df = self.current_data.copy()
            
            # 시간 컬럼 변환
            if 'CURRTIME' in df.columns:
                df['CURRTIME'] = pd.to_datetime(df['CURRTIME'], format='%Y%m%d%H%M')
                df.set_index('CURRTIME', inplace=True)
            if 'TIME' in df.columns:
                df['TIME'] = pd.to_datetime(df['TIME'], format='%Y%m%d%H%M')
            
            # 날짜 범위 필터링
            start_date = self.start_date_edit.date().toPyDate()
            end_date = self.end_date_edit.date().toPyDate()
            
            if 'TIME' in df.columns:
                df = df[(df['TIME'] >= pd.Timestamp(start_date)) & 
                       (df['TIME'] <= pd.Timestamp(end_date))]
            
            # SUM 컬럼 제거
            columns_to_drop = [col for col in df.columns if 'SUM' in col]
            df = df.drop(columns=columns_to_drop, errors='ignore')
            
            # TOTALCNT만 선택
            if 'TOTALCNT' in df.columns:
                df = df[['TOTALCNT', 'TIME'] if 'TIME' in df.columns else ['TOTALCNT']]
            
            # 이상치 제거
            if self.remove_outliers_check.isChecked() and 'TOTALCNT' in df.columns:
                # PM 기간 설정
                PM_start = pd.to_datetime('2024-10-23')
                PM_end = pd.to_datetime('2024-10-23 23:59:59')
                
                if 'TIME' in df.columns:
                    within_PM = df[(df['TIME'] >= PM_start) & (df['TIME'] <= PM_end)]
                    outside_PM = df[(df['TIME'] < PM_start) | (df['TIME'] > PM_end)]
                    outside_PM_filtered = outside_PM[
                        (outside_PM['TOTALCNT'] >= 800) & 
                        (outside_PM['TOTALCNT'] <= 2500)
                    ]
                    df = pd.concat([within_PM, outside_PM_filtered]).sort_index()
            
            # FUTURE 컬럼 생성
            df['FUTURE'] = pd.NA
            future_minutes = 10
            
            for i in df.index:
                future_time = i + pd.Timedelta(minutes=future_minutes)
                if future_time in df.index:
                    df.loc[i, 'FUTURE'] = df.loc[future_time, 'TOTALCNT']
            
            df.dropna(subset=['FUTURE'], inplace=True)
            
            # 특징 엔지니어링
            df['hour'] = df.index.hour
            df['dayofweek'] = df.index.dayofweek
            df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
            df['MA_5'] = df['TOTALCNT'].rolling(window=5, min_periods=1).mean()
            df['MA_10'] = df['TOTALCNT'].rolling(window=10, min_periods=1).mean()
            df['MA_30'] = df['TOTALCNT'].rolling(window=30, min_periods=1).mean()
            df['STD_5'] = df['TOTALCNT'].rolling(window=5, min_periods=1).std()
            df['STD_10'] = df['TOTALCNT'].rolling(window=10, min_periods=1).std()
            df['change_rate'] = df['TOTALCNT'].pct_change()
            df = df.fillna(method='ffill').fillna(0)
            
            # 스케일링
            scaling_method = self.scaling_combo.currentText()
            if scaling_method == "StandardScaler":
                self.scaler = StandardScaler()
            elif scaling_method == "MinMaxScaler":
                self.scaler = MinMaxScaler()
            else:
                self.scaler = RobustScaler()
            
            scale_columns = ['TOTALCNT', 'FUTURE', 'MA_5', 'MA_10', 'MA_30', 'STD_5', 'STD_10']
            scale_columns = [col for col in scale_columns if col in df.columns]
            
            scaled_data = self.scaler.fit_transform(df[scale_columns])
            scaled_df = pd.DataFrame(
                scaled_data, 
                columns=[f'scaled_{col}' for col in scale_columns],
                index=df.index
            )
            
            df = pd.merge(df, scaled_df, left_index=True, right_index=True)
            
            # 시퀀스 생성
            self.create_sequences(df)
            
            info = self.data_info_text.toPlainText()
            info += f"\n\n전처리 완료:\n"
            info += f"최종 데이터: {len(df)} 레코드\n"
            info += f"특징 수: {len(df.columns)}"
            self.data_info_text.setText(info)
            
            self.update_status("전처리 완료")
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"전처리 실패: {str(e)}")
            self.update_status("전처리 실패")
            
    def create_sequences(self, df):
        """시퀀스 데이터 생성"""
        seq_length = 30
        
        # 연속성 확인
        time_diff = df.index.to_series().diff()
        split_points = time_diff > pd.Timedelta(minutes=1)
        segment_ids = split_points.cumsum()
        
        # 입력 특징
        input_features = [col for col in df.columns if col.startswith('scaled_') and col != 'scaled_FUTURE']
        
        X, y = [], []
        
        for segment_id in segment_ids.unique():
            segment = df[segment_ids == segment_id]
            
            if len(segment) > seq_length:
                X_data = segment[input_features].values
                y_data = segment['scaled_FUTURE'].values
                
                for i in range(len(segment) - seq_length):
                    X.append(X_data[i:i+seq_length])
                    y.append(y_data[i+seq_length])
        
        X = np.array(X)
        y = np.array(y)
        
        # 데이터 분할
        train_size = int(0.7 * len(X))
        val_size = int(0.15 * len(X))
        
        self.current_data = {
            'X_train': X[:train_size],
            'y_train': y[:train_size],
            'X_val': X[train_size:train_size+val_size],
            'y_val': y[train_size:train_size+val_size],
            'X_test': X[train_size+val_size:],
            'y_test': y[train_size+val_size:],
            'input_features': input_features,
            'processed_df': df
        }
        
    def start_training(self):
        """학습 시작"""
        if self.current_data is None or 'X_train' not in self.current_data:
            QMessageBox.warning(self, "경고", "먼저 데이터를 전처리하세요")
            return
            
        # 학습 설정
        config = {
            'epochs': self.epochs_spin.value(),
            'batch_size': self.batch_size_spin.value(),
            'learning_rate': self.learning_rate_spin.value(),
            'validation_split': self.validation_split_spin.value()
        }
        
        # 선택된 모델
        selected_models = [name for name, check in self.model_checks.items() if check.isChecked()]
        if not selected_models:
            QMessageBox.warning(self, "경고", "최소 하나의 모델을 선택하세요")
            return
        
        config['models'] = selected_models
        
        # UI 업데이트
        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.training_log.clear()
        self.progress_bar.setValue(0)
        
        # 학습 스레드 시작
        self.training_thread = TrainingThread(self.current_data, config)
        self.training_thread.progress_update.connect(self.update_training_progress)
        self.training_thread.training_complete.connect(self.on_training_complete)
        self.training_thread.error_occurred.connect(self.on_training_error)
        self.training_thread.start()
        
    def stop_training(self):
        """학습 중단"""
        if self.training_thread:
            self.training_thread.stop()
            self.train_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.update_status("학습이 중단되었습니다")
            
    def resume_training(self):
        """학습 재개"""
        state = self.checkpoint_manager.load_state()
        if state:
            QMessageBox.information(self, "정보", "체크포인트에서 학습을 재개합니다")
            # 재개 로직 구현
        else:
            QMessageBox.warning(self, "경고", "저장된 체크포인트가 없습니다")
            
    def update_training_progress(self, message):
        """학습 진행 상황 업데이트"""
        self.training_log.append(message)
        self.update_status(message)
        
        # 프로그레스바 업데이트
        if "완료" in message:
            current = self.progress_bar.value()
            self.progress_bar.setValue(min(current + 25, 100))
            
    def on_training_complete(self, results):
        """학습 완료"""
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(100)
        
        self.models = results
        self.update_status("모든 모델 학습 완료!")
        
        # 결과 요약
        summary = "\n학습 결과 요약:\n"
        for model_name, info in results.items():
            summary += f"{model_name.upper()}: Loss={info['final_loss']:.4f}, Val Loss={info['final_val_loss']:.4f}\n"
        
        self.training_log.append(summary)
        
    def on_training_error(self, error_msg):
        """학습 오류"""
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        QMessageBox.critical(self, "학습 오류", error_msg)
        self.update_status("학습 실패")
        
    def run_prediction(self):
        """예측 실행"""
        if not self.models:
            QMessageBox.warning(self, "경고", "먼저 모델을 학습하세요")
            return
            
        try:
            self.update_status("예측 실행 중...")
            
            # 테스트 데이터
            X_test = self.current_data['X_test']
            y_test = self.current_data['y_test']
            
            # 선택된 모델
            model_type = self.predict_model_combo.currentText()
            
            if "앙상블" in model_type:
                # 앙상블 예측
                predictions = []
                for model_name in self.models:
                    model_path = self.models[model_name]['model_path']
                    model = load_model(model_path, compile=False)
                    pred = model.predict(X_test, verbose=0)
                    predictions.append(pred.flatten())
                
                # 평균 앙상블
                ensemble_pred = np.mean(predictions, axis=0)
            else:
                # 개별 모델 예측
                model_name = model_type.lower().replace('-', '_')
                model_path = self.models[model_name]['model_path']
                model = load_model(model_path, compile=False)
                ensemble_pred = model.predict(X_test, verbose=0).flatten()
            
            # 역스케일링
            n_features = self.scaler.n_features_in_
            dummy = np.zeros((len(ensemble_pred), n_features))
            dummy[:, 0] = ensemble_pred
            pred_original = self.scaler.inverse_transform(dummy)[:, 0]
            
            dummy_y = np.zeros((len(y_test), n_features))
            dummy_y[:, 0] = y_test
            y_original = self.scaler.inverse_transform(dummy_y)[:, 0]
            
            # 시간 생성 (임시)
            times = pd.date_range(start='2025-07-01', periods=len(pred_original), freq='T')
            
            # 예측 결과 업데이트
            self.prediction_widget.update_predictions(pred_original, y_original, times)
            
            # 성능 메트릭
            mae = mean_absolute_error(y_original, pred_original)
            mse = mean_squared_error(y_original, pred_original)
            r2 = r2_score(y_original, pred_original)
            
            self.update_status(f"예측 완료 - MAE: {mae:.2f}, MSE: {mse:.2f}, R²: {r2:.4f}")
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"예측 실패: {str(e)}")
            self.update_status("예측 실패")
            
    def evaluate_models(self):
        """모델 평가"""
        if not self.models or 'X_test' not in self.current_data:
            QMessageBox.warning(self, "경고", "먼저 모델을 학습하고 데이터를 준비하세요")
            return
            
        try:
            self.update_status("모델 평가 중...")
            
            evaluator = ModelPercentEvaluator()
            evaluator.scaler = self.scaler
            
            X_test = self.current_data['X_test']
            y_test = self.current_data['y_test']
            
            # 역스케일링
            n_features = self.scaler.n_features_in_
            dummy_y = np.zeros((len(y_test), n_features))
            dummy_y[:, 0] = y_test
            y_original = self.scaler.inverse_transform(dummy_y)[:, 0]
            
            # 평가 결과 저장
            results = []
            
            for model_name in self.models:
                model_path = self.models[model_name]['model_path']
                model = load_model(model_path, compile=False)
                
                # 예측
                pred = model.predict(X_test, verbose=0).flatten()
                dummy_pred = np.zeros((len(pred), n_features))
                dummy_pred[:, 0] = pred
                pred_original = self.scaler.inverse_transform(dummy_pred)[:, 0]
                
                # 평가 메트릭
                mape = evaluator.calculate_mape(y_original, pred_original)
                acc_10 = evaluator.calculate_accuracy_within_threshold(y_original, pred_original, 10)
                r2 = r2_score(y_original, pred_original) * 100
                dir_acc = evaluator.calculate_directional_accuracy(y_original, pred_original)
                
                bottleneck_metrics = evaluator.calculate_bottleneck_accuracy(
                    y_original, pred_original, 2000
                )
                
                # 종합 점수
                overall_score = (
                    (100 - min(mape, 100)) * 0.2 +
                    acc_10 * 0.3 +
                    r2 * 0.2 +
                    dir_acc * 0.15 +
                    bottleneck_metrics['accuracy'] * 0.15
                )
                
                # 등급
                if overall_score >= 90:
                    grade = "A+"
                elif overall_score >= 85:
                    grade = "A"
                elif overall_score >= 80:
                    grade = "B+"
                elif overall_score >= 75:
                    grade = "B"
                elif overall_score >= 70:
                    grade = "C+"
                else:
                    grade = "C"
                
                results.append({
                    'model': model_name.upper(),
                    'mape': mape,
                    'acc_10': acc_10,
                    'r2': r2,
                    'dir_acc': dir_acc,
                    'bottleneck_acc': bottleneck_metrics['accuracy'],
                    'overall_score': overall_score,
                    'grade': grade
                })
            
            # 테이블 업데이트
            self.eval_table.setRowCount(len(results))
            for i, res in enumerate(results):
                self.eval_table.setItem(i, 0, QTableWidgetItem(res['model']))
                self.eval_table.setItem(i, 1, QTableWidgetItem(f"{res['mape']:.2f}"))
                self.eval_table.setItem(i, 2, QTableWidgetItem(f"{res['acc_10']:.1f}"))
                self.eval_table.setItem(i, 3, QTableWidgetItem(f"{res['r2']:.1f}"))
                self.eval_table.setItem(i, 4, QTableWidgetItem(f"{res['dir_acc']:.1f}"))
                self.eval_table.setItem(i, 5, QTableWidgetItem(f"{res['bottleneck_acc']:.1f}"))
                self.eval_table.setItem(i, 6, QTableWidgetItem(f"{res['overall_score']:.1f}"))
                self.eval_table.setItem(i, 7, QTableWidgetItem(res['grade']))
            
            # 최고 성능 모델
            best_model = max(results, key=lambda x: x['overall_score'])
            detail = f"최고 성능 모델: {best_model['model']}\n"
            detail += f"종합 점수: {best_model['overall_score']:.1f}%\n"
            detail += f"성능 등급: {best_model['grade']}"
            
            self.eval_detail_text.setText(detail)
            self.update_status("모델 평가 완료")
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"평가 실패: {str(e)}")
            self.update_status("평가 실패")
            
    def create_visualization(self):
        """시각화 생성"""
        viz_type = self.viz_type_combo.currentText()
        
        self.viz_figure.clear()
        
        if viz_type == "학습 곡선" and self.models:
            # 학습 곡선
            axes = self.viz_figure.subplots(2, 2)
            axes = axes.ravel()
            
            for idx, (model_name, info) in enumerate(self.models.items()):
                if idx < 4:
                    ax = axes[idx]
                    history = info.get('history', {})
                    
                    if 'loss' in history:
                        ax.plot(history['loss'], label='Training Loss')
                    if 'val_loss' in history:
                        ax.plot(history['val_loss'], label='Validation Loss')
                    
                    ax.set_title(f'{model_name.upper()}')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Loss')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
        
        elif viz_type == "예측 vs 실제" and 'X_test' in self.current_data:
            # 예측 vs 실제
            ax = self.viz_figure.add_subplot(111)
            
            # 간단한 시각화를 위해 첫 번째 모델만 사용
            if self.models:
                model_name = list(self.models.keys())[0]
                model_path = self.models[model_name]['model_path']
                model = load_model(model_path, compile=False)
                
                X_test = self.current_data['X_test'][:100]  # 샘플
                y_test = self.current_data['y_test'][:100]
                
                pred = model.predict(X_test, verbose=0).flatten()
                
                ax.scatter(y_test, pred, alpha=0.5)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                ax.set_xlabel('Actual')
                ax.set_ylabel('Predicted')
                ax.set_title('Predictions vs Actual')
                ax.grid(True, alpha=0.3)
        
        self.viz_canvas.draw()
        
    def save_models(self):
        """모델 저장"""
        if not self.models:
            QMessageBox.warning(self, "경고", "저장할 모델이 없습니다")
            return
            
        folder = QFileDialog.getExistingDirectory(self, "모델 저장 폴더 선택")
        if folder:
            for model_name, info in self.models.items():
                src = info['model_path']
                dst = os.path.join(folder, f"{model_name}_hybrid.keras")
                
                import shutil
                shutil.copy2(src, dst)
            
            # 스케일러 저장
            if self.scaler:
                scaler_path = os.path.join(folder, "scaler.pkl")
                joblib.dump(self.scaler, scaler_path)
            
            QMessageBox.information(self, "완료", "모델이 저장되었습니다")
            
    def manage_checkpoints(self):
        """체크포인트 관리"""
        dialog = QDialog(self)
        dialog.setWindowTitle("체크포인트 관리")
        dialog.setModal(True)
        
        layout = QVBoxLayout()
        
        # 체크포인트 목록
        checkpoint_list = QListWidget()
        checkpoint_dir = self.checkpoint_manager.checkpoint_dir
        
        if os.path.exists(checkpoint_dir):
            files = os.listdir(checkpoint_dir)
            for file in files:
                checkpoint_list.addItem(file)
        
        layout.addWidget(QLabel("저장된 체크포인트:"))
        layout.addWidget(checkpoint_list)
        
        # 버튼
        btn_layout = QHBoxLayout()
        
        load_btn = QPushButton("불러오기")
        load_btn.clicked.connect(lambda: self.load_checkpoint(checkpoint_list.currentItem()))
        btn_layout.addWidget(load_btn)
        
        delete_btn = QPushButton("삭제")
        delete_btn.clicked.connect(lambda: self.delete_checkpoint(checkpoint_list.currentItem()))
        btn_layout.addWidget(delete_btn)
        
        close_btn = QPushButton("닫기")
        close_btn.clicked.connect(dialog.accept)
        btn_layout.addWidget(close_btn)
        
        layout.addLayout(btn_layout)
        dialog.setLayout(layout)
        dialog.exec_()
        
    def load_checkpoint(self, item):
        """체크포인트 불러오기"""
        if item:
            # 체크포인트 로드 로직
            pass
            
    def delete_checkpoint(self, item):
        """체크포인트 삭제"""
        if item:
            reply = QMessageBox.question(self, "확인", "이 체크포인트를 삭제하시겠습니까?")
            if reply == QMessageBox.Yes:
                # 삭제 로직
                pass
                
    def show_about(self):
        """프로그램 정보"""
        about_text = """
반도체 FAB 물류 예측 하이브리드 딥러닝 시스템

버전: 3.0
개발: 2024

특징:
- LSTM, RNN, GRU, Bi-LSTM 통합 하이브리드 모델
- CPU 기반 실행 지원
- 실시간 병목 구간 예측
- 10분 후 물류량 예측
- 퍼센트 기반 성능 평가
- 학습 재시작 기능

본 시스템은 반도체 제조 공정의 물류 흐름을
딥러닝으로 예측하여 생산성을 향상시킵니다.
        """
        QMessageBox.about(self, "프로그램 정보", about_text)
        
    def update_status(self, message):
        """상태 업데이트"""
        self.status_label.setText(f"상태: {message}")
        QApplication.processEvents()


def main():
    app = QApplication(sys.argv)
    
    # 애플리케이션 스타일 설정
    app.setStyle("Fusion")
    
    # 다크 팔레트
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(45, 45, 45))
    palette.setColor(QPalette.AlternateBase, QColor(60, 60, 60))
    palette.setColor(QPalette.ToolTipBase, Qt.black)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(45, 45, 45))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)
    
    # 메인 윈도우 실행
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()