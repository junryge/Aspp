"""
반도체 FAB 물류 예측을 위한 통합 하이브리드 딥러닝 시스템
================================================================
LSTM, RNN, GRU, ARIMA 모델을 통합한 하이브리드 예측 시스템으로
반도체 팹 간 물류 이동량과 병목 구간을 예측합니다.

주요 기능:
1. 기존 학습된 모델 불러오기 및 선택
2. 실시간 모델 학습 (재시작 가능)
3. 앙상블 예측 및 병목 구간 감지
4. 퍼센트 기반 성능 평가
5. CPU 기반 실행
6. 한글 폰트 지원

버전: 3.2 (한글 폰트 수정)
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
import glob
import platform

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

# Matplotlib 한글 폰트 설정
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns

# PyQt5
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


def setup_korean_font():
    """운영체제별 한글 폰트 자동 설정"""
    system = platform.system()
    
    # 폰트 경로 설정
    if system == 'Windows':
        font_paths = [
            'C:/Windows/Fonts/malgun.ttf',      # 맑은 고딕
            'C:/Windows/Fonts/malgunbd.ttf',    # 맑은 고딕 Bold
            'C:/Windows/Fonts/ngulim.ttf',      # 나눔고딕
            'C:/Windows/Fonts/NanumGothic.ttf',
            'C:/Windows/Fonts/gulim.ttc',       # 굴림
        ]
        font_family = 'Malgun Gothic'
        
    elif system == 'Darwin':  # macOS
        font_paths = [
            '/System/Library/Fonts/Supplemental/AppleGothic.ttf',
            '/Library/Fonts/AppleGothic.ttf',
            '/System/Library/Fonts/AppleSDGothicNeo.ttc',
            '/Library/Fonts/NanumGothic.ttf',
            '/Users/*/Library/Fonts/NanumGothic.ttf',
        ]
        font_family = 'AppleGothic'
        
    else:  # Linux
        font_paths = [
            '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
            '/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf',
            '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        ]
        font_family = 'NanumGothic'
    
    # 사용 가능한 폰트 찾기
    font_set = False
    for font_path in font_paths:
        # 와일드카드 처리
        if '*' in font_path:
            import glob
            expanded_paths = glob.glob(os.path.expanduser(font_path))
            if expanded_paths:
                font_path = expanded_paths[0]
        
        if os.path.exists(font_path):
            try:
                # 폰트 매니저에 추가
                font_prop = fm.FontProperties(fname=font_path)
                
                # matplotlib 설정
                plt.rcParams['font.family'] = font_prop.get_name()
                plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
                
                font_set = True
                print(f"✓ 한글 폰트 설정 완료: {font_prop.get_name()}")
                break
            except:
                continue
    
    # 폰트를 찾지 못한 경우 시스템 기본 폰트 시도
    if not font_set:
        try:
            # 시스템에 설치된 한글 폰트 검색
            font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
            
            # 한글 폰트 키워드
            korean_fonts = ['Malgun', 'malgun', '맑은', 'Nanum', 'nanum', '나눔', 
                          'Gulim', 'gulim', '굴림', 'Dotum', 'dotum', '돋움',
                          'Gothic', 'gothic', '고딕', 'Batang', 'batang', '바탕']
            
            for font in font_list:
                font_name = os.path.basename(font).lower()
                if any(kf.lower() in font_name for kf in korean_fonts):
                    try:
                        font_prop = fm.FontProperties(fname=font)
                        plt.rcParams['font.family'] = font_prop.get_name()
                        plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
                        font_set = True
                        print(f"✓ 한글 폰트 자동 감지: {font_prop.get_name()}")
                        break
                    except:
                        continue
        except:
            pass
    
    # 최종 폴백: 시스템 기본 폰트 사용
    if not font_set:
        if system == 'Windows':
            plt.rcParams['font.family'] = 'Malgun Gothic'
        elif system == 'Darwin':
            plt.rcParams['font.family'] = 'AppleGothic'
        else:
            # Linux에서 폰트를 찾지 못한 경우
            print("⚠ 한글 폰트를 찾을 수 없습니다. 영문으로 표시됩니다.")
            print("  NanumGothic 폰트 설치를 권장합니다:")
            print("  Ubuntu/Debian: sudo apt-get install fonts-nanum")
            print("  CentOS/RHEL: sudo yum install nhn-nanum-fonts")
            return False
    
    # 마이너스 기호 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False
    
    # 기본 스타일 설정
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    
    return True


# 프로그램 시작 시 한글 폰트 설정 실행
KOREAN_FONT_AVAILABLE = setup_korean_font()


class ModelManager:
    """모델 관리 클래스 - 기존 모델 불러오기 및 관리"""
    
    def __init__(self):
        self.loaded_models = {}
        self.model_info = {}
        self.scaler = None
        self.config = None
        
    def scan_model_directory(self, directory='model'):
        """모델 디렉토리 스캔"""
        models_found = {}
        
        if os.path.exists(directory):
            # .keras 파일 검색
            keras_files = glob.glob(os.path.join(directory, '*.keras'))
            
            for file_path in keras_files:
                filename = os.path.basename(file_path)
                model_name = filename.replace('.keras', '')
                
                # 모델 타입 추론
                if 'lstm' in model_name.lower():
                    model_type = 'LSTM'
                elif 'gru' in model_name.lower():
                    model_type = 'GRU'
                elif 'rnn' in model_name.lower():
                    model_type = 'RNN'
                elif 'bi' in model_name.lower():
                    model_type = 'Bi-LSTM'
                else:
                    model_type = 'Unknown'
                
                # 파일 정보
                file_stat = os.stat(file_path)
                models_found[model_name] = {
                    'path': file_path,
                    'type': model_type,
                    'size': file_stat.st_size / (1024 * 1024),  # MB
                    'modified': datetime.fromtimestamp(file_stat.st_mtime),
                    'loaded': False
                }
        
        return models_found
    
    def load_model(self, model_path):
        """모델 로드"""
        try:
            model = load_model(model_path, compile=False)
            # 재컴파일
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            return model
        except Exception as e:
            raise Exception(f"모델 로드 실패: {str(e)}")
    
    def load_scaler(self, scaler_path=None):
        """스케일러 로드"""
        if scaler_path and os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            return True
        
        # 기본 경로에서 찾기
        default_paths = [
            'scaler/standard_scaler_hybrid.pkl',
            'scaler/StdScaler_s30f10_0731_2079936.save',
            'scaler/scaler.pkl'
        ]
        
        for path in default_paths:
            if os.path.exists(path):
                self.scaler = joblib.load(path)
                return True
        
        return False
    
    def load_config(self, config_path=None):
        """설정 파일 로드"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            return True
        
        # 기본 경로에서 찾기
        default_path = 'results/training_config.json'
        if os.path.exists(default_path):
            with open(default_path, 'r') as f:
                self.config = json.load(f)
            return True
        
        # 기본 설정
        self.config = {
            'seq_length': 30,
            'future_minutes': 10,
            'bottleneck_threshold': 2000
        }
        return False


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
        
        total = tp + tn + fp + fn
        
        accuracy = (tp + tn) / total * 100 if total > 0 else 0
        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }


class ModelSelectionDialog(QDialog):
    """모델 선택 대화상자"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("모델 선택")
        self.setModal(True)
        self.setMinimumWidth(800)
        self.setMinimumHeight(500)
        
        self.model_manager = ModelManager()
        self.selected_models = []
        self.selected_scaler = None
        
        self.init_ui()
        self.refresh_models()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 모델 디렉토리 선택
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("모델 디렉토리:"))
        self.dir_edit = QLineEdit("model")
        dir_layout.addWidget(self.dir_edit)
        
        browse_btn = QPushButton("찾아보기")
        browse_btn.clicked.connect(self.browse_directory)
        dir_layout.addWidget(browse_btn)
        
        refresh_btn = QPushButton("새로고침")
        refresh_btn.clicked.connect(self.refresh_models)
        dir_layout.addWidget(refresh_btn)
        
        layout.addLayout(dir_layout)
        
        # 모델 목록 테이블
        self.model_table = QTableWidget()
        self.model_table.setColumnCount(6)
        self.model_table.setHorizontalHeaderLabels(['선택', '모델명', '타입', '크기(MB)', '수정일', '경로'])
        self.model_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.model_table)
        
        # 스케일러 선택
        scaler_group = QGroupBox("스케일러 선택")
        scaler_layout = QHBoxLayout()
        
        self.scaler_edit = QLineEdit()
        self.scaler_edit.setPlaceholderText("스케일러 파일 경로...")
        scaler_layout.addWidget(self.scaler_edit)
        
        scaler_browse_btn = QPushButton("찾아보기")
        scaler_browse_btn.clicked.connect(self.browse_scaler)
        scaler_layout.addWidget(scaler_browse_btn)
        
        auto_find_btn = QPushButton("자동 찾기")
        auto_find_btn.clicked.connect(self.auto_find_scaler)
        scaler_layout.addWidget(auto_find_btn)
        
        scaler_group.setLayout(scaler_layout)
        layout.addWidget(scaler_group)
        
        # 버튼
        button_layout = QHBoxLayout()
        
        load_btn = QPushButton("선택한 모델 불러오기")
        load_btn.clicked.connect(self.load_selected_models)
        button_layout.addWidget(load_btn)
        
        cancel_btn = QPushButton("취소")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
    def browse_directory(self):
        """디렉토리 선택"""
        directory = QFileDialog.getExistingDirectory(self, "모델 디렉토리 선택")
        if directory:
            self.dir_edit.setText(directory)
            self.refresh_models()
            
    def refresh_models(self):
        """모델 목록 새로고침"""
        directory = self.dir_edit.text()
        models = self.model_manager.scan_model_directory(directory)
        
        self.model_table.setRowCount(len(models))
        
        for i, (name, info) in enumerate(models.items()):
            # 체크박스
            checkbox = QCheckBox()
            checkbox_widget = QWidget()
            checkbox_layout = QHBoxLayout(checkbox_widget)
            checkbox_layout.addWidget(checkbox)
            checkbox_layout.setAlignment(Qt.AlignCenter)
            checkbox_layout.setContentsMargins(0, 0, 0, 0)
            self.model_table.setCellWidget(i, 0, checkbox_widget)
            
            # 모델 정보
            self.model_table.setItem(i, 1, QTableWidgetItem(name))
            self.model_table.setItem(i, 2, QTableWidgetItem(info['type']))
            self.model_table.setItem(i, 3, QTableWidgetItem(f"{info['size']:.2f}"))
            self.model_table.setItem(i, 4, QTableWidgetItem(info['modified'].strftime('%Y-%m-%d %H:%M')))
            self.model_table.setItem(i, 5, QTableWidgetItem(info['path']))
            
    def browse_scaler(self):
        """스케일러 파일 선택"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "스케일러 파일 선택", "",
            "Pickle Files (*.pkl *.save);;All Files (*.*)"
        )
        if filename:
            self.scaler_edit.setText(filename)
            
    def auto_find_scaler(self):
        """스케일러 자동 찾기"""
        if self.model_manager.load_scaler():
            QMessageBox.information(self, "성공", "스케일러를 자동으로 찾았습니다")
            # 찾은 경로 표시
            if os.path.exists('scaler/standard_scaler_hybrid.pkl'):
                self.scaler_edit.setText('scaler/standard_scaler_hybrid.pkl')
            elif os.path.exists('scaler/StdScaler_s30f10_0731_2079936.save'):
                self.scaler_edit.setText('scaler/StdScaler_s30f10_0731_2079936.save')
        else:
            QMessageBox.warning(self, "실패", "스케일러를 찾을 수 없습니다")
            
    def load_selected_models(self):
        """선택한 모델 불러오기"""
        self.selected_models = []
        
        for i in range(self.model_table.rowCount()):
            checkbox_widget = self.model_table.cellWidget(i, 0)
            checkbox = checkbox_widget.findChild(QCheckBox)
            
            if checkbox.isChecked():
                model_name = self.model_table.item(i, 1).text()
                model_path = self.model_table.item(i, 5).text()
                model_type = self.model_table.item(i, 2).text()
                
                self.selected_models.append({
                    'name': model_name,
                    'path': model_path,
                    'type': model_type
                })
        
        if not self.selected_models:
            QMessageBox.warning(self, "경고", "모델을 선택하세요")
            return
        
        # 스케일러 로드
        scaler_path = self.scaler_edit.text()
        if scaler_path:
            self.model_manager.load_scaler(scaler_path)
        else:
            self.model_manager.load_scaler()
        
        self.selected_scaler = self.model_manager.scaler
        
        if self.selected_scaler is None:
            reply = QMessageBox.question(
                self, "확인",
                "스케일러를 찾을 수 없습니다. 계속하시겠습니까?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
        
        self.accept()


class PredictionWidget(QWidget):
    """예측 결과 표시 위젯"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 예측 결과 테이블
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(6)
        self.result_table.setHorizontalHeaderLabels(['시간', '현재값', '예측값', '오차', '오차율(%)', '병목여부'])
        layout.addWidget(self.result_table)
        
        # 차트
        self.figure = Figure(figsize=(12, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
        
    def update_predictions(self, predictions, actual_values, times, threshold=2000):
        """예측 결과 업데이트"""
        # 테이블 업데이트
        n_rows = min(50, len(predictions))
        self.result_table.setRowCount(n_rows)
        
        for i in range(n_rows):
            # 시간
            if isinstance(times[i], pd.Timestamp):
                time_str = times[i].strftime('%Y-%m-%d %H:%M')
            else:
                time_str = str(times[i])
            self.result_table.setItem(i, 0, QTableWidgetItem(time_str))
            
            # 현재값
            self.result_table.setItem(i, 1, QTableWidgetItem(f"{actual_values[i]:.0f}"))
            
            # 예측값
            self.result_table.setItem(i, 2, QTableWidgetItem(f"{predictions[i]:.0f}"))
            
            # 오차
            error = abs(predictions[i] - actual_values[i])
            self.result_table.setItem(i, 3, QTableWidgetItem(f"{error:.1f}"))
            
            # 오차율
            error_rate = (error / actual_values[i] * 100) if actual_values[i] != 0 else 0
            self.result_table.setItem(i, 4, QTableWidgetItem(f"{error_rate:.1f}%"))
            
            # 병목여부
            bottleneck = "⚠️ 병목" if predictions[i] > threshold else "정상"
            item = QTableWidgetItem(bottleneck)
            if predictions[i] > threshold:
                item.setForeground(QColor('red'))
            self.result_table.setItem(i, 5, item)
        
        # 차트 업데이트
        self.figure.clear()
        
        # 서브플롯 생성
        ax1 = self.figure.add_subplot(211)
        ax2 = self.figure.add_subplot(212)
        
        # 상단: 예측 vs 실제
        sample_size = min(200, len(predictions))
        x_range = range(sample_size)
        
        ax1.plot(x_range, actual_values[:sample_size], label='실제값', color='blue', linewidth=2, alpha=0.7)
        ax1.plot(x_range, predictions[:sample_size], label='예측값', color='red', linewidth=1.5, alpha=0.7)
        ax1.axhline(y=threshold, color='orange', linestyle='--', label=f'병목 임계값 ({threshold})', alpha=0.5)
        
        # 병목 구간 하이라이트
        bottleneck_indices = np.where(predictions[:sample_size] > threshold)[0]
        if len(bottleneck_indices) > 0:
            ax1.scatter(bottleneck_indices, predictions[bottleneck_indices], 
                       color='red', s=30, marker='o', label='병목 예측', zorder=5)
        
        ax1.set_title('반도체 물류량 예측 결과 (10분 후)', fontsize=14)
        ax1.set_xlabel('시간 인덱스', fontsize=12)
        ax1.set_ylabel('물류량 (TOTALCNT)', fontsize=12)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 하단: 오차 분포
        errors = np.abs(predictions[:sample_size] - actual_values[:sample_size])
        ax2.bar(x_range, errors, color='green', alpha=0.5)
        ax2.axhline(y=np.mean(errors), color='red', linestyle='--', 
                   label=f'평균 오차: {np.mean(errors):.1f}')
        
        ax2.set_title('예측 오차 분포', fontsize=14)
        ax2.set_xlabel('시간 인덱스', fontsize=12)
        ax2.set_ylabel('절대 오차', fontsize=12)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        self.figure.tight_layout()
        self.canvas.draw()


class MainWindow(QMainWindow):
    """메인 윈도우"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("반도체 FAB 물류 예측 하이브리드 딥러닝 시스템 v3.2")
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
        
        # 한글 폰트 상태 표시
        if not KOREAN_FONT_AVAILABLE:
            QMessageBox.warning(self, "폰트 경고", 
                              "한글 폰트를 찾을 수 없습니다.\n"
                              "차트에서 한글이 깨질 수 있습니다.\n"
                              "NanumGothic 폰트 설치를 권장합니다.")
        
        # 중앙 위젯
        self.init_ui()
        
        # 데이터 및 모델 저장
        self.current_data = None
        self.loaded_models = {}  # 불러온 모델들
        self.trained_models = {}  # 학습한 모델들
        self.scaler = None
        self.model_manager = ModelManager()
        
    def init_ui(self):
        """UI 초기화"""
        # 메인 위젯
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # 메인 레이아웃
        layout = QVBoxLayout()
        
        # 탭 위젯
        self.tabs = QTabWidget()
        
        # 모델 관리 탭 (새로 추가)
        self.model_tab = self.create_model_management_tab()
        self.tabs.addTab(self.model_tab, "🗂️ 모델 관리")
        
        # 데이터 탭
        self.data_tab = self.create_data_tab()
        self.tabs.addTab(self.data_tab, "📊 데이터")
        
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
        
        load_models_action = QAction("모델 불러오기", self)
        load_models_action.setShortcut("Ctrl+L")
        load_models_action.triggered.connect(self.load_existing_models)
        file_menu.addAction(load_models_action)
        
        load_data_action = QAction("데이터 불러오기", self)
        load_data_action.setShortcut("Ctrl+O")
        load_data_action.triggered.connect(self.load_data)
        file_menu.addAction(load_data_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("종료", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 도움말 메뉴
        help_menu = menubar.addMenu("도움말")
        
        about_action = QAction("프로그램 정보", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
        font_info_action = QAction("폰트 정보", self)
        font_info_action.triggered.connect(self.show_font_info)
        help_menu.addAction(font_info_action)
        
    def create_model_management_tab(self):
        """모델 관리 탭 생성"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # 모델 불러오기 섹션
        load_group = QGroupBox("기존 모델 불러오기")
        load_layout = QVBoxLayout()
        
        load_btn = QPushButton("📁 모델 선택 및 불러오기")
        load_btn.clicked.connect(self.load_existing_models)
        load_layout.addWidget(load_btn)
        
        load_group.setLayout(load_layout)
        layout.addWidget(load_group)
        
        # 불러온 모델 목록
        models_group = QGroupBox("불러온 모델 목록")
        models_layout = QVBoxLayout()
        
        self.loaded_models_table = QTableWidget()
        self.loaded_models_table.setColumnCount(5)
        self.loaded_models_table.setHorizontalHeaderLabels(['모델명', '타입', '상태', '경로', '작업'])
        self.loaded_models_table.horizontalHeader().setStretchLastSection(True)
        models_layout.addWidget(self.loaded_models_table)
        
        models_group.setLayout(models_layout)
        layout.addWidget(models_group)
        
        # 모델 정보
        info_group = QGroupBox("모델 정보")
        info_layout = QVBoxLayout()
        
        self.model_info_text = QTextEdit()
        self.model_info_text.setReadOnly(True)
        self.model_info_text.setMaximumHeight(150)
        info_layout.addWidget(self.model_info_text)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
        
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
        
        # 전처리 실행 버튼
        preprocess_btn = QPushButton("전처리 실행")
        preprocess_btn.clicked.connect(self.preprocess_data)
        preprocess_layout.addWidget(preprocess_btn, 1, 0, 1, 4)
        
        preprocess_group.setLayout(preprocess_layout)
        layout.addWidget(preprocess_group)
        
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
        self.predict_model_combo.addItems(["앙상블 (전체)", "개별 모델 선택"])
        model_select_layout.addWidget(self.predict_model_combo)
        predict_layout.addLayout(model_select_layout)
        
        # 개별 모델 선택 리스트
        self.model_list_widget = QListWidget()
        self.model_list_widget.setSelectionMode(QAbstractItemView.MultiSelection)
        self.model_list_widget.setMaximumHeight(100)
        predict_layout.addWidget(self.model_list_widget)
        
        # 예측 실행
        predict_btn = QPushButton("🚀 예측 실행")
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
        eval_btn = QPushButton("📊 모델 평가 실행")
        eval_btn.clicked.connect(self.evaluate_models)
        layout.addWidget(eval_btn)
        
        # 평가 결과
        self.eval_table = QTableWidget()
        self.eval_table.setColumnCount(9)
        self.eval_table.setHorizontalHeaderLabels([
            '모델', 'MAPE(%)', '정확도(10%)', 'R² Score', 
            '방향성 정확도', '병목 정확도', '병목 F1', '종합 점수', '등급'
        ])
        layout.addWidget(self.eval_table)
        
        # 상세 결과
        self.eval_detail_text = QTextEdit()
        self.eval_detail_text.setReadOnly(True)
        self.eval_detail_text.setMaximumHeight(200)
        layout.addWidget(self.eval_detail_text)
        
        # 결과 내보내기
        export_btn = QPushButton("📥 평가 결과 내보내기")
        export_btn.clicked.connect(self.export_evaluation_results)
        layout.addWidget(export_btn)
        
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
            "예측 vs 실제", "오차 분포", "병목 구간 히트맵", 
            "모델별 성능 비교", "시계열 분해"
        ])
        viz_options.addWidget(QLabel("차트 유형:"))
        viz_options.addWidget(self.viz_type_combo)
        
        viz_btn = QPushButton("📊 시각화 생성")
        viz_btn.clicked.connect(self.create_visualization)
        viz_options.addWidget(viz_btn)
        
        save_btn = QPushButton("💾 차트 저장")
        save_btn.clicked.connect(self.save_visualization)
        viz_options.addWidget(save_btn)
        
        layout.addLayout(viz_options)
        
        # 차트 영역
        self.viz_figure = Figure(figsize=(12, 8))
        self.viz_canvas = FigureCanvas(self.viz_figure)
        layout.addWidget(self.viz_canvas)
        
        widget.setLayout(layout)
        return widget
        
    def load_existing_models(self):
        """기존 모델 불러오기"""
        dialog = ModelSelectionDialog(self)
        
        if dialog.exec_() == QDialog.Accepted:
            selected = dialog.selected_models
            
            if selected:
                self.update_status(f"{len(selected)}개 모델 로딩 중...")
                
                # 스케일러 저장
                if dialog.selected_scaler is not None:
                    self.scaler = dialog.selected_scaler
                    self.model_manager.scaler = self.scaler
                
                # 모델 로드
                for model_info in selected:
                    try:
                        model = dialog.model_manager.load_model(model_info['path'])
                        
                        # 모델 저장
                        self.loaded_models[model_info['name']] = {
                            'model': model,
                            'path': model_info['path'],
                            'type': model_info['type'],
                            'loaded_time': datetime.now()
                        }
                        
                    except Exception as e:
                        QMessageBox.warning(self, "경고", f"모델 로드 실패 ({model_info['name']}): {str(e)}")
                
                # UI 업데이트
                self.update_loaded_models_table()
                self.update_model_list()
                
                self.update_status(f"{len(self.loaded_models)}개 모델 로드 완료")
                
                # 모델 정보 표시
                info = f"불러온 모델: {len(self.loaded_models)}개\n"
                info += f"스케일러: {'로드됨' if self.scaler else '없음'}\n\n"
                
                for name, details in self.loaded_models.items():
                    info += f"• {name} ({details['type']})\n"
                
                self.model_info_text.setText(info)
                
    def update_loaded_models_table(self):
        """불러온 모델 테이블 업데이트"""
        self.loaded_models_table.setRowCount(len(self.loaded_models))
        
        for i, (name, details) in enumerate(self.loaded_models.items()):
            self.loaded_models_table.setItem(i, 0, QTableWidgetItem(name))
            self.loaded_models_table.setItem(i, 1, QTableWidgetItem(details['type']))
            self.loaded_models_table.setItem(i, 2, QTableWidgetItem("로드됨"))
            self.loaded_models_table.setItem(i, 3, QTableWidgetItem(details['path']))
            
            # 제거 버튼
            remove_btn = QPushButton("제거")
            remove_btn.clicked.connect(lambda checked, n=name: self.remove_model(n))
            self.loaded_models_table.setCellWidget(i, 4, remove_btn)
            
    def update_model_list(self):
        """예측 탭의 모델 리스트 업데이트"""
        self.model_list_widget.clear()
        
        for name in self.loaded_models.keys():
            self.model_list_widget.addItem(name)
            
    def remove_model(self, model_name):
        """모델 제거"""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            self.update_loaded_models_table()
            self.update_model_list()
            self.update_status(f"{model_name} 모델이 제거되었습니다")
            
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
                # 파일 선택 대화상자
                file_path, _ = QFileDialog.getOpenFileName(
                    self, "데이터 파일 선택", "",
                    "CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*.*)"
                )
                if not file_path:
                    return
                self.data_path_edit.setText(file_path)
            
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
            df = df.ffill().fillna(0)
            
            # 스케일링
            if self.scaler is None:
                self.scaler = StandardScaler()
                
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
        
    def run_prediction(self):
        """예측 실행"""
        if not self.loaded_models:
            QMessageBox.warning(self, "경고", "먼저 모델을 불러오세요")
            return
            
        if self.current_data is None or 'X_test' not in self.current_data:
            QMessageBox.warning(self, "경고", "먼저 데이터를 전처리하세요")
            return
            
        try:
            self.update_status("예측 실행 중...")
            
            # 테스트 데이터
            X_test = self.current_data['X_test']
            y_test = self.current_data['y_test']
            
            # 선택된 모델
            if self.predict_model_combo.currentText() == "앙상블 (전체)":
                # 모든 모델 사용
                selected_models = list(self.loaded_models.keys())
            else:
                # 선택된 모델만 사용
                selected_items = self.model_list_widget.selectedItems()
                selected_models = [item.text() for item in selected_items]
                
                if not selected_models:
                    QMessageBox.warning(self, "경고", "예측할 모델을 선택하세요")
                    return
            
            # 예측 수행
            predictions = []
            
            for model_name in selected_models:
                if model_name in self.loaded_models:
                    model = self.loaded_models[model_name]['model']
                    pred = model.predict(X_test, verbose=0)
                    predictions.append(pred.flatten())
                    self.update_status(f"{model_name} 예측 완료")
            
            # 앙상블 (평균)
            if len(predictions) > 1:
                ensemble_pred = np.mean(predictions, axis=0)
                self.update_status(f"앙상블 예측 완료 ({len(predictions)}개 모델)")
            else:
                ensemble_pred = predictions[0]
            
            # 역스케일링
            if self.scaler:
                n_features = self.scaler.n_features_in_
                dummy = np.zeros((len(ensemble_pred), n_features))
                dummy[:, 0] = ensemble_pred
                pred_original = self.scaler.inverse_transform(dummy)[:, 0]
                
                dummy_y = np.zeros((len(y_test), n_features))
                dummy_y[:, 0] = y_test
                y_original = self.scaler.inverse_transform(dummy_y)[:, 0]
            else:
                pred_original = ensemble_pred
                y_original = y_test
            
            # 시간 생성
            if 'processed_df' in self.current_data:
                df = self.current_data['processed_df']
                times = df.index[-len(pred_original):]
            else:
                times = pd.date_range(start='2025-07-01', periods=len(pred_original), freq='T')
            
            # 예측 결과 업데이트
            self.prediction_widget.update_predictions(pred_original, y_original, times)
            
            # 성능 메트릭
            mae = mean_absolute_error(y_original, pred_original)
            mse = mean_squared_error(y_original, pred_original)
            r2 = r2_score(y_original, pred_original)
            
            # 병목 예측 정확도
            bottleneck_threshold = 2000
            bottleneck_accuracy = np.mean((pred_original > bottleneck_threshold) == (y_original > bottleneck_threshold))
            
            self.update_status(f"예측 완료 - MAE: {mae:.2f}, R²: {r2:.4f}, 병목정확도: {bottleneck_accuracy:.1%}")
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"예측 실패: {str(e)}")
            self.update_status("예측 실패")
            
    def evaluate_models(self):
        """모델 평가"""
        if not self.loaded_models:
            QMessageBox.warning(self, "경고", "먼저 모델을 불러오세요")
            return
            
        if self.current_data is None or 'X_test' not in self.current_data:
            QMessageBox.warning(self, "경고", "먼저 데이터를 전처리하세요")
            return
            
        try:
            self.update_status("모델 평가 중...")
            
            evaluator = ModelPercentEvaluator()
            evaluator.scaler = self.scaler
            
            X_test = self.current_data['X_test']
            y_test = self.current_data['y_test']
            
            # 역스케일링
            if self.scaler:
                n_features = self.scaler.n_features_in_
                dummy_y = np.zeros((len(y_test), n_features))
                dummy_y[:, 0] = y_test
                y_original = self.scaler.inverse_transform(dummy_y)[:, 0]
            else:
                y_original = y_test
            
            # 평가 결과 저장
            results = []
            
            for model_name, model_info in self.loaded_models.items():
                model = model_info['model']
                
                # 예측
                pred = model.predict(X_test, verbose=0).flatten()
                
                if self.scaler:
                    dummy_pred = np.zeros((len(pred), n_features))
                    dummy_pred[:, 0] = pred
                    pred_original = self.scaler.inverse_transform(dummy_pred)[:, 0]
                else:
                    pred_original = pred
                
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
                    'model': model_name,
                    'mape': mape,
                    'acc_10': acc_10,
                    'r2': r2,
                    'dir_acc': dir_acc,
                    'bottleneck_acc': bottleneck_metrics['accuracy'],
                    'bottleneck_f1': bottleneck_metrics['f1_score'],
                    'overall_score': overall_score,
                    'grade': grade
                })
                
                self.update_status(f"{model_name} 평가 완료")
            
            # 테이블 업데이트
            self.eval_table.setRowCount(len(results))
            for i, res in enumerate(results):
                self.eval_table.setItem(i, 0, QTableWidgetItem(res['model']))
                self.eval_table.setItem(i, 1, QTableWidgetItem(f"{res['mape']:.2f}"))
                self.eval_table.setItem(i, 2, QTableWidgetItem(f"{res['acc_10']:.1f}"))
                self.eval_table.setItem(i, 3, QTableWidgetItem(f"{res['r2']:.1f}"))
                self.eval_table.setItem(i, 4, QTableWidgetItem(f"{res['dir_acc']:.1f}"))
                self.eval_table.setItem(i, 5, QTableWidgetItem(f"{res['bottleneck_acc']:.1f}"))
                self.eval_table.setItem(i, 6, QTableWidgetItem(f"{res['bottleneck_f1']:.1f}"))
                self.eval_table.setItem(i, 7, QTableWidgetItem(f"{res['overall_score']:.1f}"))
                self.eval_table.setItem(i, 8, QTableWidgetItem(res['grade']))
            
            # 최고 성능 모델
            best_model = max(results, key=lambda x: x['overall_score'])
            detail = f"📊 평가 완료 - {len(results)}개 모델\n\n"
            detail += f"🏆 최고 성능 모델: {best_model['model']}\n"
            detail += f"   종합 점수: {best_model['overall_score']:.1f}%\n"
            detail += f"   성능 등급: {best_model['grade']}\n"
            detail += f"   MAPE: {best_model['mape']:.2f}%\n"
            detail += f"   병목 정확도: {best_model['bottleneck_acc']:.1f}%"
            
            self.eval_detail_text.setText(detail)
            self.update_status("모델 평가 완료")
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"평가 실패: {str(e)}")
            self.update_status("평가 실패")
            
    def create_visualization(self):
        """시각화 생성"""
        viz_type = self.viz_type_combo.currentText()
        
        self.viz_figure.clear()
        
        if viz_type == "예측 vs 실제" and self.current_data and 'X_test' in self.current_data:
            self.create_prediction_plot()
        elif viz_type == "오차 분포" and self.current_data and 'X_test' in self.current_data:
            self.create_error_distribution_plot()
        elif viz_type == "병목 구간 히트맵" and self.current_data and 'X_test' in self.current_data:
            self.create_bottleneck_heatmap()
        elif viz_type == "모델별 성능 비교" and self.loaded_models:
            self.create_performance_comparison()
        elif viz_type == "시계열 분해":
            self.create_time_series_decomposition()
        else:
            ax = self.viz_figure.add_subplot(111)
            ax.text(0.5, 0.5, '데이터가 없거나 모델이 로드되지 않았습니다', 
                   ha='center', va='center', fontsize=12)
        
        self.viz_canvas.draw()
        
    def create_prediction_plot(self):
        """예측 vs 실제 플롯"""
        if not self.loaded_models:
            return
            
        # 첫 번째 모델 사용
        model_name = list(self.loaded_models.keys())[0]
        model = self.loaded_models[model_name]['model']
        
        X_test = self.current_data['X_test'][:200]  # 샘플
        y_test = self.current_data['y_test'][:200]
        
        pred = model.predict(X_test, verbose=0).flatten()
        
        # 역스케일링
        if self.scaler:
            n_features = self.scaler.n_features_in_
            
            dummy_pred = np.zeros((len(pred), n_features))
            dummy_pred[:, 0] = pred
            pred_original = self.scaler.inverse_transform(dummy_pred)[:, 0]
            
            dummy_y = np.zeros((len(y_test), n_features))
            dummy_y[:, 0] = y_test
            y_original = self.scaler.inverse_transform(dummy_y)[:, 0]
        else:
            pred_original = pred
            y_original = y_test
        
        ax = self.viz_figure.add_subplot(111)
        
        x_range = range(len(y_original))
        ax.plot(x_range, y_original, label='실제값', color='blue', linewidth=2, alpha=0.7)
        ax.plot(x_range, pred_original, label=f'예측값 ({model_name})', color='red', linewidth=1.5, alpha=0.7)
        ax.axhline(y=2000, color='orange', linestyle='--', label='병목 임계값', alpha=0.5)
        
        ax.set_title('예측 vs 실제 비교', fontsize=14)
        ax.set_xlabel('시간 인덱스', fontsize=12)
        ax.set_ylabel('물류량', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def create_error_distribution_plot(self):
        """오차 분포 플롯"""
        if not self.loaded_models:
            return
            
        ax = self.viz_figure.add_subplot(111)
        
        for model_name, model_info in self.loaded_models.items():
            model = model_info['model']
            
            X_test = self.current_data['X_test']
            y_test = self.current_data['y_test']
            
            pred = model.predict(X_test, verbose=0).flatten()
            
            # 역스케일링
            if self.scaler:
                n_features = self.scaler.n_features_in_
                
                dummy_pred = np.zeros((len(pred), n_features))
                dummy_pred[:, 0] = pred
                pred_original = self.scaler.inverse_transform(dummy_pred)[:, 0]
                
                dummy_y = np.zeros((len(y_test), n_features))
                dummy_y[:, 0] = y_test
                y_original = self.scaler.inverse_transform(dummy_y)[:, 0]
            else:
                pred_original = pred
                y_original = y_test
            
            errors = pred_original - y_original
            ax.hist(errors, bins=50, alpha=0.5, label=model_name)
        
        ax.set_title('모델별 오차 분포', fontsize=14)
        ax.set_xlabel('오차', fontsize=12)
        ax.set_ylabel('빈도', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def create_bottleneck_heatmap(self):
        """병목 구간 히트맵"""
        # 구현 예정
        ax = self.viz_figure.add_subplot(111)
        ax.text(0.5, 0.5, '병목 구간 히트맵 (구현 예정)', ha='center', va='center')
        
    def create_performance_comparison(self):
        """모델별 성능 비교"""
        # 구현 예정
        ax = self.viz_figure.add_subplot(111)
        ax.text(0.5, 0.5, '모델별 성능 비교 (구현 예정)', ha='center', va='center')
        
    def create_time_series_decomposition(self):
        """시계열 분해"""
        # 구현 예정
        ax = self.viz_figure.add_subplot(111)
        ax.text(0.5, 0.5, '시계열 분해 (구현 예정)', ha='center', va='center')
        
    def save_visualization(self):
        """시각화 저장"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "차트 저장", "",
            "PNG Files (*.png);;JPG Files (*.jpg);;PDF Files (*.pdf)"
        )
        if filename:
            self.viz_figure.savefig(filename, dpi=300, bbox_inches='tight')
            self.update_status(f"차트 저장 완료: {filename}")
            
    def export_evaluation_results(self):
        """평가 결과 내보내기"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "평가 결과 내보내기", "",
            "CSV Files (*.csv);;Excel Files (*.xlsx)"
        )
        
        if filename:
            # 테이블 데이터를 DataFrame으로 변환
            data = []
            for row in range(self.eval_table.rowCount()):
                row_data = []
                for col in range(self.eval_table.columnCount()):
                    item = self.eval_table.item(row, col)
                    row_data.append(item.text() if item else "")
                data.append(row_data)
            
            df = pd.DataFrame(data, columns=[
                '모델', 'MAPE(%)', '정확도(10%)', 'R² Score', 
                '방향성 정확도', '병목 정확도', '병목 F1', '종합 점수', '등급'
            ])
            
            if filename.endswith('.csv'):
                df.to_csv(filename, index=False)
            else:
                df.to_excel(filename, index=False)
            
            self.update_status(f"평가 결과 내보내기 완료: {filename}")
            
    def show_about(self):
        """프로그램 정보"""
        about_text = """
반도체 FAB 물류 예측 하이브리드 딥러닝 시스템

버전: 3.2 (한글 폰트 지원)
개발: 2024

특징:
- 기존 학습된 모델 불러오기 및 활용
- LSTM, RNN, GRU, Bi-LSTM 통합 하이브리드 모델
- CPU 기반 실행 지원
- 실시간 병목 구간 예측
- 10분 후 물류량 예측
- 퍼센트 기반 성능 평가
- 한글 폰트 자동 감지 및 설정

본 시스템은 반도체 제조 공정의 물류 흐름을
딥러닝으로 예측하여 생산성을 향상시킵니다.
        """
        QMessageBox.about(self, "프로그램 정보", about_text)
        
    def show_font_info(self):
        """폰트 정보 표시"""
        font_info = "현재 Matplotlib 폰트 설정:\n\n"
        font_info += f"Font Family: {plt.rcParams['font.family']}\n"
        font_info += f"Sans-serif: {plt.rcParams['font.sans-serif']}\n"
        font_info += f"Unicode Minus: {plt.rcParams['axes.unicode_minus']}\n\n"
        
        if KOREAN_FONT_AVAILABLE:
            font_info += "✓ 한글 폰트가 정상적으로 설정되었습니다."
        else:
            font_info += "⚠ 한글 폰트를 찾을 수 없습니다.\n"
            font_info += "NanumGothic 폰트 설치를 권장합니다."
        
        QMessageBox.information(self, "폰트 정보", font_info)
        
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