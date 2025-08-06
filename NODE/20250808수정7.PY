# -*- coding: utf-8 -*-
"""
Semiconductor Logistics Prediction System - Professional Dark Theme UI
======================================================================
LSTM, RNN, GRU, ARIMA Hybrid Model for FAB Logistics Prediction

Features:
1. Data Loading & Preprocessing
2. Individual Model Training with Checkpoints
3. Real-time Prediction & Bottleneck Detection
4. Model Performance Evaluation

Version: 2.0 - Dark Professional Edition
Date: 2024
"""

import sys
import os
import json
import pickle
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import threading
import time

# Data Processing
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, GRU, SimpleRNN, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# GUI
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# Visualization
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Dark theme for matplotlib
plt.style.use('dark_background')

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# CPU mode settings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

# Random seed
RANDOM_SEED = 2079936
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class DarkPalette:
    """Dark theme color palette"""
    # Background colors
    BG_DARKEST = "#0d1117"
    BG_DARK = "#161b22"
    BG_MEDIUM = "#21262d"
    BG_LIGHT = "#30363d"
    
    # Text colors
    TEXT_PRIMARY = "#c9d1d9"
    TEXT_SECONDARY = "#8b949e"
    TEXT_DISABLED = "#484f58"
    
    # Accent colors
    ACCENT_BLUE = "#58a6ff"
    ACCENT_GREEN = "#3fb950"
    ACCENT_RED = "#f85149"
    ACCENT_YELLOW = "#d29922"
    ACCENT_PURPLE = "#a371f7"
    
    # Button colors
    BTN_PRIMARY = "#238636"
    BTN_PRIMARY_HOVER = "#2ea043"
    BTN_SECONDARY = "#21262d"
    BTN_DANGER = "#da3633"


def get_dark_stylesheet():
    """Get complete dark theme stylesheet"""
    return f"""
    /* Main Window */
    QMainWindow {{
        background-color: {DarkPalette.BG_DARKEST};
        color: {DarkPalette.TEXT_PRIMARY};
    }}
    
    /* Central Widget */
    QWidget {{
        background-color: {DarkPalette.BG_DARKEST};
        color: {DarkPalette.TEXT_PRIMARY};
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 13px;
    }}
    
    /* Tab Widget */
    QTabWidget::pane {{
        background-color: {DarkPalette.BG_DARK};
        border: 1px solid {DarkPalette.BG_LIGHT};
        border-radius: 8px;
        padding: 10px;
    }}
    
    QTabBar::tab {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 {DarkPalette.BG_MEDIUM}, stop:1 {DarkPalette.BG_DARK});
        color: {DarkPalette.TEXT_SECONDARY};
        padding: 12px 30px;
        margin-right: 3px;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
        font-weight: bold;
        min-width: 150px;
    }}
    
    QTabBar::tab:selected {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 {DarkPalette.ACCENT_BLUE}, stop:0.5 #1f6feb, stop:1 #0969da);
        color: white;
    }}
    
    QTabBar::tab:hover:!selected {{
        background-color: {DarkPalette.BG_LIGHT};
    }}
    
    /* Group Box */
    QGroupBox {{
        background-color: {DarkPalette.BG_DARK};
        border: 2px solid {DarkPalette.BG_LIGHT};
        border-radius: 10px;
        margin-top: 15px;
        padding-top: 20px;
        font-weight: bold;
        font-size: 14px;
    }}
    
    QGroupBox::title {{
        color: {DarkPalette.ACCENT_BLUE};
        subcontrol-origin: margin;
        left: 15px;
        padding: 0 10px;
        background-color: {DarkPalette.BG_DARK};
    }}
    
    /* Push Button */
    QPushButton {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 {DarkPalette.BTN_PRIMARY}, stop:1 {DarkPalette.BTN_PRIMARY_HOVER});
        color: white;
        border: none;
        padding: 10px 25px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 13px;
        min-height: 35px;
    }}
    
    QPushButton:hover {{
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
            stop:0 {DarkPalette.BTN_PRIMARY_HOVER}, stop:1 {DarkPalette.ACCENT_GREEN});
    }}
    
    QPushButton:pressed {{
        background: #1a7f37;
    }}
    
    QPushButton:disabled {{
        background-color: {DarkPalette.BG_MEDIUM};
        color: {DarkPalette.TEXT_DISABLED};
    }}
    
    /* Input Fields */
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
        background-color: {DarkPalette.BG_DARKEST};
        color: {DarkPalette.TEXT_PRIMARY};
        border: 2px solid {DarkPalette.BG_LIGHT};
        border-radius: 6px;
        padding: 8px 12px;
        selection-background-color: {DarkPalette.ACCENT_BLUE};
        min-height: 20px;
    }}
    
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
        border: 2px solid {DarkPalette.ACCENT_BLUE};
        outline: none;
    }}
    
    /* Text Edit */
    QTextEdit {{
        background-color: {DarkPalette.BG_DARKEST};
        color: {DarkPalette.TEXT_PRIMARY};
        border: 2px solid {DarkPalette.BG_LIGHT};
        border-radius: 6px;
        padding: 8px;
        selection-background-color: {DarkPalette.ACCENT_BLUE};
    }}
    
    /* Progress Bar */
    QProgressBar {{
        background-color: {DarkPalette.BG_MEDIUM};
        border: 2px solid {DarkPalette.BG_LIGHT};
        border-radius: 6px;
        text-align: center;
        color: white;
        font-weight: bold;
        min-height: 25px;
    }}
    
    QProgressBar::chunk {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 {DarkPalette.ACCENT_BLUE}, stop:1 {DarkPalette.ACCENT_GREEN});
        border-radius: 4px;
    }}
    
    /* Check Box */
    QCheckBox {{
        color: {DarkPalette.TEXT_PRIMARY};
        spacing: 8px;
    }}
    
    QCheckBox::indicator {{
        width: 20px;
        height: 20px;
        border: 2px solid {DarkPalette.BG_LIGHT};
        border-radius: 4px;
        background-color: {DarkPalette.BG_DARKEST};
    }}
    
    QCheckBox::indicator:checked {{
        background-color: {DarkPalette.ACCENT_GREEN};
        border: 2px solid {DarkPalette.ACCENT_GREEN};
    }}
    
    /* Table Widget */
    QTableWidget {{
        background-color: {DarkPalette.BG_DARKEST};
        color: {DarkPalette.TEXT_PRIMARY};
        border: 1px solid {DarkPalette.BG_LIGHT};
        gridline-color: {DarkPalette.BG_MEDIUM};
    }}
    
    QHeaderView::section {{
        background-color: {DarkPalette.BG_DARK};
        color: {DarkPalette.ACCENT_BLUE};
        padding: 8px;
        border: none;
        font-weight: bold;
    }}
    
    /* Scroll Bar */
    QScrollBar:vertical {{
        background: {DarkPalette.BG_DARKEST};
        width: 12px;
        border-radius: 6px;
    }}
    
    QScrollBar::handle:vertical {{
        background: {DarkPalette.BG_LIGHT};
        border-radius: 6px;
        min-height: 30px;
    }}
    
    QScrollBar::handle:vertical:hover {{
        background: {DarkPalette.TEXT_DISABLED};
    }}
    
    /* Label */
    QLabel {{
        color: {DarkPalette.TEXT_PRIMARY};
    }}
    
    /* Status Bar */
    QStatusBar {{
        background-color: {DarkPalette.BG_DARKEST};
        color: {DarkPalette.TEXT_SECONDARY};
        border-top: 1px solid {DarkPalette.BG_LIGHT};
    }}
    
    /* Menu Bar */
    QMenuBar {{
        background-color: {DarkPalette.BG_DARKEST};
        color: {DarkPalette.TEXT_PRIMARY};
        border-bottom: 1px solid {DarkPalette.BG_LIGHT};
    }}
    
    QMenuBar::item:selected {{
        background-color: {DarkPalette.BG_DARK};
    }}
    
    QMenu {{
        background-color: {DarkPalette.BG_DARK};
        color: {DarkPalette.TEXT_PRIMARY};
        border: 1px solid {DarkPalette.BG_LIGHT};
    }}
    
    QMenu::item:selected {{
        background-color: {DarkPalette.ACCENT_BLUE};
    }}
    """


class WorkerSignals(QObject):
    """Worker thread signals"""
    started = pyqtSignal()
    progress = pyqtSignal(int, str)
    log = pyqtSignal(str, str)  # message, level
    result = pyqtSignal(object)
    error = pyqtSignal(str)
    finished = pyqtSignal()


class ModelCard(QFrame):
    """Model selection card widget"""
    toggled = pyqtSignal(str, bool)
    configure = pyqtSignal(str)
    
    def __init__(self, model_info, parent=None):
        super().__init__(parent)
        self.model_info = model_info
        self.is_selected = False
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup card UI"""
        self.setFixedSize(280, 320)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {DarkPalette.BG_DARK};
                border: 3px solid {DarkPalette.BG_LIGHT};
                border-radius: 12px;
                padding: 15px;
            }}
            QFrame:hover {{
                border: 3px solid {DarkPalette.ACCENT_BLUE};
            }}
        """)
        
        layout = QVBoxLayout()
        
        # Header with checkbox
        header_layout = QHBoxLayout()
        
        # Model icon
        icon_label = QLabel(self.model_info['icon'])
        icon_label.setStyleSheet("font-size: 36px;")
        header_layout.addWidget(icon_label)
        
        header_layout.addStretch()
        
        # Selection checkbox
        self.checkbox = QCheckBox()
        self.checkbox.setStyleSheet("""
            QCheckBox::indicator {
                width: 30px;
                height: 30px;
            }
        """)
        self.checkbox.toggled.connect(self.on_toggle)
        header_layout.addWidget(self.checkbox)
        
        layout.addLayout(header_layout)
        
        # Model name
        name_label = QLabel(self.model_info['name'])
        name_label.setStyleSheet(f"""
            font-size: 20px;
            font-weight: bold;
            color: {DarkPalette.ACCENT_BLUE};
        """)
        layout.addWidget(name_label)
        
        # Model type badge
        type_label = QLabel(self.model_info['type'])
        type_label.setStyleSheet(f"""
            background-color: {DarkPalette.BG_MEDIUM};
            color: {DarkPalette.TEXT_SECONDARY};
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 11px;
        """)
        type_label.setMaximumWidth(150)
        layout.addWidget(type_label)
        
        # Description
        desc_label = QLabel(self.model_info['description'])
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet(f"""
            color: {DarkPalette.TEXT_PRIMARY};
            font-size: 12px;
            margin: 10px 0;
        """)
        layout.addWidget(desc_label)
        
        # Configure button
        config_btn = QPushButton("Configure")
        config_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {DarkPalette.BG_MEDIUM};
                color: {DarkPalette.TEXT_PRIMARY};
                padding: 8px;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background-color: {DarkPalette.BG_LIGHT};
            }}
        """)
        config_btn.clicked.connect(lambda: self.configure.emit(self.model_info['name']))
        layout.addWidget(config_btn)
        
        layout.addStretch()
        self.setLayout(layout)
        
    def on_toggle(self, checked):
        """Handle selection toggle"""
        self.is_selected = checked
        if checked:
            self.setStyleSheet(f"""
                QFrame {{
                    background-color: {DarkPalette.BG_DARK};
                    border: 3px solid {DarkPalette.ACCENT_GREEN};
                    border-radius: 12px;
                    padding: 15px;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QFrame {{
                    background-color: {DarkPalette.BG_DARK};
                    border: 3px solid {DarkPalette.BG_LIGHT};
                    border-radius: 12px;
                    padding: 15px;
                }}
            """)
        self.toggled.emit(self.model_info['name'], checked)


class TrainingTab(QWidget):
    """Training tab widget"""
    def __init__(self):
        super().__init__()
        self.selected_models = {}
        self.model_configs = {}
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout()
        
        # Title section
        title_layout = QHBoxLayout()
        title_label = QLabel("Model Training Center")
        title_label.setStyleSheet(f"""
            font-size: 24px;
            font-weight: bold;
            color: {DarkPalette.ACCENT_BLUE};
            padding: 10px;
        """)
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        layout.addLayout(title_layout)
        
        # Model selection section
        model_group = QGroupBox("Select Models to Train")
        model_layout = QGridLayout()
        
        # Model cards
        self.model_cards = []
        models = [
            {
                'name': 'LSTM',
                'type': 'Recurrent Neural Network',
                'icon': '🧠',
                'description': 'Long Short-Term Memory for time series prediction. Best for transfer time forecasting.'
            },
            {
                'name': 'RNN',
                'type': 'Recurrent Neural Network',
                'icon': '🔄',
                'description': 'Simple RNN for sequential data. Suitable for bottleneck detection patterns.'
            },
            {
                'name': 'GRU',
                'type': 'Gated Recurrent Unit',
                'icon': '⚡',
                'description': 'Faster training than LSTM. Good balance between performance and speed.'
            },
            {
                'name': 'Bi-LSTM',
                'type': 'Bidirectional LSTM',
                'icon': '↔️',
                'description': 'Processes data in both directions. Enhanced pattern recognition.'
            },
            {
                'name': 'ARIMA',
                'type': 'Statistical Model',
                'icon': '📊',
                'description': 'Time series forecasting for production throughput prediction.'
            },
            {
                'name': 'Ensemble',
                'type': 'Combined Model',
                'icon': '🎯',
                'description': 'Combines multiple models for robust predictions.'
            }
        ]
        
        row, col = 0, 0
        for model_info in models:
            card = ModelCard(model_info)
            card.toggled.connect(self.on_model_toggled)
            card.configure.connect(self.configure_model)
            self.model_cards.append(card)
            model_layout.addWidget(card, row, col)
            col += 1
            if col > 2:
                col = 0
                row += 1
                
        model_group.setLayout(model_layout)
        
        # Scroll area for model cards
        scroll = QScrollArea()
        scroll.setWidget(model_group)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        
        # Training configuration
        config_group = QGroupBox("Training Configuration")
        config_layout = QGridLayout()
        
        # Data settings
        config_layout.addWidget(QLabel("Data File:"), 0, 0)
        self.data_path = QLineEdit()
        self.data_path.setPlaceholderText("Select training data file...")
        config_layout.addWidget(self.data_path, 0, 1, 1, 2)
        
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_data)
        config_layout.addWidget(browse_btn, 0, 3)
        
        # Training parameters
        config_layout.addWidget(QLabel("Epochs:"), 1, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(10, 1000)
        self.epochs_spin.setValue(100)
        config_layout.addWidget(self.epochs_spin, 1, 1)
        
        config_layout.addWidget(QLabel("Batch Size:"), 1, 2)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(16, 512)
        self.batch_spin.setValue(64)
        self.batch_spin.setSingleStep(16)
        config_layout.addWidget(self.batch_spin, 1, 3)
        
        config_layout.addWidget(QLabel("Learning Rate:"), 2, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 0.1)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setDecimals(4)
        config_layout.addWidget(self.lr_spin, 2, 1)
        
        config_layout.addWidget(QLabel("Validation Split:"), 2, 2)
        self.val_spin = QDoubleSpinBox()
        self.val_spin.setRange(0.1, 0.5)
        self.val_spin.setValue(0.2)
        self.val_spin.setSingleStep(0.05)
        config_layout.addWidget(self.val_spin, 2, 3)
        
        # Checkboxes for options
        self.checkpoint_check = QCheckBox("Use Checkpoints")
        self.checkpoint_check.setChecked(True)
        config_layout.addWidget(self.checkpoint_check, 3, 0, 1, 2)
        
        self.early_stop_check = QCheckBox("Early Stopping")
        self.early_stop_check.setChecked(True)
        config_layout.addWidget(self.early_stop_check, 3, 2, 1, 2)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # Progress section
        progress_group = QGroupBox("Training Progress")
        progress_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready to train")
        self.status_label.setStyleSheet(f"color: {DarkPalette.TEXT_SECONDARY};")
        progress_layout.addWidget(self.status_label)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        progress_layout.addWidget(self.log_text)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.train_btn = QPushButton("Start Training")
        self.train_btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {DarkPalette.ACCENT_GREEN}, stop:1 {DarkPalette.BTN_PRIMARY});
                font-size: 14px;
                padding: 12px 30px;
            }}
        """)
        self.train_btn.clicked.connect(self.start_training)
        button_layout.addWidget(self.train_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {DarkPalette.ACCENT_RED};
                font-size: 14px;
                padding: 12px 30px;
            }}
        """)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
    def on_model_toggled(self, model_name, checked):
        """Handle model selection"""
        self.selected_models[model_name] = checked
        count = sum(self.selected_models.values())
        self.status_label.setText(f"{count} model(s) selected")
        
    def configure_model(self, model_name):
        """Open model configuration dialog"""
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Configure {model_name}")
        dialog.setModal(True)
        dialog.setStyleSheet(get_dark_stylesheet())
        dialog.setFixedSize(400, 300)
        
        layout = QVBoxLayout()
        layout.addWidget(QLabel(f"Configuration for {model_name}"))
        
        # Add model-specific configuration options here
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        dialog.setLayout(layout)
        dialog.exec_()
        
    def browse_data(self):
        """Browse for training data file"""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select Training Data",
            "",
            "CSV Files (*.csv);;All Files (*.*)"
        )
        if filename:
            self.data_path.setText(filename)
            
    def start_training(self):
        """Start training selected models"""
        selected = [name for name, checked in self.selected_models.items() if checked]
        if not selected:
            QMessageBox.warning(self, "Warning", "Please select at least one model to train.")
            return
            
        if not self.data_path.text():
            QMessageBox.warning(self, "Warning", "Please select training data file.")
            return
            
        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.log_text.append(f"Starting training for: {', '.join(selected)}")
        
        # Start training in background thread
        # Implementation would go here


class PredictionTab(QWidget):
    """Prediction tab widget"""
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout()
        
        # Title
        title_label = QLabel("Real-time Prediction")
        title_label.setStyleSheet(f"""
            font-size: 24px;
            font-weight: bold;
            color: {DarkPalette.ACCENT_BLUE};
            padding: 10px;
        """)
        layout.addWidget(title_label)
        
        # Content goes here
        layout.addStretch()
        
        self.setLayout(layout)


class EvaluationTab(QWidget):
    """Evaluation tab widget"""
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout()
        
        # Title
        title_label = QLabel("Model Evaluation")
        title_label.setStyleSheet(f"""
            font-size: 24px;
            font-weight: bold;
            color: {DarkPalette.ACCENT_BLUE};
            padding: 10px;
        """)
        layout.addWidget(title_label)
        
        # Content goes here
        layout.addStretch()
        
        self.setLayout(layout)


class MainWindow(QMainWindow):
    """Main application window"""
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI"""
        self.setWindowTitle("Semiconductor Logistics AI System")
        self.setGeometry(100, 100, 1600, 900)
        
        # Apply dark theme
        self.setStyleSheet(get_dark_stylesheet())
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Header
        header_frame = QFrame()
        header_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {DarkPalette.BG_DARK};
                border-bottom: 2px solid {DarkPalette.BG_LIGHT};
                padding: 10px;
            }}
        """)
        header_layout = QHBoxLayout()
        
        logo_label = QLabel("🚀")
        logo_label.setStyleSheet("font-size: 32px;")
        header_layout.addWidget(logo_label)
        
        title_label = QLabel("Semiconductor FAB Logistics Prediction System")
        title_label.setStyleSheet(f"""
            font-size: 20px;
            font-weight: bold;
            color: {DarkPalette.ACCENT_BLUE};
        """)
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        status_label = QLabel("LSTM/RNN/GRU Hybrid Models")
        status_label.setStyleSheet(f"""
            color: {DarkPalette.TEXT_SECONDARY};
            font-size: 14px;
        """)
        header_layout.addWidget(status_label)
        
        header_frame.setLayout(header_layout)
        main_layout.addWidget(header_frame)
        
        # Tab widget
        self.tabs = QTabWidget()
        
        # Add tabs
        self.training_tab = TrainingTab()
        self.tabs.addTab(self.training_tab, "Training")
        
        self.prediction_tab = PredictionTab()
        self.tabs.addTab(self.prediction_tab, "Prediction")
        
        self.evaluation_tab = EvaluationTab()
        self.tabs.addTab(self.evaluation_tab, "Evaluation")
        
        main_layout.addWidget(self.tabs)
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
        # Menu bar
        self.create_menu_bar()
        
    def create_menu_bar(self):
        """Create menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        new_action = QAction("New Project", self)
        new_action.setShortcut("Ctrl+N")
        file_menu.addAction(new_action)
        
        open_action = QAction("Open Project", self)
        open_action.setShortcut("Ctrl+O")
        file_menu.addAction(open_action)
        
        save_action = QAction("Save Project", self)
        save_action.setShortcut("Ctrl+S")
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        
        settings_action = QAction("Settings", self)
        tools_menu.addAction(settings_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def show_about(self):
        """Show about dialog"""
        QMessageBox.information(
            self,
            "About",
            "Semiconductor Logistics Prediction System\n\n"
            "Version 2.0\n"
            "LSTM/RNN/GRU Hybrid Models\n\n"
            "Developed for FAB logistics optimization"
        )


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    
    # Set application info
    app.setApplicationName("Semiconductor Logistics AI")
    app.setOrganizationName("FAB Systems")
    
    # Set fusion style for better dark theme
    app.setStyle("Fusion")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()