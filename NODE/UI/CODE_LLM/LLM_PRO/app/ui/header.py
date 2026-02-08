#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
헤더바 - 상태 표시, 환경 전환, 모델 선택
"""

import os
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QPushButton, QComboBox
)
from PySide6.QtCore import Signal
from .theme import COLORS


class HeaderWidget(QWidget):
    """헤더 위젯"""
    env_changed = Signal(str)
    model_changed = Signal(str)
    token_reload = Signal()

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.setFixedHeight(48)
        self.setStyleSheet(f"""
            background-color: {COLORS['darker']};
            border-bottom: 1px solid {COLORS['border']};
        """)
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(10)

        # 상태 도트
        self.status_dot = QLabel()
        self.status_dot.setFixedSize(8, 8)
        self.status_dot.setStyleSheet(f"""
            background-color: {COLORS['green']};
            border-radius: 4px;
        """)
        layout.addWidget(self.status_dot)

        # 상태 텍스트
        self.status_label = QLabel("준비")
        self.status_label.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 12px;")
        layout.addWidget(self.status_label)

        layout.addStretch()

        # 환경 전환
        env_label = QLabel("환경")
        env_label.setStyleSheet(f"""
            color: {COLORS['text_muted']};
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 1px;
        """)
        layout.addWidget(env_label)

        self.env_buttons = {}
        env_labels = {"dev": "개발", "prod": "운영", "common": "공통", "local": "로컬"}
        for env_key in ["dev", "prod", "common", "local"]:
            btn = QPushButton(env_labels[env_key])
            btn.setCheckable(True)
            btn.setFixedSize(52, 28)
            btn.setStyleSheet(f"""
                QPushButton {{
                    padding: 0;
                    border-radius: 6px;
                    font-size: 11px;
                    font-weight: 600;
                    border: 1px solid {COLORS['border']};
                    background: transparent;
                    color: {COLORS['text_muted']};
                }}
                QPushButton:hover {{
                    border-color: {COLORS['primary']};
                    color: {COLORS['text']};
                    background: {COLORS['surface']};
                }}
                QPushButton:checked {{
                    background: {COLORS['primary']};
                    color: white;
                    border-color: {COLORS['primary']};
                }}
            """)
            btn.clicked.connect(lambda checked, k=env_key: self._on_env_click(k))
            self.env_buttons[env_key] = btn
            layout.addWidget(btn)

        # 구분선
        sep = QLabel()
        sep.setFixedSize(1, 20)
        sep.setStyleSheet(f"background-color: {COLORS['border']};")
        layout.addWidget(sep)

        # GGUF 모델 선택
        self.model_combo = QComboBox()
        self.model_combo.setFixedWidth(140)
        self.model_combo.setStyleSheet(f"""
            QComboBox {{
                font-size: 11px;
                padding: 4px 8px;
                border-radius: 6px;
            }}
        """)
        for key, cfg in self.config.AVAILABLE_GGUF_MODELS.items():
            exists = os.path.exists(cfg["path"])
            label = f"{cfg['name']}" + ("" if exists else " (없음)")
            self.model_combo.addItem(label, key)
        self.model_combo.currentIndexChanged.connect(self._on_model_change)
        layout.addWidget(self.model_combo)

        # 토큰 갱신
        reload_btn = QPushButton("🔑 토큰")
        reload_btn.setFixedHeight(28)
        reload_btn.setToolTip("API 토큰 새로고침")
        reload_btn.setStyleSheet(f"""
            QPushButton {{
                padding: 2px 10px;
                border-radius: 6px;
                font-size: 11px;
                border: 1px solid {COLORS['border']};
                background: transparent;
                color: {COLORS['text_muted']};
            }}
            QPushButton:hover {{
                border-color: {COLORS['yellow']};
                color: {COLORS['yellow']};
                background: {COLORS['yellow_dim']};
            }}
        """)
        reload_btn.clicked.connect(self.token_reload.emit)
        layout.addWidget(reload_btn)

        self._update_env_buttons()

    def _on_env_click(self, env_key: str):
        self.env_changed.emit(env_key)

    def _on_model_change(self, index):
        model_key = self.model_combo.itemData(index)
        if model_key:
            self.model_changed.emit(model_key)

    def _update_env_buttons(self):
        for key, btn in self.env_buttons.items():
            btn.setChecked(key == self.config.env_mode)

    def update_status(self, text: str, is_ok: bool = True):
        self.status_label.setText(text)
        color = COLORS['green'] if is_ok else COLORS['red']
        self.status_dot.setStyleSheet(f"background-color: {color}; border-radius: 4px;")

    def update_env(self, env_mode: str):
        self.config.env_mode = env_mode
        self._update_env_buttons()
        self.model_combo.setEnabled(env_mode == "local")

    def set_busy(self, busy: bool):
        if busy:
            self.update_status("처리 중...", True)
            self.status_dot.setStyleSheet(
                f"background-color: {COLORS['yellow']}; border-radius: 4px;"
            )
        else:
            mode_name = self.config.ENV_CONFIG.get(self.config.env_mode, {}).get("name", "")
            self.update_status(f"{mode_name} | {self.config.llm_mode.upper()}")
