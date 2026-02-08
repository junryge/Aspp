#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
사이드바 - 모드 버튼 (모던 디자인)
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QFrame
)
from PySide6.QtCore import Signal
from .theme import COLORS


class ModeButton(QPushButton):
    """모드 선택 버튼"""
    def __init__(self, icon: str, text: str, desc: str, mode_key: str, parent=None):
        super().__init__(parent)
        self.mode_key = mode_key
        self.setCheckable(True)
        self.setFixedHeight(50)
        self.setText(f"  {icon}  {text}")
        self.setToolTip(desc)
        self.setStyleSheet(f"""
            QPushButton {{
                text-align: left;
                padding: 8px 14px;
                border-radius: 10px;
                border: 1px solid transparent;
                background: transparent;
                color: {COLORS['text_muted']};
                font-size: 14px;
            }}
            QPushButton:hover {{
                background: {COLORS['surface']};
                color: {COLORS['text']};
                border-color: {COLORS['border']};
            }}
            QPushButton:checked {{
                background: {COLORS['primary_glow']};
                color: {COLORS['primary_hover']};
                border-color: {COLORS['primary']};
                font-weight: bold;
            }}
        """)


class SidebarWidget(QWidget):
    """사이드바 위젯"""
    mode_changed = Signal(str)

    MODES = [
        ("general", "대화", "일반 질문 / Q&A"),
        ("generate", "코드 생성", "새 코드 작성"),
        ("aider", "코드 수정", "Aider 프로젝트 수정"),
        ("analysis", "분석", "데이터 분석"),
    ]

    MODE_ICONS = {
        "general": "💬",
        "generate": "⚡",
        "aider": "🔧",
        "analysis": "📊",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(220)
        self.setStyleSheet(f"""
            background-color: {COLORS['darker']};
            border-right: 1px solid {COLORS['border']};
        """)
        self.buttons = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 20, 14, 16)
        layout.setSpacing(4)

        # 로고 영역
        logo_label = QLabel("Nomos LLM")
        logo_label.setStyleSheet(f"""
            font-size: 22px;
            font-weight: 700;
            color: {COLORS['primary']};
            padding: 0 0 2px 6px;
            letter-spacing: -0.5px;
        """)
        layout.addWidget(logo_label)

        subtitle = QLabel("코딩 어시스턴트")
        subtitle.setStyleSheet(f"""
            font-size: 11px;
            color: {COLORS['text_muted']};
            padding: 0 0 16px 6px;
        """)
        layout.addWidget(subtitle)

        # 구분선
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"background-color: {COLORS['border']}; max-height: 1px;")
        layout.addWidget(line)

        # 모드 라벨
        mode_label = QLabel("  모드")
        mode_label.setStyleSheet(f"""
            font-size: 10px;
            font-weight: 700;
            color: {COLORS['text_muted']};
            padding: 14px 0 8px 0;
            letter-spacing: 2px;
        """)
        layout.addWidget(mode_label)

        # 모드 버튼들
        for key, name, desc in self.MODES:
            icon = self.MODE_ICONS[key]
            btn = ModeButton(icon, name, desc, key)
            btn.clicked.connect(lambda checked, k=key: self._on_mode_click(k))
            self.buttons[key] = btn
            layout.addWidget(btn)
            layout.addSpacing(2)

        layout.addStretch()

        # 하단 정보
        version_label = QLabel("v1.0  |  GGUF + API")
        version_label.setStyleSheet(f"""
            font-size: 10px;
            color: {COLORS['text_muted']};
            padding: 8px 6px;
        """)
        layout.addWidget(version_label)

        # 기본 선택
        self.set_mode("general")

    def _on_mode_click(self, mode_key: str):
        self.set_mode(mode_key)
        self.mode_changed.emit(mode_key)

    def set_mode(self, mode_key: str):
        """모드 설정 (버튼 상태 업데이트)"""
        for key, btn in self.buttons.items():
            btn.setChecked(key == mode_key)
