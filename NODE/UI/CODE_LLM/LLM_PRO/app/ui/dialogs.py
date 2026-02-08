#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
다이얼로그 - 프로젝트 생성, 언어 변환 등
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QComboBox, QPushButton, QTextEdit, QFileDialog
)
from PySide6.QtCore import Qt
from .theme import COLORS


class CreateProjectDialog(QDialog):
    """프로젝트 생성 다이얼로그"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("새 프로젝트")
        self.setFixedSize(500, 380)
        self.result_data = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        # 타이틀
        title = QLabel("새 프로젝트 만들기")
        title.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {COLORS['primary']};")
        layout.addWidget(title)

        # 프로젝트 이름
        layout.addWidget(QLabel("프로젝트 이름:"))
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("예: my-web-app")
        layout.addWidget(self.name_input)

        # 유형
        layout.addWidget(QLabel("유형:"))
        self.type_combo = QComboBox()
        self.type_combo.addItem("코드 편집", "code_editing")
        self.type_combo.addItem("데이터 분석", "data_analysis")
        layout.addWidget(self.type_combo)

        # 경로
        path_layout = QHBoxLayout()
        layout.addWidget(QLabel("경로 (선택):"))
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("비워두면 자동 생성")
        path_layout.addWidget(self.path_input)
        browse_btn = QPushButton("찾아보기")
        browse_btn.setFixedWidth(70)
        browse_btn.clicked.connect(self._browse)
        path_layout.addWidget(browse_btn)
        layout.addLayout(path_layout)

        # 설명
        layout.addWidget(QLabel("설명:"))
        self.desc_input = QTextEdit()
        self.desc_input.setFixedHeight(60)
        self.desc_input.setPlaceholderText("프로젝트 설명 (선택)")
        layout.addWidget(self.desc_input)

        # 버튼
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton("취소")
        cancel_btn.setFixedSize(80, 34)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        create_btn = QPushButton("생성")
        create_btn.setObjectName("primaryBtn")
        create_btn.setFixedSize(80, 34)
        create_btn.clicked.connect(self._on_create)
        btn_layout.addWidget(create_btn)

        layout.addLayout(btn_layout)

    def _browse(self):
        folder = QFileDialog.getExistingDirectory(self, "프로젝트 폴더 선택")
        if folder:
            self.path_input.setText(folder)

    def _on_create(self):
        name = self.name_input.text().strip()
        if not name:
            self.name_input.setFocus()
            return

        self.result_data = {
            "name": name,
            "customer_type": self.type_combo.currentData(),
            "path": self.path_input.text().strip() or None,
            "description": self.desc_input.toPlainText().strip(),
        }
        self.accept()


class ConvertDialog(QDialog):
    """언어 변환 다이얼로그"""

    LANGUAGES = [
        "python", "javascript", "typescript", "java", "cpp", "c",
        "go", "rust", "csharp", "swift", "kotlin", "ruby", "php"
    ]

    def __init__(self, current_lang="python", parent=None):
        super().__init__(parent)
        self.setWindowTitle("언어 변환")
        self.setFixedSize(350, 200)
        self.result_data = None
        self.current_lang = current_lang
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("코드 언어 변환")
        title.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {COLORS['primary']};")
        layout.addWidget(title)

        # 원본 언어
        layout.addWidget(QLabel("원본 언어:"))
        self.from_combo = QComboBox()
        for lang in self.LANGUAGES:
            self.from_combo.addItem(lang)
        idx = self.from_combo.findText(self.current_lang)
        if idx >= 0:
            self.from_combo.setCurrentIndex(idx)
        layout.addWidget(self.from_combo)

        # 대상 언어
        layout.addWidget(QLabel("대상 언어:"))
        self.to_combo = QComboBox()
        for lang in self.LANGUAGES:
            self.to_combo.addItem(lang)
        layout.addWidget(self.to_combo)

        # 버튼
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton("취소")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        convert_btn = QPushButton("변환")
        convert_btn.setObjectName("primaryBtn")
        convert_btn.clicked.connect(self._on_convert)
        btn_layout.addWidget(convert_btn)

        layout.addLayout(btn_layout)

    def _on_convert(self):
        self.result_data = {
            "from_lang": self.from_combo.currentText(),
            "to_lang": self.to_combo.currentText(),
        }
        self.accept()
