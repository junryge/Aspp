#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diff 뷰어 - 코드 변경사항 표시 + 승인/거절
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextBrowser, QPushButton, QLabel
)
from PySide6.QtCore import Signal
from .theme import COLORS


class DiffViewer(QWidget):
    """Diff 표시 + 승인/거절"""
    approve_clicked = Signal(str)
    reject_clicked = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_proposal_id = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # 헤더
        header = QHBoxLayout()
        self.title_label = QLabel("변경사항")
        self.title_label.setStyleSheet(f"""
            font-size: 13px;
            font-weight: 700;
            color: {COLORS['text']};
        """)
        header.addWidget(self.title_label)
        header.addStretch()

        # 적용 버튼
        self.approve_btn = QPushButton("적용")
        self.approve_btn.setFixedHeight(32)
        self.approve_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['green_dim']};
                color: {COLORS['green']};
                border: 1px solid {COLORS['green']};
                border-radius: 6px;
                padding: 0 16px;
                font-weight: 700;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['green']};
                color: {COLORS['darkest']};
            }}
        """)
        self.approve_btn.clicked.connect(self._on_approve)
        self.approve_btn.setVisible(False)
        header.addWidget(self.approve_btn)

        # 거절 버튼
        self.reject_btn = QPushButton("거절")
        self.reject_btn.setFixedHeight(32)
        self.reject_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['red_dim']};
                color: {COLORS['red']};
                border: 1px solid {COLORS['red']};
                border-radius: 6px;
                padding: 0 16px;
                font-weight: 700;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['red']};
                color: white;
            }}
        """)
        self.reject_btn.clicked.connect(self._on_reject)
        self.reject_btn.setVisible(False)
        header.addWidget(self.reject_btn)

        layout.addLayout(header)

        # Diff 표시 영역
        self.diff_view = QTextBrowser()
        self.diff_view.setStyleSheet(f"""
            QTextBrowser {{
                background-color: {COLORS['darkest']};
                color: {COLORS['text']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                padding: 12px;
                font-family: "Consolas", "D2Coding", "Cascadia Code", monospace;
                font-size: 12px;
            }}
        """)
        self.diff_view.setHtml(f"""
            <div style="text-align:center; padding:30px; color:{COLORS['text_muted']};">
                <p style="font-size:12px;">코드 수정 후 변경사항이 여기에 표시됩니다</p>
            </div>
        """)
        layout.addWidget(self.diff_view)

    def show_diff(self, diff_text: str, proposal_id: str = None):
        """Diff 텍스트 표시"""
        self.current_proposal_id = proposal_id

        if not diff_text or diff_text == "변경사항 없음":
            self.diff_view.setHtml(f"""
                <div style="text-align:center; padding:30px; color:{COLORS['text_muted']};">
                    <p>변경사항 없음</p>
                </div>
            """)
            self.approve_btn.setVisible(False)
            self.reject_btn.setVisible(False)
            return

        html_lines = []
        for line in diff_text.split('\n'):
            escaped = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            if line.startswith('+') and not line.startswith('+++'):
                html_lines.append(
                    f'<div style="background:{COLORS["green_dim"]}; color:{COLORS["green"]};'
                    f' padding:2px 8px; border-radius:3px; margin:1px 0;">{escaped}</div>'
                )
            elif line.startswith('-') and not line.startswith('---'):
                html_lines.append(
                    f'<div style="background:{COLORS["red_dim"]}; color:{COLORS["red"]};'
                    f' padding:2px 8px; border-radius:3px; margin:1px 0;">{escaped}</div>'
                )
            elif line.startswith('@@'):
                html_lines.append(
                    f'<div style="color:{COLORS["blue"]}; font-weight:bold;'
                    f' padding:6px 8px; margin-top:8px;">{escaped}</div>'
                )
            elif line.startswith('diff ') or line.startswith('index '):
                html_lines.append(
                    f'<div style="color:{COLORS["text_muted"]}; padding:2px 8px;'
                    f' font-size:11px;">{escaped}</div>'
                )
            else:
                html_lines.append(
                    f'<div style="padding:2px 8px; color:{COLORS["text_dim"]};">{escaped}</div>'
                )

        html = f"""
        <div style="font-family:'Consolas','D2Coding','Cascadia Code',monospace;
                    font-size:12px; line-height:1.4;">
        {''.join(html_lines)}
        </div>
        """
        self.diff_view.setHtml(html)

        has_proposal = proposal_id is not None
        self.approve_btn.setVisible(has_proposal)
        self.reject_btn.setVisible(has_proposal)

    def clear(self):
        self.diff_view.setHtml(f"""
            <div style="text-align:center; padding:30px; color:{COLORS['text_muted']};">
                <p style="font-size:12px;">코드 수정 후 변경사항이 여기에 표시됩니다</p>
            </div>
        """)
        self.approve_btn.setVisible(False)
        self.reject_btn.setVisible(False)
        self.current_proposal_id = None

    def _on_approve(self):
        if self.current_proposal_id:
            self.approve_clicked.emit(self.current_proposal_id)

    def _on_reject(self):
        if self.current_proposal_id:
            self.reject_clicked.emit(self.current_proposal_id)
