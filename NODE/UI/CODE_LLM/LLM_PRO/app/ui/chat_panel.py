#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
채팅 패널 - AI 응답 표시 + 입력 (모던 UI)
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextBrowser, QLineEdit,
    QPushButton, QCheckBox, QLabel, QMenu, QApplication
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QAction
from .theme import COLORS


def _make_korean_menu_style():
    return f"""
        QMenu {{
            background-color: {COLORS['surface']};
            color: {COLORS['text']};
            border: 1px solid {COLORS['border']};
            border-radius: 6px;
            padding: 4px;
            font-size: 12px;
        }}
        QMenu::item {{
            padding: 6px 24px;
            border-radius: 4px;
        }}
        QMenu::item:selected {{
            background-color: {COLORS['primary']};
            color: white;
        }}
        QMenu::item:disabled {{
            color: {COLORS['text_muted']};
        }}
        QMenu::separator {{
            height: 1px;
            background: {COLORS['border']};
            margin: 4px 8px;
        }}
    """


class KoreanLineEdit(QLineEdit):
    """한글 우클릭 메뉴 QLineEdit"""

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        menu.setStyleSheet(_make_korean_menu_style())

        undo_act = menu.addAction("되돌리기")
        undo_act.setEnabled(self.isUndoAvailable())
        undo_act.triggered.connect(self.undo)

        redo_act = menu.addAction("다시실행")
        redo_act.setEnabled(self.isRedoAvailable())
        redo_act.triggered.connect(self.redo)

        menu.addSeparator()

        cut_act = menu.addAction("잘라내기")
        cut_act.setEnabled(self.hasSelectedText())
        cut_act.triggered.connect(self.cut)

        copy_act = menu.addAction("복사")
        copy_act.setEnabled(self.hasSelectedText())
        copy_act.triggered.connect(self.copy)

        paste_act = menu.addAction("붙여넣기")
        clipboard = QApplication.clipboard()
        paste_act.setEnabled(bool(clipboard and clipboard.text()))
        paste_act.triggered.connect(self.paste)

        delete_act = menu.addAction("삭제")
        delete_act.setEnabled(self.hasSelectedText())
        delete_act.triggered.connect(self.del_)

        menu.addSeparator()

        select_all_act = menu.addAction("전체선택")
        select_all_act.setEnabled(bool(self.text()))
        select_all_act.triggered.connect(self.selectAll)

        menu.exec(event.globalPos())


class KoreanTextBrowser(QTextBrowser):
    """한글 우클릭 메뉴 QTextBrowser"""

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        menu.setStyleSheet(_make_korean_menu_style())

        copy_act = menu.addAction("복사")
        copy_act.setEnabled(self.textCursor().hasSelection())
        copy_act.triggered.connect(self.copy)

        menu.addSeparator()

        select_all_act = menu.addAction("전체선택")
        select_all_act.triggered.connect(self.selectAll)

        menu.exec(event.globalPos())


class ChatPanel(QWidget):
    """채팅 패널"""
    send_requested = Signal(str, bool)  # (메시지, SC 사용여부)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # 응답 영역
        self.response_view = KoreanTextBrowser()
        self.response_view.setOpenExternalLinks(True)
        self.response_view.setStyleSheet(f"""
            QTextBrowser {{
                background-color: {COLORS['darker']};
                color: {COLORS['text']};
                border: 1px solid {COLORS['border']};
                border-radius: 12px;
                padding: 20px;
                font-size: 13px;
            }}
        """)
        self.response_view.setHtml(self._welcome_html())
        layout.addWidget(self.response_view, stretch=1)

        # SC 결과 뱃지 (숨김 상태로 시작)
        self.sc_badge = QLabel()
        self.sc_badge.setVisible(False)
        self.sc_badge.setStyleSheet(f"""
            background-color: {COLORS['surface']};
            color: {COLORS['text_dim']};
            border-radius: 6px;
            padding: 6px 12px;
            font-size: 11px;
        """)
        layout.addWidget(self.sc_badge)

        # 입력 영역 컨테이너
        input_container = QWidget()
        input_container.setStyleSheet(f"""
            QWidget {{
                background-color: {COLORS['surface']};
                border-radius: 12px;
                border: 1px solid {COLORS['border']};
            }}
        """)
        input_inner = QHBoxLayout(input_container)
        input_inner.setContentsMargins(6, 6, 6, 6)
        input_inner.setSpacing(8)

        self.input_field = KoreanLineEdit()
        self.input_field.setPlaceholderText("질문이나 요청을 입력하세요...")
        self.input_field.setFixedHeight(38)
        self.input_field.setStyleSheet(f"""
            QLineEdit {{
                background-color: transparent;
                color: {COLORS['text']};
                border: none;
                padding: 0 10px;
                font-size: 14px;
            }}
        """)
        self.input_field.returnPressed.connect(self._on_send)
        input_inner.addWidget(self.input_field, stretch=1)

        # SC 토글
        self.sc_check = QCheckBox("자기교정")
        self.sc_check.setToolTip("Self-Correction: 생성된 코드를 자동 검증")
        self.sc_check.setStyleSheet(f"""
            QCheckBox {{
                color: {COLORS['text_muted']};
                font-size: 11px;
                font-weight: 600;
                spacing: 4px;
            }}
            QCheckBox::indicator {{
                width: 14px;
                height: 14px;
                border: 2px solid {COLORS['border_light']};
                border-radius: 3px;
                background: {COLORS['darker']};
            }}
            QCheckBox::indicator:checked {{
                background: {COLORS['primary']};
                border-color: {COLORS['primary']};
            }}
        """)
        input_inner.addWidget(self.sc_check)

        # 전송 버튼
        self.send_btn = QPushButton("전송")
        self.send_btn.setObjectName("primaryBtn")
        self.send_btn.setFixedSize(70, 38)
        self.send_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['primary']};
                color: white;
                border: none;
                border-radius: 8px;
                font-weight: 600;
                font-size: 13px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['primary_hover']};
            }}
            QPushButton:disabled {{
                background-color: {COLORS['surface']};
                color: {COLORS['text_muted']};
            }}
        """)
        self.send_btn.clicked.connect(self._on_send)
        input_inner.addWidget(self.send_btn)

        layout.addWidget(input_container)

    def _on_send(self):
        text = self.input_field.text().strip()
        if text:
            use_sc = self.sc_check.isChecked()
            self.send_requested.emit(text, use_sc)
            self.input_field.clear()

    def show_response(self, result: dict):
        """LLM 응답 표시"""
        answer = result.get("answer", result.get("content", ""))
        html = self._markdown_to_html(answer)

        # SC 정보 표시
        if result.get("use_sc"):
            retry = result.get("retry_count", 0)
            is_valid = result.get("is_valid", False)
            badge_color = COLORS['green'] if is_valid else COLORS['red']
            badge_text = "통과" if is_valid else "실패"
            self.sc_badge.setText(f"자기교정: {badge_text} (시도 {retry}회)")
            self.sc_badge.setStyleSheet(f"""
                background-color: {badge_color}22;
                color: {badge_color};
                border: 1px solid {badge_color};
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 11px;
                font-weight: bold;
            """)
            self.sc_badge.setVisible(True)
        else:
            self.sc_badge.setVisible(False)

        self.response_view.setHtml(html)

    def show_loading(self, text: str = "처리 중..."):
        """로딩 표시"""
        self.response_view.setHtml(f"""
            <div style="text-align:center; padding:60px; color:{COLORS['text_muted']};">
                <p style="font-size:20px; color:{COLORS['primary']};">⏳</p>
                <p style="font-size:15px; margin-top:12px;">{text}</p>
                <p style="font-size:12px; margin-top:8px; color:{COLORS['text_muted']};">잠시만 기다려주세요...</p>
            </div>
        """)
        self.send_btn.setEnabled(False)
        self.input_field.setEnabled(False)

    def show_ready(self):
        """입력 가능 상태로 복원"""
        self.send_btn.setEnabled(True)
        self.input_field.setEnabled(True)
        self.input_field.setFocus()

    def show_error(self, error: str):
        """에러 표시"""
        self.response_view.setHtml(f"""
            <div style="padding:20px; background:{COLORS['red_dim']};
                        border:1px solid {COLORS['red']}; border-radius:10px; margin:10px;">
                <p style="color:{COLORS['red']}; font-weight:bold; font-size:14px;">⚠ 오류 발생</p>
                <p style="color:{COLORS['text']}; margin-top:8px; font-size:13px;">{error}</p>
            </div>
        """)

    def _markdown_to_html(self, text: str) -> str:
        """마크다운을 HTML로 변환"""
        try:
            import markdown
            html = markdown.markdown(
                text,
                extensions=['fenced_code', 'tables', 'nl2br']
            )
        except ImportError:
            html = text.replace('\n', '<br>')
            import re
            html = re.sub(
                r'```(\w*)\n(.*?)```',
                lambda m: f'<pre style="background:{COLORS["surface"]};padding:14px;border-radius:8px;'
                          f'overflow-x:auto;border:1px solid {COLORS["border"]};"><code>{m.group(2)}</code></pre>',
                html, flags=re.DOTALL
            )

        styled = f"""
        <style>
            body {{
                color: {COLORS['text']};
                font-family: "Segoe UI", "맑은 고딕", sans-serif;
                font-size: 13px;
                line-height: 1.7;
                margin: 0;
                padding: 0;
            }}
            pre {{
                background: {COLORS['darkest']};
                padding: 16px;
                border-radius: 10px;
                border: 1px solid {COLORS['border']};
                overflow-x: auto;
                font-family: "Consolas", "D2Coding", "Cascadia Code", monospace;
                font-size: 12px;
                line-height: 1.5;
                margin: 12px 0;
            }}
            code {{
                font-family: "Consolas", "D2Coding", "Cascadia Code", monospace;
                background: {COLORS['surface']};
                padding: 2px 6px;
                border-radius: 4px;
                font-size: 12px;
                color: {COLORS['peach']};
            }}
            pre code {{
                background: transparent;
                padding: 0;
                color: {COLORS['text']};
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 12px 0;
            }}
            th, td {{
                border: 1px solid {COLORS['border']};
                padding: 10px 12px;
                text-align: left;
            }}
            th {{
                background: {COLORS['surface']};
                font-weight: 600;
            }}
            h1, h2, h3, h4 {{
                color: {COLORS['primary_hover']};
                margin-top: 16px;
            }}
            h1 {{ font-size: 20px; }}
            h2 {{ font-size: 17px; }}
            h3 {{ font-size: 15px; }}
            a {{
                color: {COLORS['blue']};
                text-decoration: none;
            }}
            a:hover {{
                text-decoration: underline;
            }}
            blockquote {{
                border-left: 3px solid {COLORS['primary']};
                padding-left: 14px;
                color: {COLORS['text_dim']};
                margin: 12px 0;
            }}
            ul, ol {{
                padding-left: 24px;
            }}
            li {{
                margin: 4px 0;
            }}
            p {{
                margin: 8px 0;
            }}
        </style>
        {html}
        """
        return styled

    def _welcome_html(self) -> str:
        return f"""
        <div style="text-align:center; padding:80px 20px; color:{COLORS['text_muted']};">
            <p style="font-size:36px; color:{COLORS['primary']}; font-weight:bold;
                       letter-spacing:-1px;">Nomos LLM</p>
            <p style="font-size:14px; margin-top:8px; color:{COLORS['text_dim']};">
                코드 개발 / 수정 / 데이터 분석 도우미</p>
            <br><br>
            <div style="display:inline-block; text-align:left; background:{COLORS['surface']};
                        padding:20px 28px; border-radius:12px; border:1px solid {COLORS['border']};">
                <p style="font-size:12px; color:{COLORS['text_dim']}; margin:6px 0;">
                    💬 <b style="color:{COLORS['text']};">대화</b> — 일반 질문, 코드 설명</p>
                <p style="font-size:12px; color:{COLORS['text_dim']}; margin:6px 0;">
                    ⚡ <b style="color:{COLORS['text']};">코드 생성</b> — 새 코드 작성</p>
                <p style="font-size:12px; color:{COLORS['text_dim']}; margin:6px 0;">
                    🔧 <b style="color:{COLORS['text']};">코드 수정</b> — Aider 프로젝트 수정</p>
                <p style="font-size:12px; color:{COLORS['text_dim']}; margin:6px 0;">
                    📊 <b style="color:{COLORS['text']};">분석</b> — 데이터 분석</p>
            </div>
            <p style="font-size:11px; margin-top:20px; color:{COLORS['text_muted']};">
                왼쪽에서 모드를 선택하고 질문을 입력하세요</p>
        </div>
        """
