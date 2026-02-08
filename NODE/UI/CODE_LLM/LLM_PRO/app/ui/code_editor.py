#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
코드 에디터 - Pygments 구문 강조 + 줄번호
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton,
    QLabel, QPlainTextEdit, QTextEdit, QMenu
)
from PySide6.QtCore import Signal, Qt, QRect, QSize
from PySide6.QtGui import (
    QFont, QColor, QPainter, QTextFormat, QSyntaxHighlighter,
    QTextCharFormat, QAction
)
from .theme import COLORS

# Pygments 구문 강조
try:
    from pygments import lex
    from pygments.lexers import get_lexer_by_name, TextLexer
    from pygments.token import (
        Token, Keyword, Name, Comment, String, Error, Number,
        Operator, Punctuation, Literal
    )
    HAS_PYGMENTS = True
except ImportError:
    HAS_PYGMENTS = False

# 다크 테마용 토큰 → 색상 맵
TOKEN_COLORS = {
    Token.Keyword:              "#c678dd",  # 보라 (if, def, class, import)
    Token.Keyword.Constant:     "#d19a66",  # 주황 (True, False, None)
    Token.Keyword.Namespace:    "#c678dd",  # 보라 (import, from)
    Token.Keyword.Type:         "#e5c07b",  # 노랑 (int, str, list)
    Token.Name.Builtin:         "#61afef",  # 파랑 (print, len, range)
    Token.Name.Function:        "#61afef",  # 파랑 (함수명)
    Token.Name.Class:           "#e5c07b",  # 노랑 (클래스명)
    Token.Name.Decorator:       "#c678dd",  # 보라 (@decorator)
    Token.Name.Exception:       "#e06c75",  # 빨강 (Exception)
    Token.Comment:              "#5c6370",  # 회색 (주석)
    Token.Comment.Single:       "#5c6370",
    Token.Comment.Multiline:    "#5c6370",
    Token.String:               "#98c379",  # 초록 (문자열)
    Token.String.Doc:           "#98c379",
    Token.String.Interpol:      "#d19a66",
    Token.String.Escape:        "#d19a66",
    Token.Number:               "#d19a66",  # 주황 (숫자)
    Token.Number.Integer:       "#d19a66",
    Token.Number.Float:         "#d19a66",
    Token.Operator:             "#56b6c2",  # 청록 (연산자)
    Token.Operator.Word:        "#c678dd",  # 보라 (and, or, not)
    Token.Punctuation:          "#abb2bf",  # 밝은회색 (괄호, 콤마)
    Token.Literal:              "#d19a66",
    Token.Name.Builtin.Pseudo:  "#e5c07b",  # self, cls
}


def _get_token_color(token_type):
    """토큰 타입에 맞는 색상 찾기 (상위 토큰까지 탐색)"""
    while token_type:
        if token_type in TOKEN_COLORS:
            return TOKEN_COLORS[token_type]
        token_type = token_type.parent
    return None


class PygmentsHighlighter(QSyntaxHighlighter):
    """Pygments 기반 구문 강조기"""

    def __init__(self, parent, language="python"):
        super().__init__(parent)
        self._language = language
        self._lexer = self._get_lexer(language)

    def _get_lexer(self, language):
        if not HAS_PYGMENTS:
            return None
        try:
            return get_lexer_by_name(language, stripnl=False, ensurenl=False)
        except Exception:
            return TextLexer(stripnl=False, ensurenl=False)

    def set_language(self, language):
        self._language = language
        self._lexer = self._get_lexer(language)
        self.rehighlight()

    def highlightBlock(self, text):
        if not HAS_PYGMENTS or not self._lexer:
            return

        index = 0
        for token_type, value in lex(text, self._lexer):
            length = len(value)
            color = _get_token_color(token_type)
            if color:
                fmt = QTextCharFormat()
                fmt.setForeground(QColor(color))
                # 키워드/클래스는 볼드
                if token_type in Token.Keyword or token_type in Token.Name.Class:
                    fmt.setFontWeight(QFont.Bold)
                # 주석은 이탤릭
                if token_type in Token.Comment:
                    fmt.setFontItalic(True)
                self.setFormat(index, length, fmt)
            index += length


class LineNumberArea(QWidget):
    """줄번호 영역"""
    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor

    def sizeHint(self):
        return QSize(self.editor.line_number_area_width(), 0)

    def paintEvent(self, event):
        self.editor.line_number_area_paint(event)


class CodeTextEdit(QPlainTextEdit):
    """줄번호 + 현재줄 하이라이트 에디터"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.line_number_area = LineNumberArea(self)

        self.blockCountChanged.connect(self._update_line_number_width)
        self.updateRequest.connect(self._update_line_number_area)
        self.cursorPositionChanged.connect(self._highlight_current_line)

        self._update_line_number_width(0)
        self._highlight_current_line()

    def line_number_area_width(self):
        digits = max(1, len(str(self.blockCount())))
        return 12 + self.fontMetrics().horizontalAdvance('9') * digits

    def _update_line_number_width(self, _):
        self.setViewportMargins(self.line_number_area_width(), 0, 0, 0)

    def _update_line_number_area(self, rect, dy):
        if dy:
            self.line_number_area.scroll(0, dy)
        else:
            self.line_number_area.update(0, rect.y(), self.line_number_area.width(), rect.height())
        if rect.contains(self.viewport().rect()):
            self._update_line_number_width(0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        cr = self.contentsRect()
        self.line_number_area.setGeometry(
            QRect(cr.left(), cr.top(), self.line_number_area_width(), cr.height())
        )

    def line_number_area_paint(self, event):
        painter = QPainter(self.line_number_area)
        painter.fillRect(event.rect(), QColor(COLORS['darker']))

        block = self.firstVisibleBlock()
        block_number = block.blockNumber()
        top = round(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        bottom = top + round(self.blockBoundingRect(block).height())

        current_line = self.textCursor().blockNumber()

        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                number = str(block_number + 1)
                if block_number == current_line:
                    painter.setPen(QColor(COLORS['text']))
                else:
                    painter.setPen(QColor(COLORS['text_muted']))
                painter.drawText(
                    0, top, self.line_number_area.width() - 6,
                    self.fontMetrics().height(),
                    Qt.AlignRight, number
                )
            block = block.next()
            top = bottom
            bottom = top + round(self.blockBoundingRect(block).height())
            block_number += 1

        painter.end()

    def _highlight_current_line(self):
        selections = []
        if not self.isReadOnly():
            selection = QTextEdit.ExtraSelection()
            selection.format.setBackground(QColor(COLORS['surface']))
            selection.format.setProperty(QTextFormat.FullWidthSelection, True)
            selection.cursor = self.textCursor()
            selection.cursor.clearSelection()
            selections.append(selection)
        self.setExtraSelections(selections)

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        menu.setStyleSheet(f"""
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
        """)

        undo_act = menu.addAction("되돌리기")
        undo_act.setEnabled(self.document().isUndoAvailable())
        undo_act.triggered.connect(self.undo)

        redo_act = menu.addAction("다시실행")
        redo_act.setEnabled(self.document().isRedoAvailable())
        redo_act.triggered.connect(self.redo)

        menu.addSeparator()

        cut_act = menu.addAction("잘라내기")
        cut_act.setEnabled(self.textCursor().hasSelection())
        cut_act.triggered.connect(self.cut)

        copy_act = menu.addAction("복사")
        copy_act.setEnabled(self.textCursor().hasSelection())
        copy_act.triggered.connect(self.copy)

        paste_act = menu.addAction("붙여넣기")
        paste_act.setEnabled(self.canPaste())
        paste_act.triggered.connect(self.paste)

        delete_act = menu.addAction("삭제")
        delete_act.setEnabled(self.textCursor().hasSelection())
        delete_act.triggered.connect(lambda: self.textCursor().removeSelectedText())

        menu.addSeparator()

        select_all_act = menu.addAction("전체선택")
        select_all_act.triggered.connect(self.selectAll)

        menu.exec(event.globalPos())


class CodeEditorWidget(QWidget):
    """코드 에디터 위젯 (Pygments 구문 강조 + 줄번호)"""
    quick_action = Signal(str)

    LANGUAGES = [
        "python", "javascript", "typescript", "java", "cpp", "c",
        "go", "rust", "sql", "html", "css", "bash", "json", "yaml"
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._highlighter = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # 도구바
        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)

        lang_label = QLabel("언어:")
        lang_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")
        toolbar.addWidget(lang_label)

        self.lang_combo = QComboBox()
        self.lang_combo.setFixedWidth(120)
        for lang in self.LANGUAGES:
            self.lang_combo.addItem(lang)
        self.lang_combo.currentTextChanged.connect(self._on_lang_change)
        toolbar.addWidget(self.lang_combo)

        toolbar.addStretch()

        quick_actions = [
            ("주석 추가", "add_comments", "코드에 주석 추가"),
            ("이름 개선", "improve_names", "변수/함수명 개선"),
            ("최적화", "optimize", "코드 최적화"),
            ("타입 힌트", "type_hints", "타입 힌트 추가"),
        ]
        for label, action, tooltip in quick_actions:
            btn = QPushButton(label)
            btn.setToolTip(tooltip)
            btn.setFixedHeight(26)
            btn.setStyleSheet(f"""
                QPushButton {{
                    padding: 2px 10px;
                    border-radius: 5px;
                    font-size: 11px;
                    border: 1px solid {COLORS['border']};
                    background: {COLORS['surface']};
                    color: {COLORS['text_muted']};
                }}
                QPushButton:hover {{
                    border-color: {COLORS['primary']};
                    color: {COLORS['text']};
                    background: {COLORS['surface_hover']};
                }}
            """)
            btn.clicked.connect(lambda checked, a=action: self.quick_action.emit(a))
            toolbar.addWidget(btn)

        layout.addLayout(toolbar)

        # 에디터 (줄번호 포함)
        self.editor = CodeTextEdit()
        font = QFont("Consolas", 12)
        font.setStyleHint(QFont.Monospace)
        self.editor.setFont(font)
        self.editor.setStyleSheet(f"""
            QPlainTextEdit {{
                background-color: {COLORS['darkest']};
                color: {COLORS['text']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                padding: 4px;
                selection-background-color: {COLORS['primary']};
                selection-color: white;
            }}
        """)
        self.editor.setTabStopDistance(40)
        self.editor.setLineWrapMode(QPlainTextEdit.NoWrap)

        # Pygments 구문 강조 연결
        if HAS_PYGMENTS:
            self._highlighter = PygmentsHighlighter(self.editor.document(), "python")

        layout.addWidget(self.editor)

    def _on_lang_change(self, language):
        if self._highlighter:
            self._highlighter.set_language(language)

    def get_code(self) -> str:
        return self.editor.toPlainText()

    def set_code(self, code: str):
        self.editor.setPlainText(code)

    def get_language(self) -> str:
        return self.lang_combo.currentText()

    def set_language(self, language: str):
        index = self.lang_combo.findText(language)
        if index >= 0:
            self.lang_combo.setCurrentIndex(index)
