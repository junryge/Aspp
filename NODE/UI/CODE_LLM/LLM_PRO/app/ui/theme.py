#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
다크 테마 스타일시트 (모던 다크 UI)
"""

# 색상 팔레트
COLORS = {
    "primary": "#6366f1",
    "primary_hover": "#818cf8",
    "primary_dark": "#4f46e5",
    "primary_glow": "rgba(99, 102, 241, 0.15)",
    "dark": "#1a1b2e",
    "darker": "#13141f",
    "darkest": "#0d0e17",
    "surface": "#242538",
    "surface_hover": "#2d2e45",
    "overlay": "#3d3e58",
    "text": "#e2e4f0",
    "text_dim": "#a0a3bd",
    "text_muted": "#6b6e8a",
    "green": "#4ade80",
    "green_dim": "rgba(74, 222, 128, 0.12)",
    "red": "#f87171",
    "red_dim": "rgba(248, 113, 113, 0.12)",
    "yellow": "#fbbf24",
    "yellow_dim": "rgba(251, 191, 36, 0.12)",
    "blue": "#60a5fa",
    "teal": "#2dd4bf",
    "peach": "#fb923c",
    "border": "#2d2e45",
    "border_light": "#3d3e58",
}

DARK_THEME = f"""
/* ===== 전역 스타일 ===== */
QMainWindow, QWidget {{
    background-color: {COLORS['dark']};
    color: {COLORS['text']};
    font-family: "Segoe UI", "맑은 고딕", sans-serif;
    font-size: 13px;
}}

/* ===== 스플리터 ===== */
QSplitter::handle {{
    background-color: {COLORS['border']};
    width: 1px;
    height: 1px;
}}
QSplitter::handle:hover {{
    background-color: {COLORS['primary']};
}}

/* ===== 버튼 ===== */
QPushButton {{
    background-color: {COLORS['surface']};
    color: {COLORS['text_dim']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    padding: 6px 14px;
    font-size: 12px;
}}
QPushButton:hover {{
    background-color: {COLORS['surface_hover']};
    color: {COLORS['text']};
    border-color: {COLORS['border_light']};
}}
QPushButton:pressed {{
    background-color: {COLORS['primary_dark']};
    color: white;
}}
QPushButton:checked {{
    background-color: {COLORS['primary']};
    color: white;
    border-color: {COLORS['primary']};
}}

/* 강조 버튼 */
QPushButton#primaryBtn {{
    background-color: {COLORS['primary']};
    color: white;
    border: none;
    font-weight: 600;
    font-size: 13px;
}}
QPushButton#primaryBtn:hover {{
    background-color: {COLORS['primary_hover']};
}}
QPushButton#primaryBtn:pressed {{
    background-color: {COLORS['primary_dark']};
}}

/* ===== 입력 필드 ===== */
QLineEdit, QTextEdit, QPlainTextEdit {{
    background-color: {COLORS['darker']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    padding: 8px 12px;
    selection-background-color: {COLORS['primary']};
    selection-color: white;
}}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
    border-color: {COLORS['primary']};
    background-color: {COLORS['darkest']};
}}

/* ===== 콤보박스 ===== */
QComboBox {{
    background-color: {COLORS['surface']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    padding: 5px 10px;
    min-width: 80px;
}}
QComboBox:hover {{
    border-color: {COLORS['border_light']};
}}
QComboBox:focus {{
    border-color: {COLORS['primary']};
}}
QComboBox QAbstractItemView {{
    background-color: {COLORS['surface']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['border']};
    selection-background-color: {COLORS['primary']};
    selection-color: white;
    outline: none;
}}
QComboBox::drop-down {{
    border: none;
    width: 24px;
}}

/* ===== 스크롤바 ===== */
QScrollBar:vertical {{
    background: transparent;
    width: 8px;
    margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: {COLORS['overlay']};
    border-radius: 4px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{
    background: {COLORS['text_muted']};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: transparent;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 8px;
    margin: 2px;
}}
QScrollBar::handle:horizontal {{
    background: {COLORS['overlay']};
    border-radius: 4px;
    min-width: 24px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {COLORS['text_muted']};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
}}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
    background: transparent;
}}

/* ===== 트리 위젯 ===== */
QTreeWidget, QTreeView, QListView {{
    background-color: {COLORS['darker']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    outline: none;
    padding: 4px;
}}
QTreeWidget::item, QTreeView::item, QListView::item {{
    padding: 5px 4px;
    border-radius: 4px;
    margin: 1px 2px;
}}
QTreeWidget::item:selected, QTreeView::item:selected, QListView::item:selected {{
    background-color: {COLORS['primary']};
    color: white;
}}
QTreeWidget::item:hover, QTreeView::item:hover, QListView::item:hover {{
    background-color: {COLORS['surface']};
}}
QTreeView::branch {{
    background: transparent;
}}
QHeaderView::section {{
    background-color: {COLORS['surface']};
    color: {COLORS['text_muted']};
    border: none;
    border-bottom: 1px solid {COLORS['border']};
    padding: 6px 8px;
    font-size: 11px;
    font-weight: 600;
}}

/* ===== 라벨 ===== */
QLabel {{
    color: {COLORS['text']};
}}

/* ===== 체크박스 ===== */
QCheckBox {{
    color: {COLORS['text']};
    spacing: 6px;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 2px solid {COLORS['border_light']};
    border-radius: 4px;
    background: {COLORS['darker']};
}}
QCheckBox::indicator:hover {{
    border-color: {COLORS['primary']};
}}
QCheckBox::indicator:checked {{
    background: {COLORS['primary']};
    border-color: {COLORS['primary']};
}}

/* ===== 메뉴 ===== */
QMenu {{
    background-color: {COLORS['surface']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    padding: 6px;
}}
QMenu::item {{
    padding: 8px 20px;
    border-radius: 4px;
}}
QMenu::item:selected {{
    background-color: {COLORS['primary']};
    color: white;
}}
QMenu::separator {{
    height: 1px;
    background: {COLORS['border']};
    margin: 4px 8px;
}}

/* ===== 다이얼로그 ===== */
QDialog {{
    background-color: {COLORS['dark']};
}}

/* ===== 툴팁 ===== */
QToolTip {{
    background-color: {COLORS['surface']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 12px;
}}

/* ===== 텍스트 브라우저 (채팅) ===== */
QTextBrowser {{
    background-color: {COLORS['darker']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    padding: 12px;
}}

/* ===== 그룹박스 ===== */
QGroupBox {{
    color: {COLORS['text']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 12px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}}
"""
