#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
프로젝트 패널 - 파일 트리 (체크박스 + 초록 선택) + 프로젝트 관리
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem,
    QComboBox, QPushButton, QSplitter, QMenu, QInputDialog, QFileDialog,
    QMessageBox, QLabel
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QAction, QColor, QBrush
from .theme import COLORS
from .code_editor import CodeEditorWidget


class ProjectPanel(QWidget):
    """프로젝트 관리 패널"""
    project_changed = Signal(str)      # project_id
    file_selected = Signal(str, str)   # project_id, file_path
    create_project = Signal()          # 새 프로젝트 생성 요청

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_project_id = None
        self.projects = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 4)
        layout.setSpacing(8)

        # 프로젝트 선택 바
        proj_bar = QHBoxLayout()
        proj_bar.setSpacing(8)

        proj_label = QLabel("프로젝트")
        proj_label.setStyleSheet(f"""
            font-size: 11px;
            font-weight: 700;
            color: {COLORS['text_muted']};
            letter-spacing: 1px;
        """)
        proj_bar.addWidget(proj_label)

        self.project_combo = QComboBox()
        self.project_combo.setMinimumWidth(160)
        self.project_combo.currentIndexChanged.connect(self._on_project_change)
        proj_bar.addWidget(self.project_combo, stretch=1)

        new_btn = QPushButton("+ 새 프로젝트")
        new_btn.setFixedHeight(30)
        new_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['primary']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 0 12px;
                font-weight: 600;
                font-size: 11px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['primary_hover']};
            }}
        """)
        new_btn.clicked.connect(self.create_project.emit)
        proj_bar.addWidget(new_btn)

        layout.addLayout(proj_bar)

        # 스플리터: 파일 트리 + 파일 뷰어
        splitter = QSplitter(Qt.Horizontal)

        # 왼쪽: 파일 트리
        tree_widget = QWidget()
        tree_layout = QVBoxLayout(tree_widget)
        tree_layout.setContentsMargins(0, 0, 0, 0)
        tree_layout.setSpacing(6)

        # 선택된 파일 표시
        self.selected_label = QLabel("선택된 파일: 0개")
        self.selected_label.setStyleSheet(f"""
            font-size: 11px;
            color: {COLORS['green']};
            font-weight: 600;
            padding: 2px 4px;
        """)
        tree_layout.addWidget(self.selected_label)

        # 파일 작업 버튼
        file_bar = QHBoxLayout()
        file_bar.setSpacing(4)
        for label, tooltip, slot_name in [
            ("되돌리기", "마지막 변경 취소", "undo_requested"),
            ("변경사항", "변경사항 보기", "diff_requested"),
            ("히스토리", "커밋 히스토리", "history_requested"),
        ]:
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
            setattr(self, f"{slot_name}_btn", btn)
            file_bar.addWidget(btn)

        # 전체 선택/해제
        select_all_btn = QPushButton("전체선택")
        select_all_btn.setFixedHeight(26)
        select_all_btn.setToolTip("모든 파일 선택/해제")
        select_all_btn.setStyleSheet(f"""
            QPushButton {{
                padding: 2px 10px;
                border-radius: 5px;
                font-size: 11px;
                border: 1px solid {COLORS['green']};
                background: {COLORS['green_dim']};
                color: {COLORS['green']};
                font-weight: 600;
            }}
            QPushButton:hover {{
                background: {COLORS['green']};
                color: {COLORS['darkest']};
            }}
        """)
        select_all_btn.clicked.connect(self._toggle_select_all)
        file_bar.addWidget(select_all_btn)

        file_bar.addStretch()
        tree_layout.addLayout(file_bar)

        # 파일 트리 (체크박스 포함)
        self.file_tree = QTreeWidget()
        self.file_tree.setHeaderLabels(["파일명", "크기"])
        self.file_tree.setColumnWidth(0, 200)
        self.file_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.file_tree.customContextMenuRequested.connect(self._show_context_menu)
        self.file_tree.itemClicked.connect(self._on_file_click)
        self.file_tree.itemChanged.connect(self._on_item_check_changed)
        tree_layout.addWidget(self.file_tree)

        splitter.addWidget(tree_widget)

        # 오른쪽: 파일 뷰어
        self.file_viewer = CodeEditorWidget()
        splitter.addWidget(self.file_viewer)

        splitter.setSizes([250, 450])
        layout.addWidget(splitter)

    def update_projects(self, projects: list):
        """프로젝트 목록 업데이트"""
        self.projects = projects
        self.project_combo.blockSignals(True)
        self.project_combo.clear()
        for p in projects:
            type_labels = {"code_editing": "코드편집", "data_analysis": "데이터분석"}
            type_label = type_labels.get(p['customer_type'], p['customer_type'])
            self.project_combo.addItem(
                f"{p['name']} ({type_label})",
                p["id"]
            )
        self.project_combo.blockSignals(False)

        if projects:
            self.current_project_id = projects[0]["id"]
            self.project_changed.emit(self.current_project_id)

    def update_files(self, files: list):
        """파일 트리 업데이트 (체크박스 포함)"""
        self.file_tree.blockSignals(True)
        self.file_tree.clear()
        folders = {}

        for f in files:
            path = f["path"]
            parts = path.split("/")

            if len(parts) > 1:
                folder = "/".join(parts[:-1])
                if folder not in folders:
                    folder_item = QTreeWidgetItem(["📁 " + folder, ""])
                    folder_item.setData(0, Qt.UserRole, {"type": "folder", "path": folder})
                    folder_item.setFlags(folder_item.flags() | Qt.ItemIsUserCheckable)
                    folder_item.setCheckState(0, Qt.Unchecked)
                    self.file_tree.addTopLevelItem(folder_item)
                    folders[folder] = folder_item

                parent = folders[folder]
                file_item = QTreeWidgetItem(["📄 " + parts[-1], self._format_size(f["size"])])
                file_item.setData(0, Qt.UserRole, {"type": "file", "path": path})
                file_item.setFlags(file_item.flags() | Qt.ItemIsUserCheckable)
                file_item.setCheckState(0, Qt.Unchecked)
                parent.addChild(file_item)
            else:
                file_item = QTreeWidgetItem(["📄 " + path, self._format_size(f["size"])])
                file_item.setData(0, Qt.UserRole, {"type": "file", "path": path})
                file_item.setFlags(file_item.flags() | Qt.ItemIsUserCheckable)
                file_item.setCheckState(0, Qt.Unchecked)
                self.file_tree.addTopLevelItem(file_item)

        self.file_tree.expandAll()
        self.file_tree.blockSignals(False)
        self._update_selected_count()

    def get_checked_files(self) -> list:
        """체크된 파일 경로 목록 반환"""
        checked = []
        self._collect_checked(self.file_tree.invisibleRootItem(), checked)
        return checked

    def _collect_checked(self, parent_item, result: list):
        """재귀적으로 체크된 파일 수집"""
        for i in range(parent_item.childCount()):
            child = parent_item.child(i)
            data = child.data(0, Qt.UserRole)
            if data and data["type"] == "file" and child.checkState(0) == Qt.Checked:
                result.append(data["path"])
            self._collect_checked(child, result)

    def _on_item_check_changed(self, item, column):
        """체크 상태 변경 시 색상 업데이트"""
        data = item.data(0, Qt.UserRole)
        if not data:
            return

        # 폴더 체크 시 하위 파일도 같이 체크
        if data["type"] == "folder":
            state = item.checkState(0)
            self.file_tree.blockSignals(True)
            for i in range(item.childCount()):
                child = item.child(i)
                child.setCheckState(0, state)
                self._update_item_color(child)
            self.file_tree.blockSignals(False)

        self._update_item_color(item)
        self._update_selected_count()

    def _update_item_color(self, item):
        """체크 여부에 따라 색상 변경 (초록색 강조)"""
        is_checked = item.checkState(0) == Qt.Checked
        if is_checked:
            item.setForeground(0, QBrush(QColor(COLORS['green'])))
            item.setForeground(1, QBrush(QColor(COLORS['green'])))
            item.setBackground(0, QBrush(QColor(74, 222, 128, 20)))
            item.setBackground(1, QBrush(QColor(74, 222, 128, 20)))
        else:
            item.setForeground(0, QBrush(QColor(COLORS['text'])))
            item.setForeground(1, QBrush(QColor(COLORS['text_dim'])))
            item.setBackground(0, QBrush(QColor(0, 0, 0, 0)))
            item.setBackground(1, QBrush(QColor(0, 0, 0, 0)))

    def _update_selected_count(self):
        """선택된 파일 수 업데이트"""
        checked = self.get_checked_files()
        count = len(checked)
        if count > 0:
            self.selected_label.setText(f"✅ 선택된 파일: {count}개")
            self.selected_label.setStyleSheet(f"""
                font-size: 11px;
                color: {COLORS['green']};
                font-weight: 600;
                padding: 2px 4px;
            """)
        else:
            self.selected_label.setText("파일을 체크하여 선택하세요")
            self.selected_label.setStyleSheet(f"""
                font-size: 11px;
                color: {COLORS['text_muted']};
                font-weight: normal;
                padding: 2px 4px;
            """)

    def _toggle_select_all(self):
        """전체 선택/해제 토글"""
        checked = self.get_checked_files()
        root = self.file_tree.invisibleRootItem()
        total_files = self._count_files(root)

        # 전체 선택되어있으면 해제, 아니면 선택
        new_state = Qt.Unchecked if len(checked) == total_files and total_files > 0 else Qt.Checked

        self.file_tree.blockSignals(True)
        self._set_all_check_state(root, new_state)
        self.file_tree.blockSignals(False)
        self._update_selected_count()

    def _count_files(self, parent_item) -> int:
        count = 0
        for i in range(parent_item.childCount()):
            child = parent_item.child(i)
            data = child.data(0, Qt.UserRole)
            if data and data["type"] == "file":
                count += 1
            count += self._count_files(child)
        return count

    def _set_all_check_state(self, parent_item, state):
        for i in range(parent_item.childCount()):
            child = parent_item.child(i)
            child.setCheckState(0, state)
            self._update_item_color(child)
            self._set_all_check_state(child, state)

    def _on_project_change(self, index):
        project_id = self.project_combo.itemData(index)
        if project_id:
            self.current_project_id = project_id
            self.project_changed.emit(project_id)

    def _on_file_click(self, item, column):
        data = item.data(0, Qt.UserRole)
        if data and data["type"] == "file" and self.current_project_id:
            self.file_selected.emit(self.current_project_id, data["path"])

    def _show_context_menu(self, pos):
        item = self.file_tree.itemAt(pos)
        menu = QMenu(self)

        new_file_action = QAction("새 파일", self)
        new_file_action.triggered.connect(self._new_file)
        menu.addAction(new_file_action)

        new_folder_action = QAction("새 폴더", self)
        new_folder_action.triggered.connect(self._new_folder)
        menu.addAction(new_folder_action)

        if item:
            data = item.data(0, Qt.UserRole)
            if data:
                menu.addSeparator()
                rename_action = QAction("이름 변경", self)
                rename_action.triggered.connect(lambda: self._rename(data["path"]))
                menu.addAction(rename_action)

                delete_action = QAction("삭제", self)
                delete_action.triggered.connect(lambda: self._delete(data["path"]))
                menu.addAction(delete_action)

        menu.exec(self.file_tree.viewport().mapToGlobal(pos))

    def _new_file(self):
        name, ok = QInputDialog.getText(self, "새 파일", "파일명:")
        if ok and name and self.current_project_id:
            from app.aider.project_manager import create_new_file
            result = create_new_file(self.current_project_id, name)
            if result["success"]:
                self.project_changed.emit(self.current_project_id)

    def _new_folder(self):
        name, ok = QInputDialog.getText(self, "새 폴더", "폴더명:")
        if ok and name and self.current_project_id:
            from app.aider.project_manager import create_folder
            result = create_folder(self.current_project_id, name)
            if result["success"]:
                self.project_changed.emit(self.current_project_id)

    def _rename(self, path):
        new_name, ok = QInputDialog.getText(self, "이름 변경", "새 이름:", text=path.split("/")[-1])
        if ok and new_name and self.current_project_id:
            from app.aider.project_manager import rename_file
            result = rename_file(self.current_project_id, path, new_name)
            if result["success"]:
                self.project_changed.emit(self.current_project_id)

    def _delete(self, path):
        reply = QMessageBox.question(
            self, "삭제 확인",
            f"'{path}'를 삭제하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes and self.current_project_id:
            from app.aider.project_manager import delete_file
            result = delete_file(self.current_project_id, path)
            if result["success"]:
                self.project_changed.emit(self.current_project_id)

    @staticmethod
    def _format_size(size: int) -> str:
        if size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size / 1024:.1f} KB"
        else:
            return f"{size / (1024 * 1024):.1f} MB"
