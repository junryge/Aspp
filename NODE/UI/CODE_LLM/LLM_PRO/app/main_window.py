#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MainWindow - 전체 레이아웃 및 시그널 연결
"""

import logging
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QSplitter, QStackedWidget, QMessageBox
)
from PySide6.QtCore import Qt

from .ui.theme import DARK_THEME, COLORS
from .ui.sidebar import SidebarWidget
from .ui.header import HeaderWidget
from .ui.chat_panel import ChatPanel
from .ui.project_panel import ProjectPanel
from .ui.diff_viewer import DiffViewer
from .ui.dialogs import CreateProjectDialog
from .ui.workers import LLMWorker, AiderWorker, ModelLoadWorker

from .core.prompt_builder import SYSTEM_PROMPTS, build_prompt
from .aider import project_manager

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """메인 윈도우"""

    # 프로젝트 모드 (프로젝트 패널 표시)
    PROJECT_MODES = {"aider", "analysis"}

    def __init__(self, config, llm_provider, bridge, parent=None):
        super().__init__(parent)
        self.config = config
        self.llm_provider = llm_provider
        self.bridge = bridge
        self.current_mode = "general"
        self.current_worker = None

        self.setWindowTitle("Nomos LLM - 코딩 어시스턴트")
        self.setMinimumSize(1100, 700)
        self.setStyleSheet(DARK_THEME)

        self._setup_ui()
        self._connect_signals()
        self._update_status()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 사이드바
        self.sidebar = SidebarWidget()
        main_layout.addWidget(self.sidebar)

        # 오른쪽 영역
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        # 헤더
        self.header = HeaderWidget(self.config)
        right_layout.addWidget(self.header)

        # 콘텐츠 스택 (모드에 따라 전환)
        self.content_stack = QStackedWidget()
        self.content_stack.setStyleSheet(f"background-color: {COLORS['dark']};")

        # === 페이지 0: 채팅 전용 (general, generate 모드) ===
        self.chat_panel = ChatPanel()
        self.content_stack.addWidget(self.chat_panel)  # index 0

        # === 페이지 1: 채팅 + 프로젝트 (aider, analysis 모드) ===
        project_page = QWidget()
        project_layout = QHBoxLayout(project_page)
        project_layout.setContentsMargins(0, 0, 0, 0)
        project_layout.setSpacing(1)

        # 채팅 (좌측)
        self.project_chat = ChatPanel()

        # 프로젝트 + Diff (우측)
        right_panel = QWidget()
        right_panel_layout = QVBoxLayout(right_panel)
        right_panel_layout.setContentsMargins(0, 0, 0, 0)
        right_panel_layout.setSpacing(0)

        proj_diff_splitter = QSplitter(Qt.Vertical)
        self.project_panel = ProjectPanel()
        self.diff_viewer = DiffViewer()
        proj_diff_splitter.addWidget(self.project_panel)
        proj_diff_splitter.addWidget(self.diff_viewer)
        proj_diff_splitter.setSizes([400, 200])
        right_panel_layout.addWidget(proj_diff_splitter)

        project_splitter = QSplitter(Qt.Horizontal)
        project_splitter.addWidget(self.project_chat)
        project_splitter.addWidget(right_panel)
        project_splitter.setSizes([400, 500])

        project_layout.addWidget(project_splitter)
        self.content_stack.addWidget(project_page)  # index 1

        right_layout.addWidget(self.content_stack)
        main_layout.addWidget(right_widget, stretch=1)

    def _get_active_chat(self) -> ChatPanel:
        """현재 활성 채팅 패널 반환"""
        if self.current_mode in self.PROJECT_MODES:
            return self.project_chat
        return self.chat_panel

    def _connect_signals(self):
        # 사이드바 → 모드 전환
        self.sidebar.mode_changed.connect(self.set_mode)

        # 헤더 → 환경/모델 전환
        self.header.env_changed.connect(self._on_env_change)
        self.header.model_changed.connect(self._on_model_change)
        self.header.token_reload.connect(self._on_token_reload)

        # 채팅 → 전송 (두 채팅 패널 모두 연결)
        self.chat_panel.send_requested.connect(self._on_send)
        self.project_chat.send_requested.connect(self._on_send)

        # 프로젝트 패널
        self.project_panel.create_project.connect(self._on_create_project)
        self.project_panel.project_changed.connect(self._on_project_changed)
        self.project_panel.file_selected.connect(self._on_file_selected)

        # 프로젝트 패널 버튼들
        if hasattr(self.project_panel, 'undo_requested_btn'):
            self.project_panel.undo_requested_btn.clicked.connect(self._on_undo)
        if hasattr(self.project_panel, 'diff_requested_btn'):
            self.project_panel.diff_requested_btn.clicked.connect(self._on_show_diff)
        if hasattr(self.project_panel, 'history_requested_btn'):
            self.project_panel.history_requested_btn.clicked.connect(self._on_show_history)

        # Diff 뷰어 → 승인/거절
        self.diff_viewer.approve_clicked.connect(self._on_approve)
        self.diff_viewer.reject_clicked.connect(self._on_reject)

    def set_mode(self, mode: str):
        """모드 전환"""
        self.current_mode = mode
        self.sidebar.set_mode(mode)

        if mode in self.PROJECT_MODES:
            self.content_stack.setCurrentIndex(1)
            self._refresh_projects()
        else:
            self.content_stack.setCurrentIndex(0)

        # 입력 플레이스홀더 업데이트
        placeholders = {
            "general": "질문이나 요청을 입력하세요...",
            "generate": "생성할 코드를 설명하세요 (예: Python으로 웹 크롤러 만들어줘)...",
            "aider": "코드 수정 요청을 입력하세요...",
            "analysis": "데이터 분석 요청을 입력하세요...",
        }
        ph = placeholders.get(mode, "")
        self.chat_panel.input_field.setPlaceholderText(ph)
        self.project_chat.input_field.setPlaceholderText(ph)

    # ===== 전송 처리 =====

    def _on_send(self, message: str, use_sc: bool):
        """메시지 전송 처리"""
        if self.current_mode in self.PROJECT_MODES:
            self._on_aider_send(message)
        else:
            self._on_chat_send(message, use_sc)

    def _on_chat_send(self, message: str, use_sc: bool):
        """일반/생성 모드 전송"""
        mode = self.current_mode
        if not message.strip():
            self.chat_panel.show_error("메시지를 입력해주세요!")
            return

        prompt = build_prompt(mode, message, "", "python")
        system_prompt = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["general"])

        self.chat_panel.show_loading("LLM 처리 중...")
        self.header.set_busy(True)

        self.current_worker = LLMWorker(
            self.llm_provider, prompt, system_prompt, use_sc
        )
        self.current_worker.finished.connect(self._on_llm_result)
        self.current_worker.progress.connect(self._on_progress)
        self.current_worker.start()

    def _on_aider_send(self, message: str):
        """Aider 모드 전송"""
        chat = self._get_active_chat()
        project_id = self.project_panel.current_project_id
        if not project_id:
            chat.show_error("프로젝트를 먼저 선택하세요!")
            return

        selected = self._get_selected_files()

        chat.show_loading("Aider 코드 수정 중...")
        self.header.set_busy(True)

        self.current_worker = AiderWorker(
            self.bridge, project_id, message, selected
        )
        self.current_worker.finished.connect(self._on_aider_result)
        self.current_worker.progress.connect(self._on_progress)
        self.current_worker.start()

    # ===== 결과 처리 =====

    def _on_llm_result(self, result: dict):
        """LLM 결과 처리"""
        self.header.set_busy(False)
        chat = self._get_active_chat()
        chat.show_ready()

        if result.get("success"):
            chat.show_response(result)
        else:
            chat.show_error(result.get("answer", "알 수 없는 오류"))

    def _on_aider_result(self, result: dict):
        """Aider 결과 처리"""
        self.header.set_busy(False)
        chat = self._get_active_chat()
        chat.show_ready()

        if result.get("success"):
            chat.show_response({
                "success": True,
                "answer": result.get("response", "완료"),
                "use_sc": False
            })

            # Diff 표시
            diff = result.get("diff", "")
            proposal_id = result.get("proposal_id")
            self.diff_viewer.show_diff(diff, proposal_id)

            # 파일 트리 새로고침
            if self.project_panel.current_project_id:
                self._on_project_changed(self.project_panel.current_project_id)
        else:
            chat.show_error(result.get("error", "Aider 오류"))

    def _on_progress(self, text: str):
        """진행 상태 업데이트"""
        self.header.update_status(text)

    # ===== 환경/모델 전환 =====

    def _on_env_change(self, env_key: str):
        """환경 전환"""
        if env_key == "local":
            if self.llm_provider.local_llm is None:
                self.header.set_busy(True)
                worker = ModelLoadWorker(self.llm_provider)
                worker.finished.connect(self._on_model_loaded)
                worker.start()
                self._model_load_worker = worker
            self.config.set_env("local")
        else:
            if not self.config.api_token:
                QMessageBox.warning(self, "토큰 없음", "API 토큰이 없습니다.\ntoken.txt를 배치해주세요.")
                return
            self.config.set_env(env_key)

        self.header.update_env(self.config.env_mode)
        self._update_status()

    def _on_model_change(self, model_key: str):
        """GGUF 모델 전환"""
        self.header.set_busy(True)
        worker = ModelLoadWorker(self.llm_provider, model_key)
        worker.finished.connect(self._on_model_loaded)
        worker.start()
        self._model_load_worker = worker

    def _on_model_loaded(self, success: bool, message: str):
        """모델 로드 완료"""
        self.header.set_busy(False)
        if success:
            self._update_status()
        else:
            QMessageBox.warning(self, "모델 로드 실패", message)

    def _on_token_reload(self):
        """토큰 새로고침"""
        if self.config.load_token():
            QMessageBox.information(self, "토큰", "토큰 로드 성공")
            self._update_status()
        else:
            QMessageBox.warning(self, "토큰", "토큰 로드 실패\ntoken.txt를 확인하세요.")

    # ===== 프로젝트 관리 =====

    def _on_create_project(self):
        """프로젝트 생성"""
        dialog = CreateProjectDialog(self)
        if dialog.exec() == CreateProjectDialog.Accepted:
            data = dialog.result_data
            project = project_manager.create_project(
                name=data["name"],
                customer_type=data["customer_type"],
                description=data["description"],
                path=data["path"]
            )
            self._refresh_projects()
            chat = self._get_active_chat()
            chat.show_response({
                "success": True,
                "answer": f"프로젝트 '{project['name']}' 생성 완료!\n경로: {project['path']}",
                "use_sc": False
            })

    def _refresh_projects(self):
        """프로젝트 목록 새로고침"""
        projects = project_manager.list_projects()
        self.project_panel.update_projects(projects)

    def _on_project_changed(self, project_id: str):
        """프로젝트 선택 변경"""
        files = project_manager.get_project_files(project_id)
        self.project_panel.update_files(files)

    def _on_file_selected(self, project_id: str, file_path: str):
        """파일 선택"""
        content = project_manager.read_file(project_id, file_path)
        if content is not None:
            self.project_panel.file_viewer.set_code(content)
            ext_map = {
                ".py": "python", ".js": "javascript", ".ts": "typescript",
                ".java": "java", ".cpp": "cpp", ".c": "c", ".go": "go",
                ".rs": "rust", ".sql": "sql", ".html": "html", ".css": "css",
                ".sh": "bash", ".json": "json", ".yaml": "yaml", ".yml": "yaml"
            }
            ext = "." + file_path.split(".")[-1] if "." in file_path else ""
            lang = ext_map.get(ext, "python")
            self.project_panel.file_viewer.set_language(lang)

    def _get_selected_files(self):
        """프로젝트 패널에서 체크된 파일 목록"""
        checked = self.project_panel.get_checked_files()
        return checked if checked else None  # 빈 리스트이면 None (전체 파일)

    # ===== Git 작업 =====

    def _on_approve(self, proposal_id: str):
        """변경 승인"""
        chat = self._get_active_chat()
        result = self.bridge.approve_proposal(proposal_id)
        if result.get("success"):
            self.diff_viewer.clear()
            chat.show_response({
                "success": True,
                "answer": "변경이 승인되어 적용되었습니다!",
                "use_sc": False
            })
            if self.project_panel.current_project_id:
                self._on_project_changed(self.project_panel.current_project_id)
        else:
            chat.show_error(result.get("error", "승인 실패"))

    def _on_reject(self, proposal_id: str):
        """변경 거절"""
        chat = self._get_active_chat()
        result = self.bridge.reject_proposal(proposal_id)
        if result.get("success"):
            self.diff_viewer.clear()
            chat.show_response({
                "success": True,
                "answer": "변경이 거절되었습니다. 기존 코드가 유지됩니다.",
                "use_sc": False
            })

    def _on_undo(self):
        """마지막 변경 취소"""
        if not self.project_panel.current_project_id:
            return
        chat = self._get_active_chat()
        from .aider.git_ops import undo
        result = undo(self.project_panel.current_project_id)
        if result.get("success"):
            chat.show_response({
                "success": True,
                "answer": f"변경 취소됨: {result.get('reverted', '')}",
                "use_sc": False
            })
            self._on_project_changed(self.project_panel.current_project_id)

    def _on_show_diff(self):
        """Diff 보기"""
        if not self.project_panel.current_project_id:
            return
        from .aider.git_ops import get_diff
        result = get_diff(self.project_panel.current_project_id)
        if result.get("success"):
            self.diff_viewer.show_diff(result.get("diff", ""))

    def _on_show_history(self):
        """히스토리 보기"""
        if not self.project_panel.current_project_id:
            return
        chat = self._get_active_chat()
        from .aider.git_ops import get_history
        result = get_history(self.project_panel.current_project_id)
        if result.get("success"):
            commits = result.get("commits", [])
            if commits:
                history_text = "## Git 히스토리\n\n"
                for c in commits:
                    history_text += f"- **{c['short_hash']}** {c['message']} ({c['date'][:10]})\n"
                chat.show_response({
                    "success": True,
                    "answer": history_text,
                    "use_sc": False
                })

    # ===== 상태 =====

    def _update_status(self):
        """상태바 업데이트"""
        env_name = self.config.ENV_CONFIG.get(self.config.env_mode, {}).get("name", "")
        mode = self.config.llm_mode.upper()
        token_status = "토큰 OK" if self.config.api_token else "토큰 없음"
        local_status = "GGUF OK" if self.llm_provider.local_llm else "GGUF -"

        self.header.update_status(f"{env_name} | {mode} | {token_status} | {local_status}")
        self.header.update_env(self.config.env_mode)
