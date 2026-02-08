#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QThread 워커 - LLM/Aider 호출을 백그라운드에서 실행
"""

from PySide6.QtCore import QThread, Signal


class LLMWorker(QThread):
    """LLM 호출 워커 (자기교정 옵션 포함)"""
    finished = Signal(dict)
    progress = Signal(str)

    def __init__(self, llm_provider, prompt, system_prompt, use_sc=False, parent=None):
        super().__init__(parent)
        self.llm_provider = llm_provider
        self.prompt = prompt
        self.system_prompt = system_prompt
        self.use_sc = use_sc
        self._cancelled = False

    def run(self):
        try:
            if self.use_sc:
                self.progress.emit("자기교정 모드 실행 중...")
                from app.core.self_correction import run_self_correction
                result = run_self_correction(self.llm_provider, self.prompt, self.system_prompt)
                if result["success"]:
                    self.finished.emit({
                        "success": True,
                        "answer": result["answer"],
                        "use_sc": True,
                        "retry_count": result["retry_count"],
                        "is_valid": result["is_valid"],
                        "review": result.get("review", "")
                    })
                else:
                    self.finished.emit({"success": False, "answer": result.get("error", "오류 발생")})
            else:
                self.progress.emit("LLM 호출 중...")
                result = self.llm_provider.call(self.prompt, self.system_prompt)
                if result["success"]:
                    self.finished.emit({
                        "success": True,
                        "answer": result["content"],
                        "use_sc": False
                    })
                else:
                    self.finished.emit({"success": False, "answer": result.get("error", "오류 발생")})
        except Exception as e:
            self.finished.emit({"success": False, "answer": str(e)})

    def cancel(self):
        self._cancelled = True


class AiderWorker(QThread):
    """Aider 채팅 워커"""
    finished = Signal(dict)
    progress = Signal(str)

    def __init__(self, bridge, project_id, message, files=None, parent=None):
        super().__init__(parent)
        self.bridge = bridge
        self.project_id = project_id
        self.message = message
        self.files = files

    def run(self):
        try:
            self.progress.emit("코드 수정 중...")
            result = self.bridge.chat(self.project_id, self.message, self.files)
            self.finished.emit(result)
        except Exception as e:
            self.finished.emit({"success": False, "error": str(e)})


class ModelLoadWorker(QThread):
    """GGUF 모델 로딩 워커"""
    finished = Signal(bool, str)
    progress = Signal(str)

    def __init__(self, llm_provider, model_key=None, parent=None):
        super().__init__(parent)
        self.llm_provider = llm_provider
        self.model_key = model_key

    def run(self):
        try:
            if self.model_key:
                self.progress.emit(f"모델 전환 중: {self.model_key}...")
                success = self.llm_provider.switch_model(self.model_key)
                msg = "모델 전환 완료" if success else "모델 전환 실패"
            else:
                self.progress.emit("로컬 모델 로딩 중...")
                success = self.llm_provider.load_local_model()
                msg = "모델 로드 완료" if success else "모델 로드 실패"
            self.finished.emit(success, msg)
        except Exception as e:
            self.finished.emit(False, str(e))
