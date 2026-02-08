#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AiderBridge - aider-chat 통합 핵심 클래스
API 모드: aider Python API 사용
GGUF 모드: 직접 LLM 호출로 코드 수정
"""

import os
import re
import uuid
import subprocess
import logging
import requests
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

from .project_manager import get_project, _load_registry, _save_registry

logger = logging.getLogger(__name__)


class AiderBridge:
    """aider-chat Python API 래퍼"""

    def __init__(self, config, llm_provider):
        self.config = config
        self.llm_provider = llm_provider
        self._coders: Dict[str, object] = {}
        self._pending_proposals: Dict[str, dict] = {}

    def _get_llm_config(self) -> tuple:
        """현재 LLM 설정 반환: (model_name, api_base, api_key)"""
        # API 모드
        if self.config.api_token and self.config.llm_mode != "local":
            cfg = self.config.ENV_CONFIG.get(self.config.env_mode, {})
            url = cfg.get("url", "")
            api_base = url.rsplit("/chat/completions", 1)[0] if "/chat/completions" in url else url
            model_name = f"openai/{cfg.get('model', 'gpt-oss-20b')}"
            return model_name, api_base, self.config.api_token

        # GGUF 모드 → 로컬 서버
        model_name = f"openai/{self.config.current_gguf_model or 'local-gguf'}"
        api_base = "http://127.0.0.1:10002/v1"
        api_key = "sk-local-dummy"
        return model_name, api_base, api_key

    def _get_project_files(self, project_path: Path, files: Optional[List[str]] = None) -> List[str]:
        """프로젝트 파일 목록 반환"""
        if files:
            return [str(project_path / f) for f in files]
        code_exts = {
            ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".h",
            ".cs", ".go", ".rs", ".sql", ".html", ".css", ".json", ".yaml", ".yml",
            ".md", ".txt", ".sh", ".bat", ".ps1", ".r", ".R"
        }
        return [str(f) for f in project_path.rglob("*")
                if f.is_file() and f.suffix in code_exts and ".git" not in f.parts]

    def _chat_with_aider(self, project_path: Path, model_name: str, api_base: str,
                         api_key: str, message: str, fnames: List[str]) -> dict:
        """aider를 통한 코드 수정 (stash 기반 승인 워크플로우)"""
        try:
            from aider.coders import Coder
            from aider.models import Model
            from aider.io import InputOutput
        except ImportError:
            return {"success": False, "error": "aider-chat 미설치. pip install aider-chat"}

        os.environ["OPENAI_API_BASE"] = api_base
        os.environ["OPENAI_API_KEY"] = api_key

        original_cwd = os.getcwd()
        os.chdir(str(project_path))
        try:
            from io import StringIO
            capture = StringIO()
            io = InputOutput(yes=True, output=capture)
            model = Model(model_name)
            coder = Coder.create(
                main_model=model, fnames=fnames, io=io,
                auto_commits=False, use_git=True
            )
            result = coder.run(message)

            captured_output = capture.getvalue()
            response_text = result if result else captured_output.strip()
            if not response_text:
                response_text = "변경 제안 없음"

            # diff 수집
            diff_text = self._collect_diff(project_path)

            # stash로 임시 보관
            proposal_id = f"prop_{uuid.uuid4().hex[:8]}"
            has_changes = bool(diff_text)

            if has_changes:
                self._update_gitignore(project_path)
                self._stash_changes(project_path, proposal_id)

            return {
                "success": True,
                "response": response_text,
                "diff": diff_text,
                "proposal_id": proposal_id if has_changes else None,
                "needs_approval": has_changes,
                "model": model_name,
                "mode": "preview"
            }
        except Exception as e:
            logger.error(f"aider 오류: {e}")
            try:
                subprocess.run(["git", "-C", str(project_path), "checkout", "."], capture_output=True)
            except Exception:
                pass
            return {"success": False, "error": str(e)}
        finally:
            os.chdir(original_cwd)

    def _chat_direct_llm(self, project_path: Path, message: str, fnames: List[str]) -> dict:
        """aider 대신 직접 LLM 호출로 코드 수정 (GGUF 안정성 확보)"""
        # 선택된 파일 내용 읽기
        file_contents = {}
        for fpath in fnames:
            p = Path(fpath)
            if p.exists() and p.is_file():
                try:
                    content = p.read_text(encoding="utf-8")
                    rel = p.relative_to(project_path)
                    file_contents[str(rel)] = content
                except Exception:
                    pass

        if not file_contents:
            return {"success": False, "error": "선택된 파일을 읽을 수 없습니다"}

        # 파일 내용을 프롬프트에 포함
        files_text = ""
        for fname, content in file_contents.items():
            ext = Path(fname).suffix.lstrip(".")
            files_text += f"\n### {fname}\n```{ext}\n{content}\n```\n"

        system_prompt = """당신은 코드 수정 전문가입니다.
사용자가 요청한 변경사항을 기존 코드에 적용해주세요.

규칙:
1. 반드시 수정된 전체 파일을 ```파일명 형식으로 제공하세요
2. 수정한 부분에 주석으로 설명을 달아주세요
3. 어떤 부분을 왜 변경했는지 한국어로 설명해주세요
4. 기존 코드의 구조와 스타일을 유지하세요
5. 요청하지 않은 부분은 절대 변경하지 마세요"""

        user_prompt = f"""다음 파일을 수정해주세요.

## 현재 파일 내용
{files_text}

## 수정 요청
{message}

수정된 전체 파일 코드와 변경 설명을 제공해주세요."""

        # LLM 호출
        model_name, api_base, api_key = self._get_llm_config()

        try:
            resp = requests.post(
                f"{api_base}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model_name.replace("openai/", ""),
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "max_tokens": 4000,
                    "temperature": 0.3,
                    "stream": False
                },
                timeout=300
            )

            if resp.status_code != 200:
                return {"success": False, "error": f"LLM 호출 실패: {resp.status_code}"}

            result_data = resp.json()
            response_text = result_data["choices"][0]["message"]["content"]

            # 응답에서 코드 블록 추출 → 파일에 적용
            code_blocks = re.findall(r'```(?:\w+)?\s*\n(.*?)```', response_text, re.DOTALL)

            applied_files = []
            for fname, original in file_contents.items():
                for block in code_blocks:
                    block_stripped = block.strip()
                    if block_stripped != original.strip() and len(block_stripped) > 10:
                        target = project_path / fname
                        target.write_text(block_stripped + "\n", encoding="utf-8")
                        applied_files.append(fname)
                        break

            # diff 수집
            diff_text = self._collect_diff(project_path)

            # stash로 임시 보관
            proposal_id = f"prop_{uuid.uuid4().hex[:8]}"
            has_changes = bool(diff_text)

            if has_changes:
                self._update_gitignore(project_path)
                subprocess.run(["git", "-C", str(project_path), "add", "-A"], capture_output=True)
                subprocess.run(
                    ["git", "-C", str(project_path), "stash", "push", "-m",
                     f"aider-proposal-{proposal_id}"],
                    capture_output=True, encoding="utf-8"
                )
                self._pending_proposals[proposal_id] = {
                    "project_path": str(project_path),
                    "created": datetime.now().isoformat(),
                }

            return {
                "success": True,
                "response": response_text,
                "diff": diff_text,
                "proposal_id": proposal_id if has_changes else None,
                "needs_approval": has_changes,
                "applied_files": applied_files,
                "model": model_name,
                "mode": "preview"
            }
        except Exception as e:
            try:
                subprocess.run(["git", "-C", str(project_path), "checkout", "."], capture_output=True)
            except Exception:
                pass
            logger.error(f"직접 LLM 수정 오류: {e}")
            return {"success": False, "error": str(e)}

    def chat(self, project_id: str, message: str, files: Optional[List[str]] = None) -> dict:
        """코드 수정 요청 - GGUF면 직접 LLM, API면 aider"""
        project = get_project(project_id)
        if not project:
            return {"success": False, "error": "프로젝트를 찾을 수 없습니다"}

        model_name, api_base, api_key = self._get_llm_config()
        project_path = Path(project["path"])
        fnames = self._get_project_files(project_path, files)

        if "127.0.0.1:10002" in api_base:
            logger.info(f"직접 LLM 모드 (로컬 GGUF) - 파일 {len(fnames)}개")
            result = self._chat_direct_llm(project_path, message, fnames)
        else:
            logger.info(f"aider 모드 (API) - 파일 {len(fnames)}개")
            result = self._chat_with_aider(project_path, model_name, api_base, api_key, message, fnames)

        # last_active 업데이트
        if result.get("success"):
            reg = _load_registry()
            for p in reg["projects"]:
                if p["id"] == project_id:
                    p["last_active"] = datetime.now().isoformat()
            _save_registry(reg)

        return result

    def approve_proposal(self, proposal_id: str) -> dict:
        """승인: stash pop → 커밋"""
        proposal = self._pending_proposals.pop(proposal_id, None)
        if not proposal:
            return {"success": False, "error": "만료되었거나 존재하지 않는 제안입니다"}

        project_path = proposal["project_path"]
        try:
            # aider 캐시 정리
            import shutil
            aider_cache = Path(project_path) / ".aider.tags.cache.v4"
            if aider_cache.exists():
                shutil.rmtree(aider_cache, ignore_errors=True)

            result = subprocess.run(
                ["git", "-C", project_path, "stash", "pop"],
                capture_output=True, text=True, encoding="utf-8"
            )
            if result.returncode != 0:
                return {"success": False, "error": f"stash pop 실패: {result.stderr}"}

            # diff 수집
            diff_text = ""
            try:
                diff_result = subprocess.run(
                    ["git", "-C", project_path, "diff"],
                    capture_output=True, text=True, encoding="utf-8"
                )
                diff_text = diff_result.stdout.strip()
            except Exception:
                pass

            # 커밋
            subprocess.run(["git", "-C", project_path, "add", "-A"], capture_output=True)
            subprocess.run(
                ["git", "-C", project_path, "commit", "-m",
                 f"aider: approved changes ({proposal_id})"],
                capture_output=True, encoding="utf-8"
            )

            logger.info(f"변경 승인 완료: {proposal_id}")
            return {"success": True, "response": "변경이 적용되었습니다!", "diff": diff_text, "mode": "applied"}
        except Exception as e:
            logger.error(f"승인 적용 오류: {e}")
            return {"success": False, "error": str(e)}

    def reject_proposal(self, proposal_id: str) -> dict:
        """거부: stash drop"""
        proposal = self._pending_proposals.pop(proposal_id, None)
        if not proposal:
            return {"success": False, "error": "제안을 찾을 수 없습니다"}

        project_path = proposal["project_path"]
        try:
            result = subprocess.run(
                ["git", "-C", project_path, "stash", "list"],
                capture_output=True, text=True, encoding="utf-8"
            )
            for line in result.stdout.strip().split("\n"):
                if f"aider-proposal-{proposal_id}" in line:
                    stash_ref = line.split(":")[0]
                    subprocess.run(
                        ["git", "-C", project_path, "stash", "drop", stash_ref],
                        capture_output=True, encoding="utf-8"
                    )
                    break

            logger.info(f"변경 거부: {proposal_id}")
            return {"success": True, "message": "변경이 취소되었습니다. 기존 코드 유지."}
        except Exception as e:
            logger.error(f"거부 처리 오류: {e}")
            return {"success": False, "error": str(e)}

    # --- 유틸리티 ---

    def _collect_diff(self, project_path: Path) -> str:
        """변경사항 diff 수집"""
        diff_text = ""
        try:
            diff_result = subprocess.run(
                ["git", "-C", str(project_path), "diff"],
                capture_output=True, text=True, encoding="utf-8"
            )
            diff_text = diff_result.stdout.strip()
        except Exception:
            pass

        if not diff_text:
            try:
                diff_result = subprocess.run(
                    ["git", "-C", str(project_path), "diff", "--cached"],
                    capture_output=True, text=True, encoding="utf-8"
                )
                diff_text = diff_result.stdout.strip()
            except Exception:
                pass

        return diff_text

    def _update_gitignore(self, project_path: Path):
        """aider 캐시 파일을 gitignore에 추가"""
        gitignore = project_path / ".gitignore"
        ignore_entries = [".aider*", ".aider.tags.cache*"]
        try:
            existing = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
            added = False
            for entry in ignore_entries:
                if entry not in existing:
                    existing += f"\n{entry}"
                    added = True
            if added:
                gitignore.write_text(existing.strip() + "\n", encoding="utf-8")
        except Exception:
            pass

        subprocess.run(
            ["git", "-C", str(project_path), "rm", "-r", "--cached",
             "--ignore-unmatch", ".aider.tags.cache.v4"],
            capture_output=True, encoding="utf-8"
        )

    def _stash_changes(self, project_path: Path, proposal_id: str):
        """변경사항을 stash에 보관"""
        subprocess.run(
            ["git", "-C", str(project_path), "add", "-A"],
            capture_output=True
        )
        subprocess.run(
            ["git", "-C", str(project_path), "stash", "push", "-m",
             f"aider-proposal-{proposal_id}"],
            capture_output=True, encoding="utf-8"
        )
        logger.info(f"변경사항 stash 보관: {proposal_id}")
        self._pending_proposals[proposal_id] = {
            "project_path": str(project_path),
            "created": datetime.now().isoformat(),
        }
