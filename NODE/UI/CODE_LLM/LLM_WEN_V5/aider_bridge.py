#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aider-chat 통합 브릿지 모듈 (v3.0)
- 프로젝트 기반 코드 수정 (aider Python API)
- 프로젝트 관리 CRUD
- API 우선 → GGUF 폴백 (llama-cpp-python 서버)
"""

import os
import sys
import json
import uuid
import shutil
import subprocess
import logging
import time
import re
from pathlib import Path
from typing import Optional, List, Dict, Literal
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ========================================
# 설정
# ========================================
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECTS_ROOT = BASE_DIR / "aider_projects"
PROJECTS_FILE = PROJECTS_ROOT / "projects.json"
# LOCAL_GGUF_PORT 삭제 - 별도 서버 불필요


# ========================================
# Pydantic Models
# ========================================
class ProjectCreate(BaseModel):
    name: str
    customer_type: Literal["code_editing", "data_analysis"]
    description: str = ""
    path: Optional[str] = None  # 직접 경로 지정 (없으면 자동)


class AiderChatRequest(BaseModel):
    message: str
    files: Optional[List[str]] = None


# ========================================
# 프로젝트 레지스트리
# ========================================
def _ensure_dirs():
    """프로젝트 디렉토리 구조 생성"""
    PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)
    (PROJECTS_ROOT / "code_editing").mkdir(exist_ok=True)
    (PROJECTS_ROOT / "data_analysis").mkdir(exist_ok=True)
    if not PROJECTS_FILE.exists():
        PROJECTS_FILE.write_text(json.dumps({"projects": []}, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_registry() -> dict:
    _ensure_dirs()
    return json.loads(PROJECTS_FILE.read_text(encoding="utf-8"))


def _save_registry(data: dict):
    PROJECTS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _get_project(project_id: str) -> Optional[dict]:
    reg = _load_registry()
    for p in reg["projects"]:
        if p["id"] == project_id:
            return p
    return None


# ========================================
# AiderBridge - 핵심 클래스
# ========================================
class AiderBridge:
    """aider-chat Python API 래퍼"""

    def __init__(self):
        _ensure_dirs()
        self._coders: Dict[str, object] = {}  # project_id -> Coder
        self._pending_proposals: Dict[str, dict] = {}  # proposal_id -> 변경 제안

    def _get_llm_config(self) -> tuple:
        """현재 LLM 설정 반환: (model_name, api_base, api_key)"""
        try:
            from Coding_llm_server_v2 import API_TOKEN, ENV_MODE, ENV_CONFIG, LLM_MODE, CURRENT_GGUF_MODEL
        except ImportError:
            API_TOKEN = None
            ENV_MODE = "common"
            ENV_CONFIG = {}
            LLM_MODE = "api"
            CURRENT_GGUF_MODEL = ""

        # API 모드 → aider가 SK Hynix API 직접 호출
        if API_TOKEN and LLM_MODE != "local":
            cfg = ENV_CONFIG.get(ENV_MODE, {})
            url = cfg.get("url", "")
            api_base = url.rsplit("/chat/completions", 1)[0] if "/chat/completions" in url else url
            model_name = f"openai/{cfg.get('model', 'gpt-oss-20b')}"
            return model_name, api_base, API_TOKEN

        # GGUF 모드 → 같은 서버(10002)의 OpenAI 호환 API 사용
        model_name = f"openai/{CURRENT_GGUF_MODEL or 'local-gguf'}"
        api_base = "http://127.0.0.1:10002/v1"
        api_key = "sk-local-dummy"
        return model_name, api_base, api_key

    @staticmethod
    def scan_folder(path: str) -> dict:
        """폴더 경로 검증 + 내부 파일 미리보기"""
        p = Path(path)
        if not p.exists():
            return {"success": False, "error": f"경로가 존재하지 않습니다: {path}"}
        if not p.is_dir():
            return {"success": False, "error": "폴더가 아닙니다"}

        files = [f for f in p.rglob("*") if f.is_file() and ".git" not in f.parts]
        has_git = (p / ".git").exists()

        return {
            "success": True,
            "path": str(p),
            "file_count": len(files),
            "has_git": has_git,
            "sample_files": [str(f.relative_to(p)).replace("\\", "/") for f in files[:10]],
        }

    def create_project(self, name: str, customer_type: str, description: str = "", path: Optional[str] = None) -> dict:
        """프로젝트 생성 - 폴더 경로 지정 시 통째로 인식"""
        reg = _load_registry()
        project_id = f"proj_{uuid.uuid4().hex[:8]}"

        is_existing_folder = False

        if path:
            project_path = Path(path)
            # ★ 경로 존재 검증
            if project_path.exists() and project_path.is_dir():
                is_existing_folder = True
            elif not project_path.exists():
                # 새 폴더 생성
                project_path.mkdir(parents=True, exist_ok=True)
        else:
            safe_name = re.sub(r'[^\w\-]', '_', name.lower())
            project_path = PROJECTS_ROOT / customer_type / safe_name
            project_path.mkdir(parents=True, exist_ok=True)

        # ★ 기존 파일 카운트
        existing_files = [f for f in project_path.rglob("*") if f.is_file() and ".git" not in f.parts]

        # git init (없으면 자동 생성)
        git_dir = project_path / ".git"
        if not git_dir.exists():
            try:
                subprocess.run(["git", "init", str(project_path)], capture_output=True, check=True)
                # README 없으면 생성
                readme = project_path / "README.md"
                if not readme.exists():
                    readme.write_text(f"# {name}\n{description}\n", encoding="utf-8")
                # ★ 기존 파일 포함 전체 add + 커밋
                subprocess.run(["git", "-C", str(project_path), "add", "."], capture_output=True)
                subprocess.run(
                    ["git", "-C", str(project_path), "commit", "-m",
                     f"Initial commit ({len(existing_files)} files)"],
                    capture_output=True
                )
                logger.info(f"✅ git 자동 초기화: {project_path} ({len(existing_files)}개 파일)")
            except Exception as e:
                logger.warning(f"git init 실패 (계속 진행): {e}")

        project = {
            "id": project_id,
            "name": name,
            "customer_type": customer_type,
            "description": description,
            "path": str(project_path),
            "created": datetime.now().isoformat(),
            "last_active": datetime.now().isoformat(),
            "file_count": len(existing_files),
            "is_existing_folder": is_existing_folder,
        }

        reg["projects"].append(project)
        _save_registry(reg)
        return project

    def list_projects(self) -> list:
        return _load_registry()["projects"]

    def delete_project(self, project_id: str) -> bool:
        reg = _load_registry()
        project = None
        for p in reg["projects"]:
            if p["id"] == project_id:
                project = p
                break

        if not project:
            return False

        # Coder 인스턴스 제거
        self._coders.pop(project_id, None)

        # 파일 삭제 (자동생성 경로만)
        ppath = Path(project["path"])
        if str(PROJECTS_ROOT) in str(ppath):
            shutil.rmtree(ppath, ignore_errors=True)

        reg["projects"] = [p for p in reg["projects"] if p["id"] != project_id]
        _save_registry(reg)
        return True

    def get_project_files(self, project_id: str) -> list:
        project = _get_project(project_id)
        if not project:
            return []

        ppath = Path(project["path"])
        files = []
        for f in ppath.rglob("*"):
            if f.is_file() and ".git" not in f.parts:
                rel = f.relative_to(ppath)
                files.append({
                    "path": str(rel).replace("\\", "/"),
                    "size": f.stat().st_size,
                    "ext": f.suffix,
                })
        return sorted(files, key=lambda x: x["path"])

    def read_file(self, project_id: str, file_path: str) -> Optional[str]:
        project = _get_project(project_id)
        if not project:
            return None

        fpath = Path(project["path"]) / file_path
        if not fpath.exists() or not fpath.is_file():
            return None

        try:
            return fpath.read_text(encoding="utf-8")
        except Exception:
            return None

    def _get_project_files(self, project_path: Path, files: Optional[List[str]] = None) -> List[str]:
        """프로젝트 파일 목록 반환"""
        if files:
            return [str(project_path / f) for f in files]
        code_exts = {".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".h",
                     ".cs", ".go", ".rs", ".sql", ".html", ".css", ".json", ".yaml", ".yml",
                     ".md", ".txt", ".sh", ".bat", ".ps1", ".r", ".R"}
        return [str(f) for f in project_path.rglob("*")
                if f.is_file() and f.suffix in code_exts and ".git" not in f.parts]

    def _chat_with_aider(self, project_path: Path, model_name: str, api_base: str, api_key: str,
                         message: str, fnames: List[str]) -> dict:
        """aider를 통한 코드 수정 (변경 후 stash → 승인 시 pop 방식)"""
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
            # ★ auto_commits=False → 파일은 변경하되 커밋하지 않음
            coder = Coder.create(main_model=model, fnames=fnames, io=io,
                                 auto_commits=False, use_git=True)
            result = coder.run(message)

            captured_output = capture.getvalue()
            response_text = result if result else captured_output.strip()
            if not response_text:
                response_text = "변경 제안 없음"

            # ★ 변경된 diff 수집 (아직 커밋 안 됨)
            diff_text = ""
            try:
                diff_result = subprocess.run(
                    ["git", "-C", str(project_path), "diff"],
                    capture_output=True, text=True, encoding='utf-8'
                )
                diff_text = diff_result.stdout.strip()
            except Exception:
                pass

            # diff가 없으면 staged 확인
            if not diff_text:
                try:
                    diff_result = subprocess.run(
                        ["git", "-C", str(project_path), "diff", "--cached"],
                        capture_output=True, text=True, encoding='utf-8'
                    )
                    diff_text = diff_result.stdout.strip()
                except Exception:
                    pass

            # ★ 변경사항을 git stash로 임시 보관 (원본 복원)
            proposal_id = f"prop_{uuid.uuid4().hex[:8]}"
            has_changes = bool(diff_text)

            if has_changes:
                # .gitignore에 aider 캐시 추가 (충돌 방지)
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

                # aider 캐시 파일은 git에서 제외
                subprocess.run(
                    ["git", "-C", str(project_path), "rm", "-r", "--cached", "--ignore-unmatch", ".aider.tags.cache.v4"],
                    capture_output=True, encoding='utf-8'
                )

                # stash에 저장 (tracked 변경사항만)
                subprocess.run(
                    ["git", "-C", str(project_path), "add", "-A"],
                    capture_output=True
                )
                subprocess.run(
                    ["git", "-C", str(project_path), "stash", "push", "-m", f"aider-proposal-{proposal_id}"],
                    capture_output=True, encoding='utf-8'
                )
                logger.info(f"📦 변경사항 stash 보관: {proposal_id}")

                self._pending_proposals[proposal_id] = {
                    "project_path": str(project_path),
                    "created": datetime.now().isoformat(),
                }

            logger.info(f"📋 변경 제안: {proposal_id} | 응답: {len(response_text)}자, diff: {len(diff_text)}자")

            return {"success": True,
                    "response": response_text,
                    "diff": diff_text,
                    "proposal_id": proposal_id if has_changes else None,
                    "needs_approval": has_changes,
                    "model": model_name, "mode": "preview"}
        except Exception as e:
            logger.error(f"aider 오류: {e}")
            # 오류 시 변경사항 되돌리기
            try:
                subprocess.run(["git", "-C", str(project_path), "checkout", "."], capture_output=True)
            except Exception:
                pass
            return {"success": False, "error": str(e)}
        finally:
            os.chdir(original_cwd)

    def approve_proposal(self, proposal_id: str) -> dict:
        """승인: stash pop으로 변경사항 적용 + 커밋"""
        proposal = self._pending_proposals.pop(proposal_id, None)
        if not proposal:
            return {"success": False, "error": "만료되었거나 존재하지 않는 제안입니다"}

        project_path = proposal["project_path"]
        try:
            # aider 캐시 파일 정리 (stash pop 충돌 방지)
            import shutil as _shutil
            aider_cache = Path(project_path) / ".aider.tags.cache.v4"
            if aider_cache.exists():
                _shutil.rmtree(aider_cache, ignore_errors=True)

            # stash pop으로 변경사항 복원
            result = subprocess.run(
                ["git", "-C", project_path, "stash", "pop"],
                capture_output=True, text=True, encoding='utf-8'
            )
            if result.returncode != 0:
                return {"success": False, "error": f"stash pop 실패: {result.stderr}"}

            # diff 수집
            diff_text = ""
            try:
                diff_result = subprocess.run(
                    ["git", "-C", project_path, "diff"],
                    capture_output=True, text=True, encoding='utf-8'
                )
                diff_text = diff_result.stdout.strip()
            except Exception:
                pass

            # 커밋
            subprocess.run(
                ["git", "-C", project_path, "add", "-A"],
                capture_output=True
            )
            subprocess.run(
                ["git", "-C", project_path, "commit", "-m", f"aider: approved changes ({proposal_id})"],
                capture_output=True, encoding='utf-8'
            )

            logger.info(f"✅ 변경 승인 완료: {proposal_id}")
            return {"success": True, "response": "변경이 적용되었습니다!", "diff": diff_text, "mode": "applied"}
        except Exception as e:
            logger.error(f"승인 적용 오류: {e}")
            return {"success": False, "error": str(e)}

    def reject_proposal(self, proposal_id: str) -> dict:
        """거부: stash drop으로 변경사항 삭제"""
        proposal = self._pending_proposals.pop(proposal_id, None)
        if not proposal:
            return {"success": False, "error": "제안을 찾을 수 없습니다"}

        project_path = proposal["project_path"]
        try:
            # stash 목록에서 해당 proposal 찾아서 drop
            result = subprocess.run(
                ["git", "-C", project_path, "stash", "list"],
                capture_output=True, text=True, encoding='utf-8'
            )
            for line in result.stdout.strip().split("\n"):
                if f"aider-proposal-{proposal_id}" in line:
                    stash_ref = line.split(":")[0]  # stash@{0}
                    subprocess.run(
                        ["git", "-C", project_path, "stash", "drop", stash_ref],
                        capture_output=True, encoding='utf-8'
                    )
                    break

            logger.info(f"❌ 변경 거부: {proposal_id}")
            return {"success": True, "message": "변경이 취소되었습니다. 기존 코드 유지."}
        except Exception as e:
            logger.error(f"거부 처리 오류: {e}")
            return {"success": False, "error": str(e)}

    def _chat_direct_llm(self, project_path: Path, message: str, fnames: List[str]) -> dict:
        """aider 대신 직접 LLM 호출로 코드 수정 (로컬 GGUF 안정성 확보)"""
        import requests as _req

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

        # LLM 호출 (Coding_llm_server_v2의 /v1/chat/completions 사용)
        model_name, api_base, api_key = self._get_llm_config()

        try:
            resp = _req.post(
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
            import re as _re
            code_blocks = _re.findall(r'```(?:\w+)?\s*\n(.*?)```', response_text, _re.DOTALL)

            # 파일 변경 적용 (커밋 전)
            applied_files = []
            for fname, original in file_contents.items():
                # 응답에서 해당 파일의 수정 코드 찾기
                for block in code_blocks:
                    block_stripped = block.strip()
                    # 코드 블록이 원본과 다르고 충분히 비슷하면 적용
                    if block_stripped != original.strip() and len(block_stripped) > 10:
                        target = project_path / fname
                        target.write_text(block_stripped + "\n", encoding="utf-8")
                        applied_files.append(fname)
                        break

            # diff 수집
            diff_text = ""
            try:
                diff_result = subprocess.run(
                    ["git", "-C", str(project_path), "diff"],
                    capture_output=True, text=True, encoding='utf-8'
                )
                diff_text = diff_result.stdout.strip()
            except Exception:
                pass

            # stash로 임시 보관
            proposal_id = f"prop_{uuid.uuid4().hex[:8]}"
            has_changes = bool(diff_text)

            if has_changes:
                # .gitignore 업데이트
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

                subprocess.run(["git", "-C", str(project_path), "add", "-A"], capture_output=True)
                subprocess.run(
                    ["git", "-C", str(project_path), "stash", "push", "-m", f"aider-proposal-{proposal_id}"],
                    capture_output=True, encoding='utf-8'
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
                "model": model_name, "mode": "preview"
            }
        except Exception as e:
            # 오류 시 변경 되돌리기
            try:
                subprocess.run(["git", "-C", str(project_path), "checkout", "."], capture_output=True)
            except Exception:
                pass
            logger.error(f"직접 LLM 수정 오류: {e}")
            return {"success": False, "error": str(e)}

    def chat(self, project_id: str, message: str, files: Optional[List[str]] = None) -> dict:
        """코드 수정 요청 - 로컬 GGUF면 직접 LLM, API면 aider 사용"""
        project = _get_project(project_id)
        if not project:
            return {"success": False, "error": "프로젝트를 찾을 수 없습니다"}

        model_name, api_base, api_key = self._get_llm_config()
        project_path = Path(project["path"])
        fnames = self._get_project_files(project_path, files)

        # ★ 로컬 GGUF 모드면 직접 LLM 호출 (aider 충돌 방지)
        if "127.0.0.1:10002" in api_base:
            logger.info(f"🔧 직접 LLM 모드 (로컬 GGUF) - 파일 {len(fnames)}개")
            result = self._chat_direct_llm(project_path, message, fnames)
        else:
            logger.info(f"🔧 aider 모드 (API) - 파일 {len(fnames)}개")
            result = self._chat_with_aider(project_path, model_name, api_base, api_key, message, fnames)

        # last_active 업데이트
        if result.get("success"):
            reg = _load_registry()
            for p in reg["projects"]:
                if p["id"] == project_id:
                    p["last_active"] = datetime.now().isoformat()
            _save_registry(reg)

        return result

    def undo(self, project_id: str) -> dict:
        """마지막 aider 변경 취소 (git revert)"""
        project = _get_project(project_id)
        if not project:
            return {"success": False, "error": "프로젝트를 찾을 수 없습니다"}

        project_path = project["path"]
        try:
            result = subprocess.run(
                ["git", "-C", project_path, "log", "--oneline", "-1"],
                capture_output=True, text=True, encoding='utf-8'
            )
            last_commit = result.stdout.strip()

            subprocess.run(
                ["git", "-C", project_path, "reset", "--soft", "HEAD~1"],
                capture_output=True, check=True
            )
            subprocess.run(
                ["git", "-C", project_path, "checkout", "."],
                capture_output=True
            )

            return {"success": True, "reverted": last_commit}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_diff(self, project_id: str) -> dict:
        """현재 변경사항 diff"""
        project = _get_project(project_id)
        if not project:
            return {"success": False, "error": "프로젝트를 찾을 수 없습니다"}

        try:
            # 최근 커밋과의 diff
            result = subprocess.run(
                ["git", "-C", project["path"], "diff", "HEAD~1", "HEAD"],
                capture_output=True, text=True, encoding='utf-8'
            )
            diff_text = result.stdout

            if not diff_text:
                # unstaged 변경
                result = subprocess.run(
                    ["git", "-C", project["path"], "diff"],
                    capture_output=True, text=True, encoding='utf-8'
                )
                diff_text = result.stdout

            return {"success": True, "diff": diff_text or "변경사항 없음"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_history(self, project_id: str, limit: int = 20) -> dict:
        """git 커밋 히스토리"""
        project = _get_project(project_id)
        if not project:
            return {"success": False, "error": "프로젝트를 찾을 수 없습니다"}

        try:
            result = subprocess.run(
                ["git", "-C", project["path"], "log",
                 f"--max-count={limit}",
                 "--format=%H|%h|%s|%ai|%an"],
                capture_output=True, text=True, encoding='utf-8'
            )

            commits = []
            for line in result.stdout.strip().split("\n"):
                if "|" in line:
                    parts = line.split("|", 4)
                    commits.append({
                        "hash": parts[0],
                        "short_hash": parts[1],
                        "message": parts[2],
                        "date": parts[3],
                        "author": parts[4] if len(parts) > 4 else "",
                    })

            return {"success": True, "commits": commits}
        except Exception as e:
            return {"success": False, "error": str(e)}


# ========================================
# 싱글톤 인스턴스
# ========================================
_bridge = AiderBridge()


# ========================================
# FastAPI Router
# ========================================
def get_aider_router() -> APIRouter:
    router = APIRouter(prefix="/api/aider", tags=["aider"])

    @router.get("/scan_folder")
    async def scan_folder(path: str = Query(...)):
        """폴더 경로 스캔 - 파일 수, git 여부 미리보기"""
        return _bridge.scan_folder(path)

    @router.get("/projects")
    async def list_projects():
        return {"success": True, "projects": _bridge.list_projects()}

    @router.post("/projects")
    async def create_project(data: ProjectCreate):
        try:
            project = _bridge.create_project(
                name=data.name,
                customer_type=data.customer_type,
                description=data.description,
                path=data.path,
            )
            return {"success": True, "project": project}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @router.delete("/projects/{project_id}")
    async def delete_project(project_id: str):
        ok = _bridge.delete_project(project_id)
        return {"success": ok}

    @router.get("/projects/{project_id}/files")
    async def get_files(project_id: str):
        files = _bridge.get_project_files(project_id)
        return {"success": True, "files": files}

    @router.get("/projects/{project_id}/file")
    async def read_file(project_id: str, path: str = Query(...)):
        content = _bridge.read_file(project_id, path)
        if content is None:
            return {"success": False, "error": "파일을 찾을 수 없습니다"}
        return {"success": True, "content": content, "path": path}

    @router.post("/projects/{project_id}/files")
    async def upload_file(project_id: str, file: UploadFile = File(...), dest_path: str = Form("")):
        project = _get_project(project_id)
        if not project:
            return {"success": False, "error": "프로젝트를 찾을 수 없습니다"}

        ppath = Path(project["path"])
        if dest_path:
            target = ppath / dest_path
        else:
            target = ppath / file.filename

        target.parent.mkdir(parents=True, exist_ok=True)
        content = await file.read()
        target.write_bytes(content)

        # git add
        try:
            subprocess.run(["git", "-C", str(ppath), "add", str(target)], capture_output=True)
            subprocess.run(
                ["git", "-C", str(ppath), "commit", "-m", f"Add {file.filename}"],
                capture_output=True
            )
        except Exception:
            pass

        return {"success": True, "path": str(target.relative_to(ppath)).replace("\\", "/")}

    @router.post("/projects/{project_id}/chat")
    async def chat(project_id: str, data: AiderChatRequest):
        import asyncio
        # ★ aider는 동기 블로킹 + 자기 서버(/v1/chat/completions)에 HTTP 요청을 보내므로
        #   반드시 별도 스레드에서 실행해야 데드락 방지
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: _bridge.chat(project_id, data.message, data.files)
        )
        return result

    @router.post("/proposals/{proposal_id}/approve")
    async def approve_proposal(proposal_id: str):
        """변경 제안 승인 → 실제 적용"""
        import asyncio
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: _bridge.approve_proposal(proposal_id)
        )
        # 승인 후 프로젝트 파일 갱신을 위해 project_id 포함
        return result

    @router.post("/proposals/{proposal_id}/reject")
    async def reject_proposal(proposal_id: str):
        """변경 제안 거부"""
        return _bridge.reject_proposal(proposal_id)

    @router.put("/projects/{project_id}/file")
    async def save_file(project_id: str, data: dict):
        """파일 내용 저장 (직접 편집)"""
        project = _get_project(project_id)
        if not project:
            return {"success": False, "error": "프로젝트를 찾을 수 없습니다"}
        fpath = Path(project["path"]) / data.get("path", "")
        if not fpath.exists() or not fpath.is_file():
            return {"success": False, "error": "파일을 찾을 수 없습니다"}
        try:
            fpath.write_text(data.get("content", ""), encoding="utf-8")
            # git add + commit
            subprocess.run(["git", "-C", project["path"], "add", str(fpath)], capture_output=True)
            subprocess.run(
                ["git", "-C", project["path"], "commit", "-m", f"Edit {data.get('path', '')}"],
                capture_output=True, encoding='utf-8'
            )
            return {"success": True, "message": "저장 완료"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @router.post("/projects/{project_id}/rename")
    async def rename_file(project_id: str, data: dict):
        """파일/폴더 이름 변경"""
        project = _get_project(project_id)
        if not project:
            return {"success": False, "error": "프로젝트를 찾을 수 없습니다"}
        old_path = Path(project["path"]) / data.get("old_path", "")
        new_name = data.get("new_name", "").strip()
        if not old_path.exists():
            return {"success": False, "error": "대상을 찾을 수 없습니다"}
        if not new_name:
            return {"success": False, "error": "새 이름을 입력해주세요"}
        new_path = old_path.parent / new_name
        if new_path.exists():
            return {"success": False, "error": "같은 이름이 이미 존재합니다"}
        try:
            old_path.rename(new_path)
            subprocess.run(["git", "-C", project["path"], "add", "-A"], capture_output=True)
            subprocess.run(
                ["git", "-C", project["path"], "commit", "-m", f"Rename {data.get('old_path','')} -> {new_name}"],
                capture_output=True, encoding='utf-8'
            )
            return {"success": True, "message": "이름 변경 완료"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @router.delete("/projects/{project_id}/file")
    async def delete_file(project_id: str, path: str = Query(...)):
        """파일/폴더 삭제"""
        project = _get_project(project_id)
        if not project:
            return {"success": False, "error": "프로젝트를 찾을 수 없습니다"}
        target = Path(project["path"]) / path
        if not target.exists():
            return {"success": False, "error": "대상을 찾을 수 없습니다"}
        try:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
            subprocess.run(["git", "-C", project["path"], "add", "-A"], capture_output=True)
            subprocess.run(
                ["git", "-C", project["path"], "commit", "-m", f"Delete {path}"],
                capture_output=True, encoding='utf-8'
            )
            return {"success": True, "message": "삭제 완료"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @router.post("/projects/{project_id}/mkdir")
    async def create_folder(project_id: str, data: dict):
        """새 폴더 생성"""
        project = _get_project(project_id)
        if not project:
            return {"success": False, "error": "프로젝트를 찾을 수 없습니다"}
        folder_path = Path(project["path"]) / data.get("path", "")
        if folder_path.exists():
            return {"success": False, "error": "이미 존재하는 폴더입니다"}
        try:
            folder_path.mkdir(parents=True, exist_ok=True)
            # 빈 폴더는 git에 안 잡히므로 .gitkeep 생성
            (folder_path / ".gitkeep").write_text("", encoding="utf-8")
            subprocess.run(["git", "-C", project["path"], "add", "-A"], capture_output=True)
            subprocess.run(
                ["git", "-C", project["path"], "commit", "-m", f"Create folder {data.get('path','')}"],
                capture_output=True, encoding='utf-8'
            )
            return {"success": True, "message": "폴더 생성 완료"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @router.post("/projects/{project_id}/new_file")
    async def create_new_file(project_id: str, data: dict):
        """새 파일 생성"""
        project = _get_project(project_id)
        if not project:
            return {"success": False, "error": "프로젝트를 찾을 수 없습니다"}
        file_path = Path(project["path"]) / data.get("path", "")
        if file_path.exists():
            return {"success": False, "error": "이미 존재하는 파일입니다"}
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(data.get("content", ""), encoding="utf-8")
            subprocess.run(["git", "-C", project["path"], "add", str(file_path)], capture_output=True)
            subprocess.run(
                ["git", "-C", project["path"], "commit", "-m", f"Create {data.get('path','')}"],
                capture_output=True, encoding='utf-8'
            )
            return {"success": True, "message": "파일 생성 완료"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    @router.post("/projects/{project_id}/undo")
    async def undo(project_id: str):
        return _bridge.undo(project_id)

    @router.get("/projects/{project_id}/diff")
    async def get_diff(project_id: str):
        return _bridge.get_diff(project_id)

    @router.get("/projects/{project_id}/history")
    async def get_history(project_id: str, limit: int = 20):
        return _bridge.get_history(project_id, limit)

    return router
