#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Git 작업 - diff, stash, approve/reject, undo, history
"""

import subprocess
import logging
from .project_manager import get_project

logger = logging.getLogger(__name__)


def undo(project_id: str) -> dict:
    """마지막 변경 취소 (git reset)"""
    project = get_project(project_id)
    if not project:
        return {"success": False, "error": "프로젝트를 찾을 수 없습니다"}

    project_path = project["path"]
    try:
        result = subprocess.run(
            ["git", "-C", project_path, "log", "--oneline", "-1"],
            capture_output=True, text=True, encoding="utf-8"
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


def get_diff(project_id: str) -> dict:
    """현재 변경사항 diff"""
    project = get_project(project_id)
    if not project:
        return {"success": False, "error": "프로젝트를 찾을 수 없습니다"}

    try:
        result = subprocess.run(
            ["git", "-C", project["path"], "diff", "HEAD~1", "HEAD"],
            capture_output=True, text=True, encoding="utf-8"
        )
        diff_text = result.stdout

        if not diff_text:
            result = subprocess.run(
                ["git", "-C", project["path"], "diff"],
                capture_output=True, text=True, encoding="utf-8"
            )
            diff_text = result.stdout

        return {"success": True, "diff": diff_text or "변경사항 없음"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_history(project_id: str, limit: int = 20) -> dict:
    """git 커밋 히스토리"""
    project = get_project(project_id)
    if not project:
        return {"success": False, "error": "프로젝트를 찾을 수 없습니다"}

    try:
        result = subprocess.run(
            ["git", "-C", project["path"], "log",
             f"--max-count={limit}",
             "--format=%H|%h|%s|%ai|%an"],
            capture_output=True, text=True, encoding="utf-8"
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
