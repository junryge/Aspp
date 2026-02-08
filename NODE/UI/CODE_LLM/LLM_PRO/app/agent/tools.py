#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
에이전트 도구 - ValidateCodeTool, SaveCodeTool
"""

import re
import json
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# nanobot 사용 가능 여부
try:
    from nanobot.agent.tools.base import Tool
    HAS_NANOBOT = True
except ImportError:
    HAS_NANOBOT = False
    class Tool:
        """더미 Tool 클래스"""
        @property
        def name(self): return ""
        @property
        def description(self): return ""
        @property
        def parameters(self): return {}
        async def execute(self, **kw): return ""

WORKSPACE = Path("nanobot_workspace")


class ValidateCodeTool(Tool):
    """코드 구문 검증"""
    @property
    def name(self): return "validate_code"

    @property
    def description(self): return "코드 구문 검증 (Python/JS/HTML)"

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "검증할 코드"},
                "language": {"type": "string", "description": "언어"}
            },
            "required": ["code", "language"]
        }

    async def execute(self, code="", language="python", **kw):
        errors = []
        lang = language.lower()

        if len(code.strip()) < 5:
            return json.dumps({"valid": False, "errors": ["코드가 너무 짧음"]})

        if lang == "python":
            try:
                compile(code, "<agent>", "exec")
            except SyntaxError as e:
                errors.append(f"Line {e.lineno}: {e.msg}")

        elif lang in ("javascript", "typescript"):
            stack = []
            pairs = {'{': '}', '(': ')', '[': ']'}
            for i, ch in enumerate(code):
                if ch in pairs:
                    stack.append((ch, i))
                elif ch in pairs.values():
                    if not stack:
                        errors.append(f"Pos {i}: '{ch}' 대응없음")
                    else:
                        o, _ = stack.pop()
                        if pairs[o] != ch:
                            errors.append(f"Pos {i}: '{pairs[o]}' 예상")
            for o, p in stack:
                errors.append(f"Pos {p}: '{o}' 미닫힘")

        elif lang == "html":
            void = {
                'br', 'hr', 'img', 'input', 'meta', 'link', 'area',
                'base', 'col', 'embed', 'source', 'track', 'wbr'
            }
            opens = [
                t.lower() for t in re.findall(r'<([a-zA-Z]\w*)[^>]*(?<!/)>', code)
                if t.lower() not in void
            ]
            closes = [t.lower() for t in re.findall(r'</([a-zA-Z]\w*)>', code)]
            if len(opens) != len(closes):
                errors.append(f"태그 불일치: 열기{len(opens)} 닫기{len(closes)}")

        return json.dumps(
            {"valid": not errors, "errors": errors, "language": lang},
            ensure_ascii=False
        )


class SaveCodeTool(Tool):
    """코드를 워크스페이스에 저장"""
    @property
    def name(self): return "save_code"

    @property
    def description(self): return "코드를 워크스페이스에 저장"

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "파일명"},
                "code": {"type": "string", "description": "코드"},
                "language": {"type": "string", "description": "언어"}
            },
            "required": ["filename", "code"]
        }

    async def execute(self, filename="", code="", language="", **kw):
        out = WORKSPACE / "output"
        out.mkdir(parents=True, exist_ok=True)

        safe = "".join(c for c in filename if c.isalnum() or c in '.-_').strip()
        if not safe:
            ext_map = {
                "python": ".py", "javascript": ".js", "typescript": ".ts",
                "java": ".java", "html": ".html", "css": ".css",
                "sql": ".sql", "bash": ".sh"
            }
            ext = ext_map.get(language.lower(), ".txt")
            safe = f"code_{int(time.time())}{ext}"

        fp = out / safe
        fp.write_text(code, encoding="utf-8")

        return json.dumps(
            {"saved": True, "path": str(fp), "filename": safe, "lines": len(code.split('\n'))},
            ensure_ascii=False
        )
