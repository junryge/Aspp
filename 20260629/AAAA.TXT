"""
demos_v1/routes_agents.py - 개인 에이전트(프리셋) 저장/로드 라우트 (사용자별)

자주 쓰는 설정 묶음(환경·스킬·지식·페르소나·effort)을 사용자 아이디별로
demos_v1/personal-agents/<user_id>/<agent_id>.json 에 저장한다.
saved-prompts CRUD 패턴(routes_api.py)을 따른다.

엔드포인트:
    GET    /api/agents?user_id=...
    POST   /api/agents                 (body: {user_id, agent_id?, name, persona, envs, skills, knowledge, effort})
    DELETE /api/agents/<agent_id>?user_id=...
"""
from __future__ import annotations
import json
import os
import time
from uuid import uuid4

from flask import request, jsonify, render_template

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
AGENTS_DIR = os.path.join(_PKG_DIR, "personal-agents")


def _safe(val: str) -> str:
    val = (val or "").strip()
    if not val or "/" in val or "\\" in val or ".." in val:
        return ""
    return val


def _user_dir(user_id: str) -> str:
    """user_id 별 폴더. 빈/잘못된 값은 거부 (멀티유저 누수 방지 — '_shared' 폴백 제거)."""
    uid = _safe(user_id)
    if not uid:
        raise ValueError("user_id 필요")
    d = os.path.join(AGENTS_DIR, uid)
    os.makedirs(d, exist_ok=True)
    return d


def _path(user_id: str, aid: str) -> str:
    return os.path.join(_user_dir(user_id), f"{aid}.json")


def _save(payload: dict) -> dict:
    user_id = _safe(payload.get("user_id") or "")
    aid = _safe(payload.get("agent_id") or "") or uuid4().hex

    try:
        effort = int(payload.get("effort", 2))
    except (TypeError, ValueError):
        effort = 2

    envs = payload.get("envs") or payload.get("env") or ["auto"]
    if isinstance(envs, str):
        envs = [envs]
    envs = [e for e in envs if isinstance(e, str)] or ["auto"]

    data = {
        "agent_id": aid,
        "user_id": user_id,
        "name": (payload.get("name") or "이름 없는 에이전트").strip()[:120],
        "persona": payload.get("persona") or "",
        "envs": envs,
        "skills": [s for s in (payload.get("skills") or []) if isinstance(s, str)],
        "knowledge": bool(payload.get("knowledge")),
        "knowledge_files": [f for f in (payload.get("knowledge_files") or []) if isinstance(f, str)],
        "knowledge_scripts": [s for s in (payload.get("knowledge_scripts") or []) if isinstance(s, str)],
        "effort": effort,
        "timestamp": time.time(),
    }
    # 원자적 쓰기(임시파일 → replace) — 동시 저장/중단 시 JSON 손상 방지
    dst = _path(user_id, aid)
    tmp = dst + f".tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, dst)
    return data


def _list(user_id: str) -> list[dict]:
    d = _user_dir(user_id)
    out: list[dict] = []
    for name in os.listdir(d):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(d, name), "r", encoding="utf-8") as f:
                out.append(json.load(f))
        except Exception:
            continue
    out.sort(key=lambda a: a.get("timestamp", 0), reverse=True)
    return out


def _delete(user_id: str, aid: str) -> bool:
    aid = _safe(aid)
    if not aid:
        return False
    p = _path(user_id, aid)
    if not os.path.isfile(p):
        return False
    try:
        os.remove(p)
        return True
    except Exception:
        return False


def register_agent_routes(app):
    """개인 에이전트 CRUD 라우트 등록."""

    @app.route("/agent-window")
    def agent_window():
        """개인 에이전트 독립 관리 창 (window.open 으로 열림)."""
        return render_template("agent_window.html")

    @app.route("/api/agents", methods=["GET"])
    def api_agents_list():
        if not _safe(request.args.get("user_id", "")):
            return jsonify({"error": "user_id 필요", "agents": []}), 400
        return jsonify({"agents": _list(request.args.get("user_id", ""))})

    @app.route("/api/agents", methods=["POST"])
    def api_agents_save():
        data = request.get_json(force=True, silent=True) or {}
        if not _safe(data.get("user_id", "")):
            return jsonify({"error": "user_id 필요"}), 400
        try:
            agent = _save(data)
        except Exception as e:
            return jsonify({"error": f"저장 실패: {e}"}), 500
        return jsonify({"success": True, "agent": agent})

    @app.route("/api/agents/<agent_id>", methods=["DELETE"])
    def api_agents_delete(agent_id):
        if not _safe(request.args.get("user_id", "")):
            return jsonify({"error": "user_id 필요"}), 400
        ok = _delete(request.args.get("user_id", ""), agent_id)
        if not ok:
            return jsonify({"error": "삭제 실패 또는 에이전트 없음"}), 404
        return jsonify({"success": True, "agent_id": agent_id})
