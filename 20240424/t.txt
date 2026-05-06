"""
code_assist_v1/routes_sessions.py - 자체 세션 저장/로드 라우트

채팅 세션을 code_assist_v1/sessions/ 에 JSON 으로 저장한다.
harness_bridge 가용 여부와 무관하게 항상 동작한다.

엔드포인트:
    POST   /api/harness/session/save              (body: {session_id?, messages, skills_used?, metadata?, uploaded_files?})
    GET    /api/harness/session/load/<sid>
    GET    /api/harness/session/list
    DELETE /api/harness/session/delete/<sid>
"""
from __future__ import annotations
import json
import os
import time
from uuid import uuid4

from flask import request, jsonify

from code_assist_v1.config import SESSIONS_DIR


def _safe_sid(sid: str) -> str:
    sid = (sid or "").strip()
    if not sid or "/" in sid or "\\" in sid or ".." in sid:
        return ""
    return sid


def _session_path(sid: str) -> str:
    return os.path.join(SESSIONS_DIR, f"{sid}.json")


def _save(payload: dict) -> str:
    sid = _safe_sid(payload.get("session_id") or "") or uuid4().hex
    os.makedirs(SESSIONS_DIR, exist_ok=True)

    msgs = payload.get("messages") or []
    first_msg = ""
    for m in msgs:
        if m.get("role") == "user":
            c = m.get("content", "")
            if isinstance(c, str) and c.strip():
                first_msg = c.strip()[:80]
                break
            if isinstance(c, list):
                txt = "\n".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text").strip()
                if txt:
                    first_msg = txt[:80]
                    break

    data = {
        "session_id": sid,
        "timestamp": time.time(),
        "first_message": first_msg,
        "message_count": len(msgs),
        "messages": msgs,
        "skills_used": payload.get("skills_used") or [],
        "uploaded_files": [
            {k: v for k, v in f.items() if k != "content_full"}
            for f in (payload.get("uploaded_files") or [])
        ],
        "metadata": payload.get("metadata") or {},
    }
    with open(_session_path(sid), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return sid


def _load(sid: str) -> dict | None:
    sid = _safe_sid(sid)
    if not sid:
        return None
    p = _session_path(sid)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _list() -> list[dict]:
    if not os.path.isdir(SESSIONS_DIR):
        return []
    out: list[dict] = []
    for name in os.listdir(SESSIONS_DIR):
        if not name.endswith(".json"):
            continue
        p = os.path.join(SESSIONS_DIR, name)
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            out.append({
                "session_id": d.get("session_id", name[:-5]),
                "timestamp": d.get("timestamp", os.path.getmtime(p)),
                "first_message": d.get("first_message", ""),
                "message_count": d.get("message_count") or len(d.get("messages") or []),
                "skills_used": d.get("skills_used") or [],
            })
        except Exception:
            continue
    out.sort(key=lambda s: s["timestamp"], reverse=True)
    return out


def _delete(sid: str) -> bool:
    sid = _safe_sid(sid)
    if not sid:
        return False
    p = _session_path(sid)
    if not os.path.isfile(p):
        return False
    try:
        os.remove(p)
        return True
    except Exception:
        return False


def _route_exists(app, rule: str, method: str) -> bool:
    for r in app.url_map.iter_rules():
        if str(r) == rule and method in (r.methods or set()):
            return True
    return False


def register_session_routes(app) -> int:
    """세션 라우트 등록. 이미 동일 URL+메서드가 있으면 건너뛴다.

    Returns: 새로 등록된 라우트 수.
    """
    registered = 0

    if not _route_exists(app, "/api/harness/session/save", "POST"):
        @app.route("/api/harness/session/save", methods=["POST"], endpoint="ca_session_save")
        def _ca_session_save():
            data = request.get_json(force=True, silent=True) or {}
            try:
                sid = _save(data)
            except Exception as e:
                return jsonify({"error": f"저장 실패: {e}"}), 500
            return jsonify({"session_id": sid, "status": "saved"})
        registered += 1

    if not _route_exists(app, "/api/harness/session/load/<session_id>", "GET"):
        @app.route("/api/harness/session/load/<session_id>", endpoint="ca_session_load")
        def _ca_session_load(session_id):
            d = _load(session_id)
            if d is None:
                return jsonify({"error": "세션 없음"}), 404
            return jsonify(d)
        registered += 1

    if not _route_exists(app, "/api/harness/session/list", "GET"):
        @app.route("/api/harness/session/list", endpoint="ca_session_list")
        def _ca_session_list():
            return jsonify({"sessions": _list()})
        registered += 1

    if not _route_exists(app, "/api/harness/session/delete/<session_id>", "DELETE"):
        @app.route("/api/harness/session/delete/<session_id>", methods=["DELETE"], endpoint="ca_session_delete")
        def _ca_session_delete(session_id):
            ok = _delete(session_id)
            if not ok:
                return jsonify({"error": "삭제 실패 또는 세션 없음"}), 404
            return jsonify({"status": "deleted", "session_id": session_id})
        registered += 1

    return registered
