"""
code_assist_v1 - 코딩 어시스턴트 플랫폼 (Flask)

기존 demos_v1 (포트 10009) 와 분리된 별도 Flask 앱. 포트 10010.
"""
from __future__ import annotations
import os
from flask import Flask, send_from_directory, redirect

from code_assist_v1.config import STATIC_DIR

# 별도 Flask 인스턴스 (demos_v1 와 충돌 회피)
app = Flask(
    "code_assist_v1",
    static_folder=STATIC_DIR,
    static_url_path="/static",
)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB

_routes_registered = False


def create_app() -> Flask:
    global _routes_registered
    if _routes_registered:
        return app

    from code_assist_v1.routes_models import register_model_routes
    from code_assist_v1.routes_skills import register_skill_routes
    from code_assist_v1.routes_knowledge import register_knowledge_routes
    from code_assist_v1.routes_workspace import register_workspace_routes
    from code_assist_v1.routes_chat_stream import register_chat_stream_routes
    from code_assist_v1.routes_sessions import register_session_routes

    register_model_routes(app)
    register_skill_routes(app)
    register_knowledge_routes(app)
    register_workspace_routes(app)
    register_chat_stream_routes(app)

    # 하네스 (요구사항 #7) — scientific-skills + user_skills 통합
    # harness_bridge 가 있으면 세션 라우트를 먼저 등록해서 그 경로를 선점하고,
    # 없거나 실패해도 이어서 자체 라우트가 등록되도록 한다.
    from code_assist_v1.harness_setup import setup_harness, HARNESS_AVAILABLE
    if HARNESS_AVAILABLE:
        ok = setup_harness(app)
        if ok:
            print("  🔧 하네스: skills/ → /api/harness/* 등록 완료")
        else:
            print("  ⚠️  하네스 초기화 실패")
    else:
        print("  ℹ️  harness_bridge 미설치 — 하네스 비활성")

    # 자체 세션 라우트 (harness_bridge 가용 여부와 무관하게 폴백 보장)
    added = register_session_routes(app)
    if added:
        print(f"  💾 세션 라우트 등록: {added}개 (sessions/ 자체 저장)")

    @app.route("/")
    def root_index():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.route("/healthz")
    def healthz():
        return {"status": "ok", "service": "code_assist_v1"}

    _routes_registered = True
    return app
