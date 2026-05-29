"""
code_assist_v1/routes_chat_stream.py - SSE 스트리밍 채팅

엔드포인트: POST /api/code/chat/stream

특징:
- API 모델: 업스트림 OpenAI 호환 /v1/chat/completions 의 stream=True 응답을 토큰별로 재포장.
- GGUF 모델: llama-cpp-python 의 create_chat_completion(stream=True) 응답을 토큰별로 재포장.
- 컨텍스트 조립: prompts.CODING_SYSTEM_PROMPT + 활성 스킬 + (옵션) 도메인 지식 + (옵션) 워크스페이스 첨부.
- knowledge-search 는 수동 토글 (req body 의 enable_knowledge=true).
"""
from __future__ import annotations
import json
import time
import warnings

import requests as req
from flask import request, Response, stream_with_context

from code_assist_v1.config import (
    API_TOKEN,
    MODEL_REGISTRY,
    ENV_TO_REGISTRY,
    ENV_CONFIG,
    DEFAULT_N_CTX,
    MAX_HISTORY_TURNS,
    get_default_model,
)
from code_assist_v1.engine import (
    build_coding_system_prompt,
    build_knowledge_block,
    build_workspace_block,
    trim_message_history,
)
from code_assist_v1 import knowledge_store as ks

warnings.filterwarnings("ignore", message="Unverified HTTPS request")


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _last_user_query(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content", "")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                return "\n".join(p.get("text", "") for p in c if p.get("type") == "text")
    return ""


def _is_search_only_intent(query: str) -> bool:
    ql = query.lower()
    only_kw = ["검색해", "찾아봐", "뭐있", "목록", "리스트", "파일명", "문서 목록"]
    content_kw = ["내용", "관련", "알려", "설명", "분석", "요약", "만들어"]
    return any(k in ql for k in only_kw) and not any(k in ql for k in content_kw)


def _resolve_model(model_id: str | None) -> tuple[dict | None, str]:
    """model_id 또는 env_id 를 받아 모델 설정 반환.

    통합 모드: demos_v1.models 우선 조회 (demos_v1과 같은 모델 풀 공유).
    단독 모드 폴백: code_assist_v1 자체 config 조회.

    Returns:
        (config_dict, kind) — kind ∈ {"api", "gguf"}; 실패 시 (None, "")
    """
    if not model_id:
        model_id = get_default_model()
    if not model_id:
        return None, ""

    # 1) demos_v1.models 우선 조회 (통합 모드)
    try:
        from demos_v1.models import (
            MODEL_REGISTRY as _DV_MR,
            ENV_CONFIG as _DV_EC,
            ENV_TO_REGISTRY as _DV_E2R,
        )
        if model_id.startswith("gguf-"):
            ecfg = _DV_EC.get(model_id)
            if ecfg:
                return ecfg, "gguf"
        else:
            if model_id in _DV_MR:
                return _DV_MR[model_id], "api"
            reg_key = _DV_E2R.get(model_id)
            if reg_key and reg_key in _DV_MR:
                return _DV_MR[reg_key], "api"
    except ImportError:
        pass

    # 2) 폴백: 자체 config (단독 실행 호환)
    if model_id.startswith("gguf-"):
        ecfg = ENV_CONFIG.get(model_id)
        if not ecfg:
            return None, ""
        return ecfg, "gguf"
    if model_id in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_id], "api"
    reg_key = ENV_TO_REGISTRY.get(model_id)
    if reg_key and reg_key in MODEL_REGISTRY:
        return MODEL_REGISTRY[reg_key], "api"
    return None, ""


def _stream_api(
    cfg: dict,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    timeout_s: int,
):
    """API 모델 SSE 스트림 제너레이터."""
    url = cfg.get("url")
    model = cfg.get("model")
    headers = {"Content-Type": "application/json"}
    if API_TOKEN:
        headers["Authorization"] = f"Bearer {API_TOKEN}"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "tool_choice": "none",
    }
    upstream = req.post(
        url, headers=headers, json=payload,
        stream=True, timeout=timeout_s, verify=False,
    )
    if upstream.status_code >= 400:
        body_preview = ""
        try:
            body_preview = upstream.text[:500]
        except Exception:
            pass
        yield _sse({"error": f"HTTP {upstream.status_code}: {body_preview}", "status": upstream.status_code})
        return

    for raw_line in upstream.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        if not raw_line.startswith("data:"):
            continue
        chunk = raw_line[5:].strip()
        if chunk == "[DONE]":
            break
        if not chunk:
            continue
        try:
            ev = json.loads(chunk)
        except Exception:
            continue
        choices = ev.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        text_piece = delta.get("content") or ""
        if text_piece:
            yield _sse({"delta": text_piece})


def _stream_gguf(
    messages: list[dict],
    temperature: float,
    max_tokens: int,
):
    """GGUF 모델 SSE 스트림 — demos_v1.utils.gguf_model 공유 (통합 모드).

    demos_v1이 부팅 시 로드한 모델 인스턴스를 그대로 사용하여 메모리 중복 방지.
    Qwen3 모델이면 /no_think 자동 주입 (reasoning 토큰 낭비 차단).
    """
    import demos_v1.utils as _u
    if _u.gguf_model is None:
        yield _sse({"error": "GGUF 모델이 로드되지 않았습니다 (demos_v1 부팅 시 자동 로드됨)", "status": 503})
        return
    # Qwen3 /no_think 자동 주입 (demos_v1.gguf의 헬퍼 재사용)
    try:
        from demos_v1.gguf import _inject_no_think_for_qwen3
        _model_path = getattr(_u, "gguf_loaded_path", "") or ""
        messages = _inject_no_think_for_qwen3(messages, _model_path)
    except Exception:
        pass  # 헬퍼 없거나 실패해도 스트림은 진행
    try:
        for chunk in _u.gguf_model.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        ):
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            content = delta.get("content") or ""
            if content:
                yield _sse({"delta": content})
    except Exception as e:
        yield _sse({"error": f"GGUF 스트리밍 오류: {e}", "status": 500})


def register_chat_stream_routes(app):

    @app.route("/api/code/chat/stream", methods=["POST"])
    def api_code_chat_stream():
        try:
            data = request.get_json(force=True, silent=True) or {}
        except Exception as e:
            return Response(_sse({"error": f"잘못된 요청: {e}", "status": 400}),
                            mimetype="text/event-stream", status=400)

        # 모델 선택
        model_id = data.get("model") or data.get("env") or None
        if isinstance(model_id, list):
            model_id = model_id[0] if model_id else None
        cfg, kind = _resolve_model(model_id)
        if not cfg:
            return Response(
                _sse({"error": f"알 수 없는 모델: {model_id}", "status": 400}),
                mimetype="text/event-stream", status=400,
            )

        # 메시지·옵션
        messages = data.get("messages", []) or []
        skill_ids = data.get("skills", []) or []
        user_extra = (data.get("system_prompt") or "").strip()
        max_tokens = int(data.get("max_tokens") or 0)
        effort = int(data.get("effort", 2))
        temperature = [0.2, 0.4, 0.7, 0.9][min(max(effort, 0), 3)]

        if max_tokens <= 0:
            tier = cfg.get("cost_tier", "medium")
            max_tokens = 16384 if tier == "high" else (8192 if tier == "medium" else 4096)
            if kind == "gguf":
                max_tokens = min(max_tokens, 4096)  # GGUF 는 보수적으로

        # 히스토리 트림
        messages = trim_message_history(messages, max_turns=MAX_HISTORY_TURNS)

        # 시스템 프롬프트 (코딩 페르소나 + 스킬 본문)
        try:
            system_text = build_coding_system_prompt(
                skill_ids=skill_ids, user_extra=user_extra, n_ctx=DEFAULT_N_CTX,
            )
        except Exception as e:
            return Response(
                _sse({"error": f"시스템 프롬프트 빌드 실패: {e}", "status": 500}),
                mimetype="text/event-stream", status=500,
            )

        # 사고 모드 (think_mode 체크박스): True 면 모델이 reasoning 토큰 생성하도록 허용
        # 시스템 프롬프트 끝에 사고 지시 문구 추가 → demos_v1.gguf 헬퍼가 이 문구 검출 시
        # /no_think 자동 주입을 건너뜀 (Qwen3 reasoning 활성화).
        # think_mode=False(기본)면 기존 동작 — /no_think 자동 주입 → 빠른 답변.
        if bool(data.get("think_mode", False)):
            system_text = system_text + (
                "\n\n답변 전에 반드시 <think>...</think> 태그 안에서 충분히 사고한 후 답변하세요. "
                "사고 내용도 한국어로 작성하세요."
            )

        # 도메인 지식 (수동 토글)
        kb_block = None
        kb_files: list[str] = []
        enable_knowledge = bool(data.get("enable_knowledge", False))
        if enable_knowledge:
            q = _last_user_query(messages)
            if q.strip():
                _kb_uid = (data.get("user_id") or "").strip() or None
                results = ks.search(
                    query=q,
                    user_id=_kb_uid,
                    max_results=int(data.get("kb_max_results", 8)),
                    max_content_chars=4000,
                )
                kb_files = [r["filename"] for r in results]
                kb_block = build_knowledge_block(
                    query=q,
                    knowledge_results=results,
                    intent_search_only=_is_search_only_intent(q),
                )

        # 워크스페이스 첨부
        ws_block = None
        ws_files = data.get("workspace_files") or []
        if ws_files:
            ws_block = build_workspace_block(ws_files)

        # 최종 메시지 조립
        # 일부 vLLM 백엔드(Qwen3.6 등)의 채팅 템플릿은 system 메시지를 정확히 1개,
        # 맨 앞에만 허용한다 ("System message must be at the beginning").
        # 따라서 knowledge/workspace 블록을 별도 system 메시지로 추가하지 않고
        # 메인 system 프롬프트에 병합하여 system 메시지를 항상 1개로 유지한다.
        system_chunks = [system_text]
        if kb_block:
            system_chunks.append(kb_block["content"])
        if ws_block:
            system_chunks.append(ws_block["content"])
        final_msgs: list[dict] = [{"role": "system", "content": "\n\n".join(system_chunks)}]
        for m in messages:
            if m.get("role") == "system":
                final_msgs.append({"role": "user", "content": "[추가 지시] " + str(m.get("content", ""))})
            else:
                final_msgs.append(m)

        timeout_s = 600 if data.get("disable_fallback", True) else 120

        meta = {
            "model_used": cfg.get("model") or cfg.get("name"),
            "model_kind": kind,
            "loaded_skills": skill_ids,
            "knowledge_files": kb_files,
            "workspace_files": [f.get("filename") for f in ws_files] if ws_files else [],
            "system_prompt_length": len(system_text)
                + (len(kb_block["content"]) if kb_block else 0)
                + (len(ws_block["content"]) if ws_block else 0),
        }

        def generate():
            t0 = time.time()
            try:
                if kind == "gguf":
                    yield from _stream_gguf(final_msgs, temperature, max_tokens)
                else:
                    yield from _stream_api(cfg, final_msgs, temperature, max_tokens, timeout_s)

                yield _sse({
                    "done": True,
                    "elapsed_ms": int((time.time() - t0) * 1000),
                    **meta,
                })
            except req.exceptions.Timeout:
                yield _sse({"error": f"타임아웃 ({timeout_s}s)", "status": 504})
            except req.exceptions.ConnectionError as e:
                yield _sse({"error": f"연결 실패: {e}", "status": 502})
            except Exception as e:
                yield _sse({"error": f"스트리밍 에러: {e}", "status": 500})

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
