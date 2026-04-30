"""
demos_v1/routes_chat_stream.py - SSE streaming endpoint for Code Assistant

Phase 1 범위:
- 단일 API 모델만 (disable_fallback=True 가정)
- 폴백·멀티에이전트·드로잉/ppt 자동라우팅·지식검색 없음
- 응답을 토큰 단위 SSE 로 클라이언트에 푸시

엔드포인트: POST /api/chat/stream
요청 본문 (예):
  {
    "env": ["kimi-k25"],
    "messages": [...],
    "skills": ["agent-python-pro"],
    "system_prompt": "...",
    "max_tokens": 16384,
    "effort": 2,
    "disable_fallback": true
  }

응답: text/event-stream
이벤트 형식:
  data: {"delta": "토큰 텍스트"}\n\n      # 매 토큰마다
  data: {"done": true, "model_used": "...", "loaded_skills": [...], "system_prompt_length": 1234, "elapsed_ms": 12345}\n\n
또는 에러 시:
  data: {"error": "메시지", "status": 500}\n\n
"""
import os
import json
import time
import warnings
import requests as req
from flask import request, Response, stream_with_context

from demos_v1.config import API_TOKEN
from demos_v1.models import MODEL_REGISTRY, ENV_TO_REGISTRY
from demos_v1.skills import load_skill_content
from demos_v1.engine import _build_agent_system_prompt
from demos_v1.knowledge import search_knowledge
from demos_v1.quality import _sanitize_knowledge_content

# requests SSL warning suppression (페쇄망 self-signed)
warnings.filterwarnings("ignore", message="Unverified HTTPS request")


def _sse(obj):
    """JSON 객체를 SSE data 라인으로 직렬화."""
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def register_chat_stream_routes(app):
    """SSE 스트리밍 채팅 엔드포인트 등록."""

    @app.route("/api/chat/stream", methods=["POST"])
    def api_chat_stream():
        try:
            data = request.get_json(force=True, silent=True) or {}
        except Exception as e:
            return Response(_sse({"error": f"잘못된 요청: {e}", "status": 400}),
                            mimetype="text/event-stream", status=400)

        envs = data.get("env", ["auto"]) or ["auto"]
        env_id = envs[0] if isinstance(envs, list) else str(envs)

        # GGUF 는 Phase 1 미지원
        if env_id.startswith("gguf-") or env_id == "auto":
            return Response(
                _sse({"error": "스트리밍은 단일 API 모델만 지원합니다. (Phase 1)", "status": 400}),
                mimetype="text/event-stream", status=400,
            )

        reg_key = ENV_TO_REGISTRY.get(env_id)
        if not reg_key or reg_key not in MODEL_REGISTRY:
            return Response(
                _sse({"error": f"알 수 없는 모델: {env_id}", "status": 400}),
                mimetype="text/event-stream", status=400,
            )

        reg = MODEL_REGISTRY[reg_key]
        url = reg.get("url")
        model = reg.get("model")
        if not url or not model:
            return Response(
                _sse({"error": f"모델 설정 불완전: {env_id}", "status": 500}),
                mimetype="text/event-stream", status=500,
            )

        messages = data.get("messages", []) or []

        # 히스토리 트림 (메인 채팅과 동일 정책)
        MAX_HISTORY_MESSAGES = 12
        if len(messages) > MAX_HISTORY_MESSAGES:
            _sys_msgs = [m for m in messages[:2] if m.get("role") == "system"]
            _recent = messages[-MAX_HISTORY_MESSAGES:]
            messages = _sys_msgs + _recent

        skill_ids = data.get("skills", []) or []
        user_system_prompt = (data.get("system_prompt") or "").strip()
        max_tokens = int(data.get("max_tokens") or 0)
        effort = int(data.get("effort", 2))

        # 모델 tier 기반 기본 max_tokens
        if max_tokens <= 0:
            tier = reg.get("cost_tier", "medium")
            max_tokens = 16384 if tier == "high" else (8192 if tier == "medium" else 4096)

        # temperature: effort 0=0.2, 1=0.4, 2=0.7, 3=0.9
        temp_map = [0.2, 0.4, 0.7, 0.9]
        temperature = temp_map[min(max(effort, 0), 3)]

        # 스킬 로딩
        skill_contents = {}
        loaded_skills = []
        for sid in skill_ids:
            try:
                txt = load_skill_content(sid)
                if txt:
                    skill_contents[sid] = txt
                    loaded_skills.append(sid)
            except Exception:
                pass

        # 시스템 프롬프트 빌드 (스킬 + 사용자 추가)
        try:
            agent_system = _build_agent_system_prompt(
                loaded_skills, skill_contents, n_ctx=32768
            )
        except Exception as e:
            return Response(
                _sse({"error": f"시스템 프롬프트 빌드 실패: {e}", "status": 500}),
                mimetype="text/event-stream", status=500,
            )

        # 사용자 system_prompt 가 있으면 앞에 추가
        if user_system_prompt:
            agent_system = user_system_prompt + "\n\n" + agent_system

        # ── knowledge-search 스킬: 도메인 지식 BM25 검색 결과 주입 ──
        # routes_chat.py 의 동작과 동일하게 (수동 선택 시 활성화)
        kb_files_used = []
        last_user_query = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                _c = m.get("content", "")
                if isinstance(_c, str):
                    last_user_query = _c
                elif isinstance(_c, list):
                    last_user_query = "\n".join(p.get("text", "") for p in _c if p.get("type") == "text")
                break

        kb_system_msg = None
        if "knowledge-search" in loaded_skills and last_user_query.strip():
            try:
                _user_id = data.get("user_id", None)
                kb_results = search_knowledge(
                    last_user_query, max_results=10, max_content_chars=4000, user_id=_user_id
                )
                if kb_results:
                    kb_context = "\n\n=== 도메인 지식 검색 결과 ===\n"
                    kb_context += f"검색어: {last_user_query}\n\n"
                    total_chars = 0
                    for r in kb_results:
                        chunk = _sanitize_knowledge_content(r['content'][:4000])
                        if total_chars + len(chunk) > 12000:
                            chunk = chunk[:max(0, 12000 - total_chars)]
                            if not chunk:
                                break
                        kb_context += f"--- 📄 {r['filename']} (관련도: {r['score']}) ---\n"
                        kb_context += chunk + "\n\n"
                        total_chars += len(chunk)
                    _ql = last_user_query.lower()
                    _is_search_only = any(kw in _ql for kw in [
                        "검색해", "검색 해", "찾아봐", "찾아 봐",
                        "뭐있", "뭐 있", "목록", "리스트", "파일명", "문서 목록"
                    ])
                    _is_content_request = any(kw in _ql for kw in [
                        "내용", "관련", "알려", "설명", "분석", "요약", "만들어"
                    ])
                    if _is_search_only and not _is_content_request:
                        kb_context += (
                            "사용자가 '검색'을 요청했습니다. 파일명 목록과 관련도 점수만 간단히 보여주세요.\n"
                            "문서 내용을 분석하거나 요약하지 마세요. 파일명 리스트만 출력하세요.\n"
                        )
                    else:
                        kb_context += (
                            "위 문서를 기반으로 사용자 질문에 답변하세요. 문서에 없는 내용을 지어내지 마세요. "
                            "어떤 문서에서 정보를 찾았는지 출처를 명시하세요.\n"
                        )
                    kb_system_msg = {"role": "system", "content": kb_context}
                    kb_files_used = [r['filename'] for r in kb_results]
                    print(f"  [STREAM/KNOWLEDGE] {len(kb_results)}개 문서 주입, total_chars={total_chars}")
            except Exception as e:
                print(f"[STREAM/KNOWLEDGE] 검색 오류: {e}")

        # 최종 메시지 구성: skills system + knowledge system + 기존 messages
        final_msgs = [{"role": "system", "content": agent_system}]
        if kb_system_msg:
            final_msgs.append(kb_system_msg)
        for m in messages:
            # 다중 system 회피: 사용자가 보낸 system 도 받지만 첫 system 뒤에 user 로 변환
            if m.get("role") == "system":
                final_msgs.append({"role": "user", "content": "[추가 지시] " + str(m.get("content", ""))})
            else:
                final_msgs.append(m)

        api_key = API_TOKEN or data.get("api_key", "")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "model": model,
            "messages": final_msgs,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "tool_choice": "none",
        }

        # disable_fallback=True 면 타임아웃 600s, 아니면 120s
        timeout_s = 600 if data.get("disable_fallback") else 120

        sys_prompt_len = len(agent_system) + (len(kb_system_msg["content"]) if kb_system_msg else 0)
        meta = {
            "model_used": model,
            "loaded_skills": loaded_skills,
            "system_prompt_length": sys_prompt_len,
            "env_id": env_id,
            "knowledge_files": kb_files_used,
        }

        def generate():
            t0 = time.time()
            full_text_parts = []
            try:
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
                    yield _sse({
                        "error": f"HTTP {upstream.status_code}: {body_preview}",
                        "status": upstream.status_code,
                    })
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
                        full_text_parts.append(text_piece)
                        yield _sse({"delta": text_piece})
                    # finish_reason 무시 — [DONE] 또는 stream 종료로 끝남

                # 스트림 정상 종료
                elapsed_ms = int((time.time() - t0) * 1000)
                yield _sse({
                    "done": True,
                    "elapsed_ms": elapsed_ms,
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
                "X-Accel-Buffering": "no",  # nginx 버퍼링 방지
            },
        )
