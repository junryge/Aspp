"""Hermes 프록시 (hermes-web 전용) — GGUF 자체 추론 + HCP forward

흐름:
  Hermes CLI → http://127.0.0.1:8765/v1  (이 프록시)
    ├── HOME  → 자체 GGUF 추론 (llama-cpp-python via gguf_engine)
    └── OFFICE → http://dev.hcp.llm.skhynix.com (HCP vLLM, TOKEN.TXT 주입)

이 프록시 안에 GGUF 가 들어있어서:
  - HOME 환경에선 demos_v1 / hermes-web 등 외부 의존 없이 단독 동작
  - OFFICE 환경에선 HCP 로 forward (tools 제거, JWT 주입)
"""
import os
import sys
import json
import time

try:
    import requests
    from flask import Flask, request, Response, stream_with_context, jsonify
except ImportError:
    print("[!] 의존성 설치 필요: pip install flask requests", file=sys.stderr)
    sys.exit(1)

# GGUF 엔진 (같은 폴더의 gguf_engine.py)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf_engine


# ─────────────────────────────────────────────────────────────────────────────
# 환경 자동 감지
# ─────────────────────────────────────────────────────────────────────────────
HOME_BASE   = r"F:\M14_Q\scientific-assistant"
OFFICE_BASE = r"C:\연구과제\CODE\데모스_분석툴\scientific-assistant"

if os.path.isdir(HOME_BASE):
    ENV_MODE = "HOME"
    SCIENTIFIC_BASE = HOME_BASE
    TOKEN_FILE = os.path.join(HOME_BASE, "TOKEN.TXT")
    HCP_UPSTREAM = None  # HOME 에선 안 씀
elif os.path.isdir(OFFICE_BASE):
    ENV_MODE = "OFFICE"
    SCIENTIFIC_BASE = OFFICE_BASE
    TOKEN_FILE = os.path.join(OFFICE_BASE, "TOKEN.TXT")
    HCP_UPSTREAM = os.environ.get("HERMES_UPSTREAM", "http://common.llm.skhynix.com")
else:
    ENV_MODE = "UNKNOWN"
    SCIENTIFIC_BASE = os.path.dirname(os.path.abspath(__file__))
    TOKEN_FILE = None
    HCP_UPSTREAM = os.environ.get("HERMES_UPSTREAM", "http://common.llm.skhynix.com")

LISTEN_HOST = "127.0.0.1"
LISTEN_PORT = int(os.environ.get("HERMES_PROXY_PORT", "8765"))
TIMEOUT_S = 600
VERIFY_SSL = False

# OFFICE 에서만 TOKEN.TXT 자동 로드
_AUTO_TOKEN = ""
if ENV_MODE == "OFFICE" and TOKEN_FILE and os.path.isfile(TOKEN_FILE):
    try:
        with open(TOKEN_FILE, "r", encoding="utf-8") as _f:
            _AUTO_TOKEN = _f.read().strip()
    except Exception as _e:
        print(f"[PROXY] ! TOKEN.TXT 읽기 실패: {_e}", file=sys.stderr)

# common.llm.skhynix.com 은 tools/tool_choice/function_calling 다 지원함 (검증됨).
# 옛 엔드포인트(dev.hcp) 호환이 필요하면 HERMES_STRIP_TOOLS=1 환경변수로 켤 수 있게.
_STRIP_TOOLS = os.environ.get("HERMES_STRIP_TOOLS", "0") == "1"
if _STRIP_TOOLS:
    STRIP_FIELDS = (
        "tools", "tool_choice", "functions", "function_call",
        "parallel_tool_calls", "response_format",
    )
    print("[PROXY] ⚠ HERMES_STRIP_TOOLS=1 — tool 필드 제거 모드 (Agent 동작 안 함)")
else:
    # tool-call 통과시킴. response_format 은 일부 vLLM 이 거부할 수 있으니 유지.
    STRIP_FIELDS = ("response_format",)


app = Flask(__name__)


def _strip_unsupported(body):
    if not isinstance(body, dict):
        return body, []
    removed = []
    for k in STRIP_FIELDS:
        if k in body:
            body.pop(k, None)
            removed.append(k)
    return body, removed


def _forward_headers():
    h = {}
    for k, v in request.headers.items():
        if k.lower() in ("host", "content-length", "connection"):
            continue
        h[k] = v
    override = os.environ.get("HERMES_PROXY_OVERRIDE_KEY", "").strip()
    if override:
        h["Authorization"] = f"Bearer {override}"
    elif ENV_MODE == "OFFICE" and _AUTO_TOKEN:
        h["Authorization"] = f"Bearer {_AUTO_TOKEN}"
    return h


def _mask_auth(h):
    a = h.get("Authorization") or h.get("authorization") or ""
    if not a:
        return "(none)"
    parts = a.split(" ", 1)
    if len(parts) == 2:
        scheme, val = parts
        tail = val[:6] + "..." + val[-4:] if len(val) > 12 else val
        return f"{scheme} {tail} (len={len(val)})"
    return f"(raw, len={len(a)})"


# ─────────────────────────────────────────────────────────────────────────────
# HOME: GGUF 자체 추론
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_gguf_target(model_id):
    """model_id 를 GGUF 파일에 매핑.
    우선순위:
      1. 파일명/베이스명 정확 매치 (예: 'Qwen3.5-9B.Q5_K_M')
      2. 'gguf-N' 패턴 → 크기 내림차순 N번째 (demos 호환)
      3. 'gguf-current' → 현재 로드된 모델
      4. 현재 로드된 모델 (있으면)
      5. 첫 번째 파일 (=가장 큰 모델)
    """
    files = gguf_engine.find_gguf_files(SCIENTIFIC_BASE)
    if not files:
        return None, files

    if model_id:
        m_lower = model_id.lower().strip()

        # 1. 정확 매치
        for f in files:
            if f["id"].lower() == m_lower or f["name"].lower() == m_lower:
                return f, files

        # 2. gguf-N 패턴 (demos 호환 — 크기 내림차순 인덱스)
        import re as _re
        mm = _re.match(r"^gguf-(\d+)$", m_lower)
        if mm:
            idx = int(mm.group(1))
            if 0 <= idx < len(files):
                return files[idx], files

        # 3. gguf-current
        if m_lower == "gguf-current":
            loaded = gguf_engine.get_loaded_model()
            if loaded.get("loaded"):
                for f in files:
                    if f["path"] == loaded["path"]:
                        return f, files

    # 4. 이미 로드된 모델 우선
    loaded = gguf_engine.get_loaded_model()
    if loaded.get("loaded"):
        for f in files:
            if f["path"] == loaded["path"]:
                return f, files

    # 5. 폴백
    return files[0], files


def _serve_models_local():
    files = gguf_engine.find_gguf_files(SCIENTIFIC_BASE)
    loaded = gguf_engine.get_loaded_model()
    data = []
    for f in files:
        data.append({
            "id": f["id"],
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local-gguf",
            "size_gb": f["size_gb"],
            "loaded": loaded.get("path") == f["path"],
        })
    return jsonify({"object": "list", "data": data})


def _serve_chat_local(body):
    """HOME: gguf_engine 으로 직접 추론."""
    model_id = body.get("model") or ""
    messages = body.get("messages") or []
    temperature = float(body.get("temperature", 0.5))
    max_tokens = int(body.get("max_tokens", 4096))
    stream = bool(body.get("stream", False))

    target, files = _resolve_gguf_target(model_id)
    if target is None:
        return jsonify({"error": "GGUF 파일 못 찾음", "scientific_base": SCIENTIFIC_BASE}), 500

    ok, msg = gguf_engine.load_model(target["path"])
    if not ok:
        return jsonify({"error": msg}), 500

    if stream:
        gen, err = gguf_engine.chat_completion(messages, temperature, max_tokens, stream=True)
        if err:
            return jsonify({"error": err}), 500

        def sse():
            chunk_count = 0
            try:
                for chunk in gen:
                    chunk_count += 1
                    try:
                        # 각 chunk 안전하게 직렬화
                        payload = json.dumps(chunk, ensure_ascii=False, default=str)
                    except Exception as je:
                        print(f"[PROXY/GGUF] chunk #{chunk_count} JSON 직렬화 실패: {je}")
                        # 깨진 chunk 는 빈 delta 로 대체
                        payload = json.dumps({
                            "id": "chatcmpl-fallback", "object": "chat.completion.chunk",
                            "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": None}]
                        })
                    yield f"data: {payload}\n\n"
                # 마지막 finish_reason 보장 (모델이 stop 안 보낸 경우 대비)
                final = json.dumps({
                    "id": "chatcmpl-end", "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                }, ensure_ascii=False)
                yield f"data: {final}\n\n"
                yield "data: [DONE]\n\n"
                print(f"[PROXY/GGUF] stream 완료: {chunk_count} chunks")
            except Exception as e:
                print(f"[PROXY/GGUF] ! stream 중 예외: {type(e).__name__}: {e}")
                # 에러 발생 시에도 정상 종료 처리 — hermes 가 retry 안 하도록
                final = json.dumps({
                    "id": "chatcmpl-err", "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                }, ensure_ascii=False)
                yield f"data: {final}\n\n"
                yield "data: [DONE]\n\n"

        resp = Response(stream_with_context(sse()), mimetype="text/event-stream")
        resp.headers["Cache-Control"] = "no-cache"
        resp.headers["X-Accel-Buffering"] = "no"
        return resp

    resp, err = gguf_engine.chat_completion(messages, temperature, max_tokens, stream=False)
    if err:
        return jsonify({"error": err}), 500
    return jsonify(resp)


# ─────────────────────────────────────────────────────────────────────────────
# OFFICE: HCP forward
# ─────────────────────────────────────────────────────────────────────────────
def _forward_hcp(subpath):
    upstream_url = f"{HCP_UPSTREAM}/v1/{subpath}"
    method = request.method
    headers = _forward_headers()

    body_payload = None
    is_stream = False
    if method in ("POST", "PUT"):
        try:
            body = request.get_json(force=True, silent=True) or {}
        except Exception:
            body = {}
        body, removed = _strip_unsupported(body)
        is_stream = bool(body.get("stream"))
        # Hermes config 의 model 그대로 forward — 임의 치환 없음
        print(f"[PROXY/HCP] {method} /v1/{subpath} stream={is_stream} model={body.get('model')!r} removed={removed}")
        print(f"[PROXY/HCP] auth = {_mask_auth(headers)}")
        body_payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    else:
        print(f"[PROXY/HCP] {method} /v1/{subpath}")

    try:
        r = requests.request(
            method=method, url=upstream_url, headers=headers, data=body_payload,
            params=request.args, timeout=TIMEOUT_S, verify=VERIFY_SSL, stream=is_stream,
        )
    except requests.RequestException as e:
        return jsonify({"error": f"proxy upstream failure: {e}"}), 502

    if is_stream and r.headers.get("content-type", "").startswith("text/event-stream"):
        def _gen():
            try:
                for line in r.iter_lines(decode_unicode=True):
                    if line is None:
                        continue
                    yield (line + "\n").encode("utf-8")
                yield b"\n"
            finally:
                r.close()
        resp = Response(stream_with_context(_gen()), status=r.status_code, mimetype="text/event-stream")
        resp.headers["Cache-Control"] = "no-cache"
        resp.headers["X-Accel-Buffering"] = "no"
        return resp

    excluded = {"content-encoding", "transfer-encoding", "content-length", "connection"}
    out_headers = [(k, v) for k, v in r.headers.items() if k.lower() not in excluded]
    if r.status_code >= 400:
        print(f"[PROXY/HCP] upstream {r.status_code}: {r.text[:600]}")
    return Response(r.content, status=r.status_code, headers=out_headers)


# ─────────────────────────────────────────────────────────────────────────────
# Route — /v1/* 환경에 따라 분기
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/v1/models", methods=["GET"])
def v1_models():
    if ENV_MODE == "HOME":
        return _serve_models_local()
    return _forward_hcp("models")


@app.route("/v1/chat/completions", methods=["POST"])
def v1_chat():
    if ENV_MODE == "HOME":
        body = request.get_json(silent=True) or {}
        # HOME 에선 굳이 forward 안 함 — 그대로 GGUF 추론
        print(f"[PROXY/GGUF] POST /v1/chat/completions stream={body.get('stream')} model={body.get('model')!r}")
        return _serve_chat_local(body)
    return _forward_hcp("chat/completions")


# fallback — 그 외 /v1/* 는 OFFICE 면 forward, HOME 이면 404
@app.route("/v1/<path:subpath>", methods=["GET", "POST", "PUT", "DELETE"])
def v1_other(subpath):
    if ENV_MODE == "HOME":
        return jsonify({"error": f"HOME 모드에서 /v1/{subpath} 미지원"}), 404
    return _forward_hcp(subpath)


@app.route("/")
def root():
    info = {
        "name": "hermes-web-proxy",
        "env_mode": ENV_MODE,
        "listen": f"http://{LISTEN_HOST}:{LISTEN_PORT}",
    }
    if ENV_MODE == "HOME":
        info["mode"] = "GGUF (자체 추론, llama-cpp-python)"
        info["scientific_base"] = SCIENTIFIC_BASE
        info["loaded_model"] = gguf_engine.get_loaded_model()
    else:
        info["mode"] = "FORWARD (HCP vLLM)"
        info["upstream"] = HCP_UPSTREAM
        info["auto_token_loaded"] = bool(_AUTO_TOKEN)
    return jsonify(info)


def _read_hermes_config_model():
    """~/.hermes/config.yaml 에서 model.default 값만 정규식으로 추출.
    YAML 의존성 회피. 못 찾으면 None."""
    cfg = os.path.join(os.path.expanduser("~"), ".hermes", "config.yaml")
    if not os.path.isfile(cfg):
        return None
    try:
        with open(cfg, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception:
        return None
    # model: \n  default: VALUE  형태
    import re as _re
    m = _re.search(r"^model\s*:\s*\n(?:[ \t]+[^\n]*\n)*?[ \t]+default\s*:\s*([^\n#]+)",
                   text, _re.MULTILINE)
    if m:
        return m.group(1).strip().strip('"').strip("'")
    return None


if __name__ == "__main__":
    print("=" * 70)
    print(f"[PROXY] 환경 자동 감지: {ENV_MODE}")
    if ENV_MODE == "HOME":
        print(f"[PROXY]   - 모드: GGUF 자체 추론 (llama-cpp-python)")
        print(f"[PROXY]   - 경로: {HOME_BASE}")
        ggufs = gguf_engine.find_gguf_files(SCIENTIFIC_BASE)
        if ggufs:
            # 1순위: 환경변수, 2순위: ~/.hermes/config.yaml 의 model.default
            preferred = os.environ.get("HERMES_PROXY_GGUF", "").strip()
            source = "환경변수 HERMES_PROXY_GGUF"
            if not preferred:
                preferred = _read_hermes_config_model() or ""
                source = "~/.hermes/config.yaml 의 model.default"

            target = None
            if preferred:
                p_lower = preferred.lower()
                # 1. 파일명/베이스명 정확 매치
                for g in ggufs:
                    if g["id"].lower() == p_lower or g["name"].lower() == p_lower:
                        target = g
                        break
                # 2. gguf-N 패턴 (크기 내림차순 N번째)
                if target is None:
                    import re as _rep
                    mmp = _rep.match(r"^gguf-(\d+)$", p_lower)
                    if mmp:
                        idx = int(mmp.group(1))
                        if 0 <= idx < len(ggufs):
                            target = ggufs[idx]
                            print(f"[PROXY]   - gguf-{idx} 패턴 매칭 → {target['name']}")

            if target is None:
                print(f"[PROXY]   - 시작 모델 미지정 ({source} = {preferred!r}) → 첫 요청 시 로드")
            else:
                print(f"[PROXY]   - Hermes 설정 모델: {target['name']} ({target['size_gb']} GB)")
                print(f"[PROXY]     ({source} = {preferred!r})")
                ok, msg = gguf_engine.load_model(target["path"])
                print(f"[PROXY]     → {'OK' if ok else 'FAIL'}: {msg}")
        else:
            print(f"[PROXY]   ⚠ GGUF 파일 없음!")
    elif ENV_MODE == "OFFICE":
        print(f"[PROXY]   - 모드: HCP FORWARD")
        print(f"[PROXY]   - 업스트림: {HCP_UPSTREAM}")
        print(f"[PROXY]   - JWT 자동주입: {'예 (TOKEN.TXT, len=' + str(len(_AUTO_TOKEN)) + ')' if _AUTO_TOKEN else '아니오'}")
    else:
        print(f"[PROXY]   - 모드: UNKNOWN")
    print("-" * 70)
    print(f"[PROXY] 리스닝: http://{LISTEN_HOST}:{LISTEN_PORT}")
    print(f"[PROXY] 제거 필드(OFFICE forward 시): {list(STRIP_FIELDS)}")
    print(f"[PROXY] hermes config 의 base URL → http://{LISTEN_HOST}:{LISTEN_PORT}/v1")
    print("=" * 70)
    app.run(host=LISTEN_HOST, port=LISTEN_PORT, threaded=True, debug=False)
