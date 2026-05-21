"""Hermes ↔ 회사 vLLM(HCP) 프록시

문제: hermes-agent 가 요청 body 에 `tools` 배열 18개를 같이 보내는데
      회사 litellm/vLLM 게이트웨이가 그 schema 를 거부 → HTTP 400.

해결: hermes 가 보내는 모든 OpenAI 호환 요청에서 tools/tool_choice/
      function_call/functions 필드만 제거하고 upstream 게이트웨이로
      그대로 전달.

사용:
  1) TOKEN.TXT 가 같은 폴더에 있는지 확인.
  2) python hermes_proxy.py
     → http://localhost:8765 에서 대기 시작.
  3) hermes config edit 으로 ~/.hermes/config.yaml 의 custom provider URL 을
       http://dev.hcp.llm.skhynix.com/v1
     에서
       http://localhost:8765/v1
     로 변경.
  4) hermes chat -q "안녕"

스트리밍(SSE) / 일반 JSON 응답 둘 다 지원.
"""

import json
import os
import sys

try:
    import requests
    from flask import Flask, request, Response, stream_with_context, jsonify
except ImportError:
    print("[!] 의존성 설치 필요: pip install flask requests", file=sys.stderr)
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────────────────────
UPSTREAM = os.environ.get("HERMES_UPSTREAM", "http://dev.hcp.llm.skhynix.com")
LISTEN_HOST = "127.0.0.1"
LISTEN_PORT = int(os.environ.get("HERMES_PROXY_PORT", "8765"))
TIMEOUT_S = 600
VERIFY_SSL = False  # 사내 인증서 우회

# tools 외에 함께 거부될 수 있는 필드들 — 안전하게 같이 제거
STRIP_FIELDS = (
    "tools",
    "tool_choice",
    "functions",
    "function_call",
    "parallel_tool_calls",
    "response_format",  # 일부 vLLM 빌드에서 거부 (필요 시 주석 처리)
)


app = Flask(__name__)


def _strip_unsupported(body):
    """요청 body 에서 회사 vLLM 이 거부하는 필드 제거. 어떤 게 빠졌는지 로그."""
    if not isinstance(body, dict):
        return body, []
    removed = []
    for k in STRIP_FIELDS:
        if k in body:
            body.pop(k, None)
            removed.append(k)
    return body, removed


def _forward_headers():
    """클라이언트(hermes) 가 보낸 헤더를 그대로 upstream 에 전달."""
    h = {}
    for k, v in request.headers.items():
        if k.lower() in ("host", "content-length", "connection"):
            continue
        h[k] = v
    return h


def _mask_auth(h):
    """디버깅용: Authorization 값 마스킹해서 짧게 보여주기."""
    a = h.get("Authorization") or h.get("authorization") or ""
    if not a:
        return "(none)"
    parts = a.split(" ", 1)
    if len(parts) == 2:
        scheme, val = parts
        if len(val) <= 12:
            tail = val
        else:
            tail = val[:6] + "..." + val[-4:]
        return f"{scheme} {tail} (len={len(val)})"
    return f"(raw, len={len(a)})"


@app.route("/v1/<path:subpath>", methods=["GET", "POST", "PUT", "DELETE"])
def proxy_v1(subpath):
    upstream_url = f"{UPSTREAM}/v1/{subpath}"
    method = request.method
    headers = _forward_headers()

    # 요청 body 가공 (POST/PUT 만 의미 있음)
    body_payload = None
    is_stream = False
    if method in ("POST", "PUT"):
        try:
            body = request.get_json(force=True, silent=True) or {}
        except Exception:
            body = {}
        body, removed = _strip_unsupported(body)
        is_stream = bool(body.get("stream"))
        print(
            f"[PROXY] {method} /v1/{subpath} "
            f"stream={is_stream} "
            f"model={body.get('model')!r} "
            f"messages={len(body.get('messages') or [])} "
            f"removed={removed}"
        )
        print(f"[PROXY] auth = {_mask_auth(headers)}")
        # 추가 헤더 중 흥미로운 것만 골라서 보여줌
        extra = {k: v for k, v in headers.items()
                 if k.lower() not in ("authorization", "content-type", "user-agent", "accept", "accept-encoding")}
        if extra:
            print(f"[PROXY] extra headers = {list(extra.keys())}")
        body_payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    else:
        print(f"[PROXY] {method} /v1/{subpath}")

    # upstream 호출
    try:
        r = requests.request(
            method=method,
            url=upstream_url,
            headers=headers,
            data=body_payload,
            params=request.args,
            timeout=TIMEOUT_S,
            verify=VERIFY_SSL,
            stream=is_stream,
        )
    except requests.RequestException as e:
        print(f"[PROXY] ! upstream 실패: {e}")
        return jsonify({"error": f"proxy upstream failure: {e}"}), 502

    # SSE 스트림이면 그대로 흘려보냄
    if is_stream and r.headers.get("content-type", "").startswith("text/event-stream"):
        def _gen():
            try:
                for line in r.iter_lines(decode_unicode=True):
                    if line is None:
                        continue
                    yield (line + "\n").encode("utf-8")
                # SSE 종단 이벤트
                yield b"\n"
            finally:
                r.close()

        resp = Response(stream_with_context(_gen()), status=r.status_code, mimetype="text/event-stream")
        resp.headers["Cache-Control"] = "no-cache"
        resp.headers["X-Accel-Buffering"] = "no"
        return resp

    # 일반 응답 — 그대로 전달
    excluded = {"content-encoding", "transfer-encoding", "content-length", "connection"}
    out_headers = [(k, v) for k, v in r.headers.items() if k.lower() not in excluded]

    # 400 시 본문 로그 (디버깅용)
    if r.status_code >= 400:
        print(f"[PROXY] upstream {r.status_code}: {r.text[:600]}")

    return Response(r.content, status=r.status_code, headers=out_headers)


@app.route("/")
def root():
    return jsonify({
        "name": "hermes-vllm-proxy",
        "upstream": UPSTREAM,
        "listen": f"http://{LISTEN_HOST}:{LISTEN_PORT}",
        "strips": list(STRIP_FIELDS),
        "usage": "hermes config 의 base URL 을 http://localhost:8765/v1 로 바꾸세요.",
    })


if __name__ == "__main__":
    print(f"[PROXY] 시작: http://{LISTEN_HOST}:{LISTEN_PORT}  →  {UPSTREAM}")
    print(f"[PROXY] 제거할 필드: {list(STRIP_FIELDS)}")
    print(f"[PROXY] hermes 에는 base URL 을 http://{LISTEN_HOST}:{LISTEN_PORT}/v1 로 설정하세요.")
    app.run(host=LISTEN_HOST, port=LISTEN_PORT, threaded=True, debug=False)
