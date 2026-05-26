"""
hermes_proxy.py — Hermes Web 용 미니 LLM 프록시 (새로 작성)

- 127.0.0.1:8765 에서 listen
- 모든 /v1/* 요청을 common.llm.skhynix.com 으로 forward (하드코딩)
- TOKEN.TXT 자동 로드해서 Authorization 헤더 강제 주입
- tools / tool_choice / functions 필드 그대로 통과 (Agent 동작)
- 스트리밍 (SSE) / 비스트리밍 둘 다 지원
- 환경변수 의존 X — 그냥 띄우면 됨
"""
import os
import sys
import json
from flask import Flask, request, Response, jsonify, stream_with_context
import requests

# ─── 고정 설정 ──────────────────────────────────────────────────────────────
UPSTREAM = "http://common.llm.skhynix.com"
LISTEN_HOST = "127.0.0.1"
LISTEN_PORT = 8765
TIMEOUT_S = 600
VERIFY_SSL = False

# TOKEN.TXT 자동 로드 (있으면 Auth 헤더로 강제 주입)
_TOKEN = ""
_TOKEN_CANDIDATES = [
    r"C:\연구과제\CODE\데모스_분석툴\scientific-assistant\TOKEN.TXT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "TOKEN.TXT"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "TOKEN.TXT"),
]
for path in _TOKEN_CANDIDATES:
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                _TOKEN = f.read().strip()
            print(f"[PROXY] TOKEN.TXT 로드: {path}  (len={len(_TOKEN)})")
            break
        except Exception as e:
            print(f"[PROXY] ! TOKEN.TXT 읽기 실패 ({path}): {e}", file=sys.stderr)

if not _TOKEN:
    print("[PROXY] ⚠ TOKEN.TXT 못 찾음 — 요청의 Authorization 그대로 forward")

app = Flask(__name__)


def _build_headers():
    """클라이언트 헤더 복사 + TOKEN.TXT 가 있으면 Authorization 덮어쓰기."""
    h = {}
    for k, v in request.headers.items():
        if k.lower() in ("host", "content-length", "connection"):
            continue
        h[k] = v
    if _TOKEN:
        h["Authorization"] = f"Bearer {_TOKEN}"
    return h


def _mask_auth(h):
    a = h.get("Authorization", "(none)")
    if a.startswith("Bearer "):
        tok = a[7:]
        return f"Bearer {tok[:6]}...{tok[-4:]} (len={len(tok)})"
    return a


@app.route("/v1/<path:subpath>", methods=["GET", "POST", "PUT", "DELETE"])
def forward(subpath):
    url = f"{UPSTREAM}/v1/{subpath}"
    body = request.get_json(silent=True) or {}
    is_stream = bool(body.get("stream", False))
    model = body.get("model", "?")
    method = request.method
    headers = _build_headers()

    print(f"[PROXY] {method} /v1/{subpath}  model={model!r}  stream={is_stream}  auth={_mask_auth(headers)}")

    try:
        r = requests.request(
            method, url,
            headers=headers,
            json=body if body else None,
            stream=is_stream,
            timeout=TIMEOUT_S,
            verify=VERIFY_SSL,
        )
    except Exception as e:
        print(f"[PROXY] ✗ upstream 호출 실패: {e}")
        return jsonify({"error": str(e)}), 502

    if r.status_code >= 400:
        try:
            print(f"[PROXY] ✗ upstream {r.status_code}: {r.text[:300]}")
        except Exception:
            pass

    if is_stream:
        def gen():
            for chunk in r.iter_content(chunk_size=None):
                if chunk:
                    yield chunk
        excluded = {"content-encoding", "transfer-encoding", "content-length", "connection"}
        resp_headers = [(k, v) for k, v in r.headers.items() if k.lower() not in excluded]
        return Response(stream_with_context(gen()), status=r.status_code, headers=resp_headers)
    else:
        return Response(
            r.content,
            status=r.status_code,
            content_type=r.headers.get("Content-Type", "application/json"),
        )


@app.route("/health")
def health():
    return jsonify({
        "ok": True,
        "upstream": UPSTREAM,
        "token_loaded": bool(_TOKEN),
        "token_len": len(_TOKEN),
    })


if __name__ == "__main__":
    print("=" * 60)
    print(f"  Hermes Proxy (clean)")
    print(f"  Listen:    http://{LISTEN_HOST}:{LISTEN_PORT}")
    print(f"  Upstream:  {UPSTREAM}")
    print(f"  Token:     {'로드 (' + str(len(_TOKEN)) + '자)' if _TOKEN else '없음'}")
    print(f"  통과 필드: tools, tool_choice, functions 전부 forward")
    print("=" * 60)
    app.run(host=LISTEN_HOST, port=LISTEN_PORT, debug=False, threaded=True)
