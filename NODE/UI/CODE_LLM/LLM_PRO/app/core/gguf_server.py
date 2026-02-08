#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
최소 OpenAI 호환 HTTP 서버 - aider가 GGUF 모델과 통신할 때 사용
데몬 스레드로 실행되며 /v1/chat/completions, /v1/models만 제공
"""

import json
import time
import uuid
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

logger = logging.getLogger(__name__)

# 글로벌 참조 (핸들러에서 접근)
_llm_provider = None
_current_model = "local-gguf"


class GGUFRequestHandler(BaseHTTPRequestHandler):
    """OpenAI 호환 엔드포인트 핸들러"""

    def log_message(self, format, *args):
        # 기본 로깅 억제 (너무 시끄러움)
        pass

    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def _send_sse(self, generator):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        for chunk in generator:
            self.wfile.write(chunk.encode("utf-8"))
            self.wfile.flush()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self):
        if self.path == "/v1/models":
            models = []
            if _current_model:
                models.append({
                    "id": _current_model,
                    "object": "model",
                    "owned_by": "local"
                })
            self._send_json({"object": "list", "data": models})
        else:
            self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._send_json({"error": "Not found"}, 404)
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body.decode("utf-8"))
        except Exception:
            self._send_json({"error": {"message": "Invalid JSON"}}, 400)
            return

        if _llm_provider is None or _llm_provider.local_llm is None:
            # 자동 로드 시도
            if _llm_provider is not None:
                loaded = _llm_provider.load_local_model()
                if not loaded:
                    self._send_json(
                        {"error": {"message": "로컬 모델 로드 실패", "type": "server_error"}},
                        503
                    )
                    return
            else:
                self._send_json(
                    {"error": {"message": "LLM Provider 미설정", "type": "server_error"}},
                    503
                )
                return

        messages = data.get("messages", [])
        max_tokens = data.get("max_tokens", 4000)
        is_stream = data.get("stream", False)

        result = _llm_provider.call_gguf_chat(messages, max_tokens)

        if not result.get("success"):
            self._send_json(
                {"error": {"message": result.get("error", "LLM 호출 실패")}},
                500
            )
            return

        usage = result.get("usage", {})
        content = result["content"]
        p_tokens = usage.get("prompt_tokens", 0) or max(1, len(str(messages)) // 4)
        c_tokens = usage.get("completion_tokens", 0) or max(1, len(content) // 4)
        chat_id = f"chatcmpl-local-{uuid.uuid4().hex[:8]}"
        model_id = _current_model or "local-gguf"

        if is_stream:
            def stream_gen():
                chunk_size = 20
                for i in range(0, len(content), chunk_size):
                    chunk = content[i:i + chunk_size]
                    chunk_data = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_id,
                        "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"

                finish_data = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_id,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": p_tokens,
                        "completion_tokens": c_tokens,
                        "total_tokens": p_tokens + c_tokens
                    }
                }
                yield f"data: {json.dumps(finish_data, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            self._send_sse(stream_gen())
        else:
            self._send_json({
                "id": chat_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": p_tokens,
                    "completion_tokens": c_tokens,
                    "total_tokens": p_tokens + c_tokens
                }
            })


class GGUFServer:
    """aider용 최소 OpenAI 호환 HTTP 서버 (데몬 스레드)"""

    def __init__(self, llm_provider, port=10002):
        global _llm_provider, _current_model
        _llm_provider = llm_provider
        _current_model = llm_provider.config.current_gguf_model
        self.port = port
        self.server = None
        self.thread = None

    def start(self):
        """서버를 데몬 스레드로 시작"""
        try:
            self.server = HTTPServer(("127.0.0.1", self.port), GGUFRequestHandler)
            self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.thread.start()
            logger.info(f"GGUF 서버 시작: http://127.0.0.1:{self.port}")
        except Exception as e:
            logger.error(f"GGUF 서버 시작 실패: {e}")

    def stop(self):
        """서버 종료"""
        if self.server:
            self.server.shutdown()
            logger.info("GGUF 서버 종료")

    def update_model(self, model_key: str):
        """현재 모델 키 업데이트"""
        global _current_model
        _current_model = model_key
