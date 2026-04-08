"""LLM Provider - OpenAI-compatible API client with tool_use and fallback chains.

Reference: SKILL/scientific-assistant/app.py lines 7073-7244
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

import urllib.request
import urllib.error

from .token import load_token, print_token_status, find_token_file
from .models import (
    MODEL_REGISTRY,
    DEFAULT_MODEL,
    get_model_config,
    get_fallback_chain,
    get_max_tokens,
    classify_and_route,
)


@dataclass
class ChatMessage:
    role: str  # "system", "user", "assistant", "tool"
    content: str


@dataclass
class ToolCall:
    """A tool call returned by the LLM."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResponse:
    content: str
    model: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str = "stop"
    fallback_used: bool = False
    attempts: list[str] = field(default_factory=list)


class ProviderError(Exception):
    """Raised when all models fail."""
    pass


def build_tool_definitions() -> list[dict]:
    """Build OpenAI-compatible tool definitions for the LLM."""
    return [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file and return its contents. Use this to examine files the user mentions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path to read"}
                    },
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write content to a file. Creates parent directories if needed.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path to write"},
                        "content": {"type": "string", "description": "Content to write"}
                    },
                    "required": ["path", "content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_directory",
                "description": "List files and folders in a directory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Directory path (default: current dir)", "default": "."}
                    },
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "description": "Execute a shell command (cmd.exe on Windows) and return output. Use for: python scripts, pip, git, dir, type, etc.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command to execute"}
                    },
                    "required": ["command"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_files",
                "description": "Search for files matching a glob pattern (e.g. '*.csv', '**/*.py').",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Glob pattern to match files"}
                    },
                    "required": ["pattern"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "grep",
                "description": "Search file contents for a text pattern.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Text pattern to search for"},
                        "path": {"type": "string", "description": "Directory to search in (default: current dir)", "default": "."}
                    },
                    "required": ["pattern"]
                }
            }
        },
    ]


TOOL_DEFINITIONS = build_tool_definitions()


@dataclass
class Provider:
    """OpenAI-compatible LLM API provider with token auth, tool_use, and fallback chains."""

    token: str = ""
    default_model: str = DEFAULT_MODEL
    timeout: int = 120
    max_retries: int = 6
    _initialized: bool = False

    def __post_init__(self) -> None:
        if not self.token:
            self.token = load_token()
        self._initialized = True

    @property
    def has_token(self) -> bool:
        return bool(self.token)

    def print_status(self) -> None:
        """Print provider status at startup."""
        print_token_status(self.token, find_token_file())
        if self.has_token:
            model_cfg = get_model_config(self.default_model)
            if model_cfg:
                print(f"  Default model: {model_cfg['name']}")

    def chat(
        self,
        messages: list[ChatMessage] | list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        tools: list[dict] | None = None,
        auto_route: bool = True,
    ) -> ChatResponse:
        """Send chat completion request with tool_use support and automatic fallback."""
        if not self.has_token:
            raise ProviderError(
                "No API token. Place your key in D:\\openharness\\TOKEN.TXT"
            )

        # Normalize messages
        msg_dicts = []
        for m in messages:
            if isinstance(m, ChatMessage):
                msg_dicts.append({"role": m.role, "content": m.content})
            else:
                msg_dicts.append(m)

        # Auto-route model selection
        if model is None and auto_route:
            last_user = ""
            has_images = False
            for m in reversed(msg_dicts):
                if m["role"] == "user":
                    if isinstance(m["content"], str):
                        last_user = m["content"]
                    elif isinstance(m["content"], list):
                        parts = [p.get("text", "") for p in m["content"] if p.get("type") == "text"]
                        last_user = " ".join(parts)
                        has_images = any(p.get("type") == "image_url" for p in m["content"])
                    break
            model = classify_and_route(last_user, has_images)
        elif model is None:
            model = self.default_model

        if max_tokens is None:
            max_tokens = get_max_tokens(model)

        # Build attempt chain: primary + fallbacks
        attempt_chain = [model] + get_fallback_chain(model)
        attempt_chain = attempt_chain[:self.max_retries]

        attempts: list[str] = []
        last_error = ""

        for try_model_key in attempt_chain:
            config = get_model_config(try_model_key)
            if not config:
                continue

            try_url = config["url"]
            try_model_name = config["model"]
            attempts.append(try_model_key)

            try:
                response = self._api_call(
                    url=try_url,
                    model=try_model_name,
                    messages=msg_dicts,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                )

                # Parse tool calls if present
                tool_calls = []
                raw_tool_calls = response.get("tool_calls", [])
                for tc in raw_tool_calls:
                    fn = tc.get("function", {})
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        args = {"raw": fn.get("arguments", "")}
                    tool_calls.append(ToolCall(
                        id=tc.get("id", ""),
                        name=fn.get("name", ""),
                        arguments=args,
                    ))

                return ChatResponse(
                    content=response.get("content", "") or "",
                    model=try_model_key,
                    tool_calls=tool_calls,
                    usage=response.get("usage", {}),
                    finish_reason=response.get("finish_reason", "stop"),
                    fallback_used=len(attempts) > 1,
                    attempts=attempts,
                )
            except Exception as e:
                last_error = str(e)
                time.sleep(1)
                continue

        raise ProviderError(
            f"All {len(attempts)} models failed. "
            f"Last error: {last_error}. "
            f"Attempted: {', '.join(attempts)}"
        )

    def _api_call(
        self,
        url: str,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        tools: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Execute single API call to OpenAI-compatible endpoint."""
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        payload = json.dumps(body).encode("utf-8")

        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body_text = e.read().decode("utf-8", errors="replace") if e.fp else ""
            raise ProviderError(f"HTTP {e.code}: {body_text[:200]}")
        except urllib.error.URLError as e:
            raise ProviderError(f"Connection error: {e.reason}")
        except TimeoutError:
            raise ProviderError(f"Timeout after {self.timeout}s")

        # Parse OpenAI-compatible response
        if "choices" not in data or not data["choices"]:
            raise ProviderError(f"Invalid response format: {json.dumps(data)[:200]}")

        choice = data["choices"][0]
        message = choice.get("message", {})
        content = message.get("content", "") or ""
        tool_calls = message.get("tool_calls", [])

        return {
            "content": content,
            "tool_calls": tool_calls,
            "usage": data.get("usage", {}),
            "finish_reason": choice.get("finish_reason", "stop"),
        }


def create_provider(
    token: str | None = None,
    model: str = DEFAULT_MODEL,
) -> Provider:
    """Factory function to create a Provider instance."""
    return Provider(
        token=token or "",
        default_model=model,
    )
