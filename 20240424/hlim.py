#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hllm.py — HLLM_CLI · open llm gateway · terminal client

와이어프레임의 클래식 모노크롬 터미널 클라이언트를 진짜 CLI로 구현.
- Splash (모델 선택) → Chat 루프
- 응답 표시 방식 3종: inline / block / card  (wireframe 2A/2B/2C)
- mock 기본, --endpoint 주면 OpenAI-compatible 실제 스트리밍

사용:
    python hllm.py                      # mock 모드, 모델 선택 화면
    python hllm.py --display card       # 카드 모드로 시작
    python hllm.py --endpoint URL --token TOKEN --model qwen3-235b
    HLLM_ENDPOINT=... HLLM_TOKEN=... python hllm.py

채팅 중 슬래시 명령:
    /help        명령어 목록
    /display X   inline | block | card
    /model       모델 다시 선택
    /clear       세션 초기화
    /quit        종료  (^D / ^C 도 동일)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterator, List, Optional

# ── rich ──
try:
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.text import Text
    from rich.live import Live
    from rich.table import Table
    from rich.box import ROUNDED, MINIMAL, SIMPLE, SQUARE, Box
    from rich.padding import Padding
    from rich.rule import Rule
    from rich.prompt import Prompt, IntPrompt
    from rich.align import Align
except ImportError:
    sys.stderr.write(
        "[hllm] 'rich' 패키지가 필요합니다.\n"
        "       pip install rich\n"
    )
    sys.exit(1)

# ─────────────────────────────────────────────
#  COLORS  (wireframes wfColors)
# ─────────────────────────────────────────────
ACCENT = "#D9633B"
GREEN  = "#3F7D4F"
YELLOW = "#C9A227"
BLUE   = "#7DA3C9"
FG     = "#E8E5DC"
MID    = "#C8C2B2"
DIM    = "#9B9B9B"
FADED  = "#7A7A7A"
LOW    = "#5C5C5C"

console = Console(highlight=False)

# 좌측 세로 바만 그리는 커스텀 Box (block 모드 응답 묶음용)
#  rich.Box 는 8줄 × 4문자: top / head / head-div / body / mid-div / row / foot / bottom
#  각 행에서 1번째 문자가 left, 4번째가 right.
LEFT_BAR = Box(
    "    \n"   # top
    "│   \n"   # head
    "    \n"   # head-div
    "│   \n"   # body
    "    \n"   # mid-div
    "│   \n"   # row
    "│   \n"   # foot
    "    \n"   # bottom
)

LOGO = r"""  ┌──────────────────────────────────────────────┐
  │   _   _ _     _     __  __    ____ _     ___ │
  │  | | | | |   | |   |  \/  |  / ___| |   |_ _|│
  │  | |_| | |   | |   | |\/| | | |   | |    | | │
  │  |  _  | |___| |___| |  | | | |___| |___ | | │
  │  |_| |_|_____|_____|_|  |_|  \____|_____|___|│
  │                                              │
  │     open llm gateway · terminal client       │
  └──────────────────────────────────────────────┘"""


# ─────────────────────────────────────────────
#  CONFIG / TOKEN  로딩
# ─────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# config.json 이 없을 때 사용할 mock fallback (5개)
MOCK_MODELS_CONFIG = {
    "_mock": True,
    "models": {
        "llama-3.3-70b-instruct": {"name": "llama-3.3-70b-instruct (mock)", "model": "llama-3.3-70b-instruct",
                                    "url": "https://api.openllm.local/v1/chat/completions",
                                    "context_window": 128000, "priority": 1,
                                    "capabilities": ["text", "general"], "cost_tier": "low"},
        "qwen2.5-coder-32b":      {"name": "qwen2.5-coder-32b (mock)", "model": "qwen2.5-coder-32b",
                                    "url": "https://api.openllm.local/v1/chat/completions",
                                    "context_window": 32000,  "priority": 2,
                                    "capabilities": ["code"],  "cost_tier": "low"},
    },
    "fallback_chains": {},
}


def load_config(path: Optional[str]) -> dict:
    """config.json 로드. 없으면 mock 반환."""
    candidates = []
    if path:
        candidates.append(path)
    candidates += [
        os.path.join(os.getcwd(), "config.json"),
        os.path.join(SCRIPT_DIR, "config.json"),
    ]
    for p in candidates:
        if p and os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                if "models" in cfg:
                    cfg["_path"] = p
                    return cfg
            except Exception as e:
                console.print(f"[{YELLOW}]config.json 로드 실패 ({p}): {e}[/]")
    cfg = dict(MOCK_MODELS_CONFIG)
    cfg["_path"] = None
    return cfg


def load_token(path: Optional[str]) -> Optional[str]:
    """TOKEN.TXT 로드. 없으면 env HLLM_TOKEN 폴백."""
    candidates = []
    if path:
        candidates.append(path)
    candidates += [
        os.path.join(os.getcwd(), "TOKEN.TXT"),
        os.path.join(os.getcwd(), "token.txt"),
        os.path.join(SCRIPT_DIR, "TOKEN.TXT"),
        os.path.join(SCRIPT_DIR, "token.txt"),
    ]
    for p in candidates:
        if p and os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    tok = f.read().strip()
                if tok:
                    return tok
            except Exception:
                pass
    return os.environ.get("HLLM_TOKEN")


def models_sorted(cfg: dict) -> List[tuple]:
    """priority 오름차순(낮을수록 먼저), 동률은 이름순."""
    items = list(cfg.get("models", {}).items())
    items.sort(key=lambda kv: (kv[1].get("priority", 99), kv[0]))
    return items


def cost_tier_color(tier: str) -> str:
    return {"high": ACCENT, "medium": YELLOW, "low": GREEN}.get(tier, FADED)


def cost_tier_label(tier: str) -> str:
    return {"high": "$$$", "medium": "$$", "low": "$"}.get(tier, "?")


# ─────────────────────────────────────────────
#  STATE
# ─────────────────────────────────────────────
@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))
    metrics: dict = field(default_factory=dict)


@dataclass
class State:
    config: dict = field(default_factory=dict)        # 전체 config.json
    model_id: str = ""                                  # config.models 의 키 (예: 'qwen3-coder-480b')
    display: str = "block"
    token: Optional[str] = None
    history: List[Turn] = field(default_factory=list)
    turn_no: int = 0
    use_fallback: bool = True

    @property
    def model_meta(self) -> dict:
        return self.config.get("models", {}).get(self.model_id, {})

    @property
    def model_api_name(self) -> str:
        """API 요청의 'model' 필드 값 (예: Qwen3-Coder-480B-A35B-Instruct)."""
        return self.model_meta.get("model", self.model_id)

    @property
    def model_url(self) -> str:
        """이 모델 전용 endpoint URL."""
        return self.model_meta.get("url", "")

    @property
    def model_display(self) -> str:
        """splash/header 표시용 이름."""
        return self.model_meta.get("name", self.model_id) or self.model_id

    @property
    def is_mock(self) -> bool:
        return self.config.get("_mock") is True or not self.token


# ─────────────────────────────────────────────
#  SPLASH
# ─────────────────────────────────────────────
def show_splash(state: State) -> None:
    """ASCII 로고 + 환경 점검 + 모델 선택."""
    console.clear()
    console.print()
    console.print(Text(LOGO, style=ACCENT))
    console.print()

    console.print(f"[{FADED}]──── 환경 점검 ────[/]")

    # config 상태
    cfg_path = state.config.get("_path")
    if cfg_path:
        console.print(f"[{GREEN}][✓][/] [bold {FG}]config[/]         [{DIM}]{cfg_path}[/]")
    else:
        console.print(f"[{YELLOW}][!][/] [bold {FG}]config[/]         [{DIM}]config.json 미발견 — mock 모드[/]")

    # token 상태
    if state.token:
        masked = (state.token[:6] + "***" + state.token[-4:]) if len(state.token) > 12 else "***"
        console.print(f"[{GREEN}][✓][/] [bold {FG}]auth token[/]     [{DIM}]{masked}[/]")
    else:
        console.print(f"[{YELLOW}][!][/] [bold {FG}]auth token[/]     [{DIM}]TOKEN.TXT 미발견 — mock 모드[/]")

    # endpoint 호스트들
    hosts = sorted({m.get("url", "").split("/v1")[0] for m in state.config.get("models", {}).values() if m.get("url")})
    for h in hosts:
        if h:
            console.print(f"[{GREEN}][✓][/] [bold {FG}]endpoint[/]       [{DIM}]{h}[/]")

    n_models = len(state.config.get("models", {}))
    console.print(f"[{GREEN}][✓][/] [bold {FG}]models[/]         [{FG}]{n_models} models available[/]")

    console.print()
    console.print(f"[{FADED}]──── 모델 선택 (번호 입력 또는 ⏎ 로 1번 선택) ────[/]")
    console.print()

    sorted_models = models_sorted(state.config)

    for i, (mid, m) in enumerate(sorted_models, start=1):
        prio = m.get("priority", 99)
        star = f" [bold {YELLOW}]★[/]" if prio == 1 else ""
        ctx_k = f"{m.get('context_window', 0) // 1000}k"
        tier = m.get("cost_tier", "?")
        tier_c = cost_tier_color(tier)
        tier_l = cost_tier_label(tier)
        caps = ", ".join(m.get("capabilities", []))

        console.print(
            f"  [bold {ACCENT}]{i})[/]  "
            f"[bold {FG}]{m.get('name', mid)}[/]{star}"
            f"   [{FADED}]{ctx_k} · [{tier_c}]{tier_l}[/] · prio {prio}[/]"
        )
        console.print(
            f"      [{DIM}]{m.get('model', mid)}[/]"
            f"   [{FADED}]· {caps}[/]"
        )

    console.print()
    console.print(
        f"  [bold {FG}]⏎[/] connect   "
        f"[bold {FG}]q[/] quit   "
        f"[bold {FG}]i[/] info   "
        f"[bold {FG}]c[/] compare"
    )
    console.print()

    n = len(sorted_models)
    if n == 0:
        console.print(f"[{ACCENT}]config.models 가 비어있습니다. 종료.[/]")
        sys.exit(1)

    # 선택 루프
    while True:
        try:
            raw = console.input(f"[{ACCENT}]select model >[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print(f"\n[{FADED}]bye.[/]")
            sys.exit(0)

        if raw == "" or raw == "1":
            chosen_id, chosen = sorted_models[0]
            break
        if raw.lower() in ("q", "quit", "exit"):
            sys.exit(0)
        if raw.lower() == "c":
            show_compare()
            continue
        if raw.lower() == "i":
            show_model_info(sorted_models)
            continue
        if raw.isdigit() and 1 <= int(raw) <= n:
            chosen_id, chosen = sorted_models[int(raw) - 1]
            break
        console.print(f"[{YELLOW}]잘못된 입력입니다. 1~{n} 또는 q/c/i.[/]")

    state.model_id = chosen_id
    console.print()
    console.print(f"[{GREEN}][✓][/] connecting to [bold {FG}]{chosen.get('name', chosen_id)}[/]…")
    console.print(f"   [{FADED}]→ {chosen.get('url', '')}[/]")
    time.sleep(0.25)
    console.print()


def show_model_info(sorted_models: List[tuple]) -> None:
    """모델 상세 정보 테이블."""
    t = Table(box=SIMPLE, border_style=FADED, header_style=f"bold {FG}", padding=(0, 1))
    t.add_column("#", style=ACCENT, no_wrap=True)
    t.add_column("name", style=FG)
    t.add_column("api model", style=DIM)
    t.add_column("ctx", style=FADED, justify="right")
    t.add_column("prio", style=FADED, justify="right")
    t.add_column("cost", justify="center")
    t.add_column("caps", style=FADED)
    for i, (mid, m) in enumerate(sorted_models, start=1):
        tier = m.get("cost_tier", "?")
        t.add_row(
            str(i),
            m.get("name", mid),
            m.get("model", mid),
            f"{m.get('context_window', 0) // 1000}k",
            str(m.get("priority", "?")),
            f"[{cost_tier_color(tier)}]{cost_tier_label(tier)}[/]",
            ", ".join(m.get("capabilities", [])),
        )
    console.print(t)
    console.print()


def show_compare() -> None:
    """응답 표시 방식 비교 매트릭스."""
    t = Table(
        title=f"[bold {FG}]응답 표시 방식 — 비교[/]\n[{FADED}]같은 대화, 세 가지 무게감[/]",
        title_justify="left",
        box=SIMPLE,
        border_style=FADED,
        header_style=f"bold {FG}",
        padding=(0, 1),
    )
    t.add_column("", style=ACCENT, no_wrap=True)
    t.add_column("A · 인라인", style=FG)
    t.add_column("B · 블록  ★", style=FG)
    t.add_column("C · 카드", style=FG)
    t.add_row("시각적 무게", "가장 가벼움\nprompt 흐름", "중간\n좌측 컬러바", "가장 무거움\n컨테이너 박스")
    t.add_row("메트릭 노출", "한 줄 ↳ 요약", "응답 푸터", "4분할 그리드\n(가장 풍부)")
    t.add_row("액션 위치",   "전역 단축키만", "응답 푸터", "카드 헤더 상시")
    t.add_row("긴 대화",     f"[{GREEN}]◉ 최적[/]", "○ 적당", f"[{YELLOW}]△ 스크롤 부담[/]")
    t.add_row("비교/분석",   f"[{YELLOW}]△ 묻힘[/]", "○ 적당", f"[{GREEN}]◉ 최적[/]")
    t.add_row("추천 상황",   "빠른 Q&A,\nshell 보조", "일반 대화,\n코딩 세션 ⭐", "벤치마킹,\n모델 평가")
    console.print(t)
    console.print(
        f"[{ACCENT}]→[/] [{FG}]다음 단계: B(블록)을 기본으로, /display 로 A/C 토글.[/]"
    )
    console.print()


# ─────────────────────────────────────────────
#  RENDERERS — wireframe 2A / 2B / 2C
# ─────────────────────────────────────────────
def session_header(state: State) -> Text:
    in_tok  = sum(t.metrics.get("in_tokens",  0) for t in state.history)
    out_tok = sum(t.metrics.get("out_tokens", 0) for t in state.history)
    turns   = sum(1 for t in state.history if t.role == "assistant")
    ctx_k   = state.model_meta.get("context_window", 0) // 1000 or "?"
    mode    = "mock" if state.is_mock else "live"
    left  = f"session: [bold {FG}]{state.model_display}[/] · ctx {ctx_k}k · {mode}"
    right = f"turns {turns} · in {in_tok:,} · out {out_tok:,} tok"
    text = Text.from_markup(f"[{DIM}]{left}[/]    [{DIM}]{right}[/]")
    return text


def render_inline(turn: Turn, response: str, metrics: dict, streaming: bool) -> Group:
    """2A: prompt 다음 줄에 응답이 그냥 흐른다."""
    parts = []
    parts.append(Text.from_markup(
        f"[{ACCENT}]❯[/] [{DIM}]you[/] [{LOW}]›[/] [{FG}]{turn.content}[/]"
    ))
    body = Text.from_markup(f"[{MID}]{response}[/]" + (f"[{FG} on {LOW}] [/]" if streaming else ""))
    parts.append(Padding(body, (0, 0, 0, 4)))
    if not streaming and metrics:
        m = (
            f"[{GREEN}]↳[/] {metrics.get('out_tokens', 0)} tok · "
            f"{metrics.get('time', 0):.1f}s · "
            f"${metrics.get('cost', 0):.4f} · "
            f"{metrics.get('speed', 0)} t/s"
        )
        parts.append(Padding(Text.from_markup(m), (0, 0, 0, 4)))
    parts.append(Text(""))
    return Group(*parts)


def render_block(state: State, turn: Turn, response: str, metrics: dict, streaming: bool) -> Group:
    """2B: 좌측 컬러바 + 응답 + 푸터 메트릭."""
    parts: list = []
    parts.append(Text.from_markup(f"[{ACCENT}]❯[/] [{FG}]{turn.content}[/]"))
    parts.append(Padding(
        Text.from_markup(f"[{LOW}]{turn.timestamp} · you · {len(turn.content)} chars[/]"),
        (0, 0, 0, 2)
    ))
    parts.append(Text(""))

    bar  = YELLOW if streaming else GREEN
    head = f"▼ {state.model_display.upper()} · " + ("streaming..." if streaming else f"response #{state.turn_no}")
    inner_lines: list = [
        Text.from_markup(f"[bold {bar}]{head}[/]"),
        Text(""),
        Text.from_markup(f"[{MID}]{response}[/]" + (f"[{FG} on {LOW}] [/]" if streaming else "")),
    ]
    if not streaming and metrics:
        inner_lines.append(Text(""))
        foot = (
            f"[{bar}]●[/] {metrics.get('out_tokens', 0)} tok   "
            f"{metrics.get('time', 0):.1f}s   "
            f"{metrics.get('speed', 0)} t/s   "
            f"${metrics.get('cost', 0):.4f}    "
            f"[{FADED}][[/][bold {FG}]c[/][{FADED}]] copy · [/]"
            f"[{FADED}][[/][bold {FG}]r[/][{FADED}]] regen · [/]"
            f"[{FADED}][[/][bold {FG}]↳[/][{FADED}]] reply[/]"
        )
        inner_lines.append(Text.from_markup(foot))

    block_panel = Panel(
        Group(*inner_lines),
        box=LEFT_BAR,
        border_style=bar,
        padding=(0, 0, 0, 1),  # 바 우측 1칸 띄움
    )
    parts.append(block_panel)
    parts.append(Text(""))
    return Group(*parts)


def render_card(state: State, turn: Turn, response: str, metrics: dict, streaming: bool) -> Group:
    """2C: 둥근 카드 + 4분할 메트릭 그리드."""
    user_card = Panel(
        Text(turn.content, style=FG),
        title=Text.from_markup(f"[{ACCENT}]❯[/] [{FADED}]you[/]"),
        title_align="left",
        subtitle=Text.from_markup(f"[{FADED}]{turn.timestamp} · {len(turn.content)} chars[/]"),
        subtitle_align="right",
        box=ROUNDED,
        border_style=FADED,
        padding=(0, 1),
    )

    bar = YELLOW if streaming else GREEN
    body = Text.from_markup(f"[{MID}]{response}[/]" + (f"[{FG} on {LOW}] [/]" if streaming else ""))

    if not streaming and metrics:
        grid = Table.grid(expand=True, padding=(0, 1))
        for _ in range(4):
            grid.add_column(ratio=1)
        grid.add_row(
            Text.from_markup(f"[{LOW}]tokens[/]\n[{FG}]{metrics.get('out_tokens', 0)}[/] [{DIM}](in {metrics.get('in_tokens', 0)})[/]"),
            Text.from_markup(f"[{LOW}]latency[/]\n[{FG}]{metrics.get('time', 0):.2f}s[/]"),
            Text.from_markup(f"[{LOW}]throughput[/]\n[{FG}]{metrics.get('speed', 0)} t/s[/]"),
            Text.from_markup(f"[{LOW}]cost[/]\n[{FG}]${metrics.get('cost', 0):.5f}[/]"),
        )
        body_group = Group(body, Rule(style=bar), grid)
    else:
        body_group = body

    head_left  = f"[{bar}]◆[/] [bold {FG}]{state.model_display.upper()}[/] [{LOW}]· response #{state.turn_no} · {turn.timestamp}[/]"
    head_right = f"[{FADED}][[/][bold {FG}]c[/][{FADED}]]opy [[/][bold {FG}]r[/][{FADED}]]egen [[/][bold {FG}]s[/][{FADED}]]ave [[/][bold {FG}]↳[/][{FADED}]]reply[/]"

    resp_card = Panel(
        body_group,
        title=Text.from_markup(head_left if not streaming else f"[{bar}]◆[/] [bold {FG}]{state.model_display.upper()}[/] [{DIM}]· generating...[/]"),
        title_align="left",
        subtitle=Text.from_markup(head_right) if not streaming else Text.from_markup(f"[{bar}]● {metrics.get('time', 0):.1f}s · {metrics.get('out_tokens', 0)}/?? tok[/]"),
        subtitle_align="right",
        box=ROUNDED,
        border_style=bar if not streaming else f"dim {bar}",
        padding=(0, 1),
    )

    return Group(user_card, Text(""), resp_card, Text(""))


def render(state: State, turn: Turn, response: str, metrics: dict, streaming: bool):
    if state.display == "inline":
        return render_inline(turn, response, metrics, streaming)
    if state.display == "card":
        return render_card(state, turn, response, metrics, streaming)
    return render_block(state, turn, response, metrics, streaming)


# ─────────────────────────────────────────────
#  STREAMING  (mock | OpenAI-compatible)
# ─────────────────────────────────────────────
MOCK_REPLIES = [
    (
        ["안녕", "hi ", "hello", "헬로", "안뇽", "ㅎㅇ"],
        "안녕하세요. HLLM_CLI mock 모드입니다.\n"
        "지금은 TOKEN.TXT 가 없어서 실제 API 호출 대신 정해진 답변을 드리고 있어요.\n"
        "실제 응답을 받으시려면 작업 디렉토리에 TOKEN.TXT 를 두고 다시 실행해주세요."
    ),
    (
        ["누구", "뭐야", "정체", "who", "what are you"],
        "저는 HLLM_CLI 의 mock 응답기입니다.\n"
        "실제 대화는 config.json 의 모델 7개 (Coder-480B, Next-80B, GLM-5 등) 중 하나가 처리합니다.\n"
        "지금은 토큰이 없어서 시뮬레이션 중입니다."
    ),
    (
        ["테스트", "test", "ping"],
        "pong. 채팅 루프 / 스트리밍 / 표시 모드 / fallback chain 모두 정상 동작 중입니다."
    ),
    (
        ["help", "명령", "기능"],
        "주요 명령어:\n"
        "  /display inline|block|card   응답 표시 방식 변경\n"
        "  /model                       모델 다시 선택\n"
        "  /compare                     세 표시 방식 비교\n"
        "  /clear                       세션 초기화\n"
        "  /quit                        종료"
    ),
    (
        ["csv", "polars", "pandas"],
        "대용량 CSV는 polars 가 가장 빠릅니다. pandas 대비 5~10배 빠르고 메모리도 절반 수준입니다.\n\n"
        "  import polars as pl\n"
        "  df = pl.read_csv(\"big.csv\")\n"
        "  print(df.describe())\n\n"
        "파일 크기가 1GB 넘으면 scan_csv() 로 lazy 로딩을 권장합니다."
    ),
    (
        ["groupby", "group_by", "그룹"],
        "polars 의 groupby 는 group_by() 로 호출합니다.\n"
        "  df.group_by(\"region\").agg([pl.col(\"sales\").sum()])\n"
        "lazy 모드에서 더 빠르게 동작합니다."
    ),
    (
        ["반도체", "fab", "amhs", "oht"],
        "AMHS 흐름 분석은 보통 (1) 시계열 IDC 데이터 정합 → (2) HID/EDGE 단위 집계 → "
        "(3) surge / deadlock 룰 평가 의 3단계로 나뉩니다. 어떤 부분을 자세히 볼까요?"
    ),
    (
        ["고마워", "감사", "thanks", "thank"],
        "별말씀을요. 더 필요한 게 있으면 언제든 말씀해주세요."
    ),
]
MOCK_DEFAULT = (
    "확인했습니다. 좀 더 구체적으로 알려주시면 코드 예시까지 같이 드리겠습니다."
)


def pick_mock(prompt: str) -> str:
    p = prompt.lower()
    for kws, text in MOCK_REPLIES:
        if any(k in p for k in kws):
            return text
    return MOCK_DEFAULT


def stream_mock(prompt: str) -> Iterator[str]:
    text = pick_mock(prompt)
    # 한글 음절 단위로 나눠서 스트리밍 흉내
    for ch in text:
        yield ch
        time.sleep(0.012)


def _stream_one(url: str, api_model: str, token: Optional[str], state: State, prompt: str) -> Iterator[str]:
    """단일 endpoint 에 대한 OpenAI-compatible /chat/completions streaming.
       성공 시 chunk 들을 yield, 실패 시 RuntimeError raise."""
    import urllib.request
    import urllib.error

    msgs = [{"role": t.role, "content": t.content} for t in state.history]
    msgs.append({"role": "user", "content": prompt})

    payload = json.dumps({
        "model": api_model,
        "messages": msgs,
        "stream": True,
    }, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(
        url=url,
        data=payload,
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {token or ''}",
            "Accept": "text/event-stream",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            for raw in r:
                line = raw.decode("utf-8", errors="ignore").rstrip()
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[6:]
                elif line.startswith("data:"):
                    data = line[5:].lstrip()
                else:
                    continue
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta
                except Exception:
                    continue
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="ignore")[:200]
        except Exception:
            pass
        raise RuntimeError(f"HTTP {e.code} {e.reason} — {body}")
    except Exception as e:
        raise RuntimeError(f"{type(e).__name__}: {e}")


def stream_api(state: State, prompt: str) -> Iterator[str]:
    """모델별 URL 사용 + fallback_chains 자동 재시도.
       primary 가 데이터 한 글자도 못 보내고 실패하면 다음 모델로 넘어감.
       이미 토큰이 흐르기 시작했으면 fallback 안 함 (대화가 깨짐)."""
    primary_id = state.model_id
    chain = [primary_id]
    if state.use_fallback:
        chain += state.config.get("fallback_chains", {}).get(primary_id, [])

    last_err = None
    for idx, mid in enumerate(chain):
        meta = state.config.get("models", {}).get(mid)
        if not meta:
            continue
        url = meta.get("url")
        api_model = meta.get("model", mid)
        if not url:
            continue

        if idx > 0:
            yield f"\n[{YELLOW}]↻ fallback → {meta.get('name', mid)}[/]\n"

        produced_any = False
        try:
            for chunk in _stream_one(url, api_model, state.token, state, prompt):
                produced_any = True
                yield chunk
            return  # 정상 종료
        except RuntimeError as e:
            last_err = e
            if produced_any:
                # 스트리밍 도중 끊긴 케이스 — fallback 하면 대화 깨짐
                yield f"\n[error: {e}]\n"
                return
            # 첫 글자도 못 받음 → fallback 시도
            continue

    yield f"\n[error: 모든 모델 실패. 마지막 에러: {last_err}]\n"


# ─────────────────────────────────────────────
#  CHAT LOOP
# ─────────────────────────────────────────────
HELP_TEXT = (
    f"  [{FG}]/help[/]              명령어 목록\n"
    f"  [{FG}]/display X[/]         X = inline | block | card\n"
    f"  [{FG}]/model[/]             모델 다시 선택\n"
    f"  [{FG}]/clear[/]             세션 초기화\n"
    f"  [{FG}]/compare[/]           표시 방식 비교 매트릭스\n"
    f"  [{FG}]/quit[/]              종료  (^D / ^C 도 동일)"
)


def chat_loop(state: State) -> None:
    console.print(session_header(state))
    console.print(Rule(style=FADED))
    console.print()
    console.print(
        f"[{FADED}]display: [bold {FG}]{state.display}[/] · "
        f"명령어는 [bold {FG}]/help[/][/]"
    )
    if state.is_mock:
        console.print(
            f"[{YELLOW}][!] mock 모드[/] [{FADED}]— TOKEN.TXT 가 없어 정해진 응답이 나옵니다. "
            f"실제 API 호출은 토큰 추가 후 가능합니다.[/]"
        )
    console.print()

    while True:
        try:
            user_input = console.input(f"[{ACCENT}]❯[/] [{DIM}]you[/] [{LOW}]›[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print(f"\n[{FADED}]bye.[/]")
            return

        if not user_input:
            continue

        # ── slash 명령 ──
        if user_input.startswith("/"):
            cmd, *args = user_input[1:].split()
            cmd = cmd.lower()
            if cmd in ("q", "quit", "exit"):
                console.print(f"[{FADED}]bye.[/]")
                return
            if cmd == "help":
                console.print(HELP_TEXT); console.print(); continue
            if cmd == "display":
                if args and args[0] in ("inline", "block", "card"):
                    state.display = args[0]
                    console.print(f"[{GREEN}][✓][/] display = [bold {FG}]{state.display}[/]\n")
                else:
                    console.print(f"[{YELLOW}]usage:[/] /display inline|block|card\n")
                continue
            if cmd == "model":
                show_splash(state); continue
            if cmd == "clear":
                state.history.clear(); state.turn_no = 0
                console.clear()
                console.print(session_header(state)); console.print(Rule(style=FADED)); console.print()
                continue
            if cmd == "compare":
                show_compare(); continue
            console.print(f"[{YELLOW}]unknown command:[/] /{cmd}  ([{FG}]/help[/])\n")
            continue

        # ── 일반 메시지 ──
        # console.input() 이 echo 한 prompt 라인을 지운다.
        # render() 가 user_line 을 다시 그리므로, 안 지우면 prompt 가 두 번 보임.
        try:
            sys.stdout.write("\x1b[F\x1b[2K")  # cursor up 1 + erase line
            sys.stdout.flush()
        except Exception:
            pass

        state.turn_no += 1
        in_tokens = max(1, len(user_input) // 2)  # rough
        user_turn = Turn(
            role="user", content=user_input,
            metrics={"in_tokens": in_tokens, "out_tokens": 0},
        )
        state.history.append(user_turn)

        start = time.time()
        accumulated = ""

        if state.is_mock:
            stream = stream_mock(user_input)
        else:
            stream = stream_api(state, user_input)

        try:
            with Live(console=console, refresh_per_second=24, transient=False, vertical_overflow="visible") as live:
                live.update(render(state, user_turn, accumulated, {}, streaming=True))
                for chunk in stream:
                    accumulated += chunk
                    metrics_live = {
                        "in_tokens": in_tokens,
                        "out_tokens": max(1, len(accumulated) // 2),
                        "time": time.time() - start,
                    }
                    live.update(render(state, user_turn, accumulated, metrics_live, streaming=True))

                elapsed = max(time.time() - start, 0.001)
                out_tokens = max(1, len(accumulated) // 2)
                final_metrics = {
                    "in_tokens": in_tokens,
                    "out_tokens": out_tokens,
                    "time": elapsed,
                    "speed": int(out_tokens / elapsed),
                    "cost": out_tokens * 0.0000025,  # mock pricing
                }
                live.update(render(state, user_turn, accumulated, final_metrics, streaming=False))
        except KeyboardInterrupt:
            console.print(f"\n[{YELLOW}]^c — 중지됨[/]\n")
            continue

        state.history.append(Turn(
            role="assistant", content=accumulated,
            metrics={"in_tokens": 0, "out_tokens": final_metrics["out_tokens"]},
        ))


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(
        prog="hllm",
        description="HLLM_CLI — open llm gateway · terminal client",
    )
    p.add_argument("--config",     default=None,
                   help="config.json 경로 (기본: ./config.json 또는 스크립트 옆)")
    p.add_argument("--token-file", default=None,
                   help="API 토큰 파일 경로 (기본: ./TOKEN.TXT)")
    p.add_argument("--token",      default=None,
                   help="API 토큰을 직접 지정 (--token-file 보다 우선)")
    p.add_argument("--model",      default=None,
                   help="모델 ID (config.models 의 키. splash 건너뜀)")
    p.add_argument("--display",    default="block",
                   choices=("inline", "block", "card"),
                   help="응답 표시 방식 (default: block)")
    p.add_argument("--no-splash",  action="store_true",
                   help="splash 화면 건너뛰고 바로 chat")
    p.add_argument("--no-fallback", action="store_true",
                   help="fallback_chains 자동 재시도 끔")
    args = p.parse_args()

    cfg   = load_config(args.config)
    token = args.token or load_token(args.token_file)

    state = State(
        config=cfg,
        token=token,
        display=args.display,
        use_fallback=not args.no_fallback,
    )

    if args.model:
        if args.model in cfg.get("models", {}):
            state.model_id = args.model
        else:
            console.print(f"[{ACCENT}]모델 ID '{args.model}' 가 config.models 에 없습니다.[/]")
            console.print(f"[{FADED}]사용 가능한 ID: {', '.join(cfg.get('models', {}).keys())}[/]")
            sys.exit(2)
    elif not args.no_splash:
        show_splash(state)
    else:
        # splash 도 안 띄우고 model 지정도 없을 때 → priority 1번 자동
        sm = models_sorted(cfg)
        if not sm:
            console.print(f"[{ACCENT}]config.models 비어있음. 종료.[/]"); sys.exit(1)
        state.model_id = sm[0][0]

    chat_loop(state)


if __name__ == "__main__":
    main()
