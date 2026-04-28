#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hllm2.py — HLLM_CLI · MDIR-style commander (A형, 풀 패널)

90년대 도스 시절 MDIR 룩앤필을 닮은 풀스크린 커맨더.
- 좌측: 모델 리스트 (config.json 의 7개)
- 우측: 선택 모델 메타 + 최근 대화 미리보기
- 하단: F1~F10 펑션 바
- Enter 로 챗 진입 (hlim.chat_loop 재사용), 챗에서 빠져나오면 커맨더 복귀

사용:
    python hllm2.py
    python hllm2.py --token-file TOKEN.TXT
    python hllm2.py --display card
"""

from __future__ import annotations

import argparse
import os
import sys
import termios
import tty
from contextlib import contextmanager
from datetime import datetime

# hlim.py 의 도메인 로직 전부 재사용
from hlim import (
    State,
    chat_loop,
    console,
    cost_tier_color,
    cost_tier_label,
    load_config,
    load_token,
    models_sorted,
    show_compare,
    show_model_info,
)

try:
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.text import Text
    from rich.align import Align
    from rich.box import DOUBLE, HEAVY
except ImportError:
    sys.stderr.write("[hllm2] rich 가 필요합니다.  pip install rich\n")
    sys.exit(1)


# ── MDIR 팔레트 ────────────────────────────────────────────
BG          = "blue"
TITLE_FG    = "bright_yellow"
KEY_NUM     = "bright_yellow"
KEY_LBL     = "white on cyan"
FG          = "white"
DIM         = "bright_black"
SEL         = "black on bright_yellow"     # 선택 항목 반전
PANEL_BD    = "bright_white"
ON_BG       = f"{FG} on {BG}"
TITLE_STYLE = f"bold {TITLE_FG} on {BG}"


# ── 키 입력 (POSIX raw) ────────────────────────────────────
@contextmanager
def raw_tty():
    """cbreak 모드 진입. yield 값은 원래 cooked termios attr (복귀용)."""
    fd = sys.stdin.fileno()
    cooked = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield cooked
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, cooked)


@contextmanager
def cooked_tty(cooked):
    """raw 중간에 잠시 cooked 로 돌렸다가 다시 cbreak 로 복귀."""
    fd = sys.stdin.fileno()
    raw = termios.tcgetattr(fd)
    try:
        termios.tcsetattr(fd, termios.TCSADRAIN, cooked)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, raw)


def read_key() -> str:
    """ESC 시퀀스 / 펑션키 / 일반키를 한 토큰으로 반환.
    반환값: 'UP', 'DOWN', 'LEFT', 'RIGHT', 'ENTER', 'BACKSPACE',
            'F1'..'F12', 'ESC', 또는 일반 문자 한 글자."""
    ch = sys.stdin.read(1)
    if ch == "\x1b":
        # ESC + 짧은 시퀀스 더 읽기 (non-blocking 흉내)
        seq = ""
        try:
            import select
            while True:
                r, _, _ = select.select([sys.stdin], [], [], 0.02)
                if not r:
                    break
                seq += sys.stdin.read(1)
        except Exception:
            pass
        if not seq:
            return "ESC"
        # CSI: ESC [ ...
        if seq.startswith("[") or seq.startswith("O"):
            body = seq[1:]
            mapping = {
                "A": "UP", "B": "DOWN", "C": "RIGHT", "D": "LEFT",
                "H": "HOME", "F": "END",
                "P": "F1", "Q": "F2", "R": "F3", "S": "F4",
                "1~": "HOME", "4~": "END", "5~": "PGUP", "6~": "PGDN",
                "11~": "F1", "12~": "F2", "13~": "F3", "14~": "F4",
                "15~": "F5", "17~": "F6", "18~": "F7", "19~": "F8",
                "20~": "F9", "21~": "F10", "23~": "F11", "24~": "F12",
            }
            if body in mapping:
                return mapping[body]
        return "ESC"
    if ch in ("\r", "\n"):
        return "ENTER"
    if ch in ("\x7f", "\x08"):
        return "BACKSPACE"
    if ch == "\t":
        return "TAB"
    if ch == "\x03":  # ^C
        raise KeyboardInterrupt
    if ch == "\x04":  # ^D
        return "F10"
    return ch


# ── 화면 빌더 ──────────────────────────────────────────────
TITLE = "HLLM COMMANDER · MDIR-STYLE"


def build_header(state: State) -> Panel:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    mode = "mock" if state.is_mock else "live"
    left = Text.from_markup(f" [bold {TITLE_FG}]■ {TITLE}[/]")
    right = Text.from_markup(
        f"[{DIM}]{mode}[/]  [{TITLE_FG}]{now}[/] "
    )
    line = Text.assemble(left, Text(" " * 200), right, style=f"on {BG}")
    line.truncate(200, overflow="ellipsis")
    return Panel(line, box=DOUBLE, border_style=PANEL_BD, style=f"on {BG}", padding=(0, 0))


def build_models_panel(state: State, sorted_models, cursor: int) -> Panel:
    body = Text(style=ON_BG)
    body.append(f" MODELS [{len(sorted_models)}]\n", style=f"bold {TITLE_FG} on {BG}")
    body.append("\n")
    for i, (mid, m) in enumerate(sorted_models, start=1):
        is_cur = (i - 1 == cursor)
        prio = m.get("priority", 99)
        tier = m.get("cost_tier", "?")
        tier_l = cost_tier_label(tier)
        name = m.get("name", mid).split(" (")[0]
        line = f" {i}.{name:<22} {tier_l:>3}  p{prio} "
        if is_cur:
            body.append("▶", style=f"bold {TITLE_FG} on {BG}")
            body.append(line, style=SEL)
        else:
            body.append(" ", style=ON_BG)
            body.append(line, style=ON_BG)
        body.append("\n", style=ON_BG)
    body.append("\n")
    body.append(" ↑↓ 이동 · 1~7 점프 · Enter 접속", style=f"{DIM} on {BG}")
    return Panel(
        body,
        title=f"[{TITLE_FG}]MODELS[/]",
        title_align="left",
        border_style=PANEL_BD,
        style=f"on {BG}",
        box=HEAVY,
        padding=(0, 1),
    )


def build_meta_panel(state: State, sorted_models, cursor: int) -> Panel:
    mid, m = sorted_models[cursor]
    tier = m.get("cost_tier", "?")
    tier_c = cost_tier_color(tier)
    tier_l = cost_tier_label(tier)
    ctx_k = m.get("context_window", 0) // 1000
    caps = ", ".join(m.get("capabilities", []))
    api = m.get("model", mid)
    url = m.get("url", "")

    body = Text(style=ON_BG)
    body.append(f"  {m.get('name', mid)}\n", style=f"bold {TITLE_FG} on {BG}")
    body.append(f"  id      : ", style=f"{DIM} on {BG}"); body.append(f"{mid}\n", style=ON_BG)
    body.append(f"  api     : ", style=f"{DIM} on {BG}"); body.append(f"{api}\n", style=ON_BG)
    body.append(f"  url     : ", style=f"{DIM} on {BG}"); body.append(f"{url}\n", style=ON_BG)
    body.append(f"  ctx     : ", style=f"{DIM} on {BG}"); body.append(f"{ctx_k}k\n", style=ON_BG)
    body.append(f"  prio    : ", style=f"{DIM} on {BG}"); body.append(f"{m.get('priority','?')}\n", style=ON_BG)
    body.append(f"  cost    : ", style=f"{DIM} on {BG}"); body.append(f"{tier_l}", style=f"bold {tier_c} on {BG}"); body.append(f"  ({tier})\n", style=ON_BG)
    body.append(f"  caps    : ", style=f"{DIM} on {BG}"); body.append(f"{caps}\n", style=ON_BG)
    body.append("\n", style=ON_BG)

    body.append("  HISTORY\n", style=f"bold {TITLE_FG} on {BG}")
    if not state.history:
        body.append("    · (비어있음)\n", style=f"{DIM} on {BG}")
    else:
        for t in state.history[-6:]:
            tag = "you" if t.role == "user" else "ai "
            preview = t.content.replace("\n", " ").strip()
            if len(preview) > 64:
                preview = preview[:63] + "…"
            body.append(f"    [{t.timestamp}] {tag} ", style=f"{DIM} on {BG}")
            body.append(f"{preview}\n", style=ON_BG)
    body.append("\n", style=ON_BG)

    body.append(f"  display : ", style=f"{DIM} on {BG}")
    body.append(f"{state.display}\n", style=ON_BG)
    body.append(f"  mode    : ", style=f"{DIM} on {BG}")
    body.append(f"{'mock' if state.is_mock else 'live'}\n", style=ON_BG)

    return Panel(
        body,
        title=f"[{TITLE_FG}]META · HISTORY[/]",
        title_align="left",
        border_style=PANEL_BD,
        style=f"on {BG}",
        box=HEAVY,
        padding=(0, 1),
    )


FN_KEYS = [
    ("1",  "Help"),
    ("2",  "Model"),
    ("3",  "Display"),
    ("4",  "Clear"),
    ("5",  "Compare"),
    ("7",  "Info"),
    ("9",  "About"),
    ("10", "Quit"),
]


def build_footer() -> Panel:
    bar = Text(style=ON_BG)
    bar.append(" ", style=ON_BG)
    for num, lbl in FN_KEYS:
        bar.append(f"{num}", style=f"bold {KEY_NUM} on {BG}")
        bar.append(f"{lbl}", style=KEY_LBL)
        bar.append(" ", style=ON_BG)
    return Panel(bar, box=DOUBLE, border_style=PANEL_BD, style=f"on {BG}", padding=(0, 0))


def build_layout(state: State, sorted_models, cursor: int) -> Layout:
    root = Layout()
    root.split_column(
        Layout(build_header(state), name="head", size=3),
        Layout(name="body", ratio=1),
        Layout(build_footer(), name="foot", size=3),
    )
    root["body"].split_row(
        Layout(build_models_panel(state, sorted_models, cursor), name="models", ratio=2),
        Layout(build_meta_panel(state, sorted_models, cursor), name="meta", ratio=3),
    )
    return root


# ── 도움말 오버레이 ────────────────────────────────────────
HELP_LINES = [
    ("↑/↓",   "모델 커서 이동"),
    ("1~7",   "모델 인덱스 점프"),
    ("Enter", "선택 모델로 챗 진입"),
    ("F1 / ?", "이 도움말 토글"),
    ("F3",    "display 순환 (inline→block→card)"),
    ("F4",    "history 비우기"),
    ("F5",    "화면 다시 그리기"),
    ("F7",    "모델 상세 테이블"),
    ("F9",    "표시 방식 비교"),
    ("F10 / q / ^D", "종료"),
]


def show_help() -> None:
    console.clear()
    body = Text(style=ON_BG)
    body.append("\n  HLLM COMMANDER · 키 도움말\n\n", style=f"bold {TITLE_FG} on {BG}")
    for k, desc in HELP_LINES:
        body.append(f"   {k:<14}", style=f"bold {KEY_NUM} on {BG}")
        body.append(f"{desc}\n", style=ON_BG)
    body.append("\n  아무 키나 눌러 닫기", style=f"{DIM} on {BG}")
    console.print(Panel(body, border_style=PANEL_BD, box=HEAVY, style=f"on {BG}",
                        title=f"[{TITLE_FG}]F1 · HELP[/]", title_align="left"))
    read_key()


# ── 커맨더 루프 ────────────────────────────────────────────
DISPLAYS = ("inline", "block", "card")


def commander(state: State) -> None:
    sorted_models = models_sorted(state.config)
    if not sorted_models:
        console.print("[red]config.models 가 비어있습니다.[/]")
        return

    # 시작 모델 = state.model_id 가 있으면 그 인덱스, 아니면 0
    cursor = 0
    for i, (mid, _) in enumerate(sorted_models):
        if mid == state.model_id:
            cursor = i
            break
    state.model_id = sorted_models[cursor][0]

    with raw_tty() as cooked:
        while True:
            console.clear()
            console.print(build_layout(state, sorted_models, cursor))

            try:
                key = read_key()
            except KeyboardInterrupt:
                break

            if key == "UP":
                cursor = (cursor - 1) % len(sorted_models)
                state.model_id = sorted_models[cursor][0]
            elif key == "DOWN":
                cursor = (cursor + 1) % len(sorted_models)
                state.model_id = sorted_models[cursor][0]
            elif key == "HOME":
                cursor = 0
                state.model_id = sorted_models[cursor][0]
            elif key == "END":
                cursor = len(sorted_models) - 1
                state.model_id = sorted_models[cursor][0]
            elif key.isdigit() and key != "0":
                idx = int(key) - 1
                if 0 <= idx < len(sorted_models):
                    cursor = idx
                    state.model_id = sorted_models[cursor][0]
            elif key == "ENTER":
                state.model_id = sorted_models[cursor][0]
                # cooked 로 잠시 돌리고 chat_loop 진입
                with cooked_tty(cooked):
                    console.clear()
                    chat_loop(state)
                continue
            elif key in ("F1", "?"):
                show_help()
            elif key == "F3":
                idx = (DISPLAYS.index(state.display) + 1) % len(DISPLAYS) if state.display in DISPLAYS else 0
                state.display = DISPLAYS[idx]
            elif key == "F4":
                state.history.clear()
                state.turn_no = 0
            elif key == "F5":
                pass  # 루프 상단에서 다시 그림
            elif key == "F7":
                _showcase(lambda: show_model_info(sorted_models))
            elif key in ("F9", "F2"):
                _showcase(show_compare)
            elif key in ("F10", "q", "Q", "ESC"):
                break

    console.print()
    console.print(f"[{DIM}]bye.[/]")


def _showcase(fn) -> None:
    """현재 raw 모드 그대로 두고 일반 출력 후 키 한 번 대기."""
    console.clear()
    fn()
    console.print(f"[{DIM}]아무 키나 눌러 돌아가기...[/]")
    read_key()


# ── 메인 ───────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(
        prog="hllm2",
        description="HLLM_CLI — MDIR-style commander (A형)",
    )
    p.add_argument("--config", default=None)
    p.add_argument("--token-file", default=None)
    p.add_argument("--token", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--display", default="block",
                   choices=("inline", "block", "card"))
    p.add_argument("--no-fallback", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    token = args.token or load_token(args.token_file)

    state = State(
        config=cfg,
        token=token,
        display=args.display,
        use_fallback=not args.no_fallback,
    )

    if args.model and args.model in cfg.get("models", {}):
        state.model_id = args.model

    if not sys.stdin.isatty():
        console.print("[red]hllm2 는 인터랙티브 TTY 가 필요합니다.[/]")
        sys.exit(2)

    try:
        commander(state)
    except KeyboardInterrupt:
        console.print(f"\n[{DIM}]^C — bye.[/]")


if __name__ == "__main__":
    main()
