#!/usr/bin/env python3
"""
hllm2 — MDIR/Norton Commander 스타일 CLI (Windows)

화면 구성:
  ┌─[ hllm Commander ]──────────────────────┐
  │ 좌 패널: 모델 목록    │ 우 패널: 정보   │
  │                       │                 │
  │ [작업 로그 영역]                        │
  └─────────────────────────────────────────┘
   F1 도움  F2 모델  F3 스킬  F4 컨텍스트  F5 작업  F10 종료

전제:
  hllm.py 와 같은 폴더에 두기. config.json, token.txt 사용.
  pip install httpx rich readchar
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Windows UTF-8
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# hllm 모듈 import (같은 폴더의 hllm.py)
sys.path.insert(0, str(Path(__file__).parent))
try:
    import hllm  # type: ignore
except ImportError:
    print("[ERROR] hllm.py가 같은 폴더에 있어야 합니다.")
    sys.exit(1)

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.align import Align

try:
    import readchar
    HAS_READCHAR = True
except ImportError:
    HAS_READCHAR = False

console = Console()


# ─────────────────────────────────────────────
# DOS 컬러 테마 (MDIR 스타일)
# ─────────────────────────────────────────────
THEME = {
    "bg":       "blue",          # 배경 (파란색)
    "panel_bg": "on blue",
    "border":   "yellow on blue",
    "title":    "bold yellow on blue",
    "text":     "white on blue",
    "selected": "black on cyan",
    "func_bg":  "white on cyan",
    "func_num": "black on cyan",
    "func_lbl": "black on cyan",
    "active":   "bold yellow",
    "dim":      "bright_black on blue",
}


# ─────────────────────────────────────────────
# 상태
# ─────────────────────────────────────────────
class State:
    def __init__(self):
        self.cursor_left = 0          # 좌 패널 선택 인덱스
        self.cursor_right = 0
        self.focus = "left"           # 'left' or 'right'
        self.view = "models"          # 'models', 'skills', 'context', 'sessions'
        self.active_skills: list[str] = []
        self.log_lines: list[str] = []
        self.max_log = 8

    def add_log(self, msg: str):
        self.log_lines.append(msg)
        if len(self.log_lines) > self.max_log:
            self.log_lines = self.log_lines[-self.max_log:]


STATE = State()


# ─────────────────────────────────────────────
# 좌 패널 - 컨텐츠 종류별
# ─────────────────────────────────────────────
def render_left_panel() -> Panel:
    if STATE.view == "models":
        return _render_models_panel()
    elif STATE.view == "skills":
        return _render_skills_panel()
    elif STATE.view == "context":
        return _render_context_panel()
    elif STATE.view == "sessions":
        return _render_sessions_panel()
    return Panel("(empty)", style=THEME["text"])


def _render_models_panel() -> Panel:
    items = sorted(hllm.MODELS.items(),
                   key=lambda kv: kv[1].get("priority", 99))
    lines = []
    for i, (mid, mcfg) in enumerate(items):
        active = "★" if mid == hllm.CURRENT_MODEL else " "
        cursor = "▶" if (i == STATE.cursor_left and STATE.focus == "left") else " "
        tier = mcfg.get("cost_tier", "")
        line_text = f"{cursor} {active} {mid:25s} [{tier}]"
        if i == STATE.cursor_left and STATE.focus == "left":
            lines.append(Text(line_text, style=THEME["selected"]))
        elif mid == hllm.CURRENT_MODEL:
            lines.append(Text(line_text, style="bold yellow on blue"))
        else:
            lines.append(Text(line_text, style=THEME["text"]))

    return Panel(
        Group(*lines) if lines else Text("(no models)", style=THEME["text"]),
        title="[bold yellow]MODELS[/]",
        title_align="left",
        border_style=THEME["border"],
        style=THEME["panel_bg"],
    )


def _render_skills_panel() -> Panel:
    skills = hllm.list_skills()
    lines = []
    if not skills:
        lines.append(Text("(스킬 없음)", style=THEME["dim"]))
        lines.append(Text(f"  {hllm.SKILLS_DIR}", style=THEME["dim"]))
        lines.append(Text("  에 SKILL.md 만드세요", style=THEME["dim"]))
    else:
        for i, s in enumerate(skills):
            active = "★" if s["name"] in STATE.active_skills else " "
            cursor = "▶" if (i == STATE.cursor_left and STATE.focus == "left") else " "
            line_text = f"{cursor} {active} {s['name']}"
            if i == STATE.cursor_left and STATE.focus == "left":
                lines.append(Text(line_text, style=THEME["selected"]))
            elif s["name"] in STATE.active_skills:
                lines.append(Text(line_text, style="bold yellow on blue"))
            else:
                lines.append(Text(line_text, style=THEME["text"]))

    return Panel(
        Group(*lines) if lines else Text("(empty)", style=THEME["text"]),
        title="[bold yellow]SKILLS[/]",
        title_align="left",
        border_style=THEME["border"],
        style=THEME["panel_bg"],
    )


def _render_context_panel() -> Panel:
    lines = []

    # 글로벌 컨텍스트
    if hllm.CONTEXT_FILE.exists():
        sz = hllm.CONTEXT_FILE.stat().st_size
        lines.append(Text(f"✓ global  ({sz:,} bytes)",
                          style="bold green on blue"))
        lines.append(Text(f"  {hllm.CONTEXT_FILE}", style=THEME["dim"]))
    else:
        lines.append(Text("· global  (없음)", style=THEME["dim"]))
        lines.append(Text(f"  {hllm.CONTEXT_FILE}", style=THEME["dim"]))

    lines.append(Text("", style=THEME["text"]))

    # 프로젝트 컨텍스트
    local = Path(hllm.LOCAL_CONTEXT_FILE)
    if local.exists():
        lines.append(Text(f"✓ project ({local.stat().st_size:,} bytes)",
                          style="bold green on blue"))
        lines.append(Text(f"  ./{hllm.LOCAL_CONTEXT_FILE}", style=THEME["dim"]))
    else:
        lines.append(Text("· project (없음)", style=THEME["dim"]))
        lines.append(Text(f"  ./{hllm.LOCAL_CONTEXT_FILE}", style=THEME["dim"]))

    lines.append(Text("", style=THEME["text"]))

    # 활성 스킬
    if STATE.active_skills:
        lines.append(Text("활성 스킬:", style="bold cyan on blue"))
        for s in STATE.active_skills:
            lines.append(Text(f"  ★ {s}", style="bold yellow on blue"))
    else:
        lines.append(Text("활성 스킬: (없음)", style=THEME["dim"]))

    return Panel(
        Group(*lines),
        title="[bold yellow]CONTEXT[/]",
        title_align="left",
        border_style=THEME["border"],
        style=THEME["panel_bg"],
    )


def _render_sessions_panel() -> Panel:
    lines = []
    if hllm.SESSION_FILE.exists():
        sz = hllm.SESSION_FILE.stat().st_size
        try:
            data = json.loads(hllm.SESSION_FILE.read_text(encoding="utf-8"))
            n_msg = len(data)
            n_user = sum(1 for m in data if m.get("role") == "user")
            lines.append(Text(f"✓ last_session", style="bold green on blue"))
            lines.append(Text(f"  메시지: {n_msg}, 턴: {n_user}",
                              style=THEME["text"]))
            lines.append(Text(f"  크기: {sz:,} bytes", style=THEME["dim"]))
            lines.append(Text(f"  {hllm.SESSION_FILE}", style=THEME["dim"]))
        except Exception:
            lines.append(Text("✗ 세션 파싱 실패", style="red on blue"))
    else:
        lines.append(Text("· 저장된 세션 없음", style=THEME["dim"]))
        lines.append(Text(f"  {hllm.SESSION_FILE}", style=THEME["dim"]))

    lines.append(Text("", style=THEME["text"]))
    if hllm.LOG_FILE.exists():
        sz = hllm.LOG_FILE.stat().st_size
        lines.append(Text(f"✓ actions.log ({sz:,} bytes)",
                          style="bold green on blue"))
    else:
        lines.append(Text("· actions.log (없음)", style=THEME["dim"]))

    return Panel(
        Group(*lines),
        title="[bold yellow]SESSIONS[/]",
        title_align="left",
        border_style=THEME["border"],
        style=THEME["panel_bg"],
    )


# ─────────────────────────────────────────────
# 우 패널 - 선택된 항목의 상세 정보
# ─────────────────────────────────────────────
def render_right_panel() -> Panel:
    if STATE.view == "models":
        return _render_model_detail()
    elif STATE.view == "skills":
        return _render_skill_detail()
    elif STATE.view == "context":
        return _render_context_detail()
    elif STATE.view == "sessions":
        return _render_session_detail()
    return Panel("(empty)", style=THEME["panel_bg"])


def _render_model_detail() -> Panel:
    items = sorted(hllm.MODELS.items(),
                   key=lambda kv: kv[1].get("priority", 99))
    if not items or STATE.cursor_left >= len(items):
        return Panel(Text("(no model)", style=THEME["text"]),
                     border_style=THEME["border"], style=THEME["panel_bg"])

    mid, mcfg = items[STATE.cursor_left]
    lines = []
    lines.append(Text(f"ID:       {mid}", style="bold cyan on blue"))
    lines.append(Text(f"이름:     {mcfg.get('name', '')}", style=THEME["text"]))
    lines.append(Text(f"모델:     {mcfg.get('model', '')}", style=THEME["text"]))
    lines.append(Text(f"Tier:     {mcfg.get('cost_tier', '')}", style=THEME["text"]))
    lines.append(Text(f"우선순위: {mcfg.get('priority', '')}", style=THEME["text"]))
    lines.append(Text(f"Context:  {mcfg.get('context_window', 0):,}",
                      style=THEME["text"]))
    lines.append(Text("", style=THEME["text"]))

    caps = mcfg.get("capabilities", [])
    if caps:
        lines.append(Text("기능:", style="bold cyan on blue"))
        for c in caps:
            lines.append(Text(f"  • {c}", style=THEME["text"]))
    lines.append(Text("", style=THEME["text"]))

    url = mcfg.get("url", "")
    if url:
        lines.append(Text("URL:", style="bold cyan on blue"))
        lines.append(Text(f"  {url}", style=THEME["dim"]))
    lines.append(Text("", style=THEME["text"]))

    chain = hllm.FALLBACKS.get(mid, [])
    if chain:
        lines.append(Text("Fallback:", style="bold cyan on blue"))
        for c in chain[:5]:
            lines.append(Text(f"  → {c}", style=THEME["text"]))

    if mid == hllm.CURRENT_MODEL:
        lines.append(Text("", style=THEME["text"]))
        lines.append(Text("  ★ 현재 사용 중", style="bold yellow on blue"))
    else:
        lines.append(Text("", style=THEME["text"]))
        lines.append(Text("  Enter 또는 F2 → 활성화",
                          style="bold green on blue"))

    return Panel(
        Group(*lines),
        title=f"[bold yellow]MODEL: {mid}[/]",
        title_align="left",
        border_style=THEME["border"],
        style=THEME["panel_bg"],
    )


def _render_skill_detail() -> Panel:
    skills = hllm.list_skills()
    if not skills or STATE.cursor_left >= len(skills):
        return Panel(
            Text("스킬을 추가하려면:\n\n  /skill new <이름>\n\n또는 직접 폴더 생성:\n  ~/.hllm/skills/<이름>/SKILL.md",
                 style=THEME["text"]),
            title="[bold yellow]SKILL[/]",
            title_align="left",
            border_style=THEME["border"],
            style=THEME["panel_bg"],
        )

    s = skills[STATE.cursor_left]
    lines = []
    lines.append(Text(f"이름: {s['name']}", style="bold cyan on blue"))
    lines.append(Text(f"경로: {s['path']}", style=THEME["dim"]))
    if s["name"] in STATE.active_skills:
        lines.append(Text("상태: ★ 활성", style="bold yellow on blue"))
    else:
        lines.append(Text("상태: 비활성", style=THEME["text"]))
    lines.append(Text("", style=THEME["text"]))

    # SKILL.md 내용 미리보기
    text = hllm.load_skill(s["name"]) or ""
    preview = text.splitlines()[:15]
    for ln in preview:
        if len(ln) > 50:
            ln = ln[:50] + "…"
        lines.append(Text(ln, style=THEME["text"]))
    if len(text.splitlines()) > 15:
        lines.append(Text("…", style=THEME["dim"]))

    lines.append(Text("", style=THEME["text"]))
    if s["name"] in STATE.active_skills:
        lines.append(Text("  Enter → 비활성화", style="bold green on blue"))
    else:
        lines.append(Text("  Enter → 활성화", style="bold green on blue"))

    return Panel(
        Group(*lines),
        title=f"[bold yellow]SKILL: {s['name']}[/]",
        title_align="left",
        border_style=THEME["border"],
        style=THEME["panel_bg"],
    )


def _render_context_detail() -> Panel:
    lines = []
    lines.append(Text("Context는 hllm 시작 시", style=THEME["text"]))
    lines.append(Text("자동으로 system 프롬프트에", style=THEME["text"]))
    lines.append(Text("주입됩니다.", style=THEME["text"]))
    lines.append(Text("", style=THEME["text"]))
    lines.append(Text("편집 방법:", style="bold cyan on blue"))
    lines.append(Text(f"  notepad {hllm.CONTEXT_FILE}",
                      style=THEME["dim"]))
    lines.append(Text("", style=THEME["text"]))
    lines.append(Text("프로젝트별 (현재 폴더):", style="bold cyan on blue"))
    lines.append(Text(f"  notepad {hllm.LOCAL_CONTEXT_FILE}",
                      style=THEME["dim"]))

    # context.md 미리보기
    if hllm.CONTEXT_FILE.exists():
        lines.append(Text("", style=THEME["text"]))
        lines.append(Text("─ context.md 미리보기 ─",
                          style="bold yellow on blue"))
        try:
            text = hllm.CONTEXT_FILE.read_text(encoding="utf-8")
            for ln in text.splitlines()[:10]:
                if len(ln) > 50:
                    ln = ln[:50] + "…"
                lines.append(Text(ln, style=THEME["text"]))
        except Exception:
            pass

    return Panel(
        Group(*lines),
        title="[bold yellow]CONTEXT INFO[/]",
        title_align="left",
        border_style=THEME["border"],
        style=THEME["panel_bg"],
    )


def _render_session_detail() -> Panel:
    lines = []
    lines.append(Text("세션 관리", style="bold cyan on blue"))
    lines.append(Text("", style=THEME["text"]))
    lines.append(Text("  Enter → 세션 불러오기", style=THEME["text"]))
    lines.append(Text("  Del   → 세션 삭제", style=THEME["text"]))
    lines.append(Text("", style=THEME["text"]))

    if hllm.LOG_FILE.exists():
        lines.append(Text("─ 최근 액션 로그 ─",
                          style="bold yellow on blue"))
        try:
            log_text = hllm.LOG_FILE.read_text(encoding="utf-8")
            recent = log_text.strip().splitlines()[-10:]
            for ln in recent:
                if len(ln) > 50:
                    ln = ln[:50] + "…"
                lines.append(Text(ln, style=THEME["text"]))
        except Exception:
            pass

    return Panel(
        Group(*lines),
        title="[bold yellow]SESSION[/]",
        title_align="left",
        border_style=THEME["border"],
        style=THEME["panel_bg"],
    )


# ─────────────────────────────────────────────
# 로그 패널
# ─────────────────────────────────────────────
def render_log_panel() -> Panel:
    if not STATE.log_lines:
        content = Text("F5 또는 Enter로 작업 모드 진입",
                       style=THEME["dim"])
    else:
        content = Group(*[
            Text(ln, style=THEME["text"]) for ln in STATE.log_lines
        ])
    return Panel(
        content,
        title="[bold yellow]LOG[/]",
        title_align="left",
        border_style=THEME["border"],
        style=THEME["panel_bg"],
        height=10,
    )


# ─────────────────────────────────────────────
# 펑션 키 바
# ─────────────────────────────────────────────
def render_function_bar() -> Text:
    keys = [
        ("1", "Help"),
        ("2", "Models"),
        ("3", "Skills"),
        ("4", "Context"),
        ("5", "Run"),
        ("6", "Sessions"),
        ("7", "Reload"),
        ("8", "Edit"),
        ("9", "Yolo"),
        ("10", "Exit"),
    ]
    text = Text()
    for num, lbl in keys:
        text.append(f" F{num} ", style="black on cyan")
        text.append(f"{lbl:<8}", style="black on cyan")
    return text


# ─────────────────────────────────────────────
# 헤더
# ─────────────────────────────────────────────
def render_header() -> Panel:
    cfg = hllm.MODELS.get(hllm.CURRENT_MODEL, {})
    text = Text()
    text.append("hllm Commander ", style="bold yellow on blue")
    text.append(f"  Model: ", style=THEME["text"])
    text.append(hllm.CURRENT_MODEL, style="bold cyan on blue")
    text.append(f"  ", style=THEME["text"])
    if STATE.active_skills:
        text.append(f"Skills: ", style=THEME["text"])
        text.append(", ".join(STATE.active_skills), style="bold yellow on blue")
    text.append(f"  CWD: ", style=THEME["text"])
    text.append(os.getcwd()[:60], style=THEME["dim"])
    return Panel(
        text,
        border_style=THEME["border"],
        style=THEME["panel_bg"],
    )


# ─────────────────────────────────────────────
# 전체 레이아웃
# ─────────────────────────────────────────────
def make_layout() -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="log", size=10),
        Layout(name="func", size=1),
    )
    layout["main"].split_row(
        Layout(name="left", ratio=1),
        Layout(name="right", ratio=1),
    )
    return layout


def update_layout(layout: Layout):
    layout["header"].update(render_header())
    layout["left"].update(render_left_panel())
    layout["right"].update(render_right_panel())
    layout["log"].update(render_log_panel())
    layout["func"].update(render_function_bar())


# ─────────────────────────────────────────────
# 액션
# ─────────────────────────────────────────────
def action_select_model():
    items = sorted(hllm.MODELS.items(),
                   key=lambda kv: kv[1].get("priority", 99))
    if STATE.cursor_left < len(items):
        mid = items[STATE.cursor_left][0]
        hllm.CURRENT_MODEL = mid
        STATE.add_log(f"[모델 변경] → {mid}")


def action_toggle_skill():
    skills = hllm.list_skills()
    if STATE.cursor_left < len(skills):
        name = skills[STATE.cursor_left]["name"]
        if name in STATE.active_skills:
            STATE.active_skills.remove(name)
            STATE.add_log(f"[스킬 비활성] {name}")
        else:
            STATE.active_skills.append(name)
            STATE.add_log(f"[스킬 활성] ★ {name}")


def action_enter():
    """Enter 키 - 현재 view에 따라 다르게 동작"""
    if STATE.view == "models":
        action_select_model()
    elif STATE.view == "skills":
        action_toggle_skill()


def action_move_cursor(delta: int):
    if STATE.view == "models":
        items = list(hllm.MODELS.keys())
    elif STATE.view == "skills":
        items = hllm.list_skills()
    else:
        return
    n = len(items)
    if n == 0:
        return
    STATE.cursor_left = max(0, min(n - 1, STATE.cursor_left + delta))


# ─────────────────────────────────────────────
# 작업 모드 진입 (REPL)
# ─────────────────────────────────────────────
def enter_work_mode(live: Live):
    """Live 정지 → 일반 hllm REPL로 진입 → 끝나면 복귀"""
    live.stop()
    console.clear()
    console.print(Panel(
        f"[bold yellow]작업 모드[/]\n"
        f"모델: [cyan]{hllm.CURRENT_MODEL}[/]\n"
        f"스킬: {', '.join(STATE.active_skills) if STATE.active_skills else '(없음)'}\n\n"
        f"[dim]/exit 입력하면 Commander로 복귀[/]",
        border_style="yellow",
    ))

    # hllm REPL 호출
    cwd = os.getcwd()
    system = hllm.build_system_prompt(cwd, STATE.active_skills)
    messages: list[dict] = [{"role": "system", "content": system}]

    try:
        hllm.repl(messages, STATE.active_skills, cwd, auto_confirm=False)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        console.print(f"[red]작업 모드 오류: {e}[/]")

    STATE.add_log("[작업 모드 종료] Commander로 복귀")
    console.clear()
    live.start()


# ─────────────────────────────────────────────
# 키 처리
# ─────────────────────────────────────────────
def handle_key(key: str, live: Live) -> bool:
    """True 반환하면 종료"""
    # 펑션 키
    if key == "f1":
        STATE.view = "models"
        STATE.add_log("[F1] 도움말 - F1~F10 펑션바 사용")
    elif key == "f2":
        STATE.view = "models"
        STATE.cursor_left = 0
        STATE.add_log("[F2] 모델 view")
    elif key == "f3":
        STATE.view = "skills"
        STATE.cursor_left = 0
        STATE.add_log("[F3] 스킬 view")
    elif key == "f4":
        STATE.view = "context"
        STATE.cursor_left = 0
        STATE.add_log("[F4] 컨텍스트 view")
    elif key == "f5":
        STATE.add_log("[F5] 작업 모드 진입…")
        enter_work_mode(live)
    elif key == "f6":
        STATE.view = "sessions"
        STATE.cursor_left = 0
        STATE.add_log("[F6] 세션 view")
    elif key == "f7":
        STATE.add_log("[F7] system 프롬프트 재구성됨")
    elif key == "f8":
        # 컨텍스트/스킬 편집 (notepad 호출)
        if STATE.view == "context":
            os.system(f'notepad "{hllm.CONTEXT_FILE}"')
            STATE.add_log("[F8] context.md 편집 종료")
        elif STATE.view == "skills":
            skills = hllm.list_skills()
            if skills and STATE.cursor_left < len(skills):
                path = skills[STATE.cursor_left]["path"]
                os.system(f'notepad "{path}"')
                STATE.add_log("[F8] SKILL.md 편집 종료")
    elif key == "f9":
        STATE.add_log("[F9] (yolo 모드는 작업 시 /yolo로 토글)")
    elif key == "f10" or key == "esc":
        return True

    # 방향키
    elif key == "up":
        action_move_cursor(-1)
    elif key == "down":
        action_move_cursor(1)
    elif key == "page_up":
        action_move_cursor(-5)
    elif key == "page_down":
        action_move_cursor(5)
    elif key == "home":
        STATE.cursor_left = 0
    elif key == "end":
        if STATE.view == "models":
            STATE.cursor_left = len(hllm.MODELS) - 1
        elif STATE.view == "skills":
            STATE.cursor_left = max(0, len(hllm.list_skills()) - 1)

    # 엔터
    elif key == "enter":
        action_enter()

    # 빠른 메뉴 키
    elif key == "m":
        STATE.view = "models"
        STATE.cursor_left = 0
    elif key == "s":
        STATE.view = "skills"
        STATE.cursor_left = 0
    elif key == "c":
        STATE.view = "context"
    elif key == "q":
        return True

    return False


# ─────────────────────────────────────────────
# 키 입력 (readchar)
# ─────────────────────────────────────────────
def get_key() -> str:
    if not HAS_READCHAR:
        return input(">> ").strip().lower()

    k = readchar.readkey()
    # readchar.key 상수 매핑
    keymap = {
        readchar.key.UP: "up",
        readchar.key.DOWN: "down",
        readchar.key.LEFT: "left",
        readchar.key.RIGHT: "right",
        readchar.key.ENTER: "enter",
        readchar.key.ESC: "esc",
        readchar.key.PAGE_UP: "page_up",
        readchar.key.PAGE_DOWN: "page_down",
        readchar.key.HOME: "home",
        readchar.key.END: "end",
        readchar.key.F1: "f1",
        readchar.key.F2: "f2",
        readchar.key.F3: "f3",
        readchar.key.F4: "f4",
        readchar.key.F5: "f5",
        readchar.key.F6: "f6",
        readchar.key.F7: "f7",
        readchar.key.F8: "f8",
        readchar.key.F9: "f9",
        readchar.key.F10: "f10",
    }
    if k in keymap:
        return keymap[k]
    if isinstance(k, str) and len(k) == 1:
        return k.lower()
    return ""


# ─────────────────────────────────────────────
# 메인 루프
# ─────────────────────────────────────────────
def main():
    if not HAS_READCHAR:
        console.print("[red]readchar 패키지가 필요합니다.[/]")
        console.print("[dim]설치: pip install readchar[/]")
        sys.exit(1)

    # 초기 화면 클리어
    console.clear()

    layout = make_layout()
    update_layout(layout)

    with Live(layout, console=console, screen=True,
              refresh_per_second=10, auto_refresh=False) as live:

        STATE.add_log("hllm Commander 시작")
        STATE.add_log(f"모델: {hllm.CURRENT_MODEL}")
        STATE.add_log("F1 도움  F2 모델  F5 작업 시작  F10 종료")

        while True:
            update_layout(layout)
            live.refresh()

            try:
                key = get_key()
            except KeyboardInterrupt:
                break
            except Exception:
                continue

            if not key:
                continue

            try:
                should_exit = handle_key(key, live)
            except Exception as e:
                STATE.add_log(f"[ERROR] {e}")
                continue

            if should_exit:
                break

    console.clear()
    console.print("[bold yellow]hllm Commander 종료[/]")


if __name__ == "__main__":
    main()
