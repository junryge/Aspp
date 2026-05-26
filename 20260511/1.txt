"""
server.py — Hermes Web 백엔드 (새로 작성, 깨끗한 버전)

- Flask 로 정적 UI 서빙 + Hermes CLI subprocess 호출
- /api/health      : proxy / 환경 상태
- /api/skills      : ~/.hermes/skills 스캔 (없으면 scientific-skills 폴백)
- /api/skill/<id>  : 개별 SKILL.md 본문
- /api/chat        : SSE 스트리밍, hermes chat -q 호출
- /api/translate-skills : 영문 description → 한글 요약 (옵션)

실행: python server.py  [포트 (기본 8788)]
"""
import os
import re
import sys
import json
import socket
import subprocess
import shutil
from flask import Flask, request, Response, jsonify, stream_with_context, send_from_directory

# ─── 경로 / 포트 ─────────────────────────────────────────────────────────────
SELF_DIR = os.path.dirname(os.path.abspath(__file__))
SCIENTIFIC_BASE = os.path.dirname(SELF_DIR)

_HERMES_SKILLS = os.path.expanduser("~/.hermes/skills")
SKILLS_DIR = _HERMES_SKILLS if os.path.isdir(_HERMES_SKILLS) \
             else os.path.join(SCIENTIFIC_BASE, "scientific-skills")

PROXY_HOST = "127.0.0.1"
PROXY_PORT = 8765

# ─── Windows pywinpty (hermes prompt_toolkit 가 콘솔 요구) ─────────────────
_USE_WINPTY = False
_WINPTY_MOD = None
if sys.platform == "win32":
    try:
        import winpty as _WINPTY_MOD
        _USE_WINPTY = True
    except Exception:
        _USE_WINPTY = False

# ─── ANSI / TUI 노이즈 필터 ──────────────────────────────────────────────────
_ANSI_RE = re.compile(
    r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"
    r"|\x1b\[[\?>=!]?[0-9;]*[ -/]*[@-~]"
    r"|\x1b[()][AB012]"
    r"|\x1b[=>]"
    r"|\x1b\][0-9]+;[^\x07\x1b]*"
)
_BOXLINE_RE = re.compile(r"^[A-Z]?\s*[─═━┌┐└┘├┤┬┴┼│]{3,}\s*$")
_META_RE = re.compile(
    r"\b(Session|Elapsed|Context|Duration|Messages|Resume\sthis\ssession)\s*:|"
    r"hermes\s+--resume\s+\S+"
)
_THINK_RE = re.compile(
    r"^\s*(?:<think>|</think>|<thinking>|</thinking>|"
    r"\[?Inner monologue\]?|\[?Reasoning\]?\s*:)",
    re.IGNORECASE,
)
NOISE_PREFIXES = ("✔", "ℹ", "Initializing", "Initialized", "Query:")


def _strip_ansi(s):
    return _ANSI_RE.sub("", s or "")


def _normalize(s):
    return (s or "").strip().lstrip("​﻿\xa0")


def _is_proxy_alive():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.4)
            s.connect((PROXY_HOST, PROXY_PORT))
            return True
    except Exception:
        return False


def _find_hermes():
    for name in ("hermes", "hermes.exe"):
        p = shutil.which(name)
        if p:
            return p
    py_dir = os.path.dirname(sys.executable)
    for cand in (os.path.join(py_dir, "hermes.exe"),
                 os.path.join(py_dir, "Scripts", "hermes.exe")):
        if os.path.isfile(cand):
            return cand
    return None


# ─── 스킬 ─────────────────────────────────────────────────────────────────
_SKILL_CACHE = None
SKILLS_KR_CACHE_FILE = os.path.join(SELF_DIR, "skills_kr.json")


def _parse_skill_md(path, full=False):
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read() if full else f.read(4000)
    except Exception:
        return None
    name, desc = None, ""
    m = re.match(r"---\s*\n([\s\S]+?)\n---\s*\n?", text)
    body = text[m.end():] if m else text
    if m:
        front = m.group(1)
        nm = re.search(r"^name:\s*(.+?)$", front, re.MULTILINE)
        dm = re.search(r"^description:\s*(.+?)$", front, re.MULTILINE)
        if nm:
            name = nm.group(1).strip().strip('"').strip("'")
        if dm:
            desc = dm.group(1).strip().strip('"').strip("'")
    return name, desc[:300], (body if full else None)


def _categorize(sid):
    s = sid.lower()
    if any(k in s for k in ("review", "debug", "test", "qa-", "security", "audit", "owasp")):
        return "review"
    if any(k in s for k in ("doc", "pdf", "pptx", "xlsx", "docx", "markdown", "md-", "report",
                            "writing", "writer", "manuscript", "internal-comms", "weekly")):
        return "docs"
    if any(k in s for k in ("skill-", "prompt", "claude", "mcp", "engineer-", "anthropic",
                            "openai", "router", "brainstorm", "thinking",
                            "problem-solving", "sequential", "using-superpowers")):
        return "skill"
    if any(k in s for k in ("polars", "dask", "sql", "database", "postgres", "vaex", "data-")):
        return "data"
    if any(k in s for k in ("matplotlib", "plotly", "seaborn", "design", "ui-", "ux-",
                            "infographic", "visualization", "schematic", "drawio")):
        return "design"
    if any(k in s for k in ("devops", "guide-git", "docker", "kubernetes", "modal-",
                            "github-actions", "deployment")):
        return "ops"
    if any(k in s for k in ("biopython", "scanpy", "rdkit", "molecular", "scientific-",
                            "deepchem", "transformers", "scikit", "qiskit", "phylo",
                            "alphafold", "research", "hypothesis")):
        return "science"
    if any(k in s for k in ("python", "javascript", "typescript", "react", "frontend",
                            "backend", "guide-", "code", "developer", "-pro", "-expert")):
        return "code"
    return "etc"


SKILL_CATEGORIES = [
    {"key": "all",     "name": "전체"},
    {"key": "code",    "name": "코드 작성"},
    {"key": "review",  "name": "코드 리뷰·디버깅"},
    {"key": "docs",    "name": "문서·MD·PPT"},
    {"key": "skill",   "name": "스킬·에이전트"},
    {"key": "data",    "name": "데이터·DB"},
    {"key": "science", "name": "과학·리서치"},
    {"key": "design",  "name": "시각화·디자인"},
    {"key": "ops",     "name": "Git·DevOps"},
    {"key": "etc",     "name": "기타"},
]


def _scan_skills():
    global _SKILL_CACHE
    if _SKILL_CACHE is not None:
        return _SKILL_CACHE
    items = []
    if os.path.isdir(SKILLS_DIR):
        for entry in sorted(os.listdir(SKILLS_DIR)):
            d = os.path.join(SKILLS_DIR, entry)
            skill_md = os.path.join(d, "SKILL.md")
            if os.path.isdir(d) and os.path.isfile(skill_md):
                parsed = _parse_skill_md(skill_md)
                if parsed:
                    name, desc, _ = parsed
                    items.append({
                        "id": entry,
                        "name": name or entry,
                        "description": desc,
                        "category": _categorize(entry),
                    })
    _SKILL_CACHE = items
    return items


def _load_kr_cache():
    if not os.path.isfile(SKILLS_KR_CACHE_FILE):
        return {}
    try:
        with open(SKILLS_KR_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _save_kr_cache(d):
    try:
        with open(SKILLS_KR_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[kr-cache] 저장 실패: {e}")


# ─── Flask app ──────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static", static_url_path="/static")


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/health")
def health():
    return jsonify({
        "skills_dir": SKILLS_DIR,
        "skills_dir_exists": os.path.isdir(SKILLS_DIR),
        "proxy_alive": _is_proxy_alive(),
        "proxy_url": f"http://{PROXY_HOST}:{PROXY_PORT}",
        "hermes_binary": _find_hermes(),
        "winpty": _USE_WINPTY,
        "env_mode": "OFFICE" if os.path.exists(r"C:\연구과제") else "LOCAL",
    })


@app.route("/api/skills")
def api_skills():
    items = _scan_skills()
    kr = _load_kr_cache()
    for it in items:
        it["description_kr"] = kr.get(it["id"], "")
    return jsonify({
        "count": len(items), "skills": items,
        "kr_cached": sum(1 for it in items if it["description_kr"]),
        "categories": SKILL_CATEGORIES,
    })


@app.route("/api/skill/<sid>")
def api_skill_detail(sid):
    if not sid or "/" in sid or "\\" in sid or sid.startswith("."):
        return jsonify({"error": "invalid skill id"}), 400
    d = os.path.join(SKILLS_DIR, sid)
    skill_md = os.path.join(d, "SKILL.md")
    if not os.path.isfile(skill_md):
        return jsonify({"error": "SKILL.md not found", "id": sid}), 404
    parsed = _parse_skill_md(skill_md, full=True)
    if not parsed:
        return jsonify({"error": "parse failed"}), 500
    name, desc, body = parsed
    files = []
    for sub in ("references", "scripts", "assets"):
        sd = os.path.join(d, sub)
        if os.path.isdir(sd):
            for fn in sorted(os.listdir(sd)):
                if os.path.isfile(os.path.join(sd, fn)):
                    files.append(f"{sub}/{fn}")
    return jsonify({
        "id": sid, "name": name or sid, "description": desc,
        "body": body, "files": files,
    })


@app.route("/api/translate-skills", methods=["POST"])
def api_translate_skills():
    body = request.get_json(silent=True) or {}
    batch_size = max(1, min(20, int(body.get("batch_size", 8))))
    only_missing = bool(body.get("only_missing", True))
    items = _scan_skills()
    cache = _load_kr_cache()
    todo = []
    for it in items:
        if only_missing and cache.get(it["id"]):
            continue
        if not (it.get("description") or "").strip():
            cache[it["id"]] = ""
            continue
        todo.append(it)
    if not todo:
        return jsonify({"status": "complete", "translated": 0, "remaining": 0})
    if not _is_proxy_alive():
        return jsonify({"error": "프록시(:8765) 가 안 떠있음"}), 502

    import requests as _req
    batch = todo[:batch_size]
    prompt = (
        "다음 영문 스킬 설명을 각각 한국어 한 줄 (60자 이내) 로 요약하라.\n"
        "출력 형식 정확히:\n  1. [id] 한국어 요약\n  2. [id] 한국어 요약\n\n항목:\n"
        + "\n".join(f"{i+1}. [{it['id']}] {it['description'][:250]}"
                    for i, it in enumerate(batch))
    )
    try:
        r = _req.post(
            f"http://{PROXY_HOST}:{PROXY_PORT}/v1/chat/completions",
            json={"model": "gemma-4-31B-it",
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.1, "max_tokens": 2000, "stream": False},
            timeout=120,
        )
        r.raise_for_status()
        text = r.json().get("choices", [{}])[0].get("message", {}).get("content", "") or ""
    except Exception as e:
        return jsonify({"error": f"번역 호출 실패: {e}"}), 500
    parsed = {}
    for line in text.splitlines():
        m = re.match(r"^\d+\.\s*\[([^\]]+)\]\s*(.+)$", line.strip())
        if m:
            parsed[m.group(1).strip()] = m.group(2).strip()
    saved = 0
    for it in batch:
        if it["id"] in parsed:
            cache[it["id"]] = parsed[it["id"]]
            saved += 1
    _save_kr_cache(cache)
    return jsonify({
        "status": "ok", "translated": saved, "batch_size": len(batch),
        "remaining": max(0, len(todo) - len(batch)), "total": len(items),
        "cached_total": sum(1 for v in cache.values() if v),
    })


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.json or {}
    messages = data.get("messages") or []
    skill_ids = [s for s in (data.get("skills") or []) if s and isinstance(s, str)]
    raw_mode = bool(data.get("raw", False))
    session_id = (data.get("session_id") or "").strip()

    last_user = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content", "")
            if isinstance(c, str):
                last_user = c
            break
    echo_lines = {ln.strip() for ln in (last_user or "").splitlines() if ln.strip()}

    def gen():
        info = ("스킬: " + ", ".join(skill_ids)) if skill_ids else "자동 스킬 판단"
        yield f"data: {json.dumps({'type':'meta','info':info,'raw':raw_mode,'session_id_in':session_id or None}, ensure_ascii=False)}\n\n"

        if not _is_proxy_alive():
            yield f"data: {json.dumps({'type':'error','error':f'프록시(:{PROXY_PORT}) 안 떠있음','detail':f'python {SELF_DIR}/hermes_proxy.py'}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return
        if not last_user.strip():
            yield f"data: {json.dumps({'type':'error','error':'빈 메시지'}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return
        hermes_bin = _find_hermes()
        if not hermes_bin:
            yield f"data: {json.dumps({'type':'error','error':'hermes 명령 못 찾음'}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return

        cmd = [hermes_bin]
        if session_id:
            sess_dir = os.path.expanduser("~/.hermes/sessions")
            sess_exists = (
                os.path.isfile(os.path.join(sess_dir, f"session_{session_id}.json"))
                or os.path.isfile(os.path.join(sess_dir, f"{session_id}.json"))
            )
            if sess_exists:
                cmd += ["--resume", session_id]
                print(f"[chat] resume: {session_id}")
            else:
                print(f"[chat] session {session_id} 없음 → 새 세션")
        if skill_ids:
            cmd += ["-s", ",".join(skill_ids)]
        cmd += ["chat", "-q", last_user]

        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("TERM", "xterm-256color")
        print(f"[chat] cmd={cmd}  winpty={_USE_WINPTY}  raw={raw_mode}")

        state = {"sid": None, "body_started": False, "lines": 0, "emit": 0}

        def _filter_line(rawline):
            line = _strip_ansi(rawline.rstrip("\r\n"))
            mm = re.search(r"hermes\s+--resume\s+(\S+)", line)
            if mm:
                state["sid"] = mm.group(1)
                if not raw_mode:
                    return None
            if raw_mode:
                return (line + "\n") if line else None
            s = _normalize(line)
            if not s:
                return None
            if _BOXLINE_RE.match(s):
                return None
            if s in ("Q", "▶", "▶ "):
                return None
            if any(s.startswith(p) for p in NOISE_PREFIXES):
                return None
            if "⚕ Hermes" in s:
                return None
            if _META_RE.search(s):
                return None
            if _THINK_RE.match(s):
                return None
            if not state["body_started"] and s in echo_lines:
                return None
            state["body_started"] = True
            return s + "\n"

        def _emit(tok):
            state["emit"] += 1
            return f"data: {json.dumps({'type':'token','t':tok}, ensure_ascii=False)}\n\n"

        if _USE_WINPTY and _WINPTY_MOD is not None:
            try:
                proc = _WINPTY_MOD.PtyProcess.spawn(cmd, env=env, dimensions=(40, 240))
            except Exception as e:
                yield f"data: {json.dumps({'type':'error','error':f'winpty spawn 실패: {e}'}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return
            buf = ""
            try:
                while True:
                    try:
                        chunk = proc.read(1024)
                    except (EOFError, Exception):
                        break
                    if not chunk:
                        if not proc.isalive():
                            break
                        continue
                    buf += chunk
                    while True:
                        nl_idx, nl_len = -1, 0
                        for sep in ("\r\n", "\n", "\r"):
                            i = buf.find(sep)
                            if i != -1 and (nl_idx == -1 or i < nl_idx):
                                nl_idx, nl_len = i, len(sep)
                        if nl_idx == -1:
                            break
                        line_str = buf[:nl_idx]
                        buf = buf[nl_idx + nl_len:]
                        state["lines"] += 1
                        tok = _filter_line(line_str)
                        if tok:
                            yield _emit(tok)
                if buf.strip():
                    state["lines"] += 1
                    tok = _filter_line(buf)
                    if tok:
                        yield _emit(tok)
            finally:
                try:
                    if proc.isalive():
                        proc.terminate(force=True)
                except Exception:
                    pass
        else:
            try:
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace", bufsize=1, env=env,
                )
            except FileNotFoundError:
                yield f"data: {json.dumps({'type':'error','error':'hermes 명령 못 찾음'}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return
            try:
                for raw_line in proc.stdout:
                    state["lines"] += 1
                    tok = _filter_line(raw_line)
                    if tok:
                        yield _emit(tok)
            finally:
                try:
                    proc.stdout.close()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        proc.kill()
                    except Exception:
                        pass

        print(f"[chat] end: {state['lines']} 줄, {state['emit']} emit, sid={state['sid']}")
        yield f"data: {json.dumps({'type':'end','finish_reason':'stop','hermes_session_id':state['sid'],'stats':{'lines':state['lines'],'emitted':state['emit']}}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return Response(stream_with_context(gen()), mimetype="text/event-stream")


if __name__ == "__main__":
    port = 8788
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass

    if sys.platform == "win32":
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
        os.environ.setdefault("PYTHONUTF8", "1")
        try:
            os.system("chcp 65001 >nul 2>&1")
        except Exception:
            pass

    print("=" * 60)
    print(f"  Hermes Web (clean)")
    print(f"  Skills dir : {SKILLS_DIR}")
    print(f"               exists={os.path.isdir(SKILLS_DIR)}")
    print(f"  Proxy      : http://{PROXY_HOST}:{PROXY_PORT}  alive={_is_proxy_alive()}")
    print(f"  Hermes bin : {_find_hermes()}")
    print(f"  Winpty     : {_USE_WINPTY}")
    print(f"  Port       : {port}")
    print("=" * 60)
    print(f"  http://localhost:{port}  접속")
    print("=" * 60)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
