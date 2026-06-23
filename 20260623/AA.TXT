"""
demos_v1/routes_chat.py - /api/chat route (the massive chat endpoint) + /api/chat/stop
"""
import os
import re
import json


# ── 영문 사고 과정(Chain-of-Thought) 노출 방지 ──
# Qwen3 등 reasoning 모델이 시스템 프롬프트 지시를 무시하고 본문에 영문 사고 절차를 토함.
# 두 단계 방어: (1) <think> 태그 제거, (2) 응답 끝에서 거꾸로 한국어 본문만 추출
# 코드 블록(```)은 영문이라도 무조건 보존.
_EN_THINKING_PATTERNS = re.compile(
    r"(?:^|\s)(?:"
    r"I (?:will|'?ll|must|should|need to|cannot|don'?t|am going|have to|would|could|might)\b"
    r"|Let me\b|Let's\b|Looking at\b|Wait[,\s]|Now[,\s]|First[,\s]|Actually[,\s]"
    r"|Hmm[,\s]|Okay[,\s]|OK[,\s]|So[,\s]|The user (?:wants?|asks?|is|requested?|meant)"
    r"|Check constraint\b|Self[- ]Correction|Structure mapping"
    r"|^Generate\.\s*$|^Proceed\.\s*$|^Done\.\s*$|^\[Proceeds?\]"
    r"|Output Generation|Output matches|Output:\s|All aligned"
    r"|Here'?s (?:a|the|my)|Thinking process|Analyze User Input"
    r"|\[Output Generation\]|Step-by-step reasoning|Final (?:Check|decision|answer)"
    r"|It (?:seems|appears|looks)|This (?:means|implies|suggests)"
    r"|Therefore[,\s]|However[,\s]|Based on\b|To be (?:safe|extremely|precise|honest)"
    r"|^제목:\s|^요약:\s|^배경/목적:\s|^본문:\s*$|^결론 및 제언:\s"
    r")",
    re.IGNORECASE,
)
_KO_HEADER_RE = re.compile(r"^\s*#{1,6}\s+(?=[^\n]*[가-힯])")
_KO_CHAR_RE = re.compile(r"[가-힯]")
_EN_LETTER_RE = re.compile(r"[a-zA-Z]")


class _StreamThinkFilter:
    """스트리밍 중 사고과정(<think>) 실시간 제거. 토큰 경계로 태그가 쪼개져도 안전.
    implicit=True(예: Spark qwen3.6): 템플릿이 여는 <think> 를 프롬프트에 미리 넣어
    모델 출력이 사고로 바로 시작하고 닫는 </think> 만 나온다 → 첫 </think> 전까지 전부 사고로 보고 버린다.
    implicit=False: 명시적 <think>...</think> 만 제거(다른 모델은 평소대로 스트리밍)."""
    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self, implicit=False):
        self.buf = ""
        self.in_think = False
        self.implicit = implicit
        self.done = False   # 사고 종료(</think> 통과) 후엔 그냥 흘림

    def feed(self, tok):
        if self.done:
            return tok
        self.buf += tok
        # 암묵 모드: 첫 </think> 전까지 전부 사고 → 버리고, 그 뒤부터 본문
        if self.implicit and not self.in_think:
            i = self.buf.find(self.CLOSE)
            if i != -1:
                after = self.buf[i + len(self.CLOSE):]
                self.buf = ""
                self.done = True
                return after.lstrip("\n")
            return ""   # 아직 </think> 안 나옴 → 보류(전부 사고로 간주)
        out = ""
        while self.buf:
            if not self.in_think:
                i = self.buf.find(self.OPEN)
                if i == -1:
                    cut = self._safe_emit_len(self.buf, self.OPEN)
                    out += self.buf[:cut]
                    self.buf = self.buf[cut:]
                    break
                out += self.buf[:i]
                self.buf = self.buf[i + len(self.OPEN):]
                self.in_think = True
            else:
                j = self.buf.find(self.CLOSE)
                if j == -1:
                    keep = self._tail_partial_len(self.buf, self.CLOSE)
                    self.buf = self.buf[len(self.buf) - keep:] if keep else ""
                    break
                self.buf = self.buf[j + len(self.CLOSE):]
                self.in_think = False
        return out

    def flush(self):
        out = "" if self.in_think else self.buf
        self.buf = ""
        return out

    @staticmethod
    def _safe_emit_len(s, tag):
        for k in range(min(len(tag) - 1, len(s)), 0, -1):
            if s.endswith(tag[:k]):
                return len(s) - k
        return len(s)

    @staticmethod
    def _tail_partial_len(s, tag):
        for k in range(min(len(tag) - 1, len(s)), 0, -1):
            if s.endswith(tag[:k]):
                return k
        return 0


def _strip_thinking_artifacts(text):
    """응답에서 사고 과정 흔적 제거. 끝에서 거꾸로 한국어 본문만 추출."""
    if not text:
        return text
    cleaned = re.sub(r"<think>[\s\S]*?</think>\s*", "", text, flags=re.IGNORECASE)

    lines = cleaned.split("\n")
    code_block_idxs = set()
    in_block = False
    for i, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            code_block_idxs.add(i)
            in_block = not in_block
        elif in_block:
            code_block_idxs.add(i)

    keep_idxs = []
    body_started = False
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        stripped = line.strip()
        if i in code_block_idxs:
            keep_idxs.append(i)
            body_started = True
            continue
        if not stripped:
            if body_started:
                keep_idxs.append(i)
            continue
        if _KO_HEADER_RE.match(line):
            keep_idxs.append(i)
            body_started = True
            continue
        if _EN_THINKING_PATTERNS.search(line):
            break
        ko = len(_KO_CHAR_RE.findall(line))
        en = len(_EN_LETTER_RE.findall(line))
        if ko >= 3 and (en == 0 or ko * 2 >= en):
            keep_idxs.append(i)
            body_started = True
            continue
        if body_started and (stripped.startswith("|") or stripped == "---" or stripped.startswith(">")):
            keep_idxs.append(i)
            continue
        if body_started and en < 8:
            keep_idxs.append(i)
            continue
        break
    keep_idxs.sort()
    if not keep_idxs:
        return cleaned.strip()
    return "\n".join(lines[i] for i in keep_idxs).strip()


import time
import warnings
import requests as req
from demos_v1.llm_compat import chat_post
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import request, jsonify, Response, stream_with_context

import demos_v1.utils as _utils_mod
from demos_v1.utils import (
    BASE_DIR, SKILLS_DIR, UPLOAD_DIR,
    uploaded_csv_data, uploaded_files,
    chat_stop_flag,
    HARNESS_AVAILABLE,
)
from demos_v1.config import (
    TOKEN_SETTINGS, API_TOKEN,
    MAX_POOL_SIZE, VRAM_BUDGET_GB, _EXT_CONFIG,
)
from demos_v1.models import (
    MODEL_REGISTRY, ENV_CONFIG, ENV_TO_REGISTRY,
    FALLBACK_CHAINS, API_MODEL_TIERS,
)
from demos_v1.skills import (
    SKILL_DESC_KO, DOMAIN_SKILLS, SKILL_GROUPS, _SKILL_TO_GROUP,
    MANUAL_ONLY_SKILLS, SKILL_KEYWORDS,
    scan_skills, auto_select_skills, context_aware_skill_select,
    load_skill_content, get_skill_catalog,
    group_skills_for_parallel, apply_hierarchical_delegation,
    get_registry_key_for_env, get_model_capabilities,
)
from demos_v1.router import (
    classify_and_route, classify_format_and_style,
    build_orchestration_prompt,
    VISION_SIGNALS, COMPLEX_SIGNALS, PPT_SIGNALS, DATA_SIGNALS,
)
from logpresso_client import query_logpresso
from demos_v1.quality import (
    DOMAIN_PERSONAS, ANTI_RATIONALIZATION, VERIFICATION_GATE, ANALYSIS_LIFECYCLE,
    _extract_skill_context, _detect_repetition,
    _validate_response, _calculate_quality_score, _fix_response_issues,
    _sanitize_knowledge_content,
)
from demos_v1.engine import (
    _build_agent_system_prompt, _trim_history_for_context,
    _agent_call_gguf, _synthesize_responses_gguf,
    _api_agent_call,
    _assign_models_to_groups, _assign_api_models_to_groups,
)
from demos_v1.gguf import (
    find_gguf_files, load_gguf_model, gguf_chat,
    _pool_get_or_load, _pool_release, _pool_status,
)
from demos_v1.knowledge import search_knowledge, KNOWLEDGE_DIR, KNOWLEDGE_TRIGGERS
from demos_v1.rag_client import search_knowledge_smart
print("[routes_chat] ✅ 새 버전 로드됨 — 에이전트 지식=직접주입(read_selected) / 결과요약 제너릭 / argv")
from demos_v1.logpresso import (
    LOGPRESSO_TABLES, LOGPRESSO_TABLE_GROUPS, LOGPRESSO_FAB_FILTERS,
    _get_table_group, _filter_tables_by_groups, _fetch_table_fields,
    _llm_generate_lpql, extract_lpql_from_response, validate_lpql_readonly,
)


def _auto_save_feedback(loaded, quality, last_user_query):
    """응답 품질 점수를 피드백으로 자동 저장 (하네스 피드백 루프)"""
    try:
        from harness_bridge import _feedback_store
        if not _feedback_store or not loaded:
            return
        import time as _time
        from harness import FeedbackEntry
        score = quality.get('score', 0) if isinstance(quality, dict) else 0
        approved = score >= 60
        for sid in loaded[:5]:  # 상위 5개 스킬만
            _feedback_store.add(FeedbackEntry(
                timestamp=_time.time(),
                skill_id=sid,
                agent='app',
                quality_score=score,
                approved=approved,
                rejection_reason=None if approved else f"품질 점수 {score}/100 미달",
                query_context=(last_user_query or '')[:100],
            ))
    except Exception:
        pass


def _delatex_md(t):
    """$\\rightarrow$ 등 LaTeX 표기를 유니코드 기호로 치환 (프론트 delatexFallback 동일 매핑).
    python-markdown 은 수식을 못 다뤄 그대로 두면 HTML 다운로드 시 깨진다."""
    if not t:
        return t
    import re as _re
    _map = {
        '\\rightarrowtail': '→', '\\rightarrow': '→', '\\Rightarrow': '⇒', '\\to': '→',
        '\\leftarrow': '←', '\\Leftarrow': '⇐', '\\geq': '≥', '\\ge': '≥',
        '\\leq': '≤', '\\le': '≤', '\\neq': '≠', '\\ne': '≠', '\\approx': '≈',
        '\\times': '×', '\\cdot': '·', '\\pm': '±', '\\div': '÷', '\\infty': '∞',
        '\\alpha': 'α', '\\beta': 'β', '\\sum': '∑',
    }
    t = _re.sub(r'\\text\s*\{([^}]*)\}', r'\1', t)
    t = _re.sub(r'\\mathrm\s*\{([^}]*)\}', r'\1', t)
    for k, v in _map.items():
        t = t.replace(k, v)
    t = _re.sub(r'\$\$([\s\S]+?)\$\$', r'\1', t)
    t = _re.sub(r'\$([^$\n]+?)\$', r'\1', t)
    return t


def _maybe_generate_md_html(answer, loaded, resp_data):
    """md-to-html 스킬이 로드되었으면 MD/HTML 파일을 생성하고 다운로드 URL을 resp_data에 추가"""
    if "md-to-html" not in loaded:
        return resp_data

    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    uploads_dir = os.path.join(BASE_DIR, 'uploads')
    os.makedirs(uploads_dir, exist_ok=True)

    # 제목 자동 추출: 첫 번째 # 헤딩에서 가져옴
    title = "문서"
    for line in answer.split("\n"):
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith("## "):
            title = _delatex_md(stripped.lstrip("# ").strip())
            break

    # 1) MD 파일 저장
    md_filename = f"document_{timestamp}.md"
    md_path = os.path.join(uploads_dir, md_filename)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(answer)

    # 2) HTML 파일 생성
    try:
        import markdown as md_lib
        # 표 앞 빈 줄 보정 (표로 인식 못 해 '| ... |' 글자로 깨지는 것 방지)
        _md_in = []
        for _ln in _delatex_md(answer).split("\n"):
            if _ln.lstrip().startswith("|") and _md_in and _md_in[-1].strip() and not _md_in[-1].lstrip().startswith("|"):
                _md_in.append("")
            _md_in.append(_ln)
        extensions = ["tables", "fenced_code", "codehilite", "toc", "nl2br", "sane_lists"]
        body_html = md_lib.markdown("\n".join(_md_in), extensions=extensions)
        full_html = (
            '<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{title}</title><style>'
            'body{font-family:"Pretendard","Noto Sans KR",sans-serif;max-width:900px;margin:2rem auto;padding:0 1.5rem;color:#1a1a2e;line-height:1.7}'
            'h1,h2,h3{color:#16213e;border-bottom:2px solid #e2e8f0;padding-bottom:.3em}'
            'table{border-collapse:collapse;width:100%;margin:1em 0;font-size:.86em}'
            'th,td{border:1px solid #cbd5e1;padding:.4em .55em;text-align:left;vertical-align:top;word-break:keep-all;overflow-wrap:anywhere}'
            'th{background:#f1f5f9;font-weight:700;white-space:nowrap}'
            'code{background:#f1f5f9;padding:2px 6px;border-radius:4px;font-size:.9em}'
            'pre{background:#1e293b;color:#e2e8f0;padding:1em;border-radius:8px;overflow-x:auto}'
            'pre code{background:none;color:inherit;padding:0}'
            'blockquote{border-left:4px solid #6366f1;margin:1em 0;padding:.5em 1em;background:#f8fafc}'
            'a{color:#6366f1}img{max-width:100%;border-radius:8px}'
            '</style></head><body>'
            f'{body_html}</body></html>'
        )
        html_filename = f"document_{timestamp}.html"
        html_path = os.path.join(uploads_dir, html_filename)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(full_html)
        html_url = f"/api/download_static/document.html?id={timestamp}"
    except Exception:
        html_url = None

    md_url = f"/api/download_static/document.md?id={timestamp}"

    resp_data["md_download_url"] = md_url
    if html_url:
        resp_data["html_download_url"] = html_url
    resp_data["doc_download_id"] = timestamp

    return resp_data


# ============================================
# SSE 스트리밍 단순 경로 (단일 env + 단일 모델)
# ============================================
# 정책:
# - 사용자가 선택한 모델 한 개만 시도. 폴백 체인 없음.
#   (이전에 fallback chain 을 넣었을 때 "모든 모델 시도 실패" 가 발생함)
# - 수동 스킬: 사용자가 체크한 skill_ids 만 개별 SKILL.md 로드 (전수 enumeration 없음).
# - 자동 스킬은 이 경로에서 호출하지 않음 (이미 /api/auto_skills 에서 처리됨).
# - 하네스 라우터는 시작 시 미빌드 상태라 빈 결과 — 호출하지 않음.
# - 다중 env 병렬 합성이 필요한 경우 (env가 list 2개 이상) 는 SSE 안 함.
def _prepend_system(msgs, extra):
    """모든 system 메시지를 한 개로 합쳐 0번에 두고 나머지 user/assistant
    순서는 유지. extra 는 그 합쳐진 system 의 맨 앞에 들어감.
    vLLM/litellm 게이트웨이의 'System must be at beginning' / 다중-system
    거부 회피."""
    sys_parts = []
    if extra and str(extra).strip():
        sys_parts.append(str(extra).strip())
    rest = []
    for m in msgs or []:
        if m.get("role") == "system":
            c = m.get("content") or ""
            if isinstance(c, str) and c.strip():
                sys_parts.append(c.strip())
        else:
            rest.append(m)
    if not sys_parts:
        return rest
    return [{"role": "system", "content": "\n\n".join(sys_parts)}] + rest


def _gguf_n_ctx(model, default=32768):
    """로드된 GGUF 모델의 컨텍스트 길이(n_ctx) 반환. (메서드/속성 모두 대응)"""
    attr = getattr(model, "n_ctx", None)
    try:
        v = attr() if callable(attr) else attr
        return int(v) if v else default
    except Exception:
        return default


def _msg_text(m):
    c = m.get("content", "")
    if isinstance(c, list):
        c = "\n".join(str(p.get("text", "")) for p in c if isinstance(p, dict) and p.get("type") == "text")
    return str(c)


def _make_token_counter(model=None, tpc=2.6):
    """토큰 수 카운터 반환. 과소평가 절대 금지(과소평가하면 트림이 부족해 컨텍스트 초과).
    GGUF(크래시 위험)는 밀집 CSV 최악값 2.6 tokens/char 하한을 쓴다.
    API(128K 네이티브, 초과해도 서버가 에러로 처리)는 현실값(≈0.5)을 써서 정상 텍스트를 헛되이 자르지 않는다."""
    DENSE = tpc  # tokens per char 하한 (GGUF=2.6 보수적, API=0.5 현실적)
    def count(text):
        s = str(text)
        floor = int(len(s) * DENSE) + 1
        if model is not None:
            b = s.encode("utf-8", "ignore")
            try:
                return max(len(model.tokenize(b, add_bos=False)), floor)
            except Exception:
                try:
                    return max(len(model.tokenize(b)), floor)
                except Exception:
                    pass
        return floor
    return count


def _truncate_text_to_tokens(text, allow_tokens, count):
    """text 를 head+tail 형태로 allow_tokens 안에 들어오게 절단. (새 텍스트, 잘린_문자수) 반환.
    count 가 정확/보수적이면 루프가 실제로 맞을 때까지 줄인다."""
    total = count(text)
    if total <= allow_tokens:
        return text, 0
    ratio = len(text) / max(1, total)            # 문자/토큰 근사 (초기 추정용)
    keep = max(200, int(allow_tokens * ratio * 0.9))
    head = int(keep * 0.7)
    tail = keep - head
    marker = f"\n...[중략: 원본 {len(text):,}자 중 일부 생략]...\n"
    new = text[:head] + marker + (text[-tail:] if tail > 0 else "")
    # count 기준으로 실제 맞을 때까지 축소 (보수적 count 라 오버플로 방지 보장)
    _guard = 0
    while count(new) > allow_tokens and head > 120 and _guard < 40:
        head = int(head * 0.8)
        tail = int(tail * 0.8)
        new = text[:head] + marker + (text[-tail:] if tail > 0 else "")
        _guard += 1
    return new, max(0, len(text) - (head + max(0, tail)))


def _fit_messages_to_ctx(messages, n_ctx, reply_cap, model=None, safety=512,
                         hard_char_cap=True, tpc=2.6):
    """messages 가 (입력 + reply_cap + safety) <= n_ctx 가 되도록 트림/절단.
    1) 오래된 user/assistant 메시지 제거 → 2) 그래도 크면 가장 큰 메시지 본문 절단.
    hard_char_cap=True(GGUF): 토크나이저 무관 '하드 글자수' 안전장치까지 적용(크래시 방지).
    hard_char_cap=False(API): 토큰 추정 기반 트림만(128K 초과 시에만). 정상 텍스트를 헛되이 자르지 않음.
    반환: (트림된 messages, 응답토큰예산, 경고문)."""
    count = _make_token_counter(model, tpc)

    def mtoks(msgs):
        # 메시지당 chat 템플릿 오버헤드 넉넉히(+8), 전체 여유(+16)
        return sum(count(_msg_text(m)) + 8 for m in msgs) + 16

    msgs = [dict(m) for m in messages]
    warn = ""
    input_budget = max(512, n_ctx - reply_cap - safety)

    # 1) system·마지막 메시지 유지, 오래된 user/assistant 부터 제거
    while mtoks(msgs) > input_budget and len(msgs) > 1:
        idx = next((i for i, m in enumerate(msgs[:-1]) if m.get("role") in ("user", "assistant")), -1)
        if idx < 0:
            break
        msgs.pop(idx)

    # 2) 단일 대용량 메시지(붙여넣기 등) → 본문 절단
    if mtoks(msgs) > input_budget:
        big_i = max(range(len(msgs)), key=lambda i: len(_msg_text(msgs[i])))
        others = mtoks([m for j, m in enumerate(msgs) if j != big_i])
        allow = max(256, input_budget - others)
        new_text, removed = _truncate_text_to_tokens(_msg_text(msgs[big_i]), allow, count)
        if removed > 0:
            msgs[big_i] = dict(msgs[big_i], content=new_text)
            warn = (f"⚠️ [컨텍스트보호 v3] 입력이 한도({n_ctx:,} 토큰)를 넘어 약 {removed:,}자를 잘라 넣었습니다. "
                    f"전체를 보려면 더 큰 컨텍스트 모델(API 128K)을 쓰세요.")

    # 3) 토큰 추정과 무관한 '하드 글자수' 안전장치 — GGUF 전용(초과 시 크래시 방지).
    #    밀집 CSV(숫자·쉼표)는 문자당 토큰이 ~2.3개라, 글자수 자체를 input_budget/tpc 로 캡.
    #    API(hard_char_cap=False)는 이 단계를 건너뛴다 — 128K 네이티브라 헛절단 불필요.
    max_chars = max(400, int(input_budget / max(0.1, tpc)))
    total_chars = sum(len(_msg_text(m)) for m in msgs)
    if hard_char_cap and total_chars > max_chars:
        big_i = max(range(len(msgs)), key=lambda i: len(_msg_text(msgs[i])))
        others_chars = total_chars - len(_msg_text(msgs[big_i]))
        allow_chars = max(300, max_chars - others_chars)
        t = _msg_text(msgs[big_i])
        if len(t) > allow_chars:
            head = int(allow_chars * 0.7)
            tail = allow_chars - head
            marker = f"\n...[중략: 원본 {len(t):,}자 중 일부 생략]...\n"
            msgs[big_i] = dict(msgs[big_i], content=t[:head] + marker + (t[-tail:] if tail > 0 else ""))
            hard_removed = len(t) - (head + max(0, tail))
            if hard_removed > 0:
                warn = (f"⚠️ [컨텍스트보호 v3] 입력이 한도({n_ctx:,} 토큰)를 넘어 약 {hard_removed:,}자를 잘라 넣었습니다. "
                        f"전체를 보려면 더 큰 컨텍스트 모델(API 128K)을 쓰세요.")

    reply_budget = max(256, min(reply_cap, n_ctx - mtoks(msgs) - safety))
    return msgs, reply_budget, warn


def _extract_instruction(text):
    """대용량 붙여넣기에서 '사용자 요청' 부분을 추정(보통 앞/뒤에 위치)."""
    t = str(text).strip()
    if len(t) <= 600:
        return t
    return (t[:180].strip() + "\n…\n" + t[-450:].strip())


def _chunk_text_by_tokens(text, budget_tokens, count, hard_cap=64):
    """text 를 각 청크가 budget_tokens 이하가 되도록 순서대로 분할."""
    chunks = []
    i, n = 0, len(text)
    while i < n and len(chunks) < hard_cap:
        approx = max(400, int(budget_tokens / 1.6))   # 보수적 초기 길이(문자)
        j = min(n, i + approx)
        piece = text[i:j]
        guard = 0
        while count(piece) > budget_tokens and (j - i) > 300 and guard < 40:
            j = i + int((j - i) * 0.85)
            piece = text[i:j]
            guard += 1
        if j <= i:
            j = min(n, i + 400)
            piece = text[i:j]
        chunks.append(piece)
        i = j
    return chunks


def _plan_gguf_chunks(messages, n_ctx, reply_cap, model, safety=1536, max_chunks=20):
    """입력이 n_ctx 를 초과하면 '이어서 보기'(분할 처리) 계획 반환. 들어가면 None.
    반환: {base, instr, chunks, reply, n_ctx, safety, over}."""
    count = _make_token_counter(model)

    def mtoks(msgs):
        return sum(count(_msg_text(m)) + 8 for m in msgs) + 16

    if mtoks(messages) + reply_cap + safety <= n_ctx:
        return None   # 한 번에 들어감

    big_i = max(range(len(messages)), key=lambda i: len(_msg_text(messages[i])))
    big_text = _msg_text(messages[big_i])
    base = [dict(m) for j, m in enumerate(messages) if j != big_i]   # 시스템 등 공통 메시지
    instr = _extract_instruction(big_text)

    base_tokens = mtoks(base) if base else 16
    per_chunk_budget = n_ctx - reply_cap - safety - base_tokens - count(instr) - 80
    if per_chunk_budget < 800:
        instr = instr[-300:]
        per_chunk_budget = max(800, n_ctx - reply_cap - safety - base_tokens - count(instr) - 80)

    chunks = _chunk_text_by_tokens(big_text, per_chunk_budget, count)
    if len(chunks) <= 1:
        return None

    over = False
    if len(chunks) > max_chunks:
        chunks = chunks[:max_chunks]
        over = True
    return {"base": base, "instr": instr, "chunks": chunks,
            "reply": reply_cap, "n_ctx": n_ctx, "safety": safety, "over": over}


# ══ 실행형 지식(스크립트) 자동 실행 ═════════════════════════════════
# 개인에이전트가 '선택(체크)'한 스크립트 + 트리거어 + 업로드 데이터 → 스크립트 실행 →
# 결과 요약을 LLM 에 주입. LLM 은 선택한 문서지식(카파시톤/결과해석)으로 해석.
def _find_agent_script(user_id, query, selected_names):
    """선택된 스크립트 중 query 트리거 매칭되는 것 (meta, dir). 선택 없으면 None(안전)."""
    if not user_id or not selected_names:
        return None
    sroot = os.path.join(KNOWLEDGE_DIR, str(user_id), "scripts")
    if not os.path.isdir(sroot):
        return None
    sel = set(selected_names)
    q = query or ""
    # 후보를 모아 '가장 구체적인(긴) 트리거가 매칭된' 스크립트를 고른다.
    # (기존: 알파벳 첫 번째가 이기고, 트리거 없으면 무조건 매칭 → 엉뚱한 스크립트 오발동)
    best = None          # (matched_trigger_len, meta, dir)
    for nm in sorted(os.listdir(sroot)):
        if nm not in sel:
            continue
        mp = os.path.join(sroot, nm, "_meta.json")
        if not os.path.isfile(mp):
            continue
        try:
            meta = json.load(open(mp, encoding="utf-8"))
        except Exception:
            continue
        trig = [t for t in (meta.get("trigger") or []) if t]
        matched = [t for t in trig if t in q]
        if not matched:
            continue                              # 트리거 없거나 안 맞으면 실행 안 함(안전)
        score = max(len(t) for t in matched)      # 가장 긴(구체적) 트리거 길이
        if best is None or score > best[0]:
            best = (score, meta, os.path.join(sroot, nm))
    return (best[1], best[2]) if best else None


def _summarize_result_csv(path, max_full_rows=30, top_k=8):
    """결과 CSV → LLM용 압축 요약. 작은 표(사건단위)는 통째로, 큰 표(발동이벤트)는 분포+상위행."""
    import csv as _csv
    from collections import Counter
    try:
        rows = list(_csv.DictReader(open(path, encoding="utf-8-sig")))
    except Exception:
        return "(읽기 실패)"
    if not rows:
        return "(0행)"
    cols = list(rows[0].keys())
    n = len(rows)
    KEY = ["datetime", "time", "stage", "stage_name", "unified_risk_score", "unified_risk_level",
           "hot_area", "affected_areas", "reason", "predict_time", "start_time", "end_time",
           "lead_min", "duration_min", "max_risk_score", "max_risk_level", "triggered_rules",
           "risk_factors", "M16HUB_signals", "M14_signals", "M16A_signals", "M16B_signals"]
    keys = [c for c in KEY if c in cols] or cols[:10]
    out = [f"행수: {n}"]
    if n <= max_full_rows:
        for r in rows:
            line = " | ".join(f"{k}={r.get(k, '')}" for k in keys if str(r.get(k, '')).strip())
            if line:
                out.append(line)
    elif any(c in cols for c in KEY):
        # hubroom 스키마 결과(발동이벤트): 분포 + 위험 상위행
        for c in ("stage_name", "unified_risk_level", "hot_area"):
            if c in cols:
                cnt = Counter(str(r.get(c, "")).strip() or "-" for r in rows)
                out.append(f"[{c} 분포] " + ", ".join(f"{k}:{v}" for k, v in cnt.most_common(8)))
        if "unified_risk_score" in cols:
            def _score(r):
                try:
                    return float(r.get("unified_risk_score") or 0)
                except Exception:
                    return 0.0
            for r in sorted(rows, key=_score, reverse=True)[:top_k]:
                if _score(r) <= 0:
                    break
                out.append("· " + " | ".join(f"{k}={r.get(k, '')}" for k in keys if str(r.get(k, '')).strip()))
    else:
        # 일반(비-hubroom) 결과: 실제 행을 CSV 그대로 보여준다 (문자 예산 내). 큰 표면 앞부분만.
        out = ["컬럼: " + ",".join(cols), f"행수: {n}"]
        budget, used = 16000, 0
        for i, r in enumerate(rows):
            line = ",".join(str(r.get(c, "")) for c in cols)
            if used + len(line) + 1 > budget:
                out.append(f"...(표가 큼 — 위 {i}행까지만 표시, 총 {n}행. 전체는 결과 CSV 참조)")
                break
            out.append(line)
            used += len(line) + 1
    return "\n".join(out)


def _run_agent_script(user_id, query, csv_data, selected_names):
    """선택+트리거 매칭 스크립트 실행 → 결과 요약. 미매칭/실패 시 None."""
    found = _find_agent_script(user_id, query, selected_names)
    if not found:
        return None
    meta, sdir = found
    headers = csv_data.get("headers") or []
    rows = csv_data.get("rows") or []
    if not headers or not rows:
        return None
    import tempfile
    import subprocess
    import sys as _sys
    import glob
    import shutil
    import csv as _csv
    fd, tin = tempfile.mkstemp(suffix=".csv", prefix="an_in_")
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
        w = _csv.writer(f)
        w.writerow(headers)
        for r in rows:
            w.writerow(r)
    outd = tempfile.mkdtemp(prefix="an_out_")
    try:
        entry = meta.get("entry") or ""
        # 실행 인자: argv 템플릿 있으면 그대로(토큰 치환), 없으면 기본(hubroom식: <input> -o <outdir>).
        #   {input}=업로드 CSV 임시경로, {outdir}=결과폴더. 맨이름 동반파일은 cwd=sdir 라 자동 해석.
        argv_tpl = meta.get("argv") or []
        if argv_tpl:
            mapped = [t.replace("{input}", tin).replace("{outdir}", outd) for t in argv_tpl]
            cmd = [_sys.executable, os.path.join(sdir, entry)] + mapped
        else:
            cmd = [_sys.executable, os.path.join(sdir, entry), tin, "-o", outd]
        # encoding/errors 고정 — Windows 기본 cp949 로 한글 stdout 디코딩 시 UnicodeDecodeError 방지
        proc = subprocess.run(cmd,
                              capture_output=True, text=True, timeout=120, cwd=sdir,
                              encoding="utf-8", errors="replace",
                              env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1"))
        csvs = sorted(glob.glob(os.path.join(outd, "*.csv")) + glob.glob(os.path.join(outd, "*.CSV")))
        parts = [f"=== [분석 결과: {meta.get('name')}] — 스크립트가 계산한 결과입니다. 재계산 말고 이걸로 진단하세요 ==="]
        if csvs:
            for cf in csvs:
                parts.append(f"\n[{os.path.basename(cf)}]\n" + _summarize_result_csv(cf))
        elif (proc.stdout or "").strip():
            parts.append((proc.stdout or "")[:6000])
        else:
            parts.append(f"(산출 없음) {(proc.stderr or '')[:800]}")
        parts.append("\n\n→ 위 결과를 선택한 지식문서(카파시톤=원리 / 결과해석=컬럼사전)로 해석해 진단하세요.\n\n")
        return "\n".join(parts)
    except subprocess.TimeoutExpired:
        return f"=== [분석: {meta.get('name')}] 실행 시간 초과(120초). 데이터가 너무 큽니다. ===\n\n"
    except Exception as e:
        return f"=== [분석 오류: {meta.get('name')}] {e} ===\n\n"
    finally:
        try:
            os.remove(tin)
        except Exception:
            pass
        shutil.rmtree(outd, ignore_errors=True)


def _split_csv_text(text):
    """채팅에 '붙여넣은' CSV 추출 → (headers, rows, 지시문). 첨부 파일 없을 때 사용.
    콤마 많은 줄 = CSV(헤더+데이터), 나머지(예: '분석해줘') = 지시문."""
    lines = (text or "").splitlines()
    cand_idx = [i for i, ln in enumerate(lines) if ln.count(",") >= 5]
    if len(cand_idx) < 2:
        return None, None, text
    from collections import Counter
    ncols = Counter(lines[i].count(",") for i in cand_idx).most_common(1)[0][0]
    keep = set(i for i in cand_idx if abs(lines[i].count(",") - ncols) <= 2)
    if len(keep) < 2:
        return None, None, text
    import csv as _csv
    import io as _io
    csv_text = "\n".join(lines[i] for i in sorted(keep))
    try:
        rows = [r for r in _csv.reader(_io.StringIO(csv_text)) if len(r) >= 3]
    except Exception:
        return None, None, text
    if len(rows) < 2:
        return None, None, text
    instr = "\n".join(lines[i] for i in range(len(lines)) if i not in keep).strip()
    return rows[0], rows[1:], (instr or "이 데이터 분석해줘")


def _stream_chat_sse(data):
    chat_stop_flag["stop"] = False

    raw_env = data.get("env", "auto")
    if isinstance(raw_env, list):
        user_envs = [e for e in raw_env if e] or ["auto"]
    elif isinstance(raw_env, str):
        user_envs = [raw_env] if raw_env else ["auto"]
    else:
        user_envs = ["auto"]

    messages = data.get("messages", [])
    custom_system_prompt = (data.get("system_prompt") or "").strip()
    effort = data.get("effort", 2)
    temperature_map = {0: 0.0, 1: 0.2, 2: 0.4, 3: 0.7}

    # 단일 env만 지원. AUTO면 분류기로 해석.
    env_id = user_envs[0]
    auto_routed = False
    route_reason = None
    if env_id == "auto":
        last_query = ""
        if messages:
            last = messages[-1]
            c = last.get("content", "")
            last_query = c if isinstance(c, str) else ""
        try:
            env_id, route_reason = classify_and_route(last_query, messages, uploaded_files)
            auto_routed = True
        except Exception:
            env_id = next(iter(ENV_CONFIG.keys()), env_id)

    if env_id not in ENV_CONFIG:
        return jsonify({"error": f"알 수 없는 env: {env_id}"}), 400

    # 수동 선택 스킬 본문 로드 — 체크된 ID 만, 개별로.
    skill_ids = data.get("skills") or []
    is_gguf = str(env_id).startswith("gguf-")
    skill_section = ""
    loaded_skills = []
    if skill_ids:
        per_skill_cap = 6000 if is_gguf else 999999
        for sid in skill_ids:
            try:
                content = load_skill_content(sid)
            except Exception:
                content = ""
            if not content:
                continue
            if len(content) > per_skill_cap:
                content = content[:per_skill_cap] + f"\n... ({len(content)}자 중 {per_skill_cap}자 로드)"
            skill_section += f"\n=== SKILL: {sid} ===\n{content}\n"
            loaded_skills.append(sid)

    sys_parts = []
    if custom_system_prompt:
        sys_parts.append(custom_system_prompt)
    if skill_section:
        sys_parts.append(skill_section.strip())
    if loaded_skills:
        sys_parts.append(f"[로드된 스킬: {', '.join(loaded_skills)}]")

    # === 업로드된 CSV / 첨부 파일 주입 (JSON 경로와 동일 포맷) ===
    # 호출 시점에 demos_v1.utils 의 최신 객체를 다시 가져옴
    # (Python `from X import Y` 는 모듈 로드 시점의 바인딩이므로, 다른 코드가
    # 의도치 않게 utils.uploaded_files 를 재할당했을 경우 stale 이 될 수 있음).
    from demos_v1 import utils as _utils_now
    _csv_now = _utils_now.csv_slot(data.get("user_id"))   # ★ 사용자별 첨부 슬롯 (다중 사용자 충돌 방지)
    _files_now = _utils_now.uploaded_files

    file_section = ""
    include_csv = data.get("include_csv", True)

    # ── 스크립트 자동 실행: 데이터 출처 = 첨부 CSV(우선) 또는 채팅에 붙여넣은 CSV ──
    _q_now = ""
    _q_idx = -1
    for _i in range(len(messages) - 1, -1, -1):
        if messages[_i].get("role") == "user" and isinstance(messages[_i].get("content"), str):
            _q_now = messages[_i]["content"]
            _q_idx = _i
            break
    _data_for_script = None
    _pasted_instr = None
    if _csv_now.get("filename") and _csv_now.get("headers") and _csv_now.get("rows"):
        _data_for_script = {"headers": _csv_now["headers"], "rows": _csv_now["rows"]}   # 첨부
    else:
        _ph, _pr, _pi = _split_csv_text(_q_now)                                          # 붙여넣기
        if _ph and _pr:
            _data_for_script = {"headers": _ph, "rows": _pr}
            _pasted_instr = _pi
    _script_out = None
    if _data_for_script:
        _script_out = _run_agent_script(data.get("user_id"), _q_now, _data_for_script,
                                        data.get("knowledge_scripts") or [])
    try:
        print(f"  🐍 [SCRIPT-HOOK] user_id={data.get('user_id')!r} | knowledge_scripts={data.get('knowledge_scripts')!r} | "
              f"첨부CSV={bool(_csv_now.get('filename'))} | 붙여넣기CSV={bool(_data_for_script and _pasted_instr is not None)} | "
              f"데이터행={len((_data_for_script or {}).get('rows') or [])} | 스크립트실행={'O' if _script_out else 'X'}")
    except Exception:
        pass

    if _script_out:
        file_section += _script_out          # 스크립트가 계산 → LLM은 결과만 해석(raw 안 봄)
        if _pasted_instr is not None and _q_idx >= 0:
            # 붙여넣은 raw CSV 는 LLM 에 보내지 않고 지시문만 남김 (토큰 폭발 방지)
            messages[_q_idx] = dict(messages[_q_idx], content=_pasted_instr)
    elif include_csv and _csv_now.get("filename"):
        csv_info = _csv_now.get("summary", "")
        rows = _csv_now.get("rows", []) or []
        headers_csv = _csv_now.get("headers", []) or []
        preview_limit = min(50, len(rows))
        csv_rows_text = ",".join(headers_csv) + "\n"
        for row in rows[:preview_limit]:
            csv_rows_text += ",".join(str(c) for c in row) + "\n"
        if len(rows) > preview_limit:
            csv_rows_text += f"... (총 {len(rows)}행 중 {preview_limit}행만 표시)\n"
        file_section += f"=== 업로드된 CSV 데이터 ===\n{csv_info}\n\n데이터 미리보기:\n{csv_rows_text}\n\n"
        file_section += "사용자가 이 데이터에 대해 질문하면 위 CSV 데이터를 기반으로 분석해주세요.\n\n"

    if _files_now:
        file_section += f"=== 업로드된 파일 ({len(_files_now)}개) ===\n"
        for uf in _files_now:
            file_section += f"\n--- 파일: {uf.get('filename','?')} ({uf.get('type','?')}, {uf.get('size',0)}바이트) ---\n"
            content = uf.get("content_full", "") or ""
            cap = 8000 if is_gguf else 50000
            shown = content[:cap]
            if len(content) > cap:
                shown += f"\n... (총 {len(content)}자 중 {cap}자 표시)"
            file_section += shown + "\n"
        file_section += "\n사용자가 업로드된 파일에 대해 질문하면 위 내용을 기반으로 답변하세요.\n\n"

    # 진단 로그 — 매 SSE 호출 시 globals 상태 출력 (사용자 디버깅용)
    try:
        _fnames = [f.get("filename") for f in (_files_now or [])]
    except Exception:
        _fnames = []
    print(f"  📎 [SSE-FILES] CSV={_csv_now.get('filename') or '(none)'} | files={_fnames} | section_chars={len(file_section)}")

    if file_section:
        sys_parts.append(file_section.strip())

    sys_parts.append(
        "[응답 규칙] 사고 과정 sentinel(<think>...</think>, <|channel|>thought, "
        "<|im_start|>thinking 등)과 도구 호출 태그(<knowledge-search/>, <tool/>, "
        "<function/>, <search/> 등) 절대 출력 금지. 검색/스킬 결과는 이미 위 시스템 "
        "메시지에 제공되어 있으니, 그것만 보고 사용자 질문에 즉시 한국어로 답변 시작."
    )

    print(f"  📦 [SSE-SKILLS] 요청 skills={skill_ids!r} → 본문 로드 성공: {loaded_skills} ({len(skill_section)}자)")
    if skill_ids and not loaded_skills:
        print(f"  ⚠️ [SSE-SKILLS] 요청된 스킬 중 SKILL.md 본문 로드 0건 — 폴더 비었거나 ID 매칭 실패")

    api_messages = list(messages)
    if sys_parts:
        merged_system = "\n\n".join(sys_parts)
        api_messages = [{"role": "system", "content": merged_system}] + [
            m for m in api_messages if m.get("role") != "system"
        ]

    # knowledge-search 스킬: 도메인 지식 검색 후 결과를 system 메시지로 prepend
    last_user_query = ""
    for _m in reversed(messages):
        if _m.get("role") == "user":
            _c = _m.get("content", "")
            last_user_query = _c if isinstance(_c, str) else ""
            break
    # 개인에이전트 선택 지식문서: RAG 로 '선택한 문서의 관련 청크만' 빠르게 주입.
    #   files 필터 + 폴백버그 수정 덕에 '고른 문서'만 들어감(엉뚱한 문서 자동주입 없음).
    #   RAG 가 그 문서에서 못 찾으면 search_knowledge_smart 가 직접읽기로 폴백(무손실).
    _agent_kfiles = data.get("knowledge_files") or []
    if _agent_kfiles:
        try:
            # 검색어: 마지막 user 질문에서 '(지시: ...)' 꼬리표 제거 → RAG 매칭 정확도↑
            _q_kf = (last_user_query or "").split("\n\n(지시:")[0].strip() or (last_user_query or "")
            if _q_kf.strip():
                _kf_res = search_knowledge_smart(_q_kf, max_results=8, max_content_chars=4000,
                                                 user_id=data.get("user_id"), files=_agent_kfiles)
            else:
                from demos_v1.rag_client import read_selected as _read_selected
                _kf_res = _read_selected(data.get("user_id"), _agent_kfiles)
            if _kf_res:
                _kf_ctx = ("\n\n=== 선택한 내 지식 문서 ===\n"
                           "아래 문서 내용을 우선 근거로 답하세요. 문서에 없는 내용은 추측하지 말고 일반 지식임을 밝히세요.\n\n")
                _tot = 0
                for _r in _kf_res:
                    _seg = _r["content"][:4000]
                    if _tot + len(_seg) > 14000:
                        _seg = _seg[:max(0, 14000 - _tot)]
                        if not _seg:
                            break
                    _kf_ctx += f"--- 📄 {_r['filename']} ---\n{_seg}\n\n"
                    _tot += len(_seg)
                api_messages = _prepend_system(api_messages, _kf_ctx)
                try:
                    from demos_v1.rag_client import _healthy as _rh
                    _ksrc = "RAG" if _rh() else "BM25/직접"
                except Exception:
                    _ksrc = "?"
                print(f"  [AGENT-KFILES] 검색원={_ksrc} | {len(_kf_res)}개 파일 청크 주입 ({_tot}자) | 선택={_agent_kfiles}")
        except Exception as _e:
            print(f"  [AGENT-KFILES] {_e}")

    if "knowledge-search" in skill_ids and last_user_query.strip():
        try:
            _chat_user_id = data.get("user_id", None)
            print(f"  [KNOWLEDGE-SSE] user_id={_chat_user_id}, query={last_user_query[:50]}")
            # ★ 지식검색은 BM25/점수(키워드 매칭) 사용 — '주간 보고' 같은 키워드 질의에
            #    의미검색(RAG)이 엉뚱한 청크를 가져오던 문제로 원복.
            kb_results = search_knowledge(
                last_user_query, max_results=10, max_content_chars=4000, user_id=_chat_user_id
            )
            _src = "BM25/점수"
            print(f"  [KNOWLEDGE-SSE] 검색원={_src} | 결과 {len(kb_results)}건 | "
                  f"{sum(len(r.get('content','')) for r in kb_results)}자")
            if not kb_results:
                # 0건이면 한 줄로 끝내지 말고 일반 지식/다른 스킬로 계속 답변하도록 override.
                _other_skills = [s for s in skill_ids if s != "knowledge-search"]
                _other_note = (
                    f" 함께 선택된 다른 스킬({', '.join(_other_skills)})도 정상적으로 사용해 답변하세요."
                    if _other_skills else ""
                )
                override_msg = (
                    "[지식 검색 결과: 0건]\n"
                    f"knowledge-search 스킬은 '{last_user_query[:80]}' 에 대한 등록된 도메인 문서를 찾지 못했습니다.\n"
                    "응답 지침:\n"
                    "1. '등록된 도메인 문서에서는 관련 자료를 찾지 못했습니다.' 한 문장으로만 짧게 알리세요.\n"
                    "2. 그 후 사용자의 실제 요청을 끝까지 수행하세요 — 일반 지식으로 설명/분석, "
                    "관련 없는 다른 질문에 답하기, 새로운 코드/문서/아이디어를 창작·구성 등.\n"
                    f"3. '등록된 지식이 없습니다' 한 줄로 끝내고 응답을 종료하지 마세요.{_other_note}\n"
                )
                api_messages = _prepend_system(api_messages, override_msg)
            else:
                kb_context = "\n\n=== 도메인 지식 검색 결과 ===\n"
                kb_context += f"검색어: {last_user_query}\n\n"
                total_chars = 0
                for r in kb_results:
                    chunk = _sanitize_knowledge_content(r["content"][:4000])
                    if total_chars + len(chunk) > 12000:
                        chunk = chunk[: max(0, 12000 - total_chars)]
                        if not chunk:
                            break
                    kb_context += f"--- 📄 {r['filename']} (관련도: {r['score']}) ---\n"
                    kb_context += chunk + "\n\n"
                    total_chars += len(chunk)
                _ql = last_user_query.lower()
                _is_search_only = any(
                    kw in _ql for kw in ["검색해", "검색 해", "찾아봐", "찾아 봐", "뭐있", "뭐 있", "목록", "리스트", "파일명", "문서 목록"]
                )
                _is_content_request = any(
                    kw in _ql for kw in ["내용", "관련", "알려", "설명", "분석", "요약", "만들어"]
                )
                if _is_search_only and not _is_content_request:
                    kb_context += (
                        "사용자가 '검색'을 요청했습니다. 파일명 목록과 관련도 점수만 간단히 보여주세요.\n"
                        "문서 내용을 분석하거나 요약하지 마세요. 파일명 리스트만 출력하세요.\n"
                        "형식 예시:\n1. 📄 파일명.md (관련도: 75)\n2. 📄 파일명.md (관련도: 73)\n"
                    )
                else:
                    kb_context += (
                        "위 문서를 기반으로 사용자 질문에 답변하세요. 문서에 없는 내용을 지어내지 마세요. 어떤 문서에서 정보를 찾았는지 출처를 명시하세요.\n"
                        "중요: 프로토콜 메시지, raw 데이터, hex/binary는 절대 그대로 복사하지 마세요. 반드시 표(table) 또는 필드별 설명으로 변환하세요.\n"
                    )
                kb_context += (
                    "\n⚠️ 위 검색 결과는 이미 시스템이 제공한 최종 결과입니다. "
                    "<knowledge-search/>, <tool/>, <function/> 같은 도구 호출 태그를 출력하지 마세요. "
                    "사고 채널(<|channel|>, <think>)도 출력 금지. 바로 답변 본문부터 시작하세요.\n"
                )
                api_messages = _prepend_system(api_messages, kb_context)
        except Exception as e:
            print(f"[Knowledge Search SSE] 검색 오류: {e}")

    # === VL 모델: 이미지 첨부 시 마지막 user 메시지를 OpenAI Vision 포맷으로 변환 ===
    try:
        is_gguf_vl = str(env_id).startswith("gguf-") and "vl" in ENV_CONFIG.get(env_id, {}).get("name", "").lower()
        has_vision = ("vision" in get_model_capabilities(env_id)) or is_gguf_vl
        image_files = [f for f in (_files_now or []) if f.get("type") == "image" and f.get("img_base64")]
        if has_vision and image_files and api_messages:
            for i in range(len(api_messages) - 1, -1, -1):
                if api_messages[i].get("role") == "user":
                    text_content = api_messages[i].get("content", "")
                    if isinstance(text_content, str):
                        content_parts = [{"type": "text", "text": text_content}]
                        for img_f in image_files:
                            ext = (img_f.get("ext", "png") or "png").lower()
                            mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif",
                                    "bmp": "bmp", "webp": "webp", "svg": "svg+xml"}.get(ext, "png")
                            content_parts.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/{mime};base64,{img_f['img_base64']}"}
                            })
                        api_messages[i]["content"] = content_parts
                        print(f"  🖼️  [SSE-VL] 마지막 user 메시지에 이미지 {len(image_files)}장 첨부")
                    break
    except Exception as _ve:
        print(f"  ⚠️ [SSE-VL] 이미지 첨부 변환 실패: {_ve}")

    # ── GGUF 경로 ──
    if str(env_id).startswith("gguf-"):
        # qwen3.x GGUF: /no_think 로 사고 비활성화(템플릿이 인식) → 답만 생성. 누출/과부하/지연 방지.
        if not data.get("think_mode", False):
            for _m in reversed(api_messages):
                if _m.get("role") == "user" and isinstance(_m.get("content"), str):
                    if "/no_think" not in _m["content"]:
                        _m["content"] = _m["content"].rstrip() + " /no_think"
                    break
        gguf_path = ENV_CONFIG.get(env_id, {}).get("_gguf_path")
        user_n_ctx = data.get("n_ctx", 0)
        _load_n_ctx = user_n_ctx if user_n_ctx > 0 else 32768
        if gguf_path:
            if not load_gguf_model(gguf_path, n_ctx=_load_n_ctx):
                return jsonify({"error": f"GGUF 모델 로드 실패: {os.path.basename(gguf_path)}"}), 500
        if _utils_mod.gguf_model is None:
            return jsonify({"error": "GGUF 모델이 로드되지 않았습니다."}), 400

        gguf_reply_cap = max(256, TOKEN_SETTINGS.get("gguf_reply_cap", 4096))
        max_tokens_g = min(data.get("max_tokens", gguf_reply_cap), gguf_reply_cap)
        model_name = os.path.basename(gguf_path) if gguf_path else env_id

        _gctx = _gguf_n_ctx(_utils_mod.gguf_model)
        _greserve = max(512, TOKEN_SETTINGS.get("gguf_ctx_reserve", 1536))
        _temp = temperature_map[min(effort, 3)]
        # 컨텍스트 보호 토글. GGUF 는 n_ctx 를 물리적으로 못 넘으므로(초과=크래시) 토큰트림은 항상 유지.
        #  · ON(기본) → 토큰트림 + 하드 글자수캡(밀집CSV 안전망)
        #  · OFF      → 토큰트림만(실제 토크나이저로 n_ctx 에 맞춤) — 과한 글자수캡은 끔
        _ctx_guard = data.get("ctx_guard", True)

        def _tok_evt(t):
            return f"data: {json.dumps({'type': 'token', 't': t}, ensure_ascii=False)}\n\n"

        def _gguf_stream_once(msgs, max_toks):
            """한 번의 create_chat_completion 을 스트리밍 — (yield용 SSE, 누적텍스트, finish) 제너레이터.
            마지막에 ('__end__', acc, finish) 튜플을 yield.
            qwen3.x GGUF 는 템플릿이 <think> 를 미리 넣어 닫는 </think> 만 나오므로 암묵 사고제거 적용."""
            acc = ""
            _finish = None
            _gtf = _StreamThinkFilter(implicit=("qwen3" in (model_name or "").lower()))
            for chunk in _utils_mod.gguf_model.create_chat_completion(
                    messages=msgs, temperature=_temp, max_tokens=max_toks, stream=True):
                if chat_stop_flag.get("stop"):
                    break
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                tok = delta.get("content") or ""
                if tok:
                    vis = _gtf.feed(tok)          # 사고과정 실시간 제거
                    if vis:
                        acc += vis
                        yield _tok_evt(vis)
                fr = choices[0].get("finish_reason")
                if fr:
                    _finish = fr
            tail = _gtf.flush()
            if tail:
                acc += tail
                yield _tok_evt(tail)
            yield ("__end__", acc, _finish)

        # 분할(이어서보기) 비활성화 — 항상 단일 패스로 처리(큰 입력은 앞부분 잘라 한 번에 답변).
        # ('부분 N/20' 식 분할이 보고서/지식 답변을 조각내고 영어 사고과정 누출을 키워서 끔.)
        _plan = None

        def _meta_evt():
            _sys_len = sum(len(m.get("content", "") or "") for m in api_messages if m.get("role") == "system")
            meta = {
                "type": "meta", "env": env_id, "model": model_name,
                "loaded_skills": loaded_skills,
                "system_prompt_length": _sys_len,
                "auto_routed": auto_routed, "route_reason": route_reason,
            }
            return f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"

        if _plan is None:
            # ── 단일 패스 (필요 시 트림) ──
            _fit_msgs, _fit_reply, _fit_warn = _fit_messages_to_ctx(
                api_messages, _gctx, max_tokens_g,
                model=_utils_mod.gguf_model, safety=_greserve, hard_char_cap=_ctx_guard)

            def gen_gguf():
                yield _meta_evt()
                if _fit_warn:
                    yield _tok_evt(_fit_warn + "\n\n")
                try:
                    _finish = None
                    for ev in _gguf_stream_once(_fit_msgs, _fit_reply):
                        if isinstance(ev, tuple) and ev and ev[0] == "__end__":
                            _finish = ev[2]
                        else:
                            yield ev
                    end_evt = {"type": "end", "finish_reason": _finish, "truncated": (_finish == "length")}
                    yield f"data: {json.dumps(end_evt, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'error': f'GGUF 오류: {str(e)}'}, ensure_ascii=False)}\n\n"
        else:
            # ── 분할 처리 (이어서 보기): N조각 순차 분석 → 종합 ──
            def gen_gguf():
                yield _meta_evt()
                _chunks = _plan["chunks"]; N = len(_chunks)
                _base = _plan["base"]; _instr = _plan["instr"]
                _sys_only = [m for m in _base if m.get("role") == "system"]
                intro = (f"📚 입력이 GGUF 컨텍스트({_plan['n_ctx']:,} 토큰)를 초과해 "
                         f"**{N}개 부분으로 나눠 순서대로 분석**한 뒤 종합합니다.\n")
                if _plan["over"]:
                    intro += "⚠️ 데이터가 매우 커서 앞부분 위주로 처리합니다.\n"
                yield _tok_evt(intro)
                _partials = []
                try:
                    for idx, piece in enumerate(_chunks, 1):
                        if chat_stop_flag.get("stop"):
                            yield "data: [DONE]\n\n"; return
                        yield _tok_evt(f"\n\n━━━━━ 📄 부분 {idx}/{N} ━━━━━\n\n")
                        framed = (f"[전체 {N}개 부분 중 {idx}번째 부분입니다. 이 부분 데이터에 대해 "
                                  f"아래 요청을 수행하세요. 전체 종합은 마지막에 따로 합니다.]\n\n"
                                  f"{piece}\n\n[사용자 요청]\n{_instr}")
                        _msgs = _base + [{"role": "user", "content": framed}]
                        _msgs, _rb, _ = _fit_messages_to_ctx(
                            _msgs, _plan["n_ctx"], _plan["reply"],
                            model=_utils_mod.gguf_model, safety=_plan["safety"], hard_char_cap=_ctx_guard)
                        _acc = ""
                        for ev in _gguf_stream_once(_msgs, _rb):
                            if isinstance(ev, tuple) and ev and ev[0] == "__end__":
                                _acc = ev[1]
                            else:
                                yield ev
                        _partials.append(_acc)

                    # 종합
                    if chat_stop_flag.get("stop"):
                        yield "data: [DONE]\n\n"; return
                    yield _tok_evt("\n\n━━━━━ ✅ 종합 ━━━━━\n\n")
                    combined = "\n\n".join(f"[부분 {i+1}]\n{p}" for i, p in enumerate(_partials))
                    synth_user = (f"다음은 큰 데이터를 {N}개로 나눠 각각 분석한 결과입니다. "
                                  f"이를 하나로 종합해 사용자 요청에 대한 최종 결론을 작성하세요.\n\n"
                                  f"[사용자 요청]\n{_instr}\n\n[부분 분석 결과]\n{combined}")
                    _smsgs = _sys_only + [{"role": "user", "content": synth_user}]
                    _smsgs, _srb, _ = _fit_messages_to_ctx(
                        _smsgs, _plan["n_ctx"], _plan["reply"],
                        model=_utils_mod.gguf_model, safety=_plan["safety"], hard_char_cap=_ctx_guard)
                    _finish = None
                    for ev in _gguf_stream_once(_smsgs, _srb):
                        if isinstance(ev, tuple) and ev and ev[0] == "__end__":
                            _finish = ev[2]
                        else:
                            yield ev
                    end_evt = {"type": "end", "finish_reason": _finish, "truncated": False}
                    yield f"data: {json.dumps(end_evt, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'error': f'GGUF 분할 처리 오류: {str(e)}'}, ensure_ascii=False)}\n\n"

        return Response(stream_with_context(gen_gguf()), mimetype="text/event-stream")

    # ── API 경로 (단일 모델, 폴백 없음) ──
    api_url = ENV_CONFIG[env_id]["url"]
    model = ENV_CONFIG[env_id]["model"]
    # 모델별 전용 토큰(예: Spark) 우선 → 없으면 글로벌 → 요청값
    api_key = ENV_CONFIG[env_id].get("token") or API_TOKEN or data.get("api_key", "")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    max_tokens_a = data.get("max_tokens", TOKEN_SETTINGS.get("agent_max_tokens", 4096))

    # 컨텍스트 보호(API): 토글로 제어. 기본 ON.
    #  · ON  → 모델 context_window(128K) 안에 들어오도록 '토큰 추정' 기반으로만 트림(현실값 0.5 tpc, 하드 글자수캡 없음).
    #          → 일반 텍스트는 12만 자여도 ~6만 토큰이라 안 잘림. 진짜 128K 초과 시에만 트림.
    #  · OFF → 전혀 자르지 않고 원문 그대로 전송(128K 네이티브 신뢰).
    _ctx_guard = data.get("ctx_guard", True)
    _api_fit_warn = ""
    if _ctx_guard:
        _api_reg_key = ENV_TO_REGISTRY.get(env_id)
        _api_ctx = MODEL_REGISTRY.get(_api_reg_key, {}).get("context_window", 128000) if _api_reg_key else 128000
        api_messages, max_tokens_a, _api_fit_warn = _fit_messages_to_ctx(
            api_messages, _api_ctx, max_tokens_a, model=None, safety=1024,
            hard_char_cap=False, tpc=0.5)

    payload = {
        "model": model,
        "messages": api_messages,
        "temperature": temperature_map[min(effort, 3)],
        "max_tokens": max_tokens_a,
        "stream": True,
    }

    def gen_api():
        _tf = _StreamThinkFilter()   # 명시적 <think>...</think> 만 제거 (안전)
        _sys_len = sum(len(m.get("content","") or "") for m in api_messages if m.get("role") == "system")
        meta = {
            "type": "meta", "env": env_id, "model": model,
            "loaded_skills": loaded_skills,
            "system_prompt_length": _sys_len,
            "auto_routed": auto_routed,
            "route_reason": route_reason,
        }
        yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"
        # 입력이 잘렸으면 사용자에게 먼저 안내
        if _api_fit_warn:
            yield f"data: {json.dumps({'type': 'token', 't': _api_fit_warn + chr(10)+chr(10)}, ensure_ascii=False)}\n\n"
        try:
            r = chat_post(api_url, headers=headers, json=payload,
                         timeout=180, verify=False, stream=True)
            if r.status_code >= 400:
                detail = ""
                try:
                    detail = r.text[:500]
                except Exception:
                    pass
                err = {"type": "error", "error": f"HTTP {r.status_code}",
                       "detail": detail, "model": model}
                yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
                return
            for line in r.iter_lines(decode_unicode=True):
                if chat_stop_flag.get("stop"):
                    yield "data: [DONE]\n\n"
                    return
                if not line or not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    yield "data: [DONE]\n\n"
                    return
                try:
                    obj = json.loads(body)
                except Exception:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                # delta.reasoning_content(추론 채널)는 절대 본문으로 내보내지 않음 — 무시
                tok = delta.get("content") or ""
                if tok:
                    vis = _tf.feed(tok)          # <think> 구간 실시간 제거
                    if vis:
                        yield f"data: {json.dumps({'type': 'token', 't': vis}, ensure_ascii=False)}\n\n"
                fr = choices[0].get("finish_reason")
                if fr:
                    tail = _tf.flush()
                    if tail:
                        yield f"data: {json.dumps({'type': 'token', 't': tail}, ensure_ascii=False)}\n\n"
                    end_evt = {"type": "end", "finish_reason": fr, "truncated": (fr == "length")}
                    yield f"data: {json.dumps(end_evt, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            err = {"type": "error", "error": str(e), "model": model}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"

    return Response(stream_with_context(gen_api()), mimetype="text/event-stream")


def register_chat_routes(app):
    """Register chat routes on the Flask app."""

    @app.route("/api/chat/stop", methods=["POST"])
    def api_chat_stop():
        """진행 중인 LLM 응답을 중지"""
        chat_stop_flag["stop"] = True
        return jsonify({"stopped": True})


    @app.route("/api/chat", methods=["POST"])
    def api_chat():
        """LLM API 프록시 - 스킬을 시스템 프롬프트에 넣어서 회사 API로 전달.
        body.stream == true 이고 단일 env (auto 포함) 이면 SSE 단순 경로로 분기.
        다중 env (병렬 합성) 는 기존 비-스트리밍 흐름 유지."""
        _early_data = request.json or {}
        _dbg_stream = _early_data.get("stream")
        _dbg_env = _early_data.get("env")
        _dbg_skills = _early_data.get("skills")
        print(f"[api_chat] stream={_dbg_stream!r}, env={_dbg_env!r}, skills={_dbg_skills!r}")
        if _dbg_stream is True:
            _se_raw = _dbg_env if _dbg_env is not None else "auto"
            if isinstance(_se_raw, list):
                _se_list = [e for e in _se_raw if e]
            elif isinstance(_se_raw, str):
                _se_list = [_se_raw] if _se_raw else []
            else:
                _se_list = []
            _is_multi = len(_se_list) >= 2
            print(f"[api_chat] SSE 분기 판단: _se_list={_se_list}, _is_multi={_is_multi}")
            if not _is_multi:
                print("[api_chat] → _stream_chat_sse() 호출")
                return _stream_chat_sse(_early_data)
            print("[api_chat] → 다중 env, 기존 비-스트리밍 흐름으로 처리")
        chat_stop_flag["stop"] = False  # 새 요청 시작 시 플래그 초기화
        data = request.json
        # ★ 사용자별 첨부 CSV 슬롯 — 이 함수 내 모든 uploaded_csv_data 참조를 자기 슬롯으로 섀도잉
        #   (과거: 서버 전역 1칸 → 다중 사용자가 서로 첨부를 덮어쓰던 버그)
        from demos_v1.utils import csv_slot as _csv_slot
        uploaded_csv_data = _csv_slot(data.get("user_id"))
        # 환경 선택: 배열 또는 문자열 → 배열로 통일
        raw_env = data.get("env", "auto")
        if isinstance(raw_env, list):
            user_envs = [e for e in raw_env if e]
        elif isinstance(raw_env, str):
            user_envs = [raw_env] if raw_env else ["auto"]
        else:
            user_envs = ["auto"]
        if not user_envs:
            user_envs = ["auto"]

        auto_routed = False
        route_reason = ""
        multi_model_parallel = False  # 사용자가 모델 2개+ 선택한 수동 병렬
        selected_model_paths = []     # 수동 병렬 시 선택된 모델 경로들

        # 혼용 금지 체크: GGUF + API 동시 선택
        has_gguf = any(e.startswith("gguf-") for e in user_envs if e != "auto")
        has_api = any(not e.startswith("gguf-") for e in user_envs if e != "auto")
        if has_gguf and has_api:
            return jsonify({"error": "GGUF와 API를 동시에 선택할 수 없습니다. 같은 타입만 선택해주세요."}), 400

        # AUTO 모드
        if user_envs == ["auto"]:
            last_query = ""
            msgs = data.get("messages", [])
            if msgs:
                last_msg = msgs[-1]
                last_query = last_msg.get("content", "") if isinstance(last_msg.get("content"), str) else ""
            env_id, route_reason = classify_and_route(last_query, msgs, uploaded_files)
            auto_routed = True
        elif len(user_envs) >= 2:
            # 수동 다중 선택 → 첫 번째를 기본 env로, 병렬은 스킬이 결정
            env_id = user_envs[0]
            multi_model_parallel = True
            # GGUF 다중: 모델 경로 수집
            if has_gguf:
                for ue in user_envs:
                    p = ENV_CONFIG.get(ue, {}).get("_gguf_path")
                    if p:
                        selected_model_paths.append((p, ENV_CONFIG[ue].get("_size_gb", 0)))
            # API 다중: env_id 목록 유지 (user_envs 그대로 사용)
        elif user_envs[0] in ENV_CONFIG:
            env_id = user_envs[0]
        else:
            env_id = user_envs[0]

        if env_id and env_id in ENV_CONFIG:
            api_url = ENV_CONFIG[env_id]["url"]
            model = ENV_CONFIG[env_id]["model"]
        else:
            # env_id 가 ENV_CONFIG 에 없는 경우(낡은 router.py 등) 첫 번째 API env 로 폴백
            api_env_ids = [k for k in ENV_CONFIG.keys() if not str(k).startswith("gguf-")]
            if api_env_ids:
                fb = api_env_ids[0]
                print(f"[AUTO fallback] env_id={env_id!r} 매칭 실패 → {fb} 사용")
                env_id = fb
                api_url = ENV_CONFIG[fb]["url"]
                model = ENV_CONFIG[fb]["model"]
                auto_routed = True
                route_reason = (route_reason + " | " if route_reason else "") + f"env_id 폴백 → {fb}"
            else:
                api_url = data.get("api_url", "")
                model = data.get("model", "")
        # 모델별 전용 토큰(예: Spark) 우선 → 없으면 글로벌 → 요청값
        api_key = (ENV_CONFIG.get(env_id, {}).get("token") if env_id else "") or API_TOKEN or data.get("api_key", "")
        messages = data.get("messages", [])
        # ── 히스토리 자동 트림: 최근 6턴(12 메시지)만 유지 ──
        # 응답 잘림·느림 방지: 입력 토큰 폭증 차단
        # system 메시지는 보존, 그 외 user/assistant 만 최근 12개로 컷
        MAX_HISTORY_MESSAGES = 12
        _orig_msg_count = len(messages)
        if _orig_msg_count > MAX_HISTORY_MESSAGES:
            _sys_msgs = [m for m in messages if m.get("role") == "system"]
            _non_sys = [m for m in messages if m.get("role") != "system"]
            _recent = _non_sys[-MAX_HISTORY_MESSAGES:]
            messages = _sys_msgs + _recent
            print(f"  [HISTORY TRIM] {_orig_msg_count} → {len(messages)} 메시지 (최근 {MAX_HISTORY_MESSAGES} 턴 유지)")
        skill_ids = data.get("skills", [])
        effort = data.get("effort", 2)
        output_format = data.get("format", "code")
        writing_style = data.get("writing_style", "")
        custom_system_prompt = data.get("system_prompt", "")
        max_tokens = data.get("max_tokens", TOKEN_SETTINGS["agent_max_tokens"])
        user_n_ctx = data.get("n_ctx", 0)  # 프론트엔드에서 요청한 n_ctx (0이면 기본값 사용)
        think_mode = data.get("think_mode", False)
        requested_output_format = output_format
        is_gguf = env_id.startswith("gguf-") if env_id else False

        # ── drawio / pptx 서버측 안전장치 — 비활성화 ──
        # 사용자 요청: 수동 스킬 선택 시 응답이 작아지므로 max_tokens 강제 부스트 불필요.
        # 필요 시 UI 에서 max_tokens 명시적으로 넘기거나 모델 직접 선택.
        _srv_is_drawio = False
        _srv_is_pptx = False
        _api_timeout = 120

        # ── 토큰 자동 결정 (API/GGUF 모두) ──
        if is_gguf:
            # GGUF: 응답 캡은 위에서 받은 값(또는 TOKEN_SETTINGS) 그대로 — gguf_reply_cap/safe_max에서 추가로 캡됨
            # n_ctx는 32768 기본 (RTX 3090 24GB에서 충분). 프론트가 넘기면 그 값 사용.
            user_n_ctx = user_n_ctx if user_n_ctx > 0 else 32768
        else:
            # API: 모델 크기별 자동 설정 (속도: 상한이라 보통 답엔 영향 없고 긴 답 tail만 단축)
            _reg_key = ENV_TO_REGISTRY.get(env_id)
            _cost_tier = MODEL_REGISTRY.get(_reg_key, {}).get("cost_tier", "medium") if _reg_key else "medium"
            if _cost_tier == "high":      # 대형 (Qwen3.6-35B-A3B / VL-72B)
                max_tokens = 8192
            elif _cost_tier == "medium":  # 중형 (gemma-4-31B)
                max_tokens = 6144
            else:                         # 경량 (VL-30B / gpt-oss-20b)
                max_tokens = 4096

        # 출력형식/스타일 자동 분류 (format=auto 또는 writing_style=auto일 때)
        auto_format = False
        auto_style = False
        auto_fmt_reason = ""
        if output_format == "auto" or writing_style == "auto":
            last_q = ""
            if messages:
                last_m = messages[-1]
                last_q = last_m.get("content", "") if isinstance(last_m.get("content"), str) else ""
            auto_fmt, auto_sty, auto_fmt_reason = classify_format_and_style(
                last_q, messages, uploaded_files, skill_ids
            )
            if output_format == "auto":
                output_format = auto_fmt
                auto_format = True
            if writing_style == "auto":
                writing_style = auto_sty
                auto_style = True

        # GGUF 전용: PPT/Draw.io 생성 요청은 code 형식으로 강제 (API 경로 영향 없음)
        last_user_query = ""
        if messages:
            last_m = messages[-1]
            if isinstance(last_m.get("content"), str):
                last_user_query = last_m.get("content", "")
        ql = last_user_query.lower()
        gguf_artifact_request = is_gguf and any(kw in ql for kw in [
            "ppt", "피피티", "파워포인트", "프레젠테이션", "슬라이드", "deck",
            "draw.io", "drawio", "드로우", "다이어그램", "mxfile", "mxgraphmodel",
        ])
        if gguf_artifact_request and requested_output_format == "auto" and output_format in ("report", "analysis", "step-by-step"):
            output_format = "code"
            auto_format = True
            auto_fmt_reason = (auto_fmt_reason + " | " if auto_fmt_reason else "") + "GGUF artifact request -> force code"

        if (not api_url or not model) and not env_id.startswith("gguf-"):
            return jsonify({"error": "API URL과 모델 이름을 설정해주세요."}), 400

        # ── 주간보고 PPT 직접 생성 (LLM 코드 생성 우회) ──
        _ql_lower = last_user_query.lower()
        _is_weekly_ppt = ("주간보고" in _ql_lower or "주간 보고" in _ql_lower) and ("ppt" in _ql_lower or "PPT" in last_user_query or "피피티" in _ql_lower)
        if _is_weekly_ppt and ("weekly-report" in skill_ids or "knowledge-search" in skill_ids):
            _projects = []
            _weekly_docs = []
            try:
                _chat_user_id = data.get("user_id", None)
                _kb_results = search_knowledge(last_user_query, max_results=5, max_content_chars=8000, user_id=_chat_user_id)
                _weekly_docs = [r for r in _kb_results if "주간" in r.get("filename", "").lower() or "보고" in r.get("filename", "").lower()]
                if _weekly_docs:
                    # CSV 데이터 파싱 → projects 리스트 생성
                    _projects = []
                    for doc in _weekly_docs[:3]:
                        _content = doc.get("content", "")
                        _fname = doc.get("filename", "")
                        # 프로젝트명 추출
                        _proj_name = "프로젝트"
                        for line in _content.split("\n"):
                            line = line.strip()
                            if line and not line.startswith("---") and not line.startswith("tags") and not line.startswith("date") and not line.startswith("category") and not line.startswith("description") and not line.startswith("owner") and not line.startswith("title") and not line.startswith("updated") and not line.startswith("fab"):
                                if "주간" in line and "보고" in line:
                                    _proj_name = line.replace("주간 보고", "").replace("주간보고", "").strip().rstrip("_").strip()
                                    if "_" in _proj_name:
                                        _proj_name = _proj_name.split("_")[0].strip()
                                    if not _proj_name:
                                        _proj_name = "프로젝트"
                                    break
                        # CSV 행 파싱
                        _current = []
                        _next = []
                        _issues = ""
                        _lines = _content.split("\n")
                        _data_started = False
                        for line in _lines:
                            if "추진 내용" in line and "납기" in line:
                                _data_started = True
                                continue
                            if not _data_started:
                                continue
                            if "Issue" in line and "협의" in line:
                                _issues = ""
                                continue
                            parts = line.split(",")
                            if len(parts) >= 4:
                                _left_content = parts[0].strip()
                                _left_date = ""
                                _left_progress = ""
                                _right_content = ""
                                _right_date = ""
                                _right_progress = ""
                                # 왼쪽(금주 실적) 파싱
                                for p in parts[1:4]:
                                    p = p.strip()
                                    if "월" in p and "일" in p:
                                        _left_date = p
                                    elif "%" in p:
                                        _left_progress = p
                                # 오른쪽(차주 계획) 파싱
                                if len(parts) >= 5:
                                    _right_content = parts[4].strip() if len(parts) > 4 else ""
                                    for p in parts[5:]:
                                        p = p.strip()
                                        if "월" in p and "일" in p:
                                            _right_date = p
                                        elif "%" in p:
                                            _right_progress = p
                                if _left_content and _left_content.startswith("▶"):
                                    _current.append({"content": _left_content, "date": _left_date, "progress": _left_progress})
                                elif _left_content and _current:
                                    _current[-1]["content"] += "\n   " + _left_content
                                if _right_content and _right_content.startswith("▶"):
                                    _next.append({"content": _right_content, "date": _right_date, "progress": _right_progress})
                                elif _right_content and _next:
                                    _next[-1]["content"] += "\n   " + _right_content
                        if _current or _next:
                            _projects.append({"name": _proj_name, "current": _current, "next": _next, "issues": _issues})

                    if _projects:
                        try:
                            from pptx import Presentation as _Prs
                            from pptx.util import Inches as _In, Pt as _Pt, Cm as _Cm
                            from pptx.dml.color import RGBColor as _RGB
                            from pptx.enum.text import PP_ALIGN as _AL, MSO_ANCHOR as _AN
                            import datetime as _dt

                            def __sc(tb, r, c, tx, b=False, bg=None, al=_AL.LEFT, sz=10):
                                cl = tb.cell(r, c); cl.text = ""
                                p = cl.text_frame.paragraphs[0]; p.text = str(tx)
                                p.font.size = _Pt(sz); p.font.bold = b; p.font.name = "맑은 고딕"; p.alignment = al
                                cl.vertical_anchor = _AN.MIDDLE
                                if bg:
                                    cl.fill.solid(); cl.fill.fore_color.rgb = bg

                            def __tx(sl, l, t, w, h, tx, sz=10, b=False, al=_AL.LEFT):
                                bx = sl.shapes.add_textbox(l, t, w, h)
                                tf = bx.text_frame; tf.word_wrap = True
                                p = tf.paragraphs[0]; p.text = str(tx)
                                p.font.size = _Pt(sz); p.font.bold = b; p.font.name = "맑은 고딕"; p.alignment = al

                            _GY = _RGB(0xD9, 0xD9, 0xD9); _LG = _RGB(0xF2, 0xF2, 0xF2)
                            prs = _Prs(); prs.slide_width = _In(13.33); prs.slide_height = _In(7.5)
                            for idx, pj in enumerate(_projects):
                                sl = prs.slides.add_slide(prs.slide_layouts[6])
                                __tx(sl, _Cm(1), _Cm(0.5), _Cm(20), _Cm(1.2), f"{idx+1}. {pj['name']}", sz=24, b=True)
                                for i, c in enumerate([_RGB(0xFF,0,0), _RGB(0xFF,0xD7,0), _RGB(0x44,0x72,0xC4)]):
                                    br = sl.shapes.add_shape(1, _Cm(1+10.5*i), _Cm(2.0), _Cm(10.5), _Cm(0.25))
                                    br.fill.solid(); br.fill.fore_color.rgb = c; br.line.fill.background()
                                nc = len(pj.get("current",[])); nn = len(pj.get("next",[])); dr = max(nc,nn,1); tr = 2+dr+1
                                ts = sl.shapes.add_table(tr, 6, _Cm(1), _Cm(2.5), _Cm(31.5), _Cm(1.2*tr))
                                tb = ts.table
                                tb.columns[0].width=_Cm(12); tb.columns[1].width=_Cm(2.5); tb.columns[2].width=_Cm(2.5)
                                tb.columns[3].width=_Cm(12); tb.columns[4].width=_Cm(2.5); tb.columns[5].width=_Cm(2.5)
                                tb.cell(0,0).merge(tb.cell(0,2)); tb.cell(0,3).merge(tb.cell(0,5))
                                __sc(tb, 0, 0, "금주 실적", b=True, bg=_GY, al=_AL.CENTER, sz=12)
                                __sc(tb, 0, 3, "차주 계획", b=True, bg=_GY, al=_AL.CENTER, sz=12)
                                for ci, h in enumerate(["추진 내용","납기","진척율","추진 내용","납기","진척율"]):
                                    __sc(tb, 1, ci, h, b=True, bg=_LG, al=_AL.CENTER)
                                for ri in range(dr):
                                    r = ri+2
                                    if ri<nc:
                                        it=pj["current"][ri]; __sc(tb,r,0,it.get("content",""),sz=9); __sc(tb,r,1,it.get("date",""),al=_AL.CENTER,sz=9); __sc(tb,r,2,it.get("progress",""),al=_AL.CENTER,sz=9)
                                    if ri<nn:
                                        it=pj["next"][ri]; __sc(tb,r,3,it.get("content",""),sz=9); __sc(tb,r,4,it.get("date",""),al=_AL.CENTER,sz=9); __sc(tb,r,5,it.get("progress",""),al=_AL.CENTER,sz=9)
                                ir=2+dr; tb.cell(ir,0).merge(tb.cell(ir,5))
                                __sc(tb,ir,0,"Issue 및 협의사항"+(": "+pj["issues"] if pj.get("issues") else ""),b=True,bg=_GY)
                                __tx(sl,_Cm(1),_Cm(17),_Cm(20),_Cm(0.8),"● : 완료  ○ : 계획  ▶ : 진행중  ※ : Issue/특이사항",sz=9)
                                __tx(sl,_Cm(15),_Cm(17.5),_Cm(3),_Cm(0.6),str(idx+1),sz=10,al=_AL.CENTER)

                            _ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                            _udir = os.path.join(BASE_DIR, 'uploads'); os.makedirs(_udir, exist_ok=True)
                            _spath = os.path.join(_udir, f"weekly_report_{_ts}.pptx")
                            prs.save(_spath)
                            _dl_url = f"/api/weekly-report/download?id={_ts}"
                            _file_list = ", ".join(d.get("filename","") for d in _weekly_docs[:3])
                            return jsonify({
                                "content": f"주간보고 PPT를 생성했습니다.\n\n📄 참조 문서: {_file_list}\n📊 프로젝트: {len(_projects)}개\n\n📥 [PPT 다운로드]({_dl_url})",
                                "loaded_skills": loaded,
                                "system_prompt_length": 0,
                                "model_used": "weekly-report-direct",
                                "weekly_report_download": _dl_url,
                            })
                        except ImportError:
                            pass  # python-pptx 없으면 LLM 경로로 진행
                        except Exception as _ppt_err:
                            print(f"  [WEEKLY-REPORT] PPT 생성 실패: {_ppt_err}")
            except Exception as _wr_err:
                print(f"  [WEEKLY-REPORT] error: {_wr_err}")

        # ── knowledge-search 스킬: 도메인 지식 검색 후 LLM에게 전달 ──
        # 수동 선택 시에만 활성화 (자동 트리거 제거)
        _ql = last_user_query.lower()
        if "knowledge-search" in skill_ids and last_user_query.strip():
            try:
                _chat_user_id = data.get("user_id", None)
                print(f"  [KNOWLEDGE] user_id={_chat_user_id}, query={last_user_query[:50]}")
                # ★ BM25/점수(키워드 매칭) 사용 — RAG(의미검색)로 엉뚱하게 나오던 문제 원복
                # max_results=10: 검색 결과 누락 방지 (content 총량은 line 471 의 12000자 캡으로 보호)
                kb_results = search_knowledge(last_user_query, max_results=10, max_content_chars=4000, user_id=_chat_user_id)
                if kb_results:
                    # 검색된 문서 내용을 시스템 프롬프트에 주입하여 LLM이 답변하도록
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
                    # 검색 vs 내용 요청 구분
                    _is_search_only = any(kw in _ql for kw in ["검색해", "검색 해", "찾아봐", "찾아 봐", "뭐있", "뭐 있", "목록", "리스트", "파일명", "문서 목록"])
                    _is_content_request = any(kw in _ql for kw in ["내용", "관련", "알려", "설명", "분석", "요약", "만들어"])
                    if _is_search_only and not _is_content_request:
                        kb_context += (
                            "사용자가 '검색'을 요청했습니다. 파일명 목록과 관련도 점수만 간단히 보여주세요.\n"
                            "문서 내용을 분석하거나 요약하지 마세요. 파일명 리스트만 출력하세요.\n"
                            "형식 예시:\n"
                            "1. 📄 파일명.md (관련도: 75)\n"
                            "2. 📄 파일명.md (관련도: 73)\n"
                        )
                    else:
                        kb_context += (
                            "위 문서를 기반으로 사용자 질문에 답변하세요. 문서에 없는 내용을 지어내지 마세요. 어떤 문서에서 정보를 찾았는지 출처를 명시하세요.\n"
                            "중요: 프로토콜 메시지, raw 데이터, hex/binary는 절대 그대로 복사하지 마세요. 반드시 표(table) 또는 필드별 설명으로 변환하세요.\n"
                        )

                    # messages에 검색 결과를 system 메시지로 추가
                    kb_system_msg = {"role": "system", "content": kb_context}
                    messages = [kb_system_msg] + messages

                    # 검색된 파일 목록을 응답에 포함하기 위해 저장
                    _kb_files = [r['filename'] for r in kb_results]
            except Exception as e:
                print(f"[Knowledge Search] 검색 오류: {e}")

        # ── logpresso-search 스킬: 테이블 목록 요청 감지 ──
        _lpq_table_list_kw = ["테이블 목록", "어떤 테이블", "테이블 뭐", "테이블 리스트", "테이블 종류",
                              "테이블 있", "테이블 알려", "테이블 보여", "테이블 전부", "테이블 전체"]
        if "logpresso-search" in skill_ids and any(kw in last_user_query for kw in _lpq_table_list_kw):
            # 사용자 질문에서 카테고리 키워드 감지하여 필터링
            _q_lower = last_user_query.lower()
            _matched_groups = []
            for grp in LOGPRESSO_TABLE_GROUPS:
                # 그룹 라벨 키워드 매칭 (예: "ATLAS", "OHT", "TS", "시스템" 등)
                _grp_keywords = [grp["id"], grp["label"].split(" ")[-1].lower()]
                _grp_keywords += [p.rstrip("_").lower() for p in grp["prefix"]]
                if any(kw in _q_lower for kw in _grp_keywords):
                    _matched_groups.append(grp["id"])

            # FAB 필터 감지 (M14, M16, M14A, M16B 등)
            _matched_fabs = []
            for fab in LOGPRESSO_FAB_FILTERS:
                if fab["id"] in _q_lower or fab["label"].lower() in _q_lower:
                    _matched_fabs.append(fab["id"])

            _filtered = _filter_tables_by_groups(_matched_groups, _matched_fabs)
            _filter_label = ""
            _labels = []
            if _matched_groups:
                _labels += [g["label"] for g in LOGPRESSO_TABLE_GROUPS if g["id"] in _matched_groups]
            if _matched_fabs:
                _labels += [f["label"] for f in LOGPRESSO_FAB_FILTERS if f["id"] in _matched_fabs]
            if _labels:
                _filter_label = f" [{', '.join(_labels)}]"

            _content = f"**로그프레소 테이블 목록{_filter_label}**\n\n"
            if _matched_groups or _matched_fabs:
                _content += f"🔍 필터 적용: 카테고리={_matched_groups or '없음'}, FAB={_matched_fabs or '없음'} → **{len(_filtered)}개** 매칭\n\n"
            _content += f"총 **{len(LOGPRESSO_TABLES)}개** 중 **{len(_filtered)}개** 표시\n\n"

            # 카테고리별로 그룹화하여 표시
            _grouped = {}
            for tname in sorted(_filtered.keys()):
                gid = _get_table_group(tname)
                if gid not in _grouped:
                    _grouped[gid] = []
                _grouped[gid].append(tname)

            # 그룹 순서 유지
            _grp_order = [g["id"] for g in LOGPRESSO_TABLE_GROUPS] + ["etc"]
            _grp_labels_map = {g["id"]: g["label"] for g in LOGPRESSO_TABLE_GROUPS}
            _grp_labels_map["etc"] = "📁 기타"

            _idx = 0
            for gid in _grp_order:
                if gid not in _grouped:
                    continue
                _content += f"\n### {_grp_labels_map[gid]} ({len(_grouped[gid])}개)\n\n"
                _content += "| # | 테이블명 | 설명 | 필드(컬럼) |\n"
                _content += "|---|----------|------|----------|\n"
                for tname in _grouped[gid]:
                    _idx += 1
                    tinfo = LOGPRESSO_TABLES[tname]
                    cols = tinfo["columns"]
                    if not cols:
                        cols = _fetch_table_fields(tname, timeout=3)
                        if cols:
                            tinfo["columns"] = cols
                    cols_str = ", ".join(cols[:8]) if cols else "(서버 미접속)"
                    if len(cols) > 8:
                        cols_str += f" 외 {len(cols)-8}개"
                    _content += f"| {_idx} | `{tname}` | {tinfo['desc']} | {cols_str} |\n"

            _content += f"\n> **카테고리 필터**: `ATLAS 테이블 목록`, `OHT 테이블 목록`, `TS 테이블 보여줘`, `시스템 테이블 목록`\n"
            _content += f"> **FAB 필터**: `M14 테이블 목록`, `M14A 테이블 보여줘`, `M16B 테이블 목록`\n"
            _content += f"> **조합 가능**: `M14A TS 테이블 목록` (Transfer + M14A만)\n"
            _content += f"> 카테고리: {' / '.join(g['label'] for g in LOGPRESSO_TABLE_GROUPS)}"
            return jsonify({
                "content": _content,
                "model_used": "Logpresso Tables",
                "loaded_skills": ["logpresso-search"],
                "system_prompt_length": 0,
            })

        # ── logpresso-search 스킬: 실제 서버 조회 실행 ──
        # "쿼리 만들어줘" 등 쿼리 생성 요청이면 서버 실행 안 하고 LLM에게 넘김
        _lpq_query_gen_kw = ["쿼리 만들", "쿼리 짜", "쿼리 작성", "쿼리 생성", "LPQL 만들", "LPQL 짜", "LPQL 작성"]
        _lpq_is_query_gen = any(kw in last_user_query for kw in _lpq_query_gen_kw)
        if "logpresso-search" in skill_ids and last_user_query.strip() and not _lpq_is_query_gen:
            try:
                import pandas as pd
                from datetime import datetime as _dt

                _lpq_user_q = last_user_query.strip()

                # LLM으로 LPQL 생성
                llm_response = _llm_generate_lpql(_lpq_user_q, [
                    {"role": m.get("role", "user"), "content": m.get("content", "")}
                    for m in messages[-6:] if isinstance(m.get("content"), str)
                ])
                lpql = extract_lpql_from_response(llm_response) if llm_response else None

                if not lpql:
                    # LLM 실패 → 사용자 질문에서 테이블명 추출하여 기본 쿼리 생성
                    _found_table = None
                    _q_lower = _lpq_user_q.lower()
                    # LOGPRESSO_TABLES에서 매칭 시도
                    for tname in LOGPRESSO_TABLES:
                        if tname.lower() in _q_lower:
                            _found_table = tname
                            break
                    # 못 찾으면 질문에서 테이블명 후보 추출 (영문+숫자+언더스코어 패턴)
                    if not _found_table:
                        _table_candidates = re.findall(r'[a-zA-Z][a-zA-Z0-9_]{3,}', _lpq_user_q)
                        if _table_candidates:
                            _found_table = _table_candidates[0]

                    if _found_table:
                        _today_str = _dt.now().strftime("%Y%m%d")
                        lpql = f"table from={_today_str}000000 to={_today_str}235959 {_found_table} | limit 5"
                        # lpql이 생겼으니 아래 실행 로직으로 계속 진행
                    else:
                        _content = "**LPQL 쿼리 생성 실패**\n\n"
                        _content += "테이블명을 인식할 수 없습니다. 테이블명을 정확히 입력해주세요.\n"
                        if llm_response:
                            _content += f"\nLLM 응답:\n```\n{llm_response[:500]}\n```"
                        return jsonify({
                            "content": _content,
                            "model_used": "Logpresso Search",
                            "loaded_skills": ["logpresso-search"],
                            "auto_routed": auto_routed,
                            "route_reason": "logpresso-search",
                            "system_prompt_length": 0,
                        })

                # 시간 범위 없으면 오늘 하루 기본 적용
                if not re.search(r'(duration|from|to)\s*=', lpql, re.IGNORECASE):
                    _today = _dt.now().strftime("%Y%m%d")
                    lpql = re.sub(r'^(table|fulltext)\s+', rf'\1 from={_today}000000 to={_today}235959 ', lpql, flags=re.IGNORECASE)

                # limit 강제 5건
                lpql_lower = lpql.lower()
                if "| limit" in lpql_lower or "| head" in lpql_lower:
                    lpql = re.sub(r'\|\s*(limit|head)\s+\d+(\s+\d+)?', '| limit 5', lpql, flags=re.IGNORECASE)
                else:
                    lpql = lpql.rstrip() + " | limit 5"

                # 보안 검증
                sec_error = validate_lpql_readonly(lpql)
                if sec_error:
                    return jsonify({
                        "content": f"**보안 차단**: {sec_error}\n\n**생성된 쿼리:**\n```lpql\n{lpql}\n```",
                        "model_used": "Logpresso Search",
                        "loaded_skills": ["logpresso-search"],
                        "auto_routed": auto_routed,
                        "route_reason": "logpresso-search-blocked",
                        "system_prompt_length": 0,
                    })

                # 실제 로그프레소 서버에 쿼리 실행
                print(f"[Logpresso Search] 최종 실행 쿼리: {lpql}")
                df, err_detail = query_logpresso(lpql, timeout=180)

                if df is None:
                    # 실패 → 에러 상세 + 쿼리 표시
                    err_reason = err_detail.get("reason", "알 수 없는 오류") if err_detail else "알 수 없는 오류"
                    resp_preview = err_detail.get("response_preview", "") if err_detail else ""
                    _content = f"**로그프레소 조회 실패**\n\n"
                    _content += f"**에러:** {err_reason}\n\n"
                    _content += f"**실행한 쿼리:**\n```lpql\n{lpql}\n```\n"
                    if resp_preview:
                        _content += f"\n**서버 응답:**\n```\n{resp_preview[:300]}\n```\n"
                    _content += f"\n**원인 추정:** "
                    if "타임아웃" in err_reason or "연결" in err_reason:
                        _content += "로그프레소 서버(`10.40.42.27:8888`)에 연결할 수 없습니다. 서버 상태 또는 네트워크를 확인하세요."
                    elif "HTML" in err_reason:
                        _content += "서버가 에러 페이지를 반환했습니다. 테이블명이나 쿼리 문법을 확인하세요."
                    elif "빈 응답" in err_reason:
                        _content += "서버가 빈 응답을 반환했습니다. 테이블명이 올바른지 확인하세요."
                    else:
                        _content += "서버 연결 또는 쿼리 오류입니다."

                    return jsonify({
                        "content": _content,
                        "model_used": "Logpresso Search",
                        "loaded_skills": ["logpresso-search"],
                        "auto_routed": auto_routed,
                        "route_reason": "logpresso-search-failed",
                        "system_prompt_length": 0,
                    })

                # 성공 → 결과 + 쿼리 + 테이블 필드 정보 표시
                total = len(df)
                cols = list(df.columns)

                # 쿼리에서 테이블명 추출 → 해당 테이블의 전체 필드 정보 표시
                _queried_table = None
                _table_match = re.search(r'(?:table|fulltext)\s+(?:\S+=\S+\s+)*(\S+)', lpql, re.IGNORECASE)
                if _table_match:
                    _queried_table = _table_match.group(1).strip()

                _content = f"**로그프레소 조회 완료** (총 {total}건, 미리보기 {min(5, total)}건)\n\n"
                _content += f"**실행한 쿼리:**\n```lpql\n{lpql}\n```\n\n"

                # 테이블 필드 정보
                if _queried_table:
                    _all_fields = cols  # 실제 조회 결과의 컬럼이 가장 정확
                    # LOGPRESSO_TABLES 캐시 업데이트
                    if _queried_table in LOGPRESSO_TABLES:
                        if not LOGPRESSO_TABLES[_queried_table]["columns"] and _all_fields:
                            LOGPRESSO_TABLES[_queried_table]["columns"] = _all_fields
                        elif LOGPRESSO_TABLES[_queried_table]["columns"]:
                            _all_fields = LOGPRESSO_TABLES[_queried_table]["columns"]
                    _content += f"**테이블 `{_queried_table}` 필드** ({len(_all_fields)}개): `{'`, `'.join(_all_fields)}`\n\n"
                else:
                    _content += f"**결과 컬럼:** {', '.join(cols)}\n\n"

                # 테이블 형태로 결과 표시
                if total > 0:
                    display_cols = cols[:10]  # 컬럼 10개까지만 표시
                    _content += "| " + " | ".join(display_cols) + " |\n"
                    _content += "|" + "|".join(["---"] * len(display_cols)) + "|\n"
                    for _, row in df.head(5).iterrows():
                        _content += "| " + " | ".join(str(row.get(c, ""))[:50] for c in display_cols) + " |\n"
                    if len(cols) > 10:
                        _content += f"\n_(컬럼 {len(cols)}개 중 10개만 표시)_\n"
                else:
                    _content += "결과 0건 (데이터가 없습니다. 기간을 늘려보세요.)\n"

                return jsonify({
                    "content": _content,
                    "model_used": "Logpresso Search",
                    "loaded_skills": ["logpresso-search"],
                    "auto_routed": auto_routed,
                    "route_reason": "logpresso-search-ok",
                    "system_prompt_length": 0,
                })

            except Exception as e:
                import traceback
                traceback.print_exc()
                # 예외 발생 → 에러 표시 후 일반 LLM으로 폴백하지 않음
                return jsonify({
                    "content": f"**로그프레소 조회 중 오류 발생**\n\n`{type(e).__name__}: {e}`",
                    "model_used": "Logpresso Search",
                    "loaded_skills": ["logpresso-search"],
                    "auto_routed": auto_routed,
                    "route_reason": "logpresso-search-exception",
                    "system_prompt_length": 0,
                })

        # 시스템 프롬프트 구성
        # 출력 형식에 따라 기본 규칙 분기
        non_code_formats = ("report", "analysis", "step-by-step")
        if output_format in non_code_formats:
            code_rules = """- 사용자가 코드를 명시적으로 요청한 경우에만 코드를 포함하세요
    - 기본적으로 자연어(글)로 설명, 분석, 보고하세요"""
        else:
            code_rules = """- 코드는 즉시 실행 가능하게 (import 포함, 필요 패키지 명시)
    - 에러 처리(try/except)를 포함하세요
    - 코드 주석은 반드시 ## 형식으로 작성하세요. 단일 #이 아니라 ##을 사용하고, 구간 설명용으로만 간결하게 작성하세요. 줄마다 주석을 달지 마세요. 예: ## 데이터 전처리, ## API 호출"""

        # 사고 모드 on/off
        if think_mode:
            think_rule = (
                "- 답변 전에 반드시 <think>...</think> 태그 안에서 충분히 사고한 후 답변하세요. "
                "사고 내용도 반드시 한국어로 작성하세요. "
                "사고는 <think> 태그 안에만 작성하고, 태그 밖 본문은 사고 흔적 없이 최종 답변만 출력하세요."
            )
        else:
            think_rule = (
                "- 사고/추론 과정을 본문에 절대 출력하지 마세요. <think> 태그도 사용 금지. "
                "사고 절차를 영문/한글 어떤 형태로도 노출 금지: "
                "'Here's a thinking process', 'Let me think/analyze', 'Self-Correction', "
                "'Structure mapping', '[Output Generation]', '[Proceeds]', 'I will/must/should', "
                "'Step 1/2/3', '먼저 분석', '먼저 생각해보면' 같은 절차 안내문구 일체 금지. "
                "최종 답변(보고서·코드·설명)만 한국어로 즉시 출력하세요."
            )

        default_prompt = f"""당신은 Demos V1.0 - 과학 연구와 소프트웨어 개발을 돕는 전문 AI 어시스턴트입니다.
    370개+ 전문 스킬(과학/개발/AI/인프라/비즈니스)을 활용할 수 있습니다.

    [기본 규칙]
    - 반드시 한국어(한글)로 답변하세요. 코드 주석도 한글로 작성하세요.
    {think_rule}
    {code_rules}

    [응답 완결성]
    - 코드 블록(```)은 반드시 열고 닫아야 합니다. 코드 중간에서 끊지 마세요.
    - 문장은 반드시 완성된 형태로 끝내세요. 중간에 끊기지 않도록 하세요.
    - 응답이 길어질 것 같으면, 핵심부터 먼저 완성하고 부가 설명을 뒤에 붙이세요.
    - 토큰 한도에 가까워지면 현재 문단/코드 블록을 마무리한 후 "계속 이어서 작성해줘라고 입력하세요"로 안내하세요.

    [스킬 활용]
    - 아래에 로드된 SKILL 내용은 당신의 전문 지식입니다
    - 해당 스킬의 방법론, 코드 패턴, API, 도구를 적극 활용하세요
    - 여러 스킬이 로드된 경우 관련된 것끼리 조합하여 최적의 답변을 생성하세요

    """

        if custom_system_prompt:
            system_prompt = custom_system_prompt + "\n\n" + default_prompt
        else:
            system_prompt = default_prompt

        # ── "이어서 작성" 감지: 이전 응답 마지막 부분을 겹쳐서 이어쓰기 ──
        _continue_kw = ["이어서", "계속 작성", "계속 이어서", "이어 작성", "이어서 써", "이어서 보기", "계속해", "계속 써"]
        if last_user_query and any(kw in last_user_query for kw in _continue_kw):
            # 히스토리에서 마지막 assistant 응답 찾기
            _last_assistant = ""
            for m in reversed(messages[:-1]):  # 현재 "이어서" 메시지 제외
                if m.get("role") == "assistant":
                    _last_assistant = m.get("content", "") or ""
                    break
            if _last_assistant and len(_last_assistant) > 100:
                # 마지막 300자를 겹쳐서 이어쓰기 지시
                _overlap = _last_assistant[-300:]
                system_prompt += (
                    f"\n\n[이어서 작성 모드]\n"
                    f"이전 응답이 중간에 잘렸습니다. 아래는 이전 응답의 마지막 부분입니다:\n"
                    f"---\n{_overlap}\n---\n"
                    f"위 내용의 마지막 2~3문장을 포함하여 자연스럽게 이어서 작성하세요.\n"
                    f"절대 처음부터 다시 쓰지 마세요. 겹치는 부분부터 이어서 쓰세요."
                )

        # ===== 토큰 예산 관리 =====
        # API 대형 모델 → 제한 거의 없음 (128K+ 컨텍스트)
        # GGUF 로컬 → 32K 컨텍스트이므로 스마트하게 제한
        is_gguf = env_id.startswith("gguf-") if env_id else False
        history_chars = sum(len(m.get("content", "")) for m in messages)

        if is_gguf:
            gguf_ctx = 32768
            # GGUF: 토큰 ≈ 글자수 * 0.7 (한글 기준), 시스템+히스토리+응답 여유
            available_chars = int(gguf_ctx / 0.7) - len(system_prompt) - history_chars - 3000
            available_chars = max(available_chars, 3000)
        else:
            available_chars = 999999  # API는 사실상 무제한

        # 스킬 로드 — 사용자가 고른 skill_ids 만 개별 로드 (전수 스캔 안 함)
        loaded = []
        total_skill_chars = 0
        num_skills = max(len(skill_ids), 1)
        per_skill_limit = available_chars // num_skills if is_gguf else 999999

        for sid in skill_ids:
            content = load_skill_content(sid)
            if content:
                # GGUF만 자르기, API는 전체 로드
                if is_gguf and total_skill_chars + len(content) > available_chars:
                    remaining = available_chars - total_skill_chars
                    if remaining > 500:
                        content = content[:remaining] + f"\n... ({len(content)}자 중 {remaining}자 로드)\n"
                    else:
                        system_prompt += f"[⚠️ GGUF 컨텍스트 한도 → 스킬 생략: {', '.join(skill_ids[len(loaded):])}]\n"
                        break

                if is_gguf and len(content) > per_skill_limit:
                    content = content[:per_skill_limit] + f"\n... ({len(content)}자 중 {per_skill_limit}자 로드)\n"

                system_prompt += f"=== SKILL: {sid} ===\n{content}\n\n"
                total_skill_chars += len(content)
                loaded.append(sid)

                # scripts/references 목록 (짧으니 항상 포함) — 해당 스킬 폴더만 단발 스캔
                _sd = os.path.join(SKILLS_DIR, sid)
                scripts = []
                _scripts_dir = os.path.join(_sd, "scripts")
                if os.path.isdir(_scripts_dir):
                    for _root, _dirs, _files in os.walk(_scripts_dir):
                        for _fn in _files:
                            if _fn.endswith(".py"):
                                scripts.append({"name": _fn})
                if scripts:
                    system_prompt += f"[{sid} 스크립트: {', '.join(s['name'] for s in scripts)}]\n"
                refs = []
                _refs_dir = os.path.join(_sd, "references")
                if os.path.isdir(_refs_dir):
                    for _fn in os.listdir(_refs_dir):
                        if os.path.isfile(os.path.join(_refs_dir, _fn)):
                            refs.append({"name": _fn})
                if refs:
                    system_prompt += f"[{sid} 참고: {', '.join(r['name'] for r in refs)}]\n"
                system_prompt += "\n"

        if loaded:
            system_prompt += f"[로드된 스킬: {', '.join(loaded)}]\n\n"
            # 하네스 피드백 힌트: 이전 피드백 기반 주의사항 자동 삽입
            if HARNESS_AVAILABLE:
                try:
                    from harness_bridge import _feedback_store
                    if _feedback_store:
                        _fb_hint = _feedback_store.build_prompt_hint(loaded)
                        if _fb_hint:
                            system_prompt += _fb_hint
                except Exception:
                    pass
            if "agent-llm-architect" in loaded:
                system_prompt += (
                    "=== LLM 시스템 설계 규칙 ===\n"
                    "사용자가 LLM/에이전트/RAG 설계를 요청하면 다음 순서로 답변하세요:\n"
                    "1. 먼저 요구사항과 제약을 분석하세요(트래픽, 지연시간, 예산, 보안, 데이터 민감도).\n"
                    "2. 아키텍처 옵션 2~3개를 비교하고 트레이드오프를 명확히 제시하세요.\n"
                    "3. 최종 추천안을 제시하고, 선택 근거를 성능/비용/운영성 관점으로 설명하세요.\n"
                    "4. 구현 단계를 POC→파일럿→프로덕션으로 나눠 체크리스트 형태로 제시하세요.\n"
                    "5. 필수 운영 지표(SLO, 토큰비용, 오류율, 환각률, 안전성)를 포함하세요.\n\n"
                )
            # pptx 스킬이 로드됐으면 python-pptx 코드 생성 지시 추가
            ppt_requested = 'ql' in locals() and any(kw in ql for kw in ["ppt", "피피티", "파워포인트", "프레젠테이션", "슬라이드", "deck"])
            if "pptx" in loaded or ppt_requested:
                system_prompt += (
                    "=== PPT 생성 규칙 ===\n"
                    "사용자가 PPT/프레젠테이션/슬라이드 생성을 요청하면:\n"
                    "1. 반드시 ```python 코드블록 안에 python-pptx 코드를 작성하세요.\n"
                    "2. 코드는 `from pptx import Presentation`으로 시작하세요.\n"
                    "3. 대화 내용, 분석 결과, 업로드 파일 내용을 PPT에 반영하세요.\n"
                    "4. 각 슬라이드에 제목, 내용, 표, 차트 등을 포함하세요.\n"
                    "5. 마지막에 `prs.save('presentation.pptx')` 호출을 포함하세요.\n"
                    "6. 디자인 가이드: 볼드 색상 팔레트, 다양한 레이아웃, 데이터 시각화 활용.\n"
                    "7. 중요: 좌표/크기 값은 반드시 정수(int)로 전달하세요. "
                    "Inches(), Cm(), Pt(), Emu() 결과를 직접 사용하면 됩니다. "
                    "나눗셈(/) 등으로 float이 생기면 반드시 int()로 감싸세요.\n"
                    "8. 중요: tf.vertical_anchor에는 PP_ALIGN이 아니라 MSO_ANCHOR를 사용하세요. "
                    "예: `from pptx.enum.text import MSO_ANCHOR` 후 `tf.vertical_anchor = MSO_ANCHOR.MIDDLE`. "
                    "PP_ALIGN.CENTER를 vertical_anchor에 쓰면 에러가 납니다.\n"
                    "9. 중요: 문자열 안에 한국어 특수 따옴표(\u201c \u201d \u2018 \u2019 \u300c \u300d)를 절대 사용하지 마세요. "
                    "반드시 일반 ASCII 따옴표(\" ' )만 사용하세요. "
                    "인용문이 필요하면 작은따옴표(')로 감싸세요. 예: p.text = '기술은 훌륭하지만, 보안에 구멍이 많다.'\n"
                    "10. 코드 주석은 반드시 `## 설명내용` 형식으로 작성하세요. "
                    "단일 #이 아니라 ##을 사용하고, 구간 설명용으로만 간결하게 작성하세요. "
                    "예: `## 슬라이드 1: 표지`, `## 차트 데이터 설정`. 줄마다 주석을 달지 마세요.\n"
                    "11. 중요: 괄호 매칭에 극도로 주의하세요! "
                    "[인덱스]는 반드시 ]로 닫고, (함수호출)은 반드시 )로 닫으세요. "
                    "예: prs.slide_layouts[6] (O) / prs.slide_layouts[6) (X — SyntaxError). "
                    "코드 작성 후 모든 괄호 쌍([]/()/{})이 올바르게 매칭되는지 반드시 검증하세요.\n"
                    "12. 중요: 색상은 반드시 `from pptx.dml.color import RGBColor`를 사용하세요! "
                    "RgbColor(X), rgbColor(X) 등은 존재하지 않습니다. 반드시 RGBColor (대문자 RGB)만 사용하세요. "
                    "예: `from pptx.dml.color import RGBColor` → `fill.fore_color.rgb = RGBColor(0x1A, 0x73, 0xE8)` "
                    "또는 `run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)`. "
                    "절대 `from pptx.dml.color import RgbColor`를 쓰지 마세요 — ImportError가 발생합니다.\n"
                    "13. 중요: 대화에서 데이터/수치/통계가 있으면 반드시 python-pptx 차트로 시각화하세요! "
                    "텍스트만 나열하지 말고, 차트 슬라이드를 추가하세요.\n"
                    "14. 중요: 차트/도형/표를 추가할 때 반드시 `slide.shapes.add_chart(...)`, `slide.shapes.add_table(...)` 등 "
                    "slide 객체의 shapes를 사용하세요! placeholder나 content 변수에는 .shapes 속성이 없습니다. "
                    "예: `content = slide.placeholders[1]`한 뒤 `content.shapes.add_chart(...)` (X — AttributeError). "
                    "반드시 `slide.shapes.add_chart(...)` (O)로 작성하세요.\n"
                    "15. 중요: 표(table) 생성 시 add_table(rows, cols, ...)의 rows 수를 데이터 행 수 + 헤더 행에 정확히 맞추세요! "
                    "데이터가 N개이면 add_table(N+1, cols, ...)로 헤더 포함 N+1행을 만드세요. "
                    "행 수가 부족하면 table.cell(i, j)에서 IndexError가 발생합니다. "
                    "예: 데이터 5개 → add_table(6, 3, ...) (헤더 1행 + 데이터 5행 = 6행).\n"
                    "16. 중요: 문자열 리터럴은 반드시 같은 줄에서 열고 닫으세요! "
                    "줄바꿈이 필요하면 삼중따옴표(triple quotes)를 사용하세요. "
                    "예: p.text = '긴 텍스트' (O) / p.text = '긴 텍스트 (X — SyntaxError: unterminated string literal).\n"
                    "17. 중요 (필수): 파이썬 문법에 맞게 들여쓰기(Indentation)와 줄바꿈을 철저히 지키고 한 줄에 여러 명령어(`slide = ... title = ...`)를 이어서 쓰지 마세요. SyntaxError가 발생합니다.\n"
                    "18. 중요 (필수): 반드시 단 1개의 완성된 파이썬 스크립트만 ```python 과 ``` 사이에 출력하세요. 코드 앞뒤로 인사말, <think> 태그, 요약 설명 등 부가적인 텍스트를 절대 쓰지 마세요. 오직 파이썬 코드만 출력해야 작동합니다.\n"
                    "python-pptx 차트 코드 예시:\n"
                    "```python\n"
                    "from pptx.chart.data import CategoryChartData\n"
                    "from pptx.enum.chart import XL_CHART_TYPE\n"
                    "from pptx.util import Inches\n"
                    "\n"
                    "## 차트 데이터 설정\n"
                    "chart_data = CategoryChartData()\n"
                    "chart_data.categories = ['1분기', '2분기', '3분기', '4분기']\n"
                    "chart_data.add_series('매출', (120, 190, 300, 250))\n"
                    "chart_data.add_series('비용', (80, 100, 150, 130))\n"
                    "\n"
                    "## 차트 슬라이드 추가\n"
                    "slide = prs.slides.add_slide(prs.slide_layouts[6])\n"
                    "chart_frame = slide.shapes.add_chart(\n"
                    "    XL_CHART_TYPE.COLUMN_CLUSTERED,\n"
                    "    Inches(1), Inches(1.5), Inches(8), Inches(5),\n"
                    "    chart_data\n"
                    ")\n"
                    "chart = chart_frame.chart\n"
                    "chart.has_legend = True\n"
                    "chart.legend.include_in_layout = False\n"
                    "```\n"
                    "지원 차트: XL_CHART_TYPE.COLUMN_CLUSTERED(세로막대), LINE(꺾은선), PIE(원형), "
                    "BAR_CLUSTERED(가로막대), DOUGHNUT(도넛), AREA(영역), RADAR(방사형), "
                    "XY_SCATTER(산점도), LINE_MARKERS(꺾은선+마커)\n"
                    "여러 시리즈 비교: chart_data.add_series()를 여러 번 호출\n"
                    "원형 차트: CategoryChartData에 시리즈 1개만 추가, XL_CHART_TYPE.PIE 사용\n"
                    "프론트엔드가 코드를 감지하여 '📽️ PPT 생성 & 다운로드' 버튼을 자동 표시합니다.\n\n"
                )
                # PPT 디자인 스타일 주입 (참고PPT 우선, 없으면 프리셋)
                ppt_ref_design = data.get("ppt_ref_design", "")
                ppt_style = data.get("ppt_style", "")
                if ppt_ref_design:
                    system_prompt += (
                        "=== 참고 PPT 디자인 (반드시 이 스타일을 따르세요) ===\n"
                        f"{ppt_ref_design}\n\n"
                    )
                elif ppt_style:
                    system_prompt += (
                        "=== PPT 디자인 스타일 ===\n"
                        f"{ppt_style}\n"
                        "이 디자인 가이드를 반드시 따라 모든 슬라이드를 제작하세요.\n\n"
                    )

            # MD/HTML 문서 변환 감지
            md_html_requested = 'ql' in locals() and any(kw in ql for kw in [
                "html로", "html 로", "html변환", "html 변환", "html로 만들", "html 다운",
                "md로", "md 로", "md변환", "md 변환", "md 다운", "마크다운으로", "마크다운 다운",
                "문서로 저장", "문서로 만들", "보고서 다운", "보고서로 저장",
            ])
            if "md-to-html" in loaded or md_html_requested:
                system_prompt += (
                    "=== MD/HTML 문서 생성 규칙 ===\n"
                    "사용자가 MD, HTML, 문서 저장/다운로드를 요청하면:\n"
                    "1. 보고서 내용을 잘 구조화된 Markdown 형식으로 작성하세요.\n"
                    "2. 제목(#), 소제목(##), 표(|---|), 목록(- ), 코드블록(```) 등 Markdown 문법을 적극 활용하세요.\n"
                    "3. 시스템이 자동으로 MD 파일과 HTML 파일을 생성하여 다운로드 링크를 제공합니다.\n"
                    "4. 별도의 코드 작성은 불필요합니다. 보고서 내용에만 집중하세요.\n"
                    "5. 중요: 보고서 제목에 '(HTML 형식)', '(MD 형식)' 등 파일 형식을 언급하지 마세요. 순수한 보고서 제목만 작성하세요.\n\n"
                )

            drawio_requested = 'ql' in locals() and any(kw in ql for kw in ["drawio", "draw.io", "드로우", "드로잉", "drawingio", "다이어그램", "구조도", "흐름도", "아키텍처", "배치도", "dfd"])
            if "drawio-diagram" in loaded or drawio_requested:
                system_prompt += (
                    "=== Draw.io 다이어그램 생성 규칙 ===\n"
                    "사용자가 Draw.io, 다이어그램, 아키텍처, 구조도 생성을 요청하면:\n"
                    "1. 반드시 ```drawio 코드블록 안에 완전한 XML 코드를 작성하세요.\n"
                    "2. XML 구조 예시: ```drawio\n<mxfile><diagram><mxGraphModel><root><mxCell id=\"0\"/><mxCell id=\"1\" parent=\"0\"/>...</root></mxGraphModel></diagram></mxfile>\n```\n"
                    "3. 중요 (GGUF 필수): 다이어그램에 대한 부가적인 설명, 설치 가이드, 인사말, <think> 태그 등은 **절대 출력하지 마세요**.\n"
                    "4. 오직 단 1개의 완성된 XML 코드 블록만 출력해야 프론트엔드가 다이어그램 렌더러를 띄웁니다.\n"
                    "5. 노드의 위치(x, y), 크기(width, height) 속성을 적절히 지정해 겹치지 않게 하세요.\n\n"
                )
            # === 차트/그래프 생성 가이드 (항상 포함 - 스킬 불필요) ===
            system_prompt += (
                "=== 차트/그래프 생성 규칙 ===\n"
                "사용자가 그래프, 차트, 시각화를 요청하거나 데이터 분석 결과를 보여줄 때:\n"
                "1. ```chart 코드블록 안에 Chart.js JSON config를 작성하세요.\n"
                "2. 프론트엔드가 자동으로 인터랙티브 차트를 렌더링합니다.\n"
                "3. 지원 차트 유형: bar, line, pie, doughnut, radar, scatter, bubble, polarArea\n"
                "4. JSON 형식 예시:\n"
                "```chart\n"
                '{\n'
                '  "type": "bar",\n'
                '  "data": {\n'
                '    "labels": ["1월", "2월", "3월"],\n'
                '    "datasets": [{\n'
                '      "label": "매출",\n'
                '      "data": [120, 190, 300],\n'
                '      "backgroundColor": ["#6366f1", "#8b5cf6", "#a78bfa"]\n'
                '    }]\n'
                '  },\n'
                '  "options": {\n'
                '    "plugins": {"title": {"display": true, "text": "월별 매출"}}\n'
                '  }\n'
                '}\n'
                "```\n"
                "5. 여러 데이터셋 비교: datasets 배열에 여러 객체 추가\n"
                "6. 색상: backgroundColor, borderColor 사용. 배열로 항목별 색상 지정 가능\n"
                "7. 추천 색상 팔레트: #6366f1(인디고), #8b5cf6(보라), #ec4899(핑크), #f59e0b(앰버), #10b981(에메랄드), #3b82f6(블루), #ef4444(레드), #84cc16(라임)\n"
                "8. 데이터 분석/CSV 분석 결과를 보여줄 때 적극 활용하세요.\n"
                "9. 한 응답에 여러 ```chart 블록을 사용해 다양한 관점의 차트를 보여줄 수 있습니다.\n\n"
            )

            # 멀티에이전트 오케스트레이션 (스킬 2개 이상)
            if len(loaded) >= 2:
                loaded_set = {sid: load_skill_content(sid) or "" for sid in loaded}
                orch_prompt = build_orchestration_prompt(
                    messages[-1].get("content", "") if messages else "",
                    loaded, loaded_set
                )
                system_prompt += orch_prompt

        # CSV 데이터가 업로드되어 있으면 시스템 프롬프트에 포함
        include_csv = data.get("include_csv", True)
        if include_csv and uploaded_csv_data["filename"]:
            csv_info = uploaded_csv_data["summary"]
            # 데이터 미리보기 (최대 50행)
            preview_limit = min(50, len(uploaded_csv_data["rows"]))
            csv_rows_text = ",".join(uploaded_csv_data["headers"]) + "\n"
            for row in uploaded_csv_data["rows"][:preview_limit]:
                csv_rows_text += ",".join(str(c) for c in row) + "\n"
            if len(uploaded_csv_data["rows"]) > preview_limit:
                csv_rows_text += f"... (총 {len(uploaded_csv_data['rows'])}행 중 {preview_limit}행만 표시)\n"

            system_prompt += f"=== 업로드된 CSV 데이터 ===\n{csv_info}\n\n데이터 미리보기:\n{csv_rows_text}\n\n"
            system_prompt += "사용자가 이 데이터에 대해 질문하면 위 CSV 데이터를 기반으로 분석해주세요.\n\n"

        # 업로드된 파일 내용 주입
        if uploaded_files:
            system_prompt += f"=== 업로드된 파일 ({len(uploaded_files)}개) ===\n"
            for uf in uploaded_files:
                system_prompt += f"\n--- 파일: {uf['filename']} ({uf['type']}, {uf['size']}바이트) ---\n"
                content = uf.get("content_full", "")
                if is_gguf:
                    # GGUF: 파일당 최대 8000자로 제한
                    content = content[:8000]
                    if len(uf.get("content_full", "")) > 8000:
                        content += f"\n... (총 {len(uf['content_full'])}자 중 8000자 표시)"
                else:
                    content = content[:50000]  # API: 파일당 최대 50000자
                    if len(uf.get("content_full", "")) > 50000:
                        content += f"\n... (총 {len(uf['content_full'])}자 중 50000자 표시)"
                system_prompt += content + "\n"
            system_prompt += "\n사용자가 업로드된 파일에 대해 질문하면 위 내용을 기반으로 답변하세요.\n\n"

        # 가짜 데이터 생성 금지 + 요청 내용만 응답 규칙 (GGUF + API 공통)
        system_prompt += (
            "\n[필수 규칙]\n"
            "- 가짜 데이터를 절대 만들지 마세요! 업로드된 CSV/파일의 실제 컬럼명과 값만 사용하세요.\n"
            "- 존재하지 않는 컬럼명(Score1, Score2 등)을 지어내지 마세요.\n"
            "- 사용자가 요청한 내용에만 답변하세요. 요청하지 않은 추가 분석이나 설명을 하지 마세요.\n"
            "- 차트 데이터는 실제 데이터 기반으로 24개 이하로 요약하세요.\n"
            "- 응답이 길어질 것 같으면 핵심만 먼저 보여주고 '추가 분석이 필요하면 말씀해주세요'로 마무리하세요.\n\n"
        )

        # 응답 수준
        effort_map = [
            "매우 간결하게 핵심만 답변하세요.",
            "간결하게 답변하세요.",
            "표준적인 깊이로 설명하세요.",
            "매우 상세하고 전문적으로 분석하세요. 코드에 주석을 자세히 달고, 원리를 설명하세요.",
        ]
        format_map = {
            "code": "답변을 Python 코드 중심으로 작성하세요. import 포함, 즉시 실행 가능하게.",
            "code-fix": "문제 진단 → 수정 전/후 비교 → 수정 이유 설명 순서로 답변하세요. 최소 변경 원칙.",
            "analysis": "분석 중심으로 답변하세요. 문제 정의 → 핵심 가정 → 옵션 비교(장단점/트레이드오프) → 결론/권장안 순서. 코드보다 근거와 의사결정을 우선하고, 코드는 꼭 필요한 경우에만 최소한으로 포함하세요.",
            "report": "반드시 보고서 형식의 글(텍스트)로 작성하세요. 코드를 생성하지 마세요. 구조: 제목 → 요약(Executive Summary) → 배경/목적 → 본문(핵심 내용, 분석, 근거) → 결론 및 제언. 전문적이고 읽기 쉬운 문서 형태로 답변하세요.",
            "step-by-step": "답변을 단계별(1,2,3...)로 작성하세요. 각 단계마다 무엇을 하는지, 왜 필요한지를 글로 설명하세요. 코드보다 설명과 이해를 우선하세요.",
        }

        system_prompt += effort_map[min(effort, 3)] + "\n"
        fmt_instruction = format_map.get(output_format, "")
        if fmt_instruction:
            system_prompt += fmt_instruction + "\n"
        # 코드가 아닌 출력 형식일 때는 코드 중심 답변을 억제
        if output_format in ("report", "analysis", "step-by-step"):
            system_prompt += "중요: 사용자가 코드를 명시적으로 요청하지 않는 한, 코드 블록 없이 자연어(글)로 답변하세요.\n"
        if writing_style:
            system_prompt += f"작성 스타일: {writing_style}\n"

        # API 요청 구성
        # 일부 모델 (Qwen3.5-397B 등) 은 system 메시지가 1개 + 맨 앞에만 허용 →
        # messages 안의 모든 system 들을 system_prompt 에 통합
        _sys_extras = []
        _non_sys = []
        for _m in messages:
            if _m.get("role") == "system":
                _sys_extras.append(_m.get("content", ""))
            else:
                _non_sys.append(_m)
        _combined_system = system_prompt
        if _sys_extras:
            _combined_system = system_prompt + "\n\n" + "\n\n".join(_sys_extras)
        api_messages = [{"role": "system", "content": _combined_system}] + _non_sys
        temperature_map = [0.1, 0.3, 0.5, 0.7]

        # ===== VL 모델: 이미지 첨부 시 OpenAI Vision API 포맷 변환 (GGUF / API 공통) =====
        # env_id가 vl-로 시작하거나, gguf- 계열 중 모델명에 vl이 포함된 경우
        is_gguf_vl = env_id.startswith("gguf-") and "vl" in ENV_CONFIG.get(env_id, {}).get("name", "").lower()
        has_vision = "vision" in get_model_capabilities(env_id) or is_gguf_vl

        image_files = [f for f in uploaded_files if f.get("type") == "image" and f.get("img_base64")]

        if has_vision and image_files and api_messages:
            # 마지막 user 메시지를 멀티모달 포맷으로 변환
            for i in range(len(api_messages) - 1, -1, -1):
                if api_messages[i].get("role") == "user":
                    text_content = api_messages[i].get("content", "")
                    if isinstance(text_content, str):
                        content_parts = [{"type": "text", "text": text_content}]
                        for img_f in image_files:
                            ext = img_f.get("ext", "png").lower()
                            mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif",
                                    "bmp": "bmp", "webp": "webp", "svg": "svg+xml"}.get(ext, "png")
                            content_parts.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{mime};base64,{img_f['img_base64']}"
                                }
                            })
                        api_messages[i]["content"] = content_parts
                    break

        def _normalize_gguf_artifact_answer(answer_text, artifact_request=False):
            """GGUF 출력의 코드블록 라벨을 보정해 프론트 버튼 감지를 안정화."""
            if not artifact_request or not isinstance(answer_text, str) or not answer_text.strip():
                return answer_text

            import re as _re_norm
            normalized = answer_text
            # ```py, ```python3 -> ```python
            normalized = _re_norm.sub(r"```(?:py|python3)\s*\n", "```python\n", normalized, flags=_re_norm.IGNORECASE)

            # 무라벨 코드블록에서 PPT/Draw.io 패턴을 감지해 라벨 부여
            def _upgrade_unlabeled(m):
                body = m.group(1)
                low = body.lower()
                if any(tok in low for tok in ("from pptx", "import pptx", "presentation(", "add_slide")):
                    return f"```python\n{body}```"
                if any(tok in low for tok in ("<mxfile", "<mxgraphmodel", "<mxcell")):
                    return f"```drawio\n{body}```"
                return m.group(0)

            normalized = _re_norm.sub(r"```[ \t]*\n([\s\S]*?)```", _upgrade_unlabeled, normalized)

            # 코드블록이 전혀 없지만 python-pptx 코드가 섞여 있으면 후행 코드블록으로 보강
            low_all = normalized.lower()
            if ("from pptx" in low_all or "import pptx" in low_all) and "```python" not in low_all:
                lines = normalized.splitlines()
                start = -1
                for i, line in enumerate(lines):
                    ll = line.lower()
                    if "from pptx" in ll or "import pptx" in ll or "presentation(" in ll:
                        start = i
                        break
                if start >= 0:
                    candidate = "\n".join(lines[start:]).strip()
                    if candidate:
                        normalized += "\n\n```python\n" + candidate + "\n```"

            return normalized

        # ===== GGUF 로컬 모델: Python에서 직접 추론 =====
        if env_id.startswith("gguf-"):

            # ── 병렬 멀티에이전트 판단 ──
            # 스킬 2개+ & 다른 그룹이면 병렬 (다중 모델 선택 시에만)
            # 단일 GGUF 선택 시 병렬 안 함 (VRAM 부족 방지)
            _parallel_fallback_reason = None
            if False:  # 병렬 제거: 단일 에이전트만 사용
                _pre_skills, _par_groups, _use_parallel = group_skills_for_parallel(loaded)

                if _use_parallel:
                    try:
                        # 수동 다중 선택 시 → 선택한 모델 사용, 아니면 풀에서 자동
                        if multi_model_parallel and selected_model_paths:
                            _gguf_paths_by_size = sorted(selected_model_paths, key=lambda x: x[1], reverse=True)
                        else:
                            _PARALLEL_MIN_SIZE_GB = float(os.getenv("GGUF_PARALLEL_MIN_GB", "0.5"))
                            _gguf_envs = {k: v for k, v in ENV_CONFIG.items()
                                          if k.startswith("gguf-") and "_gguf_path" in v}
                            _gguf_paths_by_size = sorted(
                                [(v["_gguf_path"], v.get("_size_gb", 0)) for v in _gguf_envs.values()
                                 if "vl" not in os.path.basename(v["_gguf_path"]).lower()
                                 and v.get("_size_gb", 0) >= _PARALLEL_MIN_SIZE_GB],
                                key=lambda x: x[1], reverse=True,
                            )

                        if False:  # 병렬 제거: 단일 에이전트만 사용
                            # 그룹별 모델 할당
                            _assignments = _assign_models_to_groups(_par_groups, _gguf_paths_by_size)

                            # 그룹별 스킬 콘텐츠 준비
                            _group_skill_contents = {}
                            for pg in _par_groups:
                                contents = {}
                                for sid in pg["skills"]:
                                    c = load_skill_content(sid)
                                    if c:
                                        contents[sid] = c
                                _group_skill_contents[pg["group"]] = contents

                            # 마지막 사용자 메시지 추출
                            _last_query = ""
                            for m in reversed(messages):
                                if m.get("role") == "user":
                                    _last_query = m.get("content", "")
                                    if isinstance(_last_query, list):
                                        _last_query = " ".join(
                                            p.get("text", "") for p in _last_query
                                            if isinstance(p, dict) and p.get("type") == "text"
                                        )
                                    break

                            try:
                                print(f"\n  [PARALLEL] {len(_par_groups)} groups")
                            except Exception:
                                pass
                            for pg in _par_groups:
                                m_path = _assignments.get(pg["group"], "")
                                try:
                                    print(f"     [{pg['group']}] skills={pg['skills']} -> {os.path.basename(m_path)}")
                                except Exception:
                                    pass

                            # ThreadPoolExecutor로 병렬 실행
                            _agent_results = []
                            with ThreadPoolExecutor(max_workers=min(len(_par_groups), MAX_POOL_SIZE)) as executor:
                                futures = {}
                                for pg in _par_groups:
                                    model_path = _assignments[pg["group"]]
                                    future = executor.submit(
                                        _agent_call_gguf,
                                        model_path=model_path,
                                        skill_ids=pg["skills"],
                                        skill_contents=_group_skill_contents[pg["group"]],
                                        query=_last_query,
                                        history=messages[-6:],
                                        n_ctx=user_n_ctx if user_n_ctx > 0 else 16384,
                                        temperature=temperature_map[min(effort, 3)],
                                        max_tokens=TOKEN_SETTINGS["parallel_agent_max_tokens"],
                                        csv_data=uploaded_csv_data if uploaded_csv_data.get("filename") else None,
                                        uploaded_files_data=uploaded_files if uploaded_files else None,
                                    )
                                    futures[future] = pg["group"]

                                for future in as_completed(futures, timeout=180):
                                    try:
                                        result = future.result(timeout=_api_timeout)
                                        _agent_results.append(result)
                                    except Exception as e:
                                        _agent_results.append({
                                            "group": futures[future], "skills": [],
                                            "response": "", "error": str(e), "model": "",
                                        })

                            # 에이전트 결과 로그
                            for _ar in _agent_results:
                                _status = "OK" if _ar.get("response") and not _ar.get("error") else "FAIL"
                                _err_msg = f" -> {_ar['error']}" if _ar.get("error") else ""
                                try:
                                    print(f"     [{_status}] [{_ar.get('group','?')}] {_ar.get('model','?')}{_err_msg}")
                                except Exception:
                                    pass

                            # 합성 (가장 큰 모델 사용)
                            _synth_path = _gguf_paths_by_size[0][0]
                            _answer, _synth_err, _meta = _synthesize_responses_gguf(
                                _agent_results, _last_query, _synth_path,
                                temperature=temperature_map[min(effort, 3)],
                                n_ctx=user_n_ctx if user_n_ctx > 0 else 32768,
                                synth_max_tokens=max_tokens if max_tokens > 8192 else 8192,
                            )

                            if _answer and not _synth_err:
                                _answer = _normalize_gguf_artifact_answer(_answer, gguf_artifact_request)
                                try:
                                    print(f"  [PARALLEL] done: {_meta.get('agents', 0)} agents, synthesis={_meta.get('synthesis', 'N/A')}")
                                except Exception:
                                    pass
                                return jsonify({
                                    "content": _answer,
                                    "loaded_skills": loaded,
                                    "system_prompt_length": len(system_prompt),
                                    "parallel_agents": _meta.get("agents", 0),
                                    "parallel_groups": _meta.get("groups", []),
                                    "parallel_models": _meta.get("models", []),
                                    "parallel_failed": _meta.get("failed", 0),
                                    "parallel_synthesis": _meta.get("synthesis", ""),
                                    "tokens_budget": f"parallel {_meta.get('agents', 0)} agents",
                                })
                            else:
                                _parallel_fallback_reason = f"synthesis_failed: {_synth_err}"
                                try:
                                    print(f"  [PARALLEL] synthesis failed, fallback: {_synth_err}")
                                except Exception:
                                    pass

                    except Exception as _par_ex:
                        _parallel_fallback_reason = f"parallel_error: {_par_ex}"
                        try:
                            print(f"  [PARALLEL] error, fallback: {_par_ex}")
                        except Exception:
                            pass

            # ── 기존 단일모델 경로 (폴백 포함) ──
            # 선택된 환경의 모델 경로로 동적 로드/스왑
            gguf_path = ENV_CONFIG.get(env_id, {}).get("_gguf_path")
            _load_n_ctx = user_n_ctx if user_n_ctx > 0 else 32768
            if gguf_path:
                if not load_gguf_model(gguf_path, n_ctx=_load_n_ctx):
                    return jsonify({"error": f"GGUF 모델 로드 실패: {os.path.basename(gguf_path)}"}), 500
            if _utils_mod.gguf_model is None:
                return jsonify({"error": "GGUF 모델이 로드되지 않았습니다. .gguf 파일과 llama-cpp-python이 필요합니다."}), 400

            # GGUF 컨텍스트 한도 내에서 max_tokens 자동 조정 (보수적 계산)
            gguf_ctx_attr = getattr(_utils_mod.gguf_model, 'n_ctx', None)
            gguf_ctx = gguf_ctx_attr() if callable(gguf_ctx_attr) else (gguf_ctx_attr if gguf_ctx_attr is not None else 32768)
            gguf_reply_cap = max(256, TOKEN_SETTINGS["gguf_reply_cap"])
            gguf_ctx_reserve = max(512, TOKEN_SETTINGS["gguf_ctx_reserve"])

            def _estimate_gguf_prompt_tokens(msgs):
                # 1) 모델 토크나이저 기반 추정 (가능하면 사용)
                try:
                    flat_parts = []
                    for m in msgs:
                        role = m.get("role", "user")
                        content = m.get("content", "")
                        if isinstance(content, list):
                            text_parts = []
                            for p in content:
                                if isinstance(p, dict) and p.get("type") == "text":
                                    text_parts.append(str(p.get("text", "")))
                            content = "\n".join(text_parts)
                        flat_parts.append(f"{role}: {str(content)}")
                    flat_text = "\n".join(flat_parts).encode("utf-8", errors="ignore")
                    toks = _utils_mod.gguf_model.tokenize(flat_text, add_bos=True)
                    return len(toks) + 256  # chat template/여유 버퍼
                except Exception:
                    # 2) 폴백: 한국어 기준 보수 추정
                    total_chars = 0
                    for m in msgs:
                        content = m.get("content", "")
                        if isinstance(content, list):
                            text_parts = []
                            for p in content:
                                if isinstance(p, dict) and p.get("type") == "text":
                                    text_parts.append(str(p.get("text", "")))
                            content = "\n".join(text_parts)
                        total_chars += len(str(content))
                    return int(total_chars * 1.1) + 512

            prompt_tokens_est = _estimate_gguf_prompt_tokens(api_messages)

            # 컨텍스트 초과 시 자동 트림 — 시스템 메시지는 유지하고 오래된 user/assistant 페어부터 제거
            budget = gguf_ctx - gguf_ctx_reserve - 256  # 최소 응답 256토큰 보장
            trimmed = 0
            while prompt_tokens_est > budget and len(api_messages) > 1:
                removed_idx = -1
                for i, m in enumerate(api_messages):
                    if m.get("role") in ("user", "assistant"):
                        removed_idx = i
                        break
                if removed_idx < 0:
                    break
                api_messages.pop(removed_idx)
                trimmed += 1
                prompt_tokens_est = _estimate_gguf_prompt_tokens(api_messages)

            if trimmed > 0:
                print(f"[gguf] 컨텍스트 초과 → 오래된 메시지 {trimmed}개 자동 제거 (현재 ~{prompt_tokens_est}토큰)")

            safe_max = max(256, gguf_ctx - prompt_tokens_est - gguf_ctx_reserve)
            actual_max_tokens = min(max_tokens, safe_max, gguf_reply_cap)
            # 차단 가드 제거 — 트림 후에도 길면 그대로 진행 (llama.cpp가 처리)

            answer, err = gguf_chat(
                api_messages,
                temperature=temperature_map[min(effort, 3)],
                max_tokens=actual_max_tokens,
                stop_flag=chat_stop_flag,
            )
            # GGUF 디코드 실패 시: 경량 컨텍스트 + 작은 토큰으로 1회 자동 재시도
            if err and isinstance(err, str) and ("Failed completely even with batch size 1" in err or "llama.eval(decode)" in err):
                compact_system = (
                    "당신은 도움이 되는 AI입니다. "
                    "핵심만 간결하게 답하고, 코드 요청이면 실행 가능한 코드만 출력하세요."
                )
                compact_messages = [{"role": "system", "content": compact_system}]
                compact_messages.extend(messages[-6:] if len(messages) > 6 else messages)

                compact_prompt_est = _estimate_gguf_prompt_tokens(compact_messages)
                compact_safe_max = max(256, gguf_ctx - compact_prompt_est - gguf_ctx_reserve)
                retry_max = min(2048, max(256, actual_max_tokens // 2), compact_safe_max)

                retry_answer, retry_err = gguf_chat(
                    compact_messages,
                    temperature=temperature_map[min(effort, 3)],
                    max_tokens=retry_max,
                    stop_flag=chat_stop_flag,
                )
                if not retry_err and retry_answer:
                    answer = retry_answer
                    err = None
                    prompt_tokens_est = compact_prompt_est
                    actual_max_tokens = retry_max

            if err:
                return jsonify({"error": err}), 500

            # GGUF: <think> 사고만 있고 본문 없이 잘린 경우 재시도
            import re as _re
            if answer and answer.strip():
                stripped = answer.strip()
                has_open = "<think>" in stripped
                has_close = "</think>" in stripped
                think_only_gguf = False
                if has_open and not has_close:
                    think_only_gguf = True
                elif has_open and has_close:
                    after = _re.sub(r'<think>[\s\S]*?</think>\s*', '', stripped).strip()
                    if len(after) < 20:
                        think_only_gguf = True
                if think_only_gguf:
                    retry_max = min(actual_max_tokens * 2, safe_max)
                    if retry_max > actual_max_tokens:
                        retry_answer, retry_err = gguf_chat(
                            api_messages,
                            temperature=temperature_map[min(effort, 3)],
                            max_tokens=retry_max,
                            stop_flag=chat_stop_flag,
                        )
                        if not retry_err and retry_answer:
                            answer = retry_answer

            # 반복 루프 감지 및 절단 (소형 GGUF 모델 보호)
            answer, _was_rep = _detect_repetition(answer)

            answer = _normalize_gguf_artifact_answer(answer, gguf_artifact_request)

            _q_valid, _q_issues = _validate_response(answer, last_user_query)
            _quality = _calculate_quality_score(answer, last_user_query, _q_issues)
            try:
                print(f"  [QUALITY] score={_quality['score']}, grade={_quality['grade']}, issues={_q_issues}")
            except Exception:
                pass
            answer = _strip_thinking_artifacts(answer)
            _resp = {
                "content": answer,
                "loaded_skills": loaded,
                "system_prompt_length": len(system_prompt),
                "tokens_budget": f"prompt~{prompt_tokens_est}, max_tokens={actual_max_tokens}, ctx={gguf_ctx}",
                "quality": _quality,
            }
            if _parallel_fallback_reason:
                _resp["parallel_fallback"] = _parallel_fallback_reason
            _resp = _maybe_generate_md_html(answer, loaded, _resp)
            _auto_save_feedback(loaded, _quality, last_user_query)
            return jsonify(_resp)


        # ===== 회사 API: HTTP 요청 (폴백 체인 지원) =====

        # ── API 다중 선택 병렬 ──
        if False:  # 병렬 제거: 단일 에이전트만 사용
            _pre_skills, _par_groups, _use_parallel = group_skills_for_parallel(loaded)
            if _use_parallel:
                try:
                    # API 모델 목록 (선택한 순서대로)
                    _api_models = []
                    for ue in user_envs:
                        if ue in ENV_CONFIG:
                            _api_models.append({"env_id": ue, **ENV_CONFIG[ue]})
                    if len(_api_models) >= 2:
                        # 그룹 → API 모델 배정 (라운드로빈)
                        _api_assignments = {}
                        for i, pg in enumerate(_par_groups):
                            _api_assignments[pg["group"]] = _api_models[i % len(_api_models)]

                        # 그룹별 스킬 콘텐츠
                        _api_group_contents = {}
                        for pg in _par_groups:
                            contents = {}
                            for sid in pg["skills"]:
                                c = load_skill_content(sid)
                                if c:
                                    contents[sid] = c
                            _api_group_contents[pg["group"]] = contents

                        _last_query = ""
                        for m in reversed(messages):
                            if m.get("role") == "user":
                                _last_query = m.get("content", "")
                                if isinstance(_last_query, list):
                                    _last_query = " ".join(
                                        p.get("text", "") for p in _last_query
                                        if isinstance(p, dict) and p.get("type") == "text"
                                    )
                                break

                        # ThreadPoolExecutor로 API 병렬 (모듈 레벨 _api_agent_call 사용)
                        _api_results = []
                        _temp = temperature_map[min(effort, 3)]
                        with ThreadPoolExecutor(max_workers=min(len(_par_groups), 4)) as executor:
                            futures = {}
                            for pg in _par_groups:
                                api_info = _api_assignments[pg["group"]]
                                future = executor.submit(
                                    _api_agent_call,
                                    api_info=api_info,
                                    skill_ids=pg["skills"],
                                    skill_contents=_api_group_contents[pg["group"]],
                                    query=_last_query,
                                    hist=messages,
                                    api_key=api_key,
                                    temperature=_temp,
                                    csv_data=uploaded_csv_data if uploaded_csv_data.get("filename") else None,
                                    uploaded_files_data=uploaded_files if uploaded_files else None,
                                )
                                futures[future] = pg["group"]
                            for future in as_completed(futures, timeout=180):
                                try:
                                    _api_results.append(future.result(timeout=_api_timeout))
                                except Exception as e:
                                    _api_results.append({"group": futures[future], "skills": [],
                                                         "response": "", "error": str(e), "model": ""})

                        # 합성 (첫 번째 API로)
                        successes = [r for r in _api_results if r.get("response") and not r.get("error")]
                        failures = [r for r in _api_results if r.get("error")]

                        if len(successes) == 0:
                            pass  # 전부 실패 → 폴백
                        elif len(successes) == 1:
                            _sq = _calculate_quality_score(successes[0]["response"], _last_query)
                            return jsonify({
                                "content": successes[0]["response"],
                                "loaded_skills": loaded,
                                "system_prompt_length": len(system_prompt),
                                "parallel_agents": 1, "parallel_failed": len(failures),
                                "parallel_groups": [successes[0]["group"]],
                                "parallel_models": [successes[0]["model"]],
                                "parallel_synthesis": "",
                                "quality": _sq,
                            })
                        else:
                            # 합성 프롬프트
                            expert_sections = []
                            for r in successes:
                                snames = ", ".join(SKILL_DESC_KO.get(s, s) for s in r["skills"])
                                expert_sections.append(f"=== [{r['group']}] ({snames}) ===\n{r['response']}")
                            synth_system = (
                                f"여러 전문가의 분석을 통합하는 수석 연구원입니다.\n"
                                f"반드시 한국어로만 작성하세요.\n\n"
                                + "\n\n".join(expert_sections) +
                                "\n\n[통합 원칙] 핵심 포함, 중복 제거, 한국어, 코드 통합\n"
                                "[보고 구조] 핵심 결론 → 분석 근거 → 코드/시각화 → 추가 제안\n"
                                + ANTI_RATIONALIZATION
                            )
                            synth_api = _api_models[0]
                            try:
                                h = {"Content-Type": "application/json"}
                                if api_key:
                                    h["Authorization"] = f"Bearer {api_key}"
                                sr = chat_post(synth_api["url"], headers=h, json={
                                    "model": synth_api["model"],
                                    "messages": [{"role": "system", "content": synth_system},
                                                 {"role": "user", "content": _last_query}],
                                    "temperature": 0.3, "max_tokens": 8192, "stream": False,
                                }, timeout=_api_timeout, verify=False)
                                sr.raise_for_status()
                                sr_data = sr.json()
                                if "choices" in sr_data and len(sr_data["choices"]) > 0:
                                    synth_answer = sr_data["choices"][0].get("message", {}).get("content") or ""
                                    _sq = _calculate_quality_score(synth_answer, _last_query)
                                    return jsonify({
                                        "content": synth_answer,
                                        "loaded_skills": loaded,
                                        "system_prompt_length": len(system_prompt),
                                        "parallel_agents": len(successes), "parallel_failed": len(failures),
                                        "parallel_groups": [r["group"] for r in successes],
                                        "parallel_models": list(set(r["model"] for r in successes)),
                                        "parallel_synthesis": "model",
                                        "quality": _sq,
                                    })
                            except Exception:
                                pass
                            # 합성 실패 → 단순 연결
                            fallback = "\n\n---\n\n".join(
                                f"### {', '.join(SKILL_DESC_KO.get(s,s) for s in r['skills'])}\n{r['response']}"
                                for r in successes
                            )
                            _sq = _calculate_quality_score(fallback, _last_query)
                            return jsonify({
                                "content": fallback,
                                "loaded_skills": loaded,
                                "system_prompt_length": len(system_prompt),
                                "parallel_agents": len(successes), "parallel_failed": len(failures),
                                "parallel_groups": [r["group"] for r in successes],
                                "parallel_models": list(set(r["model"] for r in successes)),
                                "parallel_synthesis": "fallback_concat",
                                "quality": _sq,
                            })
                except Exception as e:
                    try:
                        print(f"  [API PARALLEL] error, fallback: {e}")
                    except Exception:
                        pass

        # ── API 자동 멀티에이전트 (AUTO 모드에서 스킬 2+개, 2+그룹) ──
        if False:  # 병렬 제거: 단일 에이전트만 사용
            _pre_skills, _par_groups, _use_parallel = group_skills_for_parallel(loaded)
            if _use_parallel:
                try:
                    # Hierarchical Delegation: 큰 그룹을 서브그룹으로 분할
                    _par_groups = apply_hierarchical_delegation(_par_groups)

                    _primary_reg_key = get_registry_key_for_env(env_id)
                    _auto_api_assignments = _assign_api_models_to_groups(_par_groups, _primary_reg_key)

                    # 그룹별 스킬 콘텐츠 준비
                    _auto_group_contents = {}
                    for pg in _par_groups:
                        contents = {}
                        for sid in pg["skills"]:
                            c = load_skill_content(sid)
                            if c:
                                contents[sid] = c
                        _auto_group_contents[pg["group"]] = contents

                    # 마지막 사용자 메시지 추출
                    _last_query = ""
                    for m in reversed(messages):
                        if m.get("role") == "user":
                            _last_query = m.get("content", "")
                            if isinstance(_last_query, list):
                                _last_query = " ".join(
                                    p.get("text", "") for p in _last_query
                                    if isinstance(p, dict) and p.get("type") == "text"
                                )
                            break

                    try:
                        print(f"\n  [API AUTO-PARALLEL] {len(_par_groups)} groups")
                    except Exception:
                        pass
                    for pg in _par_groups:
                        _asgn = _auto_api_assignments.get(pg["group"], {})
                        try:
                            print(f"     [{pg['group']}] skills={pg['skills']} -> {_asgn.get('model', '?')}")
                        except Exception:
                            pass

                    # ThreadPoolExecutor로 병렬 실행
                    _temp = temperature_map[min(effort, 3)]
                    _auto_api_results = []
                    with ThreadPoolExecutor(max_workers=min(len(_par_groups), 4)) as executor:
                        futures = {}
                        for pg in _par_groups:
                            assignment = _auto_api_assignments[pg["group"]]
                            future = executor.submit(
                                _api_agent_call,
                                api_info=assignment,
                                skill_ids=pg["skills"],
                                skill_contents=_auto_group_contents[pg["group"]],
                                query=_last_query,
                                hist=messages,
                                api_key=api_key,
                                temperature=_temp,
                                max_tokens=TOKEN_SETTINGS["parallel_agent_max_tokens"],
                                csv_data=uploaded_csv_data if uploaded_csv_data.get("filename") else None,
                                uploaded_files_data=uploaded_files if uploaded_files else None,
                            )
                            futures[future] = pg["group"]
                        for future in as_completed(futures, timeout=180):
                            try:
                                _auto_api_results.append(future.result(timeout=_api_timeout))
                            except Exception as e:
                                _auto_api_results.append({
                                    "group": futures[future], "skills": [],
                                    "response": "", "error": str(e), "model": "",
                                })

                    # 에이전트 결과 로그
                    for _ar in _auto_api_results:
                        _status = "OK" if _ar.get("response") and not _ar.get("error") else "FAIL"
                        _err_msg = f" -> {_ar['error']}" if _ar.get("error") else ""
                        try:
                            print(f"     [{_status}] [{_ar.get('group','?')}] {_ar.get('model','?')}{_err_msg}")
                        except Exception:
                            pass

                    # 결과 합성
                    successes = [r for r in _auto_api_results if r.get("response") and not r.get("error")]
                    failures = [r for r in _auto_api_results if r.get("error")]

                    if len(successes) == 1:
                        _sq = _calculate_quality_score(successes[0]["response"], _last_query)
                        try:
                            print(f"  [QUALITY] score={_sq['score']}, grade={_sq['grade']}")
                        except Exception:
                            pass
                        return jsonify({
                            "content": successes[0]["response"],
                            "loaded_skills": loaded,
                            "system_prompt_length": len(system_prompt),
                            "parallel_agents": 1, "parallel_failed": len(failures),
                            "parallel_groups": [successes[0]["group"]],
                            "parallel_models": [successes[0]["model"]],
                            "parallel_synthesis": "",
                            "auto_routed": auto_routed, "route_reason": route_reason,
                            "auto_multi_agent": True,
                            "quality": _sq,
                        })
                    elif len(successes) >= 2:
                        # 합성 모델 결정: low cost tier → 대형 모델로 업그레이드
                        _synth_reg_key = _primary_reg_key or "qwen36-35b"
                        _primary_cost = MODEL_REGISTRY.get(_synth_reg_key, {}).get("cost_tier", "medium")
                        if _primary_cost == "low":
                            _synth_reg_key = "qwen36-35b"
                        _synth_reg = MODEL_REGISTRY.get(_synth_reg_key, MODEL_REGISTRY["qwen36-35b"])

                        # 합성 프롬프트
                        expert_sections = []
                        for r in successes:
                            snames = ", ".join(SKILL_DESC_KO.get(s, s) for s in r["skills"])
                            expert_sections.append(f"=== [{r['group']}] ({snames}) ===\n{r['response']}")

                        synth_system = (
                            f"당신은 여러 전문가의 분석을 통합하는 수석 연구원입니다.\n"
                            f"중요: 반드시 모든 내용을 한국어로만 작성하세요. 영어를 사용하지 마세요.\n"
                            f"<think> 태그를 사용하지 마세요.\n\n"
                            f"아래 {len(successes)}명의 전문가가 각자의 전문 영역에서 답변했습니다.\n\n"
                            + "\n\n".join(expert_sections) +
                            "\n\n[통합 원칙]\n"
                            "1. 반드시 한국어로만 답변하세요 (코드 주석도 한국어)\n"
                            "2. 각 전문가의 핵심 내용을 빠짐없이 포함\n"
                            "3. 중복 내용은 한 번만 언급\n"
                            "4. 하나의 자연스러운 답변으로 통합 (전문가별로 분리하지 말 것)\n"
                            "5. 코드가 있으면 통합된 하나의 코드로 합쳐서 제공\n"
                            "6. 가짜 데이터를 만들지 마세요\n\n"
                            "[통합 보고 구조]\n"
                            "핵심 결론 → 분석 근거 → 코드/시각화 → 추가 제안 순서로 작성하세요.\n"
                            "각 전문가 영역의 기여를 자연스럽게 녹여내되, 출처는 명시하세요.\n\n"
                            + ANTI_RATIONALIZATION +
                            VERIFICATION_GATE
                        )

                        try:
                            _sh = {"Content-Type": "application/json"}
                            if api_key:
                                _sh["Authorization"] = f"Bearer {api_key}"
                            sr = chat_post(_synth_reg["url"], headers=_sh, json={
                                "model": _synth_reg["model"],
                                "messages": [{"role": "system", "content": synth_system},
                                             {"role": "user", "content": _last_query}],
                                "temperature": 0.3,
                                "max_tokens": max_tokens if max_tokens >= 8192 else 8192,
                                "stream": False,
                            }, timeout=180, verify=False)
                            sr.raise_for_status()
                            sr_data = sr.json()
                            if "choices" in sr_data and len(sr_data["choices"]) > 0:
                                synth_answer = sr_data["choices"][0].get("message", {}).get("content") or ""
                                # Self-evaluation: 합성 응답 품질 검증
                                _valid, _issues = _validate_response(synth_answer, _last_query)
                                if _issues:
                                    synth_answer = _fix_response_issues(synth_answer, _issues)
                                    try:
                                        print(f"  [EVAL] issues={_issues}, auto-fixed")
                                    except Exception:
                                        pass
                                _sq = _calculate_quality_score(synth_answer, _last_query, _issues if _issues else [])
                                try:
                                    print(f"  [API AUTO-PARALLEL] done: {len(successes)} agents, "
                                          f"synthesis={_synth_reg['model']}")
                                    print(f"  [QUALITY] score={_sq['score']}, grade={_sq['grade']}")
                                except Exception:
                                    pass
                                return jsonify({
                                    "content": synth_answer,
                                    "loaded_skills": loaded,
                                    "system_prompt_length": len(system_prompt),
                                    "parallel_agents": len(successes),
                                    "parallel_failed": len(failures),
                                    "parallel_groups": [r["group"] for r in successes],
                                    "parallel_models": list(set(r["model"] for r in successes)),
                                    "parallel_synthesis": _synth_reg["model"],
                                    "auto_routed": auto_routed, "route_reason": route_reason,
                                    "auto_multi_agent": True,
                                    "quality": _sq,
                                })
                        except Exception as _synth_ex:
                            try:
                                print(f"  [API AUTO-PARALLEL] synthesis failed: {_synth_ex}")
                            except Exception:
                                pass

                        # 합성 실패 → fallback concat
                        fallback = "\n\n---\n\n".join(
                            f"### {', '.join(SKILL_DESC_KO.get(s, s) for s in r['skills'])}\n{r['response']}"
                            for r in successes
                        )
                        _sq = _calculate_quality_score(fallback, _last_query)
                        return jsonify({
                            "content": fallback,
                            "loaded_skills": loaded,
                            "system_prompt_length": len(system_prompt),
                            "parallel_agents": len(successes),
                            "parallel_failed": len(failures),
                            "parallel_groups": [r["group"] for r in successes],
                            "parallel_models": list(set(r["model"] for r in successes)),
                            "parallel_synthesis": "fallback_concat",
                            "auto_routed": auto_routed, "route_reason": route_reason,
                            "auto_multi_agent": True,
                            "quality": _sq,
                        })
                    # else: 전부 실패 → 아래 단일모델 경로로 폴백

                except Exception as _auto_par_ex:
                    try:
                        print(f"  [API AUTO-PARALLEL] error, fallback to single: {_auto_par_ex}")
                    except Exception:
                        pass

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # 폴백 체인 구성: 현재 모델 → 대체 모델들
        primary_reg_key = get_registry_key_for_env(env_id)
        if primary_reg_key:
            # drawio/pptx 는 폴백 비활성 (모델 바뀌면 결과 완전히 달라져 처음부터 다시)
            if _srv_is_drawio or _srv_is_pptx:
                fallback_keys = [primary_reg_key]
                print(f"  [SERVER BOOST] {('drawio' if _srv_is_drawio else 'pptx')} — 폴백 비활성 (primary only)")
            else:
                fallback_keys = [primary_reg_key] + FALLBACK_CHAINS.get(primary_reg_key, [])
        else:
            fallback_keys = []  # GGUF 등은 폴백 없음

        # 이미지가 있으면 non-vision 폴백에서 이미지 제거 (텍스트만 전송)
        def _prepare_messages_for_model(reg_key, msgs):
            """VL 모델이 아닌 경우 멀티모달 content를 텍스트로 되돌리기"""
            cap = MODEL_REGISTRY.get(reg_key, {}).get("capabilities", set())
            if "vision" not in cap:
                clean = []
                for m in msgs:
                    if isinstance(m.get("content"), list):
                        text_parts = [p["text"] for p in m["content"] if p.get("type") == "text"]
                        clean.append({**m, "content": "\n".join(text_parts)})
                    else:
                        clean.append(m)
                return clean
            return msgs

        fallback_used = False
        fallback_from = ""
        actual_model_used = model
        last_error = None

        models_tried = []
        if fallback_keys:
            for attempt, reg_key in enumerate(fallback_keys[:6]):  # 최대 6회
                reg = MODEL_REGISTRY[reg_key]
                try_url = reg["url"]
                try_model = reg["model"]
                try_msgs = _prepare_messages_for_model(reg_key, api_messages)
                models_tried.append(try_model)

                try:
                    resp = chat_post(
                        try_url,
                        headers=headers,
                        json={
                            "model": try_model,
                            "messages": try_msgs,
                            "temperature": temperature_map[min(effort, 3)],
                            "max_tokens": max_tokens,
                            "stream": False,
                        },
                        timeout=_api_timeout,
                        verify=False,
                    )
                    resp.raise_for_status()
                    result = resp.json()

                    # 응답 추출
                    truncated = False
                    if "choices" in result and len(result["choices"]) > 0:
                        answer = result["choices"][0].get("message", {}).get("content") or ""
                        finish_reason = result["choices"][0].get("finish_reason", "")
                        if finish_reason == "length":
                            truncated = True

                        # <think> 사고 과정만 있고 본문이 없이 잘린 경우 자동 재시도
                        import re as _re
                        think_only = False
                        if truncated and answer.strip():
                            # </think> 닫히지 않았거나, </think> 후 본문이 비어있는 경우
                            stripped = answer.strip()
                            has_open_think = "<think>" in stripped
                            has_close_think = "</think>" in stripped
                            if has_open_think and not has_close_think:
                                think_only = True
                            elif has_open_think and has_close_think:
                                after_think = _re.sub(r'<think>[\s\S]*?</think>\s*', '', stripped).strip()
                                if len(after_think) < 20:
                                    think_only = True

                        if think_only and max_tokens < 32768:
                            # 토큰 2배로 늘려서 재시도 (최대 32768)
                            retry_max = min(max_tokens * 2, 32768)
                            try:
                                retry_resp = chat_post(
                                    try_url,
                                    headers=headers,
                                    json={
                                        "model": try_model,
                                        "messages": try_msgs,
                                        "temperature": temperature_map[min(effort, 3)],
                                        "max_tokens": retry_max,
                                        "stream": False,
                                    },
                                    timeout=180,
                                    verify=False,
                                )
                                retry_resp.raise_for_status()
                                retry_result = retry_resp.json()
                                if "choices" in retry_result and len(retry_result["choices"]) > 0:
                                    answer = retry_result["choices"][0].get("message", {}).get("content") or ""
                                    finish_reason = retry_result["choices"][0].get("finish_reason", "")
                                    truncated = finish_reason == "length"
                            except Exception:
                                pass  # 재시도 실패 시 원래 응답 사용

                        if attempt > 0:
                            fallback_used = True
                            fallback_from = models_tried[0]
                            actual_model_used = try_model

                        # 반복 루프 감지 및 절단 (API 응답 보호)
                        answer, _was_rep = _detect_repetition(answer)

                        # 품질 검증 및 점수 계산
                        _q_valid, _q_issues = _validate_response(answer, last_user_query)
                        if _q_issues:
                            answer = _fix_response_issues(answer, _q_issues)
                        _quality = _calculate_quality_score(answer, last_user_query, _q_issues)
                        try:
                            print(f"  [QUALITY] score={_quality['score']}, grade={_quality['grade']}, issues={_q_issues}")
                        except Exception:
                            pass

                        answer = _strip_thinking_artifacts(answer)
                        resp_data = {
                            "content": answer,
                            "loaded_skills": loaded,
                            "system_prompt_length": len(system_prompt),
                            "model_used": try_model,
                            "quality": _quality,
                        }
                        if auto_routed:
                            resp_data["auto_routed"] = True
                            resp_data["route_reason"] = route_reason
                        if auto_format or auto_style:
                            resp_data["auto_format"] = output_format if auto_format else None
                            resp_data["auto_style"] = writing_style if auto_style else None
                            resp_data["auto_fmt_reason"] = auto_fmt_reason
                        if fallback_used:
                            resp_data["fallback_used"] = True
                            resp_data["fallback_from"] = fallback_from
                        if truncated:
                            resp_data["truncated"] = True
                        resp_data = _maybe_generate_md_html(answer, loaded, resp_data)
                        _auto_save_feedback(loaded, _quality, last_user_query)
                        return jsonify(resp_data)

                    elif "error" in result:
                        last_error = f"API 에러: {result['error']}"
                        print(f"[Fallback] model={try_model} → error: {last_error} → trying next...")
                        continue  # 다음 폴백 시도
                    else:
                        last_error = f"예상치 못한 응답: {json.dumps(result, ensure_ascii=False, indent=2)}"
                        print(f"[Fallback] model={try_model} → error: unexpected response → trying next...")
                        continue

                except req.exceptions.Timeout:
                    last_error = "API 응답 시간 초과 (120초)"
                    print(f"[Fallback] model={try_model} → error: {last_error} → trying next...")
                    continue
                except req.exceptions.ConnectionError as e:
                    last_error = f"API 연결 실패: {str(e)[:200]}"
                    print(f"[Fallback] model={try_model} → error: {last_error} → trying next...")
                    continue
                except req.exceptions.HTTPError as e:
                    code = e.response.status_code if e.response is not None else 0
                    last_error = f"HTTP {code}: {str(e)}"
                    if e.response is not None:
                        try:
                            detail = json.dumps(e.response.json(), ensure_ascii=False)
                            last_error += f" - {detail[:300]}"
                        except Exception:
                            pass
                    print(f"[Fallback] model={try_model} → error: HTTP {code} → trying next...")
                    continue
                except Exception as e:
                    last_error = f"오류: {str(e)}"
                    print(f"[Fallback] model={try_model} → error: {last_error} → trying next...")
                    continue

            # 모든 폴백 실패
            return jsonify({
                "error": f"모든 모델 시도 실패 ({', '.join(models_tried)}): {last_error}"
            }), 500

        else:
            # 폴백 체인 없는 경우 (GGUF 등) → 기존 단일 요청
            try:
                resp = chat_post(
                    api_url,
                    headers=headers,
                    json={
                        "model": model,
                        "messages": api_messages,
                        "temperature": temperature_map[min(effort, 3)],
                        "max_tokens": max_tokens,
                        "stream": False,
                    },
                    timeout=_api_timeout,
                    verify=False,
                )
                resp.raise_for_status()
                result = resp.json()

                truncated = False
                if "choices" in result and len(result["choices"]) > 0:
                    answer = result["choices"][0].get("message", {}).get("content") or ""
                    finish_reason = result["choices"][0].get("finish_reason", "")
                    if finish_reason == "length":
                        truncated = True

                    # <think> 사고 과정만 있고 본문이 없이 잘린 경우 자동 재시도
                    import re as _re
                    think_only = False
                    if truncated and answer.strip():
                        stripped = answer.strip()
                        has_open_think = "<think>" in stripped
                        has_close_think = "</think>" in stripped
                        if has_open_think and not has_close_think:
                            think_only = True
                        elif has_open_think and has_close_think:
                            after_think = _re.sub(r'<think>[\s\S]*?</think>\s*', '', stripped).strip()
                            if len(after_think) < 20:
                                think_only = True

                    if think_only and max_tokens < 32768:
                        retry_max = min(max_tokens * 2, 32768)
                        try:
                            retry_resp = chat_post(
                                api_url,
                                headers=headers,
                                json={
                                    "model": model,
                                    "messages": api_messages,
                                    "temperature": temperature_map[min(effort, 3)],
                                    "max_tokens": retry_max,
                                    "stream": False,
                                },
                                timeout=180,
                                verify=False,
                            )
                            retry_resp.raise_for_status()
                            retry_result = retry_resp.json()
                            if "choices" in retry_result and len(retry_result["choices"]) > 0:
                                answer = retry_result["choices"][0].get("message", {}).get("content") or ""
                                finish_reason = retry_result["choices"][0].get("finish_reason", "")
                                truncated = finish_reason == "length"
                        except Exception:
                            pass

                elif "error" in result:
                    answer = f"API 에러: {result['error']}"
                else:
                    answer = f"예상치 못한 응답: {json.dumps(result, ensure_ascii=False, indent=2)}"

                # 품질 점수 계산
                _q_valid, _q_issues = _validate_response(answer, last_user_query)
                if _q_issues:
                    answer = _fix_response_issues(answer, _q_issues)
                _quality = _calculate_quality_score(answer, last_user_query, _q_issues)
                try:
                    print(f"  [QUALITY] score={_quality['score']}, grade={_quality['grade']}, issues={_q_issues}")
                except Exception:
                    pass

                answer = _strip_thinking_artifacts(answer)
                resp_data = {
                    "content": answer,
                    "loaded_skills": loaded,
                    "system_prompt_length": len(system_prompt),
                    "model_used": model,
                    "quality": _quality,
                }
                if auto_routed:
                    resp_data["auto_routed"] = True
                    resp_data["route_reason"] = route_reason
                if auto_format or auto_style:
                    resp_data["auto_format"] = output_format if auto_format else None
                    resp_data["auto_style"] = writing_style if auto_style else None
                    resp_data["auto_fmt_reason"] = auto_fmt_reason
                if truncated:
                    resp_data["truncated"] = True

                # 하네스: 세션 자동 저장 + 이벤트 로깅
                if HARNESS_AVAILABLE:
                    try:
                        save_chat_session(
                            messages=messages,
                            skills_used=loaded,
                            metadata={"model": model, "format": output_format},
                        )
                        log_event('chat', f'model={model} skills={len(loaded)} prompt_len={len(system_prompt)}')
                    except Exception:
                        pass

                resp_data = _maybe_generate_md_html(answer, loaded, resp_data)
                _auto_save_feedback(loaded, _quality, last_user_query)
                return jsonify(resp_data)

            except req.exceptions.Timeout:
                return jsonify({"error": "API 응답 시간 초과 (120초). max_tokens를 줄이거나 API 서버 상태를 확인하세요."}), 504
            except req.exceptions.ConnectionError as e:
                return jsonify({"error": f"API 연결 실패: {str(e)}. URL을 확인하세요."}), 502
            except req.exceptions.HTTPError as e:
                code = e.response.status_code if e.response is not None else 0
                if code == 401 or code == 403:
                    return jsonify({"error": f"인증 실패 ({code}): TOKEN.TXT의 API 키를 확인하세요."}), code
                detail = ""
                if e.response is not None:
                    try:
                        body = e.response.json()
                        detail = json.dumps(body, ensure_ascii=False, indent=2)
                    except Exception:
                        detail = e.response.text[:500] if e.response.text else ""
                prompt_chars = sum(len(m.get("content","")) for m in api_messages)
                err_msg = f"API HTTP 에러 ({code}): {str(e)}"
                if detail:
                    err_msg += f"\n서버 응답: {detail}"
                err_msg += f"\n[요청 크기: 전체 메시지={prompt_chars}자, model={model}]"
                return jsonify({"error": err_msg}), code or 500
            except Exception as e:
                return jsonify({"error": f"오류 발생: {str(e)}"}), 500


