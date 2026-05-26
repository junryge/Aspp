"""
Hermes Web — 독립 Flask 웹 UI

- demos_v1 와 완전 분리. demos 실행 여부와 무관하게 동작.
- 단, hermes-web/hermes_proxy.py (:8765) 가 떠있어야 hermes 가 응답함.
- hermes 자체는 ~/.hermes/config.yaml 의 model/base_url 사용
  (템플릿: hermes-web/hermes_config.yaml — common.llm.skhynix.com + Kimi-K2.5 디폴트).
- 환경 자동감지: F:/M14_Q/... 존재하면 HOME, C:/연구과제/... 존재하면 OFFICE.

라우트:
  GET  /                    → static/index.html
  GET  /api/health          → 프록시 + hermes 살아있나 + 환경 모드
  GET  /api/skills          → scientific-skills/ 폴더 스캔, 389 카탈로그 JSON
  POST /api/chat            → SSE 토큰 스트리밍 (hermes subprocess)

실행:
  python server.py             (기본 포트 8788)
  python server.py 9000        (포트 지정)
"""
import os
import re
import sys
import json
import socket
import subprocess
from pathlib import Path
from flask import Flask, request, Response, jsonify, stream_with_context, send_from_directory

# ─────────────────────────────────────────────────────────────────────────────
# 환경 자동 감지 (HOME / OFFICE)
# ─────────────────────────────────────────────────────────────────────────────
HOME_BASE   = r"F:\M14_Q\scientific-assistant"
OFFICE_BASE = r"C:\연구과제\CODE\데모스_분석툴\scientific-assistant"

if os.path.isdir(HOME_BASE):
    ENV_MODE = "HOME"
    SCIENTIFIC_BASE = HOME_BASE
elif os.path.isdir(OFFICE_BASE):
    ENV_MODE = "OFFICE"
    SCIENTIFIC_BASE = OFFICE_BASE
else:
    ENV_MODE = "UNKNOWN"
    SCIENTIFIC_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 스킬 디렉토리: ~/.hermes/skills (실제로 hermes 가 보는 곳) 우선,
# 없으면 옛 scientific-skills/ 폴백
_HERMES_SKILLS = os.path.expanduser("~/.hermes/skills")
if os.path.isdir(_HERMES_SKILLS):
    SKILLS_DIR = _HERMES_SKILLS
else:
    SKILLS_DIR = os.path.join(SCIENTIFIC_BASE, "scientific-skills")
PROXY_HOST = "127.0.0.1"
PROXY_PORT = 8765

# ─────────────────────────────────────────────────────────────────────────────
# Windows pywinpty (hermes prompt_toolkit 가 콘솔 요구)
# ─────────────────────────────────────────────────────────────────────────────
_USE_WINPTY = False
_WINPTY_MOD = None
if sys.platform == "win32":
    try:
        import winpty as _WINPTY_MOD
        _USE_WINPTY = True
    except Exception:
        _USE_WINPTY = False

# ─────────────────────────────────────────────────────────────────────────────
# ANSI/CSI 이스케이프 정규식 (CSI ?, OSC title, 컬러)
# ─────────────────────────────────────────────────────────────────────────────
_ANSI_RE = re.compile(
    r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"
    r"|\x1b\[[\?>=!]?[0-9;]*[ -/]*[@-~]"
    r"|\x1b[()][AB012]"
    r"|\x1b[=>]"
    r"|\x1b\][0-9]+;[^\x07\x1b]*"
)

# hermes CLI TUI 노이즈: 박스 드로잉, 메타 헤더만 — 마크다운 불릿/표는 보존
NOISE_PREFIXES = (
    "✔", "ℹ",
    "Elapsed:", "Context:", "Session:", "Resume", "Duration:",
    "Messages:", "hermes --resume", "  hermes --resume",
    "Query:", "Initializing", "Initialized",
)

# Hermes 메타 줄 (어디든 매치)
_META_LINE_RE = re.compile(
    r"\b(Session|Elapsed|Context|Duration|Messages|Resume\sthis\ssession)\s*:|"
    r"hermes\s+--resume\s+\S+",
)

# 박스 드로잉만으로 이뤄진 구분선 — 앞에 한 글자(예: "Q") 가 붙어있어도 매치
_BOXLINE_RE = re.compile(r"^[A-Z]?\s*[─═━┌┐└┘├┤┬┴┼│]{3,}\s*$")

# hermes 가 답변 전에 토하는 "Q\n" 단독 prompt-echo 라인
_Q_PROMPT_RE = re.compile(r"^Q\s*$")

# 명시적 thinking 마커 (보수적) — 일반 산문은 절대 안 잡게
_THINK_MARKER_RE = re.compile(
    r"^\s*(?:<think>|</think>|<thinking>|</thinking>|"
    r"\[?Inner monologue\]?|\[?Reasoning\]?\s*:)",
    re.IGNORECASE,
)


def _strip_ansi(s):
    return _ANSI_RE.sub("", s or "")


def _normalize_for_match(s):
    """선두/끝 비가시 문자 제거 — prefix 매치 신뢰성 확보."""
    return (s or "").strip().lstrip("​﻿\xa0")


def _is_proxy_alive():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.4)
    try:
        s.connect((PROXY_HOST, PROXY_PORT))
        return True
    except Exception:
        return False
    finally:
        try:
            s.close()
        except Exception:
            pass


def _find_hermes():
    """hermes 실행파일 경로 찾기. PATH 우선, 없으면 Python Scripts 폴더."""
    import shutil
    for name in ("hermes", "hermes.exe"):
        p = shutil.which(name)
        if p:
            return p
    # 폴백: 현재 인터프리터의 Scripts 폴더
    py_dir = os.path.dirname(sys.executable)
    for cand in (
        os.path.join(py_dir, "hermes.exe"),
        os.path.join(py_dir, "Scripts", "hermes.exe"),
    ):
        if os.path.isfile(cand):
            return cand
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 스킬 카탈로그 스캔 (Level 0 메타만)
# ─────────────────────────────────────────────────────────────────────────────
_SKILL_CACHE = None


def _parse_skill_md(path):
    """SKILL.md 의 YAML frontmatter 에서 name, description 추출."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read(4000)  # 첫 4KB 면 충분
    except Exception:
        return None
    name = None
    desc = None
    # YAML frontmatter
    m = re.match(r"---\s*\n([\s\S]+?)\n---", text)
    if m:
        body = m.group(1)
        nm = re.search(r"^name:\s*(.+?)$", body, re.MULTILINE)
        dm = re.search(r"^description:\s*(.+?)$", body, re.MULTILINE)
        if nm:
            name = nm.group(1).strip().strip('"').strip("'")
        if dm:
            desc = dm.group(1).strip().strip('"').strip("'")
    return name, (desc or "")[:300]


def _scan_skills():
    global _SKILL_CACHE
    if _SKILL_CACHE is not None:
        return _SKILL_CACHE
    items = []
    if not os.path.isdir(SKILLS_DIR):
        _SKILL_CACHE = items
        return items
    for entry in sorted(os.listdir(SKILLS_DIR)):
        d = os.path.join(SKILLS_DIR, entry)
        if not os.path.isdir(d):
            continue
        skill_md = os.path.join(d, "SKILL.md")
        if not os.path.isfile(skill_md):
            continue
        parsed = _parse_skill_md(skill_md)
        if not parsed:
            continue
        name, desc = parsed
        items.append({
            "id": entry,
            "name": name or entry,
            "description": desc,
            "category": _categorize(entry, name or entry, desc or ""),
        })
    _SKILL_CACHE = items
    return items


# ─────────────────────────────────────────────────────────────────────────────
# 스킬 대분류 (8개 + 기타)
# ─────────────────────────────────────────────────────────────────────────────
SKILL_CATEGORIES = [
    {"key": "all",     "name": "전체"},
    {"key": "code",    "name": "코드 작성"},
    {"key": "review",  "name": "코드 리뷰·디버깅"},
    {"key": "docs",    "name": "문서·MD·PPT"},
    {"key": "skill",   "name": "스킬·에이전트 만들기"},
    {"key": "data",    "name": "데이터·DB"},
    {"key": "science", "name": "과학·리서치"},
    {"key": "design",  "name": "시각화·디자인"},
    {"key": "ops",     "name": "Git·배포·DevOps"},
    {"key": "etc",     "name": "기타"},
]

# 명시 오버라이드 — 패턴으로 안 잡히는 항목들
_CATEGORY_OVERRIDE = {
    # ── 코드 리뷰·디버깅 ──
    "code-review": "review",
    "requesting-code-review": "review",
    "peer-review": "review",
    "debugging": "review",
    "systematic-debugging": "review",
    "test-driven-development": "review",
    "webapp-testing": "review",
    "verification-before-completion": "review",
    "owasp-security": "review",
    "scholar-evaluation": "review",
    "scientific-critical-thinking": "review",
    "agent-code-reviewer": "review",
    "agent-debugger": "review",
    "agent-error-coordinator": "review",
    "agent-error-detective": "review",
    "agent-test-automator": "review",
    "agent-qa-expert": "review",
    "agent-chaos-engineer": "review",
    "agent-architect-reviewer": "review",
    "agent-refactoring-specialist": "review",
    "agent-legacy-modernizer": "review",
    "agent-security-auditor": "review",
    "agent-security-engineer": "review",
    "agent-penetration-tester": "review",
    "agent-compliance-auditor": "review",

    # ── 문서·MD·PPT ──
    "markdown-mermaid-writing": "docs",
    "md-to-html": "docs",
    "pdf": "docs",
    "pptx": "docs",
    "xlsx": "docs",
    "docx": "docs",
    "markitdown": "docs",
    "repomix": "docs",
    "citation-management": "docs",
    "weekly-report": "docs",
    "scientific-writing": "docs",
    "scientific-slides": "docs",
    "clinical-reports": "docs",
    "treatment-plans": "docs",
    "latex-posters": "docs",
    "pptx-posters": "docs",
    "venue-templates": "docs",
    "research-grants": "docs",
    "market-research-reports": "docs",
    "iso-13485-certification": "docs",
    "internal-comms": "docs",
    "paper-2-web": "docs",
    "changelog-generator": "docs",
    "drawio-diagram": "docs",
    "content-research-writer": "docs",
    "writing-plans": "docs",
    "literature-review": "docs",
    "agent-documentation-engineer": "docs",
    "agent-technical-writer": "docs",
    "agent-api-documenter": "docs",
    "agent-content-marketer": "docs",

    # ── 스킬·에이전트 만들기 / 프롬프트 ──
    "skill-creator": "skill",
    "skill-share": "skill",
    "template-skill": "skill",
    "writing-skills": "skill",
    "engineer-skill-creator": "skill",
    "mcp-builder": "skill",
    "anthropic-architect": "skill",
    "anthropic-prompt-engineer": "skill",
    "openai-prompt-engineer": "skill",
    "claude-code": "skill",
    "using-superpowers": "skill",
    "guide-hooks": "skill",
    "developer-growth-analysis": "skill",
    "agent-mcp-developer": "skill",
    "agent-prompt-engineer": "skill",
    "agent-llm-architect": "skill",
    "agent-ai-engineer": "skill",
    "agent-context-manager": "skill",
    "agent-agent-organizer": "skill",
    "agent-multi-agent-coordinator": "skill",
    "agent-workflow-orchestrator": "skill",
    "agent-task-distributor": "skill",

    # ── Git · 배포 · DevOps ──
    "devops": "ops",
    "github-ecosystem": "ops",
    "guide-git": "ops",
    "guide-github-actions": "ops",
    "modal": "ops",
    "agent-deployment-engineer": "ops",
    "agent-devops-engineer": "ops",
    "agent-devops-incident-responder": "ops",
    "agent-incident-responder": "ops",
    "agent-build-engineer": "ops",
    "agent-cloud-architect": "ops",
    "agent-kubernetes-specialist": "ops",
    "agent-terraform-engineer": "ops",
    "agent-sre-engineer": "ops",
    "agent-platform-engineer": "ops",
    "agent-network-engineer": "ops",
    "agent-dependency-manager": "ops",
    "agent-tooling-engineer": "ops",
    "agent-dx-optimizer": "ops",
    "agent-git-workflow-manager": "ops",

    # ── 시각화·디자인 ──
    "matplotlib": "design",
    "plotly": "design",
    "seaborn": "design",
    "scientific-visualization": "design",
    "scientific-schematics": "design",
    "aesthetic": "design",
    "ui-styling": "design",
    "frontend-design": "design",
    "web-design-guidelines": "design",
    "brand-guidelines": "design",
    "theme-factory": "design",
    "canvas-design": "design",
    "infographics": "design",
    "generate-image": "design",
    "image-enhancer": "design",
    "slack-gif-creator": "design",
    "artifacts-builder": "design",
    "web-artifacts-builder": "design",
    "agent-ui-designer": "design",
    "agent-ux-researcher": "design",

    # ── 데이터·DB (DataFrame/대용량/금융 데이터 포함) ──
    "databases": "data",
    "polars": "data",
    "dask": "data",
    "vaex": "data",
    "zarr-python": "data",
    "edgartools": "data",
    "alpha-vantage": "data",
    "fred-economic-data": "data",
    "hedgefundmonitor": "data",
    "usfiscaldata": "data",
    "datacommons-client": "data",
    "logpresso-query": "data",
    "logpresso-search": "data",
    "tiledbvcf": "data",
    "agent-data-analyst": "data",
    "agent-data-engineer": "data",
    "agent-database-administrator": "data",
    "agent-database-optimizer": "data",
    "agent-postgres-pro": "data",
    "agent-sql-pro": "data",

    # ── 과학·리서치 ──
    "biopython": "science",
    "scanpy": "science",
    "scvi-tools": "science",
    "scvelo": "science",
    "anndata": "science",
    "cellxgene-census": "science",
    "deepchem": "science",
    "rdkit": "science",
    "datamol": "science",
    "molfeat": "science",
    "medchem": "science",
    "torchdrug": "science",
    "diffdock": "science",
    "esm": "science",
    "alphafold-database": "science",
    "molecular-dynamics": "science",
    "phylogenetics": "science",
    "etetoolkit": "science",
    "astropy": "science",
    "fluidsim": "science",
    "pymatgen": "science",
    "matlab": "science",
    "sympy": "science",
    "cobrapy": "science",
    "arboreto": "science",
    "geniml": "science",
    "gtars": "science",
    "flowio": "science",
    "histolab": "science",
    "pathml": "science",
    "matchms": "science",
    "pyopenms": "science",
    "pysam": "science",
    "deeptools": "science",
    "pydeseq2": "science",
    "scikit-bio": "science",
    "scikit-survival": "science",
    "pydicom": "science",
    "pyhealth": "science",
    "neurokit2": "science",
    "neuropixels-analysis": "science",
    "pytdc": "science",
    "geomaster": "science",
    "geopandas": "science",
    "clinical-decision-support": "science",
    "glycoengineering": "science",
    "imaging-data-commons": "science",
    "pyzotero": "science",
    "scientific-brainstorming": "science",
    "hypothesis-generation": "science",
    "hypogenic": "science",
    "denario": "science",
    "research-lookup": "science",
    "bgpt-paper-search": "science",
    "notebooklm-skill": "science",
    "open-notebook": "science",
    "perplexity-search": "science",
    "parallel-web": "science",
    "ai-multimodal": "science",
    "statistical-analysis": "science",
    "statsmodels": "science",
    "pymc": "science",
    "pymoo": "science",
    "shap": "science",
    "umap-learn": "science",
    "networkx": "science",
    "transformers": "science",
    "scikit-learn": "science",
    "pytorch-lightning": "science",
    "torch-geometric": "science",
    "stable-baselines3": "science",
    "pufferlib": "science",
    "timesfm-forecasting": "science",
    "aeon": "science",
    "qiskit": "science",
    "cirq": "science",
    "pennylane": "science",
    "qutip": "science",
    "rowan": "science",
    "simpy": "science",
    "lamindb": "science",
    "bioservices": "science",
    "gget": "science",
    "adaptyv": "science",
    "benchling-integration": "science",
    "ginkgo-cloud-lab": "science",
    "opentrons-integration": "science",
    "pylabrobot": "science",
    "protocolsio-integration": "science",
    "labarchive-integration": "science",
    "dnanexus-integration": "science",
    "latchbio-integration": "science",
    "omero-integration": "science",
    "exploratory-data-analysis": "science",
    "agent-data-scientist": "science",
    "agent-data-researcher": "science",
    "agent-research-analyst": "science",
    "agent-machine-learning-engineer": "science",
    "agent-ml-engineer": "science",
    "agent-mlops-engineer": "science",
    "agent-nlp-engineer": "science",
    "agent-quant-analyst": "science",
    "agent-knowledge-synthesizer": "science",
    "agent-market-researcher": "science",
    "agent-competitive-analyst": "science",
    "agent-trend-analyst": "science",
    "agent-business-analyst": "science",
    "agent-search-specialist": "science",

    # ── 코드 작성 (개발 가이드/프레임워크) ──
    "frontend-development": "code",
    "backend-development": "code",
    "web-frameworks": "code",
    "react-best-practices": "code",
    "python-project-skel": "code",
    "python-deprecation-fixer": "code",
    "better-auth": "code",
    "shopify": "code",
    "chrome-devtools": "code",
    "docs-seeker": "code",
    "guide-python": "code",
    "guide-react": "code",
    "guide-golang": "code",
    "guide-opus-4-5": "code",
    "guide-opus-4-5-agent": "code",
    "guide-opus-migration": "code",
    "guide-mcp-reference": "code",
    "guide-version-discovery": "code",
    "guide-claude-md": "docs",
    "guide-documentation": "docs",
    "guide-testing": "review",
    "guide-hmhco": "etc",

    # ── 기타 ──
    "common": "etc",
    "get-available-resources": "etc",
    "file-organizer": "etc",
    "invoice-organizer": "etc",
    "raffle-winner-picker": "etc",
    "video-downloader": "etc",
    "media-processing": "etc",
    "datadog-entity-generator": "ops",
    "competitive-ads-extractor": "science",
    "domain-name-brainstormer": "etc",
    "lead-research-assistant": "science",
    "meeting-insights-analyzer": "science",
    "knowledge-search": "docs",
    "offer-k-dense-web": "skill",
    "brainstorming": "skill",
    "consciousness-council": "skill",
    "sequential-thinking": "skill",
    "problem-solving": "skill",
    "what-if-oracle": "skill",
    "dhdna-profiler": "skill",
    "openalex-database": "data",

    # ── 보정 (패턴이 잘못 잡는 항목) ──
    "agent-fintech-engineer": "code",
    "agent-embedded-systems": "code",
    "agent-iot-engineer": "code",
    "agent-game-developer": "code",
    "agent-accessibility-tester": "design",
    "agent-performance-monitor": "ops",
    "agent-performance-engineer": "ops",
    "cmd-deep-research": "science",
    "cmd-explore": "skill",
    "agent-api-designer": "code",
    "depmap": "science",
    "primekg": "science",
    "google-adk-python": "skill",
}


def _categorize(sid, name, desc):
    """스킬 ID → 대분류 키. 명시 매핑 우선, 다음 패턴 규칙."""
    if sid in _CATEGORY_OVERRIDE:
        return _CATEGORY_OVERRIDE[sid]

    # 패턴: 명확한 그룹
    if sid.startswith("cmd-git-"):
        return "ops"
    if sid.startswith("cmd-cr"):
        return "review"
    if sid.endswith("-database"):
        return "data"
    if sid.startswith("guide-"):
        return "docs"

    # agent-* 패턴 분기
    if sid.startswith("agent-"):
        tail = sid[len("agent-"):]
        # ops 키워드
        if any(k in tail for k in ("devops", "deployment", "cloud", "kubernetes",
                                   "terraform", "sre", "platform", "network",
                                   "incident", "build-", "datadog")):
            return "ops"
        # review/QA/security 키워드
        if any(k in tail for k in ("security", "penetration", "auditor", "debug",
                                   "error", "test", "qa", "chaos", "reviewer",
                                   "refactor", "legacy", "compliance")):
            return "review"
        # data 키워드
        if any(k in tail for k in ("data-", "database", "postgres", "sql")):
            return "data"
        # design 키워드
        if any(k in tail for k in ("ui-", "ux-", "designer")):
            return "design"
        # docs/비즈니스 키워드
        if any(k in tail for k in ("writer", "documenter", "marketer", "scrum-",
                                   "project-manager", "product-manager",
                                   "scrum-master", "customer-", "sales-",
                                   "legal-", "risk-", "fintech",
                                   "seo-", "payment-")):
            return "docs"
        # 개발 직군
        if any(k in tail for k in ("-developer", "-pro", "-expert", "-specialist",
                                   "-architect", "-engineer", "-master")):
            return "code"
        return "etc"

    return "etc"


# ─────────────────────────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static", static_url_path="/static")


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/health")
def health():
    hermes_bin = _find_hermes()
    return jsonify({
        "env_mode": ENV_MODE,
        "scientific_base": SCIENTIFIC_BASE,
        "skills_dir": SKILLS_DIR,
        "skills_dir_exists": os.path.isdir(SKILLS_DIR),
        "proxy_alive": _is_proxy_alive(),
        "proxy_url": f"http://{PROXY_HOST}:{PROXY_PORT}",
        "hermes_binary": hermes_bin,
        "winpty": _USE_WINPTY,
    })


SKILLS_KR_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills_kr.json")


def _load_kr_cache():
    if not os.path.isfile(SKILLS_KR_CACHE):
        return {}
    try:
        with open(SKILLS_KR_CACHE, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _save_kr_cache(d):
    try:
        with open(SKILLS_KR_CACHE, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[kr-cache] 저장 실패: {e}")


@app.route("/api/skills")
def api_skills():
    items = _scan_skills()
    kr = _load_kr_cache()
    # 한글 캐시 머지: description_kr 필드 추가
    for it in items:
        it["description_kr"] = kr.get(it["id"], "")
    return jsonify({
        "count": len(items),
        "skills": items,
        "kr_cached": sum(1 for it in items if it["description_kr"]),
        "categories": SKILL_CATEGORIES,
    })


@app.route("/api/skill/<sid>")
def api_skill_detail(sid):
    """SKILL.md 전체 내용 + frontmatter 메타 반환 (progressive disclosure Tier 2)."""
    # 디렉토리 traversal 방지
    if not sid or "/" in sid or "\\" in sid or sid.startswith("."):
        return jsonify({"error": "invalid skill id"}), 400
    d = os.path.join(SKILLS_DIR, sid)
    skill_md = os.path.join(d, "SKILL.md")
    if not os.path.isfile(skill_md):
        return jsonify({"error": "SKILL.md not found", "id": sid}), 404
    try:
        with open(skill_md, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        return jsonify({"error": f"read fail: {e}"}), 500

    name, desc = None, ""
    m = re.match(r"---\s*\n([\s\S]+?)\n---\s*\n", text)
    body = text
    if m:
        front = m.group(1)
        body = text[m.end():]
        nm = re.search(r"^name:\s*(.+?)$", front, re.MULTILINE)
        dm = re.search(r"^description:\s*(.+?)$", front, re.MULTILINE)
        if nm:
            name = nm.group(1).strip().strip('"').strip("'")
        if dm:
            desc = dm.group(1).strip().strip('"').strip("'")

    # 보조 파일 리스트 (references/*, scripts/*)
    files = []
    for sub in ("references", "scripts", "assets"):
        sd = os.path.join(d, sub)
        if os.path.isdir(sd):
            for fn in sorted(os.listdir(sd)):
                fp = os.path.join(sd, fn)
                if os.path.isfile(fp):
                    files.append(f"{sub}/{fn}")

    return jsonify({
        "id": sid,
        "name": name or sid,
        "description": desc,
        "body": body,
        "files": files,
    })


@app.route("/api/translate-skills", methods=["POST"])
def api_translate_skills():
    """프록시(:8765/v1) 통해 LLM 으로 영문 description → 한글 요약. 배치 처리.
    body: {"batch_size": 10, "only_missing": true}"""
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
            cache[it["id"]] = ""  # 설명 없음 → 빈 캐시
            continue
        todo.append(it)

    if not todo:
        return jsonify({"status": "complete", "translated": 0, "remaining": 0})

    if not _is_proxy_alive():
        return jsonify({"error": "프록시(:8765) 가 안 떠있음"}), 502

    import requests as _req

    # 배치 호출 한 번 (이 요청에서 처리하는 갯수만큼)
    batch = todo[:batch_size]
    prompt_items = "\n".join(
        f"{i+1}. [{it['id']}] {it['description'][:250]}"
        for i, it in enumerate(batch)
    )
    prompt = (
        "다음은 영문 스킬 설명들이다. 각 항목을 한국어로 한 줄 (60자 이내) 요약하라.\n"
        "출력 형식 정확히 지킬 것:\n"
        "1. [id] 한국어 요약\n"
        "2. [id] 한국어 요약\n"
        "...\n\n"
        f"항목:\n{prompt_items}"
    )

    try:
        r = _req.post(
            f"http://{PROXY_HOST}:{PROXY_PORT}/v1/chat/completions",
            json={
                "model": "any",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 2000,
                "stream": False,
            },
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
    except Exception as e:
        return jsonify({"error": f"번역 호출 실패: {e}"}), 500

    # 응답 파싱: "1. [id] 한국어 요약"
    parsed = {}
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"^\d+\.\s*\[([^\]]+)\]\s*(.+)$", line)
        if m:
            sid = m.group(1).strip()
            kr = m.group(2).strip()
            parsed[sid] = kr

    # 배치의 모든 id 에 대해, 파싱된 것 있으면 저장 / 아니면 영문 그대로
    saved = 0
    for it in batch:
        if it["id"] in parsed:
            cache[it["id"]] = parsed[it["id"]]
            saved += 1
        else:
            # 파싱 실패 → 다음번에 다시 시도하도록 캐시 안 함
            pass
    _save_kr_cache(cache)

    return jsonify({
        "status": "ok",
        "translated": saved,
        "batch_size": len(batch),
        "remaining": max(0, len(todo) - len(batch)),
        "total": len(items),
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

    # 입력 에코 매치용: 사용자가 보낸 메시지의 각 비공백 줄을 집합으로
    echo_lines = set()
    for ln in (last_user or "").splitlines():
        t = ln.strip()
        if t:
            echo_lines.add(t)

    def gen():
        info = ("Hermes — 전달된 스킬: " + ", ".join(skill_ids)) \
               if skill_ids else "Hermes (자동 스킬 판단)"
        meta = {
            "type": "meta",
            "env_mode": ENV_MODE,
            "loaded_skills": skill_ids,
            "info": info,
            "session_id_in": session_id or None,
            "raw": raw_mode,
        }
        yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"

        if not _is_proxy_alive():
            err = {
                "type": "error",
                "error": f"프록시(:{PROXY_PORT}) 가 떠있지 않습니다",
                "detail": f"python {os.path.dirname(os.path.abspath(__file__))}/hermes_proxy.py",
            }
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return

        if not last_user.strip():
            yield f"data: {json.dumps({'type':'error','error':'빈 메시지'}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return

        hermes_bin = _find_hermes()
        if not hermes_bin:
            yield f"data: {json.dumps({'type':'error','error':'hermes 명령 못 찾음 (PATH 확인)'}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return

        cmd = [hermes_bin]
        # session_id 가 와도 실제 세션 파일이 없으면 --resume 안 함 (silent skip)
        # → 죽은 sid 들고 있는 브라우저 자동 구제, 새 세션 시작되어 sid 회수
        if session_id:
            sess_dir = os.path.expanduser("~/.hermes/sessions")
            possible_files = [
                os.path.join(sess_dir, f"session_{session_id}.json"),
                os.path.join(sess_dir, f"{session_id}.json"),
            ]
            sess_exists = any(os.path.isfile(p) for p in possible_files)
            if sess_exists:
                cmd += ["--resume", session_id]
                print(f"[chat] resume 가능: {session_id}")
            else:
                print(f"[chat] session_id={session_id} 파일 없음 → 새 세션 시작")
        if skill_ids:
            cmd += ["-s", ",".join(skill_ids)]
        cmd += ["chat", "-q", last_user]

        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("TERM", "xterm-256color")

        print(f"[chat] cmd={cmd}  (winpty={_USE_WINPTY}, raw={raw_mode}, sid_in={session_id})")

        new_sid = None
        line_count = 0
        emit_count = 0
        # 답변 본문이 시작되었는가 — 한 번 시작되면 에코 필터 끔
        body_started = [False]

        def _proc_line(rawline):
            nonlocal new_sid
            line = _strip_ansi(rawline.rstrip("\r\n"))
            # session_id 는 항상 추출 (raw 모드여도 sid 는 필요)
            mm = re.search(r"hermes\s+--resume\s+(\S+)", line)
            if mm:
                new_sid = mm.group(1)
                if not raw_mode:
                    return None  # 메타라인은 일반 모드에서만 숨김

            if raw_mode:
                # 그대로 통과 (디버그용)
                if line:
                    return line + "\n"
                return None

            s = _normalize_for_match(line)
            if not s:
                return None
            # 박스/구분선 (앞에 한 글자가 붙어있어도 매치)
            if _BOXLINE_RE.match(s):
                return None
            # "Q" 단독 prompt-echo
            if _Q_PROMPT_RE.match(s):
                return None
            # NOISE_PREFIXES (정규화 후 prefix 매치)
            if any(s.startswith(p) for p in NOISE_PREFIXES):
                return None
            if "⚕ Hermes" in s:
                return None
            if _META_LINE_RE.search(s):
                return None
            # 명시적 thinking 마커만 필터 (보수적)
            if _THINK_MARKER_RE.match(s):
                return None
            # 입력 에코 — 답변 본문 시작 전까지만 드롭
            if not body_started[0] and s in echo_lines:
                return None
            # 한 줄 답변이 시작된 시점부터는 에코 필터 비활성
            body_started[0] = True
            return s + "\n"

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
                    except EOFError:
                        break
                    except Exception as _re:
                        print(f"[chat] read 예외: {_re}")
                        break
                    if not chunk:
                        if not proc.isalive():
                            break
                        continue
                    buf += chunk
                    while True:
                        idx_rn = buf.find("\r\n")
                        idx_n = buf.find("\n")
                        idx_r = buf.find("\r")
                        cands = [(i, ln) for (i, ln) in
                                 [(idx_rn, 2), (idx_n, 1), (idx_r, 1)] if i != -1]
                        if not cands:
                            break
                        cands.sort()
                        nl_idx, nl_len = cands[0]
                        if idx_rn != -1 and idx_rn == nl_idx:
                            nl_len = 2
                        line_str = buf[:nl_idx]
                        buf = buf[nl_idx + nl_len:]
                        line_count += 1
                        tok = _proc_line(line_str)
                        if tok:
                            yield f"data: {json.dumps({'type':'token','t':tok}, ensure_ascii=False)}\n\n"
                            emit_count += 1
                if buf.strip():
                    line_count += 1
                    tok = _proc_line(buf)
                    if tok:
                        yield f"data: {json.dumps({'type':'token','t':tok}, ensure_ascii=False)}\n\n"
                        emit_count += 1
            finally:
                try:
                    if proc.isalive():
                        proc.terminate(force=True)
                        print("[chat] hermes proc terminated")
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
                    line_count += 1
                    tok = _proc_line(raw_line)
                    if tok:
                        yield f"data: {json.dumps({'type':'token','t':tok}, ensure_ascii=False)}\n\n"
                        emit_count += 1
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

        print(f"[chat] 종료: {line_count} 줄 읽음, {emit_count} 줄 emit, hermes_sid={new_sid}")
        end_evt = {
            "type": "end",
            "finish_reason": "stop",
            "hermes_session_id": new_sid,
            "stats": {"lines": line_count, "emitted": emit_count},
        }
        yield f"data: {json.dumps(end_evt, ensure_ascii=False)}\n\n"
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

    print("=" * 70)
    print(f"  Hermes Web")
    print("=" * 70)
    print(f"  환경 자동 감지: {ENV_MODE}")
    print(f"  Scientific 베이스: {SCIENTIFIC_BASE}")
    print(f"  스킬 디렉토리:  {SKILLS_DIR} (존재: {os.path.isdir(SKILLS_DIR)})")
    print(f"  프록시 살아있나: {_is_proxy_alive()}  ({PROXY_HOST}:{PROXY_PORT})")
    print(f"  Hermes 바이너리: {_find_hermes()}")
    print(f"  pywinpty: {_USE_WINPTY}")
    print(f"  포트: {port}")

    # 참고: GGUF 로드는 hermes_proxy.py 가 담당 (이 server.py 는 UI 만)
    print("-" * 70)
    print(f"  http://localhost:{port}  에서 접속")
    print("=" * 70)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
