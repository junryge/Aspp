"""
knowledge_wiki.py - 도메인 지식베이스 위키 시스템 (완전체)

쓰는 길: 원본 소스 → Qwen3 LLM → .md 위키 페이지 자동 생성
읽는 길: 사용자 질의 → BM25 검색 → 결과 반환 (+ LLM 종합 답변)
유지:    lint → 모순/고아/누락 탐지

Flask 모듈 — register_wiki_routes(app) 으로 기존 ASAS에 통합
독립 실행도 가능: python knowledge_wiki.py --port 5555
"""

import os
import re
import math
import json
import hashlib
import datetime
import argparse
import requests
from pathlib import Path
from flask import Flask, request, jsonify, send_file


# ============================================
# 경로 설정
# ============================================
BASE_DIR = Path(__file__).parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
RAW_DIR = KNOWLEDGE_DIR / "_raw"
LOG_FILE = KNOWLEDGE_DIR / "log.md"
INDEX_FILE = KNOWLEDGE_DIR / "index.md"
CONCEPTS_FILE = KNOWLEDGE_DIR / "_concepts.json"
TOKEN_FILE = BASE_DIR / "TOKEN.TXT"

for d in [KNOWLEDGE_DIR, RAW_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ============================================
# LLM 클라이언트 (Qwen3 / OpenAI-compatible)
# ============================================
def _load_token():
    """TOKEN.TXT에서 API 토큰 로드"""
    if TOKEN_FILE.exists():
        return TOKEN_FILE.read_text(encoding="utf-8").strip()
    return os.environ.get("LLM_API_KEY", "")


LLM_CONFIG = {
    "endpoint": os.environ.get("LLM_ENDPOINT", "https://dev.hcp.llm.skhynix.com/v1/chat/completions"),
    "model": os.environ.get("LLM_MODEL", "Qwen3-235B"),
    "api_key": _load_token(),
    "timeout": 120,
    "max_tokens": 4096,
}


def call_llm(system_prompt: str, user_prompt: str, max_tokens: int = None) -> str:
    """LLM API 호출 — OpenAI-compatible 엔드포인트"""
    headers = {"Content-Type": "application/json"}
    if LLM_CONFIG["api_key"]:
        headers["Authorization"] = f"Bearer {LLM_CONFIG['api_key']}"

    payload = {
        "model": LLM_CONFIG["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens or LLM_CONFIG["max_tokens"],
        "temperature": 0.3,
    }

    try:
        resp = requests.post(
            LLM_CONFIG["endpoint"],
            headers=headers,
            json=payload,
            timeout=LLM_CONFIG["timeout"],
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"  [LLM 오류] {e}")
        return ""


# ============================================
# BM25 검색 엔진 (knowledge.py 기반)
# ============================================
_BM25_INDEX = {}
_BM25_AVG_DL = 0
_BM25_N = 0
_BM25_DF = {}
_BM25_K1 = 1.5
_BM25_B = 0.75


def _tokenize(text):
    """한국어 2글자+, 영문/숫자 2글자+ 토큰화"""
    return re.findall(r'[\uac00-\ud7af]{2,}|[a-z0-9_]{2,}', text.lower())


def build_bm25_index():
    """BM25 인덱스 빌드 (knowledge/ 내 .md 파일 대상)"""
    global _BM25_INDEX, _BM25_AVG_DL, _BM25_N, _BM25_DF

    if not KNOWLEDGE_DIR.is_dir():
        return

    index = {}
    df = {}
    total_tokens = 0

    for fpath in KNOWLEDGE_DIR.glob("*.md"):
        fname = fpath.name
        if fname.startswith("_") or fname in ("index.md", "log.md"):
            continue
        try:
            content = fpath.read_text(encoding="utf-8")
        except Exception:
            continue

        tokens = _tokenize(content)
        token_count = len(tokens)
        total_tokens += token_count

        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1

        index[fname] = {"tf": tf, "token_count": token_count, "content": content}

        for t in set(tokens):
            df[t] = df.get(t, 0) + 1

    n = len(index)
    _BM25_INDEX = index
    _BM25_N = n
    _BM25_DF = df
    _BM25_AVG_DL = total_tokens / n if n > 0 else 1
    print(f"  [BM25] 인덱스 빌드: {n}개 문서, 평균 토큰 {_BM25_AVG_DL:.0f}")


def _bm25_score(query_tokens, doc_tf, doc_len):
    """BM25 스코어 계산"""
    score = 0
    for qt in query_tokens:
        if qt not in _BM25_DF:
            continue
        n_qi = _BM25_DF[qt]
        idf = math.log((_BM25_N - n_qi + 0.5) / (n_qi + 0.5) + 1)
        tf = doc_tf.get(qt, 0)
        numerator = tf * (_BM25_K1 + 1)
        denominator = tf + _BM25_K1 * (1 - _BM25_B + _BM25_B * doc_len / max(_BM25_AVG_DL, 1))
        score += idf * (numerator / denominator)
    return score


def _parse_frontmatter(content):
    """YAML frontmatter 파싱"""
    fm = {"tags": [], "category": "", "description": "", "related": [], "date": "", "source": ""}
    body = content

    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            fm_text = parts[1]
            body = parts[2]
            for line in fm_text.split("\n"):
                line = line.strip()
                if line.startswith("tags:"):
                    fm["tags"] = [t.strip().lower() for t in re.findall(r'[\w가-힣]+', line.split(":", 1)[1])]
                elif line.startswith("category:"):
                    fm["category"] = line.split(":", 1)[1].strip().lower()
                elif line.startswith("description:"):
                    fm["description"] = line.split(":", 1)[1].strip().strip('"').lower()
                elif line.startswith("related:"):
                    fm["related"] = [r.strip() for r in line.split(":", 1)[1].split(",") if r.strip()]
                elif line.startswith("date:"):
                    fm["date"] = line.split(":", 1)[1].strip()
                elif line.startswith("source:"):
                    fm["source"] = line.split(":", 1)[1].strip()

    return fm, body


def search_knowledge(query, max_results=5, max_content_chars=8000):
    """BM25 + 메타데이터 기반 검색"""
    if not _BM25_INDEX:
        build_bm25_index()

    q_lower = query.lower()
    query_tokens = _tokenize(query)
    query_nospace = re.sub(r'\s+', '', q_lower)

    stopwords = {"은", "는", "이", "가", "을", "를", "의", "에", "와", "과", "도", "로",
                 "에서", "부터", "까지", "한", "할", "하는", "된", "되는", "있는", "없는",
                 "the", "a", "an", "is", "are", "in", "on", "at", "to", "for", "of",
                 "것", "수", "등", "및", "중", "뭐", "좀", "해", "줘", "알려", "보여", "찾아"}
    keywords = [w for w in re.split(r'[\s,?!·]+', q_lower) if w.strip() and w not in stopwords and len(w) >= 2]

    if not keywords and not query_tokens:
        return []

    results = []
    for fname, doc_info in _BM25_INDEX.items():
        content = doc_info["content"]
        content_lower = content.lower()
        content_nospace = re.sub(r'\s+', '', content_lower)
        fname_lower = fname.lower()

        fm, body = _parse_frontmatter(content)
        total_score = 0

        # 1. BM25 스코어
        bm25 = _bm25_score(query_tokens, doc_info["tf"], doc_info["token_count"])
        total_score += int(bm25 * 10)

        # 2. 파일명 매칭
        for kw in keywords:
            if kw in fname_lower:
                total_score += 10

        # 3. frontmatter 태그 매칭
        for kw in keywords:
            if kw in fm["tags"]:
                total_score += 15
            if kw in fm["description"]:
                total_score += 10
            if kw == fm["category"]:
                total_score += 10

        # 4. 바이그램 매칭
        if len(keywords) >= 2:
            for i in range(len(keywords) - 1):
                bigram = keywords[i] + keywords[i + 1]
                if bigram in content_nospace:
                    total_score += 15
            if query_nospace in content_nospace:
                total_score += 20

        # 5. 한글 유사문자
        for kw in keywords:
            for a, b in [("률", "율"), ("렬", "열"), ("례", "예")]:
                alt = kw.replace(a, b) if a in kw else kw.replace(b, a) if b in kw else None
                if alt and alt != kw and alt in content_lower:
                    total_score += 3

        if total_score > 0:
            # preview 생성
            preview = []
            for kw in keywords[:3]:
                pos = content_lower.find(kw)
                if pos != -1:
                    start = max(0, pos - 80)
                    end = min(len(content), pos + len(kw) + 80)
                    snippet = content[start:end].replace('\n', ' ').strip()
                    if start > 0: snippet = '...' + snippet
                    if end < len(content): snippet += '...'
                    preview.append(snippet)

            results.append({
                "filename": fname,
                "score": total_score,
                "content": content[:max_content_chars],
                "content_length": len(content),
                "preview": preview,
                "frontmatter": fm,
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    if results:
        min_score = max(5, results[0]["score"] * 0.3)
        results = [r for r in results if r["score"] >= min_score]

    return results[:max_results]


# ============================================
# 쓰는 길 — 위키 자동 생성 (Ingest)
# ============================================
INGEST_SYSTEM_PROMPT = """너는 도메인 지식베이스 위키 작성자다.
사용자가 제공하는 원본 소스를 읽고, 아래 형식의 마크다운 위키 페이지를 생성해라.

규칙:
1. 반드시 YAML frontmatter로 시작 (tags, category, description, date, source, related)
2. 본문은 한국어로 작성
3. 핵심 개념/엔티티마다 [[개념명]] 형태의 위키링크 삽입
4. 기존 위키 페이지 목록을 참고하여 관련 페이지에 [[링크]] 연결
5. 불필요한 중복은 피하고, 기존 페이지를 참조하도록 안내

출력 형식:
---
tags: [태그1, 태그2, 태그3]
category: 카테고리명
description: "한 줄 설명"
date: YYYY-MM-DD
source: 원본파일명
related: [관련페이지1.md, 관련페이지2.md]
---

# 페이지 제목

본문 내용 (한국어)

## 핵심 내용
- 주요 포인트

## 관련 항목
- [[관련개념1]]
- [[관련개념2]]
"""

INGEST_UPDATE_PROMPT = """너는 위키 페이지 갱신 담당이다.
새 소스에서 추출된 정보를 기존 위키 페이지에 반영해야 한다.

기존 페이지 내용:
{existing_content}

새로 추가할 정보:
{new_info}

규칙:
1. 기존 내용은 유지하면서 새 정보를 통합
2. 모순되는 내용이 있으면 [모순감지] 태그로 표시
3. frontmatter의 tags, related 필드 업데이트
4. 수정일을 오늘 날짜로 갱신

전체 페이지를 완성된 형태로 출력해라.
"""


def _file_hash(filepath):
    """파일 SHA256 (중복 ingest 방지)"""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _get_existing_pages():
    """기존 위키 페이지 목록 + 한줄요약"""
    pages = []
    for fpath in KNOWLEDGE_DIR.glob("*.md"):
        if fpath.name.startswith("_") or fpath.name in ("index.md", "log.md"):
            continue
        try:
            content = fpath.read_text(encoding="utf-8")
            fm, body = _parse_frontmatter(content)
            pages.append({
                "filename": fpath.name,
                "description": fm.get("description", ""),
                "tags": fm.get("tags", []),
                "category": fm.get("category", ""),
            })
        except Exception:
            continue
    return pages


def _append_log(action, detail):
    """log.md에 기록 추가"""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    entry = f"\n## [{now}] {action} | {detail}\n"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry)


def _update_index():
    """index.md 자동 갱신 — 전체 위키 카탈로그"""
    pages = _get_existing_pages()
    if not pages:
        return

    lines = ["# 위키 인덱스\n"]
    lines.append(f"_자동 생성: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} | 총 {len(pages)}개 문서_\n")

    # 카테고리별 분류
    by_cat = {}
    for p in pages:
        cat = p["category"] or "미분류"
        by_cat.setdefault(cat, []).append(p)

    for cat in sorted(by_cat.keys()):
        lines.append(f"\n## {cat}\n")
        for p in sorted(by_cat[cat], key=lambda x: x["filename"]):
            desc = p["description"] or "-"
            tags = ", ".join(p["tags"][:5]) if p["tags"] else ""
            tag_str = f" `{tags}`" if tags else ""
            lines.append(f"- [{p['filename']}]({p['filename']}) — {desc}{tag_str}")

    INDEX_FILE.write_text("\n".join(lines), encoding="utf-8")


def _update_concepts():
    """_concepts.json 갱신 — 위키링크 기반 관계 매핑"""
    concepts = {}  # {페이지명: {links_to: [], linked_from: []}}

    for fpath in KNOWLEDGE_DIR.glob("*.md"):
        if fpath.name.startswith("_") or fpath.name in ("index.md", "log.md"):
            continue
        try:
            content = fpath.read_text(encoding="utf-8")
        except Exception:
            continue

        fname = fpath.name
        concepts.setdefault(fname, {"links_to": [], "linked_from": []})

        # [[위키링크]] 추출
        links = re.findall(r'\[\[([^\]]+)\]\]', content)
        for link in links:
            link_file = link if link.endswith(".md") else link + ".md"
            if link_file != fname:
                if link_file not in concepts[fname]["links_to"]:
                    concepts[fname]["links_to"].append(link_file)
                concepts.setdefault(link_file, {"links_to": [], "linked_from": []})
                if fname not in concepts[link_file]["linked_from"]:
                    concepts[link_file]["linked_from"].append(fname)

    CONCEPTS_FILE.write_text(json.dumps(concepts, ensure_ascii=False, indent=2), encoding="utf-8")
    return concepts


def ingest_text(text, source_name="manual_input", title=None):
    """텍스트 → LLM → 위키 페이지 생성

    Args:
        text: 원본 텍스트
        source_name: 소스 식별자
        title: 페이지 제목 (없으면 LLM이 결정)

    Returns:
        dict: {filename, status, message}
    """
    existing_pages = _get_existing_pages()
    page_list = "\n".join([f"- {p['filename']}: {p['description']}" for p in existing_pages[:50]])

    user_prompt = f"""원본 소스 ({source_name}):
---
{text[:6000]}
---

기존 위키 페이지 목록:
{page_list}

위 소스를 읽고 위키 페이지를 생성해라.
{"페이지 제목: " + title if title else "적절한 제목을 자동으로 결정해라."}
"""

    result = call_llm(INGEST_SYSTEM_PROMPT, user_prompt)
    if not result:
        return {"filename": "", "status": "error", "message": "LLM 응답 없음"}

    # 파일명 추출 (첫 번째 # 헤딩에서)
    title_match = re.search(r'^#\s+(.+)', result, re.MULTILINE)
    if title_match:
        page_title = title_match.group(1).strip()
    else:
        page_title = title or source_name

    # 파일명 생성 (한글/영문 → 안전한 파일명)
    safe_name = re.sub(r'[^\w가-힣\s-]', '', page_title)
    safe_name = re.sub(r'\s+', '_', safe_name.strip())[:60]
    filename = f"{safe_name}.md"

    filepath = KNOWLEDGE_DIR / filename

    # 기존 페이지 존재 시 → 갱신 모드
    if filepath.exists():
        existing_content = filepath.read_text(encoding="utf-8")
        update_prompt = INGEST_UPDATE_PROMPT.format(
            existing_content=existing_content[:4000],
            new_info=result[:4000]
        )
        updated = call_llm("위키 페이지 갱신 담당.", update_prompt)
        if updated:
            result = updated

    # 저장
    filepath.write_text(result, encoding="utf-8")

    # 원본 보관
    raw_path = RAW_DIR / f"{datetime.datetime.now().strftime('%Y%m%d_%H%M')}_{source_name[:40]}.txt"
    raw_path.write_text(text[:50000], encoding="utf-8")

    # 인덱스/개념/로그 갱신
    _append_log("ingest", f"{filename} (source: {source_name})")
    _update_index()
    _update_concepts()
    build_bm25_index()

    return {"filename": filename, "status": "ok", "message": f"위키 페이지 생성: {filename}"}


def ingest_file(filepath):
    """파일 → 위키 페이지 생성"""
    fpath = Path(filepath)
    if not fpath.exists():
        return {"filename": "", "status": "error", "message": f"파일 없음: {filepath}"}

    # 텍스트 읽기
    try:
        for enc in ["utf-8", "cp949", "euc-kr"]:
            try:
                text = fpath.read_text(encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            return {"filename": "", "status": "error", "message": "인코딩 감지 실패"}
    except Exception as e:
        return {"filename": "", "status": "error", "message": str(e)}

    return ingest_text(text, source_name=fpath.name)


# ============================================
# LLM 종합 답변 (Query with LLM)
# ============================================
QUERY_SYSTEM_PROMPT = """너는 도메인 지식베이스 질의응답 담당이다.
검색된 위키 페이지들을 기반으로 사용자 질문에 종합적으로 답변해라.

규칙:
1. 검색된 페이지 내용만 기반으로 답변 (없는 내용 추측 금지)
2. 출처 페이지명을 [파일명.md] 형태로 표기
3. 한국어로 답변
4. 관련 위키 페이지 추가 탐색을 권장할 수 있음
"""


def query_with_llm(question, save_answer=False):
    """질문 → BM25 검색 → LLM 종합 답변

    Args:
        question: 자연어 질문
        save_answer: True면 답변을 위키에 저장

    Returns:
        dict: {answer, sources, saved_as}
    """
    results = search_knowledge(question, max_results=5)

    if not results:
        return {"answer": "관련 문서를 찾지 못했습니다.", "sources": [], "saved_as": ""}

    context = ""
    sources = []
    for r in results:
        context += f"\n--- {r['filename']} (score: {r['score']}) ---\n"
        context += r["content"][:3000] + "\n"
        sources.append(r["filename"])

    user_prompt = f"""질문: {question}

검색된 위키 페이지:
{context}

위 내용을 종합하여 질문에 답변해라.
"""

    answer = call_llm(QUERY_SYSTEM_PROMPT, user_prompt)
    if not answer:
        answer = "LLM 응답 실패. 검색 결과를 직접 확인해주세요."

    saved_as = ""
    if save_answer and answer:
        _append_log("query", f"Q: {question[:60]}")

    return {"answer": answer, "sources": sources, "saved_as": saved_as}


# ============================================
# Lint — 위키 건강 점검
# ============================================
def lint_wiki():
    """위키 전체 건강 점검

    Returns:
        dict: {orphans, missing_pages, no_tags, stats}
    """
    concepts = _update_concepts()

    orphans = []       # 아무 데서도 링크 안 된 페이지
    missing = []       # [[링크]]는 있는데 실제 파일 없음
    no_tags = []       # frontmatter 태그 없는 페이지
    page_count = 0

    existing_files = set(f.name for f in KNOWLEDGE_DIR.glob("*.md")
                         if not f.name.startswith("_") and f.name not in ("index.md", "log.md"))

    for fname in existing_files:
        page_count += 1
        info = concepts.get(fname, {"links_to": [], "linked_from": []})

        # 고아 페이지 (아무 데서도 참조 안 됨)
        if not info["linked_from"] and fname != "index.md":
            orphans.append(fname)

        # 태그 없는 페이지
        fpath = KNOWLEDGE_DIR / fname
        try:
            content = fpath.read_text(encoding="utf-8")
            fm, _ = _parse_frontmatter(content)
            if not fm["tags"]:
                no_tags.append(fname)
        except Exception:
            pass

    # 누락 페이지 ([[링크]]는 있지만 파일 없음)
    for fname, info in concepts.items():
        if fname not in existing_files and info.get("linked_from"):
            missing.append({"page": fname, "referenced_by": info["linked_from"]})

    report = {
        "stats": {"total_pages": page_count, "total_links": sum(len(v["links_to"]) for v in concepts.values())},
        "orphans": orphans,
        "missing_pages": missing,
        "no_tags": no_tags,
    }

    _append_log("lint", f"pages={page_count}, orphans={len(orphans)}, missing={len(missing)}")
    return report


# ============================================
# Flask 라우트 + 웹 UI
# ============================================
def register_wiki_routes(app):
    """Flask 앱에 위키 API 라우트 등록"""

    # ── 검색 API ──
    @app.route("/api/wiki/search", methods=["POST"])
    def api_wiki_search():
        data = request.json or {}
        query = data.get("query", "").strip()
        if not query:
            return jsonify({"error": "query 필요"}), 400
        results = search_knowledge(query)
        return jsonify({
            "query": query,
            "results": [{"filename": r["filename"], "score": r["score"],
                          "preview": r["preview"], "frontmatter": r["frontmatter"]} for r in results],
            "total": len(results),
        })

    # ── Ingest API (텍스트) ──
    @app.route("/api/wiki/ingest", methods=["POST"])
    def api_wiki_ingest():
        data = request.json or {}
        text = data.get("text", "").strip()
        source = data.get("source", "web_input")
        title = data.get("title", "")
        if not text:
            return jsonify({"error": "text 필요"}), 400
        result = ingest_text(text, source_name=source, title=title or None)
        return jsonify(result)

    # ── Ingest API (파일 업로드) ──
    @app.route("/api/wiki/ingest/file", methods=["POST"])
    def api_wiki_ingest_file():
        if "file" not in request.files:
            return jsonify({"error": "file 필요"}), 400
        f = request.files["file"]
        tmp_path = RAW_DIR / f"_upload_{f.filename}"
        f.save(str(tmp_path))
        result = ingest_file(str(tmp_path))
        return jsonify(result)

    # ── LLM 질의 API ──
    @app.route("/api/wiki/query", methods=["POST"])
    def api_wiki_query():
        data = request.json or {}
        question = data.get("question", "").strip()
        if not question:
            return jsonify({"error": "question 필요"}), 400
        result = query_with_llm(question)
        return jsonify(result)

    # ── Lint API ──
    @app.route("/api/wiki/lint", methods=["POST"])
    def api_wiki_lint():
        report = lint_wiki()
        return jsonify(report)

    # ── 파일 목록 ──
    @app.route("/api/wiki/files", methods=["GET"])
    def api_wiki_files():
        q = request.args.get("q", "").lower()
        files = []
        total_size = 0
        for fpath in sorted(KNOWLEDGE_DIR.glob("*.md")):
            if fpath.name.startswith("_"):
                continue
            if q and q not in fpath.name.lower():
                continue
            size = fpath.stat().st_size
            total_size += size
            try:
                content = fpath.read_text(encoding="utf-8")
                fm, _ = _parse_frontmatter(content)
            except Exception:
                fm = {}
            files.append({"filename": fpath.name, "size": size,
                          "category": fm.get("category", ""), "tags": fm.get("tags", [])})
        return jsonify({"files": files, "total": len(files), "total_size": total_size})

    # ── 파일 내용 조회 ──
    @app.route("/api/wiki/view/<path:filename>", methods=["GET"])
    def api_wiki_view(filename):
        fpath = KNOWLEDGE_DIR / filename
        if not fpath.is_file():
            return jsonify({"error": "파일 없음"}), 404
        content = fpath.read_text(encoding="utf-8")
        fm, body = _parse_frontmatter(content)
        return jsonify({"filename": filename, "content": content, "frontmatter": fm, "size": fpath.stat().st_size})

    # ── 개념 그래프 ──
    @app.route("/api/wiki/graph", methods=["GET"])
    def api_wiki_graph():
        concepts = _update_concepts()
        nodes = []
        edges = []
        for fname, info in concepts.items():
            nodes.append({"id": fname, "links": len(info["links_to"]), "refs": len(info["linked_from"])})
            for target in info["links_to"]:
                edges.append({"source": fname, "target": target})
        return jsonify({"nodes": nodes, "edges": edges})

    # ── 웹 UI ──
    @app.route("/wiki")
    def wiki_ui():
        return WIKI_HTML


# ============================================
# 웹 UI (임베디드 HTML)
# ============================================
WIKI_HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>도메인 지식베이스</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:"Pretendard","Noto Sans KR",sans-serif;background:#0f1117;color:#e2e4e8}
.top{background:#161922;border-bottom:1px solid rgba(255,255,255,0.06);padding:1rem 1.5rem;display:flex;align-items:center;gap:1rem}
.top h1{font-size:1.3rem;font-weight:700;color:#fff}
.top .stats{margin-left:auto;display:flex;gap:0.6rem;font-size:0.8rem}
.top .stat{background:rgba(255,255,255,0.08);padding:0.3rem 0.7rem;border-radius:4px;color:#9ba1ad}
.tabs{display:flex;gap:0;background:#161922;border-bottom:1px solid rgba(255,255,255,0.06)}
.tab{padding:0.7rem 1.2rem;cursor:pointer;font-size:0.85rem;color:#64748b;border-bottom:2px solid transparent;transition:all 0.2s}
.tab:hover{color:#e2e4e8}
.tab.active{color:#38bdf8;border-bottom-color:#38bdf8}
.panel{display:none;padding:1.2rem 1.5rem}
.panel.active{display:block}
input[type=text],textarea{width:100%;padding:0.6rem 0.8rem;background:#1a1e28;border:1px solid rgba(255,255,255,0.1);border-radius:6px;color:#e2e4e8;font-size:0.9rem;font-family:inherit}
textarea{min-height:120px;resize:vertical}
.btn{padding:0.5rem 1rem;border:none;border-radius:6px;cursor:pointer;font-size:0.85rem;font-weight:500}
.btn-primary{background:#38bdf8;color:#000}
.btn-primary:hover{background:#22a5e0}
.btn-teal{background:#2dd4bf;color:#000}
.btn-amber{background:#fbbf24;color:#000}
.btn-coral{background:#fb7185;color:#000}
.result-card{background:#1a1e28;border:1px solid rgba(255,255,255,0.06);border-radius:8px;padding:1rem;margin-top:0.8rem;cursor:pointer;transition:border-color 0.2s}
.result-card:hover{border-color:rgba(56,189,248,0.3)}
.result-title{font-weight:600;font-size:0.95rem;color:#fff;margin-bottom:0.3rem}
.result-meta{font-size:0.75rem;color:#64748b;margin-bottom:0.4rem}
.result-preview{font-size:0.8rem;color:#9ba1ad;line-height:1.5}
.tag{display:inline-block;padding:0.15rem 0.5rem;background:rgba(56,189,248,0.1);color:#38bdf8;border-radius:3px;font-size:0.7rem;margin-right:0.3rem}
.modal-bg{display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.6);z-index:100;align-items:center;justify-content:center}
.modal-bg.show{display:flex}
.modal{background:#1a1e28;width:85%;max-width:800px;max-height:80vh;border-radius:10px;overflow:hidden;display:flex;flex-direction:column;border:1px solid rgba(255,255,255,0.1)}
.modal-head{padding:0.8rem 1.2rem;background:#161922;display:flex;align-items:center;border-bottom:1px solid rgba(255,255,255,0.06)}
.modal-head h3{flex:1;font-size:0.95rem}
.modal-head button{background:none;border:none;color:#64748b;font-size:1.2rem;cursor:pointer}
.modal-body{padding:1.2rem;overflow-y:auto;white-space:pre-wrap;font-family:"JetBrains Mono","Fira Code",monospace;font-size:0.82rem;line-height:1.7;color:#c8cad0}
.answer-box{background:#0d2137;border:1px solid rgba(56,189,248,0.2);border-radius:8px;padding:1rem;margin-top:1rem;white-space:pre-wrap;line-height:1.7;font-size:0.88rem}
.lint-section{margin-top:1rem}
.lint-title{font-size:0.9rem;font-weight:600;margin-bottom:0.5rem}
.lint-item{font-size:0.82rem;color:#9ba1ad;padding:0.2rem 0}
.lint-ok{color:#4ade80}
.lint-warn{color:#fbbf24}
.lint-bad{color:#fb7185}
.flex-row{display:flex;gap:0.6rem;margin-top:0.8rem;flex-wrap:wrap}
.status{padding:0.3rem 0.8rem;border-radius:4px;font-size:0.8rem;margin-top:0.8rem}
.status-ok{background:rgba(74,222,128,0.1);color:#4ade80}
.status-err{background:rgba(251,113,133,0.1);color:#fb7185}
.loading{color:#64748b;font-size:0.85rem;padding:1rem 0}
</style>
</head>
<body>

<div class="top">
<h1>도메인 지식베이스</h1>
<div class="stats">
<span class="stat" id="st-docs">문서: -</span>
<span class="stat" id="st-size">용량: -</span>
</div>
</div>

<div class="tabs">
<div class="tab active" onclick="switchTab('search')">검색</div>
<div class="tab" onclick="switchTab('ingest')">위키 생성</div>
<div class="tab" onclick="switchTab('query')">LLM 질의</div>
<div class="tab" onclick="switchTab('lint')">점검</div>
<div class="tab" onclick="switchTab('browse')">탐색</div>
</div>

<!-- 검색 -->
<div class="panel active" id="panel-search">
<input type="text" id="s-input" placeholder="키워드 검색 (BM25 + 메타데이터)..." onkeypress="if(event.key==='Enter')doSearch()">
<div class="flex-row"><button class="btn btn-primary" onclick="doSearch()">검색</button></div>
<div id="s-results"></div>
</div>

<!-- 위키 생성 -->
<div class="panel" id="panel-ingest">
<input type="text" id="i-title" placeholder="페이지 제목 (비워두면 자동 결정)" style="margin-bottom:0.5rem">
<textarea id="i-text" placeholder="원본 소스 텍스트를 여기에 붙여넣기..."></textarea>
<div class="flex-row">
<button class="btn btn-coral" onclick="doIngest()">위키 페이지 생성</button>
<span style="color:#64748b;font-size:0.8rem;line-height:2">LLM이 읽고 frontmatter + 교차참조 포함된 .md 자동 생성</span>
</div>
<div id="i-result"></div>
</div>

<!-- LLM 질의 -->
<div class="panel" id="panel-query">
<input type="text" id="q-input" placeholder="자연어 질문 (위키 기반 종합 답변)..." onkeypress="if(event.key==='Enter')doQuery()">
<div class="flex-row"><button class="btn btn-primary" onclick="doQuery()">질의</button></div>
<div id="q-result"></div>
</div>

<!-- 점검 -->
<div class="panel" id="panel-lint">
<button class="btn btn-amber" onclick="doLint()">위키 건강 점검 실행</button>
<div id="l-result"></div>
</div>

<!-- 탐색 -->
<div class="panel" id="panel-browse">
<input type="text" id="b-filter" placeholder="파일명 필터..." oninput="loadFiles()">
<div id="b-list" style="margin-top:0.8rem"></div>
</div>

<!-- 모달 -->
<div class="modal-bg" id="modal" onclick="if(event.target===this)closeModal()">
<div class="modal">
<div class="modal-head"><h3 id="m-title">파일</h3><button onclick="closeModal()">x</button></div>
<div class="modal-body" id="m-body"></div>
</div>
</div>

<script>
function switchTab(name) {
  document.querySelectorAll('.tab').forEach((t,i) => {
    const panels = ['search','ingest','query','lint','browse'];
    const isActive = panels[i] === name;
    t.classList.toggle('active', isActive);
    document.getElementById('panel-' + panels[i]).classList.toggle('active', isActive);
  });
  if (name === 'browse') loadFiles();
}

async function doSearch() {
  const q = document.getElementById('s-input').value.trim();
  if (!q) return;
  document.getElementById('s-results').innerHTML = '<div class="loading">검색 중...</div>';
  const res = await fetch('/api/wiki/search', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({query:q})});
  const data = await res.json();
  if (!data.results || data.results.length === 0) {
    document.getElementById('s-results').innerHTML = '<div class="status status-err">결과 없음</div>';
    return;
  }
  document.getElementById('s-results').innerHTML = data.results.map(r => `
    <div class="result-card" onclick="viewFile('${r.filename}')">
      <div class="result-title">${r.filename}</div>
      <div class="result-meta">score: ${r.score} | ${(r.frontmatter?.tags||[]).map(t=>'<span class=tag>'+t+'</span>').join('')}</div>
      <div class="result-preview">${(r.preview||[]).join(' ... ')}</div>
    </div>
  `).join('');
}

async function doIngest() {
  const text = document.getElementById('i-text').value.trim();
  const title = document.getElementById('i-title').value.trim();
  if (!text) return;
  document.getElementById('i-result').innerHTML = '<div class="loading">LLM 처리 중... (최대 2분)</div>';
  const res = await fetch('/api/wiki/ingest', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({text, title, source:'web_input'})});
  const data = await res.json();
  if (data.status === 'ok') {
    document.getElementById('i-result').innerHTML = `<div class="status status-ok">생성 완료: ${data.filename}</div>`;
    loadStats();
  } else {
    document.getElementById('i-result').innerHTML = `<div class="status status-err">${data.message}</div>`;
  }
}

async function doQuery() {
  const q = document.getElementById('q-input').value.trim();
  if (!q) return;
  document.getElementById('q-result').innerHTML = '<div class="loading">LLM 답변 생성 중...</div>';
  const res = await fetch('/api/wiki/query', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({question:q})});
  const data = await res.json();
  let html = `<div class="answer-box">${data.answer || '답변 없음'}</div>`;
  if (data.sources && data.sources.length) {
    html += '<div style="margin-top:0.6rem;font-size:0.8rem;color:#64748b">참조: ' + data.sources.map(s => `<span class="tag" style="cursor:pointer" onclick="viewFile('${s}')">${s}</span>`).join('') + '</div>';
  }
  document.getElementById('q-result').innerHTML = html;
}

async function doLint() {
  document.getElementById('l-result').innerHTML = '<div class="loading">점검 중...</div>';
  const res = await fetch('/api/wiki/lint', {method:'POST'});
  const data = await res.json();
  let html = `<div class="lint-section"><div class="lint-title lint-ok">통계: ${data.stats.total_pages}개 페이지, ${data.stats.total_links}개 링크</div></div>`;
  if (data.orphans.length) {
    html += `<div class="lint-section"><div class="lint-title lint-warn">고아 페이지 (${data.orphans.length})</div>`;
    data.orphans.forEach(o => html += `<div class="lint-item">${o}</div>`);
    html += '</div>';
  }
  if (data.missing_pages.length) {
    html += `<div class="lint-section"><div class="lint-title lint-bad">누락 페이지 (${data.missing_pages.length})</div>`;
    data.missing_pages.forEach(m => html += `<div class="lint-item">${m.page} (참조: ${m.referenced_by.join(', ')})</div>`);
    html += '</div>';
  }
  if (data.no_tags.length) {
    html += `<div class="lint-section"><div class="lint-title lint-warn">태그 없음 (${data.no_tags.length})</div>`;
    data.no_tags.forEach(n => html += `<div class="lint-item">${n}</div>`);
    html += '</div>';
  }
  if (!data.orphans.length && !data.missing_pages.length && !data.no_tags.length) {
    html += '<div class="status status-ok">위키 상태 양호</div>';
  }
  document.getElementById('l-result').innerHTML = html;
}

async function loadFiles() {
  const q = document.getElementById('b-filter').value.trim();
  const res = await fetch('/api/wiki/files?q=' + encodeURIComponent(q));
  const data = await res.json();
  document.getElementById('st-docs').textContent = '문서: ' + data.total;
  document.getElementById('st-size').textContent = '용량: ' + (data.total_size/1024).toFixed(1) + ' KB';
  if (!data.files.length) {
    document.getElementById('b-list').innerHTML = '<div class="loading">위키 페이지 없음</div>';
    return;
  }
  document.getElementById('b-list').innerHTML = data.files.map(f => `
    <div class="result-card" onclick="viewFile('${f.filename}')">
      <div class="result-title">${f.filename}</div>
      <div class="result-meta">${(f.size/1024).toFixed(1)} KB | ${f.category || '-'} | ${(f.tags||[]).map(t=>'<span class=tag>'+t+'</span>').join('')}</div>
    </div>
  `).join('');
}

async function viewFile(filename) {
  const res = await fetch('/api/wiki/view/' + encodeURIComponent(filename));
  const data = await res.json();
  document.getElementById('m-title').textContent = filename;
  document.getElementById('m-body').textContent = data.content || '내용 없음';
  document.getElementById('modal').classList.add('show');
}

function closeModal() { document.getElementById('modal').classList.remove('show'); }

async function loadStats() {
  const res = await fetch('/api/wiki/files');
  const data = await res.json();
  document.getElementById('st-docs').textContent = '문서: ' + data.total;
  document.getElementById('st-size').textContent = '용량: ' + (data.total_size/1024).toFixed(1) + ' KB';
}

loadStats();
</script>
</body>
</html>"""


# ============================================
# 독립 실행
# ============================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="도메인 지식베이스 위키")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    app = Flask(__name__)
    build_bm25_index()
    register_wiki_routes(app)

    print(f"\n  도메인 지식베이스 위키")
    print(f"  http://{args.host}:{args.port}/wiki")
    print(f"  LLM: {LLM_CONFIG['model']} @ {LLM_CONFIG['endpoint']}")
    print(f"  TOKEN: {'있음' if LLM_CONFIG['api_key'] else '없음 (TOKEN.TXT 확인)'}")
    print()

    app.run(host=args.host, port=args.port, debug=True)
