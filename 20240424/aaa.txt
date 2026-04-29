"""
demos_v1/knowledge.py - Domain knowledge search (BM25), triggers, API routes, and exploration page
"""
import os
import re
import math
from flask import request, jsonify, send_file
from demos_v1.utils import BASE_DIR

# ============================================
# 도메인 지식 검색 API (BM25)
# ============================================
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")

# ── BM25 인덱스 (서버 시작 시 1회 빌드) ──
_BM25_INDEX = {}  # {filename: {"tokens": [...], "token_count": int, "content": str}}
_BM25_AVG_DL = 0  # 전체 문서 평균 토큰 수
_BM25_N = 0       # 전체 문서 수
_BM25_DF = {}     # {token: 등장 문서 수}
_BM25_K1 = 1.5
_BM25_B = 0.75


def _tokenize(text):
    """한국어 2글자+, 영문/숫자 2글자+ 단위로 분리"""
    return re.findall(r'[\uac00-\ud7af]{2,}|[a-z0-9_]{2,}', text.lower())


def _build_bm25_index(search_dir=None):
    """BM25 인덱스 빌드 (서버 시작 시 또는 갱신 시 호출)"""
    global _BM25_INDEX, _BM25_AVG_DL, _BM25_N, _BM25_DF
    target_dir = search_dir or KNOWLEDGE_DIR
    if not os.path.isdir(target_dir):
        return

    index = {}
    df = {}
    total_tokens = 0

    for fname in os.listdir(target_dir):
        if not fname.endswith('.md'):
            continue
        fpath = os.path.join(target_dir, fname)
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception:
            continue

        tokens = _tokenize(content)
        token_count = len(tokens)
        total_tokens += token_count

        # 문서별 토큰 빈도
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1

        try:
            mtime = os.path.getmtime(fpath)
        except OSError:
            mtime = 0
        index[fname] = {"tokens": tf, "token_count": token_count, "content": content, "mtime": mtime}

        # DF 계산 (각 토큰이 등장하는 문서 수)
        for t in set(tokens):
            df[t] = df.get(t, 0) + 1

    n = len(index)
    _BM25_INDEX = index
    _BM25_N = n
    _BM25_DF = df
    _BM25_AVG_DL = total_tokens / n if n > 0 else 1
    print(f"  [BM25] 인덱스 빌드 완료: {n}개 문서, 평균 토큰 {_BM25_AVG_DL:.0f}")


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


# ── 서버 시작 시 인덱스 빌드 ──
_build_bm25_index()


def _get_user_knowledge_dir(user_id=None):
    """사용자별 지식 폴더 경로 반환"""
    if user_id:
        udir = os.path.join(KNOWLEDGE_DIR, user_id)
        os.makedirs(udir, exist_ok=True)
        return udir
    return KNOWLEDGE_DIR


def _build_knowledge_triggers():
    """knowledge/ 전체 하위 폴더에서 트리거 키워드 추출"""
    static_triggers = [
        "아키텍처", "architecture", "허브룸", "hubroom", "아틀라스", "atlas",
        "데드락", "deadlock", "급증", "amhs", "산출물", "컬럼", "접속정보",
        "통신방식", "예측모델", "변경사항", "변경상황", "도메인", "지식",
    ]
    dynamic_triggers = set()
    if os.path.isdir(KNOWLEDGE_DIR):
        for root, dirs, files in os.walk(KNOWLEDGE_DIR):
            for fname in files:
                if not fname.lower().endswith('.md'):
                    continue
                name = fname.rsplit('.', 1)[0]
                parts = re.split(r'[_\-\s]+', name)
                for p in parts:
                    p_lower = p.lower()
                    if re.match(r'^\d{8}$', p_lower):
                        continue
                    if p_lower in ('fab', 'md') or len(p_lower) < 2:
                        continue
                    dynamic_triggers.add(p_lower)
    return list(set(static_triggers) | dynamic_triggers)


KNOWLEDGE_TRIGGERS = _build_knowledge_triggers()


def search_knowledge(query, max_results=5, max_content_chars=8000, user_id=None):
    """사용자별 knowledge 폴더에서 BM25 + 메타데이터 기반 검색

    BM25 업그레이드 (2026-04-12):
    - IDF: 희귀 키워드일수록 높은 가중치
    - 문서 길이 보정: 긴 문서 패널티
    - k1=1.5, b=0.75
    """
    # 사용자별 폴더 검색 (user_id 없으면 전체 검색)
    search_dir = os.path.join(KNOWLEDGE_DIR, user_id) if user_id else KNOWLEDGE_DIR
    if not os.path.isdir(search_dir):
        return []

    # 사용자별 폴더: 인덱스 캐시 (파일 추가/삭제 또는 수정시간 변경 시 재빌드)
    _current_files = {}
    for f in os.listdir(search_dir):
        if f.endswith('.md'):
            try:
                _current_files[f] = os.path.getmtime(os.path.join(search_dir, f))
            except OSError:
                _current_files[f] = 0
    _indexed_files = {f: info.get('mtime', 0) for f, info in _BM25_INDEX.items()}
    if _current_files != _indexed_files:
        _build_bm25_index(search_dir)

    q_lower = query.lower()

    # ── 1단계: 컬럼명 패턴 감지 ──
    column_pattern = re.findall(r'([A-Za-z0-9]+(?:\.[A-Za-z0-9_]+){2,})', query)
    fab_prefixes = set()
    full_columns = []
    for col in column_pattern:
        parts = col.split('.')
        fab = parts[0].upper()
        fab_prefixes.add(fab)
        full_columns.append(col.lower())

    # ── 1.5단계: 자연어 날짜 → YYYYMMDD 변환 ──
    from datetime import datetime, timedelta
    today = datetime.now()
    date_keywords = []

    if "오늘" in q_lower:
        date_keywords.append(today.strftime("%Y%m%d"))
    if "어제" in q_lower:
        date_keywords.append((today - timedelta(days=1)).strftime("%Y%m%d"))

    date_patterns = re.findall(r'(\d{1,2})\s*월\s*(\d{1,2})\s*일', query)
    for month, day in date_patterns:
        try:
            dt = datetime(today.year, int(month), int(day))
            date_keywords.append(dt.strftime("%Y%m%d"))
        except ValueError:
            pass

    # 키워드 추출
    stopwords = {"은", "는", "이", "가", "을", "를", "의", "에", "와", "과", "도", "로", "으로",
                 "에서", "부터", "까지", "한", "할", "하는", "된", "되는", "있는", "없는",
                 "the", "a", "an", "is", "are", "in", "on", "at", "to", "for", "of", "and", "or",
                 "것", "수", "등", "및", "중", "뭐", "좀", "해", "줘", "알려", "보여", "찾아",
                 "que", "all", "stk", "lft", "pdt", "sfab"}
    raw_keywords = [w.strip() for w in re.split(r'[\s,?!·]+', q_lower) if w.strip()]
    keywords = []
    for w in raw_keywords:
        if '.' in w and re.match(r'[a-z0-9]+\.[a-z0-9_.]+', w):
            continue
        keywords.append(w)
    keywords = [w for w in keywords if w not in stopwords and len(w) >= 1]
    for fab in fab_prefixes:
        fab_l = fab.lower()
        if fab_l not in keywords:
            keywords.append(fab_l)
    for dk in date_keywords:
        if dk not in keywords:
            keywords.append(dk)

    query_nospace = re.sub(r'\s+', '', q_lower)
    query_tokens = _tokenize(query)

    if not keywords and not full_columns and not query_tokens:
        return []

    results = []
    for fname in os.listdir(search_dir):
        if not fname.endswith('.md'):
            continue
        fpath = os.path.join(search_dir, fname)
        fname_lower = fname.lower()
        fname_nospace = re.sub(r'[_\-\s.]', '', fname_lower)

        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception:
            continue

        # ── 프론트매터 파싱 ──
        fm_tags = []
        fm_fabs = []
        fm_category = ""
        fm_desc = ""
        fm_text = ""
        body = content
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                fm_text = parts[1]
                body = parts[2]
                for line in fm_text.split('\n'):
                    line = line.strip()
                    if line.startswith('tags:'):
                        fm_tags = [t.strip().lower() for t in re.findall(r'[\w가-힣]+', line.split(':', 1)[1])]
                    elif line.startswith('fab:'):
                        fm_fabs = [f.strip().lower() for f in re.findall(r'[\w]+', line.split(':', 1)[1])]
                    elif line.startswith('category:'):
                        fm_category = line.split(':', 1)[1].strip().lower()
                    elif line.startswith('description:'):
                        fm_desc = line.split(':', 1)[1].strip().strip('"').lower()

        content_lower = content.lower()
        content_nospace = re.sub(r'\s+', '', content_lower)

        total_score = 0

        # ── 프론트매터 날짜 매칭 ──
        fm_date = ""
        if content.startswith('---'):
            for line in fm_text.split('\n'):
                line = line.strip()
                if line.startswith('date:'):
                    fm_date = line.split(':', 1)[1].strip()
                    break
        for dk in date_keywords:
            dk_dash = f"{dk[:4]}-{dk[4:6]}-{dk[6:8]}" if len(dk) == 8 else dk
            if dk in fname_lower or dk_dash == fm_date:
                total_score += 25

        # ── 프론트매터 태그 매칭 ──
        for kw in keywords:
            if kw in fm_tags:
                total_score += 15
        # ── 프론트매터 FAB 매칭 ──
        for fab in fab_prefixes:
            if fab.lower() in fm_fabs:
                total_score += 20
        for kw in keywords:
            if kw in fm_fabs:
                total_score += 15
        # ── 카테고리 매칭 ──
        cat_keywords = {"컬럼": "column_definition", "아키텍처": "architecture", "변경": "changelog",
                        "산출물": "specification", "계획": "plan", "가이드": "guide"}
        for kw in keywords:
            if cat_keywords.get(kw) == fm_category:
                total_score += 10
        # ── description 매칭 ──
        for kw in keywords:
            if kw in fm_desc:
                total_score += 10

        # ── 컬럼명 직접 매칭 (최우선) ──
        for col in full_columns:
            if col in content_lower:
                total_score += 30

        # ── FAB 접두사 → 파일명 매칭 ──
        for fab in fab_prefixes:
            fab_l = fab.lower()
            if fab_l in fname_lower or fab_l in fname_nospace:
                total_score += 20

        # 1. 파일명 매칭 (긴 식별자는 큰 보너스 — FREE_FLOW_SPEED_HID_VALUE 같은 합성어 대응)
        for kw in keywords:
            if kw in fname_lower or kw in fname_nospace:
                # 8자 이상 언더스코어 포함 식별자는 확실한 타겟이므로 +40
                if len(kw) >= 8 and '_' in kw:
                    total_score += 40
                else:
                    total_score += 10

        # 2. BM25 스코어 (토큰 일치 기반)
        if fname in _BM25_INDEX:
            doc_info = _BM25_INDEX[fname]
            bm25 = _bm25_score(query_tokens, doc_info["tokens"], doc_info["token_count"])
            total_score += int(bm25 * 10)  # 스케일링

        # 2-b. 부분문자열 TF (BM25 토크나이저가 놓치는 FREE_FLOW_SPEED_HID_VALUE 같은 합성어 대응)
        # - BM25는 `_`로 연결된 긴 합성 토큰 vs 컨텐츠 내 분리된 짧은 토큰을 매칭 못함
        # - 그래서 BM25 점수와 별개로 substring count도 항상 합산
        for kw in keywords:
            count = content_lower.count(kw)
            if count > 0:
                total_score += min(count, 10)
            else:
                count_nospace = content_nospace.count(kw)
                if count_nospace > 0:
                    total_score += min(count_nospace, 5)

        # 3. 구문 일치 보너스
        if len(keywords) >= 2:
            for i in range(len(keywords) - 1):
                bigram = keywords[i] + keywords[i + 1]
                if bigram in content_nospace or bigram in fname_nospace:
                    total_score += 15
            if query_nospace in content_nospace or query_nospace in fname_nospace:
                total_score += 20

        # 4. 한글 유사 문자 매칭
        for kw in keywords:
            for a, b in [("률", "율"), ("렬", "열"), ("례", "예")]:
                alt_kw = kw.replace(a, b) if a in kw else kw.replace(b, a) if b in kw else None
                if alt_kw and alt_kw != kw and alt_kw in content_lower:
                    total_score += 3

        if total_score > 0:
            preview_parts = []
            for kw in keywords:
                pos = content_lower.find(kw)
                if pos == -1:
                    pos_ns = content_nospace.find(kw)
                    if pos_ns != -1:
                        pos = min(pos_ns, len(content) - 1)
                if pos != -1:
                    start = max(0, pos - 80)
                    end = min(len(content), pos + len(kw) + 80)
                    snippet = content[start:end].replace('\n', ' ').strip()
                    if start > 0:
                        snippet = '...' + snippet
                    if end < len(content):
                        snippet = snippet + '...'
                    preview_parts.append(snippet)

            results.append({
                "filename": fname,
                "score": total_score,
                "content": content[:max_content_chars],
                "content_length": len(content),
                "preview": preview_parts[:3],
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    if results:
        top_score = results[0]["score"]
        min_score = max(5, top_score * 0.3)
        # ── DEBUG: 모든 후보 점수 출력 (필요 시 주석 처리) ──
        print(f"[KNOWLEDGE DEBUG] query={query!r}, top={top_score}, min_cut={min_score:.1f}")
        for r in results:
            mark = "✓" if r["score"] >= min_score else "✗ CUT"
            print(f"  [{mark}] {r['filename']}: {r['score']}")
        results = [r for r in results if r["score"] >= min_score]
    return results[:max_results]


def register_knowledge_routes(app):
    """Register knowledge API routes."""

    @app.route("/api/knowledge/search", methods=["POST"])
    def api_knowledge_search():
        """도메인 지식 BM25 검색 API"""
        data = request.json or {}
        query = data.get("query", "").strip()
        user_id = data.get("user_id", None)
        if not query:
            return jsonify({"error": "query 파라미터가 필요합니다."}), 400

        results = search_knowledge(query, user_id=user_id)

        if not results:
            files = []
            if os.path.isdir(KNOWLEDGE_DIR):
                files = [f for f in os.listdir(KNOWLEDGE_DIR) if f.endswith('.md')]
            return jsonify({
                "query": query,
                "results": [],
                "total": 0,
                "message": f"'{query}'에 매칭되는 문서가 없습니다.",
                "available_files": files,
            })

        return jsonify({
            "query": query,
            "results": [{"filename": r["filename"], "score": r["score"], "preview": r["preview"], "content_length": r["content_length"]} for r in results],
            "total": len(results),
        })

    @app.route("/api/knowledge/files", methods=["GET"])
    def api_knowledge_files():
        """knowledge 폴더 파일 목록 반환 (q, date 필터 지원)"""
        q = request.args.get("q", "").strip().lower()
        date_filter = request.args.get("date", "").strip()
        user_id = request.args.get("user_id", None)
        target_dir = os.path.join(KNOWLEDGE_DIR, user_id) if user_id else KNOWLEDGE_DIR

        files = []
        total_size = 0
        fab_set = set()
        if os.path.isdir(target_dir):
            for fname in sorted(os.listdir(target_dir)):
                if not fname.endswith('.md'):
                    continue
                fpath = os.path.join(target_dir, fname)
                size = os.path.getsize(fpath)
                total_size += size

                # FAB 카운트
                if fname.upper().startswith("FAB_"):
                    fab_name = fname.split("_")[1] if "_" in fname else ""
                    if fab_name:
                        fab_set.add(fab_name)

                # 필터
                if q and q not in fname.lower():
                    continue
                if date_filter and date_filter not in fname:
                    continue

                files.append({"filename": fname, "size": size})

        return jsonify({
            "files": files,
            "total": len(files),
            "total_size": total_size,
            "fab_count": len(fab_set),
        })

    @app.route("/api/knowledge/view/<path:filename>", methods=["GET"])
    def api_knowledge_view(filename):
        """knowledge 파일 내용 조회"""
        user_id = request.args.get("user_id", None)
        target_dir = os.path.join(KNOWLEDGE_DIR, user_id) if user_id else KNOWLEDGE_DIR
        fpath = os.path.join(target_dir, filename)

        if not os.path.isfile(fpath):
            return jsonify({"error": f"파일을 찾을 수 없습니다: {filename}"}), 404

        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read()
            return jsonify({
                "filename": filename,
                "content": content,
                "size": os.path.getsize(fpath),
            })
        except Exception as e:
            return jsonify({"error": f"파일 읽기 실패: {str(e)}"}), 500

    @app.route("/api/knowledge/download/<path:filename>", methods=["GET"])
    def api_knowledge_download(filename):
        """knowledge 파일 다운로드"""
        user_id = request.args.get("user_id", None)
        target_dir = os.path.join(KNOWLEDGE_DIR, user_id) if user_id else KNOWLEDGE_DIR
        fpath = os.path.join(target_dir, filename)

        if not os.path.isfile(fpath):
            return jsonify({"error": f"파일을 찾을 수 없습니다: {filename}"}), 404

        return send_file(fpath, as_attachment=True, download_name=filename)

    @app.route("/knowledge", methods=["GET"])
    def knowledge_explorer():
        """지식 도메인 탐색 페이지"""
        return """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>지식 도메인 탐색</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:"Pretendard","Noto Sans KR",sans-serif;background:#f5f5f5;color:#1a1a2e}
.header{background:linear-gradient(135deg,#1a1a2e,#16213e);color:#fff;padding:1.5rem 2rem;display:flex;align-items:center;gap:1rem}
.header h1{font-size:1.5rem}
.stats{display:flex;gap:1rem;margin-left:auto;font-size:0.85rem}
.stat{background:rgba(255,255,255,0.15);padding:0.4rem 0.8rem;border-radius:8px}
.controls{padding:1rem 2rem;background:#fff;border-bottom:1px solid #e2e8f0;display:flex;gap:0.8rem;flex-wrap:wrap}
.controls input,.controls select{padding:0.5rem 0.8rem;border:1px solid #cbd5e1;border-radius:6px;font-size:0.9rem}
.controls input{flex:1;min-width:200px}
.controls button{padding:0.5rem 1rem;background:#6366f1;color:#fff;border:none;border-radius:6px;cursor:pointer}
.controls button:hover{background:#4f46e5}
.file-list{padding:1rem 2rem;display:grid;gap:0.6rem}
.file-item{background:#fff;padding:0.8rem 1.2rem;border-radius:8px;display:flex;align-items:center;gap:1rem;border:1px solid #e2e8f0;cursor:pointer;transition:all 0.15s}
.file-item:hover{border-color:#6366f1;box-shadow:0 2px 8px rgba(99,102,241,0.15)}
.file-name{flex:1;font-weight:500;font-size:0.95rem}
.file-size{color:#64748b;font-size:0.8rem}
.file-actions{display:flex;gap:0.4rem}
.file-actions button{padding:0.3rem 0.6rem;font-size:0.75rem;border:1px solid #cbd5e1;background:#fff;border-radius:4px;cursor:pointer}
.file-actions button:hover{background:#f1f5f9}
.modal-overlay{display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.5);z-index:1000;align-items:center;justify-content:center}
.modal-overlay.active{display:flex}
.modal{background:#fff;width:80%;max-width:900px;max-height:80vh;border-radius:12px;overflow:hidden;display:flex;flex-direction:column}
.modal-header{padding:1rem 1.5rem;background:#f8fafc;border-bottom:1px solid #e2e8f0;display:flex;align-items:center}
.modal-header h3{flex:1}
.modal-header button{background:none;border:none;font-size:1.2rem;cursor:pointer}
.modal-body{padding:1.5rem;overflow-y:auto;white-space:pre-wrap;font-family:"Fira Code",monospace;font-size:0.85rem;line-height:1.6}
.empty{text-align:center;padding:3rem;color:#94a3b8}
</style>
</head>
<body>
<div class="header">
    <h1>📚 지식 도메인 탐색</h1>
    <div class="stats">
        <span class="stat" id="statTotal">문서: -</span>
        <span class="stat" id="statSize">용량: -</span>
        <span class="stat" id="statFab">FAB: -</span>
    </div>
</div>
<div class="controls">
    <input type="text" id="searchInput" placeholder="키워드 검색 (파일명 + 내용)">
    <select id="catFilter">
        <option value="">전체 카테고리</option>
        <option value="FAB_">FAB</option>
        <option value="산출물">산출물</option>
        <option value="변경">변경사항</option>
        <option value="계획">계획</option>
        <option value="아키텍처">아키텍처</option>
        <option value="주간">주간보고</option>
    </select>
    <button onclick="loadFiles()">🔍 검색</button>
</div>
<div class="file-list" id="fileList"><div class="empty">로딩 중...</div></div>

<div class="modal-overlay" id="modal">
    <div class="modal">
        <div class="modal-header">
            <h3 id="modalTitle">파일</h3>
            <button onclick="closeModal()">✕</button>
        </div>
        <div class="modal-body" id="modalBody"></div>
    </div>
</div>

<script>
let allFiles = [];

async function loadFiles() {
    const q = document.getElementById('searchInput').value.trim();
    const cat = document.getElementById('catFilter').value;
    const searchQ = cat ? cat : q;
    const res = await fetch('/api/knowledge/files?q=' + encodeURIComponent(searchQ));
    const data = await res.json();
    allFiles = data.files || [];

    document.getElementById('statTotal').textContent = '문서: ' + data.total;
    document.getElementById('statSize').textContent = '용량: ' + (data.total_size / 1024).toFixed(1) + ' KB';
    document.getElementById('statFab').textContent = 'FAB: ' + (data.fab_count || 0);

    let filtered = allFiles;
    if (q && cat) {
        filtered = allFiles.filter(f => f.filename.toLowerCase().includes(q.toLowerCase()));
    }

    const list = document.getElementById('fileList');
    if (filtered.length === 0) {
        list.innerHTML = '<div class="empty">검색 결과가 없습니다.</div>';
        return;
    }
    list.innerHTML = filtered.map(f => `
        <div class="file-item" onclick="viewFile('${f.filename}')">
            <span>📄</span>
            <span class="file-name">${f.filename}</span>
            <span class="file-size">${(f.size/1024).toFixed(1)} KB</span>
            <div class="file-actions">
                <button onclick="event.stopPropagation();viewFile('${f.filename}')">보기</button>
                <button onclick="event.stopPropagation();location.href='/api/knowledge/download/${f.filename}'">다운로드</button>
            </div>
        </div>
    `).join('');
}

async function viewFile(filename) {
    const res = await fetch('/api/knowledge/view/' + encodeURIComponent(filename));
    const data = await res.json();
    document.getElementById('modalTitle').textContent = filename;
    document.getElementById('modalBody').textContent = data.content || '내용 없음';
    document.getElementById('modal').classList.add('active');
}

function closeModal() {
    document.getElementById('modal').classList.remove('active');
}

document.getElementById('modal').addEventListener('click', function(e) {
    if (e.target === this) closeModal();
});

document.getElementById('searchInput').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') loadFiles();
});

loadFiles();
</script>
</body>
</html>"""
