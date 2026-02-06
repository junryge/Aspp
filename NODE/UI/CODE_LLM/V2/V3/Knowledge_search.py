#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
knowledge_search.py
섹션 단위 MD 지식베이스 검색 모듈

기능:
1. MD 파일을 ## 헤더 기준으로 섹션 분리
2. 섹션별 키워드 인덱싱
3. 자연어 쿼리 → 관련 섹션 검색 (퍼지 매칭)
4. LLM 연동 시 검색된 섹션을 컨텍스트로 전달

사용법:
    from knowledge_search import KnowledgeBase
    kb = KnowledgeBase("./knowledge")
    results = kb.search("앙상블 모델 5개 규칙")
"""

import os
import re
import json
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("KnowledgeSearch")


# ========================================
# 데이터 클래스
# ========================================
@dataclass
class Section:
    """MD 파일의 한 섹션"""
    file: str           # 원본 파일명
    header: str         # 섹션 제목 (## 11. 모델 성능 지표...)
    level: int          # 헤더 레벨 (1=H1, 2=H2, 3=H3...)
    content: str        # 섹션 본문 (헤더 포함)
    line_start: int     # 시작 라인
    line_end: int       # 끝 라인
    keywords: List[str] = field(default_factory=list)  # 추출된 키워드
    parent_header: str = ""  # 상위 섹션 제목

    def to_dict(self):
        return {
            "file": self.file,
            "header": self.header,
            "level": self.level,
            "content": self.content[:500] + ("..." if len(self.content) > 500 else ""),
            "full_content": self.content,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "keywords": self.keywords,
            "parent_header": self.parent_header,
        }


@dataclass
class SearchResult:
    """검색 결과"""
    section: Section
    score: float        # 관련성 점수
    matched_terms: List[str]  # 매칭된 키워드들

    def to_dict(self):
        return {
            "file": self.section.file,
            "header": self.section.header,
            "level": self.section.level,
            "content": self.section.content,
            "score": self.score,
            "matched_terms": self.matched_terms,
            "parent_header": self.section.parent_header,
            "lines": f"{self.section.line_start}-{self.section.line_end}",
        }


# ========================================
# MD 파서
# ========================================
class MDParser:
    """마크다운 파일을 섹션 단위로 파싱"""

    @staticmethod
    def parse_file(filepath: str) -> List[Section]:
        """MD 파일을 섹션 리스트로 변환"""
        filename = os.path.basename(filepath)

        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except Exception as e:
            logger.error(f"파일 읽기 실패: {filepath} - {e}")
            return []

        sections = []
        current_section_lines = []
        current_header = ""
        current_level = 0
        current_start = 1
        parent_headers = {}  # level → header 매핑

        for i, line in enumerate(lines, 1):
            # 헤더 감지 (# ~ ######)
            header_match = re.match(r'^(#{1,6})\s+(.+)', line.strip())

            if header_match:
                # 이전 섹션 저장
                if current_section_lines:
                    content = ''.join(current_section_lines).strip()
                    if content and len(content) > 10:  # 너무 짧은 섹션 제외
                        section = Section(
                            file=filename,
                            header=current_header or filename,
                            level=current_level,
                            content=content,
                            line_start=current_start,
                            line_end=i - 1,
                            keywords=MDParser._extract_keywords(content),
                            parent_header=parent_headers.get(current_level - 1, ""),
                        )
                        sections.append(section)

                # 새 섹션 시작
                level = len(header_match.group(1))
                header = header_match.group(2).strip()
                parent_headers[level] = header

                current_header = header
                current_level = level
                current_section_lines = [line]
                current_start = i
            else:
                current_section_lines.append(line)

        # 마지막 섹션 저장
        if current_section_lines:
            content = ''.join(current_section_lines).strip()
            if content and len(content) > 10:
                section = Section(
                    file=filename,
                    header=current_header or filename,
                    level=current_level,
                    content=content,
                    line_start=current_start,
                    line_end=len(lines),
                    keywords=MDParser._extract_keywords(content),
                    parent_header=parent_headers.get(current_level - 1, ""),
                )
                sections.append(section)

        logger.info(f"📄 {filename}: {len(sections)}개 섹션 파싱됨")
        return sections

    @staticmethod
    def _extract_keywords(content: str) -> List[str]:
        """섹션 내용에서 키워드 추출"""
        keywords = set()

        # 1. 헤더에서 키워드 (## 뒤의 텍스트)
        for match in re.finditer(r'^#{1,6}\s+(.+)', content, re.MULTILINE):
            header_text = match.group(1).strip()
            # 번호 제거 (7-1. → 제거)
            header_text = re.sub(r'^\d+[-.]?\d*\.?\s*', '', header_text)
            keywords.add(header_text.lower())
            # 개별 단어도 추가
            for word in re.split(r'[\s/()（）\[\]]+', header_text):
                if len(word) >= 2:
                    keywords.add(word.lower())

        # 2. 볼드 텍스트 (**text**)
        for match in re.finditer(r'\*\*([^*]+)\*\*', content):
            keywords.add(match.group(1).lower().strip())

        # 3. 코드 블록 내 주요 변수명
        for match in re.finditer(r'`([^`]+)`', content):
            code = match.group(1).strip()
            if len(code) >= 2 and len(code) <= 50:
                keywords.add(code.lower())

        # 4. 테이블 헤더 (| 키 | 값 |)
        for match in re.finditer(r'\|\s*\*?\*?([^|*]+)\*?\*?\s*\|', content):
            cell = match.group(1).strip()
            if len(cell) >= 2 and cell not in ('---', '항목', '값', '설명', '#'):
                keywords.add(cell.lower())

        # 5. 한국어 핵심 명사 (2글자 이상)
        for match in re.finditer(r'[가-힣]{2,10}', content):
            word = match.group()
            # 조사/어미 제외
            if word not in ('있으면', '없으면', '아니면', '입니다', '합니다', '됩니다',
                           '에서는', '에서의', '으로의', '이므로', '때문에', '그래서'):
                keywords.add(word.lower())

        return list(keywords)[:100]  # 최대 100개


# ========================================
# 지식베이스 검색 엔진
# ========================================
class KnowledgeBase:
    """섹션 단위 지식베이스 검색"""

    def __init__(self, knowledge_dir: str):
        self.knowledge_dir = knowledge_dir
        self.sections: List[Section] = []
        self.files: List[str] = []
        self._index()

    def _index(self):
        """지식베이스 디렉토리의 모든 MD/TXT 파일 인덱싱"""
        self.sections = []
        self.files = []

        if not os.path.exists(self.knowledge_dir):
            os.makedirs(self.knowledge_dir, exist_ok=True)
            logger.warning(f"📂 지식베이스 디렉토리 생성: {self.knowledge_dir}")
            return

        for f in sorted(os.listdir(self.knowledge_dir)):
            if f.endswith(('.md', '.txt')):
                filepath = os.path.join(self.knowledge_dir, f)
                self.files.append(f)
                sections = MDParser.parse_file(filepath)
                self.sections.extend(sections)

        logger.info(f"📚 지식베이스 인덱싱 완료: {len(self.files)}개 파일, {len(self.sections)}개 섹션")

    def refresh(self):
        """인덱스 갱신"""
        self._index()

    def search(self, query: str, top_k: int = 5, min_score: int = 10) -> List[SearchResult]:
        """
        자연어 쿼리로 관련 섹션 검색

        Args:
            query: 검색어 (예: "앙상블 모델 5개 규칙")
            top_k: 최대 반환 개수
            min_score: 최소 점수 (이하 제외)

        Returns:
            SearchResult 리스트 (점수 내림차순)
        """
        if not self.sections:
            return []

        # 쿼리 토큰화
        query_tokens = self._tokenize_query(query)
        if not query_tokens:
            return []

        logger.info(f"🔍 검색: '{query}' → 토큰: {query_tokens}")

        results = []
        for section in self.sections:
            score, matched = self._score_section(section, query_tokens, query)
            if score >= min_score:
                results.append(SearchResult(
                    section=section,
                    score=score,
                    matched_terms=matched,
                ))

        # 점수 내림차순 정렬
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def search_by_topic(self, topics: List[str], top_k_per_topic: int = 3) -> Dict[str, List[SearchResult]]:
        """
        여러 토픽을 한 번에 검색

        Args:
            topics: ["모델 성능 지표", "황금 패턴", "앙상블 5개 규칙"]
            top_k_per_topic: 토픽당 최대 결과 수

        Returns:
            {"토픽": [SearchResult, ...], ...}
        """
        results = {}
        for topic in topics:
            results[topic] = self.search(topic, top_k=top_k_per_topic)
        return results

    def get_context_for_llm(self, query: str, max_tokens: int = 8000) -> str:
        """
        LLM에 전달할 컨텍스트 문자열 생성

        검색 결과를 합쳐서 LLM 프롬프트에 넣을 수 있는 형태로 반환
        """
        results = self.search(query, top_k=5)
        if not results:
            return ""

        context_parts = []
        total_len = 0

        for r in results:
            section_text = f"📄 [{r.section.file}] {r.section.header}\n{r.section.content}"
            if total_len + len(section_text) > max_tokens * 4:  # 대략적 토큰 추정
                break
            context_parts.append(section_text)
            total_len += len(section_text)

        return "\n\n---\n\n".join(context_parts)

    def list_files(self) -> List[Dict]:
        """파일 목록 반환"""
        result = []
        for f in self.files:
            filepath = os.path.join(self.knowledge_dir, f)
            size = os.path.getsize(filepath)
            sections = [s for s in self.sections if s.file == f]
            result.append({
                "filename": f,
                "size": f"{size:,}B",
                "sections": len(sections),
                "headers": [s.header for s in sections],
            })
        return result

    def list_all_sections(self) -> List[Dict]:
        """모든 섹션 목차 반환"""
        return [
            {
                "file": s.file,
                "header": s.header,
                "level": s.level,
                "parent": s.parent_header,
                "content_length": len(s.content),
            }
            for s in self.sections
        ]

    def get_section_by_header(self, header_keyword: str) -> Optional[Section]:
        """헤더 키워드로 정확한 섹션 가져오기"""
        keyword_lower = header_keyword.lower()
        for s in self.sections:
            if keyword_lower in s.header.lower():
                return s
        return None

    # ========================================
    # 내부 메서드
    # ========================================
    def _tokenize_query(self, query: str) -> List[str]:
        """쿼리를 검색 토큰으로 분리"""
        # 조사/어미 제거
        cleaned = re.sub(
            r'(을|를|이|가|은|는|의|에|에서|로|으로|부터|까지|도|만|에대해|에 대해|좀|줘|해줘|알려줘|설명해|뭐야|어떻게)',
            ' ', query
        )

        tokens = []
        # 영어 + 숫자 토큰
        for match in re.finditer(r'[a-zA-Z0-9_]+', cleaned):
            word = match.group().lower()
            if len(word) >= 2:
                tokens.append(word)

        # 한국어 토큰 (2글자 이상)
        for match in re.finditer(r'[가-힣]{2,}', cleaned):
            word = match.group()
            tokens.append(word)

        # 원본 쿼리의 핵심 구절도 추가 (연속된 한글)
        for match in re.finditer(r'[가-힣]{3,}', query):
            word = match.group()
            if word not in tokens:
                tokens.append(word)

        return list(set(tokens))

    def _score_section(self, section: Section, query_tokens: List[str], original_query: str) -> Tuple[float, List[str]]:
        """섹션의 관련성 점수 계산"""
        score = 0.0
        matched = []

        header_lower = section.header.lower()
        content_lower = section.content.lower()
        keywords_set = set(k.lower() for k in section.keywords)

        for token in query_tokens:
            token_lower = token.lower()

            # 1. 헤더 정확 매칭 (+100)
            if token_lower in header_lower:
                score += 100
                matched.append(f"헤더:{token}")

            # 2. 키워드 인덱스 매칭 (+50)
            elif token_lower in keywords_set:
                score += 50
                matched.append(f"키워드:{token}")

            # 3. 본문 포함 (+20 * 빈도, 최대 60)
            count = content_lower.count(token_lower)
            if count > 0:
                score += min(count * 20, 60)
                if f"본문:{token}" not in matched:
                    matched.append(f"본문:{token}({count}회)")

            # 4. 부분 문자열 매칭 (3글자 이상, +10)
            if len(token) >= 3:
                for kw in keywords_set:
                    if token_lower in kw or kw in token_lower:
                        score += 10
                        break

        # 5. 원본 쿼리 연속 매칭 보너스 (+50)
        # "앙상블 모델 5개 규칙" → 이 구절이 통째로 있으면 보너스
        query_lower = original_query.lower()
        if len(query_lower) >= 4 and query_lower in content_lower:
            score += 50
            matched.append(f"구절매칭:{original_query[:20]}")

        # 6. 헤더 레벨 가중치 (H2 > H3 > H4)
        if section.level == 2:
            score *= 1.2  # H2 섹션 우대
        elif section.level >= 4:
            score *= 0.8  # 하위 섹션 감점

        return score, matched


# ========================================
# FastAPI 라우터 (기존 pc_assistant에 추가 가능)
# ========================================
def create_kb_router(knowledge_dir: str) -> "APIRouter":
    """지식베이스 API 라우터 생성"""
    from fastapi import APIRouter
    from pydantic import BaseModel

    kb_router = APIRouter(prefix="/kb", tags=["knowledge-base"])
    kb = KnowledgeBase(knowledge_dir)

    class SearchRequest(BaseModel):
        query: str
        top_k: int = 5

    class MultiSearchRequest(BaseModel):
        topics: List[str]
        top_k_per_topic: int = 3

    @kb_router.get("/files")
    async def list_files():
        return {"success": True, "files": kb.list_files()}

    @kb_router.get("/sections")
    async def list_sections():
        return {"success": True, "sections": kb.list_all_sections()}

    @kb_router.post("/search")
    async def search(req: SearchRequest):
        results = kb.search(req.query, top_k=req.top_k)
        return {
            "success": True,
            "query": req.query,
            "count": len(results),
            "results": [r.to_dict() for r in results],
        }

    @kb_router.post("/multi_search")
    async def multi_search(req: MultiSearchRequest):
        all_results = kb.search_by_topic(req.topics, req.top_k_per_topic)
        return {
            "success": True,
            "results": {
                topic: [r.to_dict() for r in results]
                for topic, results in all_results.items()
            },
        }

    @kb_router.post("/context")
    async def get_context(req: SearchRequest):
        """LLM에 전달할 컨텍스트 반환"""
        context = kb.get_context_for_llm(req.query)
        return {"success": True, "context": context}

    @kb_router.post("/refresh")
    async def refresh():
        kb.refresh()
        return {
            "success": True,
            "files": len(kb.files),
            "sections": len(kb.sections),
        }

    return kb_router


# ========================================
# 테스트
# ========================================
if __name__ == "__main__":
    import sys

    # 테스트용
    test_dir = "./knowledge"
    if len(sys.argv) > 1:
        test_dir = sys.argv[1]

    kb = KnowledgeBase(test_dir)

    print(f"\n📚 파일: {len(kb.files)}개, 섹션: {len(kb.sections)}개\n")

    # 파일 목록
    for f in kb.list_files():
        print(f"  📄 {f['filename']} ({f['size']}, {f['sections']}개 섹션)")

    # 검색 테스트
    test_queries = [
        "모델 성능 지표 산출 로직",
        "예측값 활용 황금 패턴",
        "앙상블 모델 5개 규칙",
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"🔍 검색: {q}")
        print(f"{'='*60}")
        results = kb.search(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"\n  [{i}] 점수: {r.score:.0f} | {r.section.file}")
            print(f"      헤더: {r.section.header}")
            print(f"      매칭: {', '.join(r.matched_terms)}")
            print(f"      미리보기: {r.section.content[:150]}...")