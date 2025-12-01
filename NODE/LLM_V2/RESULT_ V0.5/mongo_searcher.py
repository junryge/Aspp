#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MongoDB / Logpresso 접속 정보 검색 모듈
- MD 파일에서 접속 정보 검색
- LLM 없이 템플릿 기반 포맷팅
"""

import os
import re
import logging
from typing import Tuple, Optional, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MD 파일 경로
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MD_PATH = os.path.join(SCRIPT_DIR, 'MOGO_Read', 'MOGO_Read.md')

# 전역 변수
_md_content = None
_sections = {}


def is_mongo_query(query: str) -> bool:
    """MongoDB/Logpresso 관련 쿼리인지 판단"""
    query_lower = query.lower()
    
    keywords = [
        'mongo', 'mongodb', '몽고',
        'logpresso', '로그프레소', '로그',
        'atlasread', 'mcslog',
        'pkt', 'pnt4wt', 'nwt',
        'm11', 'm15',
        '27020',  # MongoDB 포트
        '우시', 'wuxi'
    ]
    
    # 키워드 매칭
    for kw in keywords:
        if kw in query_lower:
            return True
    
    return False


def load_md() -> bool:
    """MD 파일 로드"""
    global _md_content
    
    if not os.path.exists(MD_PATH):
        logger.error(f"❌ MD 파일 없음: {MD_PATH}")
        return False
    
    try:
        with open(MD_PATH, 'r', encoding='utf-8') as f:
            _md_content = f.read()
        logger.info(f"✅ MD 로드 완료: {len(_md_content)} bytes")
        parse_sections()
        return True
    except Exception as e:
        logger.error(f"❌ MD 로드 실패: {e}")
        return False


def parse_sections():
    """MD 파일을 섹션별로 파싱"""
    global _sections
    
    if _md_content is None:
        return
    
    content = _md_content
    _sections = {}
    
    # 공통 계정
    account_start = content.find('## 공통 계정')
    mongo_start = content.find('## MongoDB')
    logpresso_start = content.find('## Logpresso')
    summary_start = content.find('## 한눈에 보기')
    
    # 계정 섹션
    if account_start != -1:
        account_end = mongo_start if mongo_start != -1 else len(content)
        _sections['계정'] = content[account_start:account_end].strip()
    
    # MongoDB 섹션
    if mongo_start != -1:
        mongo_end = logpresso_start if logpresso_start != -1 else summary_start if summary_start != -1 else len(content)
        mongo_content = content[mongo_start:mongo_end].strip()
        _sections['MongoDB_전체'] = mongo_content
        
        # MongoDB 이천
        icheon_start = mongo_content.find('### 이천')
        cheongju_start = mongo_content.find('### 청주')
        format_start = mongo_content.find('### MongoDB 접속 형식')
        
        if icheon_start != -1:
            icheon_end = cheongju_start if cheongju_start != -1 else format_start if format_start != -1 else len(mongo_content)
            _sections['MongoDB_이천'] = mongo_content[icheon_start:icheon_end].strip()
        
        if cheongju_start != -1:
            cheongju_end = format_start if format_start != -1 else len(mongo_content)
            _sections['MongoDB_청주'] = mongo_content[cheongju_start:cheongju_end].strip()
        
        if format_start != -1:
            _sections['MongoDB_접속형식'] = mongo_content[format_start:].strip()
    
    # Logpresso 섹션
    if logpresso_start != -1:
        logpresso_end = summary_start if summary_start != -1 else len(content)
        logpresso_content = content[logpresso_start:logpresso_end].strip()
        _sections['Logpresso_전체'] = logpresso_content
        
        # Logpresso 이천
        icheon_start = logpresso_content.find('### 이천')
        cheongju_start = logpresso_content.find('### 청주')
        wuxi_start = logpresso_content.find('### 우시')
        
        if icheon_start != -1:
            icheon_end = cheongju_start if cheongju_start != -1 else wuxi_start if wuxi_start != -1 else len(logpresso_content)
            _sections['Logpresso_이천'] = logpresso_content[icheon_start:icheon_end].strip()
        
        if cheongju_start != -1:
            cheongju_end = wuxi_start if wuxi_start != -1 else len(logpresso_content)
            _sections['Logpresso_청주'] = logpresso_content[cheongju_start:cheongju_end].strip()
        
        if wuxi_start != -1:
            _sections['Logpresso_우시'] = logpresso_content[wuxi_start:].strip()
    
    # 요약 섹션
    if summary_start != -1:
        summary_content = content[summary_start:].strip()
        _sections['요약'] = summary_content
        
        mongo_summary = summary_content.find('### MongoDB 요약')
        logpresso_summary = summary_content.find('### Logpresso 요약')
        
        if mongo_summary != -1:
            mongo_end = logpresso_summary if logpresso_summary != -1 else len(summary_content)
            _sections['MongoDB_요약'] = summary_content[mongo_summary:mongo_end].strip()
        
        if logpresso_summary != -1:
            _sections['Logpresso_요약'] = summary_content[logpresso_summary:].strip()
    
    logger.info(f"✅ 섹션 파싱 완료: {list(_sections.keys())}")


def format_result(section_key: str, context: str) -> str:
    """MD 테이블을 사람이 읽기 좋게 변환 - LLM 없이!"""
    
    titles = {
        '계정': '👤 공통 계정 정보',
        'MongoDB_이천': '🍃 MongoDB - 이천',
        'MongoDB_청주': '🍃 MongoDB - 청주',
        'MongoDB_전체': '🍃 MongoDB 전체',
        'MongoDB_접속형식': '🔗 MongoDB 접속 형식',
        'Logpresso_이천': '📊 Logpresso - 이천',
        'Logpresso_청주': '📊 Logpresso - 청주',
        'Logpresso_우시': '📊 Logpresso - 우시',
        'Logpresso_전체': '📊 Logpresso 전체',
        'MongoDB_요약': '📋 MongoDB 요약',
        'Logpresso_요약': '📋 Logpresso 요약',
        '요약': '📋 전체 요약'
    }
    
    title = titles.get(section_key, f'📂 {section_key}')
    
    # 맨 위에 시스템 제목 추가
    result = "🗄️ MongoDB / Logpresso 접속 정보\n"
    result += "=" * 50 + "\n\n"
    result += f"{title}\n"
    result += "-" * 50 + "\n\n"
    
    lines = context.split('\n')
    
    # 계정 정보 (항상 포함)
    result += "👤 계정: atlasread\n"
    result += "🔑 비밀번호: Readatlas1^\n\n"
    
    # 정보 수집용
    hosts = []
    current_fab = ""
    current_env = ""
    
    for line in lines:
        line = line.strip()
        
        # 빈 줄 스킵
        if not line:
            continue
        
        # 테이블 구분선 스킵
        if line.startswith('|') and '---' in line:
            continue
        
        # 헤더 스킵
        if line.startswith('|') and ('FAB' in line or '항목' in line or '사이트' in line):
            continue
        
        # 코드 블록
        if line.startswith('```'):
            continue
        
        # 접속 형식
        if 'mongodb://' in line:
            result += f"🔗 접속 형식:\n"
            result += f"   {line}\n\n"
            continue
        
        # 테이블 행 파싱 (| FAB | 환경 | 호스트/IP |)
        if line.startswith('|') and line.endswith('|'):
            parts = [p.strip() for p in line.split('|') if p.strip()]
            
            if len(parts) >= 3:
                fab = parts[0]
                env = parts[1]
                host = parts[2]
                
                # FAB 정보
                if fab and fab != '-':
                    current_fab = fab
                
                # 환경 (운영/QA)
                env_emoji = "🟢" if '운영' in env else "🟡"
                
                # 호스트 목록 파싱
                host_list = [h.strip() for h in host.split(',')]
                
                result += f"{env_emoji} {current_fab} {env}\n"
                for i, h in enumerate(host_list, 1):
                    result += f"   🖥️ {h}\n"
                result += "\n"
                
                hosts.extend(host_list)
    
    # 요약 생성
    result += "-" * 50 + "\n"
    
    if 'MongoDB' in section_key:
        if '이천' in section_key:
            result += "📝 요약: 이천 MongoDB - PKT(운영), PNT4WT(QA) 클러스터\n"
            result += "🔗 접속: mongodb://atlasread@{호스트}/?authSource=mcslog\n"
        elif '청주' in section_key:
            result += "📝 요약: 청주 MongoDB - NWT(운영), QA 클러스터\n"
            result += "🔗 접속: mongodb://atlasread@{호스트}/?authSource=mcslog\n"
        else:
            result += "📝 요약: MongoDB 접속 정보 (이천/청주)\n"
    elif 'Logpresso' in section_key:
        if '이천' in section_key:
            result += "📝 요약: 이천 Logpresso 서버 (운영/QA)\n"
        elif '청주' in section_key:
            result += "📝 요약: 청주 Logpresso 서버 (M11/M15 운영, QA)\n"
        elif '우시' in section_key:
            result += "📝 요약: 우시 Logpresso 서버 (운영)\n"
        else:
            result += "📝 요약: Logpresso 로그 서버 접속 정보\n"
    elif section_key == '계정':
        result += "📝 요약: MongoDB 공통 접속 계정 (atlasread)\n"
    else:
        result += "📝 요약: MongoDB/Logpresso 접속 정보\n"
    
    return result


def search(query: str) -> Tuple[Optional[str], str]:
    """쿼리로 검색"""
    global _sections
    
    # MD 로드 확인
    if _md_content is None:
        if not load_md():
            return None, "❌ MD 파일을 로드할 수 없습니다."
    
    query_lower = query.lower()
    
    # 1. MongoDB vs Logpresso 구분
    is_mongo = any(kw in query_lower for kw in ['mongo', '몽고', 'pkt', 'pnt4wt', 'nwt', '27020'])
    is_logpresso = any(kw in query_lower for kw in ['logpresso', '로그프레소', '로그', 'm11', 'm15'])
    
    # 2. 지역 구분
    is_icheon = any(kw in query_lower for kw in ['이천', 'icheon', 'pkt', 'pnt4wt'])
    is_cheongju = any(kw in query_lower for kw in ['청주', 'cheongju', 'nwt', 'm11', 'm15'])
    is_wuxi = any(kw in query_lower for kw in ['우시', 'wuxi'])
    
    # 3. 환경 구분
    is_prod = any(kw in query_lower for kw in ['운영', 'prod', 'production'])
    is_qa = any(kw in query_lower for kw in ['qa', 'test', '테스트'])
    
    # 4. 기타
    is_account = any(kw in query_lower for kw in ['계정', '비밀번호', 'password', 'atlasread'])
    is_summary = any(kw in query_lower for kw in ['요약', '전체', 'summary', '한눈'])
    is_format = any(kw in query_lower for kw in ['접속', '형식', 'connection', 'format'])
    
    # 섹션 결정
    section_key = None
    
    if is_account:
        section_key = '계정'
    elif is_summary:
        if is_mongo:
            section_key = 'MongoDB_요약'
        elif is_logpresso:
            section_key = 'Logpresso_요약'
        else:
            section_key = '요약'
    elif is_mongo:
        if is_format:
            section_key = 'MongoDB_접속형식'
        elif is_icheon:
            section_key = 'MongoDB_이천'
        elif is_cheongju:
            section_key = 'MongoDB_청주'
        else:
            section_key = 'MongoDB_전체'
    elif is_logpresso:
        if is_icheon:
            section_key = 'Logpresso_이천'
        elif is_cheongju:
            section_key = 'Logpresso_청주'
        elif is_wuxi:
            section_key = 'Logpresso_우시'
        else:
            section_key = 'Logpresso_전체'
    else:
        # 기본: 요약
        section_key = '요약'
    
    # 섹션 가져오기
    if section_key not in _sections:
        # 전체 내용 반환
        return '요약', format_result('요약', _sections.get('요약', _md_content))
    
    context = _sections[section_key]
    formatted = format_result(section_key, context)
    
    return section_key, formatted


# ============================================================
# 추가된 함수들 (star_searcher.py와 동일한 인터페이스)
# ============================================================

def get_summary() -> str:
    """전체 요약 정보 반환"""
    global _sections
    
    if _md_content is None:
        if not load_md():
            return "MongoDB/Logpresso 정보 없음"
    
    return _sections.get('요약', "MongoDB/Logpresso 정보 없음")


def get_full_content() -> str:
    """전체 MD 내용 반환"""
    global _md_content
    
    if _md_content is None:
        if not load_md():
            return "MongoDB/Logpresso 정보 없음"
    
    return _md_content


# 테스트
if __name__ == "__main__":
    load_md()
    
    test_queries = [
        "MongoDB 이천 접속 정보",
        "청주 Logpresso",
        "몽고 계정",
        "우시 로그프레소",
        "MongoDB 요약"
    ]
    
    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"쿼리: {q}")
        print('='*60)
        section, result = search(q)
        print(result)
    
    # 추가된 함수 테스트
    print(f"\n{'='*60}")
    print("get_summary() 테스트")
    print('='*60)
    print(get_summary()[:200] + "...")
    
    print(f"\n{'='*60}")
    print("get_full_content() 테스트")
    print('='*60)
    print(f"전체 내용 길이: {len(get_full_content())} bytes")