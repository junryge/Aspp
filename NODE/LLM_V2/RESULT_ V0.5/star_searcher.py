#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STAR DB 문서 검색 모듈 - LLM 없이 깔끔하게 출력
"""

import os
import re
import logging
from typing import Optional, Dict, Tuple

logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MD_PATH = os.path.join(SCRIPT_DIR, 'STAR_READ', 'STAR_READ.md')

_sections = None
_raw_content = None


def load_md() -> bool:
    """MD 파일 로드"""
    global _sections, _raw_content
    
    if not os.path.exists(MD_PATH):
        logger.error(f"❌ MD 파일 없음: {MD_PATH}")
        return False
    
    try:
        with open(MD_PATH, 'r', encoding='utf-8') as f:
            _raw_content = f.read()
        
        _sections = parse_sections(_raw_content)
        logger.info(f"✅ MD 로드 완료: {len(_sections)}개 섹션")
        return True
    except Exception as e:
        logger.error(f"❌ MD 로드 실패: {e}")
        return False


def parse_sections(content: str) -> Dict[str, str]:
    """MD 내용을 섹션별로 파싱 - 테이블 구분선(|---)과 섹션 구분선(---)을 구분"""
    sections = {}
    sections['전체'] = content
    
    # 공통 계정 추출
    lines = content.split('\n')
    account_lines = []
    in_account = False
    for i, line in enumerate(lines):
        if '## 공통 계정' in line:
            in_account = True
        if in_account:
            # 다음 ## 섹션이 시작되면 종료
            if line.startswith('## ') and '공통 계정' not in line:
                break
            # 단독 --- (앞뒤로 빈줄)이면 섹션 구분자 → 종료
            if line.strip() == '---':
                next_line = lines[i+1].strip() if i+1 < len(lines) else ''
                if next_line.startswith('##'):
                    break
            account_lines.append(line)
    sections['계정'] = '\n'.join(account_lines).strip()
    
    # 청주 섹션
    cheongju_start = content.find('## 청주')
    icheon_start = content.find('## 이천')
    summary_start = content.find('## 한눈에 보기')
    
    if cheongju_start != -1:
        cheongju_end = icheon_start if icheon_start != -1 else summary_start if summary_start != -1 else len(content)
        cheongju_content = content[cheongju_start:cheongju_end].strip()
        sections['청주_전체'] = cheongju_content
        
        # 청주 운영
        prod_start = cheongju_content.find('### 운영 환경')
        qa_start = cheongju_content.find('### QA 환경')
        if prod_start != -1:
            prod_end = qa_start if qa_start != -1 else len(cheongju_content)
            sections['청주_운영'] = cheongju_content[prod_start:prod_end].strip()
        if qa_start != -1:
            sections['청주_QA'] = cheongju_content[qa_start:].strip()
    
    # 이천 섹션
    if icheon_start != -1:
        icheon_end = summary_start if summary_start != -1 else len(content)
        icheon_content = content[icheon_start:icheon_end].strip()
        sections['이천_전체'] = icheon_content
        
        # 이천 운영
        prod_start = icheon_content.find('### 운영 환경')
        qa_start = icheon_content.find('### QA 환경')
        if prod_start != -1:
            prod_end = qa_start if qa_start != -1 else len(icheon_content)
            sections['이천_운영'] = icheon_content[prod_start:prod_end].strip()
        if qa_start != -1:
            sections['이천_QA'] = icheon_content[qa_start:].strip()
    
    # 한눈에 보기
    if summary_start != -1:
        failover_start = content.find('## Failover')
        summary_end = failover_start if failover_start != -1 else len(content)
        sections['요약'] = content[summary_start:summary_end].strip()
    
    # Failover
    failover_start = content.find('## Failover')
    if failover_start != -1:
        sections['Failover'] = content[failover_start:].strip()
    
    return sections


def format_result(section_key: str, context: str) -> str:
    """MD 테이블을 사람이 읽기 좋게 변환 - LLM 없이!"""
    
    titles = {
        '청주_운영': '🔵 청주 운영 환경',
        '청주_QA': '🟡 청주 QA 환경',
        '이천_운영': '🔵 이천 운영 환경',
        '이천_QA': '🟡 이천 QA 환경',
        '계정': '👤 공통 계정 정보',
        '요약': '📊 전체 요약',
        'Failover': '🔧 Failover 설정'
    }
    
    title = titles.get(section_key, f'📂 {section_key}')
    
    # 맨 위에 시스템 제목 추가
    result = "🗄️ smartSTAR Database 접속 정보\n"
    result += "=" * 45 + "\n\n"
    result += f"{title}\n"
    result += "-" * 45 + "\n\n"
    
    lines = context.split('\n')
    
    # 정보 수집용
    service_name = ""
    node_count = 0
    
    for line in lines:
        line = line.strip()
        
        # 빈 줄 스킵
        if not line:
            continue
        
        # 테이블 구분선 스킵 (|------|-----|)
        if line.startswith('|') and '---' in line:
            continue
        
        # 섹션 구분선 스킵 (단독 ---)
        if line == '---':
            continue
        
        # 헤더 제거 (## ### ####)
        if line.startswith('#'):
            continue
        
        # 테이블 행 파싱: | 항목 | 값 |
        if line.startswith('|') and line.endswith('|'):
            cells = [c.strip() for c in line.split('|') if c.strip()]
            
            if len(cells) >= 2:
                key = cells[0]
                value = cells[1]
                
                # 헤더 행 스킵
                if key in ['항목', '사이트'] or value in ['값', '환경']:
                    continue
                
                # 정보 수집
                if 'Service' in key:
                    service_name = value
                if 'Node' in key:
                    node_count += 1
                
                # 이모지 추가
                if 'Service' in key:
                    result += f"📌 {key}: {value}\n"
                elif 'Node' in key:
                    result += f"   🖥️ {key}: {value}\n"
                elif '계정' in key:
                    result += f"👤 {key}: {value}\n"
                elif '비밀번호' in key:
                    result += f"🔑 {key}: {value}\n"
                elif len(cells) >= 4:  # 요약 테이블 (4컬럼)
                    result += f"📍 {cells[0]} {cells[1]}: {cells[2]} ({cells[3]})\n"
                else:
                    result += f"   {key}: {value}\n"
        
        # 리스트 항목
        elif line.startswith('*'):
            item = line[1:].strip()
            result += f"  • {item}\n"
    
    # 📝 한글 요약 추가 (LLM 없이 템플릿!)
    result += "\n" + "-" * 45 + "\n"
    result += "📝 요약: "
    
    if section_key == '청주_운영':
        result += f"청주 운영 DB ({service_name}) - {node_count}개 노드 RAC 구성"
    elif section_key == '청주_QA':
        result += f"청주 QA DB ({service_name}) - {node_count}개 노드 RAC 구성"
    elif section_key == '이천_운영':
        result += f"이천 운영 DB ({service_name}) - {node_count}개 노드 RAC 구성"
    elif section_key == '이천_QA':
        result += f"이천 QA DB ({service_name}) - {node_count}개 노드 RAC 구성"
    elif section_key == '계정':
        result += "STAREAD 계정으로 읽기 전용 접속"
    elif section_key == '요약':
        result += "청주/이천 운영/QA 4개 환경 접속 정보"
    elif section_key == 'Failover':
        result += "장애 시 5회 재시도 (5초 간격)"
    else:
        result += "STAR DB 접속 정보"
    
    result += "\n"
    
    return result


def search(query: str) -> Tuple[Optional[str], str]:
    """STAR DB 검색 → 바로 포맷팅된 텍스트 반환"""
    global _sections
    
    if _sections is None:
        if not load_md():
            return None, "❌ STAR_READ.md 파일을 로드할 수 없습니다."
    
    query_lower = query.lower()
    
    # 사이트 + 환경 감지
    is_cheongju = any(k in query_lower for k in ['청주', 'cheongju', 'cj'])
    is_icheon = any(k in query_lower for k in ['이천', 'icheon', 'ic'])
    is_qa = any(k in query_lower for k in ['qa', '테스트', 'test'])
    is_prod = any(k in query_lower for k in ['운영', 'prod', '실서버'])
    
    # 계정 정보
    account = _sections.get('계정', '')
    
    # 섹션 선택
    if is_cheongju and is_qa:
        section_key = '청주_QA'
    elif is_cheongju:
        section_key = '청주_운영'
    elif is_icheon and is_qa:
        section_key = '이천_QA'
    elif is_icheon:
        section_key = '이천_운영'
    elif any(k in query_lower for k in ['계정', '비밀번호', 'password']):
        section_key = '계정'
    elif any(k in query_lower for k in ['failover', '페일오버', '재시도']):
        section_key = 'Failover'
    else:
        section_key = '요약'
    
    section = _sections.get(section_key, '')
    
    if not section:
        return None, "❌ 관련 정보를 찾을 수 없습니다."
    
    # 계정 + 섹션 합치기
    if section_key not in ['계정', 'Failover', '요약']:
        combined = account + "\n\n" + section
    else:
        combined = section
    
    # 포맷팅해서 반환
    formatted = format_result(section_key, combined)
    
    return section_key, formatted


def is_star_query(query: str) -> bool:
    """STAR DB 관련 쿼리인지 판단 - 명확한 키워드만!"""
    query_lower = query.lower()
    
    # STAR 전용 키워드 (이천/청주는 MongoDB에도 있으니 제외!)
    star_keywords = [
        'star', '스타', 'smartstar', 'smart star',
        'oracle', '오라클', 'rac',
        'staread', 'fc1star', 'icastar',
        'fc1starpp', 'icastarpp',  # Service Name
        'tns', '1521'  # Oracle 포트
    ]
    
    # STAR 키워드가 있으면 True
    if any(k in query_lower for k in star_keywords):
        return True
    
    # "이천/청주" + "DB/접속/운영/QA" 조합이면서 mongo/logpresso 없으면 STAR
    location_keywords = ['이천', '청주', 'icheon', 'cheongju']
    db_keywords = ['db', '데이터베이스', '접속', '운영', 'qa', '계정', '비밀번호']
    mongo_keywords = ['mongo', '몽고', 'logpresso', '로그프레소', '로그', 'pkt', 'nwt', 'm11', 'm15']
    
    has_location = any(k in query_lower for k in location_keywords)
    has_db = any(k in query_lower for k in db_keywords)
    has_mongo = any(k in query_lower for k in mongo_keywords)
    
    # 지역 + DB키워드 있고, MongoDB 키워드 없으면 → STAR
    if has_location and has_db and not has_mongo:
        return True
    
    return False


def get_summary() -> str:
    """전체 요약 정보 반환"""
    global _sections
    if _sections is None:
        if not load_md():
            return "STAR DB 정보 없음"
            
    return _sections.get('요약', "STAR DB 정보 없음")


def get_full_content() -> str:
    """전체 MD 내용 반환"""
    global _raw_content
    if _raw_content is None:
        if not load_md():
            return "STAR DB 정보 없음"
    return _raw_content


# 테스트
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("STAR DB 검색 테스트")
    print("=" * 60)
    
    for q in ["청주 운영", "청주 QA", "이천 운영", "이천 QA", "계정", "STAR DB"]:
        print(f"\n🔍 쿼리: {q}")
        print("-" * 40)
        key, result = search(q)
        print(result)