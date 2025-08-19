# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 21:30:45 2025

@author: ggg3g
"""

import os
import re
import sys
import time
import json
import base64
import queue
import threading
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging

# ------------- spaCy 관련 라이브러리 -------------
import spacy
from spacy.matcher import Matcher

# ------------- LLM & Embeddings (LangChain) -------------
from langchain_community.llms import LlamaCpp
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document

# ------------- GUI & Graph -------------
import customtkinter as ctk
from tkinter.filedialog import askopenfilename, askopenfilenames
from tkinter import messagebox, ttk, simpledialog
import fitz  # PDF 텍스트 추출
import pytesseract
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ------------- HTMLFrame (TkinterWeb) -------------
import webbrowser
import tempfile
try:
    from tkinterweb import HtmlFrame
    TKINTERWEB_AVAILABLE = True
except ImportError:
    TKINTERWEB_AVAILABLE = False

try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False
    markdown = None

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# =============================================================================
# --- spaCy 기반 자연어 쿼리 파서 ---
# =============================================================================
class SpacyQueryParser:
    """spaCy를 사용한 자연어 쿼리 파서 - 개선된 버전"""
    
    def __init__(self):
        """spaCy 모델 로드 및 매처 설정"""
        # spaCy 한국어 모델 로드
        self.nlp = spacy.load("ko_core_news_sm")
        
        # 매처 초기화
        self.matcher = Matcher(self.nlp.vocab)
        
        # 사업장 이름-코드 매핑
        self.mfg_name_to_code = {
            "본사": "1", "대불": "3", "장생포": "4", "온산": "5", 
            "모화": "6", "해양": "7", "용연": "8", "HHI": "H", 
            "ZVEZDA": "Z", "벡터": "V"
        }
        
        # 필드 그룹 매핑
        self.field_mapping = {
            '사업장': 'MFG_IND', 
            '호선': 'SHIP_NO', 
            '블록': 'BLOCK', 
            '스테이지': 'STG_CD',
            '작업상태': 'STOCK_KIND',
            '날짜': 'STOCK_DATE'
        }
        
        # 패턴 설정
        self._setup_patterns()
    
    def _setup_patterns(self):
        """spaCy 매처에 패턴 추가 - 패턴 개선"""
        # 1. 날짜 패턴 (YYYY-MM-DD, YYYY년 MM월 DD일)
        self.matcher.add("STOCK_DATE", [
            [{"SHAPE": "dddd-dd-dd"}],
            [{"TEXT": {"REGEX": r"\d{4}년"}}, {"TEXT": {"REGEX": r"\d{1,2}월"}}, {"TEXT": {"REGEX": r"\d{1,2}일"}}],
            [{"TEXT": {"REGEX": r"\d{4}년"}}, {"TEXT": {"REGEX": r"\d{1,2}월"}}],  # 년월만 있는 경우 추가
            [{"TEXT": {"REGEX": r"\d{4}-\d{2}"}}]  # YYYY-MM 형식 추가
        ])
        
        # 2. CASE_NO 패턴 (HMD 관련)
        self.matcher.add("CASE_NO_HMD", [
            [{"LOWER": "case_no"}, {"OP": "?"}, {"TEXT": ":"}, {"TEXT": {"REGEX": r".*HMD.*"}}],
            [{"LOWER": "케이스"}, {"OP": "?"}, {"TEXT": ":"}, {"TEXT": {"REGEX": r".*HMD.*"}}],
            [{"TEXT": {"REGEX": r".*HMD.*"}}, {"LOWER": "인"}, {"OP": "?"}, {"LOWER": "데이터"}],
            [{"TEXT": {"REGEX": r"HMD.*"}}, {"LOWER": "로"}, {"LOWER": "시작하는"}],
            [{"TEXT": {"REGEX": r".*HMD"}}, {"LOWER": "로"}, {"LOWER": "끝나는"}],
            [{"TEXT": {"REGEX": r".*HMD.*"}}, {"LOWER": "가"}, {"LOWER": "포함된"}],
            [{"TEXT": {"REGEX": r".*HMD.*"}}]  # HMD만 언급된 경우도 추가
        ])
        
        # 3. 호선 패턴 - 정확히 6자리 숫자만 매칭
        self.matcher.add("SHIP_NO", [
            [{"LOWER": "호선"}, {"TEXT": ":"}, {"TEXT": {"REGEX": r"\d{6}"}}],
            [{"LOWER": "호선"}, {"TEXT": {"REGEX": r"\d{6}"}}],
            [{"TEXT": {"REGEX": r"\d{6}"}}, {"LOWER": "호선"}],  # 숫자가 앞에 오는 경우
            [{"LOWER": "ship_no"}, {"TEXT": ":"}, {"TEXT": {"REGEX": r"\d{6}"}}],
            [{"LOWER": "선박"}, {"LOWER": "번호"}, {"TEXT": {"REGEX": r"\d{6}"}}],
            [{"TEXT": {"REGEX": r"\d{6}"}}]  # 6자리 숫자만 있는 경우 (호선 번호로 간주)
        ])
        
        # 4. 블록 패턴 - 다양한 형태 지원
        self.matcher.add("BLOCK", [
            [{"LOWER": "블록"}, {"TEXT": ":"}, {"TEXT": {"REGEX": r"[A-Z0-9-]+"}}],
            [{"LOWER": "블록"}, {"TEXT": {"REGEX": r"[A-Z0-9-]+"}}],
            [{"TEXT": {"REGEX": r"[A-Z0-9-]+"}}, {"LOWER": "블록"}],  # 블록 코드가 앞에 오는 경우
            [{"LOWER": "block"}, {"TEXT": ":"}, {"TEXT": {"REGEX": r"[A-Z0-9-]+"}}],
            [{"TEXT": {"REGEX": r"\d{1,2}[A-Z]\d{1,2}"}}]  # 일반적인 블록 코드 패턴 (예: 1B11)
        ])
        
        # 5. M_ACT 패턴
        self.matcher.add("M_ACT", [
            [{"LOWER": "m_act"}, {"TEXT": ":"}, {"TEXT": {"REGEX": r"[A-Z0-9-]+"}}],
            [{"LOWER": "엠액트"}, {"TEXT": ":"}, {"TEXT": {"REGEX": r"[A-Z0-9-]+"}}],
            [{"LOWER": "m_act"}, {"TEXT": {"REGEX": r"[A-Z0-9-]+"}}],
            [{"LOWER": "엠액트"}, {"TEXT": {"REGEX": r"[A-Z0-9-]+"}}]
        ])
        
        # 6. 사업장 패턴 - 다양한 형태 지원
        site_patterns = []
        for site_name in self.mfg_name_to_code.keys():
            # 사업장 + 이름
            site_patterns.append([{"LOWER": "사업장"}, {"TEXT": site_name}])
            # 이름 + 사업장
            site_patterns.append([{"TEXT": site_name}, {"LOWER": "사업장"}])
            # 이름 + '의' 패턴
            site_patterns.append([{"TEXT": site_name}, {"LOWER": "의"}])
            # 이름 + '에' 패턴
            site_patterns.append([{"TEXT": site_name}, {"LOWER": "에"}])
            # 이름 + '에서' 패턴
            site_patterns.append([{"TEXT": site_name}, {"LOWER": "에서"}])
            # 이름만 있는 경우
            site_patterns.append([{"TEXT": site_name}])
        
        # 코드 직접 지정 패턴
        site_patterns.append([{"LOWER": "mfg_ind"}, {"TEXT": ":"}, {"TEXT": {"REGEX": r"\d|[A-Z]"}}])
        
        self.matcher.add("MFG_IND", site_patterns)
        
        # 7. 타 사업장 패턴
        self.matcher.add("TO_MFG_IND", [
            [{"LOWER": "타사업장"}, {"TEXT": ":"}, {"TEXT": {"IN": list(self.mfg_name_to_code.keys())}}],
            [{"LOWER": "타사업장"}, {"TEXT": {"IN": list(self.mfg_name_to_code.keys())}}],
            [{"LOWER": "타"}, {"LOWER": "사업장"}, {"TEXT": {"IN": list(self.mfg_name_to_code.keys())}}],
            [{"LOWER": "to_mfg_ind"}, {"TEXT": ":"}, {"TEXT": {"REGEX": r"\d|[A-Z]"}}]
        ])
        
        # 8. 스테이지 코드 패턴 - 다양한 형태 지원
        self.matcher.add("STG_CD", [
            [{"LOWER": "스테이지"}, {"TEXT": ":"}, {"SHAPE": "d"}],
            [{"LOWER": "스테이지"}, {"SHAPE": "d"}],
            [{"SHAPE": "d"}, {"LOWER": "스테이지"}],  # 숫자가 앞에 오는 경우
            [{"LOWER": "stg_cd"}, {"TEXT": ":"}, {"SHAPE": "d"}],
            [{"LOWER": "stage"}, {"TEXT": ":"}, {"SHAPE": "d"}],
            [{"LOWER": "stage"}, {"SHAPE": "d"}],
            [{"LOWER": "최종"}, {"LOWER": "스테이지"}]  # 최종 스테이지 (90으로 처리)
        ])
        
        # 9. 작업 상태 패턴 - 다양한 표현 지원
        self.matcher.add("STOCK_KIND", [
            [{"LOWER": "작업중"}],
            [{"LOWER": "작업"}, {"LOWER": "중"}],
            [{"LOWER": "대기중"}],
            [{"LOWER": "대기"}, {"LOWER": "중"}],
            [{"LOWER": "작업"}, {"LOWER": "상태"}, {"LOWER": "작업중"}],
            [{"LOWER": "작업"}, {"LOWER": "상태"}, {"LOWER": "대기중"}],
            [{"LOWER": "상태"}, {"LOWER": "작업중"}],
            [{"LOWER": "상태"}, {"LOWER": "대기중"}]
        ])
        
        # 10. 총 스톡 패턴
        self.matcher.add("TOTAL_STOCK", [
            [{"LOWER": "총"}, {"LOWER": "스톡"}],
            [{"LOWER": "합계"}],
            [{"LOWER": "총합"}],
            [{"LOWER": "총"}, {"LOWER": "개수"}],
            [{"LOWER": "전체"}, {"LOWER": "개수"}],
            [{"LOWER": "스톡"}, {"LOWER": "수"}]
        ])
        
        # 11. 그룹화 패턴
        self.matcher.add("GROUP_BY", [
            [{"TEXT": {"REGEX": r"[\w가-힣]+"}, "OP": "+"}, {"LOWER": "별"}],
            [{"LOWER": "그룹화"}, {"TEXT": {"REGEX": r"[\w가-힣]+"}}],
            [{"TEXT": {"REGEX": r"[\w가-힣]+"}, "OP": "+"}, {"LOWER": "기준"}],
            [{"LOWER": "기준"}, {"TEXT": {"REGEX": r"[\w가-힣]+"}, "OP": "+"}]
        ])
    
    def parse_query(self, query):
        """쿼리 텍스트 파싱하여 파라미터 추출 - 개선된 버전"""
        params = {}
        logger.debug(f"원본 쿼리: '{query}'")
        
        # 특정 패턴 검출을 위한 사전 처리
        # 정확히 6자리 숫자만 있는 경우 호선 번호로 간주
        ship_direct_match = re.search(r'\b(\d{6})\b', query)
        if ship_direct_match:
            ship_no = ship_direct_match.group(1)
            params["SHIP_NO"] = int(ship_no)
            logger.debug(f"직접 호선 숫자 매칭: {ship_no}")
        
        # 블록 코드 직접 매칭 (nBnn 패턴)
        block_direct_match = re.search(r'\b(\d{1,2}[A-Z]\d{1,2})\b', query, re.IGNORECASE)
        if block_direct_match:
            block = block_direct_match.group(1)
            params["BLOCK"] = block.upper()
            logger.debug(f"직접 블록 코드 매칭: {block}")
        
        # 사업장 이름 직접 매칭
        for name, code in self.mfg_name_to_code.items():
            # 다양한 형태의 사업장 언급 검출
            site_patterns = [
                fr'\b{name}\s*사업장\b',  # 본사 사업장
                fr'\b사업장\s*{name}\b',  # 사업장 본사
                fr'\b{name}의\b',         # 본사의
                fr'\b{name}에\b',         # 본사에
                fr'\b{name}에서\b',       # 본사에서
                fr'\b{name}\b'            # 본사 (단독)
            ]
            
            if any(re.search(pattern, query, re.IGNORECASE) for pattern in site_patterns):
                logger.debug(f"사업장 이름 직접 매칭: {name}, 코드: {code}")
                params["MFG_IND"] = code
                break
        
        # 호선 직접 매칭 (정확히 6자리 숫자)
        ship_patterns = [
            r'호선\s*(\d{6})',            # 호선 838906
            r'(\d{6})\s*호선',            # 838906 호선
            r'호선[:\s]+(\d{6})',         # 호선: 838906
            r'SHIP_NO[:\s]+(\d{6})'       # SHIP_NO: 838906
        ]
        
        for pattern in ship_patterns:
            ship_match = re.search(pattern, query)
            if ship_match and "SHIP_NO" not in params:
                ship_no = ship_match.group(1)
                params["SHIP_NO"] = int(ship_no)
                logger.debug(f"호선 패턴 매칭: {pattern} -> {ship_no}")
                break
        
        # 블록 직접 매칭 (다양한 패턴)
        block_patterns = [
            r'블록\s*([A-Z0-9-]+)',       # 블록 1B11
            r'([A-Z0-9-]+)\s*블록',       # 1B11 블록
            r'블록[:\s]+([A-Z0-9-]+)',    # 블록: 1B11
            r'BLOCK[:\s]+([A-Z0-9-]+)'    # BLOCK: 1B11
        ]
        
        for pattern in block_patterns:
            block_match = re.search(pattern, query, re.IGNORECASE)
            if block_match and "BLOCK" not in params:
                block = block_match.group(1)
                params["BLOCK"] = block.upper()
                logger.debug(f"블록 패턴 매칭: {pattern} -> {block}")
                break
        
        # 작업 상태 직접 매칭
        if any(status in query for status in ["작업중", "작업 중"]):
            params["STOCK_KIND"] = "작업중"
            logger.debug("작업중 상태 매칭")
        elif any(status in query for status in ["대기중", "대기 중"]):
            params["STOCK_KIND"] = "대기중"
            logger.debug("대기중 상태 매칭")
        
        # 날짜 직접 매칭
        date_patterns = [
            r'(\d{4})[년-](\d{1,2})[월-](\d{1,2})일?',  # YYYY년MM월DD일 or YYYY-MM-DD
            r'(\d{4})[년-](\d{1,2})월?'                # YYYY년MM월 or YYYY-MM
        ]
        
        for pattern in date_patterns:
            date_match = re.search(pattern, query)
            if date_match:
                if len(date_match.groups()) == 3:
                    year, month, day = date_match.groups()
                    formatted_date = f"{year}-{int(month):02d}-{int(day):02d}"
                    params["STOCK_DATE"] = formatted_date
                    logger.debug(f"상세 날짜 매칭: {formatted_date}")
                elif len(date_match.groups()) == 2:
                    year, month = date_match.groups()
                    # 월 단위 검색은 날짜 범위로 변환해야 함 - 여기서는 일단 month_query 플래그 설정
                    params["month_query"] = True
                    params["year"] = int(year)
                    params["month"] = int(month)
                    logger.debug(f"월 단위 검색: {year}년 {month}월")
                break
        
        # 스테이지 직접 매칭
        stage_patterns = [
            r'스테이지\s*(\d+)',          # 스테이지 80
            r'(\d+)\s*스테이지',          # 80 스테이지
            r'스테이지[:\s]+(\d+)',       # 스테이지: 80
            r'STG_CD[:\s]+(\d+)'          # STG_CD: 80
        ]
        
        for pattern in stage_patterns:
            stage_match = re.search(pattern, query)
            if stage_match and "STG_CD" not in params:
                stage = stage_match.group(1)
                params["STG_CD"] = stage
                logger.debug(f"스테이지 패턴 매칭: {pattern} -> {stage}")
                break
        
        # 최종 스테이지 처리
        if "최종 스테이지" in query or "최종스테이지" in query:
            params["STG_CD"] = "90"
            logger.debug("최종 스테이지(90) 매칭")
        
        # spaCy를 사용한 패턴 매칭 - 직접 매칭으로 찾지 못한 경우에 보완
        doc = self.nlp(query)
        matches = self.matcher(doc)
        
        for match_id, start, end in matches:
            match_text = doc[start:end].text
            rule_name = self.nlp.vocab.strings[match_id]
            logger.debug(f"spaCy 매칭된 패턴: {rule_name}, 텍스트: {match_text}")
            
            # 이미 직접 매칭으로 처리된 필드는 건너뜀
            if rule_name == "STOCK_DATE" and "STOCK_DATE" not in params:
                date_text = match_text
                year_month_day = re.search(r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일', date_text)
                if year_month_day:
                    year, month, day = year_month_day.groups()
                    date_text = f"{year}-{int(month):02d}-{int(day):02d}"
                params["STOCK_DATE"] = date_text
                
            elif rule_name == "CASE_NO_HMD" and "CASE_NO" not in params:
                hmd_pattern = re.search(r'(.*HMD.*)', match_text)
                if hmd_pattern:
                    params["CASE_NO"] = hmd_pattern.group(1)
                else:
                    params["CASE_NO"] = "HMD"
                if "시작하는" in match_text:
                    params["HMD_SEARCH_TYPE"] = "startswith"
                elif "끝나는" in match_text:
                    params["HMD_SEARCH_TYPE"] = "endswith"
                else:
                    params["HMD_SEARCH_TYPE"] = "contains"
                    
            elif rule_name == "SHIP_NO" and "SHIP_NO" not in params:
                ship_no = re.search(r'(\d{6})', match_text)
                if ship_no:
                    params["SHIP_NO"] = int(ship_no.group(1))
                    
            elif rule_name == "BLOCK" and "BLOCK" not in params:
                block = re.search(r'([A-Z0-9-]+)', match_text, re.IGNORECASE)
                if block:
                    params["BLOCK"] = block.group(1).upper()
                    
            elif rule_name == "M_ACT" and "M_ACT" not in params:
                m_act = re.search(r'([A-Z0-9-]+)', match_text, re.IGNORECASE)
                if m_act:
                    params["M_ACT"] = m_act.group(1).upper()
                    
            elif rule_name == "MFG_IND" and "MFG_IND" not in params:
                for name in self.mfg_name_to_code.keys():
                    if name in match_text:
                        params["MFG_IND"] = self.mfg_name_to_code[name]
                        break
                mfg_code = re.search(r'(\d|[A-Z])', match_text)
                if "MFG_IND" not in params and mfg_code:
                    params["MFG_IND"] = mfg_code.group(1)
                    
            elif rule_name == "TO_MFG_IND" and "TO_MFG_IND" not in params:
                for name in self.mfg_name_to_code.keys():
                    if name in match_text:
                        params["TO_MFG_IND"] = self.mfg_name_to_code[name]
                        break
                to_mfg_code = re.search(r'(\d|[A-Z])', match_text)
                if "TO_MFG_IND" not in params and to_mfg_code:
                    params["TO_MFG_IND"] = to_mfg_code.group(1)
                    
            elif rule_name == "STG_CD" and "STG_CD" not in params:
                stg_cd = re.search(r'(\d+)', match_text)
                if stg_cd:
                    params["STG_CD"] = stg_cd.group(1)
                elif "최종" in match_text:
                    params["STG_CD"] = "90"  # 최종 스테이지는 90으로 설정
                    
            elif rule_name == "STOCK_KIND" and "STOCK_KIND" not in params:
                if "작업" in match_text:
                    params["STOCK_KIND"] = "작업중"
                elif "대기" in match_text:
                    params["STOCK_KIND"] = "대기중"
                    
            elif rule_name == "TOTAL_STOCK":
                params["TOTAL_STOCK"] = True
                    
            elif rule_name == "GROUP_BY" and "GROUP_BY" not in params:
                for field_name in self.field_mapping.keys():
                    if field_name in match_text:
                        params["GROUP_BY"] = self.field_mapping[field_name]
                        break
        
        # 월 단위 쿼리 처리
        if "month_query" in params:
            year = params["year"]
            month = params["month"]
            # 월의 첫날과 마지막 날 계산
            if month == 12:
                next_year = year + 1
                next_month = 1
            else:
                next_year = year
                next_month = month + 1
            
            start_date = f"{year}-{month:02d}-01"
            end_date = f"{next_year}-{next_month:02d}-01"
            
            params["date_range"] = (start_date, end_date)
            # 임시 플래그 제거
            del params["month_query"]
            del params["year"]
            del params["month"]
        
        logger.debug(f"최종 추출된 파라미터: {params}")
        
        # 파라미터가 비어있으면 경고 로그 추가
        if not params:
            logger.warning("파라미터가 추출되지 않았습니다. 전체 데이터가 반환될 수 있습니다.")
        
        return params

    def apply_filters(self, df, params):
        """파라미터를 기반으로 데이터프레임에 필터 적용 - 개선된 버전"""
        if df.empty:
            return df, "데이터가 비어 있습니다."
        
        # 파라미터가 비어있으면 빈 데이터프레임과 경고 메시지 반환
        if not params:
            empty_df = pd.DataFrame(columns=df.columns)
            return empty_df, "추출된 검색 조건이 없습니다. 구체적인 검색어를 입력해주세요."
        
        filtered_df = df.copy()
        # 문자열 컬럼 전처리
        string_columns = ['CASE_NO', 'M_ACT', 'BLOCK', 'MFG_IND', 'TO_MFG_IND', 'STG_CD', 'TO_STG_CD', 'STOCK_KIND']
        for col in string_columns:
            if col in filtered_df.columns:
                filtered_df[col] = filtered_df[col].astype(str).replace('nan', '')
        
        # 날짜 범위 처리 (월 단위 쿼리 등)
        if "date_range" in params:
            start_date, end_date = params["date_range"]
            if 'STOCK_DATE' in filtered_df.columns:
                filtered_df = filtered_df[(filtered_df['STOCK_DATE'] >= start_date) & (filtered_df['STOCK_DATE'] < end_date)]
                logger.debug(f"날짜 범위 필터 적용: {start_date} ~ {end_date}")
        # 단일 날짜 처리
        elif 'STOCK_DATE' in params and 'STOCK_DATE' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['STOCK_DATE'] == params['STOCK_DATE']]
            logger.debug(f"날짜 필터 적용: {params['STOCK_DATE']}")
        
        # CASE_NO 처리 (HMD 관련)
        if 'CASE_NO' in params:
            if 'CASE_NO' not in filtered_df.columns:
                return filtered_df, "데이터에 CASE_NO 열이 없습니다."
            
            case_no = params['CASE_NO']
            search_type = params.get('HMD_SEARCH_TYPE', 'contains')
            
            if case_no == 'HMD':
                if search_type == 'startswith':
                    filtered_df = filtered_df[filtered_df['CASE_NO'].str.upper().str.startswith('HMD')]
                elif search_type == 'endswith':
                    filtered_df = filtered_df[filtered_df['CASE_NO'].str.upper().str.endswith('HMD')]
                else:
                    filtered_df = filtered_df[filtered_df['CASE_NO'].str.upper().str.contains('HMD', na=False)]
            else:
                if search_type == 'startswith':
                    filtered_df = filtered_df[filtered_df['CASE_NO'].str.upper().str.startswith(case_no.upper())]
                elif search_type == 'endswith':
                    filtered_df = filtered_df[filtered_df['CASE_NO'].str.upper().str.endswith(case_no.upper())]
                else:
                    filtered_df = filtered_df[filtered_df['CASE_NO'].str.upper().str.contains(case_no.upper(), na=False)]
            
            logger.debug(f"CASE_NO 필터 적용: {case_no} (검색 타입: {search_type})")
        
        # 일반 필드 필터링
        field_filters = {
            'SHIP_NO': lambda df, val: df[df['SHIP_NO'] == val],
            'BLOCK': lambda df, val: df[df['BLOCK'].str.upper() == val.upper()],
            'M_ACT': lambda df, val: df[df['M_ACT'].str.upper() == val.upper()],
            'MFG_IND': lambda df, val: df[df['MFG_IND'] == val],  # 코드로 직접 비교
            'TO_MFG_IND': lambda df, val: df[df['TO_MFG_IND'] == val],
            'STG_CD': lambda df, val: df[df['STG_CD'] == val],
            'TO_STG_CD': lambda df, val: df[df['TO_STG_CD'] == val],
            'STOCK_KIND': lambda df, val: df[df['STOCK_KIND'] == val]
        }
        
        # 필터 적용 및 로깅
        for field, filter_func in field_filters.items():
            if field in params and field in filtered_df.columns:
                original_count = len(filtered_df)
                filtered_df = filter_func(filtered_df, params[field])
                after_count = len(filtered_df)
                logger.debug(f"{field} 필터 적용: {params[field]} (필터링 전: {original_count}, 후: {after_count})")
                
                # 필터링 후 데이터가 없으면 경고
                if after_count == 0:
                    return filtered_df, f"{field}: {params[field]} 조건에 맞는 데이터가 없습니다."
        
        # 결과 로그 기록
        logger.debug(f"최종 필터링 결과: {len(filtered_df)}개 항목")
        
        return filtered_df, None
    
    def format_results(self, filtered_df, params):
        """필터링된 결과 포맷팅 - 개선된 버전"""
        if filtered_df.empty:
            param_str = ', '.join([f'{k}: {v}' for k, v in params.items() 
                                 if k not in ['TOTAL_STOCK', 'GROUP_BY', 'HMD_SEARCH_TYPE', 'date_range']])
            return f"조건에 맞는 데이터가 없습니다. 검색 조건: {param_str}"
        
        # 결과 구성
        results = []
        total_stock = filtered_df['STOCK'].sum() if 'STOCK' in filtered_df.columns else 0
        
        # 검색 조건 문자열 생성
        condition_parts = []
        for k, v in params.items():
            if k in ['TOTAL_STOCK', 'GROUP_BY', 'HMD_SEARCH_TYPE', 'date_range']:
                continue
                
            # 사업장 코드를 이름으로 변환
            if k == 'MFG_IND' or k == 'TO_MFG_IND':
                for name, code in self.mfg_name_to_code.items():
                    if code == v:
                        condition_parts.append(f"{k}: {name}({v})")
                        break
                else:
                    condition_parts.append(f"{k}: {v}")
            else:
                condition_parts.append(f"{k}: {v}")
        
        # 날짜 범위 처리
        if 'date_range' in params:
            start, end = params['date_range']
            condition_parts.append(f"날짜 범위: {start} ~ {end}")
            
        condition_str = ', '.join(condition_parts)
        
        # 헤더 정보
        results.append("## 검색 결과 요약")
        results.append(f"* 총 항목 수: {len(filtered_df)}개")
        results.append(f"* 총 스톡 수: {total_stock}개")
        results.append(f"* 검색 조건: {condition_str}")
        
        # 대량 결과 처리
        display_df = filtered_df
        is_truncated = False
        if len(filtered_df) > 100:
            results.append(f"* 주의: 검색 결과가 많습니다 ({len(filtered_df)}개). 처음 100개만 표시합니다.")
            display_df = filtered_df.head(100)
            is_truncated = True
        
        # 그룹화 결과 표시
        if 'GROUP_BY' in params:
            group_field = params['GROUP_BY']
            if group_field in filtered_df.columns:
                # 그룹화 및 합계 계산
                group_stats = filtered_df.groupby(group_field)['STOCK'].sum().reset_index()
                group_stats = group_stats.sort_values('STOCK', ascending=False)  # 스톡 수 기준 내림차순
                
                results.append(f"\n## {group_field} 별 스톡 통계")
                for _, row in group_stats.iterrows():
                    # 사업장 코드인 경우 이름으로 변환
                    if group_field == 'MFG_IND' or group_field == 'TO_MFG_IND':
                        field_value = row[group_field]
                        for name, code in self.mfg_name_to_code.items():
                            if code == field_value:
                                field_value = f"{name}({code})"
                                break
                    else:
                        field_value = row[group_field]
                        
                    results.append(f"* {group_field}: {field_value}, 스톡 수: {row['STOCK']}개")
            else:
                results.append(f"\n* 그룹화 필드 {group_field}가 데이터에 존재하지 않습니다.")
        
        # 상세 데이터 표시
        total_displayed = len(display_df)
        results.append(f"\n## 상세 데이터 (총 {len(filtered_df)}개 중 {total_displayed}개 표시)")
        
        # 날짜별로 정렬 (가능한 경우)
        if 'STOCK_DATE' in display_df.columns:
            display_df = display_df.sort_values('STOCK_DATE', ascending=False)
        
        for _, row in display_df.iterrows():
            # 기본 정보
            line_parts = []
            
            # 날짜 정보 (있는 경우)
            if 'STOCK_DATE' in row:
                line_parts.append(f"날짜: {row['STOCK_DATE']}")
                
            # 호선 정보 (있는 경우)
            if 'SHIP_NO' in row:
                line_parts.append(f"호선: {row['SHIP_NO']}")
                
            # 블록 정보 (있는 경우)
            if 'BLOCK' in row:
                line_parts.append(f"블록: {row['BLOCK']}")
            
            # 기타 주요 필드
            for field in ['CASE_NO', 'M_ACT', 'MFG_IND', 'TO_MFG_IND', 'STG_CD', 'TO_STG_CD', 'STOCK_KIND', 'STOCK']:
                if field in row and not pd.isna(row[field]) and row[field] != '':
                    field_value = row[field]
                    
                    # 사업장 코드인 경우 이름으로 변환
                    if (field == 'MFG_IND' or field == 'TO_MFG_IND') and isinstance(field_value, str):
                        # 코드가 숫자나 문자인 경우 매핑된 이름 찾기
                        for name, code in self.mfg_name_to_code.items():
                            if code == field_value:
                                field_value = f"{name}({code})"
                                break
                        # 이미 이름인 경우 해당 코드 찾기
                        if field_value in self.mfg_name_to_code:
                            code = self.mfg_name_to_code[field_value]
                            field_value = f"{field_value}({code})"
                    
                    field_labels = {
                        'CASE_NO': 'CASE_NO',
                        'M_ACT': 'M_ACT',
                        'MFG_IND': '사업장',
                        'TO_MFG_IND': '타사업장',
                        'STG_CD': '스테이지',
                        'TO_STG_CD': '타겟 스테이지',
                        'STOCK_KIND': '작업 상태',
                        'STOCK': '스톡 수'
                    }
                    field_label = field_labels.get(field, field)
                    line_parts.append(f"{field_label}: {field_value}")
            
            line = "* " + ", ".join(line_parts)
            results.append(line)
            
        # 결과 문자열 반환
        return "\n".join(results)

# =============================================================================
# --- LangChainEquipmentSystem (RAG 시스템) ---
# =============================================================================
class LangChainEquipmentSystem:
    def __init__(self, model_path, load_existing=False):
        print(f"모델 '{model_path}' 로드 중...")
        self.equipment_code = None
        
        # csv_data 및 faiss_index 디렉토리 존재 여부 확인 및 생성
        self._ensure_directories_exist()
        
        context_size = 2048
        self.llm = LlamaCpp(
            model_path=str(model_path),
            n_ctx=context_size,
            n_batch=512,
            use_mlock=True,
            n_gpu_layers=64,
            device='CPU',#CPU,cuda
            tensor_split=[],
            rope_scaling={"type": "linear", "factor": 2.0},
            library_path=r"C:\llama.cpp\build\bin\Release\llama.dll"
        )
    
        print("임베딩 모델 로드 중...")
        self.embedding_model = HuggingFaceEmbeddings(model_name="C:/Python/AGI/koelectra-model")
        
        # 프롬프트 템플릿을 먼저 정의합니다.
        prompt_template = """
        당신은 데이터 분석 전문가입니다. 아래 제공된 정보를 바탕으로 질문에 정확하고 간결하게 답변하세요.
        
        ### 데이터
        {context}
        
        ### 사용자 질문
        {question}
        
        아래 가이드라인을 따라 답변해주세요:
        1. 가독성을 위해 날짜 형식을 그대로 유지해주세요 (예: 2025년 1월 1일 또는 2025-06-18).
        2. 수치는 적절한 소수점 자리까지 표시해주세요.
        3. 계산이 필요한 경우 정확히 계산해서 알려주세요.
        4. 특정 조건이 언급된 경우 해당 정보만 제공해주세요.
        5. 데이터 소스가 CSV인지 확인하고, 요청에 따라 해당 데이터만 사용하세요.
        6. 결과는 "결과값:"과 "분석:"으로 나누어 제시하며, 분석은 간단한 핵심 인사이트만 포함해주세요.
        
        ### 응답
        """
        self.prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        
        self.csv_df = pd.DataFrame()
        self.df = pd.DataFrame()
        self.all_csv_data = []
        self.spacy_parser = None  # spaCy 쿼리 파서 초기화
    
        # 데이터 로드 전에 벡터 스토어 관련 변수는 아직 준비되지 않은 상태이므로 분기 처리합니다.
        if load_existing and os.path.exists("faiss_index"):
            print("기존 FAISS 인덱스를 로드 중...")
            self.vector_store = FAISS.load_local("faiss_index", self.embedding_model, allow_dangerous_deserialization=True)
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 20})
            self.load_only_data()
        else:
            self.load_data()
        
        # RetrievalQA 체인 생성 (self.prompt이 이제 이미 존재합니다)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,  # 업데이트된 retriever 사용
            chain_type_kwargs={"prompt": self.prompt}
        )
        
    def extract_range_from_query(self, query):
        """쿼리에서 날짜 범위를 추출합니다."""
        # 날짜 형식: YYYY-MM-DD
        date_pattern = r'(\d{4}-\d{2}-\d{2})'
        dates = re.findall(date_pattern, query)
        
        # "부터", "까지", "에서", "~" 등의 표현으로 범위를 찾음
        if len(dates) >= 2 and any(keyword in query for keyword in ["부터", "까지", "에서", "~", "사이"]):
            return dates[0], dates[-1]  # 첫 번째와 마지막 날짜 반환
        
        return None, None
        
    def extract_csv_params(self, query):
        """CSV 데이터 쿼리에서 파라미터 추출"""
        if self.spacy_parser is None:
            self.spacy_parser = SpacyQueryParser()
        return self.spacy_parser.parse_query(query)    
    
    def load_data(self):
        # 디렉토리 존재 여부 확인
        csv_folder_path = "csv_data"
        if not os.path.exists(csv_folder_path):
            print(f"'{csv_folder_path}' 디렉토리가 존재하지 않습니다. 생성합니다.")
            try:
                os.makedirs(csv_folder_path)
                print(f"'{csv_folder_path}' 디렉토리 생성 완료. CSV 파일을 추가해주세요.")
                return  # 디렉토리가 비어있으므로 여기서 종료
            except Exception as e:
                print(f"'{csv_folder_path}' 디렉토리 생성 중 오류: {str(e)}")
                return
    
        mfg_mapping = {
            "1": "본사", "2": "대불", "3": "대불", "4": "장생포", "5": "온산", "6": "모화",
            "7": "해양", "8": "용연", "H": "HHI", "Z": "ZVEZDA", "V": "벡터"
        }
        stock_kind_mapping = {
            "21": "작업중", "30": "작업중", "25": "작업중", "26": "작업중",
            "23": "작업중", "12": "대기중", "11": "대기중"
        }
    
        self.all_csv_data = []
        csv_files = [f for f in os.listdir(csv_folder_path) if f.endswith(".csv")]
        for file in csv_files:
            file_path = os.path.join(csv_folder_path, file)
            try:
                df_csv = pd.read_csv(file_path, encoding='utf-8', sep=None, engine='python')
                df_csv.columns = [col.strip().upper() for col in df_csv.columns]
                if 'STOCK_DATE' not in df_csv.columns and 'SHIP_PLDT' in df_csv.columns:
                    df_csv['STOCK_DATE'] = df_csv['SHIP_PLDT']
                required_columns = ['SHIP_NO', 'BLOCK', 'MFG_IND', 'TO_MFG_IND', 'STG_CD', 'STOCK_KIND', 'STOCK']
                missing_columns = [col for col in required_columns if col not in df_csv.columns]
                if missing_columns:
                    print(f"경고: CSV 파일 {file}에 필수 열이 누락되었습니다: {missing_columns}")
                    continue
                df_csv['MFG_IND_NAME'] = df_csv['MFG_IND'].astype(str).map(mfg_mapping).fillna(df_csv['MFG_IND'])
                df_csv['TO_MFG_IND_NAME'] = df_csv['TO_MFG_IND'].astype(str).map(mfg_mapping).fillna(df_csv['TO_MFG_IND'])
                df_csv['STOCK_KIND'] = df_csv['STOCK_KIND'].astype(str).map(stock_kind_mapping).fillna(df_csv['STOCK_KIND'])
                if 'STOCK_DATE' in df_csv.columns:
                    df_csv['datetime'] = pd.to_datetime(df_csv['STOCK_DATE'], format="%Y-%m-%d", errors='coerce')
                if 'CASE_NO' not in df_csv.columns:
                    if 'M_ACT' in df_csv.columns:
                        df_csv['CASE_NO'] = df_csv['M_ACT']
                    else:
                        df_csv['CASE_NO'] = ''
                if 'M_ACT' not in df_csv.columns:
                    if 'CASE_NO' in df_csv.columns:
                        df_csv['M_ACT'] = df_csv['CASE_NO']
                    else:
                        df_csv['M_ACT'] = ''
                df_csv['CASE_NO'] = df_csv['CASE_NO'].astype(str).replace('nan', '')
                df_csv['M_ACT'] = df_csv['M_ACT'].astype(str).replace('nan', '')
                self.all_csv_data.append(df_csv)
            except Exception as e:
                print(f"CSV 파일 {file} 처리 중 오류 발생: {str(e)}")
                continue
    
        if self.all_csv_data:
            self.csv_df = pd.concat(self.all_csv_data, ignore_index=True)
            numeric_columns = ['SHIP_NO', 'STOCK', 'PE_OFFSET', 'STG_CD', 'TO_STG_CD']
            for col in numeric_columns:
                if col in self.csv_df.columns:
                    self.csv_df[col] = pd.to_numeric(self.csv_df[col], errors='coerce')
            if 'CASE_NO' not in self.csv_df.columns:
                if 'M_ACT' in self.csv_df.columns:
                    self.csv_df['CASE_NO'] = self.csv_df['M_ACT']
                else:
                    self.csv_df['CASE_NO'] = ''
            if 'M_ACT' not in self.csv_df.columns:
                if 'CASE_NO' in self.csv_df.columns:
                    self.csv_df['M_ACT'] = self.csv_df['CASE_NO']
                else:
                    self.csv_df['M_ACT'] = ''
            self.csv_df['CASE_NO'] = self.csv_df['CASE_NO'].astype(str).replace('nan', '')
            self.csv_df['M_ACT'] = self.csv_df['M_ACT'].astype(str).replace('nan', '')
        else:
            self.csv_df = pd.DataFrame()
    
        # CSV 데이터만으로 벡터 스토어 생성 (TXT 데이터 관련 문서는 제거)
        documents = []
        for _, row in self.csv_df.iterrows():
            content = f"호선: {row['SHIP_NO']}, 블록: {row['BLOCK']}, "
            if 'STOCK_DATE' in row:
                content += f"날짜: {row['STOCK_DATE']}, "
            content += f"사업장: {row['MFG_IND']}, 스테이지: {row['STG_CD']}, 스톡: {row['STOCK']}"
            if 'CASE_NO' in row and row['CASE_NO'] and row['CASE_NO'] != 'nan':
                content += f", CASE_NO: {row['CASE_NO']}"
            documents.append(Document(
                page_content=content,
                metadata={"source": "CSV"}
            ))
        print("벡터 스토어 생성 중...")
        self.vector_store = FAISS.from_documents(documents, self.embedding_model)
        self.vector_store.save_local("faiss_index")
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 20})
        # CSV 데이터를 주 데이터로 사용합니다.
        self.df = self.csv_df
    def _ensure_directories_exist(self):
        """필요한 디렉토리들의 존재 여부를 확인하고 없으면 생성합니다."""
        required_dirs = ["csv_data", "faiss_index"]
        for directory in required_dirs:
            if not os.path.exists(directory):
                print(f"'{directory}' 디렉토리가 존재하지 않습니다. 생성합니다.")
                try:
                    os.makedirs(directory)
                    print(f"'{directory}' 디렉토리 생성 완료.")
                except Exception as e:
                    print(f"'{directory}' 디렉토리 생성 중 오류: {str(e)}")    
    def load_only_data(self):
        """CSV 데이터만 로드하고 벡터 스토어는 생성하지 않음"""
        csv_folder_path = "csv_data"
    
        mfg_mapping = {
            "1": "본사", "2": "대불", "3": "대불", "4": "장생포", "5": "온산", "6": "모화",
            "7": "해양", "8": "용연", "H": "HHI", "Z": "ZVEZDA", "V": "벡터"
        }
        stock_kind_mapping = {
            "21": "작업중", "30": "작업중", "25": "작업중", "26": "작업중",
            "23": "작업중", "12": "대기중", "11": "대기중"
        }
    
        self.all_csv_data = []
        csv_files = [f for f in os.listdir(csv_folder_path) if f.endswith(".csv")]
        for file in csv_files:
            file_path = os.path.join(csv_folder_path, file)
            try:
                df_csv = pd.read_csv(file_path, encoding='utf-8', sep=None, engine='python')
                df_csv.columns = [col.strip().upper() for col in df_csv.columns]
                if 'STOCK_DATE' not in df_csv.columns and 'SHIP_PLDT' in df_csv.columns:
                    df_csv['STOCK_DATE'] = df_csv['SHIP_PLDT']
                required_columns = ['SHIP_NO', 'BLOCK', 'MFG_IND', 'TO_MFG_IND', 'STG_CD', 'STOCK_KIND', 'STOCK']
                missing_columns = [col for col in required_columns if col not in df_csv.columns]
                if missing_columns:
                    print(f"경고: CSV 파일 {file}에 필수 열이 누락되었습니다: {missing_columns}")
                    continue
                df_csv['MFG_IND_NAME'] = df_csv['MFG_IND'].astype(str).map(mfg_mapping).fillna(df_csv['MFG_IND'])
                df_csv['TO_MFG_IND_NAME'] = df_csv['TO_MFG_IND'].astype(str).map(mfg_mapping).fillna(df_csv['TO_MFG_IND'])
                df_csv['STOCK_KIND'] = df_csv['STOCK_KIND'].astype(str).map(stock_kind_mapping).fillna(df_csv['STOCK_KIND'])
                if 'STOCK_DATE' in df_csv.columns:
                    df_csv['datetime'] = pd.to_datetime(df_csv['STOCK_DATE'], format="%Y-%m-%d", errors='coerce')
                if 'CASE_NO' not in df_csv.columns:
                    if 'M_ACT' in df_csv.columns:
                        df_csv['CASE_NO'] = df_csv['M_ACT']
                    else:
                        df_csv['CASE_NO'] = ''
                if 'M_ACT' not in df_csv.columns:
                    if 'CASE_NO' in df_csv.columns:
                        df_csv['M_ACT'] = df_csv['CASE_NO']
                    else:
                        df_csv['M_ACT'] = ''
                df_csv['CASE_NO'] = df_csv['CASE_NO'].astype(str).replace('nan', '')
                df_csv['M_ACT'] = df_csv['M_ACT'].astype(str).replace('nan', '')
                self.all_csv_data.append(df_csv)
            except Exception as e:
                print(f"CSV 파일 {file} 처리 중 오류 발생: {str(e)}")
                continue
    
        if self.all_csv_data:
            self.csv_df = pd.concat(self.all_csv_data, ignore_index=True)
            numeric_columns = ['SHIP_NO', 'STOCK', 'PE_OFFSET', 'STG_CD', 'TO_STG_CD']
            for col in numeric_columns:
                if col in self.csv_df.columns:
                    self.csv_df[col] = pd.to_numeric(self.csv_df[col], errors='coerce')
            if 'CASE_NO' not in self.csv_df.columns:
                if 'M_ACT' in self.csv_df.columns:
                    self.csv_df['CASE_NO'] = self.csv_df['M_ACT']
                else:
                    self.csv_df['CASE_NO'] = ''
            if 'M_ACT' not in self.csv_df.columns:
                if 'CASE_NO' in self.csv_df.columns:
                    self.csv_df['M_ACT'] = self.csv_df['CASE_NO']
                else:
                    self.csv_df['M_ACT'] = ''
            self.csv_df['CASE_NO'] = self.csv_df['CASE_NO'].astype(str).replace('nan', '')
            self.csv_df['M_ACT'] = self.csv_df['M_ACT'].astype(str).replace('nan', '')
        else:
            self.csv_df = pd.DataFrame()
    
        # CSV 데이터를 주 데이터로 사용합니다.
        self.df = self.csv_df
        # RetrievalQA 체인 업데이트 (벡터 스토어는 기존 것을 사용)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 20}),
            chain_type_kwargs={"prompt": self.prompt}
        )
    
    def determine_data_source(self, query):
        # txt_data 기능 제거: 항상 CSV를 사용합니다.
        return "CSV"
    
    def query_csv_data(self, query):
        if self.csv_df.empty:
            return "CSV 데이터가 로드되지 않았습니다."
        if self.spacy_parser is None:
            self.spacy_parser = SpacyQueryParser()
        params = self.spacy_parser.parse_query(query)
        print(f"추출된 쿼리 파라미터: {params}")
        filtered_df, error = self.spacy_parser.apply_filters(self.csv_df, params)
        if error:
            return error
        return self.spacy_parser.format_results(filtered_df, params)
    
    def extract_equipment_code_from_query(self, query):
        """쿼리에서 설비 코드를 추출합니다."""
        # 설비 코드 패턴 (예: EQ001, M001 등)
        eq_pattern = r'(EQ\d{3}|M\d{3}|[A-Z]{1,2}\d{2,3})'
        eq_matches = re.findall(eq_pattern, query)
        
        # 직접적인 설비 언급 (예: "1번 설비", "2호기" 등)
        direct_pattern = r'(\d+)[번호]?\s*(설비|호기)'
        direct_matches = re.findall(direct_pattern, query)
        
        if eq_matches:
            return eq_matches[0]
        elif direct_matches:
            num, _ = direct_matches[0]
            return f"EQ{num.zfill(3)}"  # 예: 1 -> EQ001
        
        return None  
    
    def filter_by_range(self, start_date, end_date, equipment_code=None):
        """날짜 범위와 설비 코드로 데이터를 필터링합니다."""
        if self.csv_df.empty:
            return "데이터가 로드되지 않았습니다."
        
        filtered_df = self.csv_df.copy()
        
        # 날짜 필드가 있는지 확인
        if 'STOCK_DATE' not in filtered_df.columns and 'datetime' not in filtered_df.columns:
            return "날짜 정보가 없어 범위 필터링을 할 수 없습니다."
        
        # 날짜 필터링
        date_col = 'STOCK_DATE' if 'STOCK_DATE' in filtered_df.columns else 'datetime'
        if isinstance(filtered_df[date_col].iloc[0], str):
            filtered_df = filtered_df[(filtered_df[date_col] >= start_date) & (filtered_df[date_col] <= end_date)]
        else:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            filtered_df = filtered_df[(filtered_df[date_col] >= start) & (filtered_df[date_col] <= end)]
        
        # 설비 코드 필터링 (있는 경우)
        if equipment_code and 'CASE_NO' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['CASE_NO'].str.contains(equipment_code, na=False)]
        
        if filtered_df.empty:
            return f"{start_date}부터 {end_date}까지 데이터가 없습니다."
        
        # 결과 포맷팅
        total_items = len(filtered_df)
        total_stock = filtered_df['STOCK'].sum() if 'STOCK' in filtered_df.columns else 0
        
        result = f"기간: {start_date} ~ {end_date}\n"
        result += f"총 항목 수: {total_items}개\n"
        result += f"총 스톡 수: {total_stock}개\n"
        
        if equipment_code:
            result += f"설비 코드: {equipment_code}\n"
        
        return result   
    
    def generate_graph_data(self, start_date, end_date, equipment_code=None):
        """날짜 범위와 설비 코드로 그래프 데이터를 생성합니다."""
        if self.csv_df.empty:
            return None
        
        filtered_df = self.csv_df.copy()
        
        # 날짜 필드가 있는지 확인
        if 'STOCK_DATE' not in filtered_df.columns and 'datetime' not in filtered_df.columns:
            return None
        
        # 날짜 필터링
        date_col = 'STOCK_DATE' if 'STOCK_DATE' in filtered_df.columns else 'datetime'
        if isinstance(filtered_df[date_col].iloc[0], str):
            filtered_df = filtered_df[(filtered_df[date_col] >= start_date) & (filtered_df[date_col] <= end_date)]
        else:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            filtered_df = filtered_df[(filtered_df[date_col] >= start) & (filtered_df[date_col] <= end)]
        
        # 설비 코드 필터링 (있는 경우)
        if equipment_code and 'CASE_NO' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['CASE_NO'].str.contains(equipment_code, na=False)]
        
        if filtered_df.empty:
            return None
        
        # 날짜별 집계
        if 'datetime' in filtered_df.columns:
            grouped = filtered_df.groupby(pd.Grouper(key='datetime', freq='D'))['STOCK'].sum().reset_index()
            dates = [d.strftime('%Y-%m-%d') for d in grouped['datetime']]
            rates = grouped['STOCK'].tolist()
        else:
            grouped = filtered_df.groupby(date_col)['STOCK'].sum().reset_index()
            dates = grouped[date_col].tolist()
            rates = grouped['STOCK'].tolist()
        
        title = f"{start_date}부터 {end_date}까지 스톡 데이터"
        if equipment_code:
            title += f" (설비: {equipment_code})"
      
        return {"title": title, "dates": dates, "rates": rates}
    
        print("=== 로드된 MFG_IND 값 샘플 ===")
        if not self.csv_df.empty and 'MFG_IND' in self.csv_df.columns:
            print(self.csv_df['MFG_IND'].value_counts().head(10))
        else:
            print("MFG_IND 열이 존재하지 않거나 데이터가 비어 있습니다.") 
            
    def extract_threshold_from_query(self, query):
        """쿼리에서 임계값과 비교 조건을 추출합니다."""
        # 임계값 패턴 (예: "80% 이상", "50% 미만" 등)
        threshold_pattern = r'(\d+(\.\d+)?)%?\s*(이상|이하|초과|미만|같은|동일한)'
        threshold_matches = re.findall(threshold_pattern, query)
        
        if threshold_matches:
            value_str, _, condition = threshold_matches[0]
            value = float(value_str)
            
            # 조건 매핑
            condition_map = {
                "이상": ">=",
                "이하": "<=",
                "초과": ">",
                "미만": "<",
                "같은": "==",
                "동일한": "=="
            }
            
            return value, condition_map.get(condition, ">=")
        
        return None, None    
    
    
    def filter_by_threshold(self, threshold, condition, equipment_code=None):
        """임계값과 조건으로 데이터를 필터링합니다."""
        if self.csv_df.empty:
            return "데이터가 로드되지 않았습니다."
        
        filtered_df = self.csv_df.copy()
        
        # 기준 필드 (가동률, 스톡 등)
        metric_field = 'STOCK'  # 기본값으로 STOCK 사용
        
        if equipment_code and 'CASE_NO' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['CASE_NO'].str.contains(equipment_code, na=False)]
        
        if filtered_df.empty:
            return f"조건에 맞는 데이터가 없습니다."
        
        # 조건별 필터링
        if condition == ">=":
            filtered_df = filtered_df[filtered_df[metric_field] >= threshold]
        elif condition == "<=":
            filtered_df = filtered_df[filtered_df[metric_field] <= threshold]
        elif condition == ">":
            filtered_df = filtered_df[filtered_df[metric_field] > threshold]
        elif condition == "<":
            filtered_df = filtered_df[filtered_df[metric_field] < threshold]
        elif condition == "==":
            filtered_df = filtered_df[filtered_df[metric_field] == threshold]
        
        if filtered_df.empty:
            return f"임계값 {threshold} {condition} 조건에 맞는 데이터가 없습니다."
        
        # 결과 포맷팅
        total_items = len(filtered_df)
        
        result = f"임계값: {threshold} {condition}\n"
        result += f"조건을 만족하는 항목 수: {total_items}개\n"
        
        if equipment_code:
            result += f"설비 코드: {equipment_code}\n"
        
        return result    
    
    def generate_graph_data_threshold(self, threshold, condition, equipment_code=None):
        """임계값과 조건으로 그래프 데이터를 생성합니다."""
        if self.csv_df.empty:
            return None
        
        filtered_df = self.csv_df.copy()
        
        # 기준 필드 (가동률, 스톡 등)
        metric_field = 'STOCK'  # 기본값으로 STOCK 사용
        
        if equipment_code and 'CASE_NO' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['CASE_NO'].str.contains(equipment_code, na=False)]
        
        if filtered_df.empty:
            return None
        
        # 조건별 필터링
        if condition == ">=":
            filtered_df = filtered_df[filtered_df[metric_field] >= threshold]
        elif condition == "<=":
            filtered_df = filtered_df[filtered_df[metric_field] <= threshold]
        elif condition == ">":
            filtered_df = filtered_df[filtered_df[metric_field] > threshold]
        elif condition == "<":
            filtered_df = filtered_df[filtered_df[metric_field] < threshold]
        elif condition == "==":
            filtered_df = filtered_df[filtered_df[metric_field] == threshold]
        
        if filtered_df.empty:
            return None
        
        # 날짜별 집계 (날짜 필드가 있는 경우)
        if 'STOCK_DATE' in filtered_df.columns or 'datetime' in filtered_df.columns:
            date_col = 'STOCK_DATE' if 'STOCK_DATE' in filtered_df.columns else 'datetime'
            
            if 'datetime' in filtered_df.columns:
                grouped = filtered_df.groupby(pd.Grouper(key='datetime', freq='D'))[metric_field].sum().reset_index()
                dates = [d.strftime('%Y-%m-%d') for d in grouped['datetime']]
                rates = grouped[metric_field].tolist()
            else:
                grouped = filtered_df.groupby(date_col)[metric_field].sum().reset_index()
                dates = grouped[date_col].tolist()
                rates = grouped[metric_field].tolist()
        else:
            # 날짜 필드가 없는 경우 다른 유의미한 필드로 집계
            if 'SHIP_NO' in filtered_df.columns:
                grouped = filtered_df.groupby('SHIP_NO')[metric_field].sum().reset_index()
                dates = grouped['SHIP_NO'].astype(str).tolist()
                rates = grouped[metric_field].tolist()
            else:
                # 적절한 그룹화 필드가 없으면 그래프 데이터 생성 불가
                return None
        
        condition_symbol = {
            ">=": "≥", "<=": "≤", ">": ">", "<": "<", "==": "="
        }.get(condition, condition)
        
        title = f"임계값 {threshold} {condition_symbol} 조건 데이터"
        if equipment_code:
            title += f" (설비: {equipment_code})"
        
        return {"title": title, "dates": dates, "rates": rates}
    
    def direct_data_lookup(self, query):
        """직접적인 데이터 조회를 처리합니다."""
        # 특정 날짜 조회 패턴 (예: "2023년 5월 15일 데이터", "2023-05-15 정보")
        date_patterns = [
            r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일',  # 2023년 5월 15일
            r'(\d{4})-(\d{1,2})-(\d{1,2})'            # 2023-05-15
        ]
        
        for pattern in date_patterns:
            date_match = re.search(pattern, query)
            if date_match:
                if len(date_match.groups()) == 3:
                    year, month, day = date_match.groups()
                    formatted_date = f"{year}-{int(month):02d}-{int(day):02d}"
                    
                    if self.csv_df.empty:
                        return "데이터가 로드되지 않았습니다."
                    
                    # 날짜 열이 있는지 확인
                    if 'STOCK_DATE' not in self.csv_df.columns and 'datetime' not in self.csv_df.columns:
                        return "날짜 정보가 없어 날짜별 조회를 할 수 없습니다."
                    
                    # 날짜로 필터링
                    date_col = 'STOCK_DATE' if 'STOCK_DATE' in self.csv_df.columns else 'datetime'
                    filtered_df = self.csv_df[self.csv_df[date_col] == formatted_date]
                    
                    if filtered_df.empty:
                        return f"{formatted_date} 날짜의 데이터가 없습니다."
                    
                    # 결과 포맷팅
                    total_items = len(filtered_df)
                    total_stock = filtered_df['STOCK'].sum() if 'STOCK' in filtered_df.columns else 0
                    
                    result = f"날짜: {formatted_date}\n"
                    result += f"총 항목 수: {total_items}개\n"
                    result += f"총 스톡 수: {total_stock}개\n"
                    
                    # 호선별 집계 (있다면)
                    if 'SHIP_NO' in filtered_df.columns:
                        ship_counts = filtered_df.groupby('SHIP_NO')['STOCK'].sum().reset_index()
                        result += "\n호선별 스톡 수:\n"
                        for _, row in ship_counts.iterrows():
                            result += f"- 호선 {row['SHIP_NO']}: {row['STOCK']}개\n"
                    
                    return result
        
        # 직접적인 정보 조회 패턴이 없으면 None 반환
        return None    
    
    def answer_query(self, query, temperature=0.1, max_tokens=1200):
        """
        질의에 응답하며 '결과값:'과 '분석:'을 출력하고, 그래프 데이터가 필요한 경우 반환
        
        인자:
            query (str): 사용자 질의
            temperature (float): LLM 온도 매개변수
            max_tokens (int): LLM 응답 최대 토큰 수
        반환:
            tuple: (응답 문자열, 그래프 데이터 또는 None)
        """
        logger.debug(f"질의: {query}, 최대 토큰: {max_tokens}")
        
        # 쿼리 파서 초기화 (필요한 경우)
        if self.spacy_parser is None:
            self.spacy_parser = SpacyQueryParser()
        
        # 파라미터 추출
        params = self.spacy_parser.parse_query(query)
        logger.debug(f"추출된 파라미터: {params}")
        
        # CSV 데이터 처리
        if self.csv_df.empty:
            return "CSV 데이터가 로드되지 않았습니다.", None
        
        # SpacyQueryParser의 apply_filters 메서드를 사용하여 필터링
        filtered_df, error = self.spacy_parser.apply_filters(self.csv_df, params)
        
        if error:
            return f"오류: {error}", None
        
        if filtered_df.empty:
            return "조건에 맞는 데이터가 없습니다.", None
        
        # 총 스톡 수 계산 관련 질문인지 확인
        if "총" in query and "몇" in query and ("개" in query or "스톡" in query):
            total_stock = filtered_df['STOCK'].sum()
            result = f"조건에 맞는 총 스톡 수: {total_stock}개 (총 {len(filtered_df)}개 항목)"
            
            # 그래프 데이터 생성
            graph_data = self.generate_csv_graph_data(filtered_df, query)
            
            analysis_prompt = f"""
            다음 스톡 데이터를 간단히 분석해주세요:
            {result}
            
            조건: {', '.join([f'{k}: {v}' for k, v in params.items()])}
            
            분석:
            - 간결하고 핵심적인 분석만 제공하세요.
            """
            try:
                full_analysis = self.llm.invoke(analysis_prompt, max_tokens=max_tokens, temperature=temperature)
                return f"결과값:\n{result}\n\n분석:\n{full_analysis}", graph_data
            except Exception as e:
                return f"결과값:\n{result}\n\n분석:\n분석 생성 중 오류: {str(e)}", graph_data
        
        # 일반 CSV 데이터 쿼리 - format_results로 결과 포맷팅
        result = self.spacy_parser.format_results(filtered_df, params)
        
        # 그래프 데이터 생성
        graph_data = self.generate_csv_graph_data(filtered_df, query)
        
        analysis_prompt = f"""
        다음 스톡 데이터를 간단히 분석해주세요:
        {result}
        
        분석:
        - 간결하고 핵심적인 분석만 제공하세요.
        - 가장 중요한 패턴이나 특이점을 강조하세요.
        """
        try:
            full_analysis = self.llm.invoke(analysis_prompt, max_tokens=max_tokens, temperature=temperature)
            return f"결과값:\n{result}\n\n분석:\n{full_analysis}", graph_data
        except Exception as e:
            return f"결과값:\n{result}\n\n분석:\n분석 생성 중 오류: {str(e)}", graph_data
        
        # 아래 코드는 범위 질의나 임계값 질의 처리를 위한 코드지만,
        # SpacyQueryParser가 이미 처리했으므로 이 부분은 실행되지 않습니다.
        # CSV 데이터만 처리하도록 수정했기 때문에, 이 아래 코드는 불필요합니다.
        
        # 범위 질의 처리
        lower, upper = self.extract_range_from_query(query)
        if lower is not None and upper is not None:
            eq_code = self.extract_equipment_code_from_query(query)
            result = self.filter_by_range(lower, upper, eq_code)
            
            # "데이터가 없습니다" 문구 확인
            if "없습니다" in result:
                return "조건에 맞는 데이터가 없습니다.", None
            
            graph_data = self.generate_graph_data(lower, upper, eq_code)
            
            analysis_prompt = f"""
            다음 설비 가동률 데이터를 간단히 분석해주세요:
            {result}
            
            분석:
            - 간결하고 핵심적인 분석만 제공하세요.
            """
            try:
                full_analysis = self.llm.invoke(analysis_prompt, max_tokens=max_tokens, temperature=temperature)
                return f"결과값:\n{result}\n\n분석:\n{full_analysis}", graph_data
            except Exception as e:
                return f"결과값:\n{result}\n\n분석:\n분석 생성 중 오류: {str(e)}", None
        
        # 임계값 질의 처리
        threshold, condition = self.extract_threshold_from_query(query)
        if threshold is not None:
            eq_code = self.extract_equipment_code_from_query(query)
            result = self.filter_by_threshold(threshold, condition, eq_code)
            
            # "데이터가 없습니다" 문구 확인
            if "없습니다" in result:
                return "조건에 맞는 데이터가 없습니다.", None
            
            graph_data = self.generate_graph_data_threshold(threshold, condition, eq_code)
            
            analysis_prompt = f"""
            다음 설비 가동률 데이터를 간단히 분석해주세요:
            {result}
            
            분석:
            - 간결하고 핵심적인 분석만 제공하세요.
            """
            try:
                full_analysis = self.llm.invoke(analysis_prompt, max_tokens=max_tokens, temperature=temperature)
                return f"결과값:\n{result}\n\n분석:\n{full_analysis}", graph_data
            except Exception as e:
                return f"결과값:\n{result}\n\n분석:\n분석 생성 중 오류: {str(e)}", None
        
        # 직접 날짜 조회
        direct_answer = self.direct_data_lookup(query)
        if direct_answer:
            # "데이터가 없습니다" 문구 확인
            if "없습니다" in direct_answer:
                return "조건에 맞는 데이터가 없습니다.", None
            
            analysis_prompt = f"""
            다음 설비 가동률 데이터를 간단히 분석해주세요:
            {direct_answer}
            
            분석:
            - 간결하고 핵심적인 분석만 제공하세요.
            """
            try:
                full_analysis = self.llm.invoke(analysis_prompt, max_tokens=max_tokens, temperature=temperature)
                return f"결과값:\n{direct_answer}\n\n분석:\n{full_analysis}", None
            except Exception as e:
                return f"결과값:\n{direct_answer}\n\n분석:\n분석 생성 중 오류: {str(e)}", None
        
        # 일반 RAG 질의 처리 - 여기까지 오면 SpacyQueryParser가 처리하지 못한 것이므로 RAG로 시도
        try:
            answer = self.qa_chain.run(query)
            
            # 데이터 없음 확인
            if "데이터가 없습니다" in answer or "찾을 수 없습니다" in answer:
                return "조건에 맞는 데이터가 없습니다.", None
            
            if "결과값:" not in answer:
                parts = answer.split("\n\n", 1)
                if len(parts) > 1:
                    result, analysis = parts
                    answer = f"결과값:\n{result}\n\n분석:\n{analysis}"
                else:
                    analysis_prompt = f"""
                    다음 데이터를 간단히 분석해주세요:
                    {answer}
                    
                    분석:
                    - 간결하고 핵심적인 분석만 제공하세요.
                    """
                    full_analysis = self.llm.invoke(analysis_prompt, max_tokens=max_tokens, temperature=temperature)
                    answer = f"결과값:\n{answer}\n\n분석:\n{full_analysis}"
            return answer, None
        except Exception as e:
            return f"오류가 발생했습니다: {str(e)}", None
    def generate_csv_graph_data(self, filtered_df, query):
        """
        Generate graph data from filtered CSV dataframe based on the query.
        
        Args:
            filtered_df (pandas.DataFrame): Filtered dataframe with relevant data
            query (str): User query string that might contain hints about visualization needs
                
        Returns:
            dict or None: Dictionary with graph data or None if no graph is needed
        """
        if filtered_df.empty:
            return None
        
        # 그래프 관련 명시적 요청 키워드 확인
        graph_keywords = ["그래프", "차트", "시각화", "보여줘", "그려줘", "추이", "변화", "트렌드", 
                          "분포", "비교", "분석해서"]
        explicit_request = any(keyword in query for keyword in graph_keywords)
        
        # 명시적 요청이 없으면 그래프 생성하지 않음
        if not explicit_request:
            return None
        
        # 그래프 데이터 기본 구조
        graph_data = {
            "title": "데이터 분석 결과",
            "dates": [],
            "rates": []
        }
        
        # 시계열 데이터 있는지 확인
        has_time_data = 'STOCK_DATE' in filtered_df.columns or 'datetime' in filtered_df.columns
        
        # 추이/트렌드 분석 요청인 경우 시계열 그래프
        if ("추이" in query or "트렌드" in query or "변화" in query) and has_time_data:
            date_col = 'STOCK_DATE' if 'STOCK_DATE' in filtered_df.columns else 'datetime'
            
            if 'datetime' in filtered_df.columns:
                grouped = filtered_df.groupby(pd.Grouper(key='datetime', freq='D'))['STOCK'].sum().reset_index()
                graph_data["dates"] = [d.strftime('%Y-%m-%d') for d in grouped['datetime']]
                graph_data["rates"] = grouped['STOCK'].tolist()
            else:
                grouped = filtered_df.groupby(date_col)['STOCK'].sum().reset_index()
                graph_data["dates"] = grouped[date_col].tolist()
                graph_data["rates"] = grouped['STOCK'].tolist()
            
            graph_data["title"] = "날짜별 스톡 추이"
            return graph_data
        
        # 카테고리별 분포 요청인 경우
        category_map = {
            "사업장별": "MFG_IND", 
            "호선별": "SHIP_NO",
            "블록별": "BLOCK",
            "스테이지별": "STG_CD"
        }
        
        for category_name, col_name in category_map.items():
            if category_name in query and col_name in filtered_df.columns:
                # 카테고리가 너무 많으면 상위 10개만
                if filtered_df[col_name].nunique() > 10:
                    top_values = filtered_df.groupby(col_name)['STOCK'].sum().nlargest(10).index.tolist()
                    group_df = filtered_df[filtered_df[col_name].isin(top_values)]
                    grouped = group_df.groupby(col_name)['STOCK'].sum().reset_index()
                else:
                    grouped = filtered_df.groupby(col_name)['STOCK'].sum().reset_index()
                
                # 값 기준 내림차순 정렬
                grouped = grouped.sort_values('STOCK', ascending=False)
                
                graph_data["dates"] = grouped[col_name].astype(str).tolist()
                graph_data["rates"] = grouped['STOCK'].tolist()
                graph_data["title"] = f"{category_name} 스톡 분포"
                return graph_data
        
        # 일반적인 명시적 그래프 요청인 경우, 적합한 시각화 선택
        if has_time_data:
            # 날짜 데이터 있으면 시계열 그래프
            date_col = 'STOCK_DATE' if 'STOCK_DATE' in filtered_df.columns else 'datetime'
            
            if 'datetime' in filtered_df.columns:
                grouped = filtered_df.groupby(pd.Grouper(key='datetime', freq='D'))['STOCK'].sum().reset_index()
                graph_data["dates"] = [d.strftime('%Y-%m-%d') for d in grouped['datetime']]
                graph_data["rates"] = grouped['STOCK'].tolist()
            else:
                grouped = filtered_df.groupby(date_col)['STOCK'].sum().reset_index()
                graph_data["dates"] = grouped[date_col].tolist()
                graph_data["rates"] = grouped['STOCK'].tolist()
            
            graph_data["title"] = "날짜별 스톡 데이터"
            return graph_data
        elif 'SHIP_NO' in filtered_df.columns:
            # 호선별 분포 그래프
            grouped = filtered_df.groupby('SHIP_NO')['STOCK'].sum().reset_index()
            graph_data["dates"] = grouped['SHIP_NO'].astype(str).tolist()
            graph_data["rates"] = grouped['STOCK'].tolist()
            graph_data["title"] = "호선별 스톡 분포"
            return graph_data
        
        # 적절한 그래프를 생성할 수 없는 경우
        return None

# =============================================================================
# --- GUI Chat System (RAG 통합) ---
# =============================================================================
THEME = {
    'primary': '#2E86C1',
    'secondary': '#AED6F1',
    'error': '#E74C3C',
    'background': '#F4F6F7',
    'surface': '#FFFFFF',
    'text': '#2C3E50'
}

def highlight_data(text):
    pattern = r"(\d{1,3}\.\d{2}%)"
    return re.sub(pattern, r'<span style="background-color:yellow; font-weight:bold;">\1</span>', text)

class User:
    def __init__(self, username: str, password: str, department: str, role: str = "user"):
        self.username = username
        self.password = password
        self.department = department
        self.role = role
        self.login_time = None

class UserManager:
    def __init__(self):
        self._initialize_default_users()
        self.current_user = None
        self.login_attempts = {}

    def _initialize_default_users(self):
        self.users = {
            "디지털혁신부": User("디지털혁신부", "1234", "디지털혁신부", "admin"),
            "사업기획부": User("사업기획부", "1234", "사업기획부", "user"),
        }

    def login(self, username: str, password: str) -> bool:
        if username not in self.login_attempts:
            self.login_attempts[username] = {"count": 0, "lockout_until": None}
        attempts = self.login_attempts[username]
        if attempts["lockout_until"] and time.time() < attempts["lockout_until"]:
            return False
        if username in self.users and self.users[username].password == password:
            self.current_user = self.users[username]
            self.current_user.login_time = time.time()
            attempts["count"] = 0
            return True
        attempts["count"] += 1
        if attempts["count"] >= 3:
            attempts["lockout_until"] = time.time() + 300
        return False

class ModernLoginWindow(ctk.CTkToplevel):
    def __init__(self, user_manager, on_login_success):
        super().__init__()
        self.user_manager = user_manager
        self.on_login_success = on_login_success
        self.setup_window()
        self.create_widgets()
        self.bind_events()

    def setup_window(self):
        self.title("MIFO LangChain RAG DEMO")
        self.geometry("400x550")
        self.configure(fg_color=THEME['background'])
        self.withdraw()
        self.update_idletasks()
        x = (self.winfo_screenwidth() - 400) // 2
        y = (self.winfo_screenheight() - 550) // 2
        self.geometry(f"+{x}+{y}")
        self.deiconify()
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        self.logo_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.logo_frame.pack(pady=(40, 20))
        ctk.CTkLabel(
            self.logo_frame,
            text="MIFO LangChain RAG DEMO",
            font=("Helvetica", 26, "bold"),
            text_color=THEME['primary']
        ).pack()

        self.content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.content_frame.pack(pady=10, padx=40, fill="x")

        self.department_var = ctk.StringVar(value="디지털혁신부")
        ctk.CTkLabel(
            self.content_frame,
            text="Department",
            font=("Helvetica", 12, "bold"),
            text_color=THEME['text']
        ).pack(pady=(0, 5), anchor="w")
        self.department_menu = ctk.CTkOptionMenu(
            self.content_frame,
            values=["디지털혁신부", "사업기획부"],
            variable=self.department_var,
            width=320,
            font=("Helvetica", 12),
            fg_color=THEME['surface'],
            button_color=THEME['primary'],
            dropdown_fg_color=THEME['surface'],
            dropdown_font=("Helvetica", 12),
            corner_radius=12
        )
        self.department_menu.pack(pady=(0, 15))

        self.model_var = ctk.StringVar(value="oh-dcft-v3.1-claude-3-5-sonnet-20241022.Q5_K_M.gguf")
        ctk.CTkLabel(
            self.content_frame,
            text="Model",
            font=("Helvetica", 12, "bold"),
            text_color=THEME['text']
        ).pack(pady=(0, 5), anchor="w")
        self.model_menu = ctk.CTkOptionMenu(
            self.content_frame,
            values=[
                "oh-dcft-v3.1-claude-3-5-sonnet-20241022.Q5_K_M.gguf",
                "MIFO_20250115.gguf",
                "deepseek-llm-7b-chat.Q4_K_M_MIFO.gguf",
                "Meta-Llama-3-8B-Instruct.Q4_0.gguf",
            ],
            variable=self.model_var,
            width=320,
            font=("Helvetica", 12),
            fg_color=THEME['surface'],
            button_color=THEME['primary'],
            dropdown_fg_color=THEME['surface'],
            dropdown_font=("Helvetica", 12),
            corner_radius=12
        )
        self.model_menu.pack(pady=(0, 15))

        self.browse_button = ctk.CTkButton(
            self.content_frame,
            text="Browse Model...",
            command=self.browse_model_file,
            width=320,
            font=("Helvetica", 12),
            fg_color=THEME['primary'],
            hover_color=THEME['secondary'],
            corner_radius=12
        )
        self.browse_button.pack(pady=(0, 15))

        self.use_vector_store_var = ctk.BooleanVar(value=True)
        self.vector_store_checkbox = ctk.CTkCheckBox(
            self.content_frame,
            text="기존 벡터 스토어 사용 (있을 경우)",
            variable=self.use_vector_store_var,
            font=("Helvetica", 12),
            fg_color=THEME['primary'],
            hover_color=THEME['secondary'],
            corner_radius=8
        )
        self.vector_store_checkbox.pack(pady=(0, 15), anchor="w")

        ctk.CTkLabel(
            self.content_frame,
            text="Password",
            font=("Helvetica", 12, "bold"),
            text_color=THEME['text']
        ).pack(pady=(0, 5), anchor="w")
        self.password_entry = ctk.CTkEntry(
            self.content_frame,
            show="●",
            width=320,
            height=40,
            font=("Helvetica", 12),
            placeholder_text="Enter password",
            border_width=0,
            fg_color=THEME['surface'],
            text_color=THEME['text'],
            placeholder_text_color='#A0AEC0',
            corner_radius=12
        )
        self.password_entry.pack(pady=(0, 20))

        self.login_button = ctk.CTkButton(
            self.content_frame,
            text="Login",
            command=self.login,
            width=320,
            height=45,
            font=("Helvetica", 14, "bold"),
            fg_color=THEME['primary'],
            hover_color=THEME['secondary'],
            corner_radius=12
        )
        self.login_button.pack(pady=10)

        self.error_label = ctk.CTkLabel(
            self.content_frame,
            text="",
            text_color=THEME['error'],
            font=("Helvetica", 12)
        )
        self.error_label.pack(pady=5)

    def bind_events(self):
        self.bind('<Return>', lambda e: self.login())
        self.password_entry.bind('<Return>', lambda e: self.login())

    def browse_model_file(self):
        file_path = askopenfilename(
            title="Select Model File",
            filetypes=[("GGUF files", "*.gguf"), ("All files", "*.*")],
            initialdir="./models"
        )
        if file_path and Path(file_path).exists():
            self.model_var.set(file_path)
        elif file_path:
            messagebox.showerror("Error", "Selected model file does not exist!")

    def login(self):
        username = self.department_var.get()
        password = self.password_entry.get()
        model_path = self.model_var.get()
        use_vector_store = self.use_vector_store_var.get()
        if self.user_manager.login(username, password):
            self.on_login_success(model_path, use_vector_store)
            self.destroy()
        else:
            self.error_label.configure(text="Invalid credentials")
            self.password_entry.delete(0, 'end')

    def on_closing(self):
        self.quit()
        self.destroy()
        sys.exit()

class ChatMessage(ctk.CTkFrame):
    def __init__(self, master, message: str, is_user: bool = True, graph_data=None):
        bg_color = '#D6EAF8' if is_user else '#FDFEFE'
        super().__init__(master, fg_color=bg_color, border_width=0, corner_radius=16)
        self.message = self.clean_message(message)
        self.is_user = is_user
        self.graph_data = graph_data
        self.canvas = None
        self.active_notification = None
        self.setup_message()

    def setup_message(self):
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=15, pady=(10, 0))
        sender_color = THEME['primary'] if self.is_user else THEME['secondary']
        sender_text = "미포사용자님" if self.is_user else "AI에이전트"
        ctk.CTkLabel(
            header,
            text=sender_text,
            font=("Helvetica", 11, "bold"),
            text_color=sender_color
        ).pack(side="left")
        ctk.CTkButton(
            header,
            text="Copy All",
            width=60,
            height=22,
            font=("Helvetica", 10),
            command=self.copy_to_clipboard,
            fg_color=sender_color,
            hover_color=sender_color,
            corner_radius=8
        ).pack(side="right", padx=8, pady=3)
        self.create_text_block(self.message)

    def create_text_block(self, text: str):
        if not text.strip() and not self.graph_data:
            return
        content_frame = ctk.CTkFrame(self, fg_color="transparent")
        content_frame.pack(fill="x", padx=15, pady=5)
        if text.strip() and not self.is_user and MARKDOWN_AVAILABLE and TKINTERWEB_AVAILABLE:
            processed_text = highlight_data(text)
            html_content = markdown.markdown(
                processed_text,
                extensions=['nl2br', 'fenced_code', 'tables']
            )
            custom_style = """
            <style>
              body { font-family: Helvetica, sans-serif; line-height: 1.5; padding: 10px; background-color: #F4F6F7; color: #2C3E50; }
              code { background-color: #f4f4f4; padding: 2px 4px; border-radius: 4px; }
              pre { background-color: #fafafa; color: #2C3E50; padding: 10px; border-radius: 8px; overflow-x: auto; }
              table { border-collapse: collapse; width: 100%; margin-top: 10px; }
              th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
              blockquote { border-left: 4px solid #ccc; padding-left: 10px; color: #555; margin: 10px 0; }
            </style>
            """
            html_content = custom_style + html_content
            line_count = text.count('\n') + 1
            calculated_height = min(max(400, line_count * 20), 800)
            frame_for_html = ctk.CTkFrame(content_frame, fg_color="transparent", width=1500, height=calculated_height)
            frame_for_html.pack_propagate(False)
            frame_for_html.pack(fill="both", expand=True)
            html_frame = HtmlFrame(frame_for_html, messages_enabled=False, horizontal_scrollbar="auto", vertical_scrollbar="auto")
            html_frame.load_html(html_content)
            html_frame.pack(fill="both", expand=True)
        elif text.strip():
            label = ctk.CTkLabel(
                content_frame,
                text=text,
                font=("Helvetica", 13),
                wraplength=1500,
                justify="left",
                text_color=THEME['text']
            )
            label.pack(fill="x")
        if not self.is_user and self.graph_data:
            plt.rc('font', family='Malgun Gothic')
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(self.graph_data["dates"], self.graph_data["rates"], marker='o', linestyle='-')
            ax.set_title(self.graph_data["title"])
            ax.set_xlabel("날짜")
            ax.set_ylabel("가동률 (%)")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            self.canvas = FigureCanvasTkAgg(fig, master=content_frame)
            self.canvas.draw()
            canvas_widget = self.canvas.get_tk_widget()
            canvas_widget.pack(fill="x", pady=10)
            canvas_widget.bind("<MouseWheel>", lambda event: "break")

    def copy_to_clipboard(self):
        self.clipboard_clear()
        self.clipboard_append(self.message)
        self.show_copy_notification("Full message copied to clipboard!")

    def show_copy_notification(self, message):
        if self.active_notification and self.active_notification.winfo_exists():
            self.active_notification.destroy()
        notification = ctk.CTkToplevel(self)
        self.active_notification = notification
        notification.attributes('-topmost', True)
        notification.overrideredirect(True)
        width, height = 200, 40
        x = self.winfo_rootx() + (self.winfo_width() - width) // 2
        y = self.winfo_rooty() - height - 10
        notification.geometry(f"{width}x{height}+{x}+{y}")
        notify_frame = ctk.CTkFrame(notification, fg_color="#333333", corner_radius=6)
        notify_frame.pack(fill="both", expand=True)
        ctk.CTkLabel(notify_frame, text=f"✓ {message}", font=("Helvetica", 11), text_color="#FFFFFF").pack(expand=True)
        self.after(1500, notification.destroy)

    @staticmethod
    def clean_message(message: str) -> str:
        message = re.sub(r'!\[.*?\]\(.*?\)', '', message)
        patterns = [r"</?assistant>", r"</?system>", r"</?user>"]
        for pattern in patterns:
            message = re.sub(pattern, "", message)
        return message.strip()

class ModernChatUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.withdraw()
        self.user_manager = UserManager()
        self.setup_window()
        self.rag_system = None
        self.response_queue = queue.Queue()
        self.is_generating = False
        self.conversation_history = []
        self.speech_enabled = False
        self.send_count = 0
        self.license_valid = False
        self.show_login()

    def setup_window(self):
        self.title("MIFO LangChain RAG DEMO")
        self.geometry("1000x800")
        self.configure(fg_color=THEME['background'])
        
        self.header = ctk.CTkFrame(self, fg_color=THEME['surface'], height=80)
        self.header.pack(fill="x", side="top")
        self.header_left_frame = ctk.CTkFrame(self.header, fg_color=THEME['surface'])
        self.header_left_frame.pack(side="left", fill="both", expand=True, padx=20, pady=10)
        self.title_logout_frame = ctk.CTkFrame(self.header_left_frame, fg_color=THEME['surface'])
        self.title_logout_frame.pack(anchor="w", fill="x")
        
        self.title_label = ctk.CTkLabel(
            self.title_logout_frame,
            text="MIFO LangChain RAG DEMO",
            font=("Helvetica", 20, "bold"),
            text_color=THEME['primary']
        )
        self.title_label.pack(side="left", anchor="w")
        
        self.logout_button = ctk.CTkButton(
            self.title_logout_frame,
            text="로그아웃",
            command=self.logout,
            width=80,
            font=("Helvetica", 12, "bold"),
            fg_color=THEME['error'],
            hover_color="#F08080",
            corner_radius=8
        )
        self.logout_button.pack(side="left", anchor="w", padx=10, pady=5)
        
        self.model_frame = ctk.CTkFrame(self.header_left_frame, fg_color=THEME['secondary'], corner_radius=10, height=30)
        self.model_frame.pack(anchor="w", pady=(4, 0), padx=(0, 10))
        self.model_label = ctk.CTkLabel(
            self.model_frame,
            text="Loaded Model: None",
            font=("Helvetica", 14, "bold"),
            text_color=THEME['primary'],
            fg_color=THEME['secondary']
        )
        self.model_label.pack(side="left", padx=(2, 4), pady=5)
        
        self.speech_switch = ctk.CTkSwitch(self.header, text="음성 ON", command=self.toggle_speech)
        self.speech_switch.pack(side="right", padx=20, pady=10)
        
        self.main_frame = ctk.CTkFrame(self, fg_color=THEME['background'])
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=10)
        self.main_frame.grid_columnconfigure(0, weight=0)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)
        
        self.sidebar = ctk.CTkFrame(self.main_frame, width=200, fg_color=THEME['surface'], corner_radius=12)
        self.sidebar.grid(row=0, column=0, sticky="nswe", padx=(0, 10))
        
        self.upload_pdf_button = ctk.CTkButton(
            self.sidebar,
            text="PDF 요약/분석",
            font=("Helvetica", 12, "bold"),
            fg_color=THEME['primary'],
            hover_color=THEME['secondary'],
            corner_radius=8,
            command=self.choose_pdf_mode
        )
        self.upload_pdf_button.pack(pady=(0, 10), padx=15, fill="x")
        
        self.user_info = ctk.CTkLabel(
            self.sidebar,
            text="Not logged in",
            font=("Helvetica", 16, "bold"),
            text_color=THEME['text']
        )
        self.user_info.pack(pady=20, padx=15, anchor="w")
        
        ctk.CTkLabel(
            self.sidebar,
            text="Max Tokens:",
            font=("Helvetica", 12, "bold"),
            text_color=THEME['text']
        ).pack(pady=(10, 0), padx=15, anchor="w")
        self.token_slider = ctk.CTkSlider(self.sidebar, from_=128, to=9080, number_of_steps=30, command=self.update_token_label)
        self.token_slider.set(500)
        self.token_slider.pack(pady=(0, 5), padx=15, fill="x")
        self.token_value_label = ctk.CTkLabel(self.sidebar, text="500", font=("Helvetica", 12), text_color=THEME['text'])
        self.token_value_label.pack(pady=(0, 5), padx=15, anchor="w")
        self.token_count_label = ctk.CTkLabel(self.sidebar, text="Estimated tokens: 0", font=("Helvetica", 12), text_color=THEME['text'])
        self.token_count_label.pack(pady=(0, 10), padx=15, anchor="w")
        
        self.clear_button = ctk.CTkButton(
            self.sidebar,
            text="대화내용 삭제",
            font=("Helvetica", 12, "bold"),
            fg_color=THEME['error'],
            hover_color="#F08080",
            corner_radius=8,
            command=self.clear_conversation
        )
        self.clear_button.pack(pady=(0, 10), padx=15, fill="x")
        
        self.prompt_button = ctk.CTkButton(
            self.sidebar,
            text="프롬프트 템플릿 수정",
            font=("Helvetica", 12, "bold"),
            fg_color=THEME['primary'],
            hover_color=THEME['secondary'],
            corner_radius=8,
            command=self.open_prompt_window
        )
        self.prompt_button.pack(pady=(0, 10), padx=15, fill="x")
        
        self.reload_button = ctk.CTkButton(
            self.sidebar,
            text="데이터 재로드",
            font=("Helvetica", 12, "bold"),
            fg_color=THEME['primary'],
            hover_color=THEME['secondary'],
            corner_radius=8,
            command=self.reload_data
        )
        self.reload_button.pack(pady=(0, 10), padx=15, fill="x")
    
        self.chat_frame = ctk.CTkFrame(self.main_frame, fg_color=THEME['background'])
        self.chat_frame.grid(row=0, column=1, sticky="nswe")
        self.chat_container = ctk.CTkScrollableFrame(self.chat_frame, fg_color=THEME['surface'], corner_radius=12)
        self.chat_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.mode_frame = ctk.CTkFrame(self.chat_frame, fg_color=THEME['background'])
        self.mode_frame.pack(fill="x", pady=(10, 0), padx=10)
        
        mode_label = ctk.CTkLabel(
            self.mode_frame,
            text="데이터 모드 선택:",
            font=("Helvetica", 12, "bold"),
            text_color=THEME['text']
        )
        mode_label.pack(side="left", padx=(0, 10))
        
        self.selected_mode = ctk.StringVar(value="AUTO")
        
        self.auto_mode_radio = ctk.CTkRadioButton(
            self.mode_frame,
            text="자동 감지",
            variable=self.selected_mode,
            value="AUTO",
            font=("Helvetica", 12),
            fg_color=THEME['primary'],
            hover_color=THEME['secondary']
        )
        self.auto_mode_radio.pack(side="left", padx=10)
        

        
        self.csv_mode_radio = ctk.CTkRadioButton(
            self.mode_frame,
            text="CSV 모드 (호선/스톡)",
            variable=self.selected_mode,
            value="CSV",
            font=("Helvetica", 12),
            fg_color=THEME['primary'],
            hover_color=THEME['secondary']
        )
        self.csv_mode_radio.pack(side="left", padx=10)
        
        self.mode_info_button = ctk.CTkButton(
            self.mode_frame,
            text="?",
            width=20,
            height=20,
            font=("Helvetica", 12, "bold"),
            command=self.show_mode_info,
            fg_color=THEME['secondary'],
            hover_color=THEME['primary'],
            corner_radius=10
        )
        self.mode_info_button.pack(side="left", padx=10)
        
        self.input_frame = ctk.CTkFrame(self.chat_frame, fg_color=THEME['background'])
        self.input_frame.pack(fill="x", pady=(10, 10), padx=10)
        
        self.input_field = ctk.CTkTextbox(
            self.input_frame,
            height=80,
            font=("Helvetica", 13),
            wrap="word",
            border_width=1,
            border_color=THEME['primary'],
            corner_radius=12,
            fg_color=THEME['surface']
        )
        self.input_field.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.input_field.bind("<<Modified>>", self.adjust_input_height)
        
        self.stop_button = ctk.CTkButton(
            self.input_frame,
            text="STOP",
            width=80,
            height=45,
            font=("Helvetica", 14, "bold"),
            command=self.stop_generation,
            fg_color=THEME['error'],
            hover_color="#F08080",
            corner_radius=12
        )
        self.stop_button.pack(side="right", padx=(0, 10))
        
        self.send_button = ctk.CTkButton(
            self.input_frame,
            text="Send",
            width=80,
            height=45,
            font=("Helvetica", 14, "bold"),
            command=self.send_message,
            fg_color=THEME['primary'],
            hover_color=THEME['secondary'],
            corner_radius=12
        )
        self.send_button.pack(side="right")
        
        self.input_field.bind("<Return>", self.handle_return)
        self.input_field.bind("<Shift-Return>", self.handle_shift_return)

    def show_mode_info(self):
        mode_info = (
            "데이터 모드 안내:\n\n"
            "- 자동 감지: 질문 내용에 따라 자동으로 CSV 데이터를 검색합니다.\n\n"
            "- CSV 모드: 호선/블록/스톡 데이터를 검색합니다.\n"
            "  예시 질문: 'HMD로 시작하는 CASE_NO 정보', '대불 사업장의 총 스톡 수는?'"
        )
        messagebox.showinfo("모드 정보", mode_info)
    
    def _rag_query_thread(self, query, max_tokens):
        mode = self.selected_mode.get()
        if mode == "CSV":
            query = "CSV에서 " + query
        # TXT 모드 분기 제거
        answer, graph_data = self.rag_system.answer_query(query, max_tokens=max_tokens)
        self.response_queue.put((answer, graph_data))
        
    def reload_data(self):
        if self.rag_system:
            try:
                self.rag_system.load_data()
                messagebox.showinfo("데이터 재로드", "새로운 데이터가 성공적으로 로드되었습니다.")
            except Exception as e:
                messagebox.showerror("오류", f"데이터 로드 실패: {str(e)}")
        else:
            messagebox.showwarning("경고", "RAG 시스템이 초기화되지 않았습니다.")

    def add_chat_message(self, message: str, is_user: bool = True, graph_data=None):
        logger.debug(f"Adding chat message: {message[:1000]}... (total length: {len(message)})")
        chat_message = ChatMessage(self.chat_container, message, is_user, graph_data=graph_data)
        chat_message.pack(fill="x", padx=10, pady=5, anchor="n")
        self.chat_container.update_idletasks()
        self.chat_container._parent_canvas.yview_moveto(1.0)
        if is_user:
            self.conversation_history.append({"role": "user", "content": message})
        else:
            self.conversation_history.append({"role": "assistant", "content": message})
        total_tokens = sum(len(msg["content"].split()) for msg in self.conversation_history)
        self.token_count_label.configure(text=f"Estimated tokens: {total_tokens}")

    def on_login_success(self, model_path, use_vector_store=True):
        try:
            self.rag_system = LangChainEquipmentSystem(model_path=model_path, load_existing=use_vector_store)
            self.user_info.configure(text=f"Department: {self.user_manager.current_user.department}")
            self.conversation_history = []
            self.deiconify()
            model_filename = Path(model_path).stem
            self.model_label.configure(text=f"Loaded Model: {model_filename}")
            vector_store_status = "재사용됨" if use_vector_store and os.path.exists("faiss_index") else "새로 생성됨"
            welcome_message = (
                "# LangChain RAG 시스템에 오신 것을 환영합니다!\n\n"
                f"**로드된 모델:** {model_filename}  \n"
                f"**부서:** {self.user_manager.current_user.department}  \n"
                f"**벡터 스토어:** {vector_store_status}\n\n"
                "이 시스템은 LangChain을 활용한 RAG(Retrieval-Augmented Generation) 기술로 "
                "스톡 데이터를 분석하고 질문에 답변합니다.\n\n"
                "## 개발자:\n- AI 엔지니어 BLAKC J\n\n"
                "질문을 입력해 주세요."
            )
            self.add_chat_message(welcome_message, is_user=False)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.destroy()

    def show_login(self):
        ModernLoginWindow(self.user_manager, self.on_login_success)

    def logout(self):
        if messagebox.askyesno("로그아웃", "로그아웃 하시겠습니까?"):
            self.rag_system = None
            self.conversation_history = []
            self.user_info.configure(text="Not logged in")
            for widget in self.chat_container.winfo_children():
                widget.destroy()
            self.withdraw()
            self.show_login()

    def choose_pdf_mode(self):
        mode = simpledialog.askstring("PDF 처리", "모드를 선택하세요 (요약 / 상세):", parent=self)
        if mode not in ["요약", "상세"]:
            messagebox.showerror("오류", "올바른 모드를 입력하세요: 요약 또는 상세")
            return
        self.upload_and_process_pdf(mode)

    def upload_and_process_pdf(self, mode):
        file_path = askopenfilename(title="PDF 파일 선택", filetypes=[("PDF files", "*.pdf")])
        if not file_path:
            return
        try:
            pdf_text = self.extract_text_from_pdf(file_path)
            if not pdf_text.strip():
                messagebox.showerror("오류", "PDF에서 텍스트를 찾을 수 없습니다.")
                return
            self.add_chat_message(f"PDF 내용:\n\n{pdf_text[:1000]}...\n\n(이하 생략)", is_user=True)
            if mode == "요약":
                self.summarize_text(pdf_text)
            elif mode == "상세":
                self.detail_analyze_text(pdf_text)
        except Exception as e:
            messagebox.showerror("오류", f"PDF 읽기 오류: {str(e)}")

    def extract_text_from_pdf(self, file_path):
        doc = fitz.open(file_path)
        text = "".join(page.get_text("text") + "\n" for page in doc).strip()
        return re.sub(r'Boardmix', '', text)

    def summarize_text(self, text):
        user_prompt = f"다음 내용을 100자 이내로 요약해줘:\n{text[:4000]}"
        self.add_chat_message("⏳ PDF 요약을 진행 중...", is_user=False)
        thread = threading.Thread(target=self._summarize_pdf_thread, args=(user_prompt,))
        thread.start()

    def detail_analyze_text(self, text):
        user_prompt = f"다음 문서를 자세히 분석하고 주요 내용을 정리해줘:\n{text[:6000]}"
        self.add_chat_message("⏳ PDF 상세 분석을 진행 중...", is_user=False)
        thread = threading.Thread(target=self._detail_analyze_pdf_thread, args=(user_prompt,))
        thread.start()

    def _summarize_pdf_thread(self, prompt):
        try:
            response = self.rag_system.llm.invoke(prompt)
            self.response_queue.put(f"요약 결과:\n{response}")
        except Exception as e:
            self.response_queue.put(f"요약 생성 중 오류 발생: {str(e)}")

    def _detail_analyze_pdf_thread(self, prompt):
        try:
            response = self.rag_system.llm.invoke(prompt)
            self.response_queue.put(f"상세 분석 결과:\n{response}")
        except Exception as e:
            self.response_queue.put(f"상세 분석 생성 중 오류 발생: {str(e)}")

    def send_message(self):
        user_input = self.input_field.get("0.0", "end").strip()
        if not user_input or self.is_generating:
            return
        self.add_chat_message(user_input, is_user=True)
        self.input_field.delete("0.0", "end")
        self.is_generating = True
        max_tokens = min(int(self.token_slider.get()), 2048)
        thread = threading.Thread(target=self._rag_query_thread, args=(user_input, max_tokens))
        thread.start()
        self.after(100, self.check_response)

    def check_response(self):
        if not self.response_queue.empty():
            response_data = self.response_queue.get()
            if isinstance(response_data, tuple) and len(response_data) == 2:
                response_text, graph_data = response_data
                self.add_chat_message(response_text, is_user=False, graph_data=graph_data)
            else:
                self.add_chat_message(response_data, is_user=False)
            self.is_generating = False
        elif self.is_generating:
            self.after(100, self.check_response)

    def clear_conversation(self):
        self.conversation_history = []
        for widget in self.chat_container.winfo_children():
            widget.destroy()
        self.token_count_label.configure(text="Estimated tokens: 0")

    def stop_generation(self):
        messagebox.showinfo("정보", "스레드 기반 RAG, 중도 중지는 지원하지 않습니다.")

    def toggle_speech(self):
        self.speech_enabled = not self.speech_enabled
        self.speech_switch.configure(text="음성 ON" if self.speech_enabled else "음성 OFF")

    def update_token_label(self, value):
        self.token_value_label.configure(text=str(int(value)))

    def handle_return(self, event):
        self.send_message()
        return "break"

    def handle_shift_return(self, event):
        self.input_field.insert("insert", "\n")

    def adjust_input_height(self, event=None):
        if not hasattr(self, 'input_field'):
            return
        self.input_field.edit_modified(False)
        text = self.input_field.get("1.0", "end-1c")
        num_lines = text.count('\n') + 1
        min_height, max_height, line_height = 80, 250, 20
        needed_height = max(min_height, min(max_height, num_lines * line_height))
        if self.input_field.cget("height") != needed_height:
            self.input_field.configure(height=needed_height)
            self.input_frame.update_idletasks()

    def open_prompt_window(self):
        try:
            current_template = self.rag_system.prompt.template
        except AttributeError:
            messagebox.showerror("오류", "프롬프트 템플릿을 불러올 수 없습니다.")
            return
        prompt_window = ctk.CTkToplevel(self)
        prompt_window.title("프롬프트 템플릿 수정")
        prompt_window.geometry("800x600")
        prompt_window.focus_set()
        ctk.CTkLabel(
            prompt_window,
            text="프롬프트 템플릿을 수정하세요. {context}와 {question} 태그는 반드시 유지해야 합니다.",
            font=("Helvetica", 14, "bold"),
            wraplength=750
        ).pack(pady=(20, 10), padx=20, anchor='w')
        prompt_text = ctk.CTkTextbox(prompt_window, font=("Helvetica", 12), wrap="word", height=450)
        prompt_text.pack(padx=20, pady=10, fill="both", expand=True)
        prompt_text.insert("1.0", current_template)
        button_frame = ctk.CTkFrame(prompt_window, fg_color="transparent")
        button_frame.pack(pady=20, fill='x')
        cancel_button = ctk.CTkButton(
            button_frame,
            text="취소",
            font=("Helvetica", 12, "bold"),
            width=100,
            command=prompt_window.destroy,
            fg_color=THEME['error'],
            hover_color="#F08080"
        )
        cancel_button.pack(side='left', padx=20)
        def reset_to_default():
            default_template = """
            당신은 데이터 분석 전문가입니다. 아래 제공된 정보를 바탕으로 질문에 정확하고 간결하게 답변하세요.
            
            ### 데이터
            {context}
            
            ### 사용자 질문
            {question}
            
            아래 가이드라인을 따라 답변해주세요:
            1. 가독성을 위해 날짜 형식을 그대로 유지해주세요 (예: 2025년 1월 1일).
            2. 수치는 적절한 소수점 자리까지 표시해주세요.
            3. 계산이 필요한 경우 정확히 계산해서 알려주세요.
            4. 특정 조건이 언급된 경우 해당 정보만 제공해주세요.
            5. 결과는 "결과값:"과 "분석:"으로 나누어 제시하며, 분석은 간단한 핵심 인사이트만 포함해주세요.
            
            ### 응답
            """
            prompt_text.delete("1.0", "end")
            prompt_text.insert("1.0", default_template)
        reset_button = ctk.CTkButton(
            button_frame,
            text="기본값 복원",
            font=("Helvetica", 12, "bold"),
            width=100,
            command=reset_to_default,
            fg_color="#888888",
            hover_color="#AAAAAA"
        )
        reset_button.pack(side='left', padx=20)
        def save_prompt():
            new_template = prompt_text.get("1.0", "end-1c")
            if "{context}" not in new_template or "{question}" not in new_template:
                messagebox.showerror("오류", "프롬프트에는 {context}와 {question} 태그가 포함되어야 합니다.")
                return
            try:
                new_prompt = PromptTemplate(template=new_template, input_variables=["context", "question"])
                self.rag_system.prompt = new_prompt
                self.rag_system.qa_chain = RetrievalQA.from_chain_type(
                    llm=self.rag_system.llm,
                    chain_type="stuff",
                    retriever=self.rag_system.retriever,
                    chain_type_kwargs={"prompt": new_prompt}
                )
                messagebox.showinfo("저장 완료", "프롬프트 템플릿이 성공적으로 업데이트되었습니다.")
                prompt_window.destroy()
            except Exception as e:
                messagebox.showerror("오류", f"프롬프트 업데이트 중 오류 발생: {str(e)}")
        save_button = ctk.CTkButton(
            button_frame,
            text="저장",
            font=("Helvetica", 12, "bold"),
            width=100,
            command=save_prompt,
            fg_color=THEME['primary'],
            hover_color=THEME['secondary']
        )
        save_button.pack(side='right', padx=20)

if __name__ == "__main__":
    app = ModernChatUI()
    app.mainloop()