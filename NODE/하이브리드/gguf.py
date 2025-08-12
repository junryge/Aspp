 -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 08:20:20 2025

@author: 파이썬AI
"""

import os
import sys
import json
import time
import threading
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import requests
from datetime import datetime
import traceback
import re
import shutil

# GUI 라이브러리
import customtkinter as ctk
from tkinter import filedialog, messagebox, simpledialog

# 데이터 시각화
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# LangChain 및 임베딩 관련
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 파일 처리 관련
try:
    import fitz  # PyMuPDF - PDF 처리
except ImportError:
    print("PyMuPDF가 설치되지 않았습니다. PDF 기능이 제한됩니다.")

try:
    from langchain_community.llms import LlamaCpp
except ImportError:
    try:
        from langchain.llms import LlamaCpp
    except ImportError:
        print("LlamaCpp를 찾을 수 없습니다. 모델 로딩이 불가능합니다.")

# 웹 처리 관련
try:
    from bs4 import BeautifulSoup
    import wikipedia
    WEB_SUPPORT = True
except ImportError:
    print("BeautifulSoup 또는 Wikipedia 패키지가 설치되지 않았습니다. 웹 기능이 제한됩니다.")
    WEB_SUPPORT = False

# HTML 렌더링 관련
try:
    from tkinterweb import HtmlFrame
    TKINTERWEB_AVAILABLE = True
except ImportError:
    TKINTERWEB_AVAILABLE = False
    print("tkinterweb가 설치되지 않았습니다. HTML 렌더링이 제한됩니다.")

try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False
    print("markdown 패키지가 설치되지 않았습니다. 마크다운 렌더링이 제한됩니다.")

# 로깅 설정
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# 전역 테마 설정
THEME = {
    'primary': '#2E86C1',
    'secondary': '#AED6F1',
    'accent': '#F39C12',
    'error': '#E74C3C',
    'success': '#2ECC71',
    'warning': '#F1C40F',
    'background': '#F5F5F5',
    'surface': '#FFFFFF',
    'text': '#2C3E50',
    'text_secondary': '#7F8C8D'
}

# 데이터 디렉토리 생성
def create_app_directories():
    """애플리케이션 필요 디렉토리 생성"""
    dirs = [
        "data",
        "data/text",
        "data/csv",
        "data/pdf",
        "data/web",
        "vectors",
        "vectors/text",
        "vectors/csv",
        "vectors/pdf",
        "vectors/web",
        "models",
        "logs",
        "configs"
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"디렉토리 생성 또는 확인: {dir_path}")

# 설정 관리 클래스
class ConfigManager:
    """애플리케이션 설정 관리"""
    
    def __init__(self):
        self.config_file = "configs/app_config.json"
        self.defaults = {
            "theme": "system",  # system, light, dark
            "model_path": "",
            "embedding_model": "jhgan/ko-sroberta-multitask",
            "context_size": 2048,
            "temperature": 0.1,
            "max_tokens": 1000,
            "auto_save": True,
            "recent_files": [],
            "language": "ko",  # ko, en, ja, zh
            "wiki_search_enabled": True,
            "web_search_enabled": True,
            "cache_enabled": True,
            "cache_size": 100
        }
        self.config = self.load_config()
    
    def load_config(self):
        """설정 파일 로드"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    # 누락된 기본값 추가
                    for key, value in self.defaults.items():
                        if key not in config:
                            config[key] = value
                    return config
            except Exception as e:
                logger.error(f"설정 파일 로드 오류: {str(e)}")
                return self.defaults.copy()
        else:
            # 초기 설정 파일 생성
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            self.save_config(self.defaults)
            return self.defaults.copy()
    
    def save_config(self, config=None):
        """설정 파일 저장"""
        if config is None:
            config = self.config
        
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            logger.info("설정 파일 저장 완료")
            return True
        except Exception as e:
            logger.error(f"설정 파일 저장 오류: {str(e)}")
            return False
    
    def get(self, key, default=None):
        """설정 값 가져오기"""
        return self.config.get(key, default)
    
    def set(self, key, value):
        """설정 값 설정 및 저장"""
        self.config[key] = value
        if self.get("auto_save", True):
            self.save_config()
    
    def add_recent_file(self, file_path):
        """최근 파일 추가"""
        recent = self.get("recent_files", [])
        if file_path in recent:
            recent.remove(file_path)
        recent.insert(0, file_path)
        # 최대 10개만 유지
        self.set("recent_files", recent[:10])
    
    def clear_recent_files(self):
        """최근 파일 목록 초기화"""
        self.set("recent_files", [])

# 캐싱 시스템
class ResponseCache:
    """쿼리 응답 캐싱 시스템"""
    
    def __init__(self, max_size=100):
        self.cache = {}  # {query_hash: (query, response, timestamp)}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        self.load_cache()
    
    def load_cache(self):
        """캐시 파일 로드"""
        cache_file = "data/response_cache.json"
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.cache = data.get("cache", {})
                    self.hits = data.get("hits", 0)
                    self.misses = data.get("misses", 0)
                logger.info(f"캐시 로드 완료: {len(self.cache)}개 항목")
            except Exception as e:
                logger.error(f"캐시 파일 로드 오류: {str(e)}")
                self.cache = {}
                self.hits = 0
                self.misses = 0
    
    def save_cache(self):
        """캐시 파일 저장"""
        cache_file = "data/response_cache.json"
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                data = {
                    "cache": self.cache,
                    "hits": self.hits,
                    "misses": self.misses,
                    "last_saved": time.time()
                }
                json.dump(data, f, ensure_ascii=False)
            logger.info(f"캐시 저장 완료: {len(self.cache)}개 항목")
            return True
        except Exception as e:
            logger.error(f"캐시 파일 저장 오류: {str(e)}")
            return False
    
    def get(self, query):
        """쿼리에 대한 캐시된 응답 반환"""
        query_hash = self._hash_query(query)
        
        if query_hash in self.cache:
            self.hits += 1
            cached_query, response, timestamp = self.cache[query_hash]
            # 유사도 검사 (간단한 구현)
            similarity = self._calculate_similarity(query, cached_query)
            if similarity > 0.8:  # 80% 이상 유사하면 캐시 사용
                logger.info(f"캐시 히트: {query} (유사도: {similarity:.2f})")
                return response
            
        self.misses += 1
        return None
    
    def add(self, query, response):
        """응답을 캐시에 추가"""
        query_hash = self._hash_query(query)
        self.cache[query_hash] = (query, response, time.time())
        
        # 캐시 크기 제한
        if len(self.cache) > self.max_size:
            self._prune_cache()
        
        # 주기적으로 저장 (10번의 캐싱마다)
        if (self.hits + self.misses) % 10 == 0:
            self.save_cache()
    
    def clear(self):
        """캐시 전체 삭제"""
        self.cache = {}
        self.hits = 0
        self.misses = 0
        self.save_cache()
    
    def _hash_query(self, query):
        """쿼리 해시 계산"""
        import hashlib
        # 단순 문자열 해시
        query_normalized = query.lower().strip()
        return hashlib.md5(query_normalized.encode('utf-8')).hexdigest()
    
    def _calculate_similarity(self, query1, query2):
        """두 쿼리 간의 유사도 계산 (간단한 구현)"""
        # 문자열 전처리
        q1 = query1.lower().strip()
        q2 = query2.lower().strip()
        
        # 정확히 일치하면 최대 유사도
        if q1 == q2:
            return 1.0
        
        # 한 쿼리가 다른 쿼리에 포함되는 경우
        if q1 in q2:
            return len(q1) / len(q2)
        if q2 in q1:
            return len(q2) / len(q1)
        
        # 단어 수준 유사도 (공통 단어 비율)
        words1 = set(q1.split())
        words2 = set(q2.split())
        
        common_words = words1.intersection(words2)
        all_words = words1.union(words2)
        
        if not all_words:
            return 0.0
        
        return len(common_words) / len(all_words)
    
    def _prune_cache(self):
        """오래된 캐시 항목 제거"""
        # 타임스탬프 기준으로 정렬
        sorted_items = sorted(self.cache.items(), key=lambda x: x[1][2])
        
        # 가장 오래된 항목의 25%를 제거
        items_to_remove = len(sorted_items) // 4
        for i in range(items_to_remove):
            key_to_remove = sorted_items[i][0]
            del self.cache[key_to_remove]
        
        logger.info(f"캐시 정리 완료: {items_to_remove}개 항목 제거")

# 문서 처리 및 임베딩 클래스
class DocumentProcessor:
    """다양한 문서 유형을 처리하고 임베딩하는 클래스"""
    
    def __init__(self, config_manager):
        self.config = config_manager
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self._init_embedding_model()
    
    def _init_embedding_model(self):
        """임베딩 모델 초기화"""
        embedding_model_name = self.config.get("embedding_model", "jhgan/ko-sroberta-multitask")
        try:
            self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
            logger.info(f"임베딩 모델 로드 성공: {embedding_model_name}")
        except Exception as e:
            logger.error(f"임베딩 모델 로드 오류: {str(e)}")
            self.embedding_model = None
            raise RuntimeError(f"임베딩 모델 로드 실패: {str(e)}")
    
    def process_text(self, text, metadata=None):
        """텍스트 처리 및 청크로 분할"""
        if not text or not text.strip():
            return []
        
        if metadata is None:
            metadata = {"source": "text"}
        
        chunks = self.text_splitter.split_text(text)
        docs = [Document(page_content=chunk, metadata=metadata) for chunk in chunks]
        return docs
    
    def process_csv(self, df, metadata=None):
        """CSV 데이터프레임 처리"""
        if df.empty:
            return []
        
        if metadata is None:
            metadata = {"source": "csv"}
        
        docs = []
        for i, row in df.iterrows():
            # 행을 문자열로 변환
            content = " ".join([f"{col}: {row[col]}" for col in df.columns])
            doc_metadata = metadata.copy()
            doc_metadata["row"] = i
            docs.append(Document(page_content=content, metadata=doc_metadata))
        
        return docs
    
    def process_pdf(self, pdf_path, metadata=None):
        """PDF 파일 처리"""
        if not os.path.exists(pdf_path):
            logger.error(f"PDF 파일이 존재하지 않습니다: {pdf_path}")
            return []
        
        if metadata is None:
            metadata = {"source": "pdf", "file_path": pdf_path}
        
        try:
            text = ""
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text += page.get_text()
            
            return self.process_text(text, metadata)
        except Exception as e:
            logger.error(f"PDF 처리 오류: {str(e)}")
            return []
    
    def process_web_page(self, url, metadata=None):
        """웹 페이지 처리"""
        if not WEB_SUPPORT:
            logger.error("웹 처리 라이브러리가 설치되지 않았습니다.")
            return []
        
        if metadata is None:
            metadata = {"source": "web", "url": url}
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 불필요한 요소 제거
            for tag in soup(['script', 'style', 'meta', 'nav', 'footer']):
                tag.decompose()
            
            text = soup.get_text(separator='\n')
            # 여러 개의 줄바꿈 정리
            text = re.sub(r'\n+', '\n', text)
            # 여러 개의 공백 정리
            text = re.sub(r'\s+', ' ', text)
            
            return self.process_text(text, metadata)
        except Exception as e:
            logger.error(f"웹 페이지 처리 오류: {str(e)}")
            return []
    
    def process_wikipedia(self, query, language="ko", metadata=None):
        """위키피디아 페이지 처리"""
        if not WEB_SUPPORT:
            logger.error("웹 처리 라이브러리가 설치되지 않았습니다.")
            return []
        
        if metadata is None:
            metadata = {"source": "wikipedia", "query": query}
        
        try:
            # 언어 설정
            wikipedia.set_lang(language)
            
            # 검색
            search_results = wikipedia.search(query, results=3)
            if not search_results:
                return []
            
            docs = []
            for title in search_results:
                try:
                    page = wikipedia.page(title)
                    content = page.content
                    page_metadata = metadata.copy()
                    page_metadata["title"] = title
                    page_metadata["url"] = page.url
                    
                    # 텍스트 처리
                    page_docs = self.process_text(content, page_metadata)
                    docs.extend(page_docs)
                except Exception as e:
                    logger.warning(f"위키피디아 페이지 '{title}' 처리 오류: {str(e)}")
                    continue
            
            return docs
        except Exception as e:
            logger.error(f"위키피디아 처리 오류: {str(e)}")
            return []
    
    def create_or_update_vector_store(self, documents, vector_store_path, vector_store=None):
        """벡터 스토어 생성 또는 업데이트"""
        if not documents:
            logger.warning("벡터 스토어 생성할 문서가 없습니다.")
            return vector_store
        
        try:
            if vector_store is None:
                # 새 벡터 스토어 생성
                vector_store = FAISS.from_documents(documents, self.embedding_model)
                logger.info(f"새 벡터 스토어 생성 완료: {len(documents)}개 문서")
            else:
                # 기존 벡터 스토어에 문서 추가
                vector_store.add_documents(documents)
                logger.info(f"벡터 스토어 업데이트 완료: {len(documents)}개 문서 추가")
            
            # 벡터 스토어 저장
            os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)
            vector_store.save_local(vector_store_path)
            logger.info(f"벡터 스토어 저장 완료: {vector_store_path}")
            
            return vector_store
        except Exception as e:
            logger.error(f"벡터 스토어 생성/업데이트 오류: {str(e)}")
            traceback.print_exc()
            return vector_store
    
    def load_vector_store(self, vector_store_path):
        """벡터 스토어 로드"""
        if not os.path.exists(vector_store_path):
            logger.warning(f"벡터 스토어가 존재하지 않습니다: {vector_store_path}")
            return None
        
        try:
            vector_store = FAISS.load_local(vector_store_path, self.embedding_model, allow_dangerous_deserialization=True)
            logger.info(f"벡터 스토어 로드 완료: {vector_store_path}")
            return vector_store
        except Exception as e:
            logger.error(f"벡터 스토어 로드 오류: {str(e)}")
            traceback.print_exc()
            return None

# RAG 시스템 클래스
class RAGSystem:
    """검색 증강 생성 시스템"""
    
    def __init__(self, config_manager, document_processor):
        self.config = config_manager
        self.doc_processor = document_processor
        self.llm = None
        self.vector_stores = {
            "text": None,
            "csv": None,
            "pdf": None,
            "web": None
        }
        self.prompt_template = self._create_default_prompt()
        self.cache = ResponseCache(max_size=self.config.get("cache_size", 100))
        self._init_vector_stores()
        
    def _create_default_prompt(self):
        """기본 프롬프트 템플릿 생성"""
        template = """
        당신은 지식 검색 시스템입니다. 아래 제공된 컨텍스트 정보를 바탕으로 사용자의 질문에 정확하고 도움이 되는 답변을 제공하세요.
        
        ### 컨텍스트 정보:
        {context}
        
        ### 사용자 질문:
        {question}
        
        답변 가이드라인:
        1. 컨텍스트에 포함된 정보만 사용하여 답변하세요.
        2. 컨텍스트에 관련 정보가 없으면 "제공된 정보에서 관련 내용을 찾을 수 없습니다"라고 말하세요.
        3. 사실과 의견을 구분하고, 추측을 피하세요.
        4. 명확하고 간결하게 답변하되, 필요한 세부 정보는 충분히 제공하세요.
        5. 복잡한 개념은 쉽게 설명하고, 필요한 경우 예시를 제공하세요.
        6. 마크다운 형식을 사용하여 답변을 구조화하세요.
        
        ### 답변:
        """
        
        return PromptTemplate(template=template, input_variables=["context", "question"])
    
    def _init_vector_stores(self):
        """벡터 스토어 초기화"""
        vector_store_paths = {
            "text": "vectors/text",
            "csv": "vectors/csv",
            "pdf": "vectors/pdf",
            "web": "vectors/web"
        }
        
        for source_type, path in vector_store_paths.items():
            try:
                self.vector_stores[source_type] = self.doc_processor.load_vector_store(path)
                if self.vector_stores[source_type]:
                    logger.info(f"{source_type.upper()} 벡터 스토어 로드 완료")
            except Exception as e:
                logger.error(f"{source_type.upper()} 벡터 스토어 로드 오류: {str(e)}")
        
    def init_llm(self, model_path=None):
        """LLM 초기화"""
        if model_path is None:
            model_path = self.config.get("model_path", "")
        
        if not model_path or not os.path.exists(model_path):
            logger.error(f"모델 파일이 존재하지 않습니다: {model_path}")
            return False
        
        try:
            context_size = self.config.get("context_size", 2048)
            self.llm = LlamaCpp(
                model_path=model_path,
                n_ctx=context_size,
                n_batch=512,
                use_mlock=True,
                n_gpu_layers=64,  # GPU 사용 시 조정
                device='cpu',  # 'cuda'로 변경하여 GPU 사용 가능
                rope_scaling={"type": "linear", "factor": 2.0}
            )
            logger.info(f"LLM 초기화 완료: {model_path}")
            self.config.set("model_path", model_path)
            return True
        except Exception as e:
            logger.error(f"LLM 초기화 오류: {str(e)}")
            traceback.print_exc()
            return False
    
    def add_text_document(self, text, title, metadata=None):
        """텍스트 문서 추가"""
        if metadata is None:
            metadata = {"source": "text", "title": title}
        else:
            metadata["source"] = "text"
            metadata["title"] = title
        
        docs = self.doc_processor.process_text(text, metadata)
        self.vector_stores["text"] = self.doc_processor.create_or_update_vector_store(
            docs, "vectors/text", self.vector_stores["text"]
        )
        return len(docs)
    
    def add_csv_document(self, df, title, metadata=None):
        """CSV 문서 추가"""
        if metadata is None:
            metadata = {"source": "csv", "title": title}
        else:
            metadata["source"] = "csv"
            metadata["title"] = title
        
        docs = self.doc_processor.process_csv(df, metadata)
        self.vector_stores["csv"] = self.doc_processor.create_or_update_vector_store(
            docs, "vectors/csv", self.vector_stores["csv"]
        )
        return len(docs)
    
    def add_pdf_document(self, pdf_path, metadata=None):
        """PDF 문서 추가"""
        if not os.path.exists(pdf_path):
            logger.error(f"PDF 파일이 존재하지 않습니다: {pdf_path}")
            return 0
        
        title = os.path.basename(pdf_path)
        
        if metadata is None:
            metadata = {"source": "pdf", "title": title, "file_path": pdf_path}
        else:
            metadata["source"] = "pdf"
            metadata["title"] = title
            metadata["file_path"] = pdf_path
        
        docs = self.doc_processor.process_pdf(pdf_path, metadata)
        self.vector_stores["pdf"] = self.doc_processor.create_or_update_vector_store(
            docs, "vectors/pdf", self.vector_stores["pdf"]
        )
        return len(docs)
    
    def add_web_document(self, url, metadata=None):
        """웹 문서 추가"""
        if not WEB_SUPPORT:
            logger.error("웹 처리 라이브러리가 설치되지 않았습니다.")
            return 0
        
        if metadata is None:
            metadata = {"source": "web", "url": url}
        else:
            metadata["source"] = "web"
            metadata["url"] = url
        
        docs = self.doc_processor.process_web_page(url, metadata)
        self.vector_stores["web"] = self.doc_processor.create_or_update_vector_store(
            docs, "vectors/web", self.vector_stores["web"]
        )
        return len(docs)
    
    def search_documents(self, query, source_types=None, k=5):
        """문서 검색"""
        if source_types is None:
            # 모든 벡터 스토어 사용
            source_types = ["text", "csv", "pdf", "web"]
        
        all_docs = []
        for source_type in source_types:
            vector_store = self.vector_stores.get(source_type)
            if vector_store:
                try:
                    docs = vector_store.similarity_search_with_score(query, k=k)
                    all_docs.extend(docs)
                except Exception as e:
                    logger.error(f"{source_type} 문서 검색 오류: {str(e)}")
        
        # 점수 기준 정렬 (점수가 낮을수록 유사도 높음)
        all_docs.sort(key=lambda x: x[1])
        
        # 상위 k개 반환
        return all_docs[:k]
    
    def answer_query(self, query, source_types=None):
        """쿼리 응답 생성"""
        # 캐시 확인
        if self.config.get("cache_enabled", True):
            cached_response = self.cache.get(query)
            if cached_response:
                return cached_response, None  # 그래프 데이터 없음
        
        # LLM 확인
        if not self.llm:
            return "LLM이 초기화되지 않았습니다. 모델을 먼저 로드해주세요.", None
        
        # Wiki 검색
        wiki_content = None
        if self.config.get("wiki_search_enabled", True) and WEB_SUPPORT:
            try:
                language = self.config.get("language", "ko")
                wiki_docs = self.doc_processor.process_wikipedia(query, language)
                if wiki_docs:
                    wiki_content = "\n\n".join([doc.page_content for doc in wiki_docs[:2]])
            except Exception as e:
                logger.error(f"위키 검색 오류: {str(e)}")
        
        # 문서 검색
        docs = self.search_documents(query, source_types)
        
        # 검색 결과가 없을 경우
        if not docs and not wiki_content:
            message = "질문과 관련된 정보를 찾을 수 없습니다."
            # 일반 LLM 응답 생성
            try:
                temperature = self.config.get("temperature", 0.1)
                max_tokens = self.config.get("max_tokens", 1000)
                answer = self.llm.invoke(query, temperature=temperature, max_tokens=max_tokens)
                message = f"{message}\n\n일반적인 응답:\n\n{answer}"
            except Exception as e:
                logger.error(f"LLM 응답 생성 오류: {str(e)}")
            
            if self.config.get("cache_enabled", True):
                self.cache.add(query, message)
            
            return message, None
        
        # 컨텍스트 구성
        context_parts = []
        
        # 벡터 스토어 검색 결과 추가
        for doc, score in docs:
            # 점수가 1.0 이상이면 관련성이 낮다고 판단
            if score > 1.0:
                continue
            
            # 메타데이터에서 소스 정보 추출
            source_type = doc.metadata.get("source", "unknown")
            title = doc.metadata.get("title", "Unknown")
            
            # 소스 타입별 컨텍스트 포맷
            if source_type == "text":
                context_parts.append(f"[텍스트: {title}]\n{doc.page_content}")
            elif source_type == "csv":
                row = doc.metadata.get("row", "Unknown")
                context_parts.append(f"[데이터: {title}, 행: {row}]\n{doc.page_content}")
            elif source_type == "pdf":
                context_parts.append(f"[PDF: {title}]\n{doc.page_content}")
            elif source_type == "web":
                url = doc.metadata.get("url", "Unknown")
                context_parts.append(f"[웹: {title}, URL: {url}]\n{doc.page_content}")
            else:
                context_parts.append(f"[{source_type}: {title}]\n{doc.page_content}")
        
        # 위키 컨텐츠 추가
        if wiki_content:
            context_parts.append(f"[위키피디아 정보]\n{wiki_content}")
        
        # 컨텍스트 결합
        context = "\n\n".join(context_parts)
        
        # 프롬프트 구성
        prompt = self.prompt_template.format(context=context, question=query)
        
        # LLM 응답 생성
        try:
            temperature = self.config.get("temperature", 0.1)
            max_tokens = self.config.get("max_tokens", 1000)
            response = self.llm.invoke(prompt, temperature=temperature, max_tokens=max_tokens)
            
            # 캐시에 추가
            if self.config.get("cache_enabled", True):
                self.cache.add(query, response)
            
            return response, None  # 그래프 데이터는 없음 (필요시 추가)
        except Exception as e:
            error_message = f"응답 생성 중 오류가 발생했습니다: {str(e)}"
            logger.error(error_message)
            return error_message, None
    
    def generate_visualization(self, query, data_source="csv"):
        """데이터 시각화 생성"""
        if data_source != "csv" or not self.vector_stores.get("csv"):
            return "시각화는 CSV 데이터에서만 지원됩니다.", None
        
        # TODO: CSV 데이터를 기반으로 한 시각화 구현
        return "시각화 기능이 아직 구현되지 않았습니다.", None

# 메인 애플리케이션
class KnowledgeSearchApp(ctk.CTk):
    """지식 검색 애플리케이션 메인 클래스"""
    
    def __init__(self):
        super().__init__()
        
        # 앱 초기화
        self.title("지식 검색 시스템")
        self.geometry("1200x800")
        
        # 앱 디렉토리 생성
        create_app_directories()
        
        # 설정 관리자 초기화
        self.config_manager = ConfigManager()
        
        # 테마 설정
        self.apply_theme()
        
        # 문서 처리기 초기화
        try:
            self.doc_processor = DocumentProcessor(self.config_manager)
            # RAG 시스템 초기화
            self.rag_system = RAGSystem(self.config_manager, self.doc_processor)
        except Exception as e:
            self.show_error_and_exit(f"초기화 오류: {str(e)}")
        
        # 대화 이력
        self.conversation = []
        
        # 응답 생성 플래그
        self.is_generating = False
        
        # UI 구성
        self.setup_ui()
        
        # 모델 로드 확인
        self.check_model()
    
    def apply_theme(self):
        """테마 적용"""
        theme_mode = self.config_manager.get("theme", "system")
        ctk.set_appearance_mode(theme_mode)
        ctk.set_default_color_theme("blue")
    
    def show_error_and_exit(self, message):
        """오류 메시지 표시 후 종료"""
        messagebox.showerror("치명적 오류", message)
        self.quit()
        self.destroy()
        sys.exit(1)
    
    def check_model(self):
        """모델 로드 확인"""
        model_path = self.config_manager.get("model_path", "")
        if not model_path or not os.path.exists(model_path):
            self.after(500, self.show_model_dialog)
    
    def show_model_dialog(self):
        """모델 로드 대화상자 표시"""
        result = messagebox.askyesno(
            "모델 로드",
            "LLM 모델이 로드되지 않았습니다. 모델 파일을 선택하시겠습니까?"
        )
        if result:
            self.load_model()
        else:
            messagebox.showwarning(
                "제한된 기능",
                "모델이 로드되지 않으면 응답 생성이 불가능합니다. '설정 > 모델 로드'에서 나중에 모델을 로드할 수 있습니다."
            )
    
    def load_model(self):
        """모델 파일 선택 및 로드"""
        model_file = filedialog.askopenfilename(
            title="LLM 모델 파일 선택",
            filetypes=[("GGUF 파일", "*.gguf"), ("GGML 파일", "*.ggml"), ("bin 파일", "*.bin"), ("모든 파일", "*.*")],
            initialdir="models"
        )
        
        if not model_file:
            return
        
        # 로딩 대화상자 표시
        loading_dialog = self.create_loading_dialog("모델 로드 중...")
        
        # 별도 스레드에서 모델 로드
        threading.Thread(
            target=self._load_model_thread,
            args=(model_file, loading_dialog),
            daemon=True
        ).start()
    
    def _load_model_thread(self, model_file, loading_dialog):
        """모델 로드 스레드"""
        try:
            success = self.rag_system.init_llm(model_file)
            
            # UI 스레드에서 결과 처리
            self.after(100, lambda: self._handle_model_load_result(success, model_file, loading_dialog))
        except Exception as e:
            # UI 스레드에서 오류 처리
            # 수정된 부분:
            error_msg = str(e)
            def handle_error():
                self._handle_pdf_file_error(error_msg, loading_dialog)
            self.after(100, handle_error)
    
    def _handle_model_load_result(self, success, model_file, loading_dialog):
        """모델 로드 결과 처리"""
        loading_dialog.destroy()
        
        if success:
            messagebox.showinfo("성공", f"모델이 성공적으로 로드되었습니다:\n{os.path.basename(model_file)}")
            # 모델 정보 표시 업데이트
            self.model_info_label.configure(text=f"모델: {os.path.basename(model_file)}")
        else:
            messagebox.showerror("오류", f"모델 로드에 실패했습니다:\n{model_file}")
    
    def _handle_model_load_error(self, error_message, loading_dialog):
        """모델 로드 오류 처리"""
        loading_dialog.destroy()
        messagebox.showerror("오류", f"모델 로드 중 오류가 발생했습니다:\n{error_message}")
    
    def create_loading_dialog(self, message):
        """로딩 대화상자 생성"""
        dialog = ctk.CTkToplevel(self)
        dialog.title("처리 중")
        dialog.geometry("300x150")
        dialog.transient(self)
        dialog.grab_set()
        
        # 중앙에 배치
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry(f'+{x}+{y}')
        
        # 메시지 및 진행바
        ctk.CTkLabel(
            dialog,
            text=message,
            font=("Helvetica", 14, "bold")
        ).pack(pady=(20, 10))
        
        progress = ctk.CTkProgressBar(dialog, width=250)
        progress.pack(pady=(0, 20))
        progress.configure(mode="indeterminate")
        progress.start()
        
        return dialog
    
    def setup_ui(self):
        """UI 구성"""
        self.setup_menu()
        
        # 메인 레이아웃
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # 사이드바
        self.setup_sidebar()
        
        # 메인 컨텐츠
        self.setup_main_content()
    
    def setup_menu(self):
        """메뉴바 구성"""
        import tkinter as tk
        
        self.menu_bar = tk.Menu(self)
        
        # 파일 메뉴
        file_menu = tk.Menu(self.menu_bar, tearoff=0)
        file_menu.add_command(label="새 대화", command=self.clear_conversation)
        file_menu.add_separator()
        file_menu.add_command(label="텍스트 파일 추가", command=self.add_text_file)
        file_menu.add_command(label="CSV 파일 추가", command=self.add_csv_file)
        file_menu.add_command(label="PDF 파일 추가", command=self.add_pdf_file)
        file_menu.add_command(label="웹 페이지 추가", command=self.add_web_page)
        file_menu.add_separator()
        file_menu.add_command(label="대화 저장", command=self.save_conversation)
        file_menu.add_command(label="대화 불러오기", command=self.load_conversation)
        file_menu.add_separator()
        file_menu.add_command(label="종료", command=self.quit)
        self.menu_bar.add_cascade(label="파일", menu=file_menu)
        
        # 편집 메뉴
        edit_menu = tk.Menu(self.menu_bar, tearoff=0)
        edit_menu.add_command(label="복사", command=self.copy_selection)
        edit_menu.add_command(label="붙여넣기", command=self.paste_clipboard)
        edit_menu.add_separator()
        edit_menu.add_command(label="설정", command=self.show_settings)
        self.menu_bar.add_cascade(label="편집", menu=edit_menu)
        
        # 보기 메뉴
        view_menu = tk.Menu(self.menu_bar, tearoff=0)
        view_menu.add_command(label="라이트 모드", command=lambda: self.change_theme("light"))
        view_menu.add_command(label="다크 모드", command=lambda: self.change_theme("dark"))
        view_menu.add_command(label="시스템 설정", command=lambda: self.change_theme("system"))
        self.menu_bar.add_cascade(label="보기", menu=view_menu)
        
        # 도구 메뉴
        tools_menu = tk.Menu(self.menu_bar, tearoff=0)
        tools_menu.add_command(label="모델 로드", command=self.load_model)
        tools_menu.add_command(label="캐시 관리", command=self.manage_cache)
        tools_menu.add_command(label="벡터 저장소 관리", command=self.manage_vector_stores)
        self.menu_bar.add_cascade(label="도구", menu=tools_menu)
        
        # 도움말 메뉴
        help_menu = tk.Menu(self.menu_bar, tearoff=0)
        help_menu.add_command(label="사용 설명서", command=self.show_help)
        help_menu.add_command(label="정보", command=self.show_about)
        self.menu_bar.add_cascade(label="도움말", menu=help_menu)
        
        self.configure(menu=self.menu_bar)
    
    def setup_sidebar(self):
        """사이드바 구성"""
        sidebar_frame = ctk.CTkFrame(self, width=250, corner_radius=0)
        sidebar_frame.grid(row=0, column=0, sticky="nsew")
        sidebar_frame.grid_rowconfigure(10, weight=1)  # 빈 공간이 아래쪽으로 확장되도록
        sidebar_frame.grid_propagate(False)  # 크기 고정
        
        # 앱 제목
        app_title = ctk.CTkLabel(
            sidebar_frame, 
            text="지식 검색 시스템",
            font=("Helvetica", 20, "bold"),
            text_color=THEME['primary']
        )
        app_title.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")
        
        # 구분선
        separator1 = ctk.CTkFrame(sidebar_frame, height=1, fg_color=THEME["secondary"])
        separator1.grid(row=1, column=0, padx=20, pady=(0, 10), sticky="ew")
        
        # 모델 정보
        model_path = self.config_manager.get("model_path", "")
        model_name = os.path.basename(model_path) if model_path else "로드되지 않음"
        self.model_info_label = ctk.CTkLabel(
            sidebar_frame,
            text=f"모델: {model_name}",
            font=("Helvetica", 12),
            text_color=THEME['text_secondary']
        )
        self.model_info_label.grid(row=2, column=0, padx=20, pady=(0, 5), sticky="w")
        
        # 모델 로드 버튼
        model_button = ctk.CTkButton(
            sidebar_frame,
            text="모델 로드",
            command=self.load_model,
            height=30,
            font=("Helvetica", 12),
            fg_color=THEME['primary'],
            hover_color=THEME['secondary']
        )
        model_button.grid(row=3, column=0, padx=20, pady=(0, 10), sticky="ew")
        
        # 데이터 소스 라벨
        source_label = ctk.CTkLabel(
            sidebar_frame,
            text="데이터 소스",
            font=("Helvetica", 14, "bold"),
            text_color=THEME['text']
        )
        source_label.grid(row=4, column=0, padx=20, pady=(10, 5), sticky="w")
        
        # 데이터 소스 체크박스
        self.source_vars = {
            "text": ctk.BooleanVar(value=True),
            "csv": ctk.BooleanVar(value=True),
            "pdf": ctk.BooleanVar(value=True),
            "web": ctk.BooleanVar(value=True)
        }
        
        source_frame = ctk.CTkFrame(sidebar_frame, fg_color="transparent")
        source_frame.grid(row=5, column=0, padx=20, pady=(0, 10), sticky="ew")
        
        row = 0
        for i, (source, var) in enumerate(self.source_vars.items()):
            source_names = {"text": "텍스트", "csv": "CSV", "pdf": "PDF", "web": "웹/위키"}
            checkbox = ctk.CTkCheckBox(
                source_frame,
                text=source_names.get(source, source),
                variable=var,
                font=("Helvetica", 12),
                checkbox_width=20,
                checkbox_height=20,
                fg_color=THEME['primary'],
                hover_color=THEME['secondary']
            )
            checkbox.grid(row=row, column=0, padx=5, pady=3, sticky="w")
            row += 1
        
        # 구분선
        separator2 = ctk.CTkFrame(sidebar_frame, height=1, fg_color=THEME["secondary"])
        separator2.grid(row=6, column=0, padx=20, pady=(10, 10), sticky="ew")
        
        # 데이터 추가 버튼들
        add_text_button = ctk.CTkButton(
            sidebar_frame,
            text="텍스트 추가",
            command=self.show_text_input,
            height=30,
            font=("Helvetica", 12),
            fg_color=THEME['primary'],
            hover_color=THEME['secondary']
        )
        add_text_button.grid(row=7, column=0, padx=20, pady=(0, 5), sticky="ew")
        
        add_file_button = ctk.CTkButton(
            sidebar_frame,
            text="파일 추가",
            command=self.show_file_options,
            height=30,
            font=("Helvetica", 12),
            fg_color=THEME['primary'],
            hover_color=THEME['secondary']
        )
        add_file_button.grid(row=8, column=0, padx=20, pady=(0, 5), sticky="ew")
        
        add_web_button = ctk.CTkButton(
            sidebar_frame,
            text="웹 페이지 추가",
            command=self.add_web_page,
            height=30,
            font=("Helvetica", 12),
            fg_color=THEME['primary'],
            hover_color=THEME['secondary']
        )
        add_web_button.grid(row=9, column=0, padx=20, pady=(0, 10), sticky="ew")
        
        # 설정 및 도움말 버튼들 (하단)
        bottom_frame = ctk.CTkFrame(sidebar_frame, fg_color="transparent")
        bottom_frame.grid(row=11, column=0, padx=20, pady=(10, 20), sticky="ew")
        
        settings_button = ctk.CTkButton(
            bottom_frame,
            text="설정",
            command=self.show_settings,
            width=70,
            height=30,
            font=("Helvetica", 12),
            fg_color=THEME['primary'],
            hover_color=THEME['secondary']
        )
        settings_button.grid(row=0, column=0, padx=(0, 5), sticky="w")
        
        help_button = ctk.CTkButton(
            bottom_frame,
            text="도움말",
            command=self.show_help,
            width=70,
            height=30,
            font=("Helvetica", 12),
            fg_color=THEME['primary'],
            hover_color=THEME['secondary']
        )
        help_button.grid(row=0, column=1, padx=5, sticky="w")
        
        clear_button = ctk.CTkButton(
            bottom_frame,
            text="대화 초기화",
            command=self.clear_conversation,
            width=100,
            height=30,
            font=("Helvetica", 12),
            fg_color=THEME['error'],
            hover_color="#E74C3C"
        )
        clear_button.grid(row=0, column=2, padx=(5, 0), sticky="e")
    
    def setup_main_content(self):
        """메인 컨텐츠 영역 구성"""
        main_frame = ctk.CTkFrame(self, fg_color=THEME['background'])
        main_frame.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_rowconfigure(1, weight=0)
        
        # 대화 표시 영역
        self.chat_frame = ctk.CTkScrollableFrame(
            main_frame,
            fg_color=THEME['surface'],
            corner_radius=10
        )
        self.chat_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        
        # 초기 웰컴 메시지
        self.add_assistant_message(
            "안녕하세요! 지식 검색 시스템입니다. 질문이나 검색하고 싶은 내용을 입력해주세요."
        )
        
        # 입력 영역
        input_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        input_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 20))
        input_frame.grid_columnconfigure(0, weight=1)
        
        self.input_box = ctk.CTkTextbox(
            input_frame,
            height=80,
            wrap="word",
            font=("Helvetica", 14),
            border_width=1,
            border_color=THEME['primary'],
            fg_color=THEME['surface'],
            corner_radius=10
        )
        self.input_box.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.input_box.bind("<Return>", self.handle_return)
        self.input_box.bind("<Shift-Return>", self.handle_shift_return)
        
        # 입력 버튼들
        buttons_frame = ctk.CTkFrame(input_frame, fg_color="transparent", width=140)
        buttons_frame.grid(row=0, column=1, sticky="ns")
        buttons_frame.grid_rowconfigure(0, weight=1)
        buttons_frame.grid_rowconfigure(1, weight=1)
        
        self.send_button = ctk.CTkButton(
            buttons_frame,
            text="전송",
            command=self.send_message,
            width=120,
            height=35,
            font=("Helvetica", 14, "bold"),
            fg_color=THEME['primary'],
            hover_color=THEME['secondary']
        )
        self.send_button.grid(row=0, column=0, sticky="s", pady=(0, 5))
        
        self.stop_button = ctk.CTkButton(
            buttons_frame,
            text="중지",
            command=self.stop_generation,
            width=120,
            height=35,
            font=("Helvetica", 14, "bold"),
            fg_color=THEME['error'],
            hover_color="#C0392B",
            state="disabled"
        )
        self.stop_button.grid(row=1, column=0, sticky="n", pady=(5, 0))
    
    def add_user_message(self, message):
        """사용자 메시지 추가"""
        container = ctk.CTkFrame(self.chat_frame, fg_color="#D6EAF8", corner_radius=10)
        container.pack(fill="x", padx=10, pady=5, anchor="e")
        
        label = ctk.CTkLabel(
            container,
            text=message,
            font=("Helvetica", 14),
            text_color=THEME['text'],
            wraplength=750,
            justify="left"
        )
        label.pack(padx=15, pady=10, anchor="w")
        
        # 대화 이력 업데이트
        self.conversation.append({"role": "user", "content": message})
        
        # 스크롤 최하단으로
        self.after(100, lambda: self.chat_frame._parent_canvas.yview_moveto(1.0))
    
    def add_assistant_message(self, message, with_markdown=True):
        """어시스턴트 메시지 추가"""
        container = ctk.CTkFrame(self.chat_frame, fg_color=THEME['surface'], corner_radius=10)
        container.pack(fill="x", padx=10, pady=5, anchor="w")
        
        if with_markdown and MARKDOWN_AVAILABLE and TKINTERWEB_AVAILABLE:
            # 마크다운 렌더링
            html_content = markdown.markdown(
                message,
                extensions=['nl2br', 'fenced_code', 'tables']
            )
            
            custom_style = """
            <style>
              body { font-family: sans-serif; line-height: 1.5; padding: 10px; background-color: #FFFFFF; color: #2C3E50; }
              code { background-color: #f5f5f5; padding: 2px 4px; border-radius: 4px; }
              pre { background-color: #f5f5f5; color: #2C3E50; padding: 10px; border-radius: 8px; overflow-x: auto; }
              table { border-collapse: collapse; width: 100%; margin-top: 10px; }
              th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
              blockquote { border-left: 4px solid #ccc; padding-left: 10px; color: #555; margin: 10px 0; }
            </style>
            """
            
            html_content = custom_style + html_content
            
            # 줄 수에 따라 높이 조정
            line_count = message.count('\n') + 1
            calculated_height = min(max(200, line_count * 20), 600)
            
            html_frame = HtmlFrame(container, width=780, height=calculated_height, messages_enabled=False)
            html_frame.load_html(html_content)
            html_frame.pack(padx=15, pady=10, fill="both", expand=True)
        else:
            # 일반 텍스트 렌더링
            label = ctk.CTkLabel(
                container,
                text=message,
                font=("Helvetica", 14),
                text_color=THEME['text'],
                wraplength=750,
                justify="left"
            )
            label.pack(padx=15, pady=10, anchor="w")
        
        # 대화 이력 업데이트
        self.conversation.append({"role": "assistant", "content": message})
        
        # 스크롤 최하단으로
        self.after(100, lambda: self.chat_frame._parent_canvas.yview_moveto(1.0))
    
    def send_message(self):
        """메시지 전송 처리"""
        # 메시지 가져오기
        message = self.input_box.get("0.0", "end").strip()
        
        # 빈 메시지 체크
        if not message or self.is_generating:
            return
        
        # 입력 필드 초기화
        self.input_box.delete("0.0", "end")
        
        # 사용자 메시지 표시
        self.add_user_message(message)
        
        # UI 상태 업데이트
        self.is_generating = True
        self.send_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        
        # 활성화된 데이터 소스 확인
        active_sources = [source for source, var in self.source_vars.items() if var.get()]
        
        # 별도 스레드에서 응답 생성
        threading.Thread(
            target=self._generate_response_thread,
            args=(message, active_sources),
            daemon=True
        ).start()
    
    def _generate_response_thread(self, query, source_types):
        """응답 생성 스레드"""
        try:
            # 응답 생성
            response, graph_data = self.rag_system.answer_query(query, source_types)
            
            # UI 스레드에서 결과 표시
            self.after(100, lambda: self._handle_response_result(response, graph_data))
        except Exception as e:
            # 오류 처리
            error_message = f"응답 생성 중 오류가 발생했습니다: {str(e)}"
            logger.error(error_message)
            traceback.print_exc()
            
            # UI 스레드에서 오류 메시지 표시
            self.after(100, lambda: self._handle_response_error(error_message))
    
    def _handle_response_result(self, response, graph_data):
        """응답 결과 처리"""
        # 응답 표시
        self.add_assistant_message(response)
        
        # 그래프 데이터가 있으면 시각화 (미구현)
        if graph_data:
            pass
        
        # UI 상태 복원
        self.is_generating = False
        self.send_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
    
    def _handle_response_error(self, error_message):
        """응답 오류 처리"""
        # 오류 메시지 표시
        self.add_assistant_message(error_message)
        
        # UI 상태 복원
        self.is_generating = False
        self.send_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
    
    def stop_generation(self):
        """응답 생성 중지"""
        # 실제 중지 기능은 구현되지 않음 (LLM 호출을 직접 중단하기 어려움)
        messagebox.showinfo("알림", "응답 생성을 중지했습니다.")
        
        # UI 상태 복원
        self.is_generating = False
        self.send_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
    
    def handle_return(self, event):
        """Enter 키 이벤트 처리"""
        self.send_message()
        return "break"  # 이벤트 전파 중지
    
    def handle_shift_return(self, event):
        """Shift+Enter 키 이벤트 처리"""
        # 기본 동작 유지 (줄바꿈)
        return None
    
    def show_text_input(self):
        """텍스트 입력 대화상자 표시"""
        text_dialog = ctk.CTkToplevel(self)
        text_dialog.title("텍스트 추가")
        text_dialog.geometry("700x500")
        text_dialog.transient(self)
        text_dialog.grab_set()
        
        # 중앙에 배치
        text_dialog.update_idletasks()
        width = text_dialog.winfo_width()
        height = text_dialog.winfo_height()
        x = (text_dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (text_dialog.winfo_screenheight() // 2) - (height // 2)
        text_dialog.geometry(f'+{x}+{y}')
        
        # 제목 입력
        title_frame = ctk.CTkFrame(text_dialog, fg_color="transparent")
        title_frame.pack(fill="x", padx=20, pady=(20, 10))
        
        ctk.CTkLabel(
            title_frame,
            text="제목:",
            font=("Helvetica", 14),
            width=50
        ).pack(side="left")
        
        title_entry = ctk.CTkEntry(
            title_frame,
            font=("Helvetica", 14),
            width=500,
            placeholder_text="텍스트 문서의 제목을 입력하세요"
        )
        title_entry.pack(side="left", padx=(10, 0), fill="x", expand=True)
        
        # 텍스트 입력
        ctk.CTkLabel(
            text_dialog,
            text="내용:",
            font=("Helvetica", 14),
            anchor="w"
        ).pack(fill="x", padx=20, pady=(10, 5))
        
        text_box = ctk.CTkTextbox(
            text_dialog,
            font=("Helvetica", 14),
            wrap="word"
        )
        text_box.pack(fill="both", expand=True, padx=20, pady=(0, 10))
        
        # 버튼 영역
        button_frame = ctk.CTkFrame(text_dialog, fg_color="transparent")
        button_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        ctk.CTkButton(
            button_frame,
            text="취소",
            font=("Helvetica", 14),
            width=100,
            command=text_dialog.destroy,
            fg_color=THEME['error'],
            hover_color="#C0392B"
        ).pack(side="left")
        
        ctk.CTkButton(
            button_frame,
            text="추가",
            font=("Helvetica", 14),
            width=100,
            command=lambda: self._add_text_document(title_entry.get(), text_box.get("0.0", "end"), text_dialog),
            fg_color=THEME['primary'],
            hover_color=THEME['secondary']
        ).pack(side="right")
    
    def _add_text_document(self, title, content, dialog):
        """텍스트 문서 추가"""
        if not title.strip():
            messagebox.showwarning("입력 오류", "제목을 입력해주세요.", parent=dialog)
            return
        
        if not content.strip():
            messagebox.showwarning("입력 오류", "내용을 입력해주세요.", parent=dialog)
            return
        
        # 로딩 대화상자
        loading_dialog = self.create_loading_dialog("텍스트 처리 중...")
        
        # 별도 스레드에서 처리
        threading.Thread(
            target=self._process_text_thread,
            args=(title, content, dialog, loading_dialog),
            daemon=True
        ).start()
    
    def _process_text_thread(self, title, content, dialog, loading_dialog):
        """텍스트 처리 스레드"""
        try:
            # 텍스트 파일 저장
            safe_title = re.sub(r'[^\w\s가-힣]', '', title).strip()  # 특수문자 제거
            safe_title = re.sub(r'\s+', '_', safe_title)  # 공백을 언더스코어로 변환
            if not safe_title:  # 제목이 비어있을 경우
                safe_title = 'untitled'
            file_name = f"{int(time.time())}_{safe_title}.txt"
            file_path = os.path.join("data/text", file_name)
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            # RAG 시스템에 추가
            doc_count = self.rag_system.add_text_document(content, title)
            
            # UI 스레드에서 결과 처리
            self.after(100, lambda: self._handle_text_add_result(title, doc_count, dialog, loading_dialog))
        except Exception as e:
            # UI 스레드에서 오류 처리
            # 수정된 부분:
            error_msg = str(e)
            def handle_error():
                self._handle_pdf_file_error(error_msg, loading_dialog)
            self.after(100, handle_error)
    
    def _handle_text_add_result(self, title, doc_count, dialog, loading_dialog):
        """텍스트 추가 결과 처리"""
        loading_dialog.destroy()
        messagebox.showinfo("성공", f"'{title}' 문서가 성공적으로 추가되었습니다.\n{doc_count}개의 청크로 분할되었습니다.", parent=dialog)
        dialog.destroy()
    
    def _handle_text_add_error(self, error_message, dialog, loading_dialog):
        """텍스트 추가 오류 처리"""
        loading_dialog.destroy()
        messagebox.showerror("오류", f"텍스트 처리 중 오류가 발생했습니다:\n{error_message}", parent=dialog)
    
    def show_file_options(self):
        """파일 추가 옵션 메뉴 표시"""
        file_menu = ctk.CTkToplevel(self)
        file_menu.title("파일 추가")
        file_menu.geometry("300x200")
        file_menu.transient(self)
        file_menu.grab_set()
        
        # 중앙에 배치
        file_menu.update_idletasks()
        width = file_menu.winfo_width()
        height = file_menu.winfo_height()
        x = (file_menu.winfo_screenwidth() // 2) - (width // 2)
        y = (file_menu.winfo_screenheight() // 2) - (height // 2)
        file_menu.geometry(f'+{x}+{y}')
        
        # 제목
        ctk.CTkLabel(
            file_menu,
            text="추가할 파일 유형을 선택하세요",
            font=("Helvetica", 16, "bold")
        ).pack(pady=(20, 30))
        
        # 버튼들
        ctk.CTkButton(
            file_menu,
            text="텍스트 파일 (.txt)",
            font=("Helvetica", 14),
            command=lambda: [file_menu.destroy(), self.add_text_file()],
            width=200,
            height=40,
            fg_color=THEME['primary'],
            hover_color=THEME['secondary']
        ).pack(pady=(0, 10))
        
        ctk.CTkButton(
            file_menu,
            text="CSV 파일 (.csv)",
            font=("Helvetica", 14),
            command=lambda: [file_menu.destroy(), self.add_csv_file()],
            width=200,
            height=40,
            fg_color=THEME['primary'],
            hover_color=THEME['secondary']
        ).pack(pady=(0, 10))
        
        ctk.CTkButton(
            file_menu,
            text="PDF 파일 (.pdf)",
            font=("Helvetica", 14),
            command=lambda: [file_menu.destroy(), self.add_pdf_file()],
            width=200,
            height=40,
            fg_color=THEME['primary'],
            hover_color=THEME['secondary']
        ).pack(pady=(0, 20))
    
    def add_text_file(self):
        """텍스트 파일 추가"""
        file_path = filedialog.askopenfilename(
            title="텍스트 파일 선택",
            filetypes=[("텍스트 파일", "*.txt"), ("모든 파일", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            # 파일 읽기
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # 파일명을 제목으로 사용
            title = os.path.basename(file_path)
            
            # 로딩 대화상자
            loading_dialog = self.create_loading_dialog("텍스트 파일 처리 중...")
            
            # 별도 스레드에서 처리
            threading.Thread(
                target=self._process_text_file_thread,
                args=(title, content, file_path, loading_dialog),
                daemon=True
            ).start()
        except Exception as e:
            messagebox.showerror("오류", f"파일 읽기 오류: {str(e)}")
    
    def _process_text_file_thread(self, title, content, file_path, loading_dialog):
        """텍스트 파일 처리 스레드"""
        try:
            # 파일 복사
            dest_path = os.path.join("data/text", os.path.basename(file_path))
            shutil.copy2(file_path, dest_path)
            
            # RAG 시스템에 추가
            doc_count = self.rag_system.add_text_document(content, title)
            
            # UI 스레드에서 결과 처리
            self.after(100, lambda: self._handle_text_file_result(title, doc_count, loading_dialog))
        except Exception as e:
            # UI 스레드에서 오류 처리
            # 수정된 부분:
            error_msg = str(e)
            def handle_error():
                self._handle_pdf_file_error(error_msg, loading_dialog)
            self.after(100, handle_error)
    
    def _handle_text_file_result(self, title, doc_count, loading_dialog):
        """텍스트 파일 추가 결과 처리"""
        loading_dialog.destroy()
        messagebox.showinfo("성공", f"'{title}' 파일이 성공적으로 추가되었습니다.\n{doc_count}개의 청크로 분할되었습니다.")
    
    def _handle_text_file_error(self, error_message, loading_dialog):
        """텍스트 파일 추가 오류 처리"""
        loading_dialog.destroy()
        messagebox.showerror("오류", f"텍스트 파일 처리 중 오류가 발생했습니다:\n{error_message}")
    
    def add_csv_file(self):
        """CSV 파일 추가"""
        file_path = filedialog.askopenfilename(
            title="CSV 파일 선택",
            filetypes=[("CSV 파일", "*.csv"), ("모든 파일", "*.*")]
        )
        
        if not file_path:
            return
        
        # 로딩 대화상자
        loading_dialog = self.create_loading_dialog("CSV 파일 처리 중...")
        
        # 별도 스레드에서 처리
        threading.Thread(
            target=self._process_csv_file_thread,
            args=(file_path, loading_dialog),
            daemon=True
        ).start()
    
    def _process_csv_file_thread(self, file_path, loading_dialog):
        """CSV 파일 처리 스레드"""
        try:
            # CSV 파일 읽기 (다양한 인코딩 시도)
            encodings = ['utf-8', 'cp949', 'euc-kr', 'latin1']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except Exception:
                    continue
            
            if df is None:
                raise ValueError("CSV 파일을 읽을 수 없습니다. 인코딩을 확인해주세요.")
            
            # 파일 복사
            dest_path = os.path.join("data/csv", os.path.basename(file_path))
            shutil.copy2(file_path, dest_path)
            
            # 파일명을 제목으로 사용
            title = os.path.basename(file_path)
            
            # RAG 시스템에 추가
            doc_count = self.rag_system.add_csv_document(df, title)
            
            # UI 스레드에서 결과 처리
            self.after(100, lambda: self._handle_csv_file_result(title, doc_count, loading_dialog))
        except Exception as e:
            # UI 스레드에서 오류 처리
            # 수정된 부분:
            error_msg = str(e)
            def handle_error():
                self._handle_pdf_file_error(error_msg, loading_dialog)
            self.after(100, handle_error)
    
    def _handle_csv_file_result(self, title, doc_count, loading_dialog):
        """CSV 파일 추가 결과 처리"""
        loading_dialog.destroy()
        messagebox.showinfo("성공", f"'{title}' CSV 파일이 성공적으로 추가되었습니다.\n{doc_count}개의 행이 처리되었습니다.")
    
    def _handle_csv_file_error(self, error_message, loading_dialog):
        """CSV 파일 추가 오류 처리"""
        loading_dialog.destroy()
        messagebox.showerror("오류", f"CSV 파일 처리 중 오류가 발생했습니다:\n{error_message}")
    
    def add_pdf_file(self):
        """PDF 파일 추가"""
        if 'fitz' not in sys.modules:
            messagebox.showwarning("기능 제한", "PDF 처리 기능을 사용하려면 PyMuPDF(fitz) 패키지가 설치되어 있어야 합니다.")
            return
        
        file_path = filedialog.askopenfilename(
            title="PDF 파일 선택",
            filetypes=[("PDF 파일", "*.pdf"), ("모든 파일", "*.*")]
        )
        
        if not file_path:
            return
        
        # 로딩 대화상자
        loading_dialog = self.create_loading_dialog("PDF 파일 처리 중...")
        
        # 별도 스레드에서 처리
        threading.Thread(
            target=self._process_pdf_file_thread,
            args=(file_path, loading_dialog),
            daemon=True
        ).start()
    
    def _process_pdf_file_thread(self, file_path, loading_dialog):
        """PDF 파일 처리 스레드"""
        try:
            # 파일 복사
            dest_path = os.path.join("data/pdf", os.path.basename(file_path))
            shutil.copy2(file_path, dest_path)
            
            # RAG 시스템에 추가
            doc_count = self.rag_system.add_pdf_document(file_path)
            
            # UI 스레드에서 결과 처리
            title = os.path.basename(file_path)
            self.after(100, lambda: self._handle_pdf_file_result(title, doc_count, loading_dialog))
        except Exception as e:
            # UI 스레드에서 오류 처리
            # 수정된 부분:
            error_msg = str(e)
            def handle_error():
                self._handle_pdf_file_error(error_msg, loading_dialog)
            self.after(100, handle_error)
    
    def _handle_pdf_file_result(self, title, doc_count, loading_dialog):
        """PDF 파일 추가 결과 처리"""
        loading_dialog.destroy()
        messagebox.showinfo("성공", f"'{title}' PDF 파일이 성공적으로 추가되었습니다.\n{doc_count}개의 청크로 분할되었습니다.")
    
    def _handle_pdf_file_error(self, error_message, loading_dialog):
        """PDF 파일 추가 오류 처리"""
        loading_dialog.destroy()
        messagebox.showerror("오류", f"PDF 파일 처리 중 오류가 발생했습니다:\n{error_message}")
    
    def add_web_page(self):
        """웹 페이지 추가"""
        if not WEB_SUPPORT:
            messagebox.showwarning("기능 제한", "웹 페이지 처리 기능을 사용하려면 BeautifulSoup와 Requests 패키지가 설치되어 있어야 합니다.")
            return
        
        # URL 입력 대화상자
        url = simpledialog.askstring("웹 페이지 추가", "웹 페이지 URL을 입력하세요:")
        
        if not url:
            return
        
        # URL 검증 (간단히)
        if not (url.startswith('http://') or url.startswith('https://')):
            url = 'https://' + url
        
        # 로딩 대화상자
        loading_dialog = self.create_loading_dialog("웹 페이지 처리 중...")
        
        # 별도 스레드에서 처리
        threading.Thread(
            target=self._process_web_page_thread,
            args=(url, loading_dialog),
            daemon=True
        ).start()
    
    def _process_web_page_thread(self, url, loading_dialog):
        """웹 페이지 처리 스레드"""
        try:
            # URL 형식 검증
            if not (url.startswith('http://') or url.startswith('https://')):
                raise ValueError("URL은 http:// 또는 https://로 시작해야 합니다.")
            
            # RAG 시스템에 추가
            doc_count = self.rag_system.add_web_document(url)
            
            # UI 스레드에서 결과 처리
            self.after(100, lambda: self._handle_web_page_result(url, doc_count, loading_dialog))
        except Exception as e:
            # UI 스레드에서 오류 처리
            # 수정된 부분:
            error_msg = str(e)
            def handle_error():
                self._handle_pdf_file_error(error_msg, loading_dialog)
            self.after(100, handle_error)
    
    def _handle_web_page_result(self, url, doc_count, loading_dialog):
        """웹 페이지 추가 결과 처리"""
        loading_dialog.destroy()
        messagebox.showinfo("성공", f"'{url}' 웹 페이지가 성공적으로 추가되었습니다.\n{doc_count}개의 청크로 분할되었습니다.")
    
    def _handle_web_page_error(self, error_message, loading_dialog):
        """웹 페이지 추가 오류 처리"""
        loading_dialog.destroy()
        messagebox.showerror("오류", f"웹 페이지 처리 중 오류가 발생했습니다:\n{error_message}")
    
    def save_conversation(self):
        """대화 저장"""
        if not self.conversation:
            messagebox.showinfo("알림", "저장할 대화 내용이 없습니다.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="대화 저장",
            defaultextension=".json",
            filetypes=[("JSON 파일", "*.json"), ("텍스트 파일", "*.txt"), ("모든 파일", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            # 저장 형식 결정 (확장자 기반)
            _, ext = os.path.splitext(file_path)
            
            if ext.lower() == '.json':
                # JSON 형식 저장
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.conversation, f, ensure_ascii=False, indent=2)
            else:
                # 텍스트 형식 저장
                with open(file_path, 'w', encoding='utf-8') as f:
                    for message in self.conversation:
                        role = "사용자" if message["role"] == "user" else "AI"
                        f.write(f"{role}: {message['content']}\n\n")
            
            messagebox.showinfo("성공", "대화 내용이 저장되었습니다.")
        except Exception as e:
            messagebox.showerror("오류", f"대화 저장 중 오류가 발생했습니다:\n{str(e)}")
    
    def load_conversation(self):
        """대화 불러오기"""
        file_path = filedialog.askopenfilename(
            title="대화 불러오기",
            filetypes=[("JSON 파일", "*.json"), ("텍스트 파일", "*.txt"), ("모든 파일", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            # 파일 형식 결정 (확장자 기반)
            _, ext = os.path.splitext(file_path)
            
            if ext.lower() == '.json':
                # JSON 형식 로드
                with open(file_path, 'r', encoding='utf-8') as f:
                    conversation = json.load(f)
                
                # 형식 검증
                if not all(isinstance(msg, dict) and 'role' in msg and 'content' in msg for msg in conversation):
                    raise ValueError("유효하지 않은 대화 형식입니다.")
                
                # 기존 대화 초기화
                self.clear_conversation(show_message=False)
                
                # 새 대화 추가
                for message in conversation:
                    role = message["role"]
                    content = message["content"]
                    
                    if role == "user":
                        self.add_user_message(content)
                    elif role == "assistant":
                        self.add_assistant_message(content)
            else:
                # 텍스트 형식 로드 (간단한 처리)
                messagebox.showwarning("제한된 기능", "텍스트 형식의 대화 불러오기는 제한적입니다. JSON 형식을 권장합니다.")
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # 기존 대화 초기화
                self.clear_conversation(show_message=False)
                
                # 메시지 분리 시도
                messages = text.split("\n\n")
                for message in messages:
                    if not message.strip():
                        continue
                    
                    # 역할과 내용 분리 시도
                    parts = message.split(":", 1)
                    if len(parts) == 2:
                        role, content = parts[0].strip(), parts[1].strip()
                        
                        if role in ["사용자", "User", "user"]:
                            self.add_user_message(content)
                        else:
                            self.add_assistant_message(content)
            
            messagebox.showinfo("성공", "대화 내용을 불러왔습니다.")
        except Exception as e:
            messagebox.showerror("오류", f"대화 로드 중 오류가 발생했습니다:\n{str(e)}")
    
    def clear_conversation(self, show_message=True):
        """대화 초기화"""
        if not self.conversation:
            return
        
        if show_message:
            result = messagebox.askyesno("확인", "정말 대화 내용을 모두 지우시겠습니까?")
            if not result:
                return
        
        # 대화 이력 초기화
        self.conversation = []
        
        # 대화 화면 초기화
        for widget in self.chat_frame.winfo_children():
            widget.destroy()
        
        # 초기 메시지 표시
        self.add_assistant_message(
            "안녕하세요! 지식 검색 시스템입니다. 질문이나 검색하고 싶은 내용을 입력해주세요."
        )
    
    def copy_selection(self):
        """선택된 텍스트 복사"""
        try:
            self.clipboard_clear()
            
            # 입력 상자에 포커스가 있으면 선택된 텍스트 복사
            if self.focus_get() == self.input_box:
                selected_text = self.input_box.selection_get()
                self.clipboard_append(selected_text)
            else:
                # 대화 내용에서 선택된 텍스트 복사하기는 어려움 (구현 제한)
                pass
        except Exception:
            # 선택된 텍스트가 없는 경우 등의 예외 무시
            pass
    
    def paste_clipboard(self):
        """클립보드 내용 붙여넣기"""
        try:
            # 입력 상자에 포커스가 있을 때만 붙여넣기
            if self.focus_get() == self.input_box:
                clipboard_text = self.clipboard_get()
                
                # 현재 커서 위치에 붙여넣기
                self.input_box.insert("insert", clipboard_text)
        except Exception:
            # 클립보드가 비어있는 경우 등의 예외 무시
            pass
    
    def show_settings(self):
        """설정 대화상자 표시"""
        settings_dialog = ctk.CTkToplevel(self)
        settings_dialog.title("설정")
        settings_dialog.geometry("500x600")
        settings_dialog.transient(self)
        settings_dialog.grab_set()
        
        # 중앙에 배치
        settings_dialog.update_idletasks()
        width = settings_dialog.winfo_width()
        height = settings_dialog.winfo_height()
        x = (settings_dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (settings_dialog.winfo_screenheight() // 2) - (height // 2)
        settings_dialog.geometry(f'+{x}+{y}')
        
        # 설정 내용을 포함할 스크롤 프레임
        settings_frame = ctk.CTkScrollableFrame(settings_dialog)
        settings_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # 모델 설정
        model_frame = ctk.CTkFrame(settings_frame)
        model_frame.pack(fill="x", pady=(0, 15))
        
        ctk.CTkLabel(
            model_frame,
            text="모델 설정",
            font=("Helvetica", 16, "bold")
        ).pack(anchor="w", pady=(5, 10))
        
        # 모델 경로
        model_path_frame = ctk.CTkFrame(model_frame, fg_color="transparent")
        model_path_frame.pack(fill="x", pady=5)
        
        ctk.CTkLabel(
            model_path_frame,
            text="모델 경로:",
            width=100
        ).pack(side="left")
        
        model_path_var = ctk.StringVar(value=self.config_manager.get("model_path", ""))
        model_path_entry = ctk.CTkEntry(
            model_path_frame,
            textvariable=model_path_var,
            width=250
        )
        model_path_entry.pack(side="left", padx=(10, 10), fill="x", expand=True)
        
        ctk.CTkButton(
            model_path_frame,
            text="찾기",
            width=60,
            command=lambda: self._browse_model_path(model_path_var)
        ).pack(side="right")
        
        # 컨텍스트 크기
        context_frame = ctk.CTkFrame(model_frame, fg_color="transparent")
        context_frame.pack(fill="x", pady=5)
        
        ctk.CTkLabel(
            context_frame,
            text="컨텍스트 크기:",
            width=100
        ).pack(side="left")
        
        context_var = ctk.StringVar(value=str(self.config_manager.get("context_size", 2048)))
        ctk.CTkEntry(
            context_frame,
            textvariable=context_var,
            width=100
        ).pack(side="left", padx=(10, 0))
        
        # 온도
        temp_frame = ctk.CTkFrame(model_frame, fg_color="transparent")
        temp_frame.pack(fill="x", pady=5)
        
        ctk.CTkLabel(
            temp_frame,
            text="온도(0.1-1.0):",
            width=100
        ).pack(side="left")
        
        temp_var = ctk.StringVar(value=str(self.config_manager.get("temperature", 0.1)))
        ctk.CTkEntry(
            temp_frame,
            textvariable=temp_var,
            width=100
        ).pack(side="left", padx=(10, 0))
        
        # 최대 토큰 수
        token_frame = ctk.CTkFrame(model_frame, fg_color="transparent")
        token_frame.pack(fill="x", pady=5)
        
        ctk.CTkLabel(
            token_frame,
            text="최대 토큰 수:",
            width=100
        ).pack(side="left")
        
        token_var = ctk.StringVar(value=str(self.config_manager.get("max_tokens", 1000)))
        ctk.CTkEntry(
            token_frame,
            textvariable=token_var,
            width=100
        ).pack(side="left", padx=(10, 0))
        
        # 임베딩 모델
        embed_frame = ctk.CTkFrame(model_frame, fg_color="transparent")
        embed_frame.pack(fill="x", pady=5)
        
        ctk.CTkLabel(
            embed_frame,
            text="임베딩 모델:",
            width=100
        ).pack(side="left")
        
        embed_var = ctk.StringVar(value=self.config_manager.get("embedding_model", "jhgan/ko-sroberta-multitask"))
        ctk.CTkEntry(
            embed_frame,
            textvariable=embed_var,
            width=250
        ).pack(side="left", padx=(10, 0), fill="x", expand=True)
        
        # 일반 설정
        general_frame = ctk.CTkFrame(settings_frame)
        general_frame.pack(fill="x", pady=(0, 15))
        
        ctk.CTkLabel(
            general_frame,
            text="일반 설정",
            font=("Helvetica", 16, "bold")
        ).pack(anchor="w", pady=(5, 10))
        
        # 테마 설정
        theme_frame = ctk.CTkFrame(general_frame, fg_color="transparent")
        theme_frame.pack(fill="x", pady=5)
        
        ctk.CTkLabel(
            theme_frame,
            text="테마:",
            width=100
        ).pack(side="left")
        
        theme_var = ctk.StringVar(value=self.config_manager.get("theme", "system"))
        theme_combo = ctk.CTkComboBox(
            theme_frame,
            values=["system", "light", "dark"],
            variable=theme_var,
            width=150
        )
        theme_combo.pack(side="left", padx=(10, 0))
        
        # 언어 설정
        lang_frame = ctk.CTkFrame(general_frame, fg_color="transparent")
        lang_frame.pack(fill="x", pady=5)
        
        ctk.CTkLabel(
            lang_frame,
            text="언어:",
            width=100
        ).pack(side="left")
        
        lang_var = ctk.StringVar(value=self.config_manager.get("language", "ko"))
        lang_combo = ctk.CTkComboBox(
            lang_frame,
            values=["ko", "en", "ja", "zh"],
            variable=lang_var,
            width=150
        )
        lang_combo.pack(side="left", padx=(10, 0))
        
        # 자동 저장
        auto_save_var = ctk.BooleanVar(value=self.config_manager.get("auto_save", True))
        auto_save_check = ctk.CTkCheckBox(
            general_frame,
            text="설정 자동 저장",
            variable=auto_save_var
        )
        auto_save_check.pack(anchor="w", pady=5)
        
        # 검색 설정
        search_frame = ctk.CTkFrame(settings_frame)
        search_frame.pack(fill="x", pady=(0, 15))
        
        ctk.CTkLabel(
            search_frame,
            text="검색 설정",
            font=("Helvetica", 16, "bold")
        ).pack(anchor="w", pady=(5, 10))
        
        # 위키 검색
        wiki_search_var = ctk.BooleanVar(value=self.config_manager.get("wiki_search_enabled", True))
        wiki_search_check = ctk.CTkCheckBox(
            search_frame,
            text="위키피디아 검색 활성화",
            variable=wiki_search_var
        )
        wiki_search_check.pack(anchor="w", pady=5)
        
        # 웹 검색
        web_search_var = ctk.BooleanVar(value=self.config_manager.get("web_search_enabled", True))
        web_search_check = ctk.CTkCheckBox(
            search_frame,
            text="웹 검색 활성화",
            variable=web_search_var
        )
        web_search_check.pack(anchor="w", pady=5)
        
        # 캐시 설정
        cache_frame = ctk.CTkFrame(settings_frame)
        cache_frame.pack(fill="x", pady=(0, 15))
        
        ctk.CTkLabel(
            cache_frame,
            text="캐시 설정",
            font=("Helvetica", 16, "bold")
        ).pack(anchor="w", pady=(5, 10))
        
        # 캐시 활성화
        cache_enabled_var = ctk.BooleanVar(value=self.config_manager.get("cache_enabled", True))
        cache_enabled_check = ctk.CTkCheckBox(
            cache_frame,
            text="응답 캐싱 활성화",
            variable=cache_enabled_var
        )
        cache_enabled_check.pack(anchor="w", pady=5)
        
        # 캐시 크기
        cache_size_frame = ctk.CTkFrame(cache_frame, fg_color="transparent")
        cache_size_frame.pack(fill="x", pady=5)
        
        ctk.CTkLabel(
            cache_size_frame,
            text="캐시 크기:",
            width=100
        ).pack(side="left")
        
        cache_size_var = ctk.StringVar(value=str(self.config_manager.get("cache_size", 100)))
        ctk.CTkEntry(
            cache_size_frame,
            textvariable=cache_size_var,
            width=100
        ).pack(side="left", padx=(10, 0))
        
        # 버튼 영역
        button_frame = ctk.CTkFrame(settings_dialog, fg_color="transparent", height=50)
        button_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        ctk.CTkButton(
            button_frame,
            text="취소",
            command=settings_dialog.destroy,
            width=100,
            fg_color=THEME['error']
        ).pack(side="left")
        
        ctk.CTkButton(
            button_frame,
            text="저장",
            command=lambda: self._save_settings(
                settings_dialog,
                model_path_var.get(),
                context_var.get(),
                temp_var.get(),
                token_var.get(),
                embed_var.get(),
                theme_var.get(),
                lang_var.get(),
                auto_save_var.get(),
                wiki_search_var.get(),
                web_search_var.get(),
                cache_enabled_var.get(),
                cache_size_var.get()
            ),
            width=100,
            fg_color=THEME['primary']
        ).pack(side="right")
    
    def _browse_model_path(self, path_var):
        """모델 파일 경로 선택"""
        model_file = filedialog.askopenfilename(
            title="LLM 모델 파일 선택",
            filetypes=[("GGUF 파일", "*.gguf"), ("GGML 파일", "*.ggml"), ("bin 파일", "*.bin"), ("모든 파일", "*.*")],
            initialdir="models"
        )
        
        if model_file:
            path_var.set(model_file)
    
    def _save_settings(self, dialog, model_path, context_size, temperature, max_tokens, 
                       embedding_model, theme, language, auto_save, wiki_search, 
                       web_search, cache_enabled, cache_size):
        """설정 저장"""
        try:
            # 숫자 필드 검증
            try:
                context_size = int(context_size)
                if context_size < 512 or context_size > 8192:
                    raise ValueError("컨텍스트 크기는 512-8192 사이여야 합니다.")
            except ValueError:
                messagebox.showerror("입력 오류", "컨텍스트 크기는 유효한 숫자여야 합니다.", parent=dialog)
                return
            
            try:
                temperature = float(temperature)
                if temperature < 0.0 or temperature > 1.0:
                    raise ValueError("온도는 0.0-1.0 사이여야 합니다.")
            except ValueError:
                messagebox.showerror("입력 오류", "온도는 0.0-1.0 사이의 숫자여야 합니다.", parent=dialog)
                return
            
            try:
                max_tokens = int(max_tokens)
                if max_tokens < 10 or max_tokens > 4096:
                    raise ValueError("최대 토큰 수는 10-4096 사이여야 합니다.")
            except ValueError:
                messagebox.showerror("입력 오류", "최대 토큰 수는 유효한 숫자여야 합니다.", parent=dialog)
                return
            
            try:
                cache_size = int(cache_size)
                if cache_size < 10 or cache_size > 1000:
                    raise ValueError("캐시 크기는 10-1000 사이여야 합니다.")
            except ValueError:
                messagebox.showerror("입력 오류", "캐시 크기는 유효한 숫자여야 합니다.", parent=dialog)
                return
            
            # 설정 업데이트
            self.config_manager.set("model_path", model_path)
            self.config_manager.set("context_size", context_size)
            self.config_manager.set("temperature", temperature)
            self.config_manager.set("max_tokens", max_tokens)
            self.config_manager.set("embedding_model", embedding_model)
            self.config_manager.set("theme", theme)
            self.config_manager.set("language", language)
            self.config_manager.set("auto_save", auto_save)
            self.config_manager.set("wiki_search_enabled", wiki_search)
            self.config_manager.set("web_search_enabled", web_search)
            self.config_manager.set("cache_enabled", cache_enabled)
            self.config_manager.set("cache_size", cache_size)
            
            # 설정 저장
            self.config_manager.save_config()
            
            # 테마 변경 적용
            if theme != ctk.get_appearance_mode().lower():
                self.change_theme(theme)
            
            # 캐시 크기 업데이트
            if self.rag_system.cache.max_size != cache_size:
                self.rag_system.cache.max_size = cache_size
            
            # 대화상자 닫기
            dialog.destroy()
            
            messagebox.showinfo("성공", "설정이 저장되었습니다.")
        except Exception as e:
            messagebox.showerror("오류", f"설정 저장 중 오류가 발생했습니다:\n{str(e)}", parent=dialog)
    
    def change_theme(self, theme_mode):
        """테마 변경"""
        ctk.set_appearance_mode(theme_mode)
        self.config_manager.set("theme", theme_mode)
    
    def manage_cache(self):
        """캐시 관리 대화상자"""
        cache_dialog = ctk.CTkToplevel(self)
        cache_dialog.title("캐시 관리")
        cache_dialog.geometry("400x300")
        cache_dialog.transient(self)
        cache_dialog.grab_set()
        
        # 중앙에 배치
        cache_dialog.update_idletasks()
        width = cache_dialog.winfo_width()
        height = cache_dialog.winfo_height()
        x = (cache_dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (cache_dialog.winfo_screenheight() // 2) - (height // 2)
        cache_dialog.geometry(f'+{x}+{y}')
        
        # 캐시 정보 표시
        cache = self.rag_system.cache
        
        info_frame = ctk.CTkFrame(cache_dialog)
        info_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        ctk.CTkLabel(
            info_frame,
            text="캐시 정보",
            font=("Helvetica", 16, "bold")
        ).pack(anchor="w", pady=(0, 10))
        
        ctk.CTkLabel(
            info_frame,
            text=f"캐시된 항목 수: {len(cache.cache)}",
            font=("Helvetica", 14)
        ).pack(anchor="w", pady=5)
        
        ctk.CTkLabel(
            info_frame,
            text=f"최대 캐시 크기: {cache.max_size}",
            font=("Helvetica", 14)
        ).pack(anchor="w", pady=5)
        
        ctk.CTkLabel(
            info_frame,
            text=f"캐시 히트: {cache.hits}",
            font=("Helvetica", 14)
        ).pack(anchor="w", pady=5)
        
        ctk.CTkLabel(
            info_frame,
            text=f"캐시 미스: {cache.misses}",
            font=("Helvetica", 14)
        ).pack(anchor="w", pady=5)
        
        if cache.hits + cache.misses > 0:
            hit_rate = cache.hits / (cache.hits + cache.misses) * 100
            ctk.CTkLabel(
                info_frame,
                text=f"히트율: {hit_rate:.1f}%",
                font=("Helvetica", 14)
            ).pack(anchor="w", pady=5)
        
        # 버튼 영역
        button_frame = ctk.CTkFrame(cache_dialog, fg_color="transparent")
        button_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        ctk.CTkButton(
            button_frame,
            text="닫기",
            command=cache_dialog.destroy,
            width=100
        ).pack(side="left")
        
        ctk.CTkButton(
            button_frame,
            text="캐시 비우기",
            command=lambda: self._clear_cache(cache_dialog),
            width=100,
            fg_color=THEME['error']
        ).pack(side="right")
    
    def _clear_cache(self, dialog=None):
        """캐시 비우기"""
        result = messagebox.askyesno(
            "확인",
            "정말 캐시를 모두 비우시겠습니까?",
            parent=dialog
        )
        
        if result:
            self.rag_system.cache.clear()
            
            if dialog:
                dialog.destroy()
                messagebox.showinfo("성공", "캐시가 성공적으로 비워졌습니다.")
            else:
                messagebox.showinfo("성공", "캐시가 성공적으로 비워졌습니다.")
    
    def manage_vector_stores(self):
        """벡터 저장소 관리 대화상자"""
        vs_dialog = ctk.CTkToplevel(self)
        vs_dialog.title("벡터 저장소 관리")
        vs_dialog.geometry("500x400")
        vs_dialog.transient(self)
        vs_dialog.grab_set()
        
        # 중앙에 배치
        vs_dialog.update_idletasks()
        width = vs_dialog.winfo_width()
        height = vs_dialog.winfo_height()
        x = (vs_dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (vs_dialog.winfo_screenheight() // 2) - (height // 2)
        vs_dialog.geometry(f'+{x}+{y}')
        
        # 벡터 저장소 정보 표시
        info_frame = ctk.CTkScrollableFrame(vs_dialog)
        info_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        ctk.CTkLabel(
            info_frame,
            text="벡터 저장소 정보",
            font=("Helvetica", 16, "bold")
        ).pack(anchor="w", pady=(0, 10))
        
        # 각 저장소 유형별 정보
        for source_type, vector_store in self.rag_system.vector_stores.items():
            store_frame = ctk.CTkFrame(info_frame)
            store_frame.pack(fill="x", pady=5)
            
            store_count = 0
            if vector_store:
                try:
                    store_count = len(vector_store.index_to_docstore_id)
                except:
                    store_count = "알 수 없음"
            
            source_names = {
                "text": "텍스트",
                "csv": "CSV",
                "pdf": "PDF",
                "web": "웹/위키"
            }
            
            source_name = source_names.get(source_type, source_type)
            
            ctk.CTkLabel(
                store_frame,
                text=f"{source_name} 저장소:",
                font=("Helvetica", 14, "bold"),
                width=100
            ).pack(side="left", padx=10, pady=5)
            
            ctk.CTkLabel(
                store_frame,
                text=f"문서 수: {store_count}",
                font=("Helvetica", 14)
            ).pack(side="left", padx=10, pady=5)
            
            if vector_store:
                ctk.CTkButton(
                    store_frame,
                    text="초기화",
                    width=80,
                    command=lambda t=source_type: self._clear_vector_store(t, vs_dialog),
                    fg_color=THEME['error']
                ).pack(side="right", padx=10, pady=5)
        
        # 버튼 영역
        button_frame = ctk.CTkFrame(vs_dialog, fg_color="transparent")
        button_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        ctk.CTkButton(
            button_frame,
            text="닫기",
            command=vs_dialog.destroy,
            width=100
        ).pack(side="left")
        
        ctk.CTkButton(
            button_frame,
            text="모두 초기화",
            command=lambda: self._clear_all_vector_stores(vs_dialog),
            width=120,
            fg_color=THEME['error']
        ).pack(side="right")
    
    def _clear_vector_store(self, store_type, dialog=None):
        """특정 벡터 저장소 초기화"""
        source_names = {
            "text": "텍스트",
            "csv": "CSV",
            "pdf": "PDF",
            "web": "웹/위키"
        }
        
        source_name = source_names.get(store_type, store_type)
        
        result = messagebox.askyesno(
            "확인",
            f"정말 {source_name} 벡터 저장소를 초기화하시겠습니까?\n이 작업은 되돌릴 수 없습니다.",
            parent=dialog
        )
        
        if result:
            vector_path = f"vectors/{store_type}"
            try:
                # 벡터 저장소 파일 삭제
                if os.path.exists(vector_path):
                    for file in os.listdir(vector_path):
                        os.remove(os.path.join(vector_path, file))
                
                # 벡터 저장소 초기화
                self.rag_system.vector_stores[store_type] = None
                
                messagebox.showinfo("성공", f"{source_name} 벡터 저장소가 초기화되었습니다.", parent=dialog)
                
                # 대화상자 업데이트
                if dialog:
                    dialog.destroy()
                    self.manage_vector_stores()
            except Exception as e:
                messagebox.showerror("오류", f"벡터 저장소 초기화 중 오류가 발생했습니다:\n{str(e)}", parent=dialog)
    
    def _clear_all_vector_stores(self, dialog=None):
        """모든 벡터 저장소 초기화"""
        result = messagebox.askyesno(
            "확인",
            "정말 모든 벡터 저장소를 초기화하시겠습니까?\n이 작업은 되돌릴 수 없습니다.",
            parent=dialog
        )
        
        if result:
            try:
                for store_type in self.rag_system.vector_stores.keys():
                    vector_path = f"vectors/{store_type}"
                    
                    # 벡터 저장소 파일 삭제
                    if os.path.exists(vector_path):
                        for file in os.listdir(vector_path):
                            os.remove(os.path.join(vector_path, file))
                    
                    # 벡터 저장소 초기화
                    self.rag_system.vector_stores[store_type] = None
                
                messagebox.showinfo("성공", "모든 벡터 저장소가 초기화되었습니다.", parent=dialog)
                
                # 대화상자 업데이트
                if dialog:
                    dialog.destroy()
                    self.manage_vector_stores()
            except Exception as e:
                messagebox.showerror("오류", f"벡터 저장소 초기화 중 오류가 발생했습니다:\n{str(e)}", parent=dialog)
    
    def show_help(self):
        """도움말 표시"""
        help_dialog = ctk.CTkToplevel(self)
        help_dialog.title("도움말")
        help_dialog.geometry("700x600")
        help_dialog.transient(self)
        help_dialog.grab_set()
        
        # 중앙에 배치
        help_dialog.update_idletasks()
        width = help_dialog.winfo_width()
        height = help_dialog.winfo_height()
        x = (help_dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (help_dialog.winfo_screenheight() // 2) - (height // 2)
        help_dialog.geometry(f'+{x}+{y}')
        
        # 도움말 내용
        help_frame = ctk.CTkScrollableFrame(help_dialog)
        help_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        help_text = """
        # 지식 검색 시스템 사용 설명서
        
        ## 개요
        이 애플리케이션은 다양한 유형의 문서(텍스트, CSV, PDF, 웹)를 처리하고 검색할 수 있는 지식 검색 시스템입니다.
        
        ## 시작하기
        1. 처음 시작할 때는 LLM 모델을 로드해야 합니다. '모델 로드' 버튼을 클릭하여 모델 파일(.gguf, .ggml, .bin)을 선택하세요.
        2. 검색할 문서를 추가하세요. 텍스트, CSV, PDF 파일 또는 웹 페이지를 추가할 수 있습니다.
        3. 질문이나 검색하고 싶은 내용을 입력하면 시스템이 관련 정보를 검색하여 응답을 생성합니다.
        
        ## 데이터 추가하기
        * 텍스트 추가: 직접 텍스트를 입력하거나 텍스트 파일을 추가할 수 있습니다.
        * CSV 파일: 표 형태의 데이터를 추가합니다.
        * PDF 파일: PDF 문서의 텍스트를 추출하여 처리합니다.
        * 웹 페이지: URL을 입력하여 웹 페이지의 내용을 추가합니다.
        
        ## 검색 기능
        * 왼쪽 사이드바에서 검색할 데이터 유형(텍스트, CSV, PDF, 웹)을 선택할 수 있습니다.
        * 위키피디아 검색 기능이 활성화되어 있으면, 시스템은 질문과 관련된 위키피디아 정보도 검색합니다.
        
        ## 응답 생성
        * 입력창에 질문을 입력하고 '전송' 버튼을 클릭하거나 Enter 키를 누르면 응답이 생성됩니다.
        * 응답 생성 중에는 '중지' 버튼을 클릭하여 생성을 중단할 수 있습니다.
        
        ## 대화 관리
        * 대화 내용은 JSON 또는 텍스트 형식으로 저장할 수 있습니다.
        * '대화 초기화' 버튼을 클릭하여 현재 대화 내용을 모두 지울 수 있습니다.
        * 이전에 저장한 대화를 불러올 수 있습니다.
        
        ## 캐시 관리
        * 응답 캐싱 기능이 활성화되어 있으면, 동일하거나 유사한 질문에 대해 더 빠르게 응답합니다.
        * '도구 > 캐시 관리'에서 캐시 정보를 확인하거나 캐시를 비울 수 있습니다.
        
        ## 벡터 저장소 관리
        * '도구 > 벡터 저장소 관리'에서 각 데이터 유형별 벡터 저장소 정보를 확인하거나 초기화할 수 있습니다.
        
        ## 단축키
        * Enter: 메시지 전송
        * Shift+Enter: 줄바꿈
        
        ## 문제 해결
        * 모델 로드 오류: 모델 파일 경로가 올바른지 확인하세요.
        * 파일 처리 오류: 파일 형식과 인코딩이 지원되는지 확인하세요.
        * 응답 생성 실패: 모델이 로드되었는지, 관련 데이터가 추가되었는지 확인하세요.
        
        ## 추가 정보
        * 로그 파일: 애플리케이션 실행 중 발생하는 로그는 'app.log' 파일에 기록됩니다.
        * 설정 파일: 애플리케이션 설정은 'configs/app_config.json' 파일에 저장됩니다.
        """
        
        if MARKDOWN_AVAILABLE and TKINTERWEB_AVAILABLE:
            # 마크다운 렌더링
            html_content = markdown.markdown(
                help_text,
                extensions=['nl2br', 'fenced_code', 'tables']
            )
            
            custom_style = """
            <style>
              body { font-family: sans-serif; line-height: 1.5; padding: 10px; background-color: #FFFFFF; color: #2C3E50; }
              h1 { color: #2E86C1; border-bottom: 1px solid #AED6F1; padding-bottom: 5px; }
              h2 { color: #2471A3; margin-top: 20px; }
              ul { margin-left: 20px; padding-left: 20px; }
              li { margin-bottom: 5px; }
              code { background-color: #f5f5f5; padding: 2px 4px; border-radius: 4px; }
            </style>
            """
            
            html_content = custom_style + html_content
            
            html_frame = HtmlFrame(help_frame, width=650, height=550, messages_enabled=False)
            html_frame.load_html(html_content)
            html_frame.pack(fill="both", expand=True)
        else:
            # 일반 텍스트 표시
            help_text_box = ctk.CTkTextbox(help_frame, width=650, height=550, wrap="word")
            help_text_box.pack(fill="both", expand=True)
            help_text_box.insert("0.0", help_text)
            help_text_box.configure(state="disabled")
        
        # 닫기 버튼
        ctk.CTkButton(
            help_dialog,
            text="닫기",
            command=help_dialog.destroy,
            width=100
        ).pack(pady=(0, 20))
    
    def show_about(self):
        """정보 대화상자 표시"""
        about_dialog = ctk.CTkToplevel(self)
        about_dialog.title("정보")
        about_dialog.geometry("400x300")
        about_dialog.transient(self)
        about_dialog.grab_set()
        
        # 중앙에 배치
        about_dialog.update_idletasks()
        width = about_dialog.winfo_width()
        height = about_dialog.winfo_height()
        x = (about_dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (about_dialog.winfo_screenheight() // 2) - (height // 2)
        about_dialog.geometry(f'+{x}+{y}')
        
        # 앱 제목
        ctk.CTkLabel(
            about_dialog,
            text="지식 검색 시스템",
            font=("Helvetica", 20, "bold"),
            text_color=THEME['primary']
        ).pack(pady=(20, 5))
        
        # 버전
        ctk.CTkLabel(
            about_dialog,
            text="버전 1.0.0",
            font=("Helvetica", 14)
        ).pack(pady=(0, 10))
        
        # 구분선
        separator = ctk.CTkFrame(about_dialog, height=1, fg_color=THEME["secondary"])
        separator.pack(fill="x", padx=20, pady=10)
        
        # 소개
        ctk.CTkLabel(
            about_dialog,
            text="이 애플리케이션은 LLM과 벡터 검색을 활용한\n지식 검색 시스템입니다.",
            font=("Helvetica", 14),
            justify="center"
        ).pack(pady=5)
        
        # 패키지 정보
        package_frame = ctk.CTkFrame(about_dialog, fg_color="transparent")
        package_frame.pack(fill="x", padx=20, pady=5)
        
        package_info = "사용된 주요 패키지:\n"
        package_info += "- LangChain\n"
        package_info += "- CustomTkinter\n"
        package_info += "- FAISS\n"
        package_info += "- LlamaCpp\n"
        
        if 'fitz' in sys.modules:
            package_info += "- PyMuPDF\n"
        
        if WEB_SUPPORT:
            package_info += "- BeautifulSoup\n"
            package_info += "- Wikipedia\n"
        
        if TKINTERWEB_AVAILABLE:
            package_info += "- tkinterweb\n"
        
        if MARKDOWN_AVAILABLE:
            package_info += "- markdown\n"
        
        ctk.CTkLabel(
            package_frame,
            text=package_info,
            font=("Helvetica", 12),
            justify="left"
        ).pack(anchor="w")
        
        # 닫기 버튼
        ctk.CTkButton(
            about_dialog,
            text="닫기",
            command=about_dialog.destroy,
            width=100
        ).pack(pady=(10, 20))
    
    def on_closing(self):
        """앱 종료 처리"""
        # 필요한 데이터 저장
        if self.rag_system.cache.cache:
            self.rag_system.cache.save_cache()
        
        self.config_manager.save_config()
        
        # 앱 종료
        self.quit()
        self.destroy()

# 메인 실행
if __name__ == "__main__":
    try:
        app = KnowledgeSearchApp()
        app.protocol("WM_DELETE_WINDOW", app.on_closing)
        app.mainloop()
    except Exception as e:
        # 초기 오류 처리
        error_message = f"애플리케이션 시작 중 오류가 발생했습니다:\n{str(e)}\n\n상세 오류:\n{traceback.format_exc()}"
        print(error_message)
        
        # 가능하면 오류 대화상자 표시
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("오류", error_message)
            root.destroy()
        except:
            # 대화상자 표시도 실패한 경우 그냥 종료
            pass
        
        sys.exit(1)