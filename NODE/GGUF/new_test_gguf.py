# -*- coding: utf-8 -*-
"""
GGUF + LangChain CSV 검색 시스템 (개선판)
벡터 DB 저장/로드 최적화 및 검색 성능 개선
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
import time
from tqdm import tqdm
import pickle

# LangChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.prompts import PromptTemplate

# GGUF
from llama_cpp import Llama

# UI
import customtkinter as ctk
from tkinter import messagebox, filedialog, ttk
import tkinter as tk
import threading

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('csv_search.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GGUFLangChainLLM(LLM):
    """LangChain용 GGUF LLM Wrapper"""
    
    model: Llama
    
    @property
    def _llm_type(self) -> str:
        return "gguf"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """LLM 호출"""
        response = self.model(
            prompt,
            max_tokens=kwargs.get("max_tokens", 512),
            temperature=kwargs.get("temperature", 0.7),
            stop=stop or ["</s>", "\n\n"],
            echo=False
        )
        return response['choices'][0]['text']

class CSVSearchSystem:
    """CSV 검색 시스템 - LangChain 통합"""
    
    def __init__(self):
        # 경로 설정
        self.embedding_model_path = "./models/embeddings/BAAI/bge-m3"
        self.csv_dir = "./csv_data"
        self.vector_db_dir = "./vector_db"
        self.metadata_path = "./vector_db/metadata.pkl"
        
        # 모델
        self.embeddings = None
        self.vector_store = None
        self.llm = None
        self.qa_chain = None
        
        # 데이터
        self.all_data = pd.DataFrame()
        self.metadata = {}
        
        # 설정
        self.chunk_size = 500
        self.chunk_overlap = 50
        
    def load_embeddings(self):
        """임베딩 모델 로드"""
        logger.info("🔄 임베딩 모델 로드 중...")
        
        # 오프라인 모델 경로 확인
        if not Path(self.embedding_model_path).exists():
            # 온라인 모델 사용
            model_name = "BAAI/bge-m3"
            logger.info(f"로컬 모델이 없어 온라인 모델 사용: {model_name}")
        else:
            model_name = self.embedding_model_path
            
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("✅ 임베딩 모델 로드 완료")
        
    def load_csv_data(self):
        """CSV 데이터 로드"""
        csv_files = list(Path(self.csv_dir).glob("*.csv"))
        if not csv_files:
            # 테스트용 샘플 디렉토리
            csv_files = list(Path(".").glob("*.csv"))
            
        logger.info(f"📁 {len(csv_files)}개 CSV 파일 발견")
        
        all_dfs = []
        for csv_file in tqdm(csv_files, desc="CSV 로딩"):
            try:
                df = pd.read_csv(csv_file, encoding='utf-8')
                df['source_file'] = csv_file.name
                df['file_date'] = csv_file.stem
                all_dfs.append(df)
            except Exception as e:
                logger.error(f"CSV 로드 오류 ({csv_file}): {e}")
                
        if all_dfs:
            self.all_data = pd.concat(all_dfs, ignore_index=True)
            logger.info(f"✅ 총 {len(self.all_data):,}개 행 로드 완료")
        else:
            logger.warning("⚠️ 로드된 CSV 데이터가 없습니다")
            
    def create_documents(self) -> List[Document]:
        """문서 생성 - 개선된 버전"""
        documents = []
        
        if self.all_data.empty:
            return documents
            
        logger.info("📄 문서 생성 시작...")
        
        # 1. 전체 요약 문서
        overall_summary = f"""
전체 데이터 요약:
- 총 데이터 수: {len(self.all_data):,}개
- 날짜 범위: {self.all_data['file_date'].min()} ~ {self.all_data['file_date'].max()}
- 컬럼: {', '.join(self.all_data.columns.tolist())}
"""
        documents.append(Document(
            page_content=overall_summary,
            metadata={"type": "overall_summary", "doc_id": "summary_0"}
        ))
        
        # 2. 날짜별 요약
        for date in self.all_data['file_date'].unique():
            daily_data = self.all_data[self.all_data['file_date'] == date]
            
            # TOTALCNT 컬럼이 있는 경우
            if 'TOTALCNT' in daily_data.columns:
                daily_text = f"""
날짜: {date}
데이터 수: {len(daily_data):,}개
TOTALCNT 통계:
- 평균: {daily_data['TOTALCNT'].mean():.0f}
- 최대: {daily_data['TOTALCNT'].max()}
- 최소: {daily_data['TOTALCNT'].min()}
- 표준편차: {daily_data['TOTALCNT'].std():.0f}
"""
                # 최대값 시간 정보 추가
                if 'CURRTIME' in daily_data.columns:
                    max_time = daily_data.loc[daily_data['TOTALCNT'].idxmax(), 'CURRTIME']
                    daily_text += f"- 최대값 시간: {max_time}\n"
                    
            else:
                # 일반적인 요약
                daily_text = f"""
날짜: {date}
데이터 수: {len(daily_data):,}개
컬럼 정보: {', '.join(daily_data.columns.tolist())}
"""
            
            documents.append(Document(
                page_content=daily_text,
                metadata={
                    "type": "daily_summary", 
                    "date": date,
                    "doc_id": f"daily_{date}"
                }
            ))
            
        # 3. 시간대별 상세 데이터 (샘플링)
        if 'CURRTIME' in self.all_data.columns and 'TOTALCNT' in self.all_data.columns:
            # 날짜별로 그룹화
            for date in self.all_data['file_date'].unique()[:5]:  # 최근 5일만
                daily_data = self.all_data[self.all_data['file_date'] == date]
                
                # 시간대별 집계
                daily_data['hour'] = daily_data['CURRTIME'].astype(str).str[8:10]
                
                for hour in daily_data['hour'].unique():
                    hour_data = daily_data[daily_data['hour'] == hour]
                    if len(hour_data) > 3:  # 데이터가 충분한 경우만
                        hour_text = f"""
날짜: {date} {hour}시
데이터 수: {len(hour_data)}개
TOTALCNT 평균: {hour_data['TOTALCNT'].mean():.0f}
TOTALCNT 최대: {hour_data['TOTALCNT'].max()}
TOTALCNT 최소: {hour_data['TOTALCNT'].min()}
"""
                        # 다른 컬럼들의 평균값도 추가
                        numeric_cols = hour_data.select_dtypes(include=[np.number]).columns
                        for col in numeric_cols[:5]:  # 최대 5개 컬럼만
                            if col not in ['TOTALCNT', 'hour']:
                                hour_text += f"{col} 평균: {hour_data[col].mean():.0f}\n"
                                
                        documents.append(Document(
                            page_content=hour_text,
                            metadata={
                                "type": "hourly_detail",
                                "date": date,
                                "hour": hour,
                                "doc_id": f"hour_{date}_{hour}"
                            }
                        ))
        
        logger.info(f"✅ {len(documents)}개 문서 생성 완료")
        return documents
        
    def build_vector_db(self):
        """벡터 DB 구축 - 개선된 버전"""
        logger.info("🔨 벡터 DB 구축 시작...")
        
        # CSV 데이터 로드
        self.load_csv_data()
        
        if self.all_data.empty:
            raise ValueError("CSV 데이터가 없습니다")
            
        # 문서 생성
        documents = self.create_documents()
        
        # 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        split_docs = text_splitter.split_documents(documents)
        
        logger.info(f"📄 {len(documents)}개 문서를 {len(split_docs)}개 청크로 분할")
        
        # FAISS 벡터 스토어 생성
        self.vector_store = FAISS.from_documents(
            split_docs,
            self.embeddings
        )
        
        # 벡터 DB 저장
        Path(self.vector_db_dir).mkdir(exist_ok=True)
        self.vector_store.save_local(self.vector_db_dir)
        
        # 메타데이터 저장
        metadata = {
            'total_documents': len(documents),
            'total_chunks': len(split_docs),
            'chunk_size': self.chunk_size,
            'data_shape': self.all_data.shape,
            'columns': self.all_data.columns.tolist(),
            'date_range': {
                'min': str(self.all_data['file_date'].min()),
                'max': str(self.all_data['file_date'].max())
            },
            'build_time': datetime.now().isoformat()
        }
        
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
            
        # 원본 데이터도 저장 (빠른 검색용)
        self.all_data.to_pickle(Path(self.vector_db_dir) / "raw_data.pkl")
        
        logger.info(f"✅ 벡터 DB 구축 완료 (저장 위치: {self.vector_db_dir})")
        
    def load_vector_db(self) -> bool:
        """벡터 DB 로드 - 개선된 버전"""
        try:
            if not Path(self.vector_db_dir).exists():
                logger.warning("벡터 DB 디렉토리가 없습니다")
                return False
                
            logger.info("📂 벡터 DB 로드 중...")
            
            # 벡터 스토어 로드
            self.vector_store = FAISS.load_local(
                self.vector_db_dir,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            # 메타데이터 로드
            if Path(self.metadata_path).exists():
                with open(self.metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                logger.info(f"📊 메타데이터: {self.metadata}")
                
            # 원본 데이터 로드
            raw_data_path = Path(self.vector_db_dir) / "raw_data.pkl"
            if raw_data_path.exists():
                self.all_data = pd.read_pickle(raw_data_path)
                logger.info(f"✅ 원본 데이터 로드: {self.all_data.shape}")
                
            logger.info("✅ 벡터 DB 로드 완료")
            return True
            
        except Exception as e:
            logger.error(f"벡터 DB 로드 실패: {e}")
            return False
            
    def load_gguf_model(self, model_path: str):
        """GGUF 모델 로드 및 QA 체인 설정"""
        logger.info(f"🤖 GGUF 모델 로드 중: {model_path}")
        
        # GGUF 모델 로드
        llama_model = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_gpu_layers=-1,  # GPU 자동 감지
            n_threads=8,
            verbose=False
        )
        
        # LangChain LLM 래퍼
        self.llm = GGUFLangChainLLM(model=llama_model)
        
        # 프롬프트 템플릿
        prompt_template = """주어진 컨텍스트를 바탕으로 질문에 답해주세요.

컨텍스트:
{context}

질문: {question}

답변: 데이터를 분석한 결과, """

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # QA 체인 생성
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 5}  # 상위 5개 문서 검색
            ),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        logger.info("✅ GGUF 모델 및 QA 체인 설정 완료")
        
    def search(self, query: str, use_llm: bool = True) -> Dict[str, Any]:
        """통합 검색 함수"""
        results = {
            'query': query,
            'vector_results': [],
            'direct_results': '',
            'llm_answer': '',
            'source_docs': []
        }
        
        try:
            # 1. 벡터 검색
            if self.vector_store:
                docs = self.vector_store.similarity_search_with_score(query, k=5)
                results['vector_results'] = [
                    {
                        'content': doc.page_content,
                        'metadata': doc.metadata,
                        'score': float(score)
                    }
                    for doc, score in docs
                ]
                
            # 2. 직접 데이터 검색 (pandas)
            if not self.all_data.empty:
                direct_result = self._direct_search(query)
                results['direct_results'] = direct_result
                
            # 3. LLM 기반 답변
            if use_llm and self.qa_chain:
                qa_result = self.qa_chain({"query": query})
                results['llm_answer'] = qa_result['result']
                results['source_docs'] = [
                    {
                        'content': doc.page_content,
                        'metadata': doc.metadata
                    }
                    for doc in qa_result.get('source_documents', [])
                ]
                
        except Exception as e:
            logger.error(f"검색 오류: {e}")
            results['error'] = str(e)
            
        return results
        
    def _direct_search(self, query: str) -> str:
        """직접 데이터 검색"""
        results = []
        query_lower = query.lower()
        
        # TOTALCNT 관련 검색
        if 'TOTALCNT' in self.all_data.columns:
            if "최대" in query or "max" in query_lower:
                max_row = self.all_data.loc[self.all_data['TOTALCNT'].idxmax()]
                results.append(f"전체 최대값: TOTALCNT={max_row['TOTALCNT']}")
                if 'CURRTIME' in max_row:
                    results.append(f"발생 시간: {max_row['CURRTIME']}")
                if 'file_date' in max_row:
                    results.append(f"날짜: {max_row['file_date']}")
                    
            elif "1450" in query or "1500" in query:
                threshold = 1450 if "1450" in query else 1500
                filtered = self.all_data[self.all_data['TOTALCNT'] >= threshold]
                if not filtered.empty:
                    results.append(f"\nTOTALCNT {threshold} 이상:")
                    results.append(f"- 총 {len(filtered)}개 데이터")
                    
                    # 날짜별 집계
                    if 'file_date' in filtered.columns:
                        date_counts = filtered['file_date'].value_counts()
                        results.append("\n날짜별 분포:")
                        for date, count in date_counts.head(5).items():
                            results.append(f"  - {date}: {count}개")
                            
            elif "평균" in query:
                avg_total = self.all_data['TOTALCNT'].mean()
                results.append(f"전체 평균: {avg_total:.0f}")
                
                # 날짜별 평균
                if 'file_date' in self.all_data.columns:
                    date_avg = self.all_data.groupby('file_date')['TOTALCNT'].mean()
                    results.append("\n날짜별 평균:")
                    for date, avg in date_avg.head(5).items():
                        results.append(f"  - {date}: {avg:.0f}")
                        
        # 특정 날짜 검색
        for col in ['file_date', 'CURRTIME']:
            if col in self.all_data.columns:
                for date_part in query.split():
                    if date_part.replace('-', '').isdigit() and len(date_part) >= 6:
                        mask = self.all_data[col].astype(str).str.contains(date_part)
                        filtered = self.all_data[mask]
                        if not filtered.empty:
                            results.append(f"\n{date_part} 관련 데이터: {len(filtered)}개")
                            break
                            
        return "\n".join(results) if results else "관련 데이터를 찾을 수 없습니다."

class ModernSearchUI(ctk.CTk):
    """현대적인 검색 UI"""
    
    def __init__(self):
        super().__init__()
        
        self.title("GGUF + LangChain CSV 검색 시스템")
        self.geometry("1200x800")
        
        # 테마 설정
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # 검색 시스템
        self.search_system = CSVSearchSystem()
        
        # UI 구성
        self.setup_ui()
        
    def setup_ui(self):
        """UI 구성"""
        # 메인 레이아웃
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # 좌측 사이드바
        self.setup_sidebar()
        
        # 우측 메인 영역
        self.setup_main_area()
        
    def setup_sidebar(self):
        """사이드바 구성"""
        sidebar = ctk.CTkFrame(self, width=250)
        sidebar.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        sidebar.grid_propagate(False)
        
        # 타이틀
        title = ctk.CTkLabel(
            sidebar,
            text="🔍 CSV Search AI",
            font=("Arial", 24, "bold")
        )
        title.pack(pady=20)
        
        # 시스템 상태
        self.status_frame = ctk.CTkFrame(sidebar)
        self.status_frame.pack(fill="x", padx=10, pady=10)
        
        self.embedding_status = ctk.CTkLabel(
            self.status_frame,
            text="❌ 임베딩: 미로드",
            font=("Arial", 12)
        )
        self.embedding_status.pack(anchor="w", padx=10, pady=2)
        
        self.vector_status = ctk.CTkLabel(
            self.status_frame,
            text="❌ 벡터DB: 미로드",
            font=("Arial", 12)
        )
        self.vector_status.pack(anchor="w", padx=10, pady=2)
        
        self.llm_status = ctk.CTkLabel(
            self.status_frame,
            text="❌ LLM: 미로드",
            font=("Arial", 12)
        )
        self.llm_status.pack(anchor="w", padx=10, pady=2)
        
        # 버튼들
        ctk.CTkLabel(sidebar, text="초기 설정", font=("Arial", 14, "bold")).pack(pady=(20, 10))
        
        self.init_btn = ctk.CTkButton(
            sidebar,
            text="1. 임베딩 모델 로드",
            command=self.load_embeddings,
            height=35
        )
        self.init_btn.pack(fill="x", padx=10, pady=5)
        
        self.vector_btn = ctk.CTkButton(
            sidebar,
            text="2. 벡터 DB 로드/구축",
            command=self.setup_vector_db,
            height=35
        )
        self.vector_btn.pack(fill="x", padx=10, pady=5)
        
        self.llm_btn = ctk.CTkButton(
            sidebar,
            text="3. GGUF 모델 로드",
            command=self.load_gguf,
            height=35
        )
        self.llm_btn.pack(fill="x", padx=10, pady=5)
        
        # 검색 옵션
        ctk.CTkLabel(sidebar, text="검색 옵션", font=("Arial", 14, "bold")).pack(pady=(20, 10))
        
        self.use_llm_var = ctk.BooleanVar(value=True)
        self.use_llm_check = ctk.CTkCheckBox(
            sidebar,
            text="LLM 답변 사용",
            variable=self.use_llm_var
        )
        self.use_llm_check.pack(anchor="w", padx=10, pady=5)
        
        # 예제 쿼리
        ctk.CTkLabel(sidebar, text="예제 검색어", font=("Arial", 14, "bold")).pack(pady=(20, 10))
        
        examples = [
            "TOTALCNT 최대값은?",
            "1450 이상인 데이터",
            "8월 7일 15시 데이터",
            "전체 평균 통계"
        ]
        
        for ex in examples:
            ctk.CTkButton(
                sidebar,
                text=ex,
                command=lambda e=ex: self.search_box.insert("end", e),
                height=30,
                fg_color="gray40"
            ).pack(fill="x", padx=10, pady=2)
            
    def setup_main_area(self):
        """메인 영역 구성"""
        main_frame = ctk.CTkFrame(self)
        main_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)
        
        # 검색 영역
        search_frame = ctk.CTkFrame(main_frame)
        search_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        search_frame.grid_columnconfigure(0, weight=1)
        
        self.search_box = ctk.CTkEntry(
            search_frame,
            placeholder_text="검색어를 입력하세요...",
            font=("Arial", 16),
            height=40
        )
        self.search_box.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.search_box.bind("<Return>", lambda e: self.search())
        
        self.search_btn = ctk.CTkButton(
            search_frame,
            text="검색",
            command=self.search,
            width=100,
            height=40,
            font=("Arial", 16)
        )
        self.search_btn.grid(row=0, column=1)
        
        # 결과 표시 영역 (탭)
        self.tab_view = ctk.CTkTabview(main_frame)
        self.tab_view.grid(row=1, column=0, sticky="nsew", padx=10, pady=(5, 10))
        
        # 탭 생성
        self.tab_view.add("종합 결과")
        self.tab_view.add("벡터 검색")
        self.tab_view.add("직접 검색")
        self.tab_view.add("소스 문서")
        
        # 각 탭에 텍스트 위젯
        self.result_text = ctk.CTkTextbox(
            self.tab_view.tab("종합 결과"),
            font=("Consolas", 12),
            wrap="word"
        )
        self.result_text.pack(fill="both", expand=True)
        
        self.vector_text = ctk.CTkTextbox(
            self.tab_view.tab("벡터 검색"),
            font=("Consolas", 12),
            wrap="word"
        )
        self.vector_text.pack(fill="both", expand=True)
        
        self.direct_text = ctk.CTkTextbox(
            self.tab_view.tab("직접 검색"),
            font=("Consolas", 12),
            wrap="word"
        )
        self.direct_text.pack(fill="both", expand=True)
        
        self.source_text = ctk.CTkTextbox(
            self.tab_view.tab("소스 문서"),
            font=("Consolas", 12),
            wrap="word"
        )
        self.source_text.pack(fill="both", expand=True)
        
        # 진행률 표시
        self.progress = ttk.Progressbar(
            main_frame,
            mode='indeterminate',
            style="TProgressbar"
        )
        
    def load_embeddings(self):
        """임베딩 로드"""
        self.show_progress()
        
        def task():
            try:
                self.search_system.load_embeddings()
                self.after(0, lambda: self.embedding_status.configure(text="✅ 임베딩: 로드됨"))
                self.after(0, lambda: messagebox.showinfo("성공", "임베딩 모델 로드 완료"))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("오류", f"임베딩 로드 실패:\n{str(e)}"))
            finally:
                self.after(0, self.hide_progress)
                
        threading.Thread(target=task, daemon=True).start()
        
    def setup_vector_db(self):
        """벡터 DB 설정"""
        if not self.search_system.embeddings:
            messagebox.showwarning("경고", "먼저 임베딩 모델을 로드하세요")
            return
            
        # 기존 DB 확인
        if self.search_system.load_vector_db():
            result = messagebox.askyesnocancel(
                "벡터 DB",
                "기존 벡터 DB가 발견되었습니다.\n\n"
                "Yes: 기존 DB 사용\n"
                "No: 새로 구축\n"
                "Cancel: 취소"
            )
            
            if result is True:  # Yes - 기존 사용
                self.vector_status.configure(text="✅ 벡터DB: 로드됨")
                messagebox.showinfo("성공", "기존 벡터 DB를 로드했습니다")
                return
            elif result is None:  # Cancel
                return
                
        # 새로 구축
        self.show_progress()
        
        def task():
            try:
                self.search_system.build_vector_db()
                self.after(0, lambda: self.vector_status.configure(text="✅ 벡터DB: 구축됨"))
                self.after(0, lambda: messagebox.showinfo("성공", "벡터 DB 구축 완료"))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("오류", f"벡터 DB 구축 실패:\n{str(e)}"))
            finally:
                self.after(0, self.hide_progress)
                
        threading.Thread(target=task, daemon=True).start()
        
    def load_gguf(self):
        """GGUF 모델 로드"""
        if not self.search_system.vector_store:
            messagebox.showwarning("경고", "먼저 벡터 DB를 설정하세요")
            return
            
        filepath = filedialog.askopenfilename(
            title="GGUF 모델 선택",
            filetypes=[("GGUF Files", "*.gguf"), ("All Files", "*.*")]
        )
        
        if not filepath:
            return
            
        self.show_progress()
        
        def task():
            try:
                self.search_system.load_gguf_model(filepath)
                self.after(0, lambda: self.llm_status.configure(text="✅ LLM: 로드됨"))
                self.after(0, lambda: messagebox.showinfo("성공", "GGUF 모델 로드 완료"))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("오류", f"모델 로드 실패:\n{str(e)}"))
            finally:
                self.after(0, self.hide_progress)
                
        threading.Thread(target=task, daemon=True).start()
        
    def search(self):
        """검색 실행"""
        query = self.search_box.get().strip()
        if not query:
            return
            
        if not self.search_system.vector_store:
            messagebox.showwarning("경고", "먼저 벡터 DB를 설정하세요")
            return
            
        self.show_progress()
        
        # 모든 텍스트 초기화
        for text_widget in [self.result_text, self.vector_text, self.direct_text, self.source_text]:
            text_widget.delete("1.0", "end")
            
        def task():
            try:
                # 검색 실행
                results = self.search_system.search(
                    query, 
                    use_llm=self.use_llm_var.get()
                )
                
                # UI 업데이트는 메인 스레드에서
                self.after(0, lambda: self.display_results(results))
                
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("검색 오류", str(e)))
            finally:
                self.after(0, self.hide_progress)
                
        threading.Thread(target=task, daemon=True).start()
        
    def display_results(self, results: Dict[str, Any]):
        """검색 결과 표시"""
        # 종합 결과
        summary = f"🔍 검색어: {results['query']}\n"
        summary += "="*50 + "\n\n"
        
        if results.get('llm_answer'):
            summary += "🤖 AI 답변:\n"
            summary += results['llm_answer'] + "\n\n"
            
        if results.get('direct_results'):
            summary += "📊 직접 검색 결과:\n"
            summary += results['direct_results'] + "\n"
            
        self.result_text.insert("1.0", summary)
        
        # 벡터 검색 결과
        if results.get('vector_results'):
            vector_text = "🔍 유사도 검색 결과:\n\n"
            for i, doc in enumerate(results['vector_results'], 1):
                vector_text += f"[{i}] 유사도: {1-doc['score']:.3f}\n"
                vector_text += f"내용: {doc['content']}\n"
                vector_text += f"메타데이터: {doc['metadata']}\n"
                vector_text += "-"*40 + "\n\n"
            self.vector_text.insert("1.0", vector_text)
            
        # 직접 검색 결과
        if results.get('direct_results'):
            self.direct_text.insert("1.0", results['direct_results'])
            
        # 소스 문서
        if results.get('source_docs'):
            source_text = "📄 참조된 소스 문서:\n\n"
            for i, doc in enumerate(results['source_docs'], 1):
                source_text += f"[문서 {i}]\n"
                source_text += doc['content'] + "\n"
                source_text += f"출처: {doc['metadata']}\n"
                source_text += "="*50 + "\n\n"
            self.source_text.insert("1.0", source_text)
            
        # 첫 번째 탭으로 전환
        self.tab_view.set("종합 결과")
        
    def show_progress(self):
        """진행률 표시"""
        self.progress.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        self.progress.start()
        
    def hide_progress(self):
        """진행률 숨기기"""
        self.progress.stop()
        self.progress.grid_remove()

def main():
    """메인 함수"""
    # 필요한 디렉토리 생성
    for dir_name in ["csv_data", "vector_db", "models"]:
        Path(dir_name).mkdir(exist_ok=True)
        
    # UI 실행
    app = ModernSearchUI()
    app.mainloop()

if __name__ == "__main__":
    main()