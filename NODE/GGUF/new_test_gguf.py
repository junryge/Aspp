# -*- coding: utf-8 -*-
"""
폐쇄망 GGUF + CSV 검색 시스템 (간단 버전)
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import logging

# LangChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

# GGUF
from llama_cpp import Llama

# UI
import customtkinter as ctk
from tkinter import messagebox, filedialog

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleCSVSearch:
    """간단한 CSV 검색 시스템"""
    
    def __init__(self):
        # 경로 설정
        self.embedding_model_path = "./offline_models/all-MiniLM-L6-v2"
        self.csv_dir = "./output_by_date"
        self.vector_db_path = "./vector_db/faiss_index"
        
        # 모델
        self.embeddings = None
        self.vector_store = None
        self.llm = None
        
        # 데이터
        self.all_data = pd.DataFrame()
        
    def load_embeddings(self):
        """임베딩 모델 로드"""
        logger.info("임베딩 모델 로드 중...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_path,
            model_kwargs={'device': 'cpu'}
        )
        logger.info("임베딩 모델 로드 완료")
        
    def load_all_csv_data(self):
        """모든 CSV 파일을 하나의 DataFrame으로 로드"""
        csv_files = list(Path(self.csv_dir).glob("*.csv"))
        logger.info(f"CSV 파일 {len(csv_files)}개 발견")
        
        all_dfs = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            # 파일명을 날짜로 추가
            df['FILE_DATE'] = csv_file.stem
            all_dfs.append(df)
        
        self.all_data = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"총 {len(self.all_data)}개 행 로드 완료")
        
    def create_search_documents(self) -> List[Document]:
        """검색용 문서 생성 (심플 버전)"""
        documents = []
        
        # 10개 행씩 묶어서 문서화
        chunk_size = 10
        for i in range(0, len(self.all_data), chunk_size):
            chunk = self.all_data.iloc[i:i+chunk_size]
            
            # 간단한 텍스트 표현
            text = f"""
날짜: {chunk['FILE_DATE'].iloc[0]}
시간: {chunk['CURRTIME'].iloc[0]} ~ {chunk['CURRTIME'].iloc[-1]}
TOTALCNT 범위: {chunk['TOTALCNT'].min()} ~ {chunk['TOTALCNT'].max()}
TOTALCNT 평균: {chunk['TOTALCNT'].mean():.0f}

상세 데이터:
"""
            # 주요 컬럼만 포함
            for _, row in chunk.iterrows():
                text += f"시간:{row['CURRTIME']}, 총계:{row['TOTALCNT']}, "
                text += f"M14A↔M10A:{row['M14AM10ASUM']}, M14A↔M14B:{row['M14AM14BSUM']}\n"
            
            doc = Document(
                page_content=text,
                metadata={
                    "date": chunk['FILE_DATE'].iloc[0],
                    "start_time": chunk['CURRTIME'].iloc[0],
                    "end_time": chunk['CURRTIME'].iloc[-1]
                }
            )
            documents.append(doc)
            
        logger.info(f"{len(documents)}개 문서 생성 완료")
        return documents
    
    def build_vector_db(self):
        """벡터 DB 구축"""
        # CSV 로드
        self.load_all_csv_data()
        
        # 문서 생성
        documents = self.create_search_documents()
        
        # FAISS 벡터 스토어 생성
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        
        # 저장
        Path("./vector_db").mkdir(exist_ok=True)
        self.vector_store.save_local(self.vector_db_path)
        logger.info("벡터 DB 저장 완료")
        
    def load_vector_db(self):
        """기존 벡터 DB 로드"""
        if Path(self.vector_db_path).exists():
            self.vector_store = FAISS.load_local(
                self.vector_db_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("기존 벡터 DB 로드 완료")
            return True
        return False
        
    def load_gguf_model(self, model_path: str):
        """GGUF 모델 로드"""
        logger.info(f"GGUF 모델 로드 중: {model_path}")
        
        self.llm = Llama(
            model_path=model_path,
            n_ctx=8192,
            n_gpu_layers=35,  # Q6_K용
            n_threads=8,
            verbose=False
        )
        
        logger.info("GGUF 모델 로드 완료")
        
    def search(self, query: str, k: int = 3) -> List[Document]:
        """검색 실행"""
        if not self.vector_store:
            raise ValueError("벡터 DB가 로드되지 않았습니다")
        
        return self.vector_store.similarity_search(query, k=k)
    
    def answer_with_data(self, query: str, search_results: List[Document]) -> str:
        """데이터 기반 답변 생성"""
        if not self.llm:
            return "LLM이 로드되지 않았습니다."
        
        # 검색 결과 텍스트 조합
        context = "\n---\n".join([doc.page_content for doc in search_results])
        
        # Qwen2.5 프롬프트
        prompt = f"""<|im_start|>system
트래픽 데이터를 정확히 분석하여 답변합니다. 숫자는 정확히 인용합니다.<|im_end|>
<|im_start|>user
데이터:
{context}

질문: {query}<|im_end|>
<|im_start|>assistant
"""
        
        # 답변 생성
        response = self.llm(
            prompt,
            max_tokens=512,
            temperature=0.2,
            stop=["<|im_end|>"]
        )
        
        return response['choices'][0]['text']

class SimpleUI(ctk.CTk):
    """간단한 UI"""
    
    def __init__(self):
        super().__init__()
        
        self.title("CSV 데이터 검색 시스템")
        self.geometry("1000x700")
        
        ctk.set_appearance_mode("dark")
        
        # 검색 시스템
        self.search_system = SimpleCSVSearch()
        
        # UI 구성
        self.setup_ui()
        
    def setup_ui(self):
        """UI 구성"""
        # 상단 버튼들
        top_frame = ctk.CTkFrame(self)
        top_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkButton(
            top_frame,
            text="1. 임베딩 로드",
            command=self.load_embeddings,
            width=150
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            top_frame,
            text="2. 벡터DB 구축",
            command=self.build_vector_db,
            width=150
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            top_frame,
            text="3. GGUF 로드",
            command=self.load_gguf,
            width=150
        ).pack(side="left", padx=5)
        
        self.status_label = ctk.CTkLabel(top_frame, text="준비 안됨")
        self.status_label.pack(side="left", padx=20)
        
        # 검색 입력
        search_frame = ctk.CTkFrame(self)
        search_frame.pack(fill="x", padx=10, pady=10)
        
        self.search_input = ctk.CTkEntry(
            search_frame,
            placeholder_text="검색어 입력 (예: TOTALCNT 1450 이상)",
            font=("Arial", 14)
        )
        self.search_input.pack(side="left", fill="x", expand=True, padx=(0,10))
        
        ctk.CTkButton(
            search_frame,
            text="검색",
            command=self.search,
            width=100
        ).pack(side="left")
        
        # 검색 결과
        result_frame = ctk.CTkFrame(self)
        result_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(result_frame, text="검색 결과:").pack(anchor="w", padx=10, pady=5)
        
        self.result_text = ctk.CTkTextbox(
            result_frame,
            font=("Consolas", 12)
        )
        self.result_text.pack(fill="both", expand=True, padx=10, pady=5)
        
        # 예제 버튼들
        example_frame = ctk.CTkFrame(self)
        example_frame.pack(fill="x", padx=10, pady=10)
        
        examples = [
            "TOTALCNT 최대값은?",
            "15시 데이터 보여줘",
            "M14AM10A 평균은?",
            "피크 시간대는?"
        ]
        
        for ex in examples:
            ctk.CTkButton(
                example_frame,
                text=ex,
                command=lambda e=ex: self.search_input.insert(0, e),
                width=150
            ).pack(side="left", padx=5)
            
    def load_embeddings(self):
        """임베딩 로드"""
        try:
            self.search_system.load_embeddings()
            self.status_label.configure(text="✅ 임베딩 로드됨")
            messagebox.showinfo("성공", "임베딩 모델 로드 완료")
        except Exception as e:
            messagebox.showerror("오류", str(e))
            
    def build_vector_db(self):
        """벡터DB 구축"""
        try:
            if not self.search_system.embeddings:
                messagebox.showwarning("경고", "먼저 임베딩을 로드하세요")
                return
                
            # 기존 DB 확인
            if self.search_system.load_vector_db():
                if messagebox.askyesno("확인", "기존 벡터DB가 있습니다. 재구축하시겠습니까?"):
                    self.search_system.build_vector_db()
                    messagebox.showinfo("성공", "벡터DB 재구축 완료")
            else:
                self.search_system.build_vector_db()
                messagebox.showinfo("성공", "벡터DB 구축 완료")
                
            self.status_label.configure(text="✅ 임베딩+벡터DB 준비")
        except Exception as e:
            messagebox.showerror("오류", str(e))
            
    def load_gguf(self):
        """GGUF 모델 로드"""
        filepath = filedialog.askopenfilename(
            title="GGUF 모델 선택",
            filetypes=[("GGUF Files", "*.gguf")]
        )
        
        if filepath:
            try:
                self.search_system.load_gguf_model(filepath)
                self.status_label.configure(text="✅ 모든 시스템 준비 완료")
                messagebox.showinfo("성공", "GGUF 모델 로드 완료")
            except Exception as e:
                messagebox.showerror("오류", str(e))
                
    def search(self):
        """검색 실행"""
        query = self.search_input.get()
        if not query:
            return
            
        try:
            # 벡터 검색
            results = self.search_system.search(query, k=3)
            
            # 결과 표시
            self.result_text.delete("1.0", "end")
            self.result_text.insert("1.0", f"🔍 검색어: {query}\n")
            self.result_text.insert("end", "="*50 + "\n\n")
            
            # 검색 결과 표시
            for i, doc in enumerate(results, 1):
                self.result_text.insert("end", f"[결과 {i}]\n")
                self.result_text.insert("end", doc.page_content[:500] + "...\n")
                self.result_text.insert("end", "-"*30 + "\n")
            
            # LLM 답변 (옵션)
            if self.search_system.llm:
                self.result_text.insert("end", "\n💡 AI 답변:\n")
                answer = self.search_system.answer_with_data(query, results)
                self.result_text.insert("end", answer)
                
        except Exception as e:
            messagebox.showerror("오류", str(e))

if __name__ == "__main__":
    app = SimpleUI()
    app.mainloop()