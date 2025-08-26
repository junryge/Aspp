# -*- coding: utf-8 -*-
"""
폐쇄망 GGUF + CSV 검색 시스템 (대용량 데이터 고속 처리 버전)
79만개 데이터 최적화
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import logging
import time
from tqdm import tqdm

# LangChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

# GGUF
from llama_cpp import Llama

# UI
import customtkinter as ctk
from tkinter import messagebox, filedialog, ttk
import tkinter as tk

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FastCSVSearch:
    """대용량 CSV 고속 검색 시스템"""
    
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
        self.summary_data = {}  # 요약 데이터 캐시
        
    def load_embeddings(self):
        """임베딩 모델 로드"""
        logger.info("임베딩 모델 로드 중...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_path,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'batch_size': 32}  # 배치 처리
        )
        logger.info("✅ 임베딩 모델 로드 완료")
        
    def load_all_csv_data(self):
        """모든 CSV 파일을 빠르게 로드"""
        csv_files = sorted(Path(self.csv_dir).glob("*.csv"))
        logger.info(f"📁 CSV 파일 {len(csv_files)}개 발견")
        
        all_dfs = []
        for csv_file in tqdm(csv_files, desc="CSV 로딩"):
            # 필요한 컬럼만 로드 (메모리 절약)
            df = pd.read_csv(csv_file, usecols=[
                'CURRTIME', 'TOTALCNT', 
                'M14AM10ASUM', 'M14AM14BSUM', 'M14AM16SUM'
            ])
            df['FILE_DATE'] = csv_file.stem
            all_dfs.append(df)
        
        self.all_data = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"✅ 총 {len(self.all_data):,}개 행 로드 완료")
        
    def create_summary_statistics(self):
        """고속 요약 통계 생성 (벡터화 없이)"""
        logger.info("📊 요약 통계 생성 중...")
        
        for date in self.all_data['FILE_DATE'].unique():
            daily_data = self.all_data[self.all_data['FILE_DATE'] == date]
            
            # 일별 통계 저장
            self.summary_data[date] = {
                'count': len(daily_data),
                'totalcnt_mean': daily_data['TOTALCNT'].mean(),
                'totalcnt_max': daily_data['TOTALCNT'].max(),
                'totalcnt_min': daily_data['TOTALCNT'].min(),
                'peak_time': daily_data.loc[daily_data['TOTALCNT'].idxmax(), 'CURRTIME'],
                'data': daily_data  # 원본 데이터 참조
            }
        
        logger.info(f"✅ {len(self.summary_data)}개 날짜 요약 완료")
    
    def create_search_documents_fast(self) -> List[Document]:
        """초고속 문서 생성 (요약만 벡터화)"""
        documents = []
        
        logger.info("🚀 고속 문서 생성 시작")
        
        # 1. 일별 요약만 벡터화 (79만개 → 약 300개)
        for date, stats in tqdm(self.summary_data.items(), desc="문서 생성"):
            # 일별 요약
            daily_text = f"""
날짜: {date}
데이터 수: {stats['count']:,}개
TOTALCNT 평균: {stats['totalcnt_mean']:.0f}
TOTALCNT 최대: {stats['totalcnt_max']} (시간: {stats['peak_time']})
TOTALCNT 최소: {stats['totalcnt_min']}
"""
            documents.append(Document(
                page_content=daily_text,
                metadata={"date": date, "type": "daily"}
            ))
            
            # 시간대별 요약 (선택적)
            if stats['count'] > 100:  # 데이터가 많은 날만
                hourly = stats['data'].copy()
                hourly['hour'] = hourly['CURRTIME'].astype(str).str[8:10]
                
                for hour in hourly['hour'].unique():
                    hour_data = hourly[hourly['hour'] == hour]
                    if len(hour_data) > 5:  # 5개 이상 데이터가 있는 시간만
                        hour_text = f"""
날짜: {date}
시간대: {hour}시
TOTALCNT 평균: {hour_data['TOTALCNT'].mean():.0f}
TOTALCNT 최대: {hour_data['TOTALCNT'].max()}
데이터 수: {len(hour_data)}개
"""
                        documents.append(Document(
                            page_content=hour_text,
                            metadata={"date": date, "hour": hour, "type": "hourly"}
                        ))
        
        logger.info(f"✅ {len(documents)}개 문서 생성 완료 (원본 대비 {len(documents)/len(self.all_data)*100:.2f}%)")
        return documents
    
    def build_vector_db_fast(self):
        """고속 벡터 DB 구축"""
        start_time = time.time()
        
        # 1. CSV 로드
        self.load_all_csv_data()
        
        # 2. 요약 통계 생성
        self.create_summary_statistics()
        
        # 3. 문서 생성 (요약만)
        documents = self.create_search_documents_fast()
        
        # 4. FAISS 벡터 스토어 생성
        logger.info("🔨 벡터 DB 구축 중...")
        
        # 배치 처리로 빠르게
        batch_size = 100
        if len(documents) > batch_size:
            # 첫 배치로 초기화
            self.vector_store = FAISS.from_documents(
                documents[:batch_size], 
                self.embeddings
            )
            # 나머지 배치 추가
            for i in tqdm(range(batch_size, len(documents), batch_size), desc="벡터화"):
                batch = documents[i:i+batch_size]
                self.vector_store.add_documents(batch)
        else:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        
        # 5. 저장
        Path("./vector_db").mkdir(exist_ok=True)
        self.vector_store.save_local(self.vector_db_path)
        
        elapsed = time.time() - start_time
        logger.info(f"✅ 벡터 DB 구축 완료 ({elapsed:.1f}초 소요)")
        
    def search_direct_data(self, query: str) -> str:
        """직접 데이터 검색 (벡터 검색 + 실제 데이터)"""
        # 벡터 검색으로 관련 날짜/시간 찾기
        if self.vector_store:
            docs = self.vector_store.similarity_search(query, k=3)
            relevant_dates = [doc.metadata.get('date') for doc in docs]
            
            # 해당 날짜의 실제 데이터에서 검색
            results = []
            for date in relevant_dates:
                if date in self.summary_data:
                    data = self.summary_data[date]['data']
                    
                    # 쿼리에 따른 필터링
                    if "최대" in query or "max" in query.lower():
                        max_row = data.loc[data['TOTALCNT'].idxmax()]
                        results.append(f"[{date}] 최대값: {max_row['TOTALCNT']} (시간: {max_row['CURRTIME']})")
                    
                    elif "1450" in query or "1500" in query:
                        threshold = 1450 if "1450" in query else 1500
                        filtered = data[data['TOTALCNT'] >= threshold]
                        if not filtered.empty:
                            results.append(f"[{date}] {threshold} 이상: {len(filtered)}개 시간대")
                            results.append(f"  시간: {filtered['CURRTIME'].head(5).tolist()}")
                    
                    elif "평균" in query:
                        avg = data['TOTALCNT'].mean()
                        results.append(f"[{date}] 평균: {avg:.0f}")
            
            return "\n".join(results) if results else "관련 데이터를 찾을 수 없습니다."
        
        return "벡터 DB가 준비되지 않았습니다."
    
    def load_vector_db(self):
        """기존 벡터 DB 로드"""
        if Path(self.vector_db_path).exists():
            self.vector_store = FAISS.load_local(
                self.vector_db_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("✅ 기존 벡터 DB 로드 완료")
            
            # 요약 데이터도 로드 (있으면)
            summary_path = Path("./vector_db/summary_cache.pkl")
            if summary_path.exists():
                import pickle
                with open(summary_path, 'rb') as f:
                    self.summary_data = pickle.load(f)
                logger.info("✅ 요약 캐시 로드 완료")
            else:
                # 요약 데이터 재생성
                self.load_all_csv_data()
                self.create_summary_statistics()
                # 캐시 저장
                with open(summary_path, 'wb') as f:
                    pickle.dump(self.summary_data, f)
            
            return True
        return False
        
    def load_gguf_model(self, model_path: str):
        """GGUF 모델 로드"""
        logger.info(f"🤖 GGUF 모델 로드 중: {model_path}")
        
        self.llm = Llama(
            model_path=model_path,
            n_ctx=8192,
            n_gpu_layers=35,  # Q6_K용
            n_threads=8,
            verbose=False
        )
        
        logger.info("✅ GGUF 모델 로드 완료")

class FastUI(ctk.CTk):
    """고속 처리 UI"""
    
    def __init__(self):
        super().__init__()
        
        self.title("대용량 CSV 고속 검색 시스템")
        self.geometry("1000x700")
        
        ctk.set_appearance_mode("dark")
        
        # 검색 시스템
        self.search_system = FastCSVSearch()
        
        # UI 구성
        self.setup_ui()
        
    def setup_ui(self):
        """UI 구성"""
        # 상단 정보
        info_frame = ctk.CTkFrame(self)
        info_frame.pack(fill="x", padx=10, pady=10)
        
        self.info_label = ctk.CTkLabel(
            info_frame,
            text="💡 79만개 데이터를 요약하여 빠르게 검색합니다",
            font=("Arial", 14)
        )
        self.info_label.pack()
        
        # 초기화 버튼들
        btn_frame = ctk.CTkFrame(self)
        btn_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkButton(
            btn_frame,
            text="1. 임베딩 로드",
            command=self.load_embeddings,
            width=150
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            btn_frame,
            text="2. 고속 DB 구축",
            command=self.build_vector_db,
            width=150,
            fg_color="green"
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            btn_frame,
            text="3. GGUF 로드 (선택)",
            command=self.load_gguf,
            width=150
        ).pack(side="left", padx=5)
        
        self.status_label = ctk.CTkLabel(btn_frame, text="⏳ 준비 안됨")
        self.status_label.pack(side="left", padx=20)
        
        # 진행률 표시
        self.progress = ttk.Progressbar(
            btn_frame,
            mode='indeterminate',
            length=200
        )
        self.progress.pack(side="left", padx=10)
        
        # 검색 입력
        search_frame = ctk.CTkFrame(self)
        search_frame.pack(fill="x", padx=10, pady=10)
        
        self.search_input = ctk.CTkEntry(
            search_frame,
            placeholder_text="검색어 입력 (예: TOTALCNT 1450 이상, 15시 데이터)",
            font=("Arial", 14)
        )
        self.search_input.pack(side="left", fill="x", expand=True, padx=(0,10))
        
        ctk.CTkButton(
            search_frame,
            text="빠른 검색",
            command=self.quick_search,
            width=100,
            fg_color="orange"
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            search_frame,
            text="AI 검색",
            command=self.ai_search,
            width=100
        ).pack(side="left")
        
        # 검색 결과
        result_frame = ctk.CTkFrame(self)
        result_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(result_frame, text="🔍 검색 결과:").pack(anchor="w", padx=10, pady=5)
        
        self.result_text = ctk.CTkTextbox(
            result_frame,
            font=("Consolas", 12)
        )
        self.result_text.pack(fill="both", expand=True, padx=10, pady=5)
        
        # 예제 버튼들
        example_frame = ctk.CTkFrame(self)
        example_frame.pack(fill="x", padx=10, pady=10)
        
        examples = [
            "TOTALCNT 1450 이상",
            "최대값 시간대",
            "8월 7일 평균",
            "15시 데이터"
        ]
        
        for ex in examples:
            ctk.CTkButton(
                example_frame,
                text=ex,
                command=lambda e=ex: self.search_input.insert(0, e),
                width=120,
                height=30
            ).pack(side="left", padx=5)
            
    def load_embeddings(self):
        """임베딩 로드"""
        try:
            self.progress.start()
            self.search_system.load_embeddings()
            self.status_label.configure(text="✅ 임베딩 준비")
            messagebox.showinfo("성공", "임베딩 모델 로드 완료")
        except Exception as e:
            messagebox.showerror("오류", str(e))
        finally:
            self.progress.stop()
            
    def build_vector_db(self):
        """고속 벡터DB 구축"""
        try:
            if not self.search_system.embeddings:
                messagebox.showwarning("경고", "먼저 임베딩을 로드하세요")
                return
            
            # 기존 DB 확인
            if self.search_system.load_vector_db():
                response = messagebox.askyesnocancel(
                    "확인", 
                    "기존 DB가 있습니다.\n"
                    "Yes: 기존 DB 사용\n"
                    "No: 새로 구축 (10-30분 소요)\n"
                    "Cancel: 취소"
                )
                if response is True:  # Yes - 기존 사용
                    self.status_label.configure(text="✅ 검색 준비 완료")
                    return
                elif response is False:  # No - 재구축
                    self.progress.start()
                    self.info_label.configure(text="⏳ DB 구축 중... (10-30분 소요)")
                    self.update()
                    
                    self.search_system.build_vector_db_fast()
                    
                    self.info_label.configure(text="✅ 고속 DB 구축 완료!")
                    messagebox.showinfo("성공", "벡터DB 재구축 완료")
            else:
                # 새로 구축
                self.progress.start()
                self.info_label.configure(text="⏳ DB 구축 중... (첫 실행 10-30분 소요)")
                self.update()
                
                self.search_system.build_vector_db_fast()
                
                self.info_label.configure(text="✅ 고속 DB 구축 완료!")
                messagebox.showinfo("성공", "벡터DB 구축 완료")
                
            self.status_label.configure(text="✅ 검색 준비 완료")
            
        except Exception as e:
            messagebox.showerror("오류", str(e))
        finally:
            self.progress.stop()
            
    def load_gguf(self):
        """GGUF 모델 로드 (선택사항)"""
        filepath = filedialog.askopenfilename(
            title="GGUF 모델 선택",
            filetypes=[("GGUF Files", "*.gguf")]
        )
        
        if filepath:
            try:
                self.progress.start()
                self.search_system.load_gguf_model(filepath)
                self.status_label.configure(text="✅ 전체 시스템 준비")
                messagebox.showinfo("성공", "GGUF 모델 로드 완료")
            except Exception as e:
                messagebox.showerror("오류", str(e))
            finally:
                self.progress.stop()
                
    def quick_search(self):
        """빠른 검색 (벡터 + 직접 데이터)"""
        query = self.search_input.get()
        if not query:
            return
            
        try:
            self.result_text.delete("1.0", "end")
            self.result_text.insert("1.0", f"🔍 빠른 검색: {query}\n")
            self.result_text.insert("end", "="*50 + "\n\n")
            
            # 직접 데이터 검색
            result = self.search_system.search_direct_data(query)
            self.result_text.insert("end", result)
            
        except Exception as e:
            messagebox.showerror("오류", str(e))
            
    def ai_search(self):
        """AI 검색 (LLM 사용)"""
        if not self.search_system.llm:
            messagebox.showwarning("경고", "GGUF 모델을 먼저 로드하세요")
            return
            
        # 일반 검색과 동일하게 처리
        self.quick_search()

if __name__ == "__main__":
    app = FastUI()
    app.mainloop()