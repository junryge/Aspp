#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2: RAG 시스템 실행 (벡터DB 로드하여 사용)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import re
import json
import logging
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeSeriesRAG:
    """벡터DB를 로드하여 사용하는 RAG 시스템"""
    
    def __init__(self, 
                 csv_path,
                 model_path,
                 db_persist_path="./chroma_db",
                 embedding_model_path="./embeddings/all-MiniLM-L6-v2"):
        
        self.csv_path = csv_path
        self.model_path = model_path
        self.db_persist_path = db_persist_path
        self.embedding_model_path = embedding_model_path
        
        # 초기화
        self._initialize()
    
    def _initialize(self):
        """시스템 초기화"""
        logger.info("RAG 시스템 초기화 중...")
        
        # 1. CSV 데이터 로드 (쿼리용)
        self._load_csv_data()
        
        # 2. 기존 벡터DB 로드
        self._load_vectorstore()
        
        # 3. LLM 설정
        self._setup_llm()
        
        # 4. RAG 체인 설정
        self._setup_rag_chain()
        
        logger.info("초기화 완료!")
    
    def _load_csv_data(self):
        """CSV 데이터 로드"""
        logger.info(f"CSV 로드: {self.csv_path}")
        
        df = pd.read_csv(self.csv_path)
        self.df = df[['날짜', '타켓날짜', '실제값', 
                     'ExtraTrees_예측', 'ExtraTrees_오차']].copy()
        
        self.df['날짜'] = pd.to_datetime(self.df['날짜'])
        self.df['타켓날짜'] = pd.to_datetime(self.df['타켓날짜'])
        self.df.set_index('날짜', inplace=True)
        
        logger.info(f"데이터 로드 완료: {len(self.df)}개")
    
    def _load_vectorstore(self):
        """기존 벡터DB 로드"""
        if not os.path.exists(self.db_persist_path):
            raise FileNotFoundError(
                f"벡터DB가 없습니다. 먼저 step1_create_vectordb.py를 실행하세요.\n"
                f"경로: {self.db_persist_path}"
            )
        
        logger.info(f"벡터DB 로드 중: {self.db_persist_path}")
        
        # 임베딩 모델
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_path,
            model_kwargs={'device': 'cuda'}
        )
        
        # ChromaDB 로드
        self.vector_store = Chroma(
            persist_directory=self.db_persist_path,
            embedding_function=embeddings,
            collection_name="timeseries_data"
        )
        
        logger.info("벡터DB 로드 완료")
    
    def _setup_llm(self):
        """LLM 설정"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"모델 파일 없음: {self.model_path}")
        
        logger.info(f"LLM 로드: {self.model_path}")
        
        # 콜백 설정 (스트리밍 출력)
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        
        self.llm = LlamaCpp(
            model_path=self.model_path,
            n_ctx=4096,
            n_batch=512,
            n_gpu_layers=35,  # GPU 사용
            temperature=0.3,
            max_tokens=1000,
            callback_manager=callback_manager,
            verbose=False
        )
        
        logger.info("LLM 로드 완료")
    
    def _setup_rag_chain(self):
        """RAG 체인 설정"""
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 5}
            ),
            return_source_documents=True
        )
    
    def get_data_at_time(self, datetime_str):
        """특정 시간 데이터 조회"""
        try:
            dt = pd.to_datetime(datetime_str)
            
            if dt in self.df.index:
                data = self.df.loc[dt]
                return f"""
📊 {dt} 데이터:
- 실제값: {data['실제값']:.1f}
- 예측값: {data['ExtraTrees_예측']:.1f}  
- 오차: {data['ExtraTrees_오차']:.1f}
                """
            else:
                nearest_idx = self.df.index.get_indexer([dt], method='nearest')[0]
                nearest_time = self.df.index[nearest_idx]
                data = self.df.iloc[nearest_idx]
                
                return f"""
📊 가장 가까운 시간: {nearest_time}
- 실제값: {data['실제값']:.1f}
- 예측값: {data['ExtraTrees_예측']:.1f}
- 오차: {data['ExtraTrees_오차']:.1f}
                """
        except Exception as e:
            return f"오류: {e}"
    
    def get_sequence_before(self, datetime_str, length=280):
        """시퀀스 데이터 조회"""
        try:
            dt = pd.to_datetime(datetime_str)
            end_idx = self.df.index.get_indexer([dt], method='nearest')[0]
            start_idx = max(0, end_idx - length + 1)
            
            sequence = self.df.iloc[start_idx:end_idx + 1]
            
            return f"""
📈 시퀀스 분석 ({len(sequence)}개):
기간: {sequence.index[0]} ~ {sequence.index[-1]}

실제값:
- 평균: {sequence['실제값'].mean():.2f}
- 최대: {sequence['실제값'].max():.1f}  
- 최소: {sequence['실제값'].min():.1f}
- 표준편차: {sequence['실제값'].std():.2f}

예측 성능:
- 평균 오차: {sequence['ExtraTrees_오차'].mean():.2f}
- RMSE: {np.sqrt((sequence['ExtraTrees_오차']**2).mean()):.2f}

최근 5개:
{sequence.tail(5)[['실제값', 'ExtraTrees_예측']].to_string()}
            """
        except Exception as e:
            return f"오류: {e}"
    
    def analyze_query(self, query):
        """쿼리 분석 및 처리"""
        # 시간 패턴 추출
        time_pattern = r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})'
        time_match = re.search(time_pattern, query)
        
        # 실제값/예측값 조회
        if time_match and ("실제값" in query or "예측값" in query):
            return self.get_data_at_time(time_match.group(1))
        
        # 시퀀스 분석
        elif time_match and ("시퀀스" in query or "280" in query or "설명" in query):
            sequence_info = self.get_sequence_before(time_match.group(1))
            
            # LLM 설명 추가
            prompt = f"{sequence_info}\n\n위 데이터를 바탕으로 예측 이유를 간단히 설명:"
            result = self.qa_chain({"query": prompt})
            
            return f"{sequence_info}\n\n💡 설명:\n{result['result']}"
        
        # 일반 RAG 쿼리
        else:
            result = self.qa_chain({"query": query})
            return result['result']

def main():
    """메인 실행"""
    # 설정
    CSV_PATH = "./CSV파일.csv"
    MODEL_PATH = "./models/Qwen3-Coder-30B-A3B-Instruct-Q3_K_M.gguf"
    DB_PATH = "./chroma_db"
    
    # RAG 시스템 시작
    print("="*60)
    print("TimeSeriesRAG 시스템 시작")
    print("="*60)
    
    rag = TimeSeriesRAG(
        csv_path=CSV_PATH,
        model_path=MODEL_PATH,
        db_persist_path=DB_PATH
    )
    
    # 대화 루프
    while True:
        query = input("\n질문 (종료: quit): ").strip()
        
        if query.lower() in ['quit', 'exit']:
            print("종료합니다.")
            break
        
        if query:
            print("\n처리 중...")
            answer = rag.analyze_query(query)
            print(f"\n{answer}")

if __name__ == "__main__":
    main()