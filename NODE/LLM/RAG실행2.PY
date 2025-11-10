#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2: RAG 시스템 실행 (pickle 벡터DB 사용)
"""

import os
os.environ['USE_TF'] = '0'
os.environ['USE_TORCH'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_NO_TF'] = '1'

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime
import re
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeSeriesRAG:
    """pickle 벡터DB를 로드하여 사용하는 RAG 시스템"""
    
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
        
        # 3. 임베딩 모델 로드
        self._load_embedding_model()
        
        # 4. LLM 설정
        self._setup_llm()
        
        logger.info("초기화 완료!")
    
    def _load_csv_data(self):
        """CSV 데이터 로드"""
        logger.info(f"CSV 로드: {self.csv_path}")
        
        df = pd.read_csv(self.csv_path, encoding='utf-8')
        self.df = df[['현재시간', '실제시점', '실제값', 
                     '보정예측', '오차']].copy()
        
        self.df['현재시간'] = pd.to_datetime(self.df['현재시간'])
        self.df['실제시점'] = pd.to_datetime(self.df['실제시점'])
        self.df.set_index('현재시간', inplace=True)
        
        logger.info(f"데이터 로드 완료: {len(self.df)}개")
    
    def _load_vectorstore(self):
        """기존 벡터DB 로드"""
        db_file = os.path.join(self.db_persist_path, 'vectordb.pkl')
        
        if not os.path.exists(db_file):
            raise FileNotFoundError(
                f"벡터DB가 없습니다. 먼저 벡터DB.py를 실행하세요.\n"
                f"경로: {db_file}"
            )
        
        logger.info(f"벡터DB 로드 중: {db_file}")
        
        with open(db_file, 'rb') as f:
            db_data = pickle.load(f)
        
        self.documents = db_data['documents']
        self.embeddings = db_data['embeddings']
        
        logger.info(f"벡터DB 로드 완료: {len(self.documents)}개 문서")
    
    def _load_embedding_model(self):
        """임베딩 모델 로드"""
        logger.info(f"임베딩 모델 로드: {self.embedding_model_path}")
        
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except:
            device = 'cpu'
        
        logger.info(f"디바이스: {device}")
        
        from sentence_transformers import SentenceTransformer
        self.embedding_model = SentenceTransformer(self.embedding_model_path)
        if device == 'cuda':
            self.embedding_model = self.embedding_model.to('cuda')
        
        logger.info("임베딩 모델 로드 완료")
    
    def _setup_llm(self):
        """LLM 설정"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"모델 파일 없음: {self.model_path}")
        
        logger.info(f"LLM 로드: {self.model_path}")
        
        from llama_cpp import Llama
        
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=4096,
            n_batch=512,
            n_gpu_layers=35,  # GPU 사용
            temperature=0.3,
            max_tokens=1000,
            verbose=True  # 진행상황 표시
        )
        
        logger.info("LLM 로드 완료")
    
    def _cosine_similarity(self, a, b):
        """코사인 유사도 계산"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def _search_similar(self, query, k=5):
        """유사 문서 검색"""
        # 쿼리 임베딩
        query_embedding = self.embedding_model.encode([query])[0]
        
        # 코사인 유사도 계산
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            sim = self._cosine_similarity(query_embedding, doc_embedding)
            similarities.append((i, sim))
        
        # 상위 k개 선택
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:k]
        
        results = []
        for idx, score in top_k:
            results.append(self.documents[idx]['content'])
        
        return results
    
    def _call_llm(self, prompt):
        """LLM 호출"""
        print("\n🤖 AI 답변 생성 중... (30B 모델이라 시간이 좀 걸립니다)")
        response = self.llm(prompt, max_tokens=1000, temperature=0.3)
        return response['choices'][0]['text']
    
    def get_data_at_time(self, datetime_str):
        """특정 시간 데이터 조회"""
        try:
            dt = pd.to_datetime(datetime_str)
            
            if dt in self.df.index:
                data = self.df.loc[dt]
                return f"""
📊 {dt} 데이터:
- 실제값: {data['실제값']:.1f}
- 예측값: {data['보정예측']:.1f}  
- 오차: {data['오차']:.1f}
                """
            else:
                nearest_idx = self.df.index.get_indexer([dt], method='nearest')[0]
                nearest_time = self.df.index[nearest_idx]
                data = self.df.iloc[nearest_idx]
                
                return f"""
📊 가장 가까운 시간: {nearest_time}
- 실제값: {data['실제값']:.1f}
- 예측값: {data['보정예측']:.1f}
- 오차: {data['오차']:.1f}
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
- 평균 오차: {sequence['오차'].mean():.2f}
- RMSE: {np.sqrt((sequence['오차']**2).mean()):.2f}

최근 5개:
{sequence.tail(5)[['실제값', '보정예측']].to_string()}
            """
        except Exception as e:
            return f"오류: {e}"
    
    def analyze_query(self, query):
        """쿼리 분석 및 처리"""
        # 시간 패턴 추출
        time_pattern = r'(\d{4}-\d{2}-\d{2}\s+\d{1,2}:\d{2})'
        time_match = re.search(time_pattern, query)
        
        # 실제값/예측값 조회
        if time_match and ("실제값" in query or "예측값" in query):
            return self.get_data_at_time(time_match.group(1))
        
        # 시퀀스 분석
        elif time_match and ("시퀀스" in query or "280" in query or "설명" in query):
            sequence_info = self.get_sequence_before(time_match.group(1))
            
            # 유사 문서 검색
            similar_docs = self._search_similar(query, k=5)
            context = "\n\n".join(similar_docs)
            
            # LLM 설명 추가
            prompt = f"{sequence_info}\n\n관련 데이터:\n{context}\n\n위 데이터를 바탕으로 예측 이유를 간단히 설명:"
            result = self._call_llm(prompt)
            
            return f"{sequence_info}\n\n💡 설명:\n{result}"
        
        # 일반 RAG 쿼리
        else:
            # 유사 문서 검색
            similar_docs = self._search_similar(query, k=5)
            context = "\n\n".join(similar_docs)
            
            # LLM 답변 생성
            prompt = f"질문: {query}\n\n관련 정보:\n{context}\n\n답변:"
            result = self._call_llm(prompt)
            
            return result

def main():
    """메인 실행"""
    # 설정
    CSV_PATH = "./WITH.CSV"
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