#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2: TEST.CSV용 RAG 시스템 실행
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
    """TEST.CSV용 RAG 시스템"""
    
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
        
        # 1. CSV 데이터 로드
        self._load_csv_data()
        
        # 2. 벡터DB 로드
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
                     '보정예측', '오차', 'M14AM14B', 'M14BM14A', 
                     'queue_gap', 'TRANSPORT']].copy()
        
        self.df['현재시간'] = pd.to_datetime(self.df['현재시간'])
        self.df['실제시점'] = pd.to_datetime(self.df['실제시점'])
        self.df.set_index('현재시간', inplace=True)
        
        logger.info(f"데이터 로드 완료: {len(self.df)}개")
    
    def _load_vectorstore(self):
        """벡터DB 로드"""
        db_file = os.path.join(self.db_persist_path, 'vectordb.pkl')
        
        if not os.path.exists(db_file):
            raise FileNotFoundError(
                f"벡터DB가 없습니다. 먼저 벡터DB_TEST.py를 실행하세요.\n"
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
            n_ctx=4096,  # 컨텍스트 늘림
            n_batch=512,  # 배치 늘림
            n_gpu_layers=-1,  # 전체 GPU
            n_threads=4,
            temperature=0.7,
            max_tokens=512,  # 크게 늘림
            verbose=True
        )
        
        logger.info("LLM 로드 완료")
    
    def _cosine_similarity(self, a, b):
        """코사인 유사도 계산"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def _search_similar(self, query, k=5):
        """유사 문서 검색"""
        query_embedding = self.embedding_model.encode([query])[0]
        
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            sim = self._cosine_similarity(query_embedding, doc_embedding)
            similarities.append((i, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:k]
        
        results = []
        for idx, score in top_k:
            results.append(self.documents[idx]['content'])
        
        return results
    
    def _call_llm(self, prompt, max_tokens=300):
        """LLM 호출"""
        print(f"💬 토큰 생성 시작... (최대 {max_tokens}개)")
        response = self.llm(
            prompt, 
            max_tokens=max_tokens, 
            temperature=0.7,  # 더 창의적으로
            top_p=0.95,
            repeat_penalty=1.15,
            stop=["질문:", "###"]  # stop 토큰 최소화
        )
        result = response['choices'][0]['text'].strip()
        print(f"✅ 생성 완료 ({len(result)}자)")
        return result
    
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
- M14AM14B: {data['M14AM14B']}
- M14BM14A: {data['M14BM14A']}
- queue_gap: {data['queue_gap']}
- TRANSPORT: {data['TRANSPORT']}
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
- M14AM14B: {data['M14AM14B']}
- M14BM14A: {data['M14BM14A']}
                """
        except Exception as e:
            return f"오류: {e}"
    
    def analyze_query(self, query):
        """쿼리 분석 및 처리"""
        # 시간 패턴 추출
        time_pattern = r'(\d{4}-\d{2}-\d{2}\s+\d{1,2}:\d{2})'
        time_match = re.search(time_pattern, query)
        
        # 실제값/예측값 조회
        if time_match and ("실제값" in query or "예측값" in query or "데이터" in query):
            return self.get_data_at_time(time_match.group(1))
        
        # 일반 RAG 쿼리
        else:
            print("🔍 유사 문서 검색 중...")
            similar_docs = self._search_similar(query, k=5)  # 더 많은 문서 검색
            context = "\n---\n".join(similar_docs)
            
            print("🤖 AI 답변 생성 중...")
            # 개선된 프롬프트
            prompt = f"""다음은 반도체 제조 데이터입니다.

데이터:
{context}

질문: {query}

위 데이터를 바탕으로 질문에 대해 구체적으로 답변해주세요. 시간, 값, 특징을 포함해서 2-3문장으로 설명하세요.

답변:"""
            
            result = self._call_llm(prompt, max_tokens=300)
            
            return result

def main():
    """메인 실행"""
    CSV_PATH = "./TEST.CSV"
    MODEL_PATH = "./models/Qwen3-Coder-30B-A3B-Instruct-Q3_K_M.gguf"
    DB_PATH = "./chroma_db"
    
    print("="*60)
    print("TEST.CSV RAG 시스템 시작")
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