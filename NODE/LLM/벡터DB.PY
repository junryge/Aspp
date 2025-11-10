#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1: TEST.CSV로 벡터DB 생성
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
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDBCreator:
    """TEST.CSV로 벡터DB 생성"""
    
    def __init__(self, csv_path, db_path="./chroma_db", 
                 embedding_model_path="./embeddings/all-MiniLM-L6-v2"):
        self.csv_path = csv_path
        self.db_path = db_path
        self.embedding_model_path = embedding_model_path
        
        os.makedirs(self.db_path, exist_ok=True)
    
    def load_csv(self):
        """CSV 로드"""
        logger.info(f"CSV 로드: {self.csv_path}")
        df = pd.read_csv(self.csv_path, encoding='utf-8')
        
        # 필요한 컬럼만 선택
        self.df = df[['시퀀스시작', '현재시간', '현재TOTALCNT', '예측시점', 
                      '실제시점', '실제값', '보정예측', '오차', 
                      'M14AM14B', 'M14BM14A', 'queue_gap', 'TRANSPORT']].copy()
        
        self.df['현재시간'] = pd.to_datetime(self.df['현재시간'])
        self.df['실제시점'] = pd.to_datetime(self.df['실제시점'])
        
        logger.info(f"데이터 로드 완료: {len(self.df)}개 레코드")
        return self.df
    
    def create_documents(self):
        """문서 생성"""
        logger.info("문서 생성 중...")
        documents = []
        
        for idx, row in self.df.iterrows():
            # 각 행을 문서로 변환
            doc_text = f"""
시간: {row['현재시간']}
현재TOTALCNT: {row['현재TOTALCNT']}
실제값: {row['실제값']:.1f}
예측값: {row['보정예측']:.1f}
오차: {row['오차']:.1f}
M14AM14B: {row['M14AM14B']}
M14BM14A: {row['M14BM14A']}
queue_gap: {row['queue_gap']}
TRANSPORT: {row['TRANSPORT']}
            """
            
            documents.append({
                'content': doc_text.strip(),
                'metadata': {
                    '시간': str(row['현재시간']),
                    '실제값': float(row['실제값']),
                    '예측값': float(row['보정예측']),
                    '오차': float(row['오차'])
                }
            })
        
        logger.info(f"문서 생성 완료: {len(documents)}개")
        return documents
    
    def create_embeddings(self, documents):
        """임베딩 생성"""
        logger.info("임베딩 모델 로드...")
        
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except:
            device = 'cpu'
        
        logger.info(f"디바이스: {device}")
        
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(self.embedding_model_path)
        if device == 'cuda':
            model = model.to('cuda')
        
        logger.info("임베딩 생성 중...")
        texts = [doc['content'] for doc in documents]
        embeddings = model.encode(texts, show_progress_bar=True)
        
        logger.info(f"임베딩 생성 완료: {len(embeddings)}개")
        return embeddings
    
    def save_vectordb(self, documents, embeddings):
        """벡터DB 저장"""
        db_file = os.path.join(self.db_path, 'vectordb.pkl')
        
        logger.info(f"벡터DB 저장: {db_file}")
        
        with open(db_file, 'wb') as f:
            pickle.dump({
                'documents': documents,
                'embeddings': embeddings
            }, f)
        
        logger.info("저장 완료!")
    
    def run(self):
        """전체 프로세스 실행"""
        self.load_csv()
        documents = self.create_documents()
        embeddings = self.create_embeddings(documents)
        self.save_vectordb(documents, embeddings)

def main():
    """메인 실행"""
    CSV_PATH = "./TEST.CSV"
    DB_PATH = "./chroma_db"
    
    print("="*60)
    print("TEST.CSV 벡터DB 생성 시작")
    print("="*60)
    
    creator = VectorDBCreator(
        csv_path=CSV_PATH,
        db_path=DB_PATH
    )
    
    creator.run()
    
    print("="*60)
    print("✅ 벡터DB 생성 완료!")
    print("이제 RAG2_TEST.py를 실행하세요.")
    print("="*60)

if __name__ == "__main__":
    main()