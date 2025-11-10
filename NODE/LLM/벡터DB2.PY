#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1: 벡터DB 생성 (ChromaDB 없이 pickle 사용)
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
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDBCreator:
    """벡터DB 생성 전용 클래스 (pickle 기반)"""
    
    def __init__(self, csv_path, embedding_model_path, db_persist_path):
        self.csv_path = csv_path
        self.embedding_model_path = embedding_model_path
        self.db_persist_path = db_persist_path
        
    def create_vectordb(self):
        """벡터DB 생성 메인 프로세스"""
        logger.info("="*60)
        logger.info("벡터DB 생성 시작")
        logger.info("="*60)
        
        # 1. CSV 로드
        logger.info(f"CSV 로드 중: {self.csv_path}")
        df = pd.read_csv(self.csv_path, encoding='utf-8')
        
        df_filtered = df[['현재시간', '실제시점', '실제값', 
                         '보정예측', '오차']].copy()
        
        df_filtered['현재시간'] = pd.to_datetime(df_filtered['현재시간'])
        df_filtered['실제시점'] = pd.to_datetime(df_filtered['실제시점'])
        df_filtered.set_index('현재시간', inplace=True)
        
        logger.info(f"데이터 로드 완료: {len(df_filtered)}개 행")
        
        # 2. 문서 생성
        logger.info("문서 생성 중...")
        documents = self._create_documents(df_filtered)
        logger.info(f"총 {len(documents)}개 문서 생성")
        
        # 3. 임베딩 모델 로드
        logger.info(f"임베딩 모델 로드: {self.embedding_model_path}")
        
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
        
        # 4. 임베딩 생성
        logger.info("임베딩 생성 중... (시간이 걸립니다)")
        texts = [doc['content'] for doc in documents]
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
        
        # 5. 저장
        logger.info("벡터DB 저장 중...")
        os.makedirs(self.db_persist_path, exist_ok=True)
        
        db_data = {
            'documents': documents,
            'embeddings': embeddings,
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'data_count': len(df_filtered),
                'start_date': str(df_filtered.index[0]),
                'end_date': str(df_filtered.index[-1])
            }
        }
        
        with open(os.path.join(self.db_persist_path, 'vectordb.pkl'), 'wb') as f:
            pickle.dump(db_data, f)
        
        logger.info(f"벡터DB 저장 완료: {self.db_persist_path}")
        logger.info("="*60)
        logger.info("벡터DB 생성 완료!")
        logger.info("="*60)
        
    def _create_documents(self, df):
        """문서 생성"""
        documents = []
        
        # 개별 데이터 포인트
        for idx, row in df.iterrows():
            doc = {
                'content': f"""
시간: {idx}
실제값: {row['실제값']:.1f}
타겟시간: {row['실제시점']}
예측값: {row['보정예측']:.1f}
오차: {row['오차']:.1f}
                """.strip(),
                'metadata': {
                    'datetime': str(idx),
                    'type': 'datapoint',
                    'value': float(row['실제값'])
                }
            }
            documents.append(doc)
        
        # 280개 시퀀스 문서
        sequence_size = 280
        for i in range(0, max(1, len(df) - sequence_size + 1), 10):
            sequence = df.iloc[i:min(i+sequence_size, len(df))]
            
            if len(sequence) < 100:
                continue
                
            seq_doc = {
                'content': f"""
시퀀스 기간: {sequence.index[0]} ~ {sequence.index[-1]}
데이터 수: {len(sequence)}개
평균: {sequence['실제값'].mean():.2f}
최대: {sequence['실제값'].max():.1f}
최소: {sequence['실제값'].min():.1f}
표준편차: {sequence['실제값'].std():.2f}
평균오차: {sequence['오차'].mean():.2f}
                """.strip(),
                'metadata': {
                    'type': 'sequence',
                    'start_time': str(sequence.index[0]),
                    'end_time': str(sequence.index[-1])
                }
            }
            documents.append(seq_doc)
            
        return documents

if __name__ == "__main__":
    # 설정
    CSV_PATH = "./WITH.CSV"
    EMBEDDING_PATH = "./embeddings/all-MiniLM-L6-v2"
    DB_PATH = "./chroma_db"
    
    # 벡터DB 생성
    creator = VectorDBCreator(CSV_PATH, EMBEDDING_PATH, DB_PATH)
    creator.create_vectordb()