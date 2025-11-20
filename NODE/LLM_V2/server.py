#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV 데이터를 벡터DB(pickle)로 저장
"""

import os
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_vectordb(csv_path, output_path="./vector_db"):
    """CSV를 벡터DB로 변환"""
    
    logger.info(f"CSV 로드: {csv_path}")
    
    # CSV 읽기
    df = pd.read_csv(csv_path, encoding='utf-8')
    logger.info(f"데이터 로드 완료: {len(df)}행, {len(df.columns)}컬럼")
    
    # 컬럼 확인
    logger.info(f"컬럼: {list(df.columns[:5])}... (총 {len(df.columns)}개)")
    
    # 문서 생성
    documents = []
    
    for idx, row in df.iterrows():
        # 각 행을 텍스트로 변환
        stat_dt = row.get('STAT_DT', 'Unknown')
        
        # 주요 메트릭 추출
        text_parts = [f"시간: {stat_dt}"]
        
        # 모든 컬럼 값 추가 (STAT_DT 제외)
        for col in df.columns:
            if col != 'STAT_DT':
                value = row[col]
                if pd.notna(value):  # NaN이 아닌 경우만
                    text_parts.append(f"{col}: {value}")
        
        doc_text = ", ".join(text_parts)
        
        documents.append({
            'id': idx,
            'stat_dt': stat_dt,
            'content': doc_text,
            'metadata': row.to_dict()
        })
        
        if (idx + 1) % 100 == 0:
            logger.info(f"문서 생성 중... {idx + 1}/{len(df)}")
    
    logger.info(f"✅ 문서 생성 완료: {len(documents)}개")
    
    # 임베딩 모델 로드
    logger.info("임베딩 모델 로드 중...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # 로컬 모델 경로 또는 자동 다운로드
        model_path = "./embeddings/all-MiniLM-L6-v2"
        
        if os.path.exists(model_path):
            logger.info(f"로컬 모델 사용: {model_path}")
            model = SentenceTransformer(model_path)
        else:
            logger.info("온라인에서 모델 다운로드...")
            model = SentenceTransformer('all-MiniLM-L6-v2')
        
    except Exception as e:
        logger.error(f"임베딩 모델 로드 실패: {e}")
        logger.info("더미 임베딩으로 대체...")
        embeddings = [np.random.rand(384) for _ in documents]
        
        # 저장
        os.makedirs(output_path, exist_ok=True)
        db_file = os.path.join(output_path, 'vectordb.pkl')
        
        with open(db_file, 'wb') as f:
            pickle.dump({
                'documents': documents,
                'embeddings': embeddings
            }, f)
        
        logger.info(f"✅ 벡터DB 저장 완료 (더미): {db_file}")
        return
    
    # 임베딩 생성
    logger.info("임베딩 생성 중...")
    embeddings = []
    
    batch_size = 32
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        texts = [doc['content'] for doc in batch]
        batch_embeddings = model.encode(texts, show_progress_bar=False)
        embeddings.extend(batch_embeddings)
        
        if (i + batch_size) % 100 == 0:
            logger.info(f"임베딩 생성 중... {min(i+batch_size, len(documents))}/{len(documents)}")
    
    logger.info(f"✅ 임베딩 생성 완료: {len(embeddings)}개")
    
    # 벡터DB 저장
    os.makedirs(output_path, exist_ok=True)
    db_file = os.path.join(output_path, 'vectordb.pkl')
    
    logger.info(f"벡터DB 저장 중: {db_file}")
    
    with open(db_file, 'wb') as f:
        pickle.dump({
            'documents': documents,
            'embeddings': embeddings,
            'created_at': datetime.now().isoformat(),
            'total_docs': len(documents)
        }, f)
    
    logger.info(f"✅ 벡터DB 저장 완료!")
    logger.info(f"  - 파일: {db_file}")
    logger.info(f"  - 문서 수: {len(documents)}")
    logger.info(f"  - 파일 크기: {os.path.getsize(db_file) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    # CSV 파일 경로
    CSV_FILE = "data.csv"  # 여기에 실제 CSV 파일명 입력
    OUTPUT_PATH = "./vector_db"
    
    print("="*60)
    print("CSV → 벡터DB 변환")
    print("="*60)
    
    if not os.path.exists(CSV_FILE):
        logger.error(f"❌ CSV 파일 없음: {CSV_FILE}")
        logger.info("현재 폴더의 CSV 파일들:")
        for f in os.listdir("."):
            if f.endswith(".csv"):
                logger.info(f"  → {f}")
        exit(1)
    
    create_vectordb(CSV_FILE, OUTPUT_PATH)
    
    print("="*60)
    print("완료!")
    print("="*60)