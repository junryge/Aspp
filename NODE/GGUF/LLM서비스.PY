#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
벡터 검색만 하는 간단한 대화형 서비스
LLM 없이 유사도 검색 결과만 보여줌
"""

import os
import json
from typing import List, Dict

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

class SimpleSearchService:
    """벡터 검색만 하는 서비스"""
    
    def __init__(self, embedding_model_path: str, vector_store_dir: str = "./vector_stores"):
        # 임베딩 모델 로드
        print("임베딩 모델 로딩 중...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_path,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("임베딩 모델 로딩 완료!")
        
        # 벡터 저장소 로드
        self.vector_store_dir = vector_store_dir
        self.vector_store = self._load_all_vector_stores()
    
    def _load_all_vector_stores(self):
        """모든 배치의 벡터 저장소 로드 및 병합"""
        print("\n벡터 저장소 로딩 중...")
        
        # 메타데이터 읽기
        metadata_path = f"{self.vector_store_dir}/metadata.json"
        if not os.path.exists(metadata_path):
            print("메타데이터 파일이 없습니다. 먼저 prepare_vectors.py를 실행하세요.")
            return None
            
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        total_batches = metadata['total_batches']
        print(f"총 {total_batches}개 배치 발견")
        
        # 첫 번째 배치 로드
        merged_store = FAISS.load_local(
            f"{self.vector_store_dir}/batch_001", 
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print("배치 1 로드 완료")
        
        # 나머지 배치 병합
        for i in range(2, total_batches + 1):
            batch_path = f"{self.vector_store_dir}/batch_{i:03d}"
            if os.path.exists(batch_path):
                batch_store = FAISS.load_local(
                    batch_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                merged_store.merge_from(batch_store)
                print(f"배치 {i} 병합 완료")
        
        print("모든 벡터 저장소 로드 완료!")
        return merged_store
    
    def search(self, query: str, k: int = 10) -> List[Dict]:
        """유사도 검색"""
        if not self.vector_store:
            return []
        
        # 유사한 문서 검색
        docs = self.vector_store.similarity_search_with_score(query, k=k)
        
        results = []
        for doc, score in docs:
            results.append({
                "content": doc.page_content,
                "filename": doc.metadata.get("filename", "Unknown"),
                "currtime": doc.metadata.get("currtime", ""),
                "totalcnt": doc.metadata.get("totalcnt", 0),
                "score": score
            })
        
        return results

def main():
    # 설정
    MODEL_DIR = "./models"
    VECTOR_STORE_DIR = "./vector_stores"
    
    print("=== CSV 검색 서비스 (유사도 검색) ===")
    
    # 서비스 초기화
    service = SimpleSearchService(
        embedding_model_path=os.path.join(MODEL_DIR, "paraphrase-multilingual-MiniLM-L12-v2"),
        vector_store_dir=VECTOR_STORE_DIR
    )
    
    if not service.vector_store:
        return
    
    print("\n준비 완료! 검색어를 입력하세요.")
    print("(LLM 없이 유사도 검색 결과만 보여줍니다)\n")
    
    # 대화형 인터페이스
    while True:
        query = input("\n검색어 (종료: quit): ").strip()
        
        if query.lower() in ['quit', 'exit', '종료']:
            print("서비스를 종료합니다.")
            break
        
        if not query:
            continue
        
        print("\n검색 중...")
        results = service.search(query, k=5)
        
        if results:
            print(f"\n=== 검색 결과 (상위 {len(results)}개) ===")
            for idx, result in enumerate(results, 1):
                print(f"\n[{idx}] 파일: {result['filename']}")
                print(f"    시간: {result['currtime']}")
                print(f"    전체카운트: {result['totalcnt']}")
                print(f"    유사도: {1 - result['score']:.3f}")  # FAISS는 거리 반환
                print(f"    내용: {result['content'][:200]}...")
        else:
            print("검색 결과가 없습니다.")

if __name__ == "__main__":
    main()