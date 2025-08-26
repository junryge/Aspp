#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
사전 생성된 벡터 저장소를 사용한 LLM 검색 서비스
"""

import os
import json
from typing import List, Dict, Any

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from llama_cpp import Llama

class FastSearchService:
    """사전 생성된 벡터 저장소를 사용하는 검색 서비스"""
    
    def __init__(self, llm_path: str, embedding_model_path: str, vector_store_dir: str = "./vector_stores"):
        # LLM 로드 (필요할 때만)
        print("LLM 모델 로딩 중...")
        self.llm = self._load_llm(llm_path)
        print("LLM 로딩 완료!")
        
        # 임베딩 모델 로드
        print("임베딩 모델 로딩 중...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_path,
            model_kwargs={'device': 'cpu'},  # GPU 필요시 'cuda'
            encode_kwargs={'normalize_embeddings': True}
        )
        print("임베딩 모델 로딩 완료!")
        
        # 벡터 저장소 로드
        self.vector_store_dir = vector_store_dir
        self.vector_store = self._load_all_vector_stores()
        
        # QA 체인 설정
        self._setup_qa_chain()
    
    def _load_llm(self, model_path: str):
        """LLM 로드 (간단한 래퍼)"""
        from langchain.llms.base import LLM
        from langchain.callbacks.manager import CallbackManagerForLLMRun
        
        class SimpleLLM(LLM):
            def __init__(self, model_path):
                super().__init__()
                self.model = Llama(
                    model_path=model_path,
                    n_gpu_layers=0,  # CPU만 사용
                    n_ctx=4096,
                    n_threads=8,
                    verbose=False
                )
            
            @property
            def _llm_type(self) -> str:
                return "qwen2.5"
            
            def _call(self, prompt: str, stop: List[str] = None, 
                     run_manager: CallbackManagerForLLMRun = None, **kwargs: Any) -> str:
                response = self.model(
                    prompt,
                    max_tokens=512,
                    temperature=0.7,
                    stop=stop if stop else ["Human:", "\n\n"],
                    echo=False
                )
                return response['choices'][0]['text'].strip()
        
        return SimpleLLM(model_path)
    
    def _load_all_vector_stores(self):
        """모든 배치의 벡터 저장소 로드 및 병합"""
        print("\n벡터 저장소 로딩 중...")
        
        # 메타데이터 읽기
        with open(f"{self.vector_store_dir}/metadata.json", "r") as f:
            metadata = json.load(f)
        
        total_batches = metadata['total_batches']
        print(f"총 {total_batches}개 배치 발견")
        
        # 첫 번째 배치 로드
        merged_store = FAISS.load_local(
            f"{self.vector_store_dir}/batch_001", 
            self.embeddings,
            allow_dangerous_deserialization=True  # 신뢰할 수 있는 파일만
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
    
    def _setup_qa_chain(self):
        """QA 체인 설정"""
        prompt_template = """다음은 CSV 데이터에서 검색된 관련 정보입니다:

{context}

위 정보를 바탕으로 다음 질문에 한글로 자세히 답변해주세요:
질문: {question}

답변: """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            ),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
    
    def search(self, query: str) -> Dict[str, Any]:
        """검색 실행"""
        result = self.qa_chain({"query": query})
        
        sources = []
        for doc in result.get("source_documents", []):
            sources.append({
                "filename": doc.metadata.get("filename", "Unknown"),
                "content": doc.page_content[:200] + "..."
            })
        
        return {
            "answer": result["result"],
            "sources": sources
        }

def main():
    # 설정
    MODEL_DIR = "./models"
    VECTOR_STORE_DIR = "./vector_stores"
    
    # 서비스 초기화
    print("=== CSV 검색 서비스 시작 ===")
    
    service = FastSearchService(
        llm_path=os.path.join(MODEL_DIR, "Qwen2.5-14B-Instruct-Q6_K.gguf"),
        embedding_model_path=os.path.join(MODEL_DIR, "paraphrase-multilingual-MiniLM-L12-v2"),
        vector_store_dir=VECTOR_STORE_DIR
    )
    
    print("\n준비 완료! 질문을 입력하세요.\n")
    
    # 대화형 인터페이스
    while True:
        query = input("질문 (종료: quit): ").strip()
        
        if query.lower() in ['quit', 'exit', '종료']:
            print("서비스를 종료합니다.")
            break
        
        if not query:
            continue
        
        try:
            print("\n검색 중...")
            result = service.search(query)
            
            print(f"\n답변: {result['answer']}")
            print(f"\n참조 소스 ({len(result['sources'])}개):")
            for idx, source in enumerate(result['sources'][:3]):
                print(f"  {idx+1}. {source['filename']}")
            
        except Exception as e:
            print(f"오류 발생: {e}")
        
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()