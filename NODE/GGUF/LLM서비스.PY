# -*- coding: utf-8 -*-
"""
사전 생성된 벡터 저장소를 사용한 LLM 검색 서비스
"""

import os
import json
from typing import List, Dict, Any, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from llama_cpp import Llama

class SimpleLLM(LLM):
    """Llama.cpp 래퍼 - Pydantic 호환"""
    
    llama_model: Optional[Any] = None  # Pydantic 필드로 선언
    model_path: str = ""
    
    def __init__(self, model_path: str, **kwargs):
        super().__init__(**kwargs)
        self.model_path = model_path
        # 모델을 필드로 직접 할당
        self.llama_model = Llama(
            model_path=model_path,
            n_gpu_layers=0,  # CPU만 사용
            n_ctx=4096,
            n_threads=8,
            verbose=False
        )
    
    @property
    def _llm_type(self) -> str:
        return "qwen2.5"
    
    def _call(
        self, 
        prompt: str, 
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None, 
        **kwargs: Any
    ) -> str:
        """LLM 호출"""
        if self.llama_model is None:
            raise ValueError("LLM model not initialized")
            
        response = self.llama_model(
            prompt,
            max_tokens=512,
            temperature=0.7,
            stop=stop if stop else ["Human:", "\n\n"],
            echo=False
        )
        return response['choices'][0]['text'].strip()

class FastSearchService:
    """사전 생성된 벡터 저장소를 사용하는 검색 서비스"""
    
    def __init__(self, llm_path: str, embedding_model_path: str, vector_store_dir: str = "./vector_stores"):
        # LLM 로드
        print("LLM 모델 로딩 중...")
        self.llm = SimpleLLM(model_path=llm_path)
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
    
    def _load_all_vector_stores(self):
        """모든 배치의 벡터 저장소 로드 및 병합"""
        print("\n벡터 저장소 로딩 중...")
        
        # 메타데이터 읽기
        metadata_path = f"{self.vector_store_dir}/metadata.json"
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"메타데이터 파일을 찾을 수 없습니다: {metadata_path}")
            
        with open(metadata_path, "r", encoding='utf-8') as f:
            metadata = json.load(f)
        
        total_batches = metadata['total_batches']
        print(f"총 {total_batches}개 배치 발견")
        
        # 첫 번째 배치 로드
        first_batch_path = f"{self.vector_store_dir}/batch_001"
        if not os.path.exists(first_batch_path):
            raise FileNotFoundError(f"첫 번째 배치를 찾을 수 없습니다: {first_batch_path}")
            
        merged_store = FAISS.load_local(
            first_batch_path,
            self.embeddings,
            allow_dangerous_deserialization=True  # 신뢰할 수 있는 파일만
        )
        print("배치 1 로드 완료")
        
        # 나머지 배치 병합
        for i in range(2, total_batches + 1):
            batch_path = f"{self.vector_store_dir}/batch_{i:03d}"
            if os.path.exists(batch_path):
                try:
                    batch_store = FAISS.load_local(
                        batch_path,
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    merged_store.merge_from(batch_store)
                    print(f"배치 {i} 병합 완료")
                except Exception as e:
                    print(f"배치 {i} 로드 실패: {e}")
                    continue
        
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
        try:
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
        except Exception as e:
            print(f"검색 중 오류: {e}")
            return {
                "answer": f"검색 중 오류가 발생했습니다: {str(e)}",
                "sources": []
            }

def main():
    # 설정
    MODEL_DIR = "./models"
    VECTOR_STORE_DIR = "./vector_stores"
    
    # 경로 확인
    llm_path = os.path.join(MODEL_DIR, "Qwen2.5-14B-Instruct-Q6_K.gguf")
    embedding_path = os.path.join(MODEL_DIR, "paraphrase-multilingual-MiniLM-L12-v2")
    
    if not os.path.exists(llm_path):
        print(f"LLM 모델을 찾을 수 없습니다: {llm_path}")
        return
    
    if not os.path.exists(VECTOR_STORE_DIR):
        print(f"벡터 저장소 디렉토리를 찾을 수 없습니다: {VECTOR_STORE_DIR}")
        print("먼저 벡터 저장소를 생성해주세요.")
        return
    
    # 서비스 초기화
    print("=== CSV 검색 서비스 시작 ===")
    
    try:
        service = FastSearchService(
            llm_path=llm_path,
            embedding_model_path=embedding_path,
            vector_store_dir=VECTOR_STORE_DIR
        )
    except Exception as e:
        print(f"서비스 초기화 실패: {e}")
        return
    
    print("\n준비 완료! 질문을 입력하세요.\n")
    print("예시 질문:")
    print("  - 2025년 8월 7일 15시 30분의 TOTALCNT는 얼마인가요?")
    print("  - M14AM14B 값이 가장 높은 시간대는 언제인가요?")
    print("  - 전체 데이터의 평균 TOTALCNT는 얼마인가요?")
    print()
    
    # 대화형 인터페이스
    while True:
        query = input("질문 (종료: quit): ").strip()
        
        if query.lower() in ['quit', 'exit', '종료', 'q']:
            print("서비스를 종료합니다.")
            break
        
        if not query:
            continue
        
        try:
            print("\n검색 중...")
            result = service.search(query)
            
            print(f"\n답변: {result['answer']}")
            
            if result['sources']:
                print(f"\n참조 소스 ({len(result['sources'])}개):")
                for idx, source in enumerate(result['sources'][:3], 1):
                    print(f"  {idx}. {source['filename']}")
                    print(f"     내용: {source['content']}")
            
        except Exception as e:
            print(f"오류 발생: {e}")
        
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()