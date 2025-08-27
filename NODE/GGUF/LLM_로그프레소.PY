#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
로그프레소 LLM 서비스 (모델만)
Qwen2.5 + 벡터 검색 서비스
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

class QwenLLM(LLM):
    """Qwen2.5 LLM 래퍼"""
    
    llama_model: Optional[Any] = None
    model_path: str = ""
    
    def __init__(self, model_path: str, **kwargs):
        super().__init__(**kwargs)
        self.model_path = model_path
        self.llama_model = Llama(
            model_path=model_path,
            n_gpu_layers=0,  # CPU 사용
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
        response = self.llama_model(
            prompt,
            max_tokens=512,
            temperature=0.7,
            stop=stop if stop else ["Human:", "\n\n"],
            echo=False
        )
        return response['choices'][0]['text'].strip()

class LogpressoLLMService:
    """로그프레소 LLM 검색 서비스"""
    
    def __init__(self, llm_path: str, embedding_path: str, vector_store_path: str):
        # LLM 로드
        print("🔄 LLM 모델 로딩 중...")
        self.llm = QwenLLM(model_path=llm_path)
        print("✅ LLM 로딩 완료!")
        
        # 임베딩 모델 로드
        print("🔄 임베딩 모델 로딩 중...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_path,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("✅ 임베딩 모델 로딩 완료!")
        
        # 벡터 저장소 로드
        print("🔄 벡터 저장소 로딩 중...")
        self.vector_store = FAISS.load_local(
            vector_store_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print("✅ 벡터 저장소 로딩 완료!")
        
        # QA 체인 설정
        self._setup_qa_chain()
    
    def _setup_qa_chain(self):
        """QA 체인 설정"""
        prompt_template = """다음은 로그프레소 관련 정보입니다:

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
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata
                })
            
            return {
                "answer": result["result"],
                "sources": sources
            }
        except Exception as e:
            return {
                "answer": f"검색 중 오류: {str(e)}",
                "sources": []
            }

def main():
    # 모델 경로 설정
    MODEL_DIR = "./models"
    VECTOR_DIR = "./vector_stores"
    
    llm_path = os.path.join(MODEL_DIR, "Qwen2.5-14B-Instruct-Q6_K.gguf")
    embedding_path = os.path.join(MODEL_DIR, "paraphrase-multilingual-MiniLM-L12-v2")
    vector_path = os.path.join(VECTOR_DIR, "logpresso_query")
    
    # 경로 확인
    if not os.path.exists(llm_path):
        print(f"❌ LLM 모델을 찾을 수 없습니다: {llm_path}")
        return
    
    if not os.path.exists(vector_path):
        print(f"❌ 벡터 저장소를 찾을 수 없습니다: {vector_path}")
        print("먼저 벡터 저장소를 생성해주세요.")
        return
    
    # 서비스 시작
    print("🚀 로그프레소 LLM 서비스 시작...")
    
    try:
        service = LogpressoLLMService(
            llm_path=llm_path,
            embedding_path=embedding_path,
            vector_store_path=vector_path
        )
    except Exception as e:
        print(f"❌ 서비스 초기화 실패: {e}")
        return
    
    print("\n✅ 준비 완료! 질문을 입력하세요.")
    print("\n💡 예시 질문:")
    print("- 로그프레소 쿼리 사용법을 알려주세요")
    print("- table 명령어는 어떻게 사용하나요?")
    print("- CSV 파일을 어떻게 읽어오나요?")
    print("- stats 명령어로 통계를 내는 방법은?")
    print()
    
    # 대화 시작
    while True:
        query = input("❓ 질문 (종료: quit): ").strip()
        
        if query.lower() in ['quit', 'exit', '종료', 'q']:
            print("👋 서비스를 종료합니다.")
            break
        
        if not query:
            continue
        
        try:
            print("\n🔍 검색 중...")
            result = service.search(query)
            
            print(f"\n💬 답변:")
            print(result['answer'])
            
            if result['sources']:
                print(f"\n📚 참조 소스 ({len(result['sources'])}개):")
                for idx, source in enumerate(result['sources'][:3], 1):
                    print(f"  {idx}. {source['content'][:100]}...")
            
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
        
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()