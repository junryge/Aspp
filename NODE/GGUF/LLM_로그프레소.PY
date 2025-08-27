# -*- coding: utf-8 -*-
"""
CSV 데이터 + 로그프레소 문서 통합 LLM 검색 서비스
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

class IntegratedSearchService:
    """CSV 데이터 + 로그프레소 문서 통합 검색 서비스"""
    
    def __init__(self, 
                 llm_path: str, 
                 embedding_model_path: str, 
                 csv_vector_dir: str = "./vector_stores",
                 doc_vector_dir: str = "./vector_stores/logpresso"):
        
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
        self.csv_vector_store = self._load_csv_vector_stores(csv_vector_dir)
        self.doc_vector_store = self._load_doc_vector_store(doc_vector_dir)
        
        # 통합 벡터 저장소 생성 (가능한 경우)
        self.combined_vector_store = self._create_combined_store()
        
        # QA 체인 설정
        self._setup_qa_chains()
    
    def _load_csv_vector_stores(self, csv_dir: str):
        """CSV 벡터 저장소 로드"""
        try:
            print("\n=== CSV 벡터 저장소 로딩 ===")
            
            # 메타데이터 읽기
            metadata_path = f"{csv_dir}/metadata.json"
            if not os.path.exists(metadata_path):
                print("⚠️  CSV 벡터 저장소를 찾을 수 없습니다.")
                return None
                
            with open(metadata_path, "r", encoding='utf-8') as f:
                metadata = json.load(f)
            
            total_batches = metadata['total_batches']
            print(f"총 {total_batches}개 CSV 배치 발견")
            
            # 첫 번째 배치 로드
            first_batch_path = f"{csv_dir}/batch_001"
            if not os.path.exists(first_batch_path):
                print("⚠️  CSV 배치 파일을 찾을 수 없습니다.")
                return None
                
            merged_store = FAISS.load_local(
                first_batch_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print("✓ CSV 배치 1 로드 완료")
            
            # 나머지 배치 병합
            for i in range(2, total_batches + 1):
                batch_path = f"{csv_dir}/batch_{i:03d}"
                if os.path.exists(batch_path):
                    try:
                        batch_store = FAISS.load_local(
                            batch_path,
                            self.embeddings,
                            allow_dangerous_deserialization=True
                        )
                        merged_store.merge_from(batch_store)
                        print(f"✓ CSV 배치 {i} 병합 완료")
                    except Exception as e:
                        print(f"✗ CSV 배치 {i} 로드 실패: {e}")
                        continue
            
            print("✅ CSV 벡터 저장소 로드 완료!")
            return merged_store
            
        except Exception as e:
            print(f"❌ CSV 벡터 로드 실패: {e}")
            return None
    
    def _load_doc_vector_store(self, doc_dir: str):
        """로그프레소 문서 벡터 저장소 로드"""
        try:
            print("\n=== 로그프레소 문서 벡터 저장소 로딩 ===")
            
            if not os.path.exists(doc_dir):
                print("⚠️  로그프레소 문서 벡터 저장소를 찾을 수 없습니다.")
                print(f"    경로: {doc_dir}")
                print("    먼저 로그프레소 문서를 벡터화해주세요.")
                return None
            
            doc_store = FAISS.load_local(
                doc_dir,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            # 메타데이터 확인
            metadata_path = f"{doc_dir}/doc_metadata.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, "r", encoding='utf-8') as f:
                    doc_meta = json.load(f)
                print(f"✓ 문서 섹션: {doc_meta.get('total_sections', 'Unknown')}개")
                print(f"✓ 문서 청크: {doc_meta.get('total_chunks', 'Unknown')}개")
            
            print("✅ 로그프레소 문서 벡터 저장소 로드 완료!")
            return doc_store
            
        except Exception as e:
            print(f"❌ 문서 벡터 로드 실패: {e}")
            return None
    
    def _create_combined_store(self):
        """CSV + 문서 통합 벡터 저장소 생성"""
        if self.csv_vector_store and self.doc_vector_store:
            try:
                print("\n=== 통합 벡터 저장소 생성 ===")
                # CSV 저장소를 기본으로 하고 문서 저장소 병합
                combined_store = FAISS.load_local(
                    "./vector_stores/batch_001",
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                combined_store.merge_from(self.doc_vector_store)
                print("✅ CSV + 문서 통합 저장소 생성 완료!")
                return combined_store
            except Exception as e:
                print(f"⚠️  통합 저장소 생성 실패: {e}")
                return None
        
        # 하나만 있는 경우 그것을 반환
        return self.csv_vector_store or self.doc_vector_store
    
    def _setup_qa_chains(self):
        """QA 체인들 설정"""
        
        # CSV 전용 프롬프트
        csv_prompt = """다음은 CSV 데이터에서 검색된 관련 정보입니다:

{context}

위 정보를 바탕으로 다음 질문에 한글로 자세히 답변해주세요.
수치 데이터가 있다면 구체적인 값을 제시하고, 시간대별 패턴이나 변화도 분석해주세요.

질문: {question}

답변: """

        # 문서 전용 프롬프트
        doc_prompt = """다음은 로그프레소 문서에서 검색된 관련 정보입니다:

{context}

위 정보를 바탕으로 다음 질문에 한글로 자세히 답변해주세요.
로그프레소 사용법, 설정, 기능에 대해 구체적이고 실용적인 답변을 제공해주세요.

질문: {question}

답변: """

        # 통합 프롬프트
        combined_prompt = """다음은 CSV 데이터와 로그프레소 문서에서 검색된 관련 정보입니다:

{context}

위 정보를 바탕으로 다음 질문에 한글로 자세히 답변해주세요.
- 수치 데이터가 있다면 구체적인 값과 분석을 제시하세요
- 로그프레소 사용법이 관련되어 있다면 실용적인 가이드를 제공하세요
- 두 정보를 종합하여 완전한 답변을 만들어주세요

질문: {question}

답변: """

        # 각각의 QA 체인 설정
        if self.csv_vector_store:
            self.csv_qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.csv_vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 5}
                ),
                chain_type_kwargs={
                    "prompt": PromptTemplate(
                        template=csv_prompt,
                        input_variables=["context", "question"]
                    )
                },
                return_source_documents=True
            )
        
        if self.doc_vector_store:
            self.doc_qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.doc_vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 5}
                ),
                chain_type_kwargs={
                    "prompt": PromptTemplate(
                        template=doc_prompt,
                        input_variables=["context", "question"]
                    )
                },
                return_source_documents=True
            )
        
        if self.combined_vector_store:
            self.combined_qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.combined_vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 8}  # 더 많은 결과
                ),
                chain_type_kwargs={
                    "prompt": PromptTemplate(
                        template=combined_prompt,
                        input_variables=["context", "question"]
                    )
                },
                return_source_documents=True
            )
    
    def _determine_search_type(self, query: str) -> str:
        """질문 유형 자동 판단"""
        query_lower = query.lower()
        
        # 수치/데이터 관련 키워드
        data_keywords = [
            'totalcnt', 'currtime', 'm14a', 'm10a', 'm14b', 'm16',
            '시간', '값', '수치', '데이터', '분석', '통계',
            '15시', '14시', '16시', '평균', '최대', '최소', '합계'
        ]
        
        # 로그프레소 관련 키워드  
        logpresso_keywords = [
            '로그프레소', 'logpresso', '설정', '사용법', '가이드',
            '설치', '쿼리', '검색', '분석', '매뉴얼', '도움말'
        ]
        
        data_score = sum(1 for keyword in data_keywords if keyword in query_lower)
        logpresso_score = sum(1 for keyword in logpresso_keywords if keyword in query_lower)
        
        if data_score > logpresso_score and data_score > 0:
            return "csv"
        elif logpresso_score > data_score and logpresso_score > 0:
            return "doc" 
        else:
            return "combined"  # 애매하거나 둘 다 관련된 경우
    
    def search(self, query: str, search_type: str = "auto") -> Dict[str, Any]:
        """통합 검색 실행"""
        
        # 자동 유형 판단
        if search_type == "auto":
            search_type = self._determine_search_type(query)
        
        print(f"검색 유형: {search_type}")
        
        try:
            # 검색 유형에 따른 실행
            if search_type == "csv" and hasattr(self, 'csv_qa_chain'):
                result = self.csv_qa_chain({"query": query})
                search_source = "CSV 데이터"
                
            elif search_type == "doc" and hasattr(self, 'doc_qa_chain'):
                result = self.doc_qa_chain({"query": query})
                search_source = "로그프레소 문서"
                
            elif search_type == "combined" and hasattr(self, 'combined_qa_chain'):
                result = self.combined_qa_chain({"query": query})
                search_source = "CSV 데이터 + 로그프레소 문서"
                
            else:
                # 폴백: 사용 가능한 것 중 하나
                if hasattr(self, 'combined_qa_chain'):
                    result = self.combined_qa_chain({"query": query})
                    search_source = "통합 검색"
                elif hasattr(self, 'csv_qa_chain'):
                    result = self.csv_qa_chain({"query": query})
                    search_source = "CSV 데이터"
                elif hasattr(self, 'doc_qa_chain'):
                    result = self.doc_qa_chain({"query": query})
                    search_source = "로그프레소 문서"
                else:
                    raise Exception("사용 가능한 벡터 저장소가 없습니다.")
            
            # 소스 정보 정리
            sources = []
            for doc in result.get("source_documents", []):
                source_info = {
                    "content": doc.page_content[:300] + "...",
                    "metadata": doc.metadata
                }
                
                # 소스 유형별 정보 추가
                if 'filename' in doc.metadata:
                    source_info["type"] = "CSV 데이터"
                    source_info["filename"] = doc.metadata["filename"]
                elif 'source' in doc.metadata:
                    source_info["type"] = "로그프레소 문서"  
                    source_info["section"] = doc.metadata.get("title", "섹션")
                else:
                    source_info["type"] = "알 수 없음"
                
                sources.append(source_info)
            
            return {
                "answer": result["result"],
                "sources": sources,
                "search_type": search_type,
                "search_source": search_source
            }
            
        except Exception as e:
            print(f"검색 중 오류: {e}")
            return {
                "answer": f"검색 중 오류가 발생했습니다: {str(e)}",
                "sources": [],
                "search_type": search_type,
                "search_source": "오류"
            }

def main():
    # 설정
    MODEL_DIR = "./models"
    CSV_VECTOR_DIR = "./vector_stores"
    DOC_VECTOR_DIR = "./vector_stores/logpresso"
    
    # 경로 확인
    llm_path = os.path.join(MODEL_DIR, "Qwen2.5-14B-Instruct-Q6_K.gguf")
    embedding_path = os.path.join(MODEL_DIR, "paraphrase-multilingual-MiniLM-L12-v2")
    
    if not os.path.exists(llm_path):
        print(f"LLM 모델을 찾을 수 없습니다: {llm_path}")
        return
    
    # 서비스 초기화
    print("=== 통합 검색 서비스 시작 ===")
    
    try:
        service = IntegratedSearchService(
            llm_path=llm_path,
            embedding_model_path=embedding_path,
            csv_vector_dir=CSV_VECTOR_DIR,
            doc_vector_dir=DOC_VECTOR_DIR
        )
    except Exception as e:
        print(f"서비스 초기화 실패: {e}")
        return
    
    print("\n준비 완료! 질문을 입력하세요.\n")
    print("예시 질문:")
    print("  - 2025년 8월 7일 15시 30분의 TOTALCNT는 얼마인가요?")
    print("  - M14AM14B 값이 가장 높은 시간대는 언제인가요?") 
    print("  - 로그프레소 설정하는 방법을 알려주세요")
    print("  - 이 데이터를 로그프레소에서 어떻게 분석할 수 있나요?")
    print()
    
    # 대화형 인터페이스
    while True:
        query = input("질문 (종료: quit): ").strip()
        
        if query.lower() in ['quit', 'exit', '종료', 'q']:
            print("서비스를 종료합니다.")
            break
        
        if not query:
            continue
        
        # 검색 유형 지정 처리 (선택사항)
        search_type = "auto"
        
        try:
            print(f"검색 유형: {search_type}")
            result = service.search(query, search_type)
            
            print(f"\n답변: {result['answer']}")
            print(f"검색 소스: {result['search_source']}")
            
            if result['sources']:
                print(f"\n참조 소스 ({len(result['sources'])}개):")
                for idx, source in enumerate(result['sources'][:3], 1):
                    print(f"  {idx}. [{source['type']}]")
                    if source['type'] == "CSV 데이터":
                        print(f"     파일: {source.get('filename', 'Unknown')}")
                    elif source['type'] == "로그프레소 문서":
                        print(f"     섹션: {source.get('section', 'Unknown')}")
                    print(f"     내용: {source['content'][:150]}...")
            
        except Exception as e:
            print(f"오류 발생: {e}")
        
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()