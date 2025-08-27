#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
로그프레소 설명서 벡터저장 시스템
텍스트 문서를 벡터화하여 LLM 검색 가능하게 만듦
"""

import os
import json
from typing import List
from datetime import datetime
import time

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.document_loaders import TextLoader, PyPDFLoader

class LogpressoDocumentVectorizer:
    """로그프레소 문서를 벡터화하는 클래스"""
    
    def __init__(self, embedding_model_path: str = "./models/paraphrase-multilingual-MiniLM-L12-v2"):
        print("임베딩 모델 로딩 중...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_path,
            model_kwargs={
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'trust_remote_code': True
            },
            encode_kwargs={'normalize_embeddings': True}
        )
        print("임베딩 모델 로딩 완료!")
    
    def load_logpresso_documents(self, file_path: str) -> List[Document]:
        """로그프레소 텍스트 문서 로드"""
        documents = []
        
        try:
            # 텍스트 파일 로드 (다양한 인코딩 시도)
            content = None
            encodings = ['utf-8', 'cp949', 'euc-kr', 'latin1']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    print(f"✓ 파일 로드 성공 (인코딩: {encoding})")
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                raise Exception("지원하는 인코딩으로 파일을 읽을 수 없습니다")
            
            if file_path.endswith('.txt'):
                # 섹션별로 나누기
                sections = self._split_into_sections(content)
                print(f"✓ 문서를 {len(sections)}개 섹션으로 분할")
                
                for i, section in enumerate(sections):
                    if section.strip():  # 빈 섹션 제외
                        metadata = {
                            "source": "로그프레소.txt",
                            "section": i + 1,
                            "doc_type": "manual", 
                            "title": self._extract_section_title(section),
                            "char_count": len(section)
                        }
                        documents.append(Document(page_content=section, metadata=metadata))
            
            # PDF 파일도 지원
            elif file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                pdf_docs = loader.load()
                
                for i, doc in enumerate(pdf_docs):
                    doc.metadata.update({
                        "source": "로그프레소 매뉴얼",
                        "page": i + 1,
                        "doc_type": "manual"
                    })
                    documents.append(doc)
        
        except Exception as e:
            print(f"문서 로드 오류: {e}")
        
        return documents
    
    def _split_into_sections(self, content: str) -> List[str]:
        """텍스트를 섹션별로 분할"""
        # 제목이나 번호로 섹션 구분 (예: "1.", "2.", "가.", "나." 등)
        import re
        
        # 패턴들 (필요에 따라 조정)
        section_patterns = [
            r'\n\d+\.\s+',  # "1. ", "2. " 등
            r'\n[가-힣]\.\s+',  # "가. ", "나. " 등
            r'\n#{1,3}\s+',  # "# ", "## ", "### " 등 (마크다운)
            r'\n\[.*?\]',  # "[section]" 형태
        ]
        
        sections = [content]  # 기본적으로 전체 내용
        
        for pattern in section_patterns:
            new_sections = []
            for section in sections:
                split_parts = re.split(pattern, section)
                new_sections.extend(split_parts)
            sections = [s for s in new_sections if s.strip()]
        
        return sections
    
    def _extract_section_title(self, section: str) -> str:
        """섹션의 제목 추출"""
        lines = section.split('\n')
        for line in lines:
            line = line.strip()
            if line and len(line) < 100:  # 제목으로 추정되는 짧은 줄
                return line[:50]  # 최대 50자
        return "제목 없음"
    
    def create_vector_store(self, doc_file_path: str, save_path: str = "./vector_stores/logpresso"):
        """로그프레소 문서의 벡터 저장소 생성"""
        print(f"\n로그프레소 문서 처리 시작: {doc_file_path}")
        
        # 문서 로드
        documents = self.load_logpresso_documents(doc_file_path)
        print(f"문서 로드 완료: {len(documents)}개 섹션")
        
        if not documents:
            print("로드된 문서가 없습니다!")
            return
        
        # 텍스트 분할기 설정 (문서용)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # 문서는 조금 더 큰 청크
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""],
            length_function=len
        )
        
        # 문서 분할
        split_docs = text_splitter.split_documents(documents)
        print(f"문서 분할 완료: {len(split_docs)}개 청크")
        
        # 벡터 저장소 생성
        print("벡터 임베딩 생성 중...")
        vector_store = FAISS.from_documents(split_docs, self.embeddings)
        
        # 저장
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        vector_store.save_local(save_path)
        
        # 메타데이터 저장
        metadata = {
            "document_type": "logpresso_manual",
            "source_file": doc_file_path,
            "total_sections": len(documents),
            "total_chunks": len(split_docs),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        metadata_path = f"{save_path}/doc_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 로그프레소 벡터 저장소 생성 완료: {save_path}")
        return vector_store

class CombinedVectorSearch:
    """CSV 데이터 + 로그프레소 문서 통합 검색"""
    
    def __init__(self, 
                 csv_vector_dir: str = "./vector_stores", 
                 doc_vector_dir: str = "./vector_stores/logpresso",
                 embedding_model_path: str = "./models/paraphrase-multilingual-MiniLM-L12-v2"):
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_path,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # CSV 데이터 벡터 스토어 로드
        self.csv_vector_store = self._load_csv_vectors(csv_vector_dir)
        
        # 문서 벡터 스토어 로드
        self.doc_vector_store = self._load_doc_vectors(doc_vector_dir)
        
        print("통합 벡터 검색 시스템 준비 완료!")
    
    def _load_csv_vectors(self, csv_dir: str):
        """CSV 벡터 저장소 로드"""
        try:
            # 기존 CSV 벡터 로드 로직
            metadata_path = f"{csv_dir}/metadata.json"
            if not os.path.exists(metadata_path):
                print("CSV 벡터 저장소를 찾을 수 없습니다.")
                return None
            
            with open(metadata_path, "r", encoding='utf-8') as f:
                metadata = json.load(f)
            
            # 첫 번째 배치 로드 후 병합
            first_batch = f"{csv_dir}/batch_001"
            merged_store = FAISS.load_local(
                first_batch, self.embeddings, 
                allow_dangerous_deserialization=True
            )
            
            # 나머지 배치 병합
            total_batches = metadata['total_batches']
            for i in range(2, total_batches + 1):
                batch_path = f"{csv_dir}/batch_{i:03d}"
                if os.path.exists(batch_path):
                    batch_store = FAISS.load_local(
                        batch_path, self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    merged_store.merge_from(batch_store)
            
            print("CSV 벡터 저장소 로드 완료")
            return merged_store
            
        except Exception as e:
            print(f"CSV 벡터 로드 실패: {e}")
            return None
    
    def _load_doc_vectors(self, doc_dir: str):
        """문서 벡터 저장소 로드"""
        try:
            if not os.path.exists(doc_dir):
                print("로그프레소 문서 벡터 저장소를 찾을 수 없습니다.")
                return None
            
            doc_store = FAISS.load_local(
                doc_dir, self.embeddings,
                allow_dangerous_deserialization=True
            )
            print("로그프레소 문서 벡터 저장소 로드 완료")
            return doc_store
            
        except Exception as e:
            print(f"문서 벡터 로드 실패: {e}")
            return None
    
    def search(self, query: str, search_type: str = "both", k: int = 5):
        """통합 검색"""
        results = {
            "csv_results": [],
            "doc_results": [],
            "query": query
        }
        
        if search_type in ["both", "csv"] and self.csv_vector_store:
            try:
                csv_docs = self.csv_vector_store.similarity_search(query, k=k)
                results["csv_results"] = [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "type": "csv_data"
                    }
                    for doc in csv_docs
                ]
                print(f"CSV 검색 결과: {len(csv_docs)}개")
            except Exception as e:
                print(f"CSV 검색 오류: {e}")
        
        if search_type in ["both", "doc"] and self.doc_vector_store:
            try:
                doc_docs = self.doc_vector_store.similarity_search(query, k=k)
                results["doc_results"] = [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "type": "document"
                    }
                    for doc in doc_docs
                ]
                print(f"문서 검색 결과: {len(doc_docs)}개")
            except Exception as e:
                print(f"문서 검색 오류: {e}")
        
        return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="로그프레소 문서 벡터화")
    parser.add_argument("--doc-file", default="./로그프레소.txt", help="로그프레소 문서 파일")
    parser.add_argument("--save-path", default="./vector_stores/logpresso", help="저장 경로")
    parser.add_argument("--embedding-model", 
                       default="./models/paraphrase-multilingual-MiniLM-L12-v2", 
                       help="임베딩 모델 경로")
    
    args = parser.parse_args()
    
    # 문서 벡터화
    vectorizer = LogpressoDocumentVectorizer(args.embedding_model)
    vectorizer.create_vector_store(args.doc_file, args.save_path)
    
    print("\n사용법:")
    print("1. CSV + 문서 통합 검색: python integrated_search.py")
    print("2. LLM과 함께 사용: python llm_service_with_docs.py")

if __name__ == "__main__":
    import torch  # GPU 체크용
    main()