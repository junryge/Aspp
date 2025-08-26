
"""
import os
import requests
from tqdm import tqdm

def download_file(url, filepath):
    """파일 다운로드 with 진행률 표시"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as f:
        with tqdm(total=total_size, unit='iB', unit_scale=True, desc=os.path.basename(filepath)) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

def download_minilm_model():
    """MiniLM 모델 다운로드"""
    base_url = "https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/resolve/main"
    save_dir = "./models/paraphrase-multilingual-MiniLM-L12-v2"
    
    # 다운로드할 파일 목록
    files = [
        "config.json",
        "config_sentence_transformers.json", 
        "pytorch_model.bin",  # 약 470MB
        "sentence_bert_config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "modules.json",
        "1_Pooling/config.json"
    ]
    
    # 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/1_Pooling", exist_ok=True)
    
    print("MiniLM 모델 다운로드 시작...")
    
    for file in files:
        url = f"{base_url}/{file}"
        save_path = os.path.join(save_dir, file)
        
        print(f"\n다운로드 중: {file}")
        download_file(url, save_path)
    
    print(f"\n✅ 완료! 모든 파일이 {save_dir}에 저장되었습니다.")
    print(f"총 크기: 약 470MB")

if __name__ == "__main__":
    # tqdm 설치 안됐으면: pip install tqdm
    download_minilm_model()
"""
import os
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime

# LangChain imports
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import CSVLoader
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.prompts import PromptTemplate

# For Qwen2.5 model
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class QwenLLM(LLM):
    """Qwen2.5-14B-Instruct 모델을 위한 커스텀 LLM 클래스"""
    
    model_path: str = "./models/Qwen2.5-14B-Instruct-Q6_K.gguf"  # 로컬 경로
    model: Any = None
    tokenizer: Any = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __init__(self, model_path: str = None):
        super().__init__()
        if model_path:
            self.model_path = model_path
        self._load_model()
    
    def _load_model(self):
        """Qwen 모델과 토크나이저 로드"""
        print(f"모델 로딩 중: {self.model_path}...")
        
        # GGUF 형식의 경우 llama-cpp-python 사용
        # 설치: pip install llama-cpp-python
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError("llama-cpp-python이 설치되지 않았습니다. 'pip install llama-cpp-python'을 실행하세요.")
        
        # 모델 파일 존재 확인
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
        from llama_cpp import Llama
        
        # llama-cpp를 사용하여 GGUF 모델 로드
        self.model = Llama(
            model_path=self.model_path,
            n_gpu_layers=50,  # GPU에 올릴 레이어 수 (GPU 메모리에 따라 조정)
            n_ctx=4096,  # 컨텍스트 길이
            n_threads=8,  # CPU 스레드 수
            verbose=True  # 로딩 과정 표시
        )
        print("모델 로딩 완료!")
    
    @property
    def _llm_type(self) -> str:
        return "qwen2.5"
    
    def _call(
        self,
        prompt: str,
        stop: List[str] = None,
        run_manager: CallbackManagerForLLMRun = None,
        **kwargs: Any,
    ) -> str:
        """모델에서 응답 생성"""
        # llama-cpp 형식으로 호출
        response = self.model(
            prompt,
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
            stop=stop if stop else ["Human:", "\n\n"],
            echo=False  # 프롬프트를 응답에 포함하지 않음
        )
        
        # llama-cpp는 딕셔너리를 반환하므로 텍스트만 추출
        return response['choices'][0]['text'].strip()

class CSVDataProcessor:
    """CSV 파일을 처리하고 벡터 저장소를 위해 준비하는 클래스"""
    
    def __init__(self, data_folder: str = "output_by_date"):
        self.data_folder = data_folder
        self.columns = [
            "CURRTIME", "TOTALCNT", "M14AM10A", "M10AM14A", "M14AM10ASUM",
            "M14AM14B", "M14BM14A", "M14AM14BSUM", "M14AM16", "M16M14A", 
            "M14AM16SUM", "TIME"
        ]
        # 컬럼 설명 (한글)
        self.column_descriptions = {
            "CURRTIME": "현재시간",
            "TOTALCNT": "전체 카운트",
            "M14AM10A": "14A에서 10A로 이동",
            "M10AM14A": "10A에서 14A로 이동",
            "M14AM10ASUM": "14A-10A 이동 합계",
            "M14AM14B": "14A에서 14B로 이동",
            "M14BM14A": "14B에서 14A로 이동",
            "M14AM14BSUM": "14A-14B 이동 합계",
            "M14AM16": "14A에서 16으로 이동",
            "M16M14A": "16에서 14A로 이동",
            "M14AM16SUM": "14A-16 이동 합계",
            "TIME": "시간"
        }
    
    def load_all_csv_files(self) -> List[Document]:
        """폴더 내 모든 CSV 파일 로드"""
        documents = []
        
        for filename in os.listdir(self.data_folder):
            if filename.endswith('.csv'):
                filepath = os.path.join(self.data_folder, filename)
                documents.extend(self._process_single_csv(filepath, filename))
        
        return documents
    
    def _process_single_csv(self, filepath: str, filename: str) -> List[Document]:
        """단일 CSV 파일 처리"""
        try:
            df = pd.read_csv(filepath)
            documents = []
            
            # 각 행을 문서로 변환
            for idx, row in df.iterrows():
                # 한글과 영어로 내용 생성
                content_parts = [f"파일명: {filename}"]
                
                for col in self.columns:
                    if col in df.columns:
                        korean_name = self.column_descriptions.get(col, col)
                        value = row.get(col, "N/A")
                        content_parts.append(f"{korean_name}({col}): {value}")
                
                content = "\n".join(content_parts)
                
                # 메타데이터 추가
                metadata = {
                    "filename": filename,
                    "row_index": idx,
                    "currtime": row.get("CURRTIME", ""),
                    "totalcnt": row.get("TOTALCNT", 0)
                }
                
                documents.append(Document(page_content=content, metadata=metadata))
            
            print(f"처리 완료: {filename} - {len(documents)}개 문서 생성")
            return documents
            
        except Exception as e:
            print(f"파일 처리 오류 {filename}: {e}")
            return []

class CSVSearchService:
    """CSV 데이터 검색 서비스"""
    
    def __init__(self, model_path: str = "./models/Qwen2.5-14B-Instruct-Q6_K.gguf", embedding_model_path: str = None):
        # LLM 초기화
        self.llm = QwenLLM(model_path=model_path)
        
        # 폐쇄망용 로컬 임베딩 모델 경로
        if embedding_model_path is None:
            embedding_model_path = "./models/xlm-r-100langs-bert-base-nli-stsb-mean-tokens"
        
        # 한글과 영어를 모두 지원하는 임베딩 모델 사용
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_path,
            model_kwargs={
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'trust_remote_code': True  # 로컬 모델 사용시 필요
            },
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.vector_store = None
        self.qa_chain = None
        
        # 프롬프트 템플릿 (한글/영어 혼용)
        self.prompt_template = """다음은 CSV 데이터에서 검색된 관련 정보입니다:

{context}

위 정보를 바탕으로 다음 질문에 한글로 자세히 답변해주세요:
질문: {question}

답변: """
        
    def initialize_vector_store(self, data_folder: str = "output_by_date"):
        """벡터 저장소 초기화"""
        print("CSV 파일 로딩 중...")
        processor = CSVDataProcessor(data_folder)
        documents = processor.load_all_csv_files()
        
        if not documents:
            raise ValueError("로드된 문서가 없습니다!")
        
        print(f"총 {len(documents)}개 문서 로드 완료")
        
        # 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""],
            length_function=len
        )
        
        split_docs = text_splitter.split_documents(documents)
        print(f"문서 분할 완료: {len(split_docs)}개 청크")
        
        # 벡터 저장소 생성
        print("벡터 저장소 생성 중...")
        self.vector_store = FAISS.from_documents(split_docs, self.embeddings)
        print("벡터 저장소 생성 완료!")
        
        # QA 체인 설정
        prompt = PromptTemplate(
            template=self.prompt_template,
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
        """쿼리 검색"""
        if not self.qa_chain:
            raise ValueError("벡터 저장소가 초기화되지 않았습니다!")
        
        result = self.qa_chain({"query": query})
        
        # 소스 문서 정보 추가
        sources = []
        for doc in result.get("source_documents", []):
            sources.append({
                "filename": doc.metadata.get("filename", "Unknown"),
                "content": doc.page_content[:200] + "..."  # 처음 200자만
            })
        
        return {
            "answer": result["result"],
            "sources": sources
        }
    
    def save_vector_store(self, path: str = "vector_store"):
        """벡터 저장소 저장"""
        if self.vector_store:
            self.vector_store.save_local(path)
            print(f"벡터 저장소 저장 완료: {path}")
    
    def load_vector_store(self, path: str = "vector_store"):
        """저장된 벡터 저장소 로드"""
        self.vector_store = FAISS.load_local(path, self.embeddings)
        
        # QA 체인 재설정
        prompt = PromptTemplate(
            template=self.prompt_template,
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
        print(f"벡터 저장소 로드 완료: {path}")

# 사용 예제
def main():
    # 폐쇄망 환경 설정
    MODEL_DIR = "./models"  # 모델 저장 디렉토리
    DATA_DIR = "./output_by_date"  # CSV 데이터 디렉토리
    VECTOR_STORE_DIR = "./vector_stores"  # 벡터 저장소 디렉토리
    
    # 디렉토리 생성
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    
    # 서비스 초기화
    service = CSVSearchService(
        model_path=os.path.join(MODEL_DIR, "Qwen2.5-14B-Instruct-Q6_K.gguf"),
        embedding_model_path=os.path.join(MODEL_DIR, "xlm-r-100langs-bert-base-nli-stsb-mean-tokens")
    )
    
    # 벡터 저장소 초기화 (처음 실행시)
    service.initialize_vector_store(DATA_DIR)
    
    # 벡터 저장소 저장 (나중에 재사용하기 위해)
    vector_store_path = os.path.join(VECTOR_STORE_DIR, "csv_vector_store")
    service.save_vector_store(vector_store_path)
    
    # 검색 예제
    queries = [
        "TOTALCNT가 1400 이상인 데이터를 찾아줘",
        "오후 3시대의 데이터를 보여줘",
        "M14AM14B 값이 가장 높은 시간은 언제야?",
        "Find data where TOTALCNT is over 1400",
        "14A에서 16으로 이동한 수가 100 이상인 경우는?"
    ]
    
    for query in queries:
        print(f"\n질문: {query}")
        result = service.search(query)
        print(f"답변: {result['answer']}")
        print(f"참조 소스: {len(result['sources'])}개")
        for idx, source in enumerate(result['sources'][:2]):  # 처음 2개만 표시
            print(f"  - {source['filename']}: {source['content']}")

if __name__ == "__main__":
    main()