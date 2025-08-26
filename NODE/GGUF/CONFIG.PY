# CSV 검색 서비스 설정 파일
import os

# 기본 디렉토리 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "output_by_date")
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "vector_stores")

# 모델 설정
MODEL_CONFIG = {
    # Qwen LLM 설정
    "llm_model_path": os.path.join(MODEL_DIR, "Qwen2.5-14B-Instruct-Q6_K.gguf"),
    "llm_config": {
        "n_gpu_layers": 50,      # GPU에 올릴 레이어 수 (0 = CPU only)
        "n_ctx": 4096,          # 컨텍스트 길이
        "n_threads": 8,         # CPU 스레드 수
        "temperature": 0.7,
        "max_tokens": 512,
        "top_p": 0.9
    },
    
    # 임베딩 모델 설정 (MiniLM)
    "embedding_model_path": os.path.join(MODEL_DIR, "paraphrase-multilingual-MiniLM-L12-v2"),
    "embedding_config": {
        "device": "cuda",  # "cpu" 또는 "cuda"
        "normalize_embeddings": True
    }
}

# 데이터 처리 설정
DATA_CONFIG = {
    "csv_folder": DATA_DIR,
    "vector_store_path": os.path.join(VECTOR_STORE_DIR, "csv_vector_store"),
    "chunk_size": 500,
    "chunk_overlap": 50,
    "text_splitter_separators": ["\n\n", "\n", ".", " ", ""]
}

# 검색 설정
SEARCH_CONFIG = {
    "retriever_search_type": "similarity",
    "retriever_k": 5,  # 검색할 문서 수
    "chain_type": "stuff"
}

# 컬럼 정보
COLUMN_INFO = {
    "columns": [
        "CURRTIME", "TOTALCNT", "M14AM10A", "M10AM14A", "M14AM10ASUM",
        "M14AM14B", "M14BM14A", "M14AM14BSUM", "M14AM16", "M16M14A", 
        "M14AM16SUM", "TIME"
    ],
    "descriptions": {
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
}

# 프롬프트 템플릿
PROMPT_TEMPLATE = """다음은 CSV 데이터에서 검색된 관련 정보입니다:

{context}

위 정보를 바탕으로 다음 질문에 한글로 자세히 답변해주세요:
질문: {question}

답변: """

# 시스템 설정
SYSTEM_CONFIG = {
    "verbose": True,  # 상세 로그 출력
    "use_gpu": True,  # GPU 사용 여부
    "save_vector_store": True,  # 벡터 저장소 자동 저장
}

# 디렉토리 자동 생성
def create_directories():
    """필요한 디렉토리 생성"""
    for dir_path in [MODEL_DIR, DATA_DIR, VECTOR_STORE_DIR]:
        os.makedirs(dir_path, exist_ok=True)

# 설정 검증
def validate_config():
    """설정 파일 검증"""
    errors = []
    
    # 모델 파일 확인
    if not os.path.exists(MODEL_CONFIG["llm_model_path"]):
        errors.append(f"LLM 모델을 찾을 수 없습니다: {MODEL_CONFIG['llm_model_path']}")
    
    if not os.path.exists(MODEL_CONFIG["embedding_model_path"]):
        errors.append(f"임베딩 모델을 찾을 수 없습니다: {MODEL_CONFIG['embedding_model_path']}")
    
    # CSV 폴더 확인
    if not os.path.exists(DATA_CONFIG["csv_folder"]):
        errors.append(f"CSV 폴더를 찾을 수 없습니다: {DATA_CONFIG['csv_folder']}")
    
    return errors

if __name__ == "__main__":
    create_directories()
    errors = validate_config()
    
    if errors:
        print("설정 오류:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("✓ 설정 검증 완료!")