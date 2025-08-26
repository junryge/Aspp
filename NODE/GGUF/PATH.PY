#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
폐쇄망 환경 설정 및 검증 스크립트
"""

import os
import sys
import shutil
from pathlib import Path

def check_environment():
    """환경 확인"""
    print("=== 환경 확인 중 ===")
    
    # Python 버전 확인
    print(f"Python 버전: {sys.version}")
    
    # 필수 디렉토리 생성
    dirs = ["./models", "./output_by_date", "./vector_stores", "./offline_packages"]
    for dir_path in dirs:
        Path(dir_path).mkdir(exist_ok=True)
        print(f"✓ 디렉토리 생성/확인: {dir_path}")
    
    return True

def check_models():
    """모델 파일 확인"""
    print("\n=== 모델 파일 확인 ===")
    
    required_files = {
        "Qwen 모델": "./models/Qwen2.5-14B-Instruct-Q6_K.gguf",
        "임베딩 모델 설정": "./models/xlm-r-100langs-bert-base-nli-stsb-mean-tokens/config.json",
        "임베딩 모델 가중치": "./models/xlm-r-100langs-bert-base-nli-stsb-mean-tokens/pytorch_model.bin"
    }
    
    all_found = True
    for name, path in required_files.items():
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024**3)  # GB
            print(f"✓ {name}: {path} ({size:.2f} GB)")
        else:
            print(f"✗ {name}: {path} (찾을 수 없음)")
            all_found = False
    
    return all_found

def install_packages():
    """오프라인 패키지 설치"""
    print("\n=== 오프라인 패키지 설치 ===")
    
    if not os.path.exists("./offline_packages"):
        print("✗ offline_packages 폴더를 찾을 수 없습니다.")
        return False
    
    # pip 설치 명령
    install_cmd = f"{sys.executable} -m pip install --no-index --find-links ./offline_packages"
    
    packages = [
        "langchain",
        "langchain-community", 
        "faiss-cpu",
        "sentence-transformers",
        "llama-cpp-python",
        "pandas",
        "torch",
        "transformers",
        "numpy"
    ]
    
    print("다음 명령으로 패키지를 설치하세요:")
    print(f"\n{install_cmd} {' '.join(packages)}\n")
    
    return True

def test_imports():
    """필수 모듈 import 테스트"""
    print("\n=== Import 테스트 ===")
    
    modules = [
        ("langchain", "LangChain"),
        ("llama_cpp", "llama-cpp-python"),
        ("sentence_transformers", "sentence-transformers"),
        ("faiss", "FAISS"),
        ("pandas", "Pandas"),
        ("torch", "PyTorch")
    ]
    
    all_imported = True
    for module, name in modules:
        try:
            __import__(module)
            print(f"✓ {name} import 성공")
        except ImportError:
            print(f"✗ {name} import 실패")
            all_imported = False
    
    return all_imported

def create_sample_config():
    """샘플 설정 파일 생성"""
    config_content = """# CSV 검색 서비스 설정
MODEL_CONFIG = {
    "llm_model_path": "./models/Qwen2.5-14B-Instruct-Q6_K.gguf",
    "embedding_model_path": "./models/xlm-r-100langs-bert-base-nli-stsb-mean-tokens",
    "n_gpu_layers": 50,  # GPU 메모리에 따라 조정 (0 = CPU only)
    "n_ctx": 4096,       # 컨텍스트 길이
    "n_threads": 8       # CPU 스레드 수
}

DATA_CONFIG = {
    "csv_folder": "./output_by_date",
    "vector_store_path": "./vector_stores/csv_vector_store",
    "chunk_size": 500,
    "chunk_overlap": 50
}

SEARCH_CONFIG = {
    "top_k": 5,  # 검색 결과 수
    "temperature": 0.7,
    "max_tokens": 512
}
"""
    
    with open("config.py", "w", encoding="utf-8") as f:
        f.write(config_content)
    print("\n✓ config.py 파일 생성 완료")

def main():
    print("폐쇄망 LLM 서비스 설정 도구\n")
    
    # 1. 환경 확인
    check_environment()
    
    # 2. 모델 파일 확인
    models_ok = check_models()
    if not models_ok:
        print("\n⚠️  필수 모델 파일이 없습니다. 다음 파일들을 준비해주세요:")
        print("1. Qwen2.5-14B-Instruct-Q6_K.gguf")
        print("2. xlm-r-100langs-bert-base-nli-stsb-mean-tokens 폴더")
    
    # 3. 패키지 설치 안내
    install_packages()
    
    # 4. Import 테스트
    imports_ok = test_imports()
    
    # 5. 설정 파일 생성
    create_sample_config()
    
    # 결과 요약
    print("\n=== 설정 요약 ===")
    if models_ok and imports_ok:
        print("✅ 모든 준비가 완료되었습니다!")
        print("\n다음 명령으로 서비스를 실행하세요:")
        print("python csv_search_service.py")
    else:
        print("❌ 일부 구성 요소가 누락되었습니다.")
        print("위의 안내에 따라 필요한 파일과 패키지를 설치해주세요.")

if __name__ == "__main__":
    main()