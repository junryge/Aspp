# test_phi3_fixed.py
"""
Phi-3.1 모델 테스트 - integer divide by zero 오류 해결
"""

import os
import sys
import warnings

# 경고 무시
warnings.filterwarnings("ignore")

# ===== FILENO 패치 =====
if not hasattr(sys.stdout, 'fileno'):
    sys.stdout.fileno = lambda: 1
if not hasattr(sys.stderr, 'fileno'):
    sys.stderr.fileno = lambda: 2

# 환경 변수
os.environ['LLAMA_CPP_VERBOSE'] = '0'

def test_minimal():
    """최소한의 테스트"""
    
    MODEL_PATH = r"D:/LLM_MODEL/GGUF/Phi-3.1-mini-4k-instruct-IQ2_M.gguf"
    
    print("="*60)
    print("Phi-3.1 모델 최소 테스트")
    print("="*60)
    
    # 1. 파일 확인
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 파일 없음: {MODEL_PATH}")
        return None
    
    file_size = os.path.getsize(MODEL_PATH) / (1024**3)
    print(f"✅ 파일 크기: {file_size:.2f} GB")
    
    # 2. llama-cpp-python 버전 확인
    try:
        import llama_cpp
        print(f"✅ llama-cpp-python 버전: {llama_cpp.__version__}")
        
        if llama_cpp.__version__ == "0.2.20":
            print("⚠️ 0.2.20 버전에서 IQ2_M 지원 문제가 있을 수 있습니다.")
            print("다른 버전을 시도해보세요:")
            print("  pip install llama-cpp-python==0.2.32")
            print("  또는")
            print("  pip install llama-cpp-python==0.2.11")
    except:
        print("❌ llama-cpp-python 설치 필요")
        return None
    
    # 3. 여러 설정으로 시도
    from llama_cpp import Llama
    
    configs = [
        # 설정 1: 최소 설정
        {
            "n_ctx": 512,
            "n_threads": 1,
            "n_gpu_layers": 0,
            "verbose": False,
            "use_mlock": False,
            "use_mmap": False
        },
        # 설정 2: mmap 활성화
        {
            "n_ctx": 512,
            "n_threads": 2,
            "n_gpu_layers": 0,
            "verbose": False,
            "use_mlock": False,
            "use_mmap": True
        },
        # 설정 3: 기본값만
        {
            "n_ctx": 512,
            "verbose": False
        }
    ]
    
    for i, config in enumerate(configs, 1):
        print(f"\n시도 {i}: {config}")
        try:
            llm = Llama(
                model_path=MODEL_PATH,
                **config
            )
            print(f"✅ 설정 {i} 성공!")
            
            # 간단한 테스트
            result = llm("Hello", max_tokens=10, temperature=0.1)
            print(f"테스트 출력: {result['choices'][0]['text'][:50]}")
            
            return llm
            
        except Exception as e:
            print(f"❌ 설정 {i} 실패: {e}")
            continue
    
    return None

def test_different_model():
    """다른 quantization 테스트"""
    
    base_path = r"D:/LLM_MODEL/GGUF/"
    
    # 가능한 다른 파일들
    possible_files = [
        "Phi-3.1-mini-4k-instruct-Q4_K_M.gguf",
        "Phi-3.1-mini-4k-instruct-Q5_K_M.gguf",
        "Phi-3.1-mini-4k-instruct-Q8_0.gguf",
        "Phi-3.1-mini-4k-instruct.gguf"
    ]
    
    print("\n다른 Phi-3 모델 찾기:")
    for filename in possible_files:
        full_path = os.path.join(base_path, filename)
        if os.path.exists(full_path):
            print(f"✅ 발견: {filename}")
            print(f"   이 파일로 시도해보세요: {full_path}")
    
    # 전체 폴더 스캔
    print("\nGGUF 폴더의 모든 .gguf 파일:")
    if os.path.exists(base_path):
        for file in os.listdir(base_path):
            if file.endswith('.gguf'):
                size = os.path.getsize(os.path.join(base_path, file)) / (1024**3)
                print(f"  - {file} ({size:.2f} GB)")

def main():
    """메인 함수"""
    
    print("🔍 문제 진단 시작\n")
    
    # 1. Python 및 시스템 정보
    import platform
    print(f"시스템: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    print(f"아키텍처: {platform.machine()}")
    
    # 2. 최소 테스트
    llm = test_minimal()
    
    if llm:
        print("\n" + "="*60)
        print("✅ 모델 로드 성공!")
        print("="*60)
        
        # 대화 테스트
        while True:
            user_input = input("\n테스트 입력 (quit 종료): ").strip()
            if user_input.lower() == 'quit':
                break
            
            try:
                # Phi-3 형식
                prompt = f"<|user|>\n{user_input}<|end|>\n<|assistant|>"
                result = llm(prompt, max_tokens=100, stop=["<|end|>"])
                print(f"응답: {result['choices'][0]['text']}")
            except Exception as e:
                print(f"오류: {e}")
    else:
        print("\n" + "="*60)
        print("❌ 모델 로드 실패")
        print("="*60)
        
        print("\n📝 해결 방법:")
        print("\n1. llama-cpp-python 버전 변경:")
        print("   pip uninstall llama-cpp-python -y")
        print("   pip install llama-cpp-python==0.2.32 --no-cache-dir")
        print("   또는")
        print("   pip install llama-cpp-python==0.2.11 --no-cache-dir")
        
        print("\n2. 다른 quantization 사용:")
        test_different_model()
        
        print("\n3. 최신 버전 시도:")
        print("   pip install llama-cpp-python --upgrade --no-cache-dir")
        
        print("\n4. CPU 전용 빌드:")
        print("   pip install llama-cpp-python --force-reinstall --no-cache-dir")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()