# final_test.py
"""
최종 테스트 - llama-cpp-python 0.3.16용
이것도 안 되면 다른 방법 있습니다!
"""

import os
import sys

# ===== 1. 최강 FILENO 패치 =====
# stdout/stderr를 완전히 대체
import io

class DummyFile:
    def write(self, x): return len(x) if x else 0
    def flush(self): pass
    def fileno(self): return 1
    def isatty(self): return False
    def readable(self): return False
    def writable(self): return True
    def seekable(self): return False
    def close(self): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass

# 백업
original_stdout = sys.stdout
original_stderr = sys.stderr

# 완전 대체
sys.stdout = DummyFile()
sys.stderr = DummyFile()

# print 함수 재정의
def safe_print(*args, **kwargs):
    """안전한 print"""
    message = ' '.join(str(arg) for arg in args)
    original_stdout.write(message + '\n')
    original_stdout.flush()

# ===== 2. 모델 테스트 =====
def test_model():
    MODEL_PATH = r"D:/LLM_MODEL/GGUF/Phi-3.1-mini-4k-instruct-IQ2_M.gguf"
    
    safe_print("="*60)
    safe_print("🚀 llama-cpp-python 0.3.16 테스트")
    safe_print("="*60)
    
    # 파일 확인
    if not os.path.exists(MODEL_PATH):
        safe_print(f"❌ 파일 없음: {MODEL_PATH}")
        return False
    
    file_size = os.path.getsize(MODEL_PATH) / (1024**3)
    safe_print(f"✅ 파일 크기: {file_size:.2f} GB")
    
    # llama_cpp 임포트
    try:
        from llama_cpp import Llama
        import llama_cpp
        safe_print(f"✅ llama-cpp-python 버전: {llama_cpp.__version__}")
    except ImportError as e:
        safe_print(f"❌ 임포트 실패: {e}")
        return False
    
    # 여러 설정 시도
    configs = [
        # 설정 1: 0.3.x 버전 기본
        {
            "n_ctx": 2048,
            "n_batch": 512,
            "verbose": False,
        },
        # 설정 2: 최소 설정
        {
            "n_ctx": 512,
            "n_batch": 128,
            "n_threads": 4,
            "verbose": False,
            "use_mlock": False,
        },
        # 설정 3: 호환성 모드
        {
            "n_ctx": 1024,
            "verbose": False,
            "n_gpu_layers": 0,
            "seed": 1337,
            "f16_kv": True,
            "logits_all": False,
            "vocab_only": False,
            "use_mmap": True,
            "use_mlock": False,
        }
    ]
    
    for i, config in enumerate(configs, 1):
        safe_print(f"\n시도 {i}/{len(configs)}...")
        try:
            llm = Llama(
                model_path=MODEL_PATH,
                **config
            )
            safe_print(f"✅ 모델 로드 성공! (설정 {i})")
            
            # 테스트
            safe_print("\n테스트 생성 중...")
            result = llm(
                "Hello, how are you?",
                max_tokens=20,
                temperature=0.7,
                echo=False
            )
            
            response = result['choices'][0]['text']
            safe_print(f"응답: {response[:100]}")
            
            return llm
            
        except Exception as e:
            safe_print(f"❌ 설정 {i} 실패: {str(e)[:100]}")
    
    return None

def alternative_solution():
    """대체 솔루션 제안"""
    safe_print("\n" + "="*60)
    safe_print("💡 대체 솔루션")
    safe_print("="*60)
    
    safe_print("""
1. 🔧 Ollama 사용 (가장 쉬움):
   - https://ollama.ai 다운로드
   - ollama pull phi3
   - ollama run phi3
   
2. 🤗 Transformers 라이브러리:
   pip install transformers torch
   
3. 📦 LM Studio (GUI):
   - https://lmstudio.ai 다운로드
   - GGUF 파일 드래그 앤 드롭
   
4. 🖥️ koboldcpp (GUI + API):
   - https://github.com/LostRuins/koboldcpp/releases
   - exe 파일 실행 후 모델 로드
   
5. 🐍 ctransformers (대체 라이브러리):
   pip install ctransformers
   
   from ctransformers import AutoModelForCausalLM
   model = AutoModelForCausalLM.from_pretrained(
       'D:/LLM_MODEL/GGUF/Phi-3.1-mini-4k-instruct-IQ2_M.gguf',
       model_type='phi3'
   )
""")

def main():
    safe_print("\n🔬 최종 테스트 시작\n")
    
    # 버전 정보
    import platform
    safe_print(f"Python: {platform.python_version()}")
    safe_print(f"OS: {platform.system()}")
    
    # 테스트 실행
    llm = test_model()
    
    if llm:
        safe_print("\n" + "🎉"*20)
        safe_print("성공! 모델이 작동합니다!")
        safe_print("🎉"*20)
        
        # 간단한 대화
        safe_print("\n간단한 대화 테스트:")
        prompts = [
            "What is 2+2?",
            "Tell me a joke",
            "Hello!"
        ]
        
        for prompt in prompts:
            safe_print(f"\nQ: {prompt}")
            try:
                result = llm(f"User: {prompt}\nAssistant:", 
                           max_tokens=50,
                           stop=["User:", "\n\n"])
                safe_print(f"A: {result['choices'][0]['text'].strip()}")
            except:
                pass
    else:
        safe_print("\n😔 여전히 문제가 있네요...")
        alternative_solution()
        
        safe_print("\n🎯 즉시 시도할 수 있는 것:")
        safe_print("1. Koboldcpp 다운로드 (가장 쉬움)")
        safe_print("2. LM Studio 설치 (GUI, 사용 편함)")
        safe_print("3. Ollama 설치 (명령어 간단)")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        safe_print(f"\n오류: {e}")
    finally:
        # stdout 복원
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        print("\n프로그램 종료")