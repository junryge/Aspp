# test_phi3.py
"""
Phi-3.1-mini-4k-instruct 모델 전용 테스트
fileno 오류 완벽 해결 버전
"""

import os
import sys
import warnings

# 경고 무시
warnings.filterwarnings("ignore")

# ===== FILENO 패치 (가장 중요!) =====
class FakeStream:
    def __init__(self, original_stream):
        self.original = original_stream
        self.buffer = self
        
    def write(self, s):
        if self.original:
            try:
                return self.original.write(s)
            except:
                pass
        return len(s) if s else 0
    
    def flush(self):
        if self.original:
            try:
                self.original.flush()
            except:
                pass
    
    def fileno(self):
        return 1 if 'out' in str(self.original) else 2
    
    def isatty(self):
        return False
    
    def readable(self):
        return False
    
    def writable(self):
        return True
    
    def seekable(self):
        return False

# stdout/stderr 교체
original_stdout = sys.stdout
original_stderr = sys.stderr
sys.stdout = FakeStream(original_stdout)
sys.stderr = FakeStream(original_stderr)

# 환경 변수 설정
os.environ['LLAMA_CPP_VERBOSE'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# ===== 모델 테스트 =====
def test_phi3_model():
    """Phi-3.1 모델 테스트"""
    
    # 모델 경로
    MODEL_PATH = r"D:/LLM_MODEL/GGUF/Phi-3.1-mini-4k-instruct-IQ2_M.gguf"
    
    print("="*60)
    print("Phi-3.1-mini-4k-instruct 모델 테스트")
    print("="*60)
    
    # 파일 확인
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        return False
    
    file_size = os.path.getsize(MODEL_PATH) / (1024**3)
    print(f"✅ 모델 파일 발견: {os.path.basename(MODEL_PATH)}")
    print(f"📊 파일 크기: {file_size:.2f} GB")
    
    # GGUF 확인
    with open(MODEL_PATH, 'rb') as f:
        header = f.read(4)
        if header == b'GGUF':
            print("✅ GGUF 형식 확인")
        else:
            print(f"❌ GGUF 형식이 아님: {header}")
            return False
    
    # llama_cpp 임포트
    print("\n⏳ llama_cpp 모듈 로드 중...")
    try:
        from llama_cpp import Llama
        import llama_cpp
        print(f"✅ llama-cpp-python 버전: {llama_cpp.__version__}")
    except ImportError as e:
        print(f"❌ llama_cpp 임포트 실패: {e}")
        print("\n설치 명령어:")
        print("pip install llama-cpp-python==0.2.20")
        return False
    
    # 모델 로드
    print("\n⏳ Phi-3.1 모델 로드 중... (1-2분 소요)")
    try:
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=4096,         # Phi-3.1은 4k 컨텍스트
            n_threads=4,        # CPU 스레드
            n_gpu_layers=0,     # GPU 비활성화 (CPU 모드)
            seed=42,
            verbose=False,
            use_mlock=False,
            use_mmap=True,      # 메모리 맵 사용 (더 빠름)
            n_batch=512,        # 배치 크기
            f16_kv=False,
            logits_all=False,
            vocab_only=False,
            embedding=False,
            rope_scaling_type=-1,  # RoPE 스케일링 비활성화
            rope_freq_base=0,
            rope_freq_scale=0,
            numa=False          # NUMA 비활성화
        )
        print("✅ 모델 로드 성공!")
        
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        import traceback
        print("\n상세 오류:")
        traceback.print_exc()
        return False
    
    # 테스트 대화
    print("\n" + "="*60)
    print("🧪 모델 테스트")
    print("="*60)
    
    # Phi-3 형식의 프롬프트
    test_prompts = [
        "Hello! How are you?",
        "What is 2+2?",
        "Tell me a short joke."
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n테스트 {i}:")
        print(f"Q: {prompt}")
        
        # Phi-3 instruction 형식
        formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>"
        
        try:
            response = llm(
                formatted_prompt,
                max_tokens=50,
                temperature=0.7,
                top_p=0.95,
                stop=["<|end|>", "<|user|>"],
                echo=False
            )
            
            answer = response['choices'][0]['text'].strip()
            print(f"A: {answer}")
            
        except Exception as e:
            print(f"❌ 생성 오류: {e}")
    
    print("\n" + "="*60)
    print("✅ 모든 테스트 완료!")
    print("="*60)
    
    return True, llm

def interactive_mode(llm):
    """대화 모드"""
    print("\n💬 대화 모드 시작 (종료: 'quit' 또는 'exit')")
    print("-"*60)
    
    conversation_history = []
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit', '종료']:
            print("대화를 종료합니다.")
            break
        
        if not user_input:
            continue
        
        # Phi-3 프롬프트 형식
        if len(conversation_history) > 0:
            # 이전 대화 포함
            context = ""
            for h in conversation_history[-4:]:  # 최근 4개 대화
                context += h + "\n"
            prompt = f"{context}<|user|>\n{user_input}<|end|>\n<|assistant|>"
        else:
            prompt = f"<|user|>\n{user_input}<|end|>\n<|assistant|>"
        
        print("Phi-3: ", end="", flush=True)
        
        try:
            response = llm(
                prompt,
                max_tokens=200,
                temperature=0.7,
                top_p=0.95,
                stop=["<|end|>", "<|user|>", "\n\n"],
                stream=False
            )
            
            answer = response['choices'][0]['text'].strip()
            print(answer)
            
            # 대화 기록
            conversation_history.append(f"<|user|>\n{user_input}<|end|>")
            conversation_history.append(f"<|assistant|>\n{answer}<|end|>")
            
        except Exception as e:
            print(f"\n❌ 오류: {e}")

def main():
    """메인 함수"""
    print("\n🚀 Phi-3.1-mini-4k-instruct 모델 테스터")
    print("모델: D:/LLM_MODEL/GGUF/Phi-3.1-mini-4k-instruct-IQ2_M.gguf")
    
    # 시스템 정보
    import platform
    print(f"\n시스템: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    
    # 모델 테스트
    success, llm = test_phi3_model()
    
    if success:
        print("\n" + "🎉"*20)
        print("모델이 정상 작동합니다!")
        print("🎉"*20)
        
        # 대화 모드 제안
        answer = input("\n대화 모드로 진입하시겠습니까? (y/n): ").strip().lower()
        if answer == 'y':
            interactive_mode(llm)
    else:
        print("\n모델 로드에 실패했습니다.")
        print("\n해결 방법:")
        print("1. llama-cpp-python 버전 변경:")
        print("   pip install llama-cpp-python==0.2.20 --force-reinstall")
        print("2. 다른 Phi-3 quantization 시도 (Q4_K_M 등)")
        print("3. Python 3.10으로 시도")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n프로그램을 종료합니다.")
    except Exception as e:
        print(f"\n예상치 못한 오류: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # stdout 복원
        sys.stdout = original_stdout
        sys.stderr = original_stderr