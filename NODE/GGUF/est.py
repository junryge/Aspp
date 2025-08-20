# console_test.py
"""
GGUF 모델 콘솔 테스트 - fileno 오류 완벽 해결 버전
GUI 없이 순수 콘솔에서 실행
"""

import os
import sys
import warnings

# ===== 1단계: 모든 경고 무시 =====
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ===== 2단계: 완벽한 fileno 패치 =====
import io

# stdout 백업
original_stdout = sys.stdout
original_stderr = sys.stderr

class FakeStdout:
    """가짜 stdout - fileno 포함"""
    def __init__(self, original):
        self.original = original
        self.buffer = self
    
    def write(self, s):
        if self.original:
            return self.original.write(s)
        return len(s)
    
    def flush(self):
        if self.original:
            self.original.flush()
    
    def fileno(self):
        return 1
    
    def isatty(self):
        return False
    
    def readable(self):
        return False
    
    def writable(self):
        return True
    
    def seekable(self):
        return False

class FakeStderr:
    """가짜 stderr - fileno 포함"""
    def __init__(self, original):
        self.original = original
        self.buffer = self
    
    def write(self, s):
        if self.original:
            return self.original.write(s)
        return len(s)
    
    def flush(self):
        if self.original:
            self.original.flush()
    
    def fileno(self):
        return 2
    
    def isatty(self):
        return False
    
    def readable(self):
        return False
    
    def writable(self):
        return True
    
    def seekable(self):
        return False

# stdout/stderr 교체
sys.stdout = FakeStdout(original_stdout)
sys.stderr = FakeStderr(original_stderr)

# ===== 3단계: 환경 변수 설정 =====
os.environ['LLAMA_CPP_VERBOSE'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # GPU 비활성화
os.environ['LLAMA_CPP_LOG_DISABLE'] = '1'

# ===== 4단계: llama_cpp import =====
print("=" * 60)
print("GGUF 모델 테스트 시작")
print("=" * 60)

try:
    from llama_cpp import Llama
    print("✅ llama_cpp 모듈 임포트 성공")
except ImportError as e:
    print(f"❌ llama_cpp 모듈 임포트 실패: {e}")
    print("\n설치 명령어:")
    print("pip install llama-cpp-python==0.2.20")
    sys.exit(1)

# ===== 5단계: 모델 테스트 함수 =====
def test_model(model_path):
    """모델 로드 및 테스트"""
    
    # 파일 확인
    if not os.path.exists(model_path):
        print(f"❌ 파일이 존재하지 않습니다: {model_path}")
        return False
    
    # 파일 정보
    file_size = os.path.getsize(model_path) / (1024**3)  # GB
    print(f"\n📁 모델 파일: {os.path.basename(model_path)}")
    print(f"📊 파일 크기: {file_size:.2f} GB")
    
    # GGUF 헤더 확인
    with open(model_path, 'rb') as f:
        header = f.read(4)
        if header != b'GGUF':
            print(f"❌ GGUF 파일이 아닙니다. 헤더: {header}")
            return False
        print("✅ GGUF 형식 확인")
    
    # 모델 로드
    print("\n⏳ 모델 로드 중... (시간이 걸릴 수 있습니다)")
    
    try:
        # 최소 설정으로 로드
        llm = Llama(
            model_path=model_path,
            n_ctx=512,          # 작은 컨텍스트
            n_threads=2,        # 적은 스레드
            n_gpu_layers=0,     # GPU 비활성화
            seed=42,            # 고정 시드
            verbose=False,      # 출력 비활성화
            use_mlock=False,    # mlock 비활성화
            use_mmap=False,     # mmap 비활성화
            n_batch=8,          # 작은 배치
            f16_kv=False,       # f16 비활성화
            logits_all=False,   # logits 비활성화
            vocab_only=False,   # vocab만 로드 안함
            embedding=False     # 임베딩 모드 비활성화
        )
        
        print("✅ 모델 로드 성공!")
        
        # 간단한 테스트
        print("\n🧪 간단한 테스트 실행...")
        prompt = "Hello"
        
        result = llm(
            prompt,
            max_tokens=10,
            temperature=0.1,
            top_p=0.95,
            echo=False,
            stop=[]
        )
        
        response = result['choices'][0]['text']
        print(f"입력: {prompt}")
        print(f"출력: {response}")
        print("\n✅ 테스트 완료!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        
        # 상세 오류 정보
        import traceback
        print("\n상세 오류:")
        print("-" * 40)
        traceback.print_exc()
        print("-" * 40)
        
        return False

# ===== 6단계: 메인 실행 =====
def main():
    """메인 실행 함수"""
    
    # 시스템 정보
    import platform
    print("\n📋 시스템 정보:")
    print(f"  OS: {platform.system()} {platform.release()}")
    print(f"  Python: {platform.python_version()}")
    print(f"  아키텍처: {platform.machine()}")
    
    # llama-cpp-python 버전
    try:
        import llama_cpp
        print(f"  llama-cpp-python: {llama_cpp.__version__}")
    except:
        pass
    
    print("\n" + "=" * 60)
    
    # 모델 경로 입력
    while True:
        print("\nGGUF 모델 파일 경로를 입력하세요")
        print("(전체 경로 입력, 예: C:\\models\\llama-2-7b.gguf)")
        print("종료하려면 'quit' 입력")
        
        model_path = input("\n경로: ").strip()
        
        if model_path.lower() == 'quit':
            print("프로그램을 종료합니다.")
            break
        
        # 따옴표 제거
        model_path = model_path.strip('"').strip("'")
        
        if not model_path:
            continue
        
        # 테스트 실행
        success = test_model(model_path)
        
        if success:
            print("\n" + "🎉" * 20)
            print("모델이 정상적으로 작동합니다!")
            print("이제 GUI 버전도 작동할 것입니다.")
            print("🎉" * 20)
            
            # 대화 테스트
            answer = input("\n대화를 계속 테스트하시겠습니까? (y/n): ")
            if answer.lower() == 'y':
                interactive_chat(model_path)
        else:
            print("\n다른 모델로 시도해보세요.")

def interactive_chat(model_path):
    """간단한 대화 테스트"""
    print("\n💬 대화 모드 (종료: 'quit')")
    print("-" * 40)
    
    # 모델 로드
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_threads=4,
        n_gpu_layers=0,
        verbose=False,
        use_mlock=False,
        use_mmap=False
    )
    
    conversation = []
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == 'quit':
            break
        
        # 프롬프트 구성
        prompt = ""
        for msg in conversation[-4:]:  # 최근 4개 메시지
            prompt += msg + "\n"
        prompt += f"User: {user_input}\nAssistant:"
        
        # 응답 생성
        print("Assistant: ", end="", flush=True)
        result = llm(
            prompt,
            max_tokens=200,
            temperature=0.7,
            stop=["User:", "\n\n"]
        )
        
        response = result['choices'][0]['text'].strip()
        print(response)
        
        # 대화 기록
        conversation.append(f"User: {user_input}")
        conversation.append(f"Assistant: {response}")

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
        # stdout 복구
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        print("\n종료되었습니다.")