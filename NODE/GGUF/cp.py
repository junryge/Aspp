"""
가장 간단한 GGUF 테스트
이것도 안되면 llama-cpp-python 문제
"""

import os
import sys

# 1단계: 기본 확인
print("=== GGUF 테스트 시작 ===\n")

try:
    from llama_cpp import Llama
    print("✅ llama-cpp-python 임포트 성공")
except ImportError as e:
    print(f"❌ llama-cpp-python 임포트 실패: {e}")
    sys.exit(1)

# 2단계: 모델 경로
model_path = input("GGUF 파일 전체 경로 입력: ").strip()

# 경로 검증
if not os.path.exists(model_path):
    print(f"❌ 파일이 없습니다: {model_path}")
    sys.exit(1)

print(f"✅ 파일 존재 확인")
print(f"   크기: {os.path.getsize(model_path) / (1024**3):.2f} GB")

# 3단계: 가장 단순한 로드
print("\n모델 로드 시도...")

try:
    # 환경 변수로 mmap 비활성화
    os.environ['GGML_USE_MMAP'] = '0'
    
    # 절대 최소 설정
    model = Llama(model_path=model_path)
    print("✅ 모델 로드 성공!")
    
    # 간단한 생성 테스트
    print("\n생성 테스트...")
    result = model("Hello", max_tokens=10)
    print(f"결과: {result['choices'][0]['text']}")
    
except Exception as e:
    print(f"\n❌ 오류 발생: {type(e).__name__}")
    print(f"   메시지: {str(e)}")
    
    # fileno 오류인 경우 특별 처리
    if "fileno" in str(e).lower():
        print("\n=== fileno 오류 해결 방법 ===")
        print("1. 파일을 C:\\temp\\ 같은 단순한 경로로 이동")
        print("2. 파일명을 영문으로 변경 (공백 제거)")
        print("3. 다음 명령 실행:")
        print("   pip uninstall llama-cpp-python -y")
        print("   pip install llama-cpp-python==0.2.20")
        print("\n4. 그래도 안되면:")
        print("   pip install ctransformers")
        print("   (대체 라이브러리 사용)")

print("\n=== 테스트 완료 ===")
input("엔터를 눌러 종료...")