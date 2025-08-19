#!/usr/bin/env python3
"""

# 캐시까지 완전 제거
pip uninstall llama-cpp-python -y
pip cache purge

# Python 3.11 전용 설치
pip install llama-cpp-python==0.2.11 --no-cache-dir --force-reinstall
llama-cpp-python 설치 및 호환성 확인 스크립트
"""

import sys
import platform
import subprocess

def check_installation():
    print("=== 시스템 정보 ===")
    print(f"Python 버전: {sys.version}")
    print(f"플랫폼: {platform.platform()}")
    print(f"프로세서: {platform.processor()}")
    print()
    
    print("=== llama-cpp-python 확인 ===")
    try:
        import llama_cpp
        print(f"✓ llama-cpp-python 설치됨")
        print(f"  버전: {llama_cpp.__version__ if hasattr(llama_cpp, '__version__') else 'Unknown'}")
        
        # 간단한 테스트
        try:
            from llama_cpp import Llama
            print("✓ Llama 클래스 import 성공")
        except Exception as e:
            print(f"✗ Llama 클래스 import 실패: {e}")
            
    except ImportError:
        print("✗ llama-cpp-python이 설치되지 않음")
        print("\n권장 설치 명령:")
        print("pip install llama-cpp-python==0.2.11 --no-cache-dir")
    
    print("\n=== CPU 기능 확인 ===")
    try:
        # Windows에서 CPU 정보 확인
        if platform.system() == "Windows":
            result = subprocess.run(["wmic", "cpu", "get", "name"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                cpu_info = result.stdout.strip().split('\n')[1:]
                print(f"CPU: {' '.join(cpu_info)}")
    except:
        pass

if __name__ == "__main__":
    check_installation()
    
    print("\n=== 권장 설치 옵션 ===")
    print("1. 일반적인 경우:")
    print("   pip install llama-cpp-python==0.2.11")
    print("\n2. 문제가 지속될 경우:")
    print("   pip install llama-cpp-python==0.2.11 --force-reinstall --no-deps")
    print("\n3. 빌드 도구가 있는 경우:")
    print("   pip install llama-cpp-python --force-reinstall --no-binary llama-cpp-python")