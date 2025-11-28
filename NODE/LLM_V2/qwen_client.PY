# qwen_client.py
import requests

# 리눅스 GPU 서버 IP 입력
QWEN_API_URL = "http://리눅스서버IP:8000"


def chat(message: str, system: str = "You are a helpful assistant. 한국어로 답변해주세요.") -> str:
    """Qwen3-Coder API 호출"""
    try:
        resp = requests.post(
            f"{QWEN_API_URL}/chat",
            json={"message": message, "system": system},
            timeout=180
        )
        
        if resp.status_code == 200:
            return resp.json().get('response', '')
        else:
            print(f"API 에러: {resp.status_code}")
            return None
            
    except Exception as e:
        print(f"API 호출 실패: {e}")
        return None


def is_available() -> bool:
    """서버 상태 확인"""
    try:
        resp = requests.get(f"{QWEN_API_URL}/health", timeout=5)
        return resp.status_code == 200
    except:
        return False


# 테스트
if __name__ == "__main__":
    print("=" * 50)
    print("Qwen3-Coder API 클라이언트 테스트")
    print("=" * 50)
    
    if is_available():
        print("✅ Qwen3-Coder 연결됨!\n")
        
        # 대화 루프
        print("종료: quit")
        print("-" * 50)
        
        while True:
            user = input("\n사용자: ")
            if user.lower() in ['quit', 'exit', 'q']:
                break
            
            response = chat(user)
            if response:
                print(f"\nAI: {response}")
            else:
                print("\n❌ 응답 실패")
    else:
        print("❌ 연결 실패 - 서버 IP/포트 확인해")
        print(f"현재 설정: {QWEN_API_URL}")