import requests

# 서버 설정
SERVER_IP = "서버IP입력"  # 리눅스 서버 IP
SERVER_URL = f"http://{SERVER_IP}:8000"

def chat(message, system="You are a helpful coding assistant."):
    resp = requests.post(f"{SERVER_URL}/chat", json={
        "message": message,
        "system": system
    })
    return resp.json()['response']

# 대화 루프
print("="*50)
print(f"Qwen3-Coder API 클라이언트")
print(f"서버: {SERVER_URL}")
print("종료: quit")
print("="*50)

while True:
    user = input("\n사용자: ")
    if user.lower() in ['quit', 'exit', 'q']:
        break
    
    try:
        response = chat(user)
        print(f"\nAI: {response}")
    except Exception as e:
        print(f"에러: {e}")