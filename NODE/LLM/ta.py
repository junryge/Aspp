import requests
from IPython.display import display, HTML, IFrame

print("🔍 서버 상태 확인 중...")

try:
    r = requests.get("http://localhost:7870", timeout=3)
    print("✅ 서버 연결 성공!")
    print(f"📡 응답 코드: {r.status_code}")
    print("\n" + "="*50)
    
    # IFrame으로 표시
    display(IFrame("http://localhost:7870", width=1200, height=800))
    
except requests.exceptions.ConnectionRefusedError:
    print("❌ 서버가 실행되지 않았습니다!")
    print("\n터미널에서 실행하세요:")
    print("cd /project/workSpace/LLM_GGUF/LLM1/model/text-generation-webui-main")
    print("python server.py --listen --listen-port 7870 --api --api-port 5010")
    
except Exception as e:
    print(f"❌ 오류: {e}")