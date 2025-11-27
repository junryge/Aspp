"""
로컬 GGUF 모델 대화 스크립트 (30B 모델 기준, GPU 사용)
llama-cpp-python 필요: pip install llama-cpp-python
"""

from llama_cpp import Llama

# ============ 여기서 모델 경로 수정 ============
MODEL_PATH = "Qwen3-Coder-30B-A3B-Instruct.gguf"  # 사용할 모델 파일 경로
# =============================================

# 모델 로드 (30B 모델 + GPU 최적화)
print(f"모델 로딩 중: {MODEL_PATH}")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=8192,         # 컨텍스트 길이 (30B는 넉넉하게)
    n_gpu_layers=-1,    # 전체 레이어 GPU 사용
    n_batch=512,        # 배치 크기
    n_threads=8,        # CPU 스레드 (혼합 사용시)
    verbose=False
)
print("로딩 완료!\n")

# 대화 루프
print("=== 대화 시작 (종료: quit 또는 q) ===\n")

while True:
    user_input = input("You: ").strip()
    
    if user_input.lower() in ['quit', 'q', 'exit']:
        print("대화 종료")
        break
    
    if not user_input:
        continue
    
    # 응답 생성
    response = llm.create_chat_completion(
        messages=[
            {"role": "user", "content": user_input}
        ],
        max_tokens=1024,
        temperature=0.7
    )
    
    answer = response['choices'][0]['message']['content']
    print(f"AI: {answer}\n")