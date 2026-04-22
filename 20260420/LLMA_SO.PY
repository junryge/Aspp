"""
Gemma 4 26B-A4B-it GGUF 대화형 실행 스크립트
- CUDA 전체 GPU 오프로드 (RTX A5000 24GB 권장)
- 스트리밍 출력
- 멀티턴 대화 히스토리 유지
"""

import os
import sys
from llama_cpp import Llama

# ===== 설정 =====
MODEL_PATH = r"C:\models\gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"  # 모델 경로 수정
N_GPU_LAYERS = -1        # -1 = 전체 GPU 오프로드
N_CTX = 16384            # 컨텍스트 길이 (A5000이면 32768까지 OK, 3060은 8192 권장)
N_BATCH = 512            # 프롬프트 처리 배치
N_THREADS = 8            # CPU 스레드 (GPU 전체 오프로드면 영향 적음)
MAX_TOKENS = 2048        # 응답 최대 토큰

# Gemma 4 공식 권장 샘플링 파라미터
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 64
REPEAT_PENALTY = 1.0     # Gemma 4는 1.0 권장 (looping 없으면)


def init_model():
    """모델 로드"""
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] 모델 파일 없음: {MODEL_PATH}")
        sys.exit(1)

    print(f"[LOAD] {MODEL_PATH}")
    print(f"[GPU ] n_gpu_layers={N_GPU_LAYERS}, n_ctx={N_CTX}")

    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=N_GPU_LAYERS,
        n_ctx=N_CTX,
        n_batch=N_BATCH,
        n_threads=N_THREADS,
        chat_format="gemma",      # Gemma 4 채팅 템플릿
        verbose=True,             # 첫 로드 시 CUDA 인식 로그 확인용
        flash_attn=True,          # Flash Attention 활성화 (속도↑, VRAM↓)
    )
    print("[READY] 모델 로드 완료\n")
    return llm


def chat_loop(llm: Llama):
    """대화 루프"""
    messages = [
        {"role": "system", "content": "당신은 반도체 물류(AMHS/OHT) 및 머신러닝 전문가 AI 어시스턴트입니다. 한국어로 간결하고 정확하게 답변하세요."}
    ]

    print("=" * 60)
    print(" Gemma 4 26B-A4B 대화 시작")
    print(" 명령어: /exit 종료 | /reset 대화 초기화 | /save 저장")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n[You] ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[종료]")
            break

        if not user_input:
            continue

        # 명령어 처리
        if user_input == "/exit":
            print("[종료]")
            break
        if user_input == "/reset":
            messages = messages[:1]
            print("[대화 히스토리 초기화]")
            continue
        if user_input == "/save":
            with open("chat_history.txt", "w", encoding="utf-8") as f:
                for m in messages:
                    f.write(f"[{m['role']}]\n{m['content']}\n\n")
            print("[저장] chat_history.txt")
            continue

        messages.append({"role": "user", "content": user_input})

        # 스트리밍 응답
        print("\n[Gemma] ", end="", flush=True)
        full_response = ""
        try:
            stream = llm.create_chat_completion(
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                repeat_penalty=REPEAT_PENALTY,
                stream=True,
            )
            for chunk in stream:
                delta = chunk["choices"][0]["delta"]
                if "content" in delta:
                    token = delta["content"]
                    print(token, end="", flush=True)
                    full_response += token
            print()
        except Exception as e:
            print(f"\n[ERROR] {e}")
            messages.pop()  # 실패한 user 메시지 제거
            continue

        messages.append({"role": "assistant", "content": full_response})

        # 컨텍스트 관리: 32개 메시지 넘으면 오래된 것 제거 (system 유지)
        if len(messages) > 33:
            messages = [messages[0]] + messages[-32:]


def main():
    llm = init_model()
    chat_loop(llm)


if __name__ == "__main__":
    main()
