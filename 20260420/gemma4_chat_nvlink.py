"""
Gemma 4 26B-A4B-it GGUF 대화형 실행 스크립트
- NVLink 2x RTX A5000 (총 48GB VRAM) 최적화
- Layer split 모드 (NVLink 대역폭 최소화)
- 컨텍스트 64K (여유 VRAM 31GB 활용)
- 스트리밍 출력 + 멀티턴 대화
"""

import os
import sys
from llama_cpp import Llama

# ===== 모델 설정 =====
MODEL_PATH = r"C:\models\gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"  # 모델 경로 수정
N_GPU_LAYERS = -1        # -1 = 전체 GPU 오프로드

# ===== NVLink 2x A5000 멀티 GPU 설정 =====
SPLIT_MODE = 1                    # 1=layer split (NVLink 최적), 2=row split
TENSOR_SPLIT = [0.5, 0.5]         # GPU 0:1 = 50:50 분할
MAIN_GPU = 0                      # 주 GPU (토큰 샘플링/KV cache 관리)

# ===== 컨텍스트 / 배치 =====
# 17GB 모델 + 48GB VRAM = 31GB 여유 → 64K 컨텍스트 가능
# (Gemma 4 26B-A4B는 최대 256K까지 지원)
N_CTX = 65536            # 64K 컨텍스트 (장문 문서/코드베이스 분석 가능)
N_BATCH = 2048           # 프롬프트 배치 (NVLink 여유로 크게)
N_UBATCH = 512           # 물리 배치 (VRAM 여유 있으면 2048까지도 OK)
N_THREADS = 8            # CPU 스레드
MAX_TOKENS = 4096        # 응답 최대 토큰

# ===== Gemma 4 공식 권장 샘플링 =====
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 64
REPEAT_PENALTY = 1.0


def init_model():
    """모델 로드 (NVLink 2 GPU)"""
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] 모델 파일 없음: {MODEL_PATH}")
        sys.exit(1)

    print("=" * 70)
    print(f"[LOAD] {MODEL_PATH}")
    print(f"[GPU ] NVLink 2x A5000 | split={SPLIT_MODE} | ratio={TENSOR_SPLIT}")
    print(f"[CTX ] n_ctx={N_CTX:,} | batch={N_BATCH} | ubatch={N_UBATCH}")
    print("=" * 70)

    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=N_GPU_LAYERS,
        n_ctx=N_CTX,
        n_batch=N_BATCH,
        n_ubatch=N_UBATCH,
        n_threads=N_THREADS,

        # === NVLink 멀티 GPU 핵심 설정 ===
        split_mode=SPLIT_MODE,
        tensor_split=TENSOR_SPLIT,
        main_gpu=MAIN_GPU,
        # ================================

        chat_format="gemma",
        flash_attn=True,          # Flash Attention (VRAM↓ 속도↑)
        offload_kqv=True,         # KV cache를 GPU에 (NVLink 여유로 가능)
        verbose=True,             # 첫 로드 시 GPU 분배 로그 확인
    )

    print("\n[READY] 모델 로드 완료")
    print("[TIP ] nvidia-smi 실행하면 2 GPU 모두 VRAM 차있어야 정상\n")
    return llm


def chat_loop(llm: Llama):
    """대화 루프"""
    messages = [
        {"role": "system", "content": (
            "당신은 SK Hynix 반도체 FAB의 AMHS(자동 물류) 및 OHT(Overhead Hoist Transport) "
            "시스템 전문가이자, ML/XGBoost/LLM 통합 개발 전문가인 AI 어시스턴트입니다. "
            "한국어로 간결하고 정확하게 답변하며, 코드 예시는 Python/Java를 우선 사용합니다."
        )}
    ]

    print("=" * 70)
    print(" Gemma 4 26B-A4B 대화 시작 (NVLink 2x A5000)")
    print(" 명령어: /exit 종료 | /reset 초기화 | /save 저장 | /gpu GPU상태")
    print("=" * 70)

    while True:
        try:
            user_input = input("\n[You] ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[종료]")
            break

        if not user_input:
            continue

        # ===== 명령어 처리 =====
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

        if user_input == "/gpu":
            os.system("nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv")
            os.system("nvidia-smi nvlink -gt r")
            continue

        # ===== 추론 =====
        messages.append({"role": "user", "content": user_input})

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
            messages.pop()
            continue

        messages.append({"role": "assistant", "content": full_response})

        # 컨텍스트 관리: 64K면 훨씬 여유, 64개 메시지 이상만 정리
        if len(messages) > 65:
            messages = [messages[0]] + messages[-64:]


def main():
    llm = init_model()
    chat_loop(llm)


if __name__ == "__main__":
    main()
