# V100 32GB 2장 GGUF 모델 사용 가이드

## 📊 시스템 사양
| 항목 | 사양 |
|------|------|
| GPU | Tesla V100-SXM2-32GB × 2 |
| 총 VRAM | 64GB |
| 실사용 가능 | ~60GB |
| Compute Capability | 7.0 |

## 🎯 단일 GPU (32GB) 사용 가능 모델

| 양자화 | 파라미터 한계 | 실제 크기 | 추천 모델 |
|--------|--------------|-----------|-----------|
| **Q8_0** (8bit) | ~30B | ~30GB | Llama-3.1 8B, Qwen2.5 32B |
| **Q6_K** (6bit) | ~40B | ~24GB | Yi-34B, CodeLlama-34B |
| **Q5_K_M** (5bit) | ~50B | ~20GB | Falcon-40B |
| **Q4_K_M** (4bit) | ~70B | ~40GB | Llama-2 70B (타이트) |
| **Q3_K_M** (3bit) | ~90B | ~30GB | 실험적 사용 |

## 🚀 듀얼 GPU (64GB) 사용 가능 모델

| 양자화 | 파라미터 한계 | 실제 크기 | 추천 모델 |
|--------|--------------|-----------|-----------|
| **Q8_0** | ~70B | ~70GB | Llama-3.1 70B (타이트) |
| **Q6_K** | ~90B | ~54GB | Llama-3.1 70B (여유) |
| **Q5_K_M** | ~110B | ~45GB | Qwen2.5 72B |
| **Q4_K_M** | ~140B | ~40GB | Llama-3.1 70B (최적) |
| **Q3_K_M** | ~180B | ~54GB | Mixtral 8x22B |

## ⭐ 추천 구성

### 🏆 최적 성능 (속도 + 품질)
| 모델 | 양자화 | 메모리 사용 | 특징 |
|------|--------|------------|------|
| **Llama-3.1 70B** | Q4_K_M | ~40GB | 최고 밸런스 |
| **Qwen2.5 72B** | Q4_K_M | ~42GB | 한국어 강점 |
| **Mixtral 8x7B** | Q6_K | ~35GB | MoE, 빠른 속도 |
| **DeepSeek-Coder-V2** | Q5_K_M | ~45GB | 코딩 특화 |

### 💪 최대 크기 도전
| 모델 | 양자화 | 메모리 사용 | 특징 |
|------|--------|------------|------|
| **Llama-3.1 70B** | Q6_K | ~55GB | 품질 우선 |
| **Mixtral 8x22B** | Q3_K_M | ~58GB | 대규모 MoE |
| **DeepSeek-V2 236B** | Q2_K | ~60GB | 극한 압축 |

## 📈 성능 예상

### 토큰 생성 속도 (tokens/sec)
| 모델 크기 | Q4_K_M | Q5_K_M | Q6_K |
|-----------|--------|--------|------|
| 7-13B | 80-100 | 70-90 | 60-80 |
| 30-34B | 40-60 | 35-50 | 30-45 |
| 70B | 20-30 | 18-25 | 15-22 |

## ⚙️ 실행 도구별 특징

| 도구 | 멀티GPU | 특징 | 추천 용도 |
|------|---------|------|-----------|
| **vLLM** | ✅ 자동 | 최고 처리량 | API 서버 |
| **llama.cpp** | ⚠️ 수동 | CPU+GPU 혼용 | 실험/테스트 |
| **text-generation-webui** | ✅ 자동 | GUI 제공 | 대화형 사용 |
| **ExLlamaV2** | ✅ 지원 | 빠른 속도 | 고속 추론 |

## 💡 메모리 최적화 팁

### Context 길이별 추가 메모리
| Context | 추가 메모리 | 용도 |
|---------|------------|------|
| 4K | +5% | 일반 대화 |
| 8K | +10% | 문서 분석 |
| 16K | +20% | 긴 문서 |
| 32K+ | +30% | 책/논문 |

### 배치 처리 고려사항
- **단일 요청**: 기본 메모리만 사용
- **배치 크기 4**: +20% 메모리
- **배치 크기 8**: +40% 메모리
- **동시 사용자 多**: 메모리 50% 여유 권장

## 🎯 최종 추천

> **최적 선택: Llama-3.1 70B Q4_K_M**
> - 메모리 사용: 40GB/64GB (62%)
> - 품질: 우수 (Q4는 충분한 품질)
> - 속도: 20-30 tokens/sec
> - 여유 메모리로 긴 컨텍스트 처리 가능

> **실험적 선택: Mixtral 8x22B Q3_K_M**
> - MoE 아키텍처로 효율적
> - 전문가 모델 활용
> - 다양한 태스크 우수

## 📝 참고사항
- 온도 설정, 샘플링 방법에 따라 메모리 사용량 변동
- 시스템 RAM도 모델 크기의 50% 이상 권장
- NVLink 연결시 GPU간 통신 속도 향상

## 🔧 실행 예시

### vLLM 사용
```bash
# 단일 GPU
python -m vllm.entrypoints.openai.api_server \
  --model TheBloke/Llama-2-70B-GGUF \
  --quantization awq \
  --gpu-memory-utilization 0.95

# 듀얼 GPU
python -m vllm.entrypoints.openai.api_server \
  --model TheBloke/Llama-2-70B-GGUF \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.95
```

### llama.cpp 사용
```bash
# GPU 레이어 설정
./main -m llama-70b-q4_k_m.gguf \
  -ngl 80 \  # GPU 레이어 수
  -c 4096 \  # 컨텍스트 크기
  -b 512     # 배치 크기
```

### Text Generation WebUI
```bash
# 자동 멀티GPU 지원
python server.py \
  --model llama-70b-q4_k_m.gguf \
  --gpu-memory 32 32 \  # 각 GPU 메모리
  --auto-devices
```

## 📊 벤치마크 결과 예상

| 모델 | 양자화 | Perplexity | 속도 (tok/s) | VRAM |
|------|--------|------------|--------------|------|
| Llama-3.1 70B | Q4_K_M | 5.2 | 25 | 40GB |
| Llama-3.1 70B | Q5_K_M | 5.1 | 22 | 45GB |
| Llama-3.1 70B | Q6_K | 5.0 | 18 | 55GB |
| Qwen2.5 72B | Q4_K_M | 5.3 | 24 | 42GB |
| Mixtral 8x7B | Q6_K | 5.4 | 40 | 35GB |
| Mixtral 8x22B | Q3_K_M | 5.6 | 15 | 58GB |

## 🌟 고급 설정

### CUDA 최적화
```bash
export CUDA_VISIBLE_DEVICES=0,1
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ARCH_LIST="7.0"  # V100
```

### 메모리 관리
```python
import torch
torch.cuda.set_per_process_memory_fraction(0.95)
torch.cuda.empty_cache()
```

### 모델 분할 전략
```python
# 레이어별 분할
device_map = {
    "model.embed_tokens": 0,
    "model.layers.0-39": 0,
    "model.layers.40-79": 1,
    "lm_head": 1
}
```
GPU 인식 ✅

Tesla V100-SXM2-32GB 2장 모두 정상 인식
각각 32GB 메모리 (31141 MB) 확인됨
PCI 버스 ID도 다르게 잘 할당됨 (89:00.0, 8a:00.0)


성능 수치 ✅

12.53 TFLOPS 달성 (V100 이론 성능 ~15 TFLOPS의 83%)
실제 사용 환경에서 이 정도면 매우 우수한 성능입니다


병렬 처리 ✅

Multi-GPU 전략 정상 작동 (2개 디바이스 동시 사용)
GPU 간 데이터 전송 속도 7.5 GB/s (양호)
두 GPU 모두 독립적으로 연산 수행 확인


CUDA 환경 ✅

TensorFlow 2.16.1 정상 작동
CUDA 지원 및 GPU 가속 모두 활성화
Compute Capability 7.0 (V100 정확함)