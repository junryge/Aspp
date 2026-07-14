# Chronos-Bolt 란 무엇인가

> Amazon이 만든 **시계열 예측 파운데이션 모델(Time-Series Foundation Model)**.
> 한 번 학습해두면, 새 시계열에 파인튜닝 없이 바로 붙여 예측을 뽑는 zero-shot 모델이다.

---

## 1. 정체 — 한 문단 요약

Chronos-Bolt는 **사전학습된 시계열 예측 모델 패밀리**다. NLP의 언어모델이 대량 텍스트로 학습해 처음 보는 문장도 처리하듯, Chronos-Bolt는 **약 1,000억 개 시계열 관측치**로 학습해서 학습 때 본 적 없는 시계열에도 예측을 생성한다. 이걸 zero-shot 예측이라고 한다. 구조는 구글의 **T5 encoder-decoder**를 기반으로 하고, 원본 Chronos를 속도·메모리 측면에서 다시 설계한 개선판이다.

---

## 2. 계보 — Chronos 패밀리 안에서의 위치

| 세대 | 이름 | 핵심 방식 | 특징 |
|---|---|---|---|
| 1세대 | **Chronos** (원본) | 시계열을 토큰으로 변환 → LLM처럼 미래 궤적을 **샘플링**(autoregressive) | "시계열의 언어를 배운다" 컨셉. 느리고 무거움 |
| 2세대 | **Chronos-Bolt** | 시계열을 **패치**로 묶어 인코딩 → **한 번에** 여러 스텝 quantile 생성 | 250배 빠르고 20배 가벼움 |
| 3세대 | **Chronos-2** | 후속 최신판 | HuggingFace에 별도 공개 (`amazon/chronos-2`) |

원본 논문은 *"Chronos: Learning the Language of Time Series"* (Ansari et al., 2024, TMLR).

---

## 3. 원본 Chronos vs Chronos-Bolt — 뭐가 바뀌었나

| 구분 | 원본 Chronos | Chronos-Bolt |
|---|---|---|
| 입력 처리 | 관측치를 개별 토큰으로 양자화 | 여러 관측치를 **패치**로 묶어 입력 |
| 생성 방식 | autoregressive 샘플링 (한 스텝씩) | **direct multi-step** (여러 스텝 동시 생성) |
| 출력 | 샘플 경로에서 분포 추정 | **quantile 값 직접 출력** |
| 속도 | 기준 | **최대 250배** 빠름 |
| 메모리 | 기준 | **20배** 효율적 |

핵심 포인트: **Chronos-Bolt (Base)** 는 **원본 Chronos (Large)** 보다 정확도가 높으면서 **600배 이상** 빠르다. 즉, 작은 모델이 더 큰 원본을 이긴다.

---

## 4. 동작 원리

```
과거 시계열 (context)
      │
      ▼
[패치로 분할]  ← 여러 관측치를 묶음 단위로
      │
      ▼
[T5 Encoder]  → 시계열 표현(representation) 학습
      │
      ▼
[T5 Decoder]  → 여러 미래 스텝의 quantile 예측을 "직접" 생성
      │
      ▼
예측 결과: [num_series, num_quantiles, prediction_length]
```

1. **패치화**: 과거 시계열(context)을 관측치 여러 개 묶음인 패치로 자른다.
2. **인코딩**: 패치를 T5 encoder에 넣어 표현을 학습한다.
3. **직접 다중 스텝 예측**: decoder가 미래 여러 스텝의 **quantile 예측**을 한 번에 뽑는다. 스텝마다 반복(autoregressive)하지 않아 빠르다.
4. **확률적 출력**: 점 추정 하나가 아니라 여러 분위수(예: 0.1 / 0.5 / 0.9)를 줘서 불확실성 구간까지 표현한다.

---

## 5. 모델 크기 (4종)

| 모델 | 파라미터 | 베이스 |
|---|---|---|
| `chronos-bolt-tiny` | 9M | t5-efficient-tiny |
| `chronos-bolt-mini` | 21M | t5-efficient-mini |
| `chronos-bolt-small` | 48M | t5-efficient-small |
| `chronos-bolt-base` | 205M | t5-efficient-base |

- 라이선스: **Apache-2.0**
- 포맷: **safetensors** (base 기준 약 0.2B 파라미터, F32)
- **CPU/GPU 모두 실행 가능** — 16GB GPU가 필수는 아니다. Apple Silicon(mps)도 지원.

---

## 6. 성능 (벤치마크 기준)

- 벤치마크 조건: 1,024개 시계열, context length 512, 예측 구간 64스텝
- 지표: **WQL**(Weighted Quantile Loss, 확률적 예측), **MASE**(Mean Absolute Scaled Error, 점 예측)
- **27개 데이터셋** 집계 기준
- 학습 때 그 데이터셋을 전혀 안 봤는데도(zero-shot), 해당 데이터로 **직접 학습된 통계 모델·딥러닝 모델을 능가**
- 일부 데이터셋에 사전노출된 다른 파운데이션 모델보다도 우수

---

## 7. 사용법

### (A) AutoGluon — 프로덕션 권장 경로

파인튜닝, covariate(외생변수) 반영, 다른 모델과 앙상블까지 가능.

```python
# pip install autogluon
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

df = TimeSeriesDataFrame("....../m4_hourly/train.csv")

predictor = TimeSeriesPredictor(prediction_length=48).fit(
    df,
    hyperparameters={
        "Chronos": {"model_path": "amazon/chronos-bolt-base"},
    },
)
predictions = predictor.predict(df)
```

### (B) chronos-forecasting — 연구/최소 인터페이스

```python
# pip install chronos-forecasting
import pandas as pd, torch
from chronos import BaseChronosPipeline

pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-bolt-base",
    device_map="cuda",     # "cpu" 또는 "mps"(Apple Silicon)
    torch_dtype=torch.bfloat16,
)

df = pd.read_csv(".../AirPassengers.csv")

# 출력 shape: [num_series, num_quantiles, prediction_length]
forecast = pipeline.predict(
    context=torch.tensor(df["#Passengers"]),
    prediction_length=12,
)
```

### (C) SageMaker JumpStart — 엔드포인트 배포

```python
# pip install -U sagemaker
from sagemaker.jumpstart.model import JumpStartModel

model = JumpStartModel(
    model_id="autogluon-forecasting-chronos-bolt-base",
    instance_type="ml.c5.2xlarge",   # CPU/GPU 인스턴스 모두 가능
)
predictor = model.deploy()
```

> 2025-02-14 업데이트로 SageMaker JumpStart에서 몇 줄 코드로 프로덕션 엔드포인트 배포가 가능해졌다. covariate 예측도 지원.

---

## 8. 접근 경로 요약

- **HuggingFace**: `amazon/chronos-bolt-base` (그 외 tiny/mini/small)
- **AutoGluon-TimeSeries**: 파인튜닝·covariate·앙상블 지원
- **AWS SageMaker JumpStart**: CPU/GPU 엔드포인트 배포
- **Amazon Bedrock Marketplace**: Bedrock API로 호출, Agents 등과 연동
- **chronos-forecasting** (GitHub): 연구용 최소 라이브러리

---

## 9. 한눈에 정리

- **무엇**: T5 기반 시계열 예측 파운데이션 모델
- **왜 씀**: 학습 없이 새 시계열에 바로 예측(zero-shot), 게다가 빠름
- **강점**: 원본 대비 250배 속도 / 20배 메모리 / 27개 데이터셋에서 정확도 우위
- **출력**: quantile 기반 확률적 예측 (불확실성 구간 포함)
- **비용/환경**: 오픈소스(Apache-2.0), CPU에서도 구동
- **후속**: `amazon/chronos-2`가 최신판

---

## 참고

- HuggingFace 모델카드: `amazon/chronos-bolt-base`
- 논문: Ansari et al., *Chronos: Learning the Language of Time Series*, TMLR 2024 (arXiv:2403.07815)
- 기반: T5 (Raffel et al., arXiv:1910.10683)
- AWS ML Blog — Zero-shot forecasting with Chronos-Bolt and AutoGluon
