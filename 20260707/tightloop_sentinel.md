# TightLoop Sentinel 이란 무엇인가

> Chronos-Bolt 같은 예측 모델의 **"예측 → 행동(forecast-to-action)"** 계층을 담당하는
> **뉴로모픽 신경망 엔진**. 예측 분포를 실제 운영 조치로 변환한다.

출처: *Operational Synergy Between Chronos-Bolt and TightLoop Sentinel* (LEE, Kyuchul · cording.ai, Zenodo 프리프린트, 2026-07-09, DOI 10.5281/zenodo.21280208, CC-BY-4.0)

---

## 1. 왜 필요한가 — 문제 설정

시계열 파운데이션 모델(Chronos-Bolt 등)은 **zero-shot 확률 예측**을 잘 뽑는다. 그런데 실제 운영 시스템은 "예측값"만으로는 부족하다. 필요한 건 **경계가 있는 실제 조치**다:

- 예비 물량 할당 (reserve allocation)
- 경보 격상 (alert escalation)
- 버퍼 조정 (buffer adjustment)
- 캐파 스테이징 (capacity staging)

**TightLoop Sentinel**은 이 간극 — 예측 분포를 조치로 바꾸는 **forecast-to-action 계층** — 을 메우는 엔진이다.

---

## 2. 정체 — 한 문단

TightLoop Sentinel은 **뉴로모픽 신경망 엔진(neuromorphic neural network engine)** 이다. 입력으로 (1) 예측 분포와 (2) 최근 리플레이 피드백을 받아서, **인과적(causal)이고 경계가 있는(bounded) 운영 조정**으로 변환한다. 핵심은 예측 모델 자체를 개선하는 게 아니라, **예측→행동 사이 계층**을 개선한다는 점이다.

---

## 3. 역할 분담 (핵심 컨셉)

| 계층 | 담당 | 하는 일 |
|---|---|---|
| **예측** | Chronos-Bolt | 미래 값의 **분포**를 추정 (quantile 예측) |
| **행동** | TightLoop Sentinel | 그 분포를 **보수적·인과적·운영 지향 조치**로 변환 |

> 논문의 명시적 주장: TightLoop Sentinel은 **예측 모델의 정확도를 올리는 게 아니다.**
> undercoverage(과소 커버리지), shortfall(부족분), reserve waste(예비 낭비), action churn(조치 요동)처럼
> **직접 운영 비용이 걸린 forecast-to-action 계층**을 개선하는 것이다.

---

## 4. 실험 설정

- **예측 모델**: Chronos-Bolt **Base** (주 모델), Chronos-Bolt **Tiny** (비교용)
- **실행 환경**: **Jetson Orin CUDA** — 엣지 AI 환경
- **벤치마크**: GIFT-Eval 기반 4개 태스크 리플레이
  1. **BizITObs** — 애플리케이션 텔레메트리
  2. **Electricity** — 전력 수요
  3. **Jena Weather** — 기상
  4. **Bitbrains** — fast storage 트레이스
- **구현 디테일**: sentinel 로더는 **Rust**로 작성. Bitbrains에서 non-finite 라벨 지평선 문제가 발생 → 트레이스 생성 시 non-finite 행 제거 + Rust 로더에 finite-input 가드 추가로 처리.

---

## 5. 결과 (baseline_tightloop, 기본 액추에이터)

### 운영 커버리지 개선 (Chronos-Bolt Base 구간 대비, %p)

| 데이터셋 | 커버리지 개선 |
|---|---|
| BizITObs | **+32.89 %p** |
| Jena Weather | **+11.50 %p** |
| Electricity | **+10.07 %p** |
| Bitbrains | **+4.14 %p** |

→ **4개 데이터셋 전부에서 커버리지 개선.**

### 평균 운영 비용 감소

| 데이터셋 | 비용 감소 |
|---|---|
| BizITObs | **-56.45%** |
| Jena Weather | **-7.85%** |
| Electricity | **-4.26%** |

> Bitbrains는 near-zero interval-width 아웃라이어가 지배적이었지만, 수정 후 non-finite 값이 사라졌고 커버리지는 여전히 개선됨.

---

## 6. 한눈에 정리

- **무엇**: 예측 → 행동 계층을 담당하는 뉴로모픽 신경망 엔진
- **짝**: Chronos-Bolt(예측) + TightLoop Sentinel(행동)
- **하는 일**: 예측 분포 + 리플레이 피드백 → 보수적·경계 있는 운영 조치
- **개선 대상**: 예측 정확도가 아니라 **forecast-to-action 계층**(커버리지·부족분·예비낭비·조치요동)
- **검증**: Jetson Orin 엣지, GIFT-Eval 4개 태스크, 커버리지 전부 개선 + 비용 감소
- **재현성**: edge-oriented, 재현 가능한 평가로 설계됨

---

## 참고

- 논문: LEE, Kyuchul & cording.ai, *Operational Synergy Between Chronos-Bolt and TightLoop Sentinel — A Reproducible Edge-Oriented Evaluation of a Neuromorphic Neural Network Engine for Forecast-to-Action Adaptation*, Zenodo, 2026.
- DOI: 10.5281/zenodo.21280208
- 라이선스: CC-BY-4.0
- 키워드: time-series forecasting, Chronos-Bolt, GIFT-Eval, neuromorphic neural network engine, forecast-to-action adaptation, probabilistic forecasting, operational evaluation, edge AI, Jetson Orin

> 주의: 본 정리는 Zenodo 레코드의 **초록·메타데이터** 기준. 전문(.md) 본문은 서버가 바이너리로 제공해 상세 아키텍처 내부 구조까지는 확보하지 못함. 세부 구현은 원문 확인 필요.
