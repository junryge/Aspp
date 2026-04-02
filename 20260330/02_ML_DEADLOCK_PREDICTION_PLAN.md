# HID 구역 데드락 예측 — ML 모델 계획서

## 1. 개요

### 1.1 목적
룰베이스 라벨이 2~4주 축적된 후, **HID 구역별 5~10분 후 데드락 발생을 사전 예측**하는
ML 모델을 구축한다.

### 1.2 전제 조건
- 룰베이스 라벨 테이블 (`HID_DEADLOCK_LABEL`) 최소 2주 이상 축적 완료
- 라벨 검증 완료 (Precision ≥ 0.6, Recall ≥ 0.8 달성)
- 인접 HID 그래프 (`hid_adjacency_map.json`) 구축 완료

### 1.3 문제 특성

| 특성 | 설명 | 모델 선택에 미치는 영향 |
|---|---|---|
| **시계열** | 1분 단위 연속 데이터, 과거 N분 패턴이 미래 결정 | 순서 의존 모델 유리 |
| **공간/그래프** | HID 간 인접 관계, 연쇄 전파 패턴 | 그래프 구조 반영 필요 |
| **불균형** | 데드락 발생 비율 매우 낮음 (추정 1~5%) | 가중치/샘플링 전략 필수 |
| **실시간** | 1분 내 추론 완료 필요 | 경량 모델 or 배치 추론 |
| **해석성** | 현장 운영자 설명 필요 | 피처 중요도 추출 가능해야 |

---

## 2. 모델 후보 비교

### 2.1 후보 모델 목록

| # | 모델 | 유형 | 시계열 | 공간 | 해석성 | 추론속도 | 구현난이도 |
|---|---|---|---|---|---|---|---|
| 1 | **XGBoost** | 트리 앙상블 | △ (윈도우 피처) | △ (수동 피처) | ◎ | ◎ | 낮음 |
| 2 | **LightGBM** | 트리 앙상블 | △ (윈도우 피처) | △ (수동 피처) | ◎ | ◎ | 낮음 |
| 3 | **LSTM** | RNN 계열 | ◎ | △ (수동 연결) | △ | ○ | 중간 |
| 4 | **1D-CNN** | CNN 계열 | ○ | △ | △ | ◎ | 중간 |
| 5 | **Transformer** | Attention | ◎ | ○ (위치 인코딩) | △ | △ | 높음 |
| 6 | **GNN + LSTM** | 그래프+시계열 | ◎ | ◎ | × | △ | 높음 |
| 7 | **TabNet** | Attention 트리 | △ | △ | ○ | ○ | 중간 |

### 2.2 상세 분석

#### ① XGBoost / LightGBM (트리 기반)

**장점**
- 구현 간단, V7 서지 모델과 동일 프레임워크 재활용
- 피처 중요도 바로 확인 → 현장 설명 용이
- GPU 학습 지원 (V100 활용 가능)
- 결측치/이상치에 강건

**단점**
- 시계열 순서 정보를 직접 학습 못함 → 수동 윈도우 피처 필요
- HID 간 공간 관계를 명시적으로 넣어야 함
- "10분 전부터 서서히 나빠지는 패턴" 같은 건 slope 피처로 간접 표현

**적합 시나리오**: 빠른 프로토타입, 해석성 중요한 경우, 라벨 적을 때

---

#### ② LSTM (Long Short-Term Memory)

**장점**
- 시계열 순서를 직접 학습 (과거 N분 → 미래 예측)
- 서서히 악화되는 패턴, 갑자기 변하는 패턴 모두 포착
- HID별 독립 모델 or 전체 공유 모델 선택 가능

**단점**
- 피처 엔지니어링 대신 충분한 데이터 필요 (최소 4주 권장)
- 해석성 낮음 → Attention 추가로 보완 가능
- 학습 시간 길어질 수 있음

**적합 시나리오**: 시간 패턴이 핵심인 경우, 데이터 4주 이상 확보 후

**구조 예시**
```
Input: (batch, seq_len=10, features=N)  ← 과거 10분 윈도우
    │
    ▼
LSTM Layer (hidden=64, layers=2, bidirectional=True)
    │
    ▼
Attention Layer (어떤 시점이 중요한지)
    │
    ▼
FC → Sigmoid → P(deadlock in 5~10min)
```

---

#### ③ 1D-CNN

**장점**
- LSTM보다 빠른 학습/추론
- 로컬 패턴 (급격한 속도 저하, 갑작스런 유입 급증) 탐지에 강함
- 병렬 처리 효율적

**단점**
- 장기 의존성 약함 (커널 크기로 제한)
- LSTM보다 시계열 표현력 떨어짐

**적합 시나리오**: 실시간 추론 속도가 최우선인 경우

---

#### ④ Transformer (시계열 특화)

**장점**
- Self-Attention으로 어떤 시점의 어떤 피처가 중요한지 학습
- 장기 의존성 학습 가능
- 다변량 시계열에 강함

**단점**
- 데이터 많이 필요 (최소 4~8주)
- 구현 복잡도 높음
- 과적합 위험 (데이터 적을 때)

**적합 시나리오**: 데이터 충분히 쌓인 후 (8주+), 최종 고도화 단계

---

#### ⑤ GNN + LSTM (그래프 신경망 + 시계열)

**장점**
- HID 인접 관계를 그래프로 직접 모델링 → **연쇄 데드락 예측에 최강**
- 시계열(LSTM) + 공간(GNN) 동시 학습
- "HID 12가 막히면 11, 3도 막힌다" 같은 전파 패턴 학습

**단점**
- 구현 복잡도 가장 높음 (PyTorch Geometric 등 필요)
- 데이터 많이 필요
- 해석성 거의 없음
- 디버깅 어려움

**적합 시나리오**: 연쇄 데드락이 주요 문제인 경우, 8주+ 데이터, 연구 단계

**구조 예시**
```
각 HID 노드: 시계열 피처 (10분 윈도우)
    │
    ▼
LSTM per node → 시계열 임베딩
    │
    ▼
GCN/GAT Layer × 2 → 인접 HID 정보 교환
    │
    ▼
Node-level FC → P(deadlock) per HID
```

---

#### ⑥ TabNet

**장점**
- 트리 모델 수준 해석성 + 딥러닝 표현력
- 피처 선택을 자동으로 학습 (Sparse Attention)
- 표 형태 데이터에 최적화

**단점**
- 시계열 순서 학습 못함 (XGBoost와 동일 한계)
- 학습 불안정할 수 있음

**적합 시나리오**: XGBoost 대안으로 해석성 유지하면서 성능 올리고 싶을 때

---

## 3. 추천 전략: 단계별 모델 고도화

### 3.1 로드맵

```
[2~4주차: 라벨 축적]
        │
        ▼
[Phase A: Baseline]   ← LightGBM (빠른 프로토타입)
        │
        ▼
[Phase B: 시계열]     ← LSTM + Attention (시간 패턴 학습)
        │
        ▼
[Phase C: 공간+시계열] ← GNN + LSTM (연쇄 데드락, 충분한 데이터 확보 후)
        │
        ▼
[Phase D: 앙상블]     ← LightGBM + LSTM 앙상블 (실전 배포)
```

### 3.2 Phase별 상세

| Phase | 모델 | 필요 데이터 | 기간 | 목표 |
|---|---|---|---|---|
| A | LightGBM | 2주 | 1주 개발 | Recall ≥ 0.7, 빠른 검증 |
| B | LSTM + Attention | 4주 | 2주 개발 | Recall ≥ 0.8, 시계열 패턴 |
| C | GNN + LSTM | 8주+ | 3주 개발 | 연쇄 데드락 탐지 |
| D | 앙상블 | 4주+ | 1주 개발 | Recall ≥ 0.85, 실전 배포 |

### 3.3 왜 LightGBM을 Baseline으로?

XGBoost 대비:
- **학습 속도 2~5배 빠름** (Histogram 기반)
- 카테고리 피처 네이티브 지원 (HID_ID를 직접 넣기 가능)
- 메모리 효율 좋음
- 성능은 대부분 동등하거나 우위

→ V7은 XGBoost로 이미 운영 중이니, 새 프로젝트는 LightGBM으로 시작해서 비교하는 게 합리적.

---

## 4. 피처 설계

### 4.1 공통 피처 (모든 모델 공유)

#### 시간 윈도우 피처 (과거 10분)

| 그룹 | 피처 | 집계 |
|---|---|---|
| 포화도 | `saturation` | mean, max, min, slope, last |
| 속도 | `speed_ratio` | mean, min, slope, last |
| 유입유출 | `net_inflow` | mean, sum, max, last |
| 유출 | `outflow_cnt` | mean, sum, last |
| 차량밀도 | `unique_vhl` | mean, max, last |
| 혼재 | `vhl_mix` | sum (혼재 발생 횟수) |
| 룰점수 | `deadlock_score` | mean, max, last, slope |

#### 인접 HID 피처

| 피처 | 설명 |
|---|---|
| `adj_avg_saturation` | 인접 HID 평균 포화율 |
| `adj_max_saturation` | 인접 HID 최대 포화율 |
| `adj_avg_speed_ratio` | 인접 HID 평균 속도비 |
| `adj_min_speed_ratio` | 인접 HID 최소 속도비 |
| `adj_max_deadlock_score` | 인접 HID 최대 룰 점수 |
| `adj_critical_count` | 인접 HID 중 CRITICAL 수 |

#### 시간 피처

| 피처 | 설명 |
|---|---|
| `hour` | 시간대 (0~23) |
| `shift` | 교대 (주간/야간/심야) |
| `day_of_week` | 요일 |

### 4.2 LSTM/Transformer 전용

윈도우 피처 대신 **Raw 시계열 그대로 입력**:

```
Input shape: (batch, seq_len=10, features)

features per timestep:
  - saturation
  - speed_ratio
  - net_inflow
  - outflow_cnt
  - unique_vhl
  - vhl_mix
  - deadlock_score
  - adj_avg_saturation
  - adj_max_saturation
  - adj_min_speed_ratio
```

### 4.3 GNN 전용

```
Node features: 각 HID의 시계열 임베딩 (LSTM output)
Edge features: HID 간 이동 빈도, 평균 이동 시간
Graph: hid_adjacency_map.json 기반
```

---

## 5. 타겟 정의

### 5.1 Binary Classification (기본)

```python
# 향후 5~10분 내 해당 HID에서 DEADLOCK_IMMINENT 이상 발생 여부
target = 1 if max(label[t+5:t+10]) >= 0.8 else 0
```

### 5.2 Multi-class (고도화)

```python
# 향후 10분 내 최대 위험 등급
0: NORMAL
1: WATCH
2: WARNING
3: IMMINENT
4: CONFIRMED
```

### 5.3 Regression (선택)

```python
# 향후 10분 내 최대 deadlock_score 직접 예측
target = max(deadlock_score[t+1:t+10])
```

---

## 6. 불균형 처리 전략

| 방법 | 적용 모델 | 설명 |
|---|---|---|
| **클래스 가중치** | 전체 | 데드락 케이스 ×10 (V7 동일) |
| **Focal Loss** | LSTM, Transformer | 쉬운 샘플 가중치 줄이고 어려운 샘플 집중 |
| **SMOTE** | LightGBM | 소수 클래스 오버샘플링 (시계열 구조 주의) |
| **시간 기반 샘플링** | 전체 | 데드락 전후 ±30분 구간 집중 샘플링 |

---

## 7. 학습 & 평가

### 7.1 데이터 분할

```
시간 기반 분할 (절대 랜덤 셔플 금지):

|-------- Train (60%) --------|--- Val (20%) ---|--- Test (20%) ---|
|          2~3주차             |     4주차 전반    |    4주차 후반     |
```

### 7.2 평가 지표

| 지표 | 목표 | 우선순위 |
|---|---|---|
| **Recall** | ≥ 0.80 | ★★★ (미탐 최소화) |
| **Precision** | ≥ 0.60 | ★★ |
| **F1-Score** | ≥ 0.70 | ★★ |
| **FPR (오탐률)** | ≤ 0.10 | ★ |
| **조기 경보 시간** | 평균 ≥ 5분 전 | ★★★ |

### 7.3 모델 비교 프레임워크

```python
results = {}
for model_name, model in models.items():
    preds = model.predict(X_test)
    results[model_name] = {
        'recall': recall_score(y_test, preds),
        'precision': precision_score(y_test, preds),
        'f1': f1_score(y_test, preds),
        'avg_early_warning_min': calc_early_warning(y_test, preds),
        'inference_ms': measure_inference_time(model, X_test[:1])
    }
```

---

## 8. 배포 고려사항

### 8.1 추론 환경

| 항목 | 요구사항 |
|---|---|
| 추론 주기 | 1분 (집계 주기와 동일) |
| 추론 시간 | < 10초 (전체 HID 동시) |
| 하드웨어 | 내부 GPU (V100) or CPU |
| 폐쇄망 | 외부 API 불가, 모델 로컬 배포 |

### 8.2 모델 서빙

```
[1분 집계 완료]
    │
    ▼
[피처 생성 (10분 윈도우)]
    │
    ├── LightGBM: pickle 로드 → predict
    ├── LSTM: PyTorch 모델 로드 → forward
    └── 앙상블: 가중 평균
    │
    ▼
[예측 결과 DB 저장 + TibRV 알림]
```

### 8.3 모델 갱신
- 주기: 주 1회 재학습 (최근 2주 데이터)
- A/B 테스트: 신규 모델 vs 현재 모델 병렬 운영 후 전환

---

## 9. 일정 (룰베이스 라벨 확보 후)

| 주차 | 작업 | 산출물 |
|---|---|---|
| **5주차** | Phase A: LightGBM 베이스라인 | 모델, 평가 리포트 |
| **6주차** | Phase A 튜닝 + 배포 | 운영 모델 v1 |
| **7~8주차** | Phase B: LSTM + Attention | 모델, 비교 리포트 |
| **9주차** | Phase B vs A 비교 → 우수 모델 배포 | 운영 모델 v2 |
| **10~12주차** | Phase C: GNN + LSTM (선택) | 연구 리포트 |
| **12주차** | Phase D: 앙상블 최종 배포 | 운영 모델 v3 |

---

## 10. 리스크 & 대응

| 리스크 | 영향 | 대응 |
|---|---|---|
| 라벨 품질 낮음 | 모든 ML 모델 성능 저하 | 룰베이스 튜닝 우선, 현장 검증 강화 |
| 데드락 발생 빈도 극히 낮음 | 학습 데이터 부족 | Focal Loss + 시간 기반 샘플링, 기간 연장 |
| HID 구조 변경 (레이아웃 변경) | 인접 그래프 무효화 | 자동 그래프 갱신 (일 1회), 모델 재학습 |
| 실시간 추론 지연 | 알림 늦음 | 경량 모델 우선 배포 (LightGBM), GPU 추론 |
| 폐쇄망 패키지 설치 제한 | PyTorch/PyG 설치 불가 | Phase A는 sklearn/lightgbm만, Phase B 이후 패키지 사전 확보 |

---

## 11. 기대 효과

| Phase | 효과 |
|---|---|
| A (LightGBM) | 룰베이스 대비 5분 조기 예측, 오탐 감소 |
| B (LSTM) | 시계열 패턴 기반 정밀 예측, 서서히 악화되는 케이스 탐지 |
| C (GNN) | 연쇄 데드락 전파 경로 예측 |
| D (앙상블) | 단일 모델 대비 안정적 성능, 다양한 패턴 커버 |
