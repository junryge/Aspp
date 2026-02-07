# SK 하이닉스 도메인 지식 종합 정리

---

## 1. FAB 구조 및 사이트

- **운영 사이트**: 이천(Icheon), 청주(Cheongju), 우시(Wuxi) — 3개 반도체 제조 거점
- **주요 FAB 라인**: M10A, M14A, M14B, M16
- **핵심 분석 축**: **M16A(3F, 6F) ↔ M14** 간 물류 흐름이 가장 중요한 관리 대상

---

## 2. AMHS (Automated Material Handling System)

FAB 내 웨이퍼 카세트(FOUP)를 자동 운반하는 물류 시스템이다.

- **OHT(Overhead Hoist Transport)**: 천장 레일을 따라 이동하며 장비 간 FOUP를 운반
- **경로 구성요소**: Rail(일반 구간), Branch/Join(분기·합류), Conveyor(컨베이어), Transfer(이적 구간)
- **경로 탐색**: 다익스트라 알고리즘 기반으로 OHT 최적 경로를 결정
- **통신**: OHT는 UDP 기반으로 위치·상태 데이터를 실시간 송수신

### AMHS AI Agent 구조

- **SuperVisor Agent**가 상위에서 작업을 분배
- 하위 전문 에이전트: 위치 추적(Location Tracing), 유사도 검색(Embedding Search), 보고서 생성(Report Generation)

---

## 3. 급증(Surge) 개념

**급증이란**: M16A 3F의 현재 JOB 수(CURRENT_M16A_3F_JOB_2)가 **300 이상**으로 치솟는 현상

- 물류가 특정 구간에 과도하게 몰리면서 병목이 발생하는 상황
- 급증 전 **조기경보 기준은 280** — 이 수치를 넘으면 곧 급증이 올 수 있다는 신호
- 급증은 FAB 생산성에 직접 영향을 주므로 **10분 전 사전 예측**이 핵심 목표

---

## 4. 핵심 운영 지표 체계

### 4-1. Storage (저장 공간)

- **M16A_3F_STORAGE_UTIL**: M16A 3층 스토리지 사용률
- **FS Storage**: 사용량(USE), 전체 용량(TOTAL), 사용률(UTIL)로 세분화
- 스토리지가 가득 차면 FOUP를 더 이상 받을 수 없어 물류가 정체됨

### 4-2. Hub (허브룸)

- **HUBROOMTOTAL**: Hub Room 전체 수용 가능 수량
- Hub는 FOUP가 임시 대기하는 버퍼 역할
- Hub 여유 공간이 줄어들수록 급증 위험이 높아짐

### 4-3. Command (명령)

- **M16A_3F_CMD**: M16A 3F에서 발생하는 물류 명령 수
- **M16A_6F_TO_HUB_CMD**: M16A 6F에서 Hub로 향하는 명령 수
- **total_cmd**: 위 두 값의 합 — 전체 물류 부하를 나타냄
- 명령 수가 낮으면 오히려 물류 처리 능력이 부족하다는 의미 (역방향 지표)

### 4-4. Inflow / Outflow (유입·유출)

- 각 3개 지표씩 — FAB 구간별 물류 유입량과 유출량
- **net_flow = inflow - outflow** → 양수면 물류가 쌓이고 있다는 뜻

### 4-5. Max Capacity

- 구간별 최대 수용 가능량 2개 지표

---

## 5. 급증 위험 판정 체계

### 5-1. Hub 상태 등급

| 상태 | Hub Room 잔여량 | 의미 |
|---|---|---|
| **CRITICAL** | < 590 | 즉시 대응 필요 |
| **HIGH** | < 610 | 높은 위험 |
| **WARNING** | < 620 | 주의 |

### 5-2. 복합 위험 판정

- **Hub-Storage 복합 위험**: Hub < 610이면서 FS 사용률 ≥ 7 → 저장 공간과 버퍼 모두 부족
- **Hub-CMD 병목**: Hub < 610이면서 CMD < 220 → 버퍼 부족 + 명령 처리력 부족이 동시 발생

### 5-3. Storage 상태 등급

- FS 사용률 ≥ 7 → **HIGH**
- FS 사용률 ≥ 10 → **CRITICAL**
- Storage 사용률 ≥ 205 → **CRITICAL**
- Storage 사용률 ≥ 207 → **HIGH RISK**

### 5-4. Surge Risk Score (종합 위험 점수)

Hub 상태, Storage 상태, CMD 상태를 가중 합산한 종합 점수:

```
surge_risk_score = hub_high × 3 + storage_util_critical × 2 + total_cmd_low × 1 + storage_util_high × 1
```

| 점수 | 위험 등급 |
|---|---|
| ≥ 5 | **CRITICAL** |
| ≥ 3 | **HIGH** |
| ≥ 1 | **MEDIUM** |
| < 1 | **LOW** |

Hub 상태가 가중치 3으로 가장 크고, Storage Critical이 2, CMD와 Storage High가 각 1이다.

### 5-5. Surge Imminent (급증 임박 신호)

다음 3가지 조건이 **동시 충족**되면 급증 임박으로 판정:

1. 현재 JOB 수 > 280
2. 가속도(acceleration) > 0.5
3. Hub가 HIGH 상태

---

## 6. 물류 흐름의 인과관계 (도메인 해석)

급증이 발생하는 전형적 시나리오:

1. **Hub 여유 공간 감소** → FOUP 대기 버퍼 부족
2. **Storage 사용률 상승** → 저장 공간 포화
3. **CMD 감소** → 물류 처리 명령이 줄어 정체 심화
4. **Inflow > Outflow** → 유입이 유출보다 많아 물류 누적
5. **JOB 수 급등 → 300 돌파** = **Surge 발생**

핵심 인사이트: **Hub가 가장 선행 지표**이고(가중치 3), Storage와 CMD가 보조 지표 역할을 한다. 30분간의 추세를 보면 10분 뒤 급증 여부를 예측할 수 있다.

---

## 7. M16HUB ↔ M14 데이터 구조

- **AWS_IDC_DATA_HIS** 테이블에서 총 **65개 IDC 메트릭** 추출
  - M16HUB: 9개 지표
  - M14: 56개 지표
- 시간 기준(CRT_TM)으로 GROUP BY하여 시계열 구성
- 이 데이터가 M16 ↔ M14 간 물류 흐름 분석의 원천

---

## 8. 데이터 시간 구조

- **CURRTIME**: `yyyyMMdd HHmm` 형식
- 데이터에는 TOTALCNT와 함께 M14A/M10A/M14B/M16 조합 컬럼들이 포함 (SUM 포함)
- 시계열 분석의 기본 단위는 **분 단위**
