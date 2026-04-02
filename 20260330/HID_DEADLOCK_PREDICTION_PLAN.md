# HID 구역 기반 데드락 예측 시스템 계획서

## 1. 개요

### 1.1 목적
M14A FAB AMHS OHT 운영 중 발생하는 **HID 구역 데드락**을 사전 탐지·예측하여,
물류 정체로 인한 FAB 가동률 저하를 방지한다.

### 1.2 배경
- 데드락 발생 시 OHT 정체 → Lot 이동 지연 → FAB 생산성 직결
- 현재 데드락 발생 이력(라벨) 없음 → **룰베이스로 라벨 생성 후 ML 전환** 전략
- 데이터 소스: `{FAB}_ATLAS_HID_INOUT` 테이블 (1분 버퍼 플러시)

### 1.3 데이터 소스 컬럼 정의

| 컬럼 | 설명 | 활용 |
|---|---|---|
| `_time` | 플러시 시각 (1분 주기) | 시계열 기준 |
| `VHL_ID` | OHT/AGV 차량 ID | V=OHT, R=AGV 구분 |
| `FROM_HIDID` | 출발 HID 구역 번호 | 유출 집계 |
| `TO_HIDID` | 도착 HID 구역 번호 | 유입 집계 |
| `HID_VALUE` | HID 내 체류 차량 수 | 포화도 산출 |
| `FREE_FLOW_SPEED` | 자유 흐름 속도 | 정체 판단 |
| `VHL_COUNT_LIMIT` | HID 최대 수용 대수 | 포화율 분모 |
| `VHL_PRECAUTION` | 주의 임계값 | OHT(35) vs AGV(3) |
| `TRANS_CNT` | 이동 카운트 | 유출 활성도 |
| `MCP_NM` | MCP 이름 | 구역 그룹핑 |

---

## 2. 시스템 아키텍처

```
[ATLAS HID_INOUT 테이블]
        │ (1분 주기)
        ▼
[Phase 1: 1분 윈도우 집계 엔진]
   HID별 inflow / outflow / avg_speed / saturation
        │
        ▼
[Phase 2: 룰베이스 판정 엔진]
   5단계 위험도 산출 → 데드락 라벨 자동 생성
        │
        ├──▶ [알림] TibRV HID_ZONE_STATUS 퍼블리시
        │
        ▼
[Phase 3: 라벨 축적 DB]
   HID_DEADLOCK_LABEL 테이블
        │ (2~4주 축적)
        ▼
[Phase 4: ML 예측 모델]
   HID 구역별 데드락 N분 전 사전 예측
```

---

## 3. Phase 1 — HID 구역별 1분 윈도우 집계

### 3.1 집계 단위
- **시간**: 1분 윈도우 (`_time` 기준 TRUNC)
- **공간**: HID 구역 ID (FROM_HIDID, TO_HIDID에서 추출)

### 3.2 집계 피처

| 피처명 | 산출 로직 | 의미 |
|---|---|---|
| `inflow_cnt` | `COUNT(TO_HIDID = X)` | 해당 HID로 진입한 차량 수 |
| `outflow_cnt` | `COUNT(FROM_HIDID = X)` | 해당 HID에서 출발한 차량 수 |
| `net_inflow` | `inflow_cnt - outflow_cnt` | 순유입 (양수=축적) |
| `saturation` | `MAX(HID_VALUE) / VHL_COUNT_LIMIT` | HID 포화율 |
| `avg_speed` | `AVG(FREE_FLOW_SPEED) WHERE FROM_HIDID=X` | 출발 차량 평균 속도 |
| `min_speed` | `MIN(FREE_FLOW_SPEED) WHERE FROM_HIDID=X` | 최저 속도 (병목 심각도) |
| `speed_ratio` | `avg_speed / baseline_speed` | 기준 속도 대비 비율 |
| `vhl_mix` | OHT/AGV 혼재 여부 | V+R 동시 존재 시 1 |
| `trans_sum` | `SUM(TRANS_CNT)` | 총 이동 횟수 |
| `unique_vhl` | `COUNT(DISTINCT VHL_ID)` | 고유 차량 수 |

### 3.3 기준 속도 (baseline_speed)
- HID별 최근 **1시간 이동평균** 속도를 기준선으로 사용
- 초기 운영 시 전체 데이터 평균으로 시작, 점진적 갱신

---

## 4. Phase 2 — 룰베이스 데드락 판정 & 라벨 생성

### 4.1 5단계 판정 룰

#### Rule 1: HID 포화도

| 조건 | 등급 | 점수 |
|---|---|---|
| `saturation ≥ 0.95` | CRITICAL | 4 |
| `saturation ≥ 0.85` | HIGH | 3 |
| `saturation ≥ 0.70` | MEDIUM | 2 |
| `saturation ≥ 0.50` | LOW | 1 |
| `saturation < 0.50` | NORMAL | 0 |

#### Rule 2: 속도 저하

| 조건 | 등급 | 점수 |
|---|---|---|
| `speed_ratio < 0.3` | CRITICAL | 4 |
| `speed_ratio < 0.5` | HIGH | 3 |
| `speed_ratio < 0.7` | MEDIUM | 2 |

#### Rule 3: 유입/유출 불균형

| 조건 | 등급 | 점수 |
|---|---|---|
| `net_inflow ≥ 5` AND `outflow_cnt = 0` | CRITICAL | 4 |
| `net_inflow ≥ 3` AND `outflow_cnt ≤ 1` | HIGH | 3 |
| `net_inflow ≥ 3` | MEDIUM | 2 |

#### Rule 4: 정체 지속성 (시간축)

| 조건 | 등급 | 점수 |
|---|---|---|
| 직전 3분 연속 `saturation ≥ 0.85` AND `speed_ratio < 0.5` | CRITICAL | 4 |
| 직전 2분 연속 위 조건 | HIGH | 3 |

#### Rule 5: 연쇄 데드락 (공간축)

| 조건 | 등급 | 점수 |
|---|---|---|
| 인접 HID 2개 이상 동시 CRITICAL | CHAIN_CRITICAL | 5 |
| 인접 HID 1개 CRITICAL + 본 HID HIGH | CHAIN_HIGH | 4 |

### 4.2 종합 위험도 산출

```
deadlock_score = Rule1 + Rule2 + Rule3 + Rule4 + Rule5

 ≥ 12  →  DEADLOCK_CONFIRMED   (라벨: 1.0)
 ≥ 8   →  DEADLOCK_IMMINENT    (라벨: 0.8)
 ≥ 5   →  DEADLOCK_WARNING     (라벨: 0.5)
 ≥ 3   →  DEADLOCK_WATCH       (라벨: 0.2)
 < 3   →  NORMAL               (라벨: 0.0)
```

### 4.3 라벨 테이블 스키마

```sql
CREATE TABLE HID_DEADLOCK_LABEL (
    LABEL_TM        TIMESTAMP,       -- 판정 시각 (1분 단위)
    FAB_ID          VARCHAR(10),      -- M14A
    HID_ID          NUMBER,           -- HID 구역 번호
    SATURATION      NUMBER(5,4),      -- 포화율
    SPEED_RATIO     NUMBER(5,4),      -- 속도비
    NET_INFLOW      NUMBER,           -- 순유입
    OUTFLOW_CNT     NUMBER,           -- 유출 수
    RULE1_SCORE     NUMBER,
    RULE2_SCORE     NUMBER,
    RULE3_SCORE     NUMBER,
    RULE4_SCORE     NUMBER,
    RULE5_SCORE     NUMBER,
    DEADLOCK_SCORE  NUMBER,           -- 종합 점수
    RISK_LEVEL      VARCHAR(30),      -- 등급명
    DEADLOCK_LABEL  NUMBER(3,1),      -- 0.0 ~ 1.0
    CONSTRAINT PK_DEADLOCK_LABEL PRIMARY KEY (LABEL_TM, FAB_ID, HID_ID)
);
```

---

## 5. Phase 3 — 라벨 축적 & 검증

### 5.1 축적 기간
- **최소 2주**: 룰 튜닝 및 오탐/미탐 분석
- **목표 4주**: ML 학습 가능 데이터셋 확보

### 5.2 검증 방법

| 항목 | 방법 |
|---|---|
| 오탐 확인 | DEADLOCK_IMMINENT 이상 발생 시점 전후 실제 OHT 정체 여부 확인 |
| 미탐 확인 | 현장에서 데드락 발생 보고된 시점에 라벨이 찍혔는지 역추적 |
| 임계값 튜닝 | 1주차 데이터 기반 포화율/속도비 분포 분석 → 임계값 조정 |
| HID 인접 맵 | LAYOUT.XML 기반 HID 인접 관계 그래프 구축 → Rule 5 정확도 향상 |

### 5.3 튜닝 지표

```
Precision = 실제 데드락 / (DEADLOCK_IMMINENT 이상 판정 수)
Recall    = 감지된 데드락 / (실제 데드락 발생 수)

목표: Precision ≥ 0.6, Recall ≥ 0.8 (미탐보다 오탐이 나은 방향)
```

---

## 6. Phase 4 — ML 예측 모델 (라벨 확보 후)

### 6.1 목표
- HID 구역별 **5~10분 후** 데드락 발생 여부 예측
- V7 서지 예측 모델과 동일한 시계열 윈도우 접근

### 6.2 피처 설계

| 그룹 | 피처 | 윈도우 |
|---|---|---|
| 포화도 | saturation mean/max/slope | 최근 10분 |
| 속도 | speed_ratio mean/min/slope | 최근 10분 |
| 유입유출 | net_inflow mean/sum, outflow_cnt | 최근 10분 |
| 차량밀도 | unique_vhl, vhl_mix | 최근 5분 |
| 룰 점수 | deadlock_score mean/max/trend | 최근 10분 |
| 인접 HID | 인접 HID 평균 saturation, speed_ratio | 최근 5분 |

### 6.3 모델
- **XGBoost** (V7 모델과 동일 프레임워크)
- Binary classification: DEADLOCK_LABEL ≥ 0.8 → 1, else 0
- 서지 가중치 방식 동일 적용 (데드락 케이스 ×10)

### 6.4 평가
- 타겟 Detection Rate: **70%** (Recall)
- 오탐률: **30% 이하** (Precision ≥ 0.7)

---

## 7. 인접 HID 그래프 구축

### 7.1 데이터 기반 자동 구축
LAYOUT.XML 없이도 HID_INOUT 데이터에서 추출 가능:

```
인접 관계 = FROM_HIDID → TO_HIDID 이동이 빈번한 쌍
조건: 1시간 내 해당 쌍의 이동 건수 ≥ 10
```

### 7.2 그래프 활용
- Rule 5 (연쇄 데드락) 판정 시 인접 HID 참조
- 데드락 전파 경로 시각화
- ML 피처로 인접 HID 상태 포함

---

## 8. 알림 체계

### 8.1 TibRV 퍼블리시
- Subject: `HID_ZONE_STATUS` (기 설계된 4개 Subject 중 하나)
- 조건: `DEADLOCK_WARNING` 이상

### 8.2 알림 내용

```json
{
  "fab_id": "M14A",
  "hid_id": 12,
  "risk_level": "DEADLOCK_IMMINENT",
  "deadlock_score": 10,
  "saturation": 0.92,
  "speed_ratio": 0.35,
  "net_inflow": 4,
  "adjacent_risk": ["HID_11: HIGH", "HID_3: MEDIUM"],
  "timestamp": "2026-04-02T07:59:00"
}
```

---

## 9. 일정

| 주차 | 작업 | 산출물 |
|---|---|---|
| **1주차** | Phase 1: 1분 집계 엔진 개발 | 집계 테이블, Python 스크립트 |
| **1주차** | HID 인접 그래프 자동 구축 | adjacency_map.json |
| **2주차** | Phase 2: 룰베이스 판정 엔진 | 판정 로직, 라벨 테이블 |
| **2주차** | 알림 연동 (TibRV) | HID_ZONE_STATUS 퍼블리시 |
| **3~4주차** | Phase 3: 라벨 축적 & 임계값 튜닝 | 튜닝 리포트 |
| **5~6주차** | Phase 4: ML 모델 학습 & 평가 | XGBoost 모델, 평가 결과 |

---

## 10. 기대 효과

- **즉시**: 룰베이스로 데드락 위험 구간 실시간 모니터링
- **4주 후**: 축적된 라벨로 ML 예측 모델 전환, 5~10분 사전 예측
- **장기**: 데드락 발생률 감소 → AMHS 물류 효율성 향상 → FAB 가동률 개선
