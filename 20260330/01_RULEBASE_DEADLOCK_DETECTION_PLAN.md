# HID 구역 데드락 탐지 — 룰베이스 계획서

## 1. 개요

### 1.1 목적
M14A FAB AMHS OHT 운영 중 발생하는 HID 구역 데드락을 **룰베이스로 실시간 탐지**하고,
탐지 결과를 **라벨로 축적**하여 향후 ML 모델 학습 데이터로 활용한다.

### 1.2 역할
- 데드락 위험 구간 실시간 알림
- 데드락 라벨 자동 생성 (ML 학습용)
- 임계값 튜닝을 통한 점진적 정확도 향상

### 1.3 데이터 소스
- 테이블: `{FAB}_ATLAS_HID_INOUT`
- 주기: 1분 버퍼 플러시
- FAB: M14A (확장 가능: M14B, M16 등)

---

## 2. 데이터 컬럼 활용 정의

| 컬럼 | 타입 | 활용 목적 |
|---|---|---|
| `_time` | TIMESTAMP | 1분 윈도우 기준 |
| `VHL_ID` | VARCHAR | 차량 식별, V=OHT / R=AGV 구분 |
| `FROM_HIDID` | NUMBER | HID 유출 집계 |
| `TO_HIDID` | NUMBER | HID 유입 집계 |
| `HID_VALUE` | NUMBER | HID 내 체류 차량 수 |
| `FREE_FLOW_SPEED` | NUMBER | 차량 속도 → 정체 판단 |
| `VHL_COUNT_LIMIT` | NUMBER | HID 최대 수용 대수 (포화율 분모) |
| `VHL_PRECAUTION` | NUMBER | 주의 임계값 (OHT=35, AGV=3) |
| `TRANS_CNT` | NUMBER | 이동 카운트 → 유출 활성도 |
| `MCP_NM` | VARCHAR | MCP 구역 그룹핑 |

---

## 3. Phase 1 — HID 구역별 1분 윈도우 집계

### 3.1 집계 단위
- 시간: `TRUNC(_time, 'MI')` 1분 단위
- 공간: HID 구역 ID

### 3.2 집계 피처

| 피처명 | 산출 로직 | 의미 |
|---|---|---|
| `inflow_cnt` | `COUNT(TO_HIDID = X)` | HID X로 진입한 차량 수 |
| `outflow_cnt` | `COUNT(FROM_HIDID = X)` | HID X에서 출발한 차량 수 |
| `net_inflow` | `inflow_cnt - outflow_cnt` | 순유입 (양수 = 축적 중) |
| `saturation` | `MAX(HID_VALUE) / VHL_COUNT_LIMIT` | HID 포화율 (0.0 ~ 1.0+) |
| `avg_speed` | `AVG(FREE_FLOW_SPEED) WHERE FROM_HIDID=X` | 출발 차량 평균 속도 |
| `min_speed` | `MIN(FREE_FLOW_SPEED) WHERE FROM_HIDID=X` | 최저 속도 |
| `speed_ratio` | `avg_speed / baseline_speed` | 기준 대비 속도비 |
| `vhl_mix` | `1 IF both V* and R* exist` | OHT/AGV 혼재 여부 |
| `unique_vhl` | `COUNT(DISTINCT VHL_ID)` | 고유 차량 수 |
| `precaution_ratio` | `HID_VALUE / VHL_PRECAUTION` | 주의 임계 대비 비율 |

### 3.3 기준 속도 (baseline_speed)
- HID별 최근 1시간 이동평균 속도
- 초기: 전체 데이터 평균 → 1주 후 HID별 시간대 평균으로 갱신

---

## 4. Phase 2 — 5단계 룰 판정

### Rule 1: HID 포화도

| 조건 | 등급 | 점수 |
|---|---|---|
| `saturation ≥ 0.95` | CRITICAL | 4 |
| `saturation ≥ 0.85` | HIGH | 3 |
| `saturation ≥ 0.70` | MEDIUM | 2 |
| `saturation ≥ 0.50` | LOW | 1 |
| `< 0.50` | NORMAL | 0 |

### Rule 2: 속도 저하

| 조건 | 등급 | 점수 |
|---|---|---|
| `speed_ratio < 0.3` | CRITICAL | 4 |
| `speed_ratio < 0.5` | HIGH | 3 |
| `speed_ratio < 0.7` | MEDIUM | 2 |
| `< 0.7 이상` | NORMAL | 0 |

### Rule 3: 유입/유출 불균형

| 조건 | 등급 | 점수 |
|---|---|---|
| `net_inflow ≥ 5` AND `outflow_cnt = 0` | CRITICAL | 4 |
| `net_inflow ≥ 3` AND `outflow_cnt ≤ 1` | HIGH | 3 |
| `net_inflow ≥ 3` | MEDIUM | 2 |
| 그 외 | NORMAL | 0 |

### Rule 4: 정체 지속성 (시간축)

| 조건 | 등급 | 점수 |
|---|---|---|
| 직전 **3분 연속** `saturation ≥ 0.85` AND `speed_ratio < 0.5` | CRITICAL | 4 |
| 직전 **2분 연속** 위 조건 | HIGH | 3 |
| 직전 **1분** 위 조건 | MEDIUM | 1 |

### Rule 5: 연쇄 데드락 (공간축 — 인접 HID)

| 조건 | 등급 | 점수 |
|---|---|---|
| 인접 HID **2개 이상** 동시 CRITICAL | CHAIN_CRITICAL | 5 |
| 인접 HID **1개** CRITICAL + 본 HID HIGH 이상 | CHAIN_HIGH | 4 |
| 인접 HID **1개** HIGH 이상 | CHAIN_MEDIUM | 2 |

---

## 5. 종합 위험도 & 라벨 생성

### 5.1 점수 산출

```
deadlock_score = Rule1 + Rule2 + Rule3 + Rule4 + Rule5
```

### 5.2 등급 매핑

| 점수 | 등급 | 라벨값 | 의미 |
|---|---|---|---|
| ≥ 12 | `DEADLOCK_CONFIRMED` | 1.0 | 데드락 확정 |
| ≥ 8 | `DEADLOCK_IMMINENT` | 0.8 | 데드락 임박 |
| ≥ 5 | `DEADLOCK_WARNING` | 0.5 | 경고 |
| ≥ 3 | `DEADLOCK_WATCH` | 0.2 | 주의 |
| < 3 | `NORMAL` | 0.0 | 정상 |

### 5.3 라벨 테이블

```sql
CREATE TABLE HID_DEADLOCK_LABEL (
    LABEL_TM        TIMESTAMP,
    FAB_ID          VARCHAR(10),
    HID_ID          NUMBER,
    -- 집계 피처
    INFLOW_CNT      NUMBER,
    OUTFLOW_CNT     NUMBER,
    NET_INFLOW      NUMBER,
    SATURATION      NUMBER(5,4),
    AVG_SPEED       NUMBER(10,4),
    MIN_SPEED       NUMBER(10,4),
    SPEED_RATIO     NUMBER(5,4),
    UNIQUE_VHL      NUMBER,
    VHL_MIX         NUMBER(1),
    -- 룰 점수
    RULE1_SCORE     NUMBER,
    RULE2_SCORE     NUMBER,
    RULE3_SCORE     NUMBER,
    RULE4_SCORE     NUMBER,
    RULE5_SCORE     NUMBER,
    DEADLOCK_SCORE  NUMBER,
    RISK_LEVEL      VARCHAR(30),
    DEADLOCK_LABEL  NUMBER(3,1),
    CONSTRAINT PK_DL_LABEL PRIMARY KEY (LABEL_TM, FAB_ID, HID_ID)
);
```

---

## 6. 인접 HID 그래프 자동 구축

### 6.1 데이터 기반 구축 (LAYOUT.XML 불필요)

```python
# FROM_HIDID → TO_HIDID 이동 빈도 기반
adjacency = df.groupby(['FROM_HIDID', 'TO_HIDID']).size()
threshold = 10  # 1시간 내 10건 이상 이동
adjacent_pairs = adjacency[adjacency >= threshold].index.tolist()
```

### 6.2 산출물
- `hid_adjacency_map.json`: `{ "12": [11, 3, 42], "3": [12, 65, 4], ... }`
- 주기적 갱신 (일 1회)

### 6.3 검증
- LAYOUT.XML의 HID 물리적 위치와 대조 (가능한 경우)
- 이동 빈도 기반이므로 실제 운영 경로 반영

---

## 7. 알림 체계

### 7.1 TibRV 퍼블리시
- Subject: `HID_ZONE_STATUS`
- 조건: `DEADLOCK_WARNING` 이상 (score ≥ 5)

### 7.2 알림 메시지

```json
{
  "fab_id": "M14A",
  "hid_id": 12,
  "risk_level": "DEADLOCK_IMMINENT",
  "deadlock_score": 10,
  "saturation": 0.92,
  "speed_ratio": 0.35,
  "net_inflow": 4,
  "outflow_cnt": 0,
  "adjacent_risk": [
    {"hid_id": 11, "level": "HIGH"},
    {"hid_id": 3, "level": "MEDIUM"}
  ],
  "timestamp": "2026-04-02T07:59:00"
}
```

---

## 8. 라벨 검증 & 튜닝

### 8.1 검증 방법

| 항목 | 방법 |
|---|---|
| 오탐 확인 | IMMINENT 이상 발생 시점에 실제 OHT 정체 여부 (속도 로그, 현장 확인) |
| 미탐 확인 | 현장 데드락 보고 시점 역추적 → 라벨 존재 여부 |
| 분포 분석 | 1주차 전체 라벨 분포 → 임계값 적정성 판단 |

### 8.2 튜닝 대상

| 파라미터 | 초기값 | 튜닝 방향 |
|---|---|---|
| 포화율 CRITICAL 기준 | 0.95 | 데이터 분포 상위 5% 기준 |
| 속도비 CRITICAL 기준 | 0.3 | HID별 최저 속도 분포 하위 5% |
| net_inflow CRITICAL 기준 | 5 | HID별 유입 분포 상위 5% |
| 정체 지속 시간 | 3분 | 실제 데드락 지속 시간 관찰 후 |
| 인접 HID 빈도 threshold | 10건/시간 | 그래프 밀도에 따라 조정 |

### 8.3 목표 지표 (2주 후 평가)

```
Precision ≥ 0.6   (오탐 허용, 안전 우선)
Recall    ≥ 0.8   (미탐 최소화)
```

---

## 9. 일정

| 주차 | 작업 | 산출물 |
|---|---|---|
| 1주차 전반 | 1분 집계 엔진 개발 | Python 스크립트, 집계 테이블 |
| 1주차 후반 | 인접 HID 그래프 구축 | hid_adjacency_map.json |
| 2주차 전반 | 룰 판정 엔진 개발 | 5단계 룰 로직 |
| 2주차 후반 | 라벨 테이블 생성 + TibRV 연동 | DB 테이블, 알림 연동 |
| 3~4주차 | 라벨 축적 + 임계값 튜닝 | 튜닝 리포트, 확정 임계값 |

---

## 10. 기대 효과
- 라벨 없는 상태에서 즉시 운영 가능
- 데드락 위험 구간 실시간 모니터링
- 2~4주 라벨 축적 → ML 모델 학습 데이터 확보
- 임계값 튜닝으로 현장 맞춤 정확도 달성
