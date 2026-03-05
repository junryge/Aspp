# AMHS 최적 시간 예측 — 3모델 가이드

> RAIL-VEHICLEACQUIRESTARTED 데이터 기반  
> VEHICLE_ID · CARRIER_ID · TRANSFER_PORT → 최적 이송 시간 도출

---

## 모델 역할 분담

| 모델 | 유형 | 핵심 역할 |
|---|---|---|
| **Random Forest** | 회귀 / 분류 | 소요 시간 수치 예측 + Feature 중요도 해석 |
| **LSTM** | 시계열 딥러닝 | 연속 이벤트 흐름 패턴 → 다음 소요 시간 예측 |
| **Prophet** | 시계열 통계 | 포트별 혼잡 시간대 분석 → 최적 배차 시간 도출 |
| **K-Means** | 클러스터링 (보조) | 포트를 고/중/저 부하로 분류 → RF·LSTM 입력 Feature 주입 |

---

## 데이터 컬럼 → Feature 매핑

| 컬럼 | 예시 값 | 파생 Feature | 사용 모델 |
|---|---|---|---|
| MSG_TIME | 2026-02-23 16:29:40 | HOUR, DAY_OF_WEEK, SHIFT | RF / LSTM / Prophet |
| MACHINE_NAME | 4ACMD601 | FAB_ZONE, CMD_LEVEL | RF / LSTM |
| VEHICLE_ID | V456 | VHL_AVG_TIME, VHL_FAIL_CNT | RF / LSTM |
| CARRIER_ID | 4PDT4518 | CARRIER_TYPE, CARRIER_ZONE | RF |
| TRANSFER_PORT | 4ANZB101-324 | PORT_QUEUE_LEN, PORT_AVG_TIME, PORT_CLUSTER | RF / LSTM / Prophet / K-Means |
| CEID | 202 | EVENT_TYPE | RF |
| **ELAPSED_SEC** | COMPLETED − STARTED | — | 🎯 **TARGET** |

---

## 실행 파이프라인

```
STARTED ↔ COMPLETED 매핑
        ↓
   K-Means (포트 클러스터)
        ↓
Random Forest (소요 시간 예측)
        ↓
   LSTM (시퀀스 패턴 보정)
        ↓
  Prophet (혼잡 시간대 분석)
        ↓
  최적 배차 시간 도출
```

---

## 모델별 결과물

### Random Forest
- 예상 소요 시간 (초) 수치 출력
- 지연 확률 (%) 분류
- Feature Importance → 병목 원인 해석

### LSTM
- 직전 N개 이벤트 흐름 기반 다음 소요 시간 예측
- 포트 연속 점유 시 추가 지연 감지

### Prophet
- 포트별 혼잡 피크 시간대 시각화
- 권장 투입 시간 자동 산출
- 요일 / 교대 주기 자동 반영

---

## 다음 단계

1. STARTED ↔ COMPLETED 메시지 매핑 코드 작성
2. ELAPSED_SEC (Target) 컬럼 생성
3. Random Forest 베이스라인 학습
4. 결과 검증 후 LSTM / Prophet 순차 추가