# ASAS ALT/src 알람 시스템 전수 분석 보고서

> 분석 대상: `/ALT/src` 폴더 전체 (39개+ 파일)
> 분석 일자: 2026-03-23

---

## 목차

1. [알람 타입 및 카테고리](#1-알람-타입-및-카테고리)
2. [알람 데이터 구조](#2-알람-데이터-구조)
3. [알람 발생 로직 및 트리거 조건](#3-알람-발생-로직-및-트리거-조건)
4. [알람 알림 메커니즘](#4-알람-알림-메커니즘)
5. [알람 심각도 레벨](#5-알람-심각도-레벨)
6. [배치 프로세스별 알람 생성](#6-배치-프로세스별-알람-생성)
7. [알람 시스템 흐름도](#7-알람-시스템-흐름도)
8. [설정 파일 및 유틸리티 클래스](#8-설정-파일-및-유틸리티-클래스)
9. [알람 해제 메커니즘](#9-알람-해제-메커니즘)

---

## 1. 알람 타입 및 카테고리

시스템은 `SEND_SUB_SUBJECT` (TibrvService) 기준 **6가지 주요 알람 타입**을 정의합니다.

### 1.1 HID_OFF (Hidden Instruction Device 다운) - 심각도 Level 1 (CRITICAL)
- **파일**: `data/HidOffRecordItem.java`
- **트리거**: HID(Hidden Instruction Device)가 오프라인 전환 시
- **핵심 필드**: `hidId`, `alarmCode` (형식: "HID###"), `state`, `errorCode`

### 1.2 VHL_OFF (Vehicle 다운) - 심각도 Level 2 (HIGH)
- **파일**: `data/VhlOffRecordItem.java`
- **트리거**: 차량(VHL)이 정지하거나 오프라인 전환 시
- **핵심 필드**: `machineId`, `errorCode`, `stoppedFromAddress`, `stoppedToAddress`, `velocity`

### 1.3 RAIL_CUT (레일 단절) - 심각도 Level 2 (HIGH)
- **파일**: `data/RailCutRecordItem.java`
- **트리거**: 레일 연결이 끊기거나 절단될 때
- **핵심 필드**: `railCutFromAddress`, `railCutToAddress`, `railEdgeId`

### 1.4 RAIL_VIBRATION (레일 진동 이상) - 심각도 Level 3 (MEDIUM)
- **파일**: `data/RailVibrationRecordItem.java`
- **트리거**: 레일에서 비정상 진동 감지 시 (X, Y, Z 방향)
- **핵심 필드**: `address`, `term1`, `term2`, `avgX`, `avgY`, `avgZ`, `directory` (진동 방향)

### 1.5 ALARM (일반 시스템 알람)
- 배치 프로세스에서 생성 (APM, OHT Performance 등)
- 리소스 알림, 메시지 큐 알림 용도

### 1.6 VHL_AVG_SPEED (차량 평균 속도)
- 메시지 타입, 비임계 알람

---

## 2. 알람 데이터 구조

### 2.1 Record Item 공통 구조 (모든 알람 타입)
```
- id              : 고유 식별자 (타입별 키 형식 상이)
- fabId           : 공장 ID (M14A, M16A, M16E 등)
- facId           : 설비 ID (M14, M16, M15 등)
- mcpName         : MCP 이름 (A, B, E 등)
- state           : OHT_TIB_STATE.ABNORMAL 또는 OHT_TIB_STATE.NORMAL
- eventDateTime   : 알람 발생 시각 (long ms)
- recoveryDateTime: 알람 해제 시각 (VHL/HID OFF 전용)
- isChanged       : 상태 변경 플래그
- isModified      : 수정 플래그
```

### 2.2 핵심 알람 상태 Enum
```java
// OhtMsgWorkerRunnable.java
public static class OHT_TIB_STATE {
    public static final String NORMAL = "NORMAL";
    public static final String ABNORMAL = "ABNORMAL";
}
```

### 2.3 알람 심각도 레벨 (LayoutUtil.java)
```
- Level 1: HID_OFF    (CRITICAL - 인프라 다운)
- Level 2: VHL_OFF    (HIGH - 차량/경로 차단)
- Level 2: RAIL_CUT   (HIGH - 차량/경로 차단)
- Level 3: RAIL_VIBRATION (MEDIUM - 이상 감지)
```

---

## 3. 알람 발생 로직 및 트리거 조건

### 3.1 VHL OFF 감지 (`process/OhtMsgWorkerRunnable.java`)
- UDP 상태 메시지에서 비정상 상태 감지
- **3가지 케이스**:
  - Case A: 새로운 VHL OFF 이벤트 발생
  - Case B: VHL OFF 해제 (ABNORMAL → NORMAL 상태 전이)
  - Case C: 동일 위치의 다른 장치

### 3.2 HID OFF 감지 (`process/OhtMsgWorkerRunnable.java`)
- 모든 Hidden Instruction Device 모니터링
- 영향 받는 레일 주소 및 포트 기록
- ABNORMAL → NORMAL 전이 시 복구 기록

### 3.3 Rail Cut 감지 (`batch/RailCutRefreshBatch.java`)
- `inactive_SCH_1.dat` 파일을 주기적으로 다운로드
- 이전 vs 현재 railcut 세트 비교:
  - **Case A**: 복구된 RAIL CUT (α-β)
  - **Case B**: 새로운 RAIL CUT (β-α)
  - **Case C**: 유지 중인 RAIL CUT (α∩β)
- Navigator를 사용하여 영향 받는 주소/포트 계산

### 3.4 Rail Vibration 감지 (`batch/RailVibrationBatch.java`)
- IOT_M16A 데이터베이스에서 진동 데이터 쿼리
- X, Y, Z 축 진동 측정값 기록
- term1(기준) vs term2(현재) 비교
- 변화율(%) 계산

### 3.5 서버 리소스 알람 (`batch/ServerResourceApmBatch.java`)
- **임계값** (XmlUtil 환경변수):

| 항목 | 임계값 | 설명 |
|------|--------|------|
| CPU_OS_LIMIT | 30% | OS CPU 사용률 |
| CPU_USER_LIMIT | 10% | User CPU 사용률 |
| JVM_CPU_LIMIT | 1% | JVM CPU 사용률 |
| JVM_HEAP_LIMIT | 400MB | JVM 힙 메모리 |
| JVM_HEAP_LIMIT2 | 800MB | TS 프로세스 JVM 힙 |
| JVM_THREAD_LIMIT | 1000 | JVM 스레드 수 |
| TXN_TIME_LIMIT | 1분 | 트랜잭션 시간 |

### 3.6 메시지 큐 성능 알람 (`batch/OhtPerformanceTimeMinBatch.java`)
- REP (Reply) 커맨드 알람
- ASSIGN 시간 알람 (ASSIGN_LIMIT: 100%)
- TRANSFERRING 시간 알람 (TRANSFERRING_LIMIT: 100%)
- VHL 가동률 알람 (VHL_RATE_LIMIT: 95%)

### 3.7 시스템 알림 상태 (`batch/AlertingSystemStatus.java`)
- 메모리 사용률 임계값
- CPU 사용률 임계값
- 디스크 사용률 임계값
- **경고 연속 횟수**: 10회 연속 초과 시 SMS 알림 발송

### 3.8 Bridge Layout 이상 감지 (`batch/BridgeLayoutBatch.java`, `batch/BridgeJudgeRangeBatch.java`)
- **시그마 기반 이상 감지** (기본: 3 시그마)
- 히스토리 데이터 통계 분석
- 현재 값 vs 예상 범위 비교
- Trim 값: 15% (상/하위 15% 데이터 제외)

### 3.9 Stage Command 모니터링 (`batch/MonitoringControlBatch.java`)
- Stage Command의 ABNORMAL 상태 모니터링
- 특정 목적 포트의 머신 추적
- fab/목적 포트별 그룹핑

---

## 4. 알람 알림 메커니즘

### 4.1 TIBRV 메시지 브로드캐스팅 (`service/TibrvService.java`)
- **서비스**: 멀티캐스트 메시징 시스템
- **Subject 형식**: `{facId}.{environment}.{alarmType}`
- **메시지 포맷** (TibrvMsg XML):

| 필드 | 설명 |
|------|------|
| DEVICE_TYP | 알람 타입 (HIDOFF, VHLOFF, RAILCUT, VIBRATION) |
| FAB_ID | 공장 식별자 |
| FAC_ID | 설비 식별자 |
| EVENT_DT | 이벤트 타임스탬프 |
| DEVICE_NM | 장치 식별자 |
| FALR_STAT_TYP | NORMAL 또는 ABNORMAL |
| FALR_RAISE_ADDR_LVAL | 영향 받는 주소 (콤마 구분) |
| FALR_AFFECT_PORT_LVAL | 영향 받는 포트 (콤마 구분) |
| ALARM_CD | 알람 코드 (HID###, error code 등) |
| ALARM_DESC | 알람 설명 |
| ALARM_CMT | 알람 코멘트 |
| ALARM_MSG_CTN | 알람 메시지 내용 |
| ALARM_LEVEL_VAL | 심각도 레벨 (1-3) |
| ALARM_YN | Y/N 알람 발생 여부 |

### 4.2 SMS 알림 (`batch/AlertingSystemStatus.java`)
- 임계 시스템 리소스 이슈에만 사용
- 수신자: `Settings.properties` → `Sms.Receivers`
- **메시지 형식**: `[hostname] Exceed Resources ▶ Memory: X% CPU: X% Disk: X%`
- 10회 연속 초과 후에만 발송 (streak count)

### 4.3 Logpresso 데이터베이스 저장

| 테이블명 | 저장 데이터 |
|----------|------------|
| ATLAS_OHT_HID_OFF | HID OFF 이벤트 |
| ATLAS_OHT_VHL_OFF | VHL OFF 이벤트 |
| ATLAS_OHT_VHL_OFF_ONLY | VHL OFF 모니터링 전용 |
| ATLAS_OHT_STG_CMD_MNT | Stage Command 모니터링 |
| ATLAS_OHT_RAIL_CUT | Rail Cut 이벤트 |
| ATLAS_OHT_RAIL_VIBRATION | Rail Vibration 데이터 |
| bridge_judge_range | Bridge 레이아웃 통계 범위 |
| bridge_layout_test | Bridge 레이아웃 이상 감지 |
| server_resource_apm | 서버 리소스 측정값 |
| server_resource_predict | 서버 리소스 예측값 |
| ATLAS_TIB_SEND_MSG_LOG | TIBRV 메시지 송신 로그 |
| oht_cmd_count | OHT 명령 카운트 |
| oht_time_avg | OHT 메시지 타이밍 평균 |

---

## 5. 알람 심각도 레벨

| 심각도 | 알람 타입 | Level | 임계값 | 조치 |
|--------|----------|-------|--------|------|
| CRITICAL | HID_OFF | 1 | 즉시 | TIBRV + DB + UI |
| HIGH | VHL_OFF | 2 | 즉시 | TIBRV + DB + UI |
| HIGH | RAIL_CUT | 2 | 즉시 | TIBRV + DB + UI |
| MEDIUM | RAIL_VIBRATION | 3 | Term2-Term1 비교 | TIBRV + DB |
| MEDIUM | Resource APM | N/A | 다중 임계값 | DB + Prediction |
| LOW | System Status | N/A | 10회 연속 | SMS 전용 |

---

## 6. 배치 프로세스별 알람 생성

| 배치 작업 | 주기 | 알람 타입 | DB 테이블 |
|-----------|------|----------|-----------|
| AlertingSystemStatus | 1분 | System Resource | SMS 전용 |
| MonitoringControlBatch | 1분 | VHL_OFF, Stage Command | ATLAS_OHT_* |
| RailCutRefreshBatch | 가변 | RAIL_CUT | ATLAS_OHT_RAIL_CUT |
| RailVibrationBatch | 가변 | RAIL_VIBRATION | ATLAS_OHT_RAIL_VIBRATION |
| ServerResourceApmBatch | 일별 + 7일 | APM Alarm, Prediction | server_resource_* |
| OhtPerformanceTimeMinBatch | 1분 | Message Queue Alarm | oht_cmd_count, oht_time_avg |
| OhtPerformanceTimeHourBatch | 시간별 | Message Counts | oht_cmd_count, oht_time_avg |
| BridgeLayoutBatch | 1분 | Bridge Anomaly | bridge_layout_test |
| BridgeJudgeRangeBatch | 일별 | Bridge Judge Range | bridge_judge_range |
| AbnormalDetectBatch | 가변 | TS Log Pattern | ts_log_pattern |
| SystemMessageDetectBatch | 가변 | System Message Pattern | System messages |
| SwitchSystemBatch | 파일 변경 | Configuration | N/A (config only) |
| DataSetRefreshBatch | 가변 | Dataset Reload | Memory refresh |
| TrafficBatch | 가변 | Traffic Alarm | Traffic stats |
| ItsmChangeRequestBatch | 가변 | ITSM Change Request | ITSM 연동 |
| QTransferPredictBatch | 가변 | Q Transfer Prediction | Q transfer stats |

---

## 7. 알람 시스템 흐름도

### 7.1 실시간 알람 흐름 (UDP 기반)
```
UDP 메시지 수신 (OhtUdpListener)
    ↓
OhtMsgWorkerRunnable.process()
    ↓
상태 분석:
    ├→ VHL 상태 체크 → VhlOffRecordItem
    ├→ HID 상태 체크 → HidOffRecordItem
    └→ Stage Command 체크 → StageCommandRecordItem
    ↓
LayoutUtil.buildLayoutMessageDataMap()
    ↓
병렬 처리:
    ├→ TibrvService.sendMessage()
    │   ├→ TibrvRvdTransport.send() → TIBRV 브로드캐스트
    │   └→ LogpressoAPI.setInsertTuple() → 송신 로그 저장
    ├→ MonitoringControlBatch → DB 저장:
    │   ├→ ATLAS_OHT_VHL_OFF_ONLY (모니터링)
    │   ├→ ATLAS_OHT_HID_OFF (영구 저장)
    │   └→ ATLAS_OHT_STG_CMD_MNT
    └→ UI 업데이트 (WebSocket/Socket.IO)
```

### 7.2 배치 기반 알람 흐름
```
스케줄러 트리거 (주기별)
    ↓
배치 작업 실행:
    ├→ RailCutRefreshBatch → 파일 다운로드 → 비교 분석
    ├→ RailVibrationBatch → IOT DB 쿼리 → 진동 분석
    ├→ ServerResourceApmBatch → APM 데이터 수집 → 임계값 비교
    ├→ BridgeLayoutBatch → 통계 분석 → 시그마 이상 감지
    └→ AlertingSystemStatus → 리소스 체크 → SMS 발송
    ↓
Record Item 생성 → 상태 관리
    ↓
상태 검증 (ABNORMAL/NORMAL 체크)
    ↓
데이터 저장/알림:
    ├→ Logpresso DB 저장
    ├→ TIBRV 브로드캐스트 (해당 시)
    └→ SMS 알림 (임계 상황만)
```

---

## 8. 설정 파일 및 유틸리티 클래스

### 8.1 설정 파일
| 파일명 | 용도 |
|--------|------|
| Settings.properties | SMS 수신자, 리소스 임계값 |
| FabSet.properties | 공장/MCP 설정 |
| Reset.properties | 리셋 기능 트리거 |
| variable.xml | 환경 변수 임계값 |
| alarm_message.xml | 일반 알람 메시지 템플릿 |
| oht_alarm_message.xml | OHT 전용 알람 메시지 |
| customQuery.xml | Logpresso 쿼리 템플릿 |
| customQuery2.xml | 추가 쿼리 템플릿 |

### 8.2 유틸리티 클래스
| 클래스 | 역할 |
|--------|------|
| XmlUtil | 메시지/변수 로딩, Logpresso 쿼리 |
| LayoutUtil | 메시지 구성, 알람 데이터 매핑 |
| TibrvService | TIBRV 메시징 |
| LogpressoAPI | 데이터베이스 쓰기 작업 |
| SmsUtil (implied) | SMS 전송 |
| DataService | 중앙 데이터 저장소 |
| FilePathUtil | 설정 파일 경로 |
| Env | 환경 관리 |

---

## 9. 알람 해제 메커니즘

1. **자동 복구**: 조건 정상화 시 (state → NORMAL)
2. **리셋 기능**: Reset.properties를 통한 수동 복구 트리거
3. **시간 기반**: 복구 시각 타임스탬프 기록 (recoveryDateTime)
4. **연쇄 해제**: 한 알람 복구가 종속 알람 해제 가능 (예: RAIL_CUT → 레일 가용성 영향)

---

## 10. 파일별 알람 관련 코드 매핑

### 데이터 모델 (data/)
| 파일 | 알람 관련 내용 |
|------|---------------|
| `data/HidOffRecordItem.java` | HID OFF 알람 레코드 구조 |
| `data/VhlOffRecordItem.java` | VHL OFF 알람 레코드 구조 |
| `data/RailCutRecordItem.java` | Rail Cut 알람 레코드 구조 |
| `data/RailVibrationRecordItem.java` | Rail Vibration 알람 레코드 구조 |
| `data/StageCommandRecordItem.java` | Stage Command 모니터링 레코드 |
| `data/OhtStats.java` | OHT 통계 (알람 카운트 포함) |
| `data/eq/Oht.java` | OHT 장비 상태 (알람 상태 포함) |
| `data/eq/Conveyor.java` | 컨베이어 장비 알람 상태 |

### 배치 처리 (batch/)
| 파일 | 알람 관련 내용 |
|------|---------------|
| `batch/AlertingSystemStatus.java` | 시스템 리소스 SMS 알람 |
| `batch/AbnormalDetectBatch.java` | TS 로그 이상 패턴 감지 |
| `batch/MonitoringControlBatch.java` | VHL/HID/Stage 모니터링 |
| `batch/RailCutRefreshBatch.java` | Rail Cut 감지/복구 |
| `batch/RailVibrationBatch.java` | Rail Vibration 감지 |
| `batch/ServerResourceApmBatch.java` | 서버 리소스 APM 알람 |
| `batch/OhtPerformanceTimeMinBatch.java` | 1분 단위 OHT 성능 알람 |
| `batch/OhtPerformanceTimeHourBatch.java` | 시간 단위 OHT 성능 알람 |
| `batch/BridgeLayoutBatch.java` | Bridge 이상 감지 (1분 주기) |
| `batch/BridgeJudgeRangeBatch.java` | Bridge 통계 범위 계산 |
| `batch/BridgeLayoutDetailBatch.java` | Bridge 상세 레이아웃 |
| `batch/SystemMessageDetectBatch.java` | 시스템 메시지 패턴 감지 |
| `batch/SwitchSystemBatch.java` | 설정 변경 알람 |
| `batch/TrafficBatch.java` | 트래픽 알람 |
| `batch/DataSetRefreshBatch.java` | 데이터셋 리프레시 알람 |
| `batch/ItsmChangeRequestBatch.java` | ITSM 변경 요청 알람 |
| `batch/QTransferPredictBatch.java` | Q Transfer 예측 알람 |

### 처리/서비스 (process/, service/)
| 파일 | 알람 관련 내용 |
|------|---------------|
| `process/OhtMsgWorkerRunnable.java` | 실시간 알람 상태 처리 핵심 |
| `service/TibrvService.java` | TIBRV 알람 브로드캐스팅 |
| `service/UiLogpresso.java` | UI용 Logpresso 알람 조회 |

### 유틸리티/환경 (util/, environment/)
| 파일 | 알람 관련 내용 |
|------|---------------|
| `util/XmlUtil.java` | 알람 메시지 템플릿, 임계값 |
| `util/LayoutUtil.java` | 알람 메시지 구성, 레벨 매핑 |
| `util/DataService.java` | 알람 데이터 중앙 저장소 |
| `util/Util.java` | 알람 관련 유틸 함수 |
| `util/FilePathUtil.java` | 알람 설정 파일 경로 |
| `environment/Env.java` | 알람 환경 설정 |

### 쿼리/맵 (queryformat/, map/)
| 파일 | 알람 관련 내용 |
|------|---------------|
| `queryformat/MongodbCommonFilterQuery.java` | MongoDB 알람 필터 쿼리 |
| `queryformat/LogpressoMcslogQuery.java` | MCS 로그 알람 쿼리 |
| `queryformat/LogpressoCommonFilterQuery.java` | Logpresso 알람 필터 쿼리 |
| `queryformat/type/ENUM_FABLIST_GROUP.java` | 공장 그룹 enum (알람 분류) |
| `map/Vhl.java` | 차량 알람 상태 매핑 |

---

## 요약

ASAS ALT 시스템의 알람 아키텍처는 **실시간 UDP 기반 감지**와 **배치 스케줄 기반 감지** 두 축으로 구성됩니다.

- **6가지 주요 알람 타입**: HID_OFF, VHL_OFF, RAIL_CUT, RAIL_VIBRATION, ALARM, VHL_AVG_SPEED
- **3단계 심각도**: Level 1 (CRITICAL) → Level 2 (HIGH) → Level 3 (MEDIUM)
- **3가지 알림 채널**: TIBRV 브로드캐스트, Logpresso DB 저장, SMS 알림
- **16개+ 배치 프로세스**가 주기적으로 알람 조건 검사
- **13개+ DB 테이블**에 알람 데이터 저장
- **8개+ 설정 파일**로 임계값 및 알람 동작 제어
