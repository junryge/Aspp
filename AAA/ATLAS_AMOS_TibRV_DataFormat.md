# ATLAS → AMOS 시스템 TibRV 실시간 데이터 연동 설계서

> **프로젝트**: ATLAS AMHS → AMOS 실시간 데이터 연동  
> **통신 프로토콜**: TIBCO Rendezvous (TibRV)  
> **분석 기반**: ALT/src 소스코드 (TibrvService, LayoutUtil, OhtMsgWorkerRunnable 등)

---

## 1. 현재 ATLAS TibRV 전송 구조 분석

### 1.1 기존 TibRV 메시지 구조

ATLAS에서 이미 사용 중인 TibRV 전송 포맷 (`TibrvSendMsg`):

```
TibrvSendMsg {
    String key      // TibRV 라우팅 키 (예: "M14A:send:receiver1")
    String type     // Subject 타입 (SEND_SUB_SUBJECT 상수)
    Map<String, Object> data  // 실제 데이터 페이로드
}
```

### 1.2 기존 SEND_SUB_SUBJECT (TibrvService.java)

| Subject 상수 | 값 | 용도 | 전송 주기 |
|---|---|---|---|
| ALARM | "ALARM" | 통합 알람 | 이벤트 발생 시 |
| INIT | "INIT" | 초기화 데이터 | 시스템 시작 시 |
| HID_OFF | "HIDOFF" | HID OFF 이벤트 | 실시간 (UDP 수신 시) |
| VHL_OFF | "VHLOFF" | VHL OFF 이벤트 | 실시간 (UDP 수신 시) |
| RAIL_CUT | "RAILCUT" | Rail Cut 이벤트 | 배치 (RailCutRefreshBatch) |
| RAIL_VIBRATION | "VIBRATION" | Rail 진동 이상 | 배치 (RailVibrationBatch) |
| VHL_AVG_SPEED | "VHLSPEED" | OHT 평균 속도 | 배치 (TrafficBatch) |
| VHL_CNT | "VHLBEING" | HID별 VHL 수 | 배치 (VhlCnt10/30/60Batch) |

### 1.3 기존 메시지 데이터 필드 (LayoutUtil.LAYOUT_MEMBER)

| 필드명 | 상수 | 설명 |
|---|---|---|
| DEVICE_TYP | DEVICE_TYPE | 장치 타입 (HIDOFF/VHLOFF/RAILCUT 등) |
| FAB_ID | FAB_ID | FAB ID (M14A, M16 등) |
| FAC_ID | FAC_ID | FAC ID (M14, M16 등) |
| EVENT_DT | EVENT_OCCURRED_DATE | 이벤트 발생 시간 |
| DEVICE_NM | DEVICE_NAME | 장치명 |
| FALR_STAT_TYP | LAYOUT_STATE | 상태 (NORMAL / ABNORMAL) |
| FALR_RAISE_ADDR_LVAL | ADDRESS_LIST | 영향 주소 목록 (쉼표 구분) |
| FALR_AFFECT_PORT_LVAL | PORT_LIST | 영향 포트 목록 (쉼표 구분) |
| ALARM_CD | ALARM_CODE | 알람 코드 |
| ALARM_DESC | ALARM_DESCRIPTION | 알람 설명 |
| ALARM_CMT | ALARM_COMMENT | 알람 코멘트 |
| ALARM_MSG_CTN | ALARM_MESSAGE_CONTENTS | 알람 메시지 내용 |
| ALARM_LEVEL_VAL | ALARM_LEVEL | 알람 등급 (1/2/3) |
| ALARM_YN | ALARM_YN | 알람 발생 여부 |

---

## 2. AMOS 연동용 신규 Subject 설계

AMOS 시스템에 전달할 데이터를 4개 카테고리로 구분:

### 2.1 Subject 체계

```
Subject 포맷: {FAC_ID}.{ENV}.ATLAS.AMOS.{DATA_TYPE}
예시: M14.PRD.ATLAS.AMOS.OHT_LANE_STATUS
```

| # | Subject (DATA_TYPE) | 설명 | 전송 주기 | 데이터 소스 |
|---|---|---|---|---|
| 1 | OHT_LANE_STATUS | OHT LANE 이상현황 | 실시간 + 10초 | RailEdge, HidOff, RailCut |
| 2 | OHT_EQP_STATUS | 반송장비 상태현황 | 10초 | Vhl (VhlUdpState), OhtStats |
| 3 | HID_ZONE_STATUS | HID ZONE 현황 | 10초 | RailEdge(hidId), VhlCntBatch |
| 4 | CNV_STATUS | CNV 장비/Zone 현황 | 10초 | CnvPortNode, CnvTask, Conveyor |

---

## 3. 데이터 포맷 상세

### 3.1 OHT LANE 이상현황 (OHT_LANE_STATUS)

OHT 주행 레인(RailEdge) 단위의 이상 상태를 실시간 전송.

**데이터 소스**: `RailEdge`, `HidOffRecordItem`, `VhlOffRecordItem`, `RailCutRecordItem`

```json
{
  "MSG_ID": "OHT_LANE_STATUS",
  "MSG_VER": "1.0",
  "FAB_ID": "M14A",
  "FAC_ID": "M14",
  "MCP_NM": "A",
  "SEND_DT": "2026-03-21 14:30:00",
  "LANE_LIST": [
    {
      "RAIL_EDGE_ID": "M14A:A:RE_001",
      "FROM_ADDR": 1001,
      "TO_ADDR": 1002,
      "HID_ID": 5,
      "LOOP_ID": 1,
      "AREA_NM": "BAY01",
      "BAY_NM": "BAY01-A",
      "LANE_STATE": "ABNORMAL",
      "FAULT_TYPE": "HIDOFF",
      "FAULT_DETAIL": {
        "DEVICE_NM": "HID005",
        "ERROR_CD": "E001",
        "EVENT_DT": "2026-03-21 14:28:30",
        "AFFECTED_ADDR": "1001,1002,1003",
        "AFFECTED_PORT": "PORT01,PORT02",
        "ALARM_LEVEL": 1,
        "ALARM_DESC": "HID005 OFF detected"
      },
      "VELOCITY": 120.5,
      "MAX_VELOCITY": 180.0,
      "IS_AVAILABLE": false,
      "VHL_COUNT": 3,
      "RAIL_DIR": "LEFT"
    }
  ],
  "SUMMARY": {
    "TOTAL_LANE": 850,
    "NORMAL_CNT": 842,
    "HIDOFF_CNT": 3,
    "VHLOFF_CNT": 2,
    "RAILCUT_CNT": 1,
    "VIBRATION_CNT": 2
  }
}
```

**필드 설명**:

| 필드 | 타입 | 소스 | 설명 |
|---|---|---|---|
| RAIL_EDGE_ID | String | RailEdge.id | Rail Edge 고유 ID |
| FROM_ADDR | int | RailEdge.fromAddress | 시작 주소 |
| TO_ADDR | int | RailEdge.toAddress | 끝 주소 |
| HID_ID | int | RailEdge.hidId | HID Zone ID |
| LOOP_ID | int | RailEdge.loopId | Loop ID |
| AREA_NM | String | RailEdge.areaName (AbstractEdge) | 구역명 |
| BAY_NM | String | RailEdge.bayName (AbstractEdge) | Bay명 |
| LANE_STATE | String | 종합 판정 | NORMAL / ABNORMAL |
| FAULT_TYPE | String | 이벤트 타입 | HIDOFF / VHLOFF / RAILCUT / VIBRATION / NONE |
| VELOCITY | double | RailEdge.velocity | 현재 평균 속도 (분속) |
| MAX_VELOCITY | double | RailEdge.maxVelocity | 최대 속도 |
| IS_AVAILABLE | boolean | RailEdge.isAvailable | 사용 가능 여부 |
| VHL_COUNT | int | RailEdge.vhlIdMap.size() | 현재 구간 VHL 수 |
| RAIL_DIR | String | RailEdge.railDir | 레일 방향 (LEFT/RIGHT) |

**전송 조건**:
- 이상 발생/해소 시 → 즉시 전송 (기존 HID_OFF/VHL_OFF/RAIL_CUT TibRV 이벤트 연동)
- 정상 상태 → 10초 주기 요약 전송 (SUMMARY만)

---

### 3.2 반송장비(OHT) 상태현황 (OHT_EQP_STATUS)

OHT 차량(Vehicle) 단위의 실시간 상태 전송.

**데이터 소스**: `Vhl`, `Vhl.VhlUdpState`, `OhtStats`

```json
{
  "MSG_ID": "OHT_EQP_STATUS",
  "MSG_VER": "1.0",
  "FAB_ID": "M14A",
  "FAC_ID": "M14",
  "MCP_NM": "A",
  "SEND_DT": "2026-03-21 14:30:00",
  "VHL_LIST": [
    {
      "VHL_ID": "V001",
      "EQP_ID": "OHT_M14A_A",
      "VHL_TYPE": 1,
      "STATE": "TRANSFERRING",
      "DET_STATE": "MOVING",
      "IS_ONLINE": true,
      "IS_FULL": true,
      "CARRIER_ID": "FOUP001",
      "CMD_ID": "CMD_20260321_001",
      "RAIL_EDGE_ID": "M14A:A:RE_045",
      "RAIL_NODE_ID": "RN_1001",
      "NEXT_RAIL_NODE_ID": "RN_1002",
      "DISTANCE": 1250.5,
      "SRC_PORT_ID": "PORT_A01",
      "DEST_PORT_ID": "PORT_B03",
      "DEST_STATION_ID": "ST_B03",
      "ERROR_CD": "",
      "PRIORITY": 50,
      "RUN_CYCLE": "DELIVERING",
      "VHL_CYCLE": "ACTIVE",
      "RECEIVED_DT": "2026-03-21 14:29:58"
    }
  ],
  "STATS": {
    "TOTAL_VHL": 120,
    "ONLINE_CNT": 115,
    "OFFLINE_CNT": 5,
    "TRANSFERRING_CNT": 68,
    "IDLE_CNT": 35,
    "STAGE_WAIT_CNT": 7,
    "MANUAL_CNT": 3,
    "ERROR_CNT": 2,
    "AVG_VELOCITY": 125.3,
    "CENTER_VELOCITY": 130.1
  }
}
```

**필드 설명**:

| 필드 | 타입 | 소스 | 설명 |
|---|---|---|---|
| VHL_ID | String | Vhl.id | 차량 ID |
| STATE | String | VhlUdpState.state | VHL_STATE (IDLE/TRANSFERRING/REMOVING 등) |
| DET_STATE | String | VhlUdpState.detailState | 상세 상태 (MOVING/STAGE_MOVING 등) |
| IS_ONLINE | boolean | VhlUdpState.isOnline | 온라인 여부 |
| IS_FULL | boolean | VhlUdpState.isFull | 캐리어 적재 여부 |
| CARRIER_ID | String | VhlUdpState.udpCarrierId | 적재 캐리어 ID |
| RAIL_EDGE_ID | String | VhlUdpState.railEdgeId | 현재 위치 Rail Edge |
| DISTANCE | double | VhlUdpState.distance | Rail 상 이동 거리 |
| RUN_CYCLE | String | VhlUdpState.runCycle | NONE/ACQUIRING/DELIVERING/DEPOSITING |
| VHL_CYCLE | String | VhlUdpState.vhlCycle | NONE/ACTIVE/PARKING |

**전송 주기**: 10초

---

### 3.3 HID ZONE 현황 (HID_ZONE_STATUS)

HID Zone 단위의 OHT 밀집도 및 IN/OUT 흐름 전송.

**데이터 소스**: `RailEdge(hidId)`, `VhlCntBatch`, `HidEdgeInOutQueueFlushBatch`

```json
{
  "MSG_ID": "HID_ZONE_STATUS",
  "MSG_VER": "1.0",
  "FAB_ID": "M14A",
  "FAC_ID": "M14",
  "MCP_NM": "A",
  "SEND_DT": "2026-03-21 14:30:00",
  "HID_LIST": [
    {
      "HID_ID": 5,
      "VHL_COUNT": 8,
      "VHL_MAX": 15,
      "VHL_PRECAUTION": 12,
      "CONGESTION_LEVEL": "NORMAL",
      "IS_AVAILABLE": true,
      "HID_STATE": "NORMAL",
      "AVG_VELOCITY": 125.5,
      "RAIL_EDGE_COUNT": 12,
      "INOUT_FLOW": {
        "IN_COUNT": 15,
        "OUT_COUNT": 13,
        "NET_FLOW": 2,
        "PERIOD_SEC": 60
      },
      "FAULT_INFO": {
        "HIDOFF_YN": "N",
        "RAILCUT_YN": "N",
        "VHLOFF_COUNT": 0
      }
    }
  ],
  "SUMMARY": {
    "TOTAL_HID": 45,
    "NORMAL_CNT": 40,
    "CAUTION_CNT": 3,
    "WARNING_CNT": 1,
    "CRITICAL_CNT": 1
  }
}
```

**필드 설명**:

| 필드 | 타입 | 소스 | 설명 |
|---|---|---|---|
| HID_ID | int | RailEdge.hidId | HID Zone ID |
| VHL_COUNT | int | VhlCntBatch 집계 | 현재 Zone 내 VHL 수 |
| VHL_MAX | int | RawHid.vhlMax | 최대 허용 VHL 수 |
| VHL_PRECAUTION | int | RawHid.vhlPreCaution | 주의 임계값 |
| CONGESTION_LEVEL | String | 산출 로직 | NORMAL/CAUTION/WARNING/CRITICAL |
| INOUT_FLOW | Object | HidEdgeInOutQueueFlushBatch | 1분간 IN/OUT 카운트 |
| IN_COUNT | int | edgeInOutCountMap | HID 진입 VHL 수 |
| OUT_COUNT | int | edgeInOutCountMap | HID 이탈 VHL 수 |
| NET_FLOW | int | IN - OUT | 순 유입량 (양수=유입초과) |

**혼잡도 판정 로직**:
```
if VHL_COUNT >= VHL_MAX         → CRITICAL
if VHL_COUNT >= VHL_PRECAUTION  → WARNING
if VHL_COUNT >= VHL_MAX * 0.7   → CAUTION
else                            → NORMAL
```

**전송 주기**: 10초 (VhlCntBatch 연동), IN/OUT은 1분 주기

---

### 3.4 CNV 장비/Zone 현황 (CNV_STATUS)

Conveyor 장비 상태 및 Zone 단위 현황 전송.

**데이터 소스**: `CnvPortNode`, `CnvTask`, `Conveyor`, `RawCnvZone`, `CnvEdge`

```json
{
  "MSG_ID": "CNV_STATUS",
  "MSG_VER": "1.0",
  "FAB_ID": "M14A",
  "FAC_ID": "M14",
  "SEND_DT": "2026-03-21 14:30:00",
  "CNV_EQP_LIST": [
    {
      "CNV_ID": "CNV_M14A_01",
      "CNV_NM": "CONVEYOR_01",
      "IS_AVAILABLE": true,
      "LAYOUT": "INTER_BAY",
      "PORT_COUNT": 24,
      "ACTIVE_TASK_COUNT": 5
    }
  ],
  "CNV_ZONE_LIST": [
    {
      "ZONE_ID": 101,
      "EQP_NM": "CNV_M14A_01",
      "ZONE_TYPE": "ZONE",
      "DISPLAY_NM": "CNV_Z101",
      "LEVEL": 3,
      "STATE": true,
      "CARRIER_ID": "FOUP_003",
      "IS_AVAILABLE": true,
      "PHYSICAL_TYPE": 1,
      "LOGICAL_TYPE": 0,
      "DRAW_X": 150.0,
      "DRAW_Y": 320.0
    }
  ],
  "CNV_TASK_LIST": [
    {
      "TASK_ID": "TASK_001",
      "CMD_ID": "CMD_CNV_001",
      "CARRIER_ID": "FOUP_003",
      "FROM_NODE": "CN_IN_01",
      "TO_NODE": "CN_OUT_05",
      "CURRENT_NODE": "CN_Z103",
      "EVENT": "TRANSFERRING",
      "CREATE_DT": "2026-03-21 14:28:00",
      "CMD_DT": "2026-03-21 14:28:10",
      "ELAPSED_SEC": 110
    }
  ],
  "CNV_EDGE_LIST": [
    {
      "EDGE_ID": "CE_001",
      "FROM_NODE": "CN_IN_01",
      "TO_NODE": "CN_Z101",
      "IS_AVAILABLE": true,
      "AVG_TRANSFER_SEC": 150,
      "COST": 15000
    }
  ],
  "SUMMARY": {
    "TOTAL_CNV": 4,
    "AVAILABLE_CNT": 3,
    "TOTAL_ZONE": 96,
    "OCCUPIED_ZONE": 12,
    "ACTIVE_TASK": 5,
    "AVG_TRANSFER_SEC": 145
  }
}
```

**CNV Zone 필드**:

| 필드 | 타입 | 소스 | 설명 |
|---|---|---|---|
| ZONE_ID | int | RawCnvZone.zoneId / CnvPortNode.zoneNo | Zone 번호 |
| EQP_NM | String | RawCnvZone.eqpNm | 소속 Conveyor명 |
| ZONE_TYPE | String | CnvPortNode.CNV_NODE_TYPE | ZONE/BED/INPUT/OUTPUT/QS/LFT |
| STATE | boolean | RawCnvZone.state | 가동 상태 |
| CARRIER_ID | String | CnvPortNode.carrierIdList | 현재 적재 캐리어 |
| PHYSICAL_TYPE | int | RawCnvZone.physicalType | 물리 유형 |
| LOGICAL_TYPE | int | RawCnvZone.logicalType | 논리 유형 |

**CNV Task 필드**:

| 필드 | 타입 | 소스 | 설명 |
|---|---|---|---|
| TASK_ID | String | CnvTask.id | 작업 ID |
| EVENT | String | CnvTask.event (CNV_EVENT) | CREATED/INITIATED/TRANSFERRING/COMPLETED |
| FROM_NODE | String | CnvTask.frNodeId | 출발 노드 |
| TO_NODE | String | CnvTask.toNodeId | 도착 노드 |
| CURRENT_NODE | String | CnvTask.currentNodeId | 현재 위치 노드 |

**전송 주기**: 10초

---

## 4. ATLAS→AMOS 연동 아키텍처

### 4.1 데이터 흐름

```
[OHT UDP] → OhtUdpListener → OhtMsgWorkerRunnable → DataSet (메모리)
                                                         │
[CNV SIO] → CnvSocketIOListener ──────────────────→ DataSet (메모리)
                                                         │
                                                    ┌────┴────┐
                                                    │         │
                                              기존 배치    신규 배치
                                              (Traffic     (AMOS
                                               VhlCnt      DataPublish
                                               HidOff      Batch)
                                               등)              │
                                                    │         │
                                                    ▼         ▼
                                              기존 TibRV  AMOS TibRV
                                              Subject     Subject
                                                         (신규 4종)
                                                              │
                                                              ▼
                                                        AMOS 시스템
```

### 4.2 신규 배치 클래스 설계

```java
/**
 * AMOS 시스템 데이터 발행 배치
 * 주기: 10초
 * 전송: TibRV (4개 Subject)
 */
public class AmosDataPublishBatch implements Job {

    @Override
    public void execute(JobExecutionContext context) {
        // 1. OHT LANE 이상현황
        publishOhtLaneStatus();

        // 2. 반송장비 상태현황
        publishOhtEqpStatus();

        // 3. HID ZONE 현황
        publishHidZoneStatus();

        // 4. CNV 현황
        publishCnvStatus();
    }
}
```

### 4.3 TibRV Subject 등록

`TibrvService.SEND_SUB_SUBJECT`에 추가:

```java
public static class SEND_SUB_SUBJECT {
    // ... 기존 ...
    // AMOS 연동 (신규)
    public static final String AMOS_OHT_LANE    = "AMOS_OHT_LANE";
    public static final String AMOS_OHT_EQP     = "AMOS_OHT_EQP";
    public static final String AMOS_HID_ZONE    = "AMOS_HID_ZONE";
    public static final String AMOS_CNV          = "AMOS_CNV";
}
```

### 4.4 FunctionItem 스위치 추가

`environment/type/FunctionItem.java`에 추가:

```java
private Boolean useAmosPublish = null;  // AMOS 데이터 발행 ON/OFF
```

---

## 5. 데이터 수집 가능 여부 정리

| 데이터 항목 | OHT | CNV | 데이터 소스 확인 | 비고 |
|---|---|---|---|---|
| 장비 상태 (온라인/오프라인) | ✅ | ✅ | Vhl.isOnline / Conveyor.isAvailable | |
| 실시간 위치 | ✅ | ✅ | VhlUdpState.railEdgeId / CnvTask.currentNodeId | |
| 이동 속도 | ✅ | ⚠️ | RailEdge.velocity / CnvEdge.avgTransferIntervalT | CNV는 평균 반송시간만 |
| 캐리어 적재 상태 | ✅ | ✅ | VhlUdpState.isFull / CnvPortNode.carrierIdList | |
| LANE 이상 (HID OFF) | ✅ | ❌ | HidOffRecordItem | CNV에 HID 개념 없음 |
| LANE 이상 (VHL OFF) | ✅ | ❌ | VhlOffRecordItem | |
| LANE 이상 (Rail Cut) | ✅ | ❌ | RailCutRecordItem | |
| LANE 이상 (진동) | ✅ | ❌ | RailVibrationRecordItem | |
| Zone 밀집도 | ✅ | ✅ | VhlCntBatch(HID별) / CnvPortNode(Zone별) | |
| IN/OUT 흐름 | ✅ | ⚠️ | HidEdgeInOutQueueFlushBatch | CNV는 Task 기반 추적 |
| 반송 작업 현황 | ✅ | ✅ | Command+Job / CnvTask | |
| 장비 에러 코드 | ✅ | ⚠️ | VhlUdpState.errorCode / CnvTask.reasonCd | CNV는 reasonCd만 |
| 구역(Area/Bay) 정보 | ✅ | ✅ | AbstractEdge.areaName,bayName | |
| 좌표 정보 | ✅ | ✅ | RailNode.drawX,drawY / CnvPortNode.drawX,drawY | 맵 표시용 |

**요약**:
- OHT: 모든 항목 실시간 수집 가능 (UDP 기반)
- CNV: 장비상태/위치/캐리어/작업 가능, LANE 이상 관련은 OHT 전용 개념
- CNV는 Zone 기반(CnvPortNode)으로 상태 추적, 속도는 평균 반송시간으로 대체

---

## 6. 구현 체크리스트

```
[ ] TibrvService.SEND_SUB_SUBJECT에 AMOS 4종 Subject 추가
[ ] FunctionItem에 useAmosPublish 스위치 추가
[ ] SwitchSystemBatch에 AMOS 스위치 반영
[ ] AmosDataPublishBatch 클래스 생성 (batch/)
[ ] Quartz 스케줄러에 10초 주기 등록
[ ] AMOS 측 TibRV Subscriber 설정 (Subject, network, service)
[ ] JSON 직렬화 유틸 (기존 JsonUtil 활용)
[ ] 데이터 사이즈 테스트 (VHL 120대 기준 ~50KB/메시지)
[ ] 장애 시 재전송 로직 (LinkedBlockingQueue 기존 패턴)
[ ] 로그 테이블 생성 ({FAB}_ATLAS_AMOS_SEND_LOG)
```
