# HID IN/OUT 로직 상세 설명

## 1. 개요

HID(Hoisting Equipment Zone)는 레일/컨베이어 시스템의 **논리적 구역**을 나타내는 정수 ID입니다.
HID_IN_OUT 기능은 차량(OHT/Vehicle)이 한 HID 구역에서 다른 HID 구역으로 이동할 때 발생하는 **엣지 전환(Edge Transition)을 감지하고 집계**하는 기능입니다.

- **기존 기능**: HID별 VHL(차량) 재적수 카운트 (`_calculatedVhlCnt`) — 유지
- **추가 기능**: FROM_HIDID → TO_HIDID 엣지 전환 집계 (`_processHidInout`) — 신규

### HID 특수값

| HID 값 | 의미 |
|---------|------|
| `0` | 시스템 외부 (OUTSIDE) |
| `> 0` | 시스템 내부 구역 (예: HID_003, HID_187) |

---

## 2. 전체 아키텍처

```
┌──────────────────────────────────────────────────────────────────────┐
│                        실시간 메시지 처리                              │
│  OhtMsgWorkerRunnable._processMessage()                              │
│                                                                      │
│  ┌─ HID_INOUT ON? ─────────────────────────────────────────────┐     │
│  │  YES → _processHidInout()                                   │     │
│  │        previousHidId != currentHidId?                       │     │
│  │        YES → edgeInOutCountMap에 카운트 +1                   │     │
│  └─────────────────────────────────────────────────────────────┘     │
│                                                                      │
│  ┌─ VHL_CNT ON? ───────────────────────────────────────────────┐     │
│  │  YES → _calculatedVhlCnt()                                  │     │
│  │        previousHidId != currentHidId?                       │     │
│  │        YES → increaseHidVehicleCnt / decreaseHidVehicleCnt  │     │
│  │              vehicle.setHidId(currentHidId)                 │     │
│  └─────────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     매 1분 배치 (Quartz Job)                          │
│  HidEdgeInOutQueueFlushBatch.execute()                               │
│                                                                      │
│  1. edgeInOutCountMap 스냅샷 복사 → 맵 초기화                         │
│  2. Edge Key 파싱 → fabId 기준 그룹핑                                 │
│  3. {FAB}_ATLAS_HID_INOUT 테이블에 INSERT                            │
└──────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     매 24시간 배치 (Quartz Job)                        │
│  HidEdgeInOutUpdateMasterBatch.execute()                             │
│                                                                      │
│  1. _updateHidEdgeMasterInfo()                                       │
│     → {FAB}_ATLAS_INFO_HID_INOUT_MAS (엣지 마스터)                   │
│  2. _updateHidInfoMaster()                                           │
│     → {FAB}_ATLAS_HID_INFO_MAS (HID Zone 정보)                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. 실시간 처리: 엣지 전환 감지

### 3.1 호출 흐름

**파일**: `ALT/src/process/OhtMsgWorkerRunnable.java`

```java
// Line 239-241: 메시지 처리 중 HID_INOUT 기능 확인
if (functionItem.getUseFunction(FunctionType.HID_INOUT)) {
    this._processHidInout(hidId, vehicle, functionItem);
}
```

### 3.2 `_processHidInout()` 메소드 (Line 401-417)

차량의 **이전 HID**와 **현재 HID**를 비교하여 전환이 발생했는지 감지합니다.

```java
private void _processHidInout(int currentHidId, Vhl vehicle, FunctionItem functionItem) {
    int previousHidId = vehicle.getHidId();

    if (previousHidId != currentHidId) {
        // 차량 식별 정보 추출
        String vhlName = vhlIdFull.substring(vhlIdFull.lastIndexOf(':') + 1);
        String eqpName = eqpIdFull.substring(eqpIdFull.lastIndexOf(':') + 1);

        // Edge Key 생성
        String edgeKey = String.format("%03d:%03d:%s:%s:%s:%s:%s",
                previousHidId, currentHidId, this.fabId, this.mcpName,
                vehicle.getFabId(), vhlName, eqpName);

        // 메모리 맵에 카운트 +1
        DataService.getDataSet().getEdgeInOutCountMap()
            .merge(edgeKey, 1, Integer::sum);
    }
}
```

### 3.3 Edge Key 형식

```
"%03d:%03d:%s:%s:%s:%s:%s"
  │     │    │    │     │    │    │
  │     │    │    │     │    │    └─ eqpId (장비 ID)
  │     │    │    │     │    └────── vhlId (차량 ID)
  │     │    │    │     └─────────── vhlFabId (차량 FAB ID)
  │     │    │    └───────────────── mcpName (MCP 이름)
  │     │    └────────────────────── fabId (FAB ID)
  │     └─────────────────────────── toHidId (도착 HID, 3자리 zero-pad)
  └───────────────────────────────── fromHidId (출발 HID, 3자리 zero-pad)
```

**예시**: `003:187:M14A:MCP01:M14A:V00001:EQP001`

### 3.4 메모리 저장소

**파일**: `ALT/src/data/DataSet.java` (Line 55, 904-913)

```java
// ConcurrentHashMap: Edge Key → 전환 횟수
private ConcurrentMap<String, Integer> edgeInOutCountMap = new ConcurrentHashMap<>();
```

---

## 4. 기존 기능: HID별 VHL 재적수

### 4.1 `_calculatedVhlCnt()` 메소드 (Line 368-391)

HID_INOUT과 별도로, `VHL_CNT` 기능 스위치로 제어되는 **기존 로직**입니다.

```java
private void _calculatedVhlCnt(int currentHidId, String key, Vhl vehicle, FunctionItem functionItem) {
    int previousHidId = vehicle.getHidId();

    if (previousHidId != currentHidId) {
        // 새로운 HID에 진입 → 카운트 증가
        if (currentHidId > 0) {
            DataService.getDataSet().increaseHidVehicleCnt(key + ":" + currentHidId);
        }
        // 이전 HID에서 이탈 → 카운트 감소
        if (previousHidId > 0) {
            DataService.getDataSet().decreaseHidVehicleCnt(key + ":" + previousHidId);
        }
        // 차량의 현재 HID 갱신
        vehicle.setHidId(currentHidId);
    }
}
```

**핵심 차이점**:
- `_calculatedVhlCnt`: HID별 **현재 차량 수**를 관리 (실시간 대시보드용)
- `_processHidInout`: HID 간 **전환 이력**을 집계 (분석/통계용)

---

## 5. 1분 배치: 집계 데이터 저장

### 5.1 `HidEdgeInOutQueueFlushBatch` (Quartz Job)

**파일**: `ALT/src/batch/HidEdgeInOutQueueFlushBatch.java`

매 1분마다 메모리의 `edgeInOutCountMap`을 DB에 저장합니다.

### 5.2 처리 흐름

```
1. edgeInOutCountMap 스냅샷 복사
   └─ copyMap = new HashMap<>(edgeInOutCountMap)

2. edgeInOutCountMap을 새 ConcurrentHashMap으로 교체 (초기화)

3. 각 Edge Key를 파싱하여 필드 추출
   └─ parts[0]=fromHidId, parts[1]=toHidId, parts[2]=fabId,
      parts[3]=mcpName, parts[4]=vhlFabId, parts[5]=vhlId, parts[6]=eqpId

4. fabId 기준으로 Tuple 그룹핑

5. 각 fab별로 {FAB}_ATLAS_HID_INOUT 테이블에 INSERT
```

### 5.3 저장 테이블: `{FAB}_ATLAS_HID_INOUT`

**테이블명 예시**: `M14A_ATLAS_HID_INOUT`, `M16A_ATLAS_HID_INOUT`

| 컬럼명 | 타입 | 설명 | 예시 |
|--------|------|------|------|
| `EVENT_DATE` | STRING | 날짜 (파티션 키) | `2026-03-26` |
| `EVENT_DT` | STRING | 집계 시간 (1분 단위) | `2026-03-26 14:35:00` |
| `FROM_HIDID` | INT | 출발 HID Zone ID | `3` |
| `TO_HIDID` | INT | 도착 HID Zone ID | `187` |
| `TRANS_CNT` | INT | 1분간 전환 횟수 | `5` |
| `FAB_ID` | STRING | 차량 FAB ID | `M14A` |
| `VHL_ID` | STRING | 차량 ID | `V00001` |
| `EQP_ID` | STRING | 장비 ID | `EQP001` |
| `MCP_NM` | STRING | MCP 이름 | `MCP01` |
| `ENV` | STRING | 환경 구분 | `PROD` |

---

## 6. 일간 배치: 마스터 테이블 갱신

### 6.1 `HidEdgeInOutUpdateMasterBatch` (Quartz Job)

**파일**: `ALT/src/batch/HidEdgeInOutUpdateMasterBatch.java`

매일 1회 실행되며, 두 개의 마스터 테이블을 갱신합니다.

### 6.2 실행 조건

```java
// FAB별 Properties 순회
for (fabPropertiesEntry : DataService.getInstance().getFabPropertiesMap()) {
    // layout.zip 경로 확인 → 없으면 SKIP
    // MCP별 HID_INOUT 스위치 확인 → OFF면 SKIP
    if (functionItem.getUseFunction(FunctionType.HID_INOUT) == false) {
        continue;
    }
    _updateHidEdgeMasterInfo(fabId, mcpName, layoutZipFile, edgeMap);
    _updateHidInfoMaster(fabId, mcpName, edgeMap);
}
```

### 6.3 테이블 A: `{FAB}_ATLAS_INFO_HID_INOUT_MAS` (엣지 마스터)

**용도**: 시스템 내 모든 HID 간 연결(엣지) 정의 — 기준 정보

#### 엣지 감지 로직 (`_updateHidEdgeMasterInfo`)

```
1단계: RailEdge 순회 → HID > 0인 모든 HID ID와 이름 수집
       hidNameMap = { 3: "HID_003", 187: "HID_187", ... }

2단계: 인접 RailEdge 간 HID 전환 감지
       - RailEdge A (HID=3)의 toNodeId == RailEdge B (HID=187)의 fromNodeId
       - HID가 다르면 엣지 생성: 3 → 187
       - 중복 제거: processedEdges Set 사용
```

#### 엣지 유형 분류

```java
if (fromHidId == 0) {
    edgeType = "IN";        // 외부(0) → 내부: 시스템 진입
} else if (toHidId == 0) {
    edgeType = "OUT";       // 내부 → 외부(0): 시스템 퇴출
} else {
    edgeType = "INTERNAL";  // 내부 → 내부: 구역 간 이동
}
```

#### 테이블 스키마

| 컬럼명 | 타입 | 설명 | 예시 |
|--------|------|------|------|
| `FROM_HIDID` | INT | 출발 HID Zone ID | `3` |
| `TO_HIDID` | INT | 도착 HID Zone ID | `187` |
| `EDGE_ID` | STRING | 엣지 고유 ID | `003:187` |
| `FROM_HID_NM` | STRING | 출발 HID 이름 | `HID_003` 또는 `OUTSIDE` |
| `TO_HID_NM` | STRING | 도착 HID 이름 | `HID_187` 또는 `OUTSIDE` |
| `MCP_ID` | STRING | MCP 이름 | `MCP01` |
| `ZONE_ID` | STRING | Zone ID (예약) | `` |
| `EDGE_TYPE` | STRING | 엣지 유형 | `IN` / `OUT` / `INTERNAL` |
| `UPDATE_DT` | STRING | 마지막 업데이트 일시 | `2026-03-26 01:00:00` |

---

### 6.4 테이블 B: `{FAB}_ATLAS_HID_INFO_MAS` (HID Zone 정보)

**용도**: HID Zone별 물리적 특성 정보 — 레일 길이, 속도, 포트 수 등

#### 데이터 집계 로직 (`_updateHidInfoMaster`)

```
RailEdge 순회 (HID > 0인 것만):
  ├─ RAIL_LEN_TOTAL  : RailEdge.getLength() → HID별 합계
  ├─ FREE_FLOW_SPEED : RailEdge.getMaxVelocity() → HID별 평균
  └─ PORT_CNT_TOTAL  : RailEdge.getPortIdList().size() → HID별 합계
```

#### 테이블 스키마

| 컬럼명 | 타입 | 설명 | 예시 |
|--------|------|------|------|
| `HID_ID` | INT | HID Zone ID (PK) | `3` |
| `HID_NM` | STRING | HID Zone 이름 | `HID_003` |
| `MCP_ID` | STRING | MCP 이름 | `MCP01` |
| `ZONE_ID` | STRING | Zone ID (예약) | `` |
| `RAIL_LEN_TOTAL` | DOUBLE | 총 레일 길이 (mm) | `15230.5` |
| `FREE_FLOW_SPEED` | DOUBLE | 평균 최대 속도 (mm/s) | `2000.0` |
| `PORT_CNT_TOTAL` | INT | 총 포트 수 | `12` |
| `IN_CNT` | INT | IN Lane 수 (예약) | `0` |
| `OUT_CNT` | INT | OUT Lane 수 (예약) | `0` |
| `VHL_MAX` | INT | 최대 허용 차량 수 (예약) | `0` |
| `ZCU_ID` | STRING | ZCU ID (예약) | `` |
| `UPDATE_DT` | STRING | 마지막 업데이트 일시 | `2026-03-26 01:00:00` |

---

## 7. 기능 ON/OFF 제어

**파일**: `ALT/src/environment/type/FunctionItem.java`

```java
private Boolean useHidInout = null;

public boolean isUseHidInout() {
    return useHidInout != null && useHidInout;
}
```

- `FunctionType.HID_INOUT` 플래그로 **fab:mcp 단위**로 활성화/비활성화
- `Env.getSwitchMap().get(fabId + ":" + mcpName)` 으로 조회
- `true`일 때만 `_processHidInout()` 호출 및 일간 마스터 배치 실행

---

## 8. 핵심 소스 파일 요약

| 파일 | 역할 | 핵심 메소드/필드 |
|------|------|-----------------|
| `ALT/src/process/OhtMsgWorkerRunnable.java` | 실시간 메시지 처리 | `_processHidInout()`, `_calculatedVhlCnt()` |
| `ALT/src/batch/HidEdgeInOutQueueFlushBatch.java` | 1분 배치 — DB 저장 | `execute()` |
| `ALT/src/batch/HidEdgeInOutUpdateMasterBatch.java` | 일간 배치 — 마스터 갱신 | `_updateHidEdgeMasterInfo()`, `_updateHidInfoMaster()` |
| `ALT/src/data/DataSet.java` | 메모리 저장소 | `edgeInOutCountMap` |
| `ALT/src/map/Vhl.java` | 차량 객체 | `getHidId()`, `setHidId()` |
| `ALT/src/environment/type/FunctionItem.java` | 기능 플래그 | `isUseHidInout()`, `FunctionType.HID_INOUT` |

---

## 9. 테이블 요약 (FAB prefix)

| 테이블 | 테이블명 패턴 | 갱신 주기 | 용도 |
|--------|--------------|----------|------|
| 실시간 집계 | `{FAB}_ATLAS_HID_INOUT` | 매 1분 | 엣지 전환 횟수 집계 |
| 엣지 마스터 | `{FAB}_ATLAS_INFO_HID_INOUT_MAS` | 매일 1회 | HID 간 연결 정의 |
| HID 정보 | `{FAB}_ATLAS_HID_INFO_MAS` | 매일 1회 | HID Zone 물리 특성 |
