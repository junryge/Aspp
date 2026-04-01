# HID_INOUT에 FREE_FLOW_SPEED, HID_VALUE 필드 추가 변경사항

## 1. 수정 대상

| 항목 | 파일 | 수정본 위치 |
|------|------|-------------|
| 실시간 TibRV 전송 | `ALT/src/process/OhtMsgWorkerRunnable.java` | `ALT/src/process/modified/OhtMsgWorkerRunnable.java` |
| 1분 배치 Logpresso 저장 | `ALT/src/batch/HidEdgeInOutQueueFlushBatch.java` | `ALT/src/batch/modified/HidEdgeInOutQueueFlushBatch.java` |

## 2. 추가된 필드

| 필드명 | 타입 | 설명 | 데이터 소스 |
|--------|------|------|-------------|
| `FREE_FLOW_SPEED` | DOUBLE | 실시간 1분 구간 평균속도 (분속, m/min) | `RailEdge.getVelocity()` HID별 평균 |
| `HID_VALUE` | INT | 현재 HID 대기 차량 수 | `DataSet.getHidVehicleCountMap()` |

## 3. 데이터 흐름 (기존 코드에서 확인된 경로)

### FREE_FLOW_SPEED

```
UDP 메시지 수신
  └─ OhtMsgWorkerRunnable._updateVehicle()
      └─ _buildRailVelocity() → _setRailEdgeVelocity()
          └─ RailEdge.addVelocity(speed)  ← 지수평활법으로 실시간 업데이트
              └─ RailEdge.java:286~306

조회 경로:
  DataService.getDataSet().getEdgeMap()  ← 전체 RailEdge 맵
    → RailEdge.getHIDId()               ← RailEdge.java:324 (해당 edge의 HID zone ID)
    → RailEdge.getVelocity()            ← RailEdge.java:278 (실시간 지수평활 속도)
    → 해당 HID의 모든 RailEdge velocity 평균 = FREE_FLOW_SPEED
```

**속도 계산 공식** (OhtMsgWorkerRunnable.java:784~788):
```java
double ran_distance = lastRailEdge.getLength() - vehicle.getLastUdpState().distance + vehicle.getDistance();
long elapsed = vehicle.getReceivedTime() - vehicle.getLastUdpState().receivedTime;
double speed = ran_distance / (double)elapsed * 60.0;  // m/min 변환
lastRailEdge.addVelocity(speed);
```

**참조한 기존 코드 패턴** (HidEdgeInOutUpdateMasterBatch.java:190~228):
```java
// maxVelocity 기준으로 HID별 평균 계산 (하루 1회 마스터 배치)
double maxVelocity = railEdge.getMaxVelocity();
if (maxVelocity > 0) {
    maxVelMap.computeIfAbsent(hidId, k -> new ArrayList<>()).add(maxVelocity);
}
// ...
avgSpeed = sum / velocities.size();
tuple.put("FREE_FLOW_SPEED", avgSpeed);
```

### HID_VALUE

```
UDP 메시지 수신
  └─ OhtMsgWorkerRunnable._updateVehicle()
      └─ _calculatedVhlCnt()  ← OhtMsgWorkerRunnable.java:369~392
          └─ DataService.getDataSet().increaseHidVehicleCnt(key)  ← DataSet.java:1370
          └─ DataService.getDataSet().decreaseHidVehicleCnt(key)  ← DataSet.java:1385

조회 경로:
  DataService.getDataSet().getHidVehicleCountMap()  ← DataSet.java:1366
    → 키 형식: {fabId}:{mcpName}:{hidId(3자리)}
    → 값: Integer (현재 차량 수)
```

**DataSet.java:1366~1396**:
```java
public ConcurrentMap<String, Integer> getHidVehicleCountMap() {
    return hidVehicleCountMap;
}

public void increaseHidVehicleCnt(String key) {
    int count;
    if (hidVehicleCountMap.containsKey(key)) {
        count = hidVehicleCountMap.get(key);
        hidVehicleCountMap.put(key, ++count);
    } else {
        count = 1;
        hidVehicleCountMap.put(key, count);
    }
}

public void decreaseHidVehicleCnt(String key) {
    int count;
    if (hidVehicleCountMap.containsKey(key)) {
        count = hidVehicleCountMap.get(key);
        if (count > 0) {
            hidVehicleCountMap.put(key, --count);
        }
    }
}
```

## 4. 변경 전후 비교

### 파일 1: OhtMsgWorkerRunnable.java — `_processHidInout()`

#### 변경 전
```java
dataMap.put("MCP_NM", this.mcpName);
dataMap.put("ENV", Env.getEnv());

for (String tibrvKey : DataService.getInstance().getTibrvSenderLikeMap(...).keySet()) {
    DataService.getInstance().addTibrvMessageQueue(tibrvKey, type, dataMap);
}
```

#### 변경 후
```java
dataMap.put("MCP_NM", this.mcpName);
dataMap.put("ENV", Env.getEnv());

// FREE_FLOW_SPEED → 현재 HID 구간 RailEdge velocity 평균 (실시간 1분 구간 평균속도)
double sumVelocity = 0.0;
int velCount = 0;
for (AbstractEdge ae : DataService.getDataSet().getEdgeMap().values()) {
    if (ae instanceof RailEdge) {
        RailEdge re = (RailEdge) ae;
        if (re.getHIDId() == currentHidId && re.getVelocity() > 0) {
            sumVelocity += re.getVelocity();
            velCount++;
        }
    }
}
double freeFlowSpeed = velCount > 0 ? sumVelocity / velCount : 0.0;
dataMap.put("FREE_FLOW_SPEED", freeFlowSpeed);

// HID_VALUE → 현재 HID 대기 차량 수 (DataSet.hidVehicleCountMap)
String hidKey = this.fabId + ":" + this.mcpName + ":" + String.format("%03d", currentHidId);
int hidValue = DataService.getDataSet().getHidVehicleCountMap().getOrDefault(hidKey, 0);
dataMap.put("HID_VALUE", hidValue);

for (String tibrvKey : DataService.getInstance().getTibrvSenderLikeMap(...).keySet()) {
    DataService.getInstance().addTibrvMessageQueue(tibrvKey, type, dataMap);
}
```

### 파일 2: HidEdgeInOutQueueFlushBatch.java — `execute()`

#### 변경 전
```java
tuple.put("MCP_NM", mcpName);
tuple.put("ENV", Env.getEnv());

if (fabIdTuples.get(fabId) == null) {
```

#### 변경 후
```java
tuple.put("MCP_NM", mcpName);
tuple.put("ENV", Env.getEnv());

// FREE_FLOW_SPEED → 현재 HID 구간 RailEdge velocity 평균 (실시간 1분 구간 평균속도)
double sumVelocity = 0.0;
int velCount = 0;
for (AbstractEdge ae : DataService.getDataSet().getEdgeMap().values()) {
    if (ae instanceof RailEdge) {
        RailEdge re = (RailEdge) ae;
        if (re.getHIDId() == toHidId && re.getVelocity() > 0) {
            sumVelocity += re.getVelocity();
            velCount++;
        }
    }
}
double freeFlowSpeed = velCount > 0 ? sumVelocity / velCount : 0.0;
tuple.put("FREE_FLOW_SPEED", freeFlowSpeed);

// HID_VALUE → 현재 HID 대기 차량 수 (DataSet.hidVehicleCountMap)
String hidKey = fabId + ":" + mcpName + ":" + String.format("%03d", toHidId);
int hidValue = DataService.getDataSet().getHidVehicleCountMap().getOrDefault(hidKey, 0);
tuple.put("HID_VALUE", hidValue);

if (fabIdTuples.get(fabId) == null) {
```

## 5. dataMap / Tuple 전송 필드 목록 (변경 후)

| 필드 | 값 | 비고 |
|------|----|------|
| `TYPE` | `MSG_TYP.OHT + ".HID.INOUT"` | 기존 (TibRV만) |
| `FAB_ID` | fabId | 기존 |
| `EVENT_DT` | `yyyy-MM-dd HH:mm:00` | 기존 |
| `EVENT_DATE` | `yyyy-MM-dd` | 기존 |
| `FROM_HIDID` | previousHidId / fromHidId | 기존 |
| `TO_HIDID` | currentHidId / toHidId | 기존 |
| `VHL_ID` | vhlName / vhlId | 기존 |
| `EQP_ID` | eqpName / eqpId | 기존 |
| `TRANS_CNT` | transCnt | 기존 |
| `MCP_NM` | mcpName | 기존 |
| `ENV` | Env.getEnv() | 기존 |
| `FREE_FLOW_SPEED` | HID 구간 RailEdge velocity 평균 | **추가** |
| `HID_VALUE` | 현재 HID 대기 차량 수 | **추가** |

## 6. 스위치 제어

별도 스위치 추가 불필요. 기존 `FunctionType.HID_INOUT` 스위치로 제어됨:
```java
// OhtMsgWorkerRunnable.java:239
if (functionItem.getUseFunction(FunctionType.HID_INOUT)) {
    this._processHidInout(hidId, vehicle, functionItem);
}
```
