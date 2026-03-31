# _processHidInout VHL_COUNT_LIMIT, VHL_PRECAUTION 추가 변경사항

## 1. 수정 대상

| 항목 | 내용 |
|------|------|
| 파일 | `ALT/src/process/OhtMsgWorkerRunnable.java` |
| 메소드 | `_processHidInout(int currentHidId, Vhl vehicle, FunctionItem functionItem)` |
| 수정본 위치 | `ALT/src/process/modified/OhtMsgWorkerRunnable.java` |

## 2. 추가된 필드

| 필드명 | 타입 | 설명 | 데이터 소스 |
|--------|------|------|-------------|
| `VHL_COUNT_LIMIT` | INT | 최대 허용 차량 수 (HID) | `RawHid.getVhlMax()` |
| `VHL_PRECAUTION` | INT | 차량 경고 임계값 (HID) | `RawHid.getVhlPreCaution()` |

## 3. 데이터 흐름 (기존 코드에서 확인된 경로)

```
layout.xml
  └─ VEHICLE_MAX         → Mcp75Config.java:245 에서 파싱
  └─ VEHICLE_PRECAUTION  → Mcp75Config.java:247 에서 파싱
        ↓
RawHid 객체 생성 → Mcp75Config.java:261~274
  └─ rawHidMap.put(rh.getId() + ":" + rh.getSubId(), rh)
        ↓
조회 경로:
  DataService.getInstance()
    .getFabPropertiesMap().get(fabId)
    .getMcpPropertiesMap().get(mcpName)
    .getMcp75Config()
    .getRawHidMap()
```

### 참조한 기존 코드 (동일 패턴)

`ALT/src/batch/HidEdgeInOutUpdateMasterBatch.java` 라인 171~180:
```java
// RawHidMap에서 VHL_COUNT_LIMIT, VHL_PRECAUTION 조회
Map<Integer, Integer> vhlCountLimitMap = new HashMap<>();  // HID → vhlMax
Map<Integer, Integer> vhlPrecautionMap = new HashMap<>();  // HID → vhlPreCaution
McpProperties mcpProperties = DataService.getInstance().getFabPropertiesMap().get(fabId).getMcpPropertiesMap().get(mcpName);
if (mcpProperties != null && mcpProperties.getMcp75Config() != null) {
    for (RawHid rawHid : mcpProperties.getMcp75Config().getRawHidMap().values()) {
        vhlCountLimitMap.put(rawHid.getId(), rawHid.getVhlMax());
        vhlPrecautionMap.put(rawHid.getId(), rawHid.getVhlPreCaution());
    }
}
```

### RawHid 클래스 (`ALT/src/data/raw/RawHid.java`)

```java
private int vhlMax          = 0;    // layout.xml VEHICLE_MAX
private int vhlPreCaution   = 0;    // layout.xml VEHICLE_PRECAUTION

public int getVhlMax() {
    return vhlMax;
}

public int getVhlPreCaution() {
    return vhlPreCaution;
}
```

### layout.xml 파싱 (`ALT/src/data/raw/Mcp75Config.java` 라인 244~247)

```java
} else if (cfg.startsWith("VEHICLE_MAX")) {
    vhlMax              = Util.getIntOrZero(cfg.split("=")[1].trim());
} else if (cfg.startsWith("VEHICLE_PRECAUTION")) {
    vhlPreCaution       = Util.getIntOrZero(cfg.split("=")[1].trim());
}
```

## 4. 변경 전후 비교

### 변경 전 (`ALT/src/process/OhtMsgWorkerRunnable.java` 라인 402~447)

```java
private void _processHidInout(int currentHidId, Vhl vehicle, FunctionItem functionItem) {
    int previousHidId = vehicle.getHidId();

    if (previousHidId != currentHidId) {
        String vhlIdFull = vehicle.getId();
        String vhlName = vhlIdFull.substring(vhlIdFull.lastIndexOf(':') + 1);
        String eqpIdFull = vehicle.getEqpId();
        String eqpName = eqpIdFull.substring(eqpIdFull.lastIndexOf(':') + 1);
        String edgeKey = String.format("%03d:%03d:%s:%s:%s:%s:%s",
                previousHidId, currentHidId, this.fabId, this.mcpName,
                vehicle.getFabId(), vhlName, eqpName);

        int transCnt = DataService.getDataSet().getEdgeInOutCountMap()
                .merge(edgeKey, 1, Integer::sum);

        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:00");
        SimpleDateFormat dateOnlyFormat = new SimpleDateFormat("yyyy-MM-dd");
        Date now = new Date();
        String eventDt = dateFormat.format(now);
        String eventDate = dateOnlyFormat.format(now);
        String type = MSG_TYP.OHT.toString() + ".HID.INOUT";
        Map<String, Object> dataMap = new HashMap<>();

        dataMap.put("TYPE", type);
        dataMap.put("FAB_ID", this.fabId);
        dataMap.put("EVENT_DT", eventDt);
        dataMap.put("EVENT_DATE", eventDate);
        dataMap.put("FROM_HIDID", previousHidId);
        dataMap.put("TO_HIDID", currentHidId);
        dataMap.put("VHL_ID", vhlName);
        dataMap.put("EQP_ID", eqpName);
        dataMap.put("TRANS_CNT", transCnt);
        dataMap.put("MCP_NM", this.mcpName);
        dataMap.put("ENV", Env.getEnv());

        for (String tibrvKey : DataService.getInstance().getTibrvSenderLikeMap(fabId + ":send:amos").keySet()) {
            DataService.getInstance().addTibrvMessageQueue(tibrvKey, type, dataMap);
        }
    }
}
```

### 변경 후 (`ALT/src/process/modified/OhtMsgWorkerRunnable.java` 라인 402~463)

```java
private void _processHidInout(int currentHidId, Vhl vehicle, FunctionItem functionItem) {
    int previousHidId = vehicle.getHidId();

    if (previousHidId != currentHidId) {
        String vhlIdFull = vehicle.getId();
        String vhlName = vhlIdFull.substring(vhlIdFull.lastIndexOf(':') + 1);
        String eqpIdFull = vehicle.getEqpId();
        String eqpName = eqpIdFull.substring(eqpIdFull.lastIndexOf(':') + 1);
        String edgeKey = String.format("%03d:%03d:%s:%s:%s:%s:%s",
                previousHidId, currentHidId, this.fabId, this.mcpName,
                vehicle.getFabId(), vhlName, eqpName);

        int transCnt = DataService.getDataSet().getEdgeInOutCountMap()
                .merge(edgeKey, 1, Integer::sum);

        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:00");
        SimpleDateFormat dateOnlyFormat = new SimpleDateFormat("yyyy-MM-dd");
        Date now = new Date();
        String eventDt = dateFormat.format(now);
        String eventDate = dateOnlyFormat.format(now);
        String type = MSG_TYP.OHT.toString() + ".HID.INOUT";
        Map<String, Object> dataMap = new HashMap<>();

        dataMap.put("TYPE", type);
        dataMap.put("FAB_ID", this.fabId);
        dataMap.put("EVENT_DT", eventDt);
        dataMap.put("EVENT_DATE", eventDate);
        dataMap.put("FROM_HIDID", previousHidId);
        dataMap.put("TO_HIDID", currentHidId);
        dataMap.put("VHL_ID", vhlName);
        dataMap.put("EQP_ID", eqpName);
        dataMap.put("TRANS_CNT", transCnt);
        dataMap.put("MCP_NM", this.mcpName);
        dataMap.put("ENV", Env.getEnv());

        // VHL_COUNT_LIMIT, VHL_PRECAUTION → RawHid (layout.xml VEHICLE_MAX, VEHICLE_PRECAUTION)
        int vhlCountLimit = 0;
        int vhlPrecaution = 0;
        McpProperties mcpProperties = DataService.getInstance().getFabPropertiesMap().get(this.fabId).getMcpPropertiesMap().get(this.mcpName);
        if (mcpProperties != null && mcpProperties.getMcp75Config() != null) {
            for (RawHid rawHid : mcpProperties.getMcp75Config().getRawHidMap().values()) {
                if (rawHid.getId() == currentHidId) {
                    vhlCountLimit = rawHid.getVhlMax();
                    vhlPrecaution = rawHid.getVhlPreCaution();
                    break;
                }
            }
        }
        dataMap.put("VHL_COUNT_LIMIT", vhlCountLimit);
        dataMap.put("VHL_PRECAUTION", vhlPrecaution);

        for (String tibrvKey : DataService.getInstance().getTibrvSenderLikeMap(fabId + ":send:amos").keySet()) {
            DataService.getInstance().addTibrvMessageQueue(tibrvKey, type, dataMap);
        }
    }
}
```

## 5. dataMap 전송 필드 목록 (변경 후)

| 필드 | 값 | 비고 |
|------|----|------|
| `TYPE` | `MSG_TYP.OHT + ".HID.INOUT"` | 기존 |
| `FAB_ID` | `this.fabId` | 기존 |
| `EVENT_DT` | `yyyy-MM-dd HH:mm:00` | 기존 |
| `EVENT_DATE` | `yyyy-MM-dd` | 기존 |
| `FROM_HIDID` | `previousHidId` | 기존 |
| `TO_HIDID` | `currentHidId` | 기존 |
| `VHL_ID` | `vhlName` | 기존 |
| `EQP_ID` | `eqpName` | 기존 |
| `TRANS_CNT` | `transCnt` | 기존 |
| `MCP_NM` | `this.mcpName` | 기존 |
| `ENV` | `Env.getEnv()` | 기존 |
| `VHL_COUNT_LIMIT` | `RawHid.getVhlMax()` | **추가** |
| `VHL_PRECAUTION` | `RawHid.getVhlPreCaution()` | **추가** |
