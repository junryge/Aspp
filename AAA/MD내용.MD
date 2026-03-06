# OhtMsgWorkerRunnable 수정 내역

---

## 1. `_calculatedVhlCnt()` 호출부
`vehicle` 까지 3개 파라미터였던 호출부에 `functionItem` 파라미터가 추가됨

**전**
```java
if (functionItem.getUseFunction(FunctionType.VHL_CNT)) {
    this._calculatedVhlCnt(
            hidId,
            requiredKey,
            vehicle
    );
}
```

**후**
```java
if (functionItem.getUseFunction(FunctionType.VHL_CNT)) {
    this._calculatedVhlCnt(
            hidId,
            requiredKey,
            vehicle,
            functionItem  // 추가
    );
}
```

---

## 2. `_updateVehicle()` 내부 — HID_INOUT 블록 추가
VHL_CNT 블록과 Stage Command Monitoring 블록 사이에 HID_INOUT 집계 블록이 새로 추가됨

**전**
```java
// HID 구간 별 VHL 수 계산
if (functionItem.getUseFunction(FunctionType.VHL_CNT)) {
    this._calculatedVhlCnt(
            hidId,
            requiredKey,
            vehicle
    );
}
//~HID 구간 별 VHL 수 계산

// Stage Command Monitoring
if (functionItem.getUseFunction(FunctionType.MAP_FILE_REFRESH)) {
    this._processStageCommandMonitoring(
            detailStatus,
            machineKey,
            machineId,
            destinationPortId,
            systemsDateTime
    );
}
```

**후**
```java
// HID 구간 별 VHL 수 계산
if (functionItem.getUseFunction(FunctionType.VHL_CNT)) {
    this._calculatedVhlCnt(
            hidId,
            requiredKey,
            vehicle,
            functionItem
    );
}
//~HID 구간 별 VHL 수 계산

// HID IN/OUT 엣지 집계 (신규 추가)
if (functionItem.getUseFunction(FunctionType.HID_INOUT)) {
    this._processHidInout(hidId, vehicle, functionItem);
}
//~HID IN/OUT 엣지 집계

// Stage Command Monitoring
if (functionItem.getUseFunction(FunctionType.MAP_FILE_REFRESH)) {
    this._processStageCommandMonitoring(
            detailStatus,
            machineKey,
            machineId,
            destinationPortId,
            systemsDateTime
    );
}
```

---

## 3. `_calculatedVhlCnt()` 시그니처
내부 로직은 동일하고 시그니처에 `FunctionItem functionItem` 파라미터만 추가됨

**전**
```java
private void _calculatedVhlCnt(int currentHidId, String key, Vhl vehicle) {
    long timer = System.currentTimeMillis();
    int previousHidId = vehicle.getHidId();

    if (previousHidId != currentHidId) {
        if (currentHidId > 0) {
            String v = String.format("%03d", currentHidId);
            DataService.getDataSet().increaseHidVehicleCnt(key + ":" + v);
        }

        if (previousHidId > 0) {
            String v = String.format("%03d", previousHidId);
            DataService.getDataSet().decreaseHidVehicleCnt(key + ":" + v);
        }

        vehicle.setHidId(currentHidId);
    }

    long checkingTime = System.currentTimeMillis() - timer;

    if (checkingTime >= 60000) {
        logger.info("... `number of vehicles per hid section` process took more than 1 minute to complete [elapsed time: {}min]", checkingTime / 60000);
    }
}
```

**후**
```java
private void _calculatedVhlCnt(int currentHidId, String key, Vhl vehicle, FunctionItem functionItem) {
    long timer = System.currentTimeMillis();
    int previousHidId = vehicle.getHidId();

    if (previousHidId != currentHidId) {
        if (currentHidId > 0) {
            String v = String.format("%03d", currentHidId);
            DataService.getDataSet().increaseHidVehicleCnt(key + ":" + v);
        }

        if (previousHidId > 0) {
            String v = String.format("%03d", previousHidId);
            DataService.getDataSet().decreaseHidVehicleCnt(key + ":" + v);
        }

        vehicle.setHidId(currentHidId);
    }

    long checkingTime = System.currentTimeMillis() - timer;

    if (checkingTime >= 60000) {
        logger.info("... `number of vehicles per hid section` process took more than 1 minute to complete [elapsed time: {}min]", checkingTime / 60000);
    }
}
```

---

## 4. 신규 메서드 `_processHidInout()`
OHT가 HID 구간을 넘어갈 때마다 감지해서 버퍼에 카운트 누적, 1분마다 플러시 트리거, 하루 1회 마스터 테이블 업데이트 트리거

**전**
```java
// 없음
```

**후**
```java
private void _processHidInout(int currentHidId, Vhl vehicle, FunctionItem functionItem) {
    long timer = System.currentTimeMillis();
    int previousHidId = vehicle.getHidId();

    if (previousHidId != currentHidId) {
        String vhlIdFull = vehicle.getId();
        String vhlName = vhlIdFull.substring(vhlIdFull.lastIndexOf(':') + 1);
        String eqpIdFull = vehicle.getEqpId();
        String eqpName = eqpIdFull.substring(eqpIdFull.lastIndexOf(':') + 1);
        String edgeKey = String.format("%03d:%03d:%s:%s:%s",
                previousHidId, currentHidId,
                vehicle.getFabId(), vhlName, eqpName);
        synchronized (hidEdgeBufferLock) {
            hidEdgeBuffer.merge(edgeKey, 1, Integer::sum);
        }
    }

    if (timer - lastHidEdgeFlushTime >= HID_EDGE_FLUSH_INTERVAL) {
        synchronized (hidEdgeFlushLock) {
            if (timer - lastHidEdgeFlushTime >= HID_EDGE_FLUSH_INTERVAL) {
                flushHidEdgeBuffer();
                lastHidEdgeFlushTime = timer;
            }
        }
    }

    String today = new SimpleDateFormat("yyyy-MM-dd").format(new Date(timer));
    if (!today.equals(hidMasterLastUpdateDate)) {
        synchronized (hidMasterUpdateLock) {
            if (!today.equals(hidMasterLastUpdateDate)) {
                try {
                    _updateHidMasterTables();
                    hidMasterLastUpdateDate = today;
                } catch (Exception e) {
                    logger.error("HID Master Tables update failed", e);
                }
            }
        }
    }
}
```

---

## 5. 신규 메서드 `flushHidEdgeBuffer()`
버퍼에 쌓인 HID 전환 집계 데이터를 스냅샷으로 뽑아서 Logpresso `{FAB}_ATLAS_HID_INOUT` 테이블에 1분 단위로 저장

**전**
```java
// 없음
```

**후**
```java
private void flushHidEdgeBuffer() {
    if (hidEdgeBuffer.isEmpty()) {
        return;
    }

    Map<String, Integer> snapshot = new HashMap<>();
    synchronized (hidEdgeBufferLock) {
        for (Map.Entry<String, Integer> entry : hidEdgeBuffer.entrySet()) {
            int count = entry.getValue();
            if (count > 0) {
                snapshot.put(entry.getKey(), count);
            }
        }
        hidEdgeBuffer.clear();
    }

    if (snapshot.isEmpty()) {
        return;
    }

    SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:00");
    SimpleDateFormat dateOnlyFormat = new SimpleDateFormat("yyyy-MM-dd");
    Date now = new Date();
    String eventDt = dateFormat.format(now);
    String eventDate = dateOnlyFormat.format(now);

    List<Tuple> tuples = new ArrayList<>();

    for (Map.Entry<String, Integer> entry : snapshot.entrySet()) {
        String[] parts = entry.getKey().split(":");
        int fromHidId = Integer.parseInt(parts[0]);
        int toHidId = Integer.parseInt(parts[1]);
        String fabId = parts[2];
        String vhlId = parts[3];
        String eqpId = parts[4];
        int transCnt = entry.getValue();

        Tuple tuple = new Tuple();
        tuple.put("EVENT_DATE", eventDate);
        tuple.put("EVENT_DT", eventDt);
        tuple.put("FROM_HIDID", fromHidId);
        tuple.put("TO_HIDID", toHidId);
        tuple.put("TRANS_CNT", transCnt);
        tuple.put("FAB_ID", fabId);
        tuple.put("VHL_ID", vhlId);
        tuple.put("EQP_ID", eqpId);
        tuple.put("MCP_NM", this.mcpName);
        tuple.put("ENV", Env.getEnv());
        tuples.add(tuple);
    }

    String tableName = this.fabId + "_ATLAS_HID_INOUT";
    boolean success = LogpressoAPI.setInsertTuples(tableName, tuples, 100);

    if (success) {
        logger.info("HID Edge flush: {} - {} records", tableName, tuples.size());
    }
}
```

---

## 6. 신규 메서드 `_updateHidMasterTables()`
하루 1회 layout.zip 파일 경로 확인 후 엣지 마스터와 HID 상세정보 업데이트를 순서대로 호출

**전**
```java
// 없음
```

**후**
```java
private void _updateHidMasterTables() {
    logger.info("Starting HID Master Tables update [fab: {}]", this.fabId);

    FabProperties fabProperties = DataService.getInstance().getFabPropertiesMap().get(this.fabId);

    if (fabProperties == null) {
        logger.warn("[HID Master] FabProperties not found, SKIP [fab: {}]", this.fabId);
        return;
    }

    String mapDir = fabProperties.getMapDir();
    File mapDirFile = new File(mapDir);

    if (!mapDirFile.exists() || !mapDirFile.isDirectory()) {
        logger.warn("[HID Master] map directory not found, SKIP [fab: {} | path: {}]", this.fabId, mapDir);
        return;
    }

    File[] layoutZipFiles = mapDirFile.listFiles((dir, name) -> name.endsWith(".layout.zip"));

    if (layoutZipFiles == null || layoutZipFiles.length == 0) {
        logger.warn("[HID Master] *.layout.zip not found, SKIP [fab: {} | path: {}]", this.fabId, mapDir);
        return;
    }

    File layoutZipFile = layoutZipFiles[0];
    logger.info("[HID Master] layout.zip found [fab: {} | file: {}]", this.fabId, layoutZipFile.getName());

    _updateHidEdgeMasterInfo(layoutZipFile);  // 테이블 1
    _updateHidInfoMaster();                   // 테이블 2

    logger.info("HID Master Tables update completed [fab: {}]", this.fabId);
}
```

---

## 7. 신규 메서드 `_updateHidEdgeMasterInfo()`
edgeMap 순회하며 FAB 내 HID 간 전환 엣지를 구성해서 Logpresso `{FAB}_ATLAS_INFO_HID_INOUT_MAS` 테이블에 저장, 엣지 유형은 IN/OUT/INTERNAL 으로 분류

**전**
```java
// 없음
```

**후**
```java
private void _updateHidEdgeMasterInfo(File layoutZipFile) {
    List<Tuple> tuples = new ArrayList<>();
    SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
    String updateDt = dateFormat.format(new Date());

    try {
        ConcurrentMap<String, AbstractEdge> edgeMap = DataService.getDataSet().getEdgeMap();
        Map<Integer, String> hidNameMap = new HashMap<>();
        Set<String> processedEdges = new HashSet<>();

        for (Map.Entry<String, AbstractEdge> entry : edgeMap.entrySet()) {
            if (!(entry.getValue() instanceof RailEdge)) continue;
            RailEdge railEdge = (RailEdge) entry.getValue();
            if (!railEdge.getFabId().equals(this.fabId)) continue;
            int hidId = railEdge.getHIDId();
            if (hidId > 0) {
                hidNameMap.putIfAbsent(hidId, "HID_" + String.format("%03d", hidId));
            }
        }

        for (Map.Entry<String, AbstractEdge> entry : edgeMap.entrySet()) {
            if (!(entry.getValue() instanceof RailEdge)) continue;
            RailEdge railEdge = (RailEdge) entry.getValue();
            if (!railEdge.getFabId().equals(this.fabId)) continue;
            int fromHidId = railEdge.getHIDId();
            String toNodeId = railEdge.getToNodeId();

            for (Map.Entry<String, AbstractEdge> nextEntry : edgeMap.entrySet()) {
                if (!(nextEntry.getValue() instanceof RailEdge)) continue;
                RailEdge nextRailEdge = (RailEdge) nextEntry.getValue();
                if (!nextRailEdge.getFabId().equals(this.fabId)) continue;

                if (nextRailEdge.getFromNodeId().equals(toNodeId)) {
                    int toHidId = nextRailEdge.getHIDId();

                    if (fromHidId != toHidId && (fromHidId > 0 || toHidId > 0)) {
                        String edgeKey = fromHidId + ":" + toHidId;
                        if (processedEdges.contains(edgeKey)) continue;
                        processedEdges.add(edgeKey);

                        Tuple tuple = new Tuple();
                        tuple.put("FROM_HIDID", fromHidId);
                        tuple.put("TO_HIDID", toHidId);
                        tuple.put("EDGE_ID", String.format("%03d:%03d", fromHidId, toHidId));
                        tuple.put("FROM_HID_NM", fromHidId == 0 ? "OUTSIDE"
                            : hidNameMap.getOrDefault(fromHidId, "HID_" + String.format("%03d", fromHidId)));
                        tuple.put("TO_HID_NM", toHidId == 0 ? "OUTSIDE"
                            : hidNameMap.getOrDefault(toHidId, "HID_" + String.format("%03d", toHidId)));
                        tuple.put("MCP_ID", this.mcpName);
                        tuple.put("ZONE_ID", "");

                        String edgeType;
                        if (fromHidId == 0)      edgeType = "IN";
                        else if (toHidId == 0)   edgeType = "OUT";
                        else                     edgeType = "INTERNAL";

                        tuple.put("EDGE_TYPE", edgeType);
                        tuple.put("UPDATE_DT", updateDt);
                        tuples.add(tuple);
                    }
                }
            }
        }
    } catch (Exception e) {
        logger.error("[HID Master] Failed to build edge master info [fab: {}]", this.fabId, e);
        return;
    }

    if (tuples.isEmpty()) {
        logger.warn("[HID Master] No edge data found [fab: {}]", this.fabId);
        return;
    }

    String tableName = this.fabId + "_ATLAS_INFO_HID_INOUT_MAS";
    LogpressoAPI.setInsertTuples(tableName, tuples, 100);
    logger.info("[HID Master] Edge Master updated: {} - {} records", tableName, tuples.size());
}
```

---

## 8. 신규 메서드 `_updateHidInfoMaster()`
edgeMap 순회하며 HID별 레일 길이 합계, 최고속도 평균, 포트 수 합계를 집계해서 Logpresso `{FAB}_ATLAS_HID_INFO_MAS` 테이블에 저장

**전**
```java
// 없음
```

**후**
```java
private void _updateHidInfoMaster() {
    List<Tuple> tuples = new ArrayList<>();
    SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
    String updateDt = dateFormat.format(new Date());

    try {
        ConcurrentMap<String, AbstractEdge> edgeMap = DataService.getDataSet().getEdgeMap();
        Map<Integer, Double> railLenMap = new HashMap<>();
        Map<Integer, List<Double>> maxVelMap = new HashMap<>();
        Map<Integer, Integer> portCntMap = new HashMap<>();

        for (Map.Entry<String, AbstractEdge> entry : edgeMap.entrySet()) {
            if (!(entry.getValue() instanceof RailEdge)) continue;
            RailEdge railEdge = (RailEdge) entry.getValue();
            if (!railEdge.getFabId().equals(this.fabId)) continue;
            int hidId = railEdge.getHIDId();
            if (hidId <= 0) continue;

            railLenMap.merge(hidId, railEdge.getLength(), Double::sum);

            double maxVelocity = railEdge.getMaxVelocity();
            if (maxVelocity > 0) {
                maxVelMap.computeIfAbsent(hidId, k -> new ArrayList<>()).add(maxVelocity);
            }

            List<String> portList = railEdge.getPortIdList();
            if (portList != null && !portList.isEmpty()) {
                portCntMap.merge(hidId, portList.size(), Integer::sum);
            }
        }

        for (Integer hidId : railLenMap.keySet()) {
            Tuple tuple = new Tuple();
            tuple.put("HID_ID", hidId);
            tuple.put("HID_NM", "HID_" + String.format("%03d", hidId));
            tuple.put("MCP_ID", this.mcpName);
            tuple.put("ZONE_ID", "");
            tuple.put("RAIL_LEN_TOTAL", railLenMap.getOrDefault(hidId, 0.0));

            List<Double> velocities = maxVelMap.get(hidId);
            double avgSpeed = 0.0;
            if (velocities != null && !velocities.isEmpty()) {
                double sum = 0.0;
                for (Double v : velocities) sum += v;
                avgSpeed = sum / velocities.size();
            }
            tuple.put("FREE_FLOW_SPEED", avgSpeed);
            tuple.put("PORT_CNT_TOTAL", portCntMap.getOrDefault(hidId, 0));
            tuple.put("IN_CNT", 0);
            tuple.put("OUT_CNT", 0);
            tuple.put("VHL_MAX", 0);
            tuple.put("ZCU_ID", "");
            tuple.put("UPDATE_DT", updateDt);
            tuples.add(tuple);
        }
    } catch (Exception e) {
        logger.error("[HID Master] Failed to build HID info [fab: {}]", this.fabId, e);
        return;
    }

    if (tuples.isEmpty()) {
        logger.warn("[HID Master] No HID info data found [fab: {}]", this.fabId);
        return;
    }

    String tableName = this.fabId + "_ATLAS_HID_INFO_MAS";
    LogpressoAPI.setInsertTuples(tableName, tuples, 100);
    logger.info("[HID Master] HID Info Master updated: {} - {} records", tableName, tuples.size());
}
```
