# HID IN/OUT 처리 Java 코드 변경 사항

## 개요
HID IN/OUT 엣지 기반 집계 기능을 **기존 코드에 추가**합니다.
- 기존: HID별 VHL 카운트 (유지)
- 추가: FROM_HIDID → TO_HIDID 엣지 전환 집계 (2009개 엣지)

---

# 테이블 스키마 정의

## 테이블 1: ATLAS_INFO_HID_INOUT_MAS

**용도**: HID Zone 진입/진출 엣지 마스터 데이터 (기준 정보) — 하루 1회 업데이트

| 컬럼명 | 타입 | 설명 | 데이터 소스 |
|--------|------|------|-------------|
| `FROM_HIDID` | INT | 출발 HID Zone ID | `HID_Zone_Master.csv` → `IN_Lanes` 파싱 (예: "3048→3023") |
| `TO_HIDID` | INT | 도착 HID Zone ID | `HID_Zone_Master.csv` → `OUT_Lanes` 파싱 |
| `EDGE_ID` | STRING | 엣지 고유 ID (FROM:TO) | `String.format("%03d:%03d", fromHidId, toHidId)` |
| `FROM_HID_NM` | STRING | 출발 HID Zone 이름 | `HID_Zone_Master.csv` → `Full_Name` |
| `TO_HID_NM` | STRING | 도착 HID Zone 이름 | `HID_Zone_Master.csv` → `Full_Name` |
| `MCP_ID` | STRING | MCP ID | `mcp75ConfigMap.keySet()` 순회 |
| `ZONE_ID` | STRING | Zone ID | `HID_Zone_Master.csv` → `Bay_Zone` |
| `EDGE_TYPE` | STRING | 엣지 유형 | `fromHidId==0 ? "IN" : toHidId==0 ? "OUT" : "INTERNAL"` |
| `UPDATE_DT` | STRING | 마지막 업데이트 일시 | `SimpleDateFormat("yyyy-MM-dd HH:mm:ss")` |

---

## 테이블 2: ATLAS_HID_INFO_MAS

**용도**: HID 상세 정보 마스터 데이터 — 레일 길이, FREE FLOW 속도, 포트 개수 등

| 컬럼명 | 타입 | 설명 | 데이터 소스 |
|--------|------|------|-------------|
| `HID_ID` | INT | HID Zone ID (PK) | `HID_Zone_Master.csv` → `Zone_ID` |
| `HID_NM` | STRING | HID Zone 이름 | `HID_Zone_Master.csv` → `Full_Name` |
| `MCP_ID` | STRING | MCP ID | `mcp75ConfigMap.keySet()` 순회 |
| `ZONE_ID` | STRING | Zone ID | `HID_Zone_Master.csv` → `Bay_Zone` |
| `RAIL_LEN_TOTAL` | DOUBLE | 레일 길이 총합 (mm) | `RailEdge.getLength()` HID별 합계 (DataService.java:691) |
| `FREE_FLOW_SPEED` | DOUBLE | FREE FLOW 속도 (mm/s) | `RailEdge.getMaxVelocity()` HID별 평균 (RaileEdge.java:270) |
| `PORT_CNT_TOTAL` | INT | 포트 개수 총합 | `RailEdge.getPortIdList().size()` HID별 합계 |
| `IN_CNT` | INT | IN Lane 개수 | `HID_Zone_Master.csv` → `IN_Count` |
| `OUT_CNT` | INT | OUT Lane 개수 | `HID_Zone_Master.csv` → `OUT_Count` |
| `VHL_MAX` | INT | 최대 허용 차량 수 | `HID_Zone_Master.csv` → `Vehicle_Max` |
| `ZCU_ID` | STRING | ZCU ID | `HID_Zone_Master.csv` → `ZCU` |
| `UPDATE_DT` | STRING | 마지막 업데이트 일시 | `SimpleDateFormat("yyyy-MM-dd HH:mm:ss")` |

---

## 테이블 3: ATLAS_{FAB}_HID_INOUT

**용도**: HID IN/OUT 1분 집계 데이터 — FABID별 테이블 분리 (M14, M16, M17...)

| 컬럼명 | 타입 | 설명 | 데이터 소스 |
|--------|------|------|-------------|
| `EVENT_DATE` | STRING | 이벤트 날짜 | `SimpleDateFormat("yyyy-MM-dd")` |
| `EVENT_DT` | STRING | 집계 시간 (1분 단위) | `SimpleDateFormat("yyyy-MM-dd HH:mm:00")` |
| `FROM_HIDID` | INT | 출발 HID Zone ID | `vehicle.getHidId()` (previousHidId) - Vhl.java:517 |
| `TO_HIDID` | INT | 도착 HID Zone ID | `currentHidId` 파라미터 - OhtMsgWorkerRunnable.java:357 |
| `TRANS_CNT` | INT | 1분간 전환 횟수 | `hidEdgeBuffer.get(edgeKey)` 집계값 |
| `MCP_NM` | STRING | MCP 이름 | `this.mcpName` - OhtMsgWorkerRunnable.java:9 |
| `ENV` | STRING | 환경 구분 | `Env.getEnv()` - OhtMsgWorkerRunnable.java:505 |

---

# Part 1: OhtMsgWorkerRunnable.java 변경

## 1.1 클래스 필드 추가

```java
// ===== 기존 코드 유지 =====

// ===== 신규 추가: HID 엣지별 전환 카운트 집계 (1분간 모아서 배치 저장) =====
// Key: "fromHidId:toHidId", Value: 전환 횟수
private static ConcurrentMap<String, Integer> hidEdgeBuffer =
    new ConcurrentHashMap<>();
private static long lastHidEdgeFlushTime = System.currentTimeMillis();
private static final long HID_EDGE_FLUSH_INTERVAL = 60000; // 1분
private static final Object hidEdgeFlushLock = new Object();
private static final Object hidEdgeBufferLock = new Object();
```

---

## 1.2 _calculatedVhlCnt() 메소드 수정

### 기존 코드 (OhtMsgWorkerRunnable.java:357-382)
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

### 변경 코드 (기존 유지 + 엣지 집계 추가)
```java
/**
 * HID 구간별 VHL 재적수
 * @param currentHidId 현재 vehicle 이 위치한 railEdge 의 hid 값
 * @param key DataSet 에서 특정 데이터를 호출하기 위한 key 값
 * @param vehicle vehicle 객체
 */
private void _calculatedVhlCnt(int currentHidId, String key, Vhl vehicle) {
    long timer = System.currentTimeMillis();
    int previousHidId = vehicle.getHidId();

    if (previousHidId != currentHidId) {
        // ===== 기존 코드 유지: HID VHL 카운트 =====
        if (currentHidId > 0) {
            String v = String.format("%03d", currentHidId);
            DataService.getDataSet().increaseHidVehicleCnt(key + ":" + v);
        }

        if (previousHidId > 0) {
            String v = String.format("%03d", previousHidId);
            DataService.getDataSet().decreaseHidVehicleCnt(key + ":" + v);
        }
        // ===== 기존 코드 유지 끝 =====

        // ===== 신규 추가: 엣지 전환 카운트 집계 =====
        String edgeKey = String.format("%03d:%03d", previousHidId, currentHidId);
        synchronized (hidEdgeBufferLock) {
            hidEdgeBuffer.merge(edgeKey, 1, Integer::sum);
        }
        // ===== 신규 추가 끝 =====

        vehicle.setHidId(currentHidId);
    }

    // ===== 신규 추가: 1분마다 버퍼 플러시 =====
    if (timer - lastHidEdgeFlushTime >= HID_EDGE_FLUSH_INTERVAL) {
        synchronized (hidEdgeFlushLock) {
            if (timer - lastHidEdgeFlushTime >= HID_EDGE_FLUSH_INTERVAL) {
                flushHidEdgeBuffer();
                lastHidEdgeFlushTime = timer;
            }
        }
    }
    // ===== 신규 추가 끝 =====

    long checkingTime = System.currentTimeMillis() - timer;

    if (checkingTime >= 60000) {
        logger.info("... `number of vehicles per hid section` process took more than 1 minute to complete [elapsed time: {}min]", checkingTime / 60000);
    }
}
```

---

## 1.3 flushHidEdgeBuffer() 메소드 신규 추가

```java
/**
 * HID 엣지 전환 집계 데이터를 Logpresso에 1분 배치 저장
 * 테이블: ATLAS_{FABID}_HID_INOUT
 *
 * 컬럼:
 *   - EVENT_DATE: 이벤트 날짜 (파티션 키)
 *   - EVENT_DT: 집계 시간 (1분 단위)
 *   - FROM_HIDID: 출발 HID Zone ID
 *   - TO_HIDID: 도착 HID Zone ID
 *   - TRANS_CNT: 1분간 전환 횟수
 *   - MCP_NM: MCP 이름
 *   - ENV: 환경 구분
 */
private void flushHidEdgeBuffer() {
    if (hidEdgeBuffer.isEmpty()) {
        return;
    }

    // 버퍼 스냅샷 생성 및 초기화
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

    // 현재 시간 (1분 단위로 정렬)
    SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:00");
    SimpleDateFormat dateOnlyFormat = new SimpleDateFormat("yyyy-MM-dd");
    Date now = new Date();
    String eventDt = dateFormat.format(now);
    String eventDate = dateOnlyFormat.format(now);

    List<Tuple> tuples = new ArrayList<>();

    for (Map.Entry<String, Integer> entry : snapshot.entrySet()) {
        String[] hidIds = entry.getKey().split(":");
        int fromHidId = Integer.parseInt(hidIds[0]);
        int toHidId = Integer.parseInt(hidIds[1]);
        int transCnt = entry.getValue();

        Tuple tuple = new Tuple();
        tuple.put("EVENT_DATE", eventDate);
        tuple.put("EVENT_DT", eventDt);
        tuple.put("FROM_HIDID", fromHidId);
        tuple.put("TO_HIDID", toHidId);
        tuple.put("TRANS_CNT", transCnt);
        tuple.put("MCP_NM", this.mcpName);
        tuple.put("ENV", Env.getEnv());

        tuples.add(tuple);
    }

    if (tuples.isEmpty()) {
        return;
    }

    // FABID별 테이블에 저장
    String tableName = "ATLAS_" + this.fabId + "_HID_INOUT";

    boolean success = LogpressoAPI.setInsertTuples(tableName, tuples, 100);

    if (success) {
        logger.info("HID Edge transitions aggregated: {} - {} records",
                    tableName, tuples.size());
    }
}
```

---

# Part 2: HidMasterBatchJob.java 신규 메소드 추가

## 2.1 updateHidEdgeMasterInfo() 메소드 추가

```java
/**
 * HID Zone 진입/진출 엣지 마스터 데이터 업데이트
 * 테이블: ATLAS_INFO_HID_INOUT_MAS
 *
 * 컬럼:
 *   - FROM_HIDID: 출발 HID Zone ID
 *   - TO_HIDID: 도착 HID Zone ID
 *   - EDGE_ID: 엣지 고유 ID (FROM:TO)
 *   - FROM_HID_NM: 출발 HID Zone 이름
 *   - TO_HID_NM: 도착 HID Zone 이름
 *   - MCP_ID: MCP ID
 *   - ZONE_ID: Zone ID
 *   - EDGE_TYPE: 엣지 유형 (IN/OUT/INTERNAL)
 *   - UPDATE_DT: 마지막 업데이트 일시
 */
@Scheduled(cron = "0 0 0 * * ?")
public void updateHidEdgeMasterInfo() {
    String xmlPath = "/path/to/LAYOUT.XML";
    List<Tuple> tuples = new ArrayList<>();
    SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
    String updateDt = dateFormat.format(new Date());

    try {
        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        DocumentBuilder builder = factory.newDocumentBuilder();
        Document doc = builder.parse(new File(xmlPath));
        doc.getDocumentElement().normalize();

        // HID 정보 맵 구성
        Map<Integer, Element> hidMap = new HashMap<>();
        NodeList hidList = doc.getElementsByTagName("HID");
        for (int i = 0; i < hidList.getLength(); i++) {
            Element hid = (Element) hidList.item(i);
            int hidId = Integer.parseInt(hid.getAttribute("id"));
            hidMap.put(hidId, hid);
        }

        // 엣지 정보 파싱 (HID 간 연결)
        NodeList edgeList = doc.getElementsByTagName("EDGE");
        for (int i = 0; i < edgeList.getLength(); i++) {
            Element edge = (Element) edgeList.item(i);

            int fromHidId = Integer.parseInt(edge.getAttribute("fromHidId"));
            int toHidId = Integer.parseInt(edge.getAttribute("toHidId"));

            Tuple tuple = new Tuple();
            tuple.put("FROM_HIDID", fromHidId);
            tuple.put("TO_HIDID", toHidId);
            tuple.put("EDGE_ID", String.format("%03d:%03d", fromHidId, toHidId));

            // FROM HID 이름
            if (fromHidId == 0) {
                tuple.put("FROM_HID_NM", "OUTSIDE");
            } else if (hidMap.containsKey(fromHidId)) {
                tuple.put("FROM_HID_NM", hidMap.get(fromHidId).getAttribute("name"));
            } else {
                tuple.put("FROM_HID_NM", "HID_" + String.format("%03d", fromHidId));
            }

            // TO HID 이름
            if (toHidId == 0) {
                tuple.put("TO_HID_NM", "OUTSIDE");
            } else if (hidMap.containsKey(toHidId)) {
                tuple.put("TO_HID_NM", hidMap.get(toHidId).getAttribute("name"));
            } else {
                tuple.put("TO_HID_NM", "HID_" + String.format("%03d", toHidId));
            }

            // MCP_ID, ZONE_ID (TO HID 기준)
            if (toHidId > 0 && hidMap.containsKey(toHidId)) {
                Element toHid = hidMap.get(toHidId);
                tuple.put("MCP_ID", toHid.getAttribute("mcpId"));
                tuple.put("ZONE_ID", toHid.getAttribute("zoneId"));
            } else if (fromHidId > 0 && hidMap.containsKey(fromHidId)) {
                Element fromHid = hidMap.get(fromHidId);
                tuple.put("MCP_ID", fromHid.getAttribute("mcpId"));
                tuple.put("ZONE_ID", fromHid.getAttribute("zoneId"));
            } else {
                tuple.put("MCP_ID", "");
                tuple.put("ZONE_ID", "");
            }

            // 엣지 유형 결정
            String edgeType;
            if (fromHidId == 0) {
                edgeType = "IN";       // 외부에서 HID로 진입
            } else if (toHidId == 0) {
                edgeType = "OUT";      // HID에서 외부로 진출
            } else {
                edgeType = "INTERNAL"; // HID 간 이동
            }
            tuple.put("EDGE_TYPE", edgeType);

            tuple.put("UPDATE_DT", updateDt);

            tuples.add(tuple);
        }
    } catch (Exception e) {
        logger.error("Failed to parse LAYOUT.XML for edge info", e);
        return;
    }

    // Full Refresh
    LogpressoAPI.truncateTable("ATLAS_INFO_HID_INOUT_MAS");
    LogpressoAPI.setInsertTuples("ATLAS_INFO_HID_INOUT_MAS", tuples, 100);

    logger.info("HID Edge Master Info updated from LAYOUT.XML: {} records", tuples.size());
}
```

---

## 2.2 updateHidInfoMaster() 메소드 추가

```java
/**
 * HID 상세 정보 마스터 데이터 업데이트
 * 테이블: ATLAS_HID_INFO_MAS
 *
 * 컬럼:
 *   - HID_ID: HID Zone ID
 *   - HID_NM: HID Zone 이름
 *   - MCP_ID: MCP ID
 *   - ZONE_ID: Zone ID
 *   - RAIL_LEN_TOTAL: 레일 길이 총합 (mm)
 *   - FREE_FLOW_SPEED: FREE FLOW 속도 (mm/s)
 *   - PORT_CNT_TOTAL: 포트 개수 총합
 *   - IN_CNT: IN Lane 개수
 *   - OUT_CNT: OUT Lane 개수
 *   - VHL_MAX: 최대 허용 차량 수
 *   - ZCU_ID: ZCU ID
 *   - UPDATE_DT: 마지막 업데이트 일시
 */
@Scheduled(cron = "0 0 0 * * ?")
public void updateHidInfoMaster() {
    String xmlPath = "/path/to/LAYOUT.XML";
    List<Tuple> tuples = new ArrayList<>();
    SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
    String updateDt = dateFormat.format(new Date());

    try {
        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        DocumentBuilder builder = factory.newDocumentBuilder();
        Document doc = builder.parse(new File(xmlPath));
        doc.getDocumentElement().normalize();

        // HID별 레일, 포트 정보 집계용 맵
        Map<Integer, Double> railLengthMap = new HashMap<>();
        Map<Integer, Integer> portCountMap = new HashMap<>();

        // RAIL 정보 집계
        NodeList railList = doc.getElementsByTagName("RAIL");
        for (int i = 0; i < railList.getLength(); i++) {
            Element rail = (Element) railList.item(i);
            int hidId = Integer.parseInt(rail.getAttribute("hidId"));
            double length = Double.parseDouble(rail.getAttribute("length"));

            railLengthMap.merge(hidId, length, Double::sum);
        }

        // PORT 정보 집계
        NodeList portList = doc.getElementsByTagName("PORT");
        for (int i = 0; i < portList.getLength(); i++) {
            Element port = (Element) portList.item(i);
            int hidId = Integer.parseInt(port.getAttribute("hidId"));

            portCountMap.merge(hidId, 1, Integer::sum);
        }

        // HID 정보 파싱
        NodeList hidList = doc.getElementsByTagName("HID");

        for (int i = 0; i < hidList.getLength(); i++) {
            Element hid = (Element) hidList.item(i);
            int hidId = Integer.parseInt(hid.getAttribute("id"));

            Tuple tuple = new Tuple();

            tuple.put("HID_ID", hidId);
            tuple.put("HID_NM", hid.getAttribute("name"));
            tuple.put("MCP_ID", hid.getAttribute("mcpId"));
            tuple.put("ZONE_ID", hid.getAttribute("zoneId"));

            // 레일 길이 총합
            double railLenTotal = railLengthMap.getOrDefault(hidId, 0.0);
            tuple.put("RAIL_LEN_TOTAL", railLenTotal);

            // FREE FLOW 속도 (XML에서 가져오거나 기본값)
            double freeFlowSpeed = 2000.0; // 기본값 2000 mm/s
            if (hid.hasAttribute("freeFlowSpeed")) {
                freeFlowSpeed = Double.parseDouble(hid.getAttribute("freeFlowSpeed"));
            }
            tuple.put("FREE_FLOW_SPEED", freeFlowSpeed);

            // 포트 개수 총합
            int portCntTotal = portCountMap.getOrDefault(hidId, 0);
            tuple.put("PORT_CNT_TOTAL", portCntTotal);

            // 기존 컬럼
            tuple.put("IN_CNT", Integer.parseInt(hid.getAttribute("inCnt")));
            tuple.put("OUT_CNT", Integer.parseInt(hid.getAttribute("outCnt")));
            tuple.put("VHL_MAX", Integer.parseInt(hid.getAttribute("vhlMax")));
            tuple.put("ZCU_ID", hid.getAttribute("zcuId"));
            tuple.put("UPDATE_DT", updateDt);

            tuples.add(tuple);
        }
    } catch (Exception e) {
        logger.error("Failed to parse LAYOUT.XML for HID info", e);
        return;
    }

    // Full Refresh
    LogpressoAPI.truncateTable("ATLAS_HID_INFO_MAS");
    LogpressoAPI.setInsertTuples("ATLAS_HID_INFO_MAS", tuples, 100);

    logger.info("HID Info Master updated from LAYOUT.XML: {} records", tuples.size());
}
```

---

## 2.3 스케줄러 통합 (선택사항)

```java
@Scheduled(cron = "0 0 0 * * ?")
public void updateAllHidMasterTables() {
    logger.info("Starting HID Master Tables update...");

    // 1. 엣지 마스터 업데이트
    updateHidEdgeMasterInfo();

    // 2. HID 상세 정보 업데이트
    updateHidInfoMaster();

    logger.info("HID Master Tables update completed.");
}
```

---

# 변경 요약

## OhtMsgWorkerRunnable.java

| 구분 | 내용 |
|------|------|
| 기존 코드 | **유지** (HID VHL 카운트) |
| 신규 필드 | `hidEdgeBuffer`, `lastHidEdgeFlushTime`, `hidEdgeFlushLock` |
| 신규 메소드 | `flushHidEdgeBuffer()` |
| 수정 메소드 | `_calculatedVhlCnt()` (엣지 집계 로직 추가) |

## HidMasterBatchJob.java

| 구분 | 내용 |
|------|------|
| 신규 메소드 | `updateHidEdgeMasterInfo()` - 엣지 마스터 |
| 신규 메소드 | `updateHidInfoMaster()` - HID 상세 정보 |
| 신규 테이블 | `ATLAS_INFO_HID_INOUT_MAS`, `ATLAS_HID_INFO_MAS` |

---