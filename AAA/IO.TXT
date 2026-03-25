# CnvPortNode 파싱 장애 분석 보고서

**작성일**: 2026-03-25
**대상 파일**:
- `DAP/CnvSocketlOListener.java` (소켓 연결 및 JSON 파싱)
- `DAP/DataService.java` (CnvPortNode 빌드 로직)
- `DAP/zoneinfo_4AFC3201_20250324.json` (실제 Zone 데이터 샘플)
- `DAP/zoneinfo_4AFC3301_20250324.json` (실제 Zone 데이터 샘플)

**증상**: `csil.getRawCnvZoneMap().values()`가 비어있어 `CnvPortNode`가 생성되지 않음

---

## 1. 전체 데이터 흐름

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  DataService.java (line 528~531)                                            │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ for(entry : fabProperties.getConveyorToApiUrl().entrySet()) {        │  │
│  │     CnvSocketIOListener csil = new CnvSocketIOListener(...)          │  │
│  │     fabProperties.getCnvSocketIOListenerMap().put(key, csil)         │  │
│  │     csil.connectAndBuildCnvRawLayout()  ← ② 소켓 연결 시작          │  │
│  │ }                                                                    │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  CnvSocketlOListener.java - connectAndBuildCnvRawLayout() (line 77~225)    │
│                                                                             │
│  ③ socket = IO.socket(connStr)                                  (line 86)  │
│  ④ socket.emit("message", {type:"ZONE_GET_INFO"})               (line 101) │
│  ⑤ while(initialized == false) { Thread.sleep(10); }   ← 무한대기 (line 216) │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                          서버 응답 수신 (비동기)
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  socket.on("message") 핸들러 (line 108~186)                                 │
│                                                                             │
│  ⑥ type == "initializedataSend" 인 경우:                                    │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │ JSONObject zoneJo = new JSONObject(received.getString("data")) │     │
│     │           ↑ ★★★ 여기서 data가 Array면 JSONException 발생 ★★★   │     │
│     │ JSONArray zoneList = new JSONArray()                            │     │
│     │ for(iter : zoneJo.sortedKeys()) {                              │     │
│     │     zoneList.put(zoneJo.getJSONObject(key))                    │     │
│     │ }                                                              │     │
│     │ buildRawMapData(zoneList)    ← ⑦ JSON 파싱                    │     │
│     │ initialized0 = true                                            │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  ⑧ type == "UpdateZoneState" 인 경우:                                       │
│     initialized = true             ← ⑨ 이때서야 대기 루프 종료             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  buildRawMapData(JSONArray zoneList) (line 299~544)                         │
│                                                                             │
│  for(j=0; j<zoneList.length(); j++) {                                      │
│      JSONObject zo = zoneList.getJSONObject(j)                              │
│      ... 각 필드 파싱 (Level, posX, posY, ZoneID 등) ...                    │
│      RawCnvZone rcz = new RawCnvZone(...)                                  │
│      ... AttributeLD / AttributeQS / AttributeLifter 파싱 ...              │
│      rawCnvZoneMap.put(rcz.zoneId, rcz)   ← ⑩ Map에 저장                  │
│  }                                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  DataService.java - CnvPortNode Building (line 1053~1183)                   │
│                                                                             │
│  for(RawCnvZone rcz : csil.getRawCnvZoneMap().values()) {  ← ⑪ 여기가 비어있음 │
│      ... CnvPortNode cnp = new CnvPortNode(...)                             │
│      tmpNodeMap.put(id, cnp)                                                │
│  }                                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 실제 JSON 데이터 분석 결과

### 2.1 데이터 파일 구조

| 파일명 | Zone 수 | 최상위 타입 |
|--------|---------|------------|
| `zoneinfo_4AFC3201_20250324.json` | 1,891 | **JSON Array** `[...]` |
| `zoneinfo_4AFC3301_20250324.json` | 2,084 | **JSON Array** `[...]` |

### 2.2 PhysicalType 분포

| PhysicalType | 의미 | 3201 개수 | 3301 개수 |
|:---:|--------|:---:|:---:|
| 0 | Zone | 1,545 | 1,712 |
| 1 | QS/Lifter Bed | 119 | 135 |
| 2 | Input | 49 | 52 |
| 3 | Output | 57 | 48 |
| 4 | QS | 111 | 127 |
| 5 | Lifter | 4 | 4 |
| **11** | **미정의 (코드에 없음)** | **6** | **6** |

### 2.3 JSON 필드 존재 여부 (파싱 코드 대비)

| 필드명 | 코드에서 파싱 | JSON에 존재 | 비고 |
|--------|:---:|:---:|------|
| `Level` | O | O (전체) | |
| `posX` | O | O (전체) | |
| `posY` | O | O (전체) | |
| `NextZone` | O | O (전체) | |
| `PrevZone` | O | O (전체) | |
| `ZoneDrawCount` | O | O (전체) | |
| `ZoneID` | O | O (전체) | |
| `PhysicalType` | O | O (전체) | |
| `RefDirection` | O | O (전체) | |
| `DisplayName` | O | O (전체) | |
| **`CurrentNode`** | **O (line 365)** | **X (0건)** | **전체 Zone에 없음** |
| **`PrevNode`** | **O (line 369)** | **X (0건)** | **전체 Zone에 없음** |
| `LogicalType` | O (line 373) | △ (8건/1891) | 대부분 없음 |
| `AttributeLD` | O | O (231건) | BED(type=1) 전부 포함 |
| `AttributeQS` | O | O (111건) | QS(type=4) 전부 포함 |
| `AttributeLifter` | O | O (4건) | Lifter(type=5) 전부 포함 |
| `LevelZone` | O (Lifter 내부) | O (4건 모두) | 현재 데이터는 OK |
| `EtherCATID` | X (미파싱) | O (전체) | 코드에서 무시 |
| `EtherCATName` | X (미파싱) | O (전체) | 코드에서 무시 |
| `Profile` | X (미파싱) | O (전체) | 코드에서 무시 |

---

## 3. 발견된 문제점 (총 6건)

### 3.1 [Critical] 서버 데이터 포맷 불일치 — JSON Array vs Object

**위치**: `CnvSocketlOListener.java` line 129

```java
// 현재 코드: data를 JSONObject로 파싱
JSONObject zoneJo = new JSONObject(received.getString("data"));
```

**문제**:
코드는 서버의 `data` 필드가 **JSON Object** (`{"0": {...}, "1": {...}}`) 형태라고 가정합니다.
그러나 실제 JSON 데이터 파일을 보면 **JSON Array** (`[{...}, {...}]`) 형태입니다.

만약 서버가 Array 형태로 데이터를 보내면:
- `new JSONObject(arrayString)` → **JSONException 발생**
- catch 블록(line 182)에서 로그만 남기고 넘어감
- `buildRawMapData()`가 **호출되지 않음**
- `initialized0`이 **영원히 false**
- `initialized`도 **영원히 false**
- `connectAndBuildCnvRawLayout()`의 while 루프(line 216)에서 **영원히 대기**

```
[예상 에러 로그]
ERROR [MESSAGE ERROR] CnvSocketIOListener - org.json.JSONException:
A JSONObject text must begin with '{' at 1 [character 2 line 1]
```

**영향도**: 전체 Conveyor 시스템 초기화 실패. `rawCnvZoneMap`이 완전히 비어있게 됨.

---

### 3.2 [Critical] `CurrentNode`, `PrevNode` 필드 부재

**위치**: `CnvSocketlOListener.java` line 365~371 (파싱), `DataService.java` line 1090~1118 (사용)

```java
// 파싱 코드 (CnvSocketlOListener.java)
if (zo.opt("CurrentNode") != null) {       // ← 항상 null (필드 없음)
    currentNode = zo.getInt("CurrentNode"); // ← 실행 안 됨
}
if (zo.opt("PrevNode") != null) {          // ← 항상 null (필드 없음)
    prevNode = zo.getInt("PrevNode");       // ← 실행 안 됨
}
```

**문제**:
실제 JSON 데이터에 `CurrentNode`와 `PrevNode` 필드가 **단 한 건도 존재하지 않습니다**.
→ `RawCnvZone.currentNode`과 `RawCnvZone.prevNode`은 항상 **-1** (기본값)

**결과**: `DataService.java` line 1090~1118에서 노드 간 연결(currentNodeId, prevNodeId)이 전혀 구성되지 않음.

```java
// DataService.java에서 사용 (line 1090~1118)
if(rcz.currentNode >= 0) {  // ← 항상 false (-1 >= 0 은 false)
    // 이 블록 전체 실행 안 됨
}
if(rcz.prevNode >= 0) {     // ← 항상 false
    // 이 블록 전체 실행 안 됨
}
```

**JSON 데이터에 존재하는 유사 필드**: `NextZone`, `PrevZone` — 이 필드들이 `CurrentNode`/`PrevNode` 역할을 해야 하는 것은 아닌지 확인 필요.

---

### 3.3 [High] `AttributeLifter` 파싱 시 NPE 위험

**위치**: `CnvSocketlOListener.java` line 518~520

```java
JSONArray lvzs = attributeLifter.optJSONArray("LevelZone");  // null 가능

for (int k = 0; k < lvzs.length(); k++) {  // ← lvzs가 null이면 NPE!
```

**문제**:
`optJSONArray("LevelZone")`은 해당 키가 없거나 배열이 아니면 **null**을 반환합니다.
null에 대해 `.length()` 호출 시 **NullPointerException** 발생.

**NPE는 `JSONException`이 아니므로** line 540의 catch 블록에서 잡히지 않습니다.
→ `buildRawMapData()` for문 전체가 중단됨
→ 해당 zone 이후의 **모든 zone이 파싱되지 않음**

**현재 데이터 상태**: 현재 JSON 파일에서는 Lifter가 있는 4건 모두 `LevelZone`을 포함하고 있어 당장은 문제없으나, 데이터가 변경되면 즉시 장애로 이어짐.

---

### 3.4 [High] `levelZoneList.add(lvzm)` 누락

**위치**: `CnvSocketlOListener.java` line 517~527

```java
List<Map<String,Integer>> levelZoneList = new ArrayList<>();
JSONArray lvzs = attributeLifter.optJSONArray("LevelZone");

for (int k = 0; k < lvzs.length(); k++) {
    JSONObject lvz           = lvzs.optJSONObject(k);
    Map<String,Integer> lvzm = new HashMap<>();

    lvzm.put("In",       lvz.optInt("In"));
    lvzm.put("Out",      lvz.optInt("Out"));
    lvzm.put("Position", lvz.optInt("Position"));
    // ❌ 여기에 levelZoneList.add(lvzm); 가 빠져 있음!
}

rcz.lftAttr = rcz.new RawCnvLftAttr(..., levelZoneList);  // 항상 빈 리스트
```

**문제**: `lvzm`을 생성하고 값을 넣지만, `levelZoneList`에 추가하지 않아서 **항상 빈 리스트**가 전달됨.

**수정**:
```java
    lvzm.put("Position", lvz.optInt("Position"));
    levelZoneList.add(lvzm);  // ← 이 줄 추가 필요
}
```

---

### 3.5 [Medium] 예외 처리 범위 부족

**위치**: `CnvSocketlOListener.java` line 540

```java
} catch (JSONException e) {
    logger.error("[buildRawMapData]", e);
}
```

**문제**:
`JSONException`만 catch하고 있어서, `NullPointerException`, `ClassCastException` 등 런타임 예외 발생 시 **전체 for문이 중단**됩니다.

**수정 권장**:
```java
} catch (Exception e) {
    logger.error("[buildRawMapData] zone index={}, error={}", j, e.getMessage(), e);
}
```

---

### 3.6 [Medium] 타임아웃 없는 무한 대기

**위치**: `CnvSocketlOListener.java` line 216~218

```java
while(initialized == false) {
    Thread.sleep(10);
}
```

**문제**:
서버 연결 실패, 메시지 미수신, 파싱 실패 등의 이유로 `initialized`가 true가 되지 않으면 **무한 대기**에 빠짐.
이 경우 `DataService` 초기화 스레드가 영원히 블로킹됩니다.

**수정 권장**:
```java
int timeout = 0;
int maxTimeout = 30000; // 30초
while(initialized == false) {
    Thread.sleep(10);
    timeout += 10;
    if (timeout >= maxTimeout) {
        logger.error("{} initialization timeout! rawCnvZoneMap size: {}", cnvId, rawCnvZoneMap.size());
        break;
    }
}
```

---

## 4. PhysicalType=11 미처리

**위치**: `DataService.java` line 1120~1162

실제 데이터에 `PhysicalType=11`인 zone이 **12건** (3201: 6건, 3301: 6건) 존재하지만, switch문에서 `case 11`이 없어 **default(ZONE 타입)**로 처리됩니다.

```java
switch(rcz.physicalType) {
    case 1 : ... break;  // BED
    case 2 : ... break;  // INPUT
    case 3 : ... break;  // OUTPUT
    case 4 : ... break;  // QS
    case 5 : ... break;  // LFT
    default : type = CNV_NODE_TYPE.ZONE; break;
    // PhysicalType=11은 여기로 옴 — 의도된 것인지 확인 필요
}
```

의도된 동작인지 확인이 필요합니다.

---

## 5. 근본 원인 요약 및 우선순위

```
┌───────────────────────────────────────────────────────────────────────┐
│                    rawCnvZoneMap이 비어있는 원인 트리                   │
│                                                                       │
│  getRawCnvZoneMap().values() == EMPTY                                │
│  │                                                                    │
│  ├─ [원인 A] buildRawMapData()가 호출되지 않음                        │
│  │   ├─ 소켓 연결 실패 (서버 다운, URL 오류)                          │
│  │   ├─ "initializedataSend" 메시지 미수신                            │
│  │   └─ ★ data가 Array 형태 → new JSONObject() 에서 Exception (3.1) │
│  │                                                                    │
│  ├─ [원인 B] buildRawMapData() 도중 중단                              │
│  │   ├─ ★ AttributeLifter의 LevelZone이 null → NPE (3.3)            │
│  │   └─ JSONException만 catch → 다른 예외 시 전체 중단 (3.5)         │
│  │                                                                    │
│  └─ [원인 C] 타이밍/레이스 컨디션                                     │
│       ├─ UpdateZoneState가 initializedataSend보다 먼저 도착            │
│       └─ 타임아웃 없는 무한 대기로 데드락 발생 (3.6)                   │
└───────────────────────────────────────────────────────────────────────┘
```

| 순위 | 이슈 | 영향도 | 발생 확률 |
|:---:|------|:---:|:---:|
| 1 | JSON Array/Object 포맷 불일치 (3.1) | Critical | 높음 |
| 2 | LevelZone null시 NPE (3.3) | Critical | 데이터 의존 |
| 3 | levelZoneList.add() 누락 (3.4) | High | 확정 (100%) |
| 4 | 예외 처리 범위 부족 (3.5) | High | 중간 |
| 5 | 타임아웃 없는 무한 대기 (3.6) | High | 중간 |
| 6 | CurrentNode/PrevNode 필드 부재 (3.2) | High | 확정 (100%) |

---

## 6. 권장 수정 사항

### 6.1 즉시 수정 (Hot Fix)

#### (1) JSON Array/Object 호환 처리 — `CnvSocketlOListener.java` line 129~136

```java
// AS-IS
JSONObject zoneJo  = new JSONObject(received.getString("data"));
JSONArray zoneList = new JSONArray();

for (Iterator<String> iter = zoneJo.sortedKeys(); iter.hasNext();) {
    String key = iter.next();
    zoneList.put(zoneJo.getJSONObject(key));
}

// TO-BE
String dataStr = received.getString("data");
JSONArray zoneList;

if (dataStr.trim().startsWith("[")) {
    // 서버가 Array 형태로 전송한 경우
    zoneList = new JSONArray(dataStr);
} else {
    // 서버가 Object 형태로 전송한 경우 (기존 로직)
    JSONObject zoneJo = new JSONObject(dataStr);
    zoneList = new JSONArray();
    for (Iterator<String> iter = zoneJo.sortedKeys(); iter.hasNext();) {
        String key = iter.next();
        zoneList.put(zoneJo.getJSONObject(key));
    }
}
```

#### (2) levelZoneList.add() 추가 — `CnvSocketlOListener.java` line 527

```java
    lvzm.put("In",       lvz.optInt("In"));
    lvzm.put("Out",      lvz.optInt("Out"));
    lvzm.put("Position", lvz.optInt("Position"));
    levelZoneList.add(lvzm);  // ← 추가
}
```

#### (3) LevelZone null 체크 — `CnvSocketlOListener.java` line 518~520

```java
// AS-IS
JSONArray lvzs = attributeLifter.optJSONArray("LevelZone");
for (int k = 0; k < lvzs.length(); k++) {

// TO-BE
JSONArray lvzs = attributeLifter.optJSONArray("LevelZone");
if (lvzs != null) {
    for (int k = 0; k < lvzs.length(); k++) {
```

#### (4) 예외 처리 범위 확대 — `CnvSocketlOListener.java` line 540

```java
// AS-IS
} catch (JSONException e) {
    logger.error("[buildRawMapData]", e);
}

// TO-BE
} catch (Exception e) {
    logger.error("[buildRawMapData] zone index={}", j, e);
}
```

### 6.2 단기 수정

#### (5) 타임아웃 추가 — `CnvSocketlOListener.java` line 216~218

```java
// AS-IS
while(initialized == false) {
    Thread.sleep(10);
}

// TO-BE
long startTime = System.currentTimeMillis();
long timeoutMs = 30000; // 30초
while(initialized == false) {
    Thread.sleep(10);
    if (System.currentTimeMillis() - startTime > timeoutMs) {
        logger.error("[{}] Conveyor initialization TIMEOUT after {}ms. rawCnvZoneMap size: {}",
                     cnvId, timeoutMs, rawCnvZoneMap.size());
        break;
    }
}
```

### 6.3 확인 필요 사항

#### (6) CurrentNode/PrevNode vs NextZone/PrevZone 매핑 확인

JSON 데이터에 `CurrentNode`와 `PrevNode`가 없고, 대신 `NextZone`과 `PrevZone`이 존재합니다.
서버 측에서 필드명이 변경되었을 가능성이 있으므로, 서버 개발자와 확인이 필요합니다.

```
JSON 데이터 필드:  NextZone, PrevZone  (전 Zone에 존재)
코드 파싱 필드:    CurrentNode, PrevNode (0건 매칭)
```

---

## 7. 디버깅 가이드

### 로그 확인 포인트

문제 발생 시 아래 로그를 순서대로 확인하세요:

| 순서 | 로그 메시지 패턴 | 의미 | 파일:라인 |
|:---:|-----------------|------|----------|
| 1 | `started connect conveyor socket :: {url}` | 소켓 연결 시도 | line 88 |
| 2 | `{cnvId} connected : {socketId}` | 연결 성공 | line 97 |
| 3 | `{cnvId} initializedata received!!!` | 초기 데이터 수신 | line 125 |
| 4 | `[MESSAGE ERROR]` | **★ 파싱 실패** | line 183 |
| 5 | `[buildRawMapData]` | **★ 개별 zone 파싱 실패** | line 541 |
| 6 | `{cnvId} UpdateZoneState received!!!` | 상태 업데이트 수신 | line 153 |
| 7 | `finished...conveyor initializion` | 초기화 완료 | line 219 |

**3번 이후 4번이 나오면**: JSON 포맷 불일치 문제 (이슈 3.1)
**3번이 안 나오면**: 소켓 연결 실패 또는 서버 미응답
**5번이 반복되면**: 개별 zone 파싱 오류 (이슈 3.3, 3.5)

### 즉시 확인 방법

서버로부터 수신되는 `data` 필드의 형태를 확인하려면:

```java
// line 127 이후에 추가
logger.warn("{} data type check - starts with: {}", cnvId,
    received.getString("data").substring(0, Math.min(50, received.getString("data").length())));
```

- `{` 로 시작하면: 현재 코드 정상 동작
- `[` 로 시작하면: **이슈 3.1 확정** → Array 호환 처리 필요
