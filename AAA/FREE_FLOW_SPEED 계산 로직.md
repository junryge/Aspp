# FREE_FLOW_SPEED 계산 로직 (상세)

## 개요

`FREE_FLOW_SPEED`는 `{FAB}_ATLAS_HID_INFO_MAS` 테이블의 컬럼으로,
**HID 구간 내 모든 RailEdge의 maxVelocity를 단순 평균**한 값이다.

기존 원본 코드에는 이 계산이 없으며, HID 마스터 배치에서 새로 만든 로직이다.

---

## STEP 1. 속도 테이블 파싱

**파일**: `src/data/raw/Mcp75Config.java:592~622`

layout 설정파일의 `[MCP75_VEHICLE_SPEED]` 섹션을 파싱한다.

```
[MCP75_VEHICLE_SPEED]
<SPEED_TYPE_A>
  1 = 1, 2000.0 ;
  2 = 2, 1800.0 ;
  3 = 3, 1500.0 ;
>
```

- `<SPEED_TYPE_A>` → vhlSpeedType (속도 테이블 이름)
- `1 = 1, 2000.0` → level=1, speed=2000.0

파싱 코드:
```java
// src/data/raw/Mcp75Config.java:598~622
String vhl_speed_type = r.trim().replaceAll("<","").replaceAll(">", "");
ConcurrentMap<Integer, RawVhlSpeed> vm = new ConcurrentHashMap<>();

// 각 줄에서 level, speed 추출
String[] st = r.substring(r.lastIndexOf('=') + 1).split("[,]");
vm.put(
    Util.getIntOrZero(st[0].trim()),                          // level (int)
    new RawVhlSpeed(
        Util.getIntOrZero(st[0].trim()),                      // level
        Util.getDoubleOrZero(st[1].replaceAll(";","").trim())  // speed (double)
    )
);

rawVhlSpeedMap.put(vhl_speed_type, vm);
```

결과: `rawVhlSpeedMap = Map<String, Map<Integer, RawVhlSpeed>>`
- Key: 속도 테이블 이름 (예: "SPEED_TYPE_A")
- Value: 속도 등급(level) → RawVhlSpeed(level, speed)

**관련 클래스**: `src/data/raw/RawVhlSpeed.java`
- `level` (int): 속도 등급 번호
- `speed` (double): 해당 등급의 속도값

---

## STEP 2. VHL_SPEED_TYPE 결정

**파일**: `src/data/raw/Mcp75Config.java:499~500`

layout 설정파일에서 어떤 속도 테이블을 사용할지 결정한다.

```
VHL_SPEED_TYPE = "SPEED_TYPE_A";
```

파싱 코드:
```java
// src/data/raw/Mcp75Config.java:499~500
} else if (r.matches("^\\s*VHL_SPEED_TYPE =.*")){
    this.setVhlSpeedType(r.substring(r.lastIndexOf('=') + 1)
        .replaceAll(";", "").replaceAll("\"", "").trim());
}
```

결과: `vhlSpeedType = "SPEED_TYPE_A"`

---

## STEP 3. RailEdge별 maxVelocity 세팅

**파일**: `src/util/DataService.java:700~709`

RailEdge를 생성할 때, 각 RailEdge의 속도 등급(`leftSpeed`)으로 속도 테이블을 조회하여 `maxVelocity`를 세팅한다.

```java
// src/util/DataService.java:700~708
railEdge.setMaxVelocity(
    mcp75ConfigMap.get(mcpName)          // Mcp75Config 가져오기
        .getRawVhlSpeedMap()             // 속도 테이블 맵
        .get(
            mcp75ConfigMap.get(mcpName)
                .getVhlSpeedType()       // "SPEED_TYPE_A"
        )
        .get(rawPoint.getLeftSpeed())    // RawPoint의 leftSpeed (속도 등급 번호)
        .getSpeed()                      // 해당 등급의 실제 속도값
);
railEdge.setVelocity(railEdge.getMaxVelocity());
```

조회 경로:
```
Mcp75Config
  → getRawVhlSpeedMap()        // Map<String, Map<Integer, RawVhlSpeed>>
  → get("SPEED_TYPE_A")        // Map<Integer, RawVhlSpeed>
  → get(rawPoint.getLeftSpeed()) // leftSpeed=2 → RawVhlSpeed(2, 1800.0)
  → getSpeed()                 // 1800.0
```

**관련 클래스**: `src/data/raw/RawPoint.java`
- `leftSpeed` (int): 해당 포인트의 속도 등급 번호

**단위**: `src/map/edge/RailEdge.java:7` 주석에 `// 분속 단위`라고 기재됨

---

## STEP 4. HID별 평균 계산 → FREE_FLOW_SPEED

**파일**: `src/batch/HidEdgeInOutUpdateMasterBatch.java:193~228`

배치 실행 시, 같은 HID에 속한 모든 RailEdge의 maxVelocity를 수집하여 단순 평균을 구한다.

### 4-1. 수집 (193~197행)

```java
// 같은 HID의 RailEdge를 순회하면서 maxVelocity > 0인 것만 리스트에 추가
double maxVelocity = railEdge.getMaxVelocity();
if (maxVelocity > 0) {
    maxVelMap.computeIfAbsent(hidId, k -> new ArrayList<>()).add(maxVelocity);
}
```

### 4-2. 평균 계산 (218~228행)

```java
List<Double> velocities = maxVelMap.get(hidId);
double avgSpeed = 0.0;
if (velocities != null && !velocities.isEmpty()) {
    double sum = 0.0;
    for (Double v : velocities) {
        sum += v;
    }
    avgSpeed = sum / velocities.size();  // 단순 평균
}
tuple.put("FREE_FLOW_SPEED", avgSpeed);
```

### 계산 예시

HID_003에 RailEdge 4개가 속해있고:
| RailEdge | leftSpeed(등급) | maxVelocity(속도) |
|----------|----------------|-------------------|
| edge_A   | 1              | 2000.0            |
| edge_B   | 2              | 1800.0            |
| edge_C   | 1              | 2000.0            |
| edge_D   | 3              | 1500.0            |

```
FREE_FLOW_SPEED = (2000.0 + 1800.0 + 2000.0 + 1500.0) / 4 = 1825.0
```

---

## 전체 흐름 요약

```
layout 설정파일
  ├─ [MCP75_VEHICLE_SPEED] 섹션
  │     → Mcp75Config.rawVhlSpeedMap (속도등급 → 속도값)     ... STEP 1
  │
  ├─ VHL_SPEED_TYPE 설정
  │     → Mcp75Config.vhlSpeedType (사용할 속도 테이블 이름)  ... STEP 2
  │
  └─ [MCP75_POINT] 섹션
        → RawPoint.leftSpeed (각 포인트의 속도 등급)

DataService.java:700~708
  → RailEdge.maxVelocity = 속도테이블[vhlSpeedType][leftSpeed].speed ... STEP 3

HidEdgeInOutUpdateMasterBatch.java:193~228
  → HID별 maxVelocity 수집 → 합계 / 개수 = 단순 평균            ... STEP 4
  → tuple.put("FREE_FLOW_SPEED", avgSpeed)
  → {FAB}_ATLAS_HID_INFO_MAS 테이블에 저장
```

---

## 관련 파일 목록

| 파일 | 역할 |
|------|------|
| `src/data/raw/Mcp75Config.java:592~622` | layout 설정파일 [MCP75_VEHICLE_SPEED] 파싱 |
| `src/data/raw/Mcp75Config.java:499~500` | VHL_SPEED_TYPE 파싱 |
| `src/data/raw/RawVhlSpeed.java` | 속도 데이터 클래스 (level, speed) |
| `src/data/raw/RawPoint.java` | 포인트 데이터 클래스 (leftSpeed = 속도 등급) |
| `src/map/edge/RailEdge.java:7,270~276` | maxVelocity 필드 및 getter/setter |
| `src/util/DataService.java:700~708` | RailEdge별 maxVelocity 세팅 |
| `src/batch/HidEdgeInOutUpdateMasterBatch.java:193~228` | HID별 평균 계산 → FREE_FLOW_SPEED |
