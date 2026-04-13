# VHL ERROR 발생 시 영향받는 RAIL 산출 로직

## 개요

VHL(차량)이 레일 위에서 에러로 멈추면, 해당 차량으로 인해 **통행이 막히는 레일 구간**을 자동 산출하여 `AFFECT_ADDR_LIST`(DB 컬럼: `FALR_RAISE_ADDR_LVAL`)에 저장한다.

---

## 산출 알고리즘

레일 네트워크는 **노드(RailNode)** 와 **구간(RailEdge)** 으로 이루어진 그래프 구조이다.

### 노드의 핵심 속성

| 속성 | 의미 | 영향 범위 추적에서의 역할 |
|------|------|--------------------------|
| `isRailBranch` | 레일이 여러 갈래로 **나뉘는** 지점 (분기점) | **역방향 추적 종료 조건** — 분기점 이전의 VHL은 다른 경로로 우회 가능 |
| `isRailJunction` | 여러 레일이 하나로 **합쳐지는** 지점 (합류점) | **정방향 추적 종료 조건** — 합류점 이후는 다른 경로에서 진입 가능 |

### 계산 과정

```
VHL 멈춘 구간: RailEdge (FromNode 4331 → ToNode 4332)

[1] 멈춘 구간 자체를 영향 목록에 추가

[2] ToNode(4332)에서 정방향으로 추적
    → 다음 RailEdge → 다음 → ...
    → Junction(합류점) 만나면 STOP

[3] FromNode(4331)에서 역방향으로 추적
    → 이전 RailEdge → 이전 → ...
    → Branch(분기점) 만나면 STOP

결과: Branch ~ Junction 사이의 모든 RailEdge 주소 = 영향받는 RAIL
```

---

## 코드 흐름

### 1. VHL OFF 감지

- **파일**: `process/OhtMsgWorkerRunnable.java:627-638`
- UDP 메시지로 차량 상태를 수신하며, 에러코드가 VHL_OFF 목록에 포함되면 알람 발생

```java
if (alarmCodeMap.get(FunctionType.VHL_OFF.getKey()).contains(errorCode)) {
    // VHL OFF 알람 → 영향 범위 계산 시작
}
```

### 2. Navigator 생성 및 영향 범위 계산

- **파일**: `process/OhtMsgWorkerRunnable.java:731-733`
- 멈춘 RailEdge를 Navigator에 전달하여 영향받는 주소/포트를 산출

```java
Navigator navigator    = new Navigator(railEdge);
Set<String> addressSet = navigator.getAffectedRailSet();
List<String> portList  = navigator.getAffectedPortSortedList();
```

### 3. 정방향 추적 (ToNode 기준)

- **파일**: `navi/Navigator.java:157-202` (`_searchRailAffectedBasedOnToNode`)
- ToNode에서 출발하여 `FromNode2EdgeMap`을 따라 앞으로 이동
- **종료 조건**: `isRailJunction == true` (합류점 도달)

```java
if (toNode.isRailJunction()) {
    return item;  // 합류점 → 추적 종료
} else {
    // 다음 RailEdge의 address, portIdList를 수집하며 재귀 추적
}
```

### 4. 역방향 추적 (FromNode 기준)

- **파일**: `navi/Navigator.java:109-154` (`_searchRailAffectedBasedOnFromNode`)
- FromNode에서 출발하여 `ToNode2EdgeMap`을 따라 뒤로 이동
- **종료 조건**: `isRailBranch == true` (분기점 도달)

```java
if (fromNode.isRailBranch()) {
    return item;  // 분기점 → 추적 종료
} else {
    // 이전 RailEdge의 address, portIdList를 수집하며 재귀 추적
}
```

### 5. 포트 정리

- **파일**: `navi/Navigator.java:81-88`
- 수집된 포트 목록에서 중복 제거 후 `DataSet.summarizePorts()`로 축약

### 6. 결과 저장

- **파일**: `data/VhlOffRecordItem.java:111-125`
- `affectedAddress`(Set)를 쉼표로 연결한 문자열로 변환하여 DB에 저장

```java
return affectedAddress.stream()
        .filter(address -> !address.isEmpty())
        .collect(Collectors.joining(","));
```

- **DB 컬럼 매핑** (`util/LayoutUtil.java:285-286`):
  - `FALR_RAISE_ADDR_LVAL` ← 영향받는 레일 주소 목록
  - `FALR_AFFECT_PORT_LVAL` ← 영향받는 포트 목록

---

## 데이터 구조

| 구조 | 설명 |
|------|------|
| `RailEdge` | 레일 구간. `fromNodeId`, `toNodeId`, `address`, `portIdList` 보유 |
| `RailNode` | 레일 노드. `isRailBranch`(분기), `isRailJunction`(합류) 속성 보유 |
| `NodeMap` | 노드ID → RailNode 매핑 |
| `FromNode2EdgeMap` | fromNodeId → 나가는 Edge 목록 (정방향 탐색용) |
| `ToNode2EdgeMap` | toNodeId → 들어오는 Edge 목록 (역방향 탐색용) |
| `VhlOffRecordItem` | VHL OFF 알람 데이터 모델. `affectedAddress`, `affectedPort` 보유 |

---

## 무한루프 방지

- 정방향/역방향 추적 모두 `overTrackCount > 1000` 시 강제 종료 (`Navigator.java:120, 168`)

---

## 단계별 요약

| 단계 | 내용 | 파일:라인 |
|------|------|-----------|
| 감지 | 에러코드가 VHL_OFF 목록에 매칭 | `OhtMsgWorkerRunnable.java:728` |
| 계산 | 멈춘 RailEdge 기준 앞뒤 그래프 탐색 | `Navigator.java:15-106` |
| 정방향 종료 | `isRailJunction == true` | `Navigator.java:174` |
| 역방향 종료 | `isRailBranch == true` | `Navigator.java:126` |
| 포트 축약 | 중복 제거 + summarizePorts | `Navigator.java:81-88` |
| 저장 | 쉼표 연결 문자열 → DB | `VhlOffRecordItem.java:111-117` |

---

## 요약

VHL이 멈춘 RailEdge에서 **앞으로는 합류점(Junction)까지, 뒤로는 분기점(Branch)까지** 레일 그래프를 재귀적으로 추적하여, 그 사이 모든 레일 주소를 `AFFECT_ADDR_LIST`에 저장한다.
