# OHT 속도 분석 프로젝트 지침

## 프로젝트 개요
이 프로젝트는 반도체 FAB의 OHT(Overhead Hoist Transport) 시스템의 속도 데이터를 분석하고 관리하는 Smart ATLAS 시스템에 관한 것입니다.

---

## 1. 시스템 아키텍처

### 1.1 전체 데이터 흐름
```
MCP → UDP Message → OhtUdpListener.java → OhtMsgWorkerRunnable.java (Queue 적재)
                                                    ↓
                                          DataSet.java (속도 계산)
                                                    ↓
                                          RailEdgeMap → RailEdge.java (속도 저장)
                                                    ↓
                              Batch 클래스들 (주기별 동작) → Logpresso API → 각 테이블
```

### 1.2 주요 Java 클래스 역할

| 클래스명 | 역할 |
|---------|------|
| `OhtUdpListener.java` | MCP로부터 UDP 메시지 수신 |
| `OhtMsgWorkerRunnable.java` | 수신한 UDP 메시지를 Queue에 적재 |
| `DataSet.java` | Queue 데이터를 사용해 속도(velocity) 값 계산 |
| `RailEdge.java` | RailEdge 객체 단위로 속도 값 저장 |
| `RailEdgeMap` | RailEdge 객체들의 맵 구조 |
| `DataService.java` | DataSet의 인스턴스 관리 |
| `BranchTrafficBatch.java` | areaTraffic/branchTraffic 데이터 구성 및 전송 |
| `TrafficBatch.java` | railTraffic 데이터 구성 및 전송 |
| `AreaVhlCountBatch.java` | areaVhlCountBatch 데이터 구성 및 전송 |
| `DijkstraVhlRouteFind.class` | 시작점~목적지 경로 탐색 (다익스트라 알고리즘) |

### 1.3 데이터 저장소 (Logpresso 테이블)
- `ATLAS_RAW_DATA`: 원시 데이터
- `ATLAS_DATA`: 가공된 기본 데이터
- `ATLAS_AREA_TRAFFIC`: Area 단위 트래픽 정보 (속도 포함)
- `ATLAS_BRANCH_TRAFFIC`: Branch 단위 트래픽 정보
- `ATLAS_VHL_COUNT`: Vehicle 수량 집계
- `ATLAS_RAIL_TRAFFIC`: Rail 단위 트래픽 정보

---

## 2. 속도(Velocity) 계산 로직

### 2.1 기본 계산 공식
```
속도(ν) = 거리(Δx) / 시간(Δt) × 60.0
```
- **시간(Δt)**: 두 메시지의 수신 시간 차이
- **거리(Δx)**: Vehicle 위치에 따라 다른 방식 적용

### 2.2 거리 계산 케이스

#### Case 1: α의 시작점과 β의 도착지점이 동일한 경우
```
거리(Δx) = (β에 위치한 RailEdge의 길이) - (β의 현재 번지에서의 거리) + (α의 현재 번지에서의 거리)
```

#### Case 2: α, β가 서로 다른 RailEdge에 위치한 경우
```
거리(Δx) = (β의 시작점부터 α의 시작점까지의 거리 및 RailEdge 길이의 합) - (β의 현재 번지에서의 거리) + (α의 현재 번지에서의 거리)
```

※ α: 최근 수신 메시지 정보 / β: 이전(마지막) 수신 메시지 정보

### 2.3 계산 예시
```
조건: Vehicle이 서로 다른 RailEdge에 위치 (Case 2 적용)
- RailEdge 길이: 100, 200 (가정)
- β의 현재 번지 거리: 20
- α의 현재 번지 거리: 0

거리(Δx) = (100 + 200) - 20 + 0 = 280
시간(Δt) = 1
속도(ν) = 280 / 1 = 280
```

---

## 3. 속도 계산 필수 조건 (5가지)

속도 계산이 수행되기 위해서는 아래 **모든 조건**을 충족해야 합니다:

| 번호 | 조건 | 상세 |
|-----|------|------|
| 1 | 메시지 수신 시간 차이 | 현재 vehicle과 최근 정보 비교 시 **1분 미만** |
| 2 | Vehicle 상태 | `RUN(운전중)`, `OBS_STOP`, `JAM(정체)`, `E84_TIMEOUT` 중 하나 이상 |
| 3 | Cycle 일치 | 현재 vehicle이 최근 정보와 `실행 cycle`, `vehicle 실행 cycle 주기`가 각각 동일 |
| 4 | 실행 Cycle 값 | `ACQUIRE(구원주기 동안)`, `DEPOSIT(도매 사이클 중)` 중 하나 이상 |
| 5 | Vehicle 실행 Cycle 주기 | `ACQUIRE_MOVING(구원 이송 중)`, `DEPOSIT_MOVING(도매 이동)` 중 하나 이상 |

---

## 4. 속도 초기값 설정

### 4.1 초기값 형성 시점
- 서버 최초 구동 시
- `dataSetRefresh`를 통한 맵 데이터 업데이트 시

### 4.2 설정 파일 구조 (*.mcp75.cfg)

#### POINT 속성 매핑
```
[MCP75_LAYOUT@SCH.1]
POINT = 1945, 0, 1946, 2017, 64, 0, 0, 0, 0, 0
```

| POINT 값 | railEdge 속성 | 설명 |
|----------|--------------|------|
| 1945 | address | 레일의 시작점 |
| 0 | stNo | station 번호 |
| 1946 | leftAddress | 레일의 좌측 끝점 |
| 2017 | leftDistance | 레일의 좌측 길이 |
| 64 | leftSpeed | 레일의 좌측 속도 (level 키값) |
| 0 | rightAddress | 레일의 우측 끝점 |
| 0 | rightDistance | 레일의 우측 길이 |
| 0 | rightSpeed | 레일의 우측 속도 |
| - | viaPoint | - |
| - | stopResetPoint | - |

#### 속도 초기값 결정 과정
1. `[MCP75_VEHICLE]`의 `VHL_SPEED_TYPE` 값 확인 (예: "CLW07-2")
2. `[MCP75_VEHICLE_SPEED]`의 해당 하위 항목(`<CLW07-2>`)과 매칭
3. POINT의 `leftSpeed`(또는 `rightSpeed`) 값을 level 키로 사용
4. 해당 level에 매칭되는 SPEED 값 취득

```
[MCP75_VEHICLE]
VHL_SPEED_TYPE = "CLW07-2";

[MCP75_VEHICLE_SPEED]
<CLW07-2>
SPEED = 1, 1.5;      // level=1, 속도=1.5
SPEED = 60, 320;     // level=60, 속도=320
SPEED = 61, 320;
SPEED = 62, 320;
SPEED = 63, 320;
SPEED = 64, 320;     // level=64, 속도=320
```

### 4.3 RailEdge ID 명명 규칙
```
M16A:RE:BR:M16A:RN:BR:01945-M16A:RN:BR:01946
 │    │  │   │    │   │      │
 │    │  │   │    │   │      └─ 끝점(to) 주소
 │    │  │   │    │   └─ 구분자
 │    │  │   │    └─ RN (Rail Node)
 │    │  │   └─ Factory ID
 │    │  └─ 층 정보 (BR=Bridge/3층/Hubroom, RN=일반)
 │    └─ RE (Rail Edge)
 └─ Factory ID (M16A, M14A, M11A 등)
```

---

## 5. UDP 메시지 구조

### 5.1 메시지 형식
```
2,OHT,V00399,1,1,0000,1,4176,20,4177,4,4,4PDMG440,4011,00000000,0000,4ANZ16-404,4CSC1602_2,90,0,0,N4PDMG4402025020601541,B16,
```

### 5.2 토큰별 필드 설명

| 순서 | 필드명 | 설명 | 예시값 |
|-----|--------|------|--------|
| 1 | message id | 메시지 식별자 | 2 |
| 2 | mcp 명칭 | MCP 이름 | OHT |
| 3 | vehicle 명칭 | Vehicle ID | V00399 |
| 4 | 상태 | Vehicle 상태 코드 | 1 |
| 5 | 재하 정보 | 적재 상태 | 1 |
| 6 | error code | 에러 코드 | 0000 |
| 7 | 통신 상태 | 통신 상태 코드 | 1 |
| 8 | 현재 번지 | 현재 위치 주소 | 4176 |
| 9 | 현재 번지에서의 거리 | 현재 번지 내 이동 거리 | 20 |
| 10 | 차 번지 | 다음 위치 주소 | 4177 |
| 11 | 실행 cycle | 실행 사이클 코드 | 4 |
| 12 | vehicle 실행 cycle 주기 | 사이클 주기 코드 | 4 |
| 13 | carrier id | Carrier 식별자 | 4PDMG440 |
| 14 | 목적지 | 목적지 주소 | 4011 |
| 15 | E/M 상태 | E/M 상태 코드 | 00000000 |
| 16 | group id | 그룹 식별자 | 0000 |

---

## 6. 데이터 테이블 구조

### 6.1 ATLAS_AREA_TRAFFIC (속도 값 사용)

| 컬럼명 | 설명 | 예시 |
|--------|------|------|
| createTime | 데이터 작성 및 저장 일자 (=BranchTrafficBatch.java 동작 시간) | 1723082384000 |
| density | area 상 RailEdge의 vehicle 밀도<br>계산식: (vehicle 길이) × (현재 vehicle 수량) / {(RailEdge 길이) - (RailEdge 길이) % (vehicle 길이)} | 0 |
| fabId | factory id | M11A |
| id | area id | M11A:AR:B32-2 |
| passJobCnts | 최근 1분, 2분, 3분, 4분, 5분 이내 area를 지나간 Job 수 (','로 구분) | 0,0,0,0,0 |
| pathPredictQCnts | 향후 1분, 2분, 3분, 4분, 5분 이내 area를 지나갈 것으로 예상되는 Job 수 | 0,0,1,0,0 |
| railTrafficInputCnt | area상의 vehicle이 DEPOSIT 혹은 assign command의 수 | 2 |
| railTrafficOutputCnt | area상의 vehicle이 ACQUIRE 혹은 assign command의 수 | 2 |
| **speed** | **현재 BranchJoinEdge의 속도(m/min, 지수평활법)** | **117.71171** |
| vhlCnt | 현재 RailEdge의 vehicle 수량 | 0 |

### 6.2 ATLAS_BRANCH_TRAFFIC (속도 값 사용)

| 컬럼명 | 설명 | 예시 |
|--------|------|------|
| createTime | 데이터 작성 및 저장 일자 | 1723082384000 |
| density | RailEdge의 vehicle 밀도<br>계산식: (vehicle 길이) × (현재 vehicle 수량) / {(RailEdge 길이) - (RailEdge 길이) % (vehicle 길이)} | 0 |
| edgeIds | BranchJoinEdge에 해당하는 RailEdge id 목록 | ["M11A:RE:A:M11A:RN:A:02258-M11A:RN:A:30081",...] |
| elapsedTime | BranchJoinEdge를 통과하는 데 소요된 시간 | 4026 |
| fabId | factory id | M11A |
| id | BranchJoinEdge id | M11A:BJE:M11A:RN:A:02258-0-M11A:RN:A:20513 |
| isAvailable | BranchJoinEdge의 사용 가능 여부 | true |
| jobCost | Job을 통해 산출해낸 vehicle의 통과 및 소요 시간 | 637.4399648145434 |
| length | BranchJoinEdge의 길이(mm) | 3020 |
| mcpName | mcp 명칭 | A |
| passCnt | 해당 BranchJoinEdge에서 최근 10초 간 vehicle 통과 수량 | 0 |
| passJobCnts | 최근 1분, 2분, 3분, 4분, 5분 이내 BranchJoinEdge를 지나간 Job 수 | 0,0,0,0,0 |
| pathPredictQCnts | 향후 1분, 2분, 3분, 4분, 5분 이내 지나갈 것으로 예상되는 Job 수 | 0,0,0,0,0 |
| pathPredictQueueSizeAvg | pathPredictQCnts의 평균값 | 0 |
| railTrafficInOutCnt | railTrafficOutputCnt + railTrafficInputCnt의 합 | 1 |

### 6.3 ATLAS_RAIL_TRAFFIC

| 컬럼명 | 설명 | 예시 |
|--------|------|------|
| absoluteVelocity | (RailEdge의 속도) / (RailEdge의 최대 속도) | 1 |
| createTime | 데이터 작성 및 저장 일자 | 1723082383001 |
| fabId | factory id | M11A |
| intOutCnt | inputCnt + outputCnt의 합 | 0 |
| inputCnt | Job을 통해 RailEdge 상의 vehicle이 DEPOSIT 혹은 assign command의 수 | 0 |
| isAvailable | RailEdge의 사용 가능 여부 | true |
| jobCost | Job을 통해 산출해낸 vehicle의 통과 및 소요 시간 | 637.4399648145434 |
| jobPassCnt | Job을 통해 산출해낸 vehicle의 통과 수량 | 2 |
| **maxVelocity** | **RailEdge의 최대 속도** | **200** |
| outputCnt | Job을 통해 RailEdge 상의 vehicle이 ACQUIRE 혹은 assign command의 수 | 0 |
| passCnt | RailEdge를 통과한 vehicle 수량 | 1 |
| predict1_2minCnt | RailEdge의 시작점까지 예상되는 도착 시간이 1분 이상, 2분 미만인 vehicle 수량 | 1 |
| predict2_3minCnt | 도착 시간이 2분 이상, 3분 미만인 vehicle 수량 | 0 |
| predict3_4minCnt | 도착 시간이 3분 이상, 4분 미만인 vehicle 수량 | 0 |
| predict4_5minCnt | 도착 시간이 4분 이상, 5분 미만인 vehicle 수량 | 0 |
| predictOver5minCnt | 도착 시간이 5분 이상인 vehicle 수량 | 0 |
| predictQueueAllCnt | 도착이 예상되는 vehicle 수량 (=predict* 값의 전체 합) | 5 |
| predictUnder1minCnt | 도착 시간이 1분 미만인 vehicle 수량 | 3 |
| railEdgeId | RailEdge id | M11B:RE:B:M11B:RN:B:05235-M11B:RN:B:05234 |
| vhlAbnormalCnt | RailEdge 상 ABNORMAL 상태인 vehicle의 수량 | 0 |
| vhlAcquireMovingCnt | RailEdge 상 ACQUIRE 상태인 vehicle의 수량 | 0 |

### 6.4 ATLAS_VHL_COUNT

| 컬럼명 | 설명 | 예시 |
|--------|------|------|
| {fab Id}:AR:{area 명칭} | 각 area 별로 vehicle 수량을 집계 | 5 |
| createTime | 데이터 작성 및 저장 일자 | 1723082386001 |

---

## 7. Factory별 Vehicle 길이

| Factory | Vehicle 길이 (mm) |
|---------|------------------|
| M14* | 784 + 300 = **1084** |
| M16* | 943 + 300 = **1243** |
| 기타 (ETC) | 784 + 300 = **1084** |

```java
if(area.getFabId().startsWith("M14")) {
    vhlLength = 784 + 300;
}
else if(area.getFabId().startsWith("M16")) {
    vhlLength = 943 + 300;
}
else {
    vhlLength = 784 + 300;
}
```

---

## 8. 속도 계산 플로우차트

```
[UDP Message 수신]
        ↓
[Message를 통해 vehicle, railEdge 정보 취득]
        ↓
[이전(최근) vehicle 정보 조회] ← lastUdpState
        ↓
[이전(최근) vehicle의 railEdge 정보 호출] ← lastRailEdge
        ↓
[현재 railEdge와 이전 railEdge 정보 비교]
        ↓
    ┌───────────────────────────────────────┐
    │  railEdge의 종점과 lastRailEdge의     │
    │  시작점이 동일한가?                    │
    └───────────────────────────────────────┘
           │                    │
          Yes                   No
           │                    │
           ↓                    ↓
    [Case 1 계산]        [DijkstraVhlRouteFind로
                          경로 유추 후 Case 2 계산]
           │                    │
           └────────┬───────────┘
                    ↓
    [각 구간(railEdge)별로 속도 값 저장]
```

---

## 9. 용어 정리

| 용어 | 설명 |
|------|------|
| OHT | Overhead Hoist Transport - 반도체 FAB 천장 이송 시스템 |
| MCP | Material Control Point - 자재 제어 포인트 |
| RailEdge | 레일의 한 구간 (시작점 → 끝점) |
| BranchJoinEdge | 여러 RailEdge를 묶은 분기 구간 |
| Vehicle | OHT 운반체 (V00001, V00399 등) |
| Carrier | Vehicle이 운반하는 FOUP 등의 용기 |
| ACQUIRE | 물품을 집어 올리는 동작/사이클 |
| DEPOSIT | 물품을 내려놓는 동작/사이클 |
| FAB | Fabrication facility - 반도체 생산 시설 |
| Logpresso | 로그 수집 및 분석 플랫폼 |
| FOUP | Front Opening Unified Pod - 웨이퍼 운반 용기 |

---

## 10. 주의사항 및 트러블슈팅

### 10.1 속도 값이 초기값으로 유지되는 경우
- 해당 레일에 대한 UDP 메시지를 받지 못한 경우
- 5가지 필수 조건을 충족하지 못한 경우

### 10.2 속도 계산 관련 주요 체크포인트
1. Vehicle 상태 코드가 유효한지 확인 (RUN, OBS_STOP, JAM, E84_TIMEOUT)
2. 메시지 수신 시간 간격이 1분 이내인지 확인
3. 실행 cycle 값이 ACQUIRE 또는 DEPOSIT인지 확인
4. RailEdge 경로가 올바르게 연결되어 있는지 확인

### 10.3 데이터 갱신 주기
- 각 Batch 클래스가 주기별로 동작하며 Logpresso 데이터 형성 후 송신
- `dataSetRefresh` 실행 시 맵 데이터 및 초기 속도값 업데이트

---

*문서 버전: 2025.08.22*
*작성 기준: OHT_속도_분석_취합본_250822*