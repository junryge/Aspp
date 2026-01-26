# 🏭 Smart ATLAS OHT 시스템 - 3개 문서 종합 이해

## 📄 종합 대상 문서
- OHT_속도분석.MD (2025.08 버전)
- MCS_LOG_아키텍처.MD (2022.07 버전)
- OHT_VHL_분석.MD (2025.01 버전)

**🎯 시스템 목적:** 반도체 FAB의 OHT(Overhead Hoist Transport) 천장 이송 시스템의 속도 데이터를 실시간 수집, 계산, 분석하여 물류 최적화 지원

---

## 📑 목차
1. [시스템 전체 구조 이해](#1-시스템-전체-구조-이해)
2. [데이터 수집 및 통신 방식](#2-데이터-수집-및-통신-방식)
3. [메시지 처리 Worker 구조](#3-메시지-처리-worker-구조)
4. [속도(Velocity) 계산 로직](#4-속도velocity-계산-로직)
5. [데이터 저장소 (Logpresso 테이블)](#5-데이터-저장소-logpresso-테이블)
6. [Batch Job 처리](#6-batch-job-처리)
7. [경로 탐색 알고리즘](#7-경로-탐색-알고리즘)
8. [주요 클래스 및 참조 관계](#8-주요-클래스-및-참조-관계)
9. [설정 파일 구조](#9-설정-파일-구조)
10. [용어 정리](#10-용어-정리)
11. [UDP 프로토콜 정의서 (ID:2 Vehicle 상태 보고)](#11-udp-프로토콜-정의서-id2-vehicle-상태-보고)
12. [수식 종합 및 계산 예시](#12-수식-종합-및-계산-예시)

---

## 1. 시스템 전체 구조 이해

### 1.1 전체 데이터 흐름도

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                        Smart ATLAS 시스템 전체 데이터 흐름                         │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                        │
│  │  M15A MCS   │     │  M15B MCS   │     │  smartSTAR  │                        │
│  │  APP1/APP2  │     │  APP1/APP2  │     │  APP1/APP2  │    ◄── 데이터 소스      │
│  │   DB1/DB2   │     │   DB1/DB2   │     │   DB1/DB2   │                        │
│  └──────┬──────┘     └──────┬──────┘     └──────┬──────┘                        │
│         │                   │                   │                                │
│         └───────────────────┼───────────────────┘                                │
│                             │                                                    │
│                    ┌────────▼────────┐                                          │
│                    │   smartATLAS    │                                          │
│                    │    (MCSLOG)     │    ◄── 중앙 처리 시스템                   │
│                    └────────┬────────┘                                          │
│                             │                                                    │
│         ┌───────────────────┼───────────────────┐                               │
│         ▼                   ▼                   ▼                               │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                        │
│  │ M15A OHT    │     │ M15B OHT    │     │ smartATLAS  │                        │
│  │  CLWMCP     │     │  CLWMCP     │     │     UI      │    ◄── 출력/제어       │
│  └─────────────┘     └─────────────┘     └─────────────┘                        │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           상세 처리 흐름                                          │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  [1단계: 데이터 수신]                                                             │
│  ┌─────────────┐        ┌─────────────┐        ┌─────────────┐                  │
│  │   OHT MCP   │        │    MCS      │        │  smartSTAR  │                  │
│  │    (UDP)    │        │  (TIB/rv)   │        │    (DB)     │                  │
│  └──────┬──────┘        └──────┬──────┘        └──────┬──────┘                  │
│         │                      │                      │                          │
│         ▼                      ▼                      ▼                          │
│  OhtUdpListener         TIB/rv Listener        DB Connection                    │
│                                                                                  │
│  [2단계: 메시지 분배]                                                             │
│         └──────────────────────┼──────────────────────┘                          │
│                                ▼                                                 │
│                    ┌───────────────────────┐                                    │
│                    │  MessageDispatcher    │                                    │
│                    └───────────┬───────────┘                                    │
│                                │                                                 │
│  [3단계: Worker 처리]                                                             │
│    ┌───────────┬───────────┬───────────┬───────────┬───────────┐               │
│    ▼           ▼           ▼           ▼           ▼           │               │
│ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐        │               │
│ │OhtMsg  │ │EiMsg   │ │TsMsg   │ │CnvMsg  │ │UiMsg   │        │               │
│ │Worker  │ │Worker  │ │Worker  │ │Worker  │ │Worker  │        │               │
│ └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘        │               │
│     │          │          │          │          │              │               │
│     │    processOhtReport()                                    │               │
│     ▼                                                          │               │
│  [4단계: 데이터 처리 및 계산]                                                     │
│  ┌─────────────────────────────────────────────────────────┐  │               │
│  │                  DataService.java                        │  │               │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │  │               │
│  │  │ DataSet.java│  │RailEdgeMap  │  │  VhlMap     │      │  │               │
│  │  │ (속도계산)  │  │             │  │             │      │  │               │
│  │  └─────────────┘  └─────────────┘  └─────────────┘      │  │               │
│  └─────────────────────────────────────────────────────────┘  │               │
│                                │                               │               │
│  [5단계: Batch 처리 및 저장]                                                     │
│                                ▼                                                │
│  ┌────────────────┐ ┌────────────────┐ ┌────────────────┐                      │
│  │ TrafficBatch   │ │BranchTraffic   │ │AreaVhlCount   │                      │
│  │   (3/10초)     │ │ Batch(4/10초)  │ │ Batch(6/10초) │                      │
│  └───────┬────────┘ └───────┬────────┘ └───────┬────────┘                      │
│          │                  │                  │                                │
│          └──────────────────┼──────────────────┘                                │
│                             ▼                                                   │
│  [6단계: 데이터 저장]                                                            │
│  ┌──────────────────────────────────────────────────────────────────┐          │
│  │                      Logpresso API / UDP                          │          │
│  ├──────────────────────────────────────────────────────────────────┤          │
│  │ ATLAS_RAW_DATA │ ATLAS_DATA │ ATLAS_ROUTE │ ATLAS_VEHICLE        │          │
│  │ ATLAS_RAIL_TRAFFIC │ ATLAS_BRANCH_TRAFFIC │ ATLAS_AREA_TRAFFIC   │          │
│  │ ATLAS_VHL_COUNT                                                   │          │
│  └──────────────────────────────────────────────────────────────────┘          │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 데이터 제공 소스별 역할

| 소스 | 제공 데이터 | 프로토콜 | 상세 설명 |
|------|------------|----------|----------|
| **MCS** (Material Control System) | LOG(FTP/TIB), 반송 이벤트 Msg(TIB), 장비/재고상태(DB) | TIB/rv, FTP | 자재 제어 시스템에서 반송 작업 관련 이벤트 및 로그 제공 |
| **OHT** (Overhead Hoist Transport) | OHT 반송 상태, Layout Data | UDP, FTP | 천장 이송 장비의 실시간 위치, 상태, 속도 정보 |
| **smartSTAR** | 반송 이력 정보, smartFX DB | DB | 과거 반송 이력 및 통계 데이터 |

---

## 2. 데이터 수집 및 통신 방식

### 2.1 통신 방식 결정 로직

> **통신 방식은 MCP Port 개수에 따라 자동 결정됩니다:**
> - **MCP Port 0개 이하:** TIB/rv 방식 → OhtTibrvListener 사용
> - **MCP Port 1개 이상:** UDP 방식 → OhtUdpListener 사용

### 2.2 OHT Connect Listener 초기화 과정

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     OHT Connect Listener 초기화 과정                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────┐                                                   │
│  │  DataService.java   │                                                   │
│  └──────────┬──────────┘                                                   │
│             │                                                               │
│             │ .getFabPropertiesMap()                                        │
│             ▼                                                               │
│  ┌─────────────────────────────────────────┐                               │
│  │  fab properties mapper 데이터 호출 (→ fp) │                               │
│  └──────────┬──────────────────────────────┘                               │
│             │                                                               │
│             │ .onStarted()                                                  │
│             ▼                                                               │
│  ┌─────────────────────────────────────────┐                               │
│  │     LauncherListener.java               │                               │
│  └──────────┬──────────────────────────────┘                               │
│             │                                                               │
│             │ fp.getMcpPropertiesMap()의 키 값을 통해 MCP 정보 취득           │
│             ▼                                                               │
│  ┌─────────────────────────────────────────┐                               │
│  │       MCP의 Port 갯수 확인               │                               │
│  └──────────┬──────────────────────────────┘                               │
│             │                                                               │
│      ┌──────┴──────┐                                                       │
│      │             │                                                       │
│   0개 이하      1개 이상                                                     │
│      │             │                                                       │
│      ▼             ▼                                                       │
│  ┌────────┐   ┌────────────┐                                               │
│  │TIB/RV  │   │    UDP     │                                               │
│  │ 방식   │   │    방식    │                                               │
│  └───┬────┘   └─────┬──────┘                                               │
│      │              │                                                       │
│      ▼              ▼                                                       │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ TIB/RV 방식:                    │ UDP 방식:                         │    │
│  │ 1. OhtTibrvListener 인스턴스 생성│ 1. OhtUdpListener 인스턴스 생성    │    │
│  │ 2. fabOhtListenerMap에 저장     │ 2. fabOhtUdpListenerMap에 저장    │    │
│  │ 3. Tib/rv init → start          │ 3. DatagramSocket으로 UDP 연결    │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 UDP 메시지 구조

**메시지 예시:**
```
2,OHT,V00399,1,1,0000,1,4176,20,4177,4,4,4PDMG440,4011,00000000,0000,4ANZ16-404,4CSC1602_2,90,0,0,N4PDMG4402025020601541,B16,
```

| 순서 | 필드명 | 설명 | 예시값 |
|-----|--------|------|--------|
| 1 | message id | 메시지 식별자 | 2 |
| 2 | mcp 명칭 | MCP 이름 | OHT |
| 3 | vehicle 명칭 | Vehicle ID | V00399 |
| 4 | 상태 | Vehicle 상태 코드 | 1 |
| 5 | 재하 정보 | 적재 상태 (0=비적재, 1=적재) | 1 |
| 6 | error code | 에러 코드 | 0000 |
| 7 | 통신 상태 | 통신 상태 코드 | 1 |
| 8 | 현재 번지 | 현재 위치 주소 | 4176 |
| 9 | 현재 번지에서의 거리 | 현재 번지 내 이동 거리 (mm) | 20 |
| 10 | 차 번지 | 다음 위치 주소 | 4177 |
| 11 | 실행 cycle | 실행 사이클 코드 | 4 |
| 12 | vehicle 실행 cycle 주기 | 사이클 주기 코드 | 4 |
| 13 | carrier id | Carrier 식별자 | 4PDMG440 |
| 14 | 목적지 | 목적지 주소 | 4011 |
| 15 | E/M 상태 | E/M 상태 코드 | 00000000 |
| 16 | group id | 그룹 식별자 | 0000 |

---

## 3. 메시지 처리 Worker 구조

### 3.1 Worker별 역할

| Worker | 역할 | 주요 처리 내용 |
|--------|------|---------------|
| **OhtMsgWorker** | OHT 메시지 처리 | VHL 상태 추적, RailEdge/TransferEdge 속도 Update, Carrier 위치 추적 |
| **EiMsgWorker** | Command 처리 | Command 생성/완료 처리, Carrier 위치, RmEdge Cost Update |
| **TsMsgWorker** | Job 처리 | Job 생성/완료 처리 |
| **CnvMsgWorker** | Conveyor 처리 | CnvTask 추적, Conveyor 상태 Update |
| **UiMsgWorker** | UI 상태 처리 | 포트/장비 상태 Update, Job wakeup 처리 |

### 3.2 processOhtReport() 핵심 처리 흐름

> **processOhtReport()는 OhtMsgWorkerRunnable의 핵심 메서드로, UDP 메시지를 받아 Vehicle 상태 업데이트 및 속도 계산을 수행합니다.**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              processOhtReport(token) 처리 흐름                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  processOhtReport(token)                                                    │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────┐                                           │
│  │ DataService.getVhlMap()에서 │                                           │
│  │ Vehicle 객체(v) 참조        │                                           │
│  └──────────────┬──────────────┘                                           │
│                 │                                                           │
│                 ▼                                                           │
│  ┌──────────────────────────────────┐                                      │
│  │  v(vehicle)의 VHL_STATE가 REMOVING? │                                      │
│  └──────────────┬───────────────────┘                                      │
│                 │                                                           │
│       ┌─────────┴─────────┐                                                │
│       │                   │                                                │
│      Yes                  No                                               │
│       │                   │                                                │
│       ▼                   ▼                                                │
│  ┌────────────────┐  ┌────────────────────────────────────────────────┐   │
│  │ REMOVING 처리  │  │              일반 처리 로직 (6단계)              │   │
│  ├────────────────┤  ├────────────────────────────────────────────────┤   │
│  │ 1. udpState    │  │ [1단계] UDP 통신에 따른 vehicle 정보 업데이트    │   │
│  │    초기화      │  │ [2단계] Rail Edge 정보 업데이트                  │   │
│  │ 2. rail에서    │  │ [3단계] 교차점 ID 비교 및 Logpresso 저장         │   │
│  │    vehicle 삭제│  │ [4단계] Command 및 Job 정보 처리                 │   │
│  │ 3. command삭제 │  │ [5단계] Route 정보 처리 및 상태 기록             │   │
│  │ 4. return      │  │ [6단계] 최종 계산 및 저장                        │   │
│  └────────────────┘  └────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 VhlUdpState 필드 정의

| 필드명 | REMOVING 상태 값 | 일반 상태 값 | 설명 |
|--------|-----------------|-------------|------|
| railNodeId | "" (빈값) | token에서 추출 | 현재 Rail Node ID |
| udpCarrierId | "" | token | UDP Carrier ID |
| destStationId | "" | token | 목적지 Station ID |
| distance | 0 | token | 이동 거리 |
| isFull | False | token | 적재 상태 |
| runCycle | RUN_CYCLE.NONE | token | 실행 주기 |
| vhlCycle | VHL_CYCLE.NONE | token | Vehicle 주기 |
| state | VHL_STATE.REMOVING | token | Vehicle 상태 |
| receivedTime | receivedMilli | receivedMilli | 수신 시간 |
| detailState | VHL_DET_STATE.NONE | token | 상세 상태 |

---

## 4. 속도(Velocity) 계산 로직

### 4.1 기본 계산 공식

```
속도(ν) = 거리(Δx) / 시간(Δt) × 60.0
```

**단위: m/min (미터/분)**

- **시간(Δt):** 두 메시지의 수신 시간 차이 (초)
- **거리(Δx):** Vehicle 위치에 따라 다른 방식 적용
- **×60.0:** 초 단위를 분 단위로 변환

### 4.2 거리 계산 케이스

> **※ α: 최근(현재) 수신 메시지 정보 / β: 이전(마지막) 수신 메시지 정보**

#### Case 1: α의 시작점과 β의 도착지점이 동일한 경우 (연속된 RailEdge)

```
거리(Δx) = (β에 위치한 RailEdge의 길이) - (β의 현재 번지에서의 거리) + (α의 현재 번지에서의 거리)
```

#### Case 2: α, β가 서로 다른 RailEdge에 위치한 경우

```
거리(Δx) = (β의 시작점부터 α의 시작점까지의 거리 및 RailEdge 길이의 합) - (β의 현재 번지에서의 거리) + (α의 현재 번지에서의 거리)
```

> **Case 2의 경우 DijkstraVhlRouteFind 알고리즘으로 경로를 유추하여 거리 계산**

### 4.3 계산 예시

| 조건 | 값 |
|------|-----|
| 적용 케이스 | Case 2 (서로 다른 RailEdge) |
| RailEdge 길이 | 100mm, 200mm (가정) |
| β의 현재 번지 거리 | 20mm |
| α의 현재 번지 거리 | 0mm |
| 시간(Δt) | 1초 |

**계산:**
```
거리(Δx) = (100 + 200) - 20 + 0 = 280mm
속도(ν) = 280 / 1 × 60.0 = 16,800 mm/min = 16.8 m/min
```

### 4.4 속도 계산 필수 조건 (5가지 - 모두 충족 필요)

> ⚠️ **아래 5가지 조건을 모두 충족해야만 속도 계산이 수행됩니다!**

| 번호 | 조건 | 상세 | 체크 필드 |
|-----|------|------|----------|
| **1** | 메시지 수신 시간 차이 | 현재 vehicle과 최근 정보 비교 시 **1분 미만** | receivedTime |
| **2** | Vehicle 상태 | `RUN(운전중)`, `OBS_STOP`, `JAM(정체)`, `E84_TIMEOUT` 중 하나 | state (VHL_STATE) |
| **3** | Cycle 일치 | 현재 vehicle이 최근 정보와 `실행 cycle`, `vehicle 실행 cycle 주기`가 각각 동일 | runCycle, vhlCycle |
| **4** | 실행 Cycle 값 | `ACQUIRE(물품 집어올리기)`, `DEPOSIT(물품 내려놓기)` 중 하나 | runCycle (RUN_CYCLE) |
| **5** | Vehicle 실행 Cycle 주기 | `ACQUIRE_MOVING(구원 이송 중)`, `DEPOSIT_MOVING(도매 이동)` 중 하나 | vhlCycle (VHL_CYCLE) |

### 4.5 속도 계산 플로우차트

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           속도 계산 플로우차트                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [UDP Message 수신]                                                         │
│         │                                                                   │
│         ▼                                                                   │
│  [Message를 통해 vehicle, railEdge 정보 취득]                               │
│         │                                                                   │
│         ▼                                                                   │
│  [5가지 필수 조건 확인] ──────────────────────────────┐                      │
│         │                                           │                      │
│         │ 모두 충족                              미충족│                      │
│         ▼                                           ▼                      │
│  [이전(최근) vehicle 정보 조회] ← lastUdpState   [속도 계산 SKIP]            │
│         │                                                                   │
│         ▼                                                                   │
│  [이전(최근) vehicle의 railEdge 정보 호출] ← lastRailEdge                   │
│         │                                                                   │
│         ▼                                                                   │
│  [현재 railEdge와 이전 railEdge 정보 비교]                                  │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────┐                   │
│  │  railEdge의 종점과 lastRailEdge의 시작점이 동일한가? │                   │
│  └─────────────────────────┬───────────────────────────┘                   │
│                   ┌────────┴────────┐                                      │
│                  Yes                No                                      │
│                   │                  │                                      │
│                   ▼                  ▼                                      │
│           [Case 1 계산]    [DijkstraVhlRouteFind로                          │
│                             경로 유추 후 Case 2 계산]                       │
│                   │                  │                                      │
│                   └────────┬─────────┘                                      │
│                            │                                                │
│                            ▼                                                │
│           [지수평활법 적용하여 속도 값 평활화]                               │
│                            │                                                │
│                            ▼                                                │
│           [각 구간(railEdge)별로 속도 값 저장]                              │
│                            │                                                │
│                            ▼                                                │
│           [RailEdge.java에 velocity 값 업데이트]                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.6 속도 초기값 설정

#### 초기값 형성 시점
- 서버 최초 구동 시
- `dataSetRefresh`를 통한 맵 데이터 업데이트 시 (매일 10시)

#### 설정 파일 구조 (*.mcp75.cfg)

```
[MCP75_LAYOUT@SCH.1]
POINT = 1945, 0, 1946, 2017, 64, 0, 0, 0, 0, 0
```

| POINT 순서 | railEdge 속성 | 설명 | 예시값 |
|-----------|--------------|------|--------|
| 1 | address | 레일의 시작점 | 1945 |
| 2 | stNo | station 번호 | 0 |
| 3 | leftAddress | 레일의 좌측 끝점 | 1946 |
| 4 | leftDistance | 레일의 좌측 길이 | 2017 |
| 5 | leftSpeed | 레일의 좌측 속도 (level 키값) | 64 |
| 6 | rightAddress | 레일의 우측 끝점 | 0 |
| 7 | rightDistance | 레일의 우측 길이 | 0 |
| 8 | rightSpeed | 레일의 우측 속도 | 0 |

#### 속도 초기값 결정 과정

1. `[MCP75_VEHICLE]`의 `VHL_SPEED_TYPE` 값 확인 (예: "CLW07-2")
2. `[MCP75_VEHICLE_SPEED]`의 해당 하위 항목과 매칭
3. POINT의 `leftSpeed`(또는 `rightSpeed`) 값을 level 키로 사용
4. 해당 level에 매칭되는 SPEED 값 취득

```
[MCP75_VEHICLE]
VHL_SPEED_TYPE = "CLW07-2";

[MCP75_VEHICLE_SPEED]
<CLW07-2>
SPEED = 1, 1.5;      // level=1, 속도=1.5
SPEED = 60, 320;     // level=60, 속도=320
SPEED = 64, 320;     // level=64, 속도=320
```

### 4.7 Factory별 Vehicle 길이

| Factory | Vehicle 본체 (mm) | 추가 길이 (mm) | 총 길이 (mm) |
|---------|------------------|----------------|--------------|
| M14* | 784 | 300 | **1084** |
| M16* | 943 | 300 | **1243** |
| 기타 (ETC) | 784 | 300 | **1084** |

> 이 Vehicle 길이는 **density(밀도) 계산**에 사용됩니다.

---

## 5. 데이터 저장소 (Logpresso 테이블)

### 5.1 테이블 목록 및 용도

| 테이블명 | 용도 | 적재 주체 |
|---------|------|----------|
| ATLAS_RAW_DATA | 원시 데이터 | - |
| ATLAS_DATA | 가공된 기본 데이터 | - |
| ATLAS_ROUTE | 경로 정보 | RouteItem.sendToLogpresso() |
| ATLAS_VEHICLE | Vehicle 정보 | saveVehicleToLogpresso() |
| ATLAS_AREA_TRAFFIC | Area 단위 트래픽 정보 (속도 포함) | BranchTrafficBatch |
| ATLAS_BRANCH_TRAFFIC | Branch 단위 트래픽 정보 | BranchTrafficBatch |
| ATLAS_RAIL_TRAFFIC | Rail 단위 트래픽 정보 | TrafficBatch |
| ATLAS_VHL_COUNT | Vehicle 수량 집계 | AreaVhlCountBatch |

### 5.2 ATLAS_AREA_TRAFFIC 컬럼 정의

| 컬럼명 | 설명 | 예시 |
|--------|------|------|
| createTime | 데이터 작성 및 저장 일자 (=BranchTrafficBatch.java 동작 시간) | 1723082384000 |
| density | area 상 RailEdge의 vehicle 밀도 (계산식: (vehicle 길이) × (현재 vehicle 수량) / {(RailEdge 길이) - (RailEdge 길이) % (vehicle 길이)}) | 0 |
| fabId | factory id | M11A |
| id | area id | M11A:AR:B32-2 |
| passJobCnts | 최근 1분, 2분, 3분, 4분, 5분 이내 area를 지나간 Job 수 (','로 구분) | 0,0,0,0,0 |
| pathPredictQCnts | 향후 1분, 2분, 3분, 4분, 5분 이내 area를 지나갈 것으로 예상되는 Job 수 | 0,0,1,0,0 |
| railTrafficInputCnt | area상의 vehicle이 DEPOSIT 혹은 assign command의 수 | 2 |
| railTrafficOutputCnt | area상의 vehicle이 ACQUIRE 혹은 assign command의 수 | 2 |
| **speed** | **현재 BranchJoinEdge의 속도(m/min, 지수평활법)** | **117.71171** |
| vhlCnt | 현재 RailEdge의 vehicle 수량 | 0 |

### 5.3 ATLAS_BRANCH_TRAFFIC 컬럼 정의

| 컬럼명 | 설명 | 예시 |
|--------|------|------|
| createTime | 데이터 작성 및 저장 일자 | 1723082384000 |
| density | RailEdge의 vehicle 밀도 | 0 |
| edgeIds | BranchJoinEdge에 해당하는 RailEdge id 목록 | ["M11A:RE:A:M11A:RN:A:02258-M11A:RN:A:30081",...] |
| elapsedTime | BranchJoinEdge를 통과하는 데 소요된 시간 | 4026 |
| fabId | factory id | M11A |
| id | BranchJoinEdge id | M11A:BJE:M11A:RN:A:02258-0-M11A:RN:A:20513 |
| isAvailable | BranchJoinEdge의 사용 가능 여부 (LaneCut) | true |
| jobCost | Job을 통해 산출해낸 vehicle의 통과 및 소요 시간 | 637.4399648145434 |
| length | BranchJoinEdge의 길이(mm) | 3020 |
| mcpName | mcp 명칭 | A |
| passCnt | 해당 BranchJoinEdge에서 최근 10초 간 vehicle 통과 수량 | 0 |
| passJobCnts | 최근 1분, 2분, 3분, 4분, 5분 이내 지나간 Job 수 | 0,0,0,0,0 |
| pathPredictQCnts | 향후 1분, 2분, 3분, 4분, 5분 이내 지나갈 것으로 예상되는 Job 수 | 0,0,0,0,0 |
| pathPredictQueueSizeAvg | pathPredictQCnts의 평균값 | 0 |
| railTrafficInOutCnt | railTrafficOutputCnt + railTrafficInputCnt의 합 | 1 |

### 5.4 ATLAS_RAIL_TRAFFIC 컬럼 정의

| 컬럼명 | 설명 | 예시 |
|--------|------|------|
| absoluteVelocity | (RailEdge의 속도) / (RailEdge의 최대 속도) | 1 |
| createTime | 데이터 작성 및 저장 일자 | 1723082383001 |
| fabId | factory id | M11A |
| isAvailable | RailEdge의 사용 가능 여부 | true |
| jobCost | Job을 통해 산출해낸 vehicle의 통과 및 소요 시간 | 637.4399648145434 |
| **maxVelocity** | **RailEdge의 최대 속도** | **200** |
| passCnt | RailEdge를 통과한 vehicle 수량 | 1 |
| predictUnder1minCnt | 도착 시간이 1분 미만인 vehicle 수량 | 3 |
| predict1_2minCnt | 도착 시간이 1분 이상, 2분 미만인 vehicle 수량 | 1 |
| predict2_3minCnt | 도착 시간이 2분 이상, 3분 미만인 vehicle 수량 | 0 |
| predict3_4minCnt | 도착 시간이 3분 이상, 4분 미만인 vehicle 수량 | 0 |
| predict4_5minCnt | 도착 시간이 4분 이상, 5분 미만인 vehicle 수량 | 0 |
| predictOver5minCnt | 도착 시간이 5분 이상인 vehicle 수량 | 0 |
| predictQueueAllCnt | 도착이 예상되는 vehicle 수량 (=predict* 값의 전체 합) | 5 |
| railEdgeId | RailEdge id | M11B:RE:B:M11B:RN:B:05235-M11B:RN:B:05234 |
| vhlAbnormalCnt | RailEdge 상 ABNORMAL 상태인 vehicle의 수량 | 0 |
| vhlAcquireMovingCnt | RailEdge 상 ACQUIRE 상태인 vehicle의 수량 | 0 |

### 5.5 Logpresso INSERT 메서드

| 메서드 | 정의 위치 | 테이블 | 데이터 내용 |
|--------|----------|--------|-------------|
| JsonUtil.insertJsonArrayDataToLogpresso() | JsonUtil.java | 'tblNm' 값으로 동적 결정 | OhtRegBjData, OhtRegData |
| _.sendToLogpresso() | RouteItem.java | ATLAS_ROUTE | Route Item 정보 |
| _.saveVehicleToLogpresso() | OhtMsgWorkerRunnable.java | ATLAS_VEHICLE | Vehicle 정보 |

---

## 6. Batch Job 처리

### 6.1 트래픽 관련 Batch

| Batch | 주기 | 내용 | 적재 테이블 |
|-------|------|------|------------|
| **TrafficBatch** | 3/10초 | RailEdge별 통계 | ATLAS_RAIL_TRAFFIC |
| **BranchTrafficBatch** | 4/10초 | BranchJoinEdge/Area 통계 | ATLAS_BRANCH_TRAFFIC, ATLAS_AREA_TRAFFIC |
| **AreaVhlCountBatch** | 6/10초 | Area별 VHL 수량 집계 | ATLAS_VHL_COUNT |
| **VhlBatch** | 1분 | 전체 VHL 상태 | Logpresso |

### 6.2 기타 주요 Batch

| Batch | 주기 | 내용 |
|-------|------|------|
| DataSetRefreshBatch | 매일 10시 | Layout/장비/Port 정보 갱신, 초기 속도값 업데이트 |
| LaneCutRefreshBatch | 5/10초 | FTP로 lanecut.dat 파일 갱신 |
| HelloBatch | 1분 | Old Job Cleaning, Garbage Route Cleaning |
| PredictionClean | 5/10초 | 만료된 pathPredictQueue 제거 |
| ServerStatusBatch | 2초 | CPU, Memory, Thread 상태 집계 |
| ObjOnMapDiffUpdateBatch | 1초 | OHT, Command 통계 적재 |
| ObjOnMapUpdateBatch | 30초 | Port, Eqp별 상태 적재 |
| BridgeEqpMonitorBatch | 1분 | Lifter/Conveyor 처리량, 부하상태, 소요시간 모니터링 |

---

## 7. 경로 탐색 알고리즘

### 7.1 변형 Dijkstra 알고리즘

> smartATLAS는 **변형된 Dijkstra 알고리즘**을 사용합니다.

**문제점:**
RailNode와 Station을 통한 포트 연결이 복잡하여 일반 Dijkstra로는 경로탐색 불가능한 경우 발생

**해결책:**
TransferEdge_Acquire를 통과 시, 동일 RailEdge상에서는 offset이 더 큰 쪽으로만 방문 가능하도록 제한. 이후 해당 노드를 미방문 처리하여 다시 방문 가능하도록 함

### 7.2 Edge Cost 계산

| 메서드 | 설명 | 용도 |
|--------|------|------|
| getCost(carrierId) | 지수평활법 평균 속도 기반 소요시간 | UI 속도 표시 |
| getPPCost() | PathPredictQueue 수량 × 60ms 추가 | N:1 소요시간 계산 |
| getVhlCountCost() | 현재 VHL 수량 기반 penalty | VHL Assign 예측 |
| getLast1HourCost() | 최근 1시간 평균 통과 소요시간 | FutureCost 대체 |

### 7.3 FutureCost 계산 공식

```
FutureCost = baseCost
           + (futureTransCnt × transWeight)
           + (futureAcqCnt × acqTransWeight)
           + (futureDpstCnt × dpstTransWeight)
           + Junction penalty (있는 경우)
```

### 7.4 ML/DL 보정

PathPredictQueue의 시간대별 분포가 균일하지 않아 (1~3분 이내 많음) 예측시간과 실제 소요시간 차이가 큼. **AtlasPredictEnhanceTibrvReq**를 통해 ML/DL 보정 수행.

---

## 8. 주요 클래스 및 참조 관계

### 8.1 전체 클래스 역할

| 클래스명 | 역할 | 비고 |
|---------|------|------|
| OhtUdpListener.java | MCP로부터 UDP 메시지 수신 | DatagramSocket 사용 |
| OhtTibrvListener.java | TIB/rv 메시지 수신 | MCP Port 0개 이하일 때 |
| OhtMsgWorkerRunnable.java | 수신한 UDP 메시지를 Queue에 적재, processOhtReport() 실행 | 핵심 처리 클래스 |
| DataSet.java | Queue 데이터를 사용해 속도(velocity) 값 계산 | 지수평활법 적용 |
| RailEdge.java | RailEdge 객체 단위로 속도 값 저장 | 개별 레일 구간 |
| RailEdgeMap | RailEdge 객체들의 맵 구조 | 전체 레일 구간 맵핑 |
| DataService.java | DataSet의 인스턴스 관리 | 싱글톤 패턴 |
| LauncherListener.java | 시스템 시작 리스너 | onStarted() 이벤트 처리 |
| DijkstraVhlRouteFind.class | 시작점~목적지 경로 탐색 | 다익스트라 알고리즘 |
| DfkRoutePlanUpdate.java | Rail 방향 및 노드 목록 업데이트 | defaultDir, railNodeList 관리 |
| BranchTrafficBatch.java | areaTraffic/branchTraffic 데이터 구성 및 전송 | 4/10초 주기 |
| TrafficBatch.java | railTraffic 데이터 구성 및 전송 | 3/10초 주기 |
| AreaVhlCountBatch.java | areaVhlCountBatch 데이터 구성 및 전송 | 6/10초 주기 |

### 8.2 클래스 참조 관계

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              processOhtReport() 클래스 참조 관계                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                      processOhtReport(token)                                │
│                              │                                              │
│              ┌───────────────┼───────────────┐                             │
│              │               │               │                             │
│              ▼               ▼               ▼                             │
│       ┌─────────────┐ ┌─────────────┐ ┌──────────────────┐                │
│       │RailEdge.java│ │  Vhl.java   │ │VhlUdpState.class │                │
│       └──────┬──────┘ └──────┬──────┘ └──────────────────┘                │
│              │               │                                             │
│              │               └─────────┐                                   │
│              │                         │                                   │
│              │ .getRailEdgeMap()       │ .getVhlMap()                      │
│              │ .getEdgeMap()           │                                   │
│              │                         │                                   │
│              └────────────┬────────────┘                                   │
│                           │                                                │
│                           ▼                                                │
│                  ┌─────────────────┐                                       │
│                  │ DataService.java │                                       │
│                  │   (싱글톤)       │                                       │
│                  └─────────────────┘                                       │
│                           │                                                │
│           ┌───────────────┼───────────────┐                               │
│           ▼               ▼               ▼                               │
│    .getVhlMap()    .getRailEdgeMap()  .getNodeMap()                        │
│    .getEdgeMap()   .getStationPortMap()  .getFabPropertiesMap()            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.3 RailEdge ID 명명 규칙

```
M16A:RE:BR:M16A:RN:BR:01945-M16A:RN:BR:01946
```

| 위치 | 값 | 설명 |
|-----|-----|------|
| 1 | M16A | Factory ID |
| 2 | RE | Rail Edge 표시 |
| 3 | BR | 층 정보 (BR=Bridge/3층/Hubroom, RN=일반) |
| 4 | M16A | Factory ID (반복) |
| 5 | RN | Rail Node 표시 |
| 6 | BR | 층 정보 |
| 7 | 01945 | 시작점(from) 주소 |
| 8 | - | 구분자 |
| 9 | M16A:RN:BR:01946 | 끝점(to) 정보 |

---

## 9. 설정 파일 구조

### 9.1 공통 설정

| 속성 | 설명 | 값 예시 |
|------|------|--------|
| Env | 환경 설정 (TIB/rv Subject 생성에 영향) | REAL / TEST / QA |
| FabIdList | MCS기준 FAB ID 목록 (콤마 구분) | M15A,M15B |
| ProcessType | 프로세스 타입 | DATAMAKER(수집서버) / QUERY(Logpresso Query 전용) |

### 9.2 수집서버 설정

| 속성 | 설명 |
|------|------|
| Atlas.Daemon | Atlas tib/rv 통신용 daemon 서버 |
| Atlas.Gid | cmessage 통신 그룹 ID |
| Atlas.ThreadPool.size | Worker Thread 수량 (CPU 코어 수) |
| Atlas.MonitoringMode | Y/N (CPU 부하 함수 Disable) |
| Logpresso.Host / Host2 | Logpresso 서버 주소 (이중화) |
| Logpresso.UdpHost / UdpHost2 | UDP 전송 Target 주소 |
| Logpresso.UdpPort | UDP 전송 Port |

### 9.3 FAB별 설정

| 속성 | 설명 |
|------|------|
| [FABID].FacId | Factory ID (M14A → M14) |
| [FABID].BridgeFrom | 다른 FAB에서 연결되는 Bridge 장비 목록 |
| [FABID].BridgeTo | 다른 FAB에 연결하는 Bridge 장비 목록 |
| [FABID].MapDir | OHT Map 폴더 경로 |
| [FABID].daemon | MCS 통신 메시지 Listen용 rvd daemon 주소 |
| [FABID].ei.subject | MCS EI 프로세스 메시지 Listen Subject |
| [FABID].oht.[MCP].port | CLWMCP UDP Port |
| [FABID].oht.[MCP].ftp.* | CLWMCP FTP 설정 (lanecut, mcp75, station, route, layout) |

---

## 10. 용어 정리

| 용어 | 영문 | 설명 |
|------|------|------|
| OHT | Overhead Hoist Transport | 반도체 FAB 천장 이송 시스템 |
| MCP | Material Control Point | 자재 제어 포인트 |
| MCS | Material Control System | 자재 제어 시스템 |
| MHS | Material Handling System | 자재 반송 시스템 |
| VHL | Vehicle | OHT 운반체 (V00001, V00399 등) |
| RailEdge | - | 레일의 한 구간 (시작점 → 끝점) |
| BranchJoinEdge | - | 여러 RailEdge를 묶은 분기/병합 구간 |
| LongEdge | - | Branch~Junction 사이 RailEdge 집합 |
| Carrier | - | Vehicle이 운반하는 FOUP 등의 용기 |
| FOUP | Front Opening Unified Pod | 웨이퍼 운반 용기 |
| ACQUIRE | - | 물품을 집어 올리는 동작/사이클 |
| DEPOSIT | - | 물품을 내려놓는 동작/사이클 |
| FAB | Fabrication facility | 반도체 생산 시설 |
| STK | Stocker | 저장 장비 (Shelf로 Carrier 저장) |
| STB | Side Track Buffer | 측면 버퍼 |
| FIO | Factory I/O | AMHS 입출고 장치 |
| RM | RackMaster/Crane | Stocker 내부 반송 장치 |
| CLWMCP | - | OHT 제어 장비 |
| TIB/rv | TIBCO Rendezvous | 메시지 통신 프로토콜 |
| Logpresso | - | 로그 수집 및 분석 플랫폼 |
| Junction | - | 합류점 |
| Branch | - | 분기점 |
| LaneCut | - | 레일 구간 차단 |
| PathPredictQueue | - | 경로 예측 대기열 |

---

## 11. UDP 프로토콜 정의서 (ID:2 Vehicle 상태 보고)

> **📡 OCS → OSS Vehicle 상태 보고 메시지 정의**
> 이 섹션은 OHT 시스템에서 사용되는 UDP 프로토콜 ID:2 메시지의 상세 필드 정의를 포함합니다.

### 11.1 기본 데이터 항목

| 데이터 항목 | 내용 |
|------------|------|
| **텍스트 ID** | 2 |
| **MCP 명칭** | 각 MCP7의 명칭(유니크) |
| **Vehicle 명** | Vehicle 식별하기 위한 명칭 |

### 11.2 상태 코드

| 코드 | 상태 |
|------|------|
| "1" | 운전 중 |
| "2" | 정지 중 |
| "3" | 이상 |
| "4" | 수동 |
| "5" | 수출 중 |
| "6" | OBS-STOP/BZ-STOP |
| "7" | 정체 |
| "9" | E84 Timeout |
| "10" | 주회 없음 E84 Timeout |
| "11" | HT-STOP |

### 11.3 재하 정보

| 코드 | 내용 |
|------|------|
| "0" | 재하 없음 |
| "1" | 재하 있음 |

### 11.4 Error Code

- **범위:** "0000" ~ "FFFF"
- **비고:** 상태가 "3" 이외인 경우 "0000"을 설정한다

### 11.5 통신 상태

| 코드 | 내용 |
|------|------|
| "1" | 정상 |
| "2" | 통신 끊김 |

### 11.6 위치 정보

| 항목 | 설명 |
|------|------|
| **현재 번지** | 현재 번지를 알 수 없는 경우 아무것도 설정되지 않음 |
| **현재번지로부터의 거리** | 현재 번지로부터의 거리(100mm 단위). 현재 번지를 알 수 없는 경우 아무것도 설정되지 않음 |
| **다음번지** | 다음 번지를 알 수 없는 경우 아무것도 설정되지 않음 |

### 11.7 실행 Cycle

| 코드 | 내용 |
|------|------|
| "0" | Cycle 없음 |
| "1" | 위치 확인 Cycle 중 |
| "2" | 이동 Cycle 중 |
| "3" | Unload Cycle 중 |
| "4" | Load Cycle 중 |
| "5" | 수출 Cycle 중 |
| "9" | 승간 이동 Cycle 중 |
| "21" | 주회 주행 Cycle 중 |

### 11.8 Vehicle 실행 Cycle 진척

> ※ 실행 Cycle이 "0"~"5"인 경우만 유효, 그 외에는 아무것도 설정되지 않음

| 코드 | 내용 |
|------|------|
| "0" | 실행 Cycle 없음 |
| "1" | 이동 중 |
| "2" | Unload 이동 중 |
| "3" | Unload 이재 중 |
| "4" | Load 이동 중 |
| "5" | Load 이재 중 |
| "7" | 대체 지시 대기 |

### 11.9 Carrier/Destination 정보

| 항목 | 설명 |
|------|------|
| **Carrier ID** | 현재 반송 중, 반송 예정 Carrier ID. 관련 Carrier ID가 없는 경우는 아무것도 설정되지 않음 |
| **Destination** | 목적지 Station 번호. 목적지 Station이 없거나 알 수 없는 경우에는 아무것도 설정되지 않음 |

### 11.10 E/M 상태

- **기본값:** "0000000"
- **설명:** E/M 상태를 Bit 배치로 설정
- **포함 항목:** 방전 표시, 작동 가능 출력, 배터리 수명, Refresh 충전 요구, 배터리 용량 Warning

### 11.11 GroupID

- **표현:** GroupName으로 표현
- **유효 문자열:** ["A"~"Z"]["a"~"z"]["0"~"9"]
- **기본값:** GroupID가 없는 경우 "0000"으로 설정

### 11.12 Port 정보

| 항목 | 설명 |
|------|------|
| **반송원 Port** | 반송원 Port ID. 관련 Port ID가 없으면 아무것도 설정되지 않음 |
| **반송처 Port** | 반송처 Port ID. 관련 Port ID가 없으면 아무것도 설정되지 않음 |

### 11.13 반송 우선도

| 값 | 설명 |
|----|------|
| 0~99 | 범위 |
| "0" | 무효 |
| "1" | 최저 우선순위 |
| "99" | 최고 우선순위 |

### 11.14 작업 상태 상세

> ※ 실행 Cycle이 다음과 같은 경우에만 유효: "0":Cycle 없음, "2":이동 Cycle 중

| 코드 | 내용 |
|------|------|
| "0" | 작업 없음 |
| "1" | 대기 중 |
| "2" | STAGE 대기 중 |
| "3" | Standby 대기 중 |
| "4" | 반송 Load 허가 없음 대기 중 |
| "5" | Carrier 회수 대기 대기 중 |
| "6" | MAP 전송 대기 중 |
| "101" | 이동 중 |
| "102" | Parking UTS 주행 중 |
| "103" | STAGE 이동 중 |
| "104" | Standby 이동 중 |
| "105" | Balance에 의해 이동 중 |
| "106" | Parking에 의해 이동 중 |
| "107" | Forecast 이동 중 (Config 설정에서 활성화/비활성화 설정 가능) |
| "108" | 세차 이동 중 (Config 설정에서 활성화/비활성화 설정 가능) |

### 11.15 추가 항목

| 항목 | 설명 |
|------|------|
| **대차 주행거리** | 대차가 Cycle 개시 후/주행로 변경 후부터 도착까지 주행한 거리(mm). 대차가 목적지에 도착했을 때 Set 한다. 도착 이외의 상태, 주행하지 않은 경우, 주행거리 불명인 경우 0이 설정됨 |
| **Command ID** | 반송 Command ID |
| **Bay 명칭** | VHL 현재 위치의 Bay 명칭 |
| **도착 예상 시간(ETA)** | 경로 확정 시 VHL 도착 예상시간(msec). ※도착 예상 시간은 경로 확정 시 시간으로 실시간 갱신되지 않음 |
| **예약 Command ID** | 예약 중인 반송 Command ID (Config 설정에서 활성화/비활성화 설정 가능) |

---

## 12. 수식 종합 및 계산 예시

> **📐 Smart ATLAS 시스템에서 사용되는 핵심 수식 5가지와 실제 계산 예시입니다.**

### 12.1 속도(Velocity) 계산 공식

```
속도(ν) = 거리(Δx) / 시간(Δt) × 60.0
```

| 변수 | 설명 | 단위 |
|------|------|------|
| ν | 속도 | m/min (미터/분) |
| Δx | 이동 거리 | mm |
| Δt | 시간 (두 메시지 수신 시간 차이) | 초 |
| ×60.0 | 초 → 분 단위 변환 | - |

**📝 계산 예시:**
```
조건:
- 거리(Δx) = 280mm
- 시간(Δt) = 1초

계산:
속도(ν) = 280 / 1 × 60.0 = 16,800 mm/min = 16.8 m/min
```

### 12.2 거리(Δx) 계산 공식

> **※ α: 최근(현재) 수신 메시지 정보 / β: 이전(마지막) 수신 메시지 정보**

#### Case 1: 연속된 RailEdge (α의 시작점과 β의 도착지점이 동일)

```
Δx = (β RailEdge 길이) - (β의 현재 번지 거리) + (α의 현재 번지 거리)
```

**📝 Case 1 계산 예시:**
```
조건:
- β RailEdge 길이 = 2000mm
- β의 현재 번지 거리 = 500mm
- α의 현재 번지 거리 = 300mm

계산:
Δx = 2000 - 500 + 300 = 1800mm
```

#### Case 2: 서로 다른 RailEdge (Dijkstra 경로 탐색 필요)

```
Δx = (β→α 시작점까지 RailEdge 길이의 합) - (β의 현재 번지 거리) + (α의 현재 번지 거리)
```

**📝 Case 2 계산 예시:**
```
조건:
- RailEdge 1 길이 = 100mm
- RailEdge 2 길이 = 200mm
- β의 현재 번지 거리 = 20mm
- α의 현재 번지 거리 = 0mm

계산:
Δx = (100 + 200) - 20 + 0 = 280mm
```

### 12.3 밀도(Density) 계산 공식

```
density = (vehicle 길이) × (현재 vehicle 수량) / {(RailEdge 길이) - (RailEdge 길이 % vehicle 길이)}
```

> **밀도 해석:** 0에 가까울수록 여유, 1에 가까울수록 혼잡

**📝 계산 예시:**
```
조건:
- Factory: M16A (vehicle 길이 = 1243mm)
- 현재 vehicle 수량 = 3대
- RailEdge 길이 = 5000mm

계산:
유효 길이 = 5000 - (5000 % 1243) = 5000 - 271 = 4729mm
density = 1243 × 3 / 4729 = 3729 / 4729 ≈ 0.789

해석: RailEdge의 약 79%가 Vehicle로 점유됨 (혼잡 상태)
```

### 12.4 FutureCost 계산 공식

```
FutureCost = baseCost
           + (futureTransCnt × transWeight)
           + (futureAcqCnt × acqTransWeight)
           + (futureDpstCnt × dpstTransWeight)
           + Junction penalty
```

| 변수 | 설명 |
|------|------|
| baseCost | 기본 비용 (거리/시간 기반) |
| futureTransCnt | 예상 Transfer 작업 수 |
| futureAcqCnt | 예상 Acquire 작업 수 |
| futureDpstCnt | 예상 Deposit 작업 수 |
| Junction penalty | 분기점 통과 시 추가 비용 |

**📝 계산 예시:**
```
조건:
- baseCost = 100
- futureTransCnt = 2, transWeight = 10
- futureAcqCnt = 1, acqTransWeight = 15
- futureDpstCnt = 3, dpstTransWeight = 12
- Junction penalty = 5

계산:
FutureCost = 100 + (2×10) + (1×15) + (3×12) + 5
           = 100 + 20 + 15 + 36 + 5
           = 176
```

### 12.5 absoluteVelocity 계산 공식

```
absoluteVelocity = (RailEdge의 현재 속도) / (RailEdge의 최대 속도)
```

> **용도:** ATLAS_RAIL_TRAFFIC 테이블에 저장되어 레일 구간의 상대적 속도 효율을 나타냄

**📝 계산 예시:**
```
조건:
- RailEdge 현재 속도 = 160 m/min
- RailEdge 최대 속도 = 200 m/min

계산:
absoluteVelocity = 160 / 200 = 0.8

해석: 최대 속도의 80%로 운행 중
```

#### absoluteVelocity 상태 해석표

| absoluteVelocity | 상태 | 설명 |
|------------------|------|------|
| **1.0** | 최대 속도 | 정상 최고 속도로 운행 |
| **0.8** | 정상 (80%) | 양호한 흐름 |
| **0.5** | 혼잡 (50%) | 속도 저하 발생 |
| **0.2** | 심각한 지연 | 정체 또는 장애 의심 |

### 12.6 수식 요약표

| 수식명 | 공식 | 사용처 |
|--------|------|--------|
| **속도(Velocity)** | `ν = Δx / Δt × 60.0` | DataSet.java, 실시간 속도 계산 |
| **거리(Distance)** | `Δx = RailEdge길이합 - β거리 + α거리` | 속도 계산 전처리 |
| **밀도(Density)** | `vhlLen × vhlCnt / 유효길이` | ATLAS_AREA_TRAFFIC, ATLAS_BRANCH_TRAFFIC |
| **FutureCost** | `baseCost + Σ(cnt × weight) + penalty` | DijkstraVhlRouteFind, 경로 최적화 |
| **absoluteVelocity** | `현재속도 / 최대속도` | ATLAS_RAIL_TRAFFIC, 상대 속도 효율 |

---

## ⚠️ 문서 간 주의사항

- **문서 버전 차이:** MCS_LOG_아키텍처(2022.07) vs OHT_속도분석(2025.08) vs OHT_VHL_분석(2025.01) - 최신 버전 확인 필요
- **속도 계산 예시 불일치:** OHT_속도분석.MD에서 속도=280 표기, 실제 공식 적용 시 280×60=16,800 mm/min (16.8 m/min)
- **추가 구현 필요 사항:** 장애 Case별 대응 방안, 패치/배포 절차서, UI 동작 구조 설명 등 미구현

---

*작성일: 2025년 | Smart ATLAS OHT 시스템 문서 종합*