# MCS LOG Architecture 프로젝트 지침

## 프로젝트 개요
이 프로젝트는 반도체 FAB의 smartATLAS 시스템 아키텍처에 관한 것으로, MCS(Material Control System) 로그 수집 및 분석, OHT/STK/Conveyor 반송 추적, 경로 탐색 알고리즘 등을 다룹니다.

- **작성**: AMHS운영팀 (2022.07)
- **대상 시스템**: smartATLAS (M15 Ph2)

---

## 1. 시스템 구성도

### 1.1 전체 연동 구조
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   M15A MCS  │     │   M15B MCS  │     │  smartSTAR  │
│  APP1/APP2  │     │  APP1/APP2  │     │  APP1/APP2  │
│   DB1/DB2   │     │   DB1/DB2   │     │   DB1/DB2   │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       └───────────┬───────┴───────────────────┘
                   │
           ┌───────▼───────┐
           │ smartATLAS    │
           │  (MCSLOG)     │
           └───────┬───────┘
                   │
    ┌──────────────┼──────────────┐
    ▼              ▼              ▼
┌────────┐   ┌──────────┐   ┌──────────┐
│M15A OHT│   │M15B OHT  │   │smartATLAS│
│CLWMCP  │   │CLWMCP    │   │   UI     │
└────────┘   └──────────┘   └──────────┘
```

### 1.2 데이터 제공 소스

| 소스 | 제공 데이터 | 프로토콜 |
|------|------------|----------|
| MCS | LOG(FTP/TIB), 반송 이벤트 Msg(TIB), 장비/재고상태(DB) | TIB/rv, FTP |
| OHT | OHT 반송 상태, Layout Data | UDP, FTP |
| smartSTAR | 반송 이력 정보, smartFX DB | DB |

---

## 2. 객체 관계도

### 2.1 장비(EQP) 유형

| 장비 | 설명 |
|------|------|
| **EQP** | 생산장비. 양방향 Load/Unload 포트만 가진 장비로 취급 |
| **FIO** | AMHS 입/출고용. 자체 저장공간 없음, OHT를 통해 반송 |
| **OHT** | VHL 제어 장비 (OCS, CLWMCP). VHL로 장비간 반송 수행 |
| **STK** | Stocker. Shelf로 Carrier 저장, RM(Crane)으로 내부 반송. N2STK는 N2Purge 가능. LIFTER/ZIPTOWER/INTERLAYER는 층간반송 |
| **CONVEYOR** | 다수 입/출고 포트, 동시에 많은 Carrier 이동 가능. 동간 반송에 주로 사용 |

### 2.2 Node 유형

| Node | 설명 |
|------|------|
| **RailNode** | Rail 상의 노드 (Branch/Junction) |
| **StkPortNode** | Stocker 포트 (LP/BP/OP SubPort 포함) |
| **CnvPortNode** | Conveyor 포트 |
| **StbNode** | STB 포트 |
| **EqpPortNode** | 생산장비 포트 |
| **FioPortNode** | FIO 포트 (LP/OP/BP SubPort 포함) |
| **StkRmNode** | Stocker RM (Crane) |
| **StkShelfNode** | Stocker Shelf |

### 2.3 Edge 유형

| Edge | 설명 |
|------|------|
| **RailEdge** | Rail 구간 (시작점 → 끝점) |
| **TransferEdge** | Station과 Port 간 연결 (ACQUIRE/DEPOSIT) |
| **LongEdge** | Branch~Junction 사이 RailEdge 집합 |
| **BranchJoinEdge** | RailBranch~Junction 사이 RailEdge 집합 |
| **StkRmEdge** | Stocker 내부 RM 이동 경로 |
| **CnvZoneEdge** | Conveyor Zone 간 이동 경로 |

### 2.4 Job & Command 관계

```
Job (반송 작업)
 ├── OHT Command → VHL task
 ├── STK Command → RM task
 └── CNV Command → CnvTask
```

- **Job**: MCS가 생성하는 반송 작업 단위
- **Command**: Job 수행을 위해 각 장비에 내리는 세부 명령
- **JobId 변경 시**: FAB간/동간 반송 중 MCS가 바뀌거나 목적지가 바뀌는 경우. smartATLAS는 ID 변경 및 History 기록

---

## 3. 수집 서버 구조

### 3.1 메시지 처리 흐름

```
┌─────────────┐
│   OHT MCP   │──UDP──┐
└─────────────┘       │
┌─────────────┐       ▼
│  TIB/rv     │──► CMessage Listener
└─────────────┘       │
                      ▼
              ┌───────────────┐
              │MessageDispatcher│
              └───────┬───────┘
       ┌──────────────┼──────────────┬──────────────┐
       ▼              ▼              ▼              ▼
┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐
│OhtMsgWorker│ │EiMsgWorker │ │TsMsgWorker │ │CnvMsgWorker│
└────────────┘ └────────────┘ └────────────┘ └────────────┘
       │              │              │              │
       ▼              ▼              ▼              ▼
┌─────────────────────────────────────────────────────────┐
│              DataService.getDataSet()                    │
└─────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│              Logpresso API / UDP                         │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Worker별 역할

| Worker | 역할 |
|--------|------|
| **OhtMsgWorker** | VHL 상태 추적, RailEdge/TransferEdge 속도 Update, Carrier 위치 |
| **EiMsgWorker** | Command 생성/완료 처리, Carrier 위치, RmEdge Cost Update |
| **TsMsgWorker** | Job 생성/완료 처리 |
| **CnvMsgWorker** | CnvTask 추적, Conveyor 상태 Update |
| **UiMsgWorker** | 포트/장비 상태 Update, Job wakeup 처리 |

### 3.3 주요 Callback

| Callback | 소스 | 처리 내용 |
|----------|------|----------|
| UdpMessageCallback | OHT MCP | UDP 메시지 수신 |
| McsMhsMsgCallBack | MCS→MHS | 반송 이벤트 |
| MhsMcsMsgCallBack | MHS→MCS | 반송 요청 |
| TsEiMsgCallBack | TS→EI | 반송 명령 |
| EiTsMsgCallBack | EI→TS | 반송 응답 |
| McsUiMsgCallBack | MCS→UI | 상태 정보 |

---

## 4. Batch Job

### 4.1 트래픽 관련 Batch

| Batch | 주기 | 내용 | 적재 테이블 |
|-------|------|------|------------|
| **AreaVhlCountBatch** | 6/10초 | Area별 VHL 수량 집계 | ATLAS_VHL_COUNT |
| **BranchTrafficBatch** | 4/10초 | BranchJoinEdge/Area 통계 | ATLAS_BRANCH_TRAFFIC, ATLAS_AREA_TRAFFIC |
| **TrafficBatch** | 3/10초 | RailEdge별 통계 | ATLAS_RAIL_TRAFFIC |
| **VhlBatch** | 1분 | 전체 VHL 상태 | Logpresso |

### 4.2 BranchTrafficBatch 상세 Data

| 필드 | 설명 |
|------|------|
| Length | 길이(mm) |
| Speed | BranchJoinEdge 속도(m/min, 지수평활법) |
| ElapsedTime | 통과 소요시간 |
| passCnt | 지난 10초간 VHL 통과수량 |
| pathPredictQueueSizeAvg | 잔여 통과예정 수량 |
| edgeIds | railEdgeId 목록 |
| isAvailable | laneCut 여부 |
| jobCost | 최근 1분간 ATLAS_ROUTE상 itemCost 평균 |
| railTrafficInputCnt | deposit 이동중인 command 수량 |
| railTrafficOutputCnt | acquire 이동중인 command 수량 |
| vhlCnt | 현재 VHL 수량 |
| Density | vhl길이×vhlCnt / (rail길이 – rail길이 % vhl길이) |
| passJobCnts[0~4] | 과거 1~5분 내 지나간 VHL 수량 |
| pathPredictQCnts[0~4] | 향후 1~5분 내 지나갈 Job 수량 |

### 4.3 TrafficBatch 상세 Data

| 필드 | 설명 |
|------|------|
| maxVelocity | OHT 설정 속도 |
| velocity | 현재 RailEdge 속도(m/min, 지수평활법) |
| absoluteVelocity | velocity / maxVelocity |
| passCnt | VHL 통과수량 |
| vhlDensityWeight | rail상 VHL 밀도 |
| predictQueueAllCnt | 도착 예상 VHL 수량 |
| predictUnder1minCnt ~ predictOver5minCnt | 시간대별 도착 예상 수량 |
| vhlObsBzStopCnt, vhlJamCnt, vhlAbnormalCnt, vhlE84Cnt | 상태별 VHL 수량 |

### 4.4 기타 Batch

| Batch | 주기 | 내용 |
|-------|------|------|
| **BridgeEqpMonitorBatch** | 1분 | Lifter/Conveyor 처리량, 부하상태, 소요시간 모니터링 |
| **BridgeEqpNaviBatch** | 30초 | 장비별 인접 장비 순서 정보 |
| **CurrentJobListBatch** | 10초 | 현재 수행중인 Job 목록 Publish |
| **DataSetRefreshBatch** | 매일 10시 | Layout/장비/Port 정보 갱신 |
| **ExportAllLayoutToLogpressoBatch** | 매시 20분 | EDGE, LONGEDGE 등 기준정보 적재 |
| **HelloBatch** | 1분 | Old Job Cleaning, Garbage Route Cleaning |
| **PredictionClean** | 5/10초 | 만료된 pathPredictQueue 제거 |
| **LaneCutRefreshBatch** | 5/10초 | FTP로 lanecut.dat 파일 갱신 |
| **ServerStatusBatch** | 2초 | CPU, Memory, Thread 상태 집계 |
| **ObjOnMapDiffUpdateBatch** | 1초 | OHT, Command 통계 적재 |
| **ObjOnMapUpdateBatch** | 30초 | Port, Eqp별 상태 적재 |

---

## 5. 메시지 처리 상세

### 5.1 TS Msg 처리 (Job 관련)

| 메시지 | 처리 내용 |
|--------|----------|
| MCSMHS_TRANSPORT_JOB_CREATED_EVENT | Job 생성 |
| MCSMHS_TRANSPORT_JOB_CHANGED_EVENT | Job 변경 |
| MCSMHS_TRANSPORT_JOB_STARTED_EVENT | Job 시작 |
| MCSMHS_TRANSPORT_JOB_COMPLETED_EVENT | Job 완료 |
| MCSMHS_TRANSPORT_JOB_ABORTED_EVENT | Job 중지 |
| MHSMCS_MATERIAL_DEST_REP | Job 생성 전 초기정보 |
| MHSMCS_TRANSPORT_JOB_SCHEDULE_REQ | MES→MCS 반송명령 |

### 5.2 EI Msg 처리 (OHT Command)

| 메시지 | 처리 내용 | cmdState |
|--------|----------|----------|
| RAIL-CARRIERTRANSFERREPLY | Command 생성 | QUEUED |
| RAIL-VEHICLEASSIGNED | VHL 할당 | ACQMOVING |
| RAIL-VEHICLEARRIVED | Station 도착 | - |
| RAIL-VEHICLEACQUIRESTARTED | Carrier 들어올리기 시작 | ACQUIRING |
| RAIL-CARRIERINSTALLED | Carrier 위치→VHL | - |
| RAIL-VEHICLEACQUIRECOMPLETED | Carrier 들어올리기 완료 | - |
| RAIL-VEHICLEDEPARTED | VHL 출발 | DESTMOVING |
| RAIL-VEHICLEDEPOSITSTARTED | Carrier 내려놓기 시작 | DEPOSITING |
| RAIL-VEHICLEDEPOSITCOMPLETED | Carrier 내려놓기 완료 | - |
| RAIL-CARRIERREMOVED | VHL에서 Carrier 제거 | - |
| RAIL-TRANSFERCOMPLETED | Command 완료 | COMPLETED |
| RAIL-VEHICLEUNASSIGNED | VHL 할당해제 | QUEUED |

### 5.3 EI Msg 처리 (STK Command)

| 메시지 | 처리 내용 |
|--------|----------|
| STORAGE-CARRIERTRANSFER | STK 내부 반송명령 수신 |
| STORAGE-CARRIERTRANSFERREPLY | 명령 수행여부 회신 |
| STORAGE-TRANSFERINITIATED | 반송 시작 |
| STORAGE-CRANEACTIVE | RM 활성화 |
| STORAGE-CARRIERTRANSFERRING | RM 적재 후 이동 시작 |
| STORAGE-CRANEIDLE | 목적지 이재 완료 |
| STORAGE-TRANSFERCOMPLETED | 반송 완료 |
| STORAGE-CARRIERSTORED | Shelf에 저장 |
| STORAGE-CARRIERSTOREDALT | Shelf에 임시 저장 |
| STORAGE-CARRIERWAITIN | 입고포트 도착 |
| STORAGE-CARRIERWAITOUT | 출고포트 도착 |

### 5.4 Conveyor Msg 처리

| 메시지 | 처리 내용 |
|--------|----------|
| EVENT_CARRIER_DETECTED | Carrier 감지, Task 생성 |
| EVENT_READ_RFID | CarrierID 읽음 |
| EVENT_TRANSFER_INITIATED | 반송 시작 |
| EVENT_TRANSFER_TRANSFERRING | 이동중 |
| EVENT_TRANSFER_COMPLETED | 반송 완료 |
| EVENT_CARRIER_REMOVED | Carrier 제거 |
| tcmTransferInfo | Zone 이동 정보 |

---

## 6. 경로 탐색 알고리즘

### 6.1 Dijkstra 알고리즘 (변형)

smartATLAS는 **변형된 Dijkstra 알고리즘**을 사용합니다:

- **문제**: RailNode와 Station을 통한 포트 연결이 복잡하여 일반 Dijkstra로는 경로탐색 불가능한 경우 발생
- **해결**: TransferEdge_Acquire를 통과 시, 동일 RailEdge상에서는 offset이 더 큰 쪽으로만 방문 가능하도록 제한. 이후 해당 노드를 미방문 처리하여 다시 방문 가능하도록 함

### 6.2 경로 기록 (ComparableNode)

| 필드 | 설명 |
|------|------|
| Cost | Source로부터 해당 Node까지의 Cost |
| 방문여부 | 방문 여부 |
| Predecessor | 선행 노드 경로 기록 |
| Other 노드 이력 | Acquire Edge 통과 시 RailNode 재방문용 Backup |
| Other Cost | RailNode까지의 Cost Backup |

### 6.3 Navigator 경로 탐색 흐름

1. Job의 routeSelection으로 통과해야 할 LongEdge 목록 취득
2. fromContainer가 VHL인 경우 출발지를 RailNode로 설정
3. M14A↔M14B 반송 여부 확인
4. Bridge 장비 경유 시 최소 Cost 경로 탐색
5. SubPath별 RouteSelection 반영
6. 최종 경로 Return

### 6.4 RouteSelection

```
route_SCH_1.dat 파일 내용 예시:
B26 → B21 이동 시
- B26에서는 OUT01을 통해 나감
- 4926번지를 경유
- B21의 IN02를 통해 진입
```

---

## 7. Edge Cost 계산

### 7.1 Cost 유형

| 메서드 | 설명 | 용도 |
|--------|------|------|
| getCost(carrierId) | 지수평활법 평균 속도 기반 소요시간 | UI 속도 표시 |
| getPPCost() | PathPredictQueue 수량 × 60ms 추가 | N:1 소요시간 계산 |
| getVhlCountCost() | 현재 VHL 수량 기반 penalty | VHL Assign 예측 |
| getLast1HourCost() | 최근 1시간 평균 통과 소요시간 | FutureCost 대체 |

### 7.2 FutureCost 계산

```
FutureCost = baseCost 
           + (futureTransCnt × transWeight)
           + (futureAcqCnt × acqTransWeight)
           + (futureDpstCnt × dpstTransWeight)
           + Junction penalty (있는 경우)
```

### 7.3 FutureTransCount

- **TransferEdge**: after-transOverlapIntervalT ~ after+avgTransferCost+avgVhlCallCost 사이의 PathPredictQueue 크기
- **RailEdge**: 동일
- **StkRmEdge**: after-transOverlapIntervalT ~ after+avgRemovalIntervalT 사이의 PathPredictQueue 크기
- **CnvEdge**: after-transOverlapIntervalT ~ after+getCost() 사이의 PathPredictQueue 크기

---

## 8. 예상 소요시간 보정

### 8.1 ML/DL 보정

PathPredictQueue의 시간대별 분포가 균일하지 않아 (1~3분 이내 많음) 예측시간과 실제 소요시간 차이가 큼. ML/DL을 통해 보정 수행.

```java
AtlasPredictEnhanceTibrvReq.getInstance()
    .requestPredictEnhancedValue(
        fromFabId, toFabId,
        futureAcqCnt, futureDpstCnt,
        ftcs_TRANS_ACQ, ftcs_TRANS_DPST,
        ftcs_STK, ftcs_CNV, ftcs_RAIL,
        cs_TRANS_ACQ, cs_TRANS_DPST,
        cs_STK, cs_CNV, cs_RAIL,
        predictDistance
    );
```

---

## 9. OHT 위치/속도 추정

### 9.1 위치 추정 (Message 유실 대응)

약 5초 간격으로 위치 보고하지만, 특정 위치에서 보고하지 않고 건너뛰는 경우:

1. A→B 이동경로에 대해 **Dijkstra 알고리즘**으로 최단 경로 탐색
2. 가장 소요시간이 짧은 경로로 이동했을 것으로 추정
3. 가상의 이동기록 생성, 각 노드간 동일 속도로 이동한 것으로 가정하여 도착시간 계산

---

## 10. Empty Vehicle Assign 예측

### 10.1 VHL 선정 기준

1. CarrierID에 따라 POD/FOUP VHL 구분 (4RP 여부)
2. 가까운 VHL 선정
   - Deposit moving중 VHL도 대상 (목적지가 동일 LongEdge의 호출대상 Station 전에 있어야 함)
   - 동일 LongEdge의 IDLE VHL은 최소 정지거리 이전만 대상

### 10.2 Station/VHL Type 매칭

| carryType | 의미 |
|-----------|------|
| 0 | all |
| 1 (00001) | FOUP |
| 2 (00010) | POD |
| 5 (00101) | FOUP |
| 9 (01001) | FOUP |
| A (01010) | POD |

| vhlType | 의미 |
|---------|------|
| 1 | FOUP VHL |
| 2 | POD VHL |
| 4 | 청소 VHL |

---

## 11. STB 재고정보 적재

### 11.1 초기 적재

MCS DB에서 현재 STB별 Carrier 목록 Query (반송명령 없는 Carrier만)

### 11.2 실시간 추적

| 이벤트 | 처리 |
|--------|------|
| 입고 | UDP 메시지상 Carrier 위치가 VHL인 경우 해당 STB에서 제거 |
| 출고 | 목적지가 STB인 VHL이 Carrier를 갖고 있다가 Empty가 될 경우 목적지로 위치 변경 |

### 11.3 Abnormal Case 처리

| Case | 처리 |
|------|------|
| 출발지 오류 | 출발지 STB에 Carrier 없거나 다른 STB에 있는 경우 → 경보 발송 |
| 목적지 점유 | 이미 점유된 STB로 반송명령 → 경보 발송 |

---

## 12. Properties 파일 설정

### 12.1 공통 설정

| 속성 | 설명 |
|------|------|
| Env | REAL/TEST/QA (TIB/rv Subject 생성에 영향) |
| FabIdList | MCS기준 FAB ID 목록 (콤마 구분) |
| ProcessType | DATAMAKER(수집서버) / QUERY(Logpresso Query 전용) |

### 12.2 수집서버 설정

| 속성 | 설명 |
|------|------|
| Atlas.Daemon | Atlas tib/rv 통신용 daemon 서버 |
| Atlas.Gid | cmessage 통신 그룹 ID |
| Atlas.ThreadPool.size | Worker Thread 수량 (CPU 코어 수) |
| Atlas.MonitoringMode | Y/N (CPU 부하 함수 Disable) |
| Logpresso.Host/Host2 | Logpresso 서버 주소 |
| Logpresso.UdpHost/UdpHost2 | UDP 전송 Target 주소 |
| Logpresso.UdpPort | UDP 전송 Port |

### 12.3 FAB별 설정

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

## 13. 용어 정리

| 용어 | 설명 |
|------|------|
| MCS | Material Control System - 자재 제어 시스템 |
| MHS | Material Handling System - 자재 반송 시스템 |
| OHT | Overhead Hoist Transport - 천장 이송 시스템 |
| VHL | Vehicle - OHT 운반체 |
| STK | Stocker - 저장 장비 |
| STB | Side Track Buffer - 측면 버퍼 |
| FIO | Factory I/O - AMHS 입출고 장치 |
| RM | RackMaster/Crane - Stocker 내부 반송 장치 |
| CLWMCP | OHT 제어 장비 |
| TIB/rv | TIBCO Rendezvous - 메시지 통신 프로토콜 |
| Logpresso | 로그 수집/분석 플랫폼 |
| SSSP | Single Source Shortest Path - 단일 출발점 최단 경로 |
| PathPredictQueue | 경로 예측 대기열 |
| LaneCut | 레일 구간 차단 |
| Junction | 합류점 |
| Branch | 분기점 |

---

## 14. 추가 구현 필요 사항

- 장애 Case별 대응 방안
- 패치/배포 유형별 절차서
- Logpresso 내 구현사항 설명
- UI 동작 구조 설명
- 개발환경 설정 방법

---

*문서 버전: 2022.07*
*작성 기준: M15 Ph2 생산정보시스템 구축 MCSLOG Architecture 정의서*