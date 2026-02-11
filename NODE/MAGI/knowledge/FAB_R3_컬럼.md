# QUWA FAB 컬럼 사전
> **구조**: `{FAB}.{Category}.{SubCategory}.{Metric}`
> **영문 Description → 한글 번역 완료**

---

## 8. R3 계열

### R3 (144건) — AMHS 모니터링 (CPU/RAM 제외)

#### R3.OHT.ALERT — OHT 알람 (21건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `R3.OHT.ALERT.OHTSTATUSERROR` | OHT상태가 Pausing으로 장시간 대기(MCS) |
| 2 | `R3.OHT.ALERT.OHTLOADFAIL` | OHT 반송중 TransferCompleted ResultCode (1)발생(Message) |
| 3 | `R3.OHT.ALERT.OHTDISCONNECTION` | MCS-OHT간 통신이 끊어짐(Message) |
| 4 | `R3.OHT.ALERT.PIOCOMMERROR` | PIO Communication Error 발생(Message) |
| 5 | `R3.OHT.ALERT.OHTCANNOTEXECUTE` | OHT에 반송명령시 Reply (2) Currently not able to excute 발생(MCS) |
| 6 | `R3.OHT.ALERT.OHTINVALIDPARA` | OHT에 반송 명령시 Reply (3)으로 Parameter 값이 다른 경우 |
| 7 | `R3.OHT.ALERT.OHTRETRYFAILED` | OHT 3회 반송 시도 실패 발생 |
| 8 | `R3.OHT.ALERT.OHTMCPALARM` | OHTMCPAlarm 발생(Message) |
| 9 | `R3.OHT.ALERT.OHTCONNECTION` | MCS-OHT간 통신이 연결(Message) |
| 10 | `R3.OHT.ALERT.OHTPAUSED` | OHT MCP PAUSED로 State 변경 |
| 11 | `R3.OHT.ALERT.OHTAUTO` | OHT MCP AUTO로 State 변경 |
| 12 | `R3.OHT.ALERT.OHTHIDALARM` | OHT HID Error 발생 |
| 13 | `R3.OHT.ALERT.R3EOHTMCPALARMCNT` | 현재 OHT Alarm 건수(MCSDB 기준) |
| 14 | `R3.OHT.ALERT.R3EOHTHIDALARM` | R3E OHT HIDAlarm 발생 |
| 15 | `R3.OHT.ALERT.R3EOHTAUTO` | R3E OHT AUTO 발생 |
| 16 | `R3.OHT.ALERT.R3EOHTDISCONNECTION` | R3E OHT DisConnection 발생 |
| 17 | `R3.OHT.ALERT.R3EOHTPAUSED` | R3E OHT PAUSED 발생 |
| 18 | `R3.OHT.ALERT.R3EOHTMCPALARM` | R3E OHT MCP Alarm 발생 |
| 19 | `R3.OHT.ALERT.M10BOHTMCPALARMCNT` | 현재 OHT Alarm 건수(MCSDB 기준) |
| 20 | `R3.OHT.ALERT.R3OHTMCPALARMCNT` | 현재 OHT Alarm 건수(MCSDB 기준) |
| 21 | `R3.OHT.ALERT.R3EOHTCONNECTION` | R3E OHT Connection |

#### R3.OHT.ALL — OHT 전체 (1건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `R3.OHT.ALL.CLUSTER` | 체크 대상 MCP의 Clustering 상태 점검(FTP) 체크대상 MCP == current : green MCP7 <> Online : red 그외 yellow |

#### R3.OHT.OHT_P — OHT Primary 서버 (5건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `R3.OHT.OHT_P.CLUSTER` | 체크 대상 MCP의 Clustering 상태 점검(FTP) 체크대상 MCP == current : green MCP7 <> Online : red 그외 yellow |
| 2 | `R3.OHT.OHT_P.DISK_USED` | 디스크사용율(%)(FTP) |
| 3 | `R3.OHT.OHT_P.PROS_STATUS` | not_active 개수가 5개이하 : green not_active 개수가 5개 초과이고, primary, secondary 모두 상태값이 0인게 하나 이상 : yellow not_active 개수가 5개 초과이고, primary, secondary 모두 상태값이 0인게 존재하지 않음 : red |
| 4 | `R3.OHT.OHT_P.SERVICE_STATUS` | CLPSTAT의 결과 MCP7 ............: Online 이 아닐 경우(FTP) |
| 5 | `R3.OHT.OHT_P.ASSIGNEDWAITTOTAL` | ASSIGNEDWAITTOTAL VALUE(FTP) |

#### R3.OHT.OHT_S — OHT Secondary 서버 (5건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `R3.OHT.OHT_S.CLUSTER` | 체크 대상 MCP의 Clustering 상태 점검(FTP) 체크대상 MCP == current : green MCP7 <> Online : red 그외 yellow |
| 2 | `R3.OHT.OHT_S.DISK_USED` | 디스크사용율(%)(FTP) |
| 3 | `R3.OHT.OHT_S.PROS_STATUS` | not_active 개수가 5개이하 : green not_active 개수가 5개 초과이고, primary, secondary 모두 상태값이 0인게 하나 이상 : yellow not_active 개수가 5개 초과이고, primary, secondary 모두 상태값이 0인게 존재하지 않음 : red |
| 4 | `R3.OHT.OHT_S.SERVICE_STATUS` | CLPSTAT의 결과 MCP7 ............: Online 이 아닐 경우(FTP) |
| 5 | `R3.OHT.OHT_S.ASSIGNEDWAITTOTAL` | ASSIGNEDWAITTOTAL VALUE(FTP) |

#### R3.OHT.ALARM — OHT 클러스터 알람 (1건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `R3.OHT.ALARM.CLUSTERSTATE` | OHT CLUSTER SERVER가 primary 또는 secondary로 전환되거나 Down/Up 상태일 때 알람 발송 |

#### R3.OHT.SERVER_P — OHT Server Primary (1건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `R3.OHT.SERVER_P.STATE` | 체크 대상 MCP의 Clustering 대상 Server 체크 1 = server1 2 = server2 3 = server down/up |

#### R3.OHT.SERVER_S — OHT Server Secondary (1건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `R3.OHT.SERVER_S.STATE` | 체크 대상 MCP의 Clustering 대상 Server 체크 1 = server1 2 = server2 3 = server down/up |

#### R3.MCS.R3MCSA1 — MCS A1 서버 (1건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `R3.MCS.R3MCSA1.DISK_APP1` | 디스크사용율(%)(FTP) |

#### R3.MCS.R3MCSA2 — MCS A2 서버 (1건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `R3.MCS.R3MCSA2.DISK_APP1` | 디스크사용율(%)(FTP) |

#### R3.MCS.ALERT — MCS 알람 (7건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `R3.MCS.ALERT.PROCESSDOWN` | MCS에서 특정 Process의 비정상 Down을 감지(Message) |
| 2 | `R3.MCS.ALERT.PROCESSUPFAIL` | MCS내 컨트롤 서버가 Process를 StartUp시켰으나 제대로 살아나지 않을 경우(Message) |
| 3 | `R3.MCS.ALERT.PROCESSHANG` | MCS내 컨트롤 서버가 MCS에 Heartbeat메시지를 날렸으나 반응이 없을 경우 Hang으로 간주(Message) |
| 4 | `R3.MCS.ALERT.PROCESSUPSTART` | MCS내 컨트롤 서버가 Process를 StartUp시도하는 경우(Message) |
| 5 | `R3.MCS.ALERT.PROCESSSHUTDOWNFAIL` | Process에 shutdown명령을 내렸으나 종료되지 않음(Message) |
| 6 | `R3.MCS.ALERT.MCPMCSCOMMPROBLEM` | MCS가 OHT MCP로부터 일정시간 Event를 받지 않을 경우 발생 |
| 7 | `R3.MCS.ALERT.DBQUERYHANG` | No response the DB Query process |

#### R3.QUE.OHT — Queue — OHT (2건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `R3.QUE.OHT.OHTUTIL` | OHT사용율(%)(MCS) |
| 2 | `R3.QUE.OHT.CURRENTOHTQCNT` | 현재 OHT반송 Q 수 |

#### R3.QUE.LOAD — Queue — Load 반송 (8건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `R3.QUE.LOAD.CURRENTLOADQCNT` | 반송 큐 개수: 목적지가 생산장비인 건(MCS) |
| 2 | `R3.QUE.LOAD.AVGLOADTIME` | 최근 10분간 평균 Load반송시간(HySTAR) |
| 3 | `R3.QUE.LOAD.AVGLOADTIME1MIN` | 최근 1분간 평균 Load반송시간(smartSTAR) |
| 4 | `R3.QUE.LOAD.EQ_LOAD_DAILY_CNT_TWO_DAYS_AGO` | 통합 모니터링 이틀 전 EQ LOAD COUNT |
| 5 | `R3.QUE.LOAD.EQ_LOAD_DAILY_TIME_TWO_DAYS_AGO` | 통합 모니터링 이틀 전 EQ LOAD TIME |
| 6 | `R3.QUE.LOAD.EQ_LOAD_DAILY_CNT` | 통합 모니터링 전날 EQ LOAD COUNT |
| 7 | `R3.QUE.LOAD.EQ_LOAD_DAILY_TIME` | 통합 모니터링 전날 EQ LOAD TIME |
| 8 | `R3.QUE.LOAD.AVGFOUPLOADTIME` | 최근 10분간 FOUP 평균 Load반송시간(smartSTAR) |

#### R3.QUE.ALL — Queue — 전체 (3건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `R3.QUE.ALL.CURRENTQCNT` | 현재 반송 QUEUE 수 |
| 2 | `R3.QUE.ALL.CURRENTQCOMPLETED` | 최근 10분간 반송명령 완료수(MCS) |
| 3 | `R3.QUE.ALL.CURRENTQCREATED` | 최근10분간 Q생성 수(MCS) |

#### R3.QUE.ABN — Queue — 이상 (4건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `R3.QUE.ABN.QUETIMEDELAY` | 30분 이상 지연된 FOUP 목록 알림(MCS) |
| 2 | `R3.QUE.ABN.AOTRANSDELAY` | 5분마다 쿼리 검색하여 10분이상 Auto Output Port에 대기하고 있는 FOUP이 하나이상 존재하면 FOUP 개수를 알람 |
| 3 | `R3.QUE.ABN.AOTRANSDELAYDET` | 10분이상 AO Port 반송 Delay 발생되는 FOUP 정보 통지(MCS) |
| 4 | `R3.QUE.ABN.N2STOCKERDELAY` | 60분이상 N2Purge Stocker Port 내 FOUP 지연 발생 통지(MCS) |

#### R3.QUE.SFAB — Queue — SFAB 반송 (24건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `R3.QUE.SFAB.RECEIVETOM15` | R3→M15 SFAB RECEIVE 반송 QUEUE |
| 2 | `R3.QUE.SFAB.RETURNTOM15` | R3→M15 SFAB RETURN 반송 QUEUE |
| 3 | `R3.QUE.SFAB.COMPLETETOM15` | R3→M15 SFAB COMPLETE 반송 QUEUE |
| 4 | `R3.QUE.SFAB.SENDTOM15` | R3→M15 SFAB SEND 반송 QUEUE |
| 5 | `R3.QUE.SFAB.RECEIVETOM11` | R3→M11 SFAB RECEIVE 반송 QUEUE |
| 6 | `R3.QUE.SFAB.RETURNTOM11` | R3→M11 SFAB RETURN 반송 QUEUE |
| 7 | `R3.QUE.SFAB.COMPLETETOM11` | R3→M11 SFAB COMPLETE 반송 QUEUE |
| 8 | `R3.QUE.SFAB.SENDTOM11` | R3→M11 SFAB SEND 반송 QUEUE |
| 9 | `R3.QUE.SFAB.SENDTOM16` | R3→M16 SFAB SEND 반송 QUEUE |
| 10 | `R3.QUE.SFAB.SENDTOM14` | R3→M14 SFAB SEND 반송 QUEUE |
| 11 | `R3.QUE.SFAB.SENDTOM10` | R3→M10 SFAB SEND 반송 QUEUE |
| 12 | `R3.QUE.SFAB.SENDQUEUETOTAL` | R3→ALL SFAB SEND 반송 QUEUE |
| 13 | `R3.QUE.SFAB.RECEIVETOM16` | R3→M16 SFAB RECEIVE 반송 QUEUE |
| 14 | `R3.QUE.SFAB.RECEIVETOM14` | R3→M14 SFAB RECEIVE 반송 QUEUE |
| 15 | `R3.QUE.SFAB.RECEIVETOM10` | R3→M10 SFAB RECEIVE 반송 QUEUE |
| 16 | `R3.QUE.SFAB.RECEIVEQUEUETOTAL` | R3→ALL SFAB RECEIVE 반송 QUEUE |
| 17 | `R3.QUE.SFAB.RETURNTOM16` | R3→M16 SFAB RETURN 반송 QUEUE |
| 18 | `R3.QUE.SFAB.RETURNTOM14` | R3→M14 SFAB RETURN 반송 QUEUE |
| 19 | `R3.QUE.SFAB.RETURNTOM10` | R3→M10 SFAB RETURN 반송 QUEUE |
| 20 | `R3.QUE.SFAB.RETURNQUEUETOTAL` | R3→ALL SFAB RETURN 반송 QUEUE |
| 21 | `R3.QUE.SFAB.COMPLETETOM16` | R3→M16 SFAB COMPLETE 반송 QUEUE |
| 22 | `R3.QUE.SFAB.COMPLETETOM14` | R3→M14 SFAB COMPLETE 반송 QUEUE |
| 23 | `R3.QUE.SFAB.COMPLETETOM10` | R3→M10 SFAB COMPLETE 반송 QUEUE |
| 24 | `R3.QUE.SFAB.COMPLETEQUEUETOTAL` | R3→ALL SFAB COMPLETE 반송 QUEUE |

#### R3.STRATE.ALL — Storage — FAB 전체 (4건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `R3.STRATE.ALL.FABSTORAGECAPACITY` | FAB 전체저장용량 (smartSTAR) ReservedLocationCount 포함, Down된 장비 제외 |
| 2 | `R3.STRATE.ALL.FABSTORAGECOUNT` | FAB 저장수 (smartSTAR) ReservedLocationCount 포함, Down된 장비 제외 |
| 3 | `R3.STRATE.ALL.FABSTORAGERATIO` | FAB 저장율 (smartSTAR) ReservedLocationCount 포함, Down된 장비 제외 |
| 4 | `R3.STRATE.ALL.FABSTORAGESERVICECAPA` | FAB 저장가능용량 (smartSTAR) ReservedLocationCount 포함, Down된 장비 제외 |

#### R3.STRATE.STK — Storage — STK (8건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `R3.STRATE.STK.N2STORAGESERVICECAPA` | STK N2 저장가능용량 (smartSTAR) N2Stocker로 이동중인 Queue 개수 포함, Down인 장비 제외 |
| 2 | `R3.STRATE.STK.N2STORAGECOUNT` | STK N2 저장수 (smartSTAR) N2Stocker로 이동중인 Queue 개수 포함, Down인 장비 제외 |
| 3 | `R3.STRATE.STK.NONN2STORAGESERVICECAPA` | STK NONN2 저장가능용량 (smartSTAR) |
| 4 | `R3.STRATE.STK.N2STORAGERATIO` | STK N2 저장율 (smartSTAR) N2Stocker로 이동중인 Queue 개수 포함, Down인 장비 제외 |
| 5 | `R3.STRATE.STK.N2STORAGECAPACITY` | STK N2 전체저장용량 (smartSTAR) N2Stocker로 이동중인 Queue 개수 포함, Down인 장비 제외 |
| 6 | `R3.STRATE.STK.NONN2STORAGERATIO` | STK NONN2 저장율 (smartSTAR) |
| 7 | `R3.STRATE.STK.NONN2STORAGECAPACITY` | STK NONN2 전체저장용량 (smartSTAR) |
| 8 | `R3.STRATE.STK.NONN2STORAGECOUNT` | STK NONN2 저장수 (smartSTAR) |

#### R3.STRATE.STB — Storage — STB (8건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `R3.STRATE.STB.N2STORAGERATIO` | STB N2 저장율 (smartSTAR) N2Stocker로 이동중인 Queue 개수 포함, Down인 장비 제외 |
| 2 | `R3.STRATE.STB.N2STORAGECAPACITY` | STB N2 전체저장용량 (smartSTAR) N2Stocker로 이동중인 Queue 개수 포함, Down인 장비 제외 |
| 3 | `R3.STRATE.STB.NONN2STORAGECAPACITY` | STB NONN2 전체저장용량 (smartSTAR) |
| 4 | `R3.STRATE.STB.NONN2STORAGERATIO` | STB NONN2 저장율 (smartSTAR) |
| 5 | `R3.STRATE.STB.N2STORAGESERVICECAPA` | STB N2 저장가능용량 (smartSTAR) N2Stocker로 이동중인 Queue 개수 포함, Down인 장비 제외 |
| 6 | `R3.STRATE.STB.N2STORAGECOUNT` | STB N2 저장수 (smartSTAR) N2Stocker로 이동중인 Queue 개수 포함, Down인 장비 제외 |
| 7 | `R3.STRATE.STB.NONN2STORAGECOUNT` | STB NONN2 저장수 (smartSTAR) |
| 8 | `R3.STRATE.STB.NONN2STORAGESERVICECAPA` | STB NONN2 저장가능용량 (smartSTAR) |

#### R3.STRATE.N2 — Storage — N2 (4건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `R3.STRATE.N2.STORAGECAPACITY` | N2 전체저장용량 (smartSTAR) N2Stocker로 이동중인 Queue 개수 포함, Down인 장비 제외 |
| 2 | `R3.STRATE.N2.STORAGECOUNT` | N2 저장수 (smartSTAR) N2Stocker로 이동중인 Queue 개수 포함, Down인 장비 제외 |
| 3 | `R3.STRATE.N2.STORAGERATIO` | N2 저장율 (smartSTAR) N2Stocker로 이동중인 Queue 개수 포함, Down인 장비 제외 |
| 4 | `R3.STRATE.N2.STORAGESERVICECAPA` | N2 저장가능용량 (smartSTAR) N2Stocker로 이동중인 Queue 개수 포함, Down인 장비 제외 |

#### R3.STRATE.NONN2 — Storage — NONN2 (4건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `R3.STRATE.NONN2.STORAGECAPACITY` | NONN2 전체저장용량 (smartSTAR) |
| 2 | `R3.STRATE.NONN2.STORAGECOUNT` | NONN2 저장수 (smartSTAR) |
| 3 | `R3.STRATE.NONN2.STORAGERATIO` | NONN2 저장율 (smartSTAR) |
| 4 | `R3.STRATE.NONN2.STORAGESERVICECAPA` | NONN2 저장가능용량 (smartSTAR) |

#### R3.AZFS.ALERT — AZFS 알람 (2건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `R3.AZFS.ALERT.DOUBLESTORAGE` | MCS가 Carrier DoubleStorage를 감지함(MCS) |
| 2 | `R3.AZFS.ALERT.SOURCEEMPTY` | SourceEmpty 발생 |

#### R3.INV.ALERT — INV 알람 (5건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `R3.INV.ALERT.INVCONNECTION` | MCS-INV간 통신이 연결(Message) |
| 2 | `R3.INV.ALERT.INVPAUSED` | INV MCP PAUSED로 State 변경 |
| 3 | `R3.INV.ALERT.INVAUTO` | INV MCP AUTO로 State 변경 |
| 4 | `R3.INV.ALERT.INVDISCONNECTION` | MCS-INV간 통신이 끊어짐(Message) |
| 5 | `R3.INV.ALERT.INVMCPALARM` | INVMCPAlarm 발생(Message) |

#### R3.INV.PC — INV PC (2건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `R3.INV.PC.PROS_STATUS` | not_active 개수가 5개이하 : green / 5개 초과+P/S상태값0 : yellow / 5개 초과+P/S상태값≠0 : red |
| 2 | `R3.INV.PC.DISK_USED` | 디스크사용율(%)(FTP) |

#### R3.STK.ALERT — STK 알람 (10건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `R3.STK.ALERT.STKCONNECTION` | MCS-STK간 통신이 연결(Message) |
| 2 | `R3.STK.ALERT.STKPAUSED` | STK MCP PAUSED로 State 변경 |
| 3 | `R3.STK.ALERT.STKAUTO` | STK MCP AUTO로 State 변경 |
| 4 | `R3.STK.ALERT.R3ESTKAUTO` | R3E STK AUTO 발생 |
| 5 | `R3.STK.ALERT.R3ESTKMCPALARM` | R3E STK MCP Alarm 발생 |
| 6 | `R3.STK.ALERT.R3ESTKCONNECTION` | R3E Stocker Connection 발생 |
| 7 | `R3.STK.ALERT.R3ESTKPAUSED` | R3E STK PAUSED 발생 |
| 8 | `R3.STK.ALERT.STKDISCONNECTION` | MCS-STK간 통신이 끊어짐(Message) |
| 9 | `R3.STK.ALERT.STKMCPALARM` | STKMCPAlarm 발생(Message) |
| 10 | `R3.STK.ALERT.R3ESTKDISCONNECTION` | R3E STK DisConnection 발생 |

#### R3.INPOS.ALERT — INPOS 알람 (7건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `R3.INPOS.ALERT.90003` | TraceMessage Disconnect |
| 2 | `R3.INPOS.ALERT.101` | I/O Read/Write Error |
| 3 | `R3.INPOS.ALERT.103` | MC OFF Error |
| 4 | `R3.INPOS.ALERT.104` | E-STOP |
| 5 | `R3.INPOS.ALERT.112` | Purge Not Started |
| 6 | `R3.INPOS.ALERT.113` | Main DC Power Error |
| 7 | `R3.INPOS.ALERT.114` | Main PW DC Power Error |

#### R3.ETC.ALERT — ETC 알람 (4건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `R3.ETC.ALERT.CARRIERDUPLICATE` | MCS가 동일 Carrier ID 중복 존재여부 감지시 발생(Message) |
| 2 | `R3.ETC.ALERT.TRANSACTIONTIMEISOVER` | MES명령에 대한 Transaction 처리가 5초 이상 1건 발생 시 알람 송신 |
| 3 | `R3.ETC.ALERT.DESTUNITNULL` | MCS에 장비 또는 포트가 등록 되어있지 않은 명령발생(Message) |
| 4 | `R3.ETC.ALERT.N2FOUPMISMATCHDEST` | N2 FOUP이 반송이상으로 인해 Normal STB로 반송되는 경우 알람 송신 |

#### R3.RTC.ALERT — RTC 알람 (1건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `R3.RTC.ALERT.RTCDISCONNECTION` | MCS-RTC간 통신이 끊어짐(Message) |

---

---

## 부록: 주요 영문 용어 번역 참고

| 영문 | 한글 | 비고 |
|------|------|------|
| Storage Ratio | 저장률 | % |
| Storage Count | 저장 수량 | FOUP 수 |
| Storage Max Capa | 최대 용량 | 전체 수용 |
| Storage Service Capa | 가용 용량 | 실사용 가능 |
| Exclude down machine | 다운장비 제외 |  |
| Transport Q Cnt | 반송 큐 개수 |  |
| Trans Time | 반송 시간 |  |
| OHT Usage | OHT 가동률 |  |
| CPU/Memory/Disk Usage | CPU/메모리/디스크 사용률 | FTP 수집 |
| Disconnect Communication | 통신 단절 |  |
| MCP AUTO/PAUSED | MCP 자동/일시정지 |  |
| E-STOP | 비상정지 |  |
| N2 Purge | N2 퍼지 | 질소 퍼지 영역 |
| STK | 스토커 | Stocker |
| STB | STB | Stocker Buffer |
| RETICLE | 레티클 | 포토마스크 |
| Cu Area | Cu 영역 | 구리 배선 공정 |

---
> 생성일: 2026-02-09 | 총 6,111건 전수 한글화 완료 (R3 144건 포함)
