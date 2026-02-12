# PNT4_TSV AMHS 모니터링 항목 (CPU/RAM 제외)

> 총 **134개** 항목

---

#### PNT4_TSV.ETC.ALERT — ETC 알람 (5건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `PNT4_TSV.ETC.ALERT.N2FOUPMISMATCHDEST` | N2 FOUP이 반송이상으로 인해 Normal STB로 반송되는 경우 알람 송신 |
| 2 | `PNT4_TSV.ETC.ALERT.CARRIERDUPLICATE` | MCS가 동일 Carrier ID 중복 존재여부 감지시 발생(Message) |
| 3 | `PNT4_TSV.ETC.ALERT.TRANSACTIONTIMEISOVER` | MES명령에 대한 Transaction 처리가 5초 이상 10건 발생 시 알람 송신 |
| 4 | `PNT4_TSV.ETC.ALERT.DESTUNITNULL` | MCS에 장비 또는 포트가 등록 되어있지 않은 명령발생(Message) |
| 5 | `PNT4_TSV.ETC.ALERT.FIOCONNECTION` | MCS-FIO간 통신이 연결(Message) |

#### PNT4_TSV.QUE.LOAD — Queue — Load 반송 (3건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `PNT4_TSV.QUE.LOAD.AVGLOADTIME` | 최근 10분간 평균 Load Time(HySTAR) |
| 2 | `PNT4_TSV.QUE.LOAD.CURRENTLOADQCNT` | 반송 큐 개수: 목적지가 생산장비인 건(MCS) |
| 3 | `PNT4_TSV.QUE.LOAD.AVGLOADTIME1MIN` | 최근 1분간 평균 Load반송시간(smartSTAR) |

#### PNT4_TSV.QUE.ALL — Queue — 전체 (3건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `PNT4_TSV.QUE.ALL.CURRENTQCREATED` | 최근 10분간 반송명령생성수(MCS) |
| 2 | `PNT4_TSV.QUE.ALL.CURRENTQCOMPLETED` | 최근10분간 반송완료 수(MCS) |
| 3 | `PNT4_TSV.QUE.ALL.CURRENTQCNT` | 현재 반송 QUEUE 수 |

#### PNT4_TSV.INPOS.ALERT — INPOS 알람 (7건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `PNT4_TSV.INPOS.ALERT.104` | E-STOP |
| 2 | `PNT4_TSV.INPOS.ALERT.90003` | TraceMessage Disconnect |
| 3 | `PNT4_TSV.INPOS.ALERT.101` | I/O Read/Write Error |
| 4 | `PNT4_TSV.INPOS.ALERT.103` | MC OFF Error |
| 5 | `PNT4_TSV.INPOS.ALERT.112` | Purge Not Started |
| 6 | `PNT4_TSV.INPOS.ALERT.113` | Main DC Power Error |
| 7 | `PNT4_TSV.INPOS.ALERT.114` | Main PW DC Power Error |

#### PNT4_TSV.OHT.ALERT — OHT 알람 (11건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `PNT4_TSV.OHT.ALERT.OHTMCPALARMCNT` | 현재 OHT Alarm 건수(MCSDB 기준) |
| 2 | `PNT4_TSV.OHT.ALERT.PIOCOMMERROR` | PIO Communication Error 발생(Message) |
| 3 | `PNT4_TSV.OHT.ALERT.OHTSTATUSERROR` | OHT상태가 Pausing으로 장시간 대기(MCS) |
| 4 | `PNT4_TSV.OHT.ALERT.OHTLOADFAIL` | OHT 반송중 TransferCompleted ResultCode (1)발생(Message) |
| 5 | `PNT4_TSV.OHT.ALERT.OHTDISCONNECTION` | MCS-OHT간 통신이 끊어짐(Message) |
| 6 | `PNT4_TSV.OHT.ALERT.OHTMCPALARM` | OHTMCPAlarm 발생(Message) |
| 7 | `PNT4_TSV.OHT.ALERT.OHTCONNECTION` | MCS-OHT간 통신이 연결(Message) |
| 8 | `PNT4_TSV.OHT.ALERT.OHTAUTO` | OHT MCP AUTO로 State 변경 |
| 9 | `PNT4_TSV.OHT.ALERT.OHTPAUSED` | OHT MCP PAUSED로 State 변경 |
| 10 | `PNT4_TSV.OHT.ALERT.OHTHIDALARM` | OHT HID Error 발생 |
| 11 | `PNT4_TSV.OHT.ALERT.OHTCANNOTEXECUTE` | OHT에 반송명령시 Reply (2) Currently not able to excute 발생(MCS) |

#### PNT4_TSV.MCS.ALERT — MCS 알람 (6건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `PNT4_TSV.MCS.ALERT.DBQUERYHANG` | No response the DB Query process |
| 2 | `PNT4_TSV.MCS.ALERT.PROCESSDOWN` | MCS에서 특정 Process의 비정상 Down을 감지(Message) |
| 3 | `PNT4_TSV.MCS.ALERT.PROCESSSHUTDOWNFAIL` | Process에 shutdown명령을 내렸으나 종료되지 않음(Message) |
| 4 | `PNT4_TSV.MCS.ALERT.PROCESSUPFAIL` | MCS내 컨트롤 서버가 Process를 StartUp시켰으나 제대로 살아나지 않을 경우(Message) |
| 5 | `PNT4_TSV.MCS.ALERT.PROCESSUPSTART` | MCS내 컨트롤 서버가 Process를 StartUp시도하는 경우(Message) |
| 6 | `PNT4_TSV.MCS.ALERT.PROCESSHANG` | MCS내 컨트롤 서버가 MCS에 Heartbeat메시지를 날렸으나 반응이 없을 경우 Hang으로 간주(Message) |

#### PNT4_TSV.FIO.ALERT — FIO 알람 (2건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `PNT4_TSV.FIO.ALERT.FIOMCPALARM` | FIOMCPAlarm 발생(Message) |
| 2 | `PNT4_TSV.FIO.ALERT.FIODISCONNECTION` | MCS-FIO간 통신이 끊어짐 (Message) |

#### PNT4_TSV.STRATE.FOUP — Storage — FOUP (4건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `PNT4_TSV.STRATE.FOUP.STORAGESERVICECAPA` | FOUP저장가능용량 |
| 2 | `PNT4_TSV.STRATE.FOUP.STORAGERATIO` | FOUP STORAGE 사용률 (%) |
| 3 | `PNT4_TSV.STRATE.FOUP.STORAGECAPACITY` | FOUP사용저장용량 |
| 4 | `PNT4_TSV.STRATE.FOUP.STORAGECOUNT` | FOUP저장수 |

#### PNT4_TSV.STRATE.MAGAZINE — Storage — MAGAZINE (4건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `PNT4_TSV.STRATE.MAGAZINE.STORAGESERVICECAPA` | MAGAZINE저장가능용량 |
| 2 | `PNT4_TSV.STRATE.MAGAZINE.STORAGERATIO` | MAGAZINE사용률(%) |
| 3 | `PNT4_TSV.STRATE.MAGAZINE.STORAGECAPACITY` | MAGAZINE 사용저장용량 |
| 4 | `PNT4_TSV.STRATE.MAGAZINE.STORAGECOUNT` | MAGAZINE저장수 |

#### PNT4_TSV.STRATE.RINGCST — Storage — RINGCST (4건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `PNT4_TSV.STRATE.RINGCST.STORAGERATIO` | RINGCST 사용률(%) |
| 2 | `PNT4_TSV.STRATE.RINGCST.STORAGECAPACITY` | RINGCST 사용저장용량 |
| 3 | `PNT4_TSV.STRATE.RINGCST.STORAGECOUNT` | RINGCST저장수 |
| 4 | `PNT4_TSV.STRATE.RINGCST.STORAGESERVICECAPA` | RINGCST저장가능용량 |

#### PNT4_TSV.MCS.TSVMCSDB01 — MCS DB01 서버 (1건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `PNT4_TSV.MCS.TSVMCSDB01.DISK_APP1` | 디스크사용율(%)(FTP) |

#### PNT4_TSV.MCS.TSVMCSDB02 — MCS DB02 서버 (1건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `PNT4_TSV.MCS.TSVMCSDB02.DISK_APP1` | 디스크사용율(%)(FTP) |

#### PNT4_TSV.INV.ALERT — INV 알람 (5건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `PNT4_TSV.INV.ALERT.INVDISCONNECTION` | MCS-INV간 통신이 끊어짐 (Message) |
| 2 | `PNT4_TSV.INV.ALERT.INVMCPALARM` | INVMCPAlarm 발생(Message) |
| 3 | `PNT4_TSV.INV.ALERT.INVAUTO` | INV MCP AUTO로 State 변경 |
| 4 | `PNT4_TSV.INV.ALERT.INVCONNECTION` | MCS-INV간 통신이 연결(Message) |
| 5 | `PNT4_TSV.INV.ALERT.INVPAUSED` | INV MCP PAUSED로 State 변경 |

#### PNT4_TSV.AZFS.ALERT — AZFS 알람 (2건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `PNT4_TSV.AZFS.ALERT.DOUBLESTORAGE` | MCS가 Carrier DoubleStorage를 감지함(MCS) |
| 2 | `PNT4_TSV.AZFS.ALERT.SOURCEEMPTY` | SourceEmpty 발생 |

#### PNT4_TSV.STRATE.N2 — Storage — N2 (3건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `PNT4_TSV.STRATE.N2.N2STORAGERATIO` | FAB내 저장장치(STK ZFS)의 보관율(MES) ReservedLocationCount 포함 |
| 2 | `PNT4_TSV.STRATE.N2.N2STORAGECAPACITY` | FAB내 저장장치(STK ZFS)의 보관저장용량(MES) ReservedLocationCount 포함 |
| 3 | `PNT4_TSV.STRATE.N2.N2STORAGECOUNT` | FAB내 저장장치(STK ZFS)의 보관수(MES) ReservedLocationCount 포함 |

#### PNT4_TSV.STRATE.ALL — Storage — FAB 전체 (4건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `PNT4_TSV.STRATE.ALL.FABSTORAGERATIO` | FAB내 저장장치(STK ZFS)의 보관율(MES) ReservedLocationCount 포함 |
| 2 | `PNT4_TSV.STRATE.ALL.FABSTORAGESERVICECAPA` | FAB 저장가능용량 (smartSTAR) ReservedLocationCount 포함 Down된 장비 제외 |
| 3 | `PNT4_TSV.STRATE.ALL.FABSTORAGECAPACITY` | FAB내 저장장치(STK ZFS)의 보관저장용량(MES) ReservedLocationCount 포함 |
| 4 | `PNT4_TSV.STRATE.ALL.FABSTORAGECOUNT` | FAB내 저장장치(STK ZFS)의 보관수(MES) ReservedLocationCount 포함 |

#### PNT4_TSV.QUE.OHT — Queue — OHT (2건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `PNT4_TSV.QUE.OHT.OHTUTIL` | OHT사용율(%)(MCS) (INSTALLED VHL 수 - NOT UNASSGINED VHL 수)/(INSTALLED VHL 수) |
| 2 | `PNT4_TSV.QUE.OHT.CURRENTOHTQCNT` | 현재 OHT반송 Q 수(MCS) |

#### PNT4_TSV.OHT.ALL — OHT 전체 (1건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `PNT4_TSV.OHT.ALL.CLUSTER` | 체크 대상 MCP의 Clustering 상태 점검(FTP) 체크대상 MCP == current : green MCP7 <> Online : red 그외 yellow |

#### PNT4_TSV.QUE.ABN — Queue — 이상 (4건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `PNT4_TSV.QUE.ABN.N2STOCKERDELAY` | 60분이상 N2Purge Stocker Port 내 FOUP 지연 발생 통지(MCS) |
| 2 | `PNT4_TSV.QUE.ABN.AOTRANSDELAYDET` | 10분이상 AO Port 반송 Delay 발생되는 FOUP 정보 통지(MCS) |
| 3 | `PNT4_TSV.QUE.ABN.AOTRANSDELAY` | 5분마다 쿼리 검색하여 10분이상 Auto Output Port에 대기하고 있는 FOUP이 하나이상 존재하면 FOUP 개수를 알람 |
| 4 | `PNT4_TSV.QUE.ABN.QUETIMEDELAY` | 10분 이상 지연된 FOUP 목록 알림(MCS) |

#### PNT4_TSV.QUE.ALERT — Queue — Bridge/알람 (11건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `PNT4_TSV.QUE.ALERT.BRIDGESTKDISCONNECTED` | MCS-Bridge Stocker간 통신이 끊어짐(Message) |
| 2 | `PNT4_TSV.QUE.ALERT.BRIDGECNVAUTO` | Bridge Conveyor AUTO로 State 변경 |
| 3 | `PNT4_TSV.QUE.ALERT.FABTRANSJOBDELAY` | Fab간 반송 JOB 장기간 미진행(미진행 시간 8044 Option) |
| 4 | `PNT4_TSV.QUE.ALERT.BRIDGESTKCONNECTED` | MCS-BRIDGE Stocker간 통신이 연결(Message) |
| 5 | `PNT4_TSV.QUE.ALERT.BRIDGESTKAUTO` | Bridge Stocker AUTO로 State 변경 |
| 6 | `PNT4_TSV.QUE.ALERT.BRIDGESTKPAUSED` | Bridge Stocker PAUSED로 State 변경 |
| 7 | `PNT4_TSV.QUE.ALERT.BRIDGECNVPAUSED` | Bridge Conveyor PAUSED로 State 변경 |
| 8 | `PNT4_TSV.QUE.ALERT.BRIDGEAOPORTTRANSDELAY` | Bridge 장비 Auto Out Port에 대기(대기시간 8016 Option) |
| 9 | `PNT4_TSV.QUE.ALERT.NOCLASSDEFFOUNDERROR` | 데몬이 정상 동작 하지 못함 |
| 10 | `PNT4_TSV.QUE.ALERT.BRIDGECNVDISCONNECTED` | MCS-BRIDGE Conveyor간 통신이 끊어짐(Message) |
| 11 | `PNT4_TSV.QUE.ALERT.BRIDGECNVCONNECTED` | MCS-BRIDGE Conveyor간 통신이 연결(Message) |

#### PNT4_TSV.OHT.ALARM — OHT 클러스터 알람 (1건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `PNT4_TSV.OHT.ALARM.CLUSTERSTATE` | OHT CLUSTER SERVER가 primary 또는 secondary로 전환되거나 Down/Up 상태일 때 알람 발송 |

#### PNT4_TSV.OHT.SERVER_P — OHT Server Primary (1건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `PNT4_TSV.OHT.SERVER_P.STATE` | 체크 대상 MCP의 Clustering 대상 Server 체크 1 = server1 2 = server2 3 = server down/up |

#### PNT4_TSV.OHT.SERVER_S — OHT Server Secondary (1건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `PNT4_TSV.OHT.SERVER_S.STATE` | 체크 대상 MCP의 Clustering 대상 Server 체크 1 = server1 2 = server2 3 = server down/up |

#### PNT4_TSV.LFT.SENDFAB — LFT FAB간 반송 (3건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `PNT4_TSV.LFT.SENDFAB.TO_PNT4_WT_CURRENTQCNT` | 현재 PNT4_TSV->PNT4_WT Q 수 |
| 2 | `PNT4_TSV.LFT.SENDFAB.TO_ICPKG_PNT4_1F_CURRENTQCNT` | 현재 PNT4_TSV->ICPKG_PNT4_1F Q 수 |
| 3 | `PNT4_TSV.LFT.SENDFAB.TO_ICPKG_PNT4_5F_CURRENTQCNT` | 현재 PNT4_TSV->ICPKG_PNT4_5F Q 수 |

#### PNT4_TSV.STRATE.NONN2 — Storage — NONN2 (3건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `PNT4_TSV.STRATE.NONN2.NONN2STORAGERATIO` | Non N2 Area Storage 저장율(EDB) |
| 2 | `PNT4_TSV.STRATE.NONN2.NONN2STORAGECAPACITY` | Non N2 Area Storage 저장용량(EDB) |
| 3 | `PNT4_TSV.STRATE.NONN2.NONN2STORAGECOUNT` | Non N2 Area Storage 저장수(EDB) |

#### PNT4_TSV.OHT.OHT_P — OHT Primary 서버 (5건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `PNT4_TSV.OHT.OHT_P.PROS_STATUS` | not_active 개수가 5개이하 : green / 5개 초과+P/S상태값0 : yellow / 5개 초과+P/S상태값≠0 : red |
| 2 | `PNT4_TSV.OHT.OHT_P.ASSIGNEDWAITTOTAL` | ASSIGNEDWAITTOTAL VALUE(FTP) |
| 3 | `PNT4_TSV.OHT.OHT_P.CLUSTER` | 체크 대상 MCP의 Clustering 상태 점검(FTP) 체크대상 MCP == current : green MCP7 <> Online : red 그외 yellow |
| 4 | `PNT4_TSV.OHT.OHT_P.DISK_USED` | 디스크사용율(%)(FTP) |
| 5 | `PNT4_TSV.OHT.OHT_P.SERVICE_STATUS` | CLPSTAT의 결과 MCP7 ............: Online 이 아닐 경우(FTP) |

#### PNT4_TSV.OHT.OHT_S — OHT Secondary 서버 (5건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `PNT4_TSV.OHT.OHT_S.PROS_STATUS` | not_active 개수가 5개이하 : green / 5개 초과+P/S상태값0 : yellow / 5개 초과+P/S상태값≠0 : red |
| 2 | `PNT4_TSV.OHT.OHT_S.ASSIGNEDWAITTOTAL` | ASSIGNEDWAITTOTAL VALUE(FTP) |
| 3 | `PNT4_TSV.OHT.OHT_S.SERVICE_STATUS` | CLPSTAT의 결과 MCP7 ............: Online 이 아닐 경우(FTP) |
| 4 | `PNT4_TSV.OHT.OHT_S.CLUSTER` | 체크 대상 MCP의 Clustering 상태 점검(FTP) 체크대상 MCP == current : green MCP7 <> Online : red 그외 yellow |
| 5 | `PNT4_TSV.OHT.OHT_S.DISK_USED` | 디스크사용율(%)(FTP) |

#### PNT4_TSV.INV.PC — INV PC (2건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `PNT4_TSV.INV.PC.PROS_STATUS` | not_active 개수가 5개이하 : green / 5개 초과+P/S상태값0 : yellow / 5개 초과+P/S상태값≠0 : red |
| 2 | `PNT4_TSV.INV.PC.DISK_USED` | 디스크사용율(%)(FTP) |

#### PNT4_TSV.STK.ALERT — STK 알람 (5건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `PNT4_TSV.STK.ALERT.STKDISCONNECTION` | MCS-STK간 통신이 끊어짐(Message) |
| 2 | `PNT4_TSV.STK.ALERT.STKMCPALARM` | STKMCPAlarm 발생(Message) |
| 3 | `PNT4_TSV.STK.ALERT.STKPAUSED` | STK MCP PAUSED로 State변경 |
| 4 | `PNT4_TSV.STK.ALERT.STKCONNECTION` | MCS-STK간 통신이 연결(Message) |
| 5 | `PNT4_TSV.STK.ALERT.STKAUTO` | STK MCP AUTO로 State 변경 |

#### PNT4_TSV.RTC.ALERT — RTC 알람 (1건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `PNT4_TSV.RTC.ALERT.RTCDISCONNECTION` | MCS-RTC간 통신이 끊어짐(Message) |

#### PNT4_TSV.STRATE.STK — Storage — STK (8건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `PNT4_TSV.STRATE.STK.NONN2FOUPSTORAGERATIO` | NONN2FOUP 저장장치(STK)의 보관율(MCS) ReservedLocationCount 포함 Down된 장비 제외 |
| 2 | `PNT4_TSV.STRATE.STK.NONN2FOUPSTORAGECOUNT` | NONN2FOUP 저장장치(STK)의 보관수(MCS) ReservedLocationCount 포함 Down된 장비 제외 |
| 3 | `PNT4_TSV.STRATE.STK.MAGAZINESTORAGERATIO` | MAGAZINE 저장장치(STK)의 보관율(MCS) ReservedLocationCount 포함 Down된 장비 제외 |
| 4 | `PNT4_TSV.STRATE.STK.MAGAZINESTORAGECOUNT` | MAGAZINE 저장장치(STK)의 보관수(MCS) ReservedLocationCount 포함 Down된 장비 제외 |
| 5 | `PNT4_TSV.STRATE.STK.MAGAZINESTORAGESERVICECAPA` | MAGAZINE 저장장치(STK)의 저장가능용량(MCS) ReservedLocationCount 포함 Down된 장비 제외 |
| 6 | `PNT4_TSV.STRATE.STK.MAGAZINESTORAGECAPACITY` | MAGAZINE 저장장치(STK)의 보관저장용량(MCS) ReservedLocationCount 포함 Down된 장비 제외 |
| 7 | `PNT4_TSV.STRATE.STK.NONN2FOUPSTORAGESERVICECAPA` | NONN2FOUP 저장장치(STK)의 저장가능용량(MCS) ReservedLocationCount 포함 Down된 장비 제외 |
| 8 | `PNT4_TSV.STRATE.STK.NONN2FOUPSTORAGECAPACITY` | NONN2FOUP 저장장치(STK)의 보관저장용량(MCS) ReservedLocationCount 포함 Down된 장비 제외 |

#### PNT4_TSV.STRATE.STB — Storage — STB (16건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `PNT4_TSV.STRATE.STB.NONN2FOUPSTORAGERATIO` | NONN2FOUP 저장장치(STB)의 보관율(MCS) ReservedLocationCount 포함 Down된 장비 제외 |
| 2 | `PNT4_TSV.STRATE.STB.NONN2FOUPSTORAGECOUNT` | NONN2FOUP 저장장치(STB)의 보관수(MCS) ReservedLocationCount 포함 Down된 장비 제외 |
| 3 | `PNT4_TSV.STRATE.STB.NONN2FOUPSTORAGESERVICECAPA` | NONN2FOUP 저장장치(STB)의 저장가능용량(MCS) ReservedLocationCount 포함 Down된 장비 제외 |
| 4 | `PNT4_TSV.STRATE.STB.NONN2FOUPSTORAGECAPACITY` | NONN2FOUP 저장장치(STB)의 보관저장용량(MCS) ReservedLocationCount 포함 Down된 장비 제외 |
| 5 | `PNT4_TSV.STRATE.STB.N2FOUPSTORAGERATIO` | N2FOUP 저장장치(STB)의 보관율(MCS) ReservedLocationCount 포함 Down된 장비 제외 |
| 6 | `PNT4_TSV.STRATE.STB.N2FOUPSTORAGECOUNT` | N2FOUP 저장장치(STB)의 보관수(MCS) ReservedLocationCount 포함 Down된 장비 제외 |
| 7 | `PNT4_TSV.STRATE.STB.N2FOUPSTORAGESERVICECAPA` | N2FOUP 저장장치(STB)의 저장가능용량(MCS) ReservedLocationCount 포함 Down된 장비 제외 |
| 8 | `PNT4_TSV.STRATE.STB.N2FOUPSTORAGECAPACITY` | N2FOUP 저장장치(STB)의 보관저장용량(MCS) ReservedLocationCount 포함 Down된 장비 제외 |
| 9 | `PNT4_TSV.STRATE.STB.MAGAZINESTORAGERATIO` | MAGAZINE 저장장치(STB)의 보관율(MCS) ReservedLocationCount 포함 Down된 장비 제외 |
| 10 | `PNT4_TSV.STRATE.STB.MAGAZINESTORAGECOUNT` | MAGAZINE 저장장치(STB)의 보관수(MCS) ReservedLocationCount 포함 Down된 장비 제외 |
| 11 | `PNT4_TSV.STRATE.STB.MAGAZINESTORAGESERVICECAPA` | MAGAZINE 저장장치(STB)의 저장가능용량(MCS) ReservedLocationCount 포함 Down된 장비 제외 |
| 12 | `PNT4_TSV.STRATE.STB.MAGAZINESTORAGECAPACITY` | MAGAZINE 저장장치(STB)의 보관저장용량(MCS) ReservedLocationCount 포함 Down된 장비 제외 |
| 13 | `PNT4_TSV.STRATE.STB.RINGCSTSTORAGERATIO` | RINGCST 저장장치(STB)의 보관율(MCS) ReservedLocationCount 포함 Down된 장비 제외 |
| 14 | `PNT4_TSV.STRATE.STB.RINGCSTSTORAGECOUNT` | RINGCST 저장장치(STB)의 보관수(MCS) ReservedLocationCount 포함 Down된 장비 제외 |
| 15 | `PNT4_TSV.STRATE.STB.RINGCSTSTORAGESERVICECAPA` | RINGCST 저장장치(STB)의 저장가능용량(MCS) ReservedLocationCount 포함 Down된 장비 제외 |
| 16 | `PNT4_TSV.STRATE.STB.RINGCSTSTORAGECAPACITY` | RINGCST 저장장치(STB)의 보관저장용량(MCS) ReservedLocationCount 포함 Down된 장비 제외 |