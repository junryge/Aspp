# QUWA FAB 컬럼 사전
> **구조**: `{FAB}.{Category}.{SubCategory}.{Metric}`
> **영문 Description → 한글 번역 완료**

---

### ICPKG_PNT4_1F (90건)

#### ICPKG_PNT4_1F.OHT — OHT 반송 (29건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `ICPKG_PNT4_1F.OHT.ALERT.OHTMCPALARMCNT` | 현재 OHT Alarm 건수(MCSDB 기준) |
| 2 | `ICPKG_PNT4_1F.OHT.ALERT.PIOCOMMERROR` | PIO Communication Error 발생(Message) |
| 3 | `ICPKG_PNT4_1F.OHT.ALERT.OHTLOADFAIL` | OHT 반송중 TransferCompleted ResultCode (1)발생(Message) |
| 4 | `ICPKG_PNT4_1F.OHT.ALERT.OHTDISCONNECTION` | MCS-OHT간 통신이 끊어짐(Message) |
| 5 | `ICPKG_PNT4_1F.OHT.OHT_S.PROS_STATUS` | not_active 개수가 5개이하 : green not_active 개수가 5개 초과이고, primary, secondary 모두 상태값이 0인게 하나 이상 : yellow not_active 개수가 5개 초과이고, primary, secondary 모두 상태값이 0인게 존재하지 않음 : red |
| 6 | `ICPKG_PNT4_1F.OHT.OHT_P.MEM_USED` | 해당 Machine의 Memory사용율(%)(FTP) |
| 7 | `ICPKG_PNT4_1F.OHT.OHT_S.SERVICE_STATUS` | CLPSTAT의 결과 MCP7 ............: Online 이 아닐 경우(FTP) |
| 8 | `ICPKG_PNT4_1F.OHT.OHT_S.CLUSTER` | 체크 대상 MCP의 Clustering 상태 점검(FTP) 체크대상 MCP == current : green MCP7 <> Online : red 그외 yellow |
| 9 | `ICPKG_PNT4_1F.OHT.OHT_S.DISK_USED` | 디스크사용율(%)(FTP) |
| 10 | `ICPKG_PNT4_1F.OHT.OHT_P.SWAP_USED` | SWAP Memeory 사용율(%)(FTP) |
| 11 | `ICPKG_PNT4_1F.OHT.OHT_S.SWAP_USED` | SWAP Memeory 사용율(%)(FTP) |
| 12 | `ICPKG_PNT4_1F.OHT.OHT_S.CPU_TOTAL` | 해당 Machine의 CPU 사용율(%)(FTP) |
| 13 | `ICPKG_PNT4_1F.OHT.OHT_P.SERVICE_STATUS` | CLPSTAT의 결과 MCP7 ............: Online 이 아닐 경우(FTP) |
| 14 | `ICPKG_PNT4_1F.OHT.OHT_P.CPU_TOTAL` | 해당 Machine의 CPU 사용율(%)(FTP) |
| 15 | `ICPKG_PNT4_1F.OHT.SERVER_P.STATE` | 체크 대상 MCP의 Clustering 대상 Server 체크 1 = server1 2 = server2 3 = server down/up |
| 16 | `ICPKG_PNT4_1F.OHT.SERVER_S.STATE` | 체크 대상 MCP의 Clustering 대상 Server 체크 1 = server1 2 = server2 3 = server down/up |
| 17 | `ICPKG_PNT4_1F.OHT.OHT_P.ASSIGNEDWAITTOTAL` | 배차 대기 총 수량(FTP) |
| 18 | `ICPKG_PNT4_1F.OHT.OHT_S.ASSIGNEDWAITTOTAL` | 배차 대기 총 수량(FTP) |
| 19 | `ICPKG_PNT4_1F.OHT.OHT_P.DISK_USED` | 디스크사용율(%)(FTP) |
| 20 | `ICPKG_PNT4_1F.OHT.OHT_S.MEM_USED` | 해당 Machine의 Memory사용율(%)(FTP) |
| 21 | `ICPKG_PNT4_1F.OHT.ALERT.OHTMCPALARM` | OHTMCPAlarm 발생(Message) |
| 22 | `ICPKG_PNT4_1F.OHT.ALERT.OHTCONNECTION` | MCS-OHT간 통신이 연결(Message) |
| 23 | `ICPKG_PNT4_1F.OHT.ALERT.OHTAUTO` | OHT MCP AUTO로 State 변 |
| 24 | `ICPKG_PNT4_1F.OHT.ALERT.OHTPAUSED` | OHT MCP PAUSED로 State 변경 |
| 25 | `ICPKG_PNT4_1F.OHT.ALERT.OHTCANNOTEXECUTE` | OHT에 반송명령시 Reply (2) Currently not able to excute 발생(MCS) |
| 26 | `ICPKG_PNT4_1F.OHT.ALL.MAX_MEM_USED` | OHT의 Primary |
| 27 | `ICPKG_PNT4_1F.OHT.ALL.MAX_CPU_TOTAL` | OHT의 Primary |
| 28 | `ICPKG_PNT4_1F.OHT.OHT_P.CLUSTER` | 체크 대상 MCP의 Clustering 상태 점검(FTP) 체크대상 MCP == current : green MCP7 <> Online : red 그외 yellow |
| 29 | `ICPKG_PNT4_1F.OHT.OHT_P.PROS_STATUS` | not_active 개수가 5개이하 : green not_active 개수가 5개 초과이고, primary, secondary 모두 상태값이 0인게 하나 이상 : yellow not_active 개수가 5개 초과이고, primary, secondary 모두 상태값이 0인게 존재하지 않음 : red |

#### ICPKG_PNT4_1F.QUE — 반송 큐 (28건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `ICPKG_PNT4_1F.QUE.ABN.N2STOCKERDELAY` | 60분이상 N2Purge Stocker Port 내 FOUP 지연 발생 통지(MCS) |
| 2 | `ICPKG_PNT4_1F.QUE.LOAD.AVGLOADTIME` | 최근 10분간 평균 Load Time(HySTAR) |
| 3 | `ICPKG_PNT4_1F.QUE.ALL.CURRENTQCREATED` | 최근 10분간 반송명령생성수(MCS) |
| 4 | `ICPKG_PNT4_1F.QUE.ABN.AOTRANSDELAYDET` | 10분이상 AO Port 반송 Delay 발생되는 FOUP 정보 통지(MCS) |
| 5 | `ICPKG_PNT4_1F.QUE.OHT.CST270OHTUTIL` | CST(270도 선회) OHT사용율(%)(MCS) (INSTALLED VHL 수 - NOT UNASSGINED VHL 수)/(INSTALLED VHL 수) |
| 6 | `ICPKG_PNT4_1F.QUE.ALL.CURRENTQCOMPLETED` | 최근10분간 반송완료 수(MCS) |
| 7 | `ICPKG_PNT4_1F.QUE.ABN.AOTRANSDELAY` | 5분마다 쿼리 검색하여 10분이상 Auto Output Port에 대기하고 있는 FOUP이 하나이상 존재하면 FOUP 개수를 알람 |
| 8 | `ICPKG_PNT4_1F.QUE.LOAD.AVGLOADTIME1MIN` | 최근 1분간 평균 Load반송시간(smartSTAR) |
| 9 | `ICPKG_PNT4_1F.QUE.ALL.CURRENTQCNT` | 현재 반송 QUEUE 수 |
| 10 | `ICPKG_PNT4_1F.QUE.ABN.QUETIMEDELAY` | 10분 이상 지연된 FOUP 목록 알림(MCS) |
| 11 | `ICPKG_PNT4_1F.QUE.OHT.CSTOHTUTIL` | CST OHT사용율(%)(MCS) (INSTALLED VHL 수 - NOT UNASSGINED VHL 수)/(INSTALLED VHL 수) |
| 12 | `ICPKG_PNT4_1F.QUE.OHT.OHTUTIL` | OHT사용율(%)(MCS) (INSTALLED VHL 수 - NOT UNASSGINED VHL 수)/(INSTALLED VHL 수) |
| 13 | `ICPKG_PNT4_1F.QUE.ALERT.BRIDGESTKDISCONNECTED` | MCS-Bridge Stocker간 통신이 끊어짐(Message) |
| 14 | `ICPKG_PNT4_1F.QUE.ALERT.BRIDGESTKCONNECTED` | MCS-BRIDGE Stocker간 통신이 연결(Message) |
| 15 | `ICPKG_PNT4_1F.QUE.ALERT.BRIDGESTKAUTO` | Bridge Stocker AUTO로 State 변경 |
| 16 | `ICPKG_PNT4_1F.QUE.ALERT.BRIDGESTKPAUSED` | Bridge Stocker PAUSED로 State 변경 |
| 17 | `ICPKG_PNT4_1F.QUE.ALERT.BRIDGECNVDISCONNECTED` | MCS-BRIDGE Conveyor간 통신이 끊어짐(Message) |
| 18 | `ICPKG_PNT4_1F.QUE.ALERT.BRIDGECNVAUTO` | Bridge Conveyor AUTO로 State 변경 |
| 19 | `ICPKG_PNT4_1F.QUE.ALERT.BRIDGECNVPAUSED` | Bridge Conveyor PAUSED로 State 변경 |
| 20 | `ICPKG_PNT4_1F.QUE.ALERT.BRIDGEAOPORTTRANSDELAY` | Bridge 장비 Auto Out Port에 대기(대기시간 8016 Option) |
| 21 | `ICPKG_PNT4_1F.QUE.ALERT.NOCLASSDEFFOUNDERROR` | 데몬이 정상 동작 하지 못함 |
| 22 | `ICPKG_PNT4_1F.QUE.OHT.CURRENTOHTQCNT` | 현재 OHT반송 Q 수(MCS) |
| 23 | `ICPKG_PNT4_1F.QUE.LOAD.CURRENTLOADQCNT` | 반송 큐 개수: 목적지가 생산장비인 건(MCS) |
| 24 | `ICPKG_PNT4_1F.QUE.OHT.TRAYOHTUTIL` | TRAY OHT사용율(%)(MCS) (INSTALLED VHL 수 - NOT UNASSGINED VHL 수)/(INSTALLED VHL 수) |
| 25 | `ICPKG_PNT4_1F.QUE.OHT.FOSBOHTUTIL` | FOSB OHT사용율(%)(MCS) (INSTALLED VHL 수 - NOT UNASSGINED VHL 수)/(INSTALLED VHL 수) |
| 26 | `ICPKG_PNT4_1F.QUE.OHT.MGZOHTUTIL` | MGZ OHT사용율(%)(MCS) (INSTALLED VHL 수 - NOT UNASSGINED VHL 수)/(INSTALLED VHL 수) |
| 27 | `ICPKG_PNT4_1F.QUE.ALERT.FABTRANSJOBDELAY` | Fab간 반송 JOB 장기간 미진행(미진행 시간 8044 Option) |
| 28 | `ICPKG_PNT4_1F.QUE.ALERT.BRIDGECNVCONNECTED` | MCS-BRIDGE Conveyor간 통신이 연결(Message) |

#### ICPKG_PNT4_1F.STRATE — 저장 현황 (20건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `ICPKG_PNT4_1F.STRATE.ALL.FABSTORAGESERVICECAPA` | 저장장치(STK,STB)의 저장가능용량 (MCS) ReservedLocationCount 포함, Down된 장비 제외 |
| 2 | `ICPKG_PNT4_1F.STRATE.STK.STORAGECAPACITY` | 저장장치(STK)의 보관저장용량(MCS) ReservedLocationCount 포함, Down된 장비 제외 |
| 3 | `ICPKG_PNT4_1F.STRATE.STK.STORAGECOUNT` | 저장장치(STK)의 보관수(MCS) ReservedLocationCount 포함, Down된 장비 제외 |
| 4 | `ICPKG_PNT4_1F.STRATE.STK.STORAGESERVICECAPA` | 저장장치(STK)의 저장가능용량 (MCS) ReservedLocationCount 포함, Down된 장비 제외 |
| 5 | `ICPKG_PNT4_1F.STRATE.STB.STORAGERATIO` | 저장장치(STB)의 보관율(MCS) ReservedLocationCount 포함, Down된 장비 제외 |
| 6 | `ICPKG_PNT4_1F.STRATE.STB.STORAGECAPACITY` | 저장장치(STB)의 보관저장용량(MCS) ReservedLocationCount 포함, Down된 장비 제외 |
| 7 | `ICPKG_PNT4_1F.STRATE.STB.STORAGECOUNT` | 저장장치(STB)의 보관수(MCS) ReservedLocationCount 포함, Down된 장비 제외 |
| 8 | `ICPKG_PNT4_1F.STRATE.STB.STORAGESERVICECAPA` | 저장장치(STB)의 저장가능용량 (MCS) ReservedLocationCount 포함, Down된 장비 제외 |
| 9 | `ICPKG_PNT4_1F.STRATE.ALL.FABSTORAGECAPACITY` | 저장장치(STK,STB)의 보관저장용량(MCS) ReservedLocationCount 포함, Down된 장비 제외 |
| 10 | `ICPKG_PNT4_1F.STRATE.ALL.FABSTORAGECOUNT` | 저장장치(STK,STB)의 보관수(MCS) ReservedLocationCount 포함, Down된 장비 제외 |
| 11 | `ICPKG_PNT4_1F.STRATE.ALL.FABSTORAGERATIO` | 저장장치(STK,STB)의 보관율(MCS) ReservedLocationCount 포함, Down된 장비 제외 |
| 12 | `ICPKG_PNT4_1F.STRATE.STK.STORAGERATIO` | 저장장치(STK)의 보관율(MCS) ReservedLocationCount 포함, Down된 장비 제외 |
| 13 | `ICPKG_PNT4_1F.STRATE.STK.MAGAZINESTORAGERATIO` | MAGAZINE 저장장치(STK)의 보관율(MCS),ReservedLocationCount 포함,,Down된 장비 제외 |
| 14 | `ICPKG_PNT4_1F.STRATE.STK.MAGAZINESTORAGECOUNT` | MAGAZINE 저장장치(STK)의 보관수(MCS) ReservedLocationCount 포함 |
| 15 | `ICPKG_PNT4_1F.STRATE.STK.MAGAZINESTORAGESERVICECAPA` | MAGAZINE 저장장치(STK)의 저장가능용량 (MCS) ReservedLocationCount 포함 |
| 16 | `ICPKG_PNT4_1F.STRATE.STK.MAGAZINESTORAGECAPACITY` | MAGAZINE 저장장치(STK)의 보관저장용량(MCS) ReservedLocationCount 포함 |
| 17 | `ICPKG_PNT4_1F.STRATE.STB.MAGAZINESTORAGESERVICECAPA` | MAGAZINE 저장장치(STB)의 저장가능용량 (MCS) ReservedLocationCount 포함 |
| 18 | `ICPKG_PNT4_1F.STRATE.STB.MAGAZINESTORAGECOUNT` | MAGAZINE 저장장치(STB)의 보관수(MCS) ReservedLocationCount 포함 |
| 19 | `ICPKG_PNT4_1F.STRATE.STB.MAGAZINESTORAGECAPACITY` | MAGAZINE 저장장치(STB)의 보관저장용량(MCS) ReservedLocationCount 포함 |
| 20 | `ICPKG_PNT4_1F.STRATE.STB.MAGAZINESTORAGERATIO` | MAGAZINE 저장장치(STB)의 보관율(MCS) ReservedLocationCount 포함 |

#### ICPKG_PNT4_1F.MCS — 서버 인프라 (5건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `ICPKG_PNT4_1F.MCS.ALERT.PROCESSDOWN` | MCS에서 특정 Process의 비정상 Down을 감지(Message) |
| 2 | `ICPKG_PNT4_1F.MCS.ALERT.PROCESSSHUTDOWNFAIL` | Process에 shutdown명령을 내렸으나 종료되지 않음(Message) |
| 3 | `ICPKG_PNT4_1F.MCS.ALERT.PROCESSUPFAIL` | MCS내 컨트롤 서버가 Process를 StartUp시켰으나 제대로 살아나지 않을 경우(Message) |
| 4 | `ICPKG_PNT4_1F.MCS.ALERT.PROCESSUPSTART` | MCS내 컨트롤 서버가 Process를 StartUp시도하는 경우(Message) |
| 5 | `ICPKG_PNT4_1F.MCS.ALERT.PROCESSHANG` | MCS내 컨트롤 서버가 MCS에 Heartbeat메시지를 날렸으나 반응이 없을 경우 Hang으로 간주(Message) |

#### ICPKG_PNT4_1F.STK — 스토커 (5건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `ICPKG_PNT4_1F.STK.ALERT.STKDISCONNECTION` | MCS-STK간 통신이 끊어짐(Message) |
| 2 | `ICPKG_PNT4_1F.STK.ALERT.STKMCPALARM` | STKMCPAlarm 발생(Message) |
| 3 | `ICPKG_PNT4_1F.STK.ALERT.STKPAUSED` | STK MCP PAUSED로 State변경 |
| 4 | `ICPKG_PNT4_1F.STK.ALERT.STKCONNECTION` | MCS-STK간 통신이 연결(Message) |
| 5 | `ICPKG_PNT4_1F.STK.ALERT.STKAUTO` | STK MCP AUTO로 State 변경 |

#### ICPKG_PNT4_1F.LFT — 리프터 (2건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `ICPKG_PNT4_1F.LFT.SENDFAB.TO_ICPKG_PNT4_5F_CURRENTQCNT` | 현재 ICPKG_PNT4_1F->ICPKG_PNT4_5F Q 수 |
| 2 | `ICPKG_PNT4_1F.LFT.SENDFAB.TO_PNT4_TSV_CURRENTQCNT` | 현재 ICPKG_PNT4_1F->PNT4_TSV Q 수 |

#### ICPKG_PNT4_1F.ETC — 기타 (1건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `ICPKG_PNT4_1F.ETC.ALERT.CARRIERDUPLICATE` | MCS가 동일 Carrier ID 중복 존재여부 감지시 발생(Message) |

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
