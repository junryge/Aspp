# QUWA FAB 컬럼 사전
> **구조**: `{FAB}.{Category}.{SubCategory}.{Metric}`
> **영문 Description → 한글 번역 완료**

---

## 7. PKG/PRB 계열

### CJPRB (204건)

#### CJPRB.MCS — 서버 인프라 (78건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `CJPRB.MCS.ALERT.PROCESSHANG` | MCS내 컨트롤 서버가 MCS에 Heartbeat메시지를 날렸으나 반응이 없을 경우 Hang으로 간주(Message) |
| 2 | `CJPRB.MCS.ALERT.PROCESSSHUTDOWNFAIL` | Process에 shutdown명령을 내렸으나 종료되지 않음(Message) |
| 3 | `CJPRB.MCS.ALERT.PROCESSUPFAIL` | MCS내 컨트롤 서버가 Process를 StartUp시켰으나 제대로 살아나지 않을 경우(Message) |
| 4 | `CJPRB.MCS.ALERT.PROCESSUPSTART` | MCS내 컨트롤 서버가 Process를 StartUp시도하는 경우(Message) |
| 5 | `CJPRB.MCS.CJPRBMCSA1.CPU_TOTAL` | 해당 Machine의 CPU 사용율(%)(FTP) |
| 6 | `CJPRB.MCS.CJPRBMCSA1.DISK_APP1` | 디스크사용율(%)(FTP) |
| 7 | `CJPRB.MCS.CJPRBMCSA1.MEM_FREE` | 해당 Machine의 Memory 여유공간(FTP) |
| 8 | `CJPRB.MCS.CJPRBMCSA1.MEM_USED` | 해당 Machine의 Memory사용율(%)(FTP) |
| 9 | `CJPRB.MCS.CJPRBMCSA1.PS_CS01_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 10 | `CJPRB.MCS.CJPRBMCSA1.PS_CS01_RSS` | 해당 Process의 실제 Memory 사용량(FTP) |
| 11 | `CJPRB.MCS.CJPRBMCSA1.PS_CS01_VSS` | 해당 Process의 가상 Memory 사용량(FTP) |
| 12 | `CJPRB.MCS.CJPRBMCSA1.PS_DS01_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 13 | `CJPRB.MCS.CJPRBMCSA1.PS_DS01_RSS` | 해당 Process의 실제 Memory 사용량(FTP) |
| 14 | `CJPRB.MCS.CJPRBMCSA1.PS_DS01_VSS` | 해당 Process의 가상 Memory 사용량(FTP) |
| 15 | `CJPRB.MCS.CJPRBMCSA1.PS_EI01_OHT_P_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 16 | `CJPRB.MCS.CJPRBMCSA1.PS_EI01_OHT_P_RSS` | 해당 Process의 실제 Memory 사용량(FTP) |
| 17 | `CJPRB.MCS.CJPRBMCSA1.PS_EI01_OHT_P_VSS` | 해당 Process의 가상 Memory 사용량(FTP) |
| 18 | `CJPRB.MCS.CJPRBMCSA1.PS_EI02_INV_P_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 19 | `CJPRB.MCS.CJPRBMCSA1.PS_EI02_INV_P_RSS` | 해당 Process의 실제 Memory 사용량(FTP) |
| 20 | `CJPRB.MCS.CJPRBMCSA1.PS_EI02_INV_P_VSS` | 해당 Process의 가상 Memory 사용량(FTP) |
| 21 | `CJPRB.MCS.CJPRBMCSA1.PS_EI03_STK_P_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 22 | `CJPRB.MCS.CJPRBMCSA1.PS_EI03_STK_P_RSS` | 해당 Process의 실제 Memory 사용량(FTP) |
| 23 | `CJPRB.MCS.CJPRBMCSA1.PS_EI03_STK_P_VSS` | 해당 Process의 가상 Memory 사용량(FTP) |
| 24 | `CJPRB.MCS.CJPRBMCSA1.PS_EI04_STK_P_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 25 | `CJPRB.MCS.CJPRBMCSA1.PS_EI04_STK_P_RSS` | 해당 Process의 실제 Memory 사용량(FTP) |
| 26 | `CJPRB.MCS.CJPRBMCSA1.PS_EI04_STK_P_VSS` | 해당 Process의 가상 Memory 사용량(FTP) |
| 27 | `CJPRB.MCS.CJPRBMCSA1.PS_EI05_FIO_P_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 28 | `CJPRB.MCS.CJPRBMCSA1.PS_EI05_FIO_P_RSS` | 해당 Process의 실제 Memory 사용량(FTP) |
| 29 | `CJPRB.MCS.CJPRBMCSA1.PS_EI05_FIO_P_VSS` | 해당 Process의 가상 Memory 사용량(FTP) |
| 30 | `CJPRB.MCS.CJPRBMCSA1.PS_TS01_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 31 | `CJPRB.MCS.CJPRBMCSA1.PS_TS01_RSS` | 해당 Process의 실제 Memory 사용량(FTP) |
| 32 | `CJPRB.MCS.CJPRBMCSA1.PS_TS01_VSS` | 해당 Process의 가상 Memory 사용량(FTP) |
| 33 | `CJPRB.MCS.CJPRBMCSA1.PS_TS02_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 34 | `CJPRB.MCS.CJPRBMCSA1.PS_TS02_RSS` | 해당 Process의 실제 Memory 사용량(FTP) |
| 35 | `CJPRB.MCS.CJPRBMCSA1.PS_TS02_VSS` | 해당 Process의 가상 Memory 사용량(FTP) |
| 36 | `CJPRB.MCS.CJPRBMCSA1.PS_TS03_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 37 | `CJPRB.MCS.CJPRBMCSA1.PS_TS03_RSS` | 해당 Process의 실제 Memory 사용량(FTP) |
| 38 | `CJPRB.MCS.CJPRBMCSA1.PS_TS03_VSS` | 해당 Process의 가상 Memory 사용량(FTP) |
| 39 | `CJPRB.MCS.CJPRBMCSA1.SWAP_IN` | Memory Swap In 발생여부(FTP) |
| 40 | `CJPRB.MCS.CJPRBMCSA1.SWAP_OUT` | 해당 Machine에서 Memory Swap Out이 발생한 회수(FTP) |
| 41 | `CJPRB.MCS.CJPRBMCSA2.CPU_TOTAL` | 해당 Machine의 CPU 사용율(%)(FTP) |
| 42 | `CJPRB.MCS.CJPRBMCSA2.DISK_APP1` | 디스크사용율(%)(FTP) |
| 43 | `CJPRB.MCS.CJPRBMCSA2.MEM_FREE` | 해당 Machine의 Memory 여유공간(FTP) |
| 44 | `CJPRB.MCS.CJPRBMCSA2.MEM_USED` | 해당 Machine의 Memory사용율(%)(FTP) |
| 45 | `CJPRB.MCS.CJPRBMCSA2.PS_CS02_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 46 | `CJPRB.MCS.CJPRBMCSA2.PS_CS02_RSS` | 해당 Process의 실제 Memory 사용량(FTP) |
| 47 | `CJPRB.MCS.CJPRBMCSA2.PS_CS02_VSS` | 해당 Process의 가상 Memory 사용량(FTP) |
| 48 | `CJPRB.MCS.CJPRBMCSA2.PS_DS02_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 49 | `CJPRB.MCS.CJPRBMCSA2.PS_DS02_RSS` | 해당 Process의 실제 Memory 사용량(FTP) |
| 50 | `CJPRB.MCS.CJPRBMCSA2.PS_DS02_VSS` | 해당 Process의 가상 Memory 사용량(FTP) |
| 51 | `CJPRB.MCS.CJPRBMCSA2.PS_EI01_OHT_S_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 52 | `CJPRB.MCS.CJPRBMCSA2.PS_EI01_OHT_S_RSS` | 해당 Process의 실제 Memory 사용량(FTP) |
| 53 | `CJPRB.MCS.CJPRBMCSA2.PS_EI01_OHT_S_VSS` | 해당 Process의 가상 Memory 사용량(FTP) |
| 54 | `CJPRB.MCS.CJPRBMCSA2.PS_EI02_INV_S_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 55 | `CJPRB.MCS.CJPRBMCSA2.PS_EI02_INV_S_RSS` | 해당 Process의 실제 Memory 사용량(FTP) |
| 56 | `CJPRB.MCS.CJPRBMCSA2.PS_EI02_INV_S_VSS` | 해당 Process의 가상 Memory 사용량(FTP) |
| 57 | `CJPRB.MCS.CJPRBMCSA2.PS_EI03_STK_S_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 58 | `CJPRB.MCS.CJPRBMCSA2.PS_EI03_STK_S_RSS` | 해당 Process의 실제 Memory 사용량(FTP) |
| 59 | `CJPRB.MCS.CJPRBMCSA2.PS_EI03_STK_S_VSS` | 해당 Process의 가상 Memory 사용량(FTP) |
| 60 | `CJPRB.MCS.CJPRBMCSA2.PS_EI04_STK_S_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 61 | `CJPRB.MCS.CJPRBMCSA2.PS_EI04_STK_S_RSS` | 해당 Process의 실제 Memory 사용량(FTP) |
| 62 | `CJPRB.MCS.CJPRBMCSA2.PS_EI04_STK_S_VSS` | 해당 Process의 가상 Memory 사용량(FTP) |
| 63 | `CJPRB.MCS.CJPRBMCSA2.PS_EI05_FIO_S_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 64 | `CJPRB.MCS.CJPRBMCSA2.PS_EI05_FIO_S_RSS` | 해당 Process의 실제 Memory 사용량(FTP) |
| 65 | `CJPRB.MCS.CJPRBMCSA2.PS_EI05_FIO_S_VSS` | 해당 Process의 가상 Memory 사용량(FTP) |
| 66 | `CJPRB.MCS.CJPRBMCSA2.PS_TS11_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 67 | `CJPRB.MCS.CJPRBMCSA2.PS_TS11_RSS` | 해당 Process의 실제 Memory 사용량(FTP) |
| 68 | `CJPRB.MCS.CJPRBMCSA2.PS_TS11_VSS` | 해당 Process의 가상 Memory 사용량(FTP) |
| 69 | `CJPRB.MCS.CJPRBMCSA2.PS_TS12_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 70 | `CJPRB.MCS.CJPRBMCSA2.PS_TS12_RSS` | 해당 Process의 실제 Memory 사용량(FTP) |
| 71 | `CJPRB.MCS.CJPRBMCSA2.PS_TS12_VSS` | 해당 Process의 가상 Memory 사용량(FTP) |
| 72 | `CJPRB.MCS.CJPRBMCSA2.PS_TS13_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 73 | `CJPRB.MCS.CJPRBMCSA2.PS_TS13_RSS` | 해당 Process의 실제 Memory 사용량(FTP) |
| 74 | `CJPRB.MCS.CJPRBMCSA2.PS_TS13_VSS` | 해당 Process의 가상 Memory 사용량(FTP) |
| 75 | `CJPRB.MCS.CJPRBMCSA2.SWAP_IN` | Memory Swap In 발생여부(FTP) |
| 76 | `CJPRB.MCS.CJPRBMCSA2.SWAP_OUT` | 해당 Machine에서 Memory Swap Out이 발생한 회수(FTP) |
| 77 | `CJPRB.MCS.ALERT.PROCESSDOWN` | MCS에서 특정 Process의 비정상 Down을 감지(Message) |
| 78 | `CJPRB.MCS.ALERT.DBQUERYHANG` | DB 쿼리 프로세스 무응답 |

#### CJPRB.OHT — OHT 반송 (68건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `CJPRB.OHT.ALARM.CLUSTERSTATE` | OHT CLUSTER SERVER가 primary 또는 secondary로 전환되거나 Down/Up 상태일 때 알람 발송 |
| 2 | `CJPRB.OHT.SERVER_P.STATE` | 체크 대상 MCP의 Clustering 대상 Server 체크 1 = server1 2 = server2 3 = server down/up |
| 3 | `CJPRB.OHT.SERVER_S.STATE` | 체크 대상 MCP의 Clustering 대상 Server 체크 1 = server1 2 = server2 3 = server down/up |
| 4 | `CJPRB.OHT.ALERT.OHTAUTO` | OHT MCP AUTO로 State 변 |
| 5 | `CJPRB.OHT.ALERT.OHTCANNOTEXECUTE` | OHT에 반송명령시 Reply (2) Currently not able to excute 발생(MCS) |
| 6 | `CJPRB.OHT.ALERT.OHTCONNECTION` | MCS-OHT간 통신이 연결(Message) |
| 7 | `CJPRB.OHT.ALERT.OHTDISCONNECTION` | MCS-OHT간 통신이 끊어짐(Message) |
| 8 | `CJPRB.OHT.ALERT.OHTHIDALARM` | OHT HID Error 발생 |
| 9 | `CJPRB.OHT.ALERT.OHTINVALIDPARA` | OHT에 반송 명령시 Reply (3)으로 Parameter 값이 다른 경우 |
| 10 | `CJPRB.OHT.ALERT.OHTLOADFAIL` | OHT 반송중 TransferCompleted ResultCode (1)발생(Message) |
| 11 | `CJPRB.OHT.ALERT.OHTMCPALARM` | OHTMCPAlarm 발생(Message) |
| 12 | `CJPRB.OHT.ALERT.OHTMCPALARMCNT` | 현재 OHT Alarm 건수(MCSDB 기준) |
| 13 | `CJPRB.OHT.ALERT.OHTPAUSED` | OHT MCP PAUSED로 State 변경 |
| 14 | `CJPRB.OHT.ALERT.OHTSTATUSERROR` | OHT상태가 Pausing으로 장시간 대기(MCS) |
| 15 | `CJPRB.OHT.ALERT.PIOCOMMERROR` | PIO Communication Error 발생(Message) |
| 16 | `CJPRB.OHT.ALL.CLUSTER` | 체크 대상 MCP의 Clustering 상태 점검(FTP) 체크대상 MCP == current : green MCP7 <> Online : red 그외 yellow |
| 17 | `CJPRB.OHT.ALL.MAX_CPU_TOTAL` | OHT의 Primary |
| 18 | `CJPRB.OHT.ALL.MAX_MEM_USED` | OHT의 Primary |
| 19 | `CJPRB.OHT.OHT_P.CLUSTER` | 체크 대상 MCP의 Clustering 상태 점검(FTP) 체크대상 MCP == current : green MCP7 <> Online : red 그외 yellow |
| 20 | `CJPRB.OHT.OHT_P.CPU_TOTAL` | 해당 Machine의 CPU 사용율(%)(FTP) |
| 21 | `CJPRB.OHT.OHT_P.DISK_USED` | 디스크사용율(%)(FTP) |
| 22 | `CJPRB.OHT.OHT_P.MEM_USED` | 해당 Machine의 Memory사용율(%)(FTP) |
| 23 | `CJPRB.OHT.OHT_P.PROS_STATUS` | not_active 개수가 5개이하 : green not_active 개수가 5개 초과이고, primary, secondary 모두 상태값이 0인게 하나 이상 : yellow not_active 개수가 5개 초과이고, primary, secondary 모두 상태값이 0인게 존재하지 않음 : red |
| 24 | `CJPRB.OHT.OHT_P.SERVICE_STATUS` | CLPSTAT의 결과 MCP7 ............: Online 이 아닐 경우(FTP) |
| 25 | `CJPRB.OHT.OHT_P.SWAP_USED` | SWAP Memeory 사용율(%)(FTP) |
| 26 | `CJPRB.OHT.OHT_S.CLUSTER` | 체크 대상 MCP의 Clustering 상태 점검(FTP) 체크대상 MCP == current : green MCP7 <> Online : red 그외 yellow |
| 27 | `CJPRB.OHT.OHT_S.CPU_TOTAL` | 해당 Machine의 CPU 사용율(%)(FTP) |
| 28 | `CJPRB.OHT.OHT_S.DISK_USED` | 디스크사용율(%)(FTP) |
| 29 | `CJPRB.OHT.OHT_S.MEM_USED` | 해당 Machine의 Memory사용율(%)(FTP) |
| 30 | `CJPRB.OHT.OHT_S.PROS_STATUS` | not_active 개수가 5개이하 : green not_active 개수가 5개 초과이고, primary, secondary 모두 상태값이 0인게 하나 이상 : yellow not_active 개수가 5개 초과이고, primary, secondary 모두 상태값이 0인게 존재하지 않음 : red |
| 31 | `CJPRB.OHT.OHT_S.SERVICE_STATUS` | CLPSTAT의 결과 MCP7 ............: Online 이 아닐 경우(FTP) |
| 32 | `CJPRB.OHT.OHT_S.SWAP_USED` | SWAP Memeory 사용율(%)(FTP) |
| 33 | `CJPRB.OHT.OHT_P.ASSIGNEDWAITTOTAL` | 배차 대기 총 수량(FTP) |
| 34 | `CJPRB.OHT.OHT_S.ASSIGNEDWAITTOTAL` | 배차 대기 총 수량(FTP) |
| 35 | `CJPRB.OHT.3F_OHT_P.CLUSTER` | 체크 대상 MCP의 Clustering 상태 점검(FTP) 체크대상 MCP == current : green MCP7 <> Online : red 그외 yellow |
| 36 | `CJPRB.OHT.3F_OHT_P.CPU_TOTAL` | 해당 Machine의 CPU 사용율(%)(FTP) |
| 37 | `CJPRB.OHT.3F_OHT_P.DISK_USED` | 디스크사용율(%)(FTP) |
| 38 | `CJPRB.OHT.3F_OHT_P.MEM_USED` | 해당 Machine의 Memory사용율(%)(FTP) |
| 39 | `CJPRB.OHT.3F_OHT_P.PROS_STATUS` | not_active 개수가 5개이하 : green not_active 개수가 5개 초과이고, primary, secondary 모두 상태값이 0인게 하나 이상 : yellow not_active 개수가 5개 초과이고, primary, secondary 모두 상태값이 0인게 존재하지 않음 : red |
| 40 | `CJPRB.OHT.3F_OHT_P.SERVICE_STATUS` | CLPSTAT의 결과 MCP7 ............: Online 이 아닐 경우(FTP) |
| 41 | `CJPRB.OHT.3F_OHT_P.SWAP_USED` | SWAP Memeory 사용율(%)(FTP) |
| 42 | `CJPRB.OHT.3F_OHT_S.CLUSTER` | 체크 대상 MCP의 Clustering 상태 점검(FTP) 체크대상 MCP == current : green MCP7 <> Online : red 그외 yellow |
| 43 | `CJPRB.OHT.3F_OHT_S.CPU_TOTAL` | 해당 Machine의 CPU 사용율(%)(FTP) |
| 44 | `CJPRB.OHT.3F_OHT_S.DISK_USED` | 디스크사용율(%)(FTP) |
| 45 | `CJPRB.OHT.3F_OHT_S.MEM_USED` | 해당 Machine의 Memory사용율(%)(FTP) |
| 46 | `CJPRB.OHT.3F_OHT_S.PROS_STATUS` | not_active 개수가 5개이하 : green not_active 개수가 5개 초과이고, primary, secondary 모두 상태값이 0인게 하나 이상 : yellow not_active 개수가 5개 초과이고, primary, secondary 모두 상태값이 0인게 존재하지 않음 : red |
| 47 | `CJPRB.OHT.3F_OHT_S.SERVICE_STATUS` | CLPSTAT의 결과 MCP7 ............: Online 이 아닐 경우(FTP) |
| 48 | `CJPRB.OHT.3F_OHT_S.SWAP_USED` | SWAP Memeory 사용율(%)(FTP) |
| 49 | `CJPRB.OHT.3F_SERVER_P.STATE` | 체크 대상 MCP의 Clustering 대상 Server 체크 1 = server1 2 = server2 3 = server down/up |
| 50 | `CJPRB.OHT.3F_SERVER_S.STATE` | 체크 대상 MCP의 Clustering 대상 Server 체크 1 = server1 2 = server2 3 = server down/up |
| 51 | `CJPRB.OHT.ALARM.3F_CLUSTERSTATE` | OHT CLUSTER SERVER가 primary 또는 secondary로 전환되거나 Down/Up 상태일 때 알람 발송 |
| 52 | `CJPRB.OHT.ALL.3F_CLUSTER` | 체크 대상 MCP의 Clustering 상태 점검(FTP) 체크대상 MCP == current : green MCP7 <> Online : red 그외 yellow |
| 53 | `CJPRB.OHT.ALL.3F_MAX_CPU_TOTAL` | OHT의 Primary |
| 54 | `CJPRB.OHT.ALL.3F_MAX_MEM_USED` | OHT의 Primary |
| 55 | `CJPRB.OHT.4F_OHT_P.CPU_TOTAL` | 해당 Machine의 CPU 사용율(%)(FTP) |
| 56 | `CJPRB.OHT.4F_OHT_P.MEM_USED` | 해당 Machine의 Memory사용율(%)(FTP) |
| 57 | `CJPRB.OHT.4F_OHT_P.PROS_STATUS` | not_active 개수가 5개이하 : green not_active 개수가 5개 초과이고, primary, secondary 모두 상태값이 0인게 하나 이상 : yellow not_active 개수가 5개 초과이고, primary, secondary 모두 상태값이 0인게 존재하지 않음 : red |
| 58 | `CJPRB.OHT.4F_OHT_S.SERVICE_STATUS` | CLPSTAT의 결과 MCP7 ............: Online 이 아닐 경우(FTP) |
| 59 | `CJPRB.OHT.4F_OHT_P.SERVICE_STATUS` | CLPSTAT의 결과 MCP7 ............: Online 이 아닐 경우(FTP) |
| 60 | `CJPRB.OHT.4F_OHT_S.CPU_TOTAL` | 해당 Machine의 CPU 사용율(%)(FTP) |
| 61 | `CJPRB.OHT.4F_OHT_S.MEM_USED` | 해당 Machine의 Memory사용율(%)(FTP) |
| 62 | `CJPRB.OHT.4F_OHT_S.PROS_STATUS` | not_active 개수가 5개이하 : green not_active 개수가 5개 초과이고, primary, secondary 모두 상태값이 0인게 하나 이상 : yellow not_active 개수가 5개 초과이고, primary, secondary 모두 상태값이 0인게 존재하지 않음 : red |
| 63 | `CJPRB.OHT.4F_OHT_S.SWAP_USED` | SWAP Memeory 사용율(%)(FTP) |
| 64 | `CJPRB.OHT.4F_OHT_P.SWAP_USED` | SWAP Memeory 사용율(%)(FTP) |
| 65 | `CJPRB.OHT.4F_OHT_P.CLUSTER` | 체크 대상 MCP의 Clustering 상태 점검(FTP) 체크대상 MCP == current : green MCP7 <> Online : red 그외 yellow |
| 66 | `CJPRB.OHT.4F_OHT_S.CLUSTER` | 체크 대상 MCP의 Clustering 상태 점검(FTP) 체크대상 MCP == current : green MCP7 <> Online : red 그외 yellow |
| 67 | `CJPRB.OHT.4F_OHT_S.DISK_USED` | 디스크사용율(%)(FTP) |
| 68 | `CJPRB.OHT.4F_OHT_P.DISK_USED` | 디스크사용율(%)(FTP) |

#### CJPRB.QUE — 반송 큐 (20건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `CJPRB.QUE.OHT.CURRENTOHTQCNT_2F` | 현재 OHT반송 Q 수(MCS) |
| 2 | `CJPRB.QUE.OHT.CURRENTOHTQCNT_3F` | 현재 OHT반송 Q 수(MCS) |
| 3 | `CJPRB.QUE.ABN.AOTRANSDELAY` | 5분마다 쿼리 검색하여 5분이상 Auto Output Port에 대기하고 있는 FOUP이 하나이상 존재하면 FOUP 개수를 알람 |
| 4 | `CJPRB.QUE.ABN.AOTRANSDELAYDET` | 10분이상 AO Port 반송 Delay 발생되는 FOUP 정보 통지(MCS) |
| 5 | `CJPRB.QUE.ALL.CURRENTQCNT` |  |
| 6 | `CJPRB.QUE.ALL.CURRENTQCOMPLETED` | 최근 10분간 반송명령 완료수(MCS) |
| 7 | `CJPRB.QUE.ALL.CURRENTQCREATED` | 최근10분간 Q생성 수(MCS) |
| 8 | `CJPRB.QUE.LOAD.AVGLOADTIME` | 최근 10분간 평균 Load반송시간(HySTAR) |
| 9 | `CJPRB.QUE.LOAD.AVGLOADTIME1MIN` | 최근 1분간 평균 Load반송시간(smartSTAR) |
| 10 | `CJPRB.QUE.LOAD.CURRENTLOADQCNT` | 반송 큐 개수: 목적지가 생산장비인 건(MCS) |
| 11 | `CJPRB.QUE.OHT.CURRENTOHTQCNT` | 현재 OHT반송 Q 수(MCS) |
| 12 | `CJPRB.QUE.OHT.OHTUTIL` | OHT사용율(%)(MCS) (INSTALLED VHL 수 - NOT UNASSGINED VHL 수)/(INSTALLED VHL 수) |
| 13 | `CJPRB.QUE.OHT.OHTUTIL_2F` | 2F OHT사용율(%)(MCS) (INSTALLED VHL 수 - NOT UNASSGINED VHL 수)/(INSTALLED VHL 수) |
| 14 | `CJPRB.QUE.OHT.OHTUTIL_3F` | 3F OHT사용율(%)(MCS) (INSTALLED VHL 수 - NOT UNASSGINED VHL 수)/(INSTALLED VHL 수) |
| 15 | `CJPRB.QUE.OHT.OHTUTIL_3F_FOUP` | 3F OHT FOUP 사용율(%)(MCS) (INSTALLED VHL 수 - NOT UNASSGINED VHL 수)/(INSTALLED VHL 수) |
| 16 | `CJPRB.QUE.OHT.OHTUTIL_3F_RINGCST` | 3F OHT RINGCST 사용율(%)(MCS) (INSTALLED VHL 수 - NOT UNASSGINED VHL 수)/(INSTALLED VHL 수) |
| 17 | `CJPRB.QUE.OHT.OHTUTIL_2F_FOUP` | 2F OHT FOUP사용율(%)(MCS) (INSTALLED VHL 수 - NOT UNASSGINED VHL 수)/(INSTALLED VHL 수) |
| 18 | `CJPRB.QUE.ABN.AOTRANSDELAYDET_LFT331_AO01` | 5분이상 LFT331 Port 반송 Delay 발생되는 FOUP 정보 통지(MCS) |
| 19 | `CJPRB.QUE.OHT.OHTUTIL_4F` | 4F OHT사용율(%)(MCS) (INSTALLED VHL 수 - NOT UNASSGINED VHL 수)/(INSTALLED VHL 수) |
| 20 | `CJPRB.QUE.OHT.CURRENTOHTQCNT_4F` | 현재 OHT반송 Q 수(MCS) |

#### CJPRB.INV — 인버터 (20건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `CJPRB.INV.ALERT.INVAUTO` | INV MCP AUTO로 State 변경 |
| 2 | `CJPRB.INV.ALERT.INVCONNECTION` | MCS-INV간 통신이 연결(Message) |
| 3 | `CJPRB.INV.ALERT.INVDISCONNECTION` | MCS-INV간 통신이 끊어짐 (Message) |
| 4 | `CJPRB.INV.ALERT.INVMCPALARM` | INVMCPAlarm 발생(Message) |
| 5 | `CJPRB.INV.ALERT.INVPAUSED` | INV MCP PAUSED로 State 변경 |
| 6 | `CJPRB.INV.PC.CPU_TOTAL` | 해당 Machine의 CPU 사용율(%)(FTP) |
| 7 | `CJPRB.INV.PC.DISK_USED` | 디스크사용율(%)(FTP) |
| 8 | `CJPRB.INV.PC.MEM_USED` | 해당 Machine의 Memory사용율(%)(FTP) |
| 9 | `CJPRB.INV.PC.SWAP_USED` | SWAP Memeory 사용율(%)(FTP) |
| 10 | `CJPRB.INV.3F_PC.CPU_TOTAL` | 해당 Machine의 CPU 사용율(%)(FTP) |
| 11 | `CJPRB.INV.3F_PC.DISK_USED` | 디스크사용율(%)(FTP) |
| 12 | `CJPRB.INV.3F_PC.MEM_USED` | 해당 Machine의 Memory사용율(%)(FTP) |
| 13 | `CJPRB.INV.3F_PC.SWAP_USED` | SWAP Memeory 사용율(%)(FTP) |
| 14 | `CJPRB.INV.4F_PC.CPU_TOTAL` | 해당 Machine의 CPU 사용율(%)(FTP) |
| 15 | `CJPRB.INV.4F_PC.DISK_USED` | 디스크사용율(%)(FTP) |
| 16 | `CJPRB.INV.4F_PC.MEM_USED` | 해당 Machine의 Memory사용율(%)(FTP) |
| 17 | `CJPRB.INV.4F_PC.SWAP_USED` | SWAP Memeory 사용율(%)(FTP) |
| 18 | `CJPRB.INV.4F_PC.PROS_STATUS` | not_active 개수가 5개이하 : green not_active 개수가 5개 초과이고, primary, secondary 모두 상태값이 0인게 하나 이상 : yellow not_active 개수가 5개 초과이고, primary, secondary 모두 상태값이 0인게 존재하지 않음 : red |
| 19 | `CJPRB.INV.PC.PROS_STATUS` | not_active 개수가 5개이하 : green not_active 개수가 5개 초과이고, primary, secondary 모두 상태값이 0인게 하나 이상 : yellow not_active 개수가 5개 초과이고, primary, secondary 모두 상태값이 0인게 존재하지 않음 : red |
| 20 | `CJPRB.INV.3F_PC.PROS_STATUS` | not_active 개수가 5개이하 : green not_active 개수가 5개 초과이고, primary, secondary 모두 상태값이 0인게 하나 이상 : yellow not_active 개수가 5개 초과이고, primary, secondary 모두 상태값이 0인게 존재하지 않음 : red |

#### CJPRB.ETC — 기타 (5건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `CJPRB.ETC.ALERT.N2FOUPMISMATCHDEST` | N2 FOUP이 반송이상으로 인해 Normal STB로 반송되는 경우 알람 송신 |
| 2 | `CJPRB.ETC.ALERT.CARRIERDUPLICATE` | MCS가 동일 Carrier ID 중복 존재여부 감지시 발생(Message) |
| 3 | `CJPRB.ETC.ALERT.DESTUNITNULL` | MCS에 장비 또는 포트가 등록 되어있지 않은 명령발생(Message) |
| 4 | `CJPRB.ETC.ALERT.FIOCONNECTION` | MCS-FIO간 통신이 연결(Message) |
| 5 | `CJPRB.ETC.ALERT.TRANSACTIONTIMEISOVER` | MES명령에 대한 Transaction 처리가 5초 이상 10건 발생 시 알람 송신 |

#### CJPRB.STK — 스토커 (5건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `CJPRB.STK.ALERT.STKAUTO` | STK MCP AUTO로 State 변경 |
| 2 | `CJPRB.STK.ALERT.STKCONNECTION` | MCS-STK간 통신이 연결(Message) |
| 3 | `CJPRB.STK.ALERT.STKDISCONNECTION` | MCS-STK간 통신이 끊어짐(Message) |
| 4 | `CJPRB.STK.ALERT.STKMCPALARM` | STKMCPAlarm 발생(Message) |
| 5 | `CJPRB.STK.ALERT.STKPAUSED` | STK MCP PAUSED로 State변경 |

#### CJPRB.STRATE — 저장 현황 (4건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `CJPRB.STRATE.ALL.FABSTORAGECAPACITY` | FAB 전체저장용량 (smartSTAR) ReservedLocationCount 포함, Down된 장비 제외 |
| 2 | `CJPRB.STRATE.ALL.FABSTORAGECOUNT` | FAB 저장수 (smartSTAR) ReservedLocationCount 포함, Down된 장비 제외 |
| 3 | `CJPRB.STRATE.ALL.FABSTORAGERATIO` | FAB 저장율 (smartSTAR) ReservedLocationCount 포함, Down된 장비 제외 |
| 4 | `CJPRB.STRATE.ALL.FABSTORAGESERVICECAPA` | FAB 저장가능용량 (smartSTAR) ReservedLocationCount 포함, Down된 장비 제외 |

#### CJPRB.AZFS — AZFS (2건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `CJPRB.AZFS.ALERT.DOUBLESTORAGE` | MCS가 Carrier DoubleStorage를 감지함(MCS) |
| 2 | `CJPRB.AZFS.ALERT.SOURCEEMPTY` | SourceEmpty 발생 |

#### CJPRB.FIO — FIO (2건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `CJPRB.FIO.ALERT.FIODISCONNECTION` | MCS-FIO간 통신이 끊어짐 (Message) |
| 2 | `CJPRB.FIO.ALERT.FIOMCPALARM` | FIOMCPAlarm 발생(Message) |

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
