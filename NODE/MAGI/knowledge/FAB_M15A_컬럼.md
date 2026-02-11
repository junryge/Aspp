# QUWA FAB 컬럼 사전
> **구조**: `{FAB}.{Category}.{SubCategory}.{Metric}`
> **영문 Description → 한글 번역 완료**

---

### M15A (257건)

#### M15A.MCS — 서버 인프라 (99건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `M15A.MCS.ALERT.PROCESSUPSTART` | MCS내 컨트롤 서버가 Process를 StartUp시도하는 경우(Message) |
| 2 | `M15A.MCS.M15AMCSA1.CPU_TOTAL` | 해당 Machine의 CPU 사용율(%)(FTP) |
| 3 | `M15A.MCS.M15AMCSA1.DISK_APP1` | 디스크사용율(%)(FTP) |
| 4 | `M15A.MCS.M15AMCSA1.MEM_FREE` | 해당 Machine의 Memory 여유공간(FTP) |
| 5 | `M15A.MCS.M15AMCSA1.MEM_USED` | 해당 Machine의 Memory사용율(%)(FTP) |
| 6 | `M15A.MCS.M15AMCSA1.PS_CS01_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 7 | `M15A.MCS.M15AMCSA1.PS_CS01_RSS` | 해당 Process의 CPU 사용율(FTP) |
| 8 | `M15A.MCS.M15AMCSA1.PS_CS01_VSS` | 해당 Process의 CPU 사용율(FTP) |
| 9 | `M15A.MCS.M15AMCSA1.PS_DS01_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 10 | `M15A.MCS.M15AMCSA1.PS_DS01_RSS` | 해당 Process의 CPU 사용율(FTP) |
| 11 | `M15A.MCS.M15AMCSA1.PS_DS01_VSS` | 해당 Process의 CPU 사용율(FTP) |
| 12 | `M15A.MCS.M15AMCSA1.PS_EI01_OHT_P_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 13 | `M15A.MCS.M15AMCSA1.PS_EI01_OHT_P_RSS` | 해당 Process의 CPU 사용율(FTP) |
| 14 | `M15A.MCS.M15AMCSA1.PS_EI01_OHT_P_VSS` | 해당 Process의 CPU 사용율(FTP) |
| 15 | `M15A.MCS.M15AMCSA1.PS_EI02_INV_P_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 16 | `M15A.MCS.M15AMCSA1.PS_EI02_INV_P_RSS` | 해당 Process의 CPU 사용율(FTP) |
| 17 | `M15A.MCS.M15AMCSA1.PS_EI02_INV_P_VSS` | 해당 Process의 CPU 사용율(FTP) |
| 18 | `M15A.MCS.M15AMCSA1.PS_EI03_STK_P_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 19 | `M15A.MCS.M15AMCSA1.PS_EI03_STK_P_RSS` | 해당 Process의 CPU 사용율(FTP) |
| 20 | `M15A.MCS.M15AMCSA1.PS_EI03_STK_P_VSS` | 해당 Process의 CPU 사용율(FTP) |
| 21 | `M15A.MCS.M15AMCSA1.PS_EI04_STK_P_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 22 | `M15A.MCS.M15AMCSA1.PS_EI04_STK_P_VSS` | 해당 Process의 CPU 사용율(FTP) |
| 23 | `M15A.MCS.M15AMCSA1.PS_EI05_RTC_P_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 24 | `M15A.MCS.M15AMCSA1.PS_EI05_RTC_P_RSS` | 해당 Process의 CPU 사용율(FTP) |
| 25 | `M15A.MCS.M15AMCSA1.PS_EI05_RTC_P_VSS` | 해당 Process의 CPU 사용율(FTP) |
| 26 | `M15A.MCS.M15AMCSA1.PS_EI06_STK_P_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 27 | `M15A.MCS.M15AMCSA1.PS_EI06_STK_P_RSS` | 해당 Process의 CPU 사용율(FTP) |
| 28 | `M15A.MCS.M15AMCSA1.PS_EI06_STK_P_VSS` | 해당 Process의 CPU 사용율(FTP) |
| 29 | `M15A.MCS.M15AMCSA1.PS_TS01_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 30 | `M15A.MCS.M15AMCSA1.PS_TS01_RSS` | 해당 Process의 CPU 사용율(FTP) |
| 31 | `M15A.MCS.M15AMCSA1.PS_TS01_VSS` | 해당 Process의 CPU 사용율(FTP) |
| 32 | `M15A.MCS.M15AMCSA1.PS_TS02_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 33 | `M15A.MCS.M15AMCSA1.PS_TS02_RSS` | 해당 Process의 CPU 사용율(FTP) |
| 34 | `M15A.MCS.M15AMCSA1.PS_TS02_VSS` | 해당 Process의 CPU 사용율(FTP) |
| 35 | `M15A.MCS.ALERT.DBQUERYHANG` | DB 쿼리 프로세스 무응답 |
| 36 | `M15A.MCS.ALERT.PROCESSDOWN` | MCS에서 특정 Process의 비정상 Down을 감지(Message) |
| 37 | `M15A.MCS.ALERT.PROCESSHANG` | MCS내 컨트롤 서버가 MCS에 Heartbeat메시지를 날렸으나 반응이 없을 경우 Hang으로 간주(Message) |
| 38 | `M15A.MCS.ALERT.PROCESSSHUTDOWNFAIL` | Process에 shutdown명령을 내렸으나 종료되지 않음(Message) |
| 39 | `M15A.MCS.ALERT.PROCESSUPFAIL` | MCS내 컨트롤 서버가 Process를 StartUp시켰으나 제대로 살아나지 않을 경우(Message) |
| 40 | `M15A.MCS.M15AMCSA1.PS_TS03_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 41 | `M15A.MCS.M15AMCSA1.PS_TS03_RSS` | 해당 Process의 CPU 사용율(FTP) |
| 42 | `M15A.MCS.M15AMCSA1.PS_TS03_VSS` | 해당 Process의 CPU 사용율(FTP) |
| 43 | `M15A.MCS.M15AMCSA1.SWAP_IN` | Memory Swap In 발생여부(FTP) |
| 44 | `M15A.MCS.M15AMCSA1.SWAP_OUT` | 해당 Machine에서 Memory Swap Out이 발생한 회수(FTP) |
| 45 | `M15A.MCS.M15AMCSA2.CPU_TOTAL` | 해당 Machine의 CPU 사용율(%)(FTP) |
| 46 | `M15A.MCS.M15AMCSA2.DISK_APP1` | 디스크사용율(%)(FTP) |
| 47 | `M15A.MCS.M15AMCSA2.MEM_FREE` | 해당 Machine의 Memory 여유공간(FTP) |
| 48 | `M15A.MCS.M15AMCSA2.MEM_USED` | 해당 Machine의 Memory사용율(%)(FTP) |
| 49 | `M15A.MCS.M15AMCSA2.PS_CS02_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 50 | `M15A.MCS.M15AMCSA2.PS_CS02_RSS` | 해당 Process의 CPU 사용율(FTP) |
| 51 | `M15A.MCS.M15AMCSA2.PS_CS02_VSS` | 해당 Process의 CPU 사용율(FTP) |
| 52 | `M15A.MCS.M15AMCSA2.PS_EI01_OHT_S_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 53 | `M15A.MCS.M15AMCSA2.PS_EI01_OHT_S_RSS` | 해당 Process의 CPU 사용율(FTP) |
| 54 | `M15A.MCS.M15AMCSA2.PS_EI01_OHT_S_VSS` | 해당 Process의 CPU 사용율(FTP) |
| 55 | `M15A.MCS.M15AMCSA2.PS_EI02_INV_S_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 56 | `M15A.MCS.M15AMCSA2.PS_EI02_INV_S_RSS` | 해당 Process의 CPU 사용율(FTP) |
| 57 | `M15A.MCS.M15AMCSA2.PS_EI02_INV_S_VSS` | 해당 Process의 CPU 사용율(FTP) |
| 58 | `M15A.MCS.M15AMCSA2.PS_EI03_STK_S_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 59 | `M15A.MCS.M15AMCSA2.PS_EI03_STK_S_RSS` | 해당 Process의 CPU 사용율(FTP) |
| 60 | `M15A.MCS.M15AMCSA2.PS_EI03_STK_S_VSS` | 해당 Process의 CPU 사용율(FTP) |
| 61 | `M15A.MCS.M15AMCSA2.PS_EI04_STK_S_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 62 | `M15A.MCS.M15AMCSA2.PS_EI04_STK_S_RSS` | 해당 Process의 CPU 사용율(FTP) |
| 63 | `M15A.MCS.M15AMCSA2.PS_EI04_STK_S_VSS` | 해당 Process의 CPU 사용율(FTP) |
| 64 | `M15A.MCS.M15AMCSA2.PS_EI05_RTC_S_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 65 | `M15A.MCS.M15AMCSA2.PS_EI05_RTC_S_RSS` | 해당 Process의 CPU 사용율(FTP) |
| 66 | `M15A.MCS.M15AMCSA2.PS_EI05_RTC_S_VSS` | 해당 Process의 CPU 사용율(FTP) |
| 67 | `M15A.MCS.M15AMCSA2.PS_EI06_STK_S_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 68 | `M15A.MCS.M15AMCSA2.PS_EI06_STK_S_RSS` | 해당 Process의 CPU 사용율(FTP) |
| 69 | `M15A.MCS.M15AMCSA2.PS_EI06_STK_S_VSS` | 해당 Process의 CPU 사용율(FTP) |
| 70 | `M15A.MCS.M15AMCSA2.PS_TS11_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 71 | `M15A.MCS.M15AMCSA2.PS_TS11_RSS` | 해당 Process의 CPU 사용율(FTP) |
| 72 | `M15A.MCS.M15AMCSA2.PS_TS11_VSS` | 해당 Process의 CPU 사용율(FTP) |
| 73 | `M15A.MCS.M15AMCSA2.PS_TS12_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 74 | `M15A.MCS.M15AMCSA2.PS_TS12_RSS` | 해당 Process의 CPU 사용율(FTP) |
| 75 | `M15A.MCS.M15AMCSA2.PS_TS12_VSS` | 해당 Process의 CPU 사용율(FTP) |
| 76 | `M15A.MCS.M15AMCSA2.PS_TS13_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 77 | `M15A.MCS.M15AMCSA2.PS_TS13_RSS` | 해당 Process의 CPU 사용율(FTP) |
| 78 | `M15A.MCS.M15AMCSA2.PS_TS13_VSS` | 해당 Process의 CPU 사용율(FTP) |
| 79 | `M15A.MCS.M15AMCSA2.SWAP_IN` | Memory Swap In 발생여부(FTP) |
| 80 | `M15A.MCS.M15AMCSA2.SWAP_OUT` | 해당 Machine에서 Memory Swap Out이 발생한 회수(FTP) |
| 81 | `M15A.MCS.M15AMCSD1.CPU_TOTAL` | 해당 Machine의 CPU 사용율(%)(FTP) |
| 82 | `M15A.MCS.M15AMCSD1.DISK_APP1` | 디스크사용율(%)(FTP) |
| 83 | `M15A.MCS.M15AMCSD1.DISK_APP2` | 디스크사용율(%)(FTP) |
| 84 | `M15A.MCS.M15AMCSD1.DISK_USED` | 디스크사용율(%)(FTP) |
| 85 | `M15A.MCS.M15AMCSD1.MEM_FREE` | 해당 Machine의 Memory 여유공간(FTP) |
| 86 | `M15A.MCS.M15AMCSD1.MEM_USED` | 해당 Machine의 Memory사용율(%)(FTP) |
| 87 | `M15A.MCS.M15AMCSD1.SESSION_CNT` | DB Session수(FTP)oracleM15AMCS1의 프로세스개수를 집계하여 세션의 개수를 파악함 |
| 88 | `M15A.MCS.M15AMCSD1.SWAP_IN` | Memory Swap In 발생여부(FTP) |
| 89 | `M15A.MCS.M15AMCSD1.SWAP_OUT` | 해당 Machine에서 Memory Swap Out이 발생한 회수(FTP) |
| 90 | `M15A.MCS.M15AMCSD2.CPU_TOTAL` | 해당 Machine의 CPU 사용율(%)(FTP) |
| 91 | `M15A.MCS.M15AMCSD2.DISK_APP1` | 디스크사용율(%)(FTP) |
| 92 | `M15A.MCS.M15AMCSD2.DISK_APP2` | 디스크사용율(%)(FTP) |
| 93 | `M15A.MCS.M15AMCSD2.DISK_USED` | 디스크사용율(%)(FTP) |
| 94 | `M15A.MCS.M15AMCSD2.MEM_FREE` | 해당 Machine의 Memory 여유공간(FTP) |
| 95 | `M15A.MCS.M15AMCSD2.MEM_USED` | 해당 Machine의 Memory사용율(%)(FTP) |
| 96 | `M15A.MCS.M15AMCSD2.SESSION_CNT` | DB Session수(FTP)oracleM15AMCS2의 프로세스개수를 집계하여 세션의 개수를 파악함 |
| 97 | `M15A.MCS.M15AMCSD2.SWAP_IN` | Memory Swap In 발생여부(FTP) |
| 98 | `M15A.MCS.M15AMCSD2.SWAP_OUT` | 해당 Machine에서 Memory Swap Out이 발생한 회수(FTP) |
| 99 | `M15A.MCS.M15AMCSA1.PS_EI04_STK_P_RSS` | 해당 Process의 CPU 사용율(FTP) |

#### M15A.STRATE — 저장 현황 (60건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `M15A.STRATE.STK.NONN2STORAGERATIO` | NONN2STK 저장율 (smartSTAR)NON N2Stocker로 이동중인 Queue 개수 포함, Down인 장비 제외 |
| 2 | `M15A.STRATE.STK.NONN2STORAGECAPACITY` | NONN2STK 전체저장용량 (smartSTAR) |
| 3 | `M15A.STRATE.STK.NONN2STORAGECOUNT` | NONN2STK 저장수 (smartSTAR) |
| 4 | `M15A.STRATE.STB.NONN2STORAGECAPACITY` | NONN2STB 전체저장용량 (smartSTAR) |
| 5 | `M15A.STRATE.STB.NONN2STORAGECOUNT` | NONN2STB 저장수 (smartSTAR) |
| 6 | `M15A.STRATE.STK.NONN2STORAGESERVICECAPA` | NONN2STK 저장가능용량 (smartSTAR) |
| 7 | `M15A.STRATE.ALL.FABSTORAGECAPACITY` | FAB 전체저장용량 (smartSTAR) ReservedLocationCount 포함, Down된 장비 제외 |
| 8 | `M15A.STRATE.ALL.FABSTORAGECOUNT` | FAB 저장수 (smartSTAR) ReservedLocationCount 포함, Down된 장비 제외 |
| 9 | `M15A.STRATE.ALL.FABSTORAGERATIO` | FAB 저장율 (smartSTAR) ReservedLocationCount 포함, Down된 장비 제외 |
| 10 | `M15A.STRATE.ALL.FABSTORAGESERVICECAPA` | FAB 저장가능용량 (smartSTAR) ReservedLocationCount 포함, Down된 장비 제외 |
| 11 | `M15A.STRATE.CU.N2STORAGECAPACITY` | CU N2 전체저장용량 (smartSTAR) |
| 12 | `M15A.STRATE.CU.N2STORAGECOUNT` | CU N2 저장수 (smartSTAR) |
| 13 | `M15A.STRATE.CU.N2STORAGERATIO` | CU N2 저장율 (smartSTAR) |
| 14 | `M15A.STRATE.CU.N2STORAGESERVICECAPA` | CU N2 저장가능용량 (smartSTAR) |
| 15 | `M15A.STRATE.CU.NONN2STORAGECAPACITY` | CU NONN2 전체저장용량 (smartSTAR) |
| 16 | `M15A.STRATE.CU.NONN2STORAGECOUNT` | CU NONN2 저장수 (smartSTAR) |
| 17 | `M15A.STRATE.CU.NONN2STORAGERATIO` | CU NONN2 저장율 (smartSTAR) |
| 18 | `M15A.STRATE.CU.NONN2STORAGESERVICECAPA` | CU NONN2 저장가능용량 (smartSTAR) |
| 19 | `M15A.STRATE.CU.STORAGECAPACITY` | CU 전체저장용량 (smartSTAR) |
| 20 | `M15A.STRATE.CU.STORAGECOUNT` | CU 저장수 (smartSTAR) |
| 21 | `M15A.STRATE.CU.STORAGERATIO` | CU 저장율 (smartSTAR) |
| 22 | `M15A.STRATE.CU.STORAGESERVICECAPA` | CU 저장가능용량 (smartSTAR) |
| 23 | `M15A.STRATE.N2.STORAGECAPACITY` | N2 전체저장용량 (smartSTAR) |
| 24 | `M15A.STRATE.N2.STORAGECOUNT` | N2 저장수 (smartSTAR) |
| 25 | `M15A.STRATE.N2.STORAGERATIO` | N2 저장율 (smartSTAR) N2 STK로 이동중인 Queue 개수 포함, Down인 장비 제외 |
| 26 | `M15A.STRATE.N2.STORAGESERVICECAPA` | N2 저장가능용량 (smartSTAR) |
| 27 | `M15A.STRATE.NONCU.N2STORAGECAPACITY` | NONCU N2 전체저장용량 (smartSTAR) |
| 28 | `M15A.STRATE.NONCU.N2STORAGECOUNT` | NONCU N2 저장수 (smartSTAR) |
| 29 | `M15A.STRATE.NONCU.N2STORAGERATIO` | NONCU N2 저장율 (smartSTAR) |
| 30 | `M15A.STRATE.NONCU.N2STORAGESERVICECAPA` | NONCU N2 저장가능용량 (smartSTAR) |
| 31 | `M15A.STRATE.NONCU.NONN2STORAGECAPACITY` | NONCU NONN2 전체저장용량 (smartSTAR) |
| 32 | `M15A.STRATE.NONCU.NONN2STORAGECOUNT` | NONCU NONN2 저장수 (smartSTAR) |
| 33 | `M15A.STRATE.NONCU.NONN2STORAGERATIO` | NONCU NONN2 저장율 (smartSTAR) |
| 34 | `M15A.STRATE.NONCU.NONN2STORAGESERVICECAPA` | NONCU NONN2 저장가능용량 (smartSTAR) |
| 35 | `M15A.STRATE.NONCU.STORAGECAPACITY` | NONCU 전체저장용량 (smartSTAR) |
| 36 | `M15A.STRATE.NONCU.STORAGECOUNT` | NONCU 저장수 (smartSTAR) |
| 37 | `M15A.STRATE.NONCU.STORAGERATIO` | NONCU 저장율 (smartSTAR) NONCU Storage로 이동중인 Queue 개수 포함, Down인 장비 제외 |
| 38 | `M15A.STRATE.NONCU.STORAGESERVICECAPA` | NONCU 저장가능용량 (smartSTAR) |
| 39 | `M15A.STRATE.NONN2.STORAGECAPACITY` | NONN2 전체저장용량 (smartSTAR) |
| 40 | `M15A.STRATE.NONN2.STORAGECOUNT` | NONN2 저장수 (smartSTAR) |
| 41 | `M15A.STRATE.NONN2.STORAGERATIO` | NONN2 저장율 (smartSTAR) |
| 42 | `M15A.STRATE.NONN2.STORAGESERVICECAPA` | NONN2 저장가능용량 (smartSTAR) |
| 43 | `M15A.STRATE.POD.FABSTORAGECAPACITY` | POD 전체저장용량 (smartSTAR) |
| 44 | `M15A.STRATE.POD.FABSTORAGECOUNT` | POD 저장수 (smartSTAR) |
| 45 | `M15A.STRATE.POD.FABSTORAGERATIO` | POD 저장율 (smartSTAR) |
| 46 | `M15A.STRATE.POD.FABSTORAGESERVICECAPA` | POD 저장가능용량 (smartSTAR) |
| 47 | `M15A.STRATE.STB.N2STORAGECAPACITY` | N2STB 전체저장용량 (smartSTAR) |
| 48 | `M15A.STRATE.STB.N2STORAGECOUNT` | N2STB 저장수 (smartSTAR) |
| 49 | `M15A.STRATE.STB.N2STORAGERATIO` | N2STB 저장율 (smartSTAR) N2STB 로 이동중인 Queue 개수 포함, Down인 장비 제외 |
| 50 | `M15A.STRATE.STB.N2STORAGESERVICECAPA` | N2STB 저장가능용량 (smartSTAR) |
| 51 | `M15A.STRATE.STB.STORAGECAPACITY` | STB 전체저장용량 (smartSTAR) |
| 52 | `M15A.STRATE.STB.STORAGECOUNT` | STB 저장수 (smartSTAR) |
| 53 | `M15A.STRATE.STB.STORAGERATIO` | STB 저장율 (smartSTAR) |
| 54 | `M15A.STRATE.STB.STORAGESERVICECAPA` | STB 저장가능용량 (smartSTAR) |
| 55 | `M15A.STRATE.STK.N2STORAGECAPACITY` | N2STK 전체저장용량 (smartSTAR) |
| 56 | `M15A.STRATE.STK.N2STORAGECOUNT` | N2STK 저장수 (smartSTAR) |
| 57 | `M15A.STRATE.STK.N2STORAGERATIO` | N2STK 저장율 (smartSTAR) N2Stocker로 이동중인 Queue 개수 포함, Down인 장비 제외 |
| 58 | `M15A.STRATE.STK.N2STORAGESERVICECAPA` | N2STK 저장가능용량 (smartSTAR) |
| 59 | `M15A.STRATE.STB.NONN2STORAGERATIO` | NONN2STB 저장율(%) |
| 60 | `M15A.STRATE.STB.NONN2STORAGESERVICECAPA` | NONN2STB 저장가능용량 (smartSTAR) |

#### M15A.OHT — OHT 반송 (34건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `M15A.OHT.ALARM.CLUSTERSTATE` | OHT CLUSTER SERVER가 primary 또는 secondary로 전환되거나 Down/Up 상태일 때 알람 발송 |
| 2 | `M15A.OHT.SERVER_P.STATE` | 체크 대상 MCP의 Clustering 대상 Server 체크 1 = server1 2 = server2 3 = server down/up |
| 3 | `M15A.OHT.SERVER_S.STATE` | 체크 대상 MCP의 Clustering 대상 Server 체크 1 = server1 2 = server2 3 = server down/up |
| 4 | `M15A.OHT.ALL.MAX_CPU_TOTAL` | OHT의 Primary |
| 5 | `M15A.OHT.ALL.MAX_MEM_USED` | OHT의 Primary |
| 6 | `M15A.OHT.OHT_P.CLUSTER` | 체크 대상 MCP의 Clustering 상태 점검(FTP) 체크대상 MCP == current : green MCP7 <> Online : red 그외 yellow |
| 7 | `M15A.OHT.OHT_P.CPU_TOTAL` | 해당 Machine의 CPU 사용율(%)(FTP) |
| 8 | `M15A.OHT.OHT_P.DISK_USED` | 디스크사용율(%)(FTP) |
| 9 | `M15A.OHT.OHT_P.MEM_USED` | 해당 Machine의 Memory사용율(%)(FTP) |
| 10 | `M15A.OHT.OHT_P.PROS_STATUS` | not_active 개수가 5개이하 : green not_active 개수가 5개 초과이고, primary, secondary 모두 상태값이 0인게 하나 이상 : yellow not_active 개수가 5개 초과이고, primary, secondary 모두 상태값이 0인게 존재하지 않음 : red |
| 11 | `M15A.OHT.OHT_P.SERVICE_STATUS` | CLPSTAT의 결과 MCP7 ............: Online 이 아닐 경우(FTP) |
| 12 | `M15A.OHT.OHT_P.SWAP_USED` | SWAP Memeory 사용율(%)(FTP) |
| 13 | `M15A.OHT.OHT_S.CLUSTER` | 체크 대상 MCP의 Clustering 상태 점검(FTP) 체크대상 MCP == current : green MCP7 <> Online : red 그외 yellow |
| 14 | `M15A.OHT.OHT_S.CPU_TOTAL` | 해당 Machine의 CPU 사용율(%)(FTP) |
| 15 | `M15A.OHT.OHT_S.DISK_USED` | 디스크사용율(%)(FTP) |
| 16 | `M15A.OHT.OHT_S.MEM_USED` | 해당 Machine의 Memory사용율(%)(FTP) |
| 17 | `M15A.OHT.OHT_S.PROS_STATUS` | not_active 개수가 5개이하 : green not_active 개수가 5개 초과이고, primary, secondary 모두 상태값이 0인게 하나 이상 : yellow not_active 개수가 5개 초과이고, primary, secondary 모두 상태값이 0인게 존재하지 않음 : red |
| 18 | `M15A.OHT.OHT_S.SERVICE_STATUS` | CLPSTAT의 결과 MCP7 ............: Online 이 아닐 경우(FTP) |
| 19 | `M15A.OHT.OHT_S.SWAP_USED` | SWAP Memeory 사용율(%)(FTP) |
| 20 | `M15A.OHT.ALERT.OHTAUTO` | OHT MCP AUTO로 State 변경 |
| 21 | `M15A.OHT.ALERT.OHTCANNOTEXECUTE` | OHT에 반송명령시 Reply (2) Currently not able to excute 발생(MCS) |
| 22 | `M15A.OHT.ALERT.OHTCONNECTION` | MCS-OHT간 통신이 연결(Message) |
| 23 | `M15A.OHT.ALERT.OHTDISCONNECTION` | MCS-OHT간 통신이 끊어짐(Message) |
| 24 | `M15A.OHT.ALERT.OHTHIDALARM` | OHT HID Error 발생 |
| 25 | `M15A.OHT.ALERT.OHTINVALIDPARA` | OHT에 반송 명령시 Reply (3)으로 Parameter 값이 다른 경우 |
| 26 | `M15A.OHT.ALERT.OHTLOADFAIL` | OHT 반송중 TransferCompleted ResultCode (1)발생(Message) |
| 27 | `M15A.OHT.ALERT.OHTMCPALARM` | OHTMCPAlarm 발생(Message) |
| 28 | `M15A.OHT.ALERT.OHTMCPALARMCNT` | 현재 OHT Alarm 건수(MCSDB 기준) |
| 29 | `M15A.OHT.ALERT.OHTPAUSED` | OHT MCP PAUSED로 State 변경 |
| 30 | `M15A.OHT.ALERT.OHTSTATUSERROR` | OHT상태가 Pausing으로 장시간 대기(MCS) |
| 31 | `M15A.OHT.ALERT.PIOCOMMERROR` | PIO Communication Error 발생(Message) |
| 32 | `M15A.OHT.ALL.CLUSTER` | 체크 대상 MCP의 Clustering 상태 점검(FTP) 체크대상 MCP == current : green MCP7 <> Online : red 그외 yellow |
| 33 | `M15A.OHT.OHT_P.ASSIGNEDWAITTOTAL` | 배차 대기 총 수량(FTP) |
| 34 | `M15A.OHT.OHT_S.ASSIGNEDWAITTOTAL` | 배차 대기 총 수량(FTP) |

#### M15A.QUE — 반송 큐 (19건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `M15A.QUE.ABN.AOTRANSDELAY` | 5분마다 쿼리 검색하여 10분이상 Auto Output Port에 대기하고 있는 FOUP이 하나이상 존재하면 FOUP 개수를 알람 |
| 2 | `M15A.QUE.ABN.AOTRANSDELAYDET` | 10분이상 AO Port 반송 Delay 발생되는 FOUP 정보 통지(MCS) |
| 3 | `M15A.QUE.ABN.N2STOCKERDELAY` | 60분이상 N2Purge Stocker Port 내 FOUP 지연 발생 통지(MCS) |
| 4 | `M15A.QUE.ABN.QUETIMEDELAY` | 20분 이상 지연된 FOUP 목록 알림(MCS) |
| 5 | `M15A.QUE.ALL.CURRENTQCNT` |  |
| 6 | `M15A.QUE.ALL.CURRENTQCOMPLETED` | 최근 10분간 반송명령 완료수(HySTAR) |
| 7 | `M15A.QUE.ALL.CURRENTQCREATED` | 최근10분간 Q생성 수(HySTAR) |
| 8 | `M15A.QUE.ALL.CURRENTRETICLEQCNT` | 현재 POD 반송 Q 수(MCS) 출발지 장비와 목적지 장비가 다른 Queue 대상 |
| 9 | `M15A.QUE.LOAD.AVGFOUPLOADTIME` | 최근 10분간 FOUP 평균 Load반송시간(smartSTAR) |
| 10 | `M15A.QUE.LOAD.AVGLOADTIME` | 최근 10분간 평균 Load반송시간(HySTAR) |
| 11 | `M15A.QUE.LOAD.AVGLOADTIME1MIN` | 최근 1분간 평균 Load반송시간(smartSTAR) |
| 12 | `M15A.QUE.LOAD.AVGRETICLELOADTIME` | 최근 10분간 Reticle 평균 Load반송시간(smartSTAR) |
| 13 | `M15A.QUE.LOAD.CURRENTLOADQCNT` | 반송 큐 개수: 목적지가 생산장비인 건(MCS) |
| 14 | `M15A.QUE.LOAD.CURRENTRETICLELOADQCNT` | 목적지가 생산장비인 POD 반송 Q수(MCS) |
| 15 | `M15A.QUE.OHT.CURRENTOHTQCNT` | 현재 OHT반송 Q 수(MCS) |
| 16 | `M15A.QUE.OHT.CURRENTRETICLEOHTQCNT` | 현재 Reticle OHT반송 Q 수(MCS) |
| 17 | `M15A.QUE.OHT.OHTUTIL` | OHT사용율(%)(MCS) |
| 18 | `M15A.QUE.OHT.RTCOHTUTIL` | Reticle OHT사용율(%)(MCS) |
| 19 | `M15A.QUE.SENDFAB.VERTICALQUEUECOUNT` | 현재 M15A->M15B Q 수(STA) |

#### M15A.INV — 인버터 (10건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `M15A.INV.ALERT.INVAUTO` | INV MCP AUTO로 State 변경 |
| 2 | `M15A.INV.ALERT.INVCONNECTION` | MCS-INV간 통신이 연결(Message) |
| 3 | `M15A.INV.ALERT.INVDISCONNECTION` | MCS-INV간 통신이 끊어짐(Message) |
| 4 | `M15A.INV.ALERT.INVMCPALARM` | INVMCPAlarm 발생(Message) |
| 5 | `M15A.INV.ALERT.INVPAUSED` | INV MCP PAUSED로 State 변경 |
| 6 | `M15A.INV.PC.CPU_TOTAL` | 해당 Machine의 CPU 사용율(%)(FTP) |
| 7 | `M15A.INV.PC.DISK_USED` | 디스크사용율(%)(FTP) |
| 8 | `M15A.INV.PC.MEM_USED` | 해당 Machine의 Memory사용율(%)(FTP) |
| 9 | `M15A.INV.PC.PROS_STATUS` | not_active 개수가 5개이하 : green not_active 개수가 5개 초과이고, primary, secondary 모두 상태값이 0인게 하나 이상 : yellow not_active 개수가 5개 초과이고, primary, secondary 모두 상태값이 0인게 존재하지 않음 : red |
| 10 | `M15A.INV.PC.SWAP_USED` | SWAP Memeory 사용율(%)(FTP) |

#### M15A.INPOS — 포트 상태 (7건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `M15A.INPOS.ALERT.101` | I/O 읽기/쓰기 에러 |
| 2 | `M15A.INPOS.ALERT.90003` | TraceMessage 통신 단절 |
| 3 | `M15A.INPOS.ALERT.103` | MC OFF 에러 |
| 4 | `M15A.INPOS.ALERT.104` | 비상정지(E-STOP) |
| 5 | `M15A.INPOS.ALERT.113` | 메인 DC 전원 에러 |
| 6 | `M15A.INPOS.ALERT.114` | 메인 PW DC 전원 에러 |
| 7 | `M15A.INPOS.ALERT.112` | 퍼지 미시작 |

#### M15A.LFT — 리프터 (5건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `M15A.LFT.ALERT.LFTAUTO` | LFT MCP AUTO로 State 변경 |
| 2 | `M15A.LFT.ALERT.LFTCONNECTION` | MCS-LFT간 통신이 연결(Message) |
| 3 | `M15A.LFT.ALERT.LFTDISCONNECTION` | MCS-OHT간 통신이 끊어짐(Message) |
| 4 | `M15A.LFT.ALERT.LFTPAUSED` | LFT MCP PAUSED로 State 변경 |
| 5 | `M15A.LFT.SENDFAB.TO_M15C_CURRENTQCNT` | 현재 M15A->M15X Q 수 |

#### M15A.SORTER — 소터 (5건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `M15A.SORTER.ABN.SORTERWAITCOUNTOVER` | LOT의 장비 Group이 Sorter이고 Group ID가 G-TSR-02이 아니며 LOT 상태가 Released인 수량(MES) |
| 2 | `M15A.SORTER.ABN.CUSORTERWAITCOUNTOVER` | LOT의 장비 Group ID가 G-TSR-02이고 LOT 상태가 Released인 수량(MES) |
| 3 | `M15A.SORTER.ABN.PORTUNBALANCE` | ProcDescription이 OCR |
| 4 | `M15A.SORTER.ABN.SORTERRESVFAIL` | 공정 Type이 Split, Merge, Exchange이고 Lot상태가 Released, WAIT인 상태에서 Sorter_job_id가 생성되지 않았을 경우(MES) |
| 5 | `M15A.SORTER.ABN.SORTERTRANSFERFAIL` | 예약(RESV) 상태 |

#### M15A.STK — 스토커 (5건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `M15A.STK.ALERT.STKAUTO` | STK MCP AUTO로 State 변경 |
| 2 | `M15A.STK.ALERT.STKCONNECTION` | MCS-STK간 통신이 연결(Message) |
| 3 | `M15A.STK.ALERT.STKDISCONNECTION` | MCS-STK간 통신이 끊어짐(Message) |
| 4 | `M15A.STK.ALERT.STKMCPALARM` | STKMCPAlarm 발생(Message) |
| 5 | `M15A.STK.ALERT.STKPAUSED` | STK MCP PAUSED로 State 변경 |

#### M15A.ETC — 기타 (4건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `M15A.ETC.ALERT.CARRIERDUPLICATE` | MCS가 동일 Carrier ID 중복 존재여부 감지시 발생(Message) |
| 2 | `M15A.ETC.ALERT.DESTUNITNULL` | MCS에 장비 또는 포트가 등록 되어있지 않은 명령발생(Message) |
| 3 | `M15A.ETC.ALERT.N2FOUPMISMATCHDEST` | N2 FOUP이 반송이상으로 인해 Normal STB로 반송되는 경우 알람 송신 |
| 4 | `M15A.ETC.ALERT.TRANSACTIONTIMEISOVER` | MES명령에 대한 Transaction 처리가 8초 이상 5건 발생 시 알람 송신 |

#### M15A.RTC — RTC (4건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `M15A.RTC.ALERT.RTCAUTO` | STK MCP AUTO로 State 변경 |
| 2 | `M15A.RTC.ALERT.RTCCONNECTION` | MCS-STK간 통신이 연결(Message) |
| 3 | `M15A.RTC.ALERT.RTCDISCONNECTION` | MCS-RTC간 통신이 끊어짐(Message) |
| 4 | `M15A.RTC.ALERT.RTCPAUSED` | STK MCP PAUSED로 State 변경 |

#### M15A.AZFS — AZFS (2건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `M15A.AZFS.ALERT.DOUBLESTORAGE` | MCS가 Carrier DoubleStorage를 감지함(MCS) |
| 2 | `M15A.AZFS.ALERT.SOURCEEMPTY` | SourceEmpty 발생 |

#### M15A.EMPTYFOUP — Empty FOUP (2건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `M15A.EMPTYFOUP.5PNM.CURRENTCOUNT` | 빈 FOUP CURRENTCOUNT |
| 2 | `M15A.EMPTYFOUP.5PNN.CURRENTCOUNT` | 빈 FOUP CURRENTCOUNT |

#### M15A.OHTALERT — OHTALERT (1건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `M15A.OHTALERT.OHTRETRYFAILED` | OHT 3회 반송 시도 실패 발생 |

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
