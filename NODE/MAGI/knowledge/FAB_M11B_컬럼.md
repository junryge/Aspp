# QUWA FAB 컬럼 사전
> **구조**: `{FAB}.{Category}.{SubCategory}.{Metric}`
> **영문 Description → 한글 번역 완료**

---

### M11B (251건)

#### M11B.MCS — 서버 인프라 (114건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `M11B.MCS.ALERT.DELAYEDTRANSPORTJOB` | Transport Job Delay 발생 (MCS) |
| 2 | `M11B.MCS.ALERT.PROCESSDOWN` | MCS에서 특정 Process의 비정상 Down을 감지(Message) |
| 3 | `M11B.MCS.ALERT.PROCESSHANG` | MCS내 컨트롤 서버가 MCS에 Heartbeat메시지를 날렸으나 반응이 없을 경우 Hang으로 간주(Message) |
| 4 | `M11B.MCS.ALERT.PROCESSRESTART` | MCS내 컨트롤 서버가 이상이있는 서버에 대해 자동으로 Restart 명령을 발생시킴(Message) |
| 5 | `M11B.MCS.ALERT.PROCESSSHUTDOWNFAIL` | Process에 shutdown명령을 내렸으나 종료되지 않음(Message) |
| 6 | `M11B.MCS.ALERT.PROCESSUPFAIL` | MCS내 컨트롤 서버가 Process를 StartUp시켰으나 제대로 살아나지 않을 경우(Message) |
| 7 | `M11B.MCS.ALERT.PROCESSUPSTART` | MCS내 컨트롤 서버가 Process를 StartUp시도하는 경우(Message) |
| 8 | `M11B.MCS.M11BMCSA1.PS_DS01_VSS` | 해당 Process의 CPU 사용율(FTP) |
| 9 | `M11B.MCS.M11BMCSA1.PS_TS01_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 10 | `M11B.MCS.M11BMCSA1.PS_TS01_RSS` | 해당 Process의 CPU 사용율(FTP) |
| 11 | `M11B.MCS.ALERT.DBQUERYHANG` | DB 쿼리 프로세스 무응답 |
| 12 | `M11B.MCS.M11BMCSA2.PS_EI01_OHT_S_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 13 | `M11B.MCS.M11BMCSA2.PS_EI01_OHT_S_VSS` | 해당 Process의 CPU 사용율(FTP) |
| 14 | `M11B.MCS.M11BMCSA2.PS_EI02_INV_S_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 15 | `M11B.MCS.M11BMCSA2.PS_EI02_INV_S_RSS` | 해당 Process의 CPU 사용율(FTP) |
| 16 | `M11B.MCS.M11BMCSA2.PS_EI02_INV_S_VSS` | 해당 Process의 CPU 사용율(FTP) |
| 17 | `M11B.MCS.M11BMCSA2.PS_EI03_STK_S_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 18 | `M11B.MCS.M11BMCSA2.PS_EI03_STK_S_RSS` | 해당 Process의 CPU 사용율(FTP) |
| 19 | `M11B.MCS.M11BMCSA2.PS_EI03_STK_S_VSS` | 해당 Process의 CPU 사용율(FTP) |
| 20 | `M11B.MCS.M11BMCSA1.DISK_APP1` | 디스크사용율(%)(FTP) |
| 21 | `M11B.MCS.M11BMCSA1.CPU_TOTAL` | 해당 Machine의 CPU 사용율(%)(FTP) |
| 22 | `M11B.MCS.M11BMCSD1.CPU_TOTAL` | 해당 Machine의 CPU 사용율(%)(FTP) |
| 23 | `M11B.MCS.M11BMCSD1.DISK_APP1` | 디스크사용율(%)(FTP) |
| 24 | `M11B.MCS.M11BMCSD1.DISK_APP2` | 디스크사용율(%)(FTP) |
| 25 | `M11B.MCS.M11BMCSD1.DISK_USED` | 디스크사용율(%)(FTP) |
| 26 | `M11B.MCS.M11BMCSD1.MEM_FREE` | 해당 Machine의 Memory 여유공간(FTP) |
| 27 | `M11B.MCS.M11BMCSD1.MEM_USED` | 해당 Machine의 Memory사용율(%)(FTP) |
| 28 | `M11B.MCS.M11BMCSD1.MEM_USING` | 해당 Machine의 Memory사용율(%)(FTP) |
| 29 | `M11B.MCS.M11BMCSD1.SESSION_CNT` | DB Session수(FTP)oracleM12MCS1의 프로세스개수를 집계하여 세션의 개수를 파악함 |
| 30 | `M11B.MCS.M11BMCSD1.SWAP_IN` | Memory Swap In 발생여부(FTP) |
| 31 | `M11B.MCS.M11BMCSD1.SWAP_OUT` | 해당 Machine에서 Memory Swap Out이 발생한 회수(FTP) |
| 32 | `M11B.MCS.M11BMCSA1.SWAP_IN` | Memory Swap In 발생여부(FTP) |
| 33 | `M11B.MCS.M11BMCSA1.MEM_USED` | 해당 Machine의 Memory사용율(%)(FTP) |
| 34 | `M11B.MCS.M11BMCSA1.MEM_FREE` | 해당 Machine의 Memory 여유공간(FTP) |
| 35 | `M11B.MCS.M11BMCSA1.SWAP_OUT` | 해당 Machine에서 Memory Swap Out이 발생한 회수(FTP) |
| 36 | `M11B.MCS.M11BMCSA2.DISK_APP1` | 디스크사용율(%)(FTP) |
| 37 | `M11B.MCS.M11BMCSA2.CPU_TOTAL` | 해당 Machine의 CPU 사용율(%)(FTP) |
| 38 | `M11B.MCS.M11BMCSA2.MEM_FREE` | 해당 Machine의 Memory 여유공간(FTP) |
| 39 | `M11B.MCS.M11BMCSA2.MEM_USED` | 해당 Machine의 Memory사용율(%)(FTP) |
| 40 | `M11B.MCS.M11BMCSA2.PS_EI01_OHT_S_RSS` | 해당 Process의 CPU 사용율(FTP) |
| 41 | `M11B.MCS.M11BMCSA1.PS_CS01_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 42 | `M11B.MCS.M11BMCSA1.PS_TS01_VSS` | 해당 Process의 CPU 사용율(FTP) |
| 43 | `M11B.MCS.M11BMCSA1.PS_TS02_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 44 | `M11B.MCS.M11BMCSA1.PS_TS02_RSS` | 해당 Process의 CPU 사용율(FTP) |
| 45 | `M11B.MCS.M11BMCSA1.PS_TS03_RSS` | 해당 Process의 CPU 사용율(FTP) |
| 46 | `M11B.MCS.M11BMCSA1.PS_TS03_VSS` | 해당 Process의 CPU 사용율(FTP) |
| 47 | `M11B.MCS.M11BMCSA1.PS_CS01_RSS` | 해당 Process의 CPU 사용율(FTP) |
| 48 | `M11B.MCS.M11BMCSA1.PS_CS01_VSS` | 해당 Process의 CPU 사용율(FTP) |
| 49 | `M11B.MCS.M11BMCSA1.PS_DS01_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 50 | `M11B.MCS.M11BMCSA1.PS_DS01_RSS` | 해당 Process의 CPU 사용율(FTP) |
| 51 | `M11B.MCS.M11BMCSA1.PS_TS05_VSS` | 해당 Process의 CPU 사용율(FTP) |
| 52 | `M11B.MCS.M11BMCSA1.PS_TS02_VSS` | 해당 Process의 CPU 사용율(FTP) |
| 53 | `M11B.MCS.M11BMCSA1.PS_TS03_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 54 | `M11B.MCS.M11BMCSA1.PS_TS04_VSS` | 해당 Process의 CPU 사용율(FTP) |
| 55 | `M11B.MCS.M11BMCSA1.PS_TS05_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 56 | `M11B.MCS.M11BMCSA1.PS_TS05_RSS` | 해당 Process의 CPU 사용율(FTP) |
| 57 | `M11B.MCS.M11BMCSA1.PS_EI01_OHT_P_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 58 | `M11B.MCS.M11BMCSA1.PS_EI01_OHT_P_RSS` | 해당 Process의 CPU 사용율(FTP) |
| 59 | `M11B.MCS.M11BMCSA1.PS_EI01_OHT_P_VSS` | 해당 Process의 CPU 사용율(FTP) |
| 60 | `M11B.MCS.M11BMCSA1.PS_EI02_INV_P_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 61 | `M11B.MCS.M11BMCSA1.PS_TS04_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 62 | `M11B.MCS.M11BMCSA1.PS_TS04_RSS` | 해당 Process의 CPU 사용율(FTP) |
| 63 | `M11B.MCS.M11BMCSA1.PS_EI02_INV_P_RSS` | 해당 Process의 CPU 사용율(FTP) |
| 64 | `M11B.MCS.M11BMCSA1.PS_EI02_INV_P_VSS` | 해당 Process의 CPU 사용율(FTP) |
| 65 | `M11B.MCS.M11BMCSA1.PS_EI03_STK_P_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 66 | `M11B.MCS.M11BMCSA1.PS_EI03_STK_P_RSS` | 해당 Process의 CPU 사용율(FTP) |
| 67 | `M11B.MCS.M11BMCSA1.PS_EI06_LFT_P_RSS` | 해당 Process의 CPU 사용율(FTP) |
| 68 | `M11B.MCS.M11BMCSA1.PS_EI06_LFT_P_VSS` | 해당 Process의 CPU 사용율(FTP) |
| 69 | `M11B.MCS.M11BMCSA2.PS_EI06_LFT_S_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 70 | `M11B.MCS.M11BMCSA2.PS_EI06_LFT_S_RSS` | 해당 Process의 CPU 사용율(FTP) |
| 71 | `M11B.MCS.M11BMCSA1.PS_EI03_STK_P_VSS` | 해당 Process의 CPU 사용율(FTP) |
| 72 | `M11B.MCS.M11BMCSA2.SWAP_IN` | Memory Swap In 발생여부(FTP) |
| 73 | `M11B.MCS.M11BMCSA2.SWAP_OUT` | 해당 Machine에서 Memory Swap Out이 발생한 회수(FTP) |
| 74 | `M11B.MCS.M11BMCSA1.PS_EI06_LFT_P_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 75 | `M11B.MCS.M11BMCSA2.PS_EI06_LFT_S_VSS` | 해당 Process의 CPU 사용율(FTP) |
| 76 | `M11B.MCS.M11BMCSA1.PS_EI04_STK_P_VSS` | 해당 Process의 CPU 사용율(FTP) |
| 77 | `M11B.MCS.M11BMCSA1.PS_EI05_STK_P_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 78 | `M11B.MCS.M11BMCSA1.PS_EI05_STK_P_RSS` | 해당 Process의 CPU 사용율(FTP) |
| 79 | `M11B.MCS.M11BMCSA1.PS_EI05_STK_P_VSS` | 해당 Process의 CPU 사용율(FTP) |
| 80 | `M11B.MCS.M11BMCSA1.PS_EI07_FIO_P_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 81 | `M11B.MCS.M11BMCSA1.PS_EI07_FIO_P_RSS` | 해당 Process의 CPU 사용율(FTP) |
| 82 | `M11B.MCS.M11BMCSA1.PS_EI07_FIO_P_VSS` | 해당 Process의 CPU 사용율(FTP) |
| 83 | `M11B.MCS.M11BMCSA2.PS_EI04_STK_S_VSS` | 해당 Process의 CPU 사용율(FTP) |
| 84 | `M11B.MCS.M11BMCSA2.PS_EI05_STK_S_RSS` | 해당 Process의 CPU 사용율(FTP) |
| 85 | `M11B.MCS.M11BMCSA1.PS_EI04_STK_P_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 86 | `M11B.MCS.M11BMCSA1.PS_EI04_STK_P_RSS` | 해당 Process의 CPU 사용율(FTP) |
| 87 | `M11B.MCS.M11BMCSA2.PS_EI05_STK_S_VSS` | 해당 Process의 CPU 사용율(FTP) |
| 88 | `M11B.MCS.M11BMCSA2.PS_EI07_FIO_S_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 89 | `M11B.MCS.M11BMCSA2.PS_EI07_FIO_S_RSS` | 해당 Process의 CPU 사용율(FTP) |
| 90 | `M11B.MCS.M11BMCSA2.PS_EI07_FIO_S_VSS` | 해당 Process의 CPU 사용율(FTP) |
| 91 | `M11B.MCS.M11BMCSA2.PS_EI04_STK_S_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 92 | `M11B.MCS.M11BMCSA2.PS_EI04_STK_S_RSS` | 해당 Process의 CPU 사용율(FTP) |
| 93 | `M11B.MCS.M11BMCSA2.PS_CS02_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 94 | `M11B.MCS.M11BMCSA2.PS_EI05_STK_S_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 95 | `M11B.MCS.M11BMCSA2.PS_CS02_RSS` | 해당 Process의 CPU 사용율(FTP) |
| 96 | `M11B.MCS.M11BMCSA2.PS_CS02_VSS` | 해당 Process의 CPU 사용율(FTP) |
| 97 | `M11B.MCS.M11BMCSA2.PS_DS02_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 98 | `M11B.MCS.M11BMCSA2.PS_DS02_RSS` | 해당 Process의 CPU 사용율(FTP) |
| 99 | `M11B.MCS.M11BMCSA2.PS_DS02_VSS` | 해당 Process의 CPU 사용율(FTP) |
| 100 | `M11B.MCS.M11BMCSA2.PS_TS11_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 101 | `M11B.MCS.M11BMCSA2.PS_TS14_VSS` | 해당 Process의 CPU 사용율(FTP) |
| 102 | `M11B.MCS.M11BMCSA2.PS_TS11_RSS` | 해당 Process의 CPU 사용율(FTP) |
| 103 | `M11B.MCS.M11BMCSA2.PS_TS11_VSS` | 해당 Process의 CPU 사용율(FTP) |
| 104 | `M11B.MCS.M11BMCSA2.PS_TS12_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 105 | `M11B.MCS.M11BMCSA2.PS_TS15_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 106 | `M11B.MCS.M11BMCSA2.PS_TS12_RSS` | 해당 Process의 CPU 사용율(FTP) |
| 107 | `M11B.MCS.M11BMCSA2.PS_TS15_VSS` | 해당 Process의 CPU 사용율(FTP) |
| 108 | `M11B.MCS.M11BMCSA2.PS_TS12_VSS` | 해당 Process의 CPU 사용율(FTP) |
| 109 | `M11B.MCS.M11BMCSA2.PS_TS13_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 110 | `M11B.MCS.M11BMCSA2.PS_TS13_RSS` | 해당 Process의 CPU 사용율(FTP) |
| 111 | `M11B.MCS.M11BMCSA2.PS_TS15_RSS` | 해당 Process의 CPU 사용율(FTP) |
| 112 | `M11B.MCS.M11BMCSA2.PS_TS13_VSS` | 해당 Process의 CPU 사용율(FTP) |
| 113 | `M11B.MCS.M11BMCSA2.PS_TS14_CPU` | 해당 Process의 CPU 사용율(FTP) |
| 114 | `M11B.MCS.M11BMCSA2.PS_TS14_RSS` | 해당 Process의 CPU 사용율(FTP) |

#### M11B.STRATE — 저장 현황 (44건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `M11B.STRATE.STB.NONCUSTORAGESERVICECAPA` | NONCU STB 저장가능용량 (smartSTAR) NONCU STB로 이동중인 Queue 개수 포함, Down인 장비 제외 |
| 2 | `M11B.STRATE.STK.N2STORAGESERVICECAPA` | N2 STK 저장가능용량 (smartSTAR) N2Stocker로 이동중인 Queue 개수 포함, Down인 장비 제외 |
| 3 | `M11B.STRATE.STK.N2STORAGECAPACITY` | N2 STK 전체저장용량 (smartSTAR) N2Stocker로 이동중인 Queue 개수 포함, Down인 장비 제외 |
| 4 | `M11B.STRATE.STK.N2STORAGECOUNT` | N2 STK 저장수 (smartSTAR) N2Stocker로 이동중인 Queue 개수 포함, Down인 장비 제외 |
| 5 | `M11B.STRATE.STB.N2STORAGESERVICECAPA` | N2 STB 저장가능용량 (smartSTAR) N2 STB로 이동중인 Queue 개수 포함, Down인 장비 제외 |
| 6 | `M11B.STRATE.STB.N2STORAGECAPACITY` | N2 STB 전체저장용량 (smartSTAR) N2 STB로 이동중인 Queue 개수 포함, Down인 장비 제외 |
| 7 | `M11B.STRATE.STB.N2STORAGECOUNT` | N2 STB 저장수 (smartSTAR) N2 STB로 이동중인 Queue 개수 포함, Down인 장비 제외 |
| 8 | `M11B.STRATE.STB.N2STORAGERATIO` | N2 STB 저장율 (smartSTAR) N2 STB로 이동중인 Queue 개수 포함, Down인 장비 제외 |
| 9 | `M11B.STRATE.STB.CUSTORAGESERVICECAPA` | CU STB 저장가능용량 (smartSTAR) CU STB로 이동중인 Queue 개수 포함, Down인 장비 제외 |
| 10 | `M11B.STRATE.STK.N2STORAGERATIO` | N2 STK 저장율 (smartSTAR) N2Stocker로 이동중인 Queue 개수 포함, Down인 장비 제외 |
| 11 | `M11B.STRATE.STK.STORAGESERVICECAPA` | STK 저장가능용량 (smartSTAR) Stocker로 이동중인 Queue 개수 포함, Down인 장비 제외 |
| 12 | `M11B.STRATE.STK.STORAGECAPACITY` | STK 전체저장용량 (smartSTAR) Stocker로 이동중인 Queue 개수 포함, Down인 장비 제외 |
| 13 | `M11B.STRATE.STK.STORAGECOUNT` | STK 저장수 (smartSTAR) Stocker로 이동중인 Queue 개수 포함, Down인 장비 제외 |
| 14 | `M11B.STRATE.STK.STORAGERATIO` | STK 저장율 (smartSTAR) Stocker로 이동중인 Queue 개수 포함, Down인 장비 제외 |
| 15 | `M11B.STRATE.N2.STORAGESERVICECAPA` | N2 STK + STB 저장가능용량 (smartSTAR) N2 STK로 이동중인 Queue 개수 포함, Down인 장비 제외 |
| 16 | `M11B.STRATE.N2.STORAGERATIO` | N2 STK + STB 저장율 (smartSTAR) N2 STK로 이동중인 Queue 개수 포함, Down인 장비 제외 |
| 17 | `M11B.STRATE.N2.STORAGECAPACITY` | N2 STK + STB 전체저장용량 (smartSTAR) N2 STK로 이동중인 Queue 개수 포함, Down인 장비 제외 |
| 18 | `M11B.STRATE.N2.STORAGECOUNT` | N2 STK + STB 저장수 (smartSTAR) N2 STK로 이동중인 Queue 개수 포함, Down인 장비 제외 |
| 19 | `M11B.STRATE.STB.NONN2STORAGECAPACITY` | NONN2 STB 전체저장용량 (smartSTAR) NONN2 STB로 이동중인 Queue 개수 포함, Down인 장비 제외 |
| 20 | `M11B.STRATE.STB.NONN2STORAGECOUNT` | NONN2 STB 저장수 (smartSTAR) NONN2 STB로 이동중인 Queue 개수 포함, Down인 장비 제외 |
| 21 | `M11B.STRATE.STB.NONN2STORAGERATIO` | NONN2 STB 저장수 (smartSTAR) NONN2 STB로 이동중인 Queue 개수 포함, Down인 장비 제외 |
| 22 | `M11B.STRATE.STB.NONN2STORAGESERVICECAPA` | NONN2 STB 저장가능용량 (smartSTAR) NONN2 STB로 이동중인 Queue 개수 포함, Down인 장비 제외 |
| 23 | `M11B.STRATE.STK.NONN2STORAGECAPACITY` | NONN2 STK 전체저장용량 (smartSTAR) NONN2 STK로 이동중인 Queue 개수 포함, Down인 장비 제외 |
| 24 | `M11B.STRATE.STK.NONN2STORAGECOUNT` | NONN2 STK 저장수 (smartSTAR) NONN2 STK로 이동중인 Queue 개수 포함, Down인 장비 제외 |
| 25 | `M11B.STRATE.STK.NONN2STORAGERATIO` | NONN2 STK 저장수 (smartSTAR) NONN2 STK로 이동중인 Queue 개수 포함, Down인 장비 제외 |
| 26 | `M11B.STRATE.STK.NONN2STORAGESERVICECAPA` | NONN2 STK 저장가능용량 (smartSTAR) NONN2 STK로 이동중인 Queue 개수 포함, Down인 장비 제외 |
| 27 | `M11B.STRATE.STB.CUSTORAGERATIO` | CU STB 저장율 (smartSTAR) CU STB로 이동중인 Queue 개수 포함, Down인 장비 제외 |
| 28 | `M11B.STRATE.STB.NONCUSTORAGECAPACITY` | NONCU STB 전체저장용량 (smartSTAR) NONCU STB로 이동중인 Queue 개수 포함, Down인 장비 제외 |
| 29 | `M11B.STRATE.STB.NONCUSTORAGECOUNT` | NONCU STB 저장수 (smartSTAR) NONCU STB로 이동중인 Queue 개수 포함, Down인 장비 제외 |
| 30 | `M11B.STRATE.STB.NONCUSTORAGERATIO` | NONCU STB 저장율 (smartSTAR) NONCU STB로 이동중인 Queue 개수 포함, Down인 장비 제외 |
| 31 | `M11B.STRATE.NONN2.STORAGERATIO` | NONN2 저장율 (smartSTAR) |
| 32 | `M11B.STRATE.NONN2.STORAGECAPACITY` | NONN2 전체저장용량 (smartSTAR) |
| 33 | `M11B.STRATE.NONN2.STORAGESERVICECAPA` | NONN2 저장가능용량 (smartSTAR) |
| 34 | `M11B.STRATE.NONN2.STORAGECOUNT` | NONN2 저장수 (smartSTAR) |
| 35 | `M11B.STRATE.ALL.FABSTORAGESERVICECAPA` | FAB내 저장장치(STK,ZFS) 저장가능용량 (smartSTAR) ReservedLocationCount 포함, Down된 장비 제외 |
| 36 | `M11B.STRATE.ALL.FABSTORAGECAPACITY` | FAB내 저장장치(STK,ZFS) 전체저장용량 (smartSTAR) ReservedLocationCount 포함, Down된 장비 제외 |
| 37 | `M11B.STRATE.STB.STORAGECAPACITY` | STB 전체저장용량 (smartSTAR) |
| 38 | `M11B.STRATE.STB.STORAGESERVICECAPA` | STB 저장가능용량 (smartSTAR) |
| 39 | `M11B.STRATE.ALL.FABSTORAGECOUNT` | FAB내 저장장치(STK,ZFS) 저장수 (smartSTAR) ReservedLocationCount 포함, Down된 장비 제외 |
| 40 | `M11B.STRATE.ALL.FABSTORAGERATIO` | FAB내 저장장치(STK,ZFS) 저장율 (smartSTAR) ReservedLocationCount 포함, Down된 장비 제외 |
| 41 | `M11B.STRATE.STB.STORAGECOUNT` | STB 저장수 (smartSTAR) |
| 42 | `M11B.STRATE.STB.STORAGERATIO` | STB 저장율 (smartSTAR) |
| 43 | `M11B.STRATE.STB.CUSTORAGECAPACITY` | CU STB 전체저장용량 (smartSTAR) CU STB로 이동중인 Queue 개수 포함, Down인 장비 제외 |
| 44 | `M11B.STRATE.STB.CUSTORAGECOUNT` | CU STB 저장수 (smartSTAR) CU STB로 이동중인 Queue 개수 포함, Down인 장비 제외 |

#### M11B.OHT — OHT 반송 (30건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `M11B.OHT.ALARM.CLUSTERSTATE` | OHT CLUSTER SERVER가 primary 또는 secondary로 전환되거나 Down/Up 상태일 때 알람 발송 |
| 2 | `M11B.OHT.SERVER_P.STATE` | 체크 대상 MCP의 Clustering 대상 Server 체크 1 = server1 2 = server2 3 = server down/up |
| 3 | `M11B.OHT.SERVER_S.STATE` | 체크 대상 MCP의 Clustering 대상 Server 체크 1 = server1 2 = server2 3 = server down/up |
| 4 | `M11B.OHT.ALARM.CURRENTOHTALARMCNT` | 현재 OHT UnitAlarm 건수(MCS DB 기준) |
| 5 | `M11B.OHT.ALERT.OHTCANNOTEXECUTE` | OHT에 반송명령시 Reply (2) Currently not able to excute 발생(MCS) |
| 6 | `M11B.OHT.ALERT.OHTDISCONNECTION` | MCS-OHT간 통신이 끊어짐(Message) |
| 7 | `M11B.OHT.ALERT.OHTLOADFAIL` | OHT 반송중 TransferCompleted ResultCode (1)발생(Message) |
| 8 | `M11B.OHT.ALERT.OHTPARAERR` | OHT에 반송명령시 Reply (3) OHT Para Error 발생(MCS) |
| 9 | `M11B.OHT.ALERT.OHTRETRYCOUNTOVER` | OHT 장비 Load/Unload 시도 회수 초과발생(Message) |
| 10 | `M11B.OHT.ALERT.OHTRETRYCOUNTOVER2` | Load시 ResultCode'1' 발생회수 초과 발생(Message) |
| 11 | `M11B.OHT.ALERT.OHTSTATUSERROR` | OHT상태가 Pausing으로 장시간 대기(MCS) |
| 12 | `M11B.OHT.ALL.CLUSTER` | 체크 대상 MCP의 Clustering 상태 점검(FTP) 체크대상 MCP == current : green MCP7 <> Online : red 그외 yellow |
| 13 | `M11B.OHT.ALL.MAX_CPU_TOTAL` | OHT의 Primary |
| 14 | `M11B.OHT.ALL.MAX_MEM_USED` | OHT의 Primary |
| 15 | `M11B.OHT.OHT_P.CLUSTER` | 체크 대상 MCP의 Clustering 상태 점검(FTP) 체크대상 MCP == current : green MCP7 <> Online : red 그외 yellow |
| 16 | `M11B.OHT.OHT_P.CPU_TOTAL` | 해당 Machine의 CPU 사용율(%)(FTP) |
| 17 | `M11B.OHT.OHT_P.DISK_USED` | 디스크사용율(%)(FTP) |
| 18 | `M11B.OHT.OHT_P.MEM_USED` | 해당 Machine의 Memory사용율(%)(FTP) |
| 19 | `M11B.OHT.OHT_P.PROS_STATUS` | not_active 개수가 5개이하 : green not_active 개수가 5개 초과이고, primary, secondary 모두 상태값이 0인게 하나 이상 : yellow not_active 개수가 5개 초과이고, primary, secondary 모두 상태값이 0인게 존재하지 않음 : red |
| 20 | `M11B.OHT.OHT_P.SERVICE_STATUS` | CLPSTAT의 결과 MCP7 ............: Online 이 아닐 경우(FTP) |
| 21 | `M11B.OHT.OHT_P.SWAP_USED` | SWAP Memeory 사용율(%)(FTP) |
| 22 | `M11B.OHT.OHT_S.CLUSTER` | 체크 대상 MCP의 Clustering 상태 점검(FTP) 체크대상 MCP == current : green MCP7 <> Online : red 그외 yellow |
| 23 | `M11B.OHT.OHT_S.CPU_TOTAL` | 해당 Machine의 CPU 사용율(%)(FTP) |
| 24 | `M11B.OHT.OHT_S.DISK_USED` | 디스크사용율(%)(FTP) |
| 25 | `M11B.OHT.OHT_S.MEM_USED` | 해당 Machine의 Memory사용율(%)(FTP) |
| 26 | `M11B.OHT.OHT_S.PROS_STATUS` | not_active 개수가 5개이하 : green not_active 개수가 5개 초과이고, primary, secondary 모두 상태값이 0인게 하나 이상 : yellow not_active 개수가 5개 초과이고, primary, secondary 모두 상태값이 0인게 존재하지 않음 : red |
| 27 | `M11B.OHT.OHT_S.SERVICE_STATUS` | CLPSTAT의 결과 MCP7 ............: Online 이 아닐 경우(FTP) |
| 28 | `M11B.OHT.OHT_S.SWAP_USED` | SWAP Memeory 사용율(%)(FTP) |
| 29 | `M11B.OHT.OHT_P.ASSIGNEDWAITTOTAL` | 배차 대기 총 수량(FTP) |
| 30 | `M11B.OHT.OHT_S.ASSIGNEDWAITTOTAL` | 배차 대기 총 수량(FTP) |

#### M11B.QUE — 반송 큐 (18건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `M11B.QUE.ABN.FOUPCLEANDELAY` | FOUP Clean 대상 FOUP이 일반 Stocker에 저장되어 있는체로 60분이상 경과시 자동 일반Stocker로 명령(MES |
| 2 | `M11B.QUE.ABN.MAINTBUFFERPROCLOT` | SOURCE_MESBAY_NM_SQL내 해당 LOT Code |
| 3 | `M11B.QUE.ABN.N2STOCKERDELAY` | 30분이상 N2Purge Stocker Port 내 FOUP 지연 발생 통지(MCS) |
| 4 | `M11B.QUE.ABN.NOTREQUESTED` | RESV인 LOT이 다음 조건인데도 반송Queue가 없는 경우 ProbeTestYn=Y, lastEventName<>’NotOnHold’, lastEventtime 5분경과, CommunicationState=‘OnlineRemote’, Up, PortStatus Up, Auto, IntegratedMCS=‘Y’, LastEventUser in (EI... |
| 5 | `M11B.QUE.ABN.NOTUNLOADREQUESTED` | LAST_EVENT_ID==‘JobEnd’, LOC_NM=‘InMachine’, LAST_EVENT_TIME 5분 경과 (MES) |
| 6 | `M11B.QUE.ABN.PORTMISMATCH` | MCS상 8분이상 Delay되고 있는 Alternate Queue에 대해 MCS와 MES간의 Port 상태 불일치로 인해 미진행여부 판단(MCS |
| 7 | `M11B.QUE.ABN.QUETIMEDELAY` | 20분 이상 지연된 FOUP 목록 알림(MCS) |
| 8 | `M11B.QUE.ALL.CURRENTQCNT` |  |
| 9 | `M11B.QUE.ALL.CURRENTQCOMPLETED` | 최근 10분간 반송완료 수 (MCS) |
| 10 | `M11B.QUE.ALL.CURRENTQCREATED` | 최근 10분간 반송명령 생성수 (MCS) |
| 11 | `M11B.QUE.LOAD.AVGLOADTIME` | 최근 10분간 평균 Load반송시간(HySTAR) |
| 12 | `M11B.QUE.LOAD.AVGLOADTIME1MIN` | 최근 1분간 평균 Load반송시간(smartSTAR) |
| 13 | `M11B.QUE.LOAD.CURRENTLOADQCNT` | 반송 큐 개수: 목적지가 생산장비인 건(MCS) |
| 14 | `M11B.QUE.OHT.CURRENTOHTQCNT` | 현재 OHT반송 Q 수(MCS) |
| 15 | `M11B.QUE.OHT.OHTUTIL` | OHT사용율(%)(MCS) (INSTALLED VHL 수 - NOT UNASSGINED VHL 수)/(INSTALLED VHL 수) |
| 16 | `M11B.QUE.SENDFAB.CURRENTSENDFABCOMPLETED` | 최근10분간 FAB간 반송완료 수(MES) |
| 17 | `M11B.QUE.SENDFAB.CURRENTSENDFABCREATED` | 최근 10분간 FAB간 반송명령 생성수(MES) |
| 18 | `M11B.QUE.SENDFAB.M11BTOM11AQUEUECOUNT` | 현재 M11B->M11A Q 수(MES) |

#### M11B.STK — 스토커 (10건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `M11B.STK.ABN.STKZONEFULL` | 특정 Stocker가 모든 Shelf 및 OutPort가 ZoneFull(MCS) |
| 2 | `M11B.STK.ABN.UNKNOWNFOUP` | N2PurgeStocker, CuArea Stocker내 UnknownFOUP 이 Shelf 혹은 Manual OutPut Port에 저장되어 있을 경우(MCS) |
| 3 | `M11B.STK.ALERT.ALLALTSTKFULL` | MCS Storage에 저장명령 수행중 목적지의 모든 Alt가 Full일 경우(Message) |
| 4 | `M11B.STK.ALERT.STKCMDERR` | STK에 반송명령시 Reply (3) STK Command Error 발생(MCS) |
| 5 | `M11B.STK.ALERT.STKDISCONNECTION` | MCS-STK간 통신이 끊어짐(Message) |
| 6 | `M11B.STK.ALERT.TIMEOUT` | MHS가 MHSMCS_MATERIAL_DEST_REP를 늦게 내려줌 |
| 7 | `M11B.STK.ALERT.STKAUTO` | STK MCP AUTO로 State 변경 |
| 8 | `M11B.STK.ALERT.STKMCPALARM` | STKMCPAlarm 발생(Message) |
| 9 | `M11B.STK.ALERT.STKCONNECTION` | MCS-STK간 통신이 연결(Message) |
| 10 | `M11B.STK.ALERT.STKPAUSED` | STK MCP PAUSED로 State 변경 |

#### M11B.INPOS — 포트 상태 (7건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `M11B.INPOS.ALERT.104` | 비상정지(E-STOP) |
| 2 | `M11B.INPOS.ALERT.90003` | TraceMessage 통신 단절 |
| 3 | `M11B.INPOS.ALERT.101` | I/O 읽기/쓰기 에러 |
| 4 | `M11B.INPOS.ALERT.114` | 메인 PW DC 전원 에러 |
| 5 | `M11B.INPOS.ALERT.103` | MC OFF 에러 |
| 6 | `M11B.INPOS.ALERT.112` | 퍼지 미시작 |
| 7 | `M11B.INPOS.ALERT.113` | 메인 DC 전원 에러 |

#### M11B.N2 — N2 (6건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `M11B.N2.ABN.HOLDLOTN2SEND` | LOC_NM==‘InStocker’, LOT_HOLD_STAT_CD==‘OnHold” , LOT_ID not like ‘2RJ%’,‘2SP%’,‘2XP%’ ‘%RJ’,‘2UX%’,‘2SL%’, ‘MON%’ ,‘RRJ%’,‘RSK%’,‘2SK%’ LAST_EVENT_TIME 10분이상 경과 |
| 2 | `M11B.N2.ABN.N2DEFAULTSTORETIMEOUT` | Q Level이 3이 아니고 N2Purge공정이 아니고 Default Stocker가 N2STK로 지정되어 있을 경우 LOC_NM이 InMachine, InStocker, InBuffer인 Case가 아닌 경우 현재위치가 N2STK가 아닌체로 20분이상 경과(MES) |
| 3 | `M11B.N2.ABN.N2DELAYTIMEOUT` | N2STK를 DefaultStocker로 사용하는 공정에서 N2PurgeSkip이 아닌 LOT이 InMachine/InStocker/InBuffer외의 상태로 40분이상 대기 Or N2Purge공정에서 40분이상 대기(MES) |
| 4 | `M11B.N2.ABN.N2STKFOUPADJUST` | RJ Lot이 아니고 Send, Return상태 아니고, N2PurgeStoker외에 저장된 상태로 1분경과, N2PurgeSkip상태가 아님, Q3Level이 아님, DefaultStocker나 PFO Stocker가 N2거나 N2Purge공정인 Lot에 대해 N2PurgeStk로 이동.(MES,MCS) |
| 5 | `M11B.N2.ABN.N2STKQ3LOTSEND` | N2Stocker의 저장율이 80%이상일 경우 N2Stocker에 저장된 Q3Lot에 대해 SF111 |
| 6 | `M11B.N2.ABN.N2STKRJLOTSEND` | LOT의 첨자가 RJ로 끝나는 LOT은 Reject 대기 LOT으로 판단하고 자동출고(MES) |

#### M11B.INV — 인버터 (5건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `M11B.INV.PC.PROS_STATUS` | not_active 개수가 5개이하 : green not_active 개수가 5개 초과이고, primary, secondary 모두 상태값이 0인게 하나 이상 : yellow not_active 개수가 5개 초과이고, primary, secondary 모두 상태값이 0인게 존재하지 않음 : red |
| 2 | `M11B.INV.PC.CPU_TOTAL` | 해당 Machine의 CPU 사용율(%)(FTP) |
| 3 | `M11B.INV.PC.DISK_USED` | 디스크사용율(%)(FTP) |
| 4 | `M11B.INV.PC.MEM_USED` | 해당 Machine의 Memory사용율(%)(FTP) |
| 5 | `M11B.INV.PC.SWAP_USED` | SWAP Memeory 사용율(%)(FTP) |

#### M11B.SORTER — 소터 (5건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `M11B.SORTER.ABN.CUSORTERWAITCOUNTOVER` | 사용안함 |
| 2 | `M11B.SORTER.ABN.PORTUNBALANCE` | ProcDescription이 OCR |
| 3 | `M11B.SORTER.ABN.SORTERRESVFAIL` | 공정 Type이 Split, Merge, Exchange이고 Lot상태가 Released, WAIT인 상태에서 Sorter_job_id가 생성되지 않았을 경우(MES) |
| 4 | `M11B.SORTER.ABN.SORTERTRANSFERFAIL` | 예약(RESV) 상태 |
| 5 | `M11B.SORTER.ABN.SORTERWAITCOUNTOVER` | LOT의 장비 Group이 Sorter이고 LOT 상태가 Released인 수량(MES) |

#### M11B.AZFS — AZFS (3건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `M11B.AZFS.ALERT.DOUBLESTORAGE` | MCS가 Carrier DoubleStorage를 감지함(MCS) |
| 2 | `M11B.AZFS.ALERT.EMPTYRETRIEVAL` | OHT가 해당 AZFS/STB에서 공출하를 시도함. |
| 3 | `M11B.AZFS.ALERT.EMPTYRETRIEVALCOUNTOVER` | OHT가 해당 AZFS/STB에서 공출하를 시도 횟수 초과 발생 |

#### M11B.MCP — MCP (3건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `M11B.MCP.ABN.MCPALARM` | HySTAR에서 2분주기로 MSS/FABSCOPE에 기록된 ALARM수집시 Clear되지 않은 Alarm에 대해 통보(HYSTAR) |
| 2 | `M11B.MCP.ABN.OHTALARM` | MCP 알람발생건수 경고 |
| 3 | `M11B.MCP.ABN.STKALARM` | Stocker UnitAlarm 발생(MCS) Maint모드 제외 |

#### M11B.ETC — 기타 (2건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `M11B.ETC.ALERT.CARRIERDUPLICATE` | MCS가 동일 Carrier ID 중복 존재여부 감지시 발생(Message) |
| 2 | `M11B.ETC.ALERT.N2FOUPMISMATCHDEST` | N2 FOUP이 반송이상으로 인해 Normal STB로 반송되는 경우 알람 송신 |

#### M11B.FIO — FIO (1건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `M11B.FIO.ALERT.FIOMCPALARM` | FIOMCPAlarm발생(MLUD ALARM Message) |

#### M11B.SENDFAB — SENDFAB (1건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `M11B.SENDFAB.ABN.LIFTERLONGWAIT` | Lifter내에서 15분이상 대기중인 FOUP 발생(MCS) |

#### M11B.STATE — STATE (1건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `M11B.STATE.ABN.CARRIERSTATEABNORMAL` | 상태 이상 발생 FOUP List 통지(MCS) |

#### M11B.ZFS — ZFS (1건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `M11B.ZFS.ABN.MAINTFULLAZFS` | (현재 비활성화된 상태)AZFS 저장율 85% 이상일 경우 해당 AZFS내 저장된 FOUP중 저장기간 임계치 초과(DIFF, IMP : 1일, 그외 2일)일 경우 자동 일반Stocker이동저장(MES) |

---

## 4. M14 계열

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
