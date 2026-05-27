# AWS_IDC v4.1 통합 이벤트 예측 컬럼 — 전체 269개 (8개 FAB 영역)
> MAIN_TS.TXT (263개) + v3.1 M16_PKT/WT (6개) = **269개** 누락 없이 정리
> 미션: M16HUB + M14 + M14B + M16A + M16B + M16 + M16_PKT + M16_WT 통합 예측
---
## 0. 영역별 개요
| 영역 | 컬럼 수 | 역할 |
|---|---|---|
| **M16HUB** | 104개 | 중심 허브 (3F) — M14↔M16 물류 연결 |
| **M14** | 41개 | M14A (3F) — CNV로 HUB 연결, M10A 와 LFT |
| **M14B** | 41개 | M14B (7F) — LFT(4ABLD) 로 HUB 연결 |
| **M16A** | 39개 | M16A (6F) — LFT(6ABL) 로 HUB 연결 |
| **M16B** | 25개 | M16B (10F) — M16A 경유 HUB |
| **M16** | 11개 | M16 SFAB — M16↔M14/M10 FAB간 반송 |
| **M16_PKT** | 4개 | M16_PKT 브릿지 |
| **M16_WT** | 4개 | M16_WT 브릿지 (M16EUV 연결) |
| **합계** | **269개** | — |

---

## 1. M16HUB 영역 (104개)

### M16HUB-4분 이상 SLA (3개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M16HUB.QUE.ALL.TRANSPORT4MINOVERCNT` | - | 4분이상반송이력COUNT |
| 2 | `M16HUB.QUE.ALL.TRANSPORT4MINOVERRATIO` | - | 4분 이상 반송 이력 RATIO |
| 3 | `M16HUB.QUE.ALL.TRANSPORT4MINOVERTIMEAVG` | - | 4분이상반송이력TIME AVG |

### M16HUB-FAB간 반송 (1개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M16HUB.QUE.ALL.FABTRANSJOBCNT` | - | M16HUB 경유하는 FAB간 반송 JOBTREND |

### M16HUB-HUB 내 JOB (2개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M16HUB.QUE.ALL.CURRENT_M16A_3F_JOB` | - | HUBROOM 내 JOB |
| 2 | `M16HUB.QUE.ALL.CURRENT_M16A_3F_JOB_2` | - | HUBROOM 내 JOB |

### M16HUB-HUB 내부 CMD (6개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M16HUB.QUE.ALL.3F_CMD` | - | HUB 내부에서 이동중인 OHT CMD |
| 2 | `M16HUB.QUE.CNV.3F_TO_M14A_CNV_AI_CMD` | - | HUB 내부에서 이동중인 OHT CMD |
| 3 | `M16HUB.QUE.LFT.3F_TO_M14B_LFT_AI_CMD` | - | HUB 내부에서 이동중인 OHT CMD |
| 4 | `M16HUB.QUE.LFT.3F_TO_M16A_LFT_AI_CMD` | - | HUB 내부에서 이동중인 OHT CMD |
| 5 | `M16HUB.QUE.MLUD.3F_TO_M16A_MLUD_AI_CMD` | - | HUB 내부에서 이동중인 OHT CMD |
| 6 | `M16HUB.QUE.STB.3F_TO_M16A_3F_STB_CMD` | - | HUB 내부에서 이동중인 OHT CMD |

### M16HUB-HUB 출구 (5개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M16HUB.QUE.ALL.3F_TO_3F_MLUD_JOB` | - | HUBROOM에서 나가고 있거나 나갈 예정인 반송 JOB |
| 2 | `M16HUB.QUE.ALL.3F_TO_M14A_3F_JOB` | - | HUBROOM에서 나가고 있거나 나갈 예정인 반송 JOB |
| 3 | `M16HUB.QUE.ALL.3F_TO_M14B_7F_JOB` | - | HUBROOM에서 나가고 있거나 나갈 예정인 반송 JOB |
| 4 | `M16HUB.QUE.ALL.3F_TO_M16A_2F_JOB` | - | HUBROOM에서 나가고 있거나 나갈 예정인 반송 JOB |
| 5 | `M16HUB.QUE.ALL.3F_TO_M16A_6F_JOB` | - | HUBROOM에서 나가고 있거나 나갈 예정인 반송 JOB |

### M16HUB-MAXCAPA (운영자 변수) (3개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M16HUB.QUE.CNV.3F_CNV_MAXCAPA` | - | HUBROOM에서 나가기 위한 Bridge 장비의 Input MaxCapa |
| 2 | `M16HUB.QUE.LFT.3F_LFT_MAXCAPA` | - | HUBROOM에서 나가기 위한 Bridge 장비의 Input MaxCapa |
| 3 | `M16HUB.QUE.LFT.3F_M14BLFT_MAXCAPA` | - | HUBROOM에서 나가기 위한 Bridge 장비의 Input MaxCapa |

### M16HUB-OHT 가동/큐 (2개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M16HUB.QUE.OHT.CURRENTOHTQCNT` | - | 현재 OHT반송 Q 수(MCS) |
| 2 | `M16HUB.QUE.OHT.OHTUTIL` | - | OHT사용율(%)(MCS) |

### M16HUB-리프터 TOTAL 큐 (10개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M16HUB.LFT.6ABL0111.TOTAL_CURRENTQCNT` | 개수 (int) | 6ABL0111 리프터 TOTAL Q 수 |
| 2 | `M16HUB.LFT.6ABL0112.TOTAL_CURRENTQCNT` | 개수 (int) | 6ABL0112 리프터 TOTAL Q 수 |
| 3 | `M16HUB.LFT.6ABL0121.TOTAL_CURRENTQCNT` | 개수 (int) | 6ABL0121 리프터 TOTAL Q 수 |
| 4 | `M16HUB.LFT.6ABL0122.TOTAL_CURRENTQCNT` | 개수 (int) | 6ABL0122 리프터 TOTAL Q 수 |
| 5 | `M16HUB.LFT.6ABL6011.TOTAL_CURRENTQCNT` | 개수 (int) | 6ABL6011 리프터 TOTAL Q 수 |
| 6 | `M16HUB.LFT.6ABL6012.TOTAL_CURRENTQCNT` | 개수 (int) | 6ABL6012 리프터 TOTAL Q 수 |
| 7 | `M16HUB.LFT.6ABL6021.TOTAL_CURRENTQCNT` | 개수 (int) | 6ABL6021 리프터 TOTAL Q 수 |
| 8 | `M16HUB.LFT.6ABL6022.TOTAL_CURRENTQCNT` | 개수 (int) | 6ABL6022 리프터 TOTAL Q 수 |
| 9 | `M16HUB.LFT.6ABL6031.TOTAL_CURRENTQCNT` | 개수 (int) | 6ABL6031 리프터 TOTAL Q 수 |
| 10 | `M16HUB.LFT.6ABL6032.TOTAL_CURRENTQCNT` | 개수 (int) | 6ABL6032 리프터 TOTAL Q 수 |

### M16HUB-리프터 방향별 큐 (56개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M16HUB.LFT.6ABL0111.2F_TO_3F_CURRENTQCNT` | - | 현재 2F→3F Q 수 |
| 2 | `M16HUB.LFT.6ABL0111.2F_TO_6F_CURRENTQCNT` | - | 현재 2F→6F Q 수 |
| 3 | `M16HUB.LFT.6ABL0111.3F_TO_2F_CURRENTQCNT` | - | 현재 3F→2F Q 수 |
| 4 | `M16HUB.LFT.6ABL0111.3F_TO_6F_CURRENTQCNT` | - | 현재 3F→6F Q 수 |
| 5 | `M16HUB.LFT.6ABL0111.6F_TO_2F_CURRENTQCNT` | - | 현재 6F→2F Q 수 |
| 6 | `M16HUB.LFT.6ABL0111.6F_TO_3F_CURRENTQCNT` | - | 현재 6F→3F Q 수 |
| 7 | `M16HUB.LFT.6ABL0112.2F_TO_3F_CURRENTQCNT` | - | 현재 2F→3F Q 수 |
| 8 | `M16HUB.LFT.6ABL0112.2F_TO_6F_CURRENTQCNT` | - | 현재 2F→6F Q 수 |
| 9 | `M16HUB.LFT.6ABL0112.3F_TO_2F_CURRENTQCNT` | - | 현재 3F→2F Q 수 |
| 10 | `M16HUB.LFT.6ABL0112.3F_TO_6F_CURRENTQCNT` | - | 현재 3F→6F Q 수 |
| 11 | `M16HUB.LFT.6ABL0112.6F_TO_2F_CURRENTQCNT` | - | 현재 6F→2F Q 수 |
| 12 | `M16HUB.LFT.6ABL0112.6F_TO_3F_CURRENTQCNT` | - | 현재 6F→3F Q 수 |
| 13 | `M16HUB.LFT.6ABL0121.2F_TO_3F_CURRENTQCNT` | - | 현재 2F→3F Q 수 |
| 14 | `M16HUB.LFT.6ABL0121.2F_TO_6F_CURRENTQCNT` | - | 현재 2F→6F Q 수 |
| 15 | `M16HUB.LFT.6ABL0121.3F_TO_2F_CURRENTQCNT` | - | 현재 3F→2F Q 수 |
| 16 | `M16HUB.LFT.6ABL0121.3F_TO_6F_CURRENTQCNT` | - | 현재 3F→6F Q 수 |
| 17 | `M16HUB.LFT.6ABL0121.6F_TO_2F_CURRENTQCNT` | - | 현재 6F→2F Q 수 |
| 18 | `M16HUB.LFT.6ABL0122.2F_TO_6F_CURRENTQCNT` | - | 현재 2F→6F Q 수 |
| 19 | `M16HUB.LFT.6ABL0122.3F_TO_2F_CURRENTQCNT` | - | 현재 3F→2F Q 수 |
| 20 | `M16HUB.LFT.6ABL0122.3F_TO_6F_CURRENTQCNT` | - | 현재 3F→6F Q 수 |
| 21 | `M16HUB.LFT.6ABL0122.6F_TO_2F_CURRENTQCNT` | - | 현재 6F→2F Q 수 |
| 22 | `M16HUB.LFT.6ABL0122.6F_TO_3F_CURRENTQCNT` | - | 현재 6F→3F Q 수 |
| 23 | `M16HUB.LFT.6ABL6011.2F_TO_3F_CURRENTQCNT` | - | 현재 2F→3F Q 수 |
| 24 | `M16HUB.LFT.6ABL6011.2F_TO_6F_CURRENTQCNT` | - | 현재 2F→6F Q 수 |
| 25 | `M16HUB.LFT.6ABL6011.3F_TO_2F_CURRENTQCNT` | - | 현재 3F→2F Q 수 |
| 26 | `M16HUB.LFT.6ABL6011.6F_TO_2F_CURRENTQCNT` | - | 현재 6F→2F Q 수 |
| 27 | `M16HUB.LFT.6ABL6011.6F_TO_3F_CURRENTQCNT` | - | 현재 6F→3F Q 수 |
| 28 | `M16HUB.LFT.6ABL6012.2F_TO_3F_CURRENTQCNT` | - | 현재 2F→3F Q 수 |
| 29 | `M16HUB.LFT.6ABL6012.2F_TO_6F_CURRENTQCNT` | - | 현재 2F→6F Q 수 |
| 30 | `M16HUB.LFT.6ABL6012.3F_TO_2F_CURRENTQCNT` | - | 현재 3F→2F Q 수 |
| 31 | `M16HUB.LFT.6ABL6012.3F_TO_6F_CURRENTQCNT` | - | 현재 3F→6F Q 수 |
| 32 | `M16HUB.LFT.6ABL6012.6F_TO_2F_CURRENTQCNT` | - | 현재 6F→2F Q 수 |
| 33 | `M16HUB.LFT.6ABL6012.6F_TO_3F_CURRENTQCNT` | - | 현재 6F→3F Q 수 |
| 34 | `M16HUB.LFT.6ABL6021.2F_TO_3F_CURRENTQCNT` | - | 현재 2F→3F Q 수 |
| 35 | `M16HUB.LFT.6ABL6021.2F_TO_6F_CURRENTQCNT` | - | 현재 2F→6F Q 수 |
| 36 | `M16HUB.LFT.6ABL6021.3F_TO_2F_CURRENTQCNT` | - | 현재 3F→2F Q 수 |
| 37 | `M16HUB.LFT.6ABL6021.3F_TO_6F_CURRENTQCNT` | - | 현재 3F→6F Q 수 |
| 38 | `M16HUB.LFT.6ABL6021.6F_TO_2F_CURRENTQCNT` | - | 현재 6F→2F Q 수 |
| 39 | `M16HUB.LFT.6ABL6021.6F_TO_3F_CURRENTQCNT` | - | 현재 6F→3F Q 수 |
| 40 | `M16HUB.LFT.6ABL6022.2F_TO_3F_CURRENTQCNT` | - | 현재 2F→3F Q 수 |
| 41 | `M16HUB.LFT.6ABL6022.2F_TO_6F_CURRENTQCNT` | - | 현재 2F→6F Q 수 |
| 42 | `M16HUB.LFT.6ABL6022.3F_TO_2F_CURRENTQCNT` | - | 현재 3F→2F Q 수 |
| 43 | `M16HUB.LFT.6ABL6022.3F_TO_6F_CURRENTQCNT` | - | 현재 3F→6F Q 수 |
| 44 | `M16HUB.LFT.6ABL6022.6F_TO_2F_CURRENTQCNT` | - | 현재 6F→2F Q 수 |
| 45 | `M16HUB.LFT.6ABL6022.6F_TO_3F_CURRENTQCNT` | - | 현재 6F→3F Q 수 |
| 46 | `M16HUB.LFT.6ABL6031.2F_TO_3F_CURRENTQCNT` | - | 현재 2F→3F Q 수 |
| 47 | `M16HUB.LFT.6ABL6031.2F_TO_6F_CURRENTQCNT` | - | 현재 2F→6F Q 수 |
| 48 | `M16HUB.LFT.6ABL6031.3F_TO_2F_CURRENTQCNT` | - | 현재 3F→2F Q 수 |
| 49 | `M16HUB.LFT.6ABL6031.3F_TO_6F_CURRENTQCNT` | - | 현재 3F→6F Q 수 |
| 50 | `M16HUB.LFT.6ABL6031.6F_TO_2F_CURRENTQCNT` | - | 현재 6F→2F Q 수 |
| 51 | `M16HUB.LFT.6ABL6031.6F_TO_3F_CURRENTQCNT` | - | 현재 6F→3F Q 수 |
| 52 | `M16HUB.LFT.6ABL6032.2F_TO_3F_CURRENTQCNT` | - | 현재 2F→3F Q 수 |
| 53 | `M16HUB.LFT.6ABL6032.3F_TO_2F_CURRENTQCNT` | - | 현재 3F→2F Q 수 |
| 54 | `M16HUB.LFT.6ABL6032.3F_TO_6F_CURRENTQCNT` | - | 현재 3F→6F Q 수 |
| 55 | `M16HUB.LFT.6ABL6032.6F_TO_2F_CURRENTQCNT` | - | 현재 6F→2F Q 수 |
| 56 | `M16HUB.LFT.6ABL6032.6F_TO_3F_CURRENTQCNT` | - | 현재 6F→3F Q 수 |

### M16HUB-저장률 (2개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M16HUB.STRATE.ALL.FABSTORAGERATIO` | % (float) | FAB 내 저장장치 (STK+ZFS) 저장율 (Reserved 포함  Down 제외) |
| 2 | `M16HUB.STRATE.STB.3F_STORAGE_UTIL` | - | HUBROOM 잔여 STORAGE PERCENT |

### M16HUB-지연/이상 알람 (2개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M16HUB.QUE.ABN.QUETIMEDELAY` | - | FOUP List notification with delay of more than 20 minutes(MCS) |
| 2 | `M16HUB.SORTER.ABN.SORTERWAITCOUNTOVER` | - | LOT의 장비 Group이 Sorter이고 Group ID가 G-ASR-A01이 아니며 LOT 상태가 Released인 수량(MES) |

### M16HUB-큐 기본 (10개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M16HUB.CNV.SENDFAB.TO_M14A_CURRENTQCNT` | - | 현재 M16HUB->M14A Q 수 |
| 2 | `M16HUB.LFT.SENDFAB.TO_M14B_CURRENTQCNT` | - | 현재 M16HUB->M14B Q 수 |
| 3 | `M16HUB.LFT.SENDFAB.TO_M16A_CURRENTQCNT` | - | 현재 M16HUB->M16A Q 수 |
| 4 | `M16HUB.LFT.SENDFAB.TO_M16E_CURRENTQCNT` | - | 현재 M16HUB->M16E Q 수 |
| 5 | `M16HUB.QUE.ALL.CURRENTQCNT` | - | 현재 반송 QUEUE 수 |
| 6 | `M16HUB.QUE.ALL.CURRENTQCREATED` | - | 최근10분간 Q생성 수(MCS) |
| 7 | `M16HUB.QUE.ALL.M16HUBTOM14MANUAL_CURRENTQCNT` | - | M16 -> BRIDGE MLUD |
| 8 | `M16HUB.QUE.M14TOM16.MESCURRENTQCNT` | 개수 (int) | M14 → M16 브릿지 이동 반송 Q 수 (MES) |
| 9 | `M16HUB.QUE.M16TOM14A.MESCURRENTQCNT` | - | M16 -> M14A 브릿지 이동 반송 Q |
| 10 | `M16HUB.QUE.M16TOM14B.MESCURRENTQCNT` | - | M16 -> M14B 브릿지 이동 반송 Q |

### M16HUB-평균 시간 (2개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M16HUB.QUE.TIME.AVGTOTALTIME` | - | 최근 10분간 평균 TOTAL반송시간(HySTAR) |
| 2 | `M16HUB.QUE.TIME.AVGTOTALTIME1MIN` | 분 (float) | 최근 1분 평균 TOTAL 반송시간 (smartSTAR) |

---

## 2. M14 영역 (41개)

### M14-4분 이상 SLA (3개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M14.QUE.ALL.TRANSPORT4MINOVERCNT` | 개수 (int) | 4분이상반송이력COUNT |
| 2 | `M14.QUE.ALL.TRANSPORT4MINOVERRATIO` | 개수 (int) | 4분 이상 반송 이력 RATIO |
| 3 | `M14.QUE.ALL.TRANSPORT4MINOVERTIMEAVG` | 개수 (int) | 4분이상반송이력TIME AVG |

### M14-CNV (M14A→M16) (2개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M14.QUE.CNV.M14ATOM16ACURRNETQCNT` | - | M14A -> M16A 반송 Q 수 |
| 2 | `M14.QUE.CNV.M14ATOM16CURRNETQCNT` | - | M14A -> M16 2F/6F/10F |

### M14-CNV 남/북측 (8개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M14.QUE.CNV.NORTHCNVTOM14TIME` | - | 최근 10분간 평균 CNV 4AFC3301->M14 시간 |
| 2 | `M14.QUE.CNV.NORTHCNVTOM14TIME1MIN` | - | 최근 1분간 평균 CNV 4AFC3301->M14 시간 |
| 3 | `M14.QUE.CNV.NORTHM14TOCNVTIME` | - | 최근 10분간 평균 CNV M14->4AFC3301 시간 |
| 4 | `M14.QUE.CNV.NORTHM14TOCNVTIME1MIN` | - | 최근 1분간 평균 CNV M14->4AFC3301 시간 |
| 5 | `M14.QUE.CNV.SOUTHCNVTOM14TIME` | - | 최근 10분간 평균 CNV 4AFC3201->M14 시간 |
| 6 | `M14.QUE.CNV.SOUTHCNVTOM14TIME1MIN` | - | 최근 1분간 평균 CNV 4AFC3201->M14 시간 |
| 7 | `M14.QUE.CNV.SOUTHM14TOCNVTIME` | - | 최근 10분간 평균 CNV M14->4AFC3201 시간 |
| 8 | `M14.QUE.CNV.SOUTHM14TOCNVTIME1MIN` | - | 최근 1분간 평균 CNV M14->4AFC3201 시간 |

### M14-HUB 인플로 (2개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M14.QUE.ALL.3F_TO_HUB_JOB` | - | HUBROOM으로 들어가고 있거나 들어갈 예정인 반송 JOB |
| 2 | `M14.QUE.ALL.3F_TO_HUB_JOB_ALT` | - | HUBROOM으로 들어가고 있거나 들어갈 예정인 반송 JOB 중 ALTERNATED 인 JOB |

### M14-M16 SFAB (1개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M14.QUE.SFAB.SENDTOM16` | - | M14->M16 SFAB SEND 반송 QUEUE |

### M14-MAXCAPA (운영자 변수) (1개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M14.QUE.CNV.3F_CNV_MAXCAPA` | - | HUBROOM으로 들어가기 위한 Bridge 장비의 Input MaxCapa |

### M14-OHT CMD (1개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M14.QUE.OHT.3F_TO_HUB_CMD` | - | HUB 외부에서 Bridge 장비 AI로 이동중인 OHT CMD |

### M14-OHT 가동/큐 (1개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M14.QUE.OHT.OHTUTIL` | - | OHT사용율(%)(MCS) |

### M14-OHT 상태 (3개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M14.OHT.STATECNT.ABNORMAL` | 개수 (int) | 이상 상태 OHT 수 |
| 2 | `M14.OHT.STATECNT.CONGESTED` | 개수 (int) | 정체 상태 OHT 수 |
| 3 | `M14.OHT.STATECNT.HTSTOP` | 개수 (int) | HT-STOP (정체) 상태 OHT 수 |

### M14-지연/이상 알람 (4개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M14.QUE.ABN.CARRIERTRANSDELAY` | - | 10분이상 Carrier 반송지연 발생 Queue |
| 2 | `M14.QUE.ABN.QUETIMEDELAY` | - | FOUP List notification with delay of more than 20 minutes(MCS) |
| 3 | `M14.SORTER.ABN.CUSORTERWAITCOUNTOVER` | - | LOT의 장비 Group이 Cu Sorter이고 Group ID가 G-KSR-A01이 아니며 LOT 상태가 Released인 수량(MES) |
| 4 | `M14.SORTER.ABN.SORTERWAITCOUNTOVER` | - | LOT의 장비 Group이 Sorter이고 Group ID가 G-ASR-A01이 아니며 LOT 상태가 Released인 수량(MES) |

### M14-큐 기본 (13개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M14.CNV.SENDFAB.TO_M16HUB_CURRENTQCNT` | - | 현재 M14A->M16HUB Q 수 |
| 2 | `M14.QUE.ALL.CURRENTQCNT` | - | 현재 반송 QUEUE 수 |
| 3 | `M14.QUE.ALL.CURRENTQCOMPLETED` | - | 최근 10분간 반송명령 완료수(MCS) |
| 4 | `M14.QUE.ALL.CURRENTQCREATED` | - | 최근10분간 Q생성 수(MCS) |
| 5 | `M14.QUE.ALL.TOTALCNVCURRENTQCNT` | - | TOTAL 반송 Q 수 |
| 6 | `M14.QUE.CNV.ALLTONORTHCNVCURRENTQCNT` | - | ALL -> 4AFC_M16 OUT(4AFC3301) 반송 Q 수 |
| 7 | `M14.QUE.CNV.ALLTOSOUTHCNVCURRENTQCNT` | - | ALL -> 4AFC_M16 OUT(4AFC3201) 반송 Q 수 |
| 8 | `M14.QUE.CNV.M14ATONORTHCURRENTQCNT` | - | M14 북측 CNV Q 수 |
| 9 | `M14.QUE.CNV.M14ATOSOUTHCURRENTQCNT` | - | M14 남측 CNV Q 수 |
| 10 | `M14.QUE.CNV.NORTHCNVTOALLCURRENTQCNT` | - | 4AFC3301 A_IN -> ALL 반송 Q 수 |
| 11 | `M14.QUE.CNV.NORTHCURRENTQCNT` | - | 현재 반송 CNV QUEUE 수 |
| 12 | `M14.QUE.CNV.SOUTHCNVTOALLCURRENTQCNT` | - | 4AFC3201 A_IN -> ALL 반송 Q 수 |
| 13 | `M14.QUE.CNV.SOUTHCURRENTQCNT` | - | 현재 반송 CNV QUEUE 수 |

### M14-평균 시간 (2개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M14.QUE.LOAD.AVGLOADTIME` | - | 최근 10분간 평균 Load반송시간(HySTAR) |
| 2 | `M14.QUE.LOAD.AVGLOADTIME1MIN` | - | 최근 1분간 평균 Load반송시간(smartSTAR) |

---

## 3. M14B 영역 (41개)

### M14B-HUB 인플로 (2개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M14B.QUE.ALL.7F_TO_HUB_JOB` | 개수 (int) | 7F → HUB 반송 JOB |
| 2 | `M14B.QUE.ALL.7F_TO_HUB_JOB_ALT` | 개수 (int) | 7F → HUB 반송 JOB (ALT) |

### M14B-M14B → M16A LFT (1개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M14B.QUE.LFT.M14BTOM16ACURRNETQCNT` | - | M14B -> M16A 반송 Q 수 |

### M14B-OHT 가동/큐 (2개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M14B.QUE.OHT.CURRENTOHTQCNT` | - | 현재 OHT반송 Q 수(MCS) |
| 2 | `M14B.QUE.OHT.OHTUTIL` | % (float) | M14B OHT 사용률 |

### M14B-SendFab 송출 (1개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M14B.QUE.SENDFAB.VERTICALQUEUECOUNT` | - | 현재 M14B->M14A Q 수(STA) |

### M14B-리프터 TOTAL 큐 (6개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M14B.LFT.4ABLD111.TOTAL_CURRENTQCNT` | - | 현재 4ABLD111 TOTAL Q 수 |
| 2 | `M14B.LFT.4ABLD112.TOTAL_CURRENTQCNT` | - | 현재 4ABLD112 TOTAL Q 수 |
| 3 | `M14B.LFT.4ABLD121.TOTAL_CURRENTQCNT` | - | 현재 4ABLD121 TOTAL Q 수 |
| 4 | `M14B.LFT.4ABLD122.TOTAL_CURRENTQCNT` | 개수 (int) | M14B 4ABLD122 리프터 Q 수 |
| 5 | `M14B.LFT.4ABLD131.TOTAL_CURRENTQCNT` | - | 현재 4ABLD131 TOTAL Q 수 |
| 6 | `M14B.LFT.4ABLD132.TOTAL_CURRENTQCNT` | - | 현재 4ABLD132 TOTAL Q 수 |

### M14B-리프터 방향별 큐 (12개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M14B.LFT.4ABLD111.4F_TO_7F_CURRENTQCNT` | - | 현재 4F→7F Q 수 |
| 2 | `M14B.LFT.4ABLD111.7F_TO_4F_CURRENTQCNT` | - | 현재 7F→4F Q 수 |
| 3 | `M14B.LFT.4ABLD112.4F_TO_7F_CURRENTQCNT` | - | 현재 4F→7F Q 수 |
| 4 | `M14B.LFT.4ABLD112.7F_TO_4F_CURRENTQCNT` | - | 현재 7F→4F Q 수 |
| 5 | `M14B.LFT.4ABLD121.4F_TO_7F_CURRENTQCNT` | - | 현재 4F→7F Q 수 |
| 6 | `M14B.LFT.4ABLD121.7F_TO_4F_CURRENTQCNT` | - | 현재 7F→4F Q 수 |
| 7 | `M14B.LFT.4ABLD122.4F_TO_7F_CURRENTQCNT` | - | 현재 4F→7F Q 수 |
| 8 | `M14B.LFT.4ABLD122.7F_TO_4F_CURRENTQCNT` | - | 현재 7F→4F Q 수 |
| 9 | `M14B.LFT.4ABLD131.4F_TO_7F_CURRENTQCNT` | - | 현재 4F→7F Q 수 |
| 10 | `M14B.LFT.4ABLD131.7F_TO_4F_CURRENTQCNT` | - | 현재 7F→4F Q 수 |
| 11 | `M14B.LFT.4ABLD132.4F_TO_7F_CURRENTQCNT` | - | 현재 4F→7F Q 수 |
| 12 | `M14B.LFT.4ABLD132.7F_TO_4F_CURRENTQCNT` | - | 현재 7F→4F Q 수 |

### M14B-지연/이상 알람 (5개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M14B.QUE.ABN.AOTRANSDELAY` | 개수 (int) | 10분 이상 AO Port 대기 FOUP 알람 |
| 2 | `M14B.QUE.ABN.QUETIMEDELAY` | - | FOUP List notification with delay of more than 20 minutes(MCS) |
| 3 | `M14B.SORTER.ABN.CUSORTERWAITCOUNTOVER` | - | LOT의 장비 Group이 Cu Sorter이고 Group ID가 G-KSR-A01이 아니며 LOT 상태가 Released인 수량(MES) |
| 4 | `M14B.SORTER.ABN.SORTERWAITCOUNTOVER` | - | LOT의 장비 Group이 Sorter이고 Group ID가 G-ASR-A01이며 LOT 상태가 Released인 수량(MES) |
| 5 | `M14B.SORTER.ABN.SORTERWAITCOUNTOVER_B01` | - | LOT의 장비 Group이 Sorter이고 Group ID가 G-ASR-B01이며 LOT 상태가 Released인 수량(MES) |

### M14B-큐 기본 (9개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M14B.LFT.SENDFAB.TO_M14A_CURRENTQCNT` | - | 현재 M14B->M14A Q 수 |
| 2 | `M14B.LFT.SENDFAB.TO_M16HUB_CURRENTQCNT` | - | 현재 M14B->M16HUB Q 수 |
| 3 | `M14B.QUE.ALL.CURRENTQCNT` | - | 현재 반송 QUEUE 수 |
| 4 | `M14B.QUE.ALL.CURRENTQCOMPLETED` | - | 최근 10분간 반송명령 완료수(MCS) |
| 5 | `M14B.QUE.ALL.CURRENTQCREATED` | - | 최근10분간 Q생성 수(MCS) |
| 6 | `M14B.QUE.LFT.ALLTOLFTCURRENTQCNT` | - | ALL -> 4ABLB_4F OUT 반송 Q수 |
| 7 | `M14B.QUE.LFT.LFTTOALLCURRENTQCNT` | - | 4ABLD% AI -> ALL 반송 Q 수 |
| 8 | `M14B.QUE.LFT.LTFTOALLCURRENTQCNT` | - | 4ABLD% AI -> ALL 반송 Q수 |
| 9 | `M14B.QUE.LOAD.CURRENTLOADQCNT` | - | Transport Queue Count : destination is production equipment(MCS) |

### M14B-평균 시간 (3개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M14B.QUE.LOAD.AVGLOADTIME` | - | 최근 10분간 평균 Load반송시간(HySTAR) |
| 2 | `M14B.QUE.LOAD.AVGLOADTIME1MIN` | - | 최근 1분간 평균 Load반송시간(smartSTAR) |
| 3 | `M14B.QUE.TIME.AVGTOTALTIME1MIN` | 분 (float) | M14B 최근 1분 평균 반송시간 |

---

## 4. M16A 영역 (39개)

### M16A-4분 이상 SLA (3개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M16A.QUE.ALL.TRANSPORT4MINOVERCNT` | - | 4분이상반송이력COUNT |
| 2 | `M16A.QUE.ALL.TRANSPORT4MINOVERRATIO` | - | 4분 이상 반송 이력 RATIO |
| 3 | `M16A.QUE.ALL.TRANSPORT4MINOVERTIMEAVG` | - | 4분이상반송이력TIME AVG |

### M16A-CNV (M16→M14) (4개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M16A.QUE.CNV.M16ATOM14ACURRNETQCNT` | - | M16A -> M14A 반송 Q 수 |
| 2 | `M16A.QUE.CNV.M16ATOM14BCURRNETQCNT` | - | M16A -> M14 반송 Q 수 |
| 3 | `M16A.QUE.CNV.M16TOM14ACURRNETQCNT` | - | M16 2F/6F/10F |
| 4 | `M16A.QUE.CNV.M16TOM14BCURRNETQCNT` | - | M16 2F/6F/10F |

### M16A-HUB 인플로 (4개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M16A.QUE.ALL.2F_TO_HUB_JOB` | - | HUBROOM으로 들어가고 있거나 들어갈 예정인 반송 JOB |
| 2 | `M16A.QUE.ALL.2F_TO_HUB_JOB_ALT` | - | HUBROOM으로 들어가고 있거나 들어갈 예정인 반송 JOB 중 ALTERNATED 인 JOB |
| 3 | `M16A.QUE.ALL.6F_TO_HUB_JOB` | - | HUBROOM으로 들어가고 있거나 들어갈 예정인 반송 JOB |
| 4 | `M16A.QUE.ALL.6F_TO_HUB_JOB_ALT` | - | HUBROOM으로 들어가고 있거나 들어갈 예정인 반송 JOB 중 ALTERNATED 인 JOB |

### M16A-M16A 내부 흐름 (2개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M16A.QUE.ALL.2F_TO_6F_JOB` | - | M16 6F <-> 2F JOB들 |
| 2 | `M16A.QUE.ALL.6F_TO_2F_JOB` | - | M16 6F <-> 2F JOB들 |

### M16A-MAXCAPA (운영자 변수) (2개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M16A.QUE.LFT.2F_LFT_MAXCAPA` | - | HUBROOM으로 들어가기 위한 Bridge 장비의 Input MaxCapa |
| 2 | `M16A.QUE.LFT.6F_LFT_MAXCAPA` | - | HUBROOM으로 들어가기 위한 Bridge 장비의 Input MaxCapa |

### M16A-OHT CMD (2개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M16A.QUE.OHT.2F_TO_HUB_CMD` | - | HUB 외부에서 Bridge 장비 AI로 이동중인 OHT CMD |
| 2 | `M16A.QUE.OHT.6F_TO_HUB_CMD` | - | HUB 외부에서 Bridge 장비 AI로 이동중인 OHT CMD |

### M16A-OHT 가동/큐 (2개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M16A.QUE.OHT.CURRENTOHTQCNT` | - | 현재 OHT반송 Q 수(MCS) |
| 2 | `M16A.QUE.OHT.OHTUTIL` | - | OHT사용율(%)(MCS) |

### M16A-지연/이상 알람 (4개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M16A.QUE.ABN.QUETIMEDELAY` | - | FOUP List notification with delay of more than 20 minutes(MCS) |
| 2 | `M16A.SORTER.ABN.CUSORTERWAITCOUNTOVER` | - | LOT의 장비 Group이 Cu Sorter이고 Group ID가 G-KSR-A01이 아니며 LOT 상태가 Released인 수량(MES) |
| 3 | `M16A.SORTER.ABN.SORTERTRANSFERFAIL` | - | RESV |
| 4 | `M16A.SORTER.ABN.SORTERWAITCOUNTOVER` | - | LOT의 장비 Group이 Sorter이고 Group ID가 G-ASR-A01이 아니며 LOT 상태가 Released인 수량(MES) |

### M16A-큐 기본 (14개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M16A.LFT.SENDFAB.TO_M16B_CURRENTQCNT` | - | 현재 M16A->M16B Q 수 |
| 2 | `M16A.LFT.SENDFAB.TO_M16E_CURRENTQCNT` | - | 현재 M16A->M16E Q 수 |
| 3 | `M16A.LFT.SENDFAB.TO_M16HUB_CURRENTQCNT` | - | 현재 M16A->M16HUB Q 수 |
| 4 | `M16A.QUE.ALL.CURRENTQCNT` | - | "현재 반송 QUEUE 수" |
| 5 | `M16A.QUE.ALL.CURRENTQCOMPLETED` | - | 최근 10분간 반송명령 완료수(MCS) |
| 6 | `M16A.QUE.ALL.CURRENTQCREATED` | - | 최근10분간 Q생성 수(MCS) |
| 7 | `M16A.QUE.CNV.ALLTONORTHCNVCURRENTQCNT` | - | ALL -> 4AFC_M14 OUT(4AFC3301) 반송 Q 수 |
| 8 | `M16A.QUE.CNV.ALLTOSOUTHCNVCURRENTQCNT` | - | ALL -> 4AFC_M14 OUT(4AFC3201) 반송 Q 수 |
| 9 | `M16A.QUE.CNV.NORTHCNVTOALLCURRENTQCNT` | - | 4AFC3301 OUT -> ALL 반송 Q 수 |
| 10 | `M16A.QUE.CNV.SOUTHCNVTOALLCURRENTQCNT` | - | 4AFC3201 OUT -> ALL 반송 Q 수 |
| 11 | `M16A.QUE.LFT.ALLTOLFTCURRENTQCNT` | - | ALL->4ABLB_7F OUT 반송 Q수 |
| 12 | `M16A.QUE.LFT.LFTTOALLCURRENTQCNT` | - | 4ABLD->ALL 반송 Q 수 |
| 13 | `M16A.QUE.LFT.LTFTOALLCURRENTQCNT` | - | 4ABLD-> ALL 반송Q수 |
| 14 | `M16A.QUE.LOAD.CURRENTLOADQCNT` | - | Transport Queue Count : destination is production equipment(MCS) |

### M16A-평균 시간 (2개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M16A.QUE.LOAD.AVGFOUPLOADTIME` | - | 최근 10분간 FOUP 평균 Load반송시간(smartSTAR) |
| 2 | `M16A.QUE.LOAD.AVGLOADTIME1MIN` | - | 최근 1분간 평균 Load반송시간(smartSTAR) |

---

## 5. M16B 영역 (25개)

### M16B-4분 이상 SLA (3개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M16B.QUE.ALL.TRANSPORT4MINOVERCNT` | - | 4분 이상 반송 이력 COUNT |
| 2 | `M16B.QUE.ALL.TRANSPORT4MINOVERRATIO` | - | 4분 이상 반송 이력 RATIO |
| 3 | `M16B.QUE.ALL.TRANSPORT4MINOVERTIMEAVG` | - | 4분 이상 반송 이력 TIME AVG |

### M16B-CNV (M16→M14) (2개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M16B.QUE.CNV.M16BTOM14ACURRNETQCNT` | - | M16A -> M14A 반송 Q 수 |
| 2 | `M16B.QUE.CNV.M16BTOM14BCURRNETQCNT` | - | M16A -> M14 반송 Q 수 |

### M16B-HUB 인플로 (1개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M16B.QUE.ALL.10F_TO_HUB_JOB` | - | HUBROOM으로 들어가고 있거나 들어갈 예정인 반송 JOB |

### M16B-OHT 가동/큐 (2개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M16B.QUE.OHT.CURRENTOHTQCNT` | - | 현재 OHT반송 Q 수(MCS) |
| 2 | `M16B.QUE.OHT.OHTUTIL` | - | OHT사용율(%)(MCS) |

### M16B-지연/이상 알람 (4개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M16B.QUE.ABN.QUETIMEDELAY` | - | FOUP List notification with delay of more than 20 minutes(MCS) |
| 2 | `M16B.SORTER.ABN.CUSORTERWAITCOUNTOVER` | - | LOT의 장비 Group이 Cu Sorter이고 Group ID가 G-KSR-A01이 아니며 LOT 상태가 Released인 수량(MES) |
| 3 | `M16B.SORTER.ABN.SORTERTRANSFERFAIL` | - | RESV |
| 4 | `M16B.SORTER.ABN.SORTERWAITCOUNTOVER` | - | LOT의 장비 Group이 Sorter이고 Group ID가 G-ASR-A01이 아니며 LOT 상태가 Released인 수량(MES) |

### M16B-큐 기본 (11개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M16B.LFT.SENDFAB.TO_M16A_CURRENTQCNT` | - | 현재 M16B->M16A Q 수 |
| 2 | `M16B.QUE.ALL.CURRENTQCNT` | - | 현재 반송 QUEUE 수 |
| 3 | `M16B.QUE.ALL.CURRENTQCOMPLETED` | - | 최근 10분간 반송명령 완료수(MCS) |
| 4 | `M16B.QUE.ALL.CURRENTQCREATED` | - | 최근10분간 Q생성 수(MCS) |
| 5 | `M16B.QUE.CNV.ALLTONORTHCNVCURRENTQCNT` | - | ALL -> 4AFC_M14 OUT(4AFC3301) 반송 Q 수 |
| 6 | `M16B.QUE.CNV.ALLTOSOUTHCNVCURRENTQCNT` | - | ALL -> 4AFC_M14 OUT(4AFC3201) 반송 Q 수 |
| 7 | `M16B.QUE.CNV.NORTHCNVTOALLCURRENTQCNT` | - | 4AFC3301 OUT -> ALL 반송 Q 수 |
| 8 | `M16B.QUE.CNV.SOUTHCNVTOALLCURRENTQCNT` | - | 4AFC3201 OUT -> ALL 반송 Q 수 |
| 9 | `M16B.QUE.LFT.ALLTOLFTCURRENTQCNT` | - | ALL->4ABLB_7F OUT 반송 Q 수 |
| 10 | `M16B.QUE.LFT.LFTTOALLCURRENTQCNT` | - | 4ABLD->ALL 반송 Q 수 |
| 11 | `M16B.QUE.LOAD.CURRENTLOADQCNT` | - | Transport Queue Count : destination is production equipment(MCS) |

### M16B-평균 시간 (2개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M16B.QUE.LOAD.AVGFOUPLOADTIME` | - | 최근 10분간 FOUP 평균 Load반송시간(smartSTAR) |
| 2 | `M16B.QUE.LOAD.AVGLOADTIME1MIN` | - | 최근 1분간 평균 Load반송시간(smartSTAR) |

---

## 6. M16 영역 (11개)

### M16-M16 SFAB (10개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M16.QUE.SFAB.COMPLETEQUEUETOTAL` | - | M16->ALL SFAB COMPLETE 반송 QUEUE |
| 2 | `M16.QUE.SFAB.COMPLETETOM10` | - | M16->M10 SFAB COMPLETE 반송 QUEUE |
| 3 | `M16.QUE.SFAB.COMPLETETOM14` | - | M16->M14 SFAB COMPLETE 반송 QUEUE |
| 4 | `M16.QUE.SFAB.RECEIVEQUEUETOTAL` | - | M16->ALL SFAB RECEIVE 반송 QUEUE |
| 5 | `M16.QUE.SFAB.RETURNQUEUETOTAL` | - | M16->ALL SFAB RETURN 반송 QUEUE |
| 6 | `M16.QUE.SFAB.RETURNTOM10` | - | M16->M10 SFAB RETURN 반송 QUEUE |
| 7 | `M16.QUE.SFAB.RETURNTOM14` | - | M16->M14 SFAB RETURN 반송 QUEUE |
| 8 | `M16.QUE.SFAB.SENDQUEUETOTAL` | - | M16->ALL SFAB SEND 반송 QUEUE |
| 9 | `M16.QUE.SFAB.SENDTOM10` | - | M16->M10 SFAB SEND 반송 QUEUE |
| 10 | `M16.QUE.SFAB.SENDTOM14` | - | M16->M14 SFAB SEND 반송 QUEUE |

### M16-큐 기본 (1개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M16.CNV.SENDFAB.TO_M16WT_CURRENTQCNT` | - | 현재 M16->M16WT Q 수 |

---

## 7. M16_PKT 영역 (4개)

### M16_PKT-OHT 가동/큐 (1개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M16_PKT.QUE.OHT.OHTUTIL` | % (float) | M16_PKT OHT 사용률 |

### M16_PKT-OHT 알람 (1개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M16_PKT.OHT.ALERT.OHTMCPALARMCNT` | 개수 (int) | M16_PKT OHT 알람 건수 |

### M16_PKT-지연/이상 알람 (1개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M16_PKT.QUE.ABN.AOTRANSDELAY` | 개수 (int) | M16_PKT AO Port 지연 FOUP |

### M16_PKT-평균 시간 (1개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M16_PKT.QUE.TIME.AVGTOTALTIME1MIN` | 분 (float) | M16_PKT 최근 1분 평균 반송시간 |

---

## 8. M16_WT 영역 (4개)

### M16_WT-OHT 가동/큐 (1개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M16_WT.QUE.OHT.OHTUTIL` | % (float) | M16_WT OHT 사용률 |

### M16_WT-OHT 알람 (1개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M16_WT.OHT.ALERT.OHTMCPALARMCNT` | 개수 (int) | M16_WT OHT 알람 건수 |

### M16_WT-지연/이상 알람 (1개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M16_WT.QUE.ABN.AOTRANSDELAY` | 개수 (int) | M16_WT AO Port 지연 FOUP |

### M16_WT-평균 시간 (1개)

| # | 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M16_WT.QUE.TIME.AVGTOTALTIME1MIN` | 분 (float) | M16_WT 최근 1분 평균 반송시간 |

---

## 9. 통합 예측 활용 관점

### 핵심 분류

| 분류 | 개수 | 핵심 의미 |
|---|---|---|
| **운영자 변수 (MAXCAPA)** | 6개 | M16HUB/M14/M16A 의 MAXCAPA — 메신저에서 자주 변경 |
| **인플로 (HUB 진입)** | 9개 | 5개 FAB → HUB 부하 추적 |
| **HUB 출구** | 5개 | HUB → 외부 막힘 감지 |
| **HUB CMD/JOB** | 9개 | HUB 내부 OHT 명령/JOB |
| **OHT 상태/알람** | 12개 | OHT 가동률/알람/상태 |
| **4분 이상 SLA** | 12개 | 영역별 정체 직접 지표 (★ 0512 case 핵심) |
| **Sorter** | 8개 | LOT 적체 (정체 보조 지표) |
| **저장률** | 5개 | FAB/HUB 저장 포화 (R-D) |
| **리프터 큐** | 96개 | TOTAL 16 + 방향별 80 (R-C' 정밀화) |
| **CNV/LFT 흐름** | 30개+ | 영역 간 큐 추적 |

### 다음 단계

1. **SQL v4.1 작성** (269개 컬럼) — 누락 0
2. **데이터 추출** (2026-01-01 ~ 2026-05-14)
3. **영역별 정상 분포 분석** → 임계값 도출
4. **통합 룰베이스 (hubroom_predictor_main.py)** 설계

---

*본 문서는 MAIN_TS.TXT (263개) + v3.1 M16_PKT/WT (6개) = 269개 컬럼을 누락 없이 정리한 것이다.*
