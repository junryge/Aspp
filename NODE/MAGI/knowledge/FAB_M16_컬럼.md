# QUWA FAB 컬럼 사전
> **구조**: `{FAB}.{Category}.{SubCategory}.{Metric}`
> **영문 Description → 한글 번역 완료**

---

## 6. M16 계열

### M16 (33건)

#### M16.QUE — 반송 큐 (32건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `M16.QUE.LOAD.EQ_LOAD_DAILY_CNT_TWO_DAYS_AGO` | 통합 모니터링 이틀 전 EQ LOAD COUNT |
| 2 | `M16.QUE.LOAD.EQ_LOAD_DAILY_TIME_TWO_DAYS_AGO` | 통합 모니터링 이틀 전 EQ LOAD TIME |
| 3 | `M16.QUE.LOAD.EQ_LOAD_DAILY_CNT` | 통합 모니터링 전날 EQ LOAD COUNT |
| 4 | `M16.QUE.LOAD.EQ_LOAD_DAILY_TIME` | 통합 모니터링 전날 EQ LOAD TIME |
| 5 | `M16.QUE.SFAB.SENDTOR3` | M16->R3 SFAB SEND 반송 QUEUE |
| 6 | `M16.QUE.SFAB.SENDTOM14` | M16->M14 SFAB SEND 반송 QUEUE |
| 7 | `M16.QUE.SFAB.SENDTOM10` | M16->M10 SFAB SEND 반송 QUEUE |
| 8 | `M16.QUE.SFAB.SENDQUEUETOTAL` | M16->ALL SFAB SEND 반송 QUEUE |
| 9 | `M16.QUE.SFAB.RECEIVETOR3` | M16->R3 SFAB RECEIVE 반송 QUEUE |
| 10 | `M16.QUE.SFAB.RECEIVETOM14` | M16->M14 SFAB RECEIVE 반송 QUEUE |
| 11 | `M16.QUE.SFAB.RECEIVETOM10` | M16->M10 SFAB RECEIVE 반송 QUEUE |
| 12 | `M16.QUE.SFAB.RECEIVEQUEUETOTAL` | M16->ALL SFAB RECEIVE 반송 QUEUE |
| 13 | `M16.QUE.SFAB.RETURNTOR3` | M16->R3 SFAB RETURN 반송 QUEUE |
| 14 | `M16.QUE.SFAB.RETURNTOM14` | M16->M14 SFAB RETURN 반송 QUEUE |
| 15 | `M16.QUE.SFAB.RETURNTOM10` | M16->M10 SFAB RETURN 반송 QUEUE |
| 16 | `M16.QUE.SFAB.RETURNQUEUETOTAL` | M16->ALL SFAB RETURN 반송 QUEUE |
| 17 | `M16.QUE.SFAB.COMPLETETOR3` | M16->R3 SFAB COMPLETE 반송 QUEUE |
| 18 | `M16.QUE.SFAB.COMPLETETOM14` | M16->M14 SFAB COMPLETE 반송 QUEUE |
| 19 | `M16.QUE.SFAB.COMPLETETOM10` | M16->M10 SFAB COMPLETE 반송 QUEUE |
| 20 | `M16.QUE.SFAB.COMPLETEQUEUETOTAL` | M16->ALL SFAB COMPLETE 반송 QUEUE |
| 21 | `M16.QUE.ALL.M16_M10F_SENDFAB_ALL_CNT` | 통합 모니터링 전날 BRIDGE COUNT |
| 22 | `M16.QUE.ALL.M16_M14_SENDFAB_ALL_CNT` | 통합 모니터링 전날 BRIDGE COUNT |
| 23 | `M16.QUE.ALL.M16_PNT4_SENDFAB_ALL_CNT` | 통합 모니터링 전날 BRIDGE COUNT |
| 24 | `M16.QUE.ALL.M16_M10F_SENDFAB_ALL_TIME` | 통합 모니터링 전날 BRIDGE TIME |
| 25 | `M16.QUE.ALL.M16_M14_SENDFAB_ALL_TIME` | 통합 모니터링 전날 BRIDGE TIME |
| 26 | `M16.QUE.ALL.M16_PNT4_SENDFAB_ALL_TIME` | 통합 모니터링 전날 BRIDGE TIME |
| 27 | `M16.QUE.ALL.M16_M10F_SENDFAB_ALL_CNT_TWO_DAYS_AGO` | 통합 모니터링 이틀 전 BRIDGE COUNT |
| 28 | `M16.QUE.ALL.M16_M14_SENDFAB_ALL_CNT_TWO_DAYS_AGO` | 통합 모니터링 이틀 전 BRIDGE COUNT |
| 29 | `M16.QUE.ALL.M16_PNT4_SENDFAB_ALL_CNT_TWO_DAYS_AGO` | 통합 모니터링 이틀 전 BRIDGE COUNT |
| 30 | `M16.QUE.ALL.M16_M10F_SENDFAB_ALL_TIME_TWO_DAYS_AGO` | 통합 모니터링 이틀 전 BRIDGE TIME |
| 31 | `M16.QUE.ALL.M16_M14_SENDFAB_ALL_TIME_TWO_DAYS_AGO` | 통합 모니터링 이틀 전 BRIDGE TIME |
| 32 | `M16.QUE.ALL.M16_PNT4_SENDFAB_ALL_TIME_TWO_DAYS_AGO` | 통합 모니터링 이틀 전 BRIDGE TIME |

#### M16.CNV — 컨베이어 (1건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `M16.CNV.SENDFAB.TO_M16WT_CURRENTQCNT` | 현재 M16->M16WT Q 수 |

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
