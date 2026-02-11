# QUWA FAB 컬럼 사전
> **구조**: `{FAB}.{Category}.{SubCategory}.{Metric}`
> **영문 Description → 한글 번역 완료**

---

### CJPRB_WLP3 (35건)

#### CJPRB_WLP3.STRATE — 저장 현황 (20건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `CJPRB_WLP3.STRATE.ALL.FABSTORAGECAPACITY` | FAB 전체저장용량 (smartSTAR) ReservedLocationCount 포함, Down된 장비 제외 |
| 2 | `CJPRB_WLP3.STRATE.ALL.FABSTORAGECOUNT` | FAB 저장수 (smartSTAR) ReservedLocationCount 포함, Down된 장비 제외 |
| 3 | `CJPRB_WLP3.STRATE.ALL.FABSTORAGERATIO` | FAB 저장율 (smartSTAR) ReservedLocationCount 포함, Down된 장비 제외 |
| 4 | `CJPRB_WLP3.STRATE.ALL.FABSTORAGESERVICECAPA` | FAB 저장가능용량 (smartSTAR) ReservedLocationCount 포함, Down된 장비 제외 |
| 5 | `CJPRB_WLP3.STRATE.N2.STORAGECAPACITY` | N2 전체저장용량 (smartSTAR) |
| 6 | `CJPRB_WLP3.STRATE.N2.STORAGECOUNT` | N2 저장수 (smartSTAR) |
| 7 | `CJPRB_WLP3.STRATE.N2.STORAGERATIO` | N2 저장율 (smartSTAR) N2 STK로 이동중인 Queue 개수 포함, Down인 장비 제외 |
| 8 | `CJPRB_WLP3.STRATE.N2.STORAGESERVICECAPA` | N2 저장가능용량 (smartSTAR) |
| 9 | `CJPRB_WLP3.STRATE.NONN2.STORAGECAPACITY` | NONN2 전체저장용량 (smartSTAR) |
| 10 | `CJPRB_WLP3.STRATE.NONN2.STORAGECOUNT` | NONN2 저장수 (smartSTAR) |
| 11 | `CJPRB_WLP3.STRATE.NONN2.STORAGERATIO` | NONN2 저장율 (smartSTAR) |
| 12 | `CJPRB_WLP3.STRATE.NONN2.STORAGESERVICECAPA` | NONN2 저장가능용량 (smartSTAR) |
| 13 | `CJPRB_WLP3.STRATE.FOUP.NONN2STORAGECAPACITY` | NONN2 전체저장용량 (smartSTAR) |
| 14 | `CJPRB_WLP3.STRATE.FOUP.NONN2STORAGECOUNT` | NONN2 저장수 (smartSTAR) |
| 15 | `CJPRB_WLP3.STRATE.FOUP.NONN2STORAGERATIO` | NONN2 저장율 (smartSTAR) |
| 16 | `CJPRB_WLP3.STRATE.FOUP.NONN2STORAGESERVICECAPA` | NONN2 저장가능용량 (smartSTAR) |
| 17 | `CJPRB_WLP3.STRATE.RINGCST.STORAGECAPACITY` | RINGCST 전체저장용량 (smartSTAR) |
| 18 | `CJPRB_WLP3.STRATE.RINGCST.STORAGECOUNT` | RINGCST 저장수 (smartSTAR) |
| 19 | `CJPRB_WLP3.STRATE.RINGCST.STORAGERATIO` | RINGCST 저장율 (smartSTAR) |
| 20 | `CJPRB_WLP3.STRATE.RINGCST.STORAGESERVICECAPA` | RINGCST 저장가능용량 (smartSTAR) |

#### CJPRB_WLP3.QUE — 반송 큐 (15건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `CJPRB_WLP3.QUE.ALL.CURRENTQCNT` |  |
| 2 | `CJPRB_WLP3.QUE.ALL.CURRENTQCOMPLETED` | 최근 10분간 반송명령 완료수(MCS) |
| 3 | `CJPRB_WLP3.QUE.ALL.CURRENTQCREATED` | 최근10분간 Q생성 수(MCS) |
| 4 | `CJPRB_WLP3.QUE.LOAD.AVGLOADTIME` | 최근 10분간 평균 Load반송시간(HySTAR) |
| 5 | `CJPRB_WLP3.QUE.LOAD.AVGLOADTIME1MIN` | 최근 1분간 평균 Load반송시간(smartSTAR) |
| 6 | `CJPRB_WLP3.QUE.LOAD.CURRENTLOADQCNT` | 반송 큐 개수: 목적지가 생산장비인 건(MCS) |
| 7 | `CJPRB_WLP3.QUE.OHT.CURRENTOHTQCNT` | 현재 OHT반송 Q 수(MCS) |
| 8 | `CJPRB_WLP3.QUE.ABN.N2STOCKERDELAY` | 60분이상 N2Purge Stocker Port 내 FOUP 지연 발생 통지(MCS) |
| 9 | `CJPRB_WLP3.QUE.ABN.QUETIMEDELAY` | 20분 이상 지연된 FOUP 목록 알림(MCS) |
| 10 | `CJPRB_WLP3.QUE.OHT.OHTUTIL_400` | OHT사용율(%)(MCS): (INSTALLED VHL 수 - NOT UNASSGINED VHL 수)/(INSTALLED VHL 수) |
| 11 | `CJPRB_WLP3.QUE.OHT.CURRENTOHTQCNT_400` | 현재 OHT반송 Q 수(MCS) |
| 12 | `CJPRB_WLP3.QUE.LOAD.AVGFOUPLOADTIME` | 최근 10분간 FOUP 평균 Load반송시간(smartSTAR) |
| 13 | `CJPRB_WLP3.QUE.LOAD.AVGRINGCSTLOADTIME` | 최근 10분간 FOUP 평균 Load반송시간(smartSTAR) |
| 14 | `CJPRB_WLP3.QUE.OHT.CURRENTOHTQCNT_3F_RINGCST` | 현재 OHT반송 Q 수(MCS) 3F RINGCST |
| 15 | `CJPRB_WLP3.QUE.OHT.CURRENTOHTQCNT_3F_FOUP` | 현재 OHT반송 Q 수(MCS) 3F FOUP |

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
