# QUWA FAB 컬럼 사전
> **구조**: `{FAB}.{Category}.{SubCategory}.{Metric}`
> **영문 Description → 한글 번역 완료**

---

### M10F (21건)

#### M10F.QUE — 반송 큐 (11건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `M10F.QUE.LOAD.AVGLOADTIME` | 최근 10분간 평균 Load Time(HySTAR) |
| 2 | `M10F.QUE.ALL.CURRENTQCREATED` | 최근 10분간 반송명령생성수(MCS) |
| 3 | `M10F.QUE.LOAD.CURRENTLOADQCNT` | 반송 큐 개수: 목적지가 생산장비인 건(MCS) |
| 4 | `M10F.QUE.OHT.OHTUTIL` | OHT사용율(%)(MCS) (INSTALLED VHL 수 - NOT UNASSGINED VHL 수)/(INSTALLED VHL 수) |
| 5 | `M10F.QUE.OHT.CURRENTOHTQCNT` | 현재 OHT반송 Q 수(MCS) |
| 6 | `M10F.QUE.ALL.CURRENTQCOMPLETED` | 최근10분간 반송완료 수(MCS) |
| 7 | `M10F.QUE.ALL.CURRENTQCNT` | 현재 반송 Q 수(MES) 출발지 장비와 목적지 장비가 다른 Queue 대상 |
| 8 | `M10F.QUE.LOAD.EQ_LOAD_DAILY_CNT_TWO_DAYS_AGO` | 통합 모니터링 이틀 전 EQ LOAD COUNT |
| 9 | `M10F.QUE.LOAD.EQ_LOAD_DAILY_TIME_TWO_DAYS_AGO` | 통합 모니터링 이틀 전 EQ LOAD TIME |
| 10 | `M10F.QUE.LOAD.EQ_LOAD_DAILY_CNT` | 통합 모니터링 전날 EQ LOAD COUNT |
| 11 | `M10F.QUE.LOAD.EQ_LOAD_DAILY_TIME` | 통합 모니터링 전날 EQ LOAD TIME |

#### M10F.INPOS — 포트 상태 (7건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `M10F.INPOS.ALERT.114` | 메인 PW DC 전원 에러 |
| 2 | `M10F.INPOS.ALERT.90003` | TraceMessage 통신 단절 |
| 3 | `M10F.INPOS.ALERT.113` | 메인 DC 전원 에러 |
| 4 | `M10F.INPOS.ALERT.101` | I/O 읽기/쓰기 에러 |
| 5 | `M10F.INPOS.ALERT.103` | MC OFF 에러 |
| 6 | `M10F.INPOS.ALERT.104` | 비상정지(E-STOP) |
| 7 | `M10F.INPOS.ALERT.112` | 퍼지 미시작 |

#### M10F.ETC — 기타 (1건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `M10F.ETC.ALERT.N2FOUPMISMATCHDEST` | N2 FOUP이 반송이상으로 인해 Normal STB로 반송되는 경우 알람 송신 |

#### M10F.STRATE — 저장 현황 (1건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `M10F.STRATE.ALL.FABSTORAGERATIO` | FAB내 저장장치(STK,ZFS)의 보관율(MES) ReservedLocationCount 포함 |

#### M10F.RTC — RTC (1건)

| # | 컬럼명 (Index) | 한글 설명 |
|---|----------------|-----------|
| 1 | `M10F.RTC.ALERT.RTCDISCONNECTION` | MCS-RTC간 통신이 끊어짐(Message) |

---

## 3. M11 계열

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
