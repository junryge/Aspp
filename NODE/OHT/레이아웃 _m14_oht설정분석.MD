# OHT2 폴더 분석 보고서

## 개요

**대상**: SK하이닉스 M14 반도체 공장 OHT 시스템 설정 데이터

### OHT (Overhead Hoist Transport)
- 반도체 공장에서 FOUP(웨이퍼 운반 용기)를 천장 레일로 자동 운반하는 물류 시스템
- MCS(Material Control System)와 연동 운영

### MCP (Material Control Point)
- OHT 차량과 장비를 제어하는 중앙 제어 시스템

---

## 파일 구성

| 파일명 | 크기 | 설명 |
|--------|------|------|
| mcp75.cfg | ~9MB | MCP 메인 설정 파일 |
| station.dat | ~3.5MB | 스테이션 정의 파일 |
| inactive_SCH_1.dat | 378B | 비활성화된 LANE 설정 |

---

## 1. mcp75.cfg (메인 설정)

### 기본 정보

| 항목 | 값 |
|------|-----|
| 프로젝트 | HYNIX_M14 |
| 라인 | OHT |
| 맵 버전 | M14-Pro Ver.1.22.2 (2023/08/01) |
| 설정 버전 | 2025/02/07-09:09 |

### 시스템 용량

| 설정 | 값 |
|------|-----|
| 최대 장비 연결 | 4,000개 |
| 최대 차량(OHT) 연결 | 2,000대 |
| 간접 연결 장비 | 1,000개 |
| 통신 타임아웃 | 30초 |
| 상태 리포트 간격 | 10초 |

### 주요 설정 섹션 (33개)

| 섹션명 | 설명 |
|--------|------|
| [VEHICLE_MAP_INFO] | 차량 맵 버전 정보 |
| [MCP75_GLOBAL] | 글로벌 설정 (저장 경로 등) |
| [MCP75_SYSTEM_OPTION] | 시스템 옵션 |
| [MCP75_CEC] | CEC(Communication Equipment Controller) 설정 |
| [MCP75_CEC_TSC] | TSC(Transport System Controller) 설정 |
| [MCP75_CEC_GEM] | SECS/GEM 통신 설정 |
| [MCP75_SCHEDULER_GLOBAL] | 스케줄러 설정 |
| [MCP75_VEHICLE] | OHT 차량 설정 |
| [MCP75_VEHICLE_SPEED] | 차량 속도 설정 |
| [MCP75_MTL_MTS] | MTL/MTS(수직이송장치) 설정 |
| [MCP75_PIO] | PIO(Parallel I/O) 인터페이스 |
| [MCP75_BZ] | BZ(Block Zone) 설정 |
| [MCP75_BUILDING] | 빌딩/층 설정 |
| [SECS] | SECS 통신 프로토콜 |
| [EVENT_REPORT] | 이벤트 리포트 |

---

## 2. station.dat (스테이션 정의)

OHT가 FOUP를 로딩/언로딩하는 지점. 반도체 장비의 로드포트와 연결.

### 스테이션 타입별 분포

| 타입 | 개수 | 설명 |
|------|------|------|
| DUAL_ACCESS | 22,497 | 양방향 접근 가능 스테이션 |
| ZFS_RIGHT | 8,914 | ZFS 우측 스테이션 |
| ZFS_LEFT | 8,061 | ZFS 좌측 스테이션 |
| UNIVERSAL | 5,654 | 범용 스테이션 |
| ACQUIRE | 77 | 캐리어 수취 스테이션 |
| MAINTENANCE | 55 | 유지보수용 스테이션 |
| DEPOSIT | 51 | 캐리어 보관 스테이션 |
| MANUAL_ONLY | 39 | 수동 조작 전용 |
| DUMMY | 20 | 더미(테스트용) |
| MTL_SWITCHBACK | 5 | MTL 스위치백 |
| MTL_ELEVATOR | 5 | MTL 엘리베이터 |

**총 스테이션 수: 약 22,689개**

---

## 3. inactive_SCH_1.dat (비활성 LANE)

운영에서 제외된 LANE 정보 (5개)

```
[MCP75_INACTIVE]
LANE = 1293, 15121, 1, 0, 0, "", 0
LANE = 1302, 15060, 1, 0, 0, "", 0
LANE = 1465, 15178, 1, 0, 0, "", 0
LANE = 1470, 15197, 1, 0, 0, "", 0
LANE = 13178, 13179, 1, 0, 0, "", 0
```

형식: `LANE = 시작노드, 종료노드, 플래그, ...`

---

## 시스템 아키텍처

```
┌─────────────────────────────────────┐
│           MCS (Host)                │
│      Material Control System        │
└─────────────────┬───────────────────┘
                  │ SECS/GEM
                  ▼
┌─────────────────────────────────────┐
│         MCP75 (Controller)          │
│   Material Control Point - OHT Line │
│  ┌───────────┬───────────┬───────┐  │
│  │ Scheduler │ Vehicle   │Station│  │
│  │ (스케줄러)│ Mgr(차량) │Manager│  │
│  └───────────┴───────────┴───────┘  │
└─────────────────┬───────────────────┘
                  │
    ┌─────────────┼─────────────┐
    ▼             ▼             ▼
┌───────┐    ┌───────┐    ┌───────┐
│ OHT#1 │    │ OHT#2 │    │ OHT#n │
│ (차량)│    │ (차량)│    │ (차량)│
└───┬───┘    └───┬───┘    └───┬───┘
    │            │            │
    ▼            ▼            ▼
┌─────────────────────────────────────┐
│     Station (22,689개 스테이션)      │
│       로딩/언로딩 포인트             │
└─────────────────────────────────────┘
```

---

## 용어 정리

| 용어 | 설명 |
|------|------|
| OHT | Overhead Hoist Transport - 천장 물류 운반 시스템 |
| MCP | Material Control Point - 물류 제어 포인트 |
| FOUP | Front Opening Unified Pod - 웨이퍼 운반 용기 |
| SECS/GEM | 반도체 장비 통신 표준 프로토콜 |
| ZCU | Zone Control Unit - 구역 제어 장치 |
| MTL | Material Transfer Lifter - 층간 이송 장치 |
| PIO | Parallel I/O - 장비 인터페이스 |
| Lane | OHT가 이동하는 레일 구간 |
| Station | OHT가 FOUP를 적재/하역하는 지점 |

---

## 요약

| 항목 | 내용 |
|------|------|
| 대상 | SK하이닉스 M14 반도체 공장 |
| OHT 차량 | 최대 2,000대 운영 가능 |
| 스테이션 | 22,689개 관리 |
| 통신 | SECS/GEM 프로토콜로 MCS와 연동 |
| 설정 기준일 | 2025년 2월 |