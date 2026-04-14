# OHT XSOHS CSV 데이터 분석 방법론

## 1. 원본 데이터 구조

### 1.1 CSV 파일 형식

| 컬럼 | 설명 |
|------|------|
| `_host` | 호스트 (빈 값) |
| `_id` | 레코드 고유 ID (역순 번호) |
| `_table` | 테이블명 (`oht_raw_m14a`) |
| `_time` | 타임스탬프 (예: `2026-04-14 16:59:59.979+0900`) |
| `line` | OHT 메시지 본문 (콤마 구분 문자열) |

`line` 필드의 첫 번째 값이 메시지 유형(1/2/3/4)을 결정한다.
파일은 **역시간순** (최신→과거) 정렬되어 있다.

---

## 2. 메시지 유형별 구조

### 2.1 Type 1 - 시스템 상태 (5 fields)

```
1,OHT,ON-LINE,AUTO,NOALARMS
```

| 인덱스 | 필드 | 값 예시 |
|--------|------|--------|
| 0 | MessageId | 1 |
| 1 | McpName | OHT |
| 2 | 연결 상태 | ON-LINE |
| 3 | 운전 모드 | AUTO |
| 4 | 알람 상태 | NOALARMS |

### 2.2 Type 2 - VHL_STATE_REPORT (21 fields) - **핵심 데이터**

```
2,OHT,V00066,6,1,0000,1,13307,9,13308,4,4,6PDT1068,14049,00000000,0000,4ANZ55-202,4ALFF701_AI35,50,0,0
```

| 인덱스 | 필드명 | 타입 | 설명 | Enum 매핑 |
|--------|--------|------|------|-----------|
| 0 | MessageId | int | 메시지 유형 (=2) | - |
| 1 | McpName | str | MCP 이름 (=OHT) | - |
| 2 | VehicleId | str | 차량 ID (V=운반, R=예비) | - |
| 3 | **State** | int | 차량 상태 | 아래 참조 |
| 4 | **IsFull** | int | 적재 여부 (0=빈차, 1=적재) | - |
| 5 | ErrorCode | str | 에러 코드 (0000=정상) | - |
| 6 | IsOnline | int | 온라인 여부 (1=온라인) | - |
| 7 | **CurrentAddress** | int | 현재 위치 노드 번호 | - |
| 8 | **Distance** | int | 다음 노드까지 거리 (100mm 단위) | - |
| 9 | **NextAddress** | int | 다음 위치 노드 번호 | - |
| 10 | **RunCycle** | int | 실행 사이클 | 아래 참조 |
| 11 | **VhlCycle** | int | 차량 사이클 | 아래 참조 |
| 12 | CarrierId | str | 캐리어(FOUP) ID | - |
| 13 | Destination | int | 목적지 노드 번호 | - |
| 14 | EMState | str | EM 상태 비트 플래그 | - |
| 15 | GroupId | str | 그룹 ID | - |
| 16 | **SourcePort** | str | 출발 스테이션 | - |
| 17 | **DestPort** | str | 도착 스테이션 | - |
| 18 | **Priority** | int | 우선순위/속도 (0,50,70,80,90,99) | - |
| 19 | **DetailState** | int | 상세 상태 | 아래 참조 |
| 20 | RunDistance | int | 누적 주행 거리 | - |

#### State Enum (필드 3)

| 코드 | 이름 | 설명 |
|------|------|------|
| 1 | RUN | 운전 중 |
| 2 | STOP | 정지 중 |
| 3 | ABNORMAL | 상태 이상 |
| 4 | MANUAL | 수동 조치 |
| 5 | REMOVING | 분리/제거 중 |
| 6 | OBS_BZ_STOP | OBS-STOP/BZ-STOP (앞차 감지 정지) |
| 7 | JAM | 정체 |
| 8 | HT_STOP | HT-STOP |
| 9 | E84_TIMEOUT | E84 타임아웃 |

#### RunCycle Enum (필드 10)

| 코드 | 이름 | 설명 |
|------|------|------|
| 0 | NONE | 사이클 없음 |
| 1 | POSITION_DETECT | 위치 확인 |
| 2 | MOVING | 이동 중 |
| 3 | ACQUIRE | 물품 집어올리기 (Load) |
| 4 | DEPOSIT | 물품 내려놓기 (Unload) |
| 5 | SAMPLING | 샘플링 |
| 25 | WHEELDRIVE | 바퀴 주행 |

#### VhlCycle Enum (필드 11)

| 코드 | 이름 | 설명 |
|------|------|------|
| 0 | NONE | 실행 사이클 없음 |
| 1 | MOVING | 이동 중 |
| 2 | ACQUIRE_MOVING | 픽업 이동 중 |
| 3 | ACQUIRING | 픽업 수행 중 |
| 4 | DEPOSIT_MOVING | 하차 이동 중 |
| 5 | DEPOSITING | 하차 수행 중 |
| 6 | MAINT_MOVING | 유지보수 이동 |
| 7 | WAITING | 대기 |

#### DetailState Enum (필드 19)

| 코드 | 이름 | 설명 |
|------|------|------|
| 0 | NONE | 없음 (작업 수행 중) |
| 1 | WAIT | 대기 |
| 2 | STAGE_WAIT | 스테이지 대기 |
| 3 | STANDBY_WAIT | 스탠바이 대기 |
| 101 | MOVING | Idle 이동 중 |
| 103 | STAGE_MOVING | 스테이지 이동 |
| 105 | BALANCE_MOVING | 밸런싱 이동 (Idle 재배치) |

### 2.3 Type 3 - IN SERVICE (8 fields)

```
3,OHT,IN SERVICE,4KSR3201_2,4PDT4984,2,4ANZ26-244,V00600
```

| 인덱스 | 필드명 | 설명 |
|--------|--------|------|
| 0 | MessageId | 메시지 유형 (=3) |
| 1 | McpName | OHT |
| 2 | Action | 항상 "IN SERVICE" |
| 3 | Station | 서비스 진입 스테이션 |
| 4 | CarrierId | 캐리어 ID (빈 값 가능) |
| 5 | Field5 | 분류값 (빈/1/2) |
| 6 | DestStation | 목적지 스테이션 (빈 값 가능) |
| 7 | VehicleId | 차량 ID (빈 값 가능) |

### 2.4 Type 4 - MTL (6 fields)

```
4,OHT,60,MTL001,1,0000
```
극소수(5건) 발생. MTL(Material) 관련 시스템 메시지.

---

## 3. 분석 방법론

### 3.1 데이터 개요
- CSV 전체 행수 카운트
- `_time` 필드의 최소/최대값으로 시간 범위 산출
- `line` 첫 번째 필드(MessageId)별 빈도 집계

### 3.2 Fleet 현황
- VehicleId 접두어로 분류: `V`=운반차량, `R`=예비차량
- `set()`으로 고유 차량 수 산출
- 시간대별(HH) 활성 차량: `_time`에서 시 추출 → 시간별 unique VehicleId 집합

### 3.3 차량 상태 분석
- Type 2 레코드의 V-Vehicle만 대상
- State(필드3), RunCycle(필드10), VhlCycle(필드11), DetailState(필드19) 각각 빈도 집계
- 조합 분석: `State/RunCycle/VhlCycle` 조합별 빈도 → 주요 운전 패턴 파악

### 3.4 가동률 / 적재율 계산

**가동률 (Active Ratio)**:
```
가동률(%) = (DetailState가 101~106이 아닌 레코드 수) / (해당 차량 전체 레코드 수) × 100
```
- DetailState 101(MOVING), 102~106: Idle 상태로 분류
- DetailState 0(NONE), 1(WAIT) 등: Active 상태로 분류

**적재율 (Loaded Ratio)**:
```
적재율(%) = (IsFull=1 레코드 수) / (해당 차량 전체 레코드 수) × 100
```

차량별 계산 후 전체 통계(평균, 중앙값, 표준편차, 분포) 산출.

### 3.5 속도 분석
- Priority(필드18) 값을 속도로 사용 (0, 50, 70, 80, 90, 99)
- 전체 빈도 분포 및 State별 교차 분석
- State=6(OBS_BZ_STOP) 시 속도 분포 → 앞차 감속 패턴 간접 확인

### 3.6 운반 사이클 분석

**사이클 정의**: 하나의 완전한 운반 작업
```
Idle → Dispatch(빈차 이동) → Pickup(적재) → Transport(운반) → Deliver(하차) → Idle
```

**판별 로직**:
1. 차량별 시간순 정렬 (파일이 역순이므로 reverse)
2. `RunCycle=3(ACQUIRE) + VhlCycle=2(ACQUIRE_MOVING) + IsFull=0` → 사이클 시작 (Dispatch)
3. `RunCycle=4(DEPOSIT) + VhlCycle=4(DEPOSIT_MOVING) + IsFull=1` → Pickup 완료
4. `RunCycle=2(MOVING)/0(NONE) + IsFull=0` → Deliver 완료 (사이클 종료)
5. 사이클 시간 = 종료 시각 - 시작 시각 (10초~30분 범위 필터)

**구간별 소요시간**:
- Dispatch→Pickup: 빈차 이동 시간
- Pickup→Deliver: 적재 운반 시간

**Load/Unload 시간**:
- RunCycle이 3(ACQUIRE) 또는 4(DEPOSIT)인 연속 구간의 지속 시간
- 1초~120초 범위 필터

### 3.7 스테이션 분석
- SourcePort(필드16), DestPort(필드17) 빈도 집계
- OD(Origin-Destination) 쌍: (SourcePort, DestPort) 조합별 빈도
- Zone 분류: 스테이션 이름에서 `4` 다음 영문자 접두어 추출 (예: `4ALF...` → `ALF`)

### 3.8 시간대별 패턴
- `_time`에서 시(HH), 10분 단위(HH:M0) 추출
- 전체 메시지 유형 대상 빈도 집계
- 피크 시간대 식별

### 3.9 이상 상태 분석
- State=6(OBS_BZ_STOP), 7(JAM), 8(HT_STOP), 9(E84_TIMEOUT) 빈도 및 관련 차량수
- State=1(RUN)이면서 속도=0인 이벤트: 정상 운전 중 정지 → 앞차/합류점 대기 추정
- 시간대별 분포

### 3.10 Type 3 (IN SERVICE) 분석
- 총 건수, 캐리어/차량 정보 포함 비율
- 스테이션별 서비스 진입 빈도 → 고빈도 스테이션 = 물류 허브 추정

### 3.11 월드 모델 파라미터
- 속도 프로파일: 속도값별 비율 및 용도 추정
- Idle 패턴: DetailState=101(일반 idle 이동) vs 105(밸런싱 재배치)
- 8가지 월드 모델 항목별 데이터 검증 가능 여부 평가

---

## 4. 스크립트 사용법

```bash
# 기본 실행
python OHS/analyze_oht_xsohs.py

# 경로 지정 실행
python OHS/analyze_oht_xsohs.py [CSV경로] [출력리포트경로]

# 예시
python OHS/analyze_oht_xsohs.py OHS/XSOHS_extracted/raw.csv OHS/XSOHS_ANALYSIS_REPORT.md
```

### 의존성
- Python 3.6+
- 표준 라이브러리만 사용 (csv, statistics, collections, datetime)
- 외부 패키지 불필요

### 참조 소스
- `OHT2/layout/real_oht_parser.py`: Enum 매핑 원본 (VHL_STATE_MAP, RUN_CYCLE_MAP, VHL_CYCLE_MAP, VHL_DET_STATE_MAP)
- `NODE/MAGI/data_poi/oht_preprocessor.py`: 분석 패턴 참조
