# AGV / CNV UDP 인터페이스 정의서

> **데이터 출처**: `AGV_CNV/AGV_CNV_UDP_DATA.txt`
> **분석 방식**: 본 문서는 위 데이터 파일에 **실측된 값만** 기재합니다. 코드(`AgvMsgWorkerRunnable.java`, `CnvMsgWorkerRunnable.java`)에 정의된 필드 의미는 **참조용으로만 별도 표시**하며, 데이터로 확인 불가한 항목은 모두 제외합니다.

---

## 1. 데이터 파일 전체 통계 (실측)

| 항목 | 값 |
|------|----|
| 총 라인 수 | 495 |
| 패킷 수 (정상 파싱) | 496건 |
| AGV 패킷 | 155건 |
| CNV 패킷 | 341건 |
| 필드 개수 | 모든 패킷이 정확히 **20개** |
| 로그 태그 | `[ICPKT] <AGV\|CNV> 수신 패킷: <message>` |
| 구분자 | `,` (쉼표) |
| 빈 값 표현 | 연속된 쉼표 |

---

## 2. 공통 헤더 (실측)

| Idx | 관측된 값 | 비고 |
|-----|----------|------|
| 0 | `2` (496건 전부) | 단일 값 |
| 1 | `AGV` (155건) / `CNV` (341건) | 단일 값 2종 |

---

## 3. AGV 패킷 (TXT_ID=2, DEVICE_TYPE=AGV) — 155건

### 3.1 AGV 필드별 실측값 / 자리수 / 분포

> **자리수**는 데이터 파일에서 측정한 **문자 개수(min~max)** 입니다. 값이 빈 칸인 케이스는 자리수 측정에서 제외했습니다.

| Idx | 자리수 (min~max) | 채움/전체 | 실측 고유값 | 비고 |
|-----|------------------|-----------|------------|------|
| 0  | 1                | 155/155   | `2` | 고정값 |
| 1  | 3                | 155/155   | `AGV` | 고정값 |
| 2  | 8                | 155/155   | `6AAV3B01` | 고정 폭 |
| 3  | 1~4              | 155/155   | `4, 8, 10, 13, 15, 16, 22, 23, 29, 31, 32, 34, 35, 37, 41, 42, 46, 55, 57, 8131, 8134` | 21종 |
| 4  | 5~14             | 155/155   | `AGV04, AGV08, ... AGV57` (5자리) 또는 `6ARB0111-2R291`, `6ARB0312-2R231` (14자리) | |
| 5  | 1                | 155/155   | `1` | 항상 `1` |
| 6  | 1                | 155/155   | `0` 또는 `1` | |
| 7  | 1                | 155/155   | `0` 또는 `1` | |
| 8  | 2~5              | 86/155    | `19`, `25`, `46`, `47`, `130`, `407`, `12016`, `12029`, `12041`, `12104`, `19013`, `19022`, `19032`, `19035`, `19046` | 알람 코드 15종 |
| 9  | 17~32            | 86/155    | 알람 메시지 (§3.3 참조) | F8과 동시에 채워짐 |
| 10 | 3~14             | 132/155   | 3~5자리 정수(예: `446`, `2881`, `12172`) 또는 14자리 위치명(`6ARB0111-2R291`) | |
| 11 | 3~5              | 34/155    | 정수값 범위 `445`~`25591` | 채워진 34건은 모두 F8도 채워진 패킷 |
| 12 | 3~14             | 106/155   | 정수(예: `8050`, `16518`) 또는 14자리 위치명(`6ARB03ZZ-2R072`) 또는 9자리(`6B3BM615_1`) | |
| 13 | 3 또는 8         | 81/155    | 8자리 `HIRA####` / `HIRM####`, 또는 3자리 `1:1` (1건만 관측) | |
| 14 | 8                | 132/155   | 8자리 `HIRA####` / `HIRM####` | 고정 폭 |
| 15 | 1                | 109/155   | `1` (84건), `3` (24건), `5` (1건) | |
| 16 | 1                | 132/155   | `0` (86건), `1` (46건) | |
| 17 | 8                | 46/155    | `6ARB0111` (F3=8131일 때) / `6ARB0312` (F3=8134일 때) | 고정 폭 |
| 18 | 1                | 46/155    | `M` | F3=8131/8134일 때만 채워짐 |
| 19 | 4                | 46/155    | `BOTH` | F3=8131/8134일 때만 채워짐 |

### 3.2 차량 ID(F3) 분류 — 실측

| 분류 | F3 값 | F17 (BAY) | F18 | F19 |
|------|-------|-----------|-----|-----|
| 1·2자리 차량 (19종) | `4, 8, 10, 13, 15, 16, 22, 23, 29, 31, 32, 34, 35, 37, 41, 42, 46, 55, 57` | (빈 값) | (빈 값) | (빈 값) |
| 4자리 차량 (2종) | `8131` | `6ARB0111` | `M` | `BOTH` |
| 4자리 차량 (2종) | `8134` | `6ARB0312` | `M` | `BOTH` |

### 3.3 알람 코드 / 메시지 전체 (15종)

| F8 (코드) | F9 (메시지) |
|-----------|-------------|
| `19` | `1F Load Place Axis Servo   Error` |
| `25` | `2F Load Place Axis Servo   Error` |
| `46` | `Bumper Detect Alarm` |
| `47` | `1F ������ Time Out` |
| `130` | `X Axis  BW LimitError` |
| `407` | `OBS Detected Warning` |
| `12016` | `AGV ABNORMAL EXIT [Warning]` |
| `12029` | `AGV ABNORMAL EXIT [Warning]` |
| `12041` | `AGV ABNORMAL EXIT [Warning]` |
| `12104` | `AGV ABNORMAL EXIT` |
| `19013` | `Delay Move VC Warning` |
| `19022` | `Delay Move VC Warning` |
| `19032` | `Delay Move VC Warning` |
| `19035` | `Delay Move VC Warning` |
| `19046` | `Delay Move VC Warning` |

> 메시지 내 공백 폭이 불규칙(2칸/3칸 공백 포함)이며, `12104`는 `[Warning]` 접미가 없는 변형.
> 코드 `47` 메시지에 인코딩 깨진 문자(`������`)가 포함된 상태로 수신됨.

### 3.4 AGV 실측 패킷 샘플

```
2,AGV,6AAV3B01,8131,6ARB0111-2R291,1,1,1,,,6ARB0111-2R291,,6ARB03ZZ-2R072,HIRA0894,HIRA0894,,1,6ARB0111,M,BOTH
2,AGV,6AAV3B01,8134,6ARB0312-2R231,1,1,1,,,6ARB0312-2R231,,6B3BM615_1,HIRA2721,HIRA2721,,1,6ARB0312,M,BOTH
2,AGV,6AAV3B01,32,AGV32,1,1,1,19032,Delay Move VC Warning,8012,,,HIRM1620,HIRM1677,3,0,,,
2,AGV,6AAV3B01,22,AGV22,1,0,0,19022,Delay Move VC Warning,10000,,16518,,HIRA1868,1,0,,,
2,AGV,6AAV3B01,55,AGV55,1,0,0,,,,,,,,1,,,,
2,AGV,6AAV3B01,57,AGV57,1,0,0,130,X Axis  BW LimitError,17669,17668,8106,,HIRM1009,1,0,,,
2,AGV,6AAV3B01,8,AGV08,1,1,0,25,2F Load Place Axis Servo   Error,2881,,,HIRA0098,HIRM0013,3,0,,,
2,AGV,6AAV3B01,23,AGV23,1,1,1,47,1F ������ Time Out,18279,25591,175,HIRM0877,HIRM1168,1,0,,,
```

### 3.5 샘플 패킷 필드별 의미 분석 (실측 + amp DB / 코드 교차 검증)

**대상 패킷:**

```
2,AGV,6AAV3B01,8131,6ARB0111-2R291,1,1,1,,,6ARB0111-2R291,,6ARB03ZZ-2R072,HIRA0894,HIRA0894,,1,6ARB0111,M,BOTH
```

| Idx | 값 | 의미 | 의미 근거 |
|-----|----|------|-----------|
| 0 | `2` | 메시지 ID (TXT_ID) — Vehicle State Report | 코드 `MSG_ID.VHL_STATE_REPORT="2"` |
| 1 | `AGV` | 장비 타입 | 데이터 파일 1열 — 코드 정의에는 없음 |
| 2 | `6AAV3B01` | MCP_NM / EQPID (AGV 시스템 식별자) | amp DB 4개 테이블 모두 `EQPID='6AAV3B01'` 일치. `eqpid_map.txt` 기준 별칭 = `ICPKT` |
| 3 | `8131` | VHL_ID — 차량 번호 (4자리 특수 운반차량) | `amp.location.UnitID='8131'` 존재 ✓ (실측 검증) |
| 4 | `6ARB0111-2R291` | 현재 위치 식별자 | `amp.bufferlocation.HostID='6ARB0111-2R291'` 매칭 ✓ |
| 5 | `1` | (코드 정의: `FULL_IDX` — 재하 여부) | 코드 정의 기준. 본 데이터의 모든 패킷에서 항상 `1` |
| 6 | `1` | (코드 정의: `ERROR_CODE_IDX`) | 코드 정의 기준 |
| 7 | `1` | (코드 정의: `ONLINE_IDX` — 통신 상태) | 코드 정의 기준 |
| 8 | (빈 값) | 알람 코드 (없음 = 정상 운행) | F8 빈 값이면 알람 미발생 (실측: 알람 패킷에서만 채워짐) |
| 9 | (빈 값) | 알람 메시지 (없음) | F8과 동시 채움 (실측 100% 상관) |
| 10 | `6ARB0111-2R291` | 현재 번지 (`ADDRESS`) — F4와 동일 | 코드 `ADDRESS_IDX` |
| 11 | (빈 값) | 거리 (`DISTANCE`) — 알람 미발생 → 빈 값 | 실측: 정수값으로 채워진 34건은 모두 F8 알람 동반 |
| 12 | `6ARB03ZZ-2R072` | 다음 번지 (`NEXT_ADDRESS`) | `amp.bufferlocation.HostID='6ARB03ZZ-2R072'` 매칭 ✓ |
| 13 | `HIRA0894` | Carrier ID — 차량이 재하한 반송품 | 코드 `CARRIER_ID_IDX` |
| 14 | `HIRA0894` | 목적지 Carrier (`DESTINATION`) — F13과 동일 | 코드 `DESTINATION_IDX` |
| 15 | (빈 값) | E/M 상태 (`EM_STATUS`) | 코드 정의 기준 |
| 16 | `1` | 실행 사이클 (`RUN_CYCLE`) | 코드 `RUN_CYCLE_IDX` |
| 17 | `6ARB0111` | BAY 명칭 (`BAY_NM`) | `amp.bufferlocation.ZoneName` 패턴(`6ARB####`)과 일치 — 4자리 차량(8131/8134)에서만 채워짐 |
| 18 | `M` | 운전 모드 | 실측: 4자리 차량(8131/8134)에서만 `M` (109건은 빈 값). 의미 단정 불가 |
| 19 | `BOTH` | Fork 방향 (추정) | 실측: 4자리 차량에서만 `BOTH` (다른 값 미관측) |

> **근거 분류 표기**:
> - **amp DB 매칭 ✓**: `AVG/amp_map_data.json` 의 location/hostlocation/bufferlocation 테이블에서 동일 값 확인
> - **코드 정의 기준**: `AGV/AGV_JAVA/AgvMsgWorkerRunnable.java` 의 `VHL_STATE_REPORT` 인덱스 정의(+1 시프트 적용)
> - **실측**: `AGV_CNV_UDP_DATA.txt` 데이터 자체에서 직접 관측

**해석 요약 (사실 기반)**: 본 패킷은 `6AAV3B01` AGV(별칭 `ICPKT`)의 4자리 특수 운반차량 `8131`이 위치 `6ARB0111-2R291`에서 다음 위치 `6ARB03ZZ-2R072`로 진행 중이며, Carrier `HIRA0894`를 재하한 상태이고, BAY `6ARB0111`에 속함. 알람 미발생, MODE=`M`, FORK_DIR=`BOTH`.

---

## 4. CNV 패킷 (TXT_ID=2, DEVICE_TYPE=CNV) — 341건

### 4.1 CNV 필드별 실측값 / 자리수 / 분포

> **자리수**는 데이터 파일에서 측정한 **문자 개수(min~max)** 입니다. 값이 빈 칸인 케이스는 자리수 측정에서 제외했습니다.

| Idx | 자리수 (min~max) | 채움/전체 | 실측 고유값 | 비고 |
|-----|------------------|-----------|------------|------|
| 0  | 1                | 341/341   | `2` | 고정값 |
| 1  | 3                | 341/341   | `CNV` | 고정값 |
| 2  | 8~9              | 341/341   | `P4ACV5R01` (9자리), `6ACV3B01` (8자리), `6ACV3M01` (8자리) | 3종 |
| 3  | 5                | 341/341   | zero-padded 5자리 정수 (예: `10427`, `08874`, `02016`) | 고정 폭 |
| 4  | 5~21             | 341/341   | F3과 동일 5자리(327건) 또는 §4.4 특수 식별자(13/18/19/21자리) | |
| 5  | 1                | 341/341   | `1` | 항상 `1` |
| 6  | 1                | 341/341   | `0` 또는 `1` | |
| 7  | 1                | 341/341   | `0` 또는 `1` | |
| 8  | -                | 0/341     | (전체 빈 값) | **CNV 패킷 전체 알람 0건** |
| 9  | -                | 0/341     | (전체 빈 값) | **CNV 패킷 전체 알람 0건** |
| 10 | 5                | 341/341   | F3과 동일 5자리 정수 (100% 일치) | 고정 폭 |
| 11 | -                | 0/341     | (전체 빈 값) | |
| 12 | 13~21            | 137/341   | §4.3 DEST_NODE (27종) | |
| 13 | 7~8              | 172/341   | 8자리 `HIT[BDEFG]####`, `PUPD####`, `TGPD####`, `TWPD####`, 7자리 `PIN####` | |
| 14 | 8                | 93/341    | 8자리 `HIT####`, `PUPD####`, `TGPD####`, `TWPD####` 등 | 고정 폭 |
| 15 | -                | 0/341     | (전체 빈 값) | |
| 16 | 1                | 341/341   | `0` (297건), `1` (44건) | |
| 17 | 15~20            | 65/341    | §4.5 Zone (8종) | F16=`1`인 경우만 채워짐 |
| 18 | -                | 0/341     | (전체 빈 값) | |
| 19 | -                | 0/341     | (전체 빈 값) | |

### 4.2 MCP_NM(F2) 분포 (정확)

| F2 | 패킷 수 |
|----|---------|
| `P4ACV5R01` | 186 |
| `6ACV3B01` | 109 |
| `6ACV3M01` | 46 |
| **합계** | **341** |

### 4.3 DEST_NODE(F12) 전체 27종

| MCP_NM 접두 | F12 값 |
|--------------|--------|
| `6ACV3B01_*` | `6AST3B01-CO1`, `6AST3B04-CO1`, `6AST3B04-CO2`, `6AST3B06-CO1`, `6AST3B06-CO2`, `6AST3B07-CO1`, `6AST3B07-CO2`, `6AST3B08-CO2`, `6ATM3B04-CO1`, `6B3AS001-CO1`, `6B3SB010-CO1`, `6B3SC011-CO1`, `BANK1-MO1`, `TDBI-MVP` |
| `6ACV3M01_*` | `6AAT3M04-CO1`, `6AST3M03-CO2`, `6AST3M04-CO1`, `6AST3T08-CO1`, `6AST3T08-CO2`, `6M3M0302-CO1`, `CZ-ATM`, `CZ-TR-EMPTY`, `MVP-TDBI` |
| `6ACV3R01_*` / `P4ACV5R01_*` | `6ACV3R01_BR-PKT`, `6ACV3R01_COF1`, `6ACV3R01_COF3`, `P4ACV5R01_BR-LFT` |

### 4.4 NODE_NAME(F4) 특수 식별자 — F3 ≠ F4 인 7종

```
6ACV3M01_6AAT3M02-CO1
6ACV3M01_6AST3M03-CI2
6ACV3M01_6AST3T08-CO2
6ACV3M01_6ASU3M01-CO1
6ACV3R01_ARRIVED01
6ACV3R01_COT2
P4ACV5R01_ARRIVED02
```

### 4.5 CONGEST_ZONE(F17) 전체 8종 (F16=1 일 때만)

```
6ACV3B01_CZ-BANK
6ACV3B01_CZ-SRT1
6ACV3B01_CZ-SRT2
6ACV3B01_CZ-SRT3
6ACV3B01_CZ-SRT5
6ACV3M01_CZ-ATM
6ACV3M01_CZ-MK-LIS
6ACV3M01_CZ-SCHEDULE
```

> `CZ-SRT4` 는 본 데이터에 미관측.

### 4.6 CNV 실측 패킷 샘플

```
2,CNV,P4ACV5R01,10427,10427,1,0,0,,,10427,,,,HITD9817,,0,,,
2,CNV,P4ACV5R01,10444,10444,1,1,0,,,10444,,6ACV3R01_BR-PKT,HITB3621,,,0,,,
2,CNV,6ACV3B01,08234,08234,1,1,0,,,08234,,6ACV3B01_6B3SC011-CO1,HITB0076,,,0,6ACV3B01_CZ-SRT3,,
2,CNV,6ACV3B01,08874,08874,1,1,0,,,08874,,,HITE9904,,,1,6ACV3B01_CZ-SRT1,,
2,CNV,P4ACV5R01,12221,P4ACV5R01_ARRIVED02,1,0,0,,,12221,,,,TGPD0841,,0,,,
2,CNV,P4ACV5R01,12237,12237,1,1,0,,,12237,,P4ACV5R01_BR-LFT,TGPD1414,TWPD4026,,0,,,
2,CNV,6ACV3M01,02614,02614,1,1,0,,,02614,,6ACV3M01_6AST3T08-CO2,HITD8292,,,0,,,
2,CNV,P4ACV5R01,10201,6ACV3R01_COT2,1,1,0,,,10201,,,HITD5425,HITF1178,,0,,,
```

### 4.7 CNV 샘플 패킷 필드별 의미 분석 (실측 + amp/config_bridge DB / 코드 교차 검증)

**대상 패킷** (P4ACV5R01 — M16-PNT4 3F 브릿지 CNV):

```
2,CNV,P4ACV5R01,12203,12203,1,1,0,,,12203,,6ACV3R01_COF1,TGPD7011,PUPD6172,,0,,,
```

| Idx | 값 | 의미 | 의미 근거 |
|-----|----|------|-----------|
| 0 | `2` | 메시지 ID (TXT_ID) — State Report | 코드 `MSG_ID` (CNV 코드 정의는 `4`=Machine, `202`=Load 등이지만 본 데이터는 `2`로만 수신) |
| 1 | `CNV` | 장비 타입 | 데이터 파일 1열 — 코드 정의에는 없음 |
| 2 | `P4ACV5R01` | EqpID — CNV 브릿지 (M16-PNT4 3F) | `amp.cnvinformation.EqpID` 일치 (859건 모두). `config_bridge.conveyor.CnvHostID` 접두 일치 |
| 3 | `12203` | LocationID (구간 고유 ID) | `amp.cnvinformation.LocationID` 와 매칭 (zero-padded 5자리 정수) |
| 4 | `12203` | 위치 식별자 — F3과 동일(일반 구간) | F3 ≠ F4 인 경우는 §4.4 특수 식별자 (예: `P4ACV5R01_ARRIVED02`, `6ACV3R01_COT2`) |
| 5 | `1` | (코드 정의: `STATE_IDX` — 상태) | 본 데이터의 모든 패킷에서 항상 `1` |
| 6 | `1` | (코드 정의 미일치 슬롯) — 실측: `0`/`1` | F3 위치에 캐리어 존재 여부 추정 (실측: F13/F14 채워진 패킷이면 대체로 `1`) |
| 7 | `0` | (코드 정의 미일치 슬롯) — 실측: `0`/`1` | |
| 8 | (빈 값) | 알람 코드 — **CNV 패킷 전체 0건** | 실측: CNV는 알람 미전송 |
| 9 | (빈 값) | 알람 메시지 | |
| 10 | `12203` | 번지 (F3과 100% 일치) | `amp.cnvinformation.LocationID` |
| 11 | (빈 값) | (CNV에서 채워진 사례 없음) | |
| 12 | `6ACV3R01_COF1` | 다음 노드 / 출력 연결 — Carrier Out From #1 | 가이드 포트 목록 `COF1` (Carrier Out From). `config_bridge.conveyor.CnvHostID` 접두 패턴 |
| 13 | `TGPD7011` | Carrier ID #1 (구간에 적재된 캐리어) | `amp.cnvinformation.CurrentCarrier` 또는 `config_bridge.conveyor.CurrentCarrier` |
| 14 | `PUPD6172` | Carrier ID #2 (직전/관련 캐리어) | 동일 컬럼 패턴. F13≠F14 인 경우 두 캐리어가 별도로 추적됨 |
| 15 | (빈 값) | (CNV에서 채워진 사례 없음) | |
| 16 | `0` | CONGEST_FLAG — 구간이 Congestion Zone 에 속하지 않음 | 실측: `1`이면 F17에 Zone명 채워짐 (44/341건) |
| 17 | (빈 값) | Congestion Zone 명 — 본 패킷은 미소속 | F16=`1`일 때 `*_CZ-SRT[1235]`, `*_CZ-BANK`, `*_CZ-ATM`, `*_CZ-MK-LIS`, `*_CZ-SCHEDULE` 8종 (§4.5) |
| 18 | (빈 값) | (CNV에서 채워진 사례 없음) | |
| 19 | (빈 값) | (CNV에서 채워진 사례 없음) | |

**해석 요약**: 본 패킷은 `P4ACV5R01` (M16-PNT4 3F 브릿지 CNV)의 구간 `12203` 에 캐리어 `TGPD7011`이 적재되어 있고, 다음 노드는 `6ACV3R01_COF1` (Carrier Out From #1) 이며, Congestion Zone 미소속 상태.

### 4.8 CNV ↔ amp/config_bridge DB 매핑 (실측 검증)

UDP 데이터의 CNV 식별자를 amp DB / config_bridge DB 테이블에 교차 검증한 결과:

| UDP 필드 | DB 매핑 | 검증 |
|----------|---------|------|
| F2 (`P4ACV5R01`) | `amp.cnvinformation.EqpID` (859건 일치) | ✅ |
| F2 (`6ACV3B01`, `6ACV3M01`, `6ACV3R01`) | 가이드 범위 외 (다른 라인) | ⚠️ |
| F3 (LocationID, 5자리) | `amp.cnvinformation.LocationID` | ✅ |
| F4 특수값 (`P4ACV5R01_ARRIVED02`, `6ACV3R01_COT2`) | `config_bridge.conveyor.CnvHostID` 의 `<EqpID>_<포트명>` 패턴 | ✅ |
| F12 (`6ACV3R01_COF1`, `_BR-PKT`, `P4ACV5R01_BR-LFT` 등) | `config_bridge.conveyor.CnvHostID` 또는 `amp.cnvinformation.HostID` | ✅ |
| F13/F14 (CARRIER) | `cnvinformation.CurrentCarrier` 또는 `conveyor.CurrentCarrier` | ✅ |
| F16=1, F17 (Zone) | `cnvinformation.ZoneName` | (Zone 가이드 외 항목) |

#### 4.8.1 가이드의 29개 포트 vs UDP 실측 매칭

| 가이드 포트명 | UDP 실측 출현 건수 |
|---------------|--------------------|
| COT1 | **0** |
| COT2 | 2 |
| COF1 | 23 |
| COF2 | **0** |
| COF3 | 2 |
| CIF1, CIF2, CIF3 | **0** |
| AIT1 ~ AIT8 | **0** |
| AIF1, AIF2, AIF3 | **0** |
| AOF1, AOF2, AOF3 | **0** |
| AIOF1 | **0** |
| MIT1, MIT2 | **0** |
| MIOT1 | **0** |
| MOT1 | **0** |
| ARRIVED01 | 1 (`6ACV3R01_ARRIVED01`) |
| ARRIVED02 | 4 (`P4ACV5R01_ARRIVED02`) |

#### 4.8.2 가이드에 없으나 UDP에 등장한 노드 패턴

| 패턴 | 출현 | 위치 |
|------|------|------|
| `6ACV3R01_BR-PKT` | 57건 | F12 (DEST_NODE) |
| `P4ACV5R01_BR-LFT` | 3건 | F12 (DEST_NODE) |
| `6ACV3B01_*` 전체 | 109건 | F2 (B 라인 - 가이드 범위 외) |
| `6ACV3M01_*` 전체 | 46건 | F2 (M 라인 - 가이드 범위 외) |

> **결론**: 가이드의 29개 포트 중 UDP에 실측된 것은 **5종(COT2, COF1, COF3, ARRIVED01, ARRIVED02)** 뿐이며, `BR-PKT`/`BR-LFT` 등 가이드 미기재 패턴이 추가로 존재합니다. 본 데이터는 P4ACV5R01 외 `6ACV3B01`, `6ACV3M01`, `6ACV3R01` 라인 트래픽도 포함하므로, 가이드의 P4ACV5R01 단일 EQPID 범위와 차이가 있음.

#### 4.8.3 CNV 가이드 미수록 / 저장소 미보유 자료

- `cnv_data_fetch.py` — 저장소 미보유 (`AVG/SQL_CHAK.PY` 의 CNV 버전 추정)
- `cnv_map_data.json` — 저장소 미보유
- `cnv_map.html` — ✅ **저장소 보유** (`/AGV_CNV/CNV/cnv_map.html`)

### 4.9 CNV 뷰어(`cnv_map.html`) 인터페이스 정의

뷰어가 로드하는 `cnv_map_data.json` 의 최상위 키 / 필드 매핑을 정리합니다.

#### 4.9.1 JSON 최상위 구조

```json
{
  "cnv":  [ ... ],   // amp.cnvinformation 859건 (P4ACV5R01 필터링)
  "port": [ ... ]    // config_bridge.conveyor 29건
}
```

#### 4.9.2 `cnv` 배열 — `amp.cnvinformation` 매핑

| JSON 키 | DB 컬럼 | 뷰어 내부 변수 | 용도 |
|---------|---------|----------------|------|
| `EqpID` | EqpID (text) | `eqp` | 설비 ID (P4ACV5R01 고정) |
| `LocationID` | LocationID (char(30)) | `loc` | 구간 고유 ID — 인덱스 키 |
| `HostID` | HostID (char(30)) | `hid` | 호스트 ID |
| `XPos` | XPos (float) | `x` | X 좌표 |
| `YPos` | YPos (float) | `y` | Y 좌표 |
| `Width` | Width (float) | `w` | 폭 |
| `Height` | Height (float) | `h` | 높이 |
| `Direction` | Direction (text) | `dir` | 방향 (`0`,`1`,`2`,`02` 등) |
| `ObjectInList` | ObjectInList (text) | `inList` | 입력 연결 (쉼표 구분 LocationID 배열로 split) |
| `ObjectOutList` | ObjectOutList (text) | `outList` | 출력 연결 (쉼표 구분 LocationID 배열로 split) |
| `CurrentCarrier` | CurrentCarrier (text) | `carrier` | 적재 캐리어 ID |
| `ZoneName` | ZoneName (text) | `zone` | 존 이름 |
| `Kind` | Kind (int) | `kind` | 오브젝트 종류 |
| `ObjectState` | ObjectState (int) | `state` | 상태 |
| `MachineState` | MachineState (int) | `mstate` | 기계 상태 |

#### 4.9.3 `port` 배열 — `config_bridge.conveyor` 매핑

| JSON 키 | DB 컬럼 | 뷰어 내부 변수 | 용도 |
|---------|---------|----------------|------|
| `CnvUnitID` | CnvUnitID (char(60)) | `uid` | 유닛 ID — 인덱스 키 |
| `CnvHostID` | CnvHostID (char(60)) | `hid` | 포트 이름 (예: `P4ACV5R01_COT1`). 라벨 표시 시 `P4ACV5R01_` 접두 제거 |
| `ZoneName` | ZoneName (char(60)) | `zone` | 존 이름 |
| `PositionX` | PositionX (int) | `x` | X 좌표 |
| `PositionY` | PositionY (int) | `y` | Y 좌표 |
| `Width` | Width (int) | `w` | 폭 |
| `Height` | Height (int) | `h` | 높이 |
| `Direction` | Direction (int) | `dir` | 방향 |
| `LinkInner` | LinkInner (char(60)) | `linkIn` | 내부 연결 (쉼표 구분 UnitID) |
| `LinkOuter` | LinkOuter (char(60)) | `linkOut` | 외부 연결 (쉼표 구분 UnitID) |
| `CurrentCarrier` | CurrentCarrier (char(60)) | `carrier` | 적재 캐리어 |
| `ObjectKind` | ObjectKind (int) | `kind` | 오브젝트 종류 |
| `IPAddress` | IPAddress (char(60)) | `ip` | IP 주소 |
| `LineType` | LineType (char(10)) | `line` | 라인 타입 (`M16`) |

#### 4.9.4 뷰어 동작 사양

| 항목 | 사양 |
|------|------|
| 파일 입력 | 파일 선택 또는 드래그앤드롭 (`.json`) |
| 좌표계 변환 | 데이터 bounds + 5% 패딩 자동 fit, `tx(x)=x*scale+ox`, `ty(y)=y*scale+oy` |
| 줌 | 마우스 휠 1.15× 스텝, 커서 위치 기준 |
| 패닝 | 마우스 드래그 |
| 레이어 | `link` (연결선) / `cnv` (구간) / `port` (포트) / `carrier` (적재) — 토글 가능 |
| 연결선 그리기 | `cnv[i].outList` 의 각 LocationID에 대해 `cnvMap[outId]` 위치까지 직선 |
| 포트 라벨 | `scale > 0.3` 일 때 `CnvHostID.replace('P4ACV5R01_','')` 표시 |
| 툴팁 (CNV 구간) | LocationID, HostID, 좌표, Direction, In/Out, Carrier, Zone, Kind, State |
| 툴팁 (포트) | CnvHostID(이름), CnvUnitID, 좌표, Direction, LinkIn/Out, Carrier, IP, LineType |
| 통계 표시 | `CNV: N | Port: N | 캐리어: N` (CurrentCarrier 비어있지 않은 항목 합계) |

#### 4.9.5 UDP 인터페이스 ↔ 뷰어 데이터 연계

UDP 패킷 수신 시 뷰어 데이터 업데이트 흐름:

| UDP 필드 | 뷰어 갱신 대상 |
|----------|----------------|
| `F2 (P4ACV5R01)` | `EqpID` 필터 — 본 뷰어는 P4ACV5R01만 표시 |
| `F3 (LocationID)` | `cnv[].LocationID` 또는 `port[].CnvUnitID` 검색 |
| `F4 (특수 NODE_NAME)` | `port[].CnvHostID` 매칭 (`<EqpID>_<포트명>` 패턴) |
| `F12 (DEST_NODE)` | `cnv[].HostID` 또는 `port[].CnvHostID` 매칭 |
| `F13 (CARRIER_ID)` | 해당 위치의 `CurrentCarrier` 값 (실시간 갱신 필요) |
| `F16=1, F17 (Zone)` | `cnv[].ZoneName` 매칭 |

> **주의**: 본 뷰어가 사용하는 JSON은 DB 스냅샷이므로 UDP 실시간 트래픽과 동기화되지 않습니다. 가이드 명시 사항 — *"현재 DB는 스냅샷 데이터이며 실시간 갱신되지 않습니다"*.

#### 4.9.6 뷰어 파일 위치 / 사용

```
/AGV_CNV/CNV/cnv_map.html
```

브라우저에서 직접 열고 `cnv_map_data.json` 을 드래그하거나 `📂 JSON 열기` 클릭으로 로드.

---

## 5. 코드 정의 vs 실측 wire 포맷 비교 (구조적 차이만)

> **주의**: 본 절의 "코드 정의 필드명" 컬럼은 `AGV/AGV_JAVA/AgvMsgWorkerRunnable.java` 의 `VHL_STATE_REPORT` 상수에서 가져온 **코드상 명칭**입니다. 데이터 파일에서 직접 관측되는 값이 아니므로, 실측 wire의 각 필드 의미를 단정 짓는 근거로 사용하지 마십시오. 본 표의 의의는 **인덱스 위치 차이(구조적 시프트)** 한 가지 사실에 한정됩니다.

| 코드 인덱스 | 코드 정의 필드명 (참조) | 실측 wire 인덱스 |
|-------------|-------------------------|-------------------|
| 0 | TXT_ID | 0 |
| 1 | MCP_NM | 2 |
| 2 | VHL_ID | 3 |
| 3 | STATE | 4 |
| 4 | FULL | 5 |
| 5 | ERROR_CODE | 6 |
| 6 | ONLINE | 7 |
| 7 | ADDRESS | 8 |
| 8 | DISTANCE | 9 |
| 9 | NEXT_ADDRESS | 10 |
| 10 | RUN_CYCLE | 11 |
| 11 | VHL_CYCLE | 12 |
| 12 | CARRIER_ID | 13 |
| 13 | DESTINATION | 14 |
| 14 | EM_STATUS | 15 |
| 15 | GROUP_ID | 16 |
| 16 | SOURCE_PORT | 17 |
| 17 | DEST_PORT | 18 |
| 18 | PRIORITY | 19 |
| 19 | DET_STATUS | (실측 wire에 없음 — 20필드만 존재) |
| 20 | RUN_DISTANCE | (실측 wire에 없음) |
| 21 | CMD_ID | (실측 wire에 없음) |
| 22 | BAY_NM | (실측 wire에 없음) |

**팩트**:
- 실측 wire 포맷 인덱스 `1`에 `AGV` / `CNV` 가 위치 → 코드 정의에는 해당 슬롯 없음.
- 실측 wire 포맷은 20필드, 코드 정의는 23 슬롯(0~22).
- 결과적으로 인덱스 2~19는 코드 정의 대비 **+1 시프트**, 코드 정의의 19~22는 wire에 없음.

---

## 6. 미관측 항목 (참고)

본 데이터 파일에서 **관측되지 않은** 사항을 단순 기록:

- TXT_ID `1`, `3`, `4`, `13`, `14`, `15`, `51`, `201`, `202`, `203` (`2`만 존재)
- AGV F18 값 중 `M` 외
- AGV F19 값 중 `BOTH` 외
- CNV F8/F9 (알람) — 0건
- CNV F11/F15/F18/F19 — 0건 (모두 빈 값)
- `CZ-SRT4`

---

*문서 끝 — 본 문서 내용은 모두 `AGV_CNV_UDP_DATA.txt` 실측에 근거합니다.*
