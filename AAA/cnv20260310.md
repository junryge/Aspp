# ZONEINFO.JSON 분석 보고서

## 1. 파일 개요

| 항목 | 값 |
|------|-----|
| 파일 크기 | ~1.1MB |
| 데이터 형식 | JSON Array |
| 총 Zone 수 | **2,084개** |
| 고유 Profile 수 | 1개 (전 Zone 동일 프로파일) |

---

## 2. 맵 영역 / 데이터 영역 구조

맵 영역과 데이터 영역은 **분리되어 있지 않다.** 하나의 Zone 객체 안에 맵 정보와 데이터 정보가 함께 포함된 구조이다.

```json
{
  // --- 맵 영역 (화면 표시용) ---
  "Level": 1,
  "posX": 2296,
  "posY": 656,
  "ZoneDrawCount": 1,
  "RefDirection": 1,
  "DisplayName": "Zone_10101",

  // --- 데이터 영역 (제어/설정용) ---
  "ZoneID": 10101,
  "NextZone": 10102,
  "PrevZone": 10526,
  "EtherCATID": 0,
  "EtherCATName": "a01",
  "MotorReverse": 0,
  "GearRatio": 400,
  "PLCSlaveID": -1,
  "PhysicalType": 0,
  "Profile": { "MaintVel": 100, "RunFastVel": 800, ... }
}
```

### 필드 분류

| 구분 | 필드 | 용도 |
|------|------|------|
| **맵 영역** | `Level`, `posX`, `posY`, `ZoneDrawCount`, `RefDirection`, `DisplayName` | 화면에 어디에 어떻게 그릴지 |
| **데이터 영역** | `ZoneID`, `NextZone`, `PrevZone`, `EtherCATID/Name`, `MotorReverse`, `GearRatio`, `PLCSlaveID`, `PhysicalType`, `Profile` | 실제 제어 파라미터 |

> Zone 단위로 맵+데이터가 **1:1로 묶여 있는 flat 구조**이며, 별도 섹션이나 별도 파일로 분리되어 있지 않다.

---

## 3. 맵 영역 상세 (Map Area)

### 3.1 전체 맵 좌표 범위

| 축 | 최솟값 | 최댓값 | 범위 |
|----|--------|--------|------|
| posX | -47,888 | 2,460 | 50,348 |
| posY | -12,136 | 2,050 | 14,186 |

### 3.2 Level별 맵 영역

#### Level 1 (상위 레벨) - 203 zones
| 축 | 최솟값 | 최댓값 | 범위 |
|----|--------|--------|------|
| posX | 410 | 2,460 | 2,050 |
| posY | -328 | 2,050 | 2,378 |
- ZoneID 범위: 10101 ~ 11110
- 상대적으로 좁은 영역에 밀집

#### Level 0 (하위 레벨) - 1,881 zones
| 축 | 최솟값 | 최댓값 | 범위 |
|----|--------|--------|------|
| posX | -47,888 | 492 | 48,380 |
| posY | -12,136 | 82 | 12,218 |
- ZoneID 범위: 10409 ~ 34631
- 넓은 영역에 분포하는 메인 컨베이어 라인

---

## 4. 데이터 영역 상세 (Data Fields)

### 4.1 Zone 데이터 구조

각 Zone 객체는 다음 16개 필드로 구성됨:

| 필드명 | 타입 | 설명 |
|--------|------|------|
| `Level` | int | 레이어 레벨 (0 또는 1) |
| `posX` | int | 맵 상 X 좌표 |
| `posY` | int | 맵 상 Y 좌표 |
| `ZoneDrawCount` | int | 드로잉 카운트 (대부분 1, 일부 2) |
| `NextZone` | int | 다음 Zone ID (-1이면 종점) |
| `PrevZone` | int | 이전 Zone ID (-1이면 시작점) |
| `ZoneID` | int | 고유 Zone 식별자 |
| `EtherCATID` | int | EtherCAT 슬레이브 ID (0~38) |
| `EtherCATName` | str | EtherCAT 이름 (a01~a39) |
| `MotorReverse` | int | 모터 역방향 여부 (0/1) |
| `GearRatio` | int | 기어비 (400/1625/8000) |
| `PLCSlaveID` | int | PLC 슬레이브 ID (전체 -1) |
| `Profile` | dict | 속도/가감속 프로파일 (16개 파라미터) |
| `PhysicalType` | int | 물리적 장치 유형 (0~11) |
| `RefDirection` | int | 기준 방향 (0~3) |
| `DisplayName` | str | 표시 이름 |

### 4.2 PhysicalType 분류

| PhysicalType | Zone 수 | GearRatio | 추정 용도 |
|-------------|---------|-----------|-----------|
| 0 | 1,712 | 400 | 일반 컨베이어 (롤러) |
| 1 | 135 | 400 | 분기/합류 구간 |
| 2 | 52 | 400 | 입출고 포트 (IN) |
| 3 | 48 | 400 | 입출고 포트 (OUT) |
| 4 | 127 | 8,000 | 리프터/수직이송 장치 |
| 5 | 4 | 1,625 | CVLH (수평 리프터) |
| 11 | 6 | 400 | 특수 구간 |

### 4.3 RefDirection (기준 방향)

| RefDirection | Zone 수 | 비율 |
|-------------|---------|------|
| 0 | 342 | 16.4% |
| 1 | 327 | 15.7% |
| 2 | 684 | 32.8% |
| 3 | 731 | 35.1% |

### 4.4 MotorReverse

| 값 | Zone 수 |
|----|---------|
| 0 (정방향) | 132 |
| 1 (역방향) | 1,952 |

### 4.5 EtherCAT 슬레이브

- 총 39개 EtherCAT 슬레이브 (a01 ~ a39)
- EtherCATID 범위: 0 ~ 38
- PLCSlaveID: 전체 -1 (미사용)

---

## 5. Zone 체인 분석 (NextZone / PrevZone)

| 구분 | Zone 수 |
|------|---------|
| 종점 (NextZone == -1) | 307 |
| 시작점 (PrevZone == -1) | 217 |
| NextZone이 데이터에 없는 Zone | 308 |
| PrevZone이 데이터에 없는 Zone | 218 |

- 시작점보다 종점이 많음 → 분기(diverge) 구조 존재
- NextZone/PrevZone으로 컨베이어 경로 체인 구성

---

## 6. 명명된 장비 (Named Zones)

### 6.1 4AFC3301A 설비
- 109개 Zone이 `4AFC3301A_INxx`/`4AFC3301A_OUTxx` 패턴
- PhysicalType: 0, 2, 3
- 입출고 포트 관련 설비

### 6.2 CVLH (수평 리프터)
| 이름 | ZoneID | PhysicalType |
|------|--------|-------------|
| CVLH01 | 10411 | 5 |
| CVLH02 | 10611 | 5 |
| CVLH03 | 10910 | 5 |
| CVLH04 | 11110 | 5 |

모두 Level 1에 위치, GearRatio 1625 사용

---

## 7. 모션 프로파일 (Profile)

전체 2,084개 Zone이 동일한 프로파일 사용:

| 파라미터 | Maint | RunFast | RunSlow | Override |
|----------|-------|---------|---------|----------|
| Vel | 100 | 800 | 470 | 80 |
| Acc | 1,750 | 1,325 | 800 | 1,325 |
| Dcc | 3,000 | 1,850 | 510 | 1,100 |
| Jerk | 60,000 | 25,000 | 40,000 | 22,000 |

- **Maint**: 유지보수 모드 (저속)
- **RunFast**: 고속 운전 모드
- **RunSlow**: 저속 운전 모드
- **Override**: 오버라이드 모드
