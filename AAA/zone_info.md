# ZONEINFO.JSON — MAP 영역 vs DATA 영역 분리 가이드

> ZONEINFO.JSON 하나의 Zone 객체 안에 **"어디에 있나" (MAP)** 와 **"어떻게 동작하나" (DATA)** 가 섞여 있습니다.
> 이 문서는 두 영역을 깔끔하게 분리하는 방법을 설명합니다.

---

## 한눈에 보기

```
┌─────────────────────────────────────────────────┐
│              ZONEINFO.JSON (1개 Zone)             │
├────────────────────┬────────────────────────────┤
│   MAP 영역 (5개)    │      DATA 영역 (11+α개)     │
│                    │                            │
│  ZoneID            │  PhysicalType              │
│  Level             │  RefDirection              │
│  posX              │  MotorReverse              │
│  posY              │  GearRatio                 │
│  ZoneDrawCount     │  EtherCATID / Name         │
│                    │  PLCSlaveID                │
│                    │  NextZone / PrevZone       │
│                    │  Profile (모션 4종)          │
│                    │  DisplayName               │
│                    │  AttributeLD (입출고포트)     │
│                    │  AttributeQS (분기기)        │
│                    │  AttributeLifter (리프터)    │
│                    │  LogicalType (출고게이트)     │
└────────────────────┴────────────────────────────┘
```

---

## 1. MAP 영역 — "이 Zone은 어디에 있나?"

HTML 맵을 그릴 때 **이 필드만** 있으면 됩니다.

| 필드 | 예시값 | 용도 |
|------|--------|------|
| `ZoneID` | `10101` | Zone 식별자 (맵에서 라벨로 표시) |
| `Level` | `0` 또는 `1` | 어느 층에 그릴지 (0=메인, 1=상위) |
| `posX` | `2296` | X 좌표 (맵 위치) |
| `posY` | `656` | Y 좌표 (맵 위치) |
| `ZoneDrawCount` | `1` 또는 `2` | 그리기 횟수 (2이면 확장 Zone) |

### MAP 영역만 추출하면?

```json
{
  "ZoneID": 10101,
  "Level": 1,
  "posX": 2296,
  "posY": 656,
  "ZoneDrawCount": 1
}
```

### 맵 그리기 핵심 포인트

```
전체 좌표 범위:
  X: -47,888 ~ 2,460  (폭 약 50,000)
  Y: -12,136 ~ 2,050  (높이 약 14,000)

Level 0 (메인): 1,881개 Zone — 넓은 영역
Level 1 (상위):   203개 Zone — 좁은 영역 (우측 상단)
```

---

## 2. DATA 영역 — "이 Zone은 어떻게 동작하나?"

제어/통신/경로에 필요한 필드들입니다.

### 2-1. 경로 연결 (체인)

| 필드 | 예시값 | 설명 |
|------|--------|------|
| `NextZone` | `10102` 또는 `-1` | 다음 Zone (-1이면 종점) |
| `PrevZone` | `10526` 또는 `-1` | 이전 Zone (-1이면 시작점) |

> NextZone/PrevZone은 **맵 표시에도 쓸 수 있지만** (화살표 그리기),
> 본질적으로는 **경로 데이터**입니다.

### 2-2. 장치 속성

| 필드 | 예시값 | 설명 |
|------|--------|------|
| `PhysicalType` | `0` | 장치 종류 (아래 표) |
| `RefDirection` | `2` | 방향 (0=북, 1=남, 2=동, 3=서) |
| `MotorReverse` | `1` | 모터 역회전 여부 |
| `GearRatio` | `400` | 기어비 |
| `DisplayName` | `"4AFC3301A_OUT85"` | 화면 표시 이름 |

**PhysicalType 빠른 참조:**

| 값 | 장치 | 색상 추천 (맵용) |
|:--:|------|:---:|
| 0 | 일반 롤러 | 회색 |
| 1 | 분기/합류 | 노랑 |
| 2 | 입고 포트 (IN) | 초록 |
| 3 | 출고 포트 (OUT) | 빨강 |
| 4 | 리프터 | 파랑 |
| 5 | 수평 리프터 (CVLH) | 보라 |
| 11 | 특수 구간 | 주황 |

### 2-3. 통신 설정

| 필드 | 예시값 | 설명 |
|------|--------|------|
| `EtherCATID` | `5` | EtherCAT 슬레이브 ID (0~38) |
| `EtherCATName` | `"a06"` | EtherCAT 슬레이브 이름 |
| `PLCSlaveID` | `-1` | PLC ID (현재 전체 미사용) |

### 2-4. 모션 프로파일 (Profile)

모든 Zone이 동일한 값 사용:

| 모드 | 속도(Vel) | 가속(Acc) | 감속(Dcc) | 저크(Jerk) |
|------|----------:|----------:|----------:|-----------:|
| Maint (유지보수) | 100 | 1,750 | 3,000 | 60,000 |
| RunFast (고속) | 800 | 1,325 | 1,850 | 25,000 |
| RunSlow (저속) | 470 | 800 | 510 | 40,000 |
| Override (수동) | 80 | 1,325 | 1,100 | 22,000 |

### 2-5. 선택 속성 (일부 Zone만 가짐)

| 속성 | 대상 | Zone 수 | 핵심 필드 |
|------|------|--------:|-----------|
| `AttributeLD` | 입출고 포트 + 분기 포함 | 241개 | E84PortNumber, RFIDPortNumber, GroupNumber |
| `AttributeQS` | 분기기 (Quick Switch) | 127개 | North/South/East 연결, HomeOffset |
| `AttributeLifter` | 수평 리프터 | 4개 | HomeLevel, LevelZone |
| `LogicalType` | 출고 게이트 등 | 10개 | 0 또는 2 |

---

## 3. 실전 분리 예시

### 원본 (ZONEINFO.JSON에서 1개 Zone)

```json
{
  "ZoneID": 10106,
  "Level": 1,
  "posX": 2460,
  "posY": 1066,
  "ZoneDrawCount": 1,
  "NextZone": -1,
  "PrevZone": 10107,
  "PhysicalType": 3,
  "RefDirection": 3,
  "MotorReverse": 0,
  "GearRatio": 400,
  "EtherCATID": 5,
  "EtherCATName": "a06",
  "PLCSlaveID": -1,
  "Profile": { "MaintVel": 100, "..." : "..." },
  "DisplayName": "4AFC3301A_OUT85",
  "AttributeLD": {
    "E84PortNumber": 0,
    "RFIDPortNumber": 0,
    "SGTPortNumber": -1,
    "IOModuleInstalled": 1,
    "IOEtherCATID": "30",
    "GroupNumber": 2
  }
}
```

### 분리 후 → MAP 부분

```json
{
  "ZoneID": 10106,
  "Level": 1,
  "posX": 2460,
  "posY": 1066,
  "ZoneDrawCount": 1
}
```

### 분리 후 → DATA 부분

```json
{
  "ZoneID": 10106,
  "NextZone": -1,
  "PrevZone": 10107,
  "PhysicalType": 3,
  "RefDirection": 3,
  "MotorReverse": 0,
  "GearRatio": 400,
  "EtherCATID": 5,
  "EtherCATName": "a06",
  "PLCSlaveID": -1,
  "Profile": { "..." : "..." },
  "DisplayName": "4AFC3301A_OUT85",
  "AttributeLD": {
    "E84PortNumber": 0,
    "RFIDPortNumber": 0,
    "SGTPortNumber": -1,
    "IOModuleInstalled": 1,
    "IOEtherCATID": "30",
    "GroupNumber": 2
  }
}
```

---

## 4. Python으로 분리하기

```python
import json

with open('ZONEINFO.JSON') as f:
    zones = json.load(f)

MAP_KEYS = {'ZoneID', 'Level', 'posX', 'posY', 'ZoneDrawCount'}

map_data = []
ctrl_data = []

for zone in zones:
    # MAP: 위치 정보만
    map_item = {k: zone[k] for k in MAP_KEYS}
    map_data.append(map_item)

    # DATA: 나머지 전부 (ZoneID는 키로 포함)
    data_item = {k: v for k, v in zone.items() if k not in MAP_KEYS or k == 'ZoneID'}
    ctrl_data.append(data_item)

with open('ZONE_MAP.json', 'w') as f:
    json.dump(map_data, f, indent=2)

with open('ZONE_DATA.json', 'w') as f:
    json.dump(ctrl_data, f, indent=2)

print(f"MAP: {len(map_data)}개 Zone, DATA: {len(ctrl_data)}개 Zone")
```

---

## 5. HTML에서 활용할 때

```
ZONE_MAP.json  → HTML Canvas/SVG 렌더링 (좌표 기반 맵 그리기)
ZONE_DATA.json → JavaScript 데이터 테이블, 클릭 시 상세 패널 표시
```

### 맵 그리기 팁

```javascript
// 좌표 변환 (원본 → 화면)
function toScreen(posX, posY, scale, offsetX, offsetY) {
  return {
    x: (posX - (-47888)) * scale + offsetX,
    y: (posY - (-12136)) * scale + offsetY
  };
}

// PhysicalType별 색상
const COLORS = {
  0: '#999',    // 일반 롤러
  1: '#FFD700', // 분기/합류
  2: '#00CC00', // 입고 (IN)
  3: '#FF4444', // 출고 (OUT)
  4: '#4488FF', // 리프터
  5: '#AA44FF', // 수평 리프터
  11: '#FF8800' // 특수
};
```

---

## 요약

| 구분 | 필드 수 | 핵심 질문 | 용도 |
|------|--------:|-----------|------|
| **MAP** | 5개 | "어디에 그릴까?" | HTML 맵 렌더링 |
| **DATA** | 11+α개 | "어떻게 제어할까?" | 경로/장치/통신 제어 |

> **TIP**: `PhysicalType`과 `RefDirection`은 DATA이지만, 맵에서 **색상/방향 표시**에도 쓰이므로
> HTML 맵을 풍부하게 만들고 싶으면 MAP 쪽에 같이 넣어도 좋습니다.
