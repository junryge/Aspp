# OHT MAP 구축 가이드

## 1. 개요

실시간 OHT(Overhead Hoist Transport) 차량 데이터를 MAP UI에 표시하기 위한 가이드.
실제 OHT 시스템에서 0.5초 간격으로 전송되는 VHL_STATE_REPORT 메시지를 파싱하고,
layout.xml의 Address 노드 좌표를 이용해 차량 위치를 계산한다.

---

## 2. 디렉토리 구조

```
OHT2/layout/
├── real_oht_parser.py          ← 실제 OHT 데이터 파서 (이 가이드의 핵심)
├── layout_map_cre.py           ← layout.xml → layout.html 변환
├── hid_zone_csv_cre.py         ← HID Zone 마스터 CSV 생성
├── sim_server_3d_D5_D7.py      ← 시뮬레이터 서버 (참고용)
├── MAP/
│   ├── M14A/
│   │   ├── A.layout.zip        ← layout.xml 포함 ZIP
│   │   ├── A.layout/layout/layout.xml  ← 압축 해제된 XML
│   │   ├── A.station.dat       ← Station 정보
│   │   └── A.layout.html       ← 자동 생성된 노드 데이터
│   ├── M14B/
│   ├── M16A/
│   └── M16B/
└── output/                     ← CSV/JSON 출력
```

---

## 3. layout.xml 구조

layout.xml은 FAB 내 모든 주행 경로(Address)와 설비(Station) 정보를 담고 있다.

### 3.1 Address 노드

각 Address는 OHT가 지나가는 지점으로, 고유 번호와 x,y 좌표를 가진다.

```xml
<group name="Addr12340" class="address.Addr">
    <param key="address" value="12340"/>
    <param key="draw-x" value="4523.45"/>    ← X 좌표
    <param key="draw-y" value="2341.67"/>    ← Y 좌표

    <!-- 다음 주소 (연결된 경로) -->
    <group name="NextAddr12341" class="address.NextAddr">
        <param key="next-address" value="12341"/>
    </group>

    <!-- 이 주소에 있는 설비 -->
    <group name="Station7591" class="Station">
        <param key="port-id" value="4ABL3301A_OUT04"/>
        <param key="type" value="0"/>
    </group>
</group>
```

### 3.2 주요 수치
| 항목 | 값 |
|------|-----|
| 총 Address 노드 | ~9,404개 |
| 총 Station 노드 | ~22,972개 |
| 총 NextAddr 연결 | ~18,806개 |
| XML 파일 크기 | ~211 MB |

### 3.3 파싱 방법

```python
from layout_map_cre import parse_layout_xml_content

with open("layout.xml", "r", encoding="utf-8") as f:
    xml_content = f.read()

nodes_list, connections = parse_layout_xml_content(xml_content)
# nodes_list: [{'no': 12340, 'x': 4523.45, 'y': 2341.67, 'stations': []}, ...]
# connections: [[12340, 12341], [12341, 12342], ...]
```

ZIP에서 직접 추출도 가능:
```python
import zipfile
with zipfile.ZipFile("MAP/M14A/A.layout.zip", "r") as zf:
    with zf.open("layout/layout.xml") as f:
        xml_content = f.read().decode("utf-8")
```

---

## 4. OHT 메시지 포맷 (VHL_STATE_REPORT)

콤마(,)로 구분된 23개 필드. 0.5초 간격으로 차량당 1건씩 전송.

### 4.1 필드 정의

| Index | 필드명 | 타입 | 설명 | 예시 |
|-------|--------|------|------|------|
| 0 | MessageId | String | 메시지 ID (2=VHL_STATE_REPORT) | `2` |
| 1 | McpName | String | MCP 이름 | `OHT` |
| 2 | VehicleId | String | 차량 ID | `V00795` |
| 3 | State | String | 차량 상태 코드 | `1` (RUN) |
| 4 | IsFull | String | 재하정보 (0=빈차, 1=적재) | `1` |
| 5 | ErrorCode | String | 에러 코드 (4자리) | `0000` |
| 6 | IsOnline | String | 통신 상태 (0=오프, 1=온) | `1` |
| 7 | **CurrentAddress** | **Int** | **현재 번지 (Address 번호)** | **`12340`** |
| 8 | **Distance** | **Int** | **거리 (100mm 단위)** | **`14`** (=1400mm) |
| 9 | **NextAddress** | **Int** | **다음 번지** | **`12341`** |
| 10 | RunCycle | String | 실행 Cycle | `4` (DEPOSIT) |
| 11 | VhlCycle | String | Vehicle Cycle | `4` (DEPOSIT_MOVING) |
| 12 | CarrierId | String | Carrier(Wafer POD) ID | `4PDMV608` |
| 13 | Destination | Int | 목적지 Address | `36073` |
| 14 | EMState | String | E/M 상태 (8자리) | `00000000` |
| 15 | GroupId | String | Group ID (4자리) | `0000` |
| 16 | SourcePort | String | 반송원 Port 이름 | `4ABL3301A_OUT04` |
| 17 | DestPort | String | 반송처 Port 이름 | `4ANZ19-701` |
| 18 | Priority | Int | 우선도 (0~255) | `90` |
| 19 | DetailState | String | 작업 상태 상세 | `0` |
| 20 | RunDistance | Int | 주행 거리 | `0` |
| 21 | CommandId | String | Command ID | (빈값) |
| 22 | BayName | String | Bay 명 | (빈값) |

### 4.2 예시

```
2,OHT,V00795,1,1,0000,1,12340,14,12341,4,4,4PDMV608,36073,00000000,0000,4ABL3301A_OUT04,4ANZ19-701,90,0,0
```

해석:
- V00795 차량이 RUN 상태
- Carrier 4PDMV608을 적재하고
- Address 12340에서 12341 방향으로 1400mm 이동한 위치
- 4ABL3301A_OUT04에서 4ANZ19-701로 반송 중
- 목적지: Address 36073, 우선도 90

---

## 5. Enum 코드표

### 5.1 VHL_STATE (차량 상태) - 필드 [3]

| 코드 | 이름 | 설명 |
|------|------|------|
| 1 | RUN | 운전 중 |
| 2 | STOP | 정지 중 |
| 3 | ABNORMAL | 상태 이상 |
| 4 | MANUAL | 수동 조치 |
| 5 | REMOVING | 분리 및 제거 중 |
| 6 | OBS_BZ_STOP | OBS-STOP/BZ-STOP |
| 7 | JAM | 정체 |
| 8 | HT_STOP | HT-STOP |
| 9 | E84_TIMEOUT | E84 time out |

### 5.2 RUN_CYCLE (실행 주기) - 필드 [10]

| 코드 | 이름 | 설명 |
|------|------|------|
| 0 | NONE | 사이클 없음 |
| 1 | POSITION_DETECT | 위치 확인 |
| 2 | MOVING | 이동 중 |
| 3 | ACQUIRE | 물품 집어올리기 (구원) |
| 4 | DEPOSIT | 물품 내려놓기 (도매) |
| 5 | SAMPLING | 샘플링 |
| 9 | FLOOR_TRANS | 층간 이동 |
| 21 | WHEELDRIVE | 바퀴 주행 |
| 22 | MANUAL_CONTROL | 수동 조작 |
| 23 | DRIVE_TEACHING | 주행 학습 |
| 24 | TRANS_TEACHING | 이재부 학습 |
| 2E | BUILDING_TRANS | 동 간 이동 |
| 2F | EVACUATION | 대피 이동 |

### 5.3 VHL_CYCLE (Vehicle 실행 주기) - 필드 [11]

| 코드 | 이름 | 설명 |
|------|------|------|
| 0 | NONE | 실행 사이클 없음 |
| 1 | MOVING | 이동 중 |
| 2 | ACQUIRE_MOVING | 구원 이동 |
| 3 | ACQUIRING | 구원 이송 중 |
| 4 | DEPOSIT_MOVING | 도매 이동 |
| 5 | DEPOSITING | 도매 이송 중 |
| 6 | MAINT_MOVING | 유지 이동 중 |
| 7 | WAITING | 대기 |
| 8 | INPUT | 투입 중 |

### 5.4 VHL_DET_STATE (상세 상태) - 필드 [19]

| 코드 | 이름 | 설명 |
|------|------|------|
| 0 | NONE | 없음 |
| 1 | WAIT | 대기 |
| 2 | STAGE_WAIT | 스테이지 대기 |
| 3 | STANDBY_WAIT | 대기 위치 대기 |
| 4 | DEPOSIT_SIG_WAIT | 도매 신호 대기 |
| 5 | ACQ_WAIT | 구원 대기 |
| 6 | MAP_WAIT | MAP 대기 |
| 101 | MOVING | 이동 중 |
| 102 | PARKING_UTS_MOVING | 주차/UTS 이동 |
| 103 | STAGE_MOVING | 스테이지 이동 |
| 104 | STANDBY_MOVING | 대기 위치 이동 |
| 105 | BALANCE_MOVING | 밸런스 이동 |
| 106 | PARKING_MOVING | 주차 이동 |

---

## 6. 좌표 변환 로직

### 6.1 핵심 원리

OHT 메시지의 `CurrentAddress`, `NextAddress`, `Distance`를 이용하여
layout.xml 노드맵에서 x,y 좌표를 **보간(interpolation)** 계산한다.

```
CurrentAddress(12340) ----Distance(1400mm)----> NextAddress(12341)
     (x1, y1)                                       (x2, y2)
```

### 6.2 보간 공식

```python
import math

# 1. 노드맵에서 두 Address의 좌표 조회
n1 = nodes[currentAddress]   # {'x': 4523.45, 'y': 2341.67}
n2 = nodes[nextAddress]      # {'x': 4530.12, 'y': 2341.67}

# 2. 두 노드 사이의 거리 계산 (좌표 단위)
edge_dist = math.sqrt((n2['x'] - n1['x'])**2 + (n2['y'] - n1['y'])**2)

# 3. Distance를 mm로 변환 (메시지의 Distance는 100mm 단위)
distance_mm = distance_100mm * 100   # 14 * 100 = 1400mm

# 4. 보간 비율 계산
#    sim_server 기준: distance_mm = edge_dist * ratio * 1000
#    따라서: ratio = distance_mm / (edge_dist * 1000)
ratio = distance_mm / (edge_dist * 1000)
ratio = min(1.0, max(0.0, ratio))   # 0~1 범위 제한

# 5. 보간된 좌표
x = n1['x'] + (n2['x'] - n1['x']) * ratio
y = n1['y'] + (n2['y'] - n1['y']) * ratio
```

### 6.3 예외 처리

| 상황 | 처리 |
|------|------|
| CurrentAddress가 노드맵에 없음 | x=0, y=0 (미매칭 표시) |
| NextAddress가 노드맵에 없음 | CurrentAddress 좌표 사용 |
| CurrentAddress == NextAddress | CurrentAddress 좌표 사용 |
| Distance = 0 | CurrentAddress 좌표 사용 |
| 두 노드가 직접 연결 안 됨 | 직선 거리로 계산 |

---

## 7. real_oht_parser.py 사용법

### 7.1 커맨드라인 실행

모든 명령은 `OHT2/layout/` 폴더에서 실행한다.

```bash
cd OHT2/layout
```

#### (1) 메시지 1건 직접 입력

따옴표(`"`)로 감싸서 입력한다.

```bash
python real_oht_parser.py "2,OHT,V00795,1,1,0000,1,12340,14,12341,4,4,4PDMV608,36073,00000000,0000,4ABL3301A_OUT04,4ANZ19-701,90,0,0"
```

#### (2) 여러 건 직접 입력

각 메시지를 따옴표로 감싸고 공백으로 구분한다.

```bash
python real_oht_parser.py "2,OHT,V00795,1,1,0000,1,12340,14,12341,4,4,4PDMV608,36073,00000000,0000,4ABL3301A_OUT04,4ANZ19-701,90,0,0" "2,OHT,V00564,6,1,0000,1,2115,9,2116,4,4,4PDD0270,7449,00000000,0000,4ANZ25-205,4KCW3301_3,50,0,0"
```

#### (3) 파일에서 읽기 (권장 - 대량 데이터)

메신저에서 받은 메시지를 텍스트 파일에 붙여넣고 `-f` 옵션으로 읽는다.
한 줄에 메시지 1건. 빈 줄이나 `2,`로 시작하지 않는 줄은 자동 무시.

```bash
python real_oht_parser.py -f messages.txt
```

`messages.txt` 예시:
```
2,OHT,V00795,1,1,0000,1,12340,14,12341,4,4,4PDMV608,36073,00000000,0000,4ABL3301A_OUT04,4ANZ19-701,90,0,0
2,OHT,V00564,6,1,0000,1,2115,9,2116,4,4,4PDD0270,7449,00000000,0000,4ANZ25-205,4KCW3301_3,50,0,0
2,OHT,V00975,1,1,0000,1,6033,0,3294,4,4,6PDB1402,34128,00000000,0000,4EPR5301_3,4ANZ03-302,50,0,0
2,OHT,V00313,1,1,0000,1,1071,0,1072,4,4,6PDN4064,21980,00000000,0000,4CSC1603_3,4AFZ47-328,50,0,0
2,OHT,V00649,1,1,0000,1,1448,0,1449,4,4,4NDNA076,17160,00000000,0000,4ALFE001_AO31,4AFZ15-160,50,0,0
```

450대분 전부 넣어도 된다.

#### (4) 인자 없이 실행 (내장 샘플 테스트)

```bash
python real_oht_parser.py
```

내장된 샘플 데이터 5건으로 테스트한다.

### 7.2 실행 결과

어떤 방식으로 실행하든 결과는 동일하다:

1. **화면 출력** - 차량별 파싱 결과 (좌표, 상태, Carrier 등)
2. **CSV 파일** - `output/real_oht_data_YYYYMMDD_HHMMSS.csv`
3. **JSON 파일** - `output/real_oht_state_YYYYMMDD_HHMMSS.json` (MAP 호환)

### 7.3 모듈로 import (다른 Python 코드에서 사용)

```python
from real_oht_parser import RealOHTParser

# 1. 파서 초기화 (layout.xml 로딩 - 처음 1번만)
parser = RealOHTParser("layout/layout.zip")

# 2. 메시지 1건 파싱
result = parser.process_message("2,OHT,V00795,1,1,0000,1,12340,14,12341,4,4,4PDMV608,36073,00000000,0000,4ABL3301A_OUT04,4ANZ19-701,90,0,0")
print(f"좌표: ({result['x']}, {result['y']})")
print(f"상태: {result['stateName']}")

# 3. 여러 건 한꺼번에
results = parser.parse_messages([msg1, msg2, msg3, ...])

# 4. CSV 저장
parser.save_csv(results, "output/real_oht_data.csv")

# 5. MAP 호환 JSON (시뮬레이터 get_state() 포맷)
map_json = parser.to_map_json(results)
```

### 7.3 CSV 출력 컬럼

| 컬럼명 | 설명 |
|--------|------|
| timestamp | 파싱 시각 |
| vehicleId | 차량 ID |
| x, y | 변환된 좌표 |
| coordFound | 노드맵 매칭 여부 (True/False) |
| state, stateName | 상태 코드/이름 |
| isFull | 적재 여부 |
| currentAddress, nextAddress | 현재/다음 번지 |
| distance_mm | 거리 (mm) |
| carrierId | Carrier ID |
| destination | 목적지 Address |
| sourcePort, destPort | 출발/도착 Port |
| runCycle, runCycleName | 실행 주기 |
| vhlCycle, vhlCycleName | Vehicle 주기 |
| priority | 우선도 |
| detailState, detailStateName | 상세 상태 |
| errorCode | 에러 코드 |
| emState | E/M 상태 |

### 7.4 MAP 호환 JSON 구조

```json
{
  "timestamp": "2026-04-07 14:30:00.123",
  "dataSource": "REAL",
  "vehicles": [
    {
      "vehicleId": "V00795",
      "x": 4525.12,
      "y": 2341.67,
      "state": 1,
      "stateName": "RUN",
      "isLoaded": 1,
      "currentNode": 12340,
      "nextNode": 12341,
      "destination": 36073,
      "carrierId": "4PDMV608",
      "runCycle": "4",
      "runCycleName": "DEPOSIT",
      "vhlCycle": "4",
      "vhlCycleName": "DEPOSIT_MOVING",
      "distance": 1400,
      "sourcePort": "4ABL3301A_OUT04",
      "destPort": "4ANZ19-701",
      "priority": 90
    }
  ],
  "stats": {
    "total": 450,
    "running": 400,
    "stopped": 30,
    "jammed": 20,
    "loaded": 150,
    "empty": 300
  }
}
```

---

## 8. 기존 시뮬레이터와의 관계

| 항목 | 시뮬레이터 (sim_server) | 실제 데이터 (real_oht_parser) |
|------|------------------------|-------------------------------|
| 데이터 소스 | 랜덤 생성 | 실제 OHT 메시지 |
| 프론트엔드 전송 | WebSocket (/ws) | JSON 파일 또는 별도 서버 필요 |
| 좌표 계산 | positionRatio 기반 | Distance 역산 |
| 업데이트 주기 | 0.5초 (시뮬레이션) | 0.5초 (실제 수신) |
| JSON 포맷 | get_state() | to_map_json() (호환) |

**핵심**: `real_oht_parser.py`의 `to_map_json()` 출력은 시뮬레이터의 `get_state()` 출력과
동일한 구조이므로, 기존 프론트엔드 MAP UI를 그대로 사용할 수 있다.

---

## 9. 실시간 연동 방법 (향후)

현재 `real_oht_parser.py`는 메시지를 받아서 파싱하는 라이브러리.
실시간 MAP 표시를 위해서는 다음 중 하나를 추가 구현:

1. **FastAPI 엔드포인트 추가**: POST /api/oht-messages 로 메시지 수신 → WebSocket으로 프론트에 전달
2. **UDP 수신**: 기존 시뮬레이터처럼 UDP로 메시지 수신
3. **파일 감시**: 메시지가 파일로 쌓이면 watchdog으로 감시하여 파싱

어느 방법이든 `RealOHTParser.process_message()`로 파싱 → `to_map_json()`으로 변환하면 된다.
