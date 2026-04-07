# OHT 시뮬레이터 React 마이그레이션 사용 가이드

## 파일 구성

| 파일 | 역할 | 원본 |
|---|---|---|
| `layout_map_cre.js` | layout.xml → JSON (노드+연결) 파싱 | layout_map_cre.py |
| `hid_zone_csv_cre.js` | layout.xml → HID Zone JSON/CSV 생성 | hid_zone_csv_cre.py |
| `sim_server_3d_D5_D7.jsx` | React 프론트엔드 (3D 시뮬레이터 UI) | sim_server_3d_D5_D7.py 내장 HTML |

## 1. 프로젝트 셋업

```bash
# React 프로젝트 생성
npx create-react-app oht-simulator
cd oht-simulator

# 파일 복사
cp layout_map_cre.js src/
cp hid_zone_csv_cre.js src/
cp sim_server_3d_D5_D7.jsx src/

# 의존성 설치
npm install jszip          # ZIP 파일 파싱용 (layout_map_cre, hid_zone_csv_cre)
npm install three          # Three.js 3D 모드용 (선택)
```

## 2. App.js에서 사용

```jsx
// src/App.js
import OHTSimulator from './sim_server_3d_D5_D7';

function App() {
  return <OHTSimulator serverUrl="ws://localhost:8000" />;
}

export default App;
```

```bash
npm start
# → http://localhost:3000 에서 시뮬레이터 실행
```

## 3. 백엔드 (FastAPI) 실행

React 프론트엔드는 기존 Python FastAPI 백엔드에 WebSocket/REST로 연결한다.

```bash
# 백엔드 서버 (기존 그대로)
python sim_server_3d_D5_D7.py
# → ws://localhost:8000/ws (WebSocket)
# → http://localhost:8000/api/* (REST API)
```

CORS가 필요하면 백엔드에 추가:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 4. layout_map_cre.js 사용법

### 브라우저 — File input에서 파싱

```jsx
import { parseLayoutFromFile } from './layout_map_cre';

function LayoutUploader() {
  const handleFile = async (e) => {
    const file = e.target.files[0];
    const result = await parseLayoutFromFile(file);
    // result = { nodes: [...], connections: [...], meta: {...} }
    console.log(`노드: ${result.meta.nodeCount}, 연결: ${result.meta.connectionCount}`);
  };

  return <input type="file" accept=".xml,.zip,.html" onChange={handleFile} />;
}
```

### 브라우저 — XML 문자열 직접 파싱

```js
import { parseLayoutXml } from './layout_map_cre';

const xmlString = await fetch('/path/to/layout.xml').then(r => r.text());
const { nodes, connections } = parseLayoutXml(xmlString);
```

### 브라우저 — ZIP 파일 파싱

```js
import { parseLayoutZip } from './layout_map_cre';

const arrayBuffer = await fetch('/path/to/A.layout.zip').then(r => r.arrayBuffer());
const { nodes, connections } = await parseLayoutZip(arrayBuffer);
```

### Node.js CLI

```bash
# XML → JSON
node layout_map_cre.js layout.xml output.json

# ZIP → JSON
node layout_map_cre.js A.layout.zip output.json

# FAB 모드
node layout_map_cre.js --fab M14A --layout A
# → MAP/M14A/A.layout.json + A.layout.html 생성
```

### 출력 JSON 구조

```json
{
  "nodes": [
    { "no": 1, "x": 1234.56, "y": 789.01, "stations": [] },
    { "no": 2, "x": 1235.00, "y": 790.00, "stations": [] }
  ],
  "connections": [[1, 2], [2, 3]],
  "meta": {
    "nodeCount": 25000,
    "connectionCount": 28000,
    "parseTime": 1500,
    "generatedAt": "2026-04-07T12:00:00.000Z"
  }
}
```

## 5. hid_zone_csv_cre.js 사용법

### 브라우저 — File input에서 파싱

```jsx
import { parseFromFile, downloadCsv, downloadJson } from './hid_zone_csv_cre';

function HidZoneUploader() {
  const handleFile = async (e) => {
    const file = e.target.files[0];
    const result = await parseFromFile(file, 'M14 Project Ph-1');
    // result.zones = [{ Zone_ID, HID_No, Vehicle_Max, ... }, ...]
    console.log(`${result.meta.zoneCount}개 Zone, ${result.meta.rowCount}개 행`);

    // CSV 다운로드
    downloadCsv(result.zones, 'HID_Zone_Master.csv');

    // JSON 다운로드
    downloadJson(result, 'HID_Zone_Master.json');
  };

  return <input type="file" accept=".xml,.zip" onChange={handleFile} />;
}
```

### 브라우저 — XML 직접 파싱

```js
import { createHidZoneData } from './hid_zone_csv_cre';

const xmlContent = await fetch('/path/to/layout.xml').then(r => r.text());
const result = createHidZoneData(xmlContent, 'M14 Project');
```

### Node.js CLI

```bash
# XML → CSV + JSON
node hid_zone_csv_cre.js layout.xml output.csv

# FAB 모드
node hid_zone_csv_cre.js --fab M14A --layout A --project "M14 Project"
# → MAP/M14A/HID_Zone_Master_M14A_A.csv + .json 생성
```

### 출력 JSON 구조

```json
{
  "zones": [
    {
      "Zone_ID": 1,
      "HID_No": "HID-OHT-001",
      "Bay_Zone": "B01",
      "Vehicle_Max": 37,
      "Vehicle_Precaution": 26,
      "IN_Count": 2,
      "OUT_Count": 2,
      "IN_Lanes": "1234→1235; 1236→1237",
      "OUT_Lanes": "1238→1239",
      "HID_ID": "H001",
      "Addr_No": "1234",
      "Station_No": "5678"
    }
  ],
  "mcpZones": { "1": { "mcp_id": 1, "zone_id": 1, "vehicle_max": 37, "entries": [[1234, 1235]], "exits": [[1238, 1239]] } },
  "meta": { "zoneCount": 187, "rowCount": 450, "parseTime": 2000 }
}
```

## 6. sim_server_3d_D5_D7.jsx 기능 목록

### 헤더 컨트롤

| 기능 | 설명 |
|---|---|
| FAB 선택 | M14A/M14B/M16A/M16B 드롭다운 + 레이아웃 선택 + 적용 |
| 테마 토글 | 다크/화이트 모드 전환 |
| 3D 효과 | Pseudo-3D isometric 렌더링 ON/OFF |
| 곡선 모드 | 레일 코너 모따기(fillet) ON/OFF + 곡률반경 슬라이더 |
| Zone 마커 | 주의/포화 Zone 사각형 표시 ON/OFF (체크박스) |
| STK 표시 | Stocker 장비 아이콘 표시 ON/OFF (체크박스) |
| STK 배치/삭제 | +STK(맵 클릭 배치), -STK(클릭 삭제), ESC 해제 |
| 히트맵 모드 | OFF / Zone / 통과량 / 밀도 / 속도 |

### 좌측 사이드바

| 섹션 | 내용 |
|---|---|
| OHT 대수 설정 | 입력 + 적용 + Quick 버튼 (50/100/500) + Enter 키 |
| 실시간 통계 | 총 OHT, 운행, 적재, 정지, JAM |
| 속도 정보 | 평균/최대/최소 속도, 절대속도 |
| In/Out (데드락) | Total In/Out, 비율, 위험 구간, 상황만들기/초기화 |
| HID Zone 현황 | 총 Zone, 정상/주의/포화, 점유율, Zone/Station/ID/이름 토글 |
| JAM 현황 | JAM 차량 수, HIGH/MEDIUM 위험 (5초 폴링) |
| 시스템 정보 | FAB ID, 차량 길이, 최대속도, 거리단위 |
| 범례 | OHT 상태 색상 + 히트맵 범례 (모드별 동적) |
| 컨트롤 | 줌 레벨 표시 + 마우스 조작 안내 |

### 우측 사이드바

| 탭 | 기능 |
|---|---|
| OHT 상태 | 필터(전체/운행/적재/정지/JAM) + 검색 + 목록 + 클릭 확장 + 더블클릭 선택 |
| HID Zone | 필터(전체/포화/주의/정상) + 검색 + 목록 + 클릭 선택(맵 하이라이트) |

### 캔버스 인터랙션

| 조작 | 동작 |
|---|---|
| 드래그 | 맵 이동 (pan) |
| 마우스 휠 | 줌 인/아웃 (0.02x ~ 15x) |
| 더블클릭 OHT | 다중 선택 (경로선 표시, 색상별 구분) |
| 줌 50%+ | Station ID 자동 ON (수동 OFF 존중) |
| 줌 55%+ | Zone 표시 자동 ON (수동 OFF 존중) |
| STK 클릭 | 접기/펼치기 (슬롯 격자 상세뷰) |

### 렌더링

| 요소 | 설명 |
|---|---|
| 레일 | 직선/곡선(fillet), Pseudo-3D 파이프, 히트맵 색상 |
| 노드 | 원/타원 (3D 모드) |
| OHT (2D) | 글로우 + 그라데이션 + 광택 + JAM 펄스 + FOUP 표시 |
| OHT (3D) | 5면 박스 + 연결봉 + 그림자 + 화물 상자 |
| Zone Lane | IN(실선+화살표) / OUT(점선) + HID ID 라벨 + 차량수 |
| Zone 하이라이트 | 선택 Zone 노란 점선 사각형 |
| Zone 마커 | 주의(주황 점선) / 포화(빨강 실선) 사각형 + 라벨 |
| Zone 히트맵 | 점유율 기반 주황~빨강 오버레이 |
| Station | 타입별 아이콘(●■◆) + 색상 + ID(노랑)/이름 라벨 |
| STK | 접힌 아이콘(LED+ID+미니바) / 확장 패널(슬롯격자+사용량) |
| 경로 | 다중 선택 OHT 경로 점선 + 목적지 마커 |

## 7. 폴더 구조

```
프로젝트/
├── MAP/                          ← 기존 그대로 (변경 없음)
│   ├── M14A/
│   │   ├── A.layout.zip
│   │   ├── A.layout/layout/layout.xml
│   │   ├── A.station.dat
│   │   ├── A.layout.html        ← layout_map_cre.js 생성
│   │   ├── A.layout.json        ← layout_map_cre.js 생성 (신규)
│   │   ├── HID_Zone_Master_M14A_A.csv   ← hid_zone_csv_cre.js 생성
│   │   └── HID_Zone_Master_M14A_A.json  ← hid_zone_csv_cre.js 생성 (신규)
│   ├── M14B/
│   ├── M16A/
│   └── M16B/
├── src/
│   ├── layout_map_cre.js        ← XML→JSON 파서
│   ├── hid_zone_csv_cre.js      ← HID Zone 파서
│   ├── sim_server_3d_D5_D7.jsx  ← React 프론트엔드
│   └── App.js                   ← 진입점
├── sim_server_3d_D5_D7.py       ← 백엔드 (기존 그대로)
└── output/                      ← 시뮬레이션 CSV 출력
```

## 8. serverUrl Props

```jsx
// 같은 서버 (기본값: 현재 호스트)
<OHTSimulator />

// 다른 서버
<OHTSimulator serverUrl="ws://192.168.1.100:8000" />

// 폐쇄망 내부 서버
<OHTSimulator serverUrl="ws://amhs-sim.skhynix.internal:8000" />
```

## 9. ASAS 플랫폼 통합

ASAS 내 다른 모듈에서 import해서 사용:

```jsx
// ASAS Demos Alpha에서
import OHTSimulator from './modules/sim_server_3d_D5_D7';

// 특정 FAB으로 바로 연결
<OHTSimulator serverUrl={`ws://${ASAS_HOST}:8300`} />
```

layout_map_cre.js / hid_zone_csv_cre.js는 ASAS 스킬에서 독립적으로 사용 가능:

```js
// ASAS 데이터 분석 스킬에서
import { createHidZoneData } from './hid_zone_csv_cre';
import { parseLayoutXml } from './layout_map_cre';

// XML 업로드 → 분석 파이프라인
const layoutData = parseLayoutXml(xmlContent);
const hidData = createHidZoneData(xmlContent, fabName + ' Project');
```
