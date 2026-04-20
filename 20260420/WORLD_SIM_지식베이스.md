# WORLD_SIM — OHT 월드모델 시뮬레이션

> `WORD_MODEL/WORLD_SIM` 폴더. OHT 실데이터 CSV를 시간순으로 재생하면서 차량 위치·스타 지표·HID Zone·레일 차단·데드락 예측을 하나의 대시보드에서 보는 시스템.

---

## 0. 핵심 용어 (OHT / OBS / TAT / JAM)

반도체 FAB AMHS(자동 물류) 분야 전용 약어. WORLD_SIM 이해의 전제.

### OHT (Overhead Hoist Transport)
천장 레일을 따라 움직이는 무인 반송 차량. FOUP(웨이퍼 캐리어)을 장비 간에 실어 나른다. M14A FAB 기준 450대, 전체 V-Vehicle 1033대(+R-Vehicle 29대 별도).

### TAT (Turn Around Time, 반송 시간)
OHT가 작업을 할당받은 순간부터 완료까지 걸린 시간. **반송 한 건의 소요 시간**.
- **평균 2.88분 / 중앙값 2.57분** (M14A 24시간 균일)
- 3분 이내 완료 **61.5%**, 5분 이내 **91.7%**
- 큐가 1000~1800 범위 변동해도 TAT는 2.4~2.6분으로 안정 → **큐-TAT 상관 약함 (r=0.12)**
- WORLD_SIM에서는 `ts_resource`의 `RAIL-VEHICLEARRIVED`·`RAIL-VEHICLEDEPARTED` 이벤트 차이에서 산출, `oht_time_avg` 파일에 별도 평균값 저장

### OBS (Obstacle 정지, state=6 = OBS_BZ_STOP)
OHT가 장애물 또는 앞차에 막혀 **감속/정지**한 상태. 물리 구간 혼잡이 원인이므로 큐와는 거의 무관 (r=0.03).
- **정상 범위**: 120~160대 (`OBS_NORMAL_RANGE`)
- **WARNING ≥180** → 주의
- **DANGER ≥200** → 빌드업 중, 위험
- **CRITICAL/DEADLOCK ≥300** → 데드락 발생 가능
- 대시보드 데드락 위험 뱃지(`p-dl-badge`)가 이 임계값을 기준으로 색상 변경
- OBS 빌드업이 계속되면 JAM으로 발전 → 데드락으로 이어질 수 있음 (36분 빌드업 패턴 확인됨)

### JAM (교착/정체, state=7)
OHT가 **완전히 멈춰서 움직일 수 없는 상태**. OBS보다 심각한 단계.
- state=6(OBS)은 "잠시 멈춤"이지만, state=7(JAM)은 "물리적으로 막힘"
- 다수(≥5대)가 동시에 JAM → **진짜 데드락(순환 대기)** 의심 → Tarjan SCC 탐지
- 대시보드 OBS 추이 모달의 JAM 발생 기록에서 차량 수로 심각도 분류:
  - `≥5대` 🚨 데드락 발생
  - `≥2대` ⚠️ 주의 (데드락 직전)
  - `1대` 🟡 단발성 경고 (즉시 회복, 정상)
- Zone 용량 초과(`ZONE_{id}_FULL`)로 인한 JAM은 순환 대기가 아니므로 데드락에서 제외

### 세 용어의 관계
```
정상 운행 (RUN, state=1)
    ↓ 앞차/장애물
OBS (state=6) ← 물리 구조 혼잡. 큐와 무관.
    ↓ 회복 실패 + 다수 빌드업
JAM (state=7) ← 완전 정지. 1대면 단발성, ≥5대면 데드락.
    ↓ 순환 대기 (Tarjan SCC)
DEADLOCK ← 시스템이 자력 회복 불가, 운영자 개입 필요.
```

---

## 월드모델 1차 — 현재 구현 범위와 로드맵

이 시스템(WORLD_SIM)은 월드모델 구축의 **1차 단계**. 본격적인 뉴럴 월드모델(Dynamics Model) 붙이기 전의 **데이터 파이프라인 + 검증용 그라운드 트루스 재생기**에 해당한다. 이 단계가 없으면 2차에서 뭘 학습시킬지 정의 자체가 안 됨.

### 현재 WORLD_SIM에 있는 것 (1차)
- **실데이터 리플레이** — 누적 상태 기반, 시뮬 X, 그냥 재생
- **규칙 기반 데드락 탐지** — Tarjan SCC, Wait-For Graph
- **통계 기반 매크로 예측** — 이동평균 + 선형 외삽, TAT/물동량은 상수
- **결정론적 전방 step 시뮬** (`world.step()`) — Dijkstra + 단순 블로킹

### 아직 없는 것 (진짜 월드모델)
- **Dynamics Model**: OHT 다음 상태를 학습된 모델로 예측 (GNN/Transformer)
- **Counterfactual**: "이 구간 막으면 어떻게 돼?" what-if 시뮬
- **Policy / Action**: 라우팅 개입 추천, 차량 재배치
- **차량 의도 모델**: 개별 OHT가 어디로 갈지 학습된 예측
- **장기 롤아웃**: 10분 이상의 시뮬은 규칙 기반으로는 오차 누적으로 무의미

### 2차 로드맵 (예상)
1. `oht_raw_parser` + `oht_edge_aggregator`로 만든 **분 단위 상태** → GNN/Transformer 학습 데이터
2. `next_state = f(current_state, action)` **다이나믹스 학습**
3. WORLD_SIM의 `step()`을 규칙 기반에서 **뉴럴로 교체**
4. 그 위에 **정책 (강화학습 or MCTS)** 으로 개입 시뮬

### 한 줄 요약
지금 건 뉴럴 월드모델 붙이기 전 **데이터 파이프라인 + 검증용 그라운드 트루스 재생기**. 이게 2차의 학습 데이터 공급원 + 모델 성능 비교 베이스라인이 됨.

---

## 1. 실행 / 접속

```bash
cd WORD_MODEL/WORLD_SIM
python main.py
```

- 서버: FastAPI + uvicorn, 포트 **10005** (`config.SERVER_PORT`)
- 접속: `http://localhost:10005` → `dashboard.html` 자동 서빙
- WebSocket: `/ws` (0.5초 간격 상태 push, 재생 중에는 속도에 따라 간격 조절)

---

## 2. 파일 구성

| 파일 | 역할 |
|------|------|
| `main.py` | FastAPI 서버, REST API, WebSocket 엔드포인트 |
| `config.py` | 경로, 상수, 날짜 자동 스캔 (`_scan_data_dates`) |
| `data_loader.py` | CSV 7종 로딩 (`DateDataLoader`, `LayoutData`, `HIDZoneData`) |
| `replay_engine.py` | 리플레이 엔진 (play/pause/stop/jump/속도) |
| `world_model.py` | 시뮬레이션 엔진, Dijkstra, Tarjan SCC 데드락 탐지 |
| `velocity_tracker.py` | UDP 위치 변화 기반 차량별 순간 속도 (m/min) |
| `macro_predictor.py` | 큐·TAT·물동량·데드락 위험 예측 |
| `dashboard.html` | 단일 HTML 대시보드 (Canvas 맵 + 차트 + 사이드바 + 모달) |
| `data_loader_railcut_patch.py` | 레일 차단 페어 매칭(ABNORMAL/NORMAL) 패치 |
| `DATA_GUIDE.md` | 새 날짜 데이터 추가 방법 |
| `README.md` | 레일 차단 표시 패치 적용 방법 |

---

## 3. 데이터 추가 방법

`OHS_DATA_MD/` 아래에 **날짜 8자리 폴더**(예: `20260415`)를 만들고 CSV를 넣는다. 서버 재시작하면 드롭다운에 자동으로 뜬다.

| 파일명 키워드 | 내용 | 필수 |
|--------------|------|------|
| `OHT_DATA` / `OHT_날짜` | OHT 차량 raw (위치·상태·속도) | **필수** |
| `HID_INOUT` | HID 구간 통과 이벤트 (구간 속도) | **필수** |
| `OHT_RAIL_CUT` | 레일 차단 이벤트 | **필수** |
| `STAR_OHT` / `컬럼수집` | 스타 지표 (큐, OBS, 가동률) | **필수** |
| `oht_data_m14a` | 파싱된 OHT (컬럼 분리 버전, 있으면 raw 대신 우선 사용) | 선택 |
| `ts_resource` | 작업 명령 (TRANSPORTCOMMANDID) | 선택 |
| `oht_time_avg` | TAT 평균 | 선택 |

- 인코딩: utf-8 / cp949 / euc-kr 자동 시도
- 시간 포맷: `yyyy-MM-dd HH:mm:ss[.fff][±TZ]` 다중 시도
- 데이터량: 1일 약 700MB~1.2GB, 로딩 30초~1분

---

## 4. 핵심 상수 (`config.py`)

**차량 / TAT / 물동량:**
- 차량 수: M14A 450대, 전체 V-Vehicle 1033대
- TAT 평균 2.88분 (중앙값 2.57분), 3분 내 61.5%, 5분 내 91.7%
- 물동량: 시간당 ~20,000건 (주야간 동일)

**큐 / OBS 임계:**
- 큐 정상 범위: 1200~1400 (평균 1249)
- OBS 정상: 120~160
- **WARNING ≥180 / DANGER ≥200 / DEADLOCK ≥300**

**HID Zone:**
- Zone 182개, 차량 최대 37대, 주의 35대
- 혼잡 기준: 70% 이상 (`HID_CONGESTION_THRESHOLD_PCT`)

**시뮬레이션:**
- 스텝: 5초 (`SIM_STEP_SEC`)
- 예측 지평선: 10분 · 20분 (`PREDICTION_HORIZONS = [600, 1200]`)

**OHT 상태 코드 (`STATE_NAMES`):**
`1=RUN`, `2=STOP`, `3=ACCEL`, `4=DECEL`, `5=CURVE`, `6=OBS_BZ_STOP`, `7=JAM`, `8=HT_STOP`, `9=E84_TIMEOUT`

---

## 5. 주요 기능

### 5.1 실데이터 리플레이 (`replay_engine.py`)
- **누적 상태 방식**: 모든 OHT UDP 메시지를 `_cumulative_state[vid]`에 최신값으로 덮어쓰며 진행
- **프레임 스냅샷 간격**: 데이터량에 따라 자동 (2초 / 5초 / 10초)
- **컨트롤**: play / pause / stop / 속도(x1, x2, x5, x10, MAX) / 시간 점프 / 프레임 점프
- **프레임 점프 시**: 0 ~ 해당 프레임까지 누적 상태 재구축, `velocity_tracker.reset()` 호출

### 5.2 차량별 순간 속도 (`velocity_tracker.py`)
- UDP `(CurrentAddr, Distance, NextAddr)` + 수신시각 차분으로 **m/min** 계산
- 공식: `speed = delta_dm × 6 / elapsed_sec` (dm = 100mm)
- 3가지 케이스:
  - **A**: 같은 edge 안 이동 (`delta_dm = distance - prev.distance`)
  - **B**: 자연스러운 edge 전환 (`prev.next == curr.curr`)
  - **C**: 불연속 → BFS로 경로 길이 추정 (최대 20 hop)
- 범위: 0 ~ 300 m/min, 30초 초과 갭은 측정 불가
- 결과: `null`(측정 불가), `0`(정지), 실수(m/min)

### 5.3 HID Zone 포함 계산 (`data_loader.HIDZoneData`)
- `HID_Zone_Master_M14A_A.csv`에서 Zone별 IN/OUT Lane 파싱
- `in_lane_to_zone[(from, to)] = zone_id`, `out_lane_to_zone[...]` 매핑
- 차량의 현재 edge가 어느 Zone에 속하는지 즉시 조회
- Zone 상태: count / vehicleMax 비율로 `NORMAL / PRECAUTION(≥70%) / FULL(≥max)` 판정

### 5.4 매크로 예측 (`macro_predictor.py`)

| 지표 | 방법 | 비고 |
|------|------|------|
| **10분 후 큐** | 최근 10분 이동평균 + 선형 외삽(최근 5분 추세) | ±50 범위, trend_direction: up/down/stable |
| **TAT** | 상수 **2.88분** | 24시간 균일 확인됨 |
| **물동량** | 상수 **20,000건/시** | 주야간 동일 |
| **데드락 위험** | OBS 10분 이동평균 + 추세 | NORMAL/WATCH/WARNING/DANGER/CRITICAL |

- **상관관계 검증 결과** (`get_correlations`):
  - 큐-TAT: `r=0.12` (약한 양의 상관)
  - 큐-OBS: `r=0.03` (무관) — OBS는 물리 구조가 원인
  - 큐-HID속도: `r=-0.01` (무관)

### 5.5 데드락 탐지 (`world_model.py`)
- **Tarjan SCC** 알고리즘으로 Wait-For Graph(WFG)에서 순환 대기 탐지
- 조건: `stoppedTicks ≥ 2` (10초 이상 정지) + `velocity ≤ 0`
- WFG 구성:
  - `blockedBy`가 다른 stuck 차량 → 엣지 추가
  - 같은 edge 앞쪽 차량 (ratio 더 큰) 중 최근접
  - `ratio > 0.7`일 때 역방향 edge의 차량 추적
- SCC 크기 ≥ 2 → 데드락, 크기에 따라 severity:
  - `HIGH` (≥5대), `MEDIUM` (≥3대), `LOW` (2대)
- Zone 용량 초과는 `ZONE_{id}_FULL`로 별도 표시 (데드락 아님)

### 5.6 레일 차단 페어 매칭 (`data_loader_railcut_patch.py`)
- 기존: `t ≤ target_time` 모든 이벤트 누적 → 복구된 것도 계속 남음
- **패치 로직**: 시간순 재생하며 활성 집합(`active`) 관리
  - `STATE=ABNORMAL` → `active[(from, to)]` 추가
  - `STATE=NORMAL` → `active`에서 제거
  - 매칭 안 된 NORMAL은 무시 (전날 차단의 복구 등)
- 반환: `start_time`, `elapsed_sec`, `elapsed_min` 메타 포함
- **예시** (4/16 데이터):
  - `09:00:30 ABNORMAL 1470→15197` → active
  - `12:54:00 NORMAL 1761→1763` → 제거
  - `13:30 조회` → `1470→15197`만 반환 (270분째 활성)

---

## 6. 히스토리 관련 기능 (대시보드)

### 6.1 OBS / JAM 히스토리 (전체 데이터)
- **엔드포인트**: `GET /api/obs-jam-history`
- **구현**: `ReplayEngine.get_obs_jam_history()` — 전체 oht_timeline을 분 단위로 집계, 매 분마다 `state=6(OBS)` · `state=7(JAM)` 차량 수 기록
- **용도**: 재생과 무관한 전체 시계열. 사이드바 "평균 OBS / JAM 발생" 카운트, OBS 추이 모달 차트
- **클라이언트 변수**: `fullObsHistory` (데이터 로드 시 1회 fetch)

### 6.2 스타 히스토리
- **엔드포인트**: `GET /api/star-history`
- **구현**: `DateDataLoader.get_star_history()` — `star_timeline`을 `{time:"HH:MM", queue_total, obs_bz_stop, ...}` 형태로 변환
- **용도**: 하단 "큐 & OBS 추이" 차트 (`drawQueueChart`), 클릭 시 해당 시간으로 점프 가능

### 6.3 재생 중 누적 OBS 히스토리
- **클라이언트 변수**: `obsHistory` (최대 60 샘플, `OBS_HISTORY_MAX`), `jamEvents`
- **갱신**: `trackOBSHistory(obs, time, jamCount)` — 매 프레임 호출, 시각 중복 방지
- **JAM 이벤트 포착**: `jamCount > 0 && _lastJamCount === 0` (0→1 전이만 진짜 이벤트)

### 6.4 위험 레벨 이력 모달 (`openRiskModal`)
- **트리거**: 사이드바 "데드락 위험" 뱃지 더블클릭
- **구현**: `buildRiskHistory()` — `fullObsHistory` 전체를 훑으며 최근 10분 이동평균 + 추세로 위험 레벨 계산, 같은 레벨 연속 구간을 세그먼트로 묶음
- **임계값**: WATCH(상승추세>10), WARNING(≥180), DANGER(≥200), CRITICAL(≥300)
- **표시**: NORMAL 제외 모든 이벤트, 카운트 4종 + 최신순 로그

### 6.5 OBS Spike / JAM 발생 모달 (`openSpikeModal`)
- **트리거**: 사이드바 "📊 OBS 추이 상세보기" 버튼
- **내용**:
  - 현재 OBS / 평균 / 최대 / JAM 발생 수
  - OBS 시계열 차트 (초록 라인 + 파란 평균선 + 빨간 spike 임계 + **노란 현재 PLAY 세로선** + **빨간 JAM 세로선**)
  - JAM 발생 기록 — `jam ≥ 5`은 🚨데드락, `≥ 2`는 ⚠️주의, `1`은 🟡단발성 경고
- **spike 임계값**: `mean + 1.5σ` (`OBS_SPIKE_SIGMA`)

### 6.6 레일 차단 히스토리 (사이드바)
- **사이드바 "레일 차단 이벤트"** 패널: 활성 차단 개수 + 최근 5건 로그
- **상단 배지**: `cnt-railcut` (활성 edge 수), 빨간 깜빡임 (`@keyframes badgePulse`)
- **맵 표시**: 활성 edge에 빨간 점선 + ⛔ X 마커 (깜빡임, 투명도 조절)
- **토글**: `btn-railcut` (차단 표시 on/off)
- **배너**: 상단 빨간 경고 배너 (활성 차단 있을 때, 클릭하면 닫힘)

---

## 7. 주요 API 엔드포인트 (`main.py`)

| Method | Path | 설명 |
|--------|------|------|
| GET | `/` | dashboard.html |
| GET | `/api/dates` | 사용 가능한 날짜 목록 |
| GET | `/api/status` | 현재 시뮬레이션 상태 (차량 제외) |
| POST | `/api/replay/load` | `{date}` 날짜 데이터 로드 |
| POST | `/api/replay/play` / `pause` / `stop` | 재생 제어 |
| POST | `/api/replay/speed` | `{speed}` 배율 |
| POST | `/api/replay/jump` | `{time}` 또는 `{frame}` |
| GET | `/api/predict` | 매크로 예측 (큐/TAT/물동량/데드락) |
| GET | `/api/predict-deadlock` | 전방 시뮬레이션 기반 데드락 예측 |
| GET | `/api/correlations` | 데이터 간 상관관계 |
| GET | `/api/star-history` | 스타 전체 타임라인 |
| GET | `/api/hid-speeds` | HID 구간별 속도 통계 |
| GET | `/api/obs-jam-history` | OBS/JAM 분 단위 시계열 (전체) |
| GET | `/api/ts-events` | ts_resource 분 단위 집계 |
| GET | `/api/hid-zones` | HID Zone 현황 |
| GET | `/api/layout-graph` | 노드/엣지/Zone 전체 (맵 배경용, 초기 1회) |
| WS | `/ws` | 실시간 상태 스트림 + 클라이언트 명령 수신 |

**WebSocket 수신 명령**: `play`, `pause`, `stop`, `speed`, `jump`, `jump_frame`, `load`

---

## 8. 대시보드 UI 구성 (`dashboard.html`)

**상단 툴바:**
- 날짜 선택 → 재생/일시정지/정지 → 속도 (x1·x2·x5·x10·MAX) → 시간 표시 → 슬라이더 → 상태 뱃지
- 레이어 토글: Zone / 차단 / ID / 이름

**왼쪽 사이드바** (기본 접힘):
- 차량 현황, 스타 지표, 10분 후 예측, 위험 Zone, 레일 차단 이벤트, 데이터 소스

**중앙 맵** (Canvas, 확대/축소/드래그):
- 레일 네트워크 (캐시 처리)
- 차량 점 (색상: 공차=초록, 적재=파랑, OBS=주황, JAM=빨강, 정지=회색)
- HID Zone 레인 (IN=실선, OUT=점선)
- 레일 차단 표시 (빨간 점선 + ⛔ X 마커, 깜빡임)
- 상단 통계 배지 (공차/적재/OBS/정지/레일차단 실시간 카운트)
- **선택 하이라이트**:
  - OHT 더블클릭 → 노란 링 + 자동 줌인 (`selectOHT`)
  - Zone 더블클릭 → 노란 박스 + 중앙 포커스 (`selectZone`)
  - `ESC`로 선택 해제

**오른쪽 사이드바**:
- 탭: OHT 상태 / HID Zone
- 필터: 전체/운행/적재/정지/JAM, 전체/포화/주의/정상
- 검색, 정렬 (OHT는 상태 우선, Zone은 차량 수 내림차순)
- 아이템 더블클릭 → 선택 토글

**하단 패널**:
- 큐 & OBS 추이 차트 (세로선 = 현재 재생 시각, 클릭으로 점프)
- 데이터 상관관계 요약

**모달**:
- 위험 레벨 이력 (NORMAL 제외 구간)
- OBS 추이 상세 (spike 차트 + JAM 발생 기록)
- `ESC`로 닫기

---

## 9. OHT 메시지 파싱 (`data_loader.parse_oht_message`)

Raw 문자열 예: `"2,OHT,V00066,6,1,0000,1,13307,9,13308,..."`

| 필드 | 인덱스 | 의미 |
|------|--------|------|
| vid | 2 | 차량 ID (V00066 등) |
| state | 3 | 상태 코드 (1~9) |
| isFull | 4 | 적재 여부 (0/1) |
| currentNode | 7 | 현재 주소 |
| distance | 8 | 현재 edge 내 위치 (dm) |
| nextNode | 9 | 다음 주소 |
| runCycle | 10 | 주행 사이클 |
| vhlCycle | 11 | 차량 사이클 |
| carrierId | 12 | 캐리어 ID |
| destination | 13 | 목적지 노드 |
| sourcePort | 16 | 출발 포트 |
| destPort | 17 | 도착 포트 |
| speed | 18 | 메시지상 속도 (사용 안함, VelocityTracker가 실제 계산) |

`oht_data_m14a` 파일은 이미 컬럼 분리된 파싱 버전이라 있으면 우선 사용.

---

## 10. 자주 묻는 질문 / 트러블슈팅

**Q. 드롭다운에 날짜가 안 뜬다**
→ `OHS_DATA_MD/{날짜8자리}/` 폴더명이 숫자인지 확인, `OHT_DATA` 키워드 파일이 있는지 확인, 서버 재시작.

**Q. 차량 속도가 `-` 또는 `측정 불가`로 나온다**
→ 첫 관측이거나, UDP 갭이 30초 초과거나, 불연속에서 BFS 경로를 못 찾은 경우. VelocityTracker가 가짜 값을 만들지 않고 `None` 반환 (정직화).

**Q. 레일 차단이 계속 쌓이기만 한다 (복구 안됨)**
→ `data_loader.py`의 `get_frame_at` 안에서 RAIL_CUT 블록을 `_get_active_rail_cuts(target_time)`로 교체했는지 확인 (README.md 절차).

**Q. 프레임 점프 후 속도가 이상하다**
→ 점프 시 `velocity_tracker.reset()`이 호출됨. 이전 상태와의 연속성이 끊기므로 처음 몇 프레임은 `None`이 정상.

**Q. 2.5D / 3D 시각화와 어떻게 다른가?**
→ WORLD_SIM은 **2D 평면 Canvas** 기반. 2.5D/3D와는 엄격히 분리 (혼용 금지, 존의 개인 운영 규칙).

---

## 11. 향후 결정 사항

- **날짜 경계 처리**: 현재는 선택한 날짜 하루치만 재생. 전날 차단이 다음날로 이어지는 경우 `_scan_data_dates`에서 이전 날짜 연결 필요 (별도 요청 시 구현).
- **5일 이상 데이터 축적 후**: 패턴 예측(요일별/시간대별) 정확도 향상 가능.

---

## 12. 핵심 키워드 (RAG 검색용)

`OHT (Overhead Hoist Transport)`, `OBS (장애물 정지, state=6)`, `OBS_BZ_STOP`, `TAT (Turn Around Time, 반송시간)`, `JAM (교착/정체, state=7)`, `월드모델`, `월드모델 1차`, `Dynamics Model`, `Counterfactual`, `what-if`, `Policy`, `차량 의도 모델`, `장기 롤아웃`, `GNN`, `Transformer`, `MCTS`, `강화학습`, `그라운드 트루스 재생기`, `WORLD_SIM`, `AMHS`, `M14A`, `FAB`, `HID Zone`, `스타 지표`, `반송 큐`, `레일 차단`, `RAIL_CUT`, `ABNORMAL`, `NORMAL`, `데드락`, `Tarjan SCC`, `Wait-For Graph`, `VelocityTracker`, `Dijkstra`, `layout_cache`, `HID_Zone_Master`, `ts_resource`, `oht_time_avg`, `oht_raw_parser`, `oht_edge_aggregator`, `페어 매칭`, `누적 상태`, `프레임 점프`, `매크로 예측`, `spike 임계`, `위험 레벨 이력`, `dashboard.html`, `FastAPI`, `WebSocket`, `Canvas 맵`, `2.88분`, `20000건/시`, `순환 대기`, `FOUP`, `V-Vehicle`.
