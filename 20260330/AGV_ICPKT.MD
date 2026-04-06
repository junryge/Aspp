# AMP MAP 데이터 구조 및 사용 가이드

## 개요

AMP(Automated Material handling Platform) AGV 시스템의 레이아웃 데이터를 MySQL에서 JSON으로 추출하여 맵 뷰어에서 시각화합니다.

## 파일 구성

| 파일 | 용도 |
|------|------|
| amp_data_fetch.py | DB에서 JSON 데이터 추출 |
| amp_map_data.json | 추출된 데이터 파일 |
| amp_map.html | 맵 뷰어 (브라우저에서 열기) |
| eqpid_map.txt | EQPID ↔ 설비명 매핑 |

4개 파일 모두 같은 폴더에 배치합니다.

## 사용 방법

### 1단계: 데이터 추출

```
pip install pymysql
python amp_data_fetch.py
```

같은 폴더에 `amp_map_data.json`이 생성됩니다.

### 2단계: 맵 확인

`amp_map.html`을 브라우저에서 열고 📂 JSON 열기 → `amp_map_data.json` 선택, 📄 매핑 TXT → `eqpid_map.txt` 선택.

### 맵 조작법

- 드래그: 패닝
- 스크롤: 줌
- 상단 버튼: 레이어 on/off
- 마우스 오버: 상세 정보 툴팁

## DB 접속 정보

| 항목 | 값 |
|------|------|
| IP | 10.32.72.48 |
| PORT | 3306 |
| DB TYPE | MySQL |
| USER | root |
| PASSWORD | test |

레이아웃 데이터는 `amp` DB, 이동/AGV 데이터는 `acs` DB에 있습니다.

## 데이터 구조

### amp DB — 레이아웃

#### location (8,916건) — 노드 좌표

| 컬럼 | 타입 | 설명 |
|------|------|------|
| UnitID | char(60) | 노드 고유 ID |
| PositionX | int | X 좌표 |
| PositionY | int | Y 좌표 |
| Angle | int | 방향(각도) |
| Enabled | int | 활성 여부 (1/0) |
| EQPID | char(60) | 설비 ID |

#### node (22,395건) — 레인 구간

| 컬럼 | 타입 | 설명 |
|------|------|------|
| SegmentID | char(60) | 세그먼트 ID |
| FromID | char(60) | 출발 노드 (→ location.UnitID) |
| ToID | char(60) | 도착 노드 (→ location.UnitID) |
| FromXPos / FromYPos | int | 출발 좌표 |
| ToXPos / ToYPos | int | 도착 좌표 |
| Distance | int | 구간 거리 |
| Enabled | int | 활성 여부 (1/0) |

#### hostlocation (1,134건) — 설비 포트

| 컬럼 | 타입 | 설명 |
|------|------|------|
| UnitID | varchar(60) | 고유 ID |
| HostID | varchar(60) | 설비명 (예: 6B3BM043_1) |
| PositionX / PositionY | int | 좌표 |
| Enabled | int | 활성 여부 |
| Pio | int | PIO 여부 |

#### bufferlocation (1,902건) — 버퍼

| 컬럼 | 타입 | 설명 |
|------|------|------|
| UnitID | varchar(60) | 고유 ID |
| HostID | varchar(60) | 버퍼명 (예: 6ARB0410-8RC21) |
| PositionX / PositionY | int | 좌표 |
| CarrierDetect | int | 캐리어 감지 (1=있음) |
| CarrierID | varchar(60) | 캐리어 ID |
| BufferType | varchar(20) | 버퍼 타입 |
| BayID | char(60) | 베이 ID |
| GroupID | char(60) | 그룹 ID |

### 좌표 범위

| 축 | 최소 | 최대 |
|------|------|------|
| X | 72,347 | 177,366 |
| Y | 74,573 | 157,935 |

## 데이터 관계도

```
location (노드 좌표)
  ├── node.FromID / node.ToID → location.UnitID
  ├── hostlocation (설비 포트 위치)
  └── bufferlocation (버퍼 위치)
```

## EQPID 매핑 (eqpid_map.txt)

```
6AAV3B01=ICPKT
```

추가 설비 발견 시 한 줄씩 추가합니다.

## 맵 레이어 설명

| 레이어 | 색상 | 설명 |
|------|------|------|
| Node (레인) | 회색 | AGV 이동 경로 구간 |
| Location | 파란색 | 레일 위 노드/포인트 |
| Host (설비) | 주황색 | 설비 포트 (로드/언로드 지점) |
| Buffer | 초록색 | 버퍼 (캐리어 감지 시 진하게) |
| Disabled | 빨간색 | 비활성 구간 (기본 꺼짐) |

## 참고사항

- EQPID는 현재 `6AAV3B01` (ICPKT) 1종만 존재
- Enabled=0인 구간은 유지보수 중이거나 미사용 경로
- 데이터 갱신이 필요하면 `amp_data_fetch.py` 재실행
