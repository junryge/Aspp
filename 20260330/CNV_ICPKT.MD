# CNV Bridge Map 데이터 구조 및 사용 가이드

## 개요

M16-PNT4 3F 브릿지 CNV(Conveyor) 시스템의 레이아웃 데이터를 MySQL에서 JSON으로 추출하여 맵 뷰어에서 시각화합니다.

## 파일 구성

| 파일 | 용도 |
|------|------|
| cnv_data_fetch.py | DB에서 JSON 데이터 추출 |
| cnv_map_data.json | 추출된 데이터 파일 |
| cnv_map.html | 맵 뷰어 (브라우저에서 열기) |

3개 파일 모두 같은 폴더에 배치합니다.

## 사용 방법

### 1단계: 데이터 추출

```
pip install pymysql
python cnv_data_fetch.py
```

같은 폴더에 `cnv_map_data.json`이 생성됩니다.

### 2단계: 맵 확인

`cnv_map.html`을 브라우저에서 열고 📂 JSON 열기 → `cnv_map_data.json` 선택.

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

CNV 경로 데이터는 `amp` DB, 포트 이름 데이터는 `config_bridge` DB에 있습니다.

## 대상 설비

| EQPID | 설비명 | 위치 |
|------|------|------|
| P4ACV5R01 | CNV 브릿지 | M16-PNT4 3F |

## 데이터 구조

### 1. amp.cnvinformation (859건) — CNV 전체 경로

컨베이어 전체 구간(세그먼트) 정보입니다. 각 구간의 좌표와 연결 관계를 포함합니다.

| 컬럼 | 타입 | 설명 |
|------|------|------|
| EqpID | text | 설비 ID (P4ACV5R01) |
| LocationID | char(30) | 구간 고유 ID |
| HostID | char(30) | 호스트 ID |
| XPos | float | X 좌표 |
| YPos | float | Y 좌표 |
| Width | float | 폭 |
| Height | float | 높이 |
| Direction | text | 방향 (0, 1, 2, 02 등) |
| ObjectInList | text | 입력 연결 (쉼표 구분 LocationID) |
| ObjectOutList | text | 출력 연결 (쉼표 구분 LocationID) |
| CurrentCarrier | text | 현재 적재 캐리어 ID |
| Kind | int | 오브젝트 종류 |
| ObjectState | int | 상태 |
| MachineState | int | 기계 상태 |
| ZoneName | text | 존 이름 |
| MaxCount | int | 최대 수량 |
| IPAddress | text | IP 주소 |
| CVPos | int | CV 위치 |
| PLCCross | int | PLC 크로스 |
| IsPort | int | 포트 여부 |
| PlcNo | int | PLC 번호 |
| UnitNo | int | 유닛 번호 |

### 2. config_bridge.conveyor (29건) — 포트 이름

컨베이어의 주요 입출구 포트 29개의 이름과 위치 정보입니다.

| 컬럼 | 타입 | 설명 |
|------|------|------|
| CnvID | int | 컨베이어 ID |
| CnvUnitID | char(60) | 유닛 ID |
| CnvHostID | char(60) | 포트 이름 (예: P4ACV5R01_COT1) |
| ZoneName | char(60) | 존 이름 |
| ObjectKind | int | 오브젝트 종류 |
| UnitState | int | 유닛 상태 |
| UnitStatus | int | 유닛 스테이터스 |
| CurrentCarrier | char(60) | 현재 적재 캐리어 |
| PositionX | int | X 좌표 |
| PositionY | int | Y 좌표 |
| Width | int | 폭 |
| Height | int | 높이 |
| Direction | int | 방향 |
| LinkInner | char(60) | 내부 연결 (쉼표 구분 UnitID) |
| LinkOuter | char(60) | 외부 연결 (쉼표 구분 UnitID) |
| IsIDReadPosition | int | ID 읽기 위치 여부 |
| IPAddress | char(60) | IP 주소 |
| LineType | char(10) | 라인 타입 (M16) |

## 포트 목록 (29개)

| 포트명 | 용도 |
|------|------|
| COT1, COT2 | Carrier Out Terminal |
| COF1, COF2, COF3 | Carrier Out From |
| CIF1, CIF2, CIF3 | Carrier In From |
| AIT1 ~ AIT8 | A Input Terminal |
| AIF1, AIF2, AIF3 | A Input From |
| AOF1, AOF2, AOF3 | A Output From |
| AIOF1 | A Input/Output From |
| MIT1, MIT2 | M Input Terminal |
| MIOT1 | M Input/Output Terminal |
| MOT1 | M Output Terminal |
| ARRIVED01, ARRIVED02 | 도착 지점 |

## 데이터 관계도

```
amp.cnvinformation (859건, 전체 구간)
  ├── ObjectInList → LocationID (입력 연결)
  ├── ObjectOutList → LocationID (출력 연결)
  └── EqpID = P4ACV5R01

config_bridge.conveyor (29건, 포트 이름)
  ├── LinkInner → CnvUnitID (내부 연결)
  ├── LinkOuter → CnvUnitID (외부 연결)
  └── CnvHostID = P4ACV5R01_포트명
```

두 테이블은 CnvUnitID(config_bridge) ↔ LocationID(amp.cnvinformation)로 연결됩니다.

## 맵 레이어 설명

| 레이어 | 색상 | 설명 |
|------|------|------|
| 연결선 | 회색 | CNV 구간 간 In/Out 연결 |
| CNV 구간 | 보라색 | 컨베이어 전체 세그먼트 (859개) |
| 포트 | 주황색 | 주요 입출구 포트 이름 표시 (29개) |
| 캐리어 | 초록색 | 현재 캐리어 적재 위치 |

## 참고사항

- P4ACV5R01 데이터만 필터링되어 추출됩니다 (전체 cnvinformation 5,264건 중 859건)
- config_bridge와 config DB에 동일 데이터가 존재하며 config_bridge를 사용합니다
- 현재 DB는 스냅샷 데이터이며 실시간 갱신되지 않습니다
- 데이터 갱신이 필요하면 cnv_data_fetch.py 재실행
