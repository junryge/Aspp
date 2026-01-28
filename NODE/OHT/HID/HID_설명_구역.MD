# HID Zone 마스터 데이터 컬럼 정의서

> M14 Project OHT AMHS 시스템 HID(Hoisting Interface Device) Zone 구성 정보

---

## 1. 기본 식별 정보

| 컬럼명 | 데이터 타입 | 설명 | 예시 |
|--------|-------------|------|------|
| Zone_ID | Integer | HID Zone 고유 식별 번호 | 1, 2, 3 ... 187 |
| HID_No | String | HID 장비 고유 번호 | HID-OHT-001 |
| Bay_Zone | String | 소속 Bay 영역 코드 | B01, M02, OS1, OE1 |
| Sub_Region | Integer | Bay 내 세부 구역 번호 (동일 Bay에 복수 HID 존재 시 구분) | 1, 2, 3 |
| Full_Name | String | HID 전체 명칭 (Bay + Sub_Region + Zone_ID 조합) | HID-B01-1(001) |
| Territory | Integer | 관할 영역 코드 | 1 (전체 동일) |
| Type | String | 장비 유형 | HID (전체 동일) |

---

## 2. Lane 구성 정보

| 컬럼명 | 데이터 타입 | 설명 | 예시 |
|--------|-------------|------|------|
| IN_Count | Integer | 해당 HID Zone으로 진입하는 Lane 개수 | 4, 5, 10 |
| OUT_Count | Integer | 해당 HID Zone에서 진출하는 Lane 개수 | 4, 5, 12 |
| IN_Lanes | String | 진입 경로 상세 (시작노드→끝노드, 세미콜론 구분) | 3048→3023; 3091→3002 |
| OUT_Lanes | String | 진출 경로 상세 (시작노드→끝노드, 세미콜론 구분) | 3025→3026; 3043→3044 |

### Lane 표기법 설명
- `3048→3023`: OHT 주행 경로에서 노드 3048 → 노드 3023으로 연결되는 구간
- 세미콜론(`;`): 복수 Lane 구분자
- IN_Lanes: 외부에서 해당 HID Zone으로 들어오는 경로
- OUT_Lanes: 해당 HID Zone에서 외부로 나가는 경로

---

## 3. 차량 용량 정보

| 컬럼명 | 데이터 타입 | 설명 | HID4 기준값 | MiniHID 기준값 |
|--------|-------------|------|-------------|----------------|
| Vehicle_Max | Integer | 해당 Zone 내 최대 허용 OHT 대수 | 37 | 5 |
| Vehicle_Precaution | Integer | 혼잡 주의 임계값 (이 값 초과 시 혼잡 상태) | 35 | 3 |

### 용량 관리 기준
- **정상**: Vehicle 수 < Vehicle_Precaution
- **주의**: Vehicle_Precaution ≤ Vehicle 수 < Vehicle_Max
- **포화**: Vehicle 수 ≥ Vehicle_Max

---

## 4. 관리 정보

| 컬럼명 | 데이터 타입 | 설명 | 예시 |
|--------|-------------|------|------|
| Project | String | 소속 프로젝트명 | M14 Project Ph-1, 将来, M14 Bridge Project |
| ZCU | String | Zone Controller Unit 코드 (HID 제어기 식별) | ZC504A, ZC505A |
| HID_Type | String | HID 장비 타입 | HID4, MiniHID |

### Project 값 설명
| 값 | 의미 |
|----|------|
| M14 Project Ph-1 | 현재 운영 중인 HID Zone |
| 将来 (장래) | 미래 확장 예정 Zone (미가동) |
| M14 Bridge Project | Bridge 연결 구간 HID |
| (빈값) | 정의되지 않음 |

### HID_Type 설명
| 타입 | 특징 |
|------|------|
| HID4 | 대형 HID (Vehicle_Max: 31~37) |
| MiniHID | 소형 HID (Vehicle_Max: 5), 주로 보조 경로 |

---

## 5. Bay Zone 분류 체계

| 접두어 | 의미 | 예시 |
|--------|------|------|
| B## | 일반 생산 Bay | B01, B02 ... B62 |
| M## | Main 복도 / 이송 경로 | M02, M05, M60 |
| OS# | Outer Stocker 연결 | OS1, OS2 ... OS6, OSA~OSI |
| OE# | Outer East 연결 | OE1, OE2, OEA~OEC |
| ON# | Outer North 연결 | ON1, ON2 ... ON6, ONA~ONG |
| OW# | Outer West 연결 | OW1, OW2, OWA, OWB |
| BR# | Bridge 연결 | BR1, BR2 |

---

## 6. 데이터 활용 가이드

### 분석 시 제외 대상
- `Project = '将来'`: 미가동 Zone
- `Zone_ID` 178~187: HID_No, Bay_Zone 미정의

### 주요 활용 시나리오
1. **혼잡도 모니터링**: Vehicle_Max, Vehicle_Precaution 기준 실시간 체크
2. **경로 분석**: IN_Lanes, OUT_Lanes 파싱하여 네트워크 그래프 구성
3. **병목 구간 탐지**: IN_Count, OUT_Count 불균형 Zone 탐색
4. **장애 추적**: ZCU 코드 기반 컨트롤러 단위 분석

---

*작성일: 2025-01-29*