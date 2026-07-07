# M16↔M14 Bridge Que 증가 대응 SOP

> 원본: A1.zip (1.PNG ~ 4.PNG, 4장) 전체 내용 정리
> 대상: M16↔M14 Bridge 구간 OHT Queue 증가 시 대응 절차

---

## 1. 역할 분담 (기본 원칙)

| 역할 | 담당 |
|---|---|
| Max Capa 변경 | **M16 EUV 구성원** |
| M16A/M16B Storage Full 층간 이동 대응 | **해당 FAB 구성원** |

- M16↔M14 대응 시 **해당 FAB 담당자에게 Storage Rate 현황 공유** → 반송정지 대응
- 각 FAB별 Storage 현황 확인 후 **QUBE 창에 공지**
- Bridge 구성원은 **Storage 현황 확인하면서 MAX CAPA 조정** 진행

---

## 2. MAX CAPA 대응 관련 각 FAB Storage 현황 확인 SOP (기본 흐름)

```
MAX CAPA 대응
   → 각 FAB Storage 확인 (M14A/B, M16A/B)
   → M14A/M14B/M16 MAX CAPA 변경 [EUV]
   → M14A/M14B/M16 MAX CAPA 원복 [EUV]
   → 각 FAB Storage 확인 / M16A/M16B 층간 설정
```

---

## 3. 발동 기준·조치 요약표

| 방향 | 단계 | 발동 기준 | 조치 (담당 FAB) |
|---|---|---|---|
| **M14 방향 (6F→3F)** | 1단계 | Bridge Time 5분↑ **&** M16 ZT 6F→3F QUE **280개↑** | M16 6F ZT AI MAX CAPA 조정 **"3"** (M16A) |
| **M14 방향 (6F→3F)** | 2단계 | HUB Total **600↑** **&** M16 ZT 6F→3F QUE **320개↑** | M16 6F ZT AI MAX CAPA 조정 **"1"** (M16A) + M14A FAB CONV AI Port **Disable** (M14A) + M14B 7F ZT AI Port **Disable** (M14B) |
| **M16 방향 (3F→6F)** | 1단계 | Bridge Time 5분↑ **&** M16 ZT 3F→6F QUE **280개↑** | M14A FAB CONV AI MAX CAPA 조정 **50%↓** (M14A) + M14B 7F ZT AI MAX CAPA 조정 **50%↓** (M14B) |
| **M16 방향 (3F→6F)** | 2단계 | HUB Total **600↑** **&** M16 ZT 3F→6F QUE **320개↑** | M16 6F ZT AI MAX CAPA 조정 **"3"** (M16A) + M14A FAB CONV AI Port **Disable** (M14A) + M14B 7F ZT AI Port **Disable** (M14B) |
| **원복 (M14 방향 대응 후)** | - | M16 ZT **6F→3F 전체 Que 100EA 이하** | M16/M14A/B Max CAPA 원복 |
| **원복 (M16 방향 대응 후)** | - | M16 ZT **3F→6F 전체 Que 100EA 이하** | M16/M14A/B Max CAPA 원복 |

- 조치 박스 색 구분(원본): **M16A**=주황, **M14A**=파랑, **M14B**=보라 → 해당 조치 수행 FAB 표시
- [비고] 원본 M16 방향 2단계 박스 제목이 "M14 HUB OHT 몰림 발생 2단계"로 표기돼 있음. 기준이 M16 ZT 3F→6F QUE라서 문맥상 **M16 오기**로 보인다. 원문 그대로 옮기되 확인 필요.

---

## 4. M14 방향 Bridge OHT 몰림 발생 SOP (6F→3F)

### 1단계 (6F→3F)
- **기준**: Bridge Time : 5분↑ & **M16 ZT 6F→3F QUE 280개 ↑**
- **조치**: M16 6F ZT AI MAX CAPA 조정 **"3"** [M16A]

### 2단계
- **기준**: HUB Total **600 ↑** & **M16 ZT 6F→3F QUE 320개 ↑**
- **조치**:
  1. M16 6F ZT AI MAX CAPA 조정 **"1"** [M16A]
  2. M14A FAB CONV AI Port **Disable** [M14A]
  3. M14B 7F ZT AI Port **Disable** [M14B]

### 원복 조건
- **M16 ZT 6F→3F 전체 Que 100EA 이하** 시 M16/M14A/B Max CAPA 원복
- 확인 위치: ZT 장비별 Que 테이블의 **6F→3F 컬럼** 합계

---

## 5. M16 방향 Bridge OHT 몰림 발생 SOP (3F→6F)

### 1단계 (3F→6F)
- **기준**: Bridge Time : 5분↑ & **M16 ZT 3F→6F QUE 280개 ↑**
- **조치**:
  1. M14A FAB CONV AI MAX CAPA 조정 **50%↓** [M14A]
  2. M14B 7F ZT AI MAX CAPA 조정 **50%↓** [M14B]

### 2단계 (원문 표기: "M14 HUB OHT 몰림 발생 2단계")
- **기준**: HUB Total **600 ↑** & **M16 ZT 3F→6F QUE 320개 ↑**
- **조치**:
  1. M16 6F ZT AI MAX CAPA 조정 **"3"** [M16A]
  2. M14A FAB CONV AI Port **Disable** [M14A]
  3. M14B 7F ZT AI Port **Disable** [M14B]

### 원복 조건
- **M16 ZT 3F→6F 전체 Que 100EA 이하** 시 M16/M14A/B Max CAPA 원복
- 확인 위치: ZT 장비별 Que 테이블의 **3F→6F 컬럼** 합계

---

## 6. 대상 ZT 장비 (Que 확인 대상 10대)

원복 조건 판단 시 아래 장비의 방향별 Que(3F→6F / 6F→3F)를 확인한다.

| No | MACHINENAME |
|---|---|
| 1 | 6ABL0111 |
| 2 | 6ABL0112 |
| 3 | 6ABL0121 |
| 4 | 6ABL0122 |
| 5 | 6ABL6011 |
| 6 | 6ABL6012 |
| 7 | 6ABL6021 |
| 8 | 6ABL6022 |
| 9 | 6ABL6031 |
| 10 | 6ABL6032 |

---

## 7. Retry 대응 (ZT 몰림 심화 시)

```
Retry 대응
   → ZT 몰림 심화 / OHT 움직임 없음
   → 3F ZT 특정 AI PORT Close
   → Bridge 구성원 Paused 처리
   → 1회 Retry 진행 / 시간 별 2회 금지
```

- **1회만 Retry 진행**하고, **같은 시간대에 2회 금지**.

---

## 8. M16↔M14 Bridge OHT 정체 발생 대응 시나리오

대응 시 해당 FAB 담당자와 Storage Rate 현황 공유 + 반송정지 대응이 전제.

### 시나리오 구분
1. **M14 HUB 몰림 발생**
2. **M16 HUB 몰림 발생**

### Bridge 정체 해소 방안 (공통 흐름)

**메인 라인:**
```
Bridge 정체 해소 방안
   → M14A/M14B/M16A MAX CAPA "1"
   → Bridge 반송 230↓
   → MAX Capa 순차적 변경 "3" → "원복"
```

**병행 라인 (MAX CAPA "1" 설정과 동시 진행):**
```
MCS MAX CAPA 수정을 진행
   → Storage 현황 확인
   → M16A/B 층간 이동 요청
   → Storage 현황 확인
```

- 요지: MAX CAPA "1"로 조여서 Bridge 반송 230 이하로 떨어뜨린 뒤, "3" → "원복" 순으로 **순차적으로** 되돌린다. 그 사이 Storage 현황을 계속 확인하면서 M16A/B 층간 이동을 요청해 Storage Full을 방지한다.

---

## 9. 설정 방법 (MCS Managements 화면)

공통 경로: **Managements → 장비 상태 관리 → AI Port Max Capacity 조정 탭 → LOCATION 선택 → 옵션 선택(Default / Option #1~#6) → Confirm**

| 구분 | 화면 | LOCATION | 장비 예시 (MACHINE NO / NAME / PORT) |
|---|---|---|---|
| **M16A MAX CAPA 수정** | Managements - M16A (2037766) | **M16_6F** | #0111 / 6ABL0111 / 6ABL0111_AI612, 6ABL0111_AI622 |
| **M14A MAX CAPA 수정** | Managements - M14 (2037766) | **ALL** | #4AFC3201 / 4AFC3201 / 4AFC3201A_IN17, 4AFC3201A_IN18 |
| **M14B MAX CAPA 수정** | Managements - M14B (2037766) | **ALL** | #4ABLD111 / 4ABLD111 / 4ABLD111_AI75, 4ABLD111_AI77 |

- 옵션 라디오 버튼: Default, Option #1 ~ Option #6 → **Confirm / Cancel**
- 상단 메뉴: AMHS, Carrier, 이력, 통계, 경로, 현재 반송, Application, 도움말
- 툴바: 장비 상태 관리, Carrier 관리, Unable Status, 로그 조회, 반송 JOB 실패 이력

### MMDM 접속 확인
**MMDM → Enumeration Code → MHS → DelayFoupTransCount_New 접속 확인**

- Usage: Factory **M16** / TypeValue **MHS** (Desc: MHS), MINTERLOCK (M/C-ILK ON/OFF) — 총 107 Row
- EnumDef: Factory **M16** / CommonCode **DelayFoupTransCount_New** / Desc "장기 적체 FOUP Auto 반송시 Stocker당 한…" (화면 잘림)
- EnumDefValue:

| Factory | CommonCode | TypeValue | Desc | Default Value |
|---|---|---|---|---|
| M16 | DelayFoupTransCount_New | M16A | 0 | M16A |
| M16 | DelayFoupTransCount_New | M16B | 3 | M16B |
| M16 | DelayFoupTransCount_New | M16E | 0 | M16E |

---

## 10. 모니터링 화면 (smartSTAR)

**경로: smartSTAR → 물류모니터링 → FAB Monitoring → 이천 → M16HUB Monitoring**

관련 탭: 층간/FAB간 Bridge 이상 감지, M16HUB Monitoring, M16E Monitoring, M16_WT Monitoring, DASHBOARD

### 판단 기준과 모니터링 지표 매핑
| SOP 기준 | 확인 지표 |
|---|---|
| HUB Total 600↑ | **Current Queue Count → M16HUB TOTAL Q-C** (원본 예시: 546 / 576, 빨간 박스 표시 항목) |
| M16 ZT 방향별 QUE 280/320/100 | **ZT 장비별 Que 테이블** (MACHINENAME × 3F→6F / 6F→3F 컬럼) |
| Bridge Time 5분↑ | Bridge 반송 시간 (Transport Time 계열) |

### Current Queue Count 패널 항목 (원본 예시값 포함)
| 항목 | 예시값 |
|---|---|
| M16HUB FABTRANS | 847 |
| **M16HUB TOTAL Q-C** (핵심, 빨간 박스) | **546** |
| M14→M16 MESQCNT | 179 |
| M16→M14 MESQCNT | 314 |
| M16→M14A MESQC | 174 |
| M16→M14B MESQC | 140 |
| M14A TOTAL CNV Q | 754 |
| M16HUB OHT Q-CNT | 217 |
| M14B→LFT Q CNT | 82 |
| LFT→M14B Q CNT | 69 |
| M16A→LFT Q CNT | 121 |
| M16A→SOUTH CNV (강조) | 56 |
| M16A→NORTH CNV | 55 |
| LFT→M16A Q CNT | 115 |
| M14A→M16A QCNT | 219 |
| M14B→M16A Q CNT | 197 |
| M16A→M14A Q CNT | 217 |
| M16A→M14B Q CNT | 180 |
| NORTH CNV→M14A | 52 |
| SOUTH CNV→M14A | 54 |
| M16 → BRIDGE M… (강조) | 34 |

### ZT 장비별 Que 테이블 (모니터링 화면 예시, 2025-07-24 오후 3:23:07 기준)
| No | MACHINENAME | 3F→6F | 6F→3F | 2F→ |
|---|---|---|---|---|
| 1 | 6ABL0111 | 4 | 13 | |
| 2 | 6ABL0112 | 3 | 14 | |
| 3 | 6ABL0121 | 7 | 7 | |
| 4 | 6ABL0122 | 5 | 12 | 1 |
| 5 | 6ABL6011 | 5 | 19 | |
| 6 | 6ABL6012 | 5 | 6 | |
| 7 | 6ABL6021 | 4 | 9 | |
| 8 | 6ABL6022 | 5 | 16 | |
| 9 | 6ABL6031 | 5 | 4 | |
| 10 | 6ABL6032 | 9 | 12 | |

※ M16 방향 판단 시 3F→6F 컬럼(빨간 박스), M14 방향 판단 시 6F→3F 컬럼을 본다. 위 값은 화면 예시값.

### 그 외 M16HUB Monitoring 표시 항목
- **VHL Util(%)**: OHT UTIL (예시 90.98)
- **Queue Status (Last 10…)**: Q-CREATED 572, Q-COMPLETED 495, M16→M14A Q-C, M16→M14B Q-C, M14A→M16 Q-C 71, 125 등
- **Transport Time (Last 1…)**: AVG TOTAL TIME 추이
- **Stock Rate(%)**: ALL(FOUP) 5.1 / M14 HUB 30.0 / M16 HUB 1.5 (예시)
- **MCS**: MCSA1/MCSA2 CPU(%) 게이지, MCSA1_MEM 51124.3MB, MCSA2_MEM 50576.7MB (예시)
- **OHT1 / OHT3 / UTSC**: CPU(%), SERVICE / PROS_ST / CLUSTER 상태
- **이상정보 현황 (Tag/Value/Count)**: OHTMCPAlarm, STKMCPAlarm, INVMCPAlarm, STK/OHT/INV DisConnection, OHTStatusError, QueTimeDelay, PIOCommError, AOTransDelayDet, AOTransDelay, OHTLoadFail, OHTCannotExecute, ProcessUpStart, ProcessUpFail, ProcessShutDownFail, ProcessHang, ProcessDown, CarrierDuplicate, DoubleStorage, INV_CPU(5.1%), INV_MEM_USED(6.48%), INV_DISK_USED(12%), INV_SWAP_USED(0%)
- **History 로그 예시**: `M16HUB.OHT.ALERT.PIOCOMMERROR [OHT][6ECMB101][4PDMB011][PIO Communication Error Occured][From[6FIOB102][To[6ALFF402]]CarrierLoc[BV0196]]` 형식 (2025-07-24 15:16:03 등)

---

## 11. 운영 시 핵심 체크리스트

1. Bridge Time 5분↑ + 방향별 ZT QUE **280개↑** → **1단계 발동**
2. HUB Total **600↑** + 방향별 ZT QUE **320개↑** → **2단계 발동** (Port Disable 포함)
3. MAX CAPA 변경/원복은 **EUV 구성원**, 층간 이동 대응은 **해당 FAB 구성원**
4. 조정 중 **Storage Rate 현황을 QUBE 창에 공유**, Storage Full 시 M16A/B 층간 이동 요청
5. 원복은 해당 방향 **ZT 전체 QUE 100EA 이하** 확인 후 진행 (MAX CAPA "1" → "3" → "원복" 순차)
6. Retry는 **1회만**, 같은 시간대 **2회 금지**. ZT 몰림 심화·OHT 무동작 시 3F ZT 특정 AI PORT Close + Bridge 구성원 Paused 처리
