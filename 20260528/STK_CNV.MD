# STK/CNV 고정 위치 추출 방법

OHT2 의 `station.dat` 과 `layout.xml` 을 결합해서 STK/CNV/EQP 포트의 고정 위치(X, Y)를 CSV 로 추출하는 방법.

---

## 1. 개요

### 문제
- STK(Stocker), CNV(Conveyor) 같은 장비의 **고정 위치 좌표** 가 필요
- Java 코드(`Stocker.java`, `Conveyor.java`) 자체에는 X/Y 필드 없음
- Oracle DB(`SELECT_EQP_INF`)는 로컬에 없음

### 해결
**FAB 내부 파일 2개를 join** 해서 좌표 도출:
1. `station.dat` — 포트가 **어느 레일 주소(Addr)** 에 붙어있는지
2. `layout.xml` (layout.zip 내부) — 각 **Addr 의 실제 mm 좌표**

---

## 2. 데이터 소스 분석

### 2.1 `station.dat` 구조

위치: `/home/user/ASAS/OHT2/station.dat`

한 줄 예시:
```
STATION = 7596,"DUAL_ACCESS", 1,"4ABL3301A_IN01", 0x00000005, 1, 12332, 0,0,0,0,0,"UNIVERSAL", 0,0,0, 1399, 0,0,0,0,0
```

필드 매핑 (0-indexed, `=` 우측을 콤마로 split):

| idx | 값 예시 | 의미 |
|----:|---|---|
| 0 | `7596` | station_id |
| 1 | `"DUAL_ACCESS"` | station_type (ACQUIRE/DEPOSIT/DUAL_ACCESS/MAINTENANCE/...) |
| 3 | `"4ABL3301A_IN01"` | 포트 이름 |
| **6** | **`12332`** | **레일 주소(Addr) 번호** ← 외래키 |
| 12 | `"UNIVERSAL"` | access_mode (UNIVERSAL/ZFS_LEFT/ZFS_RIGHT/DUMMY/...) |
| **16** | **`1399`** | **slide 오프셋 (mm)** |

**포트 station 타입:**
- `ACQUIRE` — 출력 포트 (OHT가 짐 집어가는 곳)
- `DEPOSIT` — 입력 포트 (OHT가 짐 내려놓는 곳)
- `DUAL_ACCESS` — 양방향 포트 (STK 셀프, 브릿지)
- 그 외(`MAINTENANCE`, `DUMMY` 등)는 무시

### 2.2 `layout.xml` 구조

위치: `/home/user/ASAS/OHT2/layout/layout/layout.zip` 내부 `layout.xml` (211MB)

```xml
<group name="Addr12332" class="...address.Addr">
    <param value="12332"                key="address"/>
    <param value="70295.731219782"      key="cad-x"/>   <!-- 실제 fab CAD X (mm) -->
    <param value="-138094.17760949666"  key="cad-y"/>
    <param value="0.0"                  key="cad-z"/>
    <param value="11094.221816917998"   key="draw-x"/>  <!-- HMI 그리기 X -->
    <param value="1304.849842526659"    key="draw-y"/>
    <!-- ... 다른 param 들 ... -->
    <group name="NextAddrXX" class="...NextAddr">       <!-- 인접 Addr 정보 (이건 건너뜀) -->
        <param value="..." key="next-address"/>
        ...
    </group>
</group>
```

**좌표계:**
- `cad-x, cad-y, cad-z` (mm 단위) — 실제 fab 공간 좌표
- `draw-x, draw-y` — 화면 그리기 좌표 (cad / scale)

---

## 3. 조인 로직

```
station.dat                    layout.xml
─────────────────────          ──────────────────────────────
port_name = 4ABL3301A_IN01
address   = 12332      ──join──→  Addr12332 의 cad-x, cad-y 조회
slide     = 1399                  (slide 는 별도 컬럼 보존)
                                          │
                                          ▼
                            CSV 한 행:
                              device_type=CNV
                              port_name=4ABL3301A_IN01
                              address=12332, slide_mm=1399
                              cad_x=70295.73, cad_y=-138094.18
                              draw_x=11094.22, draw_y=1304.85
```

---

## 4. STK / CNV / EQP 분류 규칙

좌표 자체는 동일하게 계산하고, **포트 이름 패턴**으로 device_type 분류:

```python
if port_name 끝이 "-<숫자>":         → "STK"  (스토커 셀프 슬롯)
                                       예: 4AFZ01-001, 4ANZ67-540
elif equipment_id 가 [A-Z](FC|BL|LFA)[0-9]:  → "CNV"  (컨베이어/브릿지/리프터)
                                       예: 4AFC3301A, 4ABL3301A, 4ALFA201
else:                                → "EQP"  (일반 공정장비)
                                       예: 4AFS2101A, 4ANS4001A
```

### 프리픽스 의미 (추정)
| 코드 | 의미 | 분류 |
|---|---|---|
| **FC** | Front Conveyor | CNV |
| **BL** | Bridge Lifter | CNV |
| **LFA** | Lifter Assembly | CNV |
| **FZ** | FOUP Zone (storage) | STK |
| **NZ** | (storage 류) | STK |
| **FS** | FOUP Sorter | EQP |
| **NS** | (sorter 류) | EQP |
| 기타 | 공정장비 | EQP |

---

## 5. Java 원본 로직과의 관계

`Mcp75Config.java:892~930` 의 station 좌표 계산:
```java
RawPoint head = rawPointMap.get(rs.getAddress_no());          // station.dat 의 address
RawPoint tail = rawPointMap.get(head.getLeftAddress());       // 또는 RightAddress
double stDistFromHead = rs.getLeftDistance();                 // station.dat 의 slide

// head Addr 와 다음 Addr 사이를 비율 보간
double x = head.getDrawX() + ((tail.getDrawX() - head.getDrawX()) * stDistFromHead/edgeLength);
double y = head.getDrawY() + ((tail.getDrawY() - head.getDrawY()) * stDistFromHead/edgeLength);

// 포트 방향(LEFT/RIGHT) 추가 보정
if (rs.getPortLocation() == PORT_LOCATION.RIGHT) {
    x += slidePortDist * Math.sin(edgeAngle);
    y += slidePortDist * Math.cos(edgeAngle);
}
```

**현재 스크립트는 head Addr 좌표만 사용 (slide 보간 미적용).**
- 시각화/대략 위치 분류엔 충분
- mm 단위 정밀도 필요 시 layout.xml 의 `NextAddr.next-address`, `left-distance`/`right-distance` 까지 읽어서 보간 필요

---

## 6. 결과 CSV 스펙

파일: `stk_cnv_positions.csv` (3MB, 22,625 행)

### 컬럼 (13개)
| 컬럼 | 의미 | 예 |
|---|---|---|
| `device_type` | STK / CNV / EQP | `CNV` |
| `station_id` | station 고유번호 | `7596` |
| `station_type` | ACQUIRE/DEPOSIT/DUAL_ACCESS | `DUAL_ACCESS` |
| `access_mode` | UNIVERSAL/ZFS_LEFT/ZFS_RIGHT | `UNIVERSAL` |
| `port_name` | 포트 이름 | `4ABL3301A_IN01` |
| `equipment_id` | 장비 ID (포트명 `_` 앞부분) | `4ABL3301A` |
| `address` | 레일 주소번호 | `12332` |
| `slide_mm` | Addr 에서의 오프셋 (mm) | `1399` |
| `cad_x` | **실제 fab X 좌표 (mm)** | `70295.73` |
| `cad_y` | **실제 fab Y 좌표 (mm)** | `-138094.18` |
| `cad_z` | Z 좌표 (대부분 0) | `0.0` |
| `draw_x` | HMI 그리기 X | `11094.22` |
| `draw_y` | HMI 그리기 Y | `1304.85` |

### 통계 (OHT2 기준)
| device_type | 포트 수 | 고유 장비 수 |
|---|---:|---:|
| **STK** | 16,985 | 16,985 |
| **CNV** | 188 | 21 |
| **EQP** | 5,452 | 2,032 |
| **합계** | **22,625** | — |

---

## 7. 사용법

### 기본 실행 (OHT2)
```bash
cd /home/user/ASAS/OHT2/layout
python3 extract_stk_cnv_positions.py
# → stk_cnv_positions.csv 생성 (약 5초)
```

### 다른 FAB (예: M14A from OHT3)
```bash
unzip -j OHT3/MAP.zip "M14A/A.station.dat" "M14A/A.layout.zip" -d /tmp/
python3 extract_stk_cnv_positions.py \
    --station /tmp/A.station.dat \
    --layout-zip /tmp/A.layout.zip \
    --output stk_cnv_positions_M14A.csv
```

### 필터링 예시
```bash
# CNV 만
awk -F',' '$1=="CNV"' stk_cnv_positions.csv > cnv_only.csv

# 특정 장비
grep "4AFC3301A" stk_cnv_positions.csv

# 좌표 범위 확인
awk -F',' 'NR>1 {print $9, $10}' stk_cnv_positions.csv | \
    awk 'BEGIN{xmin=1e9; xmax=-1e9} {if($1<xmin)xmin=$1; if($1>xmax)xmax=$1} END{print xmin, xmax}'
```

---

## 8. 한계와 향후 개선

### 현재 한계
1. **slide 보간 미적용** — head Addr 좌표 그대로 사용. 정밀 좌표는 ±1~2m 오차 가능
2. **CNV 분류는 휴리스틱** — Oracle `SELECT_EQP_INF.EQPTYPE` 컬럼이 진짜 정답이지만 로컬에 없음
3. **AGV 미포함** — AGV 위치는 `AGV_CNV/AGV/I3AV0BMap.xml` 에 별도 형식으로 있음 (NODE/Position/X/Y)

### 향후 개선
- layout.xml 의 `NextAddr` 까지 읽어서 slide 보간 (`Mcp75Config.java` 로직 그대로 포팅)
- Oracle DB 덤프 받으면 분류 100% 정확하게 가능
- AGV 별도 추출 스크립트 추가 (I3AV0BMap.xml 파서)

---

## 9. 참고

### 관련 파일 위치
| 파일 | 경로 |
|---|---|
| 추출 스크립트 | `OHT2/layout/extract_stk_cnv_positions.py` |
| 출력 CSV | `OHT2/layout/stk_cnv_positions.csv` |
| 입력 station | `OHT2/station.dat` |
| 입력 layout | `OHT2/layout/layout/layout.zip` |
| Java 원본 좌표 계산 | `ALT/main/java/com/skhynix/smartatlas/data/raw/Mcp75Config.java:892~930` |
| Stocker 클래스 | `ALT/main/java/com/skhynix/smartatlas/data/eq/Stocker.java` |
| Conveyor 클래스 | `ALT/main/java/com/skhynix/smartatlas/data/eq/Conveyor.java` |
| EQP_TYPE 정의 | `ALT/main/java/com/skhynix/smartatlas/data/eq/Eqp.java:73` |

### OHT3 (M14A) FAB 폴더 구조
```
OHT3/MAP.zip
└── M14A/
    ├── A.layout.zip                 ← layout.xml 포함
    ├── A.MCP/*.cfg                  ← MCS 설정
    ├── A.station.dat                ← 포트 station 정보
    ├── STK_Master_M14A.csv          ← STK 직접 좌표 (6개)
    └── HID_Zone_Master_M14A_A.csv   ← HID Zone 마스터
```
