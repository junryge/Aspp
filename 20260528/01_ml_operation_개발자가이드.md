# 01_ml_operation 개발자 가이드 (상세본)

> 인계받는 개발자가 코드를 **수정 / 확장 / 디버그**할 수 있도록 작성한 레퍼런스.
> 빌드 가이드(`01_ml_operation_빌드가이드.md`)가 "무엇·왜"라면 이 문서는 "어떻게·어디를 만지나".

---

## 목차

1. [배경 & 목표](#1-배경--목표)
2. [시스템 아키텍처](#2-시스템-아키텍처)
3. [데이터 흐름](#3-데이터-흐름)
4. [환경 & 인프라 의존성](#4-환경--인프라-의존성)
5. [마스터 스케줄러 (Prediction_ml.py)](#5-마스터-스케줄러-prediction_mlpy)
6. [M14A_FAB 사이트 (Q_TRANSFER)](#6-m14a_fab-사이트-q_transfer)
7. [M16A_BR 사이트 (HUBROOM)](#7-m16a_br-사이트-hubroom)
8. [config 파일 키 전체 레퍼런스](#8-config-파일-키-전체-레퍼런스)
9. [Logpresso REST API 패턴](#9-logpresso-rest-api-패턴)
10. [Oracle dbquery 패턴](#10-oracle-dbquery-패턴)
11. [자바 → Python 매핑 (메서드 단위)](#11-자바--python-매핑-메서드-단위)
12. [확장 가이드 — 어디를 만지면 어떻게 되나](#12-확장-가이드--어디를-만지면-어떻게-되나)
13. [디버그 가이드 — 증상별 진단](#13-디버그-가이드--증상별-진단)
14. [테스트 & 검증](#14-테스트--검증)
15. [코드 컨벤션](#15-코드-컨벤션)
16. [트러블슈팅 FAQ](#16-트러블슈팅-faq)
17. [마이그레이션 히스토리 (커밋)](#17-마이그레이션-히스토리-커밋)
18. [용어집](#18-용어집)

---

## 1. 배경 & 목표

### 1.1 무엇을 대체하는가
SK Hynix FAB의 OHT/AMHS 모니터링 백엔드 **SmartAtlas** (Java/Spring/Quartz 기반) 의
**예측·알람 배치 3개**를 Python 으로 이식한다.

| 자바 클래스 | 역할 | Python |
|---|---|---|
| `HubroomTransPredictBatch.java` (282줄) | M16A Bridge Room 혼잡도 예측 + WARN_YN 판정 | `HubroomTransPredictBatch.py` |
| `QTransferPredictBatch.java` (1775줄) | M14A 반송 큐 예측 + 알람 11종 + 통계 + 적재 | `QTransferPredictBatch.py` |
| `QTransferDashBoardItemBatch.java` (216줄) | QTransfer 대시보드 행 생성 (REQUESTOR/WARNINGLOG/ERROR) | `QTransferDashBoardItemBatch.py` |

### 1.2 왜 Python 인가
- **모델 학습·추론**이 이미 Python (sklearn/Tensorflow) 으로 돌고 있음 (afagg)
- 자바 Quartz 배치는 모델을 호출하기 어렵고 운영 부담이 큼
- Logpresso REST API + numpy/pandas 로 자바 비즈니스 로직 100% 재현 가능

### 1.3 운영의 기존 분리 상태 (이미 Python 이 한 것)
| 단계 | 자바에서는 | Python (afagg) 에서는 |
|---|---|---|
| 데이터 수집 (SQL→CSV) | 자바 배치 내부 | `*_data_make.py` (CSV 생성) |
| 모델 추론 | 자바 → Python 호출 | `V8*.py` (모델 로드 + Logpresso 저장) |
| 알람·통계·대시보드 | 자바 배치 내부 | **이번 마이그레이션** |

→ 이번 작업은 **마지막 남은 ③④⑤** (알람 판정·통합 적재·대시보드) 를 Python 으로 마저 옮기는 것.

---

## 2. 시스템 아키텍처

### 2.1 전체 그림
```
                    [매분 정각]
                          │
                          ▼
              ┌─────────────────────────┐
              │  Prediction_ml.py       │  마스터 스케줄러
              │  (ThreadPoolExecutor)   │
              └─────────────────────────┘
                  │                │
        ┌─────────┘                └──────────┐
        ▼                                     ▼
  M16A_BR 사이트                         M14A_FAB 사이트
  (subprocess 순차)                      (subprocess 순차)
        │                                     │
        ├─ data_make.py        (CSV)          ├─ data_make.py       (CSV)
        ├─ Categorical 3종     (test_table)   ├─ V8.3.1 3종         (test_table2)
        ├─ Numerical 3종       (test_table)   ├─ QTransferPredict   (test_table5/6)
        └─ HubroomTransPredict (test_table)   └─ DashBoardItem      (test_table6)
```

### 2.2 사이트 분리 원칙
- M14A_FAB ↔ M16A_BR 는 **완전 독립** (서로 모름)
- 각자 `api_key.txt`, `*_config.json`, `data/`, `model/`, `__pycache__/` 보유
- 공유 모듈 없음 (`common/` 폴더 만들지 않음) — afagg 의 기존 컨벤션 유지

### 2.3 사이트 내부 실행 순서
**보장됨** (`subprocess.run` 순차):
1. data_make → CSV 생성
2. 예측기들 → 모델 추론 → Logpresso 저장
3. 알람/대시보드 배치 → 위 결과 읽어서 판정·적재

> 사이트 내 어느 한 스크립트가 실패해도 다음은 계속 진행 (`run_script()` 가 returncode 체크).

### 2.4 사이트 간 병렬
`concurrent.futures.ThreadPoolExecutor(max_workers=len(SITES))` 로 2개 사이트 동시.
사이클 시간 = max(M14A_FAB 사이트 시간, M16A_BR 사이트 시간) ≈ 5~10초.

---

## 3. 데이터 흐름

### 3.1 외부 데이터 소스

| 소스 | 접근 방식 | 용도 |
|---|---|---|
| **Logpresso** (`10.40.42.27:8888`) | HTTP REST (`/logpresso/httpexport/query.csv`) | 모든 쿼리·적재 |
| **Oracle** (mcs_m14a/m14b/m16a, star_m14, mcs_m14) | **Logpresso dbquery 경유** (oracledb 라이브러리 안 씀) | LFT 다운율, 스토리지 사용률, VHL 가동률, IDC 임계 |
| **ts_data_view_m14a / ts_current_job / bridge_layout_detail 등** | Logpresso 네이티브 테이블 | MES/MCS 이벤트, 반송 통계, 포트 상태 |
| **모델 .pkl** | 로컬 파일 (`model/`) | V8*예측기가 로드. 용량상 git 미포함. |

### 3.2 내부 적재 테이블 (현재 = 테스트)

| 테이블 | 구 운영 | 누가 씀 | 무엇이 들어가나 |
|---|---|---|---|
| `test_table` | test_hubroom_predict | V8_Categorical/Numerical 6종 + HubroomTransPredictBatch | Hubroom 예측 결과 + WARN_YN |
| `test_table2` | ATLAS_TS_PREDICT | V8.3.1 예측기 3종 | LSTM/STATE/STATE_PER/TIME (Q_TRANSFER 예측) |
| `test_table5` | test_currentjob_predict | QTransferPredictBatch | 통계 N행 + 예측 1행 + 알람 0~11행 |
| `test_table6` | qtransfer_dashboard | QTransferPredictBatch (STATE) + QTransferDashBoardItemBatch (나머지) | STATE / REQUESTOR / WARNINGLOG / ERROR 행 |
| ~~`test_table7`~~ | — | (사용 안 함) | — |

### 3.3 사이클당 적재 행 (정상)
빌드 가이드 §8 의 표 참조. 첫 사이클이 정상이면:
- test_table 9행, test_table2 3행, test_table5 ~15-20행, test_table6 ~5-8행
- ERROR 2행은 **누적 2시간 후** 생성

---

## 4. 환경 & 인프라 의존성

### 4.1 Python 런타임
- Python 3.8+
- 패키지: `requests`, `urllib3`, `pandas` (예측기만), `numpy` (예측기만), `sklearn` (예측기 모델 로드)
- 신규 배치 3개는 **표준라이브러리만** 사용 (`statistics`, `json`, `csv`, `urllib.parse`, `collections`)

### 4.2 운영 인프라 체크리스트
- [ ] `api_key.txt` 에 실제 Logpresso API 키 (M14A_FAB/, M16A_BR/ 각각)
- [ ] 모델 `.pkl` 이 `M14A_FAB/model/`, `M16A_BR/model/` 에 존재
- [ ] Logpresso `10.40.42.27:8888` 도달 가능
- [ ] Logpresso 에 `dbquery` 용 connection 등록: `mcs_m14`, `mcs_m14a`, `mcs_m14b`, `mcs_m16a`, `star_m14`
- [ ] 적재 테이블 5종 (`test_table`/`test_table2`/`test_table5`/`test_table6`) 사전 생성 (운영 테이블 스키마 그대로 — 빌드 가이드 §7 참조)
- [ ] data/ 폴더 쓰기 권한 (`M14A_FAB/data/QTRANSFER.csv` 등 생성)
- [ ] logs/ 폴더 쓰기 권한 (`Prediction_ml.log` 일자별 로테이션)

---

## 5. 마스터 스케줄러 (Prediction_ml.py)

### 5.1 책임
- 매분 **정각(00초)** 에 사이클 시작
- 2개 사이트 병렬 실행
- 사이트 내부 스크립트 순차 실행 (subprocess)
- 사이클별 결과 + 표준출력/에러를 일자별 로그 파일에 기록
- 한 스크립트 실패해도 다음 진행, 사이클 통째 예외도 다음 분 진행

### 5.2 핵심 자료구조 `SITES`
```python
SITES = {
    "M16A_BR": {
        "dir":     os.path.join(HERE, "M16A_BR"),
        "scripts": [
            "M16BR_hubroom_data_make.py --once",
            "V8_Categorical_Real_time_10min.py",
            "V8_Categorical_Real_time_15min.py",
            "V8_Categorical_Real_time_25min.py",
            "V8_Numerical_Real_time_10min.py",
            "V8_Numerical_Real_time_15min.py",
            "V8_Numerical_Real_time_25min.py",
            "HubroomTransPredictBatch.py --once",
        ],
    },
    "M14A_FAB": {
        "dir":     os.path.join(HERE, "M14A_FAB"),
        "scripts": [
            "M14A_FAB_data_make.py --once",
            "V8.3.1_Q_TRANSFER_PREDICTOR_10m.py",
            "V8.3.1_Q_TRANSFER_PREDICTOR_15m.py",
            "V8.3.1_Q_TRANSFER_PREDICTOR_25m.py",
            "QTransferPredictBatch.py --once",
            "QTransferDashBoardItemBatch.py --once",
        ],
    },
}
```

### 5.3 스크립트 추가하려면
- 단순히 `scripts` 리스트 끝에 `"새스크립트.py [args]"` 한 줄 추가
- **위치(맨 뒤/중간)** 가 의미 있음: 앞 스크립트 결과가 뒤에서 쓰이는 경우 (예: 알람 배치가 예측 결과를 읽음) 끝에 둠
- 1회만 실행돼야 하므로 반드시 `--once` 인자 받도록 작성
- 자체 while/cron 절대 금지

### 5.4 SCRIPT_TIMEOUT
```python
SCRIPT_TIMEOUT = 300  # 5분
```
사이클 60초인데 타임아웃 5분이라 한 스크립트가 늦으면 다음 사이클이 밀린다. 모니터링 포인트.

### 5.5 사이클 정각 동기화
```python
now = datetime.now()
sleep_sec = 60 - now.second - now.microsecond / 1_000_000
if sleep_sec < 0.5:
    sleep_sec += 60
time.sleep(sleep_sec)
```
사이클이 30초 걸리면 30초 sleep, 70초 걸리면 50초 sleep (다음 정각까지).

---

## 6. M14A_FAB 사이트 (Q_TRANSFER)

### 6.1 사이트 책임
- M14A FAB 의 반송 큐 (`ts_current_job`) 를 10분 후로 예측
- 임계치 초과 시 알람 11종 판정
- 통계(IQR/SD) + 예측값 + 알람 모두 `test_table5` 적재
- 예측 상태 한글 변환 + 대시보드 (REQUESTOR/WARNINGLOG/ERROR) `test_table6` 적재

### 6.2 파일 인벤토리

| 파일 | 역할 | 신규/기존 |
|---|---|---|
| `M14A_FAB_data_make.py` | Logpresso → QTRANSFER.csv (300분치) | 기존 |
| `V8.3.1_Q_TRANSFER_PREDICTOR_{10,15,25}m.py` | 모델 로드 + 추론 + test_table2 저장 | 기존 |
| `QTransferPredictBatch.py` | 통계 + 알람 11종 + 적재 + STATE 대시보드 | **★신규** |
| `QTransferDashBoardItemBatch.py` | REQUESTOR + WARNINGLOG + ERROR 대시보드 | **★신규** |
| `m14a_config.json` | 데이터수집/예측기용 (Logpresso host/port/insert_table 등) | 기존 |
| `qtransfer_alarm_config.json` | 알람배치+대시보드배치용 (임계치/메시지/쿼리) | **★신규** |
| `api_key.txt` | 빈 파일. 운영서버에서 실제 키 채워야 함 | 기존 |

### 6.3 QTransferPredictBatch.py 메서드별 상세

#### `__init__()` / `execute()` / `_run()`
- `__init__`: `self.event_dt = None` (사이클 타임스탬프 저장용)
- `execute()` → `_run()` 호출 (자바 `execute` 의 Quartz 진입점 모방)
- `_run()` 자바 L54-226 의 6단계를 그대로 따라감

#### `_fetch_prediction(self)` → dict
```python
q = f'table duration=5m {PREDICT_TABLE} | sort -TIME | limit 1'
```
- `PREDICT_TABLE` = test_table2 (V8.3.1 예측기 출력)
- 최신 1건 가져옴. 비어있으면 `{}` 반환 → `prediction=None` → 알람·통계 분기 전체 스킵
- **여기가 HTTP 500 나오면 전체 동작 불가** (가장 먼저 의심)

#### `_insert_prediction_state(self, prediction_map)` (자바 L369-443)
- `STATE` 컬럼 (NORMAL/CAUTION/CRITICAL) 을 한글 (정상/주의/심각) 로 변환
- `STATE_PER` → `VAL` (위험확률 %)
- `DASHBOARD_TABLE` = test_table6 에 1행 적재
- TYP="STATE"

#### `_get_pivot_data(self, prediction)` (자바 L450-535)
- `qTransferGroupData` 쿼리 실행 → 최신 1행 + 과거 N행
- 최신 행의 IDC 컬럼(`M14AM10A`, `M14AM16` 등) 각각에 대해:
  - `calc_average` (numpy 없이 `statistics.fmean`)
  - `calc_iqr` (자바 `_getQuantile` L1215 의 floor 보간식 그대로 — `_quantile()` 함수)
  - `calc_std` (모집단 표준편차 — `statistics.pstdev`)
- 1.5×IQR 또는 평균±3σ 벗어나면 JUDGE="T", 아니면 "F"
- 마지막에 `_alarm_valid()` 호출하여 ALARM1~5 평가

#### `_alarm_valid(self, origin, prediction)` (자바 L537-1157)
- 전제: `prediction > QTRANSFER_LIMIT(1700)` 이어야 ALARM1~5 평가 (자바 L579)
- ALARM1~5 각각 try/except 로 감싸 한 알람 실패해도 다음은 진행
- 결과: 알람 row 리스트 (`_alarm_base()` 베이스 + IDCCOL/메시지 채움)

#### ALARM1~5 빌더 상세 (자바 line refs)

##### ALARM1 (자바 L582-655)
```python
vhl_rows = query(QUERIES["vhlRunRate"])  # dbquery mcs_m14
# FOUP 가동률 ≥ 95
v24 = query(mes_q.replace("duration=1m", "duration=24h"))  # 24h MES 큐
v10 = query(mes_q.replace("duration=24h", "duration=10m")) # 10m MES 큐
# (v10 - v24) / v24 * 100 > 10
if vhl_flag and mes_flag:
    alarm["IDCCOL"] = "QTRANSFER_ALARM1"
    alarm["ALARM_MSG_CTN"] = get_message("QTRANSFER_REQ_CONTENT", ...)
```

##### ALARM2 (자바 L657-749)
- `transferCountPerMin` 10m vs 24h → M14AM16SUM 10%↑
- `bridgeDetailCnvPortDownRate` → PORT_DOWN_RATE < 10
- 만족 시 `_oracle_storage_util()` 로 M16A 스토리지값 채움

##### ALARM3 (자바 L658-1040)
- M14AM10ASUM 10%↑
- `_oracle_query("SELECT_M14TOM10LFT_DOWN_RATE", "PORT_DOWN_RATE")` < 25
- M10A↔M14A LFT 장비 다운율 부족 시 알람

##### ALARM4 (자바 L1042-1106)
- M14AM14BSUM 10%↑
- `_oracle_query("SELECT_M14LFT_DOWN_RATE", "PORT_DOWN_RATE")` < 10

##### ALARM5 (자바 L757-940)
- `_build_alarm5()` 별도 메서드. 가장 복잡함.
- `qtransferAltJobCnt` → 1시간 ALT/ERROR JOB 비율 > 50
- `altJobMachineRate` → 동일 장비 비율 ≥ 50
- `collections.defaultdict` 로 자바 `groupingBy(summingInt)` 재현
- 24h 평균은 `altJobMachineRate24h`, `qtransferAltJobCnt24h` 로 별도 조회

#### `_build_transport_alarm(self)` (자바 L1255-1369) — ALARM6~11
- `SELECT_AWS_IDC_HISTORY` 쿼리 1회 (`dbquery star_m14` Oracle)
- 8개 IDC 컬럼별 임계 비교 → ALARM6~11 판정
- ALARM10/11 은 **AND 조건 2개** (Total Que & Manual Out / 층간 대기 & STB 부하)

#### `_oracle_query(self, query_id, col)`
- `oracle.enabled=true` 이면 `oracle.queries[query_id]` 의 `dbquery mcs_m14a SELECT...` 를 Logpresso 로 전송
- Logpresso 가 Oracle 에 transparent 하게 위임 → 결과 CSV 로 받음
- **`oracledb` 라이브러리 불필요** (요청대로)

#### `_reorder_and_finalize(self, rows, prediction, prediction_map)` (자바 L130-212)
- `DESIRED_ORDER` 20개 컬럼 순서로 강제 정렬
- TOTALCNT 행에만 `LSTM` (예측값) + `LSTM_JUDGE` (실측 대비 10%↑ 차이 → Y)
- 모든 float 값 소수 2자리 반올림
- `TIME` = yyyyMMddHHmm 형식 (event_dt)

### 6.4 QTransferDashBoardItemBatch.py 메서드별 상세

#### `_run()` (자바 L51-90)
1. `_build_requestor()` → REQUESTOR 행 N개
2. `_build_mcs_error_log_count()` → WARNINGLOG 행 0~3개
3. `_build_trans_que_predict_error()` → ERROR 행 2개
4. 각 row 에 `EVENT_DT` / `DUE_GBN_CD` 부여 (yyyyMMddHHmm)
5. `_save_one()` 반복

#### `_build_requestor()` (자바 L107-130)
- `requestorCount` 쿼리 본문에 `SEARCHSYSTEM` 자리 → `REQUESTOR_LIST` 치환
- `dashboard_variables.QTRANSFER_REQUESTOR_LIST` 기본 `"RTD/RTS","EIS","ETC","OFS","MCS"`
- 결과 row: `{REQUESTOR: "RTD/RTS", REQUESTOR_CNT: 29}` → `OPER_ACT_CTN="RTD/RTS"`, `VAL=29`, `TYP="REQUESTOR"`

#### `_build_mcs_error_log_count()` (자바 L132-144)
- `mcsErrorLogCnt` 쿼리 (1h ts_data_view_m14a LEVEL 분류)
- WARN / ERROR / EXCEPTION 카운트

#### `_build_trans_que_predict_error()` (자바 L146-159) — ★ ERROR 핵심
- `quePredictError` 쿼리 본문에 박힌 `test_currentjob_predict` 를 `insert_table`(test_table5) 로 자동 치환
- 쿼리가 반환하는 1 행: `{ERROR_VALUE: 53.03, ERROR_RATE: 95.98}`
- `_build_map_unpivot("ERROR", data)` 가 (k,v) 펼침 → 2 행 생성:
  - `OPER_ACT_CTN="ERROR_VALUE", VAL=53.03, TYP="ERROR"`
  - `OPER_ACT_CTN="ERROR_RATE", VAL=95.98, TYP="ERROR"`

#### `_build_map_keyval` vs `_build_map_unpivot` (자바 L161-199)
**자바 `_buildMap` 의 두 오버로드를 Python 에서 분리한 것**

| Python | 자바 시그니처 | 동작 |
|---|---|---|
| `_build_map_keyval(content_field, value_field, type, data)` | `_buildMap(String, String, String, List)` | 지정 컬럼 2개로 1 행 → 1 OPER_ACT_CTN row |
| `_build_map_unpivot(type, data)` | `_buildMap(String, List)` | 1 행의 모든 (k,v) 펼침 → 여러 OPER_ACT_CTN row |

ERROR 행이 2개인 이유 = `_build_map_unpivot` 사용 (쿼리 결과 1행에 컬럼 2개).

---

## 7. M16A_BR 사이트 (HUBROOM)

### 7.1 사이트 책임
- M16A Bridge Room 의 OHT/캐리어 혼잡도를 10/15/25분 후로 예측
- 직전 사이클 결과와 비교해 **연속 2회 위험** 시 WARN_YN="Y" 판정
- 적재 테이블 `test_table`

### 7.2 파일 인벤토리

| 파일 | 역할 | 신규/기존 |
|---|---|---|
| `M16BR_hubroom_data_make.py` | Logpresso → HUBROOM_PIVOT_DATA.csv (30분치) | 기존 |
| `V8_Categorical_Real_time_{10,15,25}min.py` | 분류 모델 (CLASS_0/1/2) | 기존 |
| `V8_Numerical_Real_time_{10,15,25}min.py` | 회귀 모델 (JUDGEVAL) | 기존 |
| `HubroomTransPredictBatch.py` | WARN_YN 판정 | **★신규** |
| `m16br_config.json` | 데이터수집/예측기용 | 기존 |
| `hubroom_alarm_config.json` | WARN 판정용 | **★신규** |

### 7.3 HubroomTransPredictBatch.py 메서드별 상세

#### `_run()`
- `FLOWS = ["HUBROOM_PREDICT", "HUBROOM_PREDICT_15MIN", "HUBROOM_PREDICT_25MIN"]` 3개에 대해 반복
- 한 FLOW 실패해도 다음 FLOW 진행 (try/except)

#### `_process_flow(self, flow, event_dt)`
```python
# 1) 현재 예측 1건
cur_q = (f'table duration=30m {INSERT_TABLE} '
         f'| search FLOW == "{flow}" | sort -CURRENT_TIME | limit 1')
# 2) 직전까지 2건 (두번째 = 직전 예측)
prev_q = (f'table duration=30m {INSERT_TABLE} '
          f'| search FLOW == "{flow}" | sort -CURRENT_TIME | limit 2')
# 3) WARN_YN 판정
# 4) test_table 에 WARN_YN/FLOW/EVENT_DT 붙여 갱신 적재
```

#### `_valid_warn_yn(cur, prev)` (자바 L143-169)
```
prev is None        → "N"   (최초 사이클)
prev >= 2           → "N"   (위험도 미정)
prev == 1 & cur=="1" → "Y"  (연속 2회 위험 ★)
그 외                → "N"
```

### 7.4 errorVhlVal / VHL_OFF_PLUS_DATA 안 한 이유
자바에서 데이터 수집(CSV 생성) 단계 로직 — `M16BR_hubroom_data_make.py` 가 이미 담당 중.
**사용자 확인 완료**: 본 배치에서 다시 할 필요 없음.

---

## 8. config 파일 키 전체 레퍼런스

### 8.1 `qtransfer_alarm_config.json` (M14A_FAB)

```jsonc
{
  "logpresso": {
    "host": "10.40.42.27",        // Logpresso 서버
    "port": 8888,                  // httpexport 포트
    "insert_table": "test_table5", // 통계+예측+알람 적재
    "predict_table": "test_table2",// V8.3.1 예측기 출력 (읽기만)
    "dashboard_table": "test_table6" // 대시보드 적재
  },
  "thresholds": {
    // _findValueByCode 가 참조. 자바 variable.xml 의 운영값.
    "QTRANSFER_LIMIT": 1700.0,     // ALARM1~5 트리거 임계 (예측값)
    "QTRANSFER_INCREASE_RATE": 10.0, // 10m vs 24h 증가율 (%)
    "VHL_RATE_LIMIT": 95.0,        // ALARM1 VHL FOUP 가동률
    "QTRANSFER_CNV_PORT_DOWN_RATE": 10.0,  // ALARM2 CNV 포트 다운율
    "QTRANSFER_M14TOM10_PORT_DOWN_RATE": 25.0, // ALARM3
    "QTRANSFER_M14LFT_PORT_DOWN_RATE": 10.0,   // ALARM4
    "QTRANSFER_ALT_JOB_LIMIT": 50.0,           // ALARM5 ALT JOB 비율
    "QTRANSFER_ALT_JOB_MACHINE_RATE": 50.0,    // ALARM5 장비 비율
    "Q_TRANSFER_BOUNDARY_DEFAULT_6": 95.0,     // ALARM6
    "Q_TRANSFER_BOUNDARY_DEFAULT_7": 100.0,    // ALARM7
    "Q_TRANSFER_BOUNDARY_DEFAULT_8": 100.0,    // ALARM8
    "Q_TRANSFER_BOUNDARY_DEFAULT_9": 2700.0,   // ALARM9
    "Q_TRANSFER_BOUNDARY_DEFAULT_10_1": 1700.0, // ALARM10 첫번째 AND
    "Q_TRANSFER_BOUNDARY_DEFAULT_10_2": 200.0,  // ALARM10 두번째 AND
    "Q_TRANSFER_BOUNDARY_DEFAULT_11_1": 500.0,  // ALARM11
    "Q_TRANSFER_BOUNDARY_DEFAULT_11_2": 97.0
  },
  "messages": {
    // alarm_message.xml 의 메시지 코드 → 텍스트.
    // {0}{1}... 자리표시자는 get_message(code, *args) 가 치환 (자바 MessageFormat).
    "QTRANSFER_REQ_TITLE": "...",
    "QTRANSFER_REQ_CONTENT": "1. 10분 뒤 M14A 반송큐 수: {0}개 (알람 기준치 {1}개)...",
    // ... 27종
  },
  "queries": {
    // customQuery.xml / customQuery2.xml 알람용 쿼리 11종
    "qTransferGroupData": "...",       // 통계용
    "vhlRunRate": "dbquery mcs_m14 ...", // ALARM1 (Oracle dbquery)
    "mesOHTQCntAlarmValid": "...",     // ALARM1
    "completedTsJobCountPerMin": "...",// ALARM1 참고
    "transferCountPerMin": "...test_table5...", // ALARM2/3/4 (INSERT_TABLE 직접 박힘 ★)
    "bridgeDetailCnvPortDownRate": "...", // ALARM2
    "qtransferAltJobCnt": "...",       // ALARM5
    "altJobMachineRate": "...",        // ALARM5
    "altJobMachineRate24h": "...",     // ALARM5 24h 평균
    "qtransferAltJobCnt24h": "...",    // ALARM5 24h 평균
    "SELECT_AWS_IDC_HISTORY": "dbquery star_m14 SELECT ..." // ALARM6~11 (Oracle)
  },
  "dashboard_queries": {
    // QTransferDashBoardItemBatch 가 사용. customQuery2.xml.
    "requestorCount": "...",     // REQUESTOR
    "mcsErrorLogCnt": "...",     // WARNINGLOG
    "quePredictError": "...test_currentjob_predict..." // ERROR (자동 치환됨)
  },
  "dashboard_variables": {
    "QTRANSFER_REQUESTOR_LIST": "\"RTD/RTS\",\"EIS\",\"ETC\",\"OFS\",\"MCS\""
  },
  "oracle": {
    "enabled": true,
    "queries": {
      "SELECT_M14TOM10LFT_DOWN_RATE": "dbquery mcs_m14a SELECT ...", // ALARM3
      "SELECT_M14LFT_DOWN_RATE": "dbquery mcs_m14b SELECT ...",      // ALARM4
      "SELECT_M16A_STORAGE_UTIL": "dbquery mcs_m16a SELECT ..."      // ALARM2/3/4 참고
    }
  }
}
```

### 8.2 `hubroom_alarm_config.json` (M16A_BR)

```jsonc
{
  "logpresso": {
    "host": "10.40.42.27",
    "port": 8888,
    "insert_table": "test_table"
  },
  "flows": ["HUBROOM_PREDICT", "HUBROOM_PREDICT_15MIN", "HUBROOM_PREDICT_25MIN"],
  "prev_judgeval_query": "table duration=30m test_table | search FLOW == \"{FLOW}\" | sort -CURRENT_TIME | limit 1 | fields JUDGEVAL"
}
```

> 현재 `prev_judgeval_query` 는 참조만 (실제 쿼리는 `_process_flow` 안에 인라인). 향후 분리 가능.

---

## 9. Logpresso REST API 패턴

### 9.1 조회 (`query()` 헬퍼)
```python
def query(q, timeout=180):
    qs = " ".join(q.split())  # 여러 공백 1개로
    url = f"{BASE}?_apikey={API_KEY}&_q={urllib.parse.quote(qs, safe='')}"
    r = requests.get(url, verify=False, timeout=timeout)
    if r.status_code != 200 or r.text.strip().startswith("<"):
        return []  # HTML 에러 페이지 등
    return list(csv.DictReader(StringIO(r.text)))
```

**500 의 흔한 원인:**
- 존재하지 않는 테이블 (`predict_table` 오타)
- 없는 컬럼으로 sort (`sort -EVENT_DT` 인데 row 에 없음)
- `dbquery` 의 Oracle connection 미등록
- 쿼리 문법 오류 (인용부호 안 닫힘)

### 9.2 적재 (`_save_one()` 헬퍼)
```python
parts = []
for k, v in row.items():
    if v is None:
        parts.append(f"{k} = null")
    else:
        s = str(v).replace("'", "\\'")
        parts.append(f"{k} = '{s}'")
literal = "{" + ", ".join(parts) + "}"
escaped = literal.replace('"', '\\"')
q = f'json "{escaped}" | import {table}'
```

**적재 실패 흔한 원인:**
- API_KEY 빈 값
- 테이블 미존재 (사전 생성 안 됨)
- row 에 null 또는 NaN 이 있고 Logpresso 컬럼이 NOT NULL
- 컬럼 타입 mismatch (수치 컬럼에 한글 문자열 등)

### 9.3 시간 범위 — `duration=Nm` vs `from=dateadd(...) to=now()`
**전자 권장.** 후자는 일부 Logpresso 환경에서 HTTP 500.

### 9.4 인증
`api_key.txt` 1줄 (개행 trim). 환경변수 `LOGPRESSO_API_KEY` 도 fallback.

---

## 10. Oracle dbquery 패턴

### 10.1 어떻게 동작하나
```
dbquery mcs_m14a
SELECT * FROM nt_r_unit WHERE machinename LIKE '4ABL33%' ...
| eval ...
| fields ...
```
- `dbquery <connection_id>` 는 Logpresso 의 명령
- Logpresso 가 자체적으로 Oracle 에 SELECT 위임
- 결과를 일반 row stream 으로 받아서 이후 파이프 (`| eval`, `| stats`) 적용
- Python 입장에선 그냥 Logpresso 쿼리 1건

### 10.2 운영에 등록돼야 하는 connection
- `mcs_m14`, `mcs_m14a`, `mcs_m14b`, `mcs_m16a` — mybatis mapper 와 동일 이름
- `star_m14` — SELECT_AWS_IDC_HISTORY 용

### 10.3 왜 oracledb 안 썼나
- 자바 mybatis SQL 본문이 동일하게 동작
- Logpresso dbquery 가 connection pooling 도 알아서 함
- `oracledb` 패키지 추가 = Oracle Client 설치 + DSN 관리 부담

---

## 11. 자바 → Python 매핑 (메서드 단위)

### 11.1 QTransferPredictBatch
| 자바 메서드 (줄번호) | Python 메서드 | 비고 |
|---|---|---|
| `_run()` L54-226 | `_run()` | 흐름 1:1 |
| `_insertPredictionState()` L369-443 | `_insert_prediction_state()` | STATE 행만 (ERROR/WARNINGLOG/REQUESTOR 은 DashBoardItem 으로 분리) |
| `_getPivotData()` L450-535 | `_get_pivot_data()` | numpy 대신 `statistics` |
| `_alarmValid()` L537-1157 | `_alarm_valid()` | ALARM1~5 try/except 분리 |
| `_buildAlarmBase()` L1374 | `_alarm_base()` | dict factory |
| `_buildTransportAlarm()` L1255-1369 | `_build_transport_alarm()` | ALARM6~11 |
| `_calculate*()` L1159-1237 | `calc_average/std/iqr/_quantile` | statistics 표준라이브러리 |
| `_findValueByCode()` L1736 | `find_value(code, default)` | thresholds dict 조회 |
| `XmlUtil.getMessage()` | `get_message(code, *args)` | `{N}` → str.replace 치환 |
| `_collectIdcValue()` L1754 | `collect_idc_value(rows, idc_col)` | |

### 11.2 HubroomTransPredictBatch
| 자바 | Python |
|---|---|
| `_validWarnYN()` L114-177 (JUDGEVAL 분기) | `_valid_warn_yn(cur, prev)` |
| `execute()` | `execute()` → `_run()` → `_process_flow()` |

### 11.3 QTransferDashBoardItemBatch
| 자바 | Python |
|---|---|
| `_run()` L51-90 | `_run()` |
| `insertLogpressoData()` L92-105 | `_save_one()` 반복 (afagg 패턴) |
| `_buildRequestor()` L107-130 | `_build_requestor()` |
| `_buildMcsErrorLogCount()` L132-144 | `_build_mcs_error_log_count()` |
| `_buildTransQuePredictError()` L146-159 | `_build_trans_que_predict_error()` |
| `_buildMap(String, List)` L161-173 | `_build_map_unpivot()` |
| `_buildMap(String, String, String, List)` L175-199 | `_build_map_keyval()` |
| `_getMapData()` L201-215 | `_get_map_data()` |

---

## 12. 확장 가이드 — 어디를 만지면 어떻게 되나

### 12.1 임계치 조정
`qtransfer_alarm_config.json` → `thresholds.<KEY>` 만 바꾸면 즉시 반영.
재시작 필요 (모듈 로드 시점에 읽음).

### 12.2 새 알람 추가 (예: ALARM12)
1. `_alarm_valid()` 또는 `_build_transport_alarm()` 안에 새 try/except 블록
2. `thresholds` 에 `Q_TRANSFER_BOUNDARY_DEFAULT_12` 추가
3. `messages` 에 `ALARM12_TITLE`/`_CONTENT` 추가
4. 필요한 쿼리는 `queries` 에 추가
5. `_alarm_base()` 베이스로 row 생성 → `IDCCOL = "QTRANSFER_ALARM12"`
6. 결과 row 를 `alarms.append()` 또는 `result.append()`

### 12.3 새 데이터 소스 (예: 다른 Logpresso 테이블 / Oracle SQL)
- Logpresso 테이블: `queries.<새키>` 에 쿼리 본문 추가 → `query(QUERIES["<새키>"])`
- Oracle: `oracle.queries.<새키>` 에 `dbquery <conn> SELECT ...` 추가 → `_oracle_query("<새키>", "<컬럼>")`

### 12.4 새 사이트 추가
- 새 폴더 `M??_??/` 만들고 자체 `api_key.txt` / `*_config.json` / scripts 둠
- `Prediction_ml.py` 의 `SITES` 에 한 항목 추가
- 사이트 간 독립 원칙 유지 (공유 모듈 X)

### 12.5 운영 테이블로 전환 (테스트 → 운영)
config 의 다음 키만 원복:
- hubroom `insert_table`: test_table → test_hubroom_predict
- qtransfer `insert_table`: test_table5 → test_currentjob_predict
- qtransfer `predict_table`: test_table2 → ATLAS_TS_PREDICT  (V8.3.1 도 같이 바꿔야 함)
- qtransfer `dashboard_table`: test_table6 → qtransfer_dashboard
- **쿼리 본문 안에 박힌 테이블명 2곳도 같이**:
  - `queries.transferCountPerMin` (test_table5 → test_currentjob_predict)
  - `dashboard_queries.quePredictError` (test_currentjob_predict 자동 치환되므로 본문은 그대로 둬도 OK)

### 12.6 새 대시보드 행 종류 추가
1. `dashboard_queries.<새키>` 에 쿼리 추가
2. `QTransferDashBoardItemBatch.py` 의 `_run()` 에 `_build_<새타입>()` 호출 추가
3. `_build_map_keyval` 또는 `_build_map_unpivot` 으로 row 생성

---

## 13. 디버그 가이드 — 증상별 진단

### 13.1 사이클은 도는데 적재 0건
**의심 순서:**
1. `api_key.txt` 비어있나? → `cat M14A_FAB/api_key.txt` 확인
2. `test_table2` 에 V8.3.1 결과 있나? → `table duration=5m test_table2 | limit 1` Logpresso 에서 직접
3. `_fetch_prediction` 결과가 `{}` 이면 다음 분기 다 스킵 → 위 2번
4. `qTransferGroupData` 가 비어있으면 통계 0행 → ts_current_job 데이터 확인

### 13.2 ❌ 쿼리 HTTP 500
- **테이블 미존재** (predict_table 오타가 가장 흔함)
- **dbquery connection 미등록**
- **쿼리 문법 오류** (특히 `sort -<없는컬럼>`)
- **시간범위 구문** (`from=dateadd(...) to=now()` 는 `duration=Nm` 로)

### 13.3 ❌ API_KEY 없음
- `api_key.txt` 가 빈 파일
- 환경변수 `LOGPRESSO_API_KEY` 도 미설정
- → `echo "<키>" > M14A_FAB/api_key.txt`

### 13.4 알람이 안 뜬다
1. **`LSTM < 1700` 이면 정상**. ALARM1~5 안 뜸 (자바 동일)
2. 임계치 보고 진짜 충족하는지 한 줄씩 확인
3. ALARM1~11 별로 각자 try/except → 한 알람 실패해도 다른 알람은 진행 → 로그에 `⚠ ALARM? 스킵: ...` 떴는지

### 13.5 ERROR_RATE/ERROR_VALUE 가 test_table6 에 안 들어옴
1. **`quePredictError` 가 test_table5 의 지난 2시간 TOTALCNT 행을 필요로 함**
2. test_table5 가 비어있으면 결과 0 → ERROR 행 0
3. 운영 시작 후 **2시간 이상 누적** 되어야 첫 ERROR 행 나옴 (자바 동일)

### 13.6 WARN_YN 이 계속 "N" 만 뜸
- `_valid_warn_yn` 의 prev/cur 확인 (로그에 출력됨: `cur_judge=? prev_judge=?`)
- prev 가 None 이면 첫 사이클 → 정상
- prev 가 2 이상이면 무조건 "N" → V8_Numerical 의 JUDGEVAL 출력 확인

### 13.7 모듈 임포트 에러 / 표준라이브러리 못 찾음
- `statistics` 는 Python 3.4+ 표준. 운영서버 Python 버전 확인.
- `requests` 미설치 → `pip install requests`

---

## 14. 테스트 & 검증

### 14.1 1회 실행 테스트
```bash
cd 01_ml_operation
python3 Prediction_ml.py --once  # 한 사이클만 돌고 종료
```

### 14.2 개별 배치 테스트 (다른 스크립트 영향 없이)
```bash
cd M14A_FAB
python3 QTransferPredictBatch.py --once   # 통계+알람
python3 QTransferDashBoardItemBatch.py --once  # 대시보드
cd ../M16A_BR
python3 HubroomTransPredictBatch.py --once
```

### 14.3 자바 vs Python 비교 (shadow)
- 운영 자바 배치는 운영 테이블에 적재 중
- Python 은 테스트 테이블 (test_table*) 에 적재
- 같은 시각에 두 결과 row 비교:
  - 알람 건수 일치?
  - IDCCOL / ALARM_YN / 메시지 텍스트 일치?
- 1주 일치 확인 후 자바 배치 비활성화 + Python config 운영 테이블로 전환

### 14.4 단위 검증 (제안 — 아직 안 만듦)
- `_valid_warn_yn(prev=1, cur='1')` → "Y"
- `calc_iqr([1,2,3,4,5,6,7,8,9,10])` → Q1=3.25, Q3=7.75 (예시)
- `_alarm_base()` 베이스 dict 키 13종 확인
- `get_message("FOO", "a", "b")` → 자리표시자 치환 정확도

---

## 15. 코드 컨벤션

### 15.1 자바 메서드명 → Python
- camelCase → snake_case (`_alarmValid` → `_alarm_valid`)
- 자바 private (_prefix) → Python 도 _prefix (관례)
- 자바 클래스 그대로 Python 클래스 (1:1) — 단 매우 짧으면 함수만으로도

### 15.2 쿼리 보내기 전 정규화
```python
qs = " ".join(q.split())  # 모든 공백/개행을 단일 공백으로
```
Logpresso REST 가 multi-line 쿼리 거부할 수 있어서.

### 15.3 try/except 범위
- 알람 1개씩 try/except → 한 알람 실패가 다른 알람 막지 않게
- 사이클 전체는 마스터(`Prediction_ml.py`) 가 try/except → 다음 사이클 진행

### 15.4 numpy/pandas 의존성
- 신규 배치는 **표준라이브러리만** (`statistics`/`collections`)
- 이유: 운영서버 환경 변동 영향 최소화
- 기존 예측기는 numpy/pandas/sklearn 사용 (모델용)

### 15.5 print vs logger
- 개별 배치는 print (subprocess 의 stdout 으로 마스터 로그에 통합됨)
- 마스터(`Prediction_ml.py`) 만 `logging.TimedRotatingFileHandler`

### 15.6 인라인 테이블명 치환
config 의 쿼리 본문 안에 박힌 테이블명은 `.replace(old, new)` 로 자동 치환 처리.
config 키 (`insert_table` 등) 와 쿼리 본문이 일치하도록 항상 같이 수정.

---

## 16. 트러블슈팅 FAQ

**Q. test_table2 에 행이 안 들어와요.**
A. V8.3.1 예측기 (10m/15m/25m) 실패. 모델 `.pkl` 또는 데이터 CSV 또는 api_key 확인.
   → QTransferPredictBatch 가 동작 안 함 (predict_table 비어있어서).

**Q. test_table5 에 알람 행이 안 들어와요.**
A. LSTM 값이 1700 미만이면 정상 (자바 동일 동작). ALARM1~5 안 뜸. ALARM6~11 은 IDC 임계 충족 시만.

**Q. 자바와 알람 시점이 1~2분 다릅니다.**
A. Python 마스터는 매분 정각 (00초) 시작. 자바 Quartz cron 설정과 다를 수 있음.
   사이클 시간 자체는 5~10초 이므로 시점 차이는 정상.

**Q. 운영서버에 Python 3.8 이상이 없습니다.**
A. `statistics.fmean` 은 3.8+. 미만이면 `statistics.mean` 으로 대체 가능 (정밀도 약간 다름).
   `f-string =` 디버그 표현(`f"{x=}"`) 안 쓰면 3.6+ 도 동작.

**Q. Oracle dbquery 가 안 됩니다.**
A. Logpresso 에 connection 등록 안 됨. `mcs_m14a`/`mcs_m14b`/`mcs_m16a`/`mcs_m14`/`star_m14` 5종 확인.

**Q. Logpresso 가 `<html><body>` 응답을 반환합니다.**
A. 인증 실패 (API_KEY 틀림) 또는 권한 부족. `query()` 가 `< 로 시작 시 빈 리스트 반환` 처리는 이미 함.

**Q. 새 알람 메시지에 줄바꿈을 넣고 싶습니다.**
A. `messages` JSON 에서 `\n` 또는 `\\n`. Logpresso `import` 단계에서 어떻게 보존되는지 운영 자바 메시지 형식 참고.

---

## 17. 마이그레이션 히스토리 (커밋)

브랜치: `claude/decode-base64-file-dEHNN`

| 커밋 | 내용 |
|---|---|
| `b654c33` | ALT/opss/decoded 폴더 정리 + README |
| `5394d61` | config: 적재 테이블을 테스트 테이블로 변경 |
| `920c354` | decoded.zip 압축본 |
| `ac986af` | 빌드 가이드 MD 추가 |
| `cafed0d` | HTTP 500 4건 수정 (predict_table=test_table2, sort -CURRENT_TIME, duration=Nm) |
| `688e3a6` | dashboard_table test_table7 → test_table6 |
| `d66bd9b` | QTransferDashBoardItemBatch.py 신규 (REQUESTOR/WARNINGLOG/ERROR) |
| `1e87025` | 빌드 가이드 갱신 (대시보드 배치 + 매핑 변경 반영) |
| `f9c836b` | 빌드 가이드 첫 사이클 표에 test_table2 추가 |

---

## 18. 용어집

| 용어 | 뜻 |
|---|---|
| **AMHS** | Automated Material Handling System (반도체 자동물류) |
| **OHT** | Overhead Hoist Transport (천장 반송차) |
| **FOUP** | Front Opening Unified Pod (300mm 웨이퍼 캐리어) |
| **MES** | Manufacturing Execution System |
| **MCS** | Material Control System (반송 컨트롤) |
| **MCP** | Material Control Platform |
| **VHL** | Vehicle (반송차 — OHT 등) |
| **CNV** | Conveyor (컨베이어 반송) |
| **LFT** | Lifter (층간 반송 리프트) |
| **STB** | Stocker Buffer (스토커 버퍼) |
| **IDC** | Indicator (지표 컬럼) |
| **FAB** | Fabrication Plant (반도체 공장 동) |
| **Bridge Room** | M16A 의 라인 간 연결 공간 |
| **JUDGEVAL** | V8_Numerical 모델 출력의 위험도 판정값 (0/1/2) |
| **WARN_YN** | 연속 2회 위험 시 "Y" (자바 _validWarnYN 결과) |
| **LSTM** | Long Short-Term Memory (Q_TRANSFER 예측 모델 출력값 컬럼명) |
| **STATE** | 예측 상태 (NORMAL/CAUTION/CRITICAL) → 한글 (정상/주의/심각) |
| **STATE_PER** | 위험 확률 (%) |
| **dbquery** | Logpresso 가 Oracle 에 위임하는 명령 |
| **httpexport** | Logpresso REST API 엔드포인트 |
| **Quartz** | Java 의 cron-like 스케줄러 (자바 배치가 쓰던 것) |
| **afagg** | 운영중인 Python ML 시스템의 원본 폴더명 (`q.txt` 디코딩 결과) |
| **shadow** | 운영과 병행 실행해 결과 비교하는 검증 방식 |
