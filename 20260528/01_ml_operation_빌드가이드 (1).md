# 01_ml_operation 빌드 가이드 — 자바 배치 → Python 마이그레이션 상세

> SmartAtlas 자바 Quartz 배치 **3개**(`HubroomTransPredictBatch.java`, `QTransferPredictBatch.java`,
> `QTransferDashBoardItemBatch.java`)를 Python ML 운영 시스템(`Prediction_ml.py` 마스터)으로
> 이식한 전 과정.
> **무엇을 자바에서 참고했고 / 무슨 데이터를 보고 / 어디에 저장하는지** 를 코드 라인 단위로 정리.

---

## 0. 한눈에 — 전체 구조

```
01_ml_operation/
├── Prediction_ml.py              ★마스터 스케줄러 (매분 정각, 2사이트 병렬 subprocess)
│
├── M14A_FAB/                     ▼ Q_TRANSFER 사이트 (반송 큐 예측 + 알람 11종 + 대시보드)
│   ├── M14A_FAB_data_make.py            [기존] Logpresso→CSV (QTRANSFER.csv)
│   ├── V8.3.1_Q_TRANSFER_PREDICTOR_{10,15,25}m.py  [기존] 예측+저장(test_table2)
│   ├── m14a_config.json                 [기존]
│   ├── api_key.txt                      [기존, 빈 파일 — 운영키 별도]
│   ├── QTransferPredictBatch.py         ★신규 (자바 이식: 통계+알람11종+STATE 적재)
│   ├── QTransferDashBoardItemBatch.py   ★신규 (자바 이식: REQUESTOR/WARNINGLOG/ERROR)
│   └── qtransfer_alarm_config.json      ★신규 (임계치16/메시지27/알람쿼리11/대시보드쿼리3/Oracle 3)
│
└── M16A_BR/                      ▼ HUBROOM 사이트 (혼잡도 예측 + WARN 판정)
    ├── M16BR_hubroom_data_make.py    [기존] Logpresso→CSV
    ├── V8_{Categorical,Numerical}_Real_time_{10,15,25}min.py  [기존] 예측+저장
    ├── m16br_config.json             [기존]
    ├── api_key.txt                   [기존, 빈 파일]
    ├── HubroomTransPredictBatch.py   ★신규 (자바 이식: WARN_YN 판정)
    └── hubroom_alarm_config.json     ★신규 (FLOW 3종 + 판정 쿼리)
```

### 마이그레이션 대원칙
자바 배치는 **①데이터수집 → ②예측 → ③알람/경고판정 → ④적재** 를 한 클래스에서 다 했다.
Python ML 시스템(afagg)은 이미 **①②** 를 분리 완료(`*_data_make.py` + `V8*예측.py`)했으므로,
이번 마이그레이션은 **③④ + 대시보드** 만 신규 배치 3개로 이식한다.

| 단계 | 자바 위치 | Python 담당 |
|---|---|---|
| ① 데이터수집 | 자바 배치 내부 쿼리 | **기존** `*_data_make.py` (CSV 생성) |
| ② 예측(LSTM 등) | 자바 배치 내부 모델 호출 | **기존** `V8*.py` (모델 .pkl → Logpresso 저장) |
| ③ 알람/경고 판정 | `_alarmValid` / `_validWarnYN` | **★신규** `*PredictBatch.py` |
| ④ 통합 적재 (통계+예측+알람) | `insertLogpressoData` | **★신규** `QTransferPredictBatch.py` / `HubroomTransPredictBatch.py` |
| ⑤ 대시보드 (STATE/REQUESTOR/WARNINGLOG/ERROR) | `_insertPredictionState` + `QTransferDashBoardItemBatch.java` | **★신규** `QTransferPredictBatch.py._insert_prediction_state()` + `QTransferDashBoardItemBatch.py` |

---

## 1. 참고한 자바 원본

| 자바 파일 | 줄수 | 이식한 메서드 | → Python |
|---|---|---|---|
| `ALT/main/java/com/skhynix/smartatlas/batch/HubroomTransPredictBatch.java` | 282 | `_validWarnYN` (L114-177) | `HubroomTransPredictBatch.py` `_valid_warn_yn()` |
| `ALT/main/java/com/skhynix/smartatlas/batch/QTransferPredictBatch.java` | 1775 | `_run`(L54-226), `_insertPredictionState`(L369-443), `_getPivotData`(L450-535), `_alarmValid`(L537-1157), `_buildTransportAlarm`(L1255-1369), `_buildAlarmBase`(L1374) | `QTransferPredictBatch.py` 동일 이름 메서드 |
| `ALT/main/java/com/skhynix/smartatlas/batch/QTransferDashBoardItemBatch.java` | 216 | `_run`(L51-90), `insertLogpressoData`(L92-105), `_buildRequestor`(L107-130), `_buildMcsErrorLogCount`(L132-144), `_buildTransQuePredictError`(L146-159), `_buildMap`(L161-199), `_getMapData`(L201-215) | `QTransferDashBoardItemBatch.py` 동일 이름 메서드 |

### 자바에서 가져오지 **않은** 것 (이미 Python 분리 완료)
- 데이터 수집 SQL → `*_data_make.py` 가 담당
- 예측 모델 로드/추론 → `V8*.py` 가 담당
- Hubroom `errorVhlVal` / `VHL_OFF_PLUS_DATA` → CSV 생성 단계(`M16BR_hubroom_data_make.py`) 로직이라 배치와 무관 (사용자 확인)

### 설정 출처 (자바 XML → config.json)
자바는 `variable.xml`(임계치)·`alarm_message.xml`(메시지)·`customQuery.xml`(쿼리)에 의존했으나,
Python 은 **사이트별 `*_alarm_config.json` 에 직접 박아** 자바 XML 의존을 제거했다.

| 자바 XML | 추출 → config.json 키 |
|---|---|
| `variable.xml` (QTRANSFER_LIMIT 등 16종) | `thresholds` |
| `alarm_message.xml` (27종) | `messages` |
| `customQuery.xml` / `customQuery2.xml` 알람용 (11종) | `queries` |
| `customQuery2.xml` 대시보드용 (3종: requestorCount/mcsErrorLogCnt/quePredictError) | `dashboard_queries` |
| `variable.xml` QTRANSFER_REQUESTOR_LIST | `dashboard_variables` |
| mybatis `mcs_m14a/m14b/m16a.xml` (3종) | `oracle.queries` (dbquery) |

---

## 2. M14A_FAB — QTransferPredictBatch.py

### 2.1 무슨 데이터를 보나 (입력)

모두 **Logpresso httpexport REST** 로 조회 (`query()` 헬퍼, `afagg` 패턴 복제):
```
GET http://10.40.42.27:8888/logpresso/httpexport/query.csv?_apikey=<KEY>&_q=<쿼리>
```

| 입력 | 쿼리/출처 | 용도 |
|---|---|---|
| **예측값** (LSTM/STATE/STATE_PER) | `predict_table`(**test_table2**, V8.3.1 예측기 출력) 최신 1건 | 알람 트리거 기준, 대시보드 상태 |
| **반송 통계** | `qTransferGroupData` (ts_current_job) | IDC 컬럼별 IQR/표준편차/평균 계산 |
| **VHL 가동률** | `vhlRunRate` (dbquery mcs_m14) | ALARM1 |
| **MES 큐** | `mesOHTQCntAlarmValid`, `completedTsJobCountPerMin` | ALARM1 |
| **반송 카운트** | `transferCountPerMin` (test_table5 재조회) | ALARM2/3/4 |
| **CNV 포트 다운율** | `bridgeDetailCnvPortDownRate` (bridge_layout_detail) | ALARM2 |
| **ALT JOB 집계** | `qtransferAltJobCnt`, `altJobMachineRate`, `*24h` | ALARM5 |
| **IDC 임계 지표** | `SELECT_AWS_IDC_HISTORY` | ALARM6~11 |
| **Oracle LFT 다운율** | `dbquery mcs_m14a/m14b SELECT...PORT_DOWN_RATE` | ALARM3/4 |
| **Oracle Storage 사용률** | `dbquery mcs_m16a SELECT...CURR_VAL` | ALARM2/3/4 참고지표 |

> **Oracle 도 oracledb 라이브러리 없이** Logpresso `dbquery <connection> SELECT...` 명령으로
> httpexport 경유 조회 (`_oracle_query()`). 운영 Logpresso 에 mcs_m14a/m14b/m16a connection 필요.

### 2.2 처리 흐름 (`_run`, 자바 L54-226)

```
1) _build_transport_alarm()      → ALARM6~11 (예측값 무관, SELECT_AWS_IDC_HISTORY 임계 비교)
2) _fetch_prediction()           → predict_table 최신 LSTM/STATE/STATE_PER
3) prediction 있으면:
   3-1) _insert_prediction_state()→ 대시보드 테이블(**test_table6**)에 STATE 행 1건 적재 (한글 정상/주의/심각)
   3-2) _get_pivot_data(pred)     → qTransferGroupData 통계 + _alarm_valid(ALARM1~5)
4) _reorder_and_finalize()       → 20컬럼 정렬 + TOTALCNT행에 LSTM/LSTM_JUDGE + 소수2자리
5) save_rows(insert_table)       → test_table5 에 전부 적재
```

#### 적재되는 행 3종류 (test_table5)
1. **실측+통계 row (N개)** — `_get_pivot_data`: IDC 컬럼마다 `AVERAGE / IQR_Q1·Q2·LOWER·UPPER·JUDGE / STANDARDDEVIATION·SD_LOWER·SD_UPPER·SD_JUDGE`
   - IQR: `_quantile` 선형보간(자바 `_getQuantile` L1215), 1.5×IQR 경계 벗어나면 `IQR_JUDGE="T"`
   - SD: 모집단 표준편차(`statistics.pstdev`), 평균±3σ 벗어나면 `SD_JUDGE="T"`
2. **예측 row (1개)** — TOTALCNT 행에 `LSTM`(예측값), `LSTM_JUDGE`(실측 대비 10%↑ 차이→Y, 자바 L173-191)
3. **알람 row (0~11개)** — `_alarm_valid`(ALARM1~5) + `_build_transport_alarm`(ALARM6~11)

### 2.3 알람 11종 판정 조건

전제: **예측값 > QTRANSFER_LIMIT(1700)** 일 때만 ALARM1~5 평가 (자바 L579).

| 알람 | 조건 | 임계치 | Oracle |
|---|---|---|---|
| ALARM1 | VHL FOUP 가동률 ≥ 95% **&** MES 10m큐가 24h평균比 10%↑ | VHL_RATE_LIMIT=95, INCREASE_RATE=10 | — |
| ALARM2 | M14↔M16 반송 10%↑ **&** CNV 포트다운율 < 10% | CNV_PORT_DOWN_RATE=10 | storage |
| ALARM3 | M14↔M10A 반송 10%↑ **&** LFT 다운율 < 25% | M14TOM10_PORT_DOWN_RATE=25 | LFT다운율+storage |
| ALARM4 | M14↔M14B 반송 10%↑ **&** LFT 다운율 < 10% | M14LFT_PORT_DOWN_RATE=10 | LFT다운율+storage |
| ALARM5 | ALT/ERROR JOB 증가율 > 50% **&** 동일장비 비중 ≥ 50% | ALT_JOB_LIMIT=50, MACHINE_RATE=50 | — |
| ALARM6 | Normal STB Storage 부하 ≥ 95% | BOUNDARY_6=95 | — |
| ALARM7 | Sorter Wait Count ≥ 100 | BOUNDARY_7=100 | — |
| ALARM8 | Cu Sorter Wait Count ≥ 100 | BOUNDARY_8=100 | — |
| ALARM9 | Q-COMPLETED(현재Q) ≥ 2700 | BOUNDARY_9=2700 | — |
| ALARM10 | Total Que ≥ 1700 **&** Manual Out ≥ 200 | BOUNDARY_10_1=1700, _2=200 | — |
| ALARM11 | 층간 대기 QUE ≥ 500 **&** STB부하 ≥ 97% | BOUNDARY_11_1=500, _2=97 | — |

> 메시지는 `get_message(code, *args)` 로 `alarm_message.xml` 의 `{0}{1}...` 자리표시자를 치환(자바 MessageFormat).

### 2.4 어디에 저장하나 (출력)

| 적재 | config 키 | 현재 테이블(테스트) | 구 운영 테이블 | 무엇 |
|---|---|---|---|---|
| 통계+예측+알람 | `insert_table` | **test_table5** | test_currentjob_predict | 행 3종 (위 §2.2 참조) |
| 대시보드 STATE | `dashboard_table` | **test_table6** | qtransfer_dashboard | STATE 한글(정상/주의/심각)+확률 1건 |
| (예측 스크립트가 씀 — 참조만) | `predict_table` | **test_table2** | ATLAS_TS_PREDICT | LSTM/STATE/STATE_PER (V8.3.1 출력) |

저장 방식 — Logpresso httpexport `import` (afagg V8 예측기와 동일 패턴):
```
json "{COL1='v1', COL2='v2', ...}" | import test_table5
```
(`_save_one()` — 작은따옴표/큰따옴표 이스케이프 처리)

---

## 2.5 M14A_FAB — QTransferDashBoardItemBatch.py (★ 신규 추가)

자바 `QTransferDashBoardItemBatch.java` 216줄 1:1 이식. `test_table6` (qtransfer_dashboard) 에
**대시보드 행 3종** 적재 (`QTransferPredictBatch` 의 STATE 행과 같은 테이블).

### 2.5.1 처리 흐름 (`_run`, 자바 L51-90)
```
1) _build_requestor()                → requestorCount 쿼리 → TYP=REQUESTOR 행 N개
2) _build_mcs_error_log_count()      → mcsErrorLogCnt 쿼리 → TYP=WARNINGLOG 행 0~3개
3) _build_trans_que_predict_error()  → quePredictError 쿼리 → TYP=ERROR 행 2개 (ERROR_RATE/ERROR_VALUE)
4) 각 row 에 EVENT_DT/DUE_GBN_CD 부여 → _save_one(test_table6) 반복
```

### 2.5.2 적재되는 행 종류 (test_table6)

| TYP | OPER_ACT_CTN | VAL | 출처 쿼리 | 자바 메서드 |
|---|---|---|---|---|
| **STATE** | 정상/주의/심각 | 위험확률(%) | (QTransferPredictBatch 가 씀) | `_insertPredictionState` |
| **REQUESTOR** | RTD/RTS, EIS, MHS, ETC, OFS, MCS | 건수 | `requestorCount` (1h MES TRANSPORT_JOB_SCHEDULE_REQ) | `_buildRequestor` |
| **WARNINGLOG** | WARN / ERROR / EXCEPTION | 카운트 | `mcsErrorLogCnt` (1h ts_data_view_m14a) | `_buildMcsErrorLogCount` |
| **ERROR** | ERROR_RATE | 예측 정확도 % | `quePredictError` (2h test_table5 TOTALCNT vs LSTM) | `_buildTransQuePredictError` |
| **ERROR** | ERROR_VALUE | 평균 절대오차 | 같은 쿼리, 다른 컬럼 | 같음 |

### 2.5.3 ERROR 행은 누적 후 생성됨
`quePredictError` 는 **test_table5 의 지난 2시간 데이터** (TOTALCNT 실측 + LSTM 예측 페어)를
조인해 정확도를 계산합니다. → 첫 실행 직후엔 ERROR 행 0건, **2시간 누적 후** 자동 생성.
(자바도 동일 조건이라 운영 동작 일치.)

### 2.5.4 `_buildMap` 의 두 오버로드 (자바 L161-199)
- **`_build_map_unpivot(type, data)`**: 1 row 의 모든 (k, v) 쌍을 각각 OPER_ACT_CTN row 로 펼침
  → `quePredictError` 에 사용 (ERROR_VALUE, ERROR_RATE 2행 생성)
- **`_build_map_keyval(content_field, value_field, type, data)`**: 지정 컬럼 2개로 1행 매핑
  → `requestorCount`, `mcsErrorLogCnt` 에 사용

### 2.5.5 인라인 테이블명 치환
`quePredictError` 쿼리 본문 안에 `test_currentjob_predict` 가 박혀 있음 → 자동으로
`insert_table`(test_table5) 로 치환해서 보냄 (Python 코드에서 `.replace()`).

---

## 3. M16A_BR — HubroomTransPredictBatch.py

자바 282줄 중 데이터·예측 제외 → `_validWarnYN` 만 남아 ~210줄.

### 3.1 무슨 데이터를 보나
- 예측 스크립트(`V8_Numerical_Real_time_*.py`)가 `insert_table`(test_table)에 이미 저장한
  **최신 예측 row 의 `JUDGEVAL`** 을 FLOW 별로 조회.

대상 FLOW 3종 (`config.flows`): `HUBROOM_PREDICT`, `HUBROOM_PREDICT_15MIN`, `HUBROOM_PREDICT_25MIN`

### 3.2 처리 흐름 (`_process_flow`, FLOW 마다)
```
1) 현재(최신) 예측 1건 조회   : test_table FLOW필터 sort -EVENT_DT limit 1 → JUDGEVAL
2) 직전 예측 조회             : 같은 쿼리 limit 2 → 두번째 row 의 JUDGEVAL
3) _valid_warn_yn(cur, prev) : WARN_YN 판정
4) test_table 에 WARN_YN/FLOW/EVENT_DT 붙여 갱신 적재
```

### 3.3 WARN_YN 판정 규칙 (자바 `_validWarnYN` L143-169)
```
prev is None        → "N"   (최초)
prev >= 2           → "N"
prev == 1 & cur==1  → "Y"   (연속 2회 위험)
그 외               → "N"
```

### 3.4 어디에 저장하나
| 적재 | config 키 | 현재(테스트) | 구 운영 |
|---|---|---|---|
| WARN_YN 반영 예측 | `insert_table` | **test_table** | test_hubroom_predict |

---

## 4. Prediction_ml.py 마스터 등록

`SITES` dict 의 각 사이트 `scripts` 목록 **맨 뒤**에 신규 배치를 `--once` 로 추가:

```python
"M16A_BR": { "scripts": [
    "M16BR_hubroom_data_make.py --once",      # ① 수집
    "V8_Categorical_Real_time_10/15/25min.py", # ② 예측
    "V8_Numerical_Real_time_10/15/25min.py",
    "HubroomTransPredictBatch.py --once",      # ③④ ★신규
]},
"M14A_FAB": { "scripts": [
    "M14A_FAB_data_make.py --once",            # ① 수집
    "V8.3.1_Q_TRANSFER_PREDICTOR_10/15/25m.py",# ② 예측
    "QTransferPredictBatch.py --once",         # ③④ ★신규 (통계+알람+STATE → test_table5/test_table6)
    "QTransferDashBoardItemBatch.py --once",   # ⑤ ★신규 (REQUESTOR/WARNINGLOG/ERROR → test_table6)
]},
```

- 매분 정각 → 2사이트 `ThreadPoolExecutor` 병렬 → 사이트 내부는 순차(subprocess)
- 신규 배치는 **자체 스케줄러 없음**. 오직 마스터가 `--once` 로 호출 (기존 예측 스크립트와 동일 컨벤션)
- 실행 순서 보장: 수집 → 예측 → **알람배치** (예측 결과가 테이블에 있어야 배치가 읽음)

---

## 5. 빌드 시 따른 규칙 (afagg 패턴 복제)

신규 배치는 **공유 common/ 폴더 없이** 각 사이트 폴더 안에서 자체 헬퍼 보유 (기존 스크립트도 그렇게 함):

| 헬퍼 | 출처(복제) | 역할 |
|---|---|---|
| `query(q)` | `M14A_FAB_data_make.py:query()` | httpexport query.csv → list[dict] |
| `save_to_logpresso`/`_save_one` | `V8.3.1_*_10m.py:save_to_logpresso()` | `json "..." \| import {table}` |
| `load_api_key()` | afagg 공통 | api_key.txt → 1줄 키 |
| `load_config()` | 신규 | 사이트 `*_alarm_config.json` |
| `find_value`/`get_message` | 자바 `_findValueByCode`/`getMessage` | config 임계치/메시지 |
| 통계(IQR/SD) | 자바 `_calculate*` | **numpy 대신 `statistics` 표준라이브러리** (환경 무관) |

### 외부 라이브러리
```
requests   # httpexport REST
# 신규 배치는 numpy/pandas/oracledb 불필요 (statistics 표준라이브러리 + dbquery 사용)
```
예측 스크립트(`V8*.py`)는 별도로 sklearn 등 모델 의존성 필요.

---

## 6. 실행 / 검증

```bash
# 단일 테스트 (권장)
cd M14A_FAB && python3 QTransferPredictBatch.py --once
cd M16A_BR  && python3 HubroomTransPredictBatch.py --once

# 통합 1회
cd 01_ml_operation && python3 Prediction_ml.py --once

# 상시 운영
python3 Prediction_ml.py
```

### 운영 전 체크리스트
- [ ] `api_key.txt` 에 실제 Logpresso API 키 (현재 빈 파일)
- [ ] 예측 모델 `.pkl` 이 `model/` 에 존재 (V8* 가 사용)
- [ ] Logpresso `10.40.42.27:8888` 도달 가능
- [ ] Logpresso 에 `dbquery` 용 mcs_m14a/m14b/m16a connection 등록 (ALARM2/3/4)
- [ ] 패키지: requests (+예측 스크립트 sklearn 등)

### 검증 (shadow)
현재 적재 대상이 **테스트 테이블**(test_table/5/6/7)이라 운영 테이블 무영향.
같은 시각 자바 배치 결과와 **알람 건수 / IDCCOL / ALARM_YN / WARN_YN / 메시지** 비교 →
1주 일치 확인 후 config 테이블명만 운영 테이블로 원복 → 자바 배치 중단.

---

## 7. 테이블 매핑 요약 (현재 = 테스트)

| 역할 | 구 운영 | 현재 테스트 | config 키 | 누가 씀 |
|---|---|---|---|---|
| Hubroom WARN 적재 | test_hubroom_predict | **test_table** | hubroom `insert_table` | HubroomTransPredictBatch |
| QTransfer 예측 출력 (V8.3.1 → 배치가 읽음) | ATLAS_TS_PREDICT | **test_table2** | qtransfer `predict_table` | V8.3.1 예측기 (기존, 쓰기) / QTransferPredictBatch (읽기) |
| QTransfer 통계+예측+알람 적재 | test_currentjob_predict | **test_table5** | qtransfer `insert_table` | QTransferPredictBatch |
| QTransfer 대시보드 적재 (STATE/REQUESTOR/WARNINGLOG/ERROR) | qtransfer_dashboard | **test_table6** | qtransfer `dashboard_table` | QTransferPredictBatch (STATE) + QTransferDashBoardItemBatch (나머지 3종) |
| ~~test_table7~~ | — | (사용 안 함) | — | — |

> 운영 전환 = config 의 위 4개 키 값만 원복. 코드 수정 불필요(테이블명 전부 config 변수에서 읽음).
> 단, `qtransfer_alarm_config.json` 의 `queries.transferCountPerMin` 본문 안 `test_table5` 와
> `dashboard_queries.quePredictError` 본문 안 `test_currentjob_predict` 도 같이 확인할 것.

---

## 8. 운영 로그 기반 트러블슈팅 (실제 사례)

운영서버 첫 실행 시 발생했던 HTTP 500 4건과 수정 내역 (커밋 `cafed0d`, `688e3a6`, `d66bd9b`):

| 증상 | 원인 | 수정 |
|---|---|---|
| QTransfer `❌ 쿼리 HTTP 500` → 0행 적재 | `predict_table=test_table6` 인데 V8.3.1 예측기는 실제로 **test_table2** 에 저장 | config `predict_table` → `test_table2` |
| Hubroom `❌ 쿼리 HTTP 500` × 3 FLOW | `sort -EVENT_DT` 인데 V8_Numerical 출력 row 에 **`EVENT_DT` 컬럼 없음** (`CURRENT_TIME`만 있음) | 두 쿼리 모두 `sort -CURRENT_TIME` 으로 변경 |
| 두 배치 모두 시간범위 쿼리 500 | `table from=dateadd(now(),"min",-N) to=now()` 구문이 일부 Logpresso 환경에서 500 | `table duration=Nm` 호환 구문으로 통일 |
| test_table6 에 대시보드 행 안 나옴 | `dashboard_table=test_table7` 로 잘못 매핑 | config `dashboard_table` → `test_table6` |
| test_table6 에 ERROR_RATE/ERROR_VALUE 행 없음 | `QTransferDashBoardItemBatch.java` 의 `_buildTransQuePredictError` 미이식 | **QTransferDashBoardItemBatch.py** 신규 추가 + 마스터 등록 |

### 운영 첫 실행 사이클 후 들어갈 행 (정상 동작 확인용)
| 테이블 | 들어갈 것 | 누가 적재 |
|---|---|---|
| **test_table** | Hubroom Categorical 3개 + Numerical 3개 + WARN_YN 3개 = **9행** | V8_Categorical/Numerical 예측기 6종 + HubroomTransPredictBatch |
| **test_table2** | **V8.3.1 예측기 3개 × 1행 = 3행** (LSTM/STATE/STATE_PER/TIME, 10m/15m/25m) | V8.3.1_Q_TRANSFER_PREDICTOR_{10,15,25}m.py |
| **test_table5** | 통계 N행 + 예측 1행 (LSTM<1700 사이클은 알람 0) | QTransferPredictBatch |
| **test_table6** | STATE 1 + REQUESTOR 3-6 + WARNINGLOG 0-3 (ERROR 2행은 **누적 2시간 후**) | QTransferPredictBatch (STATE) + QTransferDashBoardItemBatch (나머지) |

> ⚠ test_table2 는 **QTransferPredictBatch 가 읽기만** 함 (predict_table 로). 적재는 V8.3.1 예측기 3개가 함.
> 첫 사이클에 test_table2 에 3행 안 들어가면 → V8.3.1 모델 .pkl 또는 api_key 문제. QTransfer 알람 배치는 절대 동작 안 함.
