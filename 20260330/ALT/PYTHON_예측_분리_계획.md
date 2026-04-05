# Python 예측 분리 보고서

> 작성일: 2026-04-05
> 대상: ALT/src 예측 배치 잡 3건

---

## 1. 현황

현재 ALT/src의 3개 예측 배치 잡이 **하나의 Java 프로세스 안에서** 아래 전체 흐름을 수행 중이다.

```
[현재 - 단일 프로세스]

  DB 조회 → CSV 생성 → Python 호출(ProcessBuilder) → JSON 결과 수신 → 후처리 → DB 적재
  ──────────────────── Atlas(Java) ────────────────────────────────────────────────────
```

**문제점**: Python 예측이 Atlas Java 프로세스에 강결합되어 있어 독립 실행/배포/디버깅이 불가능하다.

---

## 2. 목표

Python 예측을 독립 프로세스로 분리하되, **Atlas의 DB 적재 역할은 유지**한다.
(Atlas가 다른 시스템에 데이터를 전달하는 역할을 담당하므로)

```
[변경 후 - 분리 구조]

  [Atlas 배치 잡]                    [Python 독립 프로세스]
  ┌──────────────────┐              ┌──────────────────────┐
  │ ① DB 조회        │              │                      │
  │ ② 입력 CSV 저장  │─── CSV ───▶ │ ③ 입력 CSV 읽기      │
  │                  │              │ ④ 예측 수행           │
  │ ⑥ 결과 CSV 읽기  │◀── CSV ───  │ ⑤ 예측 결과 CSV 저장  │
  │ ⑦ 후처리/알람    │              └──────────────────────┘
  │ ⑧ DB 적재       │
  └──────────────────┘
```

---

## 3. 대상 배치 잡 현황

| # | 배치 잡 파일 | 라인수 | 입력 CSV | Python 모델 | 적재 DB |
|---|-------------|--------|----------|-------------|---------|
| 1 | `batch/HubroomTransPredictBatch.java` | 267 | `data/HUBROOM_PIVOT_DATA.csv` | HUB_ROOM_PREDICTOR_*.py (6개) | test_hubroom_predict |
| 2 | `batch/QTransferPredictBatch.java` | 1753 | `data/QTRANSFER.csv` | Q_TRANSFER_PREDICTOR_*.py (6개) | test_currentjob_predict, ATLAS_TS_PREDICT |
| 3 | `batch/ServerResourceApmBatch.java` | 375 | `data/APMWEEKDATA.csv` | SERVER_RESOURCE_MODEL.py (1개) | server_resource_predict |

### 관련 유틸리티

| 파일 | 역할 | 처리방안 |
|------|------|---------|
| `util/PythonUtil.java` (134L) | ProcessBuilder로 Python 호출 + JSON 파싱 | 사용처 없으면 삭제 |
| `util/Util.java` (1216L) | CSV 생성(`composeCsvFileContent`), 파일 저장(`createAndOverwriteFile`) | 유지 + CSV 읽기 메서드 추가 |
| `data/PredictionPara.java` (257L) | 예측 파라미터 싱글톤 | 사용처 확인 후 판단 |

---

## 4. 변경 상세

### 4.1 공통: Util.java에 CSV 읽기 메서드 추가

**파일**: `ALT/src/util/Util.java`

Python이 저장한 결과 CSV를 읽어 `List<Map<String, Object>>`로 반환하는 메서드 신규 추가.
기존 `composeCsvFileContent()`의 역방향 처리.

```java
public static List<Map<String, Object>> readCsvFile(String filePath) throws Exception
```

---

### 4.2 HubroomTransPredictBatch 변경

| 구분 | 현재 | 변경 후 |
|------|------|---------|
| 입력 CSV 생성 | `_createInputData()` | **유지** |
| Python 호출 | `_predictor()` → `PythonUtil.executeWithParam()` | **삭제** → `_loadPredictionResult()` (결과 CSV 읽기) |
| 후처리 | `_validWarnYN()` | **유지** |
| DB 적재 | `insertInLogpressoDatabase()` | **유지** |

#### 삭제 메서드
- `_predictor()` — Python 6개 모델 순차 호출
- `_preparePredictor()` — 모델 파일명 목록 로드
- `errorVhlState` 필드 + `_init()` 관련 로직 (Python 파라미터용이므로 불필요)

#### 신규 메서드
- `_loadPredictionResult()` — `data/HUBROOM_PREDICT_RESULT.csv` 읽기

---

### 4.3 QTransferPredictBatch 변경 (가장 복잡)

| 구분 | 현재 | 변경 후 |
|------|------|---------|
| 모니터링 알람 | `_buildTransportAlarm()` | **유지** (예측 무관) |
| 입력 CSV 생성 | `_createInputData()` | **유지** |
| Python 호출 | `_predictor()` → `PythonUtil.executeWithParam()` | **삭제** → `_loadPredictionResult()` (결과 CSV 읽기) |
| 예측 로그 적재 | `_buildLogpressoDataWithPrediction()` | **수정** (CSV 데이터 기반으로 변환) |
| 상태값 적재 | `_insertPredictionState()` | **유지** |
| 피벗 데이터 | `_getPivotData()` + `_alarmValid()` | **유지** |
| 컬럼 정렬/소수점 | desiredOrder 기반 정렬 | **유지** |
| DB 적재 | `insertInLogpressoDatabase()` | **유지** |

#### 삭제 메서드
- `_predictor()` — Python 6개 모델 순차 호출
- `_preparePredictor()` — 모델 파일명 목록 로드
- `_selectDisplayedPrediction()` — 대표 모델 선택

#### 신규 메서드
- `_loadPredictionResult()` — `data/QTRANSFER_PREDICT_RESULT.csv`에서 FILE_NM, LSTM, STATE, STATE_PER 파싱

#### 수정 메서드
- `_buildLogpressoDataWithPrediction()` — 기존 `Map<String, List<Map>>` 파라미터를 CSV 데이터 기반으로 변경

---

### 4.4 ServerResourceApmBatch 변경

| 구분 | 현재 | 변경 후 |
|------|------|---------|
| Oracle 조회 + 측정치 적재 | `_run()` → `getApmDataByOracle()` + `setInsertTuples()` | **유지** |
| 측정치 알람 | `getAlarmText(dataList, "MEAS")` | **유지** |
| 7일치 데이터 조회 | `getResourceDataList()` | **유지** |
| 입력 CSV 저장 | `exectePredictData()` try 블록 | **유지** |
| Python 호출 | `PythonUtil.executeWithParam("SERVER_RESOURCE_MODEL.py")` | **삭제** → `_loadPredictionResult()` |
| 예측치 알람 | `getAlarmText(result, "PREDICT")` | **유지** |
| DB 적재 | `setInsertTuples("server_resource_predict")` | **유지** |

#### 삭제 부분
- `exectePredictData()` finally 블록 내 `PythonUtil.executeWithParam()` 호출

#### 신규 메서드
- `_loadPredictionResult()` — `data/APM_PREDICT_RESULT.csv` 읽기

---

### 4.5 PythonUtil.java 처리

3개 배치 잡에서 더 이상 사용하지 않게 됨. 다른 사용처 확인(grep) 후:
- 사용처 없음 → **파일 삭제**
- 사용처 있음 → **유지**

### 4.6 import 정리

각 배치 잡에서 `PythonUtil` import 및 미사용 import 제거.

---

## 5. Python 결과 CSV 포맷 정의

Python이 예측 수행 후 저장해야 할 CSV 포맷 명세.

### 5.1 HUBROOM_PREDICT_RESULT.csv

| 컬럼 | 타입 | 설명 |
|------|------|------|
| FILE_NM | String | 모델 파일명 (ex: HUB_ROOM_PREDICTOR_10m) |
| FLOW | String | 흐름 구분 (ex: HUBROOM_PREDICT) |
| JUDGEVAL | String | 판정값 (0 또는 1) |
| 기타 | - | Python 모델이 기존 JSON으로 리턴하던 필드들 |

```csv
FILE_NM,FLOW,JUDGEVAL
HUB_ROOM_PREDICTOR_10m,HUBROOM_PREDICT,0
HUB_ROOM_PREDICTOR_15m,HUBROOM_PREDICT_15MIN,1
HUB_ROOM_PREDICTOR_20m,HUBROOM_PREDICT_20MIN,0
HUB_ROOM_PREDICTOR_25m,HUBROOM_PREDICT_25MIN,0
HUB_ROOM_PREDICTOR_30m,HUBROOM_PREDICT_30MIN,1
HUB_ROOM_PREDICTOR_35m,HUBROOM_PREDICT_35MIN,0
```

### 5.2 QTRANSFER_PREDICT_RESULT.csv

| 컬럼 | 타입 | 설명 |
|------|------|------|
| FILE_NM | String | 모델 파일명 (ex: Q_TRANSFER_PREDICTOR_10m) |
| LSTM | Double | LSTM 예측값 |
| STATE | String | 상태 (NORMAL / CAUTION / CRITICAL) |
| STATE_PER | Double | 상태 확률 (%) |

```csv
FILE_NM,LSTM,STATE,STATE_PER
Q_TRANSFER_PREDICTOR_10m,1523.5,NORMAL,85.2
Q_TRANSFER_PREDICTOR_15m,1601.2,CAUTION,72.1
Q_TRANSFER_PREDICTOR_20m,1580.0,NORMAL,90.5
Q_TRANSFER_PREDICTOR_25m,1650.3,CAUTION,68.4
Q_TRANSFER_PREDICTOR_30m,1700.1,CRITICAL,55.3
Q_TRANSFER_PREDICTOR_35m,1720.8,CRITICAL,48.9
```

### 5.3 APM_PREDICT_RESULT.csv

| 컬럼 | 타입 | 설명 |
|------|------|------|
| FAB_ID | String | FAB ID |
| SVR_NM | String | 서버명 |
| PROC_NM | String | 프로세스명 |
| IDC_NM | String | 지표명 (CPU_OS, JVM_HEAP 등) |
| MEAS_TM | String | 측정 시간 |
| FCAST_TM | String | 예측 시간 |
| FCAST_VAL | Double | 예측값 |
| MODEL_NM | String | 모델명 (ex: Prophet) |

```csv
FAB_ID,SVR_NM,PROC_NM,IDC_NM,MEAS_TM,FCAST_TM,FCAST_VAL,MODEL_NM
M14,m14mcsapp01,MCS_TS,CPU_OS,202604051000,202604051030,25.3,Prophet
M14,m14mcsapp01,MCS_TS,JVM_HEAP,202604051000,202604051030,380.5,Prophet
```

---

## 6. 영향 범위 분석

### 영향 없는 기능

| 기능 | 위치 | 사유 |
|------|------|------|
| `_buildTransportAlarm()` | QTransferPredictBatch | 예측과 무관한 독립 모니터링 알람 |
| `_validWarnYN()` | HubroomTransPredictBatch | 입력 데이터 형식 동일 (`List<Map>`) |
| `_getPivotData()` | QTransferPredictBatch | prediction 값(double)만 사용 |
| `_insertPredictionState()` | QTransferPredictBatch | STATE, STATE_PER 값만 사용 |
| `_alarmValid()` | QTransferPredictBatch | prediction 값(double)만 사용 |
| `getAlarmText()` | ServerResourceApmBatch | 결과 데이터 형식 동일 (`List<Map>`) |
| 기타 배치 잡 26개 | batch/ | Python 예측과 무관 |

### 변경되는 부분

| 항목 | 현재 | 변경 후 |
|------|------|---------|
| Python 호출 방식 | `PythonUtil.executeWithParam()` (ProcessBuilder) | CSV 파일 읽기 (`Util.readCsvFile()`) |
| 데이터 수신 형식 | JSON (Python stdout) | CSV 파일 |
| Python 실행 주체 | Atlas Java 프로세스 | 독립 프로세스 (외부에서 실행) |

---

## 7. 결과 CSV 파일 경로

모든 CSV는 `${SMARTFX_REPOSITORY}/python/data/` 디렉토리 아래 위치.

| 용도 | 파일명 | 생성 주체 |
|------|--------|----------|
| Hubroom 입력 | `HUBROOM_PIVOT_DATA.csv` | Atlas |
| Hubroom 결과 | `HUBROOM_PREDICT_RESULT.csv` | Python |
| QTransfer 입력 | `QTRANSFER.csv` | Atlas |
| QTransfer 결과 | `QTRANSFER_PREDICT_RESULT.csv` | Python |
| APM 입력 | `APMWEEKDATA.csv` | Atlas |
| APM 결과 | `APM_PREDICT_RESULT.csv` | Python |

---

## 8. 검증 항목

| # | 검증 내용 | 방법 |
|---|----------|------|
| 1 | 입력 CSV가 기존과 동일하게 생성되는지 | 변경 전/후 CSV 비교 |
| 2 | 결과 CSV 읽기가 기존 JSON 응답과 동일한 구조인지 | 데이터 타입/필드 비교 |
| 3 | DB 적재 데이터가 기존과 동일한지 | Logpresso 쿼리로 비교 |
| 4 | PythonUtil import 제거 확인 | grep 검색 |
| 5 | 다른 기능에 영향 없는지 | 전체 배치 잡 동작 확인 |
| 6 | 결과 CSV 미존재 시 에러 핸들링 | Python 미실행 상태에서 Atlas 실행 |
