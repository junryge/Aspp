# ATLAS AMHS - `_predictor()` 메서드 분석

## 개요

`_predictor()`는 ATLAS MCS(Material Control System) 내에서 **Hub-Room 물량 예측**을 수행하는 핵심 메서드이다. Java에서 Python 예측 모델(LSTM 등)을 호출하고, 결과를 후처리하여 경고 판정까지 수행한다.

---

## 메서드 시그니처

```java
private List<Map<String, Object>> _predictor()
```

- **접근 제한**: `private` — 클래스 내부에서만 호출
- **반환값**: 예측 결과 리스트 (각 항목은 예측값 + 경고 판정 결과를 포함하는 Map)

---

## 실행 흐름

```
_predictor()
│
├─ 1. _preparePredictor()
│     → Python 모델 파일 목록 조회
│     → List<Map<String, String>> (FILE_NM, FLOW)
│
├─ 2. XmlUtil.getVariableEnv("VHL_OFF_PLUS_DATA")
│     → VHL OFF 상태 보정값 조회 (기본값: "30")
│
├─ 3. [LOOP] 각 모델에 대해:
│     │
│     ├─ PythonUtil.executeWithParam(fileName, false, errorVhlState, errorAddData)
│     │   → Java → Python 프로세스 fork
│     │   → LSTM/통계 예측 모델 실행
│     │   → List<Map<String, Object>> 반환 (또는 null)
│     │
│     └─ _validWarnYN(prediction, flowName, fileName)
│         → 예측값 기반 경고 등급 판정
│         → 정제된 결과를 result에 누적
│
├─ 4. Util.printHelper(logContent)
│     → 각 모델별 실행 결과 및 소요시간 로그 출력
│
└─ 5. 총 소요시간 체크
      → DELAYED_TIME 초과 시 ERROR 로그
      → 정상 시 INFO 로그
```

---

## 주요 파라미터

| 파라미터 | 설명 | 출처 |
|---------|------|------|
| `fileName` | Python 예측 스크립트 파일명 | `_preparePredictor()` → `FILE_NM` |
| `flowName` | FAB 간 물류 방향 (예: M14→M16) | `_preparePredictor()` → `FLOW` |
| `errorVhlState` | 현재 VHL(Vehicle) 에러 상태 | 클래스 필드 |
| `errorAddData` | VHL OFF 시 추가 보정 데이터 | XML 환경변수 `VHL_OFF_PLUS_DATA` (기본 "30") |

---

## 연관 파일 (추정)

### Java 클래스

| 파일 | 역할 |
|------|------|
| **TrafficBatch.java** | `_predictor()`가 속한 클래스. 배치 스케줄러에서 주기적으로 호출됨 |
| **PythonUtil.java** | `executeWithParam()` — Java에서 Python 프로세스를 fork하여 예측 모델 실행 |
| **XmlUtil.java** | `getVariableEnv()` — XML 기반 환경변수 조회 (`VHL_OFF_PLUS_DATA` 등) |
| **Util.java** | `printHelper()` — 로그 내용 포맷 출력 |

### Python 스크립트

| 구분 | 설명 |
|------|------|
| 예측 모델 스크립트 | `_preparePredictor()`에서 조회되는 `FILE_NM`에 해당하는 Python 파일들 |
| 모델 유형 | LSTM, 통계 기반 예측 모델 (Hub-Room 물량 예측) |

---

## 호출 체인

```
[스케줄러 / Timer]
  └→ TrafficBatch.run() 또는 execute()
      └→ _predictor()
          ├→ _preparePredictor()          // 모델 목록 조회
          ├→ PythonUtil.executeWithParam() // Python 모델 실행
          └→ _validWarnYN()               // 경고 판정
      └→ 결과를 Oracle DB에 INSERT
          └→ Logpresso에서 조회 가능
              → 테이블: test_currentjob_predict
              → 필드: LSTM_TIME, LSTM_JUDGE, MEAS_TM 등
```

---

## 로그 출력 형식

### 정상 실행

```
<HUB_ROOM PREDICTION>
model_m14_to_m16.py: 10 [elapsed time: 1234ms]
model_m16_to_m14.py: 8 [elapsed time: 987ms]
```

### 모델 없음

```
model_xxx.py: - [elapsed time: -ms] [NO MODEL]
```

### 지연 발생 (DELAYED_TIME 초과)

```
!!!DELAYED!!! calculating the prediction data of hub-room has finished [elapsed time: 5m (300000ms)]
```

---

## Logpresso 연동

`_predictor()`의 최종 결과는 Oracle DB를 거쳐 Logpresso `test_currentjob_predict` 테이블에서 조회 가능하다.

### Logpresso 조회 쿼리 패턴

```
table test_currentjob_predict
| search FAB=="M14"
| eval LSTM_TIME = dateadd(TIME, "10m")
| eval LSTM_JUDGE = if(LSTM_JUDGE == "N", "F", "T")
| rename TIME as MEAS_TM
| sort MEAS_TM
```

### 주요 필드

| 필드 | 설명 |
|------|------|
| `TIME` (→ `MEAS_TM`) | 측정 시각 |
| `LSTM_TIME` | 예측 대상 시각 (TIME + 10분) |
| `LSTM_JUDGE` | 예측 판정 (N→F 정상, T 경고) |
| IQR/SD 관련 | 통계적 이상치 판정 보조 지표 |

---

## 핵심 로직 상세

### 1. `_preparePredictor()`

Python 모델 파일 목록을 DB 또는 설정에서 조회하여 반환한다. 각 항목은 다음 구조:

```java
Map<String, String> info = {
    "FILE_NM": "model_m14_to_m16.py",  // Python 스크립트 파일명
    "FLOW": "M14_TO_M16"               // FAB 간 물류 방향
};
```

### 2. `PythonUtil.executeWithParam()`

```java
PythonUtil.executeWithParam(
    fileName,        // Python 스크립트 파일명
    false,           // (추정) 디버그 모드 여부
    errorVhlState,   // 현재 VHL 에러 상태
    errorAddData     // VHL OFF 보정값 ("30")
);
```

- Java에서 Python 프로세스를 fork하여 실행
- 실행 결과를 `List<Map<String, Object>>`로 파싱하여 반환
- 모델이 없거나 실행 실패 시 `null` 반환

### 3. `_validWarnYN()`

```java
List<Map<String, Object>> refinedPrediction = _validWarnYN(prediction, flowName, fileName);
```

- 예측 결과에 경고 등급을 부여 (WARN Y/N)
- FLOW별 임계값 기준으로 판정
- 정제된 결과만 반환

---

## 확인용 grep 명령어

로컬에서 연관 파일을 찾기 위한 명령어:

```bash
# _predictor 정의 및 호출 위치
grep -rn "_predictor" ALT/src/ --include="*.java"

# 연관 메서드 전체 검색
grep -rn "_preparePredictor\|_validWarnYN\|PythonUtil.executeWithParam" ALT/src/ --include="*.java"

# 관련 클래스 파일 검색
find ALT/src/ -name "*Traffic*" -o -name "*Predict*" -o -name "*HubRoom*" | grep ".java"

# PythonUtil 클래스 위치
find ALT/src/ -name "PythonUtil.java"
```

---

## 비고

- 레포지토리: `github.com/junryge/ASAS` (private)
- 경로: `ALT/src/` 하위
- 이 문서는 코드 스니펫 및 ATLAS 도메인 지식 기반 분석이며, 실제 파일 확인은 로컬 grep으로 검증 필요
