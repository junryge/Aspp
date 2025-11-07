# V7 모델 Input 변수 정리 (~11/21)

## 1. V7 모델에 사용가능한 Input 변수 후보 List Up

---

## 📋 전체 컬럼 요약
- **총 컬럼 수**: 125개
- **데이터 기간**: 2025-11-06 15:46 ~ 16:34 (약 50분, 1분 단위)
- **데이터 행 수**: 60개

---

## 🎯 1. 필수 컬럼 (시간/식별자)

| 컬럼명 | 설명 | 비고 |
|--------|------|------|
| STAT_DT | 통계 일시 (타임스탬프) | Feature 생성용 |

**용도**: 시계열 특성 생성, 시간대별 패턴 분석

---

## 🎯 2. Target 변수 (예측 대상)

| 컬럼명 | 설명 | 범위 | 비고 |
|--------|------|------|------|
| HUBROOMTOTAL | HubRoom 총 물량 | 543 ~ 646 | **예측 목표값** |

**예측 목표**: HUBROOMTOTAL의 Surge 예측 (300 이상 또는 1700 이상)

---

## 📊 3. 기존 사용 컬럼 (구분: 카테고리별)

### 3.1 📦 Job/작업량 관련 (15개)
작업 흐름 및 물동량을 나타내는 변수들

| No | 컬럼명 | 설명 | 범위 |
|----|--------|------|------|
| 1 | CURRENT_M16A_3F_JOB | M16A 3F 현재 작업량 | 192~240 |
| 2 | CURRENT_M16A_3F_JOB_2 | M16A 3F 현재 작업량2 | 264~369 |
| 3 | M14A_3F_TO_HUB_JOB2 | M14A 3F → HUB 작업량2 | 80~112 |
| 4 | M14A_3F_TO_HUB_JOB_ALT | M14A 3F → HUB 대체 작업량 | 0~1 |
| 5 | M14B_7F_TO_HUB_JOB2 | M14B 7F → HUB 작업량2 | 34~59 |
| 6 | M14B_7F_TO_HUB_JOB_ALT | M14B 7F → HUB 대체 작업량 | 0~5 |
| 7 | M16A_2F_TO_6F_JOB | M16A 2F → 6F 작업량 | 8~19 |
| 8 | M16A_2F_TO_HUB_JOB2 | M16A 2F → HUB 작업량2 | 1~9 |
| 9 | M16A_2F_TO_HUB_JOB_ALT | M16A 2F → HUB 대체 작업량 | 0~2 |
| 10 | M16A_3F_TO_3F_MLUD_JOB | M16A 3F → 3F MLUD 작업량 | 1~4 |
| 11 | M16A_3F_TO_M14A_3F_JOB | M16A 3F → M14A 3F 작업량 | 3~16 |
| 12 | M16A_3F_TO_M14B_7F_JOB | M16A 3F → M14B 7F 작업량 | 40~75 |
| 13 | M16A_3F_TO_M16A_2F_JOB | M16A 3F → M16A 2F 작업량 | 39~55 |
| 14 | M16A_3F_TO_M16A_6F_JOB | M16A 3F → M16A 6F 작업량 | 82~115 |
| 15 | M16A_6F_TO_2F_JOB | M16A 6F → 2F 작업량 | 10~30 |

### 3.2 🔧 Command 관련 (10개)
설비 간 이동 명령/지시 관련 변수들

| No | 컬럼명 | 설명 | 범위 |
|----|--------|------|------|
| 1 | M14A_3F_TO_HUB_CMD | M14A 3F → HUB 명령 | 17~51 |
| 2 | M14B_7F_TO_HUB_CMD | M14B 7F → HUB 명령 | 29~53 |
| 3 | M16A_2F_TO_HUB_CMD | M16A 2F → HUB 명령 | 2~23 |
| 4 | M16A_3F_CMD | M16A 3F 명령 | 215~258 |
| 5 | M16A_3F_TO_M14A_CNV_AI_CMD | M16A 3F → M14A CNV AI 명령 | 40~73 |
| 6 | M16A_3F_TO_M14B_LFT_AI_CMD | M16A 3F → M14B LFT AI 명령 | 45~64 |
| 7 | M16A_3F_TO_M16A_3F_STB_CMD | M16A 3F → M16A 3F STB 명령 | 0~5 |
| 8 | M16A_3F_TO_M16A_LFT_AI_CMD | M16A 3F → M16A LFT AI 명령 | 79~114 |
| 9 | M16A_3F_TO_M16A_MLUD_AI_CMD | M16A 3F → M16A MLUD AI 명령 | 80~114 |
| 10 | M16A_6F_TO_HUB_CMD | M16A 6F → HUB 명령 | 95~125 |

### 3.3 ⚙️ Capacity/용량 관련 (9개)
설비별 최대 처리 용량 관련 변수들

| No | 컬럼명 | 설명 | 범위 |
|----|--------|------|------|
| 1 | M14A_3F_CNV_MAXCAPA | M14A 3F CNV 최대용량 | 176 (고정) |
| 2 | M14B_7F_LFT_MAXCAPA | M14B 7F LFT 최대용량 | 55 (고정) |
| 3 | M16A_2F_LFT_MAXCAPA | M16A 2F LFT 최대용량 | 60 (고정) |
| 4 | M16A_3F_CNV_MAXCAPA | M16A 3F CNV 최대용량 | 126 (고정) |
| 5 | M16A_3F_LFT_MAXCAPA | M16A 3F LFT 최대용량 | 151 (고정) |
| 6 | M16A_3F_M14BLFT_MAXCAPA | M16A 3F M14B LFT 최대용량 | 66 (고정) |
| 7 | M16A_6F_LFT_MAXCAPA | M16A 6F LFT 최대용량 | 149 (고정) |
| 8 | M16A_6F_TO_HUB_JOB | M16A 6F → HUB 작업량 | 95~128 |
| 9 | M16A_6F_TO_HUB_JOB_ALT | M16A 6F → HUB 대체 작업량 | 0~24 |

### 3.4 🚛 OFS/Offset 관련 (2개)
층간 이동 오프셋 현재값

| No | 컬럼명 | 설명 | 범위 |
|----|--------|------|------|
| 1 | M14_TO_M16_OFS_CUR | M14 → M16 오프셋 현재값 | 2~6 |
| 2 | M16_TO_M14_OFS_CUR | M16 → M14 오프셋 현재값 | 2~5 |

### 3.5 💾 Storage/저장소 관련 (4개)
저장소 사용량 및 가동률

| No | 컬럼명 | 설명 | 범위 |
|----|--------|------|------|
| 1 | M16A_3F_STORAGE_UTIL | M16A 3F 저장소 가동률(%) | 1~4 |
| 2 | CD_M163FSTORAGEUSE | M16 3F 저장소 사용량 | 1~7 |
| 3 | CD_M163FSTORAGETOTAL | M16 3F 저장소 총량 | 169 (고정) |
| 4 | CD_M163FSTORAGEUTIL | M16 3F 저장소 가동률(%) | 1~4 |

### 3.6 📋 Queue/대기열 관련 (6개)
작업 대기열 및 처리 현황

| No | 컬럼명 | 설명 | 범위 |
|----|--------|------|------|
| 1 | M14.QUE.ALL.CURRENTQCREATED | M14 전체 큐 생성 건수 | 2936~3073 |
| 2 | M14.QUE.ALL.CURRENTQCOMPLETED | M14 전체 큐 완료 건수 | 2823~2946 |
| 3 | M14.QUE.OHT.OHTUTIL | M14 OHT 가동률(%) | 74.2~80.9 |
| 4 | M14.QUE.ALL.TRANSPORT4MINOVERCNT | M14 4분 초과 운반 건수 | 93~118 |
| 5 | M14B.QUE.SENDFAB.VERTICALQUEUECOUNT | M14B 수직 큐 카운트 | 187~253 |
| 6 | **M16HUB_QUE_AVG_TOTAL_TIME** | **M16 HUB 큐 평균 총 시간** | **4.07~5.44** |

### 3.7 🚀 Velocity/속도 관련 (2개)
평균 이동 속도

| No | 컬럼명 | 설명 | 범위 |
|----|--------|------|------|
| 1 | M14A_A_AVG_VELOCITY | M14A 평균 속도 | 119.9~120.8 |
| 2 | M16A_BR_AVG_VELOCITY | M16A 브랜치 평균 속도 | 168.1~178.6 |

### 3.8 🏢 기타 (2개)

| No | 컬럼명 | 설명 | 범위 |
|----|--------|------|------|
| 1 | M16B_10F_TO_HUB_JOB | M16B 10F → HUB 작업량 | 0~43 |

**기존 컬럼 소계**: **48개**

---

## 🆕 4. 신규 추가 컬럼

### 4.1 🌿 Branch별 Count (37개) - **NEW!**
M16A 각 Branch별 작업 건수를 개별 추적

| 컬럼 범위 | 설명 | 비고 |
|-----------|------|------|
| M16A_BR_001_CNT ~ M16A_BR_037_CNT | Branch 001~037 작업 건수 | 각 경로별 물량 분포 파악 가능 |

**범위**: 0~25 (Branch별로 상이)

**활용 가치**:
- ✅ 특정 Branch의 병목 현상 탐지
- ✅ 물량 분산도 측정
- ✅ Branch 간 불균형 지표 생성 가능
- ✅ Hotspot Branch 식별

**통계 예시** (첫 데이터 기준):
- M16A_BR_001_CNT: 8
- M16A_BR_002_CNT: 23
- M16A_BR_003_CNT: 18
- ...
- M16A_BR_037_CNT: 0

### 4.2 🚀 Branch별 개별 Velocity (37개) - **NEW!**
M16A 각 Branch별 이동 속도 (HID별)

| 컬럼 범위 | 설명 | 단위 | 비고 |
|-----------|------|------|------|
| M16A_BR_HID001_VELOCITY ~ M16A_BR_HID037_VELOCITY | Branch HID 001~037 속도 | m/min (추정) | 각 경로별 속도 개별 추적 |

**범위**: 77.4~297.2 (Branch별로 상이)

**활용 가치**:
- ✅ 속도 저하 Branch 조기 감지
- ✅ 평균 속도와의 편차 계산
- ✅ 속도 변동성 지표 생성
- ✅ 속도 기반 이상치 탐지

**통계 예시** (첫 데이터 기준):
- M16A_BR_HID001_VELOCITY: 113.7
- M16A_BR_HID002_VELOCITY: 97.5
- M16A_BR_HID003_VELOCITY: 78.9
- ...
- M16A_BR_HID037_VELOCITY: 82.7

### 4.3 ⏱️ Queue 평균 시간 (1개) - **NEW!**

| 컬럼명 | 설명 | 범위 | 비고 |
|--------|------|------|------|
| **M16HUB_QUE_AVG_TOTAL_TIME** | M16 HUB 큐 평균 총 처리 시간 | 4.07~5.44 | **중요 신규 지표** |

**활용 가치**:
- ✅ 작업 처리 효율성 직접 측정
- ✅ 대기 시간 증가 → Surge 징후
- ✅ 시계열 변화율 Feature로 활용 가능
- ✅ Bottleneck 조기 경보

**신규 컬럼 소계**: **75개** (37 + 37 + 1)

---

## 📈 5. 컬럼 카테고리별 중요도 평가

### 5.1 필수 유지 (High Priority)
**Target 예측에 직접적 영향**

| 카테고리 | 컬럼 수 | 이유 |
|----------|---------|------|
| Target 변수 | 1 | HUBROOMTOTAL - 예측 목표 |
| Job 관련 | 15 | 물동량 직접 지표 |
| Queue 관련 | 6 | 대기/처리 상태, **신규 M16HUB_QUE_AVG_TOTAL_TIME 포함** |
| Command 관련 | 10 | 작업 지시량 |

**소계: 32개**

### 5.2 조건부 포함 (Medium Priority)
**파생 Feature 생성 가능**

| 카테고리 | 컬럼 수 | 이유 |
|----------|---------|------|
| Branch CNT (신규) | 37 | 물량 분산도, 병목 감지 |
| Branch Velocity (신규) | 37 | 속도 이상 감지 |
| Storage 관련 | 4 | 저장 공간 압박 |
| Velocity 평균 | 2 | 전체 속도 추세 |

**소계: 80개**

### 5.3 검토 필요 (Low Priority)
**고정값 또는 변동 미미**

| 카테고리 | 컬럼 수 | 이유 |
|----------|---------|------|
| Capacity (MAXCAPA) | 7 | 대부분 고정값 → Feature Engineering 후 제거 검토 |
| OFS/Offset | 2 | 변동 범위 작음 (2~6) |

**소계: 9개**

---

## 🔍 6. V7 모델 Input 변수 후보 우선순위

### 📌 Tier 1: 필수 포함 (32개)
```
1. HUBROOMTOTAL (Target)
2. Job 관련 15개
3. Command 관련 10개
4. Queue 관련 6개 (신규 M16HUB_QUE_AVG_TOTAL_TIME 포함)
```

### 📌 Tier 2: 신규 변수 실험 (75개)
```
1. M16A_BR_XXX_CNT (37개) - 물량 분산 패턴
2. M16A_BR_HIDXXX_VELOCITY (37개) - 속도 이상 패턴
3. M16HUB_QUE_AVG_TOTAL_TIME (1개) - 처리 효율
```

### 📌 Tier 3: 보조 변수 (12개)
```
1. Storage 관련 4개
2. Velocity 평균 2개
3. OFS 2개
4. 기타 4개
```

---

## 💡 7. 다음 단계 작업 (2단계 준비)

### 7.1 신규 변수 탐색 방향

#### 🔹 파생 Feature 생성 아이디어

**A. Branch 관련**
```python
# 물량 분산도
- Branch_CNT_STD: Branch별 Count 표준편차
- Branch_CNT_MAX_RATIO: 최대/평균 비율
- Branch_CNT_IMBALANCE: Gini 계수

# 속도 이상
- Branch_VELOCITY_CV: 속도 변동계수
- Branch_SLOW_COUNT: 평균 대비 20% 이하 속도 Branch 개수
- Branch_VELOCITY_MIN_RATIO: 최저/평균 속도 비율
```

**B. 시계열 Feature**
```python
# 변화율
- HUBROOMTOTAL_DIFF_1MIN: 1분 전 대비 변화량
- HUBROOMTOTAL_DIFF_5MIN: 5분 전 대비 변화량
- M16HUB_QUE_AVG_TIME_SLOPE: 큐 시간 변화 기울기

# Rolling 통계
- JOB_ROLLING_MEAN_5MIN: Job 합계 5분 이동평균
- CMD_ROLLING_STD_10MIN: Command 합계 10분 이동 표준편차
```

**C. 비율/압박 지표**
```python
# 용량 대비 사용률
- M16A_3F_UTILIZATION: (CURRENT_JOB / CNV_MAXCAPA) * 100
- STORAGE_PRESSURE: (STORAGEUSE / STORAGETOTAL) * Queue Time

# 작업 효율
- QUEUE_EFFICIENCY: (COMPLETED / CREATED) * 100
- TRANSPORT_DELAY_RATIO: (TRANSPORT4MINOVERCNT / CREATED) * 100
```

### 7.2 변수 중요도 분석 예정
- XGBoost Feature Importance
- SHAP Value
- Permutation Importance
- 상관관계 분석

### 7.3 다중공선성 체크
- VIF (Variance Inflation Factor) 계산
- 높은 상관관계 변수 쌍 제거 또는 통합

---

## 📝 8. 요약

| 구분 | 개수 | 비고 |
|------|------|------|
| **총 컬럼** | **125개** | STAT_DT 포함 |
| 기존 컬럼 | 48개 | V6 이전 모델 사용 |
| 신규 컬럼 | 75개 | Branch CNT/Velocity + Queue Time |
| 고정값 컬럼 | 7개 | MAXCAPA 류 |
| Target | 1개 | HUBROOMTOTAL |

### 신규 컬럼의 가치
✅ **M16A_BR_XXX_CNT (37개)**: 물량 편중 감지, Hotspot 분석  
✅ **M16A_BR_HIDXXX_VELOCITY (37개)**: 속도 저하 조기 경보  
✅ **M16HUB_QUE_AVG_TOTAL_TIME (1개)**: 처리 효율 직접 측정  

---

## 🎯 Action Items

- [ ] 1단계: Input 변수 후보 List Up ✅ **완료**
- [ ] 2단계: 파생 Feature 생성 (~11/21)
- [ ] 3단계: Feature Selection (중요도 분석)
- [ ] 4단계: V7 모델 학습 및 검증

---

**작성일**: 2025-11-07  
**목적**: V7 모델 개발을 위한 Input 변수 체계 정리
