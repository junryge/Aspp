# V8 모델 Feature 컬럼 정리

> 작성일: 2025-12-01  
> 대상: AMHS 예측 모델 V8

---

## 1. 입력 컬럼 (32개)

| 그룹 | 컬럼명 | 설명 | 개수 |
|------|--------|------|------|
| **storage** | M16A_3F_STORAGE_UTIL | 3F 스토리지 사용률 | 1 |
| **fs_storage** | CD_M163FSTORAGEUSE | 3F 스토리지 사용량 | 3 |
| | CD_M163FSTORAGETOTAL | 3F 스토리지 총량 | |
| | CD_M163FSTORAGEUTIL | 3F 스토리지 사용률(%) | |
| **hub** | HUBROOMTOTAL | 허브룸 잔여 공간 | 1 |
| **cmd** | M16A_3F_CMD | 3F 명령 수 | 2 |
| | M16A_6F_TO_HUB_CMD | 6F→허브 명령 수 | |
| **inflow** | M16A_6F_TO_HUB_JOB | 6F→허브 Job | 3 |
| | M16A_2F_TO_HUB_JOB2 | 2F→허브 Job | |
| | M14A_3F_TO_HUB_JOB2 | M14A 3F→허브 Job | |
| **outflow** | M16A_3F_TO_M16A_6F_JOB | 3F→6F Job | 3 |
| | M16A_3F_TO_M16A_2F_JOB | 3F→2F Job | |
| | M16A_3F_TO_M14A_3F_JOB | 3F→M14A Job | |
| **maxcapa** | M16A_6F_LFT_MAXCAPA | 6F 리프트 최대용량 | 2 |
| | M16A_2F_LFT_MAXCAPA | 2F 리프트 최대용량 | |
| **que** | M16HUB.QUE.ALL.CURRENTQCNT | 현재 큐 카운트 | 2 |
| | M16HUB.QUE.TIME.AVGTOTALTIME1MIN | 1분 평균 총 시간 | |
| **target_alt** | CURRENT_M16A_3F_JOB | 현재 3F Job (대체 타겟) | 1 |
| **m16hub_que** | M16HUB.QUE.ALL.CURRENTQCOMPLETED | 완료된 큐 | 5 |
| | M16HUB.QUE.ALL.FABTRANSJOBCNT | FAB 전송 Job 수 | |
| | M16HUB.QUE.TIME.AVGTOTALTIME | 평균 총 시간 | |
| | M16HUB.QUE.OHT.CURRENTOHTQCNT | OHT 큐 카운트 | |
| | M16HUB.QUE.OHT.OHTUTIL | OHT 사용률 | |
| **m16a_que** | M16A.QUE.ALL.CURRENTQCOMPLETED | 완료된 큐 | 7 |
| | M16A.QUE.ALL.CURRENTQCREATED | 생성된 큐 | |
| | M16A.QUE.OHT.CURRENTOHTQCNT | OHT 큐 카운트 | |
| | M16A.QUE.OHT.OHTUTIL | OHT 사용률 | |
| | M16A.QUE.LOAD.AVGLOADTIME1MIN | 1분 평균 로드 시간 | |
| | M16A.QUE.ALL.TRANSPORT4MINOVERCNT | 4분 초과 전송 수 | |
| | M16A.QUE.ABN.QUETIMEDELAY | 큐 시간 지연 | |

---

## 2. 버전별 컬럼 비교

| 버전 | 컬럼 수 | 추가된 그룹 |
|------|---------|-------------|
| **V6** | 17개 | 기본 (storage ~ maxcapa) |
| **V7** | 19개 | + que (2개) |
| **V8** | 32개 | + target_alt, m16hub_que, m16a_que (13개) |

---

## 3. Feature 생성 방식

| 그룹 | 생성 Feature | 개수/컬럼 |
|------|--------------|-----------|
| **maxcapa** | last_value | 1개 |
| **inflow, outflow** | mean, last_value, slope | 3개 |
| **그 외** | mean, std, max, min, last_value, last_5_mean, slope | 7개 |

---

## 4. 타겟 기본 Feature (11개)

| Feature명 | 설명 |
|-----------|------|
| target_mean | 30분 평균 |
| target_std | 30분 표준편차 |
| target_max | 30분 최대값 |
| target_min | 30분 최소값 |
| target_last_value | 현재값 |
| target_last_5_mean | 최근 5분 평균 |
| target_last_10_mean | 최근 10분 평균 |
| target_slope | 30분 기울기 |
| target_acceleration | 가속도 |
| target_is_rising | 상승 여부 (0/1) |
| target_rapid_rise | 급상승 여부 (0/1) |

---

## 5. 특수 Feature

### 5.1 FS Storage 특수 Feature

| Feature명 | 설명 |
|-----------|------|
| storage_use_rate | 사용량 변화율 |
| storage_remaining | 잔여 용량 |
| storage_util_last | 현재 사용률 |
| storage_util_high | 사용률 >= 7 (0/1) |
| storage_util_critical | 사용률 >= 10 (0/1) |

### 5.2 Hub 특수 Feature

| Feature명 | 설명 |
|-----------|------|
| hub_critical | HUBROOMTOTAL < 590 (0/1) |
| hub_high | HUBROOMTOTAL < 610 (0/1) |
| hub_warning | HUBROOMTOTAL < 620 (0/1) |
| hub_decrease_rate | 허브 감소율 |
| hub_storage_risk | 허브+스토리지 복합 위험 (0/1) |

### 5.3 복합 Feature

| Feature명 | 설명 |
|-----------|------|
| net_flow | inflow - outflow |
| total_cmd | 총 명령 수 |
| total_cmd_low | CMD < 220 (0/1) |
| total_cmd_very_low | CMD < 200 (0/1) |
| hub_cmd_bottleneck | 허브+CMD 병목 (0/1) |
| surge_risk_score | 급증 위험 점수 |
| surge_imminent | 급증 임박 (0/1) |

---

## 6. Boolean Feature (임계값 기반)

### 6.1 V7 QUE Boolean

| Feature명 | 조건 | 의미 |
|-----------|------|------|
| currentq_high | CURRENTQCNT >= 1200 | 큐 높음 |
| currentq_critical | CURRENTQCNT >= 1400 | 큐 위험 |
| avgtime1min_high | AVGTOTALTIME1MIN >= 4.0 | 시간 높음 |
| avgtime1min_critical | AVGTOTALTIME1MIN >= 4.5 | 시간 위험 |
| que_severe_bottleneck | 큐 >= 1200 AND 시간 >= 4.0 | 심각한 병목 |

### 6.2 V8 M16HUB QUE Boolean

| Feature명 | 조건 | 의미 |
|-----------|------|------|
| m16hub_ohtutil_high | OHTUTIL >= 85 | OHT 사용률 높음 |
| m16hub_ohtutil_critical | OHTUTIL >= 90 | OHT 사용률 위험 |
| m16hub_avgtime_high | AVGTOTALTIME >= 5.0 | 평균 시간 높음 |
| m16hub_avgtime_critical | AVGTOTALTIME >= 6.0 | 평균 시간 위험 |
| m16hub_severe_bottleneck | OHT >= 85 AND 시간 >= 5.0 | 심각한 병목 |

### 6.3 V8 M16A QUE Boolean

| Feature명 | 조건 | 의미 |
|-----------|------|------|
| m16a_ohtutil_high | OHTUTIL >= 85 | OHT 사용률 높음 |
| m16a_ohtutil_critical | OHTUTIL >= 90 | OHT 사용률 위험 |
| m16a_loadtime_high | AVGLOADTIME1MIN >= 2.5 | 로드 시간 높음 |
| m16a_loadtime_critical | AVGLOADTIME1MIN >= 2.8 | 로드 시간 위험 |
| m16a_transport4min_high | 4분초과 >= 40 | 전송 지연 높음 |
| m16a_transport4min_critical | 4분초과 >= 50 | 전송 지연 위험 |
| m16a_delay_warning | QUETIMEDELAY >= 1 | 지연 경고 |
| m16a_delay_critical | QUETIMEDELAY >= 3 | 지연 위험 |
| m16a_severe_bottleneck | OHT >= 85 AND 4분초과 >= 40 | 심각한 병목 |

---

## 7. 요약

| 항목 | 값 |
|------|-----|
| **입력 컬럼** | 32개 |
| **파생 Feature** | 222개 |
| **시퀀스 길이** | 30분 |
| **예측 시점** | 10분 후 |
| **타겟** | CURRENT_M16A_3F_JOB_2 |

---

## 8. 그룹별 컬럼 수

| 그룹 | V6 | V7 | V8 |
|------|----|----|-----|
| storage | 1 | 1 | 1 |
| fs_storage | 3 | 3 | 3 |
| hub | 1 | 1 | 1 |
| cmd | 2 | 2 | 2 |
| inflow | 3 | 3 | 3 |
| outflow | 3 | 3 | 3 |
| maxcapa | 2 | 2 | 2 |
| que | - | 2 | 2 |
| target_alt | - | - | 1 |
| m16hub_que | - | - | 5 |
| m16a_que | - | - | 7 |
| **합계** | **17** | **19** | **32** |

---

*End of Document*