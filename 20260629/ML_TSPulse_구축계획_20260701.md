# ML 구축 계획 — TSPulse R1 이상탐지 (30분 사전 예측)

> 작성 2026-07-01 · 결정: **TSPulse R1(이상탐지 메인) + XGBoost(지도 백업)** / 핵심 ~30피처 / 우리 정상데이터 fine-tune / 메신저 56건 = 채점지
> 목표: **운영자가 메신저에 정체를 보고하기 30분 전, ML이 "이상 급증"을 미리 알림**

---

## 0. 한 줄 정의

```
입력: 시각 t 의 핵심 설비 시계열 (룰이 보는 ~30컬럼, 과거 윈도우)
출력: 이상점수(anomaly score) [0~1] — 평소 패턴과 얼마나 벗어났나
정답: 메신저 Episode 시작시각 (운영자 실제 인지) — 학습 아님, 채점용
핵심 가설: "정체 전조 = 급증 = 이상" → 이상점수가 정체보다 30분 선행
```

---

## 1. 왜 TSPulse R1 인가 (모델 비교 결론)

| 우리 약점 | TSPulse 해법 |
|---|---|
| 라벨 56건뿐 (지도학습 빠듯) | 이상탐지 = 정상만 학습 → **라벨 불필요** |
| 메신저 불완전 정답 (조용한 정체 존재) | 라벨을 채점지로만 → **정답 오염 우회** |
| 265 다변량 시계열 | **시계열 네이티브** (롤링/델타 수작업 불필요) |
| "급증=정체" 도메인 | 이상탐지의 **정의 그 자체** |
| 폐쇄망 반입 | **1.08M 초경량**, CPU 초당 수천 |

> 적합도 정성평가 TSPulse 8.6 / TabPFN 6.2 (`ml_검증/모델비교_TabPFN_vs_TSPulse.html`)
> **관문:** 이상점수가 정체에 30분 선행하는지 (Phase 3). 선행 안 하면 → XGBoost 폴백.

---

## 2. 데이터 (이미 보유 / 회사 PC)

| 데이터 | 경로 | 규모 |
|---|---|---|
| 원본 설비 (피처 원천) | raw 265 `M16A_HUBROOM_PR_*.csv` (일자별) | 56일 × ~1440분 ≈ **8만 행** |
| 메신저 Episode (채점지) | `운영로그_분석_v2/output/*_episode.csv` | 170건 → **정체 56건** (orphan·CAPA 제외) |
| 룰 컬럼 매핑 | `그래프_분석/raw_columns.py` (RA/RB/RD/SLA/Sorter) | 핵심 30피처 근거 |

> 주의: 2026-03-24 이전 22컬럼 NULL → **학습 구간 2026-04-01 ~ 05-29** 만 사용.
> raw 일자별 다운로드: `그래프_분석/aws_idc_일자다운로드.py 20260401 20260529`

---

## 3. 핵심 30 피처 (룰이 보는 것 = raw_columns.py 그대로)

정체를 가장 먼저 드러내는 룰 근거 컬럼만 선별. (265 중 30 → 가볍고 해석 쉬움)

| 그룹 | 컬럼 | 개수 |
|---|---|---|
| **R-A 반송시간** | `{M16HUB,M14,M14B,M16A,M16B,M16_PKT,M16_WT}` 의 AVGTOTALTIME1MIN/AVGLOADTIME1MIN | 7 |
| **R-B 큐누적** | M16HUB.M14TOM16.MESCURRENTQCNT · 각 영역 `*_TO_HUB_JOB` · M16.SENDQUEUETOTAL | 6 |
| **R-D 저장/OHT** | M16HUB FABSTORAGERATIO · STB.3F_STORAGE_UTIL · {M14,M14B,M16A,M16B}.OHTUTIL | 6 |
| **SLA 4분초과** | {M16HUB,M14,M14B,M16A,M16B}.TRANSPORT4MINOVERRATIO | 5 |
| **Sorter 대기** | {M14,M14B,M16A,M16B,M16HUB}.SORTERWAITCOUNTOVER | 5 |
| **보조** | M16HUB.AOTRANSDELAY · M16HUB.QUE.OHT.CURRENTOHTQCNT (리프터 대표) | 2 |
| | **합계** | **31** |

> 리프터 역증가(R-C)는 다중 LFT 합산이라 대표 큐(CURRENTOHTQCNT)로 근사.
> 컬럼 리스트는 `raw_columns.py` 의 RA_COL/RB_COL/RD_*/SLA_COL/SORTER_COL 재사용 → 룰과 100% 일치.

---

## 4. 라벨 = 채점지 (학습 아님)

```python
# 메신저 정체 episode 시작시각 t0 → (t0-30, t0] 구간이 "정체 30분 전"
# 학습엔 안 씀. 이상점수가 이 구간에서 올랐는지 채점만.
JAM = {'정체/병목','리프터','CNV','MLUD','브릿지'}  # orphan='Y' 제외
```
- 정체 56건 / 20일 → 30분 윈도우 양성 ~5% (참고: 지도학습이면 불균형 19:1)
- 신규: `ml/labels_채점지.py`

---

## 5. TSPulse 학습 — 우리 정상데이터 fine-tune

```
1) 정상 구간 정의 = 메신저 정체 없는 시간 (± 여유 60분 제외)
2) 핵심 31 시계열 정규화 (영역별 스케일 다름 → z-score/robust)
3) TSPulse R1 (granite-timeseries-tspulse) fine-tune
   - context length 512 (≈8.5시간), 재구성(reconstruction) 기반
   - 정상 재구성 학습 → 추론 시 재구성오차 = 이상점수
4) 매분 슬라이딩 윈도우 → anomaly_score(t)
```
- **누수 차단**: 윈도우는 t 까지 과거만. fine-tune 도 train 구간 정상만.
- 신규: `ml/tspulse_train.py`, `ml/tspulse_infer.py`

---

## 6. ★ Phase 3 — 선행성 검증 (핵심 게이트)

```
이상점수 anomaly_score(t) 를 임계 θ 로 이진화 →
  · 리드타임 = (정체 시작 t0) − (θ 첫 돌파 시각), t0 직전 30분 창 안에서
  · 탐지율   = 정체 56건 중 30분 전 이상 감지된 비율
  · 오경보   = 정체 아닌데 θ 돌파 (단, 조용한 정체 감안 — 다 FP 아님)
```
| 지표 | 목표 | 미달 시 |
|---|---|---|
| 평균 리드타임 | ≥ 25분 | XGBoost 폴백 검토 |
| 탐지율 | ≥ 60% | 피처/윈도우/임계 튜닝 |
| 선행성(정체 전 상승) | 명확 | **선행 안 하면 이상탐지 무의미 → 지도학습 전환** |

> 이게 통과해야 TSPulse 확정. 신규: `ml/검증_선행성.py`

---

## 7. XGBoost 지도 백업 (같은 데이터 비교)

- 동일 31 피처 + 롤링/델타 → XGBoost binary (scale_pos_weight~19)
- 라벨 = 메신저 30분 윈도우 (이땐 정답으로 사용)
- TSPulse vs XGBoost: PR-AUC·리드타임·탐지율 표로 대조 → 승자 채택
- 기존 재활용: `train_xgboost.py`, `ml_검증/ml_검증_TabPFN_vs_XGB.py`

---

## 8. 운영 연동 (독립 병행 — 융합 X)

```
매분 → tspulse_infer (핵심31 과거윈도우) → anomaly_score
     → ml_predict/YYYYMMDD_anomaly.csv (datetime, anomaly_score, ml_level)
     → 룰베이스(test_table3) 옆 별도 (test_table4) 적재
```
- ml_level 4단계: 안전<0.3 / 관심 / 경계 / 위험≥0.7 (룰 54/75/90 과 별개)
- 룰베이스와 **병행만**, 하이브리드 융합은 나중.
- 신규: `ml/ml_runner_tspulse.py` (+ `ML_LO.py` 적재)

---

## 9. 실행 Phase

| Phase | 작업 | 산출물 | 게이트 |
|---|---|---|---|
| 0 | 회사PC 환경 (granite-tsfm 설치·GPU·백업) | 환경 OK | 라이브러리 |
| 1 | raw 56일 병합 + 31피처 추출 + 라벨 채점지 | features.csv, labels.csv | 데이터 준비 |
| 2 | TSPulse fine-tune (정상구간) + 이상점수 생성 | anomaly.csv | 점수 나옴 |
| **3** | **선행성 검증 (리드타임·탐지율)** | 검증 리포트 | **★핵심 통과** |
| 4 | XGBoost 비교 | 비교표 | 승자 결정 |
| 5 | 운영 추론기 + Logpresso 적재 | ml_runner | 병행 가동 |
| 6 | 운영 매뉴얼 + 재학습 | 문서 | 완료 |

---

## 10. 신규/수정 파일

| 파일 | 상태 | 역할 |
|---|---|---|
| `ml/features_31.py` | 신규 | raw 265 → 핵심 31피처 (raw_columns 재사용) |
| `ml/labels_채점지.py` | 신규 | 메신저 → 30분 채점 윈도우 |
| `ml/tspulse_train.py` | 신규 | 정상구간 fine-tune |
| `ml/tspulse_infer.py` | 신규 | 매분 이상점수 |
| `ml/검증_선행성.py` | 신규 | 리드타임·탐지율·선행성 (Phase 3) |
| `ml/ml_runner_tspulse.py` | 신규 | 실시간 독립 추론 |
| `ml_검증/ml_검증_TabPFN_vs_XGB.py` | 재활용 | XGBoost 비교 |
| `ML_TSPulse_운영매뉴얼.md` | 신규 | 운영자용 |

---

## 11. 위험요소 & 대비

| 위험 | 대비 |
|---|---|
| 이상탐지가 정체에 선행 안 함 | **Phase 3 게이트** — 미달 시 XGBoost 지도 폴백 |
| 이상 ≠ 정체 (설비점검 등 FP) | 조용한 정체 감안, 룰 stage 와 교차 검증 |
| 정상구간 오염 (라벨 안된 정체) | ±60분 여유 제외, robust 정규화 |
| 폐쇄망 모델 반입 | 1M 경량 — 오프라인 가중치 파일 반입 |
| 라벨 56건 (지도 비교 시) | XGBoost scale_pos_weight, 데이터 누적 재학습 |
| 누수(미래정보) | 윈도우 t까지만, 시간분할, fine-tune train만 |

---

## 12. 환경 제약 (중요)

- **이 원격 환경은 학습 불가** — numpy조차 없고 pypi 403 차단, 학습데이터(5월 raw 56일) 없음.
- **모든 학습·추론은 회사 PC** (granite-tsfm + torch + GPU 권장).
- 이 계획서 + 스크립트를 회사 PC 로 반입 → Phase 0 부터 실행.

---

## 다음 단계
이 계획으로 **Phase 1(데이터+피처+라벨)** 스크립트부터 작성 → 회사 PC 반입 → 순서대로.
Phase 3(선행성) 통과 여부가 TSPulse vs XGBoost 최종 결정.
