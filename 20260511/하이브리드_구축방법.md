# M16A HUBROOM 하이브리드 정체예측 시스템 — 운영 매뉴얼

> 룰베이스(S1/S2/S3) × XGBoost ML(0~1) 양방향 매트릭스 융합
> 실시간 분당 1행, 30분 사전 알람

---

## 0. 폴더 구조 (운영 — 한 폴더에 다 둠)

```
OHS_HYBRID_OPS/                              ← 본 패키지 (운영 한 폴더)
├── aws_idc_realtime_collector.py            (수집기)
├── hubroom_predictor.py                     (룰베이스 예측기)
├── ml_predictor.py                          (XGBoost 모델 로더)
├── ml_predict_runner.py                     (ML 실시간 예측 runner)
├── feature_builder.py                       (피처 빌더)
├── train_xgboost.py                         (모델 학습 — 1회용)
├── evaluate.py                              (성능 평가)
├── predict_all.py                           (일괄 예측)
├── make_incidents.py                        (라벨링 도구)
├── incidents_sample.json                    (라벨 샘플)
├── hybrid_predictor.py                      (하이브리드 융합기)
├── run_ml.py                                (★ 4스레드 통합 실행)
├── model.json                               (학습된 XGBoost 모델 — 별도 배포)
├── predict/
│   └── M16A_HUBROOM_PR.csv                  (수집기 실시간 출력 — 90분 윈도우)
├── predict_tobe/                            (룰베이스 출력 — 자동생성)
│   ├── YYYYMMDD_발동이벤트.csv
│   └── YYYYMMDD_사건단위.csv
├── ml_predict/                              (ML 출력 — 자동생성)
│   └── YYYYMMDD_predictions.csv
└── hybrid_predict/                          (★ 최종 융합 결과 — 자동생성)
    └── YYYYMMDD_hybrid.csv
```

---

## 1. 시스템 아키텍처

### 1.1 데이터 흐름

```
[Oracle IDC]
    ↓
수집기 (aws_idc_realtime_collector.py)         매분 00초
    ↓
predict/M16A_HUBROOM_PR.csv                    (90분 슬라이딩 윈도우 덮어쓰기)
    │
    ├──→ 룰베이스 (hubroom_predictor.py)        매분 05초
    │      ↓
    │    predict_tobe/YYYYMMDD_발동이벤트.csv
    │    predict_tobe/YYYYMMDD_사건단위.csv (S3 종료 시)
    │
    └──→ ML 예측기 (ml_predict_runner.py)       매분 폴링
           ↓
         ml_predict/YYYYMMDD_predictions.csv
           ↓
         하이브리드 (hybrid_predictor.py)        매분 폴링
           ↓
         hybrid_predict/YYYYMMDD_hybrid.csv     ★ 최종
```

### 1.2 4개 스레드 동시 실행 — run_ml.py

```python
# run_ml.py 가 한 번에 4개 띄움
[스레드1] 수집기      (백그라운드 데몬)
[스레드2] ML 예측기   (백그라운드 데몬)
[스레드3] 하이브리드  (백그라운드 데몬)
[메인]    룰베이스    (포그라운드)
```

Ctrl+C 한 번에 4개 다 종료.

---

## 2. 룰베이스 예측기 (hubroom_predictor.py)

> **S3 (사건단위 확정) = R-A' AND R-C' AND (R-B OR R-D)**
> = **시간 AND 위치 AND (양 OR 공간)**

### 2.1 4가지 룰 — 잡는 패턴

| 룰 | 잡는 패턴 | 의미 한 줄 |
|---|---|---|
| **R-A'** | 반송이 비정상적으로 느려짐 (1MIN 9분+) | 시간축 — "OHT가 느려졌나?" |
| **R-B** | M14→M16 큐 누적 (느린 폭주) | 양축 — "큐가 쌓이나?" |
| **R-C'** | 특정 리프터에 큐 몰림 (구조적 데드락) | 위치축 — "한쪽으로 쏠리나?" |
| **R-D** ★ | FAB 저장공간 가득참 (5/7 SFA 정체 같은 케이스) | 공간축 — "저장공간이 막혔나?" |

### 2.2 구조적 흐름

```
                    ┌─── R-A' (반송 시간) ────── "OHT가 느려졌나?"
                    │
   정체 발생   ──┼─── R-B  (M14→M16 큐) ──── "큐가 쌓이나?"
                    │
                    ├─── R-C' (리프터 몰림) ─── "한쪽으로 쏠리나?"
                    │
                    └─── R-D  (FAB 저장률) ──── "저장공간이 막혔나?"
                              ▼
        S3 (사건단위 확정) = R-A' AND R-C' AND (R-B OR R-D)
                             └─시간─┘└─위치─┘└─양/공간─┘
```

**핵심 철학**: 시간(R-A') + 위치(R-C') + 양/공간(R-B 또는 R-D) **셋 다 동시에** 켜져야 진짜 데드락으로 인정. 한 가지만 켜지면 그냥 경고(S1/S2)로 남고 사건은 기록 안 함 → **위양성(false positive) 적음**.

### 2.3 정체 유형별 대응

| 정체 유형 | 잡는 룰 조합 | 예시 |
|---|---|---|
| 일반 데드락 (4/21 형식) | R-A' + R-B + R-C' | 검증 케이스 |
| 폭주형 (M14에서 큐 빠르게 쌓임) | R-A' + R-B-FAST(rb_fast) + R-C' | 빠른 누적 |
| **공간 막힘형 (SFA/저장 가득) ★** | R-A' + **R-D** + R-C' | **5/7 7시 SFA 정체** |
| 단순 트래픽 변동 | 1개만 발동 → S1/S2 | 사건단위 안 잡음 |

### 2.4 3단계 룰 (S1 → S2 → S3 에스컬레이션)

| 단계 | 정의 | 의미 |
|---|---|---|
| **S1** | R-A' 2회+ OR ra_sustained | 1차 경보 — 정체 시작 신호 |
| **S2** | R-B OR rb_fast | 2차 경보 — 정체 진행 |
| **S3** | R-A' AND R-C' AND (R-B OR R-D) | 사건 확정 — 데드락 |

### 2.5 룰 컴포넌트 임계값 (운영 튜닝 포인트)

| 룰 | 정의 | 임계값 | 원천 컬럼 |
|---|---|---|---|
| **R-A'** | 1MIN ≥9분이 10분창 1회+ | 9.0분 | `*.QUE.TIME.AVGTOTALTIME1MIN` |
| **ra_sustained** | 1MIN ≥6분이 5분창 3회+ | 6.0분 ×3회 | 위와 동일 |
| **R-B** | M14→M16 +100/30분 | +100 | `*.QUE.M14TOM16.MESCURRENTQCNT` |
| **rb_fast** | M14→M16 +30/10분 | +30 | 위와 동일 |
| **R-C'** | 리프터 합 감소 + 역증가 2개+ | 역증가 ≥2 | `*.LFT.{LID}.TOTAL_CURRENTQCNT` (10개) |
| **R-D** ★ | FAB 저장률 ≥25% | 25% | `*.STRATE.ALL.FABSTORAGERATIO` |

### 2.6 LIFTER 10개 모니터링 (R-C' 입력)

```
6ABL6011, 6ABL6012, 6ABL6021, 6ABL6022,
6ABL6031, 6ABL6032, 6ABL0111, 6ABL0112,
6ABL0121, 6ABL0122
```

### 2.7 출력: `predict_tobe/YYYYMMDD_발동이벤트.csv`

| 컬럼 | 내용 |
|---|---|
| `file` | 원천 CSV 파일명 |
| `datetime` | YYYY-MM-DD HH:MM |
| `stage` | 0/1/2/3 |
| `stage_name` | 정상/1단계/2단계/3단계 |
| `prev_stage` | 직전 단계 |
| `transition` | "1→2" 같은 단계 변화 표기 |
| `reason` | 한글 사유 (예: "M14→M16 +118 (30분간)") |
| `relation` | R-A'/R-B/R-C' 근거 풀텍스트 |

---

## 3. ML 예측기 (ml_predict_runner.py)

### 3.1 설계 철학 — 룰베이스 기준에 맞춰 학습

**ML은 룰베이스를 대체하는 게 아니라 룰의 한계를 보완하도록 설계됨.**

룰의 한계:
- 룰은 **임계값 초과 시점**부터 잡음 → 늦게 잡힘
- 룰은 **단발성 노이즈**에도 발동 → 오탐 발생
- 룰은 **추세/패턴**을 못 봄 (한 시점만 봄)

ML이 보완하는 부분:
- 30분 윈도우 **추세/통계** 보고 미리 잡음 → **사전 알람**
- 과거 사건 패턴 학습 → 단발성 노이즈 거름
- 룰 임계값 직전 상태도 포착 (예: 1MIN=8.9분, R-A' 미발동이지만 위험)

학습 라벨:
- **룰베이스가 만든 S3 사건 + 운영자가 사후 검증한 실제 사건**을 정답(positive)으로 사용
- 즉, ML 은 "룰베이스 S3 가 30분 뒤에 터지는 패턴" 을 학습 → **룰 기준과 정렬된 ML**

피처:
- 룰 컴포넌트 ctx (`ra_value`, `rb_diff`, `rev_count`, `rd_fabstorage` 등) 를 그대로 피처로 사용
- + 원천 15개 컬럼의 윈도우 통계 (mean/std/slope/min/max)
- → ML의 판단 기준이 룰의 판단 기준과 **같은 축** 위에 있음

### 3.2 모델

- **XGBoost** 분류기 (`xgboost.Booster`)
- 학습 데이터: 과거 데드락 사건 라벨 (incidents_sample.json 형식)
- 출력: 0~1 점수 (1에 가까울수록 30분 뒤 정체 가능성 높음)
- 모델 파일: `model.json`
- 학습 스크립트: `train_xgboost.py`
- 평가 스크립트: `evaluate.py`

### 3.3 입력 윈도우

- 30분 슬라이딩 (최소 31행)
- 피처: `feature_builder.build_features()` 가 윈도우 통계/추세 + 룰 ctx 종합

### 3.4 ML 등급 (4단계)

| 등급 (영/한) | 임계값 |
|---|---|
| LOW / 낮음 | < 0.30 |
| MID / 중간 | 0.30 ~ 0.70 |
| HIGH / 높음 | 0.70 ~ 0.85 |
| HIGH / 강함 | ≥ 0.85 |

### 3.5 출력: `ml_predict/YYYYMMDD_predictions.csv` (47개 컬럼)

```
[시각]   datetime, prediction_for
[원천]   avgtotal1min, m14_to_m16, fabstorage_ratio,
         lft_6ABL6011 ~ lft_6ABL0122 (10개),
         m14b_aotransdelay, m14b_oht_util, m14b_4abld122,
         m14b_avgtotal1min, m14b_7f_to_hub, m14b_7f_to_hub_alt,
         m14_htstop, m14_congested, m14_abnormal,
         m16pkt_aotransdelay, m16wt_aotransdelay
[룰판정] rule_s1, rule_s2, rule_s3
[룰근거] ra_value, ra_count, ra_sustained, ra_trig,
         rb_diff, rb_diff_10, rb_fast, rb_trig,
         rc_trend, rev_count, rev_lids, rc_trig,
         rd_fabstorage, rd_7f_alt, rd_trig
[ML]    ml_score, ml_level, ml_level_kr
```

---

## 4. 하이브리드 융합 (hybrid_predictor.py) ★

### 4.0 어떻게 만들어졌나 — 설계 의도

**문제 인식**: 룰베이스만 쓰면 (1) 늦게 잡거나, (2) 오탐. ML만 쓰면 (1) 근거가 불투명, (2) 운영자가 신뢰 못함.

**해결 방향**: **양방향 매트릭스 융합** — 룰과 ML을 동시에 평가해서 4가지 케이스를 다르게 다룸.

```
                ML 강신호 (≥0.85)              ML 약/없음
              ┌─────────────────────────┬─────────────────────┐
   룰 발동   │ ★ 양합치 (신뢰도 최상)  │ 룰만 발동           │
              │  → 위험-예측/위험-확정  │  → 단발성 의심 (관심) │
              ├─────────────────────────┼─────────────────────┤
  룰 OFF    │ ML 선행 (룰 곧 발동)     │  정상               │
              │  → 사전 경보 (경보)     │                      │
              └─────────────────────────┴─────────────────────┘
```

**4가지 케이스의 의미**:

1. **양합치 (룰+ML)** : 둘 다 발동 → 신뢰도 최상 → **위험-예측/위험-확정**
2. **ML 선행 (ML만)** : ML이 룰보다 먼저 잡음 → 사전 알람 → **경보**
   - 룰 임계값 직전 상태를 ML이 패턴으로 감지 ★ 하이브리드 핵심 가치
3. **룰만 (ML 미동의)** : 룰이 깜빡 발동했지만 ML은 패턴상 사건 아님 → 오탐 의심 → **관심**
   - 룰의 위양성을 ML이 줄여줌
4. **둘 다 OFF** : 정상

**3축 통합** — 한 줄에 다 들어 있어야 함:
- **무엇이**: `final_level` (위험-확정/위험-예측/경보/주의/관심/정상)
- **어떻게**: `agreement` (both/rule_only/ml_only/none) + `direction` (sync/ml→rule/rule→?)
- **왜**: `final_reason` + `rule_reason` + `rule_relation` + 원천 영향값 24개

---

### 4.0.1 양방향 판정 — 두 방향으로 동시에 검증한다 ★

**"양방향" 의 정확한 의미**: 한쪽이 다른 쪽을 일방적으로 보정하는 게 아니라, **룰 → ML** 과 **ML → 룰** **두 방향을 동시에** 평가해서 서로 교차검증한다는 뜻.

#### 방향 ① 룰 → ML (룰이 발동했을 때 ML 이 동의하나?)

```
룰이 발동 (S1/S2/S3)  ───→  ML 점수는?
                              ├─ 강(≥0.85)  → "ML도 동의, 진짜 위험" → 위험-예측/확정
                              ├─ 중(0.70~)  → "ML 어느 정도 동의"  → 경보
                              ├─ 약(0.30~)  → "ML 약하게 동의"     → 주의
                              └─ 낮음       → "ML 미동의, 오탐 의심" → 관심
                                                  └─ direction = rule→?(ML미동의)
```

→ **룰의 오탐을 ML이 걸러줌** (rule_only + 낮은 ml_score)

#### 방향 ② ML → 룰 (ML이 강신호일 때 룰이 동의하나?)

```
ML 강신호 (≥0.85)  ───→  룰 단계는?
                            ├─ S3       → "룰도 확정" → 위험-확정
                            ├─ S2/S1    → "룰도 진행 중" → ★ 위험-예측
                            └─ OFF      → "룰 미발동, ML 선행" → 경보
                                              └─ direction = ml→rule(예측)
```

→ **룰이 미처 못 잡은 사전징후를 ML이 잡아줌** (ml_only + 강 ml_score)

#### 양방향 동시 평가의 결과

| 룰 방향 | ML 방향 | 일치도(agreement) | 방향(direction) | 의미 |
|---|---|---|---|---|
| 룰 발동 | ML 강 | `both` | `sync(양합치)` | 양쪽 다 같은 결론 → 신뢰도 최고 |
| 룰 발동 | ML 낮음 | `rule_only` | `rule→?(ML미동의)` | 룰만 발동 → 오탐 의심 |
| 룰 OFF | ML 강 | `ml_only` | `ml→rule(예측)` | ML 선행 → 룰 곧 발동 예상 |
| 룰 OFF | ML 낮음 | `none` | `-` | 양쪽 다 OFF → 정상 |

#### 양방향이 단방향과 다른 점

```
[단방향 (X)]                          [양방향 (O)]
─────────────────────────────         ─────────────────────────────
룰 → ML (룰을 ML이 보정)              룰 ↔ ML (서로 교차검증)

룰이 발동하면 ML 점수 확인하고          룰과 ML 각각 독립 판정
끝 (한 방향 흐름)                      → 두 결과를 매트릭스에서 매칭
                                       → 일치도 + 방향 동시 산출

문제:                                  장점:
- ML 단독강(룰 OFF) 케이스 누락        - ML 선행 알람 가능 (방향 ②)
- 룰 오탐을 거를 수 없음               - 룰 오탐 자동 강등 (방향 ①)
                                       - 양합치는 신뢰도 격상
```

#### 코드 구현 (`hybrid_predictor.py:64-130`)

```python
def fuse(s1, s2, s3, ml_score):
    r = _rule_level(s1, s2, s3)   # 룰 측 등급 (0/1/2/3)
    m = _ml_level(ml_score)        # ML 측 등급 (0/1/2/3)

    # 양방향 일치도
    if r > 0 and m > 0:   agreement = 'both'        # 양합치
    elif r > 0:           agreement = 'rule_only'    # 룰만
    elif m > 0:           agreement = 'ml_only'      # ML만
    else:                 agreement = 'none'         # 둘 다 OFF

    # 양방향 방향
    if r == 0 and m >= 2:   direction = 'ml→rule(예측)'    # 방향 ②
    elif r >= 2 and m == 0: direction = 'rule→?(ML미동의)' # 방향 ①
    elif r > 0 and m > 0:   direction = 'sync(양합치)'      # 양방향 합의
    else:                   direction = '-'

    # 매트릭스 4×4 = 16 셀에서 최종 등급 결정
    # (룰 등급 r × ML 등급 m 의 모든 조합)
```

→ **r 과 m 을 독립적으로 계산하고 매트릭스에서 만나게** 하는 게 양방향의 본질. 한쪽이 다른 쪽을 덮어쓰지 않음.

### 4.1 최종 등급 매트릭스

| 룰\ML | 낮음(<0.30) | 약(0.30~0.70) | 중(0.70~0.85) | 강(≥0.85) |
|---|---|---|---|---|
| **S3** | 위험-확정 | 위험-확정 | 위험-확정 | 위험-확정 |
| **S2** | 관심 | 주의 | 경보 | **★ 위험-예측** |
| **S1** | 관심 | 주의 | 경보 | **★ 위험-예측** |
| **OFF** | 정상 | 관심 | 주의 | 경보 |

### 4.2 등급별 의미

| 등급 | 의미 | 대응 |
|---|---|---|
| **위험-확정** | S3 발동 — 이미 데드락 진행 | 즉시 현장 조치 |
| **위험-예측** ★ | S1/S2 + ML 강 — 30분 내 S3 진행 예상 | 사전 대응 (하이브리드 핵심 가치) |
| **경보** | S2+ML중 / 룰OFF+ML강 / S1+ML중 | 모니터링 강화 |
| **주의** | S2+ML약 / S1+ML약 / 룰OFF+ML중 | 관찰 |
| **관심** | 단독 약신호 | 기록만 |
| **정상** | 그 외 | — |

### 4.3 일치도(agreement)

- `both` : 룰과 ML 모두 발동
- `rule_only` : 룰만 발동, ML 동의 안 함
- `ml_only` : ML만 강신호, 룰 OFF
- `none` : 둘 다 안 발동

### 4.4 방향(direction)

- `ml→rule(예측)` : ML이 먼저 잡음 → 룰이 곧 발동할 것
- `rule→?(ML미동의)` : 룰만 깜빡 발동 (오탐 가능성)
- `sync(양합치)` : 둘 다 발동 — 신뢰도 높음
- `-` : 둘 다 OFF

### 4.5 출력: `hybrid_predict/YYYYMMDD_hybrid.csv` (57개 컬럼)

```
[시각]   datetime, prediction_for                                          (2개)
[원천]   avgtotal1min, m14_to_m16, fabstorage_ratio,
         lft_6ABL6011 ~ lft_6ABL0122 (10개),
         m14b_*, m14_*, m16*_aotransdelay (11개)                          (24개)
[룰]    rule_s1, rule_s2, rule_s3,
         ra_value, ra_count, ra_sustained, ra_trig,
         rb_diff, rb_diff_10, rb_fast, rb_trig,
         rc_trend, rev_count, rev_lids, rc_trig,
         rd_fabstorage, rd_7f_alt, rd_trig,
         stage, stage_name, prev_stage, transition,
         rule_reason, rule_relation                                        (24개)
[ML]    ml_score, ml_level, ml_level_kr                                    (3개)
[융합]   final_level, agreement, direction, final_reason                   (4개)
                                                                       총 57개
```

---

## 5. 사용 설명서

### 5.1 최초 1회 — 모델 학습

```bash
# 라벨 파일 준비 (incidents.json) — 과거 사건 시각 리스트
cp incidents_sample.json incidents.json
# (운영 데이터로 실제 incidents.json 작성)

# 학습
python train_xgboost.py
  → model.json 생성

# 성능 평가 (선택)
python evaluate.py
```

### 5.2 실시간 운영 — 4스레드 동시 실행 (★ 권장)

```bash
python run_ml.py
```

콘솔 출력 예시:
```
🤖 ML 예측기 시작
   입력: predict/M16A_HUBROOM_PR.csv
   출력: ml_predict
   모델: model.json
   폴링: 60초

🔀 하이브리드 예측기 시작
   ML 입력: ml_predict
   룰 입력: predict_tobe
   출력:    hybrid_predict
   임계값:  STRONG=0.85 MID=0.70 LOW=0.30
   폴링:    60초

  [14:23] 위험-예측 | both | sync(양합치) | ★ S2+ML강 → 30분내 S3 진행예상 ml=0.87
  💾 hybrid 1건 → 20260513_hybrid.csv
```

Ctrl+C 한 번에 4개 스레드 모두 종료.

### 5.3 개별 실행 (디버깅용)

```bash
# 수집기만
python aws_idc_realtime_collector.py

# 룰베이스만
python hubroom_predictor.py --watch

# ML 예측기만
python ml_predict_runner.py

# 하이브리드만
python hybrid_predictor.py
```

### 5.4 옵션 (경로/주기 변경)

```bash
python hybrid_predictor.py \
  --input_dir ./ml_predict \
  --rule_dir  ./predict_tobe \
  --out_dir   ./hybrid_predict \
  --interval  60
```

### 5.5 임계값 튜닝

`hybrid_predictor.py` 상단:
```python
ML_STRONG = 0.85   # 강신호
ML_MID    = 0.70   # 중간
ML_LOW    = 0.30   # 약신호
```

### 5.6 일별 파일 관리

- 매일 자정에 새 파일 자동 생성 (`20260513_hybrid.csv` → `20260514_hybrid.csv`)
- 오래된 파일 (30일 경과 등) 백업/삭제는 별도 cron 으로 권장

### 5.7 문제 해결

| 증상 | 원인 | 조치 |
|---|---|---|
| `❌ 모델 없음` | model.json 없음 | `python train_xgboost.py` 실행 |
| `⚠ 입력 CSV 없음` | 수집기 정지 | 수집기 로그 / DB 연결 확인 |
| `⚠ CSV 행수 < 31` | 윈도우 미충족 | 30분 기다림 |
| `prefix 감지 실패` | CSV 컬럼명 변경 | `detect_prefix()` 로직 확인 |
| 헤더 불일치 append | 컬럼 늘어난 뒤 옛 CSV 에 추가 | 옛 파일 백업/삭제 후 재시작 |
| 한글 깨짐 | 엑셀 인코딩 | 출력은 utf-8-sig (BOM) — 더블클릭 OK |

---

## 6. 운영자가 한 줄에서 답할 수 있는 질문 ★

`hybrid_predict/YYYYMMDD_hybrid.csv` 한 줄(57개 컬럼)로 모든 질문에 답할 수 있어야 함.

### 6.1 위험도 판단

| 질문 | 확인 컬럼 | 비고 |
|---|---|---|
| 지금 위험한가? | `final_level` | 위험-예측/위험-확정이면 즉시 대응 |
| 룰과 ML 의견 같나? | `agreement` | both = 신뢰도 높음 |
| 누가 먼저 잡았나? | `direction` | `ml→rule(예측)` 이면 ML이 선행 |
| 30분 뒤 예측 시각? | `prediction_for` | datetime + 30분 |

### 6.2 판정 근거

| 질문 | 확인 컬럼 |
|---|---|
| 왜 위험으로 판정? | `final_reason` |
| 룰베이스 판정 사유? | `rule_reason`, `rule_relation` |
| 현재 룰 단계? | `stage`, `stage_name` (예: "2단계") |
| 단계 변화? | `transition` (예: "1→2") |
| ML 신뢰도? | `ml_score`, `ml_level_kr` |

### 6.3 룰 컴포넌트별 발동 여부

| 질문 | 확인 컬럼 |
|---|---|
| R-A' 발동? | `ra_trig` (1=Y) — `ra_value` (실측 분) — `ra_count` (10분창 횟수) |
| R-A' 지속발동? | `ra_sustained` (5분창 ≥6분 3회+) |
| R-B 발동? | `rb_trig` (1=Y) — `rb_diff` (30분 증가량) — `rb_diff_10` (10분 증가량) |
| R-B fast? | `rb_fast` (10분 +30 이상) |
| R-C' 발동? | `rc_trig` (1=Y) — `rev_count` (역증가 리프터 개수) — `rev_lids` (어느 리프터?) |
| R-D 발동? | `rd_trig` (1=Y) — `rd_fabstorage` (FAB 저장률 %) |

### 6.4 원천 영향값 (그 분의 실측 센서값)

| 질문 | 확인 컬럼 |
|---|---|
| T1MIN 평균? | `avgtotal1min` (분 단위, 9분 이상이면 위험) |
| M14→M16 대기? | `m14_to_m16` (대수, 급증하면 위험) |
| FAB 저장률? | `fabstorage_ratio` (%, 25% 이상이면 R-D 트리거) |
| 어느 리프터 적체? | `lft_6ABL6011` ~ `lft_6ABL0122` (대기 대수, 10개 컬럼) |
| 7F→HUB 작업? | `m14b_7f_to_hub`, `m14b_7f_to_hub_alt` |
| OHT 가동률? | `m14b_oht_util` (%) |
| OHT 이상 상태? | `m14_htstop` (정지), `m14_congested` (정체), `m14_abnormal` (이상) |
| M14B 이상지연? | `m14b_aotransdelay` |
| M16 이상지연? | `m16pkt_aotransdelay`, `m16wt_aotransdelay` |

### 6.5 운영자 의사결정 흐름

```
1. final_level 확인
   ├─ 위험-확정 → 즉시 현장 출동 (S3 진행 중)
   ├─ 위험-예측 → 사전 대응 시작 (30분 윈도우 활용)
   │    └─ rule_reason, rule_relation 으로 어느 룰 때문인지 파악
   │    └─ rev_lids 로 문제 리프터 식별
   │    └─ avgtotal1min, m14_to_m16 추세 모니터링
   ├─ 경보       → 모니터링 강화, 추세 관찰
   ├─ 주의/관심  → 로그만 기록
   └─ 정상       → 통과

2. agreement / direction 확인
   ├─ both + sync(양합치)  → 신뢰도 ↑
   ├─ ml→rule(예측)        → ML 선행 알람, 룰 발동 곧 예상
   ├─ rule→?(ML미동의)     → 단발성 가능, 오탐 의심
   └─ none                 → 안전

3. 원인 분석 (위험-예측 / 위험-확정 시)
   └─ rule_relation 안의 [R-A'/R-B/R-C'] 풀텍스트가 핵심
```

---

## 7. 파일별 책임 요약

| 파일 | 역할 | 변경 빈도 |
|---|---|---|
| `aws_idc_realtime_collector.py` | Oracle → CSV 수집 | 거의 없음 |
| `hubroom_predictor.py` | 룰베이스 S1/S2/S3 + 발동이벤트.csv | 임계값/룰 추가 시 |
| `ml_predictor.py` | XGBoost model.json 로더 | 거의 없음 |
| `ml_predict_runner.py` | ML 실시간 예측 + predictions.csv | 거의 없음 |
| `feature_builder.py` | ML 피처 빌더 + evaluate_rules | 피처 추가 시 |
| `train_xgboost.py` | 모델 학습 (1회용) | 재학습 시 |
| `evaluate.py` | 모델 성능 평가 | 검증 시 |
| `hybrid_predictor.py` | 룰×ML 융합 + hybrid.csv | 매트릭스 튜닝 시 |
| `run_ml.py` | 4스레드 통합 실행 | 거의 없음 |
| `model.json` | 학습된 XGBoost 모델 | 재학습 시 |

---

## 8. 운영 체크리스트

### 8.1 일일

- [ ] `run_ml.py` 프로세스 살아있는지 확인
- [ ] `predict/M16A_HUBROOM_PR.csv` 최신 시각 확인
- [ ] `hybrid_predict/{오늘}_hybrid.csv` 새 행 추가되는지 확인
- [ ] 콘솔에서 `위험-예측` / `위험-확정` 로그 발생 시 대응

### 8.2 주간

- [ ] 어제~오늘 hybrid CSV 의 `위험-예측` 건수 vs 실제 사건 발생 일치도 검토
- [ ] 오탐(`rule→?(ML미동의)`) / 누락 사례 정리

### 8.3 월간

- [ ] 누적된 사건 라벨로 `train_xgboost.py` 재학습 검토
- [ ] 임계값(`ML_STRONG/MID/LOW`) 튜닝 검토
- [ ] 오래된 CSV 백업 / 아카이브

---

문의: 본 매뉴얼은 hybrid_predictor 시스템 v1.0 기준
