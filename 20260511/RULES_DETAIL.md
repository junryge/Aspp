# 룰베이스 4가지 룰 — 완전 상세 명세서

> R-A' / R-B / R-C' / R-D 의 풀이, 원천 컬럼, 임계값, 트리거 조건, 잡는 케이스, 코드 위치까지

---

## 0. 공통 명명 규칙

### "R" 이란?

| 표기 | 의미 |
|---|---|
| **R** | **R**ule (룰 컴포넌트 식별자) |
| **R-A', R-B, R-C', R-D** 의 R | 모두 동일한 의미 — 그냥 "룰" 의 약자 |

### "`'` (프라임)" 이란?

| 표기 | 의미 |
|---|---|
| **`'` (프라임)** | "개선판 / 정밀화" 표시 |
| **R-A'** | 원래 R-A 였는데 임계값/조건 개선해서 R-A' 로 변경 |
| **R-C'** | 원래 R-C 였는데 "역증가 리프터" 개념 도입하면서 R-C' 로 변경 |
| **R-B, R-D** | 처음부터 정밀 정의되어 프라임 없음 |

---

## 1. R-A' — 반송이 비정상적으로 느려짐 (시간축)

### 1.1 명명 분해

| 기호 | 풀이 |
|---|---|
| **R** | Rule (룰 컴포넌트) |
| **A'** | **A**vgTotalTime (평균 반송시간) 의 개선판 — 절대값 임계 + 지속성 검증 추가 |

### 1.2 감시 축

| 감시 축 | 의미 | 운영자 질문 |
|---|---|---|
| **시간축** | OHT 가 짐 1개 옮기는 데 평소보다 오래 걸리는가 | "OHT가 느려졌나?" |

### 1.3 보는 원천 컬럼

| # | 원천 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M16HUB.QUE.TIME.AVGTOTALTIME1MIN` | 분 (float) | 최근 1분 평균 TOTAL 반송시간 (smartSTAR) |

**총 컬럼 수: 1개**

### 1.4 트리거 조건 (2종)

#### 메인 트리거: `ra_trig`

```python
# hubroom_predictor.py
TH_RA_VALUE = 9.0   # 분
ra_count    = (10분 윈도우 안에서 1MIN ≥ 9.0분 인 분의 개수)
ra_trig     = (ra_count ≥ 1)
```

→ **1MIN ≥ 9.0분 인 분이 10분 윈도우 안에 1회 이상** 발생하면 트리거

#### 보조 트리거: `ra_sustained`

```python
TH_RA_SUSTAINED_VALUE = 6.0   # 분
TH_RA_SUSTAINED_COUNT = 3     # 회
ra_sustained = (5분 윈도우 안에서 1MIN ≥ 6.0분 인 분이 3회 이상)
```

→ **1MIN ≥ 6.0분 인 분이 5분 윈도우 안에 3회 이상 지속** 되면 트리거 (덜 심한 정체가 오래 지속되는 케이스)

### 1.5 S1 / S3 에서의 역할

| 단계 | 사용 방식 |
|---|---|
| **S1** | `s1 = (ra_count ≥ 2) OR ra_sustained` — 1MIN ≥9분이 2회+ 또는 지속신호 |
| **S3** | `s3 = ra_trig AND rc_trig AND (rb_trig OR rd_trig)` — AND 조건 필수 |

### 1.6 잡는 시나리오

| 케이스 | 어떻게 잡힘 |
|---|---|
| **데드락 직전 OHT 정체** | 1MIN 9분 넘게 튀면서 ra_trig |
| **만성 슬로우다운** | 1MIN 6~9분이 5분간 3회 → ra_sustained |
| **노이즈성 일시 지연** | 1회만 튀고 끝 → 트리거 안 됨 (위양성 방어) |

### 1.7 코드 위치

| 항목 | 위치 |
|---|---|
| 컬럼 읽기 | `hubroom_predictor.py:164` |
| 임계값 정의 | `hubroom_predictor.py:70-72` (TH_RA_*) |
| 계산 로직 | `hubroom_predictor.py:204-211` (ra_count, ra_sustained 계산) |
| ML 피처 출력 | `ra_value, ra_count, ra_sustained, ra_trig` (predictions.csv) |

---

## 2. R-B — M14→M16 큐 누적 (양축)

### 2.1 명명 분해

| 기호 | 풀이 |
|---|---|
| **R** | Rule (룰 컴포넌트) |
| **B** | **B**ridge Queue (M14↔M16 브릿지 큐) |

### 2.2 감시 축

| 감시 축 | 의미 | 운영자 질문 |
|---|---|---|
| **양축** | M14 → M16 으로 가야 할 반송 큐가 빠지지 않고 누적되는가 | "큐가 쌓이나?" |

### 2.3 보는 원천 컬럼

| # | 원천 컬럼 | 단위 | 의미 |
|---|---|---|---|
| 1 | `M16HUB.QUE.M14TOM16.MESCURRENTQCNT` | 개수 (int) | M14 → M16 브릿지 이동 반송 Q 수 (MES 기준) |

**총 컬럼 수: 1개**

### 2.4 트리거 조건 (2종)

#### 메인 트리거: `rb_trig` (느린 폭주)

```python
TH_RB_DIFF_30 = 100   # 개수
rb_diff = (현재 큐 수) - (30분 전 큐 수)
rb_trig = (rb_diff ≥ 100)
```

→ **30분 동안 +100 이상 증가** 하면 트리거 (느리지만 확실한 누적)

#### 보조 트리거: `rb_fast` (빠른 폭주)

```python
TH_RB_DIFF_10 = 30   # 개수
rb_diff_10 = (현재 큐 수) - (10분 전 큐 수)
rb_fast = (rb_diff_10 ≥ 30)
```

→ **10분 동안 +30 이상 증가** 하면 트리거 (빠른 누적 — M14 폭주형)

### 2.5 S2 / S3 에서의 역할

| 단계 | 사용 방식 |
|---|---|
| **S2** | `s2 = rb_trig OR rb_fast` — R-B 또는 fast 둘 중 하나 |
| **S3** | `s3 = ra_trig AND rc_trig AND (rb_trig OR rd_trig)` — R-B 또는 R-D 만족 필수 |

### 2.6 잡는 시나리오

| 케이스 | 어떻게 잡힘 |
|---|---|
| **느린 누적** | 30분 동안 +120 → rb_trig 발동 |
| **갑작스런 폭주** | 10분만에 +50 → rb_fast 발동 (빠르게 사전 감지) |
| **단순 트래픽 증가** | +50/30분, +20/10분 → 둘 다 미발동 (위양성 방어) |

### 2.7 코드 위치

| 항목 | 위치 |
|---|---|
| 컬럼 읽기 | `hubroom_predictor.py:165` |
| 임계값 정의 | `hubroom_predictor.py:73-74` (TH_RB_DIFF_*) |
| 계산 로직 | `hubroom_predictor.py:213-220` (rb_diff, rb_diff_10 계산) |
| ML 피처 출력 | `rb_diff, rb_diff_10, rb_fast, rb_trig` (predictions.csv) |

---

## 3. R-C' — 특정 리프터에 큐 몰림 (위치축)

### 3.1 명명 분해

| 기호 | 풀이 |
|---|---|
| **R** | Rule (룰 컴포넌트) |
| **C'** | **C**ongestion at lifters (리프터 적체) 의 개선판 — "역증가" 개념 도입 |

### 3.2 감시 축

| 감시 축 | 의미 | 운영자 질문 |
|---|---|---|
| **위치축** | 전체 큐는 빠지는데 특정 리프터로 큐가 몰리는가 (구조적 데드락) | "한쪽으로 쏠리나?" |

### 3.3 보는 원천 컬럼 (10개)

| # | 원천 컬럼 | 단위 | 리프터 ID |
|---|---|---|---|
| 1 | `M16HUB.LFT.6ABL6011.TOTAL_CURRENTQCNT` | 개수 | 6ABL6011 |
| 2 | `M16HUB.LFT.6ABL6012.TOTAL_CURRENTQCNT` | 개수 | 6ABL6012 |
| 3 | `M16HUB.LFT.6ABL6021.TOTAL_CURRENTQCNT` | 개수 | 6ABL6021 |
| 4 | `M16HUB.LFT.6ABL6022.TOTAL_CURRENTQCNT` | 개수 | 6ABL6022 |
| 5 | `M16HUB.LFT.6ABL6031.TOTAL_CURRENTQCNT` | 개수 | 6ABL6031 |
| 6 | `M16HUB.LFT.6ABL6032.TOTAL_CURRENTQCNT` | 개수 | 6ABL6032 |
| 7 | `M16HUB.LFT.6ABL0111.TOTAL_CURRENTQCNT` | 개수 | 6ABL0111 |
| 8 | `M16HUB.LFT.6ABL0112.TOTAL_CURRENTQCNT` | 개수 | 6ABL0112 |
| 9 | `M16HUB.LFT.6ABL0121.TOTAL_CURRENTQCNT` | 개수 | 6ABL0121 |
| 10 | `M16HUB.LFT.6ABL0122.TOTAL_CURRENTQCNT` | 개수 | 6ABL0122 |

**총 컬럼 수: 10개 (각 리프터 TOTAL Q 수)**

### 3.4 트리거 조건

```python
TH_RC_REVERSE = 2   # 역증가 리프터 개수

# 1) 리프터 합계 추세 계산
lft_sum_now  = sum(현재 시점 리프터 10개 큐)
lft_sum_prev = sum(과거 시점 리프터 10개 큐)
rc_trend     = lft_sum_now - lft_sum_prev   # 음수면 전체 큐 감소

# 2) 역증가 리프터 카운트
rev_count    = 개수 of 리프터 (lft_now[i] > lft_prev[i])  # 거꾸로 증가한 리프터
rev_lids     = [어느 리프터들이 역증가했는지 ID 목록]

# 3) 트리거
rc_trig = (rc_trend < 0) AND (rev_count ≥ 2)
```

→ **전체 합계는 감소(rc_trend<0) 하는데, 그 와중에 2개 이상 리프터가 거꾸로 증가** 하면 트리거

### 3.5 "역증가" 개념 (R-C 와 R-C' 의 차이)

| 구분 | R-C (옛날) | R-C' (현재) ★ |
|---|---|---|
| 조건 | 단순 임계값 (특정 리프터 큐 ≥ N) | **전체 추세 감소 AND 일부만 역증가** |
| 문제점 | 평상시에도 자주 발동 | 구조적 쏠림만 잡힘 |
| 의미 | 그냥 큐 많음 | "다 빠지는데 얘만 안 빠지네?" |

### 3.6 S3 에서의 역할

| 단계 | 사용 방식 |
|---|---|
| **S3** | `s3 = ra_trig AND rc_trig AND (rb_trig OR rd_trig)` — R-C' AND 조건 필수 |

R-C' 가 안 켜지면 절대 S3 안 됨 → **위치 쏠림 없는 단순 폭주는 사건 아님**

### 3.7 잡는 시나리오

| 케이스 | 어떻게 잡힘 |
|---|---|
| **구조적 데드락** | 다 빠지는데 6ABL6011, 6ABL0121, 6ABL0122 만 거꾸로 증가 → rev_count=3, rc_trig 발동 |
| **단순 큐 증가** | 모든 리프터가 같이 증가 → rc_trend ≥ 0 → 미발동 |
| **전체 감소** | 모든 리프터 감소 → rev_count=0 → 미발동 |

### 3.8 코드 위치

| 항목 | 위치 |
|---|---|
| 컬럼 읽기 | `hubroom_predictor.py:166-169` (lft_list dict) |
| LIFTER ID 목록 | `hubroom_predictor.py:64-68` (LIFTER_IDS) |
| 임계값 정의 | `hubroom_predictor.py:75` (TH_RC_REVERSE) |
| 계산 로직 | `hubroom_predictor.py:223-234` (rc_trend, rev_count 계산) |
| ML 피처 출력 | `rc_trend, rev_count, rev_lids, rc_trig` (predictions.csv) |

---

## 4. R-D ★ — FAB 저장공간 가득참 (공간축)

### 4.1 명명 분해

| 기호 | 풀이 |
|---|---|
| **R** | Rule (룰 컴포넌트) |
| **D** | Storage **D**ensity (저장률) |

### 4.2 감시 축

| 감시 축 | 의미 | 운영자 질문 |
|---|---|---|
| **공간축** | FAB 저장공간(STK/ZFS)이 차서 큐가 빠질 곳이 없는가 | "저장공간이 막혔나?" |

### 4.3 보는 원천 컬럼 (메인 1개 + 보조 1개)

#### 메인 컬럼

| 원천 컬럼 | 단위 | 의미 | 산정 방식 |
|---|---|---|---|
| `M16HUB.STRATE.ALL.FABSTORAGERATIO` | % (float) | FAB 내 저장장치 (STK + ZFS) 저장율 | Reserved Location 포함, Down 장비 제외 (smartSTAR 기준) |

#### 보조 컬럼

| 원천 컬럼 | 단위 | 의미 |
|---|---|---|
| `M14B.QUE.ALL.7F_TO_HUB_JOB_ALT` | 개수 (int) | 7F → HUB 반송 JOB (ALT, 보조 검증) |

**총 컬럼 수: 메인 1개 + 보조 1개 = 2개**

### 4.4 트리거 조건

```python
TH_RD_FABSTORAGE = 25.0   # %
TH_RD_7F_HUB_ALT = 20      # 개수 (보조)

rd_fabstorage = M16HUB.STRATE.ALL.FABSTORAGERATIO 의 최신값
rd_7f_alt     = M14B.QUE.ALL.7F_TO_HUB_JOB_ALT 의 최신값

rd_trig = (rd_fabstorage ≥ 25.0)
```

→ **FAB 저장률 ≥ 25%** 면 트리거. 보조 컬럼은 ML 피처 / 로깅용으로 사용.

### 4.5 R-B vs R-D — OR 관계

```
S3 = R-A' AND R-C' AND (R-B OR R-D)
                       └─ 둘 중 하나만 켜져도 OK
```

| 시나리오 | R-B | R-D | 잡힘 여부 |
|---|---|---|---|
| **일반 데드락** (M14→M16 큐 폭증) | ✅ | ❌ | S3 |
| **공간 막힘형** (5/7 SFA) ★ | ❌ | ✅ | S3 |
| **둘 다** | ✅ | ✅ | S3 |
| **둘 다 OFF** | ❌ | ❌ | S2 까지만 |

### 4.6 잡는 시나리오

| 케이스 | 어떻게 잡힘 |
|---|---|
| **5/7 7시 SFA 정체** ★ | M14→M16 큐는 안 쌓이지만 FAB 저장률 28% → rd_trig 발동 → S3 |
| **저장공간 여유** | FAB 저장률 5% → rd_trig 미발동 |
| **저장률 정상이나 큐 폭증** | rd_trig 미발동이지만 rb_trig 가 발동 → S3 |

### 4.7 코드 위치

| 항목 | 위치 |
|---|---|
| 메인 컬럼 읽기 | `hubroom_predictor.py:170` (fabstorage_ratio) |
| 보조 컬럼 읽기 | `hubroom_predictor.py:176` (m14b_7f_to_hub_alt) |
| 임계값 정의 | `hubroom_predictor.py:76-77` (TH_RD_FABSTORAGE, TH_RD_7F_HUB_ALT) |
| 계산 로직 | `hubroom_predictor.py:236-244` (rd_fabstorage, rd_trig 계산) |
| ML 피처 출력 | `rd_fabstorage, rd_7f_alt, rd_trig` (predictions.csv) |

---

## 5. 4룰 한눈에 비교 — 마스터 표

| 항목 | R-A' | R-B | R-C' | R-D ★ |
|---|---|---|---|---|
| **풀네임** | Rule - AvgTotalTime' | Rule - Bridge Queue | Rule - Congestion at lifters' | Rule - storage Density |
| **감시 축** | 시간 | 양 | 위치 | 공간 |
| **운영자 질문** | OHT가 느려졌나? | 큐가 쌓이나? | 한쪽으로 쏠리나? | 저장공간 막혔나? |
| **컬럼 수** | 1개 | 1개 | 10개 | 1+1개 |
| **원천 컬럼** | QUE.TIME.AVGTOTALTIME1MIN | QUE.M14TOM16.MESCURRENTQCNT | LFT.{LID}.TOTAL_CURRENTQCNT (10개) | STRATE.ALL.FABSTORAGERATIO + M14B 7F_TO_HUB_JOB_ALT |
| **메인 임계값** | ≥9.0분 / 10분창 1회+ | +100 / 30분 | 추세감소 + 역증가 ≥2 | ≥25% |
| **보조 임계값** | ≥6.0분 / 5분창 3회+ (ra_sustained) | +30 / 10분 (rb_fast) | — | — |
| **S1 사용?** | ✅ (ra_count≥2 또는 ra_sustained) | ❌ | ❌ | ❌ |
| **S2 사용?** | ❌ | ✅ (rb_trig 또는 rb_fast) | ❌ | ❌ |
| **S3 사용?** | ✅ AND 필수 | ✅ (R-D 와 OR) | ✅ AND 필수 | ✅ (R-B 와 OR) |
| **잡는 정체 유형** | 만성 슬로우다운 | 일반 데드락 (큐 폭증형) | 구조적 데드락 (쏠림) | 공간 막힘형 (SFA 정체) |
| **ML 피처 컬럼** | ra_value, ra_count, ra_sustained, ra_trig | rb_diff, rb_diff_10, rb_fast, rb_trig | rc_trend, rev_count, rev_lids, rc_trig | rd_fabstorage, rd_7f_alt, rd_trig |

---

## 6. S1 / S2 / S3 정의 — 룰 조합

### 6.1 단계 조합

| 단계 | 정의 | 의미 | 발동 케이스 |
|---|---|---|---|
| **S0** | (어떤 트리거도 OFF) | 정상 | — |
| **S1** | (ra_count ≥ 2) OR ra_sustained | 1차 경보 — 시간 신호만 | OHT 슬로우다운 시작 |
| **S2** | rb_trig OR rb_fast | 2차 경보 — 양 신호만 | M14→M16 큐 누적 |
| **S3** | ra_trig AND rc_trig AND (rb_trig OR rd_trig) | 사건 확정 — 3축 합치 | 진짜 데드락 |

### 6.2 S3 공식 풀이

```
S3 = R-A'         AND  R-C'              AND  (R-B            OR  R-D)
   = 시간(느림)   AND  위치(쏠림)        AND  (양 폭증        OR  공간 막힘)
   = ra_trig      AND  rc_trig           AND  (rb_trig        OR  rd_trig)
   = T1MIN ≥9분  AND  (역증가≥2 + 추세↓)  AND  (M14큐 +100/30분  OR  FAB저장률 ≥25%)
```

### 6.3 정체 유형별 트리거 조합

| 정체 유형 | R-A' | R-B | R-C' | R-D | 결과 |
|---|---|---|---|---|---|
| 평상 정상 | ❌ | ❌ | ❌ | ❌ | S0 정상 |
| 일시 슬로우다운 | ✅ | ❌ | ❌ | ❌ | S1 |
| 큐만 누적 | ❌ | ✅ | ❌ | ❌ | S2 |
| 일반 데드락 (4/21 형) | ✅ | ✅ | ✅ | ❌ | **S3** ★ |
| 폭주형 | ✅ | ✅ (rb_fast) | ✅ | ❌ | **S3** ★ |
| **공간 막힘형 (5/7 SFA)** ★ | ✅ | ❌ | ✅ | ✅ | **S3** ★ |
| 단발성 노이즈 | ✅ | ❌ | ❌ | ❌ | S1 (사건 아님) |

---

## 7. 임계값 튜닝 포인트 (운영 시)

| 변수명 | 현재값 | 의미 | 튜닝 가이드 |
|---|---|---|---|
| `TH_RA_VALUE` | 9.0 분 | R-A' 메인 임계 | 평소 1MIN 분포 보고 95~99% 분위 기준 |
| `TH_RA_SUSTAINED_VALUE` | 6.0 분 | ra_sustained 임계 | TH_RA_VALUE × 0.67 정도 |
| `TH_RA_SUSTAINED_COUNT` | 3 회 | 5분창 지속 회수 | 너무 작으면 노이즈 잡힘 |
| `TH_RB_DIFF_30` | +100 / 30분 | R-B 메인 | 정상시 30분간 큐 변화 최대값 + α |
| `TH_RB_DIFF_10` | +30 / 10분 | rb_fast | TH_RB_DIFF_30 × 0.3 |
| `TH_RC_REVERSE` | 2 개 | R-C' 역증가 리프터 수 | 1개로 줄이면 위양성 급증 |
| `TH_RD_FABSTORAGE` | 25.0 % | R-D 메인 | 평소 FAB 저장률 + 20% 여유 |
| `TH_RD_7F_HUB_ALT` | 20 개 | R-D 보조 | ML 피처용, 실 트리거 아님 |

**위치**: `hubroom_predictor.py:70-77`

---

## 8. 자주 묻는 질문

**Q1: 왜 R-A' 와 R-C' 만 프라임(`'`) 이 있고 R-B, R-D 는 없나?**
A: R-A, R-C 는 옛날에 단순 임계값으로만 정의됐다가 개선되면서 프라임 붙음. R-B, R-D 는 처음부터 정밀하게 만들어서 프라임 없음.

**Q2: S3 가 안 잡히는데 위험한 상황이 발생했음. 왜?**
A: S3 는 3축 (시간 AND 위치 AND (양 OR 공간)) 모두 합치돼야 함. 한 축이라도 빠지면 S2 까지만. 그런 케이스를 잡으려고 **ML + 하이브리드** 가 보완해줌.

**Q3: R-C' 의 "역증가" 가 정확히 뭐?**
A: 전체 리프터 합은 줄고 있는데 (rc_trend<0), 그 와중에 일부 리프터만 거꾸로 증가하는 현상. 그 리프터에 큐가 몰린다는 구조적 데드락 신호.

**Q4: R-D 의 보조 컬럼 (7F_TO_HUB_JOB_ALT) 은 트리거에 안 쓰이나?**
A: 안 씀. ML 피처 / 로깅용. 실제 트리거는 FABSTORAGERATIO ≥25% 만 봄.

**Q5: 임계값 (예: 9.0분, 25%) 은 어떻게 정했나?**
A: 과거 사건 데이터 분석 → 정상 vs 사건 분포의 분리점에서 결정. 운영하면서 위양성/누락 보고 조정.

---

## 9. 파일 / 코드 참조

| 항목 | 파일:줄 |
|---|---|
| 룰 임계값 상수 정의 | `hubroom_predictor.py:70-77` |
| 원천 컬럼 매핑 | `hubroom_predictor.py:163-184` |
| 룰 평가 함수 (evaluate_rules) | `hubroom_predictor.py:188-259` |
| LIFTER ID 목록 | `hubroom_predictor.py:64-68` |
| S1/S2/S3 판정 | `hubroom_predictor.py:246-248` |
| ctx 컨텍스트 빌드 | `hubroom_predictor.py:250-258` |
| 사건 추적 FSM | `hubroom_predictor.py:262-` (IncidentTracker) |

---

*본 문서는 hubroom_predictor.py v1.0 기준이며, 룰 임계값 변경 시 운영자가 반드시 업데이트해야 함.*
