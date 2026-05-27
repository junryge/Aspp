# M16 HUBROOM 통합 이벤트 예측기 — 설계 계획서 v2.0

> `hubroom_predictor_main.py` 통합 예측기 설계 (269개 컬럼, 8개 FAB 영역)
> 입력 자료: `MAIN_UIS/` 폴더 (MAIN_TS.TXT + 1.PNG + 2.PNG) + v3.1 보강
> 컬럼 정리: `AWS_IDC_v4.1_컬럼_전체정리.md` 참조 (누락 0)

---

## 0. 입력 자료 분석 (누락 없이)

### 컬럼 출처

| 출처 | 컬럼 수 | 비고 |
|---|---|---|
| MAIN_TS.TXT (line 41-335) | 263개 | 도메인 자료 |
| v3.1 M16_PKT/WT 보강 | 6개 | OHTUTIL, T1MIN, OHTMCPALARM |
| **총 v4.1** | **269개** | **누락 0** ✓ |

### 영역별 분포

| 영역 | 컬럼 수 | 역할 |
|---|---|---|
| **M16HUB** | 104 | 중심 허브 (3F) — M14↔M16 물류 연결 |
| **M14** | 41 | M14A (3F) — CNV로 HUB 연결 |
| **M14B** | 41 | M14B (7F) — LFT(4ABLD)로 HUB 연결 |
| **M16A** | 39 | M16A (6F) — LFT(6ABL)로 HUB 연결 |
| **M16B** | 25 | M16B (10F) — M16A 경유 HUB |
| **M16** | 11 | SFAB FAB간 반송 |
| **M16_PKT** | 4 | 브릿지 |
| **M16_WT** | 4 | 브릿지 (M16EUV 연결) |
| **합계** | **269** | |

### 2.PNG 흐름도 핵심 통찰 ★★★

9개 핵심 노드의 **현재값 / 예상값 비율** 이 진짜 정체 지표:

| 노드 | 현재 | 예상 | 비율 | 평가 |
|---|---|---|---|---|
| **M14CNV → HUB** | **212** | 72 | **2.9x** | ★★★ 최대 병목 |
| **M14B → M14B LFT** | **108** | 50 | **2.2x** | ★★ 누적 |
| **M16HUBOHT** | **166** | 98 | **1.7x** | ★ HUB 적체 |
| M16LFT2F | 14 | 4 | 3.5x | 주의 (절대값 작음) |
| 나머지 5개 | — | — | <1x | 정상 |

→ **절대값 아닌 비율** 이 정체의 본질.

---

## 1. 도메인 기반 핵심 이해

### FAB 연결 구조 (10개 연결)

```
                M14B (7F) ━━LFT(4ABLD)━━━┓
                                          ↓
   M14A (3F) ━━CNV(4AFC32/33)━━━━━━━━━━► M16 HUBROOM (3F)
        │                                  │ ↑
        ↓ LFT                              ↓ │
   M10A (2F)                          ┌────┴─┴─────┐
                                      ↓            ↓
                                M16A (6F)     M16EUV (2F)
                                LFT(6ABL)         ↓ WIS STK
                                  ↓ LFT          M16WT (2F)
                              M16B (10F)
```

### 기존 한계 → 통합 해결

| 한계 (기존) | 통합 해결 (v2.0) |
|---|---|
| M16HUB 24개만 봄 | **269개 활용 (8개 영역 통합)** |
| 인접 FAB 사후 인지 | **인플로 폭증 즉시 감지** |
| 위치 모름 | **hot_area 식별 (어느 FAB 시작)** |
| 전파 경로 모름 | **propagation_chain 추적** |
| **운영자 MAXCAPA 무관심** | **MAXCAPA 6개 운영자 변수 활용** ★ |

---

## 2. 통합 예측기 설계 — 3-Layer 구조

```
┌──────────────────────────────────────────────────┐
│ Layer 3: 통합 융합 + 사건 판정 + 전파 추적         │
│   - unified_risk_score (0~500)                    │
│   - hot_area (어느 FAB 시작)                      │
│   - propagation_chain (전파 경로)                 │
│   - 통합 S3 (8개 영역 어디든 조건 만족)            │
└────────────────▲─────────────────────────────────┘
                 │
┌────────────────┴─────────────────────────────────┐
│ Layer 2: 다축 룰 (4축 + 흐름 + 운영자변수)        │
│   - 4축 룰: R-A'/R-B/R-C'/R-D × 6 영역 = 24셀      │
│   - 흐름 룰: 9개 핵심 노드 (비율 1.5x/2x/3x)      │
│   - SLA 룰: 4분초과 12개 (영역별)                 │
│   - Sorter 룰: 8개 (LOT 적체)                     │
│   - 운영자 변수 룰: MAXCAPA 6개 ★                  │
└────────────────▲─────────────────────────────────┘
                 │
┌────────────────┴─────────────────────────────────┐
│ Layer 1: 원천 데이터 수집 (269 컬럼)              │
│   8개 FAB 영역 × 카테고리                         │
└──────────────────────────────────────────────────┘
```

---

## 3. Layer 1 — 카테고리별 핵심 분류 (269개)

### ★★★ 운영자 변수 — 6개 (MAXCAPA)
> **메신저에서 운영자가 변경하는 값** = 정체 발생/조치 직접 신호

| 컬럼 | 영역 |
|---|---|
| M16HUB.QUE.LFT.3F_LFT_MAXCAPA | M16HUB |
| M16HUB.QUE.LFT.3F_M14BLFT_MAXCAPA | M16HUB |
| M16HUB.QUE.CNV.3F_CNV_MAXCAPA | M16HUB |
| M14.QUE.CNV.3F_CNV_MAXCAPA | M14 |
| M16A.QUE.LFT.2F_LFT_MAXCAPA | M16A |
| M16A.QUE.LFT.6F_LFT_MAXCAPA | M16A |

→ MAXCAPA 가 평소 100% → 50% 변경되면 운영자가 인지한 상태.

### ★★★ 4분 이상 SLA — 12개 (0512 case 핵심)
영역별 TRANSPORT4MINOVERCNT/RATIO/TIMEAVG:
- M16HUB (3), M14 (3), M16A (3), M16B (3)
- → **정체의 직접 결과 지표**

### ★★ HUB 인플로 — 9개
> 5개 FAB → HUB 로 들어오는 부하

| 컬럼 | 출발지 |
|---|---|
| M14.QUE.ALL.3F_TO_HUB_JOB | M14A 3F |
| M14.QUE.ALL.3F_TO_HUB_JOB_ALT | M14A 우회 |
| M14.QUE.OHT.3F_TO_HUB_CMD | 진행중 OHT |
| M14B.QUE.ALL.7F_TO_HUB_JOB | M14B 7F |
| M14B.QUE.ALL.7F_TO_HUB_JOB_ALT | M14B 우회 |
| M14B.QUE.OHT.7F_TO_HUB_CMD | 진행중 |
| M16A.QUE.ALL.2F_TO_HUB_JOB | M16A 2F |
| M16A.QUE.ALL.6F_TO_HUB_JOB | M16A 6F |
| M16B.QUE.ALL.10F_TO_HUB_JOB | M16B 10F |

### ★★ HUB 출구 — 5개
> HUB → 외부로 나가는 큐 (막히면 HUB 내부 적체)

```
M16HUB.QUE.ALL.3F_TO_M16A_6F_JOB
M16HUB.QUE.ALL.3F_TO_M16A_2F_JOB
M16HUB.QUE.ALL.3F_TO_M14A_3F_JOB
M16HUB.QUE.ALL.3F_TO_M14B_7F_JOB
M16HUB.QUE.ALL.3F_TO_3F_MLUD_JOB
```

### ★ Sorter — 8개 (LOT 적체)
M14, M14B, M16A, M16B 각자 SORTERWAITCOUNTOVER 등

### ★ HUB 내부 CMD/JOB — 11개
M16HUB.QUE.LFT/MLUD/STB/CNV/ALL 의 CMD/JOB

### 저장률 — 5개
M16HUB.STRATE.STK/ALL.FABSTORAGERATIO/STB.3F_STORAGE_UTIL 등

### 영역간 흐름 (CNV/LFT) — 30개+
M14↔M16 브릿지, CNV 남/북측, LFT 합계 등

### 리프터 큐 — 96개
- TOTAL_CURRENTQCNT 16개 (M16HUB 10 + M14B 6)
- 방향별 80개 (3F↔2F, 3F↔6F, 2F↔6F 등)

### OHT 가동/큐/알람 — 12개
각 영역별 OHTUTIL, CURRENTOHTQCNT, OHTMCPALARMCNT

### 평균 시간 — 12개
T1MIN, AVGTOTALTIME, AVGLOADTIME 시리즈

### M16 SFAB — 11개
SEND/RETURN/COMPLETE/RECEIVE (M16↔M14/M10)

---

## 4. Layer 2 — 룰 평가 (5종)

### 4.1 영역별 4축 룰 매트릭스 (24셀 + M16B는 R-C' 없어 23셀)

| 축 \ 영역 | **M16HUB** | **M14** (=M14A) | **M14B** | **M16A** | **M16B** | **M16** |
|---|---|---|---|---|---|---|
| **R-A' (시간)** | T1MIN ≥9분 | AVGLOADTIME1MIN | T1MIN | LOAD1MIN | LOAD1MIN | (SFAB) |
| **R-B (양)** | M14→M16 +100/30분 | CNV M14A→M16A 변화 | 7F→HUB 변화 | 2F+6F→HUB 변화 | 10F→HUB 변화 | SFAB SEND 변화 |
| **R-C' (위치)** | 리프터 10대 역증가 | CNV 남/북측 쏠림 | LFT 6대 (4ABLD) | LFT 4대 (6ABL01) | (해당없음) | (해당없음) |
| **R-D (공간)** | FAB 저장률 ≥25% | OHTUTIL ≥90% | OHTUTIL ≥90% | OHTUTIL ≥90% | OHTUTIL ≥90% | — |

### 4.2 흐름 룰 — 9개 노드 (현재/30분평균 비율)
| 비율 | 등급 |
|---|---|
| ≥ 1.5x | 주의 |
| ≥ 2.0x | 위험 |
| ≥ 3.0x | ★ 심각 (예: 2.PNG M14CNV 2.9x) |

### 4.3 SLA 룰 — 4분 이상 (영역별 12개)
- TRANSPORT4MINOVERRATIO ≥ 15%
- TRANSPORT4MINOVERCNT 10분 +20 폭증
- → 정체 직접 지표

### 4.4 Sorter 룰 — 영역별 임계값
- SORTERWAITCOUNTOVER ≥ 100 → 적체 신호
- SORTERTRANSFERFAIL 발생 → 알람

### 4.5 운영자 변수 룰 (★ 신규)
- MAXCAPA 가 100 → 50 변경 = **운영자 인지 시점**
- MAXCAPA 50 → 1 변경 = 심각도 상승
- 이걸 시계열 추적하면 운영자 행동 패턴 학습

---

## 5. Layer 3 — 통합 융합 + 전파 추적

### 5.1 통합 S3 정의

```
통합 S3 = (어느 영역이든 시간축 R-A' 발동)
       AND (어느 영역이든 위치축 R-C' 발동)
       AND (인플로 1개+ 비율 ≥2x  OR  HUB 내부 R-D 발동  OR  4분초과 ≥15%)
```

### 5.2 전파 추적 알고리즘

```python
# 시간축 분석 (t-30 ~ t)
if M14.인플로_delta(15min) > +30 또는 비율 ≥2x:
    propagation.append({
        'area': 'M14', 'time': t-15, 'signal': '인플로 폭증'
    })

# 5분 뒤
if HUB.OHT_큐_delta(5min) > +50:
    propagation.append({
        'area': 'HUB', 'time': t-10, 'signal': 'OHT 큐 누적'
    })

# 10분 뒤
if HUB.R-A'_발동:
    propagation.append({
        'area': 'HUB', 'time': t, 'signal': 'T1MIN ≥9분'
    })

# 전파 체인 자동 생성:
# "M14 인플로 폭증(t-15) → HUB 적체(t-10) → HUB R-A'(t)"
```

### 5.3 Hot Area 식별

가장 먼저 신호 발동한 영역 = hot_area
- 시간 가중치: 가장 먼저 발동 (-50점)
- 신호 강도 가중치: 비율 ≥3x (+30점)
- 운영자 인지 가중치: MAXCAPA 변경 있음 (+50점)

### 5.4 통합 위험도 점수 (확장)

기존 risk_score (0~170) → unified_risk_score (0~500)

| 카테고리 | 점수 |
|---|---|
| 영역별 4축 (5영역) | 각 영역 0~50점 × 5 = 250 |
| 흐름 룰 (9개 노드) | 각 노드 0~10점 × 9 = 90 |
| SLA 룰 (4분초과) | 12개 × 5점 = 60 |
| Sorter | 8개 × 3점 = 24 |
| 운영자 변수 (MAXCAPA) | 6개 × 10점 = 60 ★ |
| **합계** | **0~500** |

---

## 6. 운영자가 한 줄에서 보는 결과 (목표)

### 발동이벤트.csv 한 줄 예시

```
2026-05-14 14:23:00,
unified_risk_score=287, unified_risk_level=매우위험,
hot_area=M14A,
propagation_chain="M14A CNV +120/10분(t-15) → HUB OHT +50(t-10) → HUB T1MIN 9.5분(t)",
lead_time=15분,
affected_areas="M14A; HUB",
layer1_signals=[ra_hub=O, rc_hub=O, m14a_rb=O(CNV폭주)],
layer2_flows=[m14_cnv=2.9x ★★★, hub_oht=1.7x],
layer2_sla=[m14_t4over=18%, hub_t4over=12%],
operator_action=[m14_maxcapa=50% (t-5분 변경)],
ml_score=0.87,
risk_factors=ra_sustained;rb_fast;m14a_cnv_x2.9;m14_t4over_18%;maxcapa_changed
```

→ 운영자 즉시 판단: **"M14A CNV 쪽이 문제, MAXCAPA 50% 이미 변경되어 있음, HUB 임박"**

---

## 7. 기존 vs 통합 — 비교 표

| 항목 | 기존 (v3.1) | 통합 (v4.1) |
|---|---|---|
| 컬럼 수 | 64 | **269** (+205) |
| 영역 수 | 4 (HUB+M14B+PKT/WT) | **8 (전체 FAB)** |
| 룰 종류 | 4축 1세트 | **4축×5 + 흐름 + SLA + Sorter + 운영자** |
| 사전 인지 | 평균 53분 | **70~90분 예상** |
| 원인 파악 | "HUB 정체" | **"M14A 시작 → 전파 경로"** |
| 운영자 변수 | 미활용 | **MAXCAPA 6개 학습** ★ |
| Sorter 신호 | 미활용 | **8개 추적** ★ |
| 위양성 방어 | 임계값 | **다축 + 운영자 변수 교차검증** |

---

## 8. 진행 로드맵

### Phase 0 — 데이터 수집 (이번 주) ★ 현재 위치
- ✅ `AWS_IDC_v4.1_컬럼_전체정리.md` (269개 정리)
- ⏳ `AWS_IDC_QUERY_v4.1.sql` 작성 (269개 컬럼)
- ⏳ 학습 데이터 추출 (CSV)
- ⏳ 영역별 정상 분포 분석 → 임계값 도출

### Phase 1 — 룰베이스 통합 (1주)
- 영역별 4축 평가 (5영역)
- 흐름 룰 (9개 노드)
- SLA 룰 (12개)
- Sorter 룰 (8개)
- 운영자 변수 룰 (MAXCAPA 6개)

### Phase 2 — 전파 추적 (1주)
- propagation_chain 자동 생성
- hot_area 식별
- 시계열 패턴 매칭

### Phase 3 — ML 재학습 (1주)
- 피처 70개 → 200개+ (영역별 + 흐름 + SLA + 운영자)
- model.json 재학습

### Phase 4 — 하이브리드 융합 갱신 (3일)
- 통합 룰+ML 받는 hybrid_predictor.py 수정

→ **총 약 3주** (Phase 0 완료 후).

---

## 9. 데이터 / 컬럼 요구

### 새로 수집해야 할 컬럼 (수집기 v4.1)

이미 수집 중 (v3.1) ✓ — 64개

신규 추가 필요 — **205개**:
- **★★★ 운영자 변수 MAXCAPA 6개** (최우선)
- **★★ 4분초과 영역별 9개** (M14/M16A/M16B 추가)
- **★★ HUB 출구 5개** (3F_TO_*_JOB)
- **★★ HUB CMD/JOB 11개**
- **★ Sorter 8개** (LOT 적체 신호)
- **★ M14 CNV 남/북측 세부 9개**
- **★ 영역간 흐름 30개+** (CNV/LFT)
- 리프터 방향별 80개 (R-C' 정밀화 — 선택)
- 나머지 영역별 기본 큐/시간 ~50개

→ **수집기 (aws_idc_realtime_collector.py) 도 v4.1 로 업데이트 필요**.

### 학습 데이터 추출 (SQL v4.1)
- 기간: 2026-01-01 ~ 2026-05-14 (4.5개월)
- 컬럼: 269개
- 행 수: 약 192,000행
- 예상 파일 크기: 약 60~80 MB

---

## 10. 데이터 검증 — 사용자 보유

### 필요한 라벨/사건 정보 (우선순위)

1. ★★★ **8개 사건 영역별 라벨**
   - 5월 사건마다 "어느 FAB 가 시작점" 표시
   - 예: "5/8 13:00 = M14A 시작" / "5/14 10:00 = M14B 시작"

2. ★★ **0512 case 상세** (MAIN_TS.TXT 언급된 사건)
   - 5/12 사건 영역별 데이터

3. ★ **5/7 SFA 정체** 검증 (이미 R-D 잡음)

4. ★★ **메신저 → 영역 매핑**
   - 운영자가 어느 영역 MAXCAPA 변경했는지

---

## 11. 룰베이스 가능성 판단

### 룰베이스로 통합 예측 — **80% 가능** (도메인 지식 덕분)

**룰베이스 강점** (도메인 지식 기반):
- FAB 연결도 (10개) 직접 룰화 → 흐름 추적 가능
- 반송장치 7종 특성 → 각자 이상 신호 명확
- 영역별 임계값 (정상 분포 기반) → 영역별 룰 가능
- 운영자 행동 패턴 (MAXCAPA 변경) → 이미 데이터로 검증 가능

**룰베이스 한계 (ML 보완 필요)**:
- 영역 간 비선형 상호작용 (가중치 학습)
- 신호 강도 정량화 (연속값)
- 위양성 자동 감소 (학습)

**최선**: 룰베이스 단독 80% → 룰+ML 결합 90%+

---

## 12. 결론

> **269개 컬럼 / 8개 FAB / 5종 룰 → 통합 이벤트 예측기**
> **사전 인지 53분 → 70~90분**
> **원인 영역 자동 식별 (hot_area)**
> **전파 경로 자동 추적 (propagation_chain)**
> **운영자 MAXCAPA 변수까지 활용 (★ 신규)**
> **Phase 0 (데이터 수집) → Phase 1 (룰베이스) → Phase 2~4 점진 진행**

---

*본 문서는 v4.1 (269컬럼) 기반 통합 예측기 설계서이다.*
*상세 컬럼 목록: `AWS_IDC_v4.1_컬럼_전체정리.md`*
