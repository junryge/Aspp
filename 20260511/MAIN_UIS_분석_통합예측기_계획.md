# M16 HUBROOM 통합 이벤트 예측기 — 설계 계획서

> `hubroom_predictor_main.py` 통합 예측기 설계
> 입력 자료: `MAIN_UIS/` 폴더 (MAIN_TS.TXT 375줄 + 1.PNG 배치도 + 2.PNG 흐름도)

---

## 0. 입력 자료 분석 (누락 없이)

### MAIN_TS.TXT (375줄) — 5개 섹션

| 섹션 | 내용 | 핵심 |
|---|---|---|
| ① 미션 (line 1) | "M16HUB, M14B, M14, M16_PKT, M16_WT, M16A, M16, M16B **통합 이벤트 예측**" | 8개 FAB 통합 |
| ② 임계값 (line 6-18) | R-A'/R-B/R-C'/R-D + 사건/윈도우 11개 상수 | 검증된 임계값 보존 |
| ③ 시나리오 (line 21-36) | 14개 정체 유형 매트릭스 (정상~복합형 데드락) | S3 4가지 형태 |
| ④ 수집 컬럼 (line 39-335) | 24개 (현재 룰) + 13개 (0512 이상) + ~250개 (의미상 관련) | 약 290개 전체 |
| ⑤ 도메인 지식 (line 338-376) | FAB 8개 + 반송장치 7종 + 연결 10개 | 물리적 흐름 구조 ★ |

### 1.PNG — 반도체 단지 배치도
- 단지 평면도 (M14A/B, M16A/B/E/WT, R4 위치)
- 건물 간 연결 다이어그램 (LFT/CNV/STK 노선)

### 2.PNG — M16 HUB ROOM 트래픽 흐름도 ★★★

**9개 핵심 노드 — 현재값 vs 예상값**:

| 노드 | 의미 | 현재 | 예상 | **비율** | 평가 |
|---|---|---|---|---|---|
| **M14CNV** | M14A CNV → M16 HUB | **212** | 72 | **2.9x** | ★★★ 최대 병목 |
| **M14B** | M14B → M14B LFT | **108** | 50 | **2.2x** | ★★ |
| **M16HUBOHT** | M16 HUB → M16 2F/6F | **166** | 98 | **1.7x** | ★ |
| M14A | M14A → M14A CNV | 31 | 33 | 0.94x | 정상 |
| M14LFT | M14B → M16 HUB | 13 | 24 | 0.54x | 정상 |
| M16LFT | M16 → M16 6F | 89 | 89 | 1.0x | 정상 |
| M166F | M16 LFT → M16 6F | 28 | 78 | 0.36x | 정상 |
| M16HUBMLUD | HUB MLUD → M16 HUB | 1 | 7 | 0.14x | 정상 |
| M16EUV | M16 LFT → M16 2F | 0 | 2 | 0x | 정상 |
| M16LFT2F | M16 HUB → M16 2F | 14 | 4 | 3.5x | 주의 |

→ **핵심 통찰**: 절대값 아닌 **현재 / 예상 비율** 이 진짜 정체 지표.

---

## 1. 핵심 이해 — 3가지

### ① M16 HUBROOM = 8개 FAB 의 물류 허브

```
                M14B (7F)
                  │ LFT
                  ↓
   M14A (3F) ── CNV ─── M16 HUBROOM (3F) ─── LFT ──── M16A (6F)
                          │                            │
                          ├── LFT ── M16 EUV (2F)      │ LFT
                          │                            │
                          └── LFT ── M14 분석실 (B1F)   M16B (10F)
                                                       │
                                                       └─ M16 WT (2F)
                                                       └─ R4 (6F)
```

### ② 기존 룰베이스의 한계 (3가지)

| 한계 | 영향 |
|---|---|
| **M16HUB 내부만 봄** | 인접 FAB 발 정체를 사후에 잡음 → Lead time 손해 |
| **흐름 방향 안 봄** | "어디서 → 어디로" 전파됐는지 모름 |
| **24개 컬럼만 사용** | 290개 가용 컬럼 중 8% 만 활용 |

### ③ 통합 예측기가 해결할 것

| 문제 | 통합 해결 |
|---|---|
| M14A 막혀서 CNV 폭주 시 사후 인지 | **인플로 폭증 즉시 감지** (5~10분 추가 사전인지) |
| 사건 시 원인 영역 모름 | **hot_area 컬럼**: "M14A 시작" 자동 식별 |
| 정체 전파 경로 모름 | **propagation_chain**: "M14 → CNV → HUB → M16A" |

---

## 2. 통합 예측기 설계 — 3-Layer 구조

```
┌──────────────────────────────────────────────────┐
│ Layer 3: 통합 융합 + 사건 판정 + 전파 추적         │
│   - 통합 위험도 (unified_risk_score)              │
│   - 어디서 시작 (hot_area)                        │
│   - 전파 체인 (propagation_chain)                 │
│   - 사건 단위 (S3 확장)                           │
└────────────────▲─────────────────────────────────┘
                 │
┌────────────────┴─────────────────────────────────┐
│ Layer 2: 흐름 룰 (9개 핵심 노드)                  │
│   각 노드별 [현재값 / 예상값] 비율 평가           │
│   - 인플로 5개 (HUB 로 들어옴)                    │
│   - HUB 내부 3개                                  │
│   - 출구 4개 (HUB → 외부)                         │
└────────────────▲─────────────────────────────────┘
                 │
┌────────────────┴─────────────────────────────────┐
│ Layer 1: 영역별 4축 룰                            │
│   기존 R-A'/R-B/R-C'/R-D 를 각 영역별로 평가      │
│   M16HUB / M14A / M14B / M16A / M16B            │
└──────────────────────────────────────────────────┘
```

---

## 3. Layer 1 — 영역별 4축 룰 매트릭스

| 축 \ 영역 | **M16HUB** | **M14A** | **M14B** | **M16A** | **M16B** |
|---|---|---|---|---|---|
| **시간(R-A')** | T1MIN ≥9분 | M14 T1MIN | M14B T1MIN | M16A 1MIN | M16B 1MIN |
| **양(R-B)** | M14→M16 +100/30분 | M14A CNV Q 변화 | M14B LFT Q 변화 | M16A→HUB 변화 | M16B→HUB 변화 |
| **위치(R-C')** | 리프터 10대 역증가 | 4AFC3201/3301 | 4ABLD 6대 | 6ABL01 4대 | (해당 없음) |
| **공간(R-D)** | FAB 저장률 ≥25% | M14A OHTUTIL | M14B OHTUTIL | M16A OHTUTIL | M16B OHTUTIL |
| **4분초과** ★ | TRANSPORT4MINOVERCNT | M14 동일 | (없음) | M16A 동일 | M16B 동일 |

→ **5개 영역 × 5축 = 25개 셀** 동시 평가.

### 출력 ctx 컬럼 예시
```
ra_value_hub, ra_value_m14a, ra_value_m14b, ra_value_m16a, ra_value_m16b
rb_diff_hub, rb_diff_m14a, ...
rc_trig_hub, rc_trig_m14a, ...
rd_fab_hub, rd_oht_m14a, ...
t4over_hub, t4over_m14, t4over_m16a, t4over_m16b
```

---

## 4. Layer 2 — 흐름 룰 (9개 핵심 노드)

2.PNG 의 9개 노드를 룰화. **현재값 / 30분 평균 비율** 기준.

### 인플로 5개 (HUB 로 들어옴)
```python
flow_M14_in   = M14.QUE.ALL.3F_TO_HUB_JOB         # M14 → HUB
flow_M14B_in  = M14B.QUE.ALL.7F_TO_HUB_JOB        # M14B 7F → HUB
flow_M16A_2F  = M16A.QUE.ALL.2F_TO_HUB_JOB        # M16A 2F → HUB
flow_M16A_6F  = M16A.QUE.ALL.6F_TO_HUB_JOB        # M16A 6F → HUB
flow_M16B_in  = M16B.QUE.ALL.10F_TO_HUB_JOB       # M16B → HUB
```

### HUB 내부 3개
```python
hub_oht       = M16HUB.QUE.OHT.CURRENTOHTQCNT
hub_storage   = M16HUB.STRATE.STB.3F_STORAGE_UTIL
hub_mlud      = M16HUB.QUE.ALL.M16HUBTOM14MANUAL_CURRENTQCNT
```

### 출구 4개 (HUB → 외부)
```python
out_M14A     = M16HUB.QUE.ALL.3F_TO_M14A_3F_JOB
out_M14B     = M16HUB.QUE.ALL.3F_TO_M14B_7F_JOB
out_M16A_2F  = M16HUB.QUE.ALL.3F_TO_M16A_2F_JOB
out_M16A_6F  = M16HUB.QUE.ALL.3F_TO_M16A_6F_JOB
```

### 노드별 룰 (현재/30분평균 비율)

| 비율 | 등급 | 운영자 행동 |
|---|---|---|
| ≥ **1.5x** | 주의 | 모니터링 |
| ≥ **2.0x** | 위험 | 즉시 확인 |
| ≥ **3.0x** | ★ 심각 | 즉시 조치 (2.PNG M14CNV 2.9x 가 이 케이스) |

---

## 5. Layer 3 — 통합 융합 + 전파 추적

### 확장 사건 판정 (S3 강화)

```
통합 S3 = (어느 영역이든 시간축 R-A' 발동)
       AND (어느 영역이든 위치축 R-C' 발동)
       AND (인플로 1개+ 비율 ≥2x  OR  HUB 내부 R-D 발동)
```

기존 S3 = `R-A'(HUB) AND R-C'(HUB) AND (R-B OR R-D)` (HUB 만)
→ 통합 S3 = **어느 영역이든 만족** (8개 FAB 다 봄)

### 전파 추적 (★ 가장 가치)

시간축 분석으로 정체 전파 경로 자동 식별:

```python
# t-15분 ~ t 분석
if flow_M14_in.delta(15min) > +30:
    propagation_chain.append("M14 인플로 +30/15분 폭증")
    hot_area = "M14"

if 5분 뒤 hub_oht.delta(5min) > +50:
    propagation_chain.append("→ HUB OHT 큐 +50")

if 10분 뒤 hub T1MIN > 9분:
    propagation_chain.append("→ HUB R-A' 발동")

# final_reason 자동 생성:
# "M14 시작 → HUB 도달 (전파 lead_time 15분)"
```

### Layer 3 신규 출력 컬럼

| 컬럼 | 의미 | 예시 |
|---|---|---|
| `unified_risk_score` | 통합 위험도 (0~300) | 142 |
| `unified_risk_level` | 통합 위험 레벨 | "매우위험" |
| `hot_area` | 시작 영역 | "M14A" |
| `propagation_chain` | 전파 경로 | "M14A CNV +120 → HUB T1MIN +3.5" |
| `lead_time` | 전파 시작→현재 (분) | 25 |
| `affected_areas` | 영향 받은 영역 | "M14A, HUB" |

---

## 6. 운영자가 한 줄에서 보는 결과 (목표)

### 발동이벤트.csv 한 줄 예시
```
2026-05-14 14:23:00,
unified_risk_score=92, unified_risk_level=매우위험,
hot_area=M14A,
propagation_chain="M14A CNV +120/10분 → 5분 뒤 HUB OHT +50 → 10분 뒤 HUB T1MIN 9.5분",
lead_time=25분,
affected_areas="M14A; HUB",
layer1=[ra_hub=O,rc_hub=O,m14a_rb=O(CNV 폭주)],
layer2=[m14_cnv=2.9x ★★★, hub_oht=1.7x],
ml_score=0.87, risk_factors=ra_sustained;rb_fast;m14a_cnv_x2.9
```

→ 운영자 즉시 판단: **"M14A CNV 쪽 봐야 한다"**

---

## 7. 기존 vs 통합 — 비교 표

| 항목 | 기존 (hubroom_predictor.py) | 통합 (hubroom_predictor_main.py) |
|---|---|---|
| **감시 범위** | M16HUB 내부 1개 영역 | **8개 FAB 영역** |
| **컬럼 수** | 24개 | **약 60~80개** |
| **룰 개수** | 4개 (R-A'/R-B/R-C'/R-D) | **4축 × 5영역 + 흐름 9개 = 29개** |
| **사전 인지** | 평균 53분 | **70~90분 예상** (인플로 선행 신호) |
| **원인 파악** | "HUB 정체" | **"어느 FAB 에서 시작 → 어디로 전파"** |
| **운영자 행동** | "어디 봐야 하지?" | **"M14A CNV 봐라"** |
| **위양성** | R-D 단독으로 잡힌 케이스 多 | **흐름 비율 기준으로 위양성 감소** |

---

## 8. 진행 로드맵

### Phase 1 (1주) — Layer 1 영역별 4축
- 5개 영역 × 5축 컬럼 추가
- 25개 셀 동시 평가
- 출력: 영역별 ctx + 위험도 (기존 호환)

### Phase 2 (1주) — Layer 2 흐름 룰
- 9개 흐름 노드 룰 추가
- 비율 기반 발동 (1.5x/2x/3x)
- hot_area 식별

### Phase 3 (1주) — Layer 3 전파 추적
- 시계열 패턴 매칭
- propagation_chain 자동 생성
- 통합 S3 강화

### Phase 4 (1주) — 검증 + ML 재학습
- dss_data.csv 로 검증
- ML 피처 통합 (총 100개+)
- model.json 재학습

→ **총 예상 4주**.

### 추천: **Phase 1 부터 점진적 진행**

이유:
- 기존 코드 그대로 작동 (S3 정의 보존)
- 영역별 신호만 추가 — 부작용 적음
- 운영자가 점진적 적응 가능

---

## 9. 데이터 / 컬럼 요구사항

### 필수 추가 수집 컬럼 (수집기 v3.2 필요)

이미 수집 중 (v3.1) ✓:
- M14, M14B, M16A 2F/6F, M16B 인플로 5개
- M16HUB.STRATE.STB.3F_STORAGE_UTIL

신규 필요:
- **영역별 T1MIN 5개**: M14, M14B, M16A 별 T1MIN
- **영역별 OHTUTIL 5개**: M14B 만 있음 → M14, M16A, M16B 추가
- **영역별 4분초과** 3개: M14, M16A, M16B
- **HUB 출구 4개**: M16HUB.QUE.ALL.3F_TO_M16A_6F_JOB 등

→ **약 17개 추가 컬럼** 수집기에 더 넣어야 함.

---

## 10. 데이터 검증 — 사용자 줄 수 있다고 한 것

### 필요한 데이터 (우선순위)

1. ★★★ **통합 사건 라벨**: 5월 사건들의 "원인 영역" 표시
   - 예: "5/8 13:00 사건 = M14A 시작" / "5/14 10:00 사건 = M14B 시작"

2. ★★ **0512 case 상세**: 0512 (5/12) 사건의 영역별 데이터
   - MAIN_TS.TXT 에 "0512 case 이상지표" 라고 적힘
   - 이 사건이 어떤 영역에서 시작됐는지 알면 전파 추적 모델 검증 가능

3. ★ **5/7 SFA 정체 검증**: 이미 R-D 로 잡고 있는데 통합 시 더 빨리 잡는지

---

## 11. 결론 — 한 줄 요약

> **기존 룰베이스 (HUB 1개 영역) → 통합 룰베이스 (8개 FAB + 9개 흐름)**
> **사전 인지 53분 → 70~90분**
> **원인 영역 식별 가능 (hot_area)**
> **위양성 감소 (비율 기반 판정)**
> **Phase 1 부터 점진적 진행 추천**

---

*본 문서는 MAIN_UIS/ (MAIN_TS.TXT + 1.PNG + 2.PNG) 분석 기반 통합 예측기 설계서.*
