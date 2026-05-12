# ML 도입 — 왜 필요한가, 어떻게 만드나

> **핵심 메시지**: 룰베이스로 "정체 발생"은 잡지만, ML로 **"정체 발생 30분 전 사전경보"** 가능.
> 운영자가 인지하기 전에 시스템이 미리 알람 = 골든 타임 확보.

---

## 1. 왜 ML이 필요한가

### 룰베이스의 한계

룰베이스는 **임계값을 넘어야** 발동:

```
1MIN 9분 이상  → R-A' 발동
M14→M16 +100  → R-B 발동
FABSTORAGE 25%+ → R-D 발동
```

문제:
- 임계 도달 = 이미 정체 시작
- **운영자 대응이 사후 대응**
- 큐가 쌓인 후라 회복 시간 길어짐

### 5/7 실제 사례

| 시각 | 상황 | 룰 |
|---|---|---|
| 06:30 | 정상 | - |
| 07:00 | FABSTORAGE 10% (서서히 상승) | 정상 (임계 미달) |
| 07:15 | 1MIN 7~8분 (조금씩 길어짐) | S1 (조기경보) |
| 07:31 | **운영자 채팅 첫 인지** | - |
| **07:32** | **룰 S3 발동** (FABSTORAGE 24%) | **사건 확정** |
| 07:40 | FABSTORAGE 31% (피크) | 사건 진행 |

**룰이 잡았을 때 = 운영자 인지 후** → 사전 대응 불가

### ML이 해결하는 것

```
06:55 시점 데이터 (사건 37분 전)
   ↓
ML 30분 사전 예측 모델
   ↓
"07:25 정체 확률 78%"
   ↓
06:55에 사전 알람 → 30분 여유로 대응
```

---

## 2. 룰 + ML 하이브리드 구조

```
                  ┌─ 룰  ──→ 명확한 사건 (S3 확정) — 즉시 대응
1분 데이터 ───┤
                  └─ ML  ──→ 30분 사전 예측 — 미리 알람 ★
                                ↓
                            결합 로직
                                ↓
                  ┌── 룰 S3              → 확정사건 (즉시)
                  ├── ML > 0.7 + 룰 S1/S2 → ★ 사전경보 (30분 전)
                  ├── ML > 0.85 + 룰 정상  → 잠재위험 (모니터링)
                  └── 모두 낮음           → 정상
```

### 역할 분담

| | 룰 | ML |
|---|---|---|
| **언제** | 사건 발생 직후 | 사건 30분 전 |
| **확실성** | 100% (검증된 패턴) | 70~85% (확률) |
| **설명력** | 명확 ("FABSTORAGE 25% 초과") | SHAP 값으로 가능 |
| **놓치는 케이스** | 임계 미달 시 | 학습 안 된 새 패턴 |

**합치면**: 룰이 안전망, ML이 망원경.

---

## 3. ML 모델 (XGBoost) — 왜 이걸 쓰나

### XGBoost 선택 이유

| 모델 | 장단점 | 적합도 |
|---|---|---|
| LSTM | 시계열 강함, 학습 어려움, 블랙박스 | △ |
| Random Forest | 단순, 정확도 낮음 | △ |
| **XGBoost** | **빠름, 정확, SHAP으로 설명 가능, 작은 데이터에서도 작동** | ★ |
| Autoencoder | 비지도, 정상만 학습 가능 | ○ (보조) |
| Transformer | 데이터 대량 필요 | ✗ |

**XGBoost가 현실적 베스트**:
- 사건 라벨 적음(4건) → LSTM/Transformer 학습 불가
- XGBoost는 **수십~수백 라벨로도 작동**
- SHAP으로 "왜 위험하다고 판단했나" 설명 가능 → 운영자 신뢰

### 입력 (피처) 설계

```python
features = {
    # 1. 현재값 (4가지 핵심 + 보조)
    'fabstorage_now':       fabstorage[-1],
    '1min_now':             avgtotal1min[-1],
    'm14b_oht_util_now':     m14b_oht_util[-1],
    'rev_count_now':        리프터_역증가_수,

    # 2. 슬라이딩 통계 (5/10/30분)
    'fabstorage_max_30m':   max(fabstorage[-30:]),
    'fabstorage_mean_30m':  mean(fabstorage[-30:]),
    '1min_max_5m':          max(avgtotal1min[-5:]),
    '1min_std_30m':         std(avgtotal1min[-30:]),

    # 3. 변화율 (모멘텀) — 사전 신호의 핵심 ★
    'fabstorage_delta_5m':  fabstorage[-1] - fabstorage[-6],
    'fabstorage_delta_30m': fabstorage[-1] - fabstorage[-30],
    '1min_delta_10m':       avgtotal1min[-1] - avgtotal1min[-11],
    'm14_delta_30m':        m14_to_m16[-1] - m14_to_m16[-31],

    # 4. 가속도 (변화의 변화)
    'fabstorage_accel':     (fab[-1]-fab[-6]) - (fab[-6]-fab[-11]),

    # 5. 룰 발동 flag (보조 신호)
    'rule_s1': int(s1_now), 'rule_s2': int(s2_now),

    # 6. 조합 피처
    '1min_x_fabstorage':    avgtotal1min[-1] * fabstorage[-1],
}
```

**총 30~50개 피처**.

### 라벨 (정답) 만들기

```python
# 핵심: "30분 뒤" 라벨링 (사전 예측용)
def make_label(t, verified_incidents):
    future = t + timedelta(minutes=30)
    for inc_start, inc_end in verified_incidents:
        if inc_start <= future <= inc_end:
            return 1  # 30분 뒤 사건 진행 중
    return 0
```

5/7 사건(07:32~07:38) 라벨 결과:
| 시각 | 30분 뒤 시각 | 라벨 |
|---|---|---:|
| 06:30 | 07:00 | 0 |
| 06:35 | 07:05 | 0 |
| **07:02** | **07:32** | **1** ★ (사건 시작 시점) |
| 07:08 | 07:38 | 1 |
| 07:09 | 07:39 | 0 (사건 끝남) |

→ 모델이 **07:02 시점 데이터의 특징을 학습**하여 "30분 뒤 정체 발생" 신호 감지.

---

## 4. 사전경보의 가치 (왜 30분인가)

### 시간대별 대응 가능 행동

| 사전 시간 | 가능한 대응 |
|---|---|
| **45분 전** | 사람 인원 배치, 우회 경로 준비 |
| **30분 전 ★** | 운영자 알람, 모니터링 강화 |
| **15분 전** | 빠른 임시 조치 |
| 0분 (룰 발동) | 사후 대응 (이미 큐 쌓임) |
| -30분 (보통) | 회복 작업 |

### 5/7 시뮬레이션 비교

```
[룰만 사용]
  07:31 운영자 첫 인지
  07:32 룰 발동
  07:35 대응 시작
  08:00 회복 (25분 소요)
  → 정체 지속 30분

[룰 + ML 사전경보]
  07:02 ML 사전경보 (확률 0.78)
  07:05 운영자 모니터링 강화
  07:15 사전 조치 (포트 close/open 미리)
  07:32 룰 발동했지만 이미 회복 중
  07:40 회복 (8분 소요)
  → 정체 지속 8분 (★ 70% 단축)
```

---

## 5. 만드는 과정 (단계별)

### Phase 1 — 데이터 준비 (1~2일)

```bash
# 검증된 사건 4건 라벨링
verified_incidents = [
    ('2026-04-21 13:50', '2026-04-21 14:30'),
    ('2026-04-21 14:12', '2026-04-21 14:16'),
    ('2026-05-06 09:35', '2026-05-06 10:10'),
    ('2026-05-07 07:32', '2026-05-07 07:38'),
]

# 90min.csv → 피처 + 라벨 데이터프레임
df = build_features('90min.csv')
df['label'] = df['timestamp'].apply(lambda t: make_label(t, verified_incidents))
```

### Phase 2 — 모델 학습 (3~5일)

```python
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit

X = df.drop(columns=['label', 'timestamp'])
y = df['label']

# 시계열 분할 (랜덤 X)
tscv = TimeSeriesSplit(n_splits=5)

model = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    scale_pos_weight=30,    # 불균형 처리
    eval_metric='aucpr',
)

for train_idx, val_idx in tscv.split(X):
    model.fit(
        X.iloc[train_idx], y.iloc[train_idx],
        eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
        early_stopping_rounds=20,
    )

# 검증 기준
# - PR-AUC > 0.7
# - 사전경보 시점이 사건 시작 25~35분 전
```

### Phase 3 — 기존 스크립트 통합 (3~5일)

```python
# 3단계_룰베이스_사건단위.py 수정
def evaluate_rules_and_ml(t1_window, m14_window, lft_window, v3_window):
    # 기존 룰 평가
    s1, s2, s3, ctx = evaluate_rules(...)

    # 신규 ML 평가
    features = build_features_realtime(t1_window, ...)
    ml_score = ml_model.predict_proba([features])[0][1]
    ml_level = classify_ml_level(ml_score)

    # 결합
    final = combine(s3, ml_score)

    ctx.update({
        'ml_score': ml_score,
        'ml_level': ml_level,
        'final_decision': final,
    })
    return s1, s2, s3, ctx
```

**발동이벤트 CSV 컬럼 추가**:
```
기존: stage, reason, relation
신규: ml_score (0~1), ml_level (정상/주의/경보), final_decision (룰+ML 결합)
```

### Phase 4 — 운영 + 재학습 (지속)

```
주 1회:  새 사건 라벨 추가 (운영자 확인)
월 1회:  XGBoost 재학습
3달 1회: 피처 중요도 재분석 → 새 피처 추가
```

---

## 6. 필요 데이터

| 시점 | 데이터 양 | 학습 효과 |
|---|---|---|
| 지금 (1주 미만) | 사건 4건 | △ 작동만 (불안정) |
| 1주 (5/8~5/14) | 사건 추가 가능 | ○ 5/7형 정체 학습 |
| **1달** | **사건 10~20건 예상** | **★ 실용 시작** |
| 3달 | 사건 30~50건 | ★★ 안정 |
| 6달~1년 | 50~100건 + 계절성 | ★★★ 정밀 |

**최소 시작 조건**: 1달 데이터 (~30개 사건 + 정상 패턴 풍부)

---

## 7. 예상 효과

### 정량적

| 지표 | 룰만 | 룰 + ML | 개선 |
|---|---:|---:|---|
| 사건 인지 시점 | 사건 시작 0분 | **사건 시작 30분 전** | **+30분** |
| 정체 지속 시간 | ~30분 | ~10분 | **-65%** |
| 운영 대응 여유 | 사후 | **사전** | **질적 변화** |
| 위양성률 | 낮음 | 낮음 (룰 안전망) | 유지 |
| 검출률 (Recall) | 70% | 85~90% | **+15~20%** |

### 정성적

- 운영자가 모니터 안 보고 있어도 시스템이 알려줌
- 사건 발생 전 조치 → 큐 누적 방지
- 새로운 정체 패턴(처음 보는 형태)도 학습 → 자동 적응
- 데이터 누적될수록 정확도 향상

---

## 8. 리스크 & 완화

| 리스크 | 완화 |
|---|---|
| ML 블랙박스 → 운영자 불신 | SHAP 값으로 "왜 위험" 설명 표시 |
| 학습 데이터 부족 → 위양성 | 룰을 안전망으로 항상 병행 |
| 데이터 드리프트 (시간 지나면 패턴 변함) | 월 1회 재학습 |
| 새 정체 패턴 (학습 안 됨) | 룰 + Isolation Forest 보조 |

---

## 9. 도입 결정 체크리스트

- [ ] 사건 라벨 누적할 사람/프로세스 있나? (월 10건 이상 확인)
- [ ] 정상 데이터 1달치 (~43,200 행) 수집 가능한가?
- [ ] Python + XGBoost 환경 준비 가능?
- [ ] 운영자에게 "확률 점수" 개념 설명 가능?
- [ ] 재학습 주기 합의됐나? (월 1회 권장)

5개 모두 ○이면 도입 진행 가능.

---

## 10. 결론

| 단계 | 가치 |
|---|---|
| **룰베이스 (현재)** | 명확한 정체 보장 검출 |
| **+ ML 사전경보 ★** | 사건 30분 전 알람, 운영 대응 시간 확보 |

**ML은 룰을 대체하지 않고 보강.**
**둘이 같이 가야 정체 시작 전에 막을 수 있음.**

---

> 작성일: 2026-05-08
> 다음 액션: 5/8 ~ 5/14 정상 데이터 수집 → 1주일 학습 시도
