# 하네스(Harness) 기법 — LLM 스킬 평가 프레임워크

## 개요

하네스(Harness) 기법은 LLM 스킬의 품질을 **체계적으로 측정·개선**하기 위한 평가 프레임워크다.
핵심은 "Draft → Test → Evaluate → Improve" 반복 루프이며, 정량적 벤치마크와 정성적 인간 리뷰를 결합한다.

> "Today's LLMs are smart. They have good theory of mind and when given a good harness can go beyond rote instructions and really make things happen."
> — Skill Creator SKILL.md

---

## 1. 핵심 루프 (Core Loop)

```
┌─────────────┐
│  1. 의도 파악  │  ← 사용자 인터뷰, 엣지 케이스 정리
└──────┬──────┘
       ▼
┌─────────────┐
│ 2. 스킬 작성  │  ← SKILL.md 초안 + 번들 리소스
└──────┬──────┘
       ▼
┌─────────────┐
│ 3. 테스트 실행 │  ← with_skill + baseline (동시 스폰)
└──────┬──────┘
       ▼
┌─────────────┐
│ 4. 평가/채점  │  ← 정량(assertions) + 정성(human review)
└──────┬──────┘
       ▼
┌─────────────┐
│ 5. 스킬 개선  │  ← 피드백 반영, 일반화, why 설명
└──────┬──────┘
       │
       └──→ 만족할 때까지 3~5 반복
```

---

## 2. 디렉터리 구조

```
skill-name/
├── SKILL.md                    # 스킬 본체 (name, description, 지시사항)
├── evals/
│   └── evals.json              # 테스트 케이스 정의
├── scripts/                    # 번들 스크립트
├── references/                 # 참조 문서
└── assets/                     # 템플릿, 폰트, 아이콘

skill-name-workspace/
├── iteration-1/
│   ├── eval-0-descriptive-name/
│   │   ├── with_skill/
│   │   │   ├── outputs/        # 스킬 사용 결과물
│   │   │   ├── grading.json    # 채점 결과
│   │   │   └── timing.json     # 실행 시간/토큰
│   │   └── without_skill/
│   │       └── outputs/        # 베이스라인 결과물
│   ├── benchmark.json          # 벤치마크 집계
│   └── benchmark.md
├── iteration-2/
│   └── ...
└── feedback.json               # 사용자 리뷰
```

---

## 3. 평가 JSON 스키마

### 3.1 evals.json — 테스트 케이스 정의

```json
{
  "skill_name": "my-skill",
  "evals": [
    {
      "id": 1,
      "prompt": "사용자가 입력할 실제 프롬프트",
      "expected_output": "기대 결과 설명",
      "files": ["evals/files/sample.pdf"],
      "expectations": [
        "출력에 X가 포함됨",
        "Y 스크립트를 사용함"
      ]
    }
  ]
}
```

### 3.2 grading.json — 채점 결과

```json
{
  "expectations": [
    {
      "text": "assertion 내용",
      "passed": true,
      "evidence": "근거 (트랜스크립트에서 발견)"
    }
  ],
  "summary": {
    "passed": 2, "failed": 1, "total": 3, "pass_rate": 0.67
  },
  "execution_metrics": {
    "tool_calls": {"Read": 5, "Write": 2, "Bash": 8},
    "total_tool_calls": 15,
    "errors_encountered": 0
  }
}
```

### 3.3 benchmark.json — 벤치마크 집계

```json
{
  "metadata": {
    "skill_name": "my-skill",
    "runs_per_configuration": 3
  },
  "run_summary": {
    "with_skill": {
      "pass_rate": {"mean": 0.85, "stddev": 0.05},
      "time_seconds": {"mean": 45.0, "stddev": 12.0},
      "tokens": {"mean": 3800, "stddev": 400}
    },
    "without_skill": {
      "pass_rate": {"mean": 0.35, "stddev": 0.08},
      "time_seconds": {"mean": 32.0, "stddev": 8.0},
      "tokens": {"mean": 2100, "stddev": 300}
    },
    "delta": {
      "pass_rate": "+0.50",
      "time_seconds": "+13.0"
    }
  }
}
```

---

## 4. 실행 단계 상세

### Step 1: 테스트 실행 (Spawn)

- **with_skill**: 스킬을 적용한 상태에서 테스트 프롬프트 실행
- **baseline**: 스킬 없이 동일 프롬프트 실행 (신규 스킬) 또는 이전 버전 스킬로 실행 (개선 중)
- **동시 스폰**: 두 버전을 동시에 실행해 대기 시간 최소화

### Step 2: Assertions 작성

실행 중 대기하면서 정량적 assertion 초안 작성:

| 유형 | 예시 |
|------|------|
| 파일 존재 | "PDF 파일이 생성됨" |
| 콘텐츠 검증 | "출력에 'John Smith' 포함" |
| 구조 검증 | "셀 B10에 SUM 수식 존재" |
| 형식 검증 | "Markdown 헤더가 3단계" |

> **팁**: 객관적으로 검증 가능한 것만 assertion으로. 주관적 품질(문체, 디자인)은 정성 리뷰로.

### Step 3: 타이밍 캡처

```json
{
  "total_tokens": 84852,
  "duration_ms": 23332,
  "total_duration_seconds": 23.3
}
```

### Step 4: 채점 + 벤치마크 + 뷰어

1. **Grader Agent** → grading.json 생성
2. **aggregate_benchmark.py** → benchmark.json/md 생성
3. **generate_review.py** → HTML 뷰어 실행 → 사용자 리뷰

### Step 5: 피드백 수집

```json
{
  "reviews": [
    {"run_id": "eval-0-with_skill", "feedback": "축 라벨 누락"},
    {"run_id": "eval-1-with_skill", "feedback": ""}
  ],
  "status": "complete"
}
```

빈 피드백 = 만족. 구체적 불만 사항에 집중하여 개선.

---

## 5. 고급 기법

### 5.1 Blind Comparison (A/B 비교)

- 독립 에이전트가 두 출력을 익명으로 평가
- `comparison.json`: winner/loser + rubric(5점 척도) + 강점/약점
- `analysis.json`: 트랜스크립트 패턴 분석 + 개선 제안

### 5.2 Description Optimization (트리거 최적화)

스킬이 올바른 쿼리에서 올바르게 트리거되는지 최적화:

1. **Eval Set 생성**: should_trigger(8~10개) + should_not_trigger(8~10개) 쿼리
2. **사용자 검토**: HTML 에디터로 쿼리 수정/추가/삭제
3. **최적화 루프**: train 60% / test 40% 분리 → 반복 개선 (최대 5회)
4. **결과 적용**: test score 기준 best_description 선택

---

## 6. 개선 원칙 (Improvement Philosophy)

| 원칙 | 설명 |
|------|------|
| **일반화** | 테스트 케이스에 오버핏하지 말고, 범용적 개선에 집중 |
| **린 프롬프트** | 효과 없는 지시 제거, 트랜스크립트에서 비생산적 패턴 확인 |
| **WHY 설명** | ALWAYS/NEVER 대신 이유를 설명 → 모델이 맥락을 이해 |
| **반복 작업 번들링** | 테스트 케이스마다 동일 스크립트를 만들면 → scripts/에 번들 |
| **놀라움 없는 원칙** | 스킬 내용이 사용자 의도와 다르면 안 됨 |

---

## 7. 환경별 차이

| 기능 | Claude Code | Claude.ai | Cowork |
|------|------------|-----------|--------|
| 서브에이전트 | ✅ 병렬 실행 | ❌ 순차 실행 | ✅ 병렬 (타임아웃 시 순차) |
| 브라우저 뷰어 | ✅ 서버 모드 | ❌ 직접 대화 내 표시 | ❌ --static HTML |
| 베이스라인 비교 | ✅ | ❌ (무의미) | ✅ |
| 정량 벤치마크 | ✅ | ❌ | ✅ |
| Description 최적화 | ✅ (claude -p) | ❌ | ✅ |
| 패키징 (.skill) | ✅ | ✅ | ✅ |

---

## 요약

하네스 기법의 본질은 **"측정 없이 개선 없다"**이다.

1. 스킬을 작성하고
2. 구조화된 테스트로 실행하고
3. 정량 + 정성 양면에서 평가하고
4. 피드백을 반영해 개선하고
5. 만족할 때까지 반복한다

이 프레임워크를 통해 LLM 스킬을 **재현 가능하고 체계적으로** 품질 관리할 수 있다.
