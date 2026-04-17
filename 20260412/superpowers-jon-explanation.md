# Superpowers - Jon Edition 설명서

obra/superpowers 14개 스킬 중 SK Hynix ASAS 환경에 맞게 선별한 **8개 코어 스킬**에 대한 설명과 적용 방법.

---

## 목차

1. [왜 이 팩인가](#왜-이-팩인가)
2. [스킬 개요 (한눈에)](#스킬-개요-한눈에)
3. [각 스킬 상세](#각-스킬-상세)
   - [1. using-superpowers](#1-using-superpowers-메타-스킬)
   - [2. brainstorming](#2-brainstorming-설계-세션)
   - [3. writing-plans](#3-writing-plans-구현-계획)
   - [4. test-driven-development](#4-test-driven-development-tdd)
   - [5. systematic-debugging](#5-systematic-debugging-체계적-디버깅)
   - [6. verification-before-completion](#6-verification-before-completion-완료-전-검증)
   - [7. writing-skills](#7-writing-skills-스킬-작성법)
   - [8. requesting-code-review](#8-requesting-code-review-코드-리뷰)
4. [전체 워크플로우](#전체-워크플로우)
5. [설치 및 적용 방법](#설치-및-적용-방법)
6. [스크립트](#스크립트-선택)
7. [제외한 스킬](#제외한-스킬-6개)

---

## 왜 이 팩인가

obra/superpowers는 Jesse Vincent(Anthropic 출신)가 만든 152k star 급 에이전트 스킬 프레임워크. 원본은 14개인데, 그중:

- SK Hynix **폐쇄망 환경**에 맞고
- Qwen3-235B **단일 서빙** (병렬 서브에이전트 불가) 환경에 맞고
- **한국어 우선** 환경에 맞는 것

만 골라서 8개로 재구성. 안 맞는 6개는 제외.

---

## 스킬 개요 (한눈에)

| # | 스킬 | 역할 | 언제 발동 |
|---|------|------|----------|
| 1 | using-superpowers | 스킬 관리자 (메타) | 매 응답 전 |
| 2 | brainstorming | 설계 전 질문 | "만들자", "설계" |
| 3 | writing-plans | 태스크 쪼개기 | 설계 끝난 후 |
| 4 | test-driven-development | RED→GREEN→REFACTOR | 구현 중 |
| 5 | systematic-debugging | 4단계 루트코즈 | 버그/에러 |
| 6 | verification-before-completion | 완료 전 증거 | "끝났어?" |
| 7 | writing-skills | 스킬 작성법 | 새 스킬 만들 때 |
| 8 | requesting-code-review | 리뷰 체크리스트 | PR 직전 |

---

## 각 스킬 상세

### 1. using-superpowers (메타 스킬)

**역할:** 다른 7개 스킬이 제대로 발동되게 만드는 "스킬의 관리자". 진입점.

**핵심 원칙:**
사용자 메시지 받으면 응답 전에 반드시 자문 — **"이 요청에 어떤 스킬이 맞나?"** 1%라도 맞으면 무조건 invoke. "간단하니까 스킬 없이" 판단 금지.

**우선순위 룰:**
```
사용자 명시 지시 (CLAUDE.md) > 스킬 > 기본 동작
```
CLAUDE.md에 "TDD 하지마"라고 써 있으면 TDD 스킬보다 사용자 지시가 이김.

**응답 프로토콜 (5단계):**
1. 메시지 받음
2. 스킬 매칭 체크
3. "Using X skill to Y" 선언
4. 스킬 내용 그대로 따름 (체크리스트 있으면 복제)
5. 응답 생성

**한국어 트리거 매핑:**
- "왜 안 되지", "디버깅" → systematic-debugging
- "끝났어?", "배포해도 돼?" → verification-before-completion
- "만들자", "설계" → brainstorming
- "스킬 만들자" → writing-skills
- "리뷰 좀" → requesting-code-review

**적용 방법:**
- ASAS 플랫폼 session-start 프롬프트에 이 스킬 내용 주입
- Qwen3가 매 응답 시작 시 체크하도록 유도

---

### 2. brainstorming (설계 세션)

**역할:** 사용자가 "X 만들어줘" 해도 **바로 코드 안 쓰고** Socratic 질문으로 진짜 요구사항 뽑아내기.

**4단계 프로세스:**

**Phase 1 — 현재 상황 파악 (질문 5개):**
1. 목표: "해결하려는 진짜 문제가 뭔가요?"
2. 사용자: "누가, 언제, 어떤 상황에서 쓰나요?"
3. 기존 자산: "재사용 가능한 것 있나요?" (ASAS 355 스킬, V7 모델 등)
4. 제약: "폐쇄망/성능/마감/승인 중 걸리는 게?"
5. 성공 기준: "어떤 상태가 되면 끝났다고 할 수 있나요?"

**Phase 2 — 대안 최소 2개 제시 (트레이드오프 명시):**
```
안 A: [접근법] — 장점 / 단점 / 공수
안 B: [다른 접근법] — 장점 / 단점 / 공수
추천: A (이유: ...)
```

**Phase 3 — 섹션 단위 합의:**
설계 문서 한방에 투척 금지. 아키텍처 → 데이터 모델 → 컴포넌트 → 테스트 전략 → 배포 순서로 하나씩 확인.

**Phase 4 — 설계 문서 저장:**
`docs/plans/YYYY-MM-DD-<feature>.md`에 저장. 다음 스킬(writing-plans)의 입력.

**언제 skip:** 버그 수정, 이미 스펙 명확한 단순 태스크.

**적용 방법:**
- 막연한 아이디어 받을 때 바로 코드 시작하지 않고 질문 3개 이상 던지는 걸로 시작
- 설계 문서 템플릿 ASAS 문서함에 고정 저장

---

### 3. writing-plans (구현 계획)

**역할:** 설계 문서를 **2-5분 단위 작은 태스크**로 쪼개기. "주니어 개발자가 코드베이스 전혀 모른다"는 가정.

**각 태스크에 반드시 포함:**
- 정확한 파일 경로 (수정이면 라인 번호까지)
- 완전한 코드 블록 (pseudo 금지, TODO 금지)
- 검증 커맨드 + 예상 출력
- TDD 순서 (실패 테스트 → 실행 → 구현 → 통과)
- 예상 2-5분 분량

**금지 표현 (하나라도 있으면 계획 실패):**
| 표현 | 왜 금지 |
|------|--------|
| "TBD", "추후 결정" | 결정 미룸 |
| "적절한 에러 처리" | 뭐가 적절한지 불명 |
| "관련 테스트 작성" | 실제 테스트 코드 없음 |
| "Task N과 비슷하게" | 순서 뒤섞여 읽힐 수 있음 |
| "에지 케이스 처리" | 어떤 엣지? |
| "최적화" | 뭘 어떻게? |

**Self-review (완성 후):**
1. 스펙 커버리지 — 요구사항마다 태스크 매핑됐나?
2. 플레이스홀더 스캔 — TBD/TODO 단어 검색
3. 태스크 독립성 — 주니어가 읽고 이해 가능?
4. 검증 단계 있나
5. 2-5분 범위 지켰나

**적용 방법:**
- brainstorming 끝난 설계 문서 → 이 스킬 발동 → 계획 파일 생성
- 계획 파일은 체크박스 `- [ ]` 로 진행 추적

---

### 4. test-driven-development (TDD)

**역할:** 구현할 때 **테스트 먼저, 구현 나중에**. 어기면 삭제 후 재시작.

**아이언 룰:**
> "테스트 없는 프로덕션 코드는 존재할 수 없다. 테스트보다 먼저 쓴 코드는 **삭제하고** 다시 시작한다. 적응시키지 않는다."

**사이클 (RED-GREEN-REFACTOR):**

**RED:**
1. 테스트 1개 작성 (여러 개 X)
2. **실제 실행해서 FAIL 확인** — 이거 안 하면 TDD 아님
3. FAIL 메시지가 "함수 없음" 같은 기본 에러면 OK

**GREEN:**
1. 통과시키는 **최소한** 코드만 (하드코딩도 OK)
2. 실행 → PASS 확인
3. 전체 테스트 여전히 통과하는지 확인

**REFACTOR:**
1. 중복 제거, 이름 개선
2. 각 변경 후 테스트 재실행
3. 커밋

**절대 금지 4가지:**
1. 구현 먼저 → 테스트 나중
2. "간단하니까 테스트 생략"
3. 테스트 실행 안 해보고 구현 작성
4. 테스트 5개 한 번에 쓰고 구현

**적용 방법:**
- Python: `pytest` 기반
- Java (AMHS): JUnit 5
- Logpresso: 쿼리 결과를 fixture로 snapshot test
- EDA/3D 시각화 같이 TDD 어려운 건 예외 허용 (verification 스킬이 대신)

---

### 5. systematic-debugging (체계적 디버깅)

**역할:** 버그 났을 때 **루트코즈 조사 끝나기 전엔 수정 금지.**

**4단계 (건너뛰기 금지):**

**Phase 1 — Investigation:**
- 에러 메시지 완독 (스택 맨 아래 "Caused by"까지)
- 100% 재현 확인
- `git log --oneline -20`, 최근 변경 확인
- 데이터 흐름 레이어별 로그 삽입 (L1→L2→L3→L4)

**Phase 2 — Pattern Analysis:**
- 같은 버그 다른 데서도 나나? (`grep -r`)
- 리그레션인가, 처음부터 버그였나?

**Phase 3 — Hypothesis Testing:**
- 가설 1개 세움
- 검증 방법 설계
- 증거 수집
- 가설 맞으면 → Phase 4, 틀리면 → Phase 1 복귀

**Phase 4 — Implementation:**
- 실패 테스트 먼저 작성 (TDD)
- 최소 수정
- 테스트 통과 확인
- Defense-in-depth 고려

**Phase 4.5 (3번 실패 시):**
같은 버그 수정 시도 3번 실패하면 **STOP.** 가설이 틀린 게 아니라 **아키텍처가 틀린 것**. 사람과 대화.

**금지 패턴 (보이면 Phase 1 복귀):**
- "일단 try/except로 감싸자" — 증상만 숨김
- "재시도 로직 추가" — 원인 안 고침
- "타임아웃 늘리자" — 왜 느린지 모름
- "restart하면 되던데" — 재현 못 한 것
- "캐시 지우면 됨" — 캐시 왜 꼬였는지 모름

**적용 방법:**
- 디버깅 로그 `docs/debug/YYYY-MM-DD-<bug>.md`에 4단계 프로세스 기록
- Logpresso 쿼리 실패 시 특화 체크포인트 (datestr/fulltext/stats 규칙 위반 확인)
- V7 모델 예측 이상 시 특화 체크 (feature 개수, GPU fallback, threshold)

---

### 6. verification-before-completion (완료 전 검증)

**역할:** "완료" 선언 전에 **실제로 작동하는지 증거** 확보.

**금지 문장 (증거 없이 이거 말하면 위반):**
- "구현 완료"
- "테스트 다 통과해요"
- "배포 준비 됐어요"
- "작동할 겁니다"
- "아마 될 거예요"

**검증 체크리스트 (모두 실제 실행):**
1. 빌드/컴파일 성공
2. 유닛 테스트 통과
3. **엔드-투-엔드 실제 실행** (로컬 테스트 ≠ 검증)
4. 역기능 테스트 (실패해야 하는 케이스 확인)
5. 기존 기능 회귀 없음
6. 로그에 ERROR/WARNING 없음

**증거 포맷 (표):**
| # | 커맨드 | 결과 | 증거 |
|---|--------|------|------|
| 1 | `pytest -v` | 47 passed | (출력) |
| 2 | `curl /predict` | 200 OK | (로그) |

**Rationalization Table (유혹 차단):**
- "간단한 변경이니까" → 간단한 변경이 프로덕션 다 터뜨린다
- "시간 없어" → 검증 안 하고 버그 나면 더 걸린다
- "로컬에서 돌아갔어" → 로컬 ≠ 프로덕션
- "테스트 통과했어" → 테스트는 의도 검증. 실제 동작 검증 아님

**적용 방법:**
- "완료"라고 말하기 전 체크리스트 TodoWrite로 복제
- 증거 파일은 `docs/verify/YYYY-MM-DD.md` 에 저장
- V7 모델 / Logpresso 쿼리 / OHT 시뮬레이터 / Java AMHS 각 도메인별 체크포인트 따로

---

### 7. writing-skills (스킬 작성법)

**역할:** 새 SKILL.md 만들 때 지켜야 할 규칙. **"스킬 작성도 TDD다."**

**사이클:**
- **RED:** 스킬 없이 시나리오 돌려서 실패 관찰
- **GREEN:** 그 실패 막는 최소 스킬 작성
- **REFACTOR:** loophole(합리화) 닫기

**YAML frontmatter 규칙:**
```yaml
---
name: skill-identifier        # 64자 이하, kebab-case, 동명사
description: Use when ...     # 1024자 이하, 3인칭, 트리거 명시
---
```

**description 필수 요건:**
- **3인칭** (system prompt 주입됨)
- "Use when..." 으로 시작
- 무엇을 + 언제 쓸지 둘 다
- 한국어 트리거 단어 포함 (존님 환경)

**description 좋은 예 vs 나쁜 예:**

❌ 나쁨: `description: For async testing` (너무 추상적)

❌ 나쁨: `description: I can help with V7` (1인칭)

✅ 좋음:
```yaml
description: Use when working with V7 surge prediction model - retraining, threshold tuning (280 default), 11-feature engineering. Covers XGBoost 30min-to-10min prediction, encoding auto-detect. Use when user says "V7 재학습", "surge 예측".
```

**Progressive Disclosure (3단계 로딩):**
| 레벨 | 언제 로드 | 크기 |
|------|----------|------|
| Metadata (name + description) | 항상 | ~100 단어 |
| SKILL.md 본문 | 트리거 시 | **500줄 이하** |
| references/*.md | 필요 시 | 무제한 |

500줄 넘으면 references/로 쪼개라.

**적용 방법:**
- ASAS 355 스킬 품질 일관성 유지에 핵심
- 새 스킬 만들기 전 이 스킬 먼저 읽고 템플릿대로 작성
- Skill Harness (Draft→Test→Evaluate→Improve) 평가와 세트

---

### 8. requesting-code-review (코드 리뷰)

**역할:** 리뷰 요청 전 self-check, 리뷰 받았을 때 수용 방법.

**모드 A — 리뷰 요청 전 Self-Checklist:**

혼자 먼저 체크. 여기서 걸리는 거 리뷰어한테 가져가면 시간 낭비.

- **정확성:** 테스트 통과, 엣지 케이스 3개 이상, 기존 테스트 회귀 없음
- **보안:** 하드코딩 비밀 X, SQL 인젝션 방지, 민감 데이터 로그 X
- **비즈 룰 (AMHS):** OHT 경로 가중치, HID 할당 룰, V7 threshold 근거
- **입력 검증:** 인코딩 fallback, 빈 DataFrame 처리
- **성능:** N+1 X, 큰 loop 안 DB콜 X

**리뷰 요청 메시지 포맷 (4가지 필수):**
```markdown
## 리뷰 요청: [제목]

### 배경
- 해결하려는 문제
- 관련 이슈

### 변경 범위
- 파일: N개
- LOC: +X -Y

### 핵심 결정 (리뷰어 확인 필요)
1. 왜 A를 선택했나
2. 대안 B를 버린 이유

### 걱정되는 부분
- [ ] 성능 영향?
- [ ] 호환성?
```

**모드 B — 리뷰 받았을 때:**

❌ 방어적 반응 금지:
- "그건 의도한 겁니다"
- "이미 그렇게 하고 있어요"

✅ 건설적 반응:
- "그 부분 놓쳤네요. 고칠게요."
- "의도는 X였는데, 대안 A/B 중 어느 게 나을까요?"

**피드백 3버킷 분류:**
| 버킷 | 정의 | 대응 |
|------|------|------|
| Must Fix | 버그/보안/회귀 | 즉시 수정 |
| Should Fix | 가독성/네이밍 | 수정 or 이슈 기록 |
| Discussion | 디자인 의견 차이 | 대화 후 결정 |

모든 피드백에 **명시적 응답** (반영 여부 + 이유).

**적용 방법:**
- PR 올리기 전 self-checklist 5분
- 리뷰 메시지 템플릿 ASAS에 고정
- FAB 도메인 특화 체크는 별도 "fab-code-review" 스킬 만들면 더 강력

---

## 전체 워크플로우

```
사용자 요청
    ↓
[1. using-superpowers]  ← 매 응답 전 스킬 매칭 체크
    ↓
──────────────────────────────
 신 기능 개발 플로우
──────────────────────────────
    ↓
[2. brainstorming]       설계 문서 생성
    ↓
[3. writing-plans]       2-5분 태스크 분해
    ↓
[4. test-driven-development]  구현 (RED→GREEN→REFACTOR)
    ↓
[5. systematic-debugging]  버그 시 (4단계)
    ↓
[6. verification-before-completion]  완료 전 증거
    ↓
[8. requesting-code-review]  리뷰
    ↓
배포

──────────────────────────────
 스킬 작성 플로우 (별도)
──────────────────────────────
[7. writing-skills]     새 스킬 만들 때
```

---

## 설치 및 적용 방법

### 방법 1: ASAS 플랫폼에 직접 심기 (권장)

```bash
# ASAS 스킬 디렉토리에 복사
cp -r superpowers-jon/skills/* /path/to/asas/skills/

# 각 스킬 SKILL.md의 YAML frontmatter가 ASAS 스킬 로더가 이해하는 포맷인지 확인
```

ASAS가 이미 스킬 디스커버리 로직 있으면 (기존 355개 스킬 로드하는 방식) 그대로 작동.

### 방법 2: Claude Code Plugin으로 사용

```bash
# 로컬 폴더를 플러그인으로 추가
/plugin marketplace add /path/to/superpowers-jon
/plugin install superpowers-jon
```

### 방법 3: 프롬프트에 인라인 주입 (간단 테스트용)

Qwen3나 Claude 사용 시 시스템 프롬프트에:
```
You have access to the following Superpowers skills. Read the one relevant to the user's request before responding.

[SKILL.md 파일들 내용 복붙]
```

### 권장 적용 순서

1. **using-superpowers** 먼저 적용 (나머지가 발동되게)
2. **systematic-debugging + verification-before-completion** 세트로 (디버깅/배포 품질)
3. **brainstorming + writing-plans + TDD** 세트로 (신 기능 개발)
4. **writing-skills** (새 스킬 만들 때)
5. **requesting-code-review** (리뷰 프로세스 정착 시)

한꺼번에 8개 다 적용하지 말고 **1주씩 순차 도입**해서 어색함 줄이기.

---

## 스크립트 (선택)

**필수 스크립트: 없음.** SKILL.md 8개만으로 작동.

### 있으면 편한 스크립트

**1. Skill Manifest Generator** (`scripts/build-manifest.py`)
```python
# skills/*/SKILL.md frontmatter 긁어서 manifest.json 생성
# ASAS 스킬 로더가 빠르게 index 로드 가능
```

**2. Skill Validator** (`scripts/validate-skills.py`)
```python
# 각 SKILL.md 체크:
# - YAML frontmatter 유효성
# - name/description 필수
# - description 1024자 이하
# - 본문 500줄 이하
# - "Use when..." 으로 시작하는지
```

**3. Session-start Hook** (`scripts/session-start.py`)
```python
# Qwen3 세션 시작 시 "너에게 스킬 있다" 프롬프트 주입
# using-superpowers.md 내용 요약 포함
```

### obra/superpowers에 있는 스크립트 (존님 환경에는 불필요)

- `hooks/` — Claude Code 세션 훅용. ASAS는 자체 메커니즘.
- `.claude-plugin/`, `.cursor-plugin/`, `.codex/` — 각 플랫폼 플러그인 매니페스트. 불필요.
- `commands/` — 슬래시 커맨드 (/brainstorm 등). ASAS에 라우팅 별도.
- `agents/` — 서브에이전트 프롬프트. 병렬 못 쓰는 환경이니 불필요.

**결론: 지금은 SKILL.md만 있어도 충분. 운영하면서 불편하면 위 3개 스크립트 추가.**

---

## 제외한 스킬 (6개)

obra/superpowers 원본 14개 중 뺀 것들과 이유:

| 스킬 | 제외 사유 |
|------|----------|
| `executing-plans` | writing-plans에 통합 (계획과 실행 분리 불필요) |
| `receiving-code-review` | requesting-code-review에 통합 |
| `using-git-worktrees` | SK Hynix 내부 Git 정책과 불일치 |
| `finishing-a-development-branch` | 동일 사유 |
| `dispatching-parallel-agents` | 병렬 서브에이전트 사용 불가 환경 |
| `subagent-driven-development` | 동일 사유 |

---

## 출처

- **원본:** https://github.com/obra/superpowers (Jesse Vincent, Prime Radiant, 152k★)
- **라이선스:** MIT
- **재구성:** SK Hynix ASAS 환경 맞춤 (2026-04)
