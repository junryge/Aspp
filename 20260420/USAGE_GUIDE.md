# Superpowers 사용 가이드 (ASAS)

> 8개 슈퍼파워 메타 스킬을 ASAS 환경에서 어떻게 쓰는지 실전 가이드.
> 개념 설명은 [`superpowers-jon-explanation.md`](./superpowers-jon-explanation.md), 본 문서는 "ASAS 에서 어떻게" 중심.

## 1. 한 줄 요약

**4개 슈퍼파워(분야 / 선택 / 조합 / 자동) × 8개 메타 스킬 = 모든 요청에 일관 품질.**

## 2. 용어 정리

| 구분 | 도메인 스킬 (381개) | 메타 스킬 / 슈퍼파워 (8개) |
|---|---|---|
| 성격 | 무엇을 할 수 있나(What) | 어떻게 해야 하나(How) |
| 예 | `biopython`, `pytorch`, `agent-python-pro` | `brainstorming`, `systematic-debugging` |
| 위치 | `scientific-skills/<이름>/` | `scientific-skills/<이름>/` (동일) |
| UI | 24개 도메인 카드에 분산 | 새 도메인 `⚡ 슈퍼파워` 카드 |
| 호환 | YAML(`name`,`description`,`model`,`color`,`tools`) | YAML(`name`,`description`) |

## 3. Superpower #1 — 분야 선택

사용자가 쿼리 보내기 전 UI 상단 "⚡ 슈퍼파워" 카드 포함 25개 도메인 중 고른다. 판단 기준:

```
데이터 파일 있나?     → data-ml
분자/단백질/유전자?   → bioinformatics / bio-databases
반도체 FAB 로그?      → domain-knowledge (logpresso-search)
코드 품질/리뷰?       → dev-tools
막연한 아이디어?      → ⚡ 슈퍼파워 (brainstorming 단독)
```

**슈퍼파워화 포인트**: 도메인 확실치 않으면 `brainstorming` 스킬만 켜고 "뭐부터 해야 할지 모르겠어"라고 입력 → 5개 질문으로 분야 확정.

## 4. Superpower #2 — 수동 스킬 선택

도메인 고른 뒤 내부 스킬 1~3개 토글.

- **메타 단독**: PR 올리기 전 `requesting-code-review` 만 켜고 "이 diff 체크 좀" → self-checklist 가동.
- **메타 + 도메인 1**: `systematic-debugging` + `logpresso-search` → 쿼리 실패 4단계 RCA.
- **최대 3개**: 토큰·라우팅 효율상 3개 초과 금지. (응답 속도 저하, 컨텍스트 낭비)

`/api/skills` 응답에 `superpowers` 도메인이 뜨면 설치 성공.

## 5. Superpower #3 — 스킬 조합

자동 조합 추천: `/api/auto-skills` → `harness_bridge.suggest_skill_combinations()` 가 선택된 도메인 스킬 + 의도 맞는 메타 1개를 페어링.

| 쿼리 의도 | 짝지어지는 메타 |
|---|---|
| "버그/에러/왜 안 되" | `systematic-debugging` |
| "설계/만들자/새 기능" | `brainstorming` |
| "TDD/테스트 먼저" | `test-driven-development` |
| "끝났어/배포/완료" | `verification-before-completion` |
| "리뷰/PR" | `requesting-code-review` |
| "스킬 만들/SKILL.md" | `writing-skills` |
| "계획/태스크/분해" | `writing-plans` |
| (기타) | `using-superpowers` |

### 교과서 조합 8선

| 조합 | 언제 쓰나 |
|---|---|
| `brainstorming` × `biopython` | 바이오인포 파이프라인 설계 대화 |
| `writing-plans` × `agent-python-pro` | 구현을 2-5분 태스크로 분해 |
| `test-driven-development` × `pydeseq2` | 분석 코드 TDD |
| `systematic-debugging` × `logpresso-search` | 쿼리 RCA 4단계 |
| `verification-before-completion` × `pptx` | 발표자료 제출 직전 검증 |
| `requesting-code-review` × `agent-code-reviewer` | PR self-check |
| `writing-skills` × `engineer-skill-creator` | 신규 스킬 TDD |
| `using-superpowers` 단독 | 기본 라우팅/모호 쿼리 |

## 6. Superpower #4 — 자동 스킬 선택

`demos_v1/skills.py:context_aware_skill_select()` 파이프라인:

1. 현재 질문 키워드 매칭 (SKILL_KEYWORDS)
2. 최근 3턴 히스토리 보정
3. 쿼리 의도 부스트 (drawio 15, pptx 12, **메타 6–9**, llm 6, code 3)
4. 업로드 파일 확장자 매핑 (.csv → EDA, .py → python-pro ...)
5. (옵션) bge-reranker 재정렬
6. 컷오프: `max(3, top*0.3)` 미만 제외

### 한국어 메타 트리거 표

| 트리거 | 발동 메타 | 가중치 |
|---|---|---|
| 만들자, 설계, 브레인스토밍, 새 기능 | brainstorming | +8 |
| 계획, 태스크, 분해, 쪼개 | writing-plans | +6 |
| TDD, 테스트 먼저, 레드그린 | test-driven-development | +7 |
| 디버깅, 버그, 왜 안 되, 에러, 루트코즈 | systematic-debugging | +9 |
| 끝났어, 배포해도, 완료 선언, 증거 | verification-before-completion | +9 |
| 스킬 만들, SKILL.md, 스킬 작성 | writing-skills | +6 |
| 리뷰, PR, 코드리뷰, review 요청 | requesting-code-review | +6 |

`using-superpowers` 는 자동 발동 제외(모든 응답에 깔리는 메타 → 세션 고정).

### 오답 수정

자동 선택이 틀렸으면 UI 에서 직접 토글 해제·추가. 향후 `FeedbackStore` 로 학습 루프 닫는 PR 예정.

## 7. 사용 판단 가이드

```
쿼리 받음
  │
  ├─ "왜 안 되지", 에러 traceback → systematic-debugging
  ├─ "끝났어?", "배포해도" → verification-before-completion
  ├─ "리뷰 좀", PR → requesting-code-review
  ├─ "스킬 만들자", SKILL.md → writing-skills
  ├─ 모호한 아이디어 → brainstorming
  ├─ 설계 끝 → writing-plans → test-driven-development
  └─ 일반 코드/데이터 작업 → 도메인 스킬 (메타 자동 페어링)
```

### 금지 사항

- **메타 8개 동시 켜기** — 컨텍스트 폭발. 1~2개만.
- **`using-superpowers` 키워드 트리거** — 모든 쿼리에 붙어 토큰 낭비. 세션 고정 플래그로만 관리.
- **도메인 없이 메타만 켜고 "구현해줘"** — 메타는 행동 규칙이지 기능이 아님. 도메인 1개 필요.

## 8. 메타 스킬 8개 요약 카드

각 카드 원문은 `scientific-skills/<이름>/SKILL.md`.

### using-superpowers
- 역할: 모든 응답의 진입점. "어떤 스킬 발동할지" 자문 강제.
- 트리거: 세션 시작, 매 응답 전 (자동 아닌 프롬프트 고정).
- 궁합: 모든 도메인 스킬.
- Skip: 지시가 명시적이고 다른 스킬 이미 발동됐을 때.

### brainstorming
- 역할: 코드 쓰기 전 Socratic 질문 5개 → 대안 2개 → 섹션 합의.
- 트리거: "만들자", "설계", "새 기능", "어떻게".
- 궁합: `biopython`, `agent-python-pro`, `logpresso-search`.
- Skip: 버그 수정, 스펙 명확한 단순 태스크.

### writing-plans
- 역할: 설계 → 2-5분 태스크 분해 (파일/라인/검증 모두 명시).
- 트리거: "계획 짜자", "태스크로 쪼개줘".
- 궁합: `agent-python-pro`, `agent-api-designer`, `test-driven-development`.
- Skip: 계획 없이 바로 구현할 1줄 수정.

### test-driven-development
- 역할: RED→GREEN→REFACTOR 강제. 테스트 없는 코드 삭제 후 재시작.
- 트리거: "TDD", "테스트 먼저", "구현해줘".
- 궁합: `pytest`, `agent-python-pro`, `pydeseq2`.
- Skip: EDA notebook, 3D/시각화, Blender 플러그인 (judgment call).

### systematic-debugging
- 역할: 4단계 RCA(조사→패턴→가설→수정), 3회 실패 시 STOP + 아키텍처 대화.
- 트리거: "왜 안 되지", "디버깅", "버그", "에러".
- 궁합: `logpresso-search`, `agent-python-pro`, `debugging`.
- Skip: 명확한 오타·상수 수정.

### verification-before-completion
- 역할: "완료" 전 실행 증거 확보(빌드/유닛/E2E/역기능/회귀/로그).
- 트리거: "끝났어?", "배포해도 돼?", "완료".
- 궁합: `pytest`, `pptx`, `agent-devops-engineer`.
- Skip: 없음 (완료 선언 전 항상).

### writing-skills
- 역할: SKILL.md TDD. YAML frontmatter, description 작성법, Progressive Disclosure.
- 트리거: "스킬 만들자", "SKILL.md".
- 궁합: `brainstorming`, `engineer-skill-creator`, `asas-skill-authoring`(예정).
- Skip: 기존 스킬 typo 수정.

### requesting-code-review
- 역할: Self-checklist (정확성/보안/비즈룰/입력검증/성능) + 리뷰 요청 4섹션 포맷.
- 트리거: "리뷰 좀", "PR 올리기 전".
- 궁합: `agent-code-reviewer`, `owasp-security`, `github-ecosystem`.
- Skip: 혼자 쓰고 버릴 스크립트.

## 9. End-to-End 예시 3개

### 예1. V7 surge 모델 재학습

```
1. "V7 재학습 어떻게 하지"
   → brainstorming + ml 도메인 (질문: threshold 유지? feature 추가? 평가 CSV?)
2. "A안(threshold만 조정)으로 가자"
   → writing-plans (10개 2-5분 태스크 생성)
3. 각 태스크 실행
   → test-driven-development (RED→GREEN→REFACTOR)
4. "끝났어?"
   → verification-before-completion (pytest + 실제 CSV 평가)
5. "PR 올려줘"
   → requesting-code-review (self-check 5분)
```

### 예2. Logpresso 쿼리 RCA

```
1. "ts_data_view_m14a 쿼리가 빈 결과 반환"
   → systematic-debugging Phase 1 (에러 완독, 재현, git log)
2. Phase 2 (같은 패턴 grep)
3. Phase 3 (가설: table+search 조합 → 실제는 fulltext 필요)
4. Phase 4 (실패 테스트 작성 → fulltext 로 수정 → PASS)
```

### 예3. OHT 경로 최적화 스킬 신설

```
1. "OHT 경로 최적화 스킬 만들자"
   → writing-skills + brainstorming
2. RED: 기존 Claude 실패 시나리오 3개 관찰
3. GREEN: 최소 SKILL.md (아이언 룰 3개만)
4. REFACTOR: loophole 닫기
5. verification-before-completion (실제 쿼리 3개 재발동 테스트)
```

## 10. 통합 상태 체크리스트

- [x] `scientific-skills/` 에 8개 메타 스킬 폴더 존재
- [x] `DOMAIN_SKILLS["superpowers"]` 등록 (skills.py:497~)
- [x] `SKILL_DESC_KO` 에 8개 엔트리 (skills.py:197~)
- [x] `SKILL_KEYWORDS` 에 8개 엔트리 (skills.py:1334~)
- [x] `SKILL_GROUPS["meta-behavioral"]` 등록 (skills.py:686~)
- [x] `context_aware_skill_select()` 메타 트리거 블록 (skills.py:1437~)
- [x] `suggest_skill_combinations()` 도메인↔메타 페어링 (harness_bridge.py:273~)

스모크 통과(`_pick_meta_for_query` 8/8, 자동 트리거 6/6, 회귀 1/1, 페어링 3/3).

## 11. 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| `/api/skills` 에 `superpowers` 없음 | `DOMAIN_SKILLS` 삽입 실패 | `skills.py:497` 부근 dict 문법 확인 |
| 트리거 키워드 미발동 | 한국어 조합 변형 | `SKILL_KEYWORDS` 해당 엔트리에 변형 추가 |
| 메타만 뜨고 도메인 묻힘 | 가중치 과다 | `context_aware_skill_select()` 메타 가중치 재조정 |
| 조합 페어링 엉뚱 | `_pick_meta_for_query` 매칭 순서 | 구체 패턴(`스킬 만들`)을 일반(`만들자`) 앞에 유지 |
| SKILL.md 로드 안 됨 | 폴더명·파일명 불일치 | `scientific-skills/<name>/SKILL.md` 정확 확인 |

디버그 경로: `POST /api/harness/route {"query":"...", "limit":5}` → 라우터 원점수 확인.

## 12. 향후 확장

- `using-superpowers` 본문이 참조하는 ASAS 자체 스킬 작성 순서:
  1. `ml-model-retraining` (V7 재학습)
  2. `amhs-data-extraction` (Logpresso 쿼리 규칙)
  3. `closed-network-llm-integration` (Qwen3-235B 연동 패턴)
  4. `time-series-preprocessing` (CRT_TM 1분 정규화)
  5. `oht-simulator-migration` (React/3D)
  6. `fab-code-review` (도메인 특화 리뷰)
  7. `asas-skill-authoring` (ASAS 전용 스킬 작성 규칙)
- Session-start Hook (자동 `using-superpowers` 주입)
- Skill Validator 스크립트 (`scripts/validate-skills.py`)
- FeedbackStore → 자동 선택 학습 루프

## 13. 부록

- 원본: https://github.com/obra/superpowers (Jesse Vincent, MIT)
- 재구성: SK Hynix ASAS 환경 맞춤 (2026-04, `20260417_SKILL/`)
- 제외 6종: `executing-plans`, `receiving-code-review`, `using-git-worktrees`, `finishing-a-development-branch`, `dispatching-parallel-agents`, `subagent-driven-development` (사유는 [`superpowers-jon-explanation.md`](./superpowers-jon-explanation.md) §제외한 스킬 참조).
