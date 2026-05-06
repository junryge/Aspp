# Freedom 코딩 어시스턴트 사용 설명서

> 도메인 지식 + 코딩 스킬 + LLM 으로 **신규 프로그램을 완성**하는 사내 폐쇄망용 코딩 워크벤치

---

## 한눈에 보기

| 항목 | 내용 |
|------|------|
| 접속 주소 | `http://localhost:10010` |
| 실행 명령 | `python -m code_assist_v1.app_code` |
| 위치 | `code_assist_v1/` 폴더 (독립 플랫폼) |
| LLM 모델 | API 10개 + 로컬 GGUF (자동 인식) |
| 응답 방식 | SSE 토큰 스트리밍 (즉시 표시) |
| 도메인 지식 | BM25 검색 (수동 토글) |
| 코딩 스킬 | 69개 기본 + 무제한 직접 작성 (Capacity 스타일) |
| 컨텍스트 시스템 | 시스템 프롬프트 + 스킬 + 지식 + 첨부파일 자동 합성 |
| 하네스 | Tool 레지스트리·라우터·세션·피드백 운영 레이어 |

---

## 1. 빠른 시작 (3단계)

### 1단계 — 토큰 (선택)

```
code_assist_v1/TOKEN.TXT
```
파일 안에 사내 LLM API 키 한 줄. 비어있어도 GGUF 모델만으로 정상 동작.

### 2단계 — GGUF 모델 (선택)

```
code_assist_v1/MODEL_GGUF/
└── 원하는 모델.gguf
```
폴더에 `.gguf` 를 넣으면 기동 시 자동 인식됩니다. VRAM 안에서 가장 큰 모델이 자동 로드됩니다.

### 3단계 — 실행

```powershell
python -m code_assist_v1.app_code
```

브라우저에서 `http://localhost:10010` 접속.

---

## 2. 화면 구성

```
┌──── 좌 사이드바 ────┬──── 메인 채팅 ────┬── 우 워크스페이스 ──┐
│ [스킬][지식][세션] │  어시스턴트 응답  │  📁 첨부 파일 트리  │
│                    │  사용자 질문      │                    │
│  ＋ 검색           │  입력박스         │  클릭 → 미리보기   │
└────────────────────┴───────────────────┴────────────────────┘
```

### 상단바 버튼

| 버튼 | 기능 |
|------|------|
| 모델 드롭다운 | API 모델 + GGUF 통합 목록 |
| effort | 0(정확) ~ 3(창의), LLM 온도 매핑 |
| 🌙 / ☀️ | 다크/라이트 테마 토글 (브라우저에 저장) |
| ⚙️ 프롬프트 | 시스템 프롬프트 보기·편집 |
| 📖 문서 | 이 문서 |
| ＋ 세션 | 새 세션 시작 (메시지 비우기) |

### 입력박스 도구

| 도구 | 기능 |
|------|------|
| 📎 | 작업 파일 업로드 + 자동 첨부 |
| 📚 | 도메인 지식 검색 ON/OFF (수동 토글) |
| 전송 | Enter, 줄바꿈은 Shift+Enter |
| 중단 | Esc 또는 같은 버튼 |

---

## 3. 코딩 스킬

### 3.1 스킬이란?

**스킬은 LLM 의 작업 매뉴얼 한 장**입니다. `SKILL.md` 라는 마크다운에 위쪽엔 메타정보(이름·설명·태그), 아래엔 절차를 적어둡니다. 스킬을 활성화하면 그 본문이 시스템 프롬프트에 합쳐져, LLM 이 그 절차를 따라 작업합니다.

이 형식은 Anthropic 의 Claude Skills(Capacity 스타일) 와 동일합니다.

### 3.2 기본 제공 69개 스킬

첫 기동 시 코딩에 직접 도움 되는 스킬 69개가 자동으로 등록됩니다.

| 카테고리 | 예시 | 무엇을 하는가 |
|---------|------|--------------|
| 언어 전문가 | `agent-python-pro`, `agent-typescript-architect`, `agent-rust-engineer`, `agent-go-engineer` | 해당 언어 베스트 프랙티스대로 코드 작성·리뷰 |
| 프레임워크 | `agent-fastapi-architect`, `agent-react-pro`, `agent-django-pro`, `agent-vue-pro` | 프레임워크 표준 패턴으로 작업 |
| 백엔드/프론트/풀스택 | `agent-backend-developer`, `agent-frontend-developer`, `agent-fullstack-developer` | 도메인 전반 안내 |
| DevOps·인프라 | `agent-devops-engineer`, `agent-sre-engineer`, `agent-cloud-architect`, `agent-platform-engineer` | 배포·관측·인프라 |
| 데이터·ML | `agent-data-engineer`, `agent-ml-engineer`, `agent-ai-engineer` | 파이프라인·모델 |
| 보안·QA | `agent-security-engineer`, `agent-qa-engineer` | 보안 검토·테스트 |
| 아키텍처·리뷰 | `agent-code-reviewer`, `agent-architect-reviewer`, `agent-api-designer` | 설계·코드 리뷰 |
| 워크플로우 메타 | `writing-skills`, `writing-plans`, `brainstorming`, `systematic-debugging`, `test-driven-development`, `verification-before-completion`, `requesting-code-review`, `using-superpowers`, `skill-creator` | 작업 진행 방식 |
| 개발 도구 | `chrome-devtools`, `git`, `github-cli` 등 | 도구 활용 |
| 도메인 검색 | `knowledge-search` | 도메인 지식 BM25 검색 (수동) |

전체 목록은 화면 좌측 [스킬] 탭에서 확인할 수 있습니다.

### 3.3 스킬 활성화

1. 좌사이드 **[스킬]** 탭 클릭
2. 사용할 항목을 **클릭** → 좌측에 액센트바 + 입력박스 위에 🛠 칩 표시
3. 다시 클릭 → 비활성. 칩의 ✕ 클릭도 동일
4. 항목 **우클릭** → SKILL.md 본문 보기·편집·삭제

활성 스킬은 다음 채팅 요청 시 시스템 프롬프트에 본문 그대로 합쳐집니다.

### 3.4 새 스킬 직접 만들기

좌사이드 [스킬] 탭의 ＋ 버튼 → 모달에서 다음 정보를 입력:

```yaml
---
name: my-fastapi-skill
description: FastAPI 라우트와 미들웨어를 작성한다 (라우터 매칭 키워드 포함)
license: MIT
metadata:
  tags: [python, fastapi, backend]
---

# 절차

1. 첫 단계
2. 두 번째 단계
3. ...
```

저장하면 `code_assist_v1/skills/my-fastapi-skill/SKILL.md` 가 생성되고 즉시 사이드바에 등장합니다.

**작성 팁:**
- description 에는 라우팅 매칭용 키워드를 영어·한글 모두 포함
- 본문은 절차형 (1, 2, 3 단계)
- 4000자 이내 (자동 잘림)
- 환각 방지 한 줄 권장 (예: "데이터 컬럼명을 지어내지 말고 모르면 '확인 필요'로 표시")

---

## 4. 도메인 지식

### 4.1 빈 상태로 시작

`code_assist_v1/knowledge/` 폴더는 처음에 비어 있습니다. 사용자가 직접 등록합니다 (사용자별 분리 없이 평면 구조).

### 4.2 등록하는 3가지 방법

| 방법 | 절차 |
|------|------|
| 인-앱 작성 | [지식] 탭 → ＋ → 파일명 + 마크다운 본문 작성 → 저장 |
| 파일 업로드 | [지식] 탭 → 업로드 → `.md` `.txt` `.pdf` `.docx` 선택 → 텍스트 자동 추출 후 `.md` 로 저장 |
| 직접 복사 | 탐색기에서 `code_assist_v1/knowledge/` 에 `.md` 파일을 직접 복사 → 다음 검색 시 자동 인덱싱 |

### 4.3 권장 frontmatter

```markdown
---
title: M14 FAB 컬럼 정의서
tags: [m14, fab, column]
date: 2026-05-06
---

# 본문

| 컬럼 | 의미 |
|------|------|
| EQP_ID | 장비 ID |
```

### 4.4 검색·답변에 사용

1. 입력박스의 **📚 버튼** 클릭 → 그린 칩 `📚 도메인 지식 ON` 표시
2. 질문 입력 → 전송
3. 서버가 마지막 메시지로 BM25 검색 → 상위 8개 문서를 시스템 프롬프트에 주입 → LLM 이 출처를 명시하며 답변

> **자동 트리거 없음** — 📚 버튼이 켜져 있을 때만 검색이 동작합니다.

### 4.5 BM25 검색 알고리즘

- 한국어 2글자+ / 영문·숫자 2글자+ 토큰화
- BM25 (k1=1.5, b=0.75) + 파일명 부분일치 보너스 + 본문 부분일치 보너스
- 상위 점수의 30% 미만 결과 제외
- 파일 변경 자동 감지 (mtime)

---

## 5. 워크스페이스 (작업 파일 첨부)

작업 중인 코드를 LLM 에 함께 전달하려면:

1. 입력박스 **📎** → 다중 파일 선택 → 자동 업로드 + 자동 첨부 (보라 칩 `📄 main.py`)
2. 또는 우측 패널의 ＋ 버튼
3. 우측 트리에서 파일 클릭 → 미리보기. 우클릭 또는 미리보기 헤더의 📎첨부로 토글

**컨텍스트 주입 규칙:**
- 파일 1개당 최대 4000자
- 전체 합쳐서 최대 16000자
- 확장자에 따라 코드블록 언어 자동 감지

---

## 6. 시스템 프롬프트

상단바 **⚙️ 프롬프트** 클릭:

| 영역 | 설명 |
|------|------|
| 기본 시스템 프롬프트 (664자) | 코딩 어시스턴트 페르소나, 작업 원칙, 응답 형식 (읽기 전용) |
| 반-환각 가드 (176자) | 검증 게이트 (읽기 전용) |
| 사용자 추가 지시 | 자유 입력. 예: "답변은 항상 한국어로", "Python 3.11 문법 사용" |

**버튼:** 초기화 · 저장 (글자 수 토스트) · 닫기. 자동 저장도 동작 (입력 즉시 브라우저 localStorage).

---

## 7. 세션 관리

채팅 응답이 끝날 때마다 자동으로 세션이 저장됩니다.

좌사이드 **[세션]** 탭 행:
```
첫 사용자 메시지 미리보기...                  [12턴]  [✕]
2026-05-06 14:32  ·  34215181
```

- **클릭** → 메시지 복원 (현재 세션은 액센트색)
- **✕ 버튼** → 확인 후 삭제
- 상단바 **＋ 세션** → 새 세션 시작 (`session_id` 리셋)

---

## 8. 컨텍스트 시스템 (LLM 입력은 어떻게 만들어지는가)

한 번의 채팅 요청에서 시스템 메시지가 자동으로 합쳐지는 흐름:

```
[1] 기본 시스템 프롬프트 (페르소나, 664자)
   ↓
[2] 반-환각 가드 (176자)
   ↓
[3] 활성 스킬 본문 (각 SKILL.md 합산, 컨텍스트의 40% 까지)
   ↓
[4] 사용자 추가 지시 (⚙️ 프롬프트의 user_extra)
   ↓ → 여기까지가 system 메시지 #1

[5] 도메인 지식 검색 결과 (📚 ON 시, 최대 12,000자) → system 메시지 #2
[6] 워크스페이스 첨부 파일 (최대 16,000자, 파일당 4,000자) → system 메시지 #3

[7] 대화 히스토리 (최근 12턴 트림)
   ↓
LLM 호출
```

**구현 위치 (engine.py):**
- `build_coding_system_prompt(skill_ids, user_extra, n_ctx)` — 1+2+3+4
- `build_knowledge_block(query, results)` — 5
- `build_workspace_block(files)` — 6
- `trim_message_history(messages, max_turns=12)` — 7

이 모든 게 `/api/code/chat/stream` 한 호출에서 자동 처리됩니다.

---

## 9. 하네스 (운영 레이어)

**하네스 = LLM 호출 외부의 운영 레이어**입니다. Tool 레지스트리, 라우터, 세션, 권한, 피드백을 담당합니다.

### 9.1 구성요소 (`harness-mvp/harness/`)

| 모듈 | 역할 |
|------|------|
| Tool, ToolRegistry | 스킬 = Tool 로 등록·조회·실행 |
| ToolRouter | 사용자 질의 → 토큰 매칭 점수로 스킬 추천 |
| HarnessEngine | 단일 턴 실행 루프 (예산·스트림) |
| ToolPermissionContext | deny-list 기반 권한 차단 |
| StoredSession + save/load/list/delete | 세션 JSON 저장·복원 |
| HistoryLog | 이벤트 로그 |
| FeedbackStore | 스킬별 품질 피드백 누적 |
| select_experts | 멀티 에이전트 동적 선정 |

### 9.2 동작 방식

기동 시 `setup_harness(app)` 가 다음을 자동 수행:

1. **`init_harness(skills/)`** — 빈 ToolRegistry 생성
2. **`sync_skills()`** — `code_assist_v1/skills/` 의 모든 SKILL.md 를 ToolRegistry 에 Tool 로 등록 (description = frontmatter 의 description)
3. **`register_harness_routes(app)`** — `/api/harness/*` 16개 엔드포인트 등록

스킬을 만들거나 수정하거나 삭제하면 자동으로 ToolRegistry 가 동기화됩니다 (`_sync_harness()`).

### 9.3 노출되는 16개 엔드포인트

| 엔드포인트 | 용도 |
|----------|------|
| GET `/api/harness/status` | 도구 수·세션 수·이벤트 수 |
| GET `/api/harness/skills?q=` | 스킬 검색 |
| POST `/api/harness/route` | 질의 → 추천 스킬 (라우터 매칭) |
| POST `/api/harness/reload` | ToolRegistry 강제 재초기화 |
| POST `/api/harness/session/save` | 세션 저장 (채팅 종료 시 자동) |
| GET `/api/harness/session/load/<id>` | 세션 복원 |
| GET `/api/harness/session/list` | 세션 목록 |
| DELETE `/api/harness/session/delete/<id>` | 세션 삭제 |
| GET `/api/harness/history` | 이벤트 로그 |
| POST `/api/harness/suggest-combo` | 보조 스킬 추천 |
| POST `/api/harness/validate-combo` | 스킬 조합 유효성 |
| POST `/api/harness/optimize-groups` | 스킬 그룹 최적화 |
| POST `/api/harness/expert-pool` | 동적 에이전트 선정 |
| POST `/api/harness/feedback` | 피드백 저장 |
| GET `/api/harness/feedback/<skill>` | 스킬별 피드백 요약 |
| POST `/api/harness/feedback/prompt-hint` | 피드백 기반 프롬프트 힌트 |

### 9.4 사용자가 일상적으로 만나는 곳

- [세션] 탭의 모든 동작 (저장·로드·삭제·시간 표시)
- 새로 만든 스킬이 기존 agent-* 옆에 즉시 노출 (자동 sync)
- 향후 멀티에이전트·피드백 루프 확장의 기반

---

## 10. 모델

### 10.1 API 모델 (10개)

| ID | 용도 | 컨텍스트 |
|----|------|---------|
| qwen3-coder-480b | 코딩 (최강) | 128K |
| qwen3-coder-next | 코딩 (균형) | 128K |
| qwen3-coder-30b | 코딩 (빠름) | 128K |
| glm-5 | 분석·일반 | 128K |
| qwen3-next-80b | 분석·요약 | 128K |
| qwen35-397b / -fp8 | 초대형 | 128K |
| qwen25-vl-72b / qwen3-vl-30b | 비전 | 128K |
| gpt-oss-20b | 빠름 | 128K |

기본 우선순위: `qwen3-coder-480b → qwen3-coder-next → qwen3-coder-30b → glm-5 → qwen3-next-80b → gpt-oss-20b`

### 10.2 GGUF 자동 로딩

`MODEL_GGUF/` 폴더의 `.gguf` 파일을 첫 기동 시 자동 스캔:

- VRAM 예산 (기본 14GB) 안에서 가장 큰 모델 자동 로드
- `mmproj-*.gguf` 파일이 같이 있으면 비전 모드 자동 감지
- Qwen3 시리즈는 `flash_attn=True` 자동 적용
- 모델 드롭다운에 `gguf-0`, `gguf-1`, … 으로 노출

환경변수로 조정:
```powershell
$env:GGUF_VRAM_BUDGET_GB = "20"
python -m code_assist_v1.app_code
```

---

## 11. 표준 작업 흐름 (신규 프로그램 완성)

```
[1] 도메인 지식 등록
    [지식] 탭 → ＋ 또는 업로드
    예: "사내API_스펙.md", "DB_스키마.md"
    ↓
[2] 코딩 스킬 작성 (선택)
    [스킬] 탭 → ＋ 새 스킬
    예: "신규 백엔드 만들기 절차"
    ↓
[3] 모델 선택
    상단 모델 드롭다운 → qwen3-coder-480b
    ↓
[4] 활성화
    - [스킬] 탭에서 사용할 스킬 클릭 (🛠 칩 표시)
    - 입력박스의 📚 ON
    - 📎 으로 기존 코드 첨부
    ↓
[5] 질문
    "이 스펙으로 신규 프로그램 만들어줘"
    ↓ SSE 스트리밍 응답
[6] 응답 코드를 워크스페이스에 저장
    ↓
[7] 다음 턴에 다시 첨부 → 반복 → 완성
```

---

## 12. 폴더 구조

```
code_assist_v1/
├── TOKEN.TXT                ← API 토큰 (자체)
├── api_config.json          ← 모델 설정 10개 (자체)
├── MODEL_GGUF/              ← .gguf 모델 폴더 (자체)
│
├── skills/                  ← 코딩 스킬 69개 (시드됨, 평면)
├── knowledge/               ← 도메인 지식 (사용자가 직접 등록, 평면)
├── workspace/               ← 작업 파일 (평면)
├── sessions/                ← 세션 보조
│
├── static/                  ← 프론트엔드 SPA
├── docs/                    ← 문서
│
├── prompts.py               ← 시스템 프롬프트
├── engine.py                ← 컨텍스트 빌더
├── routes_*.py              ← API 엔드포인트들
├── harness_setup.py         ← 하네스 통합
└── app_code.py              ← 진입점 (포트 10010)
```

기존 `scientific-skills/`, `knowledge/ggg3g/` 같은 외부 폴더는 **첫 시드 때만** 참조됩니다. 이후엔 `code_assist_v1/` 안의 자체 자산만 봅니다.

---

## 13. 자주 묻는 질문

**Q. demos_v1 (PPT 플랫폼) 와 같이 띄울 수 있나요?**
네. 포트가 다르므로 (demos_v1: 10009, code_assist_v1: 10010) 두 터미널에서 동시 실행 가능합니다. 두 플랫폼은 코드·데이터 모두 독립입니다.

**Q. 왜 demos_v1 보다 빠른가요?**
- demos_v1 는 한 번 질문에 LLM 을 3~5번 호출 (멀티에이전트 + 합성)
- demos_v1 는 자동 knowledge-search 트리거로 매번 12K 토큰 추가 주입
- demos_v1 는 자동 스킬 선택으로 시스템 프롬프트가 50~100KB
- code_assist_v1 는 단일 모델 단일 호출, 사용자가 명시한 것만 동작 → SSE 토큰이 즉시 화면에 떨어짐

**Q. TOKEN.TXT 가 비어있어도 되나요?**
네. GGUF 모델만 사용하면 토큰 없이도 정상 동작합니다.

**Q. 폐쇄망에서 쓸 수 있나요?**
네. 외부 API 의존 없습니다. CDN(marked.js, highlight.js) 차단 시 `chat.js` 의 폴백 파서로 마크다운 표시됩니다 (syntax highlighting 만 비활성).

**Q. 도메인 지식이 BM25 로 안 잡혀요.**
- 검색어가 너무 짧으면 (1글자) 안 잡힘
- frontmatter 의 tags 와 본문에 검색 키워드를 포함시키세요
- 파일명에 핵심 키워드를 넣으면 +5 보너스

**Q. 만든 스킬을 다른 사람과 공유하려면?**
`code_assist_v1/skills/<id>/` 폴더를 통째로 복사해서 다른 PC 의 같은 위치에 두면 됩니다.

**Q. 컨텍스트와 하네스가 진짜 들어가 있는지 확인하는 방법?**
- 좌사이드 [세션] 탭에 채팅 끝날 때마다 자동 저장된 세션이 보임 → 하네스 동작 중
- ⚙️ 프롬프트 모달에서 기본 시스템 프롬프트와 사용자 추가 지시가 표시됨 → 컨텍스트 빌더 동작 중
- `http://localhost:10010/api/harness/status` 직접 호출 시 `tools_count: 69` 등의 응답 확인 가능

---

## 14. 키보드 단축

| 키 | 동작 |
|----|------|
| Enter | 메시지 전송 |
| Shift + Enter | 줄바꿈 |
| Esc | 스트리밍 중단 |
| Ctrl + F5 | 강제 새로고침 (테마·UI 즉시 반영) |

---

*Freedom 코딩 어시스턴트 · 사내 폐쇄망 전용 · Port 10010*
