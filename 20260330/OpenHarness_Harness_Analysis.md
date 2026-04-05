# OpenHarness — Agent Harness 기능 분석 문서

> HKUDS/OpenHarness GitHub 소스코드 기반 분석 | v0.1.0 | 2026-04-06  
> 작성 목적: ASAS 플랫폼 아키텍처 참고자료

---

## 1. 개요

OpenHarness는 홍콩대학교 HKUDS 연구실에서 개발한 Claude Code의 초경량 오픈소스 Python 대안이다. Claude Code의 512,664줄(TypeScript) 대비 11,733줄(Python)으로 44배 경량화를 달성했으며, 핵심 도구 43개(98%) 및 커맨드 54개(61%)를 구현한다.

**핵심 철학:** "모델은 에이전트이다. 코드는 하네스이다."

LLM이 지능을 제공하면, 하네스가 손(도구), 눈(관찰), 기억(메모리), 안전 경계(권한)를 제공한다.

**Harness = Tools + Knowledge + Observation + Action + Permissions**

---

## 2. 하네스 코어 아키텍처

하네스 코어는 7개 모듈, 총 2,230줄로 구성된다. 전체 11,733줄 중 약 19%가 순수 하네스 인프라이며, 나머지는 개별 도구 구현, UI, CLI, API 클라이언트, 멀티에이전트(swarm) 등이다.

| 모듈 | 줄수 | 역할 |
|------|------|------|
| engine | 706 | Agent Loop, 메시지 관리, 스트림 이벤트, 비용 추적 |
| tools/base | 75 | 도구 추상클래스(BaseTool) + 레지스트리(ToolRegistry) |
| permissions | 145 | 3단계 모드(Default/Plan/FullAuto) + 경로/명령 규칙 |
| hooks | 481 | PreToolUse/PostToolUse 라이프사이클 이벤트 (4타입) |
| skills | 166 | .md 파일 기반 온디맨드 지식 로딩 |
| memory | 274 | 파일 기반 세션 간 영구 기억 |
| prompts | 383 | 시스템 프롬프트 조립, 환경정보, CLAUDE.md |
| **합계** | **2,230** | **전체 11,733줄 중 19%** |

---

## 3. Agent Loop (engine/query.py — 292줄)

하네스의 핵심. `run_query()` 함수 하나가 전체 에이전트 루프를 담당한다.

### 3.1 실행 흐름

```
for _ in range(max_turns=200):
    ① auto_compact_if_needed()     # 컨텍스트 초과시 자동 압축
    ② api_client.stream_message()   # LLM에 스트리밍 요청
    ③ stop_reason != tool_use → break  # 모델이 도구 안 부르면 종료
    ④ tool_call 1개 → 순차, 2개+ → asyncio.gather 병렬
    ⑤ messages.append(tool_results) → 루프 반복
```

### 3.2 병렬 도구 실행

- 도구 호출 1개: 순차 실행 (스트림 이벤트 즉시 emit)
- 도구 호출 2개 이상: `asyncio.gather`로 병렬 실행 후 결과 일괄 emit
- 독립적인 도구 호출(grep + glob 등)의 성능을 크게 개선

### 3.3 Auto-Compact

매 턴 시작 전 컨텍스트 윈도우 초과 여부를 체크하고 2단계로 압축:

1. **Microcompact** — 오래된 tool result 내용 비우기 (저비용)
2. **Full compact** — LLM 기반 요약 (고비용)

`max_turns=200` 초과 시 `MaxTurnsExceeded` 예외 발생.

---

## 4. Tool 실행 파이프라인 (_execute_tool_call)

하나의 도구 호출이 거치는 8단계 파이프라인:

| 단계 | 처리 | 설명 |
|------|------|------|
| ① | PreToolUse 훅 | blocked=true면 즉시 거부, 도구 실행 안 함 |
| ② | ToolRegistry 조회 | 도구 없으면 "Unknown tool" 에러 반환 |
| ③ | Pydantic 검증 | model_validate()로 입력 타입 검증 |
| ④ | 경로/명령 추출 | file_path/path 필드 자동 감지, 상대→절대경로 변환 |
| ⑤ | Permission 평가 | PermissionChecker.evaluate() — 모드/경로/명령 규칙 적용 |
| ⑥ | 사용자 확인 | requires_confirmation이면 사용자 y/n 프롬프트 |
| ⑦ | tool.execute() | 실제 도구 실행 |
| ⑧ | PostToolUse 훅 | 실행 결과를 훅에 전달 |

### 4.1 경로 해석 로직

```python
# file_path 또는 path 필드를 자동 감지
for key in ("file_path", "path"):
    value = raw_input.get(key)
    if isinstance(value, str):
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = cwd / path      # 상대경로 → cwd 기준 절대경로
        return str(path.resolve())
```

---

## 5. Permission 시스템 (permissions/ — 145줄)

3단계 모드 + 규칙 기반 권한 제어 시스템.

### 5.1 권한 모드

| 모드 | 동작 | 사용 시나리오 |
|------|------|-------------|
| DEFAULT | read-only 허용, mutating은 사용자 확인 필요 | 일반 개발 작업 |
| PLAN | read-only만 허용, mutating 전면 차단 | 대규모 리팩토링 검토 |
| FULL_AUTO | 모든 도구 자동 허용 | 샌드박스 환경 |

### 5.2 추가 규칙

- **denied_tools**: 특정 도구 명시적 차단
- **allowed_tools**: 특정 도구 명시적 허용 (모드 무시)
- **path_rules**: glob 패턴으로 경로별 차단/허용 (e.g. `/etc/*` 차단)
- **denied_commands**: 명령어 패턴 차단 (e.g. `rm -rf /`, `DROP TABLE *`)

### 5.3 평가 순서

```
denied_tools → allowed_tools → path_rules → denied_commands → 모드 판단
```

### 5.4 PermissionDecision 반환 구조

```python
@dataclass(frozen=True)
class PermissionDecision:
    allowed: bool                    # 허용 여부
    requires_confirmation: bool      # 사용자 확인 필요 여부
    reason: str                      # 사유 메시지
```

---

## 6. Hook 시스템 (hooks/ — 481줄)

hooks/executor.py (230줄) 기반의 라이프사이클 이벤트 시스템.

### 6.1 이벤트

- **PRE_TOOL_USE** — 도구 실행 전. blocked=true 반환 시 도구 실행 거부
- **POST_TOOL_USE** — 도구 실행 후. 결과 로깅/감시 등에 활용

### 6.2 훅 타입 4가지

| 훅 타입 | 동작 |
|---------|------|
| CommandHook | 셸 명령 실행. 환경변수(OPENHARNESS_HOOK_EVENT, OPENHARNESS_HOOK_PAYLOAD)로 payload 전달. timeout 지원. 종료코드로 성공/실패 판단 |
| HttpHook | 외부 URL에 POST 요청. {event, payload} JSON 전송. HTTP 상태코드로 판단 |
| PromptHook | LLM에 판단 질의. `{"ok": true/false, "reason": "..."}` JSON 반환 기대 |
| AgentHook | PromptHook + "더 꼼꼼히 판단해라" 지시 추가. 복잡한 검증에 사용 |

### 6.3 매칭 및 차단

- **matcher**: fnmatch 패턴으로 도구 이름에 필터 적용 (e.g. `bash*`)
- **block_on_failure**: true면 훅 실패 시 도구 실행 자체를 거부
- **_inject_arguments**: 템플릿의 `$ARGUMENTS`를 payload JSON으로 치환

### 6.4 LLM 훅 판단 로직

```python
# PromptHook/AgentHook의 LLM 응답 파싱
def _parse_hook_json(text):
    parsed = json.loads(text)
    if isinstance(parsed.get("ok"), bool):
        return parsed
    # 파싱 실패 시 "ok"/"true"/"yes" → 통과, 나머지 → 거부
```

---

## 7. Skills 시스템 (skills/ — 166줄)

.md 파일 기반 온디맨드 지식 로딩 시스템.

### 7.1 로딩 순서

1. **번들 스킬** (skills/bundled/) — 내장 스킬 (commit, review, debug, plan, test 등)
2. **유저 스킬** (~/.openharness/skills/*.md) — 사용자 직접 추가
3. **플러그인 스킬** — 플러그인에서 로드

anthropics/skills 호환 — .md 파일을 복사하면 즉시 사용 가능.

### 7.2 마크다운 파싱 규칙

```
우선순위 1: YAML frontmatter (---) 에서 name/description 추출
우선순위 2: # 제목에서 name 추출
우선순위 3: 첫 번째 비어있지 않은 문단에서 description 추출 (200자 제한)
```

### 7.3 SkillDefinition 구조

```python
@dataclass
class SkillDefinition:
    name: str           # 스킬 이름
    description: str    # 설명
    content: str        # 전체 마크다운 내용
    source: str         # "bundled" | "user" | "plugin"
    path: str           # 파일 경로
```

---

## 8. Memory 시스템 (memory/ — 274줄)

파일 기반 세션 간 영구 기억 시스템.

### 8.1 저장 구조

```
.openharness/memory/
├── MEMORY.md          ← 인덱스 파일 (목차 역할)
├── project_setup.md   ← 개별 기억 파일
├── api_design.md
└── ...
```

### 8.2 핵심 함수

| 함수 | 동작 |
|------|------|
| add_memory_entry(cwd, title, content) | slug 생성 → .md 파일 생성 → MEMORY.md에 링크 추가 |
| remove_memory_entry(cwd, name) | .md 파일 삭제 + MEMORY.md에서 제거 |
| list_memory_files(cwd) | 프로젝트의 모든 메모리 파일 목록 |
| scan.py | 메모리 디렉토리 스캔 |
| search.py | 메모리 내용 검색 |

### 8.3 slug 생성

```python
slug = re.sub(r"[^a-zA-Z0-9]+", "_", title.lower()).strip("_") or "memory"
# "API Design Notes" → "api_design_notes"
```

---

## 9. System Prompt 조립 (prompts/ — 383줄)

### 9.1 구조

```
최종 시스템 프롬프트 = BASE_PROMPT + Environment 정보
```

### 9.2 BASE_PROMPT 핵심 지침 요약

**도구 우선 규칙:**
- read → read_file (cat/head/tail 사용 금지)
- edit → edit_file (sed/awk 사용 금지)
- write → write_file (echo/heredoc 사용 금지)
- search → glob (find/ls 사용 금지)
- content search → grep (grep/rg 사용 금지)
- Bash는 시스템 명령에만 사용

**보안 지침:**
- command injection, XSS, SQL injection 방지
- OWASP Top 10 우선

**행동 원칙:**
- 안 읽은 코드 수정 금지
- 요청 이상의 개선 금지
- 과도한 추상화 금지
- 발생 불가능한 시나리오에 대한 에러 처리 금지
- 위험 행동(force push, rm -rf, DROP TABLE 등)은 반드시 사용자 확인

**병렬 실행:**
- 독립적인 도구 호출은 병렬로 실행하라는 지침

### 9.3 Environment 정보 자동 감지

```
- OS: {os_name} {os_version}
- Architecture: {platform_machine}
- Shell: {shell}
- Working directory: {cwd}
- Date: {date}
- Python: {python_version}
- Git: yes/no (branch: {git_branch})
```

---

## 10. Tool 기본 구조 (tools/base.py — 75줄)

### 10.1 BaseTool 추상클래스

| 속성/메서드 | 설명 |
|------------|------|
| name: str | 도구 이름 (예: bash, read_file, grep) |
| description: str | 도구 설명 (LLM이 자동 이해) |
| input_model: type[BaseModel] | Pydantic BaseModel — 타입 안전 입력 검증 |
| async execute(arguments, context) → ToolResult | 실제 실행. output + is_error 반환 |
| is_read_only(arguments) → bool | 읽기 전용 여부 — 권한 판단에 사용 |
| to_api_schema() → dict | Anthropic Messages API 형식 JSON Schema 자동 생성 |

### 10.2 ToolRegistry

단순 dict 기반 name→tool 매핑.

```python
class ToolRegistry:
    _tools: dict[str, BaseTool]
    
    register(tool)        # 도구 등록
    get(name) → BaseTool  # 이름으로 조회
    list_tools() → list   # 전체 목록
    to_api_schema() → list  # API 형식 전체 스키마
```

### 10.3 구현된 43개 도구 분류

| 카테고리 | 도구 | 파일 수 |
|---------|------|--------|
| File I/O | bash, read_file, write_file, file_edit, glob, grep | 6 |
| Search | web_fetch, web_search, tool_search, lsp | 4 |
| Notebook | notebook_edit | 1 |
| Agent | agent, send_message, team_create, team_delete | 4 |
| Task | task_create/get/list/update/stop/output | 6 |
| MCP | mcp_tool, list_mcp_resources, read_mcp_resource, mcp_auth | 4 |
| Mode | enter_plan_mode, exit_plan_mode, enter_worktree, exit_worktree | 4 |
| Schedule | cron_create/list/delete/toggle, remote_trigger | 5 |
| Meta | skill, config, brief, sleep, ask_user, todo_write | 6 |

---

## 11. 기타 하네스 서브시스템

### 11.1 Swarm — 멀티에이전트 (169K)

가장 큰 서브시스템. 서브에이전트 생성, 팀 협업, 메일박스 통신, 권한 동기화 등을 담당.

| 파일 | 크기 | 역할 |
|------|------|------|
| in_process.py | 25K | 인프로세스 에이전트 실행 |
| mailbox.py | 19K | 에이전트 간 메시지 통신 |
| permission_sync.py | 37K | 에이전트 간 권한 동기화 |
| team_lifecycle.py | 29K | 팀 생성/삭제/관리 |
| worktree.py | 11K | Git worktree 기반 병렬 작업 |

### 11.2 Commands (commands/registry.py — 68K)

54개 슬래시 커맨드 레지스트리. /help, /commit, /plan, /resume 등.

### 11.3 Plugins (plugins/ — 20K)

claude-code 플러그인 호환. 로더, 스키마, 번들 플러그인 포함.

### 11.4 Config (config/ — 16K)

다층 설정 시스템. 경로 관리 + settings.json 기반.

---

## 12. ASAS 플랫폼 적용 시사점

| 항목 | OpenHarness | ASAS 참고점 |
|------|-------------|------------|
| Agent Loop | run_query() 단일 함수 292줄 | Flask 기반 루프 단순화 가능 |
| 도구 추상화 | BaseTool + Pydantic 검증 | 355-skill 체계에 동일 패턴 적용 가능 |
| 권한 체계 | 3단계 모드 + 경로/명령 규칙 | 폐쇄망 환경에서 계층별 권한 제어 참고 |
| 훅 시스템 | Pre/Post 라이프사이클 | ASAS 도구 실행 전후 감사/로깅에 활용 |
| Skills | .md 파일 기반 온디맨드 로딩 | ASAS 24카테고리 스킬과 동일 개념 |
| Memory | 파일 기반 MEMORY.md 인덱스 | ASAS 세션 간 컨텍스트 유지에 참고 |
| Auto-Compact | 2단계 컨텍스트 압축 | Qwen3-235B 컨텍스트 관리에 적용 가능 |
| 병렬 실행 | asyncio.gather 기반 | ASAS 멀티에이전트 오케스트레이션에 참고 |

### 핵심 차용 가능 개념

1. **"하네스" 용어 체계** — ASAS 아키텍처 문서에서 "LLM은 두뇌, Flask 서버는 하네스" 프레이밍
2. **Tool 실행 파이프라인 8단계** — 훅→검증→권한→실행→훅 구조를 ASAS에 도입
3. **Permission 모드** — 폐쇄망 환경에서 Default/Auto/Plan 모드 분리
4. **Skills 호환 포맷** — .md 기반 스킬 정의를 ASAS 표준으로 채택 가능

---

*문서 끝*
