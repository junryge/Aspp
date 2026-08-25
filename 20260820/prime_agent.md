# Prime Agent — Python 이식본 (`prime-agent-py`)

[PrimeIntellect-ai/prime-agent](https://github.com/PrimeIntellect-ai/prime-agent) v0.8.0 의
**RLM 하네스를 순수 Python 으로 옮긴 구현**이다.

Node.js·npm·네이티브 빌드·bash 가 **전부 필요 없다.** Windows 폐쇄망에서 `pip` 만으로 돈다.

```
pip install prime-agent-py            # 또는 오프라인: pip install --no-index --find-links wheelhouse ...
pa init --llm-base-url http://10.0.0.42:8000/v1 --llm-model Qwen3-32B
pa doctor
pa
```

---

## 왜 이식했나

원본은 TypeScript 모노레포 4패키지 + Python 커널 런타임이다. 사내 Windows 폐쇄망에서는
설치 경로가 이렇게 막힌다.

| 원본 요구사항 | 폐쇄망 Windows 에서 |
|---|---|
| Node.js 22.8+ | 별도 배포 필요 |
| `npm ci` (의존성 212건) | 사내 레지스트리 미러 필요 |
| `zeromq`·`canvas` 네이티브 빌드 | Windows 빌드 툴체인 필요, `canvas` 는 특히 까다로움 |
| TypeScript 7 프리릴리스 핀 | 사내 미러에 없으면 소스 빌드 자체가 막힘 |
| `install.sh` (POSIX sh) | Windows 경로 없음. bash 필요 |
| Prime Agent 실행 시 bash 필수 | Git Bash / Cygwin / WSL 설치 필요 |
| `uv` (astral.sh) + CPython(GitHub) + PyPI 13종 | 3개 외부 호스트 |

이 이식본은 그 전부를 없앤다. **Python 3.10+ 와 5개 순수/휠 패키지가 전부다.**

---

## 원본에서 그대로 가져온 것

| 설계 | 구현 |
|---|---|
| **툴은 `ipython` 하나뿐** | 파일·셸·데이터·스킬·위임이 전부 커널 안의 Python 코드 |
| **영속 IPython 커널** | `jupyter_client` 로 실제 IPython 커널을 띄운다. 변수·import 가 턴과 컴팩션을 넘어 유지 |
| **`rlm(...)` 재귀 서브에이전트** | admission 시점에 핸들 반환. 자식 답변은 절대 반환 안 함 |
| **결과는 `agent_message` 로만** | 자식이 `send(..., receiver_role="parent")` 하거나 파일로 |
| **가족 범위 메시징** | 부모·형제·직계 자식. 발신자 신원은 호스트가 정함 |
| **Continual Harness** | prompt/memory/skill/subagent 4종, `harness_state.json`, 원자적 쓰기 |
| **`/refine` + 롤백** | before/after 스냅샷, 역순 재생, 전역 이력 `refinements.jsonl` |
| **기본 시스템 프롬프트 불변** | `base_system_prompt` id 하드 거부 |
| **세션 JSONL 트리** | version 3 포맷. 엔트리 타입·필드명을 원본과 맞춤 |
| **컴팩션** | `contextTokens > contextWindow - reserveTokens`, `keepRecentTokens` 보존, 툴 결과에서 안 자름 |
| **goal** | `goal.complete()` 만이 성공 완료 신호 |
| **heartbeat 2종** | 사용자 `/heartbeat` 1개 + 에이전트 `rlm_heartbeat` 다수 |
| **autonomous** | **게이트 먼저** 판정 → 연속 → 턴 → 토큰 → 시간. 게이트 통과는 한도 도달을 무시하고 완료 허용 |
| **SKILL.md 스킬 계약** | 프론트매터 규칙, Python 스킬 4조건, description 없으면 미로드 |
| **재귀 깊이 제한** | 기본 2 (root → 자식 → 손자) |

**저장소 원본 스킬이 수정 없이 그대로 돈다.** 커널에 `rlm` 패키지를 같은 이름으로 노출하므로
`from rlm import host_request` 를 쓰는 `goal`·`compact`·`refine`·`agent-message`·
`agent-observe`·`rlm-heartbeat`·`edit` 스킬이 그대로 동작한다.

## 원본과 다른 것 (의도된 축소)

| 원본 | 이식본 | 이유 |
|---|---|---|
| 데몬 + 슈퍼바이저 + 워커 3프로세스 | 단일 프로세스 + 스레드 | Windows named pipe·프로세스 그룹 관리 회피 |
| Jupyter comm `host.request` | loopback TCP + 토큰 | 훨씬 단순하고 Windows 친화적 |
| 프로바이더 20여 종 | OpenAI 호환 1종 | 사내 vLLM/SGLang/게이트웨이가 대상 |
| TUI (차등 렌더링, 테마) | 스트리밍 콘솔 + prompt_toolkit | 의존성 최소화 |
| 세션 리스, 크래시 복구 저널 | 없음 | 단일 프로세스라 불필요 |
| MCP 통합 | 없음 | 필요하면 스킬로 추가 가능 |
| 확장(TypeScript) | 없음 | 스킬로 대체 |

> 데몬이 없으므로 **터미널을 닫으면 세션이 끝난다.** 백그라운드 실행이 필요하면
> `nohup` / `screen` / Windows 서비스로 감싸라. 세션 JSONL 은 계속 남으므로
> `pa -r <id>` 로 언제든 이어갈 수 있다.

---

## 설치

### 온라인

```bash
pip install prime-agent-py
```

### 폐쇄망 (wheelhouse)

```bash
pip install --no-index --find-links wheelhouse prime-agent-py
```

필요한 것: `httpx`, `jupyter-client`, `ipykernel`, `prompt-toolkit`, `rich`
(+ 커널에서 쓸 `pandas`/`numpy` 등은 선택)

### 커널 인터프리터 분리

기본은 현재 Python 을 커널로 쓴다. 별도 venv 를 쓰려면:

```bash
export PA_KERNEL_PYTHON=/opt/pa/kernel-venv/bin/python     # Windows: set PA_KERNEL_PYTHON=...
```

그 환경에는 `ipykernel` 이 있어야 한다.

---

## 설정

`pa init` 이 `models.json` 과 `settings.json` 을 만든다.

```bash
pa init --llm-base-url http://10.0.0.42:8000/v1 \
        --llm-model Qwen/Qwen3-32B \
        --context-window 262144
```

`~/.prime/agent-py/models.json` (원본 `models.json` 과 스키마 호환):

```json
{
  "providers": {
    "internal": {
      "baseUrl": "http://10.0.0.42:8000/v1",
      "api": "openai-completions",
      "apiKey": "INTERNAL_LLM_KEY",
      "authHeader": true,
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false,
        "maxTokensField": "max_tokens"
      },
      "models": [{
        "id": "Qwen/Qwen3-32B",
        "contextWindow": 262144,
        "maxTokens": 32768,
        "reasoning": true,
        "compat": { "thinkingFormat": "qwen-chat-template" }
      }]
    }
  }
}
```

`apiKey` 값 해석은 원본과 같다: `!명령` → 셸 stdout, 환경변수명 → 그 값, 그 외 → 리터럴.

> **`contextWindow` 는 서버 `--max-model-len` 과 반드시 맞춰라.** 컴팩션 트리거 계산에 직접 들어간다.
>
> **툴 콜 지원이 전제다.** 유일한 툴이 `ipython` 이므로 tool calling 을 못 하는 모델은 못 쓴다.
> vLLM 이면 `--enable-auto-tool-choice` 와 모델별 `--tool-call-parser` 를 켜라.

---

## 사용

```bash
pa                                     # 대화형
pa "src 아래 파이썬 파일 수를 세라"        # 대화형 + 초기 프롬프트
pa -p "이 저장소 구조를 요약해라"           # 한 번 실행하고 종료
pa --mode json "..."                   # 이벤트를 JSON 라인으로
pa -c                                  # 최근 세션 이어가기
pa -r a1b2c3d4                         # 특정 세션 재개
pa sessions                            # 세션 목록
pa doctor                              # 환경·엔드포인트·스킬 점검
```

### autonomous (게이트 붙은 무인 실행)

```bash
pa -p --autonomous \
   --autonomous-gate "python -m pytest -q" \
   --autonomous-gate "python -m ruff check ." \
   --autonomous-max-turns 12 \
   --autonomous-max-tokens 80000 \
   "실패하는 테스트를 고치고 검증 결과를 보고해라"
```

판정 순서는 원본과 같다: **게이트 먼저** → 연속 → 턴 → 토큰 → 시간.
게이트가 통과하면 한도에 도달했어도 완료를 허용한다.
**한도 도달은 작업 성공을 뜻하지 않는다.** 사내에서는 게이트 없이 쓰지 마라.

### 슬래시 커맨드

```
세션      /new  /sessions  /resume <id>  /name <이름>  /session  /tree
컨텍스트  /compact [지시]  /usage
하네스    /refine [지시]  /refine --global  /refine --rollback <id>  /harness
장기실행  /goal <목표>  /heartbeat every 10m <지시>
에이전트  /agents  /send <이름> <메시지>
모델      /model [패턴]  /models  /effort <레벨>
스킬      /skills  /skill:<이름> [인자]  /reload
기타      !<셸명령>  /help  /quit
```

---

## 커널 안에서 쓸 수 있는 것

```python
# 서브에이전트 — admission 시점에 반환한다. 답을 기다리지 않는다.
api  = await rlm("공개 API 를 검토해라", name="api-reviewer")
test = await rlm("빠진 회귀 테스트를 찾아라", name="test-reviewer")
# 여기서 턴을 끝낸다.

await rlm.list_subagents()
await rlm.find_models()
await rlm.delete_subagent("api-reviewer")

# 메시징 (부모·형제·직계 자식만)
await agent_message.list_agents()
await agent_message.send("결과 요약", receiver_role="parent")
await agent_message.send("이것도 봐라", receiver_role="child", receiver_name="api-reviewer")

# 관측 (읽기 전용)
await agent_observe.recent_messages("api-reviewer", limit=5)

# 하네스
await harness.overview()
await harness.create_memory("배포 절차", "릴리스는 금요일에 하지 않는다", global_=True)
await harness.create_skill("로그 파서", "MCS 로그를 파싱한다",
                           reference={"type":"python","import":"mcs_log",
                                      "call_pattern":"await mcs_log(path)"},
                           arguments={"path":{"type":"string","required":True}})

# goal / compaction / refine
await goal.get();  await goal.complete()
await compact.status();  await compact.run("인증 리팩터링에 집중")
await refine.run("반복 실패를 메모리로 남겨라")

# 에이전트 소유 heartbeat
await rlm_heartbeat.create("빌드 상태 확인", interval="10m", label="build")

# 파일 편집 (유일 일치 1회 치환)
await edit("src/app.py", "old_line", "new_line")
```

셸 규칙(원본과 동일):

- `%%bash` 는 **셀의 첫 줄**이어야 한다
- `%%bash` 는 일회용 서브셸 — `cd`/`export` 가 안 남는다
- 지속시키려면 `%cd <dir>`, `os.environ['VAR'] = '...'`

---

## 스킬

저장소 원본 스킬 디렉터리를 그대로 놓으면 된다.

```
~/.prime/agent-py/skills/<이름>/SKILL.md
<프로젝트>/.prime/agent-py/skills/<이름>/SKILL.md
```

우선순위: 명시 경로 → 프로젝트 → 전역 → 번들. 이름 충돌 시 처음 것을 유지한다.

Python 스킬은 4조건이 전부 성립해야 한다(하나라도 어긋나면 마크다운 전용으로 강등):

1. `SKILL.md` — 프론트매터에 `name`, `description` (**description 없으면 로드 안 됨**)
2. 스킬 루트에 `pyproject.toml`
3. import 이름 = 스킬명의 하이픈→언더스코어, 유효한 Python 식별자
4. `src/<import_name>/__init__.py`

`__init__.py` 에 `run()` 이 있으면 모듈 자체가 async callable 이 된다:
`await word_count("text", top=3)`.

설치는 커널 인터프리터에 editable 로:

```bash
$PA_KERNEL_PYTHON -m pip install -e ~/.prime/agent-py/skills/my-skill
# 폐쇄망이면
$PA_KERNEL_PYTHON -m pip install --no-index --find-links wheelhouse -e ~/.prime/agent-py/skills/my-skill
```

추가 후에는 **새 세션**을 시작해야 커널이 import 한다.

---

## 디렉터리 레이아웃

```
~/.prime/agent-py/
├── settings.json
├── models.json
├── history                      prompt_toolkit 입력 이력
├── skills/<이름>/SKILL.md
├── sessions/<uuid>.jsonl        세션 트랜스크립트 (원본 v3 포맷)
├── session-artifacts/<uuid>/
│   ├── harness/harness_state.json     로컬 하네스
│   └── sub-xxxxxxxx/<child>.jsonl     RLM 자식 (재귀 중첩)
├── harness/
│   ├── harness_state.json             전역 하네스
│   └── refinements.jsonl              전역 refine 이력 (롤백 소스)
└── logs/
```

환경변수:

| 변수 | 의미 |
|---|---|
| `PA_AGENT_DIR` | 설정 디렉터리 (기본 `~/.prime/agent-py`) |
| `PA_KERNEL_PYTHON` | 커널로 쓸 Python 인터프리터 |
| `PA_OFFLINE` | 오프라인 표시 (외부 호출 억제) |

---

## 보안 — 원본과 동일하게 샌드박스가 아니다

커널은 모델이 만든 Python 을 **호스트와 같은 OS 권한으로** 실행한다.
프로세스 분리는 라이프사이클 격리이지 권한 격리가 아니다.

- 전용 저권한 계정으로 돌려라
- 일회용 클론 / 깨끗한 worktree 에서 써라
- 진짜 격리가 필요하면 **외부 샌드박스**(컨테이너·전용 VM)를 바깥에 둬라
- 스킬·Python 패키지는 전부 신뢰 코드다. 사내 검토 절차를 태워라
- 세션 JSONL 과 하네스는 대화 원문을 그대로 담는다. 보존·파기 정책을 먼저 세워라

---

MIT. 원본 Prime Agent 와 그 기반인 [`pi`](https://github.com/earendil-works/pi) 의 설계를 따랐다.
