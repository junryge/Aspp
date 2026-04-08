# OpenHarness - Open Agent Harness

팀원 개인별 CLI AI 에이전트 하네스 (37개 스킬 + 2개 플러그인 내장)

---

## 1. 설치 방법

### 방법 1: 설치 스크립트 (권장)

```bash
unzip openharness.zip
cd openharness
bash install.sh
```

### 방법 2: 수동 설치

```bash
unzip openharness.zip
cd openharness
pip install -e .
mkdir -p ~/.openharness/{skills,plugins,sessions}
echo "your-api-key" > ~/.openharness/TOKEN.TXT
```

### 방법 3: PYTHONPATH 직접 사용 (pip 없이)

```bash
unzip openharness.zip
cd openharness
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
python -m openharness
```

---

## 2. API 키 설정 (필수)

LLM 서버 연결을 위해 TOKEN.TXT에 API 키를 넣어야 합니다.

```bash
# 키 파일 위치 (우선순위 순서)
~/.openharness/TOKEN.TXT    # 1순위: 개인 홈
./TOKEN.TXT                  # 2순위: 현재 디렉토리
```

```bash
echo "sk-your-api-key-here" > ~/.openharness/TOKEN.TXT
```

> TOKEN.TXT가 없으면 시작 시 경고가 표시되고 API 호출이 불가합니다.

---

## 3. 사용법

### 3-1. 인터랙티브 모드 (기본)

```bash
oh
```

```
  ╔══════════════════════════════════════════╗
  ║     OpenHarness  v0.1.0                  ║
  ║     Open Agent Harness for Teams         ║
  ╚══════════════════════════════════════════╝

  🔑 TOKEN.TXT: loaded (32 chars)
  🤖 Default model: PROD (397B)
  🔧 Tools: 45 (37 skills, 2 plugins)
  💬 Type /help for commands, /exit to quit

you> 이 코드를 리뷰해주세요
```

### 3-2. 모델 지정

```bash
oh --model qwen3-coder-480b     # 코딩 전문 (480B)
oh --model glm-5                # 빠른 범용
oh --model qwen3-vl-235b        # 비전 모델 (이미지)
oh --model gpt-oss-120b         # 경량 모델
oh --model qwen3.5-397b         # 최고 성능 (기본값)
```

### 3-3. 헤드리스 모드 (단일 명령)

```bash
# 간단한 질문
oh run "Python으로 퀵소트 구현해줘"

# JSON 출력
oh run "이 코드의 버그를 찾아줘" --json

# 여러 턴 실행
oh run "테스트를 작성하고 실행해줘" --max-turns 5
```

### 3-4. 상태 확인 커맨드

```bash
oh status     # 시스템 상태 (토큰, 모델, 도구 수)
oh models     # 사용 가능한 모델 목록
oh skills     # 로드된 스킬 목록
oh plugins    # 로드된 플러그인 목록
```

---

## 4. 슬래시 커맨드

인터랙티브 모드에서 `/`로 시작하는 커맨드를 사용합니다:

| 커맨드 | 설명 | 예시 |
|--------|------|------|
| `/help` | 사용 가능한 커맨드 전체 목록 | `/help` |
| `/models` | LLM 모델 목록 (현재 모델에 → 표시) | `/models` |
| `/model [name]` | 모델 확인/변경 | `/model qwen3-coder-480b` |
| `/skills` | 로드된 스킬 목록 | `/skills` |
| `/plugins` | 로드된 플러그인 목록 | `/plugins` |
| `/status` | 시스템 상태 요약 | `/status` |
| `/commit` | Git 커밋 생성 가이드 | `/commit` |
| `/review` | 코드 리뷰 실행 | `/review` |
| `/debug` | 버그 진단 가이드 | `/debug` |
| `/plan` | 구현 계획 수립 | `/plan` |
| `/test` | 테스트 작성 가이드 | `/test` |
| `/simplify` | 코드 리팩토링 가이드 | `/simplify` |
| `/pdf` | PDF 생성/분석 | `/pdf` |
| `/xlsx` | Excel 처리 | `/xlsx` |
| `/code-review` | 자동 PR 코드 리뷰 (플러그인) | `/code-review` |
| `/clear` | 화면 지우기 | `/clear` |
| `/exit` | 종료 | `/exit` |

### 사용 예시

```
you> /models
# Available Models

  → qwen3.5-397b              PROD (397B)          [high]
    qwen3-coder-480b           Coder-480B           [high]
    glm-5                      GLM-5                [medium]
    gpt-oss-120b               COMMON (120B)        [medium]
    glm-4.7                    GLM-4.7              [low]

you> /model qwen3-coder-480b
Model set to: qwen3-coder-480b

you> /status
# Status
  Token: loaded
  Model: qwen3-coder-480b
  Tools: 45
  Skills: 37
  Plugins: 2

you> /skills
# Skills (37 loaded)

  /commit              Create clean git commits
  /review              Code review for bugs and quality
  /debug               Diagnose and fix bugs
  /plan                Design implementation plans
  /test                Write and run tests
  /simplify            Refactor code for simplicity
  /pdf                 Create and manipulate PDF documents
  /xlsx                Create and manipulate Excel files
  /docx                Create and edit Word documents
  /pptx                Create PowerPoint presentations
  /code-review         Automated PR review (plugin)
  ...
```

---

## 5. 내장 스킬 목록 (37개)

### 개발 도구

| 스킬 | 설명 |
|------|------|
| `commit` | Git 커밋 메시지 작성 및 커밋 생성 |
| `review` | 코드 리뷰 - 버그, 품질, 모범 사례 점검 |
| `debug` | 버그 체계적 진단 및 수정 |
| `plan` | 복잡한 작업의 구현 계획 설계 |
| `test` | 단위/통합/E2E 테스트 작성 및 실행 |
| `simplify` | 코드 리팩토링 및 단순화 |
| `refactor` | 동작 변경 없이 코드 구조 개선 |
| `git-workflow` | Git 브랜치 전략, 머지, 리베이스 |
| `api-design` | RESTful/GraphQL API 설계 |
| `database-migration` | DB 스키마 마이그레이션 관리 |
| `dependency-update` | 패키지 의존성 업데이트/감사 |
| `security-audit` | 보안 취약점 스캔 (OWASP Top 10) |
| `performance` | 코드 성능 프로파일링 및 최적화 |
| `code-documentation` | 문서 자동 생성 (JSDoc, Sphinx) |

### 문서 처리

| 스킬 | 설명 |
|------|------|
| `pdf` | PDF 생성, 읽기, 병합, 분할 |
| `xlsx` | Excel 스프레드시트 생성, 차트, 수식 |
| `docx` | Word 문서 생성 및 편집 |
| `pptx` | PowerPoint 프레젠테이션 생성 |
| `doc-coauthoring` | 문서 공동 작성 |
| `data-analysis` | 데이터 분석, 시각화, 통계 |

### 인프라 / DevOps

| 스킬 | 설명 |
|------|------|
| `docker-setup` | Dockerfile, docker-compose 작성 |
| `ci-cd` | CI/CD 파이프라인 설정 (GitHub Actions) |
| `monitoring` | 로깅, 메트릭, 알림 설정 |
| `deployment` | 배포 (클라우드, 서버리스, 온프레미스) |
| `logging` | 구조화된 로깅 프레임워크 설정 |
| `caching` | 캐싱 전략 구현 (Redis, 인메모리) |
| `auth-setup` | 인증 설정 (JWT, OAuth, 세션) |
| `error-handling` | 에러 핸들링 패턴 구현 |

### 프론트엔드 / 디자인

| 스킬 | 설명 |
|------|------|
| `frontend-design` | UI 컴포넌트 및 레이아웃 디자인 |
| `canvas-design` | HTML5 Canvas 그래픽/시각화 |
| `webapp-testing` | Playwright 웹앱 테스트 |
| `web-artifacts-builder` | 단일 HTML 웹앱 빌드 |
| `theme-factory` | UI 테마 및 색상 팔레트 생성 |
| `algorithmic-art` | 알고리즘/제너러티브 아트 생성 |
| `accessibility` | 웹 접근성 (WCAG) 감사 |

### 기타

| 스킬 | 설명 |
|------|------|
| `skill-creator` | 새 스킬 생성 도우미 (Q&A 방식) |
| `claude-api` | Claude API / Anthropic SDK 사용 |
| `brand-guidelines` | 브랜드 가이드라인 문서 작성 |
| `internal-comms` | 사내 커뮤니케이션 작성 |
| `i18n` | 국제화/지역화 설정 |
| `seo-optimization` | 검색엔진 최적화 |
| `slack-gif-creator` | Slack용 GIF 생성 |

---

## 6. 내장 플러그인 (2개)

### code-review 플러그인

자동 PR 코드 리뷰를 수행합니다. 3개의 전문 에이전트가 병렬로 분석합니다:
- **bug-detector**: 버그, 엣지 케이스, 로직 오류 탐지
- **convention-checker**: 코딩 컨벤션/스타일 준수 점검
- **readability-reviewer**: 코드 가독성 및 문서화 품질 리뷰

```
you> /code-review
```

### agent-sdk-dev 플러그인

Claude Agent SDK 프로젝트 개발 키트입니다:
- `/new-sdk-app` 커맨드로 새 Agent SDK 프로젝트 생성 (Python/TypeScript)
- 프로젝트 구조 검증 에이전트 내장

```
you> /new-sdk-app
```

---

## 7. 모델 목록

### 텍스트/코드 모델

| 키 | 모델명 | 성능 | 용도 |
|----|--------|------|------|
| `qwen3.5-397b` | Qwen3.5-397B-A17B | ★★★ | 복잡한 분석, 대규모 코드 (기본값) |
| `qwen3-coder-480b` | Qwen3-Coder-480B | ★★★ | 코딩 전문 |
| `qwen3-235b-2507` | Qwen3-235B | ★★★ | 범용 대형 |
| `glm-5` | GLM-5 | ★★☆ | 빠른 범용 |
| `gpt-oss-120b` | gpt-oss-120b | ★★☆ | 경량 범용 |
| `qwen3-coder-next` | Qwen3-Coder-Next | ★★☆ | 차세대 코더 |
| `glm-4.7` | GLM-4.7 | ★☆☆ | 초고속 |
| `qwen3.5-35b` | Qwen3.5-35B | ★☆☆ | 초경량 |

### 비전 모델 (이미지 지원)

| 키 | 모델명 | 용도 |
|----|--------|------|
| `qwen3-vl-235b` | VL-235B | 복잡한 이미지 분석 |
| `qwen2.5-vl-72b` | VL-72B | 일반 이미지 분석 |
| `qwen3-vl-30b` | VL-30B | 빠른 이미지 분석 |

### 자동 라우팅

모델을 지정하지 않으면 쿼리 특성에 따라 자동 선택:
- 코딩 키워드 2개 이상 → `qwen3-coder-480b`
- 복잡한 분석 (긴 질문) → `qwen3.5-397b`
- 이미지 포함 → `qwen3-vl-*` (복잡도에 따라)
- 단순 질문 (50자 미만) → `gpt-oss-120b`

### 폴백 체인

선택된 모델 실패 시 자동으로 대체 모델을 순차 시도 (최대 6회):
```
qwen3.5-397b 실패 → qwen3-coder-480b → qwen3-235b → glm-5 → gpt-oss-120b → ...
```

---

## 8. 커스텀 스킬 만들기

```bash
mkdir -p ~/.openharness/skills/my-skill
cat > ~/.openharness/skills/my-skill/SKILL.md << 'EOF'
---
name: my-skill
description: 나만의 커스텀 스킬. TRIGGER when 사용자가 특정 작업을 요청할 때.
---
# My Custom Skill

이 스킬은 다음 작업을 수행합니다:

## 단계
1. 입력을 분석합니다
2. 결과를 생성합니다
3. 피드백을 제공합니다

## 예시
사용자: "데이터를 분석해줘"
→ 이 스킬이 자동으로 활성화됩니다.
EOF
```

스킬은 `oh` 재시작 시 자동으로 인식됩니다.

---

## 9. 커스텀 플러그인 만들기

```bash
# 1. 플러그인 디렉토리 생성
mkdir -p ~/.openharness/plugins/my-plugin/{.claude-plugin,commands,agents}

# 2. 메타데이터 작성
cat > ~/.openharness/plugins/my-plugin/.claude-plugin/plugin.json << 'EOF'
{
  "name": "my-plugin",
  "description": "나만의 플러그인",
  "version": "1.0.0"
}
EOF

# 3. 커맨드 작성
cat > ~/.openharness/plugins/my-plugin/commands/my-command.md << 'EOF'
# My Command
이 커맨드는 특정 작업을 자동화합니다.
## 사용법
/my-command [옵션]
EOF
```

---

## 10. 디렉토리 구조

```
openharness/                      # 프로젝트 루트
├── src/openharness/              # 코어 소스 (~3,000줄)
│   ├── api/                      # TOKEN.TXT 인증, 13개 모델 레지스트리, Provider
│   │   ├── token.py              # 토큰 로딩/검증
│   │   ├── models.py             # 모델 레지스트리 + 폴백 체인
│   │   └── provider.py           # OpenAI-compatible API 클라이언트
│   ├── engine/                   # Agent Loop
│   │   ├── models.py             # Tool, TurnResult, UsageSummary
│   │   ├── registry.py           # ToolRegistry (O(1) 조회)
│   │   ├── router.py             # ToolRouter (토큰 스코어링)
│   │   └── loop.py               # HarnessEngine (턴 루프, 예산, 스트리밍)
│   ├── tools/builtin.py          # 8개 빌트인 도구
│   ├── skills/loader.py          # SKILL.md 파서 + 자동 등록
│   ├── plugins/loader.py         # plugin.json 로더 + 커맨드 등록
│   ├── permissions/__init__.py   # 거부 목록 기반 권한 관리
│   ├── hooks/__init__.py         # 라이프사이클 훅
│   ├── commands/registry.py      # 슬래시 커맨드 시스템
│   ├── memory/                   # 세션, 히스토리, 트랜스크립트
│   └── cli.py                    # CLI 엔트리포인트
├── skills/anthropic/             # 37개 내장 스킬
│   ├── commit/SKILL.md
│   ├── review/SKILL.md
│   ├── pdf/SKILL.md
│   └── ... (37개)
├── plugins/anthropic/            # 2개 내장 플러그인
│   ├── code-review/              # PR 코드 리뷰 (3 에이전트)
│   └── agent-sdk-dev/            # Agent SDK 개발 키트
├── tests/                        # 60개 테스트 (전부 통과)
├── pyproject.toml                # Python 패키지 설정
├── install.sh                    # 설치 스크립트
└── TOKEN.TXT.template            # 토큰 템플릿

~/.openharness/                   # 개인 설정 (홈 디렉토리)
├── TOKEN.TXT                     # API 키
├── skills/                       # 커스텀 스킬
├── plugins/                      # 커스텀 플러그인
└── .harness_sessions/            # 세션 히스토리
```

---

## 11. 트러블슈팅

### TOKEN.TXT 관련
```
⚠️ TOKEN.TXT: missing or empty
```
**해결**: `echo "your-api-key" > ~/.openharness/TOKEN.TXT`

### 모델 연결 실패
```
All 6 models failed. Last error: Connection error
```
**해결**: 내부 LLM 서버 (`dev.hcp.llm.skhynix.com`) 접근 가능한지 확인

### pip 설치 실패 (프록시 환경)
```bash
# PYTHONPATH 방식 사용
export PYTHONPATH="/path/to/openharness/src:$PYTHONPATH"
python -m openharness
```

### 스킬이 로드되지 않음
스킬 디렉토리 구조 확인:
```
skills/my-skill/SKILL.md    (O) 올바름
skills/SKILL.md             (X) 디렉토리 없음
skills/my-skill.md          (X) 파일명이 다름
```

---

## 라이선스

Apache-2.0
