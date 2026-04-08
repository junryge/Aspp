# OpenHarness - Open Agent Harness

팀원 개인별 CLI AI 에이전트 하네스 (45개 스킬 + 2개 플러그인 내장)

---

## 1. 설치 방법 (Windows)

### 방법 1: install.bat 더블클릭 (가장 쉬움)

```
1. openharness.zip 압축 해제
2. openharness 폴더 열기
3. install.bat 더블클릭
4. 설치 완료 후 CMD 열고 oh 입력
```

### 방법 2: PowerShell 설치

```powershell
# 1. 압축 해제
Expand-Archive openharness.zip -DestinationPath .

# 2. 설치 실행
cd openharness
powershell -ExecutionPolicy Bypass -File install.ps1
```

### 방법 3: pip 수동 설치

```cmd
cd openharness
pip install -e .
mkdir %USERPROFILE%\.openharness
mkdir %USERPROFILE%\.openharness\skills
mkdir %USERPROFILE%\.openharness\plugins
mkdir %USERPROFILE%\.openharness\sessions
```

### 방법 4: PYTHONPATH 직접 사용 (pip 없이)

```cmd
cd openharness
set PYTHONPATH=%CD%\src;%PYTHONPATH%
python -m openharness
```

### Linux/Mac 사용자

```bash
unzip openharness.zip && cd openharness && bash install.sh
```

---

## 2. API 키 설정 (필수)

LLM 서버 연결을 위해 TOKEN.TXT에 API 키를 넣어야 합니다.

### Windows 경로

```
C:\Users\{사용자이름}\.openharness\TOKEN.TXT
```

### 키 입력 방법

**방법 A: 메모장으로 직접 편집**
```cmd
notepad %USERPROFILE%\.openharness\TOKEN.TXT
```
→ API 키를 붙여넣고 저장

**방법 B: CMD에서 입력**
```cmd
echo sk-your-api-key-here > %USERPROFILE%\.openharness\TOKEN.TXT
```

**방법 C: PowerShell**
```powershell
"sk-your-api-key-here" | Out-File -FilePath "$env:USERPROFILE\.openharness\TOKEN.TXT" -Encoding ascii
```

> TOKEN.TXT가 없거나 비어있으면 시작 시 경고가 표시되고 API 호출이 불가합니다.

---

## 3. 사용법

### 3-1. 인터랙티브 모드 (기본)

CMD 또는 PowerShell에서:

```cmd
oh
```

```
  ╔══════════════════════════════════════════╗
  ║     OpenHarness  v0.1.0                  ║
  ║     Open Agent Harness for Teams         ║
  ╚══════════════════════════════════════════╝

  🔑 TOKEN.TXT: loaded (32 chars)
  🤖 Default model: PROD (397B)
  🔧 Tools: 53 (45 skills, 2 plugins)
  💬 Type /help for commands, /exit to quit

you> 이 코드를 리뷰해주세요
```

### 3-2. 모델 지정

```cmd
oh --model qwen3-coder-480b     # 코딩 전문 (480B)
oh --model glm-5                # 빠른 범용
oh --model qwen3-vl-235b        # 비전 모델 (이미지)
oh --model gpt-oss-120b         # 경량 모델
oh --model qwen3.5-397b         # 최고 성능 (기본값)
```

### 3-3. 헤드리스 모드 (단일 명령)

```cmd
oh run "Python으로 퀵소트 구현해줘"
oh run "이 코드의 버그를 찾아줘" --json
oh run "테스트를 작성하고 실행해줘" --max-turns 5
```

### 3-4. 상태 확인

```cmd
oh status     # 시스템 상태
oh models     # 모델 목록
oh skills     # 스킬 목록
oh plugins    # 플러그인 목록
```

---

## 4. 슬래시 커맨드

| 커맨드 | 설명 |
|--------|------|
| `/help` | 사용 가능한 커맨드 전체 목록 |
| `/models` | LLM 모델 목록 |
| `/model [name]` | 모델 확인/변경 |
| `/skills` | 스킬 목록 |
| `/plugins` | 플러그인 목록 |
| `/status` | 시스템 상태 |
| `/commit` | Git 커밋 생성 |
| `/review` | 코드 리뷰 |
| `/debug` | 버그 진단 |
| `/plan` | 구현 계획 |
| `/test` | 테스트 작성 |
| `/simplify` | 코드 리팩토링 |
| `/pdf` | PDF 생성/분석 |
| `/xlsx` | Excel 처리 |
| `/code-review` | 자동 PR 코드 리뷰 (플러그인) |
| `/new-sdk-app` | Agent SDK 프로젝트 생성 (플러그인) |
| `/clear` | 화면 지우기 |
| `/exit` | 종료 |

---

## 5. 내장 스킬 목록 (45개)

### 개발 도구 (14개)

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

### 문서 처리 (6개)

| 스킬 | 설명 |
|------|------|
| `pdf` | PDF 생성, 읽기, 병합, 분할 |
| `xlsx` | Excel 스프레드시트 생성, 차트, 수식 |
| `docx` | Word 문서 생성 및 편집 |
| `pptx` | PowerPoint 프레젠테이션 생성 |
| `doc-coauthoring` | 문서 공동 작성 |
| `data-analysis` | 데이터 분석, 시각화, 통계 |

### 인프라 / DevOps (10개)

| 스킬 | 설명 |
|------|------|
| `docker-setup` | Dockerfile, docker-compose 작성 |
| `ci-cd` | CI/CD 파이프라인 설정 (GitHub Actions) |
| `monitoring` | 로깅, 메트릭, 알림 설정 |
| `deployment` | 배포 (클라우드, 서버리스, 온프레미스) |
| `logging` | 구조화된 로깅 프레임워크 설정 |
| `caching` | 캐싱 전략 (Redis, 인메모리) |
| `auth-setup` | 인증 설정 (JWT, OAuth, 세션) |
| `error-handling` | 에러 핸들링 패턴 구현 |
| `session-start-hook` | 프로젝트 초기화 훅 설정 |
| `update-config` | settings.json 설정 관리 |

### 프론트엔드 / 디자인 (8개)

| 스킬 | 설명 |
|------|------|
| `frontend-design` | UI 컴포넌트 및 레이아웃 디자인 |
| `canvas-design` | HTML5 Canvas 그래픽/시각화 |
| `webapp-testing` | Playwright 웹앱 테스트 |
| `web-artifacts-builder` | 단일 HTML 웹앱 빌드 |
| `theme-factory` | UI 테마 및 색상 팔레트 생성 |
| `algorithmic-art` | 알고리즘/제너러티브 아트 생성 |
| `accessibility` | 웹 접근성 (WCAG) 감사 |
| `seo-optimization` | 검색엔진 최적화 |

### 기타 (7개)

| 스킬 | 설명 |
|------|------|
| `skill-creator` | 새 스킬 생성 도우미 (Q&A 방식) |
| `claude-api` | Claude API / Anthropic SDK 사용 |
| `brand-guidelines` | 브랜드 가이드라인 문서 작성 |
| `internal-comms` | 사내 커뮤니케이션 작성 |
| `mcp-builder` | MCP 서버 빌드 가이드 |
| `i18n` | 국제화/지역화 설정 |
| `slack-gif-creator` | Slack용 GIF 생성 |
| `loop` | 프롬프트/커맨드 반복 실행 |

---

## 6. 내장 플러그인 (2개)

### code-review 플러그인

자동 코드 리뷰. 3개 전문 에이전트가 병렬 분석:
- **bug-detector**: 버그, 엣지 케이스, 로직 오류 탐지
- **convention-checker**: 코딩 컨벤션/스타일 준수 점검
- **readability-reviewer**: 가독성 및 문서화 품질 리뷰

```
you> /code-review
```

### agent-sdk-dev 플러그인

Claude Agent SDK 프로젝트 개발 키트:
- `/new-sdk-app` → 새 Agent SDK 프로젝트 생성 (Python/TypeScript)
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

모델 미지정 시 쿼리에 따라 자동 선택:
- 코딩 키워드 2개+ → `qwen3-coder-480b`
- 복잡한 분석 → `qwen3.5-397b`
- 이미지 포함 → `qwen3-vl-*`
- 단순 질문 → `gpt-oss-120b`

### 폴백 체인

모델 실패 시 자동 대체 (최대 6회):
```
qwen3.5-397b 실패 → qwen3-coder-480b → qwen3-235b → glm-5 → gpt-oss-120b → ...
```

---

## 8. 커스텀 스킬 추가

```cmd
:: Windows
mkdir %USERPROFILE%\.openharness\skills\my-skill
notepad %USERPROFILE%\.openharness\skills\my-skill\SKILL.md
```

SKILL.md 형식:
```yaml
---
name: my-skill
description: 나만의 커스텀 스킬. TRIGGER when 특정 조건.
---
# My Custom Skill

여기에 지침을 작성합니다.
```

`oh` 재시작 시 자동 인식됩니다.

---

## 9. 커스텀 플러그인 추가

```cmd
:: Windows - 플러그인 폴더 구조
mkdir %USERPROFILE%\.openharness\plugins\my-plugin\.claude-plugin
mkdir %USERPROFILE%\.openharness\plugins\my-plugin\commands
```

plugin.json:
```json
{
  "name": "my-plugin",
  "description": "나만의 플러그인",
  "version": "1.0.0"
}
```

---

## 10. 폴더 구조

```
openharness\                       # 프로젝트 루트
├── src\openharness\               # 코어 소스 (~3,000줄)
│   ├── api\                       # TOKEN.TXT 인증, 13개 모델, Provider
│   ├── engine\                    # Agent Loop (registry, router, engine)
│   ├── tools\builtin.py           # 8개 빌트인 도구
│   ├── skills\loader.py           # SKILL.md 파서
│   ├── plugins\loader.py          # plugin.json 로더
│   ├── permissions\               # 권한 관리
│   ├── hooks\                     # 라이프사이클 훅
│   ├── commands\registry.py       # 슬래시 커맨드
│   ├── memory\                    # 세션, 히스토리
│   └── cli.py                     # CLI 엔트리포인트
├── skills\anthropic\              # 45개 내장 스킬
├── plugins\anthropic\             # 2개 내장 플러그인
│   ├── code-review\               # PR 코드 리뷰 (3 에이전트)
│   └── agent-sdk-dev\             # Agent SDK 개발 키트
├── tests\                         # 60개 테스트
├── pyproject.toml                 # Python 패키지 설정
├── install.bat                    # Windows CMD 설치 스크립트
├── install.ps1                    # PowerShell 설치 스크립트
├── install.sh                     # Linux/Mac 설치 스크립트
└── TOKEN.TXT.template             # 토큰 템플릿

%USERPROFILE%\.openharness\        # 개인 설정 (Windows)
├── TOKEN.TXT                      # API 키
├── skills\                        # 커스텀 스킬
├── plugins\                       # 커스텀 플러그인
└── .harness_sessions\             # 세션 히스토리
```

---

## 11. 트러블슈팅

### TOKEN.TXT 관련
```
⚠ TOKEN.TXT: missing or empty
```
**해결**:
```cmd
echo your-api-key > %USERPROFILE%\.openharness\TOKEN.TXT
```

### 모델 연결 실패
```
All 6 models failed. Last error: Connection error
```
**해결**: 내부 LLM 서버 (`dev.hcp.llm.skhynix.com`) 접근 가능한지 확인

### pip 설치 실패
```cmd
:: PYTHONPATH 방식 사용
set PYTHONPATH=C:\path\to\openharness\src;%PYTHONPATH%
python -m openharness
```

### oh 명령어를 못 찾음
```cmd
:: Python Scripts 폴더를 PATH에 추가
set PATH=%USERPROFILE%\AppData\Local\Programs\Python\Python310\Scripts;%PATH%
```

### 스킬이 로드되지 않음
디렉토리 구조 확인:
```
skills\my-skill\SKILL.md    (O) 올바름
skills\SKILL.md             (X) 디렉토리 없음
skills\my-skill.md          (X) 파일명 다름
```

---

## 라이선스

Apache-2.0
