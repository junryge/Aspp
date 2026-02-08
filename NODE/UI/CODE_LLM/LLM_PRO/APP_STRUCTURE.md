# Nomos LLM Desktop App - 구조 문서

## 개요

PySide6 기반 데스크탑 LLM 코딩 어시스턴트.
API 우선(SK Hynix 폐쇄망), GGUF 폴백. Aider를 핵심 코드 수정 엔진으로 사용.

## 실행 방법

```bash
python -m app.main       # 직접 실행
run_app.bat              # 배치 파일 실행
```

---

## 디렉토리 구조

```
F:\M14_Q\LLM_PRO\
│
├── app/                          # 메인 애플리케이션
│   ├── main.py                   #  99줄  앱 진입점 (QApplication, 초기화)
│   ├── main_window.py            # 441줄  메인 윈도우 (레이아웃, 시그널 연결)
│   │
│   ├── core/                     # --- 백엔드 코어 ---
│   │   ├── config.py             # 128줄  환경 설정, 토큰 로딩, GGUF 모델 목록
│   │   ├── llm_provider.py       # 205줄  API + GGUF 통합 LLM 호출
│   │   ├── prompt_builder.py     #  85줄  시스템 프롬프트 (8가지 모드)
│   │   ├── self_correction.py    #  63줄  자기 교정 루프
│   │   └── gguf_server.py        # 196줄  aider용 OpenAI 호환 HTTP 서버 (port 10002)
│   │
│   ├── aider/                    # --- Aider 코드 수정 엔진 ---
│   │   ├── bridge.py             # 405줄  AiderBridge (코드 수정 핵심)
│   │   ├── project_manager.py    # 310줄  프로젝트 CRUD, 파일 관리
│   │   └── git_ops.py            #  95줄  Git: diff, stash, approve/reject, history
│   │
│   ├── agent/                    # --- 에이전트 (확장) ---
│   │   ├── sk_provider.py        # 149줄  SKHynixProvider (nanobot용)
│   │   ├── nanobot_manager.py    # 181줄  마기 에이전트 매니저
│   │   └── tools.py              # 141줄  ValidateCodeTool, SaveCodeTool
│   │
│   └── ui/                       # --- UI 위젯 ---
│       ├── theme.py              # 295줄  다크 테마 (Catppuccin 스타일 QSS)
│       ├── sidebar.py            # 146줄  모드 버튼 (대화/생성/수정/분석)
│       ├── header.py             # 170줄  상태 표시, 환경 전환, 모델 선택
│       ├── chat_panel.py         # 410줄  AI 응답 + 입력창 + 마크다운 렌더링
│       ├── code_editor.py        # 370줄  Pygments 구문 강조 + 줄번호
│       ├── project_panel.py      # 401줄  파일 트리 (체크박스 + 초록 선택)
│       ├── diff_viewer.py        # 176줄  Diff 표시 + 승인/거절
│       ├── dialogs.py            # 166줄  프로젝트 생성, 언어 변환 다이얼로그
│       └── workers.py            # 101줄  QThread 워커 (LLM/Aider 비동기)
│
├── aider_projects/               # 런타임 생성 - 프로젝트 저장소
│   ├── projects.json             # 프로젝트 레지스트리
│   ├── code_editing/             # 코드 편집 프로젝트들
│   └── data_analysis/            # 데이터 분석 프로젝트들
│
├── Qwen3-14B-Q4_K_M.gguf        # 8.4 GB  Qwen3 14B (Q4_K_M 양자화)
├── Qwen3-8B-Q6_K.gguf           # 6.3 GB  Qwen3 8B  (Q6_K 양자화)
├── qwen3-1.7b-q8_0.gguf         # 2.0 GB  Qwen3 1.7B (Q8_0 양자화)
│
├── token.txt                     # API 토큰 (회사에서 배치)
├── requirements.txt              # Python 의존성
├── run_app.bat                   # 실행 배치 파일
```

**총 26개 Python 파일 | 약 4,800줄**

---

## 핵심 모듈 설명

### 1. `app/main.py` - 앱 진입점

```
QApplication 생성 → AppConfig 로딩 → LLMProvider 초기화
→ GGUF 서버 시작 (port 10002) → AiderBridge 생성
→ MainWindow 표시
```

### 2. `app/main_window.py` - 메인 윈도우

**레이아웃:**
```
+-- 사이드바 (220px) --+-- 메인 영역 ---------------------------+
| 💬 대화              | [헤더: 상태 / 환경 / 모델 / 토큰]       |
| ⚡ 코드 생성          |                                        |
| 🔧 코드 수정          | 페이지 0: [채팅 패널]     ← 대화/생성    |
| 📊 분석              |                                        |
|                     | 페이지 1: [채팅] | [프로젝트패널]  ← 수정/분석 |
|                     |           |     | [Diff 뷰어]           |
+---------------------+--------+-----+--------------------------+
```

**모드 전환:**
- 대화/코드 생성 → 페이지 0 (채팅만)
- 코드 수정/분석 → 페이지 1 (채팅 + 프로젝트 + Diff)

### 3. `app/core/config.py` - 설정

| 항목 | 설명 |
|------|------|
| `ENV_CONFIG` | dev/prod/common/local 4개 환경 |
| `load_token()` | token.txt 탐색 (로컬 → 상위 → 홈) |
| `AVAILABLE_GGUF_MODELS` | Qwen3 14B/8B/1.7B 모델 정의 |
| `llm_mode` | "api" 또는 "gguf" |

### 4. `app/core/llm_provider.py` - LLM 호출

```
call_llm(prompt, system_prompt)
  ├── API 모드 → call_llm_api() → SK Hynix API 호출
  └── GGUF 모드 → call_local_llm() → llama-cpp 로컬 모델
```

### 5. `app/aider/bridge.py` - Aider 코드 수정

```
chat(project_id, message, selected_files)
  ├── API 모드 → aider Python API 사용
  └── GGUF 모드 → GGUF 서버(10002) 경유 → aider 호출

결과 → git stash → proposal_id 발급
  ├── 승인 → stash pop (변경 적용)
  └── 거절 → stash drop (변경 폐기)
```

### 6. `app/ui/project_panel.py` - 프로젝트 패널

- 파일 트리에 **체크박스** → 체크된 파일만 수정/분석 대상
- 체크된 파일은 **초록색 강조** 표시
- 체크 안 하면 프로젝트 전체 파일 대상
- 우클릭: 새 파일/폴더, 이름 변경, 삭제

---

## 데이터 흐름

### 일반/코드 생성 모드
```
사용자 입력 → ChatPanel.send_requested
  → MainWindow._on_chat_send()
  → LLMWorker(QThread) 실행
  → LLMProvider.call_llm()
  → 결과 → ChatPanel.show_response()
```

### 코드 수정/분석 모드
```
사용자 입력 → ChatPanel.send_requested
  → MainWindow._on_aider_send()
  → 체크된 파일 목록 수집 (get_checked_files)
  → AiderWorker(QThread) 실행
  → AiderBridge.chat()
  → 결과 → ChatPanel.show_response() + DiffViewer.show_diff()
  → 사용자 승인/거절 → git stash pop/drop
```

---

## 환경 설정

| 환경 | 설명 | LLM 모드 |
|------|------|----------|
| 개발 (dev) | SK Hynix 개발 API | API |
| 운영 (prod) | SK Hynix 운영 API | API |
| 공통 (common) | SK Hynix 공통 API | API |
| 로컬 (local) | GGUF 로컬 모델 | GGUF |

**토큰 탐색 순서:** `./token.txt` → `../token.txt` → `~/token.txt`

---

## 의존성

| 패키지 | 용도 |
|--------|------|
| PySide6 | Qt GUI 프레임워크 |
| Pygments | 코드 구문 강조 |
| markdown | 마크다운 → HTML 렌더링 |
| llama-cpp-python | GGUF 모델 로딩/추론 |
| aider-chat | 코드 수정 엔진 |
| requests | API 호출 |

---

## GGUF 서버 (port 10002)

aider는 OpenAI API 형식만 지원하므로, 로컬 GGUF 모델 사용 시
최소 OpenAI 호환 HTTP 서버를 내장 실행:

- `POST /v1/chat/completions` - 채팅 완성 (SSE 스트리밍)
- `GET /v1/models` - 모델 목록

데몬 스레드로 자동 시작/종료.
