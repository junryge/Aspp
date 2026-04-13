# 하네스 시스템 가이드

## harness_bridge.py — 메인 앱 ↔ 하네스 연결

| 기능 | 설명 |
|------|------|
| `init_harness()` | 서버 시작 시 스킬 레지스트리/라우터 초기화 |
| `harness_route()` | 사용자 질문 → 매칭 스킬 찾기 |
| `save_chat_session()` | 대화 세션 자동 저장 |
| `_feedback_store` | 피드백 저장소 (오답 노트 포함) |
| `suggest_skill_combinations()` | 스킬 조합 추천 |
| `/api/harness/*` | 하네스 관련 API 엔드포인트 등록 |

## harness-mvp/harness/ — 각 모듈 설명

| 파일 | 역할 |
|------|------|
| `feedback.py` | 피드백 루프 — 👎 시 실수 패턴 자동 기록, 다음 요청 시 "과거 실수 반복 금지" 프롬프트 주입 |
| `permissions.py` | 권한 관리 — 도구 이름 차단 + LLM 응답 코드에서 위험 패턴 검사 (DROP TABLE, rm -rf 등) |
| `engine.py` | 실행 엔진 — 스킬 라우팅 → 도구 실행 → 결과 반환, Pre-commit Hook (코드 안전성 검사) |
| `context.py` | 프로젝트 컨텍스트 — 프로젝트별 하네스 맵 로드, 금지사항/규칙/오답노트를 프롬프트에 주입 |
| `registry.py` | 스킬 등록소 — 355+개 스킬 관리, 이름/키워드로 검색 |
| `router.py` | 라우터 — 사용자 질문을 토큰화 → 스킬 매칭 점수 계산 → 상위 N개 반환 |
| `session.py` | 세션 관리 — 대화 내역 JSON 저장/로드 |
| `transcript.py` | 트랜스크립트 — 대화 기록 압축/재생 |
| `history.py` | 히스토리 — 이벤트 로그 (마크다운 렌더링) |
| `models.py` | 데이터 모델 — TurnResult, PermissionDenial, UsageSummary 등 |

## 데이터 흐름

```
사용자 질문
  ↓
harness_bridge → router.py (스킬 매칭)
  ↓
feedback.py → 오답 노트 프롬프트 주입
context.py → 프로젝트 규칙 프롬프트 주입
  ↓
LLM 응답 생성
  ↓
permissions.py → 금지 패턴 검사
engine.py → Pre-commit Hook
  ↓
feedback.py → 품질 점수 저장
session.py → 세션 저장
```

## 주요 기능 상세

### 1. 오답 노트 자동화 (feedback.py)
- 사용자가 👎 누르면 실수 패턴 자동 기록
- `add_mistake(query, error_pattern)` → JSON 저장 (최근 20개 유지)
- 다음 요청 시 `build_mistakes_prompt()` → 시스템 프롬프트에 "과거 실수 반복 금지" 주입
- `build_prompt_hint()` → 스킬별 승인율/반려사유 기반 힌트 제공

### 2. 금지 사항 강제 (permissions.py)
- `CodeGuard` 클래스: LLM 응답 코드에서 위험 패턴 검사
- 기본 금지 패턴: DROP TABLE, DELETE FROM, rm -rf, 민감정보 하드코딩 등
- `load_from_harness_map()` → 하네스 맵에서 추가 금지 규칙 자동 파싱
- `check(code)` → 위반 목록 반환, `is_safe(code)` → 안전 여부 판단

### 3. Pre-commit Hook (engine.py)
- `pre_commit_check(code, harness_map)` → 코드 실행/커밋 전 안전성 검사
- CodeGuard + 하네스 맵 규칙 결합
- 위반 시 차단 + 경고 메시지 반환

### 4. 프로젝트별 맵 관리 (context.py)
- `harness_maps/` 폴더에 프로젝트별 .md 파일 저장
- `load_project_map("smartATLAS")` → 해당 프로젝트 규칙 로드
- `build_project_prompt()` → 금지사항/규칙/오답노트를 시스템 프롬프트에 주입
- `list_project_maps()` → 사용 가능한 프로젝트 목록 반환

## 프로젝트 하네스 맵 예시 (harness_maps/smartATLAS.md)

```markdown
# 🐎 Project Harness: smartATLAS

## 1. 프로젝트 개요
- **목적:** ATLAS 소스 개발/운영 동일화
- **기술 스택:** Python, Flask, OHT 통신

## 2. 🚫 절대 금지 사항
- DB 테이블이나 컬럼을 임의로 삭제(Drop) 하지 마시오.
- `.env` 파일 등 민감한 환경 변수를 코드에 하드코딩하지 마시오.
- OHT 통신 프로토콜을 임의 변경하지 마시오.

## 3. ⚙️ 작업 규칙
- 코드는 반드시 Linter를 통과해야 합니다.
- API 호출에는 반드시 에러 핸들링을 포함하시오.

## 4. 📝 오답 노트
- [2026-04-13] API 호출 시 예외 처리 생략하여 크래시 발생. 모든 API 호출에 try-catch 필수.
```

## API 엔드포인트

| Method | URL | 설명 |
|--------|-----|------|
| GET | `/api/harness/skills` | 스킬 목록/검색 |
| POST | `/api/harness/route` | 질문 → 스킬 매칭 |
| POST | `/api/harness/reload` | 레지스트리 재초기화 |
| GET/POST/DELETE | `/api/harness/session/*` | 세션 CRUD |
| GET | `/api/harness/history` | 이벤트 로그 |
| POST | `/api/harness/suggest-combo` | 스킬 조합 추천 |
| POST | `/api/harness/validate-combo` | 스킬 조합 검증 |
| POST | `/api/harness/optimize-groups` | 스킬 병렬 그룹 최적화 |
| GET | `/api/harness/status` | 전체 상태 |
| POST | `/api/harness/expert-pool` | 동적 에이전트 선택 |
| POST/GET | `/api/harness/feedback/*` | 피드백 저장/조회/힌트 |

## 데이터 저장 위치

| 항목 | 경로 |
|------|------|
| 세션 | `.harness_sessions/{session_id}.json` |
| 피드백 | `.harness_sessions/feedback/feedback_{skill_id}.json` |
| 오답 노트 | `.harness_sessions/feedback/mistakes_log.json` |
| 프로젝트 맵 | `harness_maps/{project_name}.md` |
| 히스토리 | 메모리 (세션 중) |
