"""
demos_v1/hermes/engine.py — 헤르메스 오케스트레이션

1) build_system_prompt(user_id, query): 기존 프롬프트에 더할 헤르메스 블록 조립
   - 메모리 스냅샷(MEMORY/USER)
   - 개인 스킬 인덱스 + 질의 관련 스킬 본문(회상)
   - 텍스트 프로토콜 사용 지침 (memory / skill / ask)
2) apply_response(user_id, answer): 응답 내 프로토콜 블록 처리
   - memory: 즉시 저장 (즉시 반영 결정)
   - skill: 확인형 → 저장 안 하고 'pending'으로 사용자에게 승인 요청 반환
   - ask: 되묻기 질문 추출
   반환: {clean, memory_results, pending_skills, questions, snapshot_changed}
"""
from __future__ import annotations

from demos_v1.hermes import memory, skills, protocol, counters, builtin, sessions

# 텍스트 프로토콜 사용 지침 (시스템 프롬프트에 주입)
PROTOCOL_GUIDE = """\
=== 헤르메스 능력 (텍스트 프로토콜) ===
너는 대화에서 배우고 기억하는 에이전트다. 아래 블록을 답변 끝에 덧붙여 사용한다.
(블록은 사용자에게 안 보이게 시스템이 처리한다. 남발 금지 — 정말 가치 있을 때만.)

1) 대화에서 기억할 정보가 나오면 저장한다. 두 종류를 반드시 구분한다:
   · store: memory = 환경·프로젝트 "사실" — 담당 FAB/라인, 시스템·도구, 데이터 종류, 제약, 도메인 용어 등
   · store: user   = 사용자 "선호" — 답변 형식/말투/언어/길이 등
   ★ 사용자가 자신의 역할·담당·사용 시스템·도구·데이터·도메인을 드러내면 → 반드시 store: memory 로 저장한다.

예) 환경·프로젝트 사실(memory):
```hermes:memory
store: memory
action: add
text: 사용자는 M16_BR FAB의 OHT 반송 시스템(OHS) 정체를 분석한다
```
예) 사용자 선호(user):
```hermes:memory
store: user
action: add
text: 사용자는 답변을 표로 정리하는 것을 선호한다
```
- 선언형만(명령형 "항상 ~하라" 금지), 한 문장. 절차·방법은 메모리가 아니라 스킬로.
- 새 사실/선호가 나올 때마다 적극적으로 저장하되, 중복·사소한 잡담은 생략.
- action: add(신규) | replace(target=기존 일부) | remove(target=기존 일부)

2) 재사용 가치 있는 절차/해법을 발견하면(여러 단계 작업 완료·까다로운 오류 해결·비자명 워크플로):
```hermes:skill
action: create         # create | patch
name: <소문자-하이픈 클래스명>   # 일회성/날짜 이름 금지
when: <언제 쓰는 스킬인지 한 줄>
body: |
  1. ...
  2. ...
```
- 저장 전 사용자 승인을 받는다(시스템이 처리).

3) 요청이 모호하면 추측하지 말고 먼저 되묻는다:
```hermes:ask
- <핵심 질문1>
- <핵심 질문2>
```
"""


def build_system_prompt(user_id: str, query: str = "") -> str:
    """기존 시스템 프롬프트 뒤에 붙일 헤르메스 블록. 비었으면 빈 문자열."""
    parts = []

    mem = memory.snapshot(user_id)
    if mem:
        parts.append(mem)

    idx = skills.index_text(user_id)
    if idx:
        parts.append(idx)

    if query:
        recalled = skills.recall(user_id, query, top_k=2)
        if recalled:
            bodies = "\n\n".join(
                f"--- 스킬: {r['name']} ---\n{r['body']}" for r in recalled
            )
            parts.append("=== 관련 개인 스킬 본문 ===\n" + bodies)

        # 빌트인 스킬 팩 (Hermes 영감 — task-planning / debugging / data-analysis / verify)
        bi = builtin.recall_builtin(query, top_k=2)
        if bi:
            bbodies = "\n\n".join(f"[{r['name']}] {r['desc']}\n{r['body']}" for r in bi)
            parts.append("=== 권장 작업 방식 (헤르메스 빌트인 스킬) ===\n" + bbodies)

        # 지난 대화(과거 세션) 회상 — 사용자가 이전 대화를 물으면 근거가 되게 관련 기록 주입.
        # (기억 스냅샷=사실 요약과 별개로, 실제 대화 원문을 검색해 넣는다.)
        try:
            hits = sessions.search(user_id, query, max_hits=3)
        except Exception:
            hits = []
        if hits:
            lines = []
            for h in hits:
                d = h.get("date", "")
                for c in h.get("context", []):
                    role = "사용자" if c.get("role") == "user" else "에이전트"
                    txt = (c.get("content") or "").strip().replace("\n", " ")
                    if txt:
                        lines.append(f"[{d}] {role}: {txt[:200]}")
            if lines:
                parts.append(
                    "=== 지난 대화 관련 기록 (과거 세션 검색) ===\n"
                    + "\n".join(lines[:20])
                    + "\n(참고용 과거 대화다. 사용자가 '지난 대화/전에 얘기한 것'을 물으면 이 기록을 근거로 답한다.)"
                )

    parts.append(PROTOCOL_GUIDE)
    return "\n\n".join(parts)


def apply_response(user_id: str, answer: str) -> dict:
    """응답에서 프로토콜 블록을 처리. 사용자 표시용 clean 본문과 결과 반환."""
    clean, blocks = protocol.parse_blocks(answer)
    out = {
        "clean": clean,
        "memory_results": [],     # [{ok, msg, action, store}]
        "pending_skills": [],     # [{name, when, body, action, find, replace}]  ← 승인 대기
        "questions": [],          # 되묻기 질문
        "snapshot_changed": False,
    }

    for b in blocks:
        if b["kind"] == "memory":
            action = b.get("action", "add")
            sname = b.get("store", "memory")
            if action == "add":
                ok, msg = memory.add(user_id, sname, b.get("text", ""))
            elif action == "replace":
                ok, msg = memory.replace(user_id, sname, b.get("target", ""), b.get("text", ""))
            elif action == "remove":
                ok, msg = memory.remove(user_id, sname, b.get("target", ""))
            else:
                ok, msg = False, f"알 수 없는 메모리 액션: {action}"
            out["memory_results"].append({"ok": ok, "msg": msg, "action": action, "store": sname})
            if ok and msg not in ("이미 존재 (스킵)",):
                out["snapshot_changed"] = True   # 즉시 반영 트리거

        elif b["kind"] == "skill":
            # 확인형: 저장하지 않고 승인 대기 목록에 적재
            out["pending_skills"].append({
                "action": b.get("action", "create"),
                "name": b.get("name", ""),
                "when": b.get("when", ""),
                "body": b.get("body", ""),
                "find": b.get("find", ""),
                "replace": b.get("replace", ""),
            })

        elif b["kind"] == "ask":
            out["questions"].extend(b.get("questions", []))

    return out


def confirm_skill(user_id: str, spec: dict) -> tuple[bool, str]:
    """사용자가 승인한 스킬 저장을 실제 실행. skill_manage 실호출 → 카운터 리셋."""
    action = (spec.get("action") or "create").lower()
    name = spec.get("name", "")
    if action == "create":
        ok, msg = skills.create(user_id, name, spec.get("when", ""), spec.get("body", ""))
    elif action == "patch":
        ok, msg = skills.patch(user_id, name, spec.get("find", ""), spec.get("replace", ""))
    elif action == "edit":
        ok, msg = skills.edit(user_id, name, spec.get("when", ""), spec.get("body", ""))
    elif action == "delete":
        ok, msg = skills.delete(user_id, name)
    else:
        return False, f"알 수 없는 스킬 액션: {action}"
    if ok:
        counters.reset_skill(user_id)
    return ok, msg
