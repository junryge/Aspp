# -*- coding: utf-8 -*-
"""
SENTINEL 4-LLM 파이프라인 — 모델 교체 / 타임아웃 / JSON 파싱 수정본
====================================================================
그대로 쓰거나, MODELS / TIMEOUT / call_llm / extract_json 만 기존 코드에 옮겨 붙이면 됩니다.

수정 내용
  1) 모델 교체
     - 2차: gaia-GLM-5.1                                    → gaia-Qwen3.6-35B-A3B   (HTTP 400 회피)
     - 3차: gaia-solution-Qwen3-235B-A22B-Instruct-2507-FP8  → gaia-cc-gpt-oss-120b   (HTTP 403 회피)
  2) 최종 단계 타임아웃 90초 → 300초
  3) JSON 파싱 실패 보정
     - GLM/Qwen 계열이 앞에 붙이는 <think>...</think> 추론 블록 제거
     - ```json ... ``` 코드펜스 제거
     - 앞뒤 잡설 속에서 JSON 본문만 추출(중괄호 균형 스캔)
     - 트레일링 콤마 제거
     - response_format=json_object 우선 시도, 거부되면 자동 재시도
  4) 모델별 폴백: 400/403/타임아웃이면 다음 후보 모델로 자동 재시도
"""

import json
import re
import time
from pathlib import Path

import requests

# ── 접속 정보 ────────────────────────────────────────────────────────────────
API_BASE = "https://여기에_서버주소/v1"          # ← 기존 코드의 주소로 바꾸세요
API_KEY_FILE = Path(__file__).with_name("API KEY.TXT")
API_KEY = API_KEY_FILE.read_text(encoding="utf-8").strip() if API_KEY_FILE.exists() else ""

# ── 단계별 모델 (팀 권한 있는 모델만) ────────────────────────────────────────
MODELS = {
    "stage1": "gaia-GLM-5.2",            # 1차 데이터 훑기      (정상 동작 확인됨)
    "stage2": "gaia-Qwen3.6-35B-A3B",    # 2차 원인·전파 분석   ← 변경 (기존 gaia-GLM-5.1)
    "stage3": "gaia-cc-gpt-oss-120b",    # 3차 교차 검증        ← 변경 (기존 gaia-solution-...)
    "final":  "gaia-Qwen3.5-397B-A17B",  # 최종 통합 판정       (정상 동작 확인됨)
}

# 위 모델이 400/403/타임아웃이면 순서대로 자동 재시도 (검증된 모델을 뒤에 배치)
FALLBACK = {
    "stage1": ["gaia-Qwen3.6-35B-A3B", "gaia-cc-gpt-oss-120b", "gaia-Qwen3.5-397B-A17B"],
    "stage2": ["gaia-cc-gpt-oss-120b", "gaia-GLM-5.2", "gaia-Qwen3.5-397B-A17B"],
    "stage3": ["gaia-lst-gpt-oss-120b", "gaia-Qwen3.5-397B-A17B", "gaia-GLM-5.2"],
    "final":  ["gaia-GLM-5.2", "gaia-cc-gpt-oss-120b"],
}

# 단계별 타임아웃(초) — 최종 판정은 입력이 커서 넉넉히
TIMEOUT = {"stage1": 120, "stage2": 120, "stage3": 120, "final": 300}


# ── JSON 파싱 보정 ───────────────────────────────────────────────────────────
_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_THINK_TAG = re.compile(r"</?think>", re.I)               # 닫는 태그가 잘린 경우 태그만 제거
_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.S)


def extract_json(text):
    """모델 출력에서 JSON 만 뽑아 파싱. 실패하면 ValueError."""
    if not text:
        raise ValueError("빈 응답")
    s = str(text)

    # 1) 추론 블록 제거 (GLM/Qwen 계열이 앞에 붙임).
    #    닫는 태그가 잘린 경우엔 태그만 지우고 본문은 남겨 아래 스캔이 JSON 을 찾게 한다.
    s = _THINK.sub("", s)
    s = _THINK_TAG.sub("", s)

    # 2) 코드펜스 안이 있으면 그 내용 우선
    m = _FENCE.search(s)
    if m:
        s = m.group(1)

    s = s.strip()

    # 3) 그대로 파싱 시도
    try:
        return json.loads(s)
    except Exception:
        pass

    # 4) 잡설 속에서 첫 JSON 객체/배열만 괄호 균형으로 잘라내기.
    #    객체({)와 배열([) 중 더 앞에 나오는 쪽을 먼저 시도한다.
    pairs = [("{", "}"), ("[", "]")]
    pairs.sort(key=lambda p: (s.find(p[0]) if s.find(p[0]) >= 0 else 10 ** 9))
    for opener, closer in pairs:
        start = s.find(opener)
        if start < 0:
            continue
        depth, in_str, esc = 0, False, False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    chunk = s[start:i + 1]
                    chunk = re.sub(r",(\s*[}\]])", r"\1", chunk)   # 트레일링 콤마 제거
                    try:
                        return json.loads(chunk)
                    except Exception:
                        break
    raise ValueError("JSON 파싱 실패: " + s[:200].replace("\n", " "))


# ── LLM 호출 ────────────────────────────────────────────────────────────────
JSON_RULE = ("반드시 JSON 하나만 출력하세요. 설명·인사말·코드펜스(```)·"
             "<think> 같은 추론 과정은 절대 출력하지 마세요.")


def _post(model, messages, timeout, force_json):
    body = {"model": model, "messages": messages, "temperature": 0.2}
    if force_json:
        body["response_format"] = {"type": "json_object"}
    r = requests.post(
        f"{API_BASE}/chat/completions",
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json=body, timeout=timeout, verify=False,
    )
    return r


def call_llm(stage, messages, force_json=True, log=print):
    """단계별 모델로 호출. 400/403/타임아웃이면 폴백 모델로 자동 재시도.
       force_json 이면 JSON 으로 파싱해서 돌려주고, 아니면 원문 문자열을 돌려준다."""
    candidates = [MODELS[stage]] + [m for m in FALLBACK.get(stage, []) if m != MODELS[stage]]
    timeout = TIMEOUT.get(stage, 120)

    if force_json:
        messages = [{"role": "system", "content": JSON_RULE}] + list(messages)

    last_err = None
    for model in candidates:
        for use_json_mode in ([True, False] if force_json else [False]):
            t0 = time.time()
            try:
                r = _post(model, messages, timeout, use_json_mode)
                if r.status_code in (400, 403, 404):
                    last_err = f"HTTP {r.status_code}: {r.text[:160]}"
                    # json_object 미지원이면 끄고 재시도, 모델 문제면 다음 모델로
                    if use_json_mode and "response_format" in r.text:
                        continue
                    break
                r.raise_for_status()
                text = r.json()["choices"][0]["message"]["content"]
                log(f"  [{stage}] {model} · {time.time()-t0:.1f}초 · OK")
                return extract_json(text) if force_json else text
            except requests.exceptions.Timeout:
                last_err = f"타임아웃({timeout}초)"
                log(f"  [{stage}] {model} · 타임아웃 → 다음 모델")
                break
            except ValueError as e:      # JSON 파싱 실패 → 같은 모델로 한 번 더(모드 바꿔서)
                last_err = str(e)
                log(f"  [{stage}] {model} · {e}")
                continue
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                log(f"  [{stage}] {model} · {last_err}")
                break
    raise RuntimeError(f"[{stage}] 모든 모델 실패 — 마지막 오류: {last_err}")


# ── 사용 예시 ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    msgs = [{"role": "user", "content": "다음 구간의 이상을 JSON으로 요약해줘: ..."}]
    try:
        result = call_llm("stage1", msgs)     # 1차
        print(result)
    except Exception as e:
        print("실패:", e)
