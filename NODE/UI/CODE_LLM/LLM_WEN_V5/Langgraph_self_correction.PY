#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangGraph Self-Correction 루프 (v2.0)
- 범용 질문 지원 (코드, 데이터분석, 일반 질문 등)
- 교정 히스토리/로그 추적
- 사용자 피드백 반영 (좋아요/싫어요 + 재교정 요청)
- 엑셀 데이터 처리 도구 포함
"""

import os
import re
import json
import uuid
import datetime
import requests
import logging
from typing import TypedDict, Literal, Optional, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SelfCorrection")

# LangGraph 임포트
try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
except ImportError:
    logger.error("langgraph 미설치. 실행: pip install langgraph langchain-core")
    StateGraph = None
    END = None

# ========================================
# 설정
# ========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

API_CONFIG = {
    "url": "http://common.llm.skhynix.com/v1/chat/completions",
    "model": "gpt-oss-20b",
    "name": "COMMON(20B)"
}

# 토큰 로드
def load_token():
    paths = ["token.txt", "../token.txt", os.path.expanduser("~/token.txt")]
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    token = f.read().strip()
                    if token and "REPLACE" not in token:
                        return token
            except OSError:
                continue
    return None

API_TOKEN = load_token()


# ========================================
# 교정 히스토리 저장소
# ========================================
SC_HISTORY_DIR = os.path.join(BASE_DIR, "sc_history")
os.makedirs(SC_HISTORY_DIR, exist_ok=True)

SC_HISTORY_FILE = os.path.join(SC_HISTORY_DIR, "correction_log.json")


def _load_history() -> List[dict]:
    """교정 히스토리 로드"""
    try:
        if os.path.exists(SC_HISTORY_FILE):
            with open(SC_HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"히스토리 로드 실패: {e}")
    return []


def _save_history(history: List[dict]):
    """교정 히스토리 저장 (최대 200건)"""
    try:
        history = history[-200:]
        with open(SC_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except OSError as e:
        logger.error(f"히스토리 저장 실패: {e}")


def save_correction_record(record: dict):
    """교정 기록 1건 추가"""
    history = _load_history()
    history.append(record)
    _save_history(history)


def get_correction_history(limit: int = 20, question_filter: str = None) -> List[dict]:
    """교정 히스토리 조회"""
    history = _load_history()

    if question_filter:
        q_lower = question_filter.lower()
        history = [h for h in history if q_lower in h.get("question", "").lower()]

    return history[-limit:]


def get_correction_by_id(correction_id: str) -> Optional[dict]:
    """ID로 특정 교정 기록 조회"""
    history = _load_history()
    for record in history:
        if record.get("id") == correction_id:
            return record
    return None


def update_feedback(correction_id: str, feedback: str, feedback_text: str = "") -> bool:
    """사용자 피드백 업데이트 (좋아요/싫어요)"""
    history = _load_history()
    for record in history:
        if record.get("id") == correction_id:
            record["feedback"] = feedback  # "good" or "bad"
            record["feedback_text"] = feedback_text
            record["feedback_time"] = datetime.datetime.now().isoformat()
            _save_history(history)
            logger.info(f"피드백 저장: {correction_id} → {feedback}")
            return True
    return False


def get_feedback_stats() -> dict:
    """피드백 통계"""
    history = _load_history()
    total = len(history)
    good = sum(1 for h in history if h.get("feedback") == "good")
    bad = sum(1 for h in history if h.get("feedback") == "bad")
    no_feedback = total - good - bad
    avg_retries = sum(h.get("retry_count", 0) for h in history) / max(total, 1)
    pass_rate = sum(1 for h in history if h.get("is_valid")) / max(total, 1) * 100

    return {
        "total": total,
        "good": good,
        "bad": bad,
        "no_feedback": no_feedback,
        "avg_retries": round(avg_retries, 1),
        "pass_rate": round(pass_rate, 1)
    }


# ========================================
# 상태 정의
# ========================================
class AgentState(TypedDict):
    """에이전트 상태"""
    question: str           # 원본 질문
    context: str            # 추가 컨텍스트 (엑셀 데이터, 코드, 문서 등)
    question_type: str      # 질문 유형: general, code, data, knowledge
    answer: str             # 생성된 답변
    review: str             # 검토 결과
    is_valid: bool          # 답변 유효성
    retry_count: int        # 재시도 횟수
    max_retries: int        # 최대 재시도 횟수
    final_answer: str       # 최종 답변
    correction_log: list    # 각 시도별 기록


# ========================================
# 질문 유형 감지
# ========================================
def detect_question_type(question: str, context: str = "") -> str:
    """질문 유형 자동 감지"""
    q_lower = question.lower()

    # 코드 관련
    code_keywords = ["코드", "함수", "클래스", "버그", "에러", "디버그", "리팩토링",
                     "python", "java", "javascript", "sql", "def ", "class ",
                     "코딩", "프로그래밍", "알고리즘", "구현"]
    if any(kw in q_lower for kw in code_keywords):
        return "code"

    # 데이터 분석 관련
    data_keywords = ["데이터", "엑셀", "csv", "통계", "분석", "차트", "그래프",
                     "평균", "합계", "추세", "컬럼", "행", "테이블"]
    if any(kw in q_lower for kw in data_keywords):
        return "data"

    # 지식/문서 관련
    knowledge_keywords = ["문서", "아키텍처", "설계", "구조", "설명", "개념",
                          "프로젝트", "시스템", "모델", "워크플로우"]
    if any(kw in q_lower for kw in knowledge_keywords):
        return "knowledge"

    # 컨텍스트로 판단
    if context:
        if "컬럼" in context or "행 수" in context or "미리보기" in context:
            return "data"
        if "```" in context or "def " in context or "class " in context:
            return "code"

    return "general"


# ========================================
# 질문 유형별 시스템 프롬프트
# ========================================
SYSTEM_PROMPTS = {
    "general": """당신은 지식이 풍부한 AI 전문가입니다.
사용자의 질문에 정확하고 구체적으로 답변하세요.

규칙:
1. 질문의 핵심을 파악하여 직접적으로 답변
2. 불확실한 내용은 명시
3. 간결하고 명확하게 답변
4. 한국어로 답변""",

    "code": """당신은 시니어 소프트웨어 개발자입니다.
코드 관련 질문에 정확하고 실용적으로 답변하세요.

규칙:
1. 코드는 실행 가능한 완전한 형태로 제공
2. 코드 블록은 ```언어명 으로 감싸기
3. 에러 처리와 엣지 케이스 고려
4. 간결한 한국어 설명 포함
5. 베스트 프랙티스 준수""",

    "data": """당신은 데이터 분석 전문가입니다.
데이터를 정확히 분석하고 인사이트를 도출하세요.

규칙:
1. 데이터가 제공되면 반드시 참조하여 답변
2. 숫자나 통계는 정확하게 인용
3. 불확실한 내용은 명시
4. 분석 결과를 구조화하여 답변
5. 한국어로 답변""",

    "knowledge": """당신은 시니어 기술 문서 전문가입니다.
기술 문서와 아키텍처에 대해 정확하게 설명하세요.

규칙:
1. 문서 내용을 근거로 정확히 답변
2. 구조화된 형태로 정보 정리
3. 핵심 개념과 관계를 명확히 설명
4. 한국어로 답변"""
}

REVIEW_PROMPTS = {
    "general": """당신은 엄격한 품질 검토자입니다.

검토 기준:
1. 질문에 정확히 답변했는가?
2. 논리적 오류나 모순이 있는가?
3. 설명이 명확하고 이해하기 쉬운가?
4. 누락된 중요 정보가 있는가?

출력 형식:
- 첫 줄: PASS 또는 FAIL
- 이후: 상세 피드백""",

    "code": """당신은 시니어 코드 리뷰어입니다.

검토 기준:
1. 코드가 문법적으로 정확한가?
2. 코드가 요구사항을 충족하는가?
3. 에러 처리가 적절한가?
4. 성능 이슈나 보안 취약점이 있는가?
5. 설명이 정확한가?

출력 형식:
- 첫 줄: PASS 또는 FAIL
- 이후: 상세 피드백""",

    "data": """당신은 데이터 분석 검증 전문가입니다.

검토 기준:
1. 데이터를 올바르게 참조했는가?
2. 숫자/통계가 정확한가?
3. 분석 논리에 오류가 없는가?
4. 인사이트가 데이터로 뒷받침되는가?

출력 형식:
- 첫 줄: PASS 또는 FAIL
- 이후: 상세 피드백""",

    "knowledge": """당신은 기술 문서 검증 전문가입니다.

검토 기준:
1. 문서 내용을 정확히 인용했는가?
2. 기술적 설명에 오류가 없는가?
3. 구조화가 잘 되었는가?
4. 누락된 중요 정보가 있는가?

출력 형식:
- 첫 줄: PASS 또는 FAIL
- 이후: 상세 피드백"""
}


# ========================================
# LLM 호출 함수
# ========================================
def call_llm(prompt: str, system_prompt: str = "", max_tokens: int = 2000) -> str:
    """20B API 호출"""
    if not API_TOKEN:
        return "[ERROR] API 토큰이 없습니다."

    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    data = {
        "model": API_CONFIG["model"],
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3
    }

    try:
        response = requests.post(API_CONFIG["url"], headers=headers, json=data, timeout=120)
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
            return content.strip()
        elif response.status_code == 401:
            return "[ERROR] API 인증 실패 (401) - 토큰을 확인해주세요."
        elif response.status_code == 429:
            return "[ERROR] API 요청 한도 초과 (429) - 잠시 후 다시 시도해주세요."
        elif response.status_code == 503:
            return "[ERROR] LLM 서버 응답 없음 (503)"
        else:
            return f"[ERROR] API 오류: {response.status_code}"
    except requests.exceptions.Timeout:
        return "[ERROR] API 요청 시간 초과 (120초)"
    except requests.exceptions.ConnectionError:
        return f"[ERROR] LLM 서버 연결 실패 - {API_CONFIG['url']}"
    except Exception as e:
        logger.error(f"LLM 호출 오류: {e}", exc_info=True)
        return f"[ERROR] {type(e).__name__}: {str(e)}"


# ========================================
# 도구: 엑셀 데이터 처리
# ========================================
def read_excel_data(file_path: str, sheet_name: str = None) -> dict:
    """엑셀 파일 읽기 도구"""
    try:
        import pandas as pd

        if not os.path.exists(file_path):
            return {"success": False, "error": f"파일 없음: {file_path}"}

        if sheet_name:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        else:
            df = pd.read_excel(file_path)

        return {
            "success": True,
            "columns": df.columns.tolist(),
            "rows": len(df),
            "preview": df.head(10).to_dict('records'),
            "summary": df.describe().to_dict() if df.select_dtypes(include='number').columns.any() else {}
        }
    except ImportError:
        return {"success": False, "error": "pandas 미설치. pip install pandas openpyxl"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def analyze_excel_column(file_path: str, column: str) -> dict:
    """엑셀 특정 컬럼 분석"""
    try:
        import pandas as pd
        df = pd.read_excel(file_path)

        if column not in df.columns:
            return {"success": False, "error": f"컬럼 없음: {column}"}

        col_data = df[column]
        result = {
            "success": True,
            "column": column,
            "dtype": str(col_data.dtype),
            "non_null": int(col_data.count()),
            "null_count": int(col_data.isna().sum())
        }

        if col_data.dtype in ['int64', 'float64']:
            result.update({
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "mean": float(col_data.mean()),
                "std": float(col_data.std())
            })
        else:
            result["unique_values"] = col_data.nunique()
            result["top_values"] = col_data.value_counts().head(5).to_dict()

        return result
    except ImportError:
        return {"success": False, "error": "pandas 미설치"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ========================================
# 그래프 노드들
# ========================================
def generate_answer(state: AgentState) -> AgentState:
    """1단계: 답변 생성 (질문 유형별 최적화)"""
    q_type = state.get("question_type", "general")
    attempt = state['retry_count'] + 1
    logger.info(f"[생성] 시도 {attempt}, 유형: {q_type}")

    system_prompt = SYSTEM_PROMPTS.get(q_type, SYSTEM_PROMPTS["general"])

    # 컨텍스트가 있으면 포함
    prompt = f"질문: {state['question']}"
    if state.get('context'):
        prompt = f"[참고 자료]\n{state['context']}\n\n{prompt}"

    # 재시도인 경우 이전 피드백 포함
    if state['retry_count'] > 0 and state.get('review'):
        prompt += f"\n\n[이전 검토 피드백]\n{state['review']}\n\n위 피드백을 반영하여 다시 답변해주세요."

    # 사용자 피드백이 있으면 반영
    user_feedback = state.get("correction_log", [])
    if user_feedback:
        last_entry = user_feedback[-1] if user_feedback else None
        if last_entry and last_entry.get("user_feedback_text"):
            prompt += f"\n\n[사용자 피드백]\n{last_entry['user_feedback_text']}\n\n위 사용자 피드백을 반드시 반영하여 답변해주세요."

    answer = call_llm(prompt, system_prompt)

    # 교정 로그에 이번 시도 기록 추가
    log_entry = {
        "attempt": attempt,
        "timestamp": datetime.datetime.now().isoformat(),
        "answer_preview": answer[:200] if answer else "",
        "question_type": q_type
    }
    correction_log = list(state.get("correction_log", []))
    correction_log.append(log_entry)

    return {
        **state,
        "answer": answer,
        "retry_count": attempt,
        "correction_log": correction_log
    }


def review_answer(state: AgentState) -> AgentState:
    """2단계: 답변 검토 (질문 유형별 리뷰어)"""
    q_type = state.get("question_type", "general")
    logger.info(f"[검토] 유형: {q_type}")

    review_system = REVIEW_PROMPTS.get(q_type, REVIEW_PROMPTS["general"])

    prompt = f"""[원본 질문]
{state['question']}

[생성된 답변]
{state['answer']}
"""
    if state.get('context'):
        prompt = f"[참조 데이터]\n{state['context'][:3000]}\n\n{prompt}"

    review = call_llm(prompt, review_system, max_tokens=500)

    # PASS/FAIL 판정
    is_valid = review.strip().upper().startswith("PASS")

    logger.info(f"[검토] 결과: {'PASS' if is_valid else 'FAIL'}")

    # 로그 업데이트
    correction_log = list(state.get("correction_log", []))
    if correction_log:
        correction_log[-1]["review"] = review[:300]
        correction_log[-1]["is_valid"] = is_valid

    return {
        **state,
        "review": review,
        "is_valid": is_valid,
        "correction_log": correction_log
    }


def finalize_answer(state: AgentState) -> AgentState:
    """3단계: 최종 답변 확정"""
    logger.info(f"[완료] 시도 {state['retry_count']}회, 통과: {state['is_valid']}")

    return {
        **state,
        "final_answer": state['answer']
    }


def should_retry(state: AgentState) -> Literal["retry", "finalize"]:
    """조건부 분기: 재시도 여부 결정"""
    max_retries = state.get('max_retries', 3)

    if state['retry_count'] >= max_retries:
        logger.info(f"최대 재시도 횟수 도달 ({max_retries})")
        return "finalize"

    if state['is_valid']:
        return "finalize"

    return "retry"


# ========================================
# 그래프 구성
# ========================================
def build_graph():
    """LangGraph 구성"""
    if StateGraph is None:
        raise ImportError("langgraph가 설치되지 않았습니다. pip install langgraph langchain-core")

    graph = StateGraph(AgentState)

    graph.add_node("generate", generate_answer)
    graph.add_node("review", review_answer)
    graph.add_node("finalize", finalize_answer)

    graph.set_entry_point("generate")
    graph.add_edge("generate", "review")

    graph.add_conditional_edges(
        "review",
        should_retry,
        {
            "retry": "generate",
            "finalize": "finalize"
        }
    )

    graph.add_edge("finalize", END)

    return graph.compile()


# ========================================
# 메인 실행 함수
# ========================================
def run_self_correction(question: str, context: str = "",
                        question_type: str = None, max_retries: int = 3,
                        user_feedback_text: str = "") -> dict:
    """Self-Correction 루프 실행 (범용)"""
    correction_id = uuid.uuid4().hex[:12]

    # 질문 유형 자동 감지
    if not question_type:
        question_type = detect_question_type(question, context)

    logger.info(f"[SC-{correction_id}] 시작 | 유형: {question_type} | 최대 {max_retries}회")

    # 그래프 빌드
    app = build_graph()

    # 초기 교정 로그 (사용자 피드백이 있으면 포함)
    initial_log = []
    if user_feedback_text:
        initial_log.append({"user_feedback_text": user_feedback_text})

    # 초기 상태
    initial_state = {
        "question": question,
        "context": context,
        "question_type": question_type,
        "answer": "",
        "review": "",
        "is_valid": False,
        "retry_count": 0,
        "max_retries": max_retries,
        "final_answer": "",
        "correction_log": initial_log
    }

    # 실행
    result = app.invoke(initial_state)

    # 교정 기록 저장
    record = {
        "id": correction_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "question": question,
        "question_type": question_type,
        "context_preview": context[:200] if context else "",
        "final_answer": result['final_answer'],
        "retry_count": result['retry_count'],
        "is_valid": result['is_valid'],
        "last_review": result['review'],
        "correction_log": result.get('correction_log', []),
        "feedback": None,
        "feedback_text": "",
        "feedback_time": None
    }
    save_correction_record(record)

    logger.info(f"[SC-{correction_id}] 완료 | 시도: {result['retry_count']}회 | 통과: {result['is_valid']}")

    return {
        "success": True,
        "id": correction_id,
        "question": result['question'],
        "question_type": question_type,
        "final_answer": result['final_answer'],
        "retry_count": result['retry_count'],
        "is_valid": result['is_valid'],
        "last_review": result['review'],
        "correction_log": result.get('correction_log', [])
    }


# ========================================
# 재교정 (사용자 피드백 반영)
# ========================================
def re_correct(correction_id: str, feedback_text: str) -> dict:
    """기존 교정 결과에 대해 사용자 피드백을 반영하여 재교정"""
    original = get_correction_by_id(correction_id)
    if not original:
        return {"success": False, "error": f"교정 기록 없음: {correction_id}"}

    # 기존 결과에 '싫어요' 피드백 기록
    update_feedback(correction_id, "bad", feedback_text)

    # 사용자 피드백을 반영하여 재교정
    logger.info(f"[재교정] {correction_id} → 피드백: {feedback_text[:100]}")

    result = run_self_correction(
        question=original["question"],
        context=original.get("context_preview", ""),
        question_type=original.get("question_type"),
        user_feedback_text=feedback_text
    )

    return result


# ========================================
# 엑셀 데이터 + Self-Correction 통합
# ========================================
def analyze_excel_with_correction(file_path: str, question: str) -> dict:
    """엑셀 데이터 분석 + Self-Correction"""
    excel_data = read_excel_data(file_path)

    if not excel_data['success']:
        return {"success": False, "error": excel_data['error']}

    context = f"""엑셀 데이터 정보:
- 컬럼: {excel_data['columns']}
- 행 수: {excel_data['rows']}
- 미리보기 (상위 10행):
{json.dumps(excel_data['preview'], ensure_ascii=False, indent=2)}
"""

    if excel_data.get('summary'):
        context += f"\n- 수치 컬럼 통계:\n{json.dumps(excel_data['summary'], ensure_ascii=False, indent=2)}"

    result = run_self_correction(question, context, question_type="data")
    result['excel_info'] = {
        "columns": excel_data['columns'],
        "rows": excel_data['rows']
    }

    return result


# ========================================
# FastAPI 통합용 클래스
# ========================================
class SelfCorrectionAgent:
    """FastAPI 통합용 에이전트 클래스"""

    def __init__(self):
        self.graph = build_graph()

    def ask(self, question: str, context: str = "",
            question_type: str = None, max_retries: int = 3) -> dict:
        """범용 질문 처리"""
        return run_self_correction(question, context, question_type, max_retries)

    def ask_with_excel(self, file_path: str, question: str) -> dict:
        """엑셀 데이터와 함께 질문"""
        return analyze_excel_with_correction(file_path, question)

    def ask_with_feedback(self, correction_id: str, feedback_text: str) -> dict:
        """피드백 반영 재교정"""
        return re_correct(correction_id, feedback_text)


# ========================================
# 테스트 실행
# ========================================
if __name__ == "__main__":
    print("LangGraph Self-Correction v2.0 테스트")
    print(f"API: {API_CONFIG['name']} ({API_CONFIG['model']})")
    print(f"토큰: {'로드됨' if API_TOKEN else '없음'}")

    if not API_TOKEN:
        print("\ntoken.txt 파일에 API 토큰을 저장하세요.")
        exit(1)

    # 테스트 1: 일반 질문
    print("\n" + "=" * 60)
    print("테스트 1: 일반 질문 (자동 유형 감지)")
    result = run_self_correction(
        "Python에서 리스트와 튜플의 차이점을 설명하고, 각각 언제 사용하면 좋은지 알려줘."
    )
    print(f"유형: {result['question_type']}")
    print(f"시도: {result['retry_count']}회")
    print(f"통과: {result['is_valid']}")
    print(f"답변: {result['final_answer'][:200]}...")

    # 테스트 2: 코드 질문
    print("\n" + "=" * 60)
    print("테스트 2: 코드 질문")
    result2 = run_self_correction(
        "Python으로 이진 탐색 함수를 구현해줘."
    )
    print(f"유형: {result2['question_type']}")
    print(f"시도: {result2['retry_count']}회")

    # 테스트 3: 히스토리 확인
    print("\n" + "=" * 60)
    print("테스트 3: 교정 히스토리")
    history = get_correction_history(limit=5)
    print(f"기록: {len(history)}건")
    for h in history:
        print(f"  - [{h['id']}] {h['question'][:40]}... (시도: {h['retry_count']}회)")
