#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangGraph Self-Correction 루프 (v1.0)
- SK Hynix 내부 20B API (gpt-oss-20b) 사용
- 답변 생성 → 검토 → 재생성 루프
- 엑셀 데이터 처리 도구 포함
"""

import os
import re
import json
import requests
from typing import TypedDict, Annotated, Literal
from dataclasses import dataclass

# LangGraph 임포트
try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
except ImportError:
    print("❌ langgraph 미설치. 실행: pip install langgraph langchain-core")
    exit(1)

# ========================================
# 설정
# ========================================
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
            with open(p, "r") as f:
                token = f.read().strip()
                if token and "REPLACE" not in token:
                    return token
    return None

API_TOKEN = load_token()

# ========================================
# 상태 정의
# ========================================
class AgentState(TypedDict):
    """에이전트 상태"""
    question: str           # 원본 질문
    context: str            # 추가 컨텍스트 (엑셀 데이터 등)
    answer: str             # 생성된 답변
    review: str             # 검토 결과
    is_valid: bool          # 답변 유효성
    retry_count: int        # 재시도 횟수
    final_answer: str       # 최종 답변


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
            # <think> 태그 제거
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
            return content.strip()
        else:
            return f"[ERROR] API 오류: {response.status_code}"
    except Exception as e:
        return f"[ERROR] {str(e)}"


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
    except Exception as e:
        return {"success": False, "error": str(e)}


# ========================================
# 그래프 노드들
# ========================================
def generate_answer(state: AgentState) -> AgentState:
    """1단계: 답변 생성"""
    print(f"\n🔄 [생성] 답변 생성 중... (시도 {state['retry_count'] + 1})")
    
    system_prompt = """당신은 데이터 분석 전문가입니다.
사용자의 질문에 정확하고 구체적으로 답변하세요.

규칙:
1. 데이터가 제공되면 반드시 참조하여 답변
2. 숫자나 통계는 정확하게 인용
3. 불확실한 내용은 명시
4. 간결하고 명확하게 답변"""

    # 컨텍스트가 있으면 포함
    prompt = f"질문: {state['question']}"
    if state.get('context'):
        prompt = f"[데이터]\n{state['context']}\n\n{prompt}"
    
    # 재시도인 경우 이전 피드백 포함
    if state['retry_count'] > 0 and state.get('review'):
        prompt += f"\n\n[이전 검토 피드백]\n{state['review']}\n\n위 피드백을 반영하여 다시 답변해주세요."
    
    answer = call_llm(prompt, system_prompt)
    
    return {
        **state,
        "answer": answer,
        "retry_count": state['retry_count'] + 1
    }


def review_answer(state: AgentState) -> AgentState:
    """2단계: 답변 검토"""
    print("🔍 [검토] 답변 검토 중...")
    
    system_prompt = """당신은 엄격한 품질 검토자입니다.
생성된 답변을 검토하고 문제점을 찾아주세요.

검토 기준:
1. 질문에 정확히 답변했는가?
2. 제공된 데이터를 올바르게 사용했는가?
3. 논리적 오류나 모순이 있는가?
4. 숫자/통계가 정확한가?

출력 형식:
- 첫 줄: PASS 또는 FAIL
- 이후: 상세 피드백"""

    prompt = f"""[원본 질문]
{state['question']}

[생성된 답변]
{state['answer']}
"""
    if state.get('context'):
        prompt = f"[참조 데이터]\n{state['context']}\n\n{prompt}"
    
    review = call_llm(prompt, system_prompt, max_tokens=500)
    
    # PASS/FAIL 판정
    is_valid = review.strip().upper().startswith("PASS")
    
    print(f"   결과: {'✅ PASS' if is_valid else '❌ FAIL'}")
    
    return {
        **state,
        "review": review,
        "is_valid": is_valid
    }


def finalize_answer(state: AgentState) -> AgentState:
    """3단계: 최종 답변 확정"""
    print("✅ [완료] 최종 답변 확정")
    
    return {
        **state,
        "final_answer": state['answer']
    }


def should_retry(state: AgentState) -> Literal["retry", "finalize"]:
    """조건부 분기: 재시도 여부 결정"""
    # 최대 3회 시도
    if state['retry_count'] >= 3:
        print("⚠️  최대 재시도 횟수 도달")
        return "finalize"
    
    # 검토 통과하면 종료
    if state['is_valid']:
        return "finalize"
    
    # 검토 실패하면 재시도
    return "retry"


# ========================================
# 그래프 구성
# ========================================
def build_graph():
    """LangGraph 구성"""
    graph = StateGraph(AgentState)
    
    # 노드 추가
    graph.add_node("generate", generate_answer)
    graph.add_node("review", review_answer)
    graph.add_node("finalize", finalize_answer)
    
    # 엣지 연결
    graph.set_entry_point("generate")
    graph.add_edge("generate", "review")
    
    # 조건부 엣지 (검토 결과에 따라 분기)
    graph.add_conditional_edges(
        "review",
        should_retry,
        {
            "retry": "generate",    # 재시도
            "finalize": "finalize"  # 완료
        }
    )
    
    graph.add_edge("finalize", END)
    
    return graph.compile()


# ========================================
# 메인 실행 함수
# ========================================
def run_self_correction(question: str, context: str = "") -> dict:
    """Self-Correction 루프 실행"""
    print("=" * 50)
    print("🚀 LangGraph Self-Correction 시작")
    print(f"📝 질문: {question[:100]}...")
    print("=" * 50)
    
    # 그래프 빌드
    app = build_graph()
    
    # 초기 상태
    initial_state = {
        "question": question,
        "context": context,
        "answer": "",
        "review": "",
        "is_valid": False,
        "retry_count": 0,
        "final_answer": ""
    }
    
    # 실행
    result = app.invoke(initial_state)
    
    print("\n" + "=" * 50)
    print("🎯 최종 결과")
    print("=" * 50)
    print(f"재시도 횟수: {result['retry_count']}")
    print(f"검토 통과: {'✅' if result['is_valid'] else '❌'}")
    
    return {
        "question": result['question'],
        "final_answer": result['final_answer'],
        "retry_count": result['retry_count'],
        "is_valid": result['is_valid'],
        "last_review": result['review']
    }


# ========================================
# 엑셀 데이터 + Self-Correction 통합
# ========================================
def analyze_excel_with_correction(file_path: str, question: str) -> dict:
    """엑셀 데이터 분석 + Self-Correction"""
    print(f"\n📊 엑셀 파일 로딩: {file_path}")
    
    # 엑셀 데이터 읽기
    excel_data = read_excel_data(file_path)
    
    if not excel_data['success']:
        return {"success": False, "error": excel_data['error']}
    
    # 데이터 요약을 컨텍스트로 변환
    context = f"""엑셀 데이터 정보:
- 컬럼: {excel_data['columns']}
- 행 수: {excel_data['rows']}
- 미리보기 (상위 10행):
{json.dumps(excel_data['preview'], ensure_ascii=False, indent=2)}
"""
    
    if excel_data.get('summary'):
        context += f"\n- 수치 컬럼 통계:\n{json.dumps(excel_data['summary'], ensure_ascii=False, indent=2)}"
    
    # Self-Correction 루프 실행
    result = run_self_correction(question, context)
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
    
    def ask(self, question: str, context: str = "") -> dict:
        """질문 처리"""
        initial_state = {
            "question": question,
            "context": context,
            "answer": "",
            "review": "",
            "is_valid": False,
            "retry_count": 0,
            "final_answer": ""
        }
        
        result = self.graph.invoke(initial_state)
        
        return {
            "success": True,
            "answer": result['final_answer'],
            "retry_count": result['retry_count'],
            "is_valid": result['is_valid']
        }
    
    def ask_with_excel(self, file_path: str, question: str) -> dict:
        """엑셀 데이터와 함께 질문"""
        excel_data = read_excel_data(file_path)
        
        if not excel_data['success']:
            return {"success": False, "error": excel_data['error']}
        
        context = f"엑셀 컬럼: {excel_data['columns']}\n"
        context += f"행 수: {excel_data['rows']}\n"
        context += f"데이터 미리보기:\n{json.dumps(excel_data['preview'], ensure_ascii=False)}"
        
        return self.ask(question, context)


# ========================================
# 테스트 실행
# ========================================
if __name__ == "__main__":
    print("🔧 LangGraph Self-Correction 테스트")
    print(f"📡 API: {API_CONFIG['name']} ({API_CONFIG['model']})")
    print(f"🔑 토큰: {'✅ 로드됨' if API_TOKEN else '❌ 없음'}")
    
    if not API_TOKEN:
        print("\n⚠️ token.txt 파일에 API 토큰을 저장하세요.")
        exit(1)
    
    # 테스트 1: 일반 질문
    print("\n" + "=" * 60)
    print("테스트 1: 일반 질문")
    result = run_self_correction(
        "Python에서 리스트와 튜플의 차이점을 설명하고, 각각 언제 사용하면 좋은지 알려줘."
    )
    print(f"\n📝 최종 답변:\n{result['final_answer']}")
    
    # 테스트 2: 데이터 분석 질문 (컨텍스트 포함)
    print("\n" + "=" * 60)
    print("테스트 2: 데이터 분석 질문")
    sample_context = """
매출 데이터:
- 1월: 1,200만원
- 2월: 1,450만원  
- 3월: 1,100만원
- 4월: 1,680만원
- 5월: 1,320만원
"""
    result2 = run_self_correction(
        "이 매출 데이터에서 가장 높은 달과 낮은 달을 찾고, 전체 평균과 추세를 분석해줘.",
        sample_context
    )
    print(f"\n📝 최종 답변:\n{result2['final_answer']}")