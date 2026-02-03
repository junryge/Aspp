# LangGraph Self-Correction 가이드

## 🎯 개요

20B 모델(gpt-oss-20b)로 **자기 검토 루프**를 구현하여 답변 품질을 향상시킵니다.

```
질문 → [생성] → [검토] → PASS? → 최종답변
                    ↓ FAIL
                  [재생성] (최대 3회)
```

## 📦 설치

```bash
# LangGraph 설치
pip install langgraph langchain-core

# 엑셀 처리용 (선택)
pip install pandas openpyxl
```

## 🚀 사용법

### 1. 단독 실행 (테스트)

```bash
python langgraph_self_correction.py
```

### 2. FastAPI 서버 실행

```bash
# 통합 서버 (기존 코딩 + Self-Correction)
python coding_llm_server_v2.py

# 또는 기존 서버에 라우터 추가
# app.include_router(router, prefix="/api/sc")
```

## 📡 API 엔드포인트

### 기존 코딩 기능 (동일)
- `POST /api/ask` - 코드 생성/리뷰/디버그/설명

### Self-Correction (신규)
- `POST /api/sc/ask` - Self-Correction 루프
- `POST /api/sc/ask_excel` - 엑셀 + Self-Correction
- `POST /api/sc/excel_read` - 엑셀 미리보기

## 📝 사용 예시

### Python에서 직접 사용

```python
from langgraph_self_correction import run_self_correction

# 일반 질문
result = run_self_correction(
    "Python에서 데코레이터란 무엇인가?"
)
print(result['final_answer'])

# 데이터와 함께 질문
data = """
매출: 1월 100, 2월 150, 3월 120
"""
result = run_self_correction(
    "매출 추이를 분석해줘",
    context=data
)
```

### API 호출

```python
import requests

# Self-Correction 질문
response = requests.post(
    "http://localhost:8001/api/sc/ask",
    json={
        "question": "XGBoost와 LightGBM의 차이점",
        "context": ""  # 선택
    }
)
print(response.json())

# 엑셀 파일 분석
with open("data.xlsx", "rb") as f:
    response = requests.post(
        "http://localhost:8001/api/sc/ask_excel",
        files={"file": f},
        data={"question": "이 데이터의 이상치를 찾아줘"}
    )
print(response.json())
```

### curl 예시

```bash
# Self-Correction
curl -X POST http://localhost:8001/api/sc/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Python GIL이란?", "context": ""}'

# 엑셀 분석
curl -X POST http://localhost:8001/api/sc/ask_excel \
  -F "file=@data.xlsx" \
  -F "question=평균값을 계산해줘"
```

## 🔧 설정

### token.txt
```
YOUR_API_TOKEN_HERE
```

### 환경 전환
```bash
# DEV (30B) - 코딩 특화
curl -X POST http://localhost:8001/api/set_env -d '{"env": "dev"}'

# PROD (80B) - 범용 대형
curl -X POST http://localhost:8001/api/set_env -d '{"env": "prod"}'

# COMMON (20B) - 기본
curl -X POST http://localhost:8001/api/set_env -d '{"env": "common"}'
```

## 📊 응답 구조

```json
{
  "success": true,
  "answer": "최종 답변...",
  "retry_count": 2,        // 시도 횟수
  "is_valid": true,        // 검토 통과 여부
  "review": "PASS\n...",   // 마지막 검토 결과
  "excel_info": {          // 엑셀 사용 시
    "columns": ["A", "B"],
    "rows": 100
  }
}
```

## 💡 팁

1. **간단한 질문**: 기존 `/api/ask` 사용 (더 빠름)
2. **복잡한 분석**: `/api/sc/ask` 사용 (더 정확)
3. **엑셀 데이터**: `/api/sc/ask_excel` 사용

## 🔄 확장 가능

### MCP 도구 추가 예시
```python
# 엑셀 도구 외에 다른 도구 추가
def call_external_exe(exe_path: str, args: list) -> dict:
    """외부 EXE 실행 도구"""
    import subprocess
    result = subprocess.run([exe_path] + args, capture_output=True, text=True)
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode
    }

# 그래프에 도구 노드 추가 가능
```

## 📁 파일 구조

```
├── langgraph_self_correction.py  # 핵심 Self-Correction 로직
├── router_self_correction.py     # FastAPI 라우터 (분리용)
├── coding_llm_server_v2.py       # 통합 서버
├── token.txt                     # API 토큰
└── README_SC.md                  # 이 파일
```