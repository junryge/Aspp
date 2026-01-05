#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AMHS Log Analysis Server
- CSV upload and AMHS log analysis
- Uses system prompt and fewshot examples for analysis
"""

import os
import re
import requests
import pandas as pd
from io import StringIO
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# ========================================
# Global Variables
# ========================================
llm = None  # Local LLM

# LLM Settings
LLM_MODE = "api"  # "local" or "api"
API_TOKEN = None

# 개발/운영 환경 설정
ENV_MODE = "dev"  # "dev" or "prod"

# 환경별 설정
ENV_CONFIG = {
    "dev": {
        "url": "http://dev.assistant.llm.skhynix.com/v1/chat/completions",
        "model": "Qwen3-Coder-30B-A3B-Instruct",
        "name": "개발(30B)"
    },
    "prod": {
        "url": "http://summary.llm.skhynix.com/v1/chat/completions",
        "model": "Qwen3-Next-80B-A3B-Instruct",
        "name": "운영(80B)"
    }
}

# 현재 사용 중인 URL/Model (ENV_MODE에 따라 변경됨)
API_URL = ENV_CONFIG["dev"]["url"]
API_MODEL = ENV_CONFIG["dev"]["model"]

# AMHS Prompts (LOCAL = 14B용 경량, API = 전체)
AMHS_SYSTEM_PROMPT_LOCAL = ""
AMHS_FEWSHOT_PROMPT_LOCAL = ""
AMHS_SYSTEM_PROMPT_API = ""
AMHS_FEWSHOT_PROMPT_API = ""

# ========================================
# Load AMHS Prompts
# ========================================
def load_amhs_prompts():
    """Load AMHS system and fewshot prompts from markdown files"""
    global AMHS_SYSTEM_PROMPT_LOCAL, AMHS_FEWSHOT_PROMPT_LOCAL
    global AMHS_SYSTEM_PROMPT_API, AMHS_FEWSHOT_PROMPT_API

    # LOCAL 프롬프트 (14B 경량 버전)
    try:
        with open("AMHS_LOG_ANALYSIS_SYSTEM_PROMPT_LOCAL.md", "r", encoding="utf-8") as f:
            AMHS_SYSTEM_PROMPT_LOCAL = f.read()
        logger.info("AMHS System Prompt (LOCAL) loaded")
    except Exception as e:
        logger.error(f"Failed to load LOCAL system prompt: {e}")
        AMHS_SYSTEM_PROMPT_LOCAL = "You are an AMHS log analysis expert."

    try:
        with open("AMHS_LOG_ANALYSIS_FEWSHOT_PROMPT_LOCAL.md", "r", encoding="utf-8") as f:
            AMHS_FEWSHOT_PROMPT_LOCAL = f.read()
        logger.info("AMHS Fewshot Prompt (LOCAL) loaded")
    except Exception as e:
        logger.error(f"Failed to load LOCAL fewshot prompt: {e}")
        AMHS_FEWSHOT_PROMPT_LOCAL = ""

    # API 프롬프트 (전체 버전)
    try:
        with open("AMHS_LOG_ANALYSIS_SYSTEM_PROMPT.md", "r", encoding="utf-8") as f:
            AMHS_SYSTEM_PROMPT_API = f.read()
        logger.info("AMHS System Prompt (API) loaded")
    except Exception as e:
        logger.error(f"Failed to load API system prompt: {e}")
        AMHS_SYSTEM_PROMPT_API = AMHS_SYSTEM_PROMPT_LOCAL

    try:
        with open("AMHS_LOG_ANALYSIS_FEWSHOT_PROMPT.md", "r", encoding="utf-8") as f:
            AMHS_FEWSHOT_PROMPT_API = f.read()
        logger.info("AMHS Fewshot Prompt (API) loaded")
    except Exception as e:
        logger.error(f"Failed to load API fewshot prompt: {e}")
        AMHS_FEWSHOT_PROMPT_API = AMHS_FEWSHOT_PROMPT_LOCAL

# ========================================
# API LLM Functions
# ========================================
def load_api_token():
    """Load API token from file"""
    global API_TOKEN
    token_path = "token.txt"

    if os.path.exists(token_path):
        try:
            with open(token_path, "r", encoding='utf-8') as f:
                API_TOKEN = f.read().strip()
            if API_TOKEN and "REPLACE" not in API_TOKEN:
                logger.info("API token loaded")
                return True
            else:
                logger.warning("Token file has default value")
                return False
        except Exception as e:
            logger.error(f"Failed to load API token: {e}")
            return False
    else:
        logger.warning(f"Token file not found: {token_path}")
        return False

def call_api_llm(prompt: str, system_prompt: str = "", max_tokens: int = 4000) -> str:
    """Call API LLM"""
    global API_TOKEN

    if not API_TOKEN:
        return "API token not loaded. Please check token.txt."

    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    data = {
        "model": API_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3
    }

    # 재시도 1번 포함 (총 2번 시도)
    for attempt in range(2):
        try:
            response = requests.post(API_URL, headers=headers, json=data, timeout=300)

            if response.status_code == 200:
                result = response.json()
                answer = result["choices"][0]["message"]["content"]
                return answer
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return f"API error: {response.status_code}\n{response.text}"

        except requests.exceptions.Timeout:
            logger.warning(f"API timeout (attempt {attempt + 1}/2)")
            if attempt == 0:
                logger.info("Retrying...")
                continue
            return "API 요청 시간 초과 (5분). 서버 상태를 확인하세요."
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return f"API call failed: {e}"
    
    return "API 호출 실패"

def call_local_llm(prompt: str, max_tokens: int = 1500) -> str:
    """Call local LLM with LOCAL (lightweight) prompts"""
    global llm

    if llm is None:
        return "Local LLM not loaded."

    try:
        # LOCAL 경량 프롬프트 사용
        system_prompt = AMHS_SYSTEM_PROMPT_LOCAL + "\n\n" + AMHS_FEWSHOT_PROMPT_LOCAL
        formatted_prompt = f"""<|im_start|>system
{system_prompt}
<|im_end|>
<|im_start|>user
{prompt}
<|im_end|>
<|im_start|>assistant
"""
        response = llm(
            formatted_prompt,
            max_tokens=max_tokens,
            temperature=0.3,
            stop=["<|im_end|>", "\n\n\n"]
        )
        result = response['choices'][0]['text'].strip()

        # Qwen3 thinking 태그 제거 (reasoning 출력 제거)
        import re
        # <think>...</think> 블록 제거
        result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL)
        # 혹시 닫히지 않은 <think> 태그도 제거
        result = re.sub(r'<think>.*', '', result, flags=re.DOTALL)

        # 영어 thinking 블록 제거 (Okay, let's... / First, I need... 등으로 시작하는 부분)
        # 한글 내용이 시작되는 부분 찾기
        korean_match = re.search(r'[가-힣]', result)
        if korean_match:
            # 한글 시작 위치 찾기
            korean_start = korean_match.start()
            # 한글 시작 전에 영어가 많으면 (100자 이상) 그 부분 제거
            if korean_start > 100:
                # 한글 시작 직전의 줄바꿈 찾기
                before_korean = result[:korean_start]
                last_newline = before_korean.rfind('\n')
                if last_newline > 0:
                    result = result[last_newline+1:]
                else:
                    result = result[korean_start:]

        return result.strip()
    except Exception as e:
        logger.error(f"Local LLM call failed: {e}")
        return f"Local LLM call failed: {e}"

# ========================================
# CSV Analysis Functions
# ========================================
def parse_csv_data(csv_content: str) -> pd.DataFrame:
    """Parse CSV content to DataFrame"""
    for encoding in ['utf-8', 'cp949', 'euc-kr']:
        try:
            df = pd.read_csv(StringIO(csv_content), encoding=encoding)
            return df
        except:
            continue
    raise ValueError("Failed to parse CSV with any encoding")

def analyze_csv_basic(df: pd.DataFrame) -> dict:
    """Basic CSV analysis"""
    analysis = {
        "row_count": len(df),
        "columns": list(df.columns),
        "message_types": {},
        "time_range": {},
        "levels": {},
        "machines": set(),
        "carriers": set()
    }

    # Message type distribution
    if 'MESSAGENAME' in df.columns:
        analysis["message_types"] = df['MESSAGENAME'].value_counts().to_dict()

    # Time range
    if 'TIME_EX' in df.columns:
        times = df['TIME_EX'].dropna().tolist()
        if times:
            analysis["time_range"] = {
                "start": str(times[0]) if times else "",
                "end": str(times[-1]) if times else ""
            }

    # Level distribution
    if 'LEVEL' in df.columns:
        analysis["levels"] = df['LEVEL'].value_counts().to_dict()

    # Unique machines
    if 'MACHINENAME' in df.columns:
        analysis["machines"] = list(df['MACHINENAME'].dropna().unique()[:10])

    # Unique carriers
    if 'CARRIER' in df.columns:
        analysis["carriers"] = list(df['CARRIER'].dropna().unique()[:5])

    return analysis

def create_analysis_prompt(df: pd.DataFrame, analysis: dict, user_question: str = "") -> str:
    """Create prompt for LLM analysis (API용 - 상세)"""

    # Get sample data (first and last few rows)
    sample_head = df.head(5).to_string()
    sample_tail = df.tail(5).to_string()

    prompt = f"""## CSV 데이터 분석 요청

### 파일 기본 정보
- 총 레코드 수: {analysis['row_count']}건
- 컬럼: {', '.join(analysis['columns'][:10])}
- 시간 범위: {analysis['time_range'].get('start', 'N/A')} ~ {analysis['time_range'].get('end', 'N/A')}

### 메시지 유형 분포
{dict(list(analysis['message_types'].items())[:10])}

### LEVEL 분포
{analysis['levels']}

### 관련 장비
{analysis['machines']}

### 캐리어
{analysis['carriers']}

### 데이터 샘플 (처음 5개)
{sample_head}

### 데이터 샘플 (마지막 5개)
{sample_tail}

"""

    if user_question:
        prompt += f"""### 사용자 질문
{user_question}

위 데이터를 분석하고 사용자 질문에 답변해주세요.
"""
    else:
        prompt += """### 요청
위 AMHS 로그 데이터를 분석하고, 자연스러운 한국어로 설명해주세요.
이송 경로, 소요시간, 정상/이상 여부 등을 포함해주세요.
"""

    return prompt


def create_analysis_prompt_local(df: pd.DataFrame, analysis: dict, user_question: str = "") -> str:
    """Create prompt for LOCAL LLM (경량 - 핵심 정보만)"""

    # 핵심 컬럼만 추출
    key_cols = ['TIME_EX', 'MESSAGENAME', 'LEVEL', 'MACHINENAME', 'RESULTCODE']
    available_cols = [c for c in key_cols if c in df.columns]

    # 시퀀스 요약 (메시지 순서만)
    msg_sequence = ""
    if 'MESSAGENAME' in df.columns:
        msgs = df['MESSAGENAME'].tolist()
        # 중복 연속 제거하고 핵심만
        unique_msgs = []
        for m in msgs:
            if not unique_msgs or unique_msgs[-1] != m:
                unique_msgs.append(m)
        msg_sequence = " → ".join(unique_msgs[:15])  # 최대 15개

    # 에러 체크
    errors = []
    if 'LEVEL' in df.columns:
        error_rows = df[df['LEVEL'].isin(['ERROR', 'WARN'])]
        if len(error_rows) > 0:
            for _, row in error_rows.head(3).iterrows():
                errors.append(f"- {row.get('MESSAGENAME', 'N/A')}: {row.get('LEVEL', 'N/A')}")

    # 결과코드 이상 체크
    if 'RESULTCODE' in df.columns:
        bad_codes = df[(df['RESULTCODE'] != 0) & (df['RESULTCODE'] != 4) & (df['RESULTCODE'].notna())]
        if len(bad_codes) > 0:
            for _, row in bad_codes.head(2).iterrows():
                errors.append(f"- 결과코드 {row.get('RESULTCODE')}: {row.get('MESSAGENAME', 'N/A')}")

    # 첫/끝 데이터 (3개씩, 핵심 컬럼만)
    if available_cols:
        sample_df = df[available_cols]
        sample_head = sample_df.head(3).to_string(index=False)
        sample_tail = sample_df.tail(2).to_string(index=False)
    else:
        sample_head = df.head(3).to_string(index=False)
        sample_tail = df.tail(2).to_string(index=False)

    # 장비 유형 판별
    equip_type = "AMHS"
    msg_types = list(analysis.get('message_types', {}).keys())
    if any('RAIL-' in m and 'INTERRAIL' not in m for m in msg_types):
        equip_type = "OHT"
    elif any('INTERRAIL-' in m for m in msg_types):
        equip_type = "Conveyor"
    elif any('STORAGE-' in m for m in msg_types):
        equip_type = "Lifter"

    prompt = f"""## {equip_type} 로그 ({analysis['row_count']}건)

시간: {analysis['time_range'].get('start', '')} ~ {analysis['time_range'].get('end', '')}
장비: {', '.join(analysis['machines'][:3])}
LEVEL: {analysis['levels']}

시퀀스: {msg_sequence}

"""

    if errors:
        prompt += f"""⚠️ 이상:
{chr(10).join(errors)}

"""

    prompt += f"""처음:
{sample_head}

끝:
{sample_tail}

"""

    if user_question:
        prompt += f"질문: {user_question}\n"
    else:
        prompt += "분석: 경로, 시간, 정상여부 알려줘.\n"

    return prompt

# ========================================
# FastAPI Endpoints
# ========================================
@app.on_event("startup")
async def startup():
    """Server startup initialization"""
    global llm, LLM_MODE

    # Load AMHS prompts
    load_amhs_prompts()

    # Load API token
    if load_api_token():
        LLM_MODE = "api"
        logger.info("LLM Mode: API")
    else:
        LLM_MODE = "local"
        logger.info("No API token -> trying local mode")

    # Load local LLM (backup)
    MODEL_PATH = "Qwen3-14B-Q4_K_M.gguf"

    if os.path.exists(MODEL_PATH):
        try:
            from llama_cpp import Llama
            logger.info(f"Loading LLM: {MODEL_PATH}")
            llm = Llama(
                model_path=MODEL_PATH,
                n_ctx=32768,  # 4096 -> 32768로 확장
                n_gpu_layers=-1,
                verbose=False
            )
            logger.info("Local LLM loaded!")

            if not API_TOKEN:
                LLM_MODE = "local"
        except Exception as e:
            logger.warning(f"Local LLM load failed (API only): {e}")
    else:
        logger.warning(f"Model file not found: {MODEL_PATH}")

    logger.info(f"Server ready. Mode: {LLM_MODE}")

class Query(BaseModel):
    question: str
    mode: str = "general"

class CSVAnalysisRequest(BaseModel):
    csv_data: str
    question: Optional[str] = ""

@app.get("/")
async def home():
    return FileResponse("index.html")

@app.get("/llm_status")
async def llm_status():
    """Return LLM status"""
    return {
        "mode": LLM_MODE,
        "local_available": llm is not None,
        "api_available": API_TOKEN is not None
    }

@app.post("/set_llm_mode")
async def set_llm_mode(data: dict):
    """Set LLM mode"""
    global LLM_MODE
    new_mode = data.get("llm_mode", "api")

    if new_mode == "local" and llm is None:
        return {"success": False, "message": "Local LLM not available"}
    if new_mode == "api" and API_TOKEN is None:
        return {"success": False, "message": "API token not available"}

    LLM_MODE = new_mode
    return {"success": True, "mode": LLM_MODE, "message": f"Changed to {LLM_MODE} mode"}

@app.post("/reload_token")
async def reload_token():
    """Reload API token from file"""
    global API_TOKEN, LLM_MODE
    
    try:
        token_path = "token.txt"
        
        if not os.path.exists(token_path):
            return {"success": False, "message": f"토큰 파일이 없습니다: {token_path}"}
        
        with open(token_path, "r", encoding='utf-8') as f:
            new_token = f.read().strip()
        
        if not new_token:
            return {"success": False, "message": "토큰 파일이 비어있습니다"}
        
        if "REPLACE" in new_token:
            return {"success": False, "message": "토큰이 기본값입니다. 실제 토큰으로 교체하세요"}
        
        # 토큰 업데이트
        API_TOKEN = new_token
        LLM_MODE = "api"
        
        logger.info("API token reloaded successfully")
        return {
            "success": True, 
            "message": "토큰이 성공적으로 리로드되었습니다",
            "mode": LLM_MODE
        }
        
    except Exception as e:
        logger.error(f"Token reload failed: {e}")
        return {"success": False, "message": f"토큰 리로드 실패: {str(e)}"}

# ========================================
# 개발/운영 환경 관리 API
# ========================================
@app.get("/env_status")
async def env_status():
    """Return current environment status"""
    return {
        "env": ENV_MODE,
        "url": API_URL,
        "model": API_MODEL,
        "name": ENV_CONFIG[ENV_MODE]["name"]
    }

@app.post("/set_env_mode")
async def set_env_mode(data: dict):
    """Set environment mode (dev/prod)"""
    global ENV_MODE, API_URL, API_MODEL
    
    new_env = data.get("env", "dev")
    
    if new_env not in ENV_CONFIG:
        return {"success": False, "message": "잘못된 환경입니다. 'dev' 또는 'prod'만 가능합니다."}
    
    ENV_MODE = new_env
    API_URL = ENV_CONFIG[new_env]["url"]
    API_MODEL = ENV_CONFIG[new_env]["model"]
    
    logger.info(f"Environment changed to {new_env}: {API_URL} ({API_MODEL})")
    
    return {
        "success": True,
        "env": ENV_MODE,
        "url": API_URL,
        "model": API_MODEL,
        "message": f"{ENV_CONFIG[new_env]['name']} 환경으로 전환되었습니다."
    }

# ========================================
# 프롬프트 관리 API
# ========================================
PROMPT_FILE_MAP = {
    'system_api': 'AMHS_LOG_ANALYSIS_SYSTEM_PROMPT.md',
    'system_local': 'AMHS_LOG_ANALYSIS_SYSTEM_PROMPT_LOCAL.md',
    'fewshot_api': 'AMHS_LOG_ANALYSIS_FEWSHOT_PROMPT.md',
    'fewshot_local': 'AMHS_LOG_ANALYSIS_FEWSHOT_PROMPT_LOCAL.md'
}

@app.post("/get_prompt")
async def get_prompt(data: dict):
    """Get prompt content"""
    prompt_type = data.get("prompt_type", "")
    
    if prompt_type not in PROMPT_FILE_MAP:
        return {"success": False, "message": "잘못된 프롬프트 타입"}
    
    filepath = PROMPT_FILE_MAP[prompt_type]
    
    try:
        if not os.path.exists(filepath):
            return {"success": False, "message": f"파일이 없습니다: {filepath}"}
        
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        return {"success": True, "content": content, "filename": filepath}
    
    except Exception as e:
        logger.error(f"Prompt load failed: {e}")
        return {"success": False, "message": str(e)}

@app.post("/save_prompt")
async def save_prompt(data: dict):
    """Save prompt content"""
    global AMHS_SYSTEM_PROMPT_LOCAL, AMHS_FEWSHOT_PROMPT_LOCAL
    global AMHS_SYSTEM_PROMPT_API, AMHS_FEWSHOT_PROMPT_API
    
    prompt_type = data.get("prompt_type", "")
    content = data.get("content", "")
    
    if prompt_type not in PROMPT_FILE_MAP:
        return {"success": False, "message": "잘못된 프롬프트 타입"}
    
    if not content.strip():
        return {"success": False, "message": "내용이 비어있습니다"}
    
    filepath = PROMPT_FILE_MAP[prompt_type]
    
    try:
        # 백업 생성
        if os.path.exists(filepath):
            backup_path = filepath + ".backup"
            with open(filepath, "r", encoding="utf-8") as f:
                backup_content = f.read()
            with open(backup_path, "w", encoding="utf-8") as f:
                f.write(backup_content)
        
        # 새 내용 저장
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        
        # 메모리에 로드된 프롬프트도 업데이트
        if prompt_type == 'system_api':
            AMHS_SYSTEM_PROMPT_API = content
        elif prompt_type == 'system_local':
            AMHS_SYSTEM_PROMPT_LOCAL = content
        elif prompt_type == 'fewshot_api':
            AMHS_FEWSHOT_PROMPT_API = content
        elif prompt_type == 'fewshot_local':
            AMHS_FEWSHOT_PROMPT_LOCAL = content
        
        logger.info(f"Prompt saved: {filepath}")
        return {"success": True, "message": "저장 완료", "filename": filepath}
    
    except Exception as e:
        logger.error(f"Prompt save failed: {e}")
        return {"success": False, "message": str(e)}

@app.post("/ask")
async def ask(query: Query):
    """Question handler"""
    global LLM_MODE

    if LLM_MODE == "api":
        # API: 전체 프롬프트 사용
        system_prompt = AMHS_SYSTEM_PROMPT_API + "\n\n" + AMHS_FEWSHOT_PROMPT_API
        answer = call_api_llm(query.question, system_prompt)
    else:
        # LOCAL: 경량 프롬프트 (call_local_llm 내부에서 처리)
        answer = call_local_llm(query.question)

    return {"answer": answer}

@app.post("/analyze_csv")
async def analyze_csv(request: CSVAnalysisRequest):
    """Analyze CSV data with AMHS prompts"""
    try:
        # Parse CSV
        df = parse_csv_data(request.csv_data)

        # Basic analysis
        analysis = analyze_csv_basic(df)

        # Create prompt (모드별 다른 프롬프트)
        if LLM_MODE == "api":
            prompt = create_analysis_prompt(df, analysis, request.question)
            system_prompt = AMHS_SYSTEM_PROMPT_API + "\n\n" + AMHS_FEWSHOT_PROMPT_API
            llm_response = call_api_llm(prompt, system_prompt)
        else:
            # LOCAL: 경량 프롬프트 사용
            prompt = create_analysis_prompt_local(df, analysis, request.question)
            llm_response = call_local_llm(prompt)

        return {
            "success": True,
            "basic_info": {
                "row_count": analysis["row_count"],
                "time_range": analysis["time_range"],
                "message_types": dict(list(analysis["message_types"].items())[:5]),
                "levels": analysis["levels"]
            },
            "analysis": llm_response
        }

    except Exception as e:
        logger.error(f"CSV analysis failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...), question: str = Form("")):
    """Upload and analyze CSV file"""
    try:
        # Read file
        content = await file.read()

        # Try different encodings
        csv_text = None
        for encoding in ['utf-8', 'cp949', 'euc-kr']:
            try:
                csv_text = content.decode(encoding)
                break
            except:
                continue

        if csv_text is None:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Failed to decode CSV file"}
            )

        # Parse CSV
        df = parse_csv_data(csv_text)

        # Basic analysis
        analysis = analyze_csv_basic(df)

        # Create prompt (모드별 다른 프롬프트)
        if LLM_MODE == "api":
            prompt = create_analysis_prompt(df, analysis, question)
            system_prompt = AMHS_SYSTEM_PROMPT_API + "\n\n" + AMHS_FEWSHOT_PROMPT_API
            llm_response = call_api_llm(prompt, system_prompt)
        else:
            # LOCAL: 경량 프롬프트 사용
            prompt = create_analysis_prompt_local(df, analysis, question)
            llm_response = call_local_llm(prompt)

        return {
            "success": True,
            "filename": file.filename,
            "basic_info": {
                "row_count": analysis["row_count"],
                "time_range": analysis["time_range"],
                "message_types": dict(list(analysis["message_types"].items())[:5]),
                "levels": analysis["levels"],
                "machines": analysis["machines"][:5] if analysis["machines"] else [],
                "carriers": list(analysis["carriers"])[:5] if analysis["carriers"] else []
            },
            "analysis": llm_response
        }

    except Exception as e:
        logger.error(f"CSV upload failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

# Get list of available POI CSV files
@app.get("/poi_files")
async def get_poi_files():
    """Get list of CSV files in POI folder"""
    poi_folder = "POI"
    if not os.path.exists(poi_folder):
        return {"files": []}

    files = [f for f in os.listdir(poi_folder) if f.endswith('.csv')]
    return {"files": files}

@app.post("/analyze_poi_file")
async def analyze_poi_file(data: dict):
    """Analyze a POI CSV file"""
    filename = data.get("filename", "")
    question = data.get("question", "")

    filepath = os.path.join("POI", filename)

    if not os.path.exists(filepath):
        return {"success": False, "error": f"File not found: {filename}"}

    try:
        # Read file
        df = None
        for encoding in ['utf-8', 'cp949', 'euc-kr']:
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                break
            except:
                continue

        if df is None:
            return {"success": False, "error": "Failed to read CSV file"}

        # Basic analysis
        analysis = analyze_csv_basic(df)

        # Create prompt (모드별 다른 프롬프트)
        if LLM_MODE == "api":
            prompt = create_analysis_prompt(df, analysis, question)
            system_prompt = AMHS_SYSTEM_PROMPT_API + "\n\n" + AMHS_FEWSHOT_PROMPT_API
            llm_response = call_api_llm(prompt, system_prompt)
        else:
            # LOCAL: 경량 프롬프트 사용
            prompt = create_analysis_prompt_local(df, analysis, question)
            llm_response = call_local_llm(prompt)

        return {
            "success": True,
            "filename": filename,
            "basic_info": {
                "row_count": analysis["row_count"],
                "time_range": analysis["time_range"],
                "message_types": dict(list(analysis["message_types"].items())[:5]),
                "levels": analysis["levels"],
                "machines": analysis["machines"][:5] if analysis["machines"] else [],
                "carriers": list(analysis["carriers"])[:5] if analysis["carriers"] else []
            },
            "analysis": llm_response
        }

    except Exception as e:
        logger.error(f"POI file analysis failed: {e}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)