#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AMHS Log Analysis Server (v3.0 - AMHS 분석 전용)
- CSV upload and AMHS log analysis
- 설비별 프롬프트 자동 적용 (OHT, CONVEYOR, LIFTER, FABJOB)
- FABJOB 전처리: 시간 계산, HCACK 분석, 구간별 소요시간 자동 계산
"""

import os
import re
import requests
import pandas as pd
from io import StringIO
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import logging

# FABJOB 전처리 모듈 import
try:
    from fabjob_preprocessor import analyze_fabjob, is_fabjob_data
    FABJOB_PREPROCESSOR_AVAILABLE = True
except ImportError:
    FABJOB_PREPROCESSOR_AVAILABLE = False
    logging.warning("fabjob_preprocessor.py not found. FABJOB preprocessing disabled.")

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

API_URL = ENV_CONFIG["dev"]["url"]
API_MODEL = ENV_CONFIG["dev"]["model"]

# ========================================
# 설비별 프롬프트 설정
# ========================================
EQUIP_PROMPT_DIR = "prompts"

EQUIPMENT_TYPES = {
    "OHT": {"name": "OHT (천장 이송)", "color": "#3B82F6", "prefix": "RAIL-"},
    "CONVEYOR": {"name": "Conveyor (바닥 컨베이어)", "color": "#10B981", "prefix": "INTERRAIL-"},
    "LIFTER": {"name": "Lifter (층간 이송)", "color": "#F59E0B", "prefix": "STORAGE-"},
    "FABJOB": {"name": "FABJOB (FAB간 이송)", "color": "#8B5CF6", "prefix": "VM-"}
}


# ========================================
# 설비 감지 함수
# ========================================
def detect_equipment_type(df: pd.DataFrame) -> tuple:
    """DataFrame에서 설비 유형 감지"""
    if 'MESSAGENAME' not in df.columns:
        return "UNKNOWN", {"error": "MESSAGENAME 컬럼 없음"}
    
    messages = df['MESSAGENAME'].dropna().astype(str).tolist()
    if not messages:
        return "UNKNOWN", {"error": "메시지 없음"}
    
    counts = {"OHT": 0, "CONVEYOR": 0, "LIFTER": 0, "FABJOB": 0, "UI": 0, "INV": 0}
    
    for msg in messages:
        msg_upper = str(msg).upper()
        if msg_upper.startswith("RAIL-") and "INTERRAIL" not in msg_upper:
            counts["OHT"] += 1
        elif msg_upper.startswith("INTERRAIL-"):
            counts["CONVEYOR"] += 1
        elif msg_upper.startswith("STORAGE-"):
            counts["LIFTER"] += 1
        elif msg_upper.startswith("VM-"):
            counts["FABJOB"] += 1
        elif msg_upper.startswith("UI-"):
            counts["UI"] += 1
        elif msg_upper.startswith("INV-"):
            counts["INV"] += 1
    
    main_counts = {k: v for k, v in counts.items() if k in ["OHT", "CONVEYOR", "LIFTER", "FABJOB"]}
    total = sum(main_counts.values())
    
    if total == 0:
        return "UNKNOWN", {"counts": counts}
    
    max_type = max(main_counts, key=main_counts.get)
    max_count = main_counts[max_type]
    
    # FABJOB + 다른 설비 = FABJOB (FAB간 이송은 여러 설비 포함)
    if counts["FABJOB"] > 0 and sum(v for k, v in main_counts.items() if k != "FABJOB") > 0:
        equipment_type = "FABJOB"
    else:
        equipment_type = max_type
    
    return equipment_type, {
        "counts": counts,
        "primary": max_type,
        "ratio": round(max_count / total * 100, 1) if total > 0 else 0
    }


def detect_equipment_from_filename(filename: str) -> str:
    """파일명에서 설비 유형 추정"""
    if not filename:
        return None
    fn = filename.upper()
    if "LIFTER" in fn or "LFT" in fn:
        return "LIFTER"
    elif "CONVEYOR" in fn or "CNV" in fn:
        return "CONVEYOR"
    elif "OHT" in fn:
        return "OHT"
    elif "FABJOB" in fn:
        return "FABJOB"
    return None


def get_equipment_prompts(equipment_type: str) -> tuple:
    """설비별 프롬프트 로드 (common + system + fewshot)"""
    # 공통 프롬프트
    common_path = os.path.join(EQUIP_PROMPT_DIR, "BASE", "common.txt")
    common = ""
    if os.path.exists(common_path):
        with open(common_path, "r", encoding="utf-8") as f:
            common = f.read()
    
    # 설비별 프롬프트
    system_path = os.path.join(EQUIP_PROMPT_DIR, equipment_type, "system.txt")
    fewshot_path = os.path.join(EQUIP_PROMPT_DIR, equipment_type, "fewshot.txt")
    
    system = ""
    fewshot = ""
    
    if os.path.exists(system_path):
        with open(system_path, "r", encoding="utf-8") as f:
            system = f.read()
    
    if os.path.exists(fewshot_path):
        with open(fewshot_path, "r", encoding="utf-8") as f:
            fewshot = f.read()
    
    full_system = f"{common}\n\n{system}"
    return full_system, fewshot


def get_default_prompt() -> str:
    """기본 프롬프트 (설비 감지 실패 시)"""
    common_path = os.path.join(EQUIP_PROMPT_DIR, "BASE", "common.txt")
    if os.path.exists(common_path):
        with open(common_path, "r", encoding="utf-8") as f:
            return f.read()
    return "You are an AMHS log analysis expert. Analyze the data in Korean."


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

    for attempt in range(2):
        try:
            response = requests.post(API_URL, headers=headers, json=data, timeout=300)

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return f"API error: {response.status_code}\n{response.text}"

        except requests.exceptions.Timeout:
            logger.warning(f"API timeout (attempt {attempt + 1}/2)")
            if attempt == 0:
                continue
            return "API 요청 시간 초과 (5분). 서버 상태를 확인하세요."
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return f"API call failed: {e}"
    
    return "API 호출 실패"


def call_local_llm(prompt: str, system_prompt: str = "", max_tokens: int = 1500) -> str:
    """Call local LLM"""
    global llm

    if llm is None:
        return "Local LLM not loaded."

    try:
        if not system_prompt:
            system_prompt = get_default_prompt()
            
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

        # Qwen3 thinking 태그 제거
        result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL)
        result = re.sub(r'<think>.*', '', result, flags=re.DOTALL)

        # 영어 thinking 블록 제거
        korean_match = re.search(r'[가-힣]', result)
        if korean_match:
            korean_start = korean_match.start()
            if korean_start > 100:
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
        "machines": [],
        "carriers": []
    }

    if 'MESSAGENAME' in df.columns:
        analysis["message_types"] = df['MESSAGENAME'].value_counts().to_dict()

    if 'TIME_EX' in df.columns:
        times = df['TIME_EX'].dropna().tolist()
        if times:
            analysis["time_range"] = {"start": str(times[0]), "end": str(times[-1])}

    if 'LEVEL' in df.columns:
        analysis["levels"] = df['LEVEL'].value_counts().to_dict()

    if 'MACHINENAME' in df.columns:
        analysis["machines"] = list(df['MACHINENAME'].dropna().unique()[:10])

    if 'CARRIER' in df.columns:
        analysis["carriers"] = list(df['CARRIER'].dropna().unique()[:5])

    return analysis


def create_analysis_prompt(df: pd.DataFrame, analysis: dict, user_question: str = "", 
                          fabjob_preprocess: str = "") -> str:
    """Create prompt for LLM analysis"""
    
    # FABJOB 전처리 결과가 있으면 그걸 메인으로 사용
    if fabjob_preprocess:
        prompt = f"""## FABJOB 로그 분석 요청

{fabjob_preprocess}

"""
        if user_question:
            prompt += f"""### 추가 질문
{user_question}

위 분석 결과와 추가 질문을 바탕으로 답변해주세요.
"""
        return prompt
    
    # 일반 분석 (기존 로직)
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


# ========================================
# 공통 분석 함수
# ========================================
def analyze_amhs_log(df: pd.DataFrame, question: str = "", filename: str = "") -> dict:
    """AMHS 로그 분석 공통 함수 - 설비별 프롬프트 자동 적용 + FABJOB 전처리"""
    # 기본 분석
    analysis = analyze_csv_basic(df)
    
    # 설비 유형 감지
    equipment_type, equip_details = detect_equipment_type(df)
    filename_hint = detect_equipment_from_filename(filename)
    logger.info(f"Detected equipment: {equipment_type} (file hint: {filename_hint})")
    
    # FABJOB 전처리
    fabjob_preprocess = ""
    fabjob_analysis = None
    
    if equipment_type == "FABJOB" and FABJOB_PREPROCESSOR_AVAILABLE:
        try:
            logger.info("FABJOB detected - running preprocessor...")
            fabjob_analysis = analyze_fabjob(df)
            fabjob_preprocess = fabjob_analysis.get('preprocessed_text', '')
            logger.info(f"FABJOB preprocessing complete. Text length: {len(fabjob_preprocess)}")
        except Exception as e:
            logger.error(f"FABJOB preprocessing failed: {e}")
            fabjob_preprocess = ""
    
    # 프롬프트 생성
    prompt = create_analysis_prompt(df, analysis, question, fabjob_preprocess)
    
    # 설비별 프롬프트 로드
    equip_prompt_path = os.path.join(EQUIP_PROMPT_DIR, equipment_type)
    if equipment_type != "UNKNOWN" and os.path.exists(equip_prompt_path):
        equip_system, equip_fewshot = get_equipment_prompts(equipment_type)
        system_prompt = equip_system + "\n\n" + equip_fewshot
        logger.info(f"Using equipment-specific prompt for {equipment_type}")
    else:
        system_prompt = get_default_prompt()
        logger.info("Using default prompt")
    
    # LLM 호출
    if LLM_MODE == "api":
        llm_response = call_api_llm(prompt, system_prompt)
    else:
        llm_response = call_local_llm(prompt, system_prompt)
    
    # 결과 구성
    result = {
        "success": True,
        "equipment_type": equipment_type,
        "equipment_details": equip_details,
        "basic_info": {
            "row_count": analysis["row_count"],
            "time_range": analysis["time_range"],
            "message_types": dict(list(analysis["message_types"].items())[:5]),
            "levels": analysis["levels"],
            "machines": analysis["machines"][:5] if analysis["machines"] else [],
            "carriers": analysis["carriers"][:5] if analysis["carriers"] else []
        },
        "analysis": llm_response
    }
    
    # FABJOB 전처리 결과 추가
    if fabjob_analysis:
        result["fabjob_details"] = {
            "carrier_id": fabjob_analysis.get('carrier_id'),
            "lot_id": fabjob_analysis.get('lot_id'),
            "total_duration_sec": fabjob_analysis.get('total_duration_sec', 0),
            "final_status": fabjob_analysis.get('final_status'),
            "delays": fabjob_analysis.get('delays', []),
            "hcack_errors": len([h for h in fabjob_analysis.get('hcack_events', []) if h.get('hcack') == '2']),
        }
    
    return result


# ========================================
# FastAPI Startup
# ========================================
@app.on_event("startup")
async def startup():
    """Server startup initialization"""
    global llm, LLM_MODE

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
                n_ctx=32768,
                n_gpu_layers=-1,
                verbose=False
            )
            logger.info("Local LLM loaded!")
            if not API_TOKEN:
                LLM_MODE = "local"
        except Exception as e:
            logger.warning(f"Local LLM load failed: {e}")
    else:
        logger.warning(f"Model file not found: {MODEL_PATH}")

    # 설비별 프롬프트 폴더 확인
    if os.path.exists(EQUIP_PROMPT_DIR):
        logger.info(f"Equipment prompts found: {EQUIP_PROMPT_DIR}")
    else:
        logger.warning(f"Equipment prompts not found: {EQUIP_PROMPT_DIR}")

    # FABJOB 전처리 모듈 확인
    if FABJOB_PREPROCESSOR_AVAILABLE:
        logger.info("FABJOB preprocessor available")
    else:
        logger.warning("FABJOB preprocessor NOT available")

    logger.info(f"Server ready. Mode: {LLM_MODE}")


# ========================================
# 기본 API
# ========================================
@app.get("/")
async def home():
    return FileResponse("index.html")


@app.get("/llm_status")
async def llm_status():
    return {
        "mode": LLM_MODE,
        "local_available": llm is not None,
        "api_available": API_TOKEN is not None,
        "fabjob_preprocessor": FABJOB_PREPROCESSOR_AVAILABLE
    }


@app.post("/set_llm_mode")
async def set_llm_mode(data: dict):
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
        
        API_TOKEN = new_token
        LLM_MODE = "api"
        logger.info("API token reloaded successfully")
        return {"success": True, "message": "토큰이 성공적으로 리로드되었습니다", "mode": LLM_MODE}
    except Exception as e:
        logger.error(f"Token reload failed: {e}")
        return {"success": False, "message": f"토큰 리로드 실패: {str(e)}"}


# ========================================
# 환경 관리 API
# ========================================
@app.get("/env_status")
async def env_status():
    return {
        "env": ENV_MODE,
        "url": API_URL,
        "model": API_MODEL,
        "name": ENV_CONFIG[ENV_MODE]["name"]
    }


@app.post("/set_env_mode")
async def set_env_mode(data: dict):
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
# 설비별 프롬프트 관리 API
# ========================================
@app.get("/equipment_types")
async def get_equipment_types():
    """설비 유형 정보 반환"""
    return {"types": EQUIPMENT_TYPES}


@app.get("/equip_prompts")
async def get_equip_prompts():
    """모든 설비별 프롬프트 조회"""
    try:
        result = {}
        
        # BASE
        base_path = os.path.join(EQUIP_PROMPT_DIR, "BASE", "common.txt")
        if os.path.exists(base_path):
            with open(base_path, "r", encoding="utf-8") as f:
                result["BASE"] = {"common": f.read()}
        else:
            result["BASE"] = {"common": ""}
        
        # 각 설비별
        for equip in ["OHT", "CONVEYOR", "LIFTER", "FABJOB"]:
            result[equip] = {}
            for ptype in ["system", "fewshot"]:
                filepath = os.path.join(EQUIP_PROMPT_DIR, equip, f"{ptype}.txt")
                if os.path.exists(filepath):
                    with open(filepath, "r", encoding="utf-8") as f:
                        result[equip][ptype] = f.read()
                else:
                    result[equip][ptype] = ""
        
        return {"success": True, "prompts": result}
    except Exception as e:
        logger.error(f"Get prompts failed: {e}")
        return {"success": False, "message": str(e)}


@app.post("/save_equip_prompt")
async def save_equipment_prompt(data: dict):
    """설비별 프롬프트 저장"""
    try:
        equip_type = data.get("equipment_type", "")
        prompt_type = data.get("prompt_type", "")
        content = data.get("content", "")
        
        if not equip_type or not prompt_type:
            return {"success": False, "message": "설비 타입과 프롬프트 타입 필요"}
        
        dir_path = os.path.join(EQUIP_PROMPT_DIR, equip_type)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        filepath = os.path.join(dir_path, f"{prompt_type}.txt")
        
        # 백업
        if os.path.exists(filepath):
            backup_path = filepath + ".backup"
            with open(filepath, "r", encoding="utf-8") as f:
                with open(backup_path, "w", encoding="utf-8") as bf:
                    bf.write(f.read())
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        
        logger.info(f"Equipment prompt saved: {filepath}")
        return {"success": True, "message": "저장 완료", "filepath": filepath}
    except Exception as e:
        logger.error(f"Save prompt failed: {e}")
        return {"success": False, "message": str(e)}


# ========================================
# AMHS 분석 API
# ========================================
@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...), question: str = Form("")):
    """Upload and analyze CSV file"""
    try:
        content = await file.read()
        
        csv_text = None
        for encoding in ['utf-8', 'cp949', 'euc-kr']:
            try:
                csv_text = content.decode(encoding)
                break
            except:
                continue

        if csv_text is None:
            return JSONResponse(status_code=400, content={"success": False, "error": "Failed to decode CSV file"})

        df = parse_csv_data(csv_text)
        result = analyze_amhs_log(df, question, file.filename)
        result["filename"] = file.filename
        return result

    except Exception as e:
        logger.error(f"CSV upload failed: {e}")
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


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
        df = None
        for encoding in ['utf-8', 'cp949', 'euc-kr']:
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                break
            except:
                continue

        if df is None:
            return {"success": False, "error": "Failed to read CSV file"}

        result = analyze_amhs_log(df, question, filename)
        result["filename"] = filename
        return result

    except Exception as e:
        logger.error(f"POI file analysis failed: {e}")
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)