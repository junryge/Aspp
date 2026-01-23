#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AMHS Log Analysis Server (v5.0 - ë°°ì¹˜ ì²˜ë¦¬ ì§€ì›)
- CSV upload and AMHS log analysis
- ì„¤ë¹„ë³„ í”„ë¡¬í”„íŠ¸ ìžë™ ì ìš© (OHT, CONVEYOR, LIFTER, FABJOB)
- ì „ì„¤ë¹„ ì „ì²˜ë¦¬: ì‹œê°„ ê³„ì‚°, HCACK ë¶„ì„, êµ¬ê°„ë³„ ì†Œìš”ì‹œê°„ ìžë™ ê³„ì‚°
- ë‹¤ì¤‘ íŒŒì¼ ë°°ì¹˜ ì²˜ë¦¬ ì§€ì›
"""

import os
import re
import requests
import pandas as pd
from io import StringIO
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import logging

# ========================================
# ì „ì²˜ë¦¬ ëª¨ë“ˆ Import (4ê°œ ì„¤ë¹„)
# ========================================
# FABJOB ì „ì²˜ë¦¬ ëª¨ë“ˆ
try:
    from fabjob_preprocessor import analyze_fabjob, is_fabjob_data
    FABJOB_PREPROCESSOR_AVAILABLE = True
except ImportError:
    FABJOB_PREPROCESSOR_AVAILABLE = False
    logging.warning("fabjob_preprocessor.py not found. FABJOB preprocessing disabled.")

# OHT ì „ì²˜ë¦¬ ëª¨ë“ˆ
try:
    from oht_preprocessor import analyze_oht, is_oht_data
    OHT_PREPROCESSOR_AVAILABLE = True
except ImportError:
    OHT_PREPROCESSOR_AVAILABLE = False
    logging.warning("oht_preprocessor.py not found. OHT preprocessing disabled.")

# CONVEYOR ì „ì²˜ë¦¬ ëª¨ë“ˆ
try:
    from conveyor_preprocessor import analyze_conveyor, is_conveyor_data
    CONVEYOR_PREPROCESSOR_AVAILABLE = True
except ImportError:
    CONVEYOR_PREPROCESSOR_AVAILABLE = False
    logging.warning("conveyor_preprocessor.py not found. CONVEYOR preprocessing disabled.")

# LIFTER ì „ì²˜ë¦¬ ëª¨ë“ˆ
try:
    from lifter_preprocessor import analyze_lifter, is_lifter_data
    LIFTER_PREPROCESSOR_AVAILABLE = True
except ImportError:
    LIFTER_PREPROCESSOR_AVAILABLE = False
    logging.warning("lifter_preprocessor.py not found. LIFTER preprocessing disabled.")

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

# ê°œë°œ/ìš´ì˜ í™˜ê²½ ì„¤ì •
ENV_MODE = "dev"  # "dev" or "prod"

ENV_CONFIG = {
    "dev": {
        "url": "http://dev.assistant.llm.skhynix.com/v1/chat/completions",
        "model": "Qwen3-Coder-30B-A3B-Instruct",
        "name": "ê°œë°œ(30B)"
    },
    "prod": {
        "url": "http://summary.llm.skhynix.com/v1/chat/completions",
        "model": "Qwen3-Next-80B-A3B-Instruct",
        "name": "ìš´ì˜(80B)"
    }
}

API_URL = ENV_CONFIG["dev"]["url"]
API_MODEL = ENV_CONFIG["dev"]["model"]

# ========================================
# ì„¤ë¹„ë³„ í”„ë¡¬í”„íŠ¸ ì„¤ì •
# ========================================
EQUIP_PROMPT_DIR = "prompts"

EQUIPMENT_TYPES = {
    "OHT": {"name": "OHT (ì²œìž¥ ì´ì†¡)", "color": "#3B82F6", "prefix": "RAIL-"},
    "CONVEYOR": {"name": "Conveyor (ë°”ë‹¥ ì»¨ë² ì´ì–´)", "color": "#10B981", "prefix": "INTERRAIL-"},
    "LIFTER": {"name": "Lifter (ì¸µê°„ ì´ì†¡)", "color": "#F59E0B", "prefix": "STORAGE-"},
    "FABJOB": {"name": "FABJOB (FABê°„ ì´ì†¡)", "color": "#8B5CF6", "prefix": "VM-"}
}


# ========================================
# ìœ í‹¸ë¦¬í‹° í•¨ìˆ˜
# ========================================
def format_duration_simple(seconds: float) -> str:
    """ì´ˆë¥¼ ì½ê¸° ì¢‹ì€ í˜•ì‹ìœ¼ë¡œ ë³€í™˜"""
    if seconds < 0:
        return "N/A"
    if seconds < 60:
        return f"{seconds:.0f}ì´ˆ"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}ë¶„ {secs}ì´ˆ"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}ì‹œê°„ {mins}ë¶„"


# ========================================
# ì„¤ë¹„ ê°ì§€ í•¨ìˆ˜
# ========================================
def detect_equipment_type(df: pd.DataFrame) -> tuple:
    """DataFrameì—ì„œ ì„¤ë¹„ ìœ í˜• ê°ì§€"""
    if 'MESSAGENAME' not in df.columns:
        return "UNKNOWN", {"error": "MESSAGENAME ì»¬ëŸ¼ ì—†ìŒ"}
    
    messages = df['MESSAGENAME'].dropna().astype(str).tolist()
    if not messages:
        return "UNKNOWN", {"error": "ë©”ì‹œì§€ ì—†ìŒ"}
    
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
    
    # FABJOB + ë‹¤ë¥¸ ì„¤ë¹„ = FABJOB (FABê°„ ì´ì†¡ì€ ì—¬ëŸ¬ ì„¤ë¹„ í¬í•¨)
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
    """íŒŒì¼ëª…ì—ì„œ ì„¤ë¹„ ìœ í˜• ì¶”ì •"""
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
    """ì„¤ë¹„ë³„ í”„ë¡¬í”„íŠ¸ ë¡œë“œ (common + system + fewshot)"""
    # ê³µí†µ í”„ë¡¬í”„íŠ¸
    common_path = os.path.join(EQUIP_PROMPT_DIR, "BASE", "common.txt")
    common = ""
    if os.path.exists(common_path):
        with open(common_path, "r", encoding="utf-8") as f:
            common = f.read()
    
    # ì„¤ë¹„ë³„ í”„ë¡¬í”„íŠ¸
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
    """ê¸°ë³¸ í”„ë¡¬í”„íŠ¸ (ì„¤ë¹„ ê°ì§€ ì‹¤íŒ¨ ì‹œ)"""
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
            return "API ìš”ì²­ ì‹œê°„ ì´ˆê³¼ (5ë¶„). ì„œë²„ ìƒíƒœë¥¼ í™•ì¸í•˜ì„¸ìš”."
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return f"API call failed: {e}"
    
    return "API í˜¸ì¶œ ì‹¤íŒ¨"


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

        # Qwen3 thinking íƒœê·¸ ì œê±°
        result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL)
        result = re.sub(r'<think>.*', '', result, flags=re.DOTALL)

        # ì˜ì–´ thinking ë¸”ë¡ ì œê±°
        korean_match = re.search(r'[ê°€-íž£]', result)
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
                          preprocess_text: str = "") -> str:
    """Create prompt for LLM analysis"""
    
    # ì „ì²˜ë¦¬ ê²°ê³¼ê°€ ìžˆìœ¼ë©´ ê·¸ê±¸ ë©”ì¸ìœ¼ë¡œ ì‚¬ìš©
    if preprocess_text:
        prompt = f"""## ë¡œê·¸ ë¶„ì„ ìš”ì²­

{preprocess_text}

"""
        if user_question:
            prompt += f"""### ì¶”ê°€ ì§ˆë¬¸
{user_question}

ìœ„ ë¶„ì„ ê²°ê³¼ì™€ ì¶”ê°€ ì§ˆë¬¸ì„ ë°”íƒ•ìœ¼ë¡œ ë‹µë³€í•´ì£¼ì„¸ìš”.
"""
        return prompt
    
    # ì¼ë°˜ ë¶„ì„ (ê¸°ì¡´ ë¡œì§)
    sample_head = df.head(5).to_string()
    sample_tail = df.tail(5).to_string()

    prompt = f"""## CSV ë°ì´í„° ë¶„ì„ ìš”ì²­

### íŒŒì¼ ê¸°ë³¸ ì •ë³´
- ì´ ë ˆì½”ë“œ ìˆ˜: {analysis['row_count']}ê±´
- ì»¬ëŸ¼: {', '.join(analysis['columns'][:10])}
- ì‹œê°„ ë²”ìœ„: {analysis['time_range'].get('start', 'N/A')} ~ {analysis['time_range'].get('end', 'N/A')}

### ë©”ì‹œì§€ ìœ í˜• ë¶„í¬
{dict(list(analysis['message_types'].items())[:10])}

### LEVEL ë¶„í¬
{analysis['levels']}

### ê´€ë ¨ ìž¥ë¹„
{analysis['machines']}

### ìºë¦¬ì–´
{analysis['carriers']}

### ë°ì´í„° ìƒ˜í”Œ (ì²˜ìŒ 5ê°œ)
{sample_head}

### ë°ì´í„° ìƒ˜í”Œ (ë§ˆì§€ë§‰ 5ê°œ)
{sample_tail}

"""

    if user_question:
        prompt += f"""### ì‚¬ìš©ìž ì§ˆë¬¸
{user_question}

ìœ„ ë°ì´í„°ë¥¼ ë¶„ì„í•˜ê³  ì‚¬ìš©ìž ì§ˆë¬¸ì— ë‹µë³€í•´ì£¼ì„¸ìš”.
"""
    else:
        prompt += """### ìš”ì²­
ìœ„ AMHS ë¡œê·¸ ë°ì´í„°ë¥¼ ë¶„ì„í•˜ê³ , ìžì—°ìŠ¤ëŸ¬ìš´ í•œêµ­ì–´ë¡œ ì„¤ëª…í•´ì£¼ì„¸ìš”.
ì´ì†¡ ê²½ë¡œ, ì†Œìš”ì‹œê°„, ì •ìƒ/ì´ìƒ ì—¬ë¶€ ë“±ì„ í¬í•¨í•´ì£¼ì„¸ìš”.
"""

    return prompt


# ========================================
# ê³µí†µ ë¶„ì„ í•¨ìˆ˜ (4ê°œ ì„¤ë¹„ ì „ì²˜ë¦¬ í†µí•©)
# ========================================
def analyze_amhs_log(df: pd.DataFrame, question: str = "", filename: str = "") -> dict:
    """AMHS ë¡œê·¸ ë¶„ì„ ê³µí†µ í•¨ìˆ˜ - ì„¤ë¹„ë³„ í”„ë¡¬í”„íŠ¸ ë° ì „ì²˜ë¦¬ ìžë™ ì ìš©"""
    # ê¸°ë³¸ ë¶„ì„
    analysis = analyze_csv_basic(df)
    
    # ì„¤ë¹„ ìœ í˜• ê°ì§€
    equipment_type, equip_details = detect_equipment_type(df)
    logger.info(f"Detected equipment: {equipment_type}, details: {equip_details}")
    
    # ========================================
    # AMHS log validation (content-based)
    # ========================================
    if equipment_type == "UNKNOWN":
        logger.warning(f"Not an AMHS log file: {filename}")
        return {
            "success": False,
            "error": "\xea\xb4\x80\xea\xb3\x84\xec\x97\x86\xeb\x8a\x94 \xeb\x8d\xb0\xec\x9d\xb4\xed\x84\xb0\xec\x9e\x85\xeb\x8b\x88\xeb\x8b\xa4. AMHS \xeb\xa1\x9c\xea\xb7\xb8 \xed\x8c\x8c\xec\x9d\xbc\xec\x9d\x84 \xec\x83\x88\xeb\xa1\x9c \xec\xb0\xbe\xec\x95\x84\xec\x84\x9c \xec\x97\x85\xeb\xa1\x9c\xeb\x93\x9c \xed\x95\xb4\xec\xa3\xbc\xec\x84\xb8\xec\x9a\x94.",
            "filename": filename,
            "equipment_type": "UNKNOWN",
            "basic_info": {
                "row_count": analysis["row_count"],
                "columns": analysis["columns"][:10],
                "message_types": dict(list(analysis["message_types"].items())[:5]) if analysis["message_types"] else {}
            },
            "analysis": "이 파일은 AMHS 로그 형식이 아닙니다. OHT, Conveyor, Lifter, FABJOB 관련 CSV 파일을 업로드 해주세요."
        }
    
    # ========================================
    # ì„¤ë¹„ë³„ ì „ì²˜ë¦¬ ì‹¤í–‰
    # ========================================
    preprocess_text = ""
    preprocess_result = None
    
    if equipment_type == "FABJOB" and FABJOB_PREPROCESSOR_AVAILABLE:
        try:
            logger.info("FABJOB detected - running preprocessor...")
            preprocess_result = analyze_fabjob(df)
            preprocess_text = preprocess_result.get('preprocessed_text', '')
            logger.info(f"FABJOB preprocessing complete. Text length: {len(preprocess_text)}")
        except Exception as e:
            logger.error(f"FABJOB preprocessing failed: {e}")
    
    elif equipment_type == "OHT" and OHT_PREPROCESSOR_AVAILABLE:
        try:
            logger.info("OHT detected - running preprocessor...")
            preprocess_result = analyze_oht(df)
            preprocess_text = preprocess_result.get('preprocessed_text', '')
            logger.info(f"OHT preprocessing complete. Text length: {len(preprocess_text)}")
        except Exception as e:
            logger.error(f"OHT preprocessing failed: {e}")
    
    elif equipment_type == "CONVEYOR" and CONVEYOR_PREPROCESSOR_AVAILABLE:
        try:
            logger.info("CONVEYOR detected - running preprocessor...")
            preprocess_result = analyze_conveyor(df)
            preprocess_text = preprocess_result.get('preprocessed_text', '')
            logger.info(f"CONVEYOR preprocessing complete. Text length: {len(preprocess_text)}")
        except Exception as e:
            logger.error(f"CONVEYOR preprocessing failed: {e}")
    
    elif equipment_type == "LIFTER" and LIFTER_PREPROCESSOR_AVAILABLE:
        try:
            logger.info("LIFTER detected - running preprocessor...")
            preprocess_result = analyze_lifter(df)
            preprocess_text = preprocess_result.get('preprocessed_text', '')
            logger.info(f"LIFTER preprocessing complete. Text length: {len(preprocess_text)}")
        except Exception as e:
            logger.error(f"LIFTER preprocessing failed: {e}")
    
    # ========================================
    # í”„ë¡¬í”„íŠ¸ ìƒì„±
    # ========================================
    prompt = create_analysis_prompt(df, analysis, question, preprocess_text)
    
    # ì„¤ë¹„ë³„ í”„ë¡¬í”„íŠ¸ ë¡œë“œ
    equip_prompt_path = os.path.join(EQUIP_PROMPT_DIR, equipment_type)
    if equipment_type != "UNKNOWN" and os.path.exists(equip_prompt_path):
        equip_system, equip_fewshot = get_equipment_prompts(equipment_type)
        system_prompt = equip_system + "\n\n" + equip_fewshot
        logger.info(f"Using equipment-specific prompt for {equipment_type}")
    else:
        system_prompt = get_default_prompt()
        logger.info("Using default prompt")
    
    # LLM í˜¸ì¶œ
    if LLM_MODE == "api":
        llm_response = call_api_llm(prompt, system_prompt)
    else:
        llm_response = call_local_llm(prompt, system_prompt)
    
    # ê²°ê³¼ êµ¬ì„±
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
    
    # ì „ì²˜ë¦¬ ê²°ê³¼ ìƒì„¸ ì •ë³´ ì¶”ê°€
    if preprocess_result:
        result["preprocess_details"] = {
            "carrier_id": preprocess_result.get('carrier_id'),
            "total_duration_sec": preprocess_result.get('total_duration_sec', 0),
            "final_status": preprocess_result.get('final_status'),
            "delays": preprocess_result.get('delays', []),
        }
        
        # ì„¤ë¹„ë³„ ì¶”ê°€ ì •ë³´
        if equipment_type == "FABJOB":
            result["preprocess_details"]["lot_id"] = preprocess_result.get('lot_id')
            result["preprocess_details"]["hcack_errors"] = len([
                h for h in preprocess_result.get('hcack_events', []) 
                if h.get('hcack') == '2'
            ])
        elif equipment_type == "OHT":
            result["preprocess_details"]["vehicle_id"] = preprocess_result.get('vehicle_id')
            result["preprocess_details"]["source_port"] = preprocess_result.get('source_port')
            result["preprocess_details"]["dest_port"] = preprocess_result.get('dest_port')
        elif equipment_type == "CONVEYOR":
            result["preprocess_details"]["machine_name"] = preprocess_result.get('machine_name')
            result["preprocess_details"]["source_zone"] = preprocess_result.get('source_zone')
            result["preprocess_details"]["dest_zone"] = preprocess_result.get('dest_zone')
        elif equipment_type == "LIFTER":
            result["preprocess_details"]["machine_name"] = preprocess_result.get('machine_name')
            result["preprocess_details"]["source_floor"] = preprocess_result.get('source_floor')
            result["preprocess_details"]["dest_floor"] = preprocess_result.get('dest_floor')
    
    return result


# ========================================
# ë°°ì¹˜ ë¶„ì„ ìš”ì•½ ìƒì„±
# ========================================
def generate_batch_summary(results: list) -> dict:
    """ë°°ì¹˜ ë¶„ì„ ìš”ì•½ ìƒì„±"""
    success_results = [r for r in results if r.get('success')]
    fail_results = [r for r in results if not r.get('success')]
    
    # ì´ ì†Œìš”ì‹œê°„ í•©ê³„
    total_duration = sum(
        r.get('preprocess_details', {}).get('total_duration_sec', 0) 
        for r in success_results
    )
    
    # ì„¤ë¹„ ìœ í˜• ë¶„í¬
    equipment_counts = {}
    for r in success_results:
        eq_type = r.get('equipment_type', 'UNKNOWN')
        equipment_counts[eq_type] = equipment_counts.get(eq_type, 0) + 1
    
    # ì§€ì—° ë°œìƒ íŒŒì¼ ëª©ë¡
    delay_files = []
    for r in success_results:
        delays = r.get('preprocess_details', {}).get('delays', [])
        if delays:
            delay_files.append({
                "filename": r.get('filename'),
                "delay_count": len(delays),
                "total_duration_str": format_duration_simple(r.get('preprocess_details', {}).get('total_duration_sec', 0)),
                "main_delay": delays[0] if delays else None
            })
    
    # ì´ ë ˆì½”ë“œ ìˆ˜
    total_records = sum(
        r.get('basic_info', {}).get('row_count', 0) 
        for r in success_results
    )
    
    # ì •ìƒ ì™„ë£Œ íŒŒì¼ ìˆ˜
    normal_files = len(success_results) - len(delay_files)
    
    return {
        "success_count": len(success_results),
        "fail_count": len(fail_results),
        "total_files": len(results),
        "total_records": total_records,
        "total_duration_sec": total_duration,
        "total_duration_str": format_duration_simple(total_duration),
        "equipment_distribution": equipment_counts,
        "normal_files": normal_files,
        "delay_files": delay_files,
        "delay_file_count": len(delay_files),
        "failed_files": [{"filename": r.get('filename'), "error": r.get('error')} for r in fail_results]
    }


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
    MODEL_PATH = "Qwen3-8B-Q6_K.gguf"

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

    # ì„¤ë¹„ë³„ í”„ë¡¬í”„íŠ¸ í´ë” í™•ì¸
    if os.path.exists(EQUIP_PROMPT_DIR):
        logger.info(f"Equipment prompts found: {EQUIP_PROMPT_DIR}")
    else:
        logger.warning(f"Equipment prompts not found: {EQUIP_PROMPT_DIR}")

    # ========================================
    # ì „ì²˜ë¦¬ ëª¨ë“ˆ ìƒíƒœ ë¡œê·¸
    # ========================================
    logger.info("=== Preprocessor Modules Status ===")
    logger.info(f"  FABJOB  : {'âœ… Available' if FABJOB_PREPROCESSOR_AVAILABLE else 'âŒ Not found'}")
    logger.info(f"  OHT     : {'âœ… Available' if OHT_PREPROCESSOR_AVAILABLE else 'âŒ Not found'}")
    logger.info(f"  CONVEYOR: {'âœ… Available' if CONVEYOR_PREPROCESSOR_AVAILABLE else 'âŒ Not found'}")
    logger.info(f"  LIFTER  : {'âœ… Available' if LIFTER_PREPROCESSOR_AVAILABLE else 'âŒ Not found'}")
    logger.info("===================================")

    logger.info(f"Server ready. Mode: {LLM_MODE}")


# ========================================
# ê¸°ë³¸ API
# ========================================
@app.get("/")
async def home():
    return FileResponse("index.html")


@app.get("/llm_status")
async def llm_status():
    """LLM ë° ì „ì²˜ë¦¬ ëª¨ë“ˆ ìƒíƒœ ì¡°íšŒ"""
    return {
        "mode": LLM_MODE,
        "local_available": llm is not None,
        "api_available": API_TOKEN is not None,
        "preprocessors": {
            "fabjob": FABJOB_PREPROCESSOR_AVAILABLE,
            "oht": OHT_PREPROCESSOR_AVAILABLE,
            "conveyor": CONVEYOR_PREPROCESSOR_AVAILABLE,
            "lifter": LIFTER_PREPROCESSOR_AVAILABLE,
        }
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
            return {"success": False, "message": f"í† í° íŒŒì¼ì´ ì—†ìŠµë‹ˆë‹¤: {token_path}"}
        
        with open(token_path, "r", encoding='utf-8') as f:
            new_token = f.read().strip()
        
        if not new_token:
            return {"success": False, "message": "í† í° íŒŒì¼ì´ ë¹„ì–´ìžˆìŠµë‹ˆë‹¤"}
        if "REPLACE" in new_token:
            return {"success": False, "message": "í† í°ì´ ê¸°ë³¸ê°’ìž…ë‹ˆë‹¤. ì‹¤ì œ í† í°ìœ¼ë¡œ êµì²´í•˜ì„¸ìš”"}
        
        API_TOKEN = new_token
        LLM_MODE = "api"
        logger.info("API token reloaded successfully")
        return {"success": True, "message": "í† í°ì´ ì„±ê³µì ìœ¼ë¡œ ë¦¬ë¡œë“œë˜ì—ˆìŠµë‹ˆë‹¤", "mode": LLM_MODE}
    except Exception as e:
        logger.error(f"Token reload failed: {e}")
        return {"success": False, "message": f"í† í° ë¦¬ë¡œë“œ ì‹¤íŒ¨: {str(e)}"}


# ========================================
# í™˜ê²½ ê´€ë¦¬ API
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
        return {"success": False, "message": "ìž˜ëª»ëœ í™˜ê²½ìž…ë‹ˆë‹¤. 'dev' ë˜ëŠ” 'prod'ë§Œ ê°€ëŠ¥í•©ë‹ˆë‹¤."}
    
    ENV_MODE = new_env
    API_URL = ENV_CONFIG[new_env]["url"]
    API_MODEL = ENV_CONFIG[new_env]["model"]
    
    logger.info(f"Environment changed to {new_env}: {API_URL} ({API_MODEL})")
    return {
        "success": True,
        "env": ENV_MODE,
        "url": API_URL,
        "model": API_MODEL,
        "message": f"{ENV_CONFIG[new_env]['name']} í™˜ê²½ìœ¼ë¡œ ì „í™˜ë˜ì—ˆìŠµë‹ˆë‹¤."
    }


# ========================================
# ì„¤ë¹„ë³„ í”„ë¡¬í”„íŠ¸ ê´€ë¦¬ API
# ========================================
@app.get("/equipment_types")
async def get_equipment_types():
    """ì„¤ë¹„ ìœ í˜• ì •ë³´ ë°˜í™˜"""
    return {"types": EQUIPMENT_TYPES}


@app.get("/equip_prompts")
async def get_equip_prompts():
    """ëª¨ë“  ì„¤ë¹„ë³„ í”„ë¡¬í”„íŠ¸ ì¡°íšŒ"""
    try:
        result = {}
        
        # BASE
        base_path = os.path.join(EQUIP_PROMPT_DIR, "BASE", "common.txt")
        if os.path.exists(base_path):
            with open(base_path, "r", encoding="utf-8") as f:
                result["BASE"] = {"common": f.read()}
        else:
            result["BASE"] = {"common": ""}
        
        # ê° ì„¤ë¹„ë³„
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
    """ì„¤ë¹„ë³„ í”„ë¡¬í”„íŠ¸ ì €ìž¥"""
    try:
        equip_type = data.get("equipment_type", "")
        prompt_type = data.get("prompt_type", "")
        content = data.get("content", "")
        
        if not equip_type or not prompt_type:
            return {"success": False, "message": "ì„¤ë¹„ íƒ€ìž…ê³¼ í”„ë¡¬í”„íŠ¸ íƒ€ìž… í•„ìš”"}
        
        dir_path = os.path.join(EQUIP_PROMPT_DIR, equip_type)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        filepath = os.path.join(dir_path, f"{prompt_type}.txt")
        
        # ë°±ì—…
        if os.path.exists(filepath):
            backup_path = filepath + ".backup"
            with open(filepath, "r", encoding="utf-8") as f:
                with open(backup_path, "w", encoding="utf-8") as bf:
                    bf.write(f.read())
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        
        logger.info(f"Equipment prompt saved: {filepath}")
        return {"success": True, "message": "ì €ìž¥ ì™„ë£Œ", "filepath": filepath}
    except Exception as e:
        logger.error(f"Save prompt failed: {e}")
        return {"success": False, "message": str(e)}


# ========================================
# AMHS ë¶„ì„ API (ë‹¨ì¼ íŒŒì¼)
# ========================================
@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...), question: str = Form("")):
    """Upload and analyze single CSV file"""
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


# ========================================
# AMHS ë¶„ì„ API (ë°°ì¹˜ - ë‹¤ì¤‘ íŒŒì¼)
# ========================================
@app.post("/upload_csv_batch")
async def upload_csv_batch(files: List[UploadFile] = File(...), question: str = Form("")):
    """Upload and analyze multiple CSV files in batch"""
    results = []
    
    logger.info(f"=== Batch Analysis Started: {len(files)} files ===")
    
    for idx, file in enumerate(files):
        logger.info(f"Processing {idx+1}/{len(files)}: {file.filename}")
        
        try:
            content = await file.read()
            
            # ì¸ì½”ë”© ì²˜ë¦¬
            csv_text = None
            for encoding in ['utf-8', 'cp949', 'euc-kr']:
                try:
                    csv_text = content.decode(encoding)
                    break
                except:
                    continue
            
            if csv_text is None:
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": "CSV íŒŒì¼ ì¸ì½”ë”© ì‹¤íŒ¨ (UTF-8, CP949, EUC-KR ëª¨ë‘ ì‹¤íŒ¨)",
                    "order": idx + 1
                })
                continue
            
            df = parse_csv_data(csv_text)
            result = analyze_amhs_log(df, question, file.filename)
            result["filename"] = file.filename
            result["order"] = idx + 1
            results.append(result)
            
            logger.info(f"  âœ… {file.filename}: {result.get('equipment_type', 'UNKNOWN')}, "
                       f"{result.get('basic_info', {}).get('row_count', 0)} rows")
            
        except Exception as e:
            logger.error(f"  âŒ {file.filename} failed: {e}")
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e),
                "order": idx + 1
            })
    
    # ìš”ì•½ ìƒì„±
    summary = generate_batch_summary(results)
    
    logger.info(f"=== Batch Analysis Complete ===")
    logger.info(f"  Success: {summary['success_count']}/{summary['total_files']}")
    logger.info(f"  Delays: {summary['delay_file_count']} files")
    logger.info(f"  Equipment: {summary['equipment_distribution']}")
    
    return {
        "success": True,
        "total_files": len(files),
        "results": results,
        "summary": summary
    }


# ========================================
# POI íŒŒì¼ ê´€ë¦¬ API
# ========================================
@app.get("/poi_files")
async def get_poi_files():
    """Get list of CSV files in POI folder"""
    poi_folder = "POI"
    if not os.path.exists(poi_folder):
        return {"files": []}
    files = [f for f in os.listdir(poi_folder) if f.endswith('.csv')]
    return {"files": sorted(files)}


@app.post("/analyze_poi_file")
async def analyze_poi_file(data: dict):
    """Analyze a single POI CSV file"""
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


@app.post("/analyze_poi_batch")
async def analyze_poi_batch(data: dict):
    """Analyze multiple POI CSV files in batch"""
    filenames = data.get("filenames", [])
    question = data.get("question", "")
    
    if not filenames:
        return {"success": False, "error": "íŒŒì¼ ëª©ë¡ì´ ë¹„ì–´ìžˆìŠµë‹ˆë‹¤"}
    
    results = []
    
    logger.info(f"=== POI Batch Analysis Started: {len(filenames)} files ===")
    
    for idx, filename in enumerate(filenames):
        filepath = os.path.join("POI", filename)
        
        if not os.path.exists(filepath):
            results.append({
                "filename": filename,
                "success": False,
                "error": f"íŒŒì¼ì„ ì°¾ì„ ìˆ˜ ì—†ìŠµë‹ˆë‹¤: {filename}",
                "order": idx + 1
            })
            continue
        
        try:
            df = None
            for encoding in ['utf-8', 'cp949', 'euc-kr']:
                try:
                    df = pd.read_csv(filepath, encoding=encoding)
                    break
                except:
                    continue
            
            if df is None:
                results.append({
                    "filename": filename,
                    "success": False,
                    "error": "CSV íŒŒì¼ ì½ê¸° ì‹¤íŒ¨",
                    "order": idx + 1
                })
                continue
            
            result = analyze_amhs_log(df, question, filename)
            result["filename"] = filename
            result["order"] = idx + 1
            results.append(result)
            
            logger.info(f"  âœ… {filename}: {result.get('equipment_type', 'UNKNOWN')}")
            
        except Exception as e:
            logger.error(f"  âŒ {filename} failed: {e}")
            results.append({
                "filename": filename,
                "success": False,
                "error": str(e),
                "order": idx + 1
            })
    
    summary = generate_batch_summary(results)
    
    logger.info(f"=== POI Batch Analysis Complete ===")
    
    return {
        "success": True,
        "total_files": len(filenames),
        "results": results,
        "summary": summary
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)