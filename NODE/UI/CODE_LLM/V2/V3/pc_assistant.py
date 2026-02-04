#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PC 개인비서 AI (Moltbot 스타일 Tool Calling)
- LLM이 JSON으로 도구 호출
- 파일 검색, 내용 검색
- 시스템 정보
- 프로그램 실행
- API 및 GGUF 모두 지원
- ★ 스크린샷 웹 표시 지원
"""

import os
import re
import json
import subprocess
import platform
import psutil
import datetime
import webbrowser
import fnmatch
import requests
import pandas as pd
from typing import Optional, List
from fastapi import FastAPI, APIRouter
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PCAssistant")

# APIRouter로 변경 (메인 서버에서 include 가능)
router = APIRouter(prefix="/assistant", tags=["assistant"])

# 단독 실행용 앱 (테스트용)
app = FastAPI(title="짝퉁 몰트봇 감마버전 VER 0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================================
# Global Configuration
# ========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GGUF_MODEL_PATH = os.path.join(BASE_DIR, "Qwen3-14B-Q4_K_M.gguf")
LOCAL_LLM = None
CHAT_HISTORY = []
HISTORY_FILE = os.path.join(BASE_DIR, "chat_history.json")

# ★ 스크린샷 저장 폴더
SCREENSHOT_DIR = os.path.join(BASE_DIR, "screenshots")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# LLM 모드 설정 (api 또는 local)
LLM_MODE = "local"  # 기본값: local
API_TOKEN = None

# API 설정
ENV_CONFIG = {
    "dev": {
        "url": "http://dev.assistant.llm.skhynix.com/v1/chat/completions",
        "model": "Qwen3-Coder-30B-A3B-Instruct",
        "name": "DEV(30B)"
    },
    "prod": {
        "url": "http://summary.llm.skhynix.com/v1/chat/completions",
        "model": "Qwen3-Next-80B-A3B-Instruct",
        "name": "PROD(80B)"
    },
    "common": {
        "url": "http://common.llm.skhynix.com/v1/chat/completions",
        "model": "gpt-oss-20b",
        "name": "COMMON(20B)"
    }
}
CURRENT_ENV = "common"
API_URL = ENV_CONFIG["common"]["url"]
API_MODEL = ENV_CONFIG["common"]["model"]

# ========================================
# System Prompt (Tool Calling 방식)
# ========================================
SYSTEM_PROMPT = """당신은 '짝퉁 몰트봇 감마버전 VER 0.1'이라는 PC 개인비서 AI입니다.

[중요 규칙]
1. PC 작업(파일검색, 시스템정보 등)이 필요하면 반드시 아래 JSON 형식으로 도구를 호출하세요.
2. JSON만 출력하고, 다른 설명은 절대 붙이지 마세요.
3. keyword에는 확장자(.gguf)나 와일드카드(*) 없이 순수 키워드만 넣으세요. 예: "gguf", "txt", "python"

[도구 목록]
- 파일검색: {"tool": "search_files", "keyword": "gguf", "path": "F:/"}
- 내용검색: {"tool": "search_content", "keyword": "hello", "path": "C:/"}
- 시스템정보: {"tool": "get_system_info"}
- 폴더보기: {"tool": "list_directory", "path": "C:/Users"}
- 파일읽기: {"tool": "read_file", "path": "C:/test.txt"}
- 프로그램실행: {"tool": "run_program", "program": "notepad"}
- 프로그램종료: {"tool": "kill_program", "name": "notepad"}
- 웹열기: {"tool": "open_web", "url": "https://google.com"}
- 구글검색: {"tool": "google_search", "query": "날씨"}
- 현재시간: {"tool": "get_time"}
- 스크린샷: {"tool": "screenshot"}
- 데이터분석: {"tool": "analyze_data", "path": "C:/data.csv"}

일반 대화는 한국어로 자연스럽게 답변하세요."""


# ========================================
# Tool Functions
# ========================================
def load_local_model():
    global LOCAL_LLM
    try:
        from llama_cpp import Llama

        if not os.path.exists(GGUF_MODEL_PATH):
            logger.error(f"GGUF 파일 없음: {GGUF_MODEL_PATH}")
            return None

        logger.info("GGUF 모델 로딩 중...")
        llm = Llama(
            model_path=GGUF_MODEL_PATH,
            n_ctx=8192,
            n_threads=8,
            n_gpu_layers=50,
            n_batch=512,
            verbose=False
        )
        logger.info("GGUF 모델 로드 완료!")
        return llm
    except Exception as e:
        logger.error(f"모델 로드 실패: {e}")
        return None


def load_api_token():
    """API 토큰 로드"""
    global API_TOKEN
    token_file = os.path.join(BASE_DIR, "api_token.txt")
    if os.path.exists(token_file):
        with open(token_file, 'r') as f:
            API_TOKEN = f.read().strip()
            logger.info("API 토큰 로드됨")
            return True

    # 상위 디렉토리도 확인
    paths = ["token.txt", "../token.txt", os.path.expanduser("~/token.txt")]
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding='utf-8') as f:
                    API_TOKEN = f.read().strip()
                if API_TOKEN and "REPLACE" not in API_TOKEN:
                    logger.info(f"API 토큰 로드: {p}")
                    return True
            except:
                pass

    logger.warning("API 토큰 파일 없음")
    return False


def call_local_llm(prompt: str, system_prompt: str = "") -> dict:
    """로컬 GGUF 모델 호출"""
    global LOCAL_LLM

    if LOCAL_LLM is None:
        return {"success": False, "error": "로컬 모델이 로드되지 않았습니다"}

    full_prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""

    try:
        output = LOCAL_LLM(
            full_prompt,
            max_tokens=4096,
            temperature=0.3,
            stop=["<|im_end|>", "<|im_start|>"],
            echo=False
        )
        content = output["choices"][0]["text"].strip()
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        return {"success": True, "content": content}
    except Exception as e:
        return {"success": False, "error": str(e)}


def call_api_llm(prompt: str, system_prompt: str = "") -> dict:
    """API LLM 호출"""
    global API_TOKEN

    if not API_TOKEN:
        return {"success": False, "error": "API 토큰 없음"}

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
        "max_tokens": 4096,
        "temperature": 0.3
    }

    try:
        response = requests.post(API_URL, headers=headers, json=data, timeout=300)

        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            return {"success": True, "content": content}
        else:
            return {"success": False, "error": f"API 오류: {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def call_llm(prompt: str, system_prompt: str = "") -> dict:
    """LLM_MODE에 따라 API 또는 로컬 모델 호출"""
    if LLM_MODE == "local":
        return call_local_llm(prompt, system_prompt)
    else:
        return call_api_llm(prompt, system_prompt)


def search_files(keyword: str, path: str = "C:/", limit: int = 50) -> List[dict]:
    """파일 이름으로 검색"""
    results = []
    logger.info(f"파일 검색: '{keyword}' in '{path}'")

    try:
        for root, dirs, files in os.walk(path):
            for name in files + dirs:
                if keyword.lower() in name.lower():
                    full_path = os.path.join(root, name)
                    is_dir = os.path.isdir(full_path)
                    try:
                        size = os.path.getsize(full_path) if not is_dir else 0
                        size_str = f"{size / (1024**3):.2f}GB" if size > 1024**3 else f"{size / (1024**2):.1f}MB" if size > 1024**2 else f"{size}B"
                    except:
                        size_str = "?"

                    results.append({
                        "name": name,
                        "path": full_path,
                        "type": "폴더" if is_dir else "파일",
                        "size": size_str
                    })

                    if len(results) >= limit:
                        return results
    except Exception as e:
        logger.error(f"검색 오류: {e}")

    return results


def search_content(keyword: str, path: str = "C:/", limit: int = 30) -> List[dict]:
    """파일 내용으로 검색"""
    results = []
    extensions = ['.txt', '.py', '.md', '.json', '.html', '.css', '.js', '.csv', '.log']

    logger.info(f"내용 검색: '{keyword}' in '{path}'")

    try:
        for root, dirs, files in os.walk(path):
            for name in files:
                ext = os.path.splitext(name)[1].lower()
                if ext in extensions:
                    full_path = os.path.join(root, name)
                    try:
                        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read(50000)  # 50KB까지만
                            if keyword.lower() in content.lower():
                                idx = content.lower().find(keyword.lower())
                                snippet = content[max(0, idx-30):min(len(content), idx+70)].replace('\n', ' ')

                                results.append({
                                    "name": name,
                                    "path": full_path,
                                    "snippet": f"...{snippet}..."
                                })

                                if len(results) >= limit:
                                    return results
                    except:
                        continue
    except Exception as e:
        logger.error(f"내용 검색 오류: {e}")

    return results


def get_system_info() -> dict:
    """시스템 정보"""
    drives = []
    for p in psutil.disk_partitions():
        try:
            usage = psutil.disk_usage(p.mountpoint)
            drives.append({
                "drive": p.device,
                "total": f"{usage.total / (1024**3):.1f}GB",
                "used": f"{usage.percent}%"
            })
        except:
            pass

    return {
        "os": f"{platform.system()} {platform.release()}",
        "cpu": f"{psutil.cpu_count()}코어, {psutil.cpu_percent()}%",
        "memory": f"{psutil.virtual_memory().total // (1024**3)}GB, {psutil.virtual_memory().percent}%",
        "drives": drives
    }


def list_directory(path: str) -> List[dict]:
    """폴더 내용"""
    items = []
    try:
        for name in os.listdir(path)[:50]:
            full_path = os.path.join(path, name)
            is_dir = os.path.isdir(full_path)
            try:
                size = os.path.getsize(full_path) if not is_dir else 0
                modified = datetime.datetime.fromtimestamp(os.path.getmtime(full_path)).strftime("%Y-%m-%d %H:%M")
            except:
                size = 0
                modified = "?"

            items.append({
                "name": name,
                "type": "폴더" if is_dir else "파일",
                "size": f"{size:,}" if not is_dir else "-",
                "modified": modified
            })
    except Exception as e:
        return [{"error": str(e)}]

    return items


def read_file(path: str, max_chars: int = 5000) -> str:
    """파일 읽기"""
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(max_chars)
            if len(content) == max_chars:
                content += "\n... (파일이 너무 커서 일부만 표시)"
            return content
    except Exception as e:
        return f"파일 읽기 오류: {e}"


def run_program(program: str) -> str:
    """프로그램 실행"""
    try:
        subprocess.Popen(program, shell=True)
        return f"'{program}' 실행됨"
    except Exception as e:
        return f"실행 오류: {e}"


def kill_program(name: str) -> str:
    """프로그램 종료"""
    try:
        killed = 0
        for proc in psutil.process_iter(['name']):
            if name.lower() in proc.info['name'].lower():
                proc.kill()
                killed += 1
        return f"{killed}개 프로세스 종료됨"
    except Exception as e:
        return f"종료 오류: {e}"


def open_web(url: str) -> str:
    """웹 열기"""
    if not url.startswith('http'):
        url = 'https://' + url
    webbrowser.open(url)
    return f"'{url}' 열림"


def google_search(query: str) -> str:
    """구글 검색"""
    url = f"https://www.google.com/search?q={query}"
    webbrowser.open(url)
    return f"'{query}' 검색 중..."


def get_time() -> str:
    """현재 시간"""
    now = datetime.datetime.now()
    return f"{now.strftime('%Y년 %m월 %d일 %A %H시 %M분 %S초')}"


def take_screenshot() -> dict:
    """★ 스크린샷 찍고 웹 URL 반환"""
    try:
        from PIL import ImageGrab

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"screenshot_{timestamp}.png"
        filepath = os.path.join(SCREENSHOT_DIR, filename)

        img = ImageGrab.grab()
        img.save(filepath)

        # 웹에서 접근할 URL 반환
        web_url = f"/assistant/api/screenshot/{filename}"
        logger.info(f"스크린샷 저장: {filepath} → URL: {web_url}")

        return {
            "success": True,
            "path": filepath,
            "filename": filename,
            "url": web_url,
            "timestamp": timestamp
        }
    except ImportError:
        return {"success": False, "error": "PIL(Pillow) 라이브러리가 설치되지 않았습니다. pip install Pillow"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def analyze_data(path: str) -> str:
    """데이터 분석"""
    try:
        ext = os.path.splitext(path)[1].lower()

        if ext == '.csv':
            df = pd.read_csv(path, encoding='utf-8', errors='ignore')
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(path)
        else:
            return f"지원하지 않는 형식: {ext}"

        result = []
        result.append(f"파일: {os.path.basename(path)}")
        result.append(f"크기: {len(df):,}행 x {len(df.columns)}열")
        result.append(f"컬럼: {', '.join(df.columns.tolist()[:20])}")

        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            stats = df[numeric_cols].describe().to_string()
            result.append(f"통계:\n{stats}")

        result.append(f"샘플:\n{df.head(5).to_string()}")

        return "\n".join(result)
    except Exception as e:
        return f"분석 오류: {e}"


# Tool 실행기
def execute_tool(tool_data: dict) -> str:
    """도구 실행"""
    tool_name = tool_data.get("tool")

    if tool_name == "search_files":
        results = search_files(tool_data.get("keyword", ""), tool_data.get("path", "C:/"))
        return json.dumps(results[:20], ensure_ascii=False, indent=2)

    elif tool_name == "search_content":
        results = search_content(tool_data.get("keyword", ""), tool_data.get("path", "C:/"))
        return json.dumps(results[:10], ensure_ascii=False, indent=2)

    elif tool_name == "get_system_info":
        return json.dumps(get_system_info(), ensure_ascii=False, indent=2)

    elif tool_name == "list_directory":
        results = list_directory(tool_data.get("path", "C:/"))
        return json.dumps(results, ensure_ascii=False, indent=2)

    elif tool_name == "read_file":
        return read_file(tool_data.get("path", ""))

    elif tool_name == "run_program":
        return run_program(tool_data.get("program", ""))

    elif tool_name == "kill_program":
        return kill_program(tool_data.get("name", ""))

    elif tool_name == "open_web":
        return open_web(tool_data.get("url", ""))

    elif tool_name == "google_search":
        return google_search(tool_data.get("query", ""))

    elif tool_name == "get_time":
        return get_time()

    elif tool_name == "screenshot":
        # ★ 스크린샷은 특별 처리 - dict 반환
        result = take_screenshot()
        return json.dumps(result, ensure_ascii=False)

    elif tool_name == "analyze_data":
        return analyze_data(tool_data.get("path", ""))

    return "알 수 없는 도구"


# ========================================
# Chat Processing
# ========================================
def process_chat(user_message: str) -> dict:
    """★ 채팅 처리 - dict 반환으로 변경 (이미지 URL 포함 가능)"""
    global LOCAL_LLM, LLM_MODE

    # 모델 체크
    if LLM_MODE == "local" and LOCAL_LLM is None:
        return {"text": "로컬 모델이 로드되지 않았습니다.", "image": None}
    if LLM_MODE != "local" and not API_TOKEN:
        return {"text": "API 토큰이 없습니다.", "image": None}

    # 1차: LLM 호출 (도구 호출 여부 판단)
    try:
        result = call_llm(user_message, SYSTEM_PROMPT)
        if not result["success"]:
            return {"text": f"LLM 오류: {result.get('error', '알 수 없는 오류')}", "image": None}

        text = result["content"]

        # JSON 도구 호출 감지
        tool_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if not tool_match:
            tool_match = re.search(r'(\{[^{}]*"tool"\s*:\s*"[^"]+?"[^{}]*\})', text, re.DOTALL)

        if tool_match:
            try:
                raw_json = tool_match.group(1)
                logger.info(f"도구 호출 감지: {raw_json}")
                tool_data = json.loads(raw_json)

                # keyword에서 와일드카드 제거
                if "keyword" in tool_data:
                    tool_data["keyword"] = tool_data["keyword"].replace("*", "").replace(".", "").strip()

                logger.info(f"도구 실행: {tool_data}")
                tool_result = execute_tool(tool_data)
                logger.info(f"도구 결과: {tool_result[:200]}...")

                # ★ 스크린샷인 경우 특별 처리
                if tool_data.get("tool") == "screenshot":
                    try:
                        sc_data = json.loads(tool_result)
                        if sc_data.get("success"):
                            return {
                                "text": f"📸 스크린샷을 찍었습니다!\n\n촬영 시간: {sc_data['timestamp']}\n저장 위치: {sc_data['path']}",
                                "image": sc_data["url"]
                            }
                        else:
                            return {
                                "text": f"❌ 스크린샷 실패: {sc_data.get('error', '알 수 없는 오류')}",
                                "image": None
                            }
                    except:
                        pass

                # 2차: 결과 해석 (스크린샷 외 일반 도구)
                follow_up_system = f"""{SYSTEM_PROMPT}

[도구 실행 결과]
{tool_result}

위 결과를 사용자가 이해하기 쉽게 한국어로 자연스럽게 설명하세요.
JSON이나 원본 데이터를 그대로 보여주지 말고, 핵심 내용만 정리해서 답변하세요."""

                result2 = call_llm(user_message, follow_up_system)
                if result2["success"]:
                    return {"text": result2["content"], "image": None}
                else:
                    return {"text": f"결과 해석 오류: {result2.get('error', '')}", "image": None}

            except json.JSONDecodeError as e:
                logger.error(f"JSON 파싱 오류: {e}")
                return {"text": "명령 처리 중 오류가 발생했습니다.", "image": None}

        # 도구 호출 없으면 그냥 응답
        return {"text": text, "image": None}

    except Exception as e:
        logger.error(f"처리 오류: {e}")
        return {"text": f"오류: {e}", "image": None}


# ========================================
# 대화 기록
# ========================================
def save_history():
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(CHAT_HISTORY[-100:], f, ensure_ascii=False, indent=2)
    except:
        pass

def load_history():
    global CHAT_HISTORY
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                CHAT_HISTORY = json.load(f)
    except:
        CHAT_HISTORY = []


# ========================================
# API Models
# ========================================
class ChatRequest(BaseModel):
    message: str

class SearchRequest(BaseModel):
    keyword: str
    path: str = "C:/"
    file_content: bool = False

class EnvRequest(BaseModel):
    env: str  # "local", "dev", "prod", "common"


# ========================================
# API Endpoints
# ========================================
# 초기화 함수 (메인 서버에서 호출)
def init_assistant():
    global LOCAL_LLM, LLM_MODE
    load_history()
    if load_api_token():
        LLM_MODE = "api"
        logger.info("비서: API 모드")
    else:
        LOCAL_LLM = load_local_model()
        if LOCAL_LLM:
            LLM_MODE = "local"
            logger.info("비서: LOCAL 모드")


# Router 엔드포인트들 (메인 서버에 통합됨)
@router.get("/")
async def assistant_home():
    return FileResponse(os.path.join(BASE_DIR, "assistant_ui.html"))


@router.get("/api/status")
async def assistant_status():
    return {
        "mode": LLM_MODE,
        "env": CURRENT_ENV if LLM_MODE != "local" else "local",
        "model_loaded": LOCAL_LLM is not None if LLM_MODE == "local" else API_TOKEN is not None,
        "model_name": ENV_CONFIG.get(CURRENT_ENV, {}).get("name", "LOCAL") if LLM_MODE != "local" else "Qwen3-14B-GGUF",
        "system": get_system_info(),
        "history_count": len(CHAT_HISTORY)
    }


@router.post("/api/set_env")
async def assistant_set_env(request: EnvRequest):
    global LLM_MODE, LOCAL_LLM, CURRENT_ENV, API_URL, API_MODEL

    env = request.env.lower()

    if env == "local":
        if LOCAL_LLM is None:
            LOCAL_LLM = load_local_model()
        if LOCAL_LLM:
            LLM_MODE = "local"
            return {"success": True, "env": "local", "name": "LOCAL(14B-GGUF)"}
        else:
            return {"success": False, "error": "로컬 모델 로드 실패"}

    elif env in ENV_CONFIG:
        if not API_TOKEN:
            if not load_api_token():
                return {"success": False, "error": "API 토큰 없음"}

        LLM_MODE = "api"
        CURRENT_ENV = env
        API_URL = ENV_CONFIG[env]["url"]
        API_MODEL = ENV_CONFIG[env]["model"]
        return {"success": True, "env": env, "name": ENV_CONFIG[env]["name"]}

    return {"success": False, "error": f"알 수 없는 환경: {env}"}


@router.post("/api/chat")
async def assistant_chat(request: ChatRequest):
    """★ 이미지 URL도 함께 반환"""
    user_msg = request.message.strip()

    CHAT_HISTORY.append({"role": "user", "content": user_msg, "time": datetime.datetime.now().isoformat()})

    response = process_chat(user_msg)

    # response는 이제 dict: {"text": ..., "image": ...}
    response_text = response["text"]
    response_image = response.get("image")

    CHAT_HISTORY.append({
        "role": "assistant",
        "content": response_text,
        "image": response_image,
        "time": datetime.datetime.now().isoformat()
    })
    save_history()

    return {
        "success": True,
        "response": response_text,
        "image": response_image  # ★ 이미지 URL (없으면 None)
    }


# ★ 스크린샷 이미지 서빙 엔드포인트
@router.get("/api/screenshot/{filename}")
async def serve_screenshot(filename: str):
    """스크린샷 이미지 파일 서빙"""
    # 보안: 파일명에 경로 조작 방지
    safe_filename = os.path.basename(filename)
    filepath = os.path.join(SCREENSHOT_DIR, safe_filename)

    if not os.path.exists(filepath):
        return {"error": "스크린샷 파일을 찾을 수 없습니다"}

    return FileResponse(filepath, media_type="image/png")


# ★ 최근 스크린샷 목록
@router.get("/api/screenshots")
async def list_screenshots():
    """최근 스크린샷 목록"""
    screenshots = []
    if os.path.exists(SCREENSHOT_DIR):
        for f in sorted(os.listdir(SCREENSHOT_DIR), reverse=True)[:20]:
            if f.endswith('.png'):
                filepath = os.path.join(SCREENSHOT_DIR, f)
                screenshots.append({
                    "filename": f,
                    "url": f"/assistant/api/screenshot/{f}",
                    "size": f"{os.path.getsize(filepath) / 1024:.1f}KB",
                    "time": datetime.datetime.fromtimestamp(
                        os.path.getmtime(filepath)
                    ).strftime("%Y-%m-%d %H:%M:%S")
                })
    return {"screenshots": screenshots}


@router.post("/api/search")
async def assistant_search(request: SearchRequest):
    if request.file_content:
        results = search_content(request.keyword, request.path)
    else:
        results = search_files(request.keyword, request.path)

    return {"success": True, "results": results, "count": len(results)}


@router.get("/api/drives")
async def assistant_drives():
    drives = []
    for p in psutil.disk_partitions():
        drives.append({"device": p.device, "mountpoint": p.mountpoint})
    return {"success": True, "drives": drives}


@router.get("/api/history")
async def assistant_get_history():
    return {"history": CHAT_HISTORY[-50:]}


@router.delete("/api/history")
async def assistant_clear_history():
    global CHAT_HISTORY
    CHAT_HISTORY = []
    save_history()
    return {"success": True}


# 단독 실행용 (테스트)
if __name__ == "__main__":
    import uvicorn
    app.include_router(router)

    @app.on_event("startup")
    async def standalone_startup():
        init_assistant()

    uvicorn.run(app, host="0.0.0.0", port=8002)