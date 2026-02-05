#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pc_assistant.py
PC 개인비서 AI (Moltbot 스타일 Tool Calling) v0.2
- 스크린샷: 전용 폴더 저장 + 웹 인라인 표시
- 파일 탐색기/메모장 실행 제거
"""

import os
import re
import json
import subprocess
import platform
import psutil
import tempfile
import datetime
import webbrowser
import fnmatch
import requests
import pandas as pd
from typing import Optional, List
from fastapi import FastAPI, APIRouter, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PCAssistant")

router = APIRouter(prefix="/assistant", tags=["assistant"])
app = FastAPI(title="짝퉁 몰트봇 감마버전 VER 0.2")

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

# ★ 로컬 GGUF 모델 설정 (생성 파라미터 포함)
AVAILABLE_MODELS = {
    "qwen3-14b": {
        "path": os.path.join(BASE_DIR, "Qwen3-14B-Q4_K_M.gguf"),
        "name": "Qwen3-14B (Q4_K_M)",
        "desc": "한글 최적화 ⭐추천",
        "ctx": 8192,
        "gpu_layers": 50,
        "chat_format": "chatml",
        "repeat_penalty": 1.1,
        "temperature": 0.7,
        "korean_support": True
    },
    "mistral-nemo": {
        "path": os.path.join(BASE_DIR, "Mistral-Nemo-12B.Q6_K.gguf"),
        "name": "Mistral-Nemo-12B (Q6_K)",
        "desc": "한글 보통, 코딩 강점",
        "ctx": 8192,
        "gpu_layers": 50,
        "chat_format": "llama2",
        "repeat_penalty": 1.2,
        "temperature": 0.6,
        "korean_support": True
    },
    "phi3-mini": {
        "path": os.path.join(BASE_DIR, "Phi-3-mini-4k-instruct-Q5_K_M.gguf"),
        "name": "Phi-3-Mini (Q5_K_M)",
        "desc": "⚠️한글 불안정, 영어 전용",
        "ctx": 4096,
        "gpu_layers": 35,
        "chat_format": "phi3",
        "repeat_penalty": 1.15,
        "temperature": 0.7,
        "korean_support": False
    }
}
CURRENT_LOCAL_MODEL = "qwen3-14b"
GGUF_MODEL_PATH = AVAILABLE_MODELS[CURRENT_LOCAL_MODEL]["path"]

LOCAL_LLM = None
CHAT_HISTORY = []
HISTORY_FILE = os.path.join(BASE_DIR, "chat_history.json")

# ★ 토큰 사용량 추적
TOKEN_USAGE = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "call_count": 0
}

# ★ 스크린샷 전용 폴더
SCREENSHOT_DIR = os.path.join(BASE_DIR, "screenshots")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# ★ 지식베이스(MD 문서) 폴더
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

# ★ 과거지식 보관 폴더
KNOWLEDGE_ARCHIVE_DIR = os.path.join(BASE_DIR, "knowledge_archive")
os.makedirs(KNOWLEDGE_ARCHIVE_DIR, exist_ok=True)

# ★ LLM 생성 파라미터 (UI에서 조절 가능)
LLM_PARAMS = {
    "temperature": 0.7,
    "repeat_penalty": 1.1,
    "max_tokens": 4096
}

LLM_MODE = "local"
API_TOKEN = None

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
# System Prompt
# ========================================
SYSTEM_PROMPT = """당신은 '짝퉁 몰트봇 감마버전 VER 0.2'이라는 PC 개인비서 AI입니다.

★★★ 최우선 규칙: 지식베이스 우선 검색 ★★★
모든 질문에 대해 반드시 이 순서를 따르세요:

1단계: 지식베이스 검색 (필수)
- 사용자 질문에서 핵심 키워드를 추출하세요
- 반드시 search_knowledge로 먼저 검색하세요
- 예: {"tool": "search_knowledge", "keyword": "추출한키워드"}

2단계: 검색 결과 확인
- 관련 문서가 있으면 → 그 내용을 기반으로 답변
- 관련 문서가 없으면 → 3단계로

3단계: 일반 대화로 답변
- 지식베이스에 없는 내용은 "지식베이스에 관련 문서가 없습니다"라고 먼저 말하고
- 일반적인 지식으로 답변하거나, 모르면 "잘 모르겠습니다"라고 솔직히 답변

★ 절대 하지 말 것:
- 웹검색(google_search) 임의로 사용 금지 (사용자가 "검색해줘"라고 할 때만)
- 파일검색(search_files) 임의로 사용 금지 (사용자가 "파일 찾아줘"라고 할 때만)
- 시스템정보(get_system_info) 임의로 사용 금지 (사용자가 "시스템 정보"라고 할 때만)
- 지식베이스 검색 없이 바로 다른 도구 사용 금지

[도구 호출 형식]
JSON만 출력하세요. 다른 텍스트 붙이지 마세요.

[지식베이스 도구] - 항상 먼저 사용
- 지식검색: {"tool": "search_knowledge", "keyword": "키워드"}
- 지식목록: {"tool": "list_knowledge"}
- 지식읽기: {"tool": "read_knowledge", "filename": "파일명.md"}

[PC 도구] - 사용자가 명시적으로 요청할 때만
- 시스템정보: {"tool": "get_system_info"} ← "시스템 정보 알려줘"
- 스크린샷: {"tool": "screenshot"} ← "스크린샷 찍어줘"
- 현재시간: {"tool": "get_time"} ← "지금 몇시야"
- 프로그램실행: {"tool": "run_program", "program": "notepad"} ← "메모장 실행해줘"
- 파일검색: {"tool": "search_files", "keyword": "문서", "path": "C:/"} ← "파일 찾아줘"
- 웹열기/검색: {"tool": "google_search", "query": "검색어"} ← "검색해줘"
- 최신뉴스: {"tool": "latest_news"} ← "뉴스 보여줘"

일반 대화는 한국어로 자연스럽게 답변하세요."""


# ========================================
# LLM Functions
# ========================================
def load_local_model(model_key: str = None):
    """로컬 GGUF 모델 로드 (model_key로 모델 선택)"""
    global LOCAL_LLM, CURRENT_LOCAL_MODEL, GGUF_MODEL_PATH

    if model_key is None:
        model_key = CURRENT_LOCAL_MODEL

    if model_key not in AVAILABLE_MODELS:
        logger.error(f"알 수 없는 모델: {model_key}")
        return None

    model_config = AVAILABLE_MODELS[model_key]
    model_path = model_config["path"]

    if not os.path.exists(model_path):
        logger.error(f"GGUF 파일 없음: {model_path}")
        return None

    try:
        from llama_cpp import Llama
        logger.info(f"🔄 모델 로딩 중: {model_config['name']}...")
        llm = Llama(
            model_path=model_path,
            n_ctx=model_config.get("ctx", 8192),
            n_threads=8,
            n_gpu_layers=model_config.get("gpu_layers", 50),
            n_batch=512,
            verbose=False
        )
        CURRENT_LOCAL_MODEL = model_key
        GGUF_MODEL_PATH = model_path
        logger.info(f"✅ 모델 로드 완료: {model_config['name']}")
        return llm
    except Exception as e:
        logger.error(f"모델 로드 실패: {e}")
        return None


def get_available_local_models() -> List[dict]:
    """사용 가능한 로컬 모델 목록 반환"""
    models = []
    for key, config in AVAILABLE_MODELS.items():
        exists = os.path.exists(config["path"])
        models.append({
            "key": key,
            "name": config["name"],
            "desc": config.get("desc", ""),
            "available": exists,
            "current": key == CURRENT_LOCAL_MODEL,
            "korean_support": config.get("korean_support", True)
        })
    return models


def load_api_token():
    global API_TOKEN
    paths = [
        os.path.join(BASE_DIR, "token.txt"),
        os.path.join(BASE_DIR, "api_token.txt"),
        "token.txt",
        "../token.txt",
        os.path.expanduser("~/token.txt")
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    API_TOKEN = f.read().strip()
                if API_TOKEN and "REPLACE" not in API_TOKEN:
                    logger.info(f"✅ API 토큰 로드: {p}")
                    return True
            except Exception as e:
                logger.error(f"❌ 토큰 로드 실패: {e}")
    logger.warning("⚠️ API 토큰 파일 없음")
    return False


def call_local_llm(prompt: str, system_prompt: str = "", max_tokens: int = 4096) -> dict:
    global LOCAL_LLM, CURRENT_LOCAL_MODEL, LLM_PARAMS
    if LOCAL_LLM is None:
        return {"success": False, "error": "로컬 모델이 로드되지 않았습니다"}

    # ★ UI에서 설정한 파라미터 사용 (LLM_PARAMS 우선)
    temperature = LLM_PARAMS.get("temperature", 0.7)
    repeat_penalty = LLM_PARAMS.get("repeat_penalty", 1.1)
    actual_max_tokens = LLM_PARAMS.get("max_tokens", max_tokens)

    # ★ 모델별 프롬프트 형식
    if CURRENT_LOCAL_MODEL == "mistral-nemo":
        # Mistral-Nemo: ChatML 형식 사용 (공식 권장)
        full_prompt = f"""<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{prompt} [/INST]"""
        stop_tokens = ["</s>", "[INST]"]
    elif CURRENT_LOCAL_MODEL == "phi3-mini":
        # Phi-3: 영어 위주 모델 (한글 출력 불안정)
        # 시스템 프롬프트 영어로 변환하여 안정성 확보
        full_prompt = f"""<|system|>
You are a helpful PC assistant. Respond in Korean. Follow JSON tool format strictly.
{system_prompt}<|end|>
<|user|>
{prompt}<|end|>
<|assistant|>"""
        stop_tokens = ["<|end|>", "<|user|>", "<|system|>"]
    else:
        # Qwen3 (기본): ChatML 형식 - 한글 최적화
        full_prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""
        stop_tokens = ["<|im_end|>", "<|im_start|>"]

    try:
        output = LOCAL_LLM(
            full_prompt,
            max_tokens=actual_max_tokens,
            temperature=temperature,
            repeat_penalty=repeat_penalty,  # ★ 반복 억제
            stop=stop_tokens,
            echo=False
        )
        content = output["choices"][0]["text"].strip()
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        return {"success": True, "content": content}
    except Exception as e:
        return {"success": False, "error": str(e)}


def call_api_llm(prompt: str, system_prompt: str = "", max_tokens: int = 4096) -> dict:
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
        "max_tokens": max_tokens,
        "temperature": 0.3
    }
    try:
        response = requests.post(API_URL, headers=headers, json=data, timeout=300)
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            
            # ★ 토큰 사용량 누적
            usage = result.get("usage", {})
            if usage:
                TOKEN_USAGE["prompt_tokens"] += usage.get("prompt_tokens", 0)
                TOKEN_USAGE["completion_tokens"] += usage.get("completion_tokens", 0)
                TOKEN_USAGE["total_tokens"] += usage.get("total_tokens", 0)
                TOKEN_USAGE["call_count"] += 1
                logger.info(f"📊 토큰: +{usage.get('total_tokens', 0)} (누적: {TOKEN_USAGE['total_tokens']})")
            
            return {"success": True, "content": content}
        else:
            return {"success": False, "error": f"API 오류: {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def call_llm(prompt: str, system_prompt: str = "", max_tokens: int = 4096) -> dict:
    if LLM_MODE == "local":
        return call_local_llm(prompt, system_prompt, max_tokens)
    else:
        return call_api_llm(prompt, system_prompt, max_tokens)


# ========================================
# Tool Functions
# ========================================
def search_files(keyword: str, path: str = "C:/", limit: int = 50) -> List[dict]:
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
                        "name": name, "path": full_path,
                        "type": "폴더" if is_dir else "파일", "size": size_str
                    })
                    if len(results) >= limit:
                        return results
    except Exception as e:
        logger.error(f"검색 오류: {e}")
    return results


def search_content(keyword: str, path: str = "C:/", limit: int = 30) -> List[dict]:
    results = []
    extensions = ['.txt', '.py', '.md', '.json', '.html', '.css', '.js', '.csv', '.log']
    try:
        for root, dirs, files in os.walk(path):
            for name in files:
                ext = os.path.splitext(name)[1].lower()
                if ext in extensions:
                    full_path = os.path.join(root, name)
                    try:
                        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read(50000)
                            if keyword.lower() in content.lower():
                                idx = content.lower().find(keyword.lower())
                                snippet = content[max(0, idx-30):min(len(content), idx+70)].replace('\n', ' ')
                                results.append({"name": name, "path": full_path, "snippet": f"...{snippet}..."})
                                if len(results) >= limit:
                                    return results
                    except:
                        continue
    except Exception as e:
        logger.error(f"내용 검색 오류: {e}")
    return results


def get_system_info() -> dict:
    drives = []
    for p in psutil.disk_partitions():
        try:
            usage = psutil.disk_usage(p.mountpoint)
            drives.append({"drive": p.device, "total": f"{usage.total / (1024**3):.1f}GB", "used": f"{usage.percent}%"})
        except:
            pass
    return {
        "os": f"{platform.system()} {platform.release()}",
        "cpu": f"{psutil.cpu_count()}코어, {psutil.cpu_percent()}%",
        "memory": f"{psutil.virtual_memory().total // (1024**3)}GB, {psutil.virtual_memory().percent}%",
        "drives": drives
    }


def list_directory(path: str) -> List[dict]:
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
            items.append({"name": name, "type": "폴더" if is_dir else "파일", "size": f"{size:,}" if not is_dir else "-", "modified": modified})
    except Exception as e:
        return [{"error": str(e)}]
    return items


def read_file(path: str, max_chars: int = 5000) -> str:
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(max_chars)
            if len(content) == max_chars:
                content += "\n... (파일이 너무 커서 일부만 표시)"
            return content
    except Exception as e:
        return f"파일 읽기 오류: {e}"


def run_program(program: str) -> str:
    try:
        subprocess.Popen(program, shell=True)
        return f"'{program}' 실행됨"
    except Exception as e:
        return f"실행 오류: {e}"


def kill_program(name: str) -> str:
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
    if not url.startswith('http'):
        url = 'https://' + url
    webbrowser.open(url)
    return f"'{url}' 열림"


def google_search(query: str) -> str:
    url = f"https://www.google.com/search?q={query}"
    webbrowser.open(url)
    return f"'{query}' 검색 중..."


def get_time() -> str:
    now = datetime.datetime.now()
    return f"{now.strftime('%Y년 %m월 %d일 %A %H시 %M분 %S초')}"


# ★ 스크린샷: 전용 폴더 저장 + URL 반환
def take_screenshot() -> dict:
    """스크린샷 찍고 전용 폴더에 저장, 웹 표시용 URL 반환"""
    try:
        from PIL import ImageGrab
        filename = f"screenshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(SCREENSHOT_DIR, filename)
        img = ImageGrab.grab()
        img.save(filepath)
        logger.info(f"📸 스크린샷 저장: {filepath}")
        # 웹에서 접근 가능한 URL 반환
        return {
            "success": True,
            "filename": filename,
            "path": filepath,
            "url": f"/assistant/screenshots/{filename}"
        }
    except ImportError:
        return {"success": False, "error": "PIL(Pillow) 미설치. pip install Pillow"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ★ 최신뉴스: 독립 브라우저 창 열기 → 스크린샷 → 그 창만 닫기
def latest_news() -> dict:
    """구글뉴스를 독립 브라우저로 열고, 스크린샷 찍고, 그 창만 닫기"""
    import time
    import shutil
    
    news_proc = None
    temp_profile = None
    
    try:
        news_url = "https://news.google.com/home?hl=ko&gl=KR&ceid=KR:ko"
        
        # 임시 프로필 폴더 (독립 Chrome 인스턴스용)
        temp_profile = os.path.join(tempfile.gettempdir(), "chrome_news_temp")
        
        # 1. Chrome 찾기
        chrome_paths = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            os.path.expanduser(r"~\AppData\Local\Google\Chrome\Application\chrome.exe"),
        ]
        
        chrome_exe = None
        for p in chrome_paths:
            if os.path.exists(p):
                chrome_exe = p
                break
        
        if chrome_exe:
            # 독립 Chrome 인스턴스 (기존 Chrome과 별개, 전체화면)
            news_proc = subprocess.Popen([
                chrome_exe,
                f"--user-data-dir={temp_profile}",
                "--no-first-run",
                "--no-default-browser-check",
                "--start-maximized",
                "--disable-extensions",
                "--disable-sync",
                "--disable-translate",
                news_url
            ])
            logger.info(f"📰 구글뉴스 독립 창 열기 (PID: {news_proc.pid})")
        else:
            webbrowser.open(news_url)
            logger.info("📰 구글뉴스 열기 (기본 브라우저)")
        
        # 2. 초기 로딩 대기 (임시 프로필 첫 실행은 느림)
        time.sleep(3)
        
        # 2.5. 강제 전체화면 (임시 프로필은 최대화 무시할 수 있음)
        try:
            import ctypes
            import ctypes.wintypes
            
            # 가장 앞에 있는 Chrome 창 찾아서 최대화
            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()
            if hwnd:
                SW_MAXIMIZE = 3
                user32.ShowWindow(hwnd, SW_MAXIMIZE)
                logger.info(f"🔲 뉴스 창 최대화 완료 (hwnd: {hwnd})")
        except Exception as e:
            logger.warning(f"⚠️ 최대화 실패 (무시): {e}")
        
        # 3. 뉴스 페이지 완전히 로딩될 때까지 충분히 대기
        logger.info("⏳ 뉴스 페이지 로딩 대기 중... (8초)")
        time.sleep(8)
        
        # 3. 스크린샷 찍기
        from PIL import ImageGrab
        filename = f"news_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(SCREENSHOT_DIR, filename)
        img = ImageGrab.grab()
        img.save(filepath)
        logger.info(f"📸 뉴스 스크린샷: {filepath}")
        
        # 4. 독립 Chrome만 종료
        time.sleep(0.5)
        if news_proc and news_proc.poll() is None:
            # 자식 프로세스 포함 전체 종료
            try:
                parent = psutil.Process(news_proc.pid)
                for child in parent.children(recursive=True):
                    child.terminate()
                parent.terminate()
                logger.info(f"🔒 뉴스 창 닫기 완료 (PID: {news_proc.pid})")
            except psutil.NoSuchProcess:
                pass
        
        # 5. 임시 프로필 정리 (백그라운드)
        try:
            if temp_profile and os.path.exists(temp_profile):
                shutil.rmtree(temp_profile, ignore_errors=True)
        except:
            pass
        
        return {
            "success": True,
            "filename": filename,
            "path": filepath,
            "url": f"/assistant/screenshots/{filename}"
        }
    except Exception as e:
        # 에러 시에도 프로세스 정리
        if news_proc and news_proc.poll() is None:
            try:
                news_proc.terminate()
            except:
                pass
        return {"success": False, "error": str(e)}


# ★ 프로세스 목록 조회
def list_processes(sort_by: str = "memory", limit: int = 30) -> List[dict]:
    """실행 중인 프로세스 목록 반환"""
    processes = []
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info', 'status']):
            try:
                info = proc.info
                mem = info.get('memory_info')
                mem_mb = mem.rss / (1024 * 1024) if mem else 0
                processes.append({
                    "pid": info['pid'],
                    "name": info['name'],
                    "cpu": proc.cpu_percent(interval=0),
                    "memory_mb": round(mem_mb, 1),
                    "status": info.get('status', '?')
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # 정렬
        if sort_by == "cpu":
            processes.sort(key=lambda x: x['cpu'], reverse=True)
        else:
            processes.sort(key=lambda x: x['memory_mb'], reverse=True)

        return processes[:limit]
    except Exception as e:
        logger.error(f"프로세스 목록 오류: {e}")
        return [{"error": str(e)}]


# ★ 지식베이스 함수들
def list_knowledge() -> List[dict]:
    """지식베이스 파일 목록"""
    files = []
    try:
        for f in sorted(os.listdir(KNOWLEDGE_DIR)):
            if f.endswith(('.md', '.txt')):
                filepath = os.path.join(KNOWLEDGE_DIR, f)
                size = os.path.getsize(filepath)
                modified = datetime.datetime.fromtimestamp(os.path.getmtime(filepath)).strftime("%Y-%m-%d %H:%M")
                files.append({"filename": f, "size": f"{size:,}B", "modified": modified})
    except Exception as e:
        logger.error(f"지식 목록 오류: {e}")
    return files


def normalize_keyword(keyword: str) -> List[str]:
    """키워드를 여러 변형으로 확장 (언더스코어, 공백, 하이픈 등)"""
    keyword = keyword.strip().lower()
    variants = [keyword]

    # 언더스코어 <-> 공백 <-> 하이픈 변환
    if '_' in keyword:
        variants.append(keyword.replace('_', ' '))
        variants.append(keyword.replace('_', '-'))
        variants.append(keyword.replace('_', ''))
    if ' ' in keyword:
        variants.append(keyword.replace(' ', '_'))
        variants.append(keyword.replace(' ', '-'))
        variants.append(keyword.replace(' ', ''))
    if '-' in keyword:
        variants.append(keyword.replace('-', '_'))
        variants.append(keyword.replace('-', ' '))
        variants.append(keyword.replace('-', ''))

    # 대소문자 변형 추가
    variants.append(keyword.upper())
    variants.append(keyword.title())

    return list(set(variants))


def calculate_relevance_score(filename: str, content: str, keyword: str, variants: List[str]) -> int:
    """문서의 관련성 점수 계산 (높을수록 관련성 높음)"""
    score = 0
    filename_lower = filename.lower()
    content_lower = content.lower()

    for variant in variants:
        v_lower = variant.lower()

        # 파일명에 키워드 포함 (+50점)
        if v_lower in filename_lower:
            score += 50
            # 파일명이 키워드로 시작하면 추가 점수
            if filename_lower.startswith(v_lower):
                score += 30

        # 내용에서 키워드 등장 횟수 (최대 100점)
        count = content_lower.count(v_lower)
        score += min(count * 5, 100)

        # 제목/헤더에 키워드 있으면 추가 점수
        lines = content.split('\n')[:20]  # 상위 20줄만 확인
        for line in lines:
            if line.startswith('#') and v_lower in line.lower():
                score += 40
                break

    return score


def search_knowledge(keyword: str) -> List[dict]:
    """지식베이스에서 키워드로 파일 검색 (관련성 점수 기반 정렬)"""
    results = []
    variants = normalize_keyword(keyword)
    logger.info(f"🔍 지식검색: '{keyword}' → 변형: {variants[:5]}")

    try:
        for f in os.listdir(KNOWLEDGE_DIR):
            if not f.endswith(('.md', '.txt')):
                continue
            filepath = os.path.join(KNOWLEDGE_DIR, f)

            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as fh:
                    content = fh.read()
            except:
                continue

            # 매칭 여부 확인 (모든 변형에 대해)
            matched = False
            snippet = ""
            for variant in variants:
                v_lower = variant.lower()
                if v_lower in f.lower() or v_lower in content.lower():
                    matched = True
                    # 스니펫 추출
                    idx = content.lower().find(v_lower)
                    if idx >= 0:
                        snippet = content[max(0, idx-50):min(len(content), idx+100)].replace('\n', ' ').strip()
                    break

            if matched:
                score = calculate_relevance_score(f, content, keyword, variants)
                results.append({
                    "filename": f,
                    "snippet": f"...{snippet}..." if snippet else "(파일명 매칭)",
                    "score": score
                })

        # 관련성 점수로 정렬 (높은 순)
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        logger.info(f"📊 검색 결과: {len(results)}개 (상위: {[r['filename'] for r in results[:3]]})")

    except Exception as e:
        logger.error(f"지식 검색 오류: {e}")
    return results


def read_knowledge(filename: str) -> str:
    """지식베이스 MD 파일 읽기"""
    filepath = os.path.join(KNOWLEDGE_DIR, filename)
    if not os.path.exists(filepath):
        for f in os.listdir(KNOWLEDGE_DIR):
            if filename.lower() in f.lower():
                filepath = os.path.join(KNOWLEDGE_DIR, f)
                break
        else:
            return f"❌ '{filename}' 파일을 찾을 수 없습니다."

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(30000)
            if len(content) == 30000:
                content += "\n\n... (문서가 길어서 일부만 표시)"
            return content
    except Exception as e:
        return f"파일 읽기 오류: {e}"

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(30000)
            if len(content) == 30000:
                content += "\n\n... (문서가 길어서 일부만 표시)"
            return content
    except Exception as e:
        return f"파일 읽기 오류: {e}"


def analyze_data(path: str) -> str:
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


# ========================================
# Tool 실행기
# ========================================
def execute_tool(tool_data: dict) -> str:
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

    # ★ 스크린샷 - JSON 반환
    elif tool_name == "screenshot":
        result = take_screenshot()
        return json.dumps(result, ensure_ascii=False)

    # ★ 최신뉴스 - 구글뉴스 열고 스크린샷 찍고 닫기
    elif tool_name == "latest_news":
        result = latest_news()
        return json.dumps(result, ensure_ascii=False)

    elif tool_name == "analyze_data":
        return analyze_data(tool_data.get("path", ""))

    # ★ 지식베이스 도구들
    elif tool_name == "list_knowledge":
        results = list_knowledge()
        return json.dumps(results, ensure_ascii=False, indent=2)

    elif tool_name == "search_knowledge":
        results = search_knowledge(tool_data.get("keyword", ""))
        return json.dumps(results, ensure_ascii=False, indent=2)

    elif tool_name == "read_knowledge":
        return read_knowledge(tool_data.get("filename", ""))

    # ★ 프로세스 목록
    elif tool_name == "list_processes":
        results = list_processes(tool_data.get("sort_by", "memory"), tool_data.get("limit", 30))
        return json.dumps(results, ensure_ascii=False, indent=2)

    return "알 수 없는 도구"


# ========================================
# JSON 감지
# ========================================
def extract_tool_json(text: str) -> Optional[dict]:
    # 패턴 1: ```json 코드블록
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if "tool" in data:
                return data
        except json.JSONDecodeError:
            pass

    # 패턴 2: 인라인 {"tool": "..."}
    match = re.search(r'(\{[^{}]*"tool"\s*:\s*"[^"]+?"[^{}]*\})', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if "tool" in data:
                return data
        except json.JSONDecodeError:
            pass

    # 패턴 3: 전체가 JSON
    stripped = text.strip()
    if stripped.startswith('{') and stripped.endswith('}'):
        try:
            data = json.loads(stripped)
            if "tool" in data:
                return data
        except json.JSONDecodeError:
            pass

    # 패턴 4: 멀티라인 JSON
    match = re.search(r'\{\s*"tool"\s*:.*?\}', text, re.DOTALL)
    if match:
        try:
            json_str = re.sub(r'[\n\r\t]', ' ', match.group(0))
            json_str = re.sub(r'\s+', ' ', json_str)
            data = json.loads(json_str)
            if "tool" in data:
                return data
        except json.JSONDecodeError:
            pass

    return None


# ========================================
# ========================================
# Chat Processing
# ========================================
def get_recent_context(max_turns: int = 4) -> str:
    """최근 대화 기록을 문자열로 반환 (맥락 유지용)"""
    if not CHAT_HISTORY:
        return ""

    recent = CHAT_HISTORY[-(max_turns * 2):]  # user + assistant 쌍
    context_lines = []
    for msg in recent:
        role = "사용자" if msg["role"] == "user" else "비서"
        content = msg["content"][:500]  # 너무 길면 자르기
        context_lines.append(f"[{role}]: {content}")

    return "\n".join(context_lines)


def process_chat(user_message: str) -> str:
    global LOCAL_LLM, LLM_MODE

    if LLM_MODE == "local" and LOCAL_LLM is None:
        return "❌ 로컬 모델이 로드되지 않았습니다."
    if LLM_MODE != "local" and not API_TOKEN:
        return "❌ API 토큰이 없습니다."

    try:
        # ★ 대화 맥락 추가
        recent_context = get_recent_context(max_turns=3)
        if recent_context:
            context_prompt = f"""[이전 대화]
{recent_context}

[현재 질문]
{user_message}"""
        else:
            context_prompt = user_message

        result = call_llm(context_prompt, SYSTEM_PROMPT)
        if not result["success"]:
            return f"❌ LLM 오류: {result.get('error', '알 수 없는 오류')}"

        text = result["content"]
        logger.info(f"📝 LLM 응답: {text[:200]}")

        tool_data = extract_tool_json(text)

        if tool_data:
            try:
                if "keyword" in tool_data:
                    kw = tool_data["keyword"].replace("*", "").replace(".", "").strip()
                    if not kw:
                        return "❌ 검색 키워드가 비어있습니다."
                    tool_data["keyword"] = kw

                logger.info(f"🔧 도구 실행: {tool_data}")
                tool_result = execute_tool(tool_data)
                logger.info(f"📊 도구 결과: {tool_result[:300]}")

                tool_name = tool_data.get("tool")

                # ★ 스크린샷: 직접 포맷팅 (LLM 2차 호출 불필요)
                if tool_name == "screenshot":
                    try:
                        sc_data = json.loads(tool_result)
                        if sc_data.get("success"):
                            return f"📸 스크린샷을 찍었습니다!\n\n![스크린샷]({sc_data['url']})\n\n저장 위치: `{sc_data['path']}`"
                        else:
                            return f"❌ 스크린샷 실패: {sc_data.get('error', '?')}"
                    except:
                        return f"❌ 스크린샷 처리 오류"

                # ★ 최신뉴스: 직접 포맷팅
                if tool_name == "latest_news":
                    try:
                        news_data = json.loads(tool_result)
                        if news_data.get("success"):
                            return f"📰 **최신 뉴스** (구글뉴스)\n\n![뉴스]({news_data['url']})\n\n브라우저를 닫았습니다."
                        else:
                            return f"❌ 뉴스 확인 실패: {news_data.get('error', '?')}"
                    except:
                        return f"❌ 뉴스 처리 오류"

                # ========================================
                # ★ 지식베이스 핸들러 (3가지 구조)
                # ========================================

                # 1) read_knowledge → "이 문서를 참고해서 질문에 정확히 답변해라" + 기술 문서 전문가
                if tool_name == "read_knowledge":
                    if tool_result.startswith("❌"):
                        return tool_result
                    
                    # 문서가 너무 길면 앞부분 자르기
                    doc_content = tool_result if len(tool_result) <= 12000 else tool_result[:12000] + "\n\n... (이하 생략)"
                    
                    follow_up_prompt = f"""[사용자 질문]
{user_message}

[참고 문서]
{doc_content}

위 문서를 참고해서 사용자의 질문에 정확히 답변하세요.
문서에 있는 내용만 근거로 답변하고, 문서에 없는 내용은 추측하지 마세요."""

                    follow_up_system = """당신은 시니어 소프트웨어 엔지니어이자 기술 문서 전문가입니다.

[답변 형식]
**📋 핵심 요약**
질문에 대한 핵심 답변을 2~3줄로 요약

**📝 상세 내용**
문서에서 중요한 내용을 충분히 자세하게 정리:
- 주요 기능/목적
- 구성 요소 및 관계
- 동작 방식/흐름
- 중요한 설정이나 파라미터
- 주의사항이나 특이사항

[답변 규칙]
1. 문서 내용을 근거로 정확하고 **충분히 상세하게** 답변하세요.
2. 상세 내용은 최소 10줄 이상 작성하세요. 문서에 있는 중요 정보는 빠뜨리지 마세요.
3. 소스코드 원본은 보여주지 말고, 코드의 기능/역할/동작을 설명하세요.
4. 마크다운 표(| --- |) 사용 금지. "- 항목: 값" 형태로 나열하세요.
5. ## ### 대제목 헤더 대신 **볼드**와 이모지를 사용하세요.
6. 한국어로 답변하세요.
7. 절대 JSON을 출력하거나 도구를 호출하지 마세요."""

                    result2 = call_llm(follow_up_prompt, follow_up_system, max_tokens=6000)
                    if result2["success"]:
                        content = result2["content"].strip()
                        logger.info(f"📝 지식읽기 2차 응답: {content[:200] if content else '(빈 응답)'}")
                        if content and not extract_tool_json(content):
                            return content
                    # fallback: 문서 내용 직접 반환
                    logger.info("⚠️ 2차 LLM 응답 없음 → 문서 직접 반환")
                    return f"📄 **문서 내용:**\n\n{doc_content[:5000]}"

                # 2) search_knowledge → 관련성 높은 문서만 사용
                if tool_name == "search_knowledge":
                    try:
                        search_results = json.loads(tool_result)
                        if not search_results:
                            return "🔍 관련 문서를 찾지 못했습니다. 지식베이스에 문서를 먼저 등록해주세요."

                        # ★ 관련성 점수 기반으로 문서 선택
                        # 1위 문서와 점수 차이가 50% 이상이면 1위만 사용
                        MAX_TOTAL_LENGTH = 15000
                        merged_docs = []
                        total_length = 0
                        doc_names = []

                        top_score = search_results[0].get("score", 100)

                        for i, result in enumerate(search_results):
                            filename = result["filename"]
                            score = result.get("score", 0)

                            # 1위 문서와 점수 차이가 50% 이상이면 제외
                            if i > 0 and score < top_score * 0.5:
                                logger.info(f"⏭️ 점수 낮아 제외: {filename} (점수: {score}, 1위: {top_score})")
                                break

                            doc_content = read_knowledge(filename)

                            if doc_content.startswith("❌"):
                                continue

                            # 남은 공간에 맞게 자르기
                            remaining = MAX_TOTAL_LENGTH - total_length
                            if remaining <= 1000:
                                break

                            if len(doc_content) > remaining:
                                doc_content = doc_content[:remaining] + "\n\n... (문서 일부 생략)"

                            merged_docs.append(f"📄 **[{filename}]**\n{doc_content}")
                            doc_names.append(filename)
                            total_length += len(doc_content)

                        if not merged_docs:
                            return "🔍 문서를 읽을 수 없습니다."

                        combined_content = "\n\n---\n\n".join(merged_docs)
                        doc_list = ", ".join(doc_names)
                        logger.info(f"📚 참조 문서: {doc_list} (총 {total_length}자)")

                        follow_up_prompt = f"""[사용자 질문]
{user_message}

[참고 문서 {len(doc_names)}개: {doc_list}]
{combined_content}

위 문서들을 참고해서 사용자의 질문에 정확히 답변하세요.
여러 문서의 내용을 종합해서 답변하고, 문서에 없는 내용은 추측하지 마세요."""

                        follow_up_system = """당신은 시니어 소프트웨어 엔지니어이자 기술 문서 전문가입니다.

[답변 형식]
**📋 핵심 요약**
질문에 대한 핵심 답변을 2~3줄로 요약

**📝 상세 내용**
문서에서 중요한 내용을 충분히 자세하게 정리:
- 주요 기능/목적
- 구성 요소 및 관계
- 동작 방식/흐름
- 중요한 설정이나 파라미터
- 주의사항이나 특이사항

[답변 규칙]
1. 문서 내용을 근거로 정확하고 **충분히 상세하게** 답변하세요.
2. 상세 내용은 최소 10줄 이상 작성하세요. 문서에 있는 중요 정보는 빠뜨리지 마세요.
3. 소스코드 원본은 보여주지 말고, 코드의 기능/역할/동작을 설명하세요.
4. 마크다운 표(| --- |) 사용 금지. "- 항목: 값" 형태로 나열하세요.
5. ## ### 대제목 헤더 대신 **볼드**와 이모지를 사용하세요.
6. 한국어로 답변하세요.
7. 절대 JSON을 출력하거나 도구를 호출하지 마세요."""

                        result2 = call_llm(follow_up_prompt, follow_up_system, max_tokens=6000)
                        if result2["success"]:
                            content = result2["content"].strip()
                            logger.info(f"📝 지식검색 2차 응답: {content[:200] if content else '(빈 응답)'}")
                            if content and not extract_tool_json(content):
                                # 참조 문서 목록 추가
                                source_info = f"\n\n---\n📚 **참조 문서**: {doc_list}"
                                return content + source_info
                        # fallback: 문서 내용 직접 반환
                        logger.info("⚠️ 2차 LLM 응답 없음 → 문서 직접 반환")
                        return f"📄 **참조 문서:**\n\n{combined_content[:5000]}"
                    except Exception as e:
                        logger.error(f"지식검색 처리 오류: {e}")
                        pass

                # 3) list_knowledge → LLM 호출 없이 직접 포맷팅 (API 낭비 방지)
                if tool_name == "list_knowledge":
                    try:
                        files = json.loads(tool_result)
                        if not files:
                            return "📭 지식베이스에 등록된 문서가 없습니다.\n\n📚 지식베이스 버튼으로 MD/TXT 파일을 업로드하세요."
                        lines = [f"## 📚 지식베이스 문서 ({len(files)}개)\n"]
                        for f in files:
                            lines.append(f"- 📄 **{f['filename']}** ({f['size']}, {f['modified']})")
                        lines.append(f"\n💡 문서 내용이 궁금하면 파일명으로 질문하세요. (예: \"HID_INOUT 알려줘\")")
                        return "\n".join(lines)
                    except:
                        pass

                # 기타 도구: 2차 LLM으로 해석
                follow_up_prompt = f"""사용자 질문: {user_message}

도구 실행 결과:
{tool_result}

위 결과를 사용자가 이해하기 쉽게 한국어로 정리해서 답변하세요.
- JSON 원본을 보여주지 말고 핵심만 정리
- 도구를 다시 호출하지 마세요 (JSON 출력 금지)
- 마크다운 형식으로 보기 좋게"""

                follow_up_system = """당신은 PC 개인비서입니다.
도구 실행 결과를 한국어로 친절하게 설명합니다.
절대 JSON을 출력하지 마세요. 자연어로만 답변하세요."""

                result2 = call_llm(follow_up_prompt, follow_up_system)
                if result2["success"]:
                    response = result2["content"]
                    if extract_tool_json(response):
                        return format_tool_result_fallback(tool_data, tool_result)
                    return response
                else:
                    return format_tool_result_fallback(tool_data, tool_result)

            except json.JSONDecodeError as e:
                logger.error(f"❌ JSON 파싱 오류: {e}")
                return "❌ 명령 처리 중 오류가 발생했습니다."

        return text

    except Exception as e:
        logger.error(f"❌ 처리 오류: {e}")
        return f"❌ 오류: {e}"


# Fallback 포맷터
def format_tool_result_fallback(tool_data: dict, tool_result: str) -> str:
    tool_name = tool_data.get("tool", "")
    try:
        if tool_name == "get_system_info":
            info = json.loads(tool_result)
            lines = ["## 💻 시스템 정보", f"- **OS**: {info.get('os', '?')}", f"- **CPU**: {info.get('cpu', '?')}", f"- **메모리**: {info.get('memory', '?')}"]
            for d in info.get('drives', []):
                lines.append(f"- **{d['drive']}**: {d['total']} (사용률 {d['used']})")
            return "\n".join(lines)

        elif tool_name == "get_time":
            return f"🕐 현재 시간: {tool_result}"

        elif tool_name in ["search_files", "search_content"]:
            results = json.loads(tool_result)
            if not results:
                return f"🔍 '{tool_data.get('keyword', '')}' 검색 결과가 없습니다."
            lines = [f"🔍 검색 결과: **{len(results)}개** 발견\n"]
            for r in results[:10]:
                if "snippet" in r:
                    lines.append(f"- 📄 `{r['name']}` → {r['snippet']}")
                else:
                    lines.append(f"- {'📁' if r.get('type') == '폴더' else '📄'} `{r['name']}` ({r.get('size', '?')})")
            return "\n".join(lines)

        elif tool_name == "list_directory":
            items = json.loads(tool_result)
            lines = [f"📂 `{tool_data.get('path', '')}` 내용:\n"]
            for item in items[:20]:
                icon = "📁" if item.get("type") == "폴더" else "📄"
                lines.append(f"- {icon} `{item['name']}` ({item.get('size', '-')})")
            return "\n".join(lines)

        elif tool_name == "read_file":
            return f"📄 **파일 내용:**\n```\n{tool_result}\n```"

        elif tool_name in ["run_program", "kill_program", "open_web", "google_search"]:
            return f"✅ {tool_result}"

        elif tool_name == "list_processes":
            procs = json.loads(tool_result)
            if not procs or "error" in procs[0]:
                return "❌ 프로세스 목록을 가져올 수 없습니다."
            lines = [f"## 📋 실행 중인 프로세스 (상위 {len(procs)}개)\n"]
            lines.append("| 이름 | PID | 메모리(MB) | CPU% | 상태 |")
            lines.append("|------|-----|-----------|------|------|")
            for p in procs:
                lines.append(f"| {p['name']} | {p['pid']} | {p['memory_mb']} | {p['cpu']}% | {p['status']} |")
            return "\n".join(lines)

        elif tool_name == "analyze_data":
            return f"📊 **데이터 분석:**\n```\n{tool_result}\n```"

    except Exception as e:
        logger.error(f"포맷팅 오류: {e}")

    return f"📋 **결과:**\n```\n{tool_result}\n```"


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
    env: str

class ModelRequest(BaseModel):
    model_key: str


# ========================================
# Endpoints
# ========================================
def init_assistant():
    global LOCAL_LLM, LLM_MODE
    load_history()
    if load_api_token():
        LLM_MODE = "api"
        logger.info("✅ 비서: API 모드")
    else:
        LOCAL_LLM = load_local_model()
        if LOCAL_LLM:
            LLM_MODE = "local"
            logger.info("✅ 비서: LOCAL 모드")


@router.get("/")
async def assistant_home():
    return FileResponse(os.path.join(BASE_DIR, "assistant_ui.html"))


# ★ 스크린샷 이미지 서빙
@router.get("/screenshots/{filename}")
async def serve_screenshot(filename: str):
    filepath = os.path.join(SCREENSHOT_DIR, filename)
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="image/png")
    return {"error": "파일 없음"}


# ★ 스크린샷 목록
@router.get("/api/screenshots")
async def list_screenshots():
    files = []
    if os.path.exists(SCREENSHOT_DIR):
        for f in sorted(os.listdir(SCREENSHOT_DIR), reverse=True)[:20]:
            if f.endswith('.png'):
                filepath = os.path.join(SCREENSHOT_DIR, f)
                size = os.path.getsize(filepath)
                files.append({
                    "filename": f,
                    "url": f"/assistant/screenshots/{f}",
                    "size": f"{size / 1024:.0f}KB",
                    "time": datetime.datetime.fromtimestamp(os.path.getmtime(filepath)).strftime("%Y-%m-%d %H:%M:%S")
                })
    return {"screenshots": files}


@router.get("/api/status")
async def assistant_status():
    # 현재 로컬 모델 정보
    current_model_name = "LOCAL"
    if LLM_MODE == "local" and CURRENT_LOCAL_MODEL in AVAILABLE_MODELS:
        current_model_name = AVAILABLE_MODELS[CURRENT_LOCAL_MODEL]["name"]
    elif LLM_MODE != "local":
        current_model_name = ENV_CONFIG.get(CURRENT_ENV, {}).get("name", "API")

    return {
        "mode": LLM_MODE,
        "env": CURRENT_ENV if LLM_MODE != "local" else "local",
        "model_loaded": LOCAL_LLM is not None if LLM_MODE == "local" else API_TOKEN is not None,
        "model_name": current_model_name,
        "current_local_model": CURRENT_LOCAL_MODEL,
        "system": get_system_info(),
        "history_count": len(CHAT_HISTORY),
        "token_usage": TOKEN_USAGE
    }


# ★ 로컬 모델 목록 API
@router.get("/api/models")
async def list_local_models():
    """사용 가능한 로컬 GGUF 모델 목록"""
    return {
        "success": True,
        "models": get_available_local_models(),
        "current": CURRENT_LOCAL_MODEL
    }


# ★ 로컬 모델 변경 API
@router.post("/api/models/switch")
async def switch_local_model(request: ModelRequest):
    """로컬 GGUF 모델 변경"""
    global LOCAL_LLM, LLM_MODE

    model_key = request.model_key

    if model_key not in AVAILABLE_MODELS:
        return {"success": False, "error": f"알 수 없는 모델: {model_key}"}

    if not os.path.exists(AVAILABLE_MODELS[model_key]["path"]):
        return {"success": False, "error": f"모델 파일 없음: {AVAILABLE_MODELS[model_key]['name']}"}

    # 기존 모델 해제
    if LOCAL_LLM is not None:
        del LOCAL_LLM
        LOCAL_LLM = None
        import gc
        gc.collect()

    # 새 모델 로드
    LOCAL_LLM = load_local_model(model_key)
    if LOCAL_LLM:
        LLM_MODE = "local"
        return {
            "success": True,
            "model_key": model_key,
            "model_name": AVAILABLE_MODELS[model_key]["name"]
        }
    else:
        return {"success": False, "error": "모델 로드 실패"}


# ★ 토큰 사용량 API
@router.get("/api/tokens")
async def assistant_tokens():
    return {
        "success": True,
        "prompt_tokens": TOKEN_USAGE["prompt_tokens"],
        "completion_tokens": TOKEN_USAGE["completion_tokens"],
        "total_tokens": TOKEN_USAGE["total_tokens"],
        "call_count": TOKEN_USAGE["call_count"]
    }


@router.post("/api/tokens/reset")
async def assistant_reset_tokens():
    TOKEN_USAGE["prompt_tokens"] = 0
    TOKEN_USAGE["completion_tokens"] = 0
    TOKEN_USAGE["total_tokens"] = 0
    TOKEN_USAGE["call_count"] = 0
    return {"success": True, "message": "토큰 카운터 초기화됨"}


# ★ 파라미터 조회 API
@router.get("/api/params")
async def get_params():
    return {"success": True, "params": LLM_PARAMS}


# ★ 파라미터 변경 API
@router.post("/api/params")
async def set_params(request: dict):
    global LLM_PARAMS
    if "temperature" in request:
        LLM_PARAMS["temperature"] = float(request["temperature"])
    if "repeat_penalty" in request:
        LLM_PARAMS["repeat_penalty"] = float(request["repeat_penalty"])
    if "max_tokens" in request:
        LLM_PARAMS["max_tokens"] = int(request["max_tokens"])
    logger.info(f"⚙️ 파라미터 변경: {LLM_PARAMS}")
    return {"success": True, "params": LLM_PARAMS}


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
        return {"success": False, "error": "로컬 모델 로드 실패"}

    elif env in ENV_CONFIG:
        if not API_TOKEN and not load_api_token():
            return {"success": False, "error": "API 토큰 없음"}
        LLM_MODE = "api"
        CURRENT_ENV = env
        API_URL = ENV_CONFIG[env]["url"]
        API_MODEL = ENV_CONFIG[env]["model"]
        return {"success": True, "env": env, "name": ENV_CONFIG[env]["name"]}

    return {"success": False, "error": f"알 수 없는 환경: {env}"}


@router.post("/api/chat")
async def assistant_chat(request: ChatRequest):
    user_msg = request.message.strip()
    CHAT_HISTORY.append({"role": "user", "content": user_msg, "time": datetime.datetime.now().isoformat()})
    response = process_chat(user_msg)
    CHAT_HISTORY.append({"role": "assistant", "content": response, "time": datetime.datetime.now().isoformat()})
    save_history()
    return {"success": True, "response": response}


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


# ★ 지식베이스 API
@router.get("/api/knowledge")
async def api_list_knowledge():
    """지식베이스 문서 목록"""
    files = list_knowledge()
    return {"success": True, "files": files, "count": len(files)}


@router.post("/api/knowledge/upload")
async def api_upload_knowledge(file: UploadFile = File(...)):
    """MD/TXT 파일 업로드"""
    if not file.filename.lower().endswith(('.md', '.txt')):
        return {"success": False, "error": "md 또는 txt 파일만 업로드 가능합니다."}
    try:
        filepath = os.path.join(KNOWLEDGE_DIR, file.filename)
        content = await file.read()
        with open(filepath, 'wb') as f:
            f.write(content)
        return {"success": True, "filename": file.filename, "size": len(content)}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.delete("/api/knowledge/{filename}")
async def api_delete_knowledge(filename: str):
    """지식베이스 문서 삭제"""
    filepath = os.path.join(KNOWLEDGE_DIR, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
        return {"success": True, "message": f"'{filename}' 삭제됨"}
    return {"success": False, "error": "파일 없음"}


@router.get("/api/knowledge/download/{filename}")
async def api_download_knowledge(filename: str):
    """지식베이스 문서 다운로드"""
    filepath = os.path.join(KNOWLEDGE_DIR, filename)
    if os.path.exists(filepath):
        return FileResponse(filepath, filename=filename, media_type="application/octet-stream")
    return JSONResponse(status_code=404, content={"error": "파일 없음"})


# ========================================
# 과거지식 보관소 API
# ========================================
@router.get("/api/knowledge/archive")
async def api_list_archive():
    """과거지식 문서 목록"""
    files = []
    try:
        for f in sorted(os.listdir(KNOWLEDGE_ARCHIVE_DIR)):
            if f.lower().endswith(('.md', '.txt')):
                filepath = os.path.join(KNOWLEDGE_ARCHIVE_DIR, f)
                size = os.path.getsize(filepath)
                modified = datetime.datetime.fromtimestamp(os.path.getmtime(filepath)).strftime("%Y-%m-%d %H:%M")
                size_str = f"{size / 1024:.1f}KB" if size > 1024 else f"{size}B"
                files.append({"filename": f, "size": size_str, "modified": modified})
    except Exception as e:
        logger.error(f"과거지식 목록 오류: {e}")
    return {"success": True, "files": files, "count": len(files)}


@router.post("/api/knowledge/archive/{filename}")
async def api_archive_knowledge(filename: str):
    """지식베이스 → 과거지식으로 이동"""
    import shutil
    src = os.path.join(KNOWLEDGE_DIR, filename)
    dst = os.path.join(KNOWLEDGE_ARCHIVE_DIR, filename)
    if not os.path.exists(src):
        return {"success": False, "error": f"'{filename}' 파일 없음"}
    try:
        shutil.move(src, dst)
        return {"success": True, "message": f"'{filename}' → 과거지식으로 이동됨"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.post("/api/knowledge/restore/{filename}")
async def api_restore_knowledge(filename: str):
    """과거지식 → 지식베이스로 복원"""
    import shutil
    src = os.path.join(KNOWLEDGE_ARCHIVE_DIR, filename)
    dst = os.path.join(KNOWLEDGE_DIR, filename)
    if not os.path.exists(src):
        return {"success": False, "error": f"'{filename}' 파일 없음"}
    try:
        shutil.move(src, dst)
        return {"success": True, "message": f"'{filename}' → 지식베이스로 복원됨"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.delete("/api/knowledge/archive/{filename}")
async def api_delete_archive(filename: str):
    """과거지식 문서 완전 삭제"""
    filepath = os.path.join(KNOWLEDGE_ARCHIVE_DIR, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
        return {"success": True, "message": f"'{filename}' 완전 삭제됨"}
    return {"success": False, "error": "파일 없음"}


if __name__ == "__main__":
    import uvicorn
    app.include_router(router)

    @app.on_event("startup")
    async def standalone_startup():
        init_assistant()

    uvicorn.run(app, host="0.0.0.0", port=10003)