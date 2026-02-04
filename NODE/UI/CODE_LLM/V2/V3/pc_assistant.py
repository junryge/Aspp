#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PC 개인비서 AI (Moltbot 스타일 Tool Calling)
- LLM이 JSON으로 도구 호출
- 파일 검색, 내용 검색
- 시스템 정보
- 프로그램 실행
- API 및 GGUF 모두 지원
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
2. 도구를 호출할 때는 JSON만 출력하세요. 다른 텍스트를 JSON 앞뒤에 붙이지 마세요.
3. keyword에는 확장자(.gguf)나 와일드카드(*) 없이 순수 키워드만 넣으세요. 예: "gguf", "txt", "python"
4. 반드시 순수 JSON 객체만 출력하세요. ```json 코드블록으로 감싸지 마세요.

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

일반 대화(인사, 잡담, 코딩 질문 등)는 도구를 호출하지 말고 한국어로 자연스럽게 답변하세요."""


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


# ★ 수정 1: 토큰 로드 경로 통일 (token.txt 포함)
def load_api_token():
    """API 토큰 로드 - 메인 서버와 동일한 경로 검색"""
    global API_TOKEN
    paths = [
        os.path.join(BASE_DIR, "token.txt"),       # ★ 메인 서버와 동일
        os.path.join(BASE_DIR, "api_token.txt"),    # 기존 호환
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
            return {"success": False, "error": f"API 오류: {response.status_code} - {response.text[:200]}"}
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


def take_screenshot() -> str:
    """스크린샷"""
    try:
        from PIL import ImageGrab
        path = os.path.join(BASE_DIR, f"screenshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        img = ImageGrab.grab()
        img.save(path)
        return f"스크린샷 저장: {path}"
    except Exception as e:
        return f"스크린샷 오류: {e}"


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
        return take_screenshot()

    elif tool_name == "analyze_data":
        return analyze_data(tool_data.get("path", ""))

    return "알 수 없는 도구"


# ========================================
# ★ 수정 2: JSON 감지 강화
# ========================================
def extract_tool_json(text: str) -> Optional[dict]:
    """LLM 응답에서 도구 호출 JSON 추출 - 다양한 형식 대응"""
    
    # 패턴 1: ```json 코드블록
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if "tool" in data:
                logger.info(f"✅ JSON 감지 (코드블록): {data.get('tool')}")
                return data
        except json.JSONDecodeError:
            pass
    
    # 패턴 2: "tool" 키가 포함된 JSON 객체 (중첩 없는 단순 객체)
    match = re.search(r'(\{[^{}]*"tool"\s*:\s*"[^"]+?"[^{}]*\})', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if "tool" in data:
                logger.info(f"✅ JSON 감지 (인라인): {data.get('tool')}")
                return data
        except json.JSONDecodeError:
            pass
    
    # 패턴 3: 텍스트 전체가 JSON인 경우 (앞뒤 공백/줄바꿈만 있는 경우)
    stripped = text.strip()
    if stripped.startswith('{') and stripped.endswith('}'):
        try:
            data = json.loads(stripped)
            if "tool" in data:
                logger.info(f"✅ JSON 감지 (전체): {data.get('tool')}")
                return data
        except json.JSONDecodeError:
            pass
    
    # 패턴 4: 텍스트 안에 줄바꿈이 포함된 JSON (멀티라인)
    match = re.search(r'\{\s*"tool"\s*:.*?\}', text, re.DOTALL)
    if match:
        try:
            # 줄바꿈, 탭 정리
            json_str = match.group(0)
            json_str = re.sub(r'[\n\r\t]', ' ', json_str)
            json_str = re.sub(r'\s+', ' ', json_str)
            data = json.loads(json_str)
            if "tool" in data:
                logger.info(f"✅ JSON 감지 (멀티라인): {data.get('tool')}")
                return data
        except json.JSONDecodeError:
            pass
    
    return None


# ========================================
# Chat Processing (★ 수정된 버전)
# ========================================
def process_chat(user_message: str) -> str:
    """채팅 처리 (Tool Calling 방식)"""
    global LOCAL_LLM, LLM_MODE

    # 모델 체크
    if LLM_MODE == "local" and LOCAL_LLM is None:
        return "❌ 로컬 모델이 로드되지 않았습니다. API 모드로 전환해주세요."
    if LLM_MODE != "local" and not API_TOKEN:
        return "❌ API 토큰이 없습니다. token.txt 파일을 확인해주세요."

    # 1차: LLM 호출 (도구 호출 여부 판단)
    try:
        result = call_llm(user_message, SYSTEM_PROMPT)
        if not result["success"]:
            return f"❌ LLM 오류: {result.get('error', '알 수 없는 오류')}"

        text = result["content"]
        logger.info(f"📝 LLM 응답 (첫 200자): {text[:200]}")

        # ★ 강화된 JSON 도구 호출 감지
        tool_data = extract_tool_json(text)

        if tool_data:
            try:
                # keyword에서 와일드카드/확장자 제거
                if "keyword" in tool_data:
                    kw = tool_data["keyword"]
                    kw = kw.replace("*", "").replace(".", "").strip()
                    # 빈 키워드 방지
                    if not kw:
                        return "❌ 검색 키워드가 비어있습니다. 다시 입력해주세요."
                    tool_data["keyword"] = kw

                logger.info(f"🔧 도구 실행: {tool_data}")
                tool_result = execute_tool(tool_data)
                logger.info(f"📊 도구 결과 (첫 300자): {tool_result[:300]}")

                # 2차: 결과 해석 - 자연어로 변환
                follow_up_prompt = f"""사용자 질문: {user_message}

도구 실행 결과:
{tool_result}

위 결과를 사용자가 이해하기 쉽게 한국어로 자연스럽게 정리해서 답변하세요.
- JSON이나 원본 데이터를 그대로 보여주지 말고 핵심만 정리
- 도구를 다시 호출하지 마세요 (JSON 출력 금지)
- 마크다운 형식으로 보기 좋게 정리"""

                follow_up_system = """당신은 PC 개인비서입니다. 
도구 실행 결과를 받아서 사용자에게 친절하게 한국어로 설명합니다.
절대 JSON을 출력하지 마세요. 도구를 호출하지 마세요.
자연어로만 답변하세요."""

                result2 = call_llm(follow_up_prompt, follow_up_system)
                if result2["success"]:
                    response = result2["content"]
                    # 2차 응답에서도 혹시 JSON이 나오면 필터링
                    if extract_tool_json(response):
                        logger.warning("⚠️ 2차 응답에서도 JSON 감지 - 도구 결과 직접 포맷팅")
                        return format_tool_result_fallback(tool_data, tool_result)
                    return response
                else:
                    # 2차 호출 실패 시 직접 포맷팅
                    logger.warning(f"⚠️ 2차 LLM 실패: {result2.get('error')}")
                    return format_tool_result_fallback(tool_data, tool_result)

            except json.JSONDecodeError as e:
                logger.error(f"❌ JSON 파싱 오류: {e}")
                return "❌ 명령 처리 중 오류가 발생했습니다. 다시 시도해주세요."

        # 도구 호출 없으면 그냥 응답 (일반 대화)
        return text

    except Exception as e:
        logger.error(f"❌ 처리 오류: {e}")
        return f"❌ 오류: {e}"


def format_tool_result_fallback(tool_data: dict, tool_result: str) -> str:
    """2차 LLM 실패 시 도구 결과를 직접 포맷팅"""
    tool_name = tool_data.get("tool", "")
    
    try:
        if tool_name == "get_system_info":
            info = json.loads(tool_result)
            lines = [
                "## 💻 시스템 정보",
                f"- **OS**: {info.get('os', '?')}",
                f"- **CPU**: {info.get('cpu', '?')}",
                f"- **메모리**: {info.get('memory', '?')}",
            ]
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
                    lines.append(f"- {'📁' if r.get('type') == '폴더' else '📄'} `{r['name']}` ({r.get('size', '?')}) → `{r['path']}`")
            if len(results) > 10:
                lines.append(f"\n... 외 {len(results) - 10}개 더 있음")
            return "\n".join(lines)
        
        elif tool_name == "list_directory":
            items = json.loads(tool_result)
            if not items or (len(items) == 1 and "error" in items[0]):
                return f"❌ 폴더를 열 수 없습니다: {items[0].get('error', '?') if items else '?'}"
            lines = [f"📂 `{tool_data.get('path', '')}` 내용: **{len(items)}개**\n"]
            for item in items[:20]:
                icon = "📁" if item.get("type") == "폴더" else "📄"
                lines.append(f"- {icon} `{item['name']}` ({item.get('size', '-')}) - {item.get('modified', '?')}")
            return "\n".join(lines)
        
        elif tool_name == "read_file":
            return f"📄 **파일 내용:**\n```\n{tool_result}\n```"
        
        elif tool_name in ["run_program", "kill_program", "open_web", "google_search", "screenshot"]:
            return f"✅ {tool_result}"
        
        elif tool_name == "analyze_data":
            return f"📊 **데이터 분석 결과:**\n```\n{tool_result}\n```"
        
    except Exception as e:
        logger.error(f"포맷팅 오류: {e}")
    
    # 최후의 수단: 결과를 코드블록으로 감싸서 반환
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
    env: str  # "local", "dev", "prod", "common"


# ========================================
# API Endpoints
# ========================================
# 초기화 함수 (메인 서버에서 호출)
def init_assistant():
    global LOCAL_LLM, LLM_MODE
    load_history()
    # ★ 토큰 먼저 시도 → API 모드 우선
    if load_api_token():
        LLM_MODE = "api"
        logger.info("✅ 비서: API 모드로 시작")
    else:
        LOCAL_LLM = load_local_model()
        if LOCAL_LLM:
            LLM_MODE = "local"
            logger.info("✅ 비서: LOCAL 모드로 시작")
        else:
            logger.warning("⚠️ 비서: 모델 없음 (API 토큰도 없고 로컬 모델도 없음)")

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
        # 로컬 모드로 전환
        if LOCAL_LLM is None:
            LOCAL_LLM = load_local_model()
        if LOCAL_LLM:
            LLM_MODE = "local"
            return {"success": True, "env": "local", "name": "LOCAL(14B-GGUF)"}
        else:
            return {"success": False, "error": "로컬 모델 로드 실패"}

    elif env in ENV_CONFIG:
        # API 모드로 전환
        if not API_TOKEN:
            if not load_api_token():
                return {"success": False, "error": "API 토큰 없음. token.txt를 확인하세요."}

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


# 단독 실행용 (테스트)
if __name__ == "__main__":
    import uvicorn
    # 단독 실행 시 router를 app에 포함
    app.include_router(router)

    @app.on_event("startup")
    async def standalone_startup():
        init_assistant()

    uvicorn.run(app, host="0.0.0.0", port=8002)