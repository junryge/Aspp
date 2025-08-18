# -*- coding: utf-8 -*-

pip install llama-cpp-python --force-reinstall --no-cache-dir

# 기존 삭제
pip uninstall llama-cpp-python -y

# Python 3.11 Windows CPU 버전 직접 설치
pip install https://github.com/abetlen/llama-cpp-python/releases/download/v0.2.90/llama_cpp_python-0.2.90-cp311-cp311-win_amd64.whl

# 설치 확인 테스트
python -c "from llama_cpp import Llama; print('✅ 설치 성공!')"


# 컴파일 중단: Ctrl + C

# 캐시 삭제 후 재시도
pip cache purge
pip install llama-cpp-python --prefer-binary --no-build-isolation
"""
에이전틱 AI 시스템 - GGUF 모델 기반 자율 AI 어시스턴트
Multi-Agent Architecture with GGUF Model Integration
"""

import os
import sys
import json
import time
import threading
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import customtkinter as ctk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import logging
from datetime import datetime
import traceback
from abc import ABC, abstractmethod
import queue
import subprocess
import webbrowser
import sqlite3

# llama-cpp-python 직접 import
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print("llama-cpp-python가 설치되지 않았습니다. 설치해주세요: pip install llama-cpp-python")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("agentic_ai.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 현대적 테마 설정
MODERN_THEME = {
    'primary': '#6366F1',      # Indigo
    'secondary': '#8B5CF6',    # Purple
    'accent': '#EC4899',       # Pink
    'success': '#10B981',      # Emerald
    'warning': '#F59E0B',      # Amber
    'error': '#EF4444',        # Red
    'info': '#3B82F6',         # Blue
    'dark': '#1F2937',         # Gray-800
    'light': '#F9FAFB',        # Gray-50
    'surface': '#FFFFFF',
    'text': '#111827',         # Gray-900
    'text_secondary': '#6B7280', # Gray-500
    'border': '#E5E7EB',       # Gray-200
    'hover': '#4F46E5',        # Indigo-600
    'gradient_start': '#6366F1',
    'gradient_end': '#8B5CF6'
}

# ============= GGUF 모델 관리자 =============
class GGUFModelManager:
    """GGUF 모델 관리자"""
    
    def __init__(self):
        self.model = None
        self.model_path = ""
        self.model_loaded = False
        self.context_size = 2048
        self.temperature = 0.7
        self.max_tokens = 1000
        self.top_p = 0.95
        self.top_k = 40
        self.repeat_penalty = 1.1
        self.n_threads = 4
        self.n_gpu_layers = 0
        
    def load_model(self, model_path: str, **kwargs) -> bool:
        """GGUF 모델 로드"""
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError("llama-cpp-python이 설치되지 않았습니다.")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일이 존재하지 않습니다: {model_path}")
        
        # 파일 크기 확인 (너무 작으면 손상된 파일일 가능성)
        file_size = os.path.getsize(model_path)
        if file_size < 1024 * 1024:  # 1MB 미만
            raise ValueError(f"모델 파일이 너무 작습니다. 손상되었을 가능성이 있습니다: {file_size} bytes")
        
        try:
            # 기존 모델 정리
            if self.model:
                try:
                    del self.model
                except:
                    pass
                self.model = None
                self.model_loaded = False
            
            # 설정 업데이트
            self.context_size = kwargs.get('context_size', self.context_size)
            self.n_threads = kwargs.get('n_threads', self.n_threads)
            self.n_gpu_layers = kwargs.get('n_gpu_layers', self.n_gpu_layers)
            
            # 안전한 기본값으로 시작
            logger.info(f"모델 로드 시도: {model_path}")
            logger.info(f"설정 - Context: {self.context_size}, Threads: {self.n_threads}, GPU Layers: {self.n_gpu_layers}")
            
            # 모델 로드 (verbose=True로 디버깅 정보 출력)
            try:
                self.model = Llama(
                    model_path=model_path,
                    n_ctx=self.context_size,
                    n_threads=self.n_threads,
                    n_gpu_layers=self.n_gpu_layers,
                    verbose=True,  # 디버깅을 위해 True로 설정
                    seed=-1,
                    n_batch=512,  # 배치 크기 명시
                    f16_kv=True,  # Half precision
                    logits_all=False,
                    vocab_only=False,
                    use_mmap=True,  # 메모리 맵 사용
                    use_mlock=False  # 메모리 락 비활성화
                )
            except Exception as model_error:
                # GPU 레이어 문제일 수 있으므로 CPU 전용으로 재시도
                logger.warning(f"GPU 로드 실패, CPU 모드로 재시도: {str(model_error)}")
                self.n_gpu_layers = 0
                self.model = Llama(
                    model_path=model_path,
                    n_ctx=min(self.context_size, 512),  # 컨텍스트 크기 줄여서 재시도
                    n_threads=self.n_threads,
                    n_gpu_layers=0,  # GPU 비활성화
                    verbose=False,
                    seed=-1,
                    use_mmap=False,  # 메모리 맵 비활성화
                    use_mlock=False
                )
            
            self.model_path = model_path
            self.model_loaded = True
            logger.info(f"GGUF 모델 로드 성공: {model_path}")
            return True
            
        except OSError as e:
            error_msg = f"모델 파일 접근 오류: {str(e)}"
            logger.error(error_msg)
            self.model_loaded = False
            self.model = None
            raise RuntimeError(error_msg)
            
        except MemoryError as e:
            error_msg = f"메모리 부족: 모델이 너무 큽니다. 더 작은 모델을 사용하거나 컨텍스트 크기를 줄이세요."
            logger.error(error_msg)
            self.model_loaded = False
            self.model = None
            raise RuntimeError(error_msg)
            
        except Exception as e:
            error_msg = f"모델 로드 오류: {str(e)}"
            logger.error(error_msg)
            logger.error(f"오류 타입: {type(e).__name__}")
            self.model_loaded = False
            self.model = None
            raise RuntimeError(error_msg)
    
    def generate(self, prompt: str, **kwargs) -> str:
        """텍스트 생성"""
        if not self.model_loaded or not self.model:
            raise RuntimeError("모델이 로드되지 않았습니다.")
        
        # 생성 파라미터
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        top_p = kwargs.get('top_p', self.top_p)
        top_k = kwargs.get('top_k', self.top_k)
        repeat_penalty = kwargs.get('repeat_penalty', self.repeat_penalty)
        
        try:
            # 응답 생성
            response = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                echo=False,
                stop=["User:", "Human:", "Assistant:"]
            )
            
            # 텍스트 추출
            if isinstance(response, dict) and "choices" in response:
                return response["choices"][0]["text"].strip()
            else:
                return str(response).strip()
                
        except Exception as e:
            logger.error(f"텍스트 생성 오류: {str(e)}")
            raise
    
    def is_loaded(self) -> bool:
        """모델 로드 상태 확인"""
        return self.model_loaded

# ============= 에이전트 타입 정의 =============
class AgentType(Enum):
    """에이전트 타입 정의"""
    ORCHESTRATOR = "orchestrator"     # 전체 조정자
    PLANNER = "planner"               # 계획 수립
    EXECUTOR = "executor"             # 실행 담당
    RESEARCHER = "researcher"         # 정보 수집
    ANALYZER = "analyzer"             # 분석 담당
    CREATOR = "creator"               # 콘텐츠 생성
    REVIEWER = "reviewer"             # 검토 담당

class TaskStatus(Enum):
    """작업 상태"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

# ============= 도구 인터페이스 =============
class Tool(ABC):
    """도구 기본 클래스"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    async def execute(self, *args, **kwargs) -> Any:
        """도구 실행"""
        pass

class WebSearchTool(Tool):
    """웹 검색 도구"""
    
    def __init__(self):
        super().__init__(
            name="web_search",
            description="웹에서 정보를 검색합니다"
        )
    
    async def execute(self, query: str) -> str:
        """웹 검색 실행"""
        await asyncio.sleep(0.5)
        # 실제로는 검색 API 호출
        return f"[웹 검색 결과]\n주제: {query}\n- 관련 정보 1\n- 관련 정보 2\n- 관련 정보 3"

class FileSystemTool(Tool):
    """파일 시스템 도구"""
    
    def __init__(self):
        super().__init__(
            name="file_system",
            description="파일을 읽고 쓸 수 있습니다"
        )
    
    async def execute(self, action: str, path: str, content: str = None) -> str:
        """파일 작업 실행"""
        if action == "read":
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                return f"파일 읽기 실패: {str(e)}"
        elif action == "write":
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return f"파일 저장 완료: {path}"
            except Exception as e:
                return f"파일 쓰기 실패: {str(e)}"
        elif action == "list":
            try:
                files = os.listdir(os.path.dirname(path) or '.')
                return f"파일 목록: {', '.join(files)}"
            except Exception as e:
                return f"디렉토리 읽기 실패: {str(e)}"
        return "지원하지 않는 작업입니다."

class CodeExecutorTool(Tool):
    """코드 실행 도구"""
    
    def __init__(self):
        super().__init__(
            name="code_executor",
            description="Python 코드를 실행합니다"
        )
    
    async def execute(self, code: str) -> str:
        """코드 실행"""
        try:
            # 간단한 수식 계산만 허용 (보안상)
            if len(code) < 100 and not any(danger in code for danger in ['import', 'exec', 'eval', '__']):
                result = eval(code)
                return f"실행 결과: {result}"
            else:
                return "보안상 실행할 수 없는 코드입니다."
        except Exception as e:
            return f"실행 오류: {str(e)}"

class DatabaseTool(Tool):
    """데이터베이스 도구"""
    
    def __init__(self):
        super().__init__(
            name="database",
            description="데이터베이스 작업을 수행합니다"
        )
        self.conn = None
        self.init_db()
    
    def init_db(self):
        """데이터베이스 초기화"""
        self.conn = sqlite3.connect('agentic_ai.db', check_same_thread=False)
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT,
                status TEXT,
                agent_type TEXT,
                priority INTEGER,
                result TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT,
                content TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit()
    
    async def execute(self, action: str, **kwargs) -> str:
        """데이터베이스 작업 실행"""
        try:
            cursor = self.conn.cursor()
            
            if action == "save_task":
                cursor.execute(
                    "INSERT INTO tasks (id, title, description, status, agent_type, priority) VALUES (?, ?, ?, ?, ?, ?)",
                    (kwargs.get('id'), kwargs.get('title'), kwargs.get('description'), 
                     kwargs.get('status'), kwargs.get('agent_type'), kwargs.get('priority'))
                )
                self.conn.commit()
                return "작업이 데이터베이스에 저장되었습니다."
                
            elif action == "update_task":
                cursor.execute(
                    "UPDATE tasks SET status = ?, result = ?, completed_at = ? WHERE id = ?",
                    (kwargs.get('status'), kwargs.get('result'), datetime.now(), kwargs.get('id'))
                )
                self.conn.commit()
                return "작업 상태가 업데이트되었습니다."
                
            elif action == "get_tasks":
                cursor.execute("SELECT * FROM tasks ORDER BY created_at DESC LIMIT 10")
                tasks = cursor.fetchall()
                return f"최근 작업 {len(tasks)}개를 가져왔습니다."
                
            elif action == "save_memory":
                cursor.execute(
                    "INSERT INTO agent_memory (agent_name, content) VALUES (?, ?)",
                    (kwargs.get('agent_name'), kwargs.get('content'))
                )
                self.conn.commit()
                return "메모리가 저장되었습니다."
                
            else:
                return "지원하지 않는 데이터베이스 작업입니다."
                
        except Exception as e:
            return f"데이터베이스 오류: {str(e)}"

# ============= 작업 정의 =============
@dataclass
class Task:
    """작업 정의"""
    id: str
    title: str
    description: str
    agent_type: AgentType
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 5
    dependencies: List[str] = field(default_factory=list)
    result: Any = None
    error: str = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

# ============= 에이전트 클래스 =============
class Agent:
    """GGUF 모델 기반 에이전트"""
    
    def __init__(self, name: str, agent_type: AgentType, model_manager: GGUFModelManager = None):
        self.name = name
        self.agent_type = agent_type
        self.model_manager = model_manager
        self.tools: Dict[str, Tool] = {}
        self.memory: List[Dict] = []
        self.current_task: Optional[Task] = None
        
    def add_tool(self, tool: Tool):
        """도구 추가"""
        self.tools[tool.name] = tool
        logger.info(f"도구 추가: {tool.name} -> {self.name}")
        
    async def process_task(self, task: Task) -> Any:
        """작업 처리"""
        self.current_task = task
        task.status = TaskStatus.IN_PROGRESS
        
        try:
            # 모델이 로드되어 있으면 AI 기반 처리
            if self.model_manager and self.model_manager.is_loaded():
                result = await self.process_with_ai(task)
            else:
                # 모델이 없으면 기본 처리
                result = await self.process_without_ai(task)
            
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result
            
            # 메모리에 저장
            self.memory.append({
                'task': task.title,
                'result': result,
                'timestamp': datetime.now()
            })
            
            # 데이터베이스에 저장
            if 'database' in self.tools:
                await self.tools['database'].execute(
                    'update_task',
                    id=task.id,
                    status='completed',
                    result=str(result)
                )
            
            return result
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            logger.error(f"Agent {self.name} failed: {str(e)}")
            raise
    
    async def process_with_ai(self, task: Task) -> str:
        """AI 모델을 사용한 작업 처리"""
        # 1. 작업 분석 프롬프트 생성
        analysis_prompt = f"""
당신은 {self.agent_type.value} 역할을 수행하는 AI 에이전트입니다.

작업: {task.title}
설명: {task.description}

사용 가능한 도구: {', '.join(self.tools.keys())}

이 작업을 어떻게 수행하시겠습니까? 단계별로 설명해주세요.
"""
        
        # 2. AI 모델로 계획 생성
        try:
            plan = self.model_manager.generate(analysis_prompt, max_tokens=500)
            
            # 3. 도구 실행 여부 결정
            result_parts = [f"[{self.name}의 분석]\n{plan}\n"]
            
            # 4. 필요한 도구 실행
            for tool_name, tool in self.tools.items():
                if tool_name in task.description.lower() or tool_name in plan.lower():
                    if tool_name == "web_search":
                        search_result = await tool.execute(task.title)
                        result_parts.append(f"\n{search_result}")
                    elif tool_name == "file_system":
                        # 결과를 파일로 저장
                        file_path = f"outputs/{task.id}.txt"
                        await tool.execute("write", file_path, plan)
                        result_parts.append(f"\n결과가 {file_path}에 저장되었습니다.")
            
            # 5. 최종 응답 생성
            final_prompt = f"""
작업 분석 결과:
{plan}

위 분석을 바탕으로 사용자에게 제공할 최종 답변을 작성해주세요.
"""
            
            final_response = self.model_manager.generate(final_prompt, max_tokens=500)
            result_parts.append(f"\n[최종 답변]\n{final_response}")
            
            return "\n".join(result_parts)
            
        except Exception as e:
            logger.error(f"AI 처리 중 오류: {str(e)}")
            return await self.process_without_ai(task)
    
    async def process_without_ai(self, task: Task) -> str:
        """AI 모델 없이 기본 처리"""
        results = [f"[{self.name}가 작업을 처리중입니다]"]
        
        # 기본적인 도구 실행
        if "검색" in task.description or "search" in task.description.lower():
            if "web_search" in self.tools:
                result = await self.tools["web_search"].execute(task.title)
                results.append(result)
        
        if "파일" in task.description or "저장" in task.description:
            if "file_system" in self.tools:
                file_path = f"outputs/{task.id}.txt"
                result = await self.tools["file_system"].execute(
                    "write", file_path, f"작업: {task.title}\n내용: {task.description}"
                )
                results.append(result)
        
        if "계산" in task.description:
            if "code_executor" in self.tools:
                # 간단한 계산 추출 시도
                import re
                numbers = re.findall(r'\d+', task.description)
                if len(numbers) >= 2:
                    calc = f"{numbers[0]} + {numbers[1]}"
                    result = await self.tools["code_executor"].execute(calc)
                    results.append(result)
        
        if len(results) == 1:
            results.append(f"작업 '{task.title}'을(를) 완료했습니다.")
        
        return "\n".join(results)

# ============= 멀티 에이전트 오케스트레이터 =============
class AgentOrchestrator:
    """멀티 에이전트 시스템 조정자"""
    
    def __init__(self, model_manager: GGUFModelManager = None):
        self.model_manager = model_manager
        self.agents: Dict[str, Agent] = {}
        self.task_queue: queue.Queue = queue.Queue()
        self.completed_tasks: List[Task] = []
        self.running = False
        self.db_tool = DatabaseTool()
        
    def add_agent(self, agent: Agent):
        """에이전트 추가"""
        self.agents[agent.name] = agent
        logger.info(f"Agent added: {agent.name} ({agent.agent_type.value})")
        
    def submit_task(self, task: Task):
        """작업 제출"""
        self.task_queue.put(task)
        
        # 데이터베이스에 저장
        try:
            asyncio.create_task(self.db_tool.execute(
                'save_task',
                id=task.id,
                title=task.title,
                description=task.description,
                status=task.status.value,
                agent_type=task.agent_type.value,
                priority=task.priority
            ))
        except:
            pass
        
        logger.info(f"Task submitted: {task.title}")
        
    async def process_tasks(self):
        """작업 처리 루프"""
        self.running = True
        
        while self.running:
            if not self.task_queue.empty():
                task = self.task_queue.get()
                
                # 적절한 에이전트 선택
                agent = self.select_agent(task)
                
                if agent:
                    try:
                        result = await agent.process_task(task)
                        self.completed_tasks.append(task)
                        logger.info(f"Task completed: {task.title}")
                    except Exception as e:
                        logger.error(f"Task failed: {task.title} - {str(e)}")
                else:
                    logger.warning(f"No suitable agent for task: {task.title}")
                    
            await asyncio.sleep(0.1)
            
    def select_agent(self, task: Task) -> Optional[Agent]:
        """작업에 적합한 에이전트 선택"""
        # 지정된 타입의 에이전트 찾기
        for agent in self.agents.values():
            if agent.agent_type == task.agent_type:
                return agent
        
        # 없으면 실행자 에이전트 반환
        for agent in self.agents.values():
            if agent.agent_type == AgentType.EXECUTOR:
                return agent
                
        # 그래도 없으면 첫 번째 에이전트
        return list(self.agents.values())[0] if self.agents else None
        
    def stop(self):
        """오케스트레이터 중지"""
        self.running = False

# ============= 메인 애플리케이션 UI =============
class AgenticAIApp(ctk.CTk):
    """에이전틱 AI 시스템 UI"""
    
    def __init__(self):
        super().__init__()
        
        # 앱 초기화
        self.title("🤖 Agentic AI System - GGUF 모델 기반 자율 AI")
        self.geometry("1400x900")
        
        # 디렉토리 생성
        self.create_directories()
        
        # 모델 관리자 초기화
        self.model_manager = GGUFModelManager()
        
        # 시스템 초기화
        self.orchestrator = AgentOrchestrator(self.model_manager)
        self.setup_agents()
        
        # 비동기 이벤트 루프
        self.loop = asyncio.new_event_loop()
        self.async_thread = threading.Thread(target=self.run_async_loop, daemon=True)
        self.async_thread.start()
        
        # UI 설정
        self.setup_modern_ui()
        
        # 시작 메시지
        self.add_system_message("🚀 에이전틱 AI 시스템이 시작되었습니다!")
        self.add_system_message("⚠️ GGUF 모델을 로드하면 더 강력한 AI 기능을 사용할 수 있습니다.")
        
    def create_directories(self):
        """필요 디렉토리 생성"""
        dirs = ["models", "agents", "tasks", "outputs", "memory", "tools", "configs"]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            
    def setup_agents(self):
        """에이전트 설정"""
        # 다양한 에이전트 생성
        agents_config = [
            ("주 조정자", AgentType.ORCHESTRATOR),
            ("계획 수립자", AgentType.PLANNER),
            ("실행자", AgentType.EXECUTOR),
            ("연구원", AgentType.RESEARCHER),
            ("분석가", AgentType.ANALYZER),
            ("창작자", AgentType.CREATOR),
            ("검토자", AgentType.REVIEWER)
        ]
        
        for name, agent_type in agents_config:
            agent = Agent(name, agent_type, self.model_manager)
            
            # 도구 할당
            if agent_type in [AgentType.RESEARCHER, AgentType.EXECUTOR]:
                agent.add_tool(WebSearchTool())
                agent.add_tool(FileSystemTool())
            
            if agent_type == AgentType.EXECUTOR:
                agent.add_tool(CodeExecutorTool())
                agent.add_tool(DatabaseTool())
            
            if agent_type == AgentType.ANALYZER:
                agent.add_tool(CodeExecutorTool())
                agent.add_tool(DatabaseTool())
            
            if agent_type == AgentType.CREATOR:
                agent.add_tool(FileSystemTool())
                
            self.orchestrator.add_agent(agent)
            
    def run_async_loop(self):
        """비동기 이벤트 루프 실행"""
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.orchestrator.process_tasks())
        
    def setup_modern_ui(self):
        """현대적 UI 구성"""
        # 테마 설정
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # 그리드 설정
        self.grid_columnconfigure(0, weight=0)  # 사이드바
        self.grid_columnconfigure(1, weight=1)  # 메인 영역
        self.grid_rowconfigure(0, weight=1)
        
        # 사이드바
        self.setup_sidebar()
        
        # 메인 영역
        self.setup_main_area()
        
    def setup_sidebar(self):
        """사이드바 구성"""
        sidebar = ctk.CTkFrame(self, width=280, corner_radius=0, fg_color=MODERN_THEME['dark'])
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_rowconfigure(7, weight=1)
        
        # 로고 영역
        logo_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        logo_frame.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")
        
        ctk.CTkLabel(
            logo_frame,
            text="🤖 Agentic AI",
            font=("SF Pro Display", 28, "bold"),
            text_color=MODERN_THEME['light']
        ).pack()
        
        ctk.CTkLabel(
            logo_frame,
            text="GGUF Model Powered System",
            font=("SF Pro Text", 12),
            text_color=MODERN_THEME['text_secondary']
        ).pack()
        
        # 모델 상태
        self.model_status_frame = ctk.CTkFrame(sidebar, fg_color="#2D3748", corner_radius=10)
        self.model_status_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        
        ctk.CTkLabel(
            self.model_status_frame,
            text="GGUF Model Status",
            font=("SF Pro Display", 14, "bold"),
            text_color=MODERN_THEME['light']
        ).pack(pady=(10, 5))
        
        self.model_status_label = ctk.CTkLabel(
            self.model_status_frame,
            text="❌ No Model Loaded",
            font=("SF Pro Text", 12),
            text_color=MODERN_THEME['error']
        )
        self.model_status_label.pack(pady=(0, 5))
        
        self.model_name_label = ctk.CTkLabel(
            self.model_status_frame,
            text="",
            font=("SF Pro Text", 10),
            text_color=MODERN_THEME['text_secondary']
        )
        self.model_name_label.pack(pady=(0, 10))
        
        # 모델 로드 버튼
        self.load_model_button = ctk.CTkButton(
            sidebar,
            text="📂 Load GGUF Model",
            command=self.load_model,
            font=("SF Pro Display", 13, "bold"),
            height=40,
            corner_radius=8,
            fg_color=MODERN_THEME['accent'],
            hover_color="#D946A6"
        )
        self.load_model_button.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        
        # 에이전트 상태
        self.agent_status_frame = ctk.CTkFrame(sidebar, fg_color=MODERN_THEME['surface'], corner_radius=10)
        self.agent_status_frame.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        
        ctk.CTkLabel(
            self.agent_status_frame,
            text="Active Agents",
            font=("SF Pro Display", 14, "bold"),
            text_color=MODERN_THEME['text']
        ).pack(pady=(10, 5))
        
        # 에이전트 리스트
        for agent_name, agent in self.orchestrator.agents.items():
            agent_item = ctk.CTkFrame(self.agent_status_frame, fg_color="transparent")
            agent_item.pack(fill="x", padx=10, pady=2)
            
            status_color = MODERN_THEME['success']
            ctk.CTkLabel(
                agent_item,
                text="●",
                font=("SF Pro Text", 12),
                text_color=status_color,
                width=20
            ).pack(side="left")
            
            ctk.CTkLabel(
                agent_item,
                text=agent_name,
                font=("SF Pro Text", 12),
                text_color=MODERN_THEME['text_secondary']
            ).pack(side="left", padx=(5, 0))
        
        # 작업 통계
        stats_frame = ctk.CTkFrame(sidebar, fg_color=MODERN_THEME['surface'], corner_radius=10)
        stats_frame.grid(row=4, column=0, padx=20, pady=10, sticky="ew")
        
        ctk.CTkLabel(
            stats_frame,
            text="Task Statistics",
            font=("SF Pro Display", 14, "bold"),
            text_color=MODERN_THEME['text']
        ).pack(pady=(10, 5))
        
        self.stats_labels = {}
        stats = [
            ("Pending", "pending", MODERN_THEME['warning']),
            ("In Progress", "progress", MODERN_THEME['info']),
            ("Completed", "completed", MODERN_THEME['success']),
            ("Failed", "failed", MODERN_THEME['error'])
        ]
        
        for stat_name, stat_key, color in stats:
            stat_item = ctk.CTkFrame(stats_frame, fg_color="transparent")
            stat_item.pack(fill="x", padx=15, pady=2)
            
            ctk.CTkLabel(
                stat_item,
                text=stat_name,
                font=("SF Pro Text", 11),
                text_color=MODERN_THEME['text_secondary']
            ).pack(side="left")
            
            label = ctk.CTkLabel(
                stat_item,
                text="0",
                font=("SF Pro Display", 12, "bold"),
                text_color=color
            )
            label.pack(side="right")
            self.stats_labels[stat_key] = label
        
        # 액션 버튼들
        action_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        action_frame.grid(row=5, column=0, padx=20, pady=10, sticky="ew")
        
        ctk.CTkButton(
            action_frame,
            text="➕ New Task",
            command=self.show_new_task_dialog,
            font=("SF Pro Display", 13, "bold"),
            height=40,
            corner_radius=8,
            fg_color=MODERN_THEME['primary'],
            hover_color=MODERN_THEME['hover']
        ).pack(fill="x", pady=5)
        
        ctk.CTkButton(
            action_frame,
            text="⚙️ Model Settings",
            command=self.show_model_settings,
            font=("SF Pro Display", 13, "bold"),
            height=40,
            corner_radius=8,
            fg_color=MODERN_THEME['secondary'],
            hover_color=MODERN_THEME['hover']
        ).pack(fill="x", pady=5)
        
        # 하단 정보
        info_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        info_frame.grid(row=8, column=0, padx=20, pady=(0, 20), sticky="ew")
        
        ctk.CTkLabel(
            info_frame,
            text="Powered by llama-cpp-python",
            font=("SF Pro Text", 10),
            text_color=MODERN_THEME['text_secondary']
        ).pack()
        
    def setup_main_area(self):
        """메인 영역 구성"""
        main_frame = ctk.CTkFrame(self, fg_color=MODERN_THEME['light'])
        main_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 0), pady=0)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)
        
        # 헤더
        header_frame = ctk.CTkFrame(main_frame, height=80, fg_color=MODERN_THEME['surface'], corner_radius=0)
        header_frame.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        header_frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(
            header_frame,
            text="Task Console",
            font=("SF Pro Display", 24, "bold"),
            text_color=MODERN_THEME['text']
        ).grid(row=0, column=0, padx=30, pady=25)
        
        # 상태 표시
        self.status_label = ctk.CTkLabel(
            header_frame,
            text="🟢 System Online",
            font=("SF Pro Text", 14),
            text_color=MODERN_THEME['success']
        )
        self.status_label.grid(row=0, column=1, padx=30, pady=25, sticky="e")
        
        # 탭 뷰
        self.tabview = ctk.CTkTabview(main_frame, corner_radius=10)
        self.tabview.grid(row=1, column=0, sticky="nsew", padx=20, pady=20)
        
        # 대화 탭
        chat_tab = self.tabview.add("💬 Chat")
        self.setup_chat_tab(chat_tab)
        
        # 작업 모니터 탭
        monitor_tab = self.tabview.add("📊 Task Monitor")
        self.setup_monitor_tab(monitor_tab)
        
        # 메모리 뷰 탭
        memory_tab = self.tabview.add("🧠 Agent Memory")
        self.setup_memory_tab(memory_tab)
        
    def setup_chat_tab(self, parent):
        """대화 탭 설정"""
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)
        
        # 대화 영역
        self.chat_frame = ctk.CTkScrollableFrame(
            parent,
            fg_color=MODERN_THEME['surface'],
            corner_radius=10
        )
        self.chat_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # 입력 영역
        input_frame = ctk.CTkFrame(parent, fg_color="transparent", height=100)
        input_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        input_frame.grid_columnconfigure(0, weight=1)
        
        self.input_box = ctk.CTkTextbox(
            input_frame,
            height=80,
            wrap="word",
            font=("SF Pro Text", 14),
            border_width=2,
            border_color=MODERN_THEME['border'],
            fg_color=MODERN_THEME['surface'],
            corner_radius=10
        )
        self.input_box.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.input_box.bind("<Return>", self.handle_return)
        
        # 전송 버튼
        self.send_button = ctk.CTkButton(
            input_frame,
            text="Send Task",
            command=self.send_task,
            width=120,
            height=80,
            font=("SF Pro Display", 14, "bold"),
            corner_radius=10,
            fg_color=MODERN_THEME['primary'],
            hover_color=MODERN_THEME['hover']
        )
        self.send_button.grid(row=0, column=1)
        
    def setup_monitor_tab(self, parent):
        """작업 모니터 탭 설정"""
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)
        
        # 작업 리스트
        self.task_tree = ttk.Treeview(
            parent,
            columns=('ID', 'Title', 'Agent', 'Status', 'Priority', 'Created'),
            show='headings',
            height=20
        )
        
        # 컬럼 설정
        columns = [
            ('ID', 150),
            ('Title', 300),
            ('Agent', 120),
            ('Status', 100),
            ('Priority', 80),
            ('Created', 150)
        ]
        
        for col, width in columns:
            self.task_tree.heading(col, text=col)
            self.task_tree.column(col, width=width)
        
        self.task_tree.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # 스크롤바
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=self.task_tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns", pady=10)
        self.task_tree.configure(yscrollcommand=scrollbar.set)
        
    def setup_memory_tab(self, parent):
        """메모리 뷰 탭 설정"""
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)
        
        self.memory_text = ctk.CTkTextbox(
            parent,
            font=("SF Pro Mono", 12),
            fg_color=MODERN_THEME['dark'],
            text_color=MODERN_THEME['light']
        )
        self.memory_text.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
    def load_model(self):
        """GGUF 모델 로드"""
        model_file = filedialog.askopenfilename(
            title="GGUF 모델 파일 선택",
            filetypes=[("GGUF 파일", "*.gguf"), ("모든 파일", "*.*")],
            initialdir="models"
        )
        
        if not model_file:
            return
        
        # 로딩 대화상자
        loading_dialog = self.create_loading_dialog("GGUF 모델 로드 중...")
        
        # 별도 스레드에서 모델 로드
        threading.Thread(
            target=self._load_model_thread,
            args=(model_file, loading_dialog),
            daemon=True
        ).start()
    
    def _load_model_thread(self, model_file, loading_dialog):
        """모델 로드 스레드"""
        try:
            # 모델 로드 전 파일 검증
            if not os.path.exists(model_file):
                raise FileNotFoundError(f"파일을 찾을 수 없습니다: {model_file}")
            
            file_size_mb = os.path.getsize(model_file) / (1024 * 1024)
            logger.info(f"모델 파일 크기: {file_size_mb:.2f} MB")
            
            # 메모리 체크 (psutil이 있는 경우만)
            try:
                import psutil
                available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # GB
                logger.info(f"사용 가능한 메모리: {available_memory:.2f} GB")
                
                # 파일 크기가 사용 가능한 메모리보다 크면 경고
                if file_size_mb > available_memory * 1024 * 0.8:  # 80% 이상 사용 시 경고
                    logger.warning("메모리 부족 가능성이 있습니다. 더 작은 모델을 사용하는 것을 권장합니다.")
            except ImportError:
                logger.info("psutil이 설치되지 않아 메모리 체크를 건너뜁니다.")
            
            # 모델 로드 시도
            self.model_manager.load_model(
                model_file,
                context_size=2048,
                n_threads=4,
                n_gpu_layers=0  # 기본적으로 GPU 비활성화
            )
            
            # UI 업데이트
            self.after(100, lambda: self._handle_model_load_success(model_file, loading_dialog))
            
        except FileNotFoundError as e:
            error_msg = f"파일 오류: {str(e)}"
            logger.error(error_msg)
            self.after(100, lambda msg=error_msg: self._handle_model_load_error(msg, loading_dialog))
            
        except MemoryError:
            error_msg = "메모리 부족: 시스템 메모리가 부족합니다. 더 작은 모델을 사용해주세요."
            logger.error(error_msg)
            self.after(100, lambda msg=error_msg: self._handle_model_load_error(msg, loading_dialog))
            
        except ImportError as e:
            error_msg = f"라이브러리 오류: {str(e)}\nllama-cpp-python을 다시 설치해보세요."
            logger.error(error_msg)
            self.after(100, lambda msg=error_msg: self._handle_model_load_error(msg, loading_dialog))
            
        except RuntimeError as e:
            # GGUFModelManager에서 발생한 RuntimeError
            error_msg = str(e)
            logger.error(error_msg)
            self.after(100, lambda msg=error_msg: self._handle_model_load_error(msg, loading_dialog))
            
        except Exception as e:
            # 기타 모든 예외 처리
            error_msg = f"예상치 못한 오류: {str(e)}\n타입: {type(e).__name__}"
            logger.error(error_msg)
            logger.error(f"상세 오류:\n{traceback.format_exc()}")
            
            # Access Violation 같은 치명적 오류 처리
            if "access violation" in str(e).lower():
                error_msg += "\n\n해결 방법:\n"
                error_msg += "1. 모델 파일이 손상되지 않았는지 확인\n"
                error_msg += "2. llama-cpp-python 재설치: pip uninstall llama-cpp-python && pip install llama-cpp-python\n"
                error_msg += "3. Visual C++ Redistributable 설치 확인\n"
                error_msg += "4. 다른 GGUF 모델 파일로 시도"
            
            self.after(100, lambda msg=error_msg: self._handle_model_load_error(msg, loading_dialog))
    
    def _handle_model_load_success(self, model_file, loading_dialog):
        """모델 로드 성공 처리"""
        loading_dialog.destroy()
        
        model_name = os.path.basename(model_file)
        self.model_status_label.configure(
            text="✅ Model Loaded",
            text_color=MODERN_THEME['success']
        )
        self.model_name_label.configure(text=model_name[:30] + "..." if len(model_name) > 30 else model_name)
        
        self.add_system_message(f"✅ GGUF 모델이 성공적으로 로드되었습니다: {model_name}")
        self.add_system_message("🚀 이제 AI 에이전트들이 강화된 기능을 사용할 수 있습니다!")
        
        messagebox.showinfo("성공", f"모델이 성공적으로 로드되었습니다:\n{model_name}")
    
    def _handle_model_load_error(self, error_msg, loading_dialog):
        """모델 로드 실패 처리"""
        loading_dialog.destroy()
        
        self.add_system_message(f"❌ 모델 로드 실패: {error_msg}")
        messagebox.showerror("오류", f"모델 로드에 실패했습니다:\n{error_msg}")
    
    def create_loading_dialog(self, message):
        """로딩 대화상자 생성"""
        dialog = ctk.CTkToplevel(self)
        dialog.title("처리 중")
        dialog.geometry("300x150")
        dialog.transient(self)
        dialog.grab_set()
        
        # 중앙 배치
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry(f'+{x}+{y}')
        
        ctk.CTkLabel(
            dialog,
            text=message,
            font=("SF Pro Display", 14, "bold")
        ).pack(pady=(20, 10))
        
        progress = ctk.CTkProgressBar(dialog, width=250)
        progress.pack(pady=(0, 20))
        progress.configure(mode="indeterminate")
        progress.start()
        
        return dialog
    
    def show_model_settings(self):
        """모델 설정 대화상자"""
        dialog = ctk.CTkToplevel(self)
        dialog.title("Model Settings")
        dialog.geometry("500x600")
        dialog.transient(self)
        dialog.grab_set()
        
        # 중앙 배치
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry(f'+{x}+{y}')
        
        # 설정 프레임
        settings_frame = ctk.CTkScrollableFrame(dialog)
        settings_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        ctk.CTkLabel(
            settings_frame,
            text="GGUF Model Settings",
            font=("SF Pro Display", 20, "bold")
        ).pack(pady=(0, 20))
        
        # 설정 항목들
        settings = [
            ("Context Size", "context_size", self.model_manager.context_size, 512, 8192),
            ("Temperature", "temperature", self.model_manager.temperature, 0.0, 2.0),
            ("Max Tokens", "max_tokens", self.model_manager.max_tokens, 100, 4000),
            ("Top P", "top_p", self.model_manager.top_p, 0.0, 1.0),
            ("Top K", "top_k", self.model_manager.top_k, 1, 100),
            ("Repeat Penalty", "repeat_penalty", self.model_manager.repeat_penalty, 0.0, 2.0),
            ("Threads", "n_threads", self.model_manager.n_threads, 1, 16),
            ("GPU Layers", "n_gpu_layers", self.model_manager.n_gpu_layers, 0, 100),
        ]
        
        self.setting_vars = {}
        
        for label, key, value, min_val, max_val in settings:
            frame = ctk.CTkFrame(settings_frame, fg_color="transparent")
            frame.pack(fill="x", pady=10)
            
            ctk.CTkLabel(
                frame,
                text=label,
                font=("SF Pro Text", 14),
                width=150,
                anchor="w"
            ).pack(side="left")
            
            if key in ["temperature", "top_p", "repeat_penalty"]:
                slider = ctk.CTkSlider(
                    frame,
                    from_=min_val,
                    to=max_val,
                    width=200
                )
                slider.set(value)
                slider.pack(side="left", padx=(10, 10))
                
                value_label = ctk.CTkLabel(
                    frame,
                    text=f"{value:.2f}",
                    font=("SF Pro Mono", 12),
                    width=60
                )
                value_label.pack(side="left")
                
                slider.configure(command=lambda v, l=value_label: l.configure(text=f"{v:.2f}"))
                self.setting_vars[key] = slider
            else:
                entry = ctk.CTkEntry(
                    frame,
                    width=100,
                    font=("SF Pro Mono", 12)
                )
                entry.insert(0, str(value))
                entry.pack(side="left", padx=(10, 0))
                self.setting_vars[key] = entry
        
        # 버튼
        button_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        button_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        ctk.CTkButton(
            button_frame,
            text="Cancel",
            command=dialog.destroy,
            width=100,
            fg_color=MODERN_THEME['error']
        ).pack(side="left", padx=10)
        
        ctk.CTkButton(
            button_frame,
            text="Apply",
            command=lambda: self.apply_model_settings(dialog),
            width=100,
            fg_color=MODERN_THEME['success']
        ).pack(side="right", padx=10)
    
    def apply_model_settings(self, dialog):
        """모델 설정 적용"""
        try:
            for key, widget in self.setting_vars.items():
                if isinstance(widget, ctk.CTkSlider):
                    value = widget.get()
                else:
                    value = widget.get()
                    if key in ["context_size", "max_tokens", "top_k", "n_threads", "n_gpu_layers"]:
                        value = int(value)
                    else:
                        value = float(value)
                
                setattr(self.model_manager, key, value)
            
            self.add_system_message("✅ 모델 설정이 업데이트되었습니다.")
            dialog.destroy()
            
        except ValueError as e:
            messagebox.showerror("오류", "잘못된 값이 입력되었습니다.")
    
    def add_system_message(self, message):
        """시스템 메시지 추가"""
        container = ctk.CTkFrame(self.chat_frame, fg_color=MODERN_THEME['info'], corner_radius=10)
        container.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(
            container,
            text=message,
            font=("SF Pro Text", 13),
            text_color="white",
            wraplength=800,
            justify="left"
        ).pack(padx=15, pady=10, anchor="w")
        
        # 스크롤 다운
        self.chat_frame._parent_canvas.yview_moveto(1.0)
        
    def add_user_message(self, message):
        """사용자 메시지 추가"""
        container = ctk.CTkFrame(self.chat_frame, fg_color="#D6EAF8", corner_radius=10)
        container.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(
            container,
            text=f"👤 User",
            font=("SF Pro Display", 12, "bold"),
            text_color=MODERN_THEME['primary']
        ).pack(padx=15, pady=(10, 5), anchor="w")
        
        ctk.CTkLabel(
            container,
            text=message,
            font=("SF Pro Text", 13),
            text_color=MODERN_THEME['text'],
            wraplength=800,
            justify="left"
        ).pack(padx=15, pady=(0, 10), anchor="w")
        
        # 스크롤 다운
        self.chat_frame._parent_canvas.yview_moveto(1.0)
        
    def add_agent_message(self, agent_name, message):
        """에이전트 메시지 추가"""
        container = ctk.CTkFrame(self.chat_frame, fg_color="#E0E7FF", corner_radius=10)
        container.pack(fill="x", padx=10, pady=5)
        
        # 에이전트 이름
        ctk.CTkLabel(
            container,
            text=f"🤖 {agent_name}",
            font=("SF Pro Display", 12, "bold"),
            text_color=MODERN_THEME['primary']
        ).pack(padx=15, pady=(10, 5), anchor="w")
        
        # 메시지
        ctk.CTkLabel(
            container,
            text=message,
            font=("SF Pro Text", 13),
            text_color=MODERN_THEME['text'],
            wraplength=800,
            justify="left"
        ).pack(padx=15, pady=(0, 10), anchor="w")
        
        # 스크롤 다운
        self.chat_frame._parent_canvas.yview_moveto(1.0)
        
    def send_task(self):
        """작업 전송"""
        task_description = self.input_box.get("0.0", "end").strip()
        
        if not task_description:
            return
            
        # 입력 초기화
        self.input_box.delete("0.0", "end")
        
        # 사용자 메시지 표시
        self.add_user_message(task_description)
        
        # 작업 유형 자동 결정
        agent_type = self.determine_agent_type(task_description)
        
        # 작업 생성
        task = Task(
            id=f"TASK-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            title=task_description[:50] + "..." if len(task_description) > 50 else task_description,
            description=task_description,
            agent_type=agent_type,
            priority=5
        )
        
        # 작업 제출
        self.orchestrator.submit_task(task)
        
        # UI 업데이트
        self.add_system_message(f"📋 작업이 {agent_type.value} 에이전트에게 할당되었습니다.")
        self.update_task_monitor(task)
        self.update_stats()
        
        # 비동기로 작업 처리 및 결과 표시
        threading.Thread(
            target=self._process_task_async,
            args=(task,),
            daemon=True
        ).start()
    
    def determine_agent_type(self, description: str) -> AgentType:
        """작업 설명을 기반으로 적절한 에이전트 타입 결정"""
        desc_lower = description.lower()
        
        if any(word in desc_lower for word in ["계획", "plan", "전략", "strategy"]):
            return AgentType.PLANNER
        elif any(word in desc_lower for word in ["검색", "search", "찾", "find", "정보"]):
            return AgentType.RESEARCHER
        elif any(word in desc_lower for word in ["분석", "analyze", "통계", "데이터"]):
            return AgentType.ANALYZER
        elif any(word in desc_lower for word in ["만들", "create", "작성", "write", "생성"]):
            return AgentType.CREATOR
        elif any(word in desc_lower for word in ["검토", "review", "확인", "check"]):
            return AgentType.REVIEWER
        else:
            return AgentType.EXECUTOR
    
    def _process_task_async(self, task):
        """비동기 작업 처리"""
        # 작업이 처리될 때까지 대기
        max_wait = 30  # 최대 30초 대기
        wait_time = 0
        
        while wait_time < max_wait:
            if task.status == TaskStatus.COMPLETED:
                # 결과 표시
                agent = self.orchestrator.select_agent(task)
                if agent and task.result:
                    self.after(100, lambda: self.add_agent_message(agent.name, str(task.result)))
                break
            elif task.status == TaskStatus.FAILED:
                self.after(100, lambda: self.add_system_message(f"❌ 작업 실패: {task.error}"))
                break
            
            time.sleep(0.5)
            wait_time += 0.5
        
        # 통계 업데이트
        self.after(100, self.update_stats)
        
    def update_task_monitor(self, task):
        """작업 모니터 업데이트"""
        self.task_tree.insert('', 'end', values=(
            task.id,
            task.title,
            task.agent_type.value,
            task.status.value,
            task.priority,
            task.created_at.strftime('%Y-%m-%d %H:%M')
        ))
        
    def update_stats(self):
        """통계 업데이트"""
        # 실제 통계 계산
        stats = {
            'pending': sum(1 for t in self.orchestrator.task_queue.queue if t.status == TaskStatus.PENDING),
            'progress': 0,  # 현재 처리 중인 작업
            'completed': len(self.orchestrator.completed_tasks),
            'failed': sum(1 for t in self.orchestrator.completed_tasks if t.status == TaskStatus.FAILED)
        }
        
        for key, value in stats.items():
            if key in self.stats_labels:
                self.stats_labels[key].configure(text=str(value))
                
    def show_new_task_dialog(self):
        """새 작업 대화상자"""
        dialog = ctk.CTkToplevel(self)
        dialog.title("New Task")
        dialog.geometry("600x500")
        dialog.transient(self)
        dialog.grab_set()
        
        # 중앙 배치
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry(f'+{x}+{y}')
        
        # 내용
        ctk.CTkLabel(
            dialog,
            text="Create New Task",
            font=("SF Pro Display", 20, "bold")
        ).pack(pady=(20, 10))
        
        # 작업 제목
        ctk.CTkLabel(dialog, text="Task Title:").pack(pady=(10, 5))
        title_entry = ctk.CTkEntry(dialog, width=400)
        title_entry.pack(pady=(0, 10))
        
        # 작업 설명
        ctk.CTkLabel(dialog, text="Description:").pack(pady=(10, 5))
        desc_text = ctk.CTkTextbox(dialog, width=400, height=150)
        desc_text.pack(pady=(0, 10))
        
        # 에이전트 선택
        ctk.CTkLabel(dialog, text="Agent Type:").pack(pady=(10, 5))
        agent_var = ctk.StringVar(value=AgentType.EXECUTOR.value)
        agent_menu = ctk.CTkComboBox(
            dialog,
            values=[t.value for t in AgentType],
            variable=agent_var,
            width=400
        )
        agent_menu.pack(pady=(0, 10))
        
        # 우선순위
        ctk.CTkLabel(dialog, text="Priority (1-10):").pack(pady=(10, 5))
        priority_slider = ctk.CTkSlider(dialog, from_=1, to=10, width=400)
        priority_slider.set(5)
        priority_slider.pack(pady=(0, 20))
        
        # 버튼
        button_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        button_frame.pack(pady=20)
        
        ctk.CTkButton(
            button_frame,
            text="Cancel",
            command=dialog.destroy,
            width=100,
            fg_color=MODERN_THEME['error']
        ).pack(side="left", padx=10)
        
        ctk.CTkButton(
            button_frame,
            text="Create",
            command=lambda: self.create_task_from_dialog(
                dialog,
                title_entry.get(),
                desc_text.get("0.0", "end"),
                agent_var.get(),
                int(priority_slider.get())
            ),
            width=100,
            fg_color=MODERN_THEME['success']
        ).pack(side="left", padx=10)
        
    def create_task_from_dialog(self, dialog, title, description, agent_type, priority):
        """대화상자에서 작업 생성"""
        if not title or not description:
            messagebox.showwarning("Warning", "Please fill in all fields")
            return
            
        task = Task(
            id=f"TASK-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            title=title,
            description=description.strip(),
            agent_type=AgentType(agent_type),
            priority=priority
        )
        
        self.orchestrator.submit_task(task)
        self.add_system_message(f"📋 새 작업 생성: {task.title}")
        self.update_task_monitor(task)
        self.update_stats()
        
        dialog.destroy()
        
    def handle_return(self, event):
        """Enter 키 처리"""
        if event.state & 0x1:  # Shift key
            return None
        else:
            self.send_task()
            return "break"
            
    def on_closing(self):
        """앱 종료 처리"""
        self.orchestrator.stop()
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.quit()
        self.destroy()

# ============= 메인 실행 =============
if __name__ == "__main__":
    if not LLAMA_CPP_AVAILABLE:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "오류", 
            "llama-cpp-python이 설치되지 않았습니다.\n\n"
            "다음 명령어로 설치해주세요:\n"
            "pip install llama-cpp-python"
        )
        root.destroy()
        sys.exit(1)
    
    try:
        app = AgenticAIApp()
        app.protocol("WM_DELETE_WINDOW", app.on_closing)
        app.mainloop()
    except Exception as e:
        error_message = f"애플리케이션 시작 중 오류가 발생했습니다:\n{str(e)}\n\n{traceback.format_exc()}"
        print(error_message)
        
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("오류", error_message)
            root.destroy()
        except:
            pass
        
        sys.exit(1)