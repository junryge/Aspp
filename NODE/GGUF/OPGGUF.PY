# -*- coding: utf-8 -*-
"""
에이전틱 AI 시스템 - 자율 작업 수행 가능한 AI 어시스턴트
Multi-Agent Architecture with Tool Usage
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
        """웹 검색 실행 (시뮬레이션)"""
        await asyncio.sleep(1)  # 실제로는 API 호출
        return f"검색 결과: {query}에 대한 정보를 찾았습니다."

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
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return f"파일 저장 완료: {path}"
            except Exception as e:
                return f"파일 쓰기 실패: {str(e)}"
        return "지원하지 않는 작업입니다."

class CodeExecutorTool(Tool):
    """코드 실행 도구"""
    
    def __init__(self):
        super().__init__(
            name="code_executor",
            description="Python 코드를 실행합니다"
        )
    
    async def execute(self, code: str) -> str:
        """코드 실행 (보안상 제한적으로)"""
        try:
            # 실제 구현시에는 샌드박스 환경 필요
            result = eval(code) if len(code) < 100 else "코드가 너무 깁니다"
            return str(result)
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
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT,
                status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            )
        ''')
        self.conn.commit()
    
    async def execute(self, query: str) -> str:
        """SQL 쿼리 실행"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(query)
            if query.strip().upper().startswith('SELECT'):
                return str(cursor.fetchall())
            else:
                self.conn.commit()
                return f"쿼리 실행 완료: {cursor.rowcount}개 행 영향"
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

# ============= 에이전트 기본 클래스 =============
class Agent:
    """에이전트 기본 클래스"""
    
    def __init__(self, name: str, agent_type: AgentType, model=None):
        self.name = name
        self.agent_type = agent_type
        self.model = model
        self.tools: Dict[str, Tool] = {}
        self.memory: List[Dict] = []
        self.current_task: Optional[Task] = None
        
    def add_tool(self, tool: Tool):
        """도구 추가"""
        self.tools[tool.name] = tool
        
    async def process_task(self, task: Task) -> Any:
        """작업 처리"""
        self.current_task = task
        task.status = TaskStatus.IN_PROGRESS
        
        try:
            # 작업 계획 수립
            plan = await self.plan_task(task)
            
            # 작업 실행
            result = await self.execute_plan(plan)
            
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result
            
            # 메모리에 저장
            self.memory.append({
                'task': task.title,
                'result': result,
                'timestamp': datetime.now()
            })
            
            return result
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            logger.error(f"Agent {self.name} failed: {str(e)}")
            raise
            
    async def plan_task(self, task: Task) -> List[Dict]:
        """작업 계획 수립"""
        # 모델을 사용하여 작업 계획 생성
        if self.model:
            prompt = f"""
            작업: {task.title}
            설명: {task.description}
            
            이 작업을 수행하기 위한 단계별 계획을 수립하세요.
            사용 가능한 도구: {', '.join(self.tools.keys())}
            """
            
            # 실제로는 모델 호출
            plan = [
                {'step': '정보 수집', 'tool': 'web_search'},
                {'step': '분석', 'tool': None},
                {'step': '결과 생성', 'tool': 'file_system'}
            ]
        else:
            plan = [{'step': '기본 실행', 'tool': None}]
            
        return plan
        
    async def execute_plan(self, plan: List[Dict]) -> Any:
        """계획 실행"""
        results = []
        
        for step in plan:
            if step.get('tool'):
                tool = self.tools.get(step['tool'])
                if tool:
                    result = await tool.execute()
                    results.append(result)
            else:
                # 도구 없이 실행
                results.append(f"Step completed: {step.get('step', 'Unknown')}")
                
        return results

# ============= 멀티 에이전트 오케스트레이터 =============
class AgentOrchestrator:
    """멀티 에이전트 시스템 조정자"""
    
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.task_queue: queue.Queue = queue.Queue()
        self.completed_tasks: List[Task] = []
        self.running = False
        
    def add_agent(self, agent: Agent):
        """에이전트 추가"""
        self.agents[agent.name] = agent
        logger.info(f"Agent added: {agent.name} ({agent.agent_type.value})")
        
    def submit_task(self, task: Task):
        """작업 제출"""
        self.task_queue.put(task)
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
        for agent in self.agents.values():
            if agent.agent_type == task.agent_type:
                return agent
        return None
        
    def stop(self):
        """오케스트레이터 중지"""
        self.running = False

# ============= 메인 애플리케이션 UI =============
class AgenticAIApp(ctk.CTk):
    """에이전틱 AI 시스템 UI"""
    
    def __init__(self):
        super().__init__()
        
        # 앱 초기화
        self.title("🤖 Agentic AI System - 자율 AI 어시스턴트")
        self.geometry("1400x900")
        
        # 디렉토리 생성
        self.create_directories()
        
        # 시스템 초기화
        self.orchestrator = AgentOrchestrator()
        self.setup_agents()
        
        # 비동기 이벤트 루프
        self.loop = asyncio.new_event_loop()
        self.async_thread = threading.Thread(target=self.run_async_loop, daemon=True)
        self.async_thread.start()
        
        # UI 설정
        self.setup_modern_ui()
        
        # 시작 메시지
        self.add_system_message("🚀 에이전틱 AI 시스템이 시작되었습니다!")
        self.add_system_message("📋 작업을 입력하면 AI 에이전트들이 자동으로 처리합니다.")
        
    def create_directories(self):
        """필요 디렉토리 생성"""
        dirs = ["agents", "tasks", "outputs", "memory", "tools"]
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
            agent = Agent(name, agent_type)
            
            # 도구 할당
            if agent_type in [AgentType.RESEARCHER, AgentType.EXECUTOR]:
                agent.add_tool(WebSearchTool())
                agent.add_tool(FileSystemTool())
            
            if agent_type == AgentType.EXECUTOR:
                agent.add_tool(CodeExecutorTool())
                agent.add_tool(DatabaseTool())
                
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
        sidebar.grid_rowconfigure(6, weight=1)
        
        # 로고 영역
        logo_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        logo_frame.grid(row=0, column=0, padx=20, pady=(20, 30), sticky="ew")
        
        ctk.CTkLabel(
            logo_frame,
            text="🤖 Agentic AI",
            font=("SF Pro Display", 28, "bold"),
            text_color=MODERN_THEME['light']
        ).pack()
        
        ctk.CTkLabel(
            logo_frame,
            text="Autonomous Multi-Agent System",
            font=("SF Pro Text", 12),
            text_color=MODERN_THEME['text_secondary']
        ).pack()
        
        # 에이전트 상태
        self.agent_status_frame = ctk.CTkFrame(sidebar, fg_color=MODERN_THEME['surface'], corner_radius=10)
        self.agent_status_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        
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
        stats_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        
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
        
        # 도구 섹션
        tools_frame = ctk.CTkFrame(sidebar, fg_color=MODERN_THEME['surface'], corner_radius=10)
        tools_frame.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        
        ctk.CTkLabel(
            tools_frame,
            text="Available Tools",
            font=("SF Pro Display", 14, "bold"),
            text_color=MODERN_THEME['text']
        ).pack(pady=(10, 5))
        
        tools = ["🔍 Web Search", "📁 File System", "💻 Code Executor", "🗄️ Database"]
        for tool in tools:
            ctk.CTkLabel(
                tools_frame,
                text=tool,
                font=("SF Pro Text", 11),
                text_color=MODERN_THEME['text_secondary']
            ).pack(padx=15, pady=2, anchor="w")
        
        # 액션 버튼들
        action_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        action_frame.grid(row=4, column=0, padx=20, pady=20, sticky="ew")
        
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
            text="📊 View Reports",
            command=self.show_reports,
            font=("SF Pro Display", 13, "bold"),
            height=40,
            corner_radius=8,
            fg_color=MODERN_THEME['secondary'],
            hover_color=MODERN_THEME['hover']
        ).pack(fill="x", pady=5)
        
        # 설정 버튼
        ctk.CTkButton(
            sidebar,
            text="⚙️ Settings",
            command=self.show_settings,
            font=("SF Pro Display", 12),
            height=35,
            corner_radius=8,
            fg_color="transparent",
            hover_color=MODERN_THEME['hover'],
            text_color=MODERN_THEME['text_secondary']
        ).grid(row=7, column=0, padx=20, pady=(0, 20), sticky="ew")
        
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
            ('ID', 80),
            ('Title', 250),
            ('Agent', 150),
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
        
    def send_task(self):
        """작업 전송"""
        task_description = self.input_box.get("0.0", "end").strip()
        
        if not task_description:
            return
            
        # 입력 초기화
        self.input_box.delete("0.0", "end")
        
        # 작업 생성
        task = Task(
            id=f"TASK-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            title=task_description[:50],
            description=task_description,
            agent_type=AgentType.EXECUTOR,
            priority=5
        )
        
        # 작업 제출
        self.orchestrator.submit_task(task)
        
        # UI 업데이트
        self.add_system_message(f"📋 새 작업 제출: {task.title}")
        self.update_task_monitor(task)
        self.update_stats()
        
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
        
    def show_reports(self):
        """리포트 보기"""
        messagebox.showinfo("Reports", "작업 리포트 기능은 개발 중입니다.")
        
    def show_settings(self):
        """설정 보기"""
        messagebox.showinfo("Settings", "설정 기능은 개발 중입니다.")
        
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