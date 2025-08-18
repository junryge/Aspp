# -*- coding: utf-8 -*-
"""
완전한 GGUF 오케스트레이터 LLM 시스템
로컬 환경에서 다중 AI 모델을 지능적으로 관리하고 조율하는 시스템
"""

import os
import sys
import json
import time
import threading
import queue
import psutil
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
from pathlib import Path
import logging
from datetime import datetime
import traceback
import customtkinter as ctk
from tkinter import filedialog, messagebox, ttk
import tkinter as tk

# llama-cpp-python import
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print("llama-cpp-python 설치 필요: pip install llama-cpp-python")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("orchestrator.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== 시스템 상수 ====================
MAX_CONTEXT_LENGTH = 4096
DEFAULT_MAX_TOKENS = 1024
MEMORY_SAFETY_MARGIN = 0.8  # 메모리의 80%만 사용

# ==================== 1. 에이전트 타입 및 역할 정의 ====================
class AgentRole(Enum):
    """에이전트 역할 정의"""
    ROUTER = "router"              # 작업 분류 및 라우팅
    ANALYZER = "analyzer"          # 입력 분석
    CODER = "coder"               # 코드 생성
    DEBUGGER = "debugger"         # 코드 디버깅
    WRITER = "writer"             # 문서 작성
    TRANSLATOR = "translator"     # 번역
    SUMMARIZER = "summarizer"     # 요약
    VALIDATOR = "validator"       # 검증
    CREATIVE = "creative"         # 창작
    MATH = "math"                 # 수학/계산
    DATA_ANALYST = "data_analyst" # 데이터 분석
    GENERAL = "general"           # 일반 대화

class TaskPriority(Enum):
    """작업 우선순위"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4

class TaskStatus(Enum):
    """작업 상태"""
    PENDING = "pending"
    ANALYZING = "analyzing"
    ROUTING = "routing"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

# ==================== 2. 시스템 모니터링 ====================
class SystemMonitor:
    """시스템 리소스 모니터링"""
    
    @staticmethod
    def get_memory_info() -> Dict[str, float]:
        """메모리 정보 가져오기"""
        memory = psutil.virtual_memory()
        return {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "percent": memory.percent
        }
    
    @staticmethod
    def get_cpu_info() -> Dict[str, float]:
        """CPU 정보 가져오기"""
        return {
            "percent": psutil.cpu_percent(interval=1),
            "cores": psutil.cpu_count(logical=False),
            "threads": psutil.cpu_count(logical=True)
        }
    
    @staticmethod
    def can_load_model(model_size_gb: float) -> bool:
        """모델 로드 가능 여부 확인"""
        memory_info = SystemMonitor.get_memory_info()
        available = memory_info["available_gb"]
        required = model_size_gb * 1.2  # 20% 여유 공간
        return available > required

# ==================== 3. 작업 정의 ====================
@dataclass
class Task:
    """작업 단위"""
    id: str
    content: str
    role: Optional[AgentRole] = None
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    result: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    assigned_agent: Optional[str] = None
    
    def __lt__(self, other):
        """우선순위 비교"""
        return self.priority.value < other.priority.value

# ==================== 4. 에이전트 정의 ====================
@dataclass
class AgentProfile:
    """에이전트 프로필"""
    name: str
    role: AgentRole
    model_path: str
    model_size_gb: float
    context_size: int = 2048
    temperature: float = 0.7
    max_tokens: int = 1024
    system_prompt: str = ""
    capabilities: List[str] = field(default_factory=list)
    performance_score: float = 1.0  # 성능 점수 (0-1)
    total_tasks: int = 0
    successful_tasks: int = 0
    average_response_time: float = 0.0

class Agent:
    """지능형 에이전트"""
    
    def __init__(self, profile: AgentProfile):
        self.profile = profile
        self.model: Optional[Llama] = None
        self.is_loaded = False
        self.is_busy = False
        self.current_task: Optional[Task] = None
        self.history: List[Task] = []
        self.lock = threading.Lock()
        
    def load(self) -> bool:
        """모델 로드"""
        if self.is_loaded:
            return True
            
        if not SystemMonitor.can_load_model(self.profile.model_size_gb):
            logger.warning(f"메모리 부족으로 {self.profile.name} 로드 불가")
            return False
            
        try:
            logger.info(f"에이전트 '{self.profile.name}' 로드 시작...")
            self.model = Llama(
                model_path=self.profile.model_path,
                n_ctx=self.profile.context_size,
                n_threads=psutil.cpu_count(logical=False),
                n_gpu_layers=-1 if self._check_gpu() else 0,
                verbose=False
            )
            self.is_loaded = True
            logger.info(f"에이전트 '{self.profile.name}' 로드 완료")
            return True
        except Exception as e:
            logger.error(f"에이전트 로드 실패: {str(e)}")
            return False
    
    def unload(self):
        """모델 언로드"""
        if self.model:
            del self.model
            self.model = None
            self.is_loaded = False
            logger.info(f"에이전트 '{self.profile.name}' 언로드")
    
    def _check_gpu(self) -> bool:
        """GPU 사용 가능 여부 확인"""
        # 간단한 GPU 체크 (실제로는 더 정교한 확인 필요)
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def can_handle(self, task: Task) -> float:
        """작업 처리 가능 여부 및 적합도 반환 (0-1)"""
        if not self.is_loaded or self.is_busy:
            return 0.0
            
        # 역할 매칭
        if task.role and task.role != self.profile.role:
            return 0.0
            
        # 적합도 계산
        suitability = self.profile.performance_score
        
        # 작업 내용 분석 (키워드 기반)
        content_lower = task.content.lower()
        role_keywords = {
            AgentRole.CODER: ["code", "코드", "function", "함수", "class", "클래스"],
            AgentRole.TRANSLATOR: ["translate", "번역", "translation"],
            AgentRole.SUMMARIZER: ["summarize", "요약", "summary"],
            AgentRole.MATH: ["calculate", "계산", "math", "수학"],
            AgentRole.WRITER: ["write", "작성", "document", "문서"],
        }
        
        if self.profile.role in role_keywords:
            keywords = role_keywords[self.profile.role]
            if any(keyword in content_lower for keyword in keywords):
                suitability *= 1.5  # 키워드 매칭시 보너스
                
        return min(suitability, 1.0)
    
    def process(self, task: Task) -> str:
        """작업 처리"""
        with self.lock:
            if not self.is_loaded or self.is_busy:
                raise RuntimeError(f"에이전트 {self.profile.name}이 사용 불가능")
                
            self.is_busy = True
            self.current_task = task
            
        try:
            start_time = time.time()
            
            # 프롬프트 구성
            prompt = self._build_prompt(task)
            
            # 모델 실행
            response = self.model(
                prompt,
                max_tokens=self.profile.max_tokens,
                temperature=self.profile.temperature,
                top_p=0.95,
                repeat_penalty=1.1,
                echo=False
            )
            
            # 응답 추출
            if isinstance(response, dict) and "choices" in response:
                result = response["choices"][0]["text"].strip()
            else:
                result = str(response).strip()
            
            # 성능 통계 업데이트
            elapsed_time = time.time() - start_time
            self._update_performance(task, elapsed_time, success=True)
            
            return result
            
        except Exception as e:
            logger.error(f"작업 처리 실패: {str(e)}")
            self._update_performance(task, 0, success=False)
            raise
            
        finally:
            with self.lock:
                self.is_busy = False
                self.current_task = None
                self.history.append(task)
    
    def _build_prompt(self, task: Task) -> str:
        """프롬프트 구성"""
        parts = []
        
        # 시스템 프롬프트
        if self.profile.system_prompt:
            parts.append(f"System: {self.profile.system_prompt}")
        
        # 작업 컨텍스트
        if task.metadata.get("context"):
            parts.append(f"Context: {task.metadata['context']}")
        
        # 이전 결과 참조
        if task.dependencies:
            parts.append("Previous Results:")
            for dep_id in task.dependencies:
                if dep_id in task.metadata.get("dependency_results", {}):
                    parts.append(f"- {dep_id}: {task.metadata['dependency_results'][dep_id][:200]}...")
        
        # 메인 작업
        parts.append(f"User: {task.content}")
        parts.append("Assistant:")
        
        return "\n\n".join(parts)
    
    def _update_performance(self, task: Task, elapsed_time: float, success: bool):
        """성능 지표 업데이트"""
        self.profile.total_tasks += 1
        if success:
            self.profile.successful_tasks += 1
            
        # 평균 응답 시간 업데이트
        alpha = 0.1  # 지수 이동 평균 가중치
        if elapsed_time > 0:
            if self.profile.average_response_time == 0:
                self.profile.average_response_time = elapsed_time
            else:
                self.profile.average_response_time = (
                    alpha * elapsed_time + 
                    (1 - alpha) * self.profile.average_response_time
                )
        
        # 성능 점수 업데이트
        if self.profile.total_tasks > 0:
            success_rate = self.profile.successful_tasks / self.profile.total_tasks
            speed_score = min(1.0, 10.0 / max(self.profile.average_response_time, 0.1))
            self.profile.performance_score = (success_rate * 0.7 + speed_score * 0.3)

# ==================== 5. 지능형 라우터 ====================
class IntelligentRouter:
    """작업을 분석하고 적절한 에이전트로 라우팅"""
    
    def __init__(self):
        self.routing_rules = self._init_routing_rules()
        self.routing_history = []
        
    def _init_routing_rules(self) -> Dict[str, AgentRole]:
        """라우팅 규칙 초기화"""
        return {
            # 코딩 관련
            "code": AgentRole.CODER,
            "코드": AgentRole.CODER,
            "function": AgentRole.CODER,
            "함수": AgentRole.CODER,
            "debug": AgentRole.DEBUGGER,
            "디버그": AgentRole.DEBUGGER,
            "error": AgentRole.DEBUGGER,
            "에러": AgentRole.DEBUGGER,
            
            # 번역
            "translate": AgentRole.TRANSLATOR,
            "번역": AgentRole.TRANSLATOR,
            "translation": AgentRole.TRANSLATOR,
            
            # 요약
            "summarize": AgentRole.SUMMARIZER,
            "요약": AgentRole.SUMMARIZER,
            "summary": AgentRole.SUMMARIZER,
            
            # 수학
            "calculate": AgentRole.MATH,
            "계산": AgentRole.MATH,
            "math": AgentRole.MATH,
            "수학": AgentRole.MATH,
            
            # 데이터 분석
            "analyze": AgentRole.DATA_ANALYST,
            "분석": AgentRole.DATA_ANALYST,
            "data": AgentRole.DATA_ANALYST,
            "데이터": AgentRole.DATA_ANALYST,
            
            # 작성
            "write": AgentRole.WRITER,
            "작성": AgentRole.WRITER,
            "document": AgentRole.WRITER,
            "문서": AgentRole.WRITER,
        }
    
    def analyze_task(self, task: Task) -> AgentRole:
        """작업 분석 및 적절한 역할 결정"""
        content_lower = task.content.lower()
        
        # 키워드 기반 라우팅
        for keyword, role in self.routing_rules.items():
            if keyword in content_lower:
                logger.info(f"작업 '{task.id}'를 {role.value} 역할로 라우팅")
                return role
        
        # 기본값
        return AgentRole.GENERAL
    
    def route(self, task: Task, agents: Dict[str, Agent]) -> Optional[Agent]:
        """작업을 최적의 에이전트로 라우팅"""
        
        # 작업 역할 분석
        if not task.role:
            task.role = self.analyze_task(task)
        
        # 최적 에이전트 선택
        best_agent = None
        best_score = 0.0
        
        for agent in agents.values():
            score = agent.can_handle(task)
            if score > best_score:
                best_score = score
                best_agent = agent
        
        if best_agent:
            logger.info(f"작업 '{task.id}'를 에이전트 '{best_agent.profile.name}'로 할당 (점수: {best_score:.2f})")
            self.routing_history.append({
                "task_id": task.id,
                "agent": best_agent.profile.name,
                "score": best_score,
                "timestamp": datetime.now()
            })
            
        return best_agent

# ==================== 6. 파이프라인 매니저 ====================
class Pipeline:
    """복잡한 작업을 단계별로 처리하는 파이프라인"""
    
    def __init__(self, name: str):
        self.name = name
        self.stages: List[Dict[str, Any]] = []
        self.results: Dict[str, Any] = {}
        
    def add_stage(self, name: str, role: AgentRole, 
                  prompt_template: str, **kwargs):
        """파이프라인 단계 추가"""
        stage = {
            "name": name,
            "role": role,
            "prompt_template": prompt_template,
            "kwargs": kwargs
        }
        self.stages.append(stage)
        logger.info(f"파이프라인 '{self.name}'에 단계 '{name}' 추가")
        
    def execute(self, orchestrator, initial_input: str) -> Dict[str, Any]:
        """파이프라인 실행"""
        current_input = initial_input
        
        for i, stage in enumerate(self.stages):
            logger.info(f"파이프라인 단계 {i+1}/{len(self.stages)}: {stage['name']}")
            
            # 프롬프트 생성
            prompt = stage["prompt_template"].format(
                input=current_input,
                **self.results
            )
            
            # 작업 생성
            task = Task(
                id=f"{self.name}_{stage['name']}_{int(time.time())}",
                content=prompt,
                role=stage["role"],
                priority=TaskPriority.HIGH,
                metadata=stage.get("kwargs", {})
            )
            
            # 의존성 설정
            if i > 0:
                task.dependencies = [prev_task_id]
                task.metadata["dependency_results"] = {prev_task_id: current_input}
            
            # 작업 실행
            result = orchestrator.execute_task_sync(task)
            
            if result:
                self.results[stage["name"]] = result
                current_input = result
                prev_task_id = task.id
            else:
                raise RuntimeError(f"파이프라인 단계 실패: {stage['name']}")
        
        return self.results

# ==================== 7. 메인 오케스트레이터 ====================
class Orchestrator:
    """중앙 오케스트레이터 - 모든 에이전트와 작업을 관리"""
    
    def __init__(self, max_workers: int = 3):
        self.agents: Dict[str, Agent] = {}
        self.router = IntelligentRouter()
        self.task_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.completed_tasks: Dict[str, Task] = {}
        self.active_tasks: Dict[str, Task] = {}
        self.max_workers = max_workers
        self.workers: List[threading.Thread] = []
        self.running = False
        self.lock = threading.Lock()
        
        # 콜백
        self.on_task_complete: Optional[Callable] = None
        self.on_task_error: Optional[Callable] = None
        self.on_status_update: Optional[Callable] = None
        
        # 통계
        self.stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "average_completion_time": 0.0
        }
        
    def register_agent(self, agent: Agent) -> bool:
        """에이전트 등록"""
        if agent.load():
            self.agents[agent.profile.name] = agent
            logger.info(f"에이전트 '{agent.profile.name}' 등록 완료")
            return True
        return False
    
    def unregister_agent(self, agent_name: str):
        """에이전트 등록 해제"""
        if agent_name in self.agents:
            self.agents[agent_name].unload()
            del self.agents[agent_name]
            logger.info(f"에이전트 '{agent_name}' 등록 해제")
    
    def submit_task(self, task: Task) -> str:
        """작업 제출"""
        if not task.id:
            task.id = f"task_{int(time.time() * 1000)}"
        
        with self.lock:
            self.task_queue.put((task.priority.value, time.time(), task))
            self.active_tasks[task.id] = task
            self.stats["total_tasks"] += 1
            
        logger.info(f"작업 제출: {task.id}")
        
        if self.on_status_update:
            self.on_status_update(f"작업 제출: {task.id}")
            
        return task.id
    
    def execute_task_sync(self, task: Task, timeout: float = 60) -> Optional[str]:
        """동기적으로 작업 실행 (파이프라인용)"""
        task_id = self.submit_task(task)
        
        # 작업 완료 대기
        start_time = time.time()
        while time.time() - start_time < timeout:
            if task_id in self.completed_tasks:
                completed_task = self.completed_tasks[task_id]
                if completed_task.status == TaskStatus.COMPLETED:
                    return completed_task.result
                else:
                    return None
            time.sleep(0.1)
            
        return None
    
    def start(self):
        """오케스트레이터 시작"""
        if self.running:
            return
            
        self.running = True
        
        # 워커 스레드 생성
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"Worker-{i+1}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
            
        logger.info(f"오케스트레이터 시작 (워커: {self.max_workers}개)")
        
        if self.on_status_update:
            self.on_status_update(f"오케스트레이터 시작")
    
    def stop(self):
        """오케스트레이터 중지"""
        self.running = False
        
        for worker in self.workers:
            worker.join(timeout=5)
            
        self.workers.clear()
        logger.info("오케스트레이터 중지")
        
        if self.on_status_update:
            self.on_status_update("오케스트레이터 중지")
    
    def _worker_loop(self):
        """워커 루프"""
        while self.running:
            try:
                # 작업 가져오기
                priority, timestamp, task = self.task_queue.get(timeout=1)
                
                # 작업 처리
                self._process_task(task)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"워커 오류: {str(e)}")
    
    def _process_task(self, task: Task):
        """작업 처리"""
        try:
            task.status = TaskStatus.ROUTING
            task.started_at = datetime.now()
            
            # 에이전트 선택
            agent = self.router.route(task, self.agents)
            
            if not agent:
                raise RuntimeError("적합한 에이전트를 찾을 수 없음")
            
            task.assigned_agent = agent.profile.name
            task.status = TaskStatus.PROCESSING
            
            if self.on_status_update:
                self.on_status_update(f"작업 처리 중: {task.id} → {agent.profile.name}")
            
            # 작업 실행
            result = agent.process(task)
            
            # 작업 완료
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            
            with self.lock:
                del self.active_tasks[task.id]
                self.completed_tasks[task.id] = task
                self.stats["completed_tasks"] += 1
                
                # 평균 완료 시간 업데이트
                if task.started_at and task.completed_at:
                    completion_time = (task.completed_at - task.started_at).total_seconds()
                    alpha = 0.1
                    if self.stats["average_completion_time"] == 0:
                        self.stats["average_completion_time"] = completion_time
                    else:
                        self.stats["average_completion_time"] = (
                            alpha * completion_time + 
                            (1 - alpha) * self.stats["average_completion_time"]
                        )
            
            logger.info(f"작업 완료: {task.id}")
            
            if self.on_task_complete:
                self.on_task_complete(task)
                
        except Exception as e:
            # 작업 실패
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()
            
            with self.lock:
                if task.id in self.active_tasks:
                    del self.active_tasks[task.id]
                self.completed_tasks[task.id] = task
                self.stats["failed_tasks"] += 1
            
            logger.error(f"작업 실패: {task.id} - {str(e)}")
            
            if self.on_task_error:
                self.on_task_error(task, e)
    
    def get_statistics(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        with self.lock:
            stats = self.stats.copy()
            stats["active_tasks"] = len(self.active_tasks)
            stats["queued_tasks"] = self.task_queue.qsize()
            stats["agents"] = {
                name: {
                    "role": agent.profile.role.value,
                    "loaded": agent.is_loaded,
                    "busy": agent.is_busy,
                    "performance": agent.profile.performance_score,
                    "tasks_completed": agent.profile.successful_tasks
                }
                for name, agent in self.agents.items()
            }
            return stats

# ==================== 8. 고급 UI ====================
class OrchestratorUI(ctk.CTk):
    """오케스트레이터 고급 UI"""
    
    def __init__(self):
        super().__init__()
        
        self.title("GGUF 오케스트레이터 LLM 시스템")
        self.geometry("1400x900")
        
        # 테마 설정
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # 오케스트레이터 초기화
        self.orchestrator = Orchestrator(max_workers=3)
        self.orchestrator.on_task_complete = self.on_task_complete
        self.orchestrator.on_task_error = self.on_task_error
        self.orchestrator.on_status_update = self.on_status_update
        
        # 파이프라인
        self.pipelines: Dict[str, Pipeline] = {}
        
        # UI 구성
        self.setup_ui()
        
        # 시스템 모니터링 시작
        self.start_monitoring()
        
        # 오케스트레이터 시작
        self.orchestrator.start()
        
    def setup_ui(self):
        """UI 구성"""
        # 메인 컨테이너
        main_container = ctk.CTkFrame(self)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 상단 정보 바
        self.setup_info_bar(main_container)
        
        # 메인 콘텐츠 영역
        content_frame = ctk.CTkFrame(main_container)
        content_frame.pack(fill="both", expand=True, pady=(10, 0))
        
        # 3단 레이아웃
        # 좌측: 에이전트 관리
        left_frame = ctk.CTkFrame(content_frame, width=350)
        left_frame.pack(side="left", fill="y", padx=(0, 5))
        left_frame.pack_propagate(False)
        self.setup_agent_panel(left_frame)
        
        # 중앙: 작업 관리
        center_frame = ctk.CTkFrame(content_frame)
        center_frame.pack(side="left", fill="both", expand=True, padx=5)
        self.setup_task_panel(center_frame)
        
        # 우측: 모니터링
        right_frame = ctk.CTkFrame(content_frame, width=350)
        right_frame.pack(side="right", fill="y", padx=(5, 0))
        right_frame.pack_propagate(False)
        self.setup_monitoring_panel(right_frame)
        
    def setup_info_bar(self, parent):
        """상단 정보 바"""
        info_bar = ctk.CTkFrame(parent, height=80)
        info_bar.pack(fill="x")
        info_bar.pack_propagate(False)
        
        # 시스템 상태
        status_frame = ctk.CTkFrame(info_bar)
        status_frame.pack(side="left", padx=20, pady=10)
        
        ctk.CTkLabel(status_frame, text="시스템 상태", 
                     font=("Helvetica", 14, "bold")).pack()
        self.status_label = ctk.CTkLabel(status_frame, text="준비", 
                                         font=("Helvetica", 12))
        self.status_label.pack()
        
        # 메모리 정보
        memory_frame = ctk.CTkFrame(info_bar)
        memory_frame.pack(side="left", padx=20, pady=10)
        
        ctk.CTkLabel(memory_frame, text="메모리", 
                     font=("Helvetica", 14, "bold")).pack()
        self.memory_label = ctk.CTkLabel(memory_frame, text="0.0 GB / 0.0 GB",
                                        font=("Helvetica", 12))
        self.memory_label.pack()
        
        # CPU 정보
        cpu_frame = ctk.CTkFrame(info_bar)
        cpu_frame.pack(side="left", padx=20, pady=10)
        
        ctk.CTkLabel(cpu_frame, text="CPU", 
                     font=("Helvetica", 14, "bold")).pack()
        self.cpu_label = ctk.CTkLabel(cpu_frame, text="0%",
                                     font=("Helvetica", 12))
        self.cpu_label.pack()
        
        # 작업 통계
        stats_frame = ctk.CTkFrame(info_bar)
        stats_frame.pack(side="left", padx=20, pady=10)
        
        ctk.CTkLabel(stats_frame, text="작업 통계", 
                     font=("Helvetica", 14, "bold")).pack()
        self.stats_label = ctk.CTkLabel(stats_frame, text="완료: 0 / 실패: 0",
                                       font=("Helvetica", 12))
        self.stats_label.pack()
        
    def setup_agent_panel(self, parent):
        """에이전트 관리 패널"""
        # 제목
        ctk.CTkLabel(parent, text="에이전트 관리", 
                     font=("Helvetica", 18, "bold")).pack(pady=10)
        
        # 에이전트 추가 버튼
        add_agent_btn = ctk.CTkButton(
            parent, 
            text="+ 에이전트 추가",
            command=self.show_add_agent_dialog,
            height=35
        )
        add_agent_btn.pack(padx=10, pady=5)
        
        # 에이전트 목록
        ctk.CTkLabel(parent, text="활성 에이전트:", 
                     font=("Helvetica", 14)).pack(anchor="w", padx=10, pady=(10, 5))
        
        self.agent_listbox = ctk.CTkTextbox(parent, height=300)
        self.agent_listbox.pack(fill="both", expand=True, padx=10, pady=5)
        
        # 빠른 에이전트 추가 (예제)
        quick_add_frame = ctk.CTkFrame(parent)
        quick_add_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(quick_add_frame, text="빠른 추가:", 
                     font=("Helvetica", 12)).pack(anchor="w")
        
        ctk.CTkButton(
            quick_add_frame,
            text="일반 대화 에이전트",
            command=lambda: self.quick_add_agent("general"),
            width=140,
            height=28
        ).pack(side="left", padx=2, pady=2)
        
        ctk.CTkButton(
            quick_add_frame,
            text="코딩 에이전트",
            command=lambda: self.quick_add_agent("coder"),
            width=140,
            height=28
        ).pack(side="left", padx=2, pady=2)
        
    def setup_task_panel(self, parent):
        """작업 관리 패널"""
        # 제목
        ctk.CTkLabel(parent, text="작업 관리", 
                     font=("Helvetica", 18, "bold")).pack(pady=10)
        
        # 탭뷰
        tabview = ctk.CTkTabview(parent)
        tabview.pack(fill="both", expand=True, padx=10, pady=5)
        
        # 단일 작업 탭
        single_tab = tabview.add("단일 작업")
        self.setup_single_task_tab(single_tab)
        
        # 파이프라인 탭
        pipeline_tab = tabview.add("파이프라인")
        self.setup_pipeline_tab(pipeline_tab)
        
        # 작업 큐 탭
        queue_tab = tabview.add("작업 큐")
        self.setup_queue_tab(queue_tab)
        
    def setup_single_task_tab(self, parent):
        """단일 작업 탭"""
        # 작업 타입 선택
        type_frame = ctk.CTkFrame(parent)
        type_frame.pack(fill="x", pady=5)
        
        ctk.CTkLabel(type_frame, text="작업 타입:").pack(side="left", padx=5)
        self.task_type_var = ctk.StringVar(value=AgentRole.GENERAL.value)
        task_type_menu = ctk.CTkOptionMenu(
            type_frame,
            variable=self.task_type_var,
            values=[role.value for role in AgentRole]
        )
        task_type_menu.pack(side="left", padx=5)
        
        # 우선순위 선택
        ctk.CTkLabel(type_frame, text="우선순위:").pack(side="left", padx=(20, 5))
        self.priority_var = ctk.StringVar(value=TaskPriority.NORMAL.name)
        priority_menu = ctk.CTkOptionMenu(
            type_frame,
            variable=self.priority_var,
            values=[p.name for p in TaskPriority]
        )
        priority_menu.pack(side="left", padx=5)
        
        # 프롬프트 입력
        ctk.CTkLabel(parent, text="프롬프트:").pack(anchor="w", pady=(10, 5))
        self.prompt_input = ctk.CTkTextbox(parent, height=150)
        self.prompt_input.pack(fill="x", pady=5)
        
        # 제출 버튼
        submit_btn = ctk.CTkButton(
            parent,
            text="작업 제출",
            command=self.submit_single_task,
            height=40,
            font=("Helvetica", 14, "bold")
        )
        submit_btn.pack(pady=10)
        
        # 결과 표시
        ctk.CTkLabel(parent, text="결과:").pack(anchor="w", pady=(10, 5))
        self.result_box = ctk.CTkTextbox(parent, height=200)
        self.result_box.pack(fill="both", expand=True, pady=5)
        
    def setup_pipeline_tab(self, parent):
        """파이프라인 탭"""
        # 파이프라인 선택
        pipeline_frame = ctk.CTkFrame(parent)
        pipeline_frame.pack(fill="x", pady=5)
        
        ctk.CTkLabel(pipeline_frame, text="파이프라인:").pack(side="left", padx=5)
        self.pipeline_var = ctk.StringVar(value="새 파이프라인")
        self.pipeline_menu = ctk.CTkOptionMenu(
            pipeline_frame,
            variable=self.pipeline_var,
            values=["새 파이프라인", "코드 리뷰", "번역 파이프라인"]
        )
        self.pipeline_menu.pack(side="left", padx=5)
        
        ctk.CTkButton(
            pipeline_frame,
            text="파이프라인 생성",
            command=self.create_pipeline,
            width=120
        ).pack(side="left", padx=10)
        
        # 파이프라인 단계
        ctk.CTkLabel(parent, text="파이프라인 단계:").pack(anchor="w", pady=(10, 5))
        self.pipeline_stages_box = ctk.CTkTextbox(parent, height=150)
        self.pipeline_stages_box.pack(fill="x", pady=5)
        
        # 입력
        ctk.CTkLabel(parent, text="입력:").pack(anchor="w", pady=(10, 5))
        self.pipeline_input = ctk.CTkTextbox(parent, height=100)
        self.pipeline_input.pack(fill="x", pady=5)
        
        # 실행 버튼
        execute_btn = ctk.CTkButton(
            parent,
            text="파이프라인 실행",
            command=self.execute_pipeline,
            height=40,
            font=("Helvetica", 14, "bold")
        )
        execute_btn.pack(pady=10)
        
        # 파이프라인 결과
        ctk.CTkLabel(parent, text="파이프라인 결과:").pack(anchor="w", pady=(10, 5))
        self.pipeline_result_box = ctk.CTkTextbox(parent, height=150)
        self.pipeline_result_box.pack(fill="both", expand=True, pady=5)
        
    def setup_queue_tab(self, parent):
        """작업 큐 탭"""
        # 큐 상태
        ctk.CTkLabel(parent, text="작업 큐 상태", 
                     font=("Helvetica", 16, "bold")).pack(pady=10)
        
        # 활성 작업
        ctk.CTkLabel(parent, text="처리 중:").pack(anchor="w", pady=5)
        self.active_tasks_box = ctk.CTkTextbox(parent, height=150)
        self.active_tasks_box.pack(fill="x", pady=5)
        
        # 대기 중인 작업
        ctk.CTkLabel(parent, text="대기 중:").pack(anchor="w", pady=5)
        self.queued_tasks_box = ctk.CTkTextbox(parent, height=150)
        self.queued_tasks_box.pack(fill="x", pady=5)
        
        # 완료된 작업
        ctk.CTkLabel(parent, text="최근 완료:").pack(anchor="w", pady=5)
        self.completed_tasks_box = ctk.CTkTextbox(parent, height=150)
        self.completed_tasks_box.pack(fill="both", expand=True, pady=5)
        
    def setup_monitoring_panel(self, parent):
        """모니터링 패널"""
        # 제목
        ctk.CTkLabel(parent, text="시스템 모니터링", 
                     font=("Helvetica", 18, "bold")).pack(pady=10)
        
        # 실시간 로그
        ctk.CTkLabel(parent, text="실시간 로그:", 
                     font=("Helvetica", 14)).pack(anchor="w", padx=10, pady=5)
        
        self.log_box = ctk.CTkTextbox(parent, height=250)
        self.log_box.pack(fill="x", padx=10, pady=5)
        
        # 에이전트 성능
        ctk.CTkLabel(parent, text="에이전트 성능:", 
                     font=("Helvetica", 14)).pack(anchor="w", padx=10, pady=(10, 5))
        
        self.performance_box = ctk.CTkTextbox(parent, height=200)
        self.performance_box.pack(fill="x", padx=10, pady=5)
        
        # 제어 버튼
        control_frame = ctk.CTkFrame(parent)
        control_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkButton(
            control_frame,
            text="통계 초기화",
            command=self.reset_stats,
            width=100
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            control_frame,
            text="로그 지우기",
            command=lambda: self.log_box.delete("1.0", "end"),
            width=100
        ).pack(side="left", padx=5)
        
    def show_add_agent_dialog(self):
        """에이전트 추가 대화상자"""
        dialog = ctk.CTkToplevel(self)
        dialog.title("에이전트 추가")
        dialog.geometry("500x600")
        dialog.transient(self)
        dialog.grab_set()
        
        # 중앙 정렬
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (250)
        y = (dialog.winfo_screenheight() // 2) - (300)
        dialog.geometry(f"+{x}+{y}")
        
        # 입력 필드
        ctk.CTkLabel(dialog, text="에이전트 이름:").pack(pady=5)
        name_entry = ctk.CTkEntry(dialog, width=300)
        name_entry.pack(pady=5)
        
        ctk.CTkLabel(dialog, text="역할:").pack(pady=5)
        role_var = ctk.StringVar(value=AgentRole.GENERAL.value)
        role_menu = ctk.CTkOptionMenu(
            dialog,
            variable=role_var,
            values=[role.value for role in AgentRole],
            width=300
        )
        role_menu.pack(pady=5)
        
        ctk.CTkLabel(dialog, text="모델 파일:").pack(pady=5)
        model_path_var = ctk.StringVar()
        model_path_label = ctk.CTkLabel(dialog, text="선택되지 않음", 
                                       text_color="gray")
        model_path_label.pack(pady=5)
        
        def select_model():
            path = filedialog.askopenfilename(
                title="GGUF 모델 선택",
                filetypes=[("GGUF 파일", "*.gguf"), ("모든 파일", "*.*")]
            )
            if path:
                model_path_var.set(path)
                model_path_label.configure(text=os.path.basename(path))
        
        ctk.CTkButton(dialog, text="모델 파일 선택", 
                     command=select_model).pack(pady=5)
        
        ctk.CTkLabel(dialog, text="모델 크기 (GB):").pack(pady=5)
        size_entry = ctk.CTkEntry(dialog, width=300)
        size_entry.insert(0, "3.0")
        size_entry.pack(pady=5)
        
        ctk.CTkLabel(dialog, text="시스템 프롬프트:").pack(pady=5)
        prompt_text = ctk.CTkTextbox(dialog, width=300, height=100)
        prompt_text.pack(pady=5)
        
        # 고급 설정
        advanced_frame = ctk.CTkFrame(dialog)
        advanced_frame.pack(fill="x", padx=20, pady=10)
        
        ctk.CTkLabel(advanced_frame, text="컨텍스트 크기:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        context_entry = ctk.CTkEntry(advanced_frame, width=100)
        context_entry.insert(0, "2048")
        context_entry.grid(row=0, column=1, padx=5, pady=2)
        
        ctk.CTkLabel(advanced_frame, text="Temperature:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        temp_entry = ctk.CTkEntry(advanced_frame, width=100)
        temp_entry.insert(0, "0.7")
        temp_entry.grid(row=1, column=1, padx=5, pady=2)
        
        ctk.CTkLabel(advanced_frame, text="Max Tokens:").grid(row=2, column=0, padx=5, pady=2, sticky="w")
        tokens_entry = ctk.CTkEntry(advanced_frame, width=100)
        tokens_entry.insert(0, "1024")
        tokens_entry.grid(row=2, column=1, padx=5, pady=2)
        
        # 버튼
        button_frame = ctk.CTkFrame(dialog)
        button_frame.pack(pady=20)
        
        def add_agent():
            try:
                profile = AgentProfile(
                    name=name_entry.get(),
                    role=AgentRole(role_var.get()),
                    model_path=model_path_var.get(),
                    model_size_gb=float(size_entry.get()),
                    context_size=int(context_entry.get()),
                    temperature=float(temp_entry.get()),
                    max_tokens=int(tokens_entry.get()),
                    system_prompt=prompt_text.get("1.0", "end").strip()
                )
                
                agent = Agent(profile)
                if self.orchestrator.register_agent(agent):
                    self.update_agent_list()
                    dialog.destroy()
                    messagebox.showinfo("성공", f"에이전트 '{profile.name}' 추가 완료")
                else:
                    messagebox.showerror("오류", "에이전트 로드 실패")
                    
            except Exception as e:
                messagebox.showerror("오류", str(e))
        
        ctk.CTkButton(button_frame, text="추가", command=add_agent,
                     width=100).pack(side="left", padx=5)
        ctk.CTkButton(button_frame, text="취소", command=dialog.destroy,
                     width=100).pack(side="left", padx=5)
        
    def quick_add_agent(self, agent_type: str):
        """빠른 에이전트 추가"""
        # 예제 설정 (실제 사용시 수정 필요)
        configs = {
            "general": {
                "name": f"general_agent_{int(time.time())}",
                "role": AgentRole.GENERAL,
                "model_path": "models/llama-2-7b.gguf",  # 실제 경로로 변경
                "model_size_gb": 7.0,
                "system_prompt": "You are a helpful AI assistant."
            },
            "coder": {
                "name": f"coder_agent_{int(time.time())}",
                "role": AgentRole.CODER,
                "model_path": "models/codellama-7b.gguf",  # 실제 경로로 변경
                "model_size_gb": 7.0,
                "system_prompt": "You are an expert programmer."
            }
        }
        
        if agent_type in configs:
            try:
                config = configs[agent_type]
                profile = AgentProfile(**config)
                agent = Agent(profile)
                
                if self.orchestrator.register_agent(agent):
                    self.update_agent_list()
                    self.log(f"에이전트 '{profile.name}' 추가 완료")
                else:
                    messagebox.showerror("오류", "에이전트 로드 실패")
                    
            except Exception as e:
                messagebox.showerror("오류", str(e))
    
    def submit_single_task(self):
        """단일 작업 제출"""
        content = self.prompt_input.get("1.0", "end").strip()
        if not content:
            return
        
        task = Task(
            id="",
            content=content,
            role=AgentRole(self.task_type_var.get()),
            priority=TaskPriority[self.priority_var.get()]
        )
        
        task_id = self.orchestrator.submit_task(task)
        self.log(f"작업 제출: {task_id}")
        
        # 입력 초기화
        self.prompt_input.delete("1.0", "end")
    
    def create_pipeline(self):
        """파이프라인 생성"""
        name = self.pipeline_var.get()
        
        if name == "코드 리뷰":
            pipeline = Pipeline("code_review")
            pipeline.add_stage("analyze", AgentRole.ANALYZER, 
                             "Analyze this code: {input}")
            pipeline.add_stage("debug", AgentRole.DEBUGGER,
                             "Find potential bugs in: {input}")
            pipeline.add_stage("improve", AgentRole.CODER,
                             "Suggest improvements for: {input}")
            self.pipelines[name] = pipeline
            self.pipeline_stages_box.insert("1.0", 
                "1. 코드 분석 (Analyzer)\n2. 버그 검사 (Debugger)\n3. 개선 제안 (Coder)\n")
            
        elif name == "번역 파이프라인":
            pipeline = Pipeline("translation")
            pipeline.add_stage("translate", AgentRole.TRANSLATOR,
                             "Translate to English: {input}")
            pipeline.add_stage("validate", AgentRole.VALIDATOR,
                             "Validate translation: {input}")
            self.pipelines[name] = pipeline
            self.pipeline_stages_box.insert("1.0",
                "1. 번역 (Translator)\n2. 검증 (Validator)\n")
            
        self.log(f"파이프라인 '{name}' 생성 완료")
    
    def execute_pipeline(self):
        """파이프라인 실행"""
        pipeline_name = self.pipeline_var.get()
        if pipeline_name not in self.pipelines:
            messagebox.showwarning("경고", "먼저 파이프라인을 생성하세요")
            return
        
        input_text = self.pipeline_input.get("1.0", "end").strip()
        if not input_text:
            return
        
        pipeline = self.pipelines[pipeline_name]
        
        # 비동기 실행
        def run_pipeline():
            try:
                results = pipeline.execute(self.orchestrator, input_text)
                
                # 결과 표시
                result_text = "=== 파이프라인 실행 결과 ===\n\n"
                for stage_name, result in results.items():
                    result_text += f"[{stage_name}]\n{result}\n\n"
                
                self.after(0, lambda: self.pipeline_result_box.insert("1.0", result_text))
                
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("오류", str(e)))
        
        threading.Thread(target=run_pipeline, daemon=True).start()
        self.log(f"파이프라인 '{pipeline_name}' 실행 시작")
    
    def update_agent_list(self):
        """에이전트 목록 업데이트"""
        self.agent_listbox.delete("1.0", "end")
        
        for name, agent in self.orchestrator.agents.items():
            status = "🟢" if agent.is_loaded else "🔴"
            busy = "🔄" if agent.is_busy else "✅"
            info = f"{status} {name} ({agent.profile.role.value}) {busy}\n"
            info += f"  성능: {agent.profile.performance_score:.2f}\n"
            info += f"  작업: {agent.profile.successful_tasks}/{agent.profile.total_tasks}\n"
            info += "─" * 30 + "\n"
            self.agent_listbox.insert("end", info)
    
    def start_monitoring(self):
        """시스템 모니터링 시작"""
        def monitor():
            while True:
                try:
                    # 시스템 정보
                    memory_info = SystemMonitor.get_memory_info()
                    cpu_info = SystemMonitor.get_cpu_info()
                    
                    # 통계
                    stats = self.orchestrator.get_statistics()
                    
                    # UI 업데이트
                    self.after(0, lambda: self.update_monitoring(memory_info, cpu_info, stats))
                    
                    time.sleep(1)
                except:
                    pass
        
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
    
    def update_monitoring(self, memory_info, cpu_info, stats):
        """모니터링 정보 업데이트"""
        # 메모리
        self.memory_label.configure(
            text=f"{memory_info['used_gb']:.1f} / {memory_info['total_gb']:.1f} GB"
        )
        
        # CPU
        self.cpu_label.configure(text=f"{cpu_info['percent']:.1f}%")
        
        # 통계
        self.stats_label.configure(
            text=f"완료: {stats['completed_tasks']} / 실패: {stats['failed_tasks']}"
        )
        
        # 큐 상태
        self.active_tasks_box.delete("1.0", "end")
        for task_id, task in self.orchestrator.active_tasks.items():
            self.active_tasks_box.insert("end", 
                f"{task_id}: {task.status.value} ({task.assigned_agent})\n")
        
        # 에이전트 성능
        self.performance_box.delete("1.0", "end")
        for name, agent_stats in stats.get("agents", {}).items():
            perf_text = f"{name}:\n"
            perf_text += f"  성능: {agent_stats['performance']:.2f}\n"
            perf_text += f"  완료: {agent_stats['tasks_completed']}\n"
            self.performance_box.insert("end", perf_text)
    
    def on_task_complete(self, task: Task):
        """작업 완료 콜백"""
        self.after(0, lambda: self._handle_task_complete(task))
    
    def _handle_task_complete(self, task: Task):
        """작업 완료 처리"""
        # 결과 표시
        result_text = f"=== 작업 완료: {task.id} ===\n"
        result_text += f"에이전트: {task.assigned_agent}\n"
        result_text += f"소요 시간: {(task.completed_at - task.started_at).total_seconds():.2f}초\n"
        result_text += f"결과:\n{task.result}\n\n"
        
        self.result_box.insert("1.0", result_text)
        
        # 완료 목록 업데이트
        self.completed_tasks_box.insert("1.0", 
            f"{task.id} ({task.assigned_agent}) ✅\n")
        
        self.log(f"작업 완료: {task.id}")
        
        # 에이전트 목록 업데이트
        self.update_agent_list()
    
    def on_task_error(self, task: Task, error: Exception):
        """작업 오류 콜백"""
        self.after(0, lambda: self._handle_task_error(task, error))
    
    def _handle_task_error(self, task: Task, error: Exception):
        """작업 오류 처리"""
        self.log(f"작업 실패: {task.id} - {str(error)}", "ERROR")
        
        # 완료 목록에 실패 표시
        self.completed_tasks_box.insert("1.0", 
            f"{task.id} ({task.assigned_agent}) ❌\n")
    
    def on_status_update(self, message: str):
        """상태 업데이트"""
        self.after(0, lambda: self.status_label.configure(text=message))
    
    def log(self, message: str, level: str = "INFO"):
        """로그 메시지 추가"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {level}: {message}\n"
        self.log_box.insert("1.0", log_message)
        
        # 로거에도 기록
        if level == "ERROR":
            logger.error(message)
        else:
            logger.info(message)
    
    def reset_stats(self):
        """통계 초기화"""
        self.orchestrator.stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "average_completion_time": 0.0
        }
        
        for agent in self.orchestrator.agents.values():
            agent.profile.total_tasks = 0
            agent.profile.successful_tasks = 0
            agent.profile.average_response_time = 0.0
            agent.profile.performance_score = 1.0
        
        self.update_agent_list()
        self.log("통계 초기화 완료")
    
    def on_closing(self):
        """종료 처리"""
        self.orchestrator.stop()
        
        # 모든 에이전트 언로드
        for agent in self.orchestrator.agents.values():
            agent.unload()
        
        self.destroy()

# ==================== 9. 메인 실행 ====================
def main():
    """메인 함수"""
    if not LLAMA_CPP_AVAILABLE:
        print("오류: llama-cpp-python이 설치되지 않았습니다.")
        print("설치: pip install llama-cpp-python")
        return
    
    # 디렉토리 생성
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # UI 실행
    app = OrchestratorUI()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()

if __name__ == "__main__":
    main()