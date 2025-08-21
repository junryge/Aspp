# -*- coding: utf-8 -*-
"""
GGUF 대화 시스템 - llama-cpp-python 버전
Phi-4 모델 지원, 대화 관리 강화
"""

import os
import sys
import json
import threading
import customtkinter as ctk
from tkinter import filedialog, messagebox
import logging
from datetime import datetime
from typing import Dict, List, Optional

# llama-cpp-python import
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print("llama-cpp-python이 설치되지 않았습니다.")
    print("설치 명령어:")
    print("CPU: pip install llama-cpp-python")
    print("GPU (CUDA): CMAKE_ARGS=\"-DLLAMA_CUBLAS=on\" pip install llama-cpp-python")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 전역 테마 설정
THEME = {
    'primary': '#2E86C1',
    'secondary': '#AED6F1',
    'error': '#E74C3C',
    'success': '#2ECC71',
    'surface': '#FFFFFF',
    'text': '#2C3E50'
}

class ConversationManager:
    """대화 관리 클래스"""
    
    def __init__(self, max_history: int = 10):
        self.conversations: List[Dict] = []
        self.max_history = max_history
        self.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def add_message(self, role: str, content: str):
        """메시지 추가"""
        self.conversations.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "session_id": self.current_session_id
        })
        
        # 최대 이력 제한
        if len(self.conversations) > self.max_history * 2:
            self.conversations = self.conversations[-self.max_history * 2:]
    
    def get_context(self, num_messages: int = 6) -> List[Dict]:
        """최근 대화 컨텍스트 반환"""
        return self.conversations[-num_messages:] if len(self.conversations) > num_messages else self.conversations
    
    def clear(self):
        """대화 초기화"""
        self.conversations = []
        self.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def save_to_file(self, filepath: str):
        """대화 내용 저장"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.conversations, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"대화 저장 실패: {e}")
            return False
    
    def load_from_file(self, filepath: str):
        """대화 내용 불러오기"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.conversations = json.load(f)
            return True
        except Exception as e:
            logger.error(f"대화 불러오기 실패: {e}")
            return False

class SimpleConversationApp(ctk.CTk):
    """llama-cpp-python 기반 대화 애플리케이션"""
    
    def __init__(self):
        super().__init__()
        
        # 앱 초기화
        self.title("GGUF 대화 시스템 - Phi-4 지원")
        self.geometry("1000x750")
        
        # 테마 설정
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")
        
        # 모델 초기화
        self.model = None
        self.model_path = ""
        self.model_name = ""
        
        # 대화 관리자
        self.conversation_manager = ConversationManager(max_history=20)
        
        # 시스템 프롬프트
        self.system_prompt = "You are a helpful AI assistant."
        
        # 응답 생성 플래그
        self.is_generating = False
        self.stop_generation = False
        
        # UI 구성
        self.setup_ui()
        
    def setup_ui(self):
        """UI 구성"""
        # 메인