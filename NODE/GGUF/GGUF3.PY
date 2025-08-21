# -*- coding: utf-8 -*-
"""
GGUF 대화 시스템 - Phi-4 모델 지원 강화 버전
CTransformers 기반, 대화 관리 강화
"""

import os
import sys
import json
import threading
import customtkinter as ctk
from tkinter import filedialog, messagebox, ttk
import logging
from datetime import datetime
from typing import Dict, List, Optional

# ctransformers import
try:
    from ctransformers import AutoModelForCausalLM
    CTRANSFORMERS_AVAILABLE = True
except ImportError:
    CTRANSFORMERS_AVAILABLE = False
    print("ctransformers가 설치되지 않았습니다. 설치해주세요: pip install ctransformers")

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

# 모델 타입별 설정
MODEL_CONFIGS = {
    'phi': {
        'model_type': 'phi',
        'context_length': 4096,
        'max_new_tokens': 1024,
        'temperature': 0.7,
        'top_p': 0.95,
        'repetition_penalty': 1.1,
        'prompt_template': "<|system|>\n{system}<|end|>\n<|user|>\n{user}<|end|>\n<|assistant|>\n"
    },
    'llama': {
        'model_type': 'llama',
        'context_length': 2048,
        'max_new_tokens': 512,
        'temperature': 0.7,
        'top_p': 0.95,
        'repetition_penalty': 1.1,
        'prompt_template': "### System:\n{system}\n\n### User:\n{user}\n\n### Assistant:\n"
    },
    'mistral': {
        'model_type': 'mistral',
        'context_length': 4096,
        'max_new_tokens': 1024,
        'temperature': 0.7,
        'top_p': 0.95,
        'repetition_penalty': 1.1,
        'prompt_template': "[INST] {system}\n{user} [/INST]"
    }
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
    """CTransformers 기반 대화 애플리케이션 - Phi-4 지원"""
    
    def __init__(self):
        super().__init__()
        
        # 앱 초기화
        self.title("GGUF 대화 시스템 - Phi-4 지원 강화 버전")
        self.geometry("1000x750")
        
        # 테마 설정
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")
        
        # 모델 초기화
        self.model = None
        self.model_path = ""
        self.model_type = "llama"  # 기본값
        self.model_config = MODEL_CONFIGS['llama']
        
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
        # 메인 레이아웃
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=1)
        self.grid_rowconfigure(3, weight=0)
        
        # 상단 바
        self.setup_top_bar()
        
        # 설정 패널
        self.setup_settings_panel()
        
        # 대화 영역
        self.setup_chat_area()
        
        # 입력 영역
        self.setup_input_area()
    
    def setup_top_bar(self):
        """상단 바 구성"""
        top_frame = ctk.CTkFrame(self, height=60)
        top_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        top_frame.grid_columnconfigure(2, weight=1)
        
        # 모델 로드 버튼
        self.load_button = ctk.CTkButton(
            top_frame,
            text="모델 로드",
            command=self.load_model,
            width=100,
            height=35
        )
        self.load_button.grid(row=0, column=0, padx=5, pady=5)
        
        # 모델 타입 선택
        self.model_type_var = ctk.StringVar(value="자동 감지")
        self.model_type_menu = ctk.CTkOptionMenu(
            top_frame,
            values=["자동 감지", "phi", "llama", "mistral"],
            variable=self.model_type_var,
            width=120,
            height=35
        )
        self.model_type_menu.grid(row=0, column=1, padx=5, pady=5)
        
        # 모델 정보
        self.model_info_label = ctk.CTkLabel(
            top_frame,
            text="모델: 없음",
            font=("Arial", 12)
        )
        self.model_info_label.grid(row=0, column=2, padx=10, pady=5, sticky="w")
        
        # 대화 관리 버튼들
        button_frame = ctk.CTkFrame(top_frame, fg_color="transparent")
        button_frame.grid(row=0, column=3, padx=5, pady=5)
        
        # 대화 저장
        self.save_button = ctk.CTkButton(
            button_frame,
            text="대화 저장",
            command=self.save_conversation,
            width=80,
            height=35,
            fg_color=THEME['success']
        )
        self.save_button.grid(row=0, column=0, padx=2)
        
        # 대화 불러오기
        self.load_conv_button = ctk.CTkButton(
            button_frame,
            text="대화 열기",
            command=self.load_conversation,
            width=80,
            height=35,
            fg_color=THEME['primary']
        )
        self.load_conv_button.grid(row=0, column=1, padx=2)
        
        # 대화 초기화
        self.clear_button = ctk.CTkButton(
            button_frame,
            text="초기화",
            command=self.clear_conversation,
            width=80,
            height=35,
            fg_color=THEME['error']
        )
        self.clear_button.grid(row=0, column=2, padx=2)
    
    def setup_settings_panel(self):
        """설정 패널 구성"""
        settings_frame = ctk.CTkFrame(self, height=100)
        settings_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        settings_frame.grid_columnconfigure(1, weight=1)
        
        # 시스템 프롬프트 라벨
        prompt_label = ctk.CTkLabel(
            settings_frame,
            text="시스템 프롬프트:",
            font=("Arial", 11)
        )
        prompt_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        # 시스템 프롬프트 입력
        self.system_prompt_entry = ctk.CTkTextbox(
            settings_frame,
            height=40,
            wrap="word",
            font=("Arial", 11)
        )
        self.system_prompt_entry.grid(row=0, column=1, padx=(0, 10), pady=5, sticky="ew")
        self.system_prompt_entry.insert("0.0", self.system_prompt)
        
        # 파라미터 설정
        param_frame = ctk.CTkFrame(settings_frame, fg_color="transparent")
        param_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        
        # Temperature
        temp_label = ctk.CTkLabel(param_frame, text="Temperature:", font=("Arial", 10))
        temp_label.grid(row=0, column=0, padx=5, sticky="w")
        
        self.temp_slider = ctk.CTkSlider(
            param_frame,
            from_=0.1,
            to=2.0,
            number_of_steps=19,
            width=150
        )
        self.temp_slider.set(0.7)
        self.temp_slider.grid(row=0, column=1, padx=5)
        
        self.temp_value = ctk.CTkLabel(param_frame, text="0.7", font=("Arial", 10))
        self.temp_value.grid(row=0, column=2, padx=5)
        self.temp_slider.configure(command=lambda v: self.temp_value.configure(text=f"{v:.1f}"))
        
        # Max Tokens
        tokens_label = ctk.CTkLabel(param_frame, text="Max Tokens:", font=("Arial", 10))
        tokens_label.grid(row=0, column=3, padx=(20, 5), sticky="w")
        
        self.tokens_entry = ctk.CTkEntry(param_frame, width=80, placeholder_text="512")
        self.tokens_entry.grid(row=0, column=4, padx=5)
        self.tokens_entry.insert(0, "512")
        
        # GPU Layers
        gpu_label = ctk.CTkLabel(param_frame, text="GPU Layers:", font=("Arial", 10))
        gpu_label.grid(row=0, column=5, padx=(20, 5), sticky="w")
        
        self.gpu_entry = ctk.CTkEntry(param_frame, width=60, placeholder_text="0")
        self.gpu_entry.grid(row=0, column=6, padx=5)
        self.gpu_entry.insert(0, "0")
    
    def setup_chat_area(self):
        """대화 영역 구성"""
        # 대화 표시 영역
        self.chat_frame = ctk.CTkScrollableFrame(self)
        self.chat_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=5)
        
        # 초기 메시지
        self.add_message("시스템", "GGUF 모델을 로드한 후 대화를 시작하세요.\n지원 모델: Phi-4, Llama, Mistral 등", is_user=False)
    
    def setup_input_area(self):
        """입력 영역 구성"""
        input_frame = ctk.CTkFrame(self)
        input_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=5)
        input_frame.grid_columnconfigure(0, weight=1)
        
        # 입력창
        self.input_box = ctk.CTkTextbox(
            input_frame,
            height=80,
            wrap="word",
            font=("Arial", 12)
        )
        self.input_box.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.input_box.bind("<Return>", self.handle_return)
        self.input_box.bind("<Control-Return>", lambda e: self.input_box.insert("insert", "\n"))
        
        # 버튼 프레임
        button_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
        button_frame.grid(row=0, column=1, sticky="ns")
        
        # 전송 버튼
        self.send_button = ctk.CTkButton(
            button_frame,
            text="전송",
            command=self.send_message,
            width=80,
            height=35,
            state="disabled"
        )
        self.send_button.pack(pady=(0, 5))
        
        # 중지 버튼
        self.stop_button = ctk.CTkButton(
            button_frame,
            text="중지",
            command=self.stop_generating,
            width=80,
            height=35,
            fg_color=THEME['error'],
            state="disabled"
        )
        self.stop_button.pack()
        
        # 단축키 안내
        shortcut_label = ctk.CTkLabel(
            input_frame,
            text="Enter: 전송 | Ctrl+Enter: 줄바꿈",
            font=("Arial", 9),
            text_color="#999999"
        )
        shortcut_label.grid(row=1, column=0, sticky="w", padx=5, pady=2)
    
    def detect_model_type(self, model_path: str) -> str:
        """모델 타입 자동 감지"""
        model_name = os.path.basename(model_path).lower()
        
        if 'phi' in model_name:
            return 'phi'
        elif 'llama' in model_name:
            return 'llama'
        elif 'mistral' in model_name:
            return 'mistral'
        else:
            # 기본값
            return 'llama'
    
    def add_message(self, sender, message, is_user=True):
        """메시지 추가"""
        # 메시지 컨테이너
        container = ctk.CTkFrame(
            self.chat_frame,
            fg_color="#E3F2FD" if is_user else "#F5F5F5"
        )
        container.pack(fill="x", padx=5, pady=3)
        
        # 헤더 프레임
        header_frame = ctk.CTkFrame(container, fg_color="transparent")
        header_frame.pack(fill="x", padx=10, pady=(5, 0))
        
        # 발신자 라벨
        sender_label = ctk.CTkLabel(
            header_frame,
            text=sender,
            font=("Arial", 10, "bold"),
            text_color="#666666"
        )
        sender_label.pack(side="left")
        
        # 타임스탬프
        timestamp = datetime.now().strftime("%H:%M:%S")
        time_label = ctk.CTkLabel(
            header_frame,
            text=timestamp,
            font=("Arial", 9),
            text_color="#999999"
        )
        time_label.pack(side="right")
        
        # 메시지 라벨
        message_label = ctk.CTkLabel(
            container,
            text=message,
            font=("Arial", 12),
            justify="left",
            wraplength=850,
            anchor="w"
        )
        message_label.pack(anchor="w", padx=10, pady=(0, 5), fill="x")
        
        # 대화 관리자에 저장
        if sender != "시스템":
            role = "user" if is_user else "assistant"
            self.conversation_manager.add_message(role, message)
        
        # 스크롤 이동
        self.after(100, lambda: self.chat_frame._parent_canvas.yview_moveto(1.0))
    
    def load_model(self):
        """모델 파일 선택 및 로드"""
        if not CTRANSFORMERS_AVAILABLE:
            messagebox.showerror("오류", "ctransformers가 설치되지 않았습니다.")
            return
        
        model_file = filedialog.askopenfilename(
            title="GGUF 모델 파일 선택",
            filetypes=[("GGUF 파일", "*.gguf"), ("GGML 파일", "*.bin"), ("모든 파일", "*.*")]
        )
        
        if not model_file:
            return
        
        # 로딩 표시
        self.model_info_label.configure(text="모델 로드 중...")
        self.load_button.configure(state="disabled")
        
        # 별도 스레드에서 모델 로드
        threading.Thread(
            target=self._load_model_thread,
            args=(model_file,),
            daemon=True
        ).start()
    
    def _load_model_thread(self, model_file):
        """모델 로드 스레드"""
        try:
            # 모델 타입 결정
            if self.model_type_var.get() == "자동 감지":
                model_type = self.detect_model_type(model_file)
            else:
                model_type = self.model_type_var.get()
            
            self.model_type = model_type
            self.model_config = MODEL_CONFIGS.get(model_type, MODEL_CONFIGS['llama'])
            
            logger.info(f"모델 로드 시작: {model_file}")
            logger.info(f"모델 타입: {model_type}")
            
            # 파라미터 읽기
            try:
                max_tokens = int(self.tokens_entry.get())
            except:
                max_tokens = self.model_config['max_new_tokens']
            
            try:
                gpu_layers = int(self.gpu_entry.get())
            except:
                gpu_layers = 0
            
            temperature = float(self.temp_slider.get())
            
            # CTransformers로 모델 로드
            self.model = AutoModelForCausalLM.from_pretrained(
                model_file,
                model_type=self.model_config['model_type'],
                context_length=self.model_config['context_length'],
                max_new_tokens=max_tokens,
                threads=4,
                temperature=temperature,
                top_p=self.model_config['top_p'],
                repetition_penalty=self.model_config['repetition_penalty'],
                gpu_layers=gpu_layers
            )
            
            self.model_path = model_file
            model_name = os.path.basename(model_file)
            
            # UI 업데이트
            self.after(100, lambda: self._handle_model_loaded(model_name))
            
        except Exception as e:
            error_msg = f"모델 로드 실패: {str(e)}"
            logger.error(error_msg)
            self.after(100, lambda: self._handle_model_error(error_msg))
    
    def _handle_model_loaded(self, model_name):
        """모델 로드 완료 처리"""
        self.model_info_label.configure(text=f"모델: {model_name} ({self.model_type})")
        self.load_button.configure(state="normal")
        self.send_button.configure(state="normal")
        self.add_message("시스템", f"모델 '{model_name}'이 로드되었습니다.\n타입: {self.model_type}", is_user=False)
        logger.info("모델 로드 완료")
    
    def _handle_model_error(self, error_msg):
        """모델 로드 오류 처리"""
        self.model_info_label.configure(text="모델: 없음")
        self.load_button.configure(state="normal")
        messagebox.showerror("오류", error_msg)
    
    def send_message(self):
        """메시지 전송"""
        message = self.input_box.get("0.0", "end").strip()
        
        if not message or self.is_generating:
            return
        
        if not self.model:
            messagebox.showwarning("경고", "먼저 모델을 로드해주세요.")
            return
        
        # 시스템 프롬프트 업데이트
        self.system_prompt = self.system_prompt_entry.get("0.0", "end").strip()
        
        # 입력 초기화
        self.input_box.delete("0.0", "end")
        
        # 사용자 메시지 표시
        self.add_message("사용자", message, is_user=True)
        
        # UI 상태 업데이트
        self.is_generating = True
        self.stop_generation = False
        self.send_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        
        # 별도 스레드에서 응답 생성
        threading.Thread(
            target=self._generate_response_thread,
            args=(message,),
            daemon=True
        ).start()
    
    def _generate_response_thread(self, user_message):
        """응답 생성 스레드"""
        try:
            # 프롬프트 구성
            prompt = self._build_prompt(user_message)
            
            logger.info("응답 생성 시작")
            
            # Temperature 업데이트
            temperature = float(self.temp_slider.get())
            
            # CTransformers로 응답 생성
            response = ""
            for token in self.model(prompt, stream=True, temperature=temperature):
                if self.stop_generation:
                    break
                response += token
            
            # 응답 정리
            response = self._clean_response(response)
            
            # 응답 표시
            if response.strip():
                self.after(100, lambda: self._handle_response(response.strip()))
            else:
                self.after(100, lambda: self._handle_response("응답을 생성할 수 없습니다."))
            
        except Exception as e:
            error_msg = f"응답 생성 오류: {str(e)}"
            logger.error(error_msg)
            self.after(100, lambda: self._handle_response(error_msg))
    
    def _build_prompt(self, user_message):
        """프롬프트 구성"""
        template = self.model_config['prompt_template']
        
        if self.model_type == 'phi':
            # Phi 모델용 프롬프트
            prompt = f"<|system|>\n{self.system_prompt}<|end|>\n"
            
            # 대화 컨텍스트 추가
            context = self.conversation_manager.get_context(6)
            for msg in context[:-1]:  # 현재 메시지 제외
                if msg["role"] == "user":
                    prompt += f"<|user|>\n{msg['content']}<|end|>\n"
                else:
                    prompt += f"<|assistant|>\n{msg['content']}<|end|>\n"
            
            prompt += f"<|user|>\n{user_message}<|end|>\n<|assistant|>\n"
            
        elif self.model_type == 'mistral':
            # Mistral 모델용 프롬프트
            full_context = f"{self.system_prompt}\n\n"
            
            context = self.conversation_manager.get_context(6)
            for msg in context[:-1]:
                if msg["role"] == "user":
                    full_context += f"User: {msg['content']}\n"
                else:
                    full_context += f"Assistant: {msg['content']}\n"
            
            full_context += f"User: {user_message}"
            prompt = f"[INST] {full_context} [/INST]"
            
        else:
            # Llama 및 기본 모델용 프롬프트
            prompt = f"### System:\n{self.system_prompt}\n\n"
            
            context = self.conversation_manager.get_context(6)
            for msg in context[:-1]:
                if msg["role"] == "user":
                    prompt += f"### User:\n{msg['content']}\n\n"
                else:
                    prompt += f"### Assistant:\n{msg['content']}\n\n"
            
            prompt += f"### User:\n{user_message}\n\n### Assistant:\n"
        
        return prompt
    
    def _clean_response(self, response: str) -> str:
        """응답 정리"""
        # 모델별 종료 토큰 제거
        stop_tokens = ['<|end|>', '<|endoftext|>', '</s>', '[/INST]', '### User:', '### System:']
        
        for token in stop_tokens:
            if token in response:
                response = response.split(token)[0]
        
        return response.strip()
    
    def _handle_response(self, response):
        """응답 처리"""
        self.add_message("AI", response, is_user=False)
        
        # UI 상태 복원
        self.is_generating = False
        self.send_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        logger.info("응답 생성 완료")
    
    def stop_generating(self):
        """생성 중지"""
        self.stop_generation = True
        self.is_generating = False
        self.send_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        logger.info("응답 생성 중지")
    
    def handle_return(self, event):
        """Enter 키 처리"""
        if not event.state & 0x1:  # Shift가 눌리지 않은 경우
            self.send_message()
            return "break"
    
    def save_conversation(self):
        """대화 저장"""
        if not self.conversation_manager.conversations:
            messagebox.showwarning("경고", "저장할 대화가 없습니다.")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON 파일", "*.json"), ("모든 파일", "*.*")],
            initialfile=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        if filepath:
            if self.conversation_manager.save_to_file(filepath):
                messagebox.showinfo("성공", "대화가 저장되었습니다.")
            else:
                messagebox.showerror("오류", "대화 저장에 실패했습니다.")
    
    def load_conversation(self):
        """대화 불러오기"""
        filepath = filedialog.askopenfilename(
            filetypes=[("JSON 파일", "*.json"), ("모든 파일", "*.*")]
        )
        
        if filepath:
            if self.conversation_manager.load_from_file(filepath):
                # 화면 업데이트
                self.refresh_chat_display()
                messagebox.showinfo("성공", "대화를 불러왔습니다.")
            else:
                messagebox.showerror("오류", "대화 불러오기에 실패했습니다.")
    
    def refresh_chat_display(self):
        """대화 화면 새로고침"""
        # 화면 초기화
        for widget in self.chat_frame.winfo_children():
            widget.destroy()
        
        # 대화 재표시
        for msg in self.conversation_manager.conversations:
            is_user = msg["role"] == "user"
            sender = "사용자" if is_user else "AI"
            self.add_message(sender, msg["content"], is_user)
    
    def clear_conversation(self):
        """대화 초기화"""
        if not self.conversation_manager.conversations:
            return
        
        result = messagebox.askyesno("확인", "대화 내용을 모두 지우시겠습니까?")
        if not result:
            return
        
        # 대화 관리자 초기화
        self.conversation_manager.clear()
        
        # 화면 초기화
        for widget in self.chat_frame.winfo_children():
            widget.destroy()
        
        # 초기 메시지
        self.add_message("시스템", "대화가 초기화되었습니다.", is_user=False)
        logger.info("대화 초기화")

# 메인 실행
if __name__ == "__main__":
    if not CTRANSFORMERS_AVAILABLE:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "오류",
            "ctransformers가 설치되지 않았습니다.\n\n"
            "다음 명령어로 설치해주세요:\n"
            "pip install ctransformers\n\n"
            "GPU 사용 시:\n"
            "pip install ctransformers[cuda]"
        )
        root.destroy()
        sys.exit(1)
    
    try:
        app = SimpleConversationApp()
        app.mainloop()
    except Exception as e:
        logger.error(f"애플리케이션 실행 오류: {str(e)}")
        sys.exit(1)