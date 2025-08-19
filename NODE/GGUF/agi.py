# -*- coding: utf-8 -*-
"""
간소화된 GGUF 대화 시스템
RAG 및 LangChain 기능 제거 버전
"""

import os
import sys
import json
import time
import queue
import threading
import customtkinter as ctk
from tkinter.filedialog import askopenfilename
from tkinter import messagebox
from pathlib import Path
import logging
from datetime import datetime
import traceback

# llama-cpp-python 직접 import
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print("llama-cpp-python가 설치되지 않았습니다. 설치해주세요: pip install llama-cpp-python")

# 로깅 설정
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# 전역 테마 설정
THEME = {
    'primary': '#2E86C1',
    'secondary': '#AED6F1',
    'error': '#E74C3C',
    'background': '#F4F6F7',
    'surface': '#FFFFFF',
    'text': '#2C3E50'
}

# 디렉토리 생성
def create_app_directories():
    """애플리케이션 필요 디렉토리 생성"""
    dirs = ["models", "logs", "configs", "conversations"]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"디렉토리 생성 또는 확인: {dir_path}")

# 설정 관리 클래스
class ConfigManager:
    """애플리케이션 설정 관리"""
    
    def __init__(self):
        self.config_file = "configs/app_config.json"
        self.defaults = {
            "theme": "system",
            "model_path": "",
            "context_size": 2048,
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 0.95,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "n_threads": 4,
            "n_gpu_layers": 0,
            "seed": -1,
            "recent_conversations": []
        }
        self.config = self.load_config()
    
    def load_config(self):
        """설정 파일 로드"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    # 누락된 기본값 추가
                    for key, value in self.defaults.items():
                        if key not in config:
                            config[key] = value
                    return config
            except Exception as e:
                logger.error(f"설정 파일 로드 오류: {str(e)}")
                return self.defaults.copy()
        else:
            self.save_config(self.defaults)
            return self.defaults.copy()
    
    def save_config(self, config=None):
        """설정 파일 저장"""
        if config is None:
            config = self.config
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            logger.info("설정 파일 저장 완료")
            return True
        except Exception as e:
            logger.error(f"설정 파일 저장 오류: {str(e)}")
            return False
    
    def get(self, key, default=None):
        """설정 값 가져오기"""
        return self.config.get(key, default)
    
    def set(self, key, value):
        """설정 값 설정"""
        self.config[key] = value
        self.save_config()

# GGUF 모델 관리 클래스
class GGUFModel:
    """GGUF 모델 래퍼 클래스"""
    
    def __init__(self, config_manager):
        self.config = config_manager
        self.model = None
        self.model_path = ""
    
    def load_model(self, model_path):
        """GGUF 모델 로드"""
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError("llama-cpp-python이 설치되지 않았습니다.")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일이 존재하지 않습니다: {model_path}")
        
        try:
            # 기존 모델 정리
            if self.model:
                del self.model
                self.model = None
            
            # 설정 가져오기
            context_size = self.config.get("context_size", 2048)
            n_threads = self.config.get("n_threads", 4)
            n_gpu_layers = self.config.get("n_gpu_layers", 0)
            seed = self.config.get("seed", -1)
            
            # 모델 로드
            self.model = Llama(
                model_path=model_path,
                n_ctx=context_size,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                seed=seed,
                verbose=False
            )
            
            self.model_path = model_path
            logger.info(f"모델 로드 성공: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"모델 로드 오류: {str(e)}")
            raise
    
    def generate(self, prompt, **kwargs):
        """텍스트 생성"""
        if not self.model:
            raise RuntimeError("모델이 로드되지 않았습니다.")
        
        # 기본 설정
        temperature = kwargs.get("temperature", self.config.get("temperature", 0.7))
        max_tokens = kwargs.get("max_tokens", self.config.get("max_tokens", 1000))
        top_p = kwargs.get("top_p", self.config.get("top_p", 0.95))
        top_k = kwargs.get("top_k", self.config.get("top_k", 40))
        repeat_penalty = kwargs.get("repeat_penalty", self.config.get("repeat_penalty", 1.1))
        
        try:
            # 응답 생성
            response = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
                echo=False
            )
            
            # 텍스트 추출
            if isinstance(response, dict) and "choices" in response:
                return response["choices"][0]["text"]
            else:
                return str(response)
                
        except Exception as e:
            logger.error(f"텍스트 생성 오류: {str(e)}")
            raise
    
    def is_loaded(self):
        """모델 로드 여부 확인"""
        return self.model is not None

# 사용자 관리 클래스
class User:
    def __init__(self, username: str, password: str, department: str, role: str = "user"):
        self.username = username
        self.password = password
        self.department = department
        self.role = role
        self.login_time = None

class UserManager:
    def __init__(self):
        self._initialize_default_users()
        self.current_user = None
        self.login_attempts = {}

    def _initialize_default_users(self):
        self.users = {
            "디지털혁신부": User("디지털혁신부", "1234", "디지털혁신부", "admin"),
            "사업기획부": User("사업기획부", "1234", "사업기획부", "user"),
        }

    def login(self, username: str, password: str) -> bool:
        if username not in self.login_attempts:
            self.login_attempts[username] = {"count": 0, "lockout_until": None}
        attempts = self.login_attempts[username]
        if attempts["lockout_until"] and time.time() < attempts["lockout_until"]:
            return False
        if username in self.users and self.users[username].password == password:
            self.current_user = self.users[username]
            self.current_user.login_time = time.time()
            attempts["count"] = 0
            return True
        attempts["count"] += 1
        if attempts["count"] >= 3:
            attempts["lockout_until"] = time.time() + 300
        return False

# 로그인 창
class ModernLoginWindow(ctk.CTkToplevel):
    def __init__(self, user_manager, on_login_success):
        super().__init__()
        self.user_manager = user_manager
        self.on_login_success = on_login_success
        self.setup_window()
        self.create_widgets()
        self.bind_events()

    def setup_window(self):
        self.title("GGUF 대화 시스템")
        self.geometry("400x450")
        self.configure(fg_color=THEME['background'])
        self.withdraw()
        self.update_idletasks()
        x = (self.winfo_screenwidth() - 400) // 2
        y = (self.winfo_screenheight() - 450) // 2
        self.geometry(f"+{x}+{y}")
        self.deiconify()
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        self.logo_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.logo_frame.pack(pady=(40, 20))
        ctk.CTkLabel(
            self.logo_frame,
            text="GGUF 대화 시스템",
            font=("Helvetica", 26, "bold"),
            text_color=THEME['primary']
        ).pack()

        self.content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.content_frame.pack(pady=10, padx=40, fill="x")

        self.department_var = ctk.StringVar(value="디지털혁신부")
        ctk.CTkLabel(
            self.content_frame,
            text="부서",
            font=("Helvetica", 12, "bold"),
            text_color=THEME['text']
        ).pack(pady=(0, 5), anchor="w")
        self.department_menu = ctk.CTkOptionMenu(
            self.content_frame,
            values=["디지털혁신부", "사업기획부"],
            variable=self.department_var,
            width=320,
            font=("Helvetica", 12),
            fg_color=THEME['surface'],
            button_color=THEME['primary'],
            dropdown_fg_color=THEME['surface'],
            dropdown_font=("Helvetica", 12),
            corner_radius=12
        )
        self.department_menu.pack(pady=(0, 15))

        self.model_var = ctk.StringVar(value="모델을 선택하세요")
        ctk.CTkLabel(
            self.content_frame,
            text="모델",
            font=("Helvetica", 12, "bold"),
            text_color=THEME['text']
        ).pack(pady=(0, 5), anchor="w")
        
        self.browse_button = ctk.CTkButton(
            self.content_frame,
            text="모델 선택...",
            command=self.browse_model_file,
            width=320,
            font=("Helvetica", 12),
            fg_color=THEME['primary'],
            hover_color=THEME['secondary'],
            corner_radius=12
        )
        self.browse_button.pack(pady=(0, 15))

        self.model_label = ctk.CTkLabel(
            self.content_frame,
            text="선택된 모델: 없음",
            font=("Helvetica", 10),
            text_color=THEME['text']
        )
        self.model_label.pack(pady=(0, 15))

        ctk.CTkLabel(
            self.content_frame,
            text="비밀번호",
            font=("Helvetica", 12, "bold"),
            text_color=THEME['text']
        ).pack(pady=(0, 5), anchor="w")
        self.password_entry = ctk.CTkEntry(
            self.content_frame,
            show="●",
            width=320,
            height=40,
            font=("Helvetica", 12),
            placeholder_text="비밀번호 입력",
            border_width=0,
            fg_color=THEME['surface'],
            text_color=THEME['text'],
            placeholder_text_color='#A0AEC0',
            corner_radius=12
        )
        self.password_entry.pack(pady=(0, 20))

        self.login_button = ctk.CTkButton(
            self.content_frame,
            text="로그인",
            command=self.login,
            width=320,
            height=45,
            font=("Helvetica", 14, "bold"),
            fg_color=THEME['primary'],
            hover_color=THEME['secondary'],
            corner_radius=12
        )
        self.login_button.pack(pady=10)

        self.error_label = ctk.CTkLabel(
            self.content_frame,
            text="",
            text_color=THEME['error'],
            font=("Helvetica", 12)
        )
        self.error_label.pack(pady=5)

    def bind_events(self):
        self.bind('<Return>', lambda e: self.login())
        self.password_entry.bind('<Return>', lambda e: self.login())

    def browse_model_file(self):
        file_path = askopenfilename(
            title="GGUF 모델 선택",
            filetypes=[("GGUF files", "*.gguf"), ("All files", "*.*")],
            initialdir="./models"
        )
        if file_path and Path(file_path).exists():
            self.model_var.set(file_path)
            model_name = os.path.basename(file_path)
            self.model_label.configure(text=f"선택된 모델: {model_name}")
        elif file_path:
            messagebox.showerror("오류", "선택한 모델 파일이 존재하지 않습니다!")

    def login(self):
        username = self.department_var.get()
        password = self.password_entry.get()
        model_path = self.model_var.get()
        
        if model_path == "모델을 선택하세요":
            self.error_label.configure(text="모델을 선택해주세요")
            return
            
        if self.user_manager.login(username, password):
            self.on_login_success(model_path)
            self.destroy()
        else:
            self.error_label.configure(text="잘못된 비밀번호입니다")
            self.password_entry.delete(0, 'end')

    def on_closing(self):
        self.quit()
        self.destroy()
        sys.exit()

# 대화 메시지 클래스
class ChatMessage(ctk.CTkFrame):
    def __init__(self, master, message: str, is_user: bool = True):
        bg_color = '#D6EAF8' if is_user else '#FDFEFE'
        super().__init__(master, fg_color=bg_color, border_width=0, corner_radius=16)
        self.message = message
        self.is_user = is_user
        self.setup_message()

    def setup_message(self):
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=15, pady=(10, 0))
        sender_color = THEME['primary'] if self.is_user else THEME['secondary']
        sender_text = "사용자" if self.is_user else "AI"
        ctk.CTkLabel(
            header,
            text=sender_text,
            font=("Helvetica", 11, "bold"),
            text_color=sender_color
        ).pack(side="left")
        
        content_frame = ctk.CTkFrame(self, fg_color="transparent")
        content_frame.pack(fill="x", padx=15, pady=5)
        
        label = ctk.CTkLabel(
            content_frame,
            text=self.message,
            font=("Helvetica", 13),
            wraplength=700,
            justify="left",
            text_color=THEME['text']
        )
        label.pack(fill="x")

# 메인 UI
class ModernChatUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.withdraw()
        self.user_manager = UserManager()
        self.config_manager = ConfigManager()
        self.model = GGUFModel(self.config_manager)
        self.setup_window()
        self.response_queue = queue.Queue()
        self.is_generating = False
        self.conversation_history = []
        self.show_login()

    def setup_window(self):
        self.title("GGUF 대화 시스템")
        self.geometry("1000x800")
        self.configure(fg_color=THEME['background'])
        
        # 헤더
        self.header = ctk.CTkFrame(self, fg_color=THEME['surface'], height=80)
        self.header.pack(fill="x", side="top")
        self.header_left_frame = ctk.CTkFrame(self.header, fg_color=THEME['surface'])
        self.header_left_frame.pack(side="left", fill="both", expand=True, padx=20, pady=10)
        
        self.title_logout_frame = ctk.CTkFrame(self.header_left_frame, fg_color=THEME['surface'])
        self.title_logout_frame.pack(anchor="w", fill="x")
        
        self.title_label = ctk.CTkLabel(
            self.title_logout_frame,
            text="GGUF 대화 시스템",
            font=("Helvetica", 20, "bold"),
            text_color=THEME['primary']
        )
        self.title_label.pack(side="left", anchor="w")
        
        self.logout_button = ctk.CTkButton(
            self.title_logout_frame,
            text="로그아웃",
            command=self.logout,
            width=80,
            font=("Helvetica", 12, "bold"),
            fg_color=THEME['error'],
            hover_color="#F08080",
            corner_radius=8
        )
        self.logout_button.pack(side="left", anchor="w", padx=10, pady=5)
        
        self.model_frame = ctk.CTkFrame(self.header_left_frame, fg_color=THEME['secondary'], corner_radius=10, height=30)
        self.model_frame.pack(anchor="w", pady=(4, 0), padx=(0, 10))
        self.model_label = ctk.CTkLabel(
            self.model_frame,
            text="모델: 없음",
            font=("Helvetica", 14, "bold"),
            text_color=THEME['primary'],
            fg_color=THEME['secondary']
        )
        self.model_label.pack(side="left", padx=(2, 4), pady=5)
        
        # 메인 프레임
        self.main_frame = ctk.CTkFrame(self, fg_color=THEME['background'])
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=10)
        self.main_frame.grid_columnconfigure(0, weight=0)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)
        
        # 사이드바
        self.sidebar = ctk.CTkFrame(self.main_frame, width=200, fg_color=THEME['surface'], corner_radius=12)
        self.sidebar.grid(row=0, column=0, sticky="nswe", padx=(0, 10))
        
        self.user_info = ctk.CTkLabel(
            self.sidebar,
            text="로그인되지 않음",
            font=("Helvetica", 16, "bold"),
            text_color=THEME['text']
        )
        self.user_info.pack(pady=20, padx=15, anchor="w")
        
        ctk.CTkLabel(
            self.sidebar,
            text="최대 토큰:",
            font=("Helvetica", 12, "bold"),
            text_color=THEME['text']
        ).pack(pady=(10, 0), padx=15, anchor="w")
        self.token_slider = ctk.CTkSlider(self.sidebar, from_=128, to=2048, number_of_steps=30, command=self.update_token_label)
        self.token_slider.set(500)
        self.token_slider.pack(pady=(0, 5), padx=15, fill="x")
        self.token_value_label = ctk.CTkLabel(self.sidebar, text="500", font=("Helvetica", 12), text_color=THEME['text'])
        self.token_value_label.pack(pady=(0, 5), padx=15, anchor="w")
        
        ctk.CTkLabel(
            self.sidebar,
            text="온도:",
            font=("Helvetica", 12, "bold"),
            text_color=THEME['text']
        ).pack(pady=(10, 0), padx=15, anchor="w")
        self.temp_slider = ctk.CTkSlider(self.sidebar, from_=0, to=1, number_of_steps=20, command=self.update_temp_label)
        self.temp_slider.set(0.7)
        self.temp_slider.pack(pady=(0, 5), padx=15, fill="x")
        self.temp_value_label = ctk.CTkLabel(self.sidebar, text="0.7", font=("Helvetica", 12), text_color=THEME['text'])
        self.temp_value_label.pack(pady=(0, 10), padx=15, anchor="w")
        
        self.clear_button = ctk.CTkButton(
            self.sidebar,
            text="대화 초기화",
            font=("Helvetica", 12, "bold"),
            fg_color=THEME['error'],
            hover_color="#F08080",
            corner_radius=8,
            command=self.clear_conversation
        )
        self.clear_button.pack(pady=(0, 10), padx=15, fill="x")
        
        self.save_button = ctk.CTkButton(
            self.sidebar,
            text="대화 저장",
            font=("Helvetica", 12, "bold"),
            fg_color=THEME['primary'],
            hover_color=THEME['secondary'],
            corner_radius=8,
            command=self.save_conversation
        )
        self.save_button.pack(pady=(0, 10), padx=15, fill="x")
        
        # 대화 영역
        self.chat_frame = ctk.CTkFrame(self.main_frame, fg_color=THEME['background'])
        self.chat_frame.grid(row=0, column=1, sticky="nswe")
        self.chat_container = ctk.CTkScrollableFrame(self.chat_frame, fg_color=THEME['surface'], corner_radius=12)
        self.chat_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 입력 영역
        self.input_frame = ctk.CTkFrame(self.chat_frame, fg_color=THEME['background'])
        self.input_frame.pack(fill="x", pady=(10, 10), padx=10)
        
        self.input_field = ctk.CTkTextbox(
            self.input_frame,
            height=80,
            font=("Helvetica", 13),
            wrap="word",
            border_width=1,
            border_color=THEME['primary'],
            corner_radius=12,
            fg_color=THEME['surface']
        )
        self.input_field.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        self.stop_button = ctk.CTkButton(
            self.input_frame,
            text="중지",
            width=80,
            height=45,
            font=("Helvetica", 14, "bold"),
            command=self.stop_generation,
            fg_color=THEME['error'],
            hover_color="#F08080",
            corner_radius=12,
            state="disabled"
        )
        self.stop_button.pack(side="right", padx=(0, 10))
        
        self.send_button = ctk.CTkButton(
            self.input_frame,
            text="전송",
            width=80,
            height=45,
            font=("Helvetica", 14, "bold"),
            command=self.send_message,
            fg_color=THEME['primary'],
            hover_color=THEME['secondary'],
            corner_radius=12,
            state="disabled"
        )
        self.send_button.pack(side="right")
        
        self.input_field.bind("<Return>", self.handle_return)
        self.input_field.bind("<Shift-Return>", self.handle_shift_return)

    def show_login(self):
        ModernLoginWindow(self.user_manager, self.on_login_success)

    def on_login_success(self, model_path):
        try:
            # 모델 로드
            self.model.load_model(model_path)
            self.config_manager.set("model_path", model_path)
            
            self.user_info.configure(text=f"부서: {self.user_manager.current_user.department}")
            self.conversation_history = []
            self.deiconify()
            
            model_filename = Path(model_path).stem
            self.model_label.configure(text=f"모델: {model_filename}")
            self.send_button.configure(state="normal")
            
            welcome_message = (
                f"GGUF 대화 시스템에 오신 것을 환영합니다!\n\n"
                f"로드된 모델: {model_filename}\n"
                f"부서: {self.user_manager.current_user.department}\n\n"
                "대화를 시작해주세요."
            )
            self.add_chat_message(welcome_message, is_user=False)
        except Exception as e:
            messagebox.showerror("오류", f"모델 로드 실패: {str(e)}")
            self.destroy()

    def logout(self):
        if messagebox.askyesno("로그아웃", "로그아웃 하시겠습니까?"):
            self.model.model = None
            self.conversation_history = []
            self.user_info.configure(text="로그인되지 않음")
            for widget in self.chat_container.winfo_children():
                widget.destroy()
            self.withdraw()
            self.show_login()

    def add_chat_message(self, message: str, is_user: bool = True):
        chat_message = ChatMessage(self.chat_container, message, is_user)
        chat_message.pack(fill="x", padx=10, pady=5, anchor="n")
        self.chat_container.update_idletasks()
        self.chat_container._parent_canvas.yview_moveto(1.0)
        
        if is_user:
            self.conversation_history.append({"role": "user", "content": message})
        else:
            self.conversation_history.append({"role": "assistant", "content": message})

    def send_message(self):
        user_input = self.input_field.get("0.0", "end").strip()
        if not user_input or self.is_generating:
            return
            
        self.add_chat_message(user_input, is_user=True)
        self.input_field.delete("0.0", "end")
        
        # UI 상태 업데이트
        self.is_generating = True
        self.send_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        
        # 생성 파라미터
        max_tokens = int(self.token_slider.get())
        temperature = float(self.temp_slider.get())
        
        # 별도 스레드에서 응답 생성
        thread = threading.Thread(
            target=self._generate_response_thread, 
            args=(user_input, max_tokens, temperature)
        )
        thread.start()
        self.after(100, self.check_response)

    def _generate_response_thread(self, user_input, max_tokens, temperature):
        try:
            # 프롬프트 구성
            prompt = self._build_prompt(user_input)
            
            # 응답 생성
            response = self.model.generate(
                prompt, 
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            self.response_queue.put(response)
        except Exception as e:
            self.response_queue.put(f"오류가 발생했습니다: {str(e)}")

    def _build_prompt(self, user_message):
        """프롬프트 구성"""
        # 최근 대화 내용 포함 (최근 10개)
        recent_conversation = self.conversation_history[-10:]
        
        prompt = ""
        for msg in recent_conversation:
            if msg["role"] == "user":
                prompt += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                prompt += f"Assistant: {msg['content']}\n"
        
        prompt += f"User: {user_message}\nAssistant:"
        
        return prompt

    def check_response(self):
        if not self.response_queue.empty():
            response = self.response_queue.get()
            self.add_chat_message(response, is_user=False)
            
            # UI 상태 복원
            self.is_generating = False
            self.send_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
        elif self.is_generating:
            self.after(100, self.check_response)

    def stop_generation(self):
        self.is_generating = False
        self.send_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        messagebox.showinfo("정보", "응답 생성을 중지했습니다.")

    def clear_conversation(self):
        if not self.conversation_history:
            return
            
        if messagebox.askyesno("확인", "대화 내용을 모두 지우시겠습니까?"):
            self.conversation_history = []
            for widget in self.chat_container.winfo_children():
                widget.destroy()
            self.add_chat_message("대화가 초기화되었습니다. 새로운 대화를 시작해주세요.", is_user=False)

    def save_conversation(self):
        if not self.conversation_history:
            messagebox.showinfo("알림", "저장할 대화 내용이 없습니다.")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversations/conversation_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
            messagebox.showinfo("성공", f"대화가 저장되었습니다:\n{filename}")
        except Exception as e:
            messagebox.showerror("오류", f"대화 저장 중 오류가 발생했습니다:\n{str(e)}")

    def update_token_label(self, value):
        self.token_value_label.configure(text=str(int(value)))

    def update_temp_label(self, value):
        self.temp_value_label.configure(text=f"{value:.2f}")

    def handle_return(self, event):
        self.send_message()
        return "break"

    def handle_shift_return(self, event):
        return None

# 메인 실행
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
        create_app_directories()
        app = ModernChatUI()
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