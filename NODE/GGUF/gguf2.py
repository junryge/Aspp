# -*- coding: utf-8 -*-
"""
GGUF 간소화 UI - 폐쇄망용
RAG 및 LangChain 의존성 제거 버전
fileno 오류 해결 버전
"""

import os
import sys
import io
import json
import time
import threading
import customtkinter as ctk
from tkinter import filedialog, messagebox
from pathlib import Path
import logging
from datetime import datetime
import traceback

# ===== fileno 오류 해결 코드 =====
# stdout/stderr에 fileno 메서드가 없는 경우 추가
if not hasattr(sys.stdout, 'fileno'):
    sys.stdout.fileno = lambda: 1
if not hasattr(sys.stderr, 'fileno'):
    sys.stderr.fileno = lambda: 2
    
# buffer 속성도 없을 수 있으므로 추가
if not hasattr(sys.stdout, 'buffer'):
    sys.stdout.buffer = sys.stdout
if not hasattr(sys.stderr, 'buffer'):
    sys.stderr.buffer = sys.stderr

# 환경 변수로 llama.cpp 출력 억제
os.environ['LLAMA_CPP_VERBOSE'] = '0'
# =====================================

# llama-cpp-python 직접 import
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print("llama-cpp-python가 설치되지 않았습니다. 설치해주세요: pip install llama-cpp-python")

# 안전한 스트림 핸들러 정의
class SafeStreamHandler(logging.StreamHandler):
    """fileno 오류를 방지하는 안전한 스트림 핸들러"""
    def emit(self, record):
        try:
            super().emit(record)
        except (AttributeError, OSError):
            # fileno 관련 오류 무시
            pass

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        SafeStreamHandler()  # 안전한 핸들러 사용
    ]
)
logger = logging.getLogger(__name__)

# 전역 테마 설정
THEME = {
    'primary': '#2E86C1',
    'secondary': '#AED6F1',
    'accent': '#F39C12',
    'error': '#E74C3C',
    'success': '#2ECC71',
    'warning': '#F1C40F',
    'background': '#F5F5F5',
    'surface': '#FFFFFF',
    'text': '#2C3E50',
    'text_secondary': '#7F8C8D'
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
            
            # 추가 환경 변수 설정
            os.environ['LLAMA_CPP_LOG_DISABLE'] = '1'
            
            # 모델 로드 (안전한 설정으로)
            self.model = Llama(
                model_path=model_path,
                n_ctx=context_size,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                seed=seed if seed != -1 else None,
                verbose=False,
                use_mlock=False,  # 메모리 락 비활성화
                use_mmap=True,    # 메모리 맵 사용
                logits_all=False  # 모든 로짓 저장 비활성화
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
                echo=False,
                stream=False  # 스트리밍 비활성화
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

# 메인 애플리케이션
class SimpleGGUFApp(ctk.CTk):
    """간소화된 GGUF UI 애플리케이션"""
    
    def __init__(self):
        super().__init__()
        
        # 앱 초기화
        self.title("GGUF 대화 시스템")
        self.geometry("1000x700")
        
        # 디렉토리 생성
        create_app_directories()
        
        # 설정 관리자 초기화
        self.config_manager = ConfigManager()
        
        # 테마 설정
        self.apply_theme()
        
        # 모델 초기화
        self.model = GGUFModel(self.config_manager)
        
        # 대화 이력
        self.conversation = []
        
        # 응답 생성 플래그
        self.is_generating = False
        
        # UI 구성
        self.setup_ui()
        
        # 모델 자동 로드 시도
        self.auto_load_model()
    
    def apply_theme(self):
        """테마 적용"""
        theme_mode = self.config_manager.get("theme", "system")
        ctk.set_appearance_mode(theme_mode)
        ctk.set_default_color_theme("blue")
    
    def auto_load_model(self):
        """저장된 모델 경로로 자동 로드 시도"""
        model_path = self.config_manager.get("model_path", "")
        if model_path and os.path.exists(model_path):
            threading.Thread(
                target=self._auto_load_model_thread,
                args=(model_path,),
                daemon=True
            ).start()
    
    def _auto_load_model_thread(self, model_path):
        """자동 모델 로드 스레드"""
        try:
            self.model.load_model(model_path)
            self.after(100, lambda: self.update_model_info(os.path.basename(model_path)))
        except Exception as e:
            logger.error(f"자동 모델 로드 실패: {str(e)}")
    
    def update_model_info(self, model_name):
        """모델 정보 업데이트"""
        self.model_info_label.configure(text=f"모델: {model_name}")
        self.send_button.configure(state="normal")
    
    def setup_ui(self):
        """UI 구성"""
        # 메인 레이아웃
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)
        
        # 상단 바
        self.setup_top_bar()
        
        # 대화 영역
        self.setup_chat_area()
        
        # 입력 영역
        self.setup_input_area()
    
    def setup_top_bar(self):
        """상단 바 구성"""
        top_frame = ctk.CTkFrame(self, height=60, corner_radius=0)
        top_frame.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        top_frame.grid_columnconfigure(1, weight=1)
        
        # 앱 제목
        app_title = ctk.CTkLabel(
            top_frame,
            text="GGUF 대화 시스템",
            font=("Helvetica", 20, "bold"),
            text_color=THEME['primary']
        )
        app_title.grid(row=0, column=0, padx=20, pady=15)
        
        # 모델 정보
        model_path = self.config_manager.get("model_path", "")
        model_name = os.path.basename(model_path) if model_path else "모델 없음"
        self.model_info_label = ctk.CTkLabel(
            top_frame,
            text=f"모델: {model_name}",
            font=("Helvetica", 14),
            text_color=THEME['text_secondary']
        )
        self.model_info_label.grid(row=0, column=1, padx=20, pady=15, sticky="w")
        
        # 버튼들
        button_frame = ctk.CTkFrame(top_frame, fg_color="transparent")
        button_frame.grid(row=0, column=2, padx=20, pady=15)
        
        # 모델 로드 버튼
        load_button = ctk.CTkButton(
            button_frame,
            text="모델 로드",
            command=self.load_model,
            width=100,
            height=30,
            font=("Helvetica", 12),
            fg_color=THEME['primary'],
            hover_color=THEME['secondary']
        )
        load_button.pack(side="left", padx=(0, 5))
        
        # 설정 버튼
        settings_button = ctk.CTkButton(
            button_frame,
            text="설정",
            command=self.show_settings,
            width=80,
            height=30,
            font=("Helvetica", 12),
            fg_color=THEME['primary'],
            hover_color=THEME['secondary']
        )
        settings_button.pack(side="left", padx=5)
        
        # 대화 초기화 버튼
        clear_button = ctk.CTkButton(
            button_frame,
            text="대화 초기화",
            command=self.clear_conversation,
            width=100,
            height=30,
            font=("Helvetica", 12),
            fg_color=THEME['error'],
            hover_color="#C0392B"
        )
        clear_button.pack(side="left", padx=(5, 0))
    
    def setup_chat_area(self):
        """대화 영역 구성"""
        # 대화 표시 영역
        self.chat_frame = ctk.CTkScrollableFrame(
            self,
            fg_color=THEME['surface'],
            corner_radius=10
        )
        self.chat_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
        
        # 초기 메시지
        self.add_assistant_message("안녕하세요! GGUF 모델을 로드하고 대화를 시작해보세요.")
    
    def setup_input_area(self):
        """입력 영역 구성"""
        input_frame = ctk.CTkFrame(self, fg_color="transparent", height=100)
        input_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 20))
        input_frame.grid_columnconfigure(0, weight=1)
        
        # 입력창
        self.input_box = ctk.CTkTextbox(
            input_frame,
            height=80,
            wrap="word",
            font=("Helvetica", 14),
            border_width=1,
            border_color=THEME['primary'],
            fg_color=THEME['surface'],
            corner_radius=10
        )
        self.input_box.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.input_box.bind("<Return>", self.handle_return)
        self.input_box.bind("<Shift-Return>", self.handle_shift_return)
        
        # 버튼들
        buttons_frame = ctk.CTkFrame(input_frame, fg_color="transparent", width=140)
        buttons_frame.grid(row=0, column=1, sticky="ns")
        
        self.send_button = ctk.CTkButton(
            buttons_frame,
            text="전송",
            command=self.send_message,
            width=120,
            height=35,
            font=("Helvetica", 14, "bold"),
            fg_color=THEME['primary'],
            hover_color=THEME['secondary'],
            state="disabled"  # 모델 로드 전까지 비활성화
        )
        self.send_button.pack(pady=(0, 5))
        
        self.stop_button = ctk.CTkButton(
            buttons_frame,
            text="중지",
            command=self.stop_generation,
            width=120,
            height=35,
            font=("Helvetica", 14, "bold"),
            fg_color=THEME['error'],
            hover_color="#C0392B",
            state="disabled"
        )
        self.stop_button.pack()
    
    def add_user_message(self, message):
        """사용자 메시지 추가"""
        container = ctk.CTkFrame(self.chat_frame, fg_color="#D6EAF8", corner_radius=10)
        container.pack(fill="x", padx=10, pady=5, anchor="e")
        
        label = ctk.CTkLabel(
            container,
            text=message,
            font=("Helvetica", 14),
            text_color=THEME['text'],
            wraplength=650,
            justify="left"
        )
        label.pack(padx=15, pady=10, anchor="w")
        
        self.conversation.append({"role": "user", "content": message})
        self.after(100, lambda: self.chat_frame._parent_canvas.yview_moveto(1.0))
    
    def add_assistant_message(self, message):
        """어시스턴트 메시지 추가"""
        container = ctk.CTkFrame(self.chat_frame, fg_color="#E8F6F3", corner_radius=10)
        container.pack(fill="x", padx=10, pady=5, anchor="w")
        
        label = ctk.CTkLabel(
            container,
            text=message,
            font=("Helvetica", 14),
            text_color=THEME['text'],
            wraplength=650,
            justify="left"
        )
        label.pack(padx=15, pady=10, anchor="w")
        
        self.conversation.append({"role": "assistant", "content": message})
        self.after(100, lambda: self.chat_frame._parent_canvas.yview_moveto(1.0))
    
    def load_model(self):
        """모델 파일 선택 및 로드"""
        model_file = filedialog.askopenfilename(
            title="GGUF 모델 파일 선택",
            filetypes=[("GGUF 파일", "*.gguf"), ("모든 파일", "*.*")],
            initialdir="models"
        )
        
        if not model_file:
            return
        
        # 로딩 대화상자
        loading_dialog = self.create_loading_dialog("모델 로드 중...")
        
        # 별도 스레드에서 모델 로드
        threading.Thread(
            target=self._load_model_thread,
            args=(model_file, loading_dialog),
            daemon=True
        ).start()
    
    def _load_model_thread(self, model_file, loading_dialog):
        """모델 로드 스레드"""
        try:
            self.model.load_model(model_file)
            self.config_manager.set("model_path", model_file)
            
            self.after(100, lambda: self._handle_model_load_result(True, model_file, loading_dialog))
        except Exception as e:
            self.after(100, lambda: self._handle_model_load_result(False, str(e), loading_dialog))
    
    def _handle_model_load_result(self, success, data, loading_dialog):
        """모델 로드 결과 처리"""
        loading_dialog.destroy()
        
        if success:
            model_name = os.path.basename(data)
            self.update_model_info(model_name)
            messagebox.showinfo("성공", f"모델이 성공적으로 로드되었습니다:\n{model_name}")
        else:
            messagebox.showerror("오류", f"모델 로드에 실패했습니다:\n{data}")
    
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
            font=("Helvetica", 14, "bold")
        ).pack(pady=(20, 10))
        
        progress = ctk.CTkProgressBar(dialog, width=250)
        progress.pack(pady=(0, 20))
        progress.configure(mode="indeterminate")
        progress.start()
        
        return dialog
    
    def send_message(self):
        """메시지 전송 처리"""
        message = self.input_box.get("0.0", "end").strip()
        
        if not message or self.is_generating:
            return
        
        if not self.model.is_loaded():
            messagebox.showwarning("경고", "먼저 모델을 로드해주세요.")
            return
        
        # 입력 초기화
        self.input_box.delete("0.0", "end")
        
        # 사용자 메시지 표시
        self.add_user_message(message)
        
        # UI 상태 업데이트
        self.is_generating = True
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
            # 대화 컨텍스트 구성
            prompt = self._build_prompt(user_message)
            
            # 응답 생성
            response = self.model.generate(prompt)
            
            # UI 스레드에서 결과 표시
            self.after(100, lambda: self._handle_response_result(response))
        except Exception as e:
            error_message = f"응답 생성 중 오류가 발생했습니다: {str(e)}"
            logger.error(error_message)
            self.after(100, lambda: self._handle_response_error(error_message))
    
    def _build_prompt(self, user_message):
        """프롬프트 구성"""
        # 간단한 대화 형식 프롬프트
        # 최근 몇 개의 대화만 포함 (컨텍스트 길이 제한)
        recent_conversation = self.conversation[-10:]  # 최근 10개 메시지
        
        prompt = ""
        for msg in recent_conversation:
            if msg["role"] == "user":
                prompt += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                prompt += f"Assistant: {msg['content']}\n"
        
        prompt += f"User: {user_message}\nAssistant:"
        
        return prompt
    
    def _handle_response_result(self, response):
        """응답 결과 처리"""
        # 응답 표시
        self.add_assistant_message(response.strip())
        
        # UI 상태 복원
        self.is_generating = False
        self.send_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
    
    def _handle_response_error(self, error_message):
        """응답 오류 처리"""
        self.add_assistant_message(error_message)
        
        # UI 상태 복원
        self.is_generating = False
        self.send_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
    
    def stop_generation(self):
        """응답 생성 중지"""
        self.is_generating = False
        self.send_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        messagebox.showinfo("알림", "응답 생성을 중지했습니다.")
    
    def handle_return(self, event):
        """Enter 키 처리"""
        self.send_message()
        return "break"
    
    def handle_shift_return(self, event):
        """Shift+Enter 키 처리"""
        return None
    
    def clear_conversation(self):
        """대화 초기화"""
        if not self.conversation:
            return
        
        result = messagebox.askyesno("확인", "정말 대화 내용을 모두 지우시겠습니까?")
        if not result:
            return
        
        # 대화 초기화
        self.conversation = []
        
        # 화면 초기화
        for widget in self.chat_frame.winfo_children():
            widget.destroy()
        
        # 초기 메시지
        self.add_assistant_message("안녕하세요! GGUF 모델을 로드하고 대화를 시작해보세요.")
    
    def show_settings(self):
        """설정 대화상자"""
        settings_dialog = ctk.CTkToplevel(self)
        settings_dialog.title("설정")
        settings_dialog.geometry("500x600")
        settings_dialog.transient(self)
        settings_dialog.grab_set()
        
        # 중앙 배치
        settings_dialog.update_idletasks()
        width = settings_dialog.winfo_width()
        height = settings_dialog.winfo_height()
        x = (settings_dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (settings_dialog.winfo_screenheight() // 2) - (height // 2)
        settings_dialog.geometry(f'+{x}+{y}')
        
        # 설정 프레임
        settings_frame = ctk.CTkScrollableFrame(settings_dialog)
        settings_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # 모델 설정
        model_frame = ctk.CTkFrame(settings_frame)
        model_frame.pack(fill="x", pady=(0, 15))
        
        ctk.CTkLabel(
            model_frame,
            text="모델 설정",
            font=("Helvetica", 16, "bold")
        ).pack(anchor="w", pady=(5, 10))
        
        # 설정 항목들
        settings_items = [
            ("컨텍스트 크기:", "context_size", 2048),
            ("온도 (0.0-1.0):", "temperature", 0.7),
            ("최대 토큰:", "max_tokens", 1000),
            ("Top P:", "top_p", 0.95),
            ("Top K:", "top_k", 40),
            ("반복 페널티:", "repeat_penalty", 1.1),
            ("스레드 수:", "n_threads", 4),
            ("GPU 레이어:", "n_gpu_layers", 0),
        ]
        
        vars_dict = {}
        
        for label_text, key, default in settings_items:
            frame = ctk.CTkFrame(model_frame, fg_color="transparent")
            frame.pack(fill="x", pady=5)
            
            ctk.CTkLabel(
                frame,
                text=label_text,
                width=120
            ).pack(side="left")
            
            var = ctk.StringVar(value=str(self.config_manager.get(key, default)))
            vars_dict[key] = var
            
            ctk.CTkEntry(
                frame,
                textvariable=var,
                width=100
            ).pack(side="left", padx=(10, 0))
        
        # 테마 설정
        theme_frame = ctk.CTkFrame(settings_frame)
        theme_frame.pack(fill="x", pady=(0, 15))
        
        ctk.CTkLabel(
            theme_frame,
            text="테마 설정",
            font=("Helvetica", 16, "bold")
        ).pack(anchor="w", pady=(5, 10))
        
        theme_var = ctk.StringVar(value=self.config_manager.get("theme", "system"))
        theme_combo = ctk.CTkComboBox(
            theme_frame,
            values=["system", "light", "dark"],
            variable=theme_var,
            width=200
        )
        theme_combo.pack(anchor="w", pady=5)
        
        # 버튼들
        button_frame = ctk.CTkFrame(settings_dialog, fg_color="transparent")
        button_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        ctk.CTkButton(
            button_frame,
            text="취소",
            command=settings_dialog.destroy,
            width=100,
            fg_color=THEME['error']
        ).pack(side="left")
        
        ctk.CTkButton(
            button_frame,
            text="저장",
            command=lambda: self._save_settings(settings_dialog, vars_dict, theme_var.get()),
            width=100,
            fg_color=THEME['primary']
        ).pack(side="right")
    
    def _save_settings(self, dialog, vars_dict, theme):
        """설정 저장"""
        try:
            # 설정 값 검증 및 저장
            for key, var in vars_dict.items():
                value = var.get()
                
                # 숫자 변환
                if key in ["context_size", "max_tokens", "top_k", "n_threads", "n_gpu_layers"]:
                    value = int(value)
                elif key in ["temperature", "top_p", "repeat_penalty"]:
                    value = float(value)
                
                self.config_manager.set(key, value)
            
            # 테마 설정
            self.config_manager.set("theme", theme)
            if theme != ctk.get_appearance_mode().lower():
                self.change_theme(theme)
            
            dialog.destroy()
            messagebox.showinfo("성공", "설정이 저장되었습니다.")
        except ValueError as e:
            messagebox.showerror("오류", "잘못된 값이 입력되었습니다.", parent=dialog)
        except Exception as e:
            messagebox.showerror("오류", f"설정 저장 중 오류가 발생했습니다:\n{str(e)}", parent=dialog)
    
    def change_theme(self, theme_mode):
        """테마 변경"""
        ctk.set_appearance_mode(theme_mode)
    
    def save_conversation(self):
        """대화 저장"""
        if not self.conversation:
            messagebox.showinfo("알림", "저장할 대화 내용이 없습니다.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="대화 저장",
            defaultextension=".json",
            filetypes=[("JSON 파일", "*.json"), ("텍스트 파일", "*.txt")],
            initialdir="conversations"
        )
        
        if not file_path:
            return
        
        try:
            if file_path.endswith('.json'):
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.conversation, f, ensure_ascii=False, indent=2)
            else:
                with open(file_path, 'w', encoding='utf-8') as f:
                    for msg in self.conversation:
                        role = "User" if msg["role"] == "user" else "Assistant"
                        f.write(f"{role}: {msg['content']}\n\n")
            
            messagebox.showinfo("성공", "대화가 저장되었습니다.")
        except Exception as e:
            messagebox.showerror("오류", f"대화 저장 중 오류가 발생했습니다:\n{str(e)}")
    
    def on_closing(self):
        """앱 종료 처리"""
        self.config_manager.save_config()
        self.quit()
        self.destroy()

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
        app = SimpleGGUFApp()
        app.protocol("WM_DELETE_WINDOW", app.on_closing)
        
        # 대화 저장 메뉴 추가
        import tkinter as tk
        menu_bar = tk.Menu(app)
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="대화 저장", command=app.save_conversation)
        file_menu.add_separator()
        file_menu.add_command(label="종료", command=app.on_closing)
        menu_bar.add_cascade(label="파일", menu=file_menu)
        app.configure(menu=menu_bar)
        
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