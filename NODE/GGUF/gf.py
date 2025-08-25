# -*- coding: utf-8 -*-
"""
GGUF 대화형 LLM 시스템 - 개선 버전
프롬프트 템플릿 및 대화 기능 강화
"""

import os
import sys
import json
import time
import threading
import customtkinter as ctk
from tkinter import filedialog, messagebox, scrolledtext
from pathlib import Path
from llama_cpp import Llama
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
    'accent': '#F39C12',
    'error': '#E74C3C',
    'success': '#2ECC71',
    'warning': '#F1C40F',
    'background': '#F5F5F5',
    'surface': '#FFFFFF',
    'text': '#2C3E50',
    'text_secondary': '#7F8C8D'
}

# 기본 프롬프트 템플릿
DEFAULT_PROMPTS = {
    "일반 대화": {
        "system": "당신은 친절하고 도움이 되는 AI 어시스턴트입니다. 사용자의 질문에 명확하고 정확한 답변을 제공합니다.",
        "format": "### 사용자: {user_input}\n### 어시스턴트:"
    },
    "코드 도우미": {
        "system": "당신은 전문적인 프로그래밍 도우미입니다. 코드를 작성하고 설명하며, 버그를 찾고 최적화를 제안합니다.",
        "format": "### 요청: {user_input}\n### 코드 및 설명:"
    },
    "번역": {
        "system": "당신은 전문 번역가입니다. 정확하고 자연스러운 번역을 제공합니다.",
        "format": "### 번역 요청: {user_input}\n### 번역 결과:"
    },
    "로그프레소 쿼리": {
        "system": "당신은 로그프레소 쿼리 전문가입니다. 자연어를 로그프레소 쿼리로 변환하고 쿼리를 최적화합니다.",
        "format": "### 자연어 질의: {user_input}\n### 로그프레소 쿼리:"
    }
}

# 디렉토리 생성
def create_app_directories():
    """애플리케이션 필요 디렉토리 생성"""
    dirs = ["models", "logs", "configs", "conversations", "prompts"]
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
            "recent_conversations": [],
            "default_prompt_template": "일반 대화",
            "custom_prompts": {},
            "system_prompt": "",
            "prompt_format": ""
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

# 프롬프트 관리 클래스
class PromptManager:
    """프롬프트 템플릿 관리"""
    
    def __init__(self, config_manager):
        self.config = config_manager
        self.prompt_file = "prompts/custom_prompts.json"
        self.prompts = self.load_prompts()
    
    def load_prompts(self):
        """프롬프트 파일 로드"""
        if os.path.exists(self.prompt_file):
            try:
                with open(self.prompt_file, 'r', encoding='utf-8') as f:
                    custom_prompts = json.load(f)
                    # 기본 프롬프트와 병합
                    all_prompts = DEFAULT_PROMPTS.copy()
                    all_prompts.update(custom_prompts)
                    return all_prompts
            except Exception as e:
                logger.error(f"프롬프트 파일 로드 오류: {str(e)}")
                return DEFAULT_PROMPTS.copy()
        else:
            return DEFAULT_PROMPTS.copy()
    
    def save_prompts(self):
        """커스텀 프롬프트 저장"""
        custom_prompts = {k: v for k, v in self.prompts.items() if k not in DEFAULT_PROMPTS}
        try:
            with open(self.prompt_file, 'w', encoding='utf-8') as f:
                json.dump(custom_prompts, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"프롬프트 파일 저장 오류: {str(e)}")
            return False
    
    def add_prompt(self, name, system, format_str):
        """새 프롬프트 추가"""
        self.prompts[name] = {"system": system, "format": format_str}
        self.save_prompts()
    
    def delete_prompt(self, name):
        """프롬프트 삭제"""
        if name in self.prompts and name not in DEFAULT_PROMPTS:
            del self.prompts[name]
            self.save_prompts()
            return True
        return False
    
    def get_prompt(self, name):
        """프롬프트 가져오기"""
        return self.prompts.get(name, DEFAULT_PROMPTS["일반 대화"])

# GGUF 모델 관리 클래스
class GGUFModel:
    """GGUF 모델 래퍼 클래스"""
    
    def __init__(self, config_manager):
        self.config = config_manager
        self.model = None
        self.model_path = ""
        self.stop_generation = False
    
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
        stream = kwargs.get("stream", False)
        
        self.stop_generation = False
        
        try:
            if stream:
                # 스트리밍 생성
                return self.model(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repeat_penalty=repeat_penalty,
                    echo=False,
                    stream=True
                )
            else:
                # 일반 생성
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
    
    def stop(self):
        """생성 중지"""
        self.stop_generation = True
    
    def is_loaded(self):
        """모델 로드 여부 확인"""
        return self.model is not None

# 메인 애플리케이션
class SimpleGGUFApp(ctk.CTk):
    """GGUF 대화형 LLM 시스템"""
    
    def __init__(self):
        super().__init__()
        
        # 앱 초기화
        self.title("GGUF 대화형 LLM 시스템")
        self.geometry("1200x800")
        
        # 디렉토리 생성
        create_app_directories()
        
        # 설정 관리자 초기화
        self.config_manager = ConfigManager()
        
        # 프롬프트 관리자 초기화
        self.prompt_manager = PromptManager(self.config_manager)
        
        # 테마 설정
        self.apply_theme()
        
        # 모델 초기화
        self.model = GGUFModel(self.config_manager)
        
        # 대화 이력
        self.conversation = []
        
        # 현재 프롬프트 템플릿
        self.current_prompt_template = self.config_manager.get("default_prompt_template", "일반 대화")
        
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
        top_frame = ctk.CTkFrame(self, height=80, corner_radius=0)
        top_frame.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        top_frame.grid_columnconfigure(1, weight=1)
        
        # 앱 제목
        app_title = ctk.CTkLabel(
            top_frame,
            text="GGUF 대화형 LLM 시스템",
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
        
        # 프롬프트 템플릿 선택
        prompt_frame = ctk.CTkFrame(top_frame, fg_color="transparent")
        prompt_frame.grid(row=1, column=0, columnspan=2, padx=20, pady=(0, 10), sticky="w")
        
        ctk.CTkLabel(
            prompt_frame,
            text="프롬프트 템플릿:",
            font=("Helvetica", 12)
        ).pack(side="left", padx=(0, 10))
        
        self.prompt_combo = ctk.CTkComboBox(
            prompt_frame,
            values=list(self.prompt_manager.prompts.keys()),
            width=200,
            command=self.on_prompt_template_change
        )
        self.prompt_combo.set(self.current_prompt_template)
        self.prompt_combo.pack(side="left", padx=(0, 10))
        
        ctk.CTkButton(
            prompt_frame,
            text="템플릿 관리",
            command=self.show_prompt_manager,
            width=100,
            height=30,
            font=("Helvetica", 12),
            fg_color=THEME['secondary'],
            hover_color=THEME['primary']
        ).pack(side="left")
        
        # 버튼들
        button_frame = ctk.CTkFrame(top_frame, fg_color="transparent")
        button_frame.grid(row=0, column=2, rowspan=2, padx=20, pady=15)
        
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
    
    def on_prompt_template_change(self, choice):
        """프롬프트 템플릿 변경"""
        self.current_prompt_template = choice
        self.config_manager.set("default_prompt_template", choice)
        logger.info(f"프롬프트 템플릿 변경: {choice}")
    
    def show_prompt_manager(self):
        """프롬프트 템플릿 관리자"""
        dialog = ctk.CTkToplevel(self)
        dialog.title("프롬프트 템플릿 관리")
        dialog.geometry("800x600")
        dialog.transient(self)
        dialog.grab_set()
        
        # 중앙 배치
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry(f'+{x}+{y}')
        
        # 프레임
        main_frame = ctk.CTkFrame(dialog)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # 템플릿 리스트
        list_frame = ctk.CTkFrame(main_frame)
        list_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        ctk.CTkLabel(
            list_frame,
            text="템플릿 목록",
            font=("Helvetica", 16, "bold")
        ).pack(pady=(0, 10))
        
        self.template_listbox = ctk.CTkScrollableFrame(list_frame, width=200)
        self.template_listbox.pack(fill="both", expand=True)
        
        # 템플릿 편집기
        edit_frame = ctk.CTkFrame(main_frame)
        edit_frame.pack(side="right", fill="both", expand=True)
        
        ctk.CTkLabel(
            edit_frame,
            text="템플릿 편집",
            font=("Helvetica", 16, "bold")
        ).pack(pady=(0, 10))
        
        # 템플릿 이름
        name_frame = ctk.CTkFrame(edit_frame, fg_color="transparent")
        name_frame.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(name_frame, text="이름:").pack(side="left")
        self.template_name_var = ctk.StringVar()
        ctk.CTkEntry(
            name_frame,
            textvariable=self.template_name_var,
            width=300
        ).pack(side="left", padx=(10, 0))
        
        # 시스템 프롬프트
        ctk.CTkLabel(edit_frame, text="시스템 프롬프트:").pack(anchor="w")
        self.system_prompt_text = ctk.CTkTextbox(
            edit_frame,
            height=150,
            wrap="word"
        )
        self.system_prompt_text.pack(fill="x", pady=(5, 10))
        
        # 형식 문자열
        ctk.CTkLabel(edit_frame, text="프롬프트 형식 ({user_input}을 사용):").pack(anchor="w")
        self.format_text = ctk.CTkTextbox(
            edit_frame,
            height=100,
            wrap="word"
        )
        self.format_text.pack(fill="x", pady=(5, 10))
        
        # 버튼들
        button_frame = ctk.CTkFrame(edit_frame, fg_color="transparent")
        button_frame.pack(fill="x", pady=(10, 0))
        
        ctk.CTkButton(
            button_frame,
            text="새 템플릿",
            command=lambda: self._new_template(),
            width=100,
            fg_color=THEME['success']
        ).pack(side="left", padx=(0, 5))
        
        ctk.CTkButton(
            button_frame,
            text="저장",
            command=lambda: self._save_template(),
            width=100,
            fg_color=THEME['primary']
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            button_frame,
            text="삭제",
            command=lambda: self._delete_template(),
            width=100,
            fg_color=THEME['error']
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            button_frame,
            text="닫기",
            command=dialog.destroy,
            width=100
        ).pack(side="right")
        
        # 템플릿 목록 업데이트
        self._update_template_list()
    
    def _update_template_list(self):
        """템플릿 목록 업데이트"""
        # 기존 위젯 제거
        for widget in self.template_listbox.winfo_children():
            widget.destroy()
        
        # 템플릿 버튼 생성
        for name in self.prompt_manager.prompts:
            is_default = name in DEFAULT_PROMPTS
            btn = ctk.CTkButton(
                self.template_listbox,
                text=f"{name} {'(기본)' if is_default else ''}",
                command=lambda n=name: self._load_template(n),
                width=180,
                fg_color=THEME['secondary'] if is_default else THEME['primary']
            )
            btn.pack(pady=2)
    
    def _load_template(self, name):
        """템플릿 로드"""
        template = self.prompt_manager.get_prompt(name)
        self.template_name_var.set(name)
        self.system_prompt_text.delete("0.0", "end")
        self.system_prompt_text.insert("0.0", template.get("system", ""))
        self.format_text.delete("0.0", "end")
        self.format_text.insert("0.0", template.get("format", ""))
    
    def _new_template(self):
        """새 템플릿"""
        self.template_name_var.set("")
        self.system_prompt_text.delete("0.0", "end")
        self.format_text.delete("0.0", "end")
    
    def _save_template(self):
        """템플릿 저장"""
        name = self.template_name_var.get().strip()
        if not name:
            messagebox.showerror("오류", "템플릿 이름을 입력하세요.")
            return
        
        if name in DEFAULT_PROMPTS:
            messagebox.showerror("오류", "기본 템플릿은 수정할 수 없습니다.")
            return
        
        system = self.system_prompt_text.get("0.0", "end").strip()
        format_str = self.format_text.get("0.0", "end").strip()
        
        self.prompt_manager.add_prompt(name, system, format_str)
        self._update_template_list()
        
        # 콤보박스 업데이트
        self.prompt_combo.configure(values=list(self.prompt_manager.prompts.keys()))
        
        messagebox.showinfo("성공", "템플릿이 저장되었습니다.")
    
    def _delete_template(self):
        """템플릿 삭제"""
        name = self.template_name_var.get().strip()
        if not name:
            messagebox.showerror("오류", "삭제할 템플릿을 선택하세요.")
            return
        
        if self.prompt_manager.delete_prompt(name):
            self._update_template_list()
            self._new_template()
            
            # 콤보박스 업데이트
            self.prompt_combo.configure(values=list(self.prompt_manager.prompts.keys()))
            if self.current_prompt_template == name:
                self.prompt_combo.set("일반 대화")
                self.on_prompt_template_change("일반 대화")
            
            messagebox.showinfo("성공", "템플릿이 삭제되었습니다.")
        else:
            messagebox.showerror("오류", "기본 템플릿은 삭제할 수 없습니다.")
    
    def add_user_message(self, message):
        """사용자 메시지 추가"""
        container = ctk.CTkFrame(self.chat_frame, fg_color="#D6EAF8", corner_radius=10)
        container.pack(fill="x", padx=10, pady=5, anchor="e")
        
        label = ctk.CTkLabel(
            container,
            text=message,
            font=("Helvetica", 14),
            text_color=THEME['text'],
            wraplength=750,
            justify="left"
        )
        label.pack(padx=15, pady=10, anchor="w")
        
        self.conversation.append({"role": "user", "content": message})
        self.after(100, lambda: self.chat_frame._parent_canvas.yview_moveto(1.0))
    
    def add_assistant_message(self, message):
        """어시스턴트 메시지 추가"""
        container = ctk.CTkFrame(self.chat_frame, fg_color="#E8F6F3", corner_radius=10)
        container.pack(fill="x", padx=10, pady=5, anchor="w")
        
        self.assistant_label = ctk.CTkLabel(
            container,
            text=message,
            font=("Helvetica", 14),
            text_color=THEME['text'],
            wraplength=750,
            justify="left"
        )
        self.assistant_label.pack(padx=15, pady=10, anchor="w")
        
        self.conversation.append({"role": "assistant", "content": message})
        self.after(100, lambda: self.chat_frame._parent_canvas.yview_moveto(1.0))
        
        return self.assistant_label
    
    def update_assistant_message(self, label, message):
        """어시스턴트 메시지 업데이트 (스트리밍용)"""
        label.configure(text=message)
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
            
            # 빈 메시지로 시작
            self.after(100, lambda: self.add_assistant_message(""))
            current_label = self.assistant_label
            
            # 스트리밍 응답 생성
            full_response = ""
            stream = self.model.generate(prompt, stream=True)
            
            for output in stream:
                if self.model.stop_generation:
                    break
                
                # 텍스트 추출
                if isinstance(output, dict) and "choices" in output:
                    token = output["choices"][0]["text"]
                else:
                    token = str(output)
                
                full_response += token
                
                # UI 업데이트 (너무 자주 하지 않도록)
                if len(full_response) % 5 == 0 or token in ["\n", ".", "!", "?"]:
                    self.after(100, lambda r=full_response: self.update_assistant_message(current_label, r))
            
            # 최종 업데이트
            self.after(100, lambda: self._finalize_response(full_response, current_label))
            
        except Exception as e:
            error_message = f"응답 생성 중 오류가 발생했습니다: {str(e)}"
            logger.error(error_message)
            self.after(100, lambda: self._handle_response_error(error_message))
    
    def _build_prompt(self, user_message):
        """프롬프트 구성"""
        template = self.prompt_manager.get_prompt(self.current_prompt_template)
        system_prompt = template.get("system", "")
        format_str = template.get("format", "{user_input}")
        
        # 대화 컨텍스트 구성
        context_window = self.config_manager.get("context_size", 2048)
        max_context_length = int(context_window * 0.7)  # 70% 사용
        
        # 시스템 프롬프트 포함
        prompt = f"{system_prompt}\n\n" if system_prompt else ""
        
        # 최근 대화 내역 추가
        recent_conversation = []
        total_length = len(prompt)
        
        for msg in reversed(self.conversation[:-1]):  # 마지막 사용자 메시지 제외
            msg_text = f"{msg['role'].capitalize()}: {msg['content']}\n"
            msg_length = len(msg_text)
            
            if total_length + msg_length > max_context_length:
                break
            
            recent_conversation.insert(0, msg_text)
            total_length += msg_length
        
        # 대화 내역 추가
        if recent_conversation:
            prompt += "".join(recent_conversation) + "\n"
        
        # 현재 메시지 추가
        prompt += format_str.replace("{user_input}", user_message)
        
        return prompt
    
    def _finalize_response(self, response, label):
        """응답 최종 처리"""
        # 대화 기록 업데이트
        if self.conversation and self.conversation[-1]["role"] == "assistant":
            self.conversation[-1]["content"] = response.strip()
        
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
        self.model.stop()
        self.is_generating = False
        self.send_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
    
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
        self.add_assistant_message("대화가 초기화되었습니다. 새로운 대화를 시작해보세요!")
    
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
        ).pack(anchor="w", padx=10, pady=(10, 10))
        
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
            frame.pack(fill="x", padx=10, pady=5)
            
            ctk.CTkLabel(
                frame,
                text=label_text,
                width=150
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
        ).pack(anchor="w", padx=10, pady=(10, 10))
        
        theme_var = ctk.StringVar(value=self.config_manager.get("theme", "system"))
        theme_combo = ctk.CTkComboBox(
            theme_frame,
            values=["system", "light", "dark"],
            variable=theme_var,
            width=200
        )
        theme_combo.pack(anchor="w", padx=10, pady=5)
        
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
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conversation_data = {
                "timestamp": timestamp,
                "model": os.path.basename(self.model.model_path),
                "prompt_template": self.current_prompt_template,
                "conversation": self.conversation
            }
            
            if file_path.endswith('.json'):
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(conversation_data, f, ensure_ascii=False, indent=2)
            else:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"대화 저장 시간: {timestamp}\n")
                    f.write(f"모델: {conversation_data['model']}\n")
                    f.write(f"프롬프트 템플릿: {conversation_data['prompt_template']}\n")
                    f.write("=" * 50 + "\n\n")
                    
                    for msg in self.conversation:
                        role = "User" if msg["role"] == "user" else "Assistant"
                        f.write(f"{role}: {msg['content']}\n\n")
            
            messagebox.showinfo("성공", "대화가 저장되었습니다.")
        except Exception as e:
            messagebox.showerror("오류", f"대화 저장 중 오류가 발생했습니다:\n{str(e)}")
    
    def load_conversation(self):
        """대화 불러오기"""
        file_path = filedialog.askopenfilename(
            title="대화 불러오기",
            filetypes=[("JSON 파일", "*.json")],
            initialdir="conversations"
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 대화 초기화
            self.conversation = []
            for widget in self.chat_frame.winfo_children():
                widget.destroy()
            
            # 대화 복원
            for msg in data.get("conversation", []):
                if msg["role"] == "user":
                    self.add_user_message(msg["content"])
                else:
                    self.add_assistant_message(msg["content"])
            
            # 프롬프트 템플릿 복원
            template = data.get("prompt_template", "일반 대화")
            if template in self.prompt_manager.prompts:
                self.prompt_combo.set(template)
                self.on_prompt_template_change(template)
            
            messagebox.showinfo("성공", "대화를 불러왔습니다.")
        except Exception as e:
            messagebox.showerror("오류", f"대화 불러오기 중 오류가 발생했습니다:\n{str(e)}")
    
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
        
        # 메뉴바 추가
        import tkinter as tk
        menu_bar = tk.Menu(app)
        
        # 파일 메뉴
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="대화 저장", command=app.save_conversation)
        file_menu.add_command(label="대화 불러오기", command=app.load_conversation)
        file_menu.add_separator()
        file_menu.add_command(label="종료", command=app.on_closing)
        menu_bar.add_cascade(label="파일", menu=file_menu)
        
        # 편집 메뉴
        edit_menu = tk.Menu(menu_bar, tearoff=0)
        edit_menu.add_command(label="프롬프트 템플릿 관리", command=app.show_prompt_manager)
        edit_menu.add_command(label="설정", command=app.show_settings)
        menu_bar.add_cascade(label="편집", menu=edit_menu)
        
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