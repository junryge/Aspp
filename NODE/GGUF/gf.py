# -*- coding: utf-8 -*-
"""
Modern GGUF Chat Application - 2025
최신 기술 트렌드를 반영한 GGUF 대화형 AI 시스템
"""

import os
import sys
import json
import time
import threading
import queue
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any, Generator
import logging
from dataclasses import dataclass, field
from enum import Enum

# UI 라이브러리
import customtkinter as ctk
from tkinter import filedialog, messagebox, scrolledtext
import tkinter as tk

# GGUF 모델 라이브러리
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print("⚠️ llama-cpp-python이 설치되지 않았습니다.")
    print("설치: pip install llama-cpp-python")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gguf_chat.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 최신 프롬프트 템플릿 정의
class PromptTemplate(Enum):
    """2025년 최신 프롬프트 템플릿"""
    
    CHATML = """<|im_start|>system
{system_prompt}<|im_end|>
{chat_history}<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
"""
    
    LLAMA3 = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|>
{chat_history}<|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    ALPACA = """### System:
{system_prompt}

{chat_history}### Human:
{user_message}

### Assistant:
"""
    
    VICUNA = """A chat between a curious user and an artificial intelligence assistant.

{system_prompt}

{chat_history}USER: {user_message}
ASSISTANT: """

@dataclass
class ModelConfig:
    """모델 설정 데이터 클래스"""
    model_path: str = ""
    context_size: int = 4096
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    n_threads: int = 4
    n_gpu_layers: int = -1  # -1은 자동 감지
    seed: int = -1
    prompt_template: str = "CHATML"
    system_prompt: str = "당신은 친절하고 도움이 되는 AI 어시스턴트입니다. 사용자의 질문에 정확하고 유용한 답변을 제공합니다."
    
@dataclass
class ChatMessage:
    """채팅 메시지 데이터 클래스"""
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    tokens: int = 0

class StreamingResponse:
    """스트리밍 응답 처리 클래스"""
    
    def __init__(self, model, prompt, config: ModelConfig):
        self.model = model
        self.prompt = prompt
        self.config = config
        self.response_queue = queue.Queue()
        self.is_generating = True
        
    def generate(self) -> Generator[str, None, None]:
        """스트리밍 생성"""
        try:
            stream = self.model(
                self.prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                repeat_penalty=self.config.repeat_penalty,
                stream=True,
                stop=["<|im_end|>", "<|eot_id|>", "</s>", "###", "\n\n\n"]
            )
            
            for output in stream:
                if not self.is_generating:
                    break
                    
                token = output['choices'][0]['text']
                yield token
                
        except Exception as e:
            logger.error(f"스트리밍 생성 오류: {str(e)}")
            yield f"\n[오류: {str(e)}]"
    
    def stop(self):
        """생성 중지"""
        self.is_generating = False

class GGUFModelManager:
    """GGUF 모델 관리자 - 최신 기능 포함"""
    
    def __init__(self):
        self.model: Optional[Llama] = None
        self.config = ModelConfig()
        self.is_loaded = False
        
    def load_model(self, model_path: str, config: ModelConfig) -> bool:
        """모델 로드 with 최신 설정"""
        try:
            # 기존 모델 정리
            if self.model:
                del self.model
                self.model = None
                
            # GPU 자동 감지
            n_gpu_layers = config.n_gpu_layers
            if n_gpu_layers == -1:
                try:
                    # CUDA 사용 가능 여부 확인
                    import torch
                    if torch.cuda.is_available():
                        n_gpu_layers = 999  # 모든 레이어를 GPU로
                        logger.info(f"CUDA 감지됨: {torch.cuda.get_device_name(0)}")
                except:
                    n_gpu_layers = 0
                    
            # 모델 로드
            self.model = Llama(
                model_path=model_path,
                n_ctx=config.context_size,
                n_threads=config.n_threads,
                n_gpu_layers=n_gpu_layers,
                seed=config.seed,
                verbose=False,
                use_mmap=True,  # 메모리 매핑 사용
                use_mlock=False,  # 메모리 락 비활성화
                n_batch=512,  # 배치 크기
                rope_scaling_type=1,  # RoPE 스케일링
                mul_mat_q=True,  # 양자화된 행렬 곱셈
            )
            
            self.config = config
            self.is_loaded = True
            logger.info(f"모델 로드 성공: {os.path.basename(model_path)}")
            return True
            
        except Exception as e:
            logger.error(f"모델 로드 실패: {str(e)}")
            raise
            
    def build_prompt(self, messages: List[ChatMessage], template: PromptTemplate) -> str:
        """최신 프롬프트 템플릿 빌드"""
        chat_history = ""
        
        # 템플릿에 따른 대화 기록 포맷팅
        if template == PromptTemplate.CHATML:
            for msg in messages[:-1]:  # 마지막 메시지 제외
                if msg.role == "user":
                    chat_history += f"<|im_start|>user\n{msg.content}<|im_end|>\n"
                elif msg.role == "assistant":
                    chat_history += f"<|im_start|>assistant\n{msg.content}<|im_end|>\n"
                    
        elif template == PromptTemplate.LLAMA3:
            for msg in messages[:-1]:
                if msg.role == "user":
                    chat_history += f"<|start_header_id|>user<|end_header_id|>\n\n{msg.content}<|eot_id|>"
                elif msg.role == "assistant":
                    chat_history += f"<|start_header_id|>assistant<|end_header_id|>\n\n{msg.content}<|eot_id|>"
                    
        elif template == PromptTemplate.ALPACA:
            for msg in messages[:-1]:
                if msg.role == "user":
                    chat_history += f"### Human:\n{msg.content}\n\n"
                elif msg.role == "assistant":
                    chat_history += f"### Assistant:\n{msg.content}\n\n"
                    
        elif template == PromptTemplate.VICUNA:
            for msg in messages[:-1]:
                if msg.role == "user":
                    chat_history += f"USER: {msg.content}\n"
                elif msg.role == "assistant":
                    chat_history += f"ASSISTANT: {msg.content}\n"
        
        # 프롬프트 생성
        prompt = template.value.format(
            system_prompt=self.config.system_prompt,
            chat_history=chat_history,
            user_message=messages[-1].content if messages else ""
        )
        
        return prompt
        
    def generate_streaming(self, messages: List[ChatMessage]) -> StreamingResponse:
        """스트리밍 응답 생성"""
        if not self.is_loaded:
            raise RuntimeError("모델이 로드되지 않았습니다.")
            
        # 프롬프트 템플릿 선택
        template = PromptTemplate[self.config.prompt_template]
        prompt = self.build_prompt(messages, template)
        
        # 스트리밍 응답 객체 생성
        return StreamingResponse(self.model, prompt, self.config)

class ModernGGUFChat(ctk.CTk):
    """현대적인 GGUF 채팅 애플리케이션"""
    
    def __init__(self):
        super().__init__()
        
        # 기본 설정
        self.title("GGUF Chat AI - 2025 Edition")
        self.geometry("1200x800")
        
        # 테마 설정
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # 컴포넌트 초기화
        self.model_manager = GGUFModelManager()
        self.messages: List[ChatMessage] = []
        self.current_streaming: Optional[StreamingResponse] = None
        
        # UI 구성
        self.setup_ui()
        
        # 초기 메시지
        self.add_message("assistant", "안녕하세요! 👋 GGUF 모델을 로드하고 대화를 시작해보세요.")
        
    def setup_ui(self):
        """UI 구성"""
        # 메인 컨테이너
        self.grid_columnconfigure(0, weight=0)  # 사이드바
        self.grid_columnconfigure(1, weight=1)  # 메인 영역
        self.grid_rowconfigure(0, weight=1)
        
        # 사이드바
        self.setup_sidebar()
        
        # 메인 영역
        self.setup_main_area()
        
    def setup_sidebar(self):
        """사이드바 구성"""
        sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_rowconfigure(10, weight=1)
        
        # 타이틀
        title = ctk.CTkLabel(
            sidebar,
            text="GGUF Chat AI",
            font=("Arial", 24, "bold")
        )
        title.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        # 모델 정보
        self.model_info = ctk.CTkLabel(
            sidebar,
            text="모델: 로드되지 않음",
            font=("Arial", 12),
            text_color="gray"
        )
        self.model_info.grid(row=1, column=0, padx=20, pady=(0, 20))
        
        # 버튼들
        ctk.CTkButton(
            sidebar,
            text="모델 로드",
            command=self.load_model,
            height=40
        ).grid(row=2, column=0, padx=20, pady=5, sticky="ew")
        
        ctk.CTkButton(
            sidebar,
            text="설정",
            command=self.show_settings,
            height=40
        ).grid(row=3, column=0, padx=20, pady=5, sticky="ew")
        
        # 프롬프트 템플릿 선택
        ctk.CTkLabel(sidebar, text="프롬프트 템플릿:").grid(row=4, column=0, padx=20, pady=(20, 5), sticky="w")
        
        self.template_var = ctk.StringVar(value="CHATML")
        template_menu = ctk.CTkOptionMenu(
            sidebar,
            values=["CHATML", "LLAMA3", "ALPACA", "VICUNA"],
            variable=self.template_var,
            command=self.on_template_change
        )
        template_menu.grid(row=5, column=0, padx=20, pady=5, sticky="ew")
        
        # 온도 슬라이더
        ctk.CTkLabel(sidebar, text="Temperature:").grid(row=6, column=0, padx=20, pady=(20, 5), sticky="w")
        
        self.temp_slider = ctk.CTkSlider(
            sidebar,
            from_=0,
            to=1,
            number_of_steps=20,
            command=self.on_temp_change
        )
        self.temp_slider.set(0.7)
        self.temp_slider.grid(row=7, column=0, padx=20, pady=5, sticky="ew")
        
        self.temp_label = ctk.CTkLabel(sidebar, text="0.7")
        self.temp_label.grid(row=8, column=0, padx=20, pady=(0, 20))
        
        # 대화 관리 버튼들
        ctk.CTkButton(
            sidebar,
            text="대화 초기화",
            command=self.clear_chat,
            height=35,
            fg_color="red",
            hover_color="darkred"
        ).grid(row=9, column=0, padx=20, pady=5, sticky="ew")
        
    def setup_main_area(self):
        """메인 영역 구성"""
        main_frame = ctk.CTkFrame(self, corner_radius=0)
        main_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 0))
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        
        # 채팅 영역
        self.chat_frame = ctk.CTkScrollableFrame(main_frame)
        self.chat_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=(20, 10))
        
        # 입력 영역
        input_frame = ctk.CTkFrame(main_frame)
        input_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 20))
        input_frame.grid_columnconfigure(0, weight=1)
        
        # 입력 텍스트박스
        self.input_text = ctk.CTkTextbox(
            input_frame,
            height=100,
            wrap="word",
            font=("Arial", 14)
        )
        self.input_text.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.input_text.bind("<Control-Return>", lambda e: self.send_message())
        
        # 버튼 프레임
        btn_frame = ctk.CTkFrame(input_frame)
        btn_frame.grid(row=0, column=1, sticky="ns")
        
        self.send_btn = ctk.CTkButton(
            btn_frame,
            text="전송",
            command=self.send_message,
            width=100,
            height=40,
            state="disabled"
        )
        self.send_btn.pack(pady=(0, 5))
        
        self.stop_btn = ctk.CTkButton(
            btn_frame,
            text="중지",
            command=self.stop_generation,
            width=100,
            height=40,
            fg_color="orange",
            hover_color="darkorange",
            state="disabled"
        )
        self.stop_btn.pack()
        
    def add_message(self, role: str, content: str, streaming=False):
        """메시지 추가"""
        # 메시지 컨테이너
        msg_frame = ctk.CTkFrame(self.chat_frame, corner_radius=10)
        
        if role == "user":
            msg_frame.configure(fg_color=("gray85", "gray25"))
            msg_frame.pack(anchor="e", padx=(100, 10), pady=5, fill="x")
        else:
            msg_frame.configure(fg_color=("gray90", "gray20"))
            msg_frame.pack(anchor="w", padx=(10, 100), pady=5, fill="x")
        
        # 역할 라벨
        role_label = ctk.CTkLabel(
            msg_frame,
            text="You" if role == "user" else "AI",
            font=("Arial", 12, "bold"),
            text_color=("gray40", "gray60")
        )
        role_label.pack(anchor="w", padx=15, pady=(10, 0))
        
        # 메시지 라벨
        msg_label = ctk.CTkLabel(
            msg_frame,
            text=content,
            font=("Arial", 14),
            wraplength=600,
            justify="left"
        )
        msg_label.pack(anchor="w", padx=15, pady=(5, 10))
        
        # 메시지 저장
        if not streaming:
            self.messages.append(ChatMessage(role=role, content=content))
        
        # 스크롤 다운
        self.chat_frame._parent_canvas.yview_moveto(1.0)
        
        return msg_label
        
    def load_model(self):
        """모델 로드"""
        filepath = filedialog.askopenfilename(
            title="GGUF 모델 선택",
            filetypes=[("GGUF Files", "*.gguf"), ("All Files", "*.*")]
        )
        
        if not filepath:
            return
            
        # 로딩 다이얼로그
        self.show_loading("모델 로드 중...")
        
        # 별도 스레드에서 로드
        threading.Thread(
            target=self._load_model_thread,
            args=(filepath,),
            daemon=True
        ).start()
        
    def _load_model_thread(self, filepath):
        """모델 로드 스레드"""
        try:
            config = ModelConfig(
                model_path=filepath,
                prompt_template=self.template_var.get(),
                temperature=self.temp_slider.get()
            )
            
            self.model_manager.load_model(filepath, config)
            
            # UI 업데이트
            self.after(0, self._on_model_loaded, filepath)
            
        except Exception as e:
            self.after(0, self._on_model_error, str(e))
            
    def _on_model_loaded(self, filepath):
        """모델 로드 완료"""
        self.hide_loading()
        model_name = os.path.basename(filepath)
        self.model_info.configure(text=f"모델: {model_name}")
        self.send_btn.configure(state="normal")
        messagebox.showinfo("성공", "모델이 성공적으로 로드되었습니다!")
        
    def _on_model_error(self, error):
        """모델 로드 오류"""
        self.hide_loading()
        messagebox.showerror("오류", f"모델 로드 실패:\n{error}")
        
    def send_message(self):
        """메시지 전송"""
        content = self.input_text.get("1.0", "end").strip()
        if not content or not self.model_manager.is_loaded:
            return
            
        # 입력 초기화
        self.input_text.delete("1.0", "end")
        
        # 사용자 메시지 추가
        self.add_message("user", content)
        
        # UI 상태 변경
        self.send_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        
        # 스트리밍 응답 생성
        threading.Thread(
            target=self._generate_response,
            daemon=True
        ).start()
        
    def _generate_response(self):
        """응답 생성"""
        try:
            # AI 메시지 라벨 생성
            msg_label = None
            full_response = ""
            
            # 스트리밍 응답 시작
            self.current_streaming = self.model_manager.generate_streaming(self.messages)
            
            for token in self.current_streaming.generate():
                full_response += token
                
                # UI 업데이트
                if msg_label is None:
                    self.after(0, lambda: setattr(self, '_temp_label', 
                        self.add_message("assistant", token, streaming=True)))
                    time.sleep(0.1)  # UI 생성 대기
                    msg_label = getattr(self, '_temp_label', None)
                else:
                    self.after(0, lambda t=full_response: msg_label.configure(text=t))
                    
            # 최종 메시지 저장
            self.messages.append(ChatMessage(role="assistant", content=full_response))
            
        except Exception as e:
            error_msg = f"오류 발생: {str(e)}"
            self.after(0, lambda: self.add_message("assistant", error_msg))
            
        finally:
            # UI 상태 복원
            self.after(0, self._reset_ui_state)
            
    def stop_generation(self):
        """생성 중지"""
        if self.current_streaming:
            self.current_streaming.stop()
            
    def _reset_ui_state(self):
        """UI 상태 초기화"""
        self.send_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.current_streaming = None
        
    def clear_chat(self):
        """대화 초기화"""
        if messagebox.askyesno("확인", "대화를 초기화하시겠습니까?"):
            self.messages.clear()
            for widget in self.chat_frame.winfo_children():
                widget.destroy()
            self.add_message("assistant", "대화가 초기화되었습니다. 새로운 대화를 시작해보세요!")
            
    def show_settings(self):
        """설정 창"""
        settings_window = ctk.CTkToplevel(self)
        settings_window.title("설정")
        settings_window.geometry("600x700")
        settings_window.transient(self)
        settings_window.grab_set()
        
        # 설정 항목들
        settings = [
            ("Context Size", "context_size", 128, 32768, 4096),
            ("Max Tokens", "max_tokens", 128, 4096, 2048),
            ("Top K", "top_k", 1, 100, 40),
            ("Repeat Penalty", "repeat_penalty", 0.5, 2.0, 1.1),
            ("Threads", "n_threads", 1, 32, 4),
            ("GPU Layers", "n_gpu_layers", -1, 100, -1),
        ]
        
        row = 0
        self.setting_vars = {}
        
        for label, key, min_val, max_val, default in settings:
            ctk.CTkLabel(settings_window, text=f"{label}:").grid(
                row=row, column=0, padx=20, pady=10, sticky="w"
            )
            
            if isinstance(min_val, float):
                var = ctk.DoubleVar(value=getattr(self.model_manager.config, key, default))
            else:
                var = ctk.IntVar(value=getattr(self.model_manager.config, key, default))
                
            self.setting_vars[key] = var
            
            if key in ["context_size", "max_tokens", "n_threads", "n_gpu_layers", "top_k"]:
                spinbox = ctk.CTkEntry(settings_window, textvariable=var, width=150)
                spinbox.grid(row=row, column=1, padx=20, pady=10)
            else:
                slider = ctk.CTkSlider(
                    settings_window,
                    from_=min_val,
                    to=max_val,
                    variable=var,
                    width=200
                )
                slider.grid(row=row, column=1, padx=20, pady=10)
                
                value_label = ctk.CTkLabel(settings_window, text=f"{var.get():.2f}")
                value_label.grid(row=row, column=2, padx=10, pady=10)
                
                slider.configure(command=lambda v, l=value_label, var=var: l.configure(text=f"{var.get():.2f}"))
                
            row += 1
            
        # System Prompt
        ctk.CTkLabel(settings_window, text="System Prompt:").grid(
            row=row, column=0, columnspan=3, padx=20, pady=(20, 5), sticky="w"
        )
        row += 1
        
        self.system_prompt_text = ctk.CTkTextbox(settings_window, height=150, width=550)
        self.system_prompt_text.grid(row=row, column=0, columnspan=3, padx=20, pady=5)
        self.system_prompt_text.insert("1.0", self.model_manager.config.system_prompt)
        row += 1
        
        # 버튼들
        btn_frame = ctk.CTkFrame(settings_window)
        btn_frame.grid(row=row, column=0, columnspan=3, pady=20)
        
        ctk.CTkButton(
            btn_frame,
            text="저장",
            command=lambda: self.save_settings(settings_window)
        ).pack(side="left", padx=10)
        
        ctk.CTkButton(
            btn_frame,
            text="취소",
            command=settings_window.destroy
        ).pack(side="left")
        
    def save_settings(self, window):
        """설정 저장"""
        try:
            # 설정 업데이트
            for key, var in self.setting_vars.items():
                setattr(self.model_manager.config, key, var.get())
                
            self.model_manager.config.system_prompt = self.system_prompt_text.get("1.0", "end").strip()
            self.model_manager.config.prompt_template = self.template_var.get()
            self.model_manager.config.temperature = self.temp_slider.get()
            
            messagebox.showinfo("성공", "설정이 저장되었습니다!")
            window.destroy()
            
        except Exception as e:
            messagebox.showerror("오류", f"설정 저장 실패:\n{str(e)}")
            
    def on_template_change(self, value):
        """템플릿 변경"""
        if self.model_manager.is_loaded:
            self.model_manager.config.prompt_template = value
            
    def on_temp_change(self, value):
        """온도 변경"""
        self.temp_label.configure(text=f"{value:.2f}")
        if self.model_manager.is_loaded:
            self.model_manager.config.temperature = value
            
    def show_loading(self, message):
        """로딩 표시"""
        self.loading_window = ctk.CTkToplevel(self)
        self.loading_window.title("로딩")
        self.loading_window.geometry("300x150")
        self.loading_window.transient(self)
        self.loading_window.grab_set()
        
        # 중앙 정렬
        self.loading_window.update_idletasks()
        x = (self.loading_window.winfo_screenwidth() // 2) - 150
        y = (self.loading_window.winfo_screenheight() // 2) - 75
        self.loading_window.geometry(f"+{x}+{y}")
        
        ctk.CTkLabel(
            self.loading_window,
            text=message,
            font=("Arial", 16)
        ).pack(pady=30)
        
        progress = ctk.CTkProgressBar(self.loading_window, mode="indeterminate")
        progress.pack(padx=40)
        progress.start()
        
    def hide_loading(self):
        """로딩 숨기기"""
        if hasattr(self, 'loading_window'):
            self.loading_window.destroy()

def main():
    """메인 함수"""
    if not LLAMA_CPP_AVAILABLE:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Error",
            "llama-cpp-python is not installed.\n\n"
            "Please install it using:\n"
            "pip install llama-cpp-python"
        )
        return
        
    # 필요한 디렉토리 생성
    for dir_name in ["models", "logs", "exports"]:
        Path(dir_name).mkdir(exist_ok=True)
        
    # 앱 실행
    app = ModernGGUFChat()
    app.mainloop()

if __name__ == "__main__":
    main()