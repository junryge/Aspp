# -*- coding: utf-8 -*-
"""
GGUF 대화 시스템 - 폐쇄망용 간소화 버전
CTransformers 기반, RAG/LangChain 제거
"""

import os
import sys
import json
import threading
import customtkinter as ctk
from tkinter import filedialog, messagebox
import logging
from datetime import datetime

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

class SimpleConversationApp(ctk.CTk):
    """CTransformers 기반 간소화 대화 애플리케이션"""
    
    def __init__(self):
        super().__init__()
        
        # 앱 초기화
        self.title("GGUF 대화 시스템 - 간소화 버전")
        self.geometry("900x650")
        
        # 테마 설정
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")
        
        # 모델 초기화
        self.model = None
        self.model_path = ""
        
        # 대화 이력
        self.conversation = []
        
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
        top_frame = ctk.CTkFrame(self, height=50)
        top_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        top_frame.grid_columnconfigure(1, weight=1)
        
        # 모델 로드 버튼
        self.load_button = ctk.CTkButton(
            top_frame,
            text="모델 로드",
            command=self.load_model,
            width=100,
            height=35
        )
        self.load_button.grid(row=0, column=0, padx=5, pady=5)
        
        # 모델 정보
        self.model_info_label = ctk.CTkLabel(
            top_frame,
            text="모델: 없음",
            font=("Arial", 12)
        )
        self.model_info_label.grid(row=0, column=1, padx=10, pady=5, sticky="w")
        
        # 대화 초기화 버튼
        self.clear_button = ctk.CTkButton(
            top_frame,
            text="대화 초기화",
            command=self.clear_conversation,
            width=100,
            height=35,
            fg_color=THEME['error']
        )
        self.clear_button.grid(row=0, column=2, padx=5, pady=5)
    
    def setup_chat_area(self):
        """대화 영역 구성"""
        # 대화 표시 영역
        self.chat_frame = ctk.CTkScrollableFrame(self)
        self.chat_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        
        # 초기 메시지
        self.add_message("시스템", "GGUF 모델을 로드한 후 대화를 시작하세요.", is_user=False)
    
    def setup_input_area(self):
        """입력 영역 구성"""
        input_frame = ctk.CTkFrame(self)
        input_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        input_frame.grid_columnconfigure(0, weight=1)
        
        # 입력창
        self.input_box = ctk.CTkTextbox(
            input_frame,
            height=60,
            wrap="word",
            font=("Arial", 12)
        )
        self.input_box.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.input_box.bind("<Return>", self.handle_return)
        
        # 버튼 프레임
        button_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
        button_frame.grid(row=0, column=1, sticky="ns")
        
        # 전송 버튼
        self.send_button = ctk.CTkButton(
            button_frame,
            text="전송",
            command=self.send_message,
            width=80,
            height=30,
            state="disabled"
        )
        self.send_button.pack(pady=(0, 5))
        
        # 중지 버튼
        self.stop_button = ctk.CTkButton(
            button_frame,
            text="중지",
            command=self.stop_generating,
            width=80,
            height=30,
            fg_color=THEME['error'],
            state="disabled"
        )
        self.stop_button.pack()
    
    def add_message(self, sender, message, is_user=True):
        """메시지 추가"""
        # 메시지 컨테이너
        container = ctk.CTkFrame(
            self.chat_frame,
            fg_color="#E3F2FD" if is_user else "#F5F5F5"
        )
        container.pack(fill="x", padx=5, pady=3)
        
        # 발신자 라벨
        sender_label = ctk.CTkLabel(
            container,
            text=sender,
            font=("Arial", 10, "bold"),
            text_color="#666666"
        )
        sender_label.pack(anchor="w", padx=10, pady=(5, 0))
        
        # 메시지 라벨
        message_label = ctk.CTkLabel(
            container,
            text=message,
            font=("Arial", 12),
            justify="left",
            wraplength=750
        )
        message_label.pack(anchor="w", padx=10, pady=(0, 5))
        
        # 대화 이력 저장
        if sender != "시스템":
            self.conversation.append({
                "role": "user" if is_user else "assistant",
                "content": message
            })
        
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
            # CTransformers로 모델 로드
            logger.info(f"모델 로드 시작: {model_file}")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_file,
                model_type='llama',  # 모델 타입 (llama, gpt2, gptj 등)
                context_length=2048,
                max_new_tokens=512,
                threads=4,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.1,
                gpu_layers=0  # GPU 사용 시 레이어 수
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
        self.model_info_label.configure(text=f"모델: {model_name}")
        self.load_button.configure(state="normal")
        self.send_button.configure(state="normal")
        self.add_message("시스템", f"모델 '{model_name}'이 로드되었습니다.", is_user=False)
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
            
            # CTransformers로 응답 생성
            response = ""
            for token in self.model(prompt, stream=True):
                if self.stop_generation:
                    break
                response += token
            
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
        # 최근 대화 컨텍스트 포함 (최근 6개 메시지)
        recent_messages = self.conversation[-6:] if len(self.conversation) > 6 else self.conversation
        
        prompt = ""
        for msg in recent_messages:
            if msg["role"] == "user":
                prompt += f"User: {msg['content']}\n"
            else:
                prompt += f"Assistant: {msg['content']}\n"
        
        prompt += f"User: {user_message}\nAssistant:"
        
        return prompt
    
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
    
    def clear_conversation(self):
        """대화 초기화"""
        if not self.conversation:
            return
        
        result = messagebox.askyesno("확인", "대화 내용을 모두 지우시겠습니까?")
        if not result:
            return
        
        # 대화 초기화
        self.conversation = []
        
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
            "pip install ctransformers"
        )
        root.destroy()
        sys.exit(1)
    
    try:
        app = SimpleConversationApp()
        app.mainloop()
    except Exception as e:
        logger.error(f"애플리케이션 실행 오류: {str(e)}")
        sys.exit(1)