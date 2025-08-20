# ctransformers_gui.py
"""
ctransformers GGUF GUI
20B 모델 지원, 파일 선택 가능
fileno 오류 없음!
"""

import os
import sys
import json
import time
import threading
import psutil
import warnings
import customtkinter as ctk
from tkinter import filedialog, messagebox, scrolledtext
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# 경고 무시
warnings.filterwarnings("ignore")

# 테마 설정
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class GGUFModelManager:
    """GGUF 모델 관리자"""
    
    def __init__(self):
        self.model = None
        self.model_path = ""
        self.model_type = None
        self.is_loaded = False
        
    def load_model(self, model_path: str, model_type: str = "auto", **kwargs):
        """모델 로드"""
        try:
            from ctransformers import AutoModelForCausalLM
            
            # 기존 모델 정리
            if self.model:
                del self.model
                self.model = None
            
            # 모델 타입 시도 순서
            if model_type == "auto":
                # Phi-3 모델인 경우 여러 타입 시도
                if "phi" in os.path.basename(model_path).lower():
                    type_attempts = ["phi3", "phi", "llama", "gpt2"]
                else:
                    type_attempts = ["llama", "gptneox", "gptj", "gpt2", "falcon", "mpt"]
            else:
                type_attempts = [model_type]
            
            last_error = None
            
            # 각 모델 타입 시도
            for attempt_type in type_attempts:
                try:
                    print(f"모델 타입 '{attempt_type}' 시도 중...")
                    
                    # 새 모델 로드
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        model_type=attempt_type,
                        **kwargs
                    )
                    
                    self.model_path = model_path
                    self.model_type = attempt_type
                    self.is_loaded = True
                    print(f"✅ 모델 타입 '{attempt_type}'로 로드 성공!")
                    return True
                    
                except Exception as e:
                    last_error = e
                    print(f"❌ '{attempt_type}' 실패: {str(e)[:100]}")
                    continue
            
            # 모든 시도 실패
            raise Exception(f"모든 모델 타입 시도 실패. 마지막 오류: {last_error}")
            
        except Exception as e:
            raise Exception(f"모델 로드 실패: {str(e)}")
    
    def generate(self, prompt: str, **kwargs):
        """텍스트 생성"""
        if not self.model:
            raise Exception("모델이 로드되지 않았습니다")
        
        return self.model(prompt, **kwargs)

class CTTransformersGUI(ctk.CTk):
    """ctransformers GUI 애플리케이션"""
    
    def __init__(self):
        super().__init__()
        
        # 윈도우 설정
        self.title("GGUF Model GUI - ctransformers")
        self.geometry("1200x800")
        
        # 모델 매니저
        self.model_manager = GGUFModelManager()
        
        # 대화 기록
        self.conversation_history = []
        
        # 설정 저장 파일
        self.config_file = "gui_config.json"
        self.load_config()
        
        # UI 구성
        self.setup_ui()
        
        # 시스템 정보 표시
        self.update_system_info()
        
    def load_config(self):
        """설정 로드"""
        self.config = {
            "model_path": "",
            "model_type": "auto",
            "max_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "threads": psutil.cpu_count(logical=False),
            "gpu_layers": 0,
            "context_length": 2048,
            "batch_size": 8
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    saved_config = json.load(f)
                    self.config.update(saved_config)
            except:
                pass
    
    def save_config(self):
        """설정 저장"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except:
            pass
    
    def setup_ui(self):
        """UI 구성"""
        # 메인 레이아웃
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # 좌측 패널 (설정)
        self.setup_left_panel()
        
        # 우측 패널 (채팅)
        self.setup_right_panel()
    
    def setup_left_panel(self):
        """좌측 설정 패널"""
        left_frame = ctk.CTkFrame(self, width=350)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        left_frame.grid_propagate(False)
        
        # 스크롤 가능한 프레임
        scroll_frame = ctk.CTkScrollableFrame(left_frame)
        scroll_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 제목
        title = ctk.CTkLabel(
            scroll_frame,
            text="⚙️ GGUF Model Settings",
            font=("Arial", 20, "bold")
        )
        title.pack(pady=(0, 20))
        
        # 시스템 정보
        self.system_info_frame = ctk.CTkFrame(scroll_frame)
        self.system_info_frame.pack(fill="x", pady=(0, 15))
        
        self.ram_label = ctk.CTkLabel(self.system_info_frame, text="RAM: ")
        self.ram_label.pack(anchor="w", padx=10, pady=2)
        
        self.cpu_label = ctk.CTkLabel(self.system_info_frame, text="CPU: ")
        self.cpu_label.pack(anchor="w", padx=10, pady=2)
        
        # 모델 선택
        model_frame = ctk.CTkFrame(scroll_frame)
        model_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(
            model_frame,
            text="📁 모델 파일",
            font=("Arial", 14, "bold")
        ).pack(anchor="w", padx=10, pady=5)
        
        self.model_path_label = ctk.CTkLabel(
            model_frame,
            text="선택된 모델 없음",
            wraplength=300
        )
        self.model_path_label.pack(anchor="w", padx=10, pady=5)
        
        btn_frame = ctk.CTkFrame(model_frame, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkButton(
            btn_frame,
            text="📂 GGUF 파일 선택",
            command=self.select_model,
            width=140
        ).pack(side="left", padx=(0, 5))
        
        ctk.CTkButton(
            btn_frame,
            text="🚀 모델 로드",
            command=self.load_model,
            width=140,
            fg_color="green"
        ).pack(side="left")
        
        # 모델 타입
        ctk.CTkLabel(
            scroll_frame,
            text="모델 타입",
            font=("Arial", 12)
        ).pack(anchor="w", padx=10, pady=(10, 0))
        
        self.model_type_var = ctk.StringVar(value=self.config["model_type"])
        self.model_type_menu = ctk.CTkOptionMenu(
            scroll_frame,
            values=["auto", "llama", "gptneox", "gptj", "gpt2", "falcon", "mpt", "starcoder", "dolly-v2", "replit"],
            variable=self.model_type_var,
            command=self.on_model_type_change
        )
        self.model_type_menu.pack(fill="x", padx=10, pady=5)
        
        # 생성 설정
        gen_frame = ctk.CTkFrame(scroll_frame)
        gen_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(
            gen_frame,
            text="⚡ 생성 설정",
            font=("Arial", 14, "bold")
        ).pack(anchor="w", padx=10, pady=10)
        
        # 설정 항목들
        settings = [
            ("Max Tokens", "max_tokens", 1, 2048, self.config["max_tokens"]),
            ("Temperature", "temperature", 0.1, 2.0, self.config["temperature"]),
            ("Top P", "top_p", 0.1, 1.0, self.config["top_p"]),
            ("Top K", "top_k", 1, 100, self.config["top_k"]),
            ("Threads", "threads", 1, psutil.cpu_count(), self.config["threads"]),
            ("Context Length", "context_length", 512, 8192, self.config["context_length"]),
            ("Batch Size", "batch_size", 1, 512, self.config["batch_size"]),
            ("GPU Layers", "gpu_layers", 0, 100, self.config["gpu_layers"])
        ]
        
        self.setting_vars = {}
        
        for label, key, min_val, max_val, default in settings:
            frame = ctk.CTkFrame(gen_frame, fg_color="transparent")
            frame.pack(fill="x", padx=10, pady=5)
            
            ctk.CTkLabel(frame, text=f"{label}:", width=100).pack(side="left")
            
            if key in ["temperature", "top_p"]:
                # 슬라이더 (소수점)
                var = ctk.DoubleVar(value=default)
                slider = ctk.CTkSlider(
                    frame,
                    from_=min_val,
                    to=max_val,
                    variable=var,
                    width=150
                )
                slider.pack(side="left", padx=10)
                
                value_label = ctk.CTkLabel(frame, text=f"{default:.2f}", width=50)
                value_label.pack(side="left")
                
                # 슬라이더 값 변경 시 라벨 업데이트
                def update_label(val, lbl=value_label, k=key):
                    lbl.configure(text=f"{val:.2f}")
                    self.config[k] = val
                
                var.trace("w", lambda *args, v=var, fn=update_label: fn(v.get()))
                
            else:
                # 슬라이더 (정수)
                var = ctk.IntVar(value=default)
                slider = ctk.CTkSlider(
                    frame,
                    from_=min_val,
                    to=max_val,
                    variable=var,
                    width=150,
                    number_of_steps=max_val-min_val
                )
                slider.pack(side="left", padx=10)
                
                value_label = ctk.CTkLabel(frame, text=str(default), width=50)
                value_label.pack(side="left")
                
                # 슬라이더 값 변경 시 라벨 업데이트
                def update_label_int(val, lbl=value_label, k=key):
                    lbl.configure(text=str(int(val)))
                    self.config[k] = int(val)
                
                var.trace("w", lambda *args, v=var, fn=update_label_int: fn(v.get()))
            
            self.setting_vars[key] = var
        
        # 모델 상태
        self.status_frame = ctk.CTkFrame(scroll_frame)
        self.status_frame.pack(fill="x", pady=10)
        
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="⭕ 모델 미로드",
            font=("Arial", 14, "bold"),
            text_color="orange"
        )
        self.status_label.pack(pady=10)
    
    def setup_right_panel(self):
        """우측 채팅 패널"""
        right_frame = ctk.CTkFrame(self)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
        right_frame.grid_columnconfigure(0, weight=1)
        right_frame.grid_rowconfigure(0, weight=1)
        
        # 제목
        title_frame = ctk.CTkFrame(right_frame, height=50)
        title_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        
        ctk.CTkLabel(
            title_frame,
            text="💬 Chat Interface",
            font=("Arial", 20, "bold")
        ).pack(side="left", padx=10, pady=10)
        
        ctk.CTkButton(
            title_frame,
            text="🗑️ Clear",
            command=self.clear_chat,
            width=80
        ).pack(side="right", padx=10, pady=10)
        
        # 채팅 영역
        self.chat_frame = ctk.CTkScrollableFrame(right_frame)
        self.chat_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        
        # 입력 영역
        input_frame = ctk.CTkFrame(right_frame, height=100)
        input_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(5, 10))
        input_frame.grid_columnconfigure(0, weight=1)
        
        self.input_text = ctk.CTkTextbox(
            input_frame,
            height=70,
            wrap="word"
        )
        self.input_text.grid(row=0, column=0, sticky="ew", padx=(10, 5), pady=10)
        self.input_text.bind("<Return>", self.send_message_event)
        self.input_text.bind("<Shift-Return>", lambda e: None)
        
        self.send_button = ctk.CTkButton(
            input_frame,
            text="Send",
            command=self.send_message,
            width=100,
            height=70,
            font=("Arial", 14, "bold")
        )
        self.send_button.grid(row=0, column=1, padx=(5, 10), pady=10)
        
        # 초기 메시지
        self.add_message("assistant", "안녕하세요! GGUF 모델을 로드하고 대화를 시작해보세요. 👋")
    
    def select_model(self):
        """모델 파일 선택"""
        file_path = filedialog.askopenfilename(
            title="GGUF 모델 선택",
            filetypes=[
                ("GGUF Files", "*.gguf"),
                ("All Files", "*.*")
            ],
            initialdir=self.config.get("last_dir", ".")
        )
        
        if file_path:
            self.config["model_path"] = file_path
            self.config["last_dir"] = os.path.dirname(file_path)
            self.save_config()
            
            # 파일 정보 표시
            file_size = os.path.getsize(file_path) / (1024**3)
            file_name = os.path.basename(file_path)
            
            self.model_path_label.configure(
                text=f"📄 {file_name}\n📊 크기: {file_size:.2f} GB"
            )
            
            # 모델 타입 추측
            file_lower = file_name.lower()
            if "phi" in file_lower:
                # Phi 모델은 보통 llama 타입으로 작동
                self.model_type_var.set("llama")
            elif "llama" in file_lower:
                self.model_type_var.set("llama")
            elif "gpt" in file_lower and "neox" in file_lower:
                self.model_type_var.set("gptneox")
            elif "falcon" in file_lower:
                self.model_type_var.set("falcon")
            elif "mpt" in file_lower:
                self.model_type_var.set("mpt")
            elif "mistral" in file_lower or "mixtral" in file_lower:
                self.model_type_var.set("llama")  # Mistral도 llama 타입
            else:
                self.model_type_var.set("llama")  # 기본값을 llama로
    
    def load_model(self):
        """모델 로드"""
        model_path = self.config["model_path"]
        
        if not model_path or not os.path.exists(model_path):
            messagebox.showerror("오류", "유효한 모델 파일을 선택하세요.")
            return
        
        # 로딩 대화상자
        loading_window = ctk.CTkToplevel(self)
        loading_window.title("Loading...")
        loading_window.geometry("400x150")
        loading_window.transient(self)
        loading_window.grab_set()
        
        # 중앙 배치
        loading_window.update_idletasks()
        x = (loading_window.winfo_screenwidth() // 2) - 200
        y = (loading_window.winfo_screenheight() // 2) - 75
        loading_window.geometry(f"+{x}+{y}")
        
        ctk.CTkLabel(
            loading_window,
            text="⏳ 모델 로드 중...",
            font=("Arial", 16)
        ).pack(pady=20)
        
        progress = ctk.CTkProgressBar(loading_window, width=350)
        progress.pack(pady=10)
        progress.set(0)
        progress.start()
        
        status_label = ctk.CTkLabel(loading_window, text="초기화 중...")
        status_label.pack(pady=5)
        
        # 별도 스레드에서 로드
        def load_thread():
            error_msg = None
            file_name = None
            
            try:
                # ctransformers 확인
                status_label.configure(text="ctransformers 확인 중...")
                
                try:
                    import ctransformers
                    try:
                        version = getattr(ctransformers, '__version__', 'unknown')
                        status_label.configure(text=f"ctransformers {version} 확인")
                    except:
                        status_label.configure(text="ctransformers 확인")
                except ImportError as import_err:
                    raise Exception("ctransformers가 설치되지 않았습니다.\npip install ctransformers")
                
                # 설정 준비
                status_label.configure(text="설정 준비 중...")
                model_config = {
                    "gpu_layers": self.config["gpu_layers"],
                    "threads": self.config["threads"],
                    "context_length": self.config["context_length"],
                    "batch_size": self.config["batch_size"]
                }
                
                # 모델 로드
                file_name = os.path.basename(model_path)
                status_label.configure(text=f"{file_name} 로드 중...")
                
                self.model_manager.load_model(
                    model_path,
                    model_type=self.model_type_var.get(),
                    **model_config
                )
                
                # 성공
                self.after(100, lambda fn=file_name: self.on_model_loaded(loading_window, fn))
                
            except Exception as exc:
                # 실패 - 에러 메시지를 변수에 저장
                error_msg = str(exc)
                self.after(100, lambda err=error_msg: self.on_model_failed(loading_window, err))
        
        threading.Thread(target=load_thread, daemon=True).start()
    
    def on_model_loaded(self, loading_window, model_name):
        """모델 로드 성공"""
        loading_window.destroy()
        
        self.status_label.configure(
            text=f"✅ {model_name} 로드됨",
            text_color="green"
        )
        
        self.add_message("system", f"✅ 모델 '{model_name}'이 성공적으로 로드되었습니다!")
        messagebox.showinfo("성공", "모델이 로드되었습니다. 대화를 시작하세요!")
    
    def on_model_failed(self, loading_window, error):
        """모델 로드 실패"""
        loading_window.destroy()
        
        self.status_label.configure(
            text="❌ 로드 실패",
            text_color="red"
        )
        
        messagebox.showerror("모델 로드 실패", f"오류: {error}")
    
    def on_model_type_change(self, value):
        """모델 타입 변경"""
        self.config["model_type"] = value
        self.save_config()
    
    def send_message_event(self, event):
        """Enter 키 이벤트"""
        self.send_message()
        return "break"
    
    def send_message(self):
        """메시지 전송"""
        if not self.model_manager.is_loaded:
            messagebox.showwarning("경고", "먼저 모델을 로드하세요.")
            return
        
        message = self.input_text.get("1.0", "end").strip()
        if not message:
            return
        
        # 입력 초기화
        self.input_text.delete("1.0", "end")
        
        # 사용자 메시지 표시
        self.add_message("user", message)
        
        # 버튼 비활성화
        self.send_button.configure(state="disabled")
        
        # 응답 생성 (별도 스레드)
        def generate_response():
            try:
                # 프롬프트 구성
                prompt = self.build_prompt(message)
                
                # 생성 설정
                gen_config = {
                    "max_new_tokens": self.config["max_tokens"],
                    "temperature": self.config["temperature"],
                    "top_p": self.config["top_p"],
                    "top_k": self.config["top_k"],
                    "repetition_penalty": 1.1,
                    "stop": ["User:", "\n\n\n"]
                }
                
                # 응답 생성
                start_time = time.time()
                response = self.model_manager.generate(prompt, **gen_config)
                elapsed = time.time() - start_time
                
                # 토큰 수 계산
                tokens = len(response.split())
                speed = tokens / elapsed if elapsed > 0 else 0
                
                # 응답 표시
                self.after(100, lambda: self.display_response(response, elapsed, speed))
                
            except Exception as e:
                self.after(100, lambda: self.display_error(str(e)))
        
        threading.Thread(target=generate_response, daemon=True).start()
    
    def build_prompt(self, message):
        """프롬프트 구성"""
        # 최근 대화 포함
        context = ""
        for msg in self.conversation_history[-6:]:  # 최근 6개 메시지
            if msg["role"] == "user":
                context += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                context += f"Assistant: {msg['content']}\n"
        
        context += f"User: {message}\nAssistant:"
        return context
    
    def display_response(self, response, elapsed, speed):
        """응답 표시"""
        self.add_message("assistant", response.strip())
        self.add_message("system", f"⏱️ 생성 시간: {elapsed:.2f}초 | 속도: {speed:.1f} 토큰/초")
        self.send_button.configure(state="normal")
    
    def display_error(self, error):
        """오류 표시"""
        self.add_message("system", f"❌ 오류: {error}")
        self.send_button.configure(state="normal")
    
    def add_message(self, role, content):
        """메시지 추가"""
        # 메시지 프레임
        msg_frame = ctk.CTkFrame(self.chat_frame)
        msg_frame.pack(fill="x", padx=10, pady=5)
        
        # 역할별 스타일
        if role == "user":
            bg_color = ("#1e4d8b", "#132a4d")  # 파란색
            align = "e"
            role_text = "You"
        elif role == "assistant":
            bg_color = ("#2d7a2d", "#1a4d1a")  # 초록색
            align = "w"
            role_text = "AI"
        else:  # system
            bg_color = ("#666666", "#333333")  # 회색
            align = "center"
            role_text = "System"
        
        msg_frame.configure(fg_color=bg_color)
        
        # 역할 라벨
        role_label = ctk.CTkLabel(
            msg_frame,
            text=role_text,
            font=("Arial", 12, "bold")
        )
        role_label.pack(anchor="w", padx=10, pady=(5, 0))
        
        # 내용 라벨
        content_label = ctk.CTkLabel(
            msg_frame,
            text=content,
            font=("Arial", 12),
            wraplength=700,
            justify="left"
        )
        content_label.pack(anchor="w", padx=10, pady=(0, 5))
        
        # 대화 기록 저장
        if role != "system":
            self.conversation_history.append({"role": role, "content": content})
        
        # 스크롤 하단으로
        self.chat_frame._parent_canvas.yview_moveto(1.0)
    
    def clear_chat(self):
        """채팅 초기화"""
        for widget in self.chat_frame.winfo_children():
            widget.destroy()
        
        self.conversation_history = []
        self.add_message("assistant", "대화가 초기화되었습니다. 새로운 대화를 시작하세요!")
    
    def update_system_info(self):
        """시스템 정보 업데이트"""
        mem = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        self.ram_label.configure(
            text=f"RAM: {mem.available/(1024**3):.1f}/{mem.total/(1024**3):.1f} GB ({mem.percent:.1f}%)"
        )
        self.cpu_label.configure(
            text=f"CPU: {cpu_percent:.1f}% | {psutil.cpu_count()} cores"
        )
        
        # 1초마다 업데이트
        self.after(1000, self.update_system_info)

def main():
    """메인 함수"""
    # ctransformers 확인
    try:
        import ctransformers
        # 버전 확인 (있으면 표시, 없으면 패스)
        try:
            version = ctransformers.__version__
            print(f"✅ ctransformers {version} 확인됨")
        except AttributeError:
            print("✅ ctransformers 모듈 확인됨")
    except ImportError:
        print("❌ ctransformers가 설치되지 않았습니다.")
        print("설치: pip install ctransformers")
        
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "설치 필요",
            "ctransformers가 설치되지 않았습니다.\n\n"
            "터미널에서 다음 명령어를 실행하세요:\n"
            "pip install ctransformers customtkinter psutil"
        )
        return
    
    # GUI 실행
    app = CTTransformersGUI()
    app.mainloop()

if __name__ == "__main__":
    main()