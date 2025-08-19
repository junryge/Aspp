# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 10:51:40 2025

@author: 파이썬AI
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import anthropic
import threading
import os
import json
from pathlib import Path
import re

class ModernTheme:
    """모던 테마 설정"""
    def __init__(self, mode="light"):
        # 색상 설정
        if mode == "dark":
            self.bg_color = "#1e1e1e"  # VS Code 다크 테마 배경색
            self.fg_color = "#d4d4d4"  # 텍스트 색상
            self.accent_color = "#0e639c"  # 강조 색상
            self.secondary_color = "#333333"  # 부 배경색
            self.code_bg = "#2d2d2d"  # 코드 배경색
            self.button_bg = "#0e639c"  # 버튼 배경색
            self.button_fg = "#ffffff"  # 버튼 텍스트 색상
            self.error_color = "#f14c4c"  # 오류 색상
        else:  # light
            self.bg_color = "#f8f8f8"  # 밝은 배경색
            self.fg_color = "#333333"  # 텍스트 색상
            self.accent_color = "#007acc"  # VS Code 블루 색상
            self.secondary_color = "#e8e8e8"  # 부 배경색
            self.code_bg = "#ffffff"  # 코드 배경색
            self.button_bg = "#007acc"  # 버튼 배경색
            self.button_fg = "#ffffff"  # 버튼 텍스트 색상
            self.error_color = "#d83b01"  # 오류 색상
        
        # 폰트 설정
        self.default_font = ("Segoe UI", 10)
        self.code_font = ("Consolas", 11)
        self.header_font = ("Segoe UI", 12, "bold")
        self.title_font = ("Segoe UI", 14, "bold")

class CSharpAssistant:
    def __init__(self, root):
        self.root = root
        self.root.title("C# 코딩 어시스턴트")
        self.root.geometry("1200x800")
        
        # 테마 설정
        self.theme = ModernTheme()
        
        # root 설정
        self.root.configure(bg=self.theme.bg_color)
        
        # API 키 설정
        self.api_key = ""
        self.load_api_key()
        
        # 스타일 설정
        self.setup_styles()
        
        # 메뉴바 생성
        self.create_menu()
        
        # UI 구성
        self.create_ui()
        
        # 파일 관련 변수
        self.current_file = None
    
    def setup_styles(self):
        """ttk 스타일 설정"""
        self.style = ttk.Style()
        
        # 기본 스타일
        self.style.configure("TFrame", background=self.theme.bg_color)
        self.style.configure("TLabel", 
                            background=self.theme.bg_color, 
                            foreground=self.theme.fg_color, 
                            font=self.theme.default_font)
        
        # 버튼 스타일
        self.style.configure("TButton", 
                            background=self.theme.button_bg, 
                            foreground=self.theme.button_fg,
                            font=self.theme.default_font)
        self.style.map("TButton", 
                      background=[("active", self.theme.accent_color), 
                                  ("disabled", self.theme.secondary_color)])
        
        # 헤더 라벨 스타일
        self.style.configure("Header.TLabel", 
                            font=self.theme.header_font, 
                            background=self.theme.bg_color, 
                            foreground=self.theme.fg_color)
        
        # 기본 엔트리 스타일
        self.style.configure("TEntry", 
                            fieldbackground=self.theme.code_bg,
                            foreground=self.theme.fg_color)
        
        # 아코디언 프레임 스타일
        self.style.configure("Accord.TFrame", 
                            background=self.theme.secondary_color,
                            relief="raised")
        
        # 상태바 스타일
        self.style.configure("Status.TLabel", 
                            background=self.theme.secondary_color, 
                            foreground=self.theme.fg_color,
                            relief="sunken", 
                            anchor="w", 
                            padding=(5, 2))
        
        # 중요 버튼 스타일
        self.style.configure("Primary.TButton", 
                            background=self.theme.accent_color, 
                            foreground="white",
                            font=(self.theme.default_font[0], self.theme.default_font[1], "bold"))
    
    def create_menu(self):
        """메뉴바 생성"""
        menubar = tk.Menu(self.root)
        
        # 파일 메뉴
        file_menu = tk.Menu(menubar, tearoff=0, bg=self.theme.code_bg, fg=self.theme.fg_color, 
                           activebackground=self.theme.accent_color, activeforeground=self.theme.button_fg)
        file_menu.add_command(label="새 파일 (Ctrl+N)", command=self.new_file, accelerator="Ctrl+N")
        file_menu.add_command(label="열기... (Ctrl+O)", command=self.open_file, accelerator="Ctrl+O")
        file_menu.add_command(label="저장 (Ctrl+S)", command=self.save_file, accelerator="Ctrl+S")
        file_menu.add_command(label="다른 이름으로 저장...", command=self.save_file_as)
        file_menu.add_separator()
        file_menu.add_command(label="종료", command=self.root.quit)
        menubar.add_cascade(label="파일", menu=file_menu)
        
        # 편집 메뉴
        edit_menu = tk.Menu(menubar, tearoff=0, bg=self.theme.code_bg, fg=self.theme.fg_color, 
                           activebackground=self.theme.accent_color, activeforeground=self.theme.button_fg)
        edit_menu.add_command(label="실행 취소 (Ctrl+Z)", 
                             command=lambda: self.code_editor.event_generate("<<Undo>>"), 
                             accelerator="Ctrl+Z")
        edit_menu.add_command(label="다시 실행 (Ctrl+Y)", 
                             command=lambda: self.code_editor.event_generate("<<Redo>>"), 
                             accelerator="Ctrl+Y")
        edit_menu.add_separator()
        edit_menu.add_command(label="잘라내기 (Ctrl+X)", 
                             command=lambda: self.code_editor.event_generate("<<Cut>>"), 
                             accelerator="Ctrl+X")
        edit_menu.add_command(label="복사 (Ctrl+C)", 
                             command=lambda: self.code_editor.event_generate("<<Copy>>"), 
                             accelerator="Ctrl+C")
        edit_menu.add_command(label="붙여넣기 (Ctrl+V)", 
                             command=lambda: self.code_editor.event_generate("<<Paste>>"), 
                             accelerator="Ctrl+V")
        edit_menu.add_command(label="모두 선택 (Ctrl+A)", 
                             command=lambda: self.code_editor.tag_add("sel", "1.0", "end"), 
                             accelerator="Ctrl+A")
        menubar.add_cascade(label="편집", menu=edit_menu)
        
        # AI 메뉴
        ai_menu = tk.Menu(menubar, tearoff=0, bg=self.theme.code_bg, fg=self.theme.fg_color, 
                         activebackground=self.theme.accent_color, activeforeground=self.theme.button_fg)
        ai_menu.add_command(label="코드 제안 받기 (F5)", 
                           command=self.get_suggestion, 
                           accelerator="F5")
        ai_menu.add_separator()
        ai_menu.add_command(label="코드 최적화", 
                           command=lambda: self.get_suggestion_with_preset("이 코드를 최적화해주세요."))
        ai_menu.add_command(label="버그 찾기", 
                           command=lambda: self.get_suggestion_with_preset("이 코드에서 버그나 오류를 찾아주세요."))
        ai_menu.add_command(label="주석 추가", 
                           command=lambda: self.get_suggestion_with_preset("이 코드에 상세한 주석을 추가해주세요."))
        ai_menu.add_command(label="코드 리팩토링", 
                           command=lambda: self.get_suggestion_with_preset("이 코드를 더 깔끔하게 리팩토링해주세요."))
        menubar.add_cascade(label="AI 도우미", menu=ai_menu)
        
        # 설정 메뉴
        settings_menu = tk.Menu(menubar, tearoff=0, bg=self.theme.code_bg, fg=self.theme.fg_color, 
                               activebackground=self.theme.accent_color, activeforeground=self.theme.button_fg)
        #settings_menu.add_command(label="API 키 설정", command=self.show_api_key_dialog)
        
        # 테마 서브메뉴
        theme_menu = tk.Menu(settings_menu, tearoff=0, bg=self.theme.code_bg, fg=self.theme.fg_color, 
                           activebackground=self.theme.accent_color, activeforeground=self.theme.button_fg)
        theme_menu.add_command(label="라이트 테마", command=lambda: self.change_theme("light"))
        theme_menu.add_command(label="다크 테마", command=lambda: self.change_theme("dark"))
        settings_menu.add_cascade(label="테마 설정", menu=theme_menu)
        
        menubar.add_cascade(label="설정", menu=settings_menu)
        
        # 도움말 메뉴
        help_menu = tk.Menu(menubar, tearoff=0, bg=self.theme.code_bg, fg=self.theme.fg_color, 
                           activebackground=self.theme.accent_color, activeforeground=self.theme.button_fg)
        help_menu.add_command(label="사용법", command=self.show_help)
        help_menu.add_command(label="정보", command=self.show_about)
        menubar.add_cascade(label="도움말", menu=help_menu)
        
        self.root.config(menu=menubar)
        
        # 단축키 설정
        self.root.bind('<Control-n>', lambda event: self.new_file())
        self.root.bind('<Control-o>', lambda event: self.open_file())
        self.root.bind('<Control-s>', lambda event: self.save_file())
        self.root.bind('<F5>', lambda event: self.get_suggestion())
    
    def create_ui(self):
        """UI 구성 요소 생성"""
        # 메인 컨테이너 (PanedWindow)
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 왼쪽 패널 (코드 에디터)
        left_panel = ttk.Frame(main_paned, style="TFrame")
        
        # 코드 에디터 헤더
        editor_header = ttk.Frame(left_panel, style="TFrame")
        editor_header.pack(fill=tk.X, padx=5, pady=5)
        
        editor_label = ttk.Label(editor_header, text="C# 코드 에디터", style="Header.TLabel")
        editor_label.pack(side=tk.LEFT)
        
        # 파일 관련 버튼
        file_buttons = ttk.Frame(editor_header, style="TFrame")
        file_buttons.pack(side=tk.RIGHT)
        
        new_btn = ttk.Button(file_buttons, text="새로 만들기", command=self.new_file, width=12)
        new_btn.pack(side=tk.LEFT, padx=2)
        
        open_btn = ttk.Button(file_buttons, text="열기", command=self.open_file, width=8)
        open_btn.pack(side=tk.LEFT, padx=2)
        
        save_btn = ttk.Button(file_buttons, text="저장", command=self.save_file, width=8)
        save_btn.pack(side=tk.LEFT, padx=2)
        
        # 코드 에디터 생성
        editor_frame = ttk.Frame(left_panel, style="TFrame")
        editor_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.code_editor = scrolledtext.ScrolledText(
            editor_frame,
            wrap=tk.NONE,  # 줄 바꿈 없음
            font=self.theme.code_font,
            bg=self.theme.code_bg,
            fg=self.theme.fg_color,
            insertbackground=self.theme.fg_color,
            selectbackground=self.theme.accent_color,
            selectforeground="white",
            bd=0,
            padx=10,
            pady=10,
            undo=True  # 실행 취소 기능 활성화
        )
        self.code_editor.pack(fill=tk.BOTH, expand=True)
    
        # 바로 여기에 이벤트 바인딩 코드를 추가
        self.code_editor.bind("<KeyRelease>", lambda e: self.highlight_csharp_code())
        
        # 초기 코드에 대해서도 구문 강조 적용
        self.highlight_csharp_code()
        
        # 오른쪽 패널 (결과)
        right_panel = ttk.Frame(main_paned, style="TFrame")
        
        # 결과 헤더
        result_header = ttk.Frame(right_panel, style="TFrame")
        result_header.pack(fill=tk.X, padx=5, pady=5)
        
        result_label = ttk.Label(result_header, text="코드 제안 결과", style="Header.TLabel")
        result_label.pack(side=tk.LEFT)
        
        # 결과 제어 버튼
        result_controls = ttk.Frame(result_header, style="TFrame")
        result_controls.pack(side=tk.RIGHT)
        
        self.apply_btn = ttk.Button(result_controls, text="제안 적용", command=self.apply_suggestion, width=10)
        self.apply_btn.pack(side=tk.LEFT, padx=2)
        
        # 결과 텍스트 영역
        result_frame = ttk.Frame(right_panel, style="TFrame")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.result_text = scrolledtext.ScrolledText(
            result_frame,
            wrap=tk.WORD,
            font=self.theme.code_font,
            bg=self.theme.code_bg,
            fg=self.theme.fg_color,
            bd=0,
            padx=10,
            pady=10
        )
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
        # 코드 블록을 위한 태그 구성
        self.result_text.tag_configure("code_tag", foreground="#777777", font=self.theme.code_font)
        self.result_text.tag_configure("code", background="#f0f0f0", borderwidth=1, relief="solid", 
                                      font=self.theme.code_font, spacing1=10, spacing3=10)
        
        # PanedWindow에 두 패널 추가
        main_paned.add(left_panel, weight=1)
        main_paned.add(right_panel, weight=1)
        
        # 하단 프레임 (프롬프트 + 버튼)
        bottom_frame = ttk.Frame(self.root, style="TFrame")
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        # 프롬프트 라벨
        prompt_label = ttk.Label(bottom_frame, text="요청 내용:", style="TLabel")
        prompt_label.pack(side=tk.LEFT, padx=5, pady=5)
        
        # 프롬프트 입력 필드
        # 프롬프트 입력 필드
        # 먼저 status_var 초기화 (이전에 누락된 부분)
        self.status_var = tk.StringVar(value="준비")
        
        # 프롬프트 프레임 추가 (동적 크기 조정을 위해)
        prompt_frame = ttk.Frame(bottom_frame)
        prompt_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        
        self.prompt_entry = scrolledtext.ScrolledText(
            prompt_frame, 
            wrap=tk.WORD,
            height=3,  # 초기 높이 3줄
            font=self.theme.default_font,
            bg=self.theme.code_bg,
            fg=self.theme.fg_color,
            insertbackground=self.theme.fg_color,
            bd=1,
            padx=5,
            pady=5,
            undo=True  # 실행 취소 기능 활성화
        )
        self.prompt_entry.pack(fill=tk.X, expand=True)
        
        # 동적 높이 조정 메서드 추가
        def adjust_height(event=None):
            # 현재 라인 수 계산
            lines = self.prompt_entry.get("1.0", tk.END).count('\n')
            
            # 최소 3줄, 최대 10줄로 제한
            height = max(3, min(lines + 1, 10))
            self.prompt_entry.configure(height=height)
        
        # 키 입력 시 높이 조정
        self.prompt_entry.bind('<KeyRelease>', adjust_height)
        
        # Enter 키로 다음 위젯으로 이동, Ctrl+Enter로 코드 제안
        def on_enter(event):
            # Ctrl 키 확인을 위해 수정
            if event.state & 0x4 or (event.keysym == 'Return' and event.state & 0x1):  # Ctrl 키 또는 Ctrl+Enter
                self.get_suggestion()
                return 'break'
            
            # 현재 커서 위치의 라인 번호
            current_line = int(self.prompt_entry.index(tk.INSERT).split('.')[0])
            total_lines = int(self.prompt_entry.index(tk.END).split('.')[0])
            
            # 마지막 라인이면 다음 위젯으로 이동
            if current_line == total_lines:
                self.suggest_button.focus_set()  # 다음 위젯(제안 버튼)으로 포커스 이동
                return 'break'
            
            # 아니면 다음 라인으로 이동
            return None
        
        self.prompt_entry.bind("<Return>", on_enter)
        # Ctrl+Enter 대신 Control-Return 사용
        self.prompt_entry.bind("<Control-Return>", on_enter)
        
        # 프롬프트 내용 가져오는 메서드 추가
        def get_prompt_text():
            return self.prompt_entry.get("1.0", tk.END).strip()
        
        # 제안 버튼
        self.suggest_button = ttk.Button(
            bottom_frame,
            text="코드 제안 받기",
            command=self.get_suggestion,
            style="Primary.TButton",
            width=15
        )
        self.suggest_button.pack(side=tk.RIGHT, padx=5)
        
        # 상태 표시줄
        self.status_var = tk.StringVar()
        self.status_var.set("준비")
        self.status_bar = ttk.Label(
            self.root, 
            textvariable=self.status_var, 
            style="Status.TLabel"
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def load_api_key(self):
        """config.json 파일에서 API 키를 로드"""
        config_path = Path.home() / ".csharp_assistant" / "config.json"
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self.api_key = config.get("api_key", "")
            except Exception as e:
                messagebox.showerror("오류", f"설정 파일 로드 중 오류 발생: {e}")
    
    def save_api_key(self):
        """API 키를 config.json 파일에 저장"""
        config_path = Path.home() / ".csharp_assistant"
        config_path.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_path / "config.json", 'w') as f:
                json.dump({"api_key": self.api_key}, f)
            messagebox.showinfo("성공", "API 키가 저장되었습니다.")
        except Exception as e:
            messagebox.showerror("오류", f"설정 파일 저장 중 오류 발생: {e}")
    
    def show_api_key_dialog(self):
        """API 키 설정 다이얼로그"""
        dialog = tk.Toplevel(self.root)
        dialog.title("API 키 설정")
        dialog.geometry("450x180")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.configure(bg=self.theme.bg_color)
        
        # 다이얼로그 내용 프레임
        frame = ttk.Frame(dialog, style="TFrame", padding=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # 헤더 레이블
        header = ttk.Label(frame, text="JUN/BLACK API 키 설정", 
                          font=self.theme.header_font, style="TLabel")
        header.grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 15))
        
        # 설명 텍스트
        desc = ttk.Label(frame, text="JUN/BLACK API 사용하기 위한 API 키를 입력하세요.", 
                         style="TLabel", wraplength=400)
        desc.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))
        
        # API 키 라벨 및 입력 필드
        key_label = ttk.Label(frame, text="API 키:", style="TLabel")
        key_label.grid(row=2, column=0, sticky=tk.W, pady=10)
        
        api_key_var = tk.StringVar(value=self.api_key)
        api_key_entry = ttk.Entry(frame, textvariable=api_key_var, width=40, show="*")
        api_key_entry.grid(row=2, column=1, sticky=tk.W+tk.E, pady=10, padx=(5, 0))
        
        # 버튼 프레임
        btn_frame = ttk.Frame(frame, style="TFrame")
        btn_frame.grid(row=3, column=0, columnspan=2, pady=(15, 0))
        
        # 취소 버튼
        cancel_btn = ttk.Button(
            btn_frame, 
            text="취소", 
            command=dialog.destroy,
            width=10
        )
        cancel_btn.pack(side=tk.RIGHT, padx=5)
        
        # 저장 버튼
        save_btn = ttk.Button(
            btn_frame, 
            text="저장", 
            command=lambda: self.save_api_key_from_dialog(api_key_var.get(), dialog),
            style="Primary.TButton",
            width=10
        )
        save_btn.pack(side=tk.RIGHT, padx=5)
    
    def save_api_key_from_dialog(self, api_key, dialog):
        """다이얼로그에서 API 키 저장"""
        self.api_key = api_key
        self.save_api_key()
        dialog.destroy()
    
    def new_file(self):
        """새 파일 생성"""
        if self.code_editor.get("1.0", tk.END).strip():
            if messagebox.askyesno("확인", "현재 내용을 저장하지 않고 새 파일을 생성하시겠습니까?"):
                self.code_editor.delete("1.0", tk.END)
                self.result_text.delete("1.0", tk.END)
                self.current_file = None
                self.status_var.set("새 파일")
        else:
            self.code_editor.delete("1.0", tk.END)
            self.result_text.delete("1.0", tk.END)
            self.current_file = None
            self.status_var.set("새 파일")
    
    def open_file(self):
        """파일 열기 다이얼로그"""
        file_path = filedialog.askopenfilename(
            filetypes=[("C# 파일", "*.cs"), ("모든 파일", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    self.code_editor.delete("1.0", tk.END)
                    self.code_editor.insert(tk.END, content)
                    self.current_file = file_path
                    self.status_var.set(f"파일 열기: {file_path}")
                    # 구문 강조 적용
                    self.highlight_csharp_code()
            except Exception as e:
                messagebox.showerror("오류", f"파일 열기 중 오류 발생: {e}")
    
    def save_file(self):
        """현재 파일 저장"""
        if self.current_file:
            try:
                content = self.code_editor.get("1.0", tk.END)
                with open(self.current_file, 'w', encoding='utf-8') as file:
                    file.write(content)
                self.status_var.set(f"파일 저장됨: {self.current_file}")
            except Exception as e:
                messagebox.showerror("오류", f"파일 저장 중 오류 발생: {e}")
        else:
            self.save_file_as()

    def highlight_comments_in_code(self):
        """코드 내의 주석을 강조 표시"""
        content = self.result_text.get("1.0", tk.END)
        
        # // 스타일 한 줄 주석 찾기
        for match in re.finditer(r'(//.*?)(\n|$)', content):
            start, end = match.span(1)
            start_index = f"1.0 + {start} chars"
            end_index = f"1.0 + {end} chars"
            self.result_text.tag_add("comment", start_index, end_index)
        
        # /* */ 스타일 블록 주석 찾기
        for match in re.finditer(r'/\*([\s\S]*?)\*/', content):
            start, end = match.span()
            start_index = f"1.0 + {start} chars"
            end_index = f"1.0 + {end} chars"
            self.result_text.tag_add("block_comment", start_index, end_index)
        
        # C# 키워드 강조
        keywords = [
            r'\busing\b', r'\bnamespace\b', r'\bclass\b', r'\bpublic\b', r'\bprivate\b', 
            r'\bprotected\b', r'\binternal\b', r'\bstatic\b', r'\bvoid\b', r'\breturn\b',
            r'\bint\b', r'\bstring\b', r'\bbool\b', r'\bfloat\b', r'\bdouble\b', r'\bchar\b',
            r'\bif\b', r'\belse\b', r'\bswitch\b', r'\bcase\b', r'\bfor\b', r'\bwhile\b', 
            r'\bdo\b', r'\bforeach\b', r'\btry\b', r'\bcatch\b', r'\bfinally\b', r'\bthrow\b',
            r'\bnew\b', r'\btrue\b', r'\bfalse\b', r'\bnull\b', r'\bthis\b', r'\bbase\b'
        ]
        
        for keyword in keywords:
            keyword_pattern = re.compile(keyword)
            for match in keyword_pattern.finditer(content):
                start, end = match.span()
                start_index = f"1.0 + {start} chars"
                end_index = f"1.0 + {end} chars"
                self.result_text.tag_add("keyword", start_index, end_index)
        
        # 문자열 강조 (큰따옴표)
        for match in re.finditer(r'"[^"\\]*(\\.[^"\\]*)*"', content):
            start, end = match.span()
            start_index = f"1.0 + {start} chars"
            end_index = f"1.0 + {end} chars"
            self.result_text.tag_add("string", start_index, end_index)
        
        # 태그 색상 설정
        self.result_text.tag_configure("comment", foreground="#008000")  # 초록색 주석
        self.result_text.tag_configure("block_comment", foreground="#008000")  # 초록색 블록 주석
        self.result_text.tag_configure("keyword", foreground="#0000FF")  # 파란색 키워드
        self.result_text.tag_configure("string", foreground="#A31515")  # 적갈색 문자열

    # C# 코드 하이라이팅 함수 개선
    def highlight_csharp_code(self):
        """코드 에디터의 C# 코드 구문 강조 표시"""
        try:
            self.apply_csharp_highlighting(self.code_editor)
            self.status_var.set("구문 강조 적용됨")
        except Exception as e:
            print(f"코드 에디터 하이라이팅 오류: {e}")
    
    def save_file_as(self):
        """다른 이름으로 저장"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".cs",
            filetypes=[("C# 파일", "*.cs"), ("모든 파일", "*.*")]
        )
        if file_path:
            try:
                content = self.code_editor.get("1.0", tk.END)
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(content)
                self.current_file = file_path
                self.status_var.set(f"파일 저장됨: {file_path}")
            except Exception as e:
                messagebox.showerror("오류", f"파일 저장 중 오류 발생: {e}")
    
    def get_suggestion_with_preset(self, preset_prompt):
        """미리 정의된 프롬프트로 코드 제안 받기"""
        self.prompt_var.set(preset_prompt)
        self.get_suggestion()
    
    def get_suggestion(self):
        """JUN/BLACK J API를 사용하여 코드 제안 받기"""
        if not self.api_key:
            messagebox.showwarning("경고", "API 키가 설정되지 않았습니다. 설정 메뉴에서 API 키를 설정하세요.")
            return
        
        code = self.code_editor.get("1.0", tk.END).strip()
        prompt = self.prompt_entry.get("1.0", tk.END).strip()

        
        if not code:
            messagebox.showwarning("경고", "코드가 입력되지 않았습니다.")
            return
        
        if not prompt:
            prompt = "이 코드를 분석하고 개선점이나 최적화 방안을 제안해주세요."
        
        # 비동기적으로 API 호출
        self.status_var.set("JUN/BLACK J API 호출 중...")
        self.suggest_button.config(state=tk.DISABLED)
        
        # 프로그레스 창 표시
        self.show_progress_dialog("요청 처리 중", "JUN/BLACK J API에 요청을 처리 중입니다...")
        
        threading.Thread(target=self.call_claude_api, args=(code, prompt)).start()
    
    def show_progress_dialog(self, title, message):
        """진행 상태 창 표시"""
        self.progress_dialog = tk.Toplevel(self.root)
        self.progress_dialog.title(title)
        self.progress_dialog.geometry("350x120")
        self.progress_dialog.resizable(False, False)
        self.progress_dialog.transient(self.root)
        self.progress_dialog.grab_set()
        self.progress_dialog.configure(bg=self.theme.bg_color)
        
        # 다이얼로그 내용
        frame = ttk.Frame(self.progress_dialog, style="TFrame", padding=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # 메시지 레이블
        ttk.Label(frame, text=message, style="TLabel").pack(pady=(0, 15))
        
        # 진행 상태 바
        progress = ttk.Progressbar(frame, mode="indeterminate", length=300)
        progress.pack(pady=5)
        progress.start(10)


    def apply_csharp_highlighting(self, text_widget):
        """텍스트 위젯에 C# 구문 강조 적용"""
        # 기존 태그 제거
        for tag in ["keyword", "type", "string", "comment", "korean_comment"]:
            text_widget.tag_remove(tag, "1.0", tk.END)
        
        # C# 키워드 찾기
        keywords = ["using", "namespace", "class", "public", "private", "protected", 
                   "internal", "static", "void", "override", "partial", "this", "base"]
        
        for keyword in keywords:
            pos = "1.0"
            while True:
                pos = text_widget.search(r'\y' + keyword + r'\y', pos, tk.END, regexp=True)
                if not pos:
                    break
                end_pos = f"{pos}+{len(keyword)}c"
                text_widget.tag_add("keyword", pos, end_pos)
                pos = end_pos
        
        # 타입 찾기
        types = ["object", "string", "int", "bool", "float", "double", "var", "EventArgs"]
        for typ in types:
            pos = "1.0"
            while True:
                pos = text_widget.search(r'\y' + typ + r'\y', pos, tk.END, regexp=True)
                if not pos:
                    break
                end_pos = f"{pos}+{len(typ)}c"
                text_widget.tag_add("type", pos, end_pos)
                pos = end_pos
        
        # 주석 찾기 (// 스타일)
        pos = "1.0"
        while True:
            pos = text_widget.search(r'//.*$', pos, tk.END, regexp=True)
            if not pos:
                break
            line_end = pos.split('.')[0] + '.end'
            text_widget.tag_add("comment", pos, line_end)
            pos = text_widget.index(f"{pos}+1l")
        
        # 설조 찾기
        pos = "1.0"
        while True:
            pos = text_widget.search('설조', pos, tk.END)
            if not pos:
                break
            line_end = pos.split('.')[0] + '.end'
            text_widget.tag_add("korean_comment", pos, line_end)
            pos = text_widget.index(f"{pos}+1l")
        
        # 태그 스타일 설정
        text_widget.tag_configure("keyword", foreground="#0000FF", font=(self.theme.code_font[0], self.theme.code_font[1], "bold"))
        text_widget.tag_configure("type", foreground="#2B91AF")
        text_widget.tag_configure("string", foreground="#A31515")
        text_widget.tag_configure("comment", foreground="#008000", font=(self.theme.code_font[0], self.theme.code_font[1], "italic"))
        text_widget.tag_configure("korean_comment", foreground="#FF6600", background="#FFFFCC", font=(self.theme.code_font[0], self.theme.code_font[1], "bold"))
    
    def apply_csharp_highlighting_in_range(self, text_widget, start, end):
        """특정 범위 내에서 C# 구문 강조 적용"""
        # C# 키워드 찾기
        keywords = ["using", "namespace", "class", "public", "private", "protected", 
                   "internal", "static", "void", "override", "partial", "this", "base"]
        
        for keyword in keywords:
            pos = start
            while True:
                pos = text_widget.search(r'\y' + keyword + r'\y', pos, end, regexp=True)
                if not pos:
                    break
                end_pos = f"{pos}+{len(keyword)}c"
                text_widget.tag_add("keyword", pos, end_pos)
                pos = end_pos
        
        # 타입 찾기
        types = ["object", "string", "int", "bool", "float", "double", "var", "EventArgs"]
        for typ in types:
            pos = start
            while True:
                pos = text_widget.search(r'\y' + typ + r'\y', pos, end, regexp=True)
                if not pos:
                    break
                end_pos = f"{pos}+{len(typ)}c"
                text_widget.tag_add("type", pos, end_pos)
                pos = end_pos
        
        # 주석 찾기 (// 스타일)
        pos = start
        while True:
            pos = text_widget.search(r'//.*$', pos, end, regexp=True)
            if not pos:
                break
            line_end = pos.split('.')[0] + '.end'
            if text_widget.compare(line_end, ">", end):
                line_end = end
            text_widget.tag_add("comment", pos, line_end)
            next_line = text_widget.index(f"{pos}+1l")
            if text_widget.compare(next_line, ">=", end):
                break
            pos = next_line
        
        # 설조 찾기
        pos = start
        while True:
            pos = text_widget.search('설조', pos, end)
            if not pos:
                break
            line_end = pos.split('.')[0] + '.end'
            if text_widget.compare(line_end, ">", end):
                line_end = end
            text_widget.tag_add("korean_comment", pos, line_end)
            pos = text_widget.index(f"{pos}+1c")
            if text_widget.compare(pos, ">=", end):
                break
    
    def call_claude_api(self, code, prompt):
        """JUN/BLACK J API 호출 (스레드에서 실행)"""
        try:
            client = anthropic.Anthropic(api_key=self.api_key)
            response = client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=4000,#4000
                system="당신은 C# 전문가로서 코드 분석, 개선, 버그 수정, 최적화 등을 제안하는 어시스턴트입니다. 제안 코드는 항상 실행 가능하고 정확해야 합니다.",
                messages=[
                    {"role": "user", "content": f"다음 C# 코드를 분석해주세요:\n\n```csharp\n{code}\n```\n\n요청사항: {prompt}"}
                ]
            )
            
            # UI 스레드에서 결과 업데이트
            self.root.after(0, self.update_result, response.content[0].text)
        except Exception as e:
            self.root.after(0, self.show_error, str(e))
    
    def update_result(self, result):
        """결과 업데이트 (UI 스레드에서 호출)"""
        # 프로그레스 창 닫기
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.destroy()
    
        # 응답에 백틱(마크다운 코드 블록)이 없다면 자동으로 코드 블록 처리
        if "```" not in result:
            # 만약 "개선된 코드"라는 헤더가 있다면 그 이후부터 코드로 간주하고 코드 블록으로 감싸기
            pattern = re.compile(r"(#\s*개선된\s*코드)([\s\S]+)", re.IGNORECASE)
            match = pattern.search(result)
            if match:
                heading = match.group(1)
                code_text = match.group(2).strip()
                # 기존 응답의 해당 부분을 백틱으로 감싸도록 변환
                result = result.replace(match.group(0), f"{heading}\n```csharp\n{code_text}\n```")
            else:
                # 헤더가 없다면 응답 전체를 코드 블록으로 감싸기
                result = f"```csharp\n{result.strip()}\n```"
    
        # 결과 텍스트 위젯 업데이트
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, result)
    
        # 마크다운 코드 블록 강조 적용
        self.highlight_code_blocks()
    
        self.status_var.set("코드 제안 완료")
        self.suggest_button.config(state=tk.NORMAL)

    
    def highlight_code_blocks(self):
        """결과 텍스트에서 마크다운 코드 블록 강조 표시 (백틱, 언어 지정 포함)"""
        content = self.result_text.get("1.0", tk.END)
        self.result_text.delete("1.0", tk.END)
        
        # 수정된 정규식 패턴: 시작 백틱 세 개 뒤에 선택적으로 'csharp'와 공백 및 줄바꿈이 올 수 있음
        pattern = re.compile(r'```(?:csharp)?\s*\n([\s\S]*?)\n```')
        last_end = 0
        for match in pattern.finditer(content):
            start, end = match.span()
            
            # 코드 블록 이전의 텍스트를 그대로 추가
            self.result_text.insert(tk.END, content[last_end:start])
            
            # 매치된 코드 블록 내용만 가져옴 (마크업은 제거)
            code_content = match.group(1).rstrip()
            code_start = self.result_text.index(tk.END)
            self.result_text.insert(tk.END, code_content, "code")
            code_end = self.result_text.index(tk.END)
            
            # 코드 내부 하이라이팅 적용
            if code_content:
                try:
                    self.apply_csharp_highlighting_in_range(self.result_text, code_start, code_end)
                except Exception as e:
                    print(f"코드 블록 내부 하이라이팅 오류: {e}")
            
            last_end = end
        
        # 마지막 코드 블록 이후의 텍스트 추가
        self.result_text.insert(tk.END, content[last_end:])

    
    def show_error(self, error_message):
        """오류 표시 (UI 스레드에서 호출)"""
        # 프로그레스 창 닫기
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.destroy()
            
        messagebox.showerror("API 오류", f"JUN/BLACK J API 호출 중 오류 발생: {error_message}")
        self.status_var.set("오류 발생")
        self.suggest_button.config(state=tk.NORMAL)
    
    def apply_suggestion(self):
        """제안 결과를 에디터에 적용"""
        result = self.result_text.get("1.0", tk.END)
        
        # 코드 블록 찾기
        code_blocks = re.findall(r'```(?:csharp)?\s*([\s\S]*?)```', result)
        
        if code_blocks:
            # 첫 번째 코드 블록 사용
            code = code_blocks[0].strip()
            
            # 사용자 확인
            if messagebox.askyesno("확인", "제안된 코드를 에디터에 적용하시겠습니까?"):
                self.code_editor.delete("1.0", tk.END)
                self.code_editor.insert(tk.END, code)
                self.status_var.set("제안 코드가 적용되었습니다")
        else:
            messagebox.showinfo("정보", "제안 결과에서 코드 블록을 찾을 수 없습니다.")

    def show_search_dialog(self):
        """검색 및 바꾸기 다이얼로그 표시"""
        search_dialog = tk.Toplevel(self.root)
        search_dialog.title("검색 및 바꾸기")
        search_dialog.geometry("400x150")
        search_dialog.transient(self.root)
        search_dialog.configure(bg=self.theme.bg_color)
        
        frame = ttk.Frame(search_dialog, style="TFrame", padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # 검색어 입력
        ttk.Label(frame, text="검색:", style="TLabel").grid(row=0, column=0, sticky=tk.W, pady=5)
        search_var = tk.StringVar()
        search_entry = ttk.Entry(frame, textvariable=search_var, width=30)
        search_entry.grid(row=0, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        search_entry.focus_set()
        
        # 바꿀 텍스트 입력
        ttk.Label(frame, text="바꾸기:", style="TLabel").grid(row=1, column=0, sticky=tk.W, pady=5)
        replace_var = tk.StringVar()
        replace_entry = ttk.Entry(frame, textvariable=replace_var, width=30)
        replace_entry.grid(row=1, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        
        # 버튼 프레임
        btn_frame = ttk.Frame(frame, style="TFrame")
        btn_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        # 검색 버튼
        search_btn = ttk.Button(
            btn_frame, 
            text="검색", 
            command=lambda: self.search_text(search_var.get()),
            width=10
        )
        search_btn.pack(side=tk.LEFT, padx=5)
        
        # 바꾸기 버튼
        replace_btn = ttk.Button(
            btn_frame, 
            text="바꾸기", 
            command=lambda: self.replace_text(search_var.get(), replace_var.get()),
            width=10
        )
        replace_btn.pack(side=tk.LEFT, padx=5)
        
        # 모두 바꾸기 버튼
        replace_all_btn = ttk.Button(
            btn_frame, 
            text="모두 바꾸기", 
            command=lambda: self.replace_all_text(search_var.get(), replace_var.get()),
            width=10
        )
        replace_all_btn.pack(side=tk.LEFT, padx=5)
        
        # 취소 버튼
        cancel_btn = ttk.Button(
            btn_frame, 
            text="취소", 
            command=search_dialog.destroy,
            width=10
        )
        cancel_btn.pack(side=tk.LEFT, padx=5)
    
    def search_text(self, search_str):
        """텍스트 검색"""
        if not search_str:
            return
        
        # 이전 검색 결과 태그 삭제
        self.code_editor.tag_remove("search", "1.0", tk.END)
        
        start_pos = "1.0"
        while True:
            start_pos = self.code_editor.search(search_str, start_pos, stopindex=tk.END)
            if not start_pos:
                break
            
            end_pos = f"{start_pos}+{len(search_str)}c"
            self.code_editor.tag_add("search", start_pos, end_pos)
            start_pos = end_pos
        
        # 검색 결과 하이라이트
        self.code_editor.tag_configure("search", background="yellow", foreground="black")
    
    def change_theme(self, theme_mode):
        """테마 변경"""
        # 새 테마 설정
        self.theme = ModernTheme(theme_mode)
        
        # 기본 스타일 업데이트
        self.root.configure(bg=self.theme.bg_color)
        
        # 스타일 재설정
        self.setup_styles()
        
        # 에디터 색상 업데이트
        self.code_editor.configure(
            bg=self.theme.code_bg,
            fg=self.theme.fg_color,
            insertbackground=self.theme.fg_color,
            selectbackground=self.theme.accent_color
        )
        
        # 결과 텍스트 색상 업데이트
        self.result_text.configure(
            bg=self.theme.code_bg,
            fg=self.theme.fg_color
        )
        
        # 코드 블록 태그 업데이트
        if theme_mode == "dark":
            self.result_text.tag_configure("code", background="#2d2d2d", foreground="#e0e0e0")
            self.result_text.tag_configure("code_tag", foreground="#888888")
        else:
            self.result_text.tag_configure("code", background="#f0f0f0", foreground="#333333")
            self.result_text.tag_configure("code_tag", foreground="#777777")
        
        # 상태 메시지
        self.status_var.set(f"{theme_mode.capitalize()} 테마로 변경되었습니다")
    
    def show_help(self):
        """도움말 표시"""
        help_dialog = tk.Toplevel(self.root)
        help_dialog.title("C# 코딩 어시스턴트 사용법")
        help_dialog.geometry("600x400")
        help_dialog.transient(self.root)
        help_dialog.configure(bg=self.theme.bg_color)
        
        # 내용 프레임
        content_frame = ttk.Frame(help_dialog, style="TFrame", padding=20)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # 제목
        title_label = ttk.Label(
            content_frame, 
            text="C# 코딩 어시스턴트 사용법", 
            font=self.theme.title_font,
            style="TLabel"
        )
        title_label.pack(anchor=tk.W, pady=(0, 20))
        
        # 도움말 내용
        help_text = scrolledtext.ScrolledText(
            content_frame,
            wrap=tk.WORD,
            width=70,
            height=15,
            font=self.theme.default_font,
            bg=self.theme.code_bg,
            fg=self.theme.fg_color,
            bd=0,
            padx=10,
            pady=10
        )
        help_text.pack(fill=tk.BOTH, expand=True)
        
        help_content = """
C# 코딩 어시스턴트 사용 방법:

1. 코드 작성 및 분석
   - 왼쪽 에디터에 C# 코드를 입력하거나 파일에서 불러옵니다.
   - 하단의 "요청 내용" 필드에 분석 요청을 입력합니다.
   - "코드 제안 받기" 버튼을 클릭하거나 F5 키를 누릅니다.

2. 파일 관리
   - Ctrl+N: 새 파일 만들기
   - Ctrl+O: 파일 열기
   - Ctrl+S: 파일 저장

3. AI 기능
   - "AI 도우미" 메뉴에서 코드 최적화, 버그 찾기, 주석 추가 등의 기능을 이용할 수 있습니다.
   - 제안 결과에서 "제안 적용" 버튼을 클릭하여 코드를 바로 적용할 수 있습니다.


주의: 제안된 코드는 항상 검토 후 사용하시기 바랍니다.
        """
        
        help_text.insert(tk.END, help_content)
        help_text.configure(state=tk.DISABLED)  # 읽기 전용으로 설정
        
        # 닫기 버튼
        close_btn = ttk.Button(
            content_frame, 
            text="닫기", 
            command=help_dialog.destroy, 
            style="Primary.TButton",
            width=10
        )
        close_btn.pack(side=tk.RIGHT, pady=(10, 0))
    
    def show_about(self):
        """정보 표시"""
        about_dialog = tk.Toplevel(self.root)
        about_dialog.title("C# 코딩 어시스턴트 정보")
        about_dialog.geometry("400x300")
        about_dialog.transient(self.root)
        about_dialog.configure(bg=self.theme.bg_color)
        
        # 내용 프레임
        content_frame = ttk.Frame(about_dialog, style="TFrame", padding=20)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # 앱 이름
        ttk.Label(
            content_frame, 
            text="C# 코딩 어시스턴트", 
            font=self.theme.title_font,
            style="TLabel"
        ).pack(pady=(0, 10))
        
        # 버전
        ttk.Label(
            content_frame, 
            text="버전 1.0.0", 
            style="TLabel"
        ).pack(pady=(0, 20))
        
        # 설명
        desc = ttk.Label(
            content_frame, 
            text="JUN/BLACK J AI를 활용한 C# 코드 분석 및 제안 도구입니다.\n"
                 "코드 최적화, 버그 찾기, 리팩토링 등의 작업에 도움을 줍니다.",
            wraplength=350,
            justify=tk.CENTER,
            style="TLabel"
        )
        desc.pack(pady=(0, 20))
        
        # 크레딧
        ttk.Label(
            content_frame, 
            text="개발: JUN/BLACK J와 함께",
            style="TLabel"
        ).pack()
        
        # 닫기 버튼
        close_btn = ttk.Button(
            content_frame, 
            text="닫기", 
            command=about_dialog.destroy, 
            style="Primary.TButton",
            width=10
        )
        close_btn.pack(side=tk.BOTTOM, pady=(20, 0))



class LineNumberedText(scrolledtext.ScrolledText):
    """라인 번호가 있는 텍스트 위젯"""
    def __init__(self, *args, **kwargs):
        scrolledtext.ScrolledText.__init__(self, *args, **kwargs)
        
        self.line_numbers = tk.Text(self, width=4, bg='#f0f0f0', 
                                    fg='#606060', bd=0, 
                                    font=kwargs.get('font', ('Consolas', 10)),
                                    state='disabled')
        self.line_numbers.pack(side=tk.LEFT, fill=tk.Y)
        
        # 스크롤 연결
        self.vscrollbar = self.vbar
        self.vscrollbar.configure(command=self._on_scroll)
        
        # 수정 이벤트 바인딩
        self.bind('<<Modified>>', self._on_modified)
        self.bind('<Configure>', self._on_resize)
        
        # 초기 라인 번호 설정
        self._update_line_numbers()
    
    def _on_scroll(self, *args):
        """스크롤 시 라인 번호도 함께 스크롤"""
        self.line_numbers.yview(*args)
        scrolledtext.ScrolledText.yview(self, *args)
    
    def _on_modified(self, event=None):
        """텍스트 수정 시 라인 번호 업데이트"""
        self._update_line_numbers()
        self.edit_modified(False)  # 수정 플래그 리셋
    
    def _on_resize(self, event=None):
        """위젯 크기 변경 시 라인 번호 업데이트"""
        self._update_line_numbers()
    
    def _update_line_numbers(self):
        """라인 번호 업데이트"""
        self.line_numbers.configure(state='normal')
        self.line_numbers.delete('1.0', tk.END)
        
        lines = self.get('1.0', tk.END).count('\n')
        for i in range(1, lines + 1):
            self.line_numbers.insert(tk.END, f'{i}\n')
        
        self.line_numbers.configure(state='disabled')
        
if __name__ == "__main__":
    root = tk.Tk()
    app = CSharpAssistant(root)
    root.mainloop()