# -*- coding: utf-8 -*-
"""
Improved C# Coding Assistant
Using customtkinter for better UI and enhanced code highlighting
"""

import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox
import customtkinter as ctk
import anthropic
import threading
import os
import json
from pathlib import Path
import re
import pygments
from pygments.lexers import CSharpLexer
from pygments.formatters import HtmlFormatter
import webbrowser
from ttkwidgets import AutoHideScrollbar

# Set appearance mode and default color theme
ctk.set_appearance_mode("system")  # Options: "Light", "Dark", "System"
ctk.set_default_color_theme("blue")  # Options: "blue", "green", "dark-blue"

class SyntaxHighlightingText(ctk.CTkTextbox):
    """Custom text widget with syntax highlighting capabilities"""
    
    def __init__(self, *args, **kwargs):
        self.language = kwargs.pop('language', 'csharp')
        super().__init__(*args, **kwargs)
        
        # Configure tags for syntax highlighting
        self._configure_tags()
        
        # Bind events for updates
        self._textbox.bind("<KeyRelease>", self._on_key_release)
    
    def _configure_tags(self):
        """Configure highlighting tags"""
        # Access the internal Tkinter Text widget
        text_widget = self._textbox
        
        # Keywords
        text_widget.tag_configure("keyword", foreground="#569CD6")
        
        # Types
        text_widget.tag_configure("type", foreground="#4EC9B0")
        
        # Strings
        text_widget.tag_configure("string", foreground="#CE9178")
        
        # Comments
        text_widget.tag_configure("comment", foreground="#6A9955", font=("Consolas", 11, "italic"))
        
        # Numbers
        text_widget.tag_configure("number", foreground="#B5CEA8")
        
        # Methods
        text_widget.tag_configure("method", foreground="#DCDCAA")
        
        # Class
        text_widget.tag_configure("class", foreground="#4EC9B0")
        
        # Namespaces
        text_widget.tag_configure("namespace", foreground="#9CDCFE")
    
    def _on_key_release(self, event=None):
        """Handle key release events to update syntax highlighting"""
        if event.keysym not in ('F1', 'F2', 'F3', 'F4', 'F5', 'Alt_L', 'Alt_R', 
                               'Control_L', 'Control_R', 'Shift_L', 'Shift_R'):
            self.highlight_syntax()
    
    def highlight_syntax(self):
        """Apply syntax highlighting to the text"""
        # Access the internal Tkinter Text widget
        text_widget = self._textbox
        
        # First remove all existing tags
        for tag in text_widget.tag_names():
            if tag != "sel":  # Keep selection tags
                text_widget.tag_remove(tag, "1.0", "end")
        
        # Get the text content
        content = text_widget.get("1.0", "end-1c")
        
        # Apply C# syntax highlighting
        if self.language == 'csharp':
            self._highlight_csharp(content)
    
    def _highlight_csharp(self, content):
        """Apply C# syntax highlighting"""
        # Access the internal Tkinter Text widget
        text_widget = self._textbox
        
        # Keywords
        keywords = [
            "abstract", "as", "base", "bool", "break", "byte", "case", "catch", 
            "char", "checked", "class", "const", "continue", "decimal", "default", 
            "delegate", "do", "double", "else", "enum", "event", "explicit", 
            "extern", "false", "finally", "fixed", "float", "for", "foreach", 
            "goto", "if", "implicit", "in", "int", "interface", "internal", 
            "is", "lock", "long", "namespace", "new", "null", "object", 
            "operator", "out", "override", "params", "private", "protected", 
            "public", "readonly", "ref", "return", "sbyte", "sealed", 
            "short", "sizeof", "stackalloc", "static", "string", "struct", 
            "switch", "this", "throw", "true", "try", "typeof", "uint", "ulong", 
            "unchecked", "unsafe", "ushort", "using", "virtual", "void", 
            "volatile", "while", "add", "and", "alias", "ascending", "async", 
            "await", "by", "descending", "dynamic", "equals", "from", "get", 
            "global", "group", "into", "join", "let", "nameof", "not", 
            "notnull", "on", "or", "orderby", "partial", "remove", "select", 
            "set", "unmanaged", "value", "var", "when", "where", "with", "yield"
        ]
        
        for keyword in keywords:
            # Find all occurrences of the keyword with word boundaries
            self._highlight_pattern(keyword, "keyword")
        
        # Highlight types
        types = ["int", "string", "bool", "float", "double", "char", "void", 
                "object", "var", "dynamic", "Task", "List", "Dictionary", "IEnumerable"]
        
        for typ in types:
            self._highlight_pattern(typ, "type")
        
        # Highlight strings
        self._highlight_pattern(r'"[^"\\]*(\\.[^"\\]*)*"', "string")
        self._highlight_pattern(r'@"[^"]*(?:""[^"]*)*"', "string")  # Verbatim strings
        self._highlight_pattern(r"'[^'\\]*(\\.[^'\\]*)*'", "string")  # Character literals
        
        # Highlight numbers
        self._highlight_pattern(r'\b\d+\b', "number")
        self._highlight_pattern(r'\b\d+\.\d+\b', "number")
        self._highlight_pattern(r'\b0x[0-9a-fA-F]+\b', "number")  # Hex
        
        # Highlight comments
        self._highlight_pattern(r'//.*$', "comment", line_start=True)
        
        # Multi-line comments (/* ... */)
        content = text_widget.get("1.0", "end-1c")
        comment_start = 0
        while True:
            comment_start = content.find("/*", comment_start)
            if comment_start == -1:
                break
            
            comment_end = content.find("*/", comment_start + 2)
            if comment_end == -1:
                comment_end = len(content)
            else:
                comment_end += 2  # Include the closing */
            
            # Convert string positions to tkinter text positions
            start_line = content[:comment_start].count('\n') + 1
            start_char = comment_start - content[:comment_start].rfind('\n') - 1
            if start_line == 1:
                start_char = comment_start
            
            end_line = content[:comment_end].count('\n') + 1
            end_char = comment_end - content[:comment_end].rfind('\n') - 1
            if end_line == 1:
                end_char = comment_end
            
            text_widget.tag_add("comment", f"{start_line}.{start_char}", f"{end_line}.{end_char}")
            
            comment_start = comment_end
        
        # Highlight class names and method names
        self._highlight_pattern(r'class\s+([A-Za-z0-9_]+)', "class", group=1)
        self._highlight_pattern(r'\b([A-Za-z0-9_]+)\s*\(', "method", group=1)
        
        # Highlight namespaces
        self._highlight_pattern(r'namespace\s+([A-Za-z0-9_.]+)', "namespace", group=1)
        self._highlight_pattern(r'using\s+([A-Za-z0-9_.]+)', "namespace", group=1)
    
    def _highlight_pattern(self, pattern, tag, group=0, line_start=False):
        """Apply a highlight pattern with a specific tag"""
        # Access the internal Tkinter Text widget
        text_widget = self._textbox
        text = text_widget.get("1.0", "end-1c")
        start = "1.0"
        
        # If pattern should match only at line start
        if line_start:
            lines = text.split('\n')
            for i, line in enumerate(lines):
                line_num = i + 1
                match = re.search(pattern, line)
                if match:
                    start_idx = match.start(group)
                    end_idx = match.end(group)
                    text_widget.tag_add(tag, f"{line_num}.{start_idx}", f"{line_num}.{end_idx}")
            return
        
        # For regular patterns that can be anywhere
        while True:
            # Find the next match position
            match_pos = text_widget.search(pattern, start, regexp=True, stopindex="end")
            if not match_pos:
                break
            
            # Get line and char position
            line, char = map(int, match_pos.split('.'))
            
            # Get the text that matched the pattern
            line_text = text_widget.get(f"{line}.0", f"{line}.end")
            match = re.search(pattern, line_text[char:])
            
            if match:
                # Calculate end position
                match_end_char = char + match.end(group)
                match_start_char = char + match.start(group)
                
                # Add the tag
                text_widget.tag_add(tag, f"{line}.{match_start_char}", f"{line}.{match_end_char}")
                
                # Move to the next position
                start = f"{line}.{match_end_char}"
            else:
                # No match found, move to next line
                start = f"{line+1}.0"

class LineNumberedCodeEditor(ctk.CTkFrame):
    """Code editor with line numbers"""
    
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        # Create a frame for line numbers
        self.line_numbers = ctk.CTkTextbox(
            self,
            width=30,
            font=("Consolas", 11),
            fg_color="#2d2d2d",
            text_color="#6e7681",
            border_width=0,
            activate_scrollbars=False
        )
        self.line_numbers.grid(row=0, column=0, sticky="nsew")
        
        # Create the code editor
        self.code_editor = SyntaxHighlightingText(
            self,
            font=("Consolas", 11),
            language='csharp',
            fg_color="#1e1e1e",
            text_color="#d4d4d4", 
            border_width=0,
            wrap="none",
            undo=True,
            height=600  # 높이를 200으로 설정 (원하는 높이로 조정)
        )
        self.code_editor.grid(row=0, column=1, sticky="nsew")
        
        # Configure grid weights
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Bind events
        self.code_editor.bind("<KeyRelease>", self._on_key_release)
        self.code_editor.bind("<Button-1>", self._on_click)
        self.code_editor.bind("<MouseWheel>", self._on_mousewheel)
        
        # Synchronize scrolling
        self.code_editor._textbox.bind("<<Change>>", self._on_change)
        self.code_editor._textbox.bind("<Configure>", self._on_configure)
        
        # Initial update of line numbers
        self._update_line_numbers()
    
    def _on_key_release(self, event):
        """Handle key release event"""
        self._update_line_numbers()
    
    def _on_click(self, event):
        """Handle mouse click event"""
        self._update_line_numbers()
    
    def _on_mousewheel(self, event):
        """Handle mouse wheel event"""
        self._update_line_numbers()
    
    def _on_change(self, event):
        """Handle text change event"""
        self._update_line_numbers()
    
    def _on_configure(self, event):
        """Handle widget resize event"""
        self._update_line_numbers()
    
    def _update_line_numbers(self):
        """Update the line numbers"""
        # Get the text content
        text = self.code_editor.get("1.0", "end-1c")
        lines = text.count('\n') + 1
        
        # Get visible lines
        first_visible = self.code_editor._textbox.index("@0,0")
        first_line = int(first_visible.split('.')[0])
        
        # Create line numbers text
        line_numbers_text = '\n'.join(str(i) for i in range(first_line, lines + 1))
        
        # Update line numbers
        self.line_numbers.configure(state="normal")
        self.line_numbers.delete("1.0", "end")
        self.line_numbers.insert("1.0", line_numbers_text)
        self.line_numbers.configure(state="disabled")
        
        # Synchronize scrolling
        self.line_numbers._textbox.yview_moveto(self.code_editor._textbox.yview()[0])

class CSharpAssistantApp:
    """Improved C# Coding Assistant Application using customtkinter"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("C# 코딩 어시스턴트")
        self.root.geometry("1200x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Color scheme
        self.theme_colors = {
            "dark": {
                "bg_primary": "#1e1e1e",     # Main background (VS Code dark)
                "bg_secondary": "#252526",   # Secondary background
                "bg_tertiary": "#333333",    # Tertiary background
                "code_bg": "#1e1e1e",        # Code editor background
                "text": "#d4d4d4",           # Main text
                "accent": "#0e639c",         # Accent color (VS Code blue)
                "highlight": "#264f78",      # Selection highlight
                "border": "#444444",         # Border color
                "success": "#6a9955",        # Success color
                "error": "#f14c4c",          # Error color
                "warning": "#cca700"         # Warning color
            },
            "light": {
                "bg_primary": "#f8f8f8",     # Main background
                "bg_secondary": "#f0f0f0",   # Secondary background
                "bg_tertiary": "#e0e0e0",    # Tertiary background
                "code_bg": "#ffffff",        # Code editor background
                "text": "#333333",           # Main text
                "accent": "#007acc",         # Accent color (VS Code blue)
                "highlight": "#add6ff",      # Selection highlight
                "border": "#d0d0d0",         # Border color
                "success": "#008000",        # Success color
                "error": "#d83b01",          # Error color
                "warning": "#bf8803"         # Warning color
            }
        }
        
        # Current theme
        self.current_theme = "dark"
        
        # API key
        self.api_key = ""
        self.load_api_key()
        
        # File variables
        self.current_file = None
        self.file_changed = False
        
        # Setup UI
        self.setup_ui()
        
        # Bind keyboard shortcuts
        self.bind_shortcuts()
    
    def setup_ui(self):
        """Create the UI components"""
        # Set UI color mode
        if self.current_theme == "dark":
            ctk.set_appearance_mode("dark")
        else:
            ctk.set_appearance_mode("light")
        
        # Main container frame
        self.main_frame = ctk.CTkFrame(self.root, corner_radius=0)
        self.main_frame.pack(fill="both", expand=True)
        
        # Menu bar
        self.create_menu()
        
        # Toolbar frame
        self.toolbar_frame = ctk.CTkFrame(self.main_frame, corner_radius=0)
        self.toolbar_frame.pack(fill="x", pady=(0, 5))
        
        # Create toolbar buttons
        self.create_toolbar()
        
        # Main content with horizontal layout
        self.main_content_frame = ctk.CTkFrame(self.main_frame)
        self.main_content_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Create two columns for editor and results
        self.main_content_frame.columnconfigure(0, weight=1)
        self.main_content_frame.columnconfigure(1, weight=1)
        
        # Editor frame (left panel)
        self.editor_frame = ctk.CTkFrame(self.main_content_frame)
        self.editor_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # Editor header
        self.editor_header = ctk.CTkFrame(self.editor_frame)
        self.editor_header.pack(fill="x", padx=5, pady=5)
        
        self.editor_label = ctk.CTkLabel(
            self.editor_header,
            text="C# 코드 에디터",
            font=("Segoe UI", 14, "bold")
        )
        self.editor_label.pack(side="left")
        
        # Code editor with line numbers
        self.code_editor_frame = LineNumberedCodeEditor(self.editor_frame)
        self.code_editor_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Get a reference to the code editor
        self.code_editor = self.code_editor_frame.code_editor
        
        # Results frame (right panel)
        self.results_frame = ctk.CTkFrame(self.main_content_frame)
        self.results_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        # Results header
        self.results_header = ctk.CTkFrame(self.results_frame)
        self.results_header.pack(fill="x", padx=5, pady=5)
        
        self.results_label = ctk.CTkLabel(
            self.results_header,
            text="코드 제안 결과",
            font=("Segoe UI", 14, "bold")
        )
        self.results_label.pack(side="left")
        
        # Results control buttons
        self.results_control_frame = ctk.CTkFrame(self.results_header)
        self.results_control_frame.pack(side="right")
        
        self.apply_btn = ctk.CTkButton(
            self.results_control_frame,
            text="제안 적용",
            command=self.apply_suggestion,
            width=100
        )
        self.apply_btn.pack(side="left", padx=5)
        
        # Results text area
        self.results_text = ctk.CTkTextbox(
            self.results_frame,
            font=("Consolas", 11),
            wrap="word"
        )
        self.results_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Bottom frame for prompt
        self.bottom_frame = ctk.CTkFrame(self.main_frame)
        self.bottom_frame.pack(fill="x", padx=10, pady=10)
        
        # Prompt label
        self.prompt_label = ctk.CTkLabel(
            self.bottom_frame,
            text="요청 내용:",
            font=("Segoe UI", 12)
        )
        self.prompt_label.pack(side="left", padx=5, pady=5)
        
        # Prompt entry
        self.prompt_entry = ctk.CTkTextbox(
            self.bottom_frame,
            height=60,
            font=("Segoe UI", 12),
            wrap="word"
        )
        self.prompt_entry.pack(side="left", fill="x", expand=True, padx=5, pady=5)
        
        # Helper text in prompt
        self.prompt_entry.insert("1.0", "여기에 요청사항을 입력하세요...")
        self.prompt_entry.bind("<FocusIn>", self._on_prompt_focus_in)
        self.prompt_entry.bind("<FocusOut>", self._on_prompt_focus_out)
        
        # Submit button
        self.submit_btn = ctk.CTkButton(
            self.bottom_frame,
            text="코드 제안 받기",
            command=self.get_suggestion,
            width=120,
            height=40,
            font=("Segoe UI", 12, "bold")
        )
        self.submit_btn.pack(side="right", padx=5, pady=5)
        
        # Status bar
        self.status_bar = ctk.CTkLabel(
            self.main_frame,
            text="준비",
            anchor="w",
            font=("Segoe UI", 10),
            corner_radius=0,
            height=25
        )
        self.status_bar.pack(fill="x", side="bottom")
        
        # Set initial text in editor
        self.code_editor.insert("1.0", "using System;\n\nnamespace MyApp\n{\n    class Program\n    {\n        static void Main(string[] args)\n        {\n            Console.WriteLine(\"Hello, World!\");\n        }\n    }\n}")
        self.code_editor.highlight_syntax()
    
    def _on_prompt_focus_in(self, event):
        """Handle focus in event for prompt entry"""
        if self.prompt_entry.get("1.0", "end-1c") == "여기에 요청사항을 입력하세요...":
            self.prompt_entry.delete("1.0", "end")
    
    def _on_prompt_focus_out(self, event):
        """Handle focus out event for prompt entry"""
        if not self.prompt_entry.get("1.0", "end-1c"):
            self.prompt_entry.insert("1.0", "여기에 요청사항을 입력하세요...")
    
    def create_toolbar(self):
        """Create toolbar buttons"""
        # New file button
        self.new_btn = ctk.CTkButton(
            self.toolbar_frame,
            text="새 파일",
            command=self.new_file,
            width=80,
            height=28
        )
        self.new_btn.pack(side="left", padx=5, pady=5)
        
        # Open file button
        self.open_btn = ctk.CTkButton(
            self.toolbar_frame,
            text="열기",
            command=self.open_file,
            width=80,
            height=28
        )
        self.open_btn.pack(side="left", padx=5, pady=5)
        
        # Save file button
        self.save_btn = ctk.CTkButton(
            self.toolbar_frame,
            text="저장",
            command=self.save_file,
            width=80,
            height=28
        )
        self.save_btn.pack(side="left", padx=5, pady=5)
        
        # Separator
        self.separator = ctk.CTkFrame(
            self.toolbar_frame,
            width=2,
            height=28,
            fg_color=self.theme_colors[self.current_theme]["border"]
        )
        self.separator.pack(side="left", padx=10, pady=5)
        
        # Theme selection
        self.theme_label = ctk.CTkLabel(
            self.toolbar_frame,
            text="테마:",
            font=("Segoe UI", 12)
        )
        self.theme_label.pack(side="left", padx=5, pady=5)
        
        self.theme_var = ctk.StringVar(value="다크")
        self.theme_option = ctk.CTkOptionMenu(
            self.toolbar_frame,
            values=["다크", "라이트"],
            variable=self.theme_var,
            command=self.change_theme,
            width=80,
            height=28
        )
        self.theme_option.pack(side="left", padx=5, pady=5)
    
    def create_menu(self):
        """Create menu bar"""
        self.menu = tk.Menu(self.root, bg=self.theme_colors[self.current_theme]["bg_primary"], 
                           fg=self.theme_colors[self.current_theme]["text"],
                           activebackground=self.theme_colors[self.current_theme]["highlight"],
                           activeforeground=self.theme_colors[self.current_theme]["text"],
                           bd=0)
        
        # File menu
        self.file_menu = tk.Menu(self.menu, tearoff=0, 
                               bg=self.theme_colors[self.current_theme]["bg_primary"],
                               fg=self.theme_colors[self.current_theme]["text"],
                               activebackground=self.theme_colors[self.current_theme]["highlight"],
                               activeforeground=self.theme_colors[self.current_theme]["text"])
        self.file_menu.add_command(label="새 파일 (Ctrl+N)", command=self.new_file)
        self.file_menu.add_command(label="열기... (Ctrl+O)", command=self.open_file)
        self.file_menu.add_command(label="저장 (Ctrl+S)", command=self.save_file)
        self.file_menu.add_command(label="다른 이름으로 저장...", command=self.save_file_as)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="종료", command=self.on_closing)
        self.menu.add_cascade(label="파일", menu=self.file_menu)
        
        # Edit menu
        self.edit_menu = tk.Menu(self.menu, tearoff=0, 
                               bg=self.theme_colors[self.current_theme]["bg_primary"],
                               fg=self.theme_colors[self.current_theme]["text"],
                               activebackground=self.theme_colors[self.current_theme]["highlight"],
                               activeforeground=self.theme_colors[self.current_theme]["text"])
        self.edit_menu.add_command(label="실행 취소 (Ctrl+Z)", 
                                 command=lambda: self.code_editor._textbox.event_generate("<<Undo>>"))
        self.edit_menu.add_command(label="다시 실행 (Ctrl+Y)", 
                                 command=lambda: self.code_editor._textbox.event_generate("<<Redo>>"))
        self.edit_menu.add_separator()
        self.edit_menu.add_command(label="잘라내기 (Ctrl+X)", 
                                 command=lambda: self.code_editor._textbox.event_generate("<<Cut>>"))
        self.edit_menu.add_command(label="복사 (Ctrl+C)", 
                                 command=lambda: self.code_editor._textbox.event_generate("<<Copy>>"))
        self.edit_menu.add_command(label="붙여넣기 (Ctrl+V)", 
                                 command=lambda: self.code_editor._textbox.event_generate("<<Paste>>"))
        self.menu.add_cascade(label="편집", menu=self.edit_menu)
        
        # AI Helper menu
        self.ai_menu = tk.Menu(self.menu, tearoff=0, 
                             bg=self.theme_colors[self.current_theme]["bg_primary"],
                             fg=self.theme_colors[self.current_theme]["text"],
                             activebackground=self.theme_colors[self.current_theme]["highlight"],
                             activeforeground=self.theme_colors[self.current_theme]["text"])
        self.ai_menu.add_command(label="코드 제안 받기 (F5)", command=self.get_suggestion)
        self.ai_menu.add_separator()
        self.ai_menu.add_command(label="코드 최적화", 
                               command=lambda: self.get_suggestion_with_preset("이 코드를 최적화해주세요."))
        self.ai_menu.add_command(label="버그 찾기", 
                               command=lambda: self.get_suggestion_with_preset("이 코드에서 버그나 오류를 찾아주세요."))
        self.ai_menu.add_command(label="주석 추가", 
                               command=lambda: self.get_suggestion_with_preset("이 코드에 상세한 주석을 추가해주세요."))
        self.ai_menu.add_command(label="코드 리팩토링", 
                               command=lambda: self.get_suggestion_with_preset("이 코드를 더 깔끔하게 리팩토링해주세요."))
        self.menu.add_cascade(label="AI 도우미", menu=self.ai_menu)
        
        # Settings menu
        self.settings_menu = tk.Menu(self.menu, tearoff=0, 
                                   bg=self.theme_colors[self.current_theme]["bg_primary"],
                                   fg=self.theme_colors[self.current_theme]["text"],
                                   activebackground=self.theme_colors[self.current_theme]["highlight"],
                                   activeforeground=self.theme_colors[self.current_theme]["text"])
        self.settings_menu.add_command(label="API 키 설정", command=self.show_api_key_dialog)
        
        # Theme submenu
        self.theme_menu = tk.Menu(self.settings_menu, tearoff=0, 
                                bg=self.theme_colors[self.current_theme]["bg_primary"],
                                fg=self.theme_colors[self.current_theme]["text"],
                                activebackground=self.theme_colors[self.current_theme]["highlight"],
                                activeforeground=self.theme_colors[self.current_theme]["text"])
        self.theme_menu.add_command(label="라이트 테마", command=lambda: self.change_theme("라이트"))
        self.theme_menu.add_command(label="다크 테마", command=lambda: self.change_theme("다크"))
        self.settings_menu.add_cascade(label="테마 설정", menu=self.theme_menu)
        
        self.menu.add_cascade(label="설정", menu=self.settings_menu)
        
        # Help menu
        self.help_menu = tk.Menu(self.menu, tearoff=0, 
                               bg=self.theme_colors[self.current_theme]["bg_primary"],
                               fg=self.theme_colors[self.current_theme]["text"],
                               activebackground=self.theme_colors[self.current_theme]["highlight"],
                               activeforeground=self.theme_colors[self.current_theme]["text"])
        self.help_menu.add_command(label="사용법", command=self.show_help)
        self.help_menu.add_command(label="정보", command=self.show_about)
        self.menu.add_cascade(label="도움말", menu=self.help_menu)
        
        # Set menu to root
        self.root.config(menu=self.menu)
    
    def bind_shortcuts(self):
        """Bind keyboard shortcuts"""
        self.root.bind("<Control-n>", lambda event: self.new_file())
        self.root.bind("<Control-o>", lambda event: self.open_file())
        self.root.bind("<Control-s>", lambda event: self.save_file())
        self.root.bind("<F5>", lambda event: self.get_suggestion())
        
        # Enter key in prompt (with Ctrl for submission)
        self.prompt_entry.bind("<Control-Return>", lambda event: self.get_suggestion())
    
    def load_api_key(self):
        """Load API key from config file"""
        config_path = Path.home() / ".csharp_assistant" / "config.json"
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self.api_key = config.get("api_key", "")
            except Exception as e:
                messagebox.showerror("오류", f"설정 파일 로드 중 오류 발생: {e}")
    
    def save_api_key(self):
        """Save API key to config file"""
        config_path = Path.home() / ".csharp_assistant"
        config_path.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_path / "config.json", 'w') as f:
                json.dump({"api_key": self.api_key}, f)
            messagebox.showinfo("성공", "API 키가 저장되었습니다.")
        except Exception as e:
            messagebox.showerror("오류", f"설정 파일 저장 중 오류 발생: {e}")
    
    def show_api_key_dialog(self):
        """Show API key dialog"""
        dialog = ctk.CTkToplevel(self.root)
        dialog.title("API 키 설정")
        dialog.geometry("450x180")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.focus_set()
        
        # Center dialog
        dialog.update_idletasks()
        width = dialog.winfo_width()
        height = dialog.winfo_height()
        x = (self.root.winfo_width() // 2) - (width // 2) + self.root.winfo_x()
        y = (self.root.winfo_height() // 2) - (height // 2) + self.root.winfo_y()
        dialog.geometry(f"{width}x{height}+{x}+{y}")
        
        # Dialog content
        frame = ctk.CTkFrame(dialog)
        frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Header
        header = ctk.CTkLabel(
            frame, 
            text="JUN BLACK/J API 키 설정",
            font=("Segoe UI", 16, "bold")
        )
        header.pack(anchor="w", pady=(0, 15))
        
        # Description
        desc = ctk.CTkLabel(
            frame, 
            text="JUN BLACK/J 사용을 위한 API 키를 입력하세요.",
            font=("Segoe UI", 12),
            wraplength=400
        )
        desc.pack(anchor="w", pady=(0, 15))
        
        # API key entry
        api_key_var = tk.StringVar(value=self.api_key)
        api_key_frame = ctk.CTkFrame(frame)
        api_key_frame.pack(fill="x", pady=5)
        
        api_key_label = ctk.CTkLabel(
            api_key_frame,
            text="API 키:",
            font=("Segoe UI", 12),
            width=80
        )
        api_key_label.pack(side="left", padx=5)
        
        api_key_entry = ctk.CTkEntry(
            api_key_frame,
            textvariable=api_key_var,
            width=300,
            show="*",
            font=("Segoe UI", 12)
        )
        api_key_entry.pack(side="left", padx=5, fill="x", expand=True)
        
        # Button frame
        btn_frame = ctk.CTkFrame(frame)
        btn_frame.pack(fill="x", pady=(15, 0), anchor="e")
        
        # Cancel button
        cancel_btn = ctk.CTkButton(
            btn_frame,
            text="취소",
            command=dialog.destroy,
            width=100
        )
        cancel_btn.pack(side="right", padx=5)
        
        # Save button
        save_btn = ctk.CTkButton(
            btn_frame,
            text="저장",
            command=lambda: self.save_api_key_from_dialog(api_key_var.get(), dialog),
            width=100
        )
        save_btn.pack(side="right", padx=5)
    
    def save_api_key_from_dialog(self, api_key, dialog):
        """Save API key from dialog"""
        self.api_key = api_key
        self.save_api_key()
        dialog.destroy()
    
    def new_file(self):
        """Create a new file"""
        if self.file_changed:
            if messagebox.askyesno("확인", "현재 내용을 저장하지 않고 새 파일을 생성하시겠습니까?"):
                self.code_editor.delete("1.0", "end")
                self.results_text.delete("1.0", "end")
                self.current_file = None
                self.file_changed = False
                self.status_bar.configure(text="새 파일")
        else:
            self.code_editor.delete("1.0", "end")
            self.results_text.delete("1.0", "end")
            self.current_file = None
            self.file_changed = False
            self.status_bar.configure(text="새 파일")
    
    def open_file(self):
        """Open a file"""
        file_path = filedialog.askopenfilename(
            filetypes=[("C# 파일", "*.cs"), ("모든 파일", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    self.code_editor.delete("1.0", "end")
                    self.code_editor.insert("1.0", content)
                    self.code_editor.highlight_syntax()
                    self.current_file = file_path
                    self.file_changed = False
                    self.status_bar.configure(text=f"파일 열기: {file_path}")
            except Exception as e:
                messagebox.showerror("오류", f"파일 열기 중 오류 발생: {e}")
    
    def save_file(self):
        """Save the current file"""
        if self.current_file:
            try:
                content = self.code_editor.get("1.0", "end-1c")
                with open(self.current_file, 'w', encoding='utf-8') as file:
                    file.write(content)
                self.file_changed = False
                self.status_bar.configure(text=f"파일 저장됨: {self.current_file}")
            except Exception as e:
                messagebox.showerror("오류", f"파일 저장 중 오류 발생: {e}")
        else:
            self.save_file_as()
    
    def save_file_as(self):
        """Save as a new file"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".cs",
            filetypes=[("C# 파일", "*.cs"), ("모든 파일", "*.*")]
        )
        if file_path:
            try:
                content = self.code_editor.get("1.0", "end-1c")
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(content)
                self.current_file = file_path
                self.file_changed = False
                self.status_bar.configure(text=f"파일 저장됨: {file_path}")
            except Exception as e:
                messagebox.showerror("오류", f"파일 저장 중 오류 발생: {e}")
    
    def get_suggestion_with_preset(self, preset_prompt):
        """Get code suggestion with preset prompt"""
        # Clear any placeholder
        if self.prompt_entry.get("1.0", "end-1c") == "여기에 요청사항을 입력하세요...":
            self.prompt_entry.delete("1.0", "end")
        
        # Set the preset prompt
        self.prompt_entry.delete("1.0", "end")
        self.prompt_entry.insert("1.0", preset_prompt)
        
        # Get suggestion
        self.get_suggestion()
    
    def get_suggestion(self):
        """Get code suggestion from JUN BLACK/J API"""
        if not self.api_key:
            messagebox.showwarning("경고", "API 키가 설정되지 않았습니다. 설정 메뉴에서 API 키를 설정하세요.")
            return
        
        code = self.code_editor.get("1.0", "end-1c").strip()
        prompt = self.prompt_entry.get("1.0", "end-1c").strip()
        
        # Check for placeholder text
        if prompt == "여기에 요청사항을 입력하세요...":
            prompt = ""
        
        if not code:
            messagebox.showwarning("경고", "코드가 입력되지 않았습니다.")
            return
        
        if not prompt:
            prompt = "이 코드를 분석하고 개선점이나 최적화 방안을 제안해주세요."
        
        # Show progress dialog
        self.show_progress_dialog("요청 처리 중", "JUN BLACK/J API에 요청을 처리 중입니다...")
        
        # Disable submit button
        self.submit_btn.configure(state="disabled")
        self.status_bar.configure(text="API 호출 중...")
        
        # Call API in a separate thread
        threading.Thread(target=self.call_claude_api, args=(code, prompt)).start()
    
    def show_progress_dialog(self, title, message):
        """Show a progress dialog"""
        self.progress_dialog = ctk.CTkToplevel(self.root)
        self.progress_dialog.title(title)
        self.progress_dialog.geometry("350x150")
        self.progress_dialog.resizable(False, False)
        self.progress_dialog.transient(self.root)
        self.progress_dialog.grab_set()
        
        # Center dialog
        self.progress_dialog.update_idletasks()
        width = self.progress_dialog.winfo_width()
        height = self.progress_dialog.winfo_height()
        x = (self.root.winfo_width() // 2) - (width // 2) + self.root.winfo_x()
        y = (self.root.winfo_height() // 2) - (height // 2) + self.root.winfo_y()
        self.progress_dialog.geometry(f"{width}x{height}+{x}+{y}")
        
        # Dialog content
        frame = ctk.CTkFrame(self.progress_dialog)
        frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Message
        message_label = ctk.CTkLabel(
            frame,
            text=message,
            font=("Segoe UI", 12),
            wraplength=300
        )
        message_label.pack(pady=(0, 20))
        
        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(
            frame,
            mode="indeterminate",
            width=300
        )
        self.progress_bar.pack(pady=10)
        self.progress_bar.start()
    
    def call_claude_api(self, code, prompt):
        """Call the JUN BLACK/J API"""
        try:
            client = anthropic.Anthropic(api_key=self.api_key)
            
            response = client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=4000,
                system="당신은 C# 전문가로서 코드 분석, 개선, 버그 수정, 최적화 등을 제안하는 어시스턴트입니다. 제안 코드는 항상 실행 가능하고 정확해야 합니다. 코드 블록은 항상 ```csharp와 ``` 사이에 표시하세요.",
                messages=[
                    {"role": "user", "content": f"다음 C# 코드를 분석해주세요:\n\n```csharp\n{code}\n```\n\n요청사항: {prompt}"}
                ]
            )
            
            # Update UI in the main thread
            self.root.after(0, self.update_result, response.content[0].text)
        except Exception as e:
            self.root.after(0, self.show_error, str(e))
    
    def update_result(self, result):
        """Update result in the UI"""
        # Close progress dialog
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.destroy()
        
        # If no markdown code blocks in result, wrap it
        if "```" not in result:
            # Look for "개선된 코드" or similar header
            pattern = re.compile(r"(#\s*개선된\s*코드)([\s\S]+)", re.IGNORECASE)
            match = pattern.search(result)
            if match:
                heading = match.group(1)
                code_text = match.group(2).strip()
                # Convert to markdown code block
                result = result.replace(match.group(0), f"{heading}\n```csharp\n{code_text}\n```")
            else:
                # Wrap the entire response in a code block
                result = f"```csharp\n{result.strip()}\n```"
        
        # Update results text
        self.results_text.delete("1.0", "end")
        self.results_text.insert("1.0", result)
        
        # Apply highlighting to code blocks
        self.highlight_code_blocks_in_results()
        
        # Re-enable submit button
        self.submit_btn.configure(state="normal")
        self.status_bar.configure(text="코드 제안 완료")
    
    # def highlight_code_blocks_in_results(self):
    #     """Highlight code blocks in the results text"""
    #     # Configure code block tags
    #     self.results_text.tag_configure("code_block", 
    #                                   font=("Consolas", 11),
    #                                   background=self.theme_colors[self.current_theme]["bg_tertiary"])
        
    #     # Configure syntax highlighting tags
    #     self.results_text.tag_configure("keyword", foreground="#569CD6")
    #     self.results_text.tag_configure("type", foreground="#4EC9B0")
    #     self.results_text.tag_configure("string", foreground="#CE9178")
    #     self.results_text.tag_configure("comment", foreground="#6A9955")
    #     self.results_text.tag_configure("number", foreground="#B5CEA8")
        
    #     # Find code blocks
    #     content = self.results_text.get("1.0", "end-1c")
        
    #     # Find all code blocks with the pattern ```csharp ... ```
    #     pattern = re.compile(r'```(?:csharp)?\s*\n([\s\S]*?)\n```')
        
    #     # Process each match
    #     for match in pattern.finditer(content):
    #         start_pos = match.start()
    #         end_pos = match.end()
            
    #         # Convert string positions to tkinter positions
    #         start_line = content[:start_pos].count('\n') + 1
    #         start_col = start_pos - content[:start_pos].rfind('\n') - 1
    #         if start_line == 1:
    #             start_col = start_pos
            
    #         end_line = content[:end_pos].count('\n') + 1
    #         end_col = end_pos - content[:end_pos].rfind('\n') - 1
    #         if end_line == 1:
    #             end_col = end_pos
            
    #         # Add code block tag
    #         self.results_text.tag_add("code_block", f"{start_line}.{start_col}", f"{end_line}.{end_col}")
            
    #         # Get the code content
    #         code_content = match.group(1)
            
    #         # Highlight C# syntax within the code block
    #         self._highlight_code_within_block(code_content, start_line, start_col, end_line, end_col)
    
    def highlight_code_blocks_in_results(self):
        """향상된 C# 코드 블록 하이라이팅"""
        content = self.results_text.get("1.0", "end-1c")
        
        # 테마별 하이라이팅 색상 팔레트
        theme_colors = {
            "light": {
                "background": "#f4f4f4",
                "text": "#000000",
                "keyword": "#0000FF",        # 파란색 키워드
                "string": "#A31515",         # 다크 레드 문자열
                "comment": "#008000",        # 녹색 주석
                "type": "#2B91AF",           # 파란 청록색 타입
                "number": "#098658",         # 어두운 청록색 숫자
                "method": "#795E26",         # 갈색 메서드
            },
            "dark": {
                "background": "#1E1E1E",     # VS Code 다크 테마 배경
                "text": "#D4D4D4",           # 밝은 회색 텍스트
                "keyword": "#569CD6",        # 밝은 파란색 키워드
                "string": "#CE9178",         # 연한 주황색 문자열
                "comment": "#6A9955",        # 연한 녹색 주석
                "type": "#4EC9B0",           # 청록색 타입
                "number": "#B5CEA8",         # 연한 녹색 숫자
                "method": "#DCDCAA",         # 밝은 노란색 메서드
            }
        }
        
        # 현재 테마의 색상 선택
        colors = theme_colors[self.current_theme]
        
        # 코드 블록 찾기
        pattern = re.compile(r'```(?:csharp)?\s*\n([\s\S]*?)\n```')
        matches = list(pattern.finditer(content))
        
        if not matches:
            return
        
        # 텍스트 초기화
        self.results_text.delete("1.0", "end")
        
        # 마크다운 스타일 태그 설정
        markdown_header_style = {
            "font": ("Consolas", 12, "bold"),
            "foreground": colors["text"]
        }
        
        # 코드 블록 스타일
        code_block_style = {
            "font": ("Consolas", 10),
            "background": colors["background"],
            "foreground": colors["text"]
        }
        
        # 구문 강조 키워드
        keywords = [
            "abstract", "as", "base", "bool", "break", "byte", "case", "catch", 
            "char", "checked", "class", "const", "continue", "decimal", "default", 
            "delegate", "do", "double", "else", "enum", "event", "explicit", 
            "extern", "false", "finally", "fixed", "float", "for", "foreach", 
            "goto", "if", "implicit", "in", "int", "interface", "internal", 
            "is", "lock", "long", "namespace", "new", "null", "object", 
            "operator", "out", "override", "params", "private", "protected", 
            "public", "readonly", "ref", "return", "sbyte", "sealed", 
            "short", "sizeof", "stackalloc", "static", "string", "struct", 
            "switch", "this", "throw", "true", "try", "typeof", "uint", "ulong", 
            "unchecked", "unsafe", "ushort", "using", "virtual", "void", 
            "volatile", "while", "var", "async", "await"
        ]
        
        last_end = 0
        for match in matches:
            # 코드 블록 이전 텍스트 추가
            if match.start() > last_end:
                self.results_text.insert("end", content[last_end:match.start()], 
                                         {"font": ("Segoe UI", 10)})
            
            # 코드 블록 추가
            code_block = match.group(1).strip()
            
            # 코드 블록 헤더
            self.results_text.insert("end", "## 코드 블록\n", markdown_header_style)
            
            # 구분선
            self.results_text.insert("end", "---\n", 
                                     {"foreground": "#999999", 
                                      "font": ("Segoe UI", 10, "bold")})
            
            # 코드 블록 본문 삽입 (구문 강조 포함)
            current_tag = 0
            
            # 각 줄에 대해 구문 강조 적용
            for line in code_block.split('\n'):
                # 키워드 강조
                for keyword in keywords:
                    line = re.sub(r'\b' + keyword + r'\b', 
                                  f'<{current_tag}_keyword>{keyword}</{current_tag}_keyword>', 
                                  line)
                
                # 문자열 강조 (큰따옴표, 작은따옴표)
                line = re.sub(r'"[^"]*"', 
                              lambda m: f'<{current_tag}_string>{m.group(0)}</{current_tag}_string>', 
                              line)
                line = re.sub(r"'[^']*'", 
                              lambda m: f'<{current_tag}_string>{m.group(0)}</{current_tag}_string>', 
                              line)
                
                # 주석 강조
                line = re.sub(r'//.*$', 
                              lambda m: f'<{current_tag}_comment>{m.group(0)}</{current_tag}_comment>', 
                              line)
                
                # 숫자 강조
                line = re.sub(r'\b\d+\b', 
                              lambda m: f'<{current_tag}_number>{m.group(0)}</{current_tag}_number>', 
                              line)
                
                # 태그 삽입
                self.results_text.tag_config(f'{current_tag}_keyword', 
                                             foreground=colors["keyword"])
                self.results_text.tag_config(f'{current_tag}_string', 
                                             foreground=colors["string"])
                self.results_text.tag_config(f'{current_tag}_comment', 
                                             foreground=colors["comment"])
                self.results_text.tag_config(f'{current_tag}_number', 
                                             foreground=colors["number"])
                
                # 라인 삽입
                self.results_text.insert("end", line + "\n", code_block_style)
                
                # 태그 적용
                start_line, start_char = map(int, self.results_text.index("end-2l").split('.'))
                end_line, end_char = map(int, self.results_text.index("end-1c").split('.'))
                
                for tag_type in ['keyword', 'string', 'comment', 'number']:
                    pattern = rf'<{current_tag}_{tag_type}>(.+?)</{current_tag}_{tag_type}>'
                    for match in re.finditer(pattern, line):
                        start = match.start(1)
                        end = match.end(1)
                        self.results_text.tag_add(
                            f'{current_tag}_{tag_type}', 
                            f"{start_line}.{start_char + start}", 
                            f"{start_line}.{start_char + end}"
                        )
                
                current_tag += 1
            
            # 코드 블록 후 구분선
            self.results_text.insert("end", "---\n\n", 
                                     {"foreground": "#999999", 
                                      "font": ("Segoe UI", 10, "bold")})
            
            last_end = match.end()
        
        # 마지막 남은 텍스트 추가
        if last_end < len(content):
            self.results_text.insert("end", content[last_end:], 
                                     {"font": ("Segoe UI", 10)})
            
    def _highlight_code_within_block(self, code, start_line, start_col, end_line, end_col):
        """Apply syntax highlighting within a code block"""
        # Keywords
        keywords = [
            "abstract", "as", "base", "bool", "break", "byte", "case", "catch", 
            "char", "checked", "class", "const", "continue", "decimal", "default", 
            "delegate", "do", "double", "else", "enum", "event", "explicit", 
            "extern", "false", "finally", "fixed", "float", "for", "foreach", 
            "goto", "if", "implicit", "in", "int", "interface", "internal", 
            "is", "lock", "long", "namespace", "new", "null", "object", 
            "operator", "out", "override", "params", "private", "protected", 
            "public", "readonly", "ref", "return", "sbyte", "sealed", 
            "short", "sizeof", "stackalloc", "static", "string", "struct", 
            "switch", "this", "throw", "true", "try", "typeof", "uint", "ulong", 
            "unchecked", "unsafe", "ushort", "using", "virtual", "void", 
            "volatile", "while"
        ]
        
        # Start after the opening ```csharp tag
        start_pos = f"{start_line}.{start_col + 10}"  # Adjust for ```csharp
        
        # For each keyword, search and tag
        for keyword in keywords:
            pos = start_pos
            pattern = r'\b' + keyword + r'\b'
            
            while True:
                # Find the next occurrence
                match_pos = self.results_text.search(pattern, pos, f"{end_line}.{end_col}", 
                                                   regexp=True, stopindex=f"{end_line}.{end_col}")
                if not match_pos:
                    break
                
                # Add tag
                end_match_pos = f"{match_pos}+{len(keyword)}c"
                self.results_text.tag_add("keyword", match_pos, end_match_pos)
                
                # Move to next position
                pos = end_match_pos
    
    def show_error(self, error_message):
        """Show error message"""
        # Close progress dialog
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.destroy()
        
        messagebox.showerror("API 오류", f"JUN BLACK/J API 호출 중 오류 발생: {error_message}")
        self.status_bar.configure(text="오류 발생")
        self.submit_btn.configure(state="normal")
    
    def apply_suggestion(self):
        """Apply suggestion to the editor"""
        result = self.results_text.get("1.0", "end-1c")
        
        # Find code blocks
        code_blocks = re.findall(r'```(?:csharp)?\s*([\s\S]*?)```', result)
        
        if code_blocks:
            # Use the first code block
            code = code_blocks[0].strip()
            
            # Confirm with user
            if messagebox.askyesno("확인", "제안된 코드를 에디터에 적용하시겠습니까?"):
                self.code_editor.delete("1.0", "end")
                self.code_editor.insert("1.0", code)
                self.code_editor.highlight_syntax()
                self.file_changed = True
                self.status_bar.configure(text="제안 코드가 적용되었습니다")
        else:
            messagebox.showinfo("정보", "제안 결과에서 코드 블록을 찾을 수 없습니다.")
    
    def change_theme(self, theme):
        """Change application theme"""
        if theme == "라이트":
            self.current_theme = "light"
            ctk.set_appearance_mode("light")
        else:
            self.current_theme = "dark"
            ctk.set_appearance_mode("dark")
        
        # Update UI components
        self.status_bar.configure(text=f"{theme} 테마로 변경되었습니다")
        
        # Re-highlight code
        self.code_editor.highlight_syntax()
        self.highlight_code_blocks_in_results()
    
    def show_help(self):
        """Show help dialog"""
        help_dialog = ctk.CTkToplevel(self.root)
        help_dialog.title("C# 코딩 어시스턴트 사용법")
        help_dialog.geometry("600x400")
        help_dialog.resizable(True, True)
        help_dialog.transient(self.root)
        help_dialog.grab_set()
        
        # Center dialog
        help_dialog.update_idletasks()
        width = help_dialog.winfo_width()
        height = help_dialog.winfo_height()
        x = (self.root.winfo_width() // 2) - (width // 2) + self.root.winfo_x()
        y = (self.root.winfo_height() // 2) - (height // 2) + self.root.winfo_y()
        help_dialog.geometry(f"{width}x{height}+{x}+{y}")
        
        # Dialog content
        frame = ctk.CTkFrame(help_dialog)
        frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        title = ctk.CTkLabel(
            frame,
            text="C# 코딩 어시스턴트 사용법",
            font=("Segoe UI", 18, "bold")
        )
        title.pack(anchor="w", pady=(0, 20))
        
        # Help content
        help_text = ctk.CTkTextbox(
            frame,
            wrap="word",
            font=("Segoe UI", 12)
        )
        help_text.pack(fill="both", expand=True, pady=(0, 10))
        
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

4. 단축키
   - F5: 코드 제안 받기
   - Ctrl+Z: 실행 취소
   - Ctrl+Y: 다시 실행
   - Ctrl+X: 잘라내기
   - Ctrl+C: 복사
   - Ctrl+V: 붙여넣기

주의: 제안된 코드는 항상 검토 후 사용하시기 바랍니다.
        """
        
        help_text.insert("1.0", help_content)
        help_text.configure(state="disabled")
        
        # Close button
        close_btn = ctk.CTkButton(
            frame,
            text="닫기",
            command=help_dialog.destroy,
            width=100
        )
        close_btn.pack(side="right")
    
    def show_about(self):
        """Show about dialog"""
        about_dialog = ctk.CTkToplevel(self.root)
        about_dialog.title("C# 코딩 어시던트")
        about_dialog.geometry("400x300")
        about_dialog.resizable(False, False)
        about_dialog.transient(self.root)
        about_dialog.grab_set()
        
        # Center dialog
        about_dialog.update_idletasks()
        width = about_dialog.winfo_width()
        height = about_dialog.winfo_height()
        x = (self.root.winfo_width() // 2) - (width // 2) + self.root.winfo_x()
        y = (self.root.winfo_height() // 2) - (height // 2) + self.root.winfo_y()
        about_dialog.geometry(f"{width}x{height}+{x}+{y}")
        
        # Dialog content
        frame = ctk.CTkFrame(about_dialog)
        frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        title = ctk.CTkLabel(
            frame,
            text="C# 코딩 어시스턴트",
            font=("Segoe UI", 20, "bold")
        )
        title.pack(pady=(0, 10))
        
        # Version
        version = ctk.CTkLabel(
            frame,
            text="버전 2.0.0",
            font=("Segoe UI", 12)
        )
        version.pack(pady=(0, 20))
        
        # Description
        desc = ctk.CTkLabel(
            frame,
            text="JUN BLACK/J API를 활용한 C# 코드 분석 및 제안 도구입니다.\n"
                "코드 최적화, 버그 찾기, 리팩토링 등의 작업에 도움을 줍니다.",
            font=("Segoe UI", 12),
            wraplength=350
        )
        desc.pack(pady=(0, 20))
        
        # Credits
        credits = ctk.CTkLabel(
            frame,
            text="개발: customtkinter + JUN BLACK/J",
            font=("Segoe UI", 12)
        )
        credits.pack(pady=(0, 20))
        
        # Close button
        close_btn = ctk.CTkButton(
            frame,
            text="닫기",
            command=about_dialog.destroy,
            width=100
        )
        close_btn.pack(side="bottom")
    
    def on_closing(self):
        """Handle application closing"""
        if self.file_changed:
            if messagebox.askyesno("확인", "저장되지 않은 변경사항이 있습니다. 종료하시겠습니까?"):
                self.root.destroy()
        else:
            self.root.destroy()


if __name__ == "__main__":
    # Enable high DPI support
    ctk.deactivate_automatic_dpi_awareness()
    
    # Create the application
    root = ctk.CTk()
    app = CSharpAssistantApp(root)
    root.mainloop()