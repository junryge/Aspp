import tkinter as tk
from tkinter import ttk, filedialog, messagebox, font
import markdown
from markdown.extensions import fenced_code, tables, toc, nl2br
import webbrowser
from tkinterweb import HtmlFrame
import os
import sys

class MarkdownViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Markdown Viewer - 마크다운 뷰어")
        self.root.geometry("1200x800")
        
        # 현재 파일 경로
        self.current_file = None
        
        # 스타일 설정
        self.setup_styles()
        
        # UI 구성
        self.setup_ui()
        
        # 단축키 설정
        self.setup_shortcuts()
        
    def setup_styles(self):
        """테마와 스타일 설정"""
        self.root.configure(bg='#f0f0f0')
        style = ttk.Style()
        style.theme_use('clam')
        
        # 커스텀 스타일
        style.configure('Title.TLabel', font=('Segoe UI', 12, 'bold'))
        style.configure('Custom.TButton', padding=5)
        
    def setup_ui(self):
        """UI 구성"""
        # 메뉴바 생성
        self.create_menubar()
        
        # 툴바 생성
        self.create_toolbar()
        
        # 메인 프레임 (PanedWindow로 분할)
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 왼쪽 프레임 (마크다운 편집기)
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=1)
        
        # 마크다운 편집기 제목
        ttk.Label(left_frame, text="📝 Markdown Editor", style='Title.TLabel').pack(pady=5)
        
        # 텍스트 편집기
        text_frame = ttk.Frame(left_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 스크롤바
        text_scrollbar = ttk.Scrollbar(text_frame)
        text_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 텍스트 위젯
        self.text_editor = tk.Text(
            text_frame,
            wrap=tk.WORD,
            font=('Consolas', 11),
            bg='#ffffff',
            fg='#333333',
            insertbackground='#007ACC',
            selectbackground='#b3d9ff',
            yscrollcommand=text_scrollbar.set,
            undo=True,
            maxundo=-1
        )
        self.text_editor.pack(fill=tk.BOTH, expand=True)
        text_scrollbar.config(command=self.text_editor.yview)
        
        # 텍스트 변경 이벤트 바인딩
        self.text_editor.bind('<<Modified>>', self.on_text_change)
        self.text_editor.bind('<Control-a>', self.select_all)
        
        # 오른쪽 프레임 (HTML 미리보기)
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)
        
        # 미리보기 제목
        ttk.Label(right_frame, text="👁️ Preview", style='Title.TLabel').pack(pady=5)
        
        # HTML 프레임
        self.html_frame = HtmlFrame(right_frame, messages_enabled=False)
        self.html_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 상태바
        self.create_statusbar()
        
    def create_menubar(self):
        """메뉴바 생성"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # 파일 메뉴
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="파일", menu=file_menu)
        file_menu.add_command(label="열기... (Ctrl+O)", command=self.open_file)
        file_menu.add_command(label="저장 (Ctrl+S)", command=self.save_file)
        file_menu.add_command(label="다른 이름으로 저장...", command=self.save_as_file)
        file_menu.add_separator()
        file_menu.add_command(label="새 파일 (Ctrl+N)", command=self.new_file)
        file_menu.add_separator()
        file_menu.add_command(label="HTML로 내보내기", command=self.export_html)
        file_menu.add_separator()
        file_menu.add_command(label="종료", command=self.root.quit)
        
        # 편집 메뉴
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="편집", menu=edit_menu)
        edit_menu.add_command(label="실행 취소 (Ctrl+Z)", command=self.undo)
        edit_menu.add_command(label="다시 실행 (Ctrl+Y)", command=self.redo)
        edit_menu.add_separator()
        edit_menu.add_command(label="잘라내기 (Ctrl+X)", command=self.cut)
        edit_menu.add_command(label="복사 (Ctrl+C)", command=self.copy)
        edit_menu.add_command(label="붙여넣기 (Ctrl+V)", command=self.paste)
        edit_menu.add_separator()
        edit_menu.add_command(label="전체 선택 (Ctrl+A)", command=self.select_all_menu)
        
        # 보기 메뉴
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="보기", menu=view_menu)
        view_menu.add_command(label="새로고침 (F5)", command=self.refresh_preview)
        view_menu.add_separator()
        view_menu.add_command(label="글꼴 크기 증가", command=self.increase_font)
        view_menu.add_command(label="글꼴 크기 감소", command=self.decrease_font)
        
        # 도움말 메뉴
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="도움말", menu=help_menu)
        help_menu.add_command(label="마크다운 가이드", command=self.show_markdown_guide)
        help_menu.add_command(label="프로그램 정보", command=self.show_about)
        
    def create_toolbar(self):
        """툴바 생성"""
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        
        # 툴바 버튼들
        buttons = [
            ("📁", "열기", self.open_file),
            ("💾", "저장", self.save_file),
            ("📄", "새 파일", self.new_file),
            ("|", None, None),  # 구분선
            ("**B**", "굵게", lambda: self.insert_markdown("**", "**")),
            ("*I*", "기울임", lambda: self.insert_markdown("*", "*")),
            ("~~S~~", "취소선", lambda: self.insert_markdown("~~", "~~")),
            ("`C`", "코드", lambda: self.insert_markdown("`", "`")),
            ("|", None, None),  # 구분선
            ("H1", "제목 1", lambda: self.insert_markdown("# ", "")),
            ("H2", "제목 2", lambda: self.insert_markdown("## ", "")),
            ("H3", "제목 3", lambda: self.insert_markdown("### ", "")),
            ("|", None, None),  # 구분선
            ("•", "목록", lambda: self.insert_markdown("- ", "")),
            ("1.", "번호 목록", lambda: self.insert_markdown("1. ", "")),
            ("[ ]", "체크박스", lambda: self.insert_markdown("- [ ] ", "")),
            ("|", None, None),  # 구분선
            ("🔗", "링크", self.insert_link),
            ("🖼️", "이미지", self.insert_image),
            ("📊", "표", self.insert_table),
            ("|", None, None),  # 구분선
            ("🔄", "새로고침", self.refresh_preview),
        ]
        
        for text, tooltip, command in buttons:
            if text == "|":
                ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=3, fill=tk.Y)
            else:
                btn = ttk.Button(toolbar, text=text, command=command, style='Custom.TButton', width=6)
                btn.pack(side=tk.LEFT, padx=2)
                if tooltip:
                    self.create_tooltip(btn, tooltip)
    
    def create_statusbar(self):
        """상태바 생성"""
        self.statusbar = ttk.Frame(self.root)
        self.statusbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 상태 메시지
        self.status_msg = ttk.Label(self.statusbar, text="준비", relief=tk.SUNKEN)
        self.status_msg.pack(side=tk.LEFT, padx=5)
        
        # 커서 위치
        self.cursor_pos = ttk.Label(self.statusbar, text="줄: 1, 열: 1", relief=tk.SUNKEN)
        self.cursor_pos.pack(side=tk.RIGHT, padx=5)
        
        # 문서 정보
        self.doc_info = ttk.Label(self.statusbar, text="0 단어, 0 문자", relief=tk.SUNKEN)
        self.doc_info.pack(side=tk.RIGHT, padx=5)
        
        # 커서 위치 업데이트
        self.text_editor.bind('<KeyRelease>', self.update_cursor_position)
        self.text_editor.bind('<ButtonRelease>', self.update_cursor_position)
        
    def setup_shortcuts(self):
        """키보드 단축키 설정"""
        self.root.bind('<Control-o>', lambda e: self.open_file())
        self.root.bind('<Control-s>', lambda e: self.save_file())
        self.root.bind('<Control-n>', lambda e: self.new_file())
        self.root.bind('<F5>', lambda e: self.refresh_preview())
        self.root.bind('<Control-plus>', lambda e: self.increase_font())
        self.root.bind('<Control-minus>', lambda e: self.decrease_font())
        
    def on_text_change(self, event=None):
        """텍스트 변경 시 미리보기 업데이트"""
        if self.text_editor.edit_modified():
            self.refresh_preview()
            self.update_doc_info()
            self.text_editor.edit_modified(False)
    
    def refresh_preview(self):
        """미리보기 새로고침"""
        content = self.text_editor.get(1.0, tk.END)
        html = self.markdown_to_html(content)
        self.html_frame.load_html(html)
        self.status_msg.config(text="미리보기 업데이트됨")
        
    def markdown_to_html(self, markdown_text):
        """마크다운을 HTML로 변환"""
        # 마크다운 확장 기능 설정
        extensions = [
            'fenced_code',
            'tables',
            'toc',
            'nl2br',
            'sane_lists',
            'codehilite',
            'meta',
            'admonition'
        ]
        
        # HTML로 변환
        html_body = markdown.markdown(markdown_text, extensions=extensions)
        
        # 완전한 HTML 문서 생성
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans KR', sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 900px;
                    margin: 0 auto;
                    padding: 20px;
                    background: #fff;
                }}
                h1, h2, h3, h4, h5, h6 {{
                    color: #2c3e50;
                    margin-top: 24px;
                    margin-bottom: 16px;
                    font-weight: 600;
                    line-height: 1.25;
                }}
                h1 {{ font-size: 2em; border-bottom: 1px solid #eee; padding-bottom: 0.3em; }}
                h2 {{ font-size: 1.5em; border-bottom: 1px solid #eee; padding-bottom: 0.3em; }}
                h3 {{ font-size: 1.25em; }}
                code {{
                    background-color: #f6f8fa;
                    padding: 2px 4px;
                    border-radius: 3px;
                    font-family: 'Consolas', 'Monaco', monospace;
                    font-size: 0.9em;
                }}
                pre {{
                    background-color: #f6f8fa;
                    padding: 16px;
                    overflow: auto;
                    border-radius: 6px;
                    line-height: 1.45;
                }}
                pre code {{
                    background-color: transparent;
                    padding: 0;
                }}
                blockquote {{
                    border-left: 4px solid #dfe2e5;
                    padding: 0 1em;
                    color: #6a737d;
                    margin: 0;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 16px 0;
                }}
                table th, table td {{
                    border: 1px solid #dfe2e5;
                    padding: 6px 13px;
                }}
                table th {{
                    background-color: #f6f8fa;
                    font-weight: 600;
                }}
                table tr:nth-child(2n) {{
                    background-color: #f6f8fa;
                }}
                a {{
                    color: #0366d6;
                    text-decoration: none;
                }}
                a:hover {{
                    text-decoration: underline;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                }}
                hr {{
                    border: none;
                    border-top: 1px solid #e1e4e8;
                    margin: 24px 0;
                }}
                ul, ol {{
                    padding-left: 2em;
                    margin: 16px 0;
                }}
                li {{
                    margin: 4px 0;
                }}
                input[type="checkbox"] {{
                    margin-right: 4px;
                }}
                .admonition {{
                    padding: 12px;
                    margin: 16px 0;
                    border-left: 4px solid #4CAF50;
                    background-color: #f0f7ff;
                    border-radius: 4px;
                }}
                .admonition-title {{
                    font-weight: bold;
                    margin-bottom: 8px;
                }}
            </style>
        </head>
        <body>
            {html_body}
        </body>
        </html>
        """
        return html
        
    def open_file(self):
        """파일 열기"""
        file_path = filedialog.askopenfilename(
            title="마크다운 파일 선택",
            filetypes=[("Markdown files", "*.md *.markdown"), ("All files", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    self.text_editor.delete(1.0, tk.END)
                    self.text_editor.insert(1.0, content)
                    self.current_file = file_path
                    self.root.title(f"Markdown Viewer - {os.path.basename(file_path)}")
                    self.status_msg.config(text=f"파일 열기: {os.path.basename(file_path)}")
                    self.refresh_preview()
            except Exception as e:
                messagebox.showerror("오류", f"파일을 열 수 없습니다: {str(e)}")
                
    def save_file(self):
        """파일 저장"""
        if self.current_file:
            try:
                content = self.text_editor.get(1.0, tk.END)
                with open(self.current_file, 'w', encoding='utf-8') as file:
                    file.write(content.rstrip())
                self.status_msg.config(text=f"저장됨: {os.path.basename(self.current_file)}")
            except Exception as e:
                messagebox.showerror("오류", f"파일을 저장할 수 없습니다: {str(e)}")
        else:
            self.save_as_file()
            
    def save_as_file(self):
        """다른 이름으로 저장"""
        file_path = filedialog.asksaveasfilename(
            title="다른 이름으로 저장",
            defaultextension=".md",
            filetypes=[("Markdown files", "*.md"), ("All files", "*.*")]
        )
        if file_path:
            try:
                content = self.text_editor.get(1.0, tk.END)
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(content.rstrip())
                self.current_file = file_path
                self.root.title(f"Markdown Viewer - {os.path.basename(file_path)}")
                self.status_msg.config(text=f"저장됨: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("오류", f"파일을 저장할 수 없습니다: {str(e)}")
                
    def new_file(self):
        """새 파일"""
        if messagebox.askyesno("새 파일", "현재 문서를 저장하시겠습니까?"):
            self.save_file()
        self.text_editor.delete(1.0, tk.END)
        self.current_file = None
        self.root.title("Markdown Viewer - 제목 없음")
        self.refresh_preview()
        self.status_msg.config(text="새 파일 생성됨")
        
    def export_html(self):
        """HTML로 내보내기"""
        file_path = filedialog.asksaveasfilename(
            title="HTML로 내보내기",
            defaultextension=".html",
            filetypes=[("HTML files", "*.html"), ("All files", "*.*")]
        )
        if file_path:
            try:
                content = self.text_editor.get(1.0, tk.END)
                html = self.markdown_to_html(content)
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(html)
                self.status_msg.config(text=f"HTML 내보내기 완료: {os.path.basename(file_path)}")
                if messagebox.askyesno("HTML 내보내기", "브라우저에서 열어보시겠습니까?"):
                    webbrowser.open(file_path)
            except Exception as e:
                messagebox.showerror("오류", f"HTML 내보내기 실패: {str(e)}")
                
    def insert_markdown(self, prefix, suffix):
        """마크다운 태그 삽입"""
        try:
            sel_start = self.text_editor.index(tk.SEL_FIRST)
            sel_end = self.text_editor.index(tk.SEL_LAST)
            selected_text = self.text_editor.get(sel_start, sel_end)
            self.text_editor.delete(sel_start, sel_end)
            self.text_editor.insert(sel_start, f"{prefix}{selected_text}{suffix}")
        except:
            pos = self.text_editor.index(tk.INSERT)
            self.text_editor.insert(pos, f"{prefix}텍스트{suffix}")
            
    def insert_link(self):
        """링크 삽입"""
        dialog = tk.Toplevel(self.root)
        dialog.title("링크 삽입")
        dialog.geometry("400x150")
        
        ttk.Label(dialog, text="링크 텍스트:").grid(row=0, column=0, padx=10, pady=10, sticky='w')
        text_entry = ttk.Entry(dialog, width=40)
        text_entry.grid(row=0, column=1, padx=10, pady=10)
        
        ttk.Label(dialog, text="URL:").grid(row=1, column=0, padx=10, pady=10, sticky='w')
        url_entry = ttk.Entry(dialog, width=40)
        url_entry.grid(row=1, column=1, padx=10, pady=10)
        
        def insert():
            text = text_entry.get()
            url = url_entry.get()
            if text and url:
                pos = self.text_editor.index(tk.INSERT)
                self.text_editor.insert(pos, f"[{text}]({url})")
                dialog.destroy()
                
        ttk.Button(dialog, text="삽입", command=insert).grid(row=2, column=1, padx=10, pady=10)
        
    def insert_image(self):
        """이미지 삽입"""
        dialog = tk.Toplevel(self.root)
        dialog.title("이미지 삽입")
        dialog.geometry("400x150")
        
        ttk.Label(dialog, text="대체 텍스트:").grid(row=0, column=0, padx=10, pady=10, sticky='w')
        alt_entry = ttk.Entry(dialog, width=40)
        alt_entry.grid(row=0, column=1, padx=10, pady=10)
        
        ttk.Label(dialog, text="이미지 URL:").grid(row=1, column=0, padx=10, pady=10, sticky='w')
        url_entry = ttk.Entry(dialog, width=40)
        url_entry.grid(row=1, column=1, padx=10, pady=10)
        
        def insert():
            alt = alt_entry.get()
            url = url_entry.get()
            if alt and url:
                pos = self.text_editor.index(tk.INSERT)
                self.text_editor.insert(pos, f"![{alt}]({url})")
                dialog.destroy()
                
        ttk.Button(dialog, text="삽입", command=insert).grid(row=2, column=1, padx=10, pady=10)
        
    def insert_table(self):
        """표 삽입"""
        dialog = tk.Toplevel(self.root)
        dialog.title("표 삽입")
        dialog.geometry("300x150")
        
        ttk.Label(dialog, text="행 수:").grid(row=0, column=0, padx=10, pady=10, sticky='w')
        rows_entry = ttk.Entry(dialog, width=10)
        rows_entry.grid(row=0, column=1, padx=10, pady=10)
        rows_entry.insert(0, "3")
        
        ttk.Label(dialog, text="열 수:").grid(row=1, column=0, padx=10, pady=10, sticky='w')
        cols_entry = ttk.Entry(dialog, width=10)
        cols_entry.grid(row=1, column=1, padx=10, pady=10)
        cols_entry.insert(0, "3")
        
        def insert():
            try:
                rows = int(rows_entry.get())
                cols = int(cols_entry.get())
                
                # 표 생성
                table = ""
                # 헤더
                header = "| " + " | ".join([f"헤더{i+1}" for i in range(cols)]) + " |\n"
                separator = "| " + " | ".join(["---" for _ in range(cols)]) + " |\n"
                table += header + separator
                
                # 데이터 행
                for i in range(rows):
                    row = "| " + " | ".join([f"데이터{i+1}-{j+1}" for j in range(cols)]) + " |\n"
                    table += row
                    
                pos = self.text_editor.index(tk.INSERT)
                self.text_editor.insert(pos, table)
                dialog.destroy()
            except:
                messagebox.showerror("오류", "올바른 숫자를 입력하세요")
                
        ttk.Button(dialog, text="삽입", command=insert).grid(row=2, column=1, padx=10, pady=10)
        
    def undo(self):
        """실행 취소"""
        try:
            self.text_editor.edit_undo()
        except:
            pass
            
    def redo(self):
        """다시 실행"""
        try:
            self.text_editor.edit_redo()
        except:
            pass
            
    def cut(self):
        """잘라내기"""
        self.text_editor.event_generate("<<Cut>>")
        
    def copy(self):
        """복사"""
        self.text_editor.event_generate("<<Copy>>")
        
    def paste(self):
        """붙여넣기"""
        self.text_editor.event_generate("<<Paste>>")
        
    def select_all(self, event=None):
        """전체 선택"""
        self.text_editor.tag_add(tk.SEL, "1.0", tk.END)
        self.text_editor.mark_set(tk.INSERT, "1.0")
        self.text_editor.see(tk.INSERT)
        return 'break'
        
    def select_all_menu(self):
        """메뉴에서 전체 선택"""
        self.select_all()
        
    def increase_font(self):
        """글꼴 크기 증가"""
        current_font = font.Font(font=self.text_editor['font'])
        size = current_font['size']
        self.text_editor.configure(font=('Consolas', size + 1))
        
    def decrease_font(self):
        """글꼴 크기 감소"""
        current_font = font.Font(font=self.text_editor['font'])
        size = current_font['size']
        if size > 8:
            self.text_editor.configure(font=('Consolas', size - 1))
            
    def update_cursor_position(self, event=None):
        """커서 위치 업데이트"""
        position = self.text_editor.index(tk.INSERT)
        line, col = position.split('.')
        self.cursor_pos.config(text=f"줄: {line}, 열: {int(col)+1}")
        
    def update_doc_info(self):
        """문서 정보 업데이트"""
        content = self.text_editor.get(1.0, tk.END)
        words = len(content.split())
        chars = len(content) - 1  # END 문자 제외
        self.doc_info.config(text=f"{words} 단어, {chars} 문자")
        
    def show_markdown_guide(self):
        """마크다운 가이드 표시"""
        guide_window = tk.Toplevel(self.root)
        guide_window.title("마크다운 가이드")
        guide_window.geometry("600x500")
        
        text = tk.Text(guide_window, wrap=tk.WORD, padx=10, pady=10)
        text.pack(fill=tk.BOTH, expand=True)
        
        guide_text = """
# 마크다운 기본 문법 가이드

## 제목 (Headers)
# H1 제목
## H2 제목  
### H3 제목
#### H4 제목

## 강조 (Emphasis)
*기울임체* 또는 _기울임체_
**굵은체** 또는 __굵은체__
***굵은 기울임체*** 또는 ___굵은 기울임체___
~~취소선~~

## 목록 (Lists)
### 순서 없는 목록
- 항목 1
- 항목 2
  - 하위 항목 2.1
  - 하위 항목 2.2
- 항목 3

### 순서 있는 목록
1. 첫 번째 항목
2. 두 번째 항목
3. 세 번째 항목

### 체크리스트
- [ ] 미완료 작업
- [x] 완료된 작업

## 링크와 이미지
[링크 텍스트](https://www.example.com)
![대체 텍스트](image-url.jpg)

## 코드
인라인 코드: `code`

코드 블록:
```python
def hello():
    print("Hello, World!")
```

## 인용문
> 이것은 인용문입니다.
> 여러 줄로 작성할 수 있습니다.

## 수평선
---
또는
***

## 표 (Tables)
| 헤더 1 | 헤더 2 | 헤더 3 |
|--------|--------|--------|
| 셀 1   | 셀 2   | 셀 3   |
| 셀 4   | 셀 5   | 셀 6   |

## 이스케이프 문자
\\* 별표를 그대로 표시
\\# 샵을 그대로 표시
        """
        
        text.insert(1.0, guide_text)
        text.config(state=tk.DISABLED)
        
    def show_about(self):
        """프로그램 정보"""
        messagebox.showinfo(
            "프로그램 정보",
            "Markdown Viewer v1.0\n\n"
            "깔끔한 마크다운 뷰어 프로그램\n"
            "실시간 미리보기 기능 제공\n\n"
            "© 2024"
        )
        
    def create_tooltip(self, widget, text):
        """툴팁 생성"""
        def on_enter(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            label = tk.Label(tooltip, text=text, background="#ffffe0", 
                           relief=tk.SOLID, borderwidth=1, font=('Arial', 9))
            label.pack()
            widget.tooltip = tooltip
            
        def on_leave(event):
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()
                del widget.tooltip
                
        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)


def main():
    """메인 함수"""
    root = tk.Tk()
    
    # 아이콘 설정 (옵션)
    try:
        root.iconbitmap('markdown.ico')
    except:
        pass
    
    app = MarkdownViewer(root)
    
    # 시작 시 샘플 텍스트
    sample_text = """# Markdown Viewer에 오신 것을 환영합니다! 👋

이 프로그램은 **마크다운 파일**을 *실시간*으로 미리보기할 수 있는 편리한 뷰어입니다.

## 주요 기능 ✨

- 📝 **실시간 미리보기**: 입력하는 즉시 결과 확인
- 🎨 **깔끔한 UI**: 편집기와 미리보기 분할 화면
- 🔧 **편집 도구**: 마크다운 문법 삽입 버튼
- 💾 **파일 저장/열기**: MD 파일 관리
- 📤 **HTML 내보내기**: 웹 페이지로 변환

## 사용 방법

1. 왼쪽 편집기에 마크다운 작성
2. 오른쪽에서 실시간 미리보기 확인  
3. 툴바 버튼으로 빠른 서식 적용

### 예제 코드
```python
print("Hello, Markdown!")
```

### 체크리스트 예제
- [x] 마크다운 뷰어 실행
- [ ] 문서 작성하기
- [ ] HTML로 내보내기

### 표 예제
| 기능 | 단축키 |
|------|--------|
| 열기 | Ctrl+O |
| 저장 | Ctrl+S |
| 새로고침 | F5 |

---

> 💡 **팁**: 상단 메뉴의 '도움말 > 마크다운 가이드'에서 더 많은 문법을 확인하세요!
"""
    
    app.text_editor.insert(1.0, sample_text)
    app.refresh_preview()
    
    root.mainloop()


if __name__ == "__main__":
    main()