import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import markdown
from tkinterweb import HtmlFrame
import os
import sys
import datetime
import webbrowser

class SimpleMarkdownViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Simple MD Viewer - 마크다운 뷰어")
        self.root.geometry("1000x700")
        
        # 현재 파일 경로
        self.current_file = None
        self.last_content = None  # 마지막 로드한 내용 저장
        
        # UI 스타일 설정
        self.setup_styles()
        
        # UI 구성
        self.setup_ui()
        
        # 단축키 설정
        self.setup_shortcuts()
        
        # 명령줄 인자로 파일이 전달된 경우 열기
        if len(sys.argv) > 1:
            self.load_file(sys.argv[1])
    
    def setup_styles(self):
        """테마와 스타일 설정"""
        self.root.configure(bg='#f5f5f5')
        style = ttk.Style()
        style.theme_use('clam')
        
        # 버튼 스타일
        style.configure('Toolbar.TButton', 
                       font=('Segoe UI', 10),
                       padding=(10, 5))
        
    def setup_ui(self):
        """UI 구성"""
        # 툴바 프레임
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # 툴바 버튼들
        ttk.Button(toolbar, text="📁 파일 열기 (Ctrl+O)", 
                  command=self.open_file,
                  style='Toolbar.TButton').pack(side=tk.LEFT, padx=2)
        
        ttk.Button(toolbar, text="🔄 새로고침 (F5)", 
                  command=self.refresh_view,
                  style='Toolbar.TButton').pack(side=tk.LEFT, padx=2)
        
        ttk.Button(toolbar, text="📤 HTML 내보내기", 
                  command=self.export_html,
                  style='Toolbar.TButton').pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)
        
        ttk.Button(toolbar, text="➕ 확대", 
                  command=self.zoom_in,
                  style='Toolbar.TButton').pack(side=tk.LEFT, padx=2)
        
        ttk.Button(toolbar, text="➖ 축소", 
                  command=self.zoom_out,
                  style='Toolbar.TButton').pack(side=tk.LEFT, padx=2)
        
        ttk.Button(toolbar, text="↺ 원래 크기", 
                  command=self.reset_zoom,
                  style='Toolbar.TButton').pack(side=tk.LEFT, padx=2)
        
        # 현재 파일 표시 레이블
        self.file_label = ttk.Label(toolbar, text="파일을 선택하세요", 
                                   font=('Segoe UI', 10))
        self.file_label.pack(side=tk.RIGHT, padx=10)
        
        # 메인 뷰어 프레임
        viewer_frame = ttk.Frame(self.root)
        viewer_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # HTML 뷰어
        self.html_viewer = HtmlFrame(viewer_frame, messages_enabled=False)
        self.html_viewer.pack(fill=tk.BOTH, expand=True)
        
        # 초기 화면 표시
        self.show_welcome_screen()
        
        # 상태바
        statusbar = ttk.Frame(self.root)
        statusbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_label = ttk.Label(statusbar, text="준비", relief=tk.SUNKEN)
        self.status_label.pack(side=tk.LEFT, padx=5, pady=2)
        
        # 파일 크기 표시
        self.file_size_label = ttk.Label(statusbar, text="", relief=tk.SUNKEN)
        self.file_size_label.pack(side=tk.LEFT, padx=5, pady=2)
        
        # 줌 레벨 표시
        self.zoom_label = ttk.Label(statusbar, text="크기: 100%", relief=tk.SUNKEN)
        self.zoom_label.pack(side=tk.RIGHT, padx=5, pady=2)
        
        # 현재 줌 레벨
        self.zoom_level = 100
        
    def setup_shortcuts(self):
        """키보드 단축키 설정"""
        self.root.bind('<Control-o>', lambda e: self.open_file())
        self.root.bind('<Control-O>', lambda e: self.open_file())
        self.root.bind('<F5>', lambda e: self.refresh_view())
        self.root.bind('<Control-r>', lambda e: self.refresh_view())
        self.root.bind('<Control-R>', lambda e: self.refresh_view())
        self.root.bind('<Control-plus>', lambda e: self.zoom_in())
        self.root.bind('<Control-equal>', lambda e: self.zoom_in())  # + 키
        self.root.bind('<Control-minus>', lambda e: self.zoom_out())
        self.root.bind('<Control-0>', lambda e: self.reset_zoom())
        self.root.bind('<Control-e>', lambda e: self.export_html())
        self.root.bind('<Control-E>', lambda e: self.export_html())
        
    def show_welcome_screen(self):
        """환영 화면 표시"""
        welcome_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {
                    font-family: -apple-system, 'Segoe UI', 'Noto Sans KR', sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    min-height: 100vh;
                    margin: 0;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                }
                .container {
                    text-align: center;
                    padding: 40px;
                    background: rgba(255, 255, 255, 0.1);
                    border-radius: 20px;
                    backdrop-filter: blur(10px);
                }
                h1 {
                    font-size: 48px;
                    margin-bottom: 20px;
                    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                }
                p {
                    font-size: 20px;
                    margin: 20px 0;
                    opacity: 0.9;
                }
                .shortcuts {
                    margin-top: 40px;
                    padding: 20px;
                    background: rgba(255, 255, 255, 0.2);
                    border-radius: 10px;
                }
                .shortcut-item {
                    margin: 10px 0;
                    font-size: 16px;
                }
                .key {
                    background: rgba(255, 255, 255, 0.3);
                    padding: 3px 8px;
                    border-radius: 4px;
                    font-weight: bold;
                }
                .highlight {
                    background: rgba(255, 200, 0, 0.3);
                    padding: 5px 10px;
                    border-radius: 5px;
                    margin-top: 20px;
                    display: inline-block;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📄 Simple MD Viewer</h1>
                <p>마크다운 파일을 깔끔하게 볼 수 있는 뷰어입니다</p>
                <p>파일을 열려면 <span class="key">📁 파일 열기</span> 버튼을 클릭하세요</p>
                
                <div class="shortcuts">
                    <h3>⌨️ 단축키</h3>
                    <div class="shortcut-item"><span class="key">Ctrl + O</span> 파일 열기</div>
                    <div class="shortcut-item"><span class="key">F5</span> 파일 새로고침 (변경사항 반영)</div>
                    <div class="shortcut-item"><span class="key">Ctrl + E</span> HTML로 내보내기</div>
                    <div class="shortcut-item"><span class="key">Ctrl + +</span> 확대</div>
                    <div class="shortcut-item"><span class="key">Ctrl + -</span> 축소</div>
                    <div class="shortcut-item"><span class="key">Ctrl + 0</span> 원래 크기</div>
                </div>
                
                <div class="highlight">
                    💡 F5를 누르면 파일의 변경사항이 자동으로 반영됩니다
                </div>
            </div>
        </body>
        </html>
        """
        self.html_viewer.load_html(welcome_html)
        
    def open_file(self):
        """파일 열기"""
        file_path = filedialog.askopenfilename(
            title="마크다운 파일 선택",
            filetypes=[
                ("Markdown files", "*.md *.markdown *.mdown *.mkd"),
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.load_file(file_path)
            
    def load_file(self, file_path):
        """파일 로드"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
            self.current_file = file_path
            self.last_content = content
            self.display_markdown(content)
            
            # 파일 정보 표시
            filename = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)
            
            # 파일 크기를 읽기 쉽게 변환
            if file_size < 1024:
                size_str = f"{file_size} B"
            elif file_size < 1024 * 1024:
                size_str = f"{file_size / 1024:.1f} KB"
            else:
                size_str = f"{file_size / (1024 * 1024):.1f} MB"
            
            self.file_label.config(text=f"📄 {filename}")
            self.file_size_label.config(text=f"크기: {size_str}")
            self.root.title(f"Simple MD Viewer - {filename}")
            
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            self.status_label.config(text=f"파일 열림: {current_time}")
            
        except Exception as e:
            messagebox.showerror("오류", f"파일을 열 수 없습니다:\n{str(e)}")
            
    def refresh_view(self):
        """현재 파일 다시 로드 (F5) - 외부 변경사항 반영"""
        if not self.current_file:
            self.status_label.config(text="열린 파일이 없습니다")
            return
            
        try:
            # 파일이 존재하는지 확인
            if not os.path.exists(self.current_file):
                messagebox.showerror("오류", f"파일을 찾을 수 없습니다:\n{self.current_file}")
                self.current_file = None
                self.file_label.config(text="파일을 선택하세요")
                self.file_size_label.config(text="")
                self.show_welcome_screen()
                return
            
            # 파일 다시 읽기
            with open(self.current_file, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # 내용이 변경되었는지 확인
            if content != self.last_content:
                self.last_content = content
                self.display_markdown(content)
                
                # 파일 크기 업데이트
                file_size = os.path.getsize(self.current_file)
                if file_size < 1024:
                    size_str = f"{file_size} B"
                elif file_size < 1024 * 1024:
                    size_str = f"{file_size / 1024:.1f} KB"
                else:
                    size_str = f"{file_size / (1024 * 1024):.1f} MB"
                
                self.file_size_label.config(text=f"크기: {size_str}")
                
                current_time = datetime.datetime.now().strftime("%H:%M:%S")
                self.status_label.config(text=f"새로고침 완료 (변경됨) - {current_time}")
            else:
                # 변경사항이 없어도 줌 레벨이 변경되었을 수 있으므로 다시 렌더링
                self.display_markdown(content)
                current_time = datetime.datetime.now().strftime("%H:%M:%S")
                self.status_label.config(text=f"새로고침 완료 - {current_time}")
                
        except Exception as e:
            messagebox.showerror("오류", f"새로고침 실패:\n{str(e)}")
            
    def display_markdown(self, markdown_text):
        """마크다운을 HTML로 변환하여 표시"""
        html = self.convert_to_html(markdown_text)
        self.html_viewer.load_html(html)
        
    def convert_to_html(self, markdown_text):
        """마크다운을 스타일이 적용된 HTML로 변환"""
        # 마크다운 확장 기능
        extensions = [
            'extra',           # 테이블, 각주, 약어 등
            'codehilite',      # 코드 하이라이팅
            'toc',             # 목차
            'nl2br',           # 줄바꿈 처리
            'sane_lists',      # 개선된 목록
            'meta',            # 메타데이터
            'admonition',      # 경고/정보 박스
            'tables',          # 테이블
            'fenced_code',     # 펜스 코드 블록
        ]
        
        # 마크다운을 HTML로 변환
        md = markdown.Markdown(extensions=extensions)
        html_body = md.convert(markdown_text)
        
        # 줌 레벨에 따른 폰트 크기 계산
        base_font_size = 16 * (self.zoom_level / 100)
        
        # 완성된 HTML 문서
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                * {{
                    box-sizing: border-box;
                }}
                
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans KR', 
                                 'Helvetica Neue', Arial, sans-serif;
                    font-size: {base_font_size}px;
                    line-height: 1.7;
                    color: #2c3e50;
                    background: #ffffff;
                    max-width: 900px;
                    margin: 0 auto;
                    padding: 40px 60px;
                }}
                
                /* 제목 스타일 */
                h1, h2, h3, h4, h5, h6 {{
                    color: #1a202c;
                    font-weight: 600;
                    margin-top: 1.5em;
                    margin-bottom: 0.5em;
                    line-height: 1.3;
                }}
                
                h1 {{
                    font-size: 2.2em;
                    border-bottom: 3px solid #e2e8f0;
                    padding-bottom: 0.3em;
                }}
                
                h2 {{
                    font-size: 1.8em;
                    border-bottom: 2px solid #e2e8f0;
                    padding-bottom: 0.3em;
                }}
                
                h3 {{ font-size: 1.4em; }}
                h4 {{ font-size: 1.2em; }}
                h5 {{ font-size: 1.1em; }}
                h6 {{ font-size: 1em; color: #718096; }}
                
                /* 단락과 텍스트 */
                p {{
                    margin: 1em 0;
                }}
                
                strong {{
                    font-weight: 600;
                    color: #1a202c;
                }}
                
                em {{
                    font-style: italic;
                }}
                
                del {{
                    text-decoration: line-through;
                    color: #a0aec0;
                }}
                
                /* 링크 */
                a {{
                    color: #4299e1;
                    text-decoration: none;
                    border-bottom: 1px solid transparent;
                    transition: all 0.2s;
                }}
                
                a:hover {{
                    color: #2b6cb0;
                    border-bottom-color: #2b6cb0;
                }}
                
                /* 코드 */
                code {{
                    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                    font-size: 0.9em;
                    background: #f7fafc;
                    color: #e53e3e;
                    padding: 2px 6px;
                    border-radius: 4px;
                    border: 1px solid #e2e8f0;
                }}
                
                pre {{
                    background: #2d3748;
                    color: #e2e8f0;
                    padding: 20px;
                    border-radius: 8px;
                    overflow-x: auto;
                    line-height: 1.5;
                    margin: 1.5em 0;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }}
                
                pre code {{
                    background: transparent;
                    color: inherit;
                    padding: 0;
                    border: none;
                    font-size: 0.95em;
                }}
                
                /* 인용문 */
                blockquote {{
                    border-left: 4px solid #4299e1;
                    background: #f7fafc;
                    padding: 1em 1.5em;
                    margin: 1.5em 0;
                    font-style: italic;
                    color: #4a5568;
                    border-radius: 0 8px 8px 0;
                }}
                
                blockquote p:first-child {{
                    margin-top: 0;
                }}
                
                blockquote p:last-child {{
                    margin-bottom: 0;
                }}
                
                /* 목록 */
                ul, ol {{
                    margin: 1em 0;
                    padding-left: 2em;
                }}
                
                li {{
                    margin: 0.5em 0;
                }}
                
                ul li {{
                    list-style-type: disc;
                }}
                
                ul ul li {{
                    list-style-type: circle;
                }}
                
                ul ul ul li {{
                    list-style-type: square;
                }}
                
                /* 체크박스 */
                input[type="checkbox"] {{
                    margin-right: 0.5em;
                }}
                
                /* 테이블 */
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 1.5em 0;
                    border: 1px solid #cbd5e0;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
                }}
                
                th, td {{
                    border: 1px solid #cbd5e0;
                    padding: 12px 15px;
                    text-align: left;
                }}
                
                th {{
                    background: #edf2f7;
                    font-weight: 600;
                    color: #2d3748;
                }}
                
                tr:nth-child(even) {{
                    background: #f7fafc;
                }}
                
                tr:hover {{
                    background: #e6f7ff;
                    transition: background 0.2s;
                }}
                
                /* 구분선 */
                hr {{
                    border: none;
                    border-top: 2px solid #e2e8f0;
                    margin: 2em 0;
                }}
                
                /* 이미지 */
                img {{
                    max-width: 100%;
                    height: auto;
                    border-radius: 8px;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
                    margin: 1.5em 0;
                    display: block;
                }}
                
                /* Admonition (알림 박스) */
                .admonition {{
                    padding: 15px 20px;
                    margin: 1.5em 0;
                    border-radius: 8px;
                    border-left: 4px solid #4299e1;
                    background: #ebf8ff;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
                }}
                
                .admonition-title {{
                    font-weight: 600;
                    margin-bottom: 0.5em;
                    color: #2b6cb0;
                }}
                
                /* 애니메이션 */
                @keyframes fadeIn {{
                    from {{ opacity: 0; transform: translateY(10px); }}
                    to {{ opacity: 1; transform: translateY(0); }}
                }}
                
                body {{
                    animation: fadeIn 0.3s ease-out;
                }}
            </style>
        </head>
        <body>
            {html_body}
        </body>
        </html>
        """
        
        return html
        
    def export_html(self):
        """HTML로 내보내기 (완전한 독립 HTML 파일)"""
        if not self.current_file:
            messagebox.showwarning("경고", "먼저 마크다운 파일을 열어주세요.")
            return
            
        # 기본 파일명 제안
        default_name = os.path.splitext(os.path.basename(self.current_file))[0] + ".html"
            
        file_path = filedialog.asksaveasfilename(
            title="HTML로 내보내기",
            defaultextension=".html",
            initialfile=default_name,
            filetypes=[("HTML files", "*.html"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # 현재 표시 중인 내용을 그대로 저장 (줌 레벨 포함)
                html = self.convert_to_html(self.last_content)
                
                # 독립 실행 가능한 HTML 파일로 저장
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(html)
                    
                self.status_label.config(text=f"HTML 내보내기 완료: {os.path.basename(file_path)}")
                
                # 내보낸 파일을 브라우저에서 열지 묻기
                response = messagebox.askyesnocancel(
                    "내보내기 완료", 
                    "HTML 파일이 생성되었습니다.\n\n브라우저에서 열어보시겠습니까?"
                )
                
                if response is True:  # Yes 선택
                    webbrowser.open(file_path)
                    
            except Exception as e:
                messagebox.showerror("오류", f"HTML 내보내기 실패:\n{str(e)}")
                
    def zoom_in(self):
        """확대"""
        if self.zoom_level < 200:
            self.zoom_level += 10
            self.update_zoom()
            
    def zoom_out(self):
        """축소"""
        if self.zoom_level > 50:
            self.zoom_level -= 10
            self.update_zoom()
            
    def reset_zoom(self):
        """원래 크기로"""
        self.zoom_level = 100
        self.update_zoom()
        
    def update_zoom(self):
        """줌 레벨 업데이트"""
        self.zoom_label.config(text=f"크기: {self.zoom_level}%")
        if self.current_file and self.last_content:
            self.display_markdown(self.last_content)


def main():
    """메인 함수"""
    root = tk.Tk()
    
    # 아이콘 설정 시도
    try:
        root.iconbitmap('viewer.ico')
    except:
        pass
    
    # 윈도우를 화면 중앙에 배치
    root.update_idletasks()
    width = 1000
    height = 700
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    app = SimpleMarkdownViewer(root)
    
    # ESC 키로 종료
    root.bind('<Escape>', lambda e: root.quit())
    
    root.mainloop()


if __name__ == "__main__":
    main()