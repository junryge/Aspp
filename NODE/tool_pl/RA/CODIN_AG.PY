"""
🤖 Coding Agent Desktop App
- Ralph 에이전트들이 코드 생성 (직원)
- 관리자가 최종 확인/승인
- 로그인 없이 바로 시작
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import json
import os
from datetime import datetime
import threading
import time

# ========================================
# 색상 테마
# ========================================
COLORS = {
    'bg_dark': '#1e1e2e',
    'bg_darker': '#181825',
    'bg_surface': '#313244',
    'bg_overlay': '#45475a',
    'primary': '#6366f1',
    'secondary': '#8b5cf6',
    'success': '#10b981',
    'warning': '#f59e0b',
    'danger': '#ef4444',
    'text_light': '#cdd6f4',
    'text_muted': '#6c7086',
}

# ========================================
# 메인 애플리케이션
# ========================================
class CodingAgentApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🤖 Coding Agent - Ralph Team Manager")
        self.root.geometry("1400x900")
        self.root.configure(bg=COLORS['bg_darker'])
        
        # Ralph 에이전트들 (직원)
        self.ralphs = [
            {
                'id': 1, 
                'name': 'Backend Ralph', 
                'role': '백엔드', 
                'icon': '👨‍💻', 
                'status': 'idle',  # idle, working, done
                'specialty': ['python', 'java', 'csharp'],
            },
            {
                'id': 2, 
                'name': 'Frontend Ralph', 
                'role': '프론트엔드', 
                'icon': '🎨', 
                'status': 'idle',
                'specialty': ['javascript', 'html', 'css'],
            },
            {
                'id': 3, 
                'name': 'Test Ralph', 
                'role': 'QA', 
                'icon': '🧪', 
                'status': 'idle',
                'specialty': ['pytest', 'unittest'],
            },
        ]
        
        # 작업 목록 (승인 대기)
        self.pending_tasks = []
        
        # 승인된 작업
        self.approved_tasks = []
        
        # 거절된 작업
        self.rejected_tasks = []
        
        # 현재 선택된 작업
        self.selected_task = None
        
        # UI 생성
        self.create_ui()
        
    def run(self):
        self.root.mainloop()
    
    # ========================================
    # UI 생성
    # ========================================
    def create_ui(self):
        # 상단 헤더
        self.create_header()
        
        # 메인 영역 (3분할)
        main_frame = tk.Frame(self.root, bg=COLORS['bg_darker'])
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # 왼쪽: Ralph 팀 (직원들)
        self.create_ralph_panel(main_frame)
        
        # 중앙: 작업 입력 & 대기열
        self.create_task_panel(main_frame)
        
        # 오른쪽: 승인 패널 (관리자)
        self.create_approval_panel(main_frame)
    
    def create_header(self):
        """상단 헤더"""
        header = tk.Frame(self.root, bg=COLORS['bg_dark'], height=60)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        # 로고
        logo = tk.Label(
            header,
            text="🤖 Coding Agent",
            font=('Segoe UI', 18, 'bold'),
            bg=COLORS['bg_dark'],
            fg=COLORS['primary']
        )
        logo.pack(side='left', padx=20, pady=15)
        
        # 상태 표시
        status_frame = tk.Frame(header, bg=COLORS['bg_dark'])
        status_frame.pack(side='right', padx=20)
        
        self.pending_label = tk.Label(
            status_frame,
            text="⏳ 대기: 0",
            font=('Segoe UI', 11),
            bg=COLORS['bg_dark'],
            fg=COLORS['warning']
        )
        self.pending_label.pack(side='left', padx=10)
        
        self.approved_label = tk.Label(
            status_frame,
            text="✅ 승인: 0",
            font=('Segoe UI', 11),
            bg=COLORS['bg_dark'],
            fg=COLORS['success']
        )
        self.approved_label.pack(side='left', padx=10)
    
    def create_ralph_panel(self, parent):
        """왼쪽: Ralph 팀 패널"""
        panel = tk.Frame(parent, bg=COLORS['bg_surface'], width=300)
        panel.pack(side='left', fill='y', padx=(0, 10))
        panel.pack_propagate(False)
        
        # 제목
        title = tk.Label(
            panel,
            text="👥 Ralph 팀 (직원)",
            font=('Segoe UI', 14, 'bold'),
            bg=COLORS['bg_surface'],
            fg=COLORS['text_light']
        )
        title.pack(pady=15)
        
        # Ralph 카드들
        self.ralph_frames = {}
        for ralph in self.ralphs:
            self.create_ralph_card(panel, ralph)
    
    def create_ralph_card(self, parent, ralph):
        """Ralph 카드 생성"""
        card = tk.Frame(parent, bg=COLORS['bg_overlay'])
        card.pack(fill='x', padx=10, pady=5)
        
        # 상단: 아이콘 + 이름 + 상태
        top = tk.Frame(card, bg=COLORS['bg_overlay'])
        top.pack(fill='x', padx=10, pady=10)
        
        icon_label = tk.Label(
            top,
            text=ralph['icon'],
            font=('Segoe UI Emoji', 28),
            bg=COLORS['bg_overlay']
        )
        icon_label.pack(side='left')
        
        info_frame = tk.Frame(top, bg=COLORS['bg_overlay'])
        info_frame.pack(side='left', padx=10)
        
        name_label = tk.Label(
            info_frame,
            text=ralph['name'],
            font=('Segoe UI', 12, 'bold'),
            bg=COLORS['bg_overlay'],
            fg=COLORS['text_light']
        )
        name_label.pack(anchor='w')
        
        role_label = tk.Label(
            info_frame,
            text=ralph['role'],
            font=('Segoe UI', 10),
            bg=COLORS['bg_overlay'],
            fg=COLORS['text_muted']
        )
        role_label.pack(anchor='w')
        
        # 상태 표시
        status_colors = {
            'idle': COLORS['text_muted'],
            'working': COLORS['warning'],
            'done': COLORS['success']
        }
        status_texts = {
            'idle': '● 대기 중',
            'working': '● 작업 중...',
            'done': '● 완료'
        }
        
        status_label = tk.Label(
            top,
            text=status_texts[ralph['status']],
            font=('Segoe UI', 10),
            bg=COLORS['bg_overlay'],
            fg=status_colors[ralph['status']]
        )
        status_label.pack(side='right')
        
        # 저장 (나중에 업데이트용)
        self.ralph_frames[ralph['id']] = {
            'card': card,
            'status_label': status_label
        }
    
    def create_task_panel(self, parent):
        """중앙: 작업 입력 & 대기열"""
        panel = tk.Frame(parent, bg=COLORS['bg_darker'])
        panel.pack(side='left', fill='both', expand=True, padx=10)
        
        # === 작업 입력 영역 ===
        input_frame = tk.Frame(panel, bg=COLORS['bg_surface'])
        input_frame.pack(fill='x', pady=(0, 10))
        
        input_title = tk.Label(
            input_frame,
            text="📝 새 작업 요청",
            font=('Segoe UI', 14, 'bold'),
            bg=COLORS['bg_surface'],
            fg=COLORS['text_light']
        )
        input_title.pack(anchor='w', padx=15, pady=(15, 10))
        
        # 작업 설명 입력
        self.task_entry = tk.Entry(
            input_frame,
            font=('Segoe UI', 12),
            bg=COLORS['bg_dark'],
            fg=COLORS['text_light'],
            insertbackground=COLORS['text_light'],
            relief='flat'
        )
        self.task_entry.pack(fill='x', padx=15, pady=5, ipady=10)
        self.task_entry.insert(0, "예: Calculator 클래스 만들어줘")
        self.task_entry.bind('<FocusIn>', lambda e: self.task_entry.delete(0, 'end') if self.task_entry.get().startswith('예:') else None)
        
        # 옵션 행
        options_frame = tk.Frame(input_frame, bg=COLORS['bg_surface'])
        options_frame.pack(fill='x', padx=15, pady=10)
        
        # 담당 Ralph 선택
        tk.Label(
            options_frame,
            text="담당:",
            font=('Segoe UI', 10),
            bg=COLORS['bg_surface'],
            fg=COLORS['text_muted']
        ).pack(side='left')
        
        self.ralph_var = tk.StringVar(value='auto')
        ralph_menu = ttk.Combobox(
            options_frame,
            textvariable=self.ralph_var,
            values=['auto (자동 배정)', 'Backend Ralph', 'Frontend Ralph', 'Test Ralph'],
            state='readonly',
            width=18
        )
        ralph_menu.pack(side='left', padx=(5, 15))
        
        # 언어 선택
        tk.Label(
            options_frame,
            text="언어:",
            font=('Segoe UI', 10),
            bg=COLORS['bg_surface'],
            fg=COLORS['text_muted']
        ).pack(side='left')
        
        self.lang_var = tk.StringVar(value='python')
        lang_menu = ttk.Combobox(
            options_frame,
            textvariable=self.lang_var,
            values=['python', 'java', 'javascript', 'html', 'csharp'],
            state='readonly',
            width=12
        )
        lang_menu.pack(side='left', padx=5)
        
        # 작업 시작 버튼
        start_btn = tk.Button(
            options_frame,
            text="🚀 Ralph에게 시키기",
            font=('Segoe UI', 11, 'bold'),
            bg=COLORS['primary'],
            fg='white',
            relief='flat',
            cursor='hand2',
            command=self.assign_task
        )
        start_btn.pack(side='right')
        
        # === 작업 대기열 ===
        queue_frame = tk.Frame(panel, bg=COLORS['bg_surface'])
        queue_frame.pack(fill='both', expand=True)
        
        queue_title = tk.Label(
            queue_frame,
            text="⏳ 승인 대기 작업",
            font=('Segoe UI', 14, 'bold'),
            bg=COLORS['bg_surface'],
            fg=COLORS['text_light']
        )
        queue_title.pack(anchor='w', padx=15, pady=15)
        
        # 스크롤 가능한 작업 목록
        self.queue_container = tk.Frame(queue_frame, bg=COLORS['bg_surface'])
        self.queue_container.pack(fill='both', expand=True, padx=15, pady=(0, 15))
        
        # 빈 상태 메시지
        self.empty_label = tk.Label(
            self.queue_container,
            text="아직 대기 중인 작업이 없습니다.\n위에서 작업을 요청하세요!",
            font=('Segoe UI', 11),
            bg=COLORS['bg_surface'],
            fg=COLORS['text_muted']
        )
        self.empty_label.pack(pady=50)
    
    def create_approval_panel(self, parent):
        """오른쪽: 승인 패널 (관리자)"""
        panel = tk.Frame(parent, bg=COLORS['bg_surface'], width=450)
        panel.pack(side='right', fill='y', padx=(10, 0))
        panel.pack_propagate(False)
        
        # 제목
        title = tk.Label(
            panel,
            text="🛡️ 관리자 검토",
            font=('Segoe UI', 14, 'bold'),
            bg=COLORS['bg_surface'],
            fg=COLORS['text_light']
        )
        title.pack(pady=15)
        
        # 선택된 작업 정보
        self.review_frame = tk.Frame(panel, bg=COLORS['bg_overlay'])
        self.review_frame.pack(fill='x', padx=10, pady=5)
        
        self.review_title = tk.Label(
            self.review_frame,
            text="작업을 선택하세요",
            font=('Segoe UI', 12, 'bold'),
            bg=COLORS['bg_overlay'],
            fg=COLORS['text_light']
        )
        self.review_title.pack(anchor='w', padx=15, pady=(15, 5))
        
        self.review_info = tk.Label(
            self.review_frame,
            text="왼쪽 대기열에서 작업을 클릭하면\n코드를 검토할 수 있습니다.",
            font=('Segoe UI', 10),
            bg=COLORS['bg_overlay'],
            fg=COLORS['text_muted'],
            justify='left'
        )
        self.review_info.pack(anchor='w', padx=15, pady=(0, 15))
        
        # 코드 미리보기
        code_label = tk.Label(
            panel,
            text="📄 생성된 코드",
            font=('Segoe UI', 11, 'bold'),
            bg=COLORS['bg_surface'],
            fg=COLORS['text_light']
        )
        code_label.pack(anchor='w', padx=15, pady=(15, 5))
        
        self.code_preview = scrolledtext.ScrolledText(
            panel,
            font=('Consolas', 10),
            bg=COLORS['bg_dark'],
            fg=COLORS['text_light'],
            relief='flat',
            height=20
        )
        self.code_preview.pack(fill='both', expand=True, padx=10, pady=5)
        self.code_preview.insert('1.0', "# 작업을 선택하면 코드가 여기 표시됩니다")
        self.code_preview.config(state='disabled')
        
        # 승인/거절 버튼
        btn_frame = tk.Frame(panel, bg=COLORS['bg_surface'])
        btn_frame.pack(fill='x', padx=10, pady=15)
        
        self.reject_btn = tk.Button(
            btn_frame,
            text="❌ 거절 (재작업)",
            font=('Segoe UI', 11, 'bold'),
            bg=COLORS['danger'],
            fg='white',
            relief='flat',
            cursor='hand2',
            state='disabled',
            command=self.reject_task
        )
        self.reject_btn.pack(side='left', expand=True, fill='x', padx=(0, 5))
        
        self.approve_btn = tk.Button(
            btn_frame,
            text="✅ 승인",
            font=('Segoe UI', 11, 'bold'),
            bg=COLORS['success'],
            fg='white',
            relief='flat',
            cursor='hand2',
            state='disabled',
            command=self.approve_task
        )
        self.approve_btn.pack(side='right', expand=True, fill='x', padx=(5, 0))
    
    # ========================================
    # 작업 처리
    # ========================================
    def assign_task(self):
        """Ralph에게 작업 할당"""
        task_desc = self.task_entry.get().strip()
        if not task_desc or task_desc.startswith('예:'):
            messagebox.showwarning("경고", "작업 설명을 입력해주세요!")
            return
        
        # 담당 Ralph 결정
        ralph_choice = self.ralph_var.get()
        if ralph_choice.startswith('auto'):
            # 자동 배정: 언어에 따라
            lang = self.lang_var.get()
            if lang in ['python', 'java', 'csharp']:
                assigned_ralph = self.ralphs[0]  # Backend
            elif lang in ['javascript', 'html', 'css']:
                assigned_ralph = self.ralphs[1]  # Frontend
            else:
                assigned_ralph = self.ralphs[0]
        else:
            # 지정된 Ralph
            ralph_name = ralph_choice
            assigned_ralph = next((r for r in self.ralphs if r['name'] == ralph_name), self.ralphs[0])
        
        # 작업 생성
        task = {
            'id': len(self.pending_tasks) + len(self.approved_tasks) + 1,
            'description': task_desc,
            'language': self.lang_var.get(),
            'ralph': assigned_ralph,
            'status': 'working',
            'code': None,
            'created_at': datetime.now().strftime('%H:%M:%S'),
            'attempts': 0
        }
        
        # Ralph 상태 업데이트
        self.update_ralph_status(assigned_ralph['id'], 'working')
        
        # 입력 초기화
        self.task_entry.delete(0, 'end')
        
        # 백그라운드에서 코드 생성 시뮬레이션
        threading.Thread(target=self.simulate_code_generation, args=(task,), daemon=True).start()
    
    def simulate_code_generation(self, task):
        """코드 생성 시뮬레이션 (실제로는 LLM API 호출)"""
        time.sleep(2)  # 시뮬레이션 딜레이
        
        # 샘플 코드 생성
        sample_codes = {
            'python': '''# {desc}
class Calculator:
    """기본 계산기 클래스"""
    
    def add(self, a: float, b: float) -> float:
        """두 수를 더합니다"""
        return a + b
    
    def subtract(self, a: float, b: float) -> float:
        """두 수를 뺍니다"""
        return a - b
    
    def multiply(self, a: float, b: float) -> float:
        """두 수를 곱합니다"""
        return a * b
    
    def divide(self, a: float, b: float) -> float:
        """두 수를 나눕니다"""
        if b == 0:
            raise ValueError("0으로 나눌 수 없습니다")
        return a / b


if __name__ == "__main__":
    calc = Calculator()
    print(calc.add(10, 5))  # 15
''',
            'javascript': '''// {desc}
class Calculator {{
    add(a, b) {{
        return a + b;
    }}
    
    subtract(a, b) {{
        return a - b;
    }}
    
    multiply(a, b) {{
        return a * b;
    }}
    
    divide(a, b) {{
        if (b === 0) throw new Error("0으로 나눌 수 없습니다");
        return a / b;
    }}
}}

// 사용 예시
const calc = new Calculator();
console.log(calc.add(10, 5)); // 15
''',
            'html': '''<!-- {desc} -->
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>Login Form</title>
    <style>
        body {{ font-family: Arial; background: #1e1e2e; }}
        .login-form {{
            max-width: 400px;
            margin: 100px auto;
            padding: 30px;
            background: #313244;
            border-radius: 10px;
        }}
        input {{
            width: 100%;
            padding: 12px;
            margin: 10px 0;
            border: none;
            border-radius: 5px;
        }}
        button {{
            width: 100%;
            padding: 12px;
            background: #6366f1;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }}
    </style>
</head>
<body>
    <div class="login-form">
        <h2 style="color: white;">로그인</h2>
        <input type="email" placeholder="이메일">
        <input type="password" placeholder="비밀번호">
        <button>로그인</button>
    </div>
</body>
</html>
''',
            'java': '''// {desc}
public class Calculator {{
    
    public double add(double a, double b) {{
        return a + b;
    }}
    
    public double subtract(double a, double b) {{
        return a - b;
    }}
    
    public double multiply(double a, double b) {{
        return a * b;
    }}
    
    public double divide(double a, double b) {{
        if (b == 0) {{
            throw new ArithmeticException("0으로 나눌 수 없습니다");
        }}
        return a / b;
    }}
    
    public static void main(String[] args) {{
        Calculator calc = new Calculator();
        System.out.println(calc.add(10, 5)); // 15.0
    }}
}}
''',
            'csharp': '''// {desc}
using System;

public class Calculator
{{
    public double Add(double a, double b)
    {{
        return a + b;
    }}
    
    public double Subtract(double a, double b)
    {{
        return a - b;
    }}
    
    public double Multiply(double a, double b)
    {{
        return a * b;
    }}
    
    public double Divide(double a, double b)
    {{
        if (b == 0)
            throw new DivideByZeroException("0으로 나눌 수 없습니다");
        return a / b;
    }}
    
    public static void Main()
    {{
        var calc = new Calculator();
        Console.WriteLine(calc.Add(10, 5)); // 15
    }}
}}
'''
        }
        
        lang = task['language']
        code_template = sample_codes.get(lang, sample_codes['python'])
        task['code'] = code_template.format(desc=task['description'])
        task['status'] = 'pending_approval'
        task['attempts'] += 1
        
        # UI 업데이트 (메인 스레드에서)
        self.root.after(0, lambda: self.on_code_generated(task))
    
    def on_code_generated(self, task):
        """코드 생성 완료 후 UI 업데이트"""
        # Ralph 상태 업데이트
        self.update_ralph_status(task['ralph']['id'], 'done')
        
        # 대기열에 추가
        self.pending_tasks.append(task)
        
        # UI 업데이트
        self.refresh_queue()
        self.update_status_labels()
        
        # 2초 후 Ralph 상태 idle로
        self.root.after(2000, lambda: self.update_ralph_status(task['ralph']['id'], 'idle'))
    
    def update_ralph_status(self, ralph_id, status):
        """Ralph 상태 업데이트"""
        for ralph in self.ralphs:
            if ralph['id'] == ralph_id:
                ralph['status'] = status
                break
        
        # UI 업데이트
        if ralph_id in self.ralph_frames:
            frame_data = self.ralph_frames[ralph_id]
            status_label = frame_data['status_label']
            
            status_colors = {
                'idle': COLORS['text_muted'],
                'working': COLORS['warning'],
                'done': COLORS['success']
            }
            status_texts = {
                'idle': '● 대기 중',
                'working': '● 작업 중...',
                'done': '● 완료!'
            }
            
            status_label.config(
                text=status_texts.get(status, '● 대기 중'),
                fg=status_colors.get(status, COLORS['text_muted'])
            )
    
    def refresh_queue(self):
        """대기열 UI 새로고침"""
        # 기존 위젯 삭제
        for widget in self.queue_container.winfo_children():
            widget.destroy()
        
        if not self.pending_tasks:
            self.empty_label = tk.Label(
                self.queue_container,
                text="아직 대기 중인 작업이 없습니다.\n위에서 작업을 요청하세요!",
                font=('Segoe UI', 11),
                bg=COLORS['bg_surface'],
                fg=COLORS['text_muted']
            )
            self.empty_label.pack(pady=50)
            return
        
        # 작업 카드 생성
        for task in self.pending_tasks:
            self.create_task_card(task)
    
    def create_task_card(self, task):
        """작업 카드 생성"""
        card = tk.Frame(self.queue_container, bg=COLORS['bg_overlay'], cursor='hand2')
        card.pack(fill='x', pady=5)
        
        # 클릭 이벤트
        card.bind('<Button-1>', lambda e, t=task: self.select_task(t))
        
        inner = tk.Frame(card, bg=COLORS['bg_overlay'])
        inner.pack(fill='x', padx=15, pady=12)
        inner.bind('<Button-1>', lambda e, t=task: self.select_task(t))
        
        # 상단: 작업명 + 시간
        top = tk.Frame(inner, bg=COLORS['bg_overlay'])
        top.pack(fill='x')
        top.bind('<Button-1>', lambda e, t=task: self.select_task(t))
        
        title = tk.Label(
            top,
            text=task['description'][:40] + ('...' if len(task['description']) > 40 else ''),
            font=('Segoe UI', 11, 'bold'),
            bg=COLORS['bg_overlay'],
            fg=COLORS['text_light']
        )
        title.pack(side='left')
        title.bind('<Button-1>', lambda e, t=task: self.select_task(t))
        
        time_label = tk.Label(
            top,
            text=task['created_at'],
            font=('Segoe UI', 9),
            bg=COLORS['bg_overlay'],
            fg=COLORS['text_muted']
        )
        time_label.pack(side='right')
        time_label.bind('<Button-1>', lambda e, t=task: self.select_task(t))
        
        # 하단: Ralph + 언어 + 시도 횟수
        bottom = tk.Frame(inner, bg=COLORS['bg_overlay'])
        bottom.pack(fill='x', pady=(8, 0))
        bottom.bind('<Button-1>', lambda e, t=task: self.select_task(t))
        
        ralph_label = tk.Label(
            bottom,
            text=f"{task['ralph']['icon']} {task['ralph']['name']}",
            font=('Segoe UI', 10),
            bg=COLORS['bg_overlay'],
            fg=COLORS['primary']
        )
        ralph_label.pack(side='left')
        ralph_label.bind('<Button-1>', lambda e, t=task: self.select_task(t))
        
        lang_label = tk.Label(
            bottom,
            text=task['language'],
            font=('Segoe UI', 9),
            bg=COLORS['primary'],
            fg='white',
            padx=8,
            pady=2
        )
        lang_label.pack(side='left', padx=10)
        lang_label.bind('<Button-1>', lambda e, t=task: self.select_task(t))
        
        if task['attempts'] > 1:
            retry_label = tk.Label(
                bottom,
                text=f"재시도 {task['attempts']}회",
                font=('Segoe UI', 9),
                bg=COLORS['bg_overlay'],
                fg=COLORS['warning']
            )
            retry_label.pack(side='right')
            retry_label.bind('<Button-1>', lambda e, t=task: self.select_task(t))
    
    def select_task(self, task):
        """작업 선택"""
        self.selected_task = task
        
        # 검토 정보 업데이트
        self.review_title.config(text=task['description'][:50])
        self.review_info.config(
            text=f"담당: {task['ralph']['name']}\n"
                 f"언어: {task['language']}\n"
                 f"시도: {task['attempts']}회"
        )
        
        # 코드 미리보기 업데이트
        self.code_preview.config(state='normal')
        self.code_preview.delete('1.0', 'end')
        self.code_preview.insert('1.0', task['code'] or '코드 없음')
        self.code_preview.config(state='disabled')
        
        # 버튼 활성화
        self.approve_btn.config(state='normal')
        self.reject_btn.config(state='normal')
    
    def approve_task(self):
        """작업 승인"""
        if not self.selected_task:
            return
        
        task = self.selected_task
        
        # 대기열에서 제거
        self.pending_tasks.remove(task)
        
        # 승인 목록에 추가
        task['status'] = 'approved'
        self.approved_tasks.append(task)
        
        # UI 초기화
        self.selected_task = None
        self.review_title.config(text="작업을 선택하세요")
        self.review_info.config(text="승인되었습니다! ✅")
        self.code_preview.config(state='normal')
        self.code_preview.delete('1.0', 'end')
        self.code_preview.insert('1.0', "# 승인 완료! 다음 작업을 선택하세요.")
        self.code_preview.config(state='disabled')
        
        self.approve_btn.config(state='disabled')
        self.reject_btn.config(state='disabled')
        
        # 새로고침
        self.refresh_queue()
        self.update_status_labels()
        
        messagebox.showinfo("승인 완료", f"'{task['description'][:30]}...' 작업이 승인되었습니다!")
    
    def reject_task(self):
        """작업 거절 (재작업 요청)"""
        if not self.selected_task:
            return
        
        task = self.selected_task
        
        # 대기열에서 제거
        self.pending_tasks.remove(task)
        
        # Ralph에게 재작업 요청
        task['status'] = 'working'
        self.update_ralph_status(task['ralph']['id'], 'working')
        
        # UI 초기화
        self.selected_task = None
        self.review_title.config(text="작업을 선택하세요")
        self.review_info.config(text="재작업 요청됨 🔄")
        
        self.approve_btn.config(state='disabled')
        self.reject_btn.config(state='disabled')
        
        # 새로고침
        self.refresh_queue()
        self.update_status_labels()
        
        # 재작업 시뮬레이션
        threading.Thread(target=self.simulate_code_generation, args=(task,), daemon=True).start()
    
    def update_status_labels(self):
        """상태 레이블 업데이트"""
        self.pending_label.config(text=f"⏳ 대기: {len(self.pending_tasks)}")
        self.approved_label.config(text=f"✅ 승인: {len(self.approved_tasks)}")


# ========================================
# 앱 실행
# ========================================
if __name__ == "__main__":
    app = CodingAgentApp()
    app.run()