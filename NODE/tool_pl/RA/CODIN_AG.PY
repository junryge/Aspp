"""
🤖 Coding Agent - 워크플로우
1. 관리자: 작업 지시
2. Ralph: 코드 생성
3. Ralph: 1차 검증 (실행/테스트)
   - 실패 → 다시 수정 (자기수정 루프)
   - 성공 → 최종 승인자에게 제출
4. 최종 승인자: 승인/거절
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import time
import subprocess
import tempfile
import os

# 색상
COLORS = {
    'bg': '#1e1e2e',
    'surface': '#313244',
    'overlay': '#45475a',
    'primary': '#6366f1',
    'success': '#10b981',
    'warning': '#f59e0b',
    'danger': '#ef4444',
    'text': '#cdd6f4',
    'muted': '#6c7086',
}


class CodingAgentApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🤖 Coding Agent")
        self.root.geometry("1000x800")
        self.root.configure(bg=COLORS['bg'])
        
        # 상태
        self.generated_code = ""
        self.attempt = 0          # 현재 시도 횟수
        self.max_attempts = 10    # 최대 시도 횟수
        self.errors = []          # 에러 히스토리
        
        self.create_ui()
    
    def run(self):
        self.root.mainloop()
    
    def create_ui(self):
        # ===== 상단: 워크플로우 표시 =====
        flow_frame = tk.Frame(self.root, bg=COLORS['surface'], height=90)
        flow_frame.pack(fill='x', padx=10, pady=10)
        flow_frame.pack_propagate(False)
        
        # 워크플로우 다이어그램
        flow_label = tk.Label(
            flow_frame,
            text="🛡️ 관리자(지시) → 🤖 Ralph(생성) → 🔍 Ralph(검증) → ✅ 최종승인자(확인)",
            font=('Segoe UI', 12, 'bold'),
            bg=COLORS['surface'],
            fg=COLORS['text']
        )
        flow_label.pack(pady=10)
        
        # 현재 상태
        self.status_label = tk.Label(
            flow_frame,
            text="📍 현재: 관리자 - 작업 지시 대기중",
            font=('Segoe UI', 11),
            bg=COLORS['surface'],
            fg=COLORS['primary']
        )
        self.status_label.pack()
        
        # ===== 메인 영역 =====
        main = tk.Frame(self.root, bg=COLORS['bg'])
        main.pack(fill='both', expand=True, padx=10)
        
        # --- 왼쪽: 관리자 입력 ---
        left = tk.Frame(main, bg=COLORS['surface'], width=350)
        left.pack(side='left', fill='y', padx=(0, 5))
        left.pack_propagate(False)
        
        tk.Label(
            left,
            text="🛡️ 관리자 - 작업 지시",
            font=('Segoe UI', 13, 'bold'),
            bg=COLORS['surface'],
            fg=COLORS['primary']
        ).pack(anchor='w', padx=15, pady=(15, 10))
        
        # 프롬프트 입력
        tk.Label(left, text="무엇을 만들까요?", font=('Segoe UI', 10),
            bg=COLORS['surface'], fg=COLORS['muted']).pack(anchor='w', padx=15)
        
        self.prompt_text = scrolledtext.ScrolledText(
            left, font=('Segoe UI', 10), bg=COLORS['overlay'],
            fg=COLORS['text'], height=5, wrap='word'
        )
        self.prompt_text.pack(fill='x', padx=15, pady=5)
        self.prompt_text.insert('1.0', "Calculator 클래스를 만들어줘.\nadd, subtract, multiply, divide 메서드 포함.")
        
        # 옵션
        opt_frame = tk.Frame(left, bg=COLORS['surface'])
        opt_frame.pack(fill='x', padx=15, pady=10)
        
        tk.Label(opt_frame, text="Ralph:", font=('Segoe UI', 10),
            bg=COLORS['surface'], fg=COLORS['muted']).pack(side='left')
        self.ralph_var = tk.StringVar(value='Backend Ralph')
        ttk.Combobox(opt_frame, textvariable=self.ralph_var,
            values=['Backend Ralph', 'Frontend Ralph', 'Test Ralph'],
            state='readonly', width=14).pack(side='left', padx=5)
        
        tk.Label(opt_frame, text="언어:", font=('Segoe UI', 10),
            bg=COLORS['surface'], fg=COLORS['muted']).pack(side='left', padx=(10, 0))
        self.lang_var = tk.StringVar(value='python')
        ttk.Combobox(opt_frame, textvariable=self.lang_var,
            values=['python', 'javascript', 'java'],
            state='readonly', width=10).pack(side='left', padx=5)
        
        # 최대 시도 횟수
        retry_frame = tk.Frame(left, bg=COLORS['surface'])
        retry_frame.pack(fill='x', padx=15, pady=5)
        
        tk.Label(retry_frame, text="최대 재시도:", font=('Segoe UI', 10),
            bg=COLORS['surface'], fg=COLORS['muted']).pack(side='left')
        self.retry_var = tk.StringVar(value='10')
        tk.Spinbox(retry_frame, from_=1, to=20, textvariable=self.retry_var,
            width=5, font=('Segoe UI', 10)).pack(side='left', padx=10)
        
        # 시작 버튼
        self.start_btn = tk.Button(
            left, text="🚀 Ralph에게 시키기",
            font=('Segoe UI', 12, 'bold'),
            bg=COLORS['primary'], fg='white',
            relief='flat', cursor='hand2',
            command=self.start_work
        )
        self.start_btn.pack(fill='x', padx=15, pady=15)
        
        # Ralph 진행 상태
        tk.Label(left, text="🤖 Ralph 작업 현황",
            font=('Segoe UI', 11, 'bold'),
            bg=COLORS['surface'], fg=COLORS['warning']).pack(anchor='w', padx=15, pady=(10, 5))
        
        self.ralph_log = scrolledtext.ScrolledText(
            left, font=('Consolas', 9), bg=COLORS['overlay'],
            fg=COLORS['text'], height=12
        )
        self.ralph_log.pack(fill='both', expand=True, padx=15, pady=(0, 15))
        self.ralph_log.insert('1.0', "대기 중...")
        
        # --- 오른쪽: 코드 & 최종 승인 ---
        right = tk.Frame(main, bg=COLORS['surface'])
        right.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        tk.Label(
            right,
            text="📄 생성된 코드",
            font=('Segoe UI', 13, 'bold'),
            bg=COLORS['surface'],
            fg=COLORS['success']
        ).pack(anchor='w', padx=15, pady=(15, 5))
        
        # 시도 횟수 표시
        self.attempt_label = tk.Label(
            right,
            text="시도: 0 / 10",
            font=('Segoe UI', 10),
            bg=COLORS['surface'],
            fg=COLORS['muted']
        )
        self.attempt_label.pack(anchor='w', padx=15)
        
        # 코드 표시
        self.code_text = scrolledtext.ScrolledText(
            right, font=('Consolas', 10), bg=COLORS['overlay'],
            fg=COLORS['text'], height=18
        )
        self.code_text.pack(fill='both', expand=True, padx=15, pady=5)
        self.code_text.insert('1.0', "# 코드가 여기에 표시됩니다")
        
        # 최종 승인 영역
        approval_frame = tk.Frame(right, bg=COLORS['overlay'])
        approval_frame.pack(fill='x', padx=15, pady=10)
        
        tk.Label(
            approval_frame,
            text="🛡️ 최종 승인자 확인",
            font=('Segoe UI', 11, 'bold'),
            bg=COLORS['overlay'],
            fg=COLORS['primary']
        ).pack(anchor='w', padx=10, pady=(10, 5))
        
        self.approval_status = tk.Label(
            approval_frame,
            text="Ralph가 검증 완료 후 제출하면 여기서 확인합니다.",
            font=('Segoe UI', 10),
            bg=COLORS['overlay'],
            fg=COLORS['muted']
        )
        self.approval_status.pack(anchor='w', padx=10, pady=(0, 10))
        
        btn_frame = tk.Frame(approval_frame, bg=COLORS['overlay'])
        btn_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        self.reject_btn = tk.Button(
            btn_frame, text="❌ 거절 (재작업)",
            font=('Segoe UI', 11, 'bold'),
            bg=COLORS['danger'], fg='white',
            relief='flat', state='disabled',
            command=self.reject_code
        )
        self.reject_btn.pack(side='left', expand=True, fill='x', padx=(0, 5))
        
        self.approve_btn = tk.Button(
            btn_frame, text="✅ 최종 승인",
            font=('Segoe UI', 11, 'bold'),
            bg=COLORS['success'], fg='white',
            relief='flat', state='disabled',
            command=self.approve_code
        )
        self.approve_btn.pack(side='right', expand=True, fill='x', padx=(5, 0))
    
    def log(self, message):
        """Ralph 로그 추가"""
        self.ralph_log.insert('end', f"\n{message}")
        self.ralph_log.see('end')
    
    def start_work(self):
        """작업 시작"""
        prompt = self.prompt_text.get('1.0', 'end').strip()
        if not prompt:
            messagebox.showwarning("경고", "작업 내용을 입력해주세요!")
            return
        
        # 초기화
        self.attempt = 0
        self.max_attempts = int(self.retry_var.get())
        self.errors = []
        
        # UI 상태 변경
        self.start_btn.config(state='disabled')
        self.approve_btn.config(state='disabled')
        self.reject_btn.config(state='disabled')
        
        self.ralph_log.delete('1.0', 'end')
        self.log(f"🚀 작업 시작!")
        self.log(f"담당: {self.ralph_var.get()}")
        self.log(f"최대 시도: {self.max_attempts}회")
        
        self.status_label.config(text="📍 현재: Ralph - 코드 생성 중...")
        
        # 백그라운드에서 실행
        threading.Thread(target=self.ralph_loop, args=(prompt,), daemon=True).start()
    
    def ralph_loop(self, prompt):
        """Ralph 자기수정 루프"""
        lang = self.lang_var.get()
        
        while self.attempt < self.max_attempts:
            self.attempt += 1
            self.root.after(0, lambda a=self.attempt: self.attempt_label.config(
                text=f"시도: {a} / {self.max_attempts}"))
            
            # 1. 코드 생성
            self.root.after(0, lambda: self.log(f"\n--- 시도 {self.attempt} ---"))
            self.root.after(0, lambda: self.log("📝 코드 생성 중..."))
            self.root.after(0, lambda: self.status_label.config(
                text=f"📍 현재: Ralph - 코드 생성 중... (시도 {self.attempt})"))
            
            time.sleep(1)
            code = self.generate_code(prompt, lang)
            
            self.root.after(0, lambda c=code: self.show_code(c))
            self.root.after(0, lambda: self.log("✅ 코드 생성 완료"))
            
            # 2. 1차 검증
            self.root.after(0, lambda: self.log("🔍 검증 중... (문법/실행 테스트)"))
            self.root.after(0, lambda: self.status_label.config(
                text=f"📍 현재: Ralph - 검증 중... (시도 {self.attempt})"))
            
            time.sleep(1)
            result = self.verify_code(code, lang)
            
            if result['success']:
                # 검증 통과!
                self.root.after(0, lambda: self.log("✅ 검증 통과!"))
                self.root.after(0, lambda: self.log("📤 최종 승인자에게 제출합니다."))
                self.root.after(0, self.submit_to_admin)
                return
            else:
                # 검증 실패 → 다시 시도
                error = result['error']
                self.errors.append(error)
                self.root.after(0, lambda e=error: self.log(f"❌ 검증 실패: {e}"))
                self.root.after(0, lambda: self.log("🔄 수정 후 재시도..."))
                
                # 에러를 프롬프트에 포함시켜 재생성
                prompt = f"{prompt}\n\n[이전 에러: {error}]\n위 에러를 수정해주세요."
        
        # 최대 시도 초과
        self.root.after(0, lambda: self.log(f"\n💀 {self.max_attempts}회 시도 후 실패"))
        self.root.after(0, lambda: self.status_label.config(
            text="📍 현재: 실패 - 최대 시도 횟수 초과"))
        self.root.after(0, lambda: self.start_btn.config(state='normal'))
    
    def generate_code(self, prompt, lang):
        """코드 생성 (시뮬레이션 - 실제로는 LLM API 호출)"""
        # 시도 횟수에 따라 코드 품질 조정 (시뮬레이션)
        if self.attempt >= 2:
            # 2번째부터는 "수정된" 코드 반환
            return self.get_correct_code(prompt, lang)
        else:
            # 첫 시도는 일부러 에러 있는 코드 (시뮬레이션)
            return self.get_buggy_code(prompt, lang)
    
    def get_buggy_code(self, prompt, lang):
        """에러가 있는 코드 (시뮬레이션)"""
        if lang == 'python':
            return f'''# {prompt}

class Calculator:
    def add(self, a, b):
        return a + b
    
    def subtract(self, a, b):
        return a - b
    
    def multiply(self, a, b):
        return a * b
    
    def divide(self, a, b):
        return a / b  # 0으로 나누기 에러 가능!

# 테스트
calc = Calculator()
print(calc.divide(10, 0))  # 에러 발생!
'''
        else:
            return f"// {prompt}\nconsole.log('buggy code')"
    
    def get_correct_code(self, prompt, lang):
        """수정된 정상 코드"""
        if lang == 'python':
            return f'''# {prompt}

class Calculator:
    """사칙연산 계산기"""
    
    def add(self, a: float, b: float) -> float:
        return a + b
    
    def subtract(self, a: float, b: float) -> float:
        return a - b
    
    def multiply(self, a: float, b: float) -> float:
        return a * b
    
    def divide(self, a: float, b: float) -> float:
        if b == 0:
            raise ValueError("0으로 나눌 수 없습니다")
        return a / b


# 테스트
if __name__ == "__main__":
    calc = Calculator()
    print(calc.add(10, 5))       # 15
    print(calc.subtract(10, 5))  # 5
    print(calc.multiply(10, 5))  # 50
    print(calc.divide(10, 5))    # 2.0
'''
        elif lang == 'javascript':
            return f'''// {prompt}

class Calculator {{
    add(a, b) {{ return a + b; }}
    subtract(a, b) {{ return a - b; }}
    multiply(a, b) {{ return a * b; }}
    divide(a, b) {{
        if (b === 0) throw new Error("0으로 나눌 수 없습니다");
        return a / b;
    }}
}}

const calc = new Calculator();
console.log(calc.add(10, 5)); // 15
'''
        else:
            return f"// {prompt}\n// 정상 코드"
    
    def verify_code(self, code, lang):
        """코드 검증 (실제 실행)"""
        if lang == 'python':
            return self.verify_python(code)
        else:
            # 다른 언어는 간단히 통과 (시뮬레이션)
            return {'success': True}
    
    def verify_python(self, code):
        """Python 코드 검증"""
        # 임시 파일에 저장
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(code)
            temp_path = f.name
        
        try:
            # 1. 문법 체크
            result = subprocess.run(
                ['python', '-m', 'py_compile', temp_path],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                return {'success': False, 'error': f"문법 에러: {result.stderr}"}
            
            # 2. 실행 테스트
            result = subprocess.run(
                ['python', temp_path],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                return {'success': False, 'error': f"실행 에러: {result.stderr}"}
            
            return {'success': True, 'output': result.stdout}
            
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': "타임아웃"}
        except Exception as e:
            return {'success': False, 'error': str(e)}
        finally:
            os.unlink(temp_path)
    
    def show_code(self, code):
        """코드 표시"""
        self.generated_code = code
        self.code_text.delete('1.0', 'end')
        self.code_text.insert('1.0', code)
    
    def submit_to_admin(self):
        """최종 승인자에게 제출"""
        self.status_label.config(text="📍 현재: 최종 승인자 - 검토 대기중")
        self.approval_status.config(
            text=f"✅ Ralph 검증 완료! ({self.attempt}회 시도)\n승인 또는 거절을 선택하세요.",
            fg=COLORS['success']
        )
        self.approve_btn.config(state='normal')
        self.reject_btn.config(state='normal')
    
    def approve_code(self):
        """최종 승인"""
        # 파일 저장
        ext = {'python': '.py', 'javascript': '.js', 'java': '.java'}
        filename = f"output{ext.get(self.lang_var.get(), '.txt')}"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(self.generated_code)
        
        messagebox.showinfo("✅ 승인 완료!", f"코드가 승인되었습니다.\n저장됨: {filename}")
        
        self.reset_ui()
    
    def reject_code(self):
        """거절 → Ralph 재작업"""
        if messagebox.askyesno("재작업", "Ralph에게 다시 시킬까요?"):
            self.log("\n🔄 최종 승인자가 거절했습니다. 재작업 시작...")
            self.approval_status.config(text="재작업 요청됨", fg=COLORS['warning'])
            self.approve_btn.config(state='disabled')
            self.reject_btn.config(state='disabled')
            
            prompt = self.prompt_text.get('1.0', 'end').strip()
            threading.Thread(target=self.ralph_loop, args=(prompt,), daemon=True).start()
    
    def reset_ui(self):
        """UI 리셋"""
        self.start_btn.config(state='normal')
        self.approve_btn.config(state='disabled')
        self.reject_btn.config(state='disabled')
        self.status_label.config(text="📍 현재: 관리자 - 작업 지시 대기중")
        self.approval_status.config(
            text="Ralph가 검증 완료 후 제출하면 여기서 확인합니다.",
            fg=COLORS['muted']
        )
        self.attempt_label.config(text="시도: 0 / 10")


if __name__ == "__main__":
    app = CodingAgentApp()
    app.run()