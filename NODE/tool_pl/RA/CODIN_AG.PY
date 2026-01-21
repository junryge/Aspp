"""
🤖 Coding Agent - Workflow Map UI
워크플로우 맵 시각화 & 설정
- 스크롤 지원
- 화면 맞춤
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import json
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
    'node_prd': '#8b5cf6',
    'node_ralph': '#3b82f6',
    'node_process': '#f59e0b',
    'node_decision': '#6366f1',
    'node_done': '#10b981',
    'node_implement': '#22c55e',
}


class WorkflowMapApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🤖 Coding Agent - Workflow Map")
        
        # 화면 크기 가져와서 맞춤
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        
        # 80% 크기로 설정
        win_w = min(1200, int(screen_w * 0.85))
        win_h = min(750, int(screen_h * 0.85))
        
        # 중앙 배치
        x = (screen_w - win_w) // 2
        y = (screen_h - win_h) // 2
        
        self.root.geometry(f"{win_w}x{win_h}+{x}+{y}")
        self.root.configure(bg=COLORS['bg_darker'])
        
        # 최소 크기
        self.root.minsize(900, 600)
        
        # 워크플로우 노드 정의
        self.nodes = self.create_workflow_nodes()
        self.connections = self.create_connections()
        
        # 현재 단계
        self.current_step = 0
        self.total_steps = len(self.nodes)
        self.selected_node = None
        
        # UI 생성
        self.create_ui()
        
        # 워크플로우 그리기
        self.root.after(100, self.draw_workflow)
    
    def create_workflow_nodes(self):
        """워크플로우 노드 생성 - 더 작은 좌표"""
        return [
            {
                'id': 'prd_write',
                'type': 'admin',
                'title': 'PRD 작성',
                'desc': '만들고 싶은 것 정의',
                'x': 80, 'y': 30,
                'color': COLORS['node_prd'],
                'status': 'pending',
                'settings': {'template': 'default'}
            },
            {
                'id': 'prd_convert',
                'type': 'process',
                'title': 'prd.json 변환',
                'desc': '유저 스토리 분할',
                'x': 80, 'y': 110,
                'color': COLORS['node_process'],
                'status': 'pending',
                'settings': {}
            },
            {
                'id': 'ralph_start',
                'type': 'process',
                'title': 'Ralph 실행',
                'desc': '자기수정 루프',
                'x': 80, 'y': 190,
                'color': COLORS['node_process'],
                'status': 'pending',
                'settings': {'max_retries': 10}
            },
            {
                'id': 'pick_story',
                'type': 'ralph',
                'title': '스토리 선택',
                'desc': 'passes:false 찾기',
                'x': 80, 'y': 280,
                'color': COLORS['node_ralph'],
                'status': 'pending',
                'settings': {'assigned_ralph': 'auto'}
            },
            {
                'id': 'more_stories',
                'type': 'decision',
                'title': '더 있나?',
                'desc': '',
                'x': 50, 'y': 380,
                'color': COLORS['node_decision'],
                'status': 'pending',
                'settings': {}
            },
            {
                'id': 'implement',
                'type': 'implement',
                'title': '구현',
                'desc': '코드 작성/테스트',
                'x': 320, 'y': 220,
                'color': COLORS['node_implement'],
                'status': 'pending',
                'settings': {'language': 'python', 'max_retries': 10, 'run_tests': True}
            },
            {
                'id': 'commit',
                'type': 'process',
                'title': '커밋',
                'desc': '변경사항 저장',
                'x': 450, 'y': 220,
                'color': COLORS['node_process'],
                'status': 'pending',
                'settings': {'auto_commit': True}
            },
            {
                'id': 'update_prd',
                'type': 'process',
                'title': 'prd 업데이트',
                'desc': 'passes:true',
                'x': 320, 'y': 310,
                'color': COLORS['node_process'],
                'status': 'pending',
                'settings': {}
            },
            {
                'id': 'log_progress',
                'type': 'process',
                'title': 'progress 로그',
                'desc': '진행상황 기록',
                'x': 200, 'y': 380,
                'color': COLORS['node_process'],
                'status': 'pending',
                'settings': {}
            },
            {
                'id': 'done',
                'type': 'done',
                'title': '완료!',
                'desc': '모든 스토리 완료',
                'x': 50, 'y': 470,
                'color': COLORS['node_done'],
                'status': 'pending',
                'settings': {}
            },
        ]
    
    def create_connections(self):
        """노드 연결 정의"""
        return [
            ('prd_write', 'prd_convert'),
            ('prd_convert', 'ralph_start'),
            ('ralph_start', 'pick_story'),
            ('pick_story', 'implement'),
            ('pick_story', 'more_stories'),
            ('more_stories', 'done', 'No'),
            ('more_stories', 'log_progress', 'Yes'),
            ('implement', 'commit'),
            ('implement', 'update_prd'),
            ('update_prd', 'log_progress'),
            ('log_progress', 'pick_story'),
        ]
    
    def run(self):
        self.root.mainloop()
    
    # ========================================
    # UI 생성
    # ========================================
    def create_ui(self):
        # 상단 헤더
        self.create_header()
        
        # 메인 영역 (PanedWindow로 리사이즈 가능)
        self.paned = ttk.PanedWindow(self.root, orient='horizontal')
        self.paned.pack(fill='both', expand=True, padx=5, pady=5)
        
        # 왼쪽: 워크플로우 맵 캔버스
        self.create_canvas()
        
        # 오른쪽: 설정 패널
        self.create_settings_panel()
        
        # 하단: 네비게이션
        self.create_navigation()
    
    def create_header(self):
        """상단 헤더"""
        header = tk.Frame(self.root, bg=COLORS['bg_dark'], height=45)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        # 로고
        logo = tk.Label(
            header,
            text="🤖 Coding Agent - Workflow Map",
            font=('Segoe UI', 14, 'bold'),
            bg=COLORS['bg_dark'],
            fg=COLORS['primary']
        )
        logo.pack(side='left', padx=15, pady=10)
        
        # 상태
        self.status_label = tk.Label(
            header,
            text="Ready",
            font=('Segoe UI', 10),
            bg=COLORS['bg_dark'],
            fg=COLORS['success']
        )
        self.status_label.pack(side='right', padx=15)
    
    def create_canvas(self):
        """워크플로우 맵 캔버스 (스크롤 지원)"""
        canvas_frame = tk.Frame(self.paned, bg=COLORS['bg_surface'])
        self.paned.add(canvas_frame, weight=3)
        
        # 캔버스 제목
        title = tk.Label(
            canvas_frame,
            text="📍 Workflow Map (클릭하여 노드 선택)",
            font=('Segoe UI', 11, 'bold'),
            bg=COLORS['bg_surface'],
            fg=COLORS['text_light']
        )
        title.pack(anchor='w', padx=10, pady=8)
        
        # 스크롤 가능한 캔버스
        canvas_container = tk.Frame(canvas_frame, bg=COLORS['bg_surface'])
        canvas_container.pack(fill='both', expand=True, padx=5, pady=(0, 5))
        
        # 스크롤바
        h_scroll = tk.Scrollbar(canvas_container, orient='horizontal')
        h_scroll.pack(side='bottom', fill='x')
        
        v_scroll = tk.Scrollbar(canvas_container, orient='vertical')
        v_scroll.pack(side='right', fill='y')
        
        # 캔버스
        self.canvas = tk.Canvas(
            canvas_container,
            bg=COLORS['bg_dark'],
            highlightthickness=0,
            xscrollcommand=h_scroll.set,
            yscrollcommand=v_scroll.set
        )
        self.canvas.pack(side='left', fill='both', expand=True)
        
        h_scroll.config(command=self.canvas.xview)
        v_scroll.config(command=self.canvas.yview)
        
        # 스크롤 영역 설정
        self.canvas.config(scrollregion=(0, 0, 600, 550))
        
        # 클릭 이벤트
        self.canvas.bind('<Button-1>', self.on_canvas_click)
        
        # 마우스 휠 스크롤
        self.canvas.bind('<MouseWheel>', lambda e: self.canvas.yview_scroll(-1*(e.delta//120), "units"))
    
    def create_settings_panel(self):
        """오른쪽 설정 패널 (스크롤 지원)"""
        panel_frame = tk.Frame(self.paned, bg=COLORS['bg_surface'])
        self.paned.add(panel_frame, weight=1)
        
        # 스크롤 가능한 프레임
        canvas = tk.Canvas(panel_frame, bg=COLORS['bg_surface'], highlightthickness=0)
        scrollbar = tk.Scrollbar(panel_frame, orient="vertical", command=canvas.yview)
        self.panel = tk.Frame(canvas, bg=COLORS['bg_surface'])
        
        self.panel.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.panel, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        # 마우스 휠
        canvas.bind('<MouseWheel>', lambda e: canvas.yview_scroll(-1*(e.delta//120), "units"))
        self.panel.bind('<MouseWheel>', lambda e: canvas.yview_scroll(-1*(e.delta//120), "units"))
        
        # 노드 정보
        info_frame = tk.Frame(self.panel, bg=COLORS['bg_overlay'])
        info_frame.pack(fill='x', padx=8, pady=8)
        
        self.node_title = tk.Label(
            info_frame,
            text="노드를 선택하세요",
            font=('Segoe UI', 12, 'bold'),
            bg=COLORS['bg_overlay'],
            fg=COLORS['text_light']
        )
        self.node_title.pack(anchor='w', padx=10, pady=(10, 3))
        
        self.node_desc = tk.Label(
            info_frame,
            text="맵에서 노드 클릭",
            font=('Segoe UI', 9),
            bg=COLORS['bg_overlay'],
            fg=COLORS['text_muted'],
            justify='left'
        )
        self.node_desc.pack(anchor='w', padx=10, pady=(0, 10))
        
        # 설정 영역
        tk.Label(
            self.panel,
            text="⚙️ 노드 설정",
            font=('Segoe UI', 11, 'bold'),
            bg=COLORS['bg_surface'],
            fg=COLORS['text_light']
        ).pack(anchor='w', padx=10, pady=(10, 5))
        
        # 설정 컨테이너
        self.settings_container = tk.Frame(self.panel, bg=COLORS['bg_surface'])
        self.settings_container.pack(fill='x', padx=10)
        
        self.show_default_settings()
        
        # JSON 미리보기
        tk.Label(
            self.panel,
            text="📄 JSON",
            font=('Segoe UI', 11, 'bold'),
            bg=COLORS['bg_surface'],
            fg=COLORS['text_light']
        ).pack(anchor='w', padx=10, pady=(15, 5))
        
        self.json_preview = tk.Text(
            self.panel,
            font=('Consolas', 8),
            bg=COLORS['bg_dark'],
            fg=COLORS['text_light'],
            relief='flat',
            height=10,
            width=30
        )
        self.json_preview.pack(fill='x', padx=8, pady=(0, 10))
        self.update_json_preview()
    
    def show_default_settings(self):
        """기본 설정 표시"""
        for widget in self.settings_container.winfo_children():
            widget.destroy()
        
        tk.Label(
            self.settings_container,
            text="노드 선택 시 설정 표시",
            font=('Segoe UI', 9),
            bg=COLORS['bg_surface'],
            fg=COLORS['text_muted']
        ).pack(pady=15)
    
    def show_node_settings(self, node):
        """선택된 노드 설정 표시"""
        for widget in self.settings_container.winfo_children():
            widget.destroy()
        
        settings = node.get('settings', {})
        
        # 담당 Ralph
        if node['type'] in ['ralph', 'implement']:
            row = tk.Frame(self.settings_container, bg=COLORS['bg_surface'])
            row.pack(fill='x', pady=3)
            
            tk.Label(row, text="담당:", font=('Segoe UI', 9),
                bg=COLORS['bg_surface'], fg=COLORS['text_muted']).pack(side='left')
            
            ralph_var = tk.StringVar(value=settings.get('assigned_ralph', 'auto'))
            ralph_menu = ttk.Combobox(row, textvariable=ralph_var,
                values=['auto', 'Backend', 'Frontend', 'Test'],
                state='readonly', width=12)
            ralph_menu.pack(side='right')
            ralph_menu.bind('<<ComboboxSelected>>', 
                lambda e, n=node, v=ralph_var: self.update_node_setting(n, 'assigned_ralph', v.get()))
        
        # 최대 반복
        if 'max_retries' in settings:
            row = tk.Frame(self.settings_container, bg=COLORS['bg_surface'])
            row.pack(fill='x', pady=3)
            
            tk.Label(row, text="최대 반복:", font=('Segoe UI', 9),
                bg=COLORS['bg_surface'], fg=COLORS['text_muted']).pack(side='left')
            
            retry_var = tk.StringVar(value=str(settings.get('max_retries', 10)))
            retry_spin = tk.Spinbox(row, from_=1, to=20, textvariable=retry_var,
                width=5, font=('Segoe UI', 9), bg=COLORS['bg_dark'], fg=COLORS['text_light'])
            retry_spin.pack(side='right')
        
        # 언어
        if 'language' in settings:
            row = tk.Frame(self.settings_container, bg=COLORS['bg_surface'])
            row.pack(fill='x', pady=3)
            
            tk.Label(row, text="언어:", font=('Segoe UI', 9),
                bg=COLORS['bg_surface'], fg=COLORS['text_muted']).pack(side='left')
            
            lang_var = tk.StringVar(value=settings.get('language', 'python'))
            lang_menu = ttk.Combobox(row, textvariable=lang_var,
                values=['python', 'java', 'javascript', 'csharp'],
                state='readonly', width=12)
            lang_menu.pack(side='right')
        
        # 테스트 실행
        if 'run_tests' in settings:
            test_var = tk.BooleanVar(value=settings.get('run_tests', True))
            tk.Checkbutton(self.settings_container, text="테스트 실행", variable=test_var,
                font=('Segoe UI', 9), bg=COLORS['bg_surface'], fg=COLORS['text_light'],
                selectcolor=COLORS['bg_dark']).pack(anchor='w', pady=2)
        
        # 자동 커밋
        if 'auto_commit' in settings:
            commit_var = tk.BooleanVar(value=settings.get('auto_commit', True))
            tk.Checkbutton(self.settings_container, text="자동 커밋", variable=commit_var,
                font=('Segoe UI', 9), bg=COLORS['bg_surface'], fg=COLORS['text_light'],
                selectcolor=COLORS['bg_dark']).pack(anchor='w', pady=2)
        
        if not settings:
            tk.Label(self.settings_container, text="설정 없음", font=('Segoe UI', 9),
                bg=COLORS['bg_surface'], fg=COLORS['text_muted']).pack(pady=10)
    
    def update_node_setting(self, node, key, value):
        node['settings'][key] = value
        self.update_json_preview()
    
    def update_json_preview(self):
        if self.selected_node:
            data = {
                'id': self.selected_node['id'],
                'title': self.selected_node['title'],
                'status': self.selected_node['status'],
                'settings': self.selected_node['settings']
            }
        else:
            data = {'step': f"{self.current_step+1}/{self.total_steps}"}
        
        self.json_preview.config(state='normal')
        self.json_preview.delete('1.0', 'end')
        self.json_preview.insert('1.0', json.dumps(data, indent=2, ensure_ascii=False))
        self.json_preview.config(state='disabled')
    
    def create_navigation(self):
        """하단 네비게이션"""
        nav = tk.Frame(self.root, bg=COLORS['bg_dark'], height=50)
        nav.pack(fill='x', side='bottom')
        nav.pack_propagate(False)
        
        center = tk.Frame(nav, bg=COLORS['bg_dark'])
        center.pack(expand=True, pady=8)
        
        # Previous
        self.prev_btn = tk.Button(center, text="◀ Previous", font=('Segoe UI', 10),
            bg=COLORS['bg_surface'], fg=COLORS['text_light'], relief='flat',
            padx=15, pady=5, command=self.prev_step)
        self.prev_btn.pack(side='left', padx=3)
        
        # Step 표시
        self.step_label = tk.Label(center, text=f"Step {self.current_step+1} of {self.total_steps}",
            font=('Segoe UI', 11, 'bold'), bg=COLORS['bg_dark'], fg=COLORS['text_light'], padx=20)
        self.step_label.pack(side='left')
        
        # Next
        self.next_btn = tk.Button(center, text="Next ▶", font=('Segoe UI', 10),
            bg=COLORS['primary'], fg='white', relief='flat',
            padx=15, pady=5, command=self.next_step)
        self.next_btn.pack(side='left', padx=3)
        
        # Reset
        tk.Button(center, text="Reset", font=('Segoe UI', 10),
            bg=COLORS['bg_surface'], fg=COLORS['text_light'], relief='flat',
            padx=15, pady=5, command=self.reset_workflow).pack(side='left', padx=15)
        
        # Run All
        tk.Button(center, text="▶ Run All", font=('Segoe UI', 10, 'bold'),
            bg=COLORS['success'], fg='white', relief='flat',
            padx=15, pady=5, command=self.run_all).pack(side='left', padx=3)
    
    # ========================================
    # 워크플로우 그리기
    # ========================================
    def draw_workflow(self):
        self.canvas.delete('all')
        
        # 연결선
        for conn in self.connections:
            self.draw_connection(conn)
        
        # 노드
        for i, node in enumerate(self.nodes):
            self.draw_node(node, is_current=(i == self.current_step))
    
    def draw_connection(self, conn):
        from_id, to_id = conn[0], conn[1]
        label = conn[2] if len(conn) > 2 else None
        
        from_node = next((n for n in self.nodes if n['id'] == from_id), None)
        to_node = next((n for n in self.nodes if n['id'] == to_id), None)
        
        if not from_node or not to_node:
            return
        
        x1, y1 = from_node['x'] + 55, from_node['y'] + 25
        x2, y2 = to_node['x'] + 55, to_node['y'] + 25
        
        # 선 그리기
        self.canvas.create_line(x1, y1, x2, y2, fill=COLORS['text_muted'], width=2, arrow=tk.LAST)
        
        # 라벨
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            color = COLORS['success'] if label == 'Yes' else COLORS['danger']
            self.canvas.create_text(mx+15, my, text=label, font=('Segoe UI', 8), fill=color)
    
    def draw_node(self, node, is_current=False):
        x, y = node['x'], node['y']
        w, h = 110, 50
        
        # 상태별 색상
        if node['status'] == 'completed':
            fill = COLORS['success']
            outline = COLORS['success']
        elif node['status'] == 'running':
            fill = node['color']
            outline = COLORS['warning']
        elif is_current:
            fill = node['color']
            outline = COLORS['primary']
        else:
            fill = COLORS['bg_overlay']
            outline = node['color']
        
        outline_w = 3 if is_current else 2
        
        # 사각형
        self.canvas.create_rectangle(x, y, x+w, y+h, fill=fill, outline=outline, width=outline_w)
        
        # 제목
        text_color = 'white' if node['status'] in ['completed', 'running'] or is_current else COLORS['text_light']
        self.canvas.create_text(x+w/2, y+17, text=node['title'], font=('Segoe UI', 9, 'bold'), fill=text_color)
        
        # 설명
        self.canvas.create_text(x+w/2, y+35, text=node['desc'][:15], font=('Segoe UI', 7), fill=COLORS['text_muted'])
        
        node['canvas_coords'] = (x, y, x+w, y+h)
    
    def on_canvas_click(self, event):
        # 캔버스 스크롤 보정
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        for node in self.nodes:
            coords = node.get('canvas_coords')
            if coords:
                x1, y1, x2, y2 = coords
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.select_node(node)
                    return
        
        self.selected_node = None
        self.node_title.config(text="노드를 선택하세요")
        self.node_desc.config(text="맵에서 노드 클릭")
        self.show_default_settings()
        self.update_json_preview()
        self.draw_workflow()
    
    def select_node(self, node):
        self.selected_node = node
        self.node_title.config(text=node['title'])
        self.node_desc.config(text=f"타입: {node['type']}\n상태: {node['status']}")
        self.show_node_settings(node)
        self.update_json_preview()
        self.draw_workflow()
    
    # ========================================
    # 네비게이션
    # ========================================
    def prev_step(self):
        if self.current_step > 0:
            self.current_step -= 1
            self.update_step_ui()
    
    def next_step(self):
        if self.current_step < self.total_steps - 1:
            self.nodes[self.current_step]['status'] = 'completed'
            self.current_step += 1
            self.nodes[self.current_step]['status'] = 'running'
            self.update_step_ui()
    
    def reset_workflow(self):
        self.current_step = 0
        for node in self.nodes:
            node['status'] = 'pending'
        self.nodes[0]['status'] = 'running'
        self.update_step_ui()
    
    def update_step_ui(self):
        self.step_label.config(text=f"Step {self.current_step+1} of {self.total_steps}")
        self.status_label.config(text=f"현재: {self.nodes[self.current_step]['title']}")
        self.select_node(self.nodes[self.current_step])
        self.prev_btn.config(state='normal' if self.current_step > 0 else 'disabled')
        self.next_btn.config(state='normal' if self.current_step < self.total_steps-1 else 'disabled')
        self.draw_workflow()
    
    def run_all(self):
        def run():
            for i in range(self.current_step, self.total_steps):
                self.root.after(0, lambda idx=i: self.set_running(idx))
                time.sleep(0.8)
                self.root.after(0, lambda idx=i: self.set_completed(idx))
            self.root.after(0, lambda: messagebox.showinfo("완료", "모든 단계 완료!"))
        
        threading.Thread(target=run, daemon=True).start()
    
    def set_running(self, idx):
        self.nodes[idx]['status'] = 'running'
        self.current_step = idx
        self.update_step_ui()
    
    def set_completed(self, idx):
        self.nodes[idx]['status'] = 'completed'
        self.draw_workflow()


if __name__ == "__main__":
    app = WorkflowMapApp()
    app.run()