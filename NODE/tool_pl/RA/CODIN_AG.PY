"""
🤖 Coding Agent - Workflow Map UI
워크플로우 맵 시각화 & 설정
- 노드별 작업 설정
- 반복 횟수 설정
- 단계별 진행 (Previous/Next)
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
    'node_admin': '#ec4899',
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
        self.root.geometry("1400x900")
        self.root.configure(bg=COLORS['bg_darker'])
        
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
        """워크플로우 노드 생성"""
        return [
            {
                'id': 'prd_write',
                'type': 'admin',
                'title': 'PRD 작성',
                'desc': '만들고 싶은 것 정의',
                'x': 120, 'y': 60,
                'color': COLORS['node_prd'],
                'status': 'pending',
                'settings': {
                    'template': 'default',
                }
            },
            {
                'id': 'prd_convert',
                'type': 'process',
                'title': 'prd.json 변환',
                'desc': '유저 스토리로 분할',
                'x': 120, 'y': 160,
                'color': COLORS['node_process'],
                'status': 'pending',
                'settings': {}
            },
            {
                'id': 'ralph_start',
                'type': 'process',
                'title': 'Ralph 실행',
                'desc': '자기수정 루프 시작',
                'x': 120, 'y': 260,
                'color': COLORS['node_process'],
                'status': 'pending',
                'settings': {
                    'max_retries': 10,
                }
            },
            {
                'id': 'pick_story',
                'type': 'ralph',
                'title': '스토리 선택',
                'desc': 'passes: false 찾기',
                'x': 120, 'y': 380,
                'color': COLORS['node_ralph'],
                'status': 'pending',
                'settings': {
                    'assigned_ralph': 'auto',
                }
            },
            {
                'id': 'more_stories',
                'type': 'decision',
                'title': '더 있나?',
                'desc': '남은 스토리 확인',
                'x': 70, 'y': 510,
                'color': COLORS['node_decision'],
                'status': 'pending',
                'settings': {}
            },
            {
                'id': 'implement',
                'type': 'implement',
                'title': '구현',
                'desc': '코드 작성, 테스트 실행',
                'x': 420, 'y': 300,
                'color': COLORS['node_implement'],
                'status': 'pending',
                'settings': {
                    'language': 'python',
                    'max_retries': 10,
                    'run_tests': True,
                }
            },
            {
                'id': 'update_prd',
                'type': 'process',
                'title': 'prd.json 업데이트',
                'desc': 'passes: true 설정',
                'x': 420, 'y': 420,
                'color': COLORS['node_process'],
                'status': 'pending',
                'settings': {}
            },
            {
                'id': 'commit',
                'type': 'process',
                'title': '커밋',
                'desc': '변경사항 저장',
                'x': 580, 'y': 300,
                'color': COLORS['node_process'],
                'status': 'pending',
                'settings': {
                    'auto_commit': True,
                }
            },
            {
                'id': 'log_progress',
                'type': 'process',
                'title': 'progress.txt 로그',
                'desc': '진행 상황 기록',
                'x': 250, 'y': 510,
                'color': COLORS['node_process'],
                'status': 'pending',
                'settings': {}
            },
            {
                'id': 'done',
                'type': 'done',
                'title': '완료!',
                'desc': '모든 스토리 완료',
                'x': 70, 'y': 620,
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
            ('pick_story', 'more_stories'),
            ('pick_story', 'implement'),
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
        
        # 메인 영역
        main_frame = tk.Frame(self.root, bg=COLORS['bg_darker'])
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # 왼쪽: 워크플로우 맵 캔버스
        self.create_canvas(main_frame)
        
        # 오른쪽: 설정 패널
        self.create_settings_panel(main_frame)
        
        # 하단: 네비게이션
        self.create_navigation()
    
    def create_header(self):
        """상단 헤더"""
        header = tk.Frame(self.root, bg=COLORS['bg_dark'], height=60)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        # 로고
        logo = tk.Label(
            header,
            text="🤖 Coding Agent - Workflow Map",
            font=('Segoe UI', 18, 'bold'),
            bg=COLORS['bg_dark'],
            fg=COLORS['primary']
        )
        logo.pack(side='left', padx=20, pady=15)
        
        # 상태
        self.status_label = tk.Label(
            header,
            text="Ready",
            font=('Segoe UI', 11),
            bg=COLORS['bg_dark'],
            fg=COLORS['success']
        )
        self.status_label.pack(side='right', padx=20)
    
    def create_canvas(self, parent):
        """워크플로우 맵 캔버스"""
        canvas_frame = tk.Frame(parent, bg=COLORS['bg_surface'])
        canvas_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        # 캔버스 제목
        title = tk.Label(
            canvas_frame,
            text="📍 Workflow Map",
            font=('Segoe UI', 14, 'bold'),
            bg=COLORS['bg_surface'],
            fg=COLORS['text_light']
        )
        title.pack(anchor='w', padx=15, pady=10)
        
        # 캔버스
        self.canvas = tk.Canvas(
            canvas_frame,
            bg=COLORS['bg_dark'],
            highlightthickness=0,
            width=700,
            height=700
        )
        self.canvas.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        
        # 클릭 이벤트
        self.canvas.bind('<Button-1>', self.on_canvas_click)
    
    def create_settings_panel(self, parent):
        """오른쪽 설정 패널"""
        panel = tk.Frame(parent, bg=COLORS['bg_surface'], width=400)
        panel.pack(side='right', fill='y')
        panel.pack_propagate(False)
        
        # 노드 정보
        info_frame = tk.Frame(panel, bg=COLORS['bg_overlay'])
        info_frame.pack(fill='x', padx=10, pady=10)
        
        self.node_title = tk.Label(
            info_frame,
            text="노드를 선택하세요",
            font=('Segoe UI', 14, 'bold'),
            bg=COLORS['bg_overlay'],
            fg=COLORS['text_light']
        )
        self.node_title.pack(anchor='w', padx=15, pady=(15, 5))
        
        self.node_desc = tk.Label(
            info_frame,
            text="왼쪽 맵에서 노드를 클릭하면\n설정을 변경할 수 있습니다.",
            font=('Segoe UI', 10),
            bg=COLORS['bg_overlay'],
            fg=COLORS['text_muted'],
            justify='left'
        )
        self.node_desc.pack(anchor='w', padx=15, pady=(0, 15))
        
        # 설정 영역
        settings_label = tk.Label(
            panel,
            text="⚙️ 노드 설정",
            font=('Segoe UI', 12, 'bold'),
            bg=COLORS['bg_surface'],
            fg=COLORS['text_light']
        )
        settings_label.pack(anchor='w', padx=15, pady=(15, 10))
        
        # 설정 컨테이너
        self.settings_container = tk.Frame(panel, bg=COLORS['bg_surface'])
        self.settings_container.pack(fill='x', padx=15)
        
        # 기본 설정 표시
        self.show_default_settings()
        
        # JSON 미리보기
        json_label = tk.Label(
            panel,
            text="📄 JSON 미리보기",
            font=('Segoe UI', 12, 'bold'),
            bg=COLORS['bg_surface'],
            fg=COLORS['text_light']
        )
        json_label.pack(anchor='w', padx=15, pady=(20, 10))
        
        self.json_preview = scrolledtext.ScrolledText(
            panel,
            font=('Consolas', 9),
            bg=COLORS['bg_dark'],
            fg=COLORS['text_light'],
            relief='flat',
            height=12
        )
        self.json_preview.pack(fill='x', padx=10, pady=(0, 10))
        self.update_json_preview()
    
    def show_default_settings(self):
        """기본 설정 표시"""
        for widget in self.settings_container.winfo_children():
            widget.destroy()
        
        tk.Label(
            self.settings_container,
            text="노드를 선택하면 설정이 표시됩니다.",
            font=('Segoe UI', 10),
            bg=COLORS['bg_surface'],
            fg=COLORS['text_muted']
        ).pack(pady=20)
    
    def show_node_settings(self, node):
        """선택된 노드 설정 표시"""
        for widget in self.settings_container.winfo_children():
            widget.destroy()
        
        settings = node.get('settings', {})
        
        # 노드 타입별 설정
        if node['type'] in ['ralph', 'implement']:
            # 담당 Ralph 선택
            row = tk.Frame(self.settings_container, bg=COLORS['bg_surface'])
            row.pack(fill='x', pady=5)
            
            tk.Label(
                row, text="담당 Ralph:",
                font=('Segoe UI', 10),
                bg=COLORS['bg_surface'],
                fg=COLORS['text_muted']
            ).pack(side='left')
            
            ralph_var = tk.StringVar(value=settings.get('assigned_ralph', 'auto'))
            ralph_menu = ttk.Combobox(
                row,
                textvariable=ralph_var,
                values=['auto', 'Backend Ralph', 'Frontend Ralph', 'Test Ralph'],
                state='readonly',
                width=18
            )
            ralph_menu.pack(side='right')
            ralph_menu.bind('<<ComboboxSelected>>', 
                lambda e, n=node, v=ralph_var: self.update_node_setting(n, 'assigned_ralph', v.get()))
        
        if 'max_retries' in settings:
            # 최대 반복 횟수
            row = tk.Frame(self.settings_container, bg=COLORS['bg_surface'])
            row.pack(fill='x', pady=5)
            
            tk.Label(
                row, text="최대 반복 횟수:",
                font=('Segoe UI', 10),
                bg=COLORS['bg_surface'],
                fg=COLORS['text_muted']
            ).pack(side='left')
            
            retry_var = tk.StringVar(value=str(settings.get('max_retries', 10)))
            retry_spin = tk.Spinbox(
                row,
                from_=1, to=20,
                textvariable=retry_var,
                width=5,
                font=('Segoe UI', 10),
                bg=COLORS['bg_dark'],
                fg=COLORS['text_light']
            )
            retry_spin.pack(side='right')
            retry_spin.bind('<FocusOut>', 
                lambda e, n=node, v=retry_var: self.update_node_setting(n, 'max_retries', int(v.get())))
        
        if 'language' in settings:
            # 언어 선택
            row = tk.Frame(self.settings_container, bg=COLORS['bg_surface'])
            row.pack(fill='x', pady=5)
            
            tk.Label(
                row, text="언어:",
                font=('Segoe UI', 10),
                bg=COLORS['bg_surface'],
                fg=COLORS['text_muted']
            ).pack(side='left')
            
            lang_var = tk.StringVar(value=settings.get('language', 'python'))
            lang_menu = ttk.Combobox(
                row,
                textvariable=lang_var,
                values=['python', 'java', 'javascript', 'csharp', 'html'],
                state='readonly',
                width=15
            )
            lang_menu.pack(side='right')
            lang_menu.bind('<<ComboboxSelected>>', 
                lambda e, n=node, v=lang_var: self.update_node_setting(n, 'language', v.get()))
        
        if 'run_tests' in settings:
            # 테스트 실행 여부
            row = tk.Frame(self.settings_container, bg=COLORS['bg_surface'])
            row.pack(fill='x', pady=5)
            
            test_var = tk.BooleanVar(value=settings.get('run_tests', True))
            test_check = tk.Checkbutton(
                row,
                text="테스트 자동 실행",
                variable=test_var,
                font=('Segoe UI', 10),
                bg=COLORS['bg_surface'],
                fg=COLORS['text_light'],
                selectcolor=COLORS['bg_dark'],
                activebackground=COLORS['bg_surface'],
                command=lambda n=node, v=test_var: self.update_node_setting(n, 'run_tests', v.get())
            )
            test_check.pack(side='left')
        
        if 'auto_commit' in settings:
            # 자동 커밋 여부
            row = tk.Frame(self.settings_container, bg=COLORS['bg_surface'])
            row.pack(fill='x', pady=5)
            
            commit_var = tk.BooleanVar(value=settings.get('auto_commit', True))
            commit_check = tk.Checkbutton(
                row,
                text="자동 커밋",
                variable=commit_var,
                font=('Segoe UI', 10),
                bg=COLORS['bg_surface'],
                fg=COLORS['text_light'],
                selectcolor=COLORS['bg_dark'],
                activebackground=COLORS['bg_surface'],
                command=lambda n=node, v=commit_var: self.update_node_setting(n, 'auto_commit', v.get())
            )
            commit_check.pack(side='left')
        
        # 설정이 없는 노드
        if not settings:
            tk.Label(
                self.settings_container,
                text="이 노드는 설정이 없습니다.",
                font=('Segoe UI', 10),
                bg=COLORS['bg_surface'],
                fg=COLORS['text_muted']
            ).pack(pady=10)
    
    def update_node_setting(self, node, key, value):
        """노드 설정 업데이트"""
        node['settings'][key] = value
        self.update_json_preview()
    
    def update_json_preview(self):
        """JSON 미리보기 업데이트"""
        if self.selected_node:
            data = {
                'id': self.selected_node['id'],
                'title': self.selected_node['title'],
                'type': self.selected_node['type'],
                'status': self.selected_node['status'],
                'settings': self.selected_node['settings']
            }
        else:
            # 전체 워크플로우 요약
            data = {
                'total_nodes': len(self.nodes),
                'current_step': self.current_step + 1,
                'nodes': [{'id': n['id'], 'status': n['status']} for n in self.nodes]
            }
        
        self.json_preview.config(state='normal')
        self.json_preview.delete('1.0', 'end')
        self.json_preview.insert('1.0', json.dumps(data, indent=2, ensure_ascii=False))
        self.json_preview.config(state='disabled')
    
    def create_navigation(self):
        """하단 네비게이션"""
        nav = tk.Frame(self.root, bg=COLORS['bg_dark'], height=60)
        nav.pack(fill='x', side='bottom')
        nav.pack_propagate(False)
        
        # 중앙 정렬용 프레임
        center = tk.Frame(nav, bg=COLORS['bg_dark'])
        center.pack(expand=True)
        
        # Previous 버튼
        self.prev_btn = tk.Button(
            center,
            text="◀ Previous",
            font=('Segoe UI', 11),
            bg=COLORS['bg_surface'],
            fg=COLORS['text_light'],
            relief='flat',
            padx=20, pady=8,
            cursor='hand2',
            command=self.prev_step
        )
        self.prev_btn.pack(side='left', padx=5)
        
        # 단계 표시
        self.step_label = tk.Label(
            center,
            text=f"Step {self.current_step + 1} of {self.total_steps}",
            font=('Segoe UI', 12, 'bold'),
            bg=COLORS['bg_dark'],
            fg=COLORS['text_light'],
            padx=30
        )
        self.step_label.pack(side='left')
        
        # Next 버튼
        self.next_btn = tk.Button(
            center,
            text="Next ▶",
            font=('Segoe UI', 11),
            bg=COLORS['primary'],
            fg='white',
            relief='flat',
            padx=20, pady=8,
            cursor='hand2',
            command=self.next_step
        )
        self.next_btn.pack(side='left', padx=5)
        
        # Reset 버튼
        self.reset_btn = tk.Button(
            center,
            text="Reset",
            font=('Segoe UI', 11),
            bg=COLORS['bg_surface'],
            fg=COLORS['text_light'],
            relief='flat',
            padx=20, pady=8,
            cursor='hand2',
            command=self.reset_workflow
        )
        self.reset_btn.pack(side='left', padx=20)
        
        # Run All 버튼
        self.run_btn = tk.Button(
            center,
            text="▶ Run All",
            font=('Segoe UI', 11, 'bold'),
            bg=COLORS['success'],
            fg='white',
            relief='flat',
            padx=20, pady=8,
            cursor='hand2',
            command=self.run_all
        )
        self.run_btn.pack(side='left', padx=5)
    
    # ========================================
    # 워크플로우 그리기
    # ========================================
    def draw_workflow(self):
        """워크플로우 맵 그리기"""
        self.canvas.delete('all')
        
        # 연결선 먼저 그리기
        for conn in self.connections:
            self.draw_connection(conn)
        
        # 노드 그리기
        for i, node in enumerate(self.nodes):
            self.draw_node(node, is_current=(i == self.current_step))
    
    def draw_connection(self, conn):
        """연결선 그리기"""
        from_id = conn[0]
        to_id = conn[1]
        label = conn[2] if len(conn) > 2 else None
        
        from_node = next((n for n in self.nodes if n['id'] == from_id), None)
        to_node = next((n for n in self.nodes if n['id'] == to_id), None)
        
        if not from_node or not to_node:
            return
        
        # 시작점, 끝점 계산
        x1 = from_node['x'] + 70
        y1 = from_node['y'] + 30
        x2 = to_node['x'] + 70
        y2 = to_node['y'] + 30
        
        # 곡선 연결
        if abs(x2 - x1) > 100:
            # 수평 이동이 큰 경우 베지어 곡선
            mid_x = (x1 + x2) / 2
            self.canvas.create_line(
                x1, y1, mid_x, y1, mid_x, y2, x2, y2,
                fill=COLORS['text_muted'],
                width=2,
                smooth=True,
                arrow=tk.LAST
            )
        else:
            # 직선
            self.canvas.create_line(
                x1, y1, x2, y2,
                fill=COLORS['text_muted'],
                width=2,
                arrow=tk.LAST
            )
        
        # 라벨 (Yes/No)
        if label:
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            self.canvas.create_text(
                mid_x + 20, mid_y,
                text=label,
                font=('Segoe UI', 9),
                fill=COLORS['warning'] if label == 'Yes' else COLORS['danger']
            )
    
    def draw_node(self, node, is_current=False):
        """노드 그리기"""
        x, y = node['x'], node['y']
        w, h = 140, 60
        
        # 상태별 색상
        if node['status'] == 'completed':
            fill_color = COLORS['success']
            outline_color = COLORS['success']
        elif node['status'] == 'running':
            fill_color = node['color']
            outline_color = COLORS['warning']
        elif is_current:
            fill_color = node['color']
            outline_color = COLORS['primary']
        else:
            fill_color = COLORS['bg_overlay']
            outline_color = node['color']
        
        # 현재 단계 강조
        outline_width = 4 if is_current else 2
        
        # 둥근 사각형
        radius = 10
        self.canvas.create_rectangle(
            x, y, x + w, y + h,
            fill=fill_color,
            outline=outline_color,
            width=outline_width
        )
        
        # 제목
        self.canvas.create_text(
            x + w/2, y + 20,
            text=node['title'],
            font=('Segoe UI', 10, 'bold'),
            fill='white' if node['status'] in ['completed', 'running'] or is_current else COLORS['text_light']
        )
        
        # 설명
        self.canvas.create_text(
            x + w/2, y + 40,
            text=node['desc'][:20],
            font=('Segoe UI', 8),
            fill=COLORS['text_muted']
        )
        
        # 노드 ID 저장 (클릭 감지용)
        node['canvas_coords'] = (x, y, x + w, y + h)
    
    def on_canvas_click(self, event):
        """캔버스 클릭 이벤트"""
        x, y = event.x, event.y
        
        for node in self.nodes:
            coords = node.get('canvas_coords')
            if coords:
                x1, y1, x2, y2 = coords
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.select_node(node)
                    return
        
        # 빈 공간 클릭시 선택 해제
        self.selected_node = None
        self.node_title.config(text="노드를 선택하세요")
        self.node_desc.config(text="왼쪽 맵에서 노드를 클릭하면\n설정을 변경할 수 있습니다.")
        self.show_default_settings()
        self.update_json_preview()
        self.draw_workflow()
    
    def select_node(self, node):
        """노드 선택"""
        self.selected_node = node
        
        # 정보 업데이트
        self.node_title.config(text=f"{node['title']}")
        self.node_desc.config(text=f"타입: {node['type']}\n상태: {node['status']}\n{node['desc']}")
        
        # 설정 표시
        self.show_node_settings(node)
        
        # JSON 업데이트
        self.update_json_preview()
        
        # 맵 다시 그리기 (선택 표시)
        self.draw_workflow()
    
    # ========================================
    # 네비게이션
    # ========================================
    def prev_step(self):
        """이전 단계"""
        if self.current_step > 0:
            self.current_step -= 1
            self.update_step_ui()
    
    def next_step(self):
        """다음 단계"""
        if self.current_step < self.total_steps - 1:
            # 현재 노드 완료 처리
            self.nodes[self.current_step]['status'] = 'completed'
            self.current_step += 1
            self.nodes[self.current_step]['status'] = 'running'
            self.update_step_ui()
    
    def reset_workflow(self):
        """워크플로우 리셋"""
        self.current_step = 0
        for node in self.nodes:
            node['status'] = 'pending'
        self.nodes[0]['status'] = 'running'
        self.update_step_ui()
    
    def update_step_ui(self):
        """단계 UI 업데이트"""
        self.step_label.config(text=f"Step {self.current_step + 1} of {self.total_steps}")
        self.status_label.config(text=f"현재: {self.nodes[self.current_step]['title']}")
        
        # 선택된 노드 업데이트
        self.select_node(self.nodes[self.current_step])
        
        # 버튼 상태
        self.prev_btn.config(state='normal' if self.current_step > 0 else 'disabled')
        self.next_btn.config(state='normal' if self.current_step < self.total_steps - 1 else 'disabled')
        
        self.draw_workflow()
    
    def run_all(self):
        """전체 실행"""
        def run_steps():
            for i in range(self.current_step, self.total_steps):
                self.root.after(0, lambda idx=i: self.set_step_running(idx))
                time.sleep(1)
                self.root.after(0, lambda idx=i: self.set_step_completed(idx))
            
            self.root.after(0, lambda: messagebox.showinfo("완료", "모든 단계가 완료되었습니다!"))
        
        threading.Thread(target=run_steps, daemon=True).start()
    
    def set_step_running(self, idx):
        """단계 실행 중으로 설정"""
        self.nodes[idx]['status'] = 'running'
        self.current_step = idx
        self.update_step_ui()
    
    def set_step_completed(self, idx):
        """단계 완료로 설정"""
        self.nodes[idx]['status'] = 'completed'
        self.draw_workflow()


# ========================================
# 앱 실행
# ========================================
if __name__ == "__main__":
    app = WorkflowMapApp()
    app.run()