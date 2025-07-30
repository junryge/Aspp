import tkinter as tk
from tkinter import ttk, messagebox
import json
from datetime import datetime
import random

class Node:
    """노드 클래스 - 각 처리 단계를 나타냄"""
    def __init__(self, canvas, x, y, node_type, node_id, name):
        self.canvas = canvas
        self.x = x
        self.y = y
        self.node_type = node_type
        self.node_id = node_id
        self.name = name
        self.width = 180
        self.height = 80
        self.inputs = []
        self.outputs = []
        self.connected_nodes = []
        self.is_configured = False
        self.config = {}
        
        # 노드 타입별 색상
        self.colors = {
            'data': '#3498db',      # 파란색 - 데이터 입력
            'preprocess': '#e74c3c', # 빨간색 - 전처리
            'vector': '#f39c12',     # 주황색 - 벡터 저장
            'model': '#27ae60',      # 초록색 - 모델
            'analysis': '#9b59b6',   # 보라색 - 분석
            'prompt': '#34495e',     # 회색 - 프롬프트
            'llm': '#16a085'         # 청록색 - LLM
        }
        
        self.create_node()
        
    def create_node(self):
        """노드 시각화"""
        # 그림자 효과
        shadow_offset = 3
        self.shadow = self.canvas.create_rectangle(
            self.x + shadow_offset, self.y + shadow_offset, 
            self.x + self.width + shadow_offset, self.y + self.height + shadow_offset,
            fill='#1a1a2e',
            outline='',
            tags=f"node_{self.node_id}"
        )
        
        # 노드 배경
        self.rect = self.canvas.create_rectangle(
            self.x, self.y, 
            self.x + self.width, self.y + self.height,
            fill=self.colors.get(self.node_type, '#7f8c8d'),
            outline='white',
            width=2,
            tags=f"node_{self.node_id}"
        )
        
        # 노드 이름
        self.text = self.canvas.create_text(
            self.x + self.width/2, self.y + 20,
            text=self.name,
            fill='white',
            font=('Arial', 11, 'bold'),
            tags=f"node_{self.node_id}"
        )
        
        # 노드 타입
        self.type_text = self.canvas.create_text(
            self.x + self.width/2, self.y + 40,
            text=f"[{self.node_type}]",
            fill='#ecf0f1',
            font=('Arial', 9),
            tags=f"node_{self.node_id}"
        )
        
        # 상태 표시
        self.status = self.canvas.create_oval(
            self.x + self.width - 20, self.y + 10,
            self.x + self.width - 10, self.y + 20,
            fill='red' if not self.is_configured else 'green',
            outline='white',
            tags=f"node_{self.node_id}"
        )
        
        # 입력/출력 포트
        self.input_port = self.canvas.create_rectangle(
            self.x - 10, self.y + self.height/2 - 5,
            self.x, self.y + self.height/2 + 5,
            fill='yellow',
            outline='white',
            tags=f"port_in_{self.node_id}"
        )
        
        self.output_port = self.canvas.create_rectangle(
            self.x + self.width, self.y + self.height/2 - 5,
            self.x + self.width + 10, self.y + self.height/2 + 5,
            fill='cyan',
            outline='white',
            tags=f"port_out_{self.node_id}"
        )
        
        # 마우스 이벤트 바인딩 - 커서 변경
        for item in [self.shadow, self.rect, self.text, self.type_text, self.status]:
            self.canvas.tag_bind(item, "<Enter>", lambda e: self.canvas.config(cursor="hand2"))
            self.canvas.tag_bind(item, "<Leave>", lambda e: self.canvas.config(cursor=""))
            
        # 포트 커서
        self.canvas.tag_bind(self.input_port, "<Enter>", lambda e: self.canvas.config(cursor="crosshair"))
        self.canvas.tag_bind(self.input_port, "<Leave>", lambda e: self.canvas.config(cursor=""))
        self.canvas.tag_bind(self.output_port, "<Enter>", lambda e: self.canvas.config(cursor="crosshair"))
        self.canvas.tag_bind(self.output_port, "<Leave>", lambda e: self.canvas.config(cursor=""))
        
    def move(self, dx, dy):
        """노드 이동"""
        self.x += dx
        self.y += dy
        
        # 모든 노드 구성요소 이동
        for tag in [f"node_{self.node_id}", f"port_in_{self.node_id}", f"port_out_{self.node_id}"]:
            items = self.canvas.find_withtag(tag)
            for item in items:
                self.canvas.move(item, dx, dy)
        
    def get_output_pos(self):
        """출력 포트 위치 반환"""
        return (self.x + self.width + 5, self.y + self.height/2)
        
    def get_input_pos(self):
        """입력 포트 위치 반환"""
        return (self.x - 5, self.y + self.height/2)
        
    def set_configured(self, configured):
        """설정 상태 업데이트"""
        self.is_configured = configured
        color = 'green' if configured else 'red'
        self.canvas.itemconfig(self.status, fill=color)


class Connection:
    """노드 간 연결을 나타내는 클래스"""
    def __init__(self, canvas, start_node, end_node):
        self.canvas = canvas
        self.start_node = start_node
        self.end_node = end_node
        self.line = None
        self.create_connection()
        
    def create_connection(self):
        """연결선 생성"""
        start_x, start_y = self.start_node.get_output_pos()
        end_x, end_y = self.end_node.get_input_pos()
        
        # 베지어 곡선 스타일 연결선
        mid_x = (start_x + end_x) / 2
        self.line = self.canvas.create_line(
            start_x, start_y,
            mid_x, start_y,
            mid_x, end_y,
            end_x, end_y,
            fill='#3498db',
            width=3,
            smooth=True,
            arrow=tk.LAST,
            tags="connection"
        )
        
        # 연결선을 노드 아래로
        self.canvas.tag_lower(self.line)
        self.canvas.tag_raise(self.line, "grid")
        
    def update(self):
        """연결선 위치 업데이트"""
        if self.line:
            start_x, start_y = self.start_node.get_output_pos()
            end_x, end_y = self.end_node.get_input_pos()
            mid_x = (start_x + end_x) / 2
            
            self.canvas.coords(
                self.line,
                start_x, start_y,
                mid_x, start_y,
                mid_x, end_y,
                end_x, end_y
            )


class LogisticsPredictionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("물류이동 예측 시스템 - 비주얼 플로우 에디터")
        self.root.geometry("1400x800")
        
        # 변수 초기화
        self.nodes = {}
        self.connections = []
        self.node_counter = 0
        self.selected_node = None
        self.drag_data = {"x": 0, "y": 0}
        self.connecting = False
        self.connection_start = None
        
        self.create_ui()
        
    def create_ui(self):
        """UI 생성"""
        # 메인 프레임
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 왼쪽 패널 - 노드 팔레트
        left_panel = ttk.Frame(main_frame, width=250)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        left_panel.pack_propagate(False)
        
        # 제목
        title_label = ttk.Label(left_panel, text="노드 팔레트", 
                               font=('Arial', 14, 'bold'))
        title_label.pack(pady=10)
        
        # 노드 타입별 프레임
        self.create_node_palette(left_panel)
        
        # 중앙 캔버스
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 캔버스
        self.canvas = tk.Canvas(canvas_frame, bg='#2c3e50')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 그리드 배경 그리기
        self.draw_grid()
        
        # 캔버스 이벤트 바인딩
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Button-3>", self.on_right_click)  # 우클릭
        self.canvas.bind("<Configure>", self.on_canvas_resize)  # 크기 변경시 그리드 다시 그리기
        
        # 오른쪽 패널 - 속성/정보
        right_panel = ttk.Frame(main_frame, width=300)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        right_panel.pack_propagate(False)
        
        # 속성 패널
        self.create_properties_panel(right_panel)
        
        # 하단 툴바
        self.create_toolbar()
        
        # 초기 그리드 그리기 (창이 표시된 후)
        self.root.after(100, self.draw_grid)
        
    def create_node_palette(self, parent):
        """노드 팔레트 생성"""
        # 데이터 입력 섹션
        data_frame = ttk.LabelFrame(parent, text="데이터 입력", padding=10)
        data_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(data_frame, text="MCS 로그 데이터",
                  command=lambda: self.add_node('data', 'MCS 로그')).pack(fill=tk.X, pady=2)
        ttk.Button(data_frame, text="진동센서 데이터",
                  command=lambda: self.add_node('data', '진동센서')).pack(fill=tk.X, pady=2)
        
        # 전처리 섹션
        preprocess_frame = ttk.LabelFrame(parent, text="데이터 전처리", padding=10)
        preprocess_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(preprocess_frame, text="이상치 제거",
                  command=lambda: self.add_node('preprocess', '이상치 제거')).pack(fill=tk.X, pady=2)
        ttk.Button(preprocess_frame, text="시간별 분류",
                  command=lambda: self.add_node('preprocess', '시간별 분류')).pack(fill=tk.X, pady=2)
        ttk.Button(preprocess_frame, text="경로별 분류",
                  command=lambda: self.add_node('preprocess', '경로별 분류')).pack(fill=tk.X, pady=2)
        
        # 벡터 저장 섹션
        vector_frame = ttk.LabelFrame(parent, text="벡터 저장", padding=10)
        vector_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(vector_frame, text="RAG 벡터 저장",
                  command=lambda: self.add_node('vector', 'RAG 벡터')).pack(fill=tk.X, pady=2)
        
        # 모델 섹션
        model_frame = ttk.LabelFrame(parent, text="시계열 모델", padding=10)
        model_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(model_frame, text="LSTM 모델",
                  command=lambda: self.add_node('model', 'LSTM')).pack(fill=tk.X, pady=2)
        ttk.Button(model_frame, text="RNN 모델",
                  command=lambda: self.add_node('model', 'RNN')).pack(fill=tk.X, pady=2)
        ttk.Button(model_frame, text="ARIMA 모델",
                  command=lambda: self.add_node('model', 'ARIMA')).pack(fill=tk.X, pady=2)
        
        # 분석 섹션
        analysis_frame = ttk.LabelFrame(parent, text="분석", padding=10)
        analysis_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(analysis_frame, text="과거패턴 분석",
                  command=lambda: self.add_node('analysis', '패턴분석')).pack(fill=tk.X, pady=2)
        
        # 프롬프트 섹션
        prompt_frame = ttk.LabelFrame(parent, text="프롬프트", padding=10)
        prompt_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(prompt_frame, text="프롬프트 생성",
                  command=lambda: self.add_node('prompt', '프롬프트')).pack(fill=tk.X, pady=2)
        
        # LLM 섹션
        llm_frame = ttk.LabelFrame(parent, text="LLM", padding=10)
        llm_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(llm_frame, text="PHI-4 판단/추론",
                  command=lambda: self.add_node('llm', 'PHI-4')).pack(fill=tk.X, pady=2)
        
    def create_properties_panel(self, parent):
        """속성 패널 생성"""
        # 제목
        title_label = ttk.Label(parent, text="노드 속성", 
                               font=('Arial', 14, 'bold'))
        title_label.pack(pady=10)
        
        # 속성 표시 영역
        self.props_frame = ttk.Frame(parent)
        self.props_frame.pack(fill=tk.BOTH, expand=True, padx=10)
        
        # 시스템 정보
        info_frame = ttk.LabelFrame(parent, text="시스템 정보", padding=10)
        info_frame.pack(fill=tk.X, pady=10, padx=10)
        
        self.info_text = tk.Text(info_frame, height=8, width=30, 
                                font=('Arial', 9))
        self.info_text.pack(fill=tk.BOTH)
        self.update_system_info()
        
    def create_toolbar(self):
        """하단 툴바 생성"""
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        
        # 버튼들
        ttk.Button(toolbar, text="실행", 
                  command=self.run_pipeline).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="검증", 
                  command=self.validate_pipeline).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="저장", 
                  command=self.save_pipeline).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="불러오기", 
                  command=self.load_pipeline).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="초기화", 
                  command=self.clear_canvas).pack(side=tk.LEFT, padx=5)
        
        # 상태 표시
        self.status_label = ttk.Label(toolbar, text="준비됨", 
                                     font=('Arial', 10))
        self.status_label.pack(side=tk.RIGHT, padx=10)
        
    def draw_grid(self):
        """캔버스에 그리드 그리기"""
        # 기존 그리드 삭제
        self.canvas.delete("grid")
        
        # 그리드 간격
        grid_size = 20
        
        # 캔버스 크기 가져오기
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        
        if width <= 1 or height <= 1:
            return
            
        # 수직선 그리기
        for x in range(0, width, grid_size):
            self.canvas.create_line(x, 0, x, height, 
                                   fill='#34495e', width=1, tags="grid")
            
        # 수평선 그리기
        for y in range(0, height, grid_size):
            self.canvas.create_line(0, y, width, y, 
                                   fill='#34495e', width=1, tags="grid")
            
        # 그리드를 맨 아래로
        self.canvas.tag_lower("grid")
        
    def on_canvas_resize(self, event):
        """캔버스 크기 변경시 그리드 다시 그리기"""
        self.draw_grid()
        
    def add_node(self, node_type, name):
        """노드 추가"""
        # 캔버스 중앙 근처에 노드 생성
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            x = canvas_width // 2 + random.randint(-150, 150)
            y = canvas_height // 2 + random.randint(-100, 100)
        else:
            x = random.randint(300, 700)
            y = random.randint(100, 400)
        
        # 그리드에 맞춰 정렬
        x = (x // 20) * 20
        y = (y // 20) * 20
        
        node_id = f"node_{self.node_counter}"
        self.node_counter += 1
        
        node = Node(self.canvas, x, y, node_type, node_id, name)
        self.nodes[node_id] = node
        
        self.update_status(f"{name} 노드가 추가되었습니다")
        self.update_system_info()
        
    def on_canvas_click(self, event):
        """캔버스 클릭 이벤트"""
        # 클릭한 위치의 모든 아이템 가져오기
        clicked_items = self.canvas.find_overlapping(event.x-2, event.y-2, event.x+2, event.y+2)
        
        # 노드를 찾기 위해 태그 확인
        node_found = False
        for item in clicked_items:
            tags = self.canvas.gettags(item)
            
            for tag in tags:
                # 노드 클릭 - 드래그 시작
                if tag.startswith("node_") and not tag.startswith("node_text"):
                    node_id = tag
                    self.selected_node = node_id
                    self.drag_data["x"] = event.x
                    self.drag_data["y"] = event.y
                    self.drag_data["dragging"] = True
                    
                    # 선택된 노드 하이라이트
                    node = self.nodes[node_id]
                    self.canvas.itemconfig(node.rect, width=3)
                    
                    # 드래그 시작시 노드 투명도 조정
                    current_color = self.canvas.itemcget(node.rect, "fill")
                    self.drag_data["original_color"] = current_color
                    # 색상에 투명도 효과를 주기 위해 더 밝게
                    lighter_color = self.lighten_color(current_color)
                    self.canvas.itemconfig(node.rect, fill=lighter_color)
                    
                    # 캔버스에서 노드를 맨 위로 올림
                    self.canvas.tag_raise(f"node_{node_id}")
                    self.canvas.tag_raise(f"port_in_{node_id}")
                    self.canvas.tag_raise(f"port_out_{node_id}")
                    
                    # 마우스 커서 변경
                    self.canvas.config(cursor="fleur")
                    
                    node_found = True
                    self.show_node_properties(node_id)
                    break
                    
                # 출력 포트 클릭 - 연결 시작
                elif tag.startswith("port_out_"):
                    node_id = tag.replace("port_out_", "")
                    self.start_connection(node_id)
                    node_found = True
                    break
                    
                # 입력 포트 클릭 - 연결 완료
                elif tag.startswith("port_in_"):
                    if self.connecting:
                        node_id = tag.replace("port_in_", "")
                        self.end_connection(node_id)
                    node_found = True
                    break
                    
            if node_found:
                break
                
        # 빈 공간 클릭시 선택 해제
        if not node_found and not self.connecting:
            self.deselect_all_nodes()
                
    def on_canvas_drag(self, event):
        """캔버스 드래그 이벤트"""
        if self.selected_node and self.drag_data.get("dragging", False) and not self.connecting:
            # 이동 거리 계산
            dx = event.x - self.drag_data["x"]
            dy = event.y - self.drag_data["y"]
            
            # 노드 이동
            node = self.nodes[self.selected_node]
            node.move(dx, dy)
            
            # 연결선 실시간 업데이트
            for conn in self.connections:
                if conn.start_node.node_id == self.selected_node or conn.end_node.node_id == self.selected_node:
                    conn.update()
                    
            # 현재 위치 업데이트
            self.drag_data["x"] = event.x
            self.drag_data["y"] = event.y
            
            # 상태 표시
            self.update_status(f"노드 이동 중: {node.name} ({int(node.x)}, {int(node.y)})")
            
    def on_canvas_release(self, event):
        """캔버스 릴리즈 이벤트"""
        if self.drag_data.get("dragging", False):
            self.drag_data["dragging"] = False
            if self.selected_node:
                node = self.nodes[self.selected_node]
                
                # 그리드에 스냅
                grid_size = 20
                node.x = round(node.x / grid_size) * grid_size
                node.y = round(node.y / grid_size) * grid_size
                
                # 최종 위치로 이동
                self.canvas.coords(node.shadow, 
                                 node.x + 3, node.y + 3,
                                 node.x + node.width + 3, node.y + node.height + 3)
                self.canvas.coords(node.rect, 
                                 node.x, node.y,
                                 node.x + node.width, node.y + node.height)
                self.canvas.coords(node.text,
                                 node.x + node.width/2, node.y + 20)
                self.canvas.coords(node.type_text,
                                 node.x + node.width/2, node.y + 40)
                self.canvas.coords(node.status,
                                 node.x + node.width - 20, node.y + 10,
                                 node.x + node.width - 10, node.y + 20)
                self.canvas.coords(node.input_port,
                                 node.x - 10, node.y + node.height/2 - 5,
                                 node.x, node.y + node.height/2 + 5)
                self.canvas.coords(node.output_port,
                                 node.x + node.width, node.y + node.height/2 - 5,
                                 node.x + node.width + 10, node.y + node.height/2 + 5)
                
                # 원래 색상으로 복원
                if "original_color" in self.drag_data:
                    self.canvas.itemconfig(node.rect, fill=self.drag_data["original_color"])
                
                # 연결선 최종 업데이트
                for conn in self.connections:
                    if conn.start_node.node_id == self.selected_node or conn.end_node.node_id == self.selected_node:
                        conn.update()
                
                # 커서 복원
                self.canvas.config(cursor="")
                
                self.update_status(f"{node.name} 이동 완료")
                
    def lighten_color(self, color):
        """색상을 밝게 만들기"""
        # 간단한 구현 - 실제로는 더 정교한 색상 변환 필요
        color_map = {
            '#3498db': '#5faee3',  # 파란색
            '#e74c3c': '#ec7063',  # 빨간색
            '#f39c12': '#f5b041',  # 주황색
            '#27ae60': '#52be80',  # 초록색
            '#9b59b6': '#af7ac5',  # 보라색
            '#34495e': '#5d6d7e',  # 회색
            '#16a085': '#48c9b0'   # 청록색
        }
        return color_map.get(color, color)
                
    def deselect_all_nodes(self):
        """모든 노드 선택 해제"""
        for node_id, node in self.nodes.items():
            self.canvas.itemconfig(node.rect, width=2)
        self.selected_node = None
        
        # 속성 패널 초기화
        for widget in self.props_frame.winfo_children():
            widget.destroy()
        ttk.Label(self.props_frame, text="노드를 선택하세요", 
                 font=('Arial', 10)).pack(pady=20)
        
    def on_right_click(self, event):
        """우클릭 메뉴"""
        clicked_item = self.canvas.find_closest(event.x, event.y)[0]
        tags = self.canvas.gettags(clicked_item)
        
        for tag in tags:
            if tag.startswith("node_"):
                node_id = tag
                self.show_node_menu(event, node_id)
                return
                
    def show_node_menu(self, event, node_id):
        """노드 컨텍스트 메뉴"""
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="설정", 
                        command=lambda: self.configure_node(node_id))
        menu.add_command(label="삭제", 
                        command=lambda: self.delete_node(node_id))
        menu.add_separator()
        menu.add_command(label="정보", 
                        command=lambda: self.show_node_info(node_id))
        
        menu.post(event.x_root, event.y_root)
        
    def select_node(self, node_id):
        """노드 선택"""
        self.selected_node = node_id
        self.show_node_properties(node_id)
        
    def show_node_properties(self, node_id):
        """노드 속성 표시"""
        # 기존 위젯 제거
        for widget in self.props_frame.winfo_children():
            widget.destroy()
            
        node = self.nodes[node_id]
        
        # 노드 정보
        ttk.Label(self.props_frame, text=f"노드: {node.name}",
                 font=('Arial', 12, 'bold')).pack(pady=5)
        ttk.Label(self.props_frame, text=f"타입: {node.node_type}").pack()
        ttk.Label(self.props_frame, text=f"ID: {node_id}").pack()
        
        # 설정 상태
        status_text = "설정됨" if node.is_configured else "미설정"
        status_color = "green" if node.is_configured else "red"
        status_label = ttk.Label(self.props_frame, text=f"상태: {status_text}",
                                foreground=status_color)
        status_label.pack(pady=5)
        
        # 설정 버튼
        ttk.Button(self.props_frame, text="노드 설정",
                  command=lambda: self.configure_node(node_id)).pack(pady=10)
        
    def configure_node(self, node_id):
        """노드 설정 대화상자"""
        node = self.nodes[node_id]
        
        config_window = tk.Toplevel(self.root)
        config_window.title(f"{node.name} 설정")
        config_window.geometry("400x500")
        
        # 노드 타입별 설정
        if node.node_type == 'data':
            self.configure_data_node(config_window, node)
        elif node.node_type == 'preprocess':
            self.configure_preprocess_node(config_window, node)
        elif node.node_type == 'model':
            self.configure_model_node(config_window, node)
        elif node.node_type == 'vector':
            self.configure_vector_node(config_window, node)
        elif node.node_type == 'analysis':
            self.configure_analysis_node(config_window, node)
        elif node.node_type == 'prompt':
            self.configure_prompt_node(config_window, node)
        elif node.node_type == 'llm':
            self.configure_llm_node(config_window, node)
            
    def configure_data_node(self, window, node):
        """데이터 노드 설정"""
        ttk.Label(window, text="데이터 소스 설정", 
                 font=('Arial', 12, 'bold')).pack(pady=10)
        
        # 파일 경로
        ttk.Label(window, text="파일 경로:").pack()
        path_entry = ttk.Entry(window, width=40)
        path_entry.pack(pady=5)
        
        # 데이터 형식
        ttk.Label(window, text="데이터 형식:").pack()
        format_var = tk.StringVar(value="CSV")
        formats = ["CSV", "JSON", "Excel", "Database"]
        format_menu = ttk.OptionMenu(window, format_var, format_var.get(), *formats)
        format_menu.pack(pady=5)
        
        # 샘플링 설정
        ttk.Label(window, text="샘플링 간격 (초):").pack()
        sampling_entry = ttk.Entry(window, width=20)
        sampling_entry.insert(0, "60")
        sampling_entry.pack(pady=5)
        
        def save_config():
            node.config = {
                'path': path_entry.get(),
                'format': format_var.get(),
                'sampling': sampling_entry.get()
            }
            node.set_configured(True)
            self.update_status(f"{node.name} 설정 완료")
            window.destroy()
            
        ttk.Button(window, text="저장", command=save_config).pack(pady=20)
        
    def configure_preprocess_node(self, window, node):
        """전처리 노드 설정"""
        ttk.Label(window, text="전처리 설정", 
                 font=('Arial', 12, 'bold')).pack(pady=10)
        
        if "이상치" in node.name:
            # 이상치 제거 설정
            ttk.Label(window, text="이상치 탐지 방법:").pack()
            method_var = tk.StringVar(value="IQR")
            methods = ["IQR", "Z-Score", "Isolation Forest", "DBSCAN"]
            ttk.OptionMenu(window, method_var, method_var.get(), *methods).pack(pady=5)
            
            ttk.Label(window, text="임계값:").pack()
            threshold_entry = ttk.Entry(window, width=20)
            threshold_entry.insert(0, "1.5")
            threshold_entry.pack(pady=5)
            
        elif "시간별" in node.name:
            # 시간별 분류 설정
            ttk.Label(window, text="시간 단위:").pack()
            time_var = tk.StringVar(value="1시간")
            times = ["30분", "1시간", "2시간", "4시간", "6시간", "12시간", "24시간"]
            ttk.OptionMenu(window, time_var, time_var.get(), *times).pack(pady=5)
            
        elif "경로별" in node.name:
            # 경로별 분류 설정
            ttk.Label(window, text="경로 그룹화 기준:").pack()
            route_var = tk.StringVar(value="출발지-도착지")
            routes = ["출발지-도착지", "주요 경유지", "운송 수단", "거리별"]
            ttk.OptionMenu(window, route_var, route_var.get(), *routes).pack(pady=5)
            
        def save_config():
            node.set_configured(True)
            self.update_status(f"{node.name} 설정 완료")
            window.destroy()
            
        ttk.Button(window, text="저장", command=save_config).pack(pady=20)
        
    def configure_model_node(self, window, node):
        """모델 노드 설정"""
        ttk.Label(window, text="모델 설정", 
                 font=('Arial', 12, 'bold')).pack(pady=10)
        
        # 예측 기간
        ttk.Label(window, text="예측 기간:").pack()
        period_var = tk.StringVar(value="24시간")
        periods = ["6시간", "12시간", "24시간", "48시간", "72시간", "1주일"]
        ttk.OptionMenu(window, period_var, period_var.get(), *periods).pack(pady=5)
        
        if "LSTM" in node.name or "RNN" in node.name:
            # 딥러닝 모델 설정
            ttk.Label(window, text="은닉층 수:").pack()
            layers_entry = ttk.Entry(window, width=20)
            layers_entry.insert(0, "3")
            layers_entry.pack(pady=5)
            
            ttk.Label(window, text="유닛 수:").pack()
            units_entry = ttk.Entry(window, width=20)
            units_entry.insert(0, "128")
            units_entry.pack(pady=5)
            
            ttk.Label(window, text="배치 크기:").pack()
            batch_entry = ttk.Entry(window, width=20)
            batch_entry.insert(0, "32")
            batch_entry.pack(pady=5)
            
        elif "ARIMA" in node.name:
            # ARIMA 모델 설정
            ttk.Label(window, text="p (자기회귀):").pack()
            p_entry = ttk.Entry(window, width=20)
            p_entry.insert(0, "2")
            p_entry.pack(pady=5)
            
            ttk.Label(window, text="d (차분):").pack()
            d_entry = ttk.Entry(window, width=20)
            d_entry.insert(0, "1")
            d_entry.pack(pady=5)
            
            ttk.Label(window, text="q (이동평균):").pack()
            q_entry = ttk.Entry(window, width=20)
            q_entry.insert(0, "2")
            q_entry.pack(pady=5)
            
        def save_config():
            node.set_configured(True)
            self.update_status(f"{node.name} 설정 완료")
            window.destroy()
            
        ttk.Button(window, text="저장", command=save_config).pack(pady=20)
        
    def configure_vector_node(self, window, node):
        """벡터 저장 노드 설정"""
        ttk.Label(window, text="RAG 벡터 저장 설정", 
                 font=('Arial', 12, 'bold')).pack(pady=10)
        
        # 임베딩 모델
        ttk.Label(window, text="임베딩 모델:").pack()
        embed_var = tk.StringVar(value="OpenAI")
        embeds = ["OpenAI", "Sentence-BERT", "Custom", "Multilingual"]
        ttk.OptionMenu(window, embed_var, embed_var.get(), *embeds).pack(pady=5)
        
        # 벡터 차원
        ttk.Label(window, text="벡터 차원:").pack()
        dim_entry = ttk.Entry(window, width=20)
        dim_entry.insert(0, "768")
        dim_entry.pack(pady=5)
        
        # 저장소
        ttk.Label(window, text="벡터 저장소:").pack()
        store_var = tk.StringVar(value="ChromaDB")
        stores = ["ChromaDB", "Pinecone", "Weaviate", "FAISS"]
        ttk.OptionMenu(window, store_var, store_var.get(), *stores).pack(pady=5)
        
        def save_config():
            node.set_configured(True)
            self.update_status(f"{node.name} 설정 완료")
            window.destroy()
            
        ttk.Button(window, text="저장", command=save_config).pack(pady=20)
        
    def configure_analysis_node(self, window, node):
        """분석 노드 설정"""
        ttk.Label(window, text="패턴 분석 설정", 
                 font=('Arial', 12, 'bold')).pack(pady=10)
        
        # 분석 기간
        ttk.Label(window, text="분석 기간:").pack()
        period_var = tk.StringVar(value="30일")
        periods = ["7일", "14일", "30일", "90일", "180일", "1년"]
        ttk.OptionMenu(window, period_var, period_var.get(), *periods).pack(pady=5)
        
        # 패턴 유형
        ttk.Label(window, text="패턴 유형:").pack()
        pattern_frame = ttk.Frame(window)
        pattern_frame.pack(pady=5)
        
        patterns = ["계절성", "주기성", "트렌드", "이상패턴", "피크시간"]
        pattern_vars = {}
        for pattern in patterns:
            var = tk.BooleanVar(value=True)
            pattern_vars[pattern] = var
            ttk.Checkbutton(pattern_frame, text=pattern, variable=var).pack(anchor='w')
            
        def save_config():
            node.set_configured(True)
            self.update_status(f"{node.name} 설정 완료")
            window.destroy()
            
        ttk.Button(window, text="저장", command=save_config).pack(pady=20)
        
    def configure_prompt_node(self, window, node):
        """프롬프트 노드 설정"""
        ttk.Label(window, text="프롬프트 생성 설정", 
                 font=('Arial', 12, 'bold')).pack(pady=10)
        
        # 프롬프트 템플릿
        ttk.Label(window, text="프롬프트 템플릿:").pack()
        template_text = tk.Text(window, height=10, width=45)
        template_text.pack(pady=5)
        
        # 기본 템플릿
        default_template = """다음 물류 데이터를 분석하여 예측해주세요:
- 현재 상황: {current_status}
- 과거 패턴: {past_patterns}
- 시계열 예측: {time_series_prediction}
- 특이사항: {anomalies}

24시간 이내의 물류 이동을 예측하고,
주의해야 할 리스크를 분석해주세요."""
        
        template_text.insert('1.0', default_template)
        
        # 컨텍스트 포함 옵션
        ttk.Label(window, text="포함할 컨텍스트:").pack()
        context_frame = ttk.Frame(window)
        context_frame.pack(pady=5)
        
        contexts = ["날씨 정보", "교통 상황", "과거 지연 이력", "특별 이벤트"]
        context_vars = {}
        for context in contexts:
            var = tk.BooleanVar(value=True)
            context_vars[context] = var
            ttk.Checkbutton(context_frame, text=context, variable=var).pack(anchor='w')
            
        def save_config():
            node.config = {
                'template': template_text.get('1.0', 'end-1c'),
                'contexts': {k: v.get() for k, v in context_vars.items()}
            }
            node.set_configured(True)
            self.update_status(f"{node.name} 설정 완료")
            window.destroy()
            
        ttk.Button(window, text="저장", command=save_config).pack(pady=20)
        
    def configure_llm_node(self, window, node):
        """LLM 노드 설정"""
        ttk.Label(window, text="PHI-4 LLM 설정", 
                 font=('Arial', 12, 'bold')).pack(pady=10)
        
        # 모델 설정
        ttk.Label(window, text="Temperature:").pack()
        temp_scale = ttk.Scale(window, from_=0, to=1, orient='horizontal', length=200)
        temp_scale.set(0.7)
        temp_scale.pack(pady=5)
        
        ttk.Label(window, text="Max Tokens:").pack()
        tokens_entry = ttk.Entry(window, width=20)
        tokens_entry.insert(0, "2048")
        tokens_entry.pack(pady=5)
        
        # 추론 모드
        ttk.Label(window, text="추론 모드:").pack()
        mode_var = tk.StringVar(value="종합분석")
        modes = ["종합분석", "리스크 평가", "최적화 제안", "이상탐지"]
        ttk.OptionMenu(window, mode_var, mode_var.get(), *modes).pack(pady=5)
        
        # 출력 형식
        ttk.Label(window, text="출력 형식:").pack()
        format_var = tk.StringVar(value="구조화된 JSON")
        formats = ["구조화된 JSON", "자연어 설명", "대시보드 데이터", "리포트"]
        ttk.OptionMenu(window, format_var, format_var.get(), *formats).pack(pady=5)
        
        def save_config():
            node.config = {
                'temperature': temp_scale.get(),
                'max_tokens': tokens_entry.get(),
                'mode': mode_var.get(),
                'format': format_var.get()
            }
            node.set_configured(True)
            self.update_status(f"{node.name} 설정 완료")
            window.destroy()
            
        ttk.Button(window, text="저장", command=save_config).pack(pady=20)
        
    def start_connection(self, node_id):
        """연결 시작"""
        self.connecting = True
        self.connection_start = node_id
        self.update_status("연결할 노드를 선택하세요...")
        
    def end_connection(self, end_node_id):
        """연결 완료"""
        if self.connection_start and self.connection_start != end_node_id:
            start_node = self.nodes[self.connection_start]
            end_node = self.nodes[end_node_id]
            
            # 연결 생성
            conn = Connection(self.canvas, start_node, end_node)
            self.connections.append(conn)
            
            # 노드에 연결 정보 저장
            start_node.outputs.append(end_node_id)
            end_node.inputs.append(self.connection_start)
            
            self.update_status(f"{start_node.name} → {end_node.name} 연결됨")
            
        self.connecting = False
        self.connection_start = None
        
    def delete_node(self, node_id):
        """노드 삭제"""
        if node_id in self.nodes:
            # 관련 연결 제거
            connections_to_remove = []
            for i, conn in enumerate(self.connections):
                if conn.start_node.node_id == node_id or conn.end_node.node_id == node_id:
                    self.canvas.delete(conn.line)
                    connections_to_remove.append(i)
            
            # 연결 리스트에서 제거 (역순으로)
            for i in reversed(connections_to_remove):
                del self.connections[i]
            
            # 캔버스에서 노드 제거
            self.canvas.delete(f"node_{node_id}")
            self.canvas.delete(f"port_in_{node_id}")
            self.canvas.delete(f"port_out_{node_id}")
            
            # 노드 객체 제거
            del self.nodes[node_id]
            
            # 선택 상태 초기화
            if self.selected_node == node_id:
                self.selected_node = None
                self.deselect_all_nodes()
            
            self.update_status(f"노드가 삭제되었습니다")
            self.update_system_info()
            
    def clear_canvas(self):
        """캔버스 초기화"""
        if messagebox.askyesno("확인", "모든 노드를 삭제하시겠습니까?"):
            # 노드와 연결선만 삭제 (그리드는 유지)
            for node_id in list(self.nodes.keys()):
                self.canvas.delete(f"node_{node_id}")
                self.canvas.delete(f"port_in_{node_id}")
                self.canvas.delete(f"port_out_{node_id}")
            
            self.canvas.delete("connection")
            
            self.nodes.clear()
            self.connections.clear()
            self.node_counter = 0
            self.selected_node = None
            
            self.update_status("캔버스가 초기화되었습니다")
            self.update_system_info()
            
    def validate_pipeline(self):
        """파이프라인 검증"""
        errors = []
        warnings = []
        
        # 노드 존재 확인
        if not self.nodes:
            errors.append("노드가 없습니다")
            
        # 데이터 입력 노드 확인
        data_nodes = [n for n in self.nodes.values() if n.node_type == 'data']
        if not data_nodes:
            errors.append("데이터 입력 노드가 필요합니다")
            
        # LLM 노드 확인
        llm_nodes = [n for n in self.nodes.values() if n.node_type == 'llm']
        if not llm_nodes:
            errors.append("LLM 노드가 필요합니다")
            
        # 미설정 노드 확인
        unconfigured = [n.name for n in self.nodes.values() if not n.is_configured]
        if unconfigured:
            warnings.append(f"미설정 노드: {', '.join(unconfigured)}")
            
        # 연결 확인
        for node in self.nodes.values():
            if node.node_type != 'data' and not node.inputs:
                warnings.append(f"{node.name}에 입력이 없습니다")
                
        # 결과 표시
        if errors or warnings:
            msg = ""
            if errors:
                msg += "오류:\n" + "\n".join(f"- {e}" for e in errors) + "\n\n"
            if warnings:
                msg += "경고:\n" + "\n".join(f"- {w}" for w in warnings)
                
            messagebox.showwarning("검증 결과", msg)
        else:
            messagebox.showinfo("검증 결과", "파이프라인이 유효합니다!")
            
    def run_pipeline(self):
        """파이프라인 실행"""
        # 검증
        self.validate_pipeline()
        
        # 실행 시뮬레이션
        self.update_status("파이프라인 실행 중...")
        
        # 실제 구현시 여기서 각 노드를 순차적으로 실행
        execution_order = self.get_execution_order()
        
        result_window = tk.Toplevel(self.root)
        result_window.title("실행 결과")
        result_window.geometry("600x400")
        
        result_text = tk.Text(result_window, wrap=tk.WORD)
        result_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        result_text.insert('end', "=== 물류이동 예측 시스템 실행 결과 ===\n\n")
        result_text.insert('end', f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 실행 순서 표시
        result_text.insert('end', "실행 순서:\n")
        for i, node_id in enumerate(execution_order, 1):
            node = self.nodes[node_id]
            result_text.insert('end', f"{i}. {node.name} ({node.node_type})\n")
            
        # 예측 결과 (시뮬레이션)
        result_text.insert('end', "\n=== 예측 결과 ===\n")
        result_text.insert('end', "향후 24시간 물류 이동 예측:\n")
        result_text.insert('end', "- 예상 물동량: 1,234톤 (±5%)\n")
        result_text.insert('end', "- 주요 경로: 서울→부산 (45%), 인천→광주 (30%)\n")
        result_text.insert('end', "- 병목 구간: 경부고속도로 (14:00-18:00)\n")
        result_text.insert('end', "- 지연 위험도: 중간 (날씨 영향)\n\n")
        
        result_text.insert('end', "권장 사항:\n")
        result_text.insert('end', "1. 오후 2-6시 경부고속도로 우회 경로 활용\n")
        result_text.insert('end', "2. 긴급 화물은 오전 배송 권장\n")
        result_text.insert('end', "3. 예비 차량 20% 추가 배치 필요\n")
        
        self.update_status("파이프라인 실행 완료")
        
    def get_execution_order(self):
        """노드 실행 순서 결정 (위상 정렬)"""
        # 간단한 구현 - 실제로는 위상 정렬 알고리즘 사용
        order = []
        
        # 데이터 노드부터 시작
        for node_id, node in self.nodes.items():
            if node.node_type == 'data':
                order.append(node_id)
                
        # 나머지 노드들
        for node_id, node in self.nodes.items():
            if node_id not in order:
                order.append(node_id)
                
        return order
        
    def save_pipeline(self):
        """파이프라인 저장"""
        data = {
            'nodes': {},
            'connections': []
        }
        
        # 노드 정보 저장
        for node_id, node in self.nodes.items():
            data['nodes'][node_id] = {
                'x': node.x,
                'y': node.y,
                'type': node.node_type,
                'name': node.name,
                'config': node.config,
                'configured': node.is_configured
            }
            
        # 연결 정보 저장
        for conn in self.connections:
            data['connections'].append({
                'start': conn.start_node.node_id,
                'end': conn.end_node.node_id
            })
            
        # 파일로 저장
        from tkinter import filedialog
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.update_status(f"파이프라인이 저장되었습니다: {filename}")
            
    def load_pipeline(self):
        """파이프라인 불러오기"""
        from tkinter import filedialog
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 기존 캔버스 초기화
            self.clear_canvas()
            
            # 그리드 다시 그리기
            self.draw_grid()
            
            # 노드 생성
            for node_id, node_data in data['nodes'].items():
                node = Node(
                    self.canvas,
                    node_data['x'],
                    node_data['y'],
                    node_data['type'],
                    node_id,
                    node_data['name']
                )
                node.config = node_data.get('config', {})
                node.set_configured(node_data.get('configured', False))
                self.nodes[node_id] = node
                
            # 연결 생성
            for conn_data in data['connections']:
                start_node = self.nodes[conn_data['start']]
                end_node = self.nodes[conn_data['end']]
                conn = Connection(self.canvas, start_node, end_node)
                self.connections.append(conn)
                
            self.update_status(f"파이프라인을 불러왔습니다: {filename}")
            self.update_system_info()
            
    def show_node_info(self, node_id):
        """노드 정보 표시"""
        node = self.nodes[node_id]
        info = f"""
노드 정보
---------
이름: {node.name}
타입: {node.node_type}
ID: {node_id}
설정 상태: {'완료' if node.is_configured else '미완료'}
입력 연결: {len(node.inputs)}개
출력 연결: {len(node.outputs)}개

설정 내용:
{json.dumps(node.config, indent=2, ensure_ascii=False) if node.config else '없음'}
"""
        messagebox.showinfo("노드 정보", info)
        
    def update_status(self, message):
        """상태 메시지 업데이트"""
        self.status_label.config(text=message)
        
    def update_system_info(self):
        """시스템 정보 업데이트"""
        self.info_text.delete('1.0', tk.END)
        
        info = f"""노드 수: {len(self.nodes)}
연결 수: {len(self.connections)}

노드 타입별 개수:
"""
        # 타입별 집계
        type_count = {}
        for node in self.nodes.values():
            type_count[node.node_type] = type_count.get(node.node_type, 0) + 1
            
        for node_type, count in type_count.items():
            info += f"- {node_type}: {count}개\n"
            
        self.info_text.insert('1.0', info)


def main():
    root = tk.Tk()
    app = LogisticsPredictionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()