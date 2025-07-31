import sys
import json
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *


class NodeType(Enum):
    DATA = "data"
    PREPROCESS = "preprocess"
    VECTOR = "vector"
    MODEL = "model"
    ANALYSIS = "analysis"
    PROMPT = "prompt"
    LLM = "llm"


@dataclass
class NodeConfig:
    """노드 설정 데이터 클래스"""
    node_type: NodeType
    name: str
    color: str
    inputs: int = 1
    outputs: int = 1


# 노드 타입별 설정
NODE_CONFIGS = {
    NodeType.DATA: NodeConfig(NodeType.DATA, "데이터 입력", "#3498db", 0, 1),
    NodeType.PREPROCESS: NodeConfig(NodeType.PREPROCESS, "전처리", "#e74c3c", 1, 1),
    NodeType.VECTOR: NodeConfig(NodeType.VECTOR, "벡터 저장", "#f39c12", 1, 1),
    NodeType.MODEL: NodeConfig(NodeType.MODEL, "모델", "#27ae60", 1, 1),
    NodeType.ANALYSIS: NodeConfig(NodeType.ANALYSIS, "분석", "#9b59b6", 1, 1),
    NodeType.PROMPT: NodeConfig(NodeType.PROMPT, "프롬프트", "#34495e", 2, 1),
    NodeType.LLM: NodeConfig(NodeType.LLM, "LLM", "#16a085", 1, 1),
}


class Port(QGraphicsEllipseItem):
    """노드의 입출력 포트"""
    def __init__(self, is_output=True, parent=None):
        super().__init__(-6, -6, 12, 12, parent)
        self.is_output = is_output
        self.connections = []
        self.node = parent
        
        # 포트 스타일
        self.default_color = QColor("#00CED1") if is_output else QColor("#FFD700")
        self.hover_color = QColor("#00BFFF") if is_output else QColor("#FFA500")
        self.highlight_color = QColor("#FF6347")  # 스냅 하이라이트용
        
        self.setBrush(QBrush(self.default_color))
        self.setPen(QPen(QColor("#FFFFFF"), 2))
        self.setAcceptHoverEvents(True)
        self.setZValue(10)
        
    def hoverEnterEvent(self, event):
        self.setBrush(QBrush(self.hover_color))
        super().hoverEnterEvent(event)
        
    def hoverLeaveEvent(self, event):
        self.setBrush(QBrush(self.default_color))
        super().hoverLeaveEvent(event)
        
    def highlight(self, on=True):
        """스냅 하이라이트"""
        if on:
            self.setBrush(QBrush(self.highlight_color))
            self.setPen(QPen(QColor("#FFFFFF"), 3))
        else:
            self.setBrush(QBrush(self.default_color))
            self.setPen(QPen(QColor("#FFFFFF"), 2))
        
    def get_center(self):
        """포트 중심 좌표 반환 (씬 좌표)"""
        return self.scenePos() + self.rect().center()
        
    def can_connect_to(self, other_port):
        """다른 포트와 연결 가능한지 확인"""
        if not other_port or other_port == self:
            return False
        if self.is_output == other_port.is_output:
            return False
        if self.node == other_port.node:
            return False
        # 이미 연결되어 있는지 확인
        for conn in self.connections:
            if (conn.start_port == other_port or conn.end_port == other_port):
                return False
        return True


class Connection(QGraphicsPathItem):
    """노드 간 연결선"""
    def __init__(self, start_port=None, end_port=None):
        super().__init__()
        self.start_port = start_port
        self.end_port = end_port
        self.temp_end_pos = None
        
        # 연결선 스타일
        self.default_pen = QPen(QColor("#3498db"), 3)
        self.hover_pen = QPen(QColor("#5dade2"), 4)
        self.selected_pen = QPen(QColor("#e74c3c"), 4)
        
        # 꺾은선을 부드럽게 보이도록 조인 스타일 설정
        self.default_pen.setCapStyle(Qt.RoundCap)
        self.default_pen.setJoinStyle(Qt.RoundJoin)
        self.hover_pen.setCapStyle(Qt.RoundCap)
        self.hover_pen.setJoinStyle(Qt.RoundJoin)
        self.selected_pen.setCapStyle(Qt.RoundCap)
        self.selected_pen.setJoinStyle(Qt.RoundJoin)
        
        self.setPen(self.default_pen)
        self.setZValue(-1)
        
        # 선택 및 호버 가능
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        
        # 화살표
        self.arrow = QGraphicsPolygonItem()
        self.arrow.setBrush(QBrush(QColor("#3498db")))
        self.arrow.setPen(QPen(Qt.NoPen))
        self.arrow.setZValue(-1)
        
        if start_port and end_port:
            start_port.connections.append(self)
            end_port.connections.append(self)
            self.update_path()
            
    def hoverEnterEvent(self, event):
        """마우스 호버 시"""
        if not self.isSelected():
            self.setPen(self.hover_pen)
            self.arrow.setBrush(QBrush(QColor("#5dade2")))
        super().hoverEnterEvent(event)
        
    def hoverLeaveEvent(self, event):
        """마우스 호버 해제"""
        if not self.isSelected():
            self.setPen(self.default_pen)
            self.arrow.setBrush(QBrush(QColor("#3498db")))
        super().hoverLeaveEvent(event)
        
    def itemChange(self, change, value):
        """선택 상태 변경"""
        if change == QGraphicsItem.ItemSelectedChange:
            if value:
                self.setPen(self.selected_pen)
                self.arrow.setBrush(QBrush(QColor("#e74c3c")))
            else:
                self.setPen(self.default_pen)
                self.arrow.setBrush(QBrush(QColor("#3498db")))
        return super().itemChange(change, value)
        
    def update_path(self, end_pos=None):
        """연결선 경로 업데이트 (노드 회피 로직 포함)"""
        if not self.start_port:
            return
            
        start = self.start_port.get_center()
        
        if end_pos:
            end = end_pos
        elif self.end_port:
            end = self.end_port.get_center()
        else:
            return
            
        # 스마트 경로 계산
        path = self.calculate_smart_path(start, end)
        self.setPath(path)
        
        # 화살표 업데이트
        if self.end_port or end_pos:
            self.update_arrow(path)
            
    def calculate_smart_path(self, start, end):
        """노드를 피해가는 직각 경로 계산"""
        path = QPainterPath()
        path.moveTo(start)
        
        dx = end.x() - start.x()
        dy = end.y() - start.y()
        
        # 오프셋 거리
        offset = 50
        
        # 거리가 가까우면 그냥 직선 연결
        distance = math.sqrt(dx * dx + dy * dy)
        if distance < 150:
            path.lineTo(end)
            return path
            
        # Case 1: 정방향 연결 (왼쪽에서 오른쪽으로)
        if dx > offset:
            # 중간 지점에서 꺾기
            mid_x = start.x() + dx / 2
            path.lineTo(mid_x, start.y())
            path.lineTo(mid_x, end.y())
            path.lineTo(end)
            
        # Case 2: 역방향 연결 (오른쪽에서 왼쪽으로)
        else:
            # S자 연결
            mid_y = (start.y() + end.y()) / 2
            path.lineTo(start.x() + offset, start.y())
            path.lineTo(start.x() + offset, mid_y)
            path.lineTo(end.x() - offset, mid_y)
            path.lineTo(end.x() - offset, end.y())
            path.lineTo(end)
     
        
        return path
        
    def update_arrow(self, path):
        """화살표 업데이트"""
        if path.length() == 0:
            return
            
        # 경로의 마지막 선분에서 방향 계산
        # 끝점과 그 직전 점을 찾기
        point_count = path.elementCount()
        if point_count < 2:
            return
            
        # 마지막 두 점 가져오기
        last_element = path.elementAt(point_count - 1)
        second_last_element = path.elementAt(point_count - 2)
        
        point2 = QPointF(last_element.x, last_element.y)
        point1 = QPointF(second_last_element.x, second_last_element.y)
        
        angle = math.atan2(point2.y() - point1.y(), point2.x() - point1.x())
        
        # 화살표 폴리곤
        arrow_length = 12
        arrow_angle = math.radians(25)
        
        p1 = point2
        p2 = point2 - QPointF(
            arrow_length * math.cos(angle - arrow_angle),
            arrow_length * math.sin(angle - arrow_angle)
        )
        p3 = point2 - QPointF(
            arrow_length * math.cos(angle + arrow_angle),
            arrow_length * math.sin(angle + arrow_angle)
        )
        
        self.arrow.setPolygon(QPolygonF([p1, p2, p3]))
        
        if self.arrow.scene() != self.scene():
            if self.scene():
                self.scene().addItem(self.arrow)
                
    def shape(self):
        """클릭 영역을 넓게 설정"""
        stroker = QPainterPathStroker()
        stroker.setWidth(20)
        return stroker.createStroke(self.path())
        
    def remove(self):
        """연결 제거"""
        # 포트에서 연결 제거
        if self.start_port and self in self.start_port.connections:
            self.start_port.connections.remove(self)
        if self.end_port and self in self.end_port.connections:
            self.end_port.connections.remove(self)
            
        # 씬에서 제거
        if self.scene():
            self.scene().removeItem(self.arrow)
            self.scene().removeItem(self)


class Node(QGraphicsRectItem):
    """노드 클래스"""
    def __init__(self, node_type: NodeType, name: str, x=0, y=0):
        super().__init__(0, 0, 200, 100)
        
        self.node_id = id(self)
        self.node_type = node_type
        self.name = name
        self.config = NODE_CONFIGS[node_type]
        self.is_configured = False
        self.settings = {}
        
        # 노드 스타일
        self.setPos(x, y)
        self.setBrush(QBrush(QColor(self.config.color)))
        self.setPen(QPen(QColor("#FFFFFF"), 2))
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setCursor(Qt.PointingHandCursor)
        self.setAcceptHoverEvents(True)
        self.setZValue(1)
        
        # 그림자 효과
        shadow = QGraphicsDropShadowEffect()
        shadow.setOffset(3, 3)
        shadow.setBlurRadius(10)
        shadow.setColor(QColor(0, 0, 0, 100))
        self.setGraphicsEffect(shadow)
        
        # 텍스트 레이블
        self.title_text = QGraphicsTextItem(self.name, self)
        self.title_text.setDefaultTextColor(Qt.white)
        font = QFont("Arial", 11, QFont.Bold)
        self.title_text.setFont(font)
        self.title_text.setPos(10, 5)
        
        self.type_text = QGraphicsTextItem(f"[{node_type.value}]", self)
        self.type_text.setDefaultTextColor(QColor("#ecf0f1"))
        self.type_text.setFont(QFont("Arial", 9))
        self.type_text.setPos(10, 30)
        
        # 상태 표시
        self.status_indicator = QGraphicsEllipseItem(170, 10, 20, 20, self)
        self.update_status()
        
        # 포트 생성
        self.input_ports = []
        self.output_ports = []
        self.create_ports()
        
    def create_ports(self):
        """입출력 포트 생성"""
        # 입력 포트
        for i in range(self.config.inputs):
            y_pos = 50 + (i * 30) if self.config.inputs > 1 else 50
            port = Port(is_output=False, parent=self)
            port.setPos(0, y_pos)
            self.input_ports.append(port)
            
        # 출력 포트
        for i in range(self.config.outputs):
            y_pos = 50 + (i * 30) if self.config.outputs > 1 else 50
            port = Port(is_output=True, parent=self)
            port.setPos(200, y_pos)
            self.output_ports.append(port)
            
    def update_status(self):
        """상태 표시 업데이트"""
        color = QColor("#27ae60") if self.is_configured else QColor("#e74c3c")
        self.status_indicator.setBrush(QBrush(color))
        self.status_indicator.setPen(QPen(Qt.white, 2))
        
    def hoverEnterEvent(self, event):
        """마우스 호버 시 하이라이트"""
        self.setPen(QPen(QColor("#FFD700"), 3))
        super().hoverEnterEvent(event)
        
    def hoverLeaveEvent(self, event):
        """마우스 호버 해제"""
        self.setPen(QPen(QColor("#FFFFFF"), 2))
        super().hoverLeaveEvent(event)
        
    def itemChange(self, change, value):
        """아이템 변경 시 연결선 업데이트"""
        if change == QGraphicsItem.ItemPositionHasChanged:
            # 연결된 모든 연결선 업데이트
            for port in self.input_ports + self.output_ports:
                for connection in port.connections:
                    connection.update_path()
                    
        return super().itemChange(change, value)
        
    def mouseDoubleClickEvent(self, event):
        """더블클릭 시 설정 창 열기"""
        if hasattr(self.scene(), 'parent'):
            self.scene().parent().configure_node(self)
        super().mouseDoubleClickEvent(event)


class NodeScene(QGraphicsScene):
    """노드 에디터 씬"""
    def __init__(self):
        super().__init__()
        self.setSceneRect(-2000, -2000, 4000, 4000)
        
        # 배경 색상
        self.setBackgroundBrush(QBrush(QColor("#2c3e50")))
        
        # 그리드
        self.grid_size = 20
        self.grid_visible = True
        
        # 연결 관련
        self.current_connection = None
        self.start_port = None
        self.highlighted_port = None
        
    def drawBackground(self, painter, rect):
        """배경 그리기 (그리드 포함)"""
        super().drawBackground(painter, rect)
        
        if not self.grid_visible:
            return
            
        # 그리드 그리기
        painter.setPen(QPen(QColor("#34495e"), 1, Qt.SolidLine))
        
        # 그리드 범위 계산
        left = int(rect.left()) - (int(rect.left()) % self.grid_size)
        top = int(rect.top()) - (int(rect.top()) % self.grid_size)
        
        # 수직선
        for x in range(left, int(rect.right()), self.grid_size):
            painter.drawLine(x, rect.top(), x, rect.bottom())
            
        # 수평선
        for y in range(top, int(rect.bottom()), self.grid_size):
            painter.drawLine(rect.left(), y, rect.right(), y)
            
    def find_port_at(self, pos):
        """주어진 위치의 포트 찾기"""
        items = self.items(pos)
        for item in items:
            if isinstance(item, Port):
                return item
        return None
        
    def find_nearest_port(self, pos, port_type=None, max_distance=50):
        """주어진 위치에서 가장 가까운 포트 찾기"""
        nearest_port = None
        min_distance = max_distance
        
        for item in self.items():
            if isinstance(item, Port):
                # 포트 타입 필터링
                if port_type is not None and item.is_output != (port_type == "output"):
                    continue
                    
                # 현재 연결 중인 포트와 연결 가능한지 확인
                if self.start_port and not self.start_port.can_connect_to(item):
                    continue
                    
                port_center = item.get_center()
                distance = ((pos - port_center).x() ** 2 + (pos - port_center).y() ** 2) ** 0.5
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_port = item
                    
        return nearest_port
        
    def mousePressEvent(self, event):
        """마우스 클릭 이벤트"""
        port = self.find_port_at(event.scenePos())
        
        if port and port.is_output:
            # 출력 포트에서 연결 시작
            self.start_connection(port)
        elif port and not port.is_output and self.current_connection:
            # 입력 포트에 연결 완료
            self.end_connection(port)
        else:
            super().mousePressEvent(event)
            
    def mouseMoveEvent(self, event):
        """마우스 이동 이벤트"""
        if self.current_connection:
            self.current_connection.update_path(event.scenePos())
            
            # 가까운 포트 하이라이트
            nearest_port = self.find_nearest_port(event.scenePos(), "input", 80)
            
            if nearest_port != self.highlighted_port:
                # 이전 하이라이트 해제
                if self.highlighted_port:
                    self.highlighted_port.highlight(False)
                    
                # 새 포트 하이라이트
                if nearest_port:
                    nearest_port.highlight(True)
                    
                self.highlighted_port = nearest_port
                
        super().mouseMoveEvent(event)
        
    def mouseReleaseEvent(self, event):
        """마우스 릴리즈 이벤트"""
        if self.current_connection:
            # 가까운 입력 포트 찾기
            nearest_port = self.find_nearest_port(event.scenePos(), "input", 80)
            
            if nearest_port:
                self.end_connection(nearest_port)
            else:
                # 연결 취소
                self.removeItem(self.current_connection)
                self.current_connection = None
                self.start_port = None
                
            # 하이라이트 해제
            if self.highlighted_port:
                self.highlighted_port.highlight(False)
                self.highlighted_port = None
                
        super().mouseReleaseEvent(event)
        
    def start_connection(self, port):
        """연결 시작"""
        self.start_port = port
        self.current_connection = Connection(port)
        self.addItem(self.current_connection)
        
    def end_connection(self, end_port):
        """연결 완료"""
        if self.start_port and self.current_connection and self.start_port.can_connect_to(end_port):
            # 연결 완료
            self.current_connection.end_port = end_port
            self.current_connection.update_path()
            
            self.start_port.connections.append(self.current_connection)
            end_port.connections.append(self.current_connection)
            
            self.current_connection = None
            self.start_port = None
        else:
            # 연결 실패
            if self.current_connection:
                self.removeItem(self.current_connection)
                self.current_connection = None
                self.start_port = None


class NodeView(QGraphicsView):
    """노드 에디터 뷰"""
    def __init__(self, scene):
        super().__init__(scene)
        
        # 뷰 설정
        self.setRenderHint(QPainter.Antialiasing)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        
        # 줌 관련
        self.zoom_factor = 1.15
        self.zoom_level = 0
        self.max_zoom = 10
        self.min_zoom = -10
        
    def wheelEvent(self, event):
        """마우스 휠로 줌"""
        # 줌 인/아웃
        if event.angleDelta().y() > 0 and self.zoom_level < self.max_zoom:
            self.scale(self.zoom_factor, self.zoom_factor)
            self.zoom_level += 1
        elif event.angleDelta().y() < 0 and self.zoom_level > self.min_zoom:
            self.scale(1/self.zoom_factor, 1/self.zoom_factor)
            self.zoom_level -= 1
            
    def keyPressEvent(self, event):
        """키보드 이벤트"""
        if event.key() == Qt.Key_Delete:
            # 선택된 아이템 삭제
            for item in self.scene().selectedItems():
                if isinstance(item, Node):
                    self.delete_node(item)
                elif isinstance(item, Connection):
                    item.remove()
        elif event.key() == Qt.Key_Space:
            # 전체 보기
            self.fitInView(self.scene().itemsBoundingRect(), Qt.KeepAspectRatio)
            self.zoom_level = 0
            
        super().keyPressEvent(event)
        
    def delete_node(self, node):
        """노드 삭제"""
        # 연결된 모든 연결선 제거
        for port in node.input_ports + node.output_ports:
            for connection in port.connections[:]:
                connection.remove()
                    
        # 노드 제거
        self.scene().removeItem(node)


class LogisticsPredictionSystem(QMainWindow):
    """물류이동 예측 시스템 메인 윈도우"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("물류이동 예측 시스템 - Qt Node Editor")
        self.setGeometry(100, 100, 1400, 800)
        
        # 다크 테마 적용
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QDockWidget {
                background-color: #252525;
                color: #ffffff;
            }
            QDockWidget::title {
                background-color: #2d2d2d;
                padding: 5px;
            }
            QPushButton {
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #555555;
                padding: 5px 10px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #484848;
            }
            QPushButton:pressed {
                background-color: #2d2d2d;
            }
            QLabel {
                color: #ffffff;
            }
            QListWidget {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 1px solid #444444;
            }
            QListWidget::item:hover {
                background-color: #3c3c3c;
            }
            QTextEdit {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 1px solid #444444;
            }
        """)
        
        # 씬과 뷰 생성
        self.scene = NodeScene()
        self.view = NodeView(self.scene)
        self.setCentralWidget(self.view)
        
        # UI 초기화
        self.init_ui()
        
        # 씬 이벤트 연결
        self.scene.selectionChanged.connect(self.update_properties)
        
    def init_ui(self):
        """UI 초기화"""
        # 메뉴바
        self.create_menu_bar()
        
        # 툴바
        self.create_toolbar()
        
        # 독 위젯들
        self.create_dock_widgets()
        
        # 상태바
        self.statusBar().showMessage("준비됨")
        
    def create_menu_bar(self):
        """메뉴바 생성"""
        menubar = self.menuBar()
        
        # 파일 메뉴
        file_menu = menubar.addMenu("파일")
        
        new_action = QAction("새 파일", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.new_pipeline)
        file_menu.addAction(new_action)
        
        open_action = QAction("열기", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.load_pipeline)
        file_menu.addAction(open_action)
        
        save_action = QAction("저장", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_pipeline)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("종료", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 편집 메뉴
        edit_menu = menubar.addMenu("편집")
        
        delete_action = QAction("삭제", self)
        delete_action.setShortcut("Delete")
        delete_action.triggered.connect(self.delete_selected)
        edit_menu.addAction(delete_action)
        
        # 보기 메뉴
        view_menu = menubar.addMenu("보기")
        
        grid_action = QAction("그리드 표시", self)
        grid_action.setCheckable(True)
        grid_action.setChecked(True)
        grid_action.triggered.connect(self.toggle_grid)
        view_menu.addAction(grid_action)
        
        fit_action = QAction("전체 보기", self)
        fit_action.setShortcut("Space")
        fit_action.triggered.connect(self.fit_view)
        view_menu.addAction(fit_action)
        
    def create_toolbar(self):
        """툴바 생성"""
        toolbar = self.addToolBar("메인 툴바")
        toolbar.setMovable(False)
        
        # 실행 버튼
        run_action = QAction(QIcon(), "실행", self)
        run_action.triggered.connect(self.run_pipeline)
        toolbar.addAction(run_action)
        
        # 검증 버튼
        validate_action = QAction(QIcon(), "검증", self)
        validate_action.triggered.connect(self.validate_pipeline)
        toolbar.addAction(validate_action)
        
        toolbar.addSeparator()
        
        # 줌 컨트롤
        zoom_in_action = QAction(QIcon(), "확대", self)
        zoom_in_action.triggered.connect(lambda: self.view.scale(1.2, 1.2))
        toolbar.addAction(zoom_in_action)
        
        zoom_out_action = QAction(QIcon(), "축소", self)
        zoom_out_action.triggered.connect(lambda: self.view.scale(0.8, 0.8))
        toolbar.addAction(zoom_out_action)
        
        zoom_reset_action = QAction(QIcon(), "100%", self)
        zoom_reset_action.triggered.connect(self.reset_zoom)
        toolbar.addAction(zoom_reset_action)
        
    def create_dock_widgets(self):
        """독 위젯 생성"""
        # 노드 팔레트
        self.create_node_palette()
        
        # 속성 패널
        self.create_properties_panel()
        
        # 콘솔 출력
        self.create_console_panel()
        
    def create_node_palette(self):
        """노드 팔레트 생성"""
        dock = QDockWidget("노드 팔레트", self)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        
        widget = QWidget()
        layout = QVBoxLayout()
        
        # 노드 카테고리별 버튼 생성
        categories = {
            "데이터 입력": [
                ("MCS 로그 데이터", NodeType.DATA, "MCS 로그"),
                ("진동센서 데이터", NodeType.DATA, "진동센서"),
            ],
            "데이터 전처리": [
                ("이상치 제거", NodeType.PREPROCESS, "이상치 제거"),
                ("시간별 분류", NodeType.PREPROCESS, "시간별 분류"),
                ("경로별 분류", NodeType.PREPROCESS, "경로별 분류"),
            ],
            "벡터 저장": [
                ("RAG 벡터 저장", NodeType.VECTOR, "RAG 벡터"),
            ],
            "시계열 모델": [
                ("LSTM 모델", NodeType.MODEL, "LSTM"),
                ("RNN 모델", NodeType.MODEL, "RNN"),
                ("ARIMA 모델", NodeType.MODEL, "ARIMA"),
            ],
            "분석": [
                ("과거패턴 분석", NodeType.ANALYSIS, "패턴분석"),
            ],
            "프롬프트": [
                ("프롬프트 생성", NodeType.PROMPT, "프롬프트"),
            ],
            "LLM": [
                ("PHI-4 판단/추론", NodeType.LLM, "PHI-4"),
            ],
        }
        
        for category, nodes in categories.items():
            group = QGroupBox(category)
            group_layout = QVBoxLayout()
            
            for label, node_type, name in nodes:
                btn = QPushButton(label)
                btn.clicked.connect(lambda checked, t=node_type, n=name: self.add_node(t, n))
                group_layout.addWidget(btn)
                
            group.setLayout(group_layout)
            layout.addWidget(group)
            
        layout.addStretch()
        widget.setLayout(layout)
        
        scroll = QScrollArea()
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        dock.setWidget(scroll)
        
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        
    def create_properties_panel(self):
        """속성 패널 생성"""
        dock = QDockWidget("노드 속성", self)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        
        self.properties_widget = QTextEdit()
        self.properties_widget.setReadOnly(True)
        self.properties_widget.setPlainText("노드를 선택하세요")
        
        dock.setWidget(self.properties_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        
    def create_console_panel(self):
        """콘솔 패널 생성"""
        dock = QDockWidget("콘솔", self)
        dock.setAllowedAreas(Qt.BottomDockWidgetArea)
        
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumHeight(150)
        
        dock.setWidget(self.console)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)
        
    def add_node(self, node_type: NodeType, name: str):
        """노드 추가"""
        # 뷰 중앙에 노드 생성
        view_center = self.view.mapToScene(self.view.rect().center())
        
        node = Node(node_type, name, view_center.x() - 100, view_center.y() - 50)
        self.scene.addItem(node)
        
        self.log(f"{name} 노드가 추가되었습니다")
        
    def configure_node(self, node):
        """노드 설정 대화상자"""
        dialog = QDialog(self)
        dialog.setWindowTitle(f"{node.name} 설정")
        dialog.setModal(True)
        layout = QVBoxLayout()
        
        # 노드 타입별 설정 UI
        if node.node_type == NodeType.DATA:
            self.create_data_config(layout, node)
        elif node.node_type == NodeType.PREPROCESS:
            self.create_preprocess_config(layout, node)
        elif node.node_type == NodeType.MODEL:
            self.create_model_config(layout, node)
        elif node.node_type == NodeType.VECTOR:
            self.create_vector_config(layout, node)
        elif node.node_type == NodeType.ANALYSIS:
            self.create_analysis_config(layout, node)
        elif node.node_type == NodeType.PROMPT:
            self.create_prompt_config(layout, node)
        elif node.node_type == NodeType.LLM:
            self.create_llm_config(layout, node)
            
        # 버튼
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(lambda: self.save_node_config(dialog, node))
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        dialog.setLayout(layout)
        dialog.exec_()
        
    def create_data_config(self, layout, node):
        """데이터 노드 설정 UI"""
        layout.addWidget(QLabel("데이터 소스 설정"))
        
        # 파일 경로
        layout.addWidget(QLabel("파일 경로:"))
        path_edit = QLineEdit()
        path_edit.setObjectName("path")
        layout.addWidget(path_edit)
        
        # 데이터 형식
        layout.addWidget(QLabel("데이터 형식:"))
        format_combo = QComboBox()
        format_combo.addItems(["CSV", "JSON", "Excel", "Database"])
        format_combo.setObjectName("format")
        layout.addWidget(format_combo)
        
        # 샘플링 간격
        layout.addWidget(QLabel("샘플링 간격 (초):"))
        sampling_spin = QSpinBox()
        sampling_spin.setRange(1, 3600)
        sampling_spin.setValue(60)
        sampling_spin.setObjectName("sampling")
        layout.addWidget(sampling_spin)
        
    def create_preprocess_config(self, layout, node):
        """전처리 노드 설정 UI"""
        layout.addWidget(QLabel("전처리 설정"))
        
        if "이상치" in node.name:
            layout.addWidget(QLabel("이상치 탐지 방법:"))
            method_combo = QComboBox()
            method_combo.addItems(["IQR", "Z-Score", "Isolation Forest", "DBSCAN"])
            method_combo.setObjectName("method")
            layout.addWidget(method_combo)
            
            layout.addWidget(QLabel("임계값:"))
            threshold_spin = QDoubleSpinBox()
            threshold_spin.setRange(0.1, 5.0)
            threshold_spin.setValue(1.5)
            threshold_spin.setSingleStep(0.1)
            threshold_spin.setObjectName("threshold")
            layout.addWidget(threshold_spin)
            
        elif "시간별" in node.name:
            layout.addWidget(QLabel("시간 단위:"))
            time_combo = QComboBox()
            time_combo.addItems(["30분", "1시간", "2시간", "4시간", "6시간", "12시간", "24시간"])
            time_combo.setCurrentText("1시간")
            time_combo.setObjectName("time_unit")
            layout.addWidget(time_combo)
            
        elif "경로별" in node.name:
            layout.addWidget(QLabel("경로 그룹화 기준:"))
            route_combo = QComboBox()
            route_combo.addItems(["출발지-도착지", "주요 경유지", "운송 수단", "거리별"])
            route_combo.setObjectName("route_grouping")
            layout.addWidget(route_combo)
            
    def create_model_config(self, layout, node):
        """모델 노드 설정 UI"""
        layout.addWidget(QLabel("모델 설정"))
        
        # 예측 기간
        layout.addWidget(QLabel("예측 기간:"))
        period_combo = QComboBox()
        period_combo.addItems(["6시간", "12시간", "24시간", "48시간", "72시간", "1주일"])
        period_combo.setCurrentText("24시간")
        period_combo.setObjectName("period")
        layout.addWidget(period_combo)
        
        if "LSTM" in node.name or "RNN" in node.name:
            layout.addWidget(QLabel("은닉층 수:"))
            layers_spin = QSpinBox()
            layers_spin.setRange(1, 10)
            layers_spin.setValue(3)
            layers_spin.setObjectName("layers")
            layout.addWidget(layers_spin)
            
            layout.addWidget(QLabel("유닛 수:"))
            units_spin = QSpinBox()
            units_spin.setRange(32, 512)
            units_spin.setValue(128)
            units_spin.setSingleStep(32)
            units_spin.setObjectName("units")
            layout.addWidget(units_spin)
            
            layout.addWidget(QLabel("배치 크기:"))
            batch_spin = QSpinBox()
            batch_spin.setRange(1, 128)
            batch_spin.setValue(32)
            batch_spin.setObjectName("batch_size")
            layout.addWidget(batch_spin)
            
        elif "ARIMA" in node.name:
            layout.addWidget(QLabel("p (자기회귀):"))
            p_spin = QSpinBox()
            p_spin.setRange(0, 10)
            p_spin.setValue(2)
            p_spin.setObjectName("p")
            layout.addWidget(p_spin)
            
            layout.addWidget(QLabel("d (차분):"))
            d_spin = QSpinBox()
            d_spin.setRange(0, 5)
            d_spin.setValue(1)
            d_spin.setObjectName("d")
            layout.addWidget(d_spin)
            
            layout.addWidget(QLabel("q (이동평균):"))
            q_spin = QSpinBox()
            q_spin.setRange(0, 10)
            q_spin.setValue(2)
            q_spin.setObjectName("q")
            layout.addWidget(q_spin)
            
    def create_vector_config(self, layout, node):
        """벡터 저장 노드 설정 UI"""
        layout.addWidget(QLabel("RAG 벡터 저장 설정"))
        
        layout.addWidget(QLabel("임베딩 모델:"))
        embed_combo = QComboBox()
        embed_combo.addItems(["OpenAI", "Sentence-BERT", "Custom", "Multilingual"])
        embed_combo.setObjectName("embedding_model")
        layout.addWidget(embed_combo)
        
        layout.addWidget(QLabel("벡터 차원:"))
        dim_spin = QSpinBox()
        dim_spin.setRange(128, 2048)
        dim_spin.setValue(768)
        dim_spin.setSingleStep(128)
        dim_spin.setObjectName("vector_dim")
        layout.addWidget(dim_spin)
        
        layout.addWidget(QLabel("벡터 저장소:"))
        store_combo = QComboBox()
        store_combo.addItems(["ChromaDB", "Pinecone", "Weaviate", "FAISS"])
        store_combo.setObjectName("vector_store")
        layout.addWidget(store_combo)
        
    def create_analysis_config(self, layout, node):
        """분석 노드 설정 UI"""
        layout.addWidget(QLabel("패턴 분석 설정"))
        
        layout.addWidget(QLabel("분석 기간:"))
        period_combo = QComboBox()
        period_combo.addItems(["7일", "14일", "30일", "90일", "180일", "1년"])
        period_combo.setCurrentText("30일")
        period_combo.setObjectName("analysis_period")
        layout.addWidget(period_combo)
        
        layout.addWidget(QLabel("패턴 유형:"))
        patterns = ["계절성", "주기성", "트렌드", "이상패턴", "피크시간"]
        for i, pattern in enumerate(patterns):
            check = QCheckBox(pattern)
            check.setChecked(True)
            check.setObjectName(f"pattern_{i}")
            layout.addWidget(check)
            
    def create_prompt_config(self, layout, node):
        """프롬프트 노드 설정 UI"""
        layout.addWidget(QLabel("프롬프트 생성 설정"))
        
        layout.addWidget(QLabel("프롬프트 템플릿:"))
        template_text = QTextEdit()
        template_text.setObjectName("template")
        default_template = """다음 물류 데이터를 분석하여 예측해주세요:
- 현재 상황: {current_status}
- 과거 패턴: {past_patterns}
- 시계열 예측: {time_series_prediction}
- 특이사항: {anomalies}

24시간 이내의 물류 이동을 예측하고,
주의해야 할 리스크를 분석해주세요."""
        template_text.setPlainText(default_template)
        template_text.setMaximumHeight(150)
        layout.addWidget(template_text)
        
        layout.addWidget(QLabel("포함할 컨텍스트:"))
        contexts = ["날씨 정보", "교통 상황", "과거 지연 이력", "특별 이벤트"]
        for i, context in enumerate(contexts):
            check = QCheckBox(context)
            check.setChecked(True)
            check.setObjectName(f"context_{i}")
            layout.addWidget(check)
            
    def create_llm_config(self, layout, node):
        """LLM 노드 설정 UI"""
        layout.addWidget(QLabel("PHI-4 LLM 설정"))
        
        layout.addWidget(QLabel("Temperature:"))
        temp_slider = QSlider(Qt.Horizontal)
        temp_slider.setRange(0, 100)
        temp_slider.setValue(70)
        temp_slider.setObjectName("temperature")
        temp_label = QLabel("0.7")
        temp_slider.valueChanged.connect(lambda v: temp_label.setText(f"{v/100:.1f}"))
        
        temp_layout = QHBoxLayout()
        temp_layout.addWidget(temp_slider)
        temp_layout.addWidget(temp_label)
        layout.addLayout(temp_layout)
        
        layout.addWidget(QLabel("Max Tokens:"))
        tokens_spin = QSpinBox()
        tokens_spin.setRange(512, 8192)
        tokens_spin.setValue(2048)
        tokens_spin.setSingleStep(512)
        tokens_spin.setObjectName("max_tokens")
        layout.addWidget(tokens_spin)
        
        layout.addWidget(QLabel("추론 모드:"))
        mode_combo = QComboBox()
        mode_combo.addItems(["종합분석", "리스크 평가", "최적화 제안", "이상탐지"])
        mode_combo.setObjectName("inference_mode")
        layout.addWidget(mode_combo)
        
        layout.addWidget(QLabel("출력 형식:"))
        format_combo = QComboBox()
        format_combo.addItems(["구조화된 JSON", "자연어 설명", "대시보드 데이터", "리포트"])
        format_combo.setObjectName("output_format")
        layout.addWidget(format_combo)
        
    def save_node_config(self, dialog, node):
        """노드 설정 저장"""
        # 대화상자에서 설정 값 수집
        settings = {}
        for child in dialog.findChildren(QWidget):
            if child.objectName():
                if isinstance(child, QLineEdit):
                    settings[child.objectName()] = child.text()
                elif isinstance(child, QComboBox):
                    settings[child.objectName()] = child.currentText()
                elif isinstance(child, QSpinBox) or isinstance(child, QDoubleSpinBox):
                    settings[child.objectName()] = child.value()
                elif isinstance(child, QCheckBox):
                    settings[child.objectName()] = child.isChecked()
                elif isinstance(child, QTextEdit):
                    settings[child.objectName()] = child.toPlainText()
                elif isinstance(child, QSlider):
                    settings[child.objectName()] = child.value() / 100
                    
        node.settings = settings
        node.is_configured = True
        node.update_status()
        
        self.log(f"{node.name} 설정이 완료되었습니다")
        self.update_properties()
        
        dialog.accept()
        
    def new_pipeline(self):
        """새 파이프라인"""
        reply = QMessageBox.question(self, "확인", "현재 작업을 지우고 새로 시작하시겠습니까?")
        if reply == QMessageBox.Yes:
            self.scene.clear()
            self.log("새 파이프라인이 생성되었습니다")
            
    def save_pipeline(self):
        """파이프라인 저장"""
        filename, _ = QFileDialog.getSaveFileName(self, "파이프라인 저장", "", "JSON Files (*.json)")
        if filename:
            data = {
                "nodes": [],
                "connections": []
            }
            
            # 노드 정보 수집
            node_map = {}
            for item in self.scene.items():
                if isinstance(item, Node):
                    node_data = {
                        "id": item.node_id,
                        "type": item.node_type.value,
                        "name": item.name,
                        "x": item.x(),
                        "y": item.y(),
                        "configured": item.is_configured,
                        "settings": item.settings
                    }
                    data["nodes"].append(node_data)
                    node_map[item] = item.node_id
                    
            # 연결 정보 수집
            for item in self.scene.items():
                if isinstance(item, Connection) and item.start_port and item.end_port:
                    start_node = item.start_port.parentItem()
                    end_node = item.end_port.parentItem()
                    
                    if start_node in node_map and end_node in node_map:
                        conn_data = {
                            "start": node_map[start_node],
                            "start_port": start_node.output_ports.index(item.start_port),
                            "end": node_map[end_node],
                            "end_port": end_node.input_ports.index(item.end_port)
                        }
                        data["connections"].append(conn_data)
                        
            # JSON 파일로 저장
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            self.log(f"파이프라인이 저장되었습니다: {filename}")
            
    def load_pipeline(self):
        """파이프라인 불러오기"""
        filename, _ = QFileDialog.getOpenFileName(self, "파이프라인 열기", "", "JSON Files (*.json)")
        if filename:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 씬 초기화
            self.scene.clear()
            
            # 노드 생성
            node_map = {}
            for node_data in data["nodes"]:
                node_type = NodeType(node_data["type"])
                node = Node(node_type, node_data["name"], node_data["x"], node_data["y"])
                node.is_configured = node_data.get("configured", False)
                node.settings = node_data.get("settings", {})
                node.update_status()
                
                self.scene.addItem(node)
                node_map[node_data["id"]] = node
                
            # 연결 생성
            for conn_data in data["connections"]:
                start_node = node_map[conn_data["start"]]
                end_node = node_map[conn_data["end"]]
                
                start_port = start_node.output_ports[conn_data.get("start_port", 0)]
                end_port = end_node.input_ports[conn_data.get("end_port", 0)]
                
                connection = Connection(start_port, end_port)
                self.scene.addItem(connection)
                
            self.log(f"파이프라인을 불러왔습니다: {filename}")
            
    def delete_selected(self):
        """선택된 아이템 삭제"""
        for item in self.scene.selectedItems():
            if isinstance(item, Node):
                self.view.delete_node(item)
            elif isinstance(item, Connection):
                item.remove()
                
    def toggle_grid(self, checked):
        """그리드 표시 토글"""
        self.scene.grid_visible = checked
        self.scene.update()
        
    def fit_view(self):
        """전체 보기"""
        self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)
        self.view.zoom_level = 0
        
    def reset_zoom(self):
        """줌 초기화"""
        self.view.resetTransform()
        self.view.zoom_level = 0
        
    def validate_pipeline(self):
        """파이프라인 검증"""
        errors = []
        warnings = []
        
        # 노드 수집
        nodes = [item for item in self.scene.items() if isinstance(item, Node)]
        
        if not nodes:
            errors.append("노드가 없습니다")
        else:
            # 데이터 입력 노드 확인
            data_nodes = [n for n in nodes if n.node_type == NodeType.DATA]
            if not data_nodes:
                errors.append("데이터 입력 노드가 필요합니다")
                
            # LLM 노드 확인
            llm_nodes = [n for n in nodes if n.node_type == NodeType.LLM]
            if not llm_nodes:
                errors.append("LLM 노드가 필요합니다")
                
            # 미설정 노드 확인
            unconfigured = [n.name for n in nodes if not n.is_configured]
            if unconfigured:
                warnings.append(f"미설정 노드: {', '.join(unconfigured)}")
                
            # 연결 확인
            for node in nodes:
                if node.node_type != NodeType.DATA:
                    has_input = any(port.connections for port in node.input_ports)
                    if not has_input:
                        warnings.append(f"{node.name}에 입력이 없습니다")
                        
        # 결과 표시
        if errors or warnings:
            msg = ""
            if errors:
                msg += "오류:\n" + "\n".join(f"- {e}" for e in errors) + "\n\n"
            if warnings:
                msg += "경고:\n" + "\n".join(f"- {w}" for w in warnings)
                
            QMessageBox.warning(self, "검증 결과", msg)
            self.log("파이프라인 검증 실패")
        else:
            QMessageBox.information(self, "검증 결과", "파이프라인이 유효합니다!")
            self.log("파이프라인 검증 성공")
            
    def run_pipeline(self):
        """파이프라인 실행"""
        # 검증
        self.validate_pipeline()
        
        self.log("파이프라인 실행 시작...")
        
        # 실행 시뮬레이션
        result_dialog = QDialog(self)
        result_dialog.setWindowTitle("실행 결과")
        result_dialog.setModal(True)
        layout = QVBoxLayout()
        
        result_text = QTextEdit()
        result_text.setReadOnly(True)
        
        result = """=== 물류이동 예측 시스템 실행 결과 ===

실행 시간: 2025-07-30 10:30:00

=== 예측 결과 ===
향후 24시간 물류 이동 예측:
- 예상 물동량: 1,234톤 (±5%)
- 주요 경로: 서울→부산 (45%), 인천→광주 (30%)
- 병목 구간: 경부고속도로 (14:00-18:00)
- 지연 위험도: 중간 (날씨 영향)

권장 사항:
1. 오후 2-6시 경부고속도로 우회 경로 활용
2. 긴급 화물은 오전 배송 권장
3. 예비 차량 20% 추가 배치 필요

=== 상세 분석 ===
시계열 모델 예측 정확도: 92.3%
패턴 분석 결과: 주중 오후 시간대 물동량 증가 패턴 확인
LLM 추론: 기상 악화로 인한 지연 가능성 30%"""
        
        result_text.setPlainText(result)
        layout.addWidget(result_text)
        
        close_btn = QPushButton("닫기")
        close_btn.clicked.connect(result_dialog.accept)
        layout.addWidget(close_btn)
        
        result_dialog.setLayout(layout)
        result_dialog.resize(600, 400)
        result_dialog.exec_()
        
        self.log("파이프라인 실행 완료")
        
    def update_properties(self):
        """속성 패널 업데이트"""
        selected = self.scene.selectedItems()
        if selected and isinstance(selected[0], Node):
            node = selected[0]
            info = f"""노드 정보
---------
이름: {node.name}
타입: {node.node_type.value}
ID: {node.node_id}
설정 상태: {'완료' if node.is_configured else '미완료'}
위치: ({int(node.x())}, {int(node.y())})

설정 내용:
{json.dumps(node.settings, indent=2, ensure_ascii=False) if node.settings else '없음'}"""
            
            self.properties_widget.setPlainText(info)
        else:
            self.properties_widget.setPlainText("노드를 선택하세요")
            
    def log(self, message):
        """콘솔에 로그 출력"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.console.append(f"[{timestamp}] {message}")
        self.statusBar().showMessage(message, 3000)


def main():
    app = QApplication(sys.argv)
    
    # 애플리케이션 스타일 설정
    app.setStyle("Fusion")
    
    # 다크 팔레트
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(45, 45, 45))
    palette.setColor(QPalette.AlternateBase, QColor(60, 60, 60))
    palette.setColor(QPalette.ToolTipBase, Qt.black)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(45, 45, 45))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)
    
    window = LogisticsPredictionSystem()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()