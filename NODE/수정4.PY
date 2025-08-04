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


class MemoItem(QGraphicsRectItem):
    """메모 아이템 클래스"""
    def __init__(self, x=0, y=0, width=250, height=150):
        super().__init__(0, 0, width, height)
        
        self.memo_id = id(self)
        self.setPos(x, y)
        
        # 메모 스타일
        self.colors = [
            "#fffacd",  # 연한 노란색
            "#ffe4e1",  # 연한 분홍색
            "#e0ffff",  # 연한 하늘색
            "#f0fff0",  # 연한 초록색
            "#f5f5dc",  # 베이지색
            "#fff0f5",  # 연한 보라색
        ]
        self.current_color_index = 0
        self.setColor(self.colors[0])
        
        # 플래그 설정
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(-2)  # 노드보다 뒤에 표시
        
        # 텍스트 아이템
        self.text_item = QGraphicsTextItem(self)
        self.text_item.setPlainText("메모를 입력하세요...")
        self.text_item.setDefaultTextColor(QColor("#333333"))
        self.text_item.setPos(10, 10)
        self.text_item.setTextWidth(width - 20)
        
        # 제목 바
        self.title_height = 25
        self.is_editing = False
        
        # 리사이즈 핸들
        self.resize_handle_size = 10
        self.is_resizing = False
        self.resize_start_pos = None
        self.resize_start_rect = None
        
        # 그림자 효과
        shadow = QGraphicsDropShadowEffect()
        shadow.setOffset(2, 2)
        shadow.setBlurRadius(5)
        shadow.setColor(QColor(0, 0, 0, 50))
        self.setGraphicsEffect(shadow)
        
    def setColor(self, color):
        """메모 색상 설정"""
        self.setBrush(QBrush(QColor(color)))
        self.setPen(QPen(QColor(color).darker(120), 2))
        
    def paint(self, painter, option, widget):
        """메모 그리기"""
        super().paint(painter, option, widget)
        
        # 제목 바 그리기
        title_rect = QRectF(0, 0, self.rect().width(), self.title_height)
        painter.fillRect(title_rect, QBrush(QColor(0, 0, 0, 30)))
        
        # 제목 텍스트
        painter.setPen(QPen(QColor("#555555")))
        painter.setFont(QFont("Arial", 9))
        painter.drawText(title_rect.adjusted(10, 0, -40, 0),
                        Qt.AlignVCenter, "메모")
        
        # 닫기 버튼 영역
        close_rect = QRectF(self.rect().width() - 25, 5, 15, 15)
        painter.drawText(close_rect, Qt.AlignCenter, "×")
        
        # 색상 변경 버튼 영역
        color_rect = QRectF(self.rect().width() - 45, 5, 15, 15)
        painter.fillRect(color_rect, QBrush(QColor(self.colors[(self.current_color_index + 1) % len(self.colors)])))
        painter.drawRect(color_rect)
        
        # 리사이즈 핸들
        if self.isSelected():
            handle_rect = QRectF(
                self.rect().width() - self.resize_handle_size,
                self.rect().height() - self.resize_handle_size,
                self.resize_handle_size,
                self.resize_handle_size
            )
            # 더 눈에 띄는 리사이즈 핸들
            painter.fillRect(handle_rect, QBrush(QColor("#3498db")))
            painter.setPen(QPen(QColor("#2980b9"), 1))
            painter.drawRect(handle_rect)
            
            # 리사이즈 아이콘 그리기 (세 개의 대각선)
            painter.setPen(QPen(QColor("#ffffff"), 1))
            for i in range(3):
                offset = i * 3
                painter.drawLine(
                    handle_rect.right() - offset - 2,
                    handle_rect.bottom() - 1,
                    handle_rect.right() - 1,
                    handle_rect.bottom() - offset - 2
                )
            painter.fillRect(handle_rect, QBrush(QColor("#666666")))
            
    def mousePressEvent(self, event):
        """마우스 클릭 이벤트"""
        pos = event.pos()
        
        # 닫기 버튼 클릭
        close_rect = QRectF(self.rect().width() - 25, 5, 15, 15)
        if close_rect.contains(pos):
            self.delete_self()
            return
            
        # 색상 변경 버튼 클릭
        color_rect = QRectF(self.rect().width() - 45, 5, 15, 15)
        if color_rect.contains(pos):
            self.change_color()
            return
            
        # 리사이즈 핸들 클릭
        handle_rect = QRectF(
            self.rect().width() - self.resize_handle_size,
            self.rect().height() - self.resize_handle_size,
            self.resize_handle_size,
            self.resize_handle_size
        )
        if handle_rect.contains(pos) and self.isSelected():
            self.is_resizing = True
            self.resize_start_pos = event.scenePos()
            self.resize_start_rect = self.rect()
            event.accept()
            return
            
        super().mousePressEvent(event)
        
    def mouseMoveEvent(self, event):
        """마우스 이동 이벤트"""
        if self.is_resizing:
            # 리사이즈 처리
            diff = event.scenePos() - self.resize_start_pos
            new_width = max(150, self.resize_start_rect.width() + diff.x())
            new_height = max(100, self.resize_start_rect.height() + diff.y())
            
            self.setRect(0, 0, new_width, new_height)
            self.text_item.setTextWidth(new_width - 20)
            self.update()
        else:
            super().mouseMoveEvent(event)
            
    def mouseReleaseEvent(self, event):
        """마우스 릴리즈 이벤트"""
        self.is_resizing = False
        super().mouseReleaseEvent(event)
        
    def mouseDoubleClickEvent(self, event):
        """더블클릭으로 편집 모드"""
        self.edit_text()
        
    def hoverEnterEvent(self, event):
        """마우스 호버 시"""
        self.setCursor(Qt.PointingHandCursor)
        super().hoverEnterEvent(event)
        
    def hoverMoveEvent(self, event):
        """호버 중 마우스 이동"""
        pos = event.pos()
        
        # 리사이즈 핸들 위에서 커서 변경
        handle_rect = QRectF(
            self.rect().width() - self.resize_handle_size,
            self.rect().height() - self.resize_handle_size,
            self.resize_handle_size,
            self.resize_handle_size
        )
        if handle_rect.contains(pos) and self.isSelected():
            self.setCursor(Qt.SizeFDiagCursor)
        else:
            self.setCursor(Qt.PointingHandCursor)
            
    def edit_text(self):
        """텍스트 편집"""
        dialog = QDialog()
        dialog.setWindowTitle("메모 편집")
        dialog.setModal(True)
        layout = QVBoxLayout()
        
        # 텍스트 편집기
        text_edit = QTextEdit()
        text_edit.setPlainText(self.text_item.toPlainText())
        text_edit.setMinimumSize(400, 300)
        layout.addWidget(text_edit)
        
        # 버튼
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        dialog.setLayout(layout)
        
        if dialog.exec_() == QDialog.Accepted:
            self.text_item.setPlainText(text_edit.toPlainText())
            
    def change_color(self):
        """색상 변경"""
        self.current_color_index = (self.current_color_index + 1) % len(self.colors)
        self.setColor(self.colors[self.current_color_index])
        self.update()
        
    def delete_self(self):
        """자신을 삭제"""
        if hasattr(self.scene(), 'main_window') and self.scene().main_window:
            reply = QMessageBox.question(None, "확인",
                                       "이 메모를 삭제하시겠습니까?",
                                       QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.scene().removeItem(self)
                self.scene().main_window.log("메모가 삭제되었습니다")
                
    def get_data(self):
        """메모 데이터 반환 (저장용)"""
        return {
            "id": self.memo_id,
            "x": self.x(),
            "y": self.y(),
            "width": self.rect().width(),
            "height": self.rect().height(),
            "text": self.text_item.toPlainText(),
            "color_index": self.current_color_index
        }
        
    def set_data(self, data):
        """메모 데이터 설정 (불러오기용)"""
        self.setPos(data["x"], data["y"])
        self.setRect(0, 0, data["width"], data["height"])
        self.text_item.setPlainText(data["text"])
        self.text_item.setTextWidth(data["width"] - 20)
        self.current_color_index = data.get("color_index", 0)
        self.setColor(self.colors[self.current_color_index])


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
        if hasattr(self.scene(), 'main_window') and self.scene().main_window:
            self.scene().main_window.configure_node(self)
        super().mouseDoubleClickEvent(event)
        
    def contextMenuEvent(self, event):
        """마우스 오른쪽 클릭 컨텍스트 메뉴"""
        menu = QMenu()
        
        # 모든 노드에 공통으로 적용되는 메뉴
        configure_action = QAction("⚙️ 노드 설정", None)
        configure_action.triggered.connect(lambda: self.scene().main_window.configure_node(self) if hasattr(self.scene(), 'main_window') and self.scene().main_window else None)
        menu.addAction(configure_action)
        
        # 데이터 노드 전용 메뉴
        if self.node_type == NodeType.DATA:
            if self.is_configured and 'path' in self.settings:
                open_file_action = QAction("📁 파일 위치 열기", None)
                open_file_action.triggered.connect(lambda: self.open_file_location())
                menu.addAction(open_file_action)
        
        # 모델 노드 전용 메뉴
        elif self.node_type == NodeType.MODEL:
            if self.is_configured:
                show_params_action = QAction("📊 모델 파라미터 보기", None)
                show_params_action.triggered.connect(lambda: self.show_model_params())
                menu.addAction(show_params_action)
        
        menu.addSeparator()
        
        # 노드 복제
        duplicate_action = QAction("📑 노드 복제", None)
        duplicate_action.triggered.connect(lambda: self.duplicate_node())
        menu.addAction(duplicate_action)
        
        # 노드 삭제
        delete_action = QAction("🗑️ 노드 삭제", None)
        delete_action.triggered.connect(lambda: self.delete_self())
        menu.addAction(delete_action)
        
        # 메뉴 표시
        menu.exec_(event.screenPos())
            
    def open_file_location(self):
        """파일 위치 열기"""
        if 'path' in self.settings:
            import os
            path = self.settings['path']
            if os.path.exists(path):
                if os.path.isfile(path):
                    os.startfile(os.path.dirname(path))
                else:
                    os.startfile(path)
                    
    def show_model_params(self):
        """모델 파라미터 표시"""
        params = []
        for key, value in self.settings.items():
            params.append(f"{key}: {value}")
            
        msg = QMessageBox()
        msg.setWindowTitle(f"{self.name} 파라미터")
        msg.setText("현재 설정된 모델 파라미터:")
        msg.setDetailedText('\n'.join(params))
        msg.setIcon(QMessageBox.Information)
        msg.exec_()
        
    def duplicate_node(self):
        """노드 복제"""
        if hasattr(self.scene(), 'main_window') and self.scene().main_window:
            # 새 노드 생성
            new_node = Node(self.node_type, self.name, self.x() + 50, self.y() + 50)
            new_node.settings = self.settings.copy()
            new_node.is_configured = self.is_configured
            new_node.update_status()
            
            self.scene().addItem(new_node)
            self.scene().main_window.log(f"{self.name} 노드가 복제되었습니다")
            
    def delete_self(self):
        """자신을 삭제"""
        if hasattr(self.scene(), 'main_window') and self.scene().main_window:
            reply = QMessageBox.question(None, "확인",
                                       f"{self.name} 노드를 삭제하시겠습니까?",
                                       QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.scene().main_window.view.delete_node(self)


class NodeScene(QGraphicsScene):
    """노드 에디터 씬"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent  # 메인 윈도우 참조 저장
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
        # 클릭한 위치의 아이템 확인
        items = self.items(event.scenePos())
        port = None
        
        # 포트가 있는지 확인
        for item in items:
            if isinstance(item, Port):
                port = item
                break
        
        if port and port.is_output:
            # 출력 포트에서 연결 시작
            self.start_connection(port)
            event.accept()  # 이벤트 처리 완료
        elif port and not port.is_output and self.current_connection:
            # 입력 포트에 연결 완료
            self.end_connection(port)
            event.accept()  # 이벤트 처리 완료
        else:
            # 포트가 아닌 경우 (노드나 빈 공간) 기본 처리
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
                
    def contextMenuEvent(self, event):
        """씬 우클릭 컨텍스트 메뉴"""
        # 아이템이 없는 빈 공간에서만 동작
        items = self.items(event.scenePos())
        if not any(isinstance(item, (Node, Connection, MemoItem)) for item in items):
            menu = QMenu()
            
            # 메모 추가 메뉴
            add_memo_action = QAction("📝 메모 추가", None)
            add_memo_action.triggered.connect(lambda: self.add_memo_at(event.scenePos()))
            menu.addAction(add_memo_action)
            
            menu.exec_(event.screenPos())
        else:
            super().contextMenuEvent(event)
            
    def add_memo_at(self, pos):
        """지정된 위치에 메모 추가"""
        memo = MemoItem(pos.x() - 125, pos.y() - 75)  # 중앙에 오도록 조정
        self.addItem(memo)
        if self.main_window:
            self.main_window.log("메모가 추가되었습니다")


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
        
        # 팬(이동) 관련
        self.is_panning = False
        self.pan_start_pos = None
        self.space_pressed = False
        
    def wheelEvent(self, event):
        """마우스 휠로 줌"""
        # 줌 인/아웃
        if event.angleDelta().y() > 0 and self.zoom_level < self.max_zoom:
            self.scale(self.zoom_factor, self.zoom_factor)
            self.zoom_level += 1
        elif event.angleDelta().y() < 0 and self.zoom_level > self.min_zoom:
            self.scale(1/self.zoom_factor, 1/self.zoom_factor)
            self.zoom_level -= 1
            
    def mousePressEvent(self, event):
        """마우스 클릭 이벤트"""
        # 가운데 버튼 또는 스페이스 + 왼쪽 버튼으로 팬 시작
        if event.button() == Qt.MiddleButton or (self.space_pressed and event.button() == Qt.LeftButton):
            self.is_panning = True
            self.pan_start_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            self.setDragMode(QGraphicsView.NoDrag)
        else:
            super().mousePressEvent(event)
            
    def mouseMoveEvent(self, event):
        """마우스 이동 이벤트"""
        if self.is_panning:
            # 화면 이동
            delta = event.pos() - self.pan_start_pos
            self.pan_start_pos = event.pos()
            
            # 스크롤바 이동
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
        else:
            super().mouseMoveEvent(event)
            
    def mouseReleaseEvent(self, event):
        """마우스 릴리즈 이벤트"""
        if event.button() == Qt.MiddleButton or (self.is_panning and event.button() == Qt.LeftButton):
            self.is_panning = False
            self.setCursor(Qt.ArrowCursor)
            self.setDragMode(QGraphicsView.RubberBandDrag)
        else:
            super().mouseReleaseEvent(event)
            
    def keyPressEvent(self, event):
        """키보드 이벤트"""
        if event.key() == Qt.Key_Delete:
            # 선택된 아이템 삭제
            for item in self.scene().selectedItems():
                if isinstance(item, Node):
                    self.delete_node(item)
                elif isinstance(item, Connection):
                    item.remove()
                elif isinstance(item, MemoItem):
                    self.scene().removeItem(item)
        elif event.key() == Qt.Key_Space and not event.isAutoRepeat():
            # 스페이스바 누르면 팬 모드 활성화
            self.space_pressed = True
            self.setCursor(Qt.OpenHandCursor)
        elif event.key() == Qt.Key_F:
            # F키로 전체 보기
            self.fitInView(self.scene().itemsBoundingRect(), Qt.KeepAspectRatio)
            self.zoom_level = 0
            
        super().keyPressEvent(event)
        
    def keyReleaseEvent(self, event):
        """키보드 릴리즈 이벤트"""
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self.space_pressed = False
            if not self.is_panning:
                self.setCursor(Qt.ArrowCursor)
                
        super().keyReleaseEvent(event)
        
    def delete_node(self, node):
        """노드 삭제"""
        # 연결된 모든 연결선 제거
        for port in node.input_ports + node.output_ports:
            for connection in port.connections[:]:
                connection.remove()
                    
        # 노드 제거
        self.scene().removeItem(node)


class SemiconductorMCSSystem(QMainWindow):
    """반도체 FAB MCS 예측 시스템 메인 윈도우"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("반도체 FAB MCS 예측 시스템 - 딥러닝 Node Editor")
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
        
        # 씬과 뷰 생성 - parent로 self 전달
        self.scene = NodeScene(self)
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
        
        # 메모 추가
        add_memo_action = QAction("메모 추가", self)
        add_memo_action.setShortcut("Ctrl+M")
        add_memo_action.triggered.connect(self.add_memo)
        edit_menu.addAction(add_memo_action)
        
        edit_menu.addSeparator()
        
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
        fit_action.setShortcut("F")
        fit_action.triggered.connect(self.fit_view)
        view_menu.addAction(fit_action)
        
    def create_toolbar(self):
        """툴바 생성"""
        toolbar = self.addToolBar("메인 툴바")
        toolbar.setMovable(False)
        
        # 메모 추가 버튼
        memo_action = QAction(QIcon(), "📝 메모", self)
        memo_action.triggered.connect(self.add_memo)
        toolbar.addAction(memo_action)
        
        toolbar.addSeparator()
        
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
        
        toolbar.addSeparator()
        
        # 도움말
        help_action = QAction(QIcon(), "도움말", self)
        help_action.triggered.connect(self.show_help)
        toolbar.addAction(help_action)
        
    def create_dock_widgets(self):
        """독 위젯 생성"""
        # 노드 팔레트
        self.create_node_palette()
        
        # 속성 패널
        self.create_properties_panel()
        
        # 콘솔 출력
        self.create_console_panel()
        
    def create_node_palette(self):
        """노드 팔레트 생성 - 통합 MCS 데이터용"""
        dock = QDockWidget("노드 팔레트", self)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        
        widget = QWidget()
        layout = QVBoxLayout()
        
        # 통합 MCS 시스템용 노드 카테고리
        categories = {
            "데이터 입력": [
                ("통합 MCS 로그", NodeType.DATA, "MCS 로그"),
            ],
            "데이터 추출/전처리": [
                ("센서 데이터 추출", NodeType.PREPROCESS, "센서 추출"),
                ("이벤트 타입 필터", NodeType.PREPROCESS, "이벤트 필터"),
                ("이상치 제거", NodeType.PREPROCESS, "이상치 제거"),
                ("시간별 집계", NodeType.PREPROCESS, "시간별 집계"),
                ("베이별 분류", NodeType.PREPROCESS, "베이별 분류"),
                ("LOT별 그룹화", NodeType.PREPROCESS, "LOT 그룹화"),
                ("장비별 분류", NodeType.PREPROCESS, "장비별 분류"),
            ],
            "벡터 저장": [
                ("RAG 벡터 저장", NodeType.VECTOR, "RAG 벡터"),
                ("알람 패턴 벡터", NodeType.VECTOR, "알람 벡터"),
            ],
            "예측 모델": [
                ("LSTM (이송시간)", NodeType.MODEL, "LSTM"),
                ("RNN (병목예측)", NodeType.MODEL, "RNN"),
                ("ARIMA (처리량)", NodeType.MODEL, "ARIMA"),
                ("센서 이상탐지", NodeType.MODEL, "센서이상탐지"),
            ],
            "분석": [
                ("OHT 패턴 분석", NodeType.ANALYSIS, "OHT패턴"),
                ("장비 가동률 분석", NodeType.ANALYSIS, "가동률"),
                ("병목 구간 분석", NodeType.ANALYSIS, "병목분석"),
                ("센서 트렌드 분석", NodeType.ANALYSIS, "센서트렌드"),
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
        self.log(f"노드 추가 요청: {name} (타입: {node_type.value})")
        
        # 뷰 중앙에 노드 생성
        view_center = self.view.mapToScene(self.view.rect().center())
        
        node = Node(node_type, name, view_center.x() - 100, view_center.y() - 50)
        self.scene.addItem(node)
        
        self.log(f"{name} 노드가 추가되었습니다 (ID: {node.node_id})")
        
    def add_memo(self):
        """메모 추가"""
        view_center = self.view.mapToScene(self.view.rect().center())
        memo = MemoItem(view_center.x() - 125, view_center.y() - 75)
        self.scene.addItem(memo)
        self.log("메모가 추가되었습니다 - 더블클릭으로 편집할 수 있습니다")
        
    def configure_node(self, node):
        """노드 설정 대화상자"""
        self.log(f"노드 설정 시작: {node.name} ({node.node_type.value})")
        
        dialog = QDialog(self)
        dialog.setWindowTitle(f"{node.name} 설정")
        dialog.setModal(True)
        dialog.setMinimumWidth(600)  # 최소 너비 설정
        dialog.setMinimumHeight(500)  # 최소 높이 설정
        
        # 다이얼로그 스타일 설정
        dialog.setStyleSheet("""
            QDialog {
                background-color: #2d2d2d;
                color: #ffffff;
            }
            QLabel {
                color: #ffffff;
            }
            QGroupBox {
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        # 스크롤 가능한 영역 생성
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()
        
        # 노드 타입별 설정 UI
        if node.node_type == NodeType.DATA:
            self.create_data_config(scroll_layout, node)
        elif node.node_type == NodeType.PREPROCESS:
            self.create_preprocess_config(scroll_layout, node)
        elif node.node_type == NodeType.MODEL:
            self.create_model_config(scroll_layout, node)
        elif node.node_type == NodeType.VECTOR:
            self.create_vector_config(scroll_layout, node)
        elif node.node_type == NodeType.ANALYSIS:
            self.create_analysis_config(scroll_layout, node)
            
        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)
        
        # 메인 레이아웃
        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll, 1)
        
        # 버튼
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(lambda: self.save_node_config(dialog, node))
        buttons.rejected.connect(dialog.reject)
        main_layout.addWidget(buttons)
        
        dialog.setLayout(main_layout)
        
        self.log(f"대화상자 표시 중...")
        result = dialog.exec_()
        self.log(f"대화상자 결과: {'확인' if result else '취소'}")
        
    def create_data_config(self, layout, node):
        """데이터 노드 설정 UI - 통합 MCS 데이터용"""
        layout.addWidget(QLabel("데이터 소스 설정"))
        
        # 파일 선택
        file_layout = QHBoxLayout()
        layout.addWidget(QLabel("파일 경로:"))
        path_edit = QLineEdit()
        path_edit.setObjectName("path")
        if 'path' in node.settings:
            path_edit.setText(node.settings['path'])
        file_btn = QPushButton("찾아보기...")
        file_btn.clicked.connect(lambda: self.browse_file(path_edit))
        file_layout.addWidget(path_edit)
        file_layout.addWidget(file_btn)
        layout.addLayout(file_layout)
        
        # 데이터 형식
        layout.addWidget(QLabel("데이터 형식:"))
        format_combo = QComboBox()
        format_combo.addItems(["CSV", "JSON", "Excel", "Database"])
        format_combo.setObjectName("format")
        layout.addWidget(format_combo)
        
        # 통합 MCS 데이터 전용 설정
        if "MCS" in node.name:
            # FAB 라인 필터
            layout.addWidget(QLabel("FAB 라인:"))
            fab_combo = QComboBox()
            fab_combo.addItems(["전체", "FAB1", "FAB2", "FAB3"])
            fab_combo.setObjectName("fab_line")
            layout.addWidget(fab_combo)
            
            # 이벤트 타입 필터
            layout.addWidget(QLabel("이벤트 타입 필터:"))
            event_types = [
                "전체", "TRANSFER", "PROCESS", "SENSOR_UPDATE", 
                "ALARM", "STOCKER", "LOAD/UNLOAD"
            ]
            for i, evt in enumerate(event_types):
                check = QCheckBox(evt)
                check.setChecked(i == 0)
                check.setObjectName(f"event_filter_{i}")
                layout.addWidget(check)
            
            # 데이터 포함 옵션
            layout.addWidget(QLabel("포함할 데이터:"))
            data_options = [
                "이벤트 로그", "센서 데이터", "알람 정보", 
                "이송 시간", "장비 상태"
            ]
            for i, opt in enumerate(data_options):
                check = QCheckBox(opt)
                check.setChecked(True)
                check.setObjectName(f"data_include_{i}")
                layout.addWidget(check)
                
            # 시간 범위 필터
            layout.addWidget(QLabel("시간 범위 (선택사항):"))
            time_range_check = QCheckBox("시간 범위 필터 사용")
            time_range_check.setObjectName("use_time_filter")
            layout.addWidget(time_range_check)
        
    def create_preprocess_config(self, layout, node):
        """전처리 노드 설정 UI - 통합 MCS 데이터용"""
        layout.addWidget(QLabel("전처리 설정"))
        
        if "센서 추출" in node.name:
            layout.addWidget(QLabel("추출할 센서 데이터:"))
            sensor_types = [
                "진동 (vibration)", "온도 (temperature)", "압력 (pressure)",
                "파티클 (particle)", "습도 (humidity)", "유량 (flow_rate)"
            ]
            for i, sensor in enumerate(sensor_types):
                check = QCheckBox(sensor)
                check.setChecked(True)
                check.setObjectName(f"extract_sensor_{i}")
                layout.addWidget(check)
            
            layout.addWidget(QLabel("추출 옵션:"))
            extract_combo = QComboBox()
            extract_combo.addItems([
                "모든 센서 이벤트", "SENSOR_UPDATE만", "PROCESS 이벤트만", 
                "알람 발생 시점만", "이상치만 추출"
            ])
            extract_combo.setObjectName("extract_option")
            layout.addWidget(extract_combo)
            
        elif "이벤트 필터" in node.name:
            layout.addWidget(QLabel("필터링할 이벤트 타입:"))
            event_types = [
                "LOAD_REQUEST", "UNLOAD_REQUEST", "TRANSFER_START", "TRANSFER_COMPLETE",
                "PROCESS_START", "PROCESS_END", "SENSOR_UPDATE", "ALARM_OCCURRED",
                "STOCKER_IN", "STOCKER_OUT"
            ]
            for i, evt in enumerate(event_types):
                check = QCheckBox(evt)
                check.setChecked(False)
                check.setObjectName(f"filter_event_{i}")
                layout.addWidget(check)
                
        elif "이상치" in node.name:
            layout.addWidget(QLabel("이상치 탐지 방법:"))
            method_combo = QComboBox()
            method_combo.addItems(["IQR", "Z-Score", "Isolation Forest", "DBSCAN", "LOF"])
            method_combo.setObjectName("method")
            layout.addWidget(method_combo)
            
            layout.addWidget(QLabel("적용 대상:"))
            target_combo = QComboBox()
            target_combo.addItems([
                "전체 수치 데이터", "센서 데이터만", "이송 시간만", 
                "특정 컬럼 선택"
            ])
            target_combo.setObjectName("anomaly_target")
            layout.addWidget(target_combo)
            
            layout.addWidget(QLabel("임계값:"))
            threshold_spin = QDoubleSpinBox()
            threshold_spin.setRange(0.1, 5.0)
            threshold_spin.setValue(1.5)
            threshold_spin.setSingleStep(0.1)
            threshold_spin.setObjectName("threshold")
            layout.addWidget(threshold_spin)
            
        elif "시간별" in node.name:
            layout.addWidget(QLabel("집계 단위:"))
            time_combo = QComboBox()
            time_combo.addItems(["1분", "5분", "10분", "30분", "1시간", "4시간", "1일"])
            time_combo.setCurrentText("1시간")
            time_combo.setObjectName("time_unit")
            layout.addWidget(time_combo)
            
            layout.addWidget(QLabel("집계 방법:"))
            agg_combo = QComboBox()
            agg_combo.addItems(["평균", "합계", "최대", "최소", "중앙값", "카운트"])
            agg_combo.setObjectName("aggregation")
            layout.addWidget(agg_combo)
            
            layout.addWidget(QLabel("집계 대상:"))
            agg_target = QComboBox()
            agg_target.addItems([
                "이벤트 수", "센서 값", "이송 시간", "알람 수", "전체"
            ])
            agg_target.setObjectName("aggregation_target")
            layout.addWidget(agg_target)
            
        elif "베이별" in node.name:
            layout.addWidget(QLabel("베이 그룹화:"))
            bays = ["PHOTO", "ETCH", "DIFF", "CVD", "PVD", "CMP", "CLEAN", "TEST"]
            for i, bay in enumerate(bays):
                check = QCheckBox(bay)
                check.setChecked(True)
                check.setObjectName(f"bay_{i}")
                layout.addWidget(check)
                
        elif "LOT" in node.name:
            layout.addWidget(QLabel("LOT 필터:"))
            lot_combo = QComboBox()
            lot_combo.addItems(["전체", "HOT LOT", "SUPER HOT", "일반 LOT"])
            lot_combo.setObjectName("lot_filter")
            layout.addWidget(lot_combo)
            
        elif "장비별" in node.name:
            layout.addWidget(QLabel("장비 그룹화 기준:"))
            group_combo = QComboBox()
            group_combo.addItems([
                "장비 ID별", "장비 타입별", "베이별 장비", "제조사별"
            ])
            group_combo.setObjectName("equipment_grouping")
            layout.addWidget(group_combo)
            
            layout.addWidget(QLabel("포함할 데이터:"))
            include_checks = ["이벤트", "센서", "알람", "가동 시간"]
            for i, item in enumerate(include_checks):
                check = QCheckBox(item)
                check.setChecked(True)
                check.setObjectName(f"equipment_include_{i}")
                layout.addWidget(check)
            
    def create_model_config(self, layout, node):
        """모델 노드 설정 UI - 반도체 FAB 용"""
        layout.addWidget(QLabel("모델 설정"))
        
        # 예측 대상
        layout.addWidget(QLabel("예측 대상:"))
        target_combo = QComboBox()
        
        if "LSTM" in node.name:
            target_combo.addItems(["이송 시간", "대기 시간", "전체 사이클 타임"])
        elif "RNN" in node.name:
            target_combo.addItems(["병목 발생 확률", "지연 시간", "OHT 정체"])
        elif "ARIMA" in node.name:
            target_combo.addItems(["시간당 처리량", "일일 생산량", "가동률"])
        elif "센서이상탐지" in node.name:
            target_combo.addItems(["진동 이상", "온도 이상", "압력 이상", "파티클 이상", "복합 이상"])
            layout.addWidget(QLabel("이상탐지 알고리즘:"))
            algo_combo = QComboBox()
            algo_combo.addItems([
                "Isolation Forest", "One-Class SVM", "Autoencoder", 
                "LSTM Autoencoder", "Statistical Process Control"
            ])
            algo_combo.setObjectName("anomaly_algorithm")
            layout.addWidget(algo_combo)
            
            layout.addWidget(QLabel("민감도:"))
            sensitivity_slider = QSlider(Qt.Horizontal)
            sensitivity_slider.setRange(1, 100)
            sensitivity_slider.setValue(80)
            sensitivity_slider.setObjectName("sensitivity")
            sensitivity_label = QLabel("80%")
            sensitivity_slider.valueChanged.connect(lambda v: sensitivity_label.setText(f"{v}%"))
            
            sens_layout = QHBoxLayout()
            sens_layout.addWidget(sensitivity_slider)
            sens_layout.addWidget(sensitivity_label)
            layout.addLayout(sens_layout)
            
        target_combo.setObjectName("prediction_target")
        layout.addWidget(target_combo)
        
        # 예측 기간
        layout.addWidget(QLabel("예측 기간:"))
        period_combo = QComboBox()
        period_combo.addItems(["10분", "30분", "1시간", "4시간", "1일", "1주일"])
        period_combo.setCurrentText("1시간")
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
        layout.addWidget(QLabel("벡터 저장 설정"))
        
        layout.addWidget(QLabel("임베딩 모델:"))
        embed_combo = QComboBox()
        embed_combo.addItems(["OpenAI", "Sentence-BERT", "Custom FAB", "Multilingual"])
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
        
        if "알람" in node.name:
            layout.addWidget(QLabel("알람 코드 그룹화:"))
            group_check = QCheckBox("유사 알람 그룹화")
            group_check.setChecked(True)
            group_check.setObjectName("group_alarms")
            layout.addWidget(group_check)
        
    def create_analysis_config(self, layout, node):
        """분석 노드 설정 UI - 통합 MCS 데이터용"""
        layout.addWidget(QLabel("분석 설정"))
        
        layout.addWidget(QLabel("분석 기간:"))
        period_combo = QComboBox()
        period_combo.addItems(["1시간", "4시간", "1일", "1주일", "1개월", "3개월"])
        period_combo.setCurrentText("1일")
        period_combo.setObjectName("analysis_period")
        layout.addWidget(period_combo)
        
        if "OHT" in node.name:
            layout.addWidget(QLabel("분석 항목:"))
            patterns = ["이동 경로", "정체 구간", "평균 속도", "가동률", "충돌 위험"]
            for i, pattern in enumerate(patterns):
                check = QCheckBox(pattern)
                check.setChecked(True)
                check.setObjectName(f"oht_pattern_{i}")
                layout.addWidget(check)
                
        elif "가동률" in node.name:
            layout.addWidget(QLabel("장비 타입:"))
            equip_combo = QComboBox()
            equip_combo.addItems(["전체", "포토", "식각", "증착", "CMP", "계측"])
            equip_combo.setObjectName("equipment_type")
            layout.addWidget(equip_combo)
            
            layout.addWidget(QLabel("가동률 기준 (%):"))
            rate_spin = QSpinBox()
            rate_spin.setRange(0, 100)
            rate_spin.setValue(85)
            rate_spin.setObjectName("target_rate")
            layout.addWidget(rate_spin)
            
        elif "병목" in node.name:
            layout.addWidget(QLabel("병목 판단 기준:"))
            bottleneck_spin = QSpinBox()
            bottleneck_spin.setRange(1, 100)
            bottleneck_spin.setValue(10)
            bottleneck_spin.setSuffix(" 대 이상 대기")
            bottleneck_spin.setObjectName("bottleneck_threshold")
            layout.addWidget(bottleneck_spin)
            
        elif "센서트렌드" in node.name:
            layout.addWidget(QLabel("분석할 센서 타입:"))
            sensor_types = ["진동", "온도", "압력", "파티클", "습도", "유량"]
            for i, sensor in enumerate(sensor_types):
                check = QCheckBox(sensor)
                check.setChecked(i < 3)  # 기본적으로 진동, 온도, 압력만 선택
                check.setObjectName(f"trend_sensor_{i}")
                layout.addWidget(check)
            
            layout.addWidget(QLabel("트렌드 분석 방법:"))
            trend_combo = QComboBox()
            trend_combo.addItems([
                "이동평균", "선형회귀", "계절성 분해", "이상치 빈도", 
                "피크 검출", "변화율 분석"
            ])
            trend_combo.setObjectName("trend_method")
            layout.addWidget(trend_combo)
            
            layout.addWidget(QLabel("알림 임계값 설정:"))
            alert_check = QCheckBox("트렌드 알림 사용")
            alert_check.setObjectName("use_trend_alert")
            layout.addWidget(alert_check)
            
    def browse_file(self, line_edit):
        """파일 찾아보기 대화상자"""
        filename, _ = QFileDialog.getOpenFileName(
            self, 
            "데이터 파일 선택", 
            "", 
            "Data Files (*.csv *.json *.xlsx);;All Files (*.*)"
        )
        if filename:
            line_edit.setText(filename)
            
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
                "connections": [],
                "memos": []
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
                        
            # 메모 정보 수집
            for item in self.scene.items():
                if isinstance(item, MemoItem):
                    data["memos"].append(item.get_data())
                        
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
                
            # 메모 생성
            if "memos" in data:
                for memo_data in data["memos"]:
                    memo = MemoItem()
                    memo.set_data(memo_data)
                    self.scene.addItem(memo)
                
            self.log(f"파이프라인을 불러왔습니다: {filename}")
            
    def delete_selected(self):
        """선택된 아이템 삭제"""
        for item in self.scene.selectedItems():
            if isinstance(item, Node):
                self.view.delete_node(item)
            elif isinstance(item, Connection):
                item.remove()
            elif isinstance(item, MemoItem):
                self.scene.removeItem(item)
                
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
                
            # 예측 모델 노드 확인 (LSTM, RNN, ARIMA 중 하나)
            model_nodes = [n for n in nodes if n.node_type == NodeType.MODEL]
            if not model_nodes:
                errors.append("예측 모델 노드가 최소 하나 필요합니다")
                
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
        
        # 노드 실행 순서 결정
        execution_order = self.determine_execution_order()
        
        # 실행 결과 저장
        node_outputs = {}
        
        # 순차적으로 노드 실행
        for node in execution_order:
            self.log(f"실행 중: {node.name}")
            
            # 입력 데이터 수집
            input_data = {}
            for port in node.input_ports:
                for connection in port.connections:
                    source_node = connection.start_port.parentItem()
                    if source_node in node_outputs:
                        input_data[source_node.node_type.value] = node_outputs[source_node]
            
            # 노드 실행
            output = self.execute_node(node, input_data)
            node_outputs[node] = output
        
        # 최종 결과 표시
        self.show_execution_results(node_outputs)
        
    def determine_execution_order(self):
        """노드 실행 순서 결정 (위상 정렬)"""
        nodes = [item for item in self.scene.items() if isinstance(item, Node)]
        
        # 진입 차수 계산
        in_degree = {node: 0 for node in nodes}
        for node in nodes:
            for port in node.input_ports:
                in_degree[node] += len(port.connections)
        
        # 진입 차수가 0인 노드부터 시작
        queue = [node for node in nodes if in_degree[node] == 0]
        execution_order = []
        
        while queue:
            current = queue.pop(0)
            execution_order.append(current)
            
            # 연결된 다음 노드들의 진입 차수 감소
            for port in current.output_ports:
                for connection in port.connections:
                    next_node = connection.end_port.parentItem()
                    in_degree[next_node] -= 1
                    if in_degree[next_node] == 0:
                        queue.append(next_node)
        
        return execution_order
        
    def execute_node(self, node, input_data):
        """개별 노드 실행 - 통합 MCS 시뮬레이션"""
        import random
        import numpy as np
        from datetime import datetime, timedelta
        
        output = {}
        
        if node.node_type == NodeType.DATA:
            # 데이터 로드 시뮬레이션
            if "MCS" in node.name:
                output = {
                    "data": "통합 MCS 데이터 로드됨",
                    "records": 10000,
                    "time_range": "2025-07-01 00:00 ~ 2025-07-30 23:59",
                    "fab_lines": ["FAB1", "FAB2", "FAB3"],
                    "event_types": ["TRANSFER", "PROCESS", "SENSOR_UPDATE", "ALARM"],
                    "sensor_records": 3234,
                    "alarm_records": 156,
                    "columns": ["timestamp", "event_type", "equipment_id", "sensor_vibration_mm_s", 
                               "sensor_temperature_c", "sensor_pressure_torr", "sensor_particle_count"]
                }
                
        elif node.node_type == NodeType.PREPROCESS:
            # 전처리 실행
            if "센서 추출" in node.name:
                output = {
                    "extracted_data": "센서 데이터 추출 완료",
                    "sensor_records": 3234,
                    "sensors": {
                        "vibration": 3234,
                        "temperature": 3100,
                        "pressure": 2890,
                        "particle": 2450
                    },
                    "anomaly_count": 89,
                    "equipment_count": 45
                }
            elif "이벤트 필터" in node.name:
                output = {
                    "filtered_data": "이벤트 필터링 완료",
                    "original_records": 10000,
                    "filtered_records": 6543,
                    "filtered_events": node.settings.get("filtered_events", ["SENSOR_UPDATE", "PROCESS"])
                }
            elif "이상치" in node.name:
                output = {
                    "cleaned_data": "이상치 제거 완료",
                    "removed": 156,
                    "method": node.settings.get("method", "IQR"),
                    "anomaly_details": {
                        "vibration": 45,
                        "temperature": 67,
                        "particle": 44
                    }
                }
            elif "시간별" in node.name:
                output = {
                    "aggregated_data": "시간별 집계 완료",
                    "time_unit": node.settings.get("time_unit", "1시간"),
                    "records": 720,  # 30일 * 24시간
                    "aggregation": node.settings.get("aggregation", "평균")
                }
            elif "베이별" in node.name:
                output = {
                    "grouped_data": "베이별 분류 완료",
                    "bays": ["PHOTO", "ETCH", "CVD", "CMP"],
                    "distribution": {"PHOTO": 35, "ETCH": 30, "CVD": 20, "CMP": 15}
                }
            elif "장비별" in node.name:
                output = {
                    "equipment_grouped": "장비별 분류 완료",
                    "equipment_count": 45,
                    "top_equipment": [
                        {"id": "ASML_NXT_1980", "events": 1234},
                        {"id": "LAM_KIYO_01", "events": 987},
                        {"id": "AMAT_CENTURA_01", "events": 876}
                    ]
                }
                
        elif node.node_type == NodeType.VECTOR:
            # RAG 벡터 저장
            output = {
                "vector_store": node.settings.get("vector_store", "ChromaDB"),
                "embeddings": "벡터 저장 완료",
                "dimension": node.settings.get("vector_dim", 768),
                "total_vectors": 1234,
                "vector_types": {
                    "event_patterns": 456,
                    "sensor_patterns": 678,
                    "alarm_patterns": 100
                }
            }
            
        elif node.node_type == NodeType.MODEL:
            # 딥러닝 모델 실행
            if "LSTM" in node.name:
                # LSTM 모델로 이송 시간 예측
                prediction_target = node.settings.get("prediction_target", "이송 시간")
                period = node.settings.get("period", "1시간")
                
                # 시계열 예측 생성
                base_time = 185  # 기본 이송 시간(초)
                predictions = []
                confidence_scores = []
                
                # 시간대별 예측 생성
                time_periods = ["10분", "30분", "1시간", "4시간", "1일"]
                for tp in time_periods:
                    # 시간이 길수록 불확실성 증가
                    uncertainty = time_periods.index(tp) * 0.05
                    pred_time = base_time + random.gauss(0, 10 + uncertainty * 20)
                    conf = max(0.70, 0.95 - uncertainty)
                    
                    predictions.append({
                        "period": tp,
                        "predicted_time": round(pred_time, 1),
                        "confidence": round(conf, 3),
                        "range": f"{round(pred_time * 0.9, 1)} ~ {round(pred_time * 1.1, 1)}초"
                    })
                
                output = {
                    "model": "LSTM (Long Short-Term Memory)",
                    "prediction_target": prediction_target,
                    "model_accuracy": 94.2,
                    "training_epochs": 100,
                    "predictions": predictions,
                    "key_factors": [
                        "현재 OHT 가동률: 87.3%",
                        "PHOTO 베이 대기: 15대",
                        "최근 알람 빈도: 증가 추세"
                    ],
                    "recommendation": "PHOTO 베이 우회 경로 활성화 권장",
                    "model_params": {
                        "layers": node.settings.get("layers", 3),
                        "units": node.settings.get("units", 128),
                        "dropout": 0.2,
                        "optimizer": "adam"
                    }
                }
                
            elif "RNN" in node.name:
                # RNN 모델로 병목 예측
                prediction_target = node.settings.get("prediction_target", "병목 발생 확률")
                
                # 베이별 병목 예측
                bays = ["PHOTO", "ETCH", "CVD", "PVD", "CMP", "CLEAN", "DIFF", "IMPLANT"]
                bottleneck_predictions = []
                
                for bay in bays:
                    # 각 베이별 병목 확률 계산
                    base_prob = random.uniform(0.1, 0.8)
                    if bay in ["PHOTO", "ETCH"]:  # 주요 병목 구간
                        base_prob += 0.3
                    
                    prob = min(0.95, base_prob)
                    severity = "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.4 else "LOW"
                    
                    bottleneck_predictions.append({
                        "bay": bay,
                        "probability": round(prob, 3),
                        "severity": severity,
                        "queue_prediction": round(prob * 20),  # 예상 대기 대수
                        "impact_time": f"{round(prob * 30)}분"  # 예상 지연 시간
                    })
                
                # 시간대별 병목 패턴
                hourly_pattern = []
                for hour in range(24):
                    peak_hours = [10, 11, 14, 15, 16]  # 오전/오후 피크 시간
                    if hour in peak_hours:
                        intensity = random.uniform(0.7, 0.9)
                    else:
                        intensity = random.uniform(0.3, 0.6)
                    
                    hourly_pattern.append({
                        "hour": f"{hour:02d}:00",
                        "bottleneck_intensity": round(intensity, 2)
                    })
                
                output = {
                    "model": "RNN (Recurrent Neural Network)",
                    "prediction_target": prediction_target,
                    "model_accuracy": 89.7,
                    "bottleneck_predictions": sorted(bottleneck_predictions, 
                                                   key=lambda x: x["probability"], 
                                                   reverse=True),
                    "hourly_pattern": hourly_pattern,
                    "peak_times": ["10:00-12:00", "14:00-17:00"],
                    "critical_bays": [bp["bay"] for bp in bottleneck_predictions if bp["severity"] == "HIGH"],
                    "recommendations": [
                        "PHOTO 베이 처리 속도 향상 필요",
                        "14:00-17:00 시간대 추가 OHT 투입 권장",
                        "ETCH 베이 예방정비 일정 조정 검토"
                    ],
                    "model_params": {
                        "layers": node.settings.get("layers", 3),
                        "units": node.settings.get("units", 128),
                        "sequence_length": 24,
                        "batch_size": 32
                    }
                }
                
            elif "ARIMA" in node.name:
                # ARIMA 모델로 처리량 예측
                prediction_target = node.settings.get("prediction_target", "시간당 처리량")
                
                # ARIMA 파라미터
                p = node.settings.get("p", 2)  # 자기회귀
                d = node.settings.get("d", 1)  # 차분
                q = node.settings.get("q", 2)  # 이동평균
                
                # 과거 처리량 데이터 (시뮬레이션)
                historical_data = []
                base_throughput = 150  # 시간당 기본 처리량
                
                for i in range(168):  # 지난 1주일 (168시간)
                    # 주기적 패턴 + 노이즈
                    hour_of_day = i % 24
                    day_of_week = (i // 24) % 7
                    
                    # 시간대별 변동
                    if 6 <= hour_of_day <= 22:  # 주간
                        throughput = base_throughput + random.gauss(0, 10)
                    else:  # 야간
                        throughput = base_throughput * 0.7 + random.gauss(0, 5)
                    
                    # 주말 효과
                    if day_of_week in [5, 6]:  # 토, 일
                        throughput *= 0.8
                    
                    historical_data.append({
                        "hour": i,
                        "throughput": round(throughput, 1)
                    })
                
                # 미래 예측
                forecast_periods = 48  # 향후 48시간
                forecasts = []
                
                for i in range(forecast_periods):
                    # ARIMA 모델 시뮬레이션
                    trend = base_throughput
                    seasonal = 10 * np.sin(2 * np.pi * i / 24)  # 일일 주기
                    ar_component = 0.7 * (historical_data[-1]["throughput"] - base_throughput) if historical_data else 0
                    ma_component = random.gauss(0, 5)
                    
                    forecast_value = trend + seasonal + ar_component + ma_component
                    
                    # 신뢰 구간
                    std_error = 5 + i * 0.2  # 시간이 지날수록 불확실성 증가
                    lower_bound = forecast_value - 1.96 * std_error
                    upper_bound = forecast_value + 1.96 * std_error
                    
                    forecasts.append({
                        "hour": i + 1,
                        "forecast": round(forecast_value, 1),
                        "lower_95": round(lower_bound, 1),
                        "upper_95": round(upper_bound, 1),
                        "confidence": round(max(0.7, 0.95 - i * 0.01), 2)
                    })
                
                # 통계 요약
                avg_forecast = np.mean([f["forecast"] for f in forecasts])
                
                output = {
                    "model": "ARIMA (AutoRegressive Integrated Moving Average)",
                    "model_notation": f"ARIMA({p},{d},{q})",
                    "prediction_target": prediction_target,
                    "model_metrics": {
                        "aic": 1234.56,
                        "bic": 1256.78,
                        "mape": 5.2,  # Mean Absolute Percentage Error
                        "rmse": 7.8   # Root Mean Square Error
                    },
                    "historical_summary": {
                        "mean": round(np.mean([h["throughput"] for h in historical_data]), 1),
                        "std": round(np.std([h["throughput"] for h in historical_data]), 1),
                        "min": round(min([h["throughput"] for h in historical_data]), 1),
                        "max": round(max([h["throughput"] for h in historical_data]), 1)
                    },
                    "forecast_summary": {
                        "next_24h_avg": round(np.mean([f["forecast"] for f in forecasts[:24]]), 1),
                        "next_48h_avg": round(avg_forecast, 1),
                        "peak_forecast": round(max([f["forecast"] for f in forecasts]), 1),
                        "valley_forecast": round(min([f["forecast"] for f in forecasts]), 1)
                    },
                    "forecasts": forecasts[:24],  # 처음 24시간만 표시
                    "insights": [
                        f"향후 24시간 평균 처리량: {round(np.mean([f['forecast'] for f in forecasts[:24]]), 1)} 웨이퍼/시간",
                        f"예상 일일 생산량: {round(np.sum([f['forecast'] for f in forecasts[:24]]), 0)} 웨이퍼",
                        "주간 시간대(06:00-22:00) 처리량이 야간 대비 40% 높음",
                        "주말 처리량은 평일 대비 20% 감소 예상"
                    ],
                    "recommendations": [
                        "야간 시간대 생산성 향상 방안 검토 필요",
                        "주말 가동률 개선을 위한 인력 배치 최적화",
                        "피크 시간대 병목 현상 대비 버퍼 확보"
                    ]
                }
                
            elif "센서이상탐지" in node.name:
                # 센서 이상 탐지 모델
                target = node.settings.get("prediction_target", "복합 이상")
                algorithm = node.settings.get("anomaly_algorithm", "Isolation Forest")
                sensitivity = node.settings.get("sensitivity", 0.8)
                
                # 이상 탐지 결과 생성
                anomalies = []
                equipment_list = [
                    "ASML_NXT_1980", "ASML_NXT_2050", "LAM_KIYO_01", 
                    "AMAT_CENTURA_01", "TEL_ETCH_01", "AMAT_ENDURA_01"
                ]
                
                for eq in equipment_list:
                    # 각 장비별 센서 이상 확률
                    if random.random() < 0.3:  # 30% 확률로 이상 발생
                        sensor_type = random.choice(["vibration", "temperature", "pressure", "particle"])
                        
                        # 센서별 정상 범위
                        normal_ranges = {
                            "vibration": (0.5, 2.0),
                            "temperature": (20, 30),
                            "pressure": (750, 760),
                            "particle": (10, 100)
                        }
                        
                        # 이상값 생성
                        normal_min, normal_max = normal_ranges[sensor_type]
                        if random.random() < 0.5:  # 하한 이탈
                            value = normal_min * random.uniform(0.3, 0.8)
                        else:  # 상한 이탈
                            value = normal_max * random.uniform(1.2, 2.0)
                        
                        # 이상 점수 계산
                        anomaly_score = min(0.99, abs(value - (normal_min + normal_max) / 2) / 
                                          ((normal_max - normal_min) / 2) * sensitivity)
                        
                        anomalies.append({
                            "equipment_id": eq,
                            "sensor_type": sensor_type,
                            "current_value": round(value, 2),
                            "normal_range": f"{normal_min} ~ {normal_max}",
                            "anomaly_score": round(anomaly_score, 3),
                            "severity": "CRITICAL" if anomaly_score > 0.8 else "HIGH" if anomaly_score > 0.6 else "MEDIUM",
                            "detected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "pattern": random.choice(["급격한 상승", "지속적 증가", "주기적 이상", "간헐적 스파이크"])
                        })
                
                # 이상 패턴 분석
                pattern_analysis = {
                    "total_anomalies": len(anomalies),
                    "critical_count": len([a for a in anomalies if a["severity"] == "CRITICAL"]),
                    "sensor_distribution": {},
                    "equipment_health_score": {}
                }
                
                # 센서별 분포
                for sensor in ["vibration", "temperature", "pressure", "particle"]:
                    count = len([a for a in anomalies if a["sensor_type"] == sensor])
                    pattern_analysis["sensor_distribution"][sensor] = count
                
                # 장비별 건강도 점수
                for eq in equipment_list:
                    eq_anomalies = [a for a in anomalies if a["equipment_id"] == eq]
                    if eq_anomalies:
                        health_score = max(0, 100 - len(eq_anomalies) * 20 - 
                                         sum([a["anomaly_score"] * 10 for a in eq_anomalies]))
                    else:
                        health_score = 100
                    pattern_analysis["equipment_health_score"][eq] = round(health_score, 1)
                
                output = {
                    "model": "센서 이상탐지 (Anomaly Detection)",
                    "algorithm": algorithm,
                    "target": target,
                    "sensitivity": f"{int(sensitivity * 100)}%",
                    "detection_results": {
                        "anomalies_detected": anomalies,
                        "pattern_analysis": pattern_analysis,
                        "risk_assessment": {
                            "immediate_action_required": [a["equipment_id"] for a in anomalies if a["severity"] == "CRITICAL"],
                            "monitoring_required": [a["equipment_id"] for a in anomalies if a["severity"] == "HIGH"],
                            "maintenance_schedule": [a["equipment_id"] for a in anomalies if a["severity"] == "MEDIUM"]
                        }
                    },
                    "predictive_maintenance": {
                        "failure_probability_24h": {eq: round(min(0.95, len([a for a in anomalies if a["equipment_id"] == eq]) * 0.25), 2) 
                                                   for eq in equipment_list if any(a["equipment_id"] == eq for a in anomalies)},
                        "recommended_actions": [
                            f"{a['equipment_id']}: {a['sensor_type']} 센서 점검 필요 (이상 점수: {a['anomaly_score']})"
                            for a in sorted(anomalies, key=lambda x: x['anomaly_score'], reverse=True)[:3]
                        ]
                    },
                    "model_performance": {
                        "precision": 0.92,
                        "recall": 0.88,
                        "f1_score": 0.90,
                        "false_positive_rate": 0.08
                    }
                }
                
        elif node.node_type == NodeType.ANALYSIS:
            # 패턴 분석
            if "OHT" in node.name:
                output = {
                    "patterns": [
                        "경로 A-B 구간 주기적 정체",
                        "오후 2-4시 이송 지연 증가",
                        "FAB2 라인 OHT 가동률 저하"
                    ],
                    "avg_transfer_time": 185,  # 초
                    "congestion_points": ["NODE_A23", "NODE_B45"],
                    "utilization": 87.3
                }
            elif "가동률" in node.name:
                output = {
                    "equipment_utilization": {
                        "PHOTO": 92.3,
                        "ETCH": 87.5,
                        "CVD": 78.2,
                        "PVD": 85.1,
                        "CMP": 89.7
                    },
                    "below_target": ["CVD"],
                    "maintenance_required": ["CVD_03", "ETCH_07"]
                }
            elif "병목" in node.name:
                output = {
                    "bottlenecks": [
                        {"location": "PHOTO_BAY", "queue_size": 15, "wait_time": 23},
                        {"location": "CVD_LOADPORT_02", "queue_size": 8, "wait_time": 12}
                    ],
                    "recommendations": ["PHOTO 베이 처리 속도 향상 필요", "CVD 로드포트 추가 검토"]
                }
            elif "센서트렌드" in node.name:
                output = {
                    "trends": {
                        "vibration": {"trend": "increasing", "rate": 0.05, "alert": True},
                        "temperature": {"trend": "stable", "rate": 0.01, "alert": False},
                        "particle": {"trend": "cyclic", "period": "24h", "alert": False}
                    },
                    "predictions": {
                        "vibration_24h": 2.8,
                        "maintenance_window": "72시간 이내"
                    },
                    "method": node.settings.get("trend_method", "이동평균")
                }
                
        return output
        
    def show_execution_results(self, node_outputs):
        """실행 결과 표시 - 딥러닝 모델 중심"""
        result_dialog = QDialog(self)
        result_dialog.setWindowTitle("딥러닝 예측 결과")
        result_dialog.setModal(True)
        layout = QVBoxLayout()
        
        # 탭 위젯으로 결과 표시
        tabs = QTabWidget()
        
        # 딥러닝 모델 결과 수집
        lstm_result = None
        rnn_result = None
        arima_result = None
        anomaly_result = None
        
        for node, output in node_outputs.items():
            if node.node_type == NodeType.MODEL:
                if "LSTM" in node.name:
                    lstm_result = output
                elif "RNN" in node.name:
                    rnn_result = output
                elif "ARIMA" in node.name:
                    arima_result = output
                elif "센서이상탐지" in node.name:
                    anomaly_result = output
        
        # 종합 예측 탭
        summary_text = QTextEdit()
        summary_text.setReadOnly(True)
        
        summary_content = "=" * 60 + "\n"
        summary_content += "반도체 FAB MCS 딥러닝 예측 종합 결과\n"
        summary_content += "=" * 60 + "\n\n"
        summary_content += f"분석 시점: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # LSTM 결과
        if lstm_result:
            summary_content += "【 LSTM 이송시간 예측 】\n"
            summary_content += f"모델 정확도: {lstm_result['model_accuracy']}%\n\n"
            for pred in lstm_result['predictions'][:3]:  # 주요 3개만
                summary_content += f"  • {pred['period']} 후: {pred['predicted_time']}초 "
                summary_content += f"(신뢰도: {pred['confidence']*100:.1f}%)\n"
            summary_content += f"\n권장사항: {lstm_result['recommendation']}\n"
            summary_content += "-" * 40 + "\n\n"
        
        # RNN 결과
        if rnn_result:
            summary_content += "【 RNN 병목 예측 】\n"
            summary_content += f"모델 정확도: {rnn_result['model_accuracy']}%\n\n"
            summary_content += "주요 병목 구간:\n"
            for bp in rnn_result['bottleneck_predictions'][:3]:  # 상위 3개
                summary_content += f"  • {bp['bay']}: {bp['probability']*100:.1f}% "
                summary_content += f"({bp['severity']}) - 예상 대기: {bp['queue_prediction']}대\n"
            summary_content += f"\n피크 시간대: {', '.join(rnn_result['peak_times'])}\n"
            summary_content += "-" * 40 + "\n\n"
        
        # ARIMA 결과
        if arima_result:
            summary_content += "【 ARIMA 처리량 예측 】\n"
            summary_content += f"모델: {arima_result['model_notation']}\n"
            summary_content += f"MAPE: {arima_result['model_metrics']['mape']}%\n\n"
            summary_content += f"향후 24시간 평균 처리량: {arima_result['forecast_summary']['next_24h_avg']} 웨이퍼/시간\n"
            summary_content += f"예상 일일 생산량: {round(arima_result['forecast_summary']['next_24h_avg'] * 24)} 웨이퍼\n"
            summary_content += "\n주요 인사이트:\n"
            for insight in arima_result['insights'][:2]:
                summary_content += f"  • {insight}\n"
            summary_content += "-" * 40 + "\n\n"
        
        # 센서 이상탐지 결과
        if anomaly_result:
            summary_content += "【 센서 이상탐지 】\n"
            summary_content += f"알고리즘: {anomaly_result['algorithm']}\n"
            summary_content += f"탐지 민감도: {anomaly_result['sensitivity']}\n\n"
            
            results = anomaly_result['detection_results']
            summary_content += f"총 이상 감지: {results['pattern_analysis']['total_anomalies']}건\n"
            summary_content += f"위험 수준: {results['pattern_analysis']['critical_count']}건\n\n"
            
            if results['risk_assessment']['immediate_action_required']:
                summary_content += "즉시 조치 필요 장비:\n"
                for eq in results['risk_assessment']['immediate_action_required']:
                    summary_content += f"  • {eq}\n"
            summary_content += "\n"
        
        summary_content += "=" * 60 + "\n"
        summary_content += "【 종합 권장사항 】\n"
        summary_content += "=" * 60 + "\n"
        
        recommendations = []
        
        # 각 모델의 권장사항 통합
        if lstm_result:
            recommendations.append(lstm_result.get('recommendation', ''))
        if rnn_result and 'recommendations' in rnn_result:
            recommendations.extend(rnn_result['recommendations'][:2])
        if arima_result and 'recommendations' in arima_result:
            recommendations.extend(arima_result['recommendations'][:2])
        if anomaly_result:
            pm = anomaly_result.get('predictive_maintenance', {})
            if 'recommended_actions' in pm:
                recommendations.extend(pm['recommended_actions'][:2])
        
        for i, rec in enumerate(recommendations, 1):
            if rec:
                summary_content += f"{i}. {rec}\n"
        
        summary_text.setPlainText(summary_content)
        tabs.addTab(summary_text, "종합 예측 결과")
        
        # LSTM 상세 탭
        if lstm_result:
            lstm_text = QTextEdit()
            lstm_text.setReadOnly(True)
            lstm_content = "LSTM 이송시간 예측 상세 결과\n" + "=" * 50 + "\n\n"
            lstm_content += json.dumps(lstm_result, indent=2, ensure_ascii=False)
            lstm_text.setPlainText(lstm_content)
            tabs.addTab(lstm_text, "LSTM 예측")
        
        # RNN 상세 탭
        if rnn_result:
            rnn_text = QTextEdit()
            rnn_text.setReadOnly(True)
            rnn_content = "RNN 병목 예측 상세 결과\n" + "=" * 50 + "\n\n"
            rnn_content += json.dumps(rnn_result, indent=2, ensure_ascii=False)
            rnn_text.setPlainText(rnn_content)
            tabs.addTab(rnn_text, "RNN 예측")
        
        # ARIMA 상세 탭
        if arima_result:
            arima_text = QTextEdit()
            arima_text.setReadOnly(True)
            arima_content = "ARIMA 처리량 예측 상세 결과\n" + "=" * 50 + "\n\n"
            arima_content += json.dumps(arima_result, indent=2, ensure_ascii=False)
            arima_text.setPlainText(arima_content)
            tabs.addTab(arima_text, "ARIMA 예측")
        
        # 센서 이상탐지 상세 탭
        if anomaly_result:
            anomaly_text = QTextEdit()
            anomaly_text.setReadOnly(True)
            anomaly_content = "센서 이상탐지 상세 결과\n" + "=" * 50 + "\n\n"
            anomaly_content += json.dumps(anomaly_result, indent=2, ensure_ascii=False)
            anomaly_text.setPlainText(anomaly_content)
            tabs.addTab(anomaly_text, "이상탐지")
        
        # 전체 노드 실행 결과 탭
        all_results_text = QTextEdit()
        all_results_text.setReadOnly(True)
        all_content = "전체 파이프라인 실행 결과\n" + "=" * 50 + "\n\n"
        
        for node, output in node_outputs.items():
            all_content += f"【{node.name}】\n"
            all_content += json.dumps(output, indent=2, ensure_ascii=False)
            all_content += "\n\n" + "-" * 40 + "\n\n"
        
        all_results_text.setPlainText(all_content)
        tabs.addTab(all_results_text, "전체 결과")
        
        layout.addWidget(tabs)
        
        # 버튼 레이아웃
        button_layout = QHBoxLayout()
        
        # 결과 저장 버튼
        save_btn = QPushButton("결과 저장")
        save_btn.clicked.connect(lambda: self.save_results(node_outputs))
        button_layout.addWidget(save_btn)
        
        # 리포트 생성 버튼
        report_btn = QPushButton("리포트 생성")
        report_btn.clicked.connect(lambda: self.generate_report(node_outputs))
        button_layout.addWidget(report_btn)
        
        # 닫기 버튼
        close_btn = QPushButton("닫기")
        close_btn.clicked.connect(result_dialog.accept)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
        
        result_dialog.setLayout(layout)
        result_dialog.resize(900, 700)
        result_dialog.exec_()
        
        self.log("파이프라인 실행 완료")
        
    def save_results(self, results):
        """예측 결과 저장"""
        filename, _ = QFileDialog.getSaveFileName(
            self, 
            "예측 결과 저장", 
            f"mcs_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON Files (*.json)"
        )
        if filename:
            # 결과를 직렬화 가능한 형태로 변환
            serializable_results = {}
            for node, output in results.items():
                serializable_results[node.name] = output
                
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            
            self.log(f"예측 결과가 저장되었습니다: {filename}")
            QMessageBox.information(self, "저장 완료", "예측 결과가 성공적으로 저장되었습니다.")
            
    def generate_report(self, results):
        """예측 리포트 생성"""
        filename, _ = QFileDialog.getSaveFileName(
            self, 
            "리포트 저장", 
            f"mcs_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text Files (*.txt)"
        )
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("반도체 FAB MCS 딥러닝 예측 리포트\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"생성 일시: {datetime.now().strftime('%Y년 %m월 %d일 %H:%M:%S')}\n\n")
                
                # 각 모델별 요약
                for node, output in results.items():
                    if node.node_type == NodeType.MODEL:
                        f.write(f"\n【{node.name}】\n")
                        f.write("-" * 60 + "\n")
                        
                        if "LSTM" in node.name:
                            f.write(f"모델 정확도: {output.get('model_accuracy', 'N/A')}%\n")
                            if 'predictions' in output:
                                f.write("\n예측 결과:\n")
                                for pred in output['predictions'][:5]:
                                    f.write(f"  - {pred['period']}: {pred['predicted_time']}초 "
                                           f"(신뢰도: {pred['confidence']*100:.1f}%)\n")
                                           
                        elif "RNN" in node.name:
                            f.write(f"모델 정확도: {output.get('model_accuracy', 'N/A')}%\n")
                            if 'bottleneck_predictions' in output:
                                f.write("\n병목 예측:\n")
                                for bp in output['bottleneck_predictions'][:5]:
                                    f.write(f"  - {bp['bay']}: {bp['probability']*100:.1f}% "
                                           f"({bp['severity']})\n")
                                           
                        elif "ARIMA" in node.name:
                            if 'forecast_summary' in output:
                                f.write(f"향후 24시간 평균: {output['forecast_summary']['next_24h_avg']} 웨이퍼/시간\n")
                                f.write(f"예상 일일 생산량: {output['forecast_summary']['next_24h_avg'] * 24:.0f} 웨이퍼\n")
                                
                        elif "센서이상탐지" in node.name:
                            if 'detection_results' in output:
                                results = output['detection_results']
                                f.write(f"총 이상 감지: {results['pattern_analysis']['total_anomalies']}건\n")
                                f.write(f"위험 수준: {results['pattern_analysis']['critical_count']}건\n")
                
                f.write("\n" + "=" * 80 + "\n")
                f.write("종합 분석 결과 및 권장사항\n")
                f.write("=" * 80 + "\n")
                f.write("\n이 리포트는 딥러닝 모델을 통한 예측 결과를 담고 있습니다.\n")
                f.write("실제 운영 시에는 현장 상황을 고려하여 적용하시기 바랍니다.\n")
            
            self.log(f"리포트가 생성되었습니다: {filename}")
            QMessageBox.information(self, "리포트 생성 완료", "예측 리포트가 성공적으로 생성되었습니다.")
        
    def show_help(self):
        """도움말 표시"""
        help_text = """반도체 FAB MCS 딥러닝 예측 시스템 도움말
        
시스템 개요:
- LSTM, RNN, ARIMA 딥러닝 모델을 사용한 예측 시스템
- 통합 MCS 데이터 기반 실시간 예측 및 분석

주요 기능:
1. LSTM: 이송 시간 예측
2. RNN: 병목 구간 예측
3. ARIMA: 처리량 예측
4. 센서 이상탐지: 장비 고장 예측

마우스 조작:
- 왼쪽 클릭: 노드/연결선 선택
- 왼쪽 드래그: 선택 영역 생성
- 가운데 버튼 드래그: 화면 이동
- 스페이스 + 왼쪽 드래그: 화면 이동
- 마우스 휠: 확대/축소
- 더블클릭: 노드 설정
- 우클릭: 컨텍스트 메뉴

키보드 단축키:
- Delete: 선택 항목 삭제
- F: 전체 화면 보기
- Ctrl+N: 새 파일
- Ctrl+O: 열기
- Ctrl+S: 저장
- Ctrl+M: 메모 추가
- Ctrl+Q: 종료

노드 연결:
- 출력 포트(오른쪽)에서 입력 포트(왼쪽)로 드래그
- 연결선 클릭 후 Delete로 삭제

파이프라인 구성:
1. 데이터 입력 → 2. 전처리 → 3. 모델 → 4. 분석
각 단계를 연결하여 예측 파이프라인 구성"""
        
        QMessageBox.information(self, "도움말", help_text)
        
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
        elif selected and isinstance(selected[0], MemoItem):
            memo = selected[0]
            info = f"""메모 정보
---------
ID: {memo.memo_id}
위치: ({int(memo.x())}, {int(memo.y())})
크기: {int(memo.rect().width())} x {int(memo.rect().height())}

내용:
{memo.text_item.toPlainText()}"""
            
            self.properties_widget.setPlainText(info)
        else:
            self.properties_widget.setPlainText("노드나 메모를 선택하세요")
            
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
    
    window = SemiconductorMCSSystem()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()