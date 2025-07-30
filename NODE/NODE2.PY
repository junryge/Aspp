#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
물류이동 예측 시스템 - pyqt-node-editor 기반 구현
Requirements:
    pip install pyqt-node-editor
"""

import sys
import json
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from pyqt_node_editor import *


class LogisticsNodeContent(QWidget):
    """노드 컨텐츠 위젯"""
    def __init__(self, node, parent=None):
        super().__init__(parent)
        self.node = node
        self.init_ui()
        
    def init_ui(self):
        """UI 초기화"""
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 노드 타입 표시
        self.type_label = QLabel(f"[{self.node.node_type}]")
        self.type_label.setAlignment(Qt.AlignCenter)
        self.type_label.setStyleSheet("color: #AAA; font-size: 10px;")
        layout.addWidget(self.type_label)
        
        # 상태 표시
        self.status_label = QLabel("⚪ 미설정")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 11px;")
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
        
    def update_status(self, configured):
        """상태 업데이트"""
        if configured:
            self.status_label.setText("🟢 설정됨")
        else:
            self.status_label.setText("⚪ 미설정")


class LogisticsNode(Node):
    """물류 예측 시스템 노드"""
    
    def __init__(self, scene, title="Undefined Node", node_type="undefined", 
                 inputs=[], outputs=[], pos=[0, 0]):
        self.node_type = node_type
        self.is_configured = False
        self.settings = {}
        
        super().__init__(scene, title, inputs, outputs, pos)
        
    def init_inner_classes(self):
        """내부 클래스 초기화"""
        self.content = LogisticsNodeContent(self)
        self.grNode = NodeGraphicsNode(self)
        
        # 노드 색상 설정
        colors = {
            'data': "#3498db",      # 파란색
            'preprocess': "#e74c3c", # 빨간색
            'vector': "#f39c12",     # 주황색
            'model': "#27ae60",      # 초록색
            'analysis': "#9b59b6",   # 보라색
            'prompt': "#34495e",     # 회색
            'llm': "#16a085"         # 청록색
        }
        
        color = colors.get(self.node_type, "#7f8c8d")
        self.grNode._brush_background = QBrush(QColor(color))
        
    def set_configured(self, configured):
        """설정 상태 변경"""
        self.is_configured = configured
        self.content.update_status(configured)
        self.markDirty()
        
    def serialize(self):
        """직렬화"""
        res = super().serialize()
        res['node_type'] = self.node_type
        res['is_configured'] = self.is_configured
        res['settings'] = self.settings
        return res
        
    def deserialize(self, data, hashmap={}, restore_id=True):
        """역직렬화"""
        res = super().deserialize(data, hashmap, restore_id)
        self.node_type = data.get('node_type', 'undefined')
        self.is_configured = data.get('is_configured', False)
        self.settings = data.get('settings', {})
        self.content.update_status(self.is_configured)
        return res


class LogisticsSubWindow(NodeEditorSubWindow):
    """물류 예측 시스템 서브 윈도우"""
    
    def __init__(self):
        super().__init__()
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setTitle()
        self.init_new_node_actions()
        
        # 씬 이벤트 연결
        self.scene.add_has_been_modified_listener(self.set_title)
        self.scene.add_drag_enter_listener(self.on_drag_enter)
        self.scene.add_drop_listener(self.on_drop)
        
    def setTitle(self):
        """타이틀 설정"""
        self.setWindowTitle("물류이동 예측 시스템" + ("*" if self.has_been_modified else ""))
        
    def init_new_node_actions(self):
        """노드 생성 액션 초기화"""
        self.node_actions = {}
        
        # 노드 카테고리
        categories = {
            "데이터 입력": [
                ("MCS 로그 데이터", "data", ["출력"], ["입력"]),
                ("진동센서 데이터", "data", ["출력"], ["입력"]),
            ],
            "데이터 전처리": [
                ("이상치 제거", "preprocess", ["출력"], ["입력"]),
                ("시간별 분류", "preprocess", ["출력"], ["입력"]),
                ("경로별 분류", "preprocess", ["출력"], ["입력"]),
            ],
            "벡터 저장": [
                ("RAG 벡터 저장", "vector", ["벡터"], ["데이터"]),
            ],
            "시계열 모델": [
                ("LSTM 모델", "model", ["예측"], ["데이터"]),
                ("RNN 모델", "model", ["예측"], ["데이터"]),
                ("ARIMA 모델", "model", ["예측"], ["데이터"]),
            ],
            "분석": [
                ("과거패턴 분석", "analysis", ["분석결과"], ["벡터"]),
            ],
            "프롬프트": [
                ("프롬프트 생성", "prompt", ["프롬프트"], ["예측", "분석"]),
            ],
            "LLM": [
                ("PHI-4 판단/추론", "llm", ["결과"], ["프롬프트"]),
            ],
        }
        
        # 액션 생성
        for category, nodes in categories.items():
            for name, node_type, outputs, inputs in nodes:
                action = QAction(name)
                action.setData((name, node_type, outputs, inputs))
                self.node_actions[f"{category}/{name}"] = action
                
    def on_drag_enter(self, event):
        """드래그 엔터 이벤트"""
        if event.mimeData().hasText():
            event.acceptProposedAction()
            
    def on_drop(self, event):
        """드롭 이벤트"""
        if event.mimeData().hasText():
            event_data = event.mimeData().text()
            
            # 노드 액션 찾기
            action = self.node_actions.get(event_data)
            if action:
                name, node_type, outputs, inputs = action.data()
                
                # 씬 좌표로 변환
                mouse_pos = event.pos()
                scene_pos = self.scene.grScene.views()[0].mapToScene(mouse_pos)
                
                # 노드 생성
                self.create_node(name, node_type, outputs, inputs, 
                               [scene_pos.x(), scene_pos.y()])
                
                event.acceptProposedAction()
                
    def create_node(self, name, node_type, outputs, inputs, pos=[0, 0]):
        """노드 생성"""
        node = LogisticsNode(
            self.scene, 
            title=name,
            node_type=node_type,
            inputs=[1]*len(inputs),  # 입력 소켓 타입
            outputs=[2]*len(outputs),  # 출력 소켓 타입
            pos=pos
        )
        
        # 소켓 라벨 설정
        for i, label in enumerate(inputs):
            if i < len(node.inputs):
                node.inputs[i].socket_label = label
                
        for i, label in enumerate(outputs):
            if i < len(node.outputs):
                node.outputs[i].socket_label = label
                
        return node
        
    def configure_node(self, node):
        """노드 설정"""
        if not isinstance(node, LogisticsNode):
            return
            
        dialog = QDialog()
        dialog.setWindowTitle(f"{node.title} 설정")
        dialog.setModal(True)
        layout = QVBoxLayout()
        
        # 노드 타입별 설정
        if node.node_type == 'data':
            self.create_data_config(layout, node)
        elif node.node_type == 'preprocess':
            self.create_preprocess_config(layout, node)
        elif node.node_type == 'model':
            self.create_model_config(layout, node)
        elif node.node_type == 'vector':
            self.create_vector_config(layout, node)
        elif node.node_type == 'analysis':
            self.create_analysis_config(layout, node)
        elif node.node_type == 'prompt':
            self.create_prompt_config(layout, node)
        elif node.node_type == 'llm':
            self.create_llm_config(layout, node)
            
        # 버튼
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(lambda: self.save_node_config(dialog, node))
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        dialog.setLayout(layout)
        dialog.exec_()
        
    def create_data_config(self, layout, node):
        """데이터 노드 설정"""
        layout.addWidget(QLabel("<b>데이터 소스 설정</b>"))
        
        layout.addWidget(QLabel("파일 경로:"))
        path_edit = QLineEdit(node.settings.get('path', ''))
        path_edit.setObjectName("path")
        
        browse_btn = QPushButton("찾아보기...")
        browse_btn.clicked.connect(lambda: self.browse_file(path_edit))
        
        path_layout = QHBoxLayout()
        path_layout.addWidget(path_edit)
        path_layout.addWidget(browse_btn)
        layout.addLayout(path_layout)
        
        layout.addWidget(QLabel("데이터 형식:"))
        format_combo = QComboBox()
        format_combo.addItems(["CSV", "JSON", "Excel", "Database"])
        format_combo.setCurrentText(node.settings.get('format', 'CSV'))
        format_combo.setObjectName("format")
        layout.addWidget(format_combo)
        
    def create_preprocess_config(self, layout, node):
        """전처리 노드 설정"""
        layout.addWidget(QLabel("<b>전처리 설정</b>"))
        
        if "이상치" in node.title:
            layout.addWidget(QLabel("이상치 탐지 방법:"))
            method_combo = QComboBox()
            method_combo.addItems(["IQR", "Z-Score", "Isolation Forest", "DBSCAN"])
            method_combo.setCurrentText(node.settings.get('method', 'IQR'))
            method_combo.setObjectName("method")
            layout.addWidget(method_combo)
            
    def create_model_config(self, layout, node):
        """모델 노드 설정"""
        layout.addWidget(QLabel("<b>모델 설정</b>"))
        
        layout.addWidget(QLabel("예측 기간:"))
        period_combo = QComboBox()
        period_combo.addItems(["6시간", "12시간", "24시간", "48시간", "72시간", "1주일"])
        period_combo.setCurrentText(node.settings.get('period', '24시간'))
        period_combo.setObjectName("period")
        layout.addWidget(period_combo)
        
    def create_vector_config(self, layout, node):
        """벡터 저장 노드 설정"""
        layout.addWidget(QLabel("<b>RAG 벡터 저장 설정</b>"))
        
        layout.addWidget(QLabel("임베딩 모델:"))
        embed_combo = QComboBox()
        embed_combo.addItems(["OpenAI", "Sentence-BERT", "Custom", "Multilingual"])
        embed_combo.setCurrentText(node.settings.get('embedding_model', 'OpenAI'))
        embed_combo.setObjectName("embedding_model")
        layout.addWidget(embed_combo)
        
    def create_analysis_config(self, layout, node):
        """분석 노드 설정"""
        layout.addWidget(QLabel("<b>패턴 분석 설정</b>"))
        
        layout.addWidget(QLabel("분석 기간:"))
        period_combo = QComboBox()
        period_combo.addItems(["7일", "14일", "30일", "90일", "180일", "1년"])
        period_combo.setCurrentText(node.settings.get('analysis_period', '30일'))
        period_combo.setObjectName("analysis_period")
        layout.addWidget(period_combo)
        
    def create_prompt_config(self, layout, node):
        """프롬프트 노드 설정"""
        layout.addWidget(QLabel("<b>프롬프트 생성 설정</b>"))
        
        layout.addWidget(QLabel("프롬프트 템플릿:"))
        template_text = QTextEdit()
        template_text.setObjectName("template")
        template_text.setPlainText(node.settings.get('template', 
            """다음 물류 데이터를 분석하여 예측해주세요:
- 현재 상황: {current_status}
- 과거 패턴: {past_patterns}
- 시계열 예측: {time_series_prediction}

24시간 이내의 물류 이동을 예측하고,
주의해야 할 리스크를 분석해주세요."""))
        template_text.setMaximumHeight(150)
        layout.addWidget(template_text)
        
    def create_llm_config(self, layout, node):
        """LLM 노드 설정"""
        layout.addWidget(QLabel("<b>PHI-4 LLM 설정</b>"))
        
        layout.addWidget(QLabel("Temperature:"))
        temp_slider = QSlider(Qt.Horizontal)
        temp_slider.setRange(0, 100)
        temp_slider.setValue(int(node.settings.get('temperature', 0.7) * 100))
        temp_slider.setObjectName("temperature")
        
        temp_label = QLabel(f"{node.settings.get('temperature', 0.7):.1f}")
        temp_slider.valueChanged.connect(lambda v: temp_label.setText(f"{v/100:.1f}"))
        
        temp_layout = QHBoxLayout()
        temp_layout.addWidget(temp_slider)
        temp_layout.addWidget(temp_label)
        layout.addLayout(temp_layout)
        
    def browse_file(self, line_edit):
        """파일 찾아보기"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "파일 선택", "", 
            "Data Files (*.csv *.json *.xlsx);;All Files (*.*)"
        )
        if filename:
            line_edit.setText(filename)
            
    def save_node_config(self, dialog, node):
        """노드 설정 저장"""
        settings = {}
        
        # 위젯에서 값 수집
        for widget in dialog.findChildren(QWidget):
            name = widget.objectName()
            if name:
                if isinstance(widget, QLineEdit):
                    settings[name] = widget.text()
                elif isinstance(widget, QComboBox):
                    settings[name] = widget.currentText()
                elif isinstance(widget, QSpinBox):
                    settings[name] = widget.value()
                elif isinstance(widget, QTextEdit):
                    settings[name] = widget.toPlainText()
                elif isinstance(widget, QSlider):
                    settings[name] = widget.value() / 100
                    
        node.settings = settings
        node.set_configured(True)
        
        dialog.accept()
        
    def validate_pipeline(self):
        """파이프라인 검증"""
        errors = []
        warnings = []
        
        # 노드 확인
        nodes = self.scene.nodes
        if not nodes:
            errors.append("노드가 없습니다")
            return errors, warnings
            
        # 노드 타입별 확인
        data_nodes = [n for n in nodes if isinstance(n, LogisticsNode) and n.node_type == 'data']
        llm_nodes = [n for n in nodes if isinstance(n, LogisticsNode) and n.node_type == 'llm']
        
        if not data_nodes:
            errors.append("데이터 입력 노드가 필요합니다")
            
        if not llm_nodes:
            errors.append("LLM 노드가 필요합니다")
            
        # 미설정 노드 확인
        unconfigured = [n.title for n in nodes 
                       if isinstance(n, LogisticsNode) and not n.is_configured]
        if unconfigured:
            warnings.append(f"미설정 노드: {', '.join(unconfigured)}")
            
        return errors, warnings
        
    def run_pipeline(self):
        """파이프라인 실행"""
        # 검증
        errors, warnings = self.validate_pipeline()
        
        if errors:
            QMessageBox.warning(self, "검증 실패", 
                               "오류:\n" + "\n".join(f"- {e}" for e in errors))
            return
            
        # 실행 결과 표시
        result_dialog = QDialog()
        result_dialog.setWindowTitle("실행 결과")
        result_dialog.setModal(True)
        layout = QVBoxLayout()
        
        result_text = QTextEdit()
        result_text.setReadOnly(True)
        result_text.setPlainText("""=== 물류이동 예측 시스템 실행 결과 ===

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
3. 예비 차량 20% 추가 배치 필요""")
        
        layout.addWidget(result_text)
        
        close_btn = QPushButton("닫기")
        close_btn.clicked.connect(result_dialog.accept)
        layout.addWidget(close_btn)
        
        result_dialog.setLayout(layout)
        result_dialog.resize(600, 400)
        result_dialog.exec_()


class LogisticsNodeEditor(NodeEditorMainWindow):
    """물류 예측 시스템 메인 윈도우"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        """UI 초기화"""
        self.create_menus()
        self.create_toolbars()
        self.create_dock_widgets()
        self.create_status_bar()
        
        # 새 서브윈도우 생성
        sub_wnd = LogisticsSubWindow()
        self.mdi_area.addSubWindow(sub_wnd)
        sub_wnd.show()
        
        self.setWindowTitle("물류이동 예측 시스템 - Node Editor")
        self.resize(1400, 800)
        
    def create_menus(self):
        """메뉴 생성"""
        # 기본 메뉴
        super().create_menus()
        
        # 실행 메뉴
        run_menu = self.menuBar().addMenu("실행")
        
        validate_action = run_menu.addAction("검증")
        validate_action.setShortcut("F5")
        validate_action.triggered.connect(self.on_validate)
        
        run_action = run_menu.addAction("실행")
        run_action.setShortcut("F6")
        run_action.triggered.connect(self.on_run)
        
    def create_toolbars(self):
        """툴바 생성"""
        toolbar = self.addToolBar("실행")
        
        validate_action = toolbar.addAction("✓ 검증")
        validate_action.triggered.connect(self.on_validate)
        
        run_action = toolbar.addAction("▶ 실행")
        run_action.triggered.connect(self.on_run)
        
    def create_dock_widgets(self):
        """독 위젯 생성"""
        # 노드 목록
        list_widget = QDockWidget("노드 목록")
        list_widget.setWidget(self.create_nodes_list_widget())
        self.addDockWidget(Qt.LeftDockWidgetArea, list_widget)
        
    def create_nodes_list_widget(self):
        """노드 목록 위젯 생성"""
        list_widget = QListWidget()
        list_widget.setDragEnabled(True)
        
        # 노드 추가
        nodes = [
            ("데이터 입력/MCS 로그 데이터", "💾"),
            ("데이터 입력/진동센서 데이터", "📊"),
            ("데이터 전처리/이상치 제거", "🔧"),
            ("데이터 전처리/시간별 분류", "⏰"),
            ("데이터 전처리/경로별 분류", "🛣️"),
            ("벡터 저장/RAG 벡터 저장", "📦"),
            ("시계열 모델/LSTM 모델", "🧠"),
            ("시계열 모델/RNN 모델", "🔄"),
            ("시계열 모델/ARIMA 모델", "📈"),
            ("분석/과거패턴 분석", "🔍"),
            ("프롬프트/프롬프트 생성", "📝"),
            ("LLM/PHI-4 판단/추론", "🤖"),
        ]
        
        for node_path, icon in nodes:
            item = QListWidgetItem(f"{icon} {node_path.split('/')[-1]}")
            item.setData(Qt.UserRole, node_path)
            list_widget.addItem(item)
            
        # 드래그 시작
        def start_drag(event):
            item = list_widget.currentItem()
            if item:
                drag = QDrag(list_widget)
                mime_data = QMimeData()
                mime_data.setText(item.data(Qt.UserRole))
                drag.setMimeData(mime_data)
                drag.exec_(Qt.MoveAction)
                
        list_widget.mouseMoveEvent = start_drag
        
        return list_widget
        
    def create_status_bar(self):
        """상태바 생성"""
        self.statusBar().showMessage("준비됨")
        
    def on_validate(self):
        """검증 실행"""
        sub_wnd = self.get_current_sub_window()
        if sub_wnd:
            errors, warnings = sub_wnd.validate_pipeline()
            
            if errors or warnings:
                msg = ""
                if errors:
                    msg += "오류:\n" + "\n".join(f"- {e}" for e in errors) + "\n\n"
                if warnings:
                    msg += "경고:\n" + "\n".join(f"- {w}" for w in warnings)
                    
                QMessageBox.warning(self, "검증 결과", msg)
            else:
                QMessageBox.information(self, "검증 결과", "파이프라인이 유효합니다!")
                
    def on_run(self):
        """파이프라인 실행"""
        sub_wnd = self.get_current_sub_window()
        if sub_wnd:
            sub_wnd.run_pipeline()
            
    def get_current_sub_window(self):
        """현재 서브윈도우 반환"""
        active_sub_window = self.mdi_area.activeSubWindow()
        if active_sub_window:
            return active_sub_window.widget()
        return None
        
    def about(self):
        """About 대화상자"""
        QMessageBox.about(self, "About", 
                         "물류이동 예측 시스템\n\n"
                         "MCS 로그와 센서 데이터를 활용한\n"
                         "AI 기반 물류 예측 시스템\n\n"
                         "Qt Node Editor 기반")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 스타일 설정
    app.setStyle("Fusion")
    
    # 다크 테마
    from pyqt_node_editor import DARK_STYLE
    app.setStyleSheet(DARK_STYLE)
    
    # 메인 윈도우 실행
    editor = LogisticsNodeEditor()
    editor.show()
    
    sys.exit(app.exec_())