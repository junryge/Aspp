1. NodeView 클래스 전체 (수정됨)
pythonclass NodeView(QGraphicsView):
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
2. create_menu_bar 메서드에서 수정된 부분
python        fit_action = QAction("전체 보기", self)
        fit_action.setShortcut("F")  # Space에서 F로 변경
        fit_action.triggered.connect(self.fit_view)
        view_menu.addAction(fit_action)
3. create_toolbar 메서드에서 수정된 부분
python        zoom_reset_action = QAction(QIcon(), "100%", self)
        zoom_reset_action.triggered.connect(self.reset_zoom)
        toolbar.addAction(zoom_reset_action)
        
        toolbar.addSeparator()
        
        # 도움말 - 새로 추가
        help_action = QAction(QIcon(), "도움말", self)
        help_action.triggered.connect(self.show_help)
        toolbar.addAction(help_action)
4. show_help 메서드 추가 (update_properties 메서드 앞에)
python    def show_help(self):
        """도움말 표시"""
        help_text = """노드 에디터 조작법
        
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
- 연결선 클릭 후 Delete로 삭제"""
        
        QMessageBox.information(self, "도움말", help_text)
        
    def update_properties(self):
        # ... 기존 코드 ...
5. show_execution_results 메서드에서 수정된 부분
python        # 닫기 버튼
        close_btn = QPushButton("닫기")
        close_btn.clicked.connect(result_dialog.accept)
        layout.addWidget(close_btn)
        
        result_dialog.setLayout(layout)  # if self.isSelected(): 제거
        result_dialog.resize(800, 600)
        result_dialog.exec_()
        
        self.log("파이프라인 실행 완료")
이렇게 총 5개 부분이 변경되었습니다. 주요 변경사항은:

NodeView 클래스에 팬(화면 이동) 기능 추가
전체 보기 단축키를 Space에서 F로 변경
툴바에 도움말 버튼 추가
show_help 메서드 추가
show_execution_results의 문법 오류 수정