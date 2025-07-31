def mouseDoubleClickEvent(self, event):
        """더블클릭 시 설정 창 열기"""
        if hasattr(self.scene(), 'parent'):
            self.scene().parent().configure_node(self)
        super().mouseDoubleClickEvent(event)
        
    def contextMenuEvent(self, event):
        """마우스 오른쪽 클릭 컨텍스트 메뉴"""
        menu = QMenu()
        
        # 모든 노드에 공통으로 적용되는 메뉴
        configure_action = QAction("⚙️ 노드 설정", None)
        configure_action.triggered.connect(lambda: self.scene().parent().configure_node(self))
        menu.addAction(configure_action)
        
        # 프롬프트 노드 전용 메뉴
        if self.node_type == NodeType.PROMPT:
            if self.is_configured and 'template' in self.settings:
                preview_action = QAction("👁️ 프롬프트 미리보기", None)
                preview_action.triggered.connect(lambda: self.show_prompt_preview())
                menu.addAction(preview_action)
                
                menu.addSeparator()
                
                # 프롬프트 복사
                copy_prompt_action = QAction("📋 프롬프트 복사", None)
                copy_prompt_action.triggered.connect(lambda: self.copy_prompt_to_clipboard())
                menu.addAction(copy_prompt_action)
        
        # 데이터 노드 전용 메뉴
        elif self.node_type == NodeType.DATA:
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
            
    def show_prompt_preview(self):
        """프롬프트 미리보기"""
        if 'template' in self.settings:
            dialog = QDialog()
            dialog.setWindowTitle("프롬프트 템플릿 미리보기")
            dialog.setModal(True)
            layout = QVBoxLayout()
            
            # 프롬프트 내용
            text_edit = QTextEdit()
            text_edit.setPlainText(self.settings['template'])
            text_edit.setReadOnly(True)
            text_edit.setMinimumSize(500, 300)
            layout.addWidget(text_edit)
            
            # 포함된 컨텍스트 표시
            contexts = []
            for i in range(4):  # 최대 4개의 컨텍스트
                key = f'context_{i}'
                if key in self.settings and self.settings[key]:
                    contexts.append(['날씨 정보', '교통 상황', '과거 지연 이력', '특별 이벤트'][i])
            
            if contexts:
                context_label = QLabel(f"포함된 컨텍스트: {', '.join(contexts)}")
                layout.addWidget(context_label)
            
            # 닫기 버튼
            close_btn = QPushButton("닫기")
            close_btn.clicked.connect(dialog.accept)
            layout.addWidget(close_btn)
            
            dialog.setLayout(layout)
            dialog.exec_()
            
    def copy_prompt_to_clipboard(self):
        """프롬프트를 클립보드에 복사"""
        if 'template' in self.settings:
            clipboard = QApplication.clipboard()
            clipboard.setText(self.settings['template'])
            
            # 복사 완료 메시지
            if hasattr(self.scene(), 'parent'):
                self.scene().parent().log("프롬프트가 클립보드에 복사되었습니다")
                
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
        if hasattr(self.scene(), 'parent'):
            # 새 노드 생성
            new_node = Node(self.node_type, self.name, self.x() + 50, self.y() + 50)
            new_node.settings = self.settings.copy()
            new_node.is_configured = self.is_configured
            new_node.update_status()
            
            self.scene().addItem(new_node)
            self.scene().parent().log(f"{self.name} 노드가 복제되었습니다")
            
    def delete_self(self):
        """자신을 삭제"""
        if hasattr(self.scene(), 'parent'):
            reply = QMessageBox.question(None, "확인", 
                                       f"{self.name} 노드를 삭제하시겠습니까?",
                                       QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.scene().parent().view.delete_node(self)