"""
================================================================================
AMHS 예측 모델 패치노트 관리 프로그램 (GUI)
================================================================================
기능:
  - 패치노트 작성/수정/삭제
  - JSON 파일로 저장/로드
  - 탭으로 버전별 분리
  - MD 내보내기
================================================================================
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import json
import csv
import os
from datetime import datetime
from typing import List, Dict, Optional


class PatchNote:
    """패치노트 데이터 클래스"""
    
    def __init__(self, version: str, date: str = None, note_type: str = "Major"):
        self.version = version
        self.date = date or datetime.now().strftime("%Y-%m-%d")
        self.type = note_type
        self.new: List[str] = []
        self.improved: List[str] = []
        self.fixed: List[str] = []
        self.breaking: List[str] = []
        self.metrics: List[str] = []
        self.files: List[str] = []
        self.next: List[str] = []
        self.important: List[str] = []
    
    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "date": self.date,
            "type": self.type,
            "new": self.new,
            "improved": self.improved,
            "fixed": self.fixed,
            "breaking": self.breaking,
            "metrics": self.metrics,
            "files": self.files,
            "next": self.next,
            "important": self.important,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'PatchNote':
        note = cls(data["version"], data["date"], data.get("type", "Major"))
        note.new = data.get("new", [])
        note.fixed = data.get("fixed", [])
        note.breaking = data.get("breaking", [])
        note.files = data.get("files", [])
        note.next = data.get("next", [])
        note.important = data.get("important", [])
        
        # improved 호환성 처리
        improved_data = data.get("improved", [])
        if improved_data and isinstance(improved_data[0], dict):
            note.improved = [f"{item.get('item', '')}: {item.get('before', '')} → {item.get('after', '')}" for item in improved_data]
        else:
            note.improved = improved_data
        
        # metrics 호환성 처리
        metrics_data = data.get("metrics", [])
        if isinstance(metrics_data, dict):
            note.metrics = []
            for name, info in metrics_data.items():
                if isinstance(info, dict):
                    note.metrics.append(f"{name}: {info.get('before', '-')} → {info.get('after', '-')} ({info.get('change', '')})")
                else:
                    note.metrics.append(f"{name}: {info}")
        else:
            note.metrics = metrics_data
        
        return note


class VersionManagement:
    """버전 관리 데이터 클래스 (CSV용)"""
    
    def __init__(self):
        self.date: str = ""
        self.operation: str = ""
        self.development: str = ""
        self.fab: str = ""
        self.model_name: str = ""
        self.version: str = ""
        self.total_predict: str = ""
        self.early_detect: str = ""
        self.model_desc: str = ""
    
    def to_row(self) -> List[str]:
        return [
            self.date, self.operation, self.development, self.fab,
            self.model_name, self.version, self.total_predict,
            self.early_detect, self.model_desc
        ]
    
    @classmethod
    def from_row(cls, row: List[str]) -> 'VersionManagement':
        vm = cls()
        if len(row) >= 9:
            vm.date = row[0]
            vm.operation = row[1]
            vm.development = row[2]
            vm.fab = row[3]
            vm.model_name = row[4]
            vm.version = row[5]
            vm.total_predict = row[6]
            vm.early_detect = row[7]
            vm.model_desc = row[8]
        return vm
    
    @staticmethod
    def csv_header() -> List[str]:
        return ["날짜", "운영", "개발", "FAB", "예측모델_NAME", "버전", "전체예측", "사전감지예측", "모델내용"]


class PatchNoteApp:
    """패치노트 GUI 애플리케이션"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("AMHS 예측 모델 패치노트 관리")
        self.root.geometry("900x700")
        self.root.configure(bg="#2b2b2b")
        
        self.notes: Dict[str, PatchNote] = {}
        self.current_version: Optional[str] = None
        self.save_file = "patch_notes.json"
        
        # 스타일 설정
        self.setup_styles()
        
        # UI 구성
        self.create_widgets()
        
        # 샘플 데이터 로드
        self.load_or_create_sample()
    
    def setup_styles(self):
        """스타일 설정"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # 다크 테마 색상
        self.colors = {
            'bg': '#2b2b2b',
            'fg': '#ffffff',
            'accent': '#4a9eff',
            'accent2': '#ff6b6b',
            'card': '#3c3c3c',
            'input': '#404040',
            'border': '#555555',
        }
        
        style.configure('TFrame', background=self.colors['bg'])
        style.configure('Card.TFrame', background=self.colors['card'])
        style.configure('TLabel', background=self.colors['bg'], foreground=self.colors['fg'], font=('맑은 고딕', 10))
        style.configure('Title.TLabel', font=('맑은 고딕', 14, 'bold'))
        style.configure('Header.TLabel', font=('맑은 고딕', 12, 'bold'))
        style.configure('TButton', font=('맑은 고딕', 10))
        style.configure('Accent.TButton', font=('맑은 고딕', 10, 'bold'))
        style.configure('TNotebook', background=self.colors['bg'])
        style.configure('TNotebook.Tab', font=('맑은 고딕', 10), padding=[10, 5])
    
    def create_widgets(self):
        """위젯 생성"""
        
        # ===== 상단 헤더 =====
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(header_frame, text="📋 AMHS 예측 모델 패치노트", style='Title.TLabel').pack(side=tk.LEFT)
        
        # 버튼들
        btn_frame = ttk.Frame(header_frame)
        btn_frame.pack(side=tk.RIGHT)
        
        ttk.Button(btn_frame, text="💾 저장", command=self.save_notes).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="📂 로드", command=self.load_notes).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="📄 MD 내보내기", command=self.export_md).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="➕ 새 버전", command=self.new_version_dialog).pack(side=tk.LEFT, padx=2)
        
        # ===== 메인 영역 =====
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 좌측: 버전 목록
        left_frame = ttk.Frame(main_frame, width=150)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_frame.pack_propagate(False)
        
        ttk.Label(left_frame, text="📦 버전 목록", style='Header.TLabel').pack(pady=5)
        
        self.version_listbox = tk.Listbox(
            left_frame, 
            font=('맑은 고딕', 11),
            bg=self.colors['card'],
            fg=self.colors['fg'],
            selectbackground=self.colors['accent'],
            selectforeground='white',
            borderwidth=0,
            highlightthickness=1,
            highlightcolor=self.colors['border']
        )
        self.version_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        self.version_listbox.bind('<<ListboxSelect>>', self.on_version_select)
        
        ttk.Button(left_frame, text="🗑️ 삭제", command=self.delete_version).pack(fill=tk.X, pady=5)
        
        # 우측: 패치노트 상세
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 탭 노트북
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # 탭 1: 기본 정보
        self.tab_info = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_info, text="📝 기본 정보")
        self.create_info_tab()
        
        # 탭 2: 신규/개선
        self.tab_changes = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_changes, text="✨ 변경사항")
        self.create_changes_tab()
        
        # 탭 3: 성능/파일
        self.tab_metrics = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_metrics, text="📊 성능/파일")
        self.create_metrics_tab()
        
        # 탭 4: 버전 관리
        self.tab_version = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_version, text="🔖 버전 관리")
        self.create_version_tab()
        
        # 탭 5: 미리보기
        self.tab_preview = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_preview, text="👁️ 미리보기")
        self.create_preview_tab()
    
    def create_info_tab(self):
        """기본 정보 탭"""
        frame = ttk.Frame(self.tab_info, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # 버전
        ttk.Label(frame, text="버전:").grid(row=0, column=0, sticky='w', pady=5)
        self.entry_version = ttk.Entry(frame, font=('맑은 고딕', 11), width=20)
        self.entry_version.grid(row=0, column=1, sticky='w', pady=5, padx=5)
        
        # 날짜
        ttk.Label(frame, text="날짜:").grid(row=1, column=0, sticky='w', pady=5)
        self.entry_date = ttk.Entry(frame, font=('맑은 고딕', 11), width=20)
        self.entry_date.grid(row=1, column=1, sticky='w', pady=5, padx=5)
        
        # 유형
        ttk.Label(frame, text="유형:").grid(row=2, column=0, sticky='w', pady=5)
        self.combo_type = ttk.Combobox(frame, values=["Major", "Minor", "Patch"], font=('맑은 고딕', 11), width=17)
        self.combo_type.grid(row=2, column=1, sticky='w', pady=5, padx=5)
        self.combo_type.set("Major")
        
        # 중요 정보
        ttk.Label(frame, text="⚠️ 중요 정보:").grid(row=3, column=0, sticky='nw', pady=5)
        self.text_important = scrolledtext.ScrolledText(frame, height=4, width=50, font=('맑은 고딕', 10),
                                                        bg=self.colors['input'], fg=self.colors['fg'])
        self.text_important.grid(row=3, column=1, sticky='w', pady=5, padx=5)
        
        # Breaking Changes
        ttk.Label(frame, text="Breaking Changes:").grid(row=4, column=0, sticky='nw', pady=5)
        self.text_breaking = scrolledtext.ScrolledText(frame, height=4, width=50, font=('맑은 고딕', 10),
                                                        bg=self.colors['input'], fg=self.colors['fg'])
        self.text_breaking.grid(row=4, column=1, sticky='w', pady=5, padx=5)
        
        # 다음 버전 예고
        ttk.Label(frame, text="다음 버전 예고:").grid(row=5, column=0, sticky='nw', pady=5)
        self.text_next = scrolledtext.ScrolledText(frame, height=4, width=50, font=('맑은 고딕', 10),
                                                    bg=self.colors['input'], fg=self.colors['fg'])
        self.text_next.grid(row=5, column=1, sticky='w', pady=5, padx=5)
    
    def create_changes_tab(self):
        """변경사항 탭"""
        frame = ttk.Frame(self.tab_changes, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # 신규 기능
        ttk.Label(frame, text="✨ 신규 기능 (줄바꿈으로 구분):", style='Header.TLabel').pack(anchor='w')
        self.text_new = scrolledtext.ScrolledText(frame, height=6, font=('맑은 고딕', 10),
                                                   bg=self.colors['input'], fg=self.colors['fg'])
        self.text_new.pack(fill=tk.X, pady=(5, 15))
        
        # 개선사항
        ttk.Label(frame, text="⬆️ 개선사항 (줄바꿈으로 구분):", style='Header.TLabel').pack(anchor='w')
        self.text_improved = scrolledtext.ScrolledText(frame, height=6, font=('맑은 고딕', 10),
                                                        bg=self.colors['input'], fg=self.colors['fg'])
        self.text_improved.pack(fill=tk.X, pady=(5, 15))
        
        # 버그 수정
        ttk.Label(frame, text="🔧 버그 수정 (줄바꿈으로 구분):", style='Header.TLabel').pack(anchor='w')
        self.text_fixed = scrolledtext.ScrolledText(frame, height=6, font=('맑은 고딕', 10),
                                                     bg=self.colors['input'], fg=self.colors['fg'])
        self.text_fixed.pack(fill=tk.X, pady=(5, 15))
    
    def create_metrics_tab(self):
        """성능/파일 탭"""
        frame = ttk.Frame(self.tab_metrics, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # 성능 지표
        ttk.Label(frame, text="📊 성능 변화 (줄바꿈으로 구분):", style='Header.TLabel').pack(anchor='w')
        self.text_metrics = scrolledtext.ScrolledText(frame, height=6, font=('맑은 고딕', 10),
                                                       bg=self.colors['input'], fg=self.colors['fg'])
        self.text_metrics.pack(fill=tk.X, pady=(5, 15))
        
        # 변경 파일
        ttk.Label(frame, text="📁 변경 파일 (줄바꿈으로 구분):", style='Header.TLabel').pack(anchor='w')
        self.text_files = scrolledtext.ScrolledText(frame, height=8, font=('맑은 고딕', 10),
                                                     bg=self.colors['input'], fg=self.colors['fg'])
        self.text_files.pack(fill=tk.X, pady=(5, 15))
    
    def create_version_tab(self):
        """버전 관리 탭 (CSV 저장용, 별도 관리)"""
        frame = ttk.Frame(self.tab_version, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # 안내 문구
        ttk.Label(frame, text="📋 버전 관리 (CSV로 별도 저장)", 
                  font=('맑은 고딕', 12, 'bold')).pack(anchor='w', pady=(0, 10))
        
        # 입력 영역
        input_frame = ttk.LabelFrame(frame, text="새 항목 입력", padding=10)
        input_frame.pack(fill=tk.X, pady=(0, 15))
        
        # 상단 그리드 (날짜, 운영, 개발, FAB)
        row1 = ttk.Frame(input_frame)
        row1.pack(fill=tk.X, pady=5)
        
        ttk.Label(row1, text="📅 날짜:").pack(side=tk.LEFT)
        self.entry_vm_date = ttk.Entry(row1, font=('맑은 고딕', 10), width=12)
        self.entry_vm_date.pack(side=tk.LEFT, padx=(5, 15))
        self.entry_vm_date.insert(0, datetime.now().strftime("%Y-%m-%d"))
        
        ttk.Label(row1, text="🏭 운영:").pack(side=tk.LEFT)
        self.entry_vm_operation = ttk.Entry(row1, font=('맑은 고딕', 10), width=10)
        self.entry_vm_operation.pack(side=tk.LEFT, padx=(5, 15))
        
        ttk.Label(row1, text="💻 개발:").pack(side=tk.LEFT)
        self.entry_vm_development = ttk.Entry(row1, font=('맑은 고딕', 10), width=10)
        self.entry_vm_development.pack(side=tk.LEFT, padx=(5, 15))
        
        ttk.Label(row1, text="🏢 FAB:").pack(side=tk.LEFT)
        self.entry_vm_fab = ttk.Entry(row1, font=('맑은 고딕', 10), width=10)
        self.entry_vm_fab.pack(side=tk.LEFT, padx=(5, 0))
        
        # 중간 그리드 (예측모델_NAME, 버전)
        row2 = ttk.Frame(input_frame)
        row2.pack(fill=tk.X, pady=5)
        
        ttk.Label(row2, text="🤖 예측모델_NAME:").pack(side=tk.LEFT)
        self.entry_vm_model_name = ttk.Entry(row2, font=('맑은 고딕', 10), width=25)
        self.entry_vm_model_name.pack(side=tk.LEFT, padx=(5, 15))
        
        ttk.Label(row2, text="🔖 버전:").pack(side=tk.LEFT)
        self.entry_vm_version = ttk.Entry(row2, font=('맑은 고딕', 10), width=12)
        self.entry_vm_version.pack(side=tk.LEFT, padx=(5, 0))
        
        # 하단 그리드 (전체예측, 사전감지예측)
        row3 = ttk.Frame(input_frame)
        row3.pack(fill=tk.X, pady=5)
        
        ttk.Label(row3, text="📊 전체예측:").pack(side=tk.LEFT)
        self.entry_vm_total_predict = ttk.Entry(row3, font=('맑은 고딕', 10), width=20)
        self.entry_vm_total_predict.pack(side=tk.LEFT, padx=(5, 15))
        
        ttk.Label(row3, text="🔍 사전감지예측:").pack(side=tk.LEFT)
        self.entry_vm_early_detect = ttk.Entry(row3, font=('맑은 고딕', 10), width=20)
        self.entry_vm_early_detect.pack(side=tk.LEFT, padx=(5, 0))
        
        # 모델 내용
        row4 = ttk.Frame(input_frame)
        row4.pack(fill=tk.X, pady=5)
        
        ttk.Label(row4, text="📝 모델내용:").pack(side=tk.LEFT)
        self.entry_vm_model_desc = ttk.Entry(row4, font=('맑은 고딕', 10), width=60)
        self.entry_vm_model_desc.pack(side=tk.LEFT, padx=(5, 0), fill=tk.X, expand=True)
        
        # 버튼 영역 (항목 추가/수정/삭제)
        btn_frame = ttk.Frame(input_frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(btn_frame, text="➕ 항목 추가", command=self.add_vm_row).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="✏️ 선택 수정", command=self.update_vm_row).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="🗑️ 선택 삭제", command=self.delete_vm_row).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="🧹 입력 초기화", command=self.clear_vm_inputs).pack(side=tk.LEFT, padx=5)
        
        # 테이블 (Treeview)
        table_frame = ttk.Frame(frame)
        table_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        columns = ("날짜", "운영", "개발", "FAB", "예측모델_NAME", "버전", "전체예측", "사전감지예측", "모델내용")
        self.vm_tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=8)
        
        # 컬럼 설정
        col_widths = [90, 60, 60, 50, 150, 80, 120, 120, 200]
        for col, width in zip(columns, col_widths):
            self.vm_tree.heading(col, text=col)
            self.vm_tree.column(col, width=width, minwidth=50)
        
        # 스크롤바
        scrollbar_y = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.vm_tree.yview)
        scrollbar_x = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self.vm_tree.xview)
        self.vm_tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        
        self.vm_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 테이블 선택 이벤트
        self.vm_tree.bind('<<TreeviewSelect>>', self.on_vm_select)
        
        # CSV 저장/불러오기 버튼
        csv_frame = ttk.Frame(frame)
        csv_frame.pack(fill=tk.X)
        
        ttk.Button(csv_frame, text="💾 CSV 저장", command=self.save_vm_csv).pack(side=tk.LEFT, padx=5)
        ttk.Button(csv_frame, text="📂 CSV 불러오기", command=self.load_vm_csv).pack(side=tk.LEFT, padx=5)
        
        # 버전 관리 데이터 리스트
        self.vm_data: List[VersionManagement] = []
        self.vm_save_file = "version_management.csv"
        
        # 기존 데이터 로드 또는 샘플 추가
        self._load_vm_data()
    
    def _load_vm_data(self):
        """버전 관리 데이터 로드"""
        if os.path.exists(self.vm_save_file):
            try:
                with open(self.vm_save_file, 'r', encoding='utf-8-sig') as f:
                    reader = csv.reader(f)
                    next(reader)  # 헤더 스킵
                    for row in reader:
                        vm = VersionManagement.from_row(row)
                        self.vm_data.append(vm)
                        self.vm_tree.insert('', tk.END, values=vm.to_row())
            except Exception as e:
                print(f"버전 관리 로드 실패: {e}")
                self._add_sample_vm_data()
        else:
            # 파일 없으면 샘플 데이터
            self._add_sample_vm_data()
            self._auto_save_vm()
    
    def _auto_save_vm(self):
        """버전 관리 자동 저장"""
        try:
            with open(self.vm_save_file, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(VersionManagement.csv_header())
                for vm in self.vm_data:
                    writer.writerow(vm.to_row())
        except Exception as e:
            print(f"버전 관리 저장 실패: {e}")
    
    def _add_sample_vm_data(self):
        """샘플 버전 관리 데이터"""
        sample = VersionManagement()
        sample.date = "2025-12-01"
        sample.operation = "M16A"
        sample.development = "AI팀"
        sample.fab = "M16"
        sample.model_name = "AMHS_JOB2_예측모델"
        sample.version = "V8.0.0"
        sample.total_predict = "MAE 9.8, R² 0.68"
        sample.early_detect = "범주형 정확도 72%"
        sample.model_desc = "수치형+범주형 앙상블, 32컬럼→222Feature"
        
        self.vm_data.append(sample)
        self.vm_tree.insert('', tk.END, values=sample.to_row())
    
    def add_vm_row(self):
        """버전 관리 항목 추가"""
        vm = VersionManagement()
        vm.date = self.entry_vm_date.get().strip()
        vm.operation = self.entry_vm_operation.get().strip()
        vm.development = self.entry_vm_development.get().strip()
        vm.fab = self.entry_vm_fab.get().strip()
        vm.model_name = self.entry_vm_model_name.get().strip()
        vm.version = self.entry_vm_version.get().strip()
        vm.total_predict = self.entry_vm_total_predict.get().strip()
        vm.early_detect = self.entry_vm_early_detect.get().strip()
        vm.model_desc = self.entry_vm_model_desc.get().strip()
        
        if not vm.version:
            messagebox.showwarning("경고", "버전은 필수 입력입니다.")
            return
        
        self.vm_data.append(vm)
        self.vm_tree.insert('', tk.END, values=vm.to_row())
        self.clear_vm_inputs()
        self._auto_save_vm()  # 자동 저장
    
    def on_vm_select(self, event):
        """테이블 선택 시 입력 필드에 데이터 로드"""
        selected = self.vm_tree.selection()
        if not selected:
            return
        
        item = selected[0]
        values = self.vm_tree.item(item, 'values')
        
        if len(values) >= 9:
            self.entry_vm_date.delete(0, tk.END)
            self.entry_vm_date.insert(0, values[0])
            
            self.entry_vm_operation.delete(0, tk.END)
            self.entry_vm_operation.insert(0, values[1])
            
            self.entry_vm_development.delete(0, tk.END)
            self.entry_vm_development.insert(0, values[2])
            
            self.entry_vm_fab.delete(0, tk.END)
            self.entry_vm_fab.insert(0, values[3])
            
            self.entry_vm_model_name.delete(0, tk.END)
            self.entry_vm_model_name.insert(0, values[4])
            
            self.entry_vm_version.delete(0, tk.END)
            self.entry_vm_version.insert(0, values[5])
            
            self.entry_vm_total_predict.delete(0, tk.END)
            self.entry_vm_total_predict.insert(0, values[6])
            
            self.entry_vm_early_detect.delete(0, tk.END)
            self.entry_vm_early_detect.insert(0, values[7])
            
            self.entry_vm_model_desc.delete(0, tk.END)
            self.entry_vm_model_desc.insert(0, values[8])
    
    def update_vm_row(self):
        """선택된 항목 수정"""
        selected = self.vm_tree.selection()
        if not selected:
            messagebox.showwarning("경고", "수정할 항목을 선택하세요.")
            return
        
        item = selected[0]
        idx = self.vm_tree.index(item)
        
        if idx >= len(self.vm_data):
            return
        
        # 입력 필드에서 데이터 가져오기
        vm = self.vm_data[idx]
        vm.date = self.entry_vm_date.get().strip()
        vm.operation = self.entry_vm_operation.get().strip()
        vm.development = self.entry_vm_development.get().strip()
        vm.fab = self.entry_vm_fab.get().strip()
        vm.model_name = self.entry_vm_model_name.get().strip()
        vm.version = self.entry_vm_version.get().strip()
        vm.total_predict = self.entry_vm_total_predict.get().strip()
        vm.early_detect = self.entry_vm_early_detect.get().strip()
        vm.model_desc = self.entry_vm_model_desc.get().strip()
        
        if not vm.version:
            messagebox.showwarning("경고", "버전은 필수 입력입니다.")
            return
        
        # 테이블 업데이트
        self.vm_tree.item(item, values=vm.to_row())
        self._auto_save_vm()  # 자동 저장
        messagebox.showinfo("수정 완료", "항목이 수정되었습니다.")
    
    def delete_vm_row(self):
        """선택된 항목 삭제"""
        selected = self.vm_tree.selection()
        if not selected:
            messagebox.showwarning("경고", "삭제할 항목을 선택하세요.")
            return
        
        for item in selected:
            idx = self.vm_tree.index(item)
            self.vm_tree.delete(item)
            if idx < len(self.vm_data):
                del self.vm_data[idx]
        
        self._auto_save_vm()  # 자동 저장
    
    def clear_vm_inputs(self):
        """입력 필드 초기화"""
        self.entry_vm_date.delete(0, tk.END)
        self.entry_vm_date.insert(0, datetime.now().strftime("%Y-%m-%d"))
        self.entry_vm_operation.delete(0, tk.END)
        self.entry_vm_development.delete(0, tk.END)
        self.entry_vm_fab.delete(0, tk.END)
        self.entry_vm_model_name.delete(0, tk.END)
        self.entry_vm_version.delete(0, tk.END)
        self.entry_vm_total_predict.delete(0, tk.END)
        self.entry_vm_early_detect.delete(0, tk.END)
        self.entry_vm_model_desc.delete(0, tk.END)
    
    def save_vm_csv(self):
        """버전 관리 CSV 저장"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="version_management.csv"
        )
        if not filename:
            return
        
        with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(VersionManagement.csv_header())
            for vm in self.vm_data:
                writer.writerow(vm.to_row())
        
        messagebox.showinfo("저장 완료", f"CSV 저장: {filename}")
    
    def load_vm_csv(self):
        """버전 관리 CSV 불러오기"""
        filename = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not filename:
            return
        
        try:
            with open(filename, 'r', encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                header = next(reader)  # 헤더 스킵
                
                # 기존 데이터 클리어
                self.vm_data.clear()
                for item in self.vm_tree.get_children():
                    self.vm_tree.delete(item)
                
                # 새 데이터 로드
                for row in reader:
                    vm = VersionManagement.from_row(row)
                    self.vm_data.append(vm)
                    self.vm_tree.insert('', tk.END, values=vm.to_row())
            
            self._auto_save_vm()  # 자동 저장
            messagebox.showinfo("로드 완료", f"CSV 로드: {filename}")
        except Exception as e:
            messagebox.showerror("오류", f"CSV 로드 실패: {e}")
    
    def create_preview_tab(self):
        """미리보기 탭"""
        frame = ttk.Frame(self.tab_preview, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Button(frame, text="🔄 새로고침", command=self.refresh_preview).pack(anchor='w', pady=5)
        
        self.text_preview = scrolledtext.ScrolledText(
            frame, 
            font=('Consolas', 10),
            bg=self.colors['card'], 
            fg=self.colors['fg'],
            wrap=tk.NONE
        )
        self.text_preview.pack(fill=tk.BOTH, expand=True, pady=5)
    
    # ===== 기능 메서드 =====
    
    def load_or_create_sample(self):
        """파일 로드 또는 샘플 생성"""
        if os.path.exists(self.save_file):
            self._load_from_file(self.save_file)
        else:
            self._create_sample_data()
        self.refresh_version_list()
    
    def _create_sample_data(self):
        """샘플 데이터 생성"""
        # V8
        v8 = PatchNote("V8.0.0", "2025-12-01", "Major")
        v8.important = [
            "V7 모델과 호환되지 않음 - 반드시 V8 코드 사용",
            "기존 patch_notes.json 삭제 후 재실행 필요",
        ]
        v8.new = [
            "범주형 3-Class 분류 모델 추가 (하락/소폭/급증)",
            "M16HUB QUE 컬럼 5개 추가",
            "M16A QUE 컬럼 7개 추가",
            "CURRENT_M16A_3F_JOB 대체 타겟 컬럼 추가",
        ]
        v8.improved = [
            "입력 컬럼: 19개 → 32개",
            "Feature 수: 180개 → 222개",
            "수치형 모델: 50+ 변화 샘플 3배 가중치 적용",
            "범주형 모델: Class2(급증) 5배 가중치 적용",
        ]
        v8.metrics = [
            "MAE: 12.5 → 9.8 (↓21%)",
            "R²: 0.62 → 0.68 (↑9.7%)",
            "범주형 정확도: 72% (신규)",
        ]
        v8.breaking = ["V7 모델 파일(.pkl)과 호환되지 않음", "V8 전용 평가 코드 사용 필요"]
        v8.files = [
            "V8_수치형_학습코드.py", "V8_수치형_평가코드.py",
            "V8_범주형_학습코드.py", "V8_범주형_평가코드.py",
            "xgboost_수치형_V8.pkl", "xgboost_범주형_V8.pkl"
        ]
        v8.next = ["MLflow Tracking 연동", "하이퍼파라미터 자동 탐색 (Optuna)"]
        self.notes["V8.0.0"] = v8
        
        # V7
        v7 = PatchNote("V7.0.0", "2025-11-15", "Major")
        v7.new = ["M16HUB.QUE.ALL.CURRENTQCNT 컬럼 추가", "QUE 기반 병목 감지 Feature 추가"]
        v7.improved = ["입력 컬럼: 17개 → 19개", "병목 감지 로직 강화"]
        v7.metrics = ["MAE: 14.2 → 12.5 (↓12%)"]
        v7.fixed = ["HUBROOMTOTAL 결측치 처리 오류 수정"]
        v7.breaking = ["V6 모델 파일과 호환되지 않음"]
        v7.next = ["범주형 분류 모델 추가"]
        self.notes["V7.0.0"] = v7
        
        # V6
        v6 = PatchNote("V6.0.0", "2025-10-20", "Major")
        v6.new = ["멀티아웃풋 구조 도입 (MIN/MAX/AVG)", "10분 구간 통계 예측"]
        v6.improved = ["Feature 수: 120개 → 165개", "시퀀스 길이: 20분 → 30분"]
        v6.metrics = ["MAE: 18.5 → 14.2 (↓23%)"]
        v6.fixed = ["큰 변화 예측 누락 문제 해결"]
        v6.breaking = ["단일 출력 → 멀티 출력 구조 변경"]
        self.notes["V6.0.0"] = v6
    
    def refresh_version_list(self):
        """버전 목록 갱신"""
        self.version_listbox.delete(0, tk.END)
        for version in sorted(self.notes.keys(), reverse=True):
            self.version_listbox.insert(tk.END, version)
        
        if self.notes:
            self.version_listbox.selection_set(0)
            self.on_version_select(None)
    
    def on_version_select(self, event):
        """버전 선택 시"""
        selection = self.version_listbox.curselection()
        if not selection:
            return
        
        version = self.version_listbox.get(selection[0])
        self.current_version = version
        self.load_note_to_ui(self.notes[version])
    
    def load_note_to_ui(self, note: PatchNote):
        """패치노트를 UI에 로드"""
        # 기본 정보
        self.entry_version.delete(0, tk.END)
        self.entry_version.insert(0, note.version)
        
        self.entry_date.delete(0, tk.END)
        self.entry_date.insert(0, note.date)
        
        self.combo_type.set(note.type)
        
        self.text_important.delete('1.0', tk.END)
        self.text_important.insert('1.0', '\n'.join(note.important))
        
        self.text_breaking.delete('1.0', tk.END)
        self.text_breaking.insert('1.0', '\n'.join(note.breaking))
        
        self.text_next.delete('1.0', tk.END)
        self.text_next.insert('1.0', '\n'.join(note.next))
        
        # 변경사항
        self.text_new.delete('1.0', tk.END)
        self.text_new.insert('1.0', '\n'.join(note.new))
        
        self.text_improved.delete('1.0', tk.END)
        self.text_improved.insert('1.0', '\n'.join(note.improved))
        
        self.text_fixed.delete('1.0', tk.END)
        self.text_fixed.insert('1.0', '\n'.join(note.fixed))
        
        # 성능/파일
        self.text_metrics.delete('1.0', tk.END)
        self.text_metrics.insert('1.0', '\n'.join(note.metrics))
        
        self.text_files.delete('1.0', tk.END)
        self.text_files.insert('1.0', '\n'.join(note.files))
        
        # 미리보기
        self.refresh_preview()
    
    def save_ui_to_note(self) -> Optional[PatchNote]:
        """UI에서 패치노트로 저장"""
        version = self.entry_version.get().strip()
        if not version:
            messagebox.showerror("오류", "버전을 입력하세요")
            return None
        
        note = PatchNote(
            version,
            self.entry_date.get().strip() or datetime.now().strftime("%Y-%m-%d"),
            self.combo_type.get()
        )
        
        note.new = [x.strip() for x in self.text_new.get('1.0', tk.END).strip().split('\n') if x.strip()]
        note.improved = [x.strip() for x in self.text_improved.get('1.0', tk.END).strip().split('\n') if x.strip()]
        note.fixed = [x.strip() for x in self.text_fixed.get('1.0', tk.END).strip().split('\n') if x.strip()]
        note.breaking = [x.strip() for x in self.text_breaking.get('1.0', tk.END).strip().split('\n') if x.strip()]
        note.next = [x.strip() for x in self.text_next.get('1.0', tk.END).strip().split('\n') if x.strip()]
        note.files = [x.strip() for x in self.text_files.get('1.0', tk.END).strip().split('\n') if x.strip()]
        note.metrics = [x.strip() for x in self.text_metrics.get('1.0', tk.END).strip().split('\n') if x.strip()]
        note.important = [x.strip() for x in self.text_important.get('1.0', tk.END).strip().split('\n') if x.strip()]
        
        return note
    
    def refresh_preview(self):
        """미리보기 갱신"""
        note = self.save_ui_to_note()
        if not note:
            return
        
        preview = self._generate_preview(note)
        self.text_preview.delete('1.0', tk.END)
        self.text_preview.insert('1.0', preview)
    
    def _generate_preview(self, note: PatchNote) -> str:
        """미리보기 텍스트 생성"""
        lines = []
        w = 58
        
        lines.append("┌" + "─" * w + "┐")
        lines.append(f"│  AMHS 예측 모델 {note.version}".ljust(w) + " │")
        lines.append(f"│  {note.date} | {note.type} Update".ljust(w) + " │")
        lines.append("├" + "─" * w + "┤")
        
        if note.important:
            lines.append(f"│  [⚠️ IMPORTANT] 중요 정보".ljust(w) + " │")
            for item in note.important:
                line = f"│    • {item}"
                lines.append(line[:w].ljust(w) + " │")
        
        if note.new:
            lines.append(f"│  [NEW] 신규 기능".ljust(w) + " │")
            for item in note.new:
                line = f"│    • {item}"
                lines.append(line[:w].ljust(w) + " │")
        
        if note.improved:
            lines.append(f"│  [IMPROVED] 개선사항".ljust(w) + " │")
            for item in note.improved:
                line = f"│    • {item}"
                lines.append(line[:w].ljust(w) + " │")
        
        if note.metrics:
            lines.append(f"│  [METRICS] 성능 변화".ljust(w) + " │")
            for item in note.metrics:
                line = f"│    • {item}"
                lines.append(line[:w].ljust(w) + " │")
        
        lines.append(f"│  [FIXED] 버그 수정".ljust(w) + " │")
        if note.fixed:
            for item in note.fixed:
                line = f"│    • {item}"
                lines.append(line[:w].ljust(w) + " │")
        else:
            lines.append(f"│    • 없음".ljust(w) + " │")
        
        if note.breaking:
            lines.append(f"│  [BREAKING] 호환성".ljust(w) + " │")
            for item in note.breaking:
                line = f"│    • {item}"
                lines.append(line[:w].ljust(w) + " │")
        
        if note.files:
            lines.append(f"│  [FILES] 변경 파일".ljust(w) + " │")
            for item in note.files:
                line = f"│    • {item}"
                lines.append(line[:w].ljust(w) + " │")
        
        if note.next:
            lines.append(f"│  [NEXT] 다음 버전 예고".ljust(w) + " │")
            for item in note.next:
                line = f"│    • {item}"
                lines.append(line[:w].ljust(w) + " │")
        
        lines.append("└" + "─" * w + "┘")
        
        return '\n'.join(lines)
    
    def save_notes(self):
        """저장"""
        # 현재 편집 중인 노트 저장
        if self.current_version:
            note = self.save_ui_to_note()
            if note:
                # 버전이 변경된 경우
                if note.version != self.current_version:
                    del self.notes[self.current_version]
                self.notes[note.version] = note
                self.current_version = note.version
        
        # 파일로 저장
        data = {
            "project": "AMHS 예측 모델",
            "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "notes": {v: n.to_dict() for v, n in self.notes.items()}
        }
        
        with open(self.save_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        self.refresh_version_list()
        messagebox.showinfo("저장", f"저장 완료: {self.save_file}")
    
    def load_notes(self):
        """로드"""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            self._load_from_file(filename)
            self.refresh_version_list()
            messagebox.showinfo("로드", f"로드 완료: {filename}")
    
    def _load_from_file(self, filename):
        """파일에서 로드"""
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.notes = {}
        for version, note_data in data.get("notes", {}).items():
            self.notes[version] = PatchNote.from_dict(note_data)
    
    def export_md(self):
        """마크다운 내보내기"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".md",
            filetypes=[("Markdown files", "*.md"), ("All files", "*.*")],
            initialfile="PATCH_NOTES.md"
        )
        if not filename:
            return
        
        lines = []
        lines.append("# AMHS 예측 모델 패치노트\n")
        lines.append(f"*최종 업데이트: {datetime.now().strftime('%Y-%m-%d')}*\n")
        
        for version in sorted(self.notes.keys(), reverse=True):
            note = self.notes[version]
            lines.append(f"\n---\n")
            lines.append(f"## {note.version} ({note.date})\n")
            lines.append(f"**{note.type} Update**\n")
            
            if note.important:
                lines.append("\n### ⚠️ 중요 정보\n")
                for item in note.important:
                    lines.append(f"- {item}\n")
            
            if note.new:
                lines.append("\n### ✨ 신규 기능\n")
                for item in note.new:
                    lines.append(f"- {item}\n")
            
            if note.improved:
                lines.append("\n### ⬆️ 개선사항\n")
                for item in note.improved:
                    lines.append(f"- {item}\n")
            
            if note.metrics:
                lines.append("\n### 📊 성능 변화\n")
                for item in note.metrics:
                    lines.append(f"- {item}\n")
            
            if note.fixed:
                lines.append("\n### 🔧 버그 수정\n")
                for item in note.fixed:
                    lines.append(f"- {item}\n")
            
            if note.breaking:
                lines.append("\n### ⚠️ Breaking Changes\n")
                for item in note.breaking:
                    lines.append(f"- {item}\n")
            
            if note.files:
                lines.append("\n### 📁 변경 파일\n")
                for item in note.files:
                    lines.append(f"- `{item}`\n")
            
            if note.next:
                lines.append("\n### 🔮 다음 버전 예고\n")
                for item in note.next:
                    lines.append(f"- {item}\n")
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        messagebox.showinfo("내보내기", f"MD 저장 완료: {filename}")
    
    def new_version_dialog(self):
        """새 버전 다이얼로그"""
        dialog = tk.Toplevel(self.root)
        dialog.title("새 패치노트")
        dialog.geometry("300x150")
        dialog.configure(bg=self.colors['bg'])
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="새 버전:").pack(pady=10)
        entry = ttk.Entry(dialog, font=('맑은 고딕', 12), width=20)
        entry.pack(pady=5)
        entry.insert(0, "V9.0.0")
        entry.focus()
        
        def create():
            version = entry.get().strip()
            if version:
                if version in self.notes:
                    messagebox.showerror("오류", "이미 존재하는 버전입니다")
                    return
                self.notes[version] = PatchNote(version)
                self.refresh_version_list()
                # 새 버전 선택
                idx = list(sorted(self.notes.keys(), reverse=True)).index(version)
                self.version_listbox.selection_clear(0, tk.END)
                self.version_listbox.selection_set(idx)
                self.on_version_select(None)
                dialog.destroy()
        
        ttk.Button(dialog, text="생성", command=create).pack(pady=15)
    
    def delete_version(self):
        """버전 삭제"""
        if not self.current_version:
            return
        
        if messagebox.askyesno("삭제 확인", f"{self.current_version}을(를) 삭제하시겠습니까?"):
            del self.notes[self.current_version]
            self.current_version = None
            self.refresh_version_list()


def main():
    root = tk.Tk()
    app = PatchNoteApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()